from .base_defense import BaseDefense
from core.data_formats import TestCase
from core.unified_registry import UNIFIED_REGISTRY
from .utils import generate_output
from config.config_loader import get_model_config
from PIL import Image

from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
import threading


class LlamaGuard3Defense(BaseDefense):
    """Llama-Guard-3 defense method - GPU inference thread-safe version"""

    # Class variables: singleton instance and locks
    _instance = None
    _instance_lock = threading.Lock()
    _model_loaded = False

    # Model loading lock (protects model loading process)
    _model_init_lock = threading.Lock()

    # Inference lock (protects model inference process)
    _inference_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern: ensure only one instance"""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config):
        """Initialize model (execute only once)"""
        # Double-check locking to ensure model is loaded only once
        if not self._model_loaded:
            with self._model_init_lock:
                if not self._model_loaded:
                    # Call parent class initialization
                    super().__init__(config)

                    # Log initialization start
                    self.logger.info(
                        "Starting Llama-Guard-3 model initialization (GPU version)"
                    )

                    # Get model path
                    llama_guard_path = self.config["Llama-Guard-3"]
                    self.logger.info(f"Model path: {llama_guard_path}")

                    # Check GPU availability
                    if not torch.cuda.is_available():
                        self.logger.warning("CUDA not available, will use CPU")
                        self.device = torch.device("cpu")
                    else:
                        # Use first GPU
                        self.device = torch.device("cuda:0")
                        self.logger.info(f"Using device: {self.device}")

                        # Display GPU information
                        gpu_name = torch.cuda.get_device_name(0)
                        gpu_memory = (
                            torch.cuda.get_device_properties(0).total_memory / 1e9
                        )
                        self.logger.info(
                            f"GPU: {gpu_name}, Memory: {gpu_memory:.2f} GB"
                        )

                    try:
                        # Load processor
                        self.logger.info("Loading processor...")
                        self.judge_processor = AutoProcessor.from_pretrained(
                            llama_guard_path
                        )

                        # Load model to specified device
                        self.logger.info("Loading model...")
                        self.judge_model = AutoModelForVision2Seq.from_pretrained(
                            llama_guard_path,
                            torch_dtype=torch.bfloat16,
                            device_map=None,  # Don't use automatic device mapping to avoid competition
                        )

                        # Manually move model to device
                        self.judge_model = self.judge_model.to(self.device)
                        self.judge_model.eval()

                        # Log model information
                        model_params = sum(
                            p.numel() for p in self.judge_model.parameters()
                        )
                        self.logger.info(
                            f"Model loaded, parameters: {model_params / 1e9:.2f}B"
                        )
                        self.logger.info(
                            f"Model device: {next(self.judge_model.parameters()).device}"
                        )

                        # Mark model as loaded
                        self._model_loaded = True

                    except Exception as e:
                        self.logger.error(f"Model loading failed: {e}")
                        raise

    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:
        """Apply defense method - thread-safe version"""

        target_model_name = self.config["target_model_name"]
        # Use global function to get model config
        model_config = get_model_config(target_model_name)
        if not model_config:
            # If global lookup fails, try to get from local config
            model_params = self.config.get("model_params", {})
            model_config = model_params.get(target_model_name, {})
        target_model = UNIFIED_REGISTRY.create_model(target_model_name, model_config)
        idx = test_case.test_case_id
        attack_image_path = test_case.image_path
        attack_prompt = test_case.prompt

        # Step 1: Get original output from victim model
        self.logger.debug(f"[{idx}] Step 1: Get victim model output")
        original_output = generate_output(
            attack_image_path, attack_prompt, target_model
        )

        # Step 2: Build Llama-Guard-3 input
        self.logger.debug(f"[{idx}] Step 2: Build input")
        image = Image.open(attack_image_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": attack_prompt},
                    {"type": "image"},
                ],
            },
            {
                "role": "agent",
                "content": [{"type": "text", "text": original_output}],
            },
        ]

        # Step 3: Call Llama-Guard-3 for judgment (protected by lock)
        self.logger.debug(f"[{idx}] Step 3: Prepare model input")
        input_prompt = self.judge_processor.apply_chat_template(
            conversation, return_tensors="pt"
        )

        inputs = self.judge_processor(
            text=input_prompt, images=image, return_tensors="pt"
        ).to(self.device)

        prompt_len = len(inputs["input_ids"][0])

        # Key: Use lock to protect model inference
        self.logger.debug(
            f"[{idx}] Waiting for inference lock, thread: {threading.get_ident()}"
        )
        with self._inference_lock:
            self.logger.debug(
                f"[{idx}] Starting inference, thread: {threading.get_ident()}"
            )

            # Use no_grad to reduce memory usage
            with torch.no_grad():
                judge_outputs = self.judge_model.generate(
                    **inputs,
                    max_new_tokens=20,
                    pad_token_id=0,
                    do_sample=False,  # Greedy decoding, faster and more stable
                    num_beams=1,  # Don't use beam search
                    use_cache=True,  # Enable cache
                    temperature=0.0,  # Deterministic output
                )

            generated_tokens = judge_outputs[:, prompt_len:]
            judge_response = self.judge_processor.decode(generated_tokens[0])

        # Step 4: Make decision based on judgment result
        self.logger.debug(f"[{idx}] Step 4: Make decision based on judgment result")
        if "unsafe" in judge_response.lower():
            defended_case = self.create_defended_case(
                test_case=test_case,
                defended_prompt=test_case.prompt,
                defended_image_path=test_case.image_path,
                metadata={"should_return_default": True},
            )
        else:
            defended_case = self.create_defended_case(
                test_case=test_case,
                defended_prompt=test_case.prompt,
                defended_image_path=test_case.image_path,
                metadata={"defense_generated_response": original_output},
            )

        return defended_case
