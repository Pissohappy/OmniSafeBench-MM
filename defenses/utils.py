import json
import os

from tqdm import tqdm
import time
from torchvision import transforms
import random
import io
from openai import OpenAI
import numpy as np
from math import ceil
import requests
from io import BytesIO
import torch.nn.functional as N

import base64
from pathlib import Path
from PIL import Image
import torch
import re
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)


def generate_output(
    attack_image_path,
    attack_prompt,
    client,
    max_new_tokens=512,
    temperature=0.0,
    **kwargs,
):

    # -------------------- Unified model handling --------------------
    # All models (both closed-source and open-source) are served via vLLM
    # The only difference is the base URL and API key configuration

    # For all models served via vLLM

    messages = []
    # get the image
    if attack_image_path:
        b64_image = None
        try:
            with open(attack_image_path, "rb") as f:
                b64_image = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            return f"[generate_output] Failed to read image {attack_image_path}: {e}"

        file_extension = Path(attack_image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }
        # Set MIME type based on file extension, use default if unknown
        mime_type = mime_map.get(file_extension, "image/jpeg")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": attack_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                    },
                ],
            }
        ]
    else:  # Plain text input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": attack_prompt},
                ],
            }
        ]

    response = client.generate(
        messages, max_tokens=max_new_tokens, temperature=temperature, **kwargs
    )
    if type(response) == str:
        return response
    response = response.choices[0].message.content

    # print("response:", response, "\n")
    return response


# used for defense method uniguard
def denormalize(images, device="cpu"):
    """Actual function: Perform standard Z-score normalization. Formula: (image - mean) / std"""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    new_images = (images - mean[None, :, None, None]) / std[None, :, None, None]
    return new_images


# used for defense method uniguard
def normalize(images, device="cpu"):
    """Actual function: Perform denormalization. Formula: (image * std) + mean"""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    new_images = (images * std[None, :, None, None]) + mean[None, :, None, None]
    return new_images


# used for defense method uniguard
def add_patch_to_image_universal(
    image_path: str, patch_path: str, device: str
) -> (Image.Image, torch.Tensor):
    """
    Apply patch to image in a universal way, without requiring a specific vis_processor.
    Returns: (PIL image viewable by humans, Tensor directly inputtable to model)
    """
    if not image_path or not os.path.exists(image_path):
        print(
            f"  [Warning] Image path invalid or does not exist: {image_path}. Skipping image processing."
        )
        return None, None
    if not patch_path or not os.path.exists(patch_path):
        print(
            f"  [Warning] Patch path invalid or does not exist: {patch_path}. Skipping image processing."
        )
        return None, None

    try:
        image = Image.open(image_path).convert("RGB")
        patch_tensor = torch.load(patch_path, map_location=device)

        if patch_tensor.dim() == 4:
            patch_tensor = patch_tensor.squeeze(0)
        if patch_tensor.dim() != 3:
            raise ValueError(
                f"Patch tensor dimension should be 3 (C, H, W), currently {patch_tensor.dim()}"
            )
    except Exception as e:
        print(f"  [Failed] Error loading file: {e}")
        return None, None

    _, target_height, target_width = patch_tensor.shape
    preprocess = transforms.Compose(
        [
            transforms.Resize((target_height, target_width)),
            transforms.ToTensor(),
        ]
    )

    image_tensor_0_1 = preprocess(image).unsqueeze(0).to(device)
    image_tensor_model_ready = denormalize(image_tensor_0_1, device=device)
    image_tensor_for_patching = normalize(image_tensor_model_ready, device=device)

    patched_tensor_0_1 = image_tensor_for_patching + patch_tensor
    clipped_tensor_0_1 = torch.clamp(patched_tensor_0_1, 0, 1)

    to_pil_transform = transforms.ToPILImage()
    final_image = to_pil_transform(clipped_tensor_0_1.squeeze(0))

    final_model_ready_tensor = denormalize(clipped_tensor_0_1, device=device)

    # print("  [Success] Image patch applied.")
    return final_image, final_model_ready_tensor


# used for defense method hiddendetect
def score_and_generate_hiddendetect(
    sample,
    model,
    tokenizer,
    image_processor,
    refusal_token_ids,
    threshold,
    s=16,
    e=29,
    max_new_tokens=2048,
):
    """
    Calculate HiddenDetect score for a single sample, if passed then directly generate output.

    Returns: (score, output_text)
    """
    # --- 1. Prepare input (logic from hiddendetect_test) ---
    model_name = model.config._name_or_path

    # Construct prompt
    conv = conv_templates[find_conv_mode(model_name)].copy()
    qs = sample["txt"]
    if sample.get("img"):
        qs = adjust_query_for_images(qs, model.config)
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    # Process images
    images_tensor, image_sizes = None, None
    if sample.get("img"):
        images_tensor, image_sizes = prepare_imgs_tensor_both_cases(
            sample, image_processor, model.config
        )
        if images_tensor is not None:
            images_tensor = images_tensor.to(model.device, dtype=torch.float16)

    # --- 2. Calculate score (logic from hiddendetect_test) ---
    with torch.no_grad():
        outputs = model(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            output_hidden_states=True,
        )

    # Extract norm and lm_head
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
    else:  # Applicable to new model structures like Llama-3
        norm = model.norm

    lm_head = model.lm_head

    F = []
    for r in outputs.hidden_states[1:]:  # Skip embedding layer
        layer_output = norm(r)
        logits = lm_head(layer_output)
        next_token_logits = logits[:, -1, :]
        reference_tokens = refusal_token_ids.to(next_token_logits.device)
        cos_sim = N.cosine_similarity(next_token_logits, reference_tokens)
        F.append(cos_sim.item())

    F_selected = F[s : e + 1]
    score = np.trapz(np.array(F_selected)) if F_selected else 0.0

    # --- 3. Judge and generate ---
    if score > threshold:
        # Score exceeds threshold, intercept and return refusal message
        output_text = "I'm sorry, I can't assist with that."
        print("Response refused due to high HiddenDetect score.")
        print(f"HiddenDetect score: {score}, output: {output_text}")
        return score, output_text
    else:
        # Score passed, use same input for generation
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=False,  # Ensure deterministic output
                temperature=0.0,  # Greedy decoding
                max_new_tokens=max_new_tokens,
                use_cache=False,  # Disable cache to avoid cache_position compatibility issues
                cache_position=None,
            )

        # Decode output
        # We need to remove input part from output
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        outputs = outputs.strip().replace("\n", " ")
        print("Response generated successfully.")
        print(f"HiddenDetect score: {score}, output: {outputs}")

        return score, outputs


def find_conv_mode(model_name):
    # ... (keep as is)
    if "llama-2" in model_name.lower():
        return "llava_llama_2"
    elif "mistral" in model_name.lower():
        return "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        return "chatml_direct"
    elif "v1" in model_name.lower():
        return "llava_v1"
    elif "mpt" in model_name.lower():
        return "mpt"
    else:
        return "llava_v0"


def adjust_query_for_images(qs, model_config):
    # ... (keep as is, added model_config parameter)
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model_config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model_config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    return qs


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def prepare_imgs_tensor_both_cases(sample, image_processor, model_config):
    # ... (keep as is, but pass image_processor and model_config)
    img_prompt = [load_image(sample["img"])] if sample.get("img") else None
    if img_prompt is None:
        return None, None
    images_size = [img.size for img in img_prompt if img is not None]
    images_tensor = process_images(img_prompt, image_processor, model_config)
    return images_tensor, images_size


def generate_with_coca(case, model, processor, safety_principle, alpha, max_new_tokens):
    """
    Generate output for a single sample using CoCA defense method.

    Args:
        case (dict): Sample dictionary containing 'attack_prompt' and 'attack_image_path'.
        model: Pre-loaded victim model (e.g., LlavaForConditionalGeneration).
        processor: Corresponding processor.
        safety_principle (str): Safety principle used for guidance.
        alpha (float): CoCA amplification coefficient.
        max_new_tokens (int): Maximum generation length.

    Returns:
        str: Generated text after CoCA defense.
    """
    user_question = case.get("attack_prompt", "")
    image_path = case.get("attack_image_path", None)

    # --- 1. Prepare two types of inputs ---
    # a) Input with safety principle
    prompt_text_with_safety = f"{safety_principle} {user_question}"
    conversation_with_safety = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text_with_safety},
                {"type": "image"},
            ],
        },
    ]
    prompt_with_safety = processor.apply_chat_template(
        conversation_with_safety, add_generation_prompt=True
    )

    # b) Input without safety principle
    conversation_without_safety = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_question},
                {"type": "image"},
            ],
        },
    ]
    prompt_without_safety = processor.apply_chat_template(
        conversation_without_safety, add_generation_prompt=True
    )

    # --- 2. Load image and create input tensors ---
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return "[CoCA Error: Failed to load image]"

    inputs_with_safety = processor(
        images=raw_image, text=prompt_with_safety, return_tensors="pt"
    ).to(model.device, dtype=torch.float16)

    inputs_without_safety = processor(
        images=raw_image, text=prompt_without_safety, return_tensors="pt"
    ).to(model.device, dtype=torch.float16)

    # --- 3. Implement CoCA's custom generation loop ---
    input_ids_with_safety = inputs_with_safety["input_ids"]
    input_ids_without_safety = inputs_without_safety["input_ids"]
    pixel_values = inputs_with_safety["pixel_values"]
    eos_token_id = processor.tokenizer.eos_token_id

    with torch.inference_mode():
        for i in range(max_new_tokens):
            # Get logits from both branches
            outputs_with_safety = model(
                input_ids=input_ids_with_safety, pixel_values=pixel_values
            )
            logits_with_safety = outputs_with_safety.logits[:, -1, :]

            outputs_without_safety = model(
                input_ids=input_ids_without_safety, pixel_values=pixel_values
            )
            logits_without_safety = outputs_without_safety.logits[:, -1, :]

            # Calculate Safety Delta and apply CoCA calibration
            safety_delta = logits_with_safety - logits_without_safety
            calibrated_logits = logits_with_safety + alpha * safety_delta

            # Select next token
            next_token_id = torch.argmax(calibrated_logits, dim=-1).unsqueeze(-1)

            # Synchronously update both sequences
            input_ids_with_safety = torch.cat(
                [input_ids_with_safety, next_token_id], dim=-1
            )
            input_ids_without_safety = torch.cat(
                [input_ids_without_safety, next_token_id], dim=-1
            )

            # Check for end token
            if next_token_id.item() == eos_token_id:
                break

    # --- 4. Decode and return result ---
    initial_prompt_len = inputs_with_safety["input_ids"].shape[1]

    if input_ids_with_safety.shape[1] > initial_prompt_len:
        generated_ids = input_ids_with_safety[:, initial_prompt_len:]
        final_output = processor.decode(generated_ids[0], skip_special_tokens=True)
        print("CoCA generated new content.")
        print(f"Output: {final_output}")
    else:
        final_output = (
            "CoCA did not generate any new content."  # No new content generated
        )
        print("CoCA did not generate any new content.")

    return final_output.strip()
