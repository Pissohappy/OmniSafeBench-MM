def initialize_registry_lazy_imports():
    """Initialize lazy loading mappings to unified registry"""

    # Attack method lazy loading mappings
    attack_lazy_imports = {
        "figstep": ("attacks.figstep.attack", "FigStepAttack"),
        "cs_dj": ("attacks.cs_dj.attack", "CSDJAttack"),
        "hades": ("attacks.hades.attack", "HadesAttack"),
        "qr": ("attacks.qr.attack", "QRAttack"),
        "mml": ("attacks.mml.attack", "MMLAttack"),
        "si": ("attacks.si.attack", "SIAttack"),
        "jood": ("attacks.jood.attack", "JOODAttack"),
        "himrd": ("attacks.himrd.attack", "HIMRDAttack"),
        "bap": ("attacks.bap.attack", "BAPAttack"),
        "visual_adv": ("attacks.visual_adv.attack", "VisualAdvAttack"),
        "viscra": ("attacks.viscra.attack", "VisCRAAttack"),
        "umk": ("attacks.umk.attack", "UMKAttack"),
        "pba": ("attacks.pba.attack", "PBAAttack"),
        "imgjp": ("attacks.imgjp.attack", "ImgJPAttack"),
    }

    # Model lazy loading mappings
    model_lazy_imports = {
        "openai": ("models.openai_model", "OpenAIModel"),
        "google": ("models.google_model", "GoogleModel"),
        "anthropic": ("models.anthropic_model", "AnthropicModel"),
        "qwen": ("models.qwen_model", "QwenModel"),
        "doubao": ("models.doubao_model", "DoubaoModel"),
        "vllm": ("models.vllm_model", "VLLMModel"),
        "mistral": ("models.mistral_model", "MistralModel"),
        "any": (
            "models.openai_model",
            "OpenAIModel",
        ),  # "any" uses OpenAI-compatible API
    }

    # Defense method lazy loading mappings
    defense_lazy_imports = {
        "adashield": ("defenses.adashield", "AdaShieldDefense"),
        "dps": ("defenses.dps", "DPSDefense"),
        "ecso": ("defenses.ecso", "ECSODefense"),
        "jailguard": ("defenses.jailguard", "JailGuardDefense"),
        "llavaguard": ("defenses.llavaguard", "LlavaGuardDefense"),
        "qguard": ("defenses.qguard", "QGuardDefense"),
        "shieldlm": ("defenses.shieldlm", "ShieldLMDefense"),
        "uniguard": ("defenses.uniguard", "UniguardDefense"),
        "cider": ("defenses.cider", "CIDERDefense"),
        "mllm_protector": ("defenses.mllm_protector", "MLLMProtectorDefense"),
        "llama_guard_3": ("defenses.llama_guard_3", "LlamaGuard3Defense"),
        "hiddendetect": ("defenses.hiddendetect", "HiddenDetectDefense"),
        "guardreasoner_vl": ("defenses.guardreasoner_vl", "GuardReasonerVLDefense"),
        "llama_guard_4": ("defenses.llama_guard_4", "LlamaGuard4Defense"),
        "vlguard": ("defenses.vlguard", "VLGuardDefense"),
        "spa_vl": ("defenses.spa_vl", "SPAVLDefense"),
        "saferlhf_v": ("defenses.saferlhf_v", "SafeRLHFVDefense"),
        "coca": ("defenses.coca", "CoCADefense"),
    }

    # Evaluator lazy loading mappings
    evaluator_lazy_imports = {
        "default_judge": ("evaluators.default_judge", "DefaultJudge"),
    }

    return {
        "attacks": attack_lazy_imports,
        "models": model_lazy_imports,
        "defenses": defense_lazy_imports,
        "evaluators": evaluator_lazy_imports,
    }
