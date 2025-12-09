"""
New architecture defense methods package
Supports dynamic import, missing defense methods are set to None
"""

from .base_defense import BaseDefense

# Dynamically import all defense methods, set missing ones to None
try:
    from .adashield import AdaShieldDefense
except ImportError:
    AdaShieldDefense = None

try:
    from .dps import DPSDefense
except ImportError:
    DPSDefense = None

try:
    from .ecso import ECSODefense
except ImportError:
    ECSODefense = None

try:
    from .jailguard import JailGuardDefense
except ImportError:
    JailGuardDefense = None

try:
    from .llavaguard import LlavaGuardDefense
except ImportError:
    LlavaGuardDefense = None

try:
    from .qguard_utils import QGuardDefense
except ImportError:
    QGuardDefense = None

try:
    from .shieldlm import ShieldLMDefense
except ImportError:
    ShieldLMDefense = None

try:
    from .uniguard import UniguardDefense
except ImportError:
    UniguardDefense = None

try:
    from .cider import CIDERDefense
except ImportError:
    CIDERDefense = None

try:
    from .mllm_protector import MLLMProtectorDefense
except ImportError:
    MLLMProtectorDefense = None

try:
    from .llama_guard_3 import LlamaGuard3Defense
except ImportError:
    LlamaGuard3Defense = None

try:
    from .hiddendetect import HiddenDetectDefense
except ImportError:
    HiddenDetectDefense = None

try:
    from .guardreasoner_vl import GuardReasonerVLDefense
except ImportError:
    GuardReasonerVLDefense = None

try:
    from .llama_guard_4 import LlamaGuard4Defense
except ImportError:
    LlamaGuard4Defense = None

try:
    from .vlguard import VLGuardDefense
except ImportError:
    VLGuardDefense = None

try:
    from .coca import CoCADefense
except ImportError:
    CoCADefense = None

# Build __all__ list, including all variables (including those that may be None)
__all__ = [
    "BaseDefense",
    "AdaShieldDefense",
    "DPSDefense",
    "ECSODefense",
    "JailGuardDefense",
    "LlavaGuardDefense",
    "QGuardDefense",
    "ShieldLMDefense",
    "UniguardDefense",
    "CIDERDefense",
    "MLLMProtectorDefense",
    "LlamaGuard3Defense",
    "HiddenDetectDefense",
    "GuardReasonerVLDefense",
    "LlamaGuard4Defense",
    "VLGuardDefense",
    "SPAVLDefense",
    "SafeRLHFVDefense",
    "CoCADefense",
]
