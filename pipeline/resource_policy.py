from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class ResourcePolicy:
    """Unified resource strategy decision.

    Current goal:
    - Force *local* models to run with a single worker but process data in batches (reuse model instance).
    - Keep API models parallel by default.
    """

    strategy: str  # "batched" | "parallel"
    max_workers: int
    batched_impl: str  # "local_model" | "defense_only" | "attack_local" | "evaluator_local" | "none"
    reason: str


def infer_model_type_from_config(model_config: Optional[Dict[str, Any]]) -> str:
    """Infer model type from config without instantiating the model.

    Project convention (as requested):
    - Use ONLY `load_model` flag to decide local vs api.
    - If `load_model: true` => local (in-process)
    - Else => api
    """
    cfg = model_config or {}
    return "local" if bool(cfg.get("load_model", False)) else "api"


def policy_for_response_generation(
    model_config: Optional[Dict[str, Any]],
    defense_config: Optional[Dict[str, Any]] = None,
    default_max_workers: int = 4,
) -> ResourcePolicy:
    mcfg = model_config or {}
    dcfg = defense_config or {}

    model_load = bool(mcfg.get("load_model", False))
    defense_load = bool(dcfg.get("load_model", False))

    # Unified rule:
    # - If either model or defense needs local loading, force batched execution and single worker
    # - Otherwise, parallel with configured max_workers
    if model_load or defense_load:
        trigger = "model.load_model=true" if model_load else "defense.load_model=true"
        batched_impl = "local_model" if model_load else "defense_only"
        return ResourcePolicy(
            strategy="batched",
            max_workers=1,
            batched_impl=batched_impl,
            reason=f"{trigger} -> batched + max_workers=1 (reuse instance, avoid repeated loads)",
        )

    return ResourcePolicy(
        strategy="parallel",
        max_workers=int(default_max_workers),
        batched_impl="none",
        reason="no local loading flags -> parallel",
    )


def policy_for_test_case_generation(
    attack_config: Optional[Dict[str, Any]],
    default_max_workers: int = 4,
) -> ResourcePolicy:
    """Policy for test case generation stage.

    Project convention:
    - Use ONLY `attack_config.load_model` to decide local loading.
    - If `load_model: true` => batched + max_workers=1 (reuse attack instance)
    - Else => parallel + configured max_workers
    """
    cfg = attack_config or {}
    attack_load = bool(cfg.get("load_model", False))
    if attack_load:
        return ResourcePolicy(
            strategy="batched",
            max_workers=1,
            batched_impl="attack_local",
            reason="attack.load_model=true -> batched + max_workers=1 (reuse attack instance)",
        )
    return ResourcePolicy(
        strategy="parallel",
        max_workers=int(default_max_workers),
        batched_impl="none",
        reason="attack.load_model!=true -> parallel",
    )


def policy_for_evaluation(
    evaluator_config: Optional[Dict[str, Any]],
    default_max_workers: int = 4,
) -> ResourcePolicy:
    """Policy for evaluation stage.

    Project convention:
    - Use ONLY `evaluator_config.load_model` to decide local loading.
    - If `load_model: true` => batched + max_workers=1 (reuse evaluator instance)
    - Else => parallel + configured max_workers
    """
    cfg = evaluator_config or {}
    evaluator_load = bool(cfg.get("load_model", False))
    if evaluator_load:
        return ResourcePolicy(
            strategy="batched",
            max_workers=1,
            batched_impl="evaluator_local",
            reason="evaluator.load_model=true -> batched + max_workers=1 (reuse evaluator instance)",
        )
    return ResourcePolicy(
        strategy="parallel",
        max_workers=int(default_max_workers),
        batched_impl="none",
        reason="evaluator.load_model!=true -> parallel",
    )


