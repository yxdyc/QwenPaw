#!/usr/bin/env python3
"""Frontier-model lifecycle claims as a small, executable evidence contract."""

from dataclasses import dataclass
import hashlib
import json
from typing import Optional


@dataclass(frozen=True)
class ModelCard:
    name: str
    family: str
    base_lineage: str
    total_b: Optional[float]
    active_b: Optional[float]
    external_capacity_b: float
    architecture: str
    pretraining: str
    posttraining: str
    posttraining_recipe_disclosed: bool
    posttraining_delta_claim: bool
    weights_open: bool
    full_recipe_open: bool


CARDS = {
    "DeepSeek-V4-Pro": ModelCard(
        "DeepSeek-V4-Pro", "DeepSeek-V4", "deepseek-v4-pro", 1600, 49, 0,
        "CSA/HCA hybrid attention; mHC; MoE",
        ">32T tokens; Muon is described in the technical report",
        "specialist RL followed by multi-teacher full-vocabulary reverse-KL OPD",
        True, False, True, False,
    ),
    "Qwen3.8-Flash-Next": ModelCard(
        "Qwen3.8-Flash-Next", "Qwen3.8", "qwen3.8-flash-next", 125, 6, 51,
        "3 Gated DeltaNet : 1 sparse-attention layer; MoE; gated residual",
        "architecture candidates judged on quality, efficiency, and stability",
        "the public report does not disclose a detailed SFT/RL recipe",
        False, False, True, False,
    ),
    "Kimi K3": ModelCard(
        "Kimi K3", "Kimi K3", "kimi-k3", 2800, 104, 0,
        "KDA; attention residuals; Stable LatentMoE; native vision",
        "1M-context native multimodal base model",
        "reasoning, agentic, and general RL with long-horizon infrastructure",
        True, False, True, False,
    ),
    "GLM-5.3": ModelCard(
        "GLM-5.3", "GLM-5.3", "glm-5.2-base", None, None, 0,
        "same base-model lineage as GLM-5.2 according to the model card",
        "no new base-model claim for the 5.3 delta",
        "official card attributes the 5.3 gains to post-training",
        False, True, True, False,
    ),
    "GLM-5.3-Flash": ModelCard(
        "GLM-5.3-Flash", "GLM-5.3", "glm-5.3-flash-new-base", 320, 18, 0,
        "hybrid sparse/linear attention; mHC; native multimodality",
        "distinct base trained on 30T multimodal tokens",
        "post-training exists, but is not an isolated delta versus GLM-5.3",
        False, False, True, False,
    ),
    "GPT-6 Astra": ModelCard(
        "GPT-6 Astra", "GPT-6", "undisclosed-gpt-6-astra", None, None, 0,
        "undisclosed; official docs expose an API behavior contract",
        "undisclosed", "undisclosed", False, False, False, False,
    ),
    "Claude Fable 5.1": ModelCard(
        "Claude Fable 5.1", "Claude 5", "undisclosed-fable-5-1", None, None, 0,
        "undisclosed; official pages expose an API and safeguard contract",
        "undisclosed", "undisclosed", False, False, False, False,
    ),
}

ALIASES = {
    "dpsk": "DeepSeek-V4-Pro",
    "qwen3.8-next": "Qwen3.8-Flash-Next",
    "kimi3": "Kimi K3",
    "glm5.3": "GLM-5.3",
    "gpt-6": "GPT-6 Astra",
    "claude5.1": "Claude Fable 5.1",
}


def normalize(name: str) -> str:
    canonical = ALIASES.get(name.lower(), name)
    if canonical not in CARDS:
        raise ValueError(f"unknown model identity: {name}")
    return canonical


def active_ratio(card: ModelCard) -> Optional[float]:
    if card.total_b is None or card.active_b is None:
        return None
    return card.active_b / card.total_b


def require_posttraining_recipe(card: ModelCard) -> str:
    if not card.posttraining_recipe_disclosed:
        raise ValueError(f"post-training recipe is undisclosed: {card.name}")
    return card.posttraining


def attribution(candidate: ModelCard, parent: ModelCard) -> str:
    same_base = candidate.base_lineage == parent.base_lineage
    return "posttraining-focused" if same_base and candidate.posttraining_delta_claim else "confounded"


def approve_architecture_trial(signals: dict[str, bool]) -> bool:
    required = {"quality", "efficiency", "stability"}
    return required.issubset(signals) and all(signals[key] for key in required)


def rejected(callable_) -> bool:
    try:
        callable_()
    except ValueError:
        return True
    return False


def main() -> None:
    qwen = CARDS[normalize("qwen3.8-next")]
    glm53 = CARDS[normalize("glm5.3")]
    glm52_parent = ModelCard(
        "GLM-5.2", "GLM-5", "glm-5.2-base", None, None, 0,
        "parent architecture", "parent base", "parent post-training", True, False, True, False,
    )
    flash = CARDS["GLM-5.3-Flash"]
    closed = (CARDS["GPT-6 Astra"], CARDS["Claude Fable 5.1"])

    checks = {
        "aliases_normalized": all(normalize(alias) == target for alias, target in ALIASES.items()),
        "unknown_identity_rejected": rejected(lambda: normalize("qwen-next-unspecified")),
        "active_is_not_total": all(
            card.active_b < card.total_b
            for card in CARDS.values()
            if card.active_b is not None and card.total_b is not None
        ),
        "qwen_external_capacity_not_active": qwen.external_capacity_b == 51 and qwen.active_b == 6,
        "undisclosed_recipe_rejected": rejected(lambda: require_posttraining_recipe(qwen)),
        "open_weights_not_full_recipe": all(
            not card.full_recipe_open for card in CARDS.values() if card.weights_open
        ),
        "closed_parameters_remain_unknown": all(
            card.total_b is None and card.active_b is None for card in closed
        ),
        "api_access_is_not_open_weights": all(not card.weights_open for card in closed),
        "same_base_delta_enters_replay": attribution(glm53, glm52_parent) == "posttraining-focused",
        "new_base_comparison_is_confounded": attribution(flash, glm53) == "confounded",
        "loss_only_architecture_rejected": not approve_architecture_trial({"quality": True}),
        "three_axis_architecture_accepted": approve_architecture_trial(
            {"quality": True, "efficiency": True, "stability": True}
        ),
    }
    assert all(checks.values()), checks

    print("frontier lifecycle L0 — architecture → pretrain → post-train → serve → evaluate")
    print("snapshot=2026-09-08 | standard-library teaching contract | no model execution")
    for card in CARDS.values():
        ratio = active_ratio(card)
        ratio_text = "undisclosed" if ratio is None else f"{ratio:.3%}"
        print(
            f"CARD {card.name}: active/total={ratio_text}; "
            f"posttrain_recipe_disclosed={card.posttraining_recipe_disclosed}; "
            f"full_recipe_open={card.full_recipe_open}"
        )
    print(f"PAIR GLM-5.3 <- GLM-5.2: {attribution(glm53, glm52_parent)}")
    print(f"PAIR GLM-5.3-Flash <> GLM-5.3: {attribution(flash, glm53)}")
    print("FAILURES: unknown alias, undisclosed recipe, loss-only gate, and confounded attribution rejected")
    print(f"CHECKS {sum(checks.values())}/{len(checks)}")

    payload = {
        "schema_version": "1.0",
        "module": "frontier_model_lifecycle_l0",
        "metrics": {
            "model_cards": len(CARDS),
            "known_active_ratios": sum(active_ratio(card) is not None for card in CARDS.values()),
            "posttraining_recipes_disclosed": sum(
                card.posttraining_recipe_disclosed for card in CARDS.values()
            ),
            "posttraining_focused_pairs": 1,
            "confounded_pairs_detected": 1,
        },
        "checks": checks,
        "evidence_boundary": (
            "Dated metadata and claim-typing simulation; no weights, training, benchmark, "
            "throughput, or production capability were tested."
        ),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["digest"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    print("RESULT_JSON=" + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
