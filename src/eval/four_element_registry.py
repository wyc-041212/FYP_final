from __future__ import annotations

GENERIC_MODEL_KEYS = [
    "full_fusion",
    "meta_fusion",
    "route_meta_fusion",
    "route_meta_bucket_threshold",
    "bucket_expert_fusion",
    "bucket_meta_fusion",
    "uncertainty_residual_fusion",
    "semantic_fusion",
    "semantic_max_fusion",
    "generic_blend_fusion",
]

NO_TUNING_GENERIC_MODEL_KEYS = [
    "full_fusion",
    "meta_fusion",
    "route_meta_fusion",
    "generic_blend_fusion",
]

EXPORT_MODEL_KEYS = [
    "route_only",
    "patch_only",
    "route_patch_only",
    "pair_only",
    "full_fusion",
    "meta_fusion",
    "route_meta_fusion",
    "route_meta_bucket_threshold",
    "bucket_expert_fusion",
    "bucket_meta_fusion",
    "ambiguity_fusion",
    "uncertainty_residual_fusion",
    "semantic_fusion",
    "semantic_max_fusion",
    "generic_blend_fusion",
    "selective_pair_fusion",
]


def resolve_auto_generic_from_validation(
    validation: dict[str, float],
    threshold_search: dict[str, dict],
    available_models: list[str] | tuple[str, ...],
    candidate_keys: list[str] | tuple[str, ...] | None = None,
) -> dict:
    pool = GENERIC_MODEL_KEYS if candidate_keys is None else list(candidate_keys)
    candidates = [model for model in pool if model in available_models]
    best = None
    ranking = []
    for model in candidates:
        val_bacc = float(validation[model])
        search = threshold_search[model]
        val_auc = float(search["auc"])
        val_real_acc = float(search.get("real_accuracy", 0.0))
        val_acc = float(search["accuracy"])
        score = (val_bacc, val_auc, val_real_acc, val_acc)
        row = {
            "model_key": model,
            "val_balanced_accuracy": val_bacc,
            "val_auc": val_auc,
            "val_real_accuracy": val_real_acc,
            "val_accuracy": val_acc,
        }
        ranking.append(row)
        if best is None or score > best["score"]:
            best = {"score": score, **row}
    if best is None:
        raise KeyError("No generic candidate models found in payload.")
    ranking.sort(
        key=lambda row: (
            row["val_balanced_accuracy"],
            row["val_auc"],
            row["val_real_accuracy"],
            row["val_accuracy"],
        ),
        reverse=True,
    )
    return {
        "selected_model_key": best["model_key"],
        "val_balanced_accuracy": best["val_balanced_accuracy"],
        "val_auc": best["val_auc"],
        "val_real_accuracy": best["val_real_accuracy"],
        "val_accuracy": best["val_accuracy"],
        "ranking": ranking,
    }
