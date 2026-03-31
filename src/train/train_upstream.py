#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from prepare.cache import EmbeddingCache
from utils.device import resolve_torch_device

EPS = 1e-8
EFS_LABEL = "EFS"
REAL_LABEL = "REAL"
BASE_ALL_GROUPS = ["EFS", "FS", "FR", "FE", "REAL"]
BASE_PAIR_GROUPS = ["FS", "FR", "FE"]
NO_FR_ALL_GROUPS = ["EFS", "FS", "FE", "REAL"]
NO_FR_PAIR_GROUPS = ["FS", "FE"]
ALL_GROUPS = list(BASE_ALL_GROUPS)
PAIR_GROUPS = list(BASE_PAIR_GROUPS)
GROUP_TO_IDX = {group: idx for idx, group in enumerate(ALL_GROUPS)}
PAIR_TO_IDX = {group: idx for idx, group in enumerate(PAIR_GROUPS)}


def configure_group_scheme(no_fr: bool) -> None:
    global ALL_GROUPS, PAIR_GROUPS, GROUP_TO_IDX, PAIR_TO_IDX
    ALL_GROUPS = list(NO_FR_ALL_GROUPS if no_fr else BASE_ALL_GROUPS)
    PAIR_GROUPS = list(NO_FR_PAIR_GROUPS if no_fr else BASE_PAIR_GROUPS)
    GROUP_TO_IDX = {group: idx for idx, group in enumerate(ALL_GROUPS)}
    PAIR_TO_IDX = {group: idx for idx, group in enumerate(PAIR_GROUPS)}


class HybridManifoldModel(nn.Module):
    def __init__(
        self,
        dim: int,
        real_rank: int,
        efs_rank: int,
        real_center_init: np.ndarray,
        efs_center_init: np.ndarray,
        real_basis_init: np.ndarray,
        efs_basis_init: np.ndarray,
        fake_offset_init: np.ndarray,
        delta_proto_init: np.ndarray,
        labels: Sequence[str] | None = None,
        pair_groups: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.labels = tuple(labels or ALL_GROUPS)
        self.pair_groups = tuple(pair_groups or PAIR_GROUPS)
        self.linear = nn.Linear(dim, dim, bias=False)
        self.classifier = nn.Linear(dim, len(self.labels))
        self.real_center = nn.Parameter(torch.from_numpy(real_center_init.astype(np.float32)))
        self.efs_center = nn.Parameter(torch.from_numpy(efs_center_init.astype(np.float32)))
        self.real_basis_raw = nn.Parameter(torch.from_numpy(real_basis_init.astype(np.float32)))
        self.efs_basis_raw = nn.Parameter(torch.from_numpy(efs_basis_init.astype(np.float32)))
        self.fake_offsets = nn.Parameter(torch.from_numpy(fake_offset_init.astype(np.float32)))
        self.delta_prototypes = nn.Parameter(torch.from_numpy(delta_proto_init.astype(np.float32)))
        self.logit_scale = nn.Parameter(torch.full((len(self.labels),), -2.0, dtype=torch.float32))
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dim, dtype=torch.float32))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def real_basis(self) -> torch.Tensor:
        q, _ = torch.linalg.qr(self.real_basis_raw, mode="reduced")
        return q

    def efs_basis(self) -> torch.Tensor:
        q, _ = torch.linalg.qr(self.efs_basis_raw, mode="reduced")
        return q

    def project(self, y: torch.Tensor, center: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        shifted = y - center[None, :]
        coeff = shifted @ basis
        proj = coeff @ basis.t()
        return center[None, :] + proj

    def all_offsets(self) -> torch.Tensor:
        offsets = []
        for label in self.labels:
            if label == EFS_LABEL:
                offsets.append(self.efs_center[None, :])
            elif label == REAL_LABEL:
                offsets.append(self.real_center[None, :])
            else:
                pair_idx = self.pair_groups.index(label)
                offsets.append((self.real_center + self.fake_offsets[pair_idx])[None, :])
        return torch.cat(offsets, dim=0)

    def manifold_dist_sq(self, y: torch.Tensor) -> torch.Tensor:
        real_basis = self.real_basis()
        efs_basis = self.efs_basis()
        offsets = self.all_offsets()
        dists = []
        for idx, label in enumerate(self.labels):
            basis = efs_basis if label == EFS_LABEL else real_basis
            shifted = y - offsets[idx][None, :]
            coeff = shifted @ basis
            proj = coeff @ basis.t()
            residual = shifted - proj
            dists.append(torch.sum(residual * residual, dim=1))
        return torch.stack(dists, dim=1)

    def fused_logits(self, x: torch.Tensor, temperature: float, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.transform(x)
        manifold_logits = -alpha * self.manifold_dist_sq(y) / temperature
        linear_logits = self.classifier(x)
        gamma = F.softplus(self.logit_scale)[None, :]
        fused = linear_logits + gamma * manifold_logits
        return y, linear_logits, manifold_logits, fused


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def stratified_split(y: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit_idx = []
    val_idx = []
    for label in sorted(np.unique(y).tolist()):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        val_idx.extend(idx[:n_val].tolist())
        fit_idx.extend(idx[n_val:].tolist())
    return np.asarray(fit_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def pca_basis(x: np.ndarray, rank: int) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:rank].T.astype(np.float32, copy=False)
    if basis.shape[1] < rank:
        pad = np.zeros((basis.shape[0], rank - basis.shape[1]), dtype=np.float32)
        basis = np.concatenate([basis, pad], axis=1)
    return basis


def robust_center(x: np.ndarray, trim: float = 0.1) -> np.ndarray:
    center = x.mean(axis=0, keepdims=True)
    dist = np.sum((x - center) ** 2, axis=1)
    keep = max(32, int(round(len(x) * (1.0 - trim))))
    idx = np.argsort(dist)[:keep]
    return x[idx].mean(axis=0).astype(np.float32)


def load_split(cache_root: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    methods: List[np.ndarray] = []
    split_root = cache_root / split

    if split == "DF40_test_ood":
        for group in ALL_GROUPS[:-1]:
            group_dir = split_root / group
            if not group_dir.exists():
                continue
            for method_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
                fake_cache = EmbeddingCache.from_npz(method_dir / "cls_fake.npz")
                real_cache = EmbeddingCache.from_npz(method_dir / "cls_real.npz")
                method = method_dir.name
                vectors.append(fake_cache.cls.astype(np.float32, copy=False))
                labels.append(np.full(len(fake_cache.cls), group, dtype=object))
                methods.append(np.full(len(fake_cache.cls), method, dtype=object))
                vectors.append(real_cache.cls.astype(np.float32, copy=False))
                labels.append(np.full(len(real_cache.cls), "REAL", dtype=object))
                methods.append(np.full(len(real_cache.cls), f"real::{method}", dtype=object))
        return np.concatenate(vectors, axis=0), np.concatenate(labels, axis=0), np.concatenate(methods, axis=0)

    for group in ALL_GROUPS[:-1]:
        group_dir = split_root / group
        for cache_path in sorted(group_dir.glob("cls_*.npz")):
            cache = EmbeddingCache.from_npz(cache_path)
            method = cache_path.stem.replace("cls_", "")
            fake_mask = cache.label.astype(np.int64) == 1
            real_mask = ~fake_mask
            if np.any(fake_mask):
                vectors.append(cache.cls[fake_mask].astype(np.float32, copy=False))
                labels.append(np.full(int(np.sum(fake_mask)), group, dtype=object))
                methods.append(np.full(int(np.sum(fake_mask)), method, dtype=object))
            if np.any(real_mask):
                vectors.append(cache.cls[real_mask].astype(np.float32, copy=False))
                labels.append(np.full(int(np.sum(real_mask)), "REAL", dtype=object))
                methods.append(np.full(int(np.sum(real_mask)), f"real::{method}", dtype=object))
    return np.concatenate(vectors, axis=0), np.concatenate(labels, axis=0), np.concatenate(methods, axis=0)


def load_pair_train(cache_root: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fake_vecs: List[np.ndarray] = []
    real_vecs: List[np.ndarray] = []
    labels: List[int] = []
    split_root = cache_root / split
    for group in PAIR_GROUPS:
        group_dir = split_root / group
        for cache_path in sorted(group_dir.glob("cls_*.npz")):
            cache = EmbeddingCache.from_npz(cache_path)
            pair_ids = cache.pair_id.astype(object)
            pair_to_fake: Dict[str, np.ndarray] = {}
            pair_to_real: Dict[str, np.ndarray] = {}
            for idx, pair_id in enumerate(pair_ids.tolist()):
                if not pair_id:
                    continue
                if int(cache.label[idx]) == 1:
                    pair_to_fake[str(pair_id)] = cache.cls[idx].astype(np.float32, copy=False)
                else:
                    pair_to_real[str(pair_id)] = cache.cls[idx].astype(np.float32, copy=False)
            shared = sorted(set(pair_to_fake) & set(pair_to_real))
            for pair_id in shared:
                fake_vecs.append(pair_to_fake[pair_id])
                real_vecs.append(pair_to_real[pair_id])
                labels.append(PAIR_TO_IDX[group])
    return np.stack(fake_vecs, axis=0).astype(np.float32), np.stack(real_vecs, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)


def offset_separation_loss(fake_offsets: torch.Tensor, margin: float) -> torch.Tensor:
    penalties = []
    for idx in range(fake_offsets.shape[0]):
        penalties.append(F.relu(margin - fake_offsets[idx].norm(p=2)) ** 2)
        for jdx in range(idx + 1, fake_offsets.shape[0]):
            penalties.append(F.relu(margin - (fake_offsets[idx] - fake_offsets[jdx]).norm(p=2)) ** 2)
    return torch.mean(torch.stack(penalties)) if penalties else torch.tensor(0.0, device=fake_offsets.device)


def ortho_loss(weight: torch.Tensor) -> torch.Tensor:
    dim = weight.shape[0]
    eye = torch.eye(dim, device=weight.device, dtype=weight.dtype)
    return torch.sum((weight.t() @ weight - eye) ** 2)


def pair_logits(delta: torch.Tensor, prototypes: torch.Tensor, temperature: float) -> torch.Tensor:
    delta_sq = torch.sum(delta * delta, dim=1, keepdim=True)
    proto_sq = torch.sum(prototypes * prototypes, dim=1)[None, :]
    dist_sq = delta_sq + proto_sq - 2.0 * (delta @ prototypes.t())
    return -dist_sq / temperature


def margin_pair_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    true = logits[torch.arange(len(labels), device=logits.device), labels]
    max_other = logits.masked_fill(F.one_hot(labels, num_classes=logits.shape[1]).bool(), float("-inf")).max(dim=1).values
    return torch.mean(F.relu(margin + max_other - true))


def hard_negative_margin_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    losses = []
    efs_idx = GROUP_TO_IDX[EFS_LABEL]
    fs_idx = GROUP_TO_IDX["FS"]
    fr_idx = GROUP_TO_IDX.get("FR")
    fe_idx = GROUP_TO_IDX["FE"]
    real_idx = GROUP_TO_IDX[REAL_LABEL]

    efs_mask = labels == efs_idx
    if torch.any(efs_mask):
        true = logits[efs_mask, efs_idx]
        rival = torch.maximum(logits[efs_mask, fe_idx], logits[efs_mask, real_idx])
        losses.append(F.relu(margin + rival - true))

    if fr_idx is not None:
        fr_mask = labels == fr_idx
        if torch.any(fr_mask):
            true = logits[fr_mask, fr_idx]
            rival = torch.maximum(logits[fr_mask, fs_idx], logits[fr_mask, real_idx])
            losses.append(F.relu(margin + rival - true))

    fs_mask = labels == fs_idx
    if fr_idx is not None and torch.any(fs_mask):
        true = logits[fs_mask, fs_idx]
        rival = logits[fs_mask, fr_idx]
        losses.append(F.relu(margin + rival - true))

    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.mean(torch.cat(losses, dim=0))


def real_margin_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    real_idx = GROUP_TO_IDX[REAL_LABEL]
    real_mask = labels == real_idx
    if not torch.any(real_mask):
        return torch.tensor(0.0, device=logits.device)

    true = logits[real_mask, real_idx]
    fake_logits = logits[real_mask][:, [GROUP_TO_IDX[group] for group in ALL_GROUPS if group != REAL_LABEL]]
    rival = fake_logits.max(dim=1).values
    return torch.mean(F.relu(margin + rival - true))


def true_score_floor_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_default: float,
    target_hard: float,
    hard_weight: float,
) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)
    true_prob = prob[torch.arange(len(labels), device=logits.device), labels]
    hard_mask = labels == GROUP_TO_IDX[EFS_LABEL]
    fr_idx = GROUP_TO_IDX.get("FR")
    if fr_idx is not None:
        hard_mask = hard_mask | (labels == fr_idx)
    target = torch.full_like(true_prob, target_default)
    target = torch.where(hard_mask, torch.full_like(true_prob, target_hard), target)
    weight = torch.ones_like(true_prob)
    weight = torch.where(hard_mask, torch.full_like(true_prob, hard_weight), weight)
    return torch.mean(weight * torch.relu(target - true_prob) ** 2)


def summarize_predictions(y_true: np.ndarray, prob: np.ndarray, labels: Sequence[str]) -> dict:
    order = np.argsort(-prob, axis=1)
    pred1 = np.asarray([labels[idx] for idx in order[:, 0]], dtype=object)
    pred2 = np.asarray([labels[idx] for idx in order[:, 1]], dtype=object)
    pred3 = np.asarray([labels[idx] for idx in order[:, 2]], dtype=object)

    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for truth, pred in zip(y_true.tolist(), pred1.tolist(), strict=True):
        cm[label_to_idx[truth], label_to_idx[pred]] += 1

    top1_prob = prob[np.arange(len(prob)), order[:, 0]]
    top2_prob = prob[np.arange(len(prob)), order[:, 1]]
    true_prob = prob[np.arange(len(prob)), np.asarray([label_to_idx[y] for y in y_true.tolist()], dtype=np.int64)]

    per_class = {}
    for row_idx, label in enumerate(labels):
        row = cm[row_idx]
        total = int(row.sum())
        ranking = sorted(
            ((labels[col_idx], int(row[col_idx])) for col_idx in range(len(labels))),
            key=lambda item: (-item[1], item[0]),
        )
        mask = y_true == label
        per_class[label] = {
            "count": total,
            "top1": 0.0 if total == 0 else float(row[row_idx] / total),
            "top2": 0.0 if total == 0 else float(np.mean((pred1[mask] == label) | (pred2[mask] == label))),
            "top3": 0.0 if total == 0 else float(np.mean((pred1[mask] == label) | (pred2[mask] == label) | (pred3[mask] == label))),
            "true_prob_mean": 0.0 if total == 0 else float(np.mean(true_prob[mask])),
            "gap_top1_top2_mean": 0.0 if total == 0 else float(np.mean(top1_prob[mask] - top2_prob[mask])),
            "main_confusions": [
                {
                    "pred": pred_label,
                    "count": count,
                    "rate": 0.0 if total == 0 else float(count / total),
                }
                for pred_label, count in ranking[:3]
            ],
        }

    real_mask = y_true == "REAL"
    fake_mask = ~real_mask
    correct_mask = pred1 == y_true
    wrong_mask = ~correct_mask
    return {
        "overall_top1": float(np.mean(correct_mask)),
        "overall_top2": float(np.mean((pred1 == y_true) | (pred2 == y_true))),
        "overall_top3": float(np.mean((pred1 == y_true) | (pred2 == y_true) | (pred3 == y_true))),
        "macro_top1": float(np.mean([per_class[label]["top1"] for label in labels])),
        "macro_top2": float(np.mean([per_class[label]["top2"] for label in labels])),
        "macro_top3": float(np.mean([per_class[label]["top3"] for label in labels])),
        "fake_top1_true_group": float(np.mean(pred1[fake_mask] == y_true[fake_mask])) if np.any(fake_mask) else 0.0,
        "fake_top2_true_group": float(np.mean((pred1[fake_mask] == y_true[fake_mask]) | (pred2[fake_mask] == y_true[fake_mask]))) if np.any(fake_mask) else 0.0,
        "fake_top3_true_group": float(np.mean((pred1[fake_mask] == y_true[fake_mask]) | (pred2[fake_mask] == y_true[fake_mask]) | (pred3[fake_mask] == y_true[fake_mask]))) if np.any(fake_mask) else 0.0,
        "fake_binary_acc": float(np.mean(pred1[fake_mask] != "REAL")) if np.any(fake_mask) else 0.0,
        "real_top1": float(np.mean(pred1[real_mask] == "REAL")) if np.any(real_mask) else 0.0,
        "real_top2": float(np.mean((pred1[real_mask] == "REAL") | (pred2[real_mask] == "REAL"))) if np.any(real_mask) else 0.0,
        "real_top3": float(np.mean((pred1[real_mask] == "REAL") | (pred2[real_mask] == "REAL") | (pred3[real_mask] == "REAL"))) if np.any(real_mask) else 0.0,
        "mean_gap_top1_top2": float(np.mean(top1_prob - top2_prob)),
        "mean_gap_top1_top2_correct": float(np.mean((top1_prob - top2_prob)[correct_mask])) if np.any(correct_mask) else 0.0,
        "mean_gap_top1_top2_wrong": float(np.mean((top1_prob - top2_prob)[wrong_mask])) if np.any(wrong_mask) else 0.0,
        "true_prob_mean": float(np.mean(true_prob)),
        "per_class": per_class,
    }


def summarize_by_method(y_true: np.ndarray, prob: np.ndarray, methods: np.ndarray, labels: Sequence[str]) -> List[dict]:
    out = []
    for method in sorted(np.unique(methods).tolist()):
        mask = methods == method
        metrics = summarize_predictions(y_true[mask], prob[mask], labels)
        out.append(
            {
                "method": method,
                "count": int(np.sum(mask)),
                "overall_top1": metrics["overall_top1"],
                "overall_top2": metrics["overall_top2"],
                "overall_top3": metrics["overall_top3"],
                "fake_top1_true_group": metrics["fake_top1_true_group"],
                "fake_top2_true_group": metrics["fake_top2_true_group"],
                "fake_top3_true_group": metrics["fake_top3_true_group"],
                "fake_binary_acc": metrics["fake_binary_acc"],
                "real_top1": metrics["real_top1"],
                "mean_gap_top1_top2": metrics["mean_gap_top1_top2"],
                "true_prob_mean": metrics["true_prob_mean"],
                "per_class_top1": {label: metrics["per_class"][label]["top1"] for label in labels},
            }
        )
    return out


def evaluate(model: HybridManifoldModel, x: np.ndarray, batch_size: int, device: torch.device, temperature: float, alpha: float) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            _, _, _, fused = model.fused_logits(xb, temperature, alpha)
            out.append(torch.softmax(fused, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hybrid CLIP linear+manifold 5-way model with routing-aware evaluation.")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache" / "cls")
    parser.add_argument("--train-split", default="DF40_train")
    parser.add_argument("--test-split", default="DF40_test_ff")
    parser.add_argument("--ood-split", default="DF40_test_ood")
    parser.add_argument("--real-rank", type=int, default=40)
    parser.add_argument("--efs-rank", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--linear-warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda-real", type=float, default=1.0)
    parser.add_argument("--lambda-efs", type=float, default=1.0)
    parser.add_argument("--lambda-global", type=float, default=1.0)
    parser.add_argument("--lambda-pair", type=float, default=0.75)
    parser.add_argument("--lambda-sep", type=float, default=0.1)
    parser.add_argument("--lambda-ortho", type=float, default=0.01)
    parser.add_argument("--lambda-margin", type=float, default=0.2)
    parser.add_argument("--lambda-linear-aux", type=float, default=0.5)
    parser.add_argument("--lambda-manifold-aux", type=float, default=0.25)
    parser.add_argument("--lambda-hardneg", type=float, default=0.5)
    parser.add_argument("--lambda-real-hardneg", type=float, default=0.0)
    parser.add_argument("--lambda-score", type=float, default=0.0)
    parser.add_argument("--real-class-multiplier", type=float, default=1.0)
    parser.add_argument("--offset-margin", type=float, default=0.5)
    parser.add_argument("--pair-margin", type=float, default=0.5)
    parser.add_argument("--hardneg-margin", type=float, default=0.75)
    parser.add_argument("--real-hardneg-margin", type=float, default=0.75)
    parser.add_argument("--score-target-default", type=float, default=0.55)
    parser.add_argument("--score-target-hard", type=float, default=0.35)
    parser.add_argument("--score-hard-weight", type=float, default=2.0)
    parser.add_argument("--selection-real-weight", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-fr",
        action="store_true",
        help="Train the upstream model without FR, using labels EFS/FS/FE/REAL.",
    )
    parser.add_argument(
        "--legacy-compat",
        action="store_true",
        help="Approximate the older plain hybrid-manifold baseline by disabling later training additions and writing the older JSON layout.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "outputs" / "clip_hybrid_manifold_5way" / "summary.json",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help="Optional path to save the best hybrid-manifold checkpoint for downstream reuse.",
    )
    return parser.parse_args()


def apply_legacy_compat(args: argparse.Namespace) -> None:
    """Approximate the older plain baseline that produced the compact summary layout."""
    args.linear_warmup_epochs = 0
    args.lambda_hardneg = 0.0
    args.lambda_real_hardneg = 0.0
    args.lambda_score = 0.0
    args.selection_real_weight = 0.0
    # The original baseline ran the full schedule instead of stopping early.
    args.patience = args.epochs + 1


def build_output_settings(args: argparse.Namespace) -> dict:
    settings = {
        "cache_root": str(args.cache_root),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "ood_split": args.ood_split,
        "real_rank": args.real_rank,
        "efs_rank": args.efs_rank,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "lambda_linear_aux": args.lambda_linear_aux,
        "lambda_manifold_aux": args.lambda_manifold_aux,
        "seed": args.seed,
        "output_json": str(args.output_json),
        "labels": list(ALL_GROUPS),
        "pair_groups": list(PAIR_GROUPS),
        "no_fr": bool(args.no_fr),
    }
    if not args.legacy_compat:
        settings.update(
            {
                "linear_warmup_epochs": args.linear_warmup_epochs,
                "lambda_hardneg": args.lambda_hardneg,
                "lambda_real_hardneg": args.lambda_real_hardneg,
                "lambda_score": args.lambda_score,
                "real_class_multiplier": args.real_class_multiplier,
                "hardneg_margin": args.hardneg_margin,
                "real_hardneg_margin": args.real_hardneg_margin,
                "score_target_default": args.score_target_default,
                "score_target_hard": args.score_target_hard,
                "score_hard_weight": args.score_hard_weight,
                "selection_real_weight": args.selection_real_weight,
            }
        )
    settings["output_checkpoint"] = str(resolve_output_checkpoint(args))
    return settings


def resolve_output_checkpoint(args: argparse.Namespace) -> Path:
    if args.output_checkpoint is not None:
        return args.output_checkpoint
    return args.output_json.parent / "checkpoint_best_hybrid_manifold.pt"


def main() -> None:
    args = parse_args()
    if args.legacy_compat:
        apply_legacy_compat(args)
    configure_group_scheme(args.no_fr)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_torch_device(args.device)

    train_x_raw, train_y_names, train_methods = load_split(args.cache_root, args.train_split)
    test_x_raw, test_y_names, test_methods = load_split(args.cache_root, args.test_split)
    ood_x_raw, ood_y_names, ood_methods = load_split(args.cache_root, args.ood_split)
    pair_fake_raw, pair_real_raw, pair_y = load_pair_train(args.cache_root, args.train_split)

    mean, std = standardize_fit(train_x_raw)
    train_x = ((train_x_raw - mean) / std).astype(np.float32)
    test_x = ((test_x_raw - mean) / std).astype(np.float32)
    ood_x = ((ood_x_raw - mean) / std).astype(np.float32)
    pair_fake = ((pair_fake_raw - mean) / std).astype(np.float32)
    pair_real = ((pair_real_raw - mean) / std).astype(np.float32)

    train_y = np.asarray([GROUP_TO_IDX[label] for label in train_y_names.tolist()], dtype=np.int64)
    test_y = np.asarray([GROUP_TO_IDX[label] for label in test_y_names.tolist()], dtype=np.int64)
    ood_y = np.asarray([GROUP_TO_IDX[label] for label in ood_y_names.tolist()], dtype=np.int64)
    fit_idx, val_idx = stratified_split(train_y, args.val_ratio, args.seed)

    train_counts = Counter(train_y_names.tolist())
    class_weight = np.asarray(
        [sum(train_counts.values()) / (len(ALL_GROUPS) * train_counts[group]) for group in ALL_GROUPS],
        dtype=np.float32,
    )
    class_weight[GROUP_TO_IDX[REAL_LABEL]] *= args.real_class_multiplier
    class_weight_t = torch.from_numpy(class_weight).to(device)

    fit_x = train_x[fit_idx]
    fit_y = train_y[fit_idx]
    val_x = train_x[val_idx]
    val_y_names = train_y_names[val_idx]

    real_fit = fit_x[fit_y == GROUP_TO_IDX[REAL_LABEL]]
    efs_fit = fit_x[fit_y == GROUP_TO_IDX[EFS_LABEL]]
    real_center_init = robust_center(real_fit)
    efs_center_init = robust_center(efs_fit)
    real_basis_init = pca_basis(real_fit, args.real_rank)
    efs_basis_init = pca_basis(efs_fit, args.efs_rank)
    fake_offset_init = np.stack(
        [robust_center(fit_x[fit_y == GROUP_TO_IDX[group]]) - real_center_init for group in PAIR_GROUPS],
        axis=0,
    ).astype(np.float32)
    delta_proto_init = np.stack(
        [
            (pair_fake[pair_y == PAIR_TO_IDX[group]] - pair_real[pair_y == PAIR_TO_IDX[group]]).mean(axis=0)
            for group in PAIR_GROUPS
        ],
        axis=0,
    ).astype(np.float32)

    model = HybridManifoldModel(
        dim=train_x.shape[1],
        real_rank=args.real_rank,
        efs_rank=args.efs_rank,
        real_center_init=real_center_init,
        efs_center_init=efs_center_init,
        real_basis_init=real_basis_init,
        efs_basis_init=efs_basis_init,
        fake_offset_init=fake_offset_init,
        delta_proto_init=delta_proto_init,
        labels=ALL_GROUPS,
        pair_groups=PAIR_GROUPS,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    real_loader = DataLoader(TensorDataset(torch.from_numpy(real_fit)), batch_size=args.batch_size, shuffle=True)
    efs_loader = DataLoader(TensorDataset(torch.from_numpy(efs_fit)), batch_size=args.batch_size, shuffle=True)
    global_loader = DataLoader(TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(fit_y)), batch_size=args.batch_size, shuffle=True)
    pair_loader = DataLoader(TensorDataset(torch.from_numpy(pair_fake), torch.from_numpy(pair_real), torch.from_numpy(pair_y)), batch_size=args.batch_size, shuffle=True)

    warmup_history = []
    if args.linear_warmup_epochs > 0:
        warmup_opt = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
        warmup_loader = DataLoader(
            TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(fit_y)),
            batch_size=args.batch_size,
            shuffle=True,
        )
        for epoch in range(1, args.linear_warmup_epochs + 1):
            model.train()
            loss_buf = []
            for xb_global, yb_global in warmup_loader:
                xb_global = xb_global.to(device)
                yb_global = yb_global.to(device)
                warmup_opt.zero_grad(set_to_none=True)
                logits_linear = model.classifier(xb_global)
                loss = F.cross_entropy(logits_linear, yb_global, weight=class_weight_t)
                loss.backward()
                warmup_opt.step()
                loss_buf.append(float(loss.item()))
            val_prob = evaluate(model, val_x, args.eval_batch_size, device, args.temperature, args.alpha)
            warmup_history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(loss_buf)),
                    "val": summarize_predictions(val_y_names, val_prob, ALL_GROUPS),
                }
            )

    best_state = None
    best_metric = -1.0
    history = []
    epochs_without_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        real_iter = iter(real_loader)
        efs_iter = iter(efs_loader)
        global_iter = iter(global_loader)
        pair_iter = iter(pair_loader)
        loss_buf = []
        for _ in range(max(len(real_loader), len(efs_loader), len(global_loader), len(pair_loader))):
            try:
                (xb_real,) = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                (xb_real,) = next(real_iter)
            try:
                (xb_efs,) = next(efs_iter)
            except StopIteration:
                efs_iter = iter(efs_loader)
                (xb_efs,) = next(efs_iter)
            try:
                xb_global, yb_global = next(global_iter)
            except StopIteration:
                global_iter = iter(global_loader)
                xb_global, yb_global = next(global_iter)
            try:
                xb_fake, xb_pair_real, yb_pair = next(pair_iter)
            except StopIteration:
                pair_iter = iter(pair_loader)
                xb_fake, xb_pair_real, yb_pair = next(pair_iter)

            xb_real = xb_real.to(device)
            xb_efs = xb_efs.to(device)
            xb_global = xb_global.to(device)
            yb_global = yb_global.to(device)
            xb_fake = xb_fake.to(device)
            xb_pair_real = xb_pair_real.to(device)
            yb_pair = yb_pair.to(device)

            opt.zero_grad(set_to_none=True)

            real_basis = model.real_basis()
            efs_basis = model.efs_basis()

            y_real = model.transform(xb_real)
            real_proj = model.project(y_real, model.real_center, real_basis)
            loss_real = torch.mean(torch.sum((y_real - real_proj) ** 2, dim=1))

            y_efs = model.transform(xb_efs)
            efs_proj = model.project(y_efs, model.efs_center, efs_basis)
            loss_efs = torch.mean(torch.sum((y_efs - efs_proj) ** 2, dim=1))

            y_global, logits_linear, logits_manifold, logits_fused = model.fused_logits(xb_global, args.temperature, args.alpha)
            loss_global = F.cross_entropy(logits_fused, yb_global, weight=class_weight_t)
            loss_linear = F.cross_entropy(logits_linear, yb_global, weight=class_weight_t)
            loss_manifold = F.cross_entropy(logits_manifold, yb_global, weight=class_weight_t)
            loss_hardneg = hard_negative_margin_loss(logits_fused, yb_global, args.hardneg_margin)
            loss_real_hardneg = real_margin_loss(logits_fused, yb_global, args.real_hardneg_margin)
            loss_score = true_score_floor_loss(
                logits_fused,
                yb_global,
                target_default=args.score_target_default,
                target_hard=args.score_target_hard,
                hard_weight=args.score_hard_weight,
            )

            y_fake = model.transform(xb_fake)
            y_pair_real_t = model.transform(xb_pair_real)
            delta = y_fake - y_pair_real_t
            logits_pair = pair_logits(delta, model.delta_prototypes, args.temperature)
            loss_pair = F.cross_entropy(logits_pair, yb_pair) + margin_pair_loss(logits_pair, yb_pair, args.pair_margin)

            loss_sep = offset_separation_loss(model.fake_offsets, args.offset_margin) + F.relu(args.offset_margin - torch.norm(model.efs_center - model.real_center, p=2)) ** 2
            loss_ortho = ortho_loss(model.linear.weight)
            loss = (
                args.lambda_real * loss_real
                + args.lambda_efs * loss_efs
                + args.lambda_global * loss_global
                + args.lambda_linear_aux * loss_linear
                + args.lambda_manifold_aux * loss_manifold
                + args.lambda_hardneg * loss_hardneg
                + args.lambda_real_hardneg * loss_real_hardneg
                + args.lambda_score * loss_score
                + args.lambda_pair * loss_pair
                + args.lambda_sep * loss_sep
                + args.lambda_ortho * loss_ortho
            )
            loss.backward()
            opt.step()
            loss_buf.append(float(loss.item()))

        val_prob = evaluate(model, val_x, args.eval_batch_size, device, args.temperature, args.alpha)
        val_metrics = summarize_predictions(val_y_names, val_prob, ALL_GROUPS)
        train_loss_value = float(np.mean(loss_buf))
        history.append({"epoch": epoch, "train_loss": train_loss_value, "val": val_metrics})
        current_metric = (
            val_metrics["macro_top1"]
            + 0.25 * val_metrics["overall_top2"]
            + 0.25 * val_metrics["macro_top3"]
            + args.selection_real_weight * val_metrics["real_top1"]
            - 0.1 * val_metrics["mean_gap_top1_top2_wrong"]
        )
        if current_metric > best_metric:
            best_metric = current_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        if epochs_without_improve >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = resolve_output_checkpoint(args)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "dim": int(train_x.shape[1]),
            "real_rank": int(args.real_rank),
            "efs_rank": int(args.efs_rank),
            "classes": np.asarray(ALL_GROUPS, dtype=object),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "temperature": float(args.temperature),
            "alpha": float(args.alpha),
            "settings": build_output_settings(args),
        },
        checkpoint_path,
    )

    train_prob = evaluate(model, train_x, args.eval_batch_size, device, args.temperature, args.alpha)
    test_prob = evaluate(model, test_x, args.eval_batch_size, device, args.temperature, args.alpha)
    ood_prob = evaluate(model, ood_x, args.eval_batch_size, device, args.temperature, args.alpha)

    output = {
        "settings": build_output_settings(args),
        "train_counts": dict(train_counts),
        "class_weight": {group: float(class_weight[GROUP_TO_IDX[group]]) for group in ALL_GROUPS},
        "history": history,
        "metrics": {
            "train": summarize_predictions(train_y_names, train_prob, ALL_GROUPS),
            "test_ff": summarize_predictions(test_y_names, test_prob, ALL_GROUPS),
            "ood_test": summarize_predictions(ood_y_names, ood_prob, ALL_GROUPS),
        },
        "test_ff_by_method": summarize_by_method(test_y_names, test_prob, test_methods, ALL_GROUPS),
        "ood_by_method": summarize_by_method(ood_y_names, ood_prob, ood_methods, ALL_GROUPS),
    }
    if not args.legacy_compat:
        output["learned_manifold_scale"] = {
            group: float(v)
            for group, v in zip(ALL_GROUPS, F.softplus(model.logit_scale).detach().cpu().tolist(), strict=True)
        }
        output["warmup_history"] = warmup_history
    output["checkpoint_path"] = str(checkpoint_path)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "test_overall_top1": output["metrics"]["test_ff"]["overall_top1"],
                "test_macro_top1": output["metrics"]["test_ff"]["macro_top1"],
                "ood_overall_top1": output["metrics"]["ood_test"]["overall_top1"],
                "ood_macro_top1": output["metrics"]["ood_test"]["macro_top1"],
                "ood_overall_top2": output["metrics"]["ood_test"]["overall_top2"],
                "ood_overall_top3": output["metrics"]["ood_test"]["overall_top3"],
                "ood_per_class_top1": {k: v["top1"] for k, v in output["metrics"]["ood_test"]["per_class"].items()},
            },
            indent=2,
        )
    )
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
