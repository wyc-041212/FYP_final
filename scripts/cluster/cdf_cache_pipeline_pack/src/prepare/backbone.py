from __future__ import annotations

import importlib.machinery
import importlib.metadata as importlib_metadata
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List
import types

import numpy as np
from PIL import Image
import torch

try:
    import importlib_metadata as importlib_metadata_backport
except ImportError:  # pragma: no cover
    importlib_metadata_backport = None

_orig_version = importlib_metadata.version


def _patched_version(package: str) -> str:
    version = _orig_version(package)
    if package == "huggingface-hub":
        if version.startswith("1."):
            return "0.34.0"
        if version.startswith("0."):
            return "1.3.0"
    return version


importlib_metadata.version = _patched_version
if importlib_metadata_backport is not None:
    importlib_metadata_backport.version = _patched_version


def _install_sklearn_stub() -> None:
    sklearn_mod = types.ModuleType("sklearn")
    metrics_mod = types.ModuleType("sklearn.metrics")
    sklearn_mod.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
    metrics_mod.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)

    def _roc_curve(*args, **kwargs):
        raise RuntimeError("sklearn.metrics.roc_curve is unavailable in this environment.")

    metrics_mod.roc_curve = _roc_curve
    sklearn_mod.metrics = metrics_mod
    sys.modules.setdefault("sklearn", sklearn_mod)
    sys.modules.setdefault("sklearn.metrics", metrics_mod)


try:
    import sklearn.metrics  # type: ignore  # noqa: F401
except Exception:
    _install_sklearn_stub()


def _import_transformers():
    try:
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel

        return AutoImageProcessor, AutoModel, CLIPVisionModel
    except ImportError as exc:
        if "huggingface-hub" not in str(exc):
            raise
        importlib_metadata.version = _patched_version
        if importlib_metadata_backport is not None:
            importlib_metadata_backport.version = _patched_version
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel

        return AutoImageProcessor, AutoModel, CLIPVisionModel


AutoImageProcessor, AutoModel, CLIPVisionModel = _import_transformers()


def _resolve_model_dir(model_dir: str | Path) -> str:
    path = Path(model_dir)
    if (path / "config.json").exists():
        return str(path)
    snapshots = path / "snapshots"
    if snapshots.exists():
        snapshot_dirs = sorted(p for p in snapshots.iterdir() if p.is_dir())
        if snapshot_dirs:
            return str(snapshot_dirs[-1])
    return str(path)


class FrozenBackbone:
    def __init__(self, backbone_name: str, model_dir: str | Path, device: str) -> None:
        self.backbone_name = backbone_name
        self.model_dir = _resolve_model_dir(model_dir)
        self.device = torch.device(device)
        self.processor = None
        self.processor_cfg = None
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_dir, use_fast=False)
        except Exception:
            cfg_path = Path(self.model_dir) / "preprocessor_config.json"
            if cfg_path.exists():
                self.processor_cfg = json.loads(cfg_path.read_text())
            elif "clip" in self.backbone_name.lower() or "clip" in self.model_dir.lower():
                self.processor_cfg = {
                    "image_mean": [0.48145466, 0.4578275, 0.40821073],
                    "image_std": [0.26862954, 0.26130258, 0.27577711],
                }
            else:
                raise
        if "clip" in self.backbone_name.lower() or "clip" in self.model_dir.lower():
            self.model = CLIPVisionModel.from_pretrained(self.model_dir)
        else:
            self.model = AutoModel.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _encode_inputs(self, images: List[Image.Image], target_size: int) -> dict[str, torch.Tensor]:
        if self.processor is not None:
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                size={"height": target_size, "width": target_size},
            )
            return {key: value.to(self.device) for key, value in inputs.items()}

        cfg = self.processor_cfg or {}
        mean = np.asarray(cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.asarray(cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
        resample = Image.BICUBIC
        pixel_values = []
        for image in images:
            resized = image.resize((target_size, target_size), resample=resample)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            arr = (arr - mean[None, None, :]) / std[None, None, :]
            arr = np.transpose(arr, (2, 0, 1))
            pixel_values.append(arr)
        return {"pixel_values": torch.from_numpy(np.stack(pixel_values, axis=0)).to(self.device)}

    @torch.no_grad()
    def encode_batch(
        self,
        img_paths: Iterable[str | Path],
        target_size: int,
        layers: List[int] | None = None,
    ) -> Dict[int, np.ndarray]:
        selected_layers = layers or [-1]
        images = [Image.open(path).convert("RGB") for path in img_paths]
        inputs = self._encode_inputs(images, target_size)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        return {layer: hidden_states[layer][:, 0].detach().cpu().numpy() for layer in selected_layers}

    @torch.no_grad()
    def encode_patch_tokens_batch(
        self,
        img_paths: Iterable[str | Path],
        target_size: int,
        layer: int = -1,
    ) -> np.ndarray:
        images = [Image.open(path).convert("RGB") for path in img_paths]
        inputs = self._encode_inputs(images, target_size)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer]
        num_register = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        patch_tokens = hidden[:, 1 + num_register :, :]
        return patch_tokens.detach().cpu().numpy().astype(np.float32, copy=False)
