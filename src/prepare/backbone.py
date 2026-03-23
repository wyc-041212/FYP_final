from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from PIL import Image
from prepare.huggingface_compat import import_transformers, resolve_model_dir
from utils.device import resolve_torch_device


AutoImageProcessor, AutoModel, CLIPVisionModel = import_transformers(
    "AutoImageProcessor",
    "AutoModel",
    "CLIPVisionModel",
)


class FrozenBackbone:
    def __init__(self, backbone_name: str, model_dir: str | Path, device: str) -> None:
        self.backbone_name = backbone_name
        self.model_dir = str(resolve_model_dir(model_dir))
        self.device = resolve_torch_device(device)
        self.processor = None
        self.processor_cfg = None
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_dir, use_fast=False)
        except Exception:
            cfg_path = Path(self.model_dir) / "preprocessor_config.json"
            if cfg_path.exists():
                self.processor_cfg = json.loads(cfg_path.read_text())
            elif "clip" in self.backbone_name.lower() or "clip" in self.model_dir.lower():
                # Some local CLIP checkpoints do not ship a HF image processor config.
                # Keep preprocessing explicit so online layer sweeps can reuse one path.
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
            # DINOv3 fast processor does not reliably square-resize all aspect ratios
            # when passed an integer size alone; make the output shape explicit.
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
        return {
            "pixel_values": torch.from_numpy(np.stack(pixel_values, axis=0)).to(self.device)
        }

    @torch.no_grad()
    def encode_cls_batch(
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
    def encode_patch_batch(
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
