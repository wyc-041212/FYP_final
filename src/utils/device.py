from __future__ import annotations

import torch


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def resolve_device_name(device_name: str | None = None, *, allow_cpu: bool = True) -> str:
    requested = (device_name or "auto").strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        if allow_cpu:
            return "cpu"
        raise RuntimeError("No CUDA or MPS device is available, and CPU fallback is disabled.")

    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{device_name}', but CUDA is not available.")
        return requested

    if requested == "mps":
        if not _mps_available():
            raise RuntimeError("Requested device 'mps', but MPS is not available.")
        return requested

    if requested == "cpu":
        if not allow_cpu:
            raise RuntimeError("Requested device 'cpu', but CPU fallback is disabled.")
        return requested

    return str(torch.device(requested))


def resolve_torch_device(device_name: str | None = None, *, allow_cpu: bool = True) -> torch.device:
    return torch.device(resolve_device_name(device_name, allow_cpu=allow_cpu))
