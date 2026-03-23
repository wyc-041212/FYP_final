from __future__ import annotations

import importlib.machinery
import importlib.metadata as importlib_metadata
import sys
import types
from pathlib import Path

from PIL import Image

try:
    import importlib_metadata as importlib_metadata_backport
except ImportError:  # pragma: no cover
    importlib_metadata_backport = None

_ORIG_VERSION = importlib_metadata.version


def ensure_pillow_resampling() -> None:
    if hasattr(Image, "Resampling"):
        return

    class _Resampling:
        NEAREST = Image.NEAREST
        BOX = getattr(Image, "BOX", Image.NEAREST)
        BILINEAR = Image.BILINEAR
        HAMMING = getattr(Image, "HAMMING", Image.BILINEAR)
        BICUBIC = Image.BICUBIC
        LANCZOS = Image.LANCZOS

    Image.Resampling = _Resampling


def patched_version(package: str) -> str:
    version = _ORIG_VERSION(package)
    if package == "huggingface-hub":
        if version.startswith("1."):
            return "0.34.0"
        if version.startswith("0."):
            return "1.3.0"
    return version


def apply_hf_version_patch() -> None:
    importlib_metadata.version = patched_version
    if importlib_metadata_backport is not None:
        importlib_metadata_backport.version = patched_version


def ensure_sklearn_metrics_stub() -> None:
    try:
        import sklearn.metrics  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

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


def import_transformers(*names: str):
    apply_hf_version_patch()
    ensure_sklearn_metrics_stub()
    try:
        module = __import__("transformers", fromlist=list(names))
    except ImportError as exc:
        if "huggingface-hub" not in str(exc):
            raise
        apply_hf_version_patch()
        module = __import__("transformers", fromlist=list(names))
    return tuple(getattr(module, name) for name in names)


def resolve_model_dir(model_dir: str | Path, *, require_snapshot: bool = False) -> Path:
    path = Path(model_dir)
    if (path / "config.json").exists():
        return path

    snapshots = path / "snapshots"
    if snapshots.exists():
        snapshot_dirs = sorted(p for p in snapshots.iterdir() if p.is_dir())
        if snapshot_dirs:
            return snapshot_dirs[-1] if require_snapshot else snapshot_dirs[-1]

    if require_snapshot:
        raise RuntimeError(f"No model snapshot found under {path}")
    return path
