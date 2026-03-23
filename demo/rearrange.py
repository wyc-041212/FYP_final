from __future__ import annotations

import json
from pathlib import Path


def build_frame_index(
    frames_root: str | Path,
    output_json: str | Path,
    landmarks_root: str | Path | None = None,
    label: str = "unknown",
) -> Path:
    frames_root = Path(frames_root)
    output_json = Path(output_json)
    landmarks_root = Path(landmarks_root) if landmarks_root is not None else None

    if not frames_root.exists():
        raise FileNotFoundError(f"frames_root does not exist: {frames_root}")

    index: dict[str, dict[str, object]] = {}
    for video_dir in sorted(frames_root.iterdir()):
        if not video_dir.is_dir():
            continue

        frame_paths = sorted(str(path) for path in video_dir.glob("*.png"))
        if not frame_paths:
            continue

        entry: dict[str, object] = {
            "label": label,
            "frames": frame_paths,
        }

        if landmarks_root is not None:
            landmark_dir = landmarks_root / video_dir.name
            if landmark_dir.exists():
                entry["landmarks"] = sorted(str(path) for path in landmark_dir.glob("*.npy"))

        index[video_dir.name] = entry

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(index, indent=2))
    return output_json
