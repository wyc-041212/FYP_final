# Route Meta Fusion Pipeline

This folder is the current runnable project snapshot for the selected route-aware pipeline.

It includes:
- the upstream checkpoint
- the downstream checkpoints
- the selected `route_meta_fusion` head checkpoint
- the code needed for upstream training, downstream/head training, replay, and sample inference

## Contents

- `checkpoints/`
  - `upstream/`
  - `downstream/`
  - `heads/`
- `cache/`
  - `cls/`
  - `patch/`
  - `compact/`
  - `sampled/`
- `src/train/`
  - `train_upstream.py`
  - `train_downstream_head.py`
- `src/prepare/`
  - feature extraction and model-loading utilities
- `src/eval/`
  - legacy multi-head research code kept for reference
- `main.py`
  - unified entrypoint with `--mode replay` and `--mode sample`

## What This Folder Can Do

1. Train the upstream model from CLS cache.
2. Train the downstream patch + pair + selected head pipeline.
3. Replay `validation`, `test_ff`, and `OOD` using saved checkpoints.
4. Run sample inference on sampled folders or directly on raw folders.

## Replay Command

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n fyp python \
  /Users/wuyuchen/Desktop/FYP_final/main.py \
  --mode replay
```

## Sample Command

Sample directly from raw folders:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n fyp python \
  /Users/wuyuchen/Desktop/FYP_final/main.py \
  --mode sample \
  --data-root /Volumes/未命名/DF40 \
  --source-dir /Volumes/未命名/DF40/FR/wav2lip/cdf/frames \
  --num-folders 2 \
  --max-frames-per-folder 6 \
  --output-json /Users/wuyuchen/Desktop/tmp_wav2lip_probe_result.json
```

Sample from an existing sampled bundle:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n fyp python \
  /Users/wuyuchen/Desktop/FYP_final/main.py \
  --mode sample \
  --sample-root /Users/wuyuchen/Desktop/tmp_wav2lip_probe \
  --output-json /Users/wuyuchen/Desktop/tmp_wav2lip_probe_result.json
```

## Important Notes

- Replay depends on the existing CLS and patch cache folders on this machine.
- Raw-folder probe uses the same patch backbone family and target size as the official patch cache pipeline: `clip` at `224`.
- The current public-facing entrypoint is `main.py`.
