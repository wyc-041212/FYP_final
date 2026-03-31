# Route Meta Fusion Pipeline

This repo is the cleaned runnable snapshot of the current `patch + pair + route_meta_fusion` pipeline.

## What Matters

- `main.py`
  - unified inference entrypoint
  - supports `--mode replay` and `--mode sample`
- `src/train/train_upstream.py`
  - upstream training
  - now supports `--no-fr`
- `src/train/train_downstream_head.py`
  - downstream patch/pair/head training
  - now supports `--no-fr`
  - `pair-region-mode` supports `no_background_keep_hair`
- `sample/sample.py`
  - thin wrapper around `main.py --mode sample`
  - still referenced by `pipeline_manifest.json`, so it is kept

## Checkpoints

The repo now keeps the actual runnable checkpoints under `checkpoints/`:

- `checkpoints/upstream/`
  - `checkpoint_best_hybrid_manifold.pt`
  - `checkpoint_no_fr.pt`
  - `summary.json`
- `checkpoints/downstream/`
  - `patch_branch.joblib`
  - `pair_branch.joblib`
  - `patch_branch_no_fr.joblib`
  - `pair_branch_no_fr.joblib`
- `checkpoints/heads/`
  - `route_meta_head.joblib`
  - `route_meta_head_meta.json`
  - `route_meta_head_no_fr.joblib`
  - `route_meta_head_no_fr_meta.json`

Notes:

- `checkpoints/upstream/summary.json` is informational only. Runtime does not need it.
- `checkpoints/heads/route_meta_head_meta.json` is still used by `main.py --mode replay` to recover replay config, so keep it for now.

## Training Modes

### Full pipeline

Upstream:

```bash
python /Users/wuyuchen/Desktop/FYP_final/src/train/train_upstream.py
```

Downstream:

```bash
python /Users/wuyuchen/Desktop/FYP_final/src/train/train_downstream_head.py
```

### No-FR pipeline

Upstream:

```bash
python /Users/wuyuchen/Desktop/FYP_final/src/train/train_upstream.py --no-fr
```

Downstream:

```bash
python /Users/wuyuchen/Desktop/FYP_final/src/train/train_downstream_head.py \
  --no-fr \
  --hybrid-checkpoint /Users/wuyuchen/Desktop/FYP_final/checkpoints/upstream/checkpoint_no_fr.pt
```

## Inference

Replay:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n fyp python \
  /Users/wuyuchen/Desktop/FYP_final/main.py \
  --mode replay
```

Sample:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n fyp python \
  /Users/wuyuchen/Desktop/FYP_final/main.py \
  --mode sample \
  --sample-root /Users/wuyuchen/Desktop/tmp_probe_bundle \
  --output-json /Users/wuyuchen/Desktop/tmp_probe_result.json
```

## Folder Status

- `src/train/`
  - active code
- `src/prepare/`
  - active code
- `src/eval/`
  - legacy / research utilities
  - not required by the main runtime path
- `sample/`
  - kept because the manifest and wrapper entrypoint still use it

## Current Direction

- downstream remains `patch + pair`
- `pair` can now be trained with `no_background_keep_hair`
- `no FR` is now a first-class training switch instead of only living in temporary scripts
