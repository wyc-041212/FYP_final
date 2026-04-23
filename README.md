# FYP Final

This repository is the current runnable `within` mainline for the FYP deepfake-detection project.

The active path is:

- upstream hybrid manifold routing checkpoint in `checkpoints/upstream/`
- within-split downstream checkpoints in `within_checkpoints/`
- within experiment results in `within_outputs/`
- within experiment runners in `scripts/within_*.py`
- replay / sampling runtime in `main.py`
- demo entrypoints in `demo/`

This repository is no longer the place for:

- report drafting chapters and thesis-writing control files
- old non-within downstream checkpoints and outputs
- bulk generated outputs under the old `outputs/` tree

Those materials were intentionally moved out to desktop-side archive folders so the project itself stays aligned with the final standardized `within` pipeline.

## What Is Kept Here

### Core runtime

- `main.py`
  - shared inference / replay runtime
- `sample/sample.py`
  - thin wrapper around `main.py --mode sample`
- `demo/demo.py`
  - CLI demo pipeline for one-off video evaluation
- `demo/gradio_app.py`
  - Gradio web demo for uploading a video and recording a browser demo

### Active source code

- `src/prepare/`
  - feature extraction, cache I/O, face-region parsing helpers
- `src/train/train_upstream.py`
  - upstream hybrid manifold training
- `src/train/within_train_downstream_head.py`
  - active within downstream training / evaluation logic
- `src/train/train_downstream_head.py`
  - compatibility shim that currently forwards to the within implementation
- `src/eval/`
  - analysis utilities and supporting research scripts

### Active checkpoints

- `checkpoints/upstream/`
  - shared upstream checkpoints required by the runtime
- `within_checkpoints/downstream/`
  - active within patch / pair branches
- `within_checkpoints/heads/`
  - active within route-aware meta heads and metadata

### Active outputs

- `within_outputs/`
  - canonical within results, analysis outputs, and report-facing assets that still belong to the final project

### Validation and tests

- `tests/`
  - unit tests and structural checks

## What Was Moved Out

These are intentionally not part of the main project tree anymore:

- report drafting material
  - moved to desktop-side report bundle
- non-within training artifacts and outputs
  - moved to desktop-side archive
- old root-level `outputs/`
  - archived because the final project now centers on `within_outputs/`

One important exception remains:

- `checkpoints/upstream/` stays in this repo
  - the runtime still depends on those upstream checkpoints
  - some code paths historically referenced them via fixed assumptions, so they must remain present

## Project Layout

```text
FYP_final/
├── checkpoints/upstream/              shared upstream checkpoints kept in-project
├── within_checkpoints/                active within downstream + head checkpoints
├── within_outputs/                    active within experiment results
├── demo/                              CLI demo and Gradio demo
├── sample/                            sampling wrapper
├── scripts/within_*.py                active within experiment runners
├── scripts/cluster/                   cluster rerun / cache-sync / CDF support scripts
├── src/                               source code
├── tests/                             unit tests
├── CODEBASE_MAP.md                    longer navigation map
└── pipeline_manifest.json             retained manifest entrypoint map
```

## Main Entry Points

### Replay

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n fyp python \
  /Users/wuyuchen/Desktop/FYP_final/main.py \
  --mode replay
```

### Sample

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n fyp python \
  /Users/wuyuchen/Desktop/FYP_final/main.py \
  --mode sample \
  --sample-root /Users/wuyuchen/Desktop/tmp_probe_bundle \
  --output-json /Users/wuyuchen/Desktop/tmp_probe_result.json
```

### CLI demo

```bash
conda run --no-capture-output -n fyp python \
  /Users/wuyuchen/Desktop/FYP_final/demo/demo.py \
  --help
```

### Gradio demo

```bash
conda run --no-capture-output -n fyp python \
  /Users/wuyuchen/Desktop/FYP_final/demo/gradio_app.py
```

Open the local URL shown in the terminal, upload a video, and run the demo in the browser.

## Current Checkpoint Convention

### Shared upstream

- full upstream:
  - `checkpoints/upstream/checkpoint_best_hybrid_manifold.pt`
- no-FR upstream:
  - `checkpoints/upstream/checkpoint_no_fr.pt`

### Active within downstream

- full within patch:
  - `within_checkpoints/downstream/patch_branch.joblib`
- full within pair:
  - `within_checkpoints/downstream/pair_branch.joblib`
- no-FR within patch:
  - `within_checkpoints/downstream/patch_branch_no_fr.joblib`
- no-FR within pair:
  - `within_checkpoints/downstream/pair_branch_no_fr.joblib`

### Active within heads

- full within head:
  - `within_checkpoints/heads/route_meta_head.joblib`
- full within head metadata:
  - `within_checkpoints/heads/route_meta_head_meta.json`
- no-FR within head:
  - `within_checkpoints/heads/route_meta_head_no_fr.joblib`
- no-FR within head metadata:
  - `within_checkpoints/heads/route_meta_head_no_fr_meta.json`

Important current default:

- the within pair-region mode used by the active heads is `no_background_keep_hair`

## Experiment Scripts

The active experiment runners are the `within_*.py` scripts under `scripts/`.

Most important ones:

- `scripts/within_run_threshold_sweep.py`
- `scripts/within_run_cdf_threshold_sweep.py`
- `scripts/within_run_cdf_replay_eval.py`
- `scripts/within_run_fixed_fpr_metrics.py`
- `scripts/within_run_main_ablation_line.py`
- `scripts/within_run_main_ablation_cdf_only.py`
- `scripts/within_generate_report_assets.py`

The cluster-side support scripts live under `scripts/cluster/`.

Use those for:

- CDF cache syncing between nodes
- no-FR CDF reruns with FR groups included
- fixed-threshold `0.8` CDF ablation supplements
- standalone CDF / CelebDF-real cache preparation helpers

See `/Users/wuyuchen/Desktop/FYP_final/scripts/cluster/README.md` for details.

## Results That Matter

The canonical final within results live in `within_outputs/`.

Common files to read first:

- `within_outputs/full_threshold_sweep.csv`
- `within_outputs/no_fr_threshold_sweep.csv`
- `within_outputs/full_main_ablation_line.csv`
- `within_outputs/no_fr_main_ablation_line.csv`
- `within_outputs/full_cdf_replay_eval.csv`
- `within_outputs/no_fr_cdf_replay_eval.csv`

Important historical support files that were restored from the cluster archive:

- `within_outputs/no_fr_main_ablation_line.before_cdf_merge.*`
- `within_outputs/no_fr_main_ablation_line.before_fixed_0p8_merge.*`
- `within_outputs/no_fr_main_ablation_line_cdf_only.*`
- `within_outputs/within_no_fr_main_ablation_with_fr_cdf_cpu_gpu10.*`
- `within_outputs/within_no_fr_with_fr_cpu_gpu10_cdf_replay_eval.*`
- `within_outputs/within_no_fr_with_fr_cpu_gpu10_threshold_sweep_cdf.*`

These are useful when tracing exactly which no-FR CDF results were merged back into the final ablation table.

## Environment

Primary environment files:

- `environment.yml`
- `requirements.txt`
- `current.txt`

Typical usage assumes the `fyp` conda environment:

```bash
conda run --no-capture-output -n fyp python --version
```

## Notes

- `within_outputs/` is gitignored because it contains generated outputs and large experiment artifacts.
- `.cache/` and `__pycache__/` are runtime byproducts, not source assets.
- `docs/` currently contains planning notes and superpowers-generated plans; it is not the canonical project documentation entrypoint.
- `CODEBASE_MAP.md` is the more detailed navigation file if you need cluster-path context or experiment provenance.
