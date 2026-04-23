# FYP Codebase Map

This file is the detailed navigation map for `/Users/wuyuchen/Desktop/FYP_final`.

Use it when you need to answer:

- which code path is actually active now
- which results are canonical versus transitional
- which files were restored from the cluster archive
- which materials were intentionally moved out of the main repository

## Current Rule Of Thumb

If a file path starts with:

- `within_checkpoints/`
  - active within checkpoint material
- `within_outputs/`
  - active or preserved within results
- `scripts/within_`
  - active experiment runners
- `checkpoints/upstream/`
  - required shared upstream checkpoints

then it is part of the final standardized mainline.

If something refers to:

- old root `outputs/`
- old non-within downstream checkpoints
- report drafting files

then it is no longer part of the main repository state unless explicitly restored for provenance.

## Repository Zones

### Core runtime

- `main.py`
  - shared replay / sample runtime helpers
- `sample/sample.py`
  - simple sampling wrapper used by the manifest
- `demo/demo.py`
  - CLI demo
- `demo/gradio_app.py`
  - browser demo

### Source code

- `src/prepare/`
  - cache loading, CLIP backbone helpers, face-region parsing
- `src/train/train_upstream.py`
  - upstream hybrid manifold training
- `src/train/within_train_downstream_head.py`
  - active within downstream logic
- `src/train/train_downstream_head.py`
  - compatibility shim kept because some imports still expect this module name
- `src/eval/`
  - analysis / research utilities, not the central runtime path

### Active experiment runners

- `scripts/within_run_threshold_sweep.py`
- `scripts/within_run_cdf_threshold_sweep.py`
- `scripts/within_run_cdf_replay_eval.py`
- `scripts/within_run_fixed_fpr_metrics.py`
- `scripts/within_run_main_ablation_line.py`
- `scripts/within_run_main_ablation_cdf_only.py`
- `scripts/within_run_small_ood_sweep.py`
- `scripts/within_generate_report_assets.py`

### Cluster support

- `scripts/cluster/`
  - node-copy helpers, slurm launchers, CDF support runners
- `scripts/cluster/cdf_cache_pipeline_pack/`
  - standalone helper pack for building CDF fake-only caches and CelebDF-real caches

### Validation

- `tests/`
  - unit and structure tests

## Checkpoint Layout

### Shared upstream checkpoints

These remain under `checkpoints/upstream/` because the runtime still depends on them:

- `checkpoints/upstream/checkpoint_best_hybrid_manifold.pt`
- `checkpoints/upstream/checkpoint_no_fr.pt`
- `checkpoints/upstream/summary.json`

### Within checkpoints

The active downstream and head checkpoints live under `within_checkpoints/`.

Common files:

- `within_checkpoints/downstream/patch_branch.joblib`
- `within_checkpoints/downstream/pair_branch.joblib`
- `within_checkpoints/downstream/patch_branch_no_fr.joblib`
- `within_checkpoints/downstream/pair_branch_no_fr.joblib`
- `within_checkpoints/heads/route_meta_head.joblib`
- `within_checkpoints/heads/route_meta_head_meta.json`
- `within_checkpoints/heads/route_meta_head_no_fr.joblib`
- `within_checkpoints/heads/route_meta_head_no_fr_meta.json`

Important configuration detail:

- the current active within heads use pair-region mode `no_background_keep_hair`

There are also preserved pair-region comparison variants such as:

- `pair_branch_full_all_regions.joblib`
- `pair_branch_full_canonical.joblib`
- `pair_branch_no_fr_all_regions.joblib`
- `pair_branch_no_fr_canonical.joblib`

Those are ablation support checkpoints, not the main default.

## Output Layout

### Canonical outputs

`within_outputs/` is the main results tree that still belongs to the project.

Important result families:

- threshold sweeps
  - `full_threshold_sweep.*`
  - `no_fr_threshold_sweep.*`
- fixed-FPR metrics
  - `full_fixed_fpr_metrics.*`
  - `no_fr_fixed_fpr_metrics_*.*`
- main ablation lines
  - `full_main_ablation_line.*`
  - `no_fr_main_ablation_line.*`
- CDF replay and threshold sweeps
  - `full_cdf_replay_eval.*`
  - `no_fr_cdf_replay_eval.*`
  - `full_threshold_sweep_cdf.*`
  - `no_fr_threshold_sweep_cdf.*`
- report-facing outputs
  - `within_outputs/reports/`

### Restored provenance files

These were restored from `/Users/wuyuchen/Desktop/fyp_code_no_ckpt_data_20260422.tar.gz` because they add reproducibility value:

- `within_outputs/no_fr_main_ablation_line.before_cdf_merge.*`
- `within_outputs/no_fr_main_ablation_line.before_fixed_0p8_merge.*`
- `within_outputs/no_fr_main_ablation_line_cdf_only.*`
- `within_outputs/within_no_fr_main_ablation_with_fr_cdf_cpu_gpu10.*`
- `within_outputs/within_no_fr_with_fr_cpu_gpu10_cdf_replay_eval.*`
- `within_outputs/within_no_fr_with_fr_cpu_gpu10_threshold_sweep_cdf.*`

These files matter because they preserve:

- the state before CDF metrics were merged into the no-FR ablation line
- the fixed-threshold `0.8` rerun results from the cluster
- the detailed CDF fake-method breakdown used to support report statements

### Analysis outputs

`within_outputs/analysis/` contains additional probes, diagnostics, and demo-side analysis material.

Treat these as supporting analysis, not as the first place to read final tables from.

## Cluster Runners

The main cluster-side scripts now worth keeping track of are:

- `scripts/cluster/within_run_cdf_cluster_pipeline.slurm.sh`
  - main within CDF replay / threshold cluster pipeline
- `scripts/cluster/run_within_no_fr_with_fr_cdf.sh`
  - evaluates the within no-FR line on CDF with FR groups included
- `scripts/cluster/run_within_no_fr_main_ablation_with_fr_cdf.sh`
  - computes the CDF-only supplement for the within no-FR main ablation
- `scripts/cluster/run_within_no_fr_main_ablation_with_fr_cdf.slurm.sh`
  - slurm wrapper for the previous script
- `scripts/cluster/prefetch_within_cdf_cache.slurm.sh`
  - copies node-local CDF material when the allocated node differs from the source node

### Fixed-threshold CDF ablation

The important preserved setting is:

- `FIXED_THRESHOLD=0.8`

This setting was used for the within no-FR CDF supplement that was later merged into the final ablation line.

## CDF Cache Pipeline Pack

The restored helper pack lives at:

- `scripts/cluster/cdf_cache_pipeline_pack/`

Purpose:

- build fake-only `DF40_test_cdf` CLIP CLS caches
- build fake-only `DF40_test_cdf` CLIP patch caches
- build CelebDF-real CLS caches
- build CelebDF-real patch caches

This pack is support infrastructure, not the main repo runtime.

Useful scripts inside it:

- `prepare_cdf_cls_clip.py`
- `prepare_cdf_patch_clip.py`
- `prepare_celebdf_real_cls_clip.py`
- `prepare_celebdf_real_patch_clip.py`
- `run_cdf_cls_batch.sh`
- `run_cdf_patch_batch.sh`
- `run_celebdf_real_cls_batch.sh`
- `run_celebdf_real_patch_batch.sh`

## Files Moved Out On Purpose

The following categories were intentionally removed from the active repo tree:

- report writing bundles and thesis chapter drafts
- old non-within outputs
- old non-within downstream checkpoints and heads

The reason is simple:

- the final project now treats the standardized `within` line as canonical
- old non-within artifacts were trained using non-standardized procedures
- report drafting materials were split out to keep the code repository clean

## Environment Notes

Main environment files:

- `environment.yml`
- `requirements.txt`
- `current.txt`

Typical execution assumes:

- conda env name: `fyp`

The Gradio demo also depends on the current environment matching the repo dependencies.

## Practical Shortcuts

If you want:

- final within no-FR ablation table
  - read `within_outputs/no_fr_main_ablation_line.csv`
- full within ablation table
  - read `within_outputs/full_main_ablation_line.csv`
- restored intermediate no-FR CDF merge states
  - read `within_outputs/no_fr_main_ablation_line.before_*`
- detailed no-FR CDF method-level replay breakdown
  - read `within_outputs/within_no_fr_with_fr_cpu_gpu10_cdf_replay_eval.csv`
- threshold trade-off on that restored no-FR CDF run
  - read `within_outputs/within_no_fr_with_fr_cpu_gpu10_threshold_sweep_cdf.csv`
- browser demo entrypoint
  - run `demo/gradio_app.py`

## Boundary To Remember

When in doubt:

- trust `within_checkpoints/`, `within_outputs/`, and `scripts/within_*.py`
- keep `checkpoints/upstream/` in place
- treat cluster cache-prep helpers as support tooling
- do not move report bundle content back into this repository unless there is a specific reason
