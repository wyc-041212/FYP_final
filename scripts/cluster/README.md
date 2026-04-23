# Cluster Script Index

This directory contains the cluster-side runners and cache-sync helpers that still matter for the final `within` pipeline.

Use this folder for:

- rerunning CDF evaluation on the cluster
- syncing node-local `/tmp` CDF caches between machines
- regenerating the no-FR CDF supplement used by the final ablation line
- rebuilding CDF / CelebDF-real cache material through the standalone helper pack

Do not treat this folder as the primary project entrypoint. The local source of truth is still:

- `/Users/wuyuchen/Desktop/FYP_final/scripts/within_*.py`
- `/Users/wuyuchen/Desktop/FYP_final/within_outputs/`
- `/Users/wuyuchen/Desktop/FYP_final/within_checkpoints/`

## Main Current Scripts

- `within_run_cdf_cluster_pipeline.slurm.sh`
  - main within CDF pipeline launcher
  - runs threshold sweep and replay evaluation for the within full and within no-FR settings
  - can copy missing `/tmp` cache material from `SOURCE_NODE`

- `run_within_no_fr_with_fr_cdf.sh`
  - evaluates the within no-FR checkpoints on CDF while including FR groups
  - writes detailed threshold-sweep and replay outputs into `within_outputs/`

- `run_within_no_fr_main_ablation_with_fr_cdf.sh`
  - computes the CDF-only supplement for the within no-FR main ablation
  - this is the script behind the restored `within_no_fr_main_ablation_with_fr_cdf_cpu_gpu10.*` outputs
  - supports `FIXED_THRESHOLD=0.8` for the fixed-threshold rerun used in the final merged no-FR ablation line

- `run_within_no_fr_main_ablation_with_fr_cdf.slurm.sh`
  - slurm wrapper around the previous script
  - typical usage:

```bash
sbatch --export=ALL,FIXED_THRESHOLD=0.8 \
  /Users/wuyuchen/Desktop/FYP_final/scripts/cluster/run_within_no_fr_main_ablation_with_fr_cdf.slurm.sh
```

- `prefetch_within_cdf_cache.slurm.sh`
  - helper to copy node-local CDF caches before running a job on a different GPU node

- `within_run_cdf_cluster_pipeline_mount_tmp.slurm.sh`
  - variant for cases where `/tmp` cache material needs to be mounted or synchronized explicitly

## Older Support Scripts Kept For Reference

- `run_cdf_cache_gpu10.slurm.sh`
  - older CDF cache generation job

- `run_cdf_infer_gpu10.slurm.sh`
  - older CDF inference / cache job

- `run_cdf_cluster_pipeline.slurm.sh`
  - older generic CDF pipeline launcher retained for reference

These are not the mainline documentation target anymore, but they are still useful when tracing how earlier cluster-side cache generation was done.

## Standalone Cache Preparation Pack

The restored helper pack lives in:

- `/Users/wuyuchen/Desktop/FYP_final/scripts/cluster/cdf_cache_pipeline_pack`

Purpose:

- build fake-only `DF40_test_cdf` CLIP CLS caches
- build fake-only `DF40_test_cdf` CLIP patch caches
- build `Celeb-DF-v2` real CLS caches
- build `Celeb-DF-v2` real patch caches

Useful files in that pack:

- `prepare_cdf_cls_clip.py`
- `prepare_cdf_patch_clip.py`
- `prepare_celebdf_real_cls_clip.py`
- `prepare_celebdf_real_patch_clip.py`
- `run_cdf_cls_batch.sh`
- `run_cdf_patch_batch.sh`
- `run_celebdf_real_cls_batch.sh`
- `run_celebdf_real_patch_batch.sh`
- `launch_cdf_cache_cluster.sh`
- `launch_celebdf_real_cls_patch_cluster.sh`

Treat that pack as support tooling for cache preparation, not as the main experiment repo.

## Common Environment Assumptions

Most scripts assume:

- conda env name: `fyp`
- source node default: `gpu10`
- CDF cls cache root on cluster:
  - `/home/comp/f2256768/cdf_cache/cache_clip`
- CDF patch cache root on cluster:
  - `/tmp/f2256768/cdf_cache/cache_clip_patch`
- compact CDF patch cache on cluster:
  - `/tmp/f2256768/fyp_final_compact_cdf`
- CelebDF-real cls cache on cluster:
  - `/tmp/celebdf_real_clip`
- compact CelebDF-real patch cache on cluster:
  - `/tmp/f2256768/fyp_final_compact_celebdf_real`

Because `/tmp` is node-local on this cluster, jobs scheduled onto a different machine may need an explicit copy step before they can run successfully.

## Output Convention

These scripts write structured outputs into:

- `/Users/wuyuchen/Desktop/FYP_final/within_outputs/`

not into this `scripts/cluster/` directory.

The most important restored result families tied to these scripts are:

- `within_outputs/within_no_fr_with_fr_cpu_gpu10_cdf_replay_eval.*`
- `within_outputs/within_no_fr_with_fr_cpu_gpu10_threshold_sweep_cdf.*`
- `within_outputs/within_no_fr_main_ablation_with_fr_cdf_cpu_gpu10.*`
- `within_outputs/no_fr_main_ablation_line.before_cdf_merge.*`
- `within_outputs/no_fr_main_ablation_line.before_fixed_0p8_merge.*`
- `within_outputs/no_fr_main_ablation_line_cdf_only.*`

## Rule Of Use

Use these scripts when you need provenance or reruns for cluster-generated CDF results.

Do not use them as a substitute for:

- local runtime testing
- Gradio demo usage
- the canonical within result tables already present in `within_outputs/`
