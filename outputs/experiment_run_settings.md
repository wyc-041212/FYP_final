# Experiment Run Settings

This file records the exact local runtime settings used for the refreshed final-report experiments on `2026-04-06`.

## Environment

- Conda env: `fyp`
- Device used for all local runs: `cpu`
- Reason: upstream route inference hits `torch.linalg.qr`; current local MPS path is not stable for this runtime.
- Repository root: `/Users/wuyuchen/Desktop/FYP_final`

## Core Artifacts

### Full corrected mainline

- Upstream: `/Users/wuyuchen/Desktop/FYP_final/checkpoints/upstream/checkpoint_best_hybrid_manifold.pt`
- Patch branch: `/Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/patch_branch.joblib`
- Pair branch: `/Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/pair_branch_full_no_bg_keep_hair.joblib`
- Head: `/Users/wuyuchen/Desktop/FYP_final/checkpoints/heads/route_meta_head_full_patch_normal_pair_nobg.joblib`

### no-FR corrected mainline

- Upstream: `/Users/wuyuchen/Desktop/FYP_final/checkpoints/upstream/checkpoint_no_fr.pt`
- Patch branch: `/Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/patch_branch_no_fr.joblib`
- Pair branch: `/Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/pair_branch_no_fr_no_bg_keep_hair.joblib`
- Head: `/Users/wuyuchen/Desktop/FYP_final/checkpoints/heads/route_meta_head_no_fr_patch_normal_pair_nobg.joblib`

### Canonical pair-region variants

- Full pair branch: `/Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/pair_branch_full_canonical.joblib`
- Full head: `/Users/wuyuchen/Desktop/FYP_final/checkpoints/heads/route_meta_head_full_patch_normal_pair_canonical.joblib`
- no-FR pair branch: `/Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/pair_branch_no_fr_canonical.joblib`
- no-FR head: `/Users/wuyuchen/Desktop/FYP_final/checkpoints/heads/route_meta_head_no_fr_patch_normal_pair_canonical.joblib`

## Cache Layout Used Locally

- `cache/cls` -> `/Volumes/未命名/cache_clip`
- `cache/patch` -> `/Volumes/未命名/cache_clip_patch`
- `cache/compact` -> `/Users/wuyuchen/Desktop/FYP_final_backup/cache/compact`
- Local real-only subset used as `cdf_real` proxy:
  - `/Users/wuyuchen/Desktop/real/cls_Celeb-real.npz`
  - `/Users/wuyuchen/Desktop/real/patch_Celeb-real.npz`
  - `/Users/wuyuchen/Desktop/real/cls_YouTube-real.npz`
  - `/Users/wuyuchen/Desktop/real/patch_YouTube-real.npz`

## Commands

### Full threshold sweep

```bash
conda run -n fyp python /Users/wuyuchen/Desktop/FYP_final/scripts/run_threshold_sweep.py \
  --upstream-checkpoint /Users/wuyuchen/Desktop/FYP_final/checkpoints/upstream/checkpoint_best_hybrid_manifold.pt \
  --patch-branch /Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/patch_branch.joblib \
  --pair-branch /Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/pair_branch_full_no_bg_keep_hair.joblib \
  --route-meta-head /Users/wuyuchen/Desktop/FYP_final/checkpoints/heads/route_meta_head_full_patch_normal_pair_nobg.joblib \
  --head-meta /Users/wuyuchen/Desktop/FYP_final/checkpoints/heads/route_meta_head_full_patch_normal_pair_nobg_meta.json \
  --device cpu \
  --output-json /Users/wuyuchen/Desktop/FYP_final/outputs/full_threshold_sweep.json \
  --output-csv /Users/wuyuchen/Desktop/FYP_final/outputs/full_threshold_sweep.csv
```

### no-FR threshold sweep

```bash
conda run -n fyp python /Users/wuyuchen/Desktop/FYP_final/scripts/run_threshold_sweep.py \
  --upstream-checkpoint /Users/wuyuchen/Desktop/FYP_final/checkpoints/upstream/checkpoint_no_fr.pt \
  --patch-branch /Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/patch_branch_no_fr.joblib \
  --pair-branch /Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/pair_branch_no_fr_no_bg_keep_hair.joblib \
  --route-meta-head /Users/wuyuchen/Desktop/FYP_final/checkpoints/heads/route_meta_head_no_fr_patch_normal_pair_nobg.joblib \
  --head-meta /Users/wuyuchen/Desktop/FYP_final/checkpoints/heads/route_meta_head_no_fr_patch_normal_pair_nobg_meta.json \
  --device cpu \
  --output-json /Users/wuyuchen/Desktop/FYP_final/outputs/no_fr_threshold_sweep.json \
  --output-csv /Users/wuyuchen/Desktop/FYP_final/outputs/no_fr_threshold_sweep.csv
```

### Full main ablation line

```bash
conda run -n fyp python /Users/wuyuchen/Desktop/FYP_final/scripts/run_main_ablation_line.py \
  --upstream-checkpoint /Users/wuyuchen/Desktop/FYP_final/checkpoints/upstream/checkpoint_best_hybrid_manifold.pt \
  --patch-branch /Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/patch_branch.joblib \
  --pair-branch /Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/pair_branch_full_no_bg_keep_hair.joblib \
  --route-meta-head /Users/wuyuchen/Desktop/FYP_final/checkpoints/heads/route_meta_head_full_patch_normal_pair_nobg.joblib \
  --head-meta /Users/wuyuchen/Desktop/FYP_final/checkpoints/heads/route_meta_head_full_patch_normal_pair_nobg_meta.json \
  --device cpu \
  --output-json /Users/wuyuchen/Desktop/FYP_final/outputs/full_main_ablation_line.json \
  --output-csv /Users/wuyuchen/Desktop/FYP_final/outputs/full_main_ablation_line.csv
```

### no-FR main ablation line

```bash
conda run -n fyp python /Users/wuyuchen/Desktop/FYP_final/scripts/run_main_ablation_line.py \
  --upstream-checkpoint /Users/wuyuchen/Desktop/FYP_final/checkpoints/upstream/checkpoint_no_fr.pt \
  --patch-branch /Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/patch_branch_no_fr.joblib \
  --pair-branch /Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/pair_branch_no_fr_no_bg_keep_hair.joblib \
  --route-meta-head /Users/wuyuchen/Desktop/FYP_final/checkpoints/heads/route_meta_head_no_fr_patch_normal_pair_nobg.joblib \
  --head-meta /Users/wuyuchen/Desktop/FYP_final/checkpoints/heads/route_meta_head_no_fr_patch_normal_pair_nobg_meta.json \
  --device cpu \
  --output-json /Users/wuyuchen/Desktop/FYP_final/outputs/no_fr_main_ablation_line.json \
  --output-csv /Users/wuyuchen/Desktop/FYP_final/outputs/no_fr_main_ablation_line.csv
```

### Pair-region subset summaries

The pair-region subset summary was built from six `run_main_ablation_line.py` runs:

- Full + `all_regions`
- Full + `canonical`
- Full + `no_background_keep_hair`
- no-FR + `all_regions`
- no-FR + `canonical`
- no-FR + `no_background_keep_hair`

The resulting artifacts are:

- `/Users/wuyuchen/Desktop/FYP_final/outputs/pair_region_subset_summary.json`
- `/Users/wuyuchen/Desktop/FYP_final/outputs/pair_region_subset_summary.csv`

### Full small OOD probe sweep

```bash
conda run -n fyp python /Users/wuyuchen/Desktop/FYP_final/scripts/run_small_ood_sweep.py \
  --upstream-checkpoint /Users/wuyuchen/Desktop/FYP_final/checkpoints/upstream/checkpoint_best_hybrid_manifold.pt \
  --patch-branch /Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/patch_branch.joblib \
  --pair-branch /Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/pair_branch_full_no_bg_keep_hair.joblib \
  --route-meta-head /Users/wuyuchen/Desktop/FYP_final/checkpoints/heads/route_meta_head_full_patch_normal_pair_nobg.joblib \
  --device cpu \
  --output-json /Users/wuyuchen/Desktop/FYP_final/outputs/full_small_ood_probe_sweep.json \
  --output-csv /Users/wuyuchen/Desktop/FYP_final/outputs/full_small_ood_probe_sweep.csv
```

Additional video-level output:

- `/Users/wuyuchen/Desktop/FYP_final/outputs/full_small_ood_probe_sweep_video_level.csv`

### no-FR small OOD probe sweep

```bash
conda run -n fyp python /Users/wuyuchen/Desktop/FYP_final/scripts/run_small_ood_sweep.py \
  --upstream-checkpoint /Users/wuyuchen/Desktop/FYP_final/checkpoints/upstream/checkpoint_no_fr.pt \
  --patch-branch /Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/patch_branch_no_fr.joblib \
  --pair-branch /Users/wuyuchen/Desktop/FYP_final/checkpoints/downstream/pair_branch_no_fr_no_bg_keep_hair.joblib \
  --route-meta-head /Users/wuyuchen/Desktop/FYP_final/checkpoints/heads/route_meta_head_no_fr_patch_normal_pair_nobg.joblib \
  --device cpu \
  --output-json /Users/wuyuchen/Desktop/FYP_final/outputs/no_fr_small_ood_probe_sweep.json \
  --output-csv /Users/wuyuchen/Desktop/FYP_final/outputs/no_fr_small_ood_probe_sweep.csv
```

Additional video-level output:

- `/Users/wuyuchen/Desktop/FYP_final/outputs/no_fr_small_ood_probe_sweep_video_level.csv`

## Probe Source Mapping

### animations

- CLS source: `/Users/wuyuchen/Desktop/FYP_final_backup/outputs/demos/animations_20260324_1955/sampled/<sample>/cls_sampled.npz`
- Patch source: `/Users/wuyuchen/Desktop/FYP_final_backup/outputs/reports/domain_shift_audit_20260324/animation_patch_reextract/<sample>/patch_sampled_reextract.npz`
- Fallback patch source if repaired patch is missing: `.../sampled/<sample>/patch_sampled.npz`
- `smoke_test` bundle excluded

### kobe_test

- Bundle root: `/Users/wuyuchen/Desktop/kobe_test/kobe_prediction_samples/kobe_sample`

### tiktok

- Bundle root: `/Users/wuyuchen/Desktop/FYP_final_backup/outputs/probes/tiktok_topk_no_fr_20260329/sampled_bundle`

## Notes on Interpretation

- `animations` and `kobe_test` have historically been treated as fake-style probes in older exploratory scripts.
- `tiktok` is better treated as an open probe; the refreshed sweep therefore reports fake-hit rates and mean fake probabilities without forcing a benchmark-style label claim.
- All small OOD outputs should be framed as probe evidence rather than formal benchmark results.

## Cluster Recommendation for Remaining CDF Runs

- Start with `4090` if the remaining workload is replay/inference over already prepared CLS / patch / compact caches.
- Move to `A100` only if:
  - compact regeneration or patch re-extraction is needed at scale,
  - multiple variants need to be batched in parallel,
  - or replay memory/throughput becomes the bottleneck.
