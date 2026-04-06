# Cluster Experiment Handoff

Last updated: 2026-04-06

This file is a direct handoff for continuing the final-report experiments on the cluster. It is not report prose. It records:

- which corrected checkpoints should be treated as current;
- what has already been rerun locally;
- what still needs to be run on the cluster;
- which cache roots and output names should be used.

## 1. Current corrected mainline variants

### Full corrected mainline
- Upstream:
  - `/home/comp/f2256768/FYP_final/checkpoints/upstream/checkpoint_best_hybrid_manifold.pt`
- Patch:
  - `/home/comp/f2256768/FYP_final/checkpoints/downstream/patch_branch.joblib`
- Pair:
  - `/home/comp/f2256768/FYP_final/checkpoints/downstream/pair_branch_full_no_bg_keep_hair.joblib`
- Head:
  - `/home/comp/f2256768/FYP_final/checkpoints/heads/route_meta_head_full_patch_normal_pair_nobg.joblib`
  - `/home/comp/f2256768/FYP_final/checkpoints/heads/route_meta_head_full_patch_normal_pair_nobg_meta.json`

Meaning:
- patch branch stays normal;
- pair branch uses `no_background_keep_hair`;
- head was retrained on normal patch + no-bg pair.

### no-FR corrected mainline
- Upstream:
  - `/home/comp/f2256768/FYP_final/checkpoints/upstream/checkpoint_no_fr.pt`
- Patch:
  - `/home/comp/f2256768/FYP_final/checkpoints/downstream/patch_branch_no_fr.joblib`
- Pair:
  - `/home/comp/f2256768/FYP_final/checkpoints/downstream/pair_branch_no_fr_no_bg_keep_hair.joblib`
- Head:
  - `/home/comp/f2256768/FYP_final/checkpoints/heads/route_meta_head_no_fr_patch_normal_pair_nobg.joblib`
  - `/home/comp/f2256768/FYP_final/checkpoints/heads/route_meta_head_no_fr_patch_normal_pair_nobg_meta.json`

Important:
- do not use `patch_branch_no_fr_no_bg_keep_hair.joblib` as the final intended no-FR patch branch;
- do not use `route_meta_head_no_fr_no_bg_keep_hair.joblib` as the corrected no-FR final head.

## 2. Local reruns already completed

These outputs already exist locally and do not need to be rerun on cluster unless verification or regeneration is explicitly needed.

### Threshold sweeps
- `outputs/full_threshold_sweep.json`
- `outputs/full_threshold_sweep.csv`
- `outputs/no_fr_threshold_sweep.json`
- `outputs/no_fr_threshold_sweep.csv`

### Main ablation line
- `outputs/full_main_ablation_line.json`
- `outputs/full_main_ablation_line.csv`
- `outputs/no_fr_main_ablation_line.json`
- `outputs/no_fr_main_ablation_line.csv`

This line compares:
- `route_only`
- `patch_only`
- `pair_only`
- `route_meta_fusion`

### Pair-region subset
- `outputs/pair_region_subset_summary.json`
- `outputs/pair_region_subset_summary.csv`

The following six runs are already available locally:
- `full_pair_region_all_regions`
- `full_pair_region_canonical`
- `full_pair_region_no_background_keep_hair`
- `no_fr_pair_region_all_regions`
- `no_fr_pair_region_canonical`
- `no_fr_pair_region_no_background_keep_hair`

### Small OOD probes
- `outputs/full_small_ood_probe_sweep.json`
- `outputs/full_small_ood_probe_sweep.csv`
- `outputs/full_small_ood_probe_sweep_video_level.csv`
- `outputs/no_fr_small_ood_probe_sweep.json`
- `outputs/no_fr_small_ood_probe_sweep.csv`
- `outputs/no_fr_small_ood_probe_sweep_video_level.csv`

Retained display examples:
- `tiktok/564_1774777598`
- `animations/example--d13`

These are presentation/display examples, not official benchmark rows.

## 3. Remaining cluster-side tasks

The main remaining gap is the full CDF and larger external-real line.

### Priority A: CDF threshold sweep
Goal:
- rerun threshold sweep for:
  - full corrected mainline
  - no-FR corrected mainline
- but this time on the real cluster-side CDF cache, not only the local desktop real-only proxy

Need:
- CDF CLS cache:
  - `/home/comp/f2256768/cdf_cache/cache_clip`
- CDF patch cache:
  - `/tmp/f2256768/cdf_cache/cache_clip_patch`
- CDF compact cache:
  - `/tmp/f2256768/fyp_final_compact_cdf`

Desired outputs:
- `outputs/full_threshold_sweep_cdf.json`
- `outputs/full_threshold_sweep_cdf.csv`
- `outputs/no_fr_threshold_sweep_cdf.json`
- `outputs/no_fr_threshold_sweep_cdf.csv`

### Priority B: CDF mainline replay / infer summary
Goal:
- produce a cleaner final summary for full vs no-FR on the full CDF line at selected thresholds

Desired outputs:
- `outputs/full_cdf_replay_eval.json`
- `outputs/full_cdf_replay_eval.csv`
- `outputs/no_fr_cdf_replay_eval.json`
- `outputs/no_fr_cdf_replay_eval.csv`

### Priority C: larger external-real line if needed
Possible sources:
- `/tmp/celebdf_real_clip`
- `/tmp/f2256768/celebdf_real_outputs`
- `/tmp/f2256768/fyp_final_compact_celebdf_real`

Only do this if, after CDF, more external-real evidence is still needed for Chapter 7 or Chapter 8.

## 4. Cluster cache and data roots

### CDF roots
- Data:
  - `/tmp/f2256768/DF40_test_cdf`
- CLS cache:
  - `/home/comp/f2256768/cdf_cache/cache_clip`
- Patch cache:
  - `/tmp/f2256768/cdf_cache/cache_clip_patch`
- Compact cache:
  - `/tmp/f2256768/fyp_final_compact_cdf`

### External-real / CelebDF-real roots
- `/tmp/celebdf_real_clip`
- `/tmp/f2256768/celebdf_real_outputs`
- `/tmp/f2256768/fyp_final_compact_celebdf_real`

## 5. Scripts to use

The local reruns were driven by these repo scripts:

- `scripts/run_threshold_sweep.py`
- `scripts/run_main_ablation_line.py`
- `scripts/run_small_ood_sweep.py`

The cluster should reuse the same scripts where possible, with cluster paths substituted for local absolute paths.

## 6. Cluster command pattern

Use the `fyp` conda environment:

```bash
conda run -n fyp python /home/comp/f2256768/FYP_final/scripts/run_threshold_sweep.py ...
```

For CDF work, the important substitutions are:
- `--cls-cache-root /home/comp/f2256768/cdf_cache/cache_clip`
- `--patch-cache-root /tmp/f2256768/cdf_cache/cache_clip_patch`
- `--compact-cache-dir /tmp/f2256768/fyp_final_compact_cdf`

If the script needs explicit data-root style arguments, use the same CDF roots already hard-coded in the existing cluster slurm scripts.

## 7. GPU recommendation

For the remaining work:
- use `4090` first if the task is replay/inference/threshold sweeping over already-prepared caches;
- move to `A100` only if:
  - compact regeneration is required,
  - patch re-extraction is required,
  - or multi-variant batched throughput becomes the bottleneck.

Current expectation:
- CDF threshold sweeps and replay should be fine on `4090`.

## 8. Interpretation reminders

- `no_background_keep_hair` is the retained final policy for the pair branch, not for the patch branch.
- `route_only / patch_only / pair_only / route_meta_fusion` should be interpreted as the main ablation line, not as four equally official final systems.
- `small OOD probes` are for display/presentation support and appendix-style case-study evidence.
- The difficult unresolved issue is threshold selection across:
  - `test_ff`
  - `test_ood`
  - full CDF / external-real

Therefore, the cluster-side objective is not to invent a new evaluation protocol, but to extend the same corrected full/no-FR pipelines onto the full CDF line so that threshold trade-offs can be analyzed with the complete data.

## 9. Minimum outputs to bring back from cluster

At minimum, bring back:
- `full_threshold_sweep_cdf.json`
- `full_threshold_sweep_cdf.csv`
- `no_fr_threshold_sweep_cdf.json`
- `no_fr_threshold_sweep_cdf.csv`

Nice to have:
- `full_cdf_replay_eval.json`
- `full_cdf_replay_eval.csv`
- `no_fr_cdf_replay_eval.json`
- `no_fr_cdf_replay_eval.csv`

After those logs are synced back to local, the next step is to jointly decide the final operating thresholds and the final Chapter 7 / Chapter 8 reporting tables.
