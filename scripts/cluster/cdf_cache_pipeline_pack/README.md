# CDF Cache Pipeline Pack

This folder packages a small standalone pipeline for building fake-only
`CLIP CLS` and `CLIP patch token` caches for `/Volumes/未命名/DF40_test_cdf`.

It intentionally matches the flat `DF40_test_ff` cache layout:

- `cache_clip/DF40_test_cdf/<GROUP>/manifest_<method>.csv`
- `cache_clip/DF40_test_cdf/<GROUP>/cls_<method>.npz`
- `cache_clip_patch/DF40_test_cdf/<GROUP>/patch_<method>.npz`

Unlike the original `test_ff` paired pipeline, this CDF pack does **not**
require fake-real pair metadata.

## Inputs

- Dataset root:
  `/Volumes/未命名/DF40_test_cdf`
- Source code root used by this pack:
  `/Users/wuyuchen/Desktop/FYP/Fyp_clean/outputs/experiments/balanced_3000_seed42/20260319_022446_route_meta_fusion_pipeline_clean/src`

## Scripts

- `prepare_cdf_cls_clip.py`
  Build fake-only manifests and CLIP CLS caches.
- `prepare_cdf_patch_clip.py`
  Build fake-only CLIP patch caches with face-region labels.
- `run_cdf_cls_batch.sh`
  Batch-build all CLS caches.
- `run_cdf_patch_batch.sh`
  Batch-build all patch caches.

## Smoke Test

These commands are safe examples for a small feasibility check:

```bash
ENV_NAME=fyp \
SRC_ROOT=/Users/wuyuchen/Desktop/FYP/Fyp_clean/outputs/experiments/balanced_3000_seed42/20260319_022446_route_meta_fusion_pipeline_clean/src \
OUT_CACHE_ROOT=/Users/wuyuchen/Desktop/cdf_cache_smoke/cache_clip \
bash /Users/wuyuchen/Desktop/FYP/Fyp_clean/outputs/experiments/balanced_3000_seed42/20260320_cdf_cache_pipeline_pack/run_cdf_cls_batch.sh \
  /Volumes/未命名/DF40_test_cdf/FR/wav2lip \
  /Volumes/未命名/DF40_test_cdf/EFS/DiT \
  /Volumes/未命名/DF40_test_cdf/FS/mobileswap
```

```bash
ENV_NAME=fyp \
SRC_ROOT=/Users/wuyuchen/Desktop/FYP/Fyp_clean/outputs/experiments/balanced_3000_seed42/20260319_022446_route_meta_fusion_pipeline_clean/src \
MANIFEST_CACHE_ROOT=/Users/wuyuchen/Desktop/cdf_cache_smoke/cache_clip \
OUT_CACHE_ROOT=/Users/wuyuchen/Desktop/cdf_cache_smoke/cache_clip_patch \
bash /Users/wuyuchen/Desktop/FYP/Fyp_clean/outputs/experiments/balanced_3000_seed42/20260320_cdf_cache_pipeline_pack/run_cdf_patch_batch.sh \
  /Volumes/未命名/DF40_test_cdf/FR/wav2lip \
  /Volumes/未命名/DF40_test_cdf/EFS/DiT \
  /Volumes/未命名/DF40_test_cdf/FS/mobileswap
```

## Full Run Targets

Suggested real targets for the full build:

- CLS:
  `/Volumes/未命名/cache_clip/DF40_test_cdf`
- patch:
  `/Volumes/未命名/cache_clip_patch/DF40_test_cdf`

## Notes

- `FE` is currently empty under `/Volumes/未命名/DF40_test_cdf`.
- The pack treats `CDF` as fake-only evaluation data.
- `pair_id` in manifests is still populated with a stable folder key for later
  grouping, but it is not used as a fake-real pair link.
