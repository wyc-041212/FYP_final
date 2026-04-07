# Route Conditioned Rule Eval

口径：
- 上游用原版 route。
- `patch` 用原版 all-regions experts。
- `pair` 用 `no_background_keep_hair` experts。
- 这是 rule-based prototype，不含新 head。
- 这一版先评估 `test_ff / ood / cdf_real`；`animation` 暂时未评估，因为当前没有对应的 `cls` 重提取缓存。

## Overall

| model | test_ff_acc | ood_acc | cdf_real_acc | animation_acc |
|---|---:|---:|---:|---:|
| `route_top1_patch_only` | 85.35 | 82.50 | 46.57 | NA |
| `route_top1_grayzone_balanced` | 85.90 | 82.29 | 54.77 | NA |
| `route_top1_grayzone_realbias_fe` | 85.88 | 82.58 | 54.77 | NA |
| `route_top2_union_grayzone_balanced` | 88.49 | 82.69 | 44.72 | NA |
| `route_gap_g010_grayzone_balanced` | 87.35 | 81.83 | 52.98 | NA |
| `route_gap_g015_grayzone_balanced` | 87.40 | 81.84 | 52.33 | NA |
| `route_gap_g020_grayzone_balanced` | 87.50 | 81.89 | 51.82 | NA |

## route_top1_patch_only

| group | test_ff_acc | ood_acc |
|---|---:|---:|
| `EFS` | 92.54 | 71.84 |
| `FS` | 84.30 | 77.64 |
| `FR` | 82.52 | 84.84 |
| `FE` | 82.02 | 95.69 |

## route_top1_grayzone_balanced

| group | test_ff_acc | ood_acc |
|---|---:|---:|
| `EFS` | 94.00 | 71.70 |
| `FS` | 83.00 | 77.38 |
| `FR` | 81.39 | 84.41 |
| `FE` | 85.19 | 95.68 |

## route_top1_grayzone_realbias_fe

| group | test_ff_acc | ood_acc |
|---|---:|---:|
| `EFS` | 94.02 | 72.62 |
| `FS` | 83.00 | 77.36 |
| `FR` | 81.30 | 85.18 |
| `FE` | 85.22 | 95.17 |

## route_top2_union_grayzone_balanced

| group | test_ff_acc | ood_acc |
|---|---:|---:|
| `EFS` | 92.75 | 69.85 |
| `FS` | 88.78 | 79.82 |
| `FR` | 88.16 | 85.48 |
| `FE` | 84.28 | 95.61 |

## route_gap_g010_grayzone_balanced

| group | test_ff_acc | ood_acc |
|---|---:|---:|
| `EFS` | 92.68 | 70.83 |
| `FS` | 86.05 | 77.23 |
| `FR` | 85.89 | 83.94 |
| `FE` | 84.78 | 95.32 |

## route_gap_g015_grayzone_balanced

| group | test_ff_acc | ood_acc |
|---|---:|---:|
| `EFS` | 92.60 | 70.82 |
| `FS` | 86.20 | 77.19 |
| `FR` | 86.17 | 84.05 |
| `FE` | 84.62 | 95.29 |

## route_gap_g020_grayzone_balanced

| group | test_ff_acc | ood_acc |
|---|---:|---:|
| `EFS` | 92.57 | 70.67 |
| `FS` | 86.46 | 77.47 |
| `FR` | 86.41 | 84.13 |
| `FE` | 84.57 | 95.30 |

