# Strategy Learning Search

口径：
- 这是 strategy-level oracle search，用来找“最优行为模式”，不是最终可部署规则。
- 策略只用可观测分数：`route_gap / route_entropy / route_branch_match_any`。
- 目标函数综合了 `test_ff_nonFR bacc / ood_nonFR bacc / cdf_real_acc / animation_acc`。

## Constant Baselines

| action | obj_balanced | obj_real_guard | test_ff_nonFR_bacc | ood_nonFR_bacc | cdf_real | animation | FR_test_fake | FR_ood_fake |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `patch_top1` | 62.14 | 58.33 | 87.66 | 80.08 | 45.75 | 0.00 | 83.63 | 37.89 |
| `patch_top2` | 58.72 | 53.90 | 86.34 | 78.92 | 34.40 | 1.14 | 93.29 | 50.00 |
| `patch_top3` | 58.41 | 53.52 | 85.64 | 77.89 | 32.88 | 4.92 | 96.03 | 62.11 |
| `union_top1` | 60.60 | 56.12 | 80.39 | 72.71 | 31.70 | 42.42 | 89.13 | 53.91 |
| `union_top2` | 58.18 | 53.05 | 76.21 | 68.73 | 21.17 | 60.23 | 95.67 | 70.31 |
| `union_top3` | 62.92 | 57.82 | 75.05 | 66.69 | 19.92 | 100.00 | 97.75 | 79.30 |

## Best Balanced Objective

- policy: `{'type': 'constant', 'action': 'union_top3'}`
- metrics: `test_ff_nonFR_bacc=75.05`, `ood_nonFR_bacc=66.69`, `cdf_real=19.92`, `animation=100.00`, `FR_test_fake=97.75`, `FR_ood_fake=79.30`

## Best Real-Guard Objective

- policy: `{'type': 'constant', 'action': 'patch_top1'}`
- metrics: `test_ff_nonFR_bacc=87.66`, `ood_nonFR_bacc=80.08`, `cdf_real=45.75`, `animation=0.00`, `FR_test_fake=83.63`, `FR_ood_fake=37.89`

## Best Guarded Objective

- policy: `{'type': 'constant', 'action': 'patch_top1'}`
- metrics: `test_ff_nonFR_bacc=87.66`, `test_ff_real=83.90`, `ood_nonFR_bacc=80.08`, `ood_real=85.48`, `cdf_real=45.75`, `animation=0.00`, `FR_test_fake=83.63`, `FR_ood_fake=37.89`

## Best CDF80-Guarded Objective

- none
