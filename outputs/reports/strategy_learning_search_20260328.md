# Strategy Learning Search

口径：
- 这是 strategy-level oracle search，用来找“最优行为模式”，不是最终可部署规则。
- 策略只用可观测分数：`route_gap / route_entropy / route_branch_match_any`。
- 目标函数综合了 `test_ff bacc / ood bacc / cdf_real_acc / animation_acc`。

## Constant Baselines

| action | obj_balanced | obj_real_guard | test_ff_bacc | ood_bacc | cdf_real | animation |
|---|---:|---:|---:|---:|---:|---:|
| `patch_top1` | 60.88 | 57.30 | 86.26 | 77.02 | 45.75 | 0.00 |
| `patch_top2` | 57.92 | 53.24 | 85.75 | 76.53 | 34.40 | 1.14 |
| `patch_top3` | 57.93 | 53.13 | 85.31 | 76.46 | 32.88 | 4.92 |
| `union_top1` | 59.70 | 55.38 | 79.39 | 70.51 | 31.70 | 42.42 |
| `union_top2` | 57.74 | 52.69 | 75.79 | 67.55 | 21.17 | 60.23 |
| `union_top3` | 62.73 | 57.67 | 74.88 | 66.18 | 19.92 | 100.00 |

## Best Balanced Objective

- policy: `{'type': 'constant', 'action': 'union_top3'}`
- metrics: `test_ff_bacc=74.88`, `ood_bacc=66.18`, `cdf_real=19.92`, `animation=100.00`

## Best Real-Guard Objective

- policy: `{'type': 'constant', 'action': 'union_top3'}`
- metrics: `test_ff_bacc=74.88`, `ood_bacc=66.18`, `cdf_real=19.92`, `animation=100.00`

## Best Guarded Objective

- policy: `{'type': 'constant', 'action': 'patch_top1'}`
- metrics: `test_ff_bacc=86.26`, `test_ff_real=83.90`, `ood_bacc=77.02`, `ood_real=85.48`, `cdf_real=45.75`, `animation=0.00`

## Best CDF80-Guarded Objective

- none
