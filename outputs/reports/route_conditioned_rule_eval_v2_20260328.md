# Route Conditioned Rule Eval V2

口径：
- 上游用原版 route，但不再硬选最终 group。
- 主组改用 `patch_top`，route 只做 support/filter。
- `pair` 用 `no_background_keep_hair` experts。
- 这一版先评估 `test_ff / ood / cdf_real`；`animation` 暂时未评估。

## Overall

| model | test_ff_acc | ood_acc | cdf_real_acc | animation_acc |
|---|---:|---:|---:|---:|
| `route_top1_grayzone_realbias_fe` | 85.83 | 83.02 | 54.77 | NA |
| `patch_top_patch_only` | 87.56 | 81.22 | 32.22 | NA |
| `patch_top_grayzone_balanced` | 89.35 | 82.44 | 42.20 | NA |
| `patch_top_grayzone_realbias_fe` | 89.72 | 83.42 | 42.42 | NA |
| `patch_top_in_route_top2_else_high_balanced` | 89.40 | 82.50 | 42.38 | NA |
| `patch_top_in_route_top2_else_high_and_pair_balanced` | 89.18 | 81.98 | 45.45 | NA |
| `patch_top_supported_else_high_balanced` | 89.31 | 82.43 | 42.32 | NA |
| `patch_top_supported_else_high_and_pair_balanced` | 88.81 | 81.35 | 44.73 | NA |

