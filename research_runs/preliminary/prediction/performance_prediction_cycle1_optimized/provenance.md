# Performance Prediction Optimized Run Provenance

- run_kind: `full pilot rerun`
- optimization_mode: `cpu_cache_only`
- targets: `Average ⬆️, MMLU-PRO, MATH Lvl 5`
- gpu_used: `no`
- cuda_visible_devices: `n/a`
- device_arg: `n/a`
- measured_wall_time: `not separately recorded during the final rerun`
- chosen_final_mode: `cpu_cache_only`

## Bakeoff Decision Record

- `repeated_cpu`: `7:33.59`
- `cpu_cache_only`: `0:45.37`
- `cache_plus_gpu0` on GPU 0: `0:36.39`

## Decision

`cpu_cache_only` was kept for the final rerun. It matched the earlier pilot exactly while cutting the single-target bakeoff runtime by about `10x` relative to `repeated_cpu`. The GPU path was only modestly faster than the CPU-cached mode, so it was not kept.
