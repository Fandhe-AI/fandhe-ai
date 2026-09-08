# Layer B 集計（node=dgx。イシュー #1301）

off runs: ['layerB-dgx-off-run1.log', 'layerB-dgx-off-run2.log', 'layerB-dgx-off-run3.log', 'layerB-dgx-off-run4.log', 'layerB-dgx-off-run5.log']
on runs: ['layerB-dgx-on-clean-run1.log', 'layerB-dgx-on-clean-run2.log', 'layerB-dgx-on-clean-run3.log', 'layerB-dgx-on-clean-run4.log', 'layerB-dgx-on-clean-run5.log']

## N=1024

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 0.0306 | 5 | 0.0269 | 5 | 0.8791 |
| kernel | 4.5141 | 5 | 4.4351 | 5 | 0.9825 |
| tensor_wrap | 0.0031 | 5 | 0.0029 | 5 | 0.9355 |
| ops_gemm | 5.4276 | 5 | 5.3701 | 5 | 0.9894 |
| tape_matmul | 5.3392 | 5 | 5.5132 | 5 | 1.0326 |
| to_tensor | 0.0002 | 5 | 0.0002 | 5 | 1.0000 |
| host_copy | 1.4731 | 5 | 1.4461 | 5 | 0.9817 |
| checksum | 0.7561 | 5 | 0.7469 | 5 | 0.9878 |

## N=2048

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 3.2554 | 5 | 1.6817 | 5 | 0.5166 |
| kernel | 25.3977 | 5 | 24.9311 | 5 | 0.9816 |
| tensor_wrap | 0.0057 | 5 | 0.0052 | 5 | 0.9123 |
| ops_gemm | 26.8177 | 5 | 26.7420 | 5 | 0.9972 |
| tape_matmul | 26.5594 | 5 | 26.7057 | 5 | 1.0055 |
| to_tensor | 0.0002 | 5 | 0.0003 | 5 | 1.5000 |
| host_copy | 5.8182 | 5 | 5.7344 | 5 | 0.9856 |
| checksum | 3.0092 | 5 | 3.0053 | 5 | 0.9987 |

