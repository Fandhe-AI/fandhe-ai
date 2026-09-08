# Layer B 集計（node=dgx。イシュー #1301）

off runs: ['layerB-dgx-off-run1.log', 'layerB-dgx-off-run2.log', 'layerB-dgx-off-run3.log', 'layerB-dgx-off-run4.log', 'layerB-dgx-off-run5.log']
on runs: ['layerB-dgx-on-run1.log', 'layerB-dgx-on-run2.log', 'layerB-dgx-on-run3.log', 'layerB-dgx-on-run4.log', 'layerB-dgx-on-run5.log']

## N=512

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 0.0089 | 5 | 0.0081 | 5 | 0.9101 |
| kernel | 1.5221 | 5 | 1.4988 | 5 | 0.9847 |
| tensor_wrap | 0.0019 | 5 | 0.0026 | 5 | 1.3684 |
| ops_gemm | 1.7383 | 5 | 1.8538 | 5 | 1.0664 |
| tape_matmul | 1.6863 | 5 | 1.8716 | 5 | 1.1099 |
| to_tensor | 0.0001 | 5 | 0.0002 | 5 | 2.0000 |
| host_copy | 0.3658 | 5 | 0.3814 | 5 | 1.0426 |
| checksum | 0.1872 | 5 | 0.1870 | 5 | 0.9989 |

## N=1024

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 0.0306 | 5 | 0.0330 | 5 | 1.0784 |
| kernel | 4.5141 | 5 | 5.1841 | 5 | 1.1484 |
| tensor_wrap | 0.0031 | 5 | 0.0035 | 5 | 1.1290 |
| ops_gemm | 5.4276 | 5 | 6.1900 | 5 | 1.1405 |
| tape_matmul | 5.3392 | 5 | 7.4773 | 5 | 1.4005 |
| to_tensor | 0.0002 | 5 | 0.0002 | 5 | 1.0000 |
| host_copy | 1.4731 | 5 | 1.4989 | 5 | 1.0175 |
| checksum | 0.7561 | 5 | 0.7527 | 5 | 0.9955 |

## N=2048

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 3.2554 | 5 | 1.7901 | 5 | 0.5499 |
| kernel | 25.3977 | 5 | 25.7517 | 5 | 1.0139 |
| tensor_wrap | 0.0057 | 5 | 0.0051 | 5 | 0.8947 |
| ops_gemm | 26.8177 | 5 | 28.5179 | 5 | 1.0634 |
| tape_matmul | 26.5594 | 5 | 27.3093 | 5 | 1.0282 |
| to_tensor | 0.0002 | 5 | 0.0002 | 5 | 1.0000 |
| host_copy | 5.8182 | 5 | 5.7054 | 5 | 0.9806 |
| checksum | 3.0092 | 5 | 3.0043 | 5 | 0.9984 |

