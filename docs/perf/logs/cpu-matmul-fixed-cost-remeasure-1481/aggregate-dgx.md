# Layer B 集計（node=dgx。イシュー #1481（#1301 と同型を再利用））

off runs: ['layerB-dgx-off-run1.log', 'layerB-dgx-off-run2.log', 'layerB-dgx-off-run3.log', 'layerB-dgx-off-run4.log', 'layerB-dgx-off-run5.log']
on runs: ['layerB-dgx-on-run1.log', 'layerB-dgx-on-run2.log', 'layerB-dgx-on-run3.log', 'layerB-dgx-on-run4.log', 'layerB-dgx-on-run5.log']

## N=512

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 0.0091 | 5 | 0.0093 | 5 | 1.0220 |
| kernel | 1.2148 | 5 | 1.2437 | 5 | 1.0238 |
| tensor_wrap | 0.0017 | 5 | 0.0020 | 5 | 1.1765 |
| ops_gemm | 1.6792 | 5 | 1.6636 | 5 | 0.9907 |
| tape_matmul | 1.6160 | 5 | 1.6238 | 5 | 1.0048 |
| to_tensor | 0.0001 | 5 | 0.0001 | 5 | 1.0000 |
| host_copy | 0.3431 | 5 | 0.3446 | 5 | 1.0044 |
| checksum | 0.1872 | 5 | 0.1872 | 5 | 1.0000 |

## N=1024

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 0.0664 | 5 | 0.0375 | 5 | 0.5648 |
| kernel | 3.3738 | 5 | 3.2563 | 5 | 0.9652 |
| tensor_wrap | 0.0019 | 5 | 0.0011 | 5 | 0.5789 |
| ops_gemm | 4.2979 | 5 | 4.3574 | 5 | 1.0138 |
| tape_matmul | 4.5041 | 5 | 4.3432 | 5 | 0.9643 |
| to_tensor | 0.0003 | 5 | 0.0002 | 5 | 0.6667 |
| host_copy | 1.3694 | 5 | 1.2451 | 5 | 0.9092 |
| checksum | 0.7576 | 5 | 0.5389 | 5 | 0.7113 |

## N=2048

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 0.1969 | 5 | 0.4174 | 5 | 2.1199 |
| kernel | 13.5478 | 5 | 13.3574 | 5 | 0.9859 |
| tensor_wrap | 0.0034 | 5 | 0.0033 | 5 | 0.9706 |
| ops_gemm | 18.2343 | 5 | 14.6089 | 5 | 0.8012 |
| tape_matmul | 18.2150 | 5 | 14.3312 | 5 | 0.7868 |
| to_tensor | 0.0002 | 5 | 0.0002 | 5 | 1.0000 |
| host_copy | 5.2209 | 5 | 4.8215 | 5 | 0.9235 |
| checksum | 3.0168 | 5 | 3.0148 | 5 | 0.9993 |

