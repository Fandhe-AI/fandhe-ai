# Layer B 集計（node=m4max。イシュー #1301）

off runs: ['layerB-m4max-off-run1.log', 'layerB-m4max-off-run2.log', 'layerB-m4max-off-run3.log', 'layerB-m4max-off-run4.log', 'layerB-m4max-off-run5.log']
on runs: ['layerB-m4max-on-run1.log', 'layerB-m4max-on-run2.log', 'layerB-m4max-on-run3.log', 'layerB-m4max-on-run4.log', 'layerB-m4max-on-run5.log']

## N=512

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 0.0096 | 5 | 0.0087 | 5 | 0.9062 |
| kernel | 0.5862 | 5 | 0.5803 | 5 | 0.9899 |
| tensor_wrap | 0.0002 | 5 | 0.0003 | 5 | 1.5000 |
| ops_gemm | 0.5920 | 5 | 0.5811 | 5 | 0.9816 |
| tape_matmul | 0.5849 | 5 | 0.5895 | 5 | 1.0079 |
| to_tensor | 0.0000 | 5 | 0.0000 | 5 | nan |
| host_copy | 0.0150 | 5 | 0.0150 | 5 | 1.0000 |
| checksum | 0.1446 | 5 | 0.1445 | 5 | 0.9993 |

## N=1024

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 0.0186 | 5 | 0.0187 | 5 | 1.0054 |
| kernel | 2.8672 | 5 | 2.7980 | 5 | 0.9759 |
| tensor_wrap | 0.0004 | 5 | 0.0005 | 5 | 1.2500 |
| ops_gemm | 2.9457 | 5 | 2.8949 | 5 | 0.9828 |
| tape_matmul | 2.9765 | 5 | 2.8723 | 5 | 0.9650 |
| to_tensor | 0.0000 | 5 | 0.0000 | 5 | nan |
| host_copy | 0.2758 | 5 | 0.2800 | 5 | 1.0152 |
| checksum | 0.5690 | 5 | 0.5691 | 5 | 1.0002 |

## N=2048

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 0.0709 | 5 | 0.2110 | 5 | 2.9760 |
| kernel | 19.3451 | 5 | 19.8187 | 5 | 1.0245 |
| tensor_wrap | 0.0010 | 5 | 0.0010 | 5 | 1.0000 |
| ops_gemm | 20.1551 | 5 | 21.6416 | 5 | 1.0738 |
| tape_matmul | 20.0149 | 5 | 21.2789 | 5 | 1.0632 |
| to_tensor | 0.0001 | 5 | 0.0001 | 5 | 1.0000 |
| host_copy | 1.1222 | 5 | 1.1577 | 5 | 1.0316 |
| checksum | 2.3691 | 5 | 2.4958 | 5 | 1.0535 |

