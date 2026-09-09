# Layer B 集計（node=m4max。イシュー #1481（#1301 と同型を再利用））

off runs: ['layerB-m4max-off-run1.log', 'layerB-m4max-off-run2.log', 'layerB-m4max-off-run3.log', 'layerB-m4max-off-run4.log', 'layerB-m4max-off-run5.log']
on runs: ['layerB-m4max-on-run1.log', 'layerB-m4max-on-run2.log', 'layerB-m4max-on-run3.log', 'layerB-m4max-on-run4.log', 'layerB-m4max-on-run5.log']

## N=512

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 0.0077 | 5 | 0.0073 | 5 | 0.9481 |
| kernel | 0.5133 | 5 | 0.5225 | 5 | 1.0179 |
| tensor_wrap | 0.0002 | 5 | 0.0002 | 5 | 1.0000 |
| ops_gemm | 0.5078 | 5 | 0.4995 | 5 | 0.9837 |
| tape_matmul | 0.5152 | 5 | 0.4911 | 5 | 0.9532 |
| to_tensor | 0.0000 | 5 | 0.0000 | 5 | nan |
| host_copy | 0.0152 | 5 | 0.0149 | 5 | 0.9803 |
| checksum | 0.1480 | 5 | 0.1472 | 5 | 0.9946 |

## N=1024

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 0.0218 | 5 | 0.0209 | 5 | 0.9587 |
| kernel | 2.6477 | 5 | 2.4859 | 5 | 0.9389 |
| tensor_wrap | 0.0006 | 5 | 0.0006 | 5 | 1.0000 |
| ops_gemm | 2.9141 | 5 | 2.8353 | 5 | 0.9730 |
| tape_matmul | 2.9810 | 5 | 2.7923 | 5 | 0.9367 |
| to_tensor | 0.0001 | 5 | 0.0002 | 5 | 2.0000 |
| host_copy | 0.3067 | 5 | 0.3020 | 5 | 0.9847 |
| checksum | 0.6773 | 5 | 0.6357 | 5 | 0.9386 |

## N=2048

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 0.0787 | 5 | 0.2610 | 5 | 3.3164 |
| kernel | 18.1315 | 5 | 19.1090 | 5 | 1.0539 |
| tensor_wrap | 0.0013 | 5 | 0.0012 | 5 | 0.9231 |
| ops_gemm | 21.1289 | 5 | 20.7005 | 5 | 0.9797 |
| tape_matmul | 20.8645 | 5 | 21.1605 | 5 | 1.0142 |
| to_tensor | 0.0001 | 5 | 0.0002 | 5 | 2.0000 |
| host_copy | 1.3778 | 5 | 1.3334 | 5 | 0.9678 |
| checksum | 2.8437 | 5 | 2.7401 | 5 | 0.9636 |

