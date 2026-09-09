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

<!-- raw N=512 phase=alloc_c off_med=0.0077 on_med=0.0073 ratio=0.948051948051948 -->
<!-- raw N=512 phase=kernel off_med=0.5133 on_med=0.5225 ratio=1.017923241768946 -->
<!-- raw N=512 phase=tensor_wrap off_med=0.0002 on_med=0.0002 ratio=1.0 -->
<!-- raw N=512 phase=ops_gemm off_med=0.5078 on_med=0.4995 ratio=0.9836549822764867 -->
<!-- raw N=512 phase=tape_matmul off_med=0.5152 on_med=0.4911 ratio=0.953222049689441 -->
<!-- raw N=512 phase=host_copy off_med=0.0152 on_med=0.0149 ratio=0.9802631578947368 -->
<!-- raw N=512 phase=checksum off_med=0.148 on_med=0.1472 ratio=0.9945945945945946 -->

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

<!-- raw N=1024 phase=alloc_c off_med=0.0218 on_med=0.0209 ratio=0.9587155963302751 -->
<!-- raw N=1024 phase=kernel off_med=2.6477 on_med=2.4859 ratio=0.9388903576689203 -->
<!-- raw N=1024 phase=tensor_wrap off_med=0.0006 on_med=0.0006 ratio=1.0 -->
<!-- raw N=1024 phase=ops_gemm off_med=2.9141 on_med=2.8353 ratio=0.9729590611166399 -->
<!-- raw N=1024 phase=tape_matmul off_med=2.981 on_med=2.7923 ratio=0.9366990942636699 -->
<!-- raw N=1024 phase=to_tensor off_med=0.0001 on_med=0.0002 ratio=2.0 -->
<!-- raw N=1024 phase=host_copy off_med=0.3067 on_med=0.302 ratio=0.9846755787414412 -->
<!-- raw N=1024 phase=checksum off_med=0.6773 on_med=0.6357 ratio=0.9385796545105567 -->

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

<!-- raw N=2048 phase=alloc_c off_med=0.0787 on_med=0.261 ratio=3.3163913595933927 -->
<!-- raw N=2048 phase=kernel off_med=18.1315 on_med=19.109 ratio=1.0539117006314977 -->
<!-- raw N=2048 phase=tensor_wrap off_med=0.0013 on_med=0.0012 ratio=0.923076923076923 -->
<!-- raw N=2048 phase=ops_gemm off_med=21.1289 on_med=20.7005 ratio=0.9797244532370355 -->
<!-- raw N=2048 phase=tape_matmul off_med=20.8645 on_med=21.1605 ratio=1.014186776582233 -->
<!-- raw N=2048 phase=to_tensor off_med=0.0001 on_med=0.0002 ratio=2.0 -->
<!-- raw N=2048 phase=host_copy off_med=1.3778 on_med=1.3334 ratio=0.9677747133110757 -->
<!-- raw N=2048 phase=checksum off_med=2.8437 on_med=2.7401 ratio=0.9635685902169708 -->

