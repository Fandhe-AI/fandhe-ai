# 集計（#3646）: gb10-taskset（系列種別 taskset）

**判定対象外の系列: 数値記録のみ（RULE.txt）。帰属判定は行わない。**

checksum: 全 n・全腕・全 run で iter_total の checksum_bits が一致（hard 検査通過）。

## 判定表（5 run 中央値。単位 us）

### N=256

- fresh.iter_total = 138.9, resident_public.iter_total = 131.9, ratio = 0.950
- resident_split: 区間和 129.7 / iter_total 130.6（残差 0.9。記録のみ）

### N=512

- fresh.iter_total = 953.7, resident_public.iter_total = 989.2, ratio = 1.037
- resident_split: 区間和 1030.3 / iter_total 1034.2（残差 3.9。記録のみ）

### N=1024

- fresh.iter_total = 3339.5, resident_public.iter_total = 3472.4, ratio = 1.040
- resident_split: 区間和 3504.3 / iter_total 3507.6（残差 3.3。記録のみ）

### N=2048

- fresh.iter_total = 18516.0, resident_public.iter_total = 19143.7, ratio = 1.034
- resident_split: 区間和 19158.1 / iter_total 19162.9（残差 4.7。記録のみ）

### N=4096

- fresh.iter_total = 150833.0, resident_public.iter_total = 181209.7, ratio = 1.201
- resident_split: 区間和 181451.8 / iter_total 181344.0（残差 -107.8。記録のみ）

## 区間表（5 run 中央値 [min–max]、us）

### N=256

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.0 [0.0–0.0] |
| fresh | var_upload | 0.2 [0.2–0.2] |
| fresh | matmul | 103.8 [97.0–104.4] |
| fresh | readout | 0.1 [0.1–0.2] |
| fresh | checksum | 33.9 [33.8–34.0] |
| fresh | iter_total | 138.9 [131.9–139.5] |
| fresh | tape_drop | 0.4 [0.4–0.4] |
| resident_public | matmul | 98.0 [93.5–101.2] |
| resident_public | readout | 0.1 [0.1–0.1] |
| resident_public | checksum | 33.6 [33.6–33.6] |
| resident_public | iter_total | 131.9 [127.7–135.3] |
| resident_reupload | matmul | 119.3 [112.3–125.3] |
| resident_reupload | readout | 0.1 [0.1–0.1] |
| resident_reupload | checksum | 33.6 [33.6–33.6] |
| resident_reupload | iter_total | 153.5 [146.5–159.3] |
| resident_split | gemm_device | 90.0 [88.2–92.2] |
| resident_split | download | 5.9 [5.4–6.2] |
| resident_split | out_drop | 0.2 [0.2–0.2] |
| resident_split | readout | 0.1 [0.1–0.1] |
| resident_split | checksum | 33.6 [33.6–33.6] |
| resident_split | iter_total | 130.6 [130.0–132.1] |
| gemm_device_hold | iter_total | 86.6 [85.0–87.8] |
| gemm_device_drop_in | iter_total | 87.1 [86.7–88.9] |
| gemm_device_download | iter_total | 100.7 [99.3–103.5] |
| backend_drop | kernel_call | 102.0 [94.4–131.5] |
| backend_drop | iter_total | 102.1 [94.5–131.6] |

### N=512

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.0 [0.0–0.1] |
| fresh | var_upload | 0.5 [0.4–0.5] |
| fresh | matmul | 815.7 [677.5–843.4] |
| fresh | readout | 0.2 [0.2–0.2] |
| fresh | checksum | 134.8 [134.8–134.8] |
| fresh | iter_total | 953.7 [814.5–979.9] |
| fresh | tape_drop | 0.7 [0.7–0.8] |
| resident_public | matmul | 854.9 [809.2–871.4] |
| resident_public | readout | 0.3 [0.3–0.3] |
| resident_public | checksum | 134.6 [134.6–134.8] |
| resident_public | iter_total | 989.2 [944.9–1007.0] |
| resident_reupload | matmul | 904.2 [824.2–1002.0] |
| resident_reupload | readout | 0.3 [0.3–0.3] |
| resident_reupload | checksum | 134.7 [134.6–134.8] |
| resident_reupload | iter_total | 1039.6 [960.3–1142.2] |
| resident_split | gemm_device | 869.7 [758.2–933.1] |
| resident_split | download | 25.5 [24.0–26.9] |
| resident_split | out_drop | 0.2 [0.2–0.2] |
| resident_split | readout | 0.3 [0.3–0.3] |
| resident_split | checksum | 134.6 [134.6–134.9] |
| resident_split | iter_total | 1034.2 [933.9–1097.2] |
| gemm_device_hold | iter_total | 351.5 [347.0–455.2] |
| gemm_device_drop_in | iter_total | 352.1 [340.6–486.2] |
| gemm_device_download | iter_total | 701.2 [536.3–772.5] |
| backend_drop | kernel_call | 807.9 [588.0–934.1] |
| backend_drop | iter_total | 808.2 [588.2–935.3] |

### N=1024

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.0–0.1] |
| fresh | var_upload | 0.7 [0.6–0.8] |
| fresh | matmul | 2789.6 [2612.7–2880.2] |
| fresh | readout | 0.3 [0.2–0.3] |
| fresh | checksum | 542.0 [540.7–543.5] |
| fresh | iter_total | 3339.5 [3156.9–3443.1] |
| fresh | tape_drop | 1.1 [1.0–1.3] |
| resident_public | matmul | 2927.2 [2796.8–3068.0] |
| resident_public | readout | 0.4 [0.3–0.5] |
| resident_public | checksum | 544.9 [542.6–546.0] |
| resident_public | iter_total | 3472.4 [3337.8–3615.5] |
| resident_reupload | matmul | 3095.0 [3039.7–3161.0] |
| resident_reupload | readout | 0.5 [0.5–0.6] |
| resident_reupload | checksum | 547.3 [546.1–550.2] |
| resident_reupload | iter_total | 3641.2 [3594.5–3702.3] |
| resident_split | gemm_device | 2843.2 [2802.6–2844.8] |
| resident_split | download | 114.2 [107.4–118.3] |
| resident_split | out_drop | 0.3 [0.3–0.3] |
| resident_split | readout | 0.4 [0.4–0.5] |
| resident_split | checksum | 546.2 [543.6–548.7] |
| resident_split | iter_total | 3507.6 [3469.6–3528.1] |
| gemm_device_hold | iter_total | 2614.4 [2543.2–2637.5] |
| gemm_device_drop_in | iter_total | 2529.4 [2506.8–2727.5] |
| gemm_device_download | iter_total | 2769.7 [2693.9–2968.0] |
| backend_drop | kernel_call | 2735.7 [2679.4–2806.9] |
| backend_drop | iter_total | 2736.3 [2679.9–2807.6] |

### N=2048

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.1–0.2] |
| fresh | var_upload | 1.3 [1.2–1.3] |
| fresh | matmul | 16312.5 [16274.3–16436.0] |
| fresh | readout | 0.6 [0.6–0.6] |
| fresh | checksum | 2207.4 [2193.2–2209.7] |
| fresh | iter_total | 18516.0 [18474.7–18642.0] |
| fresh | tape_drop | 2.2 [2.0–2.2] |
| resident_public | matmul | 16933.5 [16897.8–17038.9] |
| resident_public | readout | 0.8 [0.7–0.8] |
| resident_public | checksum | 2203.1 [2182.9–2204.3] |
| resident_public | iter_total | 19143.7 [19091.3–19237.1] |
| resident_reupload | matmul | 16973.4 [16943.0–17070.8] |
| resident_reupload | readout | 0.7 [0.7–0.8] |
| resident_reupload | checksum | 2195.7 [2189.3–2202.2] |
| resident_reupload | iter_total | 19161.4 [19148.5–19272.9] |
| resident_split | gemm_device | 16328.0 [16310.8–16406.8] |
| resident_split | download | 629.3 [555.3–636.9] |
| resident_split | out_drop | 0.5 [0.5–0.6] |
| resident_split | readout | 0.7 [0.7–0.7] |
| resident_split | checksum | 2199.6 [2183.5–2203.2] |
| resident_split | iter_total | 19162.9 [19102.9–19228.2] |
| gemm_device_hold | iter_total | 16126.9 [16109.7–16254.5] |
| gemm_device_drop_in | iter_total | 16198.0 [16123.4–16257.1] |
| gemm_device_download | iter_total | 16917.3 [16873.2–16969.9] |
| backend_drop | kernel_call | 16419.7 [16294.0–16460.9] |
| backend_drop | iter_total | 16422.3 [16295.6–16462.5] |

### N=4096

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.1–0.3] |
| fresh | var_upload | 1.8 [1.5–1.9] |
| fresh | matmul | 141964.8 [141422.5–142953.7] |
| fresh | readout | 0.6 [0.6–0.6] |
| fresh | checksum | 8844.4 [8772.8–8846.3] |
| fresh | iter_total | 150833.0 [150266.4–151831.6] |
| fresh | tape_drop | 10750.3 [10520.8–10788.8] |
| resident_public | matmul | 172526.8 [171931.5–172832.6] |
| resident_public | readout | 1.1 [1.0–1.2] |
| resident_public | checksum | 8785.8 [8679.9–8800.8] |
| resident_public | iter_total | 181209.7 [180769.6–181533.4] |
| resident_reupload | matmul | 172760.5 [172361.1–173253.5] |
| resident_reupload | readout | 1.0 [1.0–1.0] |
| resident_reupload | checksum | 8776.6 [8668.2–8792.4] |
| resident_reupload | iter_total | 181605.8 [181128.3–181975.7] |
| resident_split | gemm_device | 142312.0 [142103.7–142478.0] |
| resident_split | download | 21100.7 [20739.8–21724.0] |
| resident_split | out_drop | 9332.6 [9281.5–9339.8] |
| resident_split | readout | 1.0 [0.9–1.1] |
| resident_split | checksum | 8705.5 [8679.4–8811.3] |
| resident_split | iter_total | 181344.0 [181147.4–181759.3] |
| gemm_device_hold | iter_total | 141698.2 [140726.9–142616.5] |
| gemm_device_drop_in | iter_total | 152822.7 [151995.7–153516.5] |
| gemm_device_download | iter_total | 163322.7 [162808.1–164157.2] |
| backend_drop | kernel_call | 141803.3 [141372.7–142423.7] |
| backend_drop | iter_total | 141805.0 [141374.7–142426.7] |

## run 別 iter_total（fresh / resident_public、us）

| N | arm | run1 | run2 | run3 | run4 | run5 |
|---|---|---|---|---|---|---|
| 256 | fresh | 139.0 | 136.5 | 139.5 | 131.9 | 138.9 |
| 256 | resident_public | 133.6 | 135.3 | 131.6 | 127.7 | 131.9 |
| 512 | fresh | 956.4 | 912.4 | 814.5 | 979.9 | 953.7 |
| 512 | resident_public | 1007.0 | 1005.0 | 989.2 | 973.3 | 944.9 |
| 1024 | fresh | 3341.1 | 3156.9 | 3339.5 | 3330.1 | 3443.1 |
| 1024 | resident_public | 3615.5 | 3472.4 | 3450.3 | 3493.4 | 3337.8 |
| 2048 | fresh | 18535.4 | 18474.7 | 18514.4 | 18642.0 | 18516.0 |
| 2048 | resident_public | 19237.1 | 19143.7 | 19091.3 | 19100.9 | 19159.7 |
| 4096 | fresh | 150833.0 | 150266.4 | 151831.6 | 150408.9 | 150978.5 |
| 4096 | resident_public | 180769.6 | 181533.4 | 181209.7 | 181203.4 | 181422.7 |

