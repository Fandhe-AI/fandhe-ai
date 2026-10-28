# 集計（#3646）: gb10-minflt（系列種別 minflt）

**判定対象外の系列: 数値記録のみ（RULE.txt）。帰属判定は行わない。**

checksum: 全 n・全腕・全 run で iter_total の checksum_bits が一致（hard 検査通過）。

## 判定表（1 run 中央値。単位 us）

### N=256

- fresh.iter_total = 357.1, resident_public.iter_total = 221.2, ratio = 0.620
- resident_split: 区間和 231.5 / iter_total 328.2（残差 96.7。記録のみ）

### N=512

- fresh.iter_total = 1228.8, resident_public.iter_total = 1422.1, ratio = 1.157
- resident_split: 区間和 1323.3 / iter_total 1354.5（残差 31.2。記録のみ）

### N=1024

- fresh.iter_total = 4280.2, resident_public.iter_total = 4323.6, ratio = 1.010
- resident_split: 区間和 4794.1 / iter_total 4922.1（残差 128.0。記録のみ）

### N=2048

- fresh.iter_total = 16734.8, resident_public.iter_total = 17222.5, ratio = 1.029
- resident_split: 区間和 17162.5 / iter_total 17212.8（残差 50.3。記録のみ）

### N=4096

- fresh.iter_total = 125874.7, resident_public.iter_total = 151502.1, ratio = 1.204
- resident_split: 区間和 151133.2 / iter_total 151079.7（残差 -53.5。記録のみ）

## minflt / iter（補助・判定外）

| N | arm | phase | minflt/iter |
|---|---|---|---|
| 256 | fresh | tape_build | 0.0 |
| 256 | fresh | var_upload | 0.0 |
| 256 | fresh | matmul | 1.3 |
| 256 | fresh | readout | 0.0 |
| 256 | fresh | checksum | 0.0 |
| 256 | fresh | iter_total | 1.3 |
| 256 | fresh | tape_drop | 0.0 |
| 256 | resident_public | matmul | 0.8 |
| 256 | resident_public | readout | 0.0 |
| 256 | resident_public | checksum | 0.0 |
| 256 | resident_public | iter_total | 0.8 |
| 256 | resident_reupload | matmul | 0.8 |
| 256 | resident_reupload | readout | 0.0 |
| 256 | resident_reupload | checksum | 0.0 |
| 256 | resident_reupload | iter_total | 0.8 |
| 256 | resident_split | gemm_device | 0.6 |
| 256 | resident_split | download | 2.3 |
| 256 | resident_split | out_drop | 0.0 |
| 256 | resident_split | readout | 0.0 |
| 256 | resident_split | checksum | 0.0 |
| 256 | resident_split | iter_total | 2.8 |
| 256 | gemm_device_hold | iter_total | 0.0 |
| 256 | gemm_device_drop_in | iter_total | 0.0 |
| 256 | gemm_device_download | iter_total | 0.0 |
| 256 | backend_drop | kernel_call | 0.0 |
| 256 | backend_drop | iter_total | 0.0 |
| 512 | fresh | tape_build | 0.0 |
| 512 | fresh | var_upload | 0.0 |
| 512 | fresh | matmul | 0.0 |
| 512 | fresh | readout | 0.0 |
| 512 | fresh | checksum | 0.0 |
| 512 | fresh | iter_total | 0.0 |
| 512 | fresh | tape_drop | 0.0 |
| 512 | resident_public | matmul | 0.0 |
| 512 | resident_public | readout | 0.0 |
| 512 | resident_public | checksum | 0.0 |
| 512 | resident_public | iter_total | 0.0 |
| 512 | resident_reupload | matmul | 0.0 |
| 512 | resident_reupload | readout | 0.0 |
| 512 | resident_reupload | checksum | 0.0 |
| 512 | resident_reupload | iter_total | 0.0 |
| 512 | resident_split | gemm_device | 0.0 |
| 512 | resident_split | download | 0.0 |
| 512 | resident_split | out_drop | 0.0 |
| 512 | resident_split | readout | 0.0 |
| 512 | resident_split | checksum | 0.0 |
| 512 | resident_split | iter_total | 0.0 |
| 512 | gemm_device_hold | iter_total | 0.0 |
| 512 | gemm_device_drop_in | iter_total | 0.0 |
| 512 | gemm_device_download | iter_total | 0.0 |
| 512 | backend_drop | kernel_call | 0.0 |
| 512 | backend_drop | iter_total | 0.0 |
| 1024 | fresh | tape_build | 0.0 |
| 1024 | fresh | var_upload | 0.0 |
| 1024 | fresh | matmul | 0.0 |
| 1024 | fresh | readout | 0.0 |
| 1024 | fresh | checksum | 0.0 |
| 1024 | fresh | iter_total | 0.0 |
| 1024 | fresh | tape_drop | 0.0 |
| 1024 | resident_public | matmul | 0.0 |
| 1024 | resident_public | readout | 0.0 |
| 1024 | resident_public | checksum | 0.0 |
| 1024 | resident_public | iter_total | 0.0 |
| 1024 | resident_reupload | matmul | 0.0 |
| 1024 | resident_reupload | readout | 0.0 |
| 1024 | resident_reupload | checksum | 0.0 |
| 1024 | resident_reupload | iter_total | 0.0 |
| 1024 | resident_split | gemm_device | 0.0 |
| 1024 | resident_split | download | 0.0 |
| 1024 | resident_split | out_drop | 0.0 |
| 1024 | resident_split | readout | 0.0 |
| 1024 | resident_split | checksum | 0.0 |
| 1024 | resident_split | iter_total | 0.0 |
| 1024 | gemm_device_hold | iter_total | 0.0 |
| 1024 | gemm_device_drop_in | iter_total | 0.0 |
| 1024 | gemm_device_download | iter_total | 0.0 |
| 1024 | backend_drop | kernel_call | 0.0 |
| 1024 | backend_drop | iter_total | 0.0 |
| 2048 | fresh | tape_build | 0.0 |
| 2048 | fresh | var_upload | 0.0 |
| 2048 | fresh | matmul | 0.0 |
| 2048 | fresh | readout | 0.0 |
| 2048 | fresh | checksum | 0.0 |
| 2048 | fresh | iter_total | 0.0 |
| 2048 | fresh | tape_drop | 0.0 |
| 2048 | resident_public | matmul | 0.0 |
| 2048 | resident_public | readout | 0.0 |
| 2048 | resident_public | checksum | 0.0 |
| 2048 | resident_public | iter_total | 0.0 |
| 2048 | resident_reupload | matmul | 0.0 |
| 2048 | resident_reupload | readout | 0.0 |
| 2048 | resident_reupload | checksum | 0.0 |
| 2048 | resident_reupload | iter_total | 0.0 |
| 2048 | resident_split | gemm_device | 0.0 |
| 2048 | resident_split | download | 0.0 |
| 2048 | resident_split | out_drop | 0.0 |
| 2048 | resident_split | readout | 0.0 |
| 2048 | resident_split | checksum | 0.0 |
| 2048 | resident_split | iter_total | 0.0 |
| 2048 | gemm_device_hold | iter_total | 0.0 |
| 2048 | gemm_device_drop_in | iter_total | 0.0 |
| 2048 | gemm_device_download | iter_total | 0.0 |
| 2048 | backend_drop | kernel_call | 0.0 |
| 2048 | backend_drop | iter_total | 0.0 |
| 4096 | fresh | tape_build | 0.0 |
| 4096 | fresh | var_upload | 0.0 |
| 4096 | fresh | matmul | 37811.2 |
| 4096 | fresh | readout | 0.0 |
| 4096 | fresh | checksum | 0.0 |
| 4096 | fresh | iter_total | 37811.2 |
| 4096 | fresh | tape_drop | 0.0 |
| 4096 | resident_public | matmul | 54468.7 |
| 4096 | resident_public | readout | 0.0 |
| 4096 | resident_public | checksum | 0.0 |
| 4096 | resident_public | iter_total | 54468.7 |
| 4096 | resident_reupload | matmul | 54398.9 |
| 4096 | resident_reupload | readout | 0.0 |
| 4096 | resident_reupload | checksum | 0.0 |
| 4096 | resident_reupload | iter_total | 54398.9 |
| 4096 | resident_split | gemm_device | 37806.8 |
| 4096 | resident_split | download | 16385.0 |
| 4096 | resident_split | out_drop | 0.0 |
| 4096 | resident_split | readout | 0.0 |
| 4096 | resident_split | checksum | 0.0 |
| 4096 | resident_split | iter_total | 54191.8 |
| 4096 | gemm_device_hold | iter_total | 38117.7 |
| 4096 | gemm_device_drop_in | iter_total | 37879.1 |
| 4096 | gemm_device_download | iter_total | 53655.6 |
| 4096 | backend_drop | kernel_call | 38032.7 |
| 4096 | backend_drop | iter_total | 38032.7 |

## 区間表（1 run 中央値 [min–max]、us）

### N=256

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.1–0.1] |
| fresh | var_upload | 0.4 [0.4–0.4] |
| fresh | matmul | 236.3 [236.3–236.3] |
| fresh | readout | 0.2 [0.2–0.2] |
| fresh | checksum | 34.2 [34.2–34.2] |
| fresh | iter_total | 357.1 [357.1–357.1] |
| fresh | tape_drop | 0.7 [0.7–0.7] |
| resident_public | matmul | 150.1 [150.1–150.1] |
| resident_public | readout | 0.2 [0.2–0.2] |
| resident_public | checksum | 33.6 [33.6–33.6] |
| resident_public | iter_total | 221.2 [221.2–221.2] |
| resident_reupload | matmul | 175.0 [175.0–175.0] |
| resident_reupload | readout | 0.2 [0.2–0.2] |
| resident_reupload | checksum | 33.6 [33.6–33.6] |
| resident_reupload | iter_total | 255.4 [255.4–255.4] |
| resident_split | gemm_device | 190.3 [190.3–190.3] |
| resident_split | download | 7.1 [7.1–7.1] |
| resident_split | out_drop | 0.2 [0.2–0.2] |
| resident_split | readout | 0.2 [0.2–0.2] |
| resident_split | checksum | 33.6 [33.6–33.6] |
| resident_split | iter_total | 328.2 [328.2–328.2] |
| gemm_device_hold | iter_total | 126.8 [126.8–126.8] |
| gemm_device_drop_in | iter_total | 139.1 [139.1–139.1] |
| gemm_device_download | iter_total | 152.5 [152.5–152.5] |
| backend_drop | kernel_call | 205.0 [205.0–205.0] |
| backend_drop | iter_total | 222.1 [222.1–222.1] |

### N=512

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.1–0.1] |
| fresh | var_upload | 0.6 [0.6–0.6] |
| fresh | matmul | 956.1 [956.1–956.1] |
| fresh | readout | 0.3 [0.3–0.3] |
| fresh | checksum | 139.2 [139.2–139.2] |
| fresh | iter_total | 1228.8 [1228.8–1228.8] |
| fresh | tape_drop | 1.1 [1.1–1.1] |
| resident_public | matmul | 1165.5 [1165.5–1165.5] |
| resident_public | readout | 0.3 [0.3–0.3] |
| resident_public | checksum | 139.3 [139.3–139.3] |
| resident_public | iter_total | 1422.1 [1422.1–1422.1] |
| resident_reupload | matmul | 1235.4 [1235.4–1235.4] |
| resident_reupload | readout | 0.3 [0.3–0.3] |
| resident_reupload | checksum | 188.9 [188.9–188.9] |
| resident_reupload | iter_total | 1486.3 [1486.3–1486.3] |
| resident_split | gemm_device | 1104.4 [1104.4–1104.4] |
| resident_split | download | 31.5 [31.5–31.5] |
| resident_split | out_drop | 0.3 [0.3–0.3] |
| resident_split | readout | 0.3 [0.3–0.3] |
| resident_split | checksum | 186.8 [186.8–186.8] |
| resident_split | iter_total | 1354.5 [1354.5–1354.5] |
| gemm_device_hold | iter_total | 862.6 [862.6–862.6] |
| gemm_device_drop_in | iter_total | 955.4 [955.4–955.4] |
| gemm_device_download | iter_total | 1037.9 [1037.9–1037.9] |
| backend_drop | kernel_call | 883.8 [883.8–883.8] |
| backend_drop | iter_total | 930.1 [930.1–930.1] |

### N=1024

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.1–0.1] |
| fresh | var_upload | 0.9 [0.9–0.9] |
| fresh | matmul | 3503.7 [3503.7–3503.7] |
| fresh | readout | 0.4 [0.4–0.4] |
| fresh | checksum | 747.9 [747.9–747.9] |
| fresh | iter_total | 4280.2 [4280.2–4280.2] |
| fresh | tape_drop | 1.7 [1.7–1.7] |
| resident_public | matmul | 3651.3 [3651.3–3651.3] |
| resident_public | readout | 0.5 [0.5–0.5] |
| resident_public | checksum | 553.7 [553.7–553.7] |
| resident_public | iter_total | 4323.6 [4323.6–4323.6] |
| resident_reupload | matmul | 4098.7 [4098.7–4098.7] |
| resident_reupload | readout | 0.5 [0.5–0.5] |
| resident_reupload | checksum | 754.8 [754.8–754.8] |
| resident_reupload | iter_total | 4746.0 [4746.0–4746.0] |
| resident_split | gemm_device | 3921.7 [3921.7–3921.7] |
| resident_split | download | 122.7 [122.7–122.7] |
| resident_split | out_drop | 0.6 [0.6–0.6] |
| resident_split | readout | 0.5 [0.5–0.5] |
| resident_split | checksum | 748.6 [748.6–748.6] |
| resident_split | iter_total | 4922.1 [4922.1–4922.1] |
| gemm_device_hold | iter_total | 3001.2 [3001.2–3001.2] |
| gemm_device_drop_in | iter_total | 2916.1 [2916.1–2916.1] |
| gemm_device_download | iter_total | 3198.6 [3198.6–3198.6] |
| backend_drop | kernel_call | 3656.8 [3656.8–3656.8] |
| backend_drop | iter_total | 3742.0 [3742.0–3742.0] |

### N=2048

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.2–0.2] |
| fresh | var_upload | 1.4 [1.4–1.4] |
| fresh | matmul | 13525.3 [13525.3–13525.3] |
| fresh | readout | 0.9 [0.9–0.9] |
| fresh | checksum | 3043.7 [3043.7–3043.7] |
| fresh | iter_total | 16734.8 [16734.8–16734.8] |
| fresh | tape_drop | 3.1 [3.1–3.1] |
| resident_public | matmul | 14168.5 [14168.5–14168.5] |
| resident_public | readout | 0.7 [0.7–0.7] |
| resident_public | checksum | 3004.1 [3004.1–3004.1] |
| resident_public | iter_total | 17222.5 [17222.5–17222.5] |
| resident_reupload | matmul | 14379.3 [14379.3–14379.3] |
| resident_reupload | readout | 0.7 [0.7–0.7] |
| resident_reupload | checksum | 3005.2 [3005.2–3005.2] |
| resident_reupload | iter_total | 17561.7 [17561.7–17561.7] |
| resident_split | gemm_device | 13552.6 [13552.6–13552.6] |
| resident_split | download | 603.3 [603.3–603.3] |
| resident_split | out_drop | 0.8 [0.8–0.8] |
| resident_split | readout | 0.7 [0.7–0.7] |
| resident_split | checksum | 3005.2 [3005.2–3005.2] |
| resident_split | iter_total | 17212.8 [17212.8–17212.8] |
| gemm_device_hold | iter_total | 13448.6 [13448.6–13448.6] |
| gemm_device_drop_in | iter_total | 13348.0 [13348.0–13348.0] |
| gemm_device_download | iter_total | 14048.2 [14048.2–14048.2] |
| backend_drop | kernel_call | 13646.6 [13646.6–13646.6] |
| backend_drop | iter_total | 13759.6 [13759.6–13759.6] |

### N=4096

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.2–0.2] |
| fresh | var_upload | 2.2 [2.2–2.2] |
| fresh | matmul | 113469.0 [113469.0–113469.0] |
| fresh | readout | 0.9 [0.9–0.9] |
| fresh | checksum | 12123.1 [12123.1–12123.1] |
| fresh | iter_total | 125874.7 [125874.7–125874.7] |
| fresh | tape_drop | 6602.9 [6602.9–6602.9] |
| resident_public | matmul | 142633.7 [142633.7–142633.7] |
| resident_public | readout | 0.8 [0.8–0.8] |
| resident_public | checksum | 8735.4 [8735.4–8735.4] |
| resident_public | iter_total | 151502.1 [151502.1–151502.1] |
| resident_reupload | matmul | 142582.2 [142582.2–142582.2] |
| resident_reupload | readout | 0.7 [0.7–0.7] |
| resident_reupload | checksum | 8634.6 [8634.6–8634.6] |
| resident_reupload | iter_total | 151380.8 [151380.8–151380.8] |
| resident_split | gemm_device | 113428.1 [113428.1–113428.1] |
| resident_split | download | 19753.2 [19753.2–19753.2] |
| resident_split | out_drop | 9221.8 [9221.8–9221.8] |
| resident_split | readout | 0.8 [0.8–0.8] |
| resident_split | checksum | 8729.3 [8729.3–8729.3] |
| resident_split | iter_total | 151079.7 [151079.7–151079.7] |
| gemm_device_hold | iter_total | 114812.4 [114812.4–114812.4] |
| gemm_device_drop_in | iter_total | 121292.0 [121292.0–121292.0] |
| gemm_device_download | iter_total | 133705.8 [133705.8–133705.8] |
| backend_drop | kernel_call | 114341.1 [114341.1–114341.1] |
| backend_drop | iter_total | 114408.8 [114408.8–114408.8] |

## run 別 iter_total（fresh / resident_public、us）

| N | arm | run1 |
|---|---|---|
| 256 | fresh | 357.1 |
| 256 | resident_public | 221.2 |
| 512 | fresh | 1228.8 |
| 512 | resident_public | 1422.1 |
| 1024 | fresh | 4280.2 |
| 1024 | resident_public | 4323.6 |
| 2048 | fresh | 16734.8 |
| 2048 | resident_public | 17222.5 |
| 4096 | fresh | 125874.7 |
| 4096 | resident_public | 151502.1 |

