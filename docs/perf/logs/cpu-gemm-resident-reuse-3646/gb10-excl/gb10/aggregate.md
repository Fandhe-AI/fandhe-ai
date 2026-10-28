# 集計（#3646）: gb10（系列種別 main）

checksum: 全 n・全腕・全 run で iter_total の checksum_bits が一致（hard 検査通過）。

## 判定表（5 run 中央値。単位 us）

### N=256

- fresh.iter_total = 184.9, resident_public.iter_total = 196.8, ratio = 1.064
- resident_split: 区間和 170.4 / iter_total 170.7（残差 0.3。記録のみ）
- 判定: 再現（ratio > 1.00）。ただしブリッジ確認が不成立（bridge_status.txt）。bench と診断の窓の一致が未確認のため帰属を保留する（RULE.txt）。

### N=512

- fresh.iter_total = 1181.5, resident_public.iter_total = 1256.5, ratio = 1.064
- resident_split: 区間和 1213.9 / iter_total 1187.0（残差 -26.8。記録のみ）
- 判定: 再現（ratio > 1.00）。ただしブリッジ確認が不成立（bridge_status.txt）。bench と診断の窓の一致が未確認のため帰属を保留する（RULE.txt）。

### N=1024

- fresh.iter_total = 4251.5, resident_public.iter_total = 4341.1, ratio = 1.021
- resident_split: 区間和 4588.5 / iter_total 4418.2（残差 -170.3。記録のみ）
- 判定: 再現（ratio > 1.00）。ただしブリッジ確認が不成立（bridge_status.txt）。bench と診断の窓の一致が未確認のため帰属を保留する（RULE.txt）。

### N=2048

- fresh.iter_total = 16611.3, resident_public.iter_total = 17247.8, ratio = 1.038
- resident_split: 区間和 17223.4 / iter_total 17219.2（残差 -4.2。記録のみ）
- 判定: 再現（ratio > 1.00）。ただしブリッジ確認が不成立（bridge_status.txt）。bench と診断の窓の一致が未確認のため帰属を保留する（RULE.txt）。

### N=4096

- fresh.iter_total = 125350.8, resident_public.iter_total = 150853.5, ratio = 1.203
- resident_split: 区間和 151380.9 / iter_total 151090.7（残差 -290.2。記録のみ）
- 判定: 再現（ratio > 1.00）。G = resident_public - fresh = 25502.6
- gemm_device_hold = 114589.7, +download = 132959.4, drop_in = 121021.5, backend_drop.kernel_call = 114243.3, resident_reupload = 150849.3
- rho_DL = 0.72（H1 支持（download の追加確保＋コピー））
- rho_DROP = 0.25（未達）
- rho_K = 0.01（未達）
- rho_IN = 0.00（未達）
- 帰属: 支持 rho_DL

## 区間表（5 run 中央値 [min–max]、us）

### N=256

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.0 [0.0–0.1] |
| fresh | var_upload | 0.4 [0.2–0.4] |
| fresh | matmul | 146.4 [139.5–164.7] |
| fresh | readout | 0.2 [0.1–0.2] |
| fresh | checksum | 34.0 [33.9–47.0] |
| fresh | iter_total | 184.9 [177.2–209.3] |
| fresh | tape_drop | 0.6 [0.4–0.7] |
| resident_public | matmul | 157.9 [124.5–164.4] |
| resident_public | readout | 0.2 [0.1–0.2] |
| resident_public | checksum | 33.6 [33.6–33.6] |
| resident_public | iter_total | 196.8 [159.3–205.2] |
| resident_reupload | matmul | 177.1 [160.5–231.1] |
| resident_reupload | readout | 0.2 [0.2–0.2] |
| resident_reupload | checksum | 33.6 [33.6–37.1] |
| resident_reupload | iter_total | 211.1 [196.9–276.6] |
| resident_split | gemm_device | 130.1 [122.4–161.3] |
| resident_split | download | 6.4 [5.5–6.5] |
| resident_split | out_drop | 0.2 [0.2–0.2] |
| resident_split | readout | 0.1 [0.1–0.2] |
| resident_split | checksum | 33.6 [33.6–33.6] |
| resident_split | iter_total | 170.7 [163.6–203.2] |
| gemm_device_hold | iter_total | 111.8 [107.5–115.6] |
| gemm_device_drop_in | iter_total | 105.5 [99.7–116.5] |
| gemm_device_download | iter_total | 134.7 [129.8–139.7] |
| backend_drop | kernel_call | 155.6 [142.1–174.2] |
| backend_drop | iter_total | 155.7 [142.2–174.4] |

### N=512

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.1–0.1] |
| fresh | var_upload | 0.6 [0.5–0.6] |
| fresh | matmul | 1009.9 [953.4–1081.4] |
| fresh | readout | 0.3 [0.2–0.4] |
| fresh | checksum | 187.3 [134.8–187.5] |
| fresh | iter_total | 1181.5 [1090.7–1216.3] |
| fresh | tape_drop | 1.1 [0.9–1.1] |
| resident_public | matmul | 1077.4 [923.4–1209.7] |
| resident_public | readout | 0.3 [0.3–0.3] |
| resident_public | checksum | 135.0 [134.9–188.9] |
| resident_public | iter_total | 1256.5 [1100.0–1384.4] |
| resident_reupload | matmul | 1143.2 [1049.2–1245.0] |
| resident_reupload | readout | 0.3 [0.2–0.3] |
| resident_reupload | checksum | 188.7 [134.7–188.9] |
| resident_reupload | iter_total | 1327.6 [1196.5–1387.3] |
| resident_split | gemm_device | 995.5 [966.3–1017.5] |
| resident_split | download | 31.1 [30.9–33.1] |
| resident_split | out_drop | 0.3 [0.3–0.3] |
| resident_split | readout | 0.2 [0.2–0.3] |
| resident_split | checksum | 186.8 [186.8–187.2] |
| resident_split | iter_total | 1187.0 [1161.2–1227.5] |
| gemm_device_hold | iter_total | 857.1 [755.1–958.5] |
| gemm_device_drop_in | iter_total | 887.9 [635.5–965.2] |
| gemm_device_download | iter_total | 937.4 [846.1–1035.3] |
| backend_drop | kernel_call | 1018.5 [889.7–1108.0] |
| backend_drop | iter_total | 1018.9 [889.9–1108.3] |

### N=1024

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.1–0.2] |
| fresh | var_upload | 1.0 [0.6–1.1] |
| fresh | matmul | 3507.0 [3182.0–3703.1] |
| fresh | readout | 0.4 [0.2–0.5] |
| fresh | checksum | 751.6 [548.4–756.9] |
| fresh | iter_total | 4251.5 [3810.2–4335.6] |
| fresh | tape_drop | 1.6 [1.1–2.3] |
| resident_public | matmul | 3582.4 [3420.5–4048.2] |
| resident_public | readout | 0.4 [0.3–0.4] |
| resident_public | checksum | 754.9 [544.1–765.6] |
| resident_public | iter_total | 4341.1 [4123.7–4622.2] |
| resident_reupload | matmul | 3740.0 [3577.2–3933.6] |
| resident_reupload | readout | 0.4 [0.3–0.4] |
| resident_reupload | checksum | 551.5 [539.2–765.8] |
| resident_reupload | iter_total | 4446.4 [4327.1–4501.5] |
| resident_split | gemm_device | 3717.9 [3315.2–3798.3] |
| resident_split | download | 122.3 [117.8–135.6] |
| resident_split | out_drop | 0.4 [0.4–0.5] |
| resident_split | readout | 0.4 [0.4–0.4] |
| resident_split | checksum | 747.5 [542.0–757.5] |
| resident_split | iter_total | 4418.2 [4152.7–4580.3] |
| gemm_device_hold | iter_total | 2911.0 [2624.3–2998.1] |
| gemm_device_drop_in | iter_total | 2965.7 [2810.9–3424.7] |
| gemm_device_download | iter_total | 3166.7 [2930.8–3354.3] |
| backend_drop | kernel_call | 3577.4 [3185.5–3666.0] |
| backend_drop | iter_total | 3578.2 [3186.8–3667.0] |

### N=2048

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.2–0.2] |
| fresh | var_upload | 1.5 [1.5–1.6] |
| fresh | matmul | 13571.1 [13446.9–13703.7] |
| fresh | readout | 0.8 [0.7–0.8] |
| fresh | checksum | 3045.4 [3041.5–3047.1] |
| fresh | iter_total | 16611.3 [16503.0–16715.6] |
| fresh | tape_drop | 2.8 [2.8–2.9] |
| resident_public | matmul | 14224.7 [14177.8–14377.5] |
| resident_public | readout | 0.6 [0.6–0.7] |
| resident_public | checksum | 3007.1 [3006.1–3010.8] |
| resident_public | iter_total | 17247.8 [17181.9–17391.7] |
| resident_reupload | matmul | 14262.1 [14186.8–14433.4] |
| resident_reupload | readout | 0.7 [0.7–0.7] |
| resident_reupload | checksum | 3005.8 [3005.2–3008.2] |
| resident_reupload | iter_total | 17281.9 [17195.2–17461.9] |
| resident_split | gemm_device | 13602.5 [13515.7–13675.1] |
| resident_split | download | 613.3 [603.3–630.8] |
| resident_split | out_drop | 0.7 [0.6–0.7] |
| resident_split | readout | 0.8 [0.8–0.8] |
| resident_split | checksum | 3006.1 [3004.9–3007.1] |
| resident_split | iter_total | 17219.2 [17139.4–17331.5] |
| gemm_device_hold | iter_total | 13399.1 [13180.3–13440.6] |
| gemm_device_drop_in | iter_total | 13312.8 [13193.6–13481.4] |
| gemm_device_download | iter_total | 14126.6 [14039.8–14150.0] |
| backend_drop | kernel_call | 13618.6 [13582.9–13653.7] |
| backend_drop | iter_total | 13620.4 [13585.4–13655.5] |

### N=4096

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.2–0.2] |
| fresh | var_upload | 2.2 [2.1–2.3] |
| fresh | matmul | 113214.9 [113084.1–116535.3] |
| fresh | readout | 0.8 [0.8–0.8] |
| fresh | checksum | 12124.2 [12122.1–12125.6] |
| fresh | iter_total | 125350.8 [125108.3–128668.1] |
| fresh | tape_drop | 6804.4 [6394.8–7430.5] |
| resident_public | matmul | 142237.9 [141554.8–142875.6] |
| resident_public | readout | 1.1 [1.1–1.2] |
| resident_public | checksum | 8710.1 [8622.9–8718.8] |
| resident_public | iter_total | 150853.5 [150192.6–151618.5] |
| resident_reupload | matmul | 142167.0 [141943.8–142823.7] |
| resident_reupload | readout | 1.1 [1.0–1.1] |
| resident_reupload | checksum | 8708.2 [8600.6–8732.6] |
| resident_reupload | iter_total | 150849.3 [150644.0–151483.1] |
| resident_split | gemm_device | 113722.4 [112694.8–115327.3] |
| resident_split | download | 19729.3 [19660.9–19790.9] |
| resident_split | out_drop | 9214.4 [9200.0–9227.4] |
| resident_split | readout | 1.1 [1.0–1.2] |
| resident_split | checksum | 8713.8 [8626.4–8731.3] |
| resident_split | iter_total | 151090.7 [150493.0–152041.2] |
| gemm_device_hold | iter_total | 114589.7 [114142.0–114758.5] |
| gemm_device_drop_in | iter_total | 121021.5 [120182.4–121605.3] |
| gemm_device_download | iter_total | 132959.4 [132723.5–133472.3] |
| backend_drop | kernel_call | 114243.3 [113871.0–114642.3] |
| backend_drop | iter_total | 114245.4 [113873.6–114647.1] |

## run 別 iter_total（fresh / resident_public、us）

| N | arm | run1 | run2 | run3 | run4 | run5 |
|---|---|---|---|---|---|---|
| 256 | fresh | 190.4 | 209.3 | 184.9 | 180.0 | 177.2 |
| 256 | resident_public | 205.2 | 196.8 | 168.2 | 159.3 | 200.6 |
| 512 | fresh | 1181.5 | 1090.7 | 1199.4 | 1178.1 | 1216.3 |
| 512 | resident_public | 1259.1 | 1384.4 | 1100.0 | 1256.5 | 1100.5 |
| 1024 | fresh | 4251.5 | 4259.7 | 3810.2 | 4335.6 | 4186.4 |
| 1024 | resident_public | 4597.9 | 4622.2 | 4123.7 | 4341.1 | 4302.3 |
| 2048 | fresh | 16611.3 | 16567.6 | 16503.0 | 16619.8 | 16715.6 |
| 2048 | resident_public | 17181.9 | 17247.8 | 17281.0 | 17202.9 | 17391.7 |
| 4096 | fresh | 127189.8 | 125108.3 | 125350.8 | 125331.3 | 128668.1 |
| 4096 | resident_public | 151618.5 | 150192.6 | 150351.7 | 151549.8 | 150853.5 |

