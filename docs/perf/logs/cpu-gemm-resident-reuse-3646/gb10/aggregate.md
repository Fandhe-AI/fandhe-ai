# 集計（#3646）: gb10（系列種別 main）

checksum: 全 n・全腕・全 run で iter_total の checksum_bits が一致（hard 検査通過）。

## 判定表（5 run 中央値。単位 us）

### N=256

- fresh.iter_total = 167.9, resident_public.iter_total = 162.9, ratio = 0.970
- resident_split: 区間和 162.1 / iter_total 162.8（残差 0.7。記録のみ）
- 判定: 非再現（ratio <= 1.00）。帰属は行わない。

### N=512

- fresh.iter_total = 787.8, resident_public.iter_total = 831.0, ratio = 1.055
- resident_split: 区間和 773.7 / iter_total 799.8（残差 26.1。記録のみ）
- 判定: 再現（ratio > 1.00）。ただしブリッジ確認が不成立（bridge_status.txt）。bench と診断の窓の一致が未確認のため帰属を保留する（RULE.txt）。

### N=1024

- fresh.iter_total = 3054.7, resident_public.iter_total = 3208.3, ratio = 1.050
- resident_split: 区間和 3220.0 / iter_total 3125.6（残差 -94.3。記録のみ）
- 判定: 再現（ratio > 1.00）。ただしブリッジ確認が不成立（bridge_status.txt）。bench と診断の窓の一致が未確認のため帰属を保留する（RULE.txt）。

### N=2048

- fresh.iter_total = 15951.8, resident_public.iter_total = 16583.5, ratio = 1.040
- resident_split: 区間和 16664.3 / iter_total 16729.5（残差 65.2。記録のみ）
- 判定: 再現（ratio > 1.00）。ただしブリッジ確認が不成立（bridge_status.txt）。bench と診断の窓の一致が未確認のため帰属を保留する（RULE.txt）。

### N=4096

- fresh.iter_total = 124751.6, resident_public.iter_total = 151388.7, ratio = 1.214
- resident_split: 区間和 151994.6 / iter_total 152381.6（残差 387.0。記録のみ）
- 判定: 再現（ratio > 1.00）。G = resident_public - fresh = 26637.1
- gemm_device_hold = 112978.4, +download = 133844.6, drop_in = 122616.1, backend_drop.kernel_call = 113024.4, resident_reupload = 151973.6
- rho_DL = 0.78（H1 支持（download の追加確保＋コピー））
- rho_DROP = 0.36（未達）
- rho_K = -0.00（未達）
- rho_IN = 0.02（未達）
- 帰属: 支持 rho_DL

## 区間表（5 run 中央値 [min–max]、us）

### N=256

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.0 [0.0–0.1] |
| fresh | var_upload | 0.3 [0.2–0.3] |
| fresh | matmul | 133.0 [127.8–136.3] |
| fresh | readout | 0.2 [0.1–0.2] |
| fresh | checksum | 33.9 [33.8–34.1] |
| fresh | iter_total | 167.9 [162.2–174.6] |
| fresh | tape_drop | 0.5 [0.5–0.6] |
| resident_public | matmul | 128.8 [122.3–137.0] |
| resident_public | readout | 0.1 [0.1–0.2] |
| resident_public | checksum | 33.6 [33.6–33.6] |
| resident_public | iter_total | 162.9 [156.3–175.5] |
| resident_reupload | matmul | 149.1 [142.7–168.8] |
| resident_reupload | readout | 0.2 [0.2–0.2] |
| resident_reupload | checksum | 33.6 [33.6–33.6] |
| resident_reupload | iter_total | 184.2 [180.3–205.9] |
| resident_split | gemm_device | 122.3 [116.8–131.5] |
| resident_split | download | 5.9 [5.9–8.5] |
| resident_split | out_drop | 0.2 [0.2–0.2] |
| resident_split | readout | 0.1 [0.1–0.2] |
| resident_split | checksum | 33.6 [33.6–46.7] |
| resident_split | iter_total | 162.8 [157.6–183.0] |
| gemm_device_hold | iter_total | 107.4 [103.0–111.2] |
| gemm_device_drop_in | iter_total | 108.8 [100.3–114.9] |
| gemm_device_download | iter_total | 128.9 [123.1–135.7] |
| backend_drop | kernel_call | 132.5 [127.6–142.3] |
| backend_drop | iter_total | 132.6 [127.6–142.4] |

### N=512

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.0–0.1] |
| fresh | var_upload | 0.5 [0.4–0.6] |
| fresh | matmul | 634.8 [567.8–684.6] |
| fresh | readout | 0.2 [0.2–0.3] |
| fresh | checksum | 134.9 [134.7–188.2] |
| fresh | iter_total | 787.8 [745.9–848.6] |
| fresh | tape_drop | 0.8 [0.8–1.0] |
| resident_public | matmul | 669.1 [578.7–705.1] |
| resident_public | readout | 0.3 [0.2–0.3] |
| resident_public | checksum | 137.4 [134.6–189.0] |
| resident_public | iter_total | 831.0 [753.1–863.5] |
| resident_reupload | matmul | 687.6 [652.1–692.2] |
| resident_reupload | readout | 0.3 [0.2–0.3] |
| resident_reupload | checksum | 138.4 [134.7–189.1] |
| resident_reupload | iter_total | 838.3 [813.1–878.1] |
| resident_split | gemm_device | 605.8 [529.3–648.4] |
| resident_split | download | 30.5 [26.1–31.6] |
| resident_split | out_drop | 0.3 [0.3–0.3] |
| resident_split | readout | 0.2 [0.2–0.3] |
| resident_split | checksum | 136.9 [134.6–186.8] |
| resident_split | iter_total | 799.8 [708.3–855.6] |
| gemm_device_hold | iter_total | 504.8 [455.1–557.0] |
| gemm_device_drop_in | iter_total | 513.1 [427.2–555.4] |
| gemm_device_download | iter_total | 546.9 [523.7–566.2] |
| backend_drop | kernel_call | 623.7 [552.8–655.0] |
| backend_drop | iter_total | 624.0 [553.0–655.4] |

### N=1024

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.1–0.2] |
| fresh | var_upload | 0.9 [0.7–1.0] |
| fresh | matmul | 2362.0 [2335.7–2433.7] |
| fresh | readout | 0.3 [0.2–0.4] |
| fresh | checksum | 546.2 [543.6–755.7] |
| fresh | iter_total | 3054.7 [2979.2–3131.3] |
| fresh | tape_drop | 1.3 [1.1–1.6] |
| resident_public | matmul | 2482.8 [2456.0–2646.1] |
| resident_public | readout | 0.5 [0.4–0.5] |
| resident_public | checksum | 760.1 [546.6–762.7] |
| resident_public | iter_total | 3208.3 [3077.9–3313.5] |
| resident_reupload | matmul | 2742.5 [2635.3–3010.0] |
| resident_reupload | readout | 0.4 [0.4–0.5] |
| resident_reupload | checksum | 755.3 [553.2–764.0] |
| resident_reupload | iter_total | 3445.3 [3305.8–3723.2] |
| resident_split | gemm_device | 2354.0 [2232.6–2403.2] |
| resident_split | download | 118.0 [110.6–132.6] |
| resident_split | out_drop | 0.4 [0.3–0.4] |
| resident_split | readout | 0.4 [0.3–0.5] |
| resident_split | checksum | 747.3 [550.6–749.6] |
| resident_split | iter_total | 3125.6 [3023.1–3277.9] |
| gemm_device_hold | iter_total | 2049.5 [2034.9–2075.8] |
| gemm_device_drop_in | iter_total | 2092.3 [2007.9–2153.2] |
| gemm_device_download | iter_total | 2204.9 [2155.9–2388.3] |
| backend_drop | kernel_call | 2308.3 [2222.4–2413.3] |
| backend_drop | iter_total | 2308.5 [2223.2–2413.8] |

### N=2048

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.2–0.2] |
| fresh | var_upload | 1.5 [1.5–1.6] |
| fresh | matmul | 12899.9 [12823.7–13055.8] |
| fresh | readout | 0.9 [0.9–0.9] |
| fresh | checksum | 3042.2 [3041.1–3042.6] |
| fresh | iter_total | 15951.8 [15868.2–16092.7] |
| fresh | tape_drop | 2.9 [2.9–3.0] |
| resident_public | matmul | 13570.0 [13484.0–13639.8] |
| resident_public | readout | 0.8 [0.7–0.8] |
| resident_public | checksum | 3010.0 [3009.5–3012.0] |
| resident_public | iter_total | 16583.5 [16490.5–16658.0] |
| resident_reupload | matmul | 13749.0 [13673.5–13774.1] |
| resident_reupload | readout | 0.8 [0.8–0.8] |
| resident_reupload | checksum | 3011.0 [3010.2–3011.3] |
| resident_reupload | iter_total | 16765.9 [16683.3–16792.9] |
| resident_split | gemm_device | 13053.2 [12974.2–13204.0] |
| resident_split | download | 599.5 [594.6–625.9] |
| resident_split | out_drop | 0.7 [0.6–0.7] |
| resident_split | readout | 0.8 [0.8–0.8] |
| resident_split | checksum | 3010.1 [3008.2–3012.3] |
| resident_split | iter_total | 16729.5 [16606.5–16823.3] |
| gemm_device_hold | iter_total | 12878.0 [12768.7–12895.1] |
| gemm_device_drop_in | iter_total | 12853.8 [12760.2–12904.0] |
| gemm_device_download | iter_total | 13407.4 [13270.3–13514.3] |
| backend_drop | kernel_call | 12929.7 [12867.6–12951.2] |
| backend_drop | iter_total | 12932.1 [12869.6–12952.8] |

### N=4096

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.3 [0.2–0.3] |
| fresh | var_upload | 2.2 [2.0–2.3] |
| fresh | matmul | 113003.8 [112538.7–113804.3] |
| fresh | readout | 0.9 [0.9–0.9] |
| fresh | checksum | 12134.7 [12131.3–12137.7] |
| fresh | iter_total | 124751.6 [124257.7–125811.9] |
| fresh | tape_drop | 8211.3 [7888.5–8677.3] |
| resident_public | matmul | 142572.6 [142199.7–144045.2] |
| resident_public | readout | 1.2 [1.1–1.2] |
| resident_public | checksum | 8779.6 [8693.7–8800.5] |
| resident_public | iter_total | 151388.7 [150872.9–152872.8] |
| resident_reupload | matmul | 143257.5 [142485.9–143773.1] |
| resident_reupload | readout | 1.1 [1.1–1.2] |
| resident_reupload | checksum | 8719.6 [8692.6–8798.0] |
| resident_reupload | iter_total | 151973.6 [151249.8–152596.0] |
| resident_split | gemm_device | 113010.8 [112004.9–114796.4] |
| resident_split | download | 20928.4 [20764.8–21383.6] |
| resident_split | out_drop | 9319.3 [9263.1–9323.9] |
| resident_split | readout | 1.1 [1.0–1.1] |
| resident_split | checksum | 8735.1 [8691.6–8771.1] |
| resident_split | iter_total | 152381.6 [150616.6–153586.8] |
| gemm_device_hold | iter_total | 112978.4 [111996.2–114261.0] |
| gemm_device_drop_in | iter_total | 122616.1 [120010.4–123344.7] |
| gemm_device_download | iter_total | 133844.6 [133454.6–133982.0] |
| backend_drop | kernel_call | 113024.4 [112455.1–114176.7] |
| backend_drop | iter_total | 113026.2 [112457.2–114178.6] |

## run 別 iter_total（fresh / resident_public、us）

| N | arm | run1 | run2 | run3 | run4 | run5 |
|---|---|---|---|---|---|---|
| 256 | fresh | 169.6 | 162.2 | 174.6 | 167.9 | 166.6 |
| 256 | resident_public | 162.9 | 175.5 | 162.2 | 156.3 | 164.3 |
| 512 | fresh | 750.6 | 745.9 | 787.8 | 848.6 | 815.9 |
| 512 | resident_public | 845.3 | 753.1 | 863.5 | 831.0 | 787.6 |
| 1024 | fresh | 3111.0 | 3027.0 | 3054.7 | 3131.3 | 2979.2 |
| 1024 | resident_public | 3077.9 | 3313.5 | 3208.3 | 3276.7 | 3199.1 |
| 2048 | fresh | 15868.2 | 15921.9 | 16092.7 | 16033.8 | 15951.8 |
| 2048 | resident_public | 16490.5 | 16605.7 | 16658.0 | 16528.1 | 16583.5 |
| 4096 | fresh | 124257.7 | 124671.5 | 125811.9 | 124821.0 | 124751.6 |
| 4096 | resident_public | 150872.9 | 151276.5 | 152872.8 | 151388.7 | 152176.8 |

