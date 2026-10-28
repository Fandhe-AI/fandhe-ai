# 集計（#3646）: gb10-tunables（系列種別 tunables）

**判定対象外の系列: 数値記録のみ（RULE.txt）。帰属判定は行わない。**

checksum: 全 n・全腕・全 run で iter_total の checksum_bits が一致（hard 検査通過）。

## 判定表（5 run 中央値。単位 us）

### N=256

- fresh.iter_total = 182.3, resident_public.iter_total = 154.3, ratio = 0.847
- resident_split: 区間和 174.9 / iter_total 183.5（残差 8.6。記録のみ）
- tunables 解釈: ratio <= 1.05。機構は glibc の確保とページ返却を支持（RULE.txt）。

### N=512

- fresh.iter_total = 1130.4, resident_public.iter_total = 1236.6, ratio = 1.094
- resident_split: 区間和 1265.0 / iter_total 1206.4（残差 -58.6。記録のみ）
- tunables 解釈: ratio > 1.05。glibc 確保機構だけでは説明できない。

### N=1024

- fresh.iter_total = 4373.4, resident_public.iter_total = 4473.4, ratio = 1.023
- resident_split: 区間和 4370.7 / iter_total 4403.1（残差 32.4。記録のみ）
- tunables 解釈: ratio <= 1.05。機構は glibc の確保とページ返却を支持（RULE.txt）。

### N=2048

- fresh.iter_total = 20643.0, resident_public.iter_total = 25673.0, ratio = 1.244
- resident_split: 区間和 25426.0 / iter_total 25614.5（残差 188.5。記録のみ）
- tunables 解釈: ratio > 1.05。glibc 確保機構だけでは説明できない。

### N=4096

- fresh.iter_total = 126097.6, resident_public.iter_total = 150808.9, ratio = 1.196
- resident_split: 区間和 151210.5 / iter_total 151465.2（残差 254.6。記録のみ）
- tunables 解釈: ratio > 1.05。glibc 確保機構だけでは説明できない。

## 区間表（5 run 中央値 [min–max]、us）

### N=256

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.0 [0.0–0.1] |
| fresh | var_upload | 0.3 [0.2–0.4] |
| fresh | matmul | 144.8 [138.3–150.3] |
| fresh | readout | 0.2 [0.2–0.2] |
| fresh | checksum | 34.0 [33.9–35.4] |
| fresh | iter_total | 182.3 [179.2–187.8] |
| fresh | tape_drop | 0.7 [0.7–0.8] |
| resident_public | matmul | 120.5 [116.0–139.3] |
| resident_public | readout | 0.2 [0.2–0.2] |
| resident_public | checksum | 33.6 [33.6–33.6] |
| resident_public | iter_total | 154.3 [150.1–184.9] |
| resident_reupload | matmul | 173.8 [160.9–322.3] |
| resident_reupload | readout | 0.2 [0.2–0.3] |
| resident_reupload | checksum | 33.6 [33.6–33.7] |
| resident_reupload | iter_total | 210.4 [194.9–356.8] |
| resident_split | gemm_device | 134.9 [119.5–144.6] |
| resident_split | download | 6.1 [6.0–7.3] |
| resident_split | out_drop | 0.2 [0.2–0.2] |
| resident_split | readout | 0.2 [0.1–0.2] |
| resident_split | checksum | 33.6 [33.6–33.6] |
| resident_split | iter_total | 183.5 [159.3–187.2] |
| gemm_device_hold | iter_total | 102.0 [94.5–112.2] |
| gemm_device_drop_in | iter_total | 96.6 [90.8–113.7] |
| gemm_device_download | iter_total | 123.3 [116.9–128.7] |
| backend_drop | kernel_call | 165.1 [129.0–193.5] |
| backend_drop | iter_total | 165.4 [129.1–193.8] |

### N=512

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.1–0.1] |
| fresh | var_upload | 0.6 [0.5–0.6] |
| fresh | matmul | 950.6 [900.8–1088.1] |
| fresh | readout | 0.3 [0.2–0.3] |
| fresh | checksum | 187.2 [139.1–187.6] |
| fresh | iter_total | 1130.4 [1071.9–1235.2] |
| fresh | tape_drop | 1.0 [1.0–1.1] |
| resident_public | matmul | 1065.2 [958.7–1285.5] |
| resident_public | readout | 0.3 [0.3–0.4] |
| resident_public | checksum | 188.7 [134.8–188.9] |
| resident_public | iter_total | 1236.6 [1147.3–1426.0] |
| resident_reupload | matmul | 1129.5 [934.6–1182.8] |
| resident_reupload | readout | 0.4 [0.3–0.4] |
| resident_reupload | checksum | 136.4 [134.6–189.5] |
| resident_reupload | iter_total | 1291.8 [1107.4–1364.6] |
| resident_split | gemm_device | 1047.2 [959.4–1162.7] |
| resident_split | download | 30.5 [25.0–34.1] |
| resident_split | out_drop | 0.2 [0.2–0.3] |
| resident_split | readout | 0.4 [0.3–0.4] |
| resident_split | checksum | 186.8 [134.5–187.4] |
| resident_split | iter_total | 1206.4 [1167.7–1382.8] |
| gemm_device_hold | iter_total | 732.7 [584.4–828.0] |
| gemm_device_drop_in | iter_total | 743.6 [585.1–931.2] |
| gemm_device_download | iter_total | 873.8 [627.7–920.4] |
| backend_drop | kernel_call | 976.1 [932.4–1084.6] |
| backend_drop | iter_total | 976.5 [932.7–1085.6] |

### N=1024

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.1–0.1] |
| fresh | var_upload | 0.9 [0.8–1.2] |
| fresh | matmul | 3730.5 [3126.3–3918.1] |
| fresh | readout | 0.3 [0.3–0.5] |
| fresh | checksum | 564.9 [542.9–756.7] |
| fresh | iter_total | 4373.4 [3771.1–4602.3] |
| fresh | tape_drop | 1.6 [1.3–2.1] |
| resident_public | matmul | 3755.5 [3393.2–4153.4] |
| resident_public | readout | 0.4 [0.3–0.4] |
| resident_public | checksum | 546.5 [544.1–754.5] |
| resident_public | iter_total | 4473.4 [3973.6–4794.2] |
| resident_reupload | matmul | 3881.4 [3436.8–4444.4] |
| resident_reupload | readout | 0.5 [0.4–0.5] |
| resident_reupload | checksum | 551.5 [548.5–755.6] |
| resident_reupload | iter_total | 4573.0 [4114.5–5000.4] |
| resident_split | gemm_device | 3698.6 [3143.6–3899.5] |
| resident_split | download | 121.5 [115.7–131.0] |
| resident_split | out_drop | 0.4 [0.3–0.4] |
| resident_split | readout | 0.4 [0.3–0.4] |
| resident_split | checksum | 549.9 [542.3–748.2] |
| resident_split | iter_total | 4403.1 [3860.1–4655.1] |
| gemm_device_hold | iter_total | 2905.9 [2776.6–2979.0] |
| gemm_device_drop_in | iter_total | 2818.1 [2749.3–2906.1] |
| gemm_device_download | iter_total | 3065.3 [2814.3–3427.1] |
| backend_drop | kernel_call | 3503.4 [3131.9–3899.1] |
| backend_drop | iter_total | 3503.7 [3132.9–3899.9] |

### N=2048

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.2–0.2] |
| fresh | var_upload | 1.5 [1.3–1.8] |
| fresh | matmul | 18142.3 [17446.9–18334.4] |
| fresh | readout | 0.6 [0.5–0.8] |
| fresh | checksum | 3003.1 [2218.0–3012.1] |
| fresh | iter_total | 20643.0 [20406.9–20744.2] |
| fresh | tape_drop | 2336.5 [1750.6–2383.0] |
| resident_public | matmul | 23121.7 [22779.0–23789.3] |
| resident_public | readout | 0.8 [0.7–0.8] |
| resident_public | checksum | 3031.5 [2132.7–3032.9] |
| resident_public | iter_total | 25673.0 [25475.9–26139.4] |
| resident_reupload | matmul | 20165.7 [19785.9–21150.1] |
| resident_reupload | readout | 1.0 [0.9–1.1] |
| resident_reupload | checksum | 3028.8 [2159.3–3033.6] |
| resident_reupload | iter_total | 23041.0 [22023.9–24181.2] |
| resident_split | gemm_device | 17851.8 [17227.6–18292.7] |
| resident_split | download | 5409.1 [5218.2–5902.0] |
| resident_split | out_drop | 1.0 [0.8–1.4] |
| resident_split | readout | 0.7 [0.6–0.8] |
| resident_split | checksum | 2163.4 [2131.2–3002.7] |
| resident_split | iter_total | 25614.5 [25513.2–25872.5] |
| gemm_device_hold | iter_total | 17826.7 [17586.0–18305.1] |
| gemm_device_drop_in | iter_total | 19839.0 [19538.5–20687.1] |
| gemm_device_download | iter_total | 23126.4 [22751.5–23220.7] |
| backend_drop | kernel_call | 18076.5 [17374.4–18455.6] |
| backend_drop | iter_total | 18078.2 [17376.2–18458.6] |

### N=4096

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.2–0.2] |
| fresh | var_upload | 2.1 [2.0–2.2] |
| fresh | matmul | 113967.5 [113265.2–115664.1] |
| fresh | readout | 0.8 [0.7–0.9] |
| fresh | checksum | 12122.2 [12116.7–12123.0] |
| fresh | iter_total | 126097.6 [125158.0–126464.6] |
| fresh | tape_drop | 7133.7 [7028.4–7409.3] |
| resident_public | matmul | 141870.9 [141726.2–143046.7] |
| resident_public | readout | 1.1 [1.1–1.2] |
| resident_public | checksum | 8648.1 [8612.0–8728.0] |
| resident_public | iter_total | 150808.9 [150483.8–151794.4] |
| resident_reupload | matmul | 143075.0 [142171.4–144148.4] |
| resident_reupload | readout | 1.3 [1.2–1.3] |
| resident_reupload | checksum | 8632.8 [8621.9–8720.0] |
| resident_reupload | iter_total | 151869.8 [150913.8–152838.6] |
| resident_split | gemm_device | 113604.7 [112858.0–114591.4] |
| resident_split | download | 19677.6 [19638.4–19776.6] |
| resident_split | out_drop | 9213.8 [9209.4–9251.5] |
| resident_split | readout | 1.2 [1.1–1.3] |
| resident_split | checksum | 8713.2 [8633.3–8724.2] |
| resident_split | iter_total | 151465.2 [150201.9–151902.0] |
| gemm_device_hold | iter_total | 113616.3 [113200.7–114647.0] |
| gemm_device_drop_in | iter_total | 121596.1 [121384.7–122486.8] |
| gemm_device_download | iter_total | 133906.5 [132534.2–134756.5] |
| backend_drop | kernel_call | 114361.9 [114039.7–114673.1] |
| backend_drop | iter_total | 114363.7 [114041.5–114676.1] |

## run 別 iter_total（fresh / resident_public、us）

| N | arm | run1 | run2 | run3 | run4 | run5 |
|---|---|---|---|---|---|---|
| 256 | fresh | 187.6 | 182.3 | 187.8 | 179.3 | 179.2 |
| 256 | resident_public | 184.9 | 154.3 | 158.8 | 150.1 | 153.8 |
| 512 | fresh | 1105.2 | 1130.4 | 1163.2 | 1235.2 | 1071.9 |
| 512 | resident_public | 1239.5 | 1165.1 | 1236.6 | 1426.0 | 1147.3 |
| 1024 | fresh | 4373.4 | 4459.7 | 4602.3 | 3771.1 | 4310.9 |
| 1024 | resident_public | 4473.4 | 4794.2 | 4753.6 | 3973.6 | 4336.1 |
| 2048 | fresh | 20465.5 | 20406.9 | 20643.0 | 20744.2 | 20699.7 |
| 2048 | resident_public | 25673.0 | 25872.0 | 25612.8 | 25475.9 | 26139.4 |
| 4096 | fresh | 126097.6 | 125309.1 | 126101.6 | 126464.6 | 125158.0 |
| 4096 | resident_public | 150486.5 | 150483.8 | 150808.9 | 151794.4 | 150852.4 |

