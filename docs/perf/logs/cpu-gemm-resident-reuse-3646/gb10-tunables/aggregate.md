# 集計（#3646）: gb10-tunables（系列種別 tunables）

**判定対象外の系列: 数値記録のみ（RULE.txt）。帰属判定は行わない。**

checksum: 全 n・全腕・全 run で iter_total の checksum_bits が一致（hard 検査通過）。

## 判定表（5 run 中央値。単位 us）

### N=256

- fresh.iter_total = 171.0, resident_public.iter_total = 161.6, ratio = 0.945
- resident_split: 区間和 161.6 / iter_total 162.2（残差 0.6。記録のみ）
- tunables 解釈: ratio <= 1.05。機構は glibc の確保とページ返却を支持（RULE.txt）。

### N=512

- fresh.iter_total = 766.6, resident_public.iter_total = 801.0, ratio = 1.045
- resident_split: 区間和 849.7 / iter_total 819.0（残差 -30.7。記録のみ）
- tunables 解釈: ratio <= 1.05。機構は glibc の確保とページ返却を支持（RULE.txt）。

### N=1024

- fresh.iter_total = 2868.6, resident_public.iter_total = 3276.4, ratio = 1.142
- resident_split: 区間和 3236.9 / iter_total 3206.0（残差 -30.9。記録のみ）
- tunables 解釈: ratio > 1.05。glibc 確保機構だけでは説明できない。

### N=2048

- fresh.iter_total = 20597.4, resident_public.iter_total = 25835.6, ratio = 1.254
- resident_split: 区間和 25751.6 / iter_total 25979.2（残差 227.6。記録のみ）
- tunables 解釈: ratio > 1.05。glibc 確保機構だけでは説明できない。

### N=4096

- fresh.iter_total = 124632.8, resident_public.iter_total = 152143.4, ratio = 1.221
- resident_split: 区間和 151751.2 / iter_total 151739.9（残差 -11.2。記録のみ）
- tunables 解釈: ratio > 1.05。glibc 確保機構だけでは説明できない。

## 区間表（5 run 中央値 [min–max]、us）

### N=256

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.0 [0.0–0.0] |
| fresh | var_upload | 0.3 [0.2–0.3] |
| fresh | matmul | 135.0 [130.1–135.5] |
| fresh | readout | 0.2 [0.2–0.2] |
| fresh | checksum | 34.1 [33.9–34.5] |
| fresh | iter_total | 171.0 [165.5–176.5] |
| fresh | tape_drop | 0.7 [0.6–0.8] |
| resident_public | matmul | 127.3 [124.3–133.0] |
| resident_public | readout | 0.2 [0.2–0.2] |
| resident_public | checksum | 33.6 [33.6–33.6] |
| resident_public | iter_total | 161.6 [158.3–170.1] |
| resident_reupload | matmul | 155.9 [140.5–161.2] |
| resident_reupload | readout | 0.2 [0.2–0.2] |
| resident_reupload | checksum | 33.6 [33.6–33.6] |
| resident_reupload | iter_total | 193.3 [175.3–199.6] |
| resident_split | gemm_device | 121.7 [117.5–128.9] |
| resident_split | download | 5.9 [5.8–7.7] |
| resident_split | out_drop | 0.2 [0.2–0.2] |
| resident_split | readout | 0.2 [0.1–0.2] |
| resident_split | checksum | 33.6 [33.6–33.6] |
| resident_split | iter_total | 162.2 [158.2–179.0] |
| gemm_device_hold | iter_total | 102.3 [98.2–112.6] |
| gemm_device_drop_in | iter_total | 105.4 [89.0–111.5] |
| gemm_device_download | iter_total | 127.0 [125.9–130.3] |
| backend_drop | kernel_call | 132.7 [128.3–136.3] |
| backend_drop | iter_total | 132.7 [128.3–136.4] |

### N=512

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.0–0.1] |
| fresh | var_upload | 0.5 [0.4–0.6] |
| fresh | matmul | 618.5 [585.7–630.5] |
| fresh | readout | 0.3 [0.2–0.3] |
| fresh | checksum | 142.8 [134.9–187.9] |
| fresh | iter_total | 766.6 [743.7–778.2] |
| fresh | tape_drop | 1.0 [0.8–1.1] |
| resident_public | matmul | 646.3 [595.8–653.8] |
| resident_public | readout | 0.3 [0.2–0.4] |
| resident_public | checksum | 137.2 [134.6–189.1] |
| resident_public | iter_total | 801.0 [748.0–805.8] |
| resident_reupload | matmul | 721.2 [652.1–767.0] |
| resident_reupload | readout | 0.3 [0.3–0.4] |
| resident_reupload | checksum | 139.9 [134.8–189.0] |
| resident_reupload | iter_total | 882.4 [824.8–934.7] |
| resident_split | gemm_device | 630.9 [617.0–699.5] |
| resident_split | download | 31.4 [26.8–31.6] |
| resident_split | out_drop | 0.2 [0.2–0.3] |
| resident_split | readout | 0.3 [0.3–0.3] |
| resident_split | checksum | 186.8 [134.7–187.1] |
| resident_split | iter_total | 819.0 [785.4–885.6] |
| gemm_device_hold | iter_total | 414.3 [389.0–573.1] |
| gemm_device_drop_in | iter_total | 479.3 [446.6–541.0] |
| gemm_device_download | iter_total | 553.0 [547.4–615.0] |
| backend_drop | kernel_call | 607.2 [526.3–678.2] |
| backend_drop | iter_total | 607.5 [526.5–678.5] |

### N=1024

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.1–0.1] |
| fresh | var_upload | 0.7 [0.6–0.8] |
| fresh | matmul | 2243.4 [2164.4–2417.3] |
| fresh | readout | 0.3 [0.2–0.4] |
| fresh | checksum | 544.9 [541.4–754.4] |
| fresh | iter_total | 2868.6 [2765.2–3174.2] |
| fresh | tape_drop | 1.3 [1.1–1.5] |
| resident_public | matmul | 2567.3 [2326.4–2662.2] |
| resident_public | readout | 0.5 [0.3–0.5] |
| resident_public | checksum | 755.0 [552.2–757.1] |
| resident_public | iter_total | 3276.4 [3030.9–3367.0] |
| resident_reupload | matmul | 2605.6 [2566.7–2704.4] |
| resident_reupload | readout | 0.5 [0.5–0.6] |
| resident_reupload | checksum | 756.5 [554.0–761.9] |
| resident_reupload | iter_total | 3332.0 [3266.9–3357.7] |
| resident_split | gemm_device | 2353.3 [2320.2–2431.5] |
| resident_split | download | 134.5 [116.6–135.5] |
| resident_split | out_drop | 0.3 [0.3–0.4] |
| resident_split | readout | 0.5 [0.4–0.6] |
| resident_split | checksum | 748.3 [555.8–754.4] |
| resident_split | iter_total | 3206.0 [3121.5–3317.0] |
| gemm_device_hold | iter_total | 2072.6 [2021.0–2104.6] |
| gemm_device_drop_in | iter_total | 2067.8 [2029.2–2101.5] |
| gemm_device_download | iter_total | 2244.3 [2135.2–2320.1] |
| backend_drop | kernel_call | 2282.6 [2245.6–2308.7] |
| backend_drop | iter_total | 2283.2 [2246.6–2309.1] |

### N=2048

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.2–0.2] |
| fresh | var_upload | 1.6 [1.3–1.7] |
| fresh | matmul | 18174.2 [17307.8–18387.4] |
| fresh | readout | 0.8 [0.6–0.9] |
| fresh | checksum | 3008.5 [2197.4–3013.8] |
| fresh | iter_total | 20597.4 [20341.1–20753.2] |
| fresh | tape_drop | 2298.9 [1902.6–2434.2] |
| resident_public | matmul | 23550.4 [23316.2–23675.2] |
| resident_public | readout | 0.8 [0.7–0.9] |
| resident_public | checksum | 2162.2 [2139.6–2165.6] |
| resident_public | iter_total | 25835.6 [25715.4–25909.4] |
| resident_reupload | matmul | 19911.1 [19539.5–20268.7] |
| resident_reupload | readout | 1.1 [1.0–1.2] |
| resident_reupload | checksum | 2163.7 [2161.2–2170.1] |
| resident_reupload | iter_total | 22376.7 [21864.3–23016.1] |
| resident_split | gemm_device | 17918.7 [16985.3–18234.3] |
| resident_split | download | 5666.3 [5597.6–6187.2] |
| resident_split | out_drop | 1.0 [0.9–1.4] |
| resident_split | readout | 0.7 [0.7–0.8] |
| resident_split | checksum | 2164.9 [2162.8–3006.8] |
| resident_split | iter_total | 25979.2 [25723.5–26084.2] |
| gemm_device_hold | iter_total | 17507.4 [17401.7–18496.5] |
| gemm_device_drop_in | iter_total | 19663.3 [19468.6–20198.1] |
| gemm_device_download | iter_total | 23309.5 [22916.1–23464.4] |
| backend_drop | kernel_call | 18167.2 [17721.6–18351.3] |
| backend_drop | iter_total | 18169.0 [17723.2–18353.1] |

### N=4096

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.2–0.3] |
| fresh | var_upload | 2.2 [2.1–2.3] |
| fresh | matmul | 112899.3 [112616.9–113267.6] |
| fresh | readout | 0.9 [0.9–0.9] |
| fresh | checksum | 12132.4 [12129.5–12145.2] |
| fresh | iter_total | 124632.8 [124294.9–125325.4] |
| fresh | tape_drop | 8416.3 [7595.1–8478.1] |
| resident_public | matmul | 143405.2 [142416.0–144755.9] |
| resident_public | readout | 1.1 [1.0–1.2] |
| resident_public | checksum | 8771.5 [8693.3–8792.2] |
| resident_public | iter_total | 152143.4 [151309.5–153559.4] |
| resident_reupload | matmul | 142892.1 [142163.6–143771.9] |
| resident_reupload | readout | 1.2 [1.2–1.3] |
| resident_reupload | checksum | 8720.1 [8699.5–8758.3] |
| resident_reupload | iter_total | 151480.1 [150862.8–152536.2] |
| resident_split | gemm_device | 112630.7 [112228.7–113054.2] |
| resident_split | download | 21040.2 [20774.5–21096.8] |
| resident_split | out_drop | 9297.6 [9271.4–9320.1] |
| resident_split | readout | 1.1 [1.1–1.3] |
| resident_split | checksum | 8781.5 [8693.1–8789.9] |
| resident_split | iter_total | 151739.9 [151410.7–152093.4] |
| gemm_device_hold | iter_total | 113238.2 [112493.3–113765.4] |
| gemm_device_drop_in | iter_total | 121141.3 [120610.6–122161.0] |
| gemm_device_download | iter_total | 134041.4 [133926.4–135620.7] |
| backend_drop | kernel_call | 113400.7 [112600.6–115845.7] |
| backend_drop | iter_total | 113404.3 [112603.6–115847.7] |

## run 別 iter_total（fresh / resident_public、us）

| N | arm | run1 | run2 | run3 | run4 | run5 |
|---|---|---|---|---|---|---|
| 256 | fresh | 176.5 | 165.5 | 166.7 | 171.0 | 172.8 |
| 256 | resident_public | 163.7 | 158.3 | 170.1 | 160.6 | 161.6 |
| 512 | fresh | 766.2 | 766.6 | 778.2 | 771.2 | 743.7 |
| 512 | resident_public | 794.1 | 748.0 | 801.0 | 805.8 | 804.6 |
| 1024 | fresh | 2795.3 | 2765.2 | 2868.6 | 2999.7 | 3174.2 |
| 1024 | resident_public | 3367.0 | 3276.4 | 3030.9 | 3332.9 | 3073.1 |
| 2048 | fresh | 20636.1 | 20341.1 | 20482.8 | 20753.2 | 20597.4 |
| 2048 | resident_public | 25715.4 | 25835.6 | 25842.1 | 25772.8 | 25909.4 |
| 4096 | fresh | 124294.9 | 124756.0 | 124435.6 | 124632.8 | 125325.4 |
| 4096 | resident_public | 153559.4 | 151309.5 | 153139.6 | 151602.9 | 152143.4 |

