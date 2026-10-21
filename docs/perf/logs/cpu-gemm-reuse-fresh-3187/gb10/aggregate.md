# 集計（#3187）: gb10

checksum: 全 n・全腕・全 run で iter_total の checksum_bits が一致（hard 検査通過）。

## 判定表（5 run 中央値。単位 us）

### N=256

- fresh.iter_total = 179.9, reuse.iter_total = 258.0, ratio = 1.434
- 残差 (iter_total - sum(区間)): fresh = 1.3, reuse = 0.4, reuse - fresh = -0.9（総時間差 78.1）
- 判定: 逆転が再現（ratio > 1.00）
- D_matmul = 79.6, D_readout = -0.0, D_checksum = -0.2, G = 79.4
- F_only (tape_build + var_upload) = 0.5
- 帰属: 支持 matmul→出力確保（結果蓄積・ページ確保）（100%）
- rho_FH = 0.77（結果蓄積が原因を支持）
- rho_B = 0.72（出力バッファの確保パターンが原因を支持）
- arena A/B: fresh_arena hit=780, reuse_arena hit=0（予測: 0）, reuse_arena.iter_total = 240.4
  - 既存の host arena (#2104) では reuse の逆転を是正できない

### N=512

- fresh.iter_total = 794.2, reuse.iter_total = 1241.6, ratio = 1.563
- 残差 (iter_total - sum(区間)): fresh = 34.2, reuse = 23.1, reuse - fresh = -11.1（総時間差 447.3）
- 判定: 逆転が再現（ratio > 1.00）
- D_matmul = 459.2, D_readout = -0.0, D_checksum = -0.0, G = 459.1
- F_only (tape_build + var_upload) = 0.7
- 帰属: 支持 matmul→出力確保（結果蓄積・ページ確保）（100%）
- rho_FH = 1.01（結果蓄積が原因を支持）
- rho_B = 0.94（出力バッファの確保パターンが原因を支持）
- arena A/B: fresh_arena hit=780, reuse_arena hit=0（予測: 0）, reuse_arena.iter_total = 1292.8
  - 既存の host arena (#2104) では reuse の逆転を是正できない

### N=1024

- fresh.iter_total = 3105.5, reuse.iter_total = 4641.2, ratio = 1.495
- 残差 (iter_total - sum(区間)): fresh = -62.3, reuse = 103.4, reuse - fresh = 165.7（総時間差 1535.7）
- 判定: 逆転が再現（ratio > 1.00）
- D_matmul = 1581.5, D_readout = 0.1, D_checksum = -210.6, G = 1370.9
- F_only (tape_build + var_upload) = 0.9
- 帰属: 支持 matmul→出力確保（結果蓄積・ページ確保）（115%）
- rho_FH = 0.96（結果蓄積が原因を支持）
- rho_B = 1.03（出力バッファの確保パターンが原因を支持）
- arena A/B: fresh_arena hit=780, reuse_arena hit=0（予測: 0）, reuse_arena.iter_total = 4578.0
  - 既存の host arena (#2104) では reuse の逆転を是正できない

### N=2048

- fresh.iter_total = 16091.6, reuse.iter_total = 17952.4, ratio = 1.116
- 残差 (iter_total - sum(区間)): fresh = 8.3, reuse = -10.3, reuse - fresh = -18.6（総時間差 1860.8）
- 判定: 逆転が再現（ratio > 1.00）
- D_matmul = 1891.2, D_readout = -0.1, D_checksum = -9.4, G = 1881.8
- F_only (tape_build + var_upload) = 2.5
- 帰属: 支持 matmul→出力確保（結果蓄積・ページ確保）（101%）
- rho_FH = 1.06（結果蓄積が原因を支持）
- rho_B = 0.97（出力バッファの確保パターンが原因を支持）
- arena A/B: fresh_arena hit=780, reuse_arena hit=0（予測: 0）, reuse_arena.iter_total = 17783.9
  - 既存の host arena (#2104) では reuse の逆転を是正できない

## 区間表（5 run 中央値 [min–max]、us）

### N=256

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.1–0.2] |
| fresh | var_upload | 0.3 [0.2–0.4] |
| fresh | matmul | 144.0 [129.7–145.0] |
| fresh | host_copy | 0.2 [0.1–0.2] |
| fresh | checksum | 33.9 [33.8–47.5] |
| fresh | iter_total | 179.9 [164.3–188.5] |
| fresh | tape_drop | 0.7 [0.5–0.8] |
| fresh_hold | tape_build | 0.1 [0.1–0.1] |
| fresh_hold | var_upload | 0.3 [0.2–0.4] |
| fresh_hold | matmul | 205.5 [201.6–223.5] |
| fresh_hold | host_copy | 0.1 [0.1–0.2] |
| fresh_hold | checksum | 33.8 [33.7–33.8] |
| fresh_hold | iter_total | 240.1 [236.7–263.7] |
| fresh_arena | tape_build | 0.1 [0.1–0.2] |
| fresh_arena | var_upload | 0.4 [0.4–0.5] |
| fresh_arena | matmul | 142.2 [136.5–159.0] |
| fresh_arena | host_copy | 0.2 [0.1–0.2] |
| fresh_arena | checksum | 34.0 [33.9–34.3] |
| fresh_arena | iter_total | 183.2 [172.5–199.1] |
| fresh_arena | tape_drop | 0.8 [0.7–1.0] |
| fresh_legacy | tape_build | 0.1 [0.1–0.2] |
| fresh_legacy | var_upload | 0.3 [0.2–0.5] |
| fresh_legacy | matmul | 134.3 [131.4–149.2] |
| fresh_legacy | host_copy | 6.4 [5.3–7.6] |
| fresh_legacy | checksum | 33.6 [33.6–33.6] |
| fresh_legacy | iter_total | 174.4 [172.8–196.1] |
| fresh_legacy | tape_drop | 0.8 [0.7–0.9] |
| reuse | matmul | 223.6 [215.3–232.0] |
| reuse | host_copy | 0.1 [0.1–0.2] |
| reuse | checksum | 33.8 [33.8–33.8] |
| reuse | iter_total | 258.0 [250.0–266.5] |
| reuse_arena | matmul | 206.0 [200.8–212.0] |
| reuse_arena | host_copy | 0.1 [0.1–0.2] |
| reuse_arena | checksum | 33.8 [33.8–33.8] |
| reuse_arena | iter_total | 240.4 [234.8–246.1] |
| reuse_legacy | matmul | 143.7 [127.5–148.9] |
| reuse_legacy | host_copy | 82.0 [81.3–84.4] |
| reuse_legacy | checksum | 33.6 [33.6–33.6] |
| reuse_legacy | iter_total | 254.7 [236.4–259.6] |
| backend_drop | kernel_call | 145.2 [140.1–153.0] |
| backend_drop | iter_total | 145.3 [140.1–153.1] |
| backend_hold | kernel_call | 202.9 [197.5–206.7] |
| backend_hold | iter_total | 203.1 [197.6–206.9] |

### N=512

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.1–0.2] |
| fresh | var_upload | 0.5 [0.5–0.6] |
| fresh | matmul | 624.3 [614.4–728.7] |
| fresh | host_copy | 0.2 [0.2–0.3] |
| fresh | checksum | 134.9 [134.7–188.1] |
| fresh | iter_total | 794.2 [755.2–884.0] |
| fresh | tape_drop | 0.9 [0.8–1.1] |
| fresh_hold | tape_build | 0.2 [0.1–0.2] |
| fresh_hold | var_upload | 0.6 [0.5–0.8] |
| fresh_hold | matmul | 1087.7 [1012.6–1149.5] |
| fresh_hold | host_copy | 0.2 [0.2–0.3] |
| fresh_hold | checksum | 135.0 [134.8–188.3] |
| fresh_hold | iter_total | 1279.7 [1188.5–1325.0] |
| fresh_arena | tape_build | 0.1 [0.1–0.2] |
| fresh_arena | var_upload | 0.6 [0.5–0.8] |
| fresh_arena | matmul | 649.9 [643.2–704.2] |
| fresh_arena | host_copy | 0.2 [0.2–0.2] |
| fresh_arena | checksum | 135.0 [134.8–140.1] |
| fresh_arena | iter_total | 811.6 [780.2–870.0] |
| fresh_arena | tape_drop | 1.0 [0.8–1.3] |
| fresh_legacy | tape_build | 0.2 [0.1–0.2] |
| fresh_legacy | var_upload | 0.5 [0.5–0.6] |
| fresh_legacy | matmul | 663.3 [625.5–716.8] |
| fresh_legacy | host_copy | 29.6 [25.6–31.6] |
| fresh_legacy | checksum | 137.4 [134.9–187.1] |
| fresh_legacy | iter_total | 868.9 [811.4–913.8] |
| fresh_legacy | tape_drop | 1.0 [0.9–1.1] |
| reuse | matmul | 1083.5 [1062.5–1132.5] |
| reuse | host_copy | 0.2 [0.2–0.2] |
| reuse | checksum | 134.8 [134.8–134.9] |
| reuse | iter_total | 1241.6 [1209.8–1270.8] |
| reuse_arena | matmul | 1132.0 [1090.3–1211.3] |
| reuse_arena | host_copy | 0.2 [0.2–0.3] |
| reuse_arena | checksum | 138.5 [134.8–188.0] |
| reuse_arena | iter_total | 1292.8 [1234.9–1399.8] |
| reuse_legacy | matmul | 781.3 [639.8–827.7] |
| reuse_legacy | host_copy | 367.2 [363.5–382.6] |
| reuse_legacy | checksum | 135.4 [134.9–187.0] |
| reuse_legacy | iter_total | 1304.2 [1188.3–1363.1] |
| backend_drop | kernel_call | 645.9 [578.0–718.2] |
| backend_drop | iter_total | 646.2 [578.5–718.5] |
| backend_hold | kernel_call | 1079.5 [1061.5–1187.1] |
| backend_hold | iter_total | 1079.7 [1062.3–1187.3] |

### N=1024

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.1–0.2] |
| fresh | var_upload | 0.8 [0.5–0.8] |
| fresh | matmul | 2409.3 [2310.0–2532.2] |
| fresh | host_copy | 0.3 [0.2–0.4] |
| fresh | checksum | 757.3 [541.9–762.7] |
| fresh | iter_total | 3105.5 [2964.0–3248.2] |
| fresh | tape_drop | 1.5 [0.9–1.6] |
| fresh_hold | tape_build | 0.2 [0.2–0.2] |
| fresh_hold | var_upload | 1.0 [0.9–1.2] |
| fresh_hold | matmul | 3929.9 [3878.9–4068.3] |
| fresh_hold | host_copy | 0.4 [0.3–0.5] |
| fresh_hold | checksum | 550.5 [544.6–752.9] |
| fresh_hold | iter_total | 4593.9 [4557.5–4713.8] |
| fresh_arena | tape_build | 0.2 [0.1–0.2] |
| fresh_arena | var_upload | 0.9 [0.7–1.1] |
| fresh_arena | matmul | 2381.0 [2337.4–2525.8] |
| fresh_arena | host_copy | 0.3 [0.2–0.4] |
| fresh_arena | checksum | 544.4 [541.9–762.6] |
| fresh_arena | iter_total | 3015.3 [2992.0–3155.2] |
| fresh_arena | tape_drop | 1.6 [1.3–2.2] |
| fresh_legacy | tape_build | 0.2 [0.2–0.2] |
| fresh_legacy | var_upload | 0.9 [0.9–1.1] |
| fresh_legacy | matmul | 2524.1 [2417.7–2539.1] |
| fresh_legacy | host_copy | 126.0 [114.4–131.3] |
| fresh_legacy | checksum | 556.0 [552.2–752.8] |
| fresh_legacy | iter_total | 3289.7 [3164.6–3374.1] |
| fresh_legacy | tape_drop | 1.5 [1.4–1.5] |
| reuse | matmul | 3990.8 [3944.1–4030.6] |
| reuse | host_copy | 0.4 [0.4–0.5] |
| reuse | checksum | 546.6 [542.2–758.5] |
| reuse | iter_total | 4641.2 [4567.9–4690.5] |
| reuse_arena | matmul | 3919.9 [3849.5–3942.6] |
| reuse_arena | host_copy | 0.3 [0.2–0.4] |
| reuse_arena | checksum | 755.2 [542.7–757.7] |
| reuse_arena | iter_total | 4578.0 [4489.8–4633.8] |
| reuse_legacy | matmul | 2795.4 [2643.8–2937.8] |
| reuse_legacy | host_copy | 1445.1 [1380.9–1453.1] |
| reuse_legacy | checksum | 542.6 [540.4–754.1] |
| reuse_legacy | iter_total | 5005.1 [4642.2–5135.7] |
| backend_drop | kernel_call | 2337.6 [2278.5–2466.1] |
| backend_drop | iter_total | 2338.1 [2279.2–2467.5] |
| backend_hold | kernel_call | 3961.8 [3803.2–4015.5] |
| backend_hold | iter_total | 3962.6 [3803.9–4016.9] |

### N=2048

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.2–0.3] |
| fresh | var_upload | 2.2 [2.1–2.3] |
| fresh | matmul | 13034.8 [12936.2–13186.3] |
| fresh | host_copy | 0.8 [0.8–0.8] |
| fresh | checksum | 3045.2 [3044.5–3046.6] |
| fresh | iter_total | 16091.6 [16001.1–16212.6] |
| fresh | tape_drop | 3.4 [3.4–3.5] |
| fresh_hold | tape_build | 0.2 [0.2–0.3] |
| fresh_hold | var_upload | 2.0 [1.9–2.0] |
| fresh_hold | matmul | 15045.7 [14952.3–15136.0] |
| fresh_hold | host_copy | 0.8 [0.7–0.8] |
| fresh_hold | checksum | 3017.6 [3016.3–3022.4] |
| fresh_hold | iter_total | 18081.7 [18013.7–18190.5] |
| fresh_arena | tape_build | 0.2 [0.2–0.3] |
| fresh_arena | var_upload | 1.6 [1.6–1.7] |
| fresh_arena | matmul | 13044.4 [12931.0–13083.5] |
| fresh_arena | host_copy | 0.8 [0.8–0.8] |
| fresh_arena | checksum | 3044.5 [3044.0–3046.2] |
| fresh_arena | iter_total | 16106.4 [15974.2–16137.9] |
| fresh_arena | tape_drop | 4.0 [4.0–4.1] |
| fresh_legacy | tape_build | 0.3 [0.3–0.3] |
| fresh_legacy | var_upload | 2.0 [2.0–2.1] |
| fresh_legacy | matmul | 13149.5 [13116.5–13224.1] |
| fresh_legacy | host_copy | 649.2 [594.7–679.4] |
| fresh_legacy | checksum | 3009.4 [3006.9–3011.0] |
| fresh_legacy | iter_total | 16835.0 [16771.4–16880.5] |
| fresh_legacy | tape_drop | 3.4 [3.4–3.5] |
| reuse | matmul | 14926.0 [14827.0–15102.2] |
| reuse | host_copy | 0.8 [0.7–0.8] |
| reuse | checksum | 3035.8 [3018.8–3038.1] |
| reuse | iter_total | 17952.4 [17777.4–18142.0] |
| reuse_arena | matmul | 14733.1 [14681.2–14872.1] |
| reuse_arena | host_copy | 0.8 [0.7–0.8] |
| reuse_arena | checksum | 3019.0 [3016.8–3020.1] |
| reuse_arena | iter_total | 17783.9 [17702.2–17886.9] |
| reuse_legacy | matmul | 13253.0 [13217.7–13445.5] |
| reuse_legacy | host_copy | 5579.2 [5398.3–5726.4] |
| reuse_legacy | checksum | 3009.1 [3008.1–3010.2] |
| reuse_legacy | iter_total | 22281.8 [22156.9–22454.7] |
| backend_drop | kernel_call | 13025.9 [12933.7–13081.6] |
| backend_drop | iter_total | 13028.1 [12936.1–13082.9] |
| backend_hold | kernel_call | 14857.5 [14698.1–15024.4] |
| backend_hold | iter_total | 14859.5 [14699.9–15026.1] |

## run 別 iter_total（fresh / reuse、us）

| N | arm | run1 | run2 | run3 | run4 | run5 |
|---|---|---|---|---|---|---|
| 256 | fresh | 182.6 | 164.3 | 188.5 | 179.9 | 172.7 |
| 256 | reuse | 260.2 | 266.5 | 250.0 | 258.0 | 255.6 |
| 512 | fresh | 794.2 | 858.9 | 774.4 | 755.2 | 884.0 |
| 512 | reuse | 1241.6 | 1270.8 | 1223.1 | 1209.8 | 1243.3 |
| 1024 | fresh | 3248.2 | 2964.0 | 3076.2 | 3105.5 | 3110.2 |
| 1024 | reuse | 4622.1 | 4641.2 | 4567.9 | 4661.3 | 4690.5 |
| 2048 | fresh | 16018.9 | 16091.6 | 16001.1 | 16212.6 | 16191.6 |
| 2048 | reuse | 17952.4 | 18142.0 | 17873.8 | 17777.4 | 17971.9 |

