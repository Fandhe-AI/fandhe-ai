# #2107 集計結果（RULE.txt の規則。単位 μs・5 run 中央値）

## N=1024

| 腕 | 区間 | median | min | max |
|---|---|---|---|---|
| clone_dtoh_legacy_to_vec | h2d_a | 86.3 | 84.5 | 93.1 |
| clone_dtoh_legacy_to_vec | h2d_b | 79.1 | 78.8 | 79.2 |
| clone_dtoh_legacy_to_vec | alloc_c | 0.1 | 0.1 | 0.2 |
| clone_dtoh_legacy_to_vec | launch_issue | 3.3 | 3.2 | 3.9 |
| clone_dtoh_legacy_to_vec | kernel_wait | 181.5 | 181.0 | 181.8 |
| clone_dtoh_legacy_to_vec | clone_dtoh | 75.6 | 75.6 | 76.0 |
| clone_dtoh_legacy_to_vec | d2h_sync | 0.3 | 0.3 | 0.3 |
| clone_dtoh_legacy_to_vec | host_read | 2067.6 | 1999.2 | 2227.1 |
| clone_dtoh_legacy_to_vec | Σ_matmul_equiv | 426.0 | | |
| clone_dtoh_borrowed_keep_alive | h2d_a | 84.8 | 83.0 | 84.8 |
| clone_dtoh_borrowed_keep_alive | h2d_b | 78.8 | 78.5 | 79.0 |
| clone_dtoh_borrowed_keep_alive | alloc_c | 0.1 | 0.1 | 0.1 |
| clone_dtoh_borrowed_keep_alive | launch_issue | 3.0 | 2.8 | 3.1 |
| clone_dtoh_borrowed_keep_alive | kernel_wait | 189.7 | 189.1 | 190.0 |
| clone_dtoh_borrowed_keep_alive | clone_dtoh | 36375.9 | 32565.0 | 40691.9 |
| clone_dtoh_borrowed_keep_alive | d2h_sync | 0.5 | 0.4 | 0.5 |
| clone_dtoh_borrowed_keep_alive | host_read | 540.7 | 534.4 | 540.9 |
| clone_dtoh_borrowed_keep_alive | Σ_matmul_equiv | 36732.3 | | |
| clone_dtoh_borrowed_dummy_alloc_free | h2d_a | 95.2 | 91.6 | 100.9 |
| clone_dtoh_borrowed_dummy_alloc_free | h2d_b | 79.2 | 78.2 | 79.9 |
| clone_dtoh_borrowed_dummy_alloc_free | alloc_c | 0.1 | 0.1 | 0.2 |
| clone_dtoh_borrowed_dummy_alloc_free | launch_issue | 3.3 | 3.3 | 3.5 |
| clone_dtoh_borrowed_dummy_alloc_free | kernel_wait | 181.5 | 180.9 | 182.0 |
| clone_dtoh_borrowed_dummy_alloc_free | clone_dtoh | 75.8 | 75.6 | 76.0 |
| clone_dtoh_borrowed_dummy_alloc_free | d2h_sync | 0.3 | 0.3 | 0.3 |
| clone_dtoh_borrowed_dummy_alloc_free | host_read | 1925.4 | 1802.6 | 1944.4 |
| clone_dtoh_borrowed_dummy_alloc_free | Σ_matmul_equiv | 435.7 | | |
| pretouched_fresh_split | h2d_a | 81.1 | 80.1 | 82.0 |
| pretouched_fresh_split | h2d_b | 78.8 | 78.4 | 79.4 |
| pretouched_fresh_split | alloc_c | 0.1 | 0.1 | 0.1 |
| pretouched_fresh_split | launch_issue | 3.3 | 3.2 | 3.6 |
| pretouched_fresh_split | kernel_wait | 181.3 | 181.1 | 181.8 |
| pretouched_fresh_split | dest_alloc | 4.3 | 4.2 | 5.4 |
| pretouched_fresh_split | pretouch_fill | 1335.5 | 1158.4 | 1466.2 |
| pretouched_fresh_split | d2h_issue | 81.1 | 80.8 | 81.9 |
| pretouched_fresh_split | d2h_sync | 0.5 | 0.4 | 0.6 |
| pretouched_fresh_split | host_read | 541.1 | 533.8 | 541.7 |
| pretouched_fresh_split | Σ_matmul_equiv | 1765.8 | | |
| pretouched_fresh_production | h2d_a | 81.4 | 81.0 | 82.8 |
| pretouched_fresh_production | h2d_b | 78.7 | 78.6 | 79.1 |
| pretouched_fresh_production | alloc_c | 0.1 | 0.1 | 0.1 |
| pretouched_fresh_production | launch_issue | 3.1 | 3.0 | 3.4 |
| pretouched_fresh_production | kernel_wait | 181.1 | 181.1 | 181.5 |
| pretouched_fresh_production | readback_total | 1522.8 | 1388.4 | 1681.1 |
| pretouched_fresh_production | host_read | 534.0 | 526.4 | 540.5 |
| pretouched_fresh_production | Σ_matmul_equiv | 1867.2 | | |
| prod_order_fresh | h2d_a | 81.5 | 80.1 | 82.0 |
| prod_order_fresh | h2d_b | 78.7 | 78.4 | 79.0 |
| prod_order_fresh | alloc_c | 0.1 | 0.1 | 0.1 |
| prod_order_fresh | launch_issue | 3.3 | 3.2 | 3.5 |
| prod_order_fresh | readback_total | 1450.8 | 1406.6 | 1536.3 |
| prod_order_fresh | host_read | 540.3 | 527.1 | 541.0 |
| prod_order_fresh | Σ_matmul_equiv | 1615.2 | | |
| prod_order_reused | h2d_a | 79.6 | 79.5 | 79.7 |
| prod_order_reused | h2d_b | 78.5 | 78.4 | 78.7 |
| prod_order_reused | alloc_c | 0.1 | 0.1 | 0.1 |
| prod_order_reused | launch_issue | 2.0 | 2.0 | 2.0 |
| prod_order_reused | d2h_issue | 259.2 | 259.1 | 259.5 |
| prod_order_reused | d2h_sync | 0.4 | 0.4 | 0.4 |
| prod_order_reused | host_read | 533.8 | 532.6 | 540.8 |
| prod_order_reused | Σ_matmul_equiv | 419.9 | | |
| Layer A | matmul | 1536.3 | 1393.2 | 1667.8 |

- exposed_alloc_fill = prod_order_fresh.readback_total − prod_order_reused.(d2h_issue+d2h_sync) = 1450.8 − 259.6 = 1191.2 us
- residual = LayerA.matmul − Σ_matmul_equiv(prod_order_reused) = 1116.4 us
- alloc_fill_share = 1.067（exposed_alloc_fill / residual）
- **判定: 宛先確保・事前タッチへの帰属を支持**
- 判定が示す範囲: 本番順序で表に出た宛先確保・事前タッチ費用（exposed cost）がresidual を占める割合のみ。重なりに隠れた費用・Layer B が再現しない Layer A 側の固定費は検出対象外
- 校正: LayerA.matmul − Σ_matmul_equiv(prod_order_fresh) = -78.9 us（-5.1% of LayerA。本番順序の Layer B がLayer A を再現する度合い。記録のみ）
- 重なり: Σ_matmul_equiv(直列 production) − Σ_matmul_equiv(prod_order_fresh) = 252.0 us（同一 readback 関数の直列/本番順序の壁時計差。記録のみ）
- 直列腕の dest_alloc+pretouch_fill = 1339.9 us（直列区間の値。exposed との差が重なりに隠れた分。判定には使わない・記録のみ）
- 旧 residual（§12.5 基準）= LayerA.matmul − Σ_matmul_equiv(keep_alive 直列) = -35196.0 us（連続性のための記録のみ）
- clone_dtoh 系 readback − split の (d2h_issue+d2h_sync) = 36294.7 us（直列腕同士の比較。記録のみ）
- 直列分解の整合（補助）: |split 分解和 − production readback_total| / production = 0.067（許容 0.05。超過。直列腕同士のみの整合で判定には使わない・記録のみ）
- 非後退 ratio = split.Σreadback / legacy.Σreadback = 18.719（直列腕同士。記録のみ）

## N=2048

| 腕 | 区間 | median | min | max |
|---|---|---|---|---|
| clone_dtoh_legacy_to_vec | h2d_a | 338.4 | 327.1 | 351.7 |
| clone_dtoh_legacy_to_vec | h2d_b | 319.5 | 318.0 | 334.5 |
| clone_dtoh_legacy_to_vec | alloc_c | 0.4 | 0.3 | 0.5 |
| clone_dtoh_legacy_to_vec | launch_issue | 7.8 | 6.2 | 8.1 |
| clone_dtoh_legacy_to_vec | kernel_wait | 1181.1 | 1180.3 | 1186.7 |
| clone_dtoh_legacy_to_vec | clone_dtoh | 282.4 | 281.0 | 288.6 |
| clone_dtoh_legacy_to_vec | d2h_sync | 0.4 | 0.3 | 0.4 |
| clone_dtoh_legacy_to_vec | host_read | 7432.3 | 6315.8 | 7624.5 |
| clone_dtoh_legacy_to_vec | Σ_matmul_equiv | 2127.5 | | |
| clone_dtoh_borrowed_keep_alive | h2d_a | 319.9 | 310.6 | 348.9 |
| clone_dtoh_borrowed_keep_alive | h2d_b | 319.1 | 313.3 | 328.9 |
| clone_dtoh_borrowed_keep_alive | alloc_c | 0.4 | 0.2 | 0.4 |
| clone_dtoh_borrowed_keep_alive | launch_issue | 7.0 | 3.9 | 7.1 |
| clone_dtoh_borrowed_keep_alive | kernel_wait | 1180.4 | 1179.6 | 1182.1 |
| clone_dtoh_borrowed_keep_alive | clone_dtoh | 10981.1 | 8784.1 | 149520.1 |
| clone_dtoh_borrowed_keep_alive | d2h_sync | 1.2 | 0.4 | 1.3 |
| clone_dtoh_borrowed_keep_alive | host_read | 2166.0 | 2154.9 | 2177.1 |
| clone_dtoh_borrowed_keep_alive | Σ_matmul_equiv | 12849.6 | | |
| clone_dtoh_borrowed_dummy_alloc_free | h2d_a | 365.5 | 340.1 | 384.6 |
| clone_dtoh_borrowed_dummy_alloc_free | h2d_b | 336.0 | 315.1 | 344.9 |
| clone_dtoh_borrowed_dummy_alloc_free | alloc_c | 0.4 | 0.2 | 0.5 |
| clone_dtoh_borrowed_dummy_alloc_free | launch_issue | 7.5 | 4.8 | 7.6 |
| clone_dtoh_borrowed_dummy_alloc_free | kernel_wait | 1178.2 | 1177.0 | 1181.4 |
| clone_dtoh_borrowed_dummy_alloc_free | clone_dtoh | 283.5 | 281.7 | 284.2 |
| clone_dtoh_borrowed_dummy_alloc_free | d2h_sync | 0.4 | 0.3 | 0.5 |
| clone_dtoh_borrowed_dummy_alloc_free | host_read | 7254.8 | 7023.3 | 7608.1 |
| clone_dtoh_borrowed_dummy_alloc_free | Σ_matmul_equiv | 2171.8 | | |
| pretouched_fresh_split | h2d_a | 320.4 | 313.5 | 341.1 |
| pretouched_fresh_split | h2d_b | 318.9 | 316.4 | 337.9 |
| pretouched_fresh_split | alloc_c | 0.4 | 0.2 | 0.5 |
| pretouched_fresh_split | launch_issue | 7.0 | 4.4 | 7.2 |
| pretouched_fresh_split | kernel_wait | 1161.3 | 1159.9 | 1162.5 |
| pretouched_fresh_split | dest_alloc | 8.5 | 5.7 | 8.8 |
| pretouched_fresh_split | pretouch_fill | 4934.2 | 4839.2 | 5495.0 |
| pretouched_fresh_split | d2h_issue | 301.8 | 291.5 | 303.0 |
| pretouched_fresh_split | d2h_sync | 1.3 | 0.6 | 1.3 |
| pretouched_fresh_split | host_read | 2165.9 | 2131.5 | 2170.0 |
| pretouched_fresh_split | Σ_matmul_equiv | 7033.0 | | |
| pretouched_fresh_production | h2d_a | 325.3 | 304.9 | 344.0 |
| pretouched_fresh_production | h2d_b | 318.9 | 309.5 | 331.4 |
| pretouched_fresh_production | alloc_c | 0.3 | 0.2 | 0.4 |
| pretouched_fresh_production | launch_issue | 6.8 | 4.4 | 7.0 |
| pretouched_fresh_production | kernel_wait | 1161.6 | 1160.1 | 1162.5 |
| pretouched_fresh_production | readback_total | 5333.3 | 4709.9 | 5690.5 |
| pretouched_fresh_production | host_read | 2168.1 | 2149.5 | 2169.7 |
| pretouched_fresh_production | Σ_matmul_equiv | 7168.4 | | |
| prod_order_fresh | h2d_a | 323.8 | 316.0 | 344.1 |
| prod_order_fresh | h2d_b | 316.7 | 310.7 | 334.6 |
| prod_order_fresh | alloc_c | 0.2 | 0.2 | 0.3 |
| prod_order_fresh | launch_issue | 4.4 | 4.4 | 6.6 |
| prod_order_fresh | readback_total | 5278.0 | 4369.1 | 5449.1 |
| prod_order_fresh | host_read | 2154.0 | 2121.0 | 2167.4 |
| prod_order_fresh | Σ_matmul_equiv | 5961.2 | | |
| prod_order_reused | h2d_a | 292.6 | 291.4 | 293.3 |
| prod_order_reused | h2d_b | 291.4 | 289.1 | 291.8 |
| prod_order_reused | alloc_c | 0.1 | 0.1 | 0.1 |
| prod_order_reused | launch_issue | 2.1 | 2.0 | 2.2 |
| prod_order_reused | d2h_issue | 1441.5 | 1440.3 | 1449.6 |
| prod_order_reused | d2h_sync | 0.4 | 0.4 | 0.4 |
| prod_order_reused | host_read | 2164.0 | 2152.3 | 2165.2 |
| prod_order_reused | Σ_matmul_equiv | 2027.6 | | |
| Layer A | matmul | 6201.0 | 5835.6 | 6405.7 |

- exposed_alloc_fill = prod_order_fresh.readback_total − prod_order_reused.(d2h_issue+d2h_sync) = 5278.0 − 1441.8 = 3836.1 us
- residual = LayerA.matmul − Σ_matmul_equiv(prod_order_reused) = 4173.4 us
- alloc_fill_share = 0.919（exposed_alloc_fill / residual）
- **判定: 宛先確保・事前タッチへの帰属を支持**
- 判定が示す範囲: 本番順序で表に出た宛先確保・事前タッチ費用（exposed cost）がresidual を占める割合のみ。重なりに隠れた費用・Layer B が再現しない Layer A 側の固定費は検出対象外
- 校正: LayerA.matmul − Σ_matmul_equiv(prod_order_fresh) = 239.8 us（3.9% of LayerA。本番順序の Layer B がLayer A を再現する度合い。記録のみ）
- 重なり: Σ_matmul_equiv(直列 production) − Σ_matmul_equiv(prod_order_fresh) = 1207.2 us（同一 readback 関数の直列/本番順序の壁時計差。記録のみ）
- 直列腕の dest_alloc+pretouch_fill = 4942.7 us（直列区間の値。exposed との差が重なりに隠れた分。判定には使わない・記録のみ）
- 旧 residual（§12.5 基準）= LayerA.matmul − Σ_matmul_equiv(keep_alive 直列) = -6648.5 us（連続性のための記録のみ）
- clone_dtoh 系 readback − split の (d2h_issue+d2h_sync) = 10679.3 us（直列腕同士の比較。記録のみ）
- 直列分解の整合（補助）: |split 分解和 − production readback_total| / production = 0.016（許容 0.05。OK。直列腕同士のみの整合で判定には使わない・記録のみ）
- 非後退 ratio = split.Σreadback / legacy.Σreadback = 18.510（直列腕同士。記録のみ）

## N=4096

| 腕 | 区間 | median | min | max |
|---|---|---|---|---|
| clone_dtoh_legacy_to_vec | h2d_a | 1165.8 | 1163.2 | 1188.1 |
| clone_dtoh_legacy_to_vec | h2d_b | 1156.1 | 1143.8 | 1162.3 |
| clone_dtoh_legacy_to_vec | alloc_c | 0.5 | 0.4 | 0.5 |
| clone_dtoh_legacy_to_vec | launch_issue | 8.1 | 8.0 | 8.4 |
| clone_dtoh_legacy_to_vec | kernel_wait | 9965.5 | 9726.2 | 10020.9 |
| clone_dtoh_legacy_to_vec | clone_dtoh | 21980.7 | 21220.2 | 23365.5 |
| clone_dtoh_legacy_to_vec | d2h_sync | 1.6 | 1.5 | 1.6 |
| clone_dtoh_legacy_to_vec | host_read | 37189.6 | 35467.5 | 37813.6 |
| clone_dtoh_legacy_to_vec | Σ_matmul_equiv | 34329.5 | | |
| clone_dtoh_borrowed_keep_alive | h2d_a | 1157.8 | 1155.8 | 1181.6 |
| clone_dtoh_borrowed_keep_alive | h2d_b | 1144.7 | 1138.5 | 1169.0 |
| clone_dtoh_borrowed_keep_alive | alloc_c | 0.4 | 0.3 | 0.4 |
| clone_dtoh_borrowed_keep_alive | launch_issue | 7.7 | 5.3 | 8.1 |
| clone_dtoh_borrowed_keep_alive | kernel_wait | 10012.2 | 9800.6 | 10185.1 |
| clone_dtoh_borrowed_keep_alive | clone_dtoh | 23500.5 | 23335.4 | 596670.0 |
| clone_dtoh_borrowed_keep_alive | d2h_sync | 1.5 | 0.5 | 1.8 |
| clone_dtoh_borrowed_keep_alive | host_read | 8756.9 | 8743.9 | 8772.7 |
| clone_dtoh_borrowed_keep_alive | Σ_matmul_equiv | 35992.0 | | |
| clone_dtoh_borrowed_dummy_alloc_free | h2d_a | 1169.1 | 1164.6 | 1170.9 |
| clone_dtoh_borrowed_dummy_alloc_free | h2d_b | 1146.5 | 1144.7 | 1148.5 |
| clone_dtoh_borrowed_dummy_alloc_free | alloc_c | 0.4 | 0.4 | 0.6 |
| clone_dtoh_borrowed_dummy_alloc_free | launch_issue | 8.1 | 7.6 | 8.4 |
| clone_dtoh_borrowed_dummy_alloc_free | kernel_wait | 9901.0 | 9757.5 | 10006.3 |
| clone_dtoh_borrowed_dummy_alloc_free | clone_dtoh | 22435.3 | 19929.2 | 506621.3 |
| clone_dtoh_borrowed_dummy_alloc_free | d2h_sync | 1.5 | 0.6 | 1.6 |
| clone_dtoh_borrowed_dummy_alloc_free | host_read | 34428.7 | 31024.2 | 38191.3 |
| clone_dtoh_borrowed_dummy_alloc_free | Σ_matmul_equiv | 34553.1 | | |
| pretouched_fresh_split | h2d_a | 1157.0 | 1152.7 | 1527.4 |
| pretouched_fresh_split | h2d_b | 1147.8 | 1146.3 | 1419.0 |
| pretouched_fresh_split | alloc_c | 0.4 | 0.3 | 0.5 |
| pretouched_fresh_split | launch_issue | 7.7 | 7.6 | 8.2 |
| pretouched_fresh_split | kernel_wait | 9958.3 | 9901.8 | 10493.0 |
| pretouched_fresh_split | dest_alloc | 10.4 | 9.4 | 10.5 |
| pretouched_fresh_split | pretouch_fill | 17078.6 | 14999.0 | 20170.5 |
| pretouched_fresh_split | d2h_issue | 1178.3 | 1156.6 | 1187.5 |
| pretouched_fresh_split | d2h_sync | 1.5 | 1.4 | 1.5 |
| pretouched_fresh_split | host_read | 8750.2 | 8624.6 | 8756.8 |
| pretouched_fresh_split | Σ_matmul_equiv | 31124.1 | | |
| pretouched_fresh_production | h2d_a | 1158.9 | 1150.7 | 1172.8 |
| pretouched_fresh_production | h2d_b | 1146.6 | 1144.8 | 1156.6 |
| pretouched_fresh_production | alloc_c | 0.4 | 0.3 | 0.5 |
| pretouched_fresh_production | launch_issue | 7.9 | 7.7 | 8.1 |
| pretouched_fresh_production | kernel_wait | 10202.7 | 9878.8 | 10270.8 |
| pretouched_fresh_production | readback_total | 21298.1 | 20560.5 | 21521.5 |
| pretouched_fresh_production | host_read | 8750.7 | 8657.5 | 8754.7 |
| pretouched_fresh_production | Σ_matmul_equiv | 33738.1 | | |
| prod_order_fresh | h2d_a | 1151.7 | 1151.3 | 1158.3 |
| prod_order_fresh | h2d_b | 1142.2 | 1137.6 | 1145.2 |
| prod_order_fresh | alloc_c | 0.4 | 0.4 | 0.5 |
| prod_order_fresh | launch_issue | 7.9 | 7.6 | 8.1 |
| prod_order_fresh | readback_total | 26759.1 | 21838.4 | 27483.8 |
| prod_order_fresh | host_read | 8670.2 | 8658.1 | 8749.9 |
| prod_order_fresh | Σ_matmul_equiv | 29062.6 | | |
| prod_order_reused | h2d_a | 1145.1 | 1144.4 | 1146.4 |
| prod_order_reused | h2d_b | 1142.5 | 1140.9 | 1144.2 |
| prod_order_reused | alloc_c | 0.1 | 0.1 | 0.1 |
| prod_order_reused | launch_issue | 2.1 | 2.1 | 2.2 |
| prod_order_reused | d2h_issue | 11435.8 | 11022.1 | 11528.6 |
| prod_order_reused | d2h_sync | 0.4 | 0.4 | 0.4 |
| prod_order_reused | host_read | 8751.2 | 8666.3 | 8758.0 |
| prod_order_reused | Σ_matmul_equiv | 13725.7 | | |
| Layer A | matmul | 29356.4 | 29259.3 | 29810.9 |

- exposed_alloc_fill = prod_order_fresh.readback_total − prod_order_reused.(d2h_issue+d2h_sync) = 26759.1 − 11436.2 = 15322.9 us
- residual = LayerA.matmul − Σ_matmul_equiv(prod_order_reused) = 15630.7 us
- alloc_fill_share = 0.980（exposed_alloc_fill / residual）
- **判定: 宛先確保・事前タッチへの帰属を支持**
- 判定が示す範囲: 本番順序で表に出た宛先確保・事前タッチ費用（exposed cost）がresidual を占める割合のみ。重なりに隠れた費用・Layer B が再現しない Layer A 側の固定費は検出対象外
- 校正: LayerA.matmul − Σ_matmul_equiv(prod_order_fresh) = 293.7 us（1.0% of LayerA。本番順序の Layer B がLayer A を再現する度合い。記録のみ）
- 重なり: Σ_matmul_equiv(直列 production) − Σ_matmul_equiv(prod_order_fresh) = 4675.5 us（同一 readback 関数の直列/本番順序の壁時計差。記録のみ）
- 直列腕の dest_alloc+pretouch_fill = 17089.0 us（直列区間の値。exposed との差が重なりに隠れた分。判定には使わない・記録のみ）
- 旧 residual（§12.5 基準）= LayerA.matmul − Σ_matmul_equiv(keep_alive 直列) = -6635.6 us（連続性のための記録のみ）
- clone_dtoh 系 readback − split の (d2h_issue+d2h_sync) = 22322.3 us（直列腕同士の比較。記録のみ）
- 直列分解の整合（補助）: |split 分解和 − production readback_total| / production = 0.142（許容 0.05。超過。直列腕同士のみの整合で判定には使わない・記録のみ）
- 非後退 ratio = split.Σreadback / legacy.Σreadback = 0.831（直列腕同士。記録のみ）

