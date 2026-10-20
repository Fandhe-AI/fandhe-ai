# train／infer `--phases` 5 run 中央値（registry `fandhe-ai =0.11.0`・Apple M4 (Mac mini)）

- 系列区分: **正式系列**（負荷ゲート 10/10 pass（run 5 + 同期固定費 5））

## infer_phases / cpu / fresh

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict | 201.6 | 196.7–206.1 | 99.7% |
| host_copy | 0.0 | 0.0–0.0 | 0.0% |
| checksum | 0.3 | 0.3–0.4 | 0.2% |
| iter_total | 202.2 | 197.2–207.0 | 100.0% |

- トップ 3: predict（99.7%・201.6 µs）・checksum（0.2%・0.3 µs）・host_copy（0.0%・0.0 µs）
- フェーズ和／合計: 99.9%（差分は各フェーズ中央値の非加法性を含む）

## infer_phases / cpu / reuse

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict_resident | 194.0 | 191.8–202.5 | 99.7% |
| host_copy | 0.0 | 0.0–0.0 | 0.0% |
| checksum | 0.3 | 0.3–0.3 | 0.2% |
| iter_total | 194.5 | 192.3–203.0 | 100.0% |

- トップ 3: predict_resident（99.7%・194.0 µs）・checksum（0.2%・0.3 µs）・host_copy（0.0%・0.0 µs）
- フェーズ和／合計: 99.9%（差分は各フェーズ中央値の非加法性を含む）

## train_phases / cpu / fresh

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 0.0 | 0.0–0.0 | 0.0% |
| leaf_register | 0.6 | 0.6–0.6 | 0.1% |
| forward | 210.7 | 203.5–221.9 | 29.3% |
| loss_readout | 0.0 | 0.0–0.0 | 0.0% |
| backward | 440.8 | 433.2–447.4 | 61.3% |
| param_readout | 23.4 | 23.2–25.2 | 3.3% |
| host_sgd | 34.5 | 34.3–39.5 | 4.8% |
| apply_params | 0.2 | 0.2–0.3 | 0.0% |
| tape_drop | 0.6 | 0.6–0.7 | 0.1% |
| step_total | 719.1 | 700.1–736.4 | 100.0% |

- トップ 3: backward（61.3%・440.8 µs）・forward（29.3%・210.7 µs）・host_sgd（4.8%・34.5 µs）
- フェーズ和／合計: 98.9%（差分は各フェーズ中央値の非加法性を含む）

## train_phases / cpu / reuse

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 0.0 | 0.0–0.0 | 0.0% |
| leaf_register | 0.2 | 0.1–0.2 | 0.0% |
| forward_resident | 206.4 | 183.4–208.2 | 26.5% |
| loss_readout | 0.0 | 0.0–0.0 | 0.0% |
| backward | 451.5 | 401.0–461.7 | 58.0% |
| device_update | 117.6 | 116.0–118.7 | 15.1% |
| tape_drop | 0.2 | 0.2–0.4 | 0.0% |
| step_total | 778.4 | 701.5–792.5 | 100.0% |

- トップ 3: backward（58.0%・451.5 µs）・forward_resident（26.5%・206.4 µs）・device_update（15.1%・117.6 µs）
- フェーズ和／合計: 99.7%（差分は各フェーズ中央値の非加法性を含む）

## infer_phases / metal / fresh

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| leaf_register | 0.1 | 0.1–0.1 | 0.0% |
| forward | 570.9 | 566.7–572.9 | 75.6% |
| to_tensor | 183.7 | 182.8–184.9 | 24.3% |
| host_copy | 0.1 | 0.1–0.2 | 0.0% |
| checksum | 0.3 | 0.3–0.3 | 0.0% |
| iter_total | 754.9 | 750.2–759.1 | 100.0% |

- トップ 3: forward（75.6%・570.9 µs）・to_tensor（24.3%・183.7 µs）・checksum（0.0%・0.3 µs）
- フェーズ和／合計: 100.0%（差分は各フェーズ中央値の非加法性を含む）

## infer_phases / metal / reuse

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict_resident | 330.5 | 329.1–338.5 | 99.8% |
| host_copy | 0.2 | 0.1–0.2 | 0.1% |
| checksum | 0.3 | 0.3–0.3 | 0.1% |
| iter_total | 331.0 | 329.8–339.1 | 100.0% |

- トップ 3: predict_resident（99.8%・330.5 µs）・checksum（0.1%・0.3 µs）・host_copy（0.1%・0.2 µs）
- フェーズ和／合計: 100.0%（差分は各フェーズ中央値の非加法性を含む）

## train_phases / metal / fresh

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 14.2 | 14.0–14.9 | 0.6% |
| leaf_register | 0.5 | 0.5–0.6 | 0.0% |
| forward | 958.3 | 951.2–1193.6 | 41.2% |
| loss_readout | 0.0 | 0.0–0.0 | 0.0% |
| backward | 1293.1 | 1273.7–1566.9 | 55.6% |
| param_readout | 21.7 | 20.4–22.2 | 0.9% |
| host_sgd | 35.9 | 35.6–37.0 | 1.5% |
| apply_params | 0.2 | 0.2–0.2 | 0.0% |
| tape_drop | 0.5 | 0.4–0.5 | 0.0% |
| step_total | 2324.4 | 2295.7–2782.0 | 100.0% |

- トップ 3: backward（55.6%・1293.1 µs）・forward（41.2%・958.3 µs）・host_sgd（1.5%・35.9 µs）
- フェーズ和／合計: 100.0%（差分は各フェーズ中央値の非加法性を含む）

## train_phases / metal / reuse

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 11.4 | 11.1–11.5 | 0.6% |
| leaf_register | 0.1 | 0.1–0.1 | 0.0% |
| forward_resident | 945.4 | 939.5–947.9 | 50.1% |
| loss_readout | 0.0 | 0.0–0.0 | 0.0% |
| backward | 928.4 | 921.4–931.4 | 49.2% |
| device_update | 2.2 | 2.2–2.3 | 0.1% |
| tape_drop | 0.2 | 0.2–0.2 | 0.0% |
| step_total | 1885.9 | 1876.1–1890.9 | 100.0% |

- トップ 3: forward_resident（50.1%・945.4 µs）・backward（49.2%・928.4 µs）・tape_build（0.6%・11.4 µs）
- フェーズ和／合計: 100.1%（差分は各フェーズ中央値の非加法性を含む）

# 同期回数と同期 1 回の固定費（Mac mini）

## 同期 1 回の固定費（空 dispatch の壁時間。cb+1/wait+1 の構造検査通過）

| run | 中央値 (µs) |
|---|---:|
| run1 | 25.041 |
| run2 | 11.542 |
| run3 | 11.833 |
| run4 | 14.958 |
| run5 | 14.542 |

- 5 run 中央値: **14.542 µs**（min–max 11.542–25.041 µs）

## 同期回数（steady-state）

- 1 step 全体: encode/command_buffer/wait = 11/7/7（設計 doc の机上導出 11/7/7 と一致）
- backward 窓（5 試行）: 5/3/3（期待 5/3/3 と全試行一致）
- 差し引き（forward＋loss＋update 窓）: 6/4/4

## 同期による説明量（Metal reuse・参考。原因断定はしない）

- wait 7 回 × 14.5 µs = 101.8 µs は、Mini の train Metal reuse step_total 1885.9 µs の **5.4%**
- 未説明の残差: 1784.1 µs（94.6%）
- 外挿: (train reuse − infer reuse) / (wait 7 − 1) = 259.1 µs／同期（実測固定費 14.5 µs と比較。infer の wait は診断テストの再現経路の値）
