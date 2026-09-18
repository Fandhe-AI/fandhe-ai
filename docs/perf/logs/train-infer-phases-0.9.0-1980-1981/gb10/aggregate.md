# train／infer `--phases` 5 run 中央値（registry `fandhe-ai =0.9.0`・DGX Spark GB10）

## infer_phases / cpu / fresh

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict | 176.3 | 167.1–184.6 | 99.5% |
| host_copy | 0.1 | 0.1–0.2 | 0.1% |
| checksum | 0.3 | 0.3–0.5 | 0.2% |
| iter_total | 177.1 | 167.8–186.1 | 100.0% |

- トップ 3: predict（99.5%・176.3 µs）・checksum（0.2%・0.3 µs）・host_copy（0.1%・0.1 µs）
- フェーズ和／合計: 99.8%（差分は各フェーズ中央値の非加法性を含み、固定費は未測定）

## infer_phases / cpu / reuse

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict_resident | 195.0 | 167.6–195.6 | 99.5% |
| host_copy | 0.2 | 0.1–0.2 | 0.1% |
| checksum | 0.3 | 0.3–0.3 | 0.2% |
| iter_total | 196.0 | 168.2–196.2 | 100.0% |

- トップ 3: predict_resident（99.5%・195.0 µs）・checksum（0.2%・0.3 µs）・host_copy（0.1%・0.2 µs）
- フェーズ和／合計: 99.7%（差分は各フェーズ中央値の非加法性を含み、固定費は未測定）

## train_phases / cpu / fresh

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 0.3 | 0.3–0.3 | 0.0% |
| leaf_register | 1.9 | 1.8–2.0 | 0.2% |
| forward | 252.0 | 240.6–260.8 | 22.3% |
| loss_readout | 0.1 | 0.1–0.2 | 0.0% |
| backward | 473.2 | 425.3–510.7 | 41.9% |
| param_readout | 40.2 | 37.6–44.2 | 3.6% |
| host_sgd | 65.4 | 61.4–68.5 | 5.8% |
| apply_params | 0.8 | 0.8–0.9 | 0.1% |
| tape_drop | 4.4 | 4.3–5.0 | 0.4% |
| step_total | 1130.5 | 1112.5–1178.3 | 100.0% |

- トップ 3: backward（41.9%・473.2 µs）・forward（22.3%・252.0 µs）・host_sgd（5.8%・65.4 µs）
- フェーズ和／合計: 74.2%（差分は各フェーズ中央値の非加法性を含み、固定費は未測定）

## train_phases / cpu / reuse

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 0.2 | 0.2–0.3 | 0.0% |
| leaf_register | 0.8 | 0.7–0.9 | 0.1% |
| forward_resident | 256.9 | 223.6–289.9 | 23.8% |
| loss_readout | 0.1 | 0.1–0.2 | 0.0% |
| backward | 482.7 | 406.9–518.0 | 44.7% |
| device_update | 277.5 | 273.8–280.2 | 25.7% |
| tape_drop | 2.0 | 1.9–2.0 | 0.2% |
| step_total | 1080.6 | 970.1–1140.7 | 100.0% |

- トップ 3: backward（44.7%・482.7 µs）・device_update（25.7%・277.5 µs）・forward_resident（23.8%・256.9 µs）
- フェーズ和／合計: 94.4%（差分は各フェーズ中央値の非加法性を含み、固定費は未測定）

## infer_phases / cuda / fresh

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| leaf_register | 0.2 | 0.2–0.3 | 0.2% |
| forward | 138.1 | 133.0–139.4 | 87.2% |
| to_tensor | 19.0 | 18.7–19.3 | 12.0% |
| host_copy | 0.1 | 0.1–0.1 | 0.1% |
| checksum | 0.6 | 0.6–0.8 | 0.4% |
| iter_total | 158.4 | 153.0–160.0 | 100.0% |

- トップ 3: forward（87.2%・138.1 µs）・to_tensor（12.0%・19.0 µs）・checksum（0.4%・0.6 µs）
- フェーズ和／合計: 99.9%（差分は各フェーズ中央値の非加法性を含み、固定費は未測定）

## infer_phases / cuda / reuse

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict_resident | 98.8 | 97.9–99.1 | 98.9% |
| host_copy | 0.1 | 0.1–0.1 | 0.1% |
| checksum | 0.8 | 0.8–0.9 | 0.8% |
| iter_total | 100.0 | 99.0–100.1 | 100.0% |

- トップ 3: predict_resident（98.9%・98.8 µs）・checksum（0.8%・0.8 µs）・host_copy（0.1%・0.1 µs）
- フェーズ和／合計: 99.8%（差分は各フェーズ中央値の非加法性を含み、固定費は未測定）

## train_phases / cuda / fresh

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 3.0 | 3.0–3.3 | 0.6% |
| leaf_register | 1.6 | 1.6–1.7 | 0.3% |
| forward | 176.5 | 174.9–176.7 | 34.1% |
| loss_readout | 0.0 | 0.0–0.0 | 0.0% |
| backward | 207.5 | 188.0–216.4 | 40.1% |
| param_readout | 87.0 | 66.8–92.1 | 16.8% |
| host_sgd | 54.4 | 46.2–65.0 | 10.5% |
| apply_params | 0.3 | 0.3–0.3 | 0.1% |
| tape_drop | 2.1 | 1.9–2.1 | 0.4% |
| step_total | 518.0 | 515.0–539.7 | 100.0% |

- トップ 3: backward（40.1%・207.5 µs）・forward（34.1%・176.5 µs）・param_readout（16.8%・87.0 µs）
- フェーズ和／合計: 102.8%（差分は各フェーズ中央値の非加法性を含み、固定費は未測定）

## train_phases / cuda / reuse

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 2.5 | 2.4–2.6 | 0.8% |
| leaf_register | 0.3 | 0.3–0.3 | 0.1% |
| forward_resident | 155.9 | 154.2–156.8 | 49.1% |
| loss_readout | 0.0 | 0.0–0.0 | 0.0% |
| backward | 149.1 | 147.2–149.3 | 46.9% |
| device_update | 8.0 | 7.9–8.1 | 2.5% |
| tape_drop | 0.9 | 0.8–1.0 | 0.3% |
| step_total | 317.5 | 313.6–317.7 | 100.0% |

- トップ 3: forward_resident（49.1%・155.9 µs）・backward（46.9%・149.1 µs）・device_update（2.5%・8.0 µs）
- フェーズ和／合計: 99.7%（差分は各フェーズ中央値の非加法性を含み、固定費は未測定）

