# train／infer `--phases` 5 run 中央値（registry `fandhe-ai =0.9.0`・Apple M4 Max）

## infer_phases / cpu / fresh

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict | 184.9 | 158.1–220.2 | 99.7% |
| host_copy | 0.0 | 0.0–0.1 | 0.0% |
| checksum | 0.4 | 0.3–0.4 | 0.2% |
| iter_total | 185.5 | 158.6–220.9 | 100.0% |

- トップ 3: predict（99.7%・184.9 µs）・checksum（0.2%・0.4 µs）・host_copy（0.0%・0.0 µs）
- フェーズ和／合計: 99.9%（差分は計測区間外の固定費）

## infer_phases / cpu / reuse

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict_resident | 175.3 | 163.9–199.5 | 99.6% |
| host_copy | 0.0 | 0.0–0.0 | 0.0% |
| checksum | 0.3 | 0.3–0.3 | 0.2% |
| iter_total | 176.0 | 164.5–200.1 | 100.0% |

- トップ 3: predict_resident（99.6%・175.3 µs）・checksum（0.2%・0.3 µs）・host_copy（0.0%・0.0 µs）
- フェーズ和／合計: 99.8%（差分は計測区間外の固定費）

## train_phases / cpu / fresh

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 0.1 | 0.1–0.1 | 0.0% |
| leaf_register | 0.7 | 0.5–0.7 | 0.1% |
| forward | 235.9 | 223.4–246.8 | 30.0% |
| loss_readout | 0.0 | 0.0–0.0 | 0.0% |
| backward | 476.3 | 470.0–508.3 | 60.6% |
| param_readout | 23.1 | 23.0–23.3 | 2.9% |
| host_sgd | 33.6 | 33.2–34.7 | 4.3% |
| apply_params | 0.3 | 0.3–0.4 | 0.0% |
| tape_drop | 1.0 | 1.0–1.0 | 0.1% |
| step_total | 785.9 | 780.0–800.8 | 100.0% |

- トップ 3: backward（60.6%・476.3 µs）・forward（30.0%・235.9 µs）・host_sgd（4.3%・33.6 µs）
- フェーズ和／合計: 98.1%（差分は計測区間外の固定費）

## train_phases / cpu / reuse

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 0.1 | 0.1–0.2 | 0.0% |
| leaf_register | 0.2 | 0.1–0.2 | 0.0% |
| forward_resident | 236.8 | 230.6–245.1 | 26.9% |
| loss_readout | 0.0 | 0.0–0.0 | 0.0% |
| backward | 520.7 | 511.5–564.0 | 59.0% |
| device_update | 122.3 | 121.3–123.0 | 13.9% |
| tape_drop | 0.4 | 0.4–0.5 | 0.0% |
| step_total | 881.9 | 868.1–931.6 | 100.0% |

- トップ 3: backward（59.0%・520.7 µs）・forward_resident（26.9%・236.8 µs）・device_update（13.9%・122.3 µs）
- フェーズ和／合計: 99.9%（差分は計測区間外の固定費）

## infer_phases / metal / fresh

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| leaf_register | 0.1 | 0.1–0.1 | 0.0% |
| forward | 431.6 | 311.9–507.6 | 80.2% |
| to_tensor | 110.2 | 88.9–115.4 | 20.5% |
| host_copy | 0.2 | 0.2–0.2 | 0.0% |
| checksum | 0.4 | 0.3–0.4 | 0.1% |
| iter_total | 538.2 | 398.5–634.0 | 100.0% |

- トップ 3: forward（80.2%・431.6 µs）・to_tensor（20.5%・110.2 µs）・checksum（0.1%・0.4 µs）
- フェーズ和／合計: 100.8%（差分は計測区間外の固定費）

## infer_phases / metal / reuse

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict_resident | 365.2 | 353.6–369.3 | 99.8% |
| host_copy | 0.2 | 0.2–0.2 | 0.1% |
| checksum | 0.4 | 0.4–0.4 | 0.1% |
| iter_total | 366.1 | 354.4–370.0 | 100.0% |

- トップ 3: predict_resident（99.8%・365.2 µs）・checksum（0.1%・0.4 µs）・host_copy（0.1%・0.2 µs）
- フェーズ和／合計: 99.9%（差分は計測区間外の固定費）

## train_phases / metal / fresh

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 15.7 | 15.2–19.9 | 1.0% |
| leaf_register | 0.5 | 0.5–0.7 | 0.0% |
| forward | 668.4 | 622.6–794.8 | 41.9% |
| loss_readout | 0.0 | 0.0–0.0 | 0.0% |
| backward | 854.3 | 768.8–1008.9 | 53.5% |
| param_readout | 20.8 | 20.1–21.8 | 1.3% |
| host_sgd | 30.0 | 30.0–34.3 | 1.9% |
| apply_params | 0.2 | 0.2–0.3 | 0.0% |
| tape_drop | 0.6 | 0.6–0.8 | 0.0% |
| step_total | 1596.6 | 1474.2–1914.5 | 100.0% |

- トップ 3: backward（53.5%・854.3 µs）・forward（41.9%・668.4 µs）・host_sgd（1.9%・30.0 µs）
- フェーズ和／合計: 99.6%（差分は計測区間外の固定費）

## train_phases / metal / reuse

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 14.9 | 14.5–16.0 | 1.4% |
| leaf_register | 0.2 | 0.1–0.2 | 0.0% |
| forward_resident | 516.3 | 500.1–522.2 | 48.6% |
| loss_readout | 0.0 | 0.0–0.0 | 0.0% |
| backward | 524.6 | 505.8–532.3 | 49.4% |
| device_update | 3.4 | 3.4–3.6 | 0.3% |
| tape_drop | 0.4 | 0.4–0.4 | 0.0% |
| step_total | 1061.6 | 1039.1–1077.4 | 100.0% |

- トップ 3: backward（49.4%・524.6 µs）・forward_resident（48.6%・516.3 µs）・tape_build（1.4%・14.9 µs）
- フェーズ和／合計: 99.8%（差分は計測区間外の固定費）

