# #3189 GB10 集計（正式系列）

- 負荷ゲート 15/15 pass（run 5 × armA／infer／fixedcost）

## 腕 A: train／infer `--phases`（registry `fandhe-ai =0.11.0`）

- 系列区分: **正式系列（cuda）**。cpu は参考

### train_phases / cuda / fresh

| phase | run1..5 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 3.1 | 2.9–3.7 | 0.6% |
| leaf_register | 1.7 | 1.2–2.0 | 0.3% |
| forward | 176.2 | 174.0–178.2 | 34.5% |
| loss_readout | 0.0 | 0.0–0.0 | 0.0% |
| backward | 207.5 | 190.2–217.6 | 40.6% |
| param_readout | 66.9 | 65.7–91.0 | 13.1% |
| host_sgd | 52.1 | 49.9–64.4 | 10.2% |
| apply_params | 0.3 | 0.3–0.3 | 0.1% |
| tape_drop | 2.1 | 1.7–2.2 | 0.4% |
| step_total | 511.1 | 508.6–541.3 | 100.0% |

### train_phases / cuda / reuse

| phase | run1..5 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 2.4 | 2.3–2.7 | 0.7% |
| leaf_register | 0.3 | 0.3–0.4 | 0.1% |
| forward_resident | 154.8 | 153.2–155.6 | 46.6% |
| loss_readout | 0.0 | 0.0–0.0 | 0.0% |
| backward | 165.0 | 164.7–166.4 | 49.7% |
| device_update | 7.9 | 7.8–8.0 | 2.4% |
| tape_drop | 0.9 | 0.8–0.9 | 0.3% |
| step_total | 332.3 | 330.6–334.2 | 100.0% |

### infer_phases / cuda / fresh

| phase | run1..5 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| leaf_register | 0.2 | 0.2–0.3 | 0.2% |
| forward | 133.7 | 129.9–136.8 | 86.6% |
| to_tensor | 19.0 | 18.5–19.4 | 12.3% |
| host_copy | 0.1 | 0.1–0.1 | 0.1% |
| checksum | 0.6 | 0.6–0.7 | 0.4% |
| iter_total | 154.4 | 149.8–157.4 | 100.0% |

### infer_phases / cuda / reuse

| phase | run1..5 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict_resident | 98.7 | 98.2–99.0 | 99.0% |
| host_copy | 0.1 | 0.1–0.1 | 0.1% |
| checksum | 0.8 | 0.7–0.9 | 0.8% |
| iter_total | 99.7 | 99.2–100.1 | 100.0% |

### train_phases / cpu / fresh

| phase | run1..5 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 0.3 | 0.3–0.3 | 0.0% |
| leaf_register | 2.5 | 2.4–2.8 | 0.3% |
| forward | 232.5 | 225.4–245.6 | 24.6% |
| loss_readout | 0.2 | 0.2–0.2 | 0.0% |
| backward | 449.2 | 426.0–473.7 | 47.6% |
| param_readout | 41.4 | 38.0–46.4 | 4.4% |
| host_sgd | 58.6 | 52.6–67.9 | 6.2% |
| apply_params | 1.0 | 0.9–1.1 | 0.1% |
| tape_drop | 4.5 | 4.5–4.7 | 0.5% |
| step_total | 944.1 | 931.5–1002.0 | 100.0% |

### train_phases / cpu / reuse

| phase | run1..5 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| tape_build | 0.2 | 0.2–0.2 | 0.0% |
| leaf_register | 0.8 | 0.8–0.9 | 0.1% |
| forward_resident | 215.3 | 203.6–223.5 | 22.2% |
| loss_readout | 0.1 | 0.1–0.2 | 0.0% |
| backward | 439.5 | 425.8–447.5 | 45.3% |
| device_update | 289.0 | 285.7–293.2 | 29.8% |
| tape_drop | 2.1 | 2.0–2.3 | 0.2% |
| step_total | 970.7 | 929.9–984.3 | 100.0% |

### infer_phases / cpu / fresh

| phase | run1..5 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict | 174.1 | 162.7–183.6 | 99.4% |
| host_copy | 0.1 | 0.1–0.2 | 0.1% |
| checksum | 0.3 | 0.3–0.3 | 0.2% |
| iter_total | 175.1 | 164.1–184.6 | 100.0% |

### infer_phases / cpu / reuse

| phase | run1..5 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict_resident | 182.9 | 169.4–185.7 | 99.4% |
| host_copy | 0.1 | 0.1–0.2 | 0.1% |
| checksum | 0.3 | 0.3–0.3 | 0.2% |
| iter_total | 183.9 | 170.0–186.9 | 100.0% |

## 腕 B-2: infer reuse の内部内訳

### 上位（facade。decomposed ＋ paired/residual。5 run 中央値）

| arm | phase | 中央値 (µs) | min–max (µs) |
|---|---|---:|---|
| decomposed | chain | 95.9 | 95.7–96.7 |
| decomposed | readout | 0.6 | 0.4–0.8 |
| decomposed | snapshot | 0.2 | 0.2–0.2 |
| decomposed | steps | 0.0 | 0.0–0.0 |
| decomposed | tape | 0.0 | 0.0–0.0 |
| paired | residual | 2.6 | 2.5–2.7 |
| public | predict_resident | 99.3 | 99.2–99.8 |

### 下位（backend-cuda。5 run 中央値）

| phase | 中央値 (µs) | min–max (µs) |
|---|---:|---|
| download | 69.2 | 69.1–69.5 |
| download_again | 6.1 | 5.8–6.4 |
| h2d_upload | 11.3 | 11.0–11.4 |
| kernel_wait_derived | 63.2 | 62.7–63.5 |
| l1_issue | 7.5 | 7.4–7.5 |
| l2_issue | 6.6 | 6.6–6.7 |
| total | 94.5 | 94.2–95.0 |

- fidelity = backend total / facade chain = 0.986（許容 0.8〜1.2）→ 区間は正式扱い
- 上位の固定費 = tape＋snapshot＋steps＋paired/residual = 2.9 µs
- public predict_resident = 99.3 µs（decomposed 合計 96.8 µs ＋ residual 2.6 µs）
- fidelity(上位) = decomposed 合計 / public predict_resident = 0.975（許容 0.8〜1.2）→ 上位区間は正式扱い
- 腕 A の infer reuse iter_total = 99.7 µs（版差を含む。腕をまたぐ比較）

## 腕 B-3: launch／同期の固定費（N=128。#2109 の診断を計測器として借用）

| layer | phase | 中央値 (µs) | min–max (µs) |
|---|---|---:|---|
| E | sync_idle | 0.29 | 0.3–0.3 |
| E | tiny_roundtrip | 6.14 | 5.4–6.2 |
| E | event_create_drop | 0.18 | 0.2–0.2 |
| L2 | launch_issue | 1.82 | 1.7–2.2 |
| L2S | kernel_wait | 9.65 | 9.4–9.7 |

- tiny_roundtrip は 1×1×1 GEMM の launch＋同期で、引数なしの空カーネルの代替（RULE 規則 5）

## 腕 B-4: unified memory 属性（unified_memory_probe。決定的な値・1 回）

- device_name: `NVIDIA GB10`

| 属性（CU_DEVICE_ATTRIBUTE_*） | 値 |
|---|---:|
| MANAGED_MEMORY | 1 |
| CONCURRENT_MANAGED_ACCESS | 1 |
| PAGEABLE_MEMORY_ACCESS | 1 |
| PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES | 1 |
| DIRECT_MANAGED_MEM_ACCESS_FROM_HOST | 0 |
| HOST_NATIVE_ATOMIC_SUPPORTED | 1 |
| INTEGRATED | 1 |
| CAN_MAP_HOST_MEMORY | 1 |
| CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM | 1 |

## 腕 B-1: 1 step あたりの回数（nsys・K=20／120 差分法）

| window | sync | launch | htod | dtoh | htod_sync | dtoh_sync | alloc | free | wait_event | 決定性 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fwd | 3 | 4 | 3 | 3 | 0 | 0 | 5 | 5 | 10 | 決定的 |
| fwd_bwd | 8 | 15 | 10 | 6 | 0 | 0 | 14 | 14 | 28 | 決定的 |
| full | 8 | 16 | 10 | 6 | 0 | 0 | 14 | 14 | 28 | 決定的 |
| infer_reuse | 1 | 2 | 1 | 1 | 0 | 0 | 3 | 3 | 6 | 決定的 |

### 区間への割り当て（RULE 規則 2）

| 区間 | sync | launch | htod | dtoh | htod（非 async） | dtoh（非 async） |
|---|---:|---:|---:|---:|---:|---:|
| forward | 3 | 4 | 3 | 3 | 0 | 0 |
| backward | 5 | 11 | 7 | 3 | 0 | 0 |
| update＋tape_drop | 0 | 1 | 0 | 0 | 0 | 0 |
| full（1 step） | 8 | 16 | 10 | 6 | 0 | 0 |
| infer_reuse | 1 | 2 | 1 | 1 | 0 | 0 |

## 帰属と見込み（record-only。原因の断定はしない）

- train reuse step_total = 332.3 µs、infer reuse iter_total = 99.7 µs、差 = 232.6 µs
- 同期回数: train full = 8、infer = 1

| c の種別 | c (µs) | (n_full−1)×c (µs) | step_total に対する説明率 | 未説明残差 (µs) |
|---|---:|---:|---:|---:|
| sync_idle | 0.29 | 2.0 | 0.6% | 330.3 |
| tiny_roundtrip | 6.14 | 43.0 | 12.9% | 289.3 |
| 実カーネル同期(kernel_wait_derived) | 63.18 | 442.3 | 133.1% | -110.0 |

- launch 固定費の上限 (launch_count−1)×launch_issue = (16−1)×1.82 = 27.4 µs（step_total の 8.2%）
- 1 同期あたりの外挿 (step_total−iter_total)/(n_full−n_infer) = 33.2 µs（実測固定費と比べる。約 27 µs の再計算）

