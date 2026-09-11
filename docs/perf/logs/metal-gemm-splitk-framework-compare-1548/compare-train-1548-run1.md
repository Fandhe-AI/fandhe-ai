| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 1.439 ms (min 1.391 ms / max 1.686 ms) | 1.546 ms (min 1.372 ms / max 1.668 ms) | 1.0746 | 完全一致 | 後退 | 0.9866, 1.1622, 0.8400, 1.1394, 0.9173 | いいえ |
| 64/reuse | 1.345 ms (min 1.337 ms / max 1.476 ms) | 1.365 ms (min 1.297 ms / max 1.455 ms) | 1.0151 | 完全一致 | 後退 | 0.9661, 1.0016, 1.0209, 1.0160, 0.9853 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 16.2 us | 15.1 us | 0.934 |
| leaf_register | 0.5 us | 0.5 us | 0.923 |
| forward | 732.0 us | 707.9 us | 0.967 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 770.2 us | 760.9 us | 0.988 |
| param_readout | 21.8 us | 21.3 us | 0.977 |
| host_sgd | 31.5 us | 31.2 us | 0.992 |
| apply_params | 0.2 us | 0.2 us | 1.000 |
| tape_drop | 0.6 us | 0.6 us | 0.934 |
| step_total | 1.578 ms | 1.548 ms | 0.980 |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 12.6 us | 13.9 us | 1.104 |
| leaf_register | 0.2 us | 0.1 us | 0.753 |
| forward_resident | 638.7 us | 659.8 us | 1.033 |
| loss_readout | 0.0 us | 0.0 us | 1.024 |
| backward | 693.6 us | 735.5 us | 1.060 |
| device_update | 62.0 us | 63.2 us | 1.019 |
| tape_drop | 0.2 us | 0.3 us | 1.164 |
| step_total | 1.401 ms | 1.477 ms | 1.054 |
