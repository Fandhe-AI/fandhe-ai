| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 1.884 ms (min 1.878 ms / max 1.890 ms) | 1.575 ms (min 1.572 ms / max 1.582 ms) | 0.8362 | 完全一致 | 非後退 | 0.8360, 0.8361, 0.8377, 0.8387, 0.8363 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 11.4 us | 10.5 us | 0.927 |
| leaf_register | 0.1 us | 0.1 us | 0.672 |
| forward_resident | 946.4 us | 1.050 ms | 1.109 |
| loss_readout | 0.0 us | 0.0 us | - |
| backward | 932.2 us | 509.0 us | 0.546 |
| device_update | 2.2 us | 0.5 us | 0.231 |
| tape_drop | 0.2 us | 0.2 us | 1.000 |
| step_total | 1.894 ms | 1.569 ms | 0.829 |
