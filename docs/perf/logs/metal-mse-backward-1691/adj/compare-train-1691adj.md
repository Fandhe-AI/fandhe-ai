| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 2.082 ms (min 1.648 ms / max 2.538 ms) | 1.990 ms (min 1.767 ms / max 2.178 ms) | 0.9561 | 完全一致 | 非後退（判定注意: before spread > 1.5x・負荷ノイズの疑い） | 0.9961, 0.8530, 0.8489, 1.2006, 0.9676 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 20.7 us | 18.0 us | 0.872 |
| leaf_register | 0.2 us | 0.2 us | 1.246 |
| forward_resident | 999.8 us | 784.4 us | 0.785 |
| loss_readout | 0.0 us | 0.0 us | 1.024 |
| backward | 1.078 ms | 823.9 us | 0.764 |
| device_update | 3.8 us | 3.8 us | 1.005 |
| tape_drop | 0.4 us | 0.4 us | 1.000 |
| step_total | 2.123 ms | 1.616 ms | 0.761 |
