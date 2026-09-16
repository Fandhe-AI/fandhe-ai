| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 1.767 ms (min 1.675 ms / max 1.797 ms) | 1.656 ms (min 1.583 ms / max 1.668 ms) | 0.9373 | 完全一致 | 非後退 | 0.9912, 0.9259, 0.8960, 0.9283, 0.9138 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 26.7 us | 25.2 us | 0.942 |
| leaf_register | 0.2 us | 0.2 us | 0.832 |
| forward_resident | 870.1 us | 873.4 us | 1.004 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 607.2 us | 702.8 us | 1.157 |
| device_update | 213.2 us | 4.8 us | 0.022 |
| tape_drop | 0.5 us | 0.4 us | 0.910 |
| step_total | 1.751 ms | 1.620 ms | 0.925 |
