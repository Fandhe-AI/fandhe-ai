| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/reuse | 2.006 ms (min 1.779 ms / max 2.024 ms) | 1.929 ms (min 1.713 ms / max 2.659 ms) | 0.9618 | 完全一致 | 非後退 | 0.9600, 0.9373, 0.9633, 0.9982, 1.3338 | いいえ |

### フェーズ分解（診断用・reuse・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 28.3 us | 28.0 us | 0.989 |
| leaf_register | 0.2 us | 0.3 us | 1.168 |
| forward_resident | 991.1 us | 879.4 us | 0.887 |
| loss_readout | 0.0 us | 0.0 us | 1.000 |
| backward | 1.049 ms | 1.065 ms | 1.015 |
| device_update | 5.0 us | 5.1 us | 1.034 |
| tape_drop | 0.5 us | 0.6 us | 1.078 |
| step_total | 2.094 ms | 1.993 ms | 0.952 |
