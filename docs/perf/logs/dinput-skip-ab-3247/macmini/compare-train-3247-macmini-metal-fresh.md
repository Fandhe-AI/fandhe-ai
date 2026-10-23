| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 2.336 ms (min 2.323 ms / max 2.809 ms) | 2.331 ms (min 2.327 ms / max 2.340 ms) | 0.9975 | 完全一致 | 非後退 | 0.8284, 0.9947, 1.0036, 0.9967, 1.0004 | いいえ |

### フェーズ分解（診断用・fresh・単発計測・非判定）

| phase | before | after | after/before |
|---|---|---|---|
| tape_build | 13.9 us | 14.2 us | 1.025 |
| leaf_register | 0.5 us | 0.6 us | 1.076 |
| forward | 939.6 us | 959.6 us | 1.021 |
| loss_readout | 0.0 us | 0.0 us | 0.000 |
| backward | 1.268 ms | 1.293 ms | 1.019 |
| param_readout | 20.5 us | 21.3 us | 1.043 |
| host_sgd | 34.5 us | 34.2 us | 0.993 |
| apply_params | 0.2 us | 0.2 us | 1.000 |
| tape_drop | 0.4 us | 0.4 us | 1.000 |
| step_total | 2.280 ms | 2.321 ms | 1.018 |
