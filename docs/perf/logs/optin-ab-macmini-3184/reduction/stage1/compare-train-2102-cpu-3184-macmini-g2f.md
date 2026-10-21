| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 724.2 us (min 719.3 us / max 725.2 us) | 724.0 us (min 716.2 us / max 729.7 us) | 0.9997 | 完全一致 | 非後退 | 1.0035, 0.9875, 1.0076, 0.9986, 1.0060 | いいえ |
| 64/reuse | 705.2 us (min 700.1 us / max 717.2 us) | 700.4 us (min 688.9 us / max 705.2 us) | 0.9933 | 完全一致 | 非後退 | 0.9860, 1.0005, 1.0040, 0.9956, 0.9604 | いいえ |
