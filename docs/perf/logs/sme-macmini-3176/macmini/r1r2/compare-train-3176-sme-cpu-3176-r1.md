| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 717.7 us (min 708.5 us / max 721.6 us) | 646.6 us (min 640.1 us / max 652.8 us) | 0.9009 | 完全一致 | 非後退 | 0.8986, 0.9215, 0.8986, 0.9009, 0.8915 | いいえ |
| 64/reuse | 692.4 us (min 676.2 us / max 700.5 us) | 631.2 us (min 630.6 us / max 636.2 us) | 0.9117 | 完全一致 | 非後退 | 0.9095, 0.9116, 0.9227, 0.9011, 0.9373 | いいえ |
