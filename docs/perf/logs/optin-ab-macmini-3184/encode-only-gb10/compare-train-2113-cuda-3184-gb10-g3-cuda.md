| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 509.7 us (min 506.5 us / max 534.3 us) | 513.3 us (min 507.8 us / max 543.3 us) | 1.0071 | 完全一致 | 後退 | 1.0134, 1.0169, 1.0077, 0.9949, 0.9962 | いいえ |
| 64/reuse | 314.3 us (min 313.5 us / max 319.4 us) | 312.5 us (min 312.2 us / max 313.4 us) | 0.9944 | 完全一致 | 非後退 | 0.9986, 0.9957, 0.9943, 0.9912, 0.9784 | いいえ |
