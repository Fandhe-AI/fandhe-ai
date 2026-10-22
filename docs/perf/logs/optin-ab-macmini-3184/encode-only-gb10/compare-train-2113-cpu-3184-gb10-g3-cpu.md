| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 1.109 ms (min 1.084 ms / max 1.129 ms) | 1.084 ms (min 1.058 ms / max 1.109 ms) | 0.9771 | 完全一致 | 非後退 | 1.0000, 0.9986, 0.9539, 1.0036, 0.9599 | いいえ |
| 64/reuse | 843.8 us (min 807.7 us / max 855.6 us) | 855.4 us (min 825.5 us / max 882.0 us) | 1.0138 | 完全一致 | 後退 | 1.0921, 1.0011, 0.9965, 0.9820, 1.0073 | いいえ |
