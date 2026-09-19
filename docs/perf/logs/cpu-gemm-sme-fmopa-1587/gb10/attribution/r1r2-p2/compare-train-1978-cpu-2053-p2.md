| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 1.033 ms (min 1.011 ms / max 1.118 ms) | 1.002 ms (min 945.3 us / max 1.069 ms) | 0.9698 | 完全一致 | 非後退 | 0.8965, 0.9585, 0.9148, 1.0579, 0.9987 | いいえ |
| 64/reuse | 1.041 ms (min 972.9 us / max 1.067 ms) | 1.081 ms (min 993.1 us / max 1.113 ms) | 1.0388 | 完全一致 | 後退 | 1.1114, 0.9541, 1.0218, 1.0569, 1.0501 | いいえ |
