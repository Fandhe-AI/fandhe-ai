| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 978.1 us (min 868.7 us / max 1.008 ms) | 958.4 us (min 911.3 us / max 1.013 ms) | 0.9799 | 完全一致 | 非後退 | 1.0117, 0.9799, 1.0868, 0.9608, 0.9537 | いいえ |
| 64/reuse | 711.6 us (min 665.8 us / max 940.0 us) | 742.1 us (min 673.2 us / max 767.6 us) | 1.0429 | 完全一致 | 後退 | 0.7895, 0.9393, 1.0640, 1.0571, 1.1165 | いいえ |
