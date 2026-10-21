| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 757.3 us (min 751.0 us / max 777.2 us) | 764.9 us (min 753.2 us / max 772.8 us) | 1.0100 | 完全一致 | 後退 | 1.0291, 0.9850, 1.0046, 0.9898, 1.0171 | いいえ |
| 64/reuse | 359.9 us (min 331.6 us / max 419.6 us) | 374.7 us (min 322.6 us / max 474.0 us) | 1.0412 | 完全一致 | 後退 | 1.3145, 0.9929, 1.1299, 0.7690, 1.3173 | いいえ |
