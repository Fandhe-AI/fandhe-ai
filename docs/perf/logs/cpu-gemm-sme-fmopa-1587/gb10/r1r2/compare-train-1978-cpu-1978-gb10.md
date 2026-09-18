| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） | 符号一貫（全 run >1.00） |
|---|---|---|---|---|---|---|---|
| 64/fresh | 1.116 ms (min 1.102 ms / max 1.163 ms) | 992.9 us (min 949.1 us / max 1.045 ms) | 0.8897 | 完全一致 | 非後退 | 0.8815, 0.8342, 0.9486, 0.8526, 0.8897 | いいえ |
| 64/reuse | 1.038 ms (min 955.5 us / max 1.173 ms) | 969.8 us (min 940.9 us / max 1.091 ms) | 0.9346 | 完全一致 | 非後退 | 0.8021, 0.8925, 1.0509, 1.0783, 0.9586 | いいえ |
