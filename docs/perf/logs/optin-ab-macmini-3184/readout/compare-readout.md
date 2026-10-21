| cell (task/device/size/mode/phase) | legacy median | borrowed median | borrowed/legacy | checksum | 判定 |
|---|---|---|---|---|---|
| gemm/metal/1024/fresh/None | 2.824 ms (min 2.804 ms / max 3.065 ms) | 2.755 ms (min 2.712 ms / max 2.833 ms) | 0.9754 | 完全一致 | 非後退 |
| gemm/metal/1024/reuse/None | 3.095 ms (min 3.028 ms / max 3.138 ms) | 2.929 ms (min 2.896 ms / max 3.012 ms) | 0.9462 | 完全一致 | 非後退 |
| gemm/metal/2048/fresh/None | 13.678 ms (min 13.592 ms / max 13.695 ms) | 13.256 ms (min 13.197 ms / max 13.345 ms) | 0.9691 | 完全一致 | 非後退 |
| gemm/metal/2048/reuse/None | 14.555 ms (min 14.474 ms / max 14.593 ms) | 14.057 ms (min 13.990 ms / max 14.075 ms) | 0.9658 | 完全一致 | 非後退 |
| gemm/metal/4096/fresh/None | 86.670 ms (min 85.746 ms / max 86.923 ms) | 84.933 ms (min 84.086 ms / max 85.031 ms) | 0.9800 | 完全一致 | 非後退 |
| gemm/metal/4096/reuse/None | 90.987 ms (min 90.521 ms / max 91.488 ms) | 90.334 ms (min 89.497 ms / max 91.094 ms) | 0.9928 | 完全一致 | 非後退 |

総合判定: ADOPT（借用ビューへ切替。Metal 分岐を除去して既定化する）
