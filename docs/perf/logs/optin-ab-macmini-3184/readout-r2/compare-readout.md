| cell (task/device/size/mode/phase) | legacy median | borrowed median | borrowed/legacy | checksum | 判定 |
|---|---|---|---|---|---|
| gemm/metal/1024/fresh/None | 2.868 ms (min 2.790 ms / max 3.581 ms) | 2.751 ms (min 2.671 ms / max 3.190 ms) | 0.9592 | 完全一致 | 非後退 |
| gemm/metal/1024/reuse/None | 3.059 ms (min 3.039 ms / max 3.111 ms) | 2.963 ms (min 2.918 ms / max 2.985 ms) | 0.9688 | 完全一致 | 非後退 |
| gemm/metal/2048/fresh/None | 13.586 ms (min 13.527 ms / max 13.695 ms) | 13.277 ms (min 13.241 ms / max 13.311 ms) | 0.9773 | 完全一致 | 非後退 |
| gemm/metal/2048/reuse/None | 14.479 ms (min 14.461 ms / max 14.585 ms) | 14.014 ms (min 13.973 ms / max 14.067 ms) | 0.9679 | 完全一致 | 非後退 |
| gemm/metal/4096/fresh/None | 86.311 ms (min 85.373 ms / max 86.538 ms) | 84.171 ms (min 83.925 ms / max 84.950 ms) | 0.9752 | 完全一致 | 非後退 |
| gemm/metal/4096/reuse/None | 91.377 ms (min 91.205 ms / max 91.568 ms) | 89.095 ms (min 87.376 ms / max 90.869 ms) | 0.9750 | 完全一致 | 非後退 |

総合判定: ADOPT（借用ビューへ切替。Metal 分岐を除去して既定化する）
