| cell (task/device/size/mode/phase) | legacy median | borrowed median | borrowed/legacy | checksum | 判定 |
|---|---|---|---|---|---|
| gemm/metal/1024/fresh/None | 2.122 ms (min 2.034 ms / max 2.606 ms) | 2.570 ms (min 2.275 ms / max 2.713 ms) | 1.2108 | 完全一致 | 後退 |
| gemm/metal/1024/reuse/None | 2.628 ms (min 2.310 ms / max 3.064 ms) | 2.920 ms (min 2.614 ms / max 2.930 ms) | 1.1112 | 完全一致 | 後退 |
| gemm/metal/2048/fresh/None | 8.215 ms (min 7.606 ms / max 9.049 ms) | 8.060 ms (min 7.131 ms / max 8.865 ms) | 0.9811 | 完全一致 | 非後退 |
| gemm/metal/2048/reuse/None | 9.435 ms (min 8.734 ms / max 10.655 ms) | 9.395 ms (min 8.870 ms / max 10.280 ms) | 0.9958 | 完全一致 | 非後退 |
| gemm/metal/4096/fresh/None | 34.154 ms (min 33.353 ms / max 35.508 ms) | 33.526 ms (min 32.565 ms / max 35.176 ms) | 0.9816 | 完全一致 | 非後退 |
| gemm/metal/4096/reuse/None | 42.837 ms (min 39.848 ms / max 49.079 ms) | 38.164 ms (min 37.877 ms / max 42.913 ms) | 0.8909 | 完全一致 | 非後退 |

総合判定: REJECT（legacy フォールバックを維持する）
