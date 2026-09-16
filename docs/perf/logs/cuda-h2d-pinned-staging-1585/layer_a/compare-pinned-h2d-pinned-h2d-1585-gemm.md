| cell (task/device/size/mode/phase) | off median | on median | on/off | checksum | 判定 |
|---|---|---|---|---|---|
| gemm/cuda/1024/fresh/None | 953.4 us (min 947.2 us / max 955.0 us) | 1.979 ms (min 1.759 ms / max 2.061 ms) | 2.0758 | 完全一致 | 後退（REJECT 方向） |
| gemm/cuda/1024/reuse/None | 2.078 ms (min 1.990 ms / max 2.165 ms) | 2.938 ms (min 2.731 ms / max 3.359 ms) | 1.4136 | 完全一致 | 後退（REJECT 方向） |
| gemm/cuda/2048/fresh/None | 4.177 ms (min 4.175 ms / max 4.183 ms) | 6.386 ms (min 6.279 ms / max 6.494 ms) | 1.5288 | 完全一致 | 後退（REJECT 方向） |
| gemm/cuda/2048/reuse/None | 8.343 ms (min 8.068 ms / max 8.422 ms) | 9.980 ms (min 9.807 ms / max 10.375 ms) | 1.1961 | 完全一致 | 後退（REJECT 方向） |
| gemm/cuda/4096/fresh/None | 38.080 ms (min 37.908 ms / max 38.492 ms) | 46.327 ms (min 45.140 ms / max 46.865 ms) | 1.2166 | 完全一致 | 後退（REJECT 方向） |
| gemm/cuda/4096/reuse/None | 37.737 ms (min 33.578 ms / max 38.739 ms) | 45.946 ms (min 45.448 ms / max 45.988 ms) | 1.2175 | 完全一致 | 後退（REJECT 方向） |
