## G-b（主判定）: R_dram = streamed_dram / l1_resident

### Apple M4 Max（共有負荷下）（resid-m4max-run*.txt。5 run）

| threads | R_dram 各 run | 中央値 | <=0.95 の run 数 |
|---|---|---|---|
| single | 0.8863, 0.9904, 0.8119, 0.6389, 0.5328 | 0.8119 | 4/5 |
| multi | 0.2678, 0.2691, 0.2179, 0.2671, 0.2512 | 0.2671 | 5/5 |

### DGX Spark GB10（無 pin。異種コアスケジューリング混入）（resid-dgx-run*.txt。5 run）

| threads | R_dram 各 run | 中央値 | <=0.95 の run 数 |
|---|---|---|---|
| single | 1.2985, 1.3663, 0.4063, 0.4369, 1.3508 | 1.2985 | 2/5 |
| multi | 0.5354, 0.5251, 0.5183, 0.4710, 0.4815 | 0.5183 | 5/5 |

### DGX Spark GB10（big core pin: taskset -c 5-9,15-19）（resid-dgx-bigpin-run*.txt。5 run）

| threads | R_dram 各 run | 中央値 | <=0.95 の run 数 |
|---|---|---|---|
| single | 0.5088, 0.4350, 0.4467, 0.4333, 0.4392 | 0.4392 | 5/5 |
| multi | 0.2602, 0.3409, 0.3966, 0.3183, 0.3085 | 0.3183 | 5/5 |

## G-a 文脈: 実測到達帯域（achievable_bandwidth_diag）

### Apple M4 Max（bw-m4max-run*.txt。5 run。achievable_read GB/s）

| threads | 各 run | 中央値 |
|---|---|---|
| single | 8.0685, 8.1059, 8.0132, 7.9439, 8.0665 | 8.0665 |
| multi | 75.8622, 73.5427, 75.6781, 77.9233, 69.2665 | 75.6781 |

### DGX Spark GB10（無 pin）（bw-dgx-run*.txt。5 run。achievable_read GB/s）

| threads | 各 run | 中央値 |
|---|---|---|
| single | 7.7589, 7.7899, 7.6888, 7.7732, 7.7827 | 7.7732 |
| multi | 67.1139, 64.7280, 68.4528, 69.3170, 73.0335 | 68.4528 |

