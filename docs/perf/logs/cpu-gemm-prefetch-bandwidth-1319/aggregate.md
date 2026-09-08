## G-b（主判定）: R_dram = streamed_dram / l1_resident

### Apple M4 Max（共有負荷下）（resid-m4max-run*.txt。5 run）

| threads | R_dram 各 run（参考: run 内比） | R_dram 代表値（中央値の比。事前宣言式） | <=0.95 の run 数 |
|---|---|---|---|
| single | 0.5903, 0.9104, 2.4945, 1.4250, 0.9888 | 1.4250 | 2/5 |
| multi | 0.6580, 0.3264, 1.5325, 1.5333, 0.4061 | 1.0376 | 3/5 |

### DGX Spark GB10（無 pin。異種コアスケジューリング混入）（resid-dgx-run*.txt。5 run）

| threads | R_dram 各 run（参考: run 内比） | R_dram 代表値（中央値の比。事前宣言式） | <=0.95 の run 数 |
|---|---|---|---|
| single | 0.4445, 0.4426, 0.4609, 0.4569, 0.4550 | 0.4550 | 5/5 |
| multi | 0.2731, 0.2648, 0.2456, 0.3034, 0.3029 | 0.2781 | 5/5 |

### DGX Spark GB10（big core pin: taskset -c 5-9,15-19）（resid-dgx-bigpin-run*.txt。5 run）

| threads | R_dram 各 run（参考: run 内比） | R_dram 代表値（中央値の比。事前宣言式） | <=0.95 の run 数 |
|---|---|---|---|
| single | 0.4552, 0.4535, 0.4598, 0.4681, 0.4458 | 0.4543 | 5/5 |
| multi | 0.3231, 0.1854, 0.1903, 0.1665, 0.1649 | 0.1854 | 5/5 |

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

