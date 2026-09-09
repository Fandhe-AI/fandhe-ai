# イシュー #1255 集計結果

## 1255-DONE

DONE valid_runs=5 high_runs=1 runs_executed=5 target_high_runs=3 max_runs=5 wait_result=HIGH_CONFIRMED


（run ログ検出数: 5）

## 表 A: run × size

| run | size | spread | gate | within_gate | 落ち込みラウンド(max_round_idx) | load_class | median_load1 |
|---|---|---|---|---|---|---|---|
| 1 | 256 | 6.1275e-01 | 5.0000e-02 | false | 0 | high | 5.645 |
| 1 | 512 | 3.2324e-01 | 5.0000e-02 | false | 7 | high | 5.645 |
| 1 | 1024 | 6.2647e-01 | 5.0000e-02 | false | 9 | high | 5.645 |
| 1 | 2048 | 5.4382e-01 | 5.0000e-02 | false | 0 | high | 5.645 |
| 1 | 4096 | 1.0526e-01 | 5.0000e-02 | false | 2 | high | 5.645 |
| 2 | 256 | 2.6033e-01 | 5.0000e-02 | false | 6 | mid | 2.645 |
| 2 | 512 | 5.0033e-01 | 5.0000e-02 | false | 7 | mid | 2.645 |
| 2 | 1024 | 7.4306e-01 | 5.0000e-02 | false | 0 | mid | 2.645 |
| 2 | 2048 | 2.8330e-01 | 5.0000e-02 | false | 2 | mid | 2.645 |
| 2 | 4096 | 2.8479e-02 | 5.0000e-02 | true | 3 | mid | 2.645 |
| 3 | 256 | 3.1881e-01 | 5.0000e-02 | false | 4 | low | 1.815 |
| 3 | 512 | 5.8435e-01 | 5.0000e-02 | false | 6 | low | 1.815 |
| 3 | 1024 | 7.4098e-01 | 5.0000e-02 | false | 3 | low | 1.815 |
| 3 | 2048 | 8.0719e-01 | 5.0000e-02 | false | 2 | low | 1.815 |
| 3 | 4096 | 6.0555e-02 | 5.0000e-02 | false | 1 | low | 1.815 |
| 4 | 256 | 2.4117e-01 | 5.0000e-02 | false | 2 | low | 1.98 |
| 4 | 512 | 1.4758e-01 | 5.0000e-02 | false | 1 | low | 1.98 |
| 4 | 1024 | 8.3713e-01 | 5.0000e-02 | false | 6 | low | 1.98 |
| 4 | 2048 | 6.4077e-01 | 5.0000e-02 | false | 3 | low | 1.98 |
| 4 | 4096 | 5.3835e-02 | 5.0000e-02 | false | 2 | low | 1.98 |
| 5 | 256 | 2.4162e-01 | 5.0000e-02 | false | 5 | low | 1.895 |
| 5 | 512 | 5.0390e-01 | 5.0000e-02 | false | 1 | low | 1.895 |
| 5 | 1024 | 3.3278e-01 | 5.0000e-02 | false | 0 | low | 1.895 |
| 5 | 2048 | 1.0833e+00 | 5.0000e-02 | false | 8 | low | 1.895 |
| 5 | 4096 | 1.4685e-01 | 5.0000e-02 | false | 7 | low | 1.895 |

## 表 B: サイズ別ゲート成立回数（分母: load_class=high かつ run_valid の run 数）

分母（load_class=high かつ run_valid=1 の run 数）: 1

| size | within_gate 成立数 | 分母 |
|---|---|---|
| 256 | 0 | 1 |
| 512 | 0 | 1 |
| 1024 | 0 | 1 |
| 2048 | 0 | 1 |
| 4096 | 0 | 1 |

## 表 C: スパイク位置分布（load_class=high かつ run_valid の run のみ）

| max_round_idx（0 始まり） | 出現回数 |
|---|---|
| 0 | 2 |
| 2 | 1 |
| 7 | 1 |
| 9 | 1 |

## 参考表 R: #1187 試行 4（非排他）

非排他（開始時 load 1.38・実行中 7〜11 へ上昇。#1187 試行 4。`docs/perf/metal-gemm-transpose-tiled.md` §5.4 参照）:

| size | spread | verdict | 落ち込みラウンド(argmin(round_tflops)) |
|---|---|---|---|
| 256 | 0.2877 | NG: gate 超過 | 9 |
| 512 | 0.5298 | NG: gate 超過 | 8 |
| 1024 | 2.3028 | NG: gate 超過 | 3 |
| 2048 | 1.6631 | NG: gate 超過 | 4 |
| 4096 | 0.9392 | NG: gate 超過 | 9 |

## 排他側

排他環境 valid_runs=0（#1253。`docs/perf/metal-gemm-transpose-tiled.md` §5.6 参照）
検出したマーカー: DONE_TIMEOUT_ATTEMPT1, DONE_TIMEOUT_ATTEMPT2

