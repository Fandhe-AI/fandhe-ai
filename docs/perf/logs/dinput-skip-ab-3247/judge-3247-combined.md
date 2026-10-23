# #3247 判定（rule_sha256=b4d2915487e8e020…）

| 機体 | device | mode | before 中央値 ms | after 中央値 ms | ratio | ratio_r（5 round） | 判定 | 理由 | 参考: 相手比 after（別セッション参照値・判定外） | 参考: 相手比 before |
|---|---|---|---|---|---|---|---|---|---|---|
| macmini | cpu | reuse | 0.6740 | 0.4980 | 0.739 | 0.764, 0.723, 0.741, 0.720, 0.745 | ADOPT | 効果確認（ratio<=0.95 かつ 5/5 round で ratio_r<1.00） | 2.25x (pytorch) | 3.04x (pytorch) |
| macmini | cpu | fresh | 0.6951 | 0.6956 | 1.001 | 0.991, 0.983, 1.032, 1.001, 1.008 | guard-pass | ガード（ratio<=1.03 かつ 5/5 後退でない） | 3.14x (pytorch) | 3.14x (pytorch) |
| macmini | metal | reuse | 1.8841 | 1.5754 | 0.836 | 0.836, 0.836, 0.838, 0.839, 0.836 | ADOPT | 効果確認（ratio<=0.95 かつ 5/5 round で ratio_r<1.00） | 2.29x (pytorch) | 2.74x (pytorch) |
| macmini | metal | fresh | 2.3363 | 2.3305 | 0.998 | 0.828, 0.995, 1.004, 0.997, 1.000 | guard-pass | ガード（ratio<=1.03 かつ 5/5 後退でない） | 3.38x (pytorch) | 3.39x (pytorch) |
| gb10 | cpu | reuse | 0.9181 | 0.7456 | 0.812 | 0.717, 0.675, 0.856, 0.898, 0.970 | ADOPT | 効果確認（ratio<=0.95 かつ 5/5 round で ratio_r<1.00） | 2.09x (pytorch) | 2.57x (pytorch) |
| gb10 | cpu | fresh | 0.9975 | 0.8504 | 0.853 | 0.807, 0.798, 0.958, 0.777, 0.861 | guard-pass | ガード（ratio<=1.03 かつ 5/5 後退でない） | 2.38x (pytorch) | 2.79x (pytorch) |
| gb10 | cuda | reuse | 0.3119 | 0.2778 | 0.891 | 0.884, 0.887, 0.901, 0.891, 0.869 | ADOPT | 効果確認（ratio<=0.95 かつ 5/5 round で ratio_r<1.00） | 1.02x (candle) | 1.15x (candle) |
| gb10 | cuda | fresh | 0.5260 | 0.5229 | 0.994 | 1.031, 1.024, 0.956, 0.997, 0.994 | guard-pass | ガード（ratio<=1.03 かつ 5/5 後退でない） | 1.92x (candle) | 1.94x (candle) |

終了コード: 0（処置セルがすべて ADOPT）

上の表は `judge_dinput_skip_ab.py --require-both` の stdout をそのまま収録した。実行は `docs/perf/logs/dinput-skip-ab-3247/README.md` の手順 6 のとおり（参照 JSONL は `framework-compare-0.11.0-macmini-remeasure/macmini-series-b/` と `framework-compare-0.11.0-remeasure/gb10/` の `*-0.11.0-median5.jsonl` 計 4 本）。「相手比」は fandhe-ai の median を最速の相手の median で割った値で、1 を超えるほど fandhe-ai が遅い。
