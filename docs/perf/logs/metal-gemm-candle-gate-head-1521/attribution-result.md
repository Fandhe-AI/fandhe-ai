# 帰属表（A=../../../../scripts/bench/framework-compare/results/raw/results-m4max-gemm-gate-0.8.0-ctrl-1521.jsonl / B=../../../../scripts/bench/framework-compare/results/raw/results-m4max-gemm-gate-head-cbf5488f-1521.jsonl）

| N | A中央値(min-max) | B中央値 | B/A | 分類 | candle/A | candle/B | §16参照値(candle/fandhe) | §16比 |
|---|---|---|---|---|---|---|---|---|
| 1024 | 2.997 ms (2.324-3.047) | 3.072 ms | 1.0250 | 負荷差（ノイズ帯）・構造分析と整合 | 0.651 | 0.693 | 0.743 | 0.933 |
| 2048 | 9.588 ms (9.338-10.017) | 9.386 ms | 0.9789 | 負荷差（ノイズ帯）・構造分析と整合 | 0.825 | 0.704 | 1.002 | 0.702 |
| 4096 | 46.670 ms (39.739-50.505) | 40.064 ms | 0.8585 | 負荷差（ノイズ帯）・構造分析と整合 | 0.493 | 0.573 | 0.634 | 0.904 |

