# ロバスト統計候補の再適用（イシュー #1266）

本ディレクトリは **新規計測を行わない**（既存ログの再集計のみ）ため env_info を
含まない。入力は以下の既存ログ（`docs/perf/logs/metal-gemm-transpose-route-ab-1242/`
および同ディレクトリと同階層の `metal-gemm-transpose-route-ab-1186/`・
`metal-gemm-transpose-route-ab-1187/`）:

- 正確値（秒単位。`phase1_round_stats` 行）: `1255-phase1_run{1..5}.log`（5 run）
- 参考値（TFLOPS 単位。`round_tflops` 行。`secs = 1/tflops` で近似復元）:
  `../metal-gemm-transpose-route-ab-1186/route_ab_run1.log`（1 run）・
  `../metal-gemm-transpose-route-ab-1187/route_ab_run{1..4}.log`（4 run）

## ファイル

- `reapply.py`: 再適用スクリプト（Python3 標準ライブラリのみ）。
  `python3 reapply.py --self-test` で #1255 正確値 25 セルの raw spread
  再計算値がログ `spread=` と一致するか検証し、`python3 reapply.py` で
  `reapply.md`（表）を決定的に再生成する
- `reapply.md`: 上記の固定出力（`docs/perf/metal-bench-noise-protocol.md` §8
  の表の正本）

## 再現手順

```sh
python3 docs/perf/logs/metal-bench-robust-stats-1266/reapply.py --self-test
python3 docs/perf/logs/metal-bench-robust-stats-1266/reapply.py \
  > docs/perf/logs/metal-bench-robust-stats-1266/reapply.md
```
