# fandhe-ai 0.9.0 正式系列 framework-compare 再計測（イシュー #1967）

## 目的

crates.io `fandhe-ai =0.9.0` 公開（2026-09-17。`release-all.yml` run 35136585936）を受け、
framework-compare の承認ピンを `=0.9.0` へ更新したうえで GEMM／train／infer の正式系列
再計測を行い、記録をリポジトリへ収納する（親: 低レイヤー診断・機能網羅ルート #1920）。

計測自体は 2026-09-16（UTC）に実施済み（M4 Max・GB10 とも）。本ディレクトリはその生ログ・
集計スクリプト・スコアボード artifact をリポジトリへ収納したもの。

## ディレクトリ構成

| パス | 内容 |
|---|---|
| `m4max-series-a/` | Apple M4 Max・系列 A（共有負荷下・ゲートなし）5 run。`run{1..5}/`・`results-m4max-0.9.0-median5.jsonl`（5 run 中央値集計）・`loop.log` |
| `m4max-series-b/` | Apple M4 Max・系列 B（load1 < 8.0 ゲート付き）5 run。`run{1..5}/`・`results-m4max-0.9.0-median5.jsonl`・`gate.log`（ゲート判定履歴）・`RULE.txt`（事前登録規則）・`loop.log` |
| `gb10/` | DGX Spark GB10・専有 1 セッション。`results-dgx-0.9.0.jsonl`（`run_all_cuda.sh` 全セル）・`results-dgx-0.9.0-extra.jsonl`（CPU gemm reuse／N=4096 追加計測・Python 参照 FW は含まない）・`run_all_cuda.log`・`uptime_before_all.txt`／`uptime_after_all.txt`・`tree.txt`（registry pin 確認）・`skipped-dgx-0.9.0.log`・`extra.err` |
| `scripts/` | 計測オーケストレーション（`m4max-run-090.sh`・`m4max-090-loop.sh`〈系列 A〉・`m4max-090b-loop.sh`〈系列 B・ゲート実装〉・`dgx-run-090.sh`） |
| `aggregate/` | 集計ツール（`median_rounds.py`・`compare_head.py`） |
| `scoreboard/` | フレームワーク横並びスコアボード artifact（`gen_090.py`・`body_090.html`・`style.css`）。0.8.0 比較を含む対戦成績表。Python 3 参照 FW（PyTorch／TensorFlow／SciPy）は 0.9.0 で未再計測のため 2026-09-12 計測値をそのまま流用（`gen_090.py` 冒頭 docstring・`body_090.html` の lede 参照） |

## 計測条件

- registry ピン: `fandhe-ai =0.9.0`・`candle-core =0.11.0`・`burn =0.21.0`（`deps-policy.md` 第 9 区分。`tree.txt`・各 `run.log` の `cargo tree` 出力で pin 解決を確認済み）
- **Apple M4 Max**:
  - 系列 A: 共有負荷下（他セッション並走）で 5 run。各 run 開始時 load1 = 8.46/16.92/23.46/28.05/20.47
  - 系列 B: 各 run 開始前に load1 < 8.0（系列 A の最小 8.46 未満）を最大 30 分待つゲート付きで 5 run。`gate.log` のとおり全 5 run がゲートを通過して完走した（詳細は「事前登録規則」節）
  - いずれも `SKIP_BUILD=1`（prebuild 済みバイナリを使い回し、計測からビルド残余負荷を分離）
- **DGX Spark GB10**: 専有 1 セッション（5 回計測中央値ではない。`uptime_before_all.txt` の load average 0.02/0.05/0.16 のとおり低負荷単発）

## 事前登録規則（`m4max-series-b/RULE.txt`。2026-09-16T19:43:40Z、系列 B の結果が出る前に固定）

- 系列 B は 5 run すべてがゲート（load1 < 8.0）を通過して完走した場合のみ正式値とする。1〜4 run の部分完走・GATE-TIMEOUT は参考扱いとし系列 A を正式値とする
- いずれの場合も両系列を併記し、A/B 間で verdict が反転したセルをノイズ帯として明示する
- 0.8.0 → 0.9.0 の絶対 ms 差分は M4 では負荷差の交絡があるため参考値。判定は同一 run 内の対戦相手比（verdict）を主とする

**実績**: `gate.log`（計 15 エントリ）を突き合わせると、系列 B は run1〜run5 いずれも開始直前の
ゲート判定で load1 < 8.0 を満たしてから起動しており（`loop.log` の `run{N}: done` タイムスタンプと
`gate.log` の直前エントリが一致）、5/5 run がゲートを通過して完走した。したがって
**正式値は系列 B、系列 A は参考値**として扱う（`docs/perf/cpu-gemm-candle-gate-remeasurement.md`・
`docs/perf/metal-gemm-candle-gate-remeasurement.md` への追記でこの扱いに従う）。

## 判定規則（GEMM reuse の candle 比ゲート。既存 doc と同一）

- 主判定は同一 run 内の対戦相手比: `candle fresh 中央値 / fandhe-ai reuse 中央値`。1.0 を超えれば
  fandhe-ai が candle より高速（**達成**）、1.0 未満なら**未達**
- `parity_fail_count > 0` のセルは比較データの妥当性上「判定不能」とする（REQ-2 2026-09-12 追記。
  `docs/candle-parity-tolerance-contract-decision.md` §8）。本追補で参照した全セルは
  `parity_fail_count = 0`（fandhe-ai・candle いずれも）であり判定不能セルは無かった
- `parity_scaled_abs_rescued > 0`（スケール付き絶対誤差救済項の適用）は判定不能の対象外であり、
  該当セルには「candle 救済 N 要素」と注記したうえで通常どおり達成／未達を判定する

## マスク規約

生ログ・スクリプト中の絶対パス・内部ホスト名は次のプレースホルダへマスク済み: `<repo>`
（リポジトリ作業ディレクトリ）・`<home>`（ホームディレクトリ）・`<scratch>`（スクラッチ領域）。

## 「正式値」の意味の区別（重要）

本ディレクトリ・各追補節でいう「正式値」（系列 B・GB10 の唯一セッション）は、上記の
事前登録規則（RULE.txt。同一 run 内の最速相手比〈verdict〉を主とする）上の呼称であり、
各 candle-gate doc（`docs/perf/{cuda,metal,cpu}-gemm-candle-gate-remeasurement.md`）が
従来使ってきた「正式系列」（5 回計測中央値・専有ゲート付き `run_gemm_gate_*.sh`・
`compare_gemm_gate.py` の ADOPT/REJECT 判定式）とは**別の意味**である。本計測は一般
framework-compare 実行（`run_all_cuda.sh`・`m4max-run-090.sh`）から算出した記録であり、
candle 比ゲートの達成判定（各 doc の既存ゲート判定表）はこの追補では更新しない。

## 後続 issue からの参照

正式系列 `=0.9.0` の candle 比ゲート判定（CUDA／Metal／CPU 各バックエンド）は
`docs/perf/cuda-gemm-candle-gate-remeasurement.md`・`docs/perf/metal-gemm-candle-gate-remeasurement.md`・
`docs/perf/cpu-gemm-candle-gate-remeasurement.md` の末尾追補節（イシュー #1967）に記録した。
後続イシューはこれらの節と本ディレクトリのパスを出典として参照する。
