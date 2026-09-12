# 低レイヤー診断（2026-09-12）実測成果物

本ディレクトリは `docs/perf/lowlayer-diagnosis-2026-09-12.md`（低レイヤー診断
artifact §7 の実測転記。イシュー `1574`）の生ログ・生データ・
オーケストレーションスクリプト置き場。**性能の ADOPT／REJECT を確定する
実測ではなく、後続 issue（`1575` 等。`docs/perf/lowlayer-
diagnosis-2026-09-12.md` §7 を参照）の設計・優先順位づけに使う診断・方向づけ
用の記録**である。内部ホスト名は含めない（`scripts/mac-diag.zsh` の
scratchpad 絶対パス 1 箇所のみ `<masked-scratchpad-path>` へ置換済み）。

## 系列・機体・証拠等級

| 系列 | 機体 | コード | 負荷 | 証拠等級 |
|------|------|--------|------|----------|
| DGX 0.8.0 | DGX Spark GB10（専有・sm_121） | ピン `fandhe-ai =0.8.0`（crates.io 公開版） | load average ≈ 0.05 | framework-compare 全 task はハーネス既定の単一起動・内部 20 反復（`warmup=20`/`iters` 既定）中央値。RAYON スイープは 3 プロセス起動中央値 |
| Mac HEAD | Apple M4 Max（共有） | origin/main HEAD `097bff19` の path patch＋`diag-instrumentation.patch`（backward 内訳計装・MSE/mask マイクロ計測。本番実装ではない診断専用パッチ）。CPU 計測経路自体は `f91cafa3`（HEAD）と同一 | load average ≈ 4 | backward 内訳・train phases・readout interleave は 5 プロセス起動中央値 |

## ディレクトリ構成

- `dgx/` — DGX Spark GB10 実測ログ・JSONL（`dgx-prebuild.sh`／`dgx-run.sh`／
  `dgx-run2.sh`／`dgx-py.sh` の出力）。
  - `run_all_cuda.log`・`results-dgx-0.8.0.jsonl`・`skipped-dgx-0.8.0.log` —
    framework-compare 全 task（gemm／train／infer × cpu／cuda）。
  - `dgx-run.log`・`async_ordering_real_device.log`・
    `tma_probe_real_device.log`・`gemm_transposed_parity.log`・
    `gemm_transposed_perf.log` — 実機 `#[ignore]` 診断テスト 4 種
    （非同期順序・TMA プローブ・#1214 NT/TN parity・純カーネル時間）。
  - `rayon-sweep.jsonl`／`rayon-sweep.err` — 無 pin の `RAYON_NUM_THREADS`
    ∈ {4,8,10,20} × train/infer × fresh/reuse（3 起動）。
  - `rayon-sweep-pinned.jsonl`／`rayon-sweep-pinned.err` — `taskset
    -c 5-9,15-19`（大コア pin。#1319 と同一 pin 集合）付き T ∈ {4,8,10}
    （3 起動）。
  - `results-dgx-0.8.0-extra.jsonl`・`extra.err` — CPU gemm reuse
    N=256/512/1024/2048/4096・fresh N=4096（fandhe-ai／candle／burn）追加計測。
  - `dgx-py.log`・`results-dgx-py-0.8.0.jsonl`・`py.err` — Python 参照
    フレームワーク（PyTorch CPU/CUDA・TensorFlow CPU・SciPy）を
    framework-compare と同一プロトコルで計測（`bench_py.py`）。
  - `uptime_before_*.txt`／`uptime_after_*.txt` — 各ステージ前後の `uptime`
    （専有確認の記録）。
- `mac/` — Apple M4 Max 実測ログ・JSONL（`mac-diag.zsh` の出力）。
  - `train-phases.jsonl`・`diag-{cpu,metal}-{fresh,reuse}-run{1..5}.err` —
    backward 内訳計装（`FANDHE_DIAG_BACKWARD=1`）付き train `--phases`
    × device ∈ {cpu, metal} × mode ∈ {fresh, reuse} の 5 起動。
  - `readout-phases.jsonl`・`readout.err` — Metal N=1024 gemm の
    `--readout <legacy|borrowed>` 交互計測 × mode ∈ {reuse, fresh} の 5 起動
    （#1520 の続き。matmul 区間への局所化用）。
  - `uptime_before.txt`／`uptime_after.txt` — 実行前後の `uptime`
    （共有負荷の記録。load average ≈ 4 はこのファイルの実測値）。
- `scripts/` — オーケストレーションスクリプト。
  - `dgx-prebuild.sh` — DGX 側 framework-compare／診断テストの事前ビルド
    （ベンチ実行と分離）。
  - `dgx-run.sh` — DGX 実測本体（1: 全セル、2: 診断テスト、3: RAYON スイープ）。
  - `dgx-run2.sh` — DGX 追加計測（大コア pin スイープ・CPU gemm reuse 追加形状）。
  - `dgx-venv.sh` — DGX 側 Python 参照フレームワーク用 venv 構築。
  - `dgx-py.sh` — DGX 側 Python 参照フレームワーク計測本体。
  - `mac-diag.zsh` — Mac 側 backward 内訳・readout interleave 計測本体。
- `diag-instrumentation.patch` — `crates/autodiff`／`crates/backend-cpu` へ
  当てた診断専用計装パッチ（backward 内訳ログ出力・`mse_loss_backward` の
  要素数しきい値マイクロ計測用テスト・`scripts/bench/framework-compare/
  Cargo.lock` の付随差分）。**本番実装ではない**。§7.3
  （`docs/perf/lowlayer-diagnosis-2026-09-12.md`）の数値の再現に使う。
- `bench_py.py` — PyTorch／TensorFlow／SciPy を framework-compare と同一
  プロトコル（`--framework`／`--task`／`--device`／`--size`／`--mode`・
  JSONL 出力）で計測する参照ハーネス（`--device cuda` 対応版）。リポジトリの
  `scripts/bench/framework-compare/` 配下には対応する既存ファイルが無いため
  新規配置先を提案せずそのままここへ置く。
- `mac-head-097bff19-path-patch.jsonl` — Mac HEAD path patch 計測時に
  `scripts/bench/framework-compare/results/raw/results.jsonl`
  （追跡済みファイル）へ誤って追記されていた 63 行を `git diff` から抽出し
  差し戻した後のバックアップ（`fandhe-ai` `version":"0.8.0"` ラベル付き・
  実体は HEAD `097bff19` path patch 計測）。追跡ファイル自体は
  `git checkout --` で計測前の状態へ復元済み。

## 数値の正

実測数値・表・結論の正は `docs/perf/lowlayer-diagnosis-2026-09-12.md`
（本ディレクトリのログから転記・出典欄で本ディレクトリのファイルを参照）。
本 README はファイル構成の説明のみで数値は含めない。
