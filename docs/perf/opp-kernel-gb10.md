# GB10 の相手（PyTorch／cuBLAS）純カーネル時間と Python 側 checksum の診断

イシュー #3185。規則の正は `docs/perf/logs/opp-kernel-gb10-3185/RULE.txt`、診断本体は `scripts/bench/framework-compare/opp_kernel_cuda_diag.py`。

## 状態

**診断基盤と事前登録規則まで完了。実測は未実施（unavailable）。受入基準 1〜4 は未達で、本イシューはこの doc だけではクローズできない。**
数値は推定値で埋めていない（RULE 項 14）。

- 実装時点で GB10 へは SSH 到達できた（負荷 0.15・GPU 利用率 0%）が、RULE 項 2 が要求する `~/work/.venv-bench`（`torch 2.14.0+cu130`）がノード上に存在しなかった
- ノード上の他の venv の torch は 2.13.0+cu130 と 2.11.0+cu130 のみで、0.11.0 系列の checksum 照合（RULE 項 11）の前提と版が合わないため別版へ読み替えなかった
- `torch 2.14.0+cu130` の導入は共有ノードへの大型パッケージ導入にあたり承認事項のため、実装側では行っていない

## 目的

相手（PyTorch・candle。どちらも cuBLAS SGEMM と読める）の計測窓には matmul・`.cpu().numpy()`・Python `checksum()` が混ざり、「相手のカーネルが速いのか、取り出しが速いのか」を判定できていない。次を数字で切り分ける。

- (a) cuBLAS FP32 の純カーネル時間に対する fandhe FP32 の位置（fandhe 単体は 0.9.0 の Layer B で N=2048 が 1.165 ms。`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §12.4）
- (b) TF32（`allow_tf32` ON）を使わないと超えられない壁があるか
- Python checksum の ns／要素（Issue 本文の推定 約 2.7、範囲 2.4〜3.0。Rust 側は実測 0.51）の実測

burn（cubecl）の純カーネルは対象外。判定・閾値は設けない（record_only）。

## 方法

| 項目 | 内容 |
|---|---|
| Python 側 | `opp_kernel_cuda_diag.py measure`。腕 fp32／tf32／cpu、N=256〜4096、区間 `mm_pure`（wall）・`mm_event`（CUDA event）・`readout`・`checksum`・`window` ほか。`allow_tf32` は読み戻しを検査し全行に記録する |
| fandhe 側 | 既存 `#[ignore]` テストのログを `parse-fandhe` で読む（Rust は変更しない）。N>=1024 は Layer B（`gemm_reuse_phase_diag_select`、wall、20/20）、N<=512 は `gemm_small_launch_cost_diag`（L2S は wall、D 層は event、50/200） |
| 比 | 時計ごとに列を分ける。wall 対 wall、event 対 event。N>=1024 の fandhe には event 時計の値がないため「該当計測なし」 |
| 照合 | fp32／cpu 腕の checksum を 0.11.0 系列（`docs/perf/logs/framework-compare-0.11.0-remeasure/gb10/results-dgx-py-0.11.0-median5.jsonl`）と fail-closed で照合。tf32 腕は記録のみ |
| CPU BLAS 経路 | `cpu-blas-probe`（`DNNL_VERBOSE`／`ONEDNN_VERBOSE`・`torch.__config__.show()`・`/proc/self/maps` のライブラリ名） |
| 入力 | bench_py の SEED_A／SEED_B（Python 側）。fandhe 側テストは別シード（FP32 SIMT の時間はデータにほぼ依存しない前提） |

変えないもの: `bench_py.py`・相手側ベンチ・判定規則（`checksum_contract.py`・`compare_gemm_gate.py`）・#3179 のツールと規則。

## 結果

| N | cuBLAS fp32 mm_pure [ms] | cuBLAS tf32 mm_pure [ms] | fandhe wall [ms] | wall 比 | event 比 |
|---|---|---|---|---|---|
| 256 | unavailable | unavailable | unavailable | unavailable | unavailable |
| 512 | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1024 | unavailable | unavailable | unavailable | unavailable | 該当計測なし |
| 2048 | unavailable | unavailable | unavailable（参考: 0.9.0 Layer B 1.165） | unavailable | 該当計測なし |
| 4096 | unavailable | unavailable | unavailable | unavailable | 該当計測なし |

- Python checksum の実測 ns／要素と推定 2.7 との差: unavailable
- CPU 側 BLAS 経路: 確認できなかった（実測未実施のため。経路の推測はしない）

## 受入基準との対応

| 基準 | 状態 |
|---|---|
| 1. cuBLAS 純カーネル時間（N × FP32・TF32、5 run 生ログ） | 未達（unavailable） |
| 2. Python checksum の ns／要素と推定値との差 | 未達（unavailable） |
| 3. fandhe FP32 単体と cuBLAS FP32 の比 | 未達（unavailable） |
| 4. CPU 側 BLAS 経路の確認可否 | 未達（確認できなかった。未実測） |
| 5. 相手側ベンチと判定規則を変えない | 達成（該当ファイルは変更なし） |

## #3214・#3262 への引き継ぎ

実測前のため、目標値の候補（cuBLAS FP32 純時間）と FP32 で超えられない壁の有無は未確定。両イシューへのコメントは投稿していない。

## 残課題

- `torch 2.14.0+cu130` の venv（`~/work/.venv-bench`）をノードに用意して `run-gb10.sh` を実行する（導入の承認が必要）
- N>=1024 の fandhe event 時計による計測は Rust の計装追加が要るため本イシューの対象外（必要なら別イシュー化を提案する）
