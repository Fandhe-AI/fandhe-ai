# CudaGemmAuto::run_f16 の MatrixUnit 分岐 mma 優先化 前後比較（#1156）

イシュー #1156 の受け入れ条件 R5（ユーザー承認条件）「結線前後で同一プロトコル・
5 回計測中央値の比較を取り、後退確認時は結線しない」に対応する記録。設計の正は
`docs/dispatch-rules-design.md` §5.6。

## 状態: 実測完了（DGX Spark GB10 実機。PR #1177 codex-review P1 指摘の是正）

`CudaGemmAuto::run_f16`（転送込み・H2D／カーネル起動／D2H を丸ごと計測）を
base（`0c91218`。結線前 = `MatrixUnit` 判定時に wmma を直接使用）と HEAD
（`ce4fcf0` 以降。結線後 = mma 優先・wmma フォールバック）の双方でビルドし、
同一形状（dim 512/1024/2048/4096）・同一計測プロトコルで比較した。実測環境・
計測用ベンチバイナリは下記「実機実行手順」節を参照。

先行根拠として、`docs/perf/cuda-wmma-f16-perf-triage.md` §3.1/§4.1（イシュー
#1123・2026-09-03 GB10 実機実測）のカーネル単体計測が `mma_sync_f16` が
`wmma_f16_opt`/`wmma_f16_basic` に対し形状依存で約 4.1〜10.8 倍高速であることを
既に確認済みである。本ドキュメントの実測はこれを `CudaGemmAuto::run_f16` 経由の
auto 経路（転送込み）で裏付けるものである。

## 実機実行手順（DGX Spark GB10。2026-09-04 実測時点の手順）

転送は rsync 方式（`docs/real-hardware-verification-env.md`。ホスト名等はローカル
管理値を使い本ドキュメントには書かない）。

計測用バイナリは `examples/gemm_auto_f16_mma_switch_bench.rs`（本 PR で追加。
`bench-harness::run`・`MeasurementConfig::default`〈warmup 20・計測 20〉で
`CudaGemmAuto::run_f16` を転送込みで計測する。`examples/gemm_mma_bench.rs` と
同じ計測コア・シード）。

```bash
cargo run -p fandhe-ai-backend-cuda --example gemm_auto_f16_mma_switch_bench --release
```

同一バイナリソースを base（`0c91218`）／HEAD（本 PR 反映後）それぞれのワーク
ツリーへコピーしてビルド・実行し、出力される `size=<dim> auto_f16_tflops=<値>`
を突き合わせた（base 側は `internal-diagnostics` feature 未導入のためデフォルト
feature でビルド。本バイナリはこの feature に依存しない）。

## 実測結果

実機: DGX Spark GB10（`nvidia-smi --query-gpu=name` = `NVIDIA GB10`）。
`rustc 1.97.0`・CUDA 13.0（`nvcc release 13.0, V13.0.88`）。GPU 使用率 0%（他
プロセス非稼働）を確認したうえで計測。2026-09-04 実測。

| dim | base（結線前・wmma 優先）TFLOPS 中央値 | HEAD（結線後・mma 優先）TFLOPS 中央値 | 比（HEAD/base） | 判定 |
|---|---|---|---|---|
| 512  | 2.3024 | 3.9879  | 1.73 倍 | 非後退（改善） |
| 1024 | 5.8103 | 11.5496 | 1.99 倍 | 非後退（改善） |
| 2048 | 4.5662 | 21.5223 | 4.71 倍 | 非後退（改善） |
| 4096 | 0.3715 | 0.4000  | 1.08 倍 | 非後退（僅かな改善） |

**判定: 全形状で非後退（HEAD ≥ base）を確認。結線を維持する**（#1156 R5 の
承認条件を満たす）。

dim=4096 は base・HEAD いずれも他の dim（512〜2048）と比べ著しく低い TFLOPS
（1 TFLOPS 未満）であり、mma 優先化の効果が他形状ほど現れていない。これは
`CudaGemmAuto::run_f16` の auto 経路固有の別要因（`select_gemm_kernel` の
判定・転送コスト・4096 特有のオーバーヘッド等）が支配的である可能性が高いが、
本ドキュメントのスコープ（#1156 結線の非後退判定）には影響しない値であり
（base 比で後退していないため）、原因診断は別イシューのスコープとする
（`docs/perf/cuda-fresh-gemm-n2048-overhead-diagnosis.md` 等の先例と同様、
特定形状の残存オーバーヘッド診断は個別イシューで扱う）。

## 補足: 転送込み計測の位置づけ

`examples/gemm_mma_bench.rs::measure_mma_f16` は GPU 実行のみ（H2D/D2H を
計測区間外）を計測するのに対し、本ベンチ（`gemm_auto_f16_mma_switch_bench.rs`）
は `CudaGemmAuto::run_f16` の実利用経路（転送込み）をそのまま計測する。両者は
計測対象が異なるため TFLOPS 値を直接比較しない（`gemm_mma_bench.rs` 側の
既存注記と同じ整理）。
