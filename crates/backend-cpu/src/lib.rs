//! CPU バックエンド（参照実装）。
//!
//! `tensor-core` の演算グラフノードを CPU カーネルへ変換して実行する。バックエンド切替は
//! feature フラグなしの cfg ベースを基本とし（PoC-v2-5 実証構成。REQ-2）、本バックエンドは
//! 無条件で有効化される数値一致の参照点となる。並列化は `backend-cpu` 固有の許容依存である
//! `rayon` を用いる（PoC-v2-1 で naive/blocked 比 約 6〜8.5 倍改善を実測。
//! `.claude/rules/deps-policy.md`）。
//!
//! `backend-cuda` / `backend-metal` との数値一致は統一複合判定「相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満」で検証する。丸め方針（FMA 契約）は GPU 側の既定 FMA 契約と揃えるため
//! `f32::mul_add` を用いる（PoC-v2-5 の K=4096 ストレスケースで実測確認済み。
//! `.claude/rules/coding-rust.md`）。カーネルの手動境界検査は最適化を理由に省略しない（REQ-8）。
//!
//! TASK-1.6a（#21）で GEMM カーネル（[`gemm`] モジュール。naive / blocked / rayon 並列の
//! 3 段構成）を追加した。TASK-1.6b（#22）で elementwise カーネル（二項演算 `add`・`mul`、
//! 活性化 `relu`・`exp`・`tanh`）を追加した。TASK-1.6c（#23）で `reduction`（`sum`・`max`・
//! `mean`・軸指定 reduction）を追加した。TASK-1.6d（#24）で `rayon` 並列の粒度・
//! ブロックサイズ（[`gemm::BlockSizes`]）を実測チューニングし、PoC-v2-1 比の性能改善比
//! （naive/blocked 比 約 6〜8.5 倍）が本環境でも再現することを確認した
//! （計測記録: `docs/perf/cpu-gemm-rayon-tuning.md`）。TASK-1.6f（#184）で [`gemm_blis`]
//! モジュール（BLIS/GotoBLAS2 5-loop model・`std::arch` intrinsics マイクロカーネル・
//! A/B packing）を追加した。TASK-1.6g（#185）で `gemm_blis`／`gemm_blis_parallel` の
//! マイクロカーネル選択をコンパイル時 cfg のみから実行時 CPU 機能検出（NEON／AVX2／
//! AVX-512。`gemm_blis::microkernel` の `Isa::detect`）による dispatch へ拡張した。
//! `gemm` モジュールの関数（naive/blocked/parallel/parallel_tuned）は #24 の段階比較の
//! 参照点として変更しない（公開 API 非破壊。`gemm_blis` は独立した新規追加でシグネチャも
//! #185 で変更しない）。`BackendOps` トレイトからの結線は TASK-1.9（#43）で行う
//! （spec 根拠: `docs/spec/05-tasks.md` TASK-1.1・TASK-1.6）。
//!
//! TASK-1.9a（#44）で [`device`] モジュール（[`device::CpuDeviceProvider`]）を追加した。
//! `tensor_core::device::DeviceProvider` の CPU 実装であり、CUDA／Metal 実装
//! （`backend-cuda::device::CudaDeviceProvider`／`backend-metal::device::MetalDeviceProvider`）
//! と同一 trait で列挙・選択できることを `tests/device_provider_integration.rs` で検証する。
//!
//! TASK-2.2a（#53）で [`parity`] モジュール（REQ-2 統一複合判定ユーティリティ・
//! FMA 契約参照 matmul）を追加した。#54（CPU-CUDA ペア）・#55（CPU-Metal ペア）は
//! 本モジュールの `parity::compare`／`parity::assert_parity`／
//! `parity::matmul_reference_fma` を共通利用し、ペアごとに判定ロジックを
//! 重複実装しない想定である（`docs/spec/05-tasks.md` TASK-2.2）。
//!
//! TASK-1.9b（#45）で [`memory`] モジュール（[`memory::CpuMemory`]）を追加した。
//! `tensor_core::buffer::MemoryOps` の CPU 実装であり、`upload`/`download`/
//! `alloc_zeroed` は FFI を伴わず `Vec<f32>` の複製のみで完結する
//! （`backend-cuda::CudaMemory`／`backend-metal::MetalMemory` の数値一致の
//! 参照点。`.claude/rules/coding-rust.md` の「CPU 参照実装」方針）。
//! TASK-14.1a（#174）で `tensor_core::memory_stats::MemoryStats` を実装し、
//! 確保済みバイト数のピーク値を取得できるようにした（`CpuMemory` は
//! `Arc<AllocationTracker>` を共有する非 `Copy` 型に変更。CUDA/Metal への
//! 同フック組み込みは #175）。
//!
//! **破壊的変更（`backend-cpu` 0.2.0。PR #359 codex-review 指摘 P1 を受けて
//! ここに移行手引きを明記）**: 従来 `CpuMemory` はフィールドを持たない
//! unit struct（`Copy`）だったため `let mem = CpuMemory;` のような直接構築が
//! 可能だったが、本変更でトラッカー（`Arc<AllocationTracker>`）を保持する
//! ようになり `Copy` を外した。ワークスペース内に unit struct 構築や
//! `Copy` 依存箇所が無いことは確認済みだが、外部呼び出し側は次のとおり
//! 移行する:
//! - `CpuMemory` → [`CpuMemory::new()`]（または `CpuMemory::default()`。
//!   いずれも新規の計測系列を持つトラッカーを生成する）
//! - `Copy` 依存（暗黙コピーでの使い回し）→ `Clone`（`clone()` は
//!   同一計測系列〈トラッカー〉の共有を意味し、暗黙コピーとは意味が異なる
//!   点に注意。ピークを集約したい場合は明示的に `clone()` する）
//!
//! TASK-1.9c（#46）で `ops` モジュール（[`ops::CpuBackendOps`]）を追加した。
//! `tensor_core::backend_ops::BackendOps` の CPU 実装であり、既存カーネル
//! （[`gemm_blis::gemm_blis_parallel`]・[`elementwise`] の `add`/`mul`/`relu`/
//! `exp`/`tanh`・[`reduction`] の `sum`/`max`）への薄い委譲に徹する。CUDA／
//! Metal 実装（`backend-cuda::ops::CudaBackendOps`／
//! `backend-metal::ops::MetalBackendOps`）と同一 trait でカーネルディスパッチ
//! できることを `tests/backend_ops_dispatch.rs` で検証する。
//!
//! TASK-12.1f（#203）で [`gemm_blis::gemm_blis_bias_act_parallel`]（GEMM epilogue
//! 〈bias 加算・activation〉のカーネル内融合）を追加し、[`ops::CpuBackendOps`] の
//! `gemm_bias_act`（`tensor_core::BackendOps` のデフォルトメソッド。非融合合成）を
//! オーバーライドして接続した。非融合実行（`gemm` → `add` → `relu` の 3 パス・中間
//! `Tensor` 2 個割当）に対する性能改善は `docs/perf/cpu-gemm-epilogue-fusion.md` に
//! 実測記録している（CUTLASS 系実測の動機は平均 1.38〜1.45 倍。本環境実測は 1.46〜
//! 2.56 倍）。融合版と非融合合成の bit 完全一致は `tests/gemm_epilogue_parity.rs` で
//! 検証する。CUDA／Metal は GPU カーネル内 epilogue 融合をスコープ外とし、
//! `gemm_bias_act` のデフォルト実装（elementwise 未実装のため `Unsupported`）に留める。
//!
//! TASK-12.1c（#163）で [`fused_elementwise`] モジュール
//! （[`fused_elementwise::run_fused_elementwise`]）を追加した。
//! `tensor_core::fusion`（TASK-12.1a〜c・#161〜#163）が検出・生成した
//! elementwise 連鎖（`tensor_core::FusionPlan`）を、per-op カーネル
//! （[`elementwise`]）の逐次合成ではなく単一パスのレジスタ内評価で実行
//! する CPU 参照実装である（PoC-9 `ElemwiseFuse` 方式。詳細は
//! `fused_elementwise` モジュール冒頭コメント）。`tensor_core::BackendOps::
//! run_fused`（trait への追加・[`ops::CpuBackendOps`] での override 実装）
//! への結線は #164 のスコープであり、#163 時点では関数ベースのカーネル
//! API として独立に提供する（[`gemm`]／[`elementwise`] と同じ「trait
//! 定義なし・関数ベース」構成）。数値契約は per-op カーネルと完全に
//! 揃え、融合の有無で許容誤差・演算定義を変えない（`tests/
//! fused_elementwise_parity.rs` で融合 vs 非融合の数値一致を検証する。
//! 受け入れ条件）。

mod device;
mod elementwise;
pub mod fused_elementwise;
pub mod gemm;
pub mod gemm_blis;
pub mod memory;
mod ops;
pub mod parity;
pub mod reduction;

pub use device::CpuDeviceProvider;
pub use elementwise::{
    add, add_slice, exp, exp_slice, mul, mul_slice, relu, relu_slice, tanh, tanh_slice,
};
pub use fused_elementwise::run_fused_elementwise;
pub use gemm::{
    BlockSizes, GemmError, gemm_blocked, gemm_naive, gemm_parallel, gemm_parallel_tuned,
};
pub use gemm_blis::{gemm_blis, gemm_blis_bias_act_parallel, gemm_blis_parallel};
pub use memory::CpuMemory;
pub use ops::CpuBackendOps;
pub use parity::{
    ABSOLUTE_RESCUE_THRESHOLD, CompareReport, ParityError, RELATIVE_TOLERANCE, assert_parity,
    compare, matmul_reference_fma,
};
