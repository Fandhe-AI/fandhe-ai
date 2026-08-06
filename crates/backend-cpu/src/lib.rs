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

mod device;
mod elementwise;
pub mod gemm;
pub mod gemm_blis;
pub mod parity;
pub mod reduction;

pub use device::CpuDeviceProvider;
pub use elementwise::{
    add, add_slice, exp, exp_slice, mul, mul_slice, relu, relu_slice, tanh, tanh_slice,
};
pub use gemm::{
    BlockSizes, GemmError, gemm_blocked, gemm_naive, gemm_parallel, gemm_parallel_tuned,
};
pub use gemm_blis::{gemm_blis, gemm_blis_parallel};
pub use parity::{
    ABSOLUTE_RESCUE_THRESHOLD, CompareReport, ParityError, RELATIVE_TOLERANCE, assert_parity,
    compare, matmul_reference_fma,
};
