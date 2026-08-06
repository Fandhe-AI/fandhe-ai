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
//! A/B packing）を追加した。`gemm` モジュールの関数（naive/blocked/parallel/
//! parallel_tuned）は #24 の段階比較の参照点として変更しない（公開 API 非破壊。
//! `gemm_blis` は独立した新規追加）。`BackendOps` トレイトからの結線は TASK-1.9（#43）で
//! 行う（spec 根拠: `docs/spec/05-tasks.md` TASK-1.1・TASK-1.6）。

mod elementwise;
pub mod gemm;
pub mod gemm_blis;
pub mod reduction;

pub use elementwise::{
    add, add_slice, exp, exp_slice, mul, mul_slice, relu, relu_slice, tanh, tanh_slice,
};
pub use gemm::{
    BlockSizes, GemmError, gemm_blocked, gemm_naive, gemm_parallel, gemm_parallel_tuned,
};
pub use gemm_blis::{gemm_blis, gemm_blis_parallel};
