//! Metal バックエンド。
//!
//! `tensor-core` の演算グラフノードを MSL カーネル（simdgroup 系命令を含む）へ変換して実行する。
//! バックエンド切替は feature フラグなしの cfg ベース（PoC-v2-5 実証構成。REQ-2）とし、
//! `objc2` / `objc2-foundation` / `objc2-metal` は `cfg(target_os = "macos")` で分離する
//! （非 macOS 環境のビルドに影響を与えない。`.claude/rules/deps-policy.md`）。
//!
//! `backend-cpu` との数値一致は統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」で
//! 検証する。丸め方針（FMA 契約）は Metal `simdgroup_multiply_accumulate` の既定 FMA 契約を
//! CPU 参照実装（`f32::mul_add`）と揃える（PoC-v2-5 の K=4096 ストレスケースで実測確認済み。
//! `.claude/rules/coding-rust.md`）。カーネルの手動境界検査は最適化を理由に省略しない（REQ-8）。
//! FFI 境界の `unsafe`（objc2 系）は必要最小限に留め理由コメントを付す
//! （`.claude/rules/security.md`）。
//!
//! TASK-1.8a（#38）でデバイス・コマンドキュー・バッファ管理の基盤（[`context::MetalContext`]・
//! [`buffer::MetalBuffer`]・[`error::MetalError`]）を実装済み。MSL ライブラリのコンパイル・
//! パイプライン構築・ディスパッチ経路（naive GEMM）は TASK-1.8b（#39）、simdgroup カーネルは
//! TASK-1.8c（#40）で追加する（spec 根拠: `docs/spec/05-tasks.md` TASK-1.1・TASK-1.8）。
//!
//! 非 macOS 環境ではモジュールごとコンパイル対象外になる（feature フラグなしの cfg ベース。
//! PoC-v2-5 実証構成・REQ-2）。

#[cfg(target_os = "macos")]
pub mod buffer;
#[cfg(target_os = "macos")]
pub mod context;
#[cfg(target_os = "macos")]
pub mod error;

#[cfg(target_os = "macos")]
pub use buffer::MetalBuffer;
#[cfg(target_os = "macos")]
pub use context::MetalContext;
#[cfg(target_os = "macos")]
pub use error::MetalError;
