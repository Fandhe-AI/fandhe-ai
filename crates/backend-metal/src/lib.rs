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
//! カーネル本体は TASK-1.8 で追加する（spec 根拠: `docs/spec/05-tasks.md` TASK-1.1・TASK-1.8）。
//!
//! TASK-1.9a（#44）で [`device`] モジュール（[`device::MetalDeviceProvider`]）を追加した。
//! `tensor_core::device::DeviceProvider` の Metal 実装であり、CPU／CUDA 実装
//! （`backend-cpu::CpuDeviceProvider`／`backend-cuda::device::CudaDeviceProvider`）と
//! 同一 trait で列挙・選択できることを macOS 実機上のテストで検証する。`Device::Metal`
//! 自体が `cfg(target_os = "macos")` 限定のため、本モジュールもクレート全体でこの cfg を
//! 付す（非 macOS 環境のビルドに影響を与えない）。

#[cfg(target_os = "macos")]
pub mod device;

#[cfg(target_os = "macos")]
pub use device::MetalDeviceProvider;
