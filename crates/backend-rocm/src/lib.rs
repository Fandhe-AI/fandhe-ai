//! ROCm（AMD GPU）バックエンドの雛形クレート（イシュー #3154・親 #3150）。
//!
//! 役割: HIP ランタイムの有無を probe する。`libloading`（第 12 区分）で実行時にロードし、
//! `#[link]` を使わないため ROCm 非搭載環境でもビルド・テストが通る
//! （cudarc の dynamic-loading と同じ cfg ベース方針。REQ-2・coding-rust.md）。
//! この段階ではカーネルを実行しない（HIPRTC・カーネルは #3155、facade の Device 統合は
//! #3157、unsafe 監査記録は #3158、実機 parity は #3159）。
//!
//! 設計の正本は `docs/backend-rocm-hip-ffi-decision.md`。facade からは参照しない。

mod error;
mod hip;

pub use error::RocmError;

/// HIP ランタイムをロードでき、必須シンボルを解決できるとき true。
///
/// ROCm 非搭載・ロード失敗・非対応 OS ではすべて false を返し、panic しない。
/// 雛形段階ではデバイス数までは確認しない（`hipError_t` の ABI 確認後に #3155 で追加）。
pub fn is_available() -> bool {
    hip::load().is_ok()
}
