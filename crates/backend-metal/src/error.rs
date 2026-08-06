//! `backend-metal` 公開入口の型付きエラー。
//!
//! デバイス取得・コマンドキュー生成・バッファ確保の失敗を `Option`
//! （PoC-v2-4 の `MetalGemm::new` 等が返していた形）ではなく
//! `Result<_, MetalError>` として呼び出し元（TASK-1.8b/c の GEMM・
//! simdgroup カーネル実装。#39・#40）へ伝える。本番経路で `unwrap()` /
//! `expect()` を使わない方針（`.claude/rules/coding-rust.md`）に従い、
//! [`crate::context`]・[`crate::buffer`] はここで定義するバリアントのみ
//! を返す。

use std::fmt;

/// `backend-metal` の基盤層（デバイス・キュー・バッファ）で発生しうる
/// エラー。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、TASK-1.8b 以降でパイプライン
/// 構築・ディスパッチ関連のバリアントが増えても呼び出し側の網羅的
/// match を破壊しないため（`backend-cpu::GemmError` と同方針。
/// `crates/backend-cpu/src/gemm.rs` 参照）。
#[non_exhaustive]
#[derive(Debug)]
pub enum MetalError {
    /// `MTLCreateSystemDefaultDevice` が `None` を返した
    /// （Metal 非対応環境・GPU 無効化等）。
    DeviceUnavailable,
    /// `MTLDevice::newCommandQueue` が `None` を返した。
    CommandQueueCreation,
    /// `MTLDevice::newBufferWithBytes_length_options` /
    /// `newBufferWithLength_options` が `None` を返した
    /// （メモリ不足等。要求バイト数を診断用に保持する）。
    BufferAllocation { bytes: usize },
    /// バッファ長（要素数 × `size_of::<f32>()`）の算出が `usize` の
    /// 範囲でオーバーフローする（`checked_mul` によりアクセス前に検出
    /// する。OWASP A03 観点。`.claude/rules/security.md`。将来
    /// safetensors/ONNX 由来の形状がここへ流入しうるための前段検証）。
    AllocationSizeOverflow { len: usize },
    /// バッファ確保時に長さ 0 が渡された（0 バイトバッファは Metal 側の
    /// 挙動が不定であり、呼び出し側の形状検証漏れを早期に拒否する）。
    ZeroLengthAllocation,
    /// `MTLCommandQueue::commandBuffer` が `None` を返した。
    CommandBufferCreation,
    /// `MTLCommandBuffer::computeCommandEncoder` が `None` を返した。
    ComputeEncoderCreation,
    /// `waitUntilCompleted()` 完了後、コマンドバッファの `status` が
    /// `MTLCommandBufferStatus::Error` だった（GPU 側の fault・OOM・
    /// discarded work 等）。`commit()` 自体は成功として返るため、
    /// [`crate::context::MetalContext::dispatch_sync`] は完了後にこの
    /// 状態を確認しない限り GPU 側の失敗を `Ok(())` として握り潰して
    /// しまう（今後の GEMM 実装で出力バッファの古い／不完全な内容を
    /// 読む無言の数値誤りにつながるため、型付きエラーとして呼び出し元
    /// へ伝える）。`message` は `MTLCommandBuffer::error()` の
    /// `NSError` から得た診断用の文字列表現。
    CommandBufferExecutionFailed { message: String },
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalError::DeviceUnavailable => {
                write!(
                    f,
                    "MTLCreateSystemDefaultDevice returned None (no Metal device available)"
                )
            }
            MetalError::CommandQueueCreation => {
                write!(f, "MTLDevice::newCommandQueue returned None")
            }
            MetalError::BufferAllocation { bytes } => {
                write!(f, "Metal buffer allocation failed for {bytes} bytes")
            }
            MetalError::AllocationSizeOverflow { len } => {
                write!(
                    f,
                    "buffer byte length overflows usize for len={len} elements"
                )
            }
            MetalError::ZeroLengthAllocation => {
                write!(f, "buffer allocation requested with zero length")
            }
            MetalError::CommandBufferCreation => {
                write!(f, "MTLCommandQueue::commandBuffer returned None")
            }
            MetalError::ComputeEncoderCreation => {
                write!(f, "MTLCommandBuffer::computeCommandEncoder returned None")
            }
            MetalError::CommandBufferExecutionFailed { message } => {
                write!(
                    f,
                    "Metal command buffer completed with MTLCommandBufferStatus::Error: {message}"
                )
            }
        }
    }
}

impl std::error::Error for MetalError {}
