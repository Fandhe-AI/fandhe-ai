//! f16 GEMM 用 Metal バッファ確保・アップロード・readback（TASK-8.3b・#156）。
//!
//! [`crate::buffer::MetalBuffer`]（f32 専用）とは別の独立した型として持つ。
//! `MetalBuffer` のシグネチャ（`checked_byte_len` の `size_of::<f32>()`
//! 決め打ち・`read_to_vec() -> Vec<f32>`）を変更すると `crate::gemm` の
//! 既存 f32 経路・既存パリティテスト全体へ波及するため、f16 専用の平行な
//! 型を新設して既存コードに触れない設計にする（実装計画 §2 の判断）。
//! `crate::gemm::MetalGemm::dispatch_f16` が [`MetalHalfBuffer::new_with_data`]
//! で A・B（half）を、`gemm_simdgroup_f16`（`shaders/gemm.metal`）の出力
//! C（half。累算精度契約は同カーネルのコメント参照）は
//! [`MetalHalfBuffer::new_zeroed`] で確保する。
//!
//! `StorageModeShared`・`unsafe` の使用契約は [`crate::buffer::MetalBuffer`]
//! と同一（本ファイルのコメントは差分のみ記す。詳細な SAFETY 根拠は
//! `buffer.rs` を参照）。

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::ffi::c_void;

use crate::context::MetalContext;
use crate::error::MetalError;

pub(crate) type MtlBuffer = ProtocolObject<dyn MTLBuffer>;

/// `half::f16` 要素を保持する Metal バッファのラッパー
/// （[`crate::buffer::MetalBuffer`] の f16 版）。
///
/// `Debug` 導出の理由は `crate::buffer::MetalBuffer` と同じ
/// （`#[cfg(test)]`／`#[ignore]` テストの `unwrap_err()` が `T: Debug` を
/// 要求するため）。
#[derive(Debug)]
pub struct MetalHalfBuffer {
    buffer: Retained<MtlBuffer>,
    len: usize,
}

/// `len` 要素分の `half::f16` バッファのバイト長を検証付きで算出する
/// （`crate::buffer::checked_byte_len` の f16 版）。
fn checked_byte_len(len: usize) -> Result<usize, MetalError> {
    if len == 0 {
        return Err(MetalError::ZeroLengthAllocation);
    }
    len.checked_mul(std::mem::size_of::<half::f16>())
        .ok_or(MetalError::AllocationSizeOverflow { len })
}

impl MetalHalfBuffer {
    /// `data` の内容を Metal バッファへアップロードして確保する
    /// （`crate::buffer::MetalBuffer::new_with_data` の f16 版）。
    ///
    /// # Safety 境界
    /// `crate::buffer::MetalBuffer::new_with_data` と同一の契約
    /// （`newBufferWithBytes_length_options` は `bytes_len` バイトを即座に
    /// 複製し保持しない。`bytes_len` は直前の `checked_byte_len` により
    /// `data` の実バイト長と一致することを検証済み）。
    pub fn new_with_data(ctx: &MetalContext, data: &[half::f16]) -> Result<Self, MetalError> {
        let len = data.len();
        let bytes_len = checked_byte_len(len)?;

        // SAFETY: `crate::buffer::MetalBuffer::new_with_data` の SAFETY
        // コメントと同一の契約（`&[half::f16]` の先頭ポインタは長さ 0 でも
        // 非 null。`checked_byte_len` が長さ 0 を事前拒否済み）。
        let ptr = unsafe { std::ptr::NonNull::new_unchecked(data.as_ptr() as *mut c_void) };

        // SAFETY: 上記コメント参照。ポインタ・長さともに確保直前に検証済み。
        let buffer = unsafe {
            ctx.device().newBufferWithBytes_length_options(
                ptr,
                bytes_len,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAllocation { bytes: bytes_len })?;

        Ok(Self { buffer, len })
    }

    /// `len` 要素分（ゼロ初期化）の Metal バッファを確保する
    /// （`crate::buffer::MetalBuffer::new_zeroed` の f16 版）。
    /// `gemm_simdgroup_f16` の出力バッファ（C）確保に使う。
    pub fn new_zeroed(ctx: &MetalContext, len: usize) -> Result<Self, MetalError> {
        let bytes_len = checked_byte_len(len)?;

        let buffer = ctx
            .device()
            .newBufferWithLength_options(bytes_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocation { bytes: bytes_len })?;

        Ok(Self { buffer, len })
    }

    /// 確保済みの要素数。
    pub fn len(&self) -> usize {
        self.len
    }

    /// 要素数が 0 かどうか（`crate::buffer::MetalBuffer::is_empty` と同じ
    /// 判断根拠。clippy `len_without_is_empty` 対応）。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `crate::gemm::encode_dispatch_f16` から参照される生バッファ。
    pub fn raw(&self) -> &MtlBuffer {
        &self.buffer
    }

    /// バッファの内容をホストへ読み出す
    /// （`crate::buffer::MetalBuffer::read_to_vec` の f16 版）。
    ///
    /// # Safety 境界
    /// `crate::buffer::MetalBuffer::read_to_vec` と同一の契約
    /// （`contents()` は `StorageModeShared` バッファの CPU 可視アドレス。
    /// 読み出す要素数は確保時に検証済みの `self.len` に限定する）。
    pub fn read_to_vec(&self) -> Vec<half::f16> {
        let ptr = self.buffer.contents();
        // SAFETY: 上記コメント参照。`self.len` は確保時に検証済みの要素数。
        let slice: &[half::f16] =
            unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const half::f16, self.len) };
        slice.to_vec()
    }
}
