//! `i32`／`u32` 要素を保持する Metal バッファ（gather／scatter カーネル
//! 専用の入力バッファ。イシュー #1778）。
//!
//! [`crate::buffer::MetalBuffer`]（f32 専用）・[`crate::half_buffer::
//! MetalHalfBuffer`]（f16 専用）と同じ設計判断で、既存 f32 専用型の
//! シグネチャに一切触れない独立した型として新設する。`gather_scatter.rs`
//! が `index`（i32）・`shapes`（u32）バッファの確保に使う。読み戻し API
//! は持たない（入力専用。gather／scatter の出力はいずれも f32 で
//! `crate::buffer::MetalBuffer` を使う）。

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::ffi::c_void;

use crate::context::MetalContext;
use crate::error::MetalError;

pub(crate) type MtlBuffer = ProtocolObject<dyn MTLBuffer>;

/// `i32`／`u32` の入力専用 Metal バッファ（[`crate::buffer::MetalBuffer`]
/// の整数版。読み戻しは持たない）。
#[derive(Debug)]
pub struct MetalIndexBuffer {
    buffer: Retained<MtlBuffer>,
}

/// `len` 要素分（4 バイト要素想定）のバイト長を検証付きで算出する
/// （`crate::buffer::checked_byte_len`／`crate::half_buffer::
/// checked_byte_len` と同型）。
fn checked_byte_len(len: usize) -> Result<usize, MetalError> {
    if len == 0 {
        return Err(MetalError::ZeroLengthAllocation);
    }
    len.checked_mul(std::mem::size_of::<u32>())
        .ok_or(MetalError::AllocationSizeOverflow { len })
}

impl MetalIndexBuffer {
    /// `data`（`i32`）をアップロードして確保する。
    ///
    /// # Safety 境界
    /// `crate::buffer::MetalBuffer::new_with_data` と同一の契約
    /// （`newBufferWithBytes_length_options` は `bytes_len` バイトを即座に
    /// 複製し保持しない。`bytes_len` は直前の `checked_byte_len` により
    /// `data` の実バイト長と一致することを検証済み。`&[i32]` の先頭
    /// ポインタは長さ 0 でも非 null）。
    pub fn new_with_i32(ctx: &MetalContext, data: &[i32]) -> Result<Self, MetalError> {
        let bytes_len = checked_byte_len(data.len())?;
        // SAFETY: 上記コメント参照。ポインタ・長さともに確保直前に検証済み。
        let ptr = unsafe { std::ptr::NonNull::new_unchecked(data.as_ptr() as *mut c_void) };
        // SAFETY: 上記コメント参照。
        let buffer = unsafe {
            ctx.device().newBufferWithBytes_length_options(
                ptr,
                bytes_len,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAllocation { bytes: bytes_len })?;
        Ok(Self { buffer })
    }

    /// `data`（`u32`）をアップロードして確保する（`shapes`／`rank`／
    /// `dim`／`numel` の 1 要素 `constant uint&` バッファではなく、
    /// `shapes` の複数要素配列バッファ〈`constant uint*`〉として使う。
    /// 単一スカラー引数は既存 `setBytes_length_atIndex` を使う——バッファ
    /// 確保は行わない）。
    ///
    /// # Safety 境界
    /// [`Self::new_with_i32`] と同一の契約（`&[u32]` の先頭ポインタは
    /// 長さ 0 でも非 null）。
    pub fn new_with_u32(ctx: &MetalContext, data: &[u32]) -> Result<Self, MetalError> {
        let bytes_len = checked_byte_len(data.len())?;
        // SAFETY: 上記コメント参照。
        let ptr = unsafe { std::ptr::NonNull::new_unchecked(data.as_ptr() as *mut c_void) };
        // SAFETY: 上記コメント参照。
        let buffer = unsafe {
            ctx.device().newBufferWithBytes_length_options(
                ptr,
                bytes_len,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAllocation { bytes: bytes_len })?;
        Ok(Self { buffer })
    }

    /// `crate::gather_scatter` のエンコード関数から参照される生バッファ。
    pub fn raw(&self) -> &MtlBuffer {
        &self.buffer
    }
}
