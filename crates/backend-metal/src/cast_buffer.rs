//! cast カーネル専用の要素型 generic な Metal バッファ（イシュー
//! #1751・親 #1613）。
//!
//! [`crate::buffer::MetalBuffer`]（f32 専用）・[`crate::half_buffer::
//! MetalHalfBuffer`]（f16 専用）・[`crate::index_buffer::
//! MetalIndexBuffer`]（i32／u32 専用の 4 バイト固定）と同じ設計判断
//! （既存 f32 専用型のシグネチャに一切触れない独立した型として新設
//! する）を踏襲しつつ、cast が必要とする要素サイズの種類が多い
//! （i32・i64・u8）ため、要素型ごとに構造体を複製せず [`MetalCastBuffer<T>`]
//! を要素型 `T: Copy` で generic 化する（`T` が f32・f16・i32／u32 の
//! いずれとも異なる新しい dtype の組を要求するのは cast モジュール
//! だけであり、この汎化を他モジュールへ波及させる理由がないため
//! 独立モジュールに閉じる）。
//!
//! `crate::cast::MetalCast` から `i32`／`long`（`i64`）／`uchar`（`u8`。
//! `bool` の 0／1 実体化前の生表現）バッファの確保・アップロード・
//! readback に使う（f32 側は既存 [`crate::buffer::MetalBuffer`] を
//! そのまま再利用し、本モジュールでは扱わない）。

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::ffi::c_void;
use std::marker::PhantomData;

use crate::context::MetalContext;
use crate::error::MetalError;

pub(crate) type MtlBuffer = ProtocolObject<dyn MTLBuffer>;

/// 要素型 `T` を保持する Metal バッファ（[`crate::buffer::MetalBuffer`]
/// の要素型 generic 版。cast カーネル専用）。
///
/// `T: Copy` のみを要求する（`DeviceRepr` 相当の trait を新設しない。
/// `f32`／`i32`／`i64`／`u8` はいずれも POD〈plain old data〉で
/// FFI 越しの生バイトコピーが安全な単純値型のため、境界としては
/// `Copy` で十分と判断した）。
pub struct MetalCastBuffer<T> {
    buffer: Retained<MtlBuffer>,
    len: usize,
    _marker: PhantomData<T>,
}

/// `len` 要素分の `T` バッファのバイト長を検証付きで算出する
/// （`crate::buffer::checked_byte_len`／`crate::half_buffer::
/// checked_byte_len` と同型の汎化）。
fn checked_byte_len<T>(len: usize) -> Result<usize, MetalError> {
    if len == 0 {
        return Err(MetalError::ZeroLengthAllocation);
    }
    len.checked_mul(std::mem::size_of::<T>())
        .ok_or(MetalError::AllocationSizeOverflow { len })
}

impl<T: Copy> MetalCastBuffer<T> {
    /// `data` の内容を Metal バッファへアップロードして確保する
    /// （`crate::half_buffer::MetalHalfBuffer::new_with_data` と同型）。
    ///
    /// # Safety 境界
    /// `crate::buffer::MetalBuffer::new_with_data` と同一の契約
    /// （`newBufferWithBytes_length_options` は `bytes_len` バイトを
    /// 即座に複製し保持しない。`bytes_len` は直前の `checked_byte_len`
    /// により `data` の実バイト長と一致することを検証済み。`&[T]` の
    /// 先頭ポインタは長さ 0 でも非 null だが、`checked_byte_len` が
    /// 長さ 0 を事前拒否済みのためここでは非 0 長のみ扱う）。
    pub fn new_with_data(ctx: &MetalContext, data: &[T]) -> Result<Self, MetalError> {
        let len = data.len();
        let bytes_len = checked_byte_len::<T>(len)?;

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

        Ok(Self {
            buffer,
            len,
            _marker: PhantomData,
        })
    }

    /// `len` 要素分（ゼロ初期化）の Metal バッファを確保する
    /// （`crate::half_buffer::MetalHalfBuffer::new_zeroed` と同型）。
    /// cast カーネルの出力バッファ確保に使う。
    pub fn new_zeroed(ctx: &MetalContext, len: usize) -> Result<Self, MetalError> {
        let bytes_len = checked_byte_len::<T>(len)?;

        let buffer = ctx
            .device()
            .newBufferWithLength_options(bytes_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocation { bytes: bytes_len })?;

        Ok(Self {
            buffer,
            len,
            _marker: PhantomData,
        })
    }

    /// `crate::cast::encode_cast_dispatch` から参照される生バッファ。
    pub fn raw(&self) -> &MtlBuffer {
        &self.buffer
    }

    /// 確保済みの要素数。
    pub fn len(&self) -> usize {
        self.len
    }

    /// 要素数が 0 かどうか（clippy `len_without_is_empty` 対応。
    /// `crate::buffer::MetalBuffer::is_empty` と同じ判断根拠）。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// バッファの内容をホストへ読み出す（`crate::half_buffer::
    /// MetalHalfBuffer::read_to_vec` と同型）。
    ///
    /// # Safety 境界
    /// `crate::buffer::MetalBuffer::read_to_vec` と同一の契約
    /// （呼び出し元は GPU 側書き込みが `MetalContext::dispatch_sync`
    /// 等で完了済みであることを保証すること。`contents()` は
    /// `StorageModeShared` バッファの CPU 可視アドレスを返し、読み出す
    /// 要素数は確保時に検証済みの `self.len` に限定する）。
    pub fn read_to_vec(&self) -> Vec<T> {
        let ptr = self.buffer.contents();
        // SAFETY: 上記コメント参照。`self.len` は確保時に検証済みの要素数。
        let slice: &[T] = unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const T, self.len) };
        slice.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `checked_byte_len` がオーバーフローを型付きエラーで拒否する
    /// ことの機械検証（Linux 実行可能。`objc2` FFI に触れない）。
    #[test]
    fn checked_byte_len_rejects_overflow() {
        let err = checked_byte_len::<i64>(usize::MAX).unwrap_err();
        assert!(matches!(err, MetalError::AllocationSizeOverflow { .. }));
    }

    #[test]
    fn checked_byte_len_rejects_zero_length() {
        let err = checked_byte_len::<i32>(0).unwrap_err();
        assert!(matches!(err, MetalError::ZeroLengthAllocation));
    }

    #[test]
    fn checked_byte_len_accepts_ordinary_length() {
        assert_eq!(checked_byte_len::<i32>(4).unwrap(), 16);
        assert_eq!(checked_byte_len::<i64>(4).unwrap(), 32);
        assert_eq!(checked_byte_len::<u8>(4).unwrap(), 4);
    }
}
