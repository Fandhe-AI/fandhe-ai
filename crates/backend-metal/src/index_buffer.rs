//! `i32`／`u32` 要素を保持する Metal バッファ（gather／scatter カーネル
//! 専用の入力バッファ。イシュー #1778）。
//!
//! [`crate::buffer::MetalBuffer`]（f32 専用）・[`crate::half_buffer::
//! MetalHalfBuffer`]（f16 専用）と同じ設計判断で、既存 f32 専用型の
//! シグネチャに一切触れない独立した型として新設する。`gather_scatter.rs`
//! が `index`（i32）・`shapes`（u32）バッファの確保に使う。当初は
//! 読み戻し API を持たない入力専用設計だったが、`unique`（イシュー
//! #1734）のビットニックソート結果（`u32` キー配列）を GPU 側で
//! `dedup` せずホストへ読み戻す必要があるため、[`Self::
//! read_to_vec_u32`]（`u32` キー readback 専用。イシュー #1734）のみを
//! 追加した（gather／scatter の出力自体は引き続き f32 で
//! `crate::buffer::MetalBuffer` を使う）。

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::ffi::c_void;

use crate::context::MetalContext;
use crate::error::MetalError;

pub(crate) type MtlBuffer = ProtocolObject<dyn MTLBuffer>;

/// `i32`／`u32` の入力専用 Metal バッファ（[`crate::buffer::MetalBuffer`]
/// の整数版）。`len`（要素数）は `unique.rs::MetalUnique` 相当の
/// readback（[`Self::read_to_vec_u32`]）のために保持する。
#[derive(Debug)]
pub struct MetalIndexBuffer {
    buffer: Retained<MtlBuffer>,
    len: usize,
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
        Ok(Self {
            buffer,
            len: data.len(),
        })
    }

    /// `data`（`u32`）をアップロードして確保する（`shapes`／`rank`／
    /// `dim`／`numel` の 1 要素 `constant uint&` バッファではなく、
    /// `shapes` の複数要素配列バッファ〈`constant uint*`〉として使う。
    /// 単一スカラー引数は既存 `setBytes_length_atIndex` を使う——バッファ
    /// 確保は行わない）。`unique.rs` はソート対象の `u32` キー配列
    /// （padded 長）の確保にも本メソッドを使う。
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
        Ok(Self {
            buffer,
            len: data.len(),
        })
    }

    /// `crate::gather_scatter`／`crate::unique` のエンコード関数から
    /// 参照される生バッファ。
    pub fn raw(&self) -> &MtlBuffer {
        &self.buffer
    }

    /// バッファの内容（`u32`）をホストへ読み出し新規 `Vec` として
    /// コピーする（`unique.rs::MetalUnique::run_unique_f32` 専用の
    /// readback。イシュー #1734）。
    ///
    /// # Safety 境界（`crate::buffer::MetalBuffer::read_to_vec` と同型）
    /// 呼び出し元は次の 2 点を保証すること:
    /// 1. このバッファへの GPU 側書き込みが `context.rs::
    ///    MetalContext::dispatch_sync`（同期完了を保証するディスパッチ）
    ///    等で完了していること。
    /// 2. 返却後、`self` が指す同一バッファへ新たな GPU 側書き込みを
    ///    積まないこと（本メソッド自体は借用を式の評価内に閉じ込めて
    ///    即座にコピーするため、この契約は呼び出し元の使用パターン
    ///    〈本メソッド呼び出し前に synchronize 済みであること〉にのみ
    ///    依存する）。
    ///
    /// `contents()` は `StorageModeShared` バッファの CPU 可視アドレス
    /// を返す（確保時に `MTLResourceOptions::StorageModeShared` を指定
    /// しているため CPU から直接参照可能）。読み出す要素数は確保時に
    /// 記録した `self.len` に限定しており、確保バイト数を超えて読むこと
    /// はない。
    pub fn read_to_vec_u32(&self) -> Vec<u32> {
        let ptr = self.buffer.contents();
        // SAFETY: 呼び出し元は上記 2 条件（同期済み・排他書き込みなし）
        // を満たす契約。`self.len` は確保時の `checked_byte_len` で
        // 検証済みのバイト数に対応する要素数であり、確保バイト数を
        // 超えて読むことはない。借用は本メソッド内の `.to_vec()` 呼び
        // 出しで即座にコピーされ、この式の評価が終わるまでの間のみ
        // 生存する一時値である（`crate::buffer::MetalBuffer::
        // read_to_vec` と同じ単一式内完結パターン）。
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const u32, self.len) }.to_vec()
    }
}
