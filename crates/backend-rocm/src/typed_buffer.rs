//! 要素型付きデバイスバッファ（イシュー #3439・分解案 R8）。
//!
//! 役割: `module::DeviceBuffer`（f32 専用・GEMM 経路。変更しない）に対し、gather/scatter・sort・
//! unique が必要とする `i32`・`u32`・`u64`、縮約 sum の partial 用 `f64` も扱えるようにした RAII バッファ。各演算族の実行型
//! （`scan.rs` 等）が入力・出力・補助配列（shape/stride）に使う。確保・コピーのバイト数は
//! `checked_mul` で求め、upload/download は長さが一致するときだけ通す。Drop で 1 回だけ
//! `hipFree`（失敗は無視。呼び出し側は実行後に同期済みの契約）。unsafe の監査記録は
//! `docs/backend-rocm-unsafe-audit.md` §3.5。

use crate::error::RocmError;
use crate::hip::HipRuntime;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for f64 {}
}

/// デバイスバッファの要素型（封印。ビット列として memcpy してよい POD のみ）。
pub(crate) trait DeviceElem: sealed::Sealed + Copy + Default {
    /// 1 要素のバイト数。
    const SIZE: usize;
}
impl DeviceElem for f32 {
    const SIZE: usize = 4;
}
impl DeviceElem for i32 {
    const SIZE: usize = 4;
}
impl DeviceElem for u32 {
    const SIZE: usize = 4;
}
impl DeviceElem for u64 {
    const SIZE: usize = 8;
}
/// 縮約 sum の `double` partial バッファ用（R3・#3434）。`f64` はビット列として memcpy してよい POD。
impl DeviceElem for f64 {
    const SIZE: usize = 8;
}

/// `T` 要素のデバイスバッファ。
pub(crate) struct TypedDeviceBuffer<T: DeviceElem> {
    ptr: *mut c_void,
    elems: usize,
    rt: Arc<HipRuntime>,
    _t: PhantomData<T>,
}

impl<T: DeviceElem> TypedDeviceBuffer<T> {
    /// `elems` 要素を確保する（0 要素でも有効なポインタを得るため最小 `T::SIZE` バイト）。
    pub(crate) fn new(rt: Arc<HipRuntime>, elems: usize) -> Result<Self, RocmError> {
        let bytes = checked_bytes::<T>(elems)?.max(T::SIZE);
        let ptr = rt.malloc(bytes)?;
        Ok(Self {
            ptr,
            elems,
            rt,
            _t: PhantomData,
        })
    }

    /// 確保後に 0 で初期化する。
    pub(crate) fn new_zeroed(rt: Arc<HipRuntime>, elems: usize) -> Result<Self, RocmError> {
        let buf = Self::new(rt, elems)?;
        buf.upload(&vec![T::default(); elems])?;
        Ok(buf)
    }

    /// ホスト配列から確保・転送する。
    pub(crate) fn from_slice(rt: Arc<HipRuntime>, src: &[T]) -> Result<Self, RocmError> {
        let buf = Self::new(rt, src.len())?;
        buf.upload(src)?;
        Ok(buf)
    }

    /// 確保した要素数。
    pub(crate) fn len(&self) -> usize {
        self.elems
    }

    /// カーネル引数に渡すデバイスポインタ。
    pub(crate) fn as_device_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// ホスト → デバイス。長さは確保した要素数と一致している必要がある。
    pub(crate) fn upload(&self, src: &[T]) -> Result<(), RocmError> {
        if src.len() != self.elems {
            return Err(RocmError::InvalidOpArgs {
                op: "typed_buffer.upload",
                detail: format!(
                    "長さ不一致: バッファ {} 要素, 入力 {}",
                    self.elems,
                    src.len()
                ),
            });
        }
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: (U5) `ptr` は `new` が `elems*SIZE` バイト以上確保済み（バイト数は checked_mul）、
        // `src` は `elems*SIZE` バイト読める（`T` は封印された POD。長さ一致は上で検査済み）。
        unsafe {
            self.rt
                .memcpy_htod(self.ptr, src.as_ptr().cast(), src.len() * T::SIZE)
        }
    }

    /// デバイス → ホスト。長さは確保した要素数と一致している必要がある。
    pub(crate) fn download(&self, dst: &mut [T]) -> Result<(), RocmError> {
        if dst.len() != self.elems {
            return Err(RocmError::InvalidOpArgs {
                op: "typed_buffer.download",
                detail: format!(
                    "長さ不一致: バッファ {} 要素, 出力 {}",
                    self.elems,
                    dst.len()
                ),
            });
        }
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: (U5) `dst` は `elems*SIZE` バイト書ける（長さ一致検査済み）、`ptr` は
        // `elems*SIZE` バイト以上確保済み。呼び出し側は読み戻し前に `synchronize` 済み。
        unsafe {
            self.rt
                .memcpy_dtoh(dst.as_mut_ptr().cast(), self.ptr, dst.len() * T::SIZE)
        }
    }

    /// 全要素を新しい `Vec` へ読み戻す。
    pub(crate) fn to_vec(&self) -> Result<Vec<T>, RocmError> {
        let mut v = vec![T::default(); self.elems];
        self.download(&mut v)?;
        Ok(v)
    }
}

impl<T: DeviceElem> Drop for TypedDeviceBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: (U6) `ptr` は `malloc` が返した未解放のポインタで、所有権により二重解放しない。
        unsafe {
            let _ = self.rt.free(self.ptr);
        }
    }
}

/// `elems * T::SIZE` を overflow なしで求める。
pub(crate) fn checked_bytes<T: DeviceElem>(elems: usize) -> Result<usize, RocmError> {
    elems
        .checked_mul(T::SIZE)
        .ok_or_else(|| RocmError::SizeLimitExceeded {
            op: "typed_buffer",
            n: elems,
            limit: usize::MAX / T::SIZE,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_sizes() {
        assert_eq!(<f32 as DeviceElem>::SIZE, std::mem::size_of::<f32>());
        assert_eq!(<i32 as DeviceElem>::SIZE, std::mem::size_of::<i32>());
        assert_eq!(<u32 as DeviceElem>::SIZE, std::mem::size_of::<u32>());
        assert_eq!(<u64 as DeviceElem>::SIZE, std::mem::size_of::<u64>());
        assert_eq!(<f64 as DeviceElem>::SIZE, std::mem::size_of::<f64>());
    }

    #[test]
    fn checked_bytes_rejects_overflow() {
        assert_eq!(checked_bytes::<u64>(3).unwrap(), 24);
        assert_eq!(checked_bytes::<f64>(5).unwrap(), 40);
        assert!(checked_bytes::<f64>(usize::MAX / 8 + 1).is_err());
        assert!(matches!(
            checked_bytes::<u64>(usize::MAX / 8 + 1),
            Err(RocmError::SizeLimitExceeded { .. })
        ));
        assert!(checked_bytes::<f32>(usize::MAX).is_err());
    }
}
