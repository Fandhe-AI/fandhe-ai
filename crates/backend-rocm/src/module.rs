//! HIP module・関数・デバイスバッファの所有型（RAII）。
//!
//! 役割: `gemm.rs` が使う GPU リソースの寿命を型で管理する。各型は `Arc<HipRuntime>` を持ち、
//! ライブラリがハンドルより先に解放されない（U1）。`CodeObject` は module が生きている間
//! 保持する（U4）。Drop 内の失敗は panic にせず無視する（解放の best effort。呼び出し側は
//! 実行後に同期済みである契約）。unsafe の監査記録は `docs/backend-rocm-unsafe-audit.md`（#3158）。

use crate::error::RocmError;
use crate::hip::HipRuntime;
use crate::hiprtc::CodeObject;
use std::ffi::{CString, c_void};
use std::marker::PhantomData;
use std::sync::Arc;

/// ロード済み module。Drop で同期のうえ `hipModuleUnload` する（U7）。
pub(crate) struct Module {
    handle: *mut c_void,
    /// module を作成したデバイス ID。Drop 時にこのデバイスを選んでから解放する。
    device: i32,
    // module 解放後にバイト列・ライブラリが解放されるよう、フィールド順は handle → code → rt
    // （`Drop::drop` 本体の後に宣言順で破棄される）。
    _code: CodeObject,
    rt: Arc<HipRuntime>,
}

impl Module {
    /// コードオブジェクトから module を作る。
    pub(crate) fn load(
        rt: Arc<HipRuntime>,
        code: CodeObject,
        device: i32,
    ) -> Result<Self, RocmError> {
        // SAFETY: (U4) `code` は有効なコードオブジェクトで、`Module` が `_code` として保持し
        // module 解放まで生存する。
        let handle = unsafe { rt.module_load_data(code.as_ptr())? };
        Ok(Self {
            handle,
            device,
            _code: code,
            rt,
        })
    }

    /// 固定名の関数を取り出す（`extern "C"` のため名前修飾なし）。
    pub(crate) fn function(&self, name: &'static str) -> Result<Function<'_>, RocmError> {
        let c = CString::new(name).map_err(|_| RocmError::InvalidCString {
            what: "カーネル名"
        })?;
        // SAFETY: `self.handle` は未解放の module。`c` は NUL 終端で呼び出し中生存する。
        let ptr = unsafe { self.rt.module_get_function(self.handle, c.as_ptr())? };
        Ok(Function {
            ptr,
            _module: PhantomData,
        })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // module は作成デバイスに属するため、呼び出し元が別デバイスを選んでいても作成デバイスで
        // 同期・解放する。ガードは Drop で元の選択へ戻す。選択失敗時も解放は試みる。
        let _guard = self.rt.select_device(self.device).ok();
        // 実行中カーネルがない状態でアンロードするため先に同期する（失敗は無視）。
        let _ = self.rt.synchronize();
        // SAFETY: `handle` は `module_load_data` が返した未解放の module で、所有権により
        // 二重解放しない。同期済み。
        unsafe {
            let _ = self.rt.module_unload(self.handle);
        }
    }
}

/// module 内の関数ハンドル（module より長生きできない）。
pub(crate) struct Function<'m> {
    ptr: *mut c_void,
    _module: PhantomData<&'m Module>,
}

impl Function<'_> {
    /// 起動する（非同期。完了は呼び出し側が `synchronize` で待つ）。
    ///
    /// # Safety
    /// `params` は当該カーネルの宣言どおりの個数・型・順序の引数を指す有効なポインタ配列で、
    /// 同期完了まで生存すること。
    pub(crate) unsafe fn launch(
        &self,
        rt: &HipRuntime,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        params: &mut [*mut c_void],
    ) -> Result<(), RocmError> {
        // SAFETY: `self.ptr` は生存中の module の関数。`params` の契約は呼び出し元。
        unsafe { rt.launch(self.ptr, grid, block, params.as_mut_ptr()) }
    }
}

/// デバイスバッファの要素型（crate 内限定の封印 trait。`f32`・`f64`・`i32` のみ）。
///
/// 実装はいずれも任意のビット列が有効な値になる POD 型で、`upload`／`download` の
/// バイト列コピー（`elems * size_of::<T>()`）が安全であることの前提になる（M5〜M7）。
pub(crate) trait DeviceElem: Copy {}
impl DeviceElem for f32 {}
impl DeviceElem for f64 {}
impl DeviceElem for i32 {}

/// 型付きのデバイスバッファ。Drop で 1 回だけ `hipFree`。
///
/// `hipMalloc` はメモリを 0 初期化しない。状態・出力を表すバッファは
/// [`DeviceBuffer::from_slice`] で起動前に必ず内容を書き込む（線形代数。A04/A08）。
pub(crate) struct DeviceBuffer<T: DeviceElem> {
    ptr: *mut c_void,
    elems: usize,
    rt: Arc<HipRuntime>,
    _elem: PhantomData<T>,
}

impl<T: DeviceElem> DeviceBuffer<T> {
    /// `elems` 要素分を確保する（0 要素でも有効なポインタを得るため最小 4 バイト確保）。
    pub(crate) fn new(rt: Arc<HipRuntime>, elems: usize) -> Result<Self, RocmError> {
        let bytes = elems
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| RocmError::InvalidGemmDims {
                detail: "バッファのバイト数がオーバーフローします".into(),
            })?
            .max(4);
        let ptr = rt.malloc(bytes)?;
        Ok(Self {
            ptr,
            elems,
            rt,
            _elem: PhantomData,
        })
    }

    /// 確保して `src` の内容で初期化する（未初期化メモリを起動に渡さないための入口）。
    pub(crate) fn from_slice(rt: Arc<HipRuntime>, src: &[T]) -> Result<Self, RocmError> {
        let buf = Self::new(rt, src.len())?;
        buf.upload(src)?;
        Ok(buf)
    }

    /// カーネル引数に渡すデバイスポインタ。
    pub(crate) fn as_device_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// ホスト → デバイス。長さは確保した要素数と一致している必要がある。
    pub(crate) fn upload(&self, src: &[T]) -> Result<(), RocmError> {
        if src.len() != self.elems {
            return Err(RocmError::InvalidGemmDims {
                detail: "upload の長さがバッファと一致しません".into(),
            });
        }
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: (M5) `ptr` は `elems*size_of::<T>()` バイト以上確保済み、`src` は同バイト数
        // 読める。バイト数は `new` で checked_mul 済みで、`T: DeviceElem` は POD。
        unsafe {
            self.rt
                .memcpy_htod(self.ptr, src.as_ptr().cast(), std::mem::size_of_val(src))
        }
    }

    /// デバイス → ホスト。長さは確保した要素数と一致している必要がある。
    pub(crate) fn download(&self, dst: &mut [T]) -> Result<(), RocmError> {
        if dst.len() != self.elems {
            return Err(RocmError::InvalidGemmDims {
                detail: "download の長さがバッファと一致しません".into(),
            });
        }
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: (M6) `dst` は `elems*size_of::<T>()` バイト書ける、`ptr` は同バイト数以上確保済み。
        unsafe {
            self.rt.memcpy_dtoh(
                dst.as_mut_ptr().cast(),
                self.ptr,
                std::mem::size_of_val(&*dst),
            )
        }
    }
}

impl<T: DeviceElem> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: `ptr` は `malloc` が返した未解放のポインタで、所有権により二重解放しない。
        unsafe {
            let _ = self.rt.free(self.ptr);
        }
    }
}
