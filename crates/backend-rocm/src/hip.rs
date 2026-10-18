//! HIP ランタイムの動的ロードと呼び出し（`libloading`。第 12 区分）。
//!
//! 役割: `libamdhip64` を実行時に開き、必須シンボルを解決して保持する。`lib.rs` の
//! `is_available`・`gemm.rs`・`module.rs` から呼ばれる。ビルド時リンク（`#[link]`）は使わないため
//! ROCm 非搭載でもビルドが通る（docs/backend-rocm-hip-ffi-decision.md §4）。
//!
//! 不変条件 U1: ライブラリハンドルと解決済み関数ポインタを [`HipRuntime`] に同居させ、
//! フィールドは private・ポインタを返す getter は置かない。ポインタの呼び出しは本構造の
//! メソッド経由に限り、ハンドルの寿命内でのみ起きる。生ポインタを取るメソッドは `unsafe fn`
//! とし、呼び出し側（`module.rs`）の SAFETY コメントで不変条件を示す。ハンドル型は
//! `Arc<HipRuntime>` を持ち、ライブラリがハンドルより先に解放されない。
//!
//! A03: ライブラリ名・シンボル名は固定の `&'static str` のみ。環境変数・外部入力から
//! 組み立てず、パスも連結しない（ローダーの探索規則に任せる）。
//!
//! 出典: Linux の候補は `libamdhip64.so`（ROCm 公式文書のリンク例 `-lamdhip64`。
//! <https://rocm.docs.amd.com/projects/HIP/en/latest/faq.html>）のみ。版付き soname と
//! Windows の DLL 名は一次資料で確認できなかったため採用しない（Windows・macOS は空配列）。
//! 各関数のシグネチャは HIP 公式 API リファレンス
//! （<https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api_reference.html>）
//! の C 宣言に基づく。`hipError_t` は 4 バイト enum として `c_int` で受け、0 を成功とする
//! （実機での ABI 確認は #3159）。

use crate::error::RocmError;
use std::ffi::{c_char, c_uint, c_void};
use std::os::raw::c_int;

const SYM_GET_DEVICE_COUNT: &str = "hipGetDeviceCount";
const SYM_SET_DEVICE: &str = "hipSetDevice";
const SYM_MODULE_LOAD_DATA: &str = "hipModuleLoadData";
const SYM_MODULE_GET_FUNCTION: &str = "hipModuleGetFunction";
const SYM_MODULE_UNLOAD: &str = "hipModuleUnload";
const SYM_MODULE_LAUNCH_KERNEL: &str = "hipModuleLaunchKernel";
const SYM_MALLOC: &str = "hipMalloc";
const SYM_FREE: &str = "hipFree";
const SYM_MEMCPY_HTOD: &str = "hipMemcpyHtoD";
const SYM_MEMCPY_DTOH: &str = "hipMemcpyDtoH";
const SYM_DEVICE_SYNCHRONIZE: &str = "hipDeviceSynchronize";

/// 候補ライブラリ名（Linux）。
#[cfg(target_os = "linux")]
pub(crate) const HIP_RUNTIME_CANDIDATES: &[&str] = &["libamdhip64.so"];
/// 候補ライブラリ名（HIP が存在しない・出典未確定の OS では空）。
#[cfg(not(target_os = "linux"))]
pub(crate) const HIP_RUNTIME_CANDIDATES: &[&str] = &[];

type HipGetDeviceCountFn = unsafe extern "C" fn(*mut c_int) -> c_int;
type HipSetDeviceFn = unsafe extern "C" fn(c_int) -> c_int;
type HipModuleLoadDataFn = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int;
type HipModuleGetFunctionFn =
    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int;
type HipModuleUnloadFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipModuleLaunchKernelFn = unsafe extern "C" fn(
    *mut c_void,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> c_int;
type HipMallocFn = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type HipFreeFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipMemcpyHtoDFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> c_int;
type HipMemcpyDtoHFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> c_int;
type HipDeviceSynchronizeFn = unsafe extern "C" fn() -> c_int;

/// ロード済み HIP ランタイム。ハンドルと関数ポインタの一体所有（U1）。
pub(crate) struct HipRuntime {
    // `_lib` を落とすとポインタが無効になるため、ポインタと同じ構造に保持する。
    _lib: libloading::Library,
    get_device_count: HipGetDeviceCountFn,
    set_device: HipSetDeviceFn,
    module_load_data: HipModuleLoadDataFn,
    module_get_function: HipModuleGetFunctionFn,
    module_unload: HipModuleUnloadFn,
    module_launch_kernel: HipModuleLaunchKernelFn,
    malloc: HipMallocFn,
    free: HipFreeFn,
    memcpy_htod: HipMemcpyHtoDFn,
    memcpy_dtoh: HipMemcpyDtoHFn,
    device_synchronize: HipDeviceSynchronizeFn,
}

/// `hipError_t`（0 = hipSuccess）を型付きエラーへ変換する。
pub(crate) fn check(call: &'static str, code: c_int) -> Result<(), RocmError> {
    if code == 0 {
        Ok(())
    } else {
        Err(RocmError::Hip { call, code })
    }
}

/// 固定名のシンボルを 1 つ解決する。
///
/// # Safety
/// `T` は `name` の C 宣言と一致する `extern "C"` 関数ポインタ型でなければならず、
/// 返したポインタは `lib` が生きている間だけ呼べる（呼び出し側が同じ構造へ格納する）。
unsafe fn resolve<T: Copy>(lib: &libloading::Library, name: &'static str) -> Result<T, RocmError> {
    // SAFETY: 呼び出し元の契約（`T` と宣言の一致）に従う。名前は固定の `&'static str`。
    match unsafe { lib.get::<T>(name.as_bytes()) } {
        Ok(sym) => Ok(*sym),
        Err(e) => Err(RocmError::SymbolMissing {
            symbol: name,
            detail: e.to_string(),
        }),
    }
}

/// 候補を順に試し、最初に開けたライブラリで必須シンボルをすべて解決する。
pub(crate) fn load_from(candidates: &[&'static str]) -> Result<HipRuntime, RocmError> {
    let mut last_detail = String::from("候補ライブラリがありません");
    for name in candidates {
        // SAFETY: `name` は固定の `&'static str` でパスを含まない。ロード時にライブラリの
        // 初期化子が走ることは受容する（ROCm 公式ランタイムのみが対象）。
        let lib = match unsafe { libloading::Library::new(*name) } {
            Ok(lib) => lib,
            Err(e) => {
                last_detail = e.to_string();
                continue;
            }
        };
        // SAFETY: 各関数ポインタ型は HIP の C 宣言（モジュール doc の出典）と一致させた。
        // 得たポインタは `lib` と同じ `HipRuntime` に格納し、構造の外へ出さない。
        let rt = unsafe {
            HipRuntime {
                get_device_count: resolve(&lib, SYM_GET_DEVICE_COUNT)?,
                set_device: resolve(&lib, SYM_SET_DEVICE)?,
                module_load_data: resolve(&lib, SYM_MODULE_LOAD_DATA)?,
                module_get_function: resolve(&lib, SYM_MODULE_GET_FUNCTION)?,
                module_unload: resolve(&lib, SYM_MODULE_UNLOAD)?,
                module_launch_kernel: resolve(&lib, SYM_MODULE_LAUNCH_KERNEL)?,
                malloc: resolve(&lib, SYM_MALLOC)?,
                free: resolve(&lib, SYM_FREE)?,
                memcpy_htod: resolve(&lib, SYM_MEMCPY_HTOD)?,
                memcpy_dtoh: resolve(&lib, SYM_MEMCPY_DTOH)?,
                device_synchronize: resolve(&lib, SYM_DEVICE_SYNCHRONIZE)?,
                _lib: lib,
            }
        };
        return Ok(rt);
    }
    Err(RocmError::LibraryUnavailable {
        tried: candidates.to_vec(),
        detail: last_detail,
    })
}

/// 既定の候補でロードする。
pub(crate) fn load() -> Result<HipRuntime, RocmError> {
    load_from(HIP_RUNTIME_CANDIDATES)
}

impl HipRuntime {
    /// デバイス数を返す。
    pub(crate) fn device_count(&self) -> Result<i32, RocmError> {
        let mut n: c_int = 0;
        // SAFETY: `get_device_count` は `hipError_t hipGetDeviceCount(int*)`。`n` は呼び出し中
        // 有効な書き込み先で、ポインタは `self`（ライブラリ保持）から得た。
        let code = unsafe { (self.get_device_count)(&mut n) };
        check(SYM_GET_DEVICE_COUNT, code)?;
        Ok(n)
    }

    /// 現スレッドのカレントデバイスを設定する。
    pub(crate) fn set_device(&self, id: i32) -> Result<(), RocmError> {
        // SAFETY: `hipError_t hipSetDevice(int)`。値渡しのみでメモリを触らない。
        check(SYM_SET_DEVICE, unsafe { (self.set_device)(id) })
    }

    /// 全ストリームの完了を待つ。
    pub(crate) fn synchronize(&self) -> Result<(), RocmError> {
        // SAFETY: `hipError_t hipDeviceSynchronize(void)`。引数なし。
        check(SYM_DEVICE_SYNCHRONIZE, unsafe {
            (self.device_synchronize)()
        })
    }

    /// デバイスメモリを `bytes` 確保する（0 は呼び出し側で拒否済み）。
    pub(crate) fn malloc(&self, bytes: usize) -> Result<*mut c_void, RocmError> {
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: `hipError_t hipMalloc(void**, size_t)`。`p` は呼び出し中有効な書き込み先。
        let code = unsafe { (self.malloc)(&mut p, bytes) };
        check(SYM_MALLOC, code)?;
        if p.is_null() {
            return Err(RocmError::Hip {
                call: SYM_MALLOC,
                code: -1,
            });
        }
        Ok(p)
    }

    /// デバイスメモリを解放する。
    ///
    /// # Safety
    /// `p` は [`Self::malloc`] が返し、まだ解放していないポインタであること。
    pub(crate) unsafe fn free(&self, p: *mut c_void) -> Result<(), RocmError> {
        // SAFETY: `hipError_t hipFree(void*)`。`p` の有効性は呼び出し元の契約。
        check(SYM_FREE, unsafe { (self.free)(p) })
    }

    /// ホスト → デバイスのコピー。
    ///
    /// # Safety
    /// `dst` は `bytes` 以上のデバイス確保済み領域、`src` は `bytes` バイト読めること。
    pub(crate) unsafe fn memcpy_htod(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    ) -> Result<(), RocmError> {
        // SAFETY: `hipError_t hipMemcpyHtoD(hipDeviceptr_t, void*, size_t)`（`hipDeviceptr_t`
        // は `void*`）。範囲の有効性は呼び出し元の契約。
        check(SYM_MEMCPY_HTOD, unsafe {
            (self.memcpy_htod)(dst, src, bytes)
        })
    }

    /// デバイス → ホストのコピー。
    ///
    /// # Safety
    /// `dst` は `bytes` バイト書けるホスト領域、`src` は `bytes` 以上のデバイス確保済み領域。
    pub(crate) unsafe fn memcpy_dtoh(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    ) -> Result<(), RocmError> {
        // SAFETY: `hipError_t hipMemcpyDtoH(void*, hipDeviceptr_t, size_t)`。範囲は呼び出し元の契約。
        check(SYM_MEMCPY_DTOH, unsafe {
            (self.memcpy_dtoh)(dst, src, bytes)
        })
    }

    /// コードオブジェクトのバイト列から module を作る。
    ///
    /// # Safety
    /// `image` は有効なコードオブジェクトを指し、返った module が解放されるまで生存すること。
    pub(crate) unsafe fn module_load_data(
        &self,
        image: *const c_void,
    ) -> Result<*mut c_void, RocmError> {
        let mut m: *mut c_void = std::ptr::null_mut();
        // SAFETY: `hipError_t hipModuleLoadData(hipModule_t*, const void*)`。`m` は有効な書き込み先、
        // `image` の有効性は呼び出し元の契約。
        let code = unsafe { (self.module_load_data)(&mut m, image) };
        check(SYM_MODULE_LOAD_DATA, code)?;
        if m.is_null() {
            return Err(RocmError::Hip {
                call: SYM_MODULE_LOAD_DATA,
                code: -1,
            });
        }
        Ok(m)
    }

    /// module から関数を取り出す。
    ///
    /// # Safety
    /// `module` は未解放の有効な module。`name` は NUL 終端で呼び出し中生存する。
    pub(crate) unsafe fn module_get_function(
        &self,
        module: *mut c_void,
        name: *const c_char,
    ) -> Result<*mut c_void, RocmError> {
        let mut f: *mut c_void = std::ptr::null_mut();
        // SAFETY: `hipError_t hipModuleGetFunction(hipFunction_t*, hipModule_t, const char*)`。
        // 引数の有効性は呼び出し元の契約。
        let code = unsafe { (self.module_get_function)(&mut f, module, name) };
        check(SYM_MODULE_GET_FUNCTION, code)?;
        if f.is_null() {
            return Err(RocmError::Hip {
                call: SYM_MODULE_GET_FUNCTION,
                code: -1,
            });
        }
        Ok(f)
    }

    /// module を解放する。
    ///
    /// # Safety
    /// `module` は未解放で、実行中のカーネルがない（同期済み）こと。
    pub(crate) unsafe fn module_unload(&self, module: *mut c_void) -> Result<(), RocmError> {
        // SAFETY: `hipError_t hipModuleUnload(hipModule_t)`。有効性は呼び出し元の契約。
        check(SYM_MODULE_UNLOAD, unsafe { (self.module_unload)(module) })
    }

    /// カーネルを既定ストリームで起動する。
    ///
    /// # Safety
    /// `func` は有効な関数。`params` は宣言どおりの個数・型・順序の引数への有効なポインタ配列で
    /// 呼び出し中（非同期起動のため同期完了まで）生存すること。
    pub(crate) unsafe fn launch(
        &self,
        func: *mut c_void,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        params: *mut *mut c_void,
    ) -> Result<(), RocmError> {
        // SAFETY: `hipError_t hipModuleLaunchKernel(hipFunction_t, uint gx, gy, gz, bx, by, bz,
        // uint sharedMemBytes, hipStream_t, void** kernelParams, void** extra)`。stream は null
        // （既定）、共有メモリは静的確保のみのため 0、extra は null。`params` は呼び出し元の契約。
        let code = unsafe {
            (self.module_launch_kernel)(
                func,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                std::ptr::null_mut(),
                params,
                std::ptr::null_mut(),
            )
        };
        check(SYM_MODULE_LAUNCH_KERNEL, code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonexistent_library_is_unavailable() {
        let r = load_from(&["libfandhe-ai-nonexistent-hip-for-test.so"]);
        assert!(matches!(r, Err(RocmError::LibraryUnavailable { .. })));
    }

    #[test]
    fn empty_candidates_is_unavailable() {
        assert!(matches!(
            load_from(&[]),
            Err(RocmError::LibraryUnavailable { .. })
        ));
    }

    #[test]
    fn candidates_have_no_path_separators() {
        for c in HIP_RUNTIME_CANDIDATES {
            assert!(!c.contains('/') && !c.contains('\\'), "{c}");
        }
    }

    #[test]
    fn check_maps_nonzero() {
        assert!(check("x", 0).is_ok());
        assert_eq!(check("x", 7), Err(RocmError::Hip { call: "x", code: 7 }));
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn non_linux_has_no_candidates() {
        assert!(HIP_RUNTIME_CANDIDATES.is_empty());
    }
}
