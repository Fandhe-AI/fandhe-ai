//! HIPRTC（実行時コンパイラ）の動的ロードとコンパイル経路。
//!
//! 役割: `backend-cuda::nvrtc::compile_ptx` の ROCm 側対応物。`gemm.rs` が
//! [`CompileSource`]（`GemmSource`・`KernelSource` 等の検証済みソース）を渡し、メモリ上のコードオブジェクト [`CodeObject`] を受け取る。
//! ディスクキャッシュは作らない（改竄面を持たない。キャッシュ導入は別 Issue）。
//!
//! 不変条件 U1（hip.rs と同形）: `libhiprtc` のハンドルと解決済み関数ポインタを
//! [`HiprtcRuntime`] に同居させ、フィールドは private。U2: 渡す C 文字列は NUL 終端で呼び出し中
//! 生存し、オプション配列の個数と `numOptions` が一致する。U3: コード取得は `GetCodeSize` の
//! サイズどおりにバッファを確保し、サイズ 0 を拒否する。U7: プログラムは RAII で 1 回だけ破棄する。
//!
//! A03: ソースは封印 trait [`CompileSource`] を実装する検証済みソース型（[`GemmSource`]・[`LinalgSource`]・`KernelSource`）のみ受け付け、オプションは [`GfxTarget`] allowlist 由来の
//! `--gpu-architecture=gfxNNN` 1 本のみ。include パス・環境変数・`-ffp-contract` は渡さない
//! （GEMM の FMA は明示 `__builtin_fmaf`、線形代数は各ソース先頭の `#pragma clang fp contract(off)` で
//! 決まるため不要。docs/backend-rocm-hiprtc-gemm-decision.md §2・docs/backend-gpu-linalg-decision.md §10）。
//!
//! 出典: soname は HIP 公式文書「HIPRTC is available as a separate library `libhiprtc.so`」
//! （<https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html>。版付き soname
//! `libhiprtc.so.N` は未採用）。各関数のシグネチャは同 API リファレンスの C 宣言に基づく。
//! `hiprtcResult` は 4 バイト enum として `c_int` で受け 0 を成功とする（実機確認は #3159）。

use crate::error::RocmError;
use crate::kernel_source::CompileSource;
use crate::target::GfxTarget;
use std::ffi::{CString, c_char, c_void};
use std::os::raw::c_int;

const SYM_CREATE: &str = "hiprtcCreateProgram";
const SYM_COMPILE: &str = "hiprtcCompileProgram";
const SYM_LOG_SIZE: &str = "hiprtcGetProgramLogSize";
const SYM_LOG: &str = "hiprtcGetProgramLog";
const SYM_CODE_SIZE: &str = "hiprtcGetCodeSize";
const SYM_CODE: &str = "hiprtcGetCode";
const SYM_DESTROY: &str = "hiprtcDestroyProgram";

/// 取得するコンパイラログの上限（これを超えるログは取得せずサイズのみ記録する）。
const MAX_LOG_BYTES: usize = 1 << 20;
/// コードオブジェクトのサイズ上限（異常値でメモリを使い切らないため）。
const MAX_CODE_BYTES: usize = 256 << 20;

#[cfg(target_os = "linux")]
pub(crate) const HIPRTC_CANDIDATES: &[&str] = &["libhiprtc.so"];
#[cfg(not(target_os = "linux"))]
pub(crate) const HIPRTC_CANDIDATES: &[&str] = &[];

type CreateFn = unsafe extern "C" fn(
    *mut *mut c_void,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> c_int;
type CompileFn = unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> c_int;
type SizeFn = unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int;
type FetchFn = unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int;
type DestroyFn = unsafe extern "C" fn(*mut *mut c_void) -> c_int;

/// メモリ上のコードオブジェクト（対象 ISA のバイナリ）。
#[derive(Debug)]
pub(crate) struct CodeObject(Vec<u8>);

impl CodeObject {
    pub(crate) fn as_ptr(&self) -> *const c_void {
        self.0.as_ptr().cast()
    }
}

/// ロード済み HIPRTC。ハンドルと関数ポインタの一体所有（U1）。
pub(crate) struct HiprtcRuntime {
    // hip.rs と同じく Drop しても `dlclose` しない（意図的なリーク。F1・監査記録 §5）。
    // `RocmGemmF32::new` はコンパイルのたびにローカルへロードするため、素の `Library` だと
    // 毎回 dlclose する。comgr がスレッド・atexit フックを残す場合のクラッシュを避ける。
    _lib: std::mem::ManuallyDrop<libloading::Library>,
    create: CreateFn,
    compile: CompileFn,
    log_size: SizeFn,
    log: FetchFn,
    code_size: SizeFn,
    code: FetchFn,
    destroy: DestroyFn,
}

/// 固定名のシンボルを解決する（hip.rs の `resolve` と同じ契約）。
///
/// # Safety
/// `T` は `name` の C 宣言と一致する関数ポインタ型で、結果は `lib` と同じ構造に格納すること。
unsafe fn resolve<T: Copy>(lib: &libloading::Library, name: &'static str) -> Result<T, RocmError> {
    // SAFETY: 呼び出し元の契約。名前は固定の `&'static str`。
    match unsafe { lib.get::<T>(name.as_bytes()) } {
        Ok(s) => Ok(*s),
        Err(e) => Err(RocmError::SymbolMissing {
            symbol: name,
            detail: e.to_string(),
        }),
    }
}

impl HiprtcRuntime {
    /// 候補を順に試して `libhiprtc` をロードする。候補が空なら常に `LibraryUnavailable`。
    pub(crate) fn load_from(candidates: &[&'static str]) -> Result<Self, RocmError> {
        let mut last = String::from("候補ライブラリがありません");
        for name in candidates {
            // SAFETY: 固定の `&'static str`（パスなし）。初期化子の実行は ROCm 公式ライブラリのみ
            // 対象として受容する。
            let lib = match unsafe { libloading::Library::new(*name) } {
                Ok(l) => l,
                Err(e) => {
                    last = e.to_string();
                    continue;
                }
            };
            // SAFETY: 各型は HIPRTC の C 宣言（モジュール doc の出典）と一致させた。
            // ポインタは `lib` と同じ構造に格納し外へ出さない。
            let rt = unsafe {
                Self {
                    create: resolve(&lib, SYM_CREATE)?,
                    compile: resolve(&lib, SYM_COMPILE)?,
                    log_size: resolve(&lib, SYM_LOG_SIZE)?,
                    log: resolve(&lib, SYM_LOG)?,
                    code_size: resolve(&lib, SYM_CODE_SIZE)?,
                    code: resolve(&lib, SYM_CODE)?,
                    destroy: resolve(&lib, SYM_DESTROY)?,
                    _lib: std::mem::ManuallyDrop::new(lib),
                }
            };
            return Ok(rt);
        }
        Err(RocmError::LibraryUnavailable {
            tried: candidates.to_vec(),
            detail: last,
        })
    }

    /// 既定の候補でロードする。
    pub(crate) fn load() -> Result<Self, RocmError> {
        Self::load_from(HIPRTC_CANDIDATES)
    }
}

/// `hiprtcProgram` の RAII（U7: Drop で 1 回だけ破棄）。
struct Program<'a> {
    rt: &'a HiprtcRuntime,
    handle: *mut c_void,
}

impl Drop for Program<'_> {
    fn drop(&mut self) {
        // SAFETY: `handle` は `hiprtcCreateProgram` が返した未破棄のハンドルで、所有権により
        // 二重破棄は起きない。`destroy` は `rt`（ライブラリ保持）由来。Drop 内の失敗は
        // panic させず無視する（リソース解放の best effort）。
        unsafe {
            let _ = (self.rt.destroy)(&mut self.handle);
        }
    }
}

fn check(call: &'static str, code: c_int) -> Result<(), RocmError> {
    if code == 0 {
        Ok(())
    } else {
        Err(RocmError::Hiprtc { call, code })
    }
}

fn cstring(s: &str, what: &'static str) -> Result<CString, RocmError> {
    CString::new(s).map_err(|_| RocmError::InvalidCString { what })
}

/// `--gpu-architecture=` オプション文字列（allowlist 由来の固定値のみ）。
pub(crate) fn arch_option(target: GfxTarget) -> String {
    format!("--gpu-architecture={}", target.as_str())
}

/// バッファ（NUL 終端を含みうる）をログ文字列へ変換する。
fn log_from_buffer(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// ソースをコンパイルしてコードオブジェクトを返す。ログは成功時も取得する（警告確認用。
/// 失敗時のみエラーへ含める）。
pub(crate) fn compile_code_object(
    rt: &HiprtcRuntime,
    src: &impl CompileSource,
    target: GfxTarget,
) -> Result<CodeObject, RocmError> {
    let c_src = cstring(src.as_str(), "ソース")?;
    let c_name = cstring(src.program_name(), "プログラム名")?;
    let c_opt = cstring(&arch_option(target), "オプション")?;
    let opts: [*const c_char; 1] = [c_opt.as_ptr()];

    let mut handle: *mut c_void = std::ptr::null_mut();
    // SAFETY: (U2) `c_src`・`c_name` は NUL 終端で本関数の終わりまで生存する。ヘッダは 0 件
    // （numHeaders=0・配列は null）。`handle` は有効な書き込み先。
    let code = unsafe {
        (rt.create)(
            &mut handle,
            c_src.as_ptr(),
            c_name.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    check(SYM_CREATE, code)?;
    if handle.is_null() {
        return Err(RocmError::Hiprtc {
            call: SYM_CREATE,
            code: -1,
        });
    }
    let prog = Program { rt, handle };

    // SAFETY: (U2) `prog.handle` は有効。`opts` の要素数 1 と numOptions=1 が一致し、指す C 文字列
    // （`c_opt`）は呼び出し中生存する。
    let compile_code = unsafe { (rt.compile)(prog.handle, opts.len() as c_int, opts.as_ptr()) };

    let log = fetch_log(&prog)?;
    if compile_code != 0 {
        return Err(RocmError::CompileFailed {
            code: compile_code,
            log,
        });
    }

    let mut size: usize = 0;
    // SAFETY: `prog.handle` は有効、`size` は有効な書き込み先。
    check(SYM_CODE_SIZE, unsafe {
        (rt.code_size)(prog.handle, &mut size)
    })?;
    if size == 0 {
        return Err(RocmError::CodeObjectEmpty);
    }
    if size > MAX_CODE_BYTES {
        return Err(RocmError::Hiprtc {
            call: SYM_CODE_SIZE,
            code: -2,
        });
    }
    let mut buf = vec![0u8; size];
    // SAFETY: (U3) `buf` は `GetCodeSize` が返したサイズちょうどで、`GetCode` はそのサイズまで
    // 書く。`prog.handle` は有効。
    check(SYM_CODE, unsafe {
        (rt.code)(prog.handle, buf.as_mut_ptr().cast())
    })?;
    Ok(CodeObject(buf))
}

/// コンパイラログを上限付きで取得する。上限超過時は取得せずサイズのみ返す。
fn fetch_log(prog: &Program<'_>) -> Result<String, RocmError> {
    let rt = prog.rt;
    let mut size: usize = 0;
    // SAFETY: `prog.handle` は有効、`size` は有効な書き込み先。
    check(SYM_LOG_SIZE, unsafe {
        (rt.log_size)(prog.handle, &mut size)
    })?;
    if size == 0 {
        return Ok(String::new());
    }
    if size > MAX_LOG_BYTES {
        return Ok(format!(
            "（ログが上限 {MAX_LOG_BYTES} バイトを超えたため省略: {size} バイト）"
        ));
    }
    let mut buf = vec![0u8; size];
    // SAFETY: `buf` は `GetProgramLogSize` が返したサイズちょうど（NUL 終端込み）。
    check(SYM_LOG, unsafe {
        (rt.log)(prog.handle, buf.as_mut_ptr().cast())
    })?;
    Ok(log_from_buffer(&buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels_gemm::{GemmF32Tile, render_gemm_f32_source};

    #[test]
    fn nonexistent_library_is_unavailable() {
        assert!(matches!(
            HiprtcRuntime::load_from(&["libfandhe-ai-nonexistent-hiprtc-for-test.so"]),
            Err(RocmError::LibraryUnavailable { .. })
        ));
        assert!(matches!(
            HiprtcRuntime::load_from(&[]),
            Err(RocmError::LibraryUnavailable { .. })
        ));
    }

    #[test]
    fn candidates_have_no_path_separators() {
        for c in HIPRTC_CANDIDATES {
            assert!(!c.contains('/') && !c.contains('\\'), "{c}");
        }
    }

    #[test]
    fn options_are_only_the_arch_literal() {
        assert_eq!(arch_option(GfxTarget::Gfx942), "--gpu-architecture=gfx942");
    }

    #[test]
    fn log_buffer_conversion() {
        assert_eq!(log_from_buffer(b"warn\0\0"), "warn");
        assert_eq!(log_from_buffer(b"abc"), "abc");
        assert_eq!(log_from_buffer(b""), "");
    }

    #[test]
    fn linalg_sources_have_fixed_program_names_and_no_nul() {
        use crate::kernels_linalg::render_linalg_lu_source;
        use crate::kernels_linalg_factor::render_linalg_factor_source;
        let lu = render_linalg_lu_source();
        let fa = render_linalg_factor_source();
        assert_eq!(lu.program_name(), "fandhe_linalg_lu_f64.hip");
        assert_eq!(fa.program_name(), "fandhe_linalg_factor_f64.hip");
        assert!(cstring(lu.as_str(), "ソース").is_ok());
        assert!(cstring(fa.as_str(), "ソース").is_ok());
    }

    #[test]
    fn generated_source_has_no_nul() {
        let s = render_gemm_f32_source(&GemmF32Tile::DEFAULT);
        assert!(cstring(s.as_str(), "ソース").is_ok());
        assert!(cstring("a\0b", "x").is_err());
    }
}
