//! HIP ランタイムの動的ロード（`libloading`。第 12 区分）。
//!
//! 役割: `libamdhip64` を実行時に開き、必須シンボルの解決成否を確認する。`lib.rs` の
//! `is_available` から呼ばれる。ビルド時リンク（`#[link]`）は使わないため ROCm 非搭載でも
//! ビルドが通る（docs/backend-rocm-hip-ffi-decision.md §4）。
//!
//! 不変条件 U1: ライブラリハンドルと解決済み関数ポインタを [`HipRuntime`] に同居させ、
//! フィールドは private・ポインタを返す getter は置かない。ポインタの呼び出しは本構造の
//! メソッド経由に限り、ハンドルの寿命内でのみ起きる（雛形段階では呼び出しメソッドなし）。
//!
//! A03: ライブラリ名・シンボル名は固定の `&'static str` のみ。環境変数・外部入力から
//! 組み立てず、パスも連結しない（ローダーの探索規則に任せる）。
//!
//! 出典と未確定事項: Linux の候補は `libamdhip64.so`（ROCm 公式文書のリンク例
//! `-lamdhip64`。<https://rocm.docs.amd.com/projects/HIP/en/latest/faq.html>）のみ。
//! 版付き soname と Windows の DLL 名は一次資料で確認できなかったため採用しない
//! （Windows・macOS は空配列で probe は常に false）。`hipGetDeviceCount` の
//! `hipError_t` の基底型も公式 API 文書に明記がなく、雛形では呼び出さずシンボル解決のみ行う。

use crate::error::RocmError;
use std::os::raw::c_int;

/// 必須シンボル名（固定）。
const SYM_GET_DEVICE_COUNT: &str = "hipGetDeviceCount";

/// 候補ライブラリ名（Linux）。
#[cfg(target_os = "linux")]
pub(crate) const HIP_RUNTIME_CANDIDATES: &[&str] = &["libamdhip64.so"];
/// 候補ライブラリ名（HIP が存在しない・出典未確定の OS では空）。
#[cfg(not(target_os = "linux"))]
pub(crate) const HIP_RUNTIME_CANDIDATES: &[&str] = &[];

/// `hipError_t hipGetDeviceCount(int*)` の関数ポインタ型（解決の確認のみ。呼び出さない）。
type HipGetDeviceCountFn = unsafe extern "C" fn(*mut c_int) -> c_int;

/// ロード済み HIP ランタイム。ハンドルと関数ポインタの一体所有（U1）。
pub(crate) struct HipRuntime {
    // `_lib` を落とすとポインタが無効になるため、ポインタと同じ構造に保持する。
    _lib: libloading::Library,
    // 雛形では未使用。#3155 で呼び出しメソッドを足す（getter は作らない）。
    _hip_get_device_count: HipGetDeviceCountFn,
}

/// 候補を順に試し、最初に開けたライブラリで必須シンボルを解決する。
pub(crate) fn load_from(candidates: &[&'static str]) -> Result<HipRuntime, RocmError> {
    let mut last_detail = String::from("候補ライブラリがありません");
    for name in candidates {
        // SAFETY: `name` は固定の `&'static str` でパスを含まない。ロード時にライブラリの
        // 初期化子が走ることは受容する（ROCm 公式ランタイムのみが対象）。`Library::new` は
        // 既定で RTLD_LOCAL 相当。
        let lib = match unsafe { libloading::Library::new(*name) } {
            Ok(lib) => lib,
            Err(e) => {
                last_detail = e.to_string();
                continue;
            }
        };
        // SAFETY: `HipGetDeviceCountFn` は HIP の C ABI（`hipError_t hipGetDeviceCount(int*)`。
        // https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html）
        // の宣言と一致させた。ここでは解決のみで呼ばない。得たポインタは `lib` と同じ
        // `HipRuntime` に格納し、構造の外へ出さない。
        let sym = match unsafe { lib.get::<HipGetDeviceCountFn>(SYM_GET_DEVICE_COUNT.as_bytes()) } {
            Ok(sym) => *sym,
            Err(e) => {
                return Err(RocmError::SymbolMissing {
                    symbol: SYM_GET_DEVICE_COUNT,
                    detail: e.to_string(),
                });
            }
        };
        return Ok(HipRuntime {
            _lib: lib,
            _hip_get_device_count: sym,
        });
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

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn non_linux_has_no_candidates() {
        assert!(HIP_RUNTIME_CANDIDATES.is_empty());
    }
}
