//! ROCm 非搭載環境（CI）での probe 契約: `is_available` が panic しないこと。
//! AMD 実機で壊れないよう戻り値は検証しない。

use fandhe_ai_backend_rocm::{RocmError, is_available};

#[test]
fn probe_does_not_panic() {
    let _ = is_available();
}

#[test]
fn error_display_is_non_empty() {
    let e = RocmError::Hip { call: "x", code: 1 };
    assert!(!e.to_string().is_empty());
}
