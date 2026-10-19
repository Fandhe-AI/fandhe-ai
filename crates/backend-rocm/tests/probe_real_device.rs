//! ROCm 実機でのみ実行する probe テスト（`#[ignore]`。実行は #3159）。

use fandhe_ai_backend_rocm::is_available;

#[test]
#[ignore = "ROCm 実機が必要（#3159 で実行）"]
fn probe_true_on_rocm_host() {
    assert!(is_available());
}
