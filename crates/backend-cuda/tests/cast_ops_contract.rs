//! `CastOps` の CUDA accessor 契約テスト（イシュー #1751・親 #1613）。
//!
//! `crate::ops::CudaBackendOps::cast_ops` accessor 自体は driver に
//! 一切触れない（`typed_ops_f64_contract.rs` と同じ理由: `CudaBackendOps::
//! new` はコンストラクタのみで driver へ触れないため、accessor が
//! `Some` を返すことの確認は CUDA 非搭載環境でも常時 CI 実行可能）。
//! 実際に GPU へディスパッチする数値一致検証は `cast_parity.rs`
//! （環境適応スモーク＋`#[ignore]` 実機テスト）を参照。

use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::{BackendOps, CastOps};

/// `cast_ops()` accessor が `Some` を返し、`&dyn BackendOps` 経由でも
/// 到達できることを確認する（`typed_ops_bf16_contract.rs` 等の同型
/// テストと同じ意図）。
#[test]
fn cast_ops_accessor_is_some() {
    let ops = CudaBackendOps::new(0);
    let backend_ops: &dyn BackendOps = &ops;
    assert!(backend_ops.cast_ops().is_some());
}

/// `&dyn CastOps` として扱えること（object-safety のコンパイル時検査。
/// `tensor_core::cast::tests::assert_dyn_compatible` と同型）。
#[test]
fn cast_ops_is_dyn_compatible_via_accessor() {
    let ops = CudaBackendOps::new(0);
    let cast_ops: &dyn CastOps = BackendOps::cast_ops(&ops).expect("cast_ops must be Some");
    // `&dyn CastOps` として保持できること自体が検証（呼び出しは行わない
    // ——driver に触れる呼び出しは `cast_parity.rs` の責務）。
    let _ = cast_ops;
}
