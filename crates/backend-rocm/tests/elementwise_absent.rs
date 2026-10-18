//! ROCm 非搭載環境（CI）での契約: elementwise 3 実行型（`RocmElementwise`・`RocmScalarOp`・
//! `RocmFusedElementwise`）の `new` が panic せず型付きエラーを返し、`RocmBackendOps`（Hip engine）が
//! 「HIP に触れる前に確定できる結果」を返すこと（イシュー #3433）。
//! AMD 実機でも壊れないよう、成功・失敗のどちらでも許容する箇所は環境で分岐する。

use fandhe_ai_backend_rocm::{
    GfxTarget, RocmBackendOps, RocmElementwise, RocmError, RocmFusedElementwise, RocmScalarOp,
    is_available,
};
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, DType, FusedOpKind, FusionPlan, ScalarBinaryOp, ScalarUnaryOp, Tensor,
};

fn assert_typed<T>(r: Result<T, RocmError>) {
    if let Err(e) = r {
        assert!(!e.to_string().is_empty());
    }
}

#[test]
fn constructors_do_not_panic_and_errors_are_typed() {
    assert_typed(RocmElementwise::new(GfxTarget::Gfx942));
    assert_typed(RocmScalarOp::new(GfxTarget::Gfx942));
    assert_typed(RocmFusedElementwise::new(GfxTarget::Gfx942));
}

#[test]
fn unsupported_is_decided_before_touching_hip() {
    let ops = RocmBackendOps::new(GfxTarget::Gfx942);
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    // 対象外の scalar kind は、ROCm の有無によらず `Unsupported`（`DeviceUnavailable` ではない）。
    assert!(matches!(
        ops.scalar_unary(ScalarUnaryOp::Floor, &a),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        ops.scalar_binary(ScalarBinaryOp::Atan2, &a, &a),
        Err(BackendError::Unsupported(_))
    ));
    // allowlist 外の融合プラン（Rsqrt）も同様。
    let plan = FusionPlan::from_ops(
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Rsqrt { input: 0 },
        ],
        vec![4],
        DType::F32,
        1,
    )
    .unwrap();
    assert!(matches!(
        ops.run_fused(&plan, &[&a]),
        Err(BackendError::Unsupported(_))
    ));
}

#[test]
fn supported_kernels_report_unavailable_without_rocm_and_empty_is_ok() {
    let ops = RocmBackendOps::new(GfxTarget::Gfx942);
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    if !is_available() {
        assert!(matches!(
            ops.add(&a, &a),
            Err(BackendError::DeviceUnavailable(_))
        ));
        assert!(matches!(
            ops.scalar_unary(ScalarUnaryOp::Sqrt, &a),
            Err(BackendError::DeviceUnavailable(_))
        ));
    }
    // 空は HIP に触れず Ok。
    let e = Tensor::<f32>::zeros(&[0]).unwrap();
    assert_eq!(ops.add(&e, &e).unwrap().shape(), &[0]);
    assert_eq!(
        ops.scalar_unary(ScalarUnaryOp::Sqrt, &e).unwrap().shape(),
        &[0]
    );
}
