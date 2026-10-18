//! ROCm 非搭載環境（CI）での契約: softmax 系 2 実行型（`RocmSoftmax`・`RocmLogSoftmaxBackward`）の
//! `new` が panic せず型付きエラーを返し、搭載環境では不正入力を起動前に拒否すること。あわせて
//! `RocmBackendOps` の非最終軸が `Unsupported`、形状不一致が `ShapeMismatch` になること
//! （イシュー #3435。`index_ops_absent.rs` と同型）。AMD 実機で壊れないよう、`new` の成否は問わない。

use fandhe_ai_backend_rocm::{
    GfxTarget, RocmBackendOps, RocmError, RocmLogSoftmaxBackward, RocmSoftmax,
};
use fandhe_ai_tensor_core::{BackendError, BackendOps, Tensor};

fn assert_typed<T>(r: Result<T, RocmError>) {
    if let Err(e) = r {
        assert!(!e.to_string().is_empty());
    }
}

#[test]
fn constructors_do_not_panic_and_errors_are_typed() {
    assert_typed(RocmSoftmax::new(GfxTarget::Gfx942));
    assert_typed(RocmLogSoftmaxBackward::new(GfxTarget::Gfx942));
}

#[test]
fn invalid_inputs_are_rejected_before_launch_when_available() {
    if let Ok(s) = RocmSoftmax::new(GfxTarget::Gfx942) {
        assert!(matches!(
            s.run_softmax_f32(&[0.0; 5], 2, 3),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        assert!(matches!(
            s.run_log_softmax_f32(&[0.0; 5], 2, 3),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        assert_eq!(s.run_softmax_f32(&[], 0, 3).unwrap(), Vec::<f32>::new());
    }
    if let Ok(b) = RocmLogSoftmaxBackward::new(GfxTarget::Gfx942) {
        assert!(matches!(
            b.run_log_softmax_backward_f32(&[0.0; 6], &[0.0; 5], 2, 3),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }
}

#[test]
fn backend_ops_contract_without_touching_hip() {
    let o = RocmBackendOps::new(GfxTarget::Gfx942);
    let x = Tensor::new(vec![0.5f32; 12], &[3, 4]).unwrap();
    let y = Tensor::new(vec![0.5f32; 10], &[2, 5]).unwrap();
    // 非最終軸・軸範囲外・形状不一致は HIP を触る前に型付きで返る。
    assert!(matches!(
        o.softmax(&x, 0),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        o.log_softmax(&x, 0),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        o.log_softmax_backward(&x, &x, 0),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        o.softmax(&x, 2),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        o.log_softmax_backward(&x, &y, 1),
        Err(BackendError::ShapeMismatch(_))
    ));
}
