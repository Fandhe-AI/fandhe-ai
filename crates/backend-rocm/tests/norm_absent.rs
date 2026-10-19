//! ROCm 非搭載環境（CI）での契約: 正規化の実行型 `RocmNorm::new` が panic せず型付きエラーを返し、
//! 搭載環境では不正入力（eps・長さ不一致）を起動前に `InvalidOpArgs` で拒否すること
//! （イシュー #3437・R6）。AMD 実機で壊れないよう、成功・失敗のどちらでも許容する
//! （失敗の種類は環境依存。`index_ops_absent.rs` と同形）。

use fandhe_ai_backend_rocm::{GfxTarget, RocmError, RocmNorm};

#[test]
fn constructor_does_not_panic_and_error_is_typed() {
    if let Err(e) = RocmNorm::new(GfxTarget::Gfx942) {
        assert!(!e.to_string().is_empty());
    }
}

#[test]
fn invalid_inputs_are_rejected_before_launch_when_available() {
    let Ok(n) = RocmNorm::new(GfxTarget::Gfx942) else {
        return;
    };
    let x = [1.0f32; 8];
    let invalid =
        |r: Result<(), RocmError>| assert!(matches!(r, Err(RocmError::InvalidOpArgs { .. })));
    for eps in [f32::NAN, -1.0, f32::INFINITY] {
        invalid(n.run_rmsnorm_f32(&x, None, eps, 2, 4).map(|_| ()));
        invalid(n.run_layer_norm_f32(&x, None, None, eps, 2, 4).map(|_| ()));
    }
    // x の長さ・weight／bias／dy／統計の長さ不一致。
    invalid(n.run_rmsnorm_f32(&x[..7], None, 1e-5, 2, 4).map(|_| ()));
    invalid(
        n.run_rmsnorm_f32(&x, Some(&[1.0; 3]), 1e-5, 2, 4)
            .map(|_| ()),
    );
    invalid(
        n.run_layer_norm_f32(&x, None, Some(&[1.0; 5]), 1e-5, 2, 4)
            .map(|_| ()),
    );
    invalid(
        n.run_rmsnorm_backward_f32(&x, None, &x[..7], 1e-5, 2, 4)
            .map(|_| ()),
    );
    invalid(
        n.run_layer_norm_backward_f32(&x, None, true, &x[..7], 1e-5, 2, 4)
            .map(|_| ()),
    );
    invalid(
        n.run_batch_norm_train_f32(&x, Some(&[1.0; 3]), None, 1e-5, 2, 2, 2)
            .map(|_| ()),
    );
    invalid(
        n.run_batch_norm_infer_f32(&x, &[0.0; 2], &[1.0; 3], None, None, 1e-5, 2, 2, 2)
            .map(|_| ()),
    );
    // 0 要素は HIP に触れず成功する。
    assert!(n.run_rmsnorm_f32(&[], None, 1e-5, 0, 4).unwrap().is_empty());
}
