//! ROCm 非搭載環境（CI）での契約: `RocmLoss::new` が panic せず型付きエラーを返し、搭載環境では
//! 長さ不一致・nll の範囲外添字・未知の kind を起動前に拒否すること（イシュー #3436・R5）。
//! AMD 実機で壊れないよう、成功・失敗のどちらでも許容する（失敗の種類は環境依存）。

use fandhe_ai_backend_rocm::{GfxTarget, RocmError, RocmLoss};

#[test]
fn constructor_does_not_panic_and_error_is_typed() {
    if let Err(e) = RocmLoss::new(GfxTarget::Gfx942) {
        assert!(!e.to_string().is_empty());
    }
}

#[test]
fn invalid_inputs_are_rejected_before_launch_when_available() {
    let Ok(l) = RocmLoss::new(GfxTarget::Gfx942) else {
        return;
    };
    assert!(matches!(
        l.run_mse_loss_f32(&[0.0; 3], &[0.0; 2], 1.0),
        Err(RocmError::InvalidOpArgs { .. })
    ));
    assert!(matches!(
        l.run_huber_loss_f32(&[0.0; 2], &[0.0; 2], 2, 1.0, 1.0),
        Err(RocmError::InvalidOpArgs { .. })
    ));
    assert!(matches!(
        l.run_bce_backward_f32(&[0.0; 2], &[0.0; 2], -1, 1.0),
        Err(RocmError::InvalidOpArgs { .. })
    ));
    // 範囲外の添字（C = 3 に対し t = 3）。
    assert!(matches!(
        l.run_nll_loss_f32(&[0.0; 6], &[3, 0], 2, 3, 1, 1.0),
        Err(RocmError::InvalidOpArgs { .. })
    ));
    assert!(matches!(
        l.run_nll_backward_f32(&[-1, 0], 2, 3, 1, 1.0),
        Err(RocmError::InvalidOpArgs { .. })
    ));
    // 空は HIP を起動せず成功する。
    assert_eq!(l.run_mse_loss_f32(&[], &[], 1.0).unwrap(), 0.0);
    assert!(
        l.run_kl_div_backward_f32(&[], &[], 0, 1.0)
            .unwrap()
            .is_empty()
    );
}
