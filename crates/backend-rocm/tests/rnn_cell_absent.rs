//! ROCm 非搭載環境（CI）での契約: `RocmRnnCell::new` が panic せず型付きエラーを返し、搭載環境では
//! 長さ不一致・hidden=0・端数 numel を起動前に拒否し、空入力は HIP を起動しないこと
//! （イシュー #3441・R10）。AMD 実機で壊れないよう、成功・失敗のどちらでも許容する。

use fandhe_ai_backend_rocm::{GfxTarget, RocmError, RocmRnnCell};

#[test]
fn constructor_does_not_panic_and_error_is_typed() {
    if let Err(e) = RocmRnnCell::new(GfxTarget::Gfx942) {
        assert!(!e.to_string().is_empty());
    }
}

#[test]
fn invalid_inputs_are_rejected_before_launch_when_available() {
    let Ok(r) = RocmRnnCell::new(GfxTarget::Gfx942) else {
        return;
    };
    // 長さ不一致（pre は numel * 4 が必要）。
    assert!(matches!(
        r.run_lstm_pointwise_f32(&[0.0; 7], &[0.0; 2], 1),
        Err(RocmError::InvalidOpArgs { .. })
    ));
    // hidden = 0。
    assert!(matches!(
        r.run_gru_pointwise_f32(&[0.0; 3], &[0.0; 3], &[0.0; 1], 0),
        Err(RocmError::InvalidOpArgs { .. })
    ));
    // numel が hidden の倍数でない。
    assert!(matches!(
        r.run_gru_backward_f32(&[0.0; 9], &[0.0; 3], &[0.0; 3], &[0.0; 3], 2),
        Err(RocmError::InvalidOpArgs { .. })
    ));
    assert!(matches!(
        r.run_lstm_hidden_backward_f32(&[0.0; 2], &[0.0; 3], &[0.0; 2]),
        Err(RocmError::InvalidOpArgs { .. })
    ));
    assert!(matches!(
        r.run_lstm_cell_backward_f32(&[0.0; 5], &[0.0; 2], &[0.0; 2], 1),
        Err(RocmError::InvalidOpArgs { .. })
    ));
    // 空は HIP を起動せず成功する。
    let (g, c, h) = r.run_lstm_pointwise_f32(&[], &[], 4).unwrap();
    assert!(g.is_empty() && c.is_empty() && h.is_empty());
    assert!(
        r.run_lstm_hidden_backward_f32(&[], &[], &[])
            .unwrap()
            .0
            .is_empty()
    );
}
