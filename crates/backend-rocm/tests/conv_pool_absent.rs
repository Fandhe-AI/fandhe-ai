//! ROCm 非搭載環境（CI）での契約: R7 の 3 実行型（`RocmIm2col`・`RocmConstantPad`・`RocmPooling`）の
//! `new` が panic せず型付きエラーを返し、搭載環境では不正入力を起動前に拒否すること（イシュー #3438）。
//! AMD 実機で壊れないよう、成功・失敗のどちらでも許容する（失敗の種類は環境依存）。

use fandhe_ai_backend_rocm::{GfxTarget, RocmConstantPad, RocmError, RocmIm2col, RocmPooling};
use fandhe_ai_tensor_core::Conv2dParams;

fn assert_typed<T>(r: Result<T, RocmError>) {
    if let Err(e) = r {
        assert!(!e.to_string().is_empty());
    }
}

#[test]
fn constructors_do_not_panic_and_errors_are_typed() {
    assert_typed(RocmIm2col::new(GfxTarget::Gfx942));
    assert_typed(RocmConstantPad::new(GfxTarget::Gfx942));
    assert_typed(RocmPooling::new(GfxTarget::Gfx942));
}

#[test]
fn invalid_inputs_are_rejected_before_launch_when_available() {
    if let Ok(k) = RocmIm2col::new(GfxTarget::Gfx942) {
        let params = Conv2dParams::new([3, 3], [1, 1], [0, 0], [1, 1], 1).unwrap();
        // col shape が params と整合しない（P 軸が違う）。
        assert!(matches!(
            k.run_im2col_f32(&[0.0; 25], &[1, 1, 5, 5], &[1, 1, 9, 8], &params),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        // 入力長の不一致。
        assert!(matches!(
            k.run_im2col_f32(&[0.0; 24], &[1, 1, 5, 5], &[1, 1, 9, 9], &params),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }
    if let Ok(k) = RocmConstantPad::new(GfxTarget::Gfx942) {
        // pads の長さが rank と違う。
        assert!(matches!(
            k.run_pad_f32(&[0.0; 4], &[2, 2], &[(1, 1)], 0.0),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }
    if let Ok(k) = RocmPooling::new(GfxTarget::Gfx942) {
        // padding > kernel / 2。
        assert!(matches!(
            k.run_max_pool2d_f32(&[0.0; 16], [1, 1, 4, 4], [2, 2], [1, 1], [2, 0], [1, 1]),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        // adaptive の出力サイズ 0。
        assert!(matches!(
            k.run_adaptive_avg_pool2d_f32(&[0.0; 16], [1, 1, 4, 4], [0, 2]),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        // 入力長の不一致。
        assert!(matches!(
            k.run_avg_pool2d_f32(&[0.0; 15], [1, 1, 4, 4], [2, 2], [2, 2], [0, 0], true),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }
}
