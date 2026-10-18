//! ROCm 非搭載環境（CI）での契約: R8 の 6 実行型（scan・gather/scatter・interpolate・sort・unique）の
//! `new` が panic せず型付きエラーを返し、搭載環境では不正入力を起動前に拒否すること（イシュー #3439）。
//! AMD 実機で壊れないよう、成功・失敗のどちらでも許容する（失敗の種類は環境依存）。

use fandhe_ai_backend_rocm::{
    GfxTarget, RocmError, RocmGatherScatter, RocmInterpolate, RocmScan, RocmSort, RocmUnique,
    ScatterMode,
};

fn assert_typed<T>(r: Result<T, RocmError>) {
    if let Err(e) = r {
        assert!(!e.to_string().is_empty());
    }
}

#[test]
fn constructors_do_not_panic_and_errors_are_typed() {
    assert_typed(RocmScan::new(GfxTarget::Gfx942));
    assert_typed(RocmGatherScatter::new(GfxTarget::Gfx942));
    assert_typed(RocmInterpolate::new(GfxTarget::Gfx942));
    assert_typed(RocmSort::new(GfxTarget::Gfx942));
    assert_typed(RocmUnique::new(GfxTarget::Gfx942));
}

#[test]
fn invalid_inputs_are_rejected_before_launch_when_available() {
    if let Ok(s) = RocmScan::new(GfxTarget::Gfx942) {
        assert!(matches!(
            s.run_cumsum_f32(&[0.0; 5], 1, 2, 3),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }
    if let Ok(g) = RocmGatherScatter::new(GfxTarget::Gfx942) {
        // index が範囲外。
        assert!(matches!(
            g.run_gather_f32(&[0.0; 3], &[3], &[3], &[1], 0),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        assert!(matches!(
            g.run_scatter_f32(&[0.0; 3], &[3], &[-1], &[1], &[1.0], 0, ScatterMode::Add),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        assert!(matches!(
            g.run_one_hot_f32(&[2], 2),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }
    if let Ok(i) = RocmInterpolate::new(GfxTarget::Gfx942) {
        assert!(matches!(
            i.run_bilinear_f32(&[0.0; 4], &[4], &[8], false),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }
    if let Ok(s) = RocmSort::new(GfxTarget::Gfx942) {
        assert!(matches!(
            s.run_sort_f32(&[0.0; 4], &[4], 1, false, 4),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }
}
