//! ROCm 非搭載環境（CI）での契約（イシュー #3444）: 線形代数の入口が panic せず型付きエラーを返し、
//! ゲートの既定が OFF であること。AMD 実機で壊れないよう、成功・失敗のどちらでも許容する
//! （失敗の種類は環境依存）。`BackendOps` への結線（`Unsupported` の返却）は R0 #3431 のマージ後に
//! 本ファイルへ足す。ゲートは `static` のため書き換えるテストは `GATE_LOCK` で直列化し RAII で OFF へ戻す。

use std::sync::Mutex;

use fandhe_ai_backend_rocm::linalg::{gpu_linalg_enabled, set_gpu_linalg_enabled};
use fandhe_ai_backend_rocm::{
    GfxTarget, LuSolveOutcome, RocmError, RocmLinalgFactor, RocmLinalgLu,
};

static GATE_LOCK: Mutex<()> = Mutex::new(());

struct GateGuard<'a> {
    _lock: std::sync::MutexGuard<'a, ()>,
}

impl<'a> GateGuard<'a> {
    fn new() -> Self {
        Self {
            _lock: GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner()),
        }
    }
}

impl Drop for GateGuard<'_> {
    fn drop(&mut self) {
        set_gpu_linalg_enabled(false);
    }
}

#[test]
fn gate_defaults_off_and_toggles() {
    let _g = GateGuard::new();
    assert!(!gpu_linalg_enabled());
    set_gpu_linalg_enabled(true);
    assert!(gpu_linalg_enabled());
    set_gpu_linalg_enabled(false);
    assert!(!gpu_linalg_enabled());
}

#[test]
fn constructors_do_not_panic_and_errors_are_typed() {
    match RocmLinalgLu::new(GfxTarget::Gfx942) {
        Ok(_) => {}
        Err(e) => assert!(!e.to_string().is_empty()),
    }
    match RocmLinalgFactor::new(GfxTarget::Gfx942) {
        Ok(_) => {}
        Err(e) => assert!(!e.to_string().is_empty()),
    }
}

#[test]
fn invalid_shapes_are_rejected_when_available() {
    if let Ok(lu) = RocmLinalgLu::new(GfxTarget::Gfx942) {
        assert!(matches!(
            lu.run_lu_solve_f32(&[0.0; 3], &[0.0; 2], 2, 1),
            Err(RocmError::InvalidLinalgDims { .. })
        ));
        assert!(matches!(
            lu.run_lu_det_f32(&[0.0; 3], 2),
            Err(RocmError::InvalidLinalgDims { .. })
        ));
        // n == 0 はデバイスに触れず CPU と同じ形・値。
        assert_eq!(lu.run_lu_det_f32(&[], 0).unwrap(), 1.0);
        assert_eq!(
            lu.run_lu_inv_f32(&[], 0).unwrap(),
            LuSolveOutcome::Solved(vec![])
        );
    }
    if let Ok(f) = RocmLinalgFactor::new(GfxTarget::Gfx942) {
        assert!(matches!(
            f.run_cholesky_f32(&[0.0; 3], 2),
            Err(RocmError::InvalidLinalgDims { .. })
        ));
        assert!(matches!(
            f.run_qr_f32(&[0.0; 5], 2, 3),
            Err(RocmError::InvalidLinalgDims { .. })
        ));
        assert!(matches!(
            f.run_eigh_f32(&[0.0; 3], 2),
            Err(RocmError::InvalidLinalgDims { .. })
        ));
        // 空入力は呼び出し側が短絡する契約（型付きエラー）。
        assert!(f.run_cholesky_f32(&[], 0).is_err());
        assert!(f.run_qr_f32(&[], 0, 3).is_err());
        assert!(f.run_eigh_f32(&[], 0).is_err());
    }
}
