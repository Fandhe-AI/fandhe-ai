//! ROCm 非搭載環境（CI）での契約: R3 の 2 実行型（`RocmReduce`・`RocmArgReduce`）の `new` が
//! panic せず型付きエラーを返し、搭載環境では不正入力を起動前に拒否すること（イシュー #3434）。
//! AMD 実機で壊れないよう、成功・失敗のどちらでも許容する（失敗の種類は環境依存）。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_rocm::{GfxTarget, RocmArgReduce, RocmBackendOps, RocmError, RocmReduce};
use fandhe_ai_tensor_core::{BackendError, BackendOps, Tensor};

fn assert_typed<T>(r: Result<T, RocmError>) {
    if let Err(e) = r {
        assert!(!e.to_string().is_empty());
    }
}

#[test]
fn constructors_do_not_panic_and_errors_are_typed() {
    assert_typed(RocmReduce::new(GfxTarget::Gfx942));
    assert_typed(RocmArgReduce::new(GfxTarget::Gfx942));
}

#[test]
fn invalid_inputs_are_rejected_before_launch_when_available() {
    if let Ok(r) = RocmReduce::new(GfxTarget::Gfx942) {
        // 長さ不一致。
        assert!(matches!(
            r.run_sum_axis_f32(&[0.0; 5], 1, 2, 3),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        assert!(matches!(
            r.run_max_axis_f32(&[0.0; 5], 1, 2, 3),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        // 空縮約（max/min は単位元なし）。
        assert!(matches!(
            r.run_max_all_f32(&[]),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        assert!(matches!(
            r.run_min_axis_f32(&[], 2, 0, 3),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        // sum の空縮約は 0。
        assert_eq!(r.run_sum_all_f32(&[]).unwrap(), 0.0);
        assert_eq!(r.run_sum_axis_f32(&[], 2, 0, 3).unwrap(), vec![0.0; 6]);
    }
    if let Ok(a) = RocmArgReduce::new(GfxTarget::Gfx942) {
        assert!(matches!(
            a.run_argmax_axis_f32(&[0.0; 5], 1, 2, 3),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        assert!(matches!(
            a.run_argmin_all_f32(&[]),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        assert!(matches!(
            a.run_argmax_axis_f32(&[], 2, 0, 3),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }
}

/// 公開 API（`RocmBackendOps`）経由: 空縮約は HIP に触れず CPU と同じ結果、
/// 範囲外 dim は搭載有無によらず `ShapeMismatch`、非搭載の通常入力は `DeviceUnavailable`。
#[test]
fn backend_ops_reductions_follow_cpu_contract_for_empty_and_bad_dim() {
    let o = RocmBackendOps::new(GfxTarget::Gfx942);
    let c = CpuBackendOps::new();
    let empty = Tensor::<f32>::zeros(&[0]).unwrap();
    assert_eq!(
        format!("{:?}", o.sum(&empty, None)),
        format!("{:?}", c.sum(&empty, None))
    );
    assert_eq!(
        format!("{:?}", o.max(&empty, None)),
        format!("{:?}", c.max(&empty, None))
    );
    assert_eq!(
        format!("{:?}", o.argmax(&empty, None)),
        format!("{:?}", c.argmax(&empty, None))
    );
    let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert!(matches!(
        o.sum(&x, Some(2)),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        o.argmax(&x, Some(9)),
        Err(BackendError::ShapeMismatch(_))
    ));
    if !fandhe_ai_backend_rocm::is_available() {
        assert!(matches!(
            o.sum(&x, None),
            Err(BackendError::DeviceUnavailable(_))
        ));
        assert!(matches!(
            o.min(&x, Some(0)),
            Err(BackendError::DeviceUnavailable(_))
        ));
        assert!(matches!(
            o.argmin(&x, Some(1)),
            Err(BackendError::DeviceUnavailable(_))
        ));
    }
}
