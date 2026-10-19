//! ROCm 非搭載環境（CI）での契約: GEMM 派生 4 メソッド（`gemm_resident_rhs`・`gemm_resident_lhs`・
//! `gemm_fp32_strict_into`・`linear_forward_device`）と実行型 `RocmGemmBiasAct` が、panic せず型付き
//! エラーを返し、HIP に触れる前に確定できる結果（`DeviceMismatch`）を返すこと（イシュー #3442・R11）。
//! AMD 実機でも壊れないよう、成功・失敗のどちらでも許容する箇所は環境で分岐する。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_rocm::{GfxTarget, RocmBackendOps, RocmError, RocmGemmBiasAct};
use fandhe_ai_tensor_core::{
    Activation, BackendError, BackendOps, DeviceBuffer, DeviceBufferView, Tensor,
};

fn cpu_buffer(shape: &[usize]) -> DeviceBuffer<f32> {
    let n: usize = shape.iter().product();
    let t = Tensor::new(vec![0.5f32; n], shape).unwrap();
    CpuBackendOps.memory_ops().unwrap().upload(&t).unwrap()
}

#[test]
fn constructor_does_not_panic_and_error_is_typed() {
    let r: Result<RocmGemmBiasAct, RocmError> = RocmGemmBiasAct::new(GfxTarget::Gfx942);
    if let Err(e) = r {
        assert!(!e.to_string().is_empty());
    }
}

#[test]
fn cpu_buffers_are_device_mismatch_before_touching_hip() {
    let ops = RocmBackendOps::new(GfxTarget::Gfx942);
    let w_buf = cpu_buffer(&[3, 4]);
    let ws = [3, 4];
    let a = Tensor::new(vec![1.0f32; 6], &[2, 3]).unwrap();
    // ROCm の有無によらず、CPU 由来のバッファは DeviceMismatch。
    assert!(matches!(
        ops.gemm_resident_rhs(&a, DeviceBufferView::new(&w_buf, 0, &ws).unwrap(), None),
        Err(BackendError::DeviceMismatch)
    ));
    let b = Tensor::new(vec![1.0f32; 8], &[4, 2]).unwrap();
    assert!(matches!(
        ops.gemm_resident_lhs(DeviceBufferView::new(&w_buf, 0, &ws).unwrap(), &b),
        Err(BackendError::DeviceMismatch)
    ));
    let a_buf = cpu_buffer(&[2, 3]);
    assert!(matches!(
        ops.linear_forward_device(
            &a_buf,
            DeviceBufferView::new(&w_buf, 0, &ws).unwrap(),
            None,
            Activation::None
        ),
        Err(BackendError::DeviceMismatch)
    ));
    let mut out = cpu_buffer(&[4]);
    let (x, y) = (
        Tensor::new(vec![1.0f32; 4], &[2, 2]).unwrap(),
        Tensor::new(vec![1.0f32; 4], &[2, 2]).unwrap(),
    );
    assert!(matches!(
        ops.gemm_fp32_strict_into(&x, &y, &mut out, 0),
        Err(BackendError::DeviceMismatch)
    ));
}

#[test]
fn shape_errors_are_decided_without_hip() {
    let ops = RocmBackendOps::new(GfxTarget::Gfx942);
    // 形状不一致は、ROCm の有無によらず ShapeMismatch（GEMM を起動する前に確定する）。
    let mut out = cpu_buffer(&[4]);
    let (x, y) = (
        Tensor::new(vec![1.0f32; 6], &[2, 3]).unwrap(),
        Tensor::new(vec![1.0f32; 4], &[2, 2]).unwrap(),
    );
    assert!(matches!(
        ops.gemm_fp32_strict_into(&x, &y, &mut out, 0),
        Err(BackendError::ShapeMismatch(_))
    ));
}
