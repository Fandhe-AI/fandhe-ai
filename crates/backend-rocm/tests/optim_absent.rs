//! ROCm 非搭載環境（CI）での契約: `RocmBackendOps`（Hip engine）の `sgd_step_device`／
//! `adam_step_device` が、CPU 由来のバッファを panic せず `DeviceMismatch` で拒否すること
//! （イシュー #3440）。AMD 実機でも壊れないよう、環境によらず成立する検査だけを置く。
//!
//! 限界: crate の外からは ROCm の `DeviceBuffer` を作れない（`memory_ops()` が `None`）。このため
//! 「Hip engine で numel > 0 なら `DeviceUnavailable`」や、検証の細かい順序（shape・momentum と
//! velocity・未知 kind）は外から確かめられない。それらは `ops/optim.rs` の内部テストが担う。
//! ここでは約束しない。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_rocm::{GfxTarget, RocmBackendOps};
use fandhe_ai_tensor_core::{
    AdamStepConfig, AdamStepKind, BackendError, BackendOps, SgdStepConfig, Tensor,
};

#[test]
fn cpu_buffers_are_rejected_with_device_mismatch() {
    let rocm = RocmBackendOps::new(GfxTarget::Gfx942);
    let cpu = CpuBackendOps::new();
    let mem = cpu.memory_ops().expect("CpuBackendOps は MemoryOps を持つ");
    let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    let mut p = mem.upload(&t).unwrap();
    let g = mem.upload(&t).unwrap();
    let mut m = mem.upload(&t).unwrap();
    let mut v = mem.upload(&t).unwrap();

    let sgd = SgdStepConfig {
        lr: 0.1,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        is_first_step: true,
    };
    assert!(matches!(
        rocm.sgd_step_device(&mut p, &g, None, &sgd),
        Err(BackendError::DeviceMismatch)
    ));

    let adam = AdamStepConfig {
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        decay_factor: 1.0,
        step_size: 1e-3,
        bias_correction2_sqrt: 0.1,
        kind: AdamStepKind::Coupled,
    };
    assert!(matches!(
        rocm.adam_step_device(&mut p, &g, &mut m, &mut v, &adam),
        Err(BackendError::DeviceMismatch)
    ));

    // 拒否されたバッファは書き換わっていない。
    let after = mem.download(&p).unwrap();
    assert_eq!(after.as_slice().unwrap(), t.as_slice().unwrap());
}
