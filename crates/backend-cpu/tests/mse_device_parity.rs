//! `CpuBackendOps::mse_loss_device`／`mse_loss_backward_device`（および `_into_tracked`）の
//! 受け入れ基準対応テスト（イシュー #3253。決定記録 `docs/compat-resident-train-step-decision.md` §13）。
//!
//! 比較先は同じ CPU のホスト版 `mse_loss`／`mse_loss_backward`（Tape 経路が通る核）。同じ縮約核・
//! 同じ最終式を共有するため `to_bits()` 完全一致（NaN はクラス一致）で比較する。

use fandhe_ai_backend_cpu::{CpuBackendOps, CpuMemory};
use fandhe_ai_tensor_core::buffer::{DeviceBuffer, MemoryOps};
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, DispatchFailureCell, MseReduction, ShapeError, Tensor,
};

fn data(seed: u64, len: usize) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        })
        .collect()
}

fn same_bits_or_nan(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn dev(mem: &CpuMemory, v: Vec<f32>, shape: &[usize]) -> DeviceBuffer<f32> {
    mem.upload(&Tensor::new(v, shape).unwrap()).unwrap()
}

fn read(mem: &CpuMemory, b: &DeviceBuffer<f32>) -> Vec<f32> {
    mem.download(b)
        .unwrap()
        .contiguous()
        .as_slice()
        .unwrap()
        .to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn forward_matches_host_across_sizes_and_reductions() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    for &n in &[0usize, 1, 7, 4095, 4096, 4097, 20_000] {
        for red in [MseReduction::Mean, MseReduction::Sum] {
            let (p, t) = (data(1, n), data(2, n));
            let host = ops
                .mse_loss(
                    &Tensor::new(p.clone(), &[n]).unwrap(),
                    &Tensor::new(t.clone(), &[n]).unwrap(),
                    red,
                )
                .unwrap();
            let host_v = host.get(&[]).unwrap();
            let (pd, td) = (dev(&mem, p, &[n]), dev(&mem, t, &[n]));
            let out = ops.mse_loss_device(&pd, &td, red).unwrap();
            assert_eq!(out.shape(), &[] as &[usize]);
            let v = read(&mem, &out)[0];
            assert!(
                same_bits_or_nan(v, host_v),
                "n={n} {red:?}: {v} vs {host_v}"
            );
        }
    }
}

#[test]
fn forward_and_backward_propagate_nan_and_inf_like_host() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let mut p = data(3, 100);
    let t = data(4, 100);
    p[5] = f32::NAN;
    p[9] = f32::INFINITY;
    let host = ops
        .mse_loss(
            &Tensor::new(p.clone(), &[100]).unwrap(),
            &Tensor::new(t.clone(), &[100]).unwrap(),
            MseReduction::Mean,
        )
        .unwrap()
        .get(&[])
        .unwrap();
    let (pd, td) = (dev(&mem, p.clone(), &[100]), dev(&mem, t.clone(), &[100]));
    let v = read(
        &mem,
        &ops.mse_loss_device(&pd, &td, MseReduction::Mean).unwrap(),
    )[0];
    assert!(same_bits_or_nan(v, host));
    let hb = ops
        .mse_loss_backward(
            &Tensor::new(p, &[100]).unwrap(),
            &Tensor::new(t, &[100]).unwrap(),
            0.01,
        )
        .unwrap();
    let hb = hb.contiguous().as_slice().unwrap().to_vec();
    let db = read(&mem, &ops.mse_loss_backward_device(&pd, &td, 0.01).unwrap());
    for (i, (a, b)) in db.iter().zip(&hb).enumerate() {
        assert!(same_bits_or_nan(*a, *b), "idx {i}");
    }
}

#[test]
fn backward_matches_host_and_into_overwrites_dirty_out() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let token = DispatchFailureCell::new();
    for &(r, c) in &[(1usize, 1usize), (3, 5), (64, 10), (65, 129)] {
        let n = r * c;
        let (p, t) = (data(5, n), data(6, n));
        let host = ops
            .mse_loss_backward(
                &Tensor::new(p.clone(), &[r, c]).unwrap(),
                &Tensor::new(t.clone(), &[r, c]).unwrap(),
                0.125,
            )
            .unwrap();
        let host = bits(host.contiguous().as_slice().unwrap());
        let (pd, td) = (dev(&mem, p, &[r, c]), dev(&mem, t, &[r, c]));
        let alloc = ops.mse_loss_backward_device(&pd, &td, 0.125).unwrap();
        assert_eq!(alloc.shape(), &[r, c]);
        assert_eq!(bits(&read(&mem, &alloc)), host);
        let mut out = dev(&mem, vec![f32::NAN; n], &[r, c]);
        ops.mse_loss_backward_device_into_tracked(&pd, &td, 0.125, &mut out, &token)
            .unwrap();
        assert_eq!(bits(&read(&mem, &out)), host);
    }
}

#[test]
fn forward_into_overwrites_dirty_scalar() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let token = DispatchFailureCell::new();
    let (p, t) = (data(7, 300), data(8, 300));
    let (pd, td) = (dev(&mem, p, &[300]), dev(&mem, t, &[300]));
    let expected = read(
        &mem,
        &ops.mse_loss_device(&pd, &td, MseReduction::Mean).unwrap(),
    )[0];
    let mut out = dev(&mem, vec![12345.0], &[]);
    ops.mse_loss_device_into_tracked(&pd, &td, MseReduction::Mean, &mut out, &token)
        .unwrap();
    assert_eq!(read(&mem, &out)[0].to_bits(), expected.to_bits());
}

#[test]
fn typed_errors_leave_out_untouched() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let token = DispatchFailureCell::new();
    let (pd, td) = (dev(&mem, data(1, 6), &[6]), dev(&mem, data(2, 6), &[6]));
    let bad = dev(&mem, data(2, 4), &[4]);
    assert!(matches!(
        ops.mse_loss_device(&pd, &bad, MseReduction::Mean),
        Err(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch { .. }
        ))
    ));
    assert!(matches!(
        ops.mse_loss_backward_device(&pd, &bad, 1.0),
        Err(BackendError::ShapeMismatch(_))
    ));
    // out の shape 不一致（scalar を期待）。
    let mut wrong = dev(&mem, vec![9.0, 9.0], &[2]);
    assert!(matches!(
        ops.mse_loss_device_into_tracked(&pd, &td, MseReduction::Mean, &mut wrong, &token),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert_eq!(read(&mem, &wrong), vec![9.0, 9.0]);
    let mut wrong_bwd = dev(&mem, vec![9.0; 3], &[3]);
    assert!(
        ops.mse_loss_backward_device_into_tracked(&pd, &td, 1.0, &mut wrong_bwd, &token)
            .is_err()
    );
    assert_eq!(read(&mem, &wrong_bwd), vec![9.0; 3]);
    // 空入力の _into は確保経路へ任せる Unsupported。
    let (e1, e2) = (dev(&mem, vec![], &[0]), dev(&mem, vec![], &[0]));
    let mut s = dev(&mem, vec![1.0], &[]);
    assert!(matches!(
        ops.mse_loss_device_into_tracked(&e1, &e2, MseReduction::Mean, &mut s, &token),
        Err(BackendError::Unsupported(_))
    ));
    assert_eq!(read(&mem, &s), vec![1.0]);
}
