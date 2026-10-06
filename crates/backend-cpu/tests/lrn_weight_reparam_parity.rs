//! `CpuBackendOps::{lrn_forward, weight_norm_forward, spectral_norm_forward}`（イシュー #2646）の
//! 直接呼び出しテスト。
//!
//! CPU 実装は共有ホストカーネル（`fandhe_ai_tensor_core::{lrn, weight_reparam}`。窓・軸の規則と
//! 数値契約の正）を呼ぶだけの薄い配線のため、ここでは共有カーネルとの bit 一致・非連続入力・
//! 型付きエラー・run-to-run 決定性を固定する。PyTorch fixture との突合は
//! `crates/autodiff/tests/{lrn,weight_reparam}_parity.rs`、`Tape` 経由の CPU 対 naive 比較は
//! `crates/facade/tests/lrn_weight_reparam_backend_parity.rs` が担当する。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::lrn::{self, LrnParams};
use fandhe_ai_tensor_core::weight_reparam as wr;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn ramp(n: usize, m: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * m) % 11) as f32 * 0.37 - 1.5).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn lrn_matches_shared_kernel_bit_for_bit_including_strided_input() {
    let ops = CpuBackendOps::new();
    let p = LrnParams::new(3, 1e-2, 0.75, 1.5).unwrap();
    let base = t(ramp(2 * 3 * 4, 5), &[2, 3, 4]);
    let layout = lrn::lrn_layout(&[2, 3, 4]).unwrap();
    let want = lrn::local_response_norm_host(&base.host_slice(), &layout, &p).unwrap();
    let got = ops.lrn_forward(&base, &p).unwrap();
    assert_eq!(got.shape(), &[2, 3, 4]);
    assert_eq!(bits(&got.host_slice()), bits(&want));
    // 非連続（軸入れ替え）入力は論理順で処理する。
    let tr = base.transpose(1, 2).unwrap();
    let tr_logical = tr.contiguous();
    let layout_t = lrn::lrn_layout(&[2, 4, 3]).unwrap();
    let want_t = lrn::local_response_norm_host(&tr_logical.host_slice(), &layout_t, &p).unwrap();
    assert_eq!(
        bits(&ops.lrn_forward(&tr, &p).unwrap().host_slice()),
        bits(&want_t)
    );
}

#[test]
fn weight_norm_matches_shared_kernel_and_rejects_bad_g() {
    let ops = CpuBackendOps::new();
    let v = t(ramp(3 * 4, 7), &[3, 4]);
    let g = t(vec![0.5, 1.0, 2.0], &[3, 1]);
    let layout = wr::weight_norm_layout(&[3, 4], &[3, 1], Some(0)).unwrap();
    let want = wr::weight_norm_host(&v.host_slice(), &g.host_slice(), &layout).unwrap();
    let got = ops.weight_norm_forward(&v, &g, Some(0)).unwrap();
    assert_eq!(bits(&got.host_slice()), bits(&want));
    let bad = t(vec![1.0; 3], &[3]);
    assert!(matches!(
        ops.weight_norm_forward(&v, &bad, Some(0)),
        Err(BackendError::ShapeMismatch(_))
    ));
}

#[test]
fn spectral_norm_matches_shared_kernel_and_rejects_bad_state() {
    let ops = CpuBackendOps::new();
    let w = t(ramp(3 * 2, 3), &[3, 2]);
    let u = t(vec![0.6, 0.0, 0.8], &[3]);
    let v = t(vec![1.0, 0.0], &[2]);
    let layout = wr::spectral_norm_layout(&[3, 2], 0).unwrap();
    let want =
        wr::spectral_norm_host(&w.host_slice(), &layout, &u.host_slice(), &v.host_slice()).unwrap();
    let got = ops.spectral_norm_forward(&w, &u, &v, 0).unwrap();
    assert_eq!(bits(&got.host_slice()), bits(&want));
    assert!(matches!(
        ops.spectral_norm_forward(&w, &v, &v, 0),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        ops.spectral_norm_forward(&w, &u, &v, 2),
        Err(BackendError::ShapeMismatch(_))
    ));
}

#[test]
fn invalid_shapes_are_typed_errors() {
    let ops = CpuBackendOps::new();
    let p = LrnParams::new(2, 1.0, 0.75, 1.0).unwrap();
    let rank2 = t(vec![0.0; 4], &[2, 2]);
    assert!(matches!(
        ops.lrn_forward(&rank2, &p),
        Err(BackendError::ShapeMismatch(_))
    ));
    let v = t(vec![1.0; 4], &[2, 2]);
    let g = t(vec![1.0; 2], &[2, 1]);
    assert!(matches!(
        ops.weight_norm_forward(&v, &g, Some(5)),
        Err(BackendError::ShapeMismatch(_))
    ));
}

#[test]
fn results_are_deterministic_run_to_run() {
    let ops = CpuBackendOps::new();
    let p = LrnParams::new(5, 1e-4, 0.75, 2.0).unwrap();
    let x = t(ramp(2 * 7 * 5, 13), &[2, 7, 5]);
    let a = ops.lrn_forward(&x, &p).unwrap();
    let b = ops.lrn_forward(&x, &p).unwrap();
    assert_eq!(bits(&a.host_slice()), bits(&b.host_slice()));
}
