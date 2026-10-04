//! facade 公開の高階微分 `Tape::backward_create_graph`／`CreateGraphResult`
//! （イシュー #2545。公開形は `docs/autodiff-higher-order-grad-decision.md`
//! §17.2、承認根拠はルート #2499 の一括承認）の公開経路テスト。
//!
//! 内部クレートを一切 import せず facade の公開 API だけで Hessian・HVP を
//! 組む（利用者視点の到達性確認）。計算ロジックの網羅検証は
//! `crates/autodiff/tests/create_graph.rs` が担い、本ファイルは facade 経由
//! でも同じ契約（1 階 bit 同一・fail-closed 拒否・エラー variant）が
//! 保たれることを固定する。
//!
//! 判定方式: 数値比較は REQ-2 統一複合判定（相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満。`req2_close`）。tolerance は新設・緩和しない。
//! 有限差分は `create_graph` を経由しない独立経路（素の `Tape::backward`
//! の中央差分）として使う。
//!
//! GPU 実機テストは `#[ignore]`（CI 非実行）。手順は
//! `docs/perf/logs/create-graph-facade-2545/README.md`。

use fandhe_ai::{AutodiffError, CreateGraphResult, Device, Tape, Tensor, Var};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::<f32>::new(data, shape).expect("test fixture: tensor")
}

/// REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。
fn req2_close(a: f64, b: f64) -> bool {
    let abs = (a - b).abs();
    abs < 1e-5 || abs / a.abs().max(b.abs()).max(f64::MIN_POSITIVE) < 1e-3
}

fn vals(t: &Tensor<f32>) -> Vec<f32> {
    t.as_slice().expect("CPU contiguous").to_vec()
}

/// `loss = sum(x^3)` を親テープ上に組む。
fn cubic_loss<'t>(x: &Var<'t>) -> Var<'t> {
    x.mul(x).unwrap().mul(x).unwrap().sum(None).unwrap()
}

fn hessian_diag_cubic(tape: &Tape, child: &Tape, x0: &[f32]) -> Vec<f32> {
    let x = tape.var(&t(x0.to_vec(), &[x0.len()]));
    let loss = cubic_loss(&x);
    let cg = tape
        .backward_create_graph(&loss, child)
        .expect("create_graph");
    let g = cg.grad(&x).unwrap().expect("grad");
    // sum_j dg_j/dx_i = H の行和（x^3 の Hessian は対角のため対角成分）
    let grads2 = child
        .backward(&g.sum(None).unwrap())
        .expect("child backward");
    let xc = cg.child_var(&x).unwrap().expect("mirror");
    vals(grads2.get(&xc).unwrap().expect("second order"))
}

#[test]
fn hessian_cubic_matches_closed_form_via_facade() {
    let x0 = [0.5_f32, -1.0, 2.0];
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(x0.to_vec(), &[3]));
    let loss = cubic_loss(&x);
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let first = vals(cg.first_order().get(&x).unwrap().unwrap());
    for (g, &v) in first.iter().zip(x0.iter()) {
        assert!(req2_close(*g as f64, 3.0 * (v as f64).powi(2)), "1階 {g}");
    }
    let second = hessian_diag_cubic(&fandhe_ai::tape(), &fandhe_ai::tape(), &x0);
    for (h, &v) in second.iter().zip(x0.iter()) {
        assert!(req2_close(*h as f64, 6.0 * v as f64), "2階 {h}");
    }
}

/// `loss(W) = sum(tanh(X W))` の 1 階勾配（素の `Tape::backward`。HVP の
/// 中央差分に使う独立経路）。
fn first_order_grad_w(xm: &Tensor<f32>, w: &[f32]) -> Vec<f32> {
    let tape = fandhe_ai::tape();
    let wv = tape.var(&t(w.to_vec(), &[2, 2]));
    let xv = tape.var_no_grad(xm);
    let loss = xv.matmul(&wv).unwrap().tanh().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    vals(grads.get(&wv).unwrap().unwrap())
}

#[test]
fn hvp_tanh_matmul_matches_finite_difference_via_facade() {
    let xm = t(vec![0.3, -0.7, 1.1, 0.4], &[2, 2]);
    let w0 = [0.2_f32, -0.5, 0.8, 0.1];
    let v = [0.6_f32, -0.2, 0.3, 0.9];

    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let wv = tape.var(&t(w0.to_vec(), &[2, 2]));
    let xv = tape.var_no_grad(&xm);
    let loss = xv.matmul(&wv).unwrap().tanh().sum(None).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let g = cg.grad(&wv).unwrap().unwrap();
    let vc = child.var_no_grad(&t(v.to_vec(), &[2, 2]));
    let s = g.mul(&vc).unwrap().sum(None).unwrap();
    let grads2 = child.backward(&s).unwrap();
    let hv = vals(
        grads2
            .get(&cg.child_var(&wv).unwrap().unwrap())
            .unwrap()
            .unwrap(),
    );

    let eps = 1.0e-2_f32; // f32 の丸め雑音を抑えるため先例（f64 差分）より大きく取る
    let plus: Vec<f32> = w0.iter().zip(v).map(|(&a, d)| a + eps * d).collect();
    let minus: Vec<f32> = w0.iter().zip(v).map(|(&a, d)| a - eps * d).collect();
    let gp = first_order_grad_w(&xm, &plus);
    let gm = first_order_grad_w(&xm, &minus);
    for i in 0..4 {
        let fd = (gp[i] as f64 - gm[i] as f64) / (2.0 * eps as f64);
        assert!(req2_close(hv[i] as f64, fd), "Hv[{i}] {} vs fd {fd}", hv[i]);
    }
}

#[test]
fn first_order_is_bit_identical_to_plain_backward() {
    let x0 = vec![0.5_f32, -1.0, 2.0];
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(x0.clone(), &[3]));
    let cg = tape.backward_create_graph(&cubic_loss(&x), &child).unwrap();
    let a = vals(cg.first_order().get(&x).unwrap().unwrap());

    let plain = fandhe_ai::tape();
    let xp = plain.var(&t(x0, &[3]));
    let grads = plain.backward(&cubic_loss(&xp)).unwrap();
    let b = vals(grads.get(&xp).unwrap().unwrap());
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a), bits(&b));
}

#[test]
fn rejects_same_tape_as_child() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0], &[1]));
    let loss = cubic_loss(&x);
    assert!(matches!(
        tape.backward_create_graph(&loss, &tape),
        Err(AutodiffError::Backward(_))
    ));
}

#[test]
fn rejects_non_empty_child_and_leaves_it_untouched() {
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let _leaf = child.var(&t(vec![1.0], &[1]));
    let x = tape.var(&t(vec![1.0], &[1]));
    let loss = cubic_loss(&x);
    assert!(matches!(
        tape.backward_create_graph(&loss, &child),
        Err(AutodiffError::Backward(_))
    ));
    assert_eq!(child.leaf_count(), 1, "拒否時に child を変更しない");
}

#[test]
fn rejects_unsupported_op_and_keeps_child_empty() {
    let child = fandhe_ai::tape();
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![0.1, 0.2, 0.3], &[1, 3]));
    let loss = x.softmax(1).unwrap().sum(None).unwrap();
    assert!(matches!(
        tape.backward_create_graph(&loss, &child),
        Err(AutodiffError::Backward(_))
    ));
    assert_eq!(child.leaf_count(), 0, "未対応 Op 拒否後も child は空");
    // 同じ child を、対応 Op のグラフでそのまま再利用できる。
    let tape2 = fandhe_ai::tape();
    let x2 = tape2.var(&t(vec![1.0], &[1]));
    assert!(
        tape2
            .backward_create_graph(&cubic_loss(&x2), &child)
            .is_ok()
    );
}

#[test]
fn grad_on_no_grad_leaf_returns_gradient_tracking_disabled() {
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(vec![2.0], &[1]));
    let c = tape.var_no_grad(&t(vec![3.0], &[1]));
    let loss = x.mul(&x).unwrap().mul(&c).unwrap().sum(None).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    assert!(matches!(
        cg.grad(&c),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    assert!(cg.grad(&x).unwrap().is_some());
}

#[test]
fn grad_with_foreign_tape_var_returns_tape_mismatch() {
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(vec![2.0], &[1]));
    let cg = tape.backward_create_graph(&cubic_loss(&x), &child).unwrap();
    let other = fandhe_ai::tape();
    let y = other.var(&t(vec![2.0], &[1]));
    assert!(matches!(cg.grad(&y), Err(AutodiffError::TapeMismatch)));
    assert!(matches!(cg.child_var(&y), Err(AutodiffError::TapeMismatch)));
}

/// 公開型 `CreateGraphResult` が facade ルートから名指しできること。
#[test]
fn create_graph_result_is_nameable() {
    fn first<'c>(cg: &CreateGraphResult<'c>, x: &Var<'_>) -> bool {
        cg.first_order().get(x).is_ok()
    }
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0], &[1]));
    let cg = tape.backward_create_graph(&cubic_loss(&x), &child).unwrap();
    assert!(first(&cg, &x));
}

fn gpu_hessian_matches_cpu(device: Device) {
    let x0 = [0.5_f32, -1.0, 2.0];
    let cpu = hessian_diag_cubic(&fandhe_ai::tape(), &fandhe_ai::tape(), &x0);
    let parent = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテスト");
    let child = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテスト");
    let gpu = hessian_diag_cubic(&parent, &child, &x0);
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        assert!(req2_close(*a as f64, *b as f64), "cpu {a} vs gpu {b}");
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要"]
fn create_graph_hessian_cuda_matches_cpu() {
    gpu_hessian_matches_cpu(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要"]
fn create_graph_hessian_metal_matches_cpu() {
    gpu_hessian_matches_cpu(Device::Metal);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要"]
fn create_graph_rejects_device_mismatch() {
    let parent = fandhe_ai::tape();
    let child = fandhe_ai::tape_for(Device::Cuda(0)).expect("実機が利用可能な前提のテスト");
    let x = parent.var(&t(vec![1.0], &[1]));
    assert!(matches!(
        parent.backward_create_graph(&cubic_loss(&x), &child),
        Err(AutodiffError::DeviceMismatch { .. })
    ));
}
