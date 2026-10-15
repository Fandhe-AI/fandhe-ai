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
    // イシュー #3077 で Softmax は対象になったため、未対応のまま残る
    // `LogSumExp` で fail-closed 契約を維持する。
    let loss = x.logsumexp(Some(1)).unwrap().sum(None).unwrap();
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

/// §17.2 推奨の子テープ構築（`tape_for(parent.device())`）が facade だけで
/// 完結し、ミラー取得と 2 階微分が閉形式（`6x`）と一致すること（イシュー
/// #2546。利用例テスト）。
#[test]
fn child_built_with_tape_for_parent_device_works() {
    let x0 = [0.5_f32, -1.0, 2.0];
    let parent = fandhe_ai::tape();
    let child = fandhe_ai::tape_for(parent.device()).expect("CPU テープは常に構築可能");
    let x = parent.var(&t(x0.to_vec(), &[3]));
    let loss = cubic_loss(&x);
    let cg = parent.backward_create_graph(&loss, &child).unwrap();
    let g = cg.grad(&x).unwrap().expect("子テープ上の勾配");
    // 構築した子テープ自身で 2 階微分する（x^3 の Hessian は対角のため行和＝対角）。
    let grads2 = child
        .backward(&g.sum(None).unwrap())
        .expect("child backward");
    let xc = cg.child_var(&x).unwrap().expect("ミラー取得");
    let second = vals(grads2.get(&xc).unwrap().expect("second order"));
    for (h, &v) in second.iter().zip(x0.iter()) {
        assert!(req2_close(*h as f64, 6.0 * v as f64), "2階 {h}");
    }
}

/// facade 到達可能な `Var::checkpoint_from` で親へ checkpoint 区間を登録
/// すると `backward_create_graph` が fail-closed で拒否し、子を変更しない
/// こと（イシュー #2546）。
#[test]
fn rejects_parent_with_registered_checkpoint_via_facade() {
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let h = x.sigmoid();
    let hc = h.checkpoint_from(&[&x]).expect("checkpoint 区間の登録");
    let loss = hc.sum(None).unwrap();
    assert!(matches!(
        tape.backward_create_graph(&loss, &child),
        Err(AutodiffError::Backward(_))
    ));
    assert_eq!(child.leaf_count(), 0, "拒否時に child を変更しない");
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

/// イシュー #3077: 実 CPU バックエンド（最終軸カーネル経路）での
/// Softmax／LogSoftmax の create_graph。1 階は `backward` と bit 一致し、
/// Hessian の行和は `backward` の中央差分と REQ-2 判定で一致する。
const SW: [f32; 6] = [0.3, -1.2, 0.7, 2.0, -0.5, 1.1];
const SX: [f32; 6] = [0.2, -0.4, 0.9, 1.3, -0.7, 0.1];

fn softmax_family_loss<'t>(tape: &'t Tape, x: &Var<'t>, log: bool) -> Var<'t> {
    let w = tape.var_no_grad(&t(SW.to_vec(), &[2, 3]));
    let y = if log {
        x.log_softmax(1).unwrap()
    } else {
        x.softmax(1).unwrap()
    };
    y.mul(&w).unwrap().sum(None).unwrap()
}

fn first_order_grad(x0: &[f32], log: bool) -> Vec<f32> {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(x0.to_vec(), &[2, 3]));
    let loss = softmax_family_loss(&tape, &x, log);
    let grads = tape.backward(&loss).unwrap();
    vals(grads.get(&x).unwrap().expect("grad"))
}

fn check_softmax_family(log: bool) {
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(SX.to_vec(), &[2, 3]));
    let loss = softmax_family_loss(&tape, &x, log);
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("Softmax／LogSoftmax は create_graph 対象");
    let plain = first_order_grad(&SX, log);
    let g = cg.grad(&x).unwrap().expect("grad");
    let gv = vals(&g.value());
    for (a, b) in plain.iter().zip(gv.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "1 階は backward と bit 一致");
    }
    // 2 階: sum(g) の x 勾配（= Hessian の列和）を、1 階勾配の総和の
    // 中央差分と突合する。
    let grads2 = child
        .backward(&g.sum(None).unwrap())
        .expect("child backward");
    let xc = cg.child_var(&x).unwrap().expect("mirror");
    let h = vals(grads2.get(&xc).unwrap().expect("second order"));
    let eps = 1e-2_f32;
    for i in 0..6 {
        let mut xp = SX;
        let mut xm = SX;
        xp[i] += eps;
        xm[i] -= eps;
        let sp: f64 = first_order_grad(&xp, log).iter().map(|&v| v as f64).sum();
        let sm: f64 = first_order_grad(&xm, log).iter().map(|&v| v as f64).sum();
        let num = (sp - sm) / (2.0 * eps as f64);
        assert!(
            req2_close(h[i] as f64, num),
            "i={i} analytic={} numeric={num}",
            h[i]
        );
    }
}

#[test]
fn softmax_create_graph_matches_first_order_and_finite_difference_via_facade() {
    check_softmax_family(false);
}

#[test]
fn log_softmax_create_graph_matches_first_order_and_finite_difference_via_facade() {
    check_softmax_family(true);
}
// --- 損失 Op 第 1 弾（イシュー #3082・決定記録 §26）。実 CPU バックエンド
//     （融合 `mse_loss_backward`／`nll_loss_backward` 経路を持つ `fandhe_ai::tape()`）
//     で、1 階との REQ-2 一致と Hessian 列和の中央差分突合を行う。

const LOSS_X: [f32; 6] = [0.3, -2.0, 1.9, 0.2, -0.2, 3.0];

fn mse_target(tape: &Tape) -> Var<'_> {
    tape.var_no_grad(&t(vec![0.0, 0.5, 0.4, 0.0, 0.0, 1.0], &[2, 3]))
}

fn mse_loss_at<'t>(tape: &'t Tape, x: &Var<'t>) -> Var<'t> {
    x.tanh()
        .mse_loss_with(&mse_target(tape), fandhe_ai_autodiff::Reduction::Mean)
        .expect("mse_loss")
}

fn nll_loss_at<'t>(x: &Var<'t>) -> Var<'t> {
    let targets = Tensor::new(vec![2i32, 0], &[2]).expect("targets");
    x.log_softmax(1)
        .expect("log_softmax")
        .nll_loss(&targets, 1, fandhe_ai_autodiff::Reduction::Mean)
        .expect("nll_loss")
}

fn grad_sum_at(x0: &[f32], nll: bool) -> f64 {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(x0.to_vec(), &[2, 3]));
    let loss = if nll {
        nll_loss_at(&x)
    } else {
        mse_loss_at(&tape, &x)
    };
    let grads = tape.backward(&loss).expect("backward");
    vals(grads.get(&x).unwrap().expect("grad"))
        .iter()
        .map(|&v| v as f64)
        .sum()
}

fn check_loss(nll: bool) {
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(LOSS_X.to_vec(), &[2, 3]));
    let loss = if nll {
        nll_loss_at(&x)
    } else {
        mse_loss_at(&tape, &x)
    };
    let plain = {
        let grads = tape.backward(&loss).expect("backward");
        vals(grads.get(&x).unwrap().expect("grad"))
    };
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("MseLoss／NllLoss は create_graph 対象");
    let g = cg.grad(&x).unwrap().expect("grad");
    for (a, b) in plain.iter().zip(vals(&g.value()).iter()) {
        assert!(req2_close(*a as f64, *b as f64), "1 階: {a} vs {b}");
    }
    let grads2 = child
        .backward(&g.sum(None).unwrap())
        .expect("child backward");
    let xc = cg.child_var(&x).unwrap().expect("mirror");
    let h = vals(grads2.get(&xc).unwrap().expect("second order"));
    let eps = 1e-2_f32;
    for i in 0..6 {
        let mut xp = LOSS_X;
        let mut xm = LOSS_X;
        xp[i] += eps;
        xm[i] -= eps;
        let num = (grad_sum_at(&xp, nll) - grad_sum_at(&xm, nll)) / (2.0 * eps as f64);
        assert!(
            req2_close(h[i] as f64, num),
            "i={i} analytic={} numeric={num}",
            h[i]
        );
    }
}

#[test]
fn mse_loss_create_graph_matches_first_order_and_finite_difference_via_facade() {
    check_loss(false);
}

#[test]
fn nll_loss_create_graph_matches_first_order_and_finite_difference_via_facade() {
    check_loss(true);
}
