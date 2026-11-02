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
//!
//! イシュー #3087（親 #3075。決定記録 `docs/autodiff-higher-order-grad-decision.md` §32）:
//! #3077〜#3086 で `create_graph` 対象へ広げた Op 族の代表（正規化・Conv2d・Gelu・損失・索引累積・
//! Pad／MaskedFill・プーリング・線形代数／FFT）が facade の公開経路でも二階まで通ることを固定する。
//! x について線形な Op は `sum(op(x))` だと Hessian が恒等的にゼロで二階の再生が壊れても通るため、
//! 必ず非線形な包み（tanh／要素積）を入れ、固定方向の `H v` がゼロでないことも assert する。
//! 拒否の基準 Op は恒久対象外の `Tape::custom`（§21.2 A 区分）。`LogSumExp`（#3792）・`Max`（#3789）は
//! 対象化済みのため基準にしない。GPU 実機（LayerNorm・Conv2d・MaxPool2d）は
//! `#[ignore]` で、実測は #3168・#3169 へ申し送る
//! （`docs/perf/logs/create-graph-extended-ops-3087/README.md`）。

use std::sync::Arc;

use fandhe_ai::nn::loss::{
    BceWithLogitsOptions, CrossEntropyOptions, GaussianNllOptions, MultiLabelSoftMarginOptions,
    MultiMarginOptions, PoissonNllOptions, Reduction, SigmoidFocalLossOptions,
    TripletMarginOptions,
};
use fandhe_ai::{
    AutodiffError, CreateGraphResult, CustomFunction, Device, FftNorm, Tape, Tensor, Var,
};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::<f32>::new(data, shape).expect("test fixture: tensor")
}

/// REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。
fn req2_close(a: f64, b: f64) -> bool {
    let abs = (a - b).abs();
    abs < 1e-5 || abs / a.abs().max(b.abs()).max(f64::MIN_POSITIVE) < 1e-3
}

fn vals(t: &Tensor<f32>) -> Vec<f32> {
    t.host_slice().into_owned()
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

/// 恒久対象外の `Tape::custom` 用の恒等関数（拒否テストの基準 Op。`Op::Custom` は
/// `supports_create_graph()` が恒久的に偽。決定記録 §21.2 A 区分）。
struct IdentityFn;

impl CustomFunction for IdentityFn {
    fn name(&self) -> &str {
        "identity"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        Ok(inputs[0].clone())
    }
    fn backward(
        &self,
        _inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        _requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        Ok(vec![Some(upstream.clone())])
    }
}

#[test]
fn rejects_unsupported_op_and_keeps_child_empty() {
    let child = fandhe_ai::tape();
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![0.1, 0.2, 0.3], &[1, 3]));
    // `Op::Custom` は §21.2 A 区分（恒久対象外）。LogSumExp（#3792）・Max（#3789）は
    // 対象化済みのため、拒否の基準にしない（イシュー #3087）。
    let loss = tape
        .custom(Arc::new(IdentityFn), &[x])
        .unwrap()
        .sum(None)
        .unwrap();
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
        .mse_loss_with(&mse_target(tape), Reduction::Mean)
        .expect("mse_loss")
}

fn nll_loss_at<'t>(x: &Var<'t>) -> Var<'t> {
    let targets = Tensor::new(vec![2i32, 0], &[2]).expect("targets");
    x.log_softmax(1)
        .expect("log_softmax")
        .nll_loss(&targets, 1, Reduction::Mean)
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
// --- 拡張 Op 族の facade 経由確認（イシュー #3087。決定記録 §32）。

type Build = for<'a> fn(&'a Tape, &Var<'a>) -> Var<'a>;

/// 要素ごとに異なる固定重み（対称性による打ち消しを避ける）。
fn weights<'a>(tape: &'a Tape, shape: &[usize]) -> Var<'a> {
    let n: usize = shape.iter().product();
    let w = (0..n)
        .map(|i| 0.4 + 0.37 * i as f32 - 0.05 * (i * i) as f32)
        .collect();
    tape.var_no_grad(&t(w, shape))
}

fn plain_grad(x0: &[f32], shape: &[usize], build: Build) -> Vec<f32> {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x);
    let grads = tape.backward(&loss).expect("backward");
    vals(grads.get(&x).unwrap().expect("grad"))
}

/// 二階の検証に使う固定方向 `v`（要素ごとに異なる。一様な方向だと LayerNorm・BatchNorm・
/// CrossEntropy のように 1 階勾配の総和が恒等的に一定になる Op で二階がゼロに潰れる）。
fn probe_dir(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.6 - 0.23 * i as f32 + 0.04 * (i * i) as f32)
        .collect()
}

/// 方向微分 `d/dx (g(x)·v)`（= `H v`、Hessian は対称）を `backward_create_graph` で求める。
fn hessian_dir(parent: &Tape, child: &Tape, x0: &[f32], shape: &[usize], build: Build) -> Vec<f32> {
    let x = parent.var(&t(x0.to_vec(), shape));
    let loss = build(parent, &x);
    let cg = parent
        .backward_create_graph(&loss, child)
        .expect("create_graph");
    let g = cg.grad(&x).unwrap().expect("grad");
    let v = child.var_no_grad(&t(probe_dir(x0.len()), shape));
    let s = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = child.backward(&s).expect("child backward");
    let xc = cg.child_var(&x).unwrap().expect("mirror");
    vals(grads2.get(&xc).unwrap().expect("second order"))
}

/// `build` が作るスカラー損失について検証する。
/// - `first_order()` は素の `backward` と bit 一致（決定記録 §21.1(1)・§9 (i)）。
/// - 子テープ上の `cg.grad` は REQ-2（§9 (i)。bit 一致は V3 の族だけで、`grad_exact` が真のものは bit 一致）。
/// - `H v` を素の `backward` の方向中央差分（create_graph を通らない独立経路）と REQ-2 で比べ、
///   `H v` がゼロでないことを確認する。
fn check_create_graph_family(
    label: &str,
    x0: &[f32],
    shape: &[usize],
    build: Build,
    grad_exact: bool,
    eps: f32,
) {
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x);
    let cg = tape
        .backward_create_graph(&loss, &child)
        .unwrap_or_else(|e| panic!("{label}: create_graph 対象のはず: {e:?}"));
    let plain = plain_grad(x0, shape, build);
    let first = vals(cg.first_order().get(&x).unwrap().expect("first_order"));
    for (a, b) in plain.iter().zip(first.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "{label}: first_order は bit 一致");
    }
    let g = cg.grad(&x).unwrap().expect("grad");
    for (a, b) in plain.iter().zip(vals(&g.value()).iter()) {
        if grad_exact {
            assert_eq!(a.to_bits(), b.to_bits(), "{label}: cg.grad は bit 一致");
        } else {
            assert!(
                req2_close(*a as f64, *b as f64),
                "{label}: cg.grad {a} vs {b}"
            );
        }
    }
    drop(cg);
    let child2 = fandhe_ai::tape();
    let hv = hessian_dir(&fandhe_ai::tape(), &child2, x0, shape, build);
    assert!(
        hv.iter().any(|v| v.abs() > 1e-4),
        "{label}: Hv がゼロでは二階の検証にならない: {hv:?}"
    );
    // 成分 i の `H v` = ∂(g·v)/∂x_i を、`g(x ± eps e_i)·v` の中央差分で突合する。
    let v = probe_dir(x0.len());
    let dot = |x: &[f32]| -> f64 {
        plain_grad(x, shape, build)
            .iter()
            .zip(&v)
            .map(|(&a, &d)| a as f64 * d as f64)
            .sum()
    };
    for i in 0..x0.len() {
        let mut xp = x0.to_vec();
        let mut xm = x0.to_vec();
        xp[i] += eps;
        xm[i] -= eps;
        let num = (dot(&xp) - dot(&xm)) / (2.0 * eps as f64);
        assert!(
            req2_close(hv[i] as f64, num),
            "{label}: i={i} analytic={} numeric={num}",
            hv[i]
        );
    }
}

fn layer_norm_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let w = tape.var_no_grad(&t(vec![1.2, 0.7, -0.4], &[3]));
    let b = tape.var_no_grad(&t(vec![0.1, -0.2, 0.3], &[3]));
    let y = x.layer_norm(Some(&w), Some(&b), 1e-5).unwrap();
    y.mul(&weights(tape, &[2, 3])).unwrap().sum(None).unwrap()
}

fn rms_norm_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let w = tape.var_no_grad(&t(vec![1.2, 0.7, -0.4], &[3]));
    let y = x.rms_norm(Some(&w), 1e-5).unwrap();
    y.mul(&weights(tape, &[2, 3])).unwrap().sum(None).unwrap()
}

fn batch_norm_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let w = tape.var_no_grad(&t(vec![1.2, 0.7], &[2]));
    let b = tape.var_no_grad(&t(vec![0.1, -0.2], &[2]));
    let y = x.batch_norm(Some(&w), Some(&b), 1e-5).unwrap();
    y.mul(&weights(tape, &[4, 2])).unwrap().sum(None).unwrap()
}

fn conv2d_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let w = tape.var_no_grad(&t(vec![0.5, -0.3, 0.2, 0.8], &[1, 1, 2, 2]));
    let y = x.conv2d(&w, None, [1, 1], [0, 0], [1, 1], 1).unwrap();
    y.tanh().sum(None).unwrap()
}

/// 3D 畳み込み（#3797）。軸ごとに異なる kernel `[2, 2, 2]`・stride `[1, 1, 2]` で D/H/W の取り違えを拾う。
fn conv3d_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let w = tape.var_no_grad(&t(
        vec![0.5, -0.3, 0.2, 0.8, -0.6, 0.4, 0.1, -0.7],
        &[1, 1, 2, 2, 2],
    ));
    let y = x
        .conv3d(&w, None, [1, 1, 2], [0, 0, 0], [1, 1, 1], 1)
        .unwrap();
    y.tanh().sum(None).unwrap()
}

fn conv_transpose3d_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let w = tape.var_no_grad(&t(
        vec![0.5, -0.3, 0.2, 0.8, -0.6, 0.4, 0.1, -0.7],
        &[1, 1, 2, 2, 2],
    ));
    let y = x
        .conv_transpose3d(&w, None, [1, 2, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], 1)
        .unwrap();
    y.tanh().sum(None).unwrap()
}

fn gelu_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let y = x.gelu().unwrap();
    y.mul(&weights(tape, &[2, 3])).unwrap().sum(None).unwrap()
}

fn gelu_tanh_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let y = x.gelu_tanh().unwrap();
    y.mul(&weights(tape, &[2, 3])).unwrap().sum(None).unwrap()
}

fn huber_loss_at<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    // |tanh(x) - target| が delta = 0.5 の折れ目 ± eps をまたがない値にする。
    let target = tape.var_no_grad(&t(vec![0.0, 0.0, 0.1, 0.5, 0.2, -0.2], &[2, 3]));
    x.tanh().huber_loss(&target, 0.5, Reduction::Mean).unwrap()
}

fn cross_entropy_loss_at<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let targets = Tensor::new(vec![2i32, 0], &[2]).expect("targets");
    x.cross_entropy_loss(&targets, 1, Reduction::Mean).unwrap()
}

fn gather_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let index = Tensor::new(vec![0i32, 2, 1, 1], &[2, 2]).expect("index");
    x.gather(1, &index).unwrap().tanh().sum(None).unwrap()
}

fn cumsum_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    x.cumsum(1).unwrap().tanh().sum(None).unwrap()
}

fn pad_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let p = x.pad(&[(1, 0), (0, 1)], 0.0).unwrap();
    p.mul(&p).unwrap().sum(None).unwrap()
}

fn masked_fill_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let mask =
        Tensor::<bool>::new(vec![true, false, false, false, true, false], &[2, 3]).expect("mask");
    let m = x.masked_fill(&mask, 2.0).unwrap();
    m.mul(&m).unwrap().sum(None).unwrap()
}

/// 選択縮約（イシュー #3789）: `reduce(x⊙x⊙w)`。選ばれる位置はタイにならず、`H v` は選択位置の
/// 対角 `2·w·v` でゼロにならない。
fn select_reduce_loss<'a>(tape: &'a Tape, x: &Var<'a>, reduce: fn(&Var<'a>) -> Var<'a>) -> Var<'a> {
    let shape = x.value().shape().to_vec();
    let w = weights(tape, &shape);
    reduce(&x.mul(x).unwrap().mul(&w).unwrap())
}

fn max_reduce_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    select_reduce_loss(tape, x, |y| y.max(None).unwrap())
}

fn min_reduce_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    select_reduce_loss(tape, x, |y| y.min(None).unwrap())
}

fn amax_reduce_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    select_reduce_loss(tape, x, |y| y.amax(None).unwrap())
}

fn amin_reduce_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    select_reduce_loss(tape, x, |y| y.amin(None).unwrap())
}

fn max_pool2d_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let (y, _idx) = x
        .max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1], false)
        .unwrap();
    y.tanh().sum(None).unwrap()
}

fn avg_pool2d_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    x.avg_pool2d([2, 2], Some([1, 1]), [0, 0], false, true)
        .unwrap()
        .tanh()
        .sum(None)
        .unwrap()
}

fn inv_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let y = x.inv().unwrap();
    y.mul(&weights(tape, &[2, 2])).unwrap().sum(None).unwrap()
}

fn det_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    x.det().unwrap()
}

fn rfft_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let r = x.rfft(None, None, FftNorm::Backward).unwrap();
    r.mul(&r).unwrap().sum(None).unwrap()
}

const NX: [f32; 6] = [0.3, -0.8, 1.2, 0.1, -0.4, 0.9];
const NX2: [f32; 6] = [0.5, 1.1, -0.6, 0.2, -1.3, 0.8];
const CONV_X: [f32; 9] = [0.3, -0.8, 1.2, 0.1, -0.4, 0.9, 0.5, -0.2, 0.7];
/// 値がすべて異なり差が中央差分の刻みより十分大きい入力（argmax が入れ替わらない）。
const POOL_X: [f32; 9] = [0.1, 0.9, 0.3, 0.5, 0.2, 0.8, 0.7, 0.4, 0.6];
const INV_X: [f32; 4] = [2.0, 0.5, 0.3, 1.5];

#[test]
fn layer_norm_create_graph_via_facade() {
    check_create_graph_family("layer_norm", &NX2, &[2, 3], layer_norm_loss, true, 1e-2);
}

#[test]
fn rms_norm_create_graph_via_facade() {
    check_create_graph_family("rms_norm", &NX2, &[2, 3], rms_norm_loss, true, 1e-2);
}

#[test]
fn batch_norm_create_graph_via_facade() {
    let x0 = [0.5_f32, 1.1, -0.6, 0.2, -1.3, 0.8, 0.9, -0.4];
    check_create_graph_family("batch_norm", &x0, &[4, 2], batch_norm_loss, true, 1e-2);
}

#[test]
fn conv2d_create_graph_via_facade() {
    check_create_graph_family("conv2d", &CONV_X, &[1, 1, 3, 3], conv2d_loss, true, 1e-2);
}

/// d_input は V1 合成（unfold3d／fold3d と rank 2 strict matmul）なので `cg.grad` は REQ-2 で判定する
/// （bit 一致を主張しない。決定記録 §38.5・§59）。
#[test]
fn conv3d_create_graph_via_facade() {
    let x: Vec<f32> = (0..18)
        .map(|i| (((i * 7 + 3) % 11) as f32 - 5.0) * 0.15)
        .collect();
    check_create_graph_family("conv3d", &x, &[1, 1, 2, 3, 3], conv3d_loss, false, 1e-2);
}

#[test]
fn conv_transpose3d_create_graph_via_facade() {
    let x: Vec<f32> = (0..8)
        .map(|i| (((i * 5 + 2) % 9) as f32 - 4.0) * 0.2)
        .collect();
    check_create_graph_family(
        "conv_transpose3d",
        &x,
        &[1, 1, 2, 2, 2],
        conv_transpose3d_loss,
        false,
        1e-2,
    );
}

/// 累積積（#3803）。`dim = 1` の rank 2 で lane ごとに独立に累積する。
fn cumprod_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    x.cumprod(1).unwrap().tanh().sum(None).unwrap()
}

/// 累積 logsumexp（#3803）。
fn logcumsumexp_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    x.logcumsumexp(1).unwrap().tanh().sum(None).unwrap()
}

/// `cumprod` の `cg.grad` は V3（1 階と同じホスト関数の値ノード）なので CPU では bit 一致する。
/// 零を含む lane も除算を使わない scan 構造のため二階が厳密（決定記録 §41.10・§64）。
#[test]
fn cumprod_create_graph_via_facade() {
    let x: Vec<f32> = vec![0.9, 1.1, 0.8, 1.2, 1.05, 0.95, 1.0, 0.85];
    check_create_graph_family("cumprod", &x, &[2, 4], cumprod_loss, true, 1e-2);
    let with_zero: Vec<f32> = vec![0.9, 0.0, 0.8, 1.2, 1.05, 0.95, 1.0, 0.85];
    check_create_graph_family("cumprod(零)", &with_zero, &[2, 4], cumprod_loss, true, 1e-2);
}

#[test]
fn logcumsumexp_create_graph_via_facade() {
    let x: Vec<f32> = vec![0.3, -0.5, 1.1, 0.2, -0.9, 0.7, 0.0, 0.4];
    check_create_graph_family("logcumsumexp", &x, &[2, 4], logcumsumexp_loss, true, 1e-2);
}

/// ±inf 入力は型付きエラーで拒否される（panic しない。決定記録 §33.14 論点 3・§64）。
#[test]
fn cumulative_b_create_graph_rejects_inf_input_via_facade() {
    for build in [cumprod_loss, logcumsumexp_loss] {
        let tape = fandhe_ai::tape();
        let child = fandhe_ai::tape();
        let x = tape.var(&t(vec![1.0, f32::INFINITY, 0.5, 0.25], &[1, 4]));
        let loss = build(&tape, &x);
        assert!(tape.backward_create_graph(&loss, &child).is_err());
    }
}

/// 索引付き縮約（#3802）。`index` は `[2, 3]`（input `[3, 4]` より非 dim 軸が小さい）・重複あり・
/// 触れられない行つき。`Mean`／`Amax` は非自明な係数（`1/m`・選択マスク）を通す。
fn scatter_reduce_loss_with<'a>(
    tape: &'a Tape,
    x: &Var<'a>,
    mode: fandhe_ai::ScatterReduceMode,
) -> Var<'a> {
    let s = tape.var_no_grad(&t(vec![0.35, -1.1, 0.7, 1.9, -0.45, 0.05], &[2, 3]));
    let idx = Tensor::<i32>::new(vec![1, 1, 3, 0, 2, 2], &[2, 3]).unwrap();
    let y = x.scatter_reduce(1, &idx, &s, mode, true).unwrap();
    y.tanh().sum(None).unwrap()
}

fn scatter_reduce_x() -> Vec<f32> {
    (0..12)
        .map(|i| (((i * 5 + 1) % 13) as f32 - 6.0) * 0.17)
        .collect()
}

#[test]
fn scatter_reduce_sum_create_graph_via_facade() {
    check_create_graph_family(
        "scatter_reduce_sum",
        &scatter_reduce_x(),
        &[3, 4],
        |tp, x| scatter_reduce_loss_with(tp, x, fandhe_ai::ScatterReduceMode::Sum),
        true,
        1e-2,
    );
}

#[test]
fn scatter_reduce_mean_create_graph_via_facade() {
    check_create_graph_family(
        "scatter_reduce_mean",
        &scatter_reduce_x(),
        &[3, 4],
        |tp, x| scatter_reduce_loss_with(tp, x, fandhe_ai::ScatterReduceMode::Mean),
        true,
        1e-2,
    );
}

#[test]
fn scatter_reduce_amax_create_graph_via_facade() {
    check_create_graph_family(
        "scatter_reduce_amax",
        &scatter_reduce_x(),
        &[3, 4],
        |tp, x| scatter_reduce_loss_with(tp, x, fandhe_ai::ScatterReduceMode::Amax),
        true,
        1e-2,
    );
}

#[test]
fn gelu_create_graph_via_facade() {
    check_create_graph_family("gelu", &NX, &[2, 3], gelu_loss, false, 2e-2);
}

#[test]
fn gelu_tanh_create_graph_via_facade() {
    check_create_graph_family("gelu_tanh", &NX, &[2, 3], gelu_tanh_loss, false, 2e-2);
}

#[test]
fn huber_loss_create_graph_via_facade() {
    check_create_graph_family("huber_loss", &NX, &[2, 3], huber_loss_at, false, 1e-2);
}

#[test]
fn cross_entropy_loss_create_graph_via_facade() {
    check_create_graph_family(
        "cross_entropy",
        &LOSS_X,
        &[2, 3],
        cross_entropy_loss_at,
        false,
        1e-2,
    );
}

#[test]
fn gather_create_graph_via_facade() {
    check_create_graph_family("gather", &NX, &[2, 3], gather_loss, true, 1e-2);
}

#[test]
fn cumsum_create_graph_via_facade() {
    check_create_graph_family("cumsum", &NX, &[2, 3], cumsum_loss, true, 1e-2);
}

#[test]
fn pad_create_graph_via_facade() {
    check_create_graph_family("pad", &NX, &[2, 3], pad_loss, true, 1e-2);
}

#[test]
fn masked_fill_create_graph_via_facade() {
    check_create_graph_family("masked_fill", &NX, &[2, 3], masked_fill_loss, true, 1e-2);
}

#[test]
fn max_create_graph_via_facade() {
    check_create_graph_family("max", &NX, &[2, 3], max_reduce_loss, true, 1e-2);
}

#[test]
fn min_create_graph_via_facade() {
    check_create_graph_family("min", &NX, &[2, 3], min_reduce_loss, true, 1e-2);
}

#[test]
fn amax_create_graph_via_facade() {
    check_create_graph_family("amax", &NX, &[2, 3], amax_reduce_loss, true, 1e-2);
}

#[test]
fn amin_create_graph_via_facade() {
    check_create_graph_family("amin", &NX, &[2, 3], amin_reduce_loss, true, 1e-2);
}

/// 統計縮約（イシュー #3792）: Hessian がゼロにならないよう縮約結果を二乗する。入力は NaN を含まない
/// （有限差分は NaN の座標で使えないため）。
fn logsumexp_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let w = weights(tape, &[2, 3]);
    let z = x
        .mul(x)
        .unwrap()
        .mul(&w)
        .unwrap()
        .logsumexp(Some(1))
        .unwrap();
    z.mul(&z).unwrap().sum(None).unwrap()
}

fn nansum_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let m = x.mul(x).unwrap().nansum(None).unwrap();
    m.mul(&m).unwrap()
}

fn nanmean_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let m = x.mul(x).unwrap().nanmean(Some(1)).unwrap();
    m.mul(&m).unwrap().sum(None).unwrap()
}

#[test]
fn logsumexp_create_graph_via_facade() {
    check_create_graph_family("logsumexp", &NX, &[2, 3], logsumexp_loss, true, 1e-2);
}

#[test]
fn nansum_create_graph_via_facade() {
    check_create_graph_family("nansum", &NX, &[2, 3], nansum_loss, true, 1e-2);
}

#[test]
fn nanmean_create_graph_via_facade() {
    check_create_graph_family("nanmean", &NX, &[2, 3], nanmean_loss, true, 1e-2);
}

/// 損失の二値・マージン系（イシュー #3799）。1 本の葉 `z` を `[x; target]` へ `narrow` で切り出し、
/// 損失を二乗して `H v ≠ 0` にする（`y`・focal の `target`・`pos_weight` は定数）。
fn bce_loss_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let (x, tg) = (z.narrow(0, 0, 3).unwrap(), z.narrow(0, 3, 3).unwrap());
    let l = x.bce_loss(&tg, Reduction::Mean).unwrap();
    l.mul(&l).unwrap()
}

fn bce_with_logits_pos_weight_loss_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let (x, tg) = (z.narrow(0, 0, 3).unwrap(), z.narrow(0, 3, 3).unwrap());
    let opts = BceWithLogitsOptions::default().pos_weight(t(vec![2.0, 0.5, 3.0], &[3]));
    let l = x
        .bce_with_logits_loss_with(&tg, Reduction::Mean, &opts)
        .unwrap();
    l.mul(&l).unwrap()
}

fn sigmoid_focal_loss_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let opts = SigmoidFocalLossOptions::default()
        .alpha(Some(0.25))
        .gamma(2.0);
    let l = z
        .sigmoid_focal_loss(&t(vec![1.0, 0.0, 0.7], &[3]), &opts, Reduction::Mean)
        .unwrap();
    l.mul(&l).unwrap()
}

fn margin_ranking_loss_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let (x1, x2) = (z.narrow(0, 0, 3).unwrap(), z.narrow(0, 3, 3).unwrap());
    let l = x1
        .margin_ranking_loss(&x2, &t(vec![1.0, -1.0, 1.0], &[3]), 0.5, Reduction::Mean)
        .unwrap();
    l.mul(&l).unwrap()
}

#[test]
fn bce_loss_create_graph_via_facade() {
    let z = [0.4, 0.5, 0.6, 0.9, 0.1, 0.95];
    check_create_graph_family("bce_loss", &z, &[6], bce_loss_loss, false, 5e-3);
}

#[test]
fn bce_with_logits_pos_weight_loss_create_graph_via_facade() {
    let z = [0.3, -0.8, 0.5, 0.9, 0.2, 0.5];
    check_create_graph_family(
        "bce_with_logits_pos_weight_loss",
        &z,
        &[6],
        bce_with_logits_pos_weight_loss_loss,
        true,
        3e-3,
    );
}

#[test]
fn sigmoid_focal_loss_create_graph_via_facade() {
    check_create_graph_family(
        "sigmoid_focal_loss",
        &[0.3, -0.8, 0.5],
        &[3],
        sigmoid_focal_loss_loss,
        true,
        3e-3,
    );
}

#[test]
fn margin_ranking_loss_create_graph_via_facade() {
    // raw = −y·(x1 − x2) + margin が全要素で 0 から離れる（区分境界の中央差分をまたがない）値。
    let z = [0.9, 0.2, -0.3, 0.1, 0.6, 0.4];
    check_create_graph_family(
        "margin_ranking_loss",
        &z,
        &[6],
        margin_ranking_loss_loss,
        true,
        1e-2,
    );
}

/// 損失のクラス軸系（イシュー #3800）。1 本の葉 `z` を切り出して損失を作り、二乗して `H v ≠ 0` にする。
fn cross_entropy_loss_with_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let opts = CrossEntropyOptions::default()
        .label_smoothing(0.1)
        .class_weight(t(vec![1.0, 2.0, 0.5], &[3]));
    let l = z
        .reshape(&[2, 3])
        .unwrap()
        .cross_entropy_loss_with(
            &Tensor::<i32>::new(vec![0, 2], &[2]).unwrap(),
            1,
            Reduction::Mean,
            &opts,
        )
        .unwrap();
    l.mul(&l).unwrap()
}

fn multi_margin_loss_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let opts = MultiMarginOptions::default().p(2);
    let l = z
        .reshape(&[2, 3])
        .unwrap()
        .multi_margin_loss(
            &Tensor::<i32>::new(vec![1, 0], &[2]).unwrap(),
            &opts,
            Reduction::Mean,
        )
        .unwrap();
    l.mul(&l).unwrap()
}

fn multilabel_margin_loss_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let l = z
        .reshape(&[2, 3])
        .unwrap()
        .multilabel_margin_loss(
            &Tensor::<i32>::new(vec![2, -1, 0, 0, 1, -1], &[2, 3]).unwrap(),
            Reduction::Mean,
        )
        .unwrap();
    l.mul(&l).unwrap()
}

fn multilabel_soft_margin_loss_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let l = z
        .reshape(&[2, 3])
        .unwrap()
        .multilabel_soft_margin_loss(
            &t(vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.3], &[2, 3]),
            &MultiLabelSoftMarginOptions::default(),
            Reduction::Mean,
        )
        .unwrap();
    l.mul(&l).unwrap()
}

fn cosine_embedding_loss_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let (x1, x2) = (z.narrow(0, 0, 6).unwrap(), z.narrow(0, 6, 6).unwrap());
    let l = x1
        .reshape(&[2, 3])
        .unwrap()
        .cosine_embedding_loss(
            &x2.reshape(&[2, 3]).unwrap(),
            &t(vec![1.0, -1.0], &[2]),
            0.2,
            Reduction::Mean,
        )
        .unwrap();
    l.mul(&l).unwrap()
}

fn triplet_margin_loss_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let p = |i: usize| z.narrow(0, 6 * i, 6).unwrap().reshape(&[2, 3]).unwrap();
    let l = p(0)
        .triplet_margin_loss(
            &p(1),
            &p(2),
            &TripletMarginOptions::default().margin(2.0),
            Reduction::Mean,
        )
        .unwrap();
    l.mul(&l).unwrap()
}

#[test]
fn cross_entropy_loss_with_create_graph_via_facade() {
    let z = [0.2, -0.4, 0.9, 0.5, 0.1, -0.7];
    check_create_graph_family(
        "cross_entropy_loss_with",
        &z,
        &[6],
        cross_entropy_loss_with_loss,
        true,
        1e-2,
    );
}

#[test]
fn multi_margin_loss_create_graph_via_facade() {
    let z = [0.3, 0.9, -0.2, 0.8, 0.1, 0.5];
    check_create_graph_family(
        "multi_margin_loss",
        &z,
        &[6],
        multi_margin_loss_loss,
        true,
        1e-2,
    );
}

#[test]
fn multilabel_margin_loss_create_graph_via_facade() {
    let z = [0.3, 0.9, -0.2, 0.8, 0.1, -0.5];
    check_create_graph_family(
        "multilabel_margin_loss",
        &z,
        &[6],
        multilabel_margin_loss_loss,
        true,
        1e-2,
    );
}

#[test]
fn multilabel_soft_margin_loss_create_graph_via_facade() {
    let z = [0.3, 0.9, -0.2, 0.8, 0.1, -0.5];
    check_create_graph_family(
        "multilabel_soft_margin_loss",
        &z,
        &[6],
        multilabel_soft_margin_loss_loss,
        true,
        1e-2,
    );
}

#[test]
fn cosine_embedding_loss_create_graph_via_facade() {
    let z = [
        0.5, -0.3, 0.8, 0.6, 0.2, -0.5, 0.1, 0.7, -0.4, 0.5, 0.3, -0.4,
    ];
    check_create_graph_family(
        "cosine_embedding_loss",
        &z,
        &[12],
        cosine_embedding_loss_loss,
        true,
        5e-3,
    );
}

#[test]
fn triplet_margin_loss_create_graph_via_facade() {
    let z = [
        0.5, -0.3, 0.8, 0.6, 0.2, -0.5, 0.3, -0.1, 0.5, 0.1, 0.5, -0.2, 0.9, 0.5, 0.1, 0.7, -0.4,
        0.3,
    ];
    check_create_graph_family(
        "triplet_margin_loss",
        &z,
        &[18],
        triplet_margin_loss_loss,
        true,
        5e-3,
    );
}

/// 損失の回帰系（イシュー #3798）。1 本の葉 `z` を `[x; target (; var)]` へ `narrow` で切り出し、
/// 損失を二乗して `H v ≠ 0` にする。
fn kl_div_loss_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let (x, tg) = (z.narrow(0, 0, 3).unwrap(), z.narrow(0, 3, 3).unwrap());
    let l = x.kl_div_loss(&tg, Reduction::Sum).unwrap();
    l.mul(&l).unwrap()
}

fn poisson_nll_loss_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let (x, tg) = (z.narrow(0, 0, 3).unwrap(), z.narrow(0, 3, 3).unwrap());
    let l = x
        .poisson_nll_loss(&tg, &PoissonNllOptions::default(), Reduction::Mean)
        .unwrap();
    l.mul(&l).unwrap()
}

fn gaussian_nll_loss_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let (x, tg, v) = (
        z.narrow(0, 0, 3).unwrap(),
        z.narrow(0, 3, 3).unwrap(),
        z.narrow(0, 6, 3).unwrap(),
    );
    let l = x
        .gaussian_nll_loss(&tg, &v, &GaussianNllOptions::default(), Reduction::Mean)
        .unwrap();
    l.mul(&l).unwrap()
}

#[test]
fn kl_div_loss_create_graph_via_facade() {
    let z = [0.3, -0.8, 0.5, 0.45, 0.6, 0.75];
    check_create_graph_family("kl_div_loss", &z, &[6], kl_div_loss_loss, false, 3e-3);
}

#[test]
fn poisson_nll_loss_create_graph_via_facade() {
    let z = [0.3, -0.8, 0.5, 1.0, 2.0, 0.5];
    check_create_graph_family(
        "poisson_nll_loss",
        &z,
        &[6],
        poisson_nll_loss_loss,
        true,
        3e-3,
    );
}

#[test]
fn gaussian_nll_loss_create_graph_via_facade() {
    let z = [0.3, -0.8, 0.5, 0.1, 0.4, -0.2, 0.7, 1.1, 0.9];
    check_create_graph_family(
        "gaussian_nll_loss",
        &z,
        &[9],
        gaussian_nll_loss_loss,
        true,
        1e-2,
    );
}

fn local_response_norm_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let y = z
        .reshape(&[1, 3, 2])
        .unwrap()
        .local_response_norm(3, 0.5, 0.75, 1.0)
        .unwrap();
    let l = y.mul(&y).unwrap().sum(None).unwrap();
    l.mul(&l).unwrap()
}

fn weight_norm_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    // z = [v(6); g(2)]。v と g の混合二階項を含む。
    let v = z.narrow(0, 0, 6).unwrap().reshape(&[2, 3]).unwrap();
    let g = z.narrow(0, 6, 2).unwrap().reshape(&[2, 1]).unwrap();
    // `Σ w²` は `g` だけで決まり、上流が `w` に平行になって相殺するため `tanh` を挟む。
    let w = v.weight_norm(&g, Some(0)).unwrap();
    let l = w.tanh().sum(None).unwrap();
    l.mul(&l).unwrap()
}

fn spectral_norm_loss<'a>(_tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let w = z.reshape(&[2, 3]).unwrap();
    let mut st = fandhe_ai::SpectralNormState::from_vectors(
        &[2, 3],
        0,
        &[0.6, 0.8],
        &[0.5, 0.3, 0.8],
        1,
        1e-12,
    )
    .unwrap();
    let y = w.spectral_norm(&mut st, false).unwrap();
    let l = y.mul(&y).unwrap().sum(None).unwrap();
    l.mul(&l).unwrap()
}

#[test]
fn local_response_norm_create_graph_via_facade() {
    let z = [0.3, -0.8, 0.5, 1.0, 0.7, 0.4];
    check_create_graph_family(
        "local_response_norm",
        &z,
        &[6],
        local_response_norm_loss,
        true,
        1e-2,
    );
}

#[test]
fn weight_norm_create_graph_via_facade() {
    // g = [1.5, 0.9] では group 0 の Hessian 対角（射影の項が約 7.5 で打ち消し合い値は約 0.01）の
    // f32 の最悪誤差が REQ-2 を外れうるため、総括判定が保守的に拒否する（決定記録 §57.2）。
    // 判定を通る g へ下げて facade 経由の二階を確かめる。
    let z = [0.3, -0.8, 0.5, 1.0, 0.7, 0.4, 0.8, 0.5];
    check_create_graph_family("weight_norm", &z, &[8], weight_norm_loss, true, 3e-3);
}

#[test]
fn spectral_norm_create_graph_via_facade() {
    let z = [0.9, 0.4, 0.5, 0.6, 0.7, 0.8];
    check_create_graph_family("spectral_norm", &z, &[6], spectral_norm_loss, true, 3e-3);
}

#[test]
fn max_pool2d_create_graph_via_facade() {
    check_create_graph_family(
        "max_pool2d",
        &POOL_X,
        &[1, 1, 3, 3],
        max_pool2d_loss,
        true,
        1e-2,
    );
}

#[test]
fn avg_pool2d_create_graph_via_facade() {
    check_create_graph_family(
        "avg_pool2d",
        &POOL_X,
        &[1, 1, 3, 3],
        avg_pool2d_loss,
        true,
        1e-2,
    );
}

#[test]
fn inv_create_graph_via_facade() {
    check_create_graph_family("inv", &INV_X, &[2, 2], inv_loss, false, 1e-2);
}

#[test]
fn det_create_graph_via_facade() {
    // V3（#3810・決定記録 §70）: `cg.grad` が 1 階と bit 一致（`grad_exact = true`。強めるだけで緩めない）。
    check_create_graph_family("det", &INV_X, &[2, 2], det_loss, true, 1e-2);
}

/// 行別スケール（イシュー #3810・決定記録 §46・§70）: `det` が f32 を超える `diag(1e20, 1e20)` が受理され、
/// 1 階が素の `backward` と bit 一致、`cg.grad` も bit 一致、二階が有限。真の 1 階値が f32 を超える
/// `A = I`・`g = 3e38` は拒否のままで `child` は空に保たれる。
#[test]
fn det_row_scaled_create_graph_via_facade() {
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(vec![1e20, 0.0, 0.0, 1e20], &[2, 2]));
    let loss = x.det().unwrap();
    let plain = vals(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
    let cg = tape
        .backward_create_graph(&loss, &child)
        .unwrap_or_else(|e| panic!("diag(1e20,1e20): 受理されるはず {e:?}"));
    let first = vals(cg.first_order().get(&x).unwrap().expect("first_order"));
    let g = cg.grad(&x).unwrap().expect("grad");
    let gv = vals(&g.value());
    for ((a, b), c) in plain.iter().zip(&first).zip(&gv) {
        assert_eq!(a.to_bits(), b.to_bits(), "first_order は bit 一致");
        assert_eq!(a.to_bits(), c.to_bits(), "cg.grad は bit 一致（V3）");
    }
    assert!(gv.iter().all(|v| v.is_finite()), "cg.grad {gv:?}");
    let grads2 = child.backward(&g.sum(None).unwrap()).unwrap();
    let xc = cg.child_var(&x).unwrap().unwrap();
    let h = vals(grads2.get(&xc).unwrap().unwrap());
    assert!(h.iter().all(|v| v.is_finite()), "二階 {h:?}");

    // 区分 (i): 真の 1 階値が f32 を超える入力は拒否のまま。
    let rej_child = fandhe_ai::tape();
    let eye = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let big = tape.var_no_grad(&t(vec![3e38], &[]));
    let rej_loss = eye.det().unwrap().mul(&big).unwrap();
    assert!(matches!(
        tape.backward_create_graph(&rej_loss, &rej_child),
        Err(AutodiffError::Backward(_))
    ));
    assert_eq!(rej_child.leaf_count(), 0, "拒否後は child が空");
}

#[test]
fn rfft_create_graph_via_facade() {
    check_create_graph_family("rfft", &[0.5, -1.0, 0.25, 2.0], &[4], rfft_loss, true, 1e-2);
}

// 選択縮約 3 Op（イシュー #3790。決定記録 §34）。api_surface.rs にこの族を拒否する否定ガードは無く
// （§34.1）、`grad_exact = true` の挙動テストが正ガードを担う: `supports_create_graph` を `false` へ
// 戻すと `backward_create_graph` が Err になり、ここで panic する。
const SEL_X: [f32; 12] = [0.9, 0.3, 1.4, 0.6, 1.1, 0.2, 0.8, 1.7, 0.5, 1.3, 0.4, 1.0];

fn kthvalue_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let y = x.mul(x).unwrap().kthvalue(2, 1).unwrap().0;
    y.tanh().sum(None).unwrap()
}

fn median_dim_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let y = x.mul(x).unwrap().median(Some(0)).unwrap();
    y.tanh().sum(None).unwrap()
}

fn median_all_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let y = x.mul(x).unwrap().median(None).unwrap();
    y.tanh().sum(None).unwrap()
}

fn quantile_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let y = x
        .mul(x)
        .unwrap()
        .quantile(0.4, Some(1), fandhe_ai::QuantileInterpolation::Linear)
        .unwrap();
    y.tanh().sum(None).unwrap()
}

#[test]
fn kthvalue_create_graph_via_facade() {
    check_create_graph_family("kthvalue", &SEL_X, &[3, 4], kthvalue_loss, true, 1e-2);
}

#[test]
fn median_dim_create_graph_via_facade() {
    check_create_graph_family("median_dim", &SEL_X, &[3, 4], median_dim_loss, true, 1e-2);
}

#[test]
fn median_all_create_graph_via_facade() {
    check_create_graph_family("median_all", &SEL_X, &[3, 4], median_all_loss, true, 1e-2);
}

#[test]
fn quantile_create_graph_via_facade() {
    check_create_graph_family("quantile", &SEL_X, &[3, 4], quantile_loss, true, 1e-2);
}

// 統計縮約 4 Op（イシュー #3791・決定記録 §33・§52）。`tanh` で包んで上流 g と x の混合項を
// 非自明にする（L1 は x についての二階が 0 なので、tanh の曲率で `Hv ≠ 0` を作る）。
// 孫テープ・総括判定の網羅は `crates/autodiff/tests/create_graph_stat_varnorm.rs`。
fn var_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    x.var(Some(1), 1).unwrap().tanh().sum(None).unwrap()
}

fn std_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    x.std(Some(1), 1).unwrap().tanh().sum(None).unwrap()
}

fn norm_l1_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    x.norm_l1(Some(1)).unwrap().tanh().sum(None).unwrap()
}

fn norm_l2_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    x.norm_l2(Some(1)).unwrap().tanh().sum(None).unwrap()
}

fn norm_p_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    x.norm_p(1.5, Some(1)).unwrap().tanh().sum(None).unwrap()
}

#[test]
fn var_create_graph_via_facade() {
    check_create_graph_family("var", &SEL_X, &[3, 4], var_loss, true, 1e-2);
}

#[test]
fn std_create_graph_via_facade() {
    check_create_graph_family("std", &SEL_X, &[3, 4], std_loss, true, 1e-2);
}

#[test]
fn norm_l1_create_graph_via_facade() {
    check_create_graph_family("norm_l1", &SEL_X, &[3, 4], norm_l1_loss, true, 1e-2);
}

#[test]
fn norm_l2_create_graph_via_facade() {
    check_create_graph_family("norm_l2", &SEL_X, &[3, 4], norm_l2_loss, true, 1e-2);
}

#[test]
fn norm_p_create_graph_via_facade() {
    check_create_graph_family("norm_p", &SEL_X, &[3, 4], norm_p_loss, true, 1e-2);
}

/// GPU 実機: LayerNorm・Conv2d・MaxPool2d（線形代数は forward がホスト f64 フォールバックのため
/// 代表にしない。§31.6 申し送り 5）について、CPU と device で `H v`・HVP を REQ-2 で比べる。
fn gpu_extended_ops_match_cpu(device: Device) {
    type Case = (&'static str, &'static [f32], &'static [usize], Build);
    let cases: [Case; 3] = [
        ("layer_norm", &NX2, &[2, 3], layer_norm_loss),
        ("conv2d", &CONV_X, &[1, 1, 3, 3], conv2d_loss),
        ("max_pool2d", &POOL_X, &[1, 1, 3, 3], max_pool2d_loss),
    ];
    let dev = || fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテスト");
    let hvp_ones = |parent: &Tape, child: &Tape, x0: &[f32], shape: &[usize], build: Build| {
        let x = parent.var(&t(x0.to_vec(), shape));
        let loss = build(parent, &x);
        let v = t(vec![1.0; x0.len()], shape);
        vals(&parent.hvp(&loss, &x, &v, child).expect("hvp"))
    };
    for (label, x0, shape, build) in cases {
        let h_cpu = hessian_dir(&fandhe_ai::tape(), &fandhe_ai::tape(), x0, shape, build);
        let h_gpu = hessian_dir(&dev(), &dev(), x0, shape, build);
        for (a, b) in h_cpu.iter().zip(h_gpu.iter()) {
            assert!(
                req2_close(*a as f64, *b as f64),
                "{label}: cpu {a} vs gpu {b}"
            );
        }
        let v_cpu = hvp_ones(&fandhe_ai::tape(), &fandhe_ai::tape(), x0, shape, build);
        let v_gpu = hvp_ones(&dev(), &dev(), x0, shape, build);
        for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
            assert!(
                req2_close(*a as f64, *b as f64),
                "{label}: hvp cpu {a} vs gpu {b}"
            );
        }
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/create-graph-extended-ops-3087/README.md 参照"]
fn create_graph_extended_ops_cuda_matches_cpu() {
    gpu_extended_ops_match_cpu(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/create-graph-extended-ops-3087/README.md 参照"]
fn create_graph_extended_ops_metal_matches_cpu() {
    gpu_extended_ops_match_cpu(Device::Metal);
}
// --- 構造変換 3 Op（イシュー #3793・決定記録 §35/§56） ----------------------------------

/// グローバル RNG を使う `dropout` の再現（`manual_seed` → マスク抽選）をテスト間で直列化する。
fn rng_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// 線形 Op の出力 `y` を二乗和へ流して `H v ≠ 0` にする（素の `sum(op(x))` は Hessian が 0）。
fn quad_loss<'a>(tape: &'a Tape, y: &Var<'a>) -> Var<'a> {
    let shape = y.value().shape().to_vec();
    let w = weights(tape, &shape);
    y.mul(y).unwrap().mul(&w).unwrap().sum(None).unwrap()
}

fn dropout_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    // 呼び出しのたびに同じマスクを引く（`check_create_graph_family` は build を複数回呼ぶ）。
    fandhe_ai::manual_seed(20_261_101);
    quad_loss(tape, &x.dropout(0.4, true).unwrap())
}

fn interpolate_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let y = x
        .interpolate(
            &[3, 4],
            fandhe_ai::InterpolateMode::Bilinear {
                align_corners: false,
            },
        )
        .unwrap();
    quad_loss(tape, &y)
}

fn max_unpool2d_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let (p, idx) = x
        .max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1], false)
        .unwrap();
    let y = p
        .max_unpool2d(&idx, [2, 2], Some([1, 1]), [0, 0], None)
        .unwrap();
    quad_loss(tape, &y)
}

#[test]
fn dropout_create_graph_via_facade() {
    let _g = rng_lock();
    let x = [
        0.3, -0.7, 0.9, 0.2, -0.4, 0.8, 0.5, -0.1, 0.6, -0.9, 0.4, 0.7,
    ];
    // drop 位置の符号付きゼロは `==` で同一視されるが、本ケースの上流は有限・正負混在でも
    // 1 階 `g·mask` と bit 一致する（drop 位置は `g·0` が ±0。`to_bits` ではなく値で比べる箇所は
    // autodiff 側 `create_graph_structural.rs` が担う）。ここでは REQ-2 判定で突合する。
    check_create_graph_family("dropout", &x, &[3, 4], dropout_loss, false, 1.0);
}

#[test]
fn interpolate_create_graph_via_facade() {
    let x = [
        0.3, -0.7, 0.9, 0.2, -0.4, 0.8, 0.5, -0.1, 0.6, -0.9, 0.4, 0.7,
    ];
    check_create_graph_family(
        "interpolate",
        &x,
        &[1, 2, 2, 3],
        interpolate_loss,
        true,
        1.0,
    );
}

#[test]
fn max_unpool2d_create_graph_via_facade() {
    // 勝者が入れ替わらない間隔（0.3 以上）の入力。
    let x = [0.1, 0.9, 0.4, 1.3, 0.7, 0.2, 1.1, 0.5, 1.6, 0.3, 0.8, 1.9];
    check_create_graph_family(
        "max_unpool2d",
        &x,
        &[1, 1, 3, 4],
        max_unpool2d_loss,
        true,
        0.05,
    );
}

/// `Tape::backward_create_graph` の facade 経由で、ドキュメントの Hessian 例（`Nearest` の
/// アップサンプル: 対角 = 2·（コピー回数））が成り立つ。
#[test]
fn interpolate_nearest_hessian_diagonal_is_twice_copy_count() {
    let t0 = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = t0.var(&t(vec![0.5, -1.0], &[1, 1, 2]));
    let y = x
        .interpolate(&[4], fandhe_ai::InterpolateMode::Nearest)
        .unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let cg = t0.backward_create_graph(&loss, &child).unwrap();
    let g = cg.grad(&x).unwrap().unwrap();
    let xc = cg.child_var(&x).unwrap().unwrap();
    for i in 0..2 {
        let gi = g.narrow(2, i, 1).unwrap().sum(None).unwrap();
        let h = child.backward(&gi).unwrap();
        let row = vals(h.get(&xc).unwrap().unwrap());
        // 各入力は 2 回コピーされる: d²/dx² Σ y² = 2·2 = 4。
        assert!((row[i] - 4.0).abs() < 1e-5, "{row:?}");
        assert!(row[1 - i].abs() < 1e-6, "{row:?}");
    }
}

// --- RNN セル 2 Op（イシュー #3795・決定記録 §37/§58） ------------------------------------
//
// 公開 API の `rnn_forward_seq`／`gru_forward_seq` の `x` は定数の `Tensor` で、`GateParams` は facade から
// 使えない。そのため微分する葉は初期の隠れ状態 `h0`（`[B, H]`）。x・重みについての Hessian は
// autodiff 側 `create_graph_rnn_gru_cell.rs` が担う。

fn rnn_x() -> Tensor<f32> {
    let v: Vec<f32> = (0..2 * 2 * 3)
        .map(|i| 0.5 * ((i as f32) * 0.37 + 0.2).cos())
        .collect();
    t(v, &[2, 2, 3])
}

fn rnn_loss<'a>(tape: &'a Tape, h0: &Var<'a>) -> Var<'a> {
    let rnn = fandhe_ai::nn::rnn::Rnn::new(3, 4, true, 7).unwrap();
    let h_n = tape.rnn_forward_seq(&rnn, &rnn_x(), Some(h0)).unwrap().h_n;
    quad_loss(tape, &h_n)
}

fn gru_loss<'a>(tape: &'a Tape, h0: &Var<'a>) -> Var<'a> {
    let gru = fandhe_ai::nn::rnn::Gru::new(3, 4, true, 7).unwrap();
    let h_n = tape.gru_forward_seq(&gru, &rnn_x(), Some(h0)).unwrap().h_n;
    quad_loss(tape, &h_n)
}

#[test]
fn rnn_create_graph_via_facade() {
    let h0 = [0.3, -0.7, 0.9, 0.2, -0.4, 0.8, 0.5, -0.1];
    check_create_graph_family("rnn", &h0, &[2, 4], rnn_loss, true, 0.02);
}

#[test]
fn gru_create_graph_via_facade() {
    let h0 = [0.3, -0.7, 0.9, 0.2, -0.4, 0.8, 0.5, -0.1];
    // GRU は子テープでゲートを作り直す（forward の `mul_add` と丸めがずれる）ので REQ-2 で突合する。
    check_create_graph_family("gru", &h0, &[2, 4], gru_loss, false, 0.02);
}

// --- LSTM（イシュー #3796・決定記録 §37/§61） ------------------------------------------------
//
// 葉は `z: [B, 2H]` 1 つで、`h0 = z[:, :H]`・`c0 = z[:, H:]` を切り出して両方の初期状態を微分する。
// 損失は `h_n` と `c_n` の両方に掛かる（`LstmHidden` も `LstmCell` も祖先になる）。

fn lstm_loss<'a>(tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let lstm = fandhe_ai::nn::rnn::Lstm::new(3, 4, true, 7).unwrap();
    let h0 = z.narrow(1, 0, 4).unwrap();
    let c0 = z.narrow(1, 4, 4).unwrap();
    let out = tape
        .lstm_forward_seq(&lstm, &rnn_x(), Some(&h0), Some(&c0))
        .unwrap();
    quad_loss(tape, &out.h_n)
        .add(&quad_loss(tape, &out.c_n))
        .unwrap()
}

#[test]
fn lstm_create_graph_via_facade() {
    let z = [
        0.3, -0.7, 0.9, 0.2, 0.1, 0.4, -0.3, 0.6, -0.4, 0.8, 0.5, -0.1, 0.25, -0.5, 0.7, -0.2,
    ];
    // CPU で bit 一致が観測されても主張は REQ-2 に留める（`grad_exact = false`）。
    check_create_graph_family("lstm", &z, &[2, 8], lstm_loss, false, 0.02);
}
/// STFT 族の損失用の穏やかな重み（`weights` は添字の二乗で大きく振れるため飽和を避けて別に持つ）。
fn stft_weights<'a>(tape: &'a Tape, shape: &[usize]) -> Var<'a> {
    let n: usize = shape.iter().product();
    let w = (0..n)
        .map(|i| 0.6 + 0.4 * ((i as f32) * 0.9).sin())
        .collect();
    tape.var_no_grad(&t(w, shape))
}

/// `Stft`（`stft`。イシュー #3804・決定記録 §42・§63）。窓なし（矩形窓）・`center` 既定。二次形式の損失は中央差分が厳密になる（決定記録 §31.4）。
fn stft_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let opts = fandhe_ai::StftOptions::default().with_hop_length(2);
    let y = x.stft(8, None, &opts).unwrap();
    let shape = y.value().shape().to_vec();
    let w = stft_weights(tape, &shape);
    y.mul(&y).unwrap().mul(&w).unwrap().sum(None).unwrap()
}

#[test]
fn stft_create_graph_via_facade() {
    let x0: Vec<f32> = (0..16)
        .map(|i| ((i as f32) * 0.7).sin() * 0.8 + 0.1)
        .collect();
    // 随伴 Op はホスト計算のため CPU でも主張は REQ-2 に留める（`grad_exact = false`）。
    check_create_graph_family("stft", &x0, &[16], stft_loss, false, 0.25);
}

/// `Istft`（`istft`。同上）。`[N, T, 2]` のスペクトルから信号を再構成する。
fn istft_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let opts = fandhe_ai::IstftOptions::default().with_hop_length(2);
    let y = x.istft(8, None, &opts).unwrap();
    let shape = y.value().shape().to_vec();
    let w = stft_weights(tape, &shape);
    y.mul(&y).unwrap().mul(&w).unwrap().sum(None).unwrap()
}

#[test]
fn istft_create_graph_via_facade() {
    let x0: Vec<f32> = (0..50)
        .map(|i| ((i as f32) * 0.37).cos() * 0.6 - 0.05)
        .collect();
    check_create_graph_family("istft", &x0, &[5, 5, 2], istft_loss, false, 0.25);
}

// --- 線形代数 Cholesky・Pinv・Lstsq（イシュー #3805・決定記録 §43・§65） --------------------------
//
// `cholesky` は `A = (X+Xᵀ)/2 + cI` で `X` を微分する葉にする（VJP が `dA` を対称化し forward は下三角しか
// 読まないため）。`pinv`・`lstsq` は打ち切りなし（条件の良い縦長）の入力だけ受理される。

fn spd_from<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let half = tape.var_no_grad(&t(vec![0.5; 9], &[3, 3]));
    let mut eye = vec![0.0f32; 9];
    for i in 0..3 {
        eye[i * 3 + i] = 4.0;
    }
    let eye = tape.var_no_grad(&t(eye, &[3, 3]));
    x.add(&x.transpose(0, 1).unwrap())
        .unwrap()
        .mul(&half)
        .unwrap()
        .add(&eye)
        .unwrap()
}

fn cholesky_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    quad_loss(tape, &spd_from(tape, x).cholesky().unwrap())
}

#[test]
fn cholesky_create_graph_via_facade() {
    let x0 = [0.3, -0.5, 0.2, 0.1, 0.4, -0.2, 0.6, 0.0, -0.3];
    // 損失が `L` について非線形（二次形式）なので `cg.grad` は REQ-2 で突合する（`grad_exact = false`）。
    check_create_graph_family("cholesky", &x0, &[3, 3], cholesky_loss, false, 0.05);
}

fn pinv_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    quad_loss(tape, &x.pinv(None).unwrap())
}

#[test]
fn pinv_create_graph_via_facade() {
    // 条件の良い 3x2（打ち切りなし）。
    let x0 = [2.0, 0.3, -0.4, 1.5, 0.2, -0.6];
    check_create_graph_family("pinv", &x0, &[3, 2], pinv_loss, false, 0.01);
}

fn lstsq_loss<'a>(tape: &'a Tape, z: &Var<'a>) -> Var<'a> {
    let a = z.narrow(0, 0, 6).unwrap().reshape(&[3, 2]).unwrap();
    let b = z.narrow(0, 6, 3).unwrap().reshape(&[3, 1]).unwrap();
    quad_loss(tape, &a.lstsq(&b, None).unwrap())
}

#[test]
fn lstsq_create_graph_via_facade() {
    // 前半が A（3x2）、後半が B（3x1）。A と B の混合二階も通る。
    let z = [2.0, 0.3, -0.4, 1.5, 0.2, -0.6, 0.7, -0.2, 0.5];
    check_create_graph_family("lstsq", &z, &[9], lstsq_loss, false, 0.01);
}

/// 打ち切りありの `pinv`（`A = diag(2, 1)`・`rcond = 0.75`）は型付きエラーで拒否され、`child` は空のまま。
#[test]
fn pinv_truncated_is_rejected_via_facade() {
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(vec![2.0, 0.0, 0.0, 1.0], &[2, 2]));
    let loss = x.pinv(Some(0.75)).unwrap().sum(None).unwrap();
    let r = tape.backward_create_graph(&loss, &child);
    assert!(matches!(r, Err(AutodiffError::Backward(_))));
    assert_eq!(child.leaf_count(), 0, "拒否後は child が空");
}

// --- 行列ノルム・slogdet.sign・matrix_rank（イシュー #3806・決定記録 §43・§66） ----------------------
//
// `One`・`Inf` は A に関する二階がゼロなので、外側の二次形式（`y²`）で `g` を経由する二階を非ゼロにする。
// 入力は One の最大列和・Inf の最大行和が一意（タイから離れた）値。`sign`・`matrix_rank` は区分定数で
// 単独では勾配がゼロのため、`sign · exp(logabsdet)`（= det）の二乗・`rank · Σx²` の中で検査する。

fn matrix_norm_loss<'a>(tape: &'a Tape, x: &Var<'a>, ord: fandhe_ai::MatrixNormOrd) -> Var<'a> {
    quad_loss(tape, &x.matrix_norm(ord).unwrap())
}

#[test]
fn matrix_norm_create_graph_via_facade() {
    use fandhe_ai::MatrixNormOrd::{Fro, Inf, One};
    let fro = [2.0, 0.3, -0.4, 1.5, 0.2, -0.6];
    check_create_graph_family(
        "matrix_norm_fro",
        &fro,
        &[3, 2],
        |t, x| matrix_norm_loss(t, x, Fro),
        true,
        0.01,
    );
    let tie_free = [1.0, -2.0, 3.0, 0.5, -0.7, 1.1];
    check_create_graph_family(
        "matrix_norm_one",
        &tie_free,
        &[3, 2],
        |t, x| matrix_norm_loss(t, x, One),
        true,
        0.01,
    );
    check_create_graph_family(
        "matrix_norm_inf",
        &tie_free,
        &[3, 2],
        |t, x| matrix_norm_loss(t, x, Inf),
        true,
        0.01,
    );
}

fn slogdet_sign_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let s = x.slogdet().unwrap();
    let det = s.sign.mul(&s.logabsdet.exp()).unwrap();
    quad_loss(tape, &det)
}

#[test]
fn slogdet_sign_create_graph_via_facade() {
    let x0 = [2.0, 0.3, -0.4, 0.1, 1.5, 0.2, -0.6, 0.5, 1.8];
    check_create_graph_family("slogdet_sign", &x0, &[3, 3], slogdet_sign_loss, false, 0.01);
}

fn matrix_rank_loss<'a>(_tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let r = x.matrix_rank(None).unwrap();
    r.mul(&x.mul(x).unwrap().sum(None).unwrap()).unwrap()
}

#[test]
fn matrix_rank_create_graph_via_facade() {
    let x0 = [2.0, 0.3, -0.4, 0.1, 1.5, 0.2, -0.6, 0.5, 1.8];
    check_create_graph_family("matrix_rank", &x0, &[3, 3], matrix_rank_loss, true, 0.01);
}

/// `Nuc` は型付きエラーで拒否され、`child` は空のまま。
#[test]
fn matrix_norm_nuc_is_rejected_via_facade() {
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let loss = x
        .matrix_norm(fandhe_ai::MatrixNormOrd::Nuc)
        .unwrap()
        .sum(None)
        .unwrap();
    let r = tape.backward_create_graph(&loss, &child);
    assert!(matches!(r, Err(AutodiffError::Backward(_))));
    assert_eq!(child.leaf_count(), 0, "拒否後は child が空");
}

// --- 分解 QR（イシュー #3807・決定記録 §44・§67） ----------------------------------------------

/// `QrQ`・`QrR` の両方を使う損失（重み付きにして、列ノルム 1 の制約で定数に縮まないようにする）。
fn qr_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let f = x.qr().unwrap();
    let wq = tape.var_no_grad(&t(vec![0.9, -0.4, 0.7, 0.3, -0.8, 0.5], &[3, 2]));
    let wr = tape.var_no_grad(&t(vec![0.6, -0.5, 0.2, 0.8], &[2, 2]));
    let lq = f.q.tanh().mul(&wq).unwrap().sum(None).unwrap();
    let lr = f.r.mul(&f.r).unwrap().mul(&wr).unwrap().sum(None).unwrap();
    lq.add(&lr).unwrap()
}

#[test]
fn qr_create_graph_via_facade() {
    let x0 = [2.0, 0.3, -0.4, 1.5, 0.2, 1.8];
    check_create_graph_family("qr", &x0, &[3, 2], qr_loss, false, 0.01);
}

/// 横長（`m < n`）は 1 階も未対応のため型付きエラーで拒否され、`child` は空のまま。
#[test]
fn qr_wide_is_rejected_via_facade() {
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0], &[2, 3]));
    let loss = x.qr().unwrap().r.sum(None).unwrap();
    let r = tape.backward_create_graph(&loss, &child);
    assert!(matches!(r, Err(AutodiffError::Backward(_))));
    assert_eq!(child.leaf_count(), 0, "拒否後は child が空");
}
// --- 分解 SVD・Eigh（イシュー #3808・決定記録 §44・§68） ------------------------------------------

/// `SvdU`・`SvdS`・`SvdVh` の全てを使う損失（スペクトルがよく分離した縦長入力）。
fn svd_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let f = x.svd().unwrap();
    let wu = tape.var_no_grad(&t(vec![0.9, -0.4, 0.7, 0.3, -0.8, 0.5], &[3, 2]));
    let wv = tape.var_no_grad(&t(vec![0.6, -0.5, 0.2, 0.8], &[2, 2]));
    let lu = f.u.tanh().mul(&wu).unwrap().sum(None).unwrap();
    let ls = f.s.mul(&f.s).unwrap().sum(None).unwrap();
    let lv = f.vh.tanh().mul(&wv).unwrap().sum(None).unwrap();
    lu.add(&ls).unwrap().add(&lv).unwrap()
}

#[test]
fn svd_create_graph_via_facade() {
    let x0 = [3.0, 0.3, -0.4, 1.5, 0.2, 0.9];
    check_create_graph_family("svd", &x0, &[3, 2], svd_loss, false, 0.01);
}

/// `EighValues`・`EighVectors` の両方を使う損失。入力は対称（下三角のみ読まれる）。
fn eigh_loss<'a>(tape: &'a Tape, x: &Var<'a>) -> Var<'a> {
    let f = x.eigh().unwrap();
    let wl = tape.var_no_grad(&t(vec![0.9, -0.4, 0.7], &[3]));
    let wv = tape.var_no_grad(&t(
        vec![0.6, -0.5, 0.2, 0.8, 0.3, -0.7, 0.4, 0.1, 0.5],
        &[3, 3],
    ));
    let ll = f.eigenvalues.tanh().mul(&wl).unwrap().sum(None).unwrap();
    let lv = f
        .eigenvectors
        .mul(&f.eigenvectors)
        .unwrap()
        .mul(&wv)
        .unwrap()
        .sum(None)
        .unwrap();
    ll.add(&lv).unwrap()
}

#[test]
fn eigh_create_graph_via_facade() {
    let x0 = [4.0, 0.3, 0.2, 0.3, 2.5, 0.1, 0.2, 0.1, 1.0];
    check_create_graph_family("eigh", &x0, &[3, 3], eigh_loss, false, 0.01);
}

/// 固有値の縮退は（固有値だけの損失でも）型付きエラーで拒否され、`child` は空のまま。
#[test]
fn eigh_degenerate_is_rejected_via_facade() {
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let loss = x.eigh().unwrap().eigenvalues.sum(None).unwrap();
    let r = tape.backward_create_graph(&loss, &child);
    assert!(matches!(r, Err(AutodiffError::Backward(_))));
    assert_eq!(child.leaf_count(), 0, "拒否後は child が空");
}

// --- 正規化の S0 前スケーリング（イシュー #3809・決定記録 §45・§69） ----------------------------

/// 受理された `x` の 1 階（`first_order` と `cg.grad`）・二階（`child.backward`）がすべて有限で、
/// `cg.grad` が素の `backward` と REQ-2 判定で一致することを確かめる。
fn assert_prescaled_accepted<'a>(
    ctx: &str,
    tape: &'a Tape,
    child: &'a Tape,
    x: &Var<'a>,
    loss: &Var<'a>,
) {
    let plain = tape.backward(loss).unwrap();
    let plain_x = plain
        .get(x)
        .unwrap()
        .unwrap()
        .contiguous()
        .as_slice()
        .unwrap()
        .to_vec();
    let cg = tape
        .backward_create_graph(loss, child)
        .unwrap_or_else(|e| panic!("{ctx}: 受理されるはず {e:?}"));
    let g = cg.grad(x).unwrap().unwrap();
    let gv = g.to_tensor().contiguous().as_slice().unwrap().to_vec();
    assert!(gv.iter().all(|v| v.is_finite()), "{ctx}: cg.grad {gv:?}");
    for (a, b) in gv.iter().zip(&plain_x) {
        assert!(req2_close(*a as f64, *b as f64), "{ctx}: {a} vs {b}");
    }
    let grads2 = child.backward(&g.sum(None).unwrap()).unwrap();
    let xc = cg.child_var(x).unwrap().unwrap();
    let h = grads2
        .get(&xc)
        .unwrap()
        .unwrap()
        .contiguous()
        .as_slice()
        .unwrap()
        .to_vec();
    assert!(h.iter().all(|v| v.is_finite()), "{ctx}: 二階 {h:?}");
}

/// `[3e38, -3e38]`（入口の (b)）と `[-2e38, 1e38, 1e38]`（(c)）の `layer_norm` は S0 で受理される。
/// `eps=0` の定数行は拒否のままで、`child` は空に保たれる。
#[test]
fn layer_norm_prescaled_create_graph_via_facade() {
    for (row, n) in [
        (vec![3e38_f32, -3e38], 2usize),
        (vec![-2e38, 1e38, 1e38], 3),
    ] {
        let tape = fandhe_ai::tape();
        let child = fandhe_ai::tape();
        let x = tape.var(&t(row, &[1, n]));
        let w = tape.var_no_grad(&t(vec![1.2, 0.7, -0.4][..n].to_vec(), &[n]));
        let c = tape.var_no_grad(&t((0..n).map(|i| 1.0 + i as f32).collect(), &[1, n]));
        let loss = x
            .layer_norm(Some(&w), None, 1e-5)
            .unwrap()
            .mul(&c)
            .unwrap()
            .sum(None)
            .unwrap();
        assert_prescaled_accepted("layer_norm", &tape, &child, &x, &loss);
    }
    // 拒否が残る入力。
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(vec![2.0, 2.0, 2.0], &[1, 3]));
    let loss = x.layer_norm(None, None, 0.0).unwrap().sum(None).unwrap();
    assert!(matches!(
        tape.backward_create_graph(&loss, &child),
        Err(AutodiffError::Backward(_))
    ));
    assert_eq!(child.leaf_count(), 0, "拒否後は child が空");
}

/// `batch_norm`（train）のチャネル行 `[3e38, -3e38]` は S0 で受理される。上流勾配が巨大
/// （`g = 1e38`）な例は構造段の判定で拒否のままで、`child` は空に保たれる。
#[test]
fn batch_norm_prescaled_create_graph_via_facade() {
    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(vec![3e38_f32, 1.0, -3e38, 2.0], &[2, 2]));
    let w = tape.var_no_grad(&t(vec![1.2, 0.7], &[2]));
    let c = tape.var_no_grad(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let loss = x
        .batch_norm(Some(&w), None, 1e-5)
        .unwrap()
        .mul(&c)
        .unwrap()
        .sum(None)
        .unwrap();
    assert_prescaled_accepted("batch_norm", &tape, &child, &x, &loss);

    let tape = fandhe_ai::tape();
    let child = fandhe_ai::tape();
    let x = tape.var(&t(vec![3e38_f32, -3e38], &[2, 1]));
    let w = tape.var(&t(vec![1.0], &[1]));
    let c = tape.var_no_grad(&t(vec![1e38, 1e38], &[2, 1]));
    let loss = x
        .batch_norm(Some(&w), None, 1e-5)
        .unwrap()
        .mul(&c)
        .unwrap()
        .sum(None)
        .unwrap();
    assert!(matches!(
        tape.backward_create_graph(&loss, &child),
        Err(AutodiffError::Backward(_))
    ));
    assert_eq!(child.leaf_count(), 0, "拒否後は child が空");
}
