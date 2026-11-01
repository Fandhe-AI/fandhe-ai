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

use fandhe_ai::nn::loss::{BceWithLogitsOptions, Reduction, SigmoidFocalLossOptions};
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
    check_create_graph_family("det", &INV_X, &[2, 2], det_loss, false, 1e-2);
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
