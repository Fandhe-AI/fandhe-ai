//! GELU（erf／tanh 近似）・Softplus（イシュー #1713）の受け入れ条件検証。
//!
//! `nn_softmax.rs`（イシュー #1594）と同じ構成方針:
//! - forward: 手計算値との突合（既知点）。
//! - backward: `matmul → 活性化 → mse_loss` end-to-end の解析勾配を
//!   中央差分（数値微分）と突合する。
//! - `nn::activation::Gelu`／`GeluTanh`／`Softplus` が `Var::gelu`／
//!   `gelu_tanh`／`softplus` と同一の値・テープ記録・`Module::
//!   forward_host` 経路の一致を返すことを確認する。
//! - `BackendOps::scalar_unary` が実際に呼ばれること・戻り値 shape
//!   不一致が `ShapeMismatch` で拒否されること・`Unsupported` 以外の
//!   エラーが伝播することをカウンタ付きフィクスチャで固定する
//!   （判定迂回経路を作らない。`.claude/rules/security.md` A08）。

mod common;

use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_tensor_core::{BackendError, BackendOps, ScalarUnaryOp, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

// --- 1. forward 手計算値突合 ---

#[test]
fn gelu_forward_matches_hand_computed_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0, 1.0, -1.0], &[3]));
    let y = x.gelu().unwrap();
    let out = y.to_tensor();
    // gelu(0) = 0・gelu(1) ≈ 0.8413447・gelu(-1) ≈ -0.1586553
    // （gelu は奇関数ではないが gelu(-1) = -1 - gelu(1) の関係）。
    assert!((out.get(&[0]).unwrap() - 0.0).abs() < 1e-6);
    assert!((out.get(&[1]).unwrap() - 0.841_344_7).abs() < 1e-5);
    assert!((out.get(&[2]).unwrap() - (-0.158_655_3)).abs() < 1e-5);
}

#[test]
fn gelu_tanh_forward_matches_hand_computed_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0], &[1]));
    let y = x.gelu_tanh().unwrap();
    let out = y.to_tensor();
    assert!((out.get(&[0]).unwrap() - 0.0).abs() < 1e-6);
}

#[test]
fn softplus_forward_matches_hand_computed_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0], &[1]));
    let y = x.softplus(1.0, 20.0).unwrap();
    let out = y.to_tensor();
    assert!((out.get(&[0]).unwrap() - std::f32::consts::LN_2).abs() < 1e-5);
}

// --- 2. dx >= 0 系の値域確認（極値入力での有限性） ---

#[test]
fn gelu_extreme_values_no_nan_inf() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1e4, -1e4], &[2]));
    let out = x.gelu().unwrap().to_tensor();
    assert!(out.get(&[0]).unwrap().is_finite());
    assert!(out.get(&[1]).unwrap().is_finite());
}

#[test]
fn softplus_extreme_values_no_nan_inf() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1e4, -1e4], &[2]));
    let out = x.softplus(1.0, 20.0).unwrap().to_tensor();
    assert!(out.get(&[0]).unwrap().is_finite());
    assert!(out.get(&[1]).unwrap().is_finite());
}

// --- 3. end-to-end backward: matmul → 活性化 → mse_loss ---

const H: f64 = 1e-3;
const TAU: f32 = 1e-4;
const REL_TOL: f32 = 1e-2;
const ABS_TOL: f32 = 1e-3;

fn assert_grad_close(label: &str, analytic: &Tensor<f32>, numeric: &Tensor<f32>) {
    assert_eq!(
        analytic.shape(),
        numeric.shape(),
        "{label}: shape が一致しない"
    );
    let shape = analytic.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut index = vec![0usize; shape.len()];
    for _ in 0..numel {
        let av = analytic.get(&index).unwrap_or(0.0);
        let nv = numeric.get(&index).unwrap_or(0.0);
        let diff = (av - nv).abs();
        let rel = diff / av.abs().max(nv.abs()).max(TAU);
        assert!(
            rel <= REL_TOL || diff <= ABS_TOL,
            "{label}[idx={index:?}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
        );
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

fn numeric_grad(target_tensor: &Tensor<f32>, perturb: impl Fn(Tensor<f32>) -> f32) -> Tensor<f32> {
    let shape = target_tensor.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut data: Vec<f32> = (0..numel)
        .map(|flat| {
            let mut idx = vec![0usize; shape.len()];
            let mut rem = flat;
            for axis in (0..shape.len()).rev() {
                idx[axis] = rem % shape[axis];
                rem /= shape[axis];
            }
            target_tensor.get(&idx).unwrap_or(0.0)
        })
        .collect();
    let mut grad = vec![0f32; numel];
    for i in 0..numel {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = perturb(t(data.clone(), &shape)) as f64;
        data[i] = (orig - H) as f32;
        let lm = perturb(t(data.clone(), &shape)) as f64;
        data[i] = orig as f32;
        grad[i] = ((lp - lm) / (2.0 * H)) as f32;
    }
    t(grad, &shape)
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

struct Fixture {
    x: Tensor<f32>,
    w: Tensor<f32>,
    target: Tensor<f32>,
}

fn fixture() -> Fixture {
    Fixture {
        x: t(vec![0.6, -0.4, 0.3, 0.9], &[2, 2]),
        w: t(vec![0.5, -0.7, 0.8, 0.2], &[2, 2]),
        target: t(vec![0.2, 0.6, 0.1, 0.4], &[2, 2]),
    }
}

fn forward_loss(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    target: &Tensor<f32>,
    op: impl for<'a> Fn(
        &'a fandhe_ai_autodiff::Var<'a>,
    ) -> Result<fandhe_ai_autodiff::Var<'a>, AutodiffError>,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let wv = tape.var(w);
    let tv = tape.var(target);
    let pre = xv.matmul(&wv).unwrap();
    let y = op(&pre).unwrap();
    let loss = y.mse_loss(&tv).unwrap();
    scalar(&loss.to_tensor())
}

#[test]
fn gelu_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().gelu().unwrap();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| forward_loss(&f.x, &w, &f.target, |v| v.gelu()));
    assert_grad_close("gelu e2e dW", dw, &num_dw);
}

#[test]
fn gelu_tanh_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().gelu_tanh().unwrap();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| {
        forward_loss(&f.x, &w, &f.target, |v| v.gelu_tanh())
    });
    assert_grad_close("gelu_tanh e2e dW", dw, &num_dw);
}

#[test]
fn softplus_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().softplus(1.0, 20.0).unwrap();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| {
        forward_loss(&f.x, &w, &f.target, |v| v.softplus(1.0, 20.0))
    });
    assert_grad_close("softplus e2e dW", dw, &num_dw);
}

// --- 4. nn::activation の薄いラッパー同値性（値・勾配とも一致） ---

#[test]
fn nn_activation_gelu_matches_var_gelu_end_to_end() {
    use fandhe_ai_autodiff::nn::activation::Gelu;

    let f = fixture();

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&f.x);
    let wv_a = tape_a.var(&f.w);
    let tv_a = tape_a.var(&f.target);
    let y_a = Gelu.forward(&xv_a.matmul(&wv_a).unwrap()).unwrap();
    let loss_a = y_a.mse_loss(&tv_a).unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dw_a = grads_a.get(&wv_a).unwrap().expect("到達する");

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv_b = tape_b.var(&f.x);
    let wv_b = tape_b.var(&f.w);
    let tv_b = tape_b.var(&f.target);
    let y_b = xv_b.matmul(&wv_b).unwrap().gelu().unwrap();
    let loss_b = y_b.mse_loss(&tv_b).unwrap();
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let dw_b = grads_b.get(&wv_b).unwrap().expect("到達する");

    assert_eq!(loss_a.to_tensor().get(&[]), loss_b.to_tensor().get(&[]));
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(dw_a.get(&[i, j]).unwrap(), dw_b.get(&[i, j]).unwrap());
        }
    }
}

#[test]
fn nn_activation_softplus_matches_var_softplus_end_to_end() {
    use fandhe_ai_autodiff::nn::activation::Softplus;

    let f = fixture();
    let sp = Softplus::new(1.0, 20.0).unwrap();

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&f.x);
    let wv_a = tape_a.var(&f.w);
    let tv_a = tape_a.var(&f.target);
    let y_a = sp.forward(&xv_a.matmul(&wv_a).unwrap()).unwrap();
    let loss_a = y_a.mse_loss(&tv_a).unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dw_a = grads_a.get(&wv_a).unwrap().expect("到達する");

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv_b = tape_b.var(&f.x);
    let wv_b = tape_b.var(&f.w);
    let tv_b = tape_b.var(&f.target);
    let y_b = xv_b.matmul(&wv_b).unwrap().softplus(1.0, 20.0).unwrap();
    let loss_b = y_b.mse_loss(&tv_b).unwrap();
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let dw_b = grads_b.get(&wv_b).unwrap().expect("到達する");

    assert_eq!(loss_a.to_tensor().get(&[]), loss_b.to_tensor().get(&[]));
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(dw_a.get(&[i, j]).unwrap(), dw_b.get(&[i, j]).unwrap());
        }
    }
}

// --- 5. Module::forward_host（tape 不要経路）と tape 経路の bit 一致 ---

#[test]
fn gelu_module_forward_host_matches_tape_forward() {
    use fandhe_ai_autodiff::nn::Module;
    use fandhe_ai_autodiff::nn::activation::Gelu;

    let x = t(vec![1.0, -2.0, 3.0, 0.5], &[1, 4]);
    let ops = common::naive_ops();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_tape = <Gelu as Module>::forward(&Gelu, &tape, &xv).unwrap();
    let via_host = Gelu.forward_host(ops.as_ref(), &x).unwrap();

    for c in 0..4 {
        assert_eq!(
            via_tape.to_tensor().get(&[0, c]).unwrap(),
            via_host.get(&[0, c]).unwrap()
        );
    }
}

#[test]
fn softplus_module_forward_host_matches_tape_forward() {
    use fandhe_ai_autodiff::nn::Module;
    use fandhe_ai_autodiff::nn::activation::Softplus;

    let x = t(vec![1.0, -2.0, 3.0, 0.5], &[1, 4]);
    let sp = Softplus::new(1.0, 20.0).unwrap();
    let ops = common::naive_ops();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_tape = <Softplus as Module>::forward(&sp, &tape, &xv).unwrap();
    let via_host = sp.forward_host(ops.as_ref(), &x).unwrap();

    for c in 0..4 {
        assert_eq!(
            via_tape.to_tensor().get(&[0, c]).unwrap(),
            via_host.get(&[0, c]).unwrap()
        );
    }
}

// --- 6. BackendOps::scalar_unary のディスパッチ配線検証（カウンタ付き
//        フィクスチャ）: 呼ばれること・戻り値 shape 検査・エラー伝播 ---

/// `BackendOps::scalar_unary` の呼び出し回数を記録し、必要に応じて
/// 意図的な不整合（誤った shape・任意のエラー）を返せるカウンタ付き
/// フィクスチャ（`nn_softmax.rs::SoftmaxProbeOps` と同型の最小実装。
/// 9 個の必須メソッドのみ `common::NaiveOps` へ委譲し、`scalar_unary`
/// のみ独自に振る舞いを差し替える）。
struct ScalarUnaryProbeOps {
    inner: common::NaiveOps,
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    behavior: ScalarUnaryProbeBehavior,
}

enum ScalarUnaryProbeBehavior {
    /// 正しい GELU 結果を返す（配線疎通の確認用）。
    Delegate,
    /// shape 不一致の戻り値を返す（呼び出し元の shape 検証を確認）。
    WrongShape,
    /// `Unsupported` 以外のエラーを返す（伝播することを確認）。
    OtherError,
}

impl BackendOps for ScalarUnaryProbeOps {
    fn device(&self) -> fandhe_ai_tensor_core::Device {
        self.inner.device()
    }
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.gemm(a, b)
    }
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.add(a, b)
    }
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.mul(a, b)
    }
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.relu(a)
    }
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.exp(a)
    }
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.tanh(a)
    }
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.sum(a, dim)
    }
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.max(a, dim)
    }

    fn scalar_unary(
        &self,
        op: ScalarUnaryOp,
        a: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match self.behavior {
            ScalarUnaryProbeBehavior::Delegate => {
                // `ScalarUnaryOp::apply`（`scalar_op.rs` の forward 数式
                // 単一情報源）を要素ごとに適用する（`common::mod.rs::
                // unary` は private のためここで独立に同じ意味論を
                // 再実装する）。
                let dense = a.contiguous();
                let src = dense.as_slice().unwrap_or(&[]);
                let out: Vec<f32> = src.iter().map(|&v| op.apply(v)).collect();
                Tensor::new(out, a.shape()).map_err(BackendError::ShapeMismatch)
            }
            ScalarUnaryProbeBehavior::WrongShape => {
                let numel = a.numel();
                Tensor::new(
                    vec![0.0; numel.saturating_sub(1)],
                    &[numel.saturating_sub(1)],
                )
                .map_err(BackendError::ShapeMismatch)
            }
            ScalarUnaryProbeBehavior::OtherError => Err(BackendError::KernelLaunchFailed(
                "scalar_unary probe: injected non-Unsupported error".into(),
            )),
        }
    }
}

#[test]
fn var_gelu_calls_backend_ops_scalar_unary() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let ops = ScalarUnaryProbeOps {
        inner: common::NaiveOps,
        calls: calls.clone(),
        behavior: ScalarUnaryProbeBehavior::Delegate,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let x = tape.var(&t(vec![0.0, 1.0], &[2]));
    let y = x.gelu().unwrap();

    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!((y.to_tensor().get(&[0]).unwrap() - 0.0).abs() < 1e-6);
}

#[test]
fn var_gelu_rejects_wrong_shape_from_backend() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let ops = ScalarUnaryProbeOps {
        inner: common::NaiveOps,
        calls,
        behavior: ScalarUnaryProbeBehavior::WrongShape,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let x = tape.var(&t(vec![0.0, 1.0], &[2]));
    let result = x.gelu();

    assert!(matches!(
        result,
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
}

#[test]
fn var_gelu_propagates_non_unsupported_backend_errors() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let ops = ScalarUnaryProbeOps {
        inner: common::NaiveOps,
        calls,
        behavior: ScalarUnaryProbeBehavior::OtherError,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let x = tape.var(&t(vec![0.0, 1.0], &[2]));
    let result = x.gelu();

    assert!(matches!(
        result,
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
}
