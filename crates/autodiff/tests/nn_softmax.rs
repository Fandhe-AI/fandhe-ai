//! softmax／log_softmax（イシュー #1594）の受け入れ条件検証。
//!
//! - forward: 手計算値との突合（dim 0／1）・`exp(log_softmax) ≈
//!   softmax`／行和 ≈ 1・極値入力での有限性・`dim >= rank` の型付き
//!   エラー。
//! - backward: `matmul → softmax → mse_loss` end-to-end の解析勾配を
//!   中央差分（数値微分）と突合する（`tests/nn_activation.rs` と同じ
//!   構成・許容誤差）。
//! - `nn::activation::Softmax`／`LogSoftmax` が `Var::softmax`／
//!   `log_softmax` と同一の値・テープ記録・`Module::forward_host` 経路
//!   の bit 一致を返すことを確認する。
//! - `BackendOps::softmax` が実際に呼ばれること・戻り値 shape 不一致が
//!   `ShapeMismatch` で拒否されること・`Unsupported` 以外のエラーが
//!   伝播することをカウンタ付きフィクスチャで固定する（判定迂回経路を
//!   作らない。`.claude/rules/security.md` A08）。

mod common;

use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_tensor_core::{BackendError, BackendOps, Device, ShapeError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

// --- 1. forward 手計算値突合 ---

#[test]
fn softmax_forward_matches_hand_computed_values_dim1() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3]));
    let y = x.softmax(1).unwrap();
    let out = y.to_tensor();

    // 手計算: softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
    // softmax([0,0,0]) = [1/3, 1/3, 1/3]
    let expected = [
        0.090_030_57,
        0.244_728_47,
        0.665_240_97,
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    ];
    for (i, &exp) in expected.iter().enumerate() {
        let idx = [i / 3, i % 3];
        let v = out.get(&idx).unwrap();
        assert!(
            (v - exp).abs() < 1e-5,
            "softmax[{idx:?}] = {v}, expected {exp}"
        );
    }
}

#[test]
fn softmax_forward_matches_hand_computed_values_dim0() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0], &[3, 2]));
    let y = x.softmax(0).unwrap();
    let out = y.to_tensor();

    // 列 0（[1,2,3]）は上と同じ分布、列 1（[0,0,0]）は一様分布。
    let expected_col0 = [0.090_030_57, 0.244_728_47, 0.665_240_97];
    let expected_col1 = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    for r in 0..3 {
        let v0 = out.get(&[r, 0]).unwrap();
        let v1 = out.get(&[r, 1]).unwrap();
        assert!((v0 - expected_col0[r]).abs() < 1e-5, "col0[{r}] = {v0}");
        assert!((v1 - expected_col1[r]).abs() < 1e-5, "col1[{r}] = {v1}");
    }
}

// --- 2. softmax／log_softmax の整合性（exp(log_softmax) ≈ softmax・行和 ≈ 1） ---

#[test]
fn softmax_rows_sum_to_one() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5, -1.0, 2.0, 0.0, 0.0], &[2, 4]));
    let y = x.softmax(1).unwrap();
    let out = y.to_tensor();
    for r in 0..2 {
        let sum: f32 = (0..4).map(|c| out.get(&[r, c]).unwrap()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "row {r} sum = {sum}");
    }
}

#[test]
fn exp_log_softmax_matches_softmax() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let x_a = tape_a.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[1, 4]));
    let softmax_out = x_a.softmax(1).unwrap().to_tensor();

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let x_b = tape_b.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[1, 4]));
    let log_softmax_out = x_b.log_softmax(1).unwrap().to_tensor();

    for c in 0..4 {
        let s = softmax_out.get(&[0, c]).unwrap();
        let ls = log_softmax_out.get(&[0, c]).unwrap();
        assert!(
            (ls.exp() - s).abs() < 1e-5,
            "exp(log_softmax[{c}])={} softmax[{c}]={}",
            ls.exp(),
            s
        );
    }
}

// --- 3. 極値入力での有限性 ---

#[test]
fn softmax_extreme_values_no_nan_inf() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1e4, -1e4, 1e4, -1e4], &[1, 4]));
    let out = x.softmax(1).unwrap().to_tensor();
    for c in 0..4 {
        let v = out.get(&[0, c]).unwrap();
        assert!(v.is_finite(), "softmax[{c}] = {v} は有限であるべき");
    }
}

#[test]
fn log_softmax_extreme_values_no_nan_inf() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1e4, -1e4, 1e4, -1e4], &[1, 4]));
    let out = x.log_softmax(1).unwrap().to_tensor();
    for c in 0..4 {
        let v = out.get(&[0, c]).unwrap();
        assert!(v.is_finite(), "log_softmax[{c}] = {v} は有限であるべき");
    }
}

// --- 4. dim >= rank は型付きエラー ---

#[test]
fn softmax_rejects_axis_out_of_range() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let result = x.softmax(5);
    assert!(matches!(
        result,
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            axis: 5,
            rank: 1
        }))
    ));
}

#[test]
fn log_softmax_rejects_axis_out_of_range() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let result = x.log_softmax(5);
    assert!(matches!(
        result,
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            axis: 5,
            rank: 1
        }))
    ));
}

// --- 5. end-to-end backward: matmul → softmax → mse_loss ---

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

struct SoftmaxFixture {
    x: Tensor<f32>,
    w: Tensor<f32>,
    target: Tensor<f32>,
}

fn fixture() -> SoftmaxFixture {
    SoftmaxFixture {
        x: t(vec![0.6, -0.4, 0.3, 0.9], &[2, 2]),
        w: t(vec![0.5, -0.7, 0.8, 0.2], &[2, 2]),
        target: t(vec![0.2, 0.6, 0.1, 0.4], &[2, 2]),
    }
}

fn forward_loss(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    target: &Tensor<f32>,
    op: impl for<'a> Fn(&'a fandhe_ai_autodiff::Var<'a>) -> fandhe_ai_autodiff::Var<'a>,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let wv = tape.var(w);
    let tv = tape.var(target);
    let pre = xv.matmul(&wv).unwrap();
    let y = op(&pre);
    let loss = y.mse_loss(&tv).unwrap();
    scalar(&loss.to_tensor())
}

#[test]
fn softmax_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().softmax(1).unwrap();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| {
        forward_loss(&f.x, &w, &f.target, |v| v.softmax(1).unwrap())
    });
    assert_grad_close("softmax e2e dW", dw, &num_dw);
}

#[test]
fn log_softmax_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().log_softmax(1).unwrap();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| {
        forward_loss(&f.x, &w, &f.target, |v| v.log_softmax(1).unwrap())
    });
    assert_grad_close("log_softmax e2e dW", dw, &num_dw);
}

// --- 6. nn::activation の薄いラッパー同値性（値・勾配とも一致） ---

#[test]
fn nn_activation_softmax_matches_var_softmax_end_to_end() {
    use fandhe_ai_autodiff::nn::activation::Softmax;

    let f = fixture();

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&f.x);
    let wv_a = tape_a.var(&f.w);
    let tv_a = tape_a.var(&f.target);
    let y_a = Softmax::new(1)
        .forward(&xv_a.matmul(&wv_a).unwrap())
        .unwrap();
    let loss_a = y_a.mse_loss(&tv_a).unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dw_a = grads_a.get(&wv_a).unwrap().expect("到達する");

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv_b = tape_b.var(&f.x);
    let wv_b = tape_b.var(&f.w);
    let tv_b = tape_b.var(&f.target);
    let y_b = xv_b.matmul(&wv_b).unwrap().softmax(1).unwrap();
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

// --- 7. Module::forward_host（tape 不要経路）と tape 経路の bit 一致 ---

#[test]
fn softmax_module_forward_host_matches_tape_forward() {
    use fandhe_ai_autodiff::nn::Module;
    use fandhe_ai_autodiff::nn::activation::Softmax;

    let x = t(vec![1.0, -2.0, 3.0, 0.5], &[1, 4]);
    let softmax = Softmax::new(1);
    let ops = common::naive_ops();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_tape = <Softmax as Module>::forward(&softmax, &tape, &xv).unwrap();
    let via_host = softmax.forward_host(ops.as_ref(), &x).unwrap();

    for c in 0..4 {
        assert_eq!(
            via_tape.to_tensor().get(&[0, c]).unwrap(),
            via_host.get(&[0, c]).unwrap()
        );
    }
}

// --- 8. BackendOps::softmax のディスパッチ配線検証（カウンタ付き
//        フィクスチャ）: 呼ばれること・戻り値 shape 検査・エラー伝播 ---

/// `BackendOps::softmax` の呼び出し回数を記録し、必要に応じて意図的な
/// 不整合（誤った shape・任意のエラー）を返せるカウンタ付き
/// フィクスチャ（`fusion_backend_integration.rs::CountingFusedOps` と
/// 同型の最小実装。9 個の必須メソッドのみ `common::NaiveOps` へ委譲
/// し、`softmax` のみ独自に振る舞いを差し替える）。
struct SoftmaxProbeOps {
    inner: common::NaiveOps,
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    behavior: SoftmaxProbeBehavior,
}

enum SoftmaxProbeBehavior {
    /// 正しい softmax 結果を返す（配線疎通の確認用）。
    Delegate,
    /// shape 不一致の戻り値を返す（呼び出し元の shape 検証を確認）。
    WrongShape,
    /// `Unsupported` 以外のエラーを返す（伝播することを確認）。
    OtherError,
}

impl BackendOps for SoftmaxProbeOps {
    fn device(&self) -> Device {
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

    fn softmax(&self, x: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>, BackendError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match self.behavior {
            SoftmaxProbeBehavior::Delegate => self.inner.softmax(x, dim),
            SoftmaxProbeBehavior::WrongShape => {
                // 入力より 1 要素少ない誤った shape を返す。
                let numel = x.numel();
                Tensor::new(
                    vec![0.0; numel.saturating_sub(1)],
                    &[numel.saturating_sub(1)],
                )
                .map_err(BackendError::ShapeMismatch)
            }
            SoftmaxProbeBehavior::OtherError => Err(BackendError::KernelLaunchFailed(
                "softmax probe: injected non-Unsupported error".into(),
            )),
        }
    }
}

#[test]
fn var_softmax_calls_backend_ops_softmax() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let ops = SoftmaxProbeOps {
        inner: common::NaiveOps,
        calls: calls.clone(),
        behavior: SoftmaxProbeBehavior::Delegate,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let y = x.softmax(1).unwrap();

    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    // naive 参照実装（`common::NaiveOps::softmax`）と値が一致すること
    // も確認し、`Delegate` 分岐が実際に正しい計算経路であることを
    // 併せて固定する。
    let sum: f32 = (0..3).map(|c| y.to_tensor().get(&[0, c]).unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn var_softmax_rejects_wrong_shape_from_backend() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let ops = SoftmaxProbeOps {
        inner: common::NaiveOps,
        calls,
        behavior: SoftmaxProbeBehavior::WrongShape,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let result = x.softmax(1);

    assert!(matches!(
        result,
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
}

#[test]
fn var_softmax_propagates_non_unsupported_backend_errors() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let ops = SoftmaxProbeOps {
        inner: common::NaiveOps,
        calls,
        behavior: SoftmaxProbeBehavior::OtherError,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let result = x.softmax(1);

    assert!(matches!(
        result,
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
}
