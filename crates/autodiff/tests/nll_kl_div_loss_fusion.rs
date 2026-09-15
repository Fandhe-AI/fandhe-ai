//! `Var::nll_loss`／`kl_div_loss`・`grad::vjp` の `Op::NllLoss`／
//! `Op::KlDivLoss` 分岐が、`BackendOps::nll_loss`／`nll_loss_backward`・
//! `kl_div_loss`／`kl_div_loss_backward`（イシュー #1738・親イシュー
//! #1609「損失関数の拡張」）の融合カーネルへ透過的に切り替わることの
//! 直接検証（`mse_loss_fusion.rs`・`bce_loss_fusion.rs` と同型のカウンタ
//! 付きフィクスチャ方式）。
//!
//! `autodiff` は具体バックエンドクレート（`backend-cpu` 等）へ依存しない
//! 設計上の不変条件（`docs/fusion-graph-design.md` §3.4）があるため、
//! `backend-cpu` を dev-dependency に追加せず、`common::NaiveOps` を
//! 委譲先として対象メソッドのみをオーバーライドするフィクスチャで代替
//! する。
//!
//! 4 点を固定する（NllLoss・KlDivLoss それぞれについて）:
//! 1. 融合カーネル実装ありの `BackendOps` では forward/backward とも
//!    実際にそのメソッドが呼ばれ（`Arc<AtomicUsize>` カウンタ）、
//!    フォールバック経路（`eval::nll_loss`／`kl_div_loss`）と数値一致
//!    複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で一致する。
//! 2. 常時 `Unsupported` を返すフィクスチャ（融合カーネル未実装
//!    バックエンド相当）ではフォールバックが働き、解析値と一致する。
//! 3. `Unsupported` 以外のエラー（融合カーネルの実行時失敗を模す）は
//!    フォールバックせずそのまま伝播する（判定迂回経路を作らない。
//!    `.claude/rules/security.md` A08）。forward・backward 双方で確認する。

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::{AutodiffError, Reduction, Tape};
use fandhe_ai_tensor_core::{BackendError, BackendOps, Device, KlDivTarget, MseReduction, Tensor};

/// 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
/// `.claude/rules/coding-rust.md`）。`mse_loss_fusion.rs::assert_close`
/// と同一の判定式をテストローカルに再実装する（dev-dependency 追加を
/// 避けるための既存方針を踏襲）。
fn assert_close(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

// =====================================================================
// NllLoss
// =====================================================================

/// `common::NaiveOps` に委譲しつつ `nll_loss`／`nll_loss_backward` のみ
/// を素朴な参照実装でオーバーライドし、呼び出し回数を記録する
/// フィクスチャ（`mse_loss_fusion.rs::CountingMseOps` と同型）。
struct CountingNllOps {
    inner: Box<dyn BackendOps + Send>,
    forward_calls: Arc<AtomicUsize>,
    backward_calls: Arc<AtomicUsize>,
}

impl BackendOps for CountingNllOps {
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

    fn nll_loss(
        &self,
        input: &Tensor<f32>,
        targets: &Tensor<i32>,
        class_dim: usize,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        self.forward_calls.fetch_add(1, Ordering::SeqCst);
        let shape = input.shape().to_vec();
        let outer: usize = shape[..class_dim].iter().product();
        let axis_len = shape[class_dim];
        let inner: usize = shape[class_dim + 1..].iter().product();
        let data = input.contiguous();
        let data = data.as_slice().unwrap_or(&[]);
        let targets_c = targets.contiguous();
        let target_data = targets_c.as_slice().unwrap_or(&[]);
        let n = outer * inner;
        let mut total = 0f32;
        for o in 0..outer {
            for i in 0..inner {
                let tgt = target_data[o * inner + i] as usize;
                total -= data[(o * axis_len + tgt) * inner + i];
            }
        }
        let value = match reduction {
            MseReduction::Mean if n > 0 => total / n as f32,
            MseReduction::Mean => 0.0,
            MseReduction::Sum => total,
            _ => total,
        };
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    fn nll_loss_backward(
        &self,
        input_shape: &[usize],
        targets: &Tensor<i32>,
        class_dim: usize,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        self.backward_calls.fetch_add(1, Ordering::SeqCst);
        let outer: usize = input_shape[..class_dim].iter().product();
        let axis_len = input_shape[class_dim];
        let inner: usize = input_shape[class_dim + 1..].iter().product();
        let numel: usize = input_shape.iter().product();
        let mut grad = vec![0f32; numel];
        let targets_c = targets.contiguous();
        let target_data = targets_c.as_slice().unwrap_or(&[]);
        for o in 0..outer {
            for i in 0..inner {
                let tgt = target_data[o * inner + i] as usize;
                grad[(o * axis_len + tgt) * inner + i] = -scale;
            }
        }
        Tensor::new(grad, input_shape).map_err(BackendError::ShapeMismatch)
    }
}

/// 常時 `Unsupported` を返すフィクスチャ（`nll_loss`／`nll_loss_backward`
/// を何もオーバーライドせず既定実装のまま使う）。
struct AlwaysUnsupportedNllOps {
    inner: Box<dyn BackendOps + Send>,
}

impl BackendOps for AlwaysUnsupportedNllOps {
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
}

/// `Op::NllLoss` の forward・backward の一方で `Unsupported` 以外の
/// エラーを常に返すフィクスチャ（`mse_loss_fusion.rs::
/// AlwaysFailingMseOps` と同型）。
struct AlwaysFailingNllOps {
    inner: Box<dyn BackendOps + Send>,
    fail_forward: bool,
}

impl BackendOps for AlwaysFailingNllOps {
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
    fn nll_loss(
        &self,
        _input: &Tensor<f32>,
        _targets: &Tensor<i32>,
        _class_dim: usize,
        _reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        if self.fail_forward {
            Err(BackendError::KernelLaunchFailed(
                "AlwaysFailingNllOps: simulated fused NLL forward failure (not Unsupported)".into(),
            ))
        } else {
            Err(BackendError::Unsupported("forward not under test".into()))
        }
    }
    fn nll_loss_backward(
        &self,
        _input_shape: &[usize],
        _targets: &Tensor<i32>,
        _class_dim: usize,
        _scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        if self.fail_forward {
            Err(BackendError::Unsupported("backward not under test".into()))
        } else {
            Err(BackendError::KernelLaunchFailed(
                "AlwaysFailingNllOps: simulated fused NLL backward failure (not Unsupported)"
                    .into(),
            ))
        }
    }
}

#[test]
fn fused_nll_loss_is_invoked_and_matches_fallback() {
    let input_data = vec![-0.1, -2.0, -1.5, -0.3];
    let targets = ti(vec![0, 1], &[2]);

    for reduction in [Reduction::Mean, Reduction::Sum] {
        let forward_calls = Arc::new(AtomicUsize::new(0));
        let backward_calls = Arc::new(AtomicUsize::new(0));
        let fused_ops = CountingNllOps {
            inner: common::naive_ops(),
            forward_calls: forward_calls.clone(),
            backward_calls: backward_calls.clone(),
        };
        let tape_fused = Tape::new_with_ops(Box::new(fused_ops));
        let input_fused = tape_fused.var(&t(input_data.clone(), &[2, 2]));
        let loss_fused = input_fused.nll_loss(&targets, 1, reduction).unwrap();
        assert_eq!(
            forward_calls.load(Ordering::SeqCst),
            1,
            "{reduction:?}: nll_loss は forward で 1 回呼ばれる契約"
        );
        let value_fused = scalar(&loss_fused.to_tensor());
        let grads_fused = tape_fused.backward(&loss_fused).unwrap();
        assert_eq!(
            backward_calls.load(Ordering::SeqCst),
            1,
            "{reduction:?}: nll_loss_backward は backward で 1 回呼ばれる契約"
        );
        let dinput_fused = grads_fused
            .get(&input_fused)
            .unwrap()
            .expect("input は loss に到達する")
            .clone();

        let tape_fallback = Tape::new_with_ops(common::naive_ops());
        let input_fallback = tape_fallback.var(&t(input_data.clone(), &[2, 2]));
        let loss_fallback = input_fallback.nll_loss(&targets, 1, reduction).unwrap();
        let value_fallback = scalar(&loss_fallback.to_tensor());
        let grads_fallback = tape_fallback.backward(&loss_fallback).unwrap();
        let dinput_fallback = grads_fallback
            .get(&input_fallback)
            .unwrap()
            .expect("input は loss に到達する");

        assert_close(
            value_fused,
            value_fallback,
            &format!("nll_loss forward ({reduction:?})"),
        );
        for i in 0..2 {
            for j in 0..2 {
                let a = dinput_fused.get(&[i, j]).unwrap();
                let e = dinput_fallback.get(&[i, j]).unwrap();
                assert_close(a, e, &format!("nll_loss backward ({reduction:?})[{i},{j}]"));
            }
        }
    }
}

#[test]
fn fused_nll_loss_forward_falls_back_when_unsupported() {
    let ops = AlwaysUnsupportedNllOps {
        inner: common::naive_ops(),
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let input = tape.var(&t(vec![-0.1, -2.0, -1.5, -0.3], &[2, 2]));
    let targets = ti(vec![0, 1], &[2]);
    let loss = input.nll_loss(&targets, 1, Reduction::Mean).unwrap();
    // フォールバック（`eval::nll_loss`）の解析値:
    // −(input[0,0] + input[1,1]) / 2 = −(−0.1 + −0.3) / 2 = 0.2。
    assert_close(scalar(&loss.to_tensor()), 0.2, "fallback forward mean");

    let grads = tape.backward(&loss).unwrap();
    let dinput = grads.get(&input).unwrap().expect("到達する");
    // scale = 1/2 = 0.5 → dInput はターゲット位置のみ −0.5、他は 0。
    let expected = [-0.5f32, 0.0, 0.0, -0.5];
    for (idx, &e) in expected.iter().enumerate() {
        let i = idx / 2;
        let j = idx % 2;
        assert_close(
            dinput.get(&[i, j]).unwrap(),
            e,
            &format!("fallback backward [{i},{j}]"),
        );
    }
}

#[test]
fn fused_nll_loss_forward_error_other_than_unsupported_propagates() {
    let ops = AlwaysFailingNllOps {
        inner: common::naive_ops(),
        fail_forward: true,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let input = tape.var(&t(vec![-0.1, -2.0], &[1, 2]));
    let targets = ti(vec![0], &[1]);

    let result = input.nll_loss(&targets, 1, Reduction::Mean);
    assert!(
        matches!(
            result,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ),
        "expected KernelLaunchFailed to propagate without fallback, got {result:?}"
    );
}

#[test]
fn fused_nll_loss_backward_error_other_than_unsupported_propagates() {
    let ops = AlwaysFailingNllOps {
        inner: common::naive_ops(),
        fail_forward: false,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let input = tape.var(&t(vec![-0.1, -2.0], &[1, 2]));
    let targets = ti(vec![0], &[1]);
    // forward は `Unsupported` を返す設定なので `eval::nll_loss` へ
    // フォールバックして成功する。
    let loss = input.nll_loss(&targets, 1, Reduction::Mean).unwrap();

    let result = tape.backward(&loss);
    assert!(
        matches!(
            result,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ),
        "expected KernelLaunchFailed to propagate without fallback, got {result:?}"
    );
}

// =====================================================================
// KlDivLoss
// =====================================================================

/// `common::NaiveOps` に委譲しつつ `kl_div_loss`／`kl_div_loss_backward`
/// のみを素朴な参照実装でオーバーライドし、呼び出し回数を記録する
/// フィクスチャ（`bce_loss_fusion.rs::CountingBceOps` と同型。dInput
/// のみ返し dTarget はホスト側で計算する `BackendOps` 契約に合わせる）。
struct CountingKlDivOps {
    inner: Box<dyn BackendOps + Send>,
    forward_calls: Arc<AtomicUsize>,
    backward_calls: Arc<AtomicUsize>,
}

impl BackendOps for CountingKlDivOps {
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

    fn kl_div_loss(
        &self,
        input: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: KlDivTarget,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        self.forward_calls.fetch_add(1, Ordering::SeqCst);
        let i = input.contiguous();
        let tt = target.contiguous();
        let id = i.as_slice().unwrap_or(&[]);
        let td = tt.as_slice().unwrap_or(&[]);
        let numel = id.len();
        let sum_loss: f32 = id
            .iter()
            .zip(td.iter())
            .map(|(&x, &tv)| match kind {
                KlDivTarget::Probabilities => {
                    if tv == 0.0 {
                        0.0
                    } else {
                        tv * (tv.ln() - x)
                    }
                }
                _ => tv.exp() * (tv - x),
            })
            .sum();
        let value = match reduction {
            MseReduction::Mean if numel > 0 => sum_loss / numel as f32,
            MseReduction::Mean => 0.0,
            MseReduction::Sum => sum_loss,
            _ => sum_loss,
        };
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    fn kl_div_loss_backward(
        &self,
        input: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: KlDivTarget,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        self.backward_calls.fetch_add(1, Ordering::SeqCst);
        let tt = target.contiguous();
        let td = tt.as_slice().unwrap_or(&[]);
        let dinput: Vec<f32> = td
            .iter()
            .map(|&tv| match kind {
                KlDivTarget::Probabilities => scale * -tv,
                _ => scale * -tv.exp(),
            })
            .collect();
        Tensor::new(dinput, input.shape()).map_err(BackendError::ShapeMismatch)
    }
}

/// 常時 `Unsupported` を返すフィクスチャ。
struct AlwaysUnsupportedKlDivOps {
    inner: Box<dyn BackendOps + Send>,
}

impl BackendOps for AlwaysUnsupportedKlDivOps {
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
}

/// `Op::KlDivLoss` の forward・backward の一方で `Unsupported` 以外の
/// エラーを常に返すフィクスチャ。
struct AlwaysFailingKlDivOps {
    inner: Box<dyn BackendOps + Send>,
    fail_forward: bool,
}

impl BackendOps for AlwaysFailingKlDivOps {
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
    fn kl_div_loss(
        &self,
        _input: &Tensor<f32>,
        _target: &Tensor<f32>,
        _kind: KlDivTarget,
        _reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        if self.fail_forward {
            Err(BackendError::KernelLaunchFailed(
                "AlwaysFailingKlDivOps: simulated fused KLDiv forward failure (not Unsupported)"
                    .into(),
            ))
        } else {
            Err(BackendError::Unsupported("forward not under test".into()))
        }
    }
    fn kl_div_loss_backward(
        &self,
        _input: &Tensor<f32>,
        _target: &Tensor<f32>,
        _kind: KlDivTarget,
        _scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        if self.fail_forward {
            Err(BackendError::Unsupported("backward not under test".into()))
        } else {
            Err(BackendError::KernelLaunchFailed(
                "AlwaysFailingKlDivOps: simulated fused KLDiv backward failure (not Unsupported)"
                    .into(),
            ))
        }
    }
}

#[test]
fn fused_kl_div_loss_is_invoked_and_matches_fallback() {
    let input_data = vec![-2.0, -0.5, -1.2, -0.1];
    let target_data = vec![0.2, 0.8, 0.5, 0.5];

    for reduction in [Reduction::Mean, Reduction::Sum] {
        let forward_calls = Arc::new(AtomicUsize::new(0));
        let backward_calls = Arc::new(AtomicUsize::new(0));
        let fused_ops = CountingKlDivOps {
            inner: common::naive_ops(),
            forward_calls: forward_calls.clone(),
            backward_calls: backward_calls.clone(),
        };
        let tape_fused = Tape::new_with_ops(Box::new(fused_ops));
        let input_fused = tape_fused.var(&t(input_data.clone(), &[2, 2]));
        let target_fused = tape_fused.var(&t(target_data.clone(), &[2, 2]));
        let loss_fused = input_fused.kl_div_loss(&target_fused, reduction).unwrap();
        assert_eq!(
            forward_calls.load(Ordering::SeqCst),
            1,
            "{reduction:?}: kl_div_loss は forward で 1 回呼ばれる契約"
        );
        let value_fused = scalar(&loss_fused.to_tensor());
        let grads_fused = tape_fused.backward(&loss_fused).unwrap();
        assert_eq!(
            backward_calls.load(Ordering::SeqCst),
            1,
            "{reduction:?}: kl_div_loss_backward は backward で 1 回呼ばれる契約"
        );
        let dinput_fused = grads_fused
            .get(&input_fused)
            .unwrap()
            .expect("input は loss に到達する")
            .clone();
        let dtarget_fused = grads_fused
            .get(&target_fused)
            .unwrap()
            .expect("target は loss に到達する")
            .clone();

        let tape_fallback = Tape::new_with_ops(common::naive_ops());
        let input_fallback = tape_fallback.var(&t(input_data.clone(), &[2, 2]));
        let target_fallback = tape_fallback.var(&t(target_data.clone(), &[2, 2]));
        let loss_fallback = input_fallback
            .kl_div_loss(&target_fallback, reduction)
            .unwrap();
        let value_fallback = scalar(&loss_fallback.to_tensor());
        let grads_fallback = tape_fallback.backward(&loss_fallback).unwrap();
        let dinput_fallback = grads_fallback
            .get(&input_fallback)
            .unwrap()
            .expect("input は loss に到達する");
        let dtarget_fallback = grads_fallback
            .get(&target_fallback)
            .unwrap()
            .expect("target は loss に到達する");

        assert_close(
            value_fused,
            value_fallback,
            &format!("kl_div_loss forward ({reduction:?})"),
        );
        for i in 0..2 {
            for j in 0..2 {
                let a = dinput_fused.get(&[i, j]).unwrap();
                let e = dinput_fallback.get(&[i, j]).unwrap();
                assert_close(
                    a,
                    e,
                    &format!("kl_div_loss backward dInput ({reduction:?})[{i},{j}]"),
                );
                let a = dtarget_fused.get(&[i, j]).unwrap();
                let e = dtarget_fallback.get(&[i, j]).unwrap();
                assert_close(
                    a,
                    e,
                    &format!("kl_div_loss backward dTarget ({reduction:?})[{i},{j}]"),
                );
            }
        }
    }
}

#[test]
fn fused_kl_div_loss_forward_falls_back_when_unsupported() {
    let ops = AlwaysUnsupportedKlDivOps {
        inner: common::naive_ops(),
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let input = tape.var(&t(vec![-2.0, -0.5, -1.2, -0.1], &[2, 2]));
    let target = tape.var(&t(vec![0.2, 0.8, 0.5, 0.5], &[2, 2]));
    let loss = input.kl_div_loss(&target, Reduction::Mean).unwrap();

    // フォールバック（`eval::kl_div_loss`）解析値を独立に手計算する。
    let expected_value = {
        let xs = [-2.0f32, -0.5, -1.2, -0.1];
        let ts = [0.2f32, 0.8, 0.5, 0.5];
        let sum: f32 = xs
            .iter()
            .zip(ts.iter())
            .map(|(&x, &tv)| if tv == 0.0 { 0.0 } else { tv * (tv.ln() - x) })
            .sum();
        sum / 4.0
    };
    assert_close(
        scalar(&loss.to_tensor()),
        expected_value,
        "fallback forward mean",
    );

    let grads = tape.backward(&loss).unwrap();
    let dinput = grads.get(&input).unwrap().expect("到達する");
    let dtarget = grads.get(&target).unwrap().expect("到達する");
    let ts = [0.2f32, 0.8, 0.5, 0.5];
    let xs = [-2.0f32, -0.5, -1.2, -0.1];
    for idx in 0..4 {
        let i = idx / 2;
        let j = idx % 2;
        let expected_dinput = -ts[idx] / 4.0;
        assert_close(
            dinput.get(&[i, j]).unwrap(),
            expected_dinput,
            &format!("fallback backward dInput [{i},{j}]"),
        );
        let expected_dtarget = if ts[idx] == 0.0 {
            0.0
        } else {
            (ts[idx].ln() + 1.0 - xs[idx]) / 4.0
        };
        assert_close(
            dtarget.get(&[i, j]).unwrap(),
            expected_dtarget,
            &format!("fallback backward dTarget [{i},{j}]"),
        );
    }
}

#[test]
fn fused_kl_div_loss_forward_error_other_than_unsupported_propagates() {
    let ops = AlwaysFailingKlDivOps {
        inner: common::naive_ops(),
        fail_forward: true,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let input = tape.var(&t(vec![-2.0, -0.5], &[2]));
    let target = tape.var(&t(vec![0.2, 0.8], &[2]));

    let result = input.kl_div_loss(&target, Reduction::Mean);
    assert!(
        matches!(
            result,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ),
        "expected KernelLaunchFailed to propagate without fallback, got {result:?}"
    );
}

#[test]
fn fused_kl_div_loss_backward_error_other_than_unsupported_propagates() {
    let ops = AlwaysFailingKlDivOps {
        inner: common::naive_ops(),
        fail_forward: false,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let input = tape.var(&t(vec![-2.0, -0.5], &[2]));
    let target = tape.var(&t(vec![0.2, 0.8], &[2]));
    // forward は `Unsupported` を返す設定なので `eval::kl_div_loss` へ
    // フォールバックして成功する。
    let loss = input.kl_div_loss(&target, Reduction::Mean).unwrap();

    let result = tape.backward(&loss);
    assert!(
        matches!(
            result,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ),
        "expected KernelLaunchFailed to propagate without fallback, got {result:?}"
    );
}
