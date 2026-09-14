//! `Var::bce_loss`／`bce_with_logits_loss`／`grad::vjp` の `Op::BceLoss`
//! 分岐が、`BackendOps::bce_loss`／`bce_loss_backward`（イシュー #1737・
//! 親イシュー #1609）の融合カーネルへ透過的に切り替わることの直接検証
//! （`mse_loss_fusion.rs` と同型のカウンタ付きフィクスチャ方式）。
//!
//! `autodiff` は具体バックエンドクレート（`backend-cpu` 等）へ依存しない
//! 設計上の不変条件（`docs/fusion-graph-design.md` §3.4）があるため、
//! `backend-cpu` を dev-dependency に追加せず、`common::NaiveOps` を
//! 委譲先として `bce_loss`／`bce_loss_backward` のみをオーバーライドする
//! フィクスチャで代替する（`mse_loss_fusion.rs` の代替方針）。
//!
//! 4 点を固定する（`mse_loss_fusion.rs` と同型）:
//! 1. 融合カーネル実装ありの `BackendOps` では forward/backward とも
//!    実際にそのメソッドが呼ばれ（`Arc<AtomicUsize>` カウンタ）、
//!    フォールバック経路（`eval::bce_loss`／`bce_loss_vjp`）と数値一致
//!    複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で一致する。
//! 2. 常時 `Unsupported` を返すフィクスチャではフォールバックが働き、
//!    解析値と一致する。
//! 3. `Unsupported` 以外のエラーはフォールバックせずそのまま伝播する
//!    （判定迂回経路を作らない。`.claude/rules/security.md` A08）。
//!    forward・backward 双方で確認する。

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::{AutodiffError, Reduction, Tape};
use fandhe_ai_tensor_core::{BackendError, BackendOps, BceKind, Device, MseReduction, Tensor};

/// 統一複合判定（`mse_loss_fusion.rs::assert_close` と同一）。
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

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

fn bce_elem_loss(input: f32, target: f32, kind: BceKind) -> f32 {
    match kind {
        BceKind::Probabilities => {
            let log_p = input.ln().max(-100.0);
            let log_1mp = (1.0 - input).ln().max(-100.0);
            -(target * log_p + (1.0 - target) * log_1mp)
        }
        _ => input.max(0.0) - input * target + (-input.abs()).exp().ln_1p(),
    }
}

fn bce_elem_grad_input(input: f32, target: f32, kind: BceKind) -> f32 {
    match kind {
        BceKind::Probabilities => {
            let denom = (input * (1.0 - input)).max(1e-12);
            (input - target) / denom
        }
        _ => {
            let sigmoid = if input >= 0.0 {
                1.0 / (1.0 + (-input).exp())
            } else {
                let e = input.exp();
                e / (1.0 + e)
            };
            sigmoid - target
        }
    }
}

/// `common::NaiveOps` に委譲しつつ `bce_loss`／`bce_loss_backward` の
/// みを素朴な参照実装でオーバーライドし、呼び出し回数を記録する
/// フィクスチャ（`mse_loss_fusion.rs::CountingMseOps` と同型）。
struct CountingBceOps {
    inner: Box<dyn BackendOps + Send>,
    forward_calls: Arc<AtomicUsize>,
    backward_calls: Arc<AtomicUsize>,
}

impl BackendOps for CountingBceOps {
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

    fn bce_loss(
        &self,
        input: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: BceKind,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        self.forward_calls.fetch_add(1, Ordering::SeqCst);
        let i = input.contiguous();
        let tt = target.contiguous();
        let id = i.as_slice().unwrap_or(&[]);
        let td = tt.as_slice().unwrap_or(&[]);
        let numel = id.len();
        if numel == 0 {
            return Tensor::new(vec![0.0], &[]).map_err(BackendError::ShapeMismatch);
        }
        let sum: f32 = id
            .iter()
            .zip(td.iter())
            .map(|(&x, &y)| bce_elem_loss(x, y, kind))
            .sum();
        let value = match reduction {
            MseReduction::Mean => sum / numel as f32,
            MseReduction::Sum => sum,
            _ => sum,
        };
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    fn bce_loss_backward(
        &self,
        input: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: BceKind,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        self.backward_calls.fetch_add(1, Ordering::SeqCst);
        let i = input.contiguous();
        let tt = target.contiguous();
        let id = i.as_slice().unwrap_or(&[]);
        let td = tt.as_slice().unwrap_or(&[]);
        let dinput: Vec<f32> = id
            .iter()
            .zip(td.iter())
            .map(|(&x, &y)| scale * bce_elem_grad_input(x, y, kind))
            .collect();
        Tensor::new(dinput, input.shape()).map_err(BackendError::ShapeMismatch)
    }
}

/// 常時 `Unsupported` を返すフィクスチャ（`mse_loss_fusion.rs::
/// AlwaysUnsupportedMseOps` と同型）。
struct AlwaysUnsupportedBceOps {
    inner: Box<dyn BackendOps + Send>,
}

impl BackendOps for AlwaysUnsupportedBceOps {
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

/// `Op::BceLoss` の forward・backward の一方で `Unsupported` 以外の
/// エラーを常に返すフィクスチャ（`mse_loss_fusion.rs::
/// AlwaysFailingMseOps` と同型）。
struct AlwaysFailingBceOps {
    inner: Box<dyn BackendOps + Send>,
    fail_forward: bool,
}

impl BackendOps for AlwaysFailingBceOps {
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
    fn bce_loss(
        &self,
        _input: &Tensor<f32>,
        _target: &Tensor<f32>,
        _kind: BceKind,
        _reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        if self.fail_forward {
            Err(BackendError::KernelLaunchFailed(
                "AlwaysFailingBceOps: simulated fused BCE forward failure (not Unsupported)".into(),
            ))
        } else {
            Err(BackendError::Unsupported("forward not under test".into()))
        }
    }
    fn bce_loss_backward(
        &self,
        _input: &Tensor<f32>,
        _target: &Tensor<f32>,
        _kind: BceKind,
        _scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        if self.fail_forward {
            Err(BackendError::Unsupported("backward not under test".into()))
        } else {
            Err(BackendError::KernelLaunchFailed(
                "AlwaysFailingBceOps: simulated fused BCE backward failure (not Unsupported)"
                    .into(),
            ))
        }
    }
}

#[test]
fn fused_bce_loss_is_invoked_and_matches_fallback() {
    // `Probabilities` は `[0, 1]` 範囲検査があるため確率らしい値、
    // `Logits` は範囲制約なしの値を使う。
    for kind in [BceKind::Probabilities, BceKind::Logits] {
        let (input_data, target_data): (Vec<f32>, Vec<f32>) = match kind {
            BceKind::Probabilities => (vec![0.2, 0.8, 0.5, 0.9], vec![0.0, 1.0, 1.0, 0.0]),
            _ => (vec![-2.0, 1.5, 0.0, 3.0], vec![0.0, 1.0, 1.0, 0.0]),
        };

        for reduction in [Reduction::Mean, Reduction::Sum] {
            let forward_calls = Arc::new(AtomicUsize::new(0));
            let backward_calls = Arc::new(AtomicUsize::new(0));
            let fused_ops = CountingBceOps {
                inner: common::naive_ops(),
                forward_calls: forward_calls.clone(),
                backward_calls: backward_calls.clone(),
            };
            let tape_fused = Tape::new_with_ops(Box::new(fused_ops));
            let input_fused = tape_fused.var(&t(input_data.clone(), &[2, 2]));
            let target_fused = tape_fused.var(&t(target_data.clone(), &[2, 2]));
            let loss_fused = match kind {
                BceKind::Probabilities => input_fused.bce_loss(&target_fused, reduction).unwrap(),
                _ => input_fused
                    .bce_with_logits_loss(&target_fused, reduction)
                    .unwrap(),
            };
            assert_eq!(
                forward_calls.load(Ordering::SeqCst),
                1,
                "{kind:?}/{reduction:?}: bce_loss は forward で 1 回呼ばれる契約"
            );
            let value_fused = scalar(&loss_fused.to_tensor());
            let grads_fused = tape_fused.backward(&loss_fused).unwrap();
            assert_eq!(
                backward_calls.load(Ordering::SeqCst),
                1,
                "{kind:?}/{reduction:?}: bce_loss_backward は backward で 1 回呼ばれる契約"
            );
            let dinput_fused = grads_fused
                .get(&input_fused)
                .unwrap()
                .expect("input は loss に到達する")
                .clone();

            // フォールバック経路（`NaiveOps`。既定 `Unsupported`）との突合。
            let tape_fallback = Tape::new_with_ops(common::naive_ops());
            let input_fallback = tape_fallback.var(&t(input_data.clone(), &[2, 2]));
            let target_fallback = tape_fallback.var(&t(target_data.clone(), &[2, 2]));
            let loss_fallback = match kind {
                BceKind::Probabilities => input_fallback
                    .bce_loss(&target_fallback, reduction)
                    .unwrap(),
                _ => input_fallback
                    .bce_with_logits_loss(&target_fallback, reduction)
                    .unwrap(),
            };
            let value_fallback = scalar(&loss_fallback.to_tensor());
            let grads_fallback = tape_fallback.backward(&loss_fallback).unwrap();
            let dinput_fallback = grads_fallback
                .get(&input_fallback)
                .unwrap()
                .expect("input は loss に到達する");

            assert_close(
                value_fused,
                value_fallback,
                &format!("bce_loss forward ({kind:?}/{reduction:?})"),
            );
            for i in 0..2 {
                for j in 0..2 {
                    let a = dinput_fused.get(&[i, j]).unwrap();
                    let e = dinput_fallback.get(&[i, j]).unwrap();
                    assert_close(
                        a,
                        e,
                        &format!("bce_loss backward ({kind:?}/{reduction:?})[{i},{j}]"),
                    );
                }
            }
        }
    }
}

#[test]
fn fused_bce_loss_forward_falls_back_when_unsupported() {
    let ops = AlwaysUnsupportedBceOps {
        inner: common::naive_ops(),
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let input = tape.var(&t(vec![0.2, 0.8, 0.5, 0.9], &[2, 2]));
    let target = tape.var(&t(vec![0.0, 1.0, 1.0, 0.0], &[2, 2]));
    let loss = input.bce_loss(&target, Reduction::Mean).unwrap();

    // フォールバック（`eval::bce_loss`）の解析値を手計算で突合する。
    let expected_forward: f32 = {
        let pairs = [(0.2f32, 0.0f32), (0.8, 1.0), (0.5, 1.0), (0.9, 0.0)];
        let sum: f32 = pairs
            .iter()
            .map(|&(p, y)| bce_elem_loss(p, y, BceKind::Probabilities))
            .sum();
        sum / 4.0
    };
    assert_close(
        scalar(&loss.to_tensor()),
        expected_forward,
        "fallback forward mean",
    );

    let grads = tape.backward(&loss).unwrap();
    let dinput = grads.get(&input).unwrap().expect("到達する");
    let pairs = [(0.2f32, 0.0f32), (0.8, 1.0), (0.5, 1.0), (0.9, 0.0)];
    let scale = 1.0 / 4.0;
    for (idx, &(p, y)) in pairs.iter().enumerate() {
        let i = idx / 2;
        let j = idx % 2;
        let expected = scale * bce_elem_grad_input(p, y, BceKind::Probabilities);
        assert_close(
            dinput.get(&[i, j]).unwrap(),
            expected,
            &format!("fallback backward [{i},{j}]"),
        );
    }
}

#[test]
fn fused_bce_loss_forward_error_other_than_unsupported_propagates() {
    let ops = AlwaysFailingBceOps {
        inner: common::naive_ops(),
        fail_forward: true,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let input = tape.var(&t(vec![0.2, 0.8], &[2]));
    let target = tape.var(&t(vec![0.0, 1.0], &[2]));

    let result = input.bce_loss(&target, Reduction::Mean);
    assert!(
        matches!(
            result,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ),
        "expected KernelLaunchFailed to propagate without fallback, got {result:?}"
    );
}

#[test]
fn fused_bce_loss_backward_error_other_than_unsupported_propagates() {
    let ops = AlwaysFailingBceOps {
        inner: common::naive_ops(),
        fail_forward: false,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let input = tape.var(&t(vec![0.2, 0.8], &[2]));
    let target = tape.var(&t(vec![0.0, 1.0], &[2]));
    // forward は `Unsupported` を返す設定（`fail_forward: false`）なので
    // 従来の `eval::bce_loss` へフォールバックして成功する。
    let loss = input.bce_loss(&target, Reduction::Mean).unwrap();

    let result = tape.backward(&loss);
    assert!(
        matches!(
            result,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ),
        "expected KernelLaunchFailed to propagate without fallback, got {result:?}"
    );
}
