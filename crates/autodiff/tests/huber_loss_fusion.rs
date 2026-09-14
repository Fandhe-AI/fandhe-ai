//! `Var::huber_loss`／`smooth_l1_loss`（`huber_loss_impl` の共通実体）・
//! `grad::vjp` の `Op::HuberLoss` 分岐が、`BackendOps::huber_loss`／
//! `huber_loss_backward`（イシュー #1739）の融合カーネルへ透過的に
//! 切り替わることの直接検証（`mse_loss_fusion.rs` と同型のカウンタ付き
//! フィクスチャ方式）。
//!
//! 4 点を固定する（`mse_loss_fusion.rs` と同じ構成）:
//! 1. 融合カーネル実装ありの `BackendOps` では forward/backward とも
//!    実際にそのメソッドが呼ばれ（`Arc<AtomicUsize>` カウンタ）、
//!    フォールバック経路（`eval::huber_loss`／`huber_loss_vjp`）と数値
//!    一致複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で
//!    一致する。
//! 2. 常時 `Unsupported` を返すフィクスチャ（融合カーネル未実装
//!    バックエンド相当）ではフォールバックが働き、解析値と一致する。
//! 3. `Unsupported` 以外のエラー（融合カーネルの実行時失敗を模す）は
//!    フォールバックせずそのまま伝播する（判定迂回経路を作らない。
//!    `.claude/rules/security.md` A08）。forward・backward 双方で確認する。

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::{AutodiffError, Reduction, Tape};
use fandhe_ai_tensor_core::{BackendError, BackendOps, Device, HuberKind, MseReduction, Tensor};

/// 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
/// `.claude/rules/coding-rust.md`）。`fandhe_ai_backend_cpu::parity::
/// assert_parity`（正本の判定式・定数）を dev-dependency 追加なしに
/// 再利用できないため（`autodiff` は具体バックエンドクレートへ
/// 依存しない設計上の不変条件があり、`architecture_boundaries.rs`
/// が `backend-cpu` の dev-dependency 追加も含めて機械的に禁止する。
/// `docs/fusion-graph-design.md` §3.4）、同一判定式・同一定数値を
/// テストローカルに再実装する（`mse_loss_fusion.rs::assert_close`・
/// `sgd_device_parity.rs::assert_close` と同一実装・同方針。閾値を
/// 緩和したものではない）。
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

/// 素朴参照実装（`eval::huber_elem_loss`／`huber_elem_grad` と同一の式）。
fn elem_loss(d: f32, kind: HuberKind, delta: f32) -> f32 {
    let abs_d = d.abs();
    match kind {
        HuberKind::Huber => {
            if abs_d < delta {
                0.5 * d * d
            } else {
                delta * (abs_d - 0.5 * delta)
            }
        }
        HuberKind::SmoothL1 => {
            if abs_d < delta {
                0.5 * d * d / delta
            } else {
                abs_d - 0.5 * delta
            }
        }
        _ => 0.0,
    }
}

fn elem_grad(d: f32, kind: HuberKind, delta: f32) -> f32 {
    let abs_d = d.abs();
    match kind {
        HuberKind::Huber => {
            if abs_d < delta {
                d
            } else {
                delta.copysign(d)
            }
        }
        HuberKind::SmoothL1 => {
            if abs_d < delta {
                d / delta
            } else {
                1.0f32.copysign(d)
            }
        }
        _ => 0.0,
    }
}

/// `common::NaiveOps` に委譲しつつ `huber_loss`／`huber_loss_backward`
/// のみを素朴な参照実装でオーバーライドし、呼び出し回数を
/// `Arc<AtomicUsize>` で記録するフィクスチャ（`CountingMseOps` と同型）。
struct CountingHuberOps {
    inner: Box<dyn BackendOps + Send>,
    forward_calls: Arc<AtomicUsize>,
    backward_calls: Arc<AtomicUsize>,
}

impl BackendOps for CountingHuberOps {
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

    fn huber_loss(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: HuberKind,
        delta: f32,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        self.forward_calls.fetch_add(1, Ordering::SeqCst);
        let p = pred.contiguous();
        let tt = target.contiguous();
        let pd = p.as_slice().unwrap_or(&[]);
        let td = tt.as_slice().unwrap_or(&[]);
        let numel = pd.len();
        if numel == 0 {
            return Tensor::new(vec![0.0], &[]).map_err(BackendError::ShapeMismatch);
        }
        let sum: f32 = pd
            .iter()
            .zip(td.iter())
            .map(|(&x, &y)| elem_loss(x - y, kind, delta))
            .sum();
        let value = match reduction {
            MseReduction::Mean => sum / numel as f32,
            MseReduction::Sum => sum,
            _ => sum,
        };
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    fn huber_loss_backward(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: HuberKind,
        delta: f32,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        self.backward_calls.fetch_add(1, Ordering::SeqCst);
        let p = pred.contiguous();
        let tt = target.contiguous();
        let pd = p.as_slice().unwrap_or(&[]);
        let td = tt.as_slice().unwrap_or(&[]);
        let dpred: Vec<f32> = pd
            .iter()
            .zip(td.iter())
            .map(|(&x, &y)| scale * elem_grad(x - y, kind, delta))
            .collect();
        Tensor::new(dpred, pred.shape()).map_err(BackendError::ShapeMismatch)
    }
}

/// 常時 `Unsupported` を返すフィクスチャ（融合カーネル未実装バックエンド
/// 相当。`huber_loss`／`huber_loss_backward` を何もオーバーライドせず
/// 既定実装〈fail-safe `Unsupported`〉のまま使う）。
struct AlwaysUnsupportedHuberOps {
    inner: Box<dyn BackendOps + Send>,
}

impl BackendOps for AlwaysUnsupportedHuberOps {
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

/// `Op::HuberLoss` の forward・backward の一方で `Unsupported` 以外の
/// エラーを常に返すフィクスチャ（`AlwaysFailingMseOps` と同型）。
struct AlwaysFailingHuberOps {
    inner: Box<dyn BackendOps + Send>,
    fail_forward: bool,
}

impl BackendOps for AlwaysFailingHuberOps {
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
    fn huber_loss(
        &self,
        _pred: &Tensor<f32>,
        _target: &Tensor<f32>,
        _kind: HuberKind,
        _delta: f32,
        _reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        if self.fail_forward {
            Err(BackendError::KernelLaunchFailed(
                "AlwaysFailingHuberOps: simulated fused Huber forward failure (not Unsupported)"
                    .into(),
            ))
        } else {
            Err(BackendError::Unsupported("forward not under test".into()))
        }
    }
    fn huber_loss_backward(
        &self,
        _pred: &Tensor<f32>,
        _target: &Tensor<f32>,
        _kind: HuberKind,
        _delta: f32,
        _scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        if self.fail_forward {
            Err(BackendError::Unsupported("backward not under test".into()))
        } else {
            Err(BackendError::KernelLaunchFailed(
                "AlwaysFailingHuberOps: simulated fused Huber backward failure (not Unsupported)"
                    .into(),
            ))
        }
    }
}

#[test]
fn fused_huber_loss_is_invoked_and_matches_fallback() {
    let pred_data = vec![1.5, -2.0, 3.0, 0.5];
    let target_data = vec![0.5, -1.0, 2.5, 3.0];

    for kind in [HuberKind::Huber, HuberKind::SmoothL1] {
        for delta in [0.5f32, 1.0, 2.0] {
            for reduction in [Reduction::Mean, Reduction::Sum] {
                let forward_calls = Arc::new(AtomicUsize::new(0));
                let backward_calls = Arc::new(AtomicUsize::new(0));
                let fused_ops = CountingHuberOps {
                    inner: common::naive_ops(),
                    forward_calls: forward_calls.clone(),
                    backward_calls: backward_calls.clone(),
                };
                let tape_fused = Tape::new_with_ops(Box::new(fused_ops));
                let pred_fused = tape_fused.var(&t(pred_data.clone(), &[2, 2]));
                let target_fused = tape_fused.var(&t(target_data.clone(), &[2, 2]));
                let loss_fused = match kind {
                    HuberKind::Huber => pred_fused
                        .huber_loss(&target_fused, delta, reduction)
                        .unwrap(),
                    HuberKind::SmoothL1 => pred_fused
                        .smooth_l1_loss(&target_fused, delta, reduction)
                        .unwrap(),
                    _ => unreachable!(),
                };
                assert_eq!(
                    forward_calls.load(Ordering::SeqCst),
                    1,
                    "{kind:?}/{delta}/{reduction:?}: huber_loss は forward で 1 回呼ばれる契約"
                );
                let value_fused = scalar(&loss_fused.to_tensor());
                let grads_fused = tape_fused.backward(&loss_fused).unwrap();
                assert_eq!(
                    backward_calls.load(Ordering::SeqCst),
                    1,
                    "{kind:?}/{delta}/{reduction:?}: huber_loss_backward は backward で 1 回呼ばれる契約"
                );
                let dpred_fused = grads_fused
                    .get(&pred_fused)
                    .unwrap()
                    .expect("pred は loss に到達する")
                    .clone();

                // フォールバック経路（`NaiveOps`。既定 `Unsupported`）との突合。
                let tape_fallback = Tape::new_with_ops(common::naive_ops());
                let pred_fallback = tape_fallback.var(&t(pred_data.clone(), &[2, 2]));
                let target_fallback = tape_fallback.var(&t(target_data.clone(), &[2, 2]));
                let loss_fallback = match kind {
                    HuberKind::Huber => pred_fallback
                        .huber_loss(&target_fallback, delta, reduction)
                        .unwrap(),
                    HuberKind::SmoothL1 => pred_fallback
                        .smooth_l1_loss(&target_fallback, delta, reduction)
                        .unwrap(),
                    _ => unreachable!(),
                };
                let value_fallback = scalar(&loss_fallback.to_tensor());
                let grads_fallback = tape_fallback.backward(&loss_fallback).unwrap();
                let dpred_fallback = grads_fallback
                    .get(&pred_fallback)
                    .unwrap()
                    .expect("pred は loss に到達する");

                assert_close(
                    value_fused,
                    value_fallback,
                    &format!("huber_loss forward ({kind:?}/{delta}/{reduction:?})"),
                );
                for i in 0..2 {
                    for j in 0..2 {
                        let a = dpred_fused.get(&[i, j]).unwrap();
                        let e = dpred_fallback.get(&[i, j]).unwrap();
                        assert_close(
                            a,
                            e,
                            &format!(
                                "huber_loss backward ({kind:?}/{delta}/{reduction:?})[{i},{j}]"
                            ),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn fused_huber_loss_forward_falls_back_when_unsupported() {
    let ops = AlwaysUnsupportedHuberOps {
        inner: common::naive_ops(),
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let pred = tape.var(&t(vec![1.5, -2.0, 3.0, 0.25], &[2, 2]));
    let target = tape.var(&t(vec![0.5, -1.0, 2.5, 1.0], &[2, 2]));
    let loss = pred.huber_loss(&target, 1.0, Reduction::Mean).unwrap();
    // フォールバック（`eval::huber_loss`）の解析値: d=[1.0,-1.0,0.5,-0.75]
    // → l=[0.5,0.5,0.125,0.28125]（delta=1）→ mean=1.40625/4=0.3515625。
    assert_close(
        scalar(&loss.to_tensor()),
        0.3515625,
        "fallback forward mean",
    );

    let grads = tape.backward(&loss).unwrap();
    let dpred = grads.get(&pred).unwrap().expect("到達する");
    // scale = 1/4 = 0.25 → dpred = 0.25*grad_elem(d)
    // grad_elem = [copysign(1,1)=1, copysign(1,-1)=-1, 0.5, -0.75]
    let expected = [0.25f32, -0.25, 0.125, -0.1875];
    for (idx, &e) in expected.iter().enumerate() {
        let i = idx / 2;
        let j = idx % 2;
        assert_close(
            dpred.get(&[i, j]).unwrap(),
            e,
            &format!("fallback backward [{i},{j}]"),
        );
    }
}

#[test]
fn fused_huber_loss_forward_error_other_than_unsupported_propagates() {
    let ops = AlwaysFailingHuberOps {
        inner: common::naive_ops(),
        fail_forward: true,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let pred = tape.var(&t(vec![1.0, 2.0], &[2]));
    let target = tape.var(&t(vec![0.0, 0.0], &[2]));

    let result = pred.huber_loss(&target, 1.0, Reduction::Mean);
    assert!(
        matches!(
            result,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ),
        "expected KernelLaunchFailed to propagate without fallback, got {result:?}"
    );
}

#[test]
fn fused_huber_loss_backward_error_other_than_unsupported_propagates() {
    let ops = AlwaysFailingHuberOps {
        inner: common::naive_ops(),
        fail_forward: false,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let pred = tape.var(&t(vec![1.0, 2.0], &[2]));
    let target = tape.var(&t(vec![0.0, 0.0], &[2]));
    // forward は `Unsupported` を返す設定（`fail_forward: false`）なので
    // 従来の `eval::huber_loss` へフォールバックして成功する。
    let loss = pred.huber_loss(&target, 1.0, Reduction::Mean).unwrap();

    let result = tape.backward(&loss);
    assert!(
        matches!(
            result,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ),
        "expected KernelLaunchFailed to propagate without fallback, got {result:?}"
    );
}
