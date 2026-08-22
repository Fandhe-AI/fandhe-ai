//! TASK-12.1d（#164）受け入れ条件「通常経路と融合経路が透過的に切り替
//! わる」の直接検証。
//!
//! `BackendOps::run_fused` をオーバーライドした呼び出しカウンタ付き
//! フィクスチャを `Tape::new_with_ops(ops)` へ渡し、4 段以上の elementwise 連鎖
//! （`add`/`mul`/`relu`/`exp`/`tanh`）で `run_fused` が呼ばれること（融合
//! 経路）と、常時 `Unsupported` を返すフィクスチャで per-op フォール
//! バックが働き数値一致複合判定（相対誤差 1e-3 未満 または 絶対誤差
//! 1e-5 未満。REQ-2・`.claude/rules/coding-rust.md`）を満たすこと（通常
//! 経路）の両方を固定する。両経路が同一の最終値を返すことも確認する
//! （融合の有無で数値が変わらない。`docs/fusion-graph-design.md` §4）。

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::Tape;
use fandhe_ai_tensor_core::{BackendError, BackendOps, Device, FusedOpKind, FusionPlan, Tensor};

/// `run_fused` を実際に実行する（`FusionPlan::ops()` を辿って per-op
/// 単純合成する参照実装）カウンタ付き `BackendOps`。#163（CPU 融合実行
/// 器の本実装）が未マージのため、ここでは「`run_fused` が呼ばれた
/// こと」自体をテストの主眼とし、中身は `common::NaiveOps` の per-op
/// メソッドをプラン順に適用するだけの単純な参照実装とする。
struct CountingFusedOps {
    inner: common::NaiveOps,
    fused_calls: Arc<AtomicUsize>,
}

impl BackendOps for CountingFusedOps {
    fn device(&self) -> Device {
        Device::Cpu
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

    /// 呼び出し回数を記録したうえで、`FusionPlan::ops()`（発生順）を
    /// 辿って per-op 合成する最小実装（#163 の実融合カーネルの代わり）。
    fn run_fused(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
    ) -> Result<Tensor<f32>, BackendError> {
        self.fused_calls.fetch_add(1, Ordering::SeqCst);
        // `FusionPlan::ops()` は葉ノード自身も `FusedOpKind::Input` として
        // 発生順に列挙する（`plan.rs::FusionPlan::from_ops` が葉を
        // `graph` の先頭 `leaf_count` ノードとして構築するため）。よって
        // `values` は空から始め、`ops()` が返す列をそのまま発生順に
        // 追記していけば、後続ノードの `lhs`/`rhs`/`input`（"leaves then
        // ops" のグラフ内連番）と `values` の添字が一致する。
        let mut values: Vec<Tensor<f32>> = Vec::new();
        for kind in plan.ops() {
            let v = match kind {
                FusedOpKind::Input { leaf_index } => leaves[leaf_index].clone(),
                FusedOpKind::Add { lhs, rhs } => self.inner.add(&values[lhs], &values[rhs])?,
                FusedOpKind::Mul { lhs, rhs } => self.inner.mul(&values[lhs], &values[rhs])?,
                FusedOpKind::Relu { input } => self.inner.relu(&values[input])?,
                FusedOpKind::Exp { input } => self.inner.exp(&values[input])?,
                FusedOpKind::Tanh { input } => self.inner.tanh(&values[input])?,
                // #586 で `FusedOpKind` へ追加された reduction（Sum／Max）
                // ・Rsqrt は、`fandhe_ai_autodiff::tape::build_lazy_plan` が現状これらを
                // 遅延評価対象にせず `push_eager` で実体化するため
                // （`crates/autodiff/src/tape.rs`）、本テストフィクスチャの
                // `run_fused` へ実際に渡される `ops` には現れない。到達不能
                // だが `-D warnings` の網羅性要求を満たすため、実装済み
                // カーネル未対応として型付きエラーを返す（本番経路 panic
                // 禁止方針。`.claude/rules/coding-rust.md`）。
                FusedOpKind::Rsqrt { .. } | FusedOpKind::Sum { .. } | FusedOpKind::Max { .. } => {
                    return Err(BackendError::Unsupported(
                        "CountingFusedOps::run_fused: reduction/Rsqrt not implemented \
                         (fandhe_ai_tensor_core::fusion IR extension #586; CPU kernel out of scope)"
                            .into(),
                    ));
                }
                // `fandhe_ai_tensor_core::FusedOpKind` は `#[non_exhaustive]`（codex-review
                // PR #648 P1 是正）のため、別クレートである本テストからの match
                // は将来の未知 variant に備え `_` 分岐が必須。
                _ => {
                    return Err(BackendError::Unsupported(
                        "CountingFusedOps::run_fused: unknown FusedOpKind variant \
                         (fandhe_ai_tensor_core::FusedOpKind is #[non_exhaustive]; unrecognized future \
                         variant)"
                            .into(),
                    ));
                }
            };
            values.push(v);
        }
        values
            .last()
            .cloned()
            .ok_or_else(|| BackendError::Unsupported("run_fused: empty plan".into()))
    }
}

/// 常時 `Unsupported` を返す `BackendOps`（per-op フォールバック経路の
/// 検証用）。`gemm`/`sum`/`max` は `common::NaiveOps` のまま委譲する
/// （elementwise 融合の可否のみを検証対象とするため）。
struct AlwaysUnsupportedFused {
    inner: common::NaiveOps,
}

impl BackendOps for AlwaysUnsupportedFused {
    fn device(&self) -> Device {
        Device::Cpu
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
    // `run_fused` はデフォルト実装（`Unsupported`）のまま override しない。
}

fn composite_close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs() as f64;
    let rel = diff / (a.abs() as f64).max(b.abs() as f64).max(1e-12);
    rel < 1e-3 || diff < 1e-5
}

/// 6 段の elementwise 連鎖（`add → mul → relu → exp → tanh → add`）を
/// 構築する共通ヘルパー。`x`（tape 上の入力）を起点に返す。
fn build_chain<'t>(tape: &'t Tape, x: &fandhe_ai_autodiff::Var<'t>) -> fandhe_ai_autodiff::Var<'t> {
    let bias = tape.var(&Tensor::new(vec![0.1, 0.2, 0.3, 0.4], &[4]).unwrap());
    let scale = tape.var(&Tensor::new(vec![1.1, 0.9, 1.05, 0.95], &[4]).unwrap());
    let h1 = x.add(&bias).unwrap(); // 1
    let h2 = h1.mul(&scale).unwrap(); // 2
    let h3 = h2.relu(); // 3
    let h4 = h3.exp(); // 4
    let h5 = h4.tanh(); // 5
    h5.add(&bias).unwrap() // 6 段目
}

#[test]
fn run_fused_is_called_for_elementwise_chain_beyond_min_length() {
    // 受け入れ条件（透過的切り替え・融合経路）: `run_fused` オーバーライド
    // 済み `BackendOps` を渡すと、4 段以上の elementwise 連鎖の実体化時に
    // `run_fused` が呼ばれる。
    let fused_calls = Arc::new(AtomicUsize::new(0));
    let ops = CountingFusedOps {
        inner: common::NaiveOps,
        fused_calls: fused_calls.clone(),
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let x = tape.var(&Tensor::new(vec![0.5, -0.5, 1.5, -1.5], &[4]).unwrap());
    let out = build_chain(&tape, &x);

    // `to_tensor()`（層 2）が未実体化ノードを実体化する契機。
    let result = out.to_tensor();
    assert_eq!(result.shape(), &[4]);
    assert!(
        fused_calls.load(Ordering::SeqCst) >= 1,
        "run_fused が 1 回も呼ばれなかった（透過的切り替え＝融合経路が機能していない）"
    );
}

#[test]
fn per_op_fallback_matches_fused_path_numerically() {
    // 受け入れ条件（透過的切り替え・通常経路）: `run_fused` が常時
    // `Unsupported` を返すフィクスチャでも per-op フォールバックにより
    // 同一入力に対し同一の最終値（数値一致複合判定を満たす）を返す。
    let x_data = vec![0.5f32, -0.5, 1.5, -1.5];

    let fused_calls = Arc::new(AtomicUsize::new(0));
    let fused_tape = Tape::new_with_ops(Box::new(CountingFusedOps {
        inner: common::NaiveOps,
        fused_calls: fused_calls.clone(),
    }));
    let x_fused = fused_tape.var(&Tensor::new(x_data.clone(), &[4]).unwrap());
    let fused_result = build_chain(&fused_tape, &x_fused).to_tensor();
    assert!(
        fused_calls.load(Ordering::SeqCst) >= 1,
        "融合経路の前提（run_fused 呼び出し）が満たされていない"
    );

    let fallback_tape = Tape::new_with_ops(Box::new(AlwaysUnsupportedFused {
        inner: common::NaiveOps,
    }));
    let x_fallback = fallback_tape.var(&Tensor::new(x_data, &[4]).unwrap());
    let fallback_result = build_chain(&fallback_tape, &x_fallback).to_tensor();

    let fused_slice = fused_result.contiguous();
    let fallback_slice = fallback_result.contiguous();
    let fused_data = fused_slice.as_slice().unwrap();
    let fallback_data = fallback_slice.as_slice().unwrap();
    assert_eq!(fused_data.len(), fallback_data.len());
    for (a, b) in fused_data.iter().zip(fallback_data.iter()) {
        assert!(
            composite_close(*a, *b),
            "融合経路と非融合経路（per-op フォールバック）の結果が数値一致複合判定を満たさない: {a} vs {b}"
        );
    }
}

#[test]
fn backward_through_lazy_chain_succeeds_via_materialize_fallible() {
    // `Tape::backward`（層 1）が forward 記録済みの未実体化ノードを正しく
    // 実体化しながら逆伝播できることを確認する（`matmul` 等を経由せず、
    // 遅延連鎖の末端を直接 loss にする構成）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&Tensor::new(vec![0.5, -0.5, 1.5, -1.5], &[4]).unwrap());
    let chain_out = build_chain(&tape, &x);
    let loss = chain_out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let grad_x = grads
        .get(&x)
        .unwrap()
        .expect("x は loss に寄与しているはず");
    assert_eq!(grad_x.shape(), &[4]);
}

/// `backend-cpu::run_fused_elementwise` を模した `BackendOps`
/// フィクスチャ（Cursor Bugbot・PR #403 是正の回帰テスト用）: leaf の
/// shape が `FusionPlan::output_shape()` と一致しない（broadcast を
/// 伴う）場合に `BackendError::ShapeMismatch` を返す。`backend-cpu` は
/// 実際にこの制約を持つ（`crates/backend-cpu/src/fused_elementwise.rs`
/// 「leaf i is non-contiguous (broadcast/transpose view)」コメント参照。
/// 本フィクスチャは非 contiguous 検査の代わりに shape 不一致検査のみを
/// 単純化して再現する）。
struct ShapeMismatchOnBroadcastFusedOps {
    inner: common::NaiveOps,
}

impl BackendOps for ShapeMismatchOnBroadcastFusedOps {
    fn device(&self) -> Device {
        Device::Cpu
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

    fn run_fused(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = plan.output_shape();
        if leaves.iter().any(|l| l.shape() != out_shape) {
            return Err(BackendError::ShapeMismatch(
                fandhe_ai_tensor_core::ShapeError::BroadcastIncompatible {
                    lhs: out_shape.to_vec(),
                    rhs: leaves
                        .iter()
                        .find(|l| l.shape() != out_shape)
                        .map(|l| l.shape().to_vec())
                        .unwrap_or_default(),
                },
            ));
        }
        let mut values: Vec<Tensor<f32>> = Vec::new();
        for kind in plan.ops() {
            let v = match kind {
                FusedOpKind::Input { leaf_index } => leaves[leaf_index].clone(),
                FusedOpKind::Add { lhs, rhs } => self.inner.add(&values[lhs], &values[rhs])?,
                FusedOpKind::Mul { lhs, rhs } => self.inner.mul(&values[lhs], &values[rhs])?,
                FusedOpKind::Relu { input } => self.inner.relu(&values[input])?,
                FusedOpKind::Exp { input } => self.inner.exp(&values[input])?,
                FusedOpKind::Tanh { input } => self.inner.tanh(&values[input])?,
                // #586 で `FusedOpKind` へ追加された reduction（Sum／Max）
                // ・Rsqrt は、`fandhe_ai_autodiff::tape::build_lazy_plan` が現状これらを
                // 遅延評価対象にせず `push_eager` で実体化するため
                // （`crates/autodiff/src/tape.rs`）、本テストフィクスチャの
                // `run_fused` へ実際に渡される `ops` には現れない。到達不能
                // だが `-D warnings` の網羅性要求を満たすため、実装済み
                // カーネル未対応として型付きエラーを返す（本番経路 panic
                // 禁止方針。`.claude/rules/coding-rust.md`）。
                FusedOpKind::Rsqrt { .. } | FusedOpKind::Sum { .. } | FusedOpKind::Max { .. } => {
                    return Err(BackendError::Unsupported(
                        "CountingFusedOps::run_fused: reduction/Rsqrt not implemented \
                         (fandhe_ai_tensor_core::fusion IR extension #586; CPU kernel out of scope)"
                            .into(),
                    ));
                }
                // `fandhe_ai_tensor_core::FusedOpKind` は `#[non_exhaustive]`（codex-review
                // PR #648 P1 是正）のため、別クレートである本テストからの match
                // は将来の未知 variant に備え `_` 分岐が必須。
                _ => {
                    return Err(BackendError::Unsupported(
                        "CountingFusedOps::run_fused: unknown FusedOpKind variant \
                         (fandhe_ai_tensor_core::FusedOpKind is #[non_exhaustive]; unrecognized future \
                         variant)"
                            .into(),
                    ));
                }
            };
            values.push(v);
        }
        values
            .last()
            .cloned()
            .ok_or_else(|| BackendError::Unsupported("run_fused: empty plan".into()))
    }
}

/// Cursor Bugbot 指摘（PR #403）の回帰テスト: bias broadcast
/// （`[batch, width] + [width]`）を含む遅延連鎖を、broadcast leaf の
/// 融合実行を拒否する `BackendOps`（[`ShapeMismatchOnBroadcastFusedOps`]）
/// 上で `sum`（層 1・`materialize_fallible` 経由）まで実体化しても、
/// `ShapeMismatch` が硬いエラーとして伝播せず per-op フォールバックで
/// 成功すること。修正前は `materialize_fallible` が `Unsupported` 以外の
/// `run_fused` エラーをすべて即時 `Err` 化していたため、この構成で
/// `AutodiffError::Backend(ShapeMismatch)` を返し失敗していた。
#[test]
fn broadcast_bias_add_chain_falls_back_on_fused_shape_mismatch() {
    let ops = ShapeMismatchOnBroadcastFusedOps {
        inner: common::NaiveOps,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let x = tape.var(&Tensor::new(vec![0.5; 8], &[2, 4]).unwrap());
    let bias = tape.var(&Tensor::new(vec![0.1, 0.2, 0.3, 0.4], &[4]).unwrap());
    // bias broadcast（[2,4] + [4]）→ relu の 2 段連鎖。leaf の shape が
    // 揃わないため `run_fused` は `ShapeMismatch` を返す。
    let h = x.add(&bias).unwrap();
    let h = h.relu();
    let loss = h.sum(None).unwrap(); // 層 1（`materialize_fallible`）経由
    assert_eq!(loss.to_tensor().shape(), &[] as &[usize]);

    // `Tape::backward` も同じ層 1 経路を使うため併せて確認する。
    let grads = tape.backward(&loss).unwrap();
    let grad_x = grads.get(&x).unwrap().expect("x は loss に寄与している");
    assert_eq!(grad_x.shape(), &[2, 4]);
}

/// `Tape: Send` の静的アサーション（設計書 §3.4「`Tape: Send` を維持する」）。
#[test]
fn tape_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Tape>();
}
