//! TASK-12.1d（#164）受け入れ条件「通常経路と融合経路が透過的に切り替
//! わる」の直接検証。
//!
//! `BackendOps::run_fused` をオーバーライドした呼び出しカウンタ付き
//! フィクスチャを `Tape::new(ops)` へ渡し、4 段以上の elementwise 連鎖
//! （`add`/`mul`/`relu`/`exp`/`tanh`）で `run_fused` が呼ばれること（融合
//! 経路）と、常時 `Unsupported` を返すフィクスチャで per-op フォール
//! バックが働き数値一致複合判定（相対誤差 1e-3 未満 または 絶対誤差
//! 1e-5 未満。REQ-2・`.claude/rules/coding-rust.md`）を満たすこと（通常
//! 経路）の両方を固定する。両経路が同一の最終値を返すことも確認する
//! （融合の有無で数値が変わらない。`docs/fusion-graph-design.md` §4）。

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use autodiff::Tape;
use tensor_core::{BackendError, BackendOps, Device, FusedOpKind, FusionPlan, Tensor};

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
fn build_chain<'t>(tape: &'t Tape, x: &autodiff::Var<'t>) -> autodiff::Var<'t> {
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
    let tape = Tape::new(Box::new(ops));
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
    let fused_tape = Tape::new(Box::new(CountingFusedOps {
        inner: common::NaiveOps,
        fused_calls: fused_calls.clone(),
    }));
    let x_fused = fused_tape.var(&Tensor::new(x_data.clone(), &[4]).unwrap());
    let fused_result = build_chain(&fused_tape, &x_fused).to_tensor();
    assert!(
        fused_calls.load(Ordering::SeqCst) >= 1,
        "融合経路の前提（run_fused 呼び出し）が満たされていない"
    );

    let fallback_tape = Tape::new(Box::new(AlwaysUnsupportedFused {
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
    let tape = Tape::new(common::naive_ops());
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

/// `Tape: Send` の静的アサーション（設計書 §3.4「`Tape: Send` を維持する」）。
#[test]
fn tape_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Tape>();
}
