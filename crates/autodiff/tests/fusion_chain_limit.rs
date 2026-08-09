//! 連鎖長上限（4〜6 段）の遅延評価経路（`Tape::push_lazy`）への適用の
//! 受け入れテスト（#404・`docs/fusion-graph-design.md` §3.5.4）。
//!
//! `tensor_core::MAX_FUSED_CHAIN_LEN`（= 6）到達時点で `Var::add`/`mul`/
//! `relu`/`exp`/`tanh` がその場実体化することを、`fusion_backend_
//! integration.rs` と同じフィクスチャ（`CountingFusedOps`・
//! `AlwaysUnsupportedFused`・`common::NaiveOps`）で黒箱検証する。
//! `FusionPlan::ops()`/`leaf_count()`（`tensor-core` 公開 DTO）を使い、
//! 各 `run_fused` 呼び出しの interior 数（`ops().count() - leaf_count()`）
//! が上限以下であることを実測する。数値比較は既存の複合判定
//! （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。REQ-2・
//! `.claude/rules/coding-rust.md`）を用い、許容誤差は変更しない。

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use autodiff::{Tape, Var};
use tensor_core::{
    BackendError, BackendOps, Device, FusedOpKind, FusionPlan, MAX_FUSED_CHAIN_LEN, Tensor,
};

/// `run_fused` を実行しつつ、呼び出しごとの interior 数
/// （`plan.ops().count() - plan.leaf_count()`）を記録するカウンタ付き
/// `BackendOps`（`fusion_backend_integration.rs::CountingFusedOps` と
/// 同型の参照実装。呼び出し粒度の検証のため独自に持つ）。
struct RecordingFusedOps {
    inner: common::NaiveOps,
    fused_calls: Arc<AtomicUsize>,
    interior_sizes: Arc<std::sync::Mutex<Vec<usize>>>,
}

impl BackendOps for RecordingFusedOps {
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
        self.fused_calls.fetch_add(1, Ordering::SeqCst);
        let ops: Vec<FusedOpKind> = plan.ops().collect();
        let interior = ops.len() - plan.leaf_count();
        self.interior_sizes.lock().unwrap().push(interior);

        let mut values: Vec<Tensor<f32>> = Vec::new();
        for kind in ops {
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

/// 常時 `Unsupported` を返す `BackendOps`（per-op 参照経路。
/// `fusion_backend_integration.rs::AlwaysUnsupportedFused` と同型）。
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

/// `run_fused`・`add` の両方が常時 `Err` を返す `BackendOps`（層 1
/// フォールバック経路自体の失敗伝播を検証するための最小フィクスチャ。
/// `add`/`mul`/`relu`/`exp`/`tanh` のうち `add` のみ失敗させれば十分
/// なため他メソッドは `common::NaiveOps` へ委譲する）。
struct AlwaysFailingAdd {
    inner: common::NaiveOps,
}

impl BackendOps for AlwaysFailingAdd {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.gemm(a, b)
    }
    fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "AlwaysFailingAdd: add は常に失敗する（テスト用フィクスチャ）".into(),
        ))
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
    // `run_fused` はデフォルト実装（`Unsupported`）のまま override しない
    // ため、層 1（`materialize_fallible`）は `Unsupported` を受けて
    // per-op フォールバック（`fallback_per_op`）へ委譲する。そこで
    // `add` 自体も `Unsupported` を返すため、最終的に `Err` が `?` で
    // 伝播する（`materialize_fallible` のドキュメント参照）。
}

fn composite_close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs() as f64;
    let rel = diff / (a.abs() as f64).max(b.abs() as f64).max(1e-12);
    rel < 1e-3 || diff < 1e-5
}

fn assert_tensors_close(a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape());
    let ac = a.contiguous();
    let bc = b.contiguous();
    let ad = ac.as_slice().unwrap();
    let bd = bc.as_slice().unwrap();
    for (x, y) in ad.iter().zip(bd.iter()) {
        assert!(
            composite_close(*x, *y),
            "数値一致複合判定を満たさない: {x} vs {y}"
        );
    }
}

/// `n` 段の線形 elementwise 連鎖（`add`/`relu` の交互適用）を構築する。
/// `add` は bias（全要素 0.01・shape `[4]`）との加算とし、`relu` は
/// 非 fallible 側の代表として挟む。段数 `n` は `add`/`relu` を交互に
/// 積んだ回数（各段が 1 ノード＝ 1 段）。
fn build_linear_chain<'t>(tape: &'t Tape, x: &Var<'t>, n: usize) -> Var<'t> {
    let bias = tape.var(&Tensor::new(vec![0.01, 0.02, -0.01, -0.02], &[4]).unwrap());
    let mut cur = *x;
    for i in 0..n {
        cur = if i % 2 == 0 {
            cur.add(&bias).unwrap()
        } else {
            cur.relu()
        };
    }
    cur
}

#[test]
fn deep_chain_splits_into_multiple_fused_plans_bounded_by_limit() {
    // テスト計画 §5-1: 10 段前後の線形連鎖で (a) run_fused が 2 回以上
    // 呼ばれる、(b) 各呼び出しの interior 数が MAX_FUSED_CHAIN_LEN 以下、
    // (c) 最終値が per-op 参照経路と数値一致。
    let fused_calls = Arc::new(AtomicUsize::new(0));
    let interior_sizes = Arc::new(std::sync::Mutex::new(Vec::new()));
    let ops = RecordingFusedOps {
        inner: common::NaiveOps,
        fused_calls: fused_calls.clone(),
        interior_sizes: interior_sizes.clone(),
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let x = tape.var(&Tensor::new(vec![0.5, -0.5, 1.5, -1.5], &[4]).unwrap());
    let out = build_linear_chain(&tape, &x, 10);
    let result = out.to_tensor();

    assert!(
        fused_calls.load(Ordering::SeqCst) >= 2,
        "10 段連鎖なら run_fused が複数回に分割されるはず（実測: {}）",
        fused_calls.load(Ordering::SeqCst)
    );
    for &interior in interior_sizes.lock().unwrap().iter() {
        assert!(
            interior <= MAX_FUSED_CHAIN_LEN,
            "run_fused 呼び出しの interior 数が上限を超えた: {interior} > {MAX_FUSED_CHAIN_LEN}"
        );
    }

    let ref_tape = Tape::new_with_ops(Box::new(AlwaysUnsupportedFused {
        inner: common::NaiveOps,
    }));
    let ref_x = ref_tape.var(&Tensor::new(vec![0.5, -0.5, 1.5, -1.5], &[4]).unwrap());
    let ref_result = build_linear_chain(&ref_tape, &ref_x, 10).to_tensor();
    assert_tensors_close(&result, &ref_result);
}

#[test]
fn exact_limit_chain_materializes_once_at_push_time() {
    // テスト計画 §5-2: 6 段ちょうどの連鎖は 6 段目の push 時点で自己
    // 実体化され、run_fused が 1 回・interior 6 で呼ばれる。その後の
    // to_tensor() で追加呼び出しがないこと。
    let fused_calls = Arc::new(AtomicUsize::new(0));
    let interior_sizes = Arc::new(std::sync::Mutex::new(Vec::new()));
    let ops = RecordingFusedOps {
        inner: common::NaiveOps,
        fused_calls: fused_calls.clone(),
        interior_sizes: interior_sizes.clone(),
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let x = tape.var(&Tensor::new(vec![0.5, -0.5, 1.5, -1.5], &[4]).unwrap());

    // push 直後（`to_tensor()` 呼び出し前）にすでに 1 回実体化済みで
    // あることを検証する（push 時上限適用の直接証拠）。
    let out = build_linear_chain(&tape, &x, MAX_FUSED_CHAIN_LEN);
    assert_eq!(
        fused_calls.load(Ordering::SeqCst),
        1,
        "6 段目の push 時点で自己実体化されるはず"
    );
    assert_eq!(interior_sizes.lock().unwrap().as_slice(), &[6]);

    let _ = out.to_tensor();
    assert_eq!(
        fused_calls.load(Ordering::SeqCst),
        1,
        "実体化済みノードを to_tensor() が再実体化してはいけない"
    );
}

#[test]
fn below_limit_chain_stays_lazy_until_materialize_boundary() {
    // テスト計画 §5-3: 5 段（上限未満）は push 時点では run_fused 未
    // 呼び出し（遅延維持）、to_tensor() で 1 回・interior 5。
    let fused_calls = Arc::new(AtomicUsize::new(0));
    let interior_sizes = Arc::new(std::sync::Mutex::new(Vec::new()));
    let ops = RecordingFusedOps {
        inner: common::NaiveOps,
        fused_calls: fused_calls.clone(),
        interior_sizes: interior_sizes.clone(),
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let x = tape.var(&Tensor::new(vec![0.5, -0.5, 1.5, -1.5], &[4]).unwrap());

    let out = build_linear_chain(&tape, &x, MAX_FUSED_CHAIN_LEN - 1);
    assert_eq!(
        fused_calls.load(Ordering::SeqCst),
        0,
        "5 段（上限未満）は push 時点ではまだ実体化されないはず"
    );

    let _ = out.to_tensor();
    assert_eq!(fused_calls.load(Ordering::SeqCst), 1);
    assert_eq!(interior_sizes.lock().unwrap().as_slice(), &[5]);
}

#[test]
fn add_at_limit_propagates_materialize_failure() {
    // テスト計画 §5-4: run_fused が Unsupported、per-op add も Err を
    // 返すフィクスチャで、上限到達させた Var::add が Err を返す
    // （§3.5.4「add/mul は層 1 の失敗伝播規約も併せ持つ」の検証）。
    let ops = AlwaysFailingAdd {
        inner: common::NaiveOps,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let x = tape.var(&Tensor::new(vec![0.5, -0.5, 1.5, -1.5], &[4]).unwrap());
    let bias = tape.var(&Tensor::new(vec![0.01, 0.02, -0.01, -0.02], &[4]).unwrap());

    // relu/add を交互に MAX_FUSED_CHAIN_LEN 段積み、**最終段が add**に
    // なるよう構成する（`i` が奇数の段で `add`）。最終段（add）で depth
    // が上限へ到達しその場実体化が走るため、`add` の失敗が `Var::add`
    // の戻り値 `Err` として伝播するはず（`relu`/`exp`/`tanh` は非
    // fallible な層 2 経由のため、上限がそちらに当たると本テストの
    // 主張〈add の Err 伝播〉を検証できない）。
    let mut cur = x;
    let mut last_err = None;
    for i in 0..MAX_FUSED_CHAIN_LEN {
        cur = if i % 2 == 1 {
            match cur.add(&bias) {
                Ok(v) => v,
                Err(e) => {
                    last_err = Some(e);
                    break;
                }
            }
        } else {
            cur.relu()
        };
    }
    assert!(
        last_err.is_some(),
        "上限到達時の自己実体化失敗が Var::add の Err として伝播しなかった"
    );
}

#[test]
fn backward_through_deep_chain_matches_reference_gradients() {
    // テスト計画 §5-5: 10 段超の連鎖 + sum の loss で tape.backward が
    // 成功し、勾配 shape・値が per-op 参照経路と数値一致（複数実体化窓
    // をまたぐ VJP の回帰）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&Tensor::new(vec![0.5, -0.5, 1.5, -1.5], &[4]).unwrap());
    let out = build_linear_chain(&tape, &x, 13);
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let grad_x = grads
        .get(&x)
        .unwrap()
        .expect("x は loss に寄与しているはず");
    assert_eq!(grad_x.shape(), &[4]);

    let ref_tape = Tape::new_with_ops(Box::new(AlwaysUnsupportedFused {
        inner: common::NaiveOps,
    }));
    let ref_x = ref_tape.var(&Tensor::new(vec![0.5, -0.5, 1.5, -1.5], &[4]).unwrap());
    let ref_out = build_linear_chain(&ref_tape, &ref_x, 13);
    let ref_loss = ref_out.sum(None).unwrap();
    let ref_grads = ref_tape.backward(&ref_loss).unwrap();
    let ref_grad_x = ref_grads
        .get(&ref_x)
        .unwrap()
        .expect("参照経路でも x は loss に寄与しているはず");

    assert_tensors_close(grad_x, ref_grad_x);
}

#[test]
fn fan_out_at_limit_boundary_does_not_panic() {
    // テスト計画 §5-6: depth 5 のノードを 2 消費者が共有し双方が上限
    // 到達するケースで panic せず数値一致（OnceCell 二重到達の回帰）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&Tensor::new(vec![0.5, -0.5, 1.5, -1.5], &[4]).unwrap());
    // depth 5 の共有ノード（add→relu→add→relu→add）を構築する。
    let shared = build_linear_chain(&tape, &x, MAX_FUSED_CHAIN_LEN - 1);
    // 2 つの消費者がそれぞれ shared に対し 1 演算（add）を適用し、
    // どちらも depth 6（上限）へ到達して自己実体化する。
    let bias = tape.var(&Tensor::new(vec![0.1, 0.1, 0.1, 0.1], &[4]).unwrap());
    let consumer_a = shared.add(&bias).unwrap();
    let consumer_b = shared.add(&bias).unwrap();

    let a_val = consumer_a.to_tensor();
    let b_val = consumer_b.to_tensor();
    assert_tensors_close(&a_val, &b_val);

    let ref_tape = Tape::new_with_ops(Box::new(AlwaysUnsupportedFused {
        inner: common::NaiveOps,
    }));
    let ref_x = ref_tape.var(&Tensor::new(vec![0.5, -0.5, 1.5, -1.5], &[4]).unwrap());
    let ref_shared = build_linear_chain(&ref_tape, &ref_x, MAX_FUSED_CHAIN_LEN - 1);
    let ref_bias = ref_tape.var(&Tensor::new(vec![0.1, 0.1, 0.1, 0.1], &[4]).unwrap());
    let ref_a = ref_shared.add(&ref_bias).unwrap().to_tensor();
    assert_tensors_close(&a_val, &ref_a);
}
