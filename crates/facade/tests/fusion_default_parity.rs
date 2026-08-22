//! 既定構築（`fandhe_ai::tape()`）経由の融合有効化＋REQ-2 複合判定（受入
//! 基準 3）。
//!
//! `fandhe_ai::tape()` は `fandhe_ai_backend_cpu::CpuBackendOps`（`run_fused` を
//! `run_fused_elementwise` へオーバーライド済み＝融合有効。
//! `crates/backend-cpu/src/ops.rs`）を結線する唯一の入口である
//! （`crates/facade/src/lib.rs::tape`）。本テストは
//!
//! 1. `fandhe_ai::tape()` 上で 5 段の elementwise 連鎖（`add`→`mul`→`relu`→
//!    `exp`→`tanh`。`docs/fusion-graph-design.md` の融合対象パターン）を
//!    実行し、無引数 `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`・融合を経ない
//!    per-op 逐次実装。`crates/autodiff/src/tape.rs::Tape::new`）による
//!    同一連鎖の参照値と REQ-2 統一複合判定
//!    （`fandhe_ai_backend_cpu::parity::assert_parity`。相対誤差 1e-3 未満 または
//!    絶対誤差 1e-5 未満）で比較する。**許容誤差は `parity.rs` の既存
//!    定数をそのまま使い、新規の緩和・独自閾値を導入しない**（変更は
//!    ユーザー承認必須。`.claude/rules/coding-rust.md`）。
//! 2. facade が結線する `CpuBackendOps::run_fused` が、代表的な
//!    `FusionPlan`（同じ 5 段連鎖を表す `FusedOpKind` 列）に対し `Ok`
//!    （`Unsupported` ではない）を返すことを直接観測し、融合が実際に
//!    有効であることを確認する（`crates/autodiff/tests/
//!    fusion_backend_integration.rs` のプラン構築手法を流用）。

use fandhe_ai_autodiff::{Tape, Var};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::{BackendOps, DType, FusedOpKind, FusionPlan, Tensor};

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）のいずれからも
/// `var()` を呼べるようにする、本テストファイル専用のローカル trait
/// （codex-review PR #424 P1 是正で `fandhe_ai::tape()` の戻り値型が
/// `fandhe_ai_autodiff::Tape` から `fandhe_ai::Tape` newtype へ変わったことに伴う
/// 対応。本 trait は facade の公開契約ではなく、単に「融合経路
/// （`fandhe_ai::tape()`）と非融合参照実装（`fandhe_ai_autodiff::Tape::new()`）の
/// 両方に同じ `run_chain_on` を適用したい」というテストの都合のみで
/// 導入する）。
trait VarSource {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_>;
}

impl VarSource for fandhe_ai::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

impl VarSource for Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

/// 決定的な入力値（固定値リテラル。facade へ dev-dep を増やさないため
/// `bench-harness::Xorshift64Star` 等の乱数ユーティリティは使わず、
/// 符号混在の代表値を直接列挙する。イシュー #410 実装計画 §3
/// 「入力は決定的シード」）。
fn leaf_a() -> Tensor<f32> {
    Tensor::new(vec![0.5, -1.2, 2.0, -0.3], &[2, 2]).expect("leaf_a: shape 一致")
}

fn leaf_b() -> Tensor<f32> {
    Tensor::new(vec![-0.7, 1.5, -2.2, 0.9], &[2, 2]).expect("leaf_b: shape 一致")
}

/// `fandhe_ai::tape()` 上で `add`→`mul`→`relu`→`exp`→`tanh` の 5 段連鎖を
/// 実行し、最終 `Var` を返す。[`VarSource`] 経由で `fandhe_ai::Tape`
/// （newtype）・`fandhe_ai_autodiff::Tape`（生の型）の両方に適用できる。
fn run_chain_on<T: VarSource>(tape: &T) -> Var<'_> {
    let a = tape.make_var(&leaf_a());
    let b = tape.make_var(&leaf_b());
    let added = a.add(&b).expect("add: shape 一致");
    let multiplied = added.mul(&a).expect("mul: shape 一致");
    let relued = multiplied.relu();
    let exped = relued.exp();
    exped.tanh()
}

/// 受入基準 3-1: 既定構築（融合経路）と無引数構築（`NaiveOps`・非融合
/// per-op 経路）が REQ-2 複合判定で一致する。
#[test]
fn default_facade_tape_matches_naive_reference_within_parity() {
    let fused_tape = fandhe_ai::tape();
    let fused_result = run_chain_on(&fused_tape);
    let fused_value = fused_result.to_tensor();

    let naive_tape = Tape::new();
    let naive_result = run_chain_on(&naive_tape);
    let naive_value = naive_result.to_tensor();

    assert_parity(
        "fandhe_ai::tape()（融合経路）vs fandhe_ai_autodiff::Tape::new()（NaiveOps 非融合参照）",
        fused_value
            .as_slice()
            .expect("fused_value は contiguous のはず"),
        naive_value
            .as_slice()
            .expect("naive_value は contiguous のはず"),
    );
}

/// 受入基準 3-2: facade が結線する `CpuBackendOps` の `run_fused` が
/// 代表的な `FusionPlan`（5 段連鎖）に対し `Ok` を返す（融合が実際に
/// 有効であることの直接観測）。
#[test]
fn cpu_backend_ops_run_fused_supports_representative_chain() {
    // `FusedOpKind::Input` 2 件（leaf_index 0・1）に続けて
    // add(0,1)→idx2・mul(2,0)→idx3・relu(3)→idx4・exp(4)→idx5・
    // tanh(5)→idx6 の 5 段連鎖を表す `FusionPlan` を構築する
    // （`FusionPlan::from_ops` の不変条件: 入力は常に自ノードより
    // 小さい index。`crates/tensor-core/src/fusion/plan.rs` 参照）。
    let ops = vec![
        FusedOpKind::Input { leaf_index: 0 }, // idx0 = a
        FusedOpKind::Input { leaf_index: 1 }, // idx1 = b
        FusedOpKind::Add { lhs: 0, rhs: 1 },  // idx2 = a + b
        FusedOpKind::Mul { lhs: 2, rhs: 0 },  // idx3 = (a+b) * a
        FusedOpKind::Relu { input: 3 },       // idx4 = relu(idx3)
        FusedOpKind::Exp { input: 4 },        // idx5 = exp(idx4)
        FusedOpKind::Tanh { input: 5 },       // idx6 = tanh(idx5)
    ];
    let plan = FusionPlan::from_ops(ops, vec![2, 2], DType::F32, 2)
        .expect("代表的な 5 段連鎖の FusionPlan は構築成功するはず");

    let a = leaf_a();
    let b = leaf_b();
    let cpu_ops = CpuBackendOps::new();

    let result = cpu_ops.run_fused(&plan, &[&a, &b]);

    assert!(
        result.is_ok(),
        "facade が結線する CpuBackendOps::run_fused は代表的な FusionPlan に対し \
         Ok を返すはず（Unsupported ではない）: {result:?}"
    );
}
