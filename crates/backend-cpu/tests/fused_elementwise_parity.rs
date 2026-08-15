//! `backend-cpu::fused_elementwise::run_fused_elementwise` の受け入れ基準
//! 対応テスト（TASK-12.1c・#163）。
//!
//! イシュー #163 の受け入れ条件は「融合カーネルの数値が非融合実行と
//! 一致すること」。本ファイルは PoC-9 相当パターン（ew4・fan-out・
//! fan-in 合流）で、融合カーネル（`run_fused_elementwise`）の結果を
//! per-op メソッド（`backend_cpu::{add, mul, relu, exp, tanh}`。`elementwise.rs`
//! Tensor 入口層）の逐次合成（非融合基準）と突き合わせ、
//! `backend_cpu::parity::assert_parity`（REQ-2 統一複合判定「相対誤差
//! 1e-3 未満 または 絶対誤差 1e-5 未満」）で判定する。**許容誤差の新設・
//! 緩和は行わない**（`.claude/rules/coding-rust.md`）。
//!
//! `FusionPlan` は `tensor_core::FusionPlan::from_ops`（`autodiff` 専用の
//! クレート間構築経路。`pub` + `#[doc(hidden)]`）を直接呼んで構築する
//! （`tensor_core` 内部の `pub(crate)` 型を経由しない。設計書 §3.4）。
//! `backend-cpu` は融合対象セグメントの検出（`detect_fusion`）を
//! 行わないため、本ファイルの各テストは検出済みセグメントを模した
//! `FusedOpKind` 列を直接組み立てる。

use backend_cpu::fused_elementwise::run_fused_elementwise;
use backend_cpu::parity::assert_parity;
use bench_harness::rng::Xorshift64Star;
use tensor_core::{DType, FusedOpKind, FusionPlan, ShapeError, Tensor};

fn seeded_tensor(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).unwrap()
}

/// `[-1, 1)` の一様分布を大きめの値域（`[-1e3, 1e3)`）へスケールした
/// 決定的入力（exp/tanh のオーバーフロー・飽和挙動を突く用途）。
fn seeded_tensor_scaled(seed: u64, shape: &[usize], scale: f32) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let mut rng = Xorshift64Star::new(seed);
    let data: Vec<f32> = (0..numel).map(|_| rng.next_f32() * scale).collect();
    Tensor::new(data, shape).unwrap()
}

// --- ew4: add -> relu -> exp -> tanh（4 段。2 leaves） ---

fn ew4_plan(shape: Vec<usize>) -> FusionPlan {
    // ops: 0=Input(x) 1=Input(y) 2=Add(0,1) 3=Relu(2) 4=Exp(3) 5=Tanh(4)
    FusionPlan::from_ops(
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Input { leaf_index: 1 },
            FusedOpKind::Add { lhs: 0, rhs: 1 },
            FusedOpKind::Relu { input: 2 },
            FusedOpKind::Exp { input: 3 },
            FusedOpKind::Tanh { input: 4 },
        ],
        shape,
        DType::F32,
        2,
    )
    .unwrap()
}

fn ew4_sequential(x: &Tensor<f32>, y: &Tensor<f32>) -> Tensor<f32> {
    let a = backend_cpu::add(x, y).unwrap();
    let b = backend_cpu::relu(&a).unwrap();
    let c = backend_cpu::exp(&b).unwrap();
    backend_cpu::tanh(&c).unwrap()
}

#[test]
fn ew4_matches_sequential_below_parallel_threshold() {
    let shape = [16usize];
    let x = seeded_tensor(1, &shape);
    let y = seeded_tensor(2, &shape);

    let plan = ew4_plan(shape.to_vec());

    let fused = run_fused_elementwise(&plan, &[&x, &y]).unwrap();
    let expected = ew4_sequential(&x, &y);
    assert_parity(
        "ew4 below PARALLEL_THRESHOLD",
        fused.as_slice().unwrap(),
        expected.as_slice().unwrap(),
    );
}

#[test]
fn ew4_matches_sequential_above_parallel_threshold() {
    // PARALLEL_THRESHOLD (1<<15) 以上のサイズで並列経路を確実に踏む。
    let shape = [1usize << 16];
    let x = seeded_tensor_scaled(11, &shape, 4.0);
    let y = seeded_tensor_scaled(12, &shape, 4.0);

    let plan = ew4_plan(shape.to_vec());

    let fused = run_fused_elementwise(&plan, &[&x, &y]).unwrap();
    let expected = ew4_sequential(&x, &y);
    assert_parity(
        "ew4 above PARALLEL_THRESHOLD",
        fused.as_slice().unwrap(),
        expected.as_slice().unwrap(),
    );
}

// --- fan-out: a = x + y; b = a * a; c = b + x; d = relu(c)（4 段・fan-out） ---

#[test]
fn fan_out_matches_sequential() {
    let shape = [64usize];
    let x = seeded_tensor(21, &shape);
    let y = seeded_tensor(22, &shape);

    // ops: 0=Input(x) 1=Input(y) 2=Add(0,1)=a 3=Mul(2,2)=b 4=Add(3,0)=c 5=Relu(4)=d
    let plan = FusionPlan::from_ops(
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Input { leaf_index: 1 },
            FusedOpKind::Add { lhs: 0, rhs: 1 },
            FusedOpKind::Mul { lhs: 2, rhs: 2 },
            FusedOpKind::Add { lhs: 3, rhs: 0 },
            FusedOpKind::Relu { input: 4 },
        ],
        shape.to_vec(),
        DType::F32,
        2,
    )
    .unwrap();

    let fused = run_fused_elementwise(&plan, &[&x, &y]).unwrap();

    let a = backend_cpu::add(&x, &y).unwrap();
    let b = backend_cpu::mul(&a, &a).unwrap();
    let c = backend_cpu::add(&b, &x).unwrap();
    let expected = backend_cpu::relu(&c).unwrap();

    assert_parity(
        "fan-out",
        fused.as_slice().unwrap(),
        expected.as_slice().unwrap(),
    );
}

// --- fan-in 合流: (a+b)*(c+d) の後段に relu を重ねた 4 段 ---

#[test]
fn fan_in_confluence_matches_sequential() {
    let shape = [32usize];
    let a = seeded_tensor(31, &shape);
    let b = seeded_tensor(32, &shape);
    let c = seeded_tensor(33, &shape);
    let d = seeded_tensor(34, &shape);

    // ops: 0=a 1=b 2=c 3=d 4=Add(0,1)=ab 5=Add(2,3)=cd 6=Mul(4,5)=prod 7=Relu(6)=out
    let plan = FusionPlan::from_ops(
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Input { leaf_index: 1 },
            FusedOpKind::Input { leaf_index: 2 },
            FusedOpKind::Input { leaf_index: 3 },
            FusedOpKind::Add { lhs: 0, rhs: 1 },
            FusedOpKind::Add { lhs: 2, rhs: 3 },
            FusedOpKind::Mul { lhs: 4, rhs: 5 },
            FusedOpKind::Relu { input: 6 },
        ],
        shape.to_vec(),
        DType::F32,
        4,
    )
    .unwrap();

    let fused = run_fused_elementwise(&plan, &[&a, &b, &c, &d]).unwrap();

    let ab = backend_cpu::add(&a, &b).unwrap();
    let cd = backend_cpu::add(&c, &d).unwrap();
    let prod = backend_cpu::mul(&ab, &cd).unwrap();
    let expected = backend_cpu::relu(&prod).unwrap();

    assert_parity(
        "fan-in confluence",
        fused.as_slice().unwrap(),
        expected.as_slice().unwrap(),
    );
}

// --- relu(NaN) == 0.0 の per-op カーネルとの挙動一致 ---

#[test]
fn relu_nan_matches_per_op_kernel_behavior() {
    let shape = [4usize];
    let x = Tensor::new(vec![f32::NAN, -1.0, 0.0, 2.5], &shape).unwrap();

    // ops: 0=Input(x) 1=Relu(0)
    let plan = FusionPlan::from_ops(
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Relu { input: 0 },
        ],
        shape.to_vec(),
        DType::F32,
        1,
    )
    .unwrap();

    let fused = run_fused_elementwise(&plan, &[&x]).unwrap();
    let expected = backend_cpu::relu(&x).unwrap();

    let fused_slice = fused.as_slice().unwrap();
    let expected_slice = expected.as_slice().unwrap();
    assert_eq!(fused_slice, expected_slice);
    // per-op カーネル（`elementwise.rs::relu_slice`）は NaN 入力に対し
    // `x.max(0.0)` の Rust 仕様どおり 0.0 を返す（NumPy/PyTorch とは
    // 異なる挙動。`elementwise.rs` モジュール冒頭コメント参照）。
    assert_eq!(fused_slice[0], 0.0);
}

// --- 実行前検証の拒否系 ---

#[test]
fn rejects_leaf_count_mismatch() {
    let shape = [4usize];
    let x = seeded_tensor(41, &shape);
    let plan = FusionPlan::from_ops(
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Relu { input: 0 },
        ],
        shape.to_vec(),
        DType::F32,
        1,
    )
    .unwrap();

    // leaf_count=1 だが 2 個渡す。
    let err = run_fused_elementwise(&plan, &[&x, &x]).unwrap_err();
    match err {
        tensor_core::BackendError::ShapeMismatch(ShapeError::ElementCountMismatch {
            expected,
            actual,
        }) => {
            assert_eq!(expected, 1);
            assert_eq!(actual, 2);
        }
        other => panic!("expected ElementCountMismatch, got {other:?}"),
    }
}

#[test]
fn rejects_leaf_shape_mismatch() {
    let x = seeded_tensor(42, &[4]);
    let plan = FusionPlan::from_ops(
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Relu { input: 0 },
        ],
        vec![8], // leaf の shape [4] と不一致
        DType::F32,
        1,
    )
    .unwrap();

    let err = run_fused_elementwise(&plan, &[&x]).unwrap_err();
    match err {
        tensor_core::BackendError::ShapeMismatch(ShapeError::ShapeMismatch { lhs, rhs }) => {
            assert_eq!(lhs, vec![8]);
            assert_eq!(rhs, vec![4]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn rejects_non_contiguous_leaf() {
    // 2x3 行列を転置した view（非 contiguous）を leaf として渡す。
    let base = seeded_tensor(43, &[2, 3]);
    let transposed = base.transpose_2d().unwrap();
    assert!(!transposed.is_contiguous());

    let plan = FusionPlan::from_ops(
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Relu { input: 0 },
        ],
        vec![3, 2],
        DType::F32,
        1,
    )
    .unwrap();

    let err = run_fused_elementwise(&plan, &[&transposed]).unwrap_err();
    assert!(matches!(err, tensor_core::BackendError::Unsupported(_)));
}

// --- #586: reduction（Sum/Max）・Rsqrt を含む FusionPlan の pre-scan 拒否 ---
//
// `tensor_core::fusion` の境界再定義（イシュー #586）により `FusionPlan`
// は Sum／Max／Rsqrt を含みうるが、対応する CPU カーネル実装は本イシュー
// のスコープ外（後続 G-3 以降）。`run_fused_elementwise` が pre-scan で
// fail-closed に拒否し、`eval_one` の到達不能 arm へ落ちて panic したり
// 静かに誤った値を返したりしないことを固定する（`.claude/rules/
// coding-rust.md`・`security.md` A04）。

#[test]
fn rejects_plan_containing_sum() {
    let x = seeded_tensor(50, &[4]);
    let plan = FusionPlan::from_ops(
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Relu { input: 0 },
            FusedOpKind::Sum {
                input: 1,
                axis: Some(0),
            },
        ],
        vec![4],
        DType::F32,
        1,
    )
    .unwrap();

    let err = run_fused_elementwise(&plan, &[&x]).unwrap_err();
    assert!(matches!(err, tensor_core::BackendError::Unsupported(_)));
}

#[test]
fn rejects_plan_containing_max() {
    let x = seeded_tensor(51, &[4]);
    let plan = FusionPlan::from_ops(
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Relu { input: 0 },
            FusedOpKind::Max {
                input: 1,
                axis: None,
            },
        ],
        vec![4],
        DType::F32,
        1,
    )
    .unwrap();

    let err = run_fused_elementwise(&plan, &[&x]).unwrap_err();
    assert!(matches!(err, tensor_core::BackendError::Unsupported(_)));
}

#[test]
fn rejects_plan_containing_rsqrt() {
    let x = seeded_tensor(52, &[4]);
    let plan = FusionPlan::from_ops(
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Rsqrt { input: 0 },
        ],
        vec![4],
        DType::F32,
        1,
    )
    .unwrap();

    let err = run_fused_elementwise(&plan, &[&x]).unwrap_err();
    assert!(matches!(err, tensor_core::BackendError::Unsupported(_)));
}
