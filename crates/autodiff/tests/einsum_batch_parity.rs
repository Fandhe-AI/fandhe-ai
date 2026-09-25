//! `fandhe_ai_autodiff::einsum_batch::einsum_batched`（イシュー #2149・
//! 親 #2131）の end-to-end 統合テスト。`tests/einsum.rs`（イシュー
//! #1620）と同じブルートフォース n 次元参照実装・REQ-2 統一複合判定
//! ヘルパー・中央差分ヘルパー・許容誤差定数を独立に持つ（`autodiff`
//! クレートの各テストバイナリは別プロセスでコンパイルされ、`tests/
//! einsum.rs` 側の非公開ヘルパーを直接 import できないため。新しい
//! 許容誤差は導入しない方針で同一定数を複製する
//! `.claude/rules/coding-rust.md`）。
//!
//! - forward の突合: `bij,bjk->bik`（単一 batch 添字）・
//!   `abij,abjk->abik`（複数 batch 添字）・`ibj,bjk->bik`（batch 軸が
//!   先頭にない・非恒等 permute 経由）・`bij,bjk->kbi`（出力の並べ替え）・
//!   `bi,bi->b`（left／right 空）・`bijk,bjkl->bil`（多軸 contract）・
//!   `bijx,bjk->bik`（presum を併用）。
//! - `bij,bjk->bik` が rank-3 `Var::matmul` 直接呼び出しと forward・
//!   backward とも bit 同一であること（恒等 permute・shape 不変
//!   reshape のスキップにより `MatMul` ノード 1 個に帰着する契約。
//!   `crate::einsum::einsum_matmul_path` doc 参照）。
//! - 中央差分との勾配突合（`bij,bjk->bik`・`ibj,bjk->bik` の 2 系）。
//! - `Tape::checkpoint` 区間内の batch einsum の勾配が checkpoint
//!   なしと bit 同一であること。
//! - 境界: contract 次元サイズ 0（K=0）・batch 次元サイズ 0・
//!   batch 添字の次元サイズ不一致（`InvalidArgument`）。いずれも
//!   panic しないことを確認する。
//! - `Var::einsum`（facade 公開入口）は同じ spec で引き続き
//!   `InvalidArgument` を返し、tape を壊さないこと（保留契約の固定）。
//! - `Tape::backward_create_graph` 下で batch einsum を使うと型付き
//!   エラー（`AutodiffError::Backward`）になること（panic しないこと
//!   の確認。`crate::create_graph::validate_ancestors` が rank≥3
//!   `MatMul` を拒否する既存契約）。

mod common;

use std::collections::HashMap;

use fandhe_ai_autodiff::einsum_batch::einsum_batched;
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// 多次元 index（row-major）を 1 つインクリメントする（最後方軸から
/// 繰り上げ）。`tests/einsum.rs::increment_index` と同一実装。
fn increment_index(idx: &mut [usize], shape: &[usize]) {
    for axis in (0..shape.len()).rev() {
        idx[axis] += 1;
        if idx[axis] < shape[axis] {
            return;
        }
        idx[axis] = 0;
    }
}

/// ブルートフォース n 次元 einsum 参照実装（`tests/einsum.rs::
/// brute_force_einsum` と同一実装。独立検証のため複製）。`spec` は
/// `"->"` 明示形式のみ受け付ける。
fn brute_force_einsum(spec: &str, operands: &[&Tensor<f32>]) -> Tensor<f32> {
    let (lhs, rhs) = spec
        .split_once("->")
        .expect("test fixture: brute_force_einsum は明示 '->' 形式のみ対応");
    let in_labels: Vec<Vec<char>> = lhs.split(',').map(|s| s.chars().collect()).collect();
    let out_labels: Vec<char> = rhs.chars().collect();
    assert_eq!(in_labels.len(), operands.len());

    let mut dim_of: HashMap<char, usize> = HashMap::new();
    for (labels, op) in in_labels.iter().zip(operands.iter()) {
        for (&c, &sz) in labels.iter().zip(op.shape().iter()) {
            dim_of.insert(c, sz);
        }
    }

    let mut all_labels: Vec<char> = dim_of.keys().copied().collect();
    all_labels.sort_unstable();
    let sum_labels: Vec<char> = all_labels
        .iter()
        .copied()
        .filter(|c| !out_labels.contains(c))
        .collect();

    let out_shape: Vec<usize> = out_labels.iter().map(|c| dim_of[c]).collect();
    let out_numel: usize = out_shape.iter().product();
    let sum_shape: Vec<usize> = sum_labels.iter().map(|c| dim_of[c]).collect();
    let sum_numel: usize = sum_shape.iter().product();

    let mut out_data = vec![0f32; out_numel];
    let mut out_idx = vec![0usize; out_labels.len()];
    for out_flat in out_data.iter_mut() {
        let mut assign: HashMap<char, usize> = HashMap::new();
        for (&c, &v) in out_labels.iter().zip(out_idx.iter()) {
            assign.insert(c, v);
        }

        let mut acc = 0f64;
        let mut sum_idx = vec![0usize; sum_labels.len()];
        for _ in 0..sum_numel {
            for (&c, &v) in sum_labels.iter().zip(sum_idx.iter()) {
                assign.insert(c, v);
            }
            let mut product = 1f64;
            for (labels, op) in in_labels.iter().zip(operands.iter()) {
                let idx: Vec<usize> = labels.iter().map(|c| assign[c]).collect();
                let v = op
                    .get(&idx)
                    .expect("brute_force_einsum: 添字組み立てロジックが範囲外を生成した");
                product *= v as f64;
            }
            acc += product;
            increment_index(&mut sum_idx, &sum_shape);
        }
        *out_flat = acc as f32;

        increment_index(&mut out_idx, &out_shape);
    }

    if out_shape.is_empty() {
        t(out_data, &[])
    } else {
        t(out_data, &out_shape)
    }
}

/// REQ-2 統一複合判定（`tests/einsum.rs::assert_req2_close` と同一
/// 実装。`common::req2_close` へ委譲）。
fn assert_req2_close(label: &str, actual: &Tensor<f32>, expected: &Tensor<f32>) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label}: shape が一致しない"
    );
    let shape = actual.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..numel.max(1) {
        let av = actual.get(&idx).unwrap_or(0.0);
        let ev = expected.get(&idx).unwrap_or(0.0);
        assert!(
            common::req2_close(av as f64, ev as f64),
            "{label}[{idx:?}]: actual={av} expected={ev}"
        );
        if shape.is_empty() {
            break;
        }
        increment_index(&mut idx, &shape);
    }
}

fn assert_bit_identical(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape が一致しない");
    let shape = a.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..numel.max(1) {
        let av = a.get(&idx).unwrap_or(0.0);
        let bv = b.get(&idx).unwrap_or(0.0);
        assert_eq!(
            av.to_bits(),
            bv.to_bits(),
            "{label}[{idx:?}]: a={av} b={bv}（bit 同一でない）"
        );
        if shape.is_empty() {
            break;
        }
        increment_index(&mut idx, &shape);
    }
}

// --- forward: ブルートフォース参照実装との突合 --------------------

#[test]
fn forward_matches_brute_force_single_batch_axis() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(
        (0..24).map(|i| i as f32 * 0.1 - 1.0).collect(),
        &[2, 3, 4],
    ));
    let b = tape.var(&t(
        (0..40).map(|i| i as f32 * 0.05 - 0.5).collect(),
        &[2, 4, 5],
    ));
    let expected = brute_force_einsum("bij,bjk->bik", &[&a.to_tensor(), &b.to_tensor()]);
    let out = einsum_batched("bij,bjk->bik", &[&a, &b]).unwrap();
    assert_req2_close("single_batch_axis", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_multiple_batch_axes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(
        (0..2 * 2 * 3 * 4).map(|i| i as f32 * 0.05 - 1.0).collect(),
        &[2, 2, 3, 4],
    ));
    let b = tape.var(&t(
        (0..2 * 2 * 4 * 5).map(|i| i as f32 * 0.03 - 0.5).collect(),
        &[2, 2, 4, 5],
    ));
    let expected = brute_force_einsum("abij,abjk->abik", &[&a.to_tensor(), &b.to_tensor()]);
    let out = einsum_batched("abij,abjk->abik", &[&a, &b]).unwrap();
    assert_req2_close("multiple_batch_axes", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_batch_axis_not_leading() {
    // batch 軸（b）が a 側で先頭にない（i, b, j の順）ため、
    // einsum_matmul_path の permute が非恒等になり Op::Contiguous を
    // 経由する。
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(
        (0..24).map(|i| i as f32 * 0.1 - 1.0).collect(),
        &[3, 2, 4],
    ));
    let b = tape.var(&t(
        (0..40).map(|i| i as f32 * 0.05 - 0.5).collect(),
        &[2, 4, 5],
    ));
    let expected = brute_force_einsum("ibj,bjk->bik", &[&a.to_tensor(), &b.to_tensor()]);
    let out = einsum_batched("ibj,bjk->bik", &[&a, &b]).unwrap();
    assert_req2_close("batch_axis_not_leading", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_reordered_output() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(
        (0..24).map(|i| i as f32 * 0.1 - 1.0).collect(),
        &[2, 3, 4],
    ));
    let b = tape.var(&t(
        (0..40).map(|i| i as f32 * 0.05 - 0.5).collect(),
        &[2, 4, 5],
    ));
    let expected = brute_force_einsum("bij,bjk->kbi", &[&a.to_tensor(), &b.to_tensor()]);
    let out = einsum_batched("bij,bjk->kbi", &[&a, &b]).unwrap();
    assert_req2_close("reordered_output", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_empty_left_and_right() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // b（batch・出力にも残る）=2、i（contract・両辺に現れ出力にはない）
    // =3。left／right とも空になる境界ケース。
    let a = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]));
    let b = tape.var(&t(vec![0.5, 1.0, -1.0, 2.0, -0.5, 1.5], &[2, 3]));
    let expected = brute_force_einsum("bi,bi->b", &[&a.to_tensor(), &b.to_tensor()]);
    let out = einsum_batched("bi,bi->b", &[&a, &b]).unwrap();
    assert_req2_close("empty_left_and_right", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_multi_axis_contract() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(
        (0..2 * 3 * 4 * 5).map(|i| i as f32 * 0.02 - 0.5).collect(),
        &[2, 3, 4, 5],
    ));
    let b = tape.var(&t(
        (0..2 * 4 * 5 * 6).map(|i| i as f32 * 0.015 - 0.3).collect(),
        &[2, 4, 5, 6],
    ));
    let expected = brute_force_einsum("bijk,bjkl->bil", &[&a.to_tensor(), &b.to_tensor()]);
    let out = einsum_batched("bijk,bjkl->bil", &[&a, &b]).unwrap();
    assert_req2_close("multi_axis_contract", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_with_presum() {
    // x は a のみに現れる添字（presum で先に除去される）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(
        (0..2 * 3 * 4 * 6).map(|i| i as f32 * 0.02 - 0.5).collect(),
        &[2, 3, 4, 6],
    ));
    let b = tape.var(&t(
        (0..40).map(|i| i as f32 * 0.05 - 0.5).collect(),
        &[2, 4, 5],
    ));
    let expected = brute_force_einsum("bijx,bjk->bik", &[&a.to_tensor(), &b.to_tensor()]);
    let out = einsum_batched("bijx,bjk->bik", &[&a, &b]).unwrap();
    assert_req2_close("with_presum", &out.to_tensor(), &expected);
}

// --- bit 同一性: rank-3 Var::matmul 直接呼び出しとの突合 -----------

#[test]
fn single_batch_axis_matches_var_matmul_directly_forward_and_backward() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(
        (0..24).map(|i| i as f32 * 0.1 - 1.0).collect(),
        &[2, 3, 4],
    ));
    let b = tape.var(&t(
        (0..40).map(|i| i as f32 * 0.05 - 0.5).collect(),
        &[2, 4, 5],
    ));

    // ノード数（tape へ push される `Op::MatMul` が 1 個のみである
    // こと）の検証は `crate::einsum::tests::
    // einsum_with_allow_accepts_batch_contraction_as_single_matmul_node`
    // （`tape.nodes` が `pub(crate)` のため単体テスト限定。統合テスト
    // からはアクセスできない）で行う。本テストは bit 同一性のみを
    // 検証する。
    let out = einsum_batched("bij,bjk->bik", &[&a, &b]).unwrap();

    let direct = a.matmul(&b).unwrap();
    assert_bit_identical("forward", &out.to_tensor(), &direct.to_tensor());

    let loss_out = out.sum(None).unwrap();
    let loss_direct = direct.sum(None).unwrap();
    let grads_out = tape.backward(&loss_out).unwrap();
    let grads_direct = tape.backward(&loss_direct).unwrap();
    let da_out = grads_out.get(&a).unwrap().cloned().unwrap();
    let da_direct = grads_direct.get(&a).unwrap().cloned().unwrap();
    assert_bit_identical("da", &da_out, &da_direct);
    let db_out = grads_out.get(&b).unwrap().cloned().unwrap();
    let db_direct = grads_direct.get(&b).unwrap().cloned().unwrap();
    assert_bit_identical("db", &db_out, &db_direct);
}

// --- backward: 中央差分との突合 -------------------------------------
//
// `tests/backward.rs`／`tests/einsum.rs` と同一の許容誤差定数を用いる
// （新しい許容誤差を導入しない方針。`.claude/rules/coding-rust.md`）。

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
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..numel.max(1) {
        let av = analytic.get(&idx).unwrap_or(0.0);
        let nv = numeric.get(&idx).unwrap_or(0.0);
        let diff = (av - nv).abs();
        let rel = diff / av.abs().max(nv.abs()).max(TAU);
        assert!(
            rel <= REL_TOL || diff <= ABS_TOL,
            "{label}[{idx:?}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
        );
        if shape.is_empty() {
            break;
        }
        increment_index(&mut idx, &shape);
    }
}

fn forward_einsum_batch_loss_sum(spec: &str, operands: &[&Tensor<f32>]) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars: Vec<Var<'_>> = operands.iter().map(|t| tape.var(t)).collect();
    let refs: Vec<&Var<'_>> = vars.iter().collect();
    let out = einsum_batched(spec, &refs).unwrap();
    let loss = out.sum(None).unwrap();
    loss.to_tensor()
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

fn numeric_grad(target: &Tensor<f32>, perturb: impl Fn(&Tensor<f32>) -> f32) -> Tensor<f32> {
    let shape = target.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut data: Vec<f32> = (0..numel)
        .map(|flat| {
            let mut idx = vec![0usize; shape.len()];
            let mut rem = flat;
            for axis in (0..shape.len()).rev() {
                idx[axis] = rem % shape[axis];
                rem /= shape[axis];
            }
            target.get(&idx).unwrap_or(0.0)
        })
        .collect();
    let mut grad = vec![0f32; numel];
    for i in 0..numel {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = perturb(&t(data.clone(), &shape)) as f64;
        data[i] = (orig - H) as f32;
        let lm = perturb(&t(data.clone(), &shape)) as f64;
        data[i] = orig as f32;
        grad[i] = ((lp - lm) / (2.0 * H)) as f32;
    }
    t(grad, &shape)
}

#[test]
fn backward_matches_numeric_grad_single_batch_axis() {
    let a_t = t((0..24).map(|i| i as f32 * 0.1 - 1.0).collect(), &[2, 3, 4]);
    let b_t = t((0..40).map(|i| i as f32 * 0.05 - 0.5).collect(), &[2, 4, 5]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&a_t);
    let b = tape.var(&b_t);
    let out = einsum_batched("bij,bjk->bik", &[&a, &b]).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().cloned().unwrap();
    let db = grads.get(&b).unwrap().cloned().unwrap();

    let num_da = numeric_grad(&a_t, |pa| {
        forward_einsum_batch_loss_sum("bij,bjk->bik", &[pa, &b_t])
    });
    let num_db = numeric_grad(&b_t, |pb| {
        forward_einsum_batch_loss_sum("bij,bjk->bik", &[&a_t, pb])
    });
    assert_grad_close("da_single_batch_axis", &da, &num_da);
    assert_grad_close("db_single_batch_axis", &db, &num_db);
}

#[test]
fn backward_matches_numeric_grad_batch_axis_not_leading() {
    let a_t = t((0..24).map(|i| i as f32 * 0.1 - 1.0).collect(), &[3, 2, 4]);
    let b_t = t((0..40).map(|i| i as f32 * 0.05 - 0.5).collect(), &[2, 4, 5]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&a_t);
    let b = tape.var(&b_t);
    let out = einsum_batched("ibj,bjk->bik", &[&a, &b]).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().cloned().unwrap();
    let db = grads.get(&b).unwrap().cloned().unwrap();

    let num_da = numeric_grad(&a_t, |pa| {
        forward_einsum_batch_loss_sum("ibj,bjk->bik", &[pa, &b_t])
    });
    let num_db = numeric_grad(&b_t, |pb| {
        forward_einsum_batch_loss_sum("ibj,bjk->bik", &[&a_t, pb])
    });
    assert_grad_close("da_batch_axis_not_leading", &da, &num_da);
    assert_grad_close("db_batch_axis_not_leading", &db, &num_db);
}

// --- activation checkpointing（#1624）との相互作用 --------------------

fn checkpoint_einsum_batch_compute<'t>(
    av: &Var<'t>,
    bv: &Var<'t>,
) -> Result<Var<'t>, AutodiffError> {
    let out = einsum_batched("bij,bjk->bik", &[av, bv])?;
    Ok(out.sigmoid())
}

#[test]
fn checkpoint_region_with_batch_einsum_matches_no_checkpoint() {
    let a_t = t((0..24).map(|i| i as f32 * 0.1 - 1.0).collect(), &[2, 3, 4]);
    let b_t = t((0..40).map(|i| i as f32 * 0.05 - 0.5).collect(), &[2, 4, 5]);

    let run = |use_checkpoint: bool| -> Tensor<f32> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let av = tape.var(&a_t);
        let bv = tape.var(&b_t);
        let out = if use_checkpoint {
            tape.checkpoint(|| checkpoint_einsum_batch_compute(&av, &bv))
                .unwrap()
        } else {
            checkpoint_einsum_batch_compute(&av, &bv).unwrap()
        };
        let loss = out.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        grads.get(&av).unwrap().cloned().unwrap()
    };

    let da_plain = run(false);
    let da_ckpt = run(true);
    assert_bit_identical("checkpoint_da", &da_plain, &da_ckpt);
}

// --- 境界系 ------------------------------------------------------------

#[test]
fn zero_contract_dimension_does_not_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![], &[2, 3, 0]));
    let b = tape.var(&t(vec![], &[2, 0, 5]));
    // K=0 は結果が全ゼロになるか型付きエラーになるかのいずれかで
    // よい（`.claude/rules/coding-rust.md` panic 禁止方針）。
    match einsum_batched("bij,bjk->bik", &[&a, &b]) {
        Ok(out) => {
            let tensor = out.to_tensor();
            let shape = tensor.shape();
            assert_eq!(shape, vec![2, 3, 5]);
            let numel: usize = shape.iter().product();
            let mut idx = vec![0usize; 3];
            for _ in 0..numel {
                assert_eq!(tensor.get(&idx).unwrap_or(0.0), 0.0);
                increment_index(&mut idx, shape);
            }
        }
        Err(AutodiffError::InvalidArgument(_) | AutodiffError::Shape(_)) => {}
        Err(other) => panic!("K=0 で想定外のエラー種別: {other:?}"),
    }
}

#[test]
fn zero_batch_dimension_does_not_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![], &[0, 3, 4]));
    let b = tape.var(&t(vec![], &[0, 4, 5]));
    let out = einsum_batched("bij,bjk->bik", &[&a, &b])
        .expect("batch 次元サイズ 0 は空出力として受理される");
    assert_eq!(out.to_tensor().shape(), vec![0, 3, 5]);
}

#[test]
fn mismatched_batch_dimension_size_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![0.0; 2 * 3 * 4], &[2, 3, 4]));
    let b = tape.var(&t(vec![0.0; 3 * 4 * 5], &[3, 4, 5]));
    let result = einsum_batched("bij,bjk->bik", &[&a, &b]);
    assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
}

// --- Var::einsum（facade 公開入口）の保留契約の固定 --------------------

#[test]
fn var_einsum_still_rejects_batch_contraction_after_einsum_batch_addition() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![0.0; 2 * 3 * 4], &[2, 3, 4]));
    let b = tape.var(&t(vec![0.0; 2 * 4 * 5], &[2, 4, 5]));
    // 迷子ノードが残らないこと自体の検証は
    // `crate::einsum::tests::einsum_rejects_batch_contraction_without_
    // pushing_nodes`（単体テスト限定。`tape.nodes` が `pub(crate)`）で
    // 行う。本テストは `Var::einsum` が einsum_batch 追加後も
    // `InvalidArgument` を返し続けることのみを固定する。
    let result = Var::einsum("bij,bjk->bik", &[&a, &b]);
    assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
}

// --- create_graph（高階微分）下での型付きエラー ------------------------

#[test]
fn batch_einsum_under_create_graph_returns_typed_error_not_panic() {
    let parent = Tape::new_with_ops(common::naive_ops());
    let a = parent.var(&t(
        (0..24).map(|i| i as f32 * 0.1 - 1.0).collect(),
        &[2, 3, 4],
    ));
    let b = parent.var(&t(
        (0..40).map(|i| i as f32 * 0.05 - 0.5).collect(),
        &[2, 4, 5],
    ));
    let out = einsum_batched("bij,bjk->bik", &[&a, &b]).unwrap();
    let loss = out.sum(None).unwrap();

    let child = Tape::new_with_ops(common::naive_ops());
    let result = parent.backward_create_graph(&loss, &child);
    assert!(
        matches!(result, Err(AutodiffError::Backward(_))),
        "rank>=3 MatMul は create_graph 非対応のため型付きエラーになるはず: {result:?}"
    );
}
