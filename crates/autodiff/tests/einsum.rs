//! `Var::einsum`（イシュー #1620）の end-to-end 統合テスト。
//!
//! - **ブルートフォース n 次元参照実装**（`brute_force_einsum`。効率は
//!   問わない独立実装）との forward 突合で、`Var::einsum` が正しい
//!   縮約結果を返すことを検証する（判定は REQ-2 統一複合判定「相対
//!   誤差 1e-3 未満 または 絶対誤差 1e-5 未満」の独立再実装
//!   `assert_req2_close` を使う）。
//! - `"ij,jk->ik"` が `Var::matmul` と厳密に同一の値を返すこと（分解
//!   ドライバが恒等 permute／reshape をスキップし `MatMul` ノード
//!   1 個に帰着する契約。`crate::einsum` モジュール doc 参照）。
//! - 中央差分（数値微分）との勾配突合（`H=1e-3`・相対 1e-2 または絶対
//!   1e-3・`.claude/rules/coding-rust.md` の「新しい許容誤差を導入
//!   しない」方針に沿い `tests/backward.rs` と同一の定数を用いる）。
//! - 同一 `Var` を 2 回渡した場合の勾配蓄積（複数経路からの合算）。
//! - 拒否系（ellipsis・3 オペランド以上・batch 添字を伴う縮約・
//!   同一オペランド内添字重複・rank 不一致・次元サイズ不一致・
//!   出力添字が入力に現れない）が panic せず型付きエラーを返すこと。

mod common;

use std::collections::HashMap;

use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

/// 多次元 index（row-major）を 1 つインクリメントする（最後方軸から
/// 繰り上げ）。`shape` が空（スカラー）の場合は何もしない。
fn increment_index(idx: &mut [usize], shape: &[usize]) {
    for axis in (0..shape.len()).rev() {
        idx[axis] += 1;
        if idx[axis] < shape[axis] {
            return;
        }
        idx[axis] = 0;
    }
}

/// ブルートフォース n 次元 einsum 参照実装（`Var::einsum` の分解
/// ロジックとは独立に、素朴な「全添字を総当たりして総和する」方式で
/// 計算する。効率は問わずテストの独立検証にのみ使う）。`spec` は
/// `"->"` 明示形式のみ受け付ける（省略形の検証は `src/einsum.rs` の
/// パーサ単体テストで別途カバー済み）。
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

/// REQ-2 統一複合判定（「相対誤差 1e-3 未満 または 絶対誤差 1e-5
/// 未満」。`.claude/rules/coding-rust.md`）の独立再実装。本体側の
/// 判定関数（`fandhe_ai_backend_cpu::assert_parity` 等）は使わず、
/// このテストファイル内で完結させる。
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
        let diff = (av - ev).abs();
        let rel = diff / ev.abs().max(1e-30);
        assert!(
            diff < 1e-5 || rel < 1e-3,
            "{label}[{idx:?}]: actual={av} expected={ev} diff={diff} rel={rel}"
        );
        if shape.is_empty() {
            break;
        }
        increment_index(&mut idx, &shape);
    }
}

// --- forward: ブルートフォース参照実装との突合 --------------------

#[test]
fn forward_matches_brute_force_matmul() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, -3.0, 0.5, 4.0, -1.5], &[2, 3]));
    let b = tape.var(&t(vec![1.0, -1.0, 2.0, 0.5, -2.0, 1.0], &[3, 2]));
    let expected = brute_force_einsum("ij,jk->ik", &[&a.to_tensor(), &b.to_tensor()]);
    let out = Var::einsum("ij,jk->ik", &[&a, &b]).unwrap();
    assert_req2_close("matmul", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_transpose_unary() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let expected = brute_force_einsum("ij->ji", &[&a.to_tensor()]);
    let out = Var::einsum("ij->ji", &[&a]).unwrap();
    assert_req2_close("transpose", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_full_sum_unary() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]));
    let expected = brute_force_einsum("ij->", &[&a.to_tensor()]);
    let out = Var::einsum("ij->", &[&a]).unwrap();
    assert_req2_close("full_sum", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_outer_product() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, -2.0, 3.0], &[3]));
    let b = tape.var(&t(vec![0.5, -1.5], &[2]));
    let expected = brute_force_einsum("i,j->ij", &[&a.to_tensor(), &b.to_tensor()]);
    let out = Var::einsum("i,j->ij", &[&a, &b]).unwrap();
    assert_req2_close("outer_product", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_elementwise_dot() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]));
    let b = tape.var(&t(vec![0.5, 1.0, -1.0, 2.0], &[2, 2]));
    let expected = brute_force_einsum("ij,ij->", &[&a.to_tensor(), &b.to_tensor()]);
    let out = Var::einsum("ij,ij->", &[&a, &b]).unwrap();
    assert_req2_close("elementwise_dot", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_left_multiple_axes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(
        (0..24).map(|i| i as f32 * 0.1 - 1.0).collect(),
        &[2, 3, 4],
    ));
    let b = tape.var(&t((0..8).map(|i| i as f32 * 0.2 - 0.5).collect(), &[4, 2]));
    let expected = brute_force_einsum("ijk,kl->ijl", &[&a.to_tensor(), &b.to_tensor()]);
    let out = Var::einsum("ijk,kl->ijl", &[&a, &b]).unwrap();
    assert_req2_close("left_multiple_axes", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_nt_layout() {
    // "ij,kj->ik" は右オペランドが転置扱い（NT 相当）になる縮約。
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, -1.0, 0.5, 3.0, -2.0], &[2, 3]));
    let b = tape.var(&t(vec![1.0, -1.0, 0.5, 2.0, 0.0, -0.5], &[2, 3]));
    let expected = brute_force_einsum("ij,kj->ik", &[&a.to_tensor(), &b.to_tensor()]);
    let out = Var::einsum("ij,kj->ik", &[&a, &b]).unwrap();
    assert_req2_close("nt_layout", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_presum_binary() {
    // "ijk,j->i" は b にのみ現れる j が縮約対象で、a の k は presum
    // （相手にも出力にも現れない）で先に除去される。
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(
        (0..24).map(|i| i as f32 * 0.05 - 0.5).collect(),
        &[2, 3, 4],
    ));
    let b = tape.var(&t(vec![1.0, -1.0, 0.5], &[3]));
    let expected = brute_force_einsum("ijk,j->i", &[&a.to_tensor(), &b.to_tensor()]);
    let out = Var::einsum("ijk,j->i", &[&a, &b]).unwrap();
    assert_req2_close("presum_binary", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_unary_presum_and_permute() {
    // "ijk->ki" は j を presum で除去したのち、残った i/k を出力順
    // （k, i）へ permute する。
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(
        (0..24).map(|i| i as f32 * 0.03 - 0.3).collect(),
        &[2, 3, 4],
    ));
    let expected = brute_force_einsum("ijk->ki", &[&a.to_tensor()]);
    let out = Var::einsum("ijk->ki", &[&a]).unwrap();
    assert_req2_close("unary_presum_permute", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_mul_path_all_nonempty() {
    // "ij,ik->ijk" は batch=[i]・left=[j]・right=[k] が全て非空になる
    // mul 経路（broadcast 外積）のケース。
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]));
    let b = tape.var(&t(vec![1.0, -1.0, 0.5, -0.5], &[2, 2]));
    let expected = brute_force_einsum("ij,ik->ijk", &[&a.to_tensor(), &b.to_tensor()]);
    let out = Var::einsum("ij,ik->ijk", &[&a, &b]).unwrap();
    assert_req2_close("mul_path_all_nonempty", &out.to_tensor(), &expected);
}

// --- "ij,jk->ik" と Var::matmul の bit 同一 -------------------------

#[test]
fn matmul_spec_matches_var_matmul_exactly() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, -3.0, 0.5, 4.0, -1.5], &[2, 3]));
    let b = tape.var(&t(vec![1.0, -1.0, 2.0, 0.5, -2.0, 1.0], &[3, 2]));

    let via_einsum = Var::einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let via_matmul = a.matmul(&b).unwrap();

    let ev = via_einsum.to_tensor();
    let mv = via_matmul.to_tensor();
    assert_eq!(ev.shape(), mv.shape());
    let numel: usize = ev.shape().iter().product();
    let mut idx = vec![0usize; ev.shape().len()];
    for _ in 0..numel {
        assert_eq!(
            ev.get(&idx),
            mv.get(&idx),
            "ij,jk->ik は Var::matmul と bit 同一であるべき（分解ドライバが恒等 \
             permute／reshape をスキップし MatMul ノード 1 個に帰着する契約）"
        );
        increment_index(&mut idx, ev.shape());
    }
}

// --- backward: 中央差分との突合 -------------------------------------
//
// `tests/backward.rs` と同一の許容誤差定数を用いる（新しい許容誤差を
// 導入しない方針。`.claude/rules/coding-rust.md`）。

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

/// `spec` の einsum を forward し `sum(None)` で合算した `loss` の
/// スカラー値を返す（新規 `Tape` を都度構築する使い捨てパターン。
/// `tests/backward.rs::forward_loss` と同方針）。
fn forward_einsum_loss_sum(spec: &str, operands: &[&Tensor<f32>]) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars: Vec<Var<'_>> = operands.iter().map(|t| tape.var(t)).collect();
    let refs: Vec<&Var<'_>> = vars.iter().collect();
    let out = Var::einsum(spec, &refs).unwrap();
    // 出力が非スカラーの場合は全要素和でスカラー化する
    // （`Var::sum(None)` は既にスカラーの場合も恒等的に動作する）。
    let loss = out.sum(None).unwrap();
    scalar(&loss.to_tensor())
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
fn backward_matches_numeric_grad_matmul_spec() {
    let a_t = t(vec![1.0, 2.0, -3.0, 0.5, 4.0, -1.5], &[2, 3]);
    let b_t = t(vec![1.0, -1.0, 2.0, 0.5, -2.0, 1.0], &[3, 2]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&a_t);
    let b = tape.var(&b_t);
    let out = Var::einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads
        .get(&a)
        .unwrap()
        .expect("a は loss に到達する")
        .clone();
    let db = grads
        .get(&b)
        .unwrap()
        .expect("b は loss に到達する")
        .clone();

    let num_da = numeric_grad(&a_t, |pa| forward_einsum_loss_sum("ij,jk->ik", &[pa, &b_t]));
    let num_db = numeric_grad(&b_t, |pb| forward_einsum_loss_sum("ij,jk->ik", &[&a_t, pb]));
    assert_grad_close("da_matmul", &da, &num_da);
    assert_grad_close("db_matmul", &db, &num_db);
}

#[test]
fn backward_matches_numeric_grad_left_multiple_axes() {
    let a_t = t((0..24).map(|i| i as f32 * 0.1 - 1.0).collect(), &[2, 3, 4]);
    let b_t = t((0..8).map(|i| i as f32 * 0.2 - 0.5).collect(), &[4, 2]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&a_t);
    let b = tape.var(&b_t);
    let out = Var::einsum("ijk,kl->ijl", &[&a, &b]).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads
        .get(&a)
        .unwrap()
        .expect("a は loss に到達する")
        .clone();
    let db = grads
        .get(&b)
        .unwrap()
        .expect("b は loss に到達する")
        .clone();

    let num_da = numeric_grad(&a_t, |pa| {
        forward_einsum_loss_sum("ijk,kl->ijl", &[pa, &b_t])
    });
    let num_db = numeric_grad(&b_t, |pb| {
        forward_einsum_loss_sum("ijk,kl->ijl", &[&a_t, pb])
    });
    assert_grad_close("da_left_multiple", &da, &num_da);
    assert_grad_close("db_left_multiple", &db, &num_db);
}

#[test]
fn backward_matches_numeric_grad_mul_path() {
    let a_t = t(vec![1.0, -2.0, 3.0], &[3]);
    let b_t = t(vec![0.5, -1.5], &[2]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&a_t);
    let b = tape.var(&b_t);
    let out = Var::einsum("i,j->ij", &[&a, &b]).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads
        .get(&a)
        .unwrap()
        .expect("a は loss に到達する")
        .clone();
    let db = grads
        .get(&b)
        .unwrap()
        .expect("b は loss に到達する")
        .clone();

    let num_da = numeric_grad(&a_t, |pa| forward_einsum_loss_sum("i,j->ij", &[pa, &b_t]));
    let num_db = numeric_grad(&b_t, |pb| forward_einsum_loss_sum("i,j->ij", &[&a_t, pb]));
    assert_grad_close("da_mul_path", &da, &num_da);
    assert_grad_close("db_mul_path", &db, &num_db);
}

/// 同一 `Var` を 2 回渡した場合の勾配蓄積（複数経路からの合算）を
/// 検証する。`a` を square 行列として両オペランドに渡すことで、
/// `matmul_vjp` の da/db 両方の寄与が同一 `NodeId` へ流入する
/// （`tests/backward.rs::mul_self_reference_accumulates_gradient` と
/// 同型の検証。イシュー #1620）。
#[test]
fn backward_accumulates_gradient_for_repeated_operand() {
    let a_t = t(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&a_t);
    let out = Var::einsum("ij,jk->ik", &[&a, &a]).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads
        .get(&a)
        .unwrap()
        .expect("a は loss に到達する")
        .clone();

    let num_da = numeric_grad(&a_t, |pa| forward_einsum_loss_sum("ij,jk->ik", &[pa, pa]));
    assert_grad_close("da_repeated_operand", &da, &num_da);
}

// --- 拒否系 -----------------------------------------------------------

#[test]
fn rejects_ellipsis() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0], &[2]));
    let result = Var::einsum("...i->...i", &[&a]);
    assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn rejects_three_or_more_operands() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0], &[2]));
    let b = tape.var(&t(vec![1.0, 2.0], &[2]));
    let c = tape.var(&t(vec![1.0, 2.0], &[2]));
    let result = Var::einsum("i,i,i->i", &[&a, &b, &c]);
    assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn rejects_batch_axis_contraction() {
    // batch 添字（b）を伴う縮約は rank>=3 の matmul（#1600）が未実装の
    // ため拒否する（`compute_binary_plan` の判定。モジュール doc
    // 「受理範囲」参照）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![0.0; 2 * 3 * 4], &[2, 3, 4]));
    let b = tape.var(&t(vec![0.0; 2 * 4 * 5], &[2, 4, 5]));
    let result = Var::einsum("bij,bjk->bik", &[&a, &b]);
    assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn rejects_duplicate_axis_within_operand() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let result = Var::einsum("ii->i", &[&a]);
    assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn rejects_rank_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let result = Var::einsum("ijk->ijk", &[&a]);
    assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn rejects_dimension_size_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let b = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    // a の j（サイズ 3）と b の j（サイズ 2）が食い違う。
    let result = Var::einsum("ij,jk->ik", &[&a, &b]);
    assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn rejects_output_axis_not_in_inputs() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0], &[2]));
    let result = Var::einsum("i->k", &[&a]);
    assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn rejects_operand_count_spec_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0], &[2]));
    let b = tape.var(&t(vec![1.0, 2.0], &[2]));
    // spec は 1 オペランド分だが Var を 2 個渡す。
    let result = Var::einsum("i->i", &[&a, &b]);
    assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
}

/// `Err` を返す `einsum` 呼び出しのあとも、同一 `Tape` が正常に動作
/// することを確認する（`Tape` の公開 API にはノード数を直接読む手段が
/// ないため、迷子ノードが致命的な状態を残していないことの代替検証。
/// 完全な「ノード数が増えていない」検査は `src/einsum.rs` の
/// `#[cfg(test)]` 単体テスト〈クレート内部限定〉で行う）。
#[test]
fn tape_remains_usable_after_rejected_einsum_call() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![0.0; 2 * 3 * 4], &[2, 3, 4]));
    let b = tape.var(&t(vec![0.0; 2 * 4 * 5], &[2, 4, 5]));
    let rejected = Var::einsum("bij,bjk->bik", &[&a, &b]);
    assert!(matches!(rejected, Err(AutodiffError::InvalidArgument(_))));

    // 同じ tape 上で通常の演算・backward が問題なく動く。
    let x = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]));
    let y = x.mul(&x).unwrap();
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.get(&[0, 0]).unwrap(), 2.0);
}

// --- activation checkpointing（#1624）との相互作用 --------------------
//
// `Op::Contiguous`（本イシューが新設する唯一の新規 `Op`）は checkpoint
// 非適格（`Op::is_checkpoint_eligible() == false`。`Op::Concat` と同列。
// `docs/autodiff-checkpoint-design.md` §8）だが、checkpoint 区間内に
// 現れても `Tape::checkpoint` がエラーにならず、勾配が非 checkpoint
// 実行と bit 同一になることを確認する（`crates/autodiff/tests/
// checkpoint.rs::checkpoint_region_containing_ineligible_op_does_not_
// error` と同型の検証を einsum 経路で行う）。

fn assert_bit_identical(a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape());
    let shape = a.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..numel.max(1) {
        assert_eq!(
            a.get(&idx),
            b.get(&idx),
            "idx={idx:?}: bit 同一であるべき値が食い違った"
        );
        if shape.is_empty() {
            break;
        }
        increment_index(&mut idx, &shape);
    }
}

/// `"ji,jk->ik"` は a 側の `[left..., contract...]` 目標順（`i, j`）が
/// 現在の添字順（`j, i`）と異なるため実際に `Var::permute`（非恒等）が
/// 発生し、続く `reshape` 前の `contiguous()` が `Op::Contiguous`
/// ノードを実際に 1 個積む（恒等スキップされない唯一のケース。
/// `crate::einsum::einsum_matmul_path` doc 参照）。
fn checkpoint_einsum_compute<'t>(av: &Var<'t>, bv: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let out = Var::einsum("ji,jk->ik", &[av, bv])?;
    Ok(out.sigmoid())
}

#[test]
fn checkpoint_region_with_einsum_contiguous_node_matches_no_checkpoint() {
    let a_t = t(vec![1.0, -0.5, 0.3, 2.0, -1.0, 0.4], &[3, 2]); // [j=3, i=2]
    let b_t = t(
        vec![0.2, -0.3, 0.5, 0.1, -0.4, 0.6, 1.0, -1.0, 0.5],
        &[3, 3],
    ); // [j=3, k=3]

    let run = |use_checkpoint: bool| -> Tensor<f32> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let av = tape.var(&a_t);
        let bv = tape.var(&b_t);
        let out = if use_checkpoint {
            tape.checkpoint(|| checkpoint_einsum_compute(&av, &bv))
                .unwrap()
        } else {
            checkpoint_einsum_compute(&av, &bv).unwrap()
        };
        let loss = out.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        grads.get(&av).unwrap().cloned().unwrap()
    };

    let da_plain = run(false);
    let da_ckpt = run(true);
    assert_bit_identical(&da_plain, &da_ckpt);
}
