//! `Var::matmul` のバッチ次元対応（イシュー #1715。親 #1600。spec REQ-9
//! 2026-09-12 追記 Tier 1「バッチ行列積」）の受け入れ条件対応テスト。
//!
//! - rank≥3 の `Var::matmul` forward が per-batch 2 次元 `matmul`（バッチ
//!   をほどいて個別に呼んだもの）と bit 完全一致すること。
//! - バッチ次元ブロードキャスト（lhs／rhs いずれかのバッチ 1）を
//!   forward で正しく処理すること。
//! - rank≥3 `matmul` チェーンの backward（VJP）が中央差分（数値微分）と
//!   一致すること（等バッチ・broadcast の両方）。
//! - `Tape::checkpoint` 区間に rank≥3 `matmul` を含めても、checkpoint
//!   なしの場合と勾配が bit 完全一致すること（`Op::MatMul` の
//!   checkpoint 再計算〈`tape.rs::recompute_value`〉が
//!   `matmul_forward` 経由でバッチ分岐することの回帰確認）。

mod common;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    let c = tensor.contiguous();
    c.as_slice().map(|s| s.to_vec()).unwrap_or_default()
}

fn seq(seed: u64, len: usize) -> Vec<f32> {
    // 決定的な擬似乱数列（`Xorshift64Star` 相当を使わず本ファイル内で
    // 完結させるための簡易 LCG）。値域は概ね [-1, 1)。
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state % 2000) as f32 - 1000.0) / 1000.0
        })
        .collect()
}

// --- 1. forward: per-batch 2 次元 matmul との bit 完全一致 ---

#[test]
fn matmul_batched_forward_matches_per_batch_2d_bit_exact() {
    let (b, m, k, n) = (3usize, 2usize, 3usize, 4usize);
    let a_data = seq(1, b * m * k);
    let b_data = seq(2, b * k * n);
    let a = t(a_data.clone(), &[b, m, k]);
    let bb = t(b_data.clone(), &[b, k, n]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let av = tape.var(&a);
    let bv = tape.var(&bb);
    let out = av.matmul(&bv).unwrap();
    let out_data = dense(&out.to_tensor());
    assert_eq!(out.to_tensor().shape(), &[b, m, n]);

    for i in 0..b {
        let a_i = t(a_data[i * m * k..(i + 1) * m * k].to_vec(), &[m, k]);
        let b_i = t(b_data[i * k * n..(i + 1) * k * n].to_vec(), &[k, n]);
        let tape2 = Tape::new_with_ops(common::naive_ops());
        let a2 = tape2.var(&a_i);
        let b2 = tape2.var(&b_i);
        let expected = dense(&a2.matmul(&b2).unwrap().to_tensor());
        let got = &out_data[i * m * n..(i + 1) * m * n];
        assert_eq!(got, expected.as_slice(), "batch {i} forward mismatch");
    }
}

/// lhs バッチ次元 1（NumPy 互換ブロードキャスト）が forward で正しく
/// 複製されることを確認する。
#[test]
fn matmul_batched_forward_broadcasts_lhs_batch_dim_one() {
    let (b, m, k, n) = (4usize, 2usize, 3usize, 2usize);
    let a_data = seq(3, m * k);
    let b_data = seq(4, b * k * n);
    let a = t(a_data.clone(), &[1, m, k]);
    let bb = t(b_data.clone(), &[b, k, n]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let av = tape.var(&a);
    let bv = tape.var(&bb);
    let out = av.matmul(&bv).unwrap();
    let out_data = dense(&out.to_tensor());
    assert_eq!(out.to_tensor().shape(), &[b, m, n]);

    for i in 0..b {
        let a_2d = t(a_data.clone(), &[m, k]);
        let b_i = t(b_data[i * k * n..(i + 1) * k * n].to_vec(), &[k, n]);
        let tape2 = Tape::new_with_ops(common::naive_ops());
        let a2 = tape2.var(&a_2d);
        let b2 = tape2.var(&b_i);
        let expected = dense(&a2.matmul(&b2).unwrap().to_tensor());
        let got = &out_data[i * m * n..(i + 1) * m * n];
        assert_eq!(got, expected.as_slice(), "batch {i} forward mismatch");
    }
}

// --- 2. backward: 数値微分（中央差分）との突合 ---

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
    let mut index = vec![0usize; shape.len()];
    for flat in 0..numel {
        let av = analytic.get(&index).unwrap_or(0.0);
        let nv = numeric.get(&index).unwrap_or(0.0);
        let diff = (av - nv).abs();
        let rel = diff / av.abs().max(nv.abs()).max(TAU);
        assert!(
            rel <= REL_TOL || diff <= ABS_TOL,
            "{label}[{flat:?} idx={index:?}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
        );
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

/// `loss = sum(a.matmul(b))` の forward をスカラーで再評価する
/// （中央差分の各サンプル点でテープを使い捨てる）。
fn forward_loss_sum(a: &Tensor<f32>, b: &Tensor<f32>) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let av = tape.var(a);
    let bv = tape.var(b);
    let loss = av.matmul(&bv).unwrap().sum(None).unwrap();
    loss.to_tensor()
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

fn numeric_grad(target_tensor: &Tensor<f32>, perturb: impl Fn(Tensor<f32>) -> f32) -> Tensor<f32> {
    let shape = target_tensor.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut data: Vec<f32> = dense(target_tensor);
    let mut grad = vec![0f32; numel];
    for i in 0..numel {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = perturb(t(data.clone(), &shape)) as f64;
        data[i] = (orig - H) as f32;
        let lm = perturb(t(data.clone(), &shape)) as f64;
        data[i] = orig as f32;
        grad[i] = ((lp - lm) / (2.0 * H)) as f32;
    }
    t(grad, &shape)
}

#[test]
fn matmul_batched_backward_matches_numeric_grad_equal_batch() {
    let (b, m, k, n) = (2usize, 2usize, 2usize, 2usize);
    let a = t(seq(10, b * m * k), &[b, m, k]);
    let bb = t(seq(20, b * k * n), &[b, k, n]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let av = tape.var(&a);
    let bv = tape.var(&bb);
    let loss = av.matmul(&bv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&av).unwrap().expect("a は loss に到達する");
    let db = grads.get(&bv).unwrap().expect("b は loss に到達する");

    let num_da = numeric_grad(&a, |a2| forward_loss_sum(&a2, &bb));
    let num_db = numeric_grad(&bb, |b2| forward_loss_sum(&a, &b2));
    assert_grad_close("batched da (equal batch)", da, &num_da);
    assert_grad_close("batched db (equal batch)", db, &num_db);
}

/// 空の勾配（`m == 0`／`n == 0`）を巨大なバッチ軸から縮約する backward
/// が、`axis_len` 回の空ループへ入らず直ちに返る（PR #1810 codex-review
/// P2 是正の回帰テスト）。`a = [1, 0, 1]`・`b = [B, 1, 0]`（B = 2^40）は
/// 入力・出力・勾配のいずれも実データを持たずに構成できる。
#[test]
fn matmul_batched_backward_with_empty_grad_and_huge_batch_returns_immediately() {
    let huge = 1usize << 40;
    let a = t(Vec::new(), &[1, 0, 1]);
    let bb = t(Vec::new(), &[huge, 1, 0]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let av = tape.var(&a);
    let bv = tape.var(&bb);
    let loss = av.matmul(&bv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&av).unwrap().expect("a は loss に到達する");
    let db = grads.get(&bv).unwrap().expect("b は loss に到達する");
    assert_eq!(da.shape(), &[1, 0, 1]);
    assert_eq!(db.shape(), &[huge, 1, 0]);
    assert_eq!(da.numel(), 0);
    assert_eq!(db.numel(), 0);
}

/// 空の勾配から復元する縮約先（`target_shape`）が確保不能な巨大形状
/// でも panic せず型付きエラーを返す（PR #1810 codex-review P1 是正の
/// 回帰テスト）。1 要素を `[1, H, 1]`（H = isize::MAX / 4 + 1）へ
/// broadcast した `a` と空の `b = [0, 1, 1]` では forward 出力と
/// `da_full = [0, H, 1]` は空テンソルとして成立するが、縮約先 `[1, H, 1]`
/// は H 個の f32 でバイトサイズが `isize::MAX` を超える。
#[test]
fn matmul_batched_backward_with_unallocatable_reduction_target_is_error_not_panic() {
    let huge = isize::MAX as usize / 4 + 1;
    let a = t(vec![1.0], &[1, 1, 1])
        .broadcast_to(&[1, huge, 1])
        .unwrap();
    let bb = t(Vec::new(), &[0, 1, 1]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let av = tape.var(&a);
    let bv = tape.var(&bb);
    let loss = av.matmul(&bv).unwrap().sum(None).unwrap();
    assert!(tape.backward(&loss).is_err());
}

#[test]
fn matmul_batched_backward_matches_numeric_grad_broadcast_lhs() {
    let (b, m, k, n) = (3usize, 2usize, 2usize, 2usize);
    let a = t(seq(30, m * k), &[1, m, k]);
    let bb = t(seq(40, b * k * n), &[b, k, n]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let av = tape.var(&a);
    let bv = tape.var(&bb);
    let loss = av.matmul(&bv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&av).unwrap().expect("a は loss に到達する");
    let db = grads.get(&bv).unwrap().expect("b は loss に到達する");

    let num_da = numeric_grad(&a, |a2| forward_loss_sum(&a2, &bb));
    let num_db = numeric_grad(&bb, |b2| forward_loss_sum(&a, &b2));
    assert_grad_close("batched da (broadcast lhs)", da, &num_da);
    assert_grad_close("batched db (broadcast lhs)", db, &num_db);
}

// --- 3. checkpoint 併用: rank≥3 MatMul の再計算が forward と bit 一致 ---

#[test]
fn matmul_batched_checkpoint_recompute_matches_no_checkpoint_bit_exact() {
    let (b, m, k, n) = (2usize, 2usize, 3usize, 2usize);
    let a = t(seq(50, b * m * k), &[b, m, k]);
    let bb = t(seq(60, b * k * n), &[b, k, n]);

    let run = |use_checkpoint: bool| -> (f32, Tensor<f32>, Tensor<f32>) {
        let tape = Tape::new_with_ops(common::naive_ops());
        let av = tape.var(&a);
        let bv = tape.var(&bb);
        let out = if use_checkpoint {
            tape.checkpoint(|| Ok(av.matmul(&bv)?.relu())).unwrap()
        } else {
            av.matmul(&bv).unwrap().relu()
        };
        let loss = out.sum(None).unwrap();
        let loss_val = loss
            .to_tensor()
            .get(&[])
            .expect("test fixture: スカラー shape [] のはず");
        let grads = tape.backward(&loss).unwrap();
        let da = grads
            .get(&av)
            .unwrap()
            .expect("a は loss に到達する")
            .clone();
        let db = grads
            .get(&bv)
            .unwrap()
            .expect("b は loss に到達する")
            .clone();
        (loss_val, da, db)
    };

    let (loss_no_ckpt, da_no_ckpt, db_no_ckpt) = run(false);
    let (loss_ckpt, da_ckpt, db_ckpt) = run(true);

    assert_eq!(
        loss_no_ckpt, loss_ckpt,
        "loss が checkpoint 有無で一致しない"
    );
    assert_eq!(
        dense(&da_no_ckpt),
        dense(&da_ckpt),
        "da が checkpoint 有無で bit 一致しない"
    );
    assert_eq!(
        dense(&db_no_ckpt),
        dense(&db_ckpt),
        "db が checkpoint 有無で bit 一致しない"
    );
}
