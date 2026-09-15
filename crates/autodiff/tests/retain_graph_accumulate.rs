//! `retain_graph` 契約（テープはグラフを `reset`／drop まで常時保持し、
//! 同一グラフに対する複数回 `backward` を成功させる）・
//! `Tape::backward_accumulate`（勾配蓄積 opt-in API）の契約テスト
//! （イシュー #1749）。設計は
//! `docs/autodiff-retain-graph-accumulate-decision.md` §2。
//!
//! bit 完全一致を主張してよいのは (a) 素の `backward` を同じグラフに
//! 対し複数回呼んだ場合の値（同一計算の再実行のため厳密に同値）と
//! (b) 同一 loss を 2 回蓄積した場合（`g + g` は IEEE 754 で厳密に
//! `2g`）に限る。異なる 2 つの loss を蓄積した場合は寄与の結合順が
//! `backward(l1.add(l2))` と異なるため、中央差分数値微分との統一複合
//! 判定（`tests/backward.rs` と同じ `REL_TOL`／`ABS_TOL`）で比較する。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

fn assert_bit_identical(a: &Tensor<f32>, b: &Tensor<f32>) {
    let da = dense(a);
    let db = dense(b);
    assert_eq!(da.len(), db.len(), "shape mismatch in bit-identity check");
    for (i, (x, y)) in da.iter().zip(db.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "element {i} differs: {x} (bits {:x}) vs {y} (bits {:x})",
            x.to_bits(),
            y.to_bits()
        );
    }
}

// --- 中央差分数値微分（`tests/backward.rs` と同じ様式の複製） ---

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

fn numeric_grad(target_tensor: &Tensor<f32>, perturb: impl Fn(Tensor<f32>) -> f32) -> Tensor<f32> {
    let shape = target_tensor.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut data: Vec<f32> = (0..numel)
        .map(|flat| {
            let mut idx = vec![0usize; shape.len()];
            let mut rem = flat;
            for axis in (0..shape.len()).rev() {
                idx[axis] = rem % shape[axis];
                rem /= shape[axis];
            }
            target_tensor.get(&idx).unwrap_or(0.0)
        })
        .collect();
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

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

// --- 1. 素の backward を同一グラフへ 2 回。bit 同一・ノード数不変 ---

#[test]
fn backward_twice_on_plain_graph_is_bit_identical_and_adds_no_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -0.5, 0.3, 2.0], &[2, 2]));
    let w = tape.var(&t(vec![0.5, -1.0, 1.5, 0.2], &[2, 2]));
    let target = tape.var(&t(vec![0.1, 0.0, 0.9, 0.2], &[2, 2]));

    let y = x.matmul(&w).unwrap().relu();
    let loss = y.mse_loss(&target).unwrap();

    let len_before = tape.len();
    let grads1 = tape.backward(&loss).unwrap();
    let len_after_first = tape.len();
    let grads2 = tape.backward(&loss).unwrap();
    let len_after_second = tape.len();

    assert_eq!(len_before, len_after_first);
    assert_eq!(len_before, len_after_second);

    let dx1 = grads1.get(&x).unwrap().cloned().unwrap();
    let dx2 = grads2.get(&x).unwrap().cloned().unwrap();
    assert_bit_identical(&dx1, &dx2);
    let dw1 = grads1.get(&w).unwrap().cloned().unwrap();
    let dw2 = grads2.get(&w).unwrap().cloned().unwrap();
    assert_bit_identical(&dw1, &dw2);
}

// --- 2. 遅延 elementwise チェーンでも複数回 backward が bit 同一 ---

#[test]
fn backward_twice_on_lazy_elementwise_chain_is_bit_identical() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.3, -0.2, 0.7, 1.1], &[2, 2]));

    // exp -> tanh -> mul(自己参照) の遅延 elementwise 合成。
    let e = x.exp();
    let th = e.tanh();
    let prod = th.mul(&th).unwrap();
    let loss = prod.sum(None).unwrap();

    let grads1 = tape.backward(&loss).unwrap();
    let grads2 = tape.backward(&loss).unwrap();
    let dx1 = grads1.get(&x).unwrap().cloned().unwrap();
    let dx2 = grads2.get(&x).unwrap().cloned().unwrap();
    assert_bit_identical(&dx1, &dx2);
}

// --- 3. 独立した backward 呼び出しは互いに蓄積しない ---

#[test]
fn independent_backward_calls_do_not_accumulate() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = tape.var(&t(vec![3.0, 4.0], &[2]));

    let l1 = x.mul(&x).unwrap().sum(None).unwrap();
    let l2 = y.mul(&y).unwrap().sum(None).unwrap();

    let grads1 = tape.backward(&l1).unwrap();
    let grads2 = tape.backward(&l2).unwrap();

    // l1 は y に到達しない・l2 は x に到達しない。
    assert!(grads1.get(&y).unwrap().is_none());
    assert!(grads2.get(&x).unwrap().is_none());
    // l1 の x 勾配は l2 の寄与を含まない（2x のまま）。
    let dx = grads1.get(&x).unwrap().unwrap();
    assert_eq!(dx.get(&[0]).unwrap(), 2.0);
    assert_eq!(dx.get(&[1]).unwrap(), 4.0);
}

// --- 4. backward_accumulate で同一 loss を 2 回蓄積 -> ちょうど 2g ---

#[test]
fn backward_accumulate_same_loss_twice_yields_exactly_two_g() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]));
    let loss = x.mul(&x).unwrap().sum(None).unwrap();

    let mut grads = tape.backward(&loss).unwrap();
    let g_once = grads.get(&x).unwrap().cloned().unwrap();

    tape.backward_accumulate(&loss, &mut grads).unwrap();
    let g_twice = grads.get(&x).unwrap().cloned().unwrap();

    // (g + g) は IEEE 754 で厳密に 2g なので bit 一致を主張できる。
    let expected = fandhe_ai_tensor_core::Tensor::from_shape_fill(g_once.shape(), |i| {
        let v = g_once.contiguous().as_slice().unwrap()[i];
        v + v
    })
    .unwrap();
    assert_bit_identical(&g_twice, &expected);
}

// --- 5. l1 -> l2 の順に蓄積した勾配が backward(l1+l2) および数値微分と一致 ---

#[test]
fn backward_accumulate_over_two_losses_matches_sum_loss_and_numeric_grad() {
    // pre-relu の絶対値が 0.1 以上になるよう選び、h=1e-3 の摂動で
    // ReLU のキンクを踏まない（`tests/backward.rs` と同方針）。
    let x_data = t(vec![1.0, -0.5, 0.3, 2.0], &[2, 2]);
    let w_data = t(vec![0.5, -1.0, 1.5, 0.2], &[2, 2]);
    let b_data = t(vec![0.1, -0.2], &[2]);
    let t1_data = t(vec![0.1, 0.0, 3.0, 0.0], &[2, 2]);
    let t2_data = t(vec![0.2, 0.3, -0.1, 0.5], &[2, 2]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&x_data);
    let w = tape.var(&w_data);
    let b = tape.var(&b_data);
    let target1 = tape.var(&t1_data);
    let target2 = tape.var(&t2_data);

    let y = x.matmul(&w).unwrap().add(&b).unwrap().relu();
    let l1 = y.mse_loss(&target1).unwrap();
    let l2 = y.mse_loss(&target2).unwrap();

    let mut grads = tape.backward(&l1).unwrap();
    tape.backward_accumulate(&l2, &mut grads).unwrap();
    let dx_accum = grads.get(&x).unwrap().cloned().unwrap();

    // 数値微分: forward_loss(x) = mse(relu(x@w+b), t1) + mse(relu(x@w+b), t2)
    let forward_loss = |xv: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&xv);
        let wv = tape.var(&w_data);
        let bv = tape.var(&b_data);
        let t1v = tape.var(&t1_data);
        let t2v = tape.var(&t2_data);
        let y = xv.matmul(&wv).unwrap().add(&bv).unwrap().relu();
        let l1 = y.mse_loss(&t1v).unwrap();
        let l2 = y.mse_loss(&t2v).unwrap();
        scalar(&l1.to_tensor()) + scalar(&l2.to_tensor())
    };
    let dx_numeric = numeric_grad(&x_data, forward_loss);
    assert_grad_close("dx_accum vs numeric", &dx_accum, &dx_numeric);

    // 別経路: backward(l1.add(l2)) との比較（結合順が違うため数値微分
    // と同じ許容誤差で比較する。ここでも厳密 bit 一致は主張しない）。
    let tape2 = Tape::new_with_ops(common::naive_ops());
    let x2 = tape2.var(&x_data);
    let w2 = tape2.var(&w_data);
    let b2 = tape2.var(&b_data);
    let t1v2 = tape2.var(&t1_data);
    let t2v2 = tape2.var(&t2_data);
    let y2 = x2.matmul(&w2).unwrap().add(&b2).unwrap().relu();
    let combined_loss = y2
        .mse_loss(&t1v2)
        .unwrap()
        .add(&y2.mse_loss(&t2v2).unwrap());
    // add は要素ごとの二項演算のため、mse_loss（スカラー shape []）の
    // 加算にそのまま使える。
    let combined_loss = combined_loss.unwrap();
    let grads_combined = tape2.backward(&combined_loss).unwrap();
    let dx_combined = grads_combined.get(&x2).unwrap().cloned().unwrap();
    assert_grad_close(
        "dx_accum vs combined-loss backward",
        &dx_accum,
        &dx_combined,
    );
}

// --- 6. l1 で未到達・l2 で到達するノードが蓄積後 backward(l2) と一致 ---

#[test]
fn backward_accumulate_fills_nodes_unreached_by_first_loss() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = tape.var(&t(vec![3.0, 4.0], &[2]));

    let l1 = x.mul(&x).unwrap().sum(None).unwrap();
    let l2 = y.mul(&y).unwrap().sum(None).unwrap();

    let mut grads = tape.backward(&l1).unwrap();
    assert!(grads.get(&y).unwrap().is_none());

    tape.backward_accumulate(&l2, &mut grads).unwrap();
    let dy = grads.get(&y).unwrap().cloned().unwrap();

    let grads_l2_only = tape.backward(&l2).unwrap();
    let dy_expected = grads_l2_only.get(&y).unwrap().cloned().unwrap();
    assert_bit_identical(&dy, &dy_expected);
}

// --- 7. 蓄積先の Gradients 生成後に追加されたノードも読める ---

#[test]
fn backward_accumulate_extends_for_nodes_added_between_calls() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let l1 = x.mul(&x).unwrap().sum(None).unwrap();
    let mut grads = tape.backward(&l1).unwrap();

    // grads 生成後に新しいノードを追加する。
    let y = tape.var(&t(vec![5.0, 6.0], &[2]));
    let l2 = y.mul(&y).unwrap().sum(None).unwrap();

    // panic せず成功し、新ノードの勾配が読める。
    tape.backward_accumulate(&l2, &mut grads).unwrap();
    let dy = grads.get(&y).unwrap().cloned().unwrap();
    assert_eq!(dy.get(&[0]).unwrap(), 10.0);
    assert_eq!(dy.get(&[1]).unwrap(), 12.0);
    // 既存 x の勾配は不変。
    let dx = grads.get(&x).unwrap().cloned().unwrap();
    assert_eq!(dx.get(&[0]).unwrap(), 2.0);
    assert_eq!(dx.get(&[1]).unwrap(), 4.0);
}

// --- 8. クロステープ検査 ---

#[test]
fn backward_accumulate_rejects_foreign_tape_gradients() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());

    let xa = tape_a.var(&t(vec![1.0, 2.0], &[2]));
    let la = xa.mul(&xa).unwrap().sum(None).unwrap();
    let mut grads_a = tape_a.backward(&la).unwrap();

    let xb = tape_b.var(&t(vec![3.0, 4.0], &[2]));
    let lb = xb.mul(&xb).unwrap().sum(None).unwrap();

    // tape_b 上の loss を tape_a の Gradients へ蓄積しようとする。
    let err = tape_a.backward_accumulate(&lb, &mut grads_a).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

#[test]
fn backward_accumulate_rejects_foreign_loss() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());

    let xb = tape_b.var(&t(vec![1.0, 2.0], &[2]));
    let lb = xb.mul(&xb).unwrap().sum(None).unwrap();
    let mut grads_b = tape_b.backward(&lb).unwrap();

    let xa = tape_a.var(&t(vec![3.0, 4.0], &[2]));
    let la = xa.mul(&xa).unwrap().sum(None).unwrap();

    let err = tape_b.backward_accumulate(&la, &mut grads_b).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

#[test]
fn backward_accumulate_rejects_foreign_accumulation_target() {
    // 上記 2 件はいずれも `loss` が `self` と不一致（第 1 検査）で
    // 弾かれるケース。本テストは `loss` は `self` と一致するが
    // `into`（蓄積先 `Gradients`）だけが別テープ由来という、
    // `backward_accumulate` の第 2 検査（`into.tape_id != self.id`）
    // を単独で踏む経路を網羅する（codex-review 指摘・イシュー #1749）。
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());

    let xa = tape_a.var(&t(vec![1.0, 2.0], &[2]));
    let la = xa.mul(&xa).unwrap().sum(None).unwrap();
    let mut grads_a = tape_a.backward(&la).unwrap();

    let xb = tape_b.var(&t(vec![3.0, 4.0], &[2]));
    let lb = xb.mul(&xb).unwrap().sum(None).unwrap();

    // loss (lb) は self (tape_b) と一致するが、蓄積先 grads_a は
    // tape_a 由来のため拒否される。
    let err = tape_b.backward_accumulate(&lb, &mut grads_a).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

// --- 9. reset をまたいだ Gradients への蓄積は拒否 ---

#[test]
fn backward_accumulate_rejects_gradients_from_before_reset() {
    let mut tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let l1 = x.mul(&x).unwrap().sum(None).unwrap();
    let mut grads = tape.backward(&l1).unwrap();

    tape.reset();
    let x_new = tape.leaf(0).unwrap();
    let l2 = x_new.mul(&x_new).unwrap().sum(None).unwrap();

    let err = tape.backward_accumulate(&l2, &mut grads).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

// --- 10. エラー時は into が無変更のまま返る（原子性） ---

#[test]
fn backward_accumulate_keeps_into_unchanged_on_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let l1 = x.mul(&x).unwrap().sum(None).unwrap();
    let mut grads = tape.backward(&l1).unwrap();
    let dx_before = grads.get(&x).unwrap().cloned().unwrap();

    // 追跡なし葉のみで構成された loss は backward_impl 自体が
    // Err(Backward) を返す（`Tape::var_no_grad` の契約）。
    let untracked = tape.var_no_grad(&t(vec![5.0, 6.0], &[2]));
    let untracked_loss = untracked.sum(None).unwrap();
    let err = tape
        .backward_accumulate(&untracked_loss, &mut grads)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));

    // into は全く変更されていないはず。
    let dx_after = grads.get(&x).unwrap().cloned().unwrap();
    assert_bit_identical(&dx_before, &dx_after);
}

// --- 11. var_no_grad 葉は蓄積後も GradientTrackingDisabled のまま ---

#[test]
fn backward_accumulate_leaves_no_grad_leaf_disabled() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var_no_grad(&t(vec![1.0, 2.0], &[2]));
    let w = tape.var(&t(vec![3.0, 4.0], &[2]));

    let l1 = x.mul(&w).unwrap().sum(None).unwrap();
    let mut grads = tape.backward(&l1).unwrap();
    assert!(matches!(
        grads.get(&x).unwrap_err(),
        AutodiffError::GradientTrackingDisabled
    ));

    let l2 = x.mul(&w).unwrap().sum(None).unwrap();
    tape.backward_accumulate(&l2, &mut grads).unwrap();
    assert!(matches!(
        grads.get(&x).unwrap_err(),
        AutodiffError::GradientTrackingDisabled
    ));
    // w は正常に蓄積されている（2 回分）。
    let dw = grads.get(&w).unwrap().cloned().unwrap();
    let x_val = dense(&x.to_tensor());
    assert_eq!(dw.get(&[0]).unwrap(), x_val[0] * 2.0);
    assert_eq!(dw.get(&[1]).unwrap(), x_val[1] * 2.0);
}
