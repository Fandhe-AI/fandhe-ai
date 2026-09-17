//! 子テープ方式 `create_graph`（`Tape::backward_create_graph`。イシュー
//! #1942）の受け入れ条件を直接検証する統合テスト。
//!
//! - elementwise／sum 系 Op（`Add`／`Mul`／`Relu`／`Exp`／`Tanh`／
//!   `Sigmoid`／`Sum`／`Mean`／`Reshape`／`BroadcastTo`）の二階微分が、
//!   独立な有限差分（1 階解析勾配 `Tape::backward` を中央差分した数値
//!   Hessian）と REQ-2 統一複合判定内で一致することを確認する
//!   （`common::req2_close`）。
//! - 代表的な合成関数（`x^3`・`tanh`・`sigmoid`・`relu(x)*x`）の対角
//!   Hessian を手計算の閉形式と突合する。
//! - `create_graph` を使わない既定経路（`Tape::backward`）が
//!   `backward_create_graph` 呼び出しの前後で bit 同一のまま不変である
//!   ことを確認する（受入基準 2）。
//! - fail-closed 契約（未対応 Op・同一テープ・非空子テープ・checkpoint
//!   済み親テープ・非追跡 loss・クロステープ・`GradientTrackingDisabled`）
//!   を確認する。
//!
//! 数値方式の bit 同一は主張しない（`create_graph.rs` モジュール doc
//! 参照）——正しさは本ファイルの有限差分突合・閉形式突合のみを根拠と
//! する。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn flat_len(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// 平坦添字 `idx` を `shape` の多次元添字へ変換する（行優先）。
fn unravel(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        out[d] = idx % shape[d];
        idx /= shape[d];
    }
    out
}

/// `Tape::backward`（既存 1 階・`create_graph` 非依存）を中央差分で
/// 2 回微分した数値 Hessian（`hessian[j][i] = d(grad_j)/dx_i`）。
/// 独立クロスチェック（設計 doc §9「検証方針案」の有限差分に相当。
/// `create_graph` 側の実装バグを Hessian 自己無矛盾性だけでは検出
/// できないため、`create_graph` を一切経由しない別経路で比較する）。
fn finite_diff_hessian<F>(build: F, x0: &[f32], shape: &[usize], h: f64) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Result<Var<'t>, AutodiffError>,
{
    let n = flat_len(shape);
    let grad_at = |x: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), shape));
        let loss = build(&tape, &xv).expect("build: forward 構築に失敗");
        let grads = tape.backward(&loss).expect("backward: 1 階勾配計算に失敗");
        let dx = grads
            .get(&xv)
            .expect("get: クロステープ検査は通るはず")
            .expect("x は loss に到達するはず");
        (0..n)
            .map(|i| dx.get(&unravel(i, shape)).expect("shape 範囲内のはず") as f64)
            .collect()
    };
    let mut hessian = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let mut xp = x0.to_vec();
        let mut xm = x0.to_vec();
        xp[i] += h as f32;
        xm[i] -= h as f32;
        let gp = grad_at(&xp);
        let gm = grad_at(&xm);
        for j in 0..n {
            hessian[j][i] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    hessian
}

/// `Tape::backward_create_graph` が構築する子テープ上の 1 階勾配を、
/// さらに `Tape::backward` で微分して full Hessian を得る（`x` の各
/// 成分ごとに `child.backward` を 1 回呼ぶ。`retain_graph` 契約
/// 〈イシュー #1749〉により同一子テープへの複数回 `backward` 呼び出しは
/// 無条件に成功する）。
fn analytic_hessian<F>(build: F, x0: &[f32], shape: &[usize]) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Result<Var<'t>, AutodiffError>,
{
    let n = flat_len(shape);
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("build: forward 構築に失敗");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph に失敗");
    let gx = cg
        .grad(&x)
        .expect("grad: クロステープ検査は通るはず")
        .expect("x は loss に到達するはず（1 階勾配）");
    let cx = cg
        .child_var(&x)
        .expect("child_var: クロステープ検査は通るはず")
        .expect("x は子テープ上に写しを持つはず");

    let mut hessian = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let gj = extract_scalar(&gx, shape, j).expect("extract_scalar");
        let row_grads = child.backward(&gj).expect("child.backward（2 階）に失敗");
        let row = row_grads
            .get(&cx)
            .expect("get: 子テープ上のクロステープ検査は通るはず")
            .expect("2 階勾配は x へ到達するはず");
        for i in 0..n {
            hessian[j][i] = row.get(&unravel(i, shape)).expect("shape 範囲内のはず") as f64;
        }
    }
    hessian
}

/// `v`（shape `shape`）から平坦添字 `flat_idx` が指す 1 要素をスカラー
/// `Var` として取り出す（`Var::narrow` を軸ごとに畳み込む）。`Var::
/// reshape` と異なり `Var::narrow` は非 contiguous な入力（`broadcast_to`
/// の view 結果等）でも成立する view 演算のため、`Var::contiguous`
/// （`pub(crate)` で本統合テスト〈別クレート扱い〉からは呼べない）を
/// 経由せずに済む。
fn extract_scalar<'c>(
    v: &Var<'c>,
    shape: &[usize],
    flat_idx: usize,
) -> Result<Var<'c>, AutodiffError> {
    let idx = unravel(flat_idx, shape);
    let mut cur = *v;
    for (dim, &i) in idx.iter().enumerate() {
        cur = cur.narrow(dim, i, 1)?;
    }
    cur.sum(None)
}

fn assert_hessian_close(analytic: &[Vec<f64>], numeric: &[Vec<f64>]) {
    assert_eq!(analytic.len(), numeric.len());
    for (row_a, row_n) in analytic.iter().zip(numeric.iter()) {
        assert_eq!(row_a.len(), row_n.len());
        for (&a, &n) in row_a.iter().zip(row_n.iter()) {
            assert!(
                common::req2_close(a, n),
                "hessian mismatch: analytic={a} numeric={n}"
            );
        }
    }
}

// **build 関数を `fn` 項として定義する理由**: クロージャ（`|_t, x|
// ...`）は定義時点で単独の具体シグネチャへ推論されるため、
// `for<'t> Fn(&'t Tape, &Var<'t>) -> Result<Var<'t>, AutodiffError>`
// という HRTB（両引数のライフタイムが呼び出しごとに揃う汎用形）へは
// 推論できず、`finite_diff_hessian`/`analytic_hessian`（複数回・
// 複数ライフタイムで呼ぶ）への同時渡しが型検査に通らない。ジェネリック
// な `fn` 項はこの汎用形を自然に満たすため、代わりに名前付き関数として
// 定義する。

fn build_cubic<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.mul(x)?.mul(x)?.sum(None)
}

// --- 1. x^3（`Mul` の反復・fan-out）: d²/dx² = 6x, 非対角は 0 -----------

#[test]
fn hessian_cubic_matches_finite_difference_and_closed_form() {
    let x0 = [1.0f32, -2.0, 0.5];

    let numeric = finite_diff_hessian(build_cubic, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_cubic, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);

    for i in 0..3 {
        let expected_diag = 6.0 * x0[i] as f64;
        assert!(
            common::req2_close(analytic[i][i], expected_diag),
            "diag[{i}]: analytic={} expected={}",
            analytic[i][i],
            expected_diag
        );
        for j in 0..3 {
            if i != j {
                assert!(analytic[j][i].abs() < 1e-4, "off-diag[{j}][{i}] not ~0");
            }
        }
    }
}

fn build_tanh<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.tanh().sum(None)
}

// --- 2. tanh: d²/dx² = -2*tanh(x)*(1-tanh(x)^2) -------------------------

#[test]
fn hessian_tanh_matches_finite_difference_and_closed_form() {
    let x0 = [0.3f32, -1.1, 2.0];

    let numeric = finite_diff_hessian(build_tanh, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_tanh, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);

    for i in 0..3 {
        let th = (x0[i] as f64).tanh();
        let expected_diag = -2.0 * th * (1.0 - th * th);
        assert!(
            common::req2_close(analytic[i][i], expected_diag),
            "diag[{i}]: analytic={} expected={}",
            analytic[i][i],
            expected_diag
        );
    }
}

fn build_sigmoid<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.sigmoid().sum(None)
}

// --- 3. sigmoid: d²/dx² = s*(1-s)*(1-2s) --------------------------------

#[test]
fn hessian_sigmoid_matches_finite_difference_and_closed_form() {
    let x0 = [-0.5f32, 0.2, 1.5];

    let numeric = finite_diff_hessian(build_sigmoid, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_sigmoid, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);

    for i in 0..3 {
        let s = 1.0 / (1.0 + (-(x0[i] as f64)).exp());
        let expected_diag = s * (1.0 - s) * (1.0 - 2.0 * s);
        assert!(
            common::req2_close(analytic[i][i], expected_diag),
            "diag[{i}]: analytic={} expected={}",
            analytic[i][i],
            expected_diag
        );
    }
}

// --- 4. relu(x) * x（Relu の劣勾配マスク・Mul との合成） ---------------
// x_i > 0: relu(x)*x = x^2 → d²/dx² = 2。x_i < 0: 恒等的に 0 → d²/dx² = 0。

fn build_relu_times_x<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.relu().mul(x)?.sum(None)
}

#[test]
fn hessian_relu_times_x_matches_finite_difference_and_closed_form() {
    let x0 = [1.5f32, -2.0, 0.7, -0.3];

    let numeric = finite_diff_hessian(build_relu_times_x, &x0, &[4], 1e-3);
    let analytic = analytic_hessian(build_relu_times_x, &x0, &[4]);
    assert_hessian_close(&analytic, &numeric);

    for i in 0..4 {
        let expected_diag = if x0[i] > 0.0 { 2.0 } else { 0.0 };
        assert!(
            common::req2_close(analytic[i][i], expected_diag),
            "diag[{i}]: analytic={} expected={}",
            analytic[i][i],
            expected_diag
        );
    }
}

fn build_exp<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.exp().sum(None)
}

// --- 5. exp: d²/dx² = exp(x) --------------------------------------------

#[test]
fn hessian_exp_matches_finite_difference_and_closed_form() {
    let x0 = [0.1f32, -0.5, 0.8];

    let numeric = finite_diff_hessian(build_exp, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_exp, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);

    for i in 0..3 {
        let expected_diag = (x0[i] as f64).exp();
        assert!(common::req2_close(analytic[i][i], expected_diag));
    }
}

// --- 6. Add の broadcast（`[2,2] + [2]`）: 線形演算なので Hessian は
//        全域ゼロ（`reduce_to` の broadcast 縮約経路を検証） -----------

fn build_broadcast_add<'t>(tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let b = tape.var(&t(vec![10.0, -5.0], &[2]));
    x.add(&b)?.sum(None)
}

#[test]
fn hessian_broadcast_add_is_zero() {
    // `x.add(&b)` は線形演算のため 1 階勾配 `d/dx sum(x+b) = 1`
    // （broadcast 元の `b` には依存しない定数）である。`reduce_to` の
    // 早期 return（`v_shape == target_shape` のとき写しを作らず入力
    // をそのまま返す。`create_graph.rs::reduce_to` 参照）により、
    // `cg.grad(&x)` は `x` を一切参照しない `var_no_grad` シード葉その
    // ものになる——つまり 2 階の勾配グラフ自体が存在しない（`requires_
    // grad == false` のため `child.backward` は `Err(Backward)` を
    // 返す）。これは実装の不備ではなく「線形関数の Hessian は恒等的に
    // 0」であることの構造的な証拠そのものである（勾配が `x` に一切
    // 依存しない＝ x で微分すれば 0）ため、`analytic_hessian`
    // （2 階の子テープ再逆伝播を要求する汎用ヘルパー）は使わず、1 階
    // 勾配が定数値であることのみを直接検証する。
    let x0 = [1.0f32, -2.0, 3.0, 0.5];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), &[2, 2]));
    let loss = build_broadcast_add(&tape, &x).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gx = cg.grad(&x).unwrap().expect("x は loss に到達する");
    for i in 0..4 {
        let actual = gx.value().get(&unravel(i, &[2, 2])).unwrap() as f64;
        assert!(common::req2_close(actual, 1.0));
    }
}

fn build_sum_single_axis<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.mul(x)?.sum(Some(0))?.sum(None)
}

// --- 7. Sum(Some(axis))（`unsqueeze`＋`broadcast_to` 経路）: x^2 の
//        軸縮約→全縮約は全縮約と同じ Hessian（対角 2、非対角 0） -------

#[test]
fn hessian_sum_single_axis_matches_finite_difference() {
    let x0 = [1.0f32, -1.0, 2.0, 0.5, -0.5, 1.5];
    let shape = [2usize, 3usize];

    let numeric = finite_diff_hessian(build_sum_single_axis, &x0, &shape, 1e-3);
    let analytic = analytic_hessian(build_sum_single_axis, &x0, &shape);
    assert_hessian_close(&analytic, &numeric);

    let n = flat_len(&shape);
    for i in 0..n {
        assert!(common::req2_close(analytic[i][i], 2.0));
        for j in 0..n {
            if i != j {
                assert!(analytic[j][i].abs() < 1e-4);
            }
        }
    }
}

fn build_mean<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.mul(x)?.mean(None)
}

// --- 8. Mean: d²/dx² = 2/n（`n` 要素の平均二乗） ------------------------

#[test]
fn hessian_mean_matches_finite_difference_and_closed_form() {
    let x0 = [1.0f32, -2.0, 0.5, 3.0];

    let numeric = finite_diff_hessian(build_mean, &x0, &[4], 1e-3);
    let analytic = analytic_hessian(build_mean, &x0, &[4]);
    assert_hessian_close(&analytic, &numeric);

    for i in 0..4 {
        assert!(common::req2_close(analytic[i][i], 2.0 / 4.0));
    }
}

fn build_fan_out<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let a = x.mul(x)?.sum(None)?;
    let b = x.exp().sum(None)?;
    a.add(&b)
}

// --- 9. fan-out（同一 x を 2 経路で使用）: sum(x*x) + sum(exp(x)) -------
//        d²/dx² = 2 + exp(x)（両寄与が accumulate で合算されることを検証）

#[test]
fn hessian_fan_out_matches_finite_difference_and_closed_form() {
    let x0 = [0.4f32, -0.6, 1.1];

    let numeric = finite_diff_hessian(build_fan_out, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_fan_out, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);

    for i in 0..3 {
        let expected_diag = 2.0 + (x0[i] as f64).exp();
        assert!(common::req2_close(analytic[i][i], expected_diag));
    }
}

fn build_reshape<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.reshape(&[4])?.mul(&x.reshape(&[4])?)?.sum(None)
}

// --- 10. Reshape（view 系。replay 時に `contiguous` を挟む経路） -------

#[test]
fn hessian_reshape_matches_finite_difference() {
    let x0 = [1.0f32, -1.5, 0.3, 2.2];

    let numeric = finite_diff_hessian(build_reshape, &x0, &[4], 1e-3);
    let analytic = analytic_hessian(build_reshape, &x0, &[4]);
    assert_hessian_close(&analytic, &numeric);
}

fn build_non_scalar_loss<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.mul(x)?.mul(x)
}

// --- 11. 非スカラー loss（暗黙の総和射影） ------------------------------

#[test]
fn hessian_non_scalar_loss_matches_finite_difference() {
    // `sum` を挟まず非スカラー Var をそのまま loss として渡す
    // （`Tape::backward` の「全要素 1 のシード」契約が create_graph
    // でも同じく成立することを確認）。
    let x0 = [1.0f32, -2.0, 0.5];

    let numeric = finite_diff_hessian(build_non_scalar_loss, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_non_scalar_loss, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);
}

// --- 12. 1 階勾配の bit 同一性・親テープの不変性（受入基準 2） ---------

#[test]
fn create_graph_first_order_matches_plain_backward_and_leaves_parent_intact() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 0.5], &[3]));
    let loss = x.mul(&x).unwrap().sum(None).unwrap();

    let before = tape.backward(&loss).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let after = tape.backward(&loss).unwrap();

    let dx_before = before.get(&x).unwrap().expect("x は loss に到達する");
    let dx_first_order = cg
        .first_order()
        .get(&x)
        .unwrap()
        .expect("x は loss に到達する");
    let dx_after = after.get(&x).unwrap().expect("x は loss に到達する");

    for i in 0..3 {
        let idx = [i];
        let b = dx_before.get(&idx).unwrap().to_bits();
        let f = dx_first_order.get(&idx).unwrap().to_bits();
        let a = dx_after.get(&idx).unwrap().to_bits();
        assert_eq!(b, f, "first_order は素の backward と bit 同一のはず");
        assert_eq!(
            b, a,
            "backward_create_graph 呼び出し前後で backward の結果が変わってはいけない"
        );
    }

    // 子テープ上の 1 階勾配 `Var` の値も、1 階 `Gradients` と数値的に
    // 一致する（REQ-2。数値方式が独立のため bit 同一は主張しない）。
    let gx = cg.grad(&x).unwrap().expect("x は loss に到達する");
    for i in 0..3 {
        let expected = dx_first_order.get(&[i]).unwrap() as f64;
        let actual = gx.value().get(&[i]).unwrap() as f64;
        assert!(common::req2_close(actual, expected));
    }
}

// --- 13. `var_no_grad` 葉は定数扱い（子孫が求まり、その葉自体は
//         `GradientTrackingDisabled`） --------------------------------

#[test]
fn create_graph_treats_no_grad_leaf_as_constant() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let c = tape.var_no_grad(&t(vec![3.0, 4.0], &[2]));
    let loss = x.mul(&c).unwrap().sum(None).unwrap();

    let cg = tape.backward_create_graph(&loss, &child).unwrap();

    let err = cg.grad(&c).unwrap_err();
    assert!(matches!(err, AutodiffError::GradientTrackingDisabled));

    let gx = cg
        .grad(&x)
        .unwrap()
        .expect("x は追跡対象なので勾配が求まる");
    // d/dx sum(x*c) = c（定数）
    for i in 0..2 {
        let expected = [3.0f64, 4.0][i];
        let actual = gx.value().get(&[i]).unwrap() as f64;
        assert!(common::req2_close(actual, expected));
    }
}

// --- fail-closed 契約 ----------------------------------------------------

#[test]
fn create_graph_rejects_unsupported_op_matmul() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let loss = a.matmul(&b).unwrap().sum(None).unwrap();

    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
}

#[test]
fn create_graph_rejects_unsupported_op_softmax() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let loss = x.softmax(1).unwrap().sum(None).unwrap();

    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
}

#[test]
fn create_graph_rejects_same_tape_as_child() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0], &[1]));
    let loss = x.mul(&x).unwrap().sum(None).unwrap();

    let err = tape.backward_create_graph(&loss, &tape).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
}

#[test]
fn create_graph_rejects_non_empty_child() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let _junk = child.var(&t(vec![0.0], &[1]));
    let x = tape.var(&t(vec![1.0], &[1]));
    let loss = x.mul(&x).unwrap().sum(None).unwrap();

    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
}

#[test]
fn create_graph_rejects_parent_with_registered_checkpoint() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let checkpointed = tape.checkpoint(|| Ok(x.sigmoid())).unwrap();
    let loss = checkpointed.sum(None).unwrap();

    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
}

#[test]
fn create_graph_rejects_untracked_loss() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var_no_grad(&t(vec![1.0], &[1]));
    let loss = x.mul(&x).unwrap().sum(None).unwrap();

    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
}

#[test]
fn create_graph_rejects_cross_tape_loss() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let xa = tape_a.var(&t(vec![1.0], &[1]));
    let loss_a = xa.mul(&xa).unwrap().sum(None).unwrap();

    let err = tape_b.backward_create_graph(&loss_a, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}
