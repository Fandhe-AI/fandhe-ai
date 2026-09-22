//! 子テープ方式 `create_graph`（`Tape::backward_create_graph`。イシュー
//! #1942・#1943）の受け入れ条件を直接検証する統合テスト。
//!
//! - elementwise／sum 系 Op（`Add`／`Mul`／`Relu`／`Exp`／`Tanh`／
//!   `Sigmoid`／`Sum`／`Mean`／`Reshape`／`BroadcastTo`）・`MatMul`
//!   （rank 2 × rank 2 限定。#1943）の二階微分が、独立な有限差分
//!   （1 階解析勾配 `Tape::backward` を中央差分した数値 Hessian）と
//!   REQ-2 統一複合判定内で一致することを確認する（`common::
//!   req2_close`）。
//! - 代表的な合成関数（`x^3`・`tanh`・`sigmoid`・`relu(x)*x`・matmul の
//!   二次形式・Linear〈`MatMul`→`Add` bias〉）の対角 Hessian を手計算の
//!   閉形式と突合する。
//! - 小型 MLP（`Linear`→`tanh`→`Linear`）の HVP（ヘッセ・ベクトル積）を
//!   `create_graph` を経由しない独立な方向微分の中央差分と突合する。
//! - `create_graph` を使わない既定経路（`Tape::backward`）が
//!   `backward_create_graph` 呼び出しの前後で bit 同一のまま不変である
//!   ことを確認する（受入基準 2）。
//! - fail-closed 契約（未対応 Op・rank≥3 の `MatMul`・fused `LinearAct`・
//!   同一テープ・非空子テープ・checkpoint 済み親テープ・非追跡 loss・
//!   クロステープ・`GradientTrackingDisabled`）を確認する。入口検査 7
//!   （`validate_ancestors`）が `child` へ一切書き込む前に判定すること
//!   （拒否時 `child.is_empty()` が保たれること）も併せて固定する。
//!
//! 数値方式の bit 同一は主張しない（`create_graph.rs` モジュール doc
//! 参照）——正しさは本ファイルの有限差分突合・閉形式突合のみを根拠と
//! する。

// 本ファイルの Hessian 添字ループは `hessian[j][i]` のような 2 次元
// 添字書き込み・`analytic[j][i]` の対称性検査など、`i`／`j` 自身を
// 複数の配列へ同時に使う箇所が中心で、`enumerate()` の単一要素参照
// だけでは代替できない（`nn_optim_lamb.rs` と同型の判断）。
#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::Arc;

use fandhe_ai_autodiff::nn::Linear;
use fandhe_ai_autodiff::{AutodiffError, CustomFunction, Tape, Var};
use fandhe_ai_tensor_core::{Activation, Tensor};

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

// --- 6b. Mul の broadcast 縮約が桁落ちを伴う場合に 1 階 VJP
//         （`grad.rs::reduce_to_shape`。f32 逐次和）と数値的に一致する
//         ことを確認する回帰テスト（codex-review 指摘・PR #1998）。
//         `reduce_to`（`create_graph.rs`）が `Var::sum_dims`〈CPU
//         `sum` の f64 アキュムレータ縮約〉を使っていた当初実装では、
//         `c = [1e8, 1, -1e8]` のように桁落ちを伴う broadcast 入力で
//         1 階勾配（f32 逐次和: `(1e8+1)-1e8` は `1e8+1` が丸めで
//         `1e8` のまま変わらず結果 `0.0`）と 2 階側の縮約（f64 で
//         先に総和してから 1 回丸め: 結果 `1.0`）が乖離し、REQ-2
//         統一複合判定を満たさなかった。

fn build_broadcast_mul_cancellation<'t>(
    tape: &'t Tape,
    x: &Var<'t>,
) -> Result<Var<'t>, AutodiffError> {
    let c = tape.var_no_grad(&t(vec![1.0e8, 1.0, -1.0e8], &[3]));
    x.mul(&c)?.sum(None)
}

#[test]
fn create_graph_reduce_to_matches_first_order_under_cancellation() {
    let x0 = [1.0f32];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), &[1]));
    let loss = build_broadcast_mul_cancellation(&tape, &x).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();

    let first_order = cg
        .first_order()
        .get(&x)
        .unwrap()
        .expect("x は loss に到達する");
    let first_order_val = first_order.get(&[0]).unwrap() as f64;
    // f32 逐次和（`[1e8, 1, -1e8]` の順序）は `1e8+1` が丸めで `1e8`
    // のまま変わらず、続けて `-1e8` を足すと厳密に `0.0` になる
    // （`grad.rs::reduce_to_shape` の実際の挙動をまず固定する）。
    assert!(
        common::req2_close(first_order_val, 0.0),
        "1 階 VJP（f32 逐次和）は桁落ちにより 0.0 になるはず: {first_order_val}"
    );

    let cgrad = cg.grad(&x).unwrap().expect("x は loss に到達する");
    let cgrad_val = cgrad.value().get(&[0]).unwrap() as f64;
    assert!(
        common::req2_close(cgrad_val, first_order_val),
        "create_graph の broadcast 縮約（reduce_to）が 1 階 VJP（grad.rs::\
         reduce_to_shape）と数値的に乖離した: cgrad={cgrad_val} \
         first_order={first_order_val}"
    );
}

// --- 6c. Op::Add の bias パターン（`upstream: [m, n]` → `[n]`／
//         `[1, n]` の行方向縮約）が 1 階 VJP（`grad.rs::
//         reduce_bias_grad`。f64 相当のアキュムレータ。2026-09-12
//         ユーザー承認・`.claude/rules/coding-rust.md` の勾配長軸縮約
//         契約）と数値的に一致することを確認する回帰テスト（codex-review
//         指摘・PR #1998）。`reduce_bias_grad_var` を追加する前は
//         `Op::Add` の bias パターンにも一様に `reduce_to`（f32 逐次和）
//         を適用していたため、桁落ちを伴う入力で符号レベルの不一致が
//         生じた: `x: [3, 1]`・`b: [1]`・`c = [1e8, 1, -1e8]: [3, 1]`
//         に対する `loss = ((x + b) * c).sum()` で `b` の 1 階勾配は
//         `1.0` だが旧 `reduce_to`（f32 逐次和。`(1e8+1)-1e8` が丸めで
//         `0.0` になる）は `0.0` を返していた。

#[test]
fn create_graph_bias_pattern_add_matches_first_order_under_cancellation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0, 0.0, 0.0], &[3, 1]));
    let b = tape.var(&t(vec![0.0], &[1]));
    let c = tape.var_no_grad(&t(vec![1.0e8, 1.0, -1.0e8], &[3, 1]));
    let loss = x.add(&b).unwrap().mul(&c).unwrap().sum(None).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();

    let first_order = cg
        .first_order()
        .get(&b)
        .unwrap()
        .expect("b は loss に到達する");
    let first_order_val = first_order.get(&[0]).unwrap() as f64;
    // `reduce_bias_grad`（f64 相当のアキュムレータ）は桁落ちの影響を
    // 受けず `1e8 + 1 - 1e8 = 1.0` を厳密に計算する
    // （`grad.rs::reduce_bias_grad` の実際の挙動をまず固定する）。
    assert!(
        common::req2_close(first_order_val, 1.0),
        "1 階 VJP（reduce_bias_grad。f64 相当）は桁落ちに強く 1.0 のはず: {first_order_val}"
    );

    let cgrad = cg.grad(&b).unwrap().expect("b は loss に到達する");
    let cgrad_val = cgrad.value().get(&[0]).unwrap() as f64;
    assert!(
        common::req2_close(cgrad_val, first_order_val),
        "create_graph の Op::Add bias パターン縮約（reduce_bias_grad_var）が \
         1 階 VJP（grad.rs::reduce_bias_grad）と数値的に乖離した: \
         cgrad={cgrad_val} first_order={first_order_val}"
    );
}

// --- 6d. `reduce_to`（`Op::BroadcastTo` の VJP 経路）がゼロ長軸
//         （縮約対象の要素数が 0 の合法な空テンソル）を
//         `NarrowOutOfBounds` で失敗せずゼロ勾配として処理できることを
//         確認する回帰テスト（codex-review 指摘・PR #1998。P2）。
//         `x: [3]` を `broadcast_to(&[0, 3])` した結果を縮約する場合、
//         旧 `reduce_to` は `axis_len == 0` でも無条件に
//         `Var::narrow(axis, 0, 1)` を呼んでいたため
//         `ShapeError::NarrowOutOfBounds` になっていた。1 階 VJP
//         （`grad.rs::reduce_to_shape`）は縮約ループが 0 回実行される
//         ためゼロ初期化された結果をそのまま返す。

#[test]
fn create_graph_reduce_to_handles_zero_length_axis() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let loss = x.broadcast_to(&[0, 3]).unwrap().sum(None).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();

    let first_order = cg
        .first_order()
        .get(&x)
        .unwrap()
        .expect("x は loss に到達する");
    for i in 0..3 {
        let v = first_order.get(&[i]).unwrap() as f64;
        assert!(common::req2_close(v, 0.0), "1 階勾配はゼロのはず: {v}");
    }

    // 子テープ側（`reduce_to` の `axis_len == 0` 分岐）が panic／Err
    // にならず、同じくゼロ勾配を構築できることを確認する。
    let cgrad = cg.grad(&x).unwrap().expect("x は loss に到達する");
    for i in 0..3 {
        let v = cgrad.value().get(&[i]).unwrap() as f64;
        assert!(
            common::req2_close(v, 0.0),
            "create_graph 側のゼロ長軸縮約がゼロ勾配と一致しない: {v}"
        );
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

// `Op::MatMul`（rank 2 × rank 2）は #1943 で対応済み・#1943.5 以下参照。
// `Var::sub`（`Op::ScalarBinary { op: Sub, .. }`）はイシュー #2062 で
// 対応済みになったため、本テストは代わりに `Var::gelu`（誤差関数版
// GELU。`ScalarUnaryOp::Gelu`）を未対応 Op の代表として使う——導関数
// `Φ(x) + x·φ(x)` が `erf` を要し `Var` 演算の合成だけでは再現できない
// ため `scalar_unary_replayable` が引き続き `false` を返す
// （`docs/autodiff-higher-order-grad-decision.md` §16）。入口検査 7
// （`validate_ancestors`）が `build_mirror`／`build_cgrads` より前に
// 判定するため、拒否時に `child` が一切書き込まれない（空のまま）こと
// も併せて固定する。
#[test]
fn create_graph_rejects_unsupported_op_gelu() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let loss = x.gelu().unwrap().sum(None).unwrap();

    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
    assert!(
        child.is_empty(),
        "入口検査で拒否された場合、child は無変更（空）のまま保たれるはず"
    );
}

#[test]
fn create_graph_rejects_unsupported_op_softmax() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let loss = x.softmax(1).unwrap().sum(None).unwrap();

    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
    assert!(child.is_empty());
}

/// 恒等関数として振る舞う `CustomFunction`（`Op::Custom` を経由させる
/// ためだけの最小実装。forward はそのままコピー、backward は upstream
/// をそのまま入力へ流す）。
struct IdentityCustomFn;

impl CustomFunction for IdentityCustomFn {
    fn name(&self) -> &str {
        "identity_custom_fn"
    }

    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }

    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        Ok(inputs[0].contiguous())
    }

    fn backward(
        &self,
        _inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        if !requires_grad[0] {
            return Ok(vec![None]);
        }
        Ok(vec![Some(upstream.contiguous())])
    }
}

/// `Op::Custom`（イシュー #1946。ユーザー定義 forward／backward
/// プラグイン）は `create_graph` の対象外である（`Op::
/// supports_create_graph()` が `false` を返すため `validate_ancestors`
/// が子テープへ一切書き込む前に fail-closed 拒否する契約。
/// `docs/autodiff-higher-order-grad-decision.md` §8・`docs/autodiff-
/// custom-function-decision.md` §14 参照）。
#[test]
fn create_graph_rejects_unsupported_op_custom() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let identity = tape
        .custom(Arc::new(IdentityCustomFn), &[x])
        .expect("Tape::custom 登録成功");
    let loss = identity.sum(None).unwrap();

    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
    assert!(
        child.is_empty(),
        "入口検査で拒否された場合、child は無変更（空）のまま保たれるはず"
    );
}

// --- MatMul（rank 2 × rank 2）: 二次形式・Linear（bias 込み）・fail
//     -closed 契約（イシュー #1943） -------------------------------------

// `loss = sum((X·W) ⊙ (X·W))` — `X` は非追跡の定数、`W` を微分対象と
// する（`Op::MatMul` の `db` 腕〈`a_m.transpose(0,1)?.matmul(&g)?`〉を
// 実際に微分する）。Hessian は解析的に `2・(XᵀX) ⊗ I_n` に一致する
// （`d/dW_kl sum((XW)^2) = 2・(X^T (XW))_kl`・さらに `W` で微分すると
// `2・(X^T X)_ki・δ_ln`）。
fn build_matmul_quadratic_w<'t>(tape: &'t Tape, w: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let x = tape.var_no_grad(&t(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]));
    let y = x.matmul(w)?;
    y.mul(&y)?.sum(None)
}

#[test]
fn hessian_matmul_quadratic_w_matches_finite_difference_and_closed_form() {
    let w0 = [0.3f32, -0.7, 1.1, 0.2];
    let shape = [2usize, 2usize];

    let numeric = finite_diff_hessian(build_matmul_quadratic_w, &w0, &shape, 1e-3);
    let analytic = analytic_hessian(build_matmul_quadratic_w, &w0, &shape);
    assert_hessian_close(&analytic, &numeric);

    // X = [[1,2],[-1,0.5]] → XᵀX = [[2, 1.5], [1.5, 4.25]]（手計算）。
    // Hessian[w_kl][w_mn] = 2*(XᵀX)[k][m] if l==n else 0
    // （フラット添字は行優先: idx = k*2 + l）。
    let xtx = [[2.0f64, 1.5], [1.5, 4.25]];
    for k in 0..2 {
        for l in 0..2 {
            for m in 0..2 {
                for n in 0..2 {
                    let row = k * 2 + l;
                    let col = m * 2 + n;
                    let expected = if l == n { 2.0 * xtx[k][m] } else { 0.0 };
                    assert!(
                        common::req2_close(analytic[row][col], expected),
                        "hessian[{row}][{col}]: analytic={} expected={expected}",
                        analytic[row][col]
                    );
                }
            }
        }
    }
}

// 左オペランド側（`X·W` の `X` を微分対象にする）で `Op::MatMul` の
// `da` 腕（`g.matmul(&b_m.transpose(0,1)?)?`）を網羅する。
fn build_matmul_quadratic_x<'t>(tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let w = tape.var_no_grad(&t(vec![2.0, -1.0, 0.0, 1.0], &[2, 2]));
    let y = x.matmul(&w)?;
    y.mul(&y)?.sum(None)
}

#[test]
fn hessian_matmul_quadratic_x_matches_finite_difference() {
    let x0 = [1.0f32, -0.5, 0.3, 2.0];
    let shape = [2usize, 2usize];

    let numeric = finite_diff_hessian(build_matmul_quadratic_x, &x0, &shape, 1e-3);
    let analytic = analytic_hessian(build_matmul_quadratic_x, &x0, &shape);
    assert_hessian_close(&analytic, &numeric);

    // 非ゼロであることも確認する（da 腕が実際に微分されていることの
    // 健全性チェック。全ゼロだと Hessian 比較が自明成立してしまう）。
    let any_nonzero = analytic.iter().flatten().any(|&v| v.abs() > 1e-3);
    assert!(any_nonzero, "matmul quadratic (x) の Hessian が全ゼロ");
}

// `nn::Linear`（既定 forward: `MatMul` → `Add` bias）を経由する
// Hessian。`W`・`b` それぞれを微分対象にし、`Op::MatMul`＋`Op::Add`
// （bias パターン）の合成が子テープ上で正しく再生されることを確認する。
fn build_linear_forward_w<'t>(tape: &'t Tape, w: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let x = tape.var_no_grad(&t(vec![1.0, 0.5, -0.5, 2.0], &[2, 2]));
    let bias = tape.var_no_grad(&t(vec![0.1, -0.2], &[2]));
    let y = x.matmul(w)?.add(&bias)?;
    y.tanh().mul(&y.tanh())?.sum(None)
}

#[test]
fn hessian_linear_with_bias_w_matches_finite_difference() {
    let w0 = [0.4f32, -0.3, 0.6, 0.1];
    let shape = [2usize, 2usize];

    let numeric = finite_diff_hessian(build_linear_forward_w, &w0, &shape, 1e-3);
    let analytic = analytic_hessian(build_linear_forward_w, &w0, &shape);
    assert_hessian_close(&analytic, &numeric);
}

fn build_linear_forward_bias<'t>(tape: &'t Tape, b: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let x = tape.var_no_grad(&t(vec![1.0, 0.5, -0.5, 2.0], &[2, 2]));
    let w = tape.var_no_grad(&t(vec![0.3, -0.6, 0.9, 0.2], &[2, 2]));
    let y = x.matmul(&w)?.add(b)?;
    y.tanh().mul(&y.tanh())?.sum(None)
}

#[test]
fn hessian_linear_with_bias_b_matches_finite_difference() {
    let b0 = [0.2f32, -0.1];
    let shape = [2usize];

    let numeric = finite_diff_hessian(build_linear_forward_bias, &b0, &shape, 1e-3);
    let analytic = analytic_hessian(build_linear_forward_bias, &b0, &shape);
    assert_hessian_close(&analytic, &numeric);
}

// `nn::Linear::from_parameters(..).bind(&tape).forward(..)` 経由でも
// `Op::MatMul`＋`Op::Add` として記録され、同じく二階微分できることを
// 確認する（`Var::matmul`／`add` を手動で組む上記テストとの整合確認）。
#[test]
fn hessian_via_nn_linear_forward_matches_manual_composition() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 0.5, -0.5, 2.0], &[2, 2]));
    let weight = t(vec![0.3, -0.6, 0.9, 0.2], &[2, 2]);
    let bias = t(vec![0.1, -0.2], &[2]);
    let linear = Linear::from_parameters(weight.clone(), Some(bias.clone())).unwrap();
    let vars = linear.bind(&tape);
    let y = vars.forward(&x).unwrap();
    let loss = y.tanh().mul(&y.tanh()).unwrap().sum(None).unwrap();

    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gx = cg.grad(&x).unwrap().expect("x は loss に到達する");
    let cx = cg.child_var(&x).unwrap().expect("child_var は求まるはず");
    let row = child
        .backward(&extract_scalar(&gx, &[2, 2], 0).unwrap())
        .unwrap();
    let hx = row.get(&cx).unwrap().expect("2 階勾配は x へ到達するはず");

    // 同じ計算を手動 `matmul`＋`add` で組んだ場合と数値的に一致する
    // （`nn::Linear::forward` の VJP 記録が手動合成と同一の `Op` 列に
    // 帰着することを確認）。
    let tape2 = Tape::new_with_ops(common::naive_ops());
    let child2 = Tape::new_with_ops(common::naive_ops());
    let x2 = tape2.var(&t(vec![1.0, 0.5, -0.5, 2.0], &[2, 2]));
    let w2 = tape2.var_no_grad(&weight);
    let b2 = tape2.var_no_grad(&bias);
    let y2 = x2.matmul(&w2).unwrap().add(&b2).unwrap();
    let loss2 = y2.tanh().mul(&y2.tanh()).unwrap().sum(None).unwrap();
    let cg2 = tape2.backward_create_graph(&loss2, &child2).unwrap();
    let gx2 = cg2.grad(&x2).unwrap().expect("x2 は loss2 に到達する");
    let cx2 = cg2.child_var(&x2).unwrap().expect("child_var は求まるはず");
    let row2 = child2
        .backward(&extract_scalar(&gx2, &[2, 2], 0).unwrap())
        .unwrap();
    let hx2 = row2
        .get(&cx2)
        .unwrap()
        .expect("2 階勾配は x2 へ到達するはず");

    for i in 0..4 {
        let a = hx.get(&unravel(i, &[2, 2])).unwrap() as f64;
        let b = hx2.get(&unravel(i, &[2, 2])).unwrap() as f64;
        assert!(
            common::req2_close(a, b),
            "nn::Linear 経由と手動合成の 2 階勾配が乖離した: {a} vs {b}"
        );
    }
}

// --- HVP（ヘッセ・ベクトル積）: 小型 MLP（Linear→tanh→Linear）の例
//     ---------------------------------------------------------------------
//
// `H·v` を各パラメータについて `Σ_p grad_p(θ)・v_p` を子テープ上で
// 合成し再度 `backward` することで求める（PyTorch の
// `torch.autograd.grad(grad, params, grad_outputs=v)` 相当）。独立な
// 方向微分の中央差分 `(∇L(θ+εv) − ∇L(θ−εv)) / 2ε`（`create_graph` を
// 一切経由しない別経路）と突合する。

struct MlpParams {
    w1: Tensor<f32>,
    b1: Tensor<f32>,
    w2: Tensor<f32>,
    b2: Tensor<f32>,
}

fn mlp_params(seed_scale: f32) -> MlpParams {
    MlpParams {
        w1: t(
            vec![0.2 * seed_scale, -0.3, 0.1, 0.4, -0.1, 0.2, 0.3, -0.2, 0.05],
            &[3, 3],
        ),
        b1: t(vec![0.1, -0.1, 0.05], &[3]),
        w2: t(vec![0.3, -0.2, 0.1], &[3, 1]),
        b2: t(vec![0.05], &[1]),
    }
}

/// `grad_at(θ)`（`create_graph` 非依存の素の `Tape::backward`）を返す
/// 独立経路（`analytic_hessian` と同じく create_graph 実装バグを検出
/// するための別経路）。
fn mlp_grads(
    x_const: &Tensor<f32>,
    target_const: &Tensor<f32>,
    p: &MlpParams,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let w1 = tape.var(&p.w1);
    let b1 = tape.var(&p.b1);
    let w2 = tape.var(&p.w2);
    let b2 = tape.var(&p.b2);
    let x = tape.var_no_grad(x_const);
    let target = tape.var_no_grad(target_const);
    let h = x.matmul(&w1).unwrap().add(&b1).unwrap().tanh();
    let pred = h.matmul(&w2).unwrap().add(&b2).unwrap();
    let diff = pred.sub(&target).unwrap();
    let loss = diff.mul(&diff).unwrap().mean(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let flat = |v: &Var<'_>, shape: &[usize]| -> Vec<f32> {
        let g = grads.get(v).unwrap().expect("param は loss に到達する");
        (0..flat_len(shape))
            .map(|i| g.get(&unravel(i, shape)).unwrap())
            .collect()
    };
    (
        flat(&w1, &[3, 3]),
        flat(&b1, &[3]),
        flat(&w2, &[3, 1]),
        flat(&b2, &[1]),
    )
}

fn perturb(base: &MlpParams, dirs: &MlpParams, eps: f32) -> MlpParams {
    MlpParams {
        w1: t(
            base.w1
                .contiguous()
                .as_slice()
                .unwrap()
                .iter()
                .zip(dirs.w1.contiguous().as_slice().unwrap().iter())
                .map(|(&b, &d)| b + eps * d)
                .collect(),
            base.w1.shape(),
        ),
        b1: t(
            base.b1
                .contiguous()
                .as_slice()
                .unwrap()
                .iter()
                .zip(dirs.b1.contiguous().as_slice().unwrap().iter())
                .map(|(&b, &d)| b + eps * d)
                .collect(),
            base.b1.shape(),
        ),
        w2: t(
            base.w2
                .contiguous()
                .as_slice()
                .unwrap()
                .iter()
                .zip(dirs.w2.contiguous().as_slice().unwrap().iter())
                .map(|(&b, &d)| b + eps * d)
                .collect(),
            base.w2.shape(),
        ),
        b2: t(
            base.b2
                .contiguous()
                .as_slice()
                .unwrap()
                .iter()
                .zip(dirs.b2.contiguous().as_slice().unwrap().iter())
                .map(|(&b, &d)| b + eps * d)
                .collect(),
            base.b2.shape(),
        ),
    }
}

#[test]
fn hvp_small_mlp_matches_finite_difference() {
    let x_const = t(
        vec![1.0, -0.5, 0.3, 0.2, 0.8, -0.3, -0.4, 0.6, 0.1],
        &[3, 3],
    );
    let target_const = t(vec![0.5, -0.2, 0.1], &[3, 1]);
    let theta = mlp_params(1.0);
    // 方向ベクトル v（正規化はしない。方向微分の中央差分と比較する
    // だけなので任意のスケールで良い）。
    let v = MlpParams {
        w1: t(vec![1.0, 0.0, -1.0, 0.5, 0.0, 0.0, 0.0, 1.0, -0.5], &[3, 3]),
        b1: t(vec![0.5, -0.5, 1.0], &[3]),
        w2: t(vec![1.0, -1.0, 0.5], &[3, 1]),
        b2: t(vec![1.0], &[1]),
    };

    // --- create_graph 経由の解析的 HVP ---
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let w1 = tape.var(&theta.w1);
    let b1 = tape.var(&theta.b1);
    let w2 = tape.var(&theta.w2);
    let b2 = tape.var(&theta.b2);
    let loss = mlp_loss_from_vars(&tape, &x_const, &target_const, &w1, &b1, &w2, &b2).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();

    let params: [(&Var<'_>, &Tensor<f32>, &[usize]); 4] = [
        (&w1, &v.w1, &[3, 3]),
        (&b1, &v.b1, &[3]),
        (&w2, &v.w2, &[3, 1]),
        (&b2, &v.b2, &[1]),
    ];
    // `s = Σ_p grad_p・v_p`（子テープ上で合成し、これを再度
    // `child.backward` すれば `H・v` が各パラメータ勾配として得られる）。
    let mut s: Option<Var<'_>> = None;
    for (p, vp, _shape) in params.iter() {
        let gp = cg.grad(p).unwrap().expect("param は loss に到達する");
        let vp_var = child.var_no_grad(vp);
        let term = gp.mul(&vp_var).unwrap().sum(None).unwrap();
        s = Some(match s {
            Some(acc) => acc.add(&term).unwrap(),
            None => term,
        });
    }
    let s = s.unwrap();
    let hvp_grads = child.backward(&s).unwrap();

    let hvp_flat = |p: &Var<'_>, shape: &[usize]| -> Vec<f64> {
        let cx = cg.child_var(p).unwrap().expect("child_var は求まるはず");
        let g = hvp_grads
            .get(&cx)
            .unwrap()
            .expect("2 階勾配は param へ到達するはず");
        (0..flat_len(shape))
            .map(|i| g.get(&unravel(i, shape)).unwrap() as f64)
            .collect()
    };
    let hw1 = hvp_flat(&w1, &[3, 3]);
    let hb1 = hvp_flat(&b1, &[3]);
    let hw2 = hvp_flat(&w2, &[3, 1]);
    let hb2 = hvp_flat(&b2, &[1]);

    // --- 独立経路: 方向微分の中央差分（create_graph 非依存） ---
    let eps = 1.0e-3f32;
    let theta_p = perturb(&theta, &v, eps);
    let theta_m = perturb(&theta, &v, -eps);
    let (gp_w1, gp_b1, gp_w2, gp_b2) = mlp_grads(&x_const, &target_const, &theta_p);
    let (gm_w1, gm_b1, gm_w2, gm_b2) = mlp_grads(&x_const, &target_const, &theta_m);

    let fd = |gp: &[f32], gm: &[f32]| -> Vec<f64> {
        gp.iter()
            .zip(gm.iter())
            .map(|(&p, &m)| (p as f64 - m as f64) / (2.0 * eps as f64))
            .collect()
    };
    let fd_w1 = fd(&gp_w1, &gm_w1);
    let fd_b1 = fd(&gp_b1, &gm_b1);
    let fd_w2 = fd(&gp_w2, &gm_w2);
    let fd_b2 = fd(&gp_b2, &gm_b2);

    for (a, n) in hw1.iter().zip(fd_w1.iter()) {
        assert!(common::req2_close(*a, *n), "H·v (w1) mismatch: {a} vs {n}");
    }
    for (a, n) in hb1.iter().zip(fd_b1.iter()) {
        assert!(common::req2_close(*a, *n), "H·v (b1) mismatch: {a} vs {n}");
    }
    for (a, n) in hw2.iter().zip(fd_w2.iter()) {
        assert!(common::req2_close(*a, *n), "H·v (w2) mismatch: {a} vs {n}");
    }
    for (a, n) in hb2.iter().zip(fd_b2.iter()) {
        assert!(common::req2_close(*a, *n), "H·v (b2) mismatch: {a} vs {n}");
    }
}

#[allow(clippy::too_many_arguments)]
fn mlp_loss_from_vars<'t>(
    tape: &'t Tape,
    x_const: &Tensor<f32>,
    target_const: &Tensor<f32>,
    w1: &Var<'t>,
    b1: &Var<'t>,
    w2: &Var<'t>,
    b2: &Var<'t>,
) -> Result<Var<'t>, AutodiffError> {
    let x = tape.var_no_grad(x_const);
    let target = tape.var_no_grad(target_const);
    let neg_target_data: Vec<f32> = target
        .value()
        .contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| -v)
        .collect();
    let neg_target = tape.var_no_grad(&t(neg_target_data, target_const.shape()));
    let h = x.matmul(w1)?.add(b1)?.tanh();
    let pred = h.matmul(w2)?.add(b2)?;
    let diff = pred.add(&neg_target)?;
    diff.mul(&diff)?.mean(None)
}

#[test]
fn create_graph_rejects_fused_linear_act() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let weight = t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let linear = Linear::from_parameters(weight, None).unwrap();
    let vars = linear.bind(&tape);
    let loss = vars
        .forward_with_activation(&x, Activation::None)
        .unwrap()
        .sum(None)
        .unwrap();

    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
    assert!(child.is_empty());
}

#[test]
fn create_graph_rejects_rank3_matmul() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    // batch=2 の rank 3 matmul（`Var::matmul` は #1715 でバッチ次元を
    // 受理するが、子テープ側の VJP は rank 2 限定のため事前拒否する）。
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]));
    let b = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]));
    let loss = a.matmul(&b).unwrap().sum(None).unwrap();

    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
    assert!(child.is_empty());
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

// --- checkpoint との相互作用（codex-review 指摘。PR #2003・イシュー
//     #1943） ---------------------------------------------------------

/// [`common::NaiveOps`] に薄い計装をかける `BackendOps` ラッパー。
/// `gemm`（非厳密）と `gemm_fp32_strict`（厳密）の呼び出し回数を
/// **別々の**カウンタへ記録する（`checkpoint_review_1624.rs::
/// InstrumentedOps` と同型だが、`gemm_fp32_strict` を素通しで
/// `self.gemm(..)` へ委譲する `BackendOps` の既定実装は使わず、両者を
/// 独立に計装する点が異なる——既定実装のまま `gemm` だけを計装すると
/// `gemm_fp32_strict` 呼び出しが自動的に `gemm` カウンタへ混入し、
/// 「厳密経路と非厳密経路のどちらが呼ばれたか」を区別できない）。
struct GemmPathCountingOps {
    inner: Box<dyn fandhe_ai_tensor_core::BackendOps + Send>,
    gemm_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    gemm_fp32_strict_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl fandhe_ai_tensor_core::BackendOps for GemmPathCountingOps {
    fn device(&self) -> fandhe_ai_tensor_core::Device {
        self.inner.device()
    }

    fn gemm(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.gemm_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.gemm(a, b)
    }

    fn gemm_fp32_strict(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.gemm_fp32_strict_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // `self.inner.gemm(..)` へ直接委譲する（`self.gemm(..)` を経由
        // すると `gemm_calls` も同時に加算されてしまい、厳密経路単独の
        // 呼び出し回数を計装できなくなるため）。`common::NaiveOps` は
        // TF32 の概念を持たないため forward 値自体は `gemm` と同一。
        self.inner.gemm(a, b)
    }

    fn gemm_checksum(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        readout: fandhe_ai_tensor_core::ChecksumReadout,
    ) -> Result<fandhe_ai_tensor_core::GemmChecksum, fandhe_ai_tensor_core::BackendError> {
        self.inner.gemm_checksum(a, b, readout)
    }

    fn add(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.add(a, b)
    }

    fn mul(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.mul(a, b)
    }

    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.relu(a)
    }

    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.exp(a)
    }

    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.tanh(a)
    }

    fn sum(
        &self,
        a: &Tensor<f32>,
        dim: Option<usize>,
    ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.sum(a, dim)
    }

    fn max(
        &self,
        a: &Tensor<f32>,
        dim: Option<usize>,
    ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.max(a, dim)
    }
}

/// **P1 是正の回帰**（codex-review 指摘。PR #2003）: `create_graph` の
/// 子テープ上で `Op::MatMul` VJP（`build_cgrads` が `Var::
/// matmul_fp32_strict` で記録する `da`／`db`）から作った `Var` を
/// `Var::checkpoint_from` で checkpoint 区間の外側入力にしても、
/// 後続の `Tape::backward` が非厳密な `matmul_forward`（`ops.gemm`）
/// で再計算しない（＝解放対象から除外される）ことを確認する。
///
/// **シナリオ（指摘本文と同型）**: `y = x.matmul(&w)`・
/// `loss = sum(y*y)` から `create_graph` で子テープ上の `db`
/// （`gw = cg.grad(&w)`）を得たうえで、`h = gw.mul(&gw)?.sum(None)?`
/// を構築し `h.checkpoint_from(&[])`（区間 `[0, h)`。`gw` を含む）を
/// 呼ぶ。この時点で `is_checkpoint_eligible() == true` な `Op::MatMul`
/// はそのままでは解放対象になるが、`gw` の `TapeNode::fp32_strict` が
/// 立っているため実際には解放されない（`release_checkpoint_region`
/// doc 参照）。続けて `child.backward(&h)` を呼ぶと、`Op::Mul` の VJP
/// （`grad.rs`）が `gw` の値を `materialize_fallible` で読むが、値が
/// 解放されていなければ再計算（`recompute_value` の `Op::MatMul` 分岐。
/// 常に非厳密な `matmul_forward`／`ops.gemm` を使う）は一切発生しない
/// ——子テープの `Op::MatMul` は本テストを通じて `matmul_fp32_strict`
/// でしか作られないため、`gemm_calls`（非厳密カウンタ）が 0 のまま
/// なら「非厳密経路が一度も使われなかった」ことの直接証拠になる
/// （修正前は checkpoint 解放 → `child.backward` の再計算で `gemm_
/// calls` が 1 以上へ増加し、本テストは失敗していたはずである）。
#[test]
fn create_graph_matmul_fp32_strict_grad_survives_checkpoint_release() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let parent_gemm_calls = Arc::new(AtomicUsize::new(0));
    let parent_strict_calls = Arc::new(AtomicUsize::new(0));
    let tape = Tape::new_with_ops(Box::new(GemmPathCountingOps {
        inner: common::naive_ops(),
        gemm_calls: parent_gemm_calls,
        gemm_fp32_strict_calls: parent_strict_calls,
    }));

    let child_gemm_calls = Arc::new(AtomicUsize::new(0));
    let child_strict_calls = Arc::new(AtomicUsize::new(0));
    let child = Tape::new_with_ops(Box::new(GemmPathCountingOps {
        inner: common::naive_ops(),
        gemm_calls: Arc::clone(&child_gemm_calls),
        gemm_fp32_strict_calls: Arc::clone(&child_strict_calls),
    }));

    let x = tape.var(&t(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]));
    let w = tape.var(&t(vec![0.3, -0.7, 1.1, 0.2], &[2, 2]));
    let y = x.matmul(&w).expect("matmul: forward");
    // `loss = sum(y)`（**二次形式にしない**）: `Op::Sum` の VJP は
    // upstream をそのまま `Var::broadcast_to`（view。`y` 自身の値を
    // 読まない）で広げるだけのため、`Op::MatMul(x, w)` VJP へ流れ込む
    // upstream `g` は `y` の forward mirror（`y_mirror`。`build_mirror`
    // が非厳密 `Var::matmul` で再生する、checkpoint 解放されうる別の
    // `Op::MatMul` ノード）へ一切依存しない。`loss = sum(y*y)`
    // （二次形式）だと upstream が `2・g_loss・y_mirror` になり、`gw`
    // 自身の 2 階 VJP（`child.backward` が `gw` を `Op::MatMul` として
    // 逆伝播する際の入力再構築）が `y_mirror` の値を要求してしまい、
    // `y_mirror`（元々 `fp32_strict` の対象外——`Var::matmul` 経由の
    // forward 写しであり厳密精度契約を持たない）の非厳密再計算という
    // 無関係な `gemm` 呼び出しが本テストの意図（`gw` 自身の checkpoint
    // 解放除外の検証）に混入してしまう。線形にすることでこの混入を
    // 避け、`gw` の再計算経路だけを単離する。
    let loss = y.sum(None).expect("sum");

    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph に失敗");
    let gw = cg
        .grad(&w)
        .expect("grad: クロステープ検査は通るはず")
        .expect("w は loss に到達するはず（`db` 腕）");

    // ここまでで `build_cgrads` の `Op::MatMul` 腕（`da`／`db`）が子
    // テープ上へ厳密経路で記録されているはず。`build_mirror`（forward
    // 値の写し。VJP 式自体の正しさに関わるのみで checkpoint 解放除外
    // 契約の対象外）は `Var::matmul`（非厳密）で再生するため、この
    // 時点までに非厳密 `gemm` が既に 1 回以上呼ばれていてよい——以降の
    // 比較はこの時点の値を**基準（baseline）**として、それ以上
    // 増えないことだけを検証する。
    assert!(
        child_strict_calls.load(Ordering::SeqCst) > 0,
        "build_cgrads の MatMul VJP が gemm_fp32_strict を呼んでいない"
    );
    let baseline_gemm_calls = child_gemm_calls.load(Ordering::SeqCst);

    let h = gw.mul(&gw).expect("mul").sum(None).expect("sum");
    let checkpointed = h
        .checkpoint_from(&[])
        .expect("checkpoint_from: 区間 [0, h) の登録に失敗");

    // checkpoint 解放（`register_checkpoint`。`checkpoint_from` 内で
    // forward 直後に同期実行）の直後時点では非厳密 gemm はまだ増えて
    // いない（解放自体は値を落とすだけで再計算を即座には起こさない）。
    assert_eq!(
        child_gemm_calls.load(Ordering::SeqCst),
        baseline_gemm_calls,
        "checkpoint_from の解放処理自体が非厳密な gemm を呼んでいる"
    );

    child
        .backward(&checkpointed)
        .expect("child.backward（checkpoint 解放済み gw 経由）に失敗");

    // 本テストの核心: checkpoint 解放済みの `gw`（`Op::MatMul`。
    // `fp32_strict` フラグにより解放対象から除外されているはず）の
    // 再計算が `Tape::backward` の VJP 経由で発生しても、非厳密な
    // `matmul_forward`（`ops.gemm`）は基準から一切増えない。
    assert_eq!(
        child_gemm_calls.load(Ordering::SeqCst),
        baseline_gemm_calls,
        "checkpoint 解放済み gw の再計算が非厳密な gemm（matmul_forward）を \
         使った——TapeNode::fp32_strict による checkpoint 解放除外が機能していない"
    );
}

/// [`GemmPathCountingOps`] と異なり `gemm`（非厳密）と
/// `gemm_fp32_strict`（厳密）が**数値的に異なる結果**を返す
/// `BackendOps` ラッパー。call-count 計装だけでは
/// `build_cgrads`（`Op::MatMul` 腕。既に `matmul_fp32_strict` 使用済み
/// で本 PR の対象外）由来の呼び出しと [`build_mirror`] 由来の呼び出し
/// を区別できない（後者が対象の gx ノードへ到達する経路には前者も
/// 必ず伴うため）ので、[`create_graph_nested_build_mirror_replays_matmul_fp32_strict`]
/// では `mirror`（forward 値の写し。VJP 経路を経由しない）を直接
/// 読み出して数値そのもので区別する。`gemm` を全 0 埋めにすることで、
/// もし `build_mirror` が誤って非厳密経路へ落ちれば再生値が明確に
/// 崩れる。
struct WrongNonStrictOps {
    inner: Box<dyn fandhe_ai_tensor_core::BackendOps + Send>,
}

impl fandhe_ai_tensor_core::BackendOps for WrongNonStrictOps {
    fn device(&self) -> fandhe_ai_tensor_core::Device {
        self.inner.device()
    }

    fn gemm(
        &self,
        a: &Tensor<f32>,
        _b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        // 非厳密経路は意図的に破損した値（全 0）を返す——正しい形状は
        // 維持しつつ、正しい厳密経路の結果とは必ず異なる値にする。
        let m = a.shape()[0];
        let n = _b.shape()[1];
        Ok(t(vec![0.0f32; m * n], &[m, n]))
    }

    fn gemm_fp32_strict(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.gemm(a, b)
    }

    fn gemm_checksum(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        readout: fandhe_ai_tensor_core::ChecksumReadout,
    ) -> Result<fandhe_ai_tensor_core::GemmChecksum, fandhe_ai_tensor_core::BackendError> {
        self.inner.gemm_checksum(a, b, readout)
    }

    fn add(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.add(a, b)
    }

    fn mul(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.mul(a, b)
    }

    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.relu(a)
    }

    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.exp(a)
    }

    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.tanh(a)
    }

    fn sum(
        &self,
        a: &Tensor<f32>,
        dim: Option<usize>,
    ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.sum(a, dim)
    }

    fn max(
        &self,
        a: &Tensor<f32>,
        dim: Option<usize>,
    ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
        self.inner.max(a, dim)
    }
}

/// PR #2003 codex-review 指摘（threadId `PRRT_kwDOTuUCJc6jRs6Z`）の
/// 再現テスト: `create_graph` を 2 段重ねた三階微分相当の経路
/// （`tape.backward_create_graph` → `child.backward_create_graph`）で、
/// 2 段目の [`build_mirror`]（`create_graph.rs`。非公開関数のため直接
/// 呼べず本テストは公開 API 経由で間接検証する）が 1 段目の
/// `build_cgrads` が `Var::matmul_fp32_strict` で記録した `gx`
/// （`Op::MatMul`。`TapeNode::fp32_strict == true`）を再生する際、
/// 修正前は無条件に非厳密 `Var::matmul`（`ops.gemm`）を使っていた——
/// `fp32_strict` フラグを `replay_op` へ伝播しないため。
///
/// `w` を非追跡葉（`tape.var_no_grad`）にすることで、指摘が挙げた
/// 「`transpose(w)` が非追跡のため `validate_ancestors` の rank 検査
/// を通過してしまう」具体シナリオを再現する。`loss = sum(y*z)`
/// （`z` は `y` と同形状の**別の**追跡葉。`y=x@w` に対して**線形**
/// なので `dL/dy = z` となり、`gx`〈`x` についての勾配〉の入力
/// （`g=z` の写し・`wᵀ` の写し）はいずれも `Op::Leaf`／非追跡葉
/// 由来で `build_mirror` の 段 1〈キャッシュ済み値の読み出しのみ・
/// 演算呼び出しなし〉で再生される——`loss=sum((x@w)^2)` のような
/// 二次形式だと `dL/dy` が `y` の写し〈`Op::MatMul`。`fp32_strict ==
/// false` が正しい非厳密ノード〉に依存してしまい、後述の
/// [`WrongNonStrictOps`] がその**正当な**非厳密再生まで巻き込んで
/// 壊してしまうため使えない）の勾配 `gx`（`x` について）を子テープへ
/// 作り、`h = sum(gx)`（`gx` を「使う」だけで gx 自身の値そのものを
/// ancestor mirror へ反映させれば十分——`Op::Sum` の VJP は `gx` の
/// mirror 値を読まないため build_cgrads 側の寄与を持ち込まない）へ
/// 再度 `backward_create_graph` を適用し、[`WrongNonStrictOps`] 下で
/// `cg2.child_var(&gx)`（`build_mirror` が再生した gx の写し。VJP を
/// 経由しない純粋な forward 値）を読み出す。孫テープの `gemm`
/// （非厳密）は意図的に破損した値を返すため、`build_mirror` が
/// `fp32_strict` を無視して非厳密経路へ落ちていれば写しの値が
/// `child` 側の本来の `gx` の値と食い違う。
#[test]
fn create_graph_nested_build_mirror_replays_matmul_fp32_strict() {
    // 1 段目（parent）は数値方式の計装が不要なため素の naive_ops。
    let tape = Tape::new_with_ops(common::naive_ops());
    // 2 段目の対象（1 階勾配 gx を記録する子テープ）も同様。
    let child = Tape::new_with_ops(common::naive_ops());

    let x = tape.var(&t(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]));
    // `w` は非追跡葉——指摘シナリオの前提（`transpose(w)` 側が
    // requires_grad を持たないため rank 検査を素通りする）。
    let w = tape.var_no_grad(&t(vec![0.3, -0.7, 1.1, 0.2], &[2, 2]));
    // `z` は `y` と同形状の別の追跡葉（`y` 自身への依存を断ち切る
    // ための線形化）。
    let z = tape.var(&t(vec![0.4, -1.2, 0.9, 2.1], &[2, 2]));
    let y = x.matmul(&w).expect("matmul: forward");
    let loss = y.mul(&z).expect("mul").sum(None).expect("sum");

    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph（1 段目）に失敗");
    let gx = cg
        .grad(&x)
        .expect("grad: クロステープ検査は通るはず")
        .expect("x は loss に到達するはず（da 腕）");
    // `child` 自身の naive_ops（正しい fp32_strict 経路）で計算済みの
    // 正解値。`gx` の `Op::MatMul` は `matmul_fp32_strict` 経由なので
    // `child` の gemm／gemm_fp32_strict は常に同一実装（naive_ops）
    // へ帰着し、この値は「厳密経路で計算した正しい gx」そのもの。
    let expected = gx.to_tensor();

    // 3 段目（孫テープ）: `gemm`（非厳密）が意図的に破損した値を返す。
    let grandchild = Tape::new_with_ops(Box::new(WrongNonStrictOps {
        inner: common::naive_ops(),
    }));

    // `gx` を「使う」だけの極小 loss（`Op::Sum` の VJP は mirror 値を
    // 読まないため、build_cgrads 側の gx 自身の逆伝播は本テストの
    // 数値検証に混入しない——検証対象は build_mirror が構築する
    // 写しの値そのもの）。
    let h = gx.sum(None).expect("sum");

    let cg2 = child
        .backward_create_graph(&h, &grandchild)
        .expect("backward_create_graph（2 段目）に失敗");

    let gx_mirror = cg2
        .child_var(&gx)
        .expect("child_var: クロステープ検査は通るはず")
        .expect("gx は h の祖先のはず（build_mirror が写しを構築する）");

    // 本テストの核心: `build_mirror` が `gx`（`TapeNode::fp32_strict
    // == true`）を再生した写しの値が、正しい厳密経路の値
    // （`expected`）と一致すること。修正前は非厳密 `gemm`（意図的に
    // 破損した値を返す）を使うため、この写しの値は 0 埋めになり
    // `expected` と食い違う。
    let mirror_value = gx_mirror.to_tensor();
    assert_eq!(
        mirror_value.as_slice(),
        expected.as_slice(),
        "build_mirror が fp32_strict なノード（gx）の再生に非厳密 gemm \
         を使った——TapeNode::fp32_strict が build_mirror（replay_op）\
         へ伝播していない（写しの値: {:?}・期待値: {:?}）",
        mirror_value.as_slice(),
        expected.as_slice()
    );
}

// =========================================================================
// イシュー #2062: 高階微分（create_graph）対象 Op の残り追加実装。
// `Op::supports_create_graph()` を `ScalarUnary`（`Gelu`／`GeluTanh` を
// 除く）・`ScalarBinary`（既知 13 variant）・`Transpose`・`Permute`・
// `Narrow`・`Concat`・`Contiguous`・`Where` へ拡張した分の受け入れ
// テスト。既存パターン（`finite_diff_hessian` との独立クロスチェック）
// をそのまま踏襲する。線形演算（`Transpose`／`Permute`／`Narrow`／
// `Concat`／`Contiguous`）は単体では Hessian が恒等的に 0 になるため、
// `Mul` による二次形式と合成してから検証する（kink を避けるテスト点を
// 選ぶ）。
// =========================================================================

// --- ScalarBinary: Sub（二次形式との合成で非ゼロ Hessian を検証） ------

fn build_sub_quadratic<'t>(tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let c = tape.var_no_grad(&t(vec![0.3, -0.7, 1.1], &[3]));
    let y = x.sub(&c)?;
    y.mul(&y)?.sum(None)
}

#[test]
fn hessian_sub_quadratic_matches_finite_difference_and_closed_form() {
    // loss = sum((x - c)^2) => d^2/dx_i^2 = 2, 非対角は 0。
    let x0 = [1.0f32, -2.0, 0.5];
    let numeric = finite_diff_hessian(build_sub_quadratic, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_sub_quadratic, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);
    for i in 0..3 {
        assert!(
            common::req2_close(analytic[i][i], 2.0),
            "diag[{i}]: analytic={} expected=2.0",
            analytic[i][i]
        );
    }
}

// --- ScalarBinary: Div（両入力とも追跡対象。0 除算・特異点を避ける） ---

fn build_div<'t>(tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    // `x / (x + c)`（分母も `x` に依存する非線形式。`x` のみの単純な
    // 定数除算だと 2 階微分が恒等的に 0 になり、`x` の子テープ上の
    // 写しへ 2 階勾配が到達しない〈`Gradients::get` が正しく `None`
    // を返す〉ため `analytic_hessian` の `expect` が失敗する）。
    let c = tape.var_no_grad(&t(vec![3.0, 4.0, 2.5], &[3]));
    let denom = x.add(&c)?;
    x.div(&denom)?.sum(None)
}

#[test]
fn hessian_div_matches_finite_difference() {
    let x0 = [1.0f32, -2.0, 0.5];
    let numeric = finite_diff_hessian(build_div, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_div, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);
}

// --- ScalarBinary: Pow（Var ^ Var。定義域を正に保つ） -------------------

fn build_pow<'t>(tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let exponent = tape.var_no_grad(&t(vec![2.5, 3.0, 1.7], &[3]));
    x.pow(&exponent)?.sum(None)
}

#[test]
fn hessian_pow_matches_finite_difference() {
    let x0 = [1.2f32, 2.3, 0.8];
    let numeric = finite_diff_hessian(build_pow, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_pow, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);
}

// --- ScalarBinary: 比較演算（区分定数。両勾配とも恒等的にゼロ） --------

#[test]
fn create_graph_comparison_op_grad_is_always_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let threshold = tape.var(&t(vec![1.5, 1.5, 1.5], &[3]));
    let loss = x.gt(&threshold).unwrap().sum(None).unwrap();

    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph に失敗（Op::ScalarBinary::Gt は対象）");
    let gx = cg
        .grad(&x)
        .expect("grad: クロステープ検査は通るはず")
        .expect("x は loss に到達するはず");
    let data = gx.to_tensor();
    for &v in data.contiguous().as_slice().unwrap() {
        assert_eq!(v, 0.0, "比較演算の勾配は恒等的に 0 のはず");
    }
}

// --- ScalarUnary: 滑らかな合成（Sqrt・Log・Sin・Silu） ------------------

fn build_sqrt_log_sin<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.sqrt()?.add(&x.log()?)?.add(&x.sin()?)?.sum(None)
}

#[test]
fn hessian_sqrt_log_sin_matches_finite_difference() {
    let x0 = [1.2f32, 2.5, 0.7];
    let numeric = finite_diff_hessian(build_sqrt_log_sin, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_sqrt_log_sin, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);
}

fn build_silu<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.silu()?.sum(None)
}

#[test]
fn hessian_silu_matches_finite_difference() {
    let x0 = [0.6f32, -1.3, 2.1];
    let numeric = finite_diff_hessian(build_silu, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_silu, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);
}

fn build_hardswish<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.hardswish()?.sum(None)
}

#[test]
fn hessian_hardswish_interior_matches_finite_difference() {
    // kink（±3・0 近傍）から離れたテスト点。
    let x0 = [-2.0f32, 0.5, 2.0];
    let numeric = finite_diff_hessian(build_hardswish, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_hardswish, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);
}

#[test]
fn hessian_hardswish_saturation_boundary_uses_constant_branch() {
    // `create_graph.rs::build_cgrads` の `Hardswish` 分岐マスク境界
    // （`mask_from_pred` の `<=`/`>=`）を `tensor-core::scalar_op::
    // hardswish_grad` の飽和域境界規約（`x <= -3.0` で勾配 0・
    // `x >= 3.0` で勾配 1、いずれも定数域で 2 階微分 0）へ一致させる
    // 回帰テスト（中断作業からの復旧・イシュー #2062。旧 `</>` 判定
    // だと境界ちょうどが中間式 `(2x+3)/6` の左極限側へ誤って分類され
    // 曲率 1/3 が漏れ出ていた）。
    //
    // 境界ちょうど（`x == ±3.0`）は 1 階導関数自体が不連続な kink 点
    // のため有限差分突合の対象にできない——本テストは解析 Hessian が
    // 飽和域（曲率 0）を使っていることを直接検査する。中間域内部
    // （index 2）は従来どおり closed form（傾き `1/3`）と突合する。
    let x0 = [3.0f32, -3.0, 0.6];
    let analytic = analytic_hessian(build_hardswish, &x0, &[3]);
    assert!(
        analytic[0][0].abs() < 1e-6,
        "x=3.0（hi 飽和域境界）で曲率が非ゼロ: {}",
        analytic[0][0]
    );
    assert!(
        analytic[1][1].abs() < 1e-6,
        "x=-3.0（lo 飽和域境界）で曲率が非ゼロ: {}",
        analytic[1][1]
    );
    assert!(
        common::req2_close(analytic[2][2], 1.0 / 3.0),
        "中間域内部の曲率が (2x+3)/6 の傾き 1/3 と一致しない: {}",
        analytic[2][2]
    );
}

// --- ScalarUnary: 区分定数（kink を避けたテスト点） ---------------------

fn build_abs_leaky_clamp<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    // `abs`／`leaky_relu`／`clamp` はいずれもテスト点の近傍で区分
    // *線形*（2 階微分が恒等的に 0）のため、`x.mul(x)` の二次項を
    // 加えて genuine な曲率を持たせる（さもないと `x` の子テープ上の
    // 写しへ 2 階勾配が到達せず `analytic_hessian` の `expect` が
    // 失敗する。`build_div` と同じ理由）。
    x.abs()?
        .add(&x.leaky_relu(0.1)?)?
        .add(&x.clamp(-1.0, 1.0)?)?
        .add(&x.mul(x)?)?
        .sum(None)
}

#[test]
fn hessian_abs_leaky_relu_clamp_matches_finite_difference() {
    // kink（`abs` の 0・`clamp` の境界 ±1.0）から離れたテスト点。
    let x0 = [1.7f32, -0.6, 0.3];
    let numeric = finite_diff_hessian(build_abs_leaky_clamp, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_abs_leaky_clamp, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);
}

fn build_elu_softplus<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.elu(1.3)?.add(&x.softplus(1.0, 20.0)?)?.sum(None)
}

#[test]
fn hessian_elu_softplus_matches_finite_difference() {
    let x0 = [0.9f32, -1.4, 2.2];
    let numeric = finite_diff_hessian(build_elu_softplus, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_elu_softplus, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);
}

// --- 構造系 Op（線形。`Mul` の二次形式と合成して非ゼロ Hessian を検証） -

fn build_transpose_quadratic<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    // x: [2, 2] -> y = x^T -> sum(y * y) は sum(x * x) と同値だが
    // `Op::Transpose` の replay／VJP を経由させる。
    let y = x.transpose(0, 1)?;
    y.mul(&y)?.sum(None)
}

#[test]
fn hessian_transpose_quadratic_matches_finite_difference() {
    let x0 = [1.0f32, -2.0, 0.5, 3.0];
    let numeric = finite_diff_hessian(build_transpose_quadratic, &x0, &[2, 2], 1e-3);
    let analytic = analytic_hessian(build_transpose_quadratic, &x0, &[2, 2]);
    assert_hessian_close(&analytic, &numeric);
}

fn build_permute_quadratic<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let y = x.permute(&[2, 0, 1])?;
    y.mul(&y)?.sum(None)
}

#[test]
fn hessian_permute_quadratic_matches_finite_difference() {
    let x0 = [1.0f32, -2.0, 0.5, 3.0, -1.5, 0.2, 0.7, -0.4];
    let numeric = finite_diff_hessian(build_permute_quadratic, &x0, &[2, 2, 2], 1e-3);
    let analytic = analytic_hessian(build_permute_quadratic, &x0, &[2, 2, 2]);
    assert_hessian_close(&analytic, &numeric);
}

fn build_narrow_quadratic<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    // 前後どちらにも非空区間が残る narrow（`Var::cat` の 3 分割経路）。
    let y = x.narrow(0, 1, 2)?;
    y.mul(&y)?.sum(None)
}

#[test]
fn hessian_narrow_quadratic_matches_finite_difference() {
    let x0 = [1.0f32, -2.0, 0.5, 3.0, -1.1];
    let numeric = finite_diff_hessian(build_narrow_quadratic, &x0, &[5], 1e-3);
    let analytic = analytic_hessian(build_narrow_quadratic, &x0, &[5]);
    assert_hessian_close(&analytic, &numeric);
    // narrow の外側（index 0・4）は loss に無関係なので Hessian の
    // 対応する行・列は 0 のまま。
    for j in [0usize, 4] {
        for i in 0..5 {
            assert!(
                analytic[j][i].abs() < 1e-4,
                "narrow 対象外の行 {j} が非ゼロ"
            );
            assert!(
                analytic[i][j].abs() < 1e-4,
                "narrow 対象外の列 {j} が非ゼロ"
            );
        }
    }
}

fn build_concat_quadratic<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let a = x.narrow(0, 0, 2)?;
    let b = x.narrow(0, 2, 2)?;
    // `a` を 2 回連結する（同一 NodeId の重複寄与を `accumulate` が
    // 合算する経路も検証する）。
    let y = Var::cat(&[a, b, a], 0)?;
    y.mul(&y)?.sum(None)
}

#[test]
fn hessian_concat_quadratic_matches_finite_difference() {
    let x0 = [1.0f32, -2.0, 0.5, 3.0];
    let numeric = finite_diff_hessian(build_concat_quadratic, &x0, &[4], 1e-3);
    let analytic = analytic_hessian(build_concat_quadratic, &x0, &[4]);
    assert_hessian_close(&analytic, &numeric);
    // `a`（index 0・1）は 2 回（direct + 重複連結）寄与するため
    // 対角成分は `b`（index 2・3。1 回のみ）の 2 倍になる。
    assert!(
        common::req2_close(analytic[0][0], 2.0 * analytic[2][2]),
        "重複連結の寄与が合算されていない: a={} b={}",
        analytic[0][0],
        analytic[2][2]
    );
}

// `Op::Contiguous`（`Var::contiguous`）は `pub(crate)` のため本統合
// テスト（別クレート扱い）からは直接呼べない。単体テスト
// （`crates/autodiff/src/create_graph.rs` 内の `#[cfg(test)] mod
// tests`）で検証する。

// --- Where（`Var::where_cond`。マスクは固定定数——境界依存にしない） ---

fn build_where<'t>(tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let cond = Tensor::new(vec![true, false, true], &[3]).expect("mask fixture");
    let y2 = x.mul(x)?;
    let y3 = x.mul(x)?.mul(x)?;
    // cond の真偽は forward 値に依存しない固定マスクのため、`cond`
    // 自体を微分パスに含めない（`Var::where_cond` の `cond` 引数は
    // `Tensor<bool>` で追跡対象外）。
    let _ = tape;
    Var::where_cond(&cond, &y2, &y3)?.sum(None)
}

#[test]
fn hessian_where_matches_finite_difference() {
    let x0 = [1.3f32, -0.7, 2.1];
    let numeric = finite_diff_hessian(build_where, &x0, &[3], 1e-3);
    let analytic = analytic_hessian(build_where, &x0, &[3]);
    assert_hessian_close(&analytic, &numeric);
    // index 0・2（cond == true）は y2 = x^2 の Hessian（対角 2）、
    // index 1（cond == false）は y3 = x^3 の Hessian（対角 6x）。
    assert!(common::req2_close(analytic[0][0], 2.0));
    assert!(common::req2_close(analytic[2][2], 2.0));
    assert!(common::req2_close(analytic[1][1], 6.0 * x0[1] as f64));
}

// --- MaskedFill・Gather・Scatter・Pad・MseLoss・CrossEntropyLoss は
//     イシュー #2062 のスコープ外（`Op::supports_create_graph()` は
//     引き続き `false`）。fail-closed のまま残ることを固定する。

#[test]
fn create_graph_rejects_unsupported_op_masked_fill() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 3.0], &[3]));
    let mask = Tensor::new(vec![true, false, true], &[3]).expect("mask fixture");
    let loss = x.masked_fill(&mask, 0.0).unwrap().sum(None).unwrap();

    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
    assert!(child.is_empty());
}

#[test]
fn create_graph_rejects_unsupported_op_gather() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let index = fandhe_ai_tensor_core::Tensor::<i32>::new(vec![0, 0, 1], &[3]).expect("index");
    let loss = x.gather(0, &index).unwrap().sum(None).unwrap();

    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
    assert!(child.is_empty());
}
