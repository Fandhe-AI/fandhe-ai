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
//! - `BatchNorm`（train／eval。#3079）の二階微分を有限差分・独立合成・3 段の
//!   overflow 回帰ベクトルで確認する（running stats が二重更新されないことも含む）。
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

// `Gelu`・`GeluTanh`・`Erf` はイシュー #3081（決定記録 §21.5・§22）で
// 対象になったため、未対応 Op の代表は `Softmax`・`Custom`（恒久対象外）
// と、fail-closed のまま残る隣接 Op の `Selu`（#2649 系）で固定する。
// 入口検査（`validate_ancestors`）が `build_mirror`／`build_cgrads` より
// 前に判定するため、拒否時に `child` が一切書き込まれない（空のまま）
// ことも併せて固定する。
#[test]
fn create_graph_rejects_unsupported_scalar_unary_selu() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let loss = x.selu().unwrap().sum(None).unwrap();

    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
    assert!(child.is_empty());
}

// 新たに対象になった Gelu を含むグラフでも、恒久対象外の `Custom` に
// 到達すれば fail-closed で拒否され、child は空のまま保たれる。
#[test]
fn create_graph_rejects_custom_downstream_of_gelu() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let h = x.gelu().unwrap();
    let out = tape
        .custom(Arc::new(IdentityCustomFn), &[h])
        .expect("Tape::custom 登録成功");
    let loss = out.sum(None).unwrap();

    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
    assert!(child.is_empty());
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

/// 底 `a`（行 0）・指数 `b`（行 1）の両方を追跡対象にした `a^b`
/// （`x` shape `[2, 2]`。`Var::narrow` で行ごとに分離）。`a` に `0.0`
/// を含めることで `create_graph.rs::build_cgrads` の
/// `ScalarBinaryOp::Pow` db 枝（`a_bc.log()`。`a==0` で forward 値が
/// `-inf`）を通す。`b` は `2.0`／`3.0`（0 でない正整数）のため
/// `a^b` 自体は `a==0` を含め全域で滑らか（多項式）——有限差分と
/// 突合可能な形状を保ちつつ、Hessian（child tape の再 backward）が
/// `where_cond` の不選択枝〈`log(0)=-inf`〉を通っても `NaN` を出さない
/// ことを検査する（review 指摘 2026-09-22。修正前は `0 * (1/a) =
/// 0 * inf = NaN` が db 成分へ混入していた）。
fn build_pow_zero_base<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let a = x.narrow(0, 0, 1)?;
    let b = x.narrow(0, 1, 1)?;
    a.pow(&b)?.sum(None)
}

#[test]
fn hessian_pow_db_zero_base_matches_finite_difference() {
    // 行 0 = a = [0.0, 0.0]、行 1 = b = [2.0, 3.0]（flat: [a0, a1, b0, b1]）。
    let x0 = [0.0f32, 0.0, 2.0, 3.0];
    let numeric = finite_diff_hessian(build_pow_zero_base, &x0, &[2, 2], 1e-3);
    let analytic = analytic_hessian(build_pow_zero_base, &x0, &[2, 2]);

    // 主目的の回帰検査: 修正前は db 枝の `where_cond` 逆伝播が
    // `0 * (1/a) = 0 * inf = NaN` を生み、Hessian 全体が NaN に汚染
    // されていた。まず全要素が有限であることを検査する。
    for row in &analytic {
        for &v in row {
            assert!(
                v.is_finite(),
                "Hessian に NaN／inf が混入（a==0 の log 安全化漏れ疑い）: {row:?}"
            );
        }
    }

    // 混合偏微分 d²(a^b)/da db（(j,i) = (2,0)・(3,1)。それぞれ a0-b0・
    // a1-b1 の組）は中央差分だと `a` を負側へまたぐ必要があり、
    // `db = y * ln(a)`（`eval::scalar::binary_partials`）の参照実装側が
    // `ln(負数) = NaN` を返す（`a^b` 自体は integer 指数のため負の `a`
    // でも実数値を持つが、`ln(a)` を経由する db の勾配公式は `a > 0`
    // でのみ定義される数学的な境界制約であり、本 PR が是正した
    // create_graph 側のバグとは無関係）。実測でも修正後の analytic は
    // 有限（0）だが中央差分の numeric[2][0]／numeric[3][1] は NaN に
    // なることを確認済みのため、この 2 要素のみ「解析値が 0 である
    // こと」を直接検査し、他の要素は通常どおり中央差分と突合する。
    let domain_boundary_pairs = [(2usize, 0usize), (3usize, 1usize)];
    for (j, row_a) in analytic.iter().enumerate() {
        for (i, &a) in row_a.iter().enumerate() {
            if domain_boundary_pairs.contains(&(j, i)) {
                assert_eq!(
                    a, 0.0,
                    "db 境界点（a==0）の解析 Hessian[{j}][{i}] は 0 のはず: {a}"
                );
                continue;
            }
            let n = numeric[j][i];
            assert!(
                common::req2_close(a, n),
                "hessian mismatch at [{j}][{i}]: analytic={a} numeric={n}"
            );
        }
    }
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
// ===========================================================================
// 2D 畳み込み族（`Conv2d`・`ConvTranspose2d`・`Unfold`・`Fold`）の
// create_graph 対応（イシュー #3080・決定記録 §23）。
//
// - Hessian: 1 本の葉 `p` から `narrow`＋`reshape` で x・w・(b) を切り出し、
//   純ブロックと混合ブロックをまとめて有限差分と突合する。
// - 子テープ上の 1 階勾配（`cg.grad`）を `Tape::backward` と突合する。
// - V3（d_weight／d_bias）の前方値が 1 階と bit 一致することを、
//   f32 逐次和では overflow するが f64 縮約では有限になる入力で固定する。
// - 孫テープ（3 段目）・fail-closed（Conv3d）。
// ===========================================================================

#[derive(Clone, Copy, PartialEq)]
enum ConvKind {
    Conv2d,
    ConvTranspose2d,
}

#[derive(Clone, Copy)]
struct ConvCfg {
    kind: ConvKind,
    n: usize,
    cin: usize,
    cout: usize,
    hw: [usize; 2],
    k: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    output_padding: [usize; 2],
    groups: usize,
    has_bias: bool,
}

impl ConvCfg {
    fn new(kind: ConvKind, groups: usize, has_bias: bool) -> Self {
        Self {
            kind,
            n: 2,
            cin: 2,
            cout: 2,
            hw: [3, 3],
            k: [2, 2],
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            output_padding: [0, 0],
            groups,
            has_bias,
        }
    }

    fn x_shape(&self) -> Vec<usize> {
        vec![self.n, self.cin, self.hw[0], self.hw[1]]
    }

    fn w_shape(&self) -> Vec<usize> {
        match self.kind {
            ConvKind::Conv2d => vec![self.cout, self.cin / self.groups, self.k[0], self.k[1]],
            ConvKind::ConvTranspose2d => {
                vec![self.cin, self.cout / self.groups, self.k[0], self.k[1]]
            }
        }
    }

    fn sizes(&self) -> (usize, usize, usize) {
        (
            flat_len(&self.x_shape()),
            flat_len(&self.w_shape()),
            if self.has_bias { self.cout } else { 0 },
        )
    }

    fn p_len(&self) -> usize {
        let (a, b, c) = self.sizes();
        a + b + c
    }

    /// 決定的な擬似値（[-0.8, 0.8] 程度）。
    fn p0(&self) -> Vec<f32> {
        (0..self.p_len())
            .map(|i| (((i * 37 + 11) % 17) as f32 - 8.0) * 0.1)
            .collect()
    }

    fn forward<'t>(
        &self,
        x: &Var<'t>,
        w: &Var<'t>,
        b: Option<&Var<'t>>,
    ) -> Result<Var<'t>, AutodiffError> {
        match self.kind {
            ConvKind::Conv2d => {
                x.conv2d(w, b, self.stride, self.padding, self.dilation, self.groups)
            }
            ConvKind::ConvTranspose2d => x.conv_transpose2d(
                w,
                b,
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation,
                self.groups,
            ),
        }
    }

    /// loss = sum(y ⊙ y ⊙ c)（多項式。純ブロックが非ゼロ。勾配が x・w・b の 3 次以下の
    /// 多項式のため、中央差分の h² 項が Richardson 外挿で厳密に消え、h を大きく取れる）。
    fn build<'t>(&self, tape: &'t Tape, p: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let (nx, nw, nb) = self.sizes();
        let x = p.narrow(0, 0, nx)?.reshape(&self.x_shape())?;
        let w = p.narrow(0, nx, nw)?.reshape(&self.w_shape())?;
        let b = if self.has_bias {
            Some(p.narrow(0, nx + nw, nb)?)
        } else {
            None
        };
        let y = self.forward(&x, &w, b.as_ref())?;
        let ys = y.to_tensor().shape().to_vec();
        let coef: Vec<f32> = (0..flat_len(&ys))
            .map(|i| 0.5 + ((i * 13 + 5) % 11) as f32 * 0.1)
            .collect();
        let c = tape.var_no_grad(&t(coef, &ys));
        y.mul(&y)?.mul(&c)?.sum(None)
    }
}

fn check_conv_cfg(cfg: ConvCfg) {
    let p0 = cfg.p0();
    let shape = [p0.len()];
    // f32 勾配の丸めノイズ（`ulp / 2h`）が小さい Hessian 成分で REQ-2 の絶対許容
    // （1e-5）を超えるため、h を大きく取り Richardson 外挿（`(4·D(h) − D(2h)) / 3`）で
    // 補う（多項式 loss では打ち切り誤差が厳密に消える）。tolerance 自体は変更しない。
    let d1 = finite_diff_hessian(|tp, p| cfg.build(tp, p), &p0, &shape, 1e-1);
    let d2 = finite_diff_hessian(|tp, p| cfg.build(tp, p), &p0, &shape, 2e-1);
    let numeric: Vec<Vec<f64>> = d1
        .iter()
        .zip(&d2)
        .map(|(r1, r2)| {
            r1.iter()
                .zip(r2)
                .map(|(a, b)| (4.0 * a - b) / 3.0)
                .collect()
        })
        .collect();
    let analytic = analytic_hessian(|tp, p| cfg.build(tp, p), &p0, &shape);
    assert_hessian_close(&analytic, &numeric);

    // 子テープ上の 1 階勾配（cg.grad）と Tape::backward の一致、
    // first_order() が素の backward と bit 同一であること。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let p = tape.var(&t(p0.clone(), &shape));
    let loss = cfg.build(&tape, &p).unwrap();
    let plain = tape.backward(&loss).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gp_plain = plain.get(&p).unwrap().unwrap();
    let gp_first = cg.first_order().get(&p).unwrap().unwrap();
    let gp_child = cg.grad(&p).unwrap().unwrap();
    for i in 0..p0.len() {
        assert_eq!(
            gp_plain.get(&[i]).unwrap().to_bits(),
            gp_first.get(&[i]).unwrap().to_bits(),
            "first_order は素の backward と bit 同一のはず"
        );
        let expected = gp_plain.get(&[i]).unwrap() as f64;
        let actual = gp_child.value().get(&[i]).unwrap() as f64;
        assert!(
            common::req2_close(actual, expected),
            "cg.grad[{i}]={actual} backward={expected}"
        );
    }
}

#[test]
fn hessian_conv2d_matches_finite_difference_bias_groups_matrix() {
    for groups in [1usize, 2] {
        for has_bias in [false, true] {
            check_conv_cfg(ConvCfg::new(ConvKind::Conv2d, groups, has_bias));
        }
    }
}

#[test]
fn hessian_conv2d_stride_padding_and_dilation() {
    let mut a = ConvCfg::new(ConvKind::Conv2d, 1, true);
    a.n = 1;
    a.hw = [4, 4];
    a.stride = [2, 2];
    a.padding = [1, 1];
    check_conv_cfg(a);

    let mut b = ConvCfg::new(ConvKind::Conv2d, 1, true);
    b.n = 1;
    b.hw = [4, 4];
    b.dilation = [2, 2];
    check_conv_cfg(b);
}

#[test]
fn hessian_conv_transpose2d_matches_finite_difference_bias_groups_matrix() {
    for groups in [1usize, 2] {
        for has_bias in [false, true] {
            check_conv_cfg(ConvCfg::new(ConvKind::ConvTranspose2d, groups, has_bias));
        }
    }
}

#[test]
fn hessian_conv_transpose2d_stride_output_padding_and_dilation() {
    // stride=2・output_padding=1: replay の output_padding 導出（out − base）を検証する。
    let mut a = ConvCfg::new(ConvKind::ConvTranspose2d, 1, true);
    a.n = 1;
    a.hw = [2, 2];
    a.stride = [2, 2];
    a.output_padding = [1, 1];
    check_conv_cfg(a);

    let mut b = ConvCfg::new(ConvKind::ConvTranspose2d, 1, true);
    b.n = 1;
    b.hw = [2, 2];
    b.dilation = [2, 2];
    b.padding = [1, 1];
    check_conv_cfg(b);
}

// --- Unfold／Fold 単体（窓が重なる設定） --------------------------------------

fn build_unfold_sq<'t>(tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let xr = x.reshape(&[1, 2, 3, 3])?;
    let u = xr.unfold([2, 2], [1, 1], [0, 0], [1, 1])?;
    let us = u.to_tensor().shape().to_vec();
    let coef: Vec<f32> = (0..flat_len(&us))
        .map(|i| 0.3 + (i % 7) as f32 * 0.1)
        .collect();
    let c = tape.var_no_grad(&t(coef, &us));
    u.mul(&u)?.mul(&c)?.sum(None)
}

#[test]
fn hessian_unfold_overlapping_windows_matches_finite_difference() {
    let x0: Vec<f32> = (0..18)
        .map(|i| ((i * 7 + 3) % 11) as f32 * 0.1 - 0.5)
        .collect();
    let numeric = finite_diff_hessian(build_unfold_sq, &x0, &[18], 1e-3);
    let analytic = analytic_hessian(build_unfold_sq, &x0, &[18]);
    assert_hessian_close(&analytic, &numeric);
}

fn build_fold_sq<'t>(tape: &'t Tape, z: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    // [1, 2*2*2, 4] -> fold([3, 3], k=2, s=1) -> [1, 2, 3, 3]（窓が重なる）。
    let zr = z.reshape(&[1, 8, 4])?;
    let f = zr.fold([3, 3], [2, 2], [1, 1], [0, 0], [1, 1])?;
    let fs = f.to_tensor().shape().to_vec();
    let coef: Vec<f32> = (0..flat_len(&fs))
        .map(|i| 0.4 + (i % 5) as f32 * 0.1)
        .collect();
    let c = tape.var_no_grad(&t(coef, &fs));
    f.mul(&f)?.mul(&c)?.sum(None)
}

#[test]
fn hessian_fold_overlapping_windows_matches_finite_difference() {
    let z0: Vec<f32> = (0..32)
        .map(|i| ((i * 5 + 1) % 13) as f32 * 0.1 - 0.6)
        .collect();
    let numeric = finite_diff_hessian(build_fold_sq, &z0, &[32], 1e-3);
    let analytic = analytic_hessian(build_fold_sq, &z0, &[32]);
    assert_hessian_close(&analytic, &numeric);
}

// --- V3: overflow 回帰（f32 逐次和では inf・f64 縮約では有限） -----------------

const BIG: f32 = 3.0e38;

/// N=3・1×1 カーネル・H=W=1。`x=[1,1,-1]`・`g=c=[BIG;3]`（loss = sum(y ⊙ c)）で
/// d_weight = Σ x·g = BIG（f64 縮約で有限。f32 逐次和なら 2·BIG で inf）。
/// `cg.grad(w)` の前方値が 1 階と bit 一致し、二階（∂gw/∂x = c）も有限で一致すること。
fn conv_overflow_case(kind: ConvKind) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 1.0, -1.0], &[3, 1, 1, 1]));
    let w = tape.var(&t(vec![1.0], &[1, 1, 1, 1]));
    let y = match kind {
        ConvKind::Conv2d => x.conv2d(&w, None, [1, 1], [0, 0], [1, 1], 1).unwrap(),
        ConvKind::ConvTranspose2d => x
            .conv_transpose2d(&w, None, [1, 1], [0, 0], [0, 0], [1, 1], 1)
            .unwrap(),
    };
    let c = tape.var_no_grad(&t(vec![BIG; 3], &[3, 1, 1, 1]));
    let loss = y.mul(&c).unwrap().sum(None).unwrap();

    let plain = tape.backward(&loss).unwrap();
    let dw_plain = plain.get(&w).unwrap().unwrap().get(&[0, 0, 0, 0]).unwrap();
    assert!(dw_plain.is_finite(), "1 階の d_weight は f64 縮約で有限");

    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gw = cg.grad(&w).unwrap().unwrap();
    let dw_child = gw.value().get(&[0, 0, 0, 0]).unwrap();
    assert_eq!(
        dw_plain.to_bits(),
        dw_child.to_bits(),
        "V3: cg.grad(w) の前方値は 1 階と bit 一致"
    );

    // 二階: ∂(gw)/∂x_i = g_i = BIG（有限）。
    let cx = cg.child_var(&x).unwrap().unwrap();
    let grads2 = child.backward(&gw).unwrap();
    let dx2 = grads2.get(&cx).unwrap().unwrap();
    for i in 0..3 {
        let v = dx2.get(&[i, 0, 0, 0]).unwrap();
        assert!(
            v.is_finite() && (v - BIG).abs() <= BIG * 1e-6,
            "dx2[{i}]={v}"
        );
    }

    // 3 段目（孫テープ）: gw を更に create_graph で微分できる（閉包性）。
    // 孫テープでの再生値も 1 階と同じホスト関数（f64 縮約）で再計算されるため
    // overflow しない（`Op::ConvWeightGradHost`。決定記録 §23.3）。
    let grand = Tape::new_with_ops(common::naive_ops());
    let cg2 = child
        .backward_create_graph(&gw, &grand)
        .expect("子テープ上の勾配は更に create_graph で微分できる（閉包性）");
    let g_cx = cg2.grad(&cx).unwrap().unwrap();
    for i in 0..3 {
        let v = g_cx.value().get(&[i, 0, 0, 0]).unwrap();
        assert!(v.is_finite(), "3 段目の勾配 [{i}] は有限のはず: {v}");
    }

    // 孫テープ上で gw の非線形関数 h = (gw·1e-38)² を作って更に微分する。
    // 通常の backward で有限な x 勾配は孫テープ側でも有限・一致すること
    // （補正値 gw=3e38 が孫テープ再生で inf に化けないこと）。
    let child2 = Tape::new_with_ops(common::naive_ops());
    let cg_b = tape.backward_create_graph(&loss, &child2).unwrap();
    let gw_b = cg_b.grad(&w).unwrap().unwrap();
    let s = child2.var_no_grad(&t(vec![1.0e-38], &[1, 1, 1, 1]));
    let scaled = gw_b.mul(&s).unwrap();
    let h = scaled.mul(&scaled).unwrap().sum(None).unwrap();
    let cx_b = cg_b.child_var(&x).unwrap().unwrap();
    let plain_h = child2.backward(&h).unwrap();
    let dx_plain = plain_h.get(&cx_b).unwrap().unwrap();
    let grand2 = Tape::new_with_ops(common::naive_ops());
    let cg3 = child2.backward_create_graph(&h, &grand2).unwrap();
    let dx_grand = cg3.grad(&cx_b).unwrap().unwrap();
    for i in 0..3 {
        let p = dx_plain.get(&[i, 0, 0, 0]).unwrap();
        let q = dx_grand.value().get(&[i, 0, 0, 0]).unwrap();
        assert!(p.is_finite() && q.is_finite(), "[{i}] plain={p} grand={q}");
        assert_eq!(p.to_bits(), q.to_bits(), "[{i}] 孫テープ再生値の維持");
    }
}

#[test]
fn conv2d_weight_grad_overflow_matches_first_order_bitwise() {
    conv_overflow_case(ConvKind::Conv2d);
}

#[test]
fn conv_transpose2d_weight_grad_overflow_matches_first_order_bitwise() {
    conv_overflow_case(ConvKind::ConvTranspose2d);
}

fn conv_bias_overflow_case(kind: ConvKind) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var_no_grad(&t(vec![0.5, 0.5, 0.5], &[3, 1, 1, 1]));
    let w = tape.var_no_grad(&t(vec![1.0], &[1, 1, 1, 1]));
    let b = tape.var(&t(vec![0.0], &[1]));
    let y = match kind {
        ConvKind::Conv2d => x.conv2d(&w, Some(&b), [1, 1], [0, 0], [1, 1], 1).unwrap(),
        ConvKind::ConvTranspose2d => x
            .conv_transpose2d(&w, Some(&b), [1, 1], [0, 0], [0, 0], [1, 1], 1)
            .unwrap(),
    };
    let c = tape.var_no_grad(&t(vec![BIG, BIG, -BIG], &[3, 1, 1, 1]));
    let loss = y.mul(&c).unwrap().sum(None).unwrap();
    let plain = tape.backward(&loss).unwrap();
    let db_plain = plain.get(&b).unwrap().unwrap().get(&[0]).unwrap();
    assert!(db_plain.is_finite());
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let db_child = cg.grad(&b).unwrap().unwrap().value().get(&[0]).unwrap();
    assert_eq!(db_plain.to_bits(), db_child.to_bits());
}

#[test]
fn conv2d_bias_grad_overflow_matches_first_order_bitwise() {
    conv_bias_overflow_case(ConvKind::Conv2d);
}

#[test]
fn conv_transpose2d_bias_grad_overflow_matches_first_order_bitwise() {
    conv_bias_overflow_case(ConvKind::ConvTranspose2d);
}

// --- 境界: 負の base の転置畳み込み・ゼロ次元チャネルの bias 勾配 ---------------

/// `in=1, k=1, stride=3, padding=1, dilation=1, output_padding=2`（各軸）:
/// `(in−1)·s + d(k−1) + 1 − 2p = −1` で base が負になるが forward は有効。
/// create_graph が拒否せず、1 階と一致すること（Bugbot・codex P2）。
#[test]
fn conv_transpose2d_negative_base_output_padding_is_replayable() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.5], &[1, 1, 1, 1]));
    let w = tape.var(&t(vec![2.0], &[1, 1, 1, 1]));
    let y = x
        .conv_transpose2d(&w, None, [3, 3], [1, 1], [2, 2], [1, 1], 1)
        .unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let plain = tape.backward(&loss).unwrap();
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("負の base でも有効な転置畳み込みは再生できる");
    let dx_plain = plain.get(&x).unwrap().unwrap().get(&[0, 0, 0, 0]).unwrap();
    let dx_child = cg
        .grad(&x)
        .unwrap()
        .unwrap()
        .value()
        .get(&[0, 0, 0, 0])
        .unwrap();
    assert_eq!(dx_plain.to_bits(), dx_child.to_bits());
    // 孫テープ側（再生）でも `output_padding` が導出できること。
    let grand = Tape::new_with_ops(common::naive_ops());
    let gx = cg.grad(&x).unwrap().unwrap();
    child
        .backward_create_graph(&gx.sum(None).unwrap(), &grand)
        .expect("子テープ上の転置畳み込みも再生できる");
}

/// 1×1 入力・3×3 カーネル・stride 3・padding 2・output_padding 2（Bugbot の例。
/// `(in−1)s + d(k−1) + 1 < 2p`）。
#[test]
fn conv_transpose2d_small_input_large_padding_is_replayable() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.5], &[1, 1, 1, 1]));
    let w = tape.var(&t((0..9).map(|i| i as f32 * 0.1).collect(), &[1, 1, 3, 3]));
    let y = x
        .conv_transpose2d(&w, None, [3, 3], [2, 2], [2, 2], [1, 1], 1)
        .unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gw = cg.grad(&w).unwrap().unwrap();
    let grand = Tape::new_with_ops(common::naive_ops());
    child
        .backward_create_graph(&gw.sum(None).unwrap(), &grand)
        .expect("再生で output_padding を導出できる");
}

/// 入力チャネル 0 でも bias 勾配は出力が非空なら通常の backward と一致する。
#[test]
fn conv_bias_grad_with_zero_in_channels_matches_first_order() {
    for kind in [ConvKind::Conv2d, ConvKind::ConvTranspose2d] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&Tensor::<f32>::zeros(&[1, 0, 1, 1]).unwrap());
        let (w, y);
        let b = tape.var(&t(vec![1.0], &[1]));
        match kind {
            ConvKind::Conv2d => {
                w = tape.var(&Tensor::<f32>::zeros(&[1, 0, 1, 1]).unwrap());
                y = x.conv2d(&w, Some(&b), [1, 1], [0, 0], [1, 1], 1).unwrap();
            }
            ConvKind::ConvTranspose2d => {
                w = tape.var(&Tensor::<f32>::zeros(&[0, 1, 1, 1]).unwrap());
                y = x
                    .conv_transpose2d(&w, Some(&b), [1, 1], [0, 0], [0, 0], [1, 1], 1)
                    .unwrap();
            }
        }
        let loss = y.sum(None).unwrap();
        let plain = tape.backward(&loss).unwrap();
        let db_plain = plain.get(&b).unwrap().unwrap().get(&[0]).unwrap();
        assert_eq!(db_plain, 1.0);
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let db = cg.grad(&b).unwrap().unwrap().value().get(&[0]).unwrap();
        assert_eq!(db.to_bits(), db_plain.to_bits());
    }
}

// --- 孫テープ: jvp_create_graph（親→子→孫） -----------------------------------

fn conv_jvp_case(kind: ConvKind) {
    let cfg = ConvCfg::new(kind, 1, true);
    let p0 = cfg.p0();
    let shape = [p0.len()];
    let tangent: Vec<f32> = (0..p0.len())
        .map(|i| ((i * 3 + 1) % 7) as f32 * 0.1 - 0.3)
        .collect();

    let out_at = |pv: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let p = tape.var(&t(pv.to_vec(), &shape));
        let loss = cfg.build(&tape, &p).unwrap();
        vec![loss.value().get(&[]).unwrap() as f64]
    };
    // 独立な中央差分の方向微分。
    let h = 1e-3f32;
    let plus: Vec<f32> = p0.iter().zip(&tangent).map(|(a, v)| a + h * v).collect();
    let minus: Vec<f32> = p0.iter().zip(&tangent).map(|(a, v)| a - h * v).collect();
    let numeric = (out_at(&plus)[0] - out_at(&minus)[0]) / (2.0 * h as f64);

    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let grand = Tape::new_with_ops(common::naive_ops());
    let p = tape.var(&t(p0.clone(), &shape));
    let loss = cfg.build(&tape, &p).unwrap();
    let res = fandhe_ai_autodiff::functional_ops::jvp_create_graph(
        &tape,
        &loss,
        &p,
        &t(tangent.clone(), &shape),
        &child,
        &grand,
    )
    .expect("Conv 系を含む関数でも jvp_create_graph が成立する（閉包性）");
    let jv = res.output().value().get(&[]).unwrap() as f64;
    assert!(
        common::req2_close(jv, numeric),
        "jvp={jv} numeric={numeric}"
    );
}

#[test]
fn jvp_create_graph_through_conv2d_matches_directional_derivative() {
    conv_jvp_case(ConvKind::Conv2d);
}

#[test]
fn jvp_create_graph_through_conv_transpose2d_matches_directional_derivative() {
    conv_jvp_case(ConvKind::ConvTranspose2d);
}

// --- fail-closed: 3D 族は #3080(b) まで拒否のまま ------------------------------

#[test]
fn create_graph_still_rejects_conv3d() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.5; 8], &[1, 1, 2, 2, 2]));
    let w = tape.var(&t(vec![0.25; 8], &[1, 1, 2, 2, 2]));
    let y = x
        .conv3d(&w, None, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1)
        .expect("conv3d forward");
    let loss = y.sum(None).unwrap();
    let err = tape
        .backward_create_graph(&loss, &child)
        .expect_err("Conv3d は fail-closed で拒否されるはず");
    assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
    assert!(child.is_empty());
}

// --- Gelu・GeluTanh・Erf（イシュー #3081。決定記録 §22） --------------------

fn build_gelu<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.gelu()?.sum(None)
}

fn build_gelu_tanh<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.gelu_tanh()?.sum(None)
}

fn build_erf<'t>(_tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.erf()?.sum(None)
}

// 有限差分は f32 の 1 階勾配を差分するためノイズが約 1e-4 あり、|x| が
// 大きいと 2 階導関数が REQ-2 の絶対 1e-5 を満たせない。突合点は中程度の
// |x| に限り、tolerance は変えない（§22）。
const MID_POINTS: [f32; 5] = [-1.3, -0.4, 0.0, 0.7, 1.2];

#[test]
fn hessian_gelu_matches_finite_difference() {
    let numeric = finite_diff_hessian(build_gelu, &MID_POINTS, &[5], 1e-3);
    let analytic = analytic_hessian(build_gelu, &MID_POINTS, &[5]);
    assert_hessian_close(&analytic, &numeric);
}

#[test]
fn hessian_gelu_tanh_matches_finite_difference() {
    let numeric = finite_diff_hessian(build_gelu_tanh, &MID_POINTS, &[5], 1e-3);
    let analytic = analytic_hessian(build_gelu_tanh, &MID_POINTS, &[5]);
    assert_hessian_close(&analytic, &numeric);
}

#[test]
fn hessian_erf_matches_finite_difference_and_closed_form() {
    let numeric = finite_diff_hessian(build_erf, &MID_POINTS, &[5], 1e-3);
    let analytic = analytic_hessian(build_erf, &MID_POINTS, &[5]);
    assert_hessian_close(&analytic, &numeric);
    for (i, &x) in MID_POINTS.iter().enumerate() {
        let x = x as f64;
        let expected = -2.0 * x * (2.0 / std::f64::consts::PI.sqrt()) * (-x * x).exp();
        assert!(
            common::req2_close(analytic[i][i], expected),
            "erf diag[{i}]: analytic={} expected={expected}",
            analytic[i][i]
        );
    }
}

#[test]
fn hessian_gelu_large_magnitude_matches_closed_form() {
    let x0 = [-4.0f32, -3.0, 3.0, 4.0, 5.0];
    let analytic = analytic_hessian(build_gelu, &x0, &[5]);
    for (i, &x) in x0.iter().enumerate() {
        let x = x as f64;
        let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let expected = pdf * (2.0 - x * x);
        assert!(
            common::req2_close(analytic[i][i], expected),
            "gelu diag[{i}]: analytic={} expected={expected}",
            analytic[i][i]
        );
    }
}

#[test]
fn hessian_gelu_tanh_large_magnitude_matches_closed_form() {
    const C: f64 = 0.797_884_560_802_865_4;
    const K: f64 = 0.044_715;
    let x0 = [-4.0f32, -3.0, 3.0, 4.0, 5.0];
    let analytic = analytic_hessian(build_gelu_tanh, &x0, &[5]);
    for (i, &x) in x0.iter().enumerate() {
        let x = x as f64;
        let t = (C * (x + K * x * x * x)).tanh();
        let s = 1.0 - t * t;
        let du = C * (1.0 + 3.0 * K * x * x);
        let expected = s * du + 0.5 * x * (-2.0 * t * s * du * du + s * C * 6.0 * K * x);
        // f32 の ULP 由来の誤差を許す（REQ-2 判定式そのまま）。
        assert!(
            common::req2_close(analytic[i][i], expected),
            "gelu_tanh diag[{i}]: analytic={} expected={expected}",
            analytic[i][i]
        );
    }
}

// 巨大入力でも NaN を生まない（`gelu_tanh_grad_huge_magnitude_input_does_
// not_produce_nan` と同じ観点を 2 階・孫テープまで拡張）。
#[test]
fn create_graph_gelu_family_huge_magnitude_is_nan_free() {
    type B = for<'t> fn(&'t Tape, &Var<'t>) -> Result<Var<'t>, AutodiffError>;
    let builders: [(&str, B); 3] = [
        ("gelu", build_gelu),
        ("gelu_tanh", build_gelu_tanh),
        ("erf", build_erf),
    ];
    for (name, build) in builders {
        for &v in &[1e20f32, -1e20, 1e30, -1e30, 3e38, -3e38] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(vec![v], &[1]));
            let loss = build(&tape, &x).unwrap();
            let cg = tape.backward_create_graph(&loss, &child).unwrap();
            let gx = cg.grad(&x).unwrap().unwrap();
            let g = gx.value().get(&[0]).unwrap() as f64;
            assert!(g.is_finite(), "{name}({v}): 1 階勾配が非有限 {g}");
            let fo = cg.first_order().get(&x).unwrap().unwrap();
            assert!(
                common::req2_close(g, fo.get(&[0]).unwrap() as f64),
                "{name}({v}): cg.grad と first_order が不一致"
            );
            let cx = cg.child_var(&x).unwrap().unwrap();
            let s = extract_scalar(&gx, &[1], 0).unwrap();
            let grads = child.backward(&s).unwrap();
            let h = grads.get(&cx).unwrap().unwrap().get(&[0]).unwrap();
            assert!(h.is_finite() && h.abs() < 1e-6, "{name}({v}): Hessian={h}");
        }
    }
}

// 1 階 `first_order` は素の backward と bit 同一（Gelu 系の経路不変）。
#[test]
fn create_graph_gelu_family_first_order_is_bit_identical() {
    type B = for<'t> fn(&'t Tape, &Var<'t>) -> Result<Var<'t>, AutodiffError>;
    let builders: [B; 3] = [build_gelu, build_gelu_tanh, build_erf];
    for build in builders {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(MID_POINTS.to_vec(), &[5]));
        let loss = build(&tape, &x).unwrap();
        let before = tape.backward(&loss).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let b = before.get(&x).unwrap().unwrap();
        let f = cg.first_order().get(&x).unwrap().unwrap();
        let gx = cg.grad(&x).unwrap().unwrap();
        for i in 0..5 {
            assert_eq!(
                b.get(&[i]).unwrap().to_bits(),
                f.get(&[i]).unwrap().to_bits()
            );
            assert!(common::req2_close(
                gx.value().get(&[i]).unwrap() as f64,
                b.get(&[i]).unwrap() as f64
            ));
        }
    }
}
// --- BatchNorm（イシュー #3079。決定記録 §24）-------------------------------
//
// train（バッチ統計）・eval（固定統計）の両モードで `Op::BatchNorm` を
// create_graph の対象にする。train の `sum(bn(x))` は x に対する勾配が恒等的に
// 0 で Hessian テストにならないため、損失は一様でない係数付きの曲率を持つ
// 形（`sum(y*y*c) + sum(tanh(y)*c2)`）にする。

use fandhe_ai_autodiff::functional_ops::jvp_create_graph;
use fandhe_ai_autodiff::nn::{BatchNorm1d, BatchNorm2d, Module};

/// 決定的で散らばった入力（`[3,2]`・`[2,2,2,2]` のどちらでも分散が小さすぎない）。
fn bn_input(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.37 + 0.05 * i as f32)
        .collect()
}

fn bn_coeffs(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 + ((i * 7 + salt) % 5) as f32 * 0.3)
        .collect()
}

const BN_EPS: f32 = 1e-3;

#[derive(Clone, Copy)]
struct BnCfg {
    shape: &'static [usize],
    train: bool,
    affine: bool,
}

fn bn_fixed_stats(c: usize) -> (Tensor<f32>, Tensor<f32>) {
    let mean: Vec<f32> = (0..c).map(|i| 0.2 - 0.3 * i as f32).collect();
    let var: Vec<f32> = (0..c).map(|i| 0.8 + 0.5 * i as f32).collect();
    (t(mean, &[c]), t(var, &[c]))
}

fn bn_w(c: usize) -> Tensor<f32> {
    t((0..c).map(|i| 1.3 - 0.4 * i as f32).collect(), &[c])
}

fn bn_b(c: usize) -> Tensor<f32> {
    t((0..c).map(|i| 0.2 + 0.1 * i as f32).collect(), &[c])
}

/// `y` から曲率を持つスカラー損失を作る。
fn bn_curvature_loss<'t>(
    tape: &'t Tape,
    y: &Var<'t>,
    shape: &[usize],
) -> Result<Var<'t>, AutodiffError> {
    let n = flat_len(shape);
    let c1 = tape.var_no_grad(&t(bn_coeffs(n, 0), shape));
    let c2 = tape.var_no_grad(&t(bn_coeffs(n, 3), shape));
    let a = y.mul(y)?.mul(&c1)?.sum(None)?;
    let b = y.tanh().mul(&c2)?.sum(None)?;
    a.add(&b)
}

/// `cfg` の BatchNorm を `x`（と任意の `w`／`b`）へ適用する。
fn bn_apply<'t>(
    cfg: BnCfg,
    x: &Var<'t>,
    w: Option<&Var<'t>>,
    b: Option<&Var<'t>>,
) -> Result<Var<'t>, AutodiffError> {
    if cfg.train {
        x.batch_norm(w, b, BN_EPS)
    } else {
        let (mean, var) = bn_fixed_stats(cfg.shape[1]);
        x.batch_norm_infer(w, b, &mean, &var, BN_EPS)
    }
}

fn bn_configs() -> Vec<BnCfg> {
    let mut v = Vec::new();
    for shape in [&[3usize, 2][..], &[2, 2, 2, 2][..]] {
        for train in [true, false] {
            for affine in [true, false] {
                v.push(BnCfg {
                    shape,
                    train,
                    affine,
                });
            }
        }
    }
    v
}

/// 有限差分の刻み。f32 勾配の丸め雑音（~1e-7/h）と打ち切り誤差（~h²）の釣り合いで選ぶ。
const BN_FD_H: f64 = 1e-2;

fn check_hessian<F>(x0: &[f32], shape: &[usize], build: F)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Result<Var<'t>, AutodiffError>,
{
    let numeric = finite_diff_hessian(&build, x0, shape, BN_FD_H);
    let analytic = analytic_hessian(&build, x0, shape);
    assert_hessian_close(&analytic, &numeric);
}

#[test]
fn hessian_batch_norm_x_matches_finite_difference() {
    // rank 4 は f32 勾配の丸め雑音が Hessian 成分（~1e-2）に対し無視できないため
    // 有限差分では rank 2 を、rank 4 は下の独立合成との突合で確認する。
    for cfg in bn_configs().into_iter().filter(|c| c.shape.len() == 2) {
        let c = cfg.shape[1];
        let x0 = bn_input(flat_len(cfg.shape));
        check_hessian(&x0, cfg.shape, |tape, x| {
            let w = tape.var_no_grad(&bn_w(c));
            let b = tape.var_no_grad(&bn_b(c));
            let y = if cfg.affine {
                bn_apply(cfg, x, Some(&w), Some(&b))?
            } else {
                bn_apply(cfg, x, None, None)?
            };
            bn_curvature_loss(tape, &y, cfg.shape)
        });
    }
}

#[test]
fn hessian_batch_norm_weight_and_bias_match_finite_difference() {
    for shape in [&[3usize, 2][..], &[2, 2, 2, 2][..]] {
        for train in [true, false] {
            let cfg = BnCfg {
                shape,
                train,
                affine: true,
            };
            let c = shape[1];
            let xv = bn_input(flat_len(shape));
            let w0 = bn_w(c).as_slice().unwrap().to_vec();
            let b0 = bn_b(c).as_slice().unwrap().to_vec();
            // w を微分変数にする。
            check_hessian(&w0, &[c], |tape, w| {
                let x = tape.var_no_grad(&t(xv.clone(), shape));
                let b = tape.var_no_grad(&bn_b(c));
                let y = bn_apply(cfg, &x, Some(w), Some(&b))?;
                bn_curvature_loss(tape, &y, shape)
            });
            // b を微分変数にする。
            check_hessian(&b0, &[c], |tape, b| {
                let x = tape.var_no_grad(&t(xv.clone(), shape));
                let w = tape.var_no_grad(&bn_w(c));
                let y = bn_apply(cfg, &x, Some(&w), Some(b))?;
                bn_curvature_loss(tape, &y, shape)
            });
        }
    }
}

/// 混合二階微分 `∂(∂L/∂w_j)/∂x_i`: 子テープ由来の値と、`Tape::backward` の
/// `g_w` を x 方向に中央差分した値を突合する。
#[test]
fn hessian_batch_norm_mixed_x_w_matches_finite_difference() {
    for shape in [&[3usize, 2][..], &[2, 2, 2, 2][..]] {
        for train in [true, false] {
            let cfg = BnCfg {
                shape,
                train,
                affine: true,
            };
            let c = shape[1];
            let n = flat_len(shape);
            let x0 = bn_input(n);
            let grad_w_at = |x: &[f32]| -> Vec<f64> {
                let tape = Tape::new_with_ops(common::naive_ops());
                let xv = tape.var(&t(x.to_vec(), shape));
                let w = tape.var(&bn_w(c));
                let b = tape.var_no_grad(&bn_b(c));
                let y = bn_apply(cfg, &xv, Some(&w), Some(&b)).unwrap();
                let loss = bn_curvature_loss(&tape, &y, shape).unwrap();
                let g = tape.backward(&loss).unwrap();
                let gw = g.get(&w).unwrap().unwrap();
                (0..c).map(|j| gw.get(&[j]).unwrap() as f64).collect()
            };
            let h = BN_FD_H;
            let mut numeric = vec![vec![0.0f64; n]; c];
            for i in 0..n {
                let mut xp = x0.clone();
                let mut xm = x0.clone();
                xp[i] += h as f32;
                xm[i] -= h as f32;
                let (gp, gm) = (grad_w_at(&xp), grad_w_at(&xm));
                for j in 0..c {
                    numeric[j][i] = (gp[j] - gm[j]) / (2.0 * h);
                }
            }

            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), shape));
            let w = tape.var(&bn_w(c));
            let b = tape.var_no_grad(&bn_b(c));
            let y = bn_apply(cfg, &x, Some(&w), Some(&b)).unwrap();
            let loss = bn_curvature_loss(&tape, &y, shape).unwrap();
            let cg = tape.backward_create_graph(&loss, &child).unwrap();
            let gw = cg.grad(&w).unwrap().unwrap();
            let cx = cg.child_var(&x).unwrap().unwrap();
            let mut analytic = vec![vec![0.0f64; n]; c];
            for j in 0..c {
                let row = child
                    .backward(&extract_scalar(&gw, &[c], j).unwrap())
                    .unwrap();
                let hx = row.get(&cx).unwrap().unwrap();
                for i in 0..n {
                    analytic[j][i] = hx.get(&unravel(i, shape)).unwrap() as f64;
                }
            }
            assert_hessian_close(&analytic, &numeric);
        }
    }
}

/// 独立参照: BatchNorm（train／eval・rank 2／4・affine の有無）を `mean_dims`・
/// `sub`・`mul`・`sqrt`・`div` の合成（既に create_graph 対象）で親テープに組み、
/// その Hessian と突合する。有限差分の丸め雑音に依存しない確認。
fn bn_composite<'t>(tape: &'t Tape, cfg: BnCfg, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let c = cfg.shape[1];
    let mut bshape = vec![1usize; cfg.shape.len()];
    bshape[1] = c;
    let dims: Vec<usize> = (0..cfg.shape.len()).filter(|&d| d != 1).collect();
    let col = |v: Vec<f32>| tape.var_no_grad(&t(v, &bshape));
    let (mean, rstd) = if cfg.train {
        let mu = x.mean_dims(&dims, true)?;
        let d = x.sub(&mu)?;
        let var = d.mul(&d)?.mean_dims(&dims, true)?;
        let rstd = col(vec![1.0; c]).div(&var.add(&col(vec![BN_EPS; c]))?.sqrt()?)?;
        (mu, rstd)
    } else {
        let (m, v) = bn_fixed_stats(c);
        let rstd = col(vec![1.0; c]).div(
            &col(v.as_slice().unwrap().to_vec())
                .add(&col(vec![BN_EPS; c]))?
                .sqrt()?,
        )?;
        (col(m.as_slice().unwrap().to_vec()), rstd)
    };
    let mut y = x.sub(&mean)?.mul(&rstd)?;
    if cfg.affine {
        y = y
            .mul(&col(bn_w(c).as_slice().unwrap().to_vec()))?
            .add(&col(bn_b(c).as_slice().unwrap().to_vec()))?;
    }
    bn_curvature_loss(tape, &y, cfg.shape)
}

#[test]
fn hessian_batch_norm_matches_manual_composition() {
    for cfg in bn_configs() {
        let c = cfg.shape[1];
        let x0 = bn_input(flat_len(cfg.shape));
        let analytic_bn = analytic_hessian(
            |tape, x| {
                let w = tape.var_no_grad(&bn_w(c));
                let b = tape.var_no_grad(&bn_b(c));
                let y = if cfg.affine {
                    bn_apply(cfg, x, Some(&w), Some(&b))?
                } else {
                    bn_apply(cfg, x, None, None)?
                };
                bn_curvature_loss(tape, &y, cfg.shape)
            },
            &x0,
            cfg.shape,
        );
        let analytic_ref = analytic_hessian(|tape, x| bn_composite(tape, cfg, x), &x0, cfg.shape);
        assert_hessian_close(&analytic_bn, &analytic_ref);
    }
}

/// 子テープ上の 1 階勾配（x・w・b）が `Tape::backward` と REQ-2 で一致し、
/// `first_order()` が素の `backward` と bit 同一であること。
#[test]
fn batch_norm_create_graph_first_order_matches_backward() {
    for cfg in bn_configs() {
        if !cfg.affine {
            continue;
        }
        let c = cfg.shape[1];
        let n = flat_len(cfg.shape);
        let x0 = bn_input(n);
        let run = |with_cg: bool| {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), cfg.shape));
            let w = tape.var(&bn_w(c));
            let b = tape.var(&bn_b(c));
            let y = bn_apply(cfg, &x, Some(&w), Some(&b)).unwrap();
            let loss = bn_curvature_loss(&tape, &y, cfg.shape).unwrap();
            let plain = tape.backward(&loss).unwrap();
            let dense = |g: &fandhe_ai_autodiff::Gradients, v: &Var<'_>| -> Vec<f32> {
                g.get(v)
                    .unwrap()
                    .unwrap()
                    .contiguous()
                    .as_slice()
                    .unwrap()
                    .to_vec()
            };
            let plain_x = dense(&plain, &x);
            if !with_cg {
                return (plain_x, None);
            }
            let cg = tape.backward_create_graph(&loss, &child).unwrap();
            let first_x = dense(cg.first_order(), &x);
            assert_eq!(plain_x, first_x, "first_order は backward と bit 同一");
            for (v, name) in [(&x, "x"), (&w, "w"), (&b, "b")] {
                let expected = dense(&plain, v);
                let got = cg.grad(v).unwrap().unwrap().to_tensor();
                let got = got.contiguous().as_slice().unwrap().to_vec();
                for (a, e) in got.iter().zip(expected.iter()) {
                    assert!(
                        common::req2_close(*a as f64, *e as f64),
                        "cg.grad({name}) が backward と不一致: {a} vs {e}"
                    );
                }
            }
            (plain_x, Some(()))
        };
        let _ = run(true);
    }
}

/// running stats（`nn::BatchNorm*` の buffer・`num_batches_tracked`）が子テープ
/// 構築・二階 backward・孫テープ再生で二重に更新されないこと。
#[test]
fn batch_norm_create_graph_does_not_touch_running_stats() {
    for train in [true, false] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let mut bn = BatchNorm1d::new(2, BN_EPS, 0.1).unwrap();
        bn.set_training(train);
        let x = tape.var(&t(bn_input(6), &[3, 2]));
        let y = bn.bind(&tape).forward(&x).unwrap();
        let loss = bn_curvature_loss(&tape, &y, &[3, 2]).unwrap();
        let (rm, rv, nb) = (
            bn.running_mean(),
            bn.running_var(),
            bn.num_batches_tracked(),
        );
        assert_eq!(nb, u64::from(train));

        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let gx = cg.grad(&x).unwrap().unwrap();
        let cx = cg.child_var(&x).unwrap().unwrap();
        child
            .backward(&extract_scalar(&gx, &[3, 2], 0).unwrap())
            .unwrap()
            .get(&cx)
            .unwrap()
            .unwrap();
        let s2 = extract_scalar(&gx, &[3, 2], 1).unwrap();
        let _ = child.backward_create_graph(&s2, &grand).unwrap();
        assert_eq!(bn.running_mean().as_slice(), rm.as_slice());
        assert_eq!(bn.running_var().as_slice(), rv.as_slice());
        assert_eq!(bn.num_batches_tracked(), nb);

        // jvp_create_graph（親→子→孫の再生）でも同様。
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(bn_input(6), &[3, 2]));
        let y = bn.bind(&tape).forward(&x).unwrap();
        let (rm, rv, nb) = (
            bn.running_mean(),
            bn.running_var(),
            bn.num_batches_tracked(),
        );
        let v = t(bn_coeffs(6, 1), &[3, 2]);
        let _ = jvp_create_graph(&tape, &y, &x, &v, &child, &grand).unwrap();
        assert_eq!(bn.running_mean().as_slice(), rm.as_slice());
        assert_eq!(bn.running_var().as_slice(), rv.as_slice());
        assert_eq!(bn.num_batches_tracked(), nb);
    }
    // BatchNorm2d（rank 4）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let bn = BatchNorm2d::new(2, BN_EPS, 0.1).unwrap();
    let x = tape.var(&t(bn_input(16), &[2, 2, 2, 2]));
    let y = bn.bind(&tape).forward(&x).unwrap();
    let loss = bn_curvature_loss(&tape, &y, &[2, 2, 2, 2]).unwrap();
    let (rm, nb) = (bn.running_mean(), bn.num_batches_tracked());
    let _ = tape.backward_create_graph(&loss, &child).unwrap();
    assert_eq!(bn.running_mean().as_slice(), rm.as_slice());
    assert_eq!(bn.num_batches_tracked(), nb);
}

fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// 3 段（1 階 `Tape::backward`・子テープ `cg.grad`・孫テープ再生）すべてで有限か
/// つ 1 階と REQ-2 一致を確認する。`fixed` が `Some` なら eval モード。
fn bn_apply_fixed<'t>(
    x: &Var<'t>,
    w: &Var<'t>,
    b: &Var<'t>,
    c: usize,
    eps: f32,
    fixed: &Option<(Vec<f32>, Vec<f32>)>,
) -> Result<Var<'t>, AutodiffError> {
    match fixed {
        None => x.batch_norm(Some(w), Some(b), eps),
        Some((m, v)) => x.batch_norm_infer(
            Some(w),
            Some(b),
            &t(m.clone(), &[c]),
            &t(v.clone(), &[c]),
            eps,
        ),
    }
}

/// 3 段（1 階 `Tape::backward`・子テープ `cg.grad`・孫テープ再生）すべてで有限か
/// つ 1 階と REQ-2 一致を確認する（x・w・b の勾配）。`fixed` が `Some` なら eval。
/// 損失が `y` に線形なので勾配が x に依存しない場合（eval の dx 等）は 2・3 段目を
/// 省略し、x に依存する勾配（dw 等）で 2・3 段目を確認する。
fn check_bn_three_stage(
    ctx: &str,
    shape: &[usize],
    x0: Vec<f32>,
    coeff: Vec<f32>,
    eps: f32,
    fixed: Option<(Vec<f32>, Vec<f32>)>,
) {
    let c = shape[1];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0, shape));
    let w = tape.var(&t(vec![1.0; c], &[c]));
    let b = tape.var(&t(vec![0.0; c], &[c]));
    let y = bn_apply_fixed(&x, &w, &b, c, eps, &fixed).unwrap();
    let k = tape.var_no_grad(&t(coeff, shape));
    let loss = y.mul(&k).unwrap().sum(None).unwrap();
    let plain = tape.backward(&loss).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let dense = |v: &Tensor<f32>| v.contiguous().as_slice().unwrap().to_vec();
    for (var, name) in [(&x, "x"), (&w, "w"), (&b, "b")] {
        let expected = dense(plain.get(var).unwrap().unwrap());
        if !all_finite(&expected) {
            // 1 階自体が f32 に収まらない成分（巨大な上流勾配の dw 等）は対象外。
            continue;
        }
        let gv = cg.grad(var).unwrap().unwrap();
        let gt = gv.to_tensor();
        let got = dense(&gt);
        assert!(
            all_finite(&got),
            "{ctx}/{name}: cg.grad が有限でない: {got:?}"
        );
        for (a, e) in got.iter().zip(expected.iter()) {
            assert!(
                common::req2_close(*a as f64, *e as f64),
                "{ctx}/{name}: cg.grad が 1 階と不一致 {a} vs {e}"
            );
        }
        // 2 段目: 子テープを微分し、さらに孫テープへ再生する。
        let w2 = child.var_no_grad(&t(vec![1e-3; got.len()], gt.shape()));
        let s2 = gv.mul(&w2).unwrap().sum(None).unwrap();
        // x に依存しない勾配（追跡なし）は 2 階以降の対象外。
        let hv = match child.backward(&s2) {
            Ok(hv) => hv,
            Err(AutodiffError::Backward(_)) => continue,
            Err(e) => panic!("{ctx}/{name}: child.backward が失敗: {e:?}"),
        };
        for target in [&x, &w, &b] {
            let cv = cg.child_var(target).unwrap().unwrap();
            if let Some(h) = hv.get(&cv).unwrap() {
                assert!(all_finite(&dense(h)), "{ctx}/{name}: 二階勾配が有限でない");
            }
        }
        let grand = Tape::new_with_ops(common::naive_ops());
        let cg2 = child.backward_create_graph(&s2, &grand).unwrap();
        for target in [&x, &w, &b] {
            let cv = cg.child_var(target).unwrap().unwrap();
            if let Some(g2) = cg2.grad(&cv).unwrap() {
                assert!(
                    all_finite(&dense(&g2.to_tensor())),
                    "{ctx}/{name}: 孫テープ再生の勾配が有限でない"
                );
            }
        }
    }
}
#[test]
fn batch_norm_create_graph_overflow_regression_vectors_are_finite() {
    let e = 1e-5f32;
    // x=1e20 系（f32 の二乗が overflow する）。
    check_bn_three_stage(
        "1e20",
        &[4, 1],
        vec![1e20, 2e20, -1e20, 3e20],
        vec![1.0, 2.0, -1.0, 0.5],
        e,
        None,
    );
    // 中心化の丸め。
    check_bn_three_stage(
        "centering",
        &[2, 1],
        vec![16777216.0, 16777218.0],
        vec![1.0, 2.0],
        e,
        None,
    );
    // 小スケール（rstd ≈ 316.23）。
    check_bn_three_stage(
        "small",
        &[2, 1],
        vec![1e-30, 1e-30],
        vec![1.0, 2.0],
        e,
        None,
    );
    // 定数行。
    check_bn_three_stage("const", &[2, 1], vec![1e30, 1e30], vec![1.0, 2.0], e, None);
    // 全ゼロ。
    check_bn_three_stage("zeros", &[2, 1], vec![0.0, 0.0], vec![1.0, 2.0], e, None);
    // 前置スケール（x 側の中心化が overflow しうる）。
    check_bn_three_stage(
        "prescale",
        &[2, 1],
        vec![3e38, -3e38],
        vec![1.0, 2.0],
        e,
        None,
    );
    // 上流勾配が巨大。
    check_bn_three_stage(
        "huge-g",
        &[4, 1],
        vec![1.0, 2.0, 3.0, 4.5],
        vec![1e38, -1e38, 1e38, -1e38],
        e,
        None,
    );
    check_bn_three_stage(
        "huge-g-same-sign",
        &[4, 1],
        vec![1.0, 2.0, 3.0, 4.5],
        vec![1e38; 4],
        e,
        None,
    );
    // eval: x と mean が逆符号の大値。
    check_bn_three_stage(
        "eval-opposite",
        &[2, 1],
        vec![2e38, 1e38],
        vec![1.0, 2.0],
        e,
        Some((vec![-2e38], vec![1e30])),
    );
    // 巨大な定数入力（Codex P1）: forward は有限だが逆伝播の倍率列が
    // 先に掛かると二階勾配が overflow する（train／eval 両方）。
    check_bn_three_stage(
        "const-3e38-train",
        &[2, 1],
        vec![3e38, 3e38],
        vec![1.0, 2.0],
        e,
        None,
    );
    check_bn_three_stage(
        "const-3e38-eval",
        &[2, 1],
        vec![3e38, 3e38],
        vec![1.0, 2.0],
        e,
        Some((vec![3e38], vec![0.0])),
    );
    // eval: 通常値。
    check_bn_three_stage(
        "eval-normal",
        &[3, 2],
        bn_input(6),
        bn_coeffs(6, 2),
        1e-3,
        Some((vec![0.1, -0.2], vec![0.9, 1.4])),
    );
}

/// 閉包性: 子テープの 1 階勾配をさらに `backward_create_graph` できる
/// （子テープが積む Op がすべて `supports_create_graph == true`）。
#[test]
fn batch_norm_create_graph_result_is_closed_under_create_graph() {
    for cfg in bn_configs() {
        let c = cfg.shape[1];
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(bn_input(flat_len(cfg.shape)), cfg.shape));
        let w = tape.var(&bn_w(c));
        let b = tape.var(&bn_b(c));
        let y = if cfg.affine {
            bn_apply(cfg, &x, Some(&w), Some(&b)).unwrap()
        } else {
            bn_apply(cfg, &x, None, None).unwrap()
        };
        let loss = bn_curvature_loss(&tape, &y, cfg.shape).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let gx = cg.grad(&x).unwrap().unwrap();
        let scalar = gx.sum(None).unwrap();
        child
            .backward_create_graph(&scalar, &grand)
            .expect("子テープの BatchNorm 二階 VJP 構造は孫テープへ再生できるはず");
    }
}

/// fail-closed の隣接 Op: 対象外のまま残る `local_response_norm` は入口で拒否され、
/// 子テープは無変更のまま。
#[test]
fn batch_norm_neighbor_local_response_norm_stays_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(bn_input(16), &[1, 4, 2, 2]));
    let y = x.local_response_norm(3, 1e-4, 0.75, 1.0).unwrap();
    let bn_y = y.batch_norm(None, None, BN_EPS).unwrap();
    let loss = bn_y.sum(None).unwrap();
    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
    assert!(child.is_empty());
}

/// 空テンソル（`n == 0`）の eval BatchNorm は入口検査で拒否され、子テープは無変更。
#[test]
fn batch_norm_create_graph_rejects_empty_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![], &[0, 2]));
    let (mean, var) = bn_fixed_stats(2);
    let Ok(y) = x.batch_norm_infer(None, None, &mean, &var, BN_EPS) else {
        // forward 自体が空入力を拒否する実装ならガード不要（到達しない）。
        return;
    };
    let loss = y.sum(None).unwrap();
    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
    assert!(child.is_empty());
}

/// ゼロ軸を含む形状の積が overflow する `[usize::MAX, 2, 0]` でも panic せず、
/// 型付きエラーで拒否される（形状の積ではなく `contains(&0)` で判定。Codex P1）。
#[test]
fn batch_norm_create_graph_rejects_overflowing_empty_shape() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let Ok(xt) = Tensor::new(Vec::<f32>::new(), &[usize::MAX, 2, 0]) else {
        return;
    };
    let x = tape.var(&xt);
    let (mean, var) = bn_fixed_stats(2);
    let Ok(y) = x.batch_norm_infer(None, None, &mean, &var, BN_EPS) else {
        return;
    };
    let Ok(loss) = y.sum(None) else {
        return;
    };
    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
    assert!(child.is_empty());
}

/// 巨大な定数入力（`x=mean=3e38`・`var=0`・追跡 weight=1・`loss=sum(y)`）で、
/// `dw` を重み 1 で逆伝播しても二階勾配が有限（Codex P1。逆伝播の倍率が相殺する
/// 順序で掛かること）。eval の `∂dw/∂x` は `rstd = 1/√eps ≈ 316.23`。
#[test]
fn batch_norm_create_graph_huge_constant_input_second_order_is_finite() {
    for train in [false, true] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![3e38; 2], &[2, 1]));
        let w = tape.var(&t(vec![1.0], &[1]));
        let y = if train {
            x.batch_norm(Some(&w), None, 1e-5).unwrap()
        } else {
            x.batch_norm_infer(
                Some(&w),
                None,
                &t(vec![3e38], &[1]),
                &t(vec![0.0], &[1]),
                1e-5,
            )
            .unwrap()
        };
        let loss = y.sum(None).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let dw = cg.grad(&w).unwrap().unwrap();
        let s2 = dw.sum(None).unwrap();
        let hv = child.backward(&s2).unwrap();
        let cx = cg.child_var(&x).unwrap().unwrap();
        let h = hv.get(&cx).unwrap().expect("dw は x に依存する");
        let hd = h.contiguous().as_slice().unwrap().to_vec();
        assert!(hd.iter().all(|v| v.is_finite()), "train={train}: {hd:?}");
        if !train {
            for v in &hd {
                assert!((v - 316.227_77).abs() < 0.5, "eval ∂dw/∂x = {v}");
            }
        }
    }
}

/// Codex P1 回帰: train で `s_g` の復元倍率が `inv_sg` より先に逆伝播され
/// 2^135 が単位シードへ掛かり二階勾配が NaN になった（x=[0,0]・上流 1e38）。
/// この損失の Hessian は 0 で、`child.backward` の結果は有限かつ 0。
#[test]
fn batch_norm_create_graph_train_huge_upstream_zero_input_hessian_is_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0, 0.0], &[2, 1]));
    let w = tape.var(&t(vec![1.0], &[1]));
    let y = x.batch_norm(Some(&w), None, 1e-5).unwrap();
    let k = tape.var_no_grad(&t(vec![1e38, 1e38], &[2, 1]));
    let loss = y.mul(&k).unwrap().sum(None).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let dx = cg.grad(&x).unwrap().expect("dx");
    let s = dx.sum(None).unwrap();
    let hv = child.backward(&s).unwrap();
    let cx = cg.child_var(&x).unwrap().unwrap();
    if let Some(h) = hv.get(&cx).unwrap() {
        let hd = h.contiguous().as_slice().unwrap().to_vec();
        assert!(hd.iter().all(|v| v.is_finite() && *v == 0.0), "{hd:?}");
    }
}

/// Codex P1 回帰: 上流勾配 `1e20` を含む混合二階勾配 `∂dw/∂x` が有限（eval は
/// `g·rstd ≈ 3.16e22`）。固定の前置スケール上限では単位シードしか保護できず、
/// 上流勾配に `2^(k+p)` が先に掛かり overflow した。eval と train の両経路を検証する。
#[test]
fn batch_norm_create_graph_huge_constant_input_with_huge_upstream_is_finite() {
    for train in [false, true] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![3e38; 2], &[2, 1]));
        let w = tape.var(&t(vec![1.0], &[1]));
        let y = if train {
            x.batch_norm(Some(&w), None, 1e-5).unwrap()
        } else {
            x.batch_norm_infer(
                Some(&w),
                None,
                &t(vec![3e38], &[1]),
                &t(vec![0.0], &[1]),
                1e-5,
            )
            .unwrap()
        };
        let k = tape.var_no_grad(&t(vec![1e20, 1e20], &[2, 1]));
        let loss = y.mul(&k).unwrap().sum(None).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let dw = cg.grad(&w).unwrap().unwrap();
        let s2 = dw.sum(None).unwrap();
        let hv = child.backward(&s2).unwrap();
        let cx = cg.child_var(&x).unwrap().unwrap();
        if let Some(h) = hv.get(&cx).unwrap() {
            let hd = h.contiguous().as_slice().unwrap().to_vec();
            assert!(hd.iter().all(|v| v.is_finite()), "train={train}: {hd:?}");
            if !train {
                for v in &hd {
                    assert!((v / 3.162_277_7e22 - 1.0).abs() < 1e-3, "eval ∂dw/∂x = {v}");
                }
            }
        }
    }
}
