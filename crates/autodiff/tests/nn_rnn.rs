//! RNN／LSTM／GRU セル演算・時系列ループの受け入れ条件検証（イシュー
//! #1647・設計 `docs/autodiff-rnn-cell-tape-design.md` 決定 11「#1647 へ
//! の引き継ぎ」）。
//!
//! `tests/backward.rs` と同じ「①専用 Op と合成参照実装の一致 →
//! ②多入力の数値微分突合 → ③BPTT → ④エラー経路」の構成を踏襲する。
//! `common::naive_ops()`（`BackendOps` の RNN 系メソッドはいずれも既定
//! `Unsupported`）のみを使い、`backend-cpu` 等の具体バックエンドへは
//! 依存しない（`common/mod.rs` 冒頭コメントの設計上の不変条件）。

mod common;

use fandhe_ai_autodiff::nn::{Gru, GruCell, Lstm, LstmCell, Module, Rnn, RnnCell};
use fandhe_ai_autodiff::{AutodiffError, GateParams, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

// `tests/backward.rs` と同じ許容誤差・数値微分ステップ幅（新しい許容
// 誤差は導入しない。設計 doc 決定 11 (b)）。
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
            "{label}[flat={flat} idx={index:?}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
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

/// 指定テンソルの各要素を中央差分で摂動し、`forward_loss` に対する
/// 数値勾配を計算する（`tests/backward.rs::numeric_grad` と同じ実装）。
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

// =====================================================================
// フィクスチャ（B=2, D=3, H=4。決定 12 の数式を候補 A〈合成参照実装〉
// として直接組み立てる。GRU の `1 − z` は `Var::sub` 未実装〈#1593〉の
// ため定数葉 `ones`／`neg_ones` で `ones + z*neg_ones` として表現する）。
// =====================================================================

const B: usize = 2;
const D: usize = 3;
const HID: usize = 4;

/// `rnn_fixture_data` の戻り値（x, h_prev, w_ih, w_hh, b_ih, b_hh）。
/// clippy::type_complexity 回避のための型エイリアス
/// （他ファイルで既に採用されているパターン。例:
/// `crates/backend-cpu/src/rnn_cell.rs::TripleVecOutput`）。
type RnnFixtureData = (
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
);

fn rnn_fixture_data() -> RnnFixtureData {
    let x = t(vec![0.10, -0.20, 0.30, 0.15, -0.25, 0.05], &[B, D]);
    let h_prev = t(
        vec![0.05, -0.10, 0.20, -0.15, 0.30, -0.05, 0.10, -0.20],
        &[B, HID],
    );
    let w_ih = t(
        (0..D * HID).map(|i| 0.05 * (i as f32) - 0.3).collect(),
        &[D, HID],
    );
    let w_hh = t(
        (0..HID * HID).map(|i| 0.04 * (i as f32) - 0.25).collect(),
        &[HID, HID],
    );
    let b_ih = t(vec![0.01, -0.02, 0.03, -0.01], &[HID]);
    let b_hh = t(vec![-0.02, 0.01, -0.01, 0.02], &[HID]);
    (x, h_prev, w_ih, w_hh, b_ih, b_hh)
}

// --- (a) RNN: 専用 Op と合成参照実装（決定 12）の forward 一致 ---

fn rnn_composite_forward<'t>(
    x: &Var<'t>,
    h_prev: &Var<'t>,
    w_ih: &Var<'t>,
    w_hh: &Var<'t>,
    b_ih: &Var<'t>,
    b_hh: &Var<'t>,
) -> Var<'t> {
    let pre = x
        .matmul(w_ih)
        .unwrap()
        .add(b_ih)
        .unwrap()
        .add(&h_prev.matmul(w_hh).unwrap())
        .unwrap()
        .add(b_hh)
        .unwrap();
    pre.tanh()
}

#[test]
fn rnn_cell_dedicated_op_matches_composite_reference_forward() {
    let (x_t, h_prev_t, w_ih_t, w_hh_t, b_ih_t, b_hh_t) = rnn_fixture_data();

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&x_t);
    let h_prev = tape.var(&h_prev_t);
    let w_ih = tape.var(&w_ih_t);
    let w_hh = tape.var(&w_hh_t);
    let b_ih = tape.var(&b_ih_t);
    let b_hh = tape.var(&b_hh_t);

    let dedicated = x
        .rnn_cell(
            &h_prev,
            GateParams {
                w_ih: &w_ih,
                w_hh: &w_hh,
                b_ih: Some(&b_ih),
                b_hh: Some(&b_hh),
            },
        )
        .unwrap();
    let composite = rnn_composite_forward(&x, &h_prev, &w_ih, &w_hh, &b_ih, &b_hh);

    assert_grad_close(
        "rnn forward dedicated vs composite",
        &dedicated.to_tensor(),
        &composite.to_tensor(),
    );
}

// --- (b) RNN: 多入力への数値微分突合 ---

fn rnn_forward_loss(
    x: &Tensor<f32>,
    h_prev: &Tensor<f32>,
    w_ih: &Tensor<f32>,
    w_hh: &Tensor<f32>,
    b_ih: &Tensor<f32>,
    b_hh: &Tensor<f32>,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let hv = tape.var(h_prev);
    let wih = tape.var(w_ih);
    let whh = tape.var(w_hh);
    let bih = tape.var(b_ih);
    let bhh = tape.var(b_hh);
    let h_t = xv
        .rnn_cell(
            &hv,
            GateParams {
                w_ih: &wih,
                w_hh: &whh,
                b_ih: Some(&bih),
                b_hh: Some(&bhh),
            },
        )
        .unwrap();
    let loss = h_t.sum(None).unwrap();
    scalar(&loss.to_tensor())
}

#[test]
fn rnn_cell_gradients_match_numeric_diff_for_all_inputs() {
    let (x_t, h_prev_t, w_ih_t, w_hh_t, b_ih_t, b_hh_t) = rnn_fixture_data();

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&x_t);
    let h_prev = tape.var(&h_prev_t);
    let w_ih = tape.var(&w_ih_t);
    let w_hh = tape.var(&w_hh_t);
    let b_ih = tape.var(&b_ih_t);
    let b_hh = tape.var(&b_hh_t);

    let h_out = x
        .rnn_cell(
            &h_prev,
            GateParams {
                w_ih: &w_ih,
                w_hh: &w_hh,
                b_ih: Some(&b_ih),
                b_hh: Some(&b_hh),
            },
        )
        .unwrap();
    let loss = h_out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();

    let dx = grads.get(&x).unwrap().expect("x reaches loss").clone();
    let dh_prev = grads
        .get(&h_prev)
        .unwrap()
        .expect("h_prev reaches loss")
        .clone();
    let dw_ih = grads
        .get(&w_ih)
        .unwrap()
        .expect("w_ih reaches loss")
        .clone();
    let dw_hh = grads
        .get(&w_hh)
        .unwrap()
        .expect("w_hh reaches loss")
        .clone();
    let db_ih = grads
        .get(&b_ih)
        .unwrap()
        .expect("b_ih reaches loss")
        .clone();
    let db_hh = grads
        .get(&b_hh)
        .unwrap()
        .expect("b_hh reaches loss")
        .clone();

    assert_grad_close(
        "rnn dx",
        &dx,
        &numeric_grad(&x_t, |v| {
            rnn_forward_loss(&v, &h_prev_t, &w_ih_t, &w_hh_t, &b_ih_t, &b_hh_t)
        }),
    );
    assert_grad_close(
        "rnn dh_prev",
        &dh_prev,
        &numeric_grad(&h_prev_t, |v| {
            rnn_forward_loss(&x_t, &v, &w_ih_t, &w_hh_t, &b_ih_t, &b_hh_t)
        }),
    );
    assert_grad_close(
        "rnn dw_ih",
        &dw_ih,
        &numeric_grad(&w_ih_t, |v| {
            rnn_forward_loss(&x_t, &h_prev_t, &v, &w_hh_t, &b_ih_t, &b_hh_t)
        }),
    );
    assert_grad_close(
        "rnn dw_hh",
        &dw_hh,
        &numeric_grad(&w_hh_t, |v| {
            rnn_forward_loss(&x_t, &h_prev_t, &w_ih_t, &v, &b_ih_t, &b_hh_t)
        }),
    );
    assert_grad_close(
        "rnn db_ih",
        &db_ih,
        &numeric_grad(&b_ih_t, |v| {
            rnn_forward_loss(&x_t, &h_prev_t, &w_ih_t, &w_hh_t, &v, &b_hh_t)
        }),
    );
    assert_grad_close(
        "rnn db_hh",
        &db_hh,
        &numeric_grad(&b_hh_t, |v| {
            rnn_forward_loss(&x_t, &h_prev_t, &w_ih_t, &w_hh_t, &b_ih_t, &v)
        }),
    );
}

// =====================================================================
// LSTM フィクスチャ（決定 5・12。ゲート順 i,f,g,o）。
// =====================================================================

/// `lstm_fixture_data` の戻り値（x, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh）。
/// clippy::type_complexity 回避のための型エイリアス（上記 `RnnFixtureData` 参照）。
type LstmFixtureData = (
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
);

fn lstm_fixture_data() -> LstmFixtureData {
    let x = t(vec![0.10, -0.20, 0.30, 0.15, -0.25, 0.05], &[B, D]);
    let h_prev = t(
        vec![0.05, -0.10, 0.20, -0.15, 0.30, -0.05, 0.10, -0.20],
        &[B, HID],
    );
    let c_prev = t(
        vec![0.02, -0.03, 0.01, 0.04, -0.01, 0.02, -0.02, 0.03],
        &[B, HID],
    );
    let w_ih = t(
        (0..D * 4 * HID).map(|i| 0.02 * (i as f32) - 0.4).collect(),
        &[D, 4 * HID],
    );
    let w_hh = t(
        (0..HID * 4 * HID)
            .map(|i| 0.015 * (i as f32) - 0.35)
            .collect(),
        &[HID, 4 * HID],
    );
    let b_ih = t(
        (0..4 * HID).map(|i| 0.01 * (i as f32) - 0.08).collect(),
        &[4 * HID],
    );
    let b_hh = t(
        (0..4 * HID).map(|i| -0.01 * (i as f32) + 0.05).collect(),
        &[4 * HID],
    );
    (x, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh)
}

/// LSTM の候補 A 合成参照実装（決定 5 (ii)・12）。`Var::narrow`
/// （#1599）が未実装のため、ゲートごとに事前列分割済みの重み・bias
/// （各ゲート用に切り出した `Var`）を受け取る変則版とする。決定 12 の
/// ごとの独立 GEMM 4 本で合成する（決定 5 (ii)）。
#[allow(clippy::too_many_arguments)]
fn lstm_composite_forward_vars<'t>(
    x: &Var<'t>,
    h_prev: &Var<'t>,
    c_prev: &Var<'t>,
    w_ih_gates: &[Var<'t>; 4],
    w_hh_gates: &[Var<'t>; 4],
    b_ih_gates: &[Var<'t>; 4],
    b_hh_gates: &[Var<'t>; 4],
) -> (Var<'t>, Var<'t>) {
    let pre = |g: usize| -> Var<'t> {
        x.matmul(&w_ih_gates[g])
            .unwrap()
            .add(&b_ih_gates[g])
            .unwrap()
            .add(&h_prev.matmul(&w_hh_gates[g]).unwrap())
            .unwrap()
            .add(&b_hh_gates[g])
            .unwrap()
    };
    let i = pre(0).sigmoid();
    let f = pre(1).sigmoid();
    let g = pre(2).tanh();
    let o = pre(3).sigmoid();
    let c = f.mul(c_prev).unwrap().add(&i.mul(&g).unwrap()).unwrap();
    let h = o.mul(&c.tanh()).unwrap();
    (h, c)
}

/// `[D or H, 4*HID]` の重みを 4 個の `[D or H, HID]` 列ブロックへ分割し、
/// それぞれをテープへ葉登録する（決定 5 (ii) の候補 A 合成参照実装が
/// 必要とする「ゲートごとに独立 GEMM」表現。`Var::narrow` 未実装
/// 〈#1599〉のためホスト側 `Tensor::narrow` で事前分割する）。
fn split_gate_columns<'t>(tape: &'t Tape, full: &Tensor<f32>, gates: usize) -> Vec<Var<'t>> {
    let rows = full.shape()[0];
    let total_cols = full.shape()[1];
    let width = total_cols / gates;
    (0..gates)
        .map(|g| {
            let block = full.narrow(1, g * width, width).unwrap().contiguous();
            debug_assert_eq!(block.shape(), [rows, width]);
            tape.var(&block)
        })
        .collect()
}

fn split_gate_bias<'t>(tape: &'t Tape, full: &Tensor<f32>, gates: usize) -> Vec<Var<'t>> {
    let total = full.shape()[0];
    let width = total / gates;
    (0..gates)
        .map(|g| {
            let block = full.narrow(0, g * width, width).unwrap().contiguous();
            tape.var(&block)
        })
        .collect()
}

#[test]
fn lstm_cell_dedicated_op_matches_composite_reference_forward() {
    let (x_t, h_prev_t, c_prev_t, w_ih_t, w_hh_t, b_ih_t, b_hh_t) = lstm_fixture_data();

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&x_t);
    let h_prev = tape.var(&h_prev_t);
    let c_prev = tape.var(&c_prev_t);
    let w_ih = tape.var(&w_ih_t);
    let w_hh = tape.var(&w_hh_t);
    let b_ih = tape.var(&b_ih_t);
    let b_hh = tape.var(&b_hh_t);

    let (h_ded, c_ded) = x
        .lstm_cell(
            &h_prev,
            &c_prev,
            GateParams {
                w_ih: &w_ih,
                w_hh: &w_hh,
                b_ih: Some(&b_ih),
                b_hh: Some(&b_hh),
            },
        )
        .unwrap();

    let w_ih_gates: [Var; 4] = split_gate_columns(&tape, &w_ih_t, 4).try_into().unwrap();
    let w_hh_gates: [Var; 4] = split_gate_columns(&tape, &w_hh_t, 4).try_into().unwrap();
    let b_ih_gates: [Var; 4] = split_gate_bias(&tape, &b_ih_t, 4).try_into().unwrap();
    let b_hh_gates: [Var; 4] = split_gate_bias(&tape, &b_hh_t, 4).try_into().unwrap();
    let (h_comp, c_comp) = lstm_composite_forward_vars(
        &x,
        &h_prev,
        &c_prev,
        &w_ih_gates,
        &w_hh_gates,
        &b_ih_gates,
        &b_hh_gates,
    );

    assert_grad_close(
        "lstm h dedicated vs composite",
        &h_ded.to_tensor(),
        &h_comp.to_tensor(),
    );
    assert_grad_close(
        "lstm c dedicated vs composite",
        &c_ded.to_tensor(),
        &c_comp.to_tensor(),
    );
}

fn lstm_forward_loss(
    x: &Tensor<f32>,
    h_prev: &Tensor<f32>,
    c_prev: &Tensor<f32>,
    w_ih: &Tensor<f32>,
    w_hh: &Tensor<f32>,
    b_ih: &Tensor<f32>,
    b_hh: &Tensor<f32>,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let hv = tape.var(h_prev);
    let cv = tape.var(c_prev);
    let wih = tape.var(w_ih);
    let whh = tape.var(w_hh);
    let bih = tape.var(b_ih);
    let bhh = tape.var(b_hh);
    let (h_t, c_t) = xv
        .lstm_cell(
            &hv,
            &cv,
            GateParams {
                w_ih: &wih,
                w_hh: &whh,
                b_ih: Some(&bih),
                b_hh: Some(&bhh),
            },
        )
        .unwrap();
    // h_t・c_t 双方の経路（決定 1b の 2 ノード表現）を loss に含める。
    let loss = h_t.sum(None).unwrap().add(&c_t.sum(None).unwrap()).unwrap();
    scalar(&loss.to_tensor())
}

#[test]
fn lstm_cell_gradients_match_numeric_diff_for_all_inputs_including_c_path() {
    let (x_t, h_prev_t, c_prev_t, w_ih_t, w_hh_t, b_ih_t, b_hh_t) = lstm_fixture_data();

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&x_t);
    let h_prev = tape.var(&h_prev_t);
    let c_prev = tape.var(&c_prev_t);
    let w_ih = tape.var(&w_ih_t);
    let w_hh = tape.var(&w_hh_t);
    let b_ih = tape.var(&b_ih_t);
    let b_hh = tape.var(&b_hh_t);

    let (h_t, c_t) = x
        .lstm_cell(
            &h_prev,
            &c_prev,
            GateParams {
                w_ih: &w_ih,
                w_hh: &w_hh,
                b_ih: Some(&b_ih),
                b_hh: Some(&b_hh),
            },
        )
        .unwrap();
    let loss = h_t.sum(None).unwrap().add(&c_t.sum(None).unwrap()).unwrap();
    let grads = tape.backward(&loss).unwrap();

    let dx = grads.get(&x).unwrap().unwrap().clone();
    let dh_prev = grads.get(&h_prev).unwrap().unwrap().clone();
    let dc_prev = grads.get(&c_prev).unwrap().unwrap().clone();
    let dw_ih = grads.get(&w_ih).unwrap().unwrap().clone();
    let dw_hh = grads.get(&w_hh).unwrap().unwrap().clone();
    let db_ih = grads.get(&b_ih).unwrap().unwrap().clone();
    let db_hh = grads.get(&b_hh).unwrap().unwrap().clone();

    assert_grad_close(
        "lstm dx",
        &dx,
        &numeric_grad(&x_t, |v| {
            lstm_forward_loss(&v, &h_prev_t, &c_prev_t, &w_ih_t, &w_hh_t, &b_ih_t, &b_hh_t)
        }),
    );
    assert_grad_close(
        "lstm dh_prev",
        &dh_prev,
        &numeric_grad(&h_prev_t, |v| {
            lstm_forward_loss(&x_t, &v, &c_prev_t, &w_ih_t, &w_hh_t, &b_ih_t, &b_hh_t)
        }),
    );
    assert_grad_close(
        "lstm dc_prev",
        &dc_prev,
        &numeric_grad(&c_prev_t, |v| {
            lstm_forward_loss(&x_t, &h_prev_t, &v, &w_ih_t, &w_hh_t, &b_ih_t, &b_hh_t)
        }),
    );
    assert_grad_close(
        "lstm dw_ih",
        &dw_ih,
        &numeric_grad(&w_ih_t, |v| {
            lstm_forward_loss(&x_t, &h_prev_t, &c_prev_t, &v, &w_hh_t, &b_ih_t, &b_hh_t)
        }),
    );
    assert_grad_close(
        "lstm dw_hh",
        &dw_hh,
        &numeric_grad(&w_hh_t, |v| {
            lstm_forward_loss(&x_t, &h_prev_t, &c_prev_t, &w_ih_t, &v, &b_ih_t, &b_hh_t)
        }),
    );
    assert_grad_close(
        "lstm db_ih",
        &db_ih,
        &numeric_grad(&b_ih_t, |v| {
            lstm_forward_loss(&x_t, &h_prev_t, &c_prev_t, &w_ih_t, &w_hh_t, &v, &b_hh_t)
        }),
    );
    assert_grad_close(
        "lstm db_hh",
        &db_hh,
        &numeric_grad(&b_hh_t, |v| {
            lstm_forward_loss(&x_t, &h_prev_t, &c_prev_t, &w_ih_t, &w_hh_t, &b_ih_t, &v)
        }),
    );
}

// =====================================================================
// GRU フィクスチャ（決定 1c・5・12。`reset_after=True`。ゲート順 r,z,n）。
// =====================================================================

/// `gru_fixture_data` の戻り値（x, h_prev, w_ih, w_hh, b_ih, b_hh）。
/// clippy::type_complexity 回避のための型エイリアス（上記 `RnnFixtureData` 参照）。
type GruFixtureData = (
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
);

fn gru_fixture_data() -> GruFixtureData {
    let x = t(vec![0.10, -0.20, 0.30, 0.15, -0.25, 0.05], &[B, D]);
    let h_prev = t(
        vec![0.05, -0.10, 0.20, -0.15, 0.30, -0.05, 0.10, -0.20],
        &[B, HID],
    );
    let w_ih = t(
        (0..D * 3 * HID).map(|i| 0.03 * (i as f32) - 0.35).collect(),
        &[D, 3 * HID],
    );
    let w_hh = t(
        (0..HID * 3 * HID)
            .map(|i| 0.02 * (i as f32) - 0.3)
            .collect(),
        &[HID, 3 * HID],
    );
    let b_ih = t(
        (0..3 * HID).map(|i| 0.01 * (i as f32) - 0.06).collect(),
        &[3 * HID],
    );
    let b_hh = t(
        (0..3 * HID).map(|i| -0.01 * (i as f32) + 0.04).collect(),
        &[3 * HID],
    );
    (x, h_prev, w_ih, w_hh, b_ih, b_hh)
}

/// GRU の候補 A 合成参照実装（決定 12。`n = tanh(pre_i_n + r*(h*W_hn +
/// b_hn))`・`h_t = (1-z)*n + z*h_prev`）。`1 - z` は `Var::sub`〈#1593〉
/// 未実装のため定数葉 `ones`／`neg_ones`（`z` と同 shape）で表現する
/// （`ones + z*neg_ones`）。
#[allow(clippy::too_many_arguments)]
fn gru_composite_forward_vars<'t>(
    tape: &'t Tape,
    x: &Var<'t>,
    h_prev: &Var<'t>,
    w_ih_gates: &[Var<'t>; 3],
    w_hh_gates: &[Var<'t>; 3],
    b_ih_gates: &[Var<'t>; 3],
    b_hh_gates: &[Var<'t>; 3],
) -> Var<'t> {
    let pre_i = |g: usize| -> Var<'t> {
        x.matmul(&w_ih_gates[g])
            .unwrap()
            .add(&b_ih_gates[g])
            .unwrap()
    };
    let pre_h = |g: usize| -> Var<'t> {
        h_prev
            .matmul(&w_hh_gates[g])
            .unwrap()
            .add(&b_hh_gates[g])
            .unwrap()
    };
    let r = pre_i(0).add(&pre_h(0)).unwrap().sigmoid();
    let z = pre_i(1).add(&pre_h(1)).unwrap().sigmoid();
    let n = pre_i(2).add(&r.mul(&pre_h(2)).unwrap()).unwrap().tanh();

    let ones = tape.var(&t(vec![1.0; B * HID], &[B, HID]));
    let neg_ones = tape.var(&t(vec![-1.0; B * HID], &[B, HID]));
    let one_minus_z = ones.add(&z.mul(&neg_ones).unwrap()).unwrap();
    one_minus_z
        .mul(&n)
        .unwrap()
        .add(&z.mul(h_prev).unwrap())
        .unwrap()
}

#[test]
fn gru_cell_dedicated_op_matches_composite_reference_forward() {
    let (x_t, h_prev_t, w_ih_t, w_hh_t, b_ih_t, b_hh_t) = gru_fixture_data();

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&x_t);
    let h_prev = tape.var(&h_prev_t);
    let w_ih = tape.var(&w_ih_t);
    let w_hh = tape.var(&w_hh_t);
    let b_ih = tape.var(&b_ih_t);
    let b_hh = tape.var(&b_hh_t);

    let dedicated = x
        .gru_cell(
            &h_prev,
            GateParams {
                w_ih: &w_ih,
                w_hh: &w_hh,
                b_ih: Some(&b_ih),
                b_hh: Some(&b_hh),
            },
        )
        .unwrap();

    let w_ih_gates: [Var; 3] = split_gate_columns(&tape, &w_ih_t, 3).try_into().unwrap();
    let w_hh_gates: [Var; 3] = split_gate_columns(&tape, &w_hh_t, 3).try_into().unwrap();
    let b_ih_gates: [Var; 3] = split_gate_bias(&tape, &b_ih_t, 3).try_into().unwrap();
    let b_hh_gates: [Var; 3] = split_gate_bias(&tape, &b_hh_t, 3).try_into().unwrap();
    let composite = gru_composite_forward_vars(
        &tape,
        &x,
        &h_prev,
        &w_ih_gates,
        &w_hh_gates,
        &b_ih_gates,
        &b_hh_gates,
    );

    assert_grad_close(
        "gru forward dedicated vs composite",
        &dedicated.to_tensor(),
        &composite.to_tensor(),
    );
}

fn gru_forward_loss(
    x: &Tensor<f32>,
    h_prev: &Tensor<f32>,
    w_ih: &Tensor<f32>,
    w_hh: &Tensor<f32>,
    b_ih: &Tensor<f32>,
    b_hh: &Tensor<f32>,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let hv = tape.var(h_prev);
    let wih = tape.var(w_ih);
    let whh = tape.var(w_hh);
    let bih = tape.var(b_ih);
    let bhh = tape.var(b_hh);
    let h_t = xv
        .gru_cell(
            &hv,
            GateParams {
                w_ih: &wih,
                w_hh: &whh,
                b_ih: Some(&bih),
                b_hh: Some(&bhh),
            },
        )
        .unwrap();
    let loss = h_t.sum(None).unwrap();
    scalar(&loss.to_tensor())
}

#[test]
fn gru_cell_gradients_match_numeric_diff_for_all_inputs() {
    let (x_t, h_prev_t, w_ih_t, w_hh_t, b_ih_t, b_hh_t) = gru_fixture_data();

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&x_t);
    let h_prev = tape.var(&h_prev_t);
    let w_ih = tape.var(&w_ih_t);
    let w_hh = tape.var(&w_hh_t);
    let b_ih = tape.var(&b_ih_t);
    let b_hh = tape.var(&b_hh_t);

    let h_out = x
        .gru_cell(
            &h_prev,
            GateParams {
                w_ih: &w_ih,
                w_hh: &w_hh,
                b_ih: Some(&b_ih),
                b_hh: Some(&b_hh),
            },
        )
        .unwrap();
    let loss = h_out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();

    let dx = grads.get(&x).unwrap().unwrap().clone();
    let dh_prev = grads.get(&h_prev).unwrap().unwrap().clone();
    let dw_ih = grads.get(&w_ih).unwrap().unwrap().clone();
    let dw_hh = grads.get(&w_hh).unwrap().unwrap().clone();
    let db_ih = grads.get(&b_ih).unwrap().unwrap().clone();
    let db_hh = grads.get(&b_hh).unwrap().unwrap().clone();

    assert_grad_close(
        "gru dx",
        &dx,
        &numeric_grad(&x_t, |v| {
            gru_forward_loss(&v, &h_prev_t, &w_ih_t, &w_hh_t, &b_ih_t, &b_hh_t)
        }),
    );
    assert_grad_close(
        "gru dh_prev",
        &dh_prev,
        &numeric_grad(&h_prev_t, |v| {
            gru_forward_loss(&x_t, &v, &w_ih_t, &w_hh_t, &b_ih_t, &b_hh_t)
        }),
    );
    assert_grad_close(
        "gru dw_ih",
        &dw_ih,
        &numeric_grad(&w_ih_t, |v| {
            gru_forward_loss(&x_t, &h_prev_t, &v, &w_hh_t, &b_ih_t, &b_hh_t)
        }),
    );
    assert_grad_close(
        "gru dw_hh",
        &dw_hh,
        &numeric_grad(&w_hh_t, |v| {
            gru_forward_loss(&x_t, &h_prev_t, &w_ih_t, &v, &b_ih_t, &b_hh_t)
        }),
    );
    assert_grad_close(
        "gru db_ih",
        &db_ih,
        &numeric_grad(&b_ih_t, |v| {
            gru_forward_loss(&x_t, &h_prev_t, &w_ih_t, &w_hh_t, &v, &b_hh_t)
        }),
    );
    assert_grad_close(
        "gru db_hh",
        &db_hh,
        &numeric_grad(&b_hh_t, |v| {
            gru_forward_loss(&x_t, &h_prev_t, &w_ih_t, &w_hh_t, &b_ih_t, &v)
        }),
    );
}

// =====================================================================
// (c) BPTT: Sequence レベル API（`forward_seq`）の T=1／T=3。
// =====================================================================

fn seq_input(t_len: usize) -> Tensor<f32> {
    let data: Vec<f32> = (0..t_len * B * D)
        .map(|i| 0.05 * (i as f32) - 0.4)
        .collect();
    t(data, &[t_len, B, D])
}

#[test]
fn rnn_forward_seq_bptt_completes_for_t1_and_t3_with_correct_structure() {
    // 受入基準 (c) の構造面（T=1／T>1 とも forward_seq が完走し、
    // `h_n` が最終 step の出力と一致し、`Tape::backward` が例外なく
    // 完了すること）を検証する。重み勾配が T 個の寄与和になっている
    // ことの数値的な正しさは
    // `rnn_forward_seq_weight_gradient_is_sum_of_t_contributions`
    // （T=3）が本体として検証する。
    let rnn = RnnCell::new(D, HID, true, 42).map(Rnn::from_cell).unwrap();

    for &t_len in &[1usize, 3usize] {
        let x_t = seq_input(t_len);
        let tape = Tape::new_with_ops(common::naive_ops());
        let out = rnn.forward_seq(&tape, &x_t, None).unwrap();
        assert_eq!(out.outputs.len(), t_len);

        let last_output = out.outputs.last().expect("t_len > 0").to_tensor();
        let h_n = out.h_n.to_tensor();
        for b in 0..B {
            for h in 0..HID {
                assert_eq!(
                    last_output.get(&[b, h]).unwrap().to_bits(),
                    h_n.get(&[b, h]).unwrap().to_bits(),
                    "h_n は最終 step の出力と bit-exact に一致するはず（t_len={t_len}）"
                );
            }
        }

        let loss = out.h_n.sum(None).unwrap();
        assert!(
            tape.backward(&loss).is_ok(),
            "T={t_len} の BPTT が完走すること"
        );
    }
}

/// `forward_seq` の重み勾配（decision 2 の fan-in 蓄積）が T 個の寄与和
/// になっていることを、独立に構築した「1 個の重み `Var` を T 個の
/// セル呼び出しで共有する」手組みループの数値微分と突合して直接検証
/// する（上のテストは経路が完走することのみを見ているため、本テストが
/// 受入基準 (c) の本体）。
fn rnn_manual_unrolled_loss(
    x: &Tensor<f32>,
    w_ih: &Tensor<f32>,
    w_hh: &Tensor<f32>,
    b_ih: &Tensor<f32>,
    b_hh: &Tensor<f32>,
    t_len: usize,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let wih = tape.var(w_ih);
    let whh = tape.var(w_hh);
    let bih = tape.var(b_ih);
    let bhh = tape.var(b_hh);
    let mut h = tape.var(&t(vec![0.0; B * HID], &[B, HID]));
    let mut loss_sum = tape.var(&t(vec![0.0], &[]));
    for step in 0..t_len {
        let x_t_data = x
            .narrow(0, step, 1)
            .unwrap()
            .contiguous()
            .reshape(&[B, D])
            .unwrap();
        let x_t = tape.var(&x_t_data);
        h = x_t
            .rnn_cell(
                &h,
                GateParams {
                    w_ih: &wih,
                    w_hh: &whh,
                    b_ih: Some(&bih),
                    b_hh: Some(&bhh),
                },
            )
            .unwrap();
        loss_sum = loss_sum.add(&h.sum(None).unwrap()).unwrap();
    }
    scalar(&loss_sum.to_tensor())
}

#[test]
fn rnn_forward_seq_weight_gradient_is_sum_of_t_contributions() {
    let (_, _, w_ih_t, w_hh_t, b_ih_t, b_hh_t) = rnn_fixture_data();
    let t_len = 3usize;
    let x_t = seq_input(t_len);

    let tape = Tape::new_with_ops(common::naive_ops());
    let wih = tape.var(&w_ih_t);
    let whh = tape.var(&w_hh_t);
    let bih = tape.var(&b_ih_t);
    let bhh = tape.var(&b_hh_t);
    let mut h = tape.var(&t(vec![0.0; B * HID], &[B, HID]));
    let mut loss_sum = tape.var(&t(vec![0.0], &[]));
    for step in 0..t_len {
        let x_t_data = x_t
            .narrow(0, step, 1)
            .unwrap()
            .contiguous()
            .reshape(&[B, D])
            .unwrap();
        let x_step = tape.var(&x_t_data);
        h = x_step
            .rnn_cell(
                &h,
                GateParams {
                    w_ih: &wih,
                    w_hh: &whh,
                    b_ih: Some(&bih),
                    b_hh: Some(&bhh),
                },
            )
            .unwrap();
        loss_sum = loss_sum.add(&h.sum(None).unwrap()).unwrap();
    }
    let grads = tape.backward(&loss_sum).unwrap();
    let dw_ih = grads.get(&wih).unwrap().unwrap().clone();
    let dw_hh = grads.get(&whh).unwrap().unwrap().clone();

    assert_grad_close(
        "rnn seq dw_ih (T=3 fan-in sum)",
        &dw_ih,
        &numeric_grad(&w_ih_t, |v| {
            rnn_manual_unrolled_loss(&x_t, &v, &w_hh_t, &b_ih_t, &b_hh_t, t_len)
        }),
    );
    assert_grad_close(
        "rnn seq dw_hh (T=3 fan-in sum)",
        &dw_hh,
        &numeric_grad(&w_hh_t, |v| {
            rnn_manual_unrolled_loss(&x_t, &w_ih_t, &v, &b_ih_t, &b_hh_t, t_len)
        }),
    );
}

/// `Rnn::forward_seq` が返す `RnnSeqOutput::params`（この呼び出しが
/// 実際に使ったテープ登録済みパラメータ）を使って
/// `Gradients::get(&out.params.weight_ih)` 等から直接勾配を取得できる
/// ことを検証する（イシュー #1647 codex-review P1 指摘: `forward_seq`
/// が内部で `bind` した `Var` を返さないと、呼び出し元はこの計算で
/// 使われた重み・bias の勾配を取得する手段がなく学習経路として機能
/// しなかった）。数値は上の
/// `rnn_forward_seq_weight_gradient_is_sum_of_t_contributions` と同じ
/// フィクスチャ・同じ手組みループ参照実装（`rnn_manual_unrolled_loss`）
/// で突合する。
#[test]
fn rnn_forward_seq_exposes_params_for_gradient_retrieval() {
    let (_, _, w_ih_t, w_hh_t, b_ih_t, b_hh_t) = rnn_fixture_data();
    let t_len = 3usize;
    let x_t = seq_input(t_len);

    let cell = RnnCell::from_parameters(
        w_ih_t.clone(),
        w_hh_t.clone(),
        Some(b_ih_t.clone()),
        Some(b_hh_t.clone()),
    )
    .unwrap();
    let rnn = Rnn::from_cell(cell);

    let tape = Tape::new_with_ops(common::naive_ops());
    let out = rnn.forward_seq(&tape, &x_t, None).unwrap();
    // `rnn_manual_unrolled_loss` は各 step の `h` を loss へ加算する
    // （`h_n` のみではなく全 step 分の寄与を突合するため。上の
    // `rnn_forward_seq_weight_gradient_is_sum_of_t_contributions` と
    // 同じ loss 構成に揃える）。
    let mut loss = out.outputs[0].sum(None).unwrap();
    for h in &out.outputs[1..] {
        loss = loss.add(&h.sum(None).unwrap()).unwrap();
    }
    let grads = tape.backward(&loss).unwrap();

    // `out.params` は `forward_seq` 内部で実際に計算グラフへ登録された
    // `Var` そのもの（別途 `cell.bind(tape)` した無関係な葉ではない）
    // なので `Gradients::get` が `Some` を返す。
    let dw_ih = grads
        .get(&out.params.weight_ih)
        .unwrap()
        .expect("forward_seq が使った weight_ih の勾配が取得できるはず")
        .clone();
    let dw_hh = grads
        .get(&out.params.weight_hh)
        .unwrap()
        .expect("forward_seq が使った weight_hh の勾配が取得できるはず")
        .clone();
    let db_ih = grads
        .get(out.params.bias_ih.as_ref().expect("bias=true で構築した"))
        .unwrap()
        .expect("forward_seq が使った bias_ih の勾配が取得できるはず")
        .clone();
    let db_hh = grads
        .get(out.params.bias_hh.as_ref().expect("bias=true で構築した"))
        .unwrap()
        .expect("forward_seq が使った bias_hh の勾配が取得できるはず")
        .clone();

    assert_grad_close(
        "rnn forward_seq params dw_ih",
        &dw_ih,
        &numeric_grad(&w_ih_t, |v| {
            rnn_manual_unrolled_loss(&x_t, &v, &w_hh_t, &b_ih_t, &b_hh_t, t_len)
        }),
    );
    assert_grad_close(
        "rnn forward_seq params dw_hh",
        &dw_hh,
        &numeric_grad(&w_hh_t, |v| {
            rnn_manual_unrolled_loss(&x_t, &w_ih_t, &v, &b_ih_t, &b_hh_t, t_len)
        }),
    );
    assert_grad_close(
        "rnn forward_seq params db_ih",
        &db_ih,
        &numeric_grad(&b_ih_t, |v| {
            rnn_manual_unrolled_loss(&x_t, &w_ih_t, &w_hh_t, &v, &b_hh_t, t_len)
        }),
    );
    assert_grad_close(
        "rnn forward_seq params db_hh",
        &db_hh,
        &numeric_grad(&b_hh_t, |v| {
            rnn_manual_unrolled_loss(&x_t, &w_ih_t, &w_hh_t, &b_ih_t, &v, t_len)
        }),
    );
}

// =====================================================================
// LSTM／GRU の複数ステップ（T>1）BPTT 勾配検証（イシュー #1647
// codex-review P2 指摘: 上の RNN テストのみが T>1 の重み勾配を数値
// 微分と突合しており、LSTM の `c_t` 経由の時系列勾配・GRU の複数
// ステップ勾配が未検証だった。`rnn_manual_unrolled_loss` と同じ
// 方針〈重み `Var` をループ外で 1 回だけ bind し T step 間で共有する
// ことで `forward_seq` 内部の fan-in 蓄積〈`vars = cell.bind(tape)`
// → ループ内 `vars.forward(...)`〉と同一の `Var::{lstm_cell,
// gru_cell}` 経路を通す〉で LSTM／GRU へ横展開する。決定 11 (c) の
// 受入基準〉。
// =====================================================================

fn lstm_seq_input(t_len: usize) -> Tensor<f32> {
    seq_input(t_len)
}

/// LSTM 版 `rnn_manual_unrolled_loss`。`c_t` を次 step の `c_prev` として
/// 連鎖させるため、時系列方向の勾配が `c` 経路のみを通じても正しく
/// 伝播することを検証できる（`h` は毎 step `loss_sum` へも加算する）。
#[allow(clippy::too_many_arguments)]
fn lstm_manual_unrolled_loss(
    x: &Tensor<f32>,
    w_ih: &Tensor<f32>,
    w_hh: &Tensor<f32>,
    b_ih: &Tensor<f32>,
    b_hh: &Tensor<f32>,
    h0: &Tensor<f32>,
    c0: &Tensor<f32>,
    t_len: usize,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let wih = tape.var(w_ih);
    let whh = tape.var(w_hh);
    let bih = tape.var(b_ih);
    let bhh = tape.var(b_hh);
    let mut h = tape.var(h0);
    let mut c = tape.var(c0);
    let mut loss_sum = tape.var(&t(vec![0.0], &[]));
    for step in 0..t_len {
        let x_t_data = x
            .narrow(0, step, 1)
            .unwrap()
            .contiguous()
            .reshape(&[B, D])
            .unwrap();
        let x_t = tape.var(&x_t_data);
        let (h_t, c_t) = x_t
            .lstm_cell(
                &h,
                &c,
                GateParams {
                    w_ih: &wih,
                    w_hh: &whh,
                    b_ih: Some(&bih),
                    b_hh: Some(&bhh),
                },
            )
            .unwrap();
        h = h_t;
        c = c_t;
        // `c` を直接 loss へ加算し、`c_t` 経由（`h` を介さない）の
        // 逆伝播経路も数値微分の対象に含める（決定 11 (c) の
        // 「LSTM の c_t 経由の BPTT」要求）。
        loss_sum = loss_sum
            .add(&h.sum(None).unwrap())
            .unwrap()
            .add(&c.sum(None).unwrap())
            .unwrap();
    }
    scalar(&loss_sum.to_tensor())
}

/// [`lstm_manual_unrolled_loss`] の「最終 step の `h_n`／`c_n` のみを
/// loss へ加算する」版（各 step 毎の中間和は取らない）。
/// [`lstm_forward_seq_exposes_params_for_gradient_retrieval`] が
/// `LstmSeqOutput::h_n`／`c_n`（`forward_seq` は per-step の `c` を
/// 返さないため `outputs` から中間 `c` を再構成できない）のみを使って
/// loss を組み立てるのに合わせた参照実装。
#[allow(clippy::too_many_arguments)]
fn lstm_manual_unrolled_final_loss(
    x: &Tensor<f32>,
    w_ih: &Tensor<f32>,
    w_hh: &Tensor<f32>,
    b_ih: &Tensor<f32>,
    b_hh: &Tensor<f32>,
    h0: &Tensor<f32>,
    c0: &Tensor<f32>,
    t_len: usize,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let wih = tape.var(w_ih);
    let whh = tape.var(w_hh);
    let bih = tape.var(b_ih);
    let bhh = tape.var(b_hh);
    let mut h = tape.var(h0);
    let mut c = tape.var(c0);
    for step in 0..t_len {
        let x_t_data = x
            .narrow(0, step, 1)
            .unwrap()
            .contiguous()
            .reshape(&[B, D])
            .unwrap();
        let x_t = tape.var(&x_t_data);
        let (h_t, c_t) = x_t
            .lstm_cell(
                &h,
                &c,
                GateParams {
                    w_ih: &wih,
                    w_hh: &whh,
                    b_ih: Some(&bih),
                    b_hh: Some(&bhh),
                },
            )
            .unwrap();
        h = h_t;
        c = c_t;
    }
    let loss_sum = h.sum(None).unwrap().add(&c.sum(None).unwrap()).unwrap();
    scalar(&loss_sum.to_tensor())
}

#[test]
fn lstm_forward_seq_multi_step_gradient_matches_numeric_diff() {
    let (_, h_prev, c_prev, w_ih_t, w_hh_t, b_ih_t, b_hh_t) = lstm_fixture_data();
    let t_len = 3usize;
    let x_t = lstm_seq_input(t_len);

    let tape = Tape::new_with_ops(common::naive_ops());
    let wih = tape.var(&w_ih_t);
    let whh = tape.var(&w_hh_t);
    let bih = tape.var(&b_ih_t);
    let bhh = tape.var(&b_hh_t);
    let mut h = tape.var(&h_prev);
    let mut c = tape.var(&c_prev);
    let h0_var = h;
    let c0_var = c;
    let mut loss_sum = tape.var(&t(vec![0.0], &[]));
    for step in 0..t_len {
        let x_t_data = x_t
            .narrow(0, step, 1)
            .unwrap()
            .contiguous()
            .reshape(&[B, D])
            .unwrap();
        let x_step = tape.var(&x_t_data);
        let (h_t, c_t) = x_step
            .lstm_cell(
                &h,
                &c,
                GateParams {
                    w_ih: &wih,
                    w_hh: &whh,
                    b_ih: Some(&bih),
                    b_hh: Some(&bhh),
                },
            )
            .unwrap();
        h = h_t;
        c = c_t;
        loss_sum = loss_sum
            .add(&h.sum(None).unwrap())
            .unwrap()
            .add(&c.sum(None).unwrap())
            .unwrap();
    }
    let grads = tape.backward(&loss_sum).unwrap();
    let dw_ih = grads.get(&wih).unwrap().unwrap().clone();
    let dw_hh = grads.get(&whh).unwrap().unwrap().clone();
    let dh0 = grads.get(&h0_var).unwrap().unwrap().clone();
    let dc0 = grads.get(&c0_var).unwrap().unwrap().clone();

    assert_grad_close(
        "lstm seq dw_ih (T=3 fan-in sum, c_t path included)",
        &dw_ih,
        &numeric_grad(&w_ih_t, |v| {
            lstm_manual_unrolled_loss(&x_t, &v, &w_hh_t, &b_ih_t, &b_hh_t, &h_prev, &c_prev, t_len)
        }),
    );
    assert_grad_close(
        "lstm seq dw_hh (T=3 fan-in sum, c_t path included)",
        &dw_hh,
        &numeric_grad(&w_hh_t, |v| {
            lstm_manual_unrolled_loss(&x_t, &w_ih_t, &v, &b_ih_t, &b_hh_t, &h_prev, &c_prev, t_len)
        }),
    );
    assert_grad_close(
        "lstm seq dh0 (T=3, c_t path included)",
        &dh0,
        &numeric_grad(&h_prev, |v| {
            lstm_manual_unrolled_loss(&x_t, &w_ih_t, &w_hh_t, &b_ih_t, &b_hh_t, &v, &c_prev, t_len)
        }),
    );
    assert_grad_close(
        "lstm seq dc0 (T=3, c_t path only)",
        &dc0,
        &numeric_grad(&c_prev, |v| {
            lstm_manual_unrolled_loss(&x_t, &w_ih_t, &w_hh_t, &b_ih_t, &b_hh_t, &h_prev, &v, t_len)
        }),
    );
}

/// `Lstm::forward_seq` が返す `LstmSeqOutput::params`（イシュー #1647
/// codex-review P1 指摘。[`rnn_forward_seq_exposes_params_for_gradient_retrieval`]
/// の LSTM 版）から `Gradients::get` で直接勾配を取得できることを検証
/// する。
#[test]
fn lstm_forward_seq_exposes_params_for_gradient_retrieval() {
    let (_, h_prev, c_prev, w_ih_t, w_hh_t, b_ih_t, b_hh_t) = lstm_fixture_data();
    let t_len = 3usize;
    let x_t = lstm_seq_input(t_len);

    let cell = LstmCell::from_parameters(
        w_ih_t.clone(),
        w_hh_t.clone(),
        Some(b_ih_t.clone()),
        Some(b_hh_t.clone()),
    )
    .unwrap();
    let lstm = Lstm::from_cell(cell);

    let tape = Tape::new_with_ops(common::naive_ops());
    let h0 = tape.var(&h_prev);
    let c0 = tape.var(&c_prev);
    let out = lstm.forward_seq(&tape, &x_t, Some(&h0), Some(&c0)).unwrap();
    let loss = out
        .h_n
        .sum(None)
        .unwrap()
        .add(&out.c_n.sum(None).unwrap())
        .unwrap();
    let grads = tape.backward(&loss).unwrap();

    let dw_ih = grads
        .get(&out.params.weight_ih)
        .unwrap()
        .expect("forward_seq が使った weight_ih の勾配が取得できるはず")
        .clone();

    assert_grad_close(
        "lstm forward_seq params dw_ih",
        &dw_ih,
        &numeric_grad(&w_ih_t, |v| {
            lstm_manual_unrolled_final_loss(
                &x_t, &v, &w_hh_t, &b_ih_t, &b_hh_t, &h_prev, &c_prev, t_len,
            )
        }),
    );
}

/// GRU 版 `rnn_manual_unrolled_loss`。
fn gru_manual_unrolled_loss(
    x: &Tensor<f32>,
    w_ih: &Tensor<f32>,
    w_hh: &Tensor<f32>,
    b_ih: &Tensor<f32>,
    b_hh: &Tensor<f32>,
    h0: &Tensor<f32>,
    t_len: usize,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let wih = tape.var(w_ih);
    let whh = tape.var(w_hh);
    let bih = tape.var(b_ih);
    let bhh = tape.var(b_hh);
    let mut h = tape.var(h0);
    let mut loss_sum = tape.var(&t(vec![0.0], &[]));
    for step in 0..t_len {
        let x_t_data = x
            .narrow(0, step, 1)
            .unwrap()
            .contiguous()
            .reshape(&[B, D])
            .unwrap();
        let x_t = tape.var(&x_t_data);
        h = x_t
            .gru_cell(
                &h,
                GateParams {
                    w_ih: &wih,
                    w_hh: &whh,
                    b_ih: Some(&bih),
                    b_hh: Some(&bhh),
                },
            )
            .unwrap();
        loss_sum = loss_sum.add(&h.sum(None).unwrap()).unwrap();
    }
    scalar(&loss_sum.to_tensor())
}

#[test]
fn gru_forward_seq_multi_step_gradient_matches_numeric_diff() {
    let (_, h_prev, w_ih_t, w_hh_t, b_ih_t, b_hh_t) = gru_fixture_data();
    let t_len = 3usize;
    let x_t = seq_input(t_len);

    let tape = Tape::new_with_ops(common::naive_ops());
    let wih = tape.var(&w_ih_t);
    let whh = tape.var(&w_hh_t);
    let bih = tape.var(&b_ih_t);
    let bhh = tape.var(&b_hh_t);
    let mut h = tape.var(&h_prev);
    let h0_var = h;
    let mut loss_sum = tape.var(&t(vec![0.0], &[]));
    for step in 0..t_len {
        let x_t_data = x_t
            .narrow(0, step, 1)
            .unwrap()
            .contiguous()
            .reshape(&[B, D])
            .unwrap();
        let x_step = tape.var(&x_t_data);
        h = x_step
            .gru_cell(
                &h,
                GateParams {
                    w_ih: &wih,
                    w_hh: &whh,
                    b_ih: Some(&bih),
                    b_hh: Some(&bhh),
                },
            )
            .unwrap();
        loss_sum = loss_sum.add(&h.sum(None).unwrap()).unwrap();
    }
    let grads = tape.backward(&loss_sum).unwrap();
    let dw_ih = grads.get(&wih).unwrap().unwrap().clone();
    let dw_hh = grads.get(&whh).unwrap().unwrap().clone();
    let dh0 = grads.get(&h0_var).unwrap().unwrap().clone();

    assert_grad_close(
        "gru seq dw_ih (T=3 fan-in sum)",
        &dw_ih,
        &numeric_grad(&w_ih_t, |v| {
            gru_manual_unrolled_loss(&x_t, &v, &w_hh_t, &b_ih_t, &b_hh_t, &h_prev, t_len)
        }),
    );
    assert_grad_close(
        "gru seq dw_hh (T=3 fan-in sum)",
        &dw_hh,
        &numeric_grad(&w_hh_t, |v| {
            gru_manual_unrolled_loss(&x_t, &w_ih_t, &v, &b_ih_t, &b_hh_t, &h_prev, t_len)
        }),
    );
    assert_grad_close(
        "gru seq dh0 (T=3)",
        &dh0,
        &numeric_grad(&h_prev, |v| {
            gru_manual_unrolled_loss(&x_t, &w_ih_t, &w_hh_t, &b_ih_t, &b_hh_t, &v, t_len)
        }),
    );
}

/// `Gru::forward_seq` が返す `RnnSeqOutput::params`（イシュー #1647
/// codex-review P1 指摘。[`rnn_forward_seq_exposes_params_for_gradient_retrieval`]
/// の GRU 版）から `Gradients::get` で直接勾配を取得できることを検証
/// する。
#[test]
fn gru_forward_seq_exposes_params_for_gradient_retrieval() {
    let (_, h_prev, w_ih_t, w_hh_t, b_ih_t, b_hh_t) = gru_fixture_data();
    let t_len = 3usize;
    let x_t = seq_input(t_len);

    let cell = GruCell::from_parameters(
        w_ih_t.clone(),
        w_hh_t.clone(),
        Some(b_ih_t.clone()),
        Some(b_hh_t.clone()),
    )
    .unwrap();
    let gru = Gru::from_cell(cell);

    let tape = Tape::new_with_ops(common::naive_ops());
    let h0 = tape.var(&h_prev);
    let out = gru.forward_seq(&tape, &x_t, Some(&h0)).unwrap();
    // `gru_manual_unrolled_loss` は各 step の `h` を loss へ加算する
    // ため、突合対象の loss も同じ構成に揃える（`h_n` のみではなく
    // 全 step 分の寄与を含める）。
    let mut loss = out.outputs[0].sum(None).unwrap();
    for h in &out.outputs[1..] {
        loss = loss.add(&h.sum(None).unwrap()).unwrap();
    }
    let grads = tape.backward(&loss).unwrap();

    let dw_ih = grads
        .get(&out.params.weight_ih)
        .unwrap()
        .expect("forward_seq が使った weight_ih の勾配が取得できるはず")
        .clone();

    assert_grad_close(
        "gru forward_seq params dw_ih",
        &dw_ih,
        &numeric_grad(&w_ih_t, |v| {
            gru_manual_unrolled_loss(&x_t, &v, &w_hh_t, &b_ih_t, &b_hh_t, &h_prev, t_len)
        }),
    );
}

// =====================================================================
// 決定 4a (i): セル単位の per-step 交互適用（多層スタック）の勾配連続性。
// =====================================================================

#[test]
fn two_layer_per_step_stacking_propagates_gradient_to_layer_one() {
    // 層 1: D -> HID、層 2: HID -> HID。層 2 の loss から層 1 の重みへ
    // 勾配が連続していることを数値微分で確認する（決定 4a (i)）。
    fn forward(x: &Tensor<f32>, w1: &Tensor<f32>, w2: &Tensor<f32>) -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        let w1v = tape.var(w1);
        let w2v = tape.var(w2);
        let dummy_w_hh1 = tape.var(&t(vec![0.0; HID * HID], &[HID, HID]));
        let dummy_w_hh2 = tape.var(&t(vec![0.0; HID * HID], &[HID, HID]));
        let h1_0 = tape.var(&t(vec![0.0; B * HID], &[B, HID]));
        let h2_0 = tape.var(&t(vec![0.0; B * HID], &[B, HID]));

        let h1 = xv
            .rnn_cell(
                &h1_0,
                GateParams {
                    w_ih: &w1v,
                    w_hh: &dummy_w_hh1,
                    b_ih: None,
                    b_hh: None,
                },
            )
            .unwrap();
        let h2 = h1
            .rnn_cell(
                &h2_0,
                GateParams {
                    w_ih: &w2v,
                    w_hh: &dummy_w_hh2,
                    b_ih: None,
                    b_hh: None,
                },
            )
            .unwrap();
        let loss = h2.sum(None).unwrap();
        scalar(&loss.to_tensor())
    }

    let x_t = t(vec![0.1, -0.2, 0.3, 0.15, -0.25, 0.05], &[B, D]);
    let w1_t = t(
        (0..D * HID).map(|i| 0.05 * (i as f32) - 0.3).collect(),
        &[D, HID],
    );
    let w2_t = t(
        (0..HID * HID).map(|i| 0.04 * (i as f32) - 0.25).collect(),
        &[HID, HID],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x_t);
    let w1v = tape.var(&w1_t);
    let w2v = tape.var(&w2_t);
    let dummy_w_hh1 = tape.var(&t(vec![0.0; HID * HID], &[HID, HID]));
    let dummy_w_hh2 = tape.var(&t(vec![0.0; HID * HID], &[HID, HID]));
    let h1_0 = tape.var(&t(vec![0.0; B * HID], &[B, HID]));
    let h2_0 = tape.var(&t(vec![0.0; B * HID], &[B, HID]));
    let h1 = xv
        .rnn_cell(
            &h1_0,
            GateParams {
                w_ih: &w1v,
                w_hh: &dummy_w_hh1,
                b_ih: None,
                b_hh: None,
            },
        )
        .unwrap();
    let h2 = h1
        .rnn_cell(
            &h2_0,
            GateParams {
                w_ih: &w2v,
                w_hh: &dummy_w_hh2,
                b_ih: None,
                b_hh: None,
            },
        )
        .unwrap();
    let loss = h2.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw1 = grads
        .get(&w1v)
        .unwrap()
        .expect("layer1 weight must receive gradient through layer2 (decision 4a (i))")
        .clone();

    assert_grad_close(
        "two-layer per-step stacking dw1",
        &dw1,
        &numeric_grad(&w1_t, |v| forward(&x_t, &v, &w2_t)),
    );
}

// =====================================================================
// (e) Tape::reset 後の葉保持（決定 3）。
// =====================================================================

#[test]
fn rnn_cell_weights_survive_tape_reset_when_bound_before_forward() {
    let cell = RnnCell::new(D, HID, true, 7).unwrap();
    let mut tape = Tape::new_with_ops(common::naive_ops());

    // 1 step 目: bind → forward（bind が葉プレフィックスを確定する）。
    {
        let vars = cell.bind(&tape);
        let x = tape.var(&t(vec![0.1; B * D], &[B, D]));
        let h0 = tape.var(&t(vec![0.0; B * HID], &[B, HID]));
        let _h1 = vars.forward(&x, &h0).unwrap();
    }
    tape.reset();

    // reset 後も weight_ih／weight_hh／bias は葉プレフィックスとして
    // 保持され、再度 forward できる（決定 3）。
    let vars2 = cell.bind(&tape);
    let x2 = tape.var(&t(vec![0.2; B * D], &[B, D]));
    let h0_2 = tape.var(&t(vec![0.0; B * HID], &[B, HID]));
    let h2 = vars2.forward(&x2, &h0_2);
    assert!(h2.is_ok(), "reset 後も再 forward できる（決定 3）");
}

// =====================================================================
// (i) Module::forward の無効化・forward_seq／forward_host の型検査。
// =====================================================================

#[test]
fn sequence_module_forward_is_invalid_argument() {
    let rnn = Rnn::new(D, HID, true, 1).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; B * D], &[B, D]));
    let err = Module::forward(&rnn, &tape, &x).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));

    let lstm = Lstm::new(D, HID, true, 1).unwrap();
    let err = Module::forward(&lstm, &tape, &x).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));

    let gru = Gru::new(D, HID, true, 1).unwrap();
    let err = Module::forward(&gru, &tape, &x).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn rnn_forward_host_matches_forward_seq_outputs_bit_exact() {
    let rnn = Rnn::new(D, HID, true, 3).unwrap();
    let t_len = 3usize;
    let x_t = seq_input(t_len);
    let ops = common::naive_ops();

    let host_out = rnn.forward_host(ops.as_ref(), &x_t).unwrap();
    assert_eq!(host_out.shape(), &[t_len, B, HID]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let seq_out = rnn.forward_seq(&tape, &x_t, None).unwrap();
    assert_eq!(seq_out.outputs.len(), t_len);

    for (step, out_var) in seq_out.outputs.iter().enumerate() {
        let tape_step = out_var.to_tensor();
        for b in 0..B {
            for h in 0..HID {
                let host_v = host_out.get(&[step, b, h]).unwrap();
                let tape_v = tape_step.get(&[b, h]).unwrap();
                assert_eq!(
                    host_v.to_bits(),
                    tape_v.to_bits(),
                    "forward_host と forward_seq は bit-exact のはず（決定 9。\
                     step={step} b={b} h={h}）"
                );
            }
        }
    }
}

// =====================================================================
// エラー経路。
// =====================================================================

#[test]
fn rnn_cell_rejects_mismatched_bias_presence() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; B * D], &[B, D]));
    let h_prev = tape.var(&t(vec![0.0; B * HID], &[B, HID]));
    let w_ih = tape.var(&t(vec![0.0; D * HID], &[D, HID]));
    let w_hh = tape.var(&t(vec![0.0; HID * HID], &[HID, HID]));
    let b_ih = tape.var(&t(vec![0.0; HID], &[HID]));

    let err = x
        .rnn_cell(
            &h_prev,
            GateParams {
                w_ih: &w_ih,
                w_hh: &w_hh,
                b_ih: Some(&b_ih),
                b_hh: None,
            },
        )
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn rnn_cell_rejects_cross_tape_operands() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let x = tape_a.var(&t(vec![0.0; B * D], &[B, D]));
    let h_prev = tape_b.var(&t(vec![0.0; B * HID], &[B, HID]));
    let w_ih = tape_a.var(&t(vec![0.0; D * HID], &[D, HID]));
    let w_hh = tape_a.var(&t(vec![0.0; HID * HID], &[HID, HID]));

    let err = x
        .rnn_cell(
            &h_prev,
            GateParams {
                w_ih: &w_ih,
                w_hh: &w_hh,
                b_ih: None,
                b_hh: None,
            },
        )
        .unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

#[test]
fn lstm_cell_new_rejects_zero_dims() {
    assert!(LstmCell::new(0, HID, true, 1).is_err());
    assert!(LstmCell::new(D, 0, true, 1).is_err());
}

/// イシュー #1647 codex-review P1 指摘の再現テスト: `hidden`（ここでは
/// `weight_hh.shape()[0]`）が `usize::MAX` 近傍かつ対応する軸の要素数が
/// 0（`weight_hh.shape() = [1usize << 62, 0]` は要素数 0 のまま合法な
/// 空テンソル）のとき、`gates * hidden`（`4 * (1usize << 62) == 2^64`）
/// が未検証のまま乗算されると `usize` 上で `0` へ wrap し、
/// `weight_ih.shape()[1] == 0`（同じく空テンソル）と一致してしまい
/// 本来 shape mismatch で拒否すべき不正な `LstmCell` を誤って受理して
/// しまう。`checked_gate_width`（`checked_mul` ベース）へ是正した後は
/// overflow を検出して `Err` を返すことを確認する。
#[test]
fn lstm_cell_from_parameters_rejects_overflowing_gate_width_instead_of_wrapping() {
    let huge_hidden: usize = 1usize << 62;
    // `4 * huge_hidden == 2^64` は `usize`（64bit）上で `0` へ wrap する。
    assert_eq!(4usize.wrapping_mul(huge_hidden), 0);

    let d = 3usize;
    let weight_ih = t(Vec::new(), &[d, 0]);
    let weight_hh = t(Vec::new(), &[huge_hidden, 0]);
    let err = LstmCell::from_parameters(weight_ih, weight_hh, None, None).unwrap_err();
    // wrap 後の値（0）と偶然一致して受理される（`Ok`）のではなく、
    // overflow 自体を検出したエラーで拒否されること。
    assert!(
        matches!(err, AutodiffError::InvalidArgument(_)),
        "overflow を checked_mul で検出した InvalidArgument を期待したが {err:?} だった"
    );
}

#[test]
fn gru_cell_from_parameters_rejects_rank_mismatch() {
    let bad_weight = t(vec![0.0; D * 3 * HID], &[D * 3 * HID]);
    let w_hh = t(vec![0.0; HID * 3 * HID], &[HID, 3 * HID]);
    let err = GruCell::from_parameters(bad_weight, w_hh, None, None).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn forward_seq_rejects_zero_length_sequence() {
    let rnn = Rnn::new(D, HID, true, 1).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let empty = t(Vec::new(), &[0, B, D]);
    let err = rnn.forward_seq(&tape, &empty, None).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}
