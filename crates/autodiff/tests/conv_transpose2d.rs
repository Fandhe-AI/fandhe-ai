//! `Var::conv_transpose2d`（col2im＋GEMM。イシュー #2067・設計
//! `docs/conv-ops-design.md` §15）の受け入れ条件検証。
//!
//! - forward: 出力 shape が PyTorch `_conv_input_size` と一致すること
//!   （stride／padding／output_padding／dilation の組合せ・`N=0` 受理）。
//! - backward: 数値微分（中央差分）との突合（input／weight／bias）。
//!   `tests/conv2d.rs` と同じ判定閾値（`H=1e-3`・相対 1e-2 または
//!   絶対 1e-3・`τ=1e-4`。新規の許容誤差緩和はしない）。
//! - shape／引数検査の境界（`stride=0`／`dilation=0`／`groups=0`・
//!   `output_padding >= stride`・チャンネル不整合・`H=0` 拒否・
//!   `N=0` 受理）。
//! - groups（depthwise 含む）を per-group `narrow`＋`groups=1` の合成
//!   との bit 一致で検証。
//! - bias broadcast の軸回帰。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{BackendError, ShapeError, Tensor};

const H: f64 = 1e-3;
const TAU: f64 = 1e-4;
const REL_TOL: f64 = 1e-2;
const ABS_TOL: f64 = 1e-3;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    tensor
        .contiguous()
        .as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_default()
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn assert_grad_close(label: &str, analytic: &[f32], numeric: &[f64]) {
    assert_eq!(analytic.len(), numeric.len(), "{label}: 要素数不一致");
    for (i, (&av, &nv)) in analytic.iter().zip(numeric.iter()).enumerate() {
        let av64 = av as f64;
        let diff = (av64 - nv).abs();
        let rel = diff / av64.abs().max(nv.abs()).max(TAU);
        assert!(
            rel <= REL_TOL || diff <= ABS_TOL,
            "{label}[{i}]: analytic={av64} numeric={nv} diff={diff} rel={rel}"
        );
    }
}

/// `L = Σ conv_transpose2d(x, w, b) · s` の中央差分による数値勾配
/// （f64 集計）。`param`（`0`=input・`1`=weight・`2`=bias）に対応する
/// 勾配のみ返す（`tests/conv2d.rs::numeric_conv2d_grad` と同型）。
#[allow(clippy::too_many_arguments)]
fn numeric_conv_transpose2d_grad(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    b: Option<&Tensor<f32>>,
    s: &Tensor<f32>,
    stride: [usize; 2],
    padding: [usize; 2],
    output_padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
    param: usize,
) -> Vec<f64> {
    let forward = |x: &Tensor<f32>, w: &Tensor<f32>, b: Option<&Tensor<f32>>| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        let wv = tape.var(w);
        let bv = b.map(|bt| tape.var(bt));
        let y = xv
            .conv_transpose2d(
                &wv,
                bv.as_ref(),
                stride,
                padding,
                output_padding,
                dilation,
                groups,
            )
            .unwrap();
        let out = y.to_tensor();
        dense(&out)
            .iter()
            .zip(dense(s).iter())
            .map(|(&yv, &sv)| yv as f64 * sv as f64)
            .sum()
    };
    let target: &Tensor<f32> = match param {
        0 => x,
        1 => w,
        2 => b.expect("param=2 requires bias"),
        _ => unreachable!(),
    };
    let shape = target.shape().to_vec();
    let mut data = dense(target);
    let mut grad = vec![0f64; data.len()];
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let modified = t(data.clone(), &shape);
        let lp = match param {
            0 => forward(&modified, w, b),
            1 => forward(x, &modified, b),
            2 => forward(x, w, Some(&modified)),
            _ => unreachable!(),
        };
        data[i] = (orig - H) as f32;
        let modified = t(data.clone(), &shape);
        let lm = match param {
            0 => forward(&modified, w, b),
            1 => forward(x, &modified, b),
            2 => forward(x, w, Some(&modified)),
            _ => unreachable!(),
        };
        data[i] = orig as f32;
        grad[i] = (lp - lm) / (2.0 * H);
    }
    grad
}

// --- 1. forward: 出力 shape 表 ---

type ConvTransposeShapeCase = (
    [usize; 4],
    [usize; 4],
    [usize; 2],
    [usize; 2],
    [usize; 2],
    [usize; 2],
    usize,
    [usize; 4],
);

#[test]
fn forward_output_shape_matches_pytorch_conv_input_size() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let cases: [ConvTransposeShapeCase; 4] = [
        // (input, weight, stride, padding, output_padding, dilation, groups, expected out)
        (
            [1, 4, 6, 6],
            [4, 3, 3, 3],
            [1, 1],
            [0, 0],
            [0, 0],
            [1, 1],
            1,
            [1, 3, 8, 8],
        ),
        (
            [1, 2, 4, 4],
            [2, 2, 3, 3],
            [2, 2],
            [1, 1],
            [1, 1],
            [1, 1],
            1,
            [1, 2, 8, 8],
        ),
        (
            [2, 6, 5, 5],
            [6, 2, 3, 3],
            [1, 1],
            [1, 1],
            [0, 0],
            [1, 1],
            2,
            [2, 4, 5, 5],
        ),
        (
            [1, 1, 4, 4],
            [1, 1, 3, 3],
            [1, 1],
            [0, 0],
            [0, 0],
            [2, 2],
            1,
            [1, 1, 8, 8],
        ),
    ];
    for (in_shape, w_shape, stride, padding, output_padding, dilation, groups, expected) in cases {
        let x = tape.var(&t(vec![0.1; in_shape.iter().product()], &in_shape));
        let w = tape.var(&t(vec![0.1; w_shape.iter().product()], &w_shape));
        let y = x
            .conv_transpose2d(&w, None, stride, padding, output_padding, dilation, groups)
            .unwrap();
        assert_eq!(
            y.to_tensor().shape().to_vec(),
            expected.to_vec(),
            "in={in_shape:?} w={w_shape:?}"
        );
    }
}

#[test]
fn forward_n_zero_is_accepted_and_produces_empty_output() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(Vec::new(), &[0, 4, 6, 6]));
    let w = tape.var(&t(vec![0.1; 4 * 3 * 3 * 3], &[4, 3, 3, 3]));
    let y = x
        .conv_transpose2d(&w, None, [1, 1], [0, 0], [0, 0], [1, 1], 1)
        .unwrap();
    assert_eq!(y.to_tensor().shape().to_vec(), vec![0, 3, 8, 8]);
}

// --- 2. backward: 数値微分突合 ---

#[test]
fn backward_matches_numeric_grad_basic() {
    let x = t(
        (0..numel(&[1, 2, 4, 4]))
            .map(|i| (i as f32 * 0.037).sin())
            .collect(),
        &[1, 2, 4, 4],
    );
    let w = t(
        (0..2 * 3 * 3 * 3)
            .map(|i| (i as f32 * 0.091).cos() * 0.5)
            .collect(),
        &[2, 3, 3, 3],
    );
    let b = t(vec![0.1, -0.2, 0.3], &[3]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let bv = tape.var(&b);
    let y = xv
        .conv_transpose2d(&wv, Some(&bv), [1, 1], [1, 1], [0, 0], [1, 1], 1)
        .unwrap();
    let out_shape = y.to_tensor().shape().to_vec();
    let s = t(
        (0..out_shape.iter().product::<usize>())
            .map(|i| ((i as f32) * 0.053).sin() * 0.7)
            .collect(),
        &out_shape,
    );

    let loss = y.mul(&tape.var(&s)).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("xv reaches loss");
    let dw = grads.get(&wv).unwrap().expect("wv reaches loss");
    let db = grads.get(&bv).unwrap().expect("bv reaches loss");

    let num_dx =
        numeric_conv_transpose2d_grad(&x, &w, Some(&b), &s, [1, 1], [1, 1], [0, 0], [1, 1], 1, 0);
    let num_dw =
        numeric_conv_transpose2d_grad(&x, &w, Some(&b), &s, [1, 1], [1, 1], [0, 0], [1, 1], 1, 1);
    let num_db =
        numeric_conv_transpose2d_grad(&x, &w, Some(&b), &s, [1, 1], [1, 1], [0, 0], [1, 1], 1, 2);

    assert_grad_close("dx", &dense(dx), &num_dx);
    assert_grad_close("dw", &dense(dw), &num_dw);
    assert_grad_close("db", &dense(db), &num_db);
}

#[test]
fn backward_matches_numeric_grad_groups_dilation_no_bias() {
    let x = t(
        (0..numel(&[1, 4, 4, 4]))
            .map(|i| (i as f32 * 0.029).sin())
            .collect(),
        &[1, 4, 4, 4],
    );
    let w = t(
        (0..4 * 2 * 2 * 2)
            .map(|i| (i as f32 * 0.071).cos() * 0.4)
            .collect(),
        &[4, 2, 2, 2],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let y = xv
        .conv_transpose2d(&wv, None, [1, 1], [1, 1], [0, 0], [2, 2], 2)
        .unwrap();
    let out_shape = y.to_tensor().shape().to_vec();
    let s = t(
        (0..out_shape.iter().product::<usize>())
            .map(|i| ((i as f32) * 0.061).cos() * 0.6)
            .collect(),
        &out_shape,
    );

    let loss = y.mul(&tape.var(&s)).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("xv reaches loss");
    let dw = grads.get(&wv).unwrap().expect("wv reaches loss");

    let num_dx =
        numeric_conv_transpose2d_grad(&x, &w, None, &s, [1, 1], [1, 1], [0, 0], [2, 2], 2, 0);
    let num_dw =
        numeric_conv_transpose2d_grad(&x, &w, None, &s, [1, 1], [1, 1], [0, 0], [2, 2], 2, 1);

    assert_grad_close("dx(groups,dilation)", &dense(dx), &num_dx);
    assert_grad_close("dw(groups,dilation)", &dense(dw), &num_dw);
}

#[test]
fn backward_matches_numeric_grad_output_padding() {
    // output_padding=1 は `op < stride` ゲートを通すため stride >= 2
    // の形状で検証する（設計「検証方法」§6.1）。
    let x = t(
        (0..numel(&[1, 2, 3, 3]))
            .map(|i| (i as f32 * 0.043).sin())
            .collect(),
        &[1, 2, 3, 3],
    );
    let w = t(
        (0..2 * 2 * 3 * 3)
            .map(|i| (i as f32 * 0.067).cos() * 0.4)
            .collect(),
        &[2, 2, 3, 3],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let y = xv
        .conv_transpose2d(&wv, None, [2, 2], [1, 1], [1, 1], [1, 1], 1)
        .unwrap();
    let out_shape = y.to_tensor().shape().to_vec();
    let s = t(
        (0..out_shape.iter().product::<usize>())
            .map(|i| ((i as f32) * 0.089).sin() * 0.5)
            .collect(),
        &out_shape,
    );

    let loss = y.mul(&tape.var(&s)).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("xv reaches loss");
    let dw = grads.get(&wv).unwrap().expect("wv reaches loss");

    let num_dx =
        numeric_conv_transpose2d_grad(&x, &w, None, &s, [2, 2], [1, 1], [1, 1], [1, 1], 1, 0);
    let num_dw =
        numeric_conv_transpose2d_grad(&x, &w, None, &s, [2, 2], [1, 1], [1, 1], [1, 1], 1, 1);

    assert_grad_close("dx(output_padding)", &dense(dx), &num_dx);
    assert_grad_close("dw(output_padding)", &dense(dw), &num_dw);
}

// --- 3. shape／引数検査の境界 ---

#[test]
fn rejects_zero_stride() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4, 4])], &[1, 1, 4, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 3, 3])], &[1, 1, 3, 3]));
    let err = x
        .conv_transpose2d(&w, None, [0, 1], [0, 0], [0, 0], [1, 1], 1)
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn rejects_zero_dilation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4, 4])], &[1, 1, 4, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 3, 3])], &[1, 1, 3, 3]));
    let err = x
        .conv_transpose2d(&w, None, [1, 1], [0, 0], [0, 0], [1, 0], 1)
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn rejects_zero_groups() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4, 4])], &[1, 1, 4, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 3, 3])], &[1, 1, 3, 3]));
    let err = x
        .conv_transpose2d(&w, None, [1, 1], [0, 0], [0, 0], [1, 1], 0)
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn rejects_output_padding_ge_stride() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4, 4])], &[1, 1, 4, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 3, 3])], &[1, 1, 3, 3]));
    let err = x
        .conv_transpose2d(&w, None, [1, 1], [0, 0], [1, 0], [1, 1], 1)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn rejects_cin_not_divisible_by_groups() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 3, 8, 8])], &[1, 3, 8, 8]));
    let w = tape.var(&t(vec![0.0; 3 * 2 * 3 * 3], &[3, 2, 3, 3]));
    let err = x
        .conv_transpose2d(&w, None, [1, 1], [0, 0], [0, 0], [1, 1], 2)
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn rejects_weight_cin_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 4, 8, 8])], &[1, 4, 8, 8]));
    let w = tape.var(&t(vec![0.0; 3 * 3 * 3 * 3], &[3, 3, 3, 3]));
    let err = x
        .conv_transpose2d(&w, None, [1, 1], [0, 0], [0, 0], [1, 1], 1)
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn rejects_bias_shape_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4, 4])], &[1, 1, 4, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 2, 3, 3])], &[1, 2, 3, 3]));
    let b = tape.var(&t(vec![0.0; 3], &[3]));
    let err = x
        .conv_transpose2d(&w, Some(&b), [1, 1], [0, 0], [0, 0], [1, 1], 1)
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn rejects_h_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x0 = tape.var(&t(Vec::new(), &[1, 1, 0, 4]));
    let w0 = tape.var(&t(vec![0.0; numel(&[1, 1, 1, 1])], &[1, 1, 1, 1]));
    let err = x0
        .conv_transpose2d(&w0, None, [1, 1], [0, 0], [0, 0], [1, 1], 1)
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

// --- 4. groups: per-group narrow + groups=1 合成との bit 一致 ---

#[test]
fn groups_equivalent_to_per_group_narrow_composition_depthwise() {
    let cin = 4usize;
    let groups = 4usize; // depthwise
    let cin_g = cin / groups;
    let cout = 4usize;
    let cout_g = cout / groups;
    let x = t(
        (0..numel(&[1, cin, 5, 5]))
            .map(|i| (i as f32 * 0.017).sin())
            .collect(),
        &[1, cin, 5, 5],
    );
    let w = t(
        (0..cin * cout_g * 3 * 3)
            .map(|i| (i as f32 * 0.023).cos() * 0.3)
            .collect(),
        &[cin, cout_g, 3, 3],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let y_grouped = xv
        .conv_transpose2d(&wv, None, [1, 1], [1, 1], [0, 0], [1, 1], groups)
        .unwrap();
    let out_grouped = dense(&y_grouped.to_tensor());

    // per-group narrow + groups=1 合成。weight は `[Cin, Cout_g, kH,
    // kW]` のため narrow の対象軸は `Conv2d`（`[Cout, Cin_g, kH,
    // kW]`）と逆で axis 0（Cin）になる。
    let mut per_group_outputs: Vec<Vec<f32>> = Vec::new();
    let mut out_shape_single = Vec::new();
    for g in 0..groups {
        let x_g = xv.narrow(1, g * cin_g, cin_g).unwrap();
        let w_g = wv.narrow(0, g * cin_g, cin_g).unwrap();
        let y_g = x_g
            .conv_transpose2d(&w_g, None, [1, 1], [1, 1], [0, 0], [1, 1], 1)
            .unwrap();
        out_shape_single = y_g.to_tensor().shape().to_vec();
        per_group_outputs.push(dense(&y_g.to_tensor()));
    }
    let (_n, _cout_g_single, hout, wout) = (
        out_shape_single[0],
        out_shape_single[1],
        out_shape_single[2],
        out_shape_single[3],
    );
    let mut combined = vec![0f32; cout * hout * wout];
    for (g, group_output) in per_group_outputs.iter().enumerate() {
        for co_g in 0..cout_g {
            let co = g * cout_g + co_g;
            for p in 0..hout * wout {
                combined[co * hout * wout + p] = group_output[co_g * hout * wout + p];
            }
        }
    }
    assert_eq!(out_grouped, combined);
}

// --- 5. bias broadcast の軸回帰 ---

#[test]
fn bias_adds_to_cout_axis_not_wout_axis_when_equal() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0; numel(&[1, 1, 5, 5])], &[1, 1, 5, 5]));
    let w = tape.var(&t(vec![1.0; numel(&[1, 3, 1, 1])], &[1, 3, 1, 1]));
    let b = tape.var(&t(vec![10.0, 20.0, 30.0], &[3]));
    let y = x
        .conv_transpose2d(&w, Some(&b), [1, 1], [0, 0], [0, 0], [1, 1], 1)
        .unwrap();
    assert_eq!(y.to_tensor().shape().to_vec(), vec![1, 3, 5, 5]);
    let data = dense(&y.to_tensor());
    // kernel=1x1・weight=1 なので各 (co, oh, ow) = 1.0 + bias[co]。
    for co in 0..3 {
        let expected_bias = [10.0, 20.0, 30.0][co];
        for p in 0..25 {
            let v = data[co * 25 + p];
            assert_eq!(v, 1.0 + expected_bias, "co={co} p={p}");
        }
    }
}
