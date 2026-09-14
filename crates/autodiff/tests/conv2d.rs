//! `Var::conv2d`（im2col＋GEMM。イシュー #1764・設計 `docs/conv-ops-
//! design.md`）の受け入れ条件検証。
//!
//! - forward: 出力 shape が PyTorch `_conv_output_size` と一致すること
//!   （groups／dilation／padding の組合せ）。
//! - backward: 数値微分（中央差分）との突合（input／weight／bias）。
//!   `tests/nn_linear.rs`／`tests/backward.rs` と同じ判定閾値
//!   （`H=1e-3`・相対 1e-2 または絶対 1e-3・`τ=1e-4`。新規の許容誤差
//!   緩和はしない）。
//! - shape／引数検査の境界（`stride=0`／`dilation=0`／`groups=0`・
//!   チャンネル不整合・負分子拒否ゲート・`H=0`／`W=0` 拒否・`N=0` 受理・
//!   `padding > k/2` 受理）。
//! - groups（depthwise 含む）を per-group `narrow`＋`groups=1` の合成
//!   との bit 一致で検証。
//! - bias broadcast の軸回帰（`Cout == Wout` の形状で `Cout` 軸に
//!   正しく加算されることの確認。設計 doc §5.2「実装上の注意」）。

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

/// `L = Σ conv2d(x, w, b) · s` の中央差分による数値勾配（f64 集計）。
/// `param`（`0`=input・`1`=weight・`2`=bias）に対応する勾配のみ返す。
#[allow(clippy::too_many_arguments)]
fn numeric_conv2d_grad(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    b: Option<&Tensor<f32>>,
    s: &Tensor<f32>,
    stride: [usize; 2],
    padding: [usize; 2],
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
            .conv2d(&wv, bv.as_ref(), stride, padding, dilation, groups)
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

#[test]
fn forward_output_shape_matches_pytorch_conv_output_size() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let cases: [(
        [usize; 4],
        [usize; 4],
        [usize; 2],
        [usize; 2],
        [usize; 2],
        usize,
        [usize; 4],
    ); 4] = [
        // (input, weight, stride, padding, dilation, groups, expected out)
        (
            [1, 3, 8, 8],
            [4, 3, 3, 3],
            [1, 1],
            [0, 0],
            [1, 1],
            1,
            [1, 4, 6, 6],
        ),
        (
            [2, 4, 5, 5],
            [6, 2, 3, 3],
            [1, 1],
            [1, 1],
            [1, 1],
            2,
            [2, 6, 5, 5],
        ),
        (
            [1, 4, 5, 5],
            [4, 1, 3, 3],
            [1, 1],
            [1, 1],
            [1, 1],
            4,
            [1, 4, 5, 5],
        ),
        (
            [1, 1, 7, 7],
            [1, 1, 3, 3],
            [2, 2],
            [0, 0],
            [2, 2],
            1,
            [1, 1, 2, 2],
        ),
    ];
    for (in_shape, w_shape, stride, padding, dilation, groups, expected) in cases {
        let x = tape.var(&t(vec![0.1; in_shape.iter().product()], &in_shape));
        let w = tape.var(&t(vec![0.1; w_shape.iter().product()], &w_shape));
        let y = x
            .conv2d(&w, None, stride, padding, dilation, groups)
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
    let x = tape.var(&t(Vec::new(), &[0, 3, 8, 8]));
    let w = tape.var(&t(vec![0.1; 4 * 3 * 3 * 3], &[4, 3, 3, 3]));
    let y = x.conv2d(&w, None, [1, 1], [0, 0], [1, 1], 1).unwrap();
    assert_eq!(y.to_tensor().shape().to_vec(), vec![0, 4, 6, 6]);
}

// --- 2. backward: 数値微分突合 ---

#[test]
fn backward_matches_numeric_grad_basic() {
    let x = t(
        (0..1 * 2 * 5 * 5)
            .map(|i| (i as f32 * 0.037).sin())
            .collect(),
        &[1, 2, 5, 5],
    );
    let w = t(
        (0..3 * 2 * 3 * 3)
            .map(|i| (i as f32 * 0.091).cos() * 0.5)
            .collect(),
        &[3, 2, 3, 3],
    );
    let b = t(vec![0.1, -0.2, 0.3], &[3]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let bv = tape.var(&b);
    let y = xv
        .conv2d(&wv, Some(&bv), [1, 1], [1, 1], [1, 1], 1)
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

    let num_dx = numeric_conv2d_grad(&x, &w, Some(&b), &s, [1, 1], [1, 1], [1, 1], 1, 0);
    let num_dw = numeric_conv2d_grad(&x, &w, Some(&b), &s, [1, 1], [1, 1], [1, 1], 1, 1);
    let num_db = numeric_conv2d_grad(&x, &w, Some(&b), &s, [1, 1], [1, 1], [1, 1], 1, 2);

    assert_grad_close("dx", &dense(dx), &num_dx);
    assert_grad_close("dw", &dense(dw), &num_dw);
    assert_grad_close("db", &dense(db), &num_db);
}

#[test]
fn backward_matches_numeric_grad_groups_dilation_no_bias() {
    let x = t(
        (0..1 * 4 * 6 * 6)
            .map(|i| (i as f32 * 0.029).sin())
            .collect(),
        &[1, 4, 6, 6],
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
    let y = xv.conv2d(&wv, None, [1, 1], [1, 1], [2, 2], 2).unwrap();
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

    let num_dx = numeric_conv2d_grad(&x, &w, None, &s, [1, 1], [1, 1], [2, 2], 2, 0);
    let num_dw = numeric_conv2d_grad(&x, &w, None, &s, [1, 1], [1, 1], [2, 2], 2, 1);

    assert_grad_close("dx(groups,dilation)", &dense(dx), &num_dx);
    assert_grad_close("dw(groups,dilation)", &dense(dw), &num_dw);
}

#[test]
fn backward_matches_numeric_grad_overlapping_windows() {
    // stride(1) < 有効カーネル幅(3) の重なり窓ケース。
    let x = t(
        (0..1 * 1 * 6 * 6)
            .map(|i| (i as f32 * 0.083).sin())
            .collect(),
        &[1, 1, 6, 6],
    );
    let w = t(
        vec![0.2, -0.1, 0.3, 0.05, -0.2, 0.4, 0.1, -0.05, 0.25],
        &[1, 1, 3, 3],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let y = xv.conv2d(&wv, None, [1, 1], [0, 0], [1, 1], 1).unwrap();
    let out_shape = y.to_tensor().shape().to_vec();
    let s = t(vec![1.0; out_shape.iter().product()], &out_shape);

    let loss = y.mul(&tape.var(&s)).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("xv reaches loss");

    let num_dx = numeric_conv2d_grad(&x, &w, None, &s, [1, 1], [0, 0], [1, 1], 1, 0);
    assert_grad_close("dx(overlap)", &dense(dx), &num_dx);
}

// --- 3. shape／引数検査の境界 ---

#[test]
fn rejects_zero_stride() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 1 * 1 * 4 * 4], &[1, 1, 4, 4]));
    let w = tape.var(&t(vec![0.0; 1 * 1 * 3 * 3], &[1, 1, 3, 3]));
    let err = x.conv2d(&w, None, [0, 1], [0, 0], [1, 1], 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn rejects_zero_dilation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 1 * 1 * 4 * 4], &[1, 1, 4, 4]));
    let w = tape.var(&t(vec![0.0; 1 * 1 * 3 * 3], &[1, 1, 3, 3]));
    let err = x.conv2d(&w, None, [1, 1], [0, 0], [1, 0], 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn rejects_zero_groups() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 1 * 1 * 4 * 4], &[1, 1, 4, 4]));
    let w = tape.var(&t(vec![0.0; 1 * 1 * 3 * 3], &[1, 1, 3, 3]));
    let err = x.conv2d(&w, None, [1, 1], [0, 0], [1, 1], 0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn rejects_cin_not_divisible_by_groups() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 1 * 3 * 8 * 8], &[1, 3, 8, 8]));
    let w = tape.var(&t(vec![0.0; 4 * 2 * 3 * 3], &[4, 2, 3, 3]));
    let err = x.conv2d(&w, None, [1, 1], [0, 0], [1, 1], 2).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn rejects_weight_cin_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 1 * 4 * 8 * 8], &[1, 4, 8, 8]));
    let w = tape.var(&t(vec![0.0; 4 * 3 * 3 * 3], &[4, 3, 3, 3]));
    let err = x.conv2d(&w, None, [1, 1], [0, 0], [1, 1], 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn rejects_bias_shape_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 1 * 1 * 4 * 4], &[1, 1, 4, 4]));
    let w = tape.var(&t(vec![0.0; 2 * 1 * 3 * 3], &[2, 1, 3, 3]));
    let b = tape.var(&t(vec![0.0; 3], &[3]));
    let err = x
        .conv2d(&w, Some(&b), [1, 1], [0, 0], [1, 1], 1)
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn rejects_negative_numerator_kernel_too_large() {
    // in=1, k=3, p=0, d=1 -> 分子 1 - 2 - 1 = -2 < 0（負分子拒否ゲート）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 1 * 1 * 1 * 1], &[1, 1, 1, 1]));
    let w = tape.var(&t(vec![0.0; 1 * 1 * 3 * 3], &[1, 1, 3, 3]));
    let err = x.conv2d(&w, None, [1, 1], [0, 0], [1, 1], 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn rejects_h_zero_but_accepts_padding_only_window() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // H=0 は拒否
    let x0 = tape.var(&t(Vec::new(), &[1, 1, 0, 4]));
    let w0 = tape.var(&t(vec![0.0; 1 * 1 * 1 * 1], &[1, 1, 1, 1]));
    let err = x0.conv2d(&w0, None, [1, 1], [0, 0], [1, 1], 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));

    // H=W=1, kernel=1, padding=1 は「有効入力を含まない窓」を許容
    // （設計 doc §3「訂正」）。
    let x1 = tape.var(&t(vec![2.0], &[1, 1, 1, 1]));
    let w1 = tape.var(&t(vec![3.0], &[1, 1, 1, 1]));
    let y = x1.conv2d(&w1, None, [1, 1], [1, 1], [1, 1], 1).unwrap();
    assert_eq!(y.to_tensor().shape().to_vec(), vec![1, 1, 3, 3]);
    let data = dense(&y.to_tensor());
    // 中心だけが有効入力（2*3=6）、他は padding(0)*3=0。
    assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn accepts_padding_greater_than_half_kernel() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0; 1 * 1 * 4 * 4], &[1, 1, 4, 4]));
    let w = tape.var(&t(vec![1.0; 1 * 1 * 3 * 3], &[1, 1, 3, 3]));
    // pooling は padding <= floor(k/2) の上限を持つが Conv は持たない
    // （設計 doc §0.2／§3 の意図的な差異）。
    let y = x.conv2d(&w, None, [1, 1], [5, 5], [1, 1], 1).unwrap();
    assert_eq!(y.to_tensor().shape().to_vec(), vec![1, 1, 12, 12]);
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
        (0..1 * cin * 5 * 5)
            .map(|i| (i as f32 * 0.017).sin())
            .collect(),
        &[1, cin, 5, 5],
    );
    let w = t(
        (0..cout * cin_g * 3 * 3)
            .map(|i| (i as f32 * 0.023).cos() * 0.3)
            .collect(),
        &[cout, cin_g, 3, 3],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let y_grouped = xv
        .conv2d(&wv, None, [1, 1], [1, 1], [1, 1], groups)
        .unwrap();
    let out_grouped = dense(&y_grouped.to_tensor());

    // per-group narrow + groups=1 合成
    let mut per_group_outputs: Vec<Vec<f32>> = Vec::new();
    let mut out_shape_single = Vec::new();
    for g in 0..groups {
        let x_g = xv.narrow(1, g * cin_g, cin_g).unwrap();
        let w_g = wv.narrow(0, g * cout_g, cout_g).unwrap();
        let y_g = x_g.conv2d(&w_g, None, [1, 1], [1, 1], [1, 1], 1).unwrap();
        out_shape_single = y_g.to_tensor().shape().to_vec();
        per_group_outputs.push(dense(&y_g.to_tensor()));
    }
    // 出力を [N, Cout, Hout, Wout] の Cout 軸で結合して比較。
    let (n, _cout_g_single, hout, wout) = (
        out_shape_single[0],
        out_shape_single[1],
        out_shape_single[2],
        out_shape_single[3],
    );
    let mut combined = vec![0f32; n * cout * hout * wout];
    for g in 0..groups {
        for co_g in 0..cout_g {
            let co = g * cout_g + co_g;
            for p in 0..hout * wout {
                combined[co * hout * wout + p] = per_group_outputs[g][co_g * hout * wout + p];
            }
        }
    }
    assert_eq!(out_grouped, combined);
}

// --- 5. bias broadcast の軸回帰（設計 doc §5.2） ---

#[test]
fn bias_adds_to_cout_axis_not_wout_axis_when_equal() {
    // Cout == Wout となる形状（Cout=3, Wout=3）で、bias が誤って Wout
    // 軸へ加算されないことを確認する。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0; 1 * 1 * 5 * 5], &[1, 1, 5, 5]));
    let w = tape.var(&t(vec![1.0; 3 * 1 * 3 * 3], &[3, 1, 3, 3]));
    let b = tape.var(&t(vec![10.0, 20.0, 30.0], &[3]));
    let y = x.conv2d(&w, Some(&b), [1, 1], [0, 0], [1, 1], 1).unwrap();
    assert_eq!(y.to_tensor().shape().to_vec(), vec![1, 3, 3, 3]);
    let data = dense(&y.to_tensor());
    // 各 (co, oh, ow) は sum(1*1 over 3x3 window)=9 + bias[co]。
    for co in 0..3 {
        let expected_bias = [10.0, 20.0, 30.0][co];
        for p in 0..9 {
            let v = data[co * 9 + p];
            assert_eq!(v, 9.0 + expected_bias, "co={co} p={p}");
        }
    }
}
