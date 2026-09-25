//! `conv3d_ops::conv3d`（im2col3d＋GEMM の空間 3 軸一般化。イシュー
//! #2158・設計 `docs/conv-ops-design.md` §16）の受け入れ条件検証。
//!
//! - forward: 出力 shape が [`conv3d_out_shape`] と一致すること
//!   （stride／padding／dilation／groups の組合せ・`N=0` 受理）。
//! - forward: `direct_conv3d_f64`（f64 7 重ループの直接畳み込み。
//!   im2col3d／GEMM／col2im3d を一切使わない独立参照実装）との REQ-2
//!   複合判定突合（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。
//! - backward: 数値微分（中央差分）との突合（input／weight／bias）。
//!   `tests/conv_transpose2d.rs` と同じ判定閾値（`H=1e-3`・相対 1e-2
//!   または絶対 1e-3・`τ=1e-4`。新規の許容誤差緩和はしない）。
//! - shape／引数検査の境界（`stride=0`／`dilation=0`／`groups=0`・
//!   チャンネル不整合・`D=0` 拒否・`N=0` 受理・weight rank 不一致）。
//! - kD=1・D=1 の Conv3d が reshape 経由の Conv2d と bit 一致する
//!   （`K_g`・col 行列・GEMM の呼び出しが同じになるため）。
//! - bias broadcast の軸回帰（`Cout == Wout` の形状でも W 軸へ誤って
//!   足さないこと）。

mod common;

use fandhe_ai_autodiff::conv3d_ops::conv3d;
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{BackendError, Conv3dParams, ShapeError, Tensor, conv3d_out_shape};

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

/// REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
/// `.claude/rules/coding-rust.md`）。
fn assert_parity_f32(label: &str, actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{label}: 要素数不一致");
    for (i, (&av, &ev)) in actual.iter().zip(expected.iter()).enumerate() {
        let av64 = av as f64;
        let diff = (av64 - ev).abs();
        let rel = diff / ev.abs().max(1e-30);
        assert!(
            rel < 1e-3 || diff < 1e-5,
            "{label}[{i}]: actual={av64} expected={ev} diff={diff} rel={rel}"
        );
    }
}

/// im2col3d／GEMM／col2im3d を一切使わない独立参照実装（`(c, kd, kh,
/// kw)` 昇順 `f64` 積和。設計「検証方法」§5「bit 一致は求めず REQ-2
/// 判定にだけ使う」の対象——BLIS の K 軸縮約順と異なるため）。NCDHW
/// cross-correlation。
#[allow(clippy::too_many_arguments)]
fn direct_conv3d_f64(
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    params: &Conv3dParams,
    out_shape: &[usize],
) -> Vec<f64> {
    let in_shape = input.shape();
    let (cin, d_in, h_in, w_in) = (in_shape[1], in_shape[2], in_shape[3], in_shape[4]);
    let (n_batch, cout, d_out, h_out, w_out) = (
        out_shape[0],
        out_shape[1],
        out_shape[2],
        out_shape[3],
        out_shape[4],
    );
    let groups = params.groups();
    let cin_g = cin / groups.max(1);
    let cout_g = cout / groups.max(1);
    let [kd_k, kh_k, kw_k] = params.kernel_size();
    let [sd, sh, sw] = params.stride();
    let [pd, ph, pw] = params.padding();
    let [dd, dh, dw] = params.dilation();
    let x = dense(input);
    let w = dense(weight);
    let bias_data = bias.map(dense);

    let mut out = vec![0f64; numel(out_shape)];
    for n in 0..n_batch {
        for co in 0..cout {
            let g = co / cout_g.max(1);
            for od in 0..d_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = 0f64;
                        for c_g in 0..cin_g {
                            let c = g * cin_g + c_g;
                            for kd_ in 0..kd_k {
                                let d_num = (od * sd + kd_ * dd) as i64 - pd as i64;
                                if d_num < 0 || d_num as usize >= d_in {
                                    continue;
                                }
                                let d = d_num as usize;
                                for kh_ in 0..kh_k {
                                    let h_num = (oh * sh + kh_ * dh) as i64 - ph as i64;
                                    if h_num < 0 || h_num as usize >= h_in {
                                        continue;
                                    }
                                    let h = h_num as usize;
                                    for kw_ in 0..kw_k {
                                        let w_num = (ow * sw + kw_ * dw) as i64 - pw as i64;
                                        if w_num < 0 || w_num as usize >= w_in {
                                            continue;
                                        }
                                        let ww = w_num as usize;
                                        let x_idx =
                                            (((n * cin + c) * d_in + d) * h_in + h) * w_in + ww;
                                        let w_idx =
                                            (((co * cin_g + c_g) * kd_k + kd_) * kh_k + kh_) * kw_k
                                                + kw_;
                                        acc += f64::from(x[x_idx]) * f64::from(w[w_idx]);
                                    }
                                }
                            }
                        }
                        if let Some(ref bd) = bias_data {
                            acc += f64::from(bd[co]);
                        }
                        let out_idx = (((n * cout + co) * d_out + od) * h_out + oh) * w_out + ow;
                        out[out_idx] = acc;
                    }
                }
            }
        }
    }
    out
}

/// `L = Σ conv3d(x, w, b) · s` の中央差分による数値勾配（f64 集計。
/// `tests/conv_transpose2d.rs::numeric_conv_transpose2d_grad` と同型）。
#[allow(clippy::too_many_arguments)]
fn numeric_conv3d_grad(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    b: Option<&Tensor<f32>>,
    s: &Tensor<f32>,
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
    param: usize,
) -> Vec<f64> {
    let forward = |x: &Tensor<f32>, w: &Tensor<f32>, b: Option<&Tensor<f32>>| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        let wv = tape.var(w);
        let bv = b.map(|bt| tape.var(bt));
        let y = conv3d(&xv, &wv, bv.as_ref(), stride, padding, dilation, groups).unwrap();
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

// --- 1. forward: 出力 shape・REQ-2 突合 ---

#[test]
fn forward_output_shape_and_direct_reference_match() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let in_shape = [1usize, 2, 4, 4, 4];
    let w_shape = [3usize, 2, 2, 2, 2];
    let x = t(
        (0..numel(&in_shape))
            .map(|i| (i as f32 * 0.031).sin())
            .collect(),
        &in_shape,
    );
    let w = t(
        (0..numel(&w_shape))
            .map(|i| (i as f32 * 0.057).cos() * 0.4)
            .collect(),
        &w_shape,
    );
    let b = t(vec![0.1, -0.2, 0.3], &[3]);
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let bv = tape.var(&b);
    let y = conv3d(&xv, &wv, Some(&bv), [1, 1, 1], [1, 1, 1], [1, 1, 1], 1).unwrap();

    let params = Conv3dParams::new([2, 2, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1], 1).unwrap();
    let out_shape = conv3d_out_shape(&in_shape, &w_shape, &params).unwrap();
    assert_eq!(y.to_tensor().shape().to_vec(), out_shape);

    let expected = direct_conv3d_f64(&x, &w, Some(&b), &params, &out_shape);
    assert_parity_f32("forward", &dense(&y.to_tensor()), &expected);
}

#[test]
fn forward_n_zero_is_accepted_and_produces_empty_output() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(Vec::new(), &[0, 1, 2, 2, 2]));
    let w = tape.var(&t(vec![0.1; 8], &[1, 1, 2, 2, 2]));
    let y = conv3d(&x, &w, None, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1).unwrap();
    assert_eq!(y.to_tensor().shape().to_vec(), vec![0, 1, 1, 1, 1]);
}

// --- 2. backward: 数値微分突合 ---

#[test]
fn backward_matches_numeric_grad_basic() {
    let x = t(
        (0..numel(&[1, 2, 3, 3, 3]))
            .map(|i| (i as f32 * 0.037).sin())
            .collect(),
        &[1, 2, 3, 3, 3],
    );
    let w = t(
        (0..3 * 2 * 2 * 2 * 2)
            .map(|i| (i as f32 * 0.091).cos() * 0.5)
            .collect(),
        &[3, 2, 2, 2, 2],
    );
    let b = t(vec![0.1, -0.2, 0.3], &[3]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let bv = tape.var(&b);
    let y = conv3d(&xv, &wv, Some(&bv), [1, 1, 1], [0, 0, 0], [1, 1, 1], 1).unwrap();
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

    let num_dx = numeric_conv3d_grad(&x, &w, Some(&b), &s, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1, 0);
    let num_dw = numeric_conv3d_grad(&x, &w, Some(&b), &s, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1, 1);
    let num_db = numeric_conv3d_grad(&x, &w, Some(&b), &s, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1, 2);

    assert_grad_close("dx", &dense(dx), &num_dx);
    assert_grad_close("dw", &dense(dw), &num_dw);
    assert_grad_close("db", &dense(db), &num_db);
}

#[test]
fn backward_matches_numeric_grad_groups_dilation_padding_no_bias() {
    let x = t(
        (0..numel(&[1, 4, 4, 4, 4]))
            .map(|i| (i as f32 * 0.029).sin())
            .collect(),
        &[1, 4, 4, 4, 4],
    );
    let w = t(
        (0..4 * 2 * 2 * 2 * 2)
            .map(|i| (i as f32 * 0.071).cos() * 0.4)
            .collect(),
        &[4, 2, 2, 2, 2],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let y = conv3d(&xv, &wv, None, [1, 1, 1], [1, 1, 1], [1, 1, 1], 2).unwrap();
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

    let num_dx = numeric_conv3d_grad(&x, &w, None, &s, [1, 1, 1], [1, 1, 1], [1, 1, 1], 2, 0);
    let num_dw = numeric_conv3d_grad(&x, &w, None, &s, [1, 1, 1], [1, 1, 1], [1, 1, 1], 2, 1);

    assert_grad_close("dx(groups,padding)", &dense(dx), &num_dx);
    assert_grad_close("dw(groups,padding)", &dense(dw), &num_dw);
}

// --- 3. shape／引数検査の境界 ---

#[test]
fn rejects_zero_stride() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4, 4, 4])], &[1, 1, 4, 4, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 3, 3, 3])], &[1, 1, 3, 3, 3]));
    let err = conv3d(&x, &w, None, [0, 1, 1], [0, 0, 0], [1, 1, 1], 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn rejects_zero_dilation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4, 4, 4])], &[1, 1, 4, 4, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 3, 3, 3])], &[1, 1, 3, 3, 3]));
    let err = conv3d(&x, &w, None, [1, 1, 1], [0, 0, 0], [1, 1, 0], 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn rejects_zero_groups() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4, 4, 4])], &[1, 1, 4, 4, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 3, 3, 3])], &[1, 1, 3, 3, 3]));
    let err = conv3d(&x, &w, None, [1, 1, 1], [0, 0, 0], [1, 1, 1], 0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn rejects_weight_rank_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4, 4, 4])], &[1, 1, 4, 4, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 3, 3])], &[1, 1, 3, 3]));
    let err = conv3d(&x, &w, None, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::RankMismatch { .. })
    ));
}

#[test]
fn rejects_cin_not_divisible_by_groups() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 3, 4, 4, 4])], &[1, 3, 4, 4, 4]));
    let w = tape.var(&t(vec![0.0; 3 * 2 * 2 * 2 * 2], &[3, 2, 2, 2, 2]));
    let err = conv3d(&x, &w, None, [1, 1, 1], [0, 0, 0], [1, 1, 1], 2).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn rejects_bias_shape_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4, 4, 4])], &[1, 1, 4, 4, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[2, 1, 3, 3, 3])], &[2, 1, 3, 3, 3]));
    let b = tape.var(&t(vec![0.0; 3], &[3]));
    let err = conv3d(&x, &w, Some(&b), [1, 1, 1], [0, 0, 0], [1, 1, 1], 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn rejects_d_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x0 = tape.var(&t(Vec::new(), &[1, 1, 0, 4, 4]));
    let w0 = tape.var(&t(vec![0.0; numel(&[1, 1, 1, 1, 1])], &[1, 1, 1, 1, 1]));
    let err = conv3d(&x0, &w0, None, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

// --- 4. kD=1・D=1 が reshape 経由の Conv2d と bit 一致 ---

#[test]
fn kernel_depth_one_matches_conv2d_via_reshape() {
    let in_shape_3d = [1usize, 2, 1, 4, 4];
    let in_shape_2d = [1usize, 2, 4, 4];
    let w_shape_3d = [3usize, 2, 1, 2, 2];
    let w_shape_2d = [3usize, 2, 2, 2];
    let x_data: Vec<f32> = (0..numel(&in_shape_3d))
        .map(|i| (i as f32 * 0.019).sin())
        .collect();
    let w_data: Vec<f32> = (0..numel(&w_shape_3d))
        .map(|i| (i as f32 * 0.041).cos() * 0.5)
        .collect();
    let b_data = vec![0.05f32, -0.1, 0.2];

    let tape3 = Tape::new_with_ops(common::naive_ops());
    let x3 = tape3.var(&t(x_data.clone(), &in_shape_3d));
    let w3 = tape3.var(&t(w_data.clone(), &w_shape_3d));
    let b3 = tape3.var(&t(b_data.clone(), &[3]));
    let y3 = conv3d(&x3, &w3, Some(&b3), [1, 1, 1], [0, 0, 0], [1, 1, 1], 1).unwrap();

    let tape2 = Tape::new_with_ops(common::naive_ops());
    let x2 = tape2.var(&t(x_data, &in_shape_2d));
    let w2 = tape2.var(&t(w_data, &w_shape_2d));
    let b2 = tape2.var(&t(b_data, &[3]));
    let y2 = x2
        .conv2d(&w2, Some(&b2), [1, 1], [0, 0], [1, 1], 1)
        .unwrap();

    let d3 = dense(&y3.to_tensor());
    let d2 = dense(&y2.to_tensor());
    assert_eq!(
        d3, d2,
        "kD=1 の Conv3d と reshape 経由の Conv2d は bit 一致するはず"
    );
}

// --- 5. bias broadcast の軸回帰 ---

#[test]
fn bias_adds_to_cout_axis_not_wout_axis_when_equal() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // Cout=3, Wout=3 が一致する形状で bias が W 軸へ誤って足されないことを検証する。
    let x = tape.var(&t(vec![1.0; numel(&[1, 1, 3, 3, 3])], &[1, 1, 3, 3, 3]));
    let w = tape.var(&t(vec![1.0; numel(&[3, 1, 1, 1, 1])], &[3, 1, 1, 1, 1]));
    let b = tape.var(&t(vec![10.0, 20.0, 30.0], &[3]));
    let y = conv3d(&x, &w, Some(&b), [1, 1, 1], [0, 0, 0], [1, 1, 1], 1).unwrap();
    assert_eq!(y.to_tensor().shape().to_vec(), vec![1, 3, 3, 3, 3]);
    let data = dense(&y.to_tensor());
    // kernel=1x1x1・weight=1 なので各 (co, od, oh, ow) = 1.0 + bias[co]。
    for co in 0..3 {
        let expected_bias = [10.0, 20.0, 30.0][co];
        for p in 0..27 {
            let v = data[co * 27 + p];
            assert_eq!(v, 1.0 + expected_bias, "co={co} p={p}");
        }
    }
}
