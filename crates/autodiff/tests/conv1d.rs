//! `Var::conv1d`（`Var::conv2d` の reshape 併合。イシュー #1765・設計
//! `docs/conv-ops-design.md` §2／§8）の受け入れ条件検証。
//!
//! - bit 一致（主要件）: `conv1d(..)` と手動 reshape した
//!   `[N, C, 1, L]`／`[Cout, Cin_g, 1, k]` に対する `conv2d([1, s],
//!   [0, p], [1, d], groups)` の forward／`d_input`／`d_weight`／
//!   `d_bias` が `to_bits()` で完全一致すること。
//! - 整数手計算オラクル: 非対称カーネルの cross-correlation 値。
//! - forward: 出力 shape が PyTorch `_conv_output_size` 相当と一致。
//! - backward: 数値微分（中央差分）との突合。`tests/conv2d.rs` と
//!   同じ判定閾値（`H=1e-3`・相対 1e-2 または絶対 1e-3・`τ=1e-4`。
//!   新規の許容誤差緩和はしない）。
//! - shape／引数検査の境界（rank・`stride=0`／`dilation=0`／
//!   `groups=0`・`L=0` 拒否・チャンネル不整合・bias shape 不一致・
//!   負分子拒否）。各 `Err` 経路で `Tape::len()` が呼び出し前後で
//!   不変（孤児 view ノードなし）であることを固定。
//! - 非 contiguous 入力: transpose view を渡しても contiguous 化した
//!   コピーと結果が bit 一致すること（`contiguous` 前段の契約固定）。

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

/// `L = Σ conv1d(x, w, b) · s` の中央差分による数値勾配（f64 集計）。
/// `param`（`0`=input・`1`=weight・`2`=bias）に対応する勾配のみ返す。
#[allow(clippy::too_many_arguments)]
fn numeric_conv1d_grad(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    b: Option<&Tensor<f32>>,
    s: &Tensor<f32>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    param: usize,
) -> Vec<f64> {
    let forward = |x: &Tensor<f32>, w: &Tensor<f32>, b: Option<&Tensor<f32>>| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        let wv = tape.var(w);
        let bv = b.map(|bt| tape.var(bt));
        let y = xv
            .conv1d(&wv, bv.as_ref(), stride, padding, dilation, groups)
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

// --- 1. bit 一致（conv1d ↔ 手動 reshape conv2d。設計 doc §12 主要件） ---

/// `matches_manual_reshape_conv2d_bit_exact` の 1 ケース（N・Cin・L・
/// Cout・Cin_g・k・stride・padding・dilation・groups）。tuple 型を
/// 直書きすると clippy::type_complexity に抵触するためエイリアス化
/// する（`conv2d.rs::ConvShapeCase` と同じ理由）。
type Conv1dBitExactCase = (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
);

#[test]
fn matches_manual_reshape_conv2d_bit_exact() {
    // (N, Cin, L, Cout, Cin_g, k, stride, padding, dilation, groups)
    let cases: [Conv1dBitExactCase; 3] = [
        (2, 3, 9, 4, 3, 3, 1, 0, 1, 1),
        (1, 4, 10, 4, 1, 3, 1, 1, 1, 4), // depthwise (groups=Cin=Cout)
        (1, 4, 12, 6, 2, 3, 2, 1, 2, 2), // groups + dilation + stride（重なり窓）
    ];
    for (n, cin, l, cout, cin_g, k, stride, padding, dilation, groups) in cases {
        let x = t(
            (0..numel(&[n, cin, l]))
                .map(|i| (i as f32 * 0.031).sin())
                .collect(),
            &[n, cin, l],
        );
        let w = t(
            (0..cout * cin_g * k)
                .map(|i| (i as f32 * 0.077).cos() * 0.5)
                .collect(),
            &[cout, cin_g, k],
        );
        let b = t((0..cout).map(|i| 0.1 * (i as f32 + 1.0)).collect(), &[cout]);

        // conv1d 経路
        let tape1 = Tape::new_with_ops(common::naive_ops());
        let x1 = tape1.var(&x);
        let w1 = tape1.var(&w);
        let b1 = tape1.var(&b);
        let y1 = x1
            .conv1d(&w1, Some(&b1), stride, padding, dilation, groups)
            .unwrap();
        let out_shape = y1.to_tensor().shape().to_vec();
        let s = t(
            (0..out_shape.iter().product::<usize>())
                .map(|i| ((i as f32) * 0.043).sin() * 0.6)
                .collect(),
            &out_shape,
        );
        let loss1 = y1.mul(&tape1.var(&s)).unwrap().sum(None).unwrap();
        let grads1 = tape1.backward(&loss1).unwrap();
        let dx1 = dense(grads1.get(&x1).unwrap().expect("x1 reaches loss"));
        let dw1 = dense(grads1.get(&w1).unwrap().expect("w1 reaches loss"));
        let db1 = dense(grads1.get(&b1).unwrap().expect("b1 reaches loss"));
        let fw1 = dense(&y1.to_tensor());

        // 手動 reshape → conv2d 経路
        let x4 = t(x.contiguous().as_slice().unwrap().to_vec(), &[n, cin, 1, l]);
        let w4 = t(
            w.contiguous().as_slice().unwrap().to_vec(),
            &[cout, cin_g, 1, k],
        );
        let tape2 = Tape::new_with_ops(common::naive_ops());
        let x2 = tape2.var(&x4);
        let w2 = tape2.var(&w4);
        let b2 = tape2.var(&b);
        let y2 = x2
            .conv2d(
                &w2,
                Some(&b2),
                [1, stride],
                [0, padding],
                [1, dilation],
                groups,
            )
            .unwrap();
        let out4_shape = y2.to_tensor().shape().to_vec();
        assert_eq!(
            out4_shape,
            vec![out_shape[0], out_shape[1], 1, out_shape[2]]
        );
        let s4 = t(dense(&s), &out4_shape);
        let loss2 = y2.mul(&tape2.var(&s4)).unwrap().sum(None).unwrap();
        let grads2 = tape2.backward(&loss2).unwrap();
        let dx2 = dense(grads2.get(&x2).unwrap().expect("x2 reaches loss"));
        let dw2 = dense(grads2.get(&w2).unwrap().expect("w2 reaches loss"));
        let db2 = dense(grads2.get(&b2).unwrap().expect("b2 reaches loss"));
        let fw2 = dense(&y2.to_tensor());

        assert_eq!(
            fw1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fw2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "forward bit mismatch: case={n},{cin},{l},{cout},{cin_g},{k}"
        );
        assert_eq!(
            dx1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            dx2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "d_input bit mismatch"
        );
        assert_eq!(
            dw1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            dw2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "d_weight bit mismatch"
        );
        assert_eq!(
            db1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            db2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "d_bias bit mismatch"
        );
    }
}

// --- 2. 整数手計算オラクル ---

#[test]
fn matches_hand_computed_cross_correlation_no_padding() {
    // w=[1,2,3], x=[1,2,3,4,5], stride=1, padding=0 -> cross-correlation
    // （カーネル反転なし）: [1*1+2*2+3*3, 1*2+2*3+3*4, 1*3+2*4+3*5]
    //   = [14, 20, 26]
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]));
    let w = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 1, 3]));
    let y = x.conv1d(&w, None, 1, 0, 1, 1).unwrap();
    assert_eq!(y.to_tensor().shape().to_vec(), vec![1, 1, 3]);
    assert_eq!(dense(&y.to_tensor()), vec![14.0, 20.0, 26.0]);
}

#[test]
fn matches_hand_computed_cross_correlation_with_padding() {
    // 同上・padding=1: 先頭は 0*1+1*2+2*3=8。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]));
    let w = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 1, 3]));
    let y = x.conv1d(&w, None, 1, 1, 1, 1).unwrap();
    assert_eq!(y.to_tensor().shape().to_vec(), vec![1, 1, 5]);
    let data = dense(&y.to_tensor());
    assert_eq!(data[0], 8.0);
}

// --- 3. forward: 出力 shape 表 ---

type Conv1dShapeCase = (
    [usize; 3],
    [usize; 3],
    usize,
    usize,
    usize,
    usize,
    [usize; 3],
);

#[test]
fn forward_output_shape_matches_pytorch_conv_output_size() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let cases: [Conv1dShapeCase; 4] = [
        // (input, weight, stride, padding, dilation, groups, expected out)
        ([1, 3, 8], [4, 3, 3], 1, 0, 1, 1, [1, 4, 6]),
        ([2, 4, 5], [6, 2, 3], 1, 1, 1, 2, [2, 6, 5]),
        ([1, 4, 5], [4, 1, 3], 1, 1, 1, 4, [1, 4, 5]),
        ([1, 1, 7], [1, 1, 3], 2, 0, 2, 1, [1, 1, 2]),
    ];
    for (in_shape, w_shape, stride, padding, dilation, groups, expected) in cases {
        let x = tape.var(&t(vec![0.1; in_shape.iter().product()], &in_shape));
        let w = tape.var(&t(vec![0.1; w_shape.iter().product()], &w_shape));
        let y = x
            .conv1d(&w, None, stride, padding, dilation, groups)
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
    let x = tape.var(&t(Vec::new(), &[0, 3, 8]));
    let w = tape.var(&t(vec![0.1; 4 * 3 * 3], &[4, 3, 3]));
    let y = x.conv1d(&w, None, 1, 0, 1, 1).unwrap();
    assert_eq!(y.to_tensor().shape().to_vec(), vec![0, 4, 6]);
}

// --- 4. backward: 数値微分突合 ---

#[test]
fn backward_matches_numeric_grad_basic() {
    let x = t(
        (0..numel(&[1, 2, 9]))
            .map(|i| (i as f32 * 0.037).sin())
            .collect(),
        &[1, 2, 9],
    );
    let w = t(
        (0..3 * 2 * 3)
            .map(|i| (i as f32 * 0.091).cos() * 0.5)
            .collect(),
        &[3, 2, 3],
    );
    let b = t(vec![0.1, -0.2, 0.3], &[3]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let bv = tape.var(&b);
    let y = xv.conv1d(&wv, Some(&bv), 1, 1, 1, 1).unwrap();
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

    let num_dx = numeric_conv1d_grad(&x, &w, Some(&b), &s, 1, 1, 1, 1, 0);
    let num_dw = numeric_conv1d_grad(&x, &w, Some(&b), &s, 1, 1, 1, 1, 1);
    let num_db = numeric_conv1d_grad(&x, &w, Some(&b), &s, 1, 1, 1, 1, 2);

    assert_grad_close("dx", &dense(dx), &num_dx);
    assert_grad_close("dw", &dense(dw), &num_dw);
    assert_grad_close("db", &dense(db), &num_db);
}

#[test]
fn backward_matches_numeric_grad_groups_dilation_no_bias() {
    let x = t(
        (0..numel(&[1, 4, 12]))
            .map(|i| (i as f32 * 0.029).sin())
            .collect(),
        &[1, 4, 12],
    );
    let w = t(
        (0..numel(&[4, 1, 3]))
            .map(|i| (i as f32 * 0.061).cos() * 0.5)
            .collect(),
        &[4, 1, 3],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let y = xv.conv1d(&wv, None, 1, 1, 2, 4).unwrap();
    let out_shape = y.to_tensor().shape().to_vec();
    let s = t(
        (0..out_shape.iter().product::<usize>())
            .map(|i| ((i as f32) * 0.067).cos() * 0.4)
            .collect(),
        &out_shape,
    );

    let loss = y.mul(&tape.var(&s)).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("xv reaches loss");
    let dw = grads.get(&wv).unwrap().expect("wv reaches loss");

    let num_dx = numeric_conv1d_grad(&x, &w, None, &s, 1, 1, 2, 4, 0);
    let num_dw = numeric_conv1d_grad(&x, &w, None, &s, 1, 1, 2, 4, 1);

    assert_grad_close("dx", &dense(dx), &num_dx);
    assert_grad_close("dw", &dense(dw), &num_dw);
}

#[test]
fn backward_matches_numeric_grad_overlapping_stride_window() {
    // stride < 有効カーネル幅（dilation 込み）で重なり窓を作る。
    let x = t(
        (0..numel(&[1, 1, 10]))
            .map(|i| (i as f32 * 0.043).sin())
            .collect(),
        &[1, 1, 10],
    );
    let w = t(vec![0.3, -0.5, 0.8], &[1, 1, 3]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let y = xv.conv1d(&wv, None, 1, 0, 1, 1).unwrap();
    let out_shape = y.to_tensor().shape().to_vec();
    let s = t(vec![1.0; out_shape.iter().product()], &out_shape);

    let loss = y.mul(&tape.var(&s)).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("xv reaches loss");

    let num_dx = numeric_conv1d_grad(&x, &w, None, &s, 1, 0, 1, 1, 0);
    assert_grad_close("dx(overlap)", &dense(dx), &num_dx);
}

// --- 5. shape／引数検査の境界（孤児ノードなし固定込み） ---

#[test]
fn rejects_input_rank_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4, 4])], &[1, 1, 4, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 3])], &[1, 1, 3]));
    let len_before = tape.len();
    let err = x.conv1d(&w, None, 1, 0, 1, 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 3,
            actual: 4
        })
    ));
    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

#[test]
fn rejects_weight_rank_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4])], &[1, 1, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 3, 3])], &[1, 1, 3, 3]));
    let len_before = tape.len();
    let err = x.conv1d(&w, None, 1, 0, 1, 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 3,
            actual: 4
        })
    ));
    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

#[test]
fn rejects_zero_stride() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4])], &[1, 1, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 3])], &[1, 1, 3]));
    let len_before = tape.len();
    let err = x.conv1d(&w, None, 0, 0, 1, 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

#[test]
fn rejects_zero_dilation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4])], &[1, 1, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 3])], &[1, 1, 3]));
    let len_before = tape.len();
    let err = x.conv1d(&w, None, 1, 0, 0, 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

#[test]
fn rejects_zero_groups() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4])], &[1, 1, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 3])], &[1, 1, 3]));
    let len_before = tape.len();
    let err = x.conv1d(&w, None, 1, 0, 1, 0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

#[test]
fn rejects_l_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(Vec::new(), &[1, 1, 0]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 1])], &[1, 1, 1]));
    let len_before = tape.len();
    let err = x.conv1d(&w, None, 1, 0, 1, 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

#[test]
fn rejects_cin_not_divisible_by_groups() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 3, 8])], &[1, 3, 8]));
    let w = tape.var(&t(vec![0.0; 4 * 2 * 3], &[4, 2, 3]));
    let len_before = tape.len();
    let err = x.conv1d(&w, None, 1, 0, 1, 2).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

#[test]
fn rejects_weight_cin_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 4, 8])], &[1, 4, 8]));
    let w = tape.var(&t(vec![0.0; 4 * 3 * 3], &[4, 3, 3]));
    let len_before = tape.len();
    let err = x.conv1d(&w, None, 1, 0, 1, 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

#[test]
fn rejects_bias_shape_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 4])], &[1, 1, 4]));
    let w = tape.var(&t(vec![0.0; numel(&[2, 1, 3])], &[2, 1, 3]));
    let b = tape.var(&t(vec![0.0; 3], &[3]));
    let len_before = tape.len();
    let err = x.conv1d(&w, Some(&b), 1, 0, 1, 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

#[test]
fn rejects_negative_numerator_kernel_too_large() {
    // L=1, k=3, p=0, d=1 -> 分子 1 - 2 - 1 = -2 < 0（負分子拒否ゲート）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; numel(&[1, 1, 1])], &[1, 1, 1]));
    let w = tape.var(&t(vec![0.0; numel(&[1, 1, 3])], &[1, 1, 3]));
    let len_before = tape.len();
    let err = x.conv1d(&w, None, 1, 0, 1, 1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

// --- 6. 非 contiguous 入力（contiguous 前段の契約固定） ---

#[test]
fn non_contiguous_input_matches_contiguous_copy() {
    // [N, L, C] を transpose(1, 2) して [N, C, L] の非 contiguous view
    // を作り、明示コピーした contiguous 版との結果が bit 一致すること
    // を確認する（`Var::reshape` の非 contiguous 拒否契約と `conv2d`
    // の間の非対称を `contiguous` 前段が解消している証跡）。
    let n = 1;
    let c = 3;
    let l = 8;
    let nlc = t(
        (0..numel(&[n, l, c]))
            .map(|i| (i as f32 * 0.017).sin())
            .collect(),
        &[n, l, c],
    );
    let w = t(
        (0..4 * c * 3)
            .map(|i| (i as f32 * 0.089).cos() * 0.5)
            .collect(),
        &[4, c, 3],
    );

    // 非 contiguous 経路
    let tape1 = Tape::new_with_ops(common::naive_ops());
    let nlc1 = tape1.var(&nlc);
    let ncl1 = nlc1.transpose(1, 2).unwrap();
    let w1 = tape1.var(&w);
    let y1 = ncl1.conv1d(&w1, None, 1, 1, 1, 1).unwrap();
    let fw1 = dense(&y1.to_tensor());

    // 明示コピーした contiguous 経路
    let ncl_data: Vec<f32> = {
        let src = dense(&nlc);
        let mut out = vec![0f32; n * c * l];
        for ni in 0..n {
            for ci in 0..c {
                for li in 0..l {
                    out[ni * c * l + ci * l + li] = src[ni * l * c + li * c + ci];
                }
            }
        }
        out
    };
    let ncl = t(ncl_data, &[n, c, l]);
    let tape2 = Tape::new_with_ops(common::naive_ops());
    let ncl2 = tape2.var(&ncl);
    let w2 = tape2.var(&w);
    let y2 = ncl2.conv1d(&w2, None, 1, 1, 1, 1).unwrap();
    let fw2 = dense(&y2.to_tensor());

    assert_eq!(
        fw1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        fw2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "非 contiguous 入力の結果が contiguous コピーと bit 不一致"
    );
}
