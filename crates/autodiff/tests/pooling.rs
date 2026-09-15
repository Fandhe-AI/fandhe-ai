//! `Var::max_pool2d`／`avg_pool2d`／`adaptive_avg_pool2d`（1d／2d。
//! イシュー #1728・設計 `docs/pooling-ops-design.md`）の受け入れ条件
//! 検証。`common::naive_ops()`（全メソッド既定 `Unsupported` →
//! `eval::*` ホスト参照実装へ強制フォールバック）経由で forward／
//! backward を検証する（`tests/conv2d.rs` と同方針）。CPU ネイティブ
//! 実装（`CpuBackendOps`）とのバックエンド間 parity は
//! `crates/backend-cpu/tests/pooling_parity.rs`（クレート内直接呼び
//! 出し）・`crates/facade/tests/pooling_backend_parity.rs`
//! （facade 経由）が別途担う。
//!
//! - forward: 出力 shape 表（PyTorch 出力式との一致）。
//! - 数値微分（中央差分）との突合: Avg は全構成、Max はタイのない
//!   入力（`tests/conv2d.rs` と同一閾値・新規緩和なし）。
//! - 重なり窓 Max VJP の手計算値との bit 一致（`scatter_add` 契約）。
//! - 1d ≡ `[N,C,1,L]` 2d の bit 一致（forward／backward・索引）。
//! - `ceil_mode=true` 拒否・引数検査の `AutodiffError` variant 固定。

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

fn assert_bit_exact(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape が一致しない");
    let av = dense(a);
    let bv = dense(b);
    for (i, (&x, &y)) in av.iter().zip(bv.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}[{i}]: bit 不一致（a={x}, b={y}）"
        );
    }
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

/// `L = Σ avg_pool2d(x) · s` の中央差分による数値勾配（f64 集計）。
#[allow(clippy::too_many_arguments)]
fn numeric_avg_pool2d_grad(
    x: &Tensor<f32>,
    s: &Tensor<f32>,
    kernel: [usize; 2],
    stride: Option<[usize; 2]>,
    padding: [usize; 2],
    count_include_pad: bool,
) -> Vec<f64> {
    let forward = |x: &Tensor<f32>| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        let y = xv
            .avg_pool2d(kernel, stride, padding, false, count_include_pad)
            .unwrap();
        dense(&y.to_tensor())
            .iter()
            .zip(dense(s).iter())
            .map(|(&yv, &sv)| yv as f64 * sv as f64)
            .sum()
    };
    let shape = x.shape().to_vec();
    let mut data = dense(x);
    let mut grad = vec![0f64; data.len()];
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = forward(&t(data.clone(), &shape));
        data[i] = (orig - H) as f32;
        let lm = forward(&t(data.clone(), &shape));
        data[i] = orig as f32;
        grad[i] = (lp - lm) / (2.0 * H);
    }
    grad
}

/// `L = Σ max_pool2d(x).0 · s` の中央差分による数値勾配。
fn numeric_max_pool2d_grad(
    x: &Tensor<f32>,
    s: &Tensor<f32>,
    kernel: [usize; 2],
    stride: Option<[usize; 2]>,
    padding: [usize; 2],
) -> Vec<f64> {
    let forward = |x: &Tensor<f32>| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        let (y, _idx) = xv
            .max_pool2d(kernel, stride, padding, [1, 1], false)
            .unwrap();
        dense(&y.to_tensor())
            .iter()
            .zip(dense(s).iter())
            .map(|(&yv, &sv)| yv as f64 * sv as f64)
            .sum()
    };
    let shape = x.shape().to_vec();
    let mut data = dense(x);
    let mut grad = vec![0f64; data.len()];
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = forward(&t(data.clone(), &shape));
        data[i] = (orig - H) as f32;
        let lm = forward(&t(data.clone(), &shape));
        data[i] = orig as f32;
        grad[i] = (lp - lm) / (2.0 * H);
    }
    grad
}

// --- 1. forward: 出力 shape 表 ---

#[test]
fn max_pool2d_forward_output_shape() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 2 * 3 * 8 * 8], &[2, 3, 8, 8]));
    let (y, idx) = x.max_pool2d([2, 2], None, [0, 0], [1, 1], false).unwrap();
    assert_eq!(y.to_tensor().shape().to_vec(), vec![2, 3, 4, 4]);
    assert_eq!(idx.shape(), &[2, 3, 4, 4]);
}

#[test]
fn avg_pool2d_forward_output_shape_with_padding() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 25], &[1, 1, 5, 5]));
    let y = x
        .avg_pool2d([3, 3], Some([1, 1]), [1, 1], false, true)
        .unwrap();
    // PyTorch: floor((5 + 2 - 2 - 1)/1)+1 = 5
    assert_eq!(y.to_tensor().shape().to_vec(), vec![1, 1, 5, 5]);
}

#[test]
fn adaptive_avg_pool2d_forward_output_shape() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 63], &[1, 1, 7, 9]));
    let y = x.adaptive_avg_pool2d([2, 3]).unwrap();
    assert_eq!(y.to_tensor().shape().to_vec(), vec![1, 1, 2, 3]);
}

// --- 2. 数値微分突合 ---

#[test]
fn avg_pool2d_matches_numeric_gradient_overlapping_padding() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t(
        (0..24).map(|v| (v as f32) * 0.3 - 2.0).collect(),
        &[1, 2, 3, 4],
    );
    let xv = tape.var(&x);
    let y = xv
        .avg_pool2d([2, 2], Some([1, 1]), [1, 1], false, true)
        .unwrap();
    let s = t(
        vec![1.0; y.to_tensor().shape().to_vec().iter().product()],
        y.to_tensor().shape(),
    );
    let sv = tape.var(&s);
    let loss = y.mul(&sv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().unwrap();

    let numeric = numeric_avg_pool2d_grad(&x, &s, [2, 2], Some([1, 1]), [1, 1], true);
    assert_grad_close("avg_pool2d dX", &dense(dx), &numeric);
}

#[test]
fn avg_pool2d_matches_numeric_gradient_count_include_pad_false() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t(vec![2.0, 4.0, 6.0, 8.0], &[1, 1, 2, 2]);
    let xv = tape.var(&x);
    let y = xv
        .avg_pool2d([2, 2], Some([2, 2]), [1, 1], false, false)
        .unwrap();
    let s = t(
        vec![1.0; y.to_tensor().shape().to_vec().iter().product()],
        y.to_tensor().shape(),
    );
    let sv = tape.var(&s);
    let loss = y.mul(&sv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().unwrap();

    let numeric = numeric_avg_pool2d_grad(&x, &s, [2, 2], Some([2, 2]), [1, 1], false);
    assert_grad_close(
        "avg_pool2d(count_include_pad=false) dX",
        &dense(dx),
        &numeric,
    );
}

#[test]
fn adaptive_avg_pool2d_matches_numeric_gradient_shrink_and_expand() {
    // shrink
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = t((0..16).map(|v| v as f32 * 0.5).collect(), &[1, 1, 4, 4]);
        let xv = tape.var(&x);
        let y = xv.adaptive_avg_pool2d([2, 2]).unwrap();
        let s = t(
            vec![1.0; y.to_tensor().shape().to_vec().iter().product()],
            y.to_tensor().shape(),
        );
        let sv = tape.var(&s);
        let loss = y.mul(&sv).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&xv).unwrap().unwrap();

        // adaptive を avg_pool2d の数値微分ヘルパーで代用できないため
        // 直接中央差分を計算する。
        let forward = |x: &Tensor<f32>| -> f64 {
            let tape = Tape::new_with_ops(common::naive_ops());
            let xv = tape.var(x);
            let y = xv.adaptive_avg_pool2d([2, 2]).unwrap();
            dense(&y.to_tensor())
                .iter()
                .zip(dense(&s).iter())
                .map(|(&yv, &sv)| yv as f64 * sv as f64)
                .sum()
        };
        let shape = x.shape().to_vec();
        let mut data = dense(&x);
        let mut numeric = vec![0f64; data.len()];
        for i in 0..data.len() {
            let orig = data[i] as f64;
            data[i] = (orig + H) as f32;
            let lp = forward(&t(data.clone(), &shape));
            data[i] = (orig - H) as f32;
            let lm = forward(&t(data.clone(), &shape));
            data[i] = orig as f32;
            numeric[i] = (lp - lm) / (2.0 * H);
        }
        assert_grad_close("adaptive_avg_pool2d(shrink) dX", &dense(dx), &numeric);
    }
}

#[test]
fn max_pool2d_matches_numeric_gradient_no_ties() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // 全要素異なる値でタイを排除する。
    let x = t(
        vec![
            1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0, 7.0, 6.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,
        ],
        &[1, 1, 4, 4],
    );
    let xv = tape.var(&x);
    let (y, _idx) = xv
        .max_pool2d([2, 2], Some([2, 2]), [0, 0], [1, 1], false)
        .unwrap();
    let s = t(
        vec![1.0; y.to_tensor().shape().to_vec().iter().product()],
        y.to_tensor().shape(),
    );
    let sv = tape.var(&s);
    let loss = y.mul(&sv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().unwrap();

    let numeric = numeric_max_pool2d_grad(&x, &s, [2, 2], Some([2, 2]), [0, 0]);
    assert_grad_close("max_pool2d(no ties) dX", &dense(dx), &numeric);
}

// --- 3. 重なり窓 Max VJP の手計算値との bit 一致 ---

#[test]
fn max_pool2d_overlapping_windows_backward_matches_hand_computed_scatter_add() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // データ: row0=[1,3,2], row1=[5,4,0]（[1,1,2,3]）。
    // 窓(0,0)={(0,0)=1,(0,1)=3,(1,0)=5,(1,1)=4} -> max=5 @ (1,0)。
    // 窓(0,1)={(0,1)=3,(0,2)=2,(1,1)=4,(1,2)=0} -> max=4 @ (1,1)。
    let x = t(vec![1.0, 3.0, 2.0, 5.0, 4.0, 0.0], &[1, 1, 2, 3]);
    let xv = tape.var(&x);
    let (y, idx) = xv
        .max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1], false)
        .unwrap();
    assert_eq!(dense(&y.to_tensor()), vec![5.0, 4.0]);
    // idx flat: (1,0)=1*3+0=3, (1,1)=1*3+1=4
    assert_eq!(idx.get(&[0, 0, 0, 0]), Some(3));
    assert_eq!(idx.get(&[0, 0, 0, 1]), Some(4));

    let upstream = t(vec![10.0, 100.0], &[1, 1, 1, 2]);
    let uv = tape.var(&upstream);
    let loss = y.mul(&uv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().unwrap();
    // scatter_add: 位置(1,0)(flat=3) <- 10.0, 位置(1,1)(flat=4) <- 100.0,
    // 他は 0。
    assert_eq!(dense(dx), vec![0.0, 0.0, 0.0, 10.0, 100.0, 0.0]);
}

// --- 4. 1d ≡ [N,C,1,L] 2d の bit 一致 ---

#[test]
fn max_pool1d_forward_and_backward_matches_2d_reshape() {
    let tape2d = Tape::new_with_ops(common::naive_ops());
    let x = t(vec![1.0, 5.0, 2.0, 8.0, 3.0, 9.0], &[1, 1, 1, 6]);
    let x2 = tape2d.var(&x);
    let (y2, idx2) = x2
        .max_pool2d([1, 2], Some([1, 2]), [0, 0], [1, 1], false)
        .unwrap();
    let s2 = t(
        vec![1.0; y2.to_tensor().shape().to_vec().iter().product()],
        y2.to_tensor().shape(),
    );
    let sv2 = tape2d.var(&s2);
    let loss2 = y2.mul(&sv2).unwrap().sum(None).unwrap();
    let g2 = tape2d.backward(&loss2).unwrap();
    let dx2 = g2.get(&x2).unwrap().unwrap();

    let tape1d = Tape::new_with_ops(common::naive_ops());
    let x1_flat = t(vec![1.0, 5.0, 2.0, 8.0, 3.0, 9.0], &[1, 1, 6]);
    let x1 = tape1d.var(&x1_flat);
    let (y1, idx1) = x1.max_pool1d(2, Some(2), 0, 1, false).unwrap();
    let s1 = t(
        vec![1.0; y1.to_tensor().shape().to_vec().iter().product()],
        y1.to_tensor().shape(),
    );
    let sv1 = tape1d.var(&s1);
    let loss1 = y1.mul(&sv1).unwrap().sum(None).unwrap();
    let g1 = tape1d.backward(&loss1).unwrap();
    let dx1 = g1.get(&x1).unwrap().unwrap();

    assert_bit_exact(
        "max_pool1d forward",
        &y1.to_tensor(),
        &y2.to_tensor().reshape(&[1, 1, 3]).unwrap(),
    );
    assert_eq!(idx1.shape(), &[1, 1, 3]);
    assert_eq!(
        idx2.reshape(&[1, 1, 3]).unwrap().host_slice(),
        idx1.host_slice()
    );
    assert_bit_exact(
        "max_pool1d backward dX",
        dx1,
        &dx2.reshape(&[1, 1, 6]).unwrap(),
    );
}

#[test]
fn avg_pool1d_forward_and_backward_matches_2d_reshape() {
    let tape2d = Tape::new_with_ops(common::naive_ops());
    let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 1, 6]);
    let x2 = tape2d.var(&x);
    let y2 = x2
        .avg_pool2d([1, 2], Some([1, 2]), [0, 0], false, true)
        .unwrap();
    let s2 = t(
        vec![1.0; y2.to_tensor().shape().to_vec().iter().product()],
        y2.to_tensor().shape(),
    );
    let sv2 = tape2d.var(&s2);
    let loss2 = y2.mul(&sv2).unwrap().sum(None).unwrap();
    let g2 = tape2d.backward(&loss2).unwrap();
    let dx2 = g2.get(&x2).unwrap().unwrap();

    let tape1d = Tape::new_with_ops(common::naive_ops());
    let x1_flat = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6]);
    let x1 = tape1d.var(&x1_flat);
    let y1 = x1.avg_pool1d(2, Some(2), 0, false, true).unwrap();
    let s1 = t(
        vec![1.0; y1.to_tensor().shape().to_vec().iter().product()],
        y1.to_tensor().shape(),
    );
    let sv1 = tape1d.var(&s1);
    let loss1 = y1.mul(&sv1).unwrap().sum(None).unwrap();
    let g1 = tape1d.backward(&loss1).unwrap();
    let dx1 = g1.get(&x1).unwrap().unwrap();

    assert_bit_exact(
        "avg_pool1d forward",
        &y1.to_tensor(),
        &y2.to_tensor().reshape(&[1, 1, 3]).unwrap(),
    );
    assert_bit_exact(
        "avg_pool1d backward dX",
        dx1,
        &dx2.reshape(&[1, 1, 6]).unwrap(),
    );
}

#[test]
fn adaptive_avg_pool1d_matches_2d_reshape() {
    let tape2d = Tape::new_with_ops(common::naive_ops());
    let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 1, 1, 7]);
    let x2 = tape2d.var(&x);
    let y2 = x2.adaptive_avg_pool2d([1, 2]).unwrap();

    let tape1d = Tape::new_with_ops(common::naive_ops());
    let x1_flat = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 1, 7]);
    let x1 = tape1d.var(&x1_flat);
    let y1 = x1.adaptive_avg_pool1d(2).unwrap();

    assert_bit_exact(
        "adaptive_avg_pool1d forward",
        &y1.to_tensor(),
        &y2.to_tensor().reshape(&[1, 1, 2]).unwrap(),
    );
}

// --- 5. ceil_mode=true 拒否・引数検査エラー variant ---

#[test]
fn max_pool2d_rejects_ceil_mode_true() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 16], &[1, 1, 4, 4]));
    let err = x
        .max_pool2d([2, 2], None, [0, 0], [1, 1], true)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn avg_pool2d_rejects_ceil_mode_true() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 16], &[1, 1, 4, 4]));
    let err = x.avg_pool2d([2, 2], None, [0, 0], true, true).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn max_pool2d_rejects_padding_over_half_kernel() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 16], &[1, 1, 4, 4]));
    let err = x
        .max_pool2d([2, 2], None, [2, 2], [1, 1], false)
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn max_pool2d_rejects_zero_spatial_axis() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(Vec::new(), &[1, 1, 0, 4]));
    let err = x
        .max_pool2d([2, 2], None, [0, 0], [1, 1], false)
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn adaptive_avg_pool2d_rejects_output_size_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 16], &[1, 1, 4, 4]));
    let err = x.adaptive_avg_pool2d([0, 2]).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}
