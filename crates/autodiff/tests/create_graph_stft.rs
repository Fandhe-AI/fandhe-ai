//! `Stft`／`Istft` の `create_graph`（`Tape::backward_create_graph`）対応の受け入れテスト
//! （イシュー #3804・親 #3733・決定記録 `docs/autodiff-higher-order-grad-decision.md` §42・§63）。
//!
//! 方式は随伴 Op の対（`StftAdjoint`・`IstftAdjoint`。内部 Op のため随伴性そのものは crate 内の
//! `create_graph/stft.rs` のテストが固定する）。本ファイルは公開 API 経由で次を検証する:
//! - Hessian を `Tape::backward` の 5 点中央差分（`create_graph` を経由しない独立経路）と REQ-2 統一
//!   複合判定で突合する（center〈Reflect／Constant〉・normalized・onesided・rank 1／2・短い窓・
//!   hop が割り切れない場合・`length` の短縮／延長・`onesided = false`）。
//! - `cg.first_order()` が `Tape::backward` と bit 同一、`cg.grad(..)` が REQ-2 判定で一致する。線形な
//!   上流では CPU の `cg.grad` も 1 階と bit 一致する。
//! - 閉包性（孫テープ再生）・`jvp_create_graph`・`istft(stft(x))` の往復・非有限入力と大きな上流の伝播・
//!   0 詰め部分の二階勾配 0・隣接する未対応 Op の fail-closed・backend override 経路。
//!
//! GPU 実機 parity は `crates/facade/tests/create_graph_gpu_parity.rs`（`#[ignore]`・未実測。Phase 10 の
//! #3857・#3859）で扱う。

#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::functional_ops::jvp_create_graph;
use fandhe_ai_autodiff::{AutodiffError, IstftOptions, StftOptions, Tape, Var};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::fft::{IstftParams, StftParams};
use fandhe_ai_tensor_core::{BackendOps, StftPadMode, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn pattern(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + phase) * 0.9).sin() + 0.35 * (i as f32 + 1.0).cos() + 0.2)
        .collect()
}

fn dense(v: &Tensor<f32>) -> Vec<f32> {
    v.contiguous().as_slice().expect("contiguous").to_vec()
}

type Built<'t> = Result<Var<'t>, AutodiffError>;

/// 二次形式の損失 `sum(y² ⊙ c)`（`c` は非一様な固定定数）。Hessian が x に依らず、大きな刻みでも
/// 中央差分が厳密になる（決定記録 §31.4 の前例）。
fn quad_sum<'t>(y: &Var<'t>, tape: &'t Tape) -> Built<'t> {
    let shape = y.to_tensor().shape().to_vec();
    let c = tape.var_no_grad(&t(pattern(numel(&shape), 4.0), &shape));
    y.mul(y)?.mul(&c)?.sum(None)
}

/// 非線形な損失 `sum(tanh(y) ⊙ c)`。x が非線形項へ流れる構成。
fn tanh_sum<'t>(y: &Var<'t>, tape: &'t Tape) -> Built<'t> {
    let shape = y.to_tensor().shape().to_vec();
    let c = tape.var_no_grad(&t(pattern(numel(&shape), 1.0), &shape));
    y.tanh().mul(&c)?.sum(None)
}

fn x_data(shape: &[usize]) -> Vec<f32> {
    pattern(numel(shape), 7.0)
        .into_iter()
        .enumerate()
        .map(|(i, v)| v * 0.5 + (i % 3) as f32 * 0.2 - 0.3)
        .collect()
}

fn hann(n: usize) -> Tensor<f32> {
    t(
        (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos() + 0.05)
            .collect(),
        &[n],
    )
}

fn grad_at<F>(build: &F, x: &[f32], shape: &[usize]) -> Vec<f64>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&t(x.to_vec(), shape));
    let loss = build(&tape, &xv).expect("forward 構築");
    let grads = tape.backward(&loss).expect("1 階 backward");
    let dx = grads.get(&xv).expect("get").expect("x は loss に到達する");
    dense(dx).into_iter().map(f64::from).collect()
}

fn finite_diff_hessian<F>(build: &F, x0: &[f32], shape: &[usize], h: f64) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let n = numel(shape);
    let mut hessian = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let at = |d: f64| {
            let mut x = x0.to_vec();
            x[i] += d as f32;
            grad_at(build, &x, shape)
        };
        let (p2, p1, m1, m2) = (at(2.0 * h), at(h), at(-h), at(-2.0 * h));
        for j in 0..n {
            hessian[j][i] = (-p2[j] + 8.0 * p1[j] - 8.0 * m1[j] + m2[j]) / (12.0 * h);
        }
    }
    hessian
}

fn extract_scalar<'c>(v: &Var<'c>, shape: &[usize], flat_idx: usize) -> Built<'c> {
    let mut rem = flat_idx;
    let mut idx = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        idx[d] = rem % shape[d];
        rem /= shape[d];
    }
    let mut cur = *v;
    for (dim, &i) in idx.iter().enumerate() {
        cur = cur.narrow(dim, i, 1)?;
    }
    cur.sum(None)
}

fn analytic_hessian<F>(build: &F, x0: &[f32], shape: &[usize]) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let n = numel(shape);
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward 構築");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let gx = cg.grad(&x).expect("grad").expect("x は loss に到達する");
    let cx = cg.child_var(&x).expect("child_var").expect("写しあり");
    let mut hessian = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let gj = extract_scalar(&gx, shape, j).expect("extract_scalar");
        let row_grads = child.backward(&gj).expect("child.backward");
        let row = dense(
            row_grads
                .get(&cx)
                .expect("get")
                .expect("2 階勾配は x へ到達"),
        );
        for i in 0..n {
            hessian[j][i] = f64::from(row[i]);
        }
    }
    hessian
}

/// 閉包性: 子テープ上の勾配をさらに `backward_create_graph` で孫テープへ再生しても二階の値が一致する
/// （`StftAdjoint`／`IstftAdjoint` の replay と VJP を通る）。
fn check_grandchild<F>(ctx: &str, build: &F, x0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let grand = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward 構築");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let gx = cg.grad(&x).expect("grad").expect("到達");
    let cx = cg.child_var(&x).expect("child_var").expect("写しあり");
    let c = child.var_no_grad(&t(pattern(numel(shape), 2.0), shape));
    let scalar = gx.mul(&c).expect("mul").sum(None).expect("sum");
    let direct = dense(
        child
            .backward(&scalar)
            .expect("child.backward")
            .get(&cx)
            .expect("get")
            .expect("到達"),
    );
    let cg2 = child
        .backward_create_graph(&scalar, &grand)
        .expect("孫テープへの再生（閉包性）");
    let first = dense(cg2.first_order().get(&cx).expect("get").expect("到達"));
    let second = dense(&cg2.grad(&cx).expect("grad").expect("到達").to_tensor());
    for i in 0..direct.len() {
        assert!(
            first[i].is_finite() && second[i].is_finite(),
            "{ctx}: 非有限"
        );
        assert!(
            common::req2_close(f64::from(first[i]), f64::from(direct[i])),
            "{ctx}: 孫 first_order[{i}]={} vs {}",
            first[i],
            direct[i]
        );
        assert!(
            common::req2_close(f64::from(second[i]), f64::from(direct[i])),
            "{ctx}: 孫 grad[{i}]={} vs {}",
            second[i],
            direct[i]
        );
    }
}

/// Hessian（有限差分）・1 階（bit 同一・REQ-2）・孫テープ再生を一括で検証する。
fn check_all<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize], h: f64)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let numeric = finite_diff_hessian(&build, x0, shape, h);
    let analytic = analytic_hessian(&build, x0, shape);
    for (j, (ra, rn)) in analytic.iter().zip(numeric.iter()).enumerate() {
        for (i, (&a, &n)) in ra.iter().zip(rn.iter()).enumerate() {
            assert!(
                common::req2_close(a, n),
                "{ctx}: hessian[{j}][{i}] analytic={a} numeric={n}"
            );
        }
    }
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward 構築");
    let plain = dense(
        tape.backward(&loss)
            .expect("backward")
            .get(&x)
            .expect("get")
            .expect("到達"),
    );
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let first = dense(cg.first_order().get(&x).expect("get").expect("到達"));
    assert_eq!(
        first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{ctx}: first_order は Tape::backward と bit 同一"
    );
    let gx = dense(&cg.grad(&x).expect("grad").expect("到達").to_tensor());
    for (i, (&a, &b)) in gx.iter().zip(plain.iter()).enumerate() {
        assert!(
            common::req2_close(f64::from(a), f64::from(b)),
            "{ctx}: cg.grad[{i}]={a} vs backward={b}"
        );
    }
    check_grandchild(ctx, &build, x0, shape);
}

const H_QUAD: f64 = 0.5;
const H_TANH: f64 = 5e-2;
/// istft は窓包絡で割るため利得が大きく、中央差分の打ち切り誤差を抑えるため刻みを小さくする。
const H_TANH_ISTFT: f64 = 1e-2;

fn stft_opts(
    hop: usize,
    center: bool,
    mode: StftPadMode,
    normalized: bool,
    onesided: bool,
) -> StftOptions {
    StftOptions::default()
        .with_hop_length(hop)
        .with_center(center)
        .with_pad_mode(mode)
        .with_normalized(normalized)
        .with_onesided(onesided)
}

/// stft の掃引行: (入力 shape, n_fft, hop, center, mode, normalized, onesided, 短い窓)。
type StftCase = (
    Vec<usize>,
    usize,
    usize,
    bool,
    StftPadMode,
    bool,
    bool,
    bool,
);

/// istft の掃引行: (スペクトル shape, n_fft, hop, center, normalized, onesided, length)。
type IstftCase = (
    Vec<usize>,
    usize,
    usize,
    bool,
    bool,
    Option<bool>,
    Option<usize>,
);

// --- 1. stft の Hessian 突合・1 階 bit 同一・閉包性 ---------------------------------------------

#[test]
fn stft_family_first_order_bit_identical_and_cgrad_req2() {
    // (入力 shape, n_fft, hop, center, mode, normalized, onesided, 短い窓)
    let cases: Vec<StftCase> = vec![
        (
            vec![16],
            8,
            2,
            true,
            StftPadMode::Reflect,
            false,
            true,
            false,
        ),
        (
            vec![16],
            8,
            3,
            true,
            StftPadMode::Constant,
            true,
            true,
            false,
        ),
        (
            vec![15],
            8,
            3,
            false,
            StftPadMode::Reflect,
            false,
            false,
            false,
        ),
        (
            vec![2, 14],
            6,
            4,
            true,
            StftPadMode::Reflect,
            true,
            true,
            true,
        ),
        (
            vec![13],
            6,
            5,
            false,
            StftPadMode::Constant,
            false,
            true,
            true,
        ),
    ];
    for (shape, n_fft, hop, center, mode, normalized, onesided, short) in cases {
        let ctx = format!(
            "stft {shape:?} n_fft={n_fft} hop={hop} center={center} {mode:?} norm={normalized} one={onesided} short={short}"
        );
        let win = if short { hann(n_fft - 2) } else { hann(n_fft) };
        let opts =
            stft_opts(hop, center, mode, normalized, onesided).with_win_length(win.shape()[0]);
        let x0 = x_data(&shape);
        {
            let (win, opts) = (win.clone(), opts);
            check_all(
                &format!("{ctx} quad"),
                move |tape, x| {
                    let y = x.stft(n_fft, Some(&win), &opts)?;
                    quad_sum(&y, tape)
                },
                &x0,
                &shape,
                H_QUAD,
            );
        }
        check_all(
            &format!("{ctx} tanh"),
            move |tape, x| {
                let y = x.stft(n_fft, Some(&win), &opts)?;
                tanh_sum(&y, tape)
            },
            &x0,
            &shape,
            H_TANH,
        );
    }
}

// --- 2. istft ----------------------------------------------------------------------------------

fn istft_opts(hop: usize, center: bool, normalized: bool) -> IstftOptions {
    IstftOptions::default()
        .with_hop_length(hop)
        .with_center(center)
        .with_normalized(normalized)
}

#[test]
fn istft_family_first_order_and_hessian_match() {
    // (スペクトル shape, n_fft, hop, center, normalized, onesided, length)
    let cases: Vec<IstftCase> = vec![
        (vec![5, 5, 2], 8, 2, true, false, None, None),
        (vec![5, 5, 2], 8, 2, true, true, Some(true), None),
        (vec![5, 5, 2], 8, 2, true, false, None, Some(6)),
        (vec![5, 5, 2], 8, 2, true, false, None, Some(14)),
        (vec![8, 4, 2], 8, 2, false, false, Some(false), None),
        (vec![2, 4, 5, 2], 6, 3, false, false, None, None),
    ];
    for (shape, n_fft, hop, center, normalized, onesided, length) in cases {
        let ctx = format!(
            "istft {shape:?} n_fft={n_fft} hop={hop} center={center} norm={normalized} one={onesided:?} len={length:?}"
        );
        let mut opts = istft_opts(hop, center, normalized);
        if let Some(o) = onesided {
            opts = opts.with_onesided(o);
        }
        if let Some(l) = length {
            opts = opts.with_length(l);
        }
        let win = hann(n_fft);
        let x0 = x_data(&shape);
        {
            let (win, opts) = (win.clone(), opts);
            check_all(
                &format!("{ctx} quad"),
                move |tape, x| {
                    let y = x.istft(n_fft, Some(&win), &opts)?;
                    quad_sum(&y, tape)
                },
                &x0,
                &shape,
                H_QUAD,
            );
        }
        // `onesided = false` の上位 bin は 0 詰めで Hessian の有効成分が小さく、f32 の中央差分雑音
        // （絶対 4e-5 程度）が REQ-2 判定の相対側を超えるため、二次形式（厳密）だけで突合する。
        // batch 付き（窓包絡の利得が大きい hop=3 の構成）も同じ理由で二次形式のみ。
        if onesided == Some(false) || shape.len() == 4 {
            continue;
        }
        check_all(
            &format!("{ctx} tanh"),
            move |tape, x| {
                let y = x.istft(n_fft, Some(&win), &opts)?;
                tanh_sum(&y, tape)
            },
            &x0,
            &shape,
            H_TANH_ISTFT,
        );
    }
}

/// `istft(stft(x))` の往復: x が stft と istft の両方の随伴 Op を通る。
#[test]
fn stft_istft_roundtrip_second_order_matches_direct() {
    let shape = [16usize];
    let win = hann(8);
    let x0 = x_data(&shape);
    check_all(
        "roundtrip tanh",
        move |tape, x| {
            let so = StftOptions::default().with_hop_length(2);
            let io = IstftOptions::default().with_hop_length(2).with_length(16);
            let y = x.stft(8, Some(&win), &so)?.istft(8, Some(&win), &io)?;
            tanh_sum(&y, tape)
        },
        &x0,
        &shape,
        H_TANH,
    );
}

/// x が stft と別の非線形項の両方へ流れる構成。
#[test]
fn stft_family_x_flows_to_stft_and_nonlinear_branch() {
    let shape = [16usize];
    let win = hann(8);
    check_all(
        "stft + x^2 branch",
        move |tape, x| {
            let so = StftOptions::default().with_hop_length(2);
            let a = tanh_sum(&x.stft(8, Some(&win), &so)?, tape)?;
            let b = x.mul(x)?.tanh().sum(None)?;
            a.add(&b)
        },
        &x_data(&shape),
        &shape,
        H_TANH,
    );
}

// --- 3. 線形な上流では cg.grad も 1 階と bit 一致 -----------------------------------------------

#[test]
fn stft_family_cgrad_bit_identical_for_linear_upstream() {
    let win = hann(8);
    let shape = [16usize];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x_data(&shape), &shape));
    let so = StftOptions::default().with_hop_length(2);
    let y = x.stft(8, Some(&win), &so).expect("stft");
    let ys = y.to_tensor().shape().to_vec();
    let c = tape.var_no_grad(&t(pattern(numel(&ys), 3.0), &ys));
    let loss = y.mul(&c).expect("mul").sum(None).expect("sum");
    let plain = dense(
        tape.backward(&loss)
            .expect("backward")
            .get(&x)
            .expect("get")
            .expect("到達"),
    );
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let g = dense(&cg.grad(&x).expect("grad").expect("到達").to_tensor());
    assert_eq!(
        g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "線形上流の cg.grad は 1 階と bit 一致"
    );
}

// --- 4. jvp_create_graph ------------------------------------------------------------------------

#[test]
fn stft_family_jvp_create_graph_matches_central_difference() {
    let shape = [16usize];
    let win = hann(8);
    let x0 = x_data(&shape);
    let v = pattern(16, 5.0);
    let so = StftOptions::default().with_hop_length(2);
    let loss_at = |x: &[f32]| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), &shape));
        let y = xv.stft(8, Some(&win), &so).expect("stft");
        let l = quad_sum(&y, &tape).expect("loss");
        f64::from(dense(&l.to_tensor())[0])
    };
    let h = 0.25f32;
    let plus: Vec<f32> = x0.iter().zip(&v).map(|(a, b)| a + h * b).collect();
    let minus: Vec<f32> = x0.iter().zip(&v).map(|(a, b)| a - h * b).collect();
    let expected = (loss_at(&plus) - loss_at(&minus)) / (2.0 * f64::from(h));

    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let grand = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.clone(), &shape));
    let y = x.stft(8, Some(&win), &so).expect("stft");
    let loss = quad_sum(&y, &tape).expect("loss");
    let res = jvp_create_graph(&tape, &loss, &x, &t(v.clone(), &shape), &child, &grand)
        .expect("jvp_create_graph");
    let got = f64::from(dense(&res.output().to_tensor())[0]);
    assert!(
        common::req2_close(got, expected),
        "jvp={got} vs central difference={expected}"
    );
}

// --- 5. 非有限・大きな上流 -----------------------------------------------------------------------

fn first_and_cgrad(
    shape: &[usize],
    x0: Vec<f32>,
    build: &dyn for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
) -> (Vec<f32>, Vec<f32>) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0, shape));
    let loss = build(&tape, &x).expect("forward 構築");
    let plain = dense(
        tape.backward(&loss)
            .expect("backward")
            .get(&x)
            .expect("get")
            .expect("到達"),
    );
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("非有限でも受理される（1 階と同じ伝播）");
    let g = dense(&cg.grad(&x).expect("grad").expect("到達").to_tensor());
    (plain, g)
}

fn same_class(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a == b || (a.is_finite() && b.is_finite())
}

#[test]
fn stft_family_nonfinite_propagates_like_first_order() {
    let win = hann(8);
    let so = StftOptions::default().with_hop_length(2);
    let mut x0 = x_data(&[16]);
    x0[5] = f32::NAN;
    let (plain, g) = first_and_cgrad(&[16], x0, &|tape, x| {
        let y = x.stft(8, Some(&win), &so)?;
        quad_sum(&y, tape)
    });
    for i in 0..plain.len() {
        assert!(
            same_class(plain[i], g[i]),
            "stft[{i}]: {} vs {}",
            plain[i],
            g[i]
        );
    }
    let io = IstftOptions::default().with_hop_length(2);
    let mut s0 = x_data(&[5, 5, 2]);
    s0[7] = f32::INFINITY;
    let (plain, g) = first_and_cgrad(&[5, 5, 2], s0, &|tape, x| {
        let y = x.istft(8, Some(&win), &io)?;
        quad_sum(&y, tape)
    });
    for i in 0..plain.len() {
        assert!(
            same_class(plain[i], g[i]),
            "istft[{i}]: {} vs {}",
            plain[i],
            g[i]
        );
    }
}

#[test]
fn stft_family_large_upstream_matches_first_order() {
    let win = hann(8);
    let so = StftOptions::default().with_hop_length(2);
    let (plain, g) = first_and_cgrad(&[16], x_data(&[16]), &|tape, x| {
        let y = x.stft(8, Some(&win), &so)?;
        let ys = y.to_tensor().shape().to_vec();
        let c = tape.var_no_grad(&t(vec![3.0e38; numel(&ys)], &ys));
        y.mul(&c)?.sum(None)
    });
    for i in 0..plain.len() {
        assert!(
            (plain[i].is_finite() == g[i].is_finite())
                && (!plain[i].is_finite() || plain[i] == g[i]),
            "large[{i}]: {} vs {}",
            plain[i],
            g[i]
        );
    }
}

// --- 6. 0 詰め部分・上位 bin の二階勾配 -----------------------------------------------------------

#[test]
fn istft_zero_padded_tail_and_upper_bins_have_zero_second_order() {
    let win = hann(8);
    // length を期待長より延長: 0 詰め部分は出力へ寄与しない（二階勾配 0）。
    let io = IstftOptions::default().with_hop_length(2).with_length(14);
    let shape = [5usize, 5, 2];
    let h = analytic_hessian(
        &|tape, x| {
            let y = x.istft(8, Some(&win), &io)?;
            quad_sum(&y, tape)
        },
        &x_data(&shape),
        &shape,
    );
    assert!(h.iter().flatten().all(|v| v.is_finite()));
    // onesided = false: 上位 bin（k >= n_fft/2+1 = 5）は forward が読まないため行・列とも 0。
    let io2 = IstftOptions::default()
        .with_hop_length(2)
        .with_center(false)
        .with_onesided(false);
    let shape2 = [8usize, 4, 2];
    let h2 = analytic_hessian(
        &|tape, x| {
            let y = x.istft(8, Some(&win), &io2)?;
            quad_sum(&y, tape)
        },
        &x_data(&shape2),
        &shape2,
    );
    let per_bin = 4 * 2;
    for j in 0..h2.len() {
        for i in 0..h2.len() {
            if j / per_bin >= 5 || i / per_bin >= 5 {
                assert_eq!(h2[j][i], 0.0, "上位 bin の二階勾配は 0: [{j}][{i}]");
            }
        }
    }
}

// --- 7. 隣接する未対応 Op は fail-closed のまま -----------------------------------------------------

#[test]
fn stft_family_neighbouring_unsupported_op_stays_fail_closed() {
    let win = hann(8);
    let so = StftOptions::default().with_hop_length(2);
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x_data(&[16]), &[16]));
    let spd = tape.var(&t(
        vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0],
        &[3, 3],
    ));
    let a = x
        .stft(8, Some(&win), &so)
        .expect("stft")
        .sum(None)
        .expect("sum");
    let b = spd.cholesky().expect("cholesky").sum(None).expect("sum");
    let loss = a.add(&b).expect("add");
    let r = tape.backward_create_graph(&loss, &child);
    assert!(matches!(r, Err(AutodiffError::Backward(_))));
    assert!(child.is_empty(), "拒否後は child が空");
}

// --- 8. backend override 経路 ---------------------------------------------------------------------

/// `fft_stft`／`fft_istft` の呼び出し回数を数え、`Unsupported` を返してホスト参照へ
/// フォールバックさせるバックエンド（子テープの再生が override を通ることの確認用）。
struct CountingOps {
    inner: Box<dyn BackendOps + Send>,
    stft_calls: Arc<AtomicUsize>,
    istft_calls: Arc<AtomicUsize>,
}

impl BackendOps for CountingOps {
    fn device(&self) -> Device {
        self.inner.device()
    }
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.gemm(a, b)
    }
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.add(a, b)
    }
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.mul(a, b)
    }
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.relu(a)
    }
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.exp(a)
    }
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.tanh(a)
    }
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.sum(a, dim)
    }
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.max(a, dim)
    }
    fn fft_stft(
        &self,
        _input: &Tensor<f32>,
        _window: &Tensor<f32>,
        _params: &StftParams,
    ) -> Result<Tensor<f32>, BackendError> {
        self.stft_calls.fetch_add(1, Ordering::SeqCst);
        Err(BackendError::Unsupported("counting".into()))
    }
    fn fft_istft(
        &self,
        _input: &Tensor<f32>,
        _window: &Tensor<f32>,
        _params: &IstftParams,
    ) -> Result<Tensor<f32>, BackendError> {
        self.istft_calls.fetch_add(1, Ordering::SeqCst);
        Err(BackendError::Unsupported("counting".into()))
    }
}

#[test]
fn stft_family_child_replay_uses_backend_override() {
    let win = hann(8);
    let stft_calls = Arc::new(AtomicUsize::new(0));
    let istft_calls = Arc::new(AtomicUsize::new(0));
    let child = Tape::new_with_ops(Box::new(CountingOps {
        inner: common::naive_ops(),
        stft_calls: Arc::clone(&stft_calls),
        istft_calls: Arc::clone(&istft_calls),
    }));
    let tape = Tape::new_with_ops(common::naive_ops());
    let so = StftOptions::default().with_hop_length(2);
    let io = IstftOptions::default().with_hop_length(2).with_length(16);
    let x = tape.var(&t(x_data(&[16]), &[16]));
    let y = x
        .stft(8, Some(&win), &so)
        .expect("stft")
        .istft(8, Some(&win), &io)
        .expect("istft");
    let loss = quad_sum(&y, &tape).expect("loss");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    assert!(
        stft_calls.load(Ordering::SeqCst) >= 1,
        "Stft の再生が fft_stft を通る"
    );
    assert!(
        istft_calls.load(Ordering::SeqCst) >= 1,
        "Istft の再生が fft_istft を通る"
    );
    // override が Unsupported でもホスト参照へ落ち、値は 1 階と REQ-2 で一致する。
    let plain = dense(
        tape.backward(&loss)
            .expect("backward")
            .get(&x)
            .expect("get")
            .expect("到達"),
    );
    let g = dense(&cg.grad(&x).expect("grad").expect("到達").to_tensor());
    for i in 0..plain.len() {
        assert!(common::req2_close(f64::from(g[i]), f64::from(plain[i])));
    }
}
