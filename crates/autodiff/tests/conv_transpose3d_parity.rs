//! `conv_transpose3d_ops::conv_transpose3d`（イシュー #2644）の `Tape`／`Var` を経由する
//! end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/conv-transpose3d-max-unpool-pytorch-reference/
//!   conv_transpose3d_max_unpool_reference.json`・生成条件は同ディレクトリの `README.md`）と
//!   REQ-2 統一複合判定（`common::req2_close`。tolerance 定数は新設しない。PyTorch と本実装は
//!   総和順が異なるため bit 一致は求めない）で突合する。
//! - 独立オラクル: 中心差分（`tests/conv_transpose2d.rs` と同じ `H=1e-3`・相対 1e-2 または絶対
//!   1e-3・`τ=1e-4`。緩和しない）・随伴恒等式（`conv3d` の VJP／forward との相互関係）・
//!   per-group `narrow`＋`groups=1` の合成。
//! - フォールバックは `Unsupported` のときだけで、他のバックエンドエラーは握りつぶさず伝播し、
//!   誤 shape は型付きエラーで拒否する（`col2im3d`＝forward・`im2col3d`＝VJP の両フック）。
//! - 引数検査・孤児ノード無し・`N = 0`・確保前サイズ検査・テープ記録数・決定性も固定する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::conv_transpose3d_ops::conv_transpose3d;
use fandhe_ai_autodiff::conv3d_ops::conv3d;
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, Conv3dParams, ShapeError, Tensor};
use serde::Deserialize;

const H: f64 = 1e-3;
const TAU: f64 = 1e-4;
const REL_TOL: f64 = 1e-2;
const ABS_TOL: f64 = 1e-3;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    tensor.host_slice().into_owned()
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fn arr3(v: &[usize]) -> [usize; 3] {
    [v[0], v[1], v[2]]
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// 決定的な擬似データ（`sin`／`cos` 由来。テストごとに位相を変える）。
fn wave(n: usize, freq: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * freq).sin() * amp).collect()
}

#[derive(Clone, Copy)]
struct Cfg {
    stride: [usize; 3],
    padding: [usize; 3],
    op: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
}

const UNIT: Cfg = Cfg {
    stride: [1; 3],
    padding: [0; 3],
    op: [0; 3],
    dilation: [1; 3],
    groups: 1,
};

// --- fixture ---

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    conv_transpose3d_cases: Vec<Case>,
    conv_transpose3d_error_cases: Vec<ErrCase>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    in_shape: Vec<usize>,
    weight_shape: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,
    output_padding: Vec<usize>,
    dilation: Vec<usize>,
    groups: usize,
    out_shape: Vec<usize>,
    x_bits: Vec<u32>,
    w_bits: Vec<u32>,
    b_bits: Option<Vec<u32>>,
    g_bits: Vec<u32>,
    out_bits: Vec<u32>,
    dx_bits: Vec<u32>,
    dw_bits: Vec<u32>,
    db_bits: Option<Vec<u32>>,
}

#[derive(Deserialize)]
struct ErrCase {
    name: String,
    torch_raises: bool,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/conv-transpose3d-max-unpool-pytorch-reference/conv_transpose3d_max_unpool_reference.json",
    );
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

fn assert_close_all(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            common::req2_close(f64::from(a), f64::from(e)),
            "{context}[{i}]: actual={a} expected={e}"
        );
    }
}

struct Run {
    out: Tensor<f32>,
    dx: Vec<f32>,
    dw: Vec<f32>,
    db: Option<Vec<f32>>,
}

/// `(out * g).sum()` を損失とした forward 値と全入力の勾配を返す。
fn run(
    ops: Box<dyn BackendOps + Send>,
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    b: Option<&Tensor<f32>>,
    g: Option<&Tensor<f32>>,
    cfg: Cfg,
) -> Run {
    let tape = Tape::new_with_ops(ops);
    let xv = tape.var(x);
    let wv = tape.var(w);
    let bv = b.map(|b| tape.var(b));
    let y = conv_transpose3d(
        &xv,
        &wv,
        bv.as_ref(),
        cfg.stride,
        cfg.padding,
        cfg.op,
        cfg.dilation,
        cfg.groups,
    )
    .expect("conv_transpose3d");
    let out = y.to_tensor();
    let g_default;
    let g = match g {
        Some(g) => g,
        None => {
            g_default = t(wave(numel(out.shape()), 0.053, 0.7), out.shape());
            &g_default
        }
    };
    let gv = tape.var_no_grad(g);
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    Run {
        dx: dense(grads.get(&xv).unwrap().expect("x へ勾配が届く")),
        dw: dense(grads.get(&wv).unwrap().expect("w へ勾配が届く")),
        db: bv
            .as_ref()
            .map(|bv| dense(grads.get(bv).unwrap().expect("b へ勾配が届く"))),
        out,
    }
}

// --- 1. PyTorch fixture 突合 ---

#[test]
fn fixture_forward_and_gradients_match_pytorch_reference() {
    let fixture = load_fixture();
    assert!(
        fixture.torch_version.starts_with("2.14.0"),
        "fixture は PyTorch 2.14.0 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    assert!(fixture.conv_transpose3d_cases.len() >= 10);
    let mut with_bias = 0;
    for c in &fixture.conv_transpose3d_cases {
        let cfg = Cfg {
            stride: arr3(&c.stride),
            padding: arr3(&c.padding),
            op: arr3(&c.output_padding),
            dilation: arr3(&c.dilation),
            groups: c.groups,
        };
        let x = t(from_bits(&c.x_bits), &c.in_shape);
        let w = t(from_bits(&c.w_bits), &c.weight_shape);
        let b = c
            .b_bits
            .as_ref()
            .map(|b| t(from_bits(b), &[c.out_shape[1]]));
        let g = t(from_bits(&c.g_bits), &c.out_shape);
        let r = run(common::naive_ops(), &x, &w, b.as_ref(), Some(&g), cfg);
        assert_eq!(
            r.out.shape(),
            c.out_shape.as_slice(),
            "{}: 出力 shape",
            c.name
        );
        assert_close_all(
            &dense(&r.out),
            &from_bits(&c.out_bits),
            &format!("{} forward", c.name),
        );
        assert_close_all(&r.dx, &from_bits(&c.dx_bits), &format!("{} dx", c.name));
        assert_close_all(&r.dw, &from_bits(&c.dw_bits), &format!("{} dw", c.name));
        match (&r.db, &c.db_bits) {
            (Some(got), Some(want)) => {
                with_bias += 1;
                assert_close_all(got, &from_bits(want), &format!("{} db", c.name));
            }
            (None, None) => {}
            _ => panic!("{}: bias の有無が fixture と食い違う", c.name),
        }
    }
    assert!(with_bias >= 6, "bias 付きケースが少なすぎる: {with_bias}");
}

/// 引数エラー 1 件の仕様（x／w／bias の shape と引数）。
struct ErrSpec {
    x: Vec<usize>,
    w: Vec<usize>,
    bias: Option<usize>,
    cfg: Cfg,
}

fn err_spec(name: &str) -> (ErrSpec, bool, bool) {
    // 戻り値: (仕様, PyTorch 2.14.0 が拒否するか, 本実装が受理するか)。
    let base = |x: &[usize], w: &[usize]| ErrSpec {
        x: x.to_vec(),
        w: w.to_vec(),
        bias: None,
        cfg: UNIT,
    };
    match name {
        "output_padding_eq_stride" => {
            let mut s = base(&[1, 1, 2, 2, 2], &[1, 1, 2, 2, 2]);
            s.cfg.op = [1; 3];
            (s, true, false)
        }
        // PyTorch は `output_padding < max(stride, dilation)` まで許容する（意図的な差分）。
        "output_padding_lt_dilation_ge_stride" => {
            let mut s = base(&[1, 1, 2, 2, 2], &[1, 1, 2, 2, 2]);
            s.cfg.dilation = [2; 3];
            s.cfg.op = [1; 3];
            (s, false, false)
        }
        "channel_mismatch" => (base(&[1, 2, 2, 2, 2], &[3, 1, 2, 2, 2]), true, false),
        "groups_not_divide_cin" => {
            let mut s = base(&[1, 3, 2, 2, 2], &[3, 1, 2, 2, 2]);
            s.cfg.groups = 2;
            (s, true, false)
        }
        "stride_zero" => {
            let mut s = base(&[1, 1, 2, 2, 2], &[1, 1, 2, 2, 2]);
            s.cfg.stride = [0; 3];
            (s, true, false)
        }
        "dilation_zero" => {
            let mut s = base(&[1, 1, 2, 2, 2], &[1, 1, 2, 2, 2]);
            s.cfg.dilation = [0; 3];
            (s, true, false)
        }
        "groups_zero" => {
            let mut s = base(&[1, 1, 2, 2, 2], &[1, 1, 2, 2, 2]);
            s.cfg.groups = 0;
            (s, true, false)
        }
        // PyTorch はバッチなし入力（rank 4）を受理する。本実装は rank 5 のみ（意図的な差分）。
        "input_rank4_batchless" => (base(&[1, 2, 2, 2], &[1, 1, 2, 2, 2]), false, false),
        "padding_too_large" => {
            let mut s = base(&[1, 1, 2, 2, 2], &[1, 1, 2, 2, 2]);
            s.cfg.padding = [5; 3];
            (s, true, false)
        }
        "bias_shape_mismatch" => {
            let mut s = base(&[1, 1, 2, 2, 2], &[1, 2, 2, 2, 2]);
            s.bias = Some(3);
            (s, true, false)
        }
        // `N = 0` は両方とも受理する。
        "batch_zero" => (base(&[0, 1, 2, 2, 2], &[1, 1, 2, 2, 2]), false, true),
        other => panic!("未知の error case: {other}"),
    }
}

#[test]
fn fixture_error_cases_agree_with_torch_except_documented_differences() {
    let fixture = load_fixture();
    assert!(fixture.conv_transpose3d_error_cases.len() >= 11);
    for c in &fixture.conv_transpose3d_error_cases {
        let (spec, torch_raises, ours_ok) = err_spec(&c.name);
        assert_eq!(
            c.torch_raises, torch_raises,
            "{}: fixture の torch_raises が期待表と食い違う（再生成後は表を更新する）",
            c.name
        );
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![0.5; numel(&spec.x)], &spec.x));
        let w = tape.var(&t(vec![0.25; numel(&spec.w)], &spec.w));
        let b = spec.bias.map(|n| tape.var(&t(vec![0.0; n], &[n])));
        let r = conv_transpose3d(
            &x,
            &w,
            b.as_ref(),
            spec.cfg.stride,
            spec.cfg.padding,
            spec.cfg.op,
            spec.cfg.dilation,
            spec.cfg.groups,
        );
        assert_eq!(r.is_ok(), ours_ok, "{}: 本実装の受理可否", c.name);
    }
}

// --- 2. 独立オラクル: 中心差分 ---

fn numeric_grad(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    b: Option<&Tensor<f32>>,
    s: &Tensor<f32>,
    cfg: Cfg,
    param: usize,
) -> Vec<f64> {
    let forward = |x: &Tensor<f32>, w: &Tensor<f32>, b: Option<&Tensor<f32>>| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        let wv = tape.var(w);
        let bv = b.map(|bt| tape.var(bt));
        let y = conv_transpose3d(
            &xv,
            &wv,
            bv.as_ref(),
            cfg.stride,
            cfg.padding,
            cfg.op,
            cfg.dilation,
            cfg.groups,
        )
        .unwrap();
        dense(&y.to_tensor())
            .iter()
            .zip(dense(s).iter())
            .map(|(&yv, &sv)| f64::from(yv) * f64::from(sv))
            .sum()
    };
    let target = match param {
        0 => x,
        1 => w,
        _ => b.expect("param=2 requires bias"),
    };
    let shape = target.shape().to_vec();
    let mut data = dense(target);
    let mut grad = vec![0f64; data.len()];
    for i in 0..data.len() {
        let orig = f64::from(data[i]);
        let mut eval = |delta: f64| {
            data[i] = (orig + delta) as f32;
            let m = t(data.clone(), &shape);
            match param {
                0 => forward(&m, w, b),
                1 => forward(x, &m, b),
                _ => forward(x, w, Some(&m)),
            }
        };
        let lp = eval(H);
        let lm = eval(-H);
        data[i] = orig as f32;
        grad[i] = (lp - lm) / (2.0 * H);
    }
    grad
}

fn assert_grad_close(label: &str, analytic: &[f32], numeric: &[f64]) {
    assert_eq!(analytic.len(), numeric.len(), "{label}: 要素数不一致");
    for (i, (&av, &nv)) in analytic.iter().zip(numeric).enumerate() {
        let av64 = f64::from(av);
        let diff = (av64 - nv).abs();
        let rel = diff / av64.abs().max(nv.abs()).max(TAU);
        assert!(
            rel <= REL_TOL || diff <= ABS_TOL,
            "{label}[{i}]: analytic={av64} numeric={nv} diff={diff} rel={rel}"
        );
    }
}

fn check_numeric(x_shape: &[usize], w_shape: &[usize], with_bias: bool, cfg: Cfg) {
    let x = t(wave(numel(x_shape), 0.037, 1.0), x_shape);
    let w = t(wave(numel(w_shape), 0.091, 0.5), w_shape);
    let cout = w_shape[1] * cfg.groups;
    let b = with_bias.then(|| t((0..cout).map(|i| 0.1 * i as f32 - 0.2).collect(), &[cout]));
    let g_shape = {
        let tape = Tape::new_with_ops(common::naive_ops());
        let y = conv_transpose3d(
            &tape.var(&x),
            &tape.var(&w),
            None,
            cfg.stride,
            cfg.padding,
            cfg.op,
            cfg.dilation,
            cfg.groups,
        )
        .unwrap();
        y.to_tensor().shape().to_vec()
    };
    let s = t(wave(numel(&g_shape), 0.053, 0.7), &g_shape);
    let r = run(common::naive_ops(), &x, &w, b.as_ref(), Some(&s), cfg);
    assert_grad_close("dx", &r.dx, &numeric_grad(&x, &w, b.as_ref(), &s, cfg, 0));
    assert_grad_close("dw", &r.dw, &numeric_grad(&x, &w, b.as_ref(), &s, cfg, 1));
    if with_bias {
        assert_grad_close(
            "db",
            r.db.as_ref().unwrap(),
            &numeric_grad(&x, &w, b.as_ref(), &s, cfg, 2),
        );
    }
}

#[test]
fn backward_matches_numeric_grad_basic_with_bias() {
    check_numeric(&[1, 2, 2, 3, 3], &[2, 3, 2, 2, 2], true, UNIT);
}

#[test]
fn backward_matches_numeric_grad_stride_padding_output_padding() {
    let cfg = Cfg {
        stride: [2, 2, 2],
        padding: [1, 1, 1],
        op: [1, 1, 1],
        ..UNIT
    };
    check_numeric(&[1, 2, 2, 2, 3], &[2, 2, 3, 3, 3], true, cfg);
}

#[test]
fn backward_matches_numeric_grad_groups_dilation_no_bias() {
    let cfg = Cfg {
        dilation: [2, 1, 2],
        groups: 2,
        ..UNIT
    };
    check_numeric(&[1, 4, 2, 2, 2], &[4, 2, 2, 2, 2], false, cfg);
}

#[test]
fn backward_matches_numeric_grad_anisotropic_batch2() {
    let cfg = Cfg {
        stride: [1, 2, 2],
        padding: [0, 1, 0],
        op: [0, 1, 1],
        dilation: [1, 1, 2],
        groups: 1,
    };
    check_numeric(&[2, 2, 2, 2, 3], &[2, 2, 2, 3, 2], true, cfg);
}

// --- 3. 独立オラクル: 随伴恒等式（conv3d との相互関係） ---

#[test]
fn forward_equals_conv3d_vjp_input_gradient() {
    // conv3d(a [1,2,5,5,5], W [3,2,3,3,3], s=2, p=1) -> [1,3,3,3,3]。上流を x として a へ流れる勾配は
    // conv_transpose3d(x, W)（output_padding=0）と一致する。
    let (stride, padding) = ([2; 3], [1; 3]);
    let a = t(wave(numel(&[1, 2, 5, 5, 5]), 0.021, 1.0), &[1, 2, 5, 5, 5]);
    let w = t(wave(numel(&[3, 2, 3, 3, 3]), 0.083, 0.5), &[3, 2, 3, 3, 3]);
    let x = t(wave(numel(&[1, 3, 3, 3, 3]), 0.047, 1.0), &[1, 3, 3, 3, 3]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let av = tape.var(&a);
    let wv = tape.var_no_grad(&w);
    let y = conv3d(&av, &wv, None, stride, padding, [1; 3], 1).unwrap();
    assert_eq!(y.to_tensor().shape(), &[1, 3, 3, 3, 3]);
    let xv = tape.var_no_grad(&x);
    let loss = y.mul(&xv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let want = dense(grads.get(&av).unwrap().expect("a へ勾配が届く"));

    let tape2 = Tape::new_with_ops(common::naive_ops());
    let got = conv_transpose3d(
        &tape2.var(&x),
        &tape2.var(&w),
        None,
        stride,
        padding,
        [0; 3],
        [1; 3],
        1,
    )
    .unwrap()
    .to_tensor();
    assert_eq!(got.shape(), &[1, 2, 5, 5, 5]);
    assert_close_all(&dense(&got), &want, "conv_transpose3d vs conv3d VJP");
}

#[test]
fn input_gradient_equals_conv3d_forward_of_upstream() {
    // d_input(conv_transpose3d)(g) = conv3d(g, W)（`Op::ConvTranspose3d` VJP の核心関係）。
    let cfg = Cfg {
        stride: [2; 3],
        padding: [1; 3],
        ..UNIT
    };
    let x = t(wave(numel(&[1, 3, 3, 3, 3]), 0.047, 1.0), &[1, 3, 3, 3, 3]);
    let w = t(wave(numel(&[3, 2, 3, 3, 3]), 0.083, 0.5), &[3, 2, 3, 3, 3]);
    let g = t(wave(numel(&[1, 2, 5, 5, 5]), 0.031, 0.9), &[1, 2, 5, 5, 5]);
    let r = run(common::naive_ops(), &x, &w, None, Some(&g), cfg);

    let tape = Tape::new_with_ops(common::naive_ops());
    let want = conv3d(
        &tape.var(&g),
        &tape.var(&w),
        None,
        cfg.stride,
        cfg.padding,
        [1; 3],
        1,
    )
    .unwrap()
    .to_tensor();
    assert_eq!(want.shape(), &[1, 3, 3, 3, 3]);
    assert_close_all(&r.dx, &dense(&want), "d_input vs conv3d(g, W)");
}

// --- 4. groups・bias 軸回帰・shape ---

#[test]
fn groups_equal_per_group_composition() {
    let cfg = Cfg { groups: 2, ..UNIT };
    let x = t(wave(numel(&[1, 4, 2, 3, 3]), 0.05, 1.0), &[1, 4, 2, 3, 3]);
    let w = t(wave(numel(&[4, 3, 2, 2, 2]), 0.07, 0.6), &[4, 3, 2, 2, 2]);
    let full = run(common::naive_ops(), &x, &w, None, None, cfg).out;
    assert_eq!(full.shape(), &[1, 6, 3, 4, 4]);
    for g in 0..2 {
        let xg = x.narrow(1, g * 2, 2).unwrap().contiguous();
        let wg = w.narrow(0, g * 2, 2).unwrap().contiguous();
        let part = run(common::naive_ops(), &xg, &wg, None, None, UNIT).out;
        let want = full.narrow(1, g * 3, 3).unwrap().contiguous();
        assert_close_all(&dense(&part), &dense(&want), &format!("group {g}"));
    }
}

#[test]
fn bias_adds_to_cout_axis_not_wout_axis_when_equal() {
    // Cout=3・Wout=3 の形状。右詰め broadcast なら W 軸へ誤加算されてしまう。
    let x = t(vec![0.0; 16], &[1, 2, 2, 2, 2]);
    let w = t(vec![0.0; 2 * 3 * 8], &[2, 3, 2, 2, 2]);
    let b = t(vec![1.0, 2.0, 3.0], &[3]);
    let out = run(common::naive_ops(), &x, &w, Some(&b), None, UNIT).out;
    assert_eq!(out.shape(), &[1, 3, 3, 3, 3]);
    let data = dense(&out);
    for (c, chunk) in data.chunks(27).enumerate() {
        assert!(chunk.iter().all(|&v| v == (c + 1) as f32), "channel {c}");
    }
}

#[test]
fn output_shape_follows_pytorch_formula() {
    // (in-1)*s - 2p + d*(k-1) + op + 1。
    let cfg = Cfg {
        stride: [2, 3, 1],
        padding: [1, 0, 2],
        op: [1, 2, 0],
        dilation: [1, 2, 3],
        groups: 1,
    };
    let x = t(vec![0.0; 3 * 2 * 4], &[1, 1, 3, 2, 4]);
    let w = t(vec![0.0; 18], &[1, 1, 3, 2, 3]);
    let out = run(common::naive_ops(), &x, &w, None, None, cfg).out;
    // D: 2*2-2+1*2+1+1=6, H: 1*3-0+2*1+2+1=8, W: 3*1-4+3*2+0+1=6。
    assert_eq!(out.shape(), &[1, 1, 6, 8, 6]);
}

// --- 5. 引数検査・境界 ---

#[test]
fn invalid_arguments_are_typed_errors_and_leave_no_orphan_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 8], &[1, 1, 2, 2, 2]));
    let w = tape.var(&t(vec![0.0; 8], &[1, 1, 2, 2, 2]));
    let before = tape.len();
    let call = |s: [usize; 3], op: [usize; 3], d: [usize; 3], g: usize| {
        conv_transpose3d(&x, &w, None, s, [0; 3], op, d, g).map(|_| ())
    };
    assert!(matches!(
        call([1; 3], [1, 0, 0], [1; 3], 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        call([2, 2, 2], [0, 0, 2], [1; 3], 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
    for r in [
        call([0, 1, 1], [0; 3], [1; 3], 1),
        call([1; 3], [0; 3], [1, 0, 1], 1),
        call([1; 3], [0; 3], [1; 3], 0),
    ] {
        assert!(matches!(
            r,
            Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
        ));
    }
    // rank 不一致（weight rank 4）・チャンネル不整合・bias shape。
    let w4 = tape.var(&t(vec![0.0; 4], &[1, 1, 2, 2]));
    assert!(matches!(
        conv_transpose3d(&x, &w4, None, [1; 3], [0; 3], [0; 3], [1; 3], 1),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    let w_bad_cin = tape.var(&t(vec![0.0; 16], &[2, 1, 2, 2, 2]));
    assert!(matches!(
        conv_transpose3d(&x, &w_bad_cin, None, [1; 3], [0; 3], [0; 3], [1; 3], 1),
        Err(AutodiffError::Shape(_))
    ));
    let bad_bias = tape.var(&t(vec![0.0; 2], &[2]));
    assert!(matches!(
        conv_transpose3d(&x, &w, Some(&bad_bias), [1; 3], [0; 3], [0; 3], [1; 3], 1),
        Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
    ));
    // 空間軸 0。
    let z = tape.var(&t(vec![], &[1, 1, 0, 2, 2]));
    assert!(matches!(
        conv_transpose3d(&z, &w, None, [1; 3], [0; 3], [0; 3], [1; 3], 1),
        Err(AutodiffError::Shape(_))
    ));
    // 別 tape の weight。
    let other = Tape::new_with_ops(common::naive_ops());
    let ow = other.var(&t(vec![0.0; 8], &[1, 1, 2, 2, 2]));
    assert!(conv_transpose3d(&x, &ow, None, [1; 3], [0; 3], [0; 3], [1; 3], 1).is_err());
    // 検査は tape 操作より前: 増えたのは自分で作った入力ノード 4 件（w4・w_bad_cin・bad_bias・z）のみ。
    assert_eq!(tape.len(), before + 4);
}

#[test]
fn empty_batch_is_accepted_and_backward_yields_zero_gradients() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![], &[0, 2, 2, 2, 2]));
    let w = tape.var(&t(wave(2 * 3 * 8, 0.1, 1.0), &[2, 3, 2, 2, 2]));
    let b = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let y = conv_transpose3d(&x, &w, Some(&b), [1; 3], [0; 3], [0; 3], [1; 3], 1).unwrap();
    assert_eq!(y.to_tensor().shape(), &[0, 3, 3, 3, 3]);
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    assert!(
        dense(grads.get(&w).unwrap().unwrap())
            .iter()
            .all(|&v| v == 0.0)
    );
    assert!(
        dense(grads.get(&b).unwrap().unwrap())
            .iter()
            .all(|&v| v == 0.0)
    );
}

#[test]
fn huge_broadcast_view_is_rejected_before_allocation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let base = tape.var(&t(vec![1.0], &[1, 1, 1, 1, 1]));
    let huge = base.broadcast_to(&[1, 1, 1, 1, 1usize << 61]).unwrap();
    let w = tape.var(&t(vec![1.0], &[1, 1, 1, 1, 1]));
    assert!(matches!(
        conv_transpose3d(&huge, &w, None, [1; 3], [0; 3], [0; 3], [1; 3], 1),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
}

#[test]
fn records_exactly_one_node_and_rejects_create_graph() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(wave(8, 0.3, 1.0), &[1, 1, 2, 2, 2]));
    let w = tape.var(&t(wave(8, 0.2, 1.0), &[1, 1, 2, 2, 2]));
    let before = tape.len();
    let y = conv_transpose3d(&x, &w, None, [1; 3], [0; 3], [0; 3], [1; 3], 1).unwrap();
    assert_eq!(tape.len(), before + 1);
    let loss = y.sum(None).unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
}

#[test]
fn results_are_bit_deterministic_run_to_run() {
    let cfg = Cfg {
        stride: [2; 3],
        padding: [1; 3],
        op: [1; 3],
        groups: 2,
        ..UNIT
    };
    let x = t(wave(numel(&[2, 4, 3, 3, 3]), 0.043, 1.0), &[2, 4, 3, 3, 3]);
    let w = t(wave(numel(&[4, 2, 3, 3, 3]), 0.061, 0.5), &[4, 2, 3, 3, 3]);
    let b = t(vec![0.1, -0.2, 0.3, 0.4], &[4]);
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    let a = run(common::naive_ops(), &x, &w, Some(&b), None, cfg);
    let c = run(common::naive_ops(), &x, &w, Some(&b), None, cfg);
    assert_eq!(bits(&dense(&a.out)), bits(&dense(&c.out)));
    assert_eq!(bits(&a.dx), bits(&c.dx));
    assert_eq!(bits(&a.dw), bits(&c.dw));
    assert_eq!(a.db.as_deref().map(bits), c.db.as_deref().map(bits));
}

// --- 6. フォールバックとエラー伝播（col2im3d＝forward・im2col3d＝VJP） ---

#[derive(Clone, Copy)]
enum Mode {
    Unsupported,
    LaunchFailed,
    WrongShape,
}

/// `col2im3d`／`im2col3d` だけを差し替える `BackendOps`。それ以外は naive へ委譲する。
struct ConvMock {
    inner: Box<dyn BackendOps + Send>,
    col2im: (Mode, Arc<AtomicUsize>),
    im2col: (Mode, Arc<AtomicUsize>),
}

impl BackendOps for ConvMock {
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
    fn im2col3d(
        &self,
        _input: &Tensor<f32>,
        _params: &Conv3dParams,
    ) -> Result<Tensor<f32>, BackendError> {
        self.im2col.1.fetch_add(1, Ordering::SeqCst);
        match self.im2col.0 {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            Mode::WrongShape => Ok(t(vec![0.0; 3], &[3])),
        }
    }
    fn col2im3d(
        &self,
        _d_col: &Tensor<f32>,
        _input_shape: &[usize],
        _params: &Conv3dParams,
    ) -> Result<Tensor<f32>, BackendError> {
        self.col2im.1.fetch_add(1, Ordering::SeqCst);
        match self.col2im.0 {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            Mode::WrongShape => Ok(t(vec![0.0; 3], &[3])),
        }
    }
}

fn mock_tape(col2im: Mode, im2col: Mode) -> (Tape, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let c2i = Arc::new(AtomicUsize::new(0));
    let i2c = Arc::new(AtomicUsize::new(0));
    let tape = Tape::new_with_ops(Box::new(ConvMock {
        inner: common::naive_ops(),
        col2im: (col2im, Arc::clone(&c2i)),
        im2col: (im2col, Arc::clone(&i2c)),
    }));
    (tape, c2i, i2c)
}

fn small_inputs() -> (Tensor<f32>, Tensor<f32>) {
    (
        t(wave(8, 0.4, 1.0), &[1, 1, 2, 2, 2]),
        t(wave(16, 0.3, 1.0), &[1, 2, 2, 2, 2]),
    )
}

#[test]
fn unsupported_hooks_fall_back_to_host_and_match_naive() {
    let (x, w) = small_inputs();
    let (tape, c2i, i2c) = mock_tape(Mode::Unsupported, Mode::Unsupported);
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let y = conv_transpose3d(&xv, &wv, None, [1; 3], [0; 3], [0; 3], [1; 3], 1).unwrap();
    assert_eq!(
        c2i.load(Ordering::SeqCst),
        1,
        "forward は col2im3d を先に呼ぶ"
    );
    assert_eq!(i2c.load(Ordering::SeqCst), 0);
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    assert_eq!(i2c.load(Ordering::SeqCst), 1, "VJP は im2col3d を先に呼ぶ");
    let want = run(
        common::naive_ops(),
        &x,
        &w,
        None,
        Some(&t(vec![1.0; 27 * 2], &[1, 2, 3, 3, 3])),
        UNIT,
    );
    assert_eq!(dense(&y.to_tensor()), dense(&want.out));
    assert_eq!(dense(grads.get(&xv).unwrap().unwrap()), want.dx);
    assert_eq!(dense(grads.get(&wv).unwrap().unwrap()), want.dw);
}

#[test]
fn non_unsupported_errors_are_propagated_in_forward_and_backward() {
    let (x, w) = small_inputs();
    // forward（col2im3d）。
    let (tape, _, _) = mock_tape(Mode::LaunchFailed, Mode::Unsupported);
    let r = conv_transpose3d(
        &tape.var(&x),
        &tape.var(&w),
        None,
        [1; 3],
        [0; 3],
        [0; 3],
        [1; 3],
        1,
    );
    assert!(matches!(
        r,
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    // VJP（im2col3d）。
    let (tape, _, _) = mock_tape(Mode::Unsupported, Mode::LaunchFailed);
    let y = conv_transpose3d(
        &tape.var(&x),
        &tape.var(&w),
        None,
        [1; 3],
        [0; 3],
        [0; 3],
        [1; 3],
        1,
    )
    .unwrap();
    let loss = y.sum(None).unwrap();
    assert!(matches!(
        tape.backward(&loss),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
}

#[test]
fn wrong_shape_from_backend_is_a_typed_error() {
    let (x, w) = small_inputs();
    let (tape, _, _) = mock_tape(Mode::WrongShape, Mode::Unsupported);
    let r = conv_transpose3d(
        &tape.var(&x),
        &tape.var(&w),
        None,
        [1; 3],
        [0; 3],
        [0; 3],
        [1; 3],
        1,
    );
    assert!(matches!(
        r,
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
    let (tape, _, _) = mock_tape(Mode::Unsupported, Mode::WrongShape);
    let y = conv_transpose3d(
        &tape.var(&x),
        &tape.var(&w),
        None,
        [1; 3],
        [0; 3],
        [0; 3],
        [1; 3],
        1,
    )
    .unwrap();
    let loss = y.sum(None).unwrap();
    assert!(matches!(
        tape.backward(&loss),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
}
