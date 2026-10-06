//! `fold_ops::{unfold, fold}`（イシュー #2645）の `Tape`／`Var` を経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/fold-unfold-pytorch-reference/fold_unfold_reference.json`・生成条件は同ディレクトリの
//!   `README.md`）と突合する。判定区分（tolerance 定数は新設・変更しない）:
//!   - `unfold` の forward と `fold` の d_input（= unfold）は純コピーなので **bit 一致**。
//!   - `fold` の forward と `unfold` の d_input（= fold）は、窓が重ならない設定では **bit 一致**、重なる設定では
//!     本実装が `f64` アキュムレータ・PyTorch が f32 累積のため **REQ-2 統一複合判定**（`common::req2_close`）。
//! - 独立オラクル: 中心差分（`tests/conv_transpose3d_parity.rs` と同じ `H=1e-3`・相対 1e-2 または絶対 1e-3・
//!   `τ=1e-4`。緩和しない）・随伴恒等式・`fold(unfold(x)) = x ⊙ fold(unfold(1))`。
//! - フォールバックは `Unsupported` のときだけで、他のバックエンドエラーは握りつぶさず伝播し、誤 shape は
//!   型付きエラーで拒否する（`im2col`／`col2im` の両フック）。
//! - 引数検査・孤児ノード無し・`N = 0`・確保前サイズ検査・テープ記録数・決定性も固定する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::fold_ops::{fold, unfold};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, Conv2dParams, ShapeError, Tensor};
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

fn arr2(v: &[usize]) -> [usize; 2] {
    [v[0], v[1]]
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// 決定的な擬似データ（`sin` 由来。テストごとに位相を変える）。
fn wave(n: usize, freq: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * freq).sin() * amp).collect()
}

/// unfold／fold 共通の引数束（`fold` のときだけ `output_size` を使う）。
#[derive(Clone, Copy)]
struct Cfg {
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
}

// --- fixture ---

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    unfold_cases: Vec<Case>,
    fold_cases: Vec<Case>,
    nonfinite_cases: Vec<NonFinite>,
    error_cases: Vec<ErrCase>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    in_shape: Vec<usize>,
    output_size: Option<Vec<usize>>,
    kernel: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,
    dilation: Vec<usize>,
    overlapping: Option<bool>,
    out_shape: Vec<usize>,
    x_bits: Vec<u32>,
    g_bits: Vec<u32>,
    out_bits: Vec<u32>,
    dx_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct NonFinite {
    name: String,
    kind: String,
    in_shape: Vec<usize>,
    output_size: Option<Vec<usize>>,
    kernel: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,
    dilation: Vec<usize>,
    out_shape: Vec<usize>,
    x_bits: Vec<u32>,
    g_bits: Vec<u32>,
    out_bits: Vec<u32>,
    dx_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct ErrCase {
    name: String,
    torch_raises: bool,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fold-unfold-pytorch-reference/fold_unfold_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

fn assert_bits_eq(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{context}[{i}]: actual={a} expected={e}"
        );
    }
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

/// NaN はクラス一致（payload 不問）、それ以外はビット一致。
fn assert_bits_or_nan_class(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{context}[{i}]: actual={a} expected=NaN");
        } else {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{context}[{i}]: actual={a} expected={e}"
            );
        }
    }
}

/// 有限値は REQ-2 統一複合判定、非有限値（NaN／±inf）はクラス・ビット一致。
fn assert_close_or_class(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_finite() {
            assert!(
                common::req2_close(f64::from(a), f64::from(e)),
                "{context}[{i}]: actual={a} expected={e}"
            );
        } else {
            assert_bits_or_nan_class(&[a], &[e], &format!("{context}[{i}]"));
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Unfold,
    Fold,
}

fn apply<'t>(
    kind: Kind,
    x: &fandhe_ai_autodiff::Var<'t>,
    output_size: [usize; 2],
    c: Cfg,
) -> Result<fandhe_ai_autodiff::Var<'t>, AutodiffError> {
    match kind {
        Kind::Unfold => unfold(x, c.kernel, c.stride, c.padding, c.dilation),
        Kind::Fold => fold(x, output_size, c.kernel, c.stride, c.padding, c.dilation),
    }
}

struct Run {
    out: Tensor<f32>,
    dx: Vec<f32>,
}

/// `(out * g).sum()` を損失とした forward 値と入力勾配を返す。
fn run(
    ops: Box<dyn BackendOps + Send>,
    kind: Kind,
    x: &Tensor<f32>,
    output_size: [usize; 2],
    g: Option<&Tensor<f32>>,
    c: Cfg,
) -> Run {
    let tape = Tape::new_with_ops(ops);
    let xv = tape.var(x);
    let y = apply(kind, &xv, output_size, c).expect("fold/unfold");
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
        out,
    }
}

fn cfg_of(c: &Case) -> Cfg {
    Cfg {
        kernel: arr2(&c.kernel),
        stride: arr2(&c.stride),
        padding: arr2(&c.padding),
        dilation: arr2(&c.dilation),
    }
}

// --- 1. PyTorch fixture 突合 ---

#[test]
fn fixture_unfold_matches_pytorch_reference() {
    let fixture = load_fixture();
    assert!(
        fixture.torch_version.starts_with("2.14.0"),
        "fixture は PyTorch 2.14.0 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    assert!(fixture.unfold_cases.len() >= 10);
    for c in &fixture.unfold_cases {
        let x = t(from_bits(&c.x_bits), &c.in_shape);
        let g = t(from_bits(&c.g_bits), &c.out_shape);
        let r = run(
            common::naive_ops(),
            Kind::Unfold,
            &x,
            [0, 0],
            Some(&g),
            cfg_of(c),
        );
        assert_eq!(
            r.out.shape(),
            c.out_shape.as_slice(),
            "{}: 出力 shape",
            c.name
        );
        // unfold は純コピー: bit 一致。
        assert_bits_eq(
            &dense(&r.out),
            &from_bits(&c.out_bits),
            &format!("{} forward", c.name),
        );
        // d_input = fold: 重なる窓は f64 vs f32 累積のため統一複合判定。
        assert_close_all(&r.dx, &from_bits(&c.dx_bits), &format!("{} dx", c.name));
    }
}

#[test]
fn fixture_fold_matches_pytorch_reference() {
    let fixture = load_fixture();
    assert!(fixture.fold_cases.len() >= 10);
    let (mut overlapping, mut disjoint) = (0, 0);
    for c in &fixture.fold_cases {
        let osz = arr2(
            c.output_size
                .as_ref()
                .expect("fold には output_size がある"),
        );
        let x = t(from_bits(&c.x_bits), &c.in_shape);
        let g = t(from_bits(&c.g_bits), &c.out_shape);
        let r = run(
            common::naive_ops(),
            Kind::Fold,
            &x,
            osz,
            Some(&g),
            cfg_of(c),
        );
        assert_eq!(
            r.out.shape(),
            c.out_shape.as_slice(),
            "{}: 出力 shape",
            c.name
        );
        let want = from_bits(&c.out_bits);
        if c.overlapping.expect("fold ケースは overlapping を持つ") {
            overlapping += 1;
            assert_close_all(&dense(&r.out), &want, &format!("{} forward", c.name));
        } else {
            disjoint += 1;
            assert_bits_eq(&dense(&r.out), &want, &format!("{} forward", c.name));
        }
        // d_input = unfold: 純コピーなので bit 一致。
        assert_bits_eq(&r.dx, &from_bits(&c.dx_bits), &format!("{} dx", c.name));
    }
    assert!(
        overlapping >= 5 && disjoint >= 4,
        "{overlapping}/{disjoint}"
    );
}

#[test]
fn fixture_nonfinite_cases_match_pytorch_reference() {
    let fixture = load_fixture();
    assert!(fixture.nonfinite_cases.len() >= 4);
    for c in &fixture.nonfinite_cases {
        let kind = match c.kind.as_str() {
            "unfold" => Kind::Unfold,
            "fold" => Kind::Fold,
            other => panic!("未知の kind: {other}"),
        };
        let cfg = Cfg {
            kernel: arr2(&c.kernel),
            stride: arr2(&c.stride),
            padding: arr2(&c.padding),
            dilation: arr2(&c.dilation),
        };
        let osz = c.output_size.as_deref().map_or([0, 0], arr2);
        let x = t(from_bits(&c.x_bits), &c.in_shape);
        let g = t(from_bits(&c.g_bits), &c.out_shape);
        let r = run(common::naive_ops(), kind, &x, osz, Some(&g), cfg);
        // 純コピー側（unfold forward・fold d_input・重ならない fold forward）は bit 一致（NaN はクラス一致）。
        // 加算を伴う側（unfold d_input・重なる fold forward）は f64 vs f32 累積のため、有限値は統一複合判定、
        // 非有限値（NaN／±inf）はビット／クラス一致で固定する。
        let (out_exact, dx_exact) = match (kind, c.name.contains("overlap")) {
            (Kind::Unfold, _) => (true, false),
            (Kind::Fold, overlap) => (!overlap, true),
        };
        let out_label = format!("{} forward", c.name);
        let dx_label = format!("{} dx", c.name);
        let (out, want_out) = (dense(&r.out), from_bits(&c.out_bits));
        if out_exact {
            assert_bits_or_nan_class(&out, &want_out, &out_label);
        } else {
            assert_close_or_class(&out, &want_out, &out_label);
        }
        let want_dx = from_bits(&c.dx_bits);
        if dx_exact {
            assert_bits_or_nan_class(&r.dx, &want_dx, &dx_label);
        } else {
            assert_close_or_class(&r.dx, &want_dx, &dx_label);
        }
    }
}

/// 引数エラー 1 件の仕様。
struct ErrSpec {
    kind: Kind,
    x: Vec<usize>,
    output_size: [usize; 2],
    cfg: Cfg,
}

fn unit() -> Cfg {
    Cfg {
        kernel: [2, 2],
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
    }
}

fn err_spec(name: &str) -> (ErrSpec, bool, bool) {
    // 戻り値: (仕様, PyTorch 2.14.0 が拒否するか, 本実装が受理するか)。
    let fo = |x: &[usize], osz: [usize; 2], f: fn(&mut Cfg)| {
        let mut cfg = unit();
        f(&mut cfg);
        ErrSpec {
            kind: Kind::Fold,
            x: x.to_vec(),
            output_size: osz,
            cfg,
        }
    };
    let un = |x: &[usize], f: fn(&mut Cfg)| {
        let mut cfg = unit();
        f(&mut cfg);
        ErrSpec {
            kind: Kind::Unfold,
            x: x.to_vec(),
            output_size: [0, 0],
            cfg,
        }
    };
    match name {
        "fold_k_not_divisible" => (fo(&[1, 5, 4], [3, 3], |_| {}), true, false),
        "fold_l_mismatch" => (fo(&[1, 4, 5], [3, 3], |_| {}), true, false),
        "fold_output_size_too_small" => (fo(&[1, 4, 4], [2, 2], |_| {}), true, false),
        "fold_kernel_zero" => (fo(&[1, 4, 4], [3, 3], |c| c.kernel = [0, 2]), true, false),
        "fold_stride_zero" => (fo(&[1, 4, 4], [3, 3], |c| c.stride = [0, 1]), true, false),
        "fold_dilation_zero" => (fo(&[1, 4, 4], [3, 3], |c| c.dilation = [0, 1]), true, false),
        // PyTorch はバッチなし入力を受理する。本実装はバッチ入力のみ（意図的な差分）。
        "fold_batchless_2d" => (fo(&[4, 4], [3, 3], |_| {}), false, false),
        "fold_rank4" => (fo(&[1, 1, 4, 4], [3, 3], |_| {}), true, false),
        // `N = 0` は両方とも受理する。
        "fold_batch_zero" => (fo(&[0, 4, 4], [3, 3], |_| {}), false, true),
        // PyTorch は `C = 0`（K = 0）を拒否する。本実装は空出力として受理する（意図的な差分）。
        "fold_c_zero" => (fo(&[1, 0, 4], [3, 3], |_| {}), true, true),
        "fold_output_size_zero" => (fo(&[1, 4, 1], [0, 3], |_| {}), true, false),
        "unfold_kernel_zero" => (un(&[1, 1, 3, 3], |c| c.kernel = [0, 2]), true, false),
        "unfold_stride_zero" => (un(&[1, 1, 3, 3], |c| c.stride = [0, 1]), true, false),
        "unfold_dilation_zero" => (un(&[1, 1, 3, 3], |c| c.dilation = [0, 1]), true, false),
        "unfold_kernel_gt_input" => (un(&[1, 1, 3, 3], |c| c.kernel = [5, 5]), true, false),
        "unfold_batchless_3d" => (un(&[1, 3, 3], |_| {}), false, false),
        "unfold_rank5" => (un(&[1, 1, 1, 3, 3], |_| {}), true, false),
        "unfold_batch_zero" => (un(&[0, 1, 3, 3], |_| {}), false, true),
        // PyTorch は `C = 0` を拒否する。本実装は空出力として受理する（意図的な差分）。
        "unfold_c_zero" => (un(&[1, 0, 3, 3], |_| {}), true, true),
        other => panic!("未知の error case: {other}"),
    }
}

#[test]
fn fixture_error_cases_agree_with_torch_except_documented_differences() {
    let fixture = load_fixture();
    assert!(fixture.error_cases.len() >= 19);
    for c in &fixture.error_cases {
        let (spec, torch_raises, ours_ok) = err_spec(&c.name);
        assert_eq!(
            c.torch_raises, torch_raises,
            "{}: fixture の torch_raises が期待表と食い違う（再生成後は表を更新する）",
            c.name
        );
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![0.5; numel(&spec.x)], &spec.x));
        let r = apply(spec.kind, &x, spec.output_size, spec.cfg);
        assert_eq!(r.is_ok(), ours_ok, "{}: 本実装の受理可否", c.name);
    }
}

// --- 2. 独立オラクル: 中心差分 ---

fn forward_loss(kind: Kind, x: &Tensor<f32>, osz: [usize; 2], s: &Tensor<f32>, c: Cfg) -> f64 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let y = apply(kind, &xv, osz, c).unwrap();
    dense(&y.to_tensor())
        .iter()
        .zip(dense(s).iter())
        .map(|(&yv, &sv)| f64::from(yv) * f64::from(sv))
        .sum()
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

fn check_numeric(kind: Kind, x_shape: &[usize], osz: [usize; 2], c: Cfg) {
    let x = t(wave(numel(x_shape), 0.037, 1.0), x_shape);
    let y_shape = {
        let tape = Tape::new_with_ops(common::naive_ops());
        let y = apply(kind, &tape.var(&x), osz, c).unwrap();
        y.to_tensor().shape().to_vec()
    };
    let s = t(wave(numel(&y_shape), 0.053, 0.7), &y_shape);
    let r = run(common::naive_ops(), kind, &x, osz, Some(&s), c);
    let mut data = dense(&x);
    let mut numeric = vec![0f64; data.len()];
    for i in 0..data.len() {
        let orig = f64::from(data[i]);
        data[i] = (orig + H) as f32;
        let lp = forward_loss(kind, &t(data.clone(), x_shape), osz, &s, c);
        data[i] = (orig - H) as f32;
        let lm = forward_loss(kind, &t(data.clone(), x_shape), osz, &s, c);
        data[i] = orig as f32;
        numeric[i] = (lp - lm) / (2.0 * H);
    }
    assert_grad_close("dx", &r.dx, &numeric);
}

#[test]
fn unfold_backward_matches_numeric_grad() {
    let base = Cfg {
        kernel: [2, 2],
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
    };
    check_numeric(Kind::Unfold, &[1, 2, 3, 4], [0, 0], base);
    let aniso = Cfg {
        kernel: [2, 3],
        stride: [2, 1],
        padding: [1, 0],
        dilation: [2, 1],
    };
    check_numeric(Kind::Unfold, &[2, 2, 5, 5], [0, 0], aniso);
}

#[test]
fn fold_backward_matches_numeric_grad() {
    let base = Cfg {
        kernel: [2, 2],
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
    };
    // 出力 3×4 → L = 2·3 = 6、K = 2·4 = 8。
    check_numeric(Kind::Fold, &[1, 8, 6], [3, 4], base);
    let aniso = Cfg {
        kernel: [2, 3],
        stride: [2, 1],
        padding: [1, 0],
        dilation: [2, 1],
    };
    // 出力 5×5 → H': (5+2-2-1)/2+1 = 3、W': (5-2-1)/1+1 = 3 → L = 9、K = 2·6 = 12、N = 2。
    check_numeric(Kind::Fold, &[2, 12, 9], [5, 5], aniso);
}

// --- 3. 構造的性質 ---

#[test]
fn adjoint_identity_holds() {
    // ⟨unfold(x), y⟩ = ⟨x, fold(y)⟩（f64 で内積）。
    let c = Cfg {
        kernel: [2, 3],
        stride: [1, 2],
        padding: [1, 1],
        dilation: [1, 1],
    };
    let x = t(wave(2 * 3 * 5 * 6, 0.07, 1.0), &[2, 3, 5, 6]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let ux = unfold(&tape.var(&x), c.kernel, c.stride, c.padding, c.dilation)
        .unwrap()
        .to_tensor();
    let y = t(wave(numel(ux.shape()), 0.11, 1.0), ux.shape());
    let fy = fold(
        &tape.var(&y),
        [5, 6],
        c.kernel,
        c.stride,
        c.padding,
        c.dilation,
    )
    .unwrap()
    .to_tensor();
    let dot = |a: &Tensor<f32>, b: &Tensor<f32>| -> f64 {
        dense(a)
            .iter()
            .zip(dense(b).iter())
            .map(|(&p, &q)| f64::from(p) * f64::from(q))
            .sum()
    };
    let (lhs, rhs) = (dot(&ux, &y), dot(&x, &fy));
    assert!(
        (lhs - rhs).abs() <= 1e-4 * lhs.abs().max(1.0),
        "lhs={lhs} rhs={rhs}"
    );
}

#[test]
fn fold_of_unfold_is_input_times_coverage() {
    let c = Cfg {
        kernel: [3, 3],
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
    };
    let x = t(wave(2 * 4 * 4, 0.3, 1.0), &[1, 2, 4, 4]);
    let ones = t(vec![1.0; 2 * 4 * 4], &[1, 2, 4, 4]);
    let round = |v: &Tensor<f32>| -> Vec<f32> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let u = unfold(&tape.var(v), c.kernel, c.stride, c.padding, c.dilation).unwrap();
        dense(
            &fold(&u, [4, 4], c.kernel, c.stride, c.padding, c.dilation)
                .unwrap()
                .to_tensor(),
        )
    };
    let (rx, rc) = (round(&x), round(&ones));
    for ((a, b), xv) in rx.iter().zip(&rc).zip(dense(&x)) {
        assert!((a - b * xv).abs() <= 1e-5, "{a} vs {b}*{xv}");
    }
}

#[test]
fn non_contiguous_input_is_accepted() {
    let c = unit();
    let base = t(wave(2 * 3 * 3, 0.2, 1.0), &[1, 2, 3, 3]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&base);
    let tr = xv.transpose(2, 3).unwrap();
    let got = unfold(&tr, c.kernel, c.stride, c.padding, c.dilation)
        .unwrap()
        .to_tensor();
    // 参照: transpose を実体化した入力を unfold。
    let tr_data = dense(&tr.to_tensor().contiguous());
    let tape2 = Tape::new_with_ops(common::naive_ops());
    let want = unfold(
        &tape2.var(&t(tr_data, &[1, 2, 3, 3])),
        c.kernel,
        c.stride,
        c.padding,
        c.dilation,
    )
    .unwrap()
    .to_tensor();
    assert_bits_eq(&dense(&got), &dense(&want), "transpose view");
}

// --- 4. 引数検査・境界 ---

#[test]
fn invalid_arguments_are_typed_errors_and_leave_no_orphan_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 9], &[1, 1, 3, 3]));
    let col = tape.var(&t(vec![0.0; 8], &[1, 4, 2]));
    let before = tape.len();
    for r in [
        unfold(&x, [0, 1], [1, 1], [0, 0], [1, 1]),
        unfold(&x, [2, 2], [0, 1], [0, 0], [1, 1]),
        unfold(&x, [2, 2], [1, 1], [0, 0], [0, 1]),
        unfold(&x, [2, 2], [1, 1], [usize::MAX, 0], [1, 1]),
    ] {
        assert!(matches!(r, Err(AutodiffError::Backend(_))), "引数検査");
    }
    assert!(matches!(
        unfold(&x, [4, 4], [1, 1], [0, 0], [1, 1]),
        Err(AutodiffError::Shape(_))
    ));
    assert!(matches!(
        unfold(&col, [2, 2], [1, 1], [0, 0], [1, 1]),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    assert!(matches!(
        fold(&x, [3, 3], [2, 2], [1, 1], [0, 0], [1, 1]),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    // L 不一致・K 非整除・output_size 0。
    assert!(fold(&col, [3, 3], [2, 2], [1, 1], [0, 0], [1, 1]).is_err());
    assert!(fold(&col, [3, 3], [3, 1], [1, 1], [0, 0], [1, 1]).is_err());
    assert!(fold(&col, [0, 3], [2, 2], [1, 1], [0, 0], [1, 1]).is_err());
    // 別 tape は検査対象外（入力は 1 つ）。孤児ノードを残さない。
    assert_eq!(tape.len(), before, "検査は tape 操作より前");
}

#[test]
fn empty_batch_is_accepted_and_backward_yields_zero_gradients() {
    let c = unit();
    for kind in [Kind::Unfold, Kind::Fold] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = match kind {
            Kind::Unfold => tape.var(&t(vec![], &[0, 1, 3, 3])),
            Kind::Fold => tape.var(&t(vec![], &[0, 4, 4])),
        };
        let y = apply(kind, &x, [3, 3], c).unwrap();
        let want: &[usize] = match kind {
            Kind::Unfold => &[0, 4, 4],
            Kind::Fold => &[0, 1, 3, 3],
        };
        assert_eq!(y.to_tensor().shape(), want);
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        // 空入力へ流れる勾配は空（バックエンドを呼ばずゼロ勾配）。
        if let Some(g) = grads.get(&x).unwrap() {
            assert_eq!(numel(g.shape()), 0);
        }
    }
}

#[test]
fn huge_broadcast_view_is_rejected_before_allocation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let base = tape.var(&t(vec![1.0], &[1, 1, 1, 1]));
    let huge = base.broadcast_to(&[1, 1, 1, 1usize << 61]).unwrap();
    assert!(matches!(
        unfold(&huge, [1, 1], [1, 1], [0, 0], [1, 1]),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
}

#[test]
fn huge_strided_broadcast_view_is_rejected_before_contiguous() {
    // 空間軸が巨大でも stride が大きく L が小さい（出力は小さい）ケース。入力側の検査が無いと
    // `contiguous()` が view 全体を確保してしまう。
    let tape = Tape::new_with_ops(common::naive_ops());
    let base = tape.var(&t(vec![1.0], &[1, 1, 1, 1]));
    let huge = base
        .broadcast_to(&[1, 1, 1usize << 31, 1usize << 31])
        .unwrap();
    let big = 1usize << 31;
    assert!(matches!(
        unfold(&huge, [1, 1], [big, big], [0, 0], [1, 1]),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
    // fold 側: 入力 `[1, 1, L]` を巨大 broadcast view にする（K=1・L=2^61）。
    let col = tape.var(&t(vec![1.0], &[1, 1, 1]));
    let huge_col = col.broadcast_to(&[1, 1, 1usize << 61]).unwrap();
    let before = tape.len();
    assert!(matches!(
        fold(&huge_col, [1, 1], [1, 1], [1, 1], [0, 0], [1, 1]),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
    assert_eq!(tape.len(), before);
}

#[test]
fn huge_output_size_is_rejected_before_allocation() {
    // kernel 1・stride 2^31 → L = 1。出力は 2^31·2^31 要素でバイト数が usize を超えるため確保前に拒否する。
    let tape = Tape::new_with_ops(common::naive_ops());
    let col = tape.var(&t(vec![1.0], &[1, 1, 1]));
    let big = 1usize << 31;
    let before = tape.len();
    assert!(matches!(
        fold(&col, [big, big], [1, 1], [big, big], [0, 0], [1, 1]),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
    assert!(matches!(
        fold(
            &col,
            [usize::MAX, usize::MAX],
            [1, 1],
            [big, big],
            [0, 0],
            [1, 1]
        ),
        Err(AutodiffError::Shape(_))
    ));
    assert_eq!(tape.len(), before);
}

#[test]
fn records_exactly_one_node_and_rejects_create_graph() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(wave(9, 0.3, 1.0), &[1, 1, 3, 3]));
    let before = tape.len();
    let u = unfold(&x, [2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
    assert_eq!(tape.len(), before + 1);
    let f = fold(&u, [3, 3], [2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
    assert_eq!(tape.len(), before + 2);
    let loss = f.sum(None).unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
}

#[test]
fn results_are_bit_deterministic_run_to_run() {
    let c = Cfg {
        kernel: [3, 3],
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
    };
    let x = t(wave(2 * 5 * 5, 0.13, 1.0), &[1, 2, 5, 5]);
    let a = run(common::naive_ops(), Kind::Unfold, &x, [0, 0], None, c);
    let b = run(common::naive_ops(), Kind::Unfold, &x, [0, 0], None, c);
    assert_bits_eq(&dense(&a.out), &dense(&b.out), "forward");
    assert_bits_eq(&a.dx, &b.dx, "dx");
}

// --- 5. バックエンドフック（Unsupported のときだけホストへフォールバック） ---

#[derive(Clone, Copy)]
enum Mode {
    Unsupported,
    LaunchFailed,
    WrongShape,
}

/// `im2col`／`col2im` だけを差し替える `BackendOps`。それ以外は naive へ委譲する。
struct ConvMock {
    inner: Box<dyn BackendOps + Send>,
    im2col: (Mode, Arc<AtomicUsize>),
    col2im: (Mode, Arc<AtomicUsize>),
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
    fn im2col(
        &self,
        _input: &Tensor<f32>,
        _params: &Conv2dParams,
    ) -> Result<Tensor<f32>, BackendError> {
        self.im2col.1.fetch_add(1, Ordering::SeqCst);
        match self.im2col.0 {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            Mode::WrongShape => Ok(t(vec![0.0; 3], &[3])),
        }
    }
    fn col2im(
        &self,
        _d_col: &Tensor<f32>,
        _input_shape: &[usize],
        _params: &Conv2dParams,
    ) -> Result<Tensor<f32>, BackendError> {
        self.col2im.1.fetch_add(1, Ordering::SeqCst);
        match self.col2im.0 {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            Mode::WrongShape => Ok(t(vec![0.0; 3], &[3])),
        }
    }
}

fn mock_tape(im2col: Mode, col2im: Mode) -> (Tape, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let i2c = Arc::new(AtomicUsize::new(0));
    let c2i = Arc::new(AtomicUsize::new(0));
    let tape = Tape::new_with_ops(Box::new(ConvMock {
        inner: common::naive_ops(),
        im2col: (im2col, Arc::clone(&i2c)),
        col2im: (col2im, Arc::clone(&c2i)),
    }));
    (tape, i2c, c2i)
}

fn unfold_default<'t>(
    x: &fandhe_ai_autodiff::Var<'t>,
) -> Result<fandhe_ai_autodiff::Var<'t>, AutodiffError> {
    unfold(x, [2, 2], [1, 1], [0, 0], [1, 1])
}

fn fold_default<'t>(
    x: &fandhe_ai_autodiff::Var<'t>,
) -> Result<fandhe_ai_autodiff::Var<'t>, AutodiffError> {
    fold(x, [3, 3], [2, 2], [1, 1], [0, 0], [1, 1])
}

#[test]
fn unsupported_hooks_fall_back_to_host_and_match_naive() {
    let x = t(wave(9, 0.4, 1.0), &[1, 1, 3, 3]);
    let col = t(wave(16, 0.3, 1.0), &[1, 4, 4]);
    let cfg = unit();

    // unfold: forward は im2col、VJP は col2im。
    let (tape, i2c, c2i) = mock_tape(Mode::Unsupported, Mode::Unsupported);
    let xv = tape.var(&x);
    let y = unfold_default(&xv).unwrap();
    assert_eq!(
        (i2c.load(Ordering::SeqCst), c2i.load(Ordering::SeqCst)),
        (1, 0)
    );
    let grads = tape.backward(&y.sum(None).unwrap()).unwrap();
    assert_eq!(
        (i2c.load(Ordering::SeqCst), c2i.load(Ordering::SeqCst)),
        (1, 1)
    );
    let want = run(
        common::naive_ops(),
        Kind::Unfold,
        &x,
        [0, 0],
        Some(&t(vec![1.0; 16], &[1, 4, 4])),
        cfg,
    );
    assert_bits_eq(&dense(&y.to_tensor()), &dense(&want.out), "unfold forward");
    assert_bits_eq(
        &dense(grads.get(&xv).unwrap().unwrap()),
        &want.dx,
        "unfold dx",
    );

    // fold: forward は col2im、VJP は im2col。
    let (tape, i2c, c2i) = mock_tape(Mode::Unsupported, Mode::Unsupported);
    let cv = tape.var(&col);
    let y = fold_default(&cv).unwrap();
    assert_eq!(
        (i2c.load(Ordering::SeqCst), c2i.load(Ordering::SeqCst)),
        (0, 1)
    );
    let grads = tape.backward(&y.sum(None).unwrap()).unwrap();
    assert_eq!(
        (i2c.load(Ordering::SeqCst), c2i.load(Ordering::SeqCst)),
        (1, 1)
    );
    let want = run(
        common::naive_ops(),
        Kind::Fold,
        &col,
        [3, 3],
        Some(&t(vec![1.0; 9], &[1, 1, 3, 3])),
        cfg,
    );
    assert_bits_eq(&dense(&y.to_tensor()), &dense(&want.out), "fold forward");
    assert_bits_eq(
        &dense(grads.get(&cv).unwrap().unwrap()),
        &want.dx,
        "fold dx",
    );
}

#[test]
fn non_unsupported_errors_are_propagated_in_forward_and_backward() {
    let x = t(wave(9, 0.4, 1.0), &[1, 1, 3, 3]);
    let col = t(wave(16, 0.3, 1.0), &[1, 4, 4]);
    // forward（unfold = im2col・fold = col2im）。
    let (tape, _, _) = mock_tape(Mode::LaunchFailed, Mode::Unsupported);
    assert!(matches!(
        unfold_default(&tape.var(&x)),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    let (tape, _, _) = mock_tape(Mode::Unsupported, Mode::LaunchFailed);
    assert!(matches!(
        fold_default(&tape.var(&col)),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    // VJP（unfold = col2im・fold = im2col）。
    let (tape, _, _) = mock_tape(Mode::Unsupported, Mode::LaunchFailed);
    let y = unfold_default(&tape.var(&x)).unwrap();
    assert!(matches!(
        tape.backward(&y.sum(None).unwrap()),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    let (tape, _, _) = mock_tape(Mode::LaunchFailed, Mode::Unsupported);
    let y = fold_default(&tape.var(&col)).unwrap();
    assert!(matches!(
        tape.backward(&y.sum(None).unwrap()),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
}

#[test]
fn wrong_shape_from_backend_is_a_typed_error() {
    let x = t(wave(9, 0.4, 1.0), &[1, 1, 3, 3]);
    let col = t(wave(16, 0.3, 1.0), &[1, 4, 4]);
    let (tape, _, _) = mock_tape(Mode::WrongShape, Mode::Unsupported);
    assert!(matches!(
        unfold_default(&tape.var(&x)),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
    let (tape, _, _) = mock_tape(Mode::Unsupported, Mode::WrongShape);
    assert!(matches!(
        fold_default(&tape.var(&col)),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
    let (tape, _, _) = mock_tape(Mode::Unsupported, Mode::WrongShape);
    let y = unfold_default(&tape.var(&x)).unwrap();
    assert!(matches!(
        tape.backward(&y.sum(None).unwrap()),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
    let (tape, _, _) = mock_tape(Mode::WrongShape, Mode::Unsupported);
    let y = fold_default(&tape.var(&col)).unwrap();
    assert!(matches!(
        tape.backward(&y.sum(None).unwrap()),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
}
