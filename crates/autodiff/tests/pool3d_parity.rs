//! `pool3d_ops`（イシュー #2643・`max_pool3d`／`avg_pool3d`）の `Tape`／`Var` を経由する
//! end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/pool3d-pytorch-reference/pool3d_reference.json`・生成条件は同ディレクトリの
//!   `README.md`）と突合する。f32 は NaN／inf を運べるよう u32 ビットパターンで保存されている。
//!   Max の値は選択のみなので bit 一致（NaN はクラス一致）・索引は完全一致、Avg の forward と
//!   勾配は REQ-2 統一複合判定（`common::req2_close`。tolerance 定数は新設しない。PyTorch は
//!   f32 累積・本実装は `f64` 累積のため bit 一致は求めない）。
//! - NaN を含む窓の索引は比較対象から除外する（PyTorch は最後の NaN・本実装は最初の NaN。
//!   2D と同じ意味論の決定。`docs/autodiff-pool3d-ops-decision.md` §5）。
//! - `common::naive_ops()` は `pool3d_*` を override しないため、必ず共有ホストカーネルへの
//!   フォールバック経路を通る。CPU `BackendOps` 実装との一致は
//!   `crates/facade/tests/pool3d_ops_backend_parity.rs` が担当する。
//! - フォールバックは `Unsupported` のときだけで、それ以外のバックエンドエラーは握りつぶさず
//!   伝播する。確保前サイズ検査・テープ記録数・孤児ノード無し・VJP の独立オラクル（中心差分・
//!   手計算）・run-to-run 決定性も固定する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::pool3d_ops::{avg_pool3d, max_pool3d};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::pool3d::Pool3dParams;
use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor};
use serde::Deserialize;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fn arr3(v: &[usize]) -> [usize; 3] {
    [v[0], v[1], v[2]]
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    finite_cases: Vec<Case>,
    nonfinite_cases: Vec<Case>,
    error_cases: Vec<ErrCase>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    op: String,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
    kernel: Vec<usize>,
    stride: Option<Vec<usize>>,
    padding: Vec<usize>,
    dilation: Vec<usize>,
    count_include_pad: Option<bool>,
    x_bits: Vec<u32>,
    g_bits: Vec<u32>,
    out_bits: Vec<u32>,
    grad_bits: Vec<u32>,
    index: Option<Vec<i32>>,
}

#[derive(Deserialize)]
struct ErrCase {
    name: String,
    op: String,
    in_shape: Vec<usize>,
    kernel: Vec<usize>,
    stride: Option<Vec<usize>>,
    padding: Vec<usize>,
    dilation: Vec<usize>,
    ceil_mode: bool,
    torch_raises: bool,
    out_shape: Vec<usize>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pool3d-pytorch-reference/pool3d_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

/// NaN は NaN 同士（クラス一致）、それ以外は bit 完全一致。
fn assert_class_or_bits_eq(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{context}[{i}]: NaN のはず（actual={a}）");
        } else {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{context}[{i}]: actual={a} expected={e}"
            );
        }
    }
}

/// NaN は NaN 同士、±inf は厳密、有限は REQ-2 統一複合判定。
fn class_close(a: f32, e: f32) -> bool {
    if e.is_nan() {
        a.is_nan()
    } else if e.is_infinite() {
        a == e
    } else {
        common::req2_close(f64::from(a), f64::from(e))
    }
}

fn assert_close_all(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(class_close(a, e), "{context}[{i}]: actual={a} expected={e}");
    }
}

struct Run {
    values: Vec<f32>,
    out_shape: Vec<usize>,
    index: Option<Vec<i32>>,
    grad: Vec<f32>,
}

fn apply<'t>(case: &Case, x: &Var<'t>) -> (Var<'t>, Option<Tensor<i32>>) {
    let stride = case.stride.as_deref().map(arr3);
    if case.op == "max" {
        let (v, i) = max_pool3d(
            x,
            arr3(&case.kernel),
            stride,
            arr3(&case.padding),
            arr3(&case.dilation),
            false,
        )
        .expect("max_pool3d");
        (v, Some(i))
    } else {
        let v = avg_pool3d(
            x,
            arr3(&case.kernel),
            stride,
            arr3(&case.padding),
            false,
            case.count_include_pad
                .expect("avg は count_include_pad を持つ"),
        )
        .expect("avg_pool3d");
        (v, None)
    }
}

/// `(out * g).sum()` を損失とした forward 値・索引・入力勾配を返す。
fn run_case(ops: Box<dyn BackendOps + Send>, case: &Case) -> Run {
    let tape = Tape::new_with_ops(ops);
    let xv = tape.var(&t(from_bits(&case.x_bits), &case.in_shape));
    let (y, idx) = apply(case, &xv);
    let out = y.to_tensor();
    let gv = tape.var_no_grad(&t(from_bits(&case.g_bits), out.shape()));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("入力へ勾配が届く").clone();
    Run {
        values: out.host_slice().into_owned(),
        out_shape: out.shape().to_vec(),
        index: idx.map(|i| i.contiguous().host_slice().into_owned()),
        grad: dx.host_slice().into_owned(),
    }
}

fn run_fixture_case(case: &Case) -> Run {
    run_case(common::naive_ops(), case)
}

// --- PyTorch fixture 突合 ---

#[test]
fn finite_cases_match_pytorch_reference_forward_and_backward() {
    let fixture = load_fixture();
    assert!(
        fixture.torch_version.starts_with("2.14.0"),
        "fixture は PyTorch 2.14.0 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    assert!(fixture.finite_cases.len() >= 45);
    let mut max_cases = 0;
    let mut avg_cases = 0;
    for case in &fixture.finite_cases {
        let r = run_fixture_case(case);
        assert_eq!(r.out_shape, case.out_shape, "{}: 出力 shape", case.name);
        let want = from_bits(&case.out_bits);
        if case.op == "max" {
            max_cases += 1;
            // 選択のみの演算なので値は bit 一致（±0 の符号も含む）。
            assert_class_or_bits_eq(&r.values, &want, &format!("{} forward", case.name));
            assert_eq!(
                r.index.as_deref(),
                case.index.as_deref(),
                "{}: 索引（タイ先勝ち）",
                case.name
            );
        } else {
            avg_cases += 1;
            assert_close_all(&r.values, &want, &format!("{} forward", case.name));
        }
        assert_close_all(
            &r.grad,
            &from_bits(&case.grad_bits),
            &format!("{} grad", case.name),
        );
    }
    assert!(
        max_cases >= 17 && avg_cases >= 28,
        "{max_cases}/{avg_cases}"
    );
}

#[test]
fn nonfinite_forward_and_gradients_match_pytorch_reference() {
    let fixture = load_fixture();
    assert!(fixture.nonfinite_cases.len() >= 18);
    for case in &fixture.nonfinite_cases {
        let r = run_fixture_case(case);
        let want = from_bits(&case.out_bits);
        let has_nan = from_bits(&case.x_bits).iter().any(|v| v.is_nan());
        if case.op == "max" {
            assert_class_or_bits_eq(&r.values, &want, &format!("{} forward", case.name));
            if !has_nan {
                assert_eq!(
                    r.index.as_deref(),
                    case.index.as_deref(),
                    "{}: 索引",
                    case.name
                );
                assert_close_all(
                    &r.grad,
                    &from_bits(&case.grad_bits),
                    &format!("{} grad", case.name),
                );
            }
            // NaN を含む入力の索引・勾配は意味論の決定による差分（最初／最後の NaN）。
            // 決定記録 §5 に記録しており、ここでは比較しない。
        } else {
            assert_close_all(&r.values, &want, &format!("{} forward", case.name));
            assert_close_all(
                &r.grad,
                &from_bits(&case.grad_bits),
                &format!("{} grad", case.name),
            );
        }
    }
}

/// PyTorch は最後の NaN、本実装は最初の NaN の索引を残す（意味論の決定。2D と同じ）。
#[test]
fn nan_window_index_is_the_first_nan_in_row_major_order() {
    let fixture = load_fixture();
    let case = fixture
        .nonfinite_cases
        .iter()
        .find(|c| c.name == "max_nan_multi_in_window")
        .expect("fixture に存在する");
    let r = run_fixture_case(case);
    let x = from_bits(&case.x_bits);
    let out_numel: usize = case.out_shape.iter().product();
    assert_eq!(out_numel, r.values.len());
    let index = r.index.expect("max は索引を返す");
    // 入力は [1,1,3,3,3]（平面 1 枚）なので索引はそのまま `x` の添字。
    assert_eq!(case.in_shape[..2], [1, 1]);
    let in_shape: [usize; 5] = case.in_shape[..].try_into().unwrap();
    let (_, wins) = ref_windows(
        in_shape,
        arr3(&case.kernel),
        arr3(case.stride.as_ref().unwrap()),
        arr3(&case.padding),
        arr3(&case.dilation),
    );
    let mut nan_windows = 0;
    for (o, (&v, &idx)) in r.values.iter().zip(&index).enumerate() {
        if v.is_nan() {
            nan_windows += 1;
            let first = wins[o].iter().copied().find(|&i| x[i].is_nan()).unwrap();
            assert_eq!(usize::try_from(idx).unwrap(), first, "o={o}");
        }
    }
    assert!(nan_windows >= 2, "複数の窓が NaN を含むケースのはず");
}

#[test]
fn error_cases_agree_with_torch_except_documented_differences() {
    let fixture = load_fixture();
    assert!(fixture.error_cases.len() >= 24);
    for case in &fixture.error_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let n: usize = case.in_shape.iter().product();
        let x = tape.var(&t(vec![0.0; n], &case.in_shape));
        let stride = case.stride.as_deref().map(arr3);
        let r = if case.op == "max" {
            max_pool3d(
                &x,
                arr3(&case.kernel),
                stride,
                arr3(&case.padding),
                arr3(&case.dilation),
                case.ceil_mode,
            )
            .map(|(v, _)| v)
        } else {
            avg_pool3d(
                &x,
                arr3(&case.kernel),
                stride,
                arr3(&case.padding),
                case.ceil_mode,
                true,
            )
        };
        match case.name.as_str() {
            // 意図した差分 1: バッチなし rank 4 は本実装が拒否する（NCDHW の rank 5 のみ）。
            n if n.ends_with("rank4_unbatched") => {
                assert!(
                    matches!(
                        r,
                        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
                    ),
                    "{n}"
                );
            }
            // 意図した差分 2: ceil_mode=true は v1 で拒否する。
            n if n.ends_with("ceil_mode_true") => {
                assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))), "{n}");
            }
            // 意図した差分 3: C=0 は torch が拒否するが、本実装は N=0 と同じく空出力を受理する。
            n if n.ends_with("channel_zero") => {
                assert!(case.torch_raises);
                assert_eq!(r.unwrap().to_tensor().shape(), &[1, 0, 2, 2, 2], "{n}");
            }
            _ if case.torch_raises => {
                assert!(r.is_err(), "{}: torch が拒否する構成は拒否する", case.name);
            }
            _ => {
                assert_eq!(
                    r.unwrap().to_tensor().shape(),
                    case.out_shape.as_slice(),
                    "{}",
                    case.name
                );
            }
        }
    }
}

// --- 独立オラクル（f64 リファレンス実装・中心差分・手計算） ---

/// 独立実装の窓走査（i64 座標）。出力 shape と、各出力位置の窓内有効入力の flat 添字列
/// （`(n,c)` 平面の先頭オフセット込み）を返す。
fn ref_windows(
    in_shape: [usize; 5],
    k: [usize; 3],
    s: [usize; 3],
    p: [usize; 3],
    d: [usize; 3],
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut out_dims = [0usize; 3];
    for a in 0..3 {
        let eff = d[a] * (k[a] - 1) + 1;
        out_dims[a] = (in_shape[2 + a] + 2 * p[a] - eff) / s[a] + 1;
    }
    let mut wins = Vec::new();
    for nc in 0..in_shape[0] * in_shape[1] {
        let base = nc * in_shape[2] * in_shape[3] * in_shape[4];
        for od in 0..out_dims[0] {
            for oh in 0..out_dims[1] {
                for ow in 0..out_dims[2] {
                    let mut w = Vec::new();
                    for kd in 0..k[0] {
                        for kh in 0..k[1] {
                            for kw in 0..k[2] {
                                let c = [
                                    (od * s[0] + kd * d[0]) as i64 - p[0] as i64,
                                    (oh * s[1] + kh * d[1]) as i64 - p[1] as i64,
                                    (ow * s[2] + kw * d[2]) as i64 - p[2] as i64,
                                ];
                                let inside =
                                    (0..3).all(|a| c[a] >= 0 && c[a] < in_shape[2 + a] as i64);
                                if inside {
                                    let flat = (c[0] as usize * in_shape[3] + c[1] as usize)
                                        * in_shape[4]
                                        + c[2] as usize;
                                    w.push(base + flat);
                                }
                            }
                        }
                    }
                    wins.push(w);
                }
            }
        }
    }
    (out_dims.to_vec(), wins)
}

fn pseudo(n: usize, mul: usize, modulus: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * mul) % modulus) as f32 * 0.1 - 0.7)
        .collect()
}

#[test]
fn avg_gradient_matches_central_difference_for_both_pad_modes() {
    let in_shape = [1, 2, 4, 5, 4];
    let (k, s, p) = ([2, 3, 2], [1, 2, 1], [1, 1, 1]);
    let x = pseudo(160, 7, 23);
    let (_, wins) = ref_windows(in_shape, k, s, p, [1; 3]);
    let g = pseudo(wins.len(), 5, 19);
    for cip in [true, false] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.clone(), &in_shape));
        let y = avg_pool3d(&xv, k, Some(s), p, false, cip).unwrap();
        let gv = tape.var_no_grad(&t(g.clone(), y.to_tensor().shape()));
        let loss = y.mul(&gv).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&xv).unwrap().unwrap().host_slice().into_owned();
        let kn = (k[0] * k[1] * k[2]) as f64;
        let f = |xs: &[f64]| -> f64 {
            wins.iter()
                .zip(&g)
                .map(|(w, &gi)| {
                    let div = if cip { kn } else { w.len() as f64 };
                    w.iter().map(|&i| xs[i]).sum::<f64>() / div * f64::from(gi)
                })
                .sum()
        };
        let base: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let h = 1e-3;
        for j in 0..base.len() {
            let (mut hi, mut lo) = (base.clone(), base.clone());
            hi[j] += h;
            lo[j] -= h;
            let fd = (f(&hi) - f(&lo)) / (2.0 * h);
            assert!(
                common::req2_close(f64::from(dx[j]), fd),
                "cip={cip} x[{j}]: {} vs {fd}",
                dx[j]
            );
        }
    }
}

#[test]
fn max_gradient_matches_central_difference_without_ties() {
    // 27 要素すべて異なる値（5 は 27 と互いに素）。窓重なりあり・padding あり・dilation あり。
    let in_shape = [1, 1, 3, 3, 3];
    let x: Vec<f32> = (0..27).map(|i| ((i * 5) % 27) as f32 * 0.1).collect();
    let configs: [([usize; 3], [usize; 3], [usize; 3], [usize; 3]); 3] = [
        ([2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1]),
        ([2, 2, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1]),
        ([2, 1, 2], [1, 1, 1], [0, 0, 0], [2, 1, 1]),
    ];
    for (k, s, p, d) in configs {
        let (_, wins) = ref_windows(in_shape, k, s, p, d);
        let g = pseudo(wins.len(), 3, 17);
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.clone(), &in_shape));
        let (y, _) = max_pool3d(&xv, k, Some(s), p, d, false).unwrap();
        let gv = tape.var_no_grad(&t(g.clone(), y.to_tensor().shape()));
        let loss = y.mul(&gv).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&xv).unwrap().unwrap().host_slice().into_owned();
        let f = |xs: &[f64]| -> f64 {
            wins.iter()
                .zip(&g)
                .map(|(w, &gi)| {
                    w.iter().map(|&i| xs[i]).fold(f64::NEG_INFINITY, f64::max) * f64::from(gi)
                })
                .sum()
        };
        let base: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let h = 1e-3;
        for j in 0..27 {
            let (mut hi, mut lo) = (base.clone(), base.clone());
            hi[j] += h;
            lo[j] -= h;
            let fd = (f(&hi) - f(&lo)) / (2.0 * h);
            assert!(
                common::req2_close(f64::from(dx[j]), fd),
                "k={k:?} p={p:?} x[{j}]: {} vs {fd}",
                dx[j]
            );
        }
    }
}

#[test]
fn max_gradient_accumulates_on_overlapping_windows() {
    // 窓 {1,5}・{5,2} がどちらも位置 1（値 5）を選ぶ: 勾配は 1 + 10 = 11 が位置 1 へ加算される。
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&t(vec![1.0, 5.0, 2.0], &[1, 1, 1, 1, 3]));
    let (y, idx) = max_pool3d(&xv, [1, 1, 2], Some([1, 1, 1]), [0; 3], [1; 3], false).unwrap();
    assert_eq!(idx.contiguous().host_slice().into_owned(), [1, 1]);
    let gv = tape.var_no_grad(&t(vec![1.0, 10.0], &[1, 1, 1, 1, 2]));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().unwrap().host_slice().into_owned();
    assert_eq!(dx, vec![0.0, 11.0, 0.0]);
}

#[test]
fn avg_gradient_overlap_hand_computed() {
    // [1,1,1,1,3]・k=[1,1,2]・s=1・padding 0・出力 2 要素。g=[2,4] → d_x = [1, 3, 2]。
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&t(vec![0.0; 3], &[1, 1, 1, 1, 3]));
    let y = avg_pool3d(&xv, [1, 1, 2], Some([1, 1, 1]), [0; 3], false, true).unwrap();
    let gv = tape.var_no_grad(&t(vec![2.0, 4.0], &[1, 1, 1, 1, 2]));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().unwrap().host_slice().into_owned();
    assert_eq!(dx, vec![1.0, 3.0, 2.0]);
}

#[test]
fn non_contiguous_input_is_pooled_in_logical_order() {
    let base = t((0..8).map(|i| i as f32).collect(), &[1, 1, 2, 2, 2]);
    let tr = base.transpose(3, 4).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&tr);
    let (y, idx) = max_pool3d(&xv, [1, 1, 2], None, [0; 3], [1; 3], false).unwrap();
    assert_eq!(
        y.to_tensor().host_slice().into_owned(),
        [2.0, 3.0, 6.0, 7.0]
    );
    assert_eq!(idx.contiguous().host_slice().into_owned(), [1, 3, 5, 7]);
}

// --- フォールバックとエラー伝播 ---

/// `pool3d_*` だけを差し替える `BackendOps`。それ以外は naive へ委譲する。
struct PoolMock {
    inner: Box<dyn BackendOps + Send>,
    calls: Arc<AtomicUsize>,
    mode: Mode,
}

#[derive(Clone, Copy)]
enum Mode {
    Unsupported,
    LaunchFailed,
    WrongValueShape,
    WrongIndexShape,
}

impl BackendOps for PoolMock {
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
    fn pool3d_max(
        &self,
        _input: &Tensor<f32>,
        _params: &Pool3dParams,
    ) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            // 正しい出力 shape は [1,1,1,1,1]。
            Mode::WrongValueShape => Ok((
                t(vec![0.0; 2], &[2]),
                Tensor::new(vec![0_i32; 1], &[1, 1, 1, 1, 1]).unwrap(),
            )),
            Mode::WrongIndexShape => Ok((
                t(vec![0.0; 1], &[1, 1, 1, 1, 1]),
                Tensor::new(vec![0_i32; 2], &[2]).unwrap(),
            )),
        }
    }
    fn pool3d_avg(
        &self,
        _input: &Tensor<f32>,
        _params: &Pool3dParams,
        _count_include_pad: bool,
    ) -> Result<Tensor<f32>, BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            Mode::WrongValueShape | Mode::WrongIndexShape => Ok(t(vec![0.0; 2], &[2])),
        }
    }
}

fn mock_tape(mode: Mode) -> (Tape, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let tape = Tape::new_with_ops(Box::new(PoolMock {
        inner: common::naive_ops(),
        calls: Arc::clone(&calls),
        mode,
    }));
    (tape, calls)
}

fn cube(tape: &Tape) -> Var<'_> {
    tape.var(&t((0..8).map(|i| i as f32).collect(), &[1, 1, 2, 2, 2]))
}

#[test]
fn unsupported_backend_falls_back_to_host_reference() {
    let (tape, calls) = mock_tape(Mode::Unsupported);
    let x = cube(&tape);
    let (v, i) = max_pool3d(&x, [2, 2, 2], None, [0; 3], [1; 3], false).unwrap();
    assert_eq!(v.to_tensor().host_slice().into_owned(), [7.0]);
    assert_eq!(i.contiguous().host_slice().into_owned(), [7]);
    let a = avg_pool3d(&x, [2, 2, 2], None, [0; 3], false, true).unwrap();
    assert_eq!(a.to_tensor().host_slice().into_owned(), [3.5]);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "バックエンドを先に呼んでからフォールバックしているはず"
    );
}

#[test]
fn non_unsupported_backend_errors_are_propagated() {
    let (tape, _) = mock_tape(Mode::LaunchFailed);
    let x = cube(&tape);
    let rs = [
        max_pool3d(&x, [2, 2, 2], None, [0; 3], [1; 3], false).map(|_| ()),
        avg_pool3d(&x, [2, 2, 2], None, [0; 3], false, true).map(|_| ()),
    ];
    for r in rs {
        assert!(matches!(
            r,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ));
    }
}

#[test]
fn wrong_shape_from_backend_is_a_typed_error() {
    for mode in [Mode::WrongValueShape, Mode::WrongIndexShape] {
        let (tape, _) = mock_tape(mode);
        let x = cube(&tape);
        assert!(matches!(
            max_pool3d(&x, [2, 2, 2], None, [0; 3], [1; 3], false),
            Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
        ));
        assert!(matches!(
            avg_pool3d(&x, [2, 2, 2], None, [0; 3], false, true),
            Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
        ));
    }
}

// --- 境界・テープ・決定性 ---

#[test]
fn invalid_arguments_are_typed_errors_and_leave_no_orphan_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = cube(&tape);
    let before = tape.len();
    // ceil_mode=true。
    assert!(matches!(
        max_pool3d(&x, [2, 2, 2], None, [0; 3], [1; 3], true),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        avg_pool3d(&x, [2, 2, 2], None, [0; 3], true, true),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // 0 パラメータ・padding 上限超過。
    assert!(matches!(
        max_pool3d(&x, [0, 2, 2], None, [0; 3], [1; 3], false),
        Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
    ));
    assert!(matches!(
        avg_pool3d(&x, [2, 2, 2], Some([1, 0, 1]), [0; 3], false, true),
        Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
    ));
    assert!(matches!(
        max_pool3d(&x, [2, 2, 2], None, [0; 3], [1, 1, 0], false),
        Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
    ));
    assert!(matches!(
        max_pool3d(&x, [2, 2, 2], None, [2, 0, 0], [1; 3], false),
        Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
    ));
    // カーネルが入力より大きい（負分子）・空窓（kernel=2・dilation>in）。
    assert!(matches!(
        max_pool3d(&x, [3, 2, 2], None, [0; 3], [1; 3], false),
        Err(AutodiffError::Shape(_))
    ));
    assert!(matches!(
        max_pool3d(&x, [2, 1, 1], Some([1; 3]), [0; 3], [3, 1, 1], false),
        Err(AutodiffError::Shape(_))
    ));
    // rank 不一致・空間軸 0。
    let x4 = tape.var(&t(vec![0.0; 8], &[1, 2, 2, 2]));
    assert!(matches!(
        avg_pool3d(&x4, [2, 2, 2], None, [0; 3], false, true),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    let z = tape.var(&t(vec![], &[1, 1, 0, 2, 2]));
    assert!(matches!(
        max_pool3d(&z, [1, 1, 1], None, [0; 3], [1; 3], false),
        Err(AutodiffError::Shape(_))
    ));
    // 孤児ノードを残さない（検査は tape 操作より前）。
    assert_eq!(tape.len(), before + 2, "x4・z の入力ノード 2 件のみ増える");
}

#[test]
fn empty_batch_yields_empty_outputs() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![], &[0, 2, 4, 4, 4]));
    let (v, i) = max_pool3d(&x, [2, 2, 2], None, [0; 3], [1; 3], false).unwrap();
    assert_eq!(v.to_tensor().shape(), &[0, 2, 2, 2, 2]);
    assert_eq!(i.shape(), &[0, 2, 2, 2, 2]);
    let a = avg_pool3d(&x, [2, 2, 2], None, [0; 3], false, true).unwrap();
    assert_eq!(a.to_tensor().shape(), &[0, 2, 2, 2, 2]);
}

#[test]
fn huge_broadcast_view_is_rejected_before_allocation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let base = tape.var(&t(vec![1.0], &[1, 1, 1, 1, 1]));
    let huge = base.broadcast_to(&[1, 1, 1, 1, 1usize << 61]).unwrap();
    for r in [
        max_pool3d(&huge, [1, 1, 1], None, [0; 3], [1; 3], false).map(|_| ()),
        avg_pool3d(&huge, [1, 1, 1], None, [0; 3], false, true).map(|_| ()),
    ] {
        assert!(matches!(
            r,
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
    }
}

#[test]
fn each_op_records_exactly_one_node() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = cube(&tape);
    let before = tape.len();
    max_pool3d(&x, [2, 2, 2], None, [0; 3], [1; 3], false).unwrap();
    assert_eq!(tape.len(), before + 1);
    avg_pool3d(&x, [2, 2, 2], None, [0; 3], false, false).unwrap();
    assert_eq!(tape.len(), before + 2);
}

#[test]
fn create_graph_is_a_typed_error_not_a_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = cube(&tape);
    let loss = avg_pool3d(&x, [2, 2, 2], None, [0; 3], false, true)
        .unwrap()
        .sum(None)
        .unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
    let (v, _) = max_pool3d(&x, [2, 2, 2], None, [0; 3], [1; 3], false).unwrap();
    let loss = v.sum(None).unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
}

#[test]
fn results_are_bit_deterministic_run_to_run() {
    let fixture = load_fixture();
    for case in fixture
        .finite_cases
        .iter()
        .filter(|c| c.name.contains("overlap") || c.name.contains("pad"))
    {
        let a = run_fixture_case(case);
        let b = run_fixture_case(case);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.values), bits(&b.values), "{}", case.name);
        assert_eq!(bits(&a.grad), bits(&b.grad), "{}", case.name);
        assert_eq!(a.index, b.index, "{}", case.name);
    }
}
