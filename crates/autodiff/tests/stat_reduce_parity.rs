//! `stat_reduce_ops`（イシュー #2637・`median`／`kthvalue`／`quantile`／`nansum`／
//! `nanmean`）の `Tape`／`Var` を経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/stat-reduce-pytorch-reference/stat_reduce_reference.json`・
//!   生成条件は同ディレクトリの `README.md`）と突合する。f32 は NaN／inf を運べる
//!   よう u32 ビットパターンで保存されている。選択のみの演算（`median`・`kthvalue`・
//!   `quantile` の `lower`／`higher`／`nearest`）の値は bit 一致（NaN はクラス一致）、
//!   算術を含む演算と勾配は REQ-2 統一複合判定（`common::req2_close`。tolerance
//!   定数は新設しない）で比較する。
//! - **タイの索引は受入条件にしない**（PyTorch は規定せず、本実装は安定昇順で決定的）。
//!   タイのない入力のみ索引の完全一致を要求し、タイ入力は `x[index] == value` と
//!   独自契約の手計算期待値で検証する（決定記録 §3）。
//! - `common::naive_ops()` は `stat_*` を override しないため、必ず共有ホスト
//!   カーネルへのフォールバック経路を通る。CPU `BackendOps` 実装との一致は
//!   `crates/facade/tests/stat_reduce_ops_backend_parity.rs` が担当する。
//! - フォールバックは `Unsupported` のときだけで、それ以外のバックエンドエラーは
//!   握りつぶさず伝播する。確保前サイズ検査・テープ記録数・独立オラクル（中心差分）・
//!   run-to-run 決定性も固定する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::stat_reduce_ops::{
    kthvalue, median, median_with_indices, nanmean, nansum, quantile,
};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, QuantileInterpolation, ShapeError, Tensor};
use serde::Deserialize;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    finite_cases: Vec<Case>,
    tie_cases: Vec<Case>,
    nan_cases: Vec<Case>,
    inf_cases: Vec<Case>,
    nonfinite_upstream_cases: Vec<Case>,
    error_cases: Vec<ErrCase>,
}

#[derive(Deserialize, Clone)]
struct Case {
    name: String,
    op: String,
    shape: Vec<usize>,
    dim: Option<usize>,
    k: Option<usize>,
    q_bits: Option<u32>,
    interp: Option<String>,
    x_bits: Vec<u32>,
    g_bits: Vec<u32>,
    out_shape: Vec<usize>,
    out_bits: Vec<u32>,
    grad_bits: Vec<u32>,
    index: Option<Vec<i32>>,
}

#[derive(Deserialize)]
struct ErrCase {
    name: String,
    op: String,
    shape: Vec<usize>,
    dim: Option<i64>,
    k: Option<usize>,
    q_bits: Option<u32>,
    interp: Option<String>,
    torch_raises: bool,
    out_shape: Vec<usize>,
    out_bits: Vec<u32>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stat-reduce-pytorch-reference/stat_reduce_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

fn parse_interp(name: &str) -> QuantileInterpolation {
    match name {
        "linear" => QuantileInterpolation::Linear,
        "lower" => QuantileInterpolation::Lower,
        "higher" => QuantileInterpolation::Higher,
        "midpoint" => QuantileInterpolation::Midpoint,
        "nearest" => QuantileInterpolation::Nearest,
        other => panic!("未知の補間: {other}"),
    }
}

/// 演算指定（fixture の 1 ケースの演算部分）。
#[derive(Clone)]
struct Spec {
    op: String,
    dim: Option<usize>,
    k: Option<usize>,
    q: Option<f32>,
    interp: QuantileInterpolation,
}

impl Spec {
    fn of(c: &Case) -> Self {
        Spec {
            op: c.op.clone(),
            dim: c.dim,
            k: c.k,
            q: c.q_bits.map(f32::from_bits),
            interp: c
                .interp
                .as_deref()
                .map_or(QuantileInterpolation::Linear, parse_interp),
        }
    }

    /// 値が入力要素の選択のみ（算術を通らない）か。
    fn is_selection(&self) -> bool {
        matches!(self.op.as_str(), "median_all" | "median_dim" | "kthvalue")
            || (self.op == "quantile"
                && matches!(
                    self.interp,
                    QuantileInterpolation::Lower
                        | QuantileInterpolation::Higher
                        | QuantileInterpolation::Nearest
                ))
    }
}

fn apply<'t>(s: &Spec, x: &Var<'t>) -> Result<(Var<'t>, Option<Tensor<i32>>), AutodiffError> {
    let dim = || s.dim.expect("この演算は dim が必要");
    match s.op.as_str() {
        "median_all" => Ok((median(x, None)?, None)),
        "median_dim" => {
            let (v, i) = median_with_indices(x, dim())?;
            Ok((v, Some(i)))
        }
        "kthvalue" => {
            let (v, i) = kthvalue(x, s.k.expect("k"), dim())?;
            Ok((v, Some(i)))
        }
        "quantile" => Ok((quantile(x, s.q.expect("q"), s.dim, s.interp)?, None)),
        "nansum" => Ok((nansum(x, s.dim)?, None)),
        "nanmean" => Ok((nanmean(x, s.dim)?, None)),
        other => panic!("未知の op: {other}"),
    }
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

/// NaN は NaN 同士、有限・±inf は数値の等価（`±0` は同じ組）。タイ入力用。
fn assert_class_or_value_eq(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{context}[{i}]: NaN のはず（actual={a}）");
        } else {
            assert!(a == e, "{context}[{i}]: actual={a} expected={e}");
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

/// `(out * g).sum()` を損失とした forward 値・索引・入力勾配を返す。
fn run_case(
    ops: Box<dyn BackendOps + Send>,
    s: &Spec,
    x: Vec<f32>,
    shape: &[usize],
    g: Vec<f32>,
) -> Run {
    let tape = Tape::new_with_ops(ops);
    let xv = tape.var(&t(x, shape));
    let (y, idx) = apply(s, &xv).expect("forward");
    let out_shape = y.to_tensor().shape().to_vec();
    let values = y.to_tensor().host_slice().into_owned();
    let gv = tape.var_no_grad(&t(g, &out_shape));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("入力へ勾配が届く").clone();
    Run {
        values,
        out_shape,
        index: idx.map(|i| i.contiguous().host_slice().into_owned()),
        grad: dx.host_slice().into_owned(),
    }
}

fn run_fixture_case(case: &Case) -> Run {
    run_case(
        common::naive_ops(),
        &Spec::of(case),
        from_bits(&case.x_bits),
        &case.shape,
        from_bits(&case.g_bits),
    )
}

fn check_forward(case: &Case, r: &Run, bitwise_selection: bool) {
    let s = Spec::of(case);
    let want = from_bits(&case.out_bits);
    assert_eq!(r.out_shape, case.out_shape, "{}: 出力 shape", case.name);
    if s.is_selection() {
        if bitwise_selection {
            assert_class_or_bits_eq(&r.values, &want, &format!("{} forward", case.name));
        } else {
            assert_class_or_value_eq(&r.values, &want, &format!("{} forward", case.name));
        }
    } else {
        assert_close_all(&r.values, &want, &format!("{} forward", case.name));
    }
}

// --- PyTorch fixture 突合 ---

#[test]
fn finite_cases_match_pytorch_reference_forward_index_and_backward() {
    let fixture = load_fixture();
    assert!(
        fixture.torch_version.starts_with("2.14.0"),
        "fixture は PyTorch 2.14.0 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    assert!(fixture.finite_cases.len() >= 400);
    for case in &fixture.finite_cases {
        let r = run_fixture_case(case);
        check_forward(case, &r, true);
        // タイのない入力なので索引は完全一致。
        if case.index.is_some() {
            assert_eq!(
                r.index.as_deref(),
                case.index.as_deref(),
                "{}: 索引",
                case.name
            );
        }
        assert_close_all(
            &r.grad,
            &from_bits(&case.grad_bits),
            &format!("{} grad", case.name),
        );
    }
}

#[test]
fn nan_cases_match_pytorch_reference() {
    let fixture = load_fixture();
    assert!(fixture.nan_cases.len() >= 100);
    for case in &fixture.nan_cases {
        let r = run_fixture_case(case);
        check_forward(case, &r, true);
        // NaN が 2 個以上あると PyTorch の `kthvalue` は NaN 同士の並びが不定（実測:
        // `[1,nan,2,nan,0,7]` の k=5 → 添字 3・k=6 → 添字 1）なため、その索引は比較しない
        // （本実装は安定順）。`median` は最初の NaN の位置で一致する。
        let nan_count = from_bits(&case.x_bits)
            .iter()
            .filter(|v| v.is_nan())
            .count();
        if case.index.is_some() && !(case.op == "kthvalue" && nan_count >= 2) {
            assert_eq!(
                r.index.as_deref(),
                case.index.as_deref(),
                "{}: 索引（NaN 規則）",
                case.name
            );
        }
        // 選ばれる NaN 要素が不定な lane（NaN 2 個以上）では、選択系の勾配の行き先も不定。
        // 本実装は安定順の最後の NaN へ流す（独自契約）。`nansum`／`nanmean`／`median`
        // （全要素の均等分配・軸指定は最初の NaN）は規則が決まるため比較する。
        let ambiguous = nan_count >= 2 && matches!(case.op.as_str(), "kthvalue" | "quantile");
        if !ambiguous {
            assert_close_all(
                &r.grad,
                &from_bits(&case.grad_bits),
                &format!("{} grad", case.name),
            );
        }
    }
}

/// タイ入力: 値は fixture と一致（`±0` は同じ組）。索引・選択系の勾配は PyTorch 側が
/// 規定しないため比較せず、`x[index] == value` を確認する。タイに依らない勾配
/// （`nansum`・`nanmean`・`median(None)` の均等分配）は fixture と REQ-2 判定で比較する。
#[test]
fn tie_cases_match_pytorch_where_defined() {
    let fixture = load_fixture();
    assert!(fixture.tie_cases.len() >= 100);
    for case in &fixture.tie_cases {
        let r = run_fixture_case(case);
        check_forward(case, &r, false);
        if let Some(index) = &r.index {
            let x = from_bits(&case.x_bits);
            let inner: usize = case.shape[case.dim.unwrap() + 1..].iter().product();
            let axis = case.shape[case.dim.unwrap()];
            for (lane_out, &idx) in index.iter().enumerate() {
                let (o, i) = (lane_out / inner, lane_out % inner);
                let pos = (o * axis + idx as usize) * inner + i;
                assert!(
                    x[pos] == r.values[lane_out],
                    "{}: x[index] == value",
                    case.name
                );
            }
        }
        if matches!(case.op.as_str(), "nansum" | "nanmean" | "median_all") {
            assert_close_all(
                &r.grad,
                &from_bits(&case.grad_bits),
                &format!("{} grad", case.name),
            );
        }
    }
}

/// 非有限入力（±inf）: forward のクラス一致。勾配は式の IEEE 伝播に従い（拒否・panic
/// しない）、PyTorch との完全一致は受入条件にしない。
#[test]
fn inf_cases_forward_matches_pytorch_and_gradient_is_computed() {
    let fixture = load_fixture();
    assert!(fixture.inf_cases.len() >= 150);
    for case in &fixture.inf_cases {
        let r = run_fixture_case(case);
        check_forward(case, &r, true);
        assert_eq!(r.grad.len(), case.x_bits.len(), "{}", case.name);
    }
}

/// `nansum`／`nanmean` に非有限・0 の上流勾配を流したとき、NaN 位置の勾配は乗算
/// （`0 × g`）で決まる（PyTorch 2.14.0 の実測）。全 NaN lane の `nanmean` は `g / 0`。
#[test]
fn nonfinite_upstream_gradients_match_pytorch() {
    let fixture = load_fixture();
    assert!(fixture.nonfinite_upstream_cases.len() >= 12);
    for case in &fixture.nonfinite_upstream_cases {
        let r = run_fixture_case(case);
        assert_close_all(
            &r.grad,
            &from_bits(&case.grad_bits),
            &format!("{} grad", case.name),
        );
    }
}

#[test]
fn error_cases_agree_with_torch_where_in_scope() {
    let fixture = load_fixture();
    for case in &fixture.error_cases {
        let Some(dim) = (match case.dim {
            None => Some(None),
            Some(d) => usize::try_from(d).ok().map(Some),
        }) else {
            continue; // 負の dim は受けない（決定記録 §5）
        };
        let spec = Spec {
            op: case.op.clone(),
            dim,
            k: case.k,
            q: case.q_bits.map(f32::from_bits),
            interp: case
                .interp
                .as_deref()
                .map_or(QuantileInterpolation::Linear, parse_interp),
        };
        let tape = Tape::new_with_ops(common::naive_ops());
        let n: usize = case.shape.iter().product();
        // 0 次元の fixture 入力は torch 側が 1.0（`torch.tensor(1.0)`）。それ以外は 0 埋め。
        let fill = if case.shape.is_empty() { 1.0 } else { 0.0 };
        let x = tape.var(&t(vec![fill; n], &case.shape));
        let r = apply(&spec, &x);
        if case.torch_raises {
            assert!(
                r.is_err(),
                "{}: torch が例外なら本実装も型付きエラー",
                case.name
            );
        } else if case.shape.is_empty() && spec.dim.is_some() {
            // 0 次元入力への軸指定は torch が受理するが本実装は rank 0 を拒否する（差分）。
            assert!(
                matches!(
                    r,
                    Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
                ),
                "{}",
                case.name
            );
        } else {
            let (y, _) = r.unwrap_or_else(|e| panic!("{}: 受理されるはず: {e:?}", case.name));
            assert_eq!(
                y.to_tensor().shape(),
                case.out_shape.as_slice(),
                "{}",
                case.name
            );
            assert_class_or_bits_eq(
                &y.to_tensor().host_slice(),
                &from_bits(&case.out_bits),
                &case.name,
            );
        }
    }
}

// --- 独立オラクル（中心差分・手計算） ---

/// lane（1 次元）の参照実装（`f64`・ソートは `partial_cmp`）。
fn ref_lane(op: &str, lane: &[f64], k: usize, q: f64) -> f64 {
    let mut s: Vec<f64> = lane.iter().copied().filter(|v| !v.is_nan()).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    match op {
        "median" => s[(n - 1) / 2],
        "kthvalue" => s[k - 1],
        "quantile" => {
            let rank = q * (n as f64 - 1.0);
            let (lo, hi) = (rank.floor() as usize, rank.ceil() as usize);
            s[lo] + (rank - rank.floor()) * (s[hi] - s[lo])
        }
        "nansum" => s.iter().sum(),
        "nanmean" => s.iter().sum::<f64>() / n as f64,
        other => panic!("{other}"),
    }
}

#[test]
fn gradients_match_central_difference_without_ties() {
    // [3, 5] を dim=1 で縮約（3 lane）。各 lane の値は互いに十分離れている。
    let x0: Vec<f32> = (0..15)
        .map(|i| ((i * 7) % 15) as f32 * 0.37 - 2.0)
        .collect();
    let w = [1.0_f32, -0.5, 2.0];
    let cases: Vec<(Spec, &str)> = vec![
        (spec("median_dim", Some(1), None, None), "median"),
        (spec("kthvalue", Some(1), Some(2), None), "kthvalue"),
        (spec("kthvalue", Some(1), Some(5), None), "kthvalue"),
        (spec("quantile", Some(1), None, Some(0.3)), "quantile"),
        (spec("quantile", Some(1), None, Some(0.8)), "quantile"),
        (spec("nansum", Some(1), None, None), "nansum"),
        (spec("nanmean", Some(1), None, None), "nanmean"),
    ];
    for (s, name) in cases {
        let g: Vec<f32> = w.to_vec();
        let r = run_case(common::naive_ops(), &s, x0.clone(), &[3, 5], g);
        let loss = |xs: &[f64]| -> f64 {
            (0..3)
                .map(|row| {
                    f64::from(w[row])
                        * ref_lane(
                            name,
                            &xs[row * 5..(row + 1) * 5],
                            s.k.unwrap_or(1),
                            f64::from(s.q.unwrap_or(0.0)),
                        )
                })
                .sum()
        };
        let base: Vec<f64> = x0.iter().map(|&v| f64::from(v)).collect();
        let h = 1e-3;
        for j in 0..15 {
            let (mut hi, mut lo) = (base.clone(), base.clone());
            hi[j] += h;
            lo[j] -= h;
            let fd = (loss(&hi) - loss(&lo)) / (2.0 * h);
            assert!(
                common::req2_close(f64::from(r.grad[j]), fd),
                "{name} x[{j}]: {} vs {fd}",
                r.grad[j]
            );
        }
    }
}

fn spec(op: &str, dim: Option<usize>, k: Option<usize>, q: Option<f32>) -> Spec {
    Spec {
        op: op.into(),
        dim,
        k,
        q,
        interp: QuantileInterpolation::Linear,
    }
}

#[test]
fn nan_reduction_gradients_match_hand_values_and_central_difference() {
    let x0 = vec![f32::NAN, 2.0, 4.0, f32::NAN, 1.0];
    let r = run_case(
        common::naive_ops(),
        &spec("nansum", None, None, None),
        x0.clone(),
        &[5],
        vec![3.0],
    );
    assert_eq!(r.values, vec![7.0]);
    assert_eq!(r.grad, vec![0.0, 3.0, 3.0, 0.0, 3.0]);
    let r = run_case(
        common::naive_ops(),
        &spec("nanmean", None, None, None),
        x0,
        &[5],
        vec![3.0],
    );
    assert_eq!(r.values[0].to_bits(), ((7.0_f64 / 3.0) as f32).to_bits());
    assert_eq!(r.grad, vec![0.0, 1.0, 1.0, 0.0, 1.0]);
}

// --- 独自契約（タイ・median の規則差） ---

#[test]
fn tie_index_is_stable_and_median_gradient_rules_differ_by_dim() {
    // 昇順: 1, 2(添字 1), 2(添字 2), 2(添字 3), 3 → 中央（位置 2）は添字 2。
    let x = vec![1.0_f32, 2.0, 2.0, 2.0, 3.0];
    let r = run_case(
        common::naive_ops(),
        &spec("median_dim", Some(0), None, None),
        x.clone(),
        &[5],
        vec![3.0],
    );
    assert_eq!(r.index, Some(vec![2]));
    assert_eq!(
        r.grad,
        vec![0.0, 0.0, 3.0, 0.0, 0.0],
        "軸指定は 1 要素へ全量"
    );
    let r = run_case(
        common::naive_ops(),
        &spec("median_all", None, None, None),
        x.clone(),
        &[5],
        vec![3.0],
    );
    assert_eq!(
        r.grad,
        vec![0.0, 1.0, 1.0, 1.0, 0.0],
        "全要素は等しい要素へ均等分配"
    );
    let r = run_case(
        common::naive_ops(),
        &spec("kthvalue", Some(0), Some(2), None),
        x,
        &[5],
        vec![1.0],
    );
    assert_eq!(r.index, Some(vec![1]), "k=2 は安定順で最初の 2");
}

#[test]
fn kthvalue_gradient_goes_to_selected_element_on_non_last_dim() {
    // [[4,1],[2,9],[3,5]] を dim=0 の k=2: 列 0 → 3（行 2）、列 1 → 5（行 2）。
    let x = vec![4.0_f32, 1.0, 2.0, 9.0, 3.0, 5.0];
    let r = run_case(
        common::naive_ops(),
        &spec("kthvalue", Some(0), Some(2), None),
        x,
        &[3, 2],
        vec![1.0, 2.0],
    );
    assert_eq!(r.values, vec![3.0, 5.0]);
    assert_eq!(r.index, Some(vec![2, 2]));
    assert_eq!(r.grad, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
}

// --- フォールバックとエラー伝播 ---

/// `stat_*` だけを差し替える `BackendOps`。それ以外は naive へ委譲する。
struct StatMock {
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

impl StatMock {
    fn pair(&self) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            // 4 要素入力を dim=0 で縮約すると出力は 0 次元。わざと [3]／[4] を返す。
            Mode::WrongValueShape => Ok((
                t(vec![0.0; 3], &[3]),
                Tensor::new(vec![0_i32; 1], &[]).unwrap(),
            )),
            Mode::WrongIndexShape => Ok((
                t(vec![0.0; 1], &[]),
                Tensor::new(vec![0_i32; 3], &[3]).unwrap(),
            )),
        }
    }

    fn value(&self) -> Result<Tensor<f32>, BackendError> {
        self.pair().map(|(v, _)| v)
    }
}

impl BackendOps for StatMock {
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
    fn stat_kthvalue(
        &self,
        _x: &Tensor<f32>,
        _k: usize,
        _dim: usize,
    ) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        self.pair()
    }
    fn stat_median_dim(
        &self,
        _x: &Tensor<f32>,
        _dim: usize,
    ) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        self.pair()
    }
    fn stat_median_all(&self, _x: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.value()
    }
    fn stat_quantile(
        &self,
        _x: &Tensor<f32>,
        _q: f32,
        _dim: Option<usize>,
        _interpolation: QuantileInterpolation,
    ) -> Result<Tensor<f32>, BackendError> {
        self.value()
    }
    fn stat_nansum(
        &self,
        _x: &Tensor<f32>,
        _dim: Option<usize>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.value()
    }
    fn stat_nanmean(
        &self,
        _x: &Tensor<f32>,
        _dim: Option<usize>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.value()
    }
}

fn mock_tape(mode: Mode) -> (Tape, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let tape = Tape::new_with_ops(Box::new(StatMock {
        inner: common::naive_ops(),
        calls: Arc::clone(&calls),
        mode,
    }));
    (tape, calls)
}

/// 全 6 関数を 1-D の 4 要素入力へ適用する（結果の `Result<(), _>` の列）。
fn call_all(x: &Var<'_>) -> Vec<Result<(), AutodiffError>> {
    vec![
        kthvalue(x, 2, 0).map(|_| ()),
        median_with_indices(x, 0).map(|_| ()),
        median(x, None).map(|_| ()),
        quantile(x, 0.5, Some(0), QuantileInterpolation::Linear).map(|_| ()),
        nansum(x, None).map(|_| ()),
        nanmean(x, None).map(|_| ()),
    ]
}

#[test]
fn unsupported_backend_falls_back_to_host_reference() {
    let (tape, calls) = mock_tape(Mode::Unsupported);
    let x = tape.var(&t(vec![3.0, 1.0, 5.0, 2.0], &[4]));
    let (v, i) = kthvalue(&x, 2, 0).unwrap();
    assert_eq!(v.to_tensor().host_slice().into_owned(), [2.0]);
    assert_eq!(i.contiguous().host_slice().into_owned(), [3]);
    let (v, i) = median_with_indices(&x, 0).unwrap();
    assert_eq!(v.to_tensor().host_slice().into_owned(), [2.0]);
    assert_eq!(i.contiguous().host_slice().into_owned(), [3]);
    assert_eq!(
        median(&x, None)
            .unwrap()
            .to_tensor()
            .host_slice()
            .into_owned(),
        [2.0]
    );
    let q = quantile(&x, 0.5, Some(0), QuantileInterpolation::Linear).unwrap();
    assert_eq!(q.to_tensor().host_slice().into_owned(), [2.5]);
    assert_eq!(
        nansum(&x, None)
            .unwrap()
            .to_tensor()
            .host_slice()
            .into_owned(),
        [11.0]
    );
    assert_eq!(
        nanmean(&x, Some(0))
            .unwrap()
            .to_tensor()
            .host_slice()
            .into_owned(),
        [2.75]
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        6,
        "バックエンドを先に呼んでからフォールバックしているはず"
    );
}

#[test]
fn non_unsupported_backend_errors_are_propagated() {
    let (tape, _) = mock_tape(Mode::LaunchFailed);
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    for r in call_all(&x) {
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
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        let rs = call_all(&x);
        // 値のみを返す 4 関数は値の shape、2 出力の 2 関数は値・索引の両方を検証する。
        let want_err = |idx: usize| match mode {
            Mode::WrongValueShape => true,
            Mode::WrongIndexShape => idx < 2,
            _ => false,
        };
        for (idx, r) in rs.into_iter().enumerate() {
            if want_err(idx) {
                assert!(
                    matches!(
                        r,
                        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
                    ),
                    "関数 {idx}"
                );
            } else {
                assert!(
                    r.is_ok(),
                    "関数 {idx}（索引のみ誤 shape は値のみ関数に無関係）"
                );
            }
        }
    }
}

// --- 境界・テープ・決定性 ---

#[test]
fn invalid_arguments_are_typed_errors() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    assert!(matches!(
        kthvalue(&x, 0, 0),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        kthvalue(&x, 4, 0),
        Err(AutodiffError::InvalidArgument(_))
    ));
    for q in [-0.1_f32, 1.5, f32::NAN, f32::INFINITY] {
        assert!(
            matches!(
                quantile(&x, q, None, QuantileInterpolation::Linear),
                Err(AutodiffError::InvalidArgument(_))
            ),
            "q={q}"
        );
    }
    assert!(matches!(
        median_with_indices(&x, 1),
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            axis: 1,
            rank: 1
        }))
    ));
    assert!(matches!(
        nansum(&x, Some(3)),
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
    ));
    let s = tape.var(&t(vec![1.0], &[]));
    assert!(matches!(
        nanmean(&s, Some(0)),
        Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            rank: 0,
            ..
        }))
    ));
    // 0 次元入力でも全要素版は受理する。
    assert_eq!(
        median(&s, None)
            .unwrap()
            .to_tensor()
            .host_slice()
            .into_owned(),
        [1.0]
    );
}

#[test]
fn empty_axis_rules() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![], &[0]));
    assert!(matches!(
        median_with_indices(&x, 0),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        kthvalue(&x, 1, 0),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        quantile(&x, 0.5, None, QuantileInterpolation::Linear),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert_eq!(
        nansum(&x, Some(0))
            .unwrap()
            .to_tensor()
            .host_slice()
            .into_owned(),
        [0.0]
    );
    assert!(nanmean(&x, None).unwrap().to_tensor().host_slice()[0].is_nan());
    assert!(median(&x, None).unwrap().to_tensor().host_slice()[0].is_nan());
    let y = tape.var(&t(vec![], &[0, 2]));
    assert_eq!(
        nansum(&y, Some(0))
            .unwrap()
            .to_tensor()
            .host_slice()
            .into_owned(),
        [0.0, 0.0]
    );
}

#[test]
fn huge_broadcast_view_is_rejected_before_allocation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let base = tape.var(&t(vec![1.0], &[1]));
    let huge = base.broadcast_to(&[1usize << 61]).unwrap();
    let rs = [
        kthvalue(&huge, 1, 0).map(|_| ()),
        median_with_indices(&huge, 0).map(|_| ()),
        median(&huge, None).map(|_| ()),
        quantile(&huge, 0.5, None, QuantileInterpolation::Linear).map(|_| ()),
        nansum(&huge, None).map(|_| ()),
        nanmean(&huge, Some(0)).map(|_| ()),
    ];
    for r in rs {
        assert!(matches!(
            r,
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
    }
}

#[test]
fn each_op_records_exactly_one_node() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 3.0, 2.0], &[3]));
    let mut expected = tape.len();
    kthvalue(&x, 1, 0).unwrap();
    median_with_indices(&x, 0).unwrap();
    median(&x, None).unwrap();
    quantile(&x, 0.5, None, QuantileInterpolation::Lower).unwrap();
    nansum(&x, None).unwrap();
    nanmean(&x, None).unwrap();
    for _ in 0..6 {
        expected += 1;
    }
    assert_eq!(tape.len(), expected);
    // `median(x, Some(d))` も 1 ノード（`median_with_indices` へ委譲）。
    median(&x, Some(0)).unwrap();
    assert_eq!(tape.len(), expected + 1);
}

#[test]
fn create_graph_is_a_typed_error_not_a_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let loss = nansum(&x, None).unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
}

#[test]
fn results_are_bit_deterministic_run_to_run() {
    let fixture = load_fixture();
    for case in fixture
        .finite_cases
        .iter()
        .chain(&fixture.tie_cases)
        .filter(|c| c.shape.len() == 3 || c.name.contains("tied"))
    {
        let a = run_fixture_case(case);
        let b = run_fixture_case(case);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.values), bits(&b.values), "{}", case.name);
        assert_eq!(bits(&a.grad), bits(&b.grad), "{}", case.name);
        assert_eq!(a.index, b.index, "{}", case.name);
    }
}
