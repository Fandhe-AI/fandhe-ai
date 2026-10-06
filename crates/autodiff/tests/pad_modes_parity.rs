//! `pad_ops::pad_with_mode`（イシュー #2642・reflect／replicate／circular）の
//! `Tape`／`Var` を経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/pad-modes-pytorch-reference/pad_modes_reference.json`・
//!   生成条件は同ディレクトリの `README.md`）と突合する。forward は算術を含まない
//!   コピーなので **bit 一致**（NaN はクラス一致）、勾配は REQ-2 統一複合判定
//!   （`common::req2_close`。tolerance 定数は新設しない）。
//! - `common::naive_ops()` は `pad_modes_forward` を override しないため、必ず共有
//!   ホストカーネルへのフォールバック経路を通る。CPU `BackendOps` 実装との一致は
//!   `crates/facade/tests/pad_modes_ops_backend_parity.rs` が担当する。
//! - PyTorch が拒否し本実装が受理する形（任意軸・任意 rank への一般化）は「意図的な
//!   差分」として表で明示し、独立オラクル（符号付き整数の別実装）で検証する。
//! - フォールバックは `Unsupported` のときだけで、それ以外のバックエンドエラーは
//!   握りつぶさず伝播する。確保前サイズ検査・テープ記録数・VJP の独立オラクル・
//!   内積恒等式・中心差分・run-to-run 決定性も固定する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::pad_ops::pad_with_mode;
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, PadMode, ShapeError, Tensor};
use serde::Deserialize;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fn bits_of(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn parse_mode(s: &str) -> PadMode {
    match s {
        "reflect" => PadMode::Reflect,
        "replicate" => PadMode::Replicate,
        "circular" => PadMode::Circular,
        other => panic!("未知の mode: {other}"),
    }
}

fn to_pairs(p: &[[usize; 2]]) -> Vec<(usize, usize)> {
    p.iter().map(|&[b, a]| (b, a)).collect()
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
    mode: String,
    shape: Vec<usize>,
    pads: Vec<[usize; 2]>,
    x_bits: Vec<u32>,
    #[serde(default)]
    g_bits: Vec<u32>,
    out_shape: Vec<usize>,
    out_bits: Vec<u32>,
    #[serde(default)]
    grad_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct ErrCase {
    name: String,
    mode: String,
    shape: Vec<usize>,
    pads: Vec<[usize; 2]>,
    torch_raises: bool,
    out_shape: Vec<usize>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pad-modes-pytorch-reference/pad_modes_reference.json");
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
    out_shape: Vec<usize>,
    values: Vec<f32>,
    grad: Vec<f32>,
}

/// `(out * g).sum()` を損失とした forward 値・入力勾配を返す。
fn run_case(
    ops: Box<dyn BackendOps + Send>,
    mode: PadMode,
    x: Vec<f32>,
    shape: &[usize],
    pads: &[(usize, usize)],
    g: Option<Vec<f32>>,
) -> Run {
    let tape = Tape::new_with_ops(ops);
    let xv = tape.var(&t(x, shape));
    let y = pad_with_mode(&xv, pads, mode).expect("pad_with_mode");
    let yt = y.to_tensor();
    let out_shape = yt.shape().to_vec();
    let values = yt.host_slice().into_owned();
    let g = g.unwrap_or_else(|| vec![1.0; values.len()]);
    let gv = tape.var_no_grad(&t(g, &out_shape));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("入力へ勾配が届く").clone();
    Run {
        out_shape,
        values,
        grad: dx.host_slice().into_owned(),
    }
}

fn run_fixture_case(case: &Case) -> Run {
    let g = if case.g_bits.is_empty() {
        None
    } else {
        Some(from_bits(&case.g_bits))
    };
    run_case(
        common::naive_ops(),
        parse_mode(&case.mode),
        from_bits(&case.x_bits),
        &case.shape,
        &to_pairs(&case.pads),
        g,
    )
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
    assert!(fixture.finite_cases.len() >= 100);
    for mode in ["reflect", "replicate", "circular"] {
        assert!(
            fixture.finite_cases.iter().any(|c| c.mode == mode),
            "{mode} のケースが無い"
        );
    }
    for case in &fixture.finite_cases {
        let r = run_fixture_case(case);
        assert_eq!(r.out_shape, case.out_shape, "{}: out shape", case.name);
        // 算術を含まないコピーなので bit 一致。
        assert_class_or_bits_eq(
            &r.values,
            &from_bits(&case.out_bits),
            &format!("{} forward", case.name),
        );
        assert_close_all(
            &r.grad,
            &from_bits(&case.grad_bits),
            &format!("{} grad", case.name),
        );
    }
}

#[test]
fn nonfinite_forward_matches_pytorch_reference_bits() {
    let fixture = load_fixture();
    assert!(fixture.nonfinite_cases.len() >= 6);
    for case in &fixture.nonfinite_cases {
        let r = run_fixture_case(case);
        assert_eq!(r.out_shape, case.out_shape, "{}", case.name);
        assert_class_or_bits_eq(
            &r.values,
            &from_bits(&case.out_bits),
            &format!("{} forward", case.name),
        );
    }
}

/// torch が拒否し Rust も拒否する形（型付きエラー）。
const BOTH_REJECT: &[&str] = &[
    "reflect_pad_eq_len",
    "reflect_pad_gt_len",
    "circular_pad_gt_len",
    "replicate_empty_axis",
    "reflect_empty_axis",
    "circular_empty_axis",
];

/// torch が拒否するが Rust は受理する形（任意軸・任意 rank への一般化＝意図的な差分。
/// `docs/autodiff-pad-modes-decision.md` §5）。
const RUST_ACCEPTS_SUPERSET: &[&str] = &[
    "replicate_rank1",
    "reflect_rank1",
    "circular_rank1",
    "replicate_leading_axis",
    "replicate_rank2_2ax",
    "replicate_zero_pad",
];

#[test]
fn error_cases_agree_with_torch_or_are_listed_intentional_differences() {
    let fixture = load_fixture();
    for case in &fixture.error_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let n: usize = case.shape.iter().product();
        let x = tape.var(&t(vec![0.5; n], &case.shape));
        let pads = to_pairs(&case.pads);
        let r = pad_with_mode(&x, &pads, parse_mode(&case.mode));
        if case.torch_raises {
            if BOTH_REJECT.contains(&case.name.as_str()) {
                assert!(
                    matches!(r, Err(AutodiffError::InvalidArgument(_))),
                    "{}: {:?}",
                    case.name,
                    r.map(|_| ())
                );
            } else {
                assert!(
                    RUST_ACCEPTS_SUPERSET.contains(&case.name.as_str()),
                    "{}: torch が拒否するケースは BOTH_REJECT か RUST_ACCEPTS_SUPERSET に分類する",
                    case.name
                );
                let y = r.unwrap_or_else(|e| panic!("{}: 受理するはず: {e:?}", case.name));
                let want: Vec<usize> = case
                    .shape
                    .iter()
                    .zip(&pads)
                    .map(|(&s, &(b, a))| s + b + a)
                    .collect();
                assert_eq!(y.to_tensor().shape(), want.as_slice(), "{}", case.name);
            }
        } else {
            assert!(
                !BOTH_REJECT.contains(&case.name.as_str())
                    && !RUST_ACCEPTS_SUPERSET.contains(&case.name.as_str()),
                "{}: torch が受理するケースは分類表に載せない",
                case.name
            );
            let y = r.unwrap_or_else(|e| panic!("{}: 受理するはず: {e:?}", case.name));
            assert_eq!(
                y.to_tensor().shape(),
                case.out_shape.as_slice(),
                "{}",
                case.name
            );
        }
    }
}

// --- 独立オラクル（符号付き整数での別実装） ---

fn oracle_map(mode: PadMode, len: i64, before: i64, o: i64) -> i64 {
    let p = o - before;
    match mode {
        PadMode::Reflect => {
            if len == 1 {
                return 0;
            }
            let period = 2 * (len - 1);
            let m = p.rem_euclid(period);
            if m >= len { period - m } else { m }
        }
        PadMode::Replicate => p.clamp(0, len - 1),
        PadMode::Circular => p.rem_euclid(len),
        _ => unreachable!(),
    }
}

/// 出力の平坦添字 → 入力の平坦添字の表。
fn oracle_sources(shape: &[usize], pads: &[(usize, usize)], mode: PadMode) -> Vec<usize> {
    let out_shape: Vec<usize> = shape
        .iter()
        .zip(pads)
        .map(|(&s, &(b, a))| s + b + a)
        .collect();
    let n: usize = out_shape.iter().product();
    let mut table = Vec::with_capacity(n);
    for flat in 0..n {
        let mut rem = flat;
        let mut coords = vec![0usize; shape.len()];
        for a in (0..shape.len()).rev() {
            coords[a] = rem % out_shape[a];
            rem /= out_shape[a];
        }
        let mut src = 0usize;
        for a in 0..shape.len() {
            let i = oracle_map(mode, shape[a] as i64, pads[a].0 as i64, coords[a] as i64);
            src = src * shape[a] + usize::try_from(i).unwrap();
        }
        table.push(src);
    }
    table
}

fn lcg_values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

/// (shape, pads, mode)。先頭軸・全軸・rank 1・rank 5 を含む（PyTorch が受けない形も）。
type OracleCase = (Vec<usize>, Vec<(usize, usize)>, PadMode);

fn oracle_cases() -> Vec<OracleCase> {
    let mut v = Vec::new();
    for mode in [PadMode::Reflect, PadMode::Replicate, PadMode::Circular] {
        v.push((vec![5], vec![(2, 3)], mode));
        v.push((vec![4, 3], vec![(2, 1), (1, 2)], mode));
        v.push((vec![3, 4, 5], vec![(2, 0), (0, 0), (0, 3)], mode));
        v.push((
            vec![2, 3, 2, 3, 4],
            vec![(1, 1), (2, 1), (0, 1), (1, 0), (3, 2)],
            mode,
        ));
        v.push((vec![1, 3], vec![(0, 0), (2, 2)], mode));
    }
    // reflect の 1 軸 len1 は pad 0 のみ。replicate は軸長超え。
    v.push((vec![2, 1], vec![(0, 0), (4, 5)], PadMode::Replicate));
    v.push((vec![3], vec![(7, 9)], PadMode::Replicate));
    v
}

#[test]
fn forward_and_vjp_match_independent_oracle_on_any_axis_and_rank() {
    for (shape, pads, mode) in oracle_cases() {
        let n: usize = shape.iter().product();
        let x = lcg_values(n, 7);
        let table = oracle_sources(&shape, &pads, mode);
        let g = lcg_values(table.len(), 11);
        let r = run_case(
            common::naive_ops(),
            mode,
            x.clone(),
            &shape,
            &pads,
            Some(g.clone()),
        );
        let want: Vec<f32> = table.iter().map(|&s| x[s]).collect();
        assert_eq!(
            bits_of(&r.values),
            bits_of(&want),
            "forward {mode:?} {shape:?} {pads:?}"
        );
        let mut acc = vec![0.0f64; n];
        for (o, &s) in table.iter().enumerate() {
            acc[s] += f64::from(g[o]);
        }
        for (j, (&got, &w)) in r.grad.iter().zip(&acc).enumerate() {
            assert!(
                common::req2_close(f64::from(got), w),
                "vjp {mode:?} {shape:?} {pads:?} x[{j}]: {got} vs {w}"
            );
        }
        // 内積恒等式 <pad(x), g> = <x, vjp(g)>。
        let lhs: f64 = r
            .values
            .iter()
            .zip(&g)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let rhs: f64 = x
            .iter()
            .zip(&r.grad)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        assert!(common::req2_close(lhs, rhs), "内積恒等式 {lhs} vs {rhs}");
    }
}

#[test]
fn gradient_matches_central_difference() {
    let shape = [3usize, 4];
    let pads = [(2usize, 1usize), (1, 3)];
    let x0 = lcg_values(12, 3);
    for mode in [PadMode::Reflect, PadMode::Replicate, PadMode::Circular] {
        let table = oracle_sources(&shape, &pads, mode);
        let w = lcg_values(table.len(), 5);
        let r = run_case(
            common::naive_ops(),
            mode,
            x0.clone(),
            &shape,
            &pads,
            Some(w.clone()),
        );
        let loss = |xs: &[f64]| -> f64 {
            table
                .iter()
                .zip(&w)
                .map(|(&s, &wv)| xs[s] * f64::from(wv))
                .sum()
        };
        let base: Vec<f64> = x0.iter().map(|&v| f64::from(v)).collect();
        let h = 1e-4;
        for j in 0..12 {
            let (mut hi, mut lo) = (base.clone(), base.clone());
            hi[j] += h;
            lo[j] -= h;
            let fd = (loss(&hi) - loss(&lo)) / (2.0 * h);
            assert!(
                common::req2_close(f64::from(r.grad[j]), fd),
                "{mode:?} x[{j}]: {} vs {fd}",
                r.grad[j]
            );
        }
    }
}

#[test]
fn replicate_edge_gradient_accumulates_pad_plus_one_times() {
    // (2,3) の replicate: 先頭要素は 3 回・末尾要素は 4 回分の上流を受け取る。
    let r = run_case(
        common::naive_ops(),
        PadMode::Replicate,
        vec![1.0, 2.0, 3.0, 4.0],
        &[4],
        &[(2, 3)],
        None,
    );
    assert_eq!(r.grad, vec![3.0, 1.0, 1.0, 4.0]);
}

#[test]
fn gradient_uses_f64_accumulator_for_cancelling_columns() {
    // 同一入力要素へ 1e8, 1, -1e8 が流れる。f32 蓄積なら 0 になる。
    let r = run_case(
        common::naive_ops(),
        PadMode::Replicate,
        vec![0.0],
        &[1],
        &[(1, 1)],
        Some(vec![1e8, 1.0, -1e8]),
    );
    assert_eq!(r.grad, vec![1.0]);
}

// --- フォールバックとエラー伝播 ---

/// `pad_modes_forward` だけを差し替える `BackendOps`。それ以外は naive へ委譲する。
struct PadMock {
    inner: Box<dyn BackendOps + Send>,
    calls: Arc<AtomicUsize>,
    mode: Mode,
}

#[derive(Clone, Copy)]
enum Mode {
    Unsupported,
    LaunchFailed,
    WrongShape,
}

impl BackendOps for PadMock {
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
    fn pad_modes_forward(
        &self,
        _input: &Tensor<f32>,
        _pads: &[(usize, usize)],
        _mode: PadMode,
    ) -> Result<Tensor<f32>, BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            Mode::WrongShape => Ok(t(vec![0.0; 3], &[3])),
        }
    }
}

fn mock_tape(mode: Mode) -> (Tape, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let tape = Tape::new_with_ops(Box::new(PadMock {
        inner: common::naive_ops(),
        calls: Arc::clone(&calls),
        mode,
    }));
    (tape, calls)
}

#[test]
fn unsupported_backend_falls_back_to_host_reference_exactly_once() {
    let (tape, calls) = mock_tape(Mode::Unsupported);
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let y = pad_with_mode(&x, &[(2, 3)], PadMode::Reflect).unwrap();
    assert_eq!(
        y.to_tensor().host_slice().into_owned(),
        [3., 2., 1., 2., 3., 4., 3., 2., 1.]
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "バックエンドを先に 1 回呼んでからフォールバックしているはず"
    );
}

#[test]
fn non_unsupported_backend_errors_are_propagated_and_tape_is_untouched() {
    let (tape, _) = mock_tape(Mode::LaunchFailed);
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let before = tape.len();
    assert!(matches!(
        pad_with_mode(&x, &[(1, 1)], PadMode::Replicate),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    assert_eq!(tape.len(), before);
}

#[test]
fn wrong_shape_from_backend_is_a_typed_error_and_tape_is_untouched() {
    let (tape, _) = mock_tape(Mode::WrongShape);
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let before = tape.len();
    assert!(matches!(
        pad_with_mode(&x, &[(1, 1)], PadMode::Circular),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
    assert_eq!(tape.len(), before);
}

// --- 事前検査（バックエンド呼び出し前に拒否） ---

#[test]
fn invalid_inputs_are_rejected_before_backend_call() {
    let (tape, calls) = mock_tape(Mode::Unsupported);
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let before = tape.len();
    // rank 不一致。
    assert!(matches!(
        pad_with_mode(&x, &[(1, 1)], PadMode::Replicate),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    // 各モードの pad 上限違反。
    for (mode, p) in [
        (PadMode::Reflect, (2usize, 0usize)),
        (PadMode::Reflect, (0, 2)),
        (PadMode::Circular, (3, 0)),
        (PadMode::Circular, (0, 3)),
    ] {
        assert!(
            matches!(
                pad_with_mode(&x, &[(0, 0), p], mode),
                Err(AutodiffError::InvalidArgument(_))
            ),
            "{mode:?} {p:?}"
        );
    }
    // 加算オーバーフロー・1 GiB 上限超過（実体化前の拒否）。
    let one = tape.var(&t(vec![1.0], &[1]));
    assert!(matches!(
        pad_with_mode(&one, &[(usize::MAX, 1)], PadMode::Replicate),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
    assert!(matches!(
        pad_with_mode(&one, &[(0, 1usize << 29)], PadMode::Replicate),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0, "バックエンドは呼ばれない");
    // `one` の 1 ノードだけが増え、失敗した呼び出しはノードを積まない。
    assert_eq!(tape.len(), before + 1, "tape ノードは増えない");
}

#[test]
fn huge_broadcast_view_is_rejected_before_allocation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let base = tape.var(&t(vec![1.0], &[1]));
    let huge = base.broadcast_to(&[1usize << 61]).unwrap();
    assert!(pad_with_mode(&huge, &[(1, 1)], PadMode::Replicate).is_err());
}

#[test]
fn empty_axis_zero_pads_and_rank0_are_handled() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // パディングのない軸の長さ 0 は空出力。
    let e = tape.var(&t(vec![], &[0, 3]));
    let y = pad_with_mode(&e, &[(0, 0), (1, 1)], PadMode::Reflect).unwrap();
    assert_eq!(y.to_tensor().shape(), &[0, 5]);
    // 全 0 pad は恒等コピー。
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let y = pad_with_mode(&x, &[(0, 0)], PadMode::Circular).unwrap();
    assert_eq!(y.to_tensor().host_slice().into_owned(), [1.0, 2.0, 3.0]);
    // rank 0。
    let s = tape.var(&t(vec![5.0], &[]));
    let y = pad_with_mode(&s, &[], PadMode::Reflect).unwrap();
    assert_eq!(y.to_tensor().host_slice().into_owned(), [5.0]);
}

#[test]
fn non_contiguous_view_input_is_padded_in_logical_order() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let base = tape.var(&t(vec![1.0, 4.0, 2.0, 5.0, 3.0, 0.0], &[2, 3]));
    let tr = base.transpose(0, 1).unwrap(); // [[1,5],[4,3],[2,0]]
    let y = pad_with_mode(&tr, &[(0, 0), (1, 1)], PadMode::Replicate).unwrap();
    assert_eq!(
        y.to_tensor().host_slice().into_owned(),
        [1., 1., 5., 5., 4., 4., 3., 3., 2., 2., 0., 0.]
    );
}

// --- テープ・連鎖・決定性 ---

#[test]
fn each_call_records_exactly_one_node() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 3.0, 2.0], &[3]));
    let before = tape.len();
    pad_with_mode(&x, &[(1, 2)], PadMode::Circular).unwrap();
    assert_eq!(tape.len(), before + 1);
}

#[test]
fn chained_backward_and_shared_input_accumulate() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    // 同じ入力を 2 回使う: replicate(1,1) の和の勾配 = 各 2 回分。
    let a = pad_with_mode(&x, &[(1, 1)], PadMode::Replicate).unwrap();
    let b = pad_with_mode(&x, &[(1, 1)], PadMode::Replicate).unwrap();
    let loss = a.add(&b).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("勾配").clone();
    // 1 回分: [2, 1, 2]（端は pad + 1 = 2 回）。2 回で倍。
    assert_eq!(dx.host_slice().into_owned(), vec![4.0, 2.0, 4.0]);
    // 他演算との連鎖（mul → pad → sum）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let two = tape.var_no_grad(&t(vec![2.0, 2.0, 2.0], &[3]));
    let y = pad_with_mode(&x.mul(&two).unwrap(), &[(2, 0)], PadMode::Reflect).unwrap();
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("勾配").clone();
    // reflect (2,0) of len 3: out[0]=in[2], out[1]=in[1], 続いて in[0..3]。
    assert_eq!(dx.host_slice().into_owned(), vec![2.0, 4.0, 4.0]);
}

#[test]
fn create_graph_is_a_typed_error_not_a_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let loss = pad_with_mode(&x, &[(1, 1)], PadMode::Replicate)
        .unwrap()
        .sum(None)
        .unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
}

#[test]
fn results_are_bit_deterministic_run_to_run() {
    let fixture = load_fixture();
    for case in fixture
        .finite_cases
        .iter()
        .filter(|c| c.name.contains("r5_3ax"))
    {
        let a = run_fixture_case(case);
        let b = run_fixture_case(case);
        assert_eq!(bits_of(&a.values), bits_of(&b.values), "{}", case.name);
        assert_eq!(bits_of(&a.grad), bits_of(&b.grad), "{}", case.name);
    }
}
