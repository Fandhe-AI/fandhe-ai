//! `fft_ops`（イシュー #2631・`rfft`／`irfft`）の `Tape`/`Var` を経由する
//! end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/fft-pytorch-reference/fft_reference.json`・生成条件は
//!   同ディレクトリの `README.md`）の forward 出力と入力勾配を REQ-2 統一
//!   複合判定（`common::req2_close`）で突合する。tolerance 定数は新設しない。
//! - DFT 行列積オラクル（`docs/autodiff-fft-design.md` §4 案 A）との forward・
//!   backward 突合、中心差分による勾配検査、`rfft → irfft` 往復、DC／Nyquist
//!   虚部の bit 0、run-to-run の bit 決定性、境界エラー、非有限入力の伝播、
//!   `Unsupported` 以外のバックエンドエラーを握りつぶさないことを固定する。
//!
//! `common::naive_ops()` は `fft_*` をオーバーライドしないため、必ず共有ホスト
//! カーネル（`fandhe_ai_tensor_core::fft`）へのフォールバック経路を通る。
//! CPU `BackendOps` 実装との一致は `crates/facade/tests/fft_ops_backend_parity.rs`
//! が担当する。

mod common;

use std::path::PathBuf;

use fandhe_ai_autodiff::fft_ops::{irfft, rfft};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, FftNorm, ShapeError, Tensor};
use serde::Deserialize;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    cases: Vec<Case>,
    error_cases: Vec<ErrorCase>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    op: String,
    in_shape: Vec<usize>,
    n: Option<usize>,
    dim: Option<usize>,
    norm: String,
    input: Vec<f32>,
    grad_out: Vec<f32>,
    out_shape: Vec<usize>,
    output: Vec<f32>,
    grad_in: Vec<f32>,
}

#[derive(Deserialize)]
struct ErrorCase {
    name: String,
    op: String,
    in_shape: Vec<usize>,
    n: Option<usize>,
    dim: Option<usize>,
    torch_raises: bool,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fft-pytorch-reference/fft_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

fn norm_of(s: &str) -> FftNorm {
    match s {
        "backward" => FftNorm::Backward,
        "ortho" => FftNorm::Ortho,
        "forward" => FftNorm::Forward,
        other => panic!("未知の norm: {other}"),
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

/// fixture 1 ケースを実行し、forward 出力と入力勾配を返す。
fn run_case(case: &Case) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(case.input.clone(), &case.in_shape));
    let norm = norm_of(&case.norm);
    let y = match case.op.as_str() {
        "rfft" => rfft(&x, case.n, case.dim, norm),
        "irfft" => irfft(&x, case.n, case.dim, norm),
        other => panic!("未知の op: {other}"),
    }
    .unwrap_or_else(|e| panic!("{}: forward が失敗: {e}", case.name));
    let out = y.to_tensor();
    let g = tape.var_no_grad(&t(case.grad_out.clone(), &case.out_shape));
    let loss = y.mul(&g).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("入力へ勾配が届く").clone();
    (
        out.host_slice().into_owned(),
        out.shape().to_vec(),
        dx.host_slice().into_owned(),
    )
}

#[test]
fn matches_pytorch_reference_forward_and_backward() {
    let fixture = load_fixture();
    assert!(
        fixture.torch_version.starts_with("2.14.0"),
        "fixture は PyTorch 2.14.0 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    assert!(!fixture.cases.is_empty());
    for case in &fixture.cases {
        let (out, out_shape, dx) = run_case(case);
        assert_eq!(out_shape, case.out_shape, "{}: 出力 shape", case.name);
        assert_close_all(&out, &case.output, &format!("{} forward", case.name));
        assert_eq!(dx.len(), case.grad_in.len(), "{}: 勾配長", case.name);
        assert_close_all(&dx, &case.grad_in, &format!("{} grad", case.name));
    }
}

#[test]
fn error_cases_follow_pytorch_rejections() {
    let fixture = load_fixture();
    for case in &fixture.error_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let numel: usize = case.in_shape.iter().product();
        let x = tape.var(&t(vec![0.0; numel], &case.in_shape));
        let result = match case.op.as_str() {
            "rfft" => rfft(&x, case.n, case.dim, FftNorm::Backward),
            _ => irfft(&x, case.n, case.dim, FftNorm::Backward),
        };
        assert_eq!(
            result.is_err(),
            case.torch_raises,
            "{}: torch の拒否有無と一致しない",
            case.name
        );
    }
}

// --- DFT 行列積オラクル ---

fn dft_cos(rows: usize, cols: usize, n: usize, weight: impl Fn(usize) -> f64) -> Vec<f32> {
    // 行 = 時間 j or bin k の指定は呼び出し側。ここでは (r, c) → cos(2π r c / n)。
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let th = 2.0 * std::f64::consts::PI * ((r * c) % n) as f64 / n as f64;
            out.push((weight(r) * weight(c) * th.cos()) as f32);
        }
    }
    out
}

fn dft_sin(rows: usize, cols: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let th = 2.0 * std::f64::consts::PI * ((r * c) % n) as f64 / n as f64;
            out.push(th.sin() as f32);
        }
    }
    out
}

fn seeded(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn rfft_matches_dft_matrix_oracle() {
    for (b, n, norm) in [
        (2usize, 8usize, FftNorm::Backward),
        (3, 5, FftNorm::Ortho),
        (1, 6, FftNorm::Forward),
    ] {
        let bins = n / 2 + 1;
        let scale = match norm {
            FftNorm::Backward => 1.0,
            FftNorm::Ortho => 1.0 / (n as f64).sqrt(),
            _ => 1.0 / n as f64,
        } as f32;
        let xs = seeded(b * n, 17);
        let g_re = seeded(b * bins, 18);
        let g_im = seeded(b * bins, 19);

        // 本実装
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(xs.clone(), &[b, n]));
        let y = rfft(&x, None, None, norm).unwrap();
        let mut g = vec![0.0f32; b * bins * 2];
        for i in 0..b * bins {
            g[2 * i] = g_re[i];
            g[2 * i + 1] = g_im[i];
        }
        let gv = tape.var_no_grad(&t(g, &[b, bins, 2]));
        let loss = y.mul(&gv).unwrap().sum(None).unwrap();
        let dx = tape
            .backward(&loss)
            .unwrap()
            .get(&x)
            .unwrap()
            .unwrap()
            .clone();

        // オラクル（matmul 合成）
        let tape2 = Tape::new_with_ops(common::naive_ops());
        let x2 = tape2.var(&t(xs, &[b, n]));
        let c: Vec<f32> = dft_cos(n, bins, n, |_| 1.0)
            .iter()
            .map(|v| v * scale)
            .collect();
        let s: Vec<f32> = dft_sin(n, bins, n).iter().map(|v| -v * scale).collect();
        let re = x2.matmul(&tape2.var_no_grad(&t(c, &[n, bins]))).unwrap();
        let im = x2.matmul(&tape2.var_no_grad(&t(s, &[n, bins]))).unwrap();
        let l_re = re
            .mul(&tape2.var_no_grad(&t(g_re, &[b, bins])))
            .unwrap()
            .sum(None)
            .unwrap();
        let l_im = im
            .mul(&tape2.var_no_grad(&t(g_im, &[b, bins])))
            .unwrap()
            .sum(None)
            .unwrap();
        let loss2 = l_re.add(&l_im).unwrap();
        let dx2 = tape2
            .backward(&loss2)
            .unwrap()
            .get(&x2)
            .unwrap()
            .unwrap()
            .clone();

        let (yv, re_v, im_v) = (
            y.to_tensor().host_slice().into_owned(),
            re.to_tensor().host_slice().into_owned(),
            im.to_tensor().host_slice().into_owned(),
        );
        for i in 0..b * bins {
            let ctx = format!("rfft oracle b={b} n={n} {norm:?}");
            assert!(
                common::req2_close(f64::from(yv[2 * i]), f64::from(re_v[i])),
                "{ctx} re[{i}]"
            );
            assert!(
                common::req2_close(f64::from(yv[2 * i + 1]), f64::from(im_v[i])),
                "{ctx} im[{i}]"
            );
        }
        assert_close_all(
            &dx.host_slice(),
            &dx2.host_slice(),
            &format!("rfft oracle grad n={n} {norm:?}"),
        );
    }
}

#[test]
fn irfft_matches_dft_matrix_oracle() {
    for (b, n, norm) in [
        (2usize, 8usize, FftNorm::Backward),
        (2, 7, FftNorm::Ortho),
        (1, 6, FftNorm::Forward),
    ] {
        let m = n / 2 + 1;
        let scale = match norm {
            FftNorm::Backward => 1.0 / n as f64,
            FftNorm::Ortho => 1.0 / (n as f64).sqrt(),
            _ => 1.0,
        };
        let weight = |k: usize| -> f64 {
            if k == 0 || (n % 2 == 0 && k == n / 2) {
                1.0
            } else {
                2.0
            }
        };
        let xs = seeded(b * m * 2, 21);
        let gs = seeded(b * n, 22);

        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(xs.clone(), &[b, m, 2]));
        let y = irfft(&x, Some(n), None, norm).unwrap();
        let gv = tape.var_no_grad(&t(gs.clone(), &[b, n]));
        let loss = y.mul(&gv).unwrap().sum(None).unwrap();
        let dx = tape
            .backward(&loss)
            .unwrap()
            .get(&x)
            .unwrap()
            .unwrap()
            .clone();

        // オラクル: re・im を別 Var にして [m, n] 行列で合成する。
        let mut re = vec![0.0f32; b * m];
        let mut im = vec![0.0f32; b * m];
        for i in 0..b * m {
            re[i] = xs[2 * i];
            im[i] = xs[2 * i + 1];
        }
        let tape2 = Tape::new_with_ops(common::naive_ops());
        let re_v = tape2.var(&t(re, &[b, m]));
        let im_v = tape2.var(&t(im, &[b, m]));
        let mut cm = Vec::with_capacity(m * n);
        let mut sm = Vec::with_capacity(m * n);
        for k in 0..m {
            for j in 0..n {
                let th = 2.0 * std::f64::consts::PI * ((k * j) % n) as f64 / n as f64;
                cm.push((scale * weight(k) * th.cos()) as f32);
                sm.push((-scale * weight(k) * th.sin()) as f32);
            }
        }
        let out2 = re_v
            .matmul(&tape2.var_no_grad(&t(cm, &[m, n])))
            .unwrap()
            .add(&im_v.matmul(&tape2.var_no_grad(&t(sm, &[m, n]))).unwrap())
            .unwrap();
        let loss2 = out2
            .mul(&tape2.var_no_grad(&t(gs, &[b, n])))
            .unwrap()
            .sum(None)
            .unwrap();
        let grads2 = tape2.backward(&loss2).unwrap();
        let dre = grads2
            .get(&re_v)
            .unwrap()
            .unwrap()
            .host_slice()
            .into_owned();
        let dim = grads2
            .get(&im_v)
            .unwrap()
            .unwrap()
            .host_slice()
            .into_owned();

        assert_close_all(
            &y.to_tensor().host_slice(),
            &out2.to_tensor().host_slice(),
            &format!("irfft oracle n={n} {norm:?}"),
        );
        let dxv = dx.host_slice().into_owned();
        let mut inter = vec![0.0f32; b * m * 2];
        for i in 0..b * m {
            inter[2 * i] = dre[i];
            inter[2 * i + 1] = dim[i];
        }
        assert_close_all(&dxv, &inter, &format!("irfft oracle grad n={n} {norm:?}"));
    }
}

// --- 中心差分 ---

fn loss_of(
    op: &str,
    data: &[f32],
    shape: &[usize],
    n: Option<usize>,
    norm: FftNorm,
    g: &[f32],
) -> f64 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(data.to_vec(), shape));
    let y = match op {
        "rfft" => rfft(&x, n, None, norm),
        _ => irfft(&x, n, None, norm),
    }
    .unwrap();
    y.to_tensor()
        .host_slice()
        .iter()
        .zip(g)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum()
}

#[test]
fn gradients_match_central_differences() {
    let eps = 0.05f32;
    let cases: [(&str, Vec<usize>, Option<usize>, FftNorm); 4] = [
        ("rfft", vec![6], None, FftNorm::Ortho),
        ("rfft", vec![5], Some(8), FftNorm::Backward),
        ("irfft", vec![4, 2], Some(6), FftNorm::Forward),
        ("irfft", vec![3, 2], Some(5), FftNorm::Backward),
    ];
    for (op, shape, n, norm) in cases {
        let numel: usize = shape.iter().product();
        let data = seeded(numel, 31);
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(data.clone(), &shape));
        let y = match op {
            "rfft" => rfft(&x, n, None, norm),
            _ => irfft(&x, n, None, norm),
        }
        .unwrap();
        let out_len = y.to_tensor().host_slice().len();
        let g = seeded(out_len, 32);
        let gv = tape.var_no_grad(&t(g.clone(), y.to_tensor().shape()));
        let loss = y.mul(&gv).unwrap().sum(None).unwrap();
        let dx = tape
            .backward(&loss)
            .unwrap()
            .get(&x)
            .unwrap()
            .unwrap()
            .clone();
        let dx = dx.host_slice().into_owned();
        for i in 0..numel {
            let mut hi = data.clone();
            let mut lo = data.clone();
            hi[i] += eps;
            lo[i] -= eps;
            let fd = (loss_of(op, &hi, &shape, n, norm, &g)
                - loss_of(op, &lo, &shape, n, norm, &g))
                / (2.0 * f64::from(eps));
            assert!(
                (fd - f64::from(dx[i])).abs() < 2e-3,
                "{op} {shape:?} n={n:?} {norm:?} [{i}]: fd={fd} analytic={}",
                dx[i]
            );
        }
    }
}

// --- 往復・構造的性質・決定性 ---

#[test]
fn roundtrip_rfft_irfft_for_all_norms() {
    for n in [1usize, 2, 3, 4, 7, 8] {
        for norm in [FftNorm::Backward, FftNorm::Ortho, FftNorm::Forward] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let data = seeded(2 * n, 41);
            let x = tape.var(&t(data.clone(), &[2, n]));
            let y = rfft(&x, None, None, norm).unwrap();
            let z = irfft(&y, Some(n), Some(1), norm).unwrap();
            assert_eq!(z.to_tensor().shape(), &[2, n]);
            assert_close_all(
                &z.to_tensor().host_slice(),
                &data,
                &format!("roundtrip n={n} {norm:?}"),
            );
        }
    }
}

#[test]
fn dc_and_nyquist_imag_are_positive_zero_bits() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(seeded(8, 51), &[8]));
    let y = rfft(&x, None, None, FftNorm::Backward).unwrap();
    let v = y.to_tensor().host_slice().into_owned();
    assert_eq!(v[1].to_bits(), 0, "DC 虚部");
    assert_eq!(v[2 * 4 + 1].to_bits(), 0, "Nyquist 虚部");

    // irfft の勾配側（DC・Nyquist 虚部は入力として読まれない）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let xr = tape.var(&t(seeded(10, 52), &[5, 2]));
    let out = irfft(&xr, None, None, FftNorm::Ortho).unwrap();
    let gv = tape.var_no_grad(&t(seeded(8, 53), &[8]));
    let loss = out.mul(&gv).unwrap().sum(None).unwrap();
    let dx = tape
        .backward(&loss)
        .unwrap()
        .get(&xr)
        .unwrap()
        .unwrap()
        .clone();
    let d = dx.host_slice().into_owned();
    assert_eq!(d[1].to_bits(), 0);
    assert_eq!(d[2 * 4 + 1].to_bits(), 0);
}

#[test]
fn same_input_twice_is_bit_identical() {
    let run = || {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(seeded(21, 61), &[3, 7]));
        rfft(&x, None, None, FftNorm::Ortho)
            .unwrap()
            .to_tensor()
            .host_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}

#[test]
fn nonfinite_input_propagates_without_rejection() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![f32::NAN, 1.0, 2.0, 3.0], &[4]));
    let y = rfft(&x, None, None, FftNorm::Backward).unwrap();
    let v = y.to_tensor().host_slice().into_owned();
    assert!(v.iter().any(|a| a.is_nan()), "NaN が伝播する");
    // DC 虚部はリテラル 0 のため非有限入力でも NaN にならない。
    assert_eq!(v[1].to_bits(), 0);
}

// --- 境界エラー ---

#[test]
fn boundary_errors_are_typed() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let scalar = tape.var(&t(vec![1.0], &[]));
    assert!(matches!(
        rfft(&scalar, None, None, FftNorm::Backward),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    let v4 = tape.var(&t(vec![1.0; 4], &[4]));
    assert!(matches!(
        rfft(&v4, Some(0), None, FftNorm::Backward),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        rfft(&v4, None, Some(1), FftNorm::Backward),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        irfft(&v4, None, None, FftNorm::Backward),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    let bad_last = tape.var(&t(vec![1.0; 6], &[2, 3]));
    assert!(matches!(
        irfft(&bad_last, None, None, FftNorm::Backward),
        Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
    ));
    let one_bin = tape.var(&t(vec![1.0, 0.0], &[1, 2]));
    assert!(matches!(
        irfft(&one_bin, None, None, FftNorm::Backward),
        Err(AutodiffError::InvalidArgument(_))
    ));
    let three_bins = tape.var(&t(vec![1.0; 6], &[3, 2]));
    assert!(matches!(
        irfft(&three_bins, Some(0), None, FftNorm::Backward),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        irfft(&three_bins, None, Some(1), FftNorm::Backward),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn huge_n_is_rejected_before_allocation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let v4 = tape.var(&t(vec![1.0; 4], &[4]));
    assert!(matches!(
        rfft(&v4, Some(usize::MAX / 2), None, FftNorm::Backward),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
    let c = tape.var(&t(vec![1.0; 6], &[3, 2]));
    assert!(matches!(
        irfft(&c, Some(usize::MAX), None, FftNorm::Backward),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
}

// --- Unsupported 以外のバックエンドエラーはフォールバックしない ---

/// `fft_rfft`／`fft_irfft` が指定のエラーを返すフィクスチャ。
struct FftErrOps {
    inner: Box<dyn BackendOps + Send>,
    error: fn() -> BackendError,
    wrong_shape: bool,
}

impl BackendOps for FftErrOps {
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
    fn fft_rfft(
        &self,
        _input: &Tensor<f32>,
        _n: usize,
        _dim: usize,
        _norm: FftNorm,
    ) -> Result<Tensor<f32>, BackendError> {
        if self.wrong_shape {
            return Ok(Tensor::new(vec![0.0; 3], &[3]).expect("shape"));
        }
        Err((self.error)())
    }
    fn fft_irfft(
        &self,
        _input: &Tensor<f32>,
        _n: usize,
        _dim: usize,
        _norm: FftNorm,
    ) -> Result<Tensor<f32>, BackendError> {
        if self.wrong_shape {
            return Ok(Tensor::new(vec![0.0; 3], &[3]).expect("shape"));
        }
        Err((self.error)())
    }
}

fn err_tape(error: fn() -> BackendError, wrong_shape: bool) -> Tape {
    Tape::new_with_ops(Box::new(FftErrOps {
        inner: common::naive_ops(),
        error,
        wrong_shape,
    }))
}

#[test]
fn non_unsupported_backend_error_is_propagated_not_swallowed() {
    let tape = err_tape(
        || BackendError::KernelLaunchFailed("simulated".into()),
        false,
    );
    let x = tape.var(&t(vec![1.0; 4], &[4]));
    assert!(matches!(
        rfft(&x, None, None, FftNorm::Backward),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    let c = tape.var(&t(vec![1.0; 6], &[3, 2]));
    assert!(matches!(
        irfft(&c, None, None, FftNorm::Backward),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    // InvalidArgument は AutodiffError::InvalidArgument へ写像される（握りつぶさない）。
    let tape = err_tape(|| BackendError::InvalidArgument("bad".into()), false);
    let x = tape.var(&t(vec![1.0; 4], &[4]));
    assert!(matches!(
        rfft(&x, None, None, FftNorm::Backward),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn unsupported_falls_back_to_host_kernel() {
    let tape = err_tape(|| BackendError::Unsupported("none".into()), false);
    let x = tape.var(&t(vec![1.0, 0.0, 0.0, 0.0], &[4]));
    let y = rfft(&x, None, None, FftNorm::Backward).unwrap();
    assert_eq!(
        y.to_tensor().host_slice().into_owned(),
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    );
}

#[test]
fn wrong_shape_from_backend_is_rejected() {
    let tape = err_tape(|| BackendError::Unsupported("unused".into()), true);
    let x = tape.var(&t(vec![1.0; 4], &[4]));
    assert!(matches!(
        rfft(&x, None, None, FftNorm::Backward),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
}
