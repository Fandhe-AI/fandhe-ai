//! `fft_ops::{stft, istft}`（イシュー #2633・親 #2630）の `Tape`/`Var` を経由する
//! end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/fft-pytorch-reference/stft_reference.json`・生成条件は
//!   同ディレクトリの `README.md`）の forward 出力と入力勾配を REQ-2 統一
//!   複合判定（`common::req2_close`）で突合する。tolerance 定数は新設しない。
//! - fixture の `error_cases`（torch が例外を出すか否かの実測）と Rust 側の
//!   拒否有無を突合する（差分は明示の許可リストに限る）。
//! - 独立実装の `f64` 直接 DFT オラクル（フレーム化・端パディング・重畳加算を
//!   別の書き方で再実装）との forward 突合、中心差分による勾配検査、
//!   `stft → istft` 往復、bit 決定性、境界エラー、確保前拒否、非有限入力の伝播、
//!   バックエンド契約（`Unsupported` のみフォールバック・誤 shape 拒否・
//!   NOLA 違反の拒否）を固定する。
//!
//! `common::naive_ops()` は `fft_stft`／`fft_istft` をオーバーライドしないため、
//! 必ず共有ホストカーネル（`fandhe_ai_tensor_core::fft`）へのフォールバック経路を
//! 通る。CPU `BackendOps` 実装との一致は
//! `crates/facade/tests/fft_ops_backend_parity.rs` が担当する。

mod common;

use std::path::PathBuf;

use fandhe_ai_autodiff::fft_ops::{IstftOptions, StftOptions, istft, stft};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::fft::{IstftParams, StftParams};
use fandhe_ai_tensor_core::{BackendOps, ShapeError, StftPadMode, Tensor};
use serde::Deserialize;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    cases: Vec<Case>,
    error_cases: Vec<Case>,
}

/// `cases`・`error_cases` 共通の 1 件（`error_cases` では入出力・勾配が無く、
/// `torch_raises` を持つ）。
#[derive(Deserialize)]
struct Case {
    name: String,
    op: String,
    in_shape: Vec<usize>,
    n_fft: usize,
    hop: Option<usize>,
    win_length: Option<usize>,
    window: Option<Vec<f32>>,
    center: bool,
    pad_mode: String,
    normalized: bool,
    onesided: Option<bool>,
    length: Option<usize>,
    #[serde(default)]
    input: Vec<f32>,
    #[serde(default)]
    grad_out: Vec<f32>,
    #[serde(default)]
    out_shape: Vec<usize>,
    #[serde(default)]
    output: Vec<f32>,
    #[serde(default)]
    grad_in: Vec<f32>,
    #[serde(default)]
    torch_raises: bool,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fft-pytorch-reference/stft_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

fn pad_mode_of(s: &str) -> StftPadMode {
    match s {
        "reflect" => StftPadMode::Reflect,
        "constant" => StftPadMode::Constant,
        other => panic!("未知の pad_mode: {other}"),
    }
}

fn stft_options(c: &Case) -> StftOptions {
    let mut o = StftOptions::default()
        .with_center(c.center)
        .with_pad_mode(pad_mode_of(&c.pad_mode))
        .with_normalized(c.normalized)
        .with_onesided(c.onesided.expect("stft の onesided は bool"));
    if let Some(h) = c.hop {
        o = o.with_hop_length(h);
    }
    if let Some(w) = c.win_length {
        o = o.with_win_length(w);
    }
    o
}

fn istft_options(c: &Case) -> IstftOptions {
    let mut o = IstftOptions::default()
        .with_center(c.center)
        .with_normalized(c.normalized);
    if let Some(h) = c.hop {
        o = o.with_hop_length(h);
    }
    if let Some(w) = c.win_length {
        o = o.with_win_length(w);
    }
    if let Some(s) = c.onesided {
        o = o.with_onesided(s);
    }
    if let Some(l) = c.length {
        o = o.with_length(l);
    }
    o
}

fn call<'t>(x: &Var<'t>, c: &Case) -> Result<Var<'t>, AutodiffError> {
    let window = c.window.as_ref().map(|w| t(w.clone(), &[w.len()]));
    match c.op.as_str() {
        "stft" => stft(x, c.n_fft, window.as_ref(), &stft_options(c)),
        "istft" => istft(x, c.n_fft, window.as_ref(), &istft_options(c)),
        other => panic!("未知の op: {other}"),
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

fn run_case(case: &Case) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(case.input.clone(), &case.in_shape));
    let y = call(&x, case).unwrap_or_else(|e| panic!("{}: forward が失敗: {e}", case.name));
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
    for op in ["stft", "istft"] {
        assert!(
            fixture.cases.iter().any(|c| c.op == op),
            "fixture に {op} のケースが無い"
        );
    }
    for case in &fixture.cases {
        let (out, out_shape, dx) = run_case(case);
        assert_eq!(out_shape, case.out_shape, "{}: 出力 shape", case.name);
        assert_close_all(&out, &case.output, &format!("{} forward", case.name));
        assert_eq!(dx.len(), case.grad_in.len(), "{}: 勾配長", case.name);
        assert_close_all(&dx, &case.grad_in, &format!("{} grad", case.name));
    }
}

/// torch が例外を出すが本実装は受理する（または逆の）ケースの明示的な許可リスト。
/// 差分の根拠は `docs/autodiff-fft-ops-decision.md` §13。
const RUST_ACCEPTS_WHERE_TORCH_RAISES: [&str; 1] = [
    // torch は `n_fft = 1`・`center = true`・`length` 省略で終端 `-(n_fft/2) = -0` を
    // 0 と解釈して空出力を作り、その `min()` で例外になる（実装の取りこぼし）。
    // 本実装は意味論どおり長さ L の出力を返す。
    "istft_n_fft1_center_no_length",
];
const RUST_REJECTS_WHERE_TORCH_ACCEPTS: [&str; 0] = [];

#[test]
fn error_cases_follow_pytorch_rejections() {
    let fixture = load_fixture();
    for name in RUST_ACCEPTS_WHERE_TORCH_RAISES {
        assert!(
            fixture
                .error_cases
                .iter()
                .any(|c| c.name == name && c.torch_raises),
            "許可リストの {name} が fixture に無い、または torch が例外を出さない（リストが陳腐化）"
        );
    }
    for case in &fixture.error_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let numel: usize = case.in_shape.iter().product();
        let x = tape.var(&t(vec![0.5; numel], &case.in_shape));
        let rust_rejects = call(&x, case).is_err();
        let expected = if RUST_ACCEPTS_WHERE_TORCH_RAISES.contains(&case.name.as_str()) {
            assert!(
                !rust_rejects,
                "{}: 許可リスト上は受理のはずが拒否した",
                case.name
            );
            continue;
        } else if RUST_REJECTS_WHERE_TORCH_ACCEPTS.contains(&case.name.as_str()) {
            true
        } else {
            case.torch_raises
        };
        assert_eq!(
            rust_rejects, expected,
            "{}: torch の拒否有無と一致しない",
            case.name
        );
    }
}

// --- 独立実装の f64 オラクル ---

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

fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()) as f32)
        .collect()
}

/// 長さ `win.len()` の窓を中央寄せで `n` へゼロ詰めする（独立実装）。
fn centered(win: &[f32], n: usize) -> Vec<f64> {
    let left = (n - win.len()) / 2;
    let mut out = vec![0.0; n];
    for (i, &w) in win.iter().enumerate() {
        out[left + i] = f64::from(w);
    }
    out
}

fn pad_signal(x: &[f32], pad: usize, mode: StftPadMode) -> Vec<f64> {
    let l = x.len();
    let mut out = Vec::new();
    for i in 0..pad {
        out.push(match mode {
            StftPadMode::Reflect => f64::from(x[pad - i]),
            _ => 0.0,
        });
    }
    out.extend(x.iter().map(|&v| f64::from(v)));
    for i in 0..pad {
        out.push(match mode {
            StftPadMode::Reflect => f64::from(x[l - 2 - i]),
            _ => 0.0,
        });
    }
    out
}

#[derive(Clone, Copy)]
struct Cfg {
    n: usize,
    hop: usize,
    wl: usize,
    center: bool,
    mode: StftPadMode,
    normalized: bool,
    onesided: bool,
}

/// 1 バッチ分の STFT オラクル。戻り値は `[N][T]` を `k * T + t` で並べた `(re, im)`。
fn stft_oracle(sig: &[f32], win: &[f32], c: Cfg) -> (usize, Vec<(f64, f64)>) {
    let pad = if c.center { c.n / 2 } else { 0 };
    let padded = pad_signal(sig, pad, c.mode);
    let frames = 1 + (padded.len() - c.n) / c.hop;
    let w = centered(win, c.n);
    let scale = if c.normalized {
        1.0 / (c.n as f64).sqrt()
    } else {
        1.0
    };
    let bins = if c.onesided { c.n / 2 + 1 } else { c.n };
    let mut out = vec![(0.0, 0.0); bins * frames];
    for tt in 0..frames {
        for k in 0..bins {
            let (mut re, mut im) = (0.0, 0.0);
            for j in 0..c.n {
                let v = padded[tt * c.hop + j] * w[j];
                let th = 2.0 * std::f64::consts::PI * ((k * j) % c.n) as f64 / c.n as f64;
                re += v * th.cos();
                im -= v * th.sin();
            }
            out[k * frames + tt] = (scale * re, scale * im);
        }
    }
    (frames, out)
}

/// 1 バッチ分の ISTFT オラクル。`spec` は `[N_in][T]`（`k * T + t`）。先頭
/// `n/2+1` bin だけを読み、DC・Nyquist の虚部は無視する。
fn istft_oracle(
    spec: &[(f64, f64)],
    frames: usize,
    win: &[f32],
    c: Cfg,
    length: Option<usize>,
) -> Vec<f64> {
    let n = c.n;
    let w = centered(win, n);
    let scale = if c.normalized {
        1.0 / (n as f64).sqrt()
    } else {
        1.0 / n as f64
    };
    let expected = n + c.hop * (frames - 1);
    let mut y = vec![0.0; expected];
    let mut env = vec![0.0; expected];
    for tt in 0..frames {
        // 共役対称へ延長した全 bin スペクトル。
        let full: Vec<(f64, f64)> = (0..n)
            .map(|k| {
                if k <= n / 2 {
                    let (re, mut im) = spec[k * frames + tt];
                    if k == 0 || (n.is_multiple_of(2) && k == n / 2) {
                        im = 0.0;
                    }
                    (re, im)
                } else {
                    let (re, im) = spec[(n - k) * frames + tt];
                    (re, -im)
                }
            })
            .collect();
        for j in 0..n {
            let mut acc = 0.0;
            for (k, &(re, im)) in full.iter().enumerate() {
                let th = 2.0 * std::f64::consts::PI * ((k * j) % n) as f64 / n as f64;
                acc += re * th.cos() - im * th.sin();
            }
            y[tt * c.hop + j] += scale * acc * w[j];
            env[tt * c.hop + j] += w[j] * w[j];
        }
    }
    let start = if c.center { n / 2 } else { 0 };
    let end = length.map_or(if c.center { expected - n / 2 } else { expected }, |l| {
        start + l
    });
    let mut out = Vec::new();
    for pos in start..end {
        out.push(if pos < expected {
            y[pos] / env[pos]
        } else {
            0.0
        });
    }
    out
}

fn close_f64(actual: &[f32], expected: &[f64], ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: 長さ");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            common::req2_close(f64::from(a), e),
            "{ctx}[{i}]: actual={a} expected={e}"
        );
    }
}

fn opts_of(c: Cfg) -> StftOptions {
    StftOptions::default()
        .with_hop_length(c.hop)
        .with_win_length(c.wl)
        .with_center(c.center)
        .with_pad_mode(c.mode)
        .with_normalized(c.normalized)
        .with_onesided(c.onesided)
}

fn configs() -> Vec<Cfg> {
    let mut v = Vec::new();
    for (n, hop, wl) in [
        (8usize, 2usize, 8usize),
        (7, 3, 5),
        (6, 6, 6),
        (5, 1, 5),
        (8, 3, 4),
    ] {
        for center in [true, false] {
            for mode in [StftPadMode::Reflect, StftPadMode::Constant] {
                for normalized in [false, true] {
                    for onesided in [true, false] {
                        v.push(Cfg {
                            n,
                            hop,
                            wl,
                            center,
                            mode,
                            normalized,
                            onesided,
                        });
                    }
                }
            }
        }
    }
    v
}

#[test]
fn stft_matches_f64_dft_oracle() {
    for c in configs() {
        let win = hann(c.wl);
        let (b, l) = (2usize, 29usize);
        let xs = seeded(b * l, 41);
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(xs.clone(), &[b, l]));
        let w = t(win.clone(), &[c.wl]);
        let y = stft(&x, c.n, Some(&w), &opts_of(c)).unwrap();
        let out = y.to_tensor();
        let flat = out.host_slice().into_owned();
        let (frames, _) = stft_oracle(&xs[..l], &win, c);
        let bins = if c.onesided { c.n / 2 + 1 } else { c.n };
        assert_eq!(out.shape(), &[b, bins, frames, 2]);
        for bi in 0..b {
            let (_, want) = stft_oracle(&xs[bi * l..(bi + 1) * l], &win, c);
            let got: Vec<f32> = flat[bi * bins * frames * 2..(bi + 1) * bins * frames * 2].to_vec();
            let want_flat: Vec<f64> = want.iter().flat_map(|&(re, im)| [re, im]).collect();
            close_f64(&got, &want_flat, "stft oracle");
        }
    }
}

#[test]
fn istft_matches_f64_dft_oracle() {
    for c in configs() {
        // 正の窓（center = false でも NOLA を満たす）。
        let win: Vec<f32> = seeded(c.wl, 5).iter().map(|v| 0.6 + 0.4 * v).collect();
        let frames = 6usize;
        // 入力 bin 数は onesided に従う。
        let bins = if c.onesided { c.n / 2 + 1 } else { c.n };
        let (b, spec_len) = (2usize, bins * frames * 2);
        let xs = seeded(b * spec_len, 53);
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(xs.clone(), &[b, bins, frames, 2]));
        let w = t(win.clone(), &[c.wl]);
        for length in [None, Some(11usize)] {
            let mut o = IstftOptions::default()
                .with_hop_length(c.hop)
                .with_win_length(c.wl)
                .with_center(c.center)
                .with_normalized(c.normalized)
                .with_onesided(c.onesided);
            if let Some(l) = length {
                o = o.with_length(l);
            }
            let y = match istft(&x, c.n, Some(&w), &o) {
                Ok(y) => y,
                Err(e) => {
                    // 中央寄せ窓の両端 0 により center = false で NOLA 違反になる構成は拒否される。
                    assert!(
                        matches!(e, AutodiffError::InvalidArgument(_)) && c.wl < c.n,
                        "予期しない拒否: {e}"
                    );
                    continue;
                }
            };
            let out = y.to_tensor();
            let flat = out.host_slice().into_owned();
            let l_out = out.shape()[1];
            for bi in 0..b {
                let spec: Vec<(f64, f64)> = (0..bins * frames)
                    .map(|i| {
                        let o = bi * spec_len + i * 2;
                        (f64::from(xs[o]), f64::from(xs[o + 1]))
                    })
                    .collect();
                let want = istft_oracle(&spec, frames, &win, c, length);
                close_f64(&flat[bi * l_out..(bi + 1) * l_out], &want, "istft oracle");
            }
        }
    }
}

// --- 中心差分（forward は線形のため刻みを大きく取れる） ---

fn loss_of(c: &Case, input: &[f32], g: &[f32]) -> f64 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var_no_grad(&t(input.to_vec(), &c.in_shape));
    let y = call(&x, c).unwrap().to_tensor();
    y.host_slice()
        .iter()
        .zip(g)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum()
}

#[test]
fn gradients_match_central_differences() {
    let fixture = load_fixture();
    // 小さめの代表ケースのみ（全入力要素に対して forward を 2 回走らせる）。
    for name in [
        "stft_hann",
        "stft_onesided_false_odd",
        "stft_pad_constant",
        "stft_batched_center_false",
        "istft_hann_hop_quarter",
        "istft_onesided_false_odd",
        "istft_batched",
        "istft_length_long",
    ] {
        let case = fixture
            .cases
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("fixture に {name} が無い"));
        let (_, _, dx) = run_case(case);
        let eps = 0.25f32;
        for i in 0..case.input.len() {
            let (mut p, mut m) = (case.input.clone(), case.input.clone());
            p[i] += eps;
            m[i] -= eps;
            let fd = (loss_of(case, &p, &case.grad_out) - loss_of(case, &m, &case.grad_out))
                / (2.0 * f64::from(eps));
            assert!(
                common::req2_close(f64::from(dx[i]), fd),
                "{name} grad[{i}]: autodiff={} fd={fd}",
                dx[i]
            );
        }
    }
}

// --- 往復・決定性 ---

#[test]
fn stft_istft_roundtrip_recovers_signal() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let sig = seeded(2 * 48, 9);
    let x = tape.var(&t(sig.clone(), &[2, 48]));
    let w = t(hann(16), &[16]);
    let s = stft(&x, 16, Some(&w), &StftOptions::default()).unwrap();
    let y = istft(&s, 16, Some(&w), &IstftOptions::default().with_length(48)).unwrap();
    let out = y.to_tensor();
    assert_eq!(out.shape(), &[2, 48]);
    for (a, b) in out.host_slice().iter().zip(&sig) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

#[test]
fn run_to_run_is_bit_deterministic() {
    let run = || {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(seeded(40, 3), &[40]));
        let w = t(hann(8), &[8]);
        let s = stft(
            &x,
            8,
            Some(&w),
            &StftOptions::default().with_normalized(true),
        )
        .unwrap();
        let y = istft(
            &s,
            8,
            Some(&w),
            &IstftOptions::default().with_normalized(true),
        )
        .unwrap();
        let loss = y.sum(None).unwrap();
        let g = tape.backward(&loss).unwrap();
        let dx = g.get(&x).unwrap().unwrap().clone();
        (
            s.to_tensor().host_slice().into_owned(),
            y.to_tensor().host_slice().into_owned(),
            dx.host_slice().into_owned(),
        )
    };
    let (a, b) = (run(), run());
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.0), bits(&b.0));
    assert_eq!(bits(&a.1), bits(&b.1));
    assert_eq!(bits(&a.2), bits(&b.2));
}

#[test]
fn dc_and_nyquist_imag_are_positive_zero_for_onesided() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(seeded(40, 4), &[40]));
    let y = stft(&x, 8, None, &StftOptions::default())
        .unwrap()
        .to_tensor();
    let frames = y.shape()[1];
    let d = y.host_slice().into_owned();
    for tt in 0..frames {
        assert_eq!(d[tt * 2 + 1].to_bits(), 0);
        assert_eq!(d[(4 * frames + tt) * 2 + 1].to_bits(), 0);
    }
}

// --- 境界エラー・確保前拒否・非有限入力 ---

#[test]
fn invalid_arguments_are_typed_errors() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0; 32], &[32]));
    let d = StftOptions::default();
    assert!(matches!(
        stft(&x, 0, None, &d),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        stft(&x, 8, None, &d.with_hop_length(0)),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        stft(&x, 3, None, &d),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        stft(&x, 8, None, &d.with_win_length(9)),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // 窓の長さ不一致・rank 違い。
    assert!(matches!(
        stft(&x, 8, Some(&t(vec![1.0; 7], &[7])), &d),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        stft(&x, 8, Some(&t(vec![1.0; 8], &[2, 4])), &d),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    // 信号 rank。
    let x3 = tape.var(&t(vec![1.0; 32], &[1, 2, 16]));
    assert!(matches!(
        stft(&x3, 8, None, &d),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    // 反射パディング幅。
    let short = tape.var(&t(vec![1.0; 4], &[4]));
    assert!(matches!(
        stft(&short, 8, None, &d),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // istft: 末尾次元・bin 数・hop>win・length=0・NOLA。
    let c = tape.var(&t(vec![1.0; 5 * 17 * 2], &[5, 17, 2]));
    let bad_last = tape.var(&t(vec![1.0; 5 * 17 * 3], &[5, 17, 3]));
    let di = IstftOptions::default();
    assert!(matches!(
        istft(&bad_last, 8, None, &di),
        Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
    ));
    assert!(matches!(
        istft(&c, 16, None, &di),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        istft(&c, 8, None, &di.with_hop_length(9)),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        istft(&c, 8, None, &di.with_length(0)),
        Err(AutodiffError::InvalidArgument(_))
    ));
    let h = t(hann(8), &[8]);
    assert!(matches!(
        istft(&c, 8, Some(&h), &di.with_center(false)),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // center = true なら端が切り落とされ受理される。
    assert!(istft(&c, 8, Some(&h), &di).is_ok());
}

#[test]
fn nola_threshold_boundary_matches_pytorch_pair() {
    // PyTorch 2.14.0 の実測（fixture error_cases）: 窓先頭値 3.1e-6（包絡 9.61e-12）は
    // 拒否・3.2e-6（包絡 1.024e-11）は受理。しきい値 1e-11 はこの区間にある。
    let tape = Tape::new_with_ops(common::naive_ops());
    let c = tape.var(&t(vec![0.5; 5 * 3 * 2], &[5, 3, 2]));
    let opts = IstftOptions::default()
        .with_hop_length(8)
        .with_center(false);
    let mk = |w0: f32| {
        let mut w = vec![1.0f32; 8];
        w[0] = w0;
        t(w, &[8])
    };
    assert!(istft(&c, 8, Some(&mk(3.1e-6)), &opts).is_err());
    assert!(istft(&c, 8, Some(&mk(3.2e-6)), &opts).is_ok());
}

#[test]
fn huge_arguments_are_rejected_before_allocation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0; 4], &[4]));
    let o = StftOptions::default().with_center(false);
    assert!(matches!(
        stft(&x, usize::MAX / 2, None, &o),
        Err(AutodiffError::InvalidArgument(_)) | Err(AutodiffError::Shape(_))
    ));
    let big = StftOptions::default().with_pad_mode(StftPadMode::Constant);
    assert!(stft(&x, usize::MAX, None, &big).is_err());
    let c = tape.var(&t(vec![1.0; 6], &[3, 1, 2]));
    let oi = IstftOptions::default()
        .with_hop_length(1)
        .with_length(usize::MAX);
    assert!(matches!(
        istft(&c, 4, None, &oi),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
    let huge_n = IstftOptions::default().with_hop_length(1);
    assert!(istft(&c, usize::MAX / 2, None, &huge_n).is_err());
}

#[test]
fn non_finite_input_propagates_without_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let mut xs = seeded(32, 6);
    xs[10] = f32::NAN;
    xs[20] = f32::INFINITY;
    let x = tape.var(&t(xs, &[32]));
    let y = stft(&x, 8, None, &StftOptions::default())
        .unwrap()
        .to_tensor();
    assert!(y.host_slice().iter().any(|v| !v.is_finite()));
}

// --- バックエンド契約（モック） ---

struct StftMockOps {
    inner: Box<dyn BackendOps + Send>,
    error: fn() -> BackendError,
    wrong_shape: bool,
    /// `true` なら `fft_istft` が NOLA 違反でも `Ok` を返す（迂回の模擬）。
    ok_istft: bool,
}

impl BackendOps for StftMockOps {
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
        if self.wrong_shape {
            return Ok(Tensor::new(vec![0.0; 3], &[3]).expect("shape"));
        }
        Err((self.error)())
    }
    fn fft_istft(
        &self,
        _input: &Tensor<f32>,
        _window: &Tensor<f32>,
        _params: &IstftParams,
    ) -> Result<Tensor<f32>, BackendError> {
        if self.ok_istft {
            return Ok(Tensor::new(vec![0.0; 32], &[32]).expect("shape"));
        }
        if self.wrong_shape {
            return Ok(Tensor::new(vec![0.0; 3], &[3]).expect("shape"));
        }
        Err((self.error)())
    }
}

fn mock_tape(error: fn() -> BackendError, wrong_shape: bool, ok_istft: bool) -> Tape {
    Tape::new_with_ops(Box::new(StftMockOps {
        inner: common::naive_ops(),
        error,
        wrong_shape,
        ok_istft,
    }))
}

#[test]
fn non_unsupported_backend_error_is_propagated_not_swallowed() {
    let tape = mock_tape(
        || BackendError::KernelLaunchFailed("simulated".into()),
        false,
        false,
    );
    let x = tape.var(&t(vec![1.0; 32], &[32]));
    assert!(matches!(
        stft(&x, 8, None, &StftOptions::default()),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    let c = tape.var(&t(vec![1.0; 5 * 17 * 2], &[5, 17, 2]));
    assert!(matches!(
        istft(&c, 8, None, &IstftOptions::default()),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    let tape = mock_tape(|| BackendError::InvalidArgument("bad".into()), false, false);
    let x = tape.var(&t(vec![1.0; 32], &[32]));
    assert!(matches!(
        stft(&x, 8, None, &StftOptions::default()),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn unsupported_falls_back_to_host_kernel() {
    let tape = mock_tape(|| BackendError::Unsupported("none".into()), false, false);
    let xs = seeded(32, 8);
    let x = tape.var(&t(xs.clone(), &[32]));
    let y = stft(&x, 8, None, &StftOptions::default()).unwrap();
    let want = {
        let tp = Tape::new_with_ops(common::naive_ops());
        let xv = tp.var(&t(xs, &[32]));
        stft(&xv, 8, None, &StftOptions::default())
            .unwrap()
            .to_tensor()
            .host_slice()
            .into_owned()
    };
    assert_eq!(y.to_tensor().host_slice().into_owned(), want);
}

#[test]
fn wrong_shape_from_backend_is_rejected() {
    let tape = mock_tape(|| BackendError::Unsupported("unused".into()), true, false);
    let x = tape.var(&t(vec![1.0; 32], &[32]));
    assert!(matches!(
        stft(&x, 8, None, &StftOptions::default()),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
    let c = tape.var(&t(vec![1.0; 5 * 17 * 2], &[5, 17, 2]));
    assert!(matches!(
        istft(&c, 8, None, &IstftOptions::default()),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
}

#[test]
fn nola_violation_is_rejected_even_if_backend_would_accept() {
    let tape = mock_tape(|| BackendError::Unsupported("unused".into()), false, true);
    let c = tape.var(&t(vec![1.0; 5 * 17 * 2], &[5, 17, 2]));
    let h = t(hann(8), &[8]);
    let opts = IstftOptions::default().with_center(false);
    assert!(matches!(
        istft(&c, 8, Some(&h), &opts),
        Err(AutodiffError::InvalidArgument(_))
    ));
}
