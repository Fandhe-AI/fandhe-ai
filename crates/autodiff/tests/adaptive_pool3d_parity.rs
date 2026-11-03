//! `adaptive_pool3d_ops`（イシュー #3823・`adaptive_avg_pool3d`／`adaptive_max_pool3d`）の
//! `Tape`／`Var` を経由する end-to-end 統合テスト（設計 `docs/autodiff-pool3d-ops-decision.md`
//! §18・§21）。
//!
//! - 独立オラクル: 中心差分（f64）による勾配、手計算の重なり窓、素朴な 3 重ループ参照。
//! - `common::naive_ops()` は `adaptive_pool3d_*` を override しないため、共有ホストカーネルへの
//!   フォールバック経路を通る。CPU `BackendOps` との一致は
//!   `crates/facade/tests/adaptive_pool3d_ops_backend_parity.rs` が担当する。
//! - フォールバックは `Unsupported` のときだけで、他のバックエンドエラーは伝播する。戻り shape
//!   不正は型付きエラー。孤児ノード無し・空バッチ・`N=0` でも `D·H·W > i32::MAX` を拒否する。
//! - 実 PyTorch 2.14 の実行値 fixture（`tests/fixtures/adaptive-pool3d-pytorch-reference/`）と突合する（下部の節）。

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::adaptive_pool3d_ops::{adaptive_avg_pool3d, adaptive_max_pool3d};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn pseudo(n: usize, mul: usize, modulus: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * mul) % modulus) as f32 * 0.1 - 0.7)
        .collect()
}

/// 素朴な参照: 出力位置ごとの窓（入力 flat 添字列）。式は PyTorch と同じ floor/ceil を整数で書く。
fn ref_windows(in_shape: [usize; 5], out: [usize; 3]) -> Vec<Vec<usize>> {
    let [n, c, d, h, w] = in_shape;
    let win = |o: usize, i: usize, k: usize| (o * i / k, ((o + 1) * i).div_ceil(k));
    let mut wins = Vec::new();
    for nc in 0..n * c {
        for od in 0..out[0] {
            for oh in 0..out[1] {
                for ow in 0..out[2] {
                    let (ds, de) = win(od, d, out[0]);
                    let (hs, he) = win(oh, h, out[1]);
                    let (ws, we) = win(ow, w, out[2]);
                    let mut m = Vec::new();
                    for dd in ds..de {
                        for hh in hs..he {
                            for ww in ws..we {
                                m.push(nc * d * h * w + (dd * h + hh) * w + ww);
                            }
                        }
                    }
                    wins.push(m);
                }
            }
        }
    }
    wins
}

const CASES: [([usize; 5], [usize; 3]); 5] = [
    ([1, 2, 4, 4, 4], [2, 2, 2]),
    ([1, 2, 3, 5, 4], [2, 3, 2]),
    ([2, 1, 5, 3, 7], [3, 2, 4]),
    ([1, 1, 2, 3, 2], [4, 5, 3]),
    ([1, 3, 6, 6, 6], [1, 1, 1]),
];

#[test]
fn forward_matches_naive_reference_windows() {
    for (in_shape, out) in CASES {
        let numel: usize = in_shape.iter().product();
        let x = pseudo(numel, 7, 23);
        let wins = ref_windows(in_shape, out);
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.clone(), &in_shape));
        let a = adaptive_avg_pool3d(&xv, out).unwrap();
        let (m, idx) = adaptive_max_pool3d(&xv, out).unwrap();
        let a = a.to_tensor().host_slice().into_owned();
        let m = m.to_tensor().host_slice().into_owned();
        let idx = idx.contiguous().host_slice().into_owned();
        let plane = in_shape[2] * in_shape[3] * in_shape[4];
        assert_eq!(a.len(), wins.len());
        for (k, win) in wins.iter().enumerate() {
            let mean = win.iter().map(|&i| f64::from(x[i])).sum::<f64>() / win.len() as f64;
            assert!(
                common::req2_close(f64::from(a[k]), mean),
                "{in_shape:?}->{out:?} avg[{k}]"
            );
            // Max: 先勝ちの最大。値は入力要素と bit 一致、索引は平面内 flat 添字。
            let mut best = win[0];
            for &i in win {
                if x[i] > x[best] {
                    best = i;
                }
            }
            assert_eq!(m[k].to_bits(), x[best].to_bits(), "max[{k}]");
            assert_eq!(idx[k] as usize, best % plane, "idx[{k}]");
        }
    }
}

#[test]
fn avg_gradient_matches_central_difference() {
    for (in_shape, out) in CASES {
        let numel: usize = in_shape.iter().product();
        let x = pseudo(numel, 7, 23);
        let wins = ref_windows(in_shape, out);
        let g = pseudo(wins.len(), 5, 19);
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.clone(), &in_shape));
        let y = adaptive_avg_pool3d(&xv, out).unwrap();
        let gv = tape.var_no_grad(&t(g.clone(), y.to_tensor().shape()));
        let loss = y.mul(&gv).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&xv).unwrap().unwrap().host_slice().into_owned();
        let f = |xs: &[f64]| -> f64 {
            wins.iter()
                .zip(&g)
                .map(|(w, &gi)| {
                    w.iter().map(|&i| xs[i]).sum::<f64>() / w.len() as f64 * f64::from(gi)
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
                "{in_shape:?}->{out:?} x[{j}]: {} vs {fd}",
                dx[j]
            );
        }
    }
}

#[test]
fn max_gradient_matches_central_difference_without_ties() {
    // 全要素が異なる値になるよう素数 53 で並べ替える（numel < 211 の形状のみ対象）。窓が重なる構成を含む。
    let mut checked = 0;
    for (in_shape, out) in CASES {
        let numel: usize = in_shape.iter().product();
        if numel >= 211 {
            continue;
        }
        checked += 1;
        let x: Vec<f32> = (0..numel).map(|i| ((i * 53) % 211) as f32 * 0.3).collect();
        let wins = ref_windows(in_shape, out);
        let g = pseudo(wins.len(), 5, 19);
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.clone(), &in_shape));
        let (y, _) = adaptive_max_pool3d(&xv, out).unwrap();
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
        let h = 1e-2;
        for j in 0..base.len() {
            let (mut hi, mut lo) = (base.clone(), base.clone());
            hi[j] += h;
            lo[j] -= h;
            let fd = (f(&hi) - f(&lo)) / (2.0 * h);
            assert!(
                common::req2_close(f64::from(dx[j]), fd),
                "{in_shape:?}->{out:?} x[{j}]: {} vs {fd}",
                dx[j]
            );
        }
    }
    assert!(checked >= 3, "中心差分の対象形状が少なすぎる");
}

#[test]
fn overlapping_windows_accumulate_gradients_hand_computed() {
    // D 軸 in=3,out=2: 窓 [0,2),[1,3)。Avg の勾配は [0.5, 1.0, 0.5]、Max（入力昇順）は [0, 1, 1]。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0, 1.0, 2.0], &[1, 1, 3, 1, 1]));
    let y = adaptive_avg_pool3d(&x, [2, 1, 1]).unwrap();
    let loss = y.sum(None).unwrap();
    let dx = tape.backward(&loss).unwrap();
    assert_eq!(
        dx.get(&x).unwrap().unwrap().host_slice().into_owned(),
        [0.5, 1.0, 0.5]
    );
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0, 1.0, 2.0], &[1, 1, 3, 1, 1]));
    let (y, idx) = adaptive_max_pool3d(&x, [2, 1, 1]).unwrap();
    assert_eq!(idx.contiguous().host_slice().into_owned(), [1, 2]);
    let loss = y.sum(None).unwrap();
    let dx = tape.backward(&loss).unwrap();
    assert_eq!(
        dx.get(&x).unwrap().unwrap().host_slice().into_owned(),
        [0.0, 1.0, 1.0]
    );
}

#[test]
fn non_contiguous_input_is_pooled_in_logical_order() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let x = tape.var(&t(data, &[1, 1, 2, 2, 2]));
    // D と W を入れ替えた view（非連続）。
    let xt = x.transpose(2, 4).unwrap();
    let (v, idx) = adaptive_max_pool3d(&xt, [1, 1, 1]).unwrap();
    assert_eq!(v.to_tensor().host_slice().into_owned(), [7.0]);
    assert_eq!(idx.contiguous().host_slice().into_owned(), [7]);
    // 転置後 d=0 の平面 = 元の w=0 → {0,2,4,6} 平均 3、d=1 → {1,3,5,7} 平均 4。
    let a = adaptive_avg_pool3d(&xt, [2, 1, 1]).unwrap();
    assert_eq!(a.to_tensor().host_slice().into_owned(), [3.0, 4.0]);
}

// --- フォールバックとエラー伝播 ---

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
    fn adaptive_pool3d_max(
        &self,
        _input: &Tensor<f32>,
        _output_size: [usize; 3],
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
    fn adaptive_pool3d_avg(
        &self,
        _input: &Tensor<f32>,
        _output_size: [usize; 3],
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
    let (v, i) = adaptive_max_pool3d(&x, [1, 1, 1]).unwrap();
    assert_eq!(v.to_tensor().host_slice().into_owned(), [7.0]);
    assert_eq!(i.contiguous().host_slice().into_owned(), [7]);
    let a = adaptive_avg_pool3d(&x, [1, 1, 1]).unwrap();
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
        adaptive_max_pool3d(&x, [1, 1, 1]).map(|_| ()),
        adaptive_avg_pool3d(&x, [1, 1, 1]).map(|_| ()),
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
            adaptive_max_pool3d(&x, [1, 1, 1]),
            Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
        ));
        assert!(matches!(
            adaptive_avg_pool3d(&x, [1, 1, 1]),
            Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
        ));
    }
}

// --- 境界・テープ・決定性 ---

#[test]
fn invalid_arguments_are_typed_errors_and_leave_no_orphan_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = cube(&tape);
    let x4 = tape.var(&t(vec![0.0; 8], &[1, 2, 2, 2]));
    let z = tape.var(&t(vec![], &[1, 1, 0, 2, 2]));
    let before = tape.len();
    for os in [[0, 1, 1], [1, 0, 1], [1, 1, 0]] {
        assert!(matches!(
            adaptive_avg_pool3d(&x, os),
            Err(AutodiffError::Shape(_))
        ));
        assert!(matches!(
            adaptive_max_pool3d(&x, os),
            Err(AutodiffError::Shape(_))
        ));
    }
    assert!(matches!(
        adaptive_avg_pool3d(&x4, [1, 1, 1]),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    assert!(matches!(
        adaptive_max_pool3d(&x4, [1, 1, 1]),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    assert!(adaptive_avg_pool3d(&z, [1, 1, 1]).is_err());
    assert!(adaptive_max_pool3d(&z, [1, 1, 1]).is_err());
    // 出力要素数のオーバーフロー。
    assert!(adaptive_avg_pool3d(&x, [usize::MAX, usize::MAX, 2]).is_err());
    assert_eq!(tape.len(), before, "検査は tape 操作より前なので増えない");
}

#[test]
fn empty_batch_and_channel_yield_empty_outputs() {
    let tape = Tape::new_with_ops(common::naive_ops());
    for shape in [[0usize, 2, 4, 4, 4], [2, 0, 4, 4, 4]] {
        let x = tape.var(&t(vec![], &shape));
        let (v, i) = adaptive_max_pool3d(&x, [2, 3, 1]).unwrap();
        assert_eq!(v.to_tensor().shape(), &[shape[0], shape[1], 2, 3, 1]);
        assert_eq!(i.shape(), &[shape[0], shape[1], 2, 3, 1]);
        let a = adaptive_avg_pool3d(&x, [2, 3, 1]).unwrap();
        assert_eq!(a.to_tensor().shape(), &[shape[0], shape[1], 2, 3, 1]);
    }
}

#[test]
fn empty_batch_still_rejects_plane_beyond_i32_for_max() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // 実データを持たない shape（N=0）。D·H·W = 2^31 > i32::MAX。
    let x = tape.var(&t(vec![], &[0, 1, 2048, 1024, 1024]));
    assert!(matches!(
        adaptive_max_pool3d(&x, [1, 1, 1]),
        Err(AutodiffError::Shape(ShapeError::IndexRangeOverflow { .. }))
    ));
    // Avg は索引を持たないので受理される。
    assert!(adaptive_avg_pool3d(&x, [1, 1, 1]).is_ok());
}

#[test]
fn huge_broadcast_view_is_rejected_before_allocation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let base = tape.var(&t(vec![1.0], &[1, 1, 1, 1, 1]));
    let huge = base.broadcast_to(&[1, 1, 1, 1, 1usize << 61]).unwrap();
    for r in [
        adaptive_max_pool3d(&huge, [1, 1, 1]).map(|_| ()),
        adaptive_avg_pool3d(&huge, [1, 1, 1]).map(|_| ()),
    ] {
        assert!(r.is_err());
    }
}

#[test]
fn each_op_records_exactly_one_node() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = cube(&tape);
    let before = tape.len();
    adaptive_max_pool3d(&x, [1, 1, 1]).unwrap();
    assert_eq!(tape.len(), before + 1);
    adaptive_avg_pool3d(&x, [2, 1, 1]).unwrap();
    assert_eq!(tape.len(), before + 2);
}

#[test]
fn results_are_bit_deterministic_run_to_run() {
    let run = || {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(pseudo(2 * 2 * 5 * 4 * 3, 7, 23), &[2, 2, 5, 4, 3]));
        let a = adaptive_avg_pool3d(&x, [3, 2, 2]).unwrap();
        let loss = a.mul(&a).unwrap().sum(None).unwrap();
        let g = tape.backward(&loss).unwrap();
        let bits = |v: Vec<f32>| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        (
            bits(a.to_tensor().host_slice().into_owned()),
            bits(g.get(&x).unwrap().unwrap().host_slice().into_owned()),
        )
    };
    assert_eq!(run(), run());
}

// --- 実 PyTorch 2.14 の実行値 fixture との突合 ---
//
// fixture: `tests/fixtures/adaptive-pool3d-pytorch-reference/adaptive_pool3d_reference.json`
// （生成条件・sha256 は同ディレクトリの README.md）。f32 は u32 ビットパターンで保存されている。
// Max の値は選択のみなので bit 一致（NaN はクラス一致）・索引は完全一致、Avg の forward と勾配は
// REQ-2 統一複合判定（`common::req2_close`。tolerance 定数は新設しない。PyTorch は f32 累積・本実装は
// `f64` 累積のため bit 一致は求めない）。NaN を含む窓の索引・勾配は比較対象から除外する
// （PyTorch は最後の NaN・本実装は最初の NaN。`docs/autodiff-pool3d-ops-decision.md` §5・§21）。

use serde::Deserialize;

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
    output_size: Vec<usize>,
    x_bits: Vec<u32>,
    g_bits: Vec<u32>,
    out_bits: Vec<u32>,
    grad_bits: Vec<u32>,
    index: Option<Vec<i32>>,
}

#[derive(Deserialize)]
struct ErrCase {
    name: String,
    in_shape: Vec<usize>,
    output_size: Vec<usize>,
    torch_raises: bool,
    out_shape: Vec<usize>,
}

fn load_fixture() -> Fixture {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/adaptive-pool3d-pytorch-reference/adaptive_pool3d_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fn arr3(v: &[usize]) -> [usize; 3] {
    [v[0], v[1], v[2]]
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

fn run_fixture_case(case: &Case) -> Run {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(from_bits(&case.x_bits), &case.in_shape));
    let os = arr3(&case.output_size);
    let (y, index) = if case.op == "max" {
        let (y, i) = adaptive_max_pool3d(&x, os).unwrap();
        (y, Some(i.contiguous().host_slice().into_owned()))
    } else {
        (adaptive_avg_pool3d(&x, os).unwrap(), None)
    };
    let out = y.to_tensor();
    let g = tape.var_no_grad(&t(from_bits(&case.g_bits), out.shape()));
    let loss = y.mul(&g).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    Run {
        values: out.host_slice().into_owned(),
        out_shape: out.shape().to_vec(),
        index,
        grad: grads.get(&x).unwrap().unwrap().host_slice().into_owned(),
    }
}

#[test]
fn finite_cases_match_pytorch_reference_forward_and_backward() {
    let fixture = load_fixture();
    assert!(
        fixture.torch_version.starts_with("2.14"),
        "fixture は PyTorch 2.14 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    let (mut max_cases, mut avg_cases) = (0, 0);
    for case in &fixture.finite_cases {
        let r = run_fixture_case(case);
        assert_eq!(r.out_shape, case.out_shape, "{}: 出力 shape", case.name);
        let want = from_bits(&case.out_bits);
        if case.op == "max" {
            max_cases += 1;
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
        max_cases >= 11 && avg_cases >= 11,
        "{max_cases}/{avg_cases}"
    );
}

#[test]
fn nonfinite_forward_and_gradients_match_pytorch_reference() {
    let fixture = load_fixture();
    assert!(fixture.nonfinite_cases.len() >= 12);
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

/// PyTorch は最後の NaN、本実装は最初の NaN の索引を残す（意味論の決定。2D・pool3d と同じ）。
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
    let index = r.index.expect("max は索引を返す");
    // 入力は [1,1,3,3,3]・出力 1 要素（全体窓）。NaN は x[4]・x[13]・x[14] にあり、最初は 4。
    assert_eq!(index.len(), 1);
    assert!(r.values[0].is_nan());
    let first = x.iter().position(|v| v.is_nan()).unwrap();
    assert_eq!(usize::try_from(index[0]).unwrap(), first);
}

/// torch の実測（例外の有無）との照合。意図した差分は個別に固定する。
#[test]
fn error_cases_agree_with_torch_except_documented_differences() {
    let fixture = load_fixture();
    assert!(fixture.error_cases.len() >= 16);
    for case in &fixture.error_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let n: usize = case.in_shape.iter().product();
        let x = tape.var(&t(vec![0.0; n], &case.in_shape));
        let os = arr3(&case.output_size);
        let r = if case.name.starts_with("max_") {
            adaptive_max_pool3d(&x, os).map(|(v, _)| v)
        } else {
            adaptive_avg_pool3d(&x, os)
        };
        match case.name.as_str() {
            // 意図した差分 1: バッチなし rank 4 は本実装が拒否する（NCDHW の rank 5 のみ）。
            n if n.ends_with("rank4_unbatched") => {
                assert!(!case.torch_raises);
                assert!(
                    matches!(
                        r,
                        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
                    ),
                    "{n}"
                );
            }
            // 意図した差分 2: output_size の 0 は torch が空出力で受理するが、本実装は各軸 1 以上を要求する。
            n if n.contains("output_zero") => {
                assert!(!case.torch_raises);
                assert!(matches!(r, Err(AutodiffError::Shape(_))), "{n}");
            }
            // 意図した差分 3: 空間軸 0 は torch の adaptive_avg が受理する構成があるが、本実装は常に拒否する。
            n if n.ends_with("spatial_zero_w") || n.ends_with("spatial_zero_d") => {
                assert!(matches!(r, Err(AutodiffError::Shape(_))), "{n}");
            }
            // 意図した差分 4: C=0 は torch が拒否するが、本実装は N=0 と同じく空出力を受理する。
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
