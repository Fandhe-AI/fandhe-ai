//! `indexed_update_ops`（イシュー #2641・`scatter_reduce`／`index_add`／`index_copy`／
//! `masked_scatter`）の `Tape`／`Var` を経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/indexed-update-pytorch-reference/indexed_update_reference.json`・
//!   生成条件は同ディレクトリの `README.md`）と突合する。f32 は NaN／inf を運べる
//!   よう u32 ビットパターンで保存されている。値と勾配は REQ-2 統一複合判定
//!   （`common::req2_close`。tolerance 定数は新設しない）。`Amax`／`Amin`・
//!   `index_copy`・`masked_scatter` の forward は選択・コピーのみなので bit 一致。
//! - 非有限入力・`include_self == false` で入力が偶然結果と一致する群は PyTorch との
//!   実測差分を名前で列挙して固定し、差がその集合に限られることを検査する
//!   （`docs/autodiff-indexed-update-ops-decision.md` §5）。
//! - `common::naive_ops()` は `indexed_scatter_reduce` を override しないため、必ず
//!   共有ホストカーネルへのフォールバック経路を通る。CPU `BackendOps` 実装との
//!   一致は `crates/facade/tests/indexed_update_ops_backend_parity.rs` が担当する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::indexed_update_ops::{
    index_add, index_copy, masked_scatter, scatter_reduce,
};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, ScatterReduce, ScatterReduceMode, ShapeError, Tensor};
use serde::Deserialize;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    cases: Vec<Case>,
    error_cases: Vec<ErrCase>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    op: String,
    group: String,
    x_shape: Vec<usize>,
    x_bits: Vec<u32>,
    src_shape: Vec<usize>,
    src_bits: Vec<u32>,
    g_bits: Vec<u32>,
    out_bits: Vec<u32>,
    grad_x_bits: Option<Vec<u32>>,
    grad_src_bits: Option<Vec<u32>>,
    dim: Option<usize>,
    index: Option<Vec<i32>>,
    index_shape: Option<Vec<usize>>,
    reduce: Option<String>,
    include_self: Option<bool>,
    mask_shape: Option<Vec<usize>>,
    mask: Option<Vec<u8>>,
}

#[derive(Deserialize)]
struct ErrCase {
    name: String,
    torch_raises: bool,
    out_shape: Vec<usize>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/indexed-update-pytorch-reference/indexed_update_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

fn x4(tape: &Tape) -> Var<'_> {
    tape.var(&t(vec![0.0; 4], &[4]))
}

fn mode_of(name: &str) -> ScatterReduceMode {
    match name {
        "sum" => ScatterReduceMode::Sum,
        "prod" => ScatterReduceMode::Prod,
        "mean" => ScatterReduceMode::Mean,
        "amax" => ScatterReduceMode::Amax,
        "amin" => ScatterReduceMode::Amin,
        other => panic!("未知の reduce: {other}"),
    }
}

/// NaN は NaN 同士（クラス一致）、それ以外は bit 完全一致。
fn bits_match(actual: &[f32], expected: &[f32]) -> bool {
    actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(&a, &e)| {
            if e.is_nan() {
                a.is_nan()
            } else {
                a.to_bits() == e.to_bits()
            }
        })
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

fn all_close(actual: &[f32], expected: &[f32]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(&a, &e)| class_close(a, e))
}

struct Run {
    values: Vec<f32>,
    grad_x: Vec<f32>,
    grad_src: Vec<f32>,
}

fn apply<'t>(case: &Case, x: &Var<'t>, s: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    match case.op.as_str() {
        "scatter_reduce" => {
            let idx = ti(
                case.index.clone().expect("index"),
                case.index_shape.as_ref().expect("index_shape"),
            );
            scatter_reduce(
                x,
                case.dim.expect("dim"),
                &idx,
                s,
                mode_of(case.reduce.as_deref().expect("reduce")),
                case.include_self.expect("include_self"),
            )
        }
        "index_add" | "index_copy" => {
            let idx = ti(
                case.index.clone().expect("index"),
                case.index_shape.as_ref().expect("index_shape"),
            );
            let dim = case.dim.expect("dim");
            if case.op == "index_add" {
                index_add(x, dim, &idx, s)
            } else {
                index_copy(x, dim, &idx, s)
            }
        }
        "masked_scatter" => {
            let mask = Tensor::new(
                case.mask
                    .as_ref()
                    .expect("mask")
                    .iter()
                    .map(|&b| b != 0)
                    .collect(),
                case.mask_shape.as_ref().expect("mask_shape"),
            )
            .expect("mask");
            masked_scatter(x, &mask, s)
        }
        other => panic!("未知の op: {other}"),
    }
}

/// `(out * g).sum()` を損失とした forward 値と入力・src 勾配を返す。
fn run_case(ops: Box<dyn BackendOps + Send>, case: &Case) -> Run {
    let tape = Tape::new_with_ops(ops);
    let xv = tape.var(&t(from_bits(&case.x_bits), &case.x_shape));
    let sv = tape.var(&t(from_bits(&case.src_bits), &case.src_shape));
    let y = apply(case, &xv, &sv).expect("op");
    let values = y.to_tensor().host_slice().into_owned();
    let gv = tape.var_no_grad(&t(from_bits(&case.g_bits), y.to_tensor().shape()));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let grad_of = |v: &Var<'_>, n: usize| -> Vec<f32> {
        grads
            .get(v)
            .unwrap()
            .map(|g| g.host_slice().into_owned())
            .unwrap_or_else(|| vec![0.0; n])
    };
    Run {
        values,
        grad_x: grad_of(&xv, case.x_bits.len()),
        grad_src: grad_of(&sv, case.src_bits.len()),
    }
}

/// forward・入力勾配・src 勾配のうち PyTorch 実測と一致しないものを返す。
fn mismatches(case: &Case, r: &Run) -> Vec<&'static str> {
    let exact = matches!(case.reduce.as_deref(), Some("amax") | Some("amin") | None)
        && case.op != "index_add";
    let want = from_bits(&case.out_bits);
    let fwd_ok = if exact {
        bits_match(&r.values, &want)
    } else {
        all_close(&r.values, &want)
    };
    let mut bad = Vec::new();
    if !fwd_ok {
        bad.push("forward");
    }
    if let Some(g) = &case.grad_x_bits
        && !all_close(&r.grad_x, &from_bits(g))
    {
        bad.push("grad_x");
    }
    if let Some(g) = &case.grad_src_bits
        && !all_close(&r.grad_src, &from_bits(g))
    {
        bad.push("grad_src");
    }
    bad
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
    let finite: Vec<&Case> = fixture
        .cases
        .iter()
        .filter(|c| c.group == "finite")
        .collect();
    assert!(finite.len() >= 160, "有限ケース数: {}", finite.len());
    let mut failures = Vec::new();
    for case in finite {
        let bad = mismatches(case, &run_fixture_case(case));
        if !bad.is_empty() {
            failures.push(format!("{}: {bad:?}", case.name));
        }
    }
    assert!(failures.is_empty(), "不一致: {failures:#?}");
}

/// PyTorch との実測差分（forward は一致・勾配のみ差が出る）を名前で固定する。
/// 差分の理由は決定記録 §5。ここに無いケースは完全に一致するはず。
const KNOWN_GRAD_DIFFS: &[&str] = &[
    "sr_tie_vs_self_amax_noself",
    "sr_tie_vs_self_amin_noself",
    "sr_signed_zero_amax_noself",
    "sr_signed_zero_amin_noself",
    "sr_selfeq_amax_amax_noself",
    "sr_selfeq_amin_amin_noself",
    "sr_nan_src_amax_self",
    "sr_nan_src_amax_noself",
    "sr_nan_src_amin_self",
    "sr_nan_src_amin_noself",
    "sr_nan_self_amax_self",
    "sr_nan_self_amin_self",
    "sr_nan_self_amin_noself",
    "sr_posinf_src_amin_noself",
    "sr_neginf_src_amax_noself",
    "sr_neginf_src_amin_noself",
];

#[test]
fn selfeq_and_nonfinite_cases_differ_from_pytorch_only_in_known_cases() {
    let fixture = load_fixture();
    let mut diffs: Vec<String> = Vec::new();
    for case in fixture.cases.iter().filter(|c| c.group != "finite") {
        let bad = mismatches(case, &run_fixture_case(case));
        // forward は常に一致する（差が出るのは勾配の分配のみ）。
        assert!(!bad.contains(&"forward"), "{}: forward 不一致", case.name);
        if !bad.is_empty() {
            assert!(
                matches!(case.reduce.as_deref(), Some("amax") | Some("amin")),
                "{}: 差分は amax/amin の勾配分配に限る",
                case.name
            );
            diffs.push(case.name.clone());
        }
    }
    let expected: Vec<String> = KNOWN_GRAD_DIFFS.iter().map(|s| (*s).to_string()).collect();
    assert_eq!(diffs, expected);
}

#[test]
fn error_cases_agree_with_torch_where_in_scope() {
    let fixture = load_fixture();
    for case in &fixture.error_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let one = tape.var(&t(vec![0.0], &[1]));
        let ours: Result<Var<'_>, AutodiffError> = match case.name.as_str() {
            "sr_idx_oob" => scatter_reduce(
                &x4(&tape),
                0,
                &ti(vec![4], &[1]),
                &one,
                ScatterReduceMode::Sum,
                true,
            ),
            "sr_idx_neg" => scatter_reduce(
                &x4(&tape),
                0,
                &ti(vec![-1], &[1]),
                &one,
                ScatterReduceMode::Sum,
                true,
            ),
            "sr_dim_oob" => scatter_reduce(
                &x4(&tape),
                1,
                &ti(vec![0], &[1]),
                &one,
                ScatterReduceMode::Sum,
                true,
            ),
            "sr_idx_gt_src" => scatter_reduce(
                &x4(&tape),
                0,
                &ti(vec![0, 1], &[2]),
                &one,
                ScatterReduceMode::Sum,
                true,
            ),
            "sr_idx_lt_src" => {
                let s2 = tape.var(&t(vec![0.0; 2], &[2]));
                scatter_reduce(
                    &x4(&tape),
                    0,
                    &ti(vec![0], &[1]),
                    &s2,
                    ScatterReduceMode::Sum,
                    true,
                )
            }
            "sr_idx_gt_input" => {
                let x = tape.var(&t(vec![0.0; 6], &[2, 3]));
                let s = tape.var(&t(vec![0.0; 12], &[4, 3]));
                scatter_reduce(
                    &x,
                    0,
                    &ti(vec![0; 12], &[4, 3]),
                    &s,
                    ScatterReduceMode::Sum,
                    true,
                )
            }
            "sr_rank0" => {
                let x = tape.var(&t(vec![1.0], &[]));
                scatter_reduce(
                    &x,
                    0,
                    &ti(vec![0], &[1]),
                    &one,
                    ScatterReduceMode::Sum,
                    true,
                )
            }
            "ia_idx_oob" => index_add(&x4(&tape), 0, &ti(vec![4], &[1]), &one),
            "ia_idx_neg" => index_add(&x4(&tape), 0, &ti(vec![-1], &[1]), &one),
            "ia_len_mismatch" => {
                let s = tape.var(&t(vec![0.0; 3], &[3]));
                index_add(&x4(&tape), 0, &ti(vec![0, 1], &[2]), &s)
            }
            "ia_other_dim_mismatch" => {
                let x = tape.var(&t(vec![0.0; 12], &[3, 4]));
                let s = tape.var(&t(vec![0.0; 5], &[1, 5]));
                index_add(&x, 0, &ti(vec![0], &[1]), &s)
            }
            "ia_dim_oob" => index_add(&x4(&tape), 1, &ti(vec![0], &[1]), &one),
            "ic_idx_oob" => index_copy(&x4(&tape), 0, &ti(vec![4], &[1]), &one),
            "ic_len_mismatch" => {
                let s = tape.var(&t(vec![0.0; 3], &[3]));
                index_copy(&x4(&tape), 0, &ti(vec![0, 1], &[2]), &s)
            }
            "ic_other_dim_mismatch" => {
                let x = tape.var(&t(vec![0.0; 12], &[3, 4]));
                let s = tape.var(&t(vec![0.0; 5], &[1, 5]));
                index_copy(&x, 0, &ti(vec![0], &[1]), &s)
            }
            "ms_source_short" => {
                let s = tape.var(&t(vec![0.0; 3], &[3]));
                let m = Tensor::new(vec![true; 4], &[4]).unwrap();
                masked_scatter(&x4(&tape), &m, &s)
            }
            "ms_mask_not_broadcastable" => {
                let x = tape.var(&t(vec![0.0; 3], &[3]));
                let s = tape.var(&t(vec![0.0; 3], &[3]));
                let m = Tensor::new(vec![true, false], &[2]).unwrap();
                masked_scatter(&x, &m, &s)
            }
            other => panic!("未対応のエラーケース: {other}"),
        };
        // 本実装が受理するのは torch も受理する場合のみ（差分は下記の列挙のみ）。
        // `sr_idx_lt_src`（index が src より小さい形）と `sr_rank0` は torch が受理するが
        // 本実装は fail-closed に拒否する（決定記録 §5・スコープ外）。
        let stricter = matches!(case.name.as_str(), "sr_idx_lt_src" | "sr_rank0");
        if case.torch_raises || stricter {
            assert!(ours.is_err(), "{}: エラーのはず", case.name);
        } else {
            let v = ours.unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
            assert_eq!(
                v.to_tensor().shape(),
                case.out_shape.as_slice(),
                "{}",
                case.name
            );
        }
    }
}

// --- 独立オラクル ---

#[test]
fn sum_include_self_is_bit_identical_to_scatter_add() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.1, 0.2, 0.3, 0.4, 0.5], &[5]));
    let s = tape.var(&t(vec![1e-3, 2.5, -7.25, 3.0e-2, 4.0], &[5]));
    let idx = ti(vec![0, 2, 0, 2, 4], &[5]);
    let a = scatter_reduce(&x, 0, &idx, &s, ScatterReduceMode::Sum, true).unwrap();
    let b = x.scatter_add(0, &idx, &s).unwrap();
    let bits = |v: &Var<'_>| {
        v.to_tensor()
            .host_slice()
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&a), bits(&b));
}

#[test]
fn index_add_equals_hand_expanded_scatter_add() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((0..6).map(|v| v as f32).collect(), &[3, 2]));
    let s = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));
    let a = index_add(&x, 0, &ti(vec![2, 0, 2], &[3]), &s).unwrap();
    let idx = ti(vec![2, 2, 0, 0, 2, 2], &[3, 2]);
    let b = x.scatter_add(0, &idx, &s).unwrap();
    assert_eq!(
        a.to_tensor().host_slice().into_owned(),
        b.to_tensor().host_slice().into_owned()
    );
}

#[test]
fn index_copy_duplicate_indices_last_writer_wins_with_gradient() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 3], &[3]));
    let s = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let y = index_copy(&x, 0, &ti(vec![1, 1, 0], &[3]), &s).unwrap();
    assert_eq!(y.to_tensor().host_slice().into_owned(), vec![3.0, 2.0, 0.0]);
    let g = tape.var_no_grad(&t(vec![10.0, 20.0, 30.0], &[3]));
    let loss = y.mul(&g).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    assert_eq!(
        grads.get(&s).unwrap().unwrap().host_slice().into_owned(),
        vec![0.0, 20.0, 10.0]
    );
    assert_eq!(
        grads.get(&x).unwrap().unwrap().host_slice().into_owned(),
        vec![0.0, 0.0, 30.0]
    );
}

#[test]
fn masked_scatter_extra_source_elements_get_zero_gradient() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let s = tape.var(&t(vec![7.0, 8.0, 9.0, 10.0], &[4]));
    let m = Tensor::new(vec![true, false, true], &[3]).unwrap();
    let y = masked_scatter(&x, &m, &s).unwrap();
    assert_eq!(y.to_tensor().host_slice().into_owned(), vec![7.0, 2.0, 8.0]);
    let g = tape.var_no_grad(&t(vec![1.0, 2.0, 3.0], &[3]));
    let loss = y.mul(&g).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    assert_eq!(
        grads.get(&s).unwrap().unwrap().host_slice().into_owned(),
        vec![1.0, 3.0, 0.0, 0.0]
    );
    assert_eq!(
        grads.get(&x).unwrap().unwrap().host_slice().into_owned(),
        vec![0.0, 2.0, 0.0]
    );
}

#[test]
fn scatter_reduce_gradient_matches_central_difference() {
    let x0 = [1.5_f32, -2.0, 0.75, 3.0];
    let s0 = [2.5_f32, -1.25, 4.0, 0.5, -3.5];
    let w = [0.3_f32, -0.7, 1.1, 0.9];
    let idx = ti(vec![0, 0, 2, 2, 2], &[5]);
    for mode in [
        ScatterReduceMode::Sum,
        ScatterReduceMode::Prod,
        ScatterReduceMode::Mean,
        ScatterReduceMode::Amax,
        ScatterReduceMode::Amin,
    ] {
        for inc in [true, false] {
            let eval = |xs: &[f32], ss: &[f32]| -> f64 {
                let tape = Tape::new_with_ops(common::naive_ops());
                let xv = tape.var(&t(xs.to_vec(), &[4]));
                let sv = tape.var(&t(ss.to_vec(), &[5]));
                let y = scatter_reduce(&xv, 0, &idx, &sv, mode, inc).unwrap();
                y.to_tensor()
                    .host_slice()
                    .iter()
                    .zip(&w)
                    .map(|(&o, &c)| f64::from(o) * f64::from(c))
                    .sum()
            };
            let tape = Tape::new_with_ops(common::naive_ops());
            let xv = tape.var(&t(x0.to_vec(), &[4]));
            let sv = tape.var(&t(s0.to_vec(), &[5]));
            let y = scatter_reduce(&xv, 0, &idx, &sv, mode, inc).unwrap();
            let gv = tape.var_no_grad(&t(w.to_vec(), &[4]));
            let loss = y.mul(&gv).unwrap().sum(None).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let dx = grads.get(&xv).unwrap().unwrap().host_slice().into_owned();
            let ds = grads.get(&sv).unwrap().unwrap().host_slice().into_owned();
            let h = 1e-2_f32;
            for j in 0..4 {
                let (mut a, mut b) = (x0, x0);
                a[j] += h;
                b[j] -= h;
                let fd = (eval(&a, &s0) - eval(&b, &s0)) / (2.0 * f64::from(h));
                assert!(
                    (f64::from(dx[j]) - fd).abs() < 5e-3,
                    "{mode:?} {inc} dx[{j}]"
                );
            }
            for j in 0..5 {
                let (mut a, mut b) = (s0, s0);
                a[j] += h;
                b[j] -= h;
                let fd = (eval(&x0, &a) - eval(&x0, &b)) / (2.0 * f64::from(h));
                assert!(
                    (f64::from(ds[j]) - fd).abs() < 5e-3,
                    "{mode:?} {inc} ds[{j}]"
                );
            }
        }
    }
}

// --- フォールバックとエラー伝播 ---

#[derive(Clone, Copy)]
enum Mode {
    Unsupported,
    LaunchFailed,
    WrongShape,
}

/// `indexed_scatter_reduce`／`scatter` だけを差し替える `BackendOps`。
struct Mock {
    inner: Box<dyn BackendOps + Send>,
    calls: Arc<AtomicUsize>,
    mode: Mode,
}

impl BackendOps for Mock {
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
    fn indexed_scatter_reduce(
        &self,
        _input: &Tensor<f32>,
        _dim: usize,
        _index: &Tensor<i32>,
        _src: &Tensor<f32>,
        _mode: ScatterReduceMode,
        _include_self: bool,
    ) -> Result<Tensor<f32>, BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            Mode::WrongShape => Ok(t(vec![0.0; 3], &[3])),
        }
    }
    fn scatter(
        &self,
        _input: &Tensor<f32>,
        _dim: usize,
        _index: &Tensor<i32>,
        _src: &Tensor<f32>,
        _reduce: ScatterReduce,
    ) -> Result<Tensor<f32>, BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(BackendError::Unsupported("mock scatter".into()))
    }
}

fn mock_tape(mode: Mode) -> (Tape, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let tape = Tape::new_with_ops(Box::new(Mock {
        inner: common::naive_ops(),
        calls: Arc::clone(&calls),
        mode,
    }));
    (tape, calls)
}

#[test]
fn unsupported_backend_falls_back_to_host_reference() {
    let (tape, calls) = mock_tape(Mode::Unsupported);
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let s = tape.var(&t(vec![10.0, 20.0], &[2]));
    let idx = ti(vec![0, 0], &[2]);
    let y = scatter_reduce(&x, 0, &idx, &s, ScatterReduceMode::Sum, true).unwrap();
    assert_eq!(
        y.to_tensor().host_slice().into_owned(),
        vec![31.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1, "バックエンドを先に呼ぶ");
}

#[test]
fn composed_ops_reach_host_scatter_when_backend_scatter_is_unsupported() {
    let (tape, calls) = mock_tape(Mode::Unsupported);
    let x = tape.var(&t(vec![0.0; 4], &[4]));
    let s = tape.var(&t(vec![1.0, 2.0], &[2]));
    let a = index_add(&x, 0, &ti(vec![3, 3], &[2]), &s).unwrap();
    assert_eq!(
        a.to_tensor().host_slice().into_owned(),
        vec![0.0, 0.0, 0.0, 3.0]
    );
    let c = index_copy(&x, 0, &ti(vec![1, 2], &[2]), &s).unwrap();
    assert_eq!(
        c.to_tensor().host_slice().into_owned(),
        vec![0.0, 1.0, 2.0, 0.0]
    );
    let m = Tensor::new(vec![false, true, true, false], &[4]).unwrap();
    let ms = masked_scatter(&x, &m, &s).unwrap();
    assert_eq!(
        ms.to_tensor().host_slice().into_owned(),
        vec![0.0, 1.0, 2.0, 0.0]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3, "scatter を 3 回先に呼ぶ");
}

#[test]
fn non_unsupported_backend_errors_are_propagated() {
    let (tape, _) = mock_tape(Mode::LaunchFailed);
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let s = tape.var(&t(vec![1.0], &[1]));
    let r = scatter_reduce(&x, 0, &ti(vec![0], &[1]), &s, ScatterReduceMode::Sum, true);
    assert!(matches!(
        r,
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
}

#[test]
fn wrong_shape_from_backend_is_a_typed_error() {
    let (tape, _) = mock_tape(Mode::WrongShape);
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let s = tape.var(&t(vec![1.0], &[1]));
    let r = scatter_reduce(&x, 0, &ti(vec![0], &[1]), &s, ScatterReduceMode::Sum, true);
    assert!(matches!(
        r,
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
}

// --- 境界・テープ・決定性 ---

#[test]
fn each_op_records_expected_node_counts_and_errors_record_none() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 3.0, 2.0], &[3]));
    let s = tape.var(&t(vec![1.0, 1.0], &[2]));
    let idx = ti(vec![0, 2], &[2]);
    let before = tape.len();
    scatter_reduce(&x, 0, &idx, &s, ScatterReduceMode::Prod, false).unwrap();
    assert_eq!(tape.len(), before + 1);
    index_add(&x, 0, &idx, &s).unwrap();
    assert_eq!(tape.len(), before + 2);
    index_copy(&x, 0, &idx, &s).unwrap();
    assert_eq!(tape.len(), before + 3);
    let after = tape.len();
    assert!(
        scatter_reduce(
            &x,
            0,
            &ti(vec![5, 0], &[2]),
            &s,
            ScatterReduceMode::Sum,
            true
        )
        .is_err()
    );
    assert!(index_add(&x, 0, &ti(vec![5, 0], &[2]), &s).is_err());
    assert!(index_copy(&x, 3, &idx, &s).is_err());
    let m = Tensor::new(vec![true; 3], &[3]).unwrap();
    assert!(masked_scatter(&x, &m, &s).is_err());
    assert_eq!(tape.len(), after);
}

#[test]
fn other_tape_var_is_tape_mismatch() {
    let a = Tape::new_with_ops(common::naive_ops());
    let b = Tape::new_with_ops(common::naive_ops());
    let x = a.var(&t(vec![1.0, 2.0], &[2]));
    let s = b.var(&t(vec![1.0], &[1]));
    let idx = ti(vec![0], &[1]);
    assert!(matches!(
        scatter_reduce(&x, 0, &idx, &s, ScatterReduceMode::Sum, true),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        index_add(&x, 0, &idx, &s),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        index_copy(&x, 0, &idx, &s),
        Err(AutodiffError::TapeMismatch)
    ));
    let m = Tensor::new(vec![true, false], &[2]).unwrap();
    assert!(matches!(
        masked_scatter(&x, &m, &s),
        Err(AutodiffError::TapeMismatch)
    ));
}

#[test]
fn huge_broadcast_view_is_rejected_before_allocation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let base = tape.var(&t(vec![1.0], &[1]));
    let huge = base.broadcast_to(&[1usize << 61]).unwrap();
    let s = tape.var(&t(vec![1.0], &[1]));
    let idx = ti(vec![0], &[1]);
    let r = scatter_reduce(&huge, 0, &idx, &s, ScatterReduceMode::Sum, true);
    assert!(matches!(
        r,
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
    let r = index_add(&huge, 0, &idx, &s);
    assert!(matches!(
        r,
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
    let m = Tensor::new(vec![true], &[1]).unwrap();
    assert!(matches!(
        masked_scatter(&huge, &m, &s),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
}

#[test]
fn create_graph_is_a_typed_error_not_a_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let s = tape.var(&t(vec![1.0], &[1]));
    let loss = scatter_reduce(&x, 0, &ti(vec![0], &[1]), &s, ScatterReduceMode::Sum, true)
        .unwrap()
        .sum(None)
        .unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
}

#[test]
fn results_are_bit_deterministic_run_to_run() {
    let fixture = load_fixture();
    for case in fixture.cases.iter().filter(|c| c.name.contains("r3_")) {
        let a = run_fixture_case(case);
        let b = run_fixture_case(case);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.values), bits(&b.values), "{}", case.name);
        assert_eq!(bits(&a.grad_x), bits(&b.grad_x), "{}", case.name);
        assert_eq!(bits(&a.grad_src), bits(&b.grad_src), "{}", case.name);
    }
}
