//! `merge_ops`（イシュー #2666・`merge_concatenate`／`merge_add`／`merge_multiply`／
//! `merge_average`）の `Tape`／`Var` を経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/merge-ops-pytorch-reference/merge_ops_reference.json`・生成条件は
//!   同ディレクトリの `README.md`）と突合する。forward・全入力勾配とも REQ-2 統一複合判定
//!   （`common::req2_close`。tolerance 定数は新設しない）。
//! - `error_cases` は torch が例外を出すケースが型付きエラーになること、torch が受理するが
//!   本実装が拒否するケース（`INTENDED_DIFFS`）が `docs/facade-functional-api-decision.md`
//!   §17 と一対一であることを固定する。拒否時に tape へノードが 1 つも増えないことも確認する。
//! - tape 記録数・Average の shape 不変・連鎖長上限（`MAX_FUSED_CHAIN_LEN`）を超える畳み込み・
//!   非 contiguous 入力・run-to-run の bit 決定性・クロステープ拒否も固定する。
//!   CPU `BackendOps` との一致は `crates/facade/tests/merge_ops_backend_parity.rs` が担当する。

mod common;

use std::path::PathBuf;

use fandhe_ai_autodiff::merge_ops::{merge_add, merge_average, merge_concatenate, merge_multiply};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::{MAX_FUSED_CHAIN_LEN, ShapeError, Tensor};
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
    cases: Vec<Case>,
    error_cases: Vec<ErrCase>,
}

#[derive(Deserialize)]
struct Input {
    shape: Vec<usize>,
    x_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct Out {
    shape: Vec<usize>,
    bits: Vec<u32>,
    g_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    op: String,
    dim: usize,
    uses: Vec<usize>,
    pre_transpose: Option<Vec<usize>>,
    inputs: Vec<Input>,
    out: Out,
    grads: Vec<Vec<u32>>,
}

#[derive(Deserialize)]
struct ErrCase {
    name: String,
    op: String,
    dim: usize,
    input_shapes: Vec<Vec<usize>>,
    torch_raises: bool,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/merge-ops-pytorch-reference/merge_ops_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

fn apply_op<'t>(op: &str, dim: usize, xs: &[Var<'t>]) -> Result<Var<'t>, AutodiffError> {
    match op {
        "concatenate" => merge_concatenate(xs, dim),
        "add" => merge_add(xs),
        "multiply" => merge_multiply(xs),
        "average" => merge_average(xs),
        other => panic!("未知の op: {other}"),
    }
}

fn assert_close(actual: &[f32], expected: &[f32], context: &str) {
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
    out_values: Vec<f32>,
    grads: Vec<Vec<f32>>,
}

/// `(y * g).sum()` を損失とした forward 値・葉ごとの勾配を返す。
fn run_case(c: &Case) -> Run {
    let tape = Tape::new_with_ops(common::naive_ops());
    let leaves: Vec<Var<'_>> = c
        .inputs
        .iter()
        .map(|i| tape.var(&t(from_bits(&i.x_bits), &i.shape)))
        .collect();
    let mut views: Vec<Var<'_>> = c.uses.iter().map(|&u| leaves[u]).collect();
    if let Some(pt) = &c.pre_transpose {
        views[0] = leaves[c.uses[0]]
            .transpose(pt[0], pt[1])
            .expect("pre_transpose");
    }
    let y = apply_op(&c.op, c.dim, &views).expect("forward");
    let out_shape = y.to_tensor().shape().to_vec();
    let out_values = y.to_tensor().host_slice().into_owned();
    let g = tape.var_no_grad(&t(from_bits(&c.out.g_bits), &c.out.shape));
    let loss = y.mul(&g).expect("mul").sum(None).expect("sum");
    let gs = tape.backward(&loss).expect("backward");
    let grads = leaves
        .iter()
        .map(|l| match gs.get(l).expect("get") {
            Some(g) => g.host_slice().into_owned(),
            None => vec![0.0; l.to_tensor().numel()],
        })
        .collect();
    Run {
        out_shape,
        out_values,
        grads,
    }
}

#[test]
fn fixture_matches_pytorch_2_14_0() {
    let fx = load_fixture();
    assert!(
        fx.torch_version.starts_with("2.14.0"),
        "fixture の torch バージョン: {}",
        fx.torch_version
    );
    assert!(fx.cases.len() >= 80, "fixture ケース数: {}", fx.cases.len());
    for op in ["concatenate", "add", "multiply", "average"] {
        assert!(
            fx.cases.iter().any(|c| c.op == op),
            "fixture に {op} のケースがない"
        );
    }
    for c in &fx.cases {
        let r = run_case(c);
        assert_eq!(r.out_shape, c.out.shape, "{}: 出力 shape", c.name);
        assert_close(
            &r.out_values,
            &from_bits(&c.out.bits),
            &format!("{} forward", c.name),
        );
        for (i, g) in c.grads.iter().enumerate() {
            assert_close(&r.grads[i], &from_bits(g), &format!("{} grad[{i}]", c.name));
        }
    }
}

/// torch と成否が食い違う（torch は成功／本実装はエラー）ケース。決定記録 §17 と一対一。
const INTENDED_DIFFS: [&str; 7] = [
    "add_broadcastable",
    "multiply_broadcastable",
    "average_broadcastable",
    "add_single_input",
    "multiply_single_input",
    "average_single_input",
    "concatenate_single_input",
];

#[test]
fn error_cases_are_typed_errors_and_diffs_are_intended() {
    let fx = load_fixture();
    let mut diffs: Vec<String> = Vec::new();
    for e in &fx.error_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xs: Vec<Var<'_>> = e
            .input_shapes
            .iter()
            .map(|s| tape.var(&t(vec![0.0; s.iter().product()], s)))
            .collect();
        let before = tape.len();
        let r = apply_op(&e.op, e.dim, &xs);
        assert_eq!(tape.len(), before, "{}: エラー時にノードを残さない", e.name);
        match (&r, e.torch_raises) {
            (Err(_), true) => {}
            (Err(_), false) => diffs.push(e.name.clone()),
            (Ok(_), true) => panic!("{}: torch は例外だが本実装は成功した", e.name),
            (Ok(_), false) => panic!("{}: 本実装が torch 受理ケースを成功させた", e.name),
        }
        if let Err(err) = &r {
            let kind_ok = match e.name.as_str() {
                n if n.ends_with("_single_input") => {
                    matches!(err, AutodiffError::InvalidArgument(_))
                }
                n if n.ends_with("_mismatch") || n.ends_with("_broadcastable") => {
                    matches!(err, AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
                        || e.op == "concatenate"
                }
                _ => matches!(err, AutodiffError::Shape(_)),
            };
            assert!(kind_ok, "{}: 想定外のエラー種別 {err:?}", e.name);
        }
    }
    let mut expected: Vec<String> = INTENDED_DIFFS.iter().map(|s| s.to_string()).collect();
    diffs.sort();
    expected.sort();
    assert_eq!(diffs, expected, "意図的な差分の集合が決定記録 §17 と不一致");
}

fn leaf<'t>(tape: &'t Tape, base: f32, shape: &[usize]) -> Var<'t> {
    let n: usize = shape.iter().product();
    let data = (0..n).map(|i| base + i as f32 * 0.25).collect();
    tape.var(&t(data, shape))
}

#[test]
fn rejects_zero_and_one_input_without_touching_tape() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = leaf(&tape, 1.0, &[2, 3]);
    let before = tape.len();
    for op in ["concatenate", "add", "multiply", "average"] {
        assert!(matches!(
            apply_op(op, 0, &[]),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            apply_op(op, 0, &[x]),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }
    assert_eq!(tape.len(), before);
}

#[test]
fn rejects_cross_tape_inputs_without_touching_tape() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let a = leaf(&tape_a, 1.0, &[2, 3]);
    let b = leaf(&tape_b, 2.0, &[2, 3]);
    let before = (tape_a.len(), tape_b.len());
    for op in ["concatenate", "add", "multiply", "average"] {
        assert!(matches!(
            apply_op(op, 0, &[a, b]),
            Err(AutodiffError::TapeMismatch)
        ));
        assert!(matches!(
            apply_op(op, 0, &[b, a]),
            Err(AutodiffError::TapeMismatch)
        ));
    }
    assert_eq!((tape_a.len(), tape_b.len()), before);
}

#[test]
fn rejects_shape_errors_without_touching_tape() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = leaf(&tape, 1.0, &[2, 3]);
    let broadcastable = leaf(&tape, 1.0, &[1, 3]);
    let other = leaf(&tape, 1.0, &[2, 4]);
    let before = tape.len();
    for op in ["add", "multiply", "average"] {
        for rhs in [broadcastable, other] {
            assert!(matches!(
                apply_op(op, 0, &[a, rhs]),
                Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
            ));
        }
        // 3 件目だけが不一致でも最初の演算より前に拒否する。
        assert!(apply_op(op, 0, &[a, a, other]).is_err());
    }
    assert!(merge_concatenate(&[a, a], 2).is_err());
    assert!(merge_concatenate(&[a, other], 0).is_err());
    assert_eq!(tape.len(), before);
}

#[test]
fn tape_record_counts_are_fixed() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xs: Vec<Var<'_>> = (0..4).map(|i| leaf(&tape, i as f32, &[2, 3])).collect();

    let before = tape.len();
    merge_add(&xs).expect("add");
    assert_eq!(tape.len() - before, 3, "Add は n-1 ノード");

    let before = tape.len();
    merge_multiply(&xs[..3]).expect("multiply");
    assert_eq!(tape.len() - before, 2, "Multiply は n-1 ノード");

    let before = tape.len();
    merge_average(&xs[..3]).expect("average");
    assert_eq!(
        tape.len() - before,
        2 + 1 + 1,
        "Average は n-1 + 定数葉 1 + 除算 1"
    );

    let before = tape.len();
    merge_concatenate(&xs, 1).expect("concatenate");
    assert_eq!(tape.len() - before, 1, "Concatenate は 1 ノード");
}

#[test]
fn average_keeps_shape_and_matches_oracle() {
    let tape = Tape::new_with_ops(common::naive_ops());
    for shape in [vec![], vec![3], vec![2, 3], vec![2, 1, 2]] {
        let xs: Vec<Var<'_>> = (0..3).map(|i| leaf(&tape, i as f32, &shape)).collect();
        let y = merge_average(&xs).expect("average");
        assert_eq!(y.to_tensor().shape(), shape.as_slice());
        let n: usize = shape.iter().product();
        let got = y.to_tensor().host_slice().into_owned();
        for (j, v) in got.iter().enumerate().take(n) {
            let expected: f64 = (0..3)
                .map(|i| f64::from(i as f32 + j as f32 * 0.25))
                .sum::<f64>()
                / 3.0;
            assert!(
                common::req2_close(f64::from(*v), expected),
                "{shape:?}[{j}]"
            );
        }
    }
}

#[test]
fn long_chain_beyond_fused_limit_matches_oracle() {
    let n = MAX_FUSED_CHAIN_LEN + 4;
    assert!(n >= 8);
    let tape = Tape::new_with_ops(common::naive_ops());
    let xs: Vec<Var<'_>> = (0..n)
        .map(|i| leaf(&tape, i as f32 + 1.0, &[2, 2]))
        .collect();
    let sum = merge_add(&xs).expect("add");
    let prod = merge_multiply(&xs[..n.min(8)]).expect("multiply");
    let avg = merge_average(&xs).expect("average");
    let s = sum.to_tensor().host_slice().into_owned();
    let a = avg.to_tensor().host_slice().into_owned();
    let p = prod.to_tensor().host_slice().into_owned();
    for j in 0..4 {
        let sum_o: f64 = (0..n)
            .map(|i| f64::from(i as f32 + 1.0 + j as f32 * 0.25))
            .sum();
        let prod_o: f64 = (0..n.min(8))
            .map(|i| f64::from(i as f32 + 1.0 + j as f32 * 0.25))
            .product();
        assert!(common::req2_close(f64::from(s[j]), sum_o));
        assert!(common::req2_close(f64::from(a[j]), sum_o / n as f64));
        assert!(common::req2_close(f64::from(p[j]), prod_o));
    }
    // 勾配も通る（loss = sum(avg)）。各入力の勾配は 1/n。
    let loss = avg.sum(None).expect("sum");
    let gs = tape.backward(&loss).expect("backward");
    for x in &xs {
        let g = gs.get(x).expect("get").expect("grad");
        for v in g.host_slice().iter() {
            assert!(common::req2_close(f64::from(*v), 1.0 / n as f64));
        }
    }
}

#[test]
fn non_contiguous_inputs_are_accepted() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = leaf(&tape, 1.0, &[3, 2])
        .transpose(0, 1)
        .expect("transpose");
    let b = leaf(&tape, 5.0, &[2, 3]);
    for op in ["add", "multiply", "average"] {
        let y = apply_op(op, 0, &[a, b]).expect(op);
        assert_eq!(y.to_tensor().shape(), &[2, 3]);
    }
    let c = merge_concatenate(&[a, b], 1).expect("concatenate");
    assert_eq!(c.to_tensor().shape(), &[2, 6]);
}

#[test]
fn results_are_bit_deterministic_run_to_run() {
    let fx = load_fixture();
    for c in fx
        .cases
        .iter()
        .filter(|c| c.name.contains("n7") || c.name.contains("n12"))
    {
        let r1 = run_case(c);
        let r2 = run_case(c);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&r1.out_values), bits(&r2.out_values), "{}", c.name);
        for (g1, g2) in r1.grads.iter().zip(&r2.grads) {
            assert_eq!(bits(g1), bits(g2), "{} grad", c.name);
        }
    }
}
