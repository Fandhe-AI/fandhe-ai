//! `Var::cosine_similarity`／`Var::pairwise_distance`（イシュー #3821）の実 PyTorch 2.14.0 参照値突合。
//!
//! fixture（`tests/fixtures/similarity-distance-pytorch-reference/similarity_distance_reference.json`。
//! 生成条件は同ディレクトリの `README.md`）の forward 出力と入力勾配を REQ-2 統一複合判定
//! （`common::req2_close`）で突合する。tolerance 定数は新設しない。`clamp_cases` は
//! `0 < ‖x‖ < eps` とゼロノルムで、PyTorch の straight-through 勾配と一致することを確かめる
//! （設計記録 `docs/autodiff-similarity-distance-decision.md` §5）。
//!
//! 入力は fixture の値をそのまま使う（Rust 側で再生成しない）。fixture が空・版違いなら失敗する
//! （ファイルが無いことで黙って通らない）。

mod common;

use std::path::PathBuf;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_tensor_core::Tensor;
use serde::Deserialize;

#[derive(Deserialize, Default, Clone)]
struct Params {
    #[serde(default)]
    dim: usize,
    #[serde(default)]
    p: f32,
    #[serde(default)]
    eps: f32,
    #[serde(default)]
    keepdim: bool,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    op: String,
    params: Params,
    shape: Vec<usize>,
    out_shape: Vec<usize>,
    x1: Vec<f32>,
    x2: Vec<f32>,
    g: Vec<f32>,
    out: Vec<f32>,
    grad_x1: Vec<f32>,
    grad_x2: Vec<f32>,
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    cases: Vec<Case>,
    clamp_cases: Vec<Case>,
}

fn load() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/similarity-distance-pytorch-reference/similarity_distance_reference.json",
    );
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture を読めない: {}: {e}", path.display()));
    serde_json::from_str(&text).expect("fixture の JSON が不正")
}

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data.to_vec(), shape).expect("fixture: shape とデータ長が一致")
}

fn check(c: &Case) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(&c.x1, &c.shape));
    let b = tape.var(&t(&c.x2, &c.shape));
    let y = match c.op.as_str() {
        "cosine_similarity" => a.cosine_similarity(&b, c.params.dim, c.params.eps),
        "pairwise_distance" => a.pairwise_distance(&b, c.params.p, c.params.eps, c.params.keepdim),
        other => panic!("未知の op: {other}"),
    }
    .unwrap_or_else(|e| panic!("{}: forward 失敗: {e:?}", c.name));
    let y_shape = y.value().shape().to_vec();
    assert_eq!(y_shape, c.out_shape, "{}: 出力 shape", c.name);
    let out: Vec<f64> = c.out.iter().map(|&v| v as f64).collect();
    common::assert_all_req2_close(&y.value().host_slice(), &out, &format!("{} out", c.name));
    let w = tape.var_no_grad(&t(&c.g, &y_shape));
    let loss = y.mul(&w).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    for (var, want, label) in [(&a, &c.grad_x1, "grad_x1"), (&b, &c.grad_x2, "grad_x2")] {
        let want: Vec<f64> = want.iter().map(|&v| v as f64).collect();
        let got = grads.get(var).unwrap().expect("勾配が到達する");
        common::assert_all_req2_close(&got.host_slice(), &want, &format!("{} {label}", c.name));
    }
}

#[test]
fn matches_pytorch_reference_cases() {
    let f = load();
    assert!(f.torch_version.starts_with("2.14.0"), "{}", f.torch_version);
    assert_eq!(
        f.cases.len(),
        25,
        "ケース数（cosine 6 + pairwise 18 + eps0 1）"
    );
    for c in &f.cases {
        check(c);
    }
}

#[test]
fn matches_pytorch_straight_through_gradient_in_clamp_region() {
    let f = load();
    assert_eq!(f.clamp_cases.len(), 2);
    for c in &f.clamp_cases {
        check(c);
    }
}
