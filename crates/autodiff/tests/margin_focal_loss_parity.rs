//! `margin_focal_loss_ops`（イシュー #2653・MultiMargin／MultiLabelMargin／
//! MultiLabelSoftMargin／sigmoid focal loss）の `Tape`／`Var` を経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/margin-focal-loss-pytorch-reference/margin_focal_loss_reference.json`・
//!   生成条件は同ディレクトリの `README.md`）の forward 損失値と入力勾配を、REQ-2 統一複合判定
//!   （`common::req2_close`）で突合する。tolerance 定数は新設しない。`edge_cases` は値クラス
//!   （NaN／+inf／-inf／有限）で突合し、`diverges` 付きは本実装の文書化済みの値を明示的に assert する。
//! - 契約整合: `sigmoid_focal_loss(alpha なし, gamma = 0)` と
//!   `multilabel_soft_margin_loss(weight なし, Mean)` は既存 `Var::bce_with_logits_loss(Mean)` と
//!   REQ-2 一致する。
//!
//! `common::naive_ops()` は `BackendOps` の損失系メソッドを持たないため、全 Op がホスト参照実装で
//! 評価される。CPU `BackendOps` との一致は
//! `crates/facade/tests/margin_focal_loss_ops_backend_parity.rs` が担当する。

mod common;

use std::path::PathBuf;

use fandhe_ai_autodiff::margin_focal_loss_ops::{
    MultiLabelSoftMarginOptions, MultiMarginOptions, SigmoidFocalLossOptions, multi_margin_loss,
    multilabel_margin_loss, multilabel_soft_margin_loss, sigmoid_focal_loss,
};
use fandhe_ai_autodiff::{Reduction, Tape};
use fandhe_ai_tensor_core::Tensor;
use serde::Deserialize;
use serde_json::Value;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

#[derive(Deserialize, Clone)]
struct Params {
    reduction: String,
    p: u8,
    margin: f32,
    has_weight: bool,
    alpha: Option<f32>,
    gamma: f32,
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    cases: Vec<Case>,
    edge_cases: Vec<EdgeCase>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    op: String,
    shape: Vec<usize>,
    params: Params,
    loss: f32,
    x: Vec<f32>,
    target: Vec<f32>,
    target_shape: Vec<usize>,
    weight: Option<Vec<f32>>,
    grad_x: Vec<f32>,
}

#[derive(Deserialize, Clone)]
struct Klass {
    class: String,
    value: Option<f32>,
}

#[derive(Deserialize)]
struct EdgeCase {
    name: String,
    op: String,
    shape: Vec<usize>,
    params: Params,
    loss: Klass,
    x: Vec<Klass>,
    /// 添字 target は整数、ラベル target は `Klass`。
    target: Vec<Value>,
    weight: Option<Vec<f32>>,
    grad_x: Vec<Klass>,
    #[serde(default)]
    diverges: Option<String>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/margin-focal-loss-pytorch-reference/margin_focal_loss_reference.json",
    );
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture を読めない: {}: {e}", path.display()));
    serde_json::from_str(&text).expect("fixture の JSON が不正")
}

fn klass_to_f32(k: &Klass) -> f32 {
    match k.class.as_str() {
        "nan" => f32::NAN,
        "pos_inf" => f32::INFINITY,
        "neg_inf" => f32::NEG_INFINITY,
        "finite" => k.value.expect("finite には value がある"),
        other => panic!("未知の class: {other}"),
    }
}

fn klasses(ks: &[Klass]) -> Vec<f32> {
    ks.iter().map(klass_to_f32).collect()
}

fn reduction_of(p: &Params) -> Reduction {
    match p.reduction.as_str() {
        "mean" => Reduction::Mean,
        "sum" => Reduction::Sum,
        other => panic!("未知の reduction: {other}"),
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

fn assert_class(actual: f32, expected: &Klass, context: &str) {
    match expected.class.as_str() {
        "nan" => assert!(actual.is_nan(), "{context}: NaN を期待, actual={actual}"),
        "pos_inf" => assert_eq!(actual, f32::INFINITY, "{context}"),
        "neg_inf" => assert_eq!(actual, f32::NEG_INFINITY, "{context}"),
        "finite" => {
            let e = expected.value.expect("finite には value がある");
            assert!(
                common::req2_close(f64::from(actual), f64::from(e)),
                "{context}: actual={actual} expected={e}"
            );
        }
        other => panic!("未知の class: {other}"),
    }
}

fn assert_classes(actual: &[f32], expected: &[Klass], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, e)) in actual.iter().zip(expected).enumerate() {
        assert_class(a, e, &format!("{context}[{i}]"));
    }
}

/// 1 ケースの実行結果（損失値と `input` の勾配）。
struct Run {
    loss: f32,
    grad_x: Vec<f32>,
}

/// `target` は添字系 op では整数値の `f32` 列、ラベル系 op ではラベルそのもの。
fn run(
    op: &str,
    p: &Params,
    shape: &[usize],
    x: &[f32],
    target: &[f32],
    target_shape: &[usize],
    weight: Option<&[f32]>,
) -> Run {
    let tape = Tape::new_with_ops(common::naive_ops());
    let reduction = reduction_of(p);
    let xv = tape.var(&t(x.to_vec(), shape));
    let idx = || ti(target.iter().map(|&v| v as i32).collect(), target_shape);
    let w = weight.map(|w| t(w.to_vec(), &[w.len()]));
    let loss = match op {
        "multi_margin" => {
            let mut o = MultiMarginOptions::default().p(p.p).margin(p.margin);
            if let Some(w) = w {
                o = o.weight(w);
            }
            multi_margin_loss(&xv, &idx(), &o, reduction)
        }
        "multilabel_margin" => multilabel_margin_loss(&xv, &idx(), reduction),
        "multilabel_soft_margin" => {
            let mut o = MultiLabelSoftMarginOptions::default();
            if let Some(w) = w {
                o = o.weight(w);
            }
            multilabel_soft_margin_loss(&xv, &t(target.to_vec(), target_shape), &o, reduction)
        }
        "sigmoid_focal" => {
            let o = SigmoidFocalLossOptions::default()
                .alpha(p.alpha)
                .gamma(p.gamma);
            sigmoid_focal_loss(&xv, &t(target.to_vec(), target_shape), &o, reduction)
        }
        other => panic!("未知の op: {other}"),
    }
    .unwrap_or_else(|e| panic!("{op}: forward 失敗: {e}"));
    let grads = tape.backward(&loss).unwrap();
    Run {
        loss: loss.to_tensor().host_slice()[0],
        grad_x: grads
            .get(&xv)
            .expect("勾配取得")
            .expect("入力へ勾配が届く")
            .host_slice()
            .into_owned(),
    }
}

#[test]
fn matches_pytorch_reference_forward_and_backward() {
    let fixture = load_fixture();
    assert!(
        fixture.torch_version.starts_with("2.14.0"),
        "fixture は PyTorch 2.14.0 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    for op in [
        "multi_margin",
        "multilabel_margin",
        "multilabel_soft_margin",
        "sigmoid_focal",
    ] {
        assert!(
            fixture.cases.iter().any(|c| c.op == op),
            "fixture に {op} のケースが無い"
        );
    }
    for c in &fixture.cases {
        assert_eq!(c.params.has_weight, c.weight.is_some(), "{}", c.name);
        let r = run(
            &c.op,
            &c.params,
            &c.shape,
            &c.x,
            &c.target,
            &c.target_shape,
            c.weight.as_deref(),
        );
        assert_close_all(&[r.loss], &[c.loss], &format!("{} loss", c.name));
        assert_close_all(&r.grad_x, &c.grad_x, &format!("{} grad_x", c.name));
    }
}

fn edge_target(e: &EdgeCase) -> Vec<f32> {
    e.target
        .iter()
        .map(|v| match v {
            Value::Number(n) => n.as_f64().expect("数値") as f32,
            obj => klass_to_f32(&serde_json::from_value(obj.clone()).expect("Klass")),
        })
        .collect()
}

#[test]
fn edge_cases_match_pytorch_value_class() {
    let fixture = load_fixture();
    assert_eq!(fixture.edge_cases.len(), 23, "edge_cases の件数");
    for e in &fixture.edge_cases {
        let x = klasses(&e.x);
        let target = edge_target(e);
        // 添字 target の shape: multi_margin は rank 1 入力で 0 次元・それ以外は [N]。
        let target_shape: Vec<usize> = match e.op.as_str() {
            "multi_margin" => {
                if e.shape.len() == 1 {
                    vec![]
                } else {
                    vec![e.shape[0]]
                }
            }
            _ => e.shape.clone(),
        };
        let r = run(
            &e.op,
            &e.params,
            &e.shape,
            &x,
            &target,
            &target_shape,
            e.weight.as_deref(),
        );
        if let Some(reason) = &e.diverges {
            // 文書化済みの差分（`docs/autodiff-margin-focal-loss-ops-decision.md` §5）:
            // PyTorch は f32 評価で飽和域の勾配が NaN になる。本実装は有限値（真の勾配 0 近傍）を返す。
            assert!(!reason.is_empty());
            assert_eq!(e.name, "focal_saturated_g0.5");
            assert_class(r.loss, &e.loss, &format!("{} loss", e.name));
            assert!(
                r.grad_x.iter().all(|g| g.is_finite()),
                "{}: 本実装は有限勾配",
                e.name
            );
            // 飽和していない要素（x=±40 の誤分類側）は fixture と一致する。
            assert_class(r.grad_x[0], &e.grad_x[0], "focal_saturated_g0.5 grad_x[0]");
            assert_class(r.grad_x[3], &e.grad_x[3], "focal_saturated_g0.5 grad_x[3]");
            // PyTorch が NaN にする要素の真の勾配は 0。
            for i in [1usize, 2, 4, 5] {
                assert!(r.grad_x[i].abs() < 1e-5, "grad_x[{i}] = {}", r.grad_x[i]);
            }
            continue;
        }
        assert_class(r.loss, &e.loss, &format!("{} loss", e.name));
        assert_classes(&r.grad_x, &e.grad_x, &format!("{} grad_x", e.name));
    }
}

/// `sigmoid_focal_loss(alpha なし, gamma = 0)` と `multilabel_soft_margin_loss(weight なし, Mean)` は
/// 既存 `Var::bce_with_logits_loss(Mean)` と REQ-2 一致する。
#[test]
fn bce_contract_matches_existing_bce_with_logits() {
    let xs = vec![0.3f32, -1.2, 2.5, 0.0, -0.4, 1.1];
    let ys = vec![1.0f32, 0.0, 0.7, 1.0, 0.2, 0.0];
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(xs.clone(), &[2, 3]));
    let y = tape.var(&t(ys.clone(), &[2, 3]));
    let existing = x.bce_with_logits_loss(&y, Reduction::Mean).unwrap();
    let focal = sigmoid_focal_loss(
        &x,
        &t(ys.clone(), &[2, 3]),
        &SigmoidFocalLossOptions::default().alpha(None).gamma(0.0),
        Reduction::Mean,
    )
    .unwrap();
    let mlsm = multilabel_soft_margin_loss(
        &x,
        &t(ys, &[2, 3]),
        &MultiLabelSoftMarginOptions::default(),
        Reduction::Mean,
    )
    .unwrap();
    let g_existing = tape.backward(&existing).unwrap();
    let want_loss = existing.to_tensor().host_slice()[0];
    let want_grad = g_existing
        .get(&x)
        .unwrap()
        .unwrap()
        .host_slice()
        .into_owned();
    for (name, v) in [("focal", &focal), ("mlsm", &mlsm)] {
        assert!(
            common::req2_close(
                f64::from(v.to_tensor().host_slice()[0]),
                f64::from(want_loss)
            ),
            "{name} loss"
        );
        let g = tape.backward(v).unwrap();
        let got = g.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_close_all(&got, &want_grad, &format!("{name} grad_x vs existing"));
    }
}
