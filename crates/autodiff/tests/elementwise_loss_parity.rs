//! `elementwise_loss_ops`（イシュー #2652・pos_weight 付き BCEWithLogits／
//! HingeEmbedding／SoftMargin／GaussianNLL）の `Tape`／`Var` を経由する
//! end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/elementwise-loss-pytorch-reference/elementwise_loss_reference.json`・
//!   生成条件は同ディレクトリの `README.md`）の forward 損失値と全入力勾配を、REQ-2 統一複合判定
//!   （`common::req2_close`）で突合する。tolerance 定数は新設しない。`edge_cases` は値クラス
//!   （NaN／+inf／-inf／有限）で突合し、`diverges` 付きは本実装の文書化済みの値を明示的に assert する。
//! - 契約整合: `pos_weight` なしは既存 `Var::bce_with_logits_loss` へ委譲（bit 一致・`Op::BceLoss`）、
//!   `pos_weight = ones` の新規 Op 経路は既存経路と REQ-2 一致。
//!
//! `common::naive_ops()` は `BackendOps` の損失系メソッドを持たないため、委譲経路も
//! ホスト参照実装へフォールバックする。CPU `BackendOps` との一致は
//! `crates/facade/tests/elementwise_loss_ops_backend_parity.rs` が担当する。

mod common;

use std::path::PathBuf;

use fandhe_ai_autodiff::elementwise_loss_ops::{
    BceWithLogitsOptions, GaussianNllOptions, bce_with_logits_loss_with, gaussian_nll_loss,
    hinge_embedding_loss, soft_margin_loss,
};
use fandhe_ai_autodiff::{Reduction, Tape, Var};
use fandhe_ai_tensor_core::Tensor;
use serde::Deserialize;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

#[derive(Deserialize, Default, Clone)]
struct Params {
    reduction: String,
    #[serde(default)]
    margin: f32,
    #[serde(default)]
    full: bool,
    #[serde(default)]
    eps: f32,
    #[serde(default)]
    pos_weight_shape: Vec<usize>,
    #[serde(default)]
    has_pos_weight: bool,
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
    #[serde(default)]
    y: Vec<f32>,
    #[serde(default)]
    t: Vec<f32>,
    #[serde(default)]
    var: Vec<f32>,
    #[serde(default)]
    pos_weight: Vec<f32>,
    #[serde(default)]
    grad_x: Vec<f32>,
    #[serde(default)]
    grad_y: Vec<f32>,
    #[serde(default)]
    grad_t: Vec<f32>,
    #[serde(default)]
    grad_var: Vec<f32>,
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
    params: Params,
    loss: Klass,
    x: Vec<Klass>,
    #[serde(default)]
    y: Vec<Klass>,
    #[serde(default)]
    t: Vec<Klass>,
    #[serde(default)]
    var: Vec<Klass>,
    #[serde(default)]
    pos_weight: Vec<Klass>,
    #[serde(default)]
    grad_x: Vec<Klass>,
    #[serde(default)]
    grad_y: Vec<Klass>,
    #[serde(default)]
    grad_t: Vec<Klass>,
    #[serde(default)]
    grad_var: Vec<Klass>,
    #[serde(default)]
    diverges: Option<String>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/elementwise-loss-pytorch-reference/elementwise_loss_reference.json");
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

/// 1 ケースの実行結果（損失値と、追跡した各入力の勾配）。
struct Run {
    loss: f32,
    grad_x: Vec<f32>,
    grad_y: Vec<f32>,
    grad_t: Vec<f32>,
    grad_var: Vec<f32>,
}

fn grad_of(tape_grads: &fandhe_ai_autodiff::Gradients, v: &Var<'_>) -> Vec<f32> {
    tape_grads
        .get(v)
        .expect("勾配取得")
        .expect("入力へ勾配が届く")
        .host_slice()
        .into_owned()
}

#[allow(clippy::too_many_arguments)]
fn run(
    op: &str,
    p: &Params,
    shape: &[usize],
    x: &[f32],
    y: &[f32],
    tt: &[f32],
    var: &[f32],
    pos_weight: &[f32],
) -> Run {
    let tape = Tape::new_with_ops(common::naive_ops());
    let reduction = reduction_of(p);
    let xv = tape.var(&t(x.to_vec(), shape));
    match op {
        "bce_pos_weight" => {
            let yv = tape.var(&t(y.to_vec(), shape));
            let options = if p.has_pos_weight {
                BceWithLogitsOptions::default()
                    .pos_weight(t(pos_weight.to_vec(), &p.pos_weight_shape))
            } else {
                BceWithLogitsOptions::default()
            };
            let loss = bce_with_logits_loss_with(&xv, &yv, reduction, &options)
                .unwrap_or_else(|e| panic!("{op}: forward 失敗: {e}"));
            let grads = tape.backward(&loss).unwrap();
            Run {
                loss: loss.to_tensor().host_slice()[0],
                grad_x: grad_of(&grads, &xv),
                grad_y: grad_of(&grads, &yv),
                grad_t: vec![],
                grad_var: vec![],
            }
        }
        "hinge_embedding" => {
            let loss = hinge_embedding_loss(&xv, &t(y.to_vec(), shape), p.margin, reduction)
                .unwrap_or_else(|e| panic!("{op}: forward 失敗: {e}"));
            let grads = tape.backward(&loss).unwrap();
            Run {
                loss: loss.to_tensor().host_slice()[0],
                grad_x: grad_of(&grads, &xv),
                grad_y: vec![],
                grad_t: vec![],
                grad_var: vec![],
            }
        }
        "soft_margin" => {
            let loss = soft_margin_loss(&xv, &t(y.to_vec(), shape), reduction)
                .unwrap_or_else(|e| panic!("{op}: forward 失敗: {e}"));
            let grads = tape.backward(&loss).unwrap();
            Run {
                loss: loss.to_tensor().host_slice()[0],
                grad_x: grad_of(&grads, &xv),
                grad_y: vec![],
                grad_t: vec![],
                grad_var: vec![],
            }
        }
        "gaussian_nll" => {
            let tv = tape.var(&t(tt.to_vec(), shape));
            let vv = tape.var(&t(var.to_vec(), shape));
            let options = GaussianNllOptions::default().full(p.full).eps(p.eps);
            let loss = gaussian_nll_loss(&xv, &tv, &vv, &options, reduction)
                .unwrap_or_else(|e| panic!("{op}: forward 失敗: {e}"));
            let grads = tape.backward(&loss).unwrap();
            Run {
                loss: loss.to_tensor().host_slice()[0],
                grad_x: grad_of(&grads, &xv),
                grad_y: vec![],
                grad_t: grad_of(&grads, &tv),
                grad_var: grad_of(&grads, &vv),
            }
        }
        other => panic!("未知の op: {other}"),
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
        "bce_pos_weight",
        "hinge_embedding",
        "soft_margin",
        "gaussian_nll",
    ] {
        assert!(
            fixture.cases.iter().any(|c| c.op == op),
            "fixture に {op} のケースが無い"
        );
    }
    for c in &fixture.cases {
        let r = run(
            &c.op,
            &c.params,
            &c.shape,
            &c.x,
            &c.y,
            &c.t,
            &c.var,
            &c.pos_weight,
        );
        assert_close_all(&[r.loss], &[c.loss], &format!("{} loss", c.name));
        assert_close_all(&r.grad_x, &c.grad_x, &format!("{} grad_x", c.name));
        assert_close_all(&r.grad_y, &c.grad_y, &format!("{} grad_y", c.name));
        assert_close_all(&r.grad_t, &c.grad_t, &format!("{} grad_t", c.name));
        assert_close_all(&r.grad_var, &c.grad_var, &format!("{} grad_var", c.name));
    }
}

#[test]
fn edge_cases_match_pytorch_value_class() {
    let fixture = load_fixture();
    assert_eq!(fixture.edge_cases.len(), 11, "edge_cases の件数");
    for e in &fixture.edge_cases {
        let x = klasses(&e.x);
        let n = x.len();
        let p = e.params.clone();
        let r = run(
            &e.op,
            &p,
            &[n],
            &x,
            &klasses(&e.y),
            &klasses(&e.t),
            &klasses(&e.var),
            &klasses(&e.pos_weight),
        );
        if let Some(reason) = &e.diverges {
            // 文書化済みの差分（`docs/autodiff-elementwise-loss-ops-decision.md` §5）:
            // PyTorch は inf／NaN 勾配、本実装は安定形で有限値を返す。
            assert!(!reason.is_empty());
            assert_eq!(e.name, "soft_margin_large");
            assert!(r.loss.is_finite(), "{}: 安定形は有限損失", e.name);
            assert!(
                r.grad_x.iter().all(|g| g.is_finite()),
                "{}: 安定形は有限勾配",
                e.name
            );
            // `y = -1`・`x = 200` は `z = 200`（損失 ≈ 200・勾配 `-y·σ(z)·` ではなく `+1`）、
            // `y = 1`・`x = -200` は `z = 200`（勾配 `-1`）、`y = 1`・`x = 5` は fixture と一致。
            assert!((r.loss - (400.0 + (-5.0f32).exp().ln_1p())).abs() < 1e-3);
            assert!((r.grad_x[0] - 1.0).abs() < 1e-6);
            assert!((r.grad_x[1] + 1.0).abs() < 1e-6);
            assert_class(r.grad_x[2], &e.grad_x[2], "soft_margin_large grad_x[2]");
            continue;
        }
        assert_class(r.loss, &e.loss, &format!("{} loss", e.name));
        assert_classes(&r.grad_x, &e.grad_x, &format!("{} grad_x", e.name));
        assert_classes(&r.grad_y, &e.grad_y, &format!("{} grad_y", e.name));
        assert_classes(&r.grad_t, &e.grad_t, &format!("{} grad_t", e.name));
        assert_classes(&r.grad_var, &e.grad_var, &format!("{} grad_var", e.name));
    }
}

/// `pos_weight` なしは既存 `Var::bce_with_logits_loss` へ丸ごと委譲する（bit 一致）。
/// `pos_weight = ones` の新規 Op 経路は既存経路と REQ-2 一致する。
#[test]
fn bce_contract_matches_existing_bce_with_logits() {
    let xs = vec![0.3f32, -1.2, 2.5, 0.0, -0.4, 1.1];
    let ys = vec![1.0f32, 0.0, 0.7, 1.0, 0.2, 0.0];
    for reduction in [Reduction::Mean, Reduction::Sum] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(xs.clone(), &[2, 3]));
        let y = tape.var(&t(ys.clone(), &[2, 3]));
        let existing = x.bce_with_logits_loss(&y, reduction).unwrap();
        let delegated =
            bce_with_logits_loss_with(&x, &y, reduction, &BceWithLogitsOptions::default()).unwrap();
        assert_eq!(
            existing.to_tensor().host_slice()[0].to_bits(),
            delegated.to_tensor().host_slice()[0].to_bits(),
            "委譲経路は既存経路と bit 一致"
        );
        let ones = BceWithLogitsOptions::default().pos_weight(t(vec![1.0], &[1]));
        let new_op = bce_with_logits_loss_with(&x, &y, reduction, &ones).unwrap();
        assert!(common::req2_close(
            f64::from(new_op.to_tensor().host_slice()[0]),
            f64::from(existing.to_tensor().host_slice()[0])
        ));
        let g_existing = tape.backward(&existing).unwrap();
        let g_new = tape.backward(&new_op).unwrap();
        let (gx_e, gx_n) = (grad_of(&g_existing, &x), grad_of(&g_new, &x));
        let (gy_e, gy_n) = (grad_of(&g_existing, &y), grad_of(&g_new, &y));
        assert_close_all(&gx_n, &gx_e, "grad_x ones vs existing");
        assert_close_all(&gy_n, &gy_e, "grad_y ones vs existing");
    }
}
