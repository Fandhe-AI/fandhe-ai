//! `jacobian_ops::{jacobian, hessian}`（イシュー #2670・親 #2668）の統合テスト。
//! 契約の正は `docs/autodiff-jacobian-hessian-gradcheck-decision.md` §3.2・§3.3・§3.6。
//!
//! - J1／J2: 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/jacobian-hessian-pytorch-reference/jacobian_hessian_reference.json`。
//!   生成条件は同ディレクトリの `README.md`）と REQ-2 統一複合判定
//!   （`common::req2_close`。tolerance 定数は新設しない）で全要素突合する。
//! - J3: 閉形式（二次形式の Hessian `A + Aᵀ`・`exp(x)·x` の平均）と、`Tape::backward` の勾配
//!   （スカラー出力の jacobian はその勾配と bit 一致）・対称性との整合。
//! - J4: fail-closed（テープ不一致・非追跡入力・非スカラー loss・非空の子テープ・非対象 Op・
//!   依存しない出力・要素数 0・既存ノード値と backward 結果の不変）。
//!
//! CPU `BackendOps` 実装との一致・CUDA／Metal（J5）は
//! `crates/facade/tests/jacobian_hessian_backend_parity.rs` が担当する。

mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use fandhe_ai_autodiff::jacobian_ops::{hessian, jacobian};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;
use serde::Deserialize;

#[derive(Deserialize)]
struct Packed {
    shape: Vec<usize>,
    bits: Vec<u32>,
}

impl Packed {
    fn values(&self) -> Vec<f32> {
        self.bits.iter().map(|&b| f32::from_bits(b)).collect()
    }
    fn tensor(&self) -> Tensor<f32> {
        Tensor::new(self.values(), &self.shape).expect("fixture: shape とデータ長は一致している")
    }
}

#[derive(Deserialize)]
struct Case {
    name: String,
    x: Packed,
    consts: HashMap<String, Packed>,
    out: Packed,
    expected: Packed,
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    jacobian_cases: Vec<Case>,
    hessian_cases: Vec<Case>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/jacobian-hessian-pytorch-reference/jacobian_hessian_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test: shape とデータ長は一致させている")
}

fn host(x: &Tensor<f32>) -> Vec<f32> {
    x.host_slice().into_owned()
}

fn assert_close_all(actual: &Tensor<f32>, expected: &Packed, ctx: &str) {
    assert_eq!(actual.shape(), expected.shape.as_slice(), "{ctx}: shape");
    let a = host(actual);
    let e = expected.values();
    assert_eq!(a.len(), e.len(), "{ctx}: 要素数");
    for (i, (&x, &y)) in a.iter().zip(e.iter()).enumerate() {
        assert!(
            common::req2_close(x as f64, y as f64),
            "{ctx}[{i}]: actual={x} expected={y}"
        );
    }
}

fn consts_on<'t>(tape: &'t Tape, case: &Case) -> HashMap<String, Var<'t>> {
    case.consts
        .iter()
        .map(|(k, v)| (k.clone(), tape.var_no_grad(&v.tensor())))
        .collect()
}

/// fixture 側 `gen_reference.py` の jacobian プログラムと同名・同式。
fn build_jacobian_program<'t>(
    name: &str,
    x: &Var<'t>,
    c: &HashMap<String, Var<'t>>,
) -> Result<Var<'t>, AutodiffError> {
    match name {
        "vec_elementwise" => x.tanh().mul(x)?.add(&x.exp()),
        "scalar_sum" => x.sigmoid().mul(x)?.sum(None),
        "matmul_tanh" => Ok(x.matmul(&c["w"])?.tanh()),
        "transpose_out" => x.transpose(0, 1),
        "broadcast_out" => x.broadcast_to(&[2, 3]),
        "independent_rows" => Var::cat(&[x.mul(x)?, c["k"]], 0),
        "mean_dim" => x.exp().mean(Some(1)),
        "mlp" => {
            let h = x.matmul(&c["w1"])?.add(&c["b1"])?.relu();
            h.matmul(&c["w2"])
        }
        "scalar_in" => x.exp().mul(x),
        other => panic!("未知の jacobian プログラム: {other}"),
    }
}

/// fixture 側 `gen_reference.py` の hessian プログラムと同名・同式。
fn build_hessian_program<'t>(
    name: &str,
    x: &Var<'t>,
    c: &HashMap<String, Var<'t>>,
) -> Result<Var<'t>, AutodiffError> {
    match name {
        "quadratic" => x.mul(&c["a"].matmul(x)?)?.sum(None),
        "linear" => c["k"].mul(x)?.sum(None),
        "tanh_sum" => x.tanh().mul(x)?.sum(None),
        "exp_mean" => x.exp().mul(x)?.mean(None),
        "mlp" => {
            let h = x.matmul(&c["w1"])?.add(&c["b1"])?.tanh();
            h.matmul(&c["w2"])?.sigmoid().sum(None)
        }
        "scalar_in" => x.exp().mul(x),
        "loss_1x1" => x.mul(x)?.sum(None)?.reshape(&[1, 1]),
        "cat_transpose" => {
            let z = Var::cat(&[x.sigmoid(), x.tanh()], 1)?.transpose(0, 1)?;
            z.mul(&z)?.sum(None)
        }
        "relu_cubic" => {
            let r = x.relu();
            r.mul(&r)?.mul(x)?.sum(None)
        }
        other => panic!("未知の hessian プログラム: {other}"),
    }
}

fn new_tape() -> Tape {
    Tape::new_with_ops(common::naive_ops())
}

// =====================================================================
// J1: jacobian × PyTorch fixture
// =====================================================================

#[test]
fn j1_jacobian_matches_pytorch_fixture() {
    let fx = load_fixture();
    assert!(
        fx.torch_version.starts_with("2.14.0"),
        "{}",
        fx.torch_version
    );
    assert!(!fx.jacobian_cases.is_empty());
    for case in &fx.jacobian_cases {
        let tape = new_tape();
        let x = tape.var(&case.x.tensor());
        let c = consts_on(&tape, case);
        let y = build_jacobian_program(&case.name, &x, &c).expect("program");
        // forward 値の突合（プログラムが PyTorch 側と同じ関数であることの確認）。
        assert_close_all(&y.to_tensor(), &case.out, &format!("{} forward", case.name));
        let jac = jacobian(&tape, &y, &x).unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
        assert_close_all(&jac, &case.expected, &format!("jacobian {}", case.name));
    }
}

// =====================================================================
// J2: hessian × PyTorch fixture
// =====================================================================

#[test]
fn j2_hessian_matches_pytorch_fixture() {
    let fx = load_fixture();
    assert!(!fx.hessian_cases.is_empty());
    for case in &fx.hessian_cases {
        let tape = new_tape();
        let child = new_tape();
        let x = tape.var(&case.x.tensor());
        let c = consts_on(&tape, case);
        let loss = build_hessian_program(&case.name, &x, &c).expect("program");
        assert_close_all(
            &loss.to_tensor(),
            &case.out,
            &format!("{} forward", case.name),
        );
        let h =
            hessian(&tape, &loss, &x, &child).unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
        assert_close_all(&h, &case.expected, &format!("hessian {}", case.name));
    }
}

// =====================================================================
// J3: 閉形式・backward との整合・対称性
// =====================================================================

#[test]
fn j3_scalar_output_jacobian_is_bit_equal_to_backward_gradient() {
    let fx = load_fixture();
    for case in fx.jacobian_cases.iter().filter(|c| c.out.shape.is_empty()) {
        let tape = new_tape();
        let x = tape.var(&case.x.tensor());
        let c = consts_on(&tape, case);
        let y = build_jacobian_program(&case.name, &x, &c).unwrap();
        let g = tape.backward(&y).unwrap();
        let g = g.get(&x).unwrap().expect("到達する");
        let jac = jacobian(&tape, &y, &x).unwrap();
        let (a, b) = (host(&jac), host(g));
        assert_eq!(a.len(), b.len(), "{}", case.name);
        for (u, v) in a.iter().zip(b.iter()) {
            assert_eq!(u.to_bits(), v.to_bits(), "{}", case.name);
        }
    }
}

#[test]
fn j3_quadratic_form_hessian_equals_a_plus_a_transpose() {
    let fx = load_fixture();
    let case = fx
        .hessian_cases
        .iter()
        .find(|c| c.name == "quadratic")
        .unwrap();
    let a = case.consts["a"].values();
    let tape = new_tape();
    let child = new_tape();
    let x = tape.var(&case.x.tensor());
    let c = consts_on(&tape, case);
    let loss = build_hessian_program("quadratic", &x, &c).unwrap();
    let h = hessian(&tape, &loss, &x, &child).unwrap();
    assert_eq!(h.shape(), &[3, 1, 3, 1]);
    let hv = host(&h);
    for i in 0..3 {
        for j in 0..3 {
            let expect = (a[i * 3 + j] + a[j * 3 + i]) as f64;
            assert!(
                common::req2_close(hv[i * 3 + j] as f64, expect),
                "H[{i},{j}]={} expected {expect}",
                hv[i * 3 + j]
            );
        }
    }
}

#[test]
fn j3_exp_mean_hessian_is_diagonal_closed_form() {
    let fx = load_fixture();
    let case = fx
        .hessian_cases
        .iter()
        .find(|c| c.name == "exp_mean")
        .unwrap();
    let xs = case.x.values();
    let tape = new_tape();
    let child = new_tape();
    let x = tape.var(&case.x.tensor());
    let c = consts_on(&tape, case);
    let loss = build_hessian_program("exp_mean", &x, &c).unwrap();
    let hv = host(&hessian(&tape, &loss, &x, &child).unwrap());
    for i in 0..3 {
        for j in 0..3 {
            let expect = if i == j {
                let v = xs[i] as f64;
                v.exp() * (2.0 + v) / 3.0
            } else {
                0.0
            };
            assert!(
                common::req2_close(hv[i * 3 + j] as f64, expect),
                "H[{i},{j}]={} expected {expect}",
                hv[i * 3 + j]
            );
        }
    }
}

#[test]
fn j3_hessian_is_symmetric_for_all_fixture_cases() {
    let fx = load_fixture();
    for case in &fx.hessian_cases {
        let tape = new_tape();
        let child = new_tape();
        let x = tape.var(&case.x.tensor());
        let c = consts_on(&tape, case);
        let loss = build_hessian_program(&case.name, &x, &c).unwrap();
        let h = hessian(&tape, &loss, &x, &child).unwrap();
        let n: usize = case.x.shape.iter().product();
        let hv = host(&h);
        assert_eq!(hv.len(), n * n, "{}", case.name);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    common::req2_close(hv[i * n + j] as f64, hv[j * n + i] as f64),
                    "{}: H[{i},{j}] と H[{j},{i}] が非対称",
                    case.name
                );
            }
        }
    }
}

// =====================================================================
// J4: fail-closed
// =====================================================================

#[test]
fn j4_untracked_input_is_rejected_before_touching_the_tape() {
    let tape = new_tape();
    let child = new_tape();
    let x = tape.var_no_grad(&t(vec![1.0, 2.0], &[2]));
    let w = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = w.mul(&w).unwrap();
    let before = tape.len();
    assert!(matches!(
        jacobian(&tape, &y, &x),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    let loss = y.sum(None).unwrap();
    let before_h = tape.len();
    assert!(matches!(
        hessian(&tape, &loss, &x, &child),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    assert_eq!(tape.len(), before_h);
    assert_eq!(before_h, before + 1, "検査失敗時に補助ノードを足さない");
    assert!(child.is_empty());
}

#[test]
fn j4_cross_tape_vars_are_rejected() {
    let tape = new_tape();
    let other = new_tape();
    let child = new_tape();
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = x.mul(&x).unwrap();
    let ox = other.var(&t(vec![1.0, 2.0], &[2]));
    let oy = ox.mul(&ox).unwrap();
    assert!(matches!(
        jacobian(&tape, &oy, &x),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        jacobian(&tape, &y, &ox),
        Err(AutodiffError::TapeMismatch)
    ));
    let loss = y.sum(None).unwrap();
    assert!(matches!(
        hessian(&tape, &loss, &ox, &child),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(child.is_empty());
}

#[test]
fn j4_jacobian_works_after_tape_reset() {
    let mut tape = new_tape();
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = x.mul(&x).unwrap();
    // `Var` は `&Tape` を借用するため reset 後の世代違いは同一世代の別ノード名義でしか
    // 作れない。ここでは reset 後に新しい葉を作って旧世代の NodeId と混ざらないことのみ確認する。
    let _ = y;
    tape.reset();
    let x2 = tape.var(&t(vec![3.0, 4.0], &[2]));
    let y2 = x2.mul(&x2).unwrap();
    let jac = jacobian(&tape, &y2, &x2).unwrap();
    assert_eq!(host(&jac), vec![6.0, 0.0, 0.0, 8.0]);
}

#[test]
fn j4_hessian_rejects_non_scalar_loss_and_leaves_child_untouched() {
    let tape = new_tape();
    let child = new_tape();
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = x.mul(&x).unwrap();
    assert!(matches!(
        hessian(&tape, &y, &x, &child),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(child.is_empty());
}

#[test]
fn j4_hessian_propagates_create_graph_rejections_without_touching_child() {
    // 非空の子テープ。
    let tape = new_tape();
    let child = new_tape();
    let _leaf = child.var(&t(vec![0.0], &[1]));
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let loss = x.mul(&x).unwrap().sum(None).unwrap();
    assert!(matches!(
        hessian(&tape, &loss, &x, &child),
        Err(AutodiffError::Backward(_))
    ));
    assert_eq!(child.len(), 1);

    // 非対象 Op（`Max`）。
    let tape2 = new_tape();
    let child2 = new_tape();
    let x2 = tape2.var(&t(vec![1.0, 2.0], &[2]));
    let loss2 = x2.max(None).unwrap();
    assert!(matches!(
        hessian(&tape2, &loss2, &x2, &child2),
        Err(AutodiffError::Backward(_))
    ));
    assert!(child2.is_empty());

    // rank 3 の matmul。
    let tape3 = new_tape();
    let child3 = new_tape();
    let a = tape3.var(&t(vec![1.0; 8], &[2, 2, 2]));
    let b = tape3.var(&t(vec![1.0; 8], &[2, 2, 2]));
    let loss3 = a.matmul(&b).unwrap().sum(None).unwrap();
    assert!(matches!(
        hessian(&tape3, &loss3, &a, &child3),
        Err(AutodiffError::Backward(_))
    ));
    assert!(child3.is_empty());
}

#[test]
fn j4_untracked_output_and_unrelated_rows_are_zero() {
    let tape = new_tape();
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let y_const = tape.var_no_grad(&t(vec![5.0, 6.0], &[2]));
    let jac = jacobian(&tape, &y_const, &x).unwrap();
    assert_eq!(jac.shape(), &[2, 3]);
    assert!(host(&jac).iter().all(|&v| v == 0.0));

    // 追跡される出力のうち一部の行が x に依存しない（cat の定数側）。
    let k = tape.var_no_grad(&t(vec![7.0, 8.0], &[2]));
    let y = Var::cat(&[x.mul(&x).unwrap(), k], 0).unwrap();
    let jac = jacobian(&tape, &y, &x).unwrap();
    assert_eq!(jac.shape(), &[5, 3]);
    let v = host(&jac);
    assert_eq!(&v[..9], &[2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 6.0]);
    assert!(v[9..].iter().all(|&e| e == 0.0));
}

#[test]
fn j4_hessian_of_linear_loss_and_unreachable_input_is_zero() {
    let tape = new_tape();
    let child = new_tape();
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let unrelated = tape.var(&t(vec![3.0, 4.0], &[2]));
    // loss は `unrelated` のみに依存する。x へは到達しない。
    let loss = unrelated.mul(&unrelated).unwrap().sum(None).unwrap();
    let h = hessian(&tape, &loss, &x, &child).unwrap();
    assert_eq!(h.shape(), &[2, 2]);
    assert!(host(&h).iter().all(|&v| v == 0.0));
}

#[test]
fn j4_zero_numel_input_returns_empty_result_without_panicking() {
    let tape = new_tape();
    let child = new_tape();
    let x = tape.var(&Tensor::<f32>::zeros(&[0]).unwrap());
    let y = tape.var_no_grad(&t(vec![5.0, 6.0], &[2]));
    let jac = jacobian(&tape, &y, &x).unwrap();
    assert_eq!(jac.shape(), &[2, 0]);
    assert_eq!(jac.numel(), 0);
    let w = tape.var(&t(vec![1.0], &[1]));
    let loss = w.sum(None).unwrap();
    let h = hessian(&tape, &loss, &x, &child).unwrap();
    assert_eq!(h.shape(), &[0, 0]);
}

#[test]
fn j4_existing_values_and_backward_results_are_bit_invariant() {
    let tape = new_tape();
    let x = tape.var(&t(vec![0.5, -1.0, 2.0, 0.25], &[2, 2]));
    let y = x.tanh().mul(&x).unwrap();
    let loss = y.sum(None).unwrap();
    let y_before = host(&y.to_tensor());
    let g_before = host(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());

    let _ = jacobian(&tape, &y, &x).unwrap();
    let _ = jacobian(&tape, &loss, &x).unwrap();

    assert_eq!(
        y_before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        host(&y.to_tensor())
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    let g_after = host(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
    assert_eq!(
        g_before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        g_after.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}
