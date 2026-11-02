//! facade 公開の f64 前進モード風変換 `TapeF64::{jvp, jacfwd, jvp_multi, jvp_create_graph}`／
//! `JvpCreateGraphResultF64`（イシュー #3788・親 #3719。公開形は
//! `docs/autodiff-functional-transforms-design.md` §36、実装記録は同 §40、承認は
//! <https://github.com/Fandhe-AI/fandhe-ai/issues/3766#issuecomment-6554563674>）の公開経路テスト。
//!
//! 内部クレートを一切 import せず `fandhe_ai::*` の公開 API だけで組む（利用者視点の到達性確認）。
//! 演算ごとの網羅検証は `crates/autodiff/tests/f64_functional_transforms.rs` が担い、本ファイルは
//! facade 経由でも入口検査の順序・エラー伝播・値の契約が保たれることを固定する。
//!
//! 判定方式: 数値比較は REQ-2 統一複合判定（`fandhe_ai_backend_cpu::assert_parity_f64`）。
//! tolerance は新設・緩和しない。
//!
//! GPU 実機テストは `#[ignore]`（CI 非実行）。手順は `docs/perf/logs/f64-jvp-facade-3788/README.md`。

use fandhe_ai::{AutodiffError, Device, JvpCreateGraphResultF64, Tape, TapeF64, Tensor, VarF64};
use fandhe_ai_backend_cpu::assert_parity_f64;

fn t64(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    Tensor::<f64>::new(data, shape).expect("test fixture: tensor")
}

fn vals(t: &Tensor<f64>) -> Vec<f64> {
    t.as_slice().expect("CPU contiguous").to_vec()
}

/// 公開シグネチャを関数ポインタで固定する（変更はコンパイルエラーになる）。
#[test]
fn signatures_are_fixed() {
    fn probe<'g, 't: 'g>() {
        type TensorResult = Result<Tensor<f64>, AutodiffError>;
        type JvpMultiFn<'a, 'b, 'c, 'd, 'e, 't> = fn(
            &'a TapeF64<'t>,
            &'b VarF64<'c, 't>,
            &'b [&'d VarF64<'d, 't>],
            &'e [&'e Tensor<f64>],
            &'e TapeF64<'t>,
        ) -> TensorResult;
        type JvpCreateGraphFn<'a, 'b, 'c, 'd, 'e, 'f, 't> =
            fn(
                &'a TapeF64<'t>,
                &'b VarF64<'c, 't>,
                &'b VarF64<'d, 't>,
                &'e Tensor<f64>,
                &'e TapeF64<'t>,
                &'f TapeF64<'t>,
            ) -> Result<JvpCreateGraphResultF64<'f, 't>, AutodiffError>;
        let _: for<'a, 'b, 'c, 'd, 'e> fn(
            &'a TapeF64<'t>,
            &'b VarF64<'c, 't>,
            &'b VarF64<'d, 't>,
            &'e Tensor<f64>,
            &'e TapeF64<'t>,
        ) -> TensorResult = TapeF64::jvp;
        let _: for<'a, 'b, 'c, 'd, 'e> fn(
            &'a TapeF64<'t>,
            &'b VarF64<'c, 't>,
            &'b VarF64<'d, 't>,
            &'e TapeF64<'t>,
        ) -> TensorResult = TapeF64::jacfwd;
        let _: JvpMultiFn<'_, '_, '_, '_, '_, 't> = TapeF64::jvp_multi;
        let _: JvpCreateGraphFn<'_, '_, '_, '_, '_, '_, 't> = TapeF64::jvp_create_graph;
        type Lookup<'g, 't> = Result<Option<VarF64<'g, 't>>, AutodiffError>;
        let _: for<'a> fn(&'a JvpCreateGraphResultF64<'g, 't>) -> VarF64<'g, 't> =
            JvpCreateGraphResultF64::output;
        let _: for<'a, 'b> fn(
            &'a JvpCreateGraphResultF64<'g, 't>,
            &'b VarF64<'g, 't>,
        ) -> Lookup<'g, 't> = JvpCreateGraphResultF64::child_var;
    }
    probe();
}

/// 非線形（二階が 0 にならない）な 2 入力グラフ: y = (a @ b)^2 を行ごとに和した上で `a ⊙ a` を足す。
fn graph<'g, 't>(g: &'g TapeF64<'t>) -> (VarF64<'g, 't>, VarF64<'g, 't>, VarF64<'g, 't>) {
    let a = g.var(&t64(vec![0.5, 1.5, 2.0, 1.0], &[2, 2]));
    let b = g.var(&t64(vec![1.0, -0.5, 0.25, 2.0], &[2, 2]));
    let h = a.matmul(&b).unwrap();
    let y = h.mul(&h).unwrap().add(&a.mul(&a).unwrap()).unwrap();
    (a, b, y)
}

/// jvp・jacfwd・jvp_multi・jvp_create_graph の出力を CPU で求める。
fn run_all(tape: &Tape) -> Vec<(&'static str, Vec<f64>)> {
    let (va, vb) = (
        t64(vec![1.0, 0.5, -1.0, 2.0], &[2, 2]),
        t64(vec![0.25, 1.0, 0.5, -0.5], &[2, 2]),
    );
    let mut out = Vec::new();
    {
        let (p, c) = (TapeF64::new(tape), TapeF64::new(tape));
        let (a, _b, y) = graph(&p);
        out.push(("jvp", vals(&p.jvp(&y, &a, &va, &c).unwrap())));
    }
    {
        let (p, c) = (TapeF64::new(tape), TapeF64::new(tape));
        let (a, _b, y) = graph(&p);
        out.push(("jacfwd", vals(&p.jacfwd(&y, &a, &c).unwrap())));
    }
    {
        let (p, c) = (TapeF64::new(tape), TapeF64::new(tape));
        let (a, b, y) = graph(&p);
        out.push((
            "jvp_multi",
            vals(&p.jvp_multi(&y, &[&a, &b], &[&va, &vb], &c).unwrap()),
        ));
    }
    {
        let (p, c, g) = (TapeF64::new(tape), TapeF64::new(tape), TapeF64::new(tape));
        let (a, _b, y) = graph(&p);
        let res = p.jvp_create_graph(&y, &a, &va, &c, &g).unwrap();
        out.push(("jvp_create_graph.output", vals(&res.output().value())));
        let ca = res.child_var(&a).unwrap().expect("a の写し");
        let grads = g.backward(&res.output().sum(None).unwrap()).unwrap();
        out.push((
            "jvp_create_graph.d_output_d_a",
            vals(grads.get(&ca).unwrap().expect("勾配が届く")),
        ));
    }
    out
}

#[test]
fn four_functions_are_reachable_and_consistent_on_cpu() {
    let tape = fandhe_ai::tape();
    let r = run_all(&tape);
    let get = |name: &str| r.iter().find(|(n, _)| *n == name).unwrap().1.clone();
    // jvp_create_graph の出力は jvp と同じ値（REQ-2 の範囲）。
    assert_parity_f64(
        "jvp_create_graph.output vs jvp",
        &get("jvp_create_graph.output"),
        &get("jvp"),
    );
    // jacfwd と接ベクトルの積 J·v は jvp と一致する（J は [4, 4]、v は 4 要素）。
    let j = get("jacfwd");
    let v = [1.0, 0.5, -1.0, 2.0];
    let jv: Vec<f64> = j
        .chunks(4)
        .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
        .collect();
    assert_parity_f64("jacfwd·v vs jvp", &jv, &get("jvp"));
    // a だけの jvp は、a と b の jvp_multi から b の寄与（b 側 jvp）を引いたものに一致する。
    let tape2 = fandhe_ai::tape();
    let (p, c) = (TapeF64::new(&tape2), TapeF64::new(&tape2));
    let (_a, b, y) = graph(&p);
    let vb = t64(vec![0.25, 1.0, 0.5, -0.5], &[2, 2]);
    let only_b = vals(&p.jvp(&y, &b, &vb, &c).unwrap());
    let sum: Vec<f64> = get("jvp").iter().zip(&only_b).map(|(x, y)| x + y).collect();
    assert_parity_f64("jvp(a)+jvp(b) vs jvp_multi", &sum, &get("jvp_multi"));
}

#[test]
fn second_order_through_grandchild_matches_closed_form() {
    // y = x^3、v = [1, 0.5, 2]: J·v = 3 x^2 v、Σ の x 微分は 6 x v。
    let tape = fandhe_ai::tape();
    let (p, c, g) = (
        TapeF64::new(&tape),
        TapeF64::new(&tape),
        TapeF64::new(&tape),
    );
    let xs = [1.0, 2.0, 3.0];
    let vs = [1.0, 0.5, 2.0];
    let x = p.var(&t64(xs.to_vec(), &[3]));
    let y = x.mul(&x).unwrap().mul(&x).unwrap();
    let res = p
        .jvp_create_graph(&y, &x, &t64(vs.to_vec(), &[3]), &c, &g)
        .unwrap();
    let cx = res.child_var(&x).unwrap().expect("x の写し");
    let grads = g.backward(&res.output().sum(None).unwrap()).unwrap();
    let got = vals(grads.get(&cx).unwrap().expect("勾配が届く"));
    let want: Vec<f64> = xs.iter().zip(vs).map(|(x, v)| 6.0 * x * v).collect();
    assert_parity_f64("二階 6xv", &got, &want);
}

#[test]
fn entry_checks_propagate_through_facade() {
    let tape = fandhe_ai::tape();
    let (p, c, g, other) = (
        TapeF64::new(&tape),
        TapeF64::new(&tape),
        TapeF64::new(&tape),
        TapeF64::new(&tape),
    );
    let x = p.var(&t64(vec![1.0, 2.0], &[2]));
    let y = x.mul(&x).unwrap();
    let o = other.var(&t64(vec![1.0, 2.0], &[2]));
    let nog = p.var_no_grad(&t64(vec![1.0, 2.0], &[2]));
    let v = t64(vec![1.0, 1.0], &[2]);
    let short = t64(vec![1.0], &[1]);
    assert!(matches!(
        p.jvp(&y, &o, &v, &c),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        p.jvp(&y, &nog, &v, &c),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    assert!(matches!(
        p.jvp(&y, &x, &short, &c),
        Err(AutodiffError::Shape(_))
    ));
    assert!(matches!(
        p.jacfwd(&y, &nog, &c),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    // jvp_multi: 長さ不一致・空・重複は InvalidArgument。
    assert!(matches!(
        p.jvp_multi(&y, &[&x], &[], &c),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        p.jvp_multi(&y, &[], &[], &c),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        p.jvp_multi(&y, &[&x, &x], &[&v, &v], &c),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // jvp_create_graph: 子が親と同一・孫が子と同一は Backward。
    assert!(matches!(
        p.jvp_create_graph(&y, &x, &v, &p, &g),
        Err(AutodiffError::Backward(_))
    ));
    assert!(matches!(
        p.jvp_create_graph(&y, &x, &v, &c, &c),
        Err(AutodiffError::Backward(_))
    ));
    // 失敗した入口検査では c・g は無変更なので、そのまま正しい呼び出しが通る。
    assert!(p.jvp_create_graph(&y, &x, &v, &c, &g).is_ok());
}

#[test]
fn max_is_rejected_and_child_stays_reusable() {
    let tape = fandhe_ai::tape();
    let (p, c, g) = (
        TapeF64::new(&tape),
        TapeF64::new(&tape),
        TapeF64::new(&tape),
    );
    let x = p.var(&t64(vec![1.0, 2.0], &[2]));
    let bad = x.max(None).unwrap();
    let v = t64(vec![1.0, 1.0], &[2]);
    assert!(matches!(
        p.jvp(&bad, &x, &v, &c),
        Err(AutodiffError::Backward(_))
    ));
    assert!(matches!(
        p.jvp_multi(&bad, &[&x], &[&v], &c),
        Err(AutodiffError::Backward(_))
    ));
    assert!(matches!(
        p.jvp_create_graph(&bad, &x, &v, &c, &g),
        Err(AutodiffError::Backward(_))
    ));
    let good = x.mul(&x).unwrap();
    assert!(p.jvp(&good, &x, &v, &c).is_ok());
}

#[test]
fn child_var_checks_ownership_and_tracking() {
    let tape = fandhe_ai::tape();
    let (p, c, g, other) = (
        TapeF64::new(&tape),
        TapeF64::new(&tape),
        TapeF64::new(&tape),
        TapeF64::new(&tape),
    );
    let x = p.var(&t64(vec![1.0, 2.0], &[2]));
    let unrelated = p.var(&t64(vec![1.0, 2.0], &[2]));
    let nog = p.var_no_grad(&t64(vec![1.0, 2.0], &[2]));
    let y = x.mul(&x).unwrap();
    let res = p
        .jvp_create_graph(&y, &x, &t64(vec![1.0, 1.0], &[2]), &c, &g)
        .unwrap();
    assert!(matches!(
        res.child_var(&other.var(&t64(vec![1.0], &[1]))),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        res.child_var(&nog),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    assert!(res.child_var(&unrelated).unwrap().is_none());
}

/// GPU 実機: CPU 結果と REQ-2 統一複合判定で比較する。
fn gpu_matches_cpu(device: Device) {
    let cpu = fandhe_ai::tape();
    let gpu = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテスト");
    let (c, gp) = (run_all(&cpu), run_all(&gpu));
    for ((name, want), (_, got)) in c.iter().zip(&gp) {
        assert_parity_f64(&format!("{name}(gpu vs cpu)"), got, want);
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要"]
fn f64_jvp_family_cuda_matches_cpu() {
    gpu_matches_cpu(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要"]
fn f64_jvp_family_metal_matches_cpu() {
    gpu_matches_cpu(Device::Metal);
}
