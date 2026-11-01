//! facade 公開の f64 高階微分 `TapeF64::backward_create_graph`／`CreateGraphResultF64`
//! （イシュー #3786・親 #3719。公開形は `docs/autodiff-functional-transforms-design.md`
//! §35.1、実装記録は同 §38、承認は
//! <https://github.com/Fandhe-AI/fandhe-ai/issues/3766#issuecomment-6554563674>）の
//! 公開経路テスト。
//!
//! 内部クレートを一切 import せず `fandhe_ai::*` の公開 API だけで組む（利用者視点の
//! 到達性確認）。演算ごとの VJP の網羅検証は `crates/autodiff/tests/f64_create_graph.rs`
//! が担い、本ファイルは facade 経由でも入口検査の順序・エラー伝播・値の契約が保たれる
//! ことを固定する。
//!
//! 判定方式: 数値比較は REQ-2 統一複合判定（`fandhe_ai_backend_cpu::assert_parity_f64`）。
//! tolerance は新設・緩和しない。
//!
//! GPU 実機テストは `#[ignore]`（CI 非実行）。手順は
//! `docs/perf/logs/f64-create-graph-3786/README.md`。

use fandhe_ai::{
    AutodiffError, CreateGraphResultF64, Device, GradientsF64, Tape, TapeF64, Tensor, VarF64,
};
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
        let _: for<'a, 'b, 'c, 'd> fn(
            &'a TapeF64<'t>,
            &'b VarF64<'c, 't>,
            &'d TapeF64<'t>,
        )
            -> Result<CreateGraphResultF64<'d, 't>, AutodiffError> = TapeF64::backward_create_graph;
        let _: for<'a> fn(&'a CreateGraphResultF64<'g, 't>) -> &'a GradientsF64 =
            CreateGraphResultF64::first_order;
        type Lookup<'g, 't> = Result<Option<VarF64<'g, 't>>, AutodiffError>;
        let _: for<'a, 'b> fn(
            &'a CreateGraphResultF64<'g, 't>,
            &'b VarF64<'g, 't>,
        ) -> Lookup<'g, 't> = CreateGraphResultF64::grad;
        let _: for<'a, 'b> fn(
            &'a CreateGraphResultF64<'g, 't>,
            &'b VarF64<'g, 't>,
        ) -> Lookup<'g, 't> = CreateGraphResultF64::child_var;
    }
    probe();
}

/// loss = mean((a @ b)^2) + sum(a)^3 / 8。matmul・mean・sum を通す非線形 loss。
fn loss_of<'g, 't>(a: &VarF64<'g, 't>, b: &VarF64<'g, 't>) -> VarF64<'g, 't> {
    let y = a.matmul(b).unwrap();
    let quad = y.mul(&y).unwrap().mean(None).unwrap();
    let s = a.sum(None).unwrap();
    let cube = s.mul(&s).unwrap().mul(&s).unwrap();
    let eighth = a.sum(None).unwrap().add(&cube).unwrap();
    quad.add(&eighth).unwrap()
}

fn inputs() -> (Vec<f64>, Vec<f64>) {
    (
        vec![0.5, 1.5, 2.0, 1.0, 0.7, 3.0],
        vec![1.1, 0.4, 0.9, 1.3, 0.6, 2.0],
    )
}

/// 二階（Σ grad_a の a での偏導関数）を tape 上で求める。
fn second_order_on(tape: &Tape) -> (Vec<f64>, Vec<f64>) {
    let (av, bv) = inputs();
    let parent = TapeF64::new(tape);
    let child = TapeF64::new(tape);
    let a = parent.var(&t64(av, &[2, 3]));
    let b = parent.var(&t64(bv, &[3, 2]));
    let cg = parent
        .backward_create_graph(&loss_of(&a, &b), &child)
        .expect("create_graph");
    let ga = cg.grad(&a).unwrap().expect("a へ届く");
    let gb = cg.grad(&b).unwrap().expect("b へ届く");
    let total = ga.sum(None).unwrap().add(&gb.sum(None).unwrap()).unwrap();
    let grads = child.backward(&total).expect("child backward");
    let ca = cg.child_var(&a).unwrap().expect("a の写し");
    let cb = cg.child_var(&b).unwrap().expect("b の写し");
    (
        vals(grads.get(&ca).unwrap().expect("届く")),
        vals(grads.get(&cb).unwrap().expect("届く")),
    )
}

#[test]
fn first_order_is_bit_identical_to_plain_backward() {
    let tape = fandhe_ai::tape();
    let (av, bv) = inputs();
    let (p1, c1) = (TapeF64::new(&tape), TapeF64::new(&tape));
    let a1 = p1.var(&t64(av.clone(), &[2, 3]));
    let b1 = p1.var(&t64(bv.clone(), &[3, 2]));
    let cg = p1
        .backward_create_graph(&loss_of(&a1, &b1), &c1)
        .expect("create_graph");
    let p2 = TapeF64::new(&tape);
    let a2 = p2.var(&t64(av, &[2, 3]));
    let b2 = p2.var(&t64(bv, &[3, 2]));
    let plain = p2.backward(&loss_of(&a2, &b2)).expect("backward");
    for (x, y) in [(&a1, &a2), (&b1, &b2)] {
        let got = vals(cg.first_order().get(x).unwrap().expect("届く"));
        let want = vals(plain.get(y).unwrap().expect("届く"));
        assert!(
            got.iter()
                .zip(&want)
                .all(|(g, w)| g.to_bits() == w.to_bits())
        );
        // 子テープ上の勾配は REQ-2 の範囲で一致する。
        let g = vals(&cg.grad(x).unwrap().expect("届く").value());
        assert_parity_f64("grad vs first_order", &g, &want);
    }
}

#[test]
fn second_order_matches_central_difference() {
    let (av, bv) = inputs();
    let grad_sum = |av: &[f64], bv: &[f64]| -> f64 {
        let tape = fandhe_ai::tape();
        let p = TapeF64::new(&tape);
        let a = p.var(&t64(av.to_vec(), &[2, 3]));
        let b = p.var(&t64(bv.to_vec(), &[3, 2]));
        let g = p.backward(&loss_of(&a, &b)).unwrap();
        let sum = |v: &VarF64<'_, '_>| vals(g.get(v).unwrap().unwrap()).iter().sum::<f64>();
        sum(&a) + sum(&b)
    };
    let (got_a, got_b) = second_order_on(&fandhe_ai::tape());
    let h = 1e-5;
    let (mut fd_a, mut fd_b) = (vec![0.0; 6], vec![0.0; 6]);
    for k in 0..6 {
        let (mut up, mut dn) = (av.clone(), av.clone());
        up[k] += h;
        dn[k] -= h;
        fd_a[k] = (grad_sum(&up, &bv) - grad_sum(&dn, &bv)) / (2.0 * h);
        let (mut up, mut dn) = (bv.clone(), bv.clone());
        up[k] += h;
        dn[k] -= h;
        fd_b[k] = (grad_sum(&av, &up) - grad_sum(&av, &dn)) / (2.0 * h);
    }
    assert_parity_f64("二階 a", &got_a, &fd_a);
    assert_parity_f64("二階 b", &got_b, &fd_b);
}

/// 三階（x^4 の sum）。孫テープへの再帰が facade 経由でも成立する。
#[test]
fn third_order_through_facade() {
    let tape = fandhe_ai::tape();
    let (p, c, gc) = (
        TapeF64::new(&tape),
        TapeF64::new(&tape),
        TapeF64::new(&tape),
    );
    let xv = [0.5, 1.5, 2.0];
    let x = p.var(&t64(xv.to_vec(), &[3]));
    let sq = x.mul(&x).unwrap();
    let l = sq.mul(&sq).unwrap().sum(None).unwrap();
    let cg1 = p.backward_create_graph(&l, &c).unwrap();
    let g1 = cg1.grad(&x).unwrap().unwrap();
    let cx = cg1.child_var(&x).unwrap().unwrap();
    let cg2 = c.backward_create_graph(&g1, &gc).unwrap();
    let g2 = cg2.grad(&cx).unwrap().unwrap();
    let gx = cg2.child_var(&cx).unwrap().unwrap();
    let third = gc.backward(&g2).unwrap();
    let want: Vec<f64> = xv.iter().map(|v| 24.0 * v).collect();
    assert_parity_f64("x^4 三階", &vals(third.get(&gx).unwrap().unwrap()), &want);
}

#[test]
fn entry_checks_propagate_in_order() {
    let tape = fandhe_ai::tape();
    let (p, c) = (TapeF64::new(&tape), TapeF64::new(&tape));
    let other = TapeF64::new(&tape);
    let x = p.var(&t64(vec![1.0, 2.0], &[2]));
    let loss = x.mul(&x).unwrap();
    // 1. loss が別テープ
    let foreign = other.var(&t64(vec![1.0], &[1]));
    assert!(matches!(
        p.backward_create_graph(&foreign, &c),
        Err(AutodiffError::TapeMismatch)
    ));
    // 2. 子が親と同一
    assert!(matches!(
        p.backward_create_graph(&loss, &p),
        Err(AutodiffError::Backward(_))
    ));
    // 4. 子が空でない
    let used = TapeF64::new(&tape);
    let _ = used.var(&t64(vec![1.0], &[1]));
    assert!(matches!(
        p.backward_create_graph(&loss, &used),
        Err(AutodiffError::Backward(_))
    ));
    // 追跡なしの loss は既存どおり拒否される。
    let k = p.var_no_grad(&t64(vec![1.0, 2.0], &[2]));
    assert!(p.backward_create_graph(&k.mul(&k).unwrap(), &c).is_err());
    // 拒否のあとも同じ子で正しい loss が通る。
    assert!(p.backward_create_graph(&loss, &c).is_ok());
}

/// `max` は対象外で `Backward`。子テープは汚れず再利用できる。
#[test]
fn max_is_rejected_through_facade() {
    let tape = fandhe_ai::tape();
    let (p, c) = (TapeF64::new(&tape), TapeF64::new(&tape));
    let x = p.var(&t64(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]));
    for l in [x.max(None).unwrap(), x.max(Some(1)).unwrap()] {
        assert!(matches!(
            p.backward_create_graph(&l, &c),
            Err(AutodiffError::Backward(_))
        ));
    }
    let good = x.mul(&x).unwrap();
    assert!(p.backward_create_graph(&good, &c).is_ok());
}

#[test]
fn result_accessors_check_ownership_and_tracking() {
    let tape = fandhe_ai::tape();
    let (p, c, other) = (
        TapeF64::new(&tape),
        TapeF64::new(&tape),
        TapeF64::new(&tape),
    );
    let x = p.var(&t64(vec![1.0, 2.0], &[2]));
    let k = p.var_no_grad(&t64(vec![2.0, 2.0], &[2]));
    let unused = p.var(&t64(vec![9.0], &[1]));
    let cg = p
        .backward_create_graph(&x.mul(&k).unwrap(), &c)
        .expect("create_graph");
    assert!(matches!(
        cg.grad(&k),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    assert!(matches!(
        cg.grad(&other.var(&t64(vec![1.0], &[1]))),
        Err(AutodiffError::TapeMismatch)
    ));
    // loss から届かないノードは Ok(None)。
    assert!(cg.grad(&unused).unwrap().is_none());
    assert!(cg.child_var(&unused).unwrap().is_none());
}

/// GPU 実機: CPU 結果と REQ-2 統一複合判定で比較する。
fn gpu_matches_cpu(device: Device) {
    let cpu = fandhe_ai::tape();
    let gpu = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテスト");
    let (ca, cb) = second_order_on(&cpu);
    let (ga, gb) = second_order_on(&gpu);
    assert_parity_f64("二階 a(gpu vs cpu)", &ga, &ca);
    assert_parity_f64("二階 b(gpu vs cpu)", &gb, &cb);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要"]
fn f64_create_graph_cuda_matches_cpu() {
    gpu_matches_cpu(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要"]
fn f64_create_graph_metal_matches_cpu() {
    gpu_matches_cpu(Device::Metal);
}
