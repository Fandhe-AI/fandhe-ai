//! `TapeF64::backward_create_graph`（イシュー #3785・親 #3719。設計記録
//! `docs/autodiff-functional-transforms-design.md` §35／§37）の CPU 統合テスト。
//!
//! `mod common;`（`NaiveOps`。`typed_ops_f64` は `None` でホスト経路）の上で動かす。
//! `autodiff` は具体バックエンドクレートへ依存しない不変条件があるため、REQ-2 の判定は
//! `common::req2_close`（`backend-cpu::parity::compare` の独立再実装）を使う。tolerance は
//! 新設しない。loss は第 1 段で `sum` が使えないため常に非スカラー（シードは全要素 1）で、
//! 線形 Op だけで二階が 0 になる検証（§34.3）は作らない。実機（CUDA／Metal）の
//! `#[ignore]` テストは #3786 で追加する（第 1 段に実機依存のテストはない）。

mod common;

use fandhe_ai_autodiff::f64_autograd::{TapeF64, VarF64};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{BackendError, BackendOps, Device, Tensor, TypedOps};

type R<T> = Result<T, AutodiffError>;

fn t(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn vals(x: &Tensor<f64>) -> Vec<f64> {
    x.contiguous().host_slice().to_vec()
}

fn assert_close(got: &[f64], want: &[f64], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: 要素数が違う");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(common::req2_close(*g, *w), "{ctx}[{i}]: got {g} want {w}");
    }
}

fn new_tape() -> Tape {
    Tape::new_with_ops(common::naive_ops())
}

/// 親 x:[2,3]・s:[3] の合成 loss（add／mul／div／pow とブロードキャストを全部通す）。
fn composite<'g, 't>(g: &'g TapeF64<'t>) -> R<(VarF64<'g, 't>, VarF64<'g, 't>, VarF64<'g, 't>)> {
    let x = g.var(&t(vec![0.5, 1.5, 2.0, 1.0, 0.7, 3.0], &[2, 3]));
    let s = g.var(&t(vec![1.2, 0.8, 1.7], &[3]));
    let two = g.var_no_grad(&t(vec![2.0], &[]));
    let num = x.mul(&s)?.add(&x)?;
    let den = s.add(&two)?;
    let loss = num.div(&den)?.add(&x.pow(&s)?)?;
    Ok((x, s, loss))
}

#[test]
fn first_order_is_bit_identical_to_plain_backward() -> R<()> {
    let base = new_tape();
    let (p1, c1) = (TapeF64::new(&base), TapeF64::new(&base));
    let (x1, s1, l1) = composite(&p1)?;
    let cg = p1.backward_create_graph(&l1, &c1)?;
    let p2 = TapeF64::new(&base);
    let (x2, s2, l2) = composite(&p2)?;
    let plain = p2.backward(&l2)?;
    for (a, b) in [(&x1, &x2), (&s1, &s2)] {
        let got = cg.first_order().get(a)?.expect("届く").clone();
        let want = plain.get(b)?.expect("届く").clone();
        let (g, w) = (vals(&got), vals(&want));
        assert!(g.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    Ok(())
}

#[test]
fn child_grad_matches_first_order_within_req2() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let (x, s, l) = composite(&p)?;
    let cg = p.backward_create_graph(&l, &c)?;
    for v in [&x, &s] {
        let g = cg.grad(v)?.expect("届く").value();
        let f = cg.first_order().get(v)?.expect("届く");
        assert_close(&vals(&g), &vals(f), "grad vs first_order");
    }
    Ok(())
}

/// 子テープ上の `grad(v)` を `target` の写しで微分した二階勾配を返す。
fn second<'t>(
    cg: &fandhe_ai_autodiff::f64_autograd::CreateGraphResultF64<'_, 't>,
    child: &TapeF64<'t>,
    of: &VarF64<'_, 't>,
    wrt: &VarF64<'_, 't>,
) -> R<Vec<f64>> {
    let g = cg.grad(of)?.expect("勾配が届く");
    let grads = child.backward(&g)?;
    let w = cg.child_var(wrt)?.expect("写しがある");
    Ok(vals(grads.get(&w)?.expect("二階が届く")))
}

#[test]
fn cube_second_derivative_is_six_x() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let xs = [0.5, 1.0, 2.5];
    let x = p.var(&t(xs.to_vec(), &[3]));
    let l = x.mul(&x)?.mul(&x)?;
    let cg = p.backward_create_graph(&l, &c)?;
    let want: Vec<f64> = xs.iter().map(|v| 6.0 * v).collect();
    assert_close(&second(&cg, &c, &x, &x)?, &want, "x^3");
    Ok(())
}

#[test]
fn div_second_derivatives() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let (av, bv) = ([1.0, 2.0, 3.0], [0.5, 1.5, 2.0]);
    let a = p.var(&t(av.to_vec(), &[3]));
    let b = p.var(&t(bv.to_vec(), &[3]));
    let l = a.div(&b)?;
    let cg = p.backward_create_graph(&l, &c)?;
    // of = b: d/db(-a/b^2) 系。Σ grad_b を a・b で微分する。
    let da: Vec<f64> = bv.iter().map(|b| -1.0 / (b * b)).collect();
    let db: Vec<f64> = av
        .iter()
        .zip(&bv)
        .map(|(a, b)| 2.0 * a / (b * b * b))
        .collect();
    assert_close(&second(&cg, &c, &b, &a)?, &da, "d2/dbda");
    assert_close(&second(&cg, &c, &b, &b)?, &db, "d2/db2");
    Ok(())
}

#[test]
fn constant_exponent_pow_second_derivative() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let xs = [0.5, 1.0, 2.5];
    let x = p.var(&t(xs.to_vec(), &[3]));
    let e = p.var_no_grad(&t(vec![3.5], &[]));
    let l = x.pow(&e)?;
    let cg = p.backward_create_graph(&l, &c)?;
    let want: Vec<f64> = xs.iter().map(|v| 3.5 * 2.5 * v.powf(1.5)).collect();
    assert_close(&second(&cg, &c, &x, &x)?, &want, "x^p");
    Ok(())
}

#[test]
fn tracked_base_and_exponent_mixed_partials() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let (av, bv) = ([0.5, 1.5, 2.0], [1.2, 2.5, 0.7]);
    let a = p.var(&t(av.to_vec(), &[3]));
    let b = p.var(&t(bv.to_vec(), &[3]));
    let l = a.pow(&b)?;
    let cg = p.backward_create_graph(&l, &c)?;
    // of = a: grad_a = b a^(b-1)
    let daa: Vec<f64> = av
        .iter()
        .zip(&bv)
        .map(|(a, b)| b * (b - 1.0) * a.powf(b - 2.0))
        .collect();
    let dab: Vec<f64> = av
        .iter()
        .zip(&bv)
        .map(|(a, b)| a.powf(b - 1.0) + b * a.powf(b - 1.0) * a.ln())
        .collect();
    assert_close(&second(&cg, &c, &a, &a)?, &daa, "d2/da2");
    assert_close(&second(&cg, &c, &a, &b)?, &dab, "d2/dadb");
    // of = b: grad_b = a^b ln a → d/db = a^b (ln a)^2
    let dbb: Vec<f64> = av
        .iter()
        .zip(&bv)
        .map(|(a, b)| a.powf(*b) * a.ln() * a.ln())
        .collect();
    assert_close(&second(&cg, &c, &b, &b)?, &dbb, "d2/db2");
    Ok(())
}

#[test]
fn broadcast_second_derivatives() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let xv = [0.5, 1.5, 2.0, 1.0, 0.7, 3.0];
    let sv = [1.2, 0.8, 1.7];
    let x = p.var(&t(xv.to_vec(), &[2, 3]));
    let s = p.var(&t(sv.to_vec(), &[3]));
    let xs = x.mul(&s)?;
    let l = xs.mul(&xs)?;
    let cg = p.backward_create_graph(&l, &c)?;
    // L = Σ (x s)^2。grad_s_j = Σ_i 2 x_ij^2 s_j。Σ_j を s・x で微分する。
    let ds: Vec<f64> = (0..3)
        .map(|j| 2.0 * (xv[j] * xv[j] + xv[3 + j] * xv[3 + j]))
        .collect();
    let dx: Vec<f64> = (0..6).map(|k| 4.0 * xv[k] * sv[k % 3]).collect();
    assert_close(&second(&cg, &c, &s, &s)?, &ds, "d/ds");
    assert_close(&second(&cg, &c, &s, &x)?, &dx, "d/dx");
    Ok(())
}

/// 1 階勾配の総和を再構築して中心差分と比べる（h = 1e-6）。
#[test]
fn second_order_matches_central_difference() -> R<()> {
    let xv = [0.5, 1.5, 2.0, 1.0, 0.7, 3.0];
    let grad_sum = |xs: &[f64]| -> R<f64> {
        let base = new_tape();
        let p = TapeF64::new(&base);
        let x = p.var(&t(xs.to_vec(), &[2, 3]));
        let s = p.var(&t(vec![1.2, 0.8, 1.7], &[3]));
        let two = p.var_no_grad(&t(vec![2.0], &[]));
        let l = x.mul(&s)?.add(&x)?.div(&s.add(&two)?)?.add(&x.pow(&s)?)?;
        let g = p.backward(&l)?;
        Ok(vals(g.get(&x)?.expect("届く")).iter().sum())
    };
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let x = p.var(&t(xv.to_vec(), &[2, 3]));
    let s = p.var(&t(vec![1.2, 0.8, 1.7], &[3]));
    let two = p.var_no_grad(&t(vec![2.0], &[]));
    let l = x.mul(&s)?.add(&x)?.div(&s.add(&two)?)?.add(&x.pow(&s)?)?;
    let cg = p.backward_create_graph(&l, &c)?;
    let got = second(&cg, &c, &x, &x)?;
    let h = 1e-6;
    // Σ grad_x の x_k での偏導関数（各要素は独立に寄与するため対角のみ）。
    for k in 0..6 {
        let (mut up, mut dn) = (xv.to_vec(), xv.to_vec());
        up[k] += h;
        dn[k] -= h;
        let fd = (grad_sum(&up)? - grad_sum(&dn)?) / (2.0 * h);
        assert!(
            common::req2_close(got[k], fd),
            "k={k}: got {} fd {fd}",
            got[k]
        );
    }
    Ok(())
}

/// 閉包性: 勾配グラフ自体を再度 `backward_create_graph` できる（三階）。
#[test]
fn third_order_through_mirrored_log_where_broadcast() -> R<()> {
    let base = new_tape();
    let (p, c, gc) = (
        TapeF64::new(&base),
        TapeF64::new(&base),
        TapeF64::new(&base),
    );
    // x^4（定数冪）と、ブロードキャスト付き x^3 / s。
    let xv = [0.5, 1.5, 2.0];
    let x = p.var(&t(xv.to_vec(), &[3]));
    let e4 = p.var_no_grad(&t(vec![4.0], &[]));
    let l = x.pow(&e4)?;
    let cg1 = p.backward_create_graph(&l, &c)?;
    let g1 = cg1.grad(&x)?.expect("届く");
    let cx = cg1.child_var(&x)?.expect("写し");
    let cg2 = c.backward_create_graph(&g1, &gc)?;
    let g2 = cg2.grad(&cx)?.expect("届く");
    let gx = cg2.child_var(&cx)?.expect("写し");
    let third = gc.backward(&g2)?;
    let want: Vec<f64> = xv.iter().map(|v| 24.0 * v).collect();
    assert_close(&vals(third.get(&gx)?.expect("届く")), &want, "x^4 三階");

    // 底・指数とも追跡: grad_b = a^b ln a を写した Log／Where／Pow を二段目で再生する。
    let (p, c, gc) = (
        TapeF64::new(&base),
        TapeF64::new(&base),
        TapeF64::new(&base),
    );
    let (av, bv) = ([0.5, 1.5, 2.0], [1.2, 2.5, 0.7]);
    let a = p.var(&t(av.to_vec(), &[3]));
    let b = p.var(&t(bv.to_vec(), &[3]));
    let l = a.pow(&b)?;
    let cg1 = p.backward_create_graph(&l, &c)?;
    let g1b = cg1.grad(&b)?.expect("届く");
    let cb = cg1.child_var(&b)?.expect("写し");
    let cg2 = c.backward_create_graph(&g1b, &gc)?;
    let g2b = cg2.grad(&cb)?.expect("届く");
    let gb = cg2.child_var(&cb)?.expect("写し");
    let third = gc.backward(&g2b)?;
    let want: Vec<f64> = av
        .iter()
        .zip(&bv)
        .map(|(a, b)| a.powf(*b) * a.ln().powi(3))
        .collect();
    assert_close(&vals(third.get(&gb)?.expect("届く")), &want, "a^b 三階");

    // Div + ブロードキャスト: x^3 / s の x での三階は 6 / s。
    let (p, c, gc) = (
        TapeF64::new(&base),
        TapeF64::new(&base),
        TapeF64::new(&base),
    );
    let sv = [1.2, 0.8, 1.7];
    let x = p.var(&t(vec![0.5, 1.5, 2.0, 1.0, 0.7, 3.0], &[2, 3]));
    let s = p.var(&t(sv.to_vec(), &[3]));
    let l = x.mul(&x)?.mul(&x)?.div(&s)?;
    let cg1 = p.backward_create_graph(&l, &c)?;
    let g1 = cg1.grad(&x)?.expect("届く");
    let cx = cg1.child_var(&x)?.expect("写し");
    let cg2 = c.backward_create_graph(&g1, &gc)?;
    let g2 = cg2.grad(&cx)?.expect("届く");
    let gx = cg2.child_var(&cx)?.expect("写し");
    let third = gc.backward(&g2)?;
    let want: Vec<f64> = (0..6).map(|k| 6.0 / sv[k % 3]).collect();
    assert_close(&vals(third.get(&gx)?.expect("届く")), &want, "x^3/s 三階");
    Ok(())
}

#[test]
fn pow_zero_base_and_exponent_stay_finite() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let a = p.var(&t(vec![0.0, 2.0, 0.0], &[3]));
    let b = p.var(&t(vec![0.0, 0.0, 3.0], &[3]));
    let l = a.pow(&b)?;
    let cg = p.backward_create_graph(&l, &c)?;
    let ga = vals(&cg.grad(&a)?.expect("届く").value());
    let gb = vals(&cg.grad(&b)?.expect("届く").value());
    assert_eq!(ga, vec![0.0, 0.0, 0.0]);
    assert_close(&gb, &[0.0, 2.0f64.ln(), 0.0], "grad_b");
    let first = cg.first_order();
    assert_eq!(vals(first.get(&a)?.expect("届く")), ga);
    // 二階も有限（選ばれない枝の 0*inf が NaN を作らない）。
    for (of, wrt) in [(&a, &a), (&a, &b), (&b, &a), (&b, &b)] {
        for v in second(&cg, &c, of, wrt)? {
            assert!(v.is_finite(), "二階に非有限値: {v}");
        }
    }
    Ok(())
}

/// 底が正で指数が 0 の位置でも混合二階微分を定数マスクで消さない
/// （d²(a^b)/da db = a^(b-1)(1 + b ln a) を b=0 で評価すると 1/a）。
#[test]
fn pow_positive_base_zero_exponent_keeps_mixed_partials() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let av = [2.0, 0.5, 3.0];
    let a = p.var(&t(av.to_vec(), &[3]));
    let b = p.var(&t(vec![0.0, 0.0, 0.0], &[3]));
    let l = a.pow(&b)?;
    let cg = p.backward_create_graph(&l, &c)?;
    let inv: Vec<f64> = av.iter().map(|v| 1.0 / v).collect();
    let ln2: Vec<f64> = av.iter().map(|v| v.ln() * v.ln()).collect();
    assert_close(&second(&cg, &c, &a, &b)?, &inv, "d2/da db");
    assert_close(&second(&cg, &c, &b, &a)?, &inv, "d2/db da");
    assert_close(&second(&cg, &c, &a, &a)?, &[0.0, 0.0, 0.0], "d2/da2");
    assert_close(&second(&cg, &c, &b, &b)?, &ln2, "d2/db2");
    Ok(())
}

/// 微小な正の底（1e-310）と非追跡のゼロ指数でも `a^(-1)` の overflow から
/// `0 * inf = NaN` を作らず、1 階と backward_create_graph の勾配が一致する。
#[test]
fn pow_subnormal_base_const_zero_exponent_is_finite() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let a = p.var(&t(vec![1e-310], &[1]));
    let b = p.var_no_grad(&t(vec![0.0], &[1]));
    let l = a.pow(&b)?;
    let cg = p.backward_create_graph(&l, &c)?;
    assert_eq!(vals(cg.first_order().get(&a)?.expect("届く")), vec![0.0]);
    for v in second(&cg, &c, &a, &a)? {
        assert!(v.is_finite(), "二階に非有限値: {v}");
    }
    Ok(())
}

/// device だけ差し替える `BackendOps`（`DeviceMismatch` の再現用）。
struct OtherDevice(Box<dyn BackendOps + Send>);
impl BackendOps for OtherDevice {
    fn device(&self) -> Device {
        Device::Cuda(7)
    }
    fn typed_ops_f64(&self) -> Option<&dyn TypedOps<f64>> {
        None
    }
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.0.gemm(a, b)
    }
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.0.add(a, b)
    }
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.0.mul(a, b)
    }
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.0.relu(a)
    }
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.0.exp(a)
    }
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.0.tanh(a)
    }
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.0.sum(a, dim)
    }
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.0.max(a, dim)
    }
}

#[test]
fn entry_checks_reject_in_order() -> R<()> {
    let base = new_tape();
    let p = TapeF64::new(&base);
    let x = p.var(&t(vec![1.0, 2.0], &[2]));
    let l = x.mul(&x)?;

    // 1. 別テープの loss
    let other = TapeF64::new(&base);
    let c = TapeF64::new(&base);
    let ox = other.var(&t(vec![1.0], &[1]));
    assert!(matches!(
        p.backward_create_graph(&ox.mul(&ox)?, &c),
        Err(AutodiffError::TapeMismatch)
    ));
    // 2. child が self と同一
    assert!(matches!(
        p.backward_create_graph(&l, &p),
        Err(AutodiffError::Backward(_))
    ));
    // 3. device 違い
    let alt = Tape::new_with_ops(Box::new(OtherDevice(common::naive_ops())));
    let alt_child = TapeF64::new(&alt);
    assert!(matches!(
        p.backward_create_graph(&l, &alt_child),
        Err(AutodiffError::DeviceMismatch {
            requested: Device::Cuda(7),
            actual: Device::Cpu
        })
    ));
    // 4. 空でない child
    let busy = TapeF64::new(&base);
    let _ = busy.var(&t(vec![1.0], &[1]));
    assert!(matches!(
        p.backward_create_graph(&l, &busy),
        Err(AutodiffError::Backward(_))
    ));
    Ok(())
}

#[test]
fn unsupported_ops_are_rejected_and_child_stays_reusable() -> R<()> {
    let base = new_tape();
    let p = TapeF64::new(&base);
    let c = TapeF64::new(&base);
    let m = p.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let x = p.var(&t(vec![1.0, 2.0], &[2]));
    let bad: Vec<VarF64<'_, '_>> = vec![
        m.matmul(&m)?,
        m.sum(None)?,
        m.sum(Some(0))?,
        m.mean(None)?,
        m.max(None)?,
        m.max(Some(1))?,
    ];
    for l in &bad {
        assert!(matches!(
            p.backward_create_graph(l, &c),
            Err(AutodiffError::Backward(_))
        ));
        // 何も書かれていない: 同じ child で正しい loss が通る。
    }
    let good = x.mul(&x)?;
    let cg = p.backward_create_graph(&good, &c)?;
    assert!(cg.grad(&x)?.is_some());
    Ok(())
}

#[test]
fn constant_subtree_with_unsupported_op_is_accepted() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let k = p.var_no_grad(&t(vec![3.0, 1.0, 2.0], &[3])).max(None)?; // 定数 3.0
    let x = p.var(&t(vec![1.0, 2.0], &[2]));
    let l = x.mul(&k)?.mul(&x)?; // 3 x^2
    let cg = p.backward_create_graph(&l, &c)?;
    assert_close(&second(&cg, &c, &x, &x)?, &[6.0, 6.0], "3x^2 の二階");
    Ok(())
}

#[test]
fn result_accessors_check_ownership_and_tracking() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let x = p.var(&t(vec![1.0, 2.0], &[2]));
    let k = p.var_no_grad(&t(vec![2.0, 2.0], &[2]));
    let unused = p.var(&t(vec![9.0], &[1]));
    let l = x.mul(&k)?;
    let cg = p.backward_create_graph(&l, &c)?;
    assert!(matches!(
        cg.grad(&k),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    let other = TapeF64::new(&base);
    let ov = other.var(&t(vec![1.0], &[1]));
    assert!(matches!(cg.grad(&ov), Err(AutodiffError::TapeMismatch)));
    assert!(matches!(
        cg.child_var(&ov),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(cg.grad(&unused)?.is_none());

    // 追跡なしの loss は既存どおり Backward で拒否される。
    let c2 = TapeF64::new(&base);
    assert!(matches!(
        p.backward_create_graph(&k, &c2),
        Err(AutodiffError::Backward(_))
    ));
    Ok(())
}

/// 底 1e-200・指数 0（ともに追跡）で、`a^(b-1)` の逆伝播係数 `-a^(-2)` が
/// overflow しても二階微分が NaN にならず、正しい d2/da2 = 0 を返す。
#[test]
fn pow_tiny_base_tracked_zero_exponent_second_order_is_finite() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let a = p.var(&t(vec![1e-200], &[1]));
    let b = p.var(&t(vec![0.0], &[1]));
    let l = a.pow(&b)?;
    let cg = p.backward_create_graph(&l, &c)?;
    assert_eq!(second(&cg, &c, &a, &a)?, vec![0.0]);
    // 混合二階微分は両方向とも閉形式 1/a = 1e200 に一致する（対称性）。
    let ab = second(&cg, &c, &a, &b)?;
    let ba = second(&cg, &c, &b, &a)?;
    assert_eq!(ab, vec![1e200]);
    assert_eq!(ba, vec![1e200]);
    for v in second(&cg, &c, &b, &b)? {
        assert!(v.is_finite(), "二階に非有限値: {v}");
    }
    Ok(())
}

/// 底 1e-310（非正規化数、逆数が overflow）・指数 0（ともに追跡）は NaN を
/// 流さず型付きエラーで拒否される。指数が非追跡なら d2/da2 = 0 を返す。
#[test]
fn pow_subnormal_base_tracked_zero_exponent_is_rejected_not_nan() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let a = p.var(&t(vec![1e-310], &[1]));
    let b = p.var(&t(vec![0.0], &[1]));
    let l = a.pow(&b)?;
    assert!(matches!(
        p.backward_create_graph(&l, &c),
        Err(AutodiffError::Backward(_))
    ));
    let (p2, c2) = (TapeF64::new(&base), TapeF64::new(&base));
    let a2 = p2.var(&t(vec![1e-310], &[1]));
    let k = p2.var_no_grad(&t(vec![0.0], &[1]));
    let l2 = a2.pow(&k)?;
    let cg = p2.backward_create_graph(&l2, &c2)?;
    assert_eq!(second(&cg, &c2, &a2, &a2)?, vec![0.0]);
    Ok(())
}

/// 拒否条件は broadcast 後の同一要素ペアで判定する。微小な底と指数 0 が別位置に
/// ある（拒否すべきペアが存在しない）入力は成功する。
#[test]
fn pow_tiny_base_and_zero_exponent_at_different_positions_is_accepted() -> R<()> {
    let base = new_tape();
    let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
    let a = p.var(&t(vec![1e-310, 2.0], &[2]));
    let b = p.var(&t(vec![2.0, 0.0], &[2]));
    let l = a.pow(&b)?;
    let cg = p.backward_create_graph(&l, &c)?;
    for v in second(&cg, &c, &a, &a)? {
        assert!(v.is_finite(), "二階に非有限値: {v}");
    }
    Ok(())
}
