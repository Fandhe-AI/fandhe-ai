//! double-VJP 法（`jvp`／`jacfwd` 相当）の実現可能性の検証（イシュー #2880・親 #2841）。
//!
//! 契約の正は `docs/autodiff-functional-transforms-design.md` §2・§8・§11。本ファイルは
//! §8 の判定基準 1〜4 を、`Tape::backward_create_graph` が対象とする Op 区分ごとに確かめる。
//!
//! - 手順（§8）: `u` を追跡ありの葉、`s = y ⊙ u`（暗黙の総和射影 `uᵀy`）、
//!   `g = ∂s/∂x = Jᵀu`（`u` について線形）、`t = g ⊙ v`、`∂t/∂u = J·v`。
//! - 基準 1: 二階導関数が非ゼロの関数でも `J·v` に二階項が混ざらない（閉形式で確認）。
//! - 基準 2: `jacobian_ops::jacobian` の `J·v`（ホスト f64 蓄積）と `common::req2_close` で
//!   全要素突合する（新しい tolerance 定数は作らない）。
//! - 基準 3: 機構面。`child_var(&u)` が `Some`、`J ≢ 0` の区分で `grad(&x)` が勾配追跡あり、
//!   かつ `∂t/∂u` が `Some`（`None` を暗黙のゼロとして丸めない）。
//! - 基準 4: 対象外 Op は `Err(Backward)` で拒否され子テープは無変更。
//!
//! `jvp`／`jacfwd` の公開・内部 API は作らない。`double_vjp_probe` は本ファイル内に閉じた
//! 検証用ヘルパーである（`api_surface` の関数名インベントリ対象外の名前を使う）。
//! `pub(crate)` でしか作れない Op variant は `src/double_vjp_feasibility_tests.rs`（単体）が担う。
//! 実 CPU `BackendOps`・CUDA／Metal 実機との突合は対象外（設計 §10 の 4・11）。
//! 入力値は、`build_cgrads` と `grad.rs` が独立に実装する劣勾配規約のキンク
//! （0・clamp 境界・タイ）から離して選ぶ。

mod common;

use std::sync::Arc;

use fandhe_ai_autodiff::jacobian_ops::jacobian;
use fandhe_ai_autodiff::{AutodiffError, CustomFunction, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn new_tape() -> Tape {
    Tape::new_with_ops(common::naive_ops())
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test: shape とデータ長は一致させている")
}

fn numel(shape: &[usize]) -> usize {
    shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .expect("test: 要素数が overflow しない")
}

/// 符号混在の決定的な値列（0 から離れた値になるよう位相をずらす）。
fn seq(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.37).sin() * 1.5)
        .collect()
}

/// 全要素が正の決定的な値列（`sqrt`／`log*`／`pow` の定義域用）。
fn pos(n: usize, seed: f32) -> Vec<f32> {
    seq(n, seed).iter().map(|v| v.abs() + 0.4).collect()
}

type Built<'t> = Result<Var<'t>, AutodiffError>;

/// §8 の double-VJP を実行して `J·v` を返す検証用ヘルパー（公開 API ではない）。
/// `expect_tracked` が真なら基準 3（`grad(&x)` の追跡・`∂t/∂u` の `Some`）を assert する。
fn double_vjp_probe(
    tape: &Tape,
    y: &Var<'_>,
    x: &Var<'_>,
    v: &Tensor<f32>,
    expect_tracked: bool,
    ctx: &str,
) -> Vec<f32> {
    let y_shape = y.value().shape().to_vec();
    let len_before = tape.len();
    let u = tape.var(&t(seq(numel(&y_shape), 7.0), &y_shape));
    let s = y.mul(&u).unwrap();
    let child = new_tape();
    let cg = tape
        .backward_create_graph(&s, &child)
        .unwrap_or_else(|e| panic!("{ctx}: backward_create_graph が失敗: {e:?}"));
    // 親テープへ足されるのは u と mul の 2 ノードのみ（backward_create_graph は足さない）。
    assert_eq!(tape.len(), len_before + 2, "{ctx}: 親テープの増分");

    let cu = cg.child_var(&u).unwrap();
    assert!(cu.is_some(), "{ctx}: child_var(u) が None");
    let cu = cu.unwrap();
    // 勾配追跡の有無は `Var::requires_grad`（crate 内限定）を使えないため、挙動で判定する。
    // 追跡ありの区分では `grad(x)` が `Some` で、その子テープ上の backward が成功し、
    // `∂t/∂u` が `Some` になることを要求する（`None` を暗黙のゼロへ丸めない）。
    // 比較系（J ≡ 0）は `grad(x)` が `None`／追跡なし（backward が Err）／ゼロ勾配のいずれも許す。
    // J ≡ 0 の区分では追跡の有無を assert しない（数値が全ゼロで一致することのみ確認する）。
    let Some(g) = cg.grad(x).unwrap() else {
        assert!(!expect_tracked, "{ctx}: grad(x) が None");
        return vec![0.0; numel(&y_shape)];
    };
    let prod = g.mul(&child.var_no_grad(v)).unwrap();
    let ju = match child.backward(&prod) {
        Ok(grads) => grads.get(&cu).unwrap().map(|h| {
            assert_eq!(h.shape(), y_shape.as_slice(), "{ctx}: jv の shape");
            h.host_slice().into_owned()
        }),
        Err(e) => {
            assert!(
                !expect_tracked,
                "{ctx}: 子テープ backward が失敗（grad(x) が追跡なし）: {e:?}"
            );
            return vec![0.0; numel(&y_shape)];
        }
    };
    if expect_tracked {
        assert!(ju.is_some(), "{ctx}: ∂t/∂u が None");
    }
    let m = numel(&y_shape);
    ju.unwrap_or_else(|| vec![0.0; m])
}

/// double-VJP の `J·v` を、別テープの `jacobian` の `J·v`（f64 蓄積）と全要素突合する。
fn check<F>(shape: &[usize], xv: Vec<f32>, expect_tracked: bool, ctx: &str, build: F)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let n = numel(shape);
    let vv = seq(n, 5.0);
    let v = t(vv.clone(), shape);

    let tb = new_tape();
    let xb = tb.var(&t(xv.clone(), shape));
    let yb = build(&tb, &xb).unwrap_or_else(|e| panic!("{ctx}: build 失敗: {e:?}"));
    let jac = jacobian(&tb, &yb, &xb).unwrap();
    let m = numel(yb.value().shape());
    let jh = jac.host_slice().into_owned();
    assert_eq!(jh.len(), m * n, "{ctx}: jacobian 要素数");
    let expected: Vec<f64> = (0..m)
        .map(|i| {
            (0..n)
                .map(|k| f64::from(jh[i * n + k]) * f64::from(vv[k]))
                .sum()
        })
        .collect();

    let ta = new_tape();
    let xa = ta.var(&t(xv, shape));
    let ya = build(&ta, &xa).unwrap();
    let got = double_vjp_probe(&ta, &ya, &xa, &v, expect_tracked, ctx);
    assert_eq!(got.len(), expected.len(), "{ctx}: 要素数");
    for (i, (&a, &e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            common::req2_close(f64::from(a), e),
            "{ctx}[{i}]: double-VJP {a} vs jacobian·v {e}"
        );
    }
}

fn k<'t>(tp: &'t Tape) -> Var<'t> {
    c(tp, seq(6, 2.0), &[6])
}

fn c<'t>(tape: &'t Tape, data: Vec<f32>, shape: &[usize]) -> Var<'t> {
    tape.var_no_grad(&t(data, shape))
}

// ------------------------------------------------------------ 基準 1: 閉形式

#[test]
fn c1_closed_forms_have_no_second_order_term() {
    let xv = seq(5, 1.0);
    let vv = seq(5, 5.0);
    // y = x⊙x → 2x⊙v
    let tape = new_tape();
    let x = tape.var(&t(xv.clone(), &[5]));
    let y = x.mul(&x).unwrap();
    let got = double_vjp_probe(&tape, &y, &x, &t(vv.clone(), &[5]), true, "x*x");
    for i in 0..5 {
        let e = 2.0 * f64::from(xv[i]) * f64::from(vv[i]);
        assert!(common::req2_close(f64::from(got[i]), e), "x*x[{i}]");
    }

    // y = tanh(x) → (1 - tanh²x)⊙v
    let tape = new_tape();
    let x = tape.var(&t(xv.clone(), &[5]));
    let y = x.tanh();
    let got = double_vjp_probe(&tape, &y, &x, &t(vv.clone(), &[5]), true, "tanh");
    for i in 0..5 {
        let th = f64::from(xv[i]).tanh();
        let e = (1.0 - th * th) * f64::from(vv[i]);
        assert!(common::req2_close(f64::from(got[i]), e), "tanh[{i}]");
    }

    // y = relu(x) → [x>0]⊙v
    let tape = new_tape();
    let x = tape.var(&t(xv.clone(), &[5]));
    let y = x.relu();
    let got = double_vjp_probe(&tape, &y, &x, &t(vv.clone(), &[5]), true, "relu");
    for i in 0..5 {
        let e = if xv[i] > 0.0 { f64::from(vv[i]) } else { 0.0 };
        assert!(common::req2_close(f64::from(got[i]), e), "relu[{i}]");
    }

    // y = W x（線形）→ W v。g は x に依らないが u には依存する（追跡あり）。
    let wv = seq(12, 3.0);
    let tape = new_tape();
    let x = tape.var(&t(xv[..4].to_vec(), &[4, 1]));
    let w = tape.var_no_grad(&t(wv.clone(), &[3, 4]));
    let y = w.matmul(&x).unwrap();
    let got = double_vjp_probe(&tape, &y, &x, &t(vv[..4].to_vec(), &[4, 1]), true, "Wx");
    for r in 0..3 {
        let e: f64 = (0..4)
            .map(|k| f64::from(wv[r * 4 + k]) * f64::from(vv[k]))
            .sum();
        assert!(common::req2_close(f64::from(got[r]), e), "Wx[{r}]");
    }
}

// ------------------------------------------------------------ 基準 2・3: Op 区分

#[test]
fn c2_binary_add_mul_variants() {
    let xv = seq(6, 1.0);
    check(&[2, 3], xv.clone(), true, "add const", |tp, x| {
        x.add(&c(tp, seq(6, 2.0), &[2, 3]))
    });
    // bias パターン [m,n] + [n]（reduce_bias_grad_var 経路）。x を bias 側にする。
    check(&[3], seq(3, 1.0), true, "add bias(x が [n])", |tp, x| {
        c(tp, seq(6, 2.0), &[2, 3]).add(x)
    });
    check(&[2, 3], xv.clone(), true, "add [m,n]+[n]", |tp, x| {
        x.add(&c(tp, seq(3, 4.0), &[3]))
    });
    check(&[6], xv.clone(), true, "x*x", |_, x| x.mul(x));
    check(&[6], xv.clone(), true, "x*const", |tp, x| {
        x.mul(&c(tp, seq(6, 3.0), &[6]))
    });
    check(&[2, 3], xv.clone(), true, "broadcast mul", |tp, x| {
        x.mul(&c(tp, seq(3, 3.0), &[3]))
    });
}

#[test]
fn c2_activations_and_composites() {
    let xv = seq(6, 1.0);
    check(&[6], xv.clone(), true, "relu", |_, x| Ok(x.relu()));
    check(&[6], xv.clone(), true, "exp", |_, x| Ok(x.exp()));
    check(&[6], xv.clone(), true, "tanh", |_, x| Ok(x.tanh()));
    check(&[6], xv.clone(), true, "sigmoid", |_, x| Ok(x.sigmoid()));
    check(&[6], xv.clone(), true, "relu(tanh)", |_, x| {
        Ok(x.tanh().relu())
    });
    check(&[6], xv.clone(), true, "tanh(x)*x+exp(x)", |_, x| {
        x.tanh().mul(x)?.add(&x.exp())
    });
    // MLP 形: matmul → relu → matmul
    check(&[2, 3], xv.clone(), true, "mlp", |tp, x| {
        let w1 = c(tp, seq(12, 4.0), &[3, 4]);
        let w2 = c(tp, seq(8, 6.0), &[4, 2]);
        x.matmul(&w1)?.relu().matmul(&w2)
    });
}

#[test]
fn c2_reductions_and_views() {
    let xv = seq(6, 1.0);
    check(&[2, 3], xv.clone(), true, "sum all", |_, x| x.sum(None));
    check(&[2, 3], xv.clone(), true, "sum dim", |_, x| x.sum(Some(1)));
    check(&[2, 3], xv.clone(), true, "mean all", |_, x| x.mean(None));
    check(&[2, 3], xv.clone(), true, "mean dim", |_, x| {
        x.mean(Some(0))
    });
    check(&[2, 3], xv.clone(), true, "reshape", |_, x| {
        x.reshape(&[3, 2])
    });
    check(&[3], seq(3, 2.0), true, "broadcast_to", |_, x| {
        x.broadcast_to(&[2, 3])
    });
    check(&[2, 3], xv.clone(), true, "transpose", |_, x| {
        x.transpose(0, 1)
    });
    check(&[2, 3], xv.clone(), true, "permute", |_, x| {
        x.permute(&[1, 0])
    });
    check(&[2, 3], xv.clone(), true, "narrow", |_, x| {
        x.narrow(1, 1, 2)
    });
    check(&[2, 3], xv.clone(), true, "cat const", |tp, x| {
        Var::cat(&[*x, c(tp, seq(6, 2.0), &[2, 3])], 1)
    });
    check(&[2, 3], xv.clone(), true, "cat duplicate input", |_, x| {
        Var::cat(&[*x, *x], 0)
    });
    // permute で非 contiguous になった値に Op::Contiguous を積む経路（einsum）。
    check(&[2, 3], xv.clone(), true, "einsum transpose", |_, x| {
        Var::einsum("ji->ij", &[x])
    });
    check(&[2, 3], xv.clone(), true, "einsum matmul-like", |tp, x| {
        let w = c(tp, seq(12, 4.0), &[3, 4]);
        Var::einsum("ij,jk->ik", &[x, &w])
    });
}

#[test]
fn c2_matmul_both_operand_positions() {
    // x を左オペランド（da 分岐）と右オペランド（db 分岐）の両方で確認する。
    check(&[2, 3], seq(6, 1.0), true, "matmul x left", |tp, x| {
        x.matmul(&c(tp, seq(12, 2.0), &[3, 4]))
    });
    check(&[3, 4], seq(12, 1.0), true, "matmul x right", |tp, x| {
        c(tp, seq(6, 2.0), &[2, 3]).matmul(x)
    });
    check(
        &[3, 3],
        seq(9, 1.0),
        true,
        "matmul x both (x·x)",
        |_, x| x.matmul(x),
    );
}

#[test]
fn c2_where() {
    let mask = Tensor::new(vec![true, false, true, true, false, false], &[2, 3]).unwrap();
    check(&[2, 3], seq(6, 1.0), true, "where a=x", |tp, x| {
        Var::where_cond(&mask, x, &c(tp, seq(6, 2.0), &[2, 3]))
    });
    check(&[2, 3], seq(6, 1.0), true, "where b=x", |tp, x| {
        Var::where_cond(&mask, &c(tp, seq(6, 2.0), &[2, 3]), x)
    });
    check(&[2, 3], seq(6, 1.0), true, "where a=x,b=x*x", |_, x| {
        Var::where_cond(&mask, x, &x.mul(x)?)
    });
}

#[test]
fn c2_scalar_unary_public() {
    let xv = seq(6, 1.0);
    let pv = pos(6, 1.0);
    check(&[6], xv.clone(), true, "neg", |_, x| x.neg());
    check(&[6], xv.clone(), true, "abs", |_, x| x.abs());
    check(&[6], pv.clone(), true, "sqrt", |_, x| x.sqrt());
    check(&[6], pv.clone(), true, "log", |_, x| x.log());
    check(&[6], pv.clone(), true, "log2", |_, x| x.log2());
    check(&[6], pv.clone(), true, "log10", |_, x| x.log10());
    check(&[6], xv.clone(), true, "sin", |_, x| x.sin());
    check(&[6], xv.clone(), true, "cos", |_, x| x.cos());
    check(
        &[6],
        seq(6, 1.0).iter().map(|v| v * 0.5).collect(),
        true,
        "tan",
        |_, x| x.tan(),
    );
    check(&[6], xv.clone(), true, "silu", |_, x| x.silu());
    check(&[6], xv.clone(), true, "hardswish", |_, x| x.hardswish());
    check(&[6], xv.clone(), true, "leaky_relu", |_, x| {
        x.leaky_relu(0.1)
    });
    check(&[6], xv.clone(), true, "elu", |_, x| x.elu(1.0));
    check(&[6], xv.clone(), true, "softplus", |_, x| {
        x.softplus(1.0, 20.0)
    });
    check(&[6], xv.clone(), true, "clamp", |_, x| x.clamp(-0.8, 0.8));
    check(&[6], pv.clone(), true, "pow_scalar", |_, x| {
        x.pow_scalar(3.0)
    });
}

#[test]
fn c2_scalar_binary_public() {
    let xv = seq(6, 1.0);
    let pv = pos(6, 1.0);

    check(&[6], xv.clone(), true, "sub x-c", |tp, x| x.sub(&k(tp)));
    check(&[6], xv.clone(), true, "sub c-x", |tp, x| k(tp).sub(x));
    check(&[6], xv.clone(), true, "div x/c", |tp, x| {
        x.div(&c(tp, pos(6, 2.0), &[6]))
    });
    check(&[6], pv.clone(), true, "div c/x", |tp, x| {
        c(tp, seq(6, 2.0), &[6]).div(x)
    });
    check(&[6], pv.clone(), true, "pow base=x", |tp, x| {
        x.pow(&c(tp, vec![2.0, 1.5, 3.0, 0.5, 2.5, 1.0], &[6]))
    });
    check(&[6], xv.clone(), true, "pow exponent=x", |tp, x| {
        c(tp, pos(6, 2.0), &[6]).pow(x)
    });
}

#[test]
fn c2_comparisons_are_zero_jacobian() {
    // 比較系は J ≡ 0。grad(x) が追跡なし（または None）の分岐を通し、全要素ゼロで一致する。
    let xv = seq(6, 1.0);

    check(&[6], xv.clone(), false, "gt", |tp, x| x.gt(&k(tp)));
    check(&[6], xv.clone(), false, "ge", |tp, x| x.ge(&k(tp)));
    check(&[6], xv.clone(), false, "lt", |tp, x| x.lt(&k(tp)));
    check(&[6], xv.clone(), false, "le", |tp, x| x.le(&k(tp)));
    check(&[6], xv.clone(), false, "eq", |tp, x| x.eq(&k(tp)));
    check(&[6], xv.clone(), false, "ne", |tp, x| x.ne(&k(tp)));
}

// ------------------------------------------------------------ 基準 4: 非対象 Op

struct IdentityCustomFn;

impl CustomFunction for IdentityCustomFn {
    fn name(&self) -> &str {
        "identity_custom_fn"
    }

    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }

    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        Ok(inputs[0].contiguous())
    }

    fn backward(
        &self,
        _inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        if !requires_grad[0] {
            return Ok(vec![None]);
        }
        Ok(vec![Some(upstream.contiguous())])
    }
}

/// 手順 1〜3 までを実行し、`backward_create_graph` が `Err(Backward)` で拒否すること・
/// 子テープが無変更であること・拒否呼び出しで親テープが変化しないことを確かめる。
fn assert_rejected<'t>(tape: &'t Tape, y: &Var<'t>, ctx: &str) {
    let y_shape = y.value().shape().to_vec();
    let u = tape.var(&t(seq(numel(&y_shape), 7.0), &y_shape));
    let s = y.mul(&u).unwrap();
    let child = new_tape();
    let len_before = tape.len();
    let err = tape.backward_create_graph(&s, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)), "{ctx}: {err:?}");
    assert!(child.is_empty(), "{ctx}: 子テープが無変更でない");
    assert_eq!(tape.len(), len_before, "{ctx}: 親テープが変化した");
}

#[test]
fn c4_unsupported_ops_are_rejected_fail_closed() {
    // Op::Max
    let tape = new_tape();
    let x = tape.var(&t(seq(6, 1.0), &[2, 3]));
    let y = x.max(None).unwrap();
    assert_rejected(&tape, &y, "max");

    // rank 3 の MatMul
    let tape = new_tape();
    let x = tape.var(&t(seq(8, 1.0), &[2, 2, 2]));
    let w = tape.var_no_grad(&t(seq(8, 2.0), &[2, 2, 2]));
    let y = x.matmul(&w).unwrap();
    assert_rejected(&tape, &y, "matmul rank3");

    // Op::Custom
    let tape = new_tape();
    let x = tape.var(&t(seq(3, 1.0), &[3]));
    let y = tape.custom(Arc::new(IdentityCustomFn), &[x]).unwrap();
    assert_rejected(&tape, &y, "custom");

    // 対象外の ScalarUnary（Gelu・Selu）
    let tape = new_tape();
    let x = tape.var(&t(seq(4, 1.0), &[4]));
    let y = x.gelu().unwrap();
    assert_rejected(&tape, &y, "gelu");
    let tape = new_tape();
    let x = tape.var(&t(seq(4, 1.0), &[4]));
    let y = x.selu().unwrap();
    assert_rejected(&tape, &y, "selu");
}
