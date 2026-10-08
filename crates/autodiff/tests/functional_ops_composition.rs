//! `vmap`（#2876）の出力を含む損失への `vjp`（#2874）・`hvp`（#2875）の合成検証
//! （イシュー #2878・親 #2841。設計 `docs/autodiff-functional-transforms-design.md`
//! §5・§6・§7・§10 の分解案 5）。
//!
//! 役割: `vmap` は呼び出し側テープ上の微分可能な `Var` を返す契約（§5）であり、その出力を
//! 損失の一部に組み込んで 1 階（`vjp`）・2 階（`hvp` = `backward_create_graph` の子テープ経由）
//! を取れることを、「vmap を手で展開した計算」との突合で固定する。
//!
//! 参照の 3 系統:
//! - R1: 同一テープ上の batched 等価式（vmap を使わない）。
//! - R2: 同一テープ上の明示ループ展開（`unbind` → 各スライスへ f → `Var::stack`）。
//!   `Var::contiguous` が `pub(crate)` のため、in_dim=0 かつ f の出力が contiguous な
//!   ケースに限る。
//! - R3: スライスごとに独立テープを作る参照。vjp は in_dim=0 で常に、hvp はバッチ方向に
//!   分離可能な損失（H がブロック対角）に限る。`unbind`／`stack` のグラフに依存しない。
//!
//! 判定は REQ-2 統一複合判定（`common::req2_close`。tolerance 定数は新設しない）。bit 一致は
//! 契約にしない（§11-2）。子テープ経路は 1 階 VJP と bit 同一を主張しない。
//!
//! 扱わないもの: `vmap(grad)`（§5・§11-6 により除外。テストも否定テストも置かない）・
//! double-VJP（#2880）・実 CPU／CUDA／Metal（#2877・#2881）・facade 公開。
//!
//! `hvp` は `backward_create_graph` を通るため、x から損失までの経路は
//! `Op::supports_create_graph()` が真の Op（四則・relu/exp/tanh・sum・reshape・rank 2 matmul・
//! transpose・narrow・concat 等）だけで組む。`max` は fail-closed ケース（C7）専用。

mod common;

use fandhe_ai_autodiff::functional_ops::{hvp, vjp, vmap};
use fandhe_ai_autodiff::jacobian_ops::hessian;
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

type R<'t> = Result<Var<'t>, AutodiffError>;

fn new_tape() -> Tape {
    Tape::new_with_ops(common::naive_ops())
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test: shape とデータ長は一致させている")
}

fn host(x: &Tensor<f32>) -> Vec<f32> {
    x.host_slice().into_owned()
}

fn bits(x: &Tensor<f32>) -> Vec<u32> {
    host(x).iter().map(|f| f.to_bits()).collect()
}

fn seq(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.37).sin() * 1.5)
        .collect()
}

fn assert_close_vec(actual: &[f32], expected: &[f32], ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: 要素数");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            common::req2_close(f64::from(a), f64::from(e)),
            "{ctx}[{i}]: {a} vs {e}"
        );
    }
}

fn shape_of(v: &Var<'_>) -> Vec<usize> {
    v.to_tensor().shape().to_vec()
}

/// `jacobian_ops::hessian`（facade `Tape::hessian` の実体）の `H·v`。f64 で蓄積する。
fn hessian_times_v(tape: &Tape, loss: &Var<'_>, x: &Var<'_>, v: &[f32]) -> Vec<f32> {
    let child = new_tape();
    let h = host(&hessian(tape, loss, x, &child).unwrap());
    let n = v.len();
    (0..n)
        .map(|j| {
            (0..n)
                .map(|k| f64::from(h[j * n + k]) * f64::from(v[k]))
                .sum::<f64>() as f32
        })
        .collect()
}

/// R2: 明示ループ展開（in_dim=0）。
fn unrolled<'t>(x: &Var<'t>, f: &dyn Fn(&Var<'t>) -> R<'t>) -> R<'t> {
    let outs = x.unbind(0)?.iter().map(f).collect::<Result<Vec<_>, _>>()?;
    Var::stack(&outs, 0)
}

fn weight(tape: &Tape, rows: usize, cols: usize, seed: f32) -> Var<'_> {
    tape.var_no_grad(&t(seq(rows * cols, seed), &[rows, cols]))
}

/// 1 ケース分の合成検証。`f` はスライス関数、`r1` は同値の batched 式、`loss_of` は
/// 出力（と入力）からスカラー損失を作る。`r2`／`separable` で参照系統を絞る。
struct Case {
    name: &'static str,
    shape: Vec<usize>,
    in_dim: usize,
    r2: bool,
    separable: bool,
}

fn run_case(
    c: &Case,
    f: impl for<'t> Fn(&'t Tape, &Var<'t>) -> R<'t>,
    r1: impl for<'t> Fn(&'t Tape, &Var<'t>) -> R<'t>,
    loss_of: impl for<'t> Fn(&'t Tape, &Var<'t>, &Var<'t>) -> R<'t>,
) {
    let (name, shape, in_dim) = (c.name, &c.shape[..], c.in_dim);
    let n: usize = shape.iter().product();
    let xv = seq(n, 1.0);
    let vv = seq(n, 11.0);

    // ---- vjp（1 階）
    let tape = new_tape();
    let x = tape.var(&t(xv.clone(), shape));
    let y = vmap(&tape, &x, in_dim, |s| f(&tape, s)).unwrap();
    let ys = shape_of(&y);
    let ny: usize = ys.iter().product();
    let u = t(seq(ny, 5.0), &ys);
    let g = host(&vjp(&tape, &y, &x, &u).unwrap());

    let y1 = r1(&tape, &x).unwrap();
    assert_eq!(shape_of(&y1), ys, "{name}: R1 shape");
    let g1 = host(&vjp(&tape, &y1, &x, &u).unwrap());
    assert_close_vec(&g, &g1, &format!("{name} vjp R1"));

    if c.r2 {
        let y2 = unrolled(&x, &|s| f(&tape, s)).unwrap();
        assert_eq!(shape_of(&y2), ys, "{name}: R2 shape");
        let g2 = host(&vjp(&tape, &y2, &x, &u).unwrap());
        assert_close_vec(&g, &g2, &format!("{name} vjp R2"));
    }
    if in_dim == 0 {
        // R3: スライスごとの独立テープ。余接は先頭軸方向の連続チャンク。
        let b = shape[0];
        let (sl, ul) = (n / b, ny / b);
        let uh = host(&u);
        let mut g3 = Vec::with_capacity(n);
        for i in 0..b {
            let ti = new_tape();
            let xi = ti.var(&t(xv[i * sl..(i + 1) * sl].to_vec(), &shape[1..]));
            let yi = f(&ti, &xi).unwrap();
            let ui = t(uh[i * ul..(i + 1) * ul].to_vec(), &shape_of(&yi));
            g3.extend(host(&vjp(&ti, &yi, &xi, &ui).unwrap()));
        }
        assert_close_vec(&g, &g3, &format!("{name} vjp R3"));
    }

    // ---- hvp（2 階）
    let vt = t(vv.clone(), shape);
    let loss = loss_of(&tape, &y, &x).unwrap();
    let child = new_tape();
    let h = host(&hvp(&tape, &loss, &x, &vt, &child).unwrap());
    assert_close_vec(
        &h,
        &hessian_times_v(&tape, &loss, &x, &vv),
        &format!("{name} hvp = H·v"),
    );

    let loss1 = loss_of(&tape, &y1, &x).unwrap();
    let child1 = new_tape();
    let h1 = host(&hvp(&tape, &loss1, &x, &vt, &child1).unwrap());
    assert_close_vec(&h, &h1, &format!("{name} hvp R1"));
    assert_close_vec(
        &h,
        &hessian_times_v(&tape, &loss1, &x, &vv),
        &format!("{name} H·v R1"),
    );

    if c.r2 {
        let y2 = unrolled(&x, &|s| f(&tape, s)).unwrap();
        let loss2 = loss_of(&tape, &y2, &x).unwrap();
        let child2 = new_tape();
        let h2 = host(&hvp(&tape, &loss2, &x, &vt, &child2).unwrap());
        assert_close_vec(&h, &h2, &format!("{name} hvp R2"));
    }
    if c.separable && in_dim == 0 {
        let sl = n / shape[0];
        let mut h3 = Vec::with_capacity(n);
        for i in 0..shape[0] {
            let ti = new_tape();
            let xi = ti.var(&t(xv[i * sl..(i + 1) * sl].to_vec(), &shape[1..]));
            let yi = f(&ti, &xi).unwrap();
            let li = loss_of(&ti, &yi, &xi).unwrap();
            let ci = new_tape();
            let vi = t(vv[i * sl..(i + 1) * sl].to_vec(), &shape[1..]);
            h3.extend(host(&hvp(&ti, &li, &xi, &vi, &ci).unwrap()));
        }
        assert_close_vec(&h, &h3, &format!("{name} hvp R3"));
    }
}

/// バッチ方向に分離可能な損失（要素ごとの関数の総和）。
fn sum_loss<'t>(_: &'t Tape, y: &Var<'t>, _x: &Var<'t>) -> R<'t> {
    y.mul(y)?.add(y)?.sum(None)
}

// ---------------------------------------------------------------- C1・C3

#[test]
fn c1_c3_elementwise() {
    let c = Case {
        name: "elementwise",
        shape: vec![4, 3],
        in_dim: 0,
        r2: true,
        separable: true,
    };
    run_case(&c, |_, s| s.tanh().mul(s), |_, x| x.tanh().mul(x), sum_loss);
}

#[test]
fn c1_c3_matmul_with_no_grad_weight() {
    let c = Case {
        name: "matmul",
        shape: vec![4, 3],
        in_dim: 0,
        r2: true,
        separable: true,
    };
    run_case(
        &c,
        |tape, s| {
            s.reshape(&[1, 3])?
                .matmul(&weight(tape, 3, 5, 3.0))?
                .tanh()
                .reshape(&[5])
        },
        |tape, x| Ok(x.matmul(&weight(tape, 3, 5, 3.0))?.tanh()),
        sum_loss,
    );
}

#[test]
fn c1_c3_scalar_per_slice_output() {
    let c = Case {
        name: "scalar-out",
        shape: vec![4, 3],
        in_dim: 0,
        r2: true,
        separable: true,
    };
    run_case(
        &c,
        |_, s| s.mul(s)?.sum(None),
        |_, x| x.mul(x)?.sum(Some(1)),
        sum_loss,
    );
}

#[test]
fn c1_c3_rank0_slices() {
    let c = Case {
        name: "rank0-slice",
        shape: vec![4],
        in_dim: 0,
        r2: true,
        separable: true,
    };
    run_case(&c, |_, s| s.tanh().mul(s), |_, x| x.tanh().mul(x), sum_loss);
}

#[test]
fn c1_c3_closed_forms() {
    let tape = new_tape();
    let xv = seq(12, 2.0);
    let uv = seq(12, 3.0);
    let vv = seq(12, 4.0);
    let x = tape.var(&t(xv.clone(), &[4, 3]));
    // f = s⊙s なら vjp = 2x⊙u
    let y = vmap(&tape, &x, 0, |s| s.mul(s)).unwrap();
    let g = host(&vjp(&tape, &y, &x, &t(uv.clone(), &[4, 3])).unwrap());
    let e: Vec<f32> = xv.iter().zip(&uv).map(|(a, u)| 2.0 * a * u).collect();
    assert_close_vec(&g, &e, "vjp 2x⊙u");
    // f = s⊙s⊙s の総和なら hvp = 6x⊙v
    let y3 = vmap(&tape, &x, 0, |s| s.mul(s)?.mul(s)).unwrap();
    let loss = y3.sum(None).unwrap();
    let child = new_tape();
    let h = host(&hvp(&tape, &loss, &x, &t(vv.clone(), &[4, 3]), &child).unwrap());
    let e: Vec<f32> = xv.iter().zip(&vv).map(|(a, v)| 6.0 * a * v).collect();
    assert_close_vec(&h, &e, "hvp 6x⊙v");
}

// ---------------------------------------------------------------- C2

#[test]
fn c2_vmap_output_as_part_of_output() {
    let tape = new_tape();
    let x = tape.var(&t(seq(12, 1.0), &[4, 3]));
    let z = vmap(&tape, &x, 0, |s| s.tanh().mul(s))
        .unwrap()
        .add(&x.exp())
        .unwrap();
    let z1 = x.tanh().mul(&x).unwrap().add(&x.exp()).unwrap();
    let u = t(seq(12, 6.0), &[4, 3]);
    let g = host(&vjp(&tape, &z, &x, &u).unwrap());
    assert_close_vec(&g, &host(&vjp(&tape, &z1, &x, &u).unwrap()), "R1");

    // jacobian の転置積 Jᵀu
    let j = host(&fandhe_ai_autodiff::jacobian_ops::jacobian(&tape, &z, &x).unwrap());
    let uh = host(&u);
    let e: Vec<f32> = (0..12)
        .map(|k| {
            (0..12)
                .map(|i| f64::from(j[i * 12 + k]) * f64::from(uh[i]))
                .sum::<f64>() as f32
        })
        .collect();
    assert_close_vec(&g, &e, "Jᵀu");
}

// ---------------------------------------------------------------- C4

#[test]
fn c4_non_separable_and_partial_expression() {
    // vmap 出力が部分式として入る損失（y·W2 の tanh 総和 + x の exp 総和）。損失が rank 2 の y を
    // 前提とするため R3（スライス単位）は使わない。
    let c = Case {
        name: "partial",
        shape: vec![4, 3],
        in_dim: 0,
        r2: true,
        separable: false,
    };
    run_case(
        &c,
        |tape, s| {
            s.reshape(&[1, 3])?
                .matmul(&weight(tape, 3, 5, 3.0))?
                .tanh()
                .reshape(&[5])
        },
        |tape, x| Ok(x.matmul(&weight(tape, 3, 5, 3.0))?.tanh()),
        |tape, y, x| {
            y.matmul(&weight(tape, 5, 2, 7.0))?
                .tanh()
                .sum(None)?
                .add(&x.exp().sum(None)?)
        },
    );
    // バッチ間を結合する (Σy)² 項。H の非対角ブロックが現れる（R3 は使わない）。
    let c = Case {
        name: "coupled",
        shape: vec![4, 3],
        in_dim: 0,
        r2: true,
        separable: false,
    };
    run_case(
        &c,
        |_, s| s.tanh().mul(s),
        |_, x| x.tanh().mul(x),
        |_, y, _| {
            let s = y.sum(None)?;
            s.mul(&s)?.add(&y.mul(y)?.sum(None)?)
        },
    );
}

// ---------------------------------------------------------------- C5

#[test]
fn c5_in_dim_1_copy_path() {
    let c = Case {
        name: "in_dim=1",
        shape: vec![3, 4],
        in_dim: 1,
        r2: false,
        separable: false,
    };
    run_case(
        &c,
        |_, s| s.tanh().mul(s),
        |_, x| {
            let xt = x.transpose(0, 1)?;
            xt.tanh().mul(&xt)
        },
        sum_loss,
    );
}

#[test]
fn c5_non_contiguous_closure_output() {
    let c = Case {
        name: "non-contiguous",
        shape: vec![2, 3, 4],
        in_dim: 0,
        r2: false,
        separable: true,
    };
    run_case(
        &c,
        // transpose を最後に適用し、戻り値を非 contiguous に保つ（vmap 内の `contiguous()` が
        // 恒等経路にならず `Op::Contiguous` の vjp／hvp を通る）。
        |_, s| {
            let out = s.tanh().transpose(0, 1)?;
            assert!(
                !out.value().is_contiguous(),
                "クロージャ出力が非 contiguous であること"
            );
            Ok(out)
        },
        |_, x| x.tanh().transpose(1, 2),
        sum_loss,
    );
}

#[test]
fn c5_batch_of_one() {
    let c = Case {
        name: "b=1",
        shape: vec![1, 3],
        in_dim: 0,
        r2: true,
        separable: true,
    };
    run_case(&c, |_, s| s.tanh().mul(s), |_, x| x.tanh().mul(x), sum_loss);
}

// ---------------------------------------------------------------- C6

#[test]
fn c6_stays_differentiable_and_side_effects() {
    let tape = new_tape();
    let xt = t(seq(12, 1.0), &[4, 3]);
    let x = tape.var(&xt);
    let y = vmap(&tape, &x, 0, |s| s.tanh().mul(s)).unwrap();
    let y_bits = bits(&y.to_tensor());
    let x_bits = bits(&x.to_tensor());

    // vjp は親へちょうど 2 ノード足す
    let len0 = tape.len();
    let u = t(seq(12, 6.0), &[4, 3]);
    vjp(&tape, &y, &x, &u).unwrap();
    assert_eq!(tape.len(), len0 + 2, "vjp の補助ノード");

    // hvp は親へノードを足さず、同一入力で 2 回回すと bit 一致（同一実行内の再現性）
    let loss = y.sum(None).unwrap();
    let len1 = tape.len();
    let vt = t(seq(12, 9.0), &[4, 3]);
    let run = || {
        let child = new_tape();
        bits(&hvp(&tape, &loss, &x, &vt, &child).unwrap())
    };
    let (a, b) = (run(), run());
    assert_eq!(a, b, "hvp 決定性");
    assert_eq!(tape.len(), len1, "hvp は親テープを変えない");

    // 値は不変・合成後も通常の backward が R1 と一致
    assert_eq!(bits(&y.to_tensor()), y_bits);
    assert_eq!(bits(&x.to_tensor()), x_bits);
    let g = host(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
    let r1 = x.tanh().mul(&x).unwrap().sum(None).unwrap();
    let g1 = host(tape.backward(&r1).unwrap().get(&x).unwrap().unwrap());
    assert_close_vec(&g, &g1, "backward after vjp/hvp");
}

// ---------------------------------------------------------------- C7

#[test]
fn c7_fail_closed_on_composition() {
    let tape = new_tape();
    let x = tape.var(&t(seq(12, 1.0), &[4, 3]));
    // create_graph 対象外の Op（max）を含むクロージャ: vmap と 1 階 vjp は成功し hvp は拒否。
    let y = vmap(&tape, &x, 0, |s| s.max(None)).unwrap();
    let u = t(seq(4, 2.0), &[4]);
    vjp(&tape, &y, &x, &u).unwrap();
    let loss = y.sum(None).unwrap();
    let len = tape.len();
    let child = new_tape();
    let err = hvp(&tape, &loss, &x, &t(seq(12, 3.0), &[4, 3]), &child);
    assert!(
        matches!(err, Err(AutodiffError::Backward(_))),
        "対象外 Op は Backward エラー: {err:?}"
    );
    assert!(child.is_empty(), "拒否時に子テープへ書き込まない");
    assert_eq!(tape.len(), len, "親テープ不変");

    // vmap 出力の shape と異なる余接（in_dim=1 の出力 [4, 3] へ入力レイアウト [3, 4]）。
    let tape = new_tape();
    let x = tape.var(&t(seq(12, 1.0), &[3, 4]));
    let y = vmap(&tape, &x, 1, |s| s.tanh().mul(s)).unwrap();
    assert_eq!(shape_of(&y), vec![4, 3]);
    let len = tape.len();
    let err = vjp(&tape, &y, &x, &t(seq(12, 2.0), &[3, 4]));
    assert!(
        matches!(err, Err(AutodiffError::Shape(_))),
        "余接 shape 不一致: {err:?}"
    );
    assert_eq!(tape.len(), len, "拒否時にテープ不変");
}
