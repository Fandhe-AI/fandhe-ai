//! facade `Tape::jvp`・`Tape::jacfwd`（イシュー #2956。決定記録
//! `docs/autodiff-functional-transforms-design.md` §27・§28）の CPU 結合テスト。
//!
//! 属性なし（CI で実行）: facade 経由の結果を同じ facade の `Tape::jacobian`（reverse-mode 行ごと）と
//! 接ベクトルのホスト側積（f64 蓄積）に突き合わせる。対象 Op は `supports_create_graph()` が真の
//! mul・tanh・matmul・sum・ブロードキャスト add で、非正方形と rank 0 を含む。数値の比較は REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`）を使い、tolerance は追加・変更しない。
//! 入口検査のエラー伝播・非対象 Op／空でない子テープの拒否・追跡なし `output` の非対称（`hvp` は `Err`）も固定する。
//!
//! 複数入力 `Tape::jvp_multi`（イシュー #3064・決定記録 §30）も同ファイルで CPU 結合テストする。参照は入力ごとの
//! `Tape::jacobian` と接ベクトルの積を全入力にまたがり f64 で一括合計し最後に 1 回だけ f32 へ丸めた値
//! （§30.2。入力ごとに f32 で足すと丸め順序が変わるため単一版の `reference_jv` は流用しない）。
//! 固定するエラー契約: 長さ不一致・空配列・別テープ・追跡なし入力・重複指定・shape 不一致・検査順序（§30.2 ①〜⑥）、
//! 非対象 Op（`max`）・空でない子テープ（`Backward`）、追跡なし `output` の全ゼロ。親テープに残る 2 ノードの副作用は
//! facade からノード数を観測する公開 API がないため内部テスト（`functional_ops_jvp_tests.rs` の M5）に任せる。
//! 複数入力の実機 parity は `functional_transforms_jvp_multi_backend_parity.rs`（`#[ignore]`）が担当する。
//!
//! 実機（CUDA／Metal）の parity は #2942 の既存 `#[ignore]` テスト
//! （`functional_ops_jvp_backend_parity.rs`）が担当し、本ファイルは facade 層を CPU で担当する。
//! 本ファイルのテスト関数名は宣言インベントリ（`api_surface.rs`。`crates/*/src` のみ走査）の対象外。

use fandhe_ai::{AutodiffError, ShapeError, Tape, Tensor, Var, tape};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn host(x: &Tensor<f32>) -> Vec<f32> {
    x.host_slice().into_owned()
}

/// `jacobian`（形状 `output.shape ++ input.shape`）と接ベクトルの積を f64 で蓄積した参照値。
fn reference_jv(jac: &[f32], v: &[f32]) -> Vec<f32> {
    let n = v.len();
    assert_eq!(jac.len() % n, 0);
    jac.chunks(n)
        .map(|row| {
            row.iter()
                .zip(v)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>() as f32
        })
        .collect()
}

/// `build` が作る `output` について `jvp`／`jacfwd` を `jacobian` と突き合わせる。
fn check_against_jacobian<F>(
    label: &str,
    x_data: Vec<f32>,
    x_shape: &[usize],
    v: Vec<f32>,
    build: F,
) where
    F: for<'a> Fn(&'a Tape, &Var<'a>) -> Var<'a>,
{
    let t0 = tape();
    let x = t0.var(&t(x_data, x_shape));
    let y = build(&t0, &x);
    let jac_t = t0.jacobian(&y, &x).expect("jacobian");
    let jac = host(&jac_t);
    let tangent = t(v.clone(), x_shape);

    let c1 = tape();
    let jv = t0.jvp(&y, &x, &tangent, &c1).expect("jvp");
    assert_eq!(jv.shape(), y.to_tensor().shape(), "{label}: jvp shape");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("{label}: jvp"),
        &host(&jv),
        &reference_jv(&jac, &v),
    );

    let c2 = tape();
    let jf = t0.jacfwd(&y, &x, &c2).expect("jacfwd");
    assert_eq!(jf.shape(), jac_t.shape(), "{label}: jacfwd shape");
    fandhe_ai_backend_cpu::parity::assert_parity(&format!("{label}: jacfwd"), &host(&jf), &jac);
}

#[test]
fn jvp_and_jacfwd_match_jacobian_on_supported_ops() {
    // mul（要素積）。
    check_against_jacobian(
        "mul",
        vec![1.0, 2.0, 3.0],
        &[3],
        vec![1.0, 0.5, 2.0],
        |_, x| x.mul(x).unwrap(),
    );
    // 非正方形 matmul → tanh（入力 [2, 3]・出力 [2, 4]）。
    let w = (0..12).map(|i| 0.1 * (i as f32) - 0.5).collect::<Vec<_>>();
    check_against_jacobian(
        "matmul_tanh",
        vec![0.3, -0.7, 1.1, 0.5, 0.2, -0.4],
        &[2, 3],
        vec![1.0, -0.5, 0.25, 0.0, 2.0, -1.5],
        move |tp, x| {
            let w = tp.var_no_grad(&t(w.clone(), &[3, 4]));
            x.matmul(&w).unwrap().tanh()
        },
    );
    // sum（スカラー出力）。
    check_against_jacobian(
        "sum",
        vec![0.5, -1.0, 2.0, 0.25],
        &[2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
        |_, x| x.mul(x).unwrap().sum(None).unwrap(),
    );
    // ブロードキャスト add（[2, 3] + [3]）の入力側は [3] のバイアス。
    check_against_jacobian(
        "broadcast_add",
        vec![0.1, -0.2, 0.3],
        &[3],
        vec![1.0, -1.0, 0.5],
        |tp, b| {
            let a = tp.var_no_grad(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
            a.add(b).unwrap().tanh()
        },
    );
    // rank 0。
    check_against_jacobian("rank0", vec![1.5], &[], vec![2.0], |_, x| x.mul(x).unwrap());
}

#[test]
fn input_validation_errors_propagate() {
    let t0 = tape();
    let t1 = tape();
    let x = t0.var(&t(vec![1.0, 2.0], &[2]));
    let y = x.mul(&x).unwrap();
    let v = t(vec![1.0, 1.0], &[2]);

    // 別テープの Var。
    let foreign = t1.var(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        t0.jvp(&y, &foreign, &v, &tape()),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        t0.jacfwd(&y, &foreign, &tape()),
        Err(AutodiffError::TapeMismatch)
    ));

    // 追跡なしの input。
    let nograd = t0.var_no_grad(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        t0.jvp(&y, &nograd, &v, &tape()),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    assert!(matches!(
        t0.jacfwd(&y, &nograd, &tape()),
        Err(AutodiffError::GradientTrackingDisabled)
    ));

    // tangent の shape 不一致（ブロードキャストしない）。
    let bad = t(vec![1.0, 1.0, 1.0], &[3]);
    assert!(matches!(
        t0.jvp(&y, &x, &bad, &tape()),
        Err(AutodiffError::Shape(_))
    ));
}

#[test]
fn unsupported_ops_and_dirty_child_tape_are_rejected() {
    let t0 = tape();
    let x = t0.var(&t(vec![1.0, 3.0, 2.0], &[3]));
    let v = t(vec![1.0, 1.0, 1.0], &[3]);

    // supports_create_graph() が偽の Op（max）を経由する出力は型付きの Err。
    let m = x.max(None).unwrap();
    assert!(
        t0.jvp(&m, &x, &t(vec![1.0, 1.0, 1.0], &[3]), &tape())
            .is_err()
    );
    assert!(t0.jacfwd(&m, &x, &tape()).is_err());

    // 空でない子テープは拒否される。
    let y = x.mul(&x).unwrap();
    let dirty = tape();
    let _leaf = dirty.var(&t(vec![0.0], &[1]));
    assert!(matches!(
        t0.jvp(&y, &x, &v, &dirty),
        Err(AutodiffError::Backward(_))
    ));
}

#[test]
fn untracked_output_is_zero_unlike_hvp() {
    let t0 = tape();
    let x = t0.var(&t(vec![1.0, 2.0], &[2]));
    let c = t0.var_no_grad(&t(vec![3.0, 4.0], &[2]));
    let v = t(vec![1.0, 1.0], &[2]);
    let untracked = c.mul(&c).unwrap();

    // 追跡なし output は jvp／jacfwd とも全ゼロの Ok（Tape::jacobian／Tape::vjp と同じ）。
    let jv = t0.jvp(&untracked, &x, &v, &tape()).unwrap();
    assert_eq!(jv.shape(), &[2]);
    assert_eq!(host(&jv), vec![0.0, 0.0]);
    let jf = t0.jacfwd(&untracked, &x, &tape()).unwrap();
    assert_eq!(jf.shape(), &[2, 2]);
    assert_eq!(host(&jf), vec![0.0; 4]);

    // Tape::hvp は追跡なし loss に Err を返す（非対称）。
    let loss = untracked.sum(None).unwrap();
    assert!(t0.hvp(&loss, &x, &v, &tape()).is_err());
}

/// 複数入力版の参照: `Σ_i J_i·v_i` を全入力にまたがり f64 で蓄積し、最後に 1 回だけ f32 へ丸める（§30.2）。
fn reference_multi_jv(jacs: &[Vec<f32>], vs: &[Vec<f32>]) -> Vec<f32> {
    assert_eq!(jacs.len(), vs.len());
    let m = jacs[0].len() / vs[0].len();
    (0..m)
        .map(|r| {
            let mut acc = 0.0f64;
            for (jac, v) in jacs.iter().zip(vs) {
                let n = v.len();
                for k in 0..n {
                    acc += f64::from(jac[r * n + k]) * f64::from(v[k]);
                }
            }
            acc as f32
        })
        .collect()
}

/// `build` が作る `output` の複数入力 JVP を、入力ごとの `jacobian` 由来の参照と突き合わせる。
fn check_multi_against_jacobian<F>(
    label: &str,
    xs: &[(Vec<f32>, Vec<usize>)],
    vs: &[Vec<f32>],
    build: F,
) where
    F: for<'a> Fn(&'a Tape, &[Var<'a>]) -> Var<'a>,
{
    let t0 = tape();
    let leaves: Vec<Var<'_>> = xs.iter().map(|(d, s)| t0.var(&t(d.clone(), s))).collect();
    let y = build(&t0, &leaves);
    let jacs: Vec<Vec<f32>> = leaves
        .iter()
        .map(|x| host(&t0.jacobian(&y, x).expect("jacobian")))
        .collect();
    assert!(
        jacs.iter().any(|j| j.iter().any(|v| *v != 0.0)),
        "{label}: 参照が全ゼロでは突合の意味がない"
    );
    let tangents: Vec<Tensor<f32>> = xs
        .iter()
        .zip(vs)
        .map(|((_, s), v)| t(v.clone(), s))
        .collect();
    let in_refs: Vec<&Var<'_>> = leaves.iter().collect();
    let tg_refs: Vec<&Tensor<f32>> = tangents.iter().collect();
    let out = t0
        .jvp_multi(&y, &in_refs, &tg_refs, &tape())
        .expect("jvp_multi");
    assert_eq!(out.shape(), y.to_tensor().shape(), "{label}: shape");
    fandhe_ai_backend_cpu::parity::assert_parity(
        label,
        &host(&out),
        &reference_multi_jv(&jacs, vs),
    );
}

#[test]
fn multi_input_matches_jacobian_sum_on_supported_ops() {
    // mul（rustdoc の doctest と同じ値。手計算 x2⊙v1 + x1⊙v2 = [4.0, 4.5, 9.0] とも照合）。
    let xs = vec![
        (vec![1.0, 2.0, 3.0], vec![3]),
        (vec![2.0, 0.5, 1.5], vec![3]),
    ];
    let vs = vec![vec![1.0, 0.5, 2.0], vec![0.0, 1.0, -1.0]];
    check_multi_against_jacobian("multi mul", &xs, &vs, |_, l| l[0].mul(&l[1]).unwrap());
    {
        let t0 = tape();
        let x1 = t0.var(&t(xs[0].0.clone(), &[3]));
        let x2 = t0.var(&t(xs[1].0.clone(), &[3]));
        let y = x1.mul(&x2).unwrap();
        let (v1, v2) = (t(vs[0].clone(), &[3]), t(vs[1].clone(), &[3]));
        let jv = t0.jvp_multi(&y, &[&x1, &x2], &[&v1, &v2], &tape()).unwrap();
        // x2⊙v1 + x1⊙v2 = [2*1+1*0, 0.5*0.5+2*1, 1.5*2+3*(-1)] = [2.0, 2.25, 0.0]。
        fandhe_ai_backend_cpu::parity::assert_parity(
            "multi mul hand",
            &host(&jv),
            &[2.0, 2.25, 0.0],
        );
    }
    // 非正方形 matmul → tanh（x [2,3]・W [3,4] の両方を追跡あり）。
    check_multi_against_jacobian(
        "multi matmul_tanh",
        &[
            (vec![0.3, -0.7, 1.1, 0.5, 0.2, -0.4], vec![2, 3]),
            (
                (0..12).map(|i| 0.1 * (i as f32) - 0.5).collect(),
                vec![3, 4],
            ),
        ],
        &[
            vec![1.0, -0.5, 0.25, 0.0, 2.0, -1.5],
            (0..12).map(|i| 0.05 * (i as f32) - 0.3).collect(),
        ],
        |_, l| l[0].matmul(&l[1]).unwrap().tanh(),
    );
    // ブロードキャスト add（a [2,3]・b [3] の両方を追跡あり）。
    check_multi_against_jacobian(
        "multi broadcast_add",
        &[
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            (vec![0.1, -0.2, 0.3], vec![3]),
        ],
        &[vec![0.5, -1.0, 1.0, 2.0, 0.0, -0.5], vec![1.0, -1.0, 0.5]],
        |_, l| l[0].add(&l[1]).unwrap().tanh(),
    );
    // 3 入力。
    check_multi_against_jacobian(
        "multi three_inputs",
        &[
            (vec![1.0, 2.0, 3.0], vec![3]),
            (vec![0.5, -1.0, 2.0], vec![3]),
            (vec![2.0, 1.0, -0.5], vec![3]),
        ],
        &[
            vec![1.0, 0.0, -1.0],
            vec![0.5, 0.5, 0.5],
            vec![-1.0, 2.0, 1.0],
        ],
        |_, l| l[0].mul(&l[1]).unwrap().mul(&l[2]).unwrap(),
    );
    // rank 0 の 2 入力。
    check_multi_against_jacobian(
        "multi rank0",
        &[(vec![1.5], vec![]), (vec![-2.0], vec![])],
        &[vec![2.0], vec![0.5]],
        |_, l| l[0].mul(&l[1]).unwrap(),
    );
}

#[test]
fn multi_input_zero_jacobian_input_drops_out() {
    // z は出力が依存しない追跡あり入力（J_z ≡ 0）。z を含めても z を除いた呼び出し・単一 jvp と一致する。
    let t0 = tape();
    let x = t0.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let z = t0.var(&t(vec![9.0, 8.0], &[2]));
    let y = x.mul(&x).unwrap().tanh();
    let vx = t(vec![1.0, -0.5, 2.0], &[3]);
    let vz = t(vec![5.0, 7.0], &[2]);

    let with_z = t0.jvp_multi(&y, &[&x, &z], &[&vx, &vz], &tape()).unwrap();
    let without_z = t0.jvp_multi(&y, &[&x], &[&vx], &tape()).unwrap();
    let single = t0.jvp(&y, &x, &vx, &tape()).unwrap();
    assert_eq!(with_z.shape(), y.to_tensor().shape());
    fandhe_ai_backend_cpu::parity::assert_parity(
        "zero J: with z vs without z",
        &host(&with_z),
        &host(&without_z),
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        "zero J: with z vs single",
        &host(&with_z),
        &host(&single),
    );
    assert!(
        host(&with_z).iter().any(|v| *v != 0.0),
        "自明な全ゼロ一致を避ける"
    );
}

#[test]
fn multi_input_single_input_matches_jvp() {
    // k = 1。§30.2 は bit 同一を主張しないので REQ-2 判定で比べる。
    let t0 = tape();
    let x = t0.var(&t(vec![0.3, -0.7, 1.1, 0.5, 0.2, -0.4], &[2, 3]));
    let w = t0.var_no_grad(&t(
        (0..12).map(|i| 0.1 * (i as f32) - 0.5).collect(),
        &[3, 4],
    ));
    let y = x.matmul(&w).unwrap().tanh();
    let v = t(vec![1.0, -0.5, 0.25, 0.0, 2.0, -1.5], &[2, 3]);
    let multi = t0.jvp_multi(&y, &[&x], &[&v], &tape()).unwrap();
    let single = t0.jvp(&y, &x, &v, &tape()).unwrap();
    assert_eq!(multi.shape(), single.shape());
    fandhe_ai_backend_cpu::parity::assert_parity("k=1", &host(&multi), &host(&single));
}

#[test]
fn multi_input_validation_errors() {
    let t0 = tape();
    let t1 = tape();
    let x1 = t0.var(&t(vec![1.0, 2.0], &[2]));
    let x2 = t0.var(&t(vec![3.0, 4.0], &[2]));
    let y = x1.mul(&x2).unwrap();
    let v = t(vec![1.0, 1.0], &[2]);
    let foreign = t1.var(&t(vec![1.0, 2.0], &[2]));
    let nograd = t0.var_no_grad(&t(vec![1.0, 2.0], &[2]));
    let bad = t(vec![1.0, 1.0, 1.0], &[3]);

    // ① 長さ不一致。
    assert!(matches!(
        t0.jvp_multi(&y, &[&x1, &x2], &[&v], &tape()),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // ② 空配列。
    let no_in: [&Var<'_>; 0] = [];
    let no_tg: [&Tensor<f32>; 0] = [];
    assert!(matches!(
        t0.jvp_multi(&y, &no_in, &no_tg, &tape()),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // ③ 別テープ。
    assert!(matches!(
        t0.jvp_multi(&y, &[&x1, &foreign], &[&v, &v], &tape()),
        Err(AutodiffError::TapeMismatch)
    ));
    // ④ 追跡なし入力。
    assert!(matches!(
        t0.jvp_multi(&y, &[&x1, &nograd], &[&v, &v], &tape()),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    // ⑤ 重複指定。
    assert!(matches!(
        t0.jvp_multi(&y, &[&x1, &x1], &[&v, &v], &tape()),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // ⑥ 接ベクトルの shape 不一致。
    assert!(matches!(
        t0.jvp_multi(&y, &[&x1, &x2], &[&v, &bad], &tape()),
        Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
    ));

    // 順序の固定（違反が複数なら最初のものを返す）。
    // ① が ③ より先。
    assert!(matches!(
        t0.jvp_multi(&y, &[&x1, &foreign], &[&v], &tape()),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // ③ が ④ より先。
    assert!(matches!(
        t0.jvp_multi(&y, &[&nograd, &foreign], &[&v, &v], &tape()),
        Err(AutodiffError::TapeMismatch)
    ));
    // ④ が ⑤ より先（同一の追跡なし入力を 2 回）。
    assert!(matches!(
        t0.jvp_multi(&y, &[&nograd, &nograd], &[&v, &v], &tape()),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    // ⑤ が ⑥ より先（重複かつ shape 不一致）。
    assert!(matches!(
        t0.jvp_multi(&y, &[&x1, &x1], &[&bad, &bad], &tape()),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn multi_input_unsupported_ops_and_dirty_child_are_rejected() {
    let t0 = tape();
    let x1 = t0.var(&t(vec![1.0, 3.0, 2.0], &[3]));
    let x2 = t0.var(&t(vec![0.5, 0.5, 0.5], &[3]));
    let v = t(vec![1.0, 1.0, 1.0], &[3]);

    // supports_create_graph() が偽の Op（max）を経由する出力は型付きの Err。
    let m = x1.mul(&x2).unwrap().max(None).unwrap();
    assert!(matches!(
        t0.jvp_multi(&m, &[&x1, &x2], &[&v, &v], &tape()),
        Err(AutodiffError::Backward(_))
    ));

    // 空でない子テープは拒否される。
    let y = x1.mul(&x2).unwrap();
    let dirty = tape();
    let _leaf = dirty.var(&t(vec![0.0], &[1]));
    assert!(matches!(
        t0.jvp_multi(&y, &[&x1, &x2], &[&v, &v], &dirty),
        Err(AutodiffError::Backward(_))
    ));
}

#[test]
fn multi_input_untracked_output_is_zero() {
    let t0 = tape();
    let x1 = t0.var(&t(vec![1.0, 2.0], &[2]));
    let x2 = t0.var(&t(vec![3.0, 4.0], &[2]));
    let c = t0.var_no_grad(&t(vec![3.0, 4.0], &[2]));
    let v = t(vec![1.0, 1.0], &[2]);
    let untracked = c.mul(&c).unwrap();
    let jv = t0
        .jvp_multi(&untracked, &[&x1, &x2], &[&v, &v], &tape())
        .unwrap();
    assert_eq!(jv.shape(), &[2]);
    assert_eq!(host(&jv), vec![0.0, 0.0]);
}
