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
//! 恒久対象外の Op（`Tape::custom`）・空でない子テープ（`Backward`）、追跡なし `output` の全ゼロ。親テープに残る 2 ノードの副作用は
//! facade からノード数を観測する公開 API がないため内部テスト（`functional_ops_jvp_tests.rs` の M5）に任せる。
//! 複数入力の実機 parity は `functional_transforms_jvp_multi_backend_parity.rs`（`#[ignore]`）が担当する。
//!
//! 微分可能な jvp `Tape::jvp_create_graph`／`JvpCreateGraphResult`（イシュー #3068・決定記録 §31）も同ファイルで
//! CPU 結合テストする。固定するもの: `y = x ⊙ x` で `output() = 2x ⊙ v` かつ `Tape::jvp` と REQ-2 で一致、孫テープの
//! `backward` が `child_var` の写しに `2v` を返す、スカラー損失での `Tape::hvp`／`Tape::hessian`（ホスト積）との整合、
//! `child_var` の意味論（`Ok(None)`・`TapeMismatch`・`GradientTrackingDisabled`）、入口検査 1〜4 のエラー伝播と
//! テープ事前検査（同一テープ・空でないテープ → `Backward`）、ゼロ経路、恒久対象外の Op（`Tape::custom`）の拒否。
//! `DeviceMismatch` は CPU だけでは作れないため未検証（内部 #3067 でも未検証。§31.9）。親テープに残るノード数は
//! facade から観測できないため内部テスト（C6／C7）に任せる。実機 parity は #3168・#3169 の範囲。
//!
//! 実機（CUDA／Metal）の parity は #2942 の既存 `#[ignore]` テスト
//! （`functional_ops_jvp_backend_parity.rs`）が担当し、本ファイルは facade 層を CPU で担当する。
//! イシュー #3087（決定記録 `docs/autodiff-higher-order-grad-decision.md` §32）: #3077〜#3086 で `create_graph`
//! 対象へ広げた Op の代表（softmax・layer_norm・conv2d・gelu_tanh・cumsum・pad・avg_pool2d・rfft・solve）を
//! `jvp`／`jacfwd`／`jvp_multi` で `Tape::jacobian` と突き合わせる。拒否の基準は `max`（C 区分で将来対象になりうる）
//! ではなく恒久対象外の `Tape::custom`（A 区分）。実機は #3168・#3169 へ申し送る。
//! 本ファイルのテスト関数名は宣言インベントリ（`api_surface.rs`。`crates/*/src` のみ走査）の対象外。

use std::sync::Arc;

use fandhe_ai::{AutodiffError, CustomFunction, FftNorm, ShapeError, Tape, Tensor, Var, tape};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn host(x: &Tensor<f32>) -> Vec<f32> {
    x.host_slice().into_owned()
}

/// 恒久対象外の `Tape::custom` 用の恒等関数（拒否の基準 Op。`Op::Custom` は §21.2 A 区分。
/// `LogSumExp`・`Max` は C 区分で将来の拡張で対象になりうるため基準にしない。イシュー #3087）。
struct IdentityFn;

impl CustomFunction for IdentityFn {
    fn name(&self) -> &str {
        "identity"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        Ok(inputs[0].clone())
    }
    fn backward(
        &self,
        _inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        _requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        Ok(vec![Some(upstream.clone())])
    }
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

    // 恒久対象外の Op（`Tape::custom`）を経由する出力は型付きの `Backward` エラー。
    let m = t0.custom(Arc::new(IdentityFn), &[x]).unwrap();
    assert!(matches!(
        t0.jvp(&m, &x, &t(vec![1.0, 1.0, 1.0], &[3]), &tape()),
        Err(AutodiffError::Backward(_))
    ));
    assert!(matches!(
        t0.jacfwd(&m, &x, &tape()),
        Err(AutodiffError::Backward(_))
    ));

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

    // 恒久対象外の Op（`Tape::custom`）を経由する出力は型付きの `Backward` エラー。
    let prod = x1.mul(&x2).unwrap();
    let m = t0.custom(Arc::new(IdentityFn), &[prod]).unwrap();
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

// ---------------------------------------------------------------------------------------------
// #3068: 微分可能な jvp（`Tape::jvp_create_graph`）。参照用の `jacobian`／`hvp`／`hessian` は毎回新しい子テープで呼ぶ。
// ---------------------------------------------------------------------------------------------

/// `create_graph` 版の `output()` を `jvp` と `jacobian·v` に突き合わせる（shape も `output.shape()` と一致）。
fn check_create_graph_value<F>(
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
    let jac = host(&t0.jacobian(&y, &x).expect("jacobian"));
    let tangent = t(v.clone(), x_shape);

    let (child, grand) = (tape(), tape());
    let res = t0
        .jvp_create_graph(&y, &x, &tangent, &child, &grand)
        .expect("jvp_create_graph");
    let out = res.output().to_tensor();
    assert_eq!(out.shape(), y.to_tensor().shape(), "{label}: shape");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("{label}: vs jacobian·v"),
        &host(&out),
        &reference_jv(&jac, &v),
    );
    let jv = t0.jvp(&y, &x, &tangent, &tape()).expect("jvp");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("{label}: vs jvp"),
        &host(&out),
        &host(&jv),
    );
}

#[test]
fn create_graph_value_matches_jvp_and_closed_form() {
    // y = x ⊙ x: output() = 2x ⊙ v（閉形式）。
    let t0 = tape();
    let x = t0.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let y = x.mul(&x).unwrap();
    let v = t(vec![1.0, 0.5, 2.0], &[3]);
    let (child, grand) = (tape(), tape());
    let res = t0.jvp_create_graph(&y, &x, &v, &child, &grand).unwrap();
    assert_eq!(res.output().to_tensor().shape(), y.to_tensor().shape());
    fandhe_ai_backend_cpu::parity::assert_parity(
        "create_graph: closed form",
        &host(&res.output().to_tensor()),
        &[2.0, 2.0, 12.0],
    );

    check_create_graph_value(
        "mul",
        vec![1.0, 2.0, 3.0],
        &[3],
        vec![1.0, 0.5, 2.0],
        |_, x| x.mul(x).unwrap(),
    );
    let w = (0..12).map(|i| 0.1 * (i as f32) - 0.5).collect::<Vec<_>>();
    check_create_graph_value(
        "matmul_tanh",
        vec![0.3, -0.7, 1.1, 0.5, 0.2, -0.4],
        &[2, 3],
        vec![1.0, -0.5, 0.25, 0.0, 2.0, -1.5],
        move |tp, x| {
            let w = tp.var_no_grad(&t(w.clone(), &[3, 4]));
            x.matmul(&w).unwrap().tanh()
        },
    );
    check_create_graph_value("rank0", vec![1.5], &[], vec![2.0], |_, x| x.mul(x).unwrap());
}

#[test]
fn create_graph_second_order_is_two_v_for_square() {
    let t0 = tape();
    let x = t0.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let y = x.mul(&x).unwrap();
    let v = t(vec![1.0, 0.5, 2.0], &[3]);
    let (child, grand) = (tape(), tape());
    let res = t0.jvp_create_graph(&y, &x, &v, &child, &grand).unwrap();
    let grads = grand.backward(&res.output().sum(None).unwrap()).unwrap();
    let xc = res.child_var(&x).unwrap().expect("結果は x に依存する");
    let g = grads.get(&xc).unwrap().expect("2 階勾配");
    // d(Σ 2x⊙v)/dx = 2v。
    fandhe_ai_backend_cpu::parity::assert_parity(
        "create_graph: backward = 2v",
        &host(g),
        &[2.0, 1.0, 4.0],
    );
}

#[test]
fn create_graph_matches_hvp_and_hessian_for_scalar_loss() {
    let x0 = vec![0.5_f32, -1.0, 2.0];
    let v0 = vec![1.0_f32, -0.5, 0.25];
    let n = x0.len();
    fn build<'a>(t0: &'a Tape, x0: &[f32]) -> (Var<'a>, Var<'a>) {
        let x = t0.var(&t(x0.to_vec(), &[x0.len()]));
        let loss = x
            .mul(&x)
            .unwrap()
            .mul(&x)
            .unwrap()
            .tanh()
            .sum(None)
            .unwrap();
        (x, loss)
    }
    let v = t(v0.clone(), &[n]);

    // create_graph 版: L の jvp（スカラー ∇L·v）を孫テープで x について微分すると H·v。
    let t0 = tape();
    let (x, loss) = build(&t0, &x0);
    let (child, grand) = (tape(), tape());
    let res = t0.jvp_create_graph(&loss, &x, &v, &child, &grand).unwrap();
    let grads = grand.backward(&res.output()).unwrap();
    let xc = res.child_var(&x).unwrap().expect("結果は x に依存する");
    let hv_cg = host(grads.get(&xc).unwrap().expect("H·v"));

    // Tape::hvp との整合。
    let tp = tape();
    let (xp, lossp) = build(&tp, &x0);
    let hv = tp.hvp(&lossp, &xp, &v, &tape()).expect("hvp");
    fandhe_ai_backend_cpu::parity::assert_parity("create_graph vs hvp", &hv_cg, &host(&hv));

    // Tape::hessian と v のホスト積（f64 で蓄積し最後に f32）との整合。
    let th = tape();
    let (xh, lossh) = build(&th, &x0);
    let h = host(&th.hessian(&lossh, &xh, &tape()).expect("hessian"));
    assert_eq!(h.len(), n * n);
    let hv_host: Vec<f32> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| f64::from(h[i * n + j]) * f64::from(v0[j]))
                .sum::<f64>() as f32
        })
        .collect();
    fandhe_ai_backend_cpu::parity::assert_parity("create_graph vs hessian·v", &hv_cg, &hv_host);
}

#[test]
fn create_graph_child_var_semantics() {
    let t0 = tape();
    let t1 = tape();
    let x = t0.var(&t(vec![1.0, 2.0], &[2, 1]));
    let w = t0.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let y = w.matmul(&x).unwrap();
    let v = t(vec![1.0, 1.0], &[2, 1]);
    let (child, grand) = (tape(), tape());
    let res = t0.jvp_create_graph(&y, &x, &v, &child, &grand).unwrap();
    // 線形 y = W x: J·v = W v は x に依存しない（写しなし）が W には依存する。
    assert!(res.child_var(&x).unwrap().is_none());
    assert!(res.child_var(&w).unwrap().is_some());
    // 別テープ・追跡なし。
    let foreign = t1.var(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        res.child_var(&foreign),
        Err(AutodiffError::TapeMismatch)
    ));
    let ng = t0.var_no_grad(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        res.child_var(&ng),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
}

#[test]
fn create_graph_input_validation_errors_propagate() {
    let t0 = tape();
    let t1 = tape();
    let x = t0.var(&t(vec![1.0, 2.0], &[2]));
    let y = x.mul(&x).unwrap();
    let v = t(vec![1.0, 1.0], &[2]);
    let cg = |t: &Tape, out: &Var<'_>, inp: &Var<'_>, tan: &Tensor<f32>| {
        let (c, g) = (tape(), tape());
        t.jvp_create_graph(out, inp, tan, &c, &g).map(|_| ())
    };

    // 1〜4: `Tape::jvp` と同じ variant。
    let foreign = t1.var(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        cg(&t0, &y, &foreign, &v),
        Err(AutodiffError::TapeMismatch)
    ));
    let ng = t0.var_no_grad(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        cg(&t0, &y, &ng, &v),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    let bad = t(vec![1.0, 1.0, 1.0], &[3]);
    assert!(matches!(
        cg(&t0, &y, &x, &bad),
        Err(AutodiffError::Shape(_))
    ));

    // 5: テープの事前検査（すべて `Backward`）。
    let (c, g) = (tape(), tape());
    assert!(matches!(
        t0.jvp_create_graph(&y, &x, &v, &t0, &g),
        Err(AutodiffError::Backward(_))
    ));
    assert!(matches!(
        t0.jvp_create_graph(&y, &x, &v, &c, &t0),
        Err(AutodiffError::Backward(_))
    ));
    assert!(matches!(
        t0.jvp_create_graph(&y, &x, &v, &c, &c),
        Err(AutodiffError::Backward(_))
    ));
    let dirty = tape();
    let _leaf = dirty.var(&t(vec![0.0], &[1]));
    assert!(matches!(
        t0.jvp_create_graph(&y, &x, &v, &dirty, &g),
        Err(AutodiffError::Backward(_))
    ));
    assert!(matches!(
        t0.jvp_create_graph(&y, &x, &v, &c, &dirty),
        Err(AutodiffError::Backward(_))
    ));
    // 検査順序: 入口検査 1（別テープ）はテープの事前検査（5）より先に返る。
    assert!(matches!(
        t0.jvp_create_graph(&y, &foreign, &v, &t0, &t0),
        Err(AutodiffError::TapeMismatch)
    ));
}

#[test]
fn create_graph_zero_paths() {
    let t0 = tape();
    let x = t0.var(&t(vec![1.0, 2.0], &[2]));
    let c = t0.var_no_grad(&t(vec![3.0, 4.0], &[2]));
    let v = t(vec![1.0, 1.0], &[2]);
    // 追跡なし output。
    let untracked = c.mul(&c).unwrap();
    let (child, grand) = (tape(), tape());
    let res = t0
        .jvp_create_graph(&untracked, &x, &v, &child, &grand)
        .unwrap();
    let out = res.output();
    // 追跡なし（`requires_grad` は非公開のため、孫テープでの backward が Err になることで確認する）。
    assert!(grand.backward(&out).is_err());
    assert_eq!(out.to_tensor().shape(), &[2]);
    assert_eq!(host(&out.to_tensor()), vec![0.0, 0.0]);
    assert!(res.child_var(&x).unwrap().is_none());

    // 要素数 0。
    let xe = t0.var(&t(vec![], &[0]));
    let ye = xe.mul(&xe).unwrap();
    let ve = t(vec![], &[0]);
    let (child, grand) = (tape(), tape());
    let res = t0.jvp_create_graph(&ye, &xe, &ve, &child, &grand).unwrap();
    assert!(grand.backward(&res.output()).is_err());
    assert_eq!(res.output().to_tensor().shape(), &[0]);
    assert!(res.child_var(&xe).unwrap().is_none());
}

#[test]
fn create_graph_unsupported_op_is_rejected() {
    let t0 = tape();
    let x = t0.var(&t(vec![1.0, 3.0, 2.0], &[3]));
    let v = t(vec![1.0, 1.0, 1.0], &[3]);
    let m = t0.custom(Arc::new(IdentityFn), &[x]).unwrap();
    let (child, grand) = (tape(), tape());
    assert!(matches!(
        t0.jvp_create_graph(&m, &x, &v, &child, &grand),
        Err(AutodiffError::Backward(_))
    ));
}

#[test]
fn jvp_and_jacfwd_match_jacobian_on_extended_ops() {
    let x6 = vec![0.5_f32, 1.1, -0.6, 0.2, -1.3, 0.8];
    let v6 = vec![0.6_f32, -0.2, 0.3, 0.9, -0.5, 0.4];
    check_against_jacobian("softmax", x6.clone(), &[2, 3], v6.clone(), move |tp, x| {
        let w = tp.var_no_grad(&t(vec![0.3, -1.2, 0.7, 2.0, -0.5, 1.1], &[2, 3]));
        x.softmax(1).unwrap().mul(&w).unwrap()
    });
    check_against_jacobian("layer_norm", x6.clone(), &[2, 3], v6.clone(), |tp, x| {
        let w = tp.var_no_grad(&t(vec![1.2, 0.7, -0.4], &[3]));
        let b = tp.var_no_grad(&t(vec![0.1, -0.2, 0.3], &[3]));
        x.layer_norm(Some(&w), Some(&b), 1e-5).unwrap()
    });
    let x9 = vec![0.3_f32, -0.8, 1.2, 0.1, -0.4, 0.9, 0.5, -0.2, 0.7];
    let v9 = vec![0.6_f32, -0.2, 0.3, 0.9, -0.5, 0.4, 0.1, -0.7, 0.2];
    check_against_jacobian("conv2d", x9.clone(), &[1, 1, 3, 3], v9.clone(), |tp, x| {
        let w = tp.var_no_grad(&t(vec![0.5, -0.3, 0.2, 0.8], &[1, 1, 2, 2]));
        x.conv2d(&w, None, [1, 1], [0, 0], [1, 1], 1).unwrap()
    });
    check_against_jacobian("gelu_tanh", x6.clone(), &[2, 3], v6.clone(), |_, x| {
        x.gelu_tanh().unwrap()
    });
    check_against_jacobian("cumsum", x6.clone(), &[2, 3], v6.clone(), |_, x| {
        x.cumsum(1).unwrap()
    });
    check_against_jacobian("pad", x6.clone(), &[2, 3], v6.clone(), |_, x| {
        x.pad(&[(1, 0), (0, 1)], 0.0).unwrap()
    });
    check_against_jacobian("avg_pool2d", x9, &[1, 1, 3, 3], v9, |_, x| {
        x.avg_pool2d([2, 2], Some([1, 1]), [0, 0], false, true)
            .unwrap()
    });
    check_against_jacobian(
        "rfft",
        vec![0.5, -1.0, 0.25, 2.0],
        &[4],
        vec![0.6, -0.2, 0.3, 0.9],
        |_, x| x.rfft(None, None, FftNorm::Backward).unwrap(),
    );
    check_against_jacobian(
        "solve",
        vec![2.0, 0.5, 0.3, 1.5],
        &[2, 2],
        vec![0.6, -0.2, 0.3, 0.9],
        |tp, a| {
            let b = tp.var_no_grad(&t(vec![1.0, -2.0], &[2, 1]));
            a.solve(&b).unwrap()
        },
    );
}

/// 複数入力: conv2d の入力 `x1` と重み `x2` の両方を追跡して `jvp_multi` を `jacobian` 和と突き合わせる。
#[test]
fn multi_input_matches_jacobian_sum_on_conv2d() {
    let xs = vec![
        (
            vec![0.3, -0.8, 1.2, 0.1, -0.4, 0.9, 0.5, -0.2, 0.7],
            vec![1, 1, 3, 3],
        ),
        (vec![0.5, -0.3, 0.2, 0.8], vec![1, 1, 2, 2]),
    ];
    let vs = vec![
        vec![0.6, -0.2, 0.3, 0.9, -0.5, 0.4, 0.1, -0.7, 0.2],
        vec![0.3, 0.1, -0.4, 0.2],
    ];
    check_multi_against_jacobian("multi conv2d", &xs, &vs, |_, l| {
        l[0].conv2d(&l[1], None, [1, 1], [0, 0], [1, 1], 1).unwrap()
    });
}
