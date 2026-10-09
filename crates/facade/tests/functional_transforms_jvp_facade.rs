//! facade `Tape::jvp`・`Tape::jacfwd`（イシュー #2956。決定記録
//! `docs/autodiff-functional-transforms-design.md` §27・§28）の CPU 結合テスト。
//!
//! 属性なし（CI で実行）: facade 経由の結果を同じ facade の `Tape::jacobian`（reverse-mode 行ごと）と
//! 接ベクトルのホスト側積（f64 蓄積）に突き合わせる。対象 Op は `supports_create_graph()` が真の
//! mul・tanh・matmul・sum・ブロードキャスト add で、非正方形と rank 0 を含む。数値の比較は REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`）を使い、tolerance は追加・変更しない。
//! 入口検査のエラー伝播・非対象 Op／空でない子テープの拒否・追跡なし `output` の非対称（`hvp` は `Err`）も固定する。
//!
//! 実機（CUDA／Metal）の parity は #2942 の既存 `#[ignore]` テスト
//! （`functional_ops_jvp_backend_parity.rs`）が担当し、本ファイルは facade 層を CPU で担当する。
//! 本ファイルのテスト関数名は宣言インベントリ（`api_surface.rs`。`crates/*/src` のみ走査）の対象外。

use fandhe_ai::{AutodiffError, Tape, Tensor, Var, tape};

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
