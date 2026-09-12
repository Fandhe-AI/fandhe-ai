//! 線形代数（inv／solve／det／qr／cholesky／svd）・matrix_norm の
//! end-to-end 統合テスト（イシュー #1621・親イシュー #1573
//! 「Tier 2: 線形代数」・`docs/spec/04-requirements.md` REQ-9
//! 2026-09-12 追記）。
//!
//! `backward.rs`（TASK-1.5c・#18）と同じ「`Tape`/`Var` を経由する
//! end-to-end 経路のみを対象とする」方針を踏襲する。`common::naive_ops()`
//! （`BackendOps::linalg_*` を一切オーバーライドしない）を使うため、
//! `Var::inv` 等は必ず `eval::linalg`（ホスト参照実装）へフォールバック
//! する経路を通る（`crates/backend-cpu` 側の実装は
//! `backend-cpu/tests/linalg_parity.rs` が別途担当）。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{BackendError, MatrixNormOrd, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

fn approx_eq(actual: &Tensor<f32>, expected: &[f32], tol: f32) {
    let data = actual.contiguous();
    let slice = data.as_slice().expect("test fixture: contiguous のはず");
    assert_eq!(slice.len(), expected.len(), "要素数が一致しない");
    for (a, e) in slice.iter().zip(expected.iter()) {
        assert!((a - e).abs() <= tol, "{a} vs {e} (tol={tol})");
    }
}

// --- inv: forward 値・backward の到達性 ---

#[test]
fn inv_forward_and_backward_reach_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![4.0, 1.0, 2.0, 3.0], &[2, 2]));

    let inv_a = a.inv().unwrap();
    approx_eq(&inv_a.to_tensor(), &[0.3, -0.1, -0.2, 0.4], 1e-4);

    let loss = inv_a.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    assert_eq!(da.shape(), &[2, 2]);
}

#[test]
fn inv_singular_matrix_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]));
    let result = a.inv();
    // フォールバック経路（`eval::linalg::inv`）のエラーは CPU 本番経路
    // （`BackendOps::linalg_inv`）と同じ `AutodiffError::Backend(
    // BackendError::InvalidArgument(_))` に統一される（`Var::inv` の
    // `unify_fallback_error`。codex-review／Cursor Bugbot 指摘: 実装
    // 選択〈フォールバックか本番か〉で呼び出し元から見えるエラー
    // variant が変わってはならない）。
    assert!(matches!(
        result,
        Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
    ));
}

#[test]
fn inv_rejects_non_square_shape() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let result = a.inv();
    assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
}

// --- solve ---

#[test]
fn solve_matches_inv_times_b_and_backward_reaches_both_inputs() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![3.0, 1.0, 1.0, 2.0], &[2, 2]));
    let b = tape.var(&t(vec![9.0, 8.0], &[2, 1]));

    let x = a.solve(&b).unwrap();
    approx_eq(&x.to_tensor(), &[2.0, 3.0], 1e-4);

    let loss = x.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    assert!(grads.get(&a).unwrap().is_some());
    assert!(grads.get(&b).unwrap().is_some());
}

// --- det ---

#[test]
fn det_known_value_and_singular_returns_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let d = a.det().unwrap();
    assert!((scalar(&d.to_tensor()) - (-2.0)).abs() < 1e-5);

    let singular_tape = Tape::new_with_ops(common::naive_ops());
    let s = singular_tape.var(&t(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]));
    let ds = s.det().unwrap();
    assert!(scalar(&ds.to_tensor()).abs() < 1e-6);
}

// --- cholesky ---

#[test]
fn cholesky_reconstructs_spd_and_backward_reaches_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // A = [[4,2],[2,3]] は対称正定値（固有値とも正）。
    let a = tape.var(&t(vec![4.0, 2.0, 2.0, 3.0], &[2, 2]));
    let l = a.cholesky().unwrap();
    let l_val = l.to_tensor();
    // L = [[2,0],[1,sqrt(2)]]（L L^T = [[4,2],[2,3]]）。
    approx_eq(&l_val, &[2.0, 0.0, 1.0, 2f32.sqrt()], 1e-4);

    let loss = l.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    assert!(grads.get(&a).unwrap().is_some());
}

#[test]
fn cholesky_non_positive_definite_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 2.0, 1.0], &[2, 2]));
    let result = a.cholesky();
    // `inv_singular_matrix_is_invalid_argument` と同じ理由でエラー型を
    // 統一する（`Var::cholesky` の `unify_fallback_error`）。
    assert!(matches!(
        result,
        Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
    ));
}

// --- qr: 多出力ノードの蓄積検証（sum(Q) + sum(R) を同時逆伝播） ---

#[test]
fn qr_multi_output_gradient_accumulates_to_single_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0], &[3, 2]));

    let qr = a.qr().unwrap();
    let q_sum = qr.q.sum(None).unwrap();
    let r_sum = qr.r.sum(None).unwrap();
    let loss = q_sum.add(&r_sum).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let da = grads
        .get(&a)
        .unwrap()
        .expect("a は Q・R 両方の経路から到達する");
    assert_eq!(da.shape(), &[3, 2]);
}

#[test]
fn qr_reconstructs_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0], &[3, 2]));
    let qr = a.qr().unwrap();
    let reconstructed = qr.q.matmul(&qr.r).unwrap();
    approx_eq(
        &reconstructed.to_tensor(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 7.0],
        1e-4,
    );
}

#[test]
fn qr_wide_matrix_backward_is_invalid_argument() {
    // `m < n`（wide 行列）の backward は設計文書スコープ外
    // （`docs/autodiff-linalg-design.md` §3.4「QrQ／QrR」）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let qr = a.qr().unwrap();
    let loss = qr.q.sum(None).unwrap();
    let result = tape.backward(&loss);
    assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
}

// --- svd: 多出力ノードの蓄積検証 ---

#[test]
fn svd_multi_output_gradient_accumulates_to_single_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // 特異値が相異なる可分行列（近接／重複回避）。
    let a = tape.var(&t(
        vec![3.0, 0.3, 0.1, 0.2, 2.0, 0.2, 0.1, 0.15, 1.0],
        &[3, 3],
    ));

    let svd = a.svd().unwrap();
    let u_sum = svd.u.sum(None).unwrap();
    let s_sum = svd.s.sum(None).unwrap();
    let vh_sum = svd.vh.sum(None).unwrap();
    let loss = u_sum.add(&s_sum).unwrap().add(&vh_sum).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let da = grads
        .get(&a)
        .unwrap()
        .expect("a は U・S・Vh 3 経路から到達する");
    assert_eq!(da.shape(), &[3, 3]);
}

#[test]
fn svd_reconstructs_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(
        vec![3.0, 0.3, 0.1, 0.2, 2.0, 0.2, 0.1, 0.15, 1.0],
        &[3, 3],
    ));
    let svd = a.svd().unwrap();
    let s_diag_data = {
        let s_vec = svd.s.to_tensor();
        let s_slice = s_vec.contiguous();
        s_slice.as_slice().unwrap().to_vec()
    };
    let mut u_scaled_data = svd.u.to_tensor().contiguous().as_slice().unwrap().to_vec();
    let k = s_diag_data.len();
    for r in 0..3 {
        for c in 0..k {
            u_scaled_data[r * k + c] *= s_diag_data[c];
        }
    }
    let u_scaled = tape.var(&t(u_scaled_data, &[3, k]));
    let reconstructed = u_scaled.matmul(&svd.vh).unwrap();
    approx_eq(
        &reconstructed.to_tensor(),
        &[3.0, 0.3, 0.1, 0.2, 2.0, 0.2, 0.1, 0.15, 1.0],
        1e-3,
    );
}

// --- matrix_norm（5 ord） ---

#[test]
fn matrix_norm_fro_known_value_and_backward_reaches_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![3.0, 4.0], &[1, 2]));
    let norm = a.matrix_norm(MatrixNormOrd::Fro).unwrap();
    assert!((scalar(&norm.to_tensor()) - 5.0).abs() < 1e-5);

    let grads = tape.backward(&norm).unwrap();
    assert!(grads.get(&a).unwrap().is_some());
}

#[test]
fn matrix_norm_one_and_inf_known_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![2.0, -1.0, -6.0, 3.0], &[2, 2]));
    let one = a.matrix_norm(MatrixNormOrd::One).unwrap();
    let inf = a.matrix_norm(MatrixNormOrd::Inf).unwrap();
    assert!((scalar(&one.to_tensor()) - 8.0).abs() < 1e-5);
    assert!((scalar(&inf.to_tensor()) - 9.0).abs() < 1e-5);
}

#[test]
fn matrix_norm_nuc_and_spectral_backward_reach_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(
        vec![3.0, 0.3, 0.1, 0.2, 2.0, 0.2, 0.1, 0.15, 1.0],
        &[3, 3],
    ));
    let nuc = a.matrix_norm(MatrixNormOrd::Nuc).unwrap();
    let grads = tape.backward(&nuc).unwrap();
    assert!(grads.get(&a).unwrap().is_some());

    let tape2 = Tape::new_with_ops(common::naive_ops());
    let a2 = tape2.var(&t(
        vec![3.0, 0.3, 0.1, 0.2, 2.0, 0.2, 0.1, 0.15, 1.0],
        &[3, 3],
    ));
    let spectral = a2.matrix_norm(MatrixNormOrd::Spectral).unwrap();
    let grads2 = tape2.backward(&spectral).unwrap();
    assert!(grads2.get(&a2).unwrap().is_some());
}

#[test]
fn matrix_norm_rejects_non_rank2_shape() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let result = a.matrix_norm(MatrixNormOrd::Fro);
    assert!(matches!(result, Err(AutodiffError::Shape(_))));
}

// --- 空行列（n=0）契約 ---

#[test]
fn empty_matrix_contracts() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(Vec::new(), &[0, 0]));

    let inv_a = a.inv().unwrap();
    assert_eq!(inv_a.to_tensor().shape(), &[0, 0]);

    let det_a = a.det().unwrap();
    assert!((scalar(&det_a.to_tensor()) - 1.0).abs() < 1e-6);
}

/// 空行列（`[2,0]`／`[0,2]`）の `matrix_norm` backward が
/// `One`／`Inf`／`Spectral`／`Nuc` いずれでも panic しないことを確認
/// する（codex-review 指摘。P1 #4 の修正回帰。`eval::linalg::
/// matrix_norm_vjp` の `best_col`／`best_row` 初期値 `0` や `svd` の
/// `k=0` 特異ベクトルへの添字アクセスが空バッファを踏み抜いて panic
/// していた）。forward は `0` を返す契約（設計文書 §3.5）。
#[test]
fn empty_matrix_norm_backward_does_not_panic() {
    for shape in [[2usize, 0usize], [0, 2]] {
        for ord in [
            MatrixNormOrd::Fro,
            MatrixNormOrd::One,
            MatrixNormOrd::Inf,
            MatrixNormOrd::Nuc,
            MatrixNormOrd::Spectral,
        ] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let a = tape.var(&t(Vec::new(), &shape));
            let norm = a.matrix_norm(ord).unwrap();
            assert!(
                (scalar(&norm.to_tensor())).abs() < 1e-6,
                "shape={shape:?} ord={ord:?}: forward は 0 のはず"
            );
            let grads = tape.backward(&norm).unwrap_or_else(|e| {
                panic!("shape={shape:?} ord={ord:?}: backward が panic せず Err を返した: {e:?}")
            });
            let grad = grads.get(&a).unwrap();
            if let Some(g) = grad {
                assert_eq!(g.shape(), &shape[..]);
            }
        }
    }
}
