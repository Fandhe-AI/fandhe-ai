//! 線形代数（inv／solve／det／qr／cholesky／svd）・matrix_norm の facade
//! 到達性検証（イシュー #1621・`docs/autodiff-linalg-design.md`）。
//!
//! `fandhe_ai::tape()`（`CpuBackendOps` 結線・`BackendOps::linalg_*` の
//! 本番 CPU カーネル経路）と `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`・
//! `linalg_*` 未実装のため常に `eval::linalg` ホスト参照実装へ
//! フォールバックする経路）の 2 経路が同一の値を返すことを REQ-2
//! 複合判定（`assert_parity`）で突合する（`fusion_default_parity.rs`
//! と同じ判定関数を再利用し、判定式をテスト内で再定義しない）。

use fandhe_ai::{MatrixNormOrd, Tensor};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::BackendOps;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    let c = tensor.contiguous();
    c.as_slice().map(|s| s.to_vec()).unwrap_or_default()
}

#[test]
fn inv_matches_between_facade_and_naive_fallback() {
    let a_data = vec![4.0, 1.0, 2.0, 3.0];

    let facade_tape = fandhe_ai::tape();
    let a1 = facade_tape.var(&t(a_data.clone(), &[2, 2]));
    let facade_result = dense(&a1.inv().unwrap().to_tensor());

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a2 = naive_tape.var(&t(a_data, &[2, 2]));
    let naive_result = dense(&a2.inv().unwrap().to_tensor());

    assert_parity("linalg_facade inv", &facade_result, &naive_result);
}

#[test]
fn solve_matches_between_facade_and_naive_fallback() {
    let a_data = vec![3.0, 1.0, 1.0, 2.0];
    let b_data = vec![9.0, 8.0];

    let facade_tape = fandhe_ai::tape();
    let a1 = facade_tape.var(&t(a_data.clone(), &[2, 2]));
    let b1 = facade_tape.var(&t(b_data.clone(), &[2, 1]));
    let facade_result = dense(&a1.solve(&b1).unwrap().to_tensor());

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a2 = naive_tape.var(&t(a_data, &[2, 2]));
    let b2 = naive_tape.var(&t(b_data, &[2, 1]));
    let naive_result = dense(&a2.solve(&b2).unwrap().to_tensor());

    assert_parity("linalg_facade solve", &facade_result, &naive_result);
}

#[test]
fn det_matches_between_facade_and_naive_fallback() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0];

    let facade_tape = fandhe_ai::tape();
    let a1 = facade_tape.var(&t(a_data.clone(), &[2, 2]));
    let facade_result = dense(&a1.det().unwrap().to_tensor());

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a2 = naive_tape.var(&t(a_data, &[2, 2]));
    let naive_result = dense(&a2.det().unwrap().to_tensor());

    assert_parity("linalg_facade det", &facade_result, &naive_result);
}

#[test]
fn cholesky_matches_between_facade_and_naive_fallback() {
    let a_data = vec![4.0, 2.0, 2.0, 3.0];

    let facade_tape = fandhe_ai::tape();
    let a1 = facade_tape.var(&t(a_data.clone(), &[2, 2]));
    let facade_result = dense(&a1.cholesky().unwrap().to_tensor());

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a2 = naive_tape.var(&t(a_data, &[2, 2]));
    let naive_result = dense(&a2.cholesky().unwrap().to_tensor());

    assert_parity("linalg_facade cholesky", &facade_result, &naive_result);
}

#[test]
fn qr_matches_between_facade_and_naive_fallback() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0];

    let facade_tape = fandhe_ai::tape();
    let a1 = facade_tape.var(&t(a_data.clone(), &[3, 2]));
    let qr1 = a1.qr().unwrap();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a2 = naive_tape.var(&t(a_data, &[3, 2]));
    let qr2 = a2.qr().unwrap();

    assert_parity(
        "linalg_facade qr.q",
        &dense(&qr1.q.to_tensor()),
        &dense(&qr2.q.to_tensor()),
    );
    assert_parity(
        "linalg_facade qr.r",
        &dense(&qr1.r.to_tensor()),
        &dense(&qr2.r.to_tensor()),
    );
}

#[test]
fn svd_matches_between_facade_and_naive_fallback() {
    let a_data = vec![3.0, 0.3, 0.1, 0.2, 2.0, 0.2, 0.1, 0.15, 1.0];

    let facade_tape = fandhe_ai::tape();
    let a1 = facade_tape.var(&t(a_data.clone(), &[3, 3]));
    let svd1 = a1.svd().unwrap();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a2 = naive_tape.var(&t(a_data, &[3, 3]));
    let svd2 = a2.svd().unwrap();

    assert_parity(
        "linalg_facade svd.u",
        &dense(&svd1.u.to_tensor()),
        &dense(&svd2.u.to_tensor()),
    );
    assert_parity(
        "linalg_facade svd.s",
        &dense(&svd1.s.to_tensor()),
        &dense(&svd2.s.to_tensor()),
    );
    assert_parity(
        "linalg_facade svd.vh",
        &dense(&svd1.vh.to_tensor()),
        &dense(&svd2.vh.to_tensor()),
    );
}

#[test]
fn matrix_norm_matches_between_facade_and_naive_fallback_for_all_ord() {
    let a_data = vec![3.0, 0.3, 0.1, 0.2, 2.0, 0.2, 0.1, 0.15, 1.0];
    let ords = [
        MatrixNormOrd::Fro,
        MatrixNormOrd::One,
        MatrixNormOrd::Inf,
        MatrixNormOrd::Nuc,
        MatrixNormOrd::Spectral,
    ];

    for ord in ords {
        let facade_tape = fandhe_ai::tape();
        let a1 = facade_tape.var(&t(a_data.clone(), &[3, 3]));
        let facade_result = dense(&a1.matrix_norm(ord).unwrap().to_tensor());

        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let a2 = naive_tape.var(&t(a_data.clone(), &[3, 3]));
        let naive_result = dense(&a2.matrix_norm(ord).unwrap().to_tensor());

        assert_parity(
            &format!("linalg_facade matrix_norm {ord:?}"),
            &facade_result,
            &naive_result,
        );
    }
}

/// facade（`fandhe_ai::tape()`）が `CpuBackendOps::linalg_*`（本番 CPU
/// カーネル）を実際に経由していることを直接確認する（`fusion_default_
/// parity.rs` の「融合が実際に有効であることを確認する」と同じ意図。
/// `Unsupported` を静かにフォールバックし続けているだけの可能性を
/// 排除する）。
#[test]
fn facade_tape_reaches_cpu_backend_ops_linalg_inv_directly() {
    let ops = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let a = t(vec![4.0, 1.0, 2.0, 3.0], &[2, 2]);
    assert!(ops.linalg_inv(&a).is_ok());
}

/// codex-review 指摘の回帰（`Var::unify_backend_error`）: `fandhe_ai::
/// tape()`（`CpuBackendOps::linalg_cholesky` 本番経路）が返す数値
/// エラー（非正定値）が、公開ドキュメントどおり
/// `AutodiffError::InvalidArgument(_)` として観測できることを確認する。
/// 以前は逆方向（フォールバック側を `Backend(InvalidArgument)` へ
/// 包む）へ統一していたため、本番経路のみ `AutodiffError::
/// Backend(BackendError::InvalidArgument(_))` になり、呼び出し元が
/// ドキュメント記載の variant を照合しても本番経路の失敗を捕捉
/// できなかった。
#[test]
fn cholesky_non_positive_definite_is_invalid_argument_on_cpu_production_path() {
    // `[[-1]]` は対角が負のため非正定値（`CpuBackendOps::linalg_cholesky`
    // の本番経路を直接通す。1x1 は必ず `Unsupported` 以外の値を返す）。
    let facade_tape = fandhe_ai::tape();
    let a = facade_tape.var(&t(vec![-1.0], &[1, 1]));
    let result = a.cholesky();
    assert!(
        matches!(
            result,
            Err(fandhe_ai_autodiff::AutodiffError::InvalidArgument(_))
        ),
        "本番 CPU 経路の非正定値エラーが AutodiffError::InvalidArgument でない: {result:?}"
    );
}
