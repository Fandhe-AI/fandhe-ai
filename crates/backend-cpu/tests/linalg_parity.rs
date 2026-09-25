//! 線形代数（inv／solve／det／qr／cholesky／svd）・matrix_norm の
//! `BackendOps` 経由（`CpuBackendOps`）統合テスト（イシュー #1621・
//! `docs/autodiff-linalg-design.md`）。
//!
//! `src/linalg.rs` の `#[cfg(test)] mod tests` はクレート内部の自由関数
//! （`linalg::inv` 等）を直接呼ぶユニットテストであり、`BackendOps::
//! linalg_*`（`ops.rs::CpuBackendOps` の trait 実装）を経由しない。本
//! ファイルは trait 経由の到達性・REQ-2 複合判定（`assert_parity`）・
//! 決定性（同一入力 2 回実行の bit 一致）を検証する（`docs/
//! autodiff-linalg-design.md` §4 のテスト計画 (a)(c)(d) 相当）。

use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, MatrixNormOrd, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    let c = tensor.contiguous();
    c.as_slice().map(|s| s.to_vec()).unwrap_or_default()
}

// --- inv ---

#[test]
fn linalg_inv_matches_known_solution() {
    let ops = CpuBackendOps::new();
    let a = t(vec![4.0, 7.0, 2.0, 6.0], &[2, 2]);
    let inv_a = ops.linalg_inv(&a).unwrap();
    assert_parity("linalg_inv", &dense(&inv_a), &[0.6, -0.7, -0.2, 0.4]);
}

#[test]
fn linalg_inv_singular_is_invalid_argument() {
    let ops = CpuBackendOps::new();
    let a = t(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]);
    assert!(matches!(
        ops.linalg_inv(&a),
        Err(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn linalg_inv_is_bit_deterministic_across_repeated_calls() {
    let ops = CpuBackendOps::new();
    let a = t(vec![4.0, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, -1.0, 3.5], &[3, 3]);
    let r1 = ops.linalg_inv(&a).unwrap();
    let r2 = ops.linalg_inv(&a).unwrap();
    assert_eq!(dense(&r1), dense(&r2));
}

// --- solve ---

#[test]
fn linalg_solve_matches_known_solution() {
    let ops = CpuBackendOps::new();
    let a = t(vec![3.0, 1.0, 1.0, 2.0], &[2, 2]);
    let b = t(vec![9.0, 8.0], &[2, 1]);
    let x = ops.linalg_solve(&a, &b).unwrap();
    assert_parity("linalg_solve", &dense(&x), &[2.0, 3.0]);
}

// --- det ---

#[test]
fn linalg_det_known_value_and_singular_returns_zero() {
    let ops = CpuBackendOps::new();
    let a = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_parity("linalg_det", &dense(&ops.linalg_det(&a).unwrap()), &[-2.0]);

    let singular = t(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]);
    assert_parity(
        "linalg_det singular",
        &dense(&ops.linalg_det(&singular).unwrap()),
        &[0.0],
    );
}

// --- cholesky ---

#[test]
fn linalg_cholesky_reconstructs_spd_matrix() {
    let ops = CpuBackendOps::new();
    // A = [[4,2],[2,3]] は対称正定値。
    let a = t(vec![4.0, 2.0, 2.0, 3.0], &[2, 2]);
    let l = ops.linalg_cholesky(&a).unwrap();
    assert_parity("linalg_cholesky", &dense(&l), &[2.0, 0.0, 1.0, 2f32.sqrt()]);
}

#[test]
fn linalg_cholesky_non_positive_definite_is_invalid_argument() {
    let ops = CpuBackendOps::new();
    let a = t(vec![1.0, 2.0, 2.0, 1.0], &[2, 2]);
    assert!(matches!(
        ops.linalg_cholesky(&a),
        Err(BackendError::InvalidArgument(_))
    ));
}

// --- qr ---

#[test]
fn linalg_qr_reconstructs_input_and_is_orthonormal() {
    let ops = CpuBackendOps::new();
    let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0], &[3, 2]);
    let factors = ops.linalg_qr(&a).unwrap();
    assert_eq!(factors.q.shape(), &[3, 2]);
    assert_eq!(factors.r.shape(), &[2, 2]);
    let reconstructed = ops.gemm(&factors.q, &factors.r).unwrap();
    assert_parity("linalg_qr reconstruct", &dense(&reconstructed), &dense(&a));

    let qt = factors.q.transpose(0, 1).unwrap().contiguous();
    let qtq = ops.gemm(&qt, &factors.q).unwrap();
    assert_parity("linalg_qr orthonormal", &dense(&qtq), &[1.0, 0.0, 0.0, 1.0]);
}

// --- svd ---

#[test]
fn linalg_svd_reconstructs_square_input() {
    let ops = CpuBackendOps::new();
    let mut a_data = vec![0.0f32; 9];
    let sigmas = [3.0f32, 2.0, 1.0];
    for i in 0..3 {
        a_data[i * 3 + i] = sigmas[i];
    }
    let a = t(a_data, &[3, 3]);
    let factors = ops.linalg_svd(&a).unwrap();
    assert_eq!(factors.u.shape(), &[3, 3]);
    assert_eq!(factors.s.shape(), &[3]);
    assert_eq!(factors.vh.shape(), &[3, 3]);
    assert_parity("linalg_svd singular values", &dense(&factors.s), &sigmas);
}

#[test]
fn linalg_svd_is_bit_deterministic_across_repeated_calls() {
    let ops = CpuBackendOps::new();
    let a = t(vec![3.0, 0.3, 0.1, 0.2, 2.0, 0.2, 0.1, 0.15, 1.0], &[3, 3]);
    let f1 = ops.linalg_svd(&a).unwrap();
    let f2 = ops.linalg_svd(&a).unwrap();
    assert_eq!(dense(&f1.u), dense(&f2.u));
    assert_eq!(dense(&f1.s), dense(&f2.s));
    assert_eq!(dense(&f1.vh), dense(&f2.vh));
}

// --- matrix_norm（5 ord） ---

#[test]
fn linalg_matrix_norm_fro_known_value() {
    let ops = CpuBackendOps::new();
    let a = t(vec![3.0, 4.0], &[1, 2]);
    let norm = ops.linalg_matrix_norm(&a, MatrixNormOrd::Fro).unwrap();
    assert_parity("linalg_matrix_norm fro", &dense(&norm), &[5.0]);
}

#[test]
fn linalg_matrix_norm_one_and_inf_known_values() {
    let ops = CpuBackendOps::new();
    let a = t(vec![2.0, -1.0, -6.0, 3.0], &[2, 2]);
    let one = ops.linalg_matrix_norm(&a, MatrixNormOrd::One).unwrap();
    let inf = ops.linalg_matrix_norm(&a, MatrixNormOrd::Inf).unwrap();
    assert_parity("linalg_matrix_norm one", &dense(&one), &[8.0]);
    assert_parity("linalg_matrix_norm inf", &dense(&inf), &[9.0]);
}

#[test]
fn linalg_matrix_norm_nuc_and_spectral_are_consistent_with_svd() {
    let ops = CpuBackendOps::new();
    let a = t(vec![3.0, 0.3, 0.1, 0.2, 2.0, 0.2, 0.1, 0.15, 1.0], &[3, 3]);
    let factors = ops.linalg_svd(&a).unwrap();
    let s = dense(&factors.s);
    let nuc = ops.linalg_matrix_norm(&a, MatrixNormOrd::Nuc).unwrap();
    let spectral = ops.linalg_matrix_norm(&a, MatrixNormOrd::Spectral).unwrap();
    assert_parity("linalg_matrix_norm nuc", &dense(&nuc), &[s.iter().sum()]);
    assert_parity("linalg_matrix_norm spectral", &dense(&spectral), &[s[0]]);
}

// --- 空行列（n=0） ---

#[test]
fn linalg_empty_matrix_contracts() {
    let ops = CpuBackendOps::new();
    let a = t(Vec::new(), &[0, 0]);
    let inv_a = ops.linalg_inv(&a).unwrap();
    assert_eq!(inv_a.shape(), &[0, 0]);
    let det_a = ops.linalg_det(&a).unwrap();
    assert_parity("linalg_det empty", &dense(&det_a), &[1.0]);
}

// --- eigh（イシュー #2150） ---

#[test]
fn linalg_eigh_known_symmetric_matrix() {
    let ops = CpuBackendOps::new();
    let a = t(vec![2.0, 1.0, 1.0, 2.0], &[2, 2]);
    let factors = ops.linalg_eigh(&a).unwrap();
    assert_parity(
        "linalg_eigh eigenvalues",
        &dense(&factors.eigenvalues),
        &[1.0, 3.0],
    );
}

#[test]
fn linalg_eigh_non_square_is_invalid_argument() {
    let ops = CpuBackendOps::new();
    let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert!(matches!(
        ops.linalg_eigh(&a),
        Err(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn linalg_eigh_is_bit_deterministic_across_repeated_calls() {
    let ops = CpuBackendOps::new();
    let a = t(vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0], &[3, 3]);
    let r1 = ops.linalg_eigh(&a).unwrap();
    let r2 = ops.linalg_eigh(&a).unwrap();
    assert_eq!(dense(&r1.eigenvalues), dense(&r2.eigenvalues));
    assert_eq!(dense(&r1.eigenvectors), dense(&r2.eigenvectors));
}

// --- slogdet（イシュー #2150） ---

#[test]
fn linalg_slogdet_known_value() {
    let ops = CpuBackendOps::new();
    let a = t(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let factors = ops.linalg_slogdet(&a).unwrap();
    assert_parity("linalg_slogdet sign", &dense(&factors.sign), &[1.0]);
    assert_parity(
        "linalg_slogdet logabsdet",
        &dense(&factors.logabsdet),
        &[6.0f32.ln()],
    );
}

#[test]
fn linalg_slogdet_singular_matrix() {
    let ops = CpuBackendOps::new();
    let a = t(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]);
    let factors = ops.linalg_slogdet(&a).unwrap();
    assert_eq!(dense(&factors.sign)[0], 0.0);
    assert_eq!(dense(&factors.logabsdet)[0], f32::NEG_INFINITY);
}

// --- pinv（イシュー #2150） ---

#[test]
fn linalg_pinv_full_rank_matches_inverse() {
    let ops = CpuBackendOps::new();
    let a = t(vec![4.0, 7.0, 2.0, 6.0], &[2, 2]);
    let p = ops.linalg_pinv(&a, None).unwrap();
    assert_parity("linalg_pinv", &dense(&p), &[0.6, -0.7, -0.2, 0.4]);
}

#[test]
fn linalg_pinv_is_bit_deterministic_across_repeated_calls() {
    let ops = CpuBackendOps::new();
    let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let r1 = ops.linalg_pinv(&a, None).unwrap();
    let r2 = ops.linalg_pinv(&a, None).unwrap();
    assert_eq!(dense(&r1), dense(&r2));
}

// --- matrix_rank（イシュー #2150） ---

#[test]
fn linalg_matrix_rank_known_values() {
    let ops = CpuBackendOps::new();
    let full = t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let deficient = t(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]);
    assert_parity(
        "linalg_matrix_rank full",
        &dense(&ops.linalg_matrix_rank(&full, None).unwrap()),
        &[2.0],
    );
    assert_parity(
        "linalg_matrix_rank deficient",
        &dense(&ops.linalg_matrix_rank(&deficient, None).unwrap()),
        &[1.0],
    );
}

#[test]
fn linalg_matrix_rank_rejects_invalid_rcond() {
    let ops = CpuBackendOps::new();
    let a = t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    assert!(matches!(
        ops.linalg_matrix_rank(&a, Some(-1.0)),
        Err(BackendError::InvalidArgument(_))
    ));
}

/// PR #2268 codex-review〈P2〉指摘の回帰: `pinv`／`matrix_rank`／
/// `lstsq` は空行列（`m==0` または `n==0`）の早期 return を `rcond`
/// 検証より前に置いていたため、`Some(NaN)` や負値が空行列入力では
/// 素通りしていた（`src/linalg.rs::resolve_rcond` 呼び出し順序の是正）。
#[test]
fn linalg_rejects_invalid_rcond_on_empty_matrix() {
    let ops = CpuBackendOps::new();
    let a = t(Vec::new(), &[0, 3]);
    let b = t(Vec::new(), &[0, 1]);
    for r in [-1.0f32, f32::NAN, f32::INFINITY] {
        assert!(
            matches!(
                ops.linalg_pinv(&a, Some(r)),
                Err(BackendError::InvalidArgument(_))
            ),
            "linalg_pinv: r={r}"
        );
        assert!(
            matches!(
                ops.linalg_matrix_rank(&a, Some(r)),
                Err(BackendError::InvalidArgument(_))
            ),
            "linalg_matrix_rank: r={r}"
        );
        assert!(
            matches!(
                ops.linalg_lstsq(&a, &b, Some(r)),
                Err(BackendError::InvalidArgument(_))
            ),
            "linalg_lstsq: r={r}"
        );
    }
}

// --- lstsq（イシュー #2150） ---

#[test]
fn linalg_lstsq_overdetermined_matches_known_solution() {
    let ops = CpuBackendOps::new();
    let a = t(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
    let b = t(vec![1.0, 2.0, 4.0], &[3, 1]);
    let x = ops.linalg_lstsq(&a, &b, None).unwrap();
    let data = dense(&x);
    assert!((data[0] - 4.0 / 3.0).abs() < 1e-4);
    assert!((data[1] - 7.0 / 3.0).abs() < 1e-4);
}

#[test]
fn linalg_lstsq_row_mismatch_is_invalid_argument() {
    let ops = CpuBackendOps::new();
    let a = t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = t(vec![1.0, 2.0, 3.0], &[3, 1]);
    assert!(matches!(
        ops.linalg_lstsq(&a, &b, None),
        Err(BackendError::InvalidArgument(_))
    ));
}

// --- 空行列（n=0）: 5 新演算 ---

#[test]
fn linalg_new_ops_empty_matrix_contracts() {
    let ops = CpuBackendOps::new();
    let empty = t(Vec::new(), &[0, 0]);
    let eigh_factors = ops.linalg_eigh(&empty).unwrap();
    assert_eq!(eigh_factors.eigenvalues.shape(), &[0]);
    assert_eq!(eigh_factors.eigenvectors.shape(), &[0, 0]);
    let slogdet_factors = ops.linalg_slogdet(&empty).unwrap();
    assert_parity(
        "linalg_slogdet empty sign",
        &dense(&slogdet_factors.sign),
        &[1.0],
    );
    let tall_zero = t(Vec::new(), &[0, 3]);
    let p = ops.linalg_pinv(&tall_zero, None).unwrap();
    assert_eq!(p.shape(), &[3, 0]);
    let rank = ops.linalg_matrix_rank(&tall_zero, None).unwrap();
    assert_parity("linalg_matrix_rank empty", &dense(&rank), &[0.0]);
}
