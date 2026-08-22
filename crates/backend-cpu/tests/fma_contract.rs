//! `backend-cpu::parity` の受け入れ基準対応テスト（TASK-2.2a・#53）。
//!
//! 受け入れ条件は「全ペア共通の複合判定ユーティリティが成立すること」。
//! 本ファイルは [`matmul_reference_fma`] を FMA 契約（REQ-2）の参照点として
//! 固定する側を担い、以下を検証する。
//!
//! 1. `matmul_reference_fma` 自体の正しさ（手計算・非正方境界形状）。
//! 2. 全公開 GEMM 入口（`gemm_naive`/`gemm_blocked`/`gemm_parallel`/
//!    `gemm_parallel_tuned`/`gemm_blis`/`gemm_blis_parallel`）が
//!    `matmul_reference_fma` と bit 完全一致すること（K=4096 ストレス形状。
//!    `tests/gemm_blis_parity.rs::gemm_blis_uses_mul_add_fma_contract` と
//!    同規模）。
//! 3. falsification: 非 FMA 蓄積（`acc += a * b`）の参照実装との出力差を
//!    `parity::compare` が FAIL として検出すること（判定ロジック自体が
//!    「常に PASS を返す」壊れ方をしていないこと、かつ FMA 契約退行の
//!    見逃しがないことの二重確認。PoC-v2-5 の非 FMA CPU 参照が
//!    512×512×4096 で fail_cells=7/262144 を実測した知見を踏襲し、
//!    小規模形状でも決定的に fail が出るシードを固定する）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::{
    BlockSizes, compare, gemm_blis, gemm_blis_parallel, gemm_blocked, gemm_naive, gemm_parallel,
    gemm_parallel_tuned, matmul_reference_fma,
};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

// --- 1. matmul_reference_fma 自体の正しさ ---

#[test]
fn matmul_reference_fma_matches_hand_computed_2x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];
    matmul_reference_fma(&a, &b, &mut c, 2, 2, 2).unwrap();
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn matmul_reference_fma_handles_non_square_boundary_shape() {
    // m=3（行数）× k=1（内積なし・単純スケーリング）× n=2 の縮退形状。
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![10.0, 20.0];
    let mut c = vec![0.0; 6];
    matmul_reference_fma(&a, &b, &mut c, 3, 2, 1).unwrap();
    assert_eq!(c, vec![10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
}

// --- 2. 全公開 GEMM 入口が matmul_reference_fma と bit 完全一致（FMA 契約固定） ---
//
// K=4096 は PoC-v2-5 の GPU 数値一致ストレスケースと同一規模
// （`.claude/rules/coding-rust.md`）。M=N=8 に抑えることで
// デバッグビルドでも高速に実行できる（`tests/gemm_blis_parity.rs`
// `gemm_blis_uses_mul_add_fma_contract` と同じ形状選定）。

#[test]
fn all_gemm_entrypoints_match_matmul_reference_fma_bit_exact() {
    let (m, n, k) = (8, 8, 4096);
    let a = random_matrix(50, m * k);
    let b = random_matrix(51, k * n);

    let mut c_ref = vec![0.0f32; m * n];
    matmul_reference_fma(&a, &b, &mut c_ref, m, n, k).unwrap();

    let mut c_naive = vec![0.0f32; m * n];
    gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();
    assert_eq!(
        c_ref, c_naive,
        "gemm_naive が FMA 契約参照点と bit 一致しない"
    );

    let mut c_blocked = vec![0.0f32; m * n];
    gemm_blocked(&a, &b, &mut c_blocked, m, n, k).unwrap();
    assert_eq!(
        c_ref, c_blocked,
        "gemm_blocked が FMA 契約参照点と bit 一致しない"
    );

    let mut c_parallel = vec![0.0f32; m * n];
    gemm_parallel(&a, &b, &mut c_parallel, m, n, k).unwrap();
    assert_eq!(
        c_ref, c_parallel,
        "gemm_parallel が FMA 契約参照点と bit 一致しない"
    );

    let mut c_parallel_tuned = vec![0.0f32; m * n];
    gemm_parallel_tuned(
        &a,
        &b,
        &mut c_parallel_tuned,
        m,
        n,
        k,
        BlockSizes::poc_v2_1_default(),
        1,
    )
    .unwrap();
    assert_eq!(
        c_ref, c_parallel_tuned,
        "gemm_parallel_tuned が FMA 契約参照点と bit 一致しない"
    );

    let mut c_blis = vec![0.0f32; m * n];
    gemm_blis(&a, &b, &mut c_blis, m, n, k).unwrap();
    assert_eq!(
        c_ref, c_blis,
        "gemm_blis が FMA 契約参照点と bit 一致しない"
    );

    let mut c_blis_parallel = vec![0.0f32; m * n];
    gemm_blis_parallel(&a, &b, &mut c_blis_parallel, m, n, k).unwrap();
    assert_eq!(
        c_ref, c_blis_parallel,
        "gemm_blis_parallel が FMA 契約参照点と bit 一致しない"
    );
}

// --- 3. falsification: 非 FMA 蓄積との差分を複合判定が検出する ---

/// PoC の `c += a * b`（乗算・加算を別々に丸める非 FMA 蓄積）を再現する
/// テスト専用ヘルパー。`matmul_reference_fma` との対比により FMA 契約の
/// 効果（丸め差）を注入する（本体実装には存在しない意図的な非契約実装）。
fn matmul_non_fma_reference(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        let a_row = &a[i * k..i * k + k];
        let c_row = &mut c[i * n..i * n + n];
        for (p, &a_ip) in a_row.iter().enumerate() {
            let b_row = &b[p * n..p * n + n];
            for j in 0..n {
                c_row[j] += a_ip * b_row[j];
            }
        }
    }
}

/// falsification test（PoC-v2-4/v2-5 の前例を踏襲）: 複合判定
/// （[`compare`]）が「常に PASS を返す」壊れ方をしていないこと、かつ
/// FMA 契約からの退行（非 FMA 蓄積への後退）を見逃さないことの二重確認。
///
/// 512×512×4096（PoC-v2-5 の非 FMA CPU 参照ストレスケースと同一規模。
/// `docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md` に
/// fail_cells=7/262144 の実測記録あり）でシード 60/61 を固定したところ
/// 本形状でも決定的に fail_count > 0 となることを確認済み（CPU 演算は
/// 決定的なため、一度固定すれば実行環境によらず安定して再現する）。
#[test]
fn falsification_non_fma_accumulation_is_detected_by_compare() {
    let (m, n, k) = (512, 512, 4096);
    let a = random_matrix(60, m * k);
    let b = random_matrix(61, k * n);

    let mut c_fma = vec![0.0f32; m * n];
    matmul_reference_fma(&a, &b, &mut c_fma, m, n, k).unwrap();

    let mut c_non_fma = vec![0.0f32; m * n];
    matmul_non_fma_reference(&a, &b, &mut c_non_fma, m, n, k);

    // 非 FMA 参照は FMA 参照と bit レベルでは一致しないはず（丸め差の注入
    // 自体が成立していることの前提確認）。
    assert_ne!(
        c_fma, c_non_fma,
        "非 FMA 参照が FMA 参照と bit 一致した（丸め差の注入に失敗している）"
    );

    let report = compare(&c_fma, &c_non_fma).unwrap();
    assert!(
        !report.passes(),
        "複合判定が非 FMA 蓄積との差分を FAIL として検出できなかった \
         （fail_count={}/{}, max_abs_diff={:.3e}, max_rel_err={:.3e}）",
        report.fail_count,
        report.total,
        report.max_abs_diff,
        report.max_rel_err,
    );
}
