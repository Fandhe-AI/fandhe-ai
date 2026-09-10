//! split-K 2 パス GEMM（イシュー #1474）の**自動判定入口**
//! （[`fandhe_ai_backend_metal::MetalGemm::dispatch_split_k_strided_prepared`]）
//! 自体の受け入れテスト。
//!
//! # 位置づけ（`tests/gemm_splitk_parity.rs` との違い）
//!
//! `tests/gemm_splitk_parity.rs`（AC-2）は `should_split_k` の算出した
//! 計画を `_with_plan`（`SPLIT_K_NUMERIC_CONTRACT_APPROVED` ゲートの対象外・
//! `internal-diagnostics` feature 限定）へ直接渡し、split-K 経路自体の
//! 正しさ（実測ベースライン非後退契約）を検証する。本ファイルは
//! **ゲート付き公開入口** `dispatch_split_k_strided_prepared`（既定ビルドで
//! `pub`・`required-features` なし）を直接呼び、イシュー #1513 で
//! `SPLIT_K_NUMERIC_CONTRACT_APPROVED` を `true` へ切り替えた結果として、
//!
//! - (positive) `crate::tile::should_split_k` が `Some` を返す承認済み
//!   11 形状（`common::splitk_parity_baseline::BASELINES`）× NN/NT/TN/TT
//!   で、公開入口が実際に [`SplitKRoute::Split`] を返し、かつその出力が
//!   `assert_no_split_k_parity_regression`（実測ベースライン非後退契約。
//!   tolerance 定数は変更しない）を満たすこと
//! - (negative) `should_split_k` が `None` を返す形状では、公開入口が
//!   `SplitKRoute::Classic { reason: SplitKFallbackReason::NotEligible }`
//!   を返し、**`NumericContractPendingApproval` では決してない**こと
//!   （ゲート解除自体の直接検証。`NumericContractPendingApproval` は
//!   後方互換のため enum には残っているが本テスト時点ではもう返らない）
//!
//! を確認する。`dispatch_auto`／`crate::tile::select_for_device` への
//! 本番結線はイシュー #1513 のスコープ外（#1516 へ引き継ぎ）であり、
//! `crate::ops::MetalBackendOps::gemm` は本テストが呼ぶ入口を経由しない
//! （`crates/backend-metal/src/ops.rs` に `dispatch_split_k_strided_prepared`
//! の呼び出しがないことを実装計画時点で `grep` 確認済み）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。`#[ignore]` に
//! より通常の `cargo test` からは除外される（`tests/gemm_splitk_parity.rs`
//! と同じ方針）。`required-features` は指定しない
//! （`dispatch_split_k_strided_prepared` 自体は既定ビルドで `pub`。これにより
//! `cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin`
//! の型検査対象に含まれる）。
//!
//! 実機実行（Apple Silicon 必須）:
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test gemm_splitk_auto_entry_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

mod common;

use bench_harness::rng::Xorshift64Star;
use common::splitk_parity_baseline::{assert_no_split_k_parity_regression, find_baseline};
use fandhe_ai_backend_cpu::parity::{compare, matmul_reference_fma};
use fandhe_ai_backend_metal::layout::{MatrixLayout, classify_2d};
use fandhe_ai_backend_metal::tile;
use fandhe_ai_backend_metal::{
    MetalBuffer, MetalContext, MetalGemm, SplitKFallbackReason, SplitKRoute,
};

/// `logical`（行優先の論理 `[rows, cols]`）から `[cols, rows]` 行優先の
/// 転置済み物理バッファを作る（`tests/gemm_splitk_parity.rs::
/// transpose_dense` と同一実装）。
fn transpose_dense(logical: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = logical[r * cols + c];
        }
    }
    out
}

/// `tests/gemm_splitk_parity.rs::TARGET_SHAPES` と同一の承認済み 11
/// 形状（`common::splitk_parity_baseline::BASELINES` が 1 対 1 対応）。
const TARGET_SHAPES: &[(usize, usize, usize)] = &[
    (32, 32, 2048),
    (32, 32, 4096),
    (32, 32, 8192),
    (64, 64, 2048),
    (64, 64, 4096),
    (64, 64, 8192),
    (128, 128, 2048),
    (128, 128, 4096),
    (128, 128, 8192),
    (64, 64, 2056),
    (128, 128, 2064),
];

/// `should_split_k` が `None` を返す（対象条件を満たさない）形状。
/// `crates/backend-metal/src/tile.rs` の
/// `should_split_k_rejects_large_square_and_wide_shapes`／
/// `should_split_k_rejects_k_below_max_m_n` が Linux で確認済みの事実
/// （正方 512 以上は並列度条件で除外・K が M/N 未満は Case 1 不成立）を
/// 転用する。
const NON_ELIGIBLE_SHAPES: &[(usize, usize, usize)] = &[(512, 512, 512), (64, 64, 63)];

/// (positive) 承認済み 11 形状 × NN/NT/TN/TT で、公開入口
/// `dispatch_split_k_strided_prepared` が実際に split-K 経路を実行し、
/// その出力が実測ベースライン非後退契約を満たすことを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn auto_entry_dispatches_split_k_for_eligible_shapes_and_matches_baseline() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in TARGET_SHAPES {
        let baseline = find_baseline(m, n, k).unwrap_or_else(|| {
            panic!(
                "m={m}, n={n}, k={k} に対応する記録済みベースラインが見つからない \
                 （TARGET_SHAPES と common::splitk_parity_baseline::BASELINES の対応が \
                 崩れている）"
            )
        });
        let expected_plan = tile::should_split_k(m, n, k).unwrap_or_else(|| {
            panic!(
                "should_split_k が None を返した（承認済み形状の前提が崩れている）: \
                 m={m}, n={n}, k={k}"
            )
        });

        let a_logical = Xorshift64Star::new(m as u64 * 7 + k as u64 + 1).fill_vec(m * k);
        let b_logical = Xorshift64Star::new(n as u64 * 11 + k as u64 + 2).fill_vec(k * n);
        let mut expected = vec![0.0f32; m * n];
        matmul_reference_fma(&a_logical, &b_logical, &mut expected, m, n, k)
            .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

        for (trans_a, trans_b) in [(false, false), (false, true), (true, false), (true, true)] {
            let (a_phys, a_layout): (Vec<f32>, MatrixLayout) = if trans_a {
                (
                    transpose_dense(&a_logical, m, k),
                    classify_2d(&[m, k], &[1, m as isize]).unwrap(),
                )
            } else {
                (
                    a_logical.clone(),
                    classify_2d(&[m, k], &[k as isize, 1]).unwrap(),
                )
            };
            let (b_phys, b_layout): (Vec<f32>, MatrixLayout) = if trans_b {
                (
                    transpose_dense(&b_logical, k, n),
                    classify_2d(&[k, n], &[1, k as isize]).unwrap(),
                )
            } else {
                (
                    b_logical.clone(),
                    classify_2d(&[k, n], &[n as isize, 1]).unwrap(),
                )
            };

            let a_buf =
                MetalBuffer::new_with_data(&ctx, &a_phys).expect("A バッファ確保に失敗した");
            let b_buf =
                MetalBuffer::new_with_data(&ctx, &b_phys).expect("B バッファ確保に失敗した");
            let c_buf = MetalBuffer::new_zeroed(&ctx, m * n).expect("C バッファ確保に失敗した");

            // ゲート付き公開入口（`should_split_k` の自動判定を経由する）
            // を直接呼ぶ。ここが `_with_plan` を使う `gemm_splitk_parity.rs`
            // との違い（イシュー #1513 のゲート解除自体の検証）。
            let route = gemm
                .dispatch_split_k_strided_prepared(
                    &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf, m, n, k,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "dispatch_split_k_strided_prepared failed (trans_a={trans_a}, \
                         trans_b={trans_b}, m={m}, n={n}, k={k}): {e}"
                    )
                });

            let partitions = match route {
                SplitKRoute::Split { partitions, .. } => partitions,
                SplitKRoute::Classic { reason, .. } => panic!(
                    "trans_a={trans_a}, trans_b={trans_b}, m={m}, n={n}, k={k}: ゲート解除\
                     済みのはずが classic 経路へフォールバックした（reason={reason:?}）。\
                     `SPLIT_K_NUMERIC_CONTRACT_APPROVED` が true であることを確認せよ。"
                ),
            };
            assert_eq!(
                partitions, expected_plan.partitions,
                "trans_a={trans_a}, trans_b={trans_b}, m={m}, n={n}, k={k}: 自動判定入口が\
                 選んだ partitions が should_split_k の算出値と一致しない"
            );

            let actual = c_buf.read_to_vec();
            let report = compare(&actual, &expected).unwrap_or_else(|e| {
                panic!(
                    "parity compare failed (trans_a={trans_a}, trans_b={trans_b}, m={m}, n={n}, \
                     k={k}): {e}"
                )
            });

            println!(
                "[auto-entry] m={m} n={n} k={k} trans_a={trans_a} trans_b={trans_b} \
                 fail_count={}/{} max_abs_diff={:.10} mean_abs_diff={:.10} max_rel_err={:.8}",
                report.fail_count,
                report.total,
                report.max_abs_diff,
                report.mean_abs_diff,
                report.max_rel_err
            );

            let context = format!(
                "auto entry split-K parity vs CPU reference (trans_a={trans_a}, \
                 trans_b={trans_b}, m={m}, n={n}, k={k})"
            );
            assert_no_split_k_parity_regression(&context, &report, baseline);
        }
    }
}

/// (negative) `should_split_k` が `None` を返す形状では、公開入口が
/// classic 経路（`SplitKFallbackReason::NotEligible`）へフォールバック
/// し、**`NumericContractPendingApproval` では決してない**こと
/// （ゲート解除自体の直接検証）。classic 経路の出力は既存 bit 一致群
/// （`tests/gemm_splitk_bit_match.rs` 等）が別途カバーするが、ここでも
/// CPU 参照実装との bit 完全一致（fail_count=0）を併せて確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn auto_entry_falls_back_to_classic_not_eligible_for_non_split_k_shapes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in NON_ELIGIBLE_SHAPES {
        assert!(
            tile::should_split_k(m, n, k).is_none(),
            "m={m}, n={n}, k={k}: NON_ELIGIBLE_SHAPES の前提（should_split_k が None を \
             返すこと）が崩れている"
        );

        let a_logical = Xorshift64Star::new(m as u64 * 13 + k as u64 + 3).fill_vec(m * k);
        let b_logical = Xorshift64Star::new(n as u64 * 17 + k as u64 + 5).fill_vec(k * n);
        let mut expected = vec![0.0f32; m * n];
        matmul_reference_fma(&a_logical, &b_logical, &mut expected, m, n, k)
            .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

        let a_layout = classify_2d(&[m, k], &[k as isize, 1]).unwrap();
        let b_layout = classify_2d(&[k, n], &[n as isize, 1]).unwrap();
        let a_buf = MetalBuffer::new_with_data(&ctx, &a_logical).expect("A バッファ確保に失敗した");
        let b_buf = MetalBuffer::new_with_data(&ctx, &b_logical).expect("B バッファ確保に失敗した");
        let c_buf = MetalBuffer::new_zeroed(&ctx, m * n).expect("C バッファ確保に失敗した");

        let route = gemm
            .dispatch_split_k_strided_prepared(
                &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf, m, n, k,
            )
            .unwrap_or_else(|e| {
                panic!("dispatch_split_k_strided_prepared failed (m={m}, n={n}, k={k}): {e}")
            });

        match route {
            SplitKRoute::Split { .. } => panic!(
                "m={m}, n={n}, k={k}: NON_ELIGIBLE 形状のはずが split-K 経路を実行した \
                 （should_split_k の判定と自動判定入口の分岐が食い違っている）"
            ),
            SplitKRoute::Classic { reason, .. } => {
                assert_eq!(
                    reason,
                    SplitKFallbackReason::NotEligible,
                    "m={m}, n={n}, k={k}: フォールバック理由が NotEligible ではない \
                     （reason={reason:?}）。NumericContractPendingApproval が返るなら \
                     SPLIT_K_NUMERIC_CONTRACT_APPROVED のゲート解除が反映されていない。"
                );
            }
        }

        let actual = c_buf.read_to_vec();
        let report = compare(&actual, &expected)
            .unwrap_or_else(|e| panic!("parity compare failed (m={m}, n={n}, k={k}): {e}"));
        assert_eq!(
            report.fail_count, 0,
            "m={m}, n={n}, k={k}: classic 経路が CPU 参照実装と bit 完全一致しない \
             （fail_count={}）",
            report.fail_count
        );
    }
}
