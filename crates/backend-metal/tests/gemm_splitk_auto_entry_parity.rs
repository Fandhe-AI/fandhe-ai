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
//! - (negative・事前条件は満たす) `should_split_k` が `None` を返し、
//!   かつ `strided_tiled_eligibility`（事前条件ゲート。イシュー #1899）
//!   は満たす形状では、公開入口が
//!   `SplitKRoute::Classic { reason: SplitKFallbackReason::NotEligible }`
//!   を返し、**`NumericContractPendingApproval` では決してない**こと
//!   （ゲート解除自体の直接検証。`NumericContractPendingApproval` は
//!   後方互換のため enum には残っているが本テスト時点ではもう返らない）
//! - (negative・事前条件違反) `strided_tiled_eligibility` の事前条件
//!   （m/n/k が 8 の倍数等）**に違反する**形状（例: `(64, 64, 63)`）では、
//!   `should_split_k` の判定結果に関わらず公開入口が型付き
//!   `Err(`[`fandhe_ai_backend_metal::MetalError::StridedTiledIneligible`]`)`
//!   を返し `SplitKRoute::Classic` へは分類しないこと（契約 (A)。イシュー
//!   #1899・2026-09-16 ユーザー承認。`docs/backend-metal-splitk-decision.md`
//!   §5「自動判定入口の事前条件契約（#1899）」参照）
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
    MetalBuffer, MetalContext, MetalError, MetalGemm, SplitKFallbackReason, SplitKRoute,
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

/// classic 経路（非 split-K）の CPU 参照実装に対する bit 完全一致を
/// `to_bits()` 経由で検証する（`tests/gemm_splitk_bit_match.rs::
/// assert_bit_exact` と同じ理由: `compare()` は REQ-2 統一複合判定
/// 「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」を用いるため
/// `fail_count == 0` は許容誤差内での一致を意味するに過ぎず、bit 完全
/// 一致を意味しない。classic 経路は CPU 参照実装〈`f32::mul_add`〉と
/// 同一の FMA 契約〈`.claude/rules/coding-rust.md`〉で丸めるため bit
/// 完全一致を期待できる回帰検出用の厳密判定として用いる）。
fn assert_bit_exact_vs_reference(actual: &[f32], expected: &[f32], context: &str) {
    let actual_bits: Vec<u32> = actual.iter().map(|v| v.to_bits()).collect();
    let expected_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        actual_bits, expected_bits,
        "{context}: classic 経路の出力が CPU 参照実装と bit 単位で一致しなかった\
         （fail_count ベースの許容誤差判定では検出できない回帰の疑いがある）。"
    );
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

/// `should_split_k` が `None` を返し、かつ
/// `strided_tiled_eligibility`（`backend-metal` 内部の非公開関数。
/// `dispatch_split_k_strided_prepared` の事前条件）**も満たす**形状。
/// `crates/backend-metal/src/tile.rs` の
/// `should_split_k_rejects_large_square_and_wide_shapes` が Linux で
/// 確認済みの事実（正方 512 以上は並列度条件で除外）を転用する。
/// この定数は `SplitKRoute::Classic { reason: NotEligible }` を返す
/// 既存 negative テストの対象であり続ける（イシュー #1899 で `(64, 64,
/// 63)` を下記 [`PRECONDITION_VIOLATING_SHAPES`] へ分離する前の挙動を
/// そのまま引き継ぐ）。
const NON_ELIGIBLE_PRECONDITION_OK_SHAPES: &[(usize, usize, usize)] = &[(512, 512, 512)];

/// `should_split_k` が `None` を返す**が**、
/// `strided_tiled_eligibility` の事前条件（m/n/k が 8 の倍数）には
/// **違反する**形状（イシュー #1899・#1513 (b)）。
///
/// 2026-09-16 の M4 Max 実測（#1904。
/// `docs/perf/logs/metal-gemm-splitk-auto-entry-1513/auto_entry.log`）で、
/// `(64, 64, 63)`（k=63 が 8 の倍数でない）が
/// `Ok(SplitKRoute::Classic { reason: NotEligible })` を期待する旧
/// `NON_ELIGIBLE_SHAPES` fixture に含まれていたため FAIL
/// （実際には `Err(MetalError::StridedTiledIneligible)` を返す）した。
/// 契約 (A)（イシュー #1899 コメント・2026-09-16 ユーザー承認）は
/// 「事前条件違反は `should_split_k` の判定結果に関わらず型付き `Err`
/// のまま維持し、`SplitKRoute::Classic` へは分類しない」と確定した
/// ため、この fixture は
/// `auto_entry_rejects_precondition_violating_shapes_with_typed_err`
/// （別テスト）の対象とし、`SplitKRoute::Classic { NotEligible }` を
/// 期待する対象からは外す（`crates/backend-metal/src/tile.rs::
/// should_split_k_rejects_k_below_max_m_n` が `should_split_k` 自体は
/// `None` を返すことを Linux で別途確認済み。ここでは
/// `strided_tiled_eligibility` 側の事前条件違反を検証する）。
const PRECONDITION_VIOLATING_SHAPES: &[(usize, usize, usize)] = &[(64, 64, 63)];

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

/// (negative) `should_split_k` が `None` を返し、かつ
/// `strided_tiled_eligibility` の事前条件は満たす形状（
/// [`NON_ELIGIBLE_PRECONDITION_OK_SHAPES`]）では、公開入口が classic
/// 経路（`SplitKFallbackReason::NotEligible`）へフォールバックし、
/// **`NumericContractPendingApproval` では決してない**こと（ゲート解除
/// 自体の直接検証）。classic 経路の出力は既存 bit 一致群
/// （`tests/gemm_splitk_bit_match.rs` 等）が別途カバーするが、ここでも
/// CPU 参照実装との bit 完全一致を `to_bits()` 経由（許容誤差を用いる
/// `compare()`/`fail_count` ではなく `assert_bit_exact_vs_reference`）で
/// 併せて確認する。
///
/// `strided_tiled_eligibility` の事前条件**に違反する**形状
/// （[`PRECONDITION_VIOLATING_SHAPES`]。例: `(64, 64, 63)`）は本テストの
/// 対象ではない（`Ok(Classic)` ではなく型付き `Err` を返す契約 (A)。
/// イシュー #1899）。その契約は
/// `auto_entry_rejects_precondition_violating_shapes_with_typed_err`
/// （別テスト）が検証する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn auto_entry_falls_back_to_classic_not_eligible_for_non_split_k_shapes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in NON_ELIGIBLE_PRECONDITION_OK_SHAPES {
        assert!(
            tile::should_split_k(m, n, k).is_none(),
            "m={m}, n={n}, k={k}: NON_ELIGIBLE_PRECONDITION_OK_SHAPES の前提\
             （should_split_k が None を返すこと）が崩れている"
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
        let context = format!("auto entry classic fallback vs CPU reference (m={m}, n={n}, k={k})");
        assert_bit_exact_vs_reference(&actual, &expected, &context);
    }
}

/// (negative) `strided_tiled_eligibility` の事前条件（m/n/k が 8 の倍数
/// 等）に**違反する**形状（[`PRECONDITION_VIOLATING_SHAPES`]。例:
/// `(64, 64, 63)`）では、`should_split_k` の判定結果に関わらず公開入口
/// が型付き `Err(MetalError::StridedTiledIneligible)` を返し、
/// [`SplitKRoute::Classic`] へは分類しないこと（契約 (A)。イシュー
/// #1899・2026-09-16 ユーザー承認）。
///
/// 2026-09-16 の M4 Max 実測（#1904。
/// `docs/perf/logs/metal-gemm-splitk-auto-entry-1513/auto_entry.log`）で、
/// `(64, 64, 63)` を旧 `NON_ELIGIBLE_SHAPES`（`Ok(Classic{NotEligible})`
/// を期待）に含めていたため FAIL していた事象の是正: `(64, 64, 63)` を
/// 本テスト専用の [`PRECONDITION_VIOLATING_SHAPES`] へ切り出し、
/// `Err` を返すことそのものを期待値とする。
///
/// また、事前条件検査は encode より前に完結する
/// （`dispatch_split_k_strided_prepared` 内の両分岐が
/// `strided_tiled_eligibility(...)?` を encode 呼び出しより先に評価する
/// ため）契約を、`c_buf`（`MetalBuffer::new_zeroed` で確保したゼロ埋め
/// バッファ）が `Err` 後も全要素ゼロのまま変化しないことで直接確認する
/// （`Err` を返した呼び出しが出力バッファへ一切書き込まないという
/// fail-closed 契約の裏付け。`.claude/rules/security.md` A03）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn auto_entry_rejects_precondition_violating_shapes_with_typed_err() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in PRECONDITION_VIOLATING_SHAPES {
        assert!(
            tile::should_split_k(m, n, k).is_none(),
            "m={m}, n={n}, k={k}: PRECONDITION_VIOLATING_SHAPES の前提\
             （should_split_k が None を返すこと）が崩れている"
        );
        assert!(
            !m.is_multiple_of(8) || !n.is_multiple_of(8) || !k.is_multiple_of(8),
            "m={m}, n={n}, k={k}: PRECONDITION_VIOLATING_SHAPES の前提\
             （m/n/k のいずれかが 8 の倍数でないこと。事前条件違反 fixture である\
             こと）が崩れている"
        );

        let a_logical = Xorshift64Star::new(m as u64 * 19 + k as u64 + 7).fill_vec(m * k);
        let b_logical = Xorshift64Star::new(n as u64 * 23 + k as u64 + 11).fill_vec(k * n);

        let a_layout = classify_2d(&[m, k], &[k as isize, 1]).unwrap();
        let b_layout = classify_2d(&[k, n], &[n as isize, 1]).unwrap();
        let a_buf = MetalBuffer::new_with_data(&ctx, &a_logical).expect("A バッファ確保に失敗した");
        let b_buf = MetalBuffer::new_with_data(&ctx, &b_logical).expect("B バッファ確保に失敗した");
        let c_buf = MetalBuffer::new_zeroed(&ctx, m * n).expect("C バッファ確保に失敗した");

        let result = gemm.dispatch_split_k_strided_prepared(
            &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf, m, n, k,
        );

        println!("[auto-entry] precondition-violation m={m} n={n} k={k} result={result:?}");

        match result {
            Ok(SplitKRoute::Split { .. }) => panic!(
                "m={m}, n={n}, k={k}: 事前条件違反形状のはずが split-K 経路を実行した \
                 （契約 (A) 違反。イシュー #1899）"
            ),
            Ok(SplitKRoute::Classic { reason, .. }) => panic!(
                "m={m}, n={n}, k={k}: 事前条件違反形状のはずが classic 経路 \
                 （reason={reason:?}）へ分類された（契約 (A) 違反。事前条件違反は \
                 SplitKRoute::Classic ではなく型付き Err を返すべき。イシュー #1899）"
            ),
            Err(err) => {
                assert!(
                    matches!(err, MetalError::StridedTiledIneligible { .. }),
                    "m={m}, n={n}, k={k}: 事前条件違反時のエラーが \
                     StridedTiledIneligible ではない（err={err:?}）"
                );
            }
        }

        let c_after = c_buf.read_to_vec();
        assert!(
            c_after.iter().all(|v| v.to_bits() == 0u32),
            "m={m}, n={n}, k={k}: Err を返したにも関わらず c_buf が書き換えられた \
             （事前条件検査が encode より前に完結し出力バッファへ一切副作用を \
             及ぼさないという契約〈イシュー #1899〉が崩れている）"
        );
    }
}
