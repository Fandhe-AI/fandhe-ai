//! split-K 2 パス GEMM（イシュー #1474。#1516 で `dispatch_auto` へ
//! 定数ゲート付きで結線済み・既定 OFF〈有効化は #1515 の ADOPT 確定後〉）
//! の AC-2: split 経路（[`fandhe_ai_backend_metal::SplitKRoute::Split`]）
//! の出力が CPU 参照実装（`matmul_reference_fma`）と一致することを、
//! `docs/backend-metal-splitk-decision.md` §3 の対象 9 形状 × NN/NT/TN/TT
//! で確認する受け入れテスト。K 端数（`tile.bk` の非整除）を含む追加形状
//! `(64,64,2056)`／`(128,128,2064)` も併せて確認する（イシュー #1474
//! 計画 §7.2）。
//!
//! **判定方式（実測ベースライン非後退方式。イシュー #1512。承認記録は
//! `docs/backend-metal-splitk-parity-judgment-decision.md` §7・
//! 2026-09-10 ユーザー承認）**: 本テストは
//! `fandhe_ai_backend_cpu::parity::compare` の集計結果（`CompareReport`）を
//! `crates/backend-metal/tests/common/splitk_parity_baseline.rs::
//! assert_no_split_k_parity_regression`（CUDA 側 `ParityBaseline` と同型の
//! 4 指標連言〈`total` 一致・`fail_count`／`mean_abs_diff`／`max_abs_diff`／
//! `max_rel_err` の非後退〉）で判定する。対象 11 形状すべてへ一律適用する
//! （承認記録 §3・§7。CUDA 側のような「厳密ゼロ fail 成立形状は厳密判定・
//! 不成立形状のみ baseline」という形状二分方式は採らない）。
//!
//! 実機実測（M4 Max）では、split-K は K 方向を複数パーティションへ分割し
//! 独立に部分和を求めてから固定順序で結合するため、単一の連続 K ループで
//! 求める classic 経路とは加算の結合順序が異なり、丸め誤差の生じ方も異なる
//! ことを確認している。classic 経路は全対象形状で CPU 参照実装と bit
//! 完全一致する一方、split-K 経路は対象 11 形状のうち大半で REQ-2 統一
//! 複合判定の要素単位 fail が発生する（詳細は `docs/perf/metal-gemm-
//! splitk-two-pass.md` §5 を参照）。この特性は分割そのものに起因する構造的
//! 特性であり、縮約アルゴリズムの改善だけでは解消できないことを確認済み
//! （同 §5.2）。
//!
//! **経緯（差し戻し→再承認）**: 当初 baseline 方式を実装したが、PR #1496 の
//! codex-review 指摘（イシュー #1474）により、当時参照していた spec REQ-2
//! 2026-09-02 追記は TF32/f16 Tensor Core 経路限定であり Metal f32 split-K
//! への適用拡張には別途ユーザー承認が必要と判明したため、いったん
//! `assert_parity`（厳密ゼロ fail 判定）へ差し戻した
//! （`docs/perf/metal-gemm-splitk-two-pass.md` §5.5）。その後イシュー #1511
//! で適用拡張・baseline 値がユーザー承認され（`docs/backend-metal-splitk-
//! parity-judgment-decision.md` §7）、本イシュー（#1512）で baseline 方式へ
//! 再切替した。tolerance 定数（`RELATIVE_TOLERANCE`/
//! `ABSOLUTE_RESCUE_THRESHOLD`）自体は一貫して変更していない。
//!
//! 本テストは `crate::tile::should_split_k` の算出した計画を
//! `dispatch_split_k_strided_prepared_with_plan`（`tests/
//! gemm_splitk_bit_match.rs`〈AC-1〉と同じ明示計画版。`gemm.rs::
//! SPLIT_K_NUMERIC_CONTRACT_APPROVED` ゲートの対象外）へ直接渡す。
//! 自動判定入口 `dispatch_split_k_strided_prepared`（イシュー #1513 で
//! `SPLIT_K_NUMERIC_CONTRACT_APPROVED` を `true` へ切替済み・ゲート解除
//! 完了。`gemm.rs` 該当ドキュメンテーションコメント参照）は本テストの
//! 対象外で、split-K 経路自体の正しさ検証には明示計画版を使う。公開
//! 入口自体の到達確認は `tests/gemm_splitk_auto_entry_parity.rs`
//! （#1513 で新設）が担う。
//!
//! いずれのケースも戻り値が `SplitKRoute::Split` であることを assert し、
//! フォールバック（classic 経路）による自明合格を排除する。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。`#[ignore]` に
//! より通常の `cargo test` からは除外される
//! （`tests/gemm_strided_parity.rs` と同じ方針）。
//!
//! 実機実行（Apple Silicon 必須）:
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --features internal-diagnostics --test gemm_splitk_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

mod common;

use bench_harness::rng::Xorshift64Star;
use common::splitk_parity_baseline::{assert_no_split_k_parity_regression, find_baseline};
use fandhe_ai_backend_cpu::parity::{compare, matmul_reference_fma};
use fandhe_ai_backend_metal::layout::{MatrixLayout, classify_2d};
use fandhe_ai_backend_metal::tile;
use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm, SplitKRoute};

/// `logical`（行優先の論理 `[rows, cols]`）から `[cols, rows]` 行優先の
/// 転置済み物理バッファを作る（`tests/gemm_strided_parity.rs::
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

/// AC-2 の対象形状: `docs/backend-metal-splitk-decision.md` §3 の対象 9
/// 形状に加え、K 端数（`tile.bk`=16 の非整除）を含む境界ケース 2 点
/// （イシュー #1474 計画 §7.2）。`common::splitk_parity_baseline::BASELINES`
/// が同じ 11 形状を 1 対 1 でカバーする。
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

/// NN/NT/TN/TT の 4 パターンで `dispatch_split_k_strided_prepared_with_plan`
/// を直接呼び、CPU 参照実装との非後退契約（`assert_no_split_k_parity_
/// regression`。REQ-2 統一複合判定の集計値 `CompareReport` を記録済み
/// ベースラインと比較する）を検証する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn split_k_matches_classic_and_cpu_reference_for_target_shapes_and_transpose_patterns() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in TARGET_SHAPES {
        // `(m, n, k)` に対応する記録済みベースライン行。未登録形状での
        // 呼び出しは fail-open で素通りさせず panic する（`find_baseline`
        // の doc コメント参照。TARGET_SHAPES と BASELINES は 1 対 1 で
        // 対応する前提が崩れていないことをここで機械的に保証する）。
        let baseline = find_baseline(m, n, k).unwrap_or_else(|| {
            panic!(
                "m={m}, n={n}, k={k} に対応する記録済みベースラインが見つからない \
                 （TARGET_SHAPES と common::splitk_parity_baseline::BASELINES の対応が \
                 崩れている。baseline 行の追加は実機実測とセットでユーザー承認が必要）"
            )
        });

        // `should_split_k`（自動判定）が算出する計画をそのまま
        // `_with_plan`（`SPLIT_K_NUMERIC_CONTRACT_APPROVED` ゲート対象外）
        // へ明示的に渡す。対象形状はいずれも `should_split_k` が `Some`
        // を返す前提（`docs/backend-metal-splitk-decision.md` §3）。
        let plan = tile::should_split_k(m, n, k).unwrap_or_else(|| {
            panic!(
                "should_split_k が None を返した（対象形状の前提が崩れている）: m={m}, n={n}, k={k}"
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

            let route = gemm
                .dispatch_split_k_strided_prepared_with_plan(
                    &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf, m, n, k, plan,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "dispatch_split_k_strided_prepared_with_plan failed (trans_a={trans_a}, \
                         trans_b={trans_b}, m={m}, n={n}, k={k}): {e}"
                    )
                });
            assert!(
                matches!(route, SplitKRoute::Split { .. }),
                "trans_a={trans_a}, trans_b={trans_b}, m={m}, n={n}, k={k}: classic 経路へ\
                 フォールバックした（route={route:?}）。フォールバックによる自明合格を\
                 排除するため split-K 経路への到達を要求する（対象 9 形状は\
                 `should_split_k` が Some を返す前提）。"
            );

            let actual = c_buf.read_to_vec();
            let report = compare(&actual, &expected).unwrap_or_else(|e| {
                panic!(
                    "parity compare failed (trans_a={trans_a}, trans_b={trans_b}, m={m}, n={n}, \
                     k={k}): {e}"
                )
            });

            // #1474 実測時の survey ログ（`docs/perf/logs/metal-gemm-
            // splitk-two-pass-1474/parity_survey_all_shapes.log`）と同一
            // 形式で 1 行出力する。Mac 実機ログ（`docs/perf/logs/
            // metal-gemm-splitk-parity-baseline-1512/`）で承認済み
            // ベースラインとの一致を目視確認できるようにするため
            // （`--nocapture` 併用が前提）。
            println!(
                "m={m} n={n} k={k} trans_a={trans_a} trans_b={trans_b} \
                 fail_count={}/{} max_abs_diff={:.10} mean_abs_diff={:.10} max_rel_err={:.8}",
                report.fail_count,
                report.total,
                report.max_abs_diff,
                report.mean_abs_diff,
                report.max_rel_err
            );

            let context = format!(
                "split-K parity vs CPU reference (trans_a={trans_a}, trans_b={trans_b}, m={m}, \
                 n={n}, k={k})"
            );
            assert_no_split_k_parity_regression(&context, &report, baseline);
        }
    }
}
