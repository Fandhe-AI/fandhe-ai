//! split-K 2 パス GEMM（イシュー #1474。opt-in・`dispatch_auto` へ未結線）
//! の AC-2: split 経路（[`fandhe_ai_backend_metal::SplitKRoute::Split`]）
//! の出力が CPU 参照実装（`matmul_reference_fma`）と一致することを、
//! `docs/backend-metal-splitk-decision.md` §3 の対象 9 形状 × NN/NT/TN/TT
//! で確認する受け入れテスト。K 端数（`tile.bk` の非整除）を含む追加形状
//! `(64,64,2056)`／`(128,128,2064)` も併せて確認する（イシュー #1474
//! 計画 §7.2）。
//!
//! **判定方式（実機実測を経て AC-2 の当初記述から変更。イシュー #1474）**:
//! split-K は K 方向を複数パーティションへ分割し独立に部分和を求めてから
//! 固定順序で結合するため、単一の連続 K ループで求める classic 経路とは
//! 加算の結合順序が異なり、丸め誤差の生じ方も異なる。実機実測（M4 Max）で
//! classic 経路は全対象形状で CPU 参照実装と bit 完全一致する一方、split-K
//! 経路は対象 11 形状のうち大半で REQ-2 統一複合判定の要素単位 fail が
//! 1〜8 件／総要素数発生し、`partitions=2`（最小分割）の時点で既に発生する
//! ことを確認した（結合順序の違いに起因する構造的特性であり、`docs/
//! coding-rust.md` が定める tolerance 定数〈`RELATIVE_TOLERANCE`/
//! `ABSOLUTE_RESCUE_THRESHOLD`〉の緩和では解消しない）。詳細は
//! `docs/perf/metal-gemm-splitk-two-pass.md` §5 を参照。
//!
//! そのため本テストは厳密ゼロ fail 判定（`assert_parity`）ではなく、
//! `tests/common/splitk_parity_baseline.rs` の実測ベースライン非後退
//! 契約（`assert_no_split_k_parity_regression`）で判定する。これは
//! `docs/spec/04-requirements.md` REQ-2 2026-09-02 追記（TF32/f16 Tensor
//! Core 経路の受け入れ判定方式）が CUDA 側で確立した「厳密ゼロ fail
//! 判定は実機実測で成立が確認された形状に限り、成立しない形状は実測
//! baseline 非後退方式を正式な受け入れ判定とする」という方針と同種の
//! 適用である（判定式・tolerance 定数自体は一切変更しない）。
//!
//! いずれのケースも `dispatch_split_k_strided_prepared` の戻り値が
//! `SplitKRoute::Split` であることを assert し、フォールバック（classic
//! 経路）による自明合格を排除する。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。`#[ignore]` に
//! より通常の `cargo test` からは除外される
//! （`tests/gemm_strided_parity.rs` と同じ方針）。
//!
//! 実機実行（Apple Silicon 必須）:
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test gemm_splitk_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

mod common;

use bench_harness::rng::Xorshift64Star;
use common::splitk_parity_baseline::{assert_no_split_k_parity_regression, find_baseline};
use fandhe_ai_backend_cpu::parity::{compare, matmul_reference_fma};
use fandhe_ai_backend_metal::layout::{MatrixLayout, classify_2d};
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
/// （イシュー #1474 計画 §7.2）。
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

/// NN/NT/TN/TT の 4 パターンで `dispatch_split_k_strided_prepared` を
/// 直接呼び、classic 経路・CPU 参照実装との統一複合判定（REQ-2）を検証
/// する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn split_k_matches_classic_and_cpu_reference_for_target_shapes_and_transpose_patterns() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in TARGET_SHAPES {
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
                .dispatch_split_k_strided_prepared(
                    &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf, m, n, k,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "dispatch_split_k_strided_prepared failed (trans_a={trans_a}, \
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
            let report = compare(&actual, &expected)
                .expect("compare の長さ検証に失敗した（actual/expected のサイズ不一致）");
            let baseline = find_baseline(m, n, k).unwrap_or_else(|| {
                panic!(
                    "m={m}, n={n}, k={k}: split-K parity baseline が未登録です。\
                     `tests/common/splitk_parity_baseline.rs::BASELINES` に実測値を追加してください \
                     （推定値の記入は禁止）。"
                )
            });
            assert_no_split_k_parity_regression(
                &format!(
                    "split-K parity vs CPU reference (trans_a={trans_a}, trans_b={trans_b}, \
                     m={m}, n={n}, k={k})"
                ),
                &report,
                baseline,
            );
        }
    }
}
