//! `MetalGemm::dispatch_strided_bias_act_prepared`（転置パターン別・stride
//! 付き GEMM 入口。イシュー #1040）の実機テスト。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test gemm_strided_parity -- --ignored --nocapture
//! ```
//!
//! 検証項目:
//! - NN/NT/TN/TT の 4 転置パターンで CPU 参照実装（`fandhe_ai_backend_cpu::
//!   parity::matmul_reference_fma`）との統一複合判定（REQ-2）
//! - NN 構成（`dispatch_bias_act_prepared` 経由）と本テストが直接呼ぶ
//!   `dispatch_strided_bias_act_prepared`（`GemmStrides::nn` 相当）が
//!   ビット完全一致すること（後方互換委譲の非後退確認）
//! - `[B, M, K] @ [K, N]` の先頭次元 collapse（`MetalBackendOps::
//!   gemm_collapsed_lhs`）が CPU 参照実装（バッチ次元ごとに独立 GEMM）と
//!   一致すること
//!
//! `MetalBackendOps::gemm_resident_lhs`／`gemm_resident_rhs` に転置 view を
//! 渡した場合の「ホスト側転置コピー 0 回」確認は、`ops::
//! RESIDENT_HOST_REPACK_COUNT` が `pub(crate)`（クレート境界外の本ファイル
//! からは参照できない。`gemm::BIAS_ACT_FUSED_LAUNCH_COUNT` と同じ可視性
//! 方針。`gemm_bias_act_parity.rs` コメント参照）のため、
//! `crates/backend-metal/src/ops.rs` の `#[cfg(test)]` 内クレート内テスト
//! （`gemm_resident_lhs_transposed_b_does_not_increment_repack_counter`）に
//! 委ねる。本ファイルは数値一致のみを検証する。

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::{assert_parity, matmul_reference_fma};
use fandhe_ai_backend_metal::layout::{MatrixLayout, classify_2d};
use fandhe_ai_backend_metal::tile;
use fandhe_ai_backend_metal::{MetalBackendOps, MetalBuffer, MetalContext, MetalGemm};
use fandhe_ai_tensor_core::Tensor;

/// `logical`（行優先の論理 `[rows, cols]`）から `[cols, rows]` 行優先の
/// 転置済み物理バッファを作る（`crate::layout::classify_2d` が
/// `transposed: true` と分類する view の実データを CPU 側で明示的に
/// 構築するため）。
fn transpose_dense(logical: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = logical[r * cols + c];
        }
    }
    out
}

/// NN/NT/TN/TT の 4 パターンで `dispatch_strided_bias_act_prepared` を
/// 直接呼び、CPU 参照実装との統一複合判定（REQ-2）を検証する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_strided_bias_act_prepared_matches_cpu_reference_for_all_transpose_patterns() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in &[(1usize, 1usize, 1usize), (4, 5, 3), (37, 29, 41)] {
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

            gemm.dispatch_strided_bias_act_prepared(
                &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, None, false, &c_buf, m, n, k,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "dispatch_strided_bias_act_prepared failed (trans_a={trans_a}, \
                     trans_b={trans_b}, m={m}, n={n}, k={k}): {e}"
                )
            });
            let actual = c_buf.read_to_vec();

            assert_parity(
                &format!(
                    "NN/NT/TN/TT parity (trans_a={trans_a}, trans_b={trans_b}, m={m}, n={n}, k={k})"
                ),
                &actual,
                &expected,
            );
        }
    }
}

/// NN 構成での `dispatch_strided_bias_act_prepared` 直接呼び出しが、
/// 既存の `dispatch_bias_act_prepared`（`GemmStrides::nn` へ委譲する
/// 後方互換入口）とビット完全一致することを確認する（イシュー #1040
/// 実装計画§2.2「NN 指定時は既存と完全同一の添字」の非後退確認）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_bias_act_prepared_nn_is_bit_identical_to_strided_nn() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");
    let (m, n, k) = (13usize, 17usize, 19usize);
    let a = Xorshift64Star::new(101).fill_vec(m * k);
    let b = Xorshift64Star::new(202).fill_vec(k * n);

    let a_buf = MetalBuffer::new_with_data(&ctx, &a).unwrap();
    let b_buf = MetalBuffer::new_with_data(&ctx, &b).unwrap();
    let c_buf_legacy = MetalBuffer::new_zeroed(&ctx, m * n).unwrap();
    gemm.dispatch_bias_act_prepared(
        &ctx,
        &a_buf,
        0,
        &b_buf,
        0,
        None,
        false,
        &c_buf_legacy,
        m,
        n,
        k,
    )
    .unwrap();
    let legacy = c_buf_legacy.read_to_vec();

    let a_layout = classify_2d(&[m, k], &[k as isize, 1]).unwrap();
    let b_layout = classify_2d(&[k, n], &[n as isize, 1]).unwrap();
    let c_buf_strided = MetalBuffer::new_zeroed(&ctx, m * n).unwrap();
    gemm.dispatch_strided_bias_act_prepared(
        &ctx,
        &a_buf,
        0,
        a_layout,
        &b_buf,
        0,
        b_layout,
        None,
        false,
        &c_buf_strided,
        m,
        n,
        k,
    )
    .unwrap();
    let strided = c_buf_strided.read_to_vec();

    assert_eq!(
        legacy, strided,
        "NN 構成では dispatch_bias_act_prepared と dispatch_strided_bias_act_prepared \
         はビット完全一致する契約（後方互換委譲の非後退確認）"
    );
}

/// `MetalBackendOps::gemm_collapsed_lhs` が `[B, M, K] @ [K, N]` を
/// バッチ次元ごとの独立 GEMM（CPU 参照実装）と一致させて計算することを
/// 検証する（イシュー #1040「先頭次元 collapse」）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_collapsed_lhs_matches_per_batch_cpu_reference() {
    let ops = MetalBackendOps::new();
    let (batch, m, k, n) = (3usize, 4usize, 5usize, 6usize);
    let a_data = Xorshift64Star::new(303).fill_vec(batch * m * k);
    let b_data = Xorshift64Star::new(404).fill_vec(k * n);

    let a = Tensor::new(a_data.clone(), &[batch, m, k]).unwrap();
    let b_tensor = Tensor::new(b_data.clone(), &[k, n]).unwrap();

    let actual = ops
        .gemm_collapsed_lhs(&a, &b_tensor)
        .expect("gemm_collapsed_lhs must succeed on Metal-equipped test runner");
    assert_eq!(actual.shape(), &[batch, m, n]);

    let actual_contiguous = actual.contiguous();
    let actual_slice = actual_contiguous.as_slice().unwrap();
    for b in 0..batch {
        let a_batch = &a_data[b * m * k..(b + 1) * m * k];
        let mut expected_batch = vec![0.0f32; m * n];
        matmul_reference_fma(a_batch, &b_data, &mut expected_batch, m, n, k)
            .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");
        let actual_batch = &actual_slice[b * m * n..(b + 1) * m * n];
        assert_parity(
            &format!("gemm_collapsed_lhs batch {b}"),
            actual_batch,
            &expected_batch,
        );
    }
}

/// イシュー #1138: `dispatch_strided_tiled_prepared`（NN・8 整除形状）が
/// `dispatch_tiled_prepared`（既存 `gemm_simdgroup_tiled` 高速経路）と
/// ビット完全一致することを確認する（`TRANS_A`/`TRANS_B` 追加による
/// 既存 NN 経路の非後退確認。`docs/backend-metal-transpose-collapse-design.md`
/// §2）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_strided_tiled_prepared_nn_is_bit_identical_to_dispatch_tiled_prepared() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");
    let (m, n, k) = (64usize, 96usize, 48usize);
    let cfg = tile::select(m, n, k);

    let a = Xorshift64Star::new(11).fill_vec(m * k);
    let b = Xorshift64Star::new(22).fill_vec(k * n);
    let a_buf = MetalBuffer::new_with_data(&ctx, &a).unwrap();
    let b_buf = MetalBuffer::new_with_data(&ctx, &b).unwrap();

    let c_buf_legacy = MetalBuffer::new_zeroed(&ctx, m * n).unwrap();
    gemm.dispatch_tiled_prepared(&ctx, &a_buf, &b_buf, &c_buf_legacy, m, n, k, cfg)
        .unwrap();
    let legacy = c_buf_legacy.read_to_vec();

    let a_layout = classify_2d(&[m, k], &[k as isize, 1]).unwrap();
    let b_layout = classify_2d(&[k, n], &[n as isize, 1]).unwrap();
    let c_buf_strided = MetalBuffer::new_zeroed(&ctx, m * n).unwrap();
    gemm.dispatch_strided_tiled_prepared(
        &ctx,
        &a_buf,
        0,
        a_layout,
        &b_buf,
        0,
        b_layout,
        &c_buf_strided,
        m,
        n,
        k,
        cfg,
    )
    .unwrap();
    let strided = c_buf_strided.read_to_vec();

    assert_eq!(
        legacy, strided,
        "NN 構成では dispatch_tiled_prepared と dispatch_strided_tiled_prepared \
         はビット完全一致する契約（TRANS_A/TRANS_B 導入の非後退確認。イシュー #1138）"
    );
}

/// イシュー #1138: NN/NT/TN/TT の 4 転置パターンで
/// `dispatch_strided_tiled_prepared` が CPU 参照実装と統一複合判定
/// （REQ-2）で一致することを確認する。形状はいずれも 8 整除（適格性
/// ゲートを通過する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_strided_tiled_prepared_matches_cpu_reference_for_all_transpose_patterns() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in &[(64usize, 64usize, 64usize), (72, 88, 104), (256, 256, 256)] {
        let cfg = tile::select(m, n, k);
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

            gemm.dispatch_strided_tiled_prepared(
                &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf, m, n, k, cfg,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "dispatch_strided_tiled_prepared failed (trans_a={trans_a}, \
                     trans_b={trans_b}, m={m}, n={n}, k={k}): {e}"
                )
            });
            let actual = c_buf.read_to_vec();

            assert_parity(
                &format!(
                    "dispatch_strided_tiled_prepared NN/NT/TN/TT parity \
                     (trans_a={trans_a}, trans_b={trans_b}, m={m}, n={n}, k={k})"
                ),
                &actual,
                &expected,
            );
        }
    }
}

/// イシュー #1138: 適格性ゲート（`strided_tiled_eligibility`）が拒否する
/// 入力（非 8 整除次元）では `dispatch_strided_tiled_prepared` が
/// `Err(StridedTiledIneligible)` を返す一方、classic strided 経路
/// （`dispatch_strided_bias_act_prepared`）は同じ入力を問題なく処理できる
/// ことを確認する（fail-closed フォールバックの健全性）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_strided_tiled_prepared_rejects_non_eight_divisible_shape_while_classic_succeeds() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");
    let (m, n, k) = (37usize, 29usize, 41usize); // いずれも非 8 整除

    let a = Xorshift64Star::new(55).fill_vec(m * k);
    let b = Xorshift64Star::new(66).fill_vec(k * n);
    let a_buf = MetalBuffer::new_with_data(&ctx, &a).unwrap();
    let b_buf = MetalBuffer::new_with_data(&ctx, &b).unwrap();
    let a_layout = classify_2d(&[m, k], &[k as isize, 1]).unwrap();
    let b_layout = classify_2d(&[k, n], &[n as isize, 1]).unwrap();

    let c_buf_tiled = MetalBuffer::new_zeroed(&ctx, m * n).unwrap();
    let cfg = tile::select(m, n, k);
    let err = gemm
        .dispatch_strided_tiled_prepared(
            &ctx,
            &a_buf,
            0,
            a_layout,
            &b_buf,
            0,
            b_layout,
            &c_buf_tiled,
            m,
            n,
            k,
            cfg,
        )
        .expect_err("非 8 整除形状は StridedTiledIneligible で拒否される契約");
    assert!(matches!(
        err,
        fandhe_ai_backend_metal::MetalError::StridedTiledIneligible { .. }
    ));

    let c_buf_classic = MetalBuffer::new_zeroed(&ctx, m * n).unwrap();
    gemm.dispatch_strided_bias_act_prepared(
        &ctx,
        &a_buf,
        0,
        a_layout,
        &b_buf,
        0,
        b_layout,
        None,
        false,
        &c_buf_classic,
        m,
        n,
        k,
    )
    .expect("classic strided 経路（gemm_tiled_bias_act）は非 8 整除形状も処理できる契約");
}
