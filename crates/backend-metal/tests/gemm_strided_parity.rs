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
use fandhe_ai_backend_cpu::parity::{assert_parity, compare, matmul_reference_fma};
use fandhe_ai_backend_metal::layout::{MatrixLayout, classify_2d};
use fandhe_ai_backend_metal::tile;
use fandhe_ai_backend_metal::{MetalBackendOps, MetalBuffer, MetalContext, MetalGemm, TileConfig};
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

/// `logical`（行優先の論理 `[rows, cols]`）を、行あたり `ld`（`>= cols`）
/// 要素のパディング付き行優先バッファへ埋め込む。パディング列（`cols..ld`）
/// には論理値と大きく異なる番兵値（`f32::MAX`）を書き込み、シェーダが
/// `st.lda`/`st.ldb`（padded leading dimension）ではなく非パディング値
/// （`dims.k`/`dims.n`）で読み出す誤りがあれば、パディング領域を読んで
/// 結果が大きく崩れ `assert_parity` が確実に検出できるようにする
/// （イシュー #1138 codex-review/Cursor Bugbot 指摘の回帰テスト）。
fn pad_dense_row_major(logical: &[f32], rows: usize, cols: usize, ld: usize) -> Vec<f32> {
    assert!(ld >= cols, "ld は cols 以上である契約");
    let mut out = vec![f32::MAX; rows * ld];
    for r in 0..rows {
        out[r * ld..r * ld + cols].copy_from_slice(&logical[r * cols..(r + 1) * cols]);
    }
    out
}

/// イシュー #1138（codex-review P1・Cursor Bugbot 指摘の回帰）:
/// `gemm_simdgroup_tiled` の非転置ロード分岐（staged・direct-load 両経路）
/// が `st.lda`/`st.ldb`（`MatrixLayout::ld`。padded leading dimension を
/// 許容する）を無視し `dims.k`/`dims.n`（非パディング論理次元）で device
/// メモリを読んでいたバグの回帰テスト。`strided_tiled_eligibility` は
/// `ld` が 4 の倍数であることしか要求せず `ld == k`/`ld == n`（密行列）を
/// 強制しないため、NN/NT/TN（非転置側を 1 つ以上含む全パターン）で
/// padded leading dimension を持つ入力を CPU 参照実装と統一複合判定
/// （REQ-2）で照合する。修正前は非転置側のパディング列（番兵値
/// `f32::MAX`）を誤って読み込み結果が大きく崩れるため本テストで検出
/// できる（TT は非転置側を持たないため対象外。修正内容は
/// `crates/backend-metal/src/shaders/gemm.metal` の非転置ロード分岐参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_strided_tiled_prepared_handles_padded_leading_dimension_for_non_transposed_side() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    let (m, n, k) = (64usize, 96usize, 48usize); // いずれも 8 整除（適格性ゲート通過）
    let cfg = tile::select(m, n, k);
    let a_logical = Xorshift64Star::new(1138).fill_vec(m * k);
    let b_logical = Xorshift64Star::new(2276).fill_vec(k * n);
    let mut expected = vec![0.0f32; m * n];
    matmul_reference_fma(&a_logical, &b_logical, &mut expected, m, n, k)
        .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

    // padded ld（4 の倍数を維持したまま論理次元より大きくする）。
    let lda_padded = k + 4;
    let ldb_padded = n + 4;

    // NN: A・B とも非転置・両方 padded ld。
    // NT: A 非転置（padded ld）・B 転置（padded ld は転置側に非適用）。
    // TN: A 転置・B 非転置（padded ld）。
    // TT は非転置側を持たないため本バグの対象外（既存 all_transpose_patterns
    // テストで別途 parity 確認済み）。
    for (trans_a, trans_b) in [(false, false), (false, true), (true, false)] {
        let (a_phys, a_layout): (Vec<f32>, MatrixLayout) = if trans_a {
            (
                transpose_dense(&a_logical, m, k),
                classify_2d(&[m, k], &[1, m as isize]).unwrap(),
            )
        } else {
            (
                pad_dense_row_major(&a_logical, m, k, lda_padded),
                classify_2d(&[m, k], &[lda_padded as isize, 1]).unwrap(),
            )
        };
        let (b_phys, b_layout): (Vec<f32>, MatrixLayout) = if trans_b {
            (
                transpose_dense(&b_logical, k, n),
                classify_2d(&[k, n], &[1, k as isize]).unwrap(),
            )
        } else {
            (
                pad_dense_row_major(&b_logical, k, n, ldb_padded),
                classify_2d(&[k, n], &[ldb_padded as isize, 1]).unwrap(),
            )
        };

        let a_buf = MetalBuffer::new_with_data(&ctx, &a_phys).expect("A バッファ確保に失敗した");
        let b_buf = MetalBuffer::new_with_data(&ctx, &b_phys).expect("B バッファ確保に失敗した");
        let c_buf = MetalBuffer::new_zeroed(&ctx, m * n).expect("C バッファ確保に失敗した");

        gemm.dispatch_strided_tiled_prepared(
            &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf, m, n, k, cfg,
        )
        .unwrap_or_else(|e| {
            panic!(
                "dispatch_strided_tiled_prepared failed (padded ld, trans_a={trans_a}, \
                 trans_b={trans_b}, m={m}, n={n}, k={k}): {e}"
            )
        });
        let actual = c_buf.read_to_vec();

        assert_parity(
            &format!(
                "dispatch_strided_tiled_prepared padded-ld parity \
                 (trans_a={trans_a}, trans_b={trans_b}, m={m}, n={n}, k={k}, \
                 lda={}, ldb={})",
                a_layout.ld, b_layout.ld
            ),
            &actual,
            &expected,
        );
    }
}
// --- イシュー #1329（E7・親 #1324）: 64x64x32（wm2wn2）を CANDIDATES[9] へ
//     追加。全形状 × 転置 4 種の parity 確認・CANDIDATES[0]（bk=16）との
//     複合判定内一致確認。性能比較・`tile::select` への組み込み判断は
//     後続イシュー #1330 のスコープ（本ファイルは正確性確認のみ）。

/// `CANDIDATES[9]`（イシュー #1329。64,64,32,2,2 — `CANDIDATES[0]` の
/// bk=32 版）を明示指定した `dispatch_strided_tiled_prepared` の 1 形状 ×
/// 1 転置パターンを実行し、CPU 参照実装との複合判定（REQ-2）で一致する
/// ことを確認する。`expected`（CPU 参照）は形状ごとに 1 回だけ計算し
/// 呼び出し元（4 パターン分）で共有する契約（`matmul_reference_fma` は
/// スカラー実装で計算コストが高いため）。`resolved == cfg` を assert し
/// サイレントフォールバックを検知する。
#[allow(clippy::too_many_arguments)]
fn assert_e7_candidate_matches_reference_for_pattern(
    ctx: &MetalContext,
    gemm: &MetalGemm,
    cfg: TileConfig,
    a_logical: &[f32],
    b_logical: &[f32],
    expected: &[f32],
    shape: (usize, usize, usize),
    transpose: (bool, bool),
) {
    let (m, n, k) = shape;
    let (trans_a, trans_b) = transpose;
    let (a_phys, a_layout): (Vec<f32>, MatrixLayout) = if trans_a {
        (
            transpose_dense(a_logical, m, k),
            classify_2d(&[m, k], &[1, m as isize]).unwrap(),
        )
    } else {
        (
            a_logical.to_vec(),
            classify_2d(&[m, k], &[k as isize, 1]).unwrap(),
        )
    };
    let (b_phys, b_layout): (Vec<f32>, MatrixLayout) = if trans_b {
        (
            transpose_dense(b_logical, k, n),
            classify_2d(&[k, n], &[1, k as isize]).unwrap(),
        )
    } else {
        (
            b_logical.to_vec(),
            classify_2d(&[k, n], &[n as isize, 1]).unwrap(),
        )
    };

    let a_buf = MetalBuffer::new_with_data(ctx, &a_phys).expect("A バッファ確保に失敗した");
    let b_buf = MetalBuffer::new_with_data(ctx, &b_phys).expect("B バッファ確保に失敗した");
    let c_buf = MetalBuffer::new_zeroed(ctx, m * n).expect("C バッファ確保に失敗した");

    let resolved = gemm
        .dispatch_strided_tiled_prepared(
            ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf, m, n, k, cfg,
        )
        .unwrap_or_else(|e| {
            panic!(
                "dispatch_strided_tiled_prepared failed (cfg={cfg:?}, trans_a={trans_a}, \
                 trans_b={trans_b}, m={m}, n={n}, k={k}): {e}"
            )
        });
    assert_eq!(
        resolved, cfg,
        "CANDIDATES[9] がサイレントフォールバックした（trans_a={trans_a}, trans_b={trans_b}, \
         m={m}, n={n}, k={k}）"
    );

    let actual = c_buf.read_to_vec();
    assert_parity(
        &format!(
            "CANDIDATES[9]（64,64,32,2,2。イシュー #1329）dispatch_strided_tiled_prepared \
             parity (trans_a={trans_a}, trans_b={trans_b}, m={m}, n={n}, k={k})"
        ),
        &actual,
        expected,
    );
}

/// イシュー #1329 の AC (c): `CANDIDATES[9]`（64,64,32,2,2）が全形状（正方
/// 立方・K 未実測正方出力・準正方長方形・縦長横長の代表点。
/// `examples/gemm_transpose_tile_sweep.rs::shapes()` と同一の 10 点のうち
/// 純 4096³ を除く 9 点）＋ 8 整除境界形状 (72,88,104) × NN/NT/TN/TT の
/// 4 転置パターンで parity 0 fail であることを確認する。純 4096³
/// （スカラー CPU 参照の計算コストが突出して大きい）は
/// `bk32_64x64_candidate_matches_cpu_reference_for_n4096_cubic_shape` へ
/// 分離する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn bk32_64x64_candidate_matches_cpu_reference_for_all_shapes_and_transpose_patterns() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");
    let cfg = TileConfig {
        bm: 64,
        bn: 64,
        bk: 32,
        wm: 2,
        wn: 2,
        staged: true,
    };

    let shapes: &[(usize, usize, usize)] = &[
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (2048, 2048, 64),
        (2048, 2048, 512),
        (1536, 1024, 1024),
        (1024, 1536, 1536),
        (4096, 1024, 1024),
        (1024, 4096, 1024),
        (72, 88, 104),
    ];

    for &(m, n, k) in shapes {
        let a_logical = Xorshift64Star::new(m as u64 * 7 + k as u64 + 101).fill_vec(m * k);
        let b_logical = Xorshift64Star::new(n as u64 * 11 + k as u64 + 102).fill_vec(k * n);
        let mut expected = vec![0.0f32; m * n];
        matmul_reference_fma(&a_logical, &b_logical, &mut expected, m, n, k)
            .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

        for pattern in [(false, false), (false, true), (true, false), (true, true)] {
            assert_e7_candidate_matches_reference_for_pattern(
                &ctx,
                &gemm,
                cfg,
                &a_logical,
                &b_logical,
                &expected,
                (m, n, k),
                pattern,
            );
        }
    }
}

/// 上記テストから分離した純 4096³ ケース（スカラー CPU 参照の計算コストが
/// 突出して大きいため実行時間管理のため個別実行可能にする。イシュー
/// #1329 計画「リスク・注意」節）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない（実行時間が長い）"]
fn bk32_64x64_candidate_matches_cpu_reference_for_n4096_cubic_shape() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");
    let cfg = TileConfig {
        bm: 64,
        bn: 64,
        bk: 32,
        wm: 2,
        wn: 2,
        staged: true,
    };
    let (m, n, k) = (4096usize, 4096usize, 4096usize);

    let a_logical = Xorshift64Star::new(201).fill_vec(m * k);
    let b_logical = Xorshift64Star::new(202).fill_vec(k * n);
    let mut expected = vec![0.0f32; m * n];
    matmul_reference_fma(&a_logical, &b_logical, &mut expected, m, n, k)
        .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

    for pattern in [(false, false), (false, true), (true, false), (true, true)] {
        assert_e7_candidate_matches_reference_for_pattern(
            &ctx,
            &gemm,
            cfg,
            &a_logical,
            &b_logical,
            &expected,
            (m, n, k),
            pattern,
        );
    }
}

/// イシュー #1329 の AC (e): `CANDIDATES[9]`（bk=32）と `CANDIDATES[0]`
/// （bk=16。同じ 64x64・wm2wn2 タイル形状）の出力が、統一複合判定（REQ-2）
/// の範囲内で一致することを確認する（K チャンク順を変える構成変更が
/// 数値面で許容範囲内であることの直接証跡）。bit 完全一致は assert せず
/// （K の分割粒度が異なるため丸め順が変わりうる）、一致した場合は
/// 観察としてのみ記録する（`docs/perf/metal-gemm-n4096-kernel-gap.md`
/// §13 に転記）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn bk32_64x64_candidate_agrees_with_bk16_counterpart_within_composite_tolerance() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");
    let cfg_bk32 = TileConfig {
        bm: 64,
        bn: 64,
        bk: 32,
        wm: 2,
        wn: 2,
        staged: true,
    };
    let cfg_bk16 = TileConfig {
        bm: 64,
        bn: 64,
        bk: 16,
        wm: 2,
        wn: 2,
        staged: true,
    };

    for &n in &[512usize, 1024, 2048, 4096] {
        let (m, k) = (n, n);
        let a_logical = Xorshift64Star::new(n as u64 * 13 + 301).fill_vec(m * k);
        let b_logical = Xorshift64Star::new(n as u64 * 17 + 302).fill_vec(k * n);

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

            let c_buf_32 = MetalBuffer::new_zeroed(&ctx, m * n).expect("C バッファ確保に失敗した");
            gemm.dispatch_strided_tiled_prepared(
                &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf_32, m, n, k, cfg_bk32,
            )
            .expect("CANDIDATES[9]（bk=32）のディスパッチに失敗した");
            let out_bk32 = c_buf_32.read_to_vec();

            let c_buf_16 = MetalBuffer::new_zeroed(&ctx, m * n).expect("C バッファ確保に失敗した");
            gemm.dispatch_strided_tiled_prepared(
                &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf_16, m, n, k, cfg_bk16,
            )
            .expect("CANDIDATES[0]（bk=16）のディスパッチに失敗した");
            let out_bk16 = c_buf_16.read_to_vec();

            let report = compare(&out_bk32, &out_bk16)
                .expect("compare の形状検証に失敗した（同一 m,n のため発生しないはず）");
            assert!(
                report.passes(),
                "CANDIDATES[9]（bk=32）と CANDIDATES[0]（bk=16）の出力が複合判定 FAIL \
                 （trans_a={trans_a}, trans_b={trans_b}, n={n}, fail_count={}/{}, \
                 max_abs_diff={:.3e}, max_rel_err={:.3e}）",
                report.fail_count,
                report.total,
                report.max_abs_diff,
                report.max_rel_err
            );
        }
    }
}
