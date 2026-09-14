//! `gemm_simdgroup_tiled_te`（`thread_elements()` 方式 BlockMMA 候補
//! カーネル。イシュー #1693・親 #1586）の全形状 × 転置 4 パターンの
//! parity 確認・本番 `gemm_simdgroup_tiled` との bit 一致・非 staged 拒否。
//!
//! 本候補は既存の `MetalGemm::pipeline_for_tile`（classic 経路）自身が
//! `mma_frag_load` フィールド（`MetalGemm::new_with_mma_frag_load` で
//! 明示できる instance ゲート）に応じてカーネル関数名を切り替える設計
//! （`crates/backend-metal/src/gemm.rs::pipeline_for_tile` 冒頭コメント
//! 参照）のため、専用の dispatch 関数は持たない。本ファイルは
//! `ThreadElements` インスタンスに対して既存の本番公開入口
//! （`dispatch_strided_tiled_prepared`・`dispatch_variant`）をそのまま
//! 呼ぶことで候補カーネルを検証する（`tests/gemm_strided_parity.rs` と
//! 同型の構成）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。全ケース
//! `#[ignore]`（実機依存テストの分離。`.claude/rules/coding-rust.md`）。
//! 実行するには macOS 実機で以下を叩く:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test gemm_te_parity \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! **性能実測・本番結線可否判断は兄弟イシュー #1694 のスコープ**
//! （`docs/perf/metal-gemm-thread-elements-candidate.md`）。

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::{assert_parity, matmul_reference_fma};
use fandhe_ai_backend_metal::gemm::GemmVariant;
use fandhe_ai_backend_metal::layout::{MatrixLayout, classify_2d};
use fandhe_ai_backend_metal::tile::{self, MmaFragLoad, TileConfig};
use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm};

/// `src`（`rows`×`cols`、行優先）の転置（`cols`×`rows`、行優先）を返す
/// （`tests/gemm_strided_parity.rs::transpose_dense` と同一）。
fn transpose_dense(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

/// `tile::select` は小さい／端数形状に対して非 staged 構成
/// （`TileConfig::SINGLE_SIMDGROUP_8X8` 等）を選びうるが、te は staged
/// 経路のみを実装する契約（`gemm.rs::pipeline_for_tile` の te ガードが
/// 非 staged 候補を拒否する。`te_rejects_non_staged_candidate` 参照）。
/// `select` が非 staged を返す構成向けの各 parity テストは、代わりに
/// 常に staged な固定構成を使う（`tests/gemm_hfrag_parity.rs::
/// staged_fallback_cfg` と同一値）。
fn staged_fallback_cfg() -> TileConfig {
    TileConfig {
        bm: 32,
        bn: 32,
        bk: 16,
        wm: 2,
        wn: 2,
        staged: true,
    }
}

/// `tile::select(m, n, k)` の結果が staged ならそのまま、非 staged なら
/// [`staged_fallback_cfg`] を返す（te が対応する構成のみを選ぶ）。
fn select_staged(m: usize, n: usize, k: usize) -> TileConfig {
    let cfg = tile::select(m, n, k);
    if cfg.staged {
        cfg
    } else {
        staged_fallback_cfg()
    }
}

const PATTERNS: [(bool, bool); 4] = [(false, false), (false, true), (true, false), (true, true)];

/// 8 整除形状（`dispatch_strided_tiled_prepared` の適格性ゲートを通過
/// する形状）× 全転置パターンの parity を確認する（`tests/
/// gemm_strided_parity.rs::dispatch_strided_tiled_prepared_matches_
/// cpu_reference_for_all_transpose_patterns` と同型の構成。`gemm` には
/// `MetalGemm::new_with_mma_frag_load(&ctx, ThreadElements)` を渡す）。
fn run_strided_case(ctx: &MetalContext, gemm: &MetalGemm, m: usize, n: usize, k: usize, seed: u64) {
    let cfg = select_staged(m, n, k);
    let a_logical = Xorshift64Star::new(seed).fill_vec(m * k);
    let b_logical = Xorshift64Star::new(seed + 1).fill_vec(k * n);
    let mut expected = vec![0.0f32; m * n];
    matmul_reference_fma(&a_logical, &b_logical, &mut expected, m, n, k)
        .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

    for (trans_a, trans_b) in PATTERNS {
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

        let a_buf = MetalBuffer::new_with_data(ctx, &a_phys).expect("A バッファ確保に失敗した");
        let b_buf = MetalBuffer::new_with_data(ctx, &b_phys).expect("B バッファ確保に失敗した");
        let c_buf = MetalBuffer::new_zeroed(ctx, m * n).expect("C バッファ確保に失敗した");

        let resolved = gemm
            .dispatch_strided_tiled_prepared(
                ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf, m, n, k, cfg,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "te dispatch_strided_tiled_prepared failed (trans_a={trans_a}, \
                     trans_b={trans_b}, m={m}, n={n}, k={k}): {e}"
                )
            });
        assert_eq!(
            resolved, cfg,
            "te（trans_a={trans_a}, trans_b={trans_b}, m={m}, n={n}, k={k}）が \
             フォールバックせず指定 cfg を採用する想定"
        );

        let actual = c_buf.read_to_vec();
        assert_parity(
            &format!(
                "te dispatch_strided_tiled_prepared parity \
                 (trans_a={trans_a}, trans_b={trans_b}, m={m}, n={n}, k={k})"
            ),
            &actual,
            &expected,
        );
    }
}

/// 正方 8 整列形状 × 全転置パターンの parity を確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn te_square_shapes_all_patterns() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new_with_mma_frag_load(&ctx, MmaFragLoad::ThreadElements)
        .expect("GEMM パイプラインの構築に失敗した");

    for &size in &[64usize, 128, 512, 1024, 2048] {
        run_strided_case(&ctx, &gemm, size, size, size, 0x1693_1000 + size as u64);
    }
}

/// 縦長・横長・K 末尾（いずれも 8 整除）形状 × 全転置パターンの parity
/// を確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn te_tall_wide_and_k_tail_shapes_all_patterns() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new_with_mma_frag_load(&ctx, MmaFragLoad::ThreadElements)
        .expect("GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in &[
        (2048usize, 256usize, 512usize),
        (256, 2048, 512),
        (96, 96, 40),
    ] {
        run_strided_case(
            &ctx,
            &gemm,
            m,
            n,
            k,
            0x1693_5000 + m as u64 + n as u64 + k as u64,
        );
    }
}

/// 端数形状（`pad8` 経由の 0 パディング経路）を NN 限定で確認する
/// （`dispatch_strided_tiled_prepared` は m/n/k いずれも 8 整除を要求
/// するため対象外。`dispatch_variant` は `pad8` で内部的にパディングする
/// ため端数形状も扱える）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn te_ragged_shapes_nn() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new_with_mma_frag_load(&ctx, MmaFragLoad::ThreadElements)
        .expect("GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in &[(60usize, 68usize, 36usize), (68, 60, 20), (63, 65, 33)] {
        // `resolve_tile_config` は `pub(crate)`（クレート境界の外である
        // 本ファイルからは参照できない）ため、フォールバック非経由の
        // 明示検証は行わず出力の parity のみを確認する（クレート内テスト
        // `crate::gemm::tests::all_staged_candidates_match_te_cpu_
        // reference_512_nn` が正方形状でフォールバック非経由を確認する）。
        let cfg = select_staged(m, n, k);
        let a = Xorshift64Star::new(0x1693_3000 + m as u64).fill_vec(m * k);
        let b = Xorshift64Star::new(0x1693_4000 + n as u64).fill_vec(k * n);
        let out = gemm
            .dispatch_variant(&ctx, GemmVariant::SimdgroupTiled(cfg), &a, &b, m, n, k)
            .unwrap_or_else(|e| panic!("te dispatch_variant failed (m={m}, n={n}, k={k}): {e}"));

        let mut expected = vec![0.0f32; m * n];
        matmul_reference_fma(&a, &b, &mut expected, m, n, k)
            .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");
        assert_parity(
            &format!("te ragged NN gemm m={m} n={n} k={k}"),
            &out,
            &expected,
        );
    }
}

// 全 staged `CANDIDATES` の 512³ NN 総当たりは `CANDIDATES` が
// `pub(crate)`（クレート内部表現）のため本ファイル（クレート境界の外）
// からは参照できない。`crate::gemm::tests::
// all_staged_candidates_match_te_cpu_reference_512_nn`
// （`crates/backend-metal/src/gemm.rs` の `#[cfg(test)] mod tests`。
// クレート内テスト）が同じ役割を担う（`tests/gemm_hfrag_parity.rs` の
// 同種コメントと同じ設計判断）。同様に R0 前提ゲート（`thread_elements()`
// レイアウト probe。`diag_probe_thread_elements_layout` が `#[cfg(test)]`
// 限定のため）は `crate::gemm::tests::te_layout_probe_matches_model` が
// 担う。

/// 非 staged 候補（`SINGLE_SIMDGROUP_8X8` を含む）が
/// `pipeline_for_tile`（te ガード）から fail-closed に拒否されることを
/// 確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn te_rejects_non_staged_candidate() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new_with_mma_frag_load(&ctx, MmaFragLoad::ThreadElements)
        .expect("GEMM パイプラインの構築に失敗した");

    #[allow(clippy::assertions_on_constants)]
    {
        assert!(
            !TileConfig::SINGLE_SIMDGROUP_8X8.staged,
            "本テストの前提（SINGLE_SIMDGROUP_8X8 は非 staged）が崩れています"
        );
    }

    // `resolve_tile_config` は `pub(crate)` のため本ファイル（クレート
    // 境界の外）からは参照できない。`pub fn dispatch_strided_tiled_
    // prepared`（8 整除の 64³ NN・適格性ゲートは通過する）へ非 staged
    // 候補を明示的に渡すことで、`pipeline_for_tile` の te ガードが
    // fail-closed に拒否することを外側から確認する。
    let (m, n, k) = (64usize, 64usize, 64usize);
    let a = vec![0.0f32; m * k];
    let b = vec![0.0f32; k * n];
    let a_buf = MetalBuffer::new_with_data(&ctx, &a).expect("A バッファ確保に失敗した");
    let b_buf = MetalBuffer::new_with_data(&ctx, &b).expect("B バッファ確保に失敗した");
    let c_buf = MetalBuffer::new_zeroed(&ctx, m * n).expect("C バッファ確保に失敗した");
    let a_layout = classify_2d(&[m, k], &[k as isize, 1]).unwrap();
    let b_layout = classify_2d(&[k, n], &[n as isize, 1]).unwrap();

    let err = gemm
        .dispatch_strided_tiled_prepared(
            &ctx,
            &a_buf,
            0,
            a_layout,
            &b_buf,
            0,
            b_layout,
            &c_buf,
            m,
            n,
            k,
            TileConfig::SINGLE_SIMDGROUP_8X8,
        )
        .expect_err("非 staged 候補（SINGLE_SIMDGROUP_8X8）は te インスタンスで拒否される想定");
    let message = format!("{err}");
    assert!(
        message.contains("(te)"),
        "非 staged 候補拒否時のエラーメッセージに te の言及がない: {message}"
    );
}

/// 本番 `gemm_simdgroup_tiled`（`MetalGemm::new`。`dispatch_variant`
/// 経由）と候補 `gemm_simdgroup_tiled_te`（同一 `cfg`・NN・正方形状）の
/// 出力が bit 完全一致することを確認する（`docs/perf/
/// metal-gemm-thread-elements-candidate.md` §2「数値契約」: 演算オペランド
/// 列・共有メモリへ格納する値は本番 staged 経路と完全に同一のため、
/// レーン→要素レイアウトが実機で `thread_elements_coord` モデルと一致
/// すれば bit 同一が期待できる、という設計仮説を実機で検証する）。
/// 不一致は調査対象であり緩和しない（正式ゲートは REQ-2 だが、本テストは
/// より厳格な bit 一致を追加で確認する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn te_bit_match_with_production_dispatch_auto() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let base = MetalGemm::new(&ctx).expect("base GEMM パイプラインの構築に失敗した");
    let head = MetalGemm::new_with_mma_frag_load(&ctx, MmaFragLoad::ThreadElements)
        .expect("head（te）GEMM パイプラインの構築に失敗した");

    for &size in &[512usize, 1024, 2048] {
        let cfg = select_staged(size, size, size);
        let a = Xorshift64Star::new(0x1693_7000 + size as u64).fill_vec(size * size);
        let b = Xorshift64Star::new(0x1693_8000 + size as u64).fill_vec(size * size);

        let base_out = base
            .dispatch_variant(
                &ctx,
                GemmVariant::SimdgroupTiled(cfg),
                &a,
                &b,
                size,
                size,
                size,
            )
            .unwrap_or_else(|err| {
                panic!("base（本番）ディスパッチに失敗した（size={size}）: {err}")
            });
        let head_out = head
            .dispatch_variant(
                &ctx,
                GemmVariant::SimdgroupTiled(cfg),
                &a,
                &b,
                size,
                size,
                size,
            )
            .unwrap_or_else(|err| panic!("head（te）ディスパッチに失敗した（size={size}）: {err}"));

        assert_eq!(
            base_out.len(),
            head_out.len(),
            "base/head の出力長が一致しない（size={size}）"
        );
        for (idx, (&b_val, &h_val)) in base_out.iter().zip(head_out.iter()).enumerate() {
            assert_eq!(
                b_val.to_bits(),
                h_val.to_bits(),
                "size={size} idx={idx}: base（本番 {b_val}）と head（te {h_val}）が bit 不一致"
            );
        }
    }
}
