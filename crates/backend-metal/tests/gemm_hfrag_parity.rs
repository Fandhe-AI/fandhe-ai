//! `gemm_simdgroup_tiled_hfrag`（half フラグメント／f32 累算候補。
//! イシュー #1369・親 #1368 E9）の全形状 × 転置 4 パターンの parity 確認。
//!
//! REQ-2 の「Tensor Core 経路の受け入れ判定方式」（`docs/spec/
//! 04-requirements.md` REQ-2「2026-09-02 追記」・`.claude/rules/
//! coding-rust.md`「TF32/f16 Tensor Core 経路の parity テスト判定方式」）に
//! 従い、本ファイルは 2 系統の参照で判定する:
//!
//! 1. **正しさゲート（fail-closed・厳密）**: 入力を `half::f16::from_f32`
//!    で丸めて f32 へ戻した配列に対する `matmul_reference_fma` と候補出力を
//!    `assert_parity`（REQ-2 統一複合判定）で照合する。本カーネルは丸め
//!    済み入力に対して f32 累算するため、この参照とは統一複合判定で一致
//!    するべきという設計（`shaders/gemm.metal::gemm_simdgroup_tiled_hfrag`
//!    冒頭コメント「数値契約」参照）。あわせて resolved [`TileConfig`] が
//!    指定 `cfg` と一致すること（フォールバック非経由）も assert する。
//! 2. **REQ-2 形状別判定用の実測記録（ゲートしない）**: 丸めなし f32 入力の
//!    `matmul_reference_fma` と `fandhe_ai_backend_cpu::parity::compare` で
//!    `CompareReport` を取得し標準出力へ記録する。ベースライン行の追加・
//!    非後退ゲート化は人間承認必須（coding-rust.md）のため本ファイルでは
//!    行わない。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。全ケース
//! `#[ignore]`（実機依存テストの分離。`.claude/rules/coding-rust.md`）。
//! 実行するには macOS 実機で以下を叩く:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test gemm_hfrag_parity \
//!   -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::{assert_parity, compare, matmul_reference_fma};
use fandhe_ai_backend_metal::layout::TransposePattern;
use fandhe_ai_backend_metal::tile::{self, TileConfig};
use fandhe_ai_backend_metal::{MetalContext, MetalGemm};

/// `src`（`rows`×`cols`、行優先）の転置（`cols`×`rows`、行優先）を返す。
/// `dispatch_hfrag_tiled_unverified` へ渡すストレージ（`TRANS_A`/`TRANS_B`
/// 時の `[k,m]`/`[n,k]` 行優先ストレージ）を、テスト側の論理形状（`[m,k]`/
/// `[k,n]`）から機械的に作るためのヘルパ（GPU 側の添字規約と対応する。
/// `crate::gemm::MetalGemm::dispatch_hfrag_tiled_unverified` doc comment
/// 「ストレージ形状」参照）。
fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

/// 1 ケース（`pattern`・`(m, n, k)`・`cfg`）を実行し、2 系統の参照との
/// 突合結果を返す（正しさゲートは呼び出し側で assert する。統計記録は
/// 呼び出し側が println する）。
struct CaseResult {
    resolved: TileConfig,
    rounded_gate_out: Vec<f32>,
    rounded_gate_expected: Vec<f32>,
    unrounded_report: fandhe_ai_backend_cpu::parity::CompareReport,
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    ctx: &MetalContext,
    gemm: &MetalGemm,
    pattern: TransposePattern,
    cfg: TileConfig,
    seed_a: u64,
    seed_b: u64,
    m: usize,
    n: usize,
    k: usize,
) -> CaseResult {
    use TransposePattern::{Nn, Nt, Tn, Tt};

    let mut rng_a = Xorshift64Star::new(seed_a);
    let mut rng_b = Xorshift64Star::new(seed_b);
    // 論理形状（転置前の意味論）: A は `[m, k]`、B は `[k, n]`。
    let a_logical = rng_a.fill_vec(m * k);
    let b_logical = rng_b.fill_vec(k * n);

    // `dispatch_hfrag_tiled_unverified` が要求するストレージ形状へ変換する
    // （`Tn`/`Tt` は A を `[k, m]` へ、`Nt`/`Tt` は B を `[n, k]` へ転置。
    // `crate::gemm::MetalGemm::dispatch_hfrag_tiled_unverified` doc comment
    // 参照）。
    let a_storage = match pattern {
        Nn | Nt => a_logical.clone(),
        Tn | Tt => transpose(&a_logical, m, k),
    };
    let b_storage = match pattern {
        Nn | Tn => b_logical.clone(),
        Nt | Tt => transpose(&b_logical, k, n),
    };

    let (out, resolved) = gemm
        .dispatch_hfrag_tiled_unverified(ctx, &a_storage, &b_storage, m, n, k, pattern, cfg)
        .unwrap_or_else(|err| {
            panic!(
                "hfrag({pattern:?}, cfg={cfg:?}) m={m} n={n} k={k}: ディスパッチに失敗した: {err}"
            )
        });

    // 正しさゲート（丸め済み入力参照）。
    let a_h: Vec<f32> = a_logical
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
    let b_h: Vec<f32> = b_logical
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
    let mut rounded_gate_expected = vec![0.0f32; m * n];
    matmul_reference_fma(&a_h, &b_h, &mut rounded_gate_expected, m, n, k)
        .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した（丸め済み入力）");

    // REQ-2 形状別判定用の実測記録（丸めなし入力参照。ゲートしない）。
    let mut unrounded_expected = vec![0.0f32; m * n];
    matmul_reference_fma(&a_logical, &b_logical, &mut unrounded_expected, m, n, k)
        .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した（丸めなし入力）");
    let unrounded_report =
        compare(&out, &unrounded_expected).expect("compare の長さ検証に失敗した");

    CaseResult {
        resolved,
        rounded_gate_out: out,
        rounded_gate_expected,
        unrounded_report,
    }
}

/// 1 ケースを実行し、正しさゲート（丸め済み入力参照との統一複合判定）を
/// assert したうえで、REQ-2 形状別判定用の実測統計を 1 行 println する。
#[allow(clippy::too_many_arguments)]
fn run_and_report(
    ctx: &MetalContext,
    gemm: &MetalGemm,
    label: &str,
    pattern: TransposePattern,
    cfg: TileConfig,
    seed_a: u64,
    seed_b: u64,
    m: usize,
    n: usize,
    k: usize,
) {
    let result = run_case(ctx, gemm, pattern, cfg, seed_a, seed_b, m, n, k);
    assert_eq!(
        result.resolved, cfg,
        "{label}: hfrag がフォールバックせず指定 cfg を採用する想定（cfg={cfg:?} resolved={:?}）",
        result.resolved
    );
    assert_parity(
        &format!(
            "{label}: metal hfrag {pattern:?} gemm m={m} n={n} k={k}（丸め済み入力参照・正しさゲート）"
        ),
        &result.rounded_gate_out,
        &result.rounded_gate_expected,
    );

    let r = result.unrounded_report;
    println!(
        "hfrag_parity label={label} pattern={pattern:?} m={m} n={n} k={k} cfg={cfg:?} \
         strict_exact={} total={} fail_count={} mean_abs_diff={:.6e} max_abs_diff={:.6e} \
         max_rel_err={:.6e}",
        r.passes(),
        r.total,
        r.fail_count,
        r.mean_abs_diff,
        r.max_abs_diff,
        r.max_rel_err,
    );
}

/// `tile::select` は小さい／端数形状に対して非 staged 構成
/// （`TileConfig::SINGLE_SIMDGROUP_8X8` 等）を選びうるが、hfrag は staged
/// 経路のみ実装する契約（`pipeline_for_tile_hfrag` が非 staged 候補を
/// 拒否する。`non_staged_candidate_is_rejected` 参照）。`select` が非
/// staged を返す構成向けの各 parity テストは、代わりに常に staged な
/// 固定構成（`square_shapes_all_patterns` の 64³ ケースと同一値）を使う。
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
/// [`staged_fallback_cfg`] を返す（hfrag が対応する構成のみを選ぶ）。
fn select_staged(m: usize, n: usize, k: usize) -> TileConfig {
    let cfg = tile::select(m, n, k);
    if cfg.staged {
        cfg
    } else {
        staged_fallback_cfg()
    }
}

const PATTERNS: [TransposePattern; 4] = [
    TransposePattern::Nn,
    TransposePattern::Nt,
    TransposePattern::Tn,
    TransposePattern::Tt,
];

/// 正方 8 整列形状（本番選択構成。`tile::select` が選ぶ cfg）× 全転置
/// パターンの parity を確認する（実装計画 §2.4 形状セット 1 項目）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn square_shapes_all_patterns() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for &size in &[64usize, 128, 512, 1024] {
        let cfg = select_staged(size, size, size);
        for (i, &pattern) in PATTERNS.iter().enumerate() {
            run_and_report(
                &ctx,
                &gemm,
                "square",
                pattern,
                cfg,
                0x1369_1000 + size as u64 * 10 + i as u64,
                0x1369_2000 + size as u64 * 10 + i as u64,
                size,
                size,
                size,
            );
        }
    }
}

/// 端数形状（`pad8` 経由の 0 パディング経路）× 全転置パターンの parity
/// を確認する（実装計画 §2.4 形状セット 2 項目）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn ragged_shapes_all_patterns() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in &[(60usize, 68usize, 36usize), (68, 60, 20), (63, 65, 33)] {
        let cfg = select_staged(m, n, k);
        for (i, &pattern) in PATTERNS.iter().enumerate() {
            run_and_report(
                &ctx,
                &gemm,
                "ragged",
                pattern,
                cfg,
                0x1369_3000 + i as u64,
                0x1369_4000 + i as u64,
                m,
                n,
                k,
            );
        }
    }
}

/// 縦長・横長・K 末尾形状 × 全転置パターンの parity を確認する
/// （実装計画 §2.4 形状セット 3 項目）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tall_wide_and_k_tail_shapes_all_patterns() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for &(label, m, n, k) in &[
        ("tall", 2048usize, 256usize, 512usize),
        ("wide", 256, 2048, 512),
        ("k_tail", 96, 96, 40),
    ] {
        let cfg = select_staged(m, n, k);
        for (i, &pattern) in PATTERNS.iter().enumerate() {
            run_and_report(
                &ctx,
                &gemm,
                label,
                pattern,
                cfg,
                0x1369_5000 + i as u64,
                0x1369_6000 + i as u64,
                m,
                n,
                k,
            );
        }
    }
}

// 全 staged `CANDIDATES` の 512³ NN 総当たりは `CANDIDATES` が
// `pub(crate)`（クレート内部表現）のため本ファイル（クレート境界の外）
// からは参照できない。`crate::gemm::tests::
// all_staged_candidates_match_hfrag_cpu_reference_512_nn`
// （`crates/backend-metal/src/gemm.rs` の `#[cfg(test)] mod tests`。
// クレート内テスト）が同じ役割を担う（実装計画 §2.4 形状セット 4 項目。
// `resolve_tile_config_f16` 等の既存クレート内テストと同じ設計判断）。

/// 非 staged 候補（`SINGLE_SIMDGROUP_8X8` を含む）が
/// `pipeline_for_tile_hfrag` から fail-closed に拒否されることを確認する
/// （実装計画 §2.1「direct-load 経路は実装しない」の Rust 側検証）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn non_staged_candidate_is_rejected() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    #[allow(clippy::assertions_on_constants)]
    {
        assert!(
            !TileConfig::SINGLE_SIMDGROUP_8X8.staged,
            "本テストの前提（SINGLE_SIMDGROUP_8X8 は非 staged）が崩れています"
        );
    }

    let mut rng_a = Xorshift64Star::new(0x1369_9001);
    let mut rng_b = Xorshift64Star::new(0x1369_9002);
    let (m, n, k) = (64usize, 64usize, 64usize);
    let a = rng_a.fill_vec(m * k);
    let b = rng_b.fill_vec(k * n);

    let err = gemm
        .dispatch_hfrag_tiled_unverified(
            &ctx,
            &a,
            &b,
            m,
            n,
            k,
            TransposePattern::Nn,
            TileConfig::SINGLE_SIMDGROUP_8X8,
        )
        .expect_err("非 staged 候補（SINGLE_SIMDGROUP_8X8）は拒否される想定");
    let message = format!("{err}");
    assert!(
        message.contains("hfrag"),
        "非 staged 候補拒否時のエラーメッセージに hfrag の言及がない: {message}"
    );
}
