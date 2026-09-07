//! E2 特殊化版（`crate::spec_source`／`MetalGemm::
//! new_with_source_specialization`。イシュー #1288 で試作・bit 一致を
//! 自己検証済み）の `MTLComputePipelineState` 反射値・N=1024/2048/4096
//! 純カーネル時間（GPU タイムスタンプ）を base（function constant 経路。
//! `source_specialized=false`）/head（ソーステキスト特殊化経路。
//! `source_specialized=true`）で before/after 比較する診断テスト
//! （イシュー #1289）。
//!
//! `docs/perf/metal-gemm-n4096-kernel-gap.md` §8.3「#1289 への引き継ぎ」
//! の指示に従い、以下 2 点を本ファイルの入口とする:
//!
//! 1. [`spec_source_reflection_dump_all_candidates`][]: `tile::CANDIDATES`
//!    全 10 候補 × base/head × NN で `MetalGemm::
//!    diag_tile_pipeline_reflection`（`gemm.rs`。`#[cfg(test)]
//!    pub(crate)`。本イシューで新設）を呼び、`maxTotalThreadsPerThreadgroup`／
//!    `threadExecutionWidth`／`staticThreadgroupMemoryLength` を取得する。
//!    ディスパッチを伴わないため秒未満で完了する（§2 の H1 仮説検証と
//!    同じプロトコル）。
//! 2. [`spec_source_kernel_gpu_ab_production_sizes`][]: N=1024/2048/4096 で
//!    base/head の `MetalGemm` インスタンスへ `gemm_reuse_phase_diag_
//!    tests::measure_one_phase_trial`（`pub(crate)`。本イシューで
//!    再利用可能にした）を trial ごとに interleaved（偶数反復
//!    base→head、奇数反復 head→base）で呼び、`kernel_gpu_secs`
//!    （イシュー #1276 の GPU タイムスタンプ変種）の中央値・
//!    `head_over_base_kernel_gpu` 比を出力する。20 warmup が head 側の
//!    候補ごと MSL 再コンパイル（初回のみ）を吸収する（`docs/perf/
//!    metal-bench-noise-protocol.md` §2 と同じ order-bias 相殺の
//!    考え方）。
//!
//! # 配置理由（`gemm_reuse_phase_diag_tests.rs` と同じ判断）
//!
//! `gemm::MetalGemm::{new_with_source_specialization,
//! diag_tile_pipeline_reflection}`（いずれも本イシュー・#1288 で追加した
//! `#[cfg(test)]` 面）・`gemm_reuse_phase_diag_tests::{measure_one_phase_
//! trial, PhaseSample, WARMUP_TRIALS, MEASURED_TRIALS, gen_square_ab,
//! median_of}`（本イシューで `pub(crate)` 化）へ到達するため、
//! integration test ではなく `lib.rs` の兄弟モジュールとして配置する。
//!
//! # 実行時は必ず `--test-threads=1`
//!
//! 同一プロセス内に base/head 2 つの `MetalGemm`（独立キャッシュ）を
//! 構築するが、いずれも `MetalContext::new()`（`cached_context()` では
//! ない専用コンテキスト）を使うため `context_cache` の singleton とは
//! 競合しない。ただし GPU 上での複数テストスレッド競合を避けるため
//! `gemm_reuse_phase_diag_tests.rs` と同じ理由で `--test-threads=1` を
//! 前提とする。
//!
//! # gating しない方針（`gemm_reuse_phase_diag_tests.rs` と同じ理由）
//!
//! 両テストとも実行が成功すること（例外なく完了すること）のみを検証
//! 条件とし、反射値・カーネル時間比の大小関係への `assert!` は行わない
//! （E2 の有効性判断はドキュメント側〈`docs/perf/metal-gemm-n4096-
//! kernel-gap.md` §9〉で人間が行う。環境揺らぎによる flaky 化防止）。
//! `resolved_cfg == requested_cfg`（フォールバック非経由）・
//! `source_specialized_cache_len`/`function_constant_cache_len` による
//! 経路切り替えの証跡のみを assert する（`gemm.rs::mod tests` の
//! `source_specialized_route_populates_only_spec_cache` と同じ判断根拠:
//! 反射値・カーネル時間が「実際にどちらの経路の値か」を出力一致とは
//! 独立に保証するため）。
//!
//! # プロダクションコード不変
//!
//! `gemm.rs` への変更は新規 `#[cfg(test)] pub(crate)` アクセサ
//! （`diag_tile_pipeline_reflection`）の追加のみ（既存 `pipeline_
//! for_tile`／`dispatch_auto`／`dispatch_variant` は無変更）。
//! `tile::SOURCE_SPECIALIZATION_ENABLED` は `false` のまま（本ファイルは
//! 明示的な `source_specialized` 引数を渡す `new_with_source_
//! specialization` 経由でのみ head 経路へアクセスする）。

use crate::context::MetalContext;
use crate::gemm::MetalGemm;
use crate::gemm_reuse_phase_diag_tests::{
    MEASURED_TRIALS, WARMUP_TRIALS, gen_square_ab, measure_one_phase_trial, median_of,
};
use crate::layout::TransposePattern;
use crate::tile;

/// [`spec_source_kernel_gpu_ab_production_sizes`] が対象とするサイズ
/// （`docs/perf/metal-gemm-reuse-phase-1277` の Phase 2 分母表と同一。
/// §8.3「#1289 への引き継ぎ」参照）。
const SIZES: [usize; 3] = [1024, 2048, 4096];

/// AC-1: `tile::CANDIDATES` 全 10 候補 × base/head（NN）の
/// `MTLComputePipelineState` 反射値をダンプする。ディスパッチを伴わない
/// ため 1 プロセス実行で足りる（プロトコル文書化は
/// `docs/perf/metal-gemm-n4096-kernel-gap.md` §9.1 を正とする）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn spec_source_reflection_dump_all_candidates() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let base_gemm = MetalGemm::new_with_source_specialization(&ctx, false)
        .expect("base GEMM パイプラインの構築に失敗した");
    let head_gemm = MetalGemm::new_with_source_specialization(&ctx, true)
        .expect("head GEMM パイプラインの構築に失敗した");

    println!(
        "candidate_index side requested_thread_count max_total_threads_per_threadgroup thread_execution_width static_threadgroup_memory_length resolved_tile"
    );
    for (i, cfg) in tile::CANDIDATES.iter().copied().enumerate() {
        for (side, gemm) in [("base", &base_gemm), ("head", &head_gemm)] {
            let r = gemm
                .diag_tile_pipeline_reflection(&ctx, cfg, TransposePattern::Nn)
                .unwrap_or_else(|e| panic!("index={i} side={side}: 反射値取得に失敗した: {e:?}"));
            assert_eq!(
                r.resolved_cfg, r.requested_cfg,
                "index={i} side={side}: フォールバックが発生した（検証が空振りする）"
            );
            println!(
                "{i} {side} {} {} {} {} {:?}",
                r.requested_thread_count,
                r.max_total_threads_per_threadgroup,
                r.thread_execution_width,
                r.static_threadgroup_memory_length,
                r.resolved_cfg,
            );
        }
    }
}

/// イシュー #1329 の AC (d): `CANDIDATES[9]`（64,64,32,2,2。E7・親 #1324）の
/// `MTLComputePipelineState` 反射値を NN/NT/TN/TT の 4 転置パターンで取得し、
/// フォールバックが発生していないこと（`resolved_cfg == requested_cfg`）・
/// `max_total_threads_per_threadgroup >= 128`（`thread_count()`＝
/// `wm*wn*32`＝128 を満たすこと）・`thread_execution_width == 32`
/// （Apple GPU SIMD 幅の定数値）を確認する。`base`（function constant
/// 経路。`SOURCE_SPECIALIZATION_ENABLED` 本番既定）のみで十分
/// （`spec_source_reflection_dump_all_candidates` が base/head 両方を
/// 巡回する形と異なり、本テストは E7 候補固有の反射値証跡が目的）。
/// `static_threadgroup_memory_length` と `TileConfig::shared_mem_bytes_for`
/// の値（`tile.rs::candidate_9_shared_mem_bytes_for_every_transpose_
/// pattern_within_32kib_and_16_aligned` で固定済み）を並べて出力し、
/// `docs/perf/metal-gemm-n4096-kernel-gap.md` §13.2 へ転記する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn candidate_9_reflection_shows_no_fallback_for_every_transpose_pattern() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");
    let cfg = tile::CANDIDATES[9];

    println!(
        "pattern requested_thread_count max_total_threads_per_threadgroup thread_execution_width static_threadgroup_memory_length shared_mem_bytes_for resolved_tile"
    );
    for pattern in [
        TransposePattern::Nn,
        TransposePattern::Nt,
        TransposePattern::Tn,
        TransposePattern::Tt,
    ] {
        let r = gemm
            .diag_tile_pipeline_reflection(&ctx, cfg, pattern)
            .unwrap_or_else(|e| panic!("pattern={pattern:?}: 反射値取得に失敗した: {e:?}"));
        assert_eq!(
            r.resolved_cfg, r.requested_cfg,
            "pattern={pattern:?}: フォールバックが発生した（検証が空振りする）"
        );
        assert!(
            r.max_total_threads_per_threadgroup >= 128,
            "pattern={pattern:?}: max_total_threads_per_threadgroup={} が 128 未満",
            r.max_total_threads_per_threadgroup
        );
        assert_eq!(
            r.thread_execution_width, 32,
            "pattern={pattern:?}: thread_execution_width が 32 ではない"
        );
        println!(
            "{pattern:?} {} {} {} {} {} {:?}",
            r.requested_thread_count,
            r.max_total_threads_per_threadgroup,
            r.thread_execution_width,
            r.static_threadgroup_memory_length,
            cfg.shared_mem_bytes_for(pattern),
            r.resolved_cfg,
        );
    }
}

/// AC-2: N=1024/2048/4096 で base/head の `kernel_gpu`（GPU タイム
/// スタンプによる純カーネル専有時間。イシュー #1276）を trial ごとに
/// interleaved で計測し、5 プロセス起動の 1 回分として中央値・
/// `head_over_base_kernel_gpu` 比を出力する（複数プロセス起動・集計は
/// `docs/perf/metal-gemm-n4096-kernel-gap.md` §9.1 の手順書側で行う）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn spec_source_kernel_gpu_ab_production_sizes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let base_gemm = MetalGemm::new_with_source_specialization(&ctx, false)
        .expect("base GEMM パイプラインの構築に失敗した");
    let head_gemm = MetalGemm::new_with_source_specialization(&ctx, true)
        .expect("head GEMM パイプラインの構築に失敗した");

    for n in SIZES {
        let cfg = tile::select_for_device(n, n, n, ctx.verified_m4_max_gpu_core_count());
        let (a, b) = gen_square_ab(0x1289_a000 ^ (n as u64), n);

        let mut base_keep_alive: Vec<Vec<f32>> =
            Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);
        let mut head_keep_alive: Vec<Vec<f32>> =
            Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);

        // warmup: head 側の初回 MSL 再コンパイル（候補ごとの特殊化
        // ソース再コンパイル）を吸収する（ファイル冒頭コメント参照）。
        for _ in 0..WARMUP_TRIALS {
            let _ = measure_one_phase_trial(&ctx, &base_gemm, &a, &b, n, cfg, &mut base_keep_alive);
            let _ = measure_one_phase_trial(&ctx, &head_gemm, &a, &b, n, cfg, &mut head_keep_alive);
        }

        let mut base_kernel_gpu = Vec::with_capacity(MEASURED_TRIALS);
        let mut head_kernel_gpu = Vec::with_capacity(MEASURED_TRIALS);
        let mut base_resolved: Option<tile::TileConfig> = None;
        let mut head_resolved: Option<tile::TileConfig> = None;

        for i in 0..MEASURED_TRIALS {
            // trial 偶奇で計測順を反転し order-bias（サーマル・スケジュ
            // ーラ由来のドリフト）を相殺する（`docs/perf/metal-bench-
            // noise-protocol.md` §2 と同じ手法。ファイル冒頭コメント
            // 参照）。
            let (base_sample, head_sample) = if i % 2 == 0 {
                let b_s =
                    measure_one_phase_trial(&ctx, &base_gemm, &a, &b, n, cfg, &mut base_keep_alive);
                let h_s =
                    measure_one_phase_trial(&ctx, &head_gemm, &a, &b, n, cfg, &mut head_keep_alive);
                (b_s, h_s)
            } else {
                let h_s =
                    measure_one_phase_trial(&ctx, &head_gemm, &a, &b, n, cfg, &mut head_keep_alive);
                let b_s =
                    measure_one_phase_trial(&ctx, &base_gemm, &a, &b, n, cfg, &mut base_keep_alive);
                (b_s, h_s)
            };
            // codex-review 指摘（イシュー #1289 PR #1379）: 要求構成
            // `cfg` と実際にディスパッチされた構成（`pipeline_for_tile`
            // フォールバック解決後）が base/head 双方で一致することを
            // サンプル追加前に毎反復検証する。`gemm_reuse_phase_diag_
            // tests` の `fallback_occurred` 記録（#1189 指摘対応）は
            // フォールバックを許容したうえで注記するだけだが、本テスト
            // は base/head の「同一構成での」性能比較が前提のため、
            // 片側だけフォールバックした反復を性能比較として集計する
            // と E2 特殊化の有効性判断を誤らせる。よって記録ではなく
            // fail-closed な assert とする。
            assert_eq!(
                base_sample.resolved_cfg, cfg,
                "N={n} trial={i}: base 側で pipeline_for_tile フォールバックが発生した(requested={cfg:?}, resolved={:?})。性能比較の前提が崩れるため中断する",
                base_sample.resolved_cfg
            );
            assert_eq!(
                head_sample.resolved_cfg, cfg,
                "N={n} trial={i}: head 側で pipeline_for_tile フォールバックが発生した(requested={cfg:?}, resolved={:?})。性能比較の前提が崩れるため中断する",
                head_sample.resolved_cfg
            );
            base_kernel_gpu.push(base_sample.kernel_gpu_secs);
            head_kernel_gpu.push(head_sample.kernel_gpu_secs);
            base_resolved = Some(base_sample.resolved_cfg);
            head_resolved = Some(head_sample.resolved_cfg);
        }

        // 経路切り替えの証跡（出力一致だけでは両経路が同じ実装へ倒れた
        // false-green を検出できないため。`gemm.rs::mod tests::
        // source_specialized_route_populates_only_spec_cache` と同じ
        // 判断根拠）。
        let base_spec_len = base_gemm
            .source_specialized_cache_len()
            .expect("base のキャッシュ長取得に失敗した");
        let base_fc_len = base_gemm
            .function_constant_cache_len()
            .expect("base のキャッシュ長取得に失敗した");
        let head_spec_len = head_gemm
            .source_specialized_cache_len()
            .expect("head のキャッシュ長取得に失敗した");
        let head_fc_len = head_gemm
            .function_constant_cache_len()
            .expect("head のキャッシュ長取得に失敗した");
        assert_eq!(
            base_spec_len, 0,
            "N={n}: base は tiled_spec_cache が空のはず"
        );
        assert!(
            base_fc_len > 0,
            "N={n}: base は function_constant_cache が増えているはず"
        );
        assert!(
            head_spec_len > 0,
            "N={n}: head は tiled_spec_cache が増えているはず"
        );
        assert_eq!(
            head_fc_len, 0,
            "N={n}: head は function_constant_cache が空のはず"
        );

        let base_q = median_of(&base_kernel_gpu);
        let head_q = median_of(&head_kernel_gpu);
        let ratio = head_q.median / base_q.median;

        println!(
            "N={n} requested_tile={cfg:?} base_resolved_tile={:?} head_resolved_tile={:?}",
            base_resolved.expect("MEASURED_TRIALS > 0 のため必ず Some"),
            head_resolved.expect("MEASURED_TRIALS > 0 のため必ず Some"),
        );
        println!(
            "  side=base kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
            base_q.median * 1e3,
            base_q.q1 * 1e3,
            base_q.q3 * 1e3
        );
        println!(
            "  side=head kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
            head_q.median * 1e3,
            head_q.q1 * 1e3,
            head_q.q3 * 1e3
        );
        println!(
            "N={n} head_over_base_kernel_gpu={:.6} base_spec_cache_len={base_spec_len} base_fc_cache_len={base_fc_len} head_spec_cache_len={head_spec_len} head_fc_cache_len={head_fc_len}",
            ratio
        );
    }
}
