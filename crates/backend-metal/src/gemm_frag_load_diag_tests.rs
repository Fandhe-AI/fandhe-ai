//! E3 フラグメントロード方式候補（`tile::FragLoadConfig`。イシュー #1293
//! で実装・bit 一致を自己検証済み）の N=1024/2048/4096 純カーネル時間
//! （GPU タイムスタンプ。イシュー #1276 の `kernel_gpu` 変種）を M4 Max
//! で 5 回計測し、有効性（`tile::select` への組み込み可否・採用候補）を
//! 判定する診断テスト（イシュー #1295）。
//!
//! `docs/perf/metal-gemm-frag-load-candidates.md` §5「#1295 への引き継ぎ」
//! の指示に従い、以下 1 点を本ファイルの入口とする:
//!
//! - [`frag_load_kernel_gpu_ab_production_sizes`][]: N=1024/2048/4096 で
//!   5 候補（`tgp-k1`〈= 本番既定 `tile::FRAG_LOAD_CONFIG`〉・`tgp-k2`・
//!   `device-legacy`・`device-hoisted-k1`・`device-hoisted-k2`）の
//!   `MetalGemm` インスタンスへ `gemm_reuse_phase_diag_tests::
//!   measure_one_phase_trial`（`pub(crate)`）を trial ごとに交互
//!   （trial index による開始オフセット回転）で呼び、`kernel_gpu_secs`
//!   の中央値・base（`tgp-k1`）比・`device-legacy` 比を出力する。
//!
//! # 候補と `TileConfig` の組み立て方
//!
//! `tgp-k1`／`tgp-k2` は `staged=true` の本番選択構成（`tile::
//! select_for_device`）をそのまま使う（`FragLoadConfig.device_hoisted` は
//! staged 経路では no-op。`docs/perf/metal-gemm-frag-load-candidates.md`
//! §1 候補表）。`device-legacy`／`device-hoisted-k1`／`device-hoisted-k2`
//! は `TileConfig { staged: false, ..cfg }`（同ドキュメント §5「`{staged:
//! false}` twin の作り方」）を使う。twin は `validate` が通らない場合
//! 対象外とする（`gemm.rs::frag_load_tgp_vs_device_same_shape_bit_match`
//! と同じ設計）。
//!
//! # 配置理由（`gemm_spec_source_diag_tests.rs` と同じ判断）
//!
//! `gemm::MetalGemm::new_with_frag_load`（イシュー #1293 で追加した `pub`
//! 面）・`gemm_reuse_phase_diag_tests::{measure_one_phase_trial,
//! PhaseSample, WARMUP_TRIALS, MEASURED_TRIALS, gen_square_ab, median_of}`
//! （いずれも `pub(crate)`）へ到達するため、integration test ではなく
//! `lib.rs` の兄弟モジュールとして配置する。
//!
//! # 実行時は必ず `--test-threads=1`
//!
//! 同一プロセス内に 5 つの `MetalGemm`（いずれも `MetalContext::new()`。
//! `cached_context()` ではない専用コンテキスト 1 個を共有）を構築するが、
//! GPU 上での複数テストスレッド競合を避けるため `gemm_spec_source_diag_
//! tests.rs` と同じ理由で `--test-threads=1` を前提とする。
//!
//! # メモリ使用量
//!
//! ホスト側キープアライブ（`Vec<Vec<f32>>`）は N=4096・5 候補同時保持で
//! (20 warmup + 20 測定) × 4096² × 4 bytes × 5 候補 ≈ 13.4 GiB
//! （統合メモリ上）。N ごとのループスコープで各候補の `keep_alive` を
//! drop してから次の N へ進むことでピークを 1 N 分に抑える
//! （`gemm_spec_source_diag_tests.rs` の base/head 2 インスタンス版と
//! 同じ考え方の 5 候補版）。
//!
//! # gating しない方針（`gemm_spec_source_diag_tests.rs` と同じ理由）
//!
//! 実行が成功すること（例外なく完了すること）のみを検証条件とし、
//! `kernel_gpu` の大小関係への `assert!` は行わない（E3 の有効性判断は
//! ドキュメント側〈`docs/perf/metal-gemm-n4096-kernel-gap.md` §10〉で
//! 人間が行う。環境揺らぎによる flaky 化防止）。`resolved_cfg ==
//! requested_cfg`（フォールバック非経由。base は本番選択構成・device 系
//! は twin）のみを fail-closed に assert する（片側だけフォールバックした
//! 反復を性能比較として集計すると E3 の有効性判断を誤らせるため。
//! `gemm_spec_source_diag_tests.rs` の同種 assert と同じ根拠）。
//!
//! # プロダクションコード不変
//!
//! 本ファイルは `gemm.rs`／`tile.rs`／`shaders/gemm.metal` への変更を
//! 一切含まない（イシュー #1293 が既に追加済みの `#[cfg(test)] pub`／
//! `pub(crate)` 面のみを利用する）。`tile::select` の候補表・本番既定
//! （`MetalGemm::new` の `tile::FRAG_LOAD_CONFIG`）への組み込みは本イシュー
//! のスコープ外（兄弟イシュー #1302 が担う。`docs/perf/metal-gemm-
//! frag-load-candidates.md` §5・本ファイル冒頭コメント参照）。

use crate::context::MetalContext;
use crate::gemm::MetalGemm;
use crate::gemm_reuse_phase_diag_tests::{
    MEASURED_TRIALS, WARMUP_TRIALS, gen_square_ab, measure_one_phase_trial, median_of,
};
use crate::tile::{self, FragLoadConfig, FragLoadKSteps, TileConfig};
// `MTLDevice::maxThreadgroupMemoryLength`（twin の `validate` に渡す
// デバイス上限取得。`gemm.rs::frag_load_tgp_vs_device_same_shape_bit_match`
// と同じ import）。
use objc2_metal::MTLDevice;

/// [`frag_load_kernel_gpu_ab_production_sizes`] が対象とするサイズ
/// （`docs/perf/metal-gemm-reuse-phase-1277` の Phase 2 分母表と同一。
/// `gemm_spec_source_diag_tests::SIZES` と同一値）。
const SIZES: [usize; 3] = [1024, 2048, 4096];

/// E3 候補ラベル・`FragLoadConfig`・「device 系か（twin 構成が必要か）」の
/// 対応（`docs/perf/metal-gemm-frag-load-candidates.md` §1 候補表と同じ
/// 5 候補。`tgp-k1` は本番既定 `tile::FRAG_LOAD_CONFIG` と同値）。
const CANDIDATE_LABELS: [&str; 5] = [
    "tgp-k1",
    "tgp-k2",
    "device-legacy",
    "device-hoisted-k1",
    "device-hoisted-k2",
];

const CANDIDATE_CONFIGS: [FragLoadConfig; 5] = [
    tile::FRAG_LOAD_CONFIG,
    FragLoadConfig {
        device_hoisted: false,
        ksteps: FragLoadKSteps::Two,
    },
    FragLoadConfig {
        device_hoisted: false,
        ksteps: FragLoadKSteps::One,
    },
    FragLoadConfig {
        device_hoisted: true,
        ksteps: FragLoadKSteps::One,
    },
    FragLoadConfig {
        device_hoisted: true,
        ksteps: FragLoadKSteps::Two,
    },
];

/// `CANDIDATE_LABELS`/`CANDIDATE_CONFIGS` のうち device 系（`staged=false`
/// twin を要する）候補の index（`docs/perf/metal-gemm-frag-load-
/// candidates.md` §1 候補表の `USE_TGP_STAGING=false` 行）。
const DEVICE_CANDIDATE_INDICES: [usize; 3] = [2, 3, 4];

/// AC-1/AC-2: N=1024/2048/4096 で 5 候補の `kernel_gpu`（GPU タイム
/// スタンプによる純カーネル専有時間。イシュー #1276）を trial ごとに
/// 交互（開始オフセット回転）で計測し、5 プロセス起動の 1 回分として
/// 中央値・`head_over_base_kernel_gpu`／`head_over_device_legacy_
/// kernel_gpu` 比を出力する（複数プロセス起動・集計は
/// `docs/perf/metal-gemm-n4096-kernel-gap.md` §10 の手順書側で行う）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn frag_load_kernel_gpu_ab_production_sizes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");

    // 5 候補すべての `MetalGemm` を先に構築する（`new_with_frag_load` は
    // MSL パイプラインの遅延構築のみで確保コストは小さい。ファイル冒頭
    // 「実行時は必ず `--test-threads=1`」参照）。
    let gemms: Vec<MetalGemm> = CANDIDATE_CONFIGS
        .iter()
        .map(|&cfg| {
            MetalGemm::new_with_frag_load(&ctx, cfg)
                .expect("frag_load 候補 GEMM パイプラインの構築に失敗した")
        })
        .collect();

    let max_threads_per_tg = 1024u32;
    let max_shared_mem_bytes = ctx.device().maxThreadgroupMemoryLength() as u32;

    for n in SIZES {
        // base 選択構成（`tile::select_for_device`。`dispatch_auto` と同一
        // 解決。`tgp-k1`／`tgp-k2` はこの構成をそのまま使う）。
        let base_cfg = tile::select_for_device(n, n, n, ctx.verified_m4_max_gpu_core_count());
        let device_twin = TileConfig {
            staged: false,
            ..base_cfg
        };

        // device 系候補が `validate` を通るかを事前に確認し、通らない
        // 候補は本 N では計測対象外とする（`docs/perf/metal-gemm-
        // frag-load-candidates.md` §5「twin の作り方」・
        // `gemm.rs::frag_load_tgp_vs_device_same_shape_bit_match` と同じ
        // 設計）。
        let device_twin_validates = device_twin
            .validate(max_threads_per_tg, max_shared_mem_bytes)
            .is_ok();

        // 候補 index -> 要求構成（`tgp-*` は `base_cfg`・device 系は
        // `device_twin`）。
        let requested_cfg = |idx: usize| -> TileConfig {
            if DEVICE_CANDIDATE_INDICES.contains(&idx) {
                device_twin
            } else {
                base_cfg
            }
        };

        // 実際に計測する候補 index（device 系は twin が validate を通った
        // 場合のみ含める）。
        let active_indices: Vec<usize> = (0..CANDIDATE_LABELS.len())
            .filter(|&i| !DEVICE_CANDIDATE_INDICES.contains(&i) || device_twin_validates)
            .collect();
        assert!(
            !active_indices.is_empty(),
            "N={n}: 計測対象の候補が 0 件だった（検証が空振りする）"
        );
        if !device_twin_validates {
            println!(
                "N={n}: device twin ({device_twin:?}) が validate を通らなかったため \
                 device-legacy/device-hoisted-k1/device-hoisted-k2 は本 N で計測対象外"
            );
        }

        let (a, b) = gen_square_ab(0x1295_a000 ^ (n as u64), n);
        let num_active = active_indices.len();

        // 候補ごとの keep_alive（N ごとのスコープで drop することで
        // ピークメモリを 1 N 分に抑える。ファイル冒頭「メモリ使用量」
        // 参照）。
        let mut keep_alives: Vec<Vec<Vec<f32>>> = (0..CANDIDATE_LABELS.len())
            .map(|_| Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS))
            .collect();

        // warmup: 各候補の初回 MSL パイプライン構築コストを吸収する
        // （`gemm_spec_source_diag_tests.rs` と同じ考え方）。
        for _ in 0..WARMUP_TRIALS {
            for &idx in &active_indices {
                let cfg = requested_cfg(idx);
                let _ = measure_one_phase_trial(
                    &ctx,
                    &gemms[idx],
                    &a,
                    &b,
                    n,
                    cfg,
                    &mut keep_alives[idx],
                );
            }
        }

        let mut kernel_gpu: Vec<Vec<f64>> = (0..CANDIDATE_LABELS.len())
            .map(|_| Vec::with_capacity(MEASURED_TRIALS))
            .collect();
        let mut resolved: Vec<Option<TileConfig>> = vec![None; CANDIDATE_LABELS.len()];

        for trial in 0..MEASURED_TRIALS {
            // trial index による開始オフセット回転（order-bias 相殺。
            // ファイル冒頭「候補と `TileConfig` の組み立て方」参照）。
            let offset = trial % num_active;
            for step in 0..num_active {
                let idx = active_indices[(offset + step) % num_active];
                let cfg = requested_cfg(idx);
                let sample = measure_one_phase_trial(
                    &ctx,
                    &gemms[idx],
                    &a,
                    &b,
                    n,
                    cfg,
                    &mut keep_alives[idx],
                );
                // フォールバック非経由の fail-closed 検証（ファイル冒頭
                // 「gating しない方針」参照。片側フォールバックした反復を
                // 性能比較として集計しない）。
                assert_eq!(
                    sample.resolved_cfg, cfg,
                    "N={n} trial={trial} cand={}: pipeline_for_tile フォールバックが\
                     発生した(requested={cfg:?}, resolved={:?})。性能比較の前提が\
                     崩れるため中断する",
                    CANDIDATE_LABELS[idx], sample.resolved_cfg
                );
                kernel_gpu[idx].push(sample.kernel_gpu_secs);
                resolved[idx] = Some(sample.resolved_cfg);
            }
        }

        for &idx in &active_indices {
            let q = median_of(&kernel_gpu[idx]);
            println!(
                "N={n} cand={} requested_tile={:?} resolved_tile={:?} \
                 kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
                CANDIDATE_LABELS[idx],
                requested_cfg(idx),
                resolved[idx].expect("MEASURED_TRIALS > 0 のため必ず Some"),
                q.median * 1e3,
                q.q1 * 1e3,
                q.q3 * 1e3
            );
        }

        // base（`tgp-k1`。index 0）比。`tgp-k1` は常に active
        // （device twin 非依存）なので分母は必ず存在する。
        let base_median = median_of(&kernel_gpu[0]).median;
        for &idx in &active_indices {
            let ratio = median_of(&kernel_gpu[idx]).median / base_median;
            println!(
                "N={n} cand={} head_over_base_kernel_gpu={ratio:.6}",
                CANDIDATE_LABELS[idx]
            );
        }

        // device-legacy（index 2）比: staged→device 切替の効果と
        // hoisting／ksteps の効果を分離して帰属するための副次比
        // （ファイル冒頭コメント参照。採否の一次指標にはしない）。
        if device_twin_validates {
            let device_legacy_median = median_of(&kernel_gpu[2]).median;
            for &idx in &[3usize, 4usize] {
                let ratio = median_of(&kernel_gpu[idx]).median / device_legacy_median;
                println!(
                    "N={n} cand={} head_over_device_legacy_kernel_gpu={ratio:.6}",
                    CANDIDATE_LABELS[idx]
                );
            }
        }
    }
}
