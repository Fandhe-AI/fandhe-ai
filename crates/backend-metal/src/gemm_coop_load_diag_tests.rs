//! E4 協調ロードレイアウト候補（`tile::CoopLoadConfig`。イシュー #1298
//! で実装・bit 一致を自己検証済み）の N=1024/2048/4096 純カーネル時間
//! （GPU タイムスタンプ。イシュー #1276 の `kernel_gpu` 変種）を M4 Max
//! で 5 回計測し、有効性（`tile::select` への組み込み可否・採用候補）を
//! 判定する診断テスト（イシュー #1300）。
//!
//! `docs/perf/metal-gemm-coop-load-candidates.md` §5「#1300 への
//! 引き継ぎ」の指示に従い、以下 1 点を本ファイルの入口とする:
//!
//! - [`coop_load_kernel_gpu_ab_production_sizes`][]: N=1024/2048/4096 で
//!   6 候補（`L0-P4`〈= 本番既定 `tile::COOP_LOAD_CONFIG`。
//!   `layout=RowLinear, pad=Four`〉・`L0-P0`・`L0-P8`・`L1-P0`・`L1-P4`・
//!   `L1-P8`。`gemm::tests::coop_load_required_heads` と同一の 5 head +
//!   base）の `MetalGemm` インスタンスへ `gemm_reuse_phase_diag_tests::
//!   measure_one_phase_trial`（`pub(crate)`）を trial ごとに交互
//!   （trial index による開始オフセット回転）で呼び、`kernel_gpu_secs`
//!   の中央値・base（`L0-P4`）比を出力する。
//!
//! # 候補と `TileConfig` の組み立て方（`gemm_frag_load_diag_tests.rs`
//! との差分）
//!
//! E3（`FragLoadConfig`）と異なり、E4 の 6 候補はいずれも同一の
//! `base_cfg = tile::select_for_device(...)`（`staged=true` の本番選択
//! 構成。`dispatch_auto` と同一解決）をそのまま使う（`{staged: false}`
//! twin は不要）。`CoopLoadConfig::pad_elems` が `!cfg.staged` の場合に
//! 常に `0` を返す契約（`tile.rs::CoopLoadConfig::pad_elems` doc
//! comment）のため、`staged=false` では P0/P4/P8 が同一カーネルへ縮退し
//! 比較が空振りするので、`base_cfg.staged` を fail-closed に assert する。
//!
//! かわりに E4 固有の懸念は「`TGP_PAD=8` が共有メモリ使用量を増やし、
//! `pipeline_for_tile` の事前検証（`shared_mem_bytes_for_pad` が
//! デバイス上限を超える場合の fallback chain 遷移）で base と異なる
//! `resolved_cfg` に落ちうる」点にある。本ファイルは候補ごとに
//! `base_cfg.shared_mem_bytes_for_pad(NN, cfg.pad_elems(base_cfg))` が
//! デバイス上限（`MTLDevice::maxThreadgroupMemoryLength`）以下かを
//! 事前フィルタし、超える候補は当該 N で計測対象外とする（`gemm_frag_
//! load_diag_tests.rs` の device twin `validate` 事前フィルタと同型の
//! 設計だが、E4 は `TileConfig` 自体を差し替えず同一 `base_cfg` の
//! 共有メモリ見積りだけを候補ごとに変えて判定する点が異なる）。
//! 事前フィルタを通過した active 候補には、E3 と同じく `resolved_cfg ==
//! requested_cfg`（= `base_cfg`）の fail-closed assert を維持する
//! （片側だけフォールバックした反復を性能比較に混入させないため）。
//!
//! # 配置理由（`gemm_frag_load_diag_tests.rs` と同じ判断）
//!
//! `gemm::MetalGemm::new_with_coop_load`（イシュー #1298 で追加した
//! `pub` 面）・`gemm_reuse_phase_diag_tests::{measure_one_phase_trial,
//! WARMUP_TRIALS, MEASURED_TRIALS, gen_square_ab, median_of}`（いずれも
//! `pub(crate)`）・`tile::CoopLoadConfig::pad_elems`／
//! `TileConfig::shared_mem_bytes_for_pad`（いずれも `pub(crate)`）へ
//! 到達するため、integration test ではなく `lib.rs` の兄弟モジュールと
//! して配置する。
//!
//! # 実行時は必ず `--test-threads=1`
//!
//! 同一プロセス内に 6 つの `MetalGemm`（いずれも `MetalContext::new()`。
//! `cached_context()` ではない専用コンテキスト 1 個を共有）を構築するが、
//! GPU 上での複数テストスレッド競合を避けるため `gemm_frag_load_diag_
//! tests.rs` と同じ理由で `--test-threads=1` を前提とする。
//!
//! # メモリ使用量
//!
//! ホスト側キープアライブ（`Vec<Vec<f32>>`）は N=4096・6 候補同時保持で
//! (20 warmup + 20 測定) × 4096² × 4 bytes × 6 候補 ≈ 16.1 GiB
//! （統合メモリ上。本機 64 GiB のため truncate 不要）。N ごとのループ
//! スコープで各候補の `keep_alive` を drop してから次の N へ進むことで
//! ピークを 1 N 分に抑える（`gemm_frag_load_diag_tests.rs` と同じ考え方）。
//!
//! # gating しない方針（`gemm_frag_load_diag_tests.rs` と同じ理由）
//!
//! 実行が成功すること（例外なく完了すること）のみを検証条件とし、
//! `kernel_gpu` の大小関係への `assert!` は行わない（E4 の有効性判断は
//! ドキュメント側〈`docs/perf/metal-gemm-n4096-kernel-gap.md` §11〉で
//! 人間が行う。環境揺らぎによる flaky 化防止）。`resolved_cfg ==
//! requested_cfg`（フォールバック非経由）のみを fail-closed に assert
//! する。
//!
//! # プロダクションコード不変
//!
//! 本ファイルは `gemm.rs`／`tile.rs`／`shaders/gemm.metal` への変更を
//! 一切含まない（イシュー #1298 が既に追加済みの `#[cfg(test)] pub`／
//! `pub(crate)` 面のみを利用する）。`tile::select` の候補表・本番既定
//! （`MetalGemm::new` の `tile::COOP_LOAD_CONFIG`）への組み込みは本
//! イシューのスコープ外（兄弟イシュー #1302／#1304 が担う。
//! `docs/perf/metal-gemm-coop-load-candidates.md` §5・本ファイル冒頭
//! コメント参照）。

use crate::context::MetalContext;
use crate::gemm::MetalGemm;
use crate::gemm_reuse_phase_diag_tests::{
    MEASURED_TRIALS, WARMUP_TRIALS, gen_square_ab, measure_one_phase_trial, median_of,
};
use crate::layout::TransposePattern;
use crate::tile::{self, CoopLoadConfig, CoopLoadLayout, TgpPad};
// `MTLDevice::maxThreadgroupMemoryLength`（共有メモリ事前フィルタに渡す
// デバイス上限取得。`gemm_frag_load_diag_tests.rs` と同じ import）。
use objc2_metal::MTLDevice;

/// [`coop_load_kernel_gpu_ab_production_sizes`] が対象とするサイズ
/// （`docs/perf/metal-gemm-reuse-phase-1277` の Phase 2 分母表と同一。
/// `gemm_frag_load_diag_tests::SIZES` と同一値）。
const SIZES: [usize; 3] = [1024, 2048, 4096];

/// E4 候補ラベル・`CoopLoadConfig` の対応（`docs/perf/
/// metal-gemm-coop-load-candidates.md` §1 候補表と同じ 6 候補。index 0
/// = base = `tile::COOP_LOAD_CONFIG`〈= `L0-P4`〉。index 1〜5 は
/// `gemm::tests::coop_load_required_heads` と同一値・同一順序）。
const CANDIDATE_LABELS: [&str; 6] = ["L0-P4", "L0-P0", "L0-P8", "L1-P0", "L1-P4", "L1-P8"];

const CANDIDATE_CONFIGS: [CoopLoadConfig; 6] = [
    tile::COOP_LOAD_CONFIG,
    CoopLoadConfig {
        layout: CoopLoadLayout::RowLinear,
        pad: TgpPad::Zero,
    },
    CoopLoadConfig {
        layout: CoopLoadLayout::RowLinear,
        pad: TgpPad::Eight,
    },
    CoopLoadConfig {
        layout: CoopLoadLayout::RowStrided,
        pad: TgpPad::Zero,
    },
    CoopLoadConfig {
        layout: CoopLoadLayout::RowStrided,
        pad: TgpPad::Four,
    },
    CoopLoadConfig {
        layout: CoopLoadLayout::RowStrided,
        pad: TgpPad::Eight,
    },
];

/// AC-1/AC-2: N=1024/2048/4096 で 6 候補の `kernel_gpu`（GPU タイム
/// スタンプによる純カーネル専有時間。イシュー #1276）を trial ごとに
/// 交互（開始オフセット回転）で計測し、5 プロセス起動の 1 回分として
/// 中央値・`head_over_base_kernel_gpu` 比を出力する（複数プロセス起動・
/// 集計は `docs/perf/metal-gemm-n4096-kernel-gap.md` §11 の手順書側で
/// 行う）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn coop_load_kernel_gpu_ab_production_sizes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");

    // 6 候補すべての `MetalGemm` を先に構築する（`new_with_coop_load` は
    // MSL パイプラインの遅延構築のみで確保コストは小さい。ファイル冒頭
    // 「実行時は必ず `--test-threads=1`」参照）。
    let gemms: Vec<MetalGemm> = CANDIDATE_CONFIGS
        .iter()
        .map(|&cfg| {
            MetalGemm::new_with_coop_load(&ctx, cfg)
                .expect("coop_load 候補 GEMM パイプラインの構築に失敗した")
        })
        .collect();
    assert_eq!(
        gemms[0].coop_load(),
        tile::COOP_LOAD_CONFIG,
        "index 0（L0-P4）は本番既定と一致するはず"
    );

    let max_shared_mem_bytes = ctx.device().maxThreadgroupMemoryLength() as u32;

    for n in SIZES {
        // 全候補共通の選択構成（`tile::select_for_device`。`dispatch_auto`
        // と同一解決）。E4 は `TileConfig` 自体を差し替えないため、E3の
        // ような device twin は不要（ファイル冒頭「候補と `TileConfig`
        // の組み立て方」参照）。
        let base_cfg = tile::select_for_device(n, n, n, ctx.verified_m4_max_gpu_core_count());
        assert!(
            base_cfg.staged,
            "N={n}: 選択構成が staged=false だった。CoopLoadConfig::pad_elems は \
             !staged で常に 0 を返すため P0/P4/P8 が同一カーネルへ縮退し比較が \
             空振りする（ファイル冒頭コメント参照）"
        );

        // 候補ごとの共有メモリ事前フィルタ（`TGP_PAD=8` がデバイス上限を
        // 超える場合、`pipeline_for_tile` が base と異なる `resolved_cfg`
        // へフォールバックしうるため、事前に除外する。ファイル冒頭
        // コメント参照）。
        let active_indices: Vec<usize> = (0..CANDIDATE_LABELS.len())
            .filter(|&i| {
                let cfg = CANDIDATE_CONFIGS[i];
                let pad_elems = cfg.pad_elems(base_cfg);
                base_cfg.shared_mem_bytes_for_pad(TransposePattern::Nn, pad_elems)
                    <= max_shared_mem_bytes
            })
            .collect();
        assert!(
            active_indices.contains(&0),
            "N={n}: base（L0-P4）自体が共有メモリ事前フィルタで除外された。\
             本番既定が動作しない環境のため計測を中断する"
        );
        for (i, label) in CANDIDATE_LABELS.iter().enumerate() {
            if !active_indices.contains(&i) {
                println!(
                    "N={n}: cand={label} は共有メモリ事前フィルタで対象外\
                     （shared_mem_bytes_for_pad > maxThreadgroupMemoryLength）"
                );
            }
        }

        let (a, b) = gen_square_ab(0x1300_a000 ^ (n as u64), n);
        let num_active = active_indices.len();

        // 候補ごとの keep_alive（N ごとのスコープで drop することで
        // ピークメモリを 1 N 分に抑える。ファイル冒頭「メモリ使用量」
        // 参照）。
        let mut keep_alives: Vec<Vec<Vec<f32>>> = (0..CANDIDATE_LABELS.len())
            .map(|_| Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS))
            .collect();

        // warmup: 各候補の初回 MSL パイプライン構築コストを吸収する
        // （`gemm_frag_load_diag_tests.rs` と同じ考え方）。
        for _ in 0..WARMUP_TRIALS {
            for &idx in &active_indices {
                let _ = measure_one_phase_trial(
                    &ctx,
                    &gemms[idx],
                    &a,
                    &b,
                    n,
                    base_cfg,
                    &mut keep_alives[idx],
                );
            }
        }

        let mut kernel_gpu: Vec<Vec<f64>> = (0..CANDIDATE_LABELS.len())
            .map(|_| Vec::with_capacity(MEASURED_TRIALS))
            .collect();
        let mut resolved: Vec<Option<tile::TileConfig>> = vec![None; CANDIDATE_LABELS.len()];

        for trial in 0..MEASURED_TRIALS {
            // trial index による開始オフセット回転（order-bias 相殺。
            // ファイル冒頭「候補と `TileConfig` の組み立て方」参照）。
            let offset = trial % num_active;
            for step in 0..num_active {
                let idx = active_indices[(offset + step) % num_active];
                let sample = measure_one_phase_trial(
                    &ctx,
                    &gemms[idx],
                    &a,
                    &b,
                    n,
                    base_cfg,
                    &mut keep_alives[idx],
                );
                // フォールバック非経由の fail-closed 検証（ファイル冒頭
                // 「gating しない方針」参照。片側フォールバックした反復を
                // 性能比較として集計しない）。
                assert_eq!(
                    sample.resolved_cfg, base_cfg,
                    "N={n} trial={trial} cand={}: pipeline_for_tile フォールバックが\
                     発生した(requested={base_cfg:?}, resolved={:?})。性能比較の前提が\
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
                "N={n} cand={} coop_load={:?} resolved_tile={:?} \
                 kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
                CANDIDATE_LABELS[idx],
                CANDIDATE_CONFIGS[idx],
                resolved[idx].expect("MEASURED_TRIALS > 0 のため必ず Some"),
                q.median * 1e3,
                q.q1 * 1e3,
                q.q3 * 1e3
            );
        }

        // base（`L0-P4`。index 0）比。base は共有メモリ事前フィルタで
        // 必ず active（上記 assert 参照）なので分母は必ず存在する。
        let base_median = median_of(&kernel_gpu[0]).median;
        for &idx in &active_indices {
            let ratio = median_of(&kernel_gpu[idx]).median / base_median;
            println!(
                "N={n} cand={} head_over_base_kernel_gpu={ratio:.6}",
                CANDIDATE_LABELS[idx]
            );
        }
    }
}
