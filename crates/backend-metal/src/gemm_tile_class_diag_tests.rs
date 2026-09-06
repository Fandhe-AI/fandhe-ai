//! E6 タイルクラス分割（`tile::TileClassMode`。イシュー #1327・PR #1388
//! で opt-in 機構を追加・bit 一致を自己検証済み）の 2 クラス経路
//! （`TileClassMode::Split`）と現行経路（`TileClassMode::Legacy`）の
//! N=1024/2048/4096 純カーネル時間（GPU タイムスタンプ。イシュー #1276
//! の `kernel_gpu` 変種）を候補 0/4/5/8（`tile::CANDIDATES`）で M4 Max
//! で 5 回計測し、有効性（本番結線可否）を判定する診断テスト
//! （イシュー #1328）。
//!
//! # AC（受け入れ基準）形状での構造的縮退（事前に明記する作業仮説）
//!
//! `tile::CANDIDATES[0]`（64,64,16,2,2）・`[4]`（64,64,16,1,2）・`[5]`
//! （64,32,32,2,2）・`[8]`（32,64,16,1,2）はいずれも `staged: true` で、
//! N=1024/2048/4096 は各候補の `bm`/`bn`/`bk` すべての倍数（`bk` が最大
//! でも 32、N が最小でも 1024 のため必ず割り切れる）。よって
//! `tile::tile_class_plan` は interior＝grid 全体・端ストリップ 2 本とも
//! 空を返し、`TileClassMode::Split` は「**Interior クラス（`tile::
//! TileClassMode` ドキュメンテーションコメント参照。direct-load 強制）を
//! grid 全体へ適用した 1 dispatch**＋領域ガード」へ縮退する
//! （`TILE_CLASS_EDGE_DISPATCH_COUNT`／`TILE_CLASS_INTERIOR_DISPATCH_
//! COUNT` の増分で本テストが可観測にする。実測では edge 増分は常に 0・
//! interior 増分のみ発生することを確認済み）。これは
//! `docs/perf/metal-gemm-transpose-tiled.md`（E3）の `device-legacy`
//! twin（`staged=false`）と実質同一の direct-load 経路であり、E3 では
//! 全 N で staged 比 1.6〜3.3 倍**遅かった**ため、当初はこの構造的縮退が
//! 「後退（REJECT）」に直結すると予想していた。しかし実測（本ファイル
//! 冒頭 doc の想定に反して）は候補 0/4/8 で Split（direct-load 縮退）が
//! Legacy（staged）より速い場面が多く、候補 5（`bk=32`）のみ逆に遅いという、
//! E3 単体の傾向とは異なる結果が出た（形状・候補依存で単純に予測できない。
//! 詳細な数値は `docs/perf/metal-gemm-n4096-kernel-gap.md` §12 を参照）。
//! **採否の最終判断は同 §12.4 で人間が行う。本ファイル自体は gating しない**
//! （大小関係への `assert!` は行わず、フォールバック非経由・bit 一致
//! のみを fail-closed に検証する）。
//!
//! 「現行経路」は本番 `select_for_device` の選択構成（M4 Max では
//! N=1024→`CANDIDATES[6]`・2048→`[1]`・4096→`[2]`）ではなく、**同一候補
//! の `TileClassMode::Legacy`（1 dispatch）** を指す（候補 0/4/5/8 は
//! いずれも本番選択構成ではないため）。
//!
//! # 配置理由（`gemm_coop_load_diag_tests.rs` と同じ判断）
//!
//! `gemm::MetalGemm::new_with_tile_class`（`pub`。イシュー #1327）・
//! `gemm::MetalGemm::tile_class_mode`（`#[cfg(test)] pub(crate)`）・
//! `gemm_reuse_phase_diag_tests::{measure_one_phase_trial, WARMUP_TRIALS,
//! MEASURED_TRIALS, gen_square_ab, median_of}`（いずれも `pub(crate)`）・
//! `tile::{CANDIDATES, TileClassMode, tile_class_plan}`（`pub(crate)`）へ
//! 到達するため、integration test ではなく `lib.rs` の兄弟モジュールと
//! して配置する。
//!
//! # 実行時は必ず `--test-threads=1`
//!
//! 同一プロセス内に base（`Legacy`）/head（`Split`）2 つの `MetalGemm`
//! （`MetalContext::new()` 専用コンテキスト 1 個を共有）を構築するが、
//! GPU 上での複数テストスレッド競合を避けるため `gemm_coop_load_diag_
//! tests.rs` と同じ理由で `--test-threads=1` を前提とする。
//!
//! # メモリ使用量
//!
//! ホスト側キープアライブ（`Vec<Vec<f32>>`）は N=4096・2 arm（base/head）
//! 同時保持で (20 warmup + 20 測定) × 4096² × 4 bytes × 2 arm ≈ 5.4 GiB
//! （統合メモリ上。本機 64 GiB のため truncate 不要）。候補ごとのループ
//! スコープで両 arm の `keep_alive` を drop してから次の候補へ進むことで
//! ピークを 1 候補分に抑える（4 候補同時保持だと約 21 GiB になるため
//! 避ける。ファイル冒頭「配置理由」上の設計判断）。
//!
//! # gating しない方針（`gemm_coop_load_diag_tests.rs` と同じ理由）
//!
//! 実行が成功すること（例外なく完了すること）のみを検証条件とし、
//! `kernel_gpu` の大小関係への `assert!` は行わない（有効性判断は
//! ドキュメント側〈`docs/perf/metal-gemm-n4096-kernel-gap.md` §12〉で
//! 人間が行う。環境揺らぎによる flaky 化防止）。ただし以下は fail-closed
//! に検証する:
//!
//! - `resolved_cfg == cfg`（フォールバック非経由。片側だけ別構成へ
//!   フォールバックした反復を性能比較に混入させない）
//! - `TILE_CLASS_SPLIT_FALLBACK_COUNT` の増分が 0（head 側で Edge/Interior
//!   解決構成が食い違う fail-closed フォールバックが発生していない）
//! - head 側 trial 0 の出力が base 側 trial 0 の出力と bit 完全一致
//!   （同一入力 `gen_square_ab(seed ^ n)` を使う安価な正確性検査。PR
//!   #1388 の T1〜T3 自己検証と独立に、本ハーネス自身の経路でも
//!   Legacy/Split が同一結果を生むことを確認する）
//!
//! # プロダクションコード不変
//!
//! 本ファイルは `tile.rs`／`shaders/gemm.metal` への変更を一切含まない
//! （`gemm.rs` は本イシューで `encode_tiled_by_class` を挙動不変の
//! `plan_tiled_by_class`／`encode_tiled_plan` へリファクタしたのみ）。
//! `tile::TILE_CLASS_MODE`（本番既定 `Legacy`）・`tile::select` の候補表
//! への組み込みは本ファイルの実測結果を受けて別途 docs 側で判断する
//! （`docs/perf/metal-gemm-n4096-kernel-gap.md` §12.4）。

use crate::context::MetalContext;
use crate::gemm::MetalGemm;
use crate::gemm_reuse_phase_diag_tests::{
    MEASURED_TRIALS, WARMUP_TRIALS, gen_square_ab, measure_one_phase_trial, median_of,
};
use crate::tile::{self, TileClassMode};
// `MTLDevice::maxThreadgroupMemoryLength`（候補ごとの共有メモリ事前
// フィルタに渡すデバイス上限取得。`gemm_coop_load_diag_tests.rs` と同じ
// import）。
use objc2_metal::MTLDevice;

/// [`tile_class_kernel_gpu_ab_production_sizes`] が対象とするサイズ
/// （`docs/perf/metal-gemm-reuse-phase-1277` の Phase 2 分母表と同一。
/// `gemm_coop_load_diag_tests::SIZES` と同一値）。
const SIZES: [usize; 3] = [1024, 2048, 4096];

/// イシュー #1328 が対象とする `tile::CANDIDATES` の添字（ファイル冒頭
/// 「AC 形状での構造的縮退」参照。候補ラベルは `cand<index>`）。
const CANDIDATE_INDICES: [usize; 4] = [0, 4, 5, 8];

/// `MTLComputePipelineState::maxTotalThreadsPerThreadgroup` は構築後にしか
/// 取得できないため、`gemm::MetalGemm::pipeline_for_tile` と同じ Apple GPU
/// 一般上限（1024）を事前検証に使う（`gemm.rs::pipeline_for_tile` ドキュ
/// メンテーションコメント参照）。
const MAX_THREADS_PER_TG_ESTIMATE: u32 = 1024;

/// AC-(a): N=1024/2048/4096 × 候補 0/4/5/8 で base（`TileClassMode::
/// Legacy`）/head（`TileClassMode::Split`）の `kernel_gpu`（GPU タイム
/// スタンプによる純カーネル専有時間）を trial ごとに交互（開始オフセット
/// 回転）で計測し、5 プロセス起動の 1 回分として中央値・
/// `split_over_legacy_kernel_gpu` 比を出力する（複数プロセス起動・集計は
/// `docs/perf/metal-gemm-n4096-kernel-gap.md` §12 の手順書側で行う）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tile_class_kernel_gpu_ab_production_sizes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");

    let base_gemm = MetalGemm::new_with_tile_class(&ctx, TileClassMode::Legacy)
        .expect("base（Legacy）GEMM パイプラインの構築に失敗した");
    let head_gemm = MetalGemm::new_with_tile_class(&ctx, TileClassMode::Split)
        .expect("head（Split）GEMM パイプラインの構築に失敗した");
    assert_eq!(base_gemm.tile_class_mode(), TileClassMode::Legacy);
    assert_eq!(head_gemm.tile_class_mode(), TileClassMode::Split);

    let max_shared_mem_bytes = ctx.device().maxThreadgroupMemoryLength() as u32;

    for n in SIZES {
        // 候補ごとの共有メモリ事前フィルタ（`pipeline_for_tile` が
        // フォールバックする候補を当該 N・当該デバイスでは対象外とする。
        // `gemm_coop_load_diag_tests.rs` と同じ設計）。
        let active_indices: Vec<usize> = CANDIDATE_INDICES
            .iter()
            .copied()
            .filter(|&i| {
                tile::CANDIDATES[i]
                    .validate(MAX_THREADS_PER_TG_ESTIMATE, max_shared_mem_bytes)
                    .is_ok()
            })
            .collect();
        for &i in &CANDIDATE_INDICES {
            if !active_indices.contains(&i) {
                println!(
                    "N={n}: cand{i} は共有メモリ／スレッド数事前フィルタで対象外\
                     （TileConfig::validate 失敗）"
                );
            }
        }
        if active_indices.is_empty() {
            println!("N={n}: 対象候補が全て事前フィルタで除外されたためスキップ");
            continue;
        }

        let (a, b) = gen_square_ab(0x1328_a000 ^ (n as u64), n);

        // 候補ごとの keep_alive（候補ごとのスコープで drop することで
        // ピークメモリを 1 候補分に抑える。ファイル冒頭「メモリ使用量」
        // 参照）。
        for &idx in &active_indices {
            let cfg = tile::CANDIDATES[idx];

            // AC 形状では Split が「Interior 1 回 dispatch」へ縮退する
            // ことをここで確認する（ファイル冒頭「AC 形状での構造的
            // 縮退」参照）。空振り検知のため、計測開始前に increment 前
            // カウンタを記録する。
            let interior_before = crate::gemm::TILE_CLASS_INTERIOR_DISPATCH_COUNT.with(|c| c.get());
            let edge_before = crate::gemm::TILE_CLASS_EDGE_DISPATCH_COUNT.with(|c| c.get());
            let fallback_before = crate::gemm::TILE_CLASS_SPLIT_FALLBACK_COUNT.with(|c| c.get());

            let mut keep_alive_base: Vec<Vec<f32>> =
                Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);
            let mut keep_alive_head: Vec<Vec<f32>> =
                Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);

            // warmup: 各 arm の初回 MSL パイプライン構築コストを吸収する。
            for _ in 0..WARMUP_TRIALS {
                let _ =
                    measure_one_phase_trial(&ctx, &base_gemm, &a, &b, n, cfg, &mut keep_alive_base);
                let _ =
                    measure_one_phase_trial(&ctx, &head_gemm, &a, &b, n, cfg, &mut keep_alive_head);
            }

            let mut kernel_gpu_base: Vec<f64> = Vec::with_capacity(MEASURED_TRIALS);
            let mut kernel_gpu_head: Vec<f64> = Vec::with_capacity(MEASURED_TRIALS);
            let mut first_output_base: Option<Vec<f32>> = None;
            let mut first_output_head: Option<Vec<f32>> = None;

            for trial in 0..MEASURED_TRIALS {
                // trial index による開始オフセット回転（order-bias 相殺。
                // `num_active` ではなく 2 arm 間の交互のみだが、`offset`
                // で base/head どちらを先に計測するかを反転させる）。
                let base_first = trial % 2 == 0;
                let (sample_base, sample_head) = if base_first {
                    let sb = measure_one_phase_trial(
                        &ctx,
                        &base_gemm,
                        &a,
                        &b,
                        n,
                        cfg,
                        &mut keep_alive_base,
                    );
                    let sh = measure_one_phase_trial(
                        &ctx,
                        &head_gemm,
                        &a,
                        &b,
                        n,
                        cfg,
                        &mut keep_alive_head,
                    );
                    (sb, sh)
                } else {
                    let sh = measure_one_phase_trial(
                        &ctx,
                        &head_gemm,
                        &a,
                        &b,
                        n,
                        cfg,
                        &mut keep_alive_head,
                    );
                    let sb = measure_one_phase_trial(
                        &ctx,
                        &base_gemm,
                        &a,
                        &b,
                        n,
                        cfg,
                        &mut keep_alive_base,
                    );
                    (sb, sh)
                };

                // フォールバック非経由の fail-closed 検証（ファイル冒頭
                // 「gating しない方針」参照。片側だけフォールバックした
                // 反復を性能比較として集計しない）。
                assert_eq!(
                    sample_base.resolved_cfg, cfg,
                    "N={n} trial={trial} cand{idx} base（Legacy）: pipeline_for_tile \
                     フォールバックが発生した(requested={cfg:?}, \
                     resolved={:?})。性能比較の前提が崩れるため中断する",
                    sample_base.resolved_cfg
                );
                assert_eq!(
                    sample_head.resolved_cfg, cfg,
                    "N={n} trial={trial} cand{idx} head（Split）: pipeline_for_tile \
                     フォールバックが発生した(requested={cfg:?}, \
                     resolved={:?})。性能比較の前提が崩れるため中断する",
                    sample_head.resolved_cfg
                );

                kernel_gpu_base.push(sample_base.kernel_gpu_secs);
                kernel_gpu_head.push(sample_head.kernel_gpu_secs);

                if trial == 0 {
                    // trial 0 の出力を退避し、末尾で base/head の bit 一致を
                    // 検証する（ファイル冒頭「gating しない方針」参照）。
                    // `keep_alive_{base,head}` は `measure_one_phase_trial`
                    // が `copied`（出力の複製）を push した直後なので、
                    // 直前に push された要素（trial 0 分の warmup 後、
                    // measured の先頭）を参照する。
                    first_output_base = keep_alive_base.last().cloned();
                    first_output_head = keep_alive_head.last().cloned();
                }
            }

            // head 側で Edge/Interior 解決構成が食い違う fail-closed
            // フォールバックが発生していないことを確認する（ファイル冒頭
            // 「gating しない方針」参照）。
            let fallback_after = crate::gemm::TILE_CLASS_SPLIT_FALLBACK_COUNT.with(|c| c.get());
            assert_eq!(
                fallback_after, fallback_before,
                "N={n} cand{idx}: TILE_CLASS_SPLIT_FALLBACK_COUNT が増加した\
                 （head 側で Edge/Interior 解決構成が食い違い Legacy 単一 \
                 dispatch へフォールバックした）。性能比較の前提が崩れるため \
                 中断する"
            );

            // head 側が空振り（Legacy と同一の 1 dispatch のみ）に
            // なっていないかを確認する（`new_with_tile_class(Split)` が
            // 正しく尊重されているかの検査。`plan_tiled_by_class` が
            // `tile_class_mode` を見ずに常に Legacy 分岐へ落ちるリグレッ
            // ションを検出する）。
            let interior_after = crate::gemm::TILE_CLASS_INTERIOR_DISPATCH_COUNT.with(|c| c.get());
            let edge_after = crate::gemm::TILE_CLASS_EDGE_DISPATCH_COUNT.with(|c| c.get());
            let interior_increment = interior_after - interior_before;
            let edge_increment = edge_after - edge_before;
            assert!(
                interior_increment > 0 || edge_increment > 0,
                "N={n} cand{idx}: head（Split）側で TILE_CLASS_INTERIOR_\
                 DISPATCH_COUNT／TILE_CLASS_EDGE_DISPATCH_COUNT のいずれも \
                 増加しなかった（Split 分岐が呼ばれていない疑いがある）"
            );
            // AC 形状での構造的縮退（ファイル冒頭コメント参照）: interior
            // は常に 0 増分・edge のみ増分するはず。値自体は assert せず
            // 記録する（gating しない方針）。
            println!(
                "N={n} cand{idx} tile_class_edge_dispatch_increment={edge_increment} \
                 tile_class_interior_dispatch_increment={interior_increment}"
            );

            // 正確性の安価な fail-closed 検査: trial 0 の base/head 出力
            // を bit 完全一致で比較する（ファイル冒頭「gating しない方針」
            // 参照）。
            let out_base = first_output_base
                .expect("MEASURED_TRIALS > 0 のため trial 0 の base 出力は必ず Some");
            let out_head = first_output_head
                .expect("MEASURED_TRIALS > 0 のため trial 0 の head 出力は必ず Some");
            assert_eq!(
                out_base.len(),
                out_head.len(),
                "N={n} cand{idx}: base/head の出力長が一致しない"
            );
            let mismatch = out_base
                .iter()
                .zip(out_head.iter())
                .enumerate()
                .find(|(_, (b, h))| b.to_bits() != h.to_bits());
            assert!(
                mismatch.is_none(),
                "N={n} cand{idx}: base（Legacy）/head（Split）の出力が bit 一致しない \
                 (最初の不一致 index={:?})",
                mismatch.map(|(i, _)| i)
            );

            let q_base = median_of(&kernel_gpu_base);
            let q_head = median_of(&kernel_gpu_head);
            println!(
                "N={n} cand{idx} mode=legacy resolved_tile={cfg:?} \
                 kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
                q_base.median * 1e3,
                q_base.q1 * 1e3,
                q_base.q3 * 1e3
            );
            println!(
                "N={n} cand{idx} mode=split resolved_tile={cfg:?} \
                 kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
                q_head.median * 1e3,
                q_head.q1 * 1e3,
                q_head.q3 * 1e3
            );
            let ratio = q_head.median / q_base.median;
            println!("N={n} cand{idx} split_over_legacy_kernel_gpu={ratio:.6}");
        }
    }
}

/// AC-(a) 追補: 候補 0/4/5/8 の実測結果が候補依存で符号が割れた
/// （ファイル冒頭「AC 形状での構造的縮退」参照。cand0/4/8 は改善方向・
/// cand5 は後退方向）ため、**本番 `dispatch_auto` が実際に選択する構成**
/// （`tile::select_for_device`。M4 Max では N=512→`CANDIDATES[3]`・
/// 1024→`[6]`・2048→`[1]`・4096→`[2]`。いずれも候補 0/4/5/8 とは異なる）
/// で base（Legacy）/head（Split）を直接比較し、`tile::TILE_CLASS_MODE`
/// を実際に切り替えた場合の効果を検証する（イシュー #1328 の計画
/// 「§3.4 採否・本番結線の判断規則」が要求する「ADOPT 相当の場合の
/// 追加 A/B」に相当。候補 0/4/5/8 の結果だけでは本番構成への外挿が
/// できないため必須）。
///
/// `gemm_reuse_phase_diag_tests::run_size_with`（`pub(crate)`。イシュー
/// #1289 で base/head 比較用に汎用化済み）が `select_for_device` で
/// 構成を解決し `kernel_gpu` を含む全フェーズを `println!` する。
/// 大小関係への `assert!` は行わない（ファイル冒頭「gating しない方針」
/// と同じ理由）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tile_class_production_select_kernel_gpu_ab() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");

    let base_gemm = MetalGemm::new_with_tile_class(&ctx, TileClassMode::Legacy)
        .expect("base（Legacy）GEMM パイプラインの構築に失敗した");
    let head_gemm = MetalGemm::new_with_tile_class(&ctx, TileClassMode::Split)
        .expect("head（Split）GEMM パイプラインの構築に失敗した");

    // `dispatch_auto` の実測帯域（512/1024/2048/4096。`select_for_device`
    // ドキュメンテーションコメント参照）を網羅する。
    for n in [512usize, 1024, 2048, 4096] {
        crate::gemm_reuse_phase_diag_tests::run_size_with(&ctx, &base_gemm, n, "legacy");
        crate::gemm_reuse_phase_diag_tests::run_size_with(&ctx, &head_gemm, n, "split");
    }
}
