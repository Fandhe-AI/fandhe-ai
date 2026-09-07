//! hfrag 候補（half フラグメント／f32 累算。イシュー #1369・親 #1368 E9）
//! の純カーネル専有時間（GPU タイムスタンプ。`kernel_gpu`。イシュー
//! #1276）を M4 Max で 5 回中央値比較し、opt-in 候補として前進させる
//! 価値があるかを判定する診断テスト群（イシュー #1370。兄弟 E7/E8
//! （`gemm_bk32_diag_tests.rs`／`gemm_bm128_diag_tests.rs`）と同型の
//! 構成）。
//!
//! # 系列構成（`docs/perf/metal-gemm-hfrag-candidate.md` §9 参照）
//!
//! - **S（候補スイープ）**: hfrag × 全 staged `tile::CANDIDATES[0..=10]`
//!   を N=1024/2048/4096 で単独計測し、N ごとの hfrag 最良タイルを
//!   確定する（`select_for_device` の構成は f32 向けにチューニングされて
//!   おり、hfrag の SMEM 使用量は f32 版のちょうど半分〈`tile::
//!   TileConfig::shared_mem_bytes_hfrag_for`〉のため、hfrag に最適な
//!   タイルが f32 と同じとは限らない）。
//! - **A（同一タイル比較）**: `tile::select_for_device` が選ぶ構成を
//!   base（f32）/head（hfrag）両 arm に使い、N=512/1024/2048/4096 で
//!   「タイル形状を揃えた純粋な half MMA 効果」を直接比較する。
//! - **B（結線判断の根拠）**: head を S 系列で確定した N 別最良タイル、
//!   base を本番選択構成（`select_for_device`）とし、N=512/1024/2048/
//!   4096 で採否判断の主根拠とする（`docs/perf/metal-gemm-hfrag-
//!   candidate.md` §2.4 のとおり、本番結線自体は opt-in 公開 API 設計＋
//!   ユーザー承認が前提。本イシューの「採否」は「opt-in 候補として前進
//!   させる価値があるか」の判定に限る）。
//!
//! # 入力の丸め（hfrag 側の複合判定を成立させるための前提）
//!
//! `gemm_simdgroup_tiled_hfrag` は協調ロード時に f32→half 変換するため、
//! 丸めなし f32 参照に対しては統一複合判定（相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満）が一般に成立しない（`tests/gemm_hfrag_parity.rs`・
//! `docs/perf/metal-gemm-hfrag-candidate.md` §2.4/§4 で実測済み）。本
//! ファイルは A/B 両系列で `round_to_half_representable` により入力を
//! half 表現可能な f32 へ事前丸めしてから両 arm（f32／hfrag）へ同一
//! 入力として渡す。half への変換自体は恒等写像（丸め済み値は f32→half
//! で値が変わらない）で積は f32 で厳密なため、f32 経路との差は K 方向
//! 加算順序のみとなり複合判定は pass する設計（S 系列は hfrag 単独計測
//! のため正確性検査を行わない — 単独計測に「比較先」が存在しないため）。
//! この丸めは計測時間に対しては中立（同一データサイズ・同一メモリ
//! アクセスパターンのため、値がどう丸められているかは kernel_gpu に
//! 影響しない）。
//!
//! # 配置理由（E7/E8 と同じ判断）
//!
//! `gemm::MetalGemm::diag_encode_tiled_hfrag_nn`（`#[cfg(test)]
//! pub(crate)`。イシュー #1369）・`gemm_reuse_phase_diag_tests::
//! {DiagKernel, measure_one_phase_trial_with, WARMUP_TRIALS,
//! MEASURED_TRIALS, gen_square_ab, median_of}`（いずれも `pub(crate)`。
//! イシュー #1370 で `DiagKernel`／`measure_one_phase_trial_with` を
//! 新設）・`gemm_bk32_diag_tests::run_ab_pair_kernels`（`pub(crate)`。
//! イシュー #1370 で base/head 独立カーネル指定へ一般化）・
//! `tile::{CANDIDATES, select_for_device}` へ到達するため、integration
//! test ではなく `lib.rs` の兄弟モジュールとして配置する。`objc2` 系 FFI
//! 型に触れるため `cfg(all(test, target_os = "macos"))` を付ける。
//!
//! # 実行時は必ず `--test-threads=1`
//!
//! `measure_one_phase_trial_with` は `ctx.synchronize_with_gpu_
//! timestamps()` でプロセスワイドの完了バッチ数を検証するため、GPU 上
//! での複数テストスレッド競合を避ける必要がある（既存診断テスト群と
//! 同じ理由）。
//!
//! # メモリ使用量
//!
//! S 系列は N=4096・11 候補を N ごとのループスコープで `keep_alive` を
//! drop してから次の候補へ進むため、ピークは 1 候補分
//! （(20 warmup + 20 測定) × 4096² × 4 bytes ≈ 2.7 GiB）に抑える。A/B
//! 系列は E7/E8 と同じ 2 arm 同時保持（N=4096 で最大 約 5.4 GiB）を
//! N ごとのループスコープで抑える。本機 64 GiB のため truncate 不要
//! （既存診断テスト群と同じ設計）。
//!
//! # gating しない方針（既存 diag テストと同じ理由）
//!
//! S 系列・A/B 系列とも実行が成功すること（例外なく完了すること）のみを
//! 検証条件とし、`kernel_gpu` の大小関係への `assert!` は行わない
//! （有効性判断は `docs/perf/metal-gemm-hfrag-candidate.md` §9 で人間が
//! 行う。環境揺らぎによる flaky 化防止）。ただし以下は fail-closed に
//! 検証する:
//!
//! - `resolved_cfg == cfg`（フォールバック非経由。`measure_one_phase_
//!   trial_with` 内・`run_ab_pair_kernels` 内の既存 assert がそのまま
//!   適用される）
//! - A/B 系列: trial 0 の base（f32）/head（hfrag、丸め済み入力）出力が
//!   複合判定に pass する（`run_ab_pair_kernels` 内の既存 assert）
//!
//! # プロダクションコード不変
//!
//! 本ファイルは `tile.rs`／`gemm.rs`／`shaders/gemm.metal` への変更を
//! 一切含まない（診断テスト追加のみ）。opt-in 候補としての前進可否は
//! `docs/perf/metal-gemm-hfrag-candidate.md` §9 側で判断する。

use crate::context::MetalContext;
use crate::gemm::MetalGemm;
use crate::gemm_bk32_diag_tests::run_ab_pair_kernels;
use crate::gemm_reuse_phase_diag_tests::{
    DiagKernel, MEASURED_TRIALS, WARMUP_TRIALS, gen_square_ab, measure_one_phase_trial_with,
    median_of,
};
use crate::tile::{self, TileConfig};
// `MTLDevice::maxThreadgroupMemoryLength`（hfrag 候補の共有メモリ事前
// フィルタに渡すデバイス上限取得。`gemm_bk32_diag_tests.rs` と同じ
// import）。
use objc2_metal::MTLDevice;

/// [`hfrag_kernel_gpu_sweep_all_staged_candidates`] が対象とするサイズ
/// （`docs/perf/metal-gemm-hfrag-candidate.md` §9.2「S 系列」）。
const SIZES_S: [usize; 3] = [1024, 2048, 4096];

/// [`hfrag_kernel_gpu_ab_same_tile_vs_f32`]・
/// [`hfrag_kernel_gpu_ab_best_vs_production_select`] が対象とするサイズ
/// （`tile::select_for_device` の実測帯域全体。E7/E8 の `SIZES_B` と
/// 同一）。
const SIZES_AB: [usize; 4] = [512, 1024, 2048, 4096];

/// `MTLComputePipelineState::maxTotalThreadsPerThreadgroup` は構築後に
/// しか取得できないため、`gemm::MetalGemm::pipeline_for_tile_hfrag` と
/// 同じ Apple GPU 一般上限（1024）を事前検証に使う（E7/E8 と同じ判断）。
const MAX_THREADS_PER_TG_ESTIMATE: u32 = 1024;

/// S 系列（`docs/perf/metal-gemm-hfrag-candidate.md` §9.2）の M4 Max
/// 実機実測（5 プロセス起動・5 回中央値の中央値。`docs/perf/logs/
/// metal-gemm-hfrag-kernel-gpu-ab-1370/sweep_raw_combined.txt`）で確定
/// した、N 別の hfrag 最良タイル index（`tile::CANDIDATES` の添字）:
///
/// - N=1024 → `[6]`（`(bm=64,bn=32,bk=8,wm=4,wn=1)`。中央値 0.2312 ms）
/// - N=2048 → `[0]`（`(bm=64,bn=64,bk=16,wm=2,wn=2)`。中央値 1.5443 ms）
/// - N=4096 → `[9]`（`(bm=64,bn=64,bk=32,wm=2,wn=2)`。中央値 12.2568 ms）
///
/// N=512 は S 系列の対象外（`SIZES_S` 参照）のため、本番選択構成
/// （`tile::select_for_device` が返す `[5]`）をそのまま用いる（この N
/// では A 系列と B 系列が同一ペアになる。ファイル冒頭「系列構成」参照）。
/// M4 Max 本番選択構成（`tile::select_for_device` 実測）: 512→`[5]`・
/// 1024→`[6]`・2048→`[1]`・4096→`[2]`（`docs/perf/metal-gemm-n4096-
/// kernel-gap.md` §16.3）。
const HFRAG_BEST_BY_N: [(usize, usize); 4] = [(512, 5), (1024, 6), (2048, 0), (4096, 9)];

fn hfrag_best_cfg_for(n: usize) -> TileConfig {
    let idx = HFRAG_BEST_BY_N
        .iter()
        .find(|&&(size, _)| size == n)
        .unwrap_or_else(|| panic!("N={n} は HFRAG_BEST_BY_N に未登録"))
        .1;
    tile::CANDIDATES[idx]
}

/// `values` の各要素を half（RTE）へ丸めてから f32 へ戻す（ファイル冒頭
/// 「入力の丸め」参照）。`half::f16::from_f32`／`to_f32` は
/// `tests/gemm_hfrag_parity.rs` が正しさゲートの参照生成に使うのと同じ
/// 変換であり、本ファイルはそれを A/B 両系列の入力生成に転用する。
fn round_to_half_representable(values: &mut [f32]) {
    for v in values.iter_mut() {
        *v = half::f16::from_f32(*v).to_f32();
    }
}

/// hfrag 候補が受理する構成（`tile::CANDIDATES` は全 staged。デバイス
/// 上限超過のみ拒否対象）を事前検証する（E7/E8 と同じ判断: パイプライン
/// 構築失敗による性能比較前提の崩れを防ぐ）。
fn validate_hfrag_cfg(ctx: &MetalContext, cfg: TileConfig) {
    let max_shared_mem_bytes = ctx.device().maxThreadgroupMemoryLength() as u32;
    // hfrag の SMEM 使用量は f32 版のちょうど半分（`shared_mem_bytes_
    // hfrag_for`）のため、f32 版の `validate`（`shared_mem_bytes_for`
    // 相当の暗黙 SMEM 検査を含まない整除制約チェック）に加えて hfrag
    // 側の実 SMEM 使用量も上限内であることを明示検証する。
    cfg.validate(MAX_THREADS_PER_TG_ESTIMATE, max_shared_mem_bytes)
        .unwrap_or_else(|e| {
            panic!(
                "cfg={cfg:?} が共有メモリ／スレッド数事前フィルタで対象外 \
                 （TileConfig::validate 失敗: {e:?}）。本テストの前提が \
                 崩れるため中断する"
            )
        });
    let hfrag_smem = cfg.shared_mem_bytes_hfrag_for(crate::layout::TransposePattern::Nn);
    assert!(
        hfrag_smem <= max_shared_mem_bytes,
        "cfg={cfg:?} の hfrag SMEM 使用量（{hfrag_smem} バイト）がデバイス上限 \
         （{max_shared_mem_bytes} バイト）を超過する。本テストの前提が崩れる \
         ため中断する"
    );
}

/// S 系列: hfrag × 全 staged `tile::CANDIDATES` を N=1024/2048/4096 で
/// 単独計測する（`docs/perf/metal-gemm-hfrag-candidate.md` §9.2）。
/// N ごとに候補をループし、`kernel_gpu` 中央値を `println!` する
/// （比較対象を持たない単独計測のため、正確性検査は行わない — 正しさは
/// `tests/gemm_hfrag_parity.rs`・`gemm.rs` の実機 `#[ignore]` テストで
/// 既に確認済み。ファイル冒頭「配置理由」参照）。出力は
/// `aggregate.md` の抽出コマンドが
/// `N=<n> series=sweep cand=<idx> resolved_tile=<cfg> kernel_gpu_median_ms=<v> q1=<v> q3=<v>`
/// 形式で拾う。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn hfrag_kernel_gpu_sweep_all_staged_candidates() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    // hfrag は staged 経路のみ実装する契約（`gemm::MetalGemm::
    // pipeline_for_tile_hfrag` が `crate::tile::fallback_chain` 巡回中に
    // `staged=false` の候補を fail-closed に拒否する。`docs/perf/
    // metal-gemm-hfrag-candidate.md` §2.1「スコープ境界」参照）。
    // `tile::CANDIDATES[7]`（`TileConfig::SINGLE_SIMDGROUP_8X8`）は
    // `staged=false` の唯一の要素のため、スイープ対象から除外する
    // （関数名の「全 staged 候補」はこの除外後の集合を指す）。
    for &cfg in tile::CANDIDATES.iter().filter(|c| c.staged) {
        validate_hfrag_cfg(&ctx, cfg);
    }

    for n in SIZES_S {
        // N ごとにキープアライブのスコープを閉じ、ピークメモリを 1 候補分
        // に抑える（ファイル冒頭「メモリ使用量」参照）。
        let (a, b) = {
            let (mut a, mut b) = gen_square_ab(0x1370_5000 ^ (n as u64), n);
            round_to_half_representable(&mut a);
            round_to_half_representable(&mut b);
            (a, b)
        };

        for (idx, &cfg) in tile::CANDIDATES
            .iter()
            .enumerate()
            .filter(|(_, c)| c.staged)
        {
            let mut keep_alive: Vec<Vec<f32>> = Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);

            for _ in 0..WARMUP_TRIALS {
                let _ = measure_one_phase_trial_with(
                    &ctx,
                    &gemm,
                    &a,
                    &b,
                    n,
                    cfg,
                    &mut keep_alive,
                    DiagKernel::Hfrag,
                );
            }

            let mut kernel_gpu: Vec<f64> = Vec::with_capacity(MEASURED_TRIALS);
            for _ in 0..MEASURED_TRIALS {
                let s = measure_one_phase_trial_with(
                    &ctx,
                    &gemm,
                    &a,
                    &b,
                    n,
                    cfg,
                    &mut keep_alive,
                    DiagKernel::Hfrag,
                );
                assert_eq!(
                    s.resolved_cfg, cfg,
                    "N={n} cand={idx} cfg={cfg:?}: pipeline_for_tile_hfrag \
                     フォールバックが発生した(resolved={:?})。性能比較の前提が \
                     崩れるため中断する",
                    s.resolved_cfg
                );
                kernel_gpu.push(s.kernel_gpu_secs);
            }

            let q = median_of(&kernel_gpu);
            println!(
                "N={n} series=sweep cand={idx} resolved_tile={cfg:?} \
                 kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
                q.median * 1e3,
                q.q1 * 1e3,
                q.q3 * 1e3
            );
        }
    }
}

/// A 系列: `tile::select_for_device` が選ぶ構成（本番選択タイル）を
/// base（f32）/head（hfrag）両 arm に使い、N=512/1024/2048/4096 で
/// 「タイル形状を揃えた純粋な half MMA 効果」を比較する
/// （`docs/perf/metal-gemm-hfrag-candidate.md` §9.3）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn hfrag_kernel_gpu_ab_same_tile_vs_f32() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for n in SIZES_AB {
        let cfg = tile::select_for_device(n, n, n, ctx.verified_m4_max_gpu_core_count());
        validate_hfrag_cfg(&ctx, cfg);
        println!("N={n} pair=same_tile production_select_resolved={cfg:?}");

        let (a, b) = {
            let (mut a, mut b) = gen_square_ab(0x1370_a000 ^ (n as u64), n);
            round_to_half_representable(&mut a);
            round_to_half_representable(&mut b);
            (a, b)
        };

        run_ab_pair_kernels(
            &ctx,
            &gemm,
            &a,
            &b,
            n,
            cfg,
            cfg,
            "hfrag_same_tile",
            DiagKernel::F32Tiled,
            DiagKernel::Hfrag,
        );
    }
}

/// B 系列（結線判断の主根拠）: head を S 系列で確定した N 別最良タイル
/// （[`HFRAG_BEST_BY_N`]）、base を本番選択構成（`tile::select_for_
/// device`）とし、N=512/1024/2048/4096 で比較する
/// （`docs/perf/metal-gemm-hfrag-candidate.md` §9.4）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn hfrag_kernel_gpu_ab_best_vs_production_select() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for n in SIZES_AB {
        let base_cfg = tile::select_for_device(n, n, n, ctx.verified_m4_max_gpu_core_count());
        let head_cfg = hfrag_best_cfg_for(n);
        validate_hfrag_cfg(&ctx, head_cfg);
        println!(
            "N={n} pair=best_vs_production_select production_select_resolved={base_cfg:?} \
             hfrag_best_resolved={head_cfg:?}"
        );

        let (a, b) = {
            let (mut a, mut b) = gen_square_ab(0x1370_b000 ^ (n as u64), n);
            round_to_half_representable(&mut a);
            round_to_half_representable(&mut b);
            (a, b)
        };

        run_ab_pair_kernels(
            &ctx,
            &gemm,
            &a,
            &b,
            n,
            base_cfg,
            head_cfg,
            "hfrag_best_vs_production",
            DiagKernel::F32Tiled,
            DiagKernel::Hfrag,
        );
    }
}

/// スモークテスト（N=512・少数反復）: 丸め済み入力に対して hfrag/f32 が
/// 複合判定に pass することを短時間で確認する（ファイル冒頭「入力の
/// 丸め」参照。フル A/B 系列の実行前に前提を検証する導線）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn hfrag_smoke_rounded_inputs_pass_composite_vs_f32() {
    const N: usize = 512;

    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    let cfg = tile::select_for_device(N, N, N, ctx.verified_m4_max_gpu_core_count());
    validate_hfrag_cfg(&ctx, cfg);

    let (mut a, mut b) = gen_square_ab(0x1370_5a0c, N);
    round_to_half_representable(&mut a);
    round_to_half_representable(&mut b);

    run_ab_pair_kernels(
        &ctx,
        &gemm,
        &a,
        &b,
        N,
        cfg,
        cfg,
        "hfrag_smoke",
        DiagKernel::F32Tiled,
        DiagKernel::Hfrag,
    );
}
