//! E7 実測（イシュー #1324/#1329 に続く採否判断。イシュー #1330）:
//! `tile::CANDIDATES[9]`（64×64・`bk=32`・wm2wn2。イシュー #1329・
//! PR #1391 で追加。K ループ 1 反復あたりの `threadgroup_barrier` 往復を
//! 半減させる狙い）の純カーネル時間（GPU タイムスタンプ。イシュー #1276
//! の `kernel_gpu` 変種）を、
//!
//! - **A 系列**: `tile::CANDIDATES[0]`（同じ 64×64 タイル・`bk=16`。
//!   `[9]` の直接比較対象）
//! - **B 系列**: `tile::select_for_device` が実際に選ぶ本番選択構成
//!   （M4 Max では N=512→`[5]`・1024→`[6]`・2048→`[1]`・4096→`[2]`）
//!
//! の 2 系列で N=1024/2048/4096（B 系列は 512 も含む）× 5 プロセス
//!起動・5 回中央値比較し、`tile::select` への `CANDIDATES[9]` 組み込み
//! 可否を判定するための診断テスト。
//!
//! # A 系列単独では「組み込み可否」を判断できない（計画上の前提）
//!
//! 本番 `dispatch_auto`（`tile::select_for_device` の M4 Max 厳密一致
//! テーブル）は N=2048/4096 で `CANDIDATES[0]` を選ばない（`docs/perf/
//! metal-gemm-n4096-kernel-gap.md` §12 実測: `[0]` は N=4096 で本番選択
//! 構成 `[2]` より約 7.7 倍遅い）。したがって `[9]` が `[0]` に勝っても
//! `select` を触る根拠にはならない。A 系列は #1324 の「バリア半減
//! 仮説」に対する直接回答（`[9]` vs `[0]` の isolated 比較）に限定し、
//! B 系列（`[9]` vs 本番選択構成）を組み込み判断の唯一の根拠とする
//! （`docs/perf/metal-gemm-n4096-kernel-gap.md` §14 参照）。
//!
//! # 両 arm とも同一の `MetalGemm::new`（既定 Legacy 構成）インスタンス
//!
//! E6（`gemm_tile_class_diag_tests.rs`）は base/head で `TileClassMode`
//! が異なるため 2 個の `MetalGemm` インスタンスを必要としたが、本テスト
//! は base/head とも実行モード自体は同一（既定構成）で `TileConfig`
//! （タイル形状）のみが異なる。`gemm::MetalGemm::diag_encode_tiled_nn`
//! は `cfg: tile::TileConfig` を呼び出しごとの引数として受け取るため、
//! 1 個の `MetalGemm::new(&ctx)` インスタンスを base/head 双方で共有
//! できる（`gemm_reuse_phase_diag_tests::measure_one_phase_trial` の
//! 呼び出し方をそのまま踏襲。プロダクションコードは無変更）。
//!
//! # 配置理由（`gemm_tile_class_diag_tests.rs` と同じ判断）
//!
//! `gemm::MetalGemm::diag_encode_tiled_nn`（`#[cfg(test)] pub(crate)`）・
//! `gemm_reuse_phase_diag_tests::{measure_one_phase_trial, WARMUP_TRIALS,
//! MEASURED_TRIALS, gen_square_ab, median_of}`（いずれも `pub(crate)`）・
//! `tile::{CANDIDATES, select_for_device}`（`pub(crate)`／`pub`）へ到達
//! するため、integration test ではなく `lib.rs` の兄弟モジュールとして
//! 配置する。
//!
//! # 実行時は必ず `--test-threads=1`
//!
//! `measure_one_phase_trial` は `ctx.synchronize_with_gpu_timestamps()`
//! でプロセスワイドの完了バッチ数を検証するため、GPU 上での複数テスト
//! スレッド競合を避ける必要がある（`gemm_tile_class_diag_tests.rs` と
//! 同じ理由）。
//!
//! # メモリ使用量
//!
//! ホスト側キープアライブは A 系列 N=4096・2 arm 同時保持で
//! (20 warmup + 20 測定) × 4096² × 4 bytes × 2 arm ≈ 5.4 GiB（統合
//! メモリ上。本機 64 GiB のため truncate 不要）。N ごとのループスコープで
//! 両 arm の `keep_alive` を drop してから次の N へ進むことでピークを
//! 1 N 分に抑える（`gemm_tile_class_diag_tests.rs` と同じ設計）。
//!
//! # gating しない方針（既存 diag テストと同じ理由）
//!
//! 実行が成功すること（例外なく完了すること）のみを検証条件とし、
//! `kernel_gpu` の大小関係への `assert!` は行わない（有効性判断は
//! `docs/perf/metal-gemm-n4096-kernel-gap.md` §14 で人間が行う。環境
//! 揺らぎによる flaky 化防止）。ただし以下は fail-closed に検証する:
//!
//! - `resolved_cfg == cfg`（フォールバック非経由。片側だけ別構成へ
//!   フォールバックした反復を性能比較に混入させない。`[9]` が
//!   `pipeline_for_tile` でフォールバックした場合はここで panic する
//!   ため、その事実自体が REJECT の根拠になる）
//! - trial 0 の base/head 出力が
//!   [`fandhe_ai_backend_cpu::parity::compare`] の複合判定（相対誤差
//!   1e-3 未満 または 絶対誤差 1e-5 未満）に pass する（**bit 完全一致は
//!   要求しない**。`[9]` は `bk=32` で K 分割粒度が `[0]`／本番選択構成
//!   〈いずれも `bk=16` または `bk=32`〈`[5]`〉〉と異なりうるため、
//!   浮動小数点加算順序が変わり bit 一致は契約外——`coding-rust.md`
//!   「バックエンド間数値一致は統一複合判定」を単一バックエンド内の
//!   タイル形状違い比較にも適用する）
//!
//! # プロダクションコード不変
//!
//! 本ファイルは `tile.rs`／`gemm.rs`／`shaders/gemm.metal` への変更を
//! 一切含まない（診断テスト追加のみ）。`tile::select`／
//! `select_with_occupancy_for_device` の候補表への `CANDIDATES[9]` 組み
//! 込みは本ファイルの実測結果を受けて別途 `docs/perf/
//! metal-gemm-n4096-kernel-gap.md` §14 側で判断する。ADOPT の場合の
//! `tile.rs` 変更は別コミット（同一 PR 内の別ステップ）で行う。

use crate::context::MetalContext;
use crate::gemm::MetalGemm;
use crate::gemm_reuse_phase_diag_tests::{
    MEASURED_TRIALS, WARMUP_TRIALS, gen_square_ab, measure_one_phase_trial, median_of,
};
use crate::tile::{self, TileConfig};
// `MTLDevice::maxThreadgroupMemoryLength`（`CANDIDATES[9]` の共有メモリ
// 事前フィルタに渡すデバイス上限取得。`gemm_tile_class_diag_tests.rs`
// と同じ import）。
use objc2_metal::MTLDevice;

/// [`bk32_kernel_gpu_ab_vs_candidate0`] が対象とするサイズ（Issue の
/// 主対象は 2048/4096。1024 は参考値として追加する）。
const SIZES_A: [usize; 3] = [1024, 2048, 4096];

/// [`bk32_kernel_gpu_ab_vs_production_select`] が対象とするサイズ
/// （`tile::select_for_device` の実測帯域全体。512/1024 は非後退情報
/// として追加する）。
const SIZES_B: [usize; 4] = [512, 1024, 2048, 4096];

/// `MTLComputePipelineState::maxTotalThreadsPerThreadgroup` は構築後に
/// しか取得できないため、`gemm::MetalGemm::pipeline_for_tile` と同じ
/// Apple GPU 一般上限（1024）を事前検証に使う（`gemm_tile_class_diag_
/// tests.rs` と同じ判断）。
const MAX_THREADS_PER_TG_ESTIMATE: u32 = 1024;

/// 1 回の base/head trial 交互ループを実行し、`kernel_gpu` 中央値・
/// 複合判定 pass・フォールバック非経由を検証したうえで
/// `(base_median_ms, head_median_ms)` を返す共通ヘルパ（A/B 両系列で
/// 交互測定ロジックを重複させない。`gemm_tile_class_diag_tests.rs` の
/// trial 交互ループを 2 系列で再利用できる形に切り出したもの）。
///
/// `pair_label` は出力の識別子（`aggregate.md` の抽出パターンに
/// 対応。例: `cand0` や `production_select`）。
#[allow(clippy::too_many_arguments)]
fn run_ab_pair(
    ctx: &MetalContext,
    gemm: &MetalGemm,
    a: &[f32],
    b: &[f32],
    n: usize,
    base_cfg: TileConfig,
    head_cfg: TileConfig,
    pair_label: &str,
) {
    let mut keep_alive_base: Vec<Vec<f32>> = Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);
    let mut keep_alive_head: Vec<Vec<f32>> = Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);

    // warmup: 各 arm の初回 MSL パイプライン構築コストを吸収する。
    for _ in 0..WARMUP_TRIALS {
        let _ = measure_one_phase_trial(ctx, gemm, a, b, n, base_cfg, &mut keep_alive_base);
        let _ = measure_one_phase_trial(ctx, gemm, a, b, n, head_cfg, &mut keep_alive_head);
    }

    let mut kernel_gpu_base: Vec<f64> = Vec::with_capacity(MEASURED_TRIALS);
    let mut kernel_gpu_head: Vec<f64> = Vec::with_capacity(MEASURED_TRIALS);
    let mut first_output_base: Option<Vec<f32>> = None;
    let mut first_output_head: Option<Vec<f32>> = None;

    for trial in 0..MEASURED_TRIALS {
        // trial index による開始オフセット回転（order-bias 相殺。
        // `gemm_tile_class_diag_tests.rs` と同じ設計）。
        let base_first = trial % 2 == 0;
        let (sample_base, sample_head) = if base_first {
            let sb = measure_one_phase_trial(ctx, gemm, a, b, n, base_cfg, &mut keep_alive_base);
            let sh = measure_one_phase_trial(ctx, gemm, a, b, n, head_cfg, &mut keep_alive_head);
            (sb, sh)
        } else {
            let sh = measure_one_phase_trial(ctx, gemm, a, b, n, head_cfg, &mut keep_alive_head);
            let sb = measure_one_phase_trial(ctx, gemm, a, b, n, base_cfg, &mut keep_alive_base);
            (sb, sh)
        };

        // フォールバック非経由の fail-closed 検証（ファイル冒頭
        // 「gating しない方針」参照）。
        assert_eq!(
            sample_base.resolved_cfg, base_cfg,
            "N={n} trial={trial} pair={pair_label} base: pipeline_for_tile \
             フォールバックが発生した(requested={base_cfg:?}, \
             resolved={:?})。性能比較の前提が崩れるため中断する",
            sample_base.resolved_cfg
        );
        assert_eq!(
            sample_head.resolved_cfg, head_cfg,
            "N={n} trial={trial} pair={pair_label} head: pipeline_for_tile \
             フォールバックが発生した(requested={head_cfg:?}, \
             resolved={:?})。性能比較の前提が崩れるため中断する",
            sample_head.resolved_cfg
        );

        kernel_gpu_base.push(sample_base.kernel_gpu_secs);
        kernel_gpu_head.push(sample_head.kernel_gpu_secs);

        if trial == 0 {
            first_output_base = keep_alive_base.last().cloned();
            first_output_head = keep_alive_head.last().cloned();
        }
    }

    // 正確性の fail-closed 検査: trial 0 の base/head 出力を複合判定
    // （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で比較する
    // （ファイル冒頭「gating しない方針」参照。bit 完全一致は要求
    // しない — `bk` が異なりうるため K 分割の加算順序が変わりうる）。
    let out_base =
        first_output_base.expect("MEASURED_TRIALS > 0 のため trial 0 の base 出力は必ず Some");
    let out_head =
        first_output_head.expect("MEASURED_TRIALS > 0 のため trial 0 の head 出力は必ず Some");
    let report = fandhe_ai_backend_cpu::parity::compare(&out_base, &out_head)
        .expect("parity::compare は要素数一致・NaN 非混入の入力に対して常に Ok を返す");
    assert!(
        report.passes(),
        "N={n} pair={pair_label}: base/head の出力が複合判定（相対誤差 1e-3 \
         未満 または 絶対誤差 1e-5 未満）に pass しない（report={report:?}）"
    );

    let q_base = median_of(&kernel_gpu_base);
    let q_head = median_of(&kernel_gpu_head);
    println!(
        "N={n} pair={pair_label} mode=base resolved_tile={base_cfg:?} \
         kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
        q_base.median * 1e3,
        q_base.q1 * 1e3,
        q_base.q3 * 1e3
    );
    println!(
        "N={n} pair={pair_label} mode=head resolved_tile={head_cfg:?} \
         kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
        q_head.median * 1e3,
        q_head.q1 * 1e3,
        q_head.q3 * 1e3
    );
    let ratio = q_head.median / q_base.median;
    println!("N={n} pair={pair_label} head_over_base_kernel_gpu={ratio:.6}");
}

/// A 系列（Issue 受け入れ基準）: `CANDIDATES[9]`（bk=32・head）vs
/// `CANDIDATES[0]`（bk=16・base）を N=1024/2048/4096 で比較する
/// （#1324「バリア半減仮説」への直接回答。ファイル冒頭「A 系列単独では
/// 組み込み可否を判断できない」参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn bk32_kernel_gpu_ab_vs_candidate0() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    let base_cfg = tile::CANDIDATES[0];
    let head_cfg = tile::CANDIDATES[9];

    let max_shared_mem_bytes = ctx.device().maxThreadgroupMemoryLength() as u32;
    for &cfg in &[base_cfg, head_cfg] {
        cfg.validate(MAX_THREADS_PER_TG_ESTIMATE, max_shared_mem_bytes)
            .unwrap_or_else(|e| {
                panic!(
                    "cfg={cfg:?} が共有メモリ／スレッド数事前フィルタで対象外 \
                     （TileConfig::validate 失敗: {e:?}）。本テストの前提が \
                     崩れるため中断する"
                )
            });
    }

    for n in SIZES_A {
        let (a, b) = gen_square_ab(0x1330_a000 ^ (n as u64), n);
        run_ab_pair(&ctx, &gemm, &a, &b, n, base_cfg, head_cfg, "cand9_vs_cand0");
    }
}

/// B 系列（結線判断に必須）: `CANDIDATES[9]`（head）vs
/// `tile::select_for_device` の本番選択構成（base）を N=512/1024/2048/
/// 4096 で比較する（ファイル冒頭「A 系列単独では組み込み可否を判断
/// できない」参照。`select` のテーブルを変更する唯一の根拠）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn bk32_kernel_gpu_ab_vs_production_select() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    let head_cfg = tile::CANDIDATES[9];
    let max_shared_mem_bytes = ctx.device().maxThreadgroupMemoryLength() as u32;
    head_cfg
        .validate(MAX_THREADS_PER_TG_ESTIMATE, max_shared_mem_bytes)
        .unwrap_or_else(|e| {
            panic!(
                "head_cfg={head_cfg:?} が共有メモリ／スレッド数事前フィルタで \
                 対象外（TileConfig::validate 失敗: {e:?}）。本テストの前提が \
                 崩れるため中断する"
            )
        });

    for n in SIZES_B {
        // `dispatch_auto` と同一の構成解決（`gemm_reuse_phase_diag_
        // tests::run_size_with` と同じ呼び出し方）。解決結果を
        // `println!` で残し、docs 側での突合を容易にする。
        let base_cfg = tile::select_for_device(n, n, n, ctx.verified_m4_max_gpu_core_count());
        println!("N={n} pair=production_select production_select_resolved={base_cfg:?}");

        let (a, b) = gen_square_ab(0x1330_b000 ^ (n as u64), n);
        run_ab_pair(
            &ctx,
            &gemm,
            &a,
            &b,
            n,
            base_cfg,
            head_cfg,
            "production_select",
        );
    }
}
