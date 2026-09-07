//! E8 実測（イシュー #1325/#1331 に続く採否判断。イシュー #1332）:
//! `tile::CANDIDATES[10]`（128×64・`bk=16`・2×2 simdgroup・simdgroup
//! あたり acc 32。イシュー #1331・PR #1393 で追加。threadgroup タイル
//! 拡張によるタイル再利用率向上の狙い）の純カーネル時間（GPU
//! タイムスタンプ。イシュー #1276 の `kernel_gpu` 変種）を、
//!
//! - **A 系列**: (a) `tile::CANDIDATES[0]`（64×64・`bk=16`。`[10]` の
//!   構造上の直接対応。#1331/§15.6 の引き継ぎ）、(b)
//!   `tile::CANDIDATES[3]`（イシュー本文が「現行 N=4096 最良候補」と
//!   呼ぶ #744 時点の候補。参考ペア）
//! - **B 系列**: `tile::select_for_device` が実際に選ぶ本番選択構成
//!   （M4 Max では N=512→`[5]`・1024→`[6]`・2048→`[1]`・4096→`[2]`。
//!   `docs/perf/metal-gemm-n4096-kernel-gap.md` §14）
//!
//! の 2 系列で N=1024/2048/4096（A 系列。B 系列は 512 も含む）×
//! 5 プロセス起動・5 回中央値比較し、`tile::select` への
//! `CANDIDATES[10]` 組み込み可否を判定するための診断テスト
//! （E7 `gemm_bk32_diag_tests.rs` と同型構成）。
//!
//! # イシュー文言の「現行最良候補」注記（計画上の前提）
//!
//! イシュー #1332 本文は「現行 N=4096 最良候補（`CANDIDATES[3]`／
//! `[0]`）」と書くが、これは #744 時点の古い記述である。`tile.rs` の
//! M4 Max 厳密一致テーブル（`select_with_occupancy_for_device` 内）と
//! `docs/perf/metal-gemm-n4096-kernel-gap.md` §14.3 実測では
//! **2048→`CANDIDATES[1]`（64,32,16,2,2）・4096→`CANDIDATES[2]`
//! （32,64,16,2,2）** が現行本番選択構成であり、`[3]` は #1039 以降
//! 最良ではない。したがって「現行最良候補」は B 系列（`select_for_
//! device` の実選択構成）で表現し、`[0]`（構造上の直接対応。§15.6 の
//! 引き継ぎ）を A 系列の主対象、`[3]` をイシュー文言突合用の参考
//! ペアとして追加する。この整理は `docs/perf/metal-gemm-n4096-
//! kernel-gap.md` §16.0 にも記載する。
//!
//! # A 系列単独では「組み込み可否」を判断できない（E7 と同じ理由）
//!
//! 本番 `dispatch_auto`（`tile::select_for_device` の M4 Max 厳密一致
//! テーブル）は N=2048/4096 で `CANDIDATES[0]`／`[3]` を選ばない。
//! したがって `[10]` が `[0]`／`[3]` に勝っても `select` を触る根拠には
//! ならない。A 系列は #1325 の「タイル再利用率向上仮説」に対する直接
//! 回答（`[10]` vs `[0]`／`[3]` の isolated 比較）に限定し、B 系列
//! （`[10]` vs 本番選択構成）を組み込み判断の唯一の根拠とする
//! （`docs/perf/metal-gemm-n4096-kernel-gap.md` §16 参照）。
//!
//! # 両 arm とも同一の `MetalGemm::new`（既定 Legacy 構成）インスタンス
//!
//! `gemm::MetalGemm::diag_encode_tiled_nn` は `cfg: tile::TileConfig`
//! を呼び出しごとの引数として受け取るため、1 個の `MetalGemm::new(&ctx)`
//! インスタンスを base/head 双方で共有できる（E7 と同じ設計。
//! プロダクションコードは無変更）。
//!
//! # 配置理由（E7 と同じ判断）
//!
//! `gemm::MetalGemm::diag_encode_tiled_nn`（`#[cfg(test)] pub(crate)`）・
//! `gemm_reuse_phase_diag_tests::{measure_one_phase_trial, WARMUP_TRIALS,
//! MEASURED_TRIALS, gen_square_ab, median_of}`（いずれも `pub(crate)`）・
//! `gemm_bk32_diag_tests::run_ab_pair`（`pub(crate)`。E7/E8 で共有する
//! 交互測定・fail-closed 検証ヘルパ。本イシューで複製せず共有へ
//! 昇格した）・`tile::{CANDIDATES, select_for_device}`（`pub(crate)`／
//! `pub`）へ到達するため、integration test ではなく `lib.rs` の兄弟
//! モジュールとして配置する。
//!
//! # 実行時は必ず `--test-threads=1`
//!
//! `measure_one_phase_trial` は `ctx.synchronize_with_gpu_timestamps()`
//! でプロセスワイドの完了バッチ数を検証するため、GPU 上での複数テスト
//! スレッド競合を避ける必要がある（E7 と同じ理由）。
//!
//! # メモリ使用量
//!
//! ホスト側キープアライブは A 系列 N=4096・2 arm 同時保持で
//! (20 warmup + 20 測定) × 4096² × 4 bytes × 2 arm ≈ 5.4 GiB（統合
//! メモリ上。本機 64 GiB のため truncate 不要）。N ごとのループスコープで
//! 両 arm の `keep_alive` を drop してから次の N へ進むことでピークを
//! 1 N 分に抑える（E7 と同じ設計）。
//!
//! # gating しない方針（E7 と同じ理由）
//!
//! 実行が成功すること（例外なく完了すること）のみを検証条件とし、
//! `kernel_gpu` の大小関係への `assert!` は行わない（有効性判断は
//! `docs/perf/metal-gemm-n4096-kernel-gap.md` §16 で人間が行う。環境
//! 揺らぎによる flaky 化防止）。ただし以下は fail-closed に検証する:
//!
//! - `resolved_cfg == cfg`（フォールバック非経由。片側だけ別構成へ
//!   フォールバックした反復を性能比較に混入させない）
//! - trial 0 の base/head 出力が
//!   [`fandhe_ai_backend_cpu::parity::compare`] の複合判定（相対誤差
//!   1e-3 未満 または 絶対誤差 1e-5 未満）に pass する（**bit 完全一致は
//!   要求しない**。`[10]` はタイル形状（simdgroup 数・acc レイアウト）
//!   が `[0]`／`[3]`／本番選択構成と異なりうるため、浮動小数点加算順序が
//!   変わり bit 一致は契約外——`coding-rust.md` の「バックエンド間数値
//!   一致は統一複合判定」を単一バックエンド内のタイル形状違い比較にも
//!   適用する）
//!
//! # プロダクションコード不変
//!
//! 本ファイルは `tile.rs`／`gemm.rs`／`shaders/gemm.metal` への変更を
//! 一切含まない（診断テスト追加のみ）。`tile::select`／
//! `select_with_occupancy_for_device` の候補表への `CANDIDATES[10]` 組み
//! 込みは本ファイルの実測結果を受けて別途 `docs/perf/
//! metal-gemm-n4096-kernel-gap.md` §16 側で判断する。ADOPT の場合の
//! `tile.rs` 変更は別コミット（同一 PR 内の別ステップ）で行う。

use crate::context::MetalContext;
use crate::gemm::MetalGemm;
use crate::gemm_bk32_diag_tests::run_ab_pair;
use crate::gemm_reuse_phase_diag_tests::gen_square_ab;
use crate::tile::{self, TileConfig};
// `MTLDevice::maxThreadgroupMemoryLength`（`CANDIDATES[10]` の共有メモリ
// 事前フィルタに渡すデバイス上限取得。E7 と同じ import）。
use objc2_metal::MTLDevice;

/// [`bm128_kernel_gpu_ab_vs_candidate0`] が対象とするサイズ（Issue の
/// 主対象は 2048/4096。1024 は参考値として追加する）。
const SIZES_A: [usize; 3] = [1024, 2048, 4096];

/// [`bm128_kernel_gpu_ab_vs_production_select`] が対象とするサイズ
/// （`tile::select_for_device` の実測帯域全体。512/1024 は非後退情報
/// として追加する）。
const SIZES_B: [usize; 4] = [512, 1024, 2048, 4096];

/// `MTLComputePipelineState::maxTotalThreadsPerThreadgroup` は構築後に
/// しか取得できないため、`gemm::MetalGemm::pipeline_for_tile` と同じ
/// Apple GPU 一般上限（1024）を事前検証に使う（E7 と同じ判断）。
const MAX_THREADS_PER_TG_ESTIMATE: u32 = 1024;

/// A 系列（Issue 受け入れ基準＋参考ペア）: `CANDIDATES[10]`（head）を
/// `CANDIDATES[0]`（(a) base。構造上の直接対応）・`CANDIDATES[3]`
/// （(b) base。イシュー文言突合用の参考）の 2 通りと N=1024/2048/4096
/// で比較する（#1325「タイル再利用率向上仮説」への直接回答。ファイル
/// 冒頭「A 系列単独では組み込み可否を判断できない」参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn bm128_kernel_gpu_ab_vs_candidate0() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    let head_cfg = tile::CANDIDATES[10];
    let base_cfg_a = tile::CANDIDATES[0];
    let base_cfg_b = tile::CANDIDATES[3];

    let max_shared_mem_bytes = ctx.device().maxThreadgroupMemoryLength() as u32;
    for &cfg in &[head_cfg, base_cfg_a, base_cfg_b] {
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
        let (a, b) = gen_square_ab(0x1332_a000 ^ (n as u64), n);
        run_ab_pair(
            &ctx,
            &gemm,
            &a,
            &b,
            n,
            base_cfg_a,
            head_cfg,
            "cand10_vs_cand0",
        );
        run_ab_pair(
            &ctx,
            &gemm,
            &a,
            &b,
            n,
            base_cfg_b,
            head_cfg,
            "cand10_vs_cand3",
        );
    }
}

/// B 系列（結線判断に必須）: `CANDIDATES[10]`（head）vs
/// `tile::select_for_device` の本番選択構成（base）を N=512/1024/2048/
/// 4096 で比較する（ファイル冒頭「A 系列単独では組み込み可否を判断
/// できない」参照。`select` のテーブルを変更する唯一の根拠）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn bm128_kernel_gpu_ab_vs_production_select() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    let head_cfg = tile::CANDIDATES[10];
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
        // `dispatch_auto` と同一の構成解決（E7 と同じ呼び出し方）。
        // 解決結果を `println!` で残し、docs 側での突合を容易にする。
        let base_cfg: TileConfig =
            tile::select_for_device(n, n, n, ctx.verified_m4_max_gpu_core_count());
        println!("N={n} pair=production_select production_select_resolved={base_cfg:?}");

        let (a, b) = gen_square_ab(0x1332_b000 ^ (n as u64), n);
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
