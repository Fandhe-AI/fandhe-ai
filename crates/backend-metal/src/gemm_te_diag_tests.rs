//! `thread_elements()` 方式 BlockMMA 候補（イシュー #1693。
//! `gemm_simdgroup_tiled_te`／`tile::MmaFragLoad::ThreadElements`）の
//! 純カーネル専有時間（GPU タイムスタンプ。`kernel_gpu`。イシュー
//! #1276）を本番選択構成（`tile::select_for_device`）と M4 Max 実機で
//! A/B 比較する診断テスト（イシュー #1694）。
//!
//! # 位置づけ・前提ゲート
//!
//! `docs/perf/metal-gemm-thread-elements-candidate.md` §4「実機記入欄」
//! が定義する R0（probe）→ R1（parity・正しさ）→ R2（非 staged 拒否）→
//! R3（本番との bit 一致）を Mac セッションが先に通してから、本ファイル
//! の A/B（性能）を実行する契約（イシュー #1694 issue コメントの事前
//! 登録判定規則 1 を参照）。R0/R1 が FAIL の場合は性能 A/B を実施しない
//! （緩和・再試行での救済なし）。
//!
//! # 両 arm は独立インスタンス（E2 spec_source と同型。E7/E8/hfrag の
//! 単一インスタンス切替とは異なる）
//!
//! `gemm_bk32_diag_tests::run_ab_pair_kernels`（E7/E8/hfrag が使う交互
//! 測定ヘルパ）は単一 `MetalGemm` インスタンス上で `TileConfig`／
//! `DiagKernel` を呼び出しごとに切り替える設計だが、本候補は
//! `MetalGemm` インスタンス自身が保持する `mma_frag_load` フィールド
//! （`MetalGemm::new_with_mma_frag_load` でのみ指定可能）でカーネル
//! 関数名を切り替える（`pipeline_for_tile` 自身が
//! `self.mma_frag_load` を見る設計。`docs/perf/metal-gemm-thread-
//! elements-candidate.md` §2 参照）。したがって本ファイルは
//! `gemm_spec_source_diag_tests.rs`（base/head 2 個の `MetalGemm`
//! インスタンスを独立に構築する型）と同じ構成を踏襲する。
//!
//! - base: `MetalGemm::new(&ctx)`（`MmaFragLoad::SimdgroupLoad`。本番
//!   既定）。
//! - head: `MetalGemm::new_with_mma_frag_load(&ctx,
//!   MmaFragLoad::ThreadElements)`。
//!
//! `diag_encode_tiled_nn`（f32 版・既存の計測境界専用入口）は
//! `pipeline_for_tile` を経由するため、`ThreadElements` インスタンス
//! でもそのまま使える（新規 `DiagKernel` variant・`gemm.rs`／`tile.rs`／
//! `shaders/gemm.metal` の変更は不要）。
//!
//! # false-green 防止（出力が bit 同一になりうるため出力比較だけでは
//! 経路切替を証明できない）
//!
//! 以下を fail-closed な `assert!` として検証する:
//!
//! 1. `base.mma_frag_load() == SimdgroupLoad`・
//!    `head.mma_frag_load() == ThreadElements`（`#[cfg(test)]
//!    pub(crate)` アクセサ。#1693 が「#1694 の実機 A/B が使う」と
//!    明記した面）。
//! 2. `cfg.staged`（te ガードは非 staged 候補を `pipeline_for_tile` が
//!    fail-closed に拒否するため、非 staged 構成が選ばれた場合に
//!    head 側だけ panic して初めて気づくのではなく、本テスト側で
//!    事前に構成の前提を検証する）。
//! 3. 毎 trial `base_sample.resolved_cfg == cfg` かつ
//!    `head_sample.resolved_cfg == cfg`
//!    （`measure_one_phase_trial_with` は自身では `resolved_cfg` を
//!    assert しないため、本ファイル側でフォールバック非経由を検証
//!    する。`gemm_bk32_diag_tests::run_ab_pair_kernels` と同じ判断）。
//!
//! # 正しさ・checksum
//!
//! trial 0 の head 出力を `matmul_reference_fma`（CPU 参照）と
//! `assert_parity`（REQ-2 統一複合判定。相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満）で fail-closed 検証する（E8 と同型。
//! `tests/gemm_te_parity.rs` は N=2048 までのため、本テストの
//! N=4096 は新規カバレッジ。CPU 参照実装〈逐次 3 重ループ〉は
//! 4096³ で数十秒〜分単位かかりうる — README 参照）。
//!
//! trial 0 の base／head 出力の checksum（f64 逐次和）と
//! `bit_identical=<bool>`（要素ごと `to_bits()` 比較）を `println!`
//! する。checksum 完全一致は本テスト内では abort させず、集計側
//! （`docs/perf/logs/metal-gemm-thread-elements-ab-1694/aggregate.py`）
//! の事前登録規則として適用する（性能データを残すため。
//! `gemm_bk32_diag_tests::run_ab_pair_kernels` は複合判定 FAIL で
//! abort するが、本ファイルは bit 一致という厳格な検査を集計側の
//! 規則に委ねる設計判断）。
//!
//! # 配置理由（既存診断テスト群と同じ判断）
//!
//! `gemm::MetalGemm::{new, new_with_mma_frag_load, mma_frag_load}`
//! （`new_with_mma_frag_load`／`mma_frag_load` はいずれも #1693 で
//! 追加。前者は `pub`、後者は `#[cfg(test)] pub(crate)`）・
//! `gemm_reuse_phase_diag_tests::{measure_one_phase_trial,
//! WARMUP_TRIALS, MEASURED_TRIALS, gen_square_ab, median_of}`・
//! `tile::select_for_device` へ到達するため、integration test では
//! なく `lib.rs` の兄弟モジュールとして配置する。`objc2` 系 FFI 型に
//! 触れるため `cfg(all(test, target_os = "macos"))` を付ける。
//!
//! # 実行時は必ず `--test-threads=1`
//!
//! `measure_one_phase_trial` は `ctx.synchronize_with_gpu_
//! timestamps()` でプロセスワイドの完了バッチ数を検証するため、GPU
//! 上での複数テストスレッド競合を避ける必要がある（既存診断テスト群
//! と同じ理由）。
//!
//! # メモリ使用量
//!
//! ホスト側キープアライブ（`Vec<Vec<f32>>`）は N=4096・2 arm 同時保持で
//! (20 warmup + 20 測定) × 4096² × 4 bytes × 2 arm ≈ 5.4 GiB（統合
//! メモリ上。本機 64 GiB のため truncate 不要。E7/E8/hfrag と同じ
//! 見積り）。N ごとのループスコープで両 arm の `keep_alive` を drop
//! してから次の N へ進むことでピークを 1 N 分に抑える。
//!
//! # gating しない方針（既存診断テスト群と同じ理由）
//!
//! 実行が成功すること（例外なく完了すること）のみを検証条件とし、
//! `kernel_gpu` の大小関係への `assert!` は行わない（有効性判断は
//! `docs/perf/metal-gemm-thread-elements-candidate.md` §7 で人間が
//! 行う。環境揺らぎによる flaky 化防止）。フォールバック非経由・
//! 正しさ（REQ-2）は上記のとおり fail-closed に検証する。
//!
//! # プロダクションコード不変
//!
//! 本ファイルは `tile.rs`／`gemm.rs`／`shaders/gemm.metal` への変更を
//! 一切含まない（診断テスト追加のみ）。`tile::select`／
//! `dispatch_auto` への組み込み可否は本ファイルの実測結果を受けて
//! 別途ユーザー承認のうえ判断する（`docs/perf/metal-gemm-thread-
//! elements-candidate.md` §5〜§7 参照）。

use crate::context::MetalContext;
use crate::gemm::MetalGemm;
use crate::gemm_reuse_phase_diag_tests::{
    MEASURED_TRIALS, WARMUP_TRIALS, gen_square_ab, measure_one_phase_trial, median_of,
};
use crate::tile::{self, MmaFragLoad};
use fandhe_ai_backend_cpu::parity::{assert_parity, matmul_reference_fma};

/// 本番選択構成（`tile::select_for_device`）が実測帯域全体をカバー
/// するよう、`gemm_bk32_diag_tests::SIZES_B`（B 系列）と同一の対象
/// サイズを使う。
const SIZES_AB: [usize; 4] = [512, 1024, 2048, 4096];

/// checksum（f64 逐次和。`docs/perf/device-checksum-readback-ab.md`
/// と同じ「ホスト f64 逐次和」定義。index 順で加算し決定的にする）。
fn checksum_f64(values: &[f32]) -> f64 {
    let mut acc = 0.0f64;
    for &v in values {
        acc += v as f64;
    }
    acc
}

/// 要素ごと `to_bits()` 比較（`checksum_f64` の f64 値そのものではなく
/// 出力 `Vec<f32>` 全体の bit 一致を見る。checksum は `println!` での
/// 突合用の補助情報であり、実際の bit 一致判定は本関数が担う）。
fn bit_identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// R0〜R3 前提ゲート・実機（Metal）依存の性能 A/B 診断テスト（イシュー
/// #1694。事前登録判定規則は issue コメント参照）。`--test-threads=1`
/// 必須（ファイル冒頭コメント参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn te_kernel_gpu_ab_vs_production_select() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let base = MetalGemm::new(&ctx).expect("base（本番既定 SimdgroupLoad）GEMM 構築に失敗した");
    let head = MetalGemm::new_with_mma_frag_load(&ctx, MmaFragLoad::ThreadElements)
        .expect("head（ThreadElements 候補）GEMM 構築に失敗した");

    // false-green 防止 (1): 両 arm が意図した mma_frag_load を保持して
    // いることを構築直後に検証する（ファイル冒頭コメント参照）。
    assert_eq!(
        base.mma_frag_load(),
        MmaFragLoad::SimdgroupLoad,
        "base は本番既定（SimdgroupLoad）でなければならない"
    );
    assert_eq!(
        head.mma_frag_load(),
        MmaFragLoad::ThreadElements,
        "head は候補経路（ThreadElements）でなければならない"
    );

    for n in SIZES_AB {
        let cfg = tile::select_for_device(n, n, n, ctx.verified_m4_max_gpu_core_count());
        // false-green 防止 (2): te ガードは非 staged 候補を拒否する
        // ため、本番選択構成が非 staged であれば head 側が別構成へ
        // フォールバックしうる（構造上該当しない想定だが、本番選択
        // テーブルの将来変更に備え明示的に検証する）。
        assert!(
            cfg.staged,
            "N={n}: tile::select_for_device が非 staged 構成 {cfg:?} を \
             選んだ。te ガードは非 staged 候補を拒否するため本 A/B の \
             前提が崩れる"
        );
        println!("N={n} pair=te_vs_production_select production_select_resolved={cfg:?}");

        let (a, b) = gen_square_ab(0x1694_a000 ^ (n as u64), n);

        let mut keep_alive_base: Vec<Vec<f32>> =
            Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);
        let mut keep_alive_head: Vec<Vec<f32>> =
            Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);

        // warmup: head 側の初回 MSL パイプライン構築コストを吸収する。
        for _ in 0..WARMUP_TRIALS {
            let _ = measure_one_phase_trial(&ctx, &base, &a, &b, n, cfg, &mut keep_alive_base);
            let _ = measure_one_phase_trial(&ctx, &head, &a, &b, n, cfg, &mut keep_alive_head);
        }

        let mut kernel_gpu_base: Vec<f64> = Vec::with_capacity(MEASURED_TRIALS);
        let mut kernel_gpu_head: Vec<f64> = Vec::with_capacity(MEASURED_TRIALS);
        let mut first_output_base: Option<Vec<f32>> = None;
        let mut first_output_head: Option<Vec<f32>> = None;

        for trial in 0..MEASURED_TRIALS {
            // trial 偶奇で計測順を反転し order-bias を相殺する
            // （既存診断テスト群と同じ手法）。
            let (sample_base, sample_head) = if trial % 2 == 0 {
                let sb = measure_one_phase_trial(&ctx, &base, &a, &b, n, cfg, &mut keep_alive_base);
                let sh = measure_one_phase_trial(&ctx, &head, &a, &b, n, cfg, &mut keep_alive_head);
                (sb, sh)
            } else {
                let sh = measure_one_phase_trial(&ctx, &head, &a, &b, n, cfg, &mut keep_alive_head);
                let sb = measure_one_phase_trial(&ctx, &base, &a, &b, n, cfg, &mut keep_alive_base);
                (sb, sh)
            };

            // false-green 防止 (3): フォールバック非経由の fail-closed
            // 検証（ファイル冒頭コメント参照）。
            assert_eq!(
                sample_base.resolved_cfg, cfg,
                "N={n} trial={trial}: base 側で pipeline_for_tile フォール \
                 バックが発生した(requested={cfg:?}, resolved={:?})。性能 \
                 比較の前提が崩れるため中断する",
                sample_base.resolved_cfg
            );
            assert_eq!(
                sample_head.resolved_cfg, cfg,
                "N={n} trial={trial}: head 側で pipeline_for_tile フォール \
                 バックが発生した(requested={cfg:?}, resolved={:?})。性能 \
                 比較の前提が崩れるため中断する",
                sample_head.resolved_cfg
            );

            kernel_gpu_base.push(sample_base.kernel_gpu_secs);
            kernel_gpu_head.push(sample_head.kernel_gpu_secs);

            if trial == 0 {
                first_output_base = keep_alive_base.last().cloned();
                first_output_head = keep_alive_head.last().cloned();
            }
        }

        let out_base =
            first_output_base.expect("MEASURED_TRIALS > 0 のため trial 0 の base 出力は必ず Some");
        let out_head =
            first_output_head.expect("MEASURED_TRIALS > 0 のため trial 0 の head 出力は必ず Some");

        // 正しさ（REQ-2）: head 出力を CPU 参照（`matmul_reference_
        // fma`）と複合判定で fail-closed 検証する（ファイル冒頭
        // 「正しさ・checksum」参照。N=4096 は数十秒〜分単位かかりうる）。
        let mut expected = vec![0.0f32; n * n];
        matmul_reference_fma(&a, &b, &mut expected, n, n, n)
            .expect("CPU 参照実装（matmul_reference_fma）は正方形状に対し常に Ok を返す");
        assert_parity(
            &format!("N={n}: head（ThreadElements）vs CPU 参照"),
            &out_head,
            &expected,
        );

        let checksum_base = checksum_f64(&out_base);
        let checksum_head = checksum_f64(&out_head);
        let identical = bit_identical(&out_base, &out_head);

        let q_base = median_of(&kernel_gpu_base);
        let q_head = median_of(&kernel_gpu_head);
        println!(
            "N={n} pair=te_vs_production_select mode=base kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
            q_base.median * 1e3,
            q_base.q1 * 1e3,
            q_base.q3 * 1e3
        );
        println!(
            "N={n} pair=te_vs_production_select mode=head kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
            q_head.median * 1e3,
            q_head.q1 * 1e3,
            q_head.q3 * 1e3
        );
        // 機械判定用の比は丸めずに記録する（codex-review 指摘・イシュー
        // #1694 PR レビュー是正）。`{ratio:.6}` のように小数 6 桁へ丸めて
        // しまうと、実際の比が 1.0000004 のような境界値でも「1.000000」
        // へ丸められ、`aggregate.py::judge_size` の `r <= 1.00` 判定が
        // 本来 REJECT／undetermined となるべき計測を ADOPT-as-opt-in-
        // candidate と誤判定しうる（`docs/perf/logs/
        // metal-gemm-thread-elements-ab-1694/aggregate.py` §7.1 の事前
        // 登録判定規則を保持するため）。表示用の丸めは集計表描画時
        // （`aggregate.py` の `.4f` 整形）にのみ行い、ここでは Rust の
        // `f64` デフォルト `Display`（最短の往復可能表現）で全精度を
        // 出力する。
        let ratio = q_head.median / q_base.median;
        println!(
            "N={n} head_over_base_kernel_gpu={ratio} base_checksum={checksum_base:e} \
             head_checksum={checksum_head:e} bit_identical={identical}"
        );
    }
}
