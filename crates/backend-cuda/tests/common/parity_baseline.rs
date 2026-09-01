//! parity 非後退契約（イシュー #491・GEMM 性能改善ツリー B-0・親 #490）の
//! ベースライン fixture と検査ユーティリティ。
//!
//! # 位置づけ
//!
//! `wmma_tf32`・`wmma_tf32_opt`・`mma_f16` 系経路は REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）で恒常
//! fail の既知状態にある（#186 由来。`docs/backend-cuda-real-device-testing.md`
//! §5.3 に実測記録済み）。以降の Phase B/C カーネル改修は「parity green」を
//! 受け入れ条件にできないため、本モジュールは「fail 比率・平均絶対誤差が
//! 記録済みベースラインを上回らない」ことを機械検査する**非後退契約**を提供
//! する。判定式・閾値定数（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）
//! 自体は一切変更しない（`fandhe_ai_backend_cpu::parity` を単に呼ぶだけで複製しない）。
//!
//! ## 承認記録（PR #640 codex-review 指摘対応）
//!
//! `baseline_fail_count`/`baseline_mean_abs_diff_ceiling` は
//! `RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`（tolerance 定数）とは
//! **別の概念**であり、両者を混同しない: tolerance 定数は「1 要素あたりの
//! 合否判定式」そのもの、baseline のこの 2 値は「その判定式を通した後の
//! 集計結果（fail 件数・平均絶対誤差）が既知の恒常 fail 状態から悪化して
//! いないか」を見る非後退の上限である（`assert_no_parity_regression` は
//! `report.total`・`fail_count`・`mean_abs_diff` の 3 点比較のみを行い、
//! tolerance 定数自体には一切触れない。`tolerance_constants_are_pinned` が
//! 定数の無断変更を別途 bit 等値で検知する）。
//!
//! この非後退基準（「恒常 fail 経路は green を要求せず、記録済み
//! ベースラインを上回らないことを機械検査する」設計）自体は、本モジュールが
//! 実装するイシュー #491 の受け入れ基準に明記された**ユーザー承認済みの
//! 仕様**であり、本 PR が新設した緩和ではない（イシュー #491 本文
//! 「§1.2 parity 非後退契約」2 番目の受け入れ基準を単一の正として参照する。
//! `docs/perf/cuda-parity-baseline.md` §6「ベースライン更新規約」に
//! 「上方更新（緩和）はユーザー承認必須」と明記し、本表の初期値（下方
//! 更新にすら当たらない初回記録）とは区別している）。
//!
//! `crates/backend-cuda/tests/*.rs` は独立クレート扱いのため、各テスト
//! ファイルから `mod common;` で本モジュールを共有する
//! （`crates/autodiff/tests/common/mod.rs` と同型のパターン）。
//!
//! 正本ドキュメントは `docs/perf/cuda-parity-baseline.md`
//! （ベースライン更新規約・出典・実測環境の記録）。本ファイルは機械検査の
//! 正であり、ドキュメント側と二重管理しない。

#![allow(dead_code)] // テストファイルごとに使う関数・定数が異なるため。

/// 非後退契約の対象経路（Issue #491 が対象とする経路。イシュー #500 で
/// `WmmaTf32Staged` を追加し 4 経路とした）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityPath {
    /// `CudaGemm::run_wmma_tf32`（基本 WMMA(TF32) カーネル）。
    WmmaTf32,
    /// opt カーネル（`__syncthreads()` ベース・共有メモリタイル最適化。
    /// TASK-11.1d・#63）**単独**の非後退ゲート。
    ///
    /// **イシュー #500 でのルーティング変更（PR #678 codex-review P1
    /// 指摘対応）**: `run_wmma_tf32`（公開 API）は #500 で追加された TF32
    /// opt-staged カーネル（cp.async 多段パイプライン）が利用可能かつ
    /// cp.async 16 バイト整列条件（`n%4==0 && k%4==0`）を満たす形状では
    /// staged 経路を最優先で選ぶ（`gemm.rs::run_wmma_tf32` 3 段選択）ため、
    /// 公開 API 経由では opt カーネル単独を強制実行できない形状が生じた。
    /// この行の非後退検査は公開 API を経由せず、`fandhe_ai_backend_cuda::gemm::tests::
    /// wmma_tf32_opt_kernel_parity_does_not_regress`（`src/gemm.rs` 内の
    /// ライブラリ単体テスト。private field `CudaGemm::wmma_tf32_opt`・
    /// private fn `run_wmma_tf32_opt_kernel` へ同一モジュール内から直接
    /// アクセスし、3 段選択を経由せず opt カーネルを強制実行する）で行う
    /// （基本版カーネル専用ゲート `wmma_tf32_basic_kernel_parity_does_not_regress`
    /// と同型のパターン。`tests/parity_nonregression.rs` はこの経路を
    /// 検査しない——公開 API では opt を強制できないため）。
    ///
    /// **イシュー #1106（案 A）で判定方式を形状別に再割り当て済み**: 当初
    /// （PR #678）は `BASELINES` に実測値を持つ 3 形状（64×64×64・
    /// 512×512×512〈seed=0x7A0〉・512×512×4096）のみを本非後退ゲートで
    /// 検査し、残りの opt カーネル固有タイル境界網羅（ブロックタイル
    /// 倍数・非倍数境界・非正方・極小）は `fandhe_ai_backend_cuda::
    /// gemm::tests::wmma_tf32_opt_kernel_matches_reference_across_shapes`・
    /// `wmma_tf32_opt_kernel_k4096_stress`（いずれも `src/gemm.rs`）が
    /// `assert_no_parity_regression` ではなく `fandhe_ai_backend_cpu::assert_parity`
    /// （厳密ゼロ fail 判定）で検査していた。GB10 実機実測（#1106
    /// reopen コメント・診断ダンプ `wmma_tf32_opt_kernel_parity_diagnostic_dump_issue_1106`）
    /// で、64x64x64・512x512x4096 を含む 8/9 ケースが TF32 丸めに由来する
    /// 既知の非ゼロ fail_count を持ち（opt/basic bit-identical・sm_86/GB10
    /// 世代間差分なし。`docs/perf/cuda-tensor-core-tolerance-opt-remeasurement.md`
    /// §5〜§7）、厳密ゼロ fail 判定はそもそも成立しない（カーネルのバグでは
    /// なくテスト設計の不整合）ことが判明した。ゼロ fail が実際に成立する
    /// のは 1x1x1（sub-K-tile）のみ。本 PR（#1106 案 A）で、非ゼロ fail が
    /// 判明した 6 形状（128x128x128・512x512x512〈別シード〉・63x65x33・
    /// 65x63x17・64x96x256・4096x4096x4096）を `BASELINES` へ実測値付きで
    /// 追加し、`wmma_tf32_opt_kernel_matches_reference_across_shapes`・
    /// `wmma_tf32_opt_kernel_k4096_stress` からは削除した（1x1x1 のみ残す）。
    /// これにより本非後退ゲートが 9 形状中 8 形状（1x1x1 を除く全て）を
    /// カバーする。tolerance 定数（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）
    /// は変更していない（ユーザー承認 2026-09-02。詳細は
    /// `docs/perf/cuda-parity-baseline.md` §9.11）。
    WmmaTf32Opt,
    /// opt-staged カーネル（cp.async 多段パイプライン・fragment 先読み。
    /// イシュー #500）単独の非後退ゲート。
    ///
    /// `run_wmma_tf32`（公開 API）は staged カーネルが利用可能かつ整列
    /// 条件を満たす形状では staged 経路を最優先で選ぶため、`WmmaTf32Opt`
    /// とは逆に、この経路は公開 API 経由（`tests/parity_nonregression.rs::
    /// check_wmma_tf32_staged_baseline`）で正しく強制実行できる（事前に
    /// `wmma_tf32_staged_available()` を assert し、整列条件を満たす形状を
    /// 選ぶ）。
    ///
    /// 現時点では実機未到達のため記録値は未確定
    /// （`ParityBaseline::baseline_provenance_unconfirmed == true`）で
    /// あり、fail-closed 契約（`assert_no_parity_regression`）により実機
    /// 実行のたびに fail し続ける。実機再測定でのフォローアップはイシュー
    /// #502（Phase B 完了時点の f32/f16 再計測）へ引き継ぐ。
    WmmaTf32Staged,
    /// `CudaMmaGemm::run_f16`（`mma.sync`/`ldmatrix`/`cp.async` 経路）。
    MmaF16,
    /// `CudaWmmaGemm::run_f16`（基本 WMMA(f16) カーネル）。
    ///
    /// **イシュー #1106（GB10 全件洗い出し）で追加**: `tests/cpu_cuda_wmma_parity.rs::
    /// wmma_f16_k4096_stress`（256×256×4096・seed=8888）は f16→f32→
    /// `matmul_reference_fma`→f16 丸め→f32 の量子化込み参照でも K=4096
    /// 蓄積により既知の tail 超過（`docs/backend-cuda-real-device-testing.md`
    /// §5.3）を持つ。本体テスト自体は `assert_parity`（REQ-2 受け入れ条件）
    /// を維持したまま、非後退監視を `wmma_f16_k4096_stress_non_regression`
    /// として別テストで併設する（`ParityPath::MmaF16`・`tensor_core_parity_record`
    /// tf32 行と同型の「元の受け入れ条件は置き換えない」設計。
    /// `docs/perf/cuda-parity-baseline.md` §10.4 参照）。
    WmmaF16,
    /// `CudaWmmaGemm::run_f16_opt`（opt WMMA(f16) カーネル）。
    ///
    /// `WmmaF16` と同じ理由で追加（`tests/gemm_wmma_f16_opt.rs::
    /// wmma_f16_opt_k4096_stress`。256×256×4096・seed=8889 の非後退監視を
    /// `wmma_f16_opt_k4096_stress_non_regression` として併設する）。
    WmmaF16Opt,
}

impl std::fmt::Display for ParityPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ParityPath::WmmaTf32 => "wmma_tf32",
            ParityPath::WmmaTf32Opt => "wmma_tf32_opt",
            ParityPath::WmmaTf32Staged => "wmma_tf32_staged",
            ParityPath::MmaF16 => "mma_f16",
            ParityPath::WmmaF16 => "wmma_f16",
            ParityPath::WmmaF16Opt => "wmma_f16_opt",
        };
        f.write_str(s)
    }
}

/// 経路・形状ごとの記録済みベースライン 1 行。
///
/// `total`/`baseline_fail_count`/`baseline_mean_abs_diff_ceiling` は
/// `docs/backend-cuda-real-device-testing.md` §5.3（DGX Spark GB10・sm_121
/// 実機実測）からの転記であり、推定値は含まない。各行の出典・実測日は
/// `docs/perf/cuda-parity-baseline.md` の表に対応する。
#[derive(Debug, Clone, Copy)]
pub struct ParityBaseline {
    pub path: ParityPath,
    pub context: &'static str,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub seed: u64,
    pub total: usize,
    pub baseline_fail_count: usize,
    /// 表記丸め対応の天井値（`docs/perf/cuda-parity-baseline.md` §「表記丸め
    /// 対応」）。§5.3 記録値の最終桁を切り上げた値を格納する（許容誤差
    /// 〈tolerance 定数〉の緩和ではなく、表記丸め誤差の吸収のみ。判定式・
    /// 閾値定数は一切変更しない）。
    pub baseline_mean_abs_diff_ceiling: f64,
    /// この行の記録値が「意図したカーネル版で実測されたか」の provenance
    /// 保証が取れていない場合に `true`（PR #640 codex-review 指摘対応・
    /// イシュー #491。PR #678 codex-review P1 指摘対応で
    /// `basic_kernel_baseline_unconfirmed` から改名: `WmmaTf32Staged`
    /// 行にも同じ「実機未到達で数値未確定」の意味で使うため、`WmmaTf32`
    /// 〈基本版〉専用を想起させる旧名を一般化した）。
    ///
    /// **fail-closed 契約（PR #640 codex-review P1 再指摘対応。旧
    /// `pending_basic_remeasurement` からの改名）**: `true` の行は
    /// `assert_no_parity_regression` が判定を試みず**必ず panic する**
    /// （黙って skip しない。旧実装は provenance 不確実な行をスキップし
    /// 「実機テストは正常終了と shape 一致だけで通過する」状態を許してい
    /// たが、これは非後退ゲートが機能していないのに green に見える回帰
    /// だった）。`WmmaTf32`（基本版）行は記録元テストが opt カーネル
    /// 可用性を確認せず `run_wmma_tf32` を呼んだため、記録値が opt
    /// カーネルの実測結果である可能性が高く、基本版カーネル専用の実測経路
    /// （`fandhe_ai_backend_cuda::gemm::tests::wmma_tf32_basic_kernel_parity_does_not_regress`。
    /// `src/gemm.rs` 内のライブラリ単体テスト）と比較すると、カーネルが
    /// 異なり fail_count・mean_abs_diff の分布も異なるため、後退していない
    /// 変更を拒否する false-fail と、基本版の後退を見逃す false-pass の
    /// 両方が生じ非後退ゲートとして成立しない。`WmmaTf32Staged` 行は
    /// PR #678（イシュー #500）で staged カーネルを追加した時点では実機
    /// 未到達のため記録値そのものが存在しない（推定値の捏造をしない。
    /// `docs/perf/cuda-parity-baseline.md` §6「ベースライン更新規約」）。
    /// 実機再測定でこの provenance 不確実性を解消したら `false` へ更新し、
    /// 各経路専用の実測値へ差し替える。それまでの間、該当行を含む実機必須
    /// テストは実行のたびに fail し続ける契約であり、これは意図した挙動
    /// である（本リポで既知の受け入れ済み状態。
    /// `docs/backend-cuda-real-device-testing.md` §5.3・§7 参照）。
    pub baseline_provenance_unconfirmed: bool,
}

/// 記録済みベースライン一覧（`docs/perf/cuda-parity-baseline.md` の表・7 行）。
///
/// 出典: `docs/backend-cuda-real-device-testing.md` §5.3（DGX Spark GB10・
/// sm_121・2026 年 8 月時点実測）。§5.3 の記録は `assert_parity` が最初の
/// fail で panic する契約のため「各テストで最初に fail した (形状, シード)
/// の値」のみが実測されている。未計測形状の行追加は実機実測とセットでのみ
/// 行う（推定値の捏造をしない。`docs/perf/cuda-parity-baseline.md`
/// 「ベースライン更新規約」）。
pub static BASELINES: &[ParityBaseline] = &[
    // wmma_tf32: 形状網羅の最小ケース（32x32x32、seed=2000）。
    // `tests/gemm_wmma_tf32.rs::wmma_tf32_matches_reference_across_shapes` の
    // 先頭ケース（`assert_wmma_tf32_parity(&gemm, ctx, 2000 + 0, 32, 32, 32)`）。
    //
    // 既知の限界は解消済み（イシュー #1106・GB10 実機実測）: 記録元
    // （`run_wmma_tf32`。opt 可用なら opt 優先）が実際に opt カーネルを
    // 計測していた可能性を懸念していたが、基本版カーネル専用の単体テスト
    // （`fandhe_ai_backend_cuda::gemm::tests::wmma_tf32_basic_kernel_parity_does_not_regress`。
    // `src/gemm.rs`。private field 経由で 3 段選択を経由せず基本版
    // カーネルを直接強制実行）を DGX Spark GB10（sm_121・CUDA 13.0）実機で
    // release 2 回実行し、いずれも fail_count=154/1024・
    // mean_abs_diff=3.697936e-4 の完全一致を確認した（実行間の値の安定性
    // 確認込み。#726 の前例と同じ手順）。この実測値は記録済みの値
    // （opt 実測の疑いがあったもの）と一致しており、基本版・opt カーネルが
    // 同一の parity 分布を持つという #995 の GB10 実測結果（basic/opt/staged
    // 数値完全一致）を裏付ける。よって既存の記録値をそのまま確定値として
    // 採用し `baseline_provenance_unconfirmed: false` へ更新する（推定値の
    // 記入ではなく実測による確認。詳細は
    // `docs/backend-cuda-real-device-testing.md` §5.3・
    // `docs/perf/cuda-parity-baseline.md` §3「既知の限界」参照）。
    ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "wmma_tf32 32x32x32 seed=2000",
        m: 32,
        n: 32,
        k: 32,
        seed: 2000,
        total: 32 * 32,
        baseline_fail_count: 154,
        baseline_mean_abs_diff_ceiling: 3.699e-4,
        // イシュー #1106・GB10 実機実測で基本版カーネル専用の確定測定を
        // 完了（上記コメント参照）。provenance 不確実性は解消済み。
        baseline_provenance_unconfirmed: false,
    },
    // wmma_tf32: K=4096 ストレスケース先頭（256x256x4096、seed=8888）。
    // `tests/gemm_wmma_tf32.rs::wmma_tf32_k4096_stress_poc_v2_5` の先頭呼出し。
    //
    // 既知の限界は解消済み（イシュー #1106）: 上の 32x32x32 行と同じ手順で
    // GB10 実機 release 2 回実行し、いずれも fail_count=10647/65536・
    // mean_abs_diff=4.476030e-3 の完全一致を確認した。記録済みの値と一致。
    ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "wmma_tf32 256x256x4096 seed=8888 (PoC-v2-5 stress)",
        m: 256,
        n: 256,
        k: 4096,
        seed: 8888,
        total: 256 * 256,
        baseline_fail_count: 10647,
        baseline_mean_abs_diff_ceiling: 4.477e-3,
        // イシュー #1106・GB10 実機実測で確定（上記コメント参照）。
        baseline_provenance_unconfirmed: false,
    },
    // wmma_tf32: 案 A（イシュー #1106・GB10 全件洗い出し）で追加した
    // 形状網羅の残り 7 ケース。
    //
    // 背景: `tests/gemm_wmma_tf32.rs::wmma_tf32_matches_reference_across_shapes`・
    // `wmma_tf32_k4096_stress_poc_v2_5` は `assert_parity`（厳密ゼロ fail
    // 判定）を使っていたが、GB10 実機実測（2026-09-02 の診断ダンプ
    // `wmma_tf32_parity_diagnostic_dump_issue_1106`。修正確定に伴い削除
    // 済み）で 64x64x64・128x128x128・512x512x512・64x96x128・
    // 17x23x19・33x31x65・512x512x4096（seed=0xFACADE）の 7 形状すべてが
    // 非ゼロ fail_count を持つことが判明した（`wmma_tf32_opt`・
    // `wmma_tf32_staged` と同じ TF32 丸めの既知の恒常特性）。ゼロ fail が
    // 実際に成立するのは 1x1x1（sub-K-tile）のみであり、上記 2 テストから
    // 当該形状を削除したうえで本 7 行をここへ移し、
    // `tests/gemm_wmma_tf32.rs::wmma_tf32_routed_path_baselines_do_not_regress`
    // （公開 API `run_wmma_tf32` 経由の非後退監視）の対象に含める。
    // ceiling は §4「表記丸め対応」と同じ規約（表示 4 桁の最終桁 +1）で
    // 算出した。
    ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "wmma_tf32 64x64x64 seed=2001 (#1106 diagnostic)",
        m: 64,
        n: 64,
        k: 64,
        seed: 2001,
        total: 64 * 64,
        baseline_fail_count: 687,
        baseline_mean_abs_diff_ceiling: 5.542e-4,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "wmma_tf32 128x128x128 seed=2002 (#1106 diagnostic)",
        m: 128,
        n: 128,
        k: 128,
        seed: 2002,
        total: 128 * 128,
        baseline_fail_count: 2559,
        baseline_mean_abs_diff_ceiling: 7.858e-4,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "wmma_tf32 512x512x512 seed=2003 (#1106 diagnostic)",
        m: 512,
        n: 512,
        k: 512,
        seed: 2003,
        total: 512 * 512,
        baseline_fail_count: 42550,
        baseline_mean_abs_diff_ceiling: 1.565e-3,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "wmma_tf32 64x96x128 seed=2004 (#1106 diagnostic)",
        m: 64,
        n: 96,
        k: 128,
        seed: 2004,
        total: 64 * 96,
        baseline_fail_count: 1027,
        baseline_mean_abs_diff_ceiling: 7.895e-4,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "wmma_tf32 17x23x19 seed=2006 (#1106 diagnostic)",
        m: 17,
        n: 23,
        k: 19,
        seed: 2006,
        total: 17 * 23,
        baseline_fail_count: 52,
        baseline_mean_abs_diff_ceiling: 3.059e-4,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "wmma_tf32 33x31x65 seed=2007 (#1106 diagnostic)",
        m: 33,
        n: 31,
        k: 65,
        seed: 2007,
        total: 33 * 31,
        baseline_fail_count: 171,
        baseline_mean_abs_diff_ceiling: 5.599e-4,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "wmma_tf32 512x512x4096 seed=0xFACADE (#1106 diagnostic)",
        m: 512,
        n: 512,
        k: 4096,
        seed: 0xFACADE,
        total: 512 * 512,
        baseline_fail_count: 42688,
        baseline_mean_abs_diff_ceiling: 4.464e-3,
        baseline_provenance_unconfirmed: false,
    },
    // wmma_tf32_opt: 512x512x512。記録元 `tensor_core_real_device.rs::
    // tensor_core_parity_record` TF32 部分は事前に `wmma_tf32_opt_available()`
    // を assert してから `gemm.run_wmma_tf32(...)` を計測している
    // （seed=0x7A0）。記録時点（イシュー #500 の staged カーネル追加前）は
    // opt 可用性さえ確認すれば `run_wmma_tf32` が実際に opt 経路を通ったが、
    // #500 以降 `run_wmma_tf32` は整列形状で staged を最優先するため、この
    // 記録値の非後退検査は公開 API 経由では行わない（PR #678 codex-review
    // P1 指摘対応）: `fandhe_ai_backend_cuda::gemm::tests::
    // wmma_tf32_opt_kernel_parity_does_not_regress`（`src/gemm.rs`）が
    // private field 経由で opt カーネルを直接強制実行して検査する
    // （`ParityPath::WmmaTf32Opt` ドキュメンテーションコメント参照）。
    ParityBaseline {
        path: ParityPath::WmmaTf32Opt,
        context: "wmma_tf32_opt 512x512x512 seed=0x7A0 (tensor_core_parity_record)",
        m: 512,
        n: 512,
        k: 512,
        seed: 0x7A0,
        total: 512 * 512,
        baseline_fail_count: 42493,
        baseline_mean_abs_diff_ceiling: 1.575e-3,
        // opt 可用性を計測前に assert 済み（記録元コメント参照）。
        // provenance 不確実性なし。
        baseline_provenance_unconfirmed: false,
    },
    // wmma_tf32_opt: 形状網羅の先頭ケース（64x64x64、seed=3000）。
    // `tests/gemm_wmma_tf32_opt.rs::wmma_tf32_opt_matches_reference_across_shapes`
    // の先頭ケース（`assert_wmma_tf32_opt_parity(&gemm, ctx, 3000 + 0, 64, 64, 64)`）。
    //
    // 記録時点（#500 の staged カーネル追加前）は `gemm_wmma_tf32_opt.rs` が
    // opt 経路専用だったため provenance 不確実性はないが、上記 512x512x512
    // 行と同じ理由で非後退検査自体は `gemm.rs` の内部テストへ移設済み
    // （このコメントは記録値の provenance についてのみ言及し、現在の検査
    // 経路は `ParityPath::WmmaTf32Opt` ドキュメンテーションコメント参照）。
    ParityBaseline {
        path: ParityPath::WmmaTf32Opt,
        context: "wmma_tf32_opt 64x64x64 seed=3000",
        m: 64,
        n: 64,
        k: 64,
        seed: 3000,
        total: 64 * 64,
        baseline_fail_count: 699,
        baseline_mean_abs_diff_ceiling: 5.677e-4,
        baseline_provenance_unconfirmed: false,
    },
    // wmma_tf32_opt: K=4096 ストレスケース先頭（512x512x4096、seed=0xC0FFEE）。
    // `tests/gemm_wmma_tf32_opt.rs::wmma_tf32_opt_k4096_stress` の先頭呼出し。
    // 上の 64x64x64 行と同じ理由で provenance 不確実性なし・検査経路は
    // `gemm.rs` の内部テストへ移設済み。
    ParityBaseline {
        path: ParityPath::WmmaTf32Opt,
        context: "wmma_tf32_opt 512x512x4096 seed=0xC0FFEE",
        m: 512,
        n: 512,
        k: 4096,
        seed: 0xC0FFEE,
        total: 512 * 512,
        baseline_fail_count: 43019,
        baseline_mean_abs_diff_ceiling: 4.464e-3,
        baseline_provenance_unconfirmed: false,
    },
    // wmma_tf32_opt: 案 A（イシュー #1106）で追加した形状網羅の残り 6 ケース。
    //
    // 背景: `fandhe_ai_backend_cuda::gemm::tests::
    // wmma_tf32_opt_kernel_matches_reference_across_shapes`／
    // `wmma_tf32_opt_kernel_k4096_stress` は `assert_parity`（厳密ゼロ fail
    // 判定）を使っていたが、GB10 実機実測（#1106 reopen コメント・診断
    // ダンプ `wmma_tf32_opt_kernel_parity_diagnostic_dump_issue_1106`）で
    // 128x128x128・512x512x512（別シード）・63x65x33・65x63x17・
    // 64x96x256・4096x4096x4096 の 6 形状すべてが非ゼロ fail_count を持つ
    // ことが判明した（TF32 丸めによる既知の恒常特性。opt/basic カーネルで
    // bit-identical、sm_86/GB10 世代間でも差分なし。
    // `docs/perf/cuda-tensor-core-tolerance-opt-remeasurement.md` §5〜§7）。
    // ゼロ fail が実際に成立するのは 1x1x1（sub-K-tile）のみであり、他は
    // `assert_parity` を適用すること自体が誤りだった（テスト設計の不整合。
    // カーネル側の数値バグではない）。本 6 行は上記 2 テストから当該
    // 形状を削除したうえでここへ移し、`wmma_tf32_opt_kernel_parity_does_not_regress`
    // （baseline 非後退方式）の対象に含める。
    //
    // 実測: DGX Spark GB10（sm_121・CUDA 13.0）、GPU アイドル
    // （`nvidia-smi --query-gpu=utilization.gpu` 0%）を確認したうえで
    // 2026-09-01 に初回計測、2026-09-02 に GPU アイドル再確認のうえ再計測し、
    // 2 回とも fail_count・max_abs_diff・max_rel_err・mean_abs_diff が
    // 完全一致することを確認した（`docs/perf/cuda-parity-baseline.md` §9.11
    // 参照）。ceiling は表示 4 桁の最終桁を +1 した天井値（既存行と同じ
    // 表記丸め対応の規約。`ParityBaseline::baseline_mean_abs_diff_ceiling`
    // ドキュメンテーションコメント参照）。
    ParityBaseline {
        path: ParityPath::WmmaTf32Opt,
        context: "wmma_tf32_opt 128x128x128 seed=0xBB9 (#1106 diagnostic)",
        m: 128,
        n: 128,
        k: 128,
        seed: 0xBB9,
        total: 128 * 128,
        baseline_fail_count: 2638,
        baseline_mean_abs_diff_ceiling: 7.776e-4,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32Opt,
        context: "wmma_tf32_opt 512x512x512 seed=0xBBA (#1106 diagnostic)",
        m: 512,
        n: 512,
        k: 512,
        seed: 0xBBA,
        total: 512 * 512,
        baseline_fail_count: 42799,
        baseline_mean_abs_diff_ceiling: 1.569e-3,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32Opt,
        context: "wmma_tf32_opt 63x65x33 seed=0xBBB (#1106 diagnostic)",
        m: 63,
        n: 65,
        k: 33,
        seed: 0xBBB,
        total: 63 * 65,
        baseline_fail_count: 698,
        baseline_mean_abs_diff_ceiling: 3.944e-4,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32Opt,
        context: "wmma_tf32_opt 65x63x17 seed=0xBBC (#1106 diagnostic)",
        m: 65,
        n: 63,
        k: 17,
        seed: 0xBBC,
        total: 65 * 63,
        baseline_fail_count: 635,
        baseline_mean_abs_diff_ceiling: 2.778e-4,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32Opt,
        context: "wmma_tf32_opt 64x96x256 seed=0xBBD (#1106 diagnostic)",
        m: 64,
        n: 96,
        k: 256,
        seed: 0xBBD,
        total: 64 * 96,
        baseline_fail_count: 967,
        baseline_mean_abs_diff_ceiling: 1.118e-3,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32Opt,
        context: "wmma_tf32_opt 4096x4096x4096 seed=0xBEEF (#1106 diagnostic)",
        m: 4096,
        n: 4096,
        k: 4096,
        seed: 0xBEEF,
        total: 4096 * 4096,
        baseline_fail_count: 2725617,
        baseline_mean_abs_diff_ceiling: 4.454e-3,
        baseline_provenance_unconfirmed: false,
    },
    // wmma_tf32_staged: K=4096 ストレスケース（512x512x4096）。既存
    // wmma_tf32_opt ストレス行（直上、seed=0xC0FFEE）と同形状に揃え、
    // opt 対比の改善量を直接比較できるようにする（PR #678 codex-review P1
    // 指摘対応・イシュー #500）。#500 時点では実機未到達のためプレース
    // ホルダ + `baseline_provenance_unconfirmed: true` の fail-closed 行
    // だったが、イシュー #726 で DGX Spark GB10 実機（コミット 06b24b4・
    // 2026-08-19）にて `parity_baselines_do_not_regress` の staged 検査
    // （公開 API `run_wmma_tf32` 経由・`wmma_tf32_staged_available()`
    // assert 済み）を release/debug 各 2 回実行し、4 回とも同一の
    // fail_count=43019/262144・mean_abs_diff=4.463436e-3 を確認して確定値
    // へ差し替えた。値は直上の wmma_tf32_opt 同形状行と一致する（staged は
    // opt と同一の FMA 契約・積和順序を保つ cp.async 二重バッファ版であり、
    // 数値結果が変わらないことの実測裏付け）。ceiling は 4 有効桁表記
    // 4.463e-3 の最終桁切り上げ天井値（`docs/perf/cuda-parity-baseline.md`
    // §4）。
    ParityBaseline {
        path: ParityPath::WmmaTf32Staged,
        context: "wmma_tf32_staged 512x512x4096 seed=0xC0FFEE",
        m: 512,
        n: 512,
        k: 4096,
        seed: 0xC0FFEE,
        total: 512 * 512,
        baseline_fail_count: 43019,
        baseline_mean_abs_diff_ceiling: 4.464e-3,
        baseline_provenance_unconfirmed: false,
    },
    // wmma_tf32_staged: 案 A（イシュー #1106・GB10 全件洗い出し）で追加
    // した形状網羅の残り 7 ケース。
    //
    // 背景: `tests/gemm_wmma_tf32_staged.rs::wmma_tf32_staged_matches_reference_across_shapes`・
    // `wmma_tf32_staged_k4096_stress` は `assert_parity`（厳密ゼロ fail
    // 判定）を使っていたが、GB10 実機実測（2026-09-02 の診断ダンプ
    // `wmma_tf32_staged_parity_diagnostic_dump_issue_1106`。修正確定に
    // 伴い削除済み）で 64x64x64・128x128x128・512x512x512・60x68x36・
    // 68x60x20・64x96x256・4096x4096x4096（seed=0xBEEF）の 7 形状すべてが
    // 非ゼロ fail_count を持つことが判明した（`wmma_tf32_opt` と同じ
    // TF32 丸めの既知の恒常特性）。ゼロ fail が実際に成立するのは 1x1x1
    // （sub-K-tile）のみであり、上記 2 テストから当該形状を削除したうえで
    // 本 7 行をここへ移し、`tests/parity_nonregression.rs::
    // parity_baselines_do_not_regress`（`check_wmma_tf32_staged_baseline`。
    // 公開 API `run_wmma_tf32` 経由。既存の走査対象でありコード変更不要で
    // 対象拡大した）の対象に含める。ceiling は §4 と同じ規約で算出した。
    ParityBaseline {
        path: ParityPath::WmmaTf32Staged,
        context: "wmma_tf32_staged 64x64x64 seed=0xFA0 (#1106 diagnostic)",
        m: 64,
        n: 64,
        k: 64,
        seed: 0xFA0,
        total: 64 * 64,
        baseline_fail_count: 633,
        baseline_mean_abs_diff_ceiling: 5.426e-4,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32Staged,
        context: "wmma_tf32_staged 128x128x128 seed=0xFA1 (#1106 diagnostic)",
        m: 128,
        n: 128,
        k: 128,
        seed: 0xFA1,
        total: 128 * 128,
        baseline_fail_count: 2631,
        baseline_mean_abs_diff_ceiling: 7.834e-4,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32Staged,
        context: "wmma_tf32_staged 512x512x512 seed=0xFA2 (#1106 diagnostic)",
        m: 512,
        n: 512,
        k: 512,
        seed: 0xFA2,
        total: 512 * 512,
        baseline_fail_count: 42782,
        baseline_mean_abs_diff_ceiling: 1.572e-3,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32Staged,
        context: "wmma_tf32_staged 60x68x36 seed=0xFA3 (#1106 diagnostic)",
        m: 60,
        n: 68,
        k: 36,
        seed: 0xFA3,
        total: 60 * 68,
        baseline_fail_count: 691,
        baseline_mean_abs_diff_ceiling: 4.096e-4,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32Staged,
        context: "wmma_tf32_staged 68x60x20 seed=0xFA4 (#1106 diagnostic)",
        m: 68,
        n: 60,
        k: 20,
        seed: 0xFA4,
        total: 68 * 60,
        baseline_fail_count: 620,
        baseline_mean_abs_diff_ceiling: 2.973e-4,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32Staged,
        context: "wmma_tf32_staged 64x96x256 seed=0xFA5 (#1106 diagnostic)",
        m: 64,
        n: 96,
        k: 256,
        seed: 0xFA5,
        total: 64 * 96,
        baseline_fail_count: 1008,
        baseline_mean_abs_diff_ceiling: 1.122e-3,
        baseline_provenance_unconfirmed: false,
    },
    ParityBaseline {
        path: ParityPath::WmmaTf32Staged,
        context: "wmma_tf32_staged 4096x4096x4096 seed=0xBEEF (#1106 diagnostic)",
        m: 4096,
        n: 4096,
        k: 4096,
        seed: 0xBEEF,
        total: 4096 * 4096,
        baseline_fail_count: 2725617,
        baseline_mean_abs_diff_ceiling: 4.454e-3,
        baseline_provenance_unconfirmed: false,
    },
    // mma_f16: K=4096 ストレスケース（256x256x4096、seed=9999）。
    // `tests/cpu_cuda_mma_parity.rs::mma_f16_k4096_stress` の先頭呼出し
    // （`assert_mma_f16_parity(&gemm, ctx, 9999, 256, 256, 4096)`）。
    ParityBaseline {
        path: ParityPath::MmaF16,
        context: "mma_f16 256x256x4096 seed=9999",
        m: 256,
        n: 256,
        k: 4096,
        seed: 9999,
        total: 256 * 256,
        baseline_fail_count: 101,
        baseline_mean_abs_diff_ceiling: 7.647e-5,
        // `run_f16` は基本/opt の分岐を持たないため provenance 不確実性なし。
        baseline_provenance_unconfirmed: false,
    },
    // wmma_f16: K=4096 ストレスケース（256x256x4096、seed=8888）。
    // `tests/cpu_cuda_wmma_parity.rs::wmma_f16_k4096_stress` の唯一の
    // 呼出し（`assert_wmma_f16_parity(&gemm, ctx, 8888, 256, 256, 4096)`）。
    // イシュー #1106（GB10 全件洗い出し）で追加。実測: DGX Spark GB10
    // （sm_121・CUDA 13.0）・GPU アイドル・直列実行（`--test-threads=1`）
    // 2026-09-02。
    ParityBaseline {
        path: ParityPath::WmmaF16,
        context: "wmma_f16 256x256x4096 seed=8888",
        m: 256,
        n: 256,
        k: 4096,
        seed: 8888,
        total: 256 * 256,
        baseline_fail_count: 99,
        baseline_mean_abs_diff_ceiling: 7.563e-5,
        // `run_f16`（基本 WMMA）は分岐を持たないため provenance 不確実性
        // なし。
        baseline_provenance_unconfirmed: false,
    },
    // wmma_f16_opt: K=4096 ストレスケース（256x256x4096、seed=8889）。
    // `tests/gemm_wmma_f16_opt.rs::wmma_f16_opt_k4096_stress` の唯一の
    // 呼出し（`assert_wmma_f16_opt_parity(&gemm, ctx, 8889, 256, 256,
    // 4096)`）。イシュー #1106（GB10 全件洗い出し）で追加。実測環境は
    // 上記 `wmma_f16` 行と同一（2026-09-02）。
    ParityBaseline {
        path: ParityPath::WmmaF16Opt,
        context: "wmma_f16_opt 256x256x4096 seed=8889",
        m: 256,
        n: 256,
        k: 4096,
        seed: 8889,
        total: 256 * 256,
        baseline_fail_count: 81,
        baseline_mean_abs_diff_ceiling: 7.628e-5,
        baseline_provenance_unconfirmed: false,
    },
];

/// 非後退判定: `report` がベースラインを上回っていないことを assert する。
///
/// `fandhe_ai_backend_cpu::compare` が返す `CompareReport` をそのまま使い、判定式
/// （複合判定の合否）を複製・改変しない（`.claude/rules/security.md` A08
/// 「判定の迂回経路を作らない」）。
///
/// 3 点を fail-closed で検査する:
/// 1. `report.total == baseline.total`（形状・比較対象のずれの検出。
///    fixture の `total` 定義とテスト側の実測形状がずれていた場合に
///    「たまたま非後退に見える」誤判定を防ぐ）
/// 2. `report.fail_count <= baseline.baseline_fail_count`
/// 3. `report.mean_abs_diff <= baseline.baseline_mean_abs_diff_ceiling`
///
/// # Panics
///
/// いずれかの検査に失敗した場合、`fandhe_ai_backend_cpu::assert_parity` と同水準の
/// 診断情報（fail_count・mean_abs_diff 等の分布統計）を付けて panic する。
/// `baseline.baseline_provenance_unconfirmed == true` の場合は上記 3 点の
/// 比較を行わず、実機再測定が必要である旨のメッセージで即座に panic する
/// （fail-closed。`ParityBaseline::baseline_provenance_unconfirmed` の
/// ドキュメンテーションコメント参照。PR #640 codex-review P1 再指摘対応:
/// 呼び出し側で判定をスキップさせず、この関数自身が迂回不能な唯一の
/// 判定経路として fail-closed を保証する）。
#[track_caller]
pub fn assert_no_parity_regression(
    context: &str,
    report: &fandhe_ai_backend_cpu::CompareReport,
    baseline: &ParityBaseline,
) {
    assert!(
        !baseline.baseline_provenance_unconfirmed,
        "{context}: parity 非後退契約 FAIL — この行は基本版カーネル専用の \
         確定ベースラインが未整備です（baseline_provenance_unconfirmed \
         == true）。provenance 不確実性により非後退判定に使えないため、\
         黙って skip せず fail-closed で失敗させています。CUDA 実機で \
         基本版カーネル単独を再測定し、確定値を記録したうえで \
         baseline_provenance_unconfirmed: false へ更新してください \
         （推定値の記入は禁止。docs/perf/cuda-parity-baseline.md \
         §「既知の限界」・§「ベースライン更新規約」参照。実測 fail_count=\
         {}/{}, mean_abs_diff={:.6e} は参考値であり合否判定には使っていません）",
        report.fail_count, report.total, report.mean_abs_diff,
    );
    assert_eq!(
        report.total, baseline.total,
        "{context}: 比較対象の要素数が baseline({}) と一致しません（形状・\
         比較対象がずれている可能性があります。baseline={:?}）",
        baseline.total, baseline
    );
    assert!(
        report.fail_count <= baseline.baseline_fail_count,
        "{context}: parity 非後退契約 FAIL — fail_count が後退しました \
         (actual={}/{}, baseline={}/{}, mean_abs_diff={:.6e}, \
         baseline_mean_abs_diff_ceiling={:.6e})",
        report.fail_count,
        report.total,
        baseline.baseline_fail_count,
        baseline.total,
        report.mean_abs_diff,
        baseline.baseline_mean_abs_diff_ceiling,
    );
    assert!(
        report.mean_abs_diff <= baseline.baseline_mean_abs_diff_ceiling,
        "{context}: parity 非後退契約 FAIL — mean_abs_diff が後退しました \
         (actual={:.6e}, baseline_ceiling={:.6e}, fail_count={}/{}, \
         baseline_fail_count={})",
        report.mean_abs_diff,
        baseline.baseline_mean_abs_diff_ceiling,
        report.fail_count,
        report.total,
        baseline.baseline_fail_count,
    );
}

/// tolerance 定数（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）の
/// 無断変更を機械検知する（受け入れ基準 3）。
///
/// bit 等値で assert する: 浮動小数点定数の意図しない再定義（丸め違い等）
/// も検出対象に含めるため、`==` による厳密比較を用いる。
#[track_caller]
pub fn assert_tolerance_constants_pinned() {
    assert_eq!(
        fandhe_ai_backend_cpu::RELATIVE_TOLERANCE,
        1e-3,
        "RELATIVE_TOLERANCE が無断変更されています（ガードレール閾値の変更は \
         ユーザー承認必須。.claude/rules/security.md A08）"
    );
    assert_eq!(
        fandhe_ai_backend_cpu::ABSOLUTE_RESCUE_THRESHOLD,
        1e-5,
        "ABSOLUTE_RESCUE_THRESHOLD が無断変更されています（ガードレール閾値の \
         変更はユーザー承認必須。.claude/rules/security.md A08）"
    );
}
