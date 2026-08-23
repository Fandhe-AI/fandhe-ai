//! バックエンド非依存の interleaved A/B・安定性計測ユーティリティ（イシュー #746）。
//!
//! `crate::protocol::run`（計測コア。TASK-8.1a）はそのまま「1 ラウンド分の
//! 計測」として使い、本モジュールは複数ラウンドの計測を挟み込む上位プロトコル
//! を提供する。位置づけ:
//!
//! - `guardrail`／`self-repair` の検証ゲートが依存する [`crate::run`]・
//!   [`crate::MeasurementConfig`] のセマンティクスは変更しない
//!   （`crate::protocol` モジュールドキュメント参照）。本モジュールは
//!   `protocol::run` を呼ぶ側として新規追加するのみ。
//! - 動機は Metal 実機（M4 Max）での tgid swizzle（#540）A/B 計測が
//!   サーマル・GPU クロック（DVFS）挙動の系統誤差で「劣化中央値 5% 以内」の
//!   判定が成立しなかったこと（#746 イシュー本文）。checkout 切替方式（base/head
//!   計測が時間的に分離されドリフトがそのまま系統誤差になる）ではなく、
//!   同一プロセス内で base/head の 2 インスタンスを構築し
//!   ラウンド交互（順序反転）で interleaved 計測することでドリフトを
//!   ラウンド間で相殺する（CUDA 側 `CudaMmaGemm::new_with_swizzle`
//!   〈`crates/backend-cuda/examples/gemm_mma_swizzle_bench.rs`〉と同型の
//!   「同一プロセス内 2 インスタンス」設計、Metal 側
//!   `examples/gemm_bench.rs` occupancy 比較の「ラウンド交互・偶数ラウンド」
//!   先例を踏襲）。
//!
//! 判定統計（中央値ベース）・許容誤差・#540 既存の採否判定基準は本モジュールでは
//! 変更しない（`.claude/rules/security.md`: ガードレール閾値・テスト許容誤差の
//! 変更はユーザー承認必須）。`relative_spread`（`crate::stats`）による
//! ばらつきの定量報告を追加するのみで、判定閾値自体は呼び出し側 example
//! （`crates/backend-metal/examples/gemm_swizzle_ab_bench.rs`）が
//! `docs/perf/metal-bench-noise-protocol.md` の基準に沿って出力するに留める
//! （安定性ゲート不成立時は「判定不可」を明示し、判定へ進まない安全側設計）。

use crate::protocol::{self, MeasurementConfig};
use crate::stats::{self, BenchError};
use std::time::{Duration, Instant};

/// 安定性ゲート（[`StabilityResult::spread`] の許容上限）の単一真実源。
///
/// `docs/perf/metal-bench-noise-protocol.md`「5. 安定性ゲートと不成立時の中断規定」の
/// 「spread が概ね 5% を超えるサイズがある計測セッションは A/B 判定へ進まない」基準値。
/// 呼び出し側（`crates/backend-metal/examples/gemm_swizzle_ab_bench.rs`）が同じ値を
/// example 内へ直接定義すると、閾値変更時にコードと文書が独立に乖離しうる
/// （codex-review 指摘対応。イシュー #746 PR #763）。値を変更する場合は本定数と
/// 上記文書の記述を両方更新すること（ガードレール閾値相当のためユーザー承認必須。
/// `.claude/rules/security.md`）。
pub const STABILITY_SPREAD_GATE: f64 = 0.05;

/// [`run_ab`]／[`run_stability`] のラウンド構成。
///
/// `rounds` を偶数必須とするのは、ラウンド交互（A→B / B→A の順序反転）が
/// サーマルドリフトを相殺する前提が「A 先頭ラウンド数 == B 先頭ラウンド数」に
/// 依存するため（`crates/backend-metal/examples/gemm_bench.rs` の occupancy
/// 比較・`ROUNDS=6` 偶数固定と同じ order-bias 相殺根拠）。奇数を
/// `BenchError::ProtocolViolation` で fail-closed に拒否し、非対称な
/// interleave を黙って許容しない。
///
/// フィールドは非公開とし [`AbConfig::new`] の検証（偶数・2 以上）を経由した
/// 構築のみを許す（codex-review 指摘対応。イシュー #746 PR #763: 全フィールドが
/// `pub` だと `AbConfig { rounds: 3, .. }` のような直接構築で検証をバイパスでき、
/// `run_ab`／`run_stability` 側は入口で再検証しないため、奇数 rounds のまま
/// order-bias 相殺前提が崩れた計測が fail-closed 契約を経由せず実行されてしまう）。
/// 値の参照は [`AbConfig::rounds`]／[`AbConfig::cooldown`]／[`AbConfig::min_warmup`]
/// の getter を使う。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AbConfig {
    /// A/B（または単一系列）を計測するラウンド数。偶数・2 以上が必須。
    rounds: usize,
    /// ラウンド間（各 side の計測直後）に挟む待機時間。
    /// `protocol::run` の計測区間（`Instant` 計測ループ）の外側でのみ
    /// 発生させるため、計測サンプルへは混入しない（本モジュール実装の
    /// `std::thread::sleep` 呼び出し位置がその契約を担保する）。
    cooldown: Duration,
    /// 各ラウンドの計測直前に追加実行するウォームアップの最低経過時間。
    /// `MeasurementConfig::warmup`（回数下限）に加え、小サイズワークロードで
    /// GPU クロックが未昇圧のまま計測へ入るのを防ぐための時間ベース下限
    /// （`docs/perf/metal-bench-noise-protocol.md` 参照）。
    min_warmup: Duration,
}

impl AbConfig {
    /// `rounds` が偶数・2 以上であることを検証して構築する。
    ///
    /// # Errors
    ///
    /// `rounds` が奇数、または 0 の場合は `BenchError::ProtocolViolation`。
    pub fn new(
        rounds: usize,
        cooldown: Duration,
        min_warmup: Duration,
    ) -> Result<Self, BenchError> {
        if rounds == 0 || !rounds.is_multiple_of(2) {
            return Err(BenchError::ProtocolViolation(format!(
                "rounds は 2 以上の偶数が必須（順序反転による order-bias 相殺の前提）。指定値: {rounds}"
            )));
        }
        Ok(Self {
            rounds,
            cooldown,
            min_warmup,
        })
    }

    /// 検証済みのラウンド数（偶数・2 以上。[`AbConfig::new`] が保証）。
    pub fn rounds(&self) -> usize {
        self.rounds
    }

    /// ラウンド間の待機時間。
    pub fn cooldown(&self) -> Duration {
        self.cooldown
    }

    /// 時間ベースの追加ウォームアップ下限。
    pub fn min_warmup(&self) -> Duration {
        self.min_warmup
    }
}

/// `min_warmup` の経過時間、かつ `min_calls` 回以上のいずれも満たすまで
/// `workload` を呼び続ける（時間ベースの追加ウォームアップ）。
///
/// `protocol::run` 自身が行う回数ベースのウォームアップ（`MeasurementConfig::warmup`）
/// より前段で呼び、GPU クロック（DVFS）が定常状態に達するまでの時間を稼ぐ
/// （モジュール冒頭ドキュメント参照）。`std::hint::black_box` で `protocol::run`
/// と同じ最適化除去防止を行う（`crate::protocol::run` ドキュメンテーション
/// コメント「`black_box` による保護」節と同一の理由）。
fn extended_warmup<F: FnMut()>(min_warmup: Duration, min_calls: usize, mut workload: F) {
    let start = Instant::now();
    let mut calls = 0usize;
    #[allow(clippy::unit_arg)]
    while calls < min_calls || start.elapsed() < min_warmup {
        std::hint::black_box(workload());
        calls += 1;
    }
}

/// 単一ワークロードの安定性計測結果（[`run_stability`] の戻り値）。
#[derive(Debug, Clone)]
pub struct StabilityResult {
    /// 各ラウンドの中央値（秒）。`protocol::run` の `median_secs` をラウンド数分集めたもの。
    pub round_medians_secs: Vec<f64>,
    /// `round_medians_secs` の相対ばらつき（[`crate::relative_spread()`]）。
    pub spread: f64,
}

/// 同一ワークロードを `ab_config.rounds` ラウンド計測し、ラウンド間の
/// ばらつき（[`StabilityResult::spread`]）を返す。
///
/// ノイズ対策プロトコルの「対照カーネルのばらつきが安定しているか」
/// （安定性セルフチェック。`docs/perf/metal-bench-noise-protocol.md`）を
/// 判定するための材料を提供する。閾値判定自体は行わない（モジュール冒頭
/// ドキュメント参照）。
///
/// # Errors
///
/// - `measurement_config` が warmup・iters とも 20 回未満の場合、
///   `protocol::run` が返す `BenchError::ProtocolViolation` をそのまま伝播する
/// - ラウンド中央値の集計（[`crate::relative_spread()`]）が失敗した場合
///   （NaN 混入等）はそのエラーを伝播する
pub fn run_stability<F: FnMut()>(
    ab_config: &AbConfig,
    measurement_config: &MeasurementConfig,
    mut workload: F,
) -> Result<StabilityResult, BenchError> {
    let mut round_medians_secs = Vec::with_capacity(ab_config.rounds);

    for round in 0..ab_config.rounds {
        extended_warmup(
            ab_config.min_warmup,
            measurement_config.warmup,
            &mut workload,
        );
        let measurement = protocol::run(measurement_config, &mut workload)?;
        round_medians_secs.push(measurement.median_secs);

        let is_last_round = round + 1 == ab_config.rounds;
        if !is_last_round {
            std::thread::sleep(ab_config.cooldown);
        }
    }

    let spread = stats::relative_spread(&round_medians_secs)?;
    Ok(StabilityResult {
        round_medians_secs,
        spread,
    })
}

/// [`run_ab`] の戻り値。`a` を base（例: swizzle off）、`b` を head
/// （例: swizzle on）として扱う想定（呼び出し側の意味付けは自由）。
#[derive(Debug, Clone)]
pub struct AbResult {
    /// side A の各ラウンド中央値（秒）。
    pub a_round_medians_secs: Vec<f64>,
    /// side B の各ラウンド中央値（秒）。
    pub b_round_medians_secs: Vec<f64>,
    /// side A の全ラウンド中央値（[`crate::median_q1_q3()`] の median）。
    pub median_a_secs: f64,
    /// side B の全ラウンド中央値。
    pub median_b_secs: f64,
    /// `median_b_secs / median_a_secs`。**実行時間（レイテンシ）の比**であり、
    /// 1.0 未満なら B が A より速い。TFLOPS 等スループット指標の head/base 比は
    /// 時間の逆数のためこの値の**逆数**（`median_a_secs / median_b_secs`）になる。
    /// #540 の採否判定基準（`head_over_base` > 1.0 で採用）はスループット比を
    /// 前提とするため、呼び出し側でそのまま `head_over_base` として使わない
    /// こと（codex-review・Cursor Bugbot 指摘対応。イシュー #746 PR #763:
    /// `crates/backend-metal/examples/gemm_swizzle_ab_bench.rs` がこの値を
    /// そのまま `head_over_base` として出力し判定を逆転させていた）。
    pub b_over_a_ratio: f64,
    /// side A のラウンド間ばらつき（[`crate::relative_spread()`]）。
    pub spread_a: f64,
    /// side B のラウンド間ばらつき。
    pub spread_b: f64,
}

/// [`run_ab`] が `b_over_a_ratio`（`median_b_secs / median_a_secs`）を計算する前に
/// 両中央値の符号・有限性を検証する（`crate::threshold::judge` が own.median_secs に
/// 課す検証と同じ理由・同じ fail-closed 方針）。
///
/// `stats::relative_spread` は全サンプルが 0 のときのみ median=0.0 を許容して
/// `Ok(0.0)` を返す設計のため、`protocol::run` の計測精度がタイマー分解能を下回る
/// 極小ワークロードでは `median_a_secs`／`median_b_secs` が 0.0 になりうる。ここで
/// 検証せずに比を計算すると NaN（両方 0）や無限大（片方のみ 0）を含む `Ok(AbResult)`
/// を返してしまい、呼び出し側（`gemm_swizzle_ab_bench.rs`）の `head_over_base`
/// 計算へ非有限値が silent に伝播する（codex-review 指摘対応。イシュー #746 PR #763）。
///
/// `median_b_secs` も 0.0 未満に加え **0.0 そのものを拒否**する: 呼び出し側は
/// `median_b_secs` を `b_over_a_ratio` の分子だけでなく `tflops(size, median_b_secs)`
/// （時間の逆数）の分母としても使うため、0.0 を非負として素通しすると
/// `head_median_tflops` が無限大になり `head_over_base` へ非有限値が伝播する。
/// 「非負なら許容」では本関数のドキュメント上の目的（非有限値の伝播防止）を
/// 達成できないため、`median_a_secs` と同じ「正の有限値」を要求する
/// （codex-review 指摘対応。イシュー #746 PR #763 再指摘）。
///
/// 単体テスト（`tests::validate_ab_medians_*`）から直接呼べるよう、[`run_ab`] 本体
/// （実測タイマー依存で 0 秒を決定的に再現できない）とは切り離した関数にする。
///
/// # Errors
///
/// `median_a_secs`・`median_b_secs` のいずれかが正の有限値でない場合、
/// `BenchError::ProtocolViolation`。
fn validate_ab_medians(median_a_secs: f64, median_b_secs: f64) -> Result<(), BenchError> {
    if !(median_a_secs.is_finite() && median_a_secs > 0.0) {
        return Err(BenchError::ProtocolViolation(format!(
            "median_a_secs は正の有限値が必須（b_over_a_ratio の分母）。実際: {median_a_secs}"
        )));
    }
    if !(median_b_secs.is_finite() && median_b_secs > 0.0) {
        return Err(BenchError::ProtocolViolation(format!(
            "median_b_secs は正の有限値が必須（呼び出し側の tflops 計算で分母にも \
             使われるため 0.0 も拒否する）。実際: {median_b_secs}"
        )));
    }
    Ok(())
}

/// `workload_a`（例: base = swizzle off）・`workload_b`（例: head = swizzle on）を
/// `ab_config.rounds` ラウンド interleaved に計測する。
///
/// ラウンドごとに A→B / B→A の順序を反転させる（偶数ラウンドは A 先頭、
/// 奇数ラウンドは B 先頭）ことで、時間経過に伴うサーマルドリフト・GPU
/// クロック変動が A・B いずれか一方だけに系統的に乗るのを防ぐ（モジュール
/// 冒頭ドキュメント参照）。`ab_config.rounds` が偶数であることは
/// [`AbConfig::new`] が保証するため、A 先頭ラウンド数と B 先頭ラウンド数は
/// 必ず等しくなる。
///
/// 各ラウンドの各 side 計測の直前に `extended_warmup` を挟み、直後に
/// `ab_config.cooldown` を待機する（最終ラウンドの最終 side の後は待機しない。
/// 呼び出し元がこの後すぐ関数を抜けるため）。
///
/// # Errors
///
/// [`run_stability`] と同じ（`protocol::run`・`stats::relative_spread` の
/// エラーをそのまま伝播する）。
pub fn run_ab<FA: FnMut(), FB: FnMut()>(
    ab_config: &AbConfig,
    measurement_config: &MeasurementConfig,
    mut workload_a: FA,
    mut workload_b: FB,
) -> Result<AbResult, BenchError> {
    let mut a_round_medians_secs = Vec::with_capacity(ab_config.rounds);
    let mut b_round_medians_secs = Vec::with_capacity(ab_config.rounds);

    for round in 0..ab_config.rounds {
        let a_first = round % 2 == 0;
        let is_last_round = round + 1 == ab_config.rounds;

        let mut measure_a = |is_last_side: bool| -> Result<(), BenchError> {
            extended_warmup(
                ab_config.min_warmup,
                measurement_config.warmup,
                &mut workload_a,
            );
            let measurement = protocol::run(measurement_config, &mut workload_a)?;
            a_round_medians_secs.push(measurement.median_secs);
            if !(is_last_round && is_last_side) {
                std::thread::sleep(ab_config.cooldown);
            }
            Ok(())
        };

        // クロージャの可変借用が同時に 2 つ生存しないよう、A/B 計測を
        // ブロックで分けて呼ぶ（`measure_a` は `workload_a`／
        // `a_round_medians_secs` を可変借用するため、`measure_b` 定義と
        // スコープが重ならないようにする）。
        if a_first {
            measure_a(false)?;
        }

        {
            let is_last_side_b = a_first;
            extended_warmup(
                ab_config.min_warmup,
                measurement_config.warmup,
                &mut workload_b,
            );
            let measurement = protocol::run(measurement_config, &mut workload_b)?;
            b_round_medians_secs.push(measurement.median_secs);
            if !(is_last_round && is_last_side_b) {
                std::thread::sleep(ab_config.cooldown);
            }
        }

        if !a_first {
            measure_a(true)?;
        }
    }

    let spread_a = stats::relative_spread(&a_round_medians_secs)?;
    let spread_b = stats::relative_spread(&b_round_medians_secs)?;
    let median_a_secs = stats::median_q1_q3(&a_round_medians_secs)?.median;
    let median_b_secs = stats::median_q1_q3(&b_round_medians_secs)?.median;

    validate_ab_medians(median_a_secs, median_b_secs)?;

    Ok(AbResult {
        a_round_medians_secs,
        b_round_medians_secs,
        median_a_secs,
        median_b_secs,
        b_over_a_ratio: median_b_secs / median_a_secs,
        spread_a,
        spread_b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// タイマー分解能未満の 0 秒計測を回避するための最小限の実作業ワークロード。
    ///
    /// 無 op クロージャ（`|| {}`）は `protocol::run` の計測ループで測ると、
    /// macOS ローカル環境の並列テスト実行下（他テストとの CPU 競合）で
    /// 中央値がタイマー分解能（数十 ns 〜 1µs 程度）以下の 0 になることがあり、
    /// `run_ab`／`run_stability` の 0 秒拒否（[`validate_ab_medians`]）が
    /// `ProtocolViolation` を返して間欠的にテストが失敗していた（イシュー #904）。
    /// `std::hint::black_box` で最適化除去を防ぎつつ固定回数の演算を行うことで、
    /// 負荷下でも計測時間が確実にタイマー分解能を超えるようにする（`run_ab`／
    /// `run_stability` の 0 秒拒否契約自体は変更しない）。
    fn spin_workload() {
        let mut acc: u64 = 0;
        for i in 0..2_000u64 {
            acc = std::hint::black_box(acc.wrapping_add(std::hint::black_box(i)));
        }
        std::hint::black_box(acc);
    }

    #[test]
    fn new_rejects_odd_rounds() {
        let err = AbConfig::new(3, Duration::ZERO, Duration::ZERO)
            .expect_err("奇数 rounds は拒否されるはず");
        assert!(matches!(err, BenchError::ProtocolViolation(_)));
    }

    #[test]
    fn new_rejects_zero_rounds() {
        let err = AbConfig::new(0, Duration::ZERO, Duration::ZERO)
            .expect_err("0 rounds は拒否されるはず");
        assert!(matches!(err, BenchError::ProtocolViolation(_)));
    }

    #[test]
    fn new_accepts_even_rounds() {
        let config =
            AbConfig::new(6, Duration::ZERO, Duration::ZERO).expect("偶数 rounds は受理されるはず");
        assert_eq!(config.rounds, 6);
    }

    #[test]
    fn run_stability_collects_one_median_per_round() {
        let ab_config = AbConfig::new(4, Duration::ZERO, Duration::ZERO).unwrap();
        let measurement_config = MeasurementConfig::new(20, 20).unwrap();
        let result = run_stability(&ab_config, &measurement_config, spin_workload).unwrap();
        assert_eq!(result.round_medians_secs.len(), 4);
        // 軽量な固定回数ワークロードは全ラウンドほぼ同じ短時間のため spread は 0 近傍。
        assert!(result.spread >= 0.0);
    }

    #[test]
    fn run_ab_a_first_and_b_first_counts_are_equal() {
        // A 先頭ラウンド数と B 先頭ラウンド数が一致することを、全呼び出し
        // シーケンスを記録したうえで round・side 単位のバッチへ分解して検証する。
        //
        // `extended_warmup`（本テストの `min_warmup=ZERO` では `min_calls` 回
        // ちょうど）+ `protocol::run`（`warmup`+`iters`）で 1 round の 1 side
        // あたりの呼び出し回数は 20+40=60 回に決定的に固定される（下記
        // `CALLS_PER_BATCH` 参照）。ラウンド境界をまたぐと同じ side が
        // 連続することがある（例: round0 が `A,B`・round1 が `B,A` の順序なら
        // 呼び出しシーケンス全体は `A,B,B,A,...` になり `B,B` が連続する）ため、
        // 単純な「直前と side が変わった箇所」の検出ではラウンド境界を
        // 見誤る。呼び出し回数が固定であることを使い、`CALLS_PER_BATCH` 個
        // ずつのバッチへ機械的に分割することで正しくラウンド境界を復元する。
        let call_log: RefCell<Vec<char>> = RefCell::new(Vec::new());
        let ab_config = AbConfig::new(6, Duration::ZERO, Duration::ZERO).unwrap();
        let measurement_config = MeasurementConfig::new(20, 20).unwrap();

        run_ab(
            &ab_config,
            &measurement_config,
            || {
                spin_workload();
                call_log.borrow_mut().push('a');
            },
            || {
                spin_workload();
                call_log.borrow_mut().push('b');
            },
        )
        .unwrap();

        const CALLS_PER_BATCH: usize = 60; // extended_warmup 20 + protocol::run (warmup 20 + iters 20)
        let log = call_log.borrow();
        assert_eq!(log.len(), CALLS_PER_BATCH * ab_config.rounds * 2);
        assert!(
            log.chunks(CALLS_PER_BATCH)
                .all(|batch| batch.iter().all(|&c| c == batch[0])),
            "各バッチ内は単一 side の呼び出しのみのはず: {log:?}"
        );

        let batch_first_sides: Vec<char> = log.chunks(CALLS_PER_BATCH).map(|b| b[0]).collect();
        assert_eq!(batch_first_sides.len(), ab_config.rounds * 2);

        let mut a_first_count = 0;
        let mut b_first_count = 0;
        for round_batches in batch_first_sides.chunks(2) {
            assert_ne!(
                round_batches[0], round_batches[1],
                "1 round 内の 2 バッチは異なる side のはず: {batch_first_sides:?}"
            );
            match round_batches[0] {
                'a' => a_first_count += 1,
                'b' => b_first_count += 1,
                other => unreachable!("side は 'a'/'b' のみのはず: {other:?}"),
            }
        }
        assert_eq!(
            a_first_count, b_first_count,
            "偶数ラウンドなら A 先頭・B 先頭は同数のはず: {batch_first_sides:?}"
        );
        assert_eq!(a_first_count, ab_config.rounds / 2);
    }

    #[test]
    fn run_ab_cooldown_is_outside_measured_region() {
        // cooldown を大きめに設定しても、計測サンプル（`protocol::run` の
        // `median_secs`）には cooldown の待機時間が混入しないことを検証する
        // （計測ループは `Instant` を workload 呼び出しの前後でしか取らない
        // `crate::protocol::run` の契約。cooldown の sleep はその外側で呼ぶ設計）。
        let ab_config = AbConfig::new(2, Duration::from_millis(5), Duration::ZERO).unwrap();
        let measurement_config = MeasurementConfig::new(20, 20).unwrap();
        let result = run_ab(
            &ab_config,
            &measurement_config,
            spin_workload,
            spin_workload,
        )
        .unwrap();
        // 軽量な固定回数ワークロードの中央値は cooldown（5ms）よりはるかに小さいはず。
        assert!(result.median_a_secs < 0.001);
        assert!(result.median_b_secs < 0.001);
    }

    #[test]
    fn b_over_a_ratio_is_a_latency_ratio_not_a_throughput_ratio() {
        // `AbResult::b_over_a_ratio` が「実行時間（レイテンシ）の比」であり、
        // スループット（TFLOPS 等）比の head/base とは逆数の関係になることを
        // 固定する回帰テスト（codex-review・Cursor Bugbot 指摘対応。イシュー
        // #746 PR #763: `gemm_swizzle_ab_bench.rs` がこの値をそのまま
        // `head_over_base`〈スループット比〉として出力し #540 の採否判定
        // 〈> 1.0 で採用〉を逆転させていた）。B（head 相当）を A（base 相当）
        // より明確に遅くし、`b_over_a_ratio` が 1.0 を大きく超える
        // （= B が遅い）ことを検証する。これはスループット比では
        // `head_over_base < 1.0`（悪化）に対応するはずの状況であり、
        // 両者が同じ向きの値だと誤って扱わないことをロックする。
        let ab_config = AbConfig::new(2, Duration::ZERO, Duration::ZERO).unwrap();
        let measurement_config = MeasurementConfig::new(20, 20).unwrap();
        let result = run_ab(&ab_config, &measurement_config, spin_workload, || {
            std::thread::sleep(Duration::from_micros(200))
        })
        .unwrap();
        assert!(
            result.b_over_a_ratio > 1.0,
            "B（head 相当）を遅くしたので b_over_a_ratio（レイテンシ比）は 1.0 を超えるはず: {}",
            result.b_over_a_ratio
        );
        // スループット比（head_over_base 相当）はレイテンシ比の逆数であるべき。
        let throughput_ratio = result.median_a_secs / result.median_b_secs;
        assert!(
            (throughput_ratio - result.b_over_a_ratio.recip()).abs() < 1e-9,
            "スループット比は b_over_a_ratio の逆数であるべき: throughput_ratio={throughput_ratio} \
             b_over_a_ratio={}",
            result.b_over_a_ratio
        );
        assert!(
            throughput_ratio < 1.0,
            "B が遅い（head 悪化）状況ではスループット比は 1.0 未満（不採用方向）のはず: {throughput_ratio}"
        );
    }

    #[test]
    fn extended_warmup_extends_call_count_until_min_warmup_elapsed() {
        // min_warmup が満たされるまで warmup が延長されることを、呼び出し回数
        // カウントで検証する（インライン `extended_warmup` の単体テスト）。
        let call_count = AtomicUsize::new(0);
        extended_warmup(Duration::from_millis(5), 20, || {
            call_count.fetch_add(1, Ordering::SeqCst);
        });
        // min_calls=20 のみなら 20 回で終わるはずだが、5ms 分の追加ループが
        // 生じるため、軽量クロージャでは 20 回を大幅に超えるはず。
        assert!(
            call_count.load(Ordering::SeqCst) > 20,
            "min_warmup による時間ベース延長が機能していれば 20 回を超えるはず: {}",
            call_count.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn validate_ab_medians_rejects_zero_median_a_secs() {
        // median_a_secs=0（タイマー分解能未満の極小ワークロード等で発生しうる）は
        // b_over_a_ratio の分母がゼロになり NaN/無限大混入の AbResult を Ok で
        // 返してしまうため、fail-closed に拒否されることを確認する
        // （codex-review 指摘対応。イシュー #746 PR #763）。
        let err = validate_ab_medians(0.0, 1.0)
            .expect_err("median_a_secs=0 は拒否されるはず（ゼロ除算防止）");
        assert!(matches!(err, BenchError::ProtocolViolation(_)));
    }

    #[test]
    fn validate_ab_medians_rejects_both_zero() {
        // 両中央値が 0 だと比率計算は 0.0/0.0 = NaN になるため拒否されるはず。
        let err = validate_ab_medians(0.0, 0.0)
            .expect_err("median_a_secs=median_b_secs=0 は拒否されるはず");
        assert!(matches!(err, BenchError::ProtocolViolation(_)));
    }

    #[test]
    fn validate_ab_medians_rejects_zero_median_b_secs() {
        // median_b_secs=0.0 は b_over_a_ratio 上は非負として許容できて見えるが、
        // 呼び出し側の tflops(size, median_b_secs) の分母にもなるため無限大が
        // 伝播しうる。0.0 も拒否されることを固定する（codex-review 再指摘対応。
        // イシュー #746 PR #763）。
        let err = validate_ab_medians(1.0, 0.0)
            .expect_err("median_b_secs=0 は拒否されるはず（tflops 計算の分母保護）");
        assert!(matches!(err, BenchError::ProtocolViolation(_)));
    }

    #[test]
    fn validate_ab_medians_rejects_non_finite_values() {
        assert!(validate_ab_medians(f64::NAN, 1.0).is_err());
        assert!(validate_ab_medians(1.0, f64::INFINITY).is_err());
        assert!(validate_ab_medians(-1.0, 1.0).is_err());
        assert!(validate_ab_medians(1.0, -1.0).is_err());
    }

    #[test]
    fn validate_ab_medians_accepts_positive_finite_values() {
        assert!(validate_ab_medians(1.0, 2.0).is_ok());
    }

    #[test]
    fn extended_warmup_respects_min_calls_when_min_warmup_is_zero() {
        let call_count = AtomicUsize::new(0);
        extended_warmup(Duration::ZERO, 20, || {
            call_count.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(call_count.load(Ordering::SeqCst), 20);
    }
}
