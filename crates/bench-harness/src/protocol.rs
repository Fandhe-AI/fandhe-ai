//! バックエンド非依存の計測プロトコル（TASK-8.1a）。
//!
//! `backend-cpu` / `backend-cuda` / `backend-metal` いずれのワークロードも
//! クロージャとして受け取ることで、バックエンド抽象層（TASK-1.9・未完了）の完成を
//! 待たずに計測コアを実装する（イシュー #27 実装計画の安全側判断）。
//! warmup 20 回以上・計測 20 回以上・中央値採用・Q1/Q3 記録という
//! `docs/spec/05-tasks.md` TASK-8.1 の受け入れ条件を満たす `run` を提供する。
//!
//! ## スコープ境界
//!
//! - バックエンド別の同期統一（CUDA `stream.synchronize()` / Metal コマンドバッファ完了待ち）や
//!   決定的シード（xorshift64*）は TASK-8.1b（イシュー #28）のスコープ。本モジュールは
//!   「`workload` クロージャの呼び出しが返った時点で計測対象処理が完了している」ことを
//!   呼び出し側の責務として前提とするのみで、同期処理そのものは持たない
//!   （呼び出し側が同期フックとして `workload` の中に同期呼び出しを含める設計）。
//! - 構造化出力（JSON 等）・プロトコル遵守回帰テストは TASK-8.1c（イシュー #29）のスコープ。
//!   `Measurement` はプレーンな `f64`/`usize` フィールドに留め、`serde` derive は付与しない
//!   （`.claude/rules/deps-policy.md`: 依存追加はユーザー承認必須のため、自動運転下では追加しない）。

use crate::stats::{self, BenchError, Quartiles};
use std::hint::black_box;
use std::time::Instant;

/// TASK-8.1 が定める計測プロトコルの下限（warmup・計測回数とも 20 回以上）。
const MIN_ITERATIONS: usize = 20;

/// 計測設定（warmup 回数・計測回数）。
///
/// `Default` は spec 最低値（20/20）を採用する。`run` はこの下限を下回る設定を
/// `BenchError::ProtocolViolation` で拒否し、下限を回避する API を設けない
/// （ガードレール・許容値の単独緩和禁止の趣旨に整合。`.claude/rules/security.md`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasurementConfig {
    /// 計測対象外で先行実行する回数（キャッシュ・JIT 等のウォームアップ）。
    pub warmup: usize,
    /// 計測対象として所要時間を記録する回数。
    pub iters: usize,
}

impl Default for MeasurementConfig {
    fn default() -> Self {
        Self {
            warmup: MIN_ITERATIONS,
            iters: MIN_ITERATIONS,
        }
    }
}

impl MeasurementConfig {
    /// `warmup`・`iters` を指定して設定を構築する。
    ///
    /// spec 下限（20 回以上）を満たさない場合は `BenchError::ProtocolViolation` を返す。
    pub fn new(warmup: usize, iters: usize) -> Result<Self, BenchError> {
        if warmup < MIN_ITERATIONS {
            return Err(BenchError::ProtocolViolation(format!(
                "warmup は {MIN_ITERATIONS} 回以上が必須（TASK-8.1）。指定値: {warmup}"
            )));
        }
        if iters < MIN_ITERATIONS {
            return Err(BenchError::ProtocolViolation(format!(
                "計測回数（iters）は {MIN_ITERATIONS} 回以上が必須（TASK-8.1）。指定値: {iters}"
            )));
        }
        Ok(Self { warmup, iters })
    }
}

/// 1 回の計測プロトコル実行結果。
///
/// 受け入れ条件（「計測結果に中央値・Q1/Q3 が記録される」）を満たすため、
/// `median_secs`・`q1_secs`・`q3_secs` を必ず保持する。TFLOPS 等ドメイン固有の換算は
/// 本クレートの関心事ではなく呼び出し側（TASK-8.2 以降）の責務とし、秒単位に留める。
#[derive(Debug, Clone)]
pub struct Measurement {
    pub median_secs: f64,
    pub q1_secs: f64,
    pub q3_secs: f64,
    /// 集計前の全計測サンプル（秒）。TASK-8.1c（イシュー #29）の構造化出力等、
    /// 中央値・Q1/Q3 以外の再集計が必要になった場合に備えて保持する。
    pub samples_secs: Vec<f64>,
    pub warmup: usize,
    pub iters: usize,
}

/// `config` に従って `workload` を warmup 実行 → 計測実行し、`Measurement` を返す。
///
/// `workload` は 1 回の計測対象処理を表すクロージャで、呼び出しが返った時点で
/// 処理が完了している（同期済みである）ことを前提とする。GPU バックエンドで
/// 非同期実行を使う場合は、呼び出し側が `workload` 内で同期呼び出しを行う必要がある
/// （バックエンド別同期の統一自体は TASK-8.1b・イシュー #28 のスコープ）。
///
/// `workload` はジェネリック（`F: FnMut()`）であるため単相化・インライン化の対象になり、
/// 副作用を伴わない呼び出しはコンパイラに最適化除去されうる（`criterion` が計測対象の
/// 呼び出しに `black_box` を必須とするのと同じ理由）。`run` は `workload()` の呼び出しを
/// `std::hint::black_box` で包み、呼び出し自体が最適化で消えないようにする。ただし
/// `black_box` はクロージャ内部の計算過程までは保護しないため、`workload` が引数を
/// 使わずに定数を返す・メモ化結果を返す等の実装であれば内部計算自体は依然として
/// 最適化されうる。計測対象の入出力は呼び出し側が `workload` 内で `black_box` する
/// （あるいは計測対象データへの副作用を伴う）ことが前提となる。
///
/// # Errors
///
/// - `config` が下限（20/20）未満の場合は `BenchError::ProtocolViolation`
///   （`MeasurementConfig::new` を経由していれば発生しないが、`Default` 以外の
///   構築経路を将来追加した場合に備えて `run` 側でも防御的に検証する）
/// - 計測サンプルの集計に失敗した場合は `stats::median_q1_q3` のエラーをそのまま返す
pub fn run<F: FnMut()>(
    config: &MeasurementConfig,
    mut workload: F,
) -> Result<Measurement, BenchError> {
    if config.warmup < MIN_ITERATIONS || config.iters < MIN_ITERATIONS {
        return Err(BenchError::ProtocolViolation(format!(
            "warmup・iters とも {MIN_ITERATIONS} 回以上が必須（TASK-8.1）。指定値: warmup={}, iters={}",
            config.warmup, config.iters
        )));
    }

    // `workload()` の戻り値（`()`）を `black_box` に渡すのは意図的な設計であり、
    // `unit_arg` lint の対象になるが安易な握り潰しではない。`black_box` の役割は
    // 「呼び出しの結果を最適化に対して不透明にする」ことにあり、戻り値が unit でも
    // 呼び出しそのものを最適化除去から保護する効果は保持される（criterion が計測対象の
    // 呼び出しを `black_box` で包むのと同じ理由。上記 `run` のドキュメンテーションコメント参照）。
    #[allow(clippy::unit_arg)]
    {
        for _ in 0..config.warmup {
            black_box(workload());
        }
    }

    let mut samples_secs = Vec::with_capacity(config.iters);
    #[allow(clippy::unit_arg)]
    for _ in 0..config.iters {
        let start = Instant::now();
        black_box(workload());
        samples_secs.push(start.elapsed().as_secs_f64());
    }

    let Quartiles { median, q1, q3 } = stats::median_q1_q3(&samples_secs)?;

    Ok(Measurement {
        median_secs: median,
        q1_secs: q1,
        q3_secs: q3,
        samples_secs,
        warmup: config.warmup,
        iters: config.iters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn default_config_is_20_20() {
        let config = MeasurementConfig::default();
        assert_eq!(config.warmup, 20);
        assert_eq!(config.iters, 20);
    }

    #[test]
    fn new_rejects_warmup_below_minimum() {
        let err =
            MeasurementConfig::new(19, 20).expect_err("warmup 19 は下限未満のため拒否されるはず");
        assert!(matches!(err, BenchError::ProtocolViolation(_)));
    }

    #[test]
    fn new_rejects_iters_below_minimum() {
        let err =
            MeasurementConfig::new(20, 19).expect_err("iters 19 は下限未満のため拒否されるはず");
        assert!(matches!(err, BenchError::ProtocolViolation(_)));
    }

    #[test]
    fn new_accepts_minimum_20_20() {
        let config =
            MeasurementConfig::new(20, 20).expect("20/20 は下限ちょうどのため成功するはず");
        assert_eq!(config.warmup, 20);
        assert_eq!(config.iters, 20);
    }

    #[test]
    fn run_rejects_config_below_minimum_defensively() {
        // MeasurementConfig::new を経由しない構築（フィールドが pub のため直接構築可能）でも
        // run 側の防御的検証で下限未満は拒否される。
        let config = MeasurementConfig {
            warmup: 1,
            iters: 1,
        };
        let err = run(&config, || {}).expect_err("下限未満は run 側でも拒否されるはず");
        assert!(matches!(err, BenchError::ProtocolViolation(_)));
    }

    #[test]
    fn run_calls_workload_warmup_plus_iters_times() {
        let call_count = AtomicUsize::new(0);
        let config = MeasurementConfig::new(20, 20).unwrap();
        let measurement = run(&config, || {
            call_count.fetch_add(1, Ordering::SeqCst);
        })
        .expect("軽量ダミーワークロードは成功するはず");

        assert_eq!(call_count.load(Ordering::SeqCst), 40); // warmup 20 + iters 20
        assert_eq!(measurement.samples_secs.len(), 20);
        assert_eq!(measurement.warmup, 20);
        assert_eq!(measurement.iters, 20);
    }

    #[test]
    fn run_records_median_q1_q3() {
        // 受け入れ条件「計測結果に中央値・Q1/Q3 が記録される」の直接検証。
        let config = MeasurementConfig::new(20, 20).unwrap();
        let measurement = run(&config, || {
            // カウンタ加算のみの軽量ダミーワークロード。所要時間は 0 以上であればよい。
        })
        .expect("軽量ダミーワークロードは成功するはず");

        assert!(measurement.median_secs >= 0.0);
        assert!(measurement.q1_secs >= 0.0);
        assert!(measurement.q3_secs >= 0.0);
        // median-of-halves 定義（stats::median_q1_q3）により q1 <= median <= q3 が成立する。
        assert!(measurement.q1_secs <= measurement.median_secs);
        assert!(measurement.median_secs <= measurement.q3_secs);
    }

    #[test]
    fn run_does_not_optimize_workload_away() {
        // `workload` はジェネリックのため単相化・インライン化されうる。副作用を伴わない
        // クロージャの呼び出しがコンパイラに丸ごと除去されると計測値がゼロ近傍に潰れ、
        // TASK-8.2・TASK-3.2 の合否判定を誤らせる（Review 指摘）。`black_box` による保護が
        // 機能していれば、ある程度の計算量を持つワークロードで所要時間が確実に正の値になる。
        let config = MeasurementConfig::new(20, 20).unwrap();
        let measurement = run(&config, || {
            let mut acc: u64 = 0;
            for i in 0..10_000u64 {
                acc = black_box(acc.wrapping_add(black_box(i)));
            }
            black_box(acc);
        })
        .expect("計算量のあるワークロードは成功するはず");

        // 最適化除去されていれば samples はほぼ 0 のままになりうるため、
        // 全サンプルが厳密に正であることまで確認する（受け入れ条件の直接検証）。
        assert!(
            measurement.samples_secs.iter().all(|&s| s > 0.0),
            "black_box 保護が機能していれば全サンプルは正の所要時間を記録するはず: {:?}",
            measurement.samples_secs
        );
        assert!(measurement.median_secs > 0.0);
    }
}
