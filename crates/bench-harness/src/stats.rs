//! 計測サンプル列から中央値・Q1/Q3 を求める純粋関数群。
//!
//! `protocol::run`（同クレート `protocol` モジュール）から呼ばれ、
//! ウォームアップ後の計測サンプル（秒単位の所要時間）を集計する責務を持つ。
//! 分位点の定義は PoC-v2-1 参照実装
//! （`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/rust/src/bin/gemm_bench.rs:17-25`）の
//! median-of-halves 方式をそのまま踏襲する（TASK-8.1a・REQ-8）。
//! 値の意味（TFLOPS 換算等）はこのモジュールの関心事ではなく、呼び出し側（TASK-8.2 以降）に委ねる。

use std::fmt;

/// 統計計算の失敗を表す型付きエラー。
///
/// 本番経路で `unwrap()` / `expect()` を使わない方針（`.claude/rules/coding-rust.md`）に基づき、
/// 空スライス・NaN 混入といった不正入力は fail-closed にこのエラーで弾く。
#[derive(Debug, Clone, PartialEq)]
pub enum BenchError {
    /// サンプル列が空で分位点を計算できない。
    EmptySamples,
    /// サンプル列に NaN が混入しており、決定的な順序付けができない。
    ///
    /// 計測結果は将来 guardrail／self-repair の合否判定（TASK-8.2・TASK-3.2）の入力になるため、
    /// NaN を黙って無視・0 扱いせず拒否する（REQ-8 の意図を汲んだ安全側判断）。
    NanSample,
    /// `MeasurementConfig` が spec 下限（warmup 20 回以上・計測 20 回以上）を満たさない。
    ProtocolViolation(String),
    /// 環境ガード（[`crate::env_guard`]）のバックオフ再試行が上限回数に達しても
    /// `overall == GuardVerdict::Fail`（[`crate::env_guard::EnvGuardReport::is_blocking`]）
    /// のまま解消しなかった（イシュー #1265）。`attempts` は実行した試行回数
    /// （`max_attempts` と一致）、`detail` は最終試行の実測値・上限・flagged
    /// プロセス名等の人が読める要約（ホスト名・ユーザー名・機体識別子は含まない。
    /// `docs/real-hardware-verification-env.md` 方針）。呼び出し側
    /// （`examples/gemm_transpose_route_ab_bench.rs`）はこのエラーで計測を
    /// 中断し `verdict=undetermined` として記録する。
    EnvGuardExhausted { attempts: usize, detail: String },
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchError::EmptySamples => write!(f, "サンプル列が空のため分位点を計算できない"),
            BenchError::NanSample => write!(f, "サンプル列に NaN が混入している"),
            BenchError::ProtocolViolation(msg) => write!(f, "計測プロトコル違反: {msg}"),
            BenchError::EnvGuardExhausted { attempts, detail } => write!(
                f,
                "環境ガードが {attempts} 回の再試行上限に達した: {detail}"
            ),
        }
    }
}

impl std::error::Error for BenchError {}

/// 中央値・Q1・Q3 の組。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quartiles {
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
}

/// サンプル列（秒単位の所要時間等）から中央値・Q1・Q3 を求める。
///
/// 分位点は「ソート後、`idx = round(p * (n-1))` 番目の要素を採用する」
/// median-of-halves 方式（PoC-v2-1 実測踏襲。`p=0.5/0.25/0.75`）で定義する。
/// 線形補間方式とは値が異なりうるため、この定義自体をテスト期待値に固定する
/// （`.claude/rules/coding-rust.md`: バックエンド間許容誤差と同様、定義の単独変更を避ける）。
///
/// # Errors
///
/// - `samples` が空の場合は `BenchError::EmptySamples`
/// - NaN が混入している場合は `BenchError::NanSample`
pub fn median_q1_q3(samples: &[f64]) -> Result<Quartiles, BenchError> {
    if samples.is_empty() {
        return Err(BenchError::EmptySamples);
    }
    if samples.iter().any(|x| x.is_nan()) {
        return Err(BenchError::NanSample);
    }

    // `partial_cmp().unwrap()`（PoC-v2-1 参照実装）は NaN 混入時に panic しうるため、
    // 本番経路 unwrap 禁止方針（coding-rust.md）に従い `f64::total_cmp` を用いる。
    // NaN は上で既に弾いているため、total_cmp と partial_cmp は本関数内で同じ順序を返す。
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(f64::total_cmp);

    let n = sorted.len();
    let pick = |p: f64| -> f64 {
        let idx = (p * (n as f64 - 1.0)).round() as usize;
        sorted[idx.min(n - 1)]
    };

    Ok(Quartiles {
        median: pick(0.5),
        q1: pick(0.25),
        q3: pick(0.75),
    })
}

/// サンプル列の相対ばらつき `(max − min) / median` を求める。
///
/// `ab::run_stability`（同クレート `ab` モジュール。イシュー #746）が
/// 「対照カーネルの複数ラウンド計測がどの程度ばらついたか」を定量化する
/// ために呼ぶ。ノイズ対策プロトコル（`docs/perf/metal-bench-noise-protocol.md`）
/// の安定性ゲート（spread ≤5% 程度）の判定材料であり、本関数自体は
/// 閾値判定を行わない（判定は呼び出し側 example の責務。ガードレール
/// 閾値・許容誤差の単独緩和はユーザー承認必須という方針
/// `.claude/rules/security.md` に触れないよう、閾値をこのクレートに
/// 埋め込まない設計）。
///
/// `median` の定義は [`median_q1_q3`] と同一（median-of-halves 方式）。
///
/// # Errors
///
/// - `samples` が空の場合は `BenchError::EmptySamples`
/// - NaN が混入している場合は `BenchError::NanSample`
///
/// `median` が 0.0 の場合（全サンプルが 0 秒。理論上は起こりうるが実務では
/// 到達しない）は `max == min == 0.0` のときのみ spread を `0.0` として返し、
/// それ以外（0 除算で無限大・NaN になるケース）は `BenchError::NanSample`
/// として fail-closed に拒否する（本番経路 `unwrap()`/`expect()` 禁止方針
/// `.claude/rules/coding-rust.md` に基づき、無限大・NaN を判定結果として
/// 黙って伝播させない）。
pub fn relative_spread(samples: &[f64]) -> Result<f64, BenchError> {
    let Quartiles { median, .. } = median_q1_q3(samples)?;

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &s in samples {
        min = min.min(s);
        max = max.max(s);
    }

    if median == 0.0 {
        return if max == 0.0 && min == 0.0 {
            Ok(0.0)
        } else {
            Err(BenchError::NanSample)
        };
    }

    let spread = (max - min) / median;
    if spread.is_nan() {
        return Err(BenchError::NanSample);
    }
    Ok(spread)
}

/// `numerator / median` の 0 除算・非有限伝播を [`relative_spread`] と同じ
/// fail-closed 契約で判定する私的ヘルパー。
///
/// [`trimmed_relative_spread`]・[`iqr_over_median`]・[`mad2_over_median`] の
/// 3 関数はいずれも「分子（レンジ・IQR幅・2×MAD）を `median` で正規化する」
/// という構造が共通するため、`median == 0.0` 時の扱い（分子も 0 のときのみ
/// `Ok(0.0)`、それ以外は `NanSample`）と結果 NaN の拒否をここへ集約する。
/// [`relative_spread`] 本体はこのヘルパーを使わず既存のまま独立に保つ
/// （REQ-8・イシュー #1483: 既存の判定経路をバイト単位で不変に保つため）。
fn ratio_over_median(numerator: f64, median: f64) -> Result<f64, BenchError> {
    if median == 0.0 {
        return if numerator == 0.0 {
            Ok(0.0)
        } else {
            Err(BenchError::NanSample)
        };
    }
    let ratio = numerator / median;
    if ratio.is_nan() {
        return Err(BenchError::NanSample);
    }
    Ok(ratio)
}

/// サンプル列の「トリム済みレンジ」相対ばらつきを求める（判定に使わない補助統計量。
/// `docs/perf/metal-bench-noise-protocol.md` §8 案 A）。
///
/// 昇順ソート後、上下各 `trim_per_side` 個を除いた残り `n - 2*trim_per_side` 個の
/// レンジ（max − min）を、**全系列**（トリム前）の median（[`median_q1_q3`] と同一定義）
/// で正規化する。分母をトリム前 median に固定するのは
/// `docs/perf/logs/metal-bench-robust-stats-1266/reapply.py` の参照実装
/// （`trimmed_range_spread`）と定義を一致させ、既存の実測再適用値
/// （`docs/perf/metal-bench-noise-protocol.md` §8.3）と突合可能にするため。
///
/// [`ab::auxiliary_spread`](crate::ab) から呼ばれ、[`crate::ab::AuxiliarySpread::trimmed`]
/// を埋める。**判定に使わない補助レポート値であり、[`crate::ab::STABILITY_SPREAD_GATE`]
/// と比較して安定性ゲート判定に転用してはならない**（トリムはスパイクを機械的に
/// 除外するため、`relative_spread` と同じ閾値に対して実質的なゲート緩和になる。
/// `docs/perf/metal-bench-noise-protocol.md` §8.2）。
///
/// # Errors
///
/// - `samples` が空の場合は `BenchError::EmptySamples`
/// - NaN が混入している場合は `BenchError::NanSample`
/// - `samples.len() < 2 * trim_per_side + 2`（トリム後に 2 要素未満しか残らず
///   レンジを定義できない。`reapply.py` の `n - 2k < 2` 判定と同一式）の場合は
///   `BenchError::ProtocolViolation`
/// - median が 0.0 でトリム後の max/min のいずれかが非 0 の場合、
///   または結果が NaN になる場合は `BenchError::NanSample`
///   （[`relative_spread`] と同じ fail-closed 方針）
pub fn trimmed_relative_spread(samples: &[f64], trim_per_side: usize) -> Result<f64, BenchError> {
    // `median_q1_q3` を先に呼び、空・NaN のエラー優先順位を `relative_spread` と揃える。
    let Quartiles { median, .. } = median_q1_q3(samples)?;

    let n = samples.len();
    let required = 2 * trim_per_side + 2;
    if n < required {
        return Err(BenchError::ProtocolViolation(format!(
            "トリム済みレンジには samples.len() >= 2*trim_per_side+2 が必須。\
             n={n}, trim_per_side={trim_per_side}, 必要下限={required}"
        )));
    }

    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let trimmed = &sorted[trim_per_side..n - trim_per_side];

    let min = trimmed[0];
    let max = trimmed[trimmed.len() - 1];
    ratio_over_median(max - min, median)
}

/// サンプル列の IQR（四分位範囲）相対ばらつきを求める（判定に使わない補助統計量。
/// `docs/perf/metal-bench-noise-protocol.md` §8 案 C）。
///
/// `(q3 - q1) / median`（いずれも [`median_q1_q3`] の定義）。分位点自体が
/// 極端値の影響を受けにくいため、外れ値 1 点によるスパイクを自然に抑える。
///
/// [`ab::auxiliary_spread`](crate::ab) から呼ばれる。**判定に使わない補助
/// レポート値**である点は [`trimmed_relative_spread`] と同じ契約
/// （`crate::ab::STABILITY_SPREAD_GATE` への転用禁止）。
///
/// # Errors
///
/// [`relative_spread`] と同じ（空 → `EmptySamples`、NaN 混入 → `NanSample`、
/// median==0 かつ q3==q1 のときのみ `Ok(0.0)`・それ以外は `NanSample`）。
pub fn iqr_over_median(samples: &[f64]) -> Result<f64, BenchError> {
    let Quartiles { median, q1, q3 } = median_q1_q3(samples)?;
    ratio_over_median(q3 - q1, median)
}

/// サンプル列の 2×MAD（中央絶対偏差）相対ばらつきを求める（判定に使わない補助統計量。
/// `docs/perf/metal-bench-noise-protocol.md` §8 案 D）。
///
/// `2 * median(|x - median(samples)|) / median(samples)`。内側の
/// `median(|x - median|)` も [`median_q1_q3`] と同一の median-of-halves 定義で
/// 求める（偏差列に対して改めて `median_q1_q3` を適用する）。係数 2 は
/// 正規分布下で IQR とスケールを揃えるための慣用的な補正（`reapply.py`
/// `mad_over_median` を参照）。
///
/// [`ab::auxiliary_spread`](crate::ab) から呼ばれる。**判定に使わない補助
/// レポート値**である点は [`trimmed_relative_spread`] と同じ契約。
///
/// # Errors
///
/// [`relative_spread`] と同じ（空 → `EmptySamples`、NaN 混入 → `NanSample`、
/// median==0 かつ MAD==0 のときのみ `Ok(0.0)`・それ以外は `NanSample`）。
pub fn mad2_over_median(samples: &[f64]) -> Result<f64, BenchError> {
    let Quartiles { median, .. } = median_q1_q3(samples)?;
    let deviations: Vec<f64> = samples.iter().map(|&x| (x - median).abs()).collect();
    let mad = median_q1_q3(&deviations)?.median;
    ratio_over_median(2.0 * mad, median)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_count_known_distribution() {
        // 1..=9 の奇数個サンプル。中央値は 5（idx=4）。
        let samples: Vec<f64> = (1..=9).map(f64::from).collect();
        let q = median_q1_q3(&samples).expect("非空・非 NaN のため成功するはず");
        assert_eq!(q.median, 5.0);
        // n=9: q1 idx = round(0.25*8)=2 -> value 3.0, q3 idx = round(0.75*8)=6 -> value 7.0
        assert_eq!(q.q1, 3.0);
        assert_eq!(q.q3, 7.0);
    }

    #[test]
    fn even_count_known_distribution() {
        // 1..=20 の偶数個サンプル（TASK 記述例に合わせる）。
        let samples: Vec<f64> = (1..=20).map(f64::from).collect();
        let q = median_q1_q3(&samples).expect("非空・非 NaN のため成功するはず");
        // n=20: median idx = round(0.5*19)=10(round-half-to-even の丸めは Rust std の round に従う) -> value 11.0
        assert_eq!(q.median, 11.0);
        // q1 idx = round(0.25*19)=round(4.75)=5 -> value 6.0
        assert_eq!(q.q1, 6.0);
        // q3 idx = round(0.75*19)=round(14.25)=14 -> value 15.0
        assert_eq!(q.q3, 15.0);
    }

    #[test]
    fn unsorted_input_is_sorted_before_picking() {
        let samples = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        let q = median_q1_q3(&samples).expect("非空・非 NaN のため成功するはず");
        assert_eq!(q.median, 3.0);
    }

    #[test]
    fn empty_samples_is_error() {
        let samples: Vec<f64> = Vec::new();
        assert_eq!(median_q1_q3(&samples), Err(BenchError::EmptySamples));
    }

    #[test]
    fn nan_sample_is_error() {
        let samples = vec![1.0, f64::NAN, 3.0];
        assert_eq!(median_q1_q3(&samples), Err(BenchError::NanSample));
    }

    #[test]
    fn relative_spread_all_same_value_is_zero() {
        // 境界値: 全同値サンプルは max == min のため spread は必ず 0。
        let samples = vec![2.0, 2.0, 2.0, 2.0];
        assert_eq!(relative_spread(&samples), Ok(0.0));
    }

    #[test]
    fn relative_spread_monotonic_sequence_matches_expected_value() {
        // 境界値: 単調列（median-of-halves 方式で median=5, idx=4 は
        // odd_count_known_distribution と同じ n=9 系列）は
        // (max-min)/median = (9-1)/5 = 1.6 になる。
        let samples: Vec<f64> = (1..=9).map(f64::from).collect();
        let spread = relative_spread(&samples).expect("非空・非 NaN のため成功するはず");
        assert!((spread - 1.6).abs() < 1e-12);
    }

    #[test]
    fn relative_spread_propagates_empty_and_nan_errors() {
        assert_eq!(relative_spread(&[]), Err(BenchError::EmptySamples));
        assert_eq!(
            relative_spread(&[1.0, f64::NAN]),
            Err(BenchError::NanSample)
        );
    }

    // --- 補助 spread 統計量（イシュー #1483。判定に使わない） ---

    #[test]
    fn trimmed_relative_spread_k0_matches_relative_spread() {
        // k=0 は「トリムなし」であり relative_spread と完全一致するはず。
        let samples: Vec<f64> = (1..=9).map(f64::from).collect();
        let expected = relative_spread(&samples).expect("成功するはず");
        let actual = trimmed_relative_spread(&samples, 0).expect("成功するはず");
        assert!((actual - expected).abs() < 1e-12);

        let samples20: Vec<f64> = (1..=20).map(f64::from).collect();
        let expected20 = relative_spread(&samples20).expect("成功するはず");
        let actual20 = trimmed_relative_spread(&samples20, 0).expect("成功するはず");
        assert!((actual20 - expected20).abs() < 1e-12);
    }

    #[test]
    fn trimmed_relative_spread_odd_count_known_distribution() {
        // 1..=9（median=5, idx=4）。k=1 でトリム後は 2..=8（7 要素）→ レンジ 6。
        let samples: Vec<f64> = (1..=9).map(f64::from).collect();
        let actual = trimmed_relative_spread(&samples, 1).expect("成功するはず");
        // (8-2)/5 = 1.2
        assert!((actual - 1.2).abs() < 1e-12);
    }

    #[test]
    fn iqr_over_median_odd_count_known_distribution() {
        // 1..=9: q1=3.0, q3=7.0, median=5.0 -> (7-3)/5 = 0.8
        let samples: Vec<f64> = (1..=9).map(f64::from).collect();
        let actual = iqr_over_median(&samples).expect("成功するはず");
        assert!((actual - 0.8).abs() < 1e-12);
    }

    #[test]
    fn mad2_over_median_odd_count_known_distribution() {
        // 1..=9: median=5.0, 偏差=[4,3,2,1,0,1,2,3,4] をソートすると
        // [0,1,1,2,2,3,3,4,4] で median-of-halves idx=4 -> 2.0。2*2/5 = 0.8。
        let samples: Vec<f64> = (1..=9).map(f64::from).collect();
        let actual = mad2_over_median(&samples).expect("成功するはず");
        assert!((actual - 0.8).abs() < 1e-12);
    }

    #[test]
    fn even_count_iqr_and_trimmed_known_distribution() {
        // 1..=20: median=11.0, q1=6.0, q3=15.0（even_count_known_distribution 参照）。
        let samples: Vec<f64> = (1..=20).map(f64::from).collect();
        let iqr = iqr_over_median(&samples).expect("成功するはず");
        assert!((iqr - (15.0 - 6.0) / 11.0).abs() < 1e-12);

        // k=1 トリム後は 2..=19（18 要素）-> レンジ 17。
        let trimmed = trimmed_relative_spread(&samples, 1).expect("成功するはず");
        assert!((trimmed - (19.0 - 2.0) / 11.0).abs() < 1e-12);
    }

    #[test]
    fn trimmed_relative_spread_rejects_insufficient_samples() {
        // n=3, k=1 -> 必要下限 2*1+2=4 を満たさない。
        let samples = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            trimmed_relative_spread(&samples, 1),
            Err(BenchError::ProtocolViolation(_))
        ));

        // n=4, k=1 -> 必要下限 4 をちょうど満たす（中央 2 要素のレンジ）。
        let samples4 = vec![1.0, 2.0, 3.0, 4.0];
        assert!(trimmed_relative_spread(&samples4, 1).is_ok());

        // n=4, k=2 -> 必要下限 2*2+2=6 を満たさない。
        assert!(matches!(
            trimmed_relative_spread(&samples4, 2),
            Err(BenchError::ProtocolViolation(_))
        ));
    }

    #[test]
    fn auxiliary_stats_all_same_value_is_zero() {
        let samples = vec![2.0, 2.0, 2.0, 2.0];
        assert_eq!(trimmed_relative_spread(&samples, 1), Ok(0.0));
        assert_eq!(iqr_over_median(&samples), Ok(0.0));
        assert_eq!(mad2_over_median(&samples), Ok(0.0));
    }

    #[test]
    fn auxiliary_stats_all_zero_is_zero() {
        let samples = vec![0.0, 0.0, 0.0, 0.0];
        assert_eq!(trimmed_relative_spread(&samples, 1), Ok(0.0));
        assert_eq!(iqr_over_median(&samples), Ok(0.0));
        assert_eq!(mad2_over_median(&samples), Ok(0.0));
    }

    #[test]
    fn auxiliary_stats_propagate_empty_and_nan_errors() {
        assert_eq!(
            trimmed_relative_spread(&[], 1),
            Err(BenchError::EmptySamples)
        );
        assert_eq!(iqr_over_median(&[]), Err(BenchError::EmptySamples));
        assert_eq!(mad2_over_median(&[]), Err(BenchError::EmptySamples));

        let nan_samples = [1.0, f64::NAN, 3.0, 4.0];
        assert_eq!(
            trimmed_relative_spread(&nan_samples, 1),
            Err(BenchError::NanSample)
        );
        assert_eq!(iqr_over_median(&nan_samples), Err(BenchError::NanSample));
        assert_eq!(mad2_over_median(&nan_samples), Err(BenchError::NanSample));
    }

    #[test]
    fn auxiliary_stats_never_exceed_relative_spread() {
        // トリム・IQR・MAD はいずれもレンジより小さいかスパイクの影響を受けにくいため、
        // 素の relative_spread 以下になるはず（複数系列で機械的に確認）。
        let series: Vec<Vec<f64>> = vec![
            (1..=9).map(f64::from).collect(),
            (1..=20).map(f64::from).collect(),
            vec![1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0],
        ];
        for samples in series {
            let raw = relative_spread(&samples).expect("成功するはず");
            let trimmed = trimmed_relative_spread(&samples, 1).expect("成功するはず");
            let iqr = iqr_over_median(&samples).expect("成功するはず");
            assert!(trimmed <= raw + 1e-12, "trimmed={trimmed} raw={raw}");
            assert!(iqr <= raw + 1e-12, "iqr={iqr} raw={raw}");
        }
    }

    #[test]
    fn auxiliary_stats_match_reapply_reference_1255_run4_size1024() {
        // docs/perf/logs/metal-gemm-transpose-route-ab-1242/1255-phase1_run4.log:429
        // の size=1024 系列（round_medians_secs）を
        // docs/perf/logs/metal-bench-robust-stats-1266/reapply.md の該当行
        // （1255-run4 | 1024 | 0.8371 | 0.0184✓ | 0.0110✓ | ... | 0.0139✓）と突合する。
        let samples = [
            1.155958e-3,
            1.168250e-3,
            1.146875e-3,
            1.160167e-3,
            1.151459e-3,
            1.162667e-3,
            2.112750e-3,
            1.153833e-3,
            1.141542e-3,
            1.164250e-3,
        ];
        let raw = relative_spread(&samples).expect("成功するはず");
        assert!((raw - 0.8371).abs() < 1e-4, "raw={raw}");

        let trimmed = trimmed_relative_spread(&samples, 1).expect("成功するはず");
        assert!((trimmed - 0.0184).abs() < 1e-4, "trimmed={trimmed}");

        let iqr = iqr_over_median(&samples).expect("成功するはず");
        assert!((iqr - 0.0110).abs() < 1e-4, "iqr={iqr}");

        let mad2 = mad2_over_median(&samples).expect("成功するはず");
        assert!((mad2 - 0.0139).abs() < 1e-4, "mad2={mad2}");
    }
}
