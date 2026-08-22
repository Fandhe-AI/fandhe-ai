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
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchError::EmptySamples => write!(f, "サンプル列が空のため分位点を計算できない"),
            BenchError::NanSample => write!(f, "サンプル列に NaN が混入している"),
            BenchError::ProtocolViolation(msg) => write!(f, "計測プロトコル違反: {msg}"),
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
}
