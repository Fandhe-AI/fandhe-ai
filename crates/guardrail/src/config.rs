//! 閾値体系（TASK-4.1b・イシュー #105）。
//!
//! REQ-4（`docs/spec/04-requirements.md`）が定める初期推奨閾値
//! （変更行数 200 行以内・ベンチ劣化中央値 5% 以内〈5 回以上計測〉）を保持する
//! [`Thresholds`] 型と、その値域検証を提供する。PoC-3
//! （`docs/spec/03-poc/poc-3-guardrail-validity/code/guardrail.sh:22-38`）の
//! strict / default / loose 3 プリセットをそのまま踏襲し、既定値そのものは
//! 本モジュールでは変更しない（変更にはユーザー承認が必須。security.md・
//! delegation-impl.md 禁止事項）。
//!
//! v1（`rust-ai-library-v1/crates/guardrail/src/config.rs`）は `guardrail.toml`
//! の読み込み・パース（`toml` クレート依存）まで担うが、`toml` は v2 の
//! 許容依存 8 区分（`.claude/rules/deps-policy.md`）に含まれず追加は
//! ユーザー承認が必須のため、本イシューのスコープからファイル I/O を除外する
//! （実装計画 §3）。本モジュールが提供するのは「検証済み `Thresholds` 型」
//! 「組み込み既定値」「raw 値からの検証付き構築 API（[`Thresholds::from_raw`]）」
//! までであり、`guardrail.toml` パース手段の選定・接続は #104（TASK-4.1a）が
//! 行う契約とする。[`ThresholdsRaw`] はそのまま `serde::Deserialize` 可能な
//! 契約型として公開し、#104 側のパーサ実装（JSON 化・手書きパーサ・`toml`
//! 追加承認のいずれか）が決まり次第、追加コストなく接続できるようにする。
//!
//! ここで公開する [`Thresholds`] は `decision` モジュール（判定ロジック）が
//! 受け取って条件判定に使う契約 API である。

use serde::Deserialize;

use crate::error::GuardrailError;

/// ベンチ計測回数の下限。REQ-4「ベンチマーク計測は 5 回以上実施し...中央値を
/// 採用する」（`docs/spec/04-requirements.md`）に対応。単発計測での閾値判定は
/// 許可しない。
pub const MIN_BENCH_RUNS: u32 = 5;

/// 閾値プリセット名。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresetName {
    Strict,
    Default,
    Loose,
}

impl PresetName {
    /// 設定ファイル・CLI 引数上の表記（`strict`/`default`/`loose`）に対応する
    /// 文字列を返す。
    pub fn as_str(self) -> &'static str {
        match self {
            PresetName::Strict => "strict",
            PresetName::Default => "default",
            PresetName::Loose => "loose",
        }
    }
}

impl std::str::FromStr for PresetName {
    type Err = GuardrailError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "strict" => Ok(PresetName::Strict),
            "default" => Ok(PresetName::Default),
            "loose" => Ok(PresetName::Loose),
            other => Err(GuardrailError::UnknownPreset {
                preset: other.to_string(),
            }),
        }
    }
}

/// 単一プリセットの閾値の未検証形（REQ-4 の 5 条件のうち行数・ベンチの 2 条件を
/// 保持する）。
///
/// 公開 API 非破壊・build/test/clippy 全通過・ゲーミング疑いなしの残り 3 条件は
/// 数値閾値を持たないため、本構造体には含めない（`decision` モジュールの
/// 判定ロジックがそれぞれ別途判定する）。`serde(deny_unknown_fields)` により、
/// 設定ファイル側のタイポ由来の未知キーを検証前に拒否する
/// （security.md A03: 外部入力の検証）。#104（TASK-4.1a）のパーサ実装が
/// この型へ直接デシリアライズできるよう `pub` フィールドとして公開する。
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdsRaw {
    pub lines_max: u32,
    pub bench_max_pct: f64,
    pub bench_runs: u32,
}

/// 検証済みの閾値。[`ThresholdsRaw`] を値域検証した後のものだけがこの型の
/// インスタンスとして存在しうる。フィールドは非公開とし、
/// [`Thresholds::from_raw`]（値域検証を必ず経由する）以外の経路で構築できない
/// ようにすることで不変条件を型システムで強制する（v1 と同じ流儀。
/// 判定の迂回経路を作らない。security.md A08）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    lines_max: u32,
    bench_max_pct: f64,
    bench_runs: u32,
}

impl Thresholds {
    /// 変更行数の上限（REQ-4）。超過はエスカレーション対象。
    pub fn lines_max(&self) -> u32 {
        self.lines_max
    }

    /// ベンチ劣化許容％（5 回以上計測の中央値がこれを超えるとエスカレーション/却下）。
    pub fn bench_max_pct(&self) -> f64 {
        self.bench_max_pct
    }

    /// ベンチ計測回数の下限（REQ-4「5 回以上計測」。常に [`MIN_BENCH_RUNS`] 以上）。
    pub fn bench_runs(&self) -> u32 {
        self.bench_runs
    }

    /// [`ThresholdsRaw`] を値域検証し、不正なら理由付きで拒否する（fail-closed）。
    ///
    /// 設定ファイル由来の raw 値（#104 が接続するパーサの出力）・組み込み
    /// 既定値（[`Thresholds::builtin`]）のいずれもこの経路を通る。
    pub fn from_raw(preset: PresetName, raw: ThresholdsRaw) -> Result<Self, GuardrailError> {
        if raw.lines_max == 0 {
            return Err(GuardrailError::ConfigInvalidValue {
                preset: preset.as_str().to_string(),
                reason: "lines_max は 1 以上である必要があります".to_string(),
            });
        }
        if !raw.bench_max_pct.is_finite() || raw.bench_max_pct <= 0.0 {
            return Err(GuardrailError::ConfigInvalidValue {
                preset: preset.as_str().to_string(),
                reason: "bench_max_pct は有限かつ正の値である必要があります".to_string(),
            });
        }
        if raw.bench_runs < MIN_BENCH_RUNS {
            return Err(GuardrailError::ConfigInvalidValue {
                preset: preset.as_str().to_string(),
                reason: format!(
                    "bench_runs は {MIN_BENCH_RUNS} 以上である必要があります（REQ-4: 5 回以上計測）"
                ),
            });
        }
        Ok(Thresholds {
            lines_max: raw.lines_max,
            bench_max_pct: raw.bench_max_pct,
            bench_runs: raw.bench_runs,
        })
    }

    /// PoC-3（`guardrail.sh:22-38`）の組み込み既定値を返す。設定ファイル未指定
    /// （`--config` 省略かつリポジトリルートに `guardrail.toml` が存在しない）
    /// 場合に #104 側が使う安全側の初期値。数値は spec 確定値そのままであり、
    /// ここでも変更しない。組み込み定数を [`Thresholds::from_raw`] に通す
    /// ことで検証経路を 1 本化し（config.rs 内で検証をバイパスする特別扱いを
    /// 作らない）、値域を満たさない定数への劣化を回帰テストで検知できるように
    /// する。
    pub fn builtin(preset: PresetName) -> Result<Self, GuardrailError> {
        let raw = match preset {
            PresetName::Strict => ThresholdsRaw {
                lines_max: 100,
                bench_max_pct: 3.0,
                bench_runs: 5,
            },
            PresetName::Default => ThresholdsRaw {
                lines_max: 200,
                bench_max_pct: 5.0,
                bench_runs: 5,
            },
            PresetName::Loose => ThresholdsRaw {
                lines_max: 400,
                bench_max_pct: 10.0,
                bench_runs: 5,
            },
        };
        Thresholds::from_raw(preset, raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_defaults_match_poc3_default_preset() {
        let thresholds =
            Thresholds::builtin(PresetName::Default).expect("組み込み既定値の検証に失敗");
        assert_eq!(thresholds.lines_max(), 200);
        assert_eq!(thresholds.bench_max_pct(), 5.0);
        assert_eq!(thresholds.bench_runs(), 5);
    }

    #[test]
    fn builtin_defaults_match_poc3_strict_preset() {
        let thresholds =
            Thresholds::builtin(PresetName::Strict).expect("組み込み既定値の検証に失敗");
        assert_eq!(thresholds.lines_max(), 100);
        assert_eq!(thresholds.bench_max_pct(), 3.0);
        assert_eq!(thresholds.bench_runs(), 5);
    }

    #[test]
    fn builtin_defaults_match_poc3_loose_preset() {
        let thresholds =
            Thresholds::builtin(PresetName::Loose).expect("組み込み既定値の検証に失敗");
        assert_eq!(thresholds.lines_max(), 400);
        assert_eq!(thresholds.bench_max_pct(), 10.0);
        assert_eq!(thresholds.bench_runs(), 5);
    }

    fn valid_raw() -> ThresholdsRaw {
        ThresholdsRaw {
            lines_max: 200,
            bench_max_pct: 5.0,
            bench_runs: 5,
        }
    }

    #[test]
    fn from_raw_accepts_valid_values() {
        let thresholds =
            Thresholds::from_raw(PresetName::Default, valid_raw()).expect("正常系の値域検証に失敗");
        assert_eq!(thresholds.lines_max(), 200);
        assert_eq!(thresholds.bench_max_pct(), 5.0);
    }

    #[test]
    fn rejects_lines_max_zero() {
        let raw = ThresholdsRaw {
            lines_max: 0,
            ..valid_raw()
        };
        let err = Thresholds::from_raw(PresetName::Strict, raw).unwrap_err();
        assert!(matches!(err, GuardrailError::ConfigInvalidValue { .. }));
    }

    #[test]
    fn rejects_bench_runs_below_minimum() {
        let raw = ThresholdsRaw {
            bench_runs: MIN_BENCH_RUNS - 1,
            ..valid_raw()
        };
        let err = Thresholds::from_raw(PresetName::Strict, raw).unwrap_err();
        assert!(matches!(err, GuardrailError::ConfigInvalidValue { .. }));
    }

    #[test]
    fn rejects_bench_max_pct_infinite() {
        let raw = ThresholdsRaw {
            bench_max_pct: f64::INFINITY,
            ..valid_raw()
        };
        let err = Thresholds::from_raw(PresetName::Strict, raw).unwrap_err();
        assert!(matches!(err, GuardrailError::ConfigInvalidValue { .. }));
    }

    #[test]
    fn rejects_bench_max_pct_nan() {
        let raw = ThresholdsRaw {
            bench_max_pct: f64::NAN,
            ..valid_raw()
        };
        let err = Thresholds::from_raw(PresetName::Strict, raw).unwrap_err();
        assert!(matches!(err, GuardrailError::ConfigInvalidValue { .. }));
    }

    #[test]
    fn rejects_bench_max_pct_non_positive() {
        let raw = ThresholdsRaw {
            bench_max_pct: 0.0,
            ..valid_raw()
        };
        let err = Thresholds::from_raw(PresetName::Strict, raw).unwrap_err();
        assert!(matches!(err, GuardrailError::ConfigInvalidValue { .. }));
    }

    #[test]
    fn unknown_preset_name_is_rejected_at_parse() {
        let err = "yolo".parse::<PresetName>().unwrap_err();
        assert!(matches!(err, GuardrailError::UnknownPreset { .. }));
    }

    #[test]
    fn preset_name_round_trips_through_as_str_and_from_str() {
        for preset in [PresetName::Strict, PresetName::Default, PresetName::Loose] {
            let parsed: PresetName = preset.as_str().parse().expect("as_str の逆変換に失敗");
            assert_eq!(parsed, preset);
        }
    }
}
