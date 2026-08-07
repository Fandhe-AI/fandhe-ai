//! `guardrail.toml` のスキーマ・値域検証・組み込み既定値。
//!
//! v1（`Fandhe-AI/rust-ai-library-v1/crates/guardrail/src/config.rs`）の
//! `PresetName`／`ThresholdsRaw` 相当を移植する。数値そのもの（REQ-4 初期推奨値:
//! 変更行数 200 行以内・ベンチ劣化中央値 5% 以内・5 回以上計測、strict/default/loose
//! 3 プリセット）は spec 確定値をそのまま踏襲し、本イシューでは変更しない
//! （`.claude/rules/security.md`「ガードレール閾値の変更はユーザー承認必須」）。
//!
//! `cli.rs` から `--config`／`--repo`／`--preset` を受け取り、
//! `docs/guardrail-self-repair-cli.md` 2.4 節の探索順序
//! （`--config` 指定 → `--repo` 直下の `guardrail.toml` → 組み込み既定値）で
//! 解決される（`main.rs::resolve_config` から呼ばれる）。

use std::fs;
use std::path::{Path, PathBuf};

use crate::error::GuardrailError;
use crate::toml_lite::{self, TomlValue};

/// 3 段階の判定閾値プリセット（PoC-3 由来。2.4 節）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresetName {
    Strict,
    Default,
    Loose,
}

impl PresetName {
    /// `--preset` の文字列表現から解決する。未知の値は usage エラー（終了コード `2`）。
    pub fn parse(s: &str) -> Result<Self, GuardrailError> {
        match s {
            "strict" => Ok(PresetName::Strict),
            "default" => Ok(PresetName::Default),
            "loose" => Ok(PresetName::Loose),
            other => Err(GuardrailError::UsageError(format!(
                "unknown --preset value '{other}' (expected strict|default|loose)"
            ))),
        }
    }
}

/// 判定閾値（5 条件のうち数値で表現できるもの）。
///
/// REQ-4 初期推奨値をそのまま組み込み既定値とする（v1・PoC-3 準拠。数値の
/// 変更はユーザー承認必須のため、本モジュールでは定数として固定し CLI 引数
/// からの直接注入経路を設けない。2.4 節「非対称設計」）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// 変更行数の上限（この値以内なら自動適用の候補になりうる）。
    pub lines_max: u64,
    /// ベンチ劣化中央値の許容上限（%）。
    pub bench_median_max_pct: f64,
    /// ベンチ計測の最低回数（REQ-4「5 回以上」の受け入れ基準）。
    pub bench_runs_min: u32,
}

impl Thresholds {
    /// 組み込み既定値（REQ-4 初期推奨値・v1/PoC-3 準拠）をプリセットごとに返す。
    /// strict はより厳しく（閾値を絞る）、loose はより緩い側に振る PoC-3 の
    /// 3 段階設計を踏襲する。
    pub fn builtin(preset: PresetName) -> Self {
        match preset {
            PresetName::Strict => Thresholds {
                lines_max: 100,
                bench_median_max_pct: 2.5,
                bench_runs_min: 5,
            },
            PresetName::Default => Thresholds {
                lines_max: 200,
                bench_median_max_pct: 5.0,
                bench_runs_min: 5,
            },
            PresetName::Loose => Thresholds {
                lines_max: 400,
                bench_median_max_pct: 10.0,
                bench_runs_min: 5,
            },
        }
    }

    /// 値域検証。`bench_runs_min < 5` は REQ-4 の受け入れ基準に反するため拒否する。
    fn validate(&self) -> Result<(), GuardrailError> {
        if self.lines_max == 0 {
            return Err(GuardrailError::InvalidInput(
                "lines_max must be greater than 0".to_string(),
            ));
        }
        if !(self.bench_median_max_pct.is_finite() && self.bench_median_max_pct >= 0.0) {
            return Err(GuardrailError::InvalidInput(
                "bench_median_max_pct must be a finite value >= 0".to_string(),
            ));
        }
        if self.bench_runs_min < 5 {
            return Err(GuardrailError::InvalidInput(
                "bench_runs_min must be >= 5 (REQ-4 acceptance criteria)".to_string(),
            ));
        }
        Ok(())
    }
}

/// 解決済み `guardrail.toml` 設定。TASK-4.1a では `Thresholds` のみを保持する
/// （REQ-5 除外ルールは `policy-exclusion.toml` として #118 で別途扱う。2.4 節）。
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub thresholds: Thresholds,
}

/// `--config` → `--repo` 直下 → 組み込み既定値、の順で `guardrail.toml` を解決する
/// （2.4 節・計画 4 節ステップ 4）。`main.rs` から呼ばれる。
pub fn resolve(
    explicit_path: Option<&Path>,
    repo_root: &Path,
    preset: PresetName,
) -> Result<Config, GuardrailError> {
    if let Some(path) = explicit_path {
        return load_file(path, preset);
    }
    let candidate = repo_root.join("guardrail.toml");
    if candidate.is_file() {
        return load_file(&candidate, preset);
    }
    Ok(Config {
        thresholds: Thresholds::builtin(preset),
    })
}

fn load_file(path: &Path, preset: PresetName) -> Result<Config, GuardrailError> {
    let metadata = fs::metadata(path).map_err(|source| GuardrailError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.len() > toml_lite::MAX_INPUT_BYTES as u64 {
        return Err(GuardrailError::InvalidInput(format!(
            "{} exceeds {} byte limit",
            path.display(),
            toml_lite::MAX_INPUT_BYTES
        )));
    }
    let content = fs::read_to_string(path).map_err(|source| GuardrailError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_config(&content, preset)
}

/// テーブル名 `preset.<name>`（`toml_lite` はネストを解釈しないため、
/// `[preset.strict]` はテーブル名そのものとして扱われる）から
/// `lines_max`／`bench_median_max_pct`／`bench_runs_min` の 3 フィールドのみを
/// 明示照合で読み取る。それ以外のキーは未知フィールドとして拒否する
/// （2.5 節「`deny_unknown_fields` 相当」）。
fn parse_config(content: &str, preset: PresetName) -> Result<Config, GuardrailError> {
    let doc = toml_lite::parse(content)?;
    let table_name = format!("preset.{}", preset_key(preset));
    let Some(table) = doc.get(&table_name) else {
        // 該当プリセットのテーブルが無い設定ファイルは、組み込み既定値をそのまま使う
        // （設定ファイルは「一部のプリセットだけ上書きする」用途を許容する）。
        let thresholds = Thresholds::builtin(preset);
        thresholds.validate()?;
        return Ok(Config { thresholds });
    };

    let mut lines_max: Option<u64> = None;
    let mut bench_median_max_pct: Option<f64> = None;
    let mut bench_runs_min: Option<u32> = None;

    for (key, value) in table {
        match key.as_str() {
            "lines_max" => {
                lines_max = Some(expect_u64(value, key)?);
            }
            "bench_median_max_pct" => {
                bench_median_max_pct = Some(expect_f64(value, key)?);
            }
            "bench_runs_min" => {
                bench_runs_min = Some(expect_u64(value, key)? as u32);
            }
            unknown => {
                return Err(GuardrailError::InvalidInput(format!(
                    "unknown field '{unknown}' in [{table_name}] (deny_unknown_fields)"
                )));
            }
        }
    }

    let builtin = Thresholds::builtin(preset);
    let thresholds = Thresholds {
        lines_max: lines_max.unwrap_or(builtin.lines_max),
        bench_median_max_pct: bench_median_max_pct.unwrap_or(builtin.bench_median_max_pct),
        bench_runs_min: bench_runs_min.unwrap_or(builtin.bench_runs_min),
    };
    thresholds.validate()?;
    Ok(Config { thresholds })
}

fn preset_key(preset: PresetName) -> &'static str {
    match preset {
        PresetName::Strict => "strict",
        PresetName::Default => "default",
        PresetName::Loose => "loose",
    }
}

fn expect_u64(value: &TomlValue, key: &str) -> Result<u64, GuardrailError> {
    match value {
        TomlValue::Integer(i) if *i >= 0 => Ok(*i as u64),
        _ => Err(GuardrailError::InvalidInput(format!(
            "field '{key}' must be a non-negative integer"
        ))),
    }
}

fn expect_f64(value: &TomlValue, key: &str) -> Result<f64, GuardrailError> {
    match value {
        TomlValue::Float(f) => Ok(*f),
        TomlValue::Integer(i) => Ok(*i as f64),
        _ => Err(GuardrailError::InvalidInput(format!(
            "field '{key}' must be a number"
        ))),
    }
}

/// テスト・`main.rs` の共通ヘルパ: `--config` が指すパスを `PathBuf` へ正規化する。
pub fn as_explicit_path(raw: &str) -> PathBuf {
    PathBuf::from(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_defaults_match_req4_initial_recommendation() {
        let t = Thresholds::builtin(PresetName::Default);
        assert_eq!(t.lines_max, 200);
        assert_eq!(t.bench_median_max_pct, 5.0);
        assert_eq!(t.bench_runs_min, 5);
    }

    #[test]
    fn parse_config_accepts_partial_override() {
        let cfg = parse_config("[preset.default]\nlines_max = 150\n", PresetName::Default).unwrap();
        assert_eq!(cfg.thresholds.lines_max, 150);
        // 未指定フィールドは組み込み既定値にフォールバックする。
        assert_eq!(cfg.thresholds.bench_median_max_pct, 5.0);
    }

    #[test]
    fn parse_config_rejects_unknown_field() {
        let err =
            parse_config("[preset.default]\nunknown_field = 1\n", PresetName::Default).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn parse_config_rejects_bench_runs_below_five() {
        let err = parse_config(
            "[preset.default]\nbench_runs_min = 3\n",
            PresetName::Default,
        )
        .unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn resolve_falls_back_to_builtin_when_no_file_present() {
        let dir = std::env::temp_dir().join(format!(
            "guardrail-config-test-{}-{}",
            std::process::id(),
            "resolve-fallback"
        ));
        let _ = fs::create_dir_all(&dir);
        let cfg = resolve(None, &dir, PresetName::Default).unwrap();
        assert_eq!(cfg.thresholds, Thresholds::builtin(PresetName::Default));
        let _ = fs::remove_dir_all(&dir);
    }
}
