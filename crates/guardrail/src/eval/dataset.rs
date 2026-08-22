//! ラベル付きデータセット（`tests/fixtures/labeled-changes/changes/`）の
//! 列挙・パース・判定入力への変換。
//!
//! `guardrail eval`（`crate::eval::run`。TASK-4.3a・イシュー #115）から呼ばれる。
//! `changes/<change_id>/{meta.toml,poc3-result.json}` の 2 ファイルを読み、
//! `crate::decision::DecisionInput` を構築できる形（[`LabeledChange`]）へ写像する。
//! 判定ロジック自体（`decide` の呼び出し）は本モジュールの責務外であり
//! `crate::eval`（`mod.rs`）側が担う（本モジュールは「読み取り・検証・写像」に徹する）。
//!
//! # スコープ境界
//! - `meta.toml` は `change_id`・`category`・`expected_verdict`・
//!   `known_blindspot` の 4 フィールドのみを読む。
//!   `poc3_default_verdict`・`origin`・`summary`・`expected_exclusion_rule_ids`・
//!   `expected_verdict_after_exclusions` は v1 参照値・REQ-5 除外リスト系
//!   フィールドであり `eval`（判定器単体の経路。1.1 節「設計制約」）の判定には
//!   使わない（存在は許容するが値は無視する。ファイル自体には他所〈`tests/labeled_changes_labels.rs`〉
//!   がスキーマ完全一致で検証済み）。
//! - `poc3-result.json` は必須フィールド欠落の検出のみを行う前方互換パース
//!   （`docs/guardrail-self-repair-cli.md` 2.5 節「`--dataset` は必須フィールド
//!   欠落の検出のみ」）。`preset`・`lines_max`・`bench_max_pct`・`reasons` 等の
//!   v1 参照値フィールドは判定に使わない。
//!
//! # セキュリティ（A03。`.claude/rules/security.md`）
//! `change_id`（`changes/` 配下のディレクトリ名）はパス連結**前**に文字クラス
//! （`[A-Za-z0-9._-]+`・64 字以内）を検証し、`--dataset` ルート外参照
//! （パストラバーサル）を遮断する（`is_valid_change_id`。
//! `tests/labeled_changes_fixtures.rs`／`tests/labeled_changes_labels.rs` の
//! 同名関数と同一契約）。`meta.toml`／`poc3-result.json` はいずれも読み込み前に
//! 64 KiB 上限（[`crate::toml_lite::MAX_INPUT_BYTES`]）を検査する。

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::decision::{BenchSignal, GateSignal, GateSignals, Verdict};
use crate::error::GuardrailError;
use crate::toml_lite::{self, MAX_INPUT_BYTES, TomlValue};

/// `changes/` 配下 1 件分の評価入力（判定入力＋正解ラベル）。
///
/// `gates`・`bench` は [`crate::decision::DecisionInput::new`] へそのまま渡す
/// ための形に整形済み（`Copy` 型のみで構成）。
#[derive(Debug, Clone)]
pub struct LabeledChange {
    pub change_id: String,
    /// `"safe"` | `"dangerous"` | `"gray"`（README「ラベル基準」節の 3 分類）。
    pub category: String,
    /// 判定器単体の正解ラベル（REQ-4 受け入れ基準。除外リスト適用前。
    /// `docs/guardrail-self-repair-cli.md` 1.1 節「設計制約」）。
    pub expected_verdict: Verdict,
    pub known_blind_spot: bool,
    pub lines_changed: u64,
    pub gates: GateSignals,
    pub api_broken: bool,
    pub gaming_suspect: bool,
    pub bench: BenchSignal,
}

/// change_id（`changes/` 配下のディレクトリ名）の文字クラス契約。
/// `tests/labeled_changes_fixtures.rs::is_valid_change_id` と同一契約
/// （英数字始まり・`[A-Za-z0-9._-]` のみ・64 字以内。パストラバーサル対策）。
fn is_valid_change_id(id: &str) -> bool {
    if id.is_empty() || id.len() > 64 {
        return false;
    }
    let mut chars = id.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_alphanumeric() {
        return false;
    }
    id.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
}

/// `--dataset` ディレクトリ配下の `changes/*/` を列挙し、文字クラス検証を
/// 通過した change_id のみを昇順で返す（検証前の名前を path join に使わない。
/// A03 対策）。
pub fn list_change_ids(dataset_dir: &Path) -> Result<Vec<String>, GuardrailError> {
    if !dataset_dir.is_dir() {
        return Err(GuardrailError::InvalidInput(format!(
            "dataset directory '{}' does not exist",
            dataset_dir.display()
        )));
    }
    let changes_dir = dataset_dir.join("changes");
    if !changes_dir.is_dir() {
        return Err(GuardrailError::InvalidInput(format!(
            "'{}' に 'changes' ディレクトリが存在しません",
            dataset_dir.display()
        )));
    }

    let entries = fs::read_dir(&changes_dir).map_err(|source| GuardrailError::Io {
        path: changes_dir.clone(),
        source,
    })?;

    let mut ids = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| GuardrailError::Io {
            path: changes_dir.clone(),
            source,
        })?;
        let file_type = entry.file_type().map_err(|source| GuardrailError::Io {
            path: changes_dir.clone(),
            source,
        })?;
        if !file_type.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if !is_valid_change_id(&name) {
            return Err(GuardrailError::InvalidInput(format!(
                "changes/ 配下のディレクトリ名 '{name}' が文字クラス契約\
                 （[A-Za-z0-9._-]+・64 字以内）を満たさない（A03: path join 前に遮断）"
            )));
        }
        ids.push(name);
    }
    ids.sort();
    Ok(ids)
}

/// `change_id` 1 件分の `meta.toml`／`poc3-result.json` を読み、
/// [`LabeledChange`] へ変換する。`change_id` は呼び出し前提として
/// `is_valid_change_id` を満たすこと（[`list_change_ids`] が返す値を
/// そのまま渡す想定。未検証の外部文字列を直接渡さない）。
pub fn load_change(dataset_dir: &Path, change_id: &str) -> Result<LabeledChange, GuardrailError> {
    if !is_valid_change_id(change_id) {
        return Err(GuardrailError::InvalidInput(format!(
            "change_id '{change_id}' が文字クラス契約を満たさない（A03: path join 前に遮断）"
        )));
    }
    let dir = dataset_dir.join("changes").join(change_id);

    let meta = load_meta(&dir.join("meta.toml"), change_id)?;
    let poc3 = load_poc3(&dir.join("poc3-result.json"), change_id)?;

    let gates = GateSignals {
        build: bool_to_gate(poc3.build_ok),
        test: bool_to_gate(poc3.test_ok),
        clippy: bool_to_gate(poc3.clippy_ok),
    };

    // PoC-3 の実行順序契約（「ベンチはゲート全通過時のみ計測する」）が
    // `poc3-result.json` の記録済み値としても成立していることをここで検証する
    // （`DecisionInput::new` が同じ契約を構築時に再検証するが、ここでは
    // 「ベンチ計測ありなのに `bench_median_pct` が欠落」という dataset 自体の
    // 記録不備を明確な理由で先に検出する。fail-closed。`.claude/rules/security.md` A08）。
    let bench = match (poc3.bench_ran, poc3.bench_median_pct) {
        (true, Some(median_pct)) => BenchSignal::Measured { median_pct },
        (true, None) => {
            return Err(GuardrailError::InvalidInput(format!(
                "change_id '{change_id}': poc3-result.json の bench_ran=true に対し\
                 bench_median_pct が欠落（null）している"
            )));
        }
        (false, _) => BenchSignal::NotRun,
    };

    Ok(LabeledChange {
        change_id: meta.change_id,
        category: meta.category,
        expected_verdict: meta.expected_verdict,
        known_blind_spot: meta.known_blind_spot,
        lines_changed: poc3.lines_changed,
        gates,
        api_broken: poc3.api_broken,
        gaming_suspect: poc3.gaming_suspect,
        bench,
    })
}

fn bool_to_gate(ok: bool) -> GateSignal {
    if ok {
        GateSignal::Passed
    } else {
        GateSignal::Failed
    }
}

/// `meta.toml` から読み取る最小フィールド集合。
struct MetaFields {
    change_id: String,
    category: String,
    expected_verdict: Verdict,
    known_blind_spot: bool,
}

const CATEGORY_VALUES: &[&str] = &["safe", "dangerous", "gray"];

/// `meta.toml` を読み `MetaFields` を返す。README の 9 フィールドスキーマ
/// （`tests/labeled_changes_labels.rs::REQUIRED_META_KEYS`）のうち、判定に
/// 使う 4 フィールドのみを明示照合で読み取る（モジュールコメント「スコープ
/// 境界」節）。`crate::toml_lite::parse` は未知フィールドを拒否しない汎用
/// パーサのため、他フィールド（`origin`・`summary` 等）が存在してもここでは
/// エラーにしない。
fn load_meta(path: &Path, change_id: &str) -> Result<MetaFields, GuardrailError> {
    let metadata = fs::metadata(path).map_err(|source| GuardrailError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.len() > MAX_INPUT_BYTES as u64 {
        return Err(GuardrailError::InvalidInput(format!(
            "{} exceeds {MAX_INPUT_BYTES} byte limit",
            path.display()
        )));
    }
    let content = fs::read_to_string(path).map_err(|source| GuardrailError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let doc = toml_lite::parse(&content)?;
    let table = doc.get("").ok_or_else(|| {
        GuardrailError::InvalidInput(format!("{}: ルートテーブルが存在しない", path.display()))
    })?;

    let meta_change_id = expect_string(table, "change_id", path)?;
    if meta_change_id != change_id {
        return Err(GuardrailError::InvalidInput(format!(
            "{}: change_id フィールド '{meta_change_id}' がディレクトリ名 '{change_id}' と一致しない",
            path.display()
        )));
    }

    let category = expect_string(table, "category", path)?;
    if !CATEGORY_VALUES.contains(&category.as_str()) {
        return Err(GuardrailError::InvalidInput(format!(
            "{}: category '{category}' が許可値 {CATEGORY_VALUES:?} に含まれない",
            path.display()
        )));
    }

    let expected_verdict_raw = expect_string(table, "expected_verdict", path)?;
    let expected_verdict = parse_verdict(&expected_verdict_raw, path)?;

    let known_blind_spot = expect_bool(table, "known_blindspot", path)?;

    Ok(MetaFields {
        change_id: meta_change_id,
        category,
        expected_verdict,
        known_blind_spot,
    })
}

fn expect_string(
    table: &BTreeMap<String, TomlValue>,
    key: &str,
    path: &Path,
) -> Result<String, GuardrailError> {
    match table.get(key) {
        Some(TomlValue::String(s)) => Ok(s.clone()),
        Some(_) => Err(GuardrailError::InvalidInput(format!(
            "{}: field '{key}' must be a string",
            path.display()
        ))),
        None => Err(GuardrailError::InvalidInput(format!(
            "{}: missing required field '{key}'",
            path.display()
        ))),
    }
}

fn expect_bool(
    table: &BTreeMap<String, TomlValue>,
    key: &str,
    path: &Path,
) -> Result<bool, GuardrailError> {
    match table.get(key) {
        Some(TomlValue::Bool(b)) => Ok(*b),
        Some(_) => Err(GuardrailError::InvalidInput(format!(
            "{}: field '{key}' must be a bool",
            path.display()
        ))),
        None => Err(GuardrailError::InvalidInput(format!(
            "{}: missing required field '{key}'",
            path.display()
        ))),
    }
}

/// `meta.toml` の `expected_verdict`（ハイフン表記。README・`decision::Verdict`
/// との語彙差は `tests/labeled_changes_labels.rs::verdict_id_to_ja` と同種）を
/// `Verdict` へ変換する。`match` は網羅列挙とし `_ =>` を使わない
/// （fail-closed。`.claude/rules/security.md` A05）。
fn parse_verdict(raw: &str, path: &Path) -> Result<Verdict, GuardrailError> {
    match raw {
        "auto-apply" => Ok(Verdict::AutoApply),
        "escalate" => Ok(Verdict::Escalate),
        "reject" => Ok(Verdict::Reject),
        other => Err(GuardrailError::InvalidInput(format!(
            "{}: unknown expected_verdict value '{other}' (expected auto-apply|escalate|reject)",
            path.display()
        ))),
    }
}

/// `poc3-result.json` から読み取る最小フィールド集合（前方互換パース。
/// モジュールコメント「スコープ境界」節）。
struct Poc3Fields {
    lines_changed: u64,
    api_broken: bool,
    gaming_suspect: bool,
    build_ok: bool,
    test_ok: bool,
    clippy_ok: bool,
    bench_ran: bool,
    bench_median_pct: Option<f64>,
}

fn load_poc3(path: &Path, change_id: &str) -> Result<Poc3Fields, GuardrailError> {
    let metadata = fs::metadata(path).map_err(|source| GuardrailError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.len() > MAX_INPUT_BYTES as u64 {
        return Err(GuardrailError::InvalidInput(format!(
            "{} exceeds {MAX_INPUT_BYTES} byte limit",
            path.display()
        )));
    }
    let content = fs::read_to_string(path).map_err(|source| GuardrailError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let value: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
        GuardrailError::InvalidInput(format!("{}: invalid JSON: {e}", path.display()))
    })?;
    let obj = value.as_object().ok_or_else(|| {
        GuardrailError::InvalidInput(format!(
            "{}: JSON のルートがオブジェクトでない",
            path.display()
        ))
    })?;

    let json_change_id = obj
        .get("change_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            GuardrailError::InvalidInput(format!(
                "{}: missing or non-string required field 'change_id'",
                path.display()
            ))
        })?;
    if json_change_id != change_id {
        return Err(GuardrailError::InvalidInput(format!(
            "{}: change_id フィールド '{json_change_id}' がディレクトリ名 '{change_id}' と一致しない",
            path.display()
        )));
    }

    let lines_changed = expect_json_u64(obj, "lines_changed", path)?;
    let api_broken = expect_json_bool(obj, "api_broken", path)?;
    let gaming_suspect = expect_json_bool(obj, "gaming_suspect", path)?;
    let build_ok = expect_json_bool(obj, "build_ok", path)?;
    let test_ok = expect_json_bool(obj, "test_ok", path)?;
    let clippy_ok = expect_json_bool(obj, "clippy_ok", path)?;
    let bench_ran = expect_json_bool(obj, "bench_ran", path)?;
    let bench_median_pct = match obj.get("bench_median_pct") {
        Some(v) if v.is_null() => None,
        Some(v) => Some(v.as_f64().ok_or_else(|| {
            GuardrailError::InvalidInput(format!(
                "{}: field 'bench_median_pct' must be a number or null",
                path.display()
            ))
        })?),
        None => {
            return Err(GuardrailError::InvalidInput(format!(
                "{}: missing required field 'bench_median_pct'",
                path.display()
            )));
        }
    };

    Ok(Poc3Fields {
        lines_changed,
        api_broken,
        gaming_suspect,
        build_ok,
        test_ok,
        clippy_ok,
        bench_ran,
        bench_median_pct,
    })
}

fn expect_json_u64(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    path: &Path,
) -> Result<u64, GuardrailError> {
    obj.get(key).and_then(|v| v.as_u64()).ok_or_else(|| {
        GuardrailError::InvalidInput(format!(
            "{}: missing or non-u64 required field '{key}'",
            path.display()
        ))
    })
}

fn expect_json_bool(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    path: &Path,
) -> Result<bool, GuardrailError> {
    obj.get(key).and_then(|v| v.as_bool()).ok_or_else(|| {
        GuardrailError::InvalidInput(format!(
            "{}: missing or non-bool required field '{key}'",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// 実 dataset（15 件）に対する列挙・読み込みの疎通確認。詳細な正誤判定は
    /// `crate::eval::run` 経由の統合テスト（`tests/eval_harness.rs`）で行う。
    fn real_dataset_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/labeled-changes")
    }

    #[test]
    fn list_change_ids_finds_all_fifteen_real_fixtures() {
        let ids = list_change_ids(&real_dataset_dir()).expect("列挙に失敗");
        assert_eq!(ids.len(), 15, "実 dataset は 15 件の想定: {ids:?}");
    }

    #[test]
    fn load_change_reads_dangerous_reject_case() {
        let change =
            load_change(&real_dataset_dir(), "D1-relu-sigmoid-swap").expect("読み込みに失敗");
        assert_eq!(change.category, "dangerous");
        assert_eq!(change.expected_verdict, Verdict::Reject);
        assert!(!change.known_blind_spot);
        assert_eq!(change.gates.test, GateSignal::Failed);
        assert_eq!(change.bench, BenchSignal::NotRun);
    }

    #[test]
    fn load_change_reads_known_blindspot_case() {
        let change =
            load_change(&real_dataset_dir(), "G2-hidden-dim-increase").expect("読み込みに失敗");
        assert_eq!(change.category, "gray");
        assert_eq!(change.expected_verdict, Verdict::Escalate);
        assert!(change.known_blind_spot);
        assert_eq!(
            change.gates,
            GateSignals {
                build: GateSignal::Passed,
                test: GateSignal::Passed,
                clippy: GateSignal::Passed,
            }
        );
        assert!(matches!(change.bench, BenchSignal::Measured { .. }));
    }

    #[test]
    fn list_change_ids_rejects_missing_dataset_dir() {
        let missing = std::env::temp_dir().join(format!(
            "guardrail-eval-dataset-test-missing-{}",
            std::process::id()
        ));
        let err = list_change_ids(&missing).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn load_change_rejects_invalid_change_id_before_path_join() {
        let err = load_change(&real_dataset_dir(), "../../etc").unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }
}
