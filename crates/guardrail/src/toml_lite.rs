//! `guardrail.toml` が必要とする限定サブセットのみを解釈する自作 TOML パーサ。
//!
//! `toml` クレートは `.claude/rules/deps-policy.md` の許容依存 8 区分に含まれず、
//! 依存追加はユーザー承認事項のため（TASK-4.1a 計画 2.1 節）、std のみで
//! 「`[table]` ヘッダ＋ `key = value`（整数／浮動小数／文字列／真偽値）の
//! フラットな key-value」という限定文法のみを実装する。ネスト配列・インライン
//! テーブル・複数行文字列等の TOML 本来の機能は対象外（`config.rs` が読む
//! `guardrail.toml` の形が要求しないため）。
//!
//! `config.rs` から呼ばれ、`GuardrailError::InvalidInput` を返すことで
//! 呼び出し元が終了コード `1`（内部エラー）へ変換できるようにする
//! （`docs/guardrail-self-repair-cli.md` 2.5 節 A03 対策: 外部入力はパース時に検証する）。

use std::collections::BTreeMap;

use crate::error::GuardrailError;

/// パース後の値。`guardrail.toml` の用途では整数・浮動小数・文字列・真偽値のみで足りる。
#[derive(Debug, Clone, PartialEq)]
pub enum TomlValue {
    Integer(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

/// テーブル名（ルートは空文字列）→ key → 値、のフラットな解釈結果。
///
/// `config.rs` はこの構造を明示フィールド照合で読み取り、未知フィールドを
/// 拒否する（2.5 節「未知フィールドを拒否」）。未知フィールド拒否は
/// `toml_lite` 自体ではなく呼び出し元の責務とする（テーブル構造自体は
/// 汎用的に保つため）。
pub type TomlDocument = BTreeMap<String, BTreeMap<String, TomlValue>>;

/// 外部入力のサイズ上限（`docs/guardrail-self-repair-cli.md` 2.5 節・
/// 計画 4 節ステップ 4「64 KiB 読み込み上限」）。DoS 的な巨大入力を
/// パース前に拒否する（A03 対策）。
pub const MAX_INPUT_BYTES: usize = 64 * 1024;

/// 限定サブセット TOML 文字列をパースする。
///
/// 対応文法:
/// - 空行・`#` 行コメントの無視
/// - `[section]` テーブルヘッダ（`section` は不透明な文字列として扱う。
///   `[preset.default]` のようなドット表記は TOML 本来のネスト解釈をせず、
///   テーブル名そのものとして 1 対 1 で保持する）
/// - `key = value` （`value` は整数・浮動小数・`true`/`false`・二重引用符文字列）
/// - 同一テーブル内の重複キーは拒否（設定の意図しない上書きを防ぐ）
pub fn parse(input: &str) -> Result<TomlDocument, GuardrailError> {
    if input.len() > MAX_INPUT_BYTES {
        return Err(GuardrailError::InvalidInput(format!(
            "input exceeds {MAX_INPUT_BYTES} byte limit ({} bytes)",
            input.len()
        )));
    }

    let mut doc: TomlDocument = TomlDocument::new();
    let mut current_table = String::new();
    doc.insert(current_table.clone(), BTreeMap::new());

    for (lineno, raw_line) in input.lines().enumerate() {
        let line_number = lineno + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if let Some(stripped) = line.strip_prefix('[') {
            let name = stripped.strip_suffix(']').ok_or_else(|| {
                GuardrailError::InvalidInput(format!(
                    "line {line_number}: malformed table header (missing ']')"
                ))
            })?;
            let name = name.trim();
            // ドット（例: `[preset.default]`）は TOML 本来のネスト解釈をせず、
            // テーブル名そのものに含まれる不透明な文字として扱う（`config.rs` が
            // `preset.<name>` という文字列キーで直接照合するため。真のネスト構造
            // （`[a.b]` を `a` テーブル内の `b` サブテーブルと解釈する動作）は
            // 対象外）。角括弧の再出現のみ拒否する。
            if name.is_empty() || name.contains(['[', ']']) {
                return Err(GuardrailError::InvalidInput(format!(
                    "line {line_number}: unsupported table name '{name}'"
                )));
            }
            current_table = name.to_string();
            doc.entry(current_table.clone()).or_default();
            continue;
        }

        let (key, value_str) = line.split_once('=').ok_or_else(|| {
            GuardrailError::InvalidInput(format!("line {line_number}: expected 'key = value'"))
        })?;
        let key = key.trim();
        let value_str = value_str.trim();
        if key.is_empty() {
            return Err(GuardrailError::InvalidInput(format!(
                "line {line_number}: empty key"
            )));
        }

        let value = parse_value(value_str).ok_or_else(|| {
            GuardrailError::InvalidInput(format!(
                "line {line_number}: unsupported value '{value_str}'"
            ))
        })?;

        let table = doc.entry(current_table.clone()).or_default();
        if table.contains_key(key) {
            return Err(GuardrailError::InvalidInput(format!(
                "line {line_number}: duplicate key '{key}' in table '[{current_table}]'"
            )));
        }
        table.insert(key.to_string(), value);
    }

    Ok(doc)
}

/// `#` 以降をコメントとして取り除く。文字列リテラル内の `#` は考慮しない
/// （`guardrail.toml` の値に `#` を含む文字列は現状の用途で不要なため）。
fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(idx) => &line[..idx],
        None => line,
    }
}

fn parse_value(s: &str) -> Option<TomlValue> {
    if let Some(inner) = s.strip_prefix('"').and_then(|r| r.strip_suffix('"')) {
        return Some(TomlValue::String(inner.to_string()));
    }
    match s {
        "true" => return Some(TomlValue::Bool(true)),
        "false" => return Some(TomlValue::Bool(false)),
        _ => {}
    }
    if let Ok(i) = s.parse::<i64>() {
        return Some(TomlValue::Integer(i));
    }
    if let Ok(f) = s.parse::<f64>() {
        return Some(TomlValue::Float(f));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_root_and_table_keys() {
        let doc = parse(
            "root_key = 1\n\n[preset.default]\nlines_max = 200\nbench_max_pct = 5.0\nname = \"default\"\nenabled = true\n",
        )
        .unwrap();
        assert_eq!(doc[""]["root_key"], TomlValue::Integer(1));
        assert_eq!(doc["preset.default"]["lines_max"], TomlValue::Integer(200));
        assert_eq!(
            doc["preset.default"]["bench_max_pct"],
            TomlValue::Float(5.0)
        );
        assert_eq!(
            doc["preset.default"]["name"],
            TomlValue::String("default".to_string())
        );
        assert_eq!(doc["preset.default"]["enabled"], TomlValue::Bool(true));
    }

    #[test]
    fn rejects_duplicate_key_in_same_table() {
        let err = parse("a = 1\na = 2\n").unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_malformed_table_header() {
        let err = parse("[unterminated\n").unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_oversized_input() {
        let huge = "a = 1\n".repeat(MAX_INPUT_BYTES);
        let err = parse(&huge).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn ignores_comments_and_blank_lines() {
        let doc = parse("# comment\n\nkey = 1 # trailing comment\n").unwrap();
        assert_eq!(doc[""]["key"], TomlValue::Integer(1));
    }
}
