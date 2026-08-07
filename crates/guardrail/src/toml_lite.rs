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
///
/// `StringArray` は `guardrail::eval::dataset`（TASK-4.3a・イシュー #115）が読む
/// `meta.toml` の `expected_exclusion_rule_ids = [...]` フィールドに対応するため
/// 追加した（本体の `guardrail.toml` パース〈`config.rs`〉はこの型を消費しない。
/// 既存の `guardrail.toml` パース挙動・既存テストは不変）。単一行の文字列配列
/// のみを受理し、ネスト配列・非文字列要素は非対応（`meta.toml` の実際の記法が
/// これらを使わないため）。
#[derive(Debug, Clone, PartialEq)]
pub enum TomlValue {
    Integer(i64),
    Float(f64),
    String(String),
    Bool(bool),
    StringArray(Vec<String>),
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

/// `#` 以降をコメントとして取り除く。二重引用符文字列リテラルの内側（トグル
/// 方式。エスケープされた `\"` は非対応）にある `#` はコメント開始とみなさない。
///
/// `guardrail.toml` は本来 `#` を含む文字列値を使わない想定だったが、
/// `guardrail::eval::dataset`（TASK-4.3a・イシュー #115）が本パーサで
/// `meta.toml` を読むようになり、`origin`（例: `"PoC-2 #1 流用"`）等の
/// フィールドが文字列内に `#` を含むため、文字列トグルを実装する
/// （既存の `guardrail.toml` は文字列内に `#` を含まないため挙動は不変）。
fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    for (idx, ch) in line.char_indices() {
        match ch {
            '"' => in_string = !in_string,
            '#' if !in_string => return &line[..idx],
            _ => {}
        }
    }
    line
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
    if let Some(inner) = s.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
        return parse_string_array(inner.trim());
    }
    None
}

/// `[ "a", "b" ]`／`[]` の単一行文字列配列のみを受理する（`TomlValue` の
/// ドキュメント参照）。要素が二重引用符文字列でない場合は非対応として `None`
/// を返し、呼び出し元 `parse` が行番号付きの `InvalidInput` に変換する。
fn parse_string_array(inner: &str) -> Option<TomlValue> {
    if inner.is_empty() {
        return Some(TomlValue::StringArray(Vec::new()));
    }
    let mut items = Vec::new();
    for part in inner.split(',') {
        let item = part
            .trim()
            .strip_prefix('"')
            .and_then(|r| r.strip_suffix('"'))?;
        items.push(item.to_string());
    }
    Some(TomlValue::StringArray(items))
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

    /// 回帰テスト: `#` を含む文字列値（`meta.toml` の `origin = "PoC-2 #1 流用"`
    /// 等）が、コメント除去により途中で切り詰められないこと（`strip_comment`
    /// の文字列トグル対応。TASK-4.3a・イシュー #115 で発覚した実データ由来の
    /// 回帰）。
    #[test]
    fn hash_inside_string_literal_is_not_treated_as_comment_start() {
        let doc = parse("origin = \"PoC-2 #1 流用\"\n").unwrap();
        assert_eq!(
            doc[""]["origin"],
            TomlValue::String("PoC-2 #1 流用".to_string())
        );
    }

    /// 上記と行末コメントの併用（文字列を閉じた**後**の `#` は引き続き
    /// コメントとして扱われること）。
    #[test]
    fn trailing_comment_after_closed_string_with_hash_is_still_stripped() {
        let doc = parse("origin = \"PoC-2 #1 流用\" # trailing note\n").unwrap();
        assert_eq!(
            doc[""]["origin"],
            TomlValue::String("PoC-2 #1 流用".to_string())
        );
    }

    #[test]
    fn parses_string_array_field() {
        let doc = parse("ids = [\"a\", \"b\"]\n").unwrap();
        assert_eq!(
            doc[""]["ids"],
            TomlValue::StringArray(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn parses_empty_string_array_field() {
        let doc = parse("ids = []\n").unwrap();
        assert_eq!(doc[""]["ids"], TomlValue::StringArray(Vec::new()));
    }

    /// `config.rs` の既存挙動（`guardrail.toml` は整数・浮動小数・文字列・
    /// 真偽値のみ）に対する回帰確認: `StringArray` 追加後も既存文法の解釈は
    /// 変わらないこと。
    #[test]
    fn existing_scalar_value_parsing_is_unaffected_by_array_support() {
        let doc = parse(
            "[preset.default]\nlines_max = 200\nbench_median_max_pct = 5.0\nname = \"default\"\nenabled = true\n",
        )
        .unwrap();
        assert_eq!(doc["preset.default"]["lines_max"], TomlValue::Integer(200));
        assert_eq!(
            doc["preset.default"]["bench_median_max_pct"],
            TomlValue::Float(5.0)
        );
        assert_eq!(
            doc["preset.default"]["name"],
            TomlValue::String("default".to_string())
        );
        assert_eq!(doc["preset.default"]["enabled"], TomlValue::Bool(true));
    }
}
