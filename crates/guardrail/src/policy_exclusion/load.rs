//! `policy-exclusion.toml`（REQ-5・TASK-5.1a・#119 が定義するリポジトリ直下の
//! ルール定義ファイル）専用のミニ TOML ローダ（TASK-5.2c・#124）。
//!
//! # なぜ `crate::toml_lite` を再利用しないか（#124 判断理由。計画 3.1 節）
//! `policy-exclusion.toml` は `[[exclusion]]`（array-of-tables）で複数ルールを
//! 列挙し、各要素内に `[exclusion.match]` サブテーブルをネストする。
//! `toml_lite`（`config.rs`／`eval::dataset` 用）は「テーブル名 → key への
//! フラットな 1 対 1 マップ」を前提とするため、同名テーブルヘッダが複数回
//! 現れると後勝ちで上書きされ、`[[exclusion]]` を複数件保持できない
//! （`toml_lite::parse` は `[exclusion]` のような角括弧の再出現も
//! 拒否する）。この構造差はテーブル走査ロジックそのものの作り直しを要する
//! ため、`toml_lite` の関数は再利用せず本ファイルで専用実装する。
//! 一方で「依存追加をしない」「入力サイズ上限で DoS を防ぐ」「未知フィールド・
//! 値域外を fail-closed で拒否する」という**方針**は `toml_lite`
//! （`crate::toml_lite::MAX_INPUT_BYTES`）と共有し、上限定数もそのまま再利用する。
//!
//! # 呼び出し元・責務境界
//! [`load_from_str`] は `policy-exclusion.toml` の内容文字列を受け取り
//! [`super::PolicyExclusionConfig`] を返す。呼び出し元（`main.rs`・
//! `self-repair` からの lib 直接呼び出し）がファイル読み込み（I/O）を担い、
//! 本関数はパース・検証のみに専念する（`config.rs`／`toml_lite::parse` と
//! 同じ分担）。
//!
//! 結果は [`super::builtin_defaults`] と完全一致することを
//! `tests/policy_exclusion_toml_consistency.rs`（回帰テスト）で保証する
//! 設計である（`policy-exclusion.toml` ヘッダコメント参照）。乖離は
//! 判定迂回経路になり得るため、両者は同一 PR で同時変更しユーザー承認を
//! 得ること（`.claude/rules/security.md`）。
//!
//! # fail-closed 検証（A03: 外部入力はパース時に検証する）
//! - `schema_version == 1` 必須（それ以外は将来のスキーマ変更を無条件では
//!   受理しない安全側の既定）
//! - 未知テーブル・未知キーを拒否する（`[[exclusion]]`・`[exclusion.match]`
//!   以外のテーブル、上記 2 テーブルが持たないキーはいずれもエラー）
//! - `paths`／`assertion_patterns` の空配列・空文字列要素を拒否する
//!   （「全パスに一致しない」「全行に一致しない」という無意味な除外ルールを
//!   黙って受理しない）
//! - `category`／`action`／`match.type` は既知の値のみを受理する（未知の値を
//!   受理すると `super::Category`／`super::Action`／`super::MatchRule` への
//!   写像先が定まらず、後続の `match` 網羅性が崩れる）
//! - 入力サイズ上限（`crate::toml_lite::MAX_INPUT_BYTES` = 64 KiB）超過を拒否する

use super::path_match::PathPattern;
use super::{Action, Category, ExclusionRule, MatchRule, PolicyExclusionConfig};
use crate::error::GuardrailError;
use crate::toml_lite::MAX_INPUT_BYTES;

/// `[[exclusion]]` 1 要素分のパース中間表現。全フィールドが揃って初めて
/// [`ExclusionRule`] へ変換できる（[`finalize_pending`] 参照）。
#[derive(Default)]
struct PendingExclusion {
    id: Option<String>,
    category: Option<String>,
    /// `description`／`rationale` は [`ExclusionRule`] が保持しないフィールド
    /// だが、fail-closed 検証（必須キー・非空文字列）の対象として読み取る
    /// （`policy-exclusion.toml` のスキーマが要求する形を守っていることの
    /// 確認。値そのものは破棄する）。
    description: Option<String>,
    rationale: Option<String>,
    paths: Option<Vec<String>>,
    action: Option<String>,
    in_match_table: bool,
    match_type: Option<String>,
    assertion_patterns: Option<Vec<String>>,
}

/// `load_from_str` の走査状態。`root`（`schema_version` のみを許す最上位）と
/// `exclusion`（`[[exclusion]]` 〜 次のヘッダ直前までの領域。`[exclusion.match]`
/// も同一要素内として扱う）の 2 状態のみを持つ。
enum State {
    Root,
    Exclusion(PendingExclusion),
}

/// `policy-exclusion.toml` の内容文字列をパース・検証し
/// [`PolicyExclusionConfig`] を返す。
///
/// パース失敗・スキーマ検証失敗はいずれも [`GuardrailError::InvalidInput`]
/// として返す（`config.rs`・`toml_lite::parse` と同じエラー区分。終了コード
/// 契約は `main.rs` 側の写像に委ねる）。
pub fn load_from_str(input: &str) -> Result<PolicyExclusionConfig, GuardrailError> {
    if input.len() > MAX_INPUT_BYTES {
        return Err(GuardrailError::InvalidInput(format!(
            "policy-exclusion.toml exceeds {MAX_INPUT_BYTES} byte limit ({} bytes)",
            input.len()
        )));
    }

    let mut schema_version: Option<i64> = None;
    let mut rules = Vec::new();
    let mut state = State::Root;

    for (lineno, raw_line) in input.lines().enumerate() {
        let line_number = lineno + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if line == "[[exclusion]]" {
            if let State::Exclusion(pending) = state {
                rules.push(finalize_pending(pending, line_number)?);
            }
            state = State::Exclusion(PendingExclusion::default());
            continue;
        }

        if line == "[exclusion.match]" {
            match &mut state {
                State::Exclusion(pending) => {
                    pending.in_match_table = true;
                }
                State::Root => {
                    return Err(GuardrailError::InvalidInput(format!(
                        "line {line_number}: '[exclusion.match]' appeared before any '[[exclusion]]'"
                    )));
                }
            }
            continue;
        }

        if line.starts_with('[') {
            return Err(GuardrailError::InvalidInput(format!(
                "line {line_number}: unsupported table header '{line}'"
            )));
        }

        let (key, value_str) = line.split_once('=').ok_or_else(|| {
            GuardrailError::InvalidInput(format!("line {line_number}: expected 'key = value'"))
        })?;
        let key = key.trim();
        let value_str = value_str.trim();

        match &mut state {
            State::Root => {
                if key != "schema_version" {
                    return Err(GuardrailError::InvalidInput(format!(
                        "line {line_number}: unknown root key '{key}'"
                    )));
                }
                if schema_version.is_some() {
                    return Err(GuardrailError::InvalidInput(format!(
                        "line {line_number}: duplicate 'schema_version'"
                    )));
                }
                schema_version = Some(parse_integer(value_str, line_number)?);
            }
            State::Exclusion(pending) => {
                assign_exclusion_field(pending, key, value_str, line_number)?;
            }
        }
    }

    if let State::Exclusion(pending) = state {
        rules.push(finalize_pending(pending, input.lines().count())?);
    }

    let schema_version = schema_version.ok_or_else(|| {
        GuardrailError::InvalidInput("missing required root key 'schema_version'".to_string())
    })?;
    if schema_version != 1 {
        return Err(GuardrailError::InvalidInput(format!(
            "unsupported schema_version {schema_version} (expected 1)"
        )));
    }
    if rules.is_empty() {
        return Err(GuardrailError::InvalidInput(
            "policy-exclusion.toml defines no [[exclusion]] rules".to_string(),
        ));
    }
    ensure_required_category_coverage(&rules)?;

    Ok(PolicyExclusionConfig { rules })
}

/// `architecture_change`・`test_tolerance_loosening`・`dependency_change` の
/// 3 カテゴリがそれぞれ最低 1 件のルールを持つことを検査する（Bugbot 指摘
/// #330。無条件エスカレーション対象〈G5・依存変更ルール等〉が設定漏れの
/// まま自動適用へ進む経路を fail-closed で塞ぐ。`super::builtin_defaults`
/// との一致は別途 `tests/policy_exclusion_toml_consistency.rs` が保証する）。
fn ensure_required_category_coverage(rules: &[ExclusionRule]) -> Result<(), GuardrailError> {
    const REQUIRED: [(Category, &str); 3] = [
        (Category::ArchitectureChange, "architecture_change"),
        (Category::TestToleranceLoosening, "test_tolerance_loosening"),
        (Category::DependencyChange, "dependency_change"),
    ];
    for (category, label) in REQUIRED {
        if !rules.iter().any(|rule| rule.category == category) {
            return Err(GuardrailError::InvalidInput(format!(
                "policy-exclusion.toml defines no [[exclusion]] rule for required category '{label}'"
            )));
        }
    }
    Ok(())
}

/// `[[exclusion]]`／`[exclusion.match]` 領域内の 1 行（`key = value`）を
/// `pending` へ書き込む。`in_match_table` フラグで両テーブルのキー空間を
/// 分離し、`[exclusion.match]` 側のキー（`type`／`assertion_patterns`）が
/// 誤って `[[exclusion]]` 直下のキーとして扱われる取り違えを防ぐ。
fn assign_exclusion_field(
    pending: &mut PendingExclusion,
    key: &str,
    value_str: &str,
    line_number: usize,
) -> Result<(), GuardrailError> {
    if pending.in_match_table {
        return match key {
            "type" => {
                reject_duplicate_key(pending.match_type.is_some(), key, line_number)?;
                pending.match_type = Some(parse_string(value_str, line_number)?);
                Ok(())
            }
            "assertion_patterns" => {
                reject_duplicate_key(pending.assertion_patterns.is_some(), key, line_number)?;
                pending.assertion_patterns = Some(parse_string_array(
                    value_str,
                    line_number,
                    /* allow_empty */ false,
                )?);
                Ok(())
            }
            other => Err(GuardrailError::InvalidInput(format!(
                "line {line_number}: unknown key '{other}' in '[exclusion.match]'"
            ))),
        };
    }
    match key {
        "id" => {
            reject_duplicate_key(pending.id.is_some(), key, line_number)?;
            pending.id = Some(parse_string(value_str, line_number)?);
        }
        "category" => {
            reject_duplicate_key(pending.category.is_some(), key, line_number)?;
            pending.category = Some(parse_string(value_str, line_number)?);
        }
        "description" => {
            reject_duplicate_key(pending.description.is_some(), key, line_number)?;
            pending.description = Some(parse_string(value_str, line_number)?);
        }
        "rationale" => {
            reject_duplicate_key(pending.rationale.is_some(), key, line_number)?;
            pending.rationale = Some(parse_string(value_str, line_number)?);
        }
        "paths" => {
            reject_duplicate_key(pending.paths.is_some(), key, line_number)?;
            pending.paths = Some(parse_string_array(
                value_str,
                line_number,
                /* allow_empty */ false,
            )?)
        }
        "action" => {
            reject_duplicate_key(pending.action.is_some(), key, line_number)?;
            pending.action = Some(parse_string(value_str, line_number)?);
        }
        other => {
            return Err(GuardrailError::InvalidInput(format!(
                "line {line_number}: unknown key '{other}' in '[[exclusion]]'"
            )));
        }
    }
    Ok(())
}

/// 同一キーの重複指定を fail-closed で拒否する（Bugbot 指摘 #330・`toml_lite`
/// と同じ「重複キー拒否」方針を `assign_exclusion_field` にも揃える）。
/// `already_set` は該当フィールドの `Option` が既に `Some` かどうかを渡す。
fn reject_duplicate_key(
    already_set: bool,
    key: &str,
    line_number: usize,
) -> Result<(), GuardrailError> {
    if already_set {
        return Err(GuardrailError::InvalidInput(format!(
            "line {line_number}: duplicate key '{key}'"
        )));
    }
    Ok(())
}

/// パース完了した `pending` を検証しつつ [`ExclusionRule`] へ変換する。
/// 必須キー欠落・未知の列挙値・空配列はすべてここで fail-closed に拒否する。
fn finalize_pending(
    pending: PendingExclusion,
    line_number: usize,
) -> Result<ExclusionRule, GuardrailError> {
    let id = pending.id.ok_or_else(|| missing_field("id", line_number))?;
    let category_raw = pending
        .category
        .ok_or_else(|| missing_field("category", line_number))?;
    pending
        .description
        .as_ref()
        .ok_or_else(|| missing_field("description", line_number))?;
    pending
        .rationale
        .as_ref()
        .ok_or_else(|| missing_field("rationale", line_number))?;
    let paths_raw = pending
        .paths
        .ok_or_else(|| missing_field("paths", line_number))?;
    let action_raw = pending
        .action
        .ok_or_else(|| missing_field("action", line_number))?;
    let match_type = pending
        .match_type
        .ok_or_else(|| missing_field("match.type", line_number))?;

    let category = match category_raw.as_str() {
        "architecture_change" => Category::ArchitectureChange,
        "test_tolerance_loosening" => Category::TestToleranceLoosening,
        "dependency_change" => Category::DependencyChange,
        other => {
            return Err(GuardrailError::InvalidInput(format!(
                "rule '{id}': unknown category '{other}'"
            )));
        }
    };
    let action = match action_raw.as_str() {
        "human_approval" => Action::HumanApproval,
        other => {
            return Err(GuardrailError::InvalidInput(format!(
                "rule '{id}': unknown action '{other}'"
            )));
        }
    };

    let match_rule = match match_type.as_str() {
        "any_diff_in_paths" => {
            if pending.assertion_patterns.is_some() {
                return Err(GuardrailError::InvalidInput(format!(
                    "rule '{id}': 'assertion_patterns' is not valid for match type 'any_diff_in_paths'"
                )));
            }
            let paths = paths_raw
                .iter()
                .map(|p| PathPattern::compile(p))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    GuardrailError::InvalidInput(format!("rule '{id}': invalid path pattern: {e}"))
                })?;
            MatchRule::AnyDiffInPaths { paths }
        }
        "test_assertion_relaxation_without_prod_change" => {
            let assertion_patterns = pending
                .assertion_patterns
                .ok_or_else(|| missing_field("match.assertion_patterns", line_number))?;
            MatchRule::TestAssertionRelaxationWithoutProdChange { assertion_patterns }
        }
        other => {
            return Err(GuardrailError::InvalidInput(format!(
                "rule '{id}': unknown match type '{other}'"
            )));
        }
    };

    Ok(ExclusionRule {
        id,
        category,
        action,
        match_rule,
    })
}

fn missing_field(field: &str, line_number: usize) -> GuardrailError {
    GuardrailError::InvalidInput(format!(
        "line {line_number}: missing required field '{field}' in '[[exclusion]]'"
    ))
}

/// 二重引用符文字列 1 個のみを受理する（`toml_lite::parse_value` の文字列
/// 分岐と同じ制約）。
fn parse_string(value_str: &str, line_number: usize) -> Result<String, GuardrailError> {
    value_str
        .strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .filter(|inner| !inner.contains('"'))
        .map(str::to_string)
        .ok_or_else(|| {
            GuardrailError::InvalidInput(format!(
                "line {line_number}: expected double-quoted string, got '{value_str}'"
            ))
        })
}

/// `["a", "b"]` の単一行文字列配列を受理する。`allow_empty` が `false` の
/// 場合、空配列・空文字列要素のいずれも拒否する（「全てに一致／全てに不一致」
/// という無意味な除外ルールを黙って受理しない。ローダ冒頭コメント参照）。
fn parse_string_array(
    value_str: &str,
    line_number: usize,
    allow_empty: bool,
) -> Result<Vec<String>, GuardrailError> {
    let inner = value_str
        .strip_prefix('[')
        .and_then(|r| r.strip_suffix(']'))
        .ok_or_else(|| {
            GuardrailError::InvalidInput(format!(
                "line {line_number}: expected array literal, got '{value_str}'"
            ))
        })?;
    let trimmed = inner.trim();
    if trimmed.is_empty() {
        if allow_empty {
            return Ok(Vec::new());
        }
        return Err(GuardrailError::InvalidInput(format!(
            "line {line_number}: empty array is not allowed here"
        )));
    }
    let items = trimmed
        .split(',')
        .map(|part| parse_string(part.trim(), line_number))
        .collect::<Result<Vec<_>, _>>()?;
    if items.iter().any(|s| s.is_empty()) {
        return Err(GuardrailError::InvalidInput(format!(
            "line {line_number}: empty string element is not allowed"
        )));
    }
    Ok(items)
}

fn parse_integer(value_str: &str, line_number: usize) -> Result<i64, GuardrailError> {
    value_str.parse::<i64>().map_err(|_| {
        GuardrailError::InvalidInput(format!(
            "line {line_number}: expected integer, got '{value_str}'"
        ))
    })
}

/// `#` 以降をコメントとして取り除く（二重引用符文字列内はコメント開始と
/// みなさないトグル方式。`toml_lite::strip_comment` と同一契約だが、
/// クレート横断の依存を避けるため本モジュールで独立実装する）。
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

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_valid_toml() -> String {
        r#"
schema_version = 1

[[exclusion]]
id = "arch-hyperparameter-change"
category = "architecture_change"
description = "d"
rationale = "r"
paths = ["**/src/model*.rs"]
action = "human_approval"

[exclusion.match]
type = "any_diff_in_paths"
"#
        .to_string()
    }

    /// `ensure_required_category_coverage` を満たすための埋め草ルールを
    /// 生成する（成功系テストが `minimal_valid_toml` 等 1 カテゴリのみでは
    /// 網羅チェックに落ちるため、テスト対象外の 2 カテゴリを補う）。
    fn filler_rule(id: &str, category: &str) -> String {
        format!(
            "\n[[exclusion]]\nid = \"{id}\"\ncategory = \"{category}\"\ndescription = \"d\"\nrationale = \"r\"\npaths = [\"**/filler-{id}\"]\naction = \"human_approval\"\n\n[exclusion.match]\ntype = \"any_diff_in_paths\"\n"
        )
    }

    /// `architecture_change`・`test_tolerance_loosening`・`dependency_change`
    /// のうち `except_category` 以外を埋め草ルールとして付加した文字列を返す。
    fn with_other_categories_filled(toml: &str, except_category: &str) -> String {
        let mut out = toml.to_string();
        for (id, category) in [
            ("filler-arch", "architecture_change"),
            ("filler-tol", "test_tolerance_loosening"),
            ("filler-dep", "dependency_change"),
        ] {
            if category != except_category {
                out.push_str(&filler_rule(id, category));
            }
        }
        out
    }

    #[test]
    fn parses_any_diff_in_paths_rule() {
        let toml = with_other_categories_filled(&minimal_valid_toml(), "architecture_change");
        let config = load_from_str(&toml).unwrap();
        assert_eq!(config.rules.len(), 3);
        let rule = &config.rules[0];
        assert_eq!(rule.id, "arch-hyperparameter-change");
        assert_eq!(rule.category, Category::ArchitectureChange);
        assert_eq!(rule.action, Action::HumanApproval);
        match &rule.match_rule {
            MatchRule::AnyDiffInPaths { paths } => {
                assert_eq!(paths.len(), 1);
                assert_eq!(paths[0].as_str(), "**/src/model*.rs");
            }
            other => panic!("expected AnyDiffInPaths, got {other:?}"),
        }
    }

    #[test]
    fn parses_test_assertion_relaxation_rule_with_patterns() {
        let base = r#"
schema_version = 1

[[exclusion]]
id = "test-tolerance-loosening"
category = "test_tolerance_loosening"
description = "d"
rationale = "r"
paths = ["**/*.rs"]
action = "human_approval"

[exclusion.match]
type = "test_assertion_relaxation_without_prod_change"
assertion_patterns = ["assert!", "abs() <", "1e-[0-9]"]
"#;
        let toml = with_other_categories_filled(base, "test_tolerance_loosening");
        let config = load_from_str(&toml).unwrap();
        assert_eq!(config.rules.len(), 3);
        match &config.rules[0].match_rule {
            MatchRule::TestAssertionRelaxationWithoutProdChange { assertion_patterns } => {
                assert_eq!(
                    assertion_patterns,
                    &vec![
                        "assert!".to_string(),
                        "abs() <".to_string(),
                        "1e-[0-9]".to_string()
                    ]
                );
            }
            other => panic!("expected TestAssertionRelaxationWithoutProdChange, got {other:?}"),
        }
    }

    #[test]
    fn parses_multiple_exclusion_entries() {
        let toml = format!(
            "{}\n[[exclusion]]\nid = \"dependency-change\"\ncategory = \"dependency_change\"\ndescription = \"d\"\nrationale = \"r\"\npaths = [\"**/Cargo.toml\", \"**/Cargo.lock\"]\naction = \"human_approval\"\n\n[exclusion.match]\ntype = \"any_diff_in_paths\"\n{}",
            minimal_valid_toml(),
            filler_rule("filler-tol", "test_tolerance_loosening")
        );
        let config = load_from_str(&toml).unwrap();
        assert_eq!(config.rules.len(), 3);
        assert_eq!(config.rules[1].id, "dependency-change");
        assert_eq!(config.rules[1].category, Category::DependencyChange);
    }

    #[test]
    fn rejects_missing_required_category_coverage() {
        // `dependency_change` カテゴリのルールが 1 件も存在しない場合、
        // ルール自体は妥当でも `ensure_required_category_coverage` で
        // fail-closed に拒否される（Bugbot 指摘 #330）。
        let toml = with_other_categories_filled(&minimal_valid_toml(), "architecture_change")
            .replace(
                "category = \"dependency_change\"",
                "category = \"architecture_change\"",
            );
        let err = load_from_str(&toml).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_duplicate_key_in_exclusion_table() {
        // フィクスチャに `with_other_categories_filled` で必須カテゴリ
        // 3 種を揃える（Bugbot 指摘 #330・PR #330 レビュー再指摘）。
        // 揃えないと `reject_duplicate_key` が機能しなくなり duplicate な
        // `id` が後勝ちで静かに上書きされても、後続の
        // `ensure_required_category_coverage` が別の理由（カテゴリ欠落）で
        // 先に `InvalidInput` を返してしまい、このテストは
        // `reject_duplicate_key` の破壊を検出できないまま green で
        // 居座ってしまう。フィクスチャで網羅チェックを通過させたうえで、
        // エラーメッセージが `reject_duplicate_key`
        // （`load.rs:269` 付近）由来の "duplicate key" であることまで
        // 直接検証し、`ensure_required_category_coverage` 由来のエラーとの
        // 取り違えを防ぐ。
        let base = r#"
schema_version = 1

[[exclusion]]
id = "x"
id = "y"
category = "architecture_change"
description = "d"
rationale = "r"
paths = ["**/*.rs"]
action = "human_approval"

[exclusion.match]
type = "any_diff_in_paths"
"#;
        let toml = with_other_categories_filled(base, "architecture_change");
        let err = load_from_str(&toml).unwrap_err();
        match err {
            GuardrailError::InvalidInput(message) => {
                assert!(
                    message.contains("duplicate key 'id'"),
                    "expected duplicate-key rejection message, got: {message}"
                );
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn rejects_duplicate_key_in_match_table() {
        // 上記 `rejects_duplicate_key_in_exclusion_table` と同じ理由で
        // 必須カテゴリを埋め草で揃え、`reject_duplicate_key` 由来の
        // エラーメッセージであることまで検証する（Bugbot 指摘 #330）。
        let base = r#"
schema_version = 1

[[exclusion]]
id = "x"
category = "architecture_change"
description = "d"
rationale = "r"
paths = ["**/*.rs"]
action = "human_approval"

[exclusion.match]
type = "any_diff_in_paths"
type = "any_diff_in_paths"
"#;
        let toml = with_other_categories_filled(base, "architecture_change");
        let err = load_from_str(&toml).unwrap_err();
        match err {
            GuardrailError::InvalidInput(message) => {
                assert!(
                    message.contains("duplicate key 'type'"),
                    "expected duplicate-key rejection message, got: {message}"
                );
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn rejects_missing_schema_version() {
        let toml = "[[exclusion]]\nid = \"x\"\n";
        let err = load_from_str(toml).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_unsupported_schema_version() {
        let toml = "schema_version = 2\n";
        let err = load_from_str(toml).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_unknown_key_in_exclusion_table() {
        let toml = r#"
schema_version = 1

[[exclusion]]
id = "x"
category = "architecture_change"
description = "d"
rationale = "r"
paths = ["**/*.rs"]
action = "human_approval"
unknown_key = "z"

[exclusion.match]
type = "any_diff_in_paths"
"#;
        let err = load_from_str(toml).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_unknown_category() {
        let toml = r#"
schema_version = 1

[[exclusion]]
id = "x"
category = "not_a_real_category"
description = "d"
rationale = "r"
paths = ["**/*.rs"]
action = "human_approval"

[exclusion.match]
type = "any_diff_in_paths"
"#;
        let err = load_from_str(toml).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_unknown_action() {
        let toml = r#"
schema_version = 1

[[exclusion]]
id = "x"
category = "architecture_change"
description = "d"
rationale = "r"
paths = ["**/*.rs"]
action = "auto_apply"

[exclusion.match]
type = "any_diff_in_paths"
"#;
        let err = load_from_str(toml).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_unknown_match_type() {
        let toml = r#"
schema_version = 1

[[exclusion]]
id = "x"
category = "architecture_change"
description = "d"
rationale = "r"
paths = ["**/*.rs"]
action = "human_approval"

[exclusion.match]
type = "not_a_real_match_type"
"#;
        let err = load_from_str(toml).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_empty_paths_array() {
        let toml = r#"
schema_version = 1

[[exclusion]]
id = "x"
category = "architecture_change"
description = "d"
rationale = "r"
paths = []
action = "human_approval"

[exclusion.match]
type = "any_diff_in_paths"
"#;
        let err = load_from_str(toml).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_empty_assertion_patterns_array() {
        let toml = r#"
schema_version = 1

[[exclusion]]
id = "x"
category = "test_tolerance_loosening"
description = "d"
rationale = "r"
paths = ["**/*.rs"]
action = "human_approval"

[exclusion.match]
type = "test_assertion_relaxation_without_prod_change"
assertion_patterns = []
"#;
        let err = load_from_str(toml).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_assertion_patterns_on_any_diff_in_paths() {
        let toml = r#"
schema_version = 1

[[exclusion]]
id = "x"
category = "architecture_change"
description = "d"
rationale = "r"
paths = ["**/*.rs"]
action = "human_approval"

[exclusion.match]
type = "any_diff_in_paths"
assertion_patterns = ["assert!"]
"#;
        let err = load_from_str(toml).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_missing_required_field() {
        let toml = r#"
schema_version = 1

[[exclusion]]
id = "x"
category = "architecture_change"
description = "d"
rationale = "r"
action = "human_approval"

[exclusion.match]
type = "any_diff_in_paths"
"#;
        let err = load_from_str(toml).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_no_exclusion_rules() {
        let toml = "schema_version = 1\n";
        let err = load_from_str(toml).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_input_exceeding_size_limit() {
        let huge = "#".repeat(MAX_INPUT_BYTES + 1);
        let err = load_from_str(&huge).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }
}
