//! TASK-5.3a（イシュー #126）受け入れ条件の機械検証: `tests/fixtures/labeled-changes/`
//! の 2 層ラベルモデルが持つ不変条件 (1)「空 → 一致」の回帰テスト。
//!
//! README（`crates/guardrail/tests/fixtures/labeled-changes/README.md`
//! 「2 層ラベルモデル」節）が定義する不変条件は次の 2 本:
//! - (1) 空 → 一致: `expected_exclusion_rule_ids` が空なら
//!   `expected_verdict_after_exclusions == expected_verdict`（本ファイルの
//!   スコープ）
//! - (2) 非空 → 安全側単調性: `severity(after) >= max(severity(expected),
//!   Escalate)`（兄弟イシュー #127 のスコープ。本ファイルでは検証しない）
//!
//! 不変条件 (1) は「除外ルールに一切 match しない変更では、除外リスト適用が
//! 判定結果を変えてはならない」ことをラベル上で保証する回帰テストであり、
//! TASK-6.1（判定器自己回帰 CI・#146）がこのテストスイートを前提とする。
//!
//! # スコープ境界（`labeled_changes_labels.rs`・TASK-5.3 系との重複防止）
//! - `labeled_changes_labels.rs`（TASK-4.2b・#110）は 2 層ラベルの**形式**
//!   （スキーマ・enum 値・README 記載値とのピン留め）のみを検証し、モジュール
//!   コメントで不変条件の検証を明示的に対象外としている。本ファイルはその
//!   欠落を埋める（不変条件 (1) の**意味**検証）。
//! - 不変条件 (2)（非空 → 安全側単調性）は #127 のスコープ。
//! - `expected_exclusion_rule_ids` の各 id と `policy-exclusion.toml`
//!   （TASK-5.1a・#312 で移植済み）との参照整合性テストは TASK-5.3 系の
//!   残スコープであり、本ファイルでは検証しない（README「本イシューの
//!   スコープ外」）。
//! - 除外リスト match 実装（TASK-5.2）との結合検証（本番経路での判定一致）は
//!   TASK-6.1（#146）のスコープ。
//!
//! # 依存（追加なし）
//! `meta.toml` は `guardrail::toml_lite` が配列
//! （`expected_exclusion_rule_ids`）非対応のため、本ファイル内の std-only
//! ミニパーサでパースする。`labeled_changes_labels.rs` と同一の設計判断
//! だが、兄弟イシュー #127 との並行編集を避けるため（`.claude/rules/
//! delegation-impl.md`「複数 Agent に同一ファイルを並行編集させない」）
//! 共有ヘルパーへは切り出さず、本ファイル内で完結させる（意図的な重複）。
//! `Cargo.toml`／`Cargo.lock` は本ファイルの追加によって変更しない。
//!
//! # セキュリティ（A03 インジェクション対策。`.claude/rules/security.md`）
//! `meta.toml` はリポジトリ内データだが外部フォーマットパースとして扱い、
//! fail-closed で検証する: change_id をディレクトリ名から path join する
//! **前**に文字クラス検証してパストラバーサルを遮断
//! （[`is_valid_change_id`]。`labeled_changes_labels.rs` と同一契約）、
//! 64 KiB サイズ上限（[`MAX_META_BYTES`]）で DoS 的入力を拒否、verdict 値は
//! 許可リスト完全一致で照合する。サブプロセス起動・シェル文字列展開は
//! 行わない。
//!
//! # 整合性（A08。`.claude/rules/security.md`）
//! 本テストは 2 層ラベルの改竄検知（見逃し方向の無断改定をテスト差分なしに
//! 通せなくする）として機能する。fixture のラベル値・ガードレール閾値・
//! テスト許容誤差は一切変更しない（変更はユーザー承認必須）。パース失敗・
//! スキーマ逸脱は fail-closed（panic）で扱い、fixture 消失による**空虚な
//! 通過**は件数ガード（[`empty_exclusion_ids_imply_identical_verdicts`]
//! 内の 15 件アサーション・[`at_least_one_fixture_has_empty_exclusion_ids`]）
//! で防ぐ。

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// `meta.toml` の外部入力サイズ上限（DoS 的な巨大入力を拒否する。
/// `guardrail::toml_lite::MAX_INPUT_BYTES`・`labeled_changes_labels.rs::
/// MAX_META_BYTES` と同一値）。
const MAX_META_BYTES: usize = 64 * 1024;

/// README「15 件一覧」表に記載された fixture の総数。fixture 消失を検知する
/// ための件数ガードに用いる（空リストになって不変条件テストが空虚に通過する
/// ことを防ぐ）。
const EXPECTED_FIXTURE_COUNT: usize = 15;

/// verdict 系フィールドの許可値（`decision::Verdict` の 3 分岐と同一語彙。
/// `labeled_changes_labels.rs::VERDICT_VALUES` と同一契約）。
const VERDICT_VALUES: &[&str] = &["auto-apply", "escalate", "reject"];

/// `crates/guardrail/tests/fixtures/labeled-changes` への絶対パス。
fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/labeled-changes")
}

/// change_id（`changes/` 配下のディレクトリ名）の文字クラス契約。
/// `labeled_changes_labels.rs::is_valid_change_id` と同一契約（英数字始まり・
/// `[A-Za-z0-9._-]` のみ・64 字以内。パストラバーサル対策）。
fn is_valid_change_id(id: &str) -> bool {
    if id.is_empty() || id.len() > 64 {
        return false;
    }
    let mut chars = id.chars();
    let first = chars.next().expect("空文字列は上で弾いている");
    if !first.is_ascii_alphanumeric() {
        return false;
    }
    id.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
}

/// `changes/` 配下の change_id 一覧を、文字クラス検証を通過したものだけ
/// 列挙する（検証前のディレクトリ名を path join に使わない。A03 対策）。
fn list_change_ids() -> Vec<String> {
    let changes_dir = fixtures_root().join("changes");
    let mut ids: Vec<String> = fs::read_dir(&changes_dir)
        .unwrap_or_else(|e| panic!("changes/ ディレクトリの読み取りに失敗: {changes_dir:?}: {e}"))
        .filter_map(|entry| {
            let entry = entry.expect("read_dir エントリの取得に失敗");
            if entry.file_type().expect("file_type の取得に失敗").is_dir() {
                Some(entry.file_name().to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .collect();
    ids.sort();
    for id in &ids {
        assert!(
            is_valid_change_id(id),
            "change_id '{id}' が文字クラス契約を満たさない（A03: path join 前に遮断）"
        );
    }
    ids
}

/// `meta.toml` 限定サブセットのパース結果 1 値。本ファイルが値として
/// 抽出するのは文字列と文字列配列のみだが、`meta.toml` 自体には
/// `known_blindspot` 等の真偽値フィールドも含まれるため、パーサとしては
/// 真偽値も受理する（`Bool` は読み捨てるだけの目的で保持する。
/// `labeled_changes_labels.rs::RawValue` と同型）。
#[derive(Debug, Clone, PartialEq)]
enum RawValue {
    Str(String),
    Bool(bool),
    Array(Vec<String>),
}

/// `meta.toml` 限定サブセット（フラットな `key = value`＋文字列配列＋
/// `#` 行コメント）のミニパーサ。対応文法・非対応理由は
/// `labeled_changes_labels.rs::parse_flat_toml` と同一（本ファイル冒頭
/// `//!` コメント「依存」節参照）。
fn parse_flat_toml(input: &str) -> Result<BTreeMap<String, RawValue>, String> {
    if input.len() > MAX_META_BYTES {
        return Err(format!(
            "input exceeds {MAX_META_BYTES} byte limit ({} bytes)",
            input.len()
        ));
    }

    let mut map = BTreeMap::new();
    for (lineno, raw_line) in input.lines().enumerate() {
        let line_number = lineno + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let (key, value_str) = trimmed
            .split_once('=')
            .ok_or_else(|| format!("line {line_number}: expected 'key = value'"))?;
        let key = key.trim().to_string();
        let value_str = value_str.trim();
        if key.is_empty() {
            return Err(format!("line {line_number}: empty key"));
        }
        if map.contains_key(&key) {
            return Err(format!("line {line_number}: duplicate key '{key}'"));
        }

        let value = parse_toml_value(value_str)
            .ok_or_else(|| format!("line {line_number}: unsupported value '{value_str}'"))?;
        map.insert(key, value);
    }
    Ok(map)
}

/// 値パーサ。`meta.toml` は `known_blindspot` 等の真偽値フィールドも
/// 含むため、行単位のパース自体は真偽値も受理する（本ファイルが
/// 実際に抽出するのは文字列・文字列配列のみで、真偽値は
/// [`RawValue::Bool`] として読み捨てる）。
fn parse_toml_value(s: &str) -> Option<RawValue> {
    if let Some(inner) = s.strip_prefix('"').and_then(|r| r.strip_suffix('"')) {
        return Some(RawValue::Str(inner.to_string()));
    }
    match s {
        "true" => return Some(RawValue::Bool(true)),
        "false" => return Some(RawValue::Bool(false)),
        _ => {}
    }
    if let Some(inner) = s.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
        let inner = inner.trim();
        if inner.is_empty() {
            return Some(RawValue::Array(Vec::new()));
        }
        let mut items = Vec::new();
        for part in inner.split(',') {
            let item = part
                .trim()
                .strip_prefix('"')
                .and_then(|r| r.strip_suffix('"'))?;
            items.push(item.to_string());
        }
        return Some(RawValue::Array(items));
    }
    None
}

fn expect_str(map: &BTreeMap<String, RawValue>, key: &str, id: &str) -> String {
    match map.get(key) {
        Some(RawValue::Str(s)) => s.clone(),
        other => panic!("change_id '{id}': フィールド '{key}' は文字列である想定だが {other:?}"),
    }
}

fn expect_array(map: &BTreeMap<String, RawValue>, key: &str, id: &str) -> Vec<String> {
    match map.get(key) {
        Some(RawValue::Array(a)) => a.clone(),
        other => {
            panic!("change_id '{id}': フィールド '{key}' は文字列配列である想定だが {other:?}")
        }
    }
}

/// 不変条件 (1) の検証に必要な最小限のラベル値（`meta.toml` の 9 フィールド
/// 中 3 フィールドのみを読む。他フィールドの形式検証は
/// `labeled_changes_labels.rs` の責務であり本ファイルでは重複しない）。
struct EmptyExclusionCheck {
    change_id: String,
    expected_verdict: String,
    expected_exclusion_rule_ids: Vec<String>,
    expected_verdict_after_exclusions: String,
}

/// `meta.toml` を読み取り、不変条件 (1) 検証に必要な 3 フィールドを
/// 抽出する。verdict 値は許可リスト照合で fail-closed に扱う（A03）。
fn load_for_empty_exclusion_check(id: &str) -> EmptyExclusionCheck {
    let path = fixtures_root().join("changes").join(id).join("meta.toml");
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?} の読み取りに失敗: {e}"));
    let raw = parse_flat_toml(&text)
        .unwrap_or_else(|e| panic!("{path:?} のパースに失敗（不正な meta.toml 構文）: {e}"));

    let change_id = expect_str(&raw, "change_id", id);
    assert_eq!(
        change_id, id,
        "change_id フィールド '{change_id}' がディレクトリ名 '{id}' と一致しない"
    );

    let expected_verdict = expect_str(&raw, "expected_verdict", id);
    let expected_verdict_after_exclusions =
        expect_str(&raw, "expected_verdict_after_exclusions", id);
    for (field_name, value) in [
        ("expected_verdict", &expected_verdict),
        (
            "expected_verdict_after_exclusions",
            &expected_verdict_after_exclusions,
        ),
    ] {
        assert!(
            VERDICT_VALUES.contains(&value.as_str()),
            "change_id '{id}': {field_name} '{value}' が許可値 {VERDICT_VALUES:?} に含まれない"
        );
    }

    let expected_exclusion_rule_ids = expect_array(&raw, "expected_exclusion_rule_ids", id);

    EmptyExclusionCheck {
        change_id,
        expected_verdict,
        expected_exclusion_rule_ids,
        expected_verdict_after_exclusions,
    }
}

/// 受け入れ条件の本体: 不変条件 (1)「`expected_exclusion_rule_ids` が空なら
/// `expected_verdict_after_exclusions == expected_verdict`」が全 15 fixture
/// で成立することを検証する。
///
/// 件数ガード（`ids.len() == EXPECTED_FIXTURE_COUNT`）は fixture 消失に
/// よってループ本体が実質空回りし、テストが**空虚に**通過することを防ぐ
/// （A08: 改竄検知の一環）。
#[test]
fn empty_exclusion_ids_imply_identical_verdicts() {
    let ids = list_change_ids();
    assert_eq!(
        ids.len(),
        EXPECTED_FIXTURE_COUNT,
        "changes/ 配下の fixture 総数が README「15 件一覧」と一致しない: {ids:?}"
    );

    for id in &ids {
        let checked = load_for_empty_exclusion_check(id);
        if checked.expected_exclusion_rule_ids.is_empty() {
            assert_eq!(
                checked.expected_verdict_after_exclusions,
                checked.expected_verdict,
                "change_id '{}': 不変条件 (1)（空 → 一致）違反。\
                 expected_exclusion_rule_ids が空なのに \
                 expected_verdict_after_exclusions='{}' が \
                 expected_verdict='{}' と一致しない（README「2 層ラベル\
                 モデル」節）",
                checked.change_id,
                checked.expected_verdict_after_exclusions,
                checked.expected_verdict,
            );
        }
    }
}

/// 非空虚性ガード: `expected_exclusion_rule_ids` が空の fixture が
/// 1 件以上存在することを検証する。全件が非空化されて
/// [`empty_exclusion_ids_imply_identical_verdicts`] の対象が消滅したこと
/// （＝実質的に何も検証しなくなったこと）に気づけるようにする（README
/// 「15 件一覧」表時点で空リストは 10 件）。
#[test]
fn at_least_one_fixture_has_empty_exclusion_ids() {
    let ids = list_change_ids();
    let empty_count = ids
        .iter()
        .filter(|id| {
            load_for_empty_exclusion_check(id)
                .expected_exclusion_rule_ids
                .is_empty()
        })
        .count();
    assert!(
        empty_count >= 1,
        "expected_exclusion_rule_ids が空の fixture が 1 件も存在しない。\
         不変条件 (1) の検証対象が消滅している可能性がある（empty_count=0）"
    );
}
