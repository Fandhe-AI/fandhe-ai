//! TASK-4.2b（イシュー #110）受け入れ条件の機械検証: `tests/fixtures/labeled-changes/`
//! 全 15 件の `meta.toml` 2 層ラベル（`expected_verdict`／
//! `expected_exclusion_rule_ids`／`expected_verdict_after_exclusions`）の
//! **形式検証**と、`poc3-result.json` の**参照値の整理**を行う。
//!
//! TASK-4.2a（#109・`labeled_changes_fixtures.rs`）は `crates/guardrail/src/`・
//! `Cargo.toml` を TASK-4.1（#103）と並行編集しない制約から std-only の
//! 構造検証（ファイル存在・patch 適用可否・行数境界）に留めていた。本ファイルは
//! その後継として、TOML／JSON の**中身**を検証する（README「検証範囲の注記」
//! 「meta.toml の 2 層ラベルの意味検証・`lines_changed` 等参照値の整理は
//! TASK-4.2b（#110）のスコープ」に対応）。
//!
//! # スコープ境界（TASK-5.3・#125 との重複防止）
//! - 本ファイルが検証するのは 2 層ラベルの**形式**（スキーマ・enum 値・
//!   ファイル間整合・README 記載値とのピン留め）のみ。
//! - 空 → 一致・安全側単調性という**不変条件**（`severity(...) >=
//!   max(severity(expected_verdict), Escalate)`）の回帰テストは TASK-5.3
//!   （#125/#126/#127）のスコープであり、本ファイルでは検証しない。
//! - `expected_exclusion_rule_ids` の `policy-exclusion.toml` 参照整合も
//!   TASK-5.3 系のスコープ（README 明記）。
//!
//! # 依存（追加なし）
//! `meta.toml` は `guardrail::toml_lite` が配列（`expected_exclusion_rule_ids`）
//! 非対応のため、本ファイル内の std-only ミニパーサでパースする
//! （[`parse_flat_toml`]）。`poc3-result.json` は `guardrail` の既存
//! `[dependencies]`（`Cargo.toml`）にある `serde_json` をそのまま使う
//! （integration test は package の依存解決を共有するため追加宣言は不要）。
//! `Cargo.toml`／`Cargo.lock` は本ファイルの追加によって変更しない。
//!
//! # セキュリティ（A03 インジェクション対策。`.claude/rules/security.md`）
//! `meta.toml`／`poc3-result.json` はリポジトリ内データだが外部フォーマット
//! パースとして扱い、fail-closed で検証する: 64 KiB サイズ上限
//! （[`MAX_META_BYTES`]／[`MAX_POC3_BYTES`]）・未知フィールド拒否
//! （スキーマのキー集合完全一致）・enum 値の許可リスト照合。change_id は
//! ディレクトリ名を path join する**前に**文字クラス検証し、パス
//! トラバーサルを遮断する（[`is_valid_change_id`]。`labeled_changes_fixtures.rs`
//! の同名関数と同一契約）。サブプロセス起動・シェル文字列展開は行わない。
//!
//! # 整合性（A08。`.claude/rules/security.md`）
//! [`GROUND_TRUTH`] は README「15 件一覧」表の値をテスト内 const としてピン
//! 留めしたものであり、ガードレール正解ラベルの無断変更（緩和を含む）を
//! テスト差分なしには通せなくする改竄検知の役割を持つ。ラベル値・許容誤差は
//! 一切変更しない（PoC-3 実測由来の正本。README「由来と v2 移植方針」参照）。

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// `meta.toml` の外部入力サイズ上限（DoS 的な巨大入力を拒否する。
/// `guardrail::toml_lite::MAX_INPUT_BYTES` と同一値）。
const MAX_META_BYTES: usize = 64 * 1024;

/// `poc3-result.json` の外部入力サイズ上限（同上の理由）。
const MAX_POC3_BYTES: usize = 64 * 1024;

/// 判定順序契約の行数閾値（`labeled_changes_fixtures.rs::LINES_MAX` と同値。
/// テスト対象ファイルが異なるため独立して定義する）。
const LINES_MAX: u64 = 200;

/// `meta.toml` スキーマのキー集合（README「`meta.toml` スキーマ」節）。
/// 未知フィールド・欠落フィールドの双方を拒否するため完全一致で照合する。
const REQUIRED_META_KEYS: &[&str] = &[
    "change_id",
    "category",
    "expected_verdict",
    "poc3_default_verdict",
    "known_blindspot",
    "origin",
    "summary",
    "expected_exclusion_rule_ids",
    "expected_verdict_after_exclusions",
];

/// `category` の許可値（README「ラベル基準」節の 3 分類）。
const CATEGORY_VALUES: &[&str] = &["safe", "dangerous", "gray"];

/// verdict 系フィールド（`expected_verdict`／`poc3_default_verdict`／
/// `expected_verdict_after_exclusions`）の許可値。`decision::Verdict` の
/// 3 分岐（`auto-apply`/`escalate`/`reject`）と同一語彙。
const VERDICT_VALUES: &[&str] = &["auto-apply", "escalate", "reject"];

/// `poc3-result.json` スキーマのキー集合（README「重要な注記」節・
/// PoC-3 実測の生データが持つ全フィールド）。
const REQUIRED_POC3_KEYS: &[&str] = &[
    "change_id",
    "preset",
    "lines_changed",
    "lines_max",
    "api_broken",
    "gaming_suspect",
    "build_ok",
    "test_ok",
    "clippy_ok",
    "bench_ran",
    "bench_median_pct",
    "bench_samples_pct",
    "bench_max_pct",
    "verdict",
    "reasons",
];

/// `crates/guardrail/tests/fixtures/labeled-changes` への絶対パス。
/// `labeled_changes_fixtures.rs::fixtures_root` と同一定義（独立した
/// テストバイナリのため重複定義するが、参照先ディレクトリは共有する）。
fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/labeled-changes")
}

/// change_id（`changes/` 配下のディレクトリ名）の文字クラス契約。
/// `labeled_changes_fixtures.rs::is_valid_change_id` と同一契約
/// （英数字始まり・`[A-Za-z0-9._-]` のみ・64 字以内。パストラバーサル対策）。
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

/// `meta.toml`／`poc3-result.json` の限定サブセットパース結果 1 値。
/// `guardrail::toml_lite::TomlValue` と異なり数値型は扱わない
/// （`meta.toml` の全フィールドが文字列・真偽値・文字列配列のいずれかのため）。
#[derive(Debug, Clone, PartialEq)]
enum RawValue {
    Str(String),
    Bool(bool),
    Array(Vec<String>),
}

/// `meta.toml` 限定サブセット（フラットな `key = value`＋文字列配列＋
/// `#` 行コメント）のミニパーサ。`guardrail::toml_lite` を使わない理由は
/// 本ファイル冒頭 `//!` コメント「依存」節を参照。
///
/// 対応文法: 空行・`#` で始まる行コメントの無視、`key = "string"`、
/// `key = true`/`false`、`key = ["a", "b"]`（空配列 `[]` 可）。ネスト
/// テーブル・複数行文字列・行末コメントは非対応（`meta.toml` の実際の
/// 記法がこれらを使わないため。実装時に全 15 ファイルで確認済み）。
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

fn expect_bool(map: &BTreeMap<String, RawValue>, key: &str, id: &str) -> bool {
    match map.get(key) {
        Some(RawValue::Bool(b)) => *b,
        other => panic!("change_id '{id}': フィールド '{key}' は真偽値である想定だが {other:?}"),
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

/// `meta.toml` を読み取り 2 層ラベルの形式検証（受け入れ条件の本体）を
/// 行ったうえで、比較用の型付き値を返す。スキーマのキー集合完全一致・
/// change_id 一致・enum 値の許可リスト照合・`known_blindspot` の導出条件を
/// ここで検証する（`meta_labels_pass_two_layer_format_validation` と
/// `meta_labels_match_v1_ground_truth_table` の両方から呼ばれる共有ヘルパ）。
struct ValidatedMeta {
    change_id: String,
    category: String,
    expected_verdict: String,
    poc3_default_verdict: String,
    known_blindspot: bool,
    expected_exclusion_rule_ids: Vec<String>,
    expected_verdict_after_exclusions: String,
}

fn load_and_validate_meta(id: &str) -> ValidatedMeta {
    let path = fixtures_root().join("changes").join(id).join("meta.toml");
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?} の読み取りに失敗: {e}"));
    let raw = parse_flat_toml(&text)
        .unwrap_or_else(|e| panic!("{path:?} のパースに失敗（不正な meta.toml 構文）: {e}"));

    // 未知フィールド・欠落フィールドを両方拒否する（スキーマのキー集合完全一致）。
    let mut actual_keys: Vec<&str> = raw.keys().map(String::as_str).collect();
    actual_keys.sort_unstable();
    let mut expected_keys: Vec<&str> = REQUIRED_META_KEYS.to_vec();
    expected_keys.sort_unstable();
    assert_eq!(
        actual_keys, expected_keys,
        "change_id '{id}': meta.toml のキー集合が README のスキーマ（9 フィールド）と一致しない"
    );

    let change_id = expect_str(&raw, "change_id", id);
    assert_eq!(
        change_id, id,
        "change_id フィールド '{change_id}' がディレクトリ名 '{id}' と一致しない"
    );

    let category = expect_str(&raw, "category", id);
    assert!(
        CATEGORY_VALUES.contains(&category.as_str()),
        "change_id '{id}': category '{category}' が許可値 {CATEGORY_VALUES:?} に含まれない"
    );

    let expected_verdict = expect_str(&raw, "expected_verdict", id);
    let poc3_default_verdict = expect_str(&raw, "poc3_default_verdict", id);
    let expected_verdict_after_exclusions =
        expect_str(&raw, "expected_verdict_after_exclusions", id);
    for (field_name, value) in [
        ("expected_verdict", &expected_verdict),
        ("poc3_default_verdict", &poc3_default_verdict),
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

    let known_blindspot = expect_bool(&raw, "known_blindspot", id);
    assert_eq!(
        known_blindspot,
        expected_verdict != poc3_default_verdict,
        "change_id '{id}': known_blindspot は (expected_verdict != poc3_default_verdict) と\
         一致する想定（README「乖離する G2・G5 のみ known_blindspot=true」契約）"
    );

    // origin・summary は文字列型であることのみ検証する（内容の意味検証は対象外）。
    let _origin = expect_str(&raw, "origin", id);
    let _summary = expect_str(&raw, "summary", id);

    let expected_exclusion_rule_ids = expect_array(&raw, "expected_exclusion_rule_ids", id);
    let mut seen = std::collections::BTreeSet::new();
    for rule_id in &expected_exclusion_rule_ids {
        assert!(
            !rule_id.is_empty()
                && rule_id
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
            "change_id '{id}': rule id '{rule_id}' が文字クラス契約（[a-z0-9-]+）を満たさない"
        );
        assert!(
            seen.insert(rule_id.clone()),
            "change_id '{id}': rule id '{rule_id}' が expected_exclusion_rule_ids 内で重複している"
        );
    }

    ValidatedMeta {
        change_id,
        category,
        expected_verdict,
        poc3_default_verdict,
        known_blindspot,
        expected_exclusion_rule_ids,
        expected_verdict_after_exclusions,
    }
}

/// テストケース 1（受け入れ条件の本体）: 全 15 件の `meta.toml` が 2 層
/// ラベルモデルの形式検証を通過する。
#[test]
fn meta_labels_pass_two_layer_format_validation() {
    let ids = list_change_ids();
    assert!(ids.len() >= 15, "changes/ 配下が 15 件未満: {ids:?}");
    for id in &ids {
        // 検証自体は load_and_validate_meta 内の assert 群が担う。
        // ここでは呼び出しの網羅性（全件で panic なく完走すること）を保証する。
        let _ = load_and_validate_meta(id);
    }
}

/// README「15 件一覧」表の値をそのままピン留めした正解表（A08 改竄検知）。
/// ラベル・許容誤差の緩和をテスト差分なしに通せなくする（本ファイル冒頭
/// `//!` コメント「整合性」節参照）。
struct GroundTruthRow {
    change_id: &'static str,
    category: &'static str,
    expected_verdict: &'static str,
    poc3_default_verdict: &'static str,
    known_blindspot: bool,
    expected_exclusion_rule_ids: &'static [&'static str],
    expected_verdict_after_exclusions: &'static str,
}

const GROUND_TRUTH: &[GroundTruthRow] = &[
    GroundTruthRow {
        change_id: "S1-doc-comments",
        category: "safe",
        expected_verdict: "auto-apply",
        poc3_default_verdict: "auto-apply",
        known_blindspot: false,
        expected_exclusion_rule_ids: &["arch-hyperparameter-change"],
        expected_verdict_after_exclusions: "escalate",
    },
    GroundTruthRow {
        change_id: "S2-gelu-add",
        category: "safe",
        expected_verdict: "auto-apply",
        poc3_default_verdict: "auto-apply",
        known_blindspot: false,
        expected_exclusion_rule_ids: &[],
        expected_verdict_after_exclusions: "auto-apply",
    },
    GroundTruthRow {
        change_id: "S3-const-extract",
        category: "safe",
        expected_verdict: "auto-apply",
        poc3_default_verdict: "auto-apply",
        known_blindspot: false,
        expected_exclusion_rule_ids: &["arch-hyperparameter-change"],
        expected_verdict_after_exclusions: "escalate",
    },
    GroundTruthRow {
        change_id: "S4-S5-cosmetic-comments",
        category: "safe",
        expected_verdict: "auto-apply",
        poc3_default_verdict: "auto-apply",
        known_blindspot: false,
        expected_exclusion_rule_ids: &[],
        expected_verdict_after_exclusions: "auto-apply",
    },
    GroundTruthRow {
        change_id: "S5-inline-attr",
        category: "safe",
        expected_verdict: "auto-apply",
        poc3_default_verdict: "auto-apply",
        known_blindspot: false,
        expected_exclusion_rule_ids: &[],
        expected_verdict_after_exclusions: "auto-apply",
    },
    GroundTruthRow {
        change_id: "D1-relu-sigmoid-swap",
        category: "dangerous",
        expected_verdict: "reject",
        poc3_default_verdict: "reject",
        known_blindspot: false,
        expected_exclusion_rule_ids: &[],
        expected_verdict_after_exclusions: "reject",
    },
    GroundTruthRow {
        change_id: "D2-private-method",
        category: "dangerous",
        expected_verdict: "reject",
        poc3_default_verdict: "reject",
        known_blindspot: false,
        expected_exclusion_rule_ids: &[],
        expected_verdict_after_exclusions: "reject",
    },
    GroundTruthRow {
        change_id: "D3-redundant-calc",
        category: "dangerous",
        expected_verdict: "escalate",
        poc3_default_verdict: "escalate",
        known_blindspot: false,
        expected_exclusion_rule_ids: &["arch-hyperparameter-change"],
        expected_verdict_after_exclusions: "escalate",
    },
    GroundTruthRow {
        change_id: "D4-leaky-relu-sign-bug",
        category: "dangerous",
        expected_verdict: "reject",
        poc3_default_verdict: "reject",
        known_blindspot: false,
        expected_exclusion_rule_ids: &[],
        expected_verdict_after_exclusions: "reject",
    },
    GroundTruthRow {
        change_id: "D5-lr-bug",
        category: "dangerous",
        expected_verdict: "reject",
        poc3_default_verdict: "reject",
        known_blindspot: false,
        expected_exclusion_rule_ids: &[],
        expected_verdict_after_exclusions: "reject",
    },
    GroundTruthRow {
        change_id: "G1-gaming",
        category: "gray",
        expected_verdict: "reject",
        poc3_default_verdict: "reject",
        known_blindspot: false,
        expected_exclusion_rule_ids: &[],
        expected_verdict_after_exclusions: "reject",
    },
    GroundTruthRow {
        change_id: "G2-hidden-dim-increase",
        category: "gray",
        expected_verdict: "escalate",
        poc3_default_verdict: "auto-apply",
        known_blindspot: true,
        expected_exclusion_rule_ids: &["arch-hyperparameter-change"],
        expected_verdict_after_exclusions: "escalate",
    },
    GroundTruthRow {
        change_id: "G3-api-break",
        category: "gray",
        expected_verdict: "escalate",
        poc3_default_verdict: "escalate",
        known_blindspot: false,
        expected_exclusion_rule_ids: &[],
        expected_verdict_after_exclusions: "escalate",
    },
    GroundTruthRow {
        change_id: "G4-large-comment-refactor",
        category: "gray",
        expected_verdict: "escalate",
        poc3_default_verdict: "escalate",
        known_blindspot: false,
        expected_exclusion_rule_ids: &[],
        expected_verdict_after_exclusions: "escalate",
    },
    GroundTruthRow {
        change_id: "G5-test-only-loosen",
        category: "gray",
        expected_verdict: "escalate",
        poc3_default_verdict: "auto-apply",
        known_blindspot: true,
        expected_exclusion_rule_ids: &["test-tolerance-loosening"],
        expected_verdict_after_exclusions: "escalate",
    },
];

/// テストケース 2: 全 15 件のラベルが README「15 件一覧」表（v1 正本）と
/// 完全一致する（A08 改竄検知ピン）。
#[test]
fn meta_labels_match_v1_ground_truth_table() {
    let ids = list_change_ids();
    let ground_truth_ids: Vec<&str> = GROUND_TRUTH.iter().map(|r| r.change_id).collect();
    let mut sorted_fixture_ids = ids.clone();
    sorted_fixture_ids.sort();
    let mut sorted_ground_truth_ids: Vec<String> =
        ground_truth_ids.iter().map(|s| s.to_string()).collect();
    sorted_ground_truth_ids.sort();
    assert_eq!(
        sorted_fixture_ids, sorted_ground_truth_ids,
        "changes/ 配下の change_id 集合が GROUND_TRUTH の change_id 集合と一致しない\
         （新規追加・削除があれば GROUND_TRUTH も更新する必要がある）"
    );

    for row in GROUND_TRUTH {
        let meta = load_and_validate_meta(row.change_id);
        assert_eq!(
            meta.category, row.category,
            "change_id '{}': category",
            row.change_id
        );
        assert_eq!(
            meta.expected_verdict, row.expected_verdict,
            "change_id '{}': expected_verdict",
            row.change_id
        );
        assert_eq!(
            meta.poc3_default_verdict, row.poc3_default_verdict,
            "change_id '{}': poc3_default_verdict",
            row.change_id
        );
        assert_eq!(
            meta.known_blindspot, row.known_blindspot,
            "change_id '{}': known_blindspot",
            row.change_id
        );
        assert_eq!(
            meta.expected_exclusion_rule_ids, row.expected_exclusion_rule_ids,
            "change_id '{}': expected_exclusion_rule_ids",
            row.change_id
        );
        assert_eq!(
            meta.expected_verdict_after_exclusions, row.expected_verdict_after_exclusions,
            "change_id '{}': expected_verdict_after_exclusions",
            row.change_id
        );
    }
}

/// `poc3-result.json` を読み取り、64 KiB 上限チェック後に `serde_json::Value`
/// としてパースする。
fn load_poc3_json(id: &str) -> Value {
    let path = fixtures_root()
        .join("changes")
        .join(id)
        .join("poc3-result.json");
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?} の読み取りに失敗: {e}"));
    assert!(
        text.len() <= MAX_POC3_BYTES,
        "{path:?} が {MAX_POC3_BYTES} byte 上限を超過している（A03: DoS 的な巨大入力を拒否）"
    );
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path:?} の JSON パースに失敗: {e}"))
}

/// verdict の機械可読 ID（`meta.toml` 語彙）を `Verdict::as_ja`（`decision.rs`）
/// と同一の日本語表記へ変換する。網羅 match とし `_ =>` ワイルドカードを
/// 使わない（fail-closed。`.claude/rules/security.md` A05・`decision.rs` と
/// 同じ設計方針）。
fn verdict_id_to_ja(id: &str) -> &'static str {
    match id {
        "auto-apply" => "自動適用",
        "escalate" => "エスカレーション",
        "reject" => "却下",
        other => panic!("未知の verdict 語彙 '{other}'（VERDICT_VALUES の許可リストと不整合）"),
    }
}

/// サンプル列の統計的中央値（奇数個・偶数個どちらにも対応する一般形）。
/// `poc3-result.json` の `bench_samples_pct` は 3 件（S1）または 5 件
/// （他 9 件の `bench_ran = true` ケース）と件数が一定でないため、
/// 「5 回計測」を固定長として仮定しない（実測データに基づく設計判断）。
fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("非有限値は参照データに含まれない"));
    let n = sorted.len();
    assert!(n > 0, "空配列の中央値は定義されない");
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// テストケース 3（参照値の整理・検証）: 全 15 件の `poc3-result.json` が
/// README 記載のスキーマ（15 フィールド）・型・固定参照値を満たす。
#[test]
fn poc3_reference_values_are_well_formed() {
    for id in list_change_ids() {
        let value = load_poc3_json(&id);
        let obj = value.as_object().unwrap_or_else(|| {
            panic!("change_id '{id}': poc3-result.json のルートがオブジェクトでない")
        });

        let mut actual_keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        actual_keys.sort_unstable();
        let mut expected_keys: Vec<&str> = REQUIRED_POC3_KEYS.to_vec();
        expected_keys.sort_unstable();
        assert_eq!(
            actual_keys, expected_keys,
            "change_id '{id}': poc3-result.json のキー集合が README のスキーマ（15 フィールド）と一致しない"
        );

        let json_change_id = obj["change_id"]
            .as_str()
            .unwrap_or_else(|| panic!("change_id '{id}': change_id フィールドが文字列でない"));
        assert_eq!(
            json_change_id, id,
            "poc3-result.json の change_id フィールドがディレクトリ名と不一致"
        );

        // 参照閾値の固定値検証: v1 default プリセット由来の参照値であることの明示
        // （README「重要な注記」節。実測プリセットは常に `default`）。
        assert_eq!(
            obj["preset"].as_str(),
            Some("default"),
            "change_id '{id}': preset は 'default' 固定の想定"
        );
        assert_eq!(
            obj["lines_max"].as_u64(),
            Some(LINES_MAX),
            "change_id '{id}': lines_max は {LINES_MAX} 固定の想定"
        );
        assert_eq!(
            obj["bench_max_pct"].as_f64(),
            Some(5.0),
            "change_id '{id}': bench_max_pct は 5.0 固定の想定"
        );

        assert!(
            obj["lines_changed"].as_u64().is_some(),
            "change_id '{id}': lines_changed が非負整数でない"
        );
        for bool_field in [
            "api_broken",
            "gaming_suspect",
            "build_ok",
            "test_ok",
            "clippy_ok",
            "bench_ran",
        ] {
            assert!(
                obj[bool_field].as_bool().is_some(),
                "change_id '{id}': フィールド '{bool_field}' が真偽値でない"
            );
        }
        assert!(
            obj["bench_median_pct"].is_null() || obj["bench_median_pct"].as_f64().is_some(),
            "change_id '{id}': bench_median_pct は null または浮動小数の想定"
        );
        let samples = obj["bench_samples_pct"]
            .as_array()
            .unwrap_or_else(|| panic!("change_id '{id}': bench_samples_pct が配列でない"));
        for sample in samples {
            assert!(
                sample.as_f64().is_some(),
                "change_id '{id}': bench_samples_pct の要素が浮動小数でない"
            );
        }
        assert!(
            obj["verdict"].as_str().is_some(),
            "change_id '{id}': verdict が文字列でない"
        );
        assert!(
            obj["reasons"].as_str().is_some(),
            "change_id '{id}': reasons が文字列でない"
        );

        // change_id の三者一致（ディレクトリ名・meta.toml・poc3-result.json）。
        let meta = load_and_validate_meta(&id);
        assert_eq!(
            meta.change_id, json_change_id,
            "change_id '{id}': meta.toml と poc3-result.json の change_id が不一致"
        );
    }
}

/// テストケース 4: `poc3-result.json` の参照値内部の整合性（ベンチ計測の
/// 有無と median/samples の整合・verdict と `poc3_default_verdict` の
/// マッピング一致・行数境界）。**再構築 patch の実測行数との等値比較は
/// 行わない**（README「参照値」契約の維持。行数境界の patch 側検証は
/// `labeled_changes_fixtures.rs::line_count_boundary_matches_labels` の管轄）。
#[test]
fn poc3_reference_values_internal_consistency() {
    for id in list_change_ids() {
        let value = load_poc3_json(&id);
        let obj = value.as_object().expect("well_formed テストで検証済み");

        let bench_ran = obj["bench_ran"]
            .as_bool()
            .expect("well_formed テストで検証済み");
        let median_is_null = obj["bench_median_pct"].is_null();
        let samples = obj["bench_samples_pct"]
            .as_array()
            .expect("well_formed テストで検証済み");

        if bench_ran {
            assert!(
                !median_is_null,
                "change_id '{id}': bench_ran=true なのに bench_median_pct が null"
            );
            assert!(
                !samples.is_empty(),
                "change_id '{id}': bench_ran=true なのに bench_samples_pct が空"
            );
            let sample_values: Vec<f64> = samples
                .iter()
                .map(|v| v.as_f64().expect("well_formed テストで検証済み"))
                .collect();
            let expected_median = obj["bench_median_pct"]
                .as_f64()
                .expect("well_formed テストで検証済み");
            let computed_median = median(&sample_values);
            assert!(
                (expected_median - computed_median).abs() < 1e-9,
                "change_id '{id}': bench_median_pct（{expected_median}）が \
                 bench_samples_pct の統計的中央値（{computed_median}）と一致しない"
            );
        } else {
            assert!(
                median_is_null,
                "change_id '{id}': bench_ran=false なのに bench_median_pct が非 null"
            );
            assert!(
                samples.is_empty(),
                "change_id '{id}': bench_ran=false なのに bench_samples_pct が非空"
            );
        }

        // verdict（日本語）↔ meta.toml poc3_default_verdict のマッピング一致。
        let meta = load_and_validate_meta(&id);
        let expected_ja = verdict_id_to_ja(&meta.poc3_default_verdict);
        assert_eq!(
            obj["verdict"]
                .as_str()
                .expect("well_formed テストで検証済み"),
            expected_ja,
            "change_id '{id}': poc3-result.json の verdict が meta.toml の \
             poc3_default_verdict とマッピング不一致"
        );

        // 参照値内部の行数境界: G4 のみ超過・他 14 件は以内
        // （poc3-result.json 自身の lines_changed フィールドを使う。
        // 再構築 patch の実測行数とは独立した検証）。
        let lines_changed = obj["lines_changed"]
            .as_u64()
            .expect("well_formed テストで検証済み");
        if id == "G4-large-comment-refactor" {
            assert!(
                lines_changed > LINES_MAX,
                "change_id '{id}': lines_changed（{lines_changed}）が {LINES_MAX} 超過の想定"
            );
        } else {
            assert!(
                lines_changed <= LINES_MAX,
                "change_id '{id}': lines_changed（{lines_changed}）が {LINES_MAX} 以内の想定"
            );
        }
    }
}
