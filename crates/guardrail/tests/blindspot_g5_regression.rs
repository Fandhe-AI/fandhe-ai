//! TASK-5.4b（イシュー #130）受け入れ条件の機械検証: G5（テスト許容誤差
//! 単独緩和）ブラインドスポットの回帰テストを新実装向けに移植する。
//!
//! 受け入れ条件: 「G5 ケースが人間承認判定になる」。
//!
//! # 背景（README「発見事項5」・`fixtures/labeled-changes/changes/
//! G5-test-only-loosen/meta.toml`）
//! G5（`leaky_relu` の既知値テストの許容誤差のみを 1e-6→1e-2 に単独で緩める・
//! 本番コード変更を伴わない）は、build/test/clippy/bench/行数/API 破壊/
//! ゲーミング同時変更のいずれの機械的シグナルも回避するため、判定器単体
//! （除外リスト適用前）では `Verdict::AutoApply` に見逃す既知のブラインド
//! スポットである（`known_blindspot = true`）。この見逃しは REQ-5 の
//! ポリシー除外リスト（`policy-exclusion.toml` の `test-tolerance-loosening`
//! ルール。`policy_exclusion::builtin_defaults`・#313 で移植済み）で補う
//! 設計になっている。
//!
//! # 本ファイルのスコープ
//! `guardrail::eval::mod.rs` の既存テスト群は「除外リスト適用**前**の
//! 判定器単体では見逃す」ことのみを確認する（`eval` は REQ-5 除外リスト
//! 評価を行わない設計制約）。`label_invariant_safe_side_monotonicity.rs`
//! はラベル値どうしの severity 比較（`meta.toml` の数値のみ）に留まり、
//! 「G2/G5 ブラインドスポットの検知確認は TASK-5.4 のスコープ」と明記して
//! 本イシュー・兄弟イシュー #129 へ委譲している。
//!
//! 本ファイルはその欠落を埋め、`guardrail::decision::decide` を実 dataset の
//! G5 シグナル（`tests/fixtures/labeled-changes/changes/
//! G5-test-only-loosen/`）で実際に呼び出し、
//! - 除外リスト未適用（`exclusion_rule_ids = []`）では `Verdict::AutoApply`
//!   に見逃す（ブラインドスポットの再現。回帰の基準点）
//! - `meta.toml` の `expected_exclusion_rule_ids`（`test-tolerance-loosening`）
//!   を適用すると `Verdict::Escalate`（人間承認）になる（受け入れ条件の本体）
//!
//! ことの両方を機械検証する。G2（隠れ層次元数変更）の同種回帰テストは
//! 兄弟イシュー #129（TASK-5.4a）の `blindspot_g2_regression.rs` で扱っており
//! 本ファイルでは扱わない。
//!
//! # 除外リスト match 実装との結合範囲についての注記
//! `policy-exclusion.toml` の各ルール（パスパターン・変更種別等）から
//! `exclusion_rule_ids` を実際に導出するマッチング処理（#123
//! `feat/123-test-assertion-relaxation-rule` 等）自体の実装は本ファイルの
//! 対象外。本ファイルは `meta.toml` の `expected_exclusion_rule_ids`
//! （マッチング処理が正しく動作した場合に出力される想定の match 結果）を
//! 「既に match 済みの入力」として `DecisionInput::new` へ直接渡し、
//! `decide` 側の判定ロジック（除外リスト match は判定順序契約により却下に
//! 次ぐ優先度で無条件エスカレーションへ回る。`decision.rs`）が G5 の
//! ブラインドスポットを実際に解消することを確認する
//! （`guardrail::decision` の
//! `exclusion_match_yields_escalate_even_when_all_signals_clean` は合成入力
//! による汎用テストであり、本ファイルは実 G5 シグナルでの具体例として
//! 補完する）。マッチング処理と `decide` の結合検証は TASK-6.1（#146）の
//! スコープ（`blindspot_g2_regression.rs` と同じ整理）。
//!
//! # 依存（追加なし）
//! `meta.toml` の `expected_exclusion_rule_ids`（配列）は `guardrail::toml_lite`
//! が非対応のため、`blindspot_g2_regression.rs`・
//! `label_invariant_empty_exclusions.rs` と同一設計の std-only ミニパーサで
//! パースする（共有ヘルパー化はせずファイル内完結。
//! `.claude/rules/delegation-impl.md`「複数 Agent に同一ファイルを並行
//! 編集させない」を踏まえた既存方針を踏襲）。`guardrail::eval::dataset`・
//! `guardrail::decision`・`guardrail::config` は既存の `[dependencies]` 経由
//! （lib クレートを直接呼ぶユニット結合テスト）で追加宣言不要。
//!
//! # セキュリティ（A03。`.claude/rules/security.md`）
//! `meta.toml` はリポジトリ内データだが外部フォーマットパースとして扱い、
//! change_id は文字クラス検証済みの固定文字列リテラルのみを path join に
//! 使う（ディレクトリ列挙は行わないため既存ファイル群の
//! `is_valid_change_id` は不要）。64 KiB サイズ上限で DoS 的入力を拒否する。
//!
//! # 整合性（A08。`.claude/rules/security.md`）
//! ガードレール閾値・除外リストルール・テスト許容誤差は一切変更しない
//! （変更はユーザー承認必須）。`decide` へ渡す全シグナルは実 dataset
//! （`poc3-result.json`・`meta.toml`）由来であり、本ファイル内で合成しない。

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use guardrail::config::{PresetName, Thresholds};
use guardrail::decision::{DecisionInput, Verdict, decide};
use guardrail::eval::dataset;

/// `meta.toml` の外部入力サイズ上限（`label_invariant_*.rs` と同一値）。
const MAX_META_BYTES: usize = 64 * 1024;

/// 検証対象の change_id（G5 固定。本ファイルのスコープは G5 単体）。
const CHANGE_ID: &str = "G5-test-only-loosen";

/// `crates/guardrail/tests/fixtures/labeled-changes` への絶対パス。
fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/labeled-changes")
}

/// `meta.toml` 限定サブセットのパース結果 1 値
/// （`blindspot_g2_regression.rs::RawValue` と同型）。
#[derive(Debug, Clone, PartialEq)]
enum RawValue {
    Str(String),
    Bool(bool),
    Array(Vec<String>),
}

/// `meta.toml` 限定サブセット（フラットな `key = value`＋文字列配列＋
/// `#` 行コメント）のミニパーサ。対応文法・非対応理由は
/// `blindspot_g2_regression.rs::parse_flat_toml` と同一。
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

fn expect_array(map: &BTreeMap<String, RawValue>, key: &str, id: &str) -> Vec<String> {
    match map.get(key) {
        Some(RawValue::Array(a)) => a.clone(),
        other => {
            panic!("change_id '{id}': フィールド '{key}' は文字列配列である想定だが {other:?}")
        }
    }
}

fn expect_str(map: &BTreeMap<String, RawValue>, key: &str, id: &str) -> String {
    match map.get(key) {
        Some(RawValue::Str(s)) => s.clone(),
        other => panic!("change_id '{id}': フィールド '{key}' は文字列である想定だが {other:?}"),
    }
}

/// `meta.toml` の verdict 語彙（ハイフン区切り）を `guardrail::decision::Verdict`
/// へ変換する。網羅 match とし `_ =>` ワイルドカードは使わない（fail-closed。
/// `labeled_changes_labels.rs::verdict_id_to_ja` と同じ設計方針）。ラベルが
/// 見逃し方向（`auto-apply`）へ改竄されても本ファイルのアサーションが
/// サイレントに緩まないよう、ハードコードではなくこの変換経由で
/// `decide()` の結果と突き合わせる。
fn verdict_id_to_verdict(id: &str) -> Verdict {
    match id {
        "auto-apply" => Verdict::AutoApply,
        "escalate" => Verdict::Escalate,
        "reject" => Verdict::Reject,
        other => panic!("未知の verdict 語彙 '{other}'（許可値 auto-apply/escalate/reject）"),
    }
}

/// G5 の `meta.toml` から `expected_exclusion_rule_ids`／
/// `expected_verdict_after_exclusions` を読み取る（他フィールドの形式検証は
/// `labeled_changes_labels.rs` の責務であり本ファイルでは重複しない）。
struct G5ExclusionLabels {
    expected_exclusion_rule_ids: Vec<String>,
    expected_verdict_after_exclusions: Verdict,
}

fn load_g5_exclusion_labels() -> G5ExclusionLabels {
    let path = fixtures_root()
        .join("changes")
        .join(CHANGE_ID)
        .join("meta.toml");
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?} の読み取りに失敗: {e}"));
    let raw = parse_flat_toml(&text)
        .unwrap_or_else(|e| panic!("{path:?} のパースに失敗（不正な meta.toml 構文）: {e}"));
    let expected_exclusion_rule_ids = expect_array(&raw, "expected_exclusion_rule_ids", CHANGE_ID);
    let expected_verdict_after_exclusions = verdict_id_to_verdict(&expect_str(
        &raw,
        "expected_verdict_after_exclusions",
        CHANGE_ID,
    ));
    G5ExclusionLabels {
        expected_exclusion_rule_ids,
        expected_verdict_after_exclusions,
    }
}

fn default_thresholds() -> Thresholds {
    Thresholds::builtin(PresetName::Default)
}

/// 実 dataset の G5 シグナル（`poc3-result.json`・`meta.toml` の
/// `expected_verdict`／`known_blindspot`）を読み込み、`exclusion_rule_ids`
/// のみ差し替えて `decide` を呼ぶ。
fn decide_g5_with_exclusion_rule_ids(exclusion_rule_ids: Vec<String>) -> Verdict {
    let thresholds = default_thresholds();
    let change = dataset::load_change(&fixtures_root(), CHANGE_ID)
        .unwrap_or_else(|e| panic!("G5 dataset の読み込みに失敗: {e}"));

    let input = DecisionInput::new(
        &thresholds,
        change.lines_changed,
        change.gates,
        change.api_broken,
        change.gaming_suspect,
        change.bench,
        exclusion_rule_ids,
    )
    .expect("G5 の実シグナルは矛盾なし入力の想定（poc3-result.json 実測由来）");

    decide(&input).expect("G5 の判定に失敗").verdict()
}

/// 回帰の基準点: 除外リスト未適用では、判定器単体は G5 を `AutoApply` に
/// 見逃す（既知のブラインドスポット。`known_blindspot = true` の再現）。
/// この事実自体が変わらないことを確認しておくことで、次のテスト
/// （除外リスト適用後に `Escalate` へ変わること）が「除外リストの適用が
/// 効いた結果」であると区別できる。
#[test]
fn g5_without_exclusion_list_is_still_missed_as_auto_apply() {
    let change = dataset::load_change(&fixtures_root(), CHANGE_ID)
        .unwrap_or_else(|e| panic!("G5 dataset の読み込みに失敗: {e}"));
    assert_eq!(
        change.expected_verdict,
        Verdict::Escalate,
        "G5 の正解ラベル（除外リスト適用前）は escalate の想定"
    );
    assert!(
        change.known_blind_spot,
        "G5 は known_blindspot = true の想定（README「発見事項5」）"
    );

    let verdict = decide_g5_with_exclusion_rule_ids(Vec::new());
    assert_eq!(
        verdict,
        Verdict::AutoApply,
        "G5 は除外リスト未適用では判定器単体が AutoApply に見逃す想定\
         （ブラインドスポットの再現。回帰の基準点）"
    );
}

/// 受け入れ条件の本体（イシュー #130）: `meta.toml` の
/// `expected_exclusion_rule_ids`（`test-tolerance-loosening`）を適用する
/// と、G5 は `Verdict::Escalate`（人間承認）になる。
#[test]
fn g5_case_becomes_human_approval_after_policy_exclusion() {
    let labels = load_g5_exclusion_labels();
    assert!(
        !labels.expected_exclusion_rule_ids.is_empty(),
        "G5 の expected_exclusion_rule_ids が空。fixture のラベルが変更され\
         テストの前提が崩れている可能性がある"
    );
    assert!(
        labels
            .expected_exclusion_rule_ids
            .iter()
            .any(|id| id == "test-tolerance-loosening"),
        "G5 は test-tolerance-loosening ルールに match する想定だが\
         実際の expected_exclusion_rule_ids は {:?}",
        labels.expected_exclusion_rule_ids
    );
    // 受け入れ条件「G5 ケースが人間承認判定になる」は Escalate を指す。
    // ハードコードではなく meta.toml 由来の値と突き合わせ、ラベルが
    // 見逃し方向へ改竄された場合にも検知できるようにする（A08）。
    assert_eq!(
        labels.expected_verdict_after_exclusions,
        Verdict::Escalate,
        "G5 の expected_verdict_after_exclusions が escalate でない。\
         fixture のラベルが見逃し方向へ変更されている可能性がある\
         （ガードレール閾値・許容誤差の緩和はユーザー承認必須。\
         `.claude/rules/security.md`）"
    );

    let verdict = decide_g5_with_exclusion_rule_ids(labels.expected_exclusion_rule_ids);
    assert_eq!(
        verdict, labels.expected_verdict_after_exclusions,
        "受け入れ条件違反: 除外リスト（test-tolerance-loosening）適用後の\
         G5 は expected_verdict_after_exclusions（{:?}）になる想定だが\
         {verdict:?} だった（TASK-5.4b・イシュー #130\
         「G5 ケースが人間承認判定になる」）",
        labels.expected_verdict_after_exclusions
    );
}
