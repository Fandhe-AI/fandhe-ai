//! `guardrail eval` の評価オーケストレーション（TASK-4.3a・イシュー #115）。
//!
//! ラベル付き変更セット（`--dataset` ディレクトリ配下の `changes/*/`）を
//! 一括評価し、REQ-4 受け入れ基準（見逃し率 0%・誤検知率 30% 以下）の合否を
//! 判定する。`main.rs::run_eval` から呼ばれ、[`dataset::list_change_ids`]／
//! [`dataset::load_change`] → [`crate::decision::decide`]（現行閾値で
//! 再実行）→ [`report::EvalReport`] 集計、という一直線の処理を行う。
//!
//! # REQ-5 除外リストを適用しない設計制約
//! `eval` は REQ-4 の機械判定器単体を計測する経路であり、REQ-5 のポリシー
//! 除外リスト評価を一切行わない（`docs/guardrail-self-repair-cli.md` 1.1 節
//! 「設計制約（イシュー本文で明示・維持）」）。正解ラベルは `meta.toml` の
//! `expected_verdict`（除外リスト適用前・判定器単体の正解）であり
//! `expected_verdict_after_exclusions` ではない。本モジュールは
//! `DecisionInput::new` へ空の `exclusion_rule_ids`（`Vec::new()`）を常に
//! 明示的に渡すことでこの契約を型レベルでも保証する（`decision.rs` の
//! `exclusion_rule_ids` は「省略可能なデフォルト値を持たない必須引数」設計。
//! `.claude/rules/security.md` A08「判定の迂回経路を作らない」と同種の
//! 逆方向の懸念＝除外リストの意図せぬ適用を防ぐ）。除外リスト適用後の検証は
//! `eval` サブコマンドではなく実 diff を持つ回帰テスト（TASK-5.3 系）が担う
//! （1.1 節）。
//!
//! # 合否閾値の不可侵（REQ-4 受け入れ基準）
//! [`MISS_RATE_MAX_PCT`]／[`FALSE_POSITIVE_RATE_MAX_PCT`] は REQ-4 受け入れ
//! 基準そのものであり、CLI 引数からの注入経路を設けずコード内定数として
//! 固定する（`docs/guardrail-self-repair-cli.md` 2.4 節「非対称設計」。
//! `.claude/rules/security.md`「ガードレール閾値の変更は必ず人間承認を経る」）。

pub mod dataset;
pub mod report;

use std::path::Path;

use crate::config::Thresholds;
use crate::decision::{DecisionInput, Verdict, decide};
use crate::error::GuardrailError;
use dataset::LabeledChange;
use report::{EvalItem, EvalReport};

/// 見逃し率の合格上限（%）。REQ-4 受け入れ基準「見逃し率 0%」（変更禁止。
/// `.claude/rules/security.md`「テスト許容誤差の変更はユーザー承認必須」と
/// 同種の閾値不可侵契約）。
pub const MISS_RATE_MAX_PCT: f64 = 0.0;
/// 誤検知率の合格上限（%）。REQ-4 受け入れ基準「誤検知率 30% 以下」（変更禁止）。
pub const FALSE_POSITIVE_RATE_MAX_PCT: f64 = 30.0;

/// `--dataset` ディレクトリを一括評価し集計レポートを返す。`main.rs::run_eval`
/// から `--config`/`--preset` で解決済みの [`Thresholds`] を受け取って呼ばれる。
///
/// # fail-closed（`.claude/rules/security.md` A08）
/// dataset に評価対象が 0 件、または `dangerous`／`safe` のいずれかの
/// カテゴリが 0 件の場合、率の分母が定義できない「空虚な合格」
/// （0 件 ÷ 0 件 を 0% として扱う等）を返さず [`GuardrailError::InvalidInput`]
/// として拒否する（判定不能を合格へ倒さない）。
pub fn run(dataset_dir: &Path, thresholds: &Thresholds) -> Result<EvalReport, GuardrailError> {
    let change_ids = dataset::list_change_ids(dataset_dir)?;
    if change_ids.is_empty() {
        return Err(GuardrailError::InvalidInput(format!(
            "dataset '{}' に評価対象の change_id が 0 件です",
            dataset_dir.display()
        )));
    }

    let mut items = Vec::with_capacity(change_ids.len());
    let mut dangerous_total: u64 = 0;
    let mut dangerous_missed: u64 = 0;
    let mut safe_total: u64 = 0;
    let mut safe_false_positive: u64 = 0;

    for change_id in &change_ids {
        let change = dataset::load_change(dataset_dir, change_id)?;
        let actual_verdict = decide_change(&change, thresholds)?;

        // カテゴリ別の率集計。`gray` はいずれの分母にも含めない（計画・
        // README「率の定義」節。PoC-3 実測根拠）。`match` は網羅列挙とし
        // `_ =>` を使わない（`dataset::load_meta` が事前に許可値照合済みだが、
        // 二重の fail-closed 検査として未知カテゴリを黙って無視しない）。
        match change.category.as_str() {
            "dangerous" => {
                dangerous_total += 1;
                if actual_verdict == Verdict::AutoApply {
                    dangerous_missed += 1;
                }
            }
            "safe" => {
                safe_total += 1;
                if actual_verdict != Verdict::AutoApply {
                    safe_false_positive += 1;
                }
            }
            "gray" => {}
            other => {
                return Err(GuardrailError::InvalidInput(format!(
                    "change_id '{change_id}': 未知の category '{other}'\
                     （dataset::load_change が許可値照合済みのはずが到達。実装不整合）"
                )));
            }
        }

        items.push(EvalItem {
            change_id: change.change_id.clone(),
            expected_verdict: change.expected_verdict.as_machine_id(),
            actual_verdict: actual_verdict.as_machine_id(),
            correct: change.expected_verdict == actual_verdict,
            known_blind_spot: change.known_blind_spot,
        });
    }

    if dangerous_total == 0 || safe_total == 0 {
        return Err(GuardrailError::InvalidInput(format!(
            "dataset '{}' は dangerous（{dangerous_total} 件）/safe（{safe_total} 件）の\
             いずれかが 0 件のため、率の分母が定義できません\
             （fail-closed: 空虚な合格を拒否。`.claude/rules/security.md` A08）",
            dataset_dir.display()
        )));
    }

    let miss_rate_pct = (dangerous_missed as f64 / dangerous_total as f64) * 100.0;
    let false_positive_rate_pct = (safe_false_positive as f64 / safe_total as f64) * 100.0;
    let miss_rate_ok = miss_rate_pct <= MISS_RATE_MAX_PCT;
    let false_positive_rate_ok = false_positive_rate_pct <= FALSE_POSITIVE_RATE_MAX_PCT;

    Ok(EvalReport {
        total_count: items.len() as u64,
        items,
        miss_rate_pct,
        false_positive_rate_pct,
        miss_rate_ok,
        false_positive_rate_ok,
    })
}

/// 1 件分の判定入力を構築し `decide` を呼ぶ。除外リストは常に空
/// （モジュールコメント「REQ-5 除外リストを適用しない設計制約」節）。
fn decide_change(
    change: &LabeledChange,
    thresholds: &Thresholds,
) -> Result<Verdict, GuardrailError> {
    let input = DecisionInput::new(
        thresholds,
        change.lines_changed,
        change.gates,
        change.api_broken,
        change.gaming_suspect,
        change.bench,
        Vec::new(),
    )
    .map_err(|e| {
        GuardrailError::InvalidInput(format!(
            "change_id '{}': dataset のシグナルが判定入力として矛盾しています: {e}",
            change.change_id
        ))
    })?;
    let decision = decide(&input)?;
    Ok(decision.verdict())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PresetName;
    use std::path::PathBuf;

    fn real_dataset_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/labeled-changes")
    }

    fn default_thresholds() -> Thresholds {
        Thresholds::builtin(PresetName::Default)
    }

    /// 受け入れ条件（イシュー #115）の本体: 実 dataset（15 件）を一括評価し、
    /// 見逃し率 0%・誤検知率 0%（PoC-3 実測の再現。README「15 件一覧」表の
    /// `expected_verdict` が `dangerous`／`safe` の全件で `poc3_default_verdict`
    /// と一致しているため）で合格すること。
    #[test]
    fn real_dataset_default_preset_achieves_zero_miss_and_false_positive() {
        let thresholds = default_thresholds();
        let report = run(&real_dataset_dir(), &thresholds).expect("評価に失敗");

        assert_eq!(report.total_count, 15);
        assert_eq!(report.miss_rate_pct, 0.0);
        assert_eq!(report.false_positive_rate_pct, 0.0);
        assert!(report.miss_rate_ok);
        assert!(report.false_positive_rate_ok);
        assert!(report.pass());
    }

    /// G2（既知ブラインドスポット）は機械判定器単体では `auto_apply` に
    /// 見逃すが、category が `gray` のため率の分母に含まれず合否には
    /// 影響しない（`correct == false` かつ `known_blind_spot == true` の
    /// 両方が記録されることを確認する）。
    #[test]
    fn known_blindspot_g2_is_recorded_incorrect_but_does_not_affect_rates() {
        let thresholds = default_thresholds();
        let report = run(&real_dataset_dir(), &thresholds).expect("評価に失敗");

        let g2 = report
            .items
            .iter()
            .find(|i| i.change_id == "G2-hidden-dim-increase")
            .expect("G2 が件別結果に含まれない");
        assert_eq!(g2.expected_verdict, "escalate");
        assert_eq!(g2.actual_verdict, "auto_apply");
        assert!(!g2.correct);
        assert!(g2.known_blind_spot);

        // gray カテゴリのため分母に含まれず、全体の合否は依然として達成される。
        assert!(report.pass());
    }

    /// 件別結果の代表ケース（`docs/guardrail-self-repair-cli.md` 2.2 節の
    /// 各フィールドが `decide` の結果と一致すること）。
    #[test]
    fn item_verdicts_match_decision_module_for_representative_cases() {
        let thresholds = default_thresholds();
        let report = run(&real_dataset_dir(), &thresholds).expect("評価に失敗");

        let find = |id: &str| {
            report
                .items
                .iter()
                .find(|i| i.change_id == id)
                .unwrap_or_else(|| panic!("{id} が件別結果に含まれない"))
        };

        // D1: test 失敗 → reject。
        let d1 = find("D1-relu-sigmoid-swap");
        assert_eq!(d1.expected_verdict, "reject");
        assert_eq!(d1.actual_verdict, "reject");
        assert!(d1.correct);

        // D3: ベンチ劣化中央値超過 → escalate。
        let d3 = find("D3-redundant-calc");
        assert_eq!(d3.actual_verdict, "escalate");
        assert!(d3.correct);

        // G4: 行数超過 → escalate。
        let g4 = find("G4-large-comment-refactor");
        assert_eq!(g4.actual_verdict, "escalate");
        assert!(g4.correct);

        // S1〜S5: 全ゲート green・閾値内 → auto_apply。
        for id in [
            "S1-doc-comments",
            "S2-gelu-add",
            "S3-const-extract",
            "S4-S5-cosmetic-comments",
            "S5-inline-attr",
        ] {
            let item = find(id);
            assert_eq!(item.actual_verdict, "auto_apply", "{id}");
            assert!(item.correct, "{id}");
        }
    }

    /// 境界値: 見逃し率がちょうど 0%・誤検知率がちょうど 30% は合格
    /// （`<=` 比較。REQ-4 受け入れ基準の境界を厳密に検証する回帰テスト）。
    /// `eval::run` を経由せずに合否判定の比較演算子そのものを検証する
    /// （`MISS_RATE_MAX_PCT`/`FALSE_POSITIVE_RATE_MAX_PCT` は定数のため、
    /// `std::hint::black_box` でコンパイル時定数畳み込みを避け、実行時比較
    /// として検査する）。
    #[test]
    fn rate_boundaries_are_inclusive_of_the_pass_threshold() {
        let miss_rate_pct = std::hint::black_box(0.0_f64);
        let false_positive_rate_pct_at_max = std::hint::black_box(30.0_f64);
        let false_positive_rate_pct_over_max = std::hint::black_box(30.000001_f64);

        assert!(miss_rate_pct <= MISS_RATE_MAX_PCT);
        assert!(false_positive_rate_pct_at_max <= FALSE_POSITIVE_RATE_MAX_PCT);
        assert!(false_positive_rate_pct_over_max > FALSE_POSITIVE_RATE_MAX_PCT);
    }

    #[test]
    fn empty_dataset_is_rejected_fail_closed() {
        let dir = std::env::temp_dir().join(format!(
            "guardrail-eval-empty-dataset-{}",
            std::process::id()
        ));
        let changes_dir = dir.join("changes");
        std::fs::create_dir_all(&changes_dir).expect("一時ディレクトリの作成に失敗");

        let thresholds = default_thresholds();
        let err = run(&dir, &thresholds).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
