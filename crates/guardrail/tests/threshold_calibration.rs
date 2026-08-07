//! `guardrail eval` の 3 プリセット掃引・候補閾値ピン留め回帰テスト
//! （TASK-4.3b・イシュー #116）。
//!
//! `tests/eval_harness.rs`（TASK-4.3a・イシュー #115）は `default` プリセット
//! 単独の CLI プロセス境界検証を担う。本ファイルはその上に「strict/default/
//! loose 3 プリセットいずれも実 dataset で REQ-4 受け入れ基準（見逃し率 0%・
//! 誤検知率 30% 以下）を満たす」という掃引結果と、
//! `docs/guardrail-threshold-recalibration.md` に記録した候補閾値（`default`）
//! の実測値（0.0%/0.0%）を実装側にピン留めする（記録と実装の乖離検知。
//! `.claude/rules/security.md`「ガードレール閾値の変更は必ず人間承認を経る」
//! 契約下で、閾値定数自体は変えず「記録した数値が実装からずれていないか」を
//! 継続監視する TASK-6.1（判定器自己回帰 CI）の先行資産）。
//!
//! `--repo` は空の一時ディレクトリを明示的に渡す。デフォルト値 `.`（リポジトリ
//! ルート）に `guardrail.toml` が存在すると `config::resolve` がそれを拾って
//! しまい、#117（TASK-4.3c）が確定閾値を反映した `guardrail.toml` を追加した
//! 時点で本テストがピン留めしている数値が暗黙に変わる（`--preset` 経由の
//! 組み込み既定値ではなくファイル上書き値を意図せず検証してしまう）。

use std::path::{Path, PathBuf};
use std::process::Command;

use guardrail::config::{self, PresetName, Thresholds};

fn guardrail_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_guardrail"))
}

fn real_dataset_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/labeled-changes")
}

/// `guardrail.toml` を含まないことが保証された空リポジトリルート
/// （`--repo` に渡し `config::resolve` の探索順序 2 段目〈`--repo` 直下〉を
/// 常に「ファイルなし → 組み込み既定値」へ倒すため）。
fn empty_repo_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "guardrail-threshold-calibration-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("空 --repo 用一時ディレクトリの作成に失敗");
    dir
}

struct PresetExpectation {
    preset: &'static str,
    expected_exit_code: i32,
    expected_miss_rate_pct: f64,
    expected_false_positive_rate_pct: f64,
}

/// 3 プリセット掃引（計画ステップ 2 の本体）。strict/default/loose のいずれも
/// 見逃し率 0%・誤検知率 0%（30% 以下の基準を厳格に上回る余裕を持って達成）
/// で合格することを検証する。率だけでなく件別 verdict も確認するのは、
/// 率の分母（dangerous/safe のみ。gray を含まない）が prese 間の判定差
/// （後続テスト参照）を隠しうるため（`docs/guardrail-threshold-recalibration.md`
/// 「プリセット間で率が同一の理由」節）。
#[test]
fn eval_sweep_across_three_presets_all_pass_with_zero_rates() {
    let expectations = [
        PresetExpectation {
            preset: "strict",
            expected_exit_code: 0,
            expected_miss_rate_pct: 0.0,
            expected_false_positive_rate_pct: 0.0,
        },
        PresetExpectation {
            preset: "default",
            expected_exit_code: 0,
            expected_miss_rate_pct: 0.0,
            expected_false_positive_rate_pct: 0.0,
        },
        PresetExpectation {
            preset: "loose",
            expected_exit_code: 0,
            expected_miss_rate_pct: 0.0,
            expected_false_positive_rate_pct: 0.0,
        },
    ];

    for exp in expectations {
        let repo_dir = empty_repo_dir(exp.preset);
        let output = guardrail_bin()
            .args([
                "eval",
                "--dataset",
                real_dataset_dir().to_str().expect("非 UTF-8 パス"),
                "--repo",
                repo_dir.to_str().expect("非 UTF-8 パス"),
                "--preset",
                exp.preset,
                "--format",
                "json",
            ])
            .output()
            .expect("failed to run guardrail binary");

        assert_eq!(
            output.status.code(),
            Some(exp.expected_exit_code),
            "preset={} stdout={} stderr={}",
            exp.preset,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let json: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("stdout が有効な JSON でない");
        assert_eq!(
            json["total_count"], 15,
            "preset={} total_count が 15 件でない",
            exp.preset
        );
        assert_eq!(
            json["miss_rate_pct"], exp.expected_miss_rate_pct,
            "preset={} miss_rate_pct が候補閾値記録と乖離",
            exp.preset
        );
        assert_eq!(
            json["false_positive_rate_pct"], exp.expected_false_positive_rate_pct,
            "preset={} false_positive_rate_pct が候補閾値記録と乖離",
            exp.preset
        );
        assert_eq!(json["miss_rate_ok"], true, "preset={}", exp.preset);
        assert_eq!(
            json["false_positive_rate_ok"], true,
            "preset={}",
            exp.preset
        );

        let _ = std::fs::remove_dir_all(&repo_dir);
    }
}

/// プリセット間で率が同一でも、`loose`（`lines_max=400`）は `default`／`strict`
/// （`lines_max=200`／`100`）と異なり G4（実測 lines_changed=210。
/// `docs/guardrail-recalibration/v2-measured-signals.json`）を `escalate` から
/// `auto_apply` へ取りこぼす。G4 は `category=gray` のため miss/false-positive
/// 率の分母（dangerous/safe のみ）に含まれず合否には現れない（率だけを見て
/// 3 プリセットを同列に扱わない、という記録文書の結論を実装側でも固定する）。
#[test]
fn loose_preset_misses_g4_via_item_level_verdict_while_default_and_strict_catch_it() {
    let strict_repo = empty_repo_dir("g4-strict");
    let default_repo = empty_repo_dir("g4-default");
    let loose_repo = empty_repo_dir("g4-loose");

    let g4_verdict = |preset: &str, repo_dir: &Path| -> String {
        let output = guardrail_bin()
            .args([
                "eval",
                "--dataset",
                real_dataset_dir().to_str().expect("非 UTF-8 パス"),
                "--repo",
                repo_dir.to_str().expect("非 UTF-8 パス"),
                "--preset",
                preset,
                "--format",
                "json",
            ])
            .output()
            .expect("failed to run guardrail binary");
        let json: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("stdout が有効な JSON でない");
        json["items"]
            .as_array()
            .expect("items が配列でない")
            .iter()
            .find(|i| i["change_id"] == "G4-large-comment-refactor")
            .expect("G4 が件別結果に含まれない")["actual_verdict"]
            .as_str()
            .expect("actual_verdict が文字列でない")
            .to_string()
    };

    assert_eq!(g4_verdict("strict", &strict_repo), "escalate");
    assert_eq!(g4_verdict("default", &default_repo), "escalate");
    assert_eq!(
        g4_verdict("loose", &loose_repo),
        "auto_apply",
        "loose（lines_max=400）は G4（>200 行超過のみを検知条件とする境界例）を \
         見逃す設計上の弱点を持つ。gray カテゴリのため率には出ないが、\
         件別 verdict では観測できる（記録文書『候補閾値』節の選定根拠）"
    );

    let _ = std::fs::remove_dir_all(&strict_repo);
    let _ = std::fs::remove_dir_all(&default_repo);
    let _ = std::fs::remove_dir_all(&loose_repo);
}

/// リポジトリルート直下の `guardrail.toml`（イシュー #117・TASK-4.3c で新設）が
/// `config::resolve` の探索順序 2 段目（`docs/guardrail-self-repair-cli.md` 2.4
/// 節）で実際に解決され、承認済み確定値（`docs/guardrail-threshold-recalibration.md`
/// 「5. 確定記録」節・200／5.0／5）と組み込み既定値
/// （`Thresholds::builtin(PresetName::Default)`）の両方に一致することをピン留めする。
/// `guardrail.toml` の無断改変や既定値との意図しない乖離を fail-closed で検知する
/// （`.claude/rules/security.md`「ガードレール閾値の変更は必ず人間承認を経る」）。
#[test]
fn committed_root_guardrail_toml_resolves_to_approved_default_thresholds() {
    // `crates/guardrail` から見たリポジトリルート（`CARGO_MANIFEST_DIR/../..`）。
    // 他テストが `--repo` に空の一時ディレクトリを渡しコミット済み設定を意図的に
    // 迂回するのに対し、本テストは逆にコミット済み `guardrail.toml` そのものを
    // 対象とする（探索順序 2 段目の実地検証）。
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("リポジトリルートの解決に失敗");
    assert!(
        repo_root.join("guardrail.toml").is_file(),
        "リポジトリルート直下に guardrail.toml が見つからない: {}",
        repo_root.display()
    );

    let resolved = config::resolve(None, &repo_root, PresetName::Default)
        .expect("guardrail.toml の解決に失敗");
    let approved = Thresholds {
        lines_max: 200,
        bench_median_max_pct: 5.0,
        bench_runs_min: 5,
    };
    assert_eq!(
        resolved.thresholds, approved,
        "guardrail.toml から解決された値が承認済み確定値（200/5.0/5）と乖離"
    );
    assert_eq!(
        resolved.thresholds,
        Thresholds::builtin(PresetName::Default),
        "guardrail.toml の値が組み込み既定値と乖離（本イシューでは数値変更なしのはず）"
    );
}
