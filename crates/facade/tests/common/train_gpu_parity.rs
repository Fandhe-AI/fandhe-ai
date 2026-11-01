//! NN 層・最適化・ONNX の学習の実機 parity テスト共通部品（イシュー #3778）。
//!
//! **役割**: `tests/train_gpu_parity.rs` の登録表（`CASES`）が返す測定量を、腕 A
//! （被検腕。`tape_for(<device>)`）と腕 B（参照腕。既定は CPU）で比較し、結果を
//! `TRAIN_PARITY` 行として記録する部品を 1 か所に集約する。Phase 6〜8 の実装 issue
//! は `CASES` に行を足すだけで、判定・記録の書式は本ファイルが揃える。Phase 10 の
//! 実機測定 issue（#3860〜#3865）は `--nocapture` ログの `TRAIN_PARITY` 行を集計する。
//!
//! **判定契約**: 判定式は書かない。`Judge::Req2` は
//! `fandhe_ai_backend_cpu::parity::compare`（REQ-2 統一複合判定の唯一の実体。閾値の
//! 正本は `fandhe_ai_tensor_core::tolerance`）を呼び `passes()` を見るだけである。
//! tolerance・baseline は本ファイルで持たず、変更もしない（ユーザー承認必須）。
//! `Judge::Bit` は設計が bit 一致を約束する経路（param groups の常駐更新等）用の
//! 構造上の検査で、閾値を持たない。
//!
//! **fail-closed**: 名前集合の不一致・長さ不一致・全ゼロ（`require_nonzero`）・run 関数の
//! `Err`・登録 0 件の族はいずれも `undetermined` であり、pass として扱わない。
//!
//! `#![allow(dead_code)]` の理由: `#[path]` で取り込むテストファイルごとに使う項目が
//! 異なる（`tests/common/mod.rs` と同じ事情）ため。

#![allow(dead_code)]

use fandhe_ai::Device;
use fandhe_ai_backend_cpu::parity::{CompareReport, compare};

/// 経路の区分。H の pass は「経路の parity」であってカーネル同士の parity ではない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// デバイスカーネル経路。
    K,
    /// ホストへの実体化を含むフォールバック（`Tape::custom` の `Erf`・`Softmax`・
    /// `LayerNormalization` 等）。
    H,
    /// opt-in ゲートの後ろ（既定 OFF なので H として測る）。
    G,
}

/// 判定方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Judge {
    /// REQ-2 統一複合判定（`compare` を呼ぶだけ）。
    Req2,
    /// bit 一致（NaN のみクラス一致）。tolerance を持たない構造上の検査。
    Bit,
}

/// 1 ケースが返す測定量。
pub struct Quantity {
    pub name: &'static str,
    pub values: Vec<f32>,
    /// 全ゼロで自明に通るのを防ぐガード。
    pub require_nonzero: bool,
}

pub type RunFn = fn(Device) -> Result<Vec<Quantity>, String>;
pub type RefFn = fn() -> Result<Vec<Quantity>, String>;

/// 登録表の 1 行。
pub struct Case {
    pub name: &'static str,
    pub family: &'static str,
    pub route: Route,
    pub judge: Judge,
    /// RULE.txt と README に写す形状の宣言文字列。
    pub shape: &'static str,
    /// 腕 A。`run(device)`。
    pub run: RunFn,
    /// 腕 B。`None` なら `run(Device::Cpu)`。
    pub reference: Option<RefFn>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
}

/// 量ごとの記録。
pub struct Record {
    pub case: &'static str,
    pub family: &'static str,
    pub quantity: &'static str,
    pub arm: String,
    pub route: Route,
    pub judge: Judge,
    pub report: CompareReport,
    pub verdict: Verdict,
}

/// 1 行・key=value 形式の記録行。`--nocapture` ログから grep で集計できる。
pub fn record_line(r: &Record) -> String {
    format!(
        "TRAIN_PARITY case={} family={} quantity={} arm={} route={:?} judge={:?} total={} fail_count={} max_abs_diff={:.6e} mean_abs_diff={:.6e} max_rel_err={:.6e} verdict={}",
        r.case,
        r.family,
        r.quantity,
        r.arm,
        r.route,
        r.judge,
        r.report.total,
        r.report.fail_count,
        r.report.max_abs_diff,
        r.report.mean_abs_diff,
        r.report.max_rel_err,
        match r.verdict {
            Verdict::Pass => "pass",
            Verdict::Fail => "fail",
        }
    )
}

/// undetermined 行（判定に至らなかった理由つき）。
pub fn undetermined_line(case: &str, family: &str, arm: &str, reason: &str) -> String {
    format!(
        "TRAIN_PARITY case={case} family={family} arm={arm} verdict=undetermined reason={reason}"
    )
}

/// bit 一致判定（NaN は NaN 同士をクラス一致）。`+0.0` と `-0.0` は区別する。
fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits())
}

/// 2 つの測定量列を比較して量ごとの記録を返す。undetermined 要因は `Err`。
pub fn compare_quantities(
    case: &Case,
    arm_label: &str,
    a: &[Quantity],
    b: &[Quantity],
) -> Result<Vec<Record>, String> {
    if a.is_empty() || b.is_empty() {
        return Err("測定量が 0 件（空の測定を pass 扱いしない）".to_string());
    }
    if let Some(q) = a.iter().chain(b).find(|q| q.values.is_empty()) {
        return Err(format!(
            "量 {} の values が空（total=0 の自明な一致を許さない）",
            q.name
        ));
    }
    let names_a: Vec<_> = a.iter().map(|q| q.name).collect();
    let names_b: Vec<_> = b.iter().map(|q| q.name).collect();
    if names_a != names_b {
        return Err(format!(
            "測定量の名前集合・順序が腕間で一致しない: {names_a:?} vs {names_b:?}"
        ));
    }
    let mut out = Vec::new();
    for (qa, qb) in a.iter().zip(b) {
        if qa.require_nonzero
            && (qa.values.iter().all(|v| *v == 0.0) || qb.values.iter().all(|v| *v == 0.0))
        {
            return Err(format!("量 {} が全ゼロ（自明な一致を許さない）", qa.name));
        }
        let report = compare(&qa.values, &qb.values)
            .map_err(|e| format!("量 {} の比較入力が不正: {e}", qa.name))?;
        let ok = match case.judge {
            Judge::Req2 => report.passes(),
            Judge::Bit => bits_equal(&qa.values, &qb.values),
        };
        out.push(Record {
            case: case.name,
            family: case.family,
            quantity: qa.name,
            arm: arm_label.to_string(),
            route: case.route,
            judge: case.judge,
            report,
            verdict: if ok { Verdict::Pass } else { Verdict::Fail },
        });
    }
    Ok(out)
}

/// 1 ケースを測る（腕 A と腕 B を走らせて比較）。
pub fn run_case(case: &Case, device: Device) -> Result<Vec<Record>, String> {
    let a = (case.run)(device).map_err(|e| format!("腕 A の実行が失敗: {e}"))?;
    let b = match case.reference {
        Some(f) => f(),
        None => (case.run)(Device::Cpu),
    }
    .map_err(|e| format!("腕 B の実行が失敗: {e}"))?;
    compare_quantities(case, &format!("{device:?}"), &a, &b)
}

/// 族の全ケースを測り、記録行を出力して fail / undetermined を 1 つの panic にまとめる。
pub fn run_family(cases: &[Case], family: &str, device: Device) {
    let selected: Vec<&Case> = cases.iter().filter(|c| c.family == family).collect();
    if selected.is_empty() {
        panic!("undetermined: 族 {family} の登録が 0 件");
    }
    let arm = format!("{device:?}");
    let mut problems = Vec::new();
    for case in selected {
        match run_case(case, device) {
            Ok(records) => {
                for r in &records {
                    println!("{}", record_line(r));
                    if r.verdict == Verdict::Fail {
                        problems.push(format!(
                            "fail case={} quantity={} fail_count={} max_abs_diff={:.6e} max_rel_err={:.6e}",
                            r.case,
                            r.quantity,
                            r.report.fail_count,
                            r.report.max_abs_diff,
                            r.report.max_rel_err
                        ));
                    }
                }
            }
            Err(reason) => {
                println!("{}", undetermined_line(case.name, family, &arm, &reason));
                problems.push(format!("undetermined case={} reason={reason}", case.name));
            }
        }
    }
    assert!(
        problems.is_empty(),
        "族 {family} ({arm}) に未達あり:\n{}",
        problems.join("\n")
    );
}
