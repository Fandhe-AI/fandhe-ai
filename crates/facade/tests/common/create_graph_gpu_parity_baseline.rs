//! 二階微分（`create_graph`）の GPU parity テスト
//! （`crates/facade/tests/create_graph_gpu_parity.rs`）が baseline 方式で
//! 判定するケースの記録を置く場所。**値は 1 行も入れていない空の置き場**。
//!
//! 役割: 判定方式が `Judge::Baseline` のケース（結合順序が単一の連続 K
//! ループと異なる経路を含む族）に対し、実機実測値に基づく非後退天井を引く。
//! 呼び出し元は `create_graph_gpu_parity.rs` の `judge`。形は
//! `crates/backend-cuda/tests/common/parity_baseline.rs::ParityBaseline` に倣う
//! （`fail_count`・総要素数一致・`mean_abs_diff`／`max_abs_diff`／`max_rel_err`
//! ceiling の fail-closed 非後退検査）。
//!
//! 契約（`docs/autodiff-higher-order-grad-decision.md` §48.4・§48.5、
//! `.claude/rules/coding-rust.md`「テスト・ベンチ」）:
//! - baseline に行が無いケースを `Baseline` 方式に指定したら、黙って通さず
//!   undetermined（`Err`）として fail-closed に止める
//! - 行の追加・更新は **実機の実測値に限り、所有者の承認 URL を添えた場合だけ**
//!   行える。推定値・Linux／M4 Max 開発機の代替値は入れない
//! - `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD` はここで持たない
//!   （正本は `crates/tensor-core/src/tolerance.rs`）

/// 1 ケース分の記録済み baseline。全フィールドは実機実測値の転記に限る。
#[derive(Debug, Clone, Copy)]
pub struct GpuParityBaseline {
    /// 登録表のケース名（`CASES[i].name`）と逐語一致させる。
    pub case: &'static str,
    /// 比較した全要素数。形状がずれたら不一致として止める。
    pub total: usize,
    pub fail_count: usize,
    pub mean_abs_diff_ceiling: f64,
    pub max_abs_diff_ceiling: f64,
    pub max_rel_err_ceiling: f64,
    /// 追加を承認した所有者コメントの URL。空文字の行は無効（fail-closed）。
    pub approval_url: &'static str,
}

/// 記録済み baseline。**空**（所有者承認と実機実測が揃うまで入れない）。
pub const BASELINES: &[GpuParityBaseline] = &[];

/// `case` の baseline を引く。無い・承認 URL が空のときは undetermined。
pub fn lookup(case: &str) -> Result<&'static GpuParityBaseline, String> {
    let row = BASELINES
        .iter()
        .find(|b| b.case == case)
        .ok_or_else(|| format!("undetermined: baseline に `{case}` の行がありません"))?;
    if row.approval_url.is_empty() {
        return Err(format!(
            "undetermined: baseline `{case}` に所有者の承認 URL がありません"
        ));
    }
    Ok(row)
}

/// 実測 `report` が `row` から後退していなければ `Ok`。
pub fn check_no_regression(
    row: &GpuParityBaseline,
    report: &fandhe_ai_backend_cpu::CompareReport,
) -> Result<(), String> {
    if report.total != row.total {
        return Err(format!(
            "{}: 要素数が baseline と不一致 (actual={}, baseline={})",
            row.case, report.total, row.total
        ));
    }
    // 非有限は ceiling との比較で偽陰性にならないよう明示的に弾く。
    let finite = report.mean_abs_diff.is_finite()
        && report.max_abs_diff.is_finite()
        && report.max_rel_err.is_finite();
    if !finite
        || report.fail_count > row.fail_count
        || report.mean_abs_diff > row.mean_abs_diff_ceiling
        || report.max_abs_diff > row.max_abs_diff_ceiling
        || report.max_rel_err > row.max_rel_err_ceiling
    {
        return Err(format!(
            "{}: baseline から後退 (fail_count={}/{}, mean_abs_diff={:.6e}, \
             max_abs_diff={:.6e}, max_rel_err={:.6e})",
            row.case,
            report.fail_count,
            report.total,
            report.mean_abs_diff,
            report.max_abs_diff,
            report.max_rel_err
        ));
    }
    Ok(())
}
