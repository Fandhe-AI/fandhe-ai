//! paged attention（K-3。イシュー #3113・親 #3108）の実機 parity テストが使う「形状マトリクス」と
//! 「判定方式の選択」の共有フィクスチャ。
//!
//! 役割: `paged_kv_real_device.rs`（実機 `#[ignore]` テスト。呼び出し元）へ形状マトリクス
//! [`CASES`] と判定 [`judge`]・ログ 1 行 [`metric_line`] を供給し、
//! `paged_attention_judgement_contract.rs`（CI で動く契約テスト）が本モジュールの自己整合を固定する。
//! `crates/backend-metal/tests/common/paged_attention_judgement.rs` と内容は同一（baseline は
//! ハードウェアごとに別の実測値になるため、先例 `parity_baseline.rs`／`splitk_parity_baseline.rs`
//! どおりクレートごとに 1 ファイル置く）。
//!
//! 判定方針（設計 `docs/device-resident-kv-paged-attention-design.md` §5・coding-rust.md「テスト・ベンチ」）:
//! - 既定は統一複合判定（`fandhe_ai_backend_cpu::parity::compare`）のゼロ fail（[`Judgement::Strict`]）。
//!   合否式は複製せず `CompareReport` を正本とする。tolerance 定数は本ファイルで変更しない。
//! - 実機で fail が出た形状だけを [`BASELINES`] へ登録して baseline 非後退判定へ切り替えられる。
//!   登録は**実機実測値のみ・人間承認必須**で、本 PR 時点の登録は 0 件（未実測。実測は #3171・#3172）。
#![allow(dead_code)] // 使う項目がテストバイナリごとに異なる（契約テストと実機テスト）ため

use fandhe_ai_backend_cpu::parity::CompareReport;

/// 形状マトリクスの 1 ケース。`id` は一意で安定（ログ集計と baseline の参照キー）。
#[derive(Debug, Clone, Copy)]
pub struct PagedAttentionCase {
    pub id: &'static str,
    pub block_size: usize,
    pub batch: usize,
    pub embed: usize,
    pub heads: usize,
    /// 各ステップの `L_new`（最初が prefill・以降 decode／複数トークン追記）。
    pub steps: &'static [usize],
    /// K/V/Q 入力の振幅倍率（`expf` と `f32::exp` の差が softmax へ出るかを見る用）。
    pub input_scale: f32,
    /// `(step 後の添字, 巻き戻し先の長さ)`。巻き戻し後に物理ブロックが論理順と一致しなくなる。
    pub rewind: Option<(usize, usize)>,
}

const fn case(
    id: &'static str,
    block_size: usize,
    batch: usize,
    embed: usize,
    heads: usize,
    steps: &'static [usize],
) -> PagedAttentionCase {
    PagedAttentionCase {
        id,
        block_size,
        batch,
        embed,
        heads,
        steps,
        input_scale: 1.0,
        rewind: None,
    }
}

/// 形状マトリクス。カーネルのスレッド数 T=128 を跨ぐ形状（s_max>128・Dh>128）、ブロック境界 ±1、
/// 複数トークン追記、巻き戻し後の再追記、大振幅入力を含む。プールと作業バッファは小さく保つ。
pub static CASES: &[PagedAttentionCase] = &[
    // #3111 の既存 4 形状
    case("legacy_bs1_b1", 1, 1, 8, 2, &[3, 1, 1]),
    case("legacy_bs4_b2", 4, 2, 16, 2, &[5, 1, 1, 4]),
    case("legacy_bs16_b1", 16, 1, 64, 4, &[17, 1, 1]),
    case("legacy_bs16_b3", 16, 3, 48, 3, &[16, 1, 15]),
    // ブロック境界（bs-1／bs／bs+1 の prefill → decode 2 回）
    case("bs1_prefill1", 1, 2, 16, 2, &[1, 1, 1]),
    case("bs1_prefill2", 1, 2, 16, 2, &[2, 1, 1]),
    case("bs2_prefill1", 2, 2, 16, 2, &[1, 1, 1]),
    case("bs2_prefill2", 2, 2, 16, 2, &[2, 1, 1]),
    case("bs2_prefill3", 2, 2, 16, 2, &[3, 1, 1]),
    case("bs16_prefill15", 16, 2, 16, 2, &[15, 1, 1]),
    case("bs16_prefill16", 16, 2, 16, 2, &[16, 1, 1]),
    case("bs16_prefill17", 16, 2, 16, 2, &[17, 1, 1]),
    case("bs256_prefill255", 256, 2, 16, 2, &[255, 1, 1]),
    case("bs256_prefill256", 256, 2, 16, 2, &[256, 1, 1]),
    case("bs256_prefill257", 256, 2, 16, 2, &[257, 1, 1]),
    // 複数ブロック・s_max>128（j 軸のスレッドストライド）
    case("long_bs16_s301", 16, 2, 32, 2, &[100, 100, 100, 1]),
    // Dh>128（d 軸のストライド）・Dh=1
    case("dh160_h1", 16, 1, 160, 1, &[5, 1, 1]),
    case("dh1_e4_h4", 4, 2, 4, 4, &[6, 1, 1]),
    // L_new>1 の複数回追記（因果オフセット j <= s_prev + i）
    case("multi_prefill_bs4", 4, 2, 16, 2, &[3, 4, 5, 1]),
    // 巻き戻し後の再追記（ブロック途中・ブロック境界ちょうど）
    PagedAttentionCase {
        rewind: Some((1, 6)),
        ..case("rewind_mid_block", 4, 2, 16, 2, &[9, 1, 5])
    },
    PagedAttentionCase {
        rewind: Some((0, 4)),
        ..case("rewind_to_boundary", 4, 2, 16, 2, &[9, 2])
    },
    // 大振幅入力
    PagedAttentionCase {
        input_scale: 4.0,
        ..case("large_amplitude", 16, 2, 32, 2, &[17, 1, 1])
    },
];

/// ケースの `step` 番目の出力要素数（`B * L_new * E`）。
pub fn step_output_elems(c: &PagedAttentionCase, step: usize) -> usize {
    c.batch * c.steps[step] * c.embed
}

/// ケースが必要とする物理ブロック数（巻き戻しの再利用を見込まず安全側に多め）。
pub fn pool_blocks(c: &PagedAttentionCase) -> usize {
    let total: usize = c.steps.iter().sum();
    c.batch * total.div_ceil(c.block_size) + 2
}

/// baseline 非後退判定の 1 行（ケース × ステップ）。**実機実測値のみ・人間承認必須**。
/// tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）とは別の概念で、
/// 判定式は変えず「実測で確定した既知の劣化を後退させない」ための ceiling。
/// 全 ceiling は必須（coding-rust.md の fail-closed 非後退検査）。
#[derive(Debug, Clone, Copy)]
pub struct PagedParityBaseline {
    pub case_id: &'static str,
    pub step: usize,
    pub total: usize,
    pub baseline_fail_count: usize,
    pub baseline_mean_abs_diff_ceiling: f64,
    pub baseline_max_abs_diff_ceiling: f64,
    pub baseline_max_rel_err_ceiling: f64,
    /// 実測環境（例 "DGX Spark GB10"）。
    pub measured_on: &'static str,
    /// 実測ログのパス（`docs/perf/logs/paged-attention-parity-3113/` 配下）。
    pub log_path: &'static str,
    /// 人間承認の参照（Issue／PR コメント URL）。
    pub approval: &'static str,
}

/// 登録済み baseline。**空で開始**する（本 PR は未実測。追加は実測値＋人間承認の別 PR）。
pub static BASELINES: &[PagedParityBaseline] = &[];

#[derive(Debug, Clone, Copy)]
pub enum Judgement<'a> {
    Strict,
    Baseline(&'a PagedParityBaseline),
}

/// 表に無ければ `Strict`（fail-closed）。
pub fn judgement_for(case_id: &str, step: usize) -> Judgement<'static> {
    BASELINES
        .iter()
        .find(|b| b.case_id == case_id && b.step == step)
        .map_or(Judgement::Strict, Judgement::Baseline)
}

/// 合否判定。fail の集約は呼び出し側（1 回の実行で全ケースの指標をログに残すため）。
pub fn judge(context: &str, report: &CompareReport, j: Judgement<'_>) -> Result<(), String> {
    match j {
        Judgement::Strict => {
            if report.fail_count == 0 {
                Ok(())
            } else {
                Err(format!(
                    "{context}: strict 判定で {} / {} 要素が fail (max_abs={:e}, max_rel={:e})",
                    report.fail_count, report.total, report.max_abs_diff, report.max_rel_err
                ))
            }
        }
        Judgement::Baseline(b) => {
            let mut errs = Vec::new();
            if report.total != b.total {
                errs.push(format!("total {} != baseline {}", report.total, b.total));
            }
            if report.fail_count > b.baseline_fail_count {
                errs.push(format!(
                    "fail_count {} > baseline {}",
                    report.fail_count, b.baseline_fail_count
                ));
            }
            // NaN は `<=` が偽になるため否定形で比較し fail-closed にする。
            for (name, got, ceil) in [
                (
                    "mean_abs_diff",
                    report.mean_abs_diff,
                    b.baseline_mean_abs_diff_ceiling,
                ),
                (
                    "max_abs_diff",
                    report.max_abs_diff,
                    b.baseline_max_abs_diff_ceiling,
                ),
                (
                    "max_rel_err",
                    report.max_rel_err,
                    b.baseline_max_rel_err_ceiling,
                ),
            ] {
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(got <= ceil) {
                    errs.push(format!("{name} {got:e} > ceiling {ceil:e}"));
                }
            }
            if errs.is_empty() {
                Ok(())
            } else {
                Err(format!(
                    "{context}: baseline 非後退違反: {}",
                    errs.join("; ")
                ))
            }
        }
    }
}

/// Phase 8（#3171・#3172）のログ集計用 1 行。書式は README に固定している。
pub fn metric_line(
    backend: &str,
    case_id: &str,
    step: usize,
    report: &CompareReport,
    j: Judgement<'_>,
) -> String {
    let kind = match j {
        Judgement::Strict => "strict",
        Judgement::Baseline(_) => "baseline",
    };
    format!(
        "PAGED_PARITY backend={backend} case={case_id} step={step} total={} fail={} max_abs={:e} max_rel={:e} mean_abs={:e} judgement={kind}",
        report.total,
        report.fail_count,
        report.max_abs_diff,
        report.max_rel_err,
        report.mean_abs_diff
    )
}
