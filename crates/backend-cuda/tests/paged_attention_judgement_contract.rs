//! paged attention 実機 parity の判定構成（`common::paged_attention_judgement`。イシュー #3113）の
//! デバイス不要な契約テスト。
//!
//! 実機テスト（`paged_kv_real_device.rs`）は `#[ignore]` のため CI では動かない。本ファイルは
//! cfg も `#[ignore]` も持たず、通常の `cargo test` で次を固定する（`splitk_parity_baseline_contract.rs`
//! と同型）:
//! - 形状マトリクス `CASES` の一意性と、各ケースが `PagedKvLayout`／`PagedAttentionArgs` の検証を通ること
//! - `BASELINES` が現時点で空であること（行を追加する PR はここも変えるため、レビューで
//!   「実機実測値のみ・人間承認」を確認できる）と、行が入った場合の自己整合
//! - 既定が `Strict`、`judge` の falsification（5 種）と正例、ログ 1 行の書式
//! - tolerance 定数の bit 等値固定

mod common;

use common::paged_attention_judgement::{
    BASELINES, CASES, Judgement, PagedParityBaseline, judge, judgement_for, metric_line,
    pool_blocks, step_output_elems,
};
use fandhe_ai_backend_cpu::parity::{CompareReport, compare};
use fandhe_ai_tensor_core::{PagedAttentionArgs, PagedKvLayout};

#[test]
fn case_ids_are_unique_and_shapes_validate() {
    let mut seen = std::collections::HashSet::new();
    for c in CASES {
        assert!(seen.insert(c.id), "case id 重複: {}", c.id);
        assert!(c.embed % c.heads == 0, "{}: embed % heads", c.id);
        assert!(c.steps.iter().all(|&l| l > 0), "{}: L_new > 0", c.id);
        let layout = PagedKvLayout::new(pool_blocks(c), c.block_size, c.embed)
            .unwrap_or_else(|e| panic!("{}: layout {e:?}", c.id));
        assert!(layout.num_blocks <= 512, "{}: プールを小さく保つ", c.id);
        let mut seq = 0usize;
        for (si, &l) in c.steps.iter().enumerate() {
            seq += l;
            let mbps = seq.div_ceil(c.block_size);
            PagedAttentionArgs::new(layout, c.batch, l, c.heads, mbps, 0.5)
                .unwrap_or_else(|e| panic!("{} step {si}: args {e:?}", c.id));
            if let Some((after, new_len)) = c.rewind.filter(|&(a, _)| a == si) {
                assert!(
                    after < c.steps.len() - 1,
                    "{}: 巻き戻し後に再追記が要る",
                    c.id
                );
                assert!(new_len < seq, "{}: 巻き戻しは短縮のみ", c.id);
                seq = new_len;
            }
        }
        if let Some((after, _)) = c.rewind {
            assert!(after < c.steps.len(), "{}: rewind の step 添字", c.id);
        }
    }
}

#[test]
fn matrix_covers_thread_stride_and_block_boundaries() {
    // カーネルのスレッド数 T=128 を跨ぐ形状（j 軸・d 軸）が欠けると実機の取りこぼしになる。
    assert!(CASES.iter().any(|c| c.steps.iter().sum::<usize>() > 128));
    assert!(CASES.iter().any(|c| c.embed / c.heads > 128));
    assert!(CASES.iter().any(|c| c.embed == c.heads));
    assert!(CASES.iter().any(|c| c.rewind.is_some()));
    assert!(CASES.iter().any(|c| c.input_scale > 1.0));
    for bs in [1usize, 2, 16, 256] {
        assert!(CASES.iter().any(|c| c.block_size == bs), "bs={bs}");
    }
}

#[test]
fn baselines_are_empty_until_measured_and_approved() {
    // 実機実測値と人間承認なしに baseline を入れない。行を追加する PR はこの assert を変える。
    assert!(BASELINES.is_empty());
    let mut seen = std::collections::HashSet::new();
    for b in BASELINES {
        let c = CASES
            .iter()
            .find(|c| c.id == b.case_id)
            .unwrap_or_else(|| panic!("未知の case_id: {}", b.case_id));
        assert!(b.step < c.steps.len());
        assert!(seen.insert((b.case_id, b.step)), "baseline 重複");
        assert_eq!(b.total, step_output_elems(c, b.step));
        for v in [
            b.baseline_mean_abs_diff_ceiling,
            b.baseline_max_abs_diff_ceiling,
            b.baseline_max_rel_err_ceiling,
        ] {
            assert!(v.is_finite() && v >= 0.0);
        }
        assert!(!b.measured_on.is_empty() && !b.log_path.is_empty() && !b.approval.is_empty());
    }
}

#[test]
fn unregistered_case_defaults_to_strict() {
    for c in CASES {
        for si in 0..c.steps.len() {
            if BASELINES
                .iter()
                .all(|b| !(b.case_id == c.id && b.step == si))
            {
                assert!(matches!(judgement_for(c.id, si), Judgement::Strict));
            }
        }
    }
    assert!(matches!(
        judgement_for("no-such-case", 0),
        Judgement::Strict
    ));
}

fn failing_report() -> CompareReport {
    let a = [0.0f32, 0.0, 0.0, 0.0];
    let b = [1.0f32, 0.0, 0.0, 0.0];
    compare(&a, &b).unwrap()
}

#[test]
fn strict_rejects_any_fail_and_accepts_zero() {
    let bad = failing_report();
    assert_eq!(bad.fail_count, 1);
    assert!(judge("t", &bad, Judgement::Strict).is_err());
    let ok = compare(&[1.0f32, 2.0], &[1.0f32, 2.0]).unwrap();
    assert!(judge("t", &ok, Judgement::Strict).is_ok());
}

fn baseline_from(r: &CompareReport) -> PagedParityBaseline {
    PagedParityBaseline {
        case_id: "synthetic",
        step: 0,
        total: r.total,
        baseline_fail_count: r.fail_count,
        baseline_mean_abs_diff_ceiling: r.mean_abs_diff,
        baseline_max_abs_diff_ceiling: r.max_abs_diff,
        baseline_max_rel_err_ceiling: r.max_rel_err,
        measured_on: "synthetic",
        log_path: "synthetic",
        approval: "synthetic",
    }
}

#[test]
fn baseline_judgement_falsification_and_boundary() {
    let r = failing_report();
    assert!(r.mean_abs_diff > 0.0 && r.max_abs_diff > 0.0 && r.max_rel_err > 0.0);
    // 正例: ceiling ちょうどは後退ではない
    let base = baseline_from(&r);
    assert!(judge("t", &r, Judgement::Baseline(&base)).is_ok());
    // falsification 5 種
    let mut b = base;
    b.total += 1;
    assert!(judge("t", &r, Judgement::Baseline(&b)).is_err(), "total");
    let mut b = base;
    b.baseline_fail_count = 0;
    assert!(
        judge("t", &r, Judgement::Baseline(&b)).is_err(),
        "fail_count"
    );
    let mut b = base;
    b.baseline_mean_abs_diff_ceiling /= 2.0;
    assert!(judge("t", &r, Judgement::Baseline(&b)).is_err(), "mean_abs");
    let mut b = base;
    b.baseline_max_abs_diff_ceiling /= 2.0;
    assert!(judge("t", &r, Judgement::Baseline(&b)).is_err(), "max_abs");
    let mut b = base;
    b.baseline_max_rel_err_ceiling /= 2.0;
    assert!(judge("t", &r, Judgement::Baseline(&b)).is_err(), "max_rel");
}

#[test]
fn metric_line_format_is_pinned() {
    let r = compare(&[1.0f32, 2.0], &[1.0f32, 2.0]).unwrap();
    assert_eq!(
        metric_line("cuda", "case-x", 2, &r, Judgement::Strict),
        "PAGED_PARITY backend=cuda case=case-x step=2 total=2 fail=0 max_abs=0e0 max_rel=0e0 mean_abs=0e0 judgement=strict"
    );
}

#[test]
fn tolerance_constants_are_pinned() {
    common::parity_baseline::assert_tolerance_constants_pinned();
}
