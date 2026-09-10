//! Metal f32 split-K 経路の parity 非後退契約（`common::splitk_parity_
//! baseline`。イシュー #1512）の Linux 実行可能な型検査・falsification
//! テスト。
//!
//! `tests/gemm_splitk_parity.rs`（判定の実利用側）は `#![cfg(target_os =
//! "macos")]` かつ `required-features = ["internal-diagnostics"]` のため
//! Linux CI（GitHub ホステッド・ubuntu-latest）ではコンパイルすら
//! されず、`common::splitk_parity_baseline` モジュール自体の型検査
//! カバレッジがゼロだった（イシュー #1512 の実装計画で確認した事実）。
//! 本ファイルは macOS cfg・`required-features` のいずれも持たず、通常の
//! `cargo test -p fandhe-ai-backend-metal` で実行されることで、
//! - (a) `BASELINES` が承認済み 11 形状ちょうどであり、各行の実測値
//!   （`docs/backend-metal-splitk-parity-judgment-decision.md` §7 の表・
//!   2026-09-10 ユーザー承認）から機械的に転記した ceiling と一致する
//!   こと（黙った改竄・緩和の検出。`crates/backend-cuda/tests/
//!   parity_nonregression.rs::baseline_fixture_is_self_consistent` と
//!   同方針）
//! - (b) `find_baseline` が承認済み 11 形状すべてで `Some` を返し、
//!   未登録形状では `None` を返すこと
//! - (c) `assert_no_split_k_parity_regression` が `CUDA` 側
//!   `parity_nonregression.rs` と同型の 5 種 falsification
//!   （total 不一致・fail_count／mean_abs_diff／max_abs_diff／
//!   max_rel_err の後退）でそれぞれ確実に panic すること（「常に pass
//!   する」壊れ方の防止）
//! - (d) ceiling ちょうど以下の合成 report は pass すること（正例。
//!   境界値そのものを後退と誤検出しないことの確認）
//!   を検証する。`BASELINES` の数値自体は一切変更しない。

mod common;

use common::splitk_parity_baseline::{
    BASELINES, assert_no_split_k_parity_regression, find_baseline,
};
use fandhe_ai_backend_cpu::parity::{CompareReport, compare};

/// 承認記録（`docs/backend-metal-splitk-parity-judgment-decision.md` §7
/// の表・2026-09-10 ユーザー承認）からの転記。`(m, n, k, total,
/// fail_count, mean_abs_diff_ceiling, max_abs_diff_ceiling,
/// max_rel_err_ceiling)`。`BASELINES` と全一致することを (a) で検査する
/// （`BASELINES` 側の改竄・緩和を fail-closed に検出する独立の転記元）。
/// `(m, n, k, total, fail_count, mean_abs_diff_ceiling, max_abs_diff_ceiling,
/// max_rel_err_ceiling)`。clippy::type_complexity 回避のための factoring
/// （`common::splitk_parity_baseline::Baseline` とは独立の転記元用タプル型）。
type ApprovedBaselineRow = (usize, usize, usize, usize, usize, f64, f64, f64);

const APPROVED_BASELINES: &[ApprovedBaselineRow] = &[
    (32, 32, 2048, 1024, 0, 1.0e-5, 9.0e-5, 3.0e-4),
    (32, 32, 4096, 1024, 0, 2.0e-5, 3.0e-4, 2.0e-4),
    (32, 32, 8192, 1024, 1, 4.0e-5, 4.0e-4, 1.2e-3),
    (64, 64, 2048, 4096, 2, 1.0e-5, 1.0e-4, 4.0e-3),
    (64, 64, 4096, 4096, 0, 2.0e-5, 3.0e-4, 8.0e-4),
    (64, 64, 8192, 4096, 2, 4.0e-5, 4.0e-4, 2.3e-3),
    (128, 128, 2048, 16384, 2, 1.0e-5, 1.2e-4, 3.4e-3),
    (128, 128, 4096, 16384, 8, 2.0e-5, 1.7e-4, 8.6e-3),
    (128, 128, 8192, 16384, 2, 4.0e-5, 4.0e-4, 1.7e-3),
    (64, 64, 2056, 4096, 1, 1.0e-5, 1.3e-4, 1.3e-3),
    (128, 128, 2064, 16384, 2, 1.0e-5, 1.3e-4, 0.14),
];

/// (a) `BASELINES` が承認済み 11 形状ちょうどであり、承認記録からの
/// 独立転記（`APPROVED_BASELINES`）と全一致すること。
#[test]
fn baselines_match_approved_ceiling_table() {
    assert_eq!(
        BASELINES.len(),
        APPROVED_BASELINES.len(),
        "BASELINES の行数が承認済み 11 形状と一致しない（承認記録: \
         docs/backend-metal-splitk-parity-judgment-decision.md §7）"
    );

    for &(m, n, k, total, fail_count, mean_ceil, max_abs_ceil, max_rel_ceil) in APPROVED_BASELINES {
        let b = find_baseline(m, n, k).unwrap_or_else(|| {
            panic!("承認済み形状 (m={m}, n={n}, k={k}) が BASELINES に見つからない")
        });
        assert_eq!(
            b.total, total,
            "(m={m}, n={n}, k={k}): total が承認値と不一致"
        );
        assert_eq!(
            b.baseline_fail_count, fail_count,
            "(m={m}, n={n}, k={k}): fail_count が承認値と不一致（緩和・改竄の疑い）"
        );
        assert_eq!(
            b.baseline_mean_abs_diff_ceiling, mean_ceil,
            "(m={m}, n={n}, k={k}): mean_abs_diff_ceiling が承認値と不一致"
        );
        assert_eq!(
            b.baseline_max_abs_diff_ceiling, max_abs_ceil,
            "(m={m}, n={n}, k={k}): max_abs_diff_ceiling が承認値と不一致"
        );
        assert_eq!(
            b.baseline_max_rel_err_ceiling, max_rel_ceil,
            "(m={m}, n={n}, k={k}): max_rel_err_ceiling が承認値と不一致"
        );
    }
}

/// fixture 自体の妥当性検査: 各行 `total == m*n`・`fail_count <= total`・
/// 各 ceiling が有限の非負値であること
/// （`crates/backend-cuda/tests/parity_nonregression.rs::
/// baseline_fixture_is_self_consistent` と同方針）。
#[test]
fn baseline_fixture_is_self_consistent() {
    assert!(!BASELINES.is_empty(), "BASELINES must not be empty");
    for b in BASELINES {
        let expected_total = b.m * b.n;
        assert_eq!(
            b.total, expected_total,
            "(m={}, n={}, k={}): total は m*n と一致する必要がある（total={}, m*n={}）",
            b.m, b.n, b.k, b.total, expected_total
        );
        assert!(
            b.baseline_fail_count <= b.total,
            "(m={}, n={}, k={}): baseline_fail_count({}) が total({}) を超えている",
            b.m,
            b.n,
            b.k,
            b.baseline_fail_count,
            b.total
        );
        for (name, v) in [
            ("mean_abs_diff_ceiling", b.baseline_mean_abs_diff_ceiling),
            ("max_abs_diff_ceiling", b.baseline_max_abs_diff_ceiling),
            ("max_rel_err_ceiling", b.baseline_max_rel_err_ceiling),
        ] {
            assert!(
                v >= 0.0 && v.is_finite(),
                "(m={}, n={}, k={}): {name} は有限の非負値である必要がある（値={v}）",
                b.m,
                b.n,
                b.k
            );
        }
    }
}

/// (b) `find_baseline` が承認済み 11 形状すべてで `Some` を返すこと。
#[test]
fn find_baseline_resolves_all_approved_shapes() {
    for &(m, n, k, ..) in APPROVED_BASELINES {
        assert!(
            find_baseline(m, n, k).is_some(),
            "承認済み形状 (m={m}, n={n}, k={k}) の find_baseline が None を返した"
        );
    }
}

/// (b) 未登録形状では `find_baseline` が `None` を返すこと（fail-open で
/// 素通りさせない契約の型検査）。
#[test]
fn find_baseline_returns_none_for_unregistered_shape() {
    assert!(find_baseline(3, 5, 7).is_none());
}

/// テスト用の合成 `CompareReport` を作る。`compare` を経由して構築する
/// ことで、実運用（`gemm_splitk_parity.rs`）と同じ経路の値を使う。
fn synthetic_report(a: &[f32], b: &[f32]) -> CompareReport {
    compare(a, b).expect("length must match")
}

/// (d) 正例: ceiling ちょうど以下の合成 report は pass すること
/// （`BASELINES` の最初の行 `(32,32,2048)` を使い、all-zero 同士の比較
/// つまり fail_count=0・全指標 0.0 という「余裕を持って ceiling 以下」の
/// 報告で panic しないことを確認する）。
#[test]
fn assert_no_split_k_parity_regression_passes_on_zero_diff_report() {
    let baseline = find_baseline(32, 32, 2048).expect("承認済み形状");
    let a = vec![0.0f32; baseline.total];
    let b = vec![0.0f32; baseline.total];
    let report = synthetic_report(&a, &b);
    assert_no_split_k_parity_regression("synthetic zero diff", &report, baseline);
}

/// (c) fail-closed 契約の falsification テスト（5 種）。`crates/
/// backend-cuda/tests/parity_nonregression.rs` の
/// `assert_no_parity_regression_panics_on_*` 群と同方針: 「常に pass する」
/// 壊れ方をしていないことを固定する。合成 baseline を使い、対象の 1 指標
/// のみをベースライン超過にする（他指標は余裕を持たせる）。

#[test]
#[should_panic(expected = "後退しています")]
fn assert_no_split_k_parity_regression_panics_on_fail_count_regression() {
    let baseline = common::splitk_parity_baseline::SplitKParityBaseline {
        m: 4,
        n: 4,
        k: 4,
        total: 16,
        baseline_fail_count: 2,
        baseline_mean_abs_diff_ceiling: 1.0,
        baseline_max_abs_diff_ceiling: 1.0,
        baseline_max_rel_err_ceiling: 1.0,
    };
    let a = vec![0.0f32; baseline.total];
    let mut b = vec![0.0f32; baseline.total];
    // 3 セルを大きく乖離させ、fail_count=3 > baseline_fail_count(2) にする
    // （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満のいずれも満たさない値）。
    b[0] = 1.0;
    b[1] = 1.0;
    b[2] = 1.0;
    let report = synthetic_report(&a, &b);
    assert_no_split_k_parity_regression("synthetic fail_count", &report, &baseline);
}

#[test]
#[should_panic(expected = "後退しています")]
fn assert_no_split_k_parity_regression_panics_on_mean_abs_diff_regression() {
    let baseline = common::splitk_parity_baseline::SplitKParityBaseline {
        m: 2,
        n: 2,
        k: 2,
        total: 4,
        // fail_count は緩く設定し、mean_abs_diff 側のみで fail させる。
        baseline_fail_count: 4,
        baseline_mean_abs_diff_ceiling: 1e-6,
        baseline_max_abs_diff_ceiling: 1.0,
        baseline_max_rel_err_ceiling: 1.0,
    };
    let a = vec![0.0f32; baseline.total];
    // 絶対誤差救済閾値(1e-5)未満のため複合判定は pass するが、
    // mean_abs_diff(約 5e-6) は baseline_mean_abs_diff_ceiling(1e-6) を上回る。
    let b = vec![5e-6f32; baseline.total];
    let report = synthetic_report(&a, &b);
    assert_no_split_k_parity_regression("synthetic mean_abs_diff", &report, &baseline);
}

#[test]
#[should_panic(expected = "一致しません")]
fn assert_no_split_k_parity_regression_panics_on_total_mismatch() {
    let baseline = common::splitk_parity_baseline::SplitKParityBaseline {
        m: 4,
        n: 4,
        k: 4,
        total: 16,
        baseline_fail_count: 100,
        baseline_mean_abs_diff_ceiling: 1.0,
        baseline_max_abs_diff_ceiling: 1.0,
        baseline_max_rel_err_ceiling: 1.0,
    };
    // total が baseline(16) と異なる合成レポート。
    let a = vec![0.0f32; 9];
    let b = vec![0.0f32; 9];
    let report = synthetic_report(&a, &b);
    assert_no_split_k_parity_regression("synthetic total mismatch", &report, &baseline);
}

#[test]
#[should_panic(expected = "後退しています")]
fn assert_no_split_k_parity_regression_panics_on_max_abs_diff_regression() {
    let baseline = common::splitk_parity_baseline::SplitKParityBaseline {
        m: 2,
        n: 2,
        k: 2,
        total: 4,
        // fail_count・mean_abs_diff は緩く設定し、max_abs_diff 側のみで
        // fail させる。
        baseline_fail_count: 4,
        baseline_mean_abs_diff_ceiling: 1.0,
        baseline_max_abs_diff_ceiling: 1e-7,
        baseline_max_rel_err_ceiling: 1.0,
    };
    let a = vec![0.0f32; baseline.total];
    // abs_diff=5e-6 は絶対誤差救済閾値(1e-5)未満のため複合判定は pass
    // する（fail_count は増えない）が、max_abs_diff(約 5e-6) は
    // baseline_max_abs_diff_ceiling(1e-7) を上回る。
    let b = vec![5e-6f32; baseline.total];
    let report = synthetic_report(&a, &b);
    assert_no_split_k_parity_regression("synthetic max_abs_diff", &report, &baseline);
}

#[test]
#[should_panic(expected = "後退しています")]
fn assert_no_split_k_parity_regression_panics_on_max_rel_err_regression() {
    let baseline = common::splitk_parity_baseline::SplitKParityBaseline {
        m: 2,
        n: 2,
        k: 2,
        total: 4,
        baseline_fail_count: 4,
        baseline_mean_abs_diff_ceiling: 1.0,
        // abs_diff(1e-6) は緩い上限のため通過するが、rel_err(0.5) は
        // baseline_max_rel_err_ceiling(0.1) を上回る。
        baseline_max_abs_diff_ceiling: 1.0,
        baseline_max_rel_err_ceiling: 0.1,
    };
    let a = vec![1e-6f32; baseline.total];
    // abs_diff=1e-6 は絶対誤差救済閾値(1e-5)未満のため複合判定は pass
    // する（fail_count は増えない）が、rel_err = diff/max(|x|,|y|,1e-12)
    // = 1e-6/2e-6 = 0.5 は baseline_max_rel_err_ceiling(0.1) を上回る。
    let b = vec![2e-6f32; baseline.total];
    let report = synthetic_report(&a, &b);
    assert_no_split_k_parity_regression("synthetic max_rel_err", &report, &baseline);
}

/// イシュー #1513: `SPLIT_K_NUMERIC_CONTRACT_APPROVED` を `true` へ切り替え
/// 済みのため、`MetalGemm::dispatch_split_k_strided_prepared`（自動判定
/// 入口）は `crate::tile::should_split_k` が `Some` を返す形状で実際に
/// split-K 経路（`SplitKRoute::Split`）を実行するようになった。この
/// 前提が崩れていないこと（承認済み `BASELINES` 11 形状 ⊂
/// `should_split_k` の `Some` 集合であること）を Linux で機械的に固定
/// する。macOS 実機での公開入口自体の到達確認（`SplitKRoute::Split` を
/// 実際に返すこと）は `tests/gemm_splitk_auto_entry_parity.rs`（`#[ignore]`）
/// が担う。
#[test]
fn approved_baseline_shapes_are_split_k_eligible() {
    for b in BASELINES {
        let plan = fandhe_ai_backend_metal::should_split_k(b.m, b.n, b.k).unwrap_or_else(|| {
            panic!(
                "承認済み形状 (m={}, n={}, k={}) で should_split_k が None を返した \
                     （BASELINES と自動判定条件の前提が崩れている。baseline 行の追加・\
                     `should_split_k` の判定条件変更は実機実測とセットでユーザー承認が必要）",
                b.m, b.n, b.k
            )
        });
        assert!(
            plan.partitions >= 2,
            "(m={}, n={}, k={}): should_split_k が返した partitions({}) が 2 未満（split-K \
             として機能する分割数の前提が崩れている）",
            b.m,
            b.n,
            b.k,
            plan.partitions
        );
    }
}
