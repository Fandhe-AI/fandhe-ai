//! イシュー #2117: GB10 大コア affinity A/B の「機構が実際に発火したか」の事前確認用結合テスト。
//!
//! `backend-cpu` の公開診断 API `gb10_affinity_report()` を呼び、`enabled`・`env_override`・
//! `detected_big_core_ids`・`pool_active` を 1 行の `key=value` として stderr へ出力する。
//! `scripts/bench/framework-compare/run_ab_gb10_affinity_cpu.sh` が before／after 各 checkout で
//! 本テストを実行し、出力を grep で取り込む（両腕が実質同一バイナリのまま「非後退」で通過する
//! 偽 ADOPT を遮断するための fail-closed 前提確認。判定規則の正は
//! `docs/perf/logs/cpu-gb10-affinity-ab-2117/RULE.txt`）。
//!
//! 環境変数 `EXPECT_GB10_AFFINITY_POOL_ACTIVE` で検査を切り替える。
//! - 未設定: 出力のみ（x86 smoke・手動確認用。assert しない）。
//! - `1`: after 腕（`GB10_AFFINITY_ENABLED = true`）。専用プールが実際に有効で、大コア 10 個を
//!   検出していることを要求する。
//! - `0`: before 腕（既定 OFF）。専用プールが不活性であることを要求する。
//! - それ以外: panic で拒否する（fail-closed）。
//!
//! 実機（GB10）依存のため `#[ignore]`。after checkout では `--lib` の単体テスト
//! （`const { assert!(!GB10_AFFINITY_ENABLED) }` を持つ）が落ちるため、本テストは
//! `--test gb10_affinity_report` で単独実行すること。

use fandhe_ai_backend_cpu::gb10_affinity_report;

/// GB10（Cortex-X925 x10 + A725 x10）の大コア数。
const EXPECTED_BIG_CORES: usize = 10;

#[test]
#[ignore = "GB10 実機依存（イシュー #2117）。run_ab_gb10_affinity_cpu.sh が --ignored で実行する"]
fn gb10_affinity_report_dump() {
    let r = gb10_affinity_report();
    let ids = match &r.detected_big_core_ids {
        Some(v) => format!("{v:?}").replace(' ', ""),
        None => "None".to_string(),
    };
    eprintln!(
        "gb10_affinity_report enabled={} env_override={} detected_big_core_ids={} pool_active={}",
        r.enabled, r.env_override, ids, r.pool_active
    );

    match std::env::var("EXPECT_GB10_AFFINITY_POOL_ACTIVE")
        .ok()
        .as_deref()
    {
        None => {}
        Some("1") => {
            assert!(
                r.enabled,
                "after 腕なのに enabled=false（on-arm.patch 未適用）"
            );
            assert!(
                !r.env_override,
                "RAYON_NUM_THREADS により機構が無効化されている"
            );
            assert_eq!(
                r.detected_big_core_ids.as_ref().map(Vec::len),
                Some(EXPECTED_BIG_CORES),
                "大コア検出が 10 個ではない"
            );
            assert!(r.pool_active, "専用 affinity プールが有効化されていない");
        }
        Some("0") => {
            assert!(!r.pool_active, "before 腕なのに専用プールが有効");
        }
        Some(other) => panic!("EXPECT_GB10_AFFINITY_POOL_ACTIVE は 0/1 のみ（got: {other}）"),
    }
}
