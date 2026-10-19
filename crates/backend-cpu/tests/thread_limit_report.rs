//! イシュー #3178: Mac mini の CPU スレッド方針 A/B（`BIG_CORE_LIMIT_ENABLED` 反転）の
//! 「機構が実際に発火したか」の事前確認用結合テスト。
//!
//! `backend-cpu` の公開診断 API `thread_limit_report()` を呼び、`current`・
//! `detected_big_cores`・`env_override`・`effective` を 1 行の `key=value` として stderr へ出力する。
//! `scripts/bench/framework-compare/run_ab_thread_policy_macmini_cpu.sh`（`AB_GATE=big-core`）が
//! before／after 各 checkout で本テストを実行し、出力を取り込む（両腕が実質同一動作のまま
//! 「非後退」で通る偽 ADOPT を遮断する fail-closed の前提確認。判定規則の正は
//! `docs/perf/logs/cpu-thread-policy-macmini-3178/RULE.txt`）。
//!
//! 環境変数 `EXPECT_BIG_CORE_LIMIT` で検査を切り替える。
//! - 未設定: 出力のみ（Mac mini 以外の smoke・手動確認用。assert しない）。
//! - `0`: before 腕（既定 OFF）。`effective == current`（並列度が絞られていない）を要求する。
//! - `1`: after 腕（`BIG_CORE_LIMIT_ENABLED = true`）。Mac mini（P 4 + E 6 = 10 論理 CPU）で
//!   `effective == 4`（`detected_big_cores == Some(4)`・`current == 10`）を要求する。
//! - それ以外: panic で拒否する（fail-closed）。
//!
//! 注意: `effective` が絞るのは `job_grid` に渡す並列度だけで、job 自体はグローバルプール
//! （10 スレッド）上で走る。affinity もプール縮小もしない。
//!
//! 実機依存のため `#[ignore]`。after checkout では `--lib` の単体テストが落ちうるため、
//! 本テストは `--test thread_limit_report` で単独実行すること。

use fandhe_ai_backend_cpu::thread_limit_report;

/// Mac mini（M4）の P コア数と論理 CPU 総数。
const EXPECTED_BIG_CORES: usize = 4;
const EXPECTED_LOGICAL_CPUS: usize = 10;

#[test]
#[ignore = "Mac mini 実機依存（イシュー #3178）。run_ab_thread_policy_macmini_cpu.sh が --ignored で実行する"]
fn thread_limit_report_dump() {
    let r = thread_limit_report();
    eprintln!(
        "thread_limit_report current={} detected_big_cores={:?} env_override={} effective={}",
        r.current, r.detected_big_cores, r.env_override, r.effective
    );

    match std::env::var("EXPECT_BIG_CORE_LIMIT").ok().as_deref() {
        None => {}
        Some("0") => {
            assert_eq!(
                r.effective, r.current,
                "before 腕なのに並列度が絞られている（BIG_CORE_LIMIT_ENABLED が true）"
            );
        }
        Some("1") => {
            assert!(
                !r.env_override,
                "RAYON_NUM_THREADS により機構が無効化されている"
            );
            assert_eq!(
                r.detected_big_cores,
                Some(EXPECTED_BIG_CORES),
                "P コア検出が 4 個ではない"
            );
            assert_eq!(
                r.current, EXPECTED_LOGICAL_CPUS,
                "グローバルプールが 10 スレッドではない"
            );
            assert_eq!(
                r.effective, EXPECTED_BIG_CORES,
                "after 腕なのに effective が P コア数へ絞られていない（patch 未適用）"
            );
        }
        Some(other) => panic!("EXPECT_BIG_CORE_LIMIT は 0 か 1 のみ（got: {other}）"),
    }
}
