//! イシュー #3178: Mac mini の CPU スレッド方針 A/B（`SMALL_SHAPE_CAP_ENABLED` 反転）の
//! 「機構が実際に発火したか」の事前確認用結合テスト。
//!
//! `small_shape_cap_report()` は専用プールを生成せず `POOL.get()` を読むだけなので、`pool_active`
//! は cap が実際に発火した GEMM を 1 回実行した後でないと true にならない。そのため本テストは
//! 先に cap 対象の学習形状（64x256x784 = 12,845,056 < 2^24）を公開 GEMM 入口
//! `gemm_blis_parallel` で 1 回実行してから report を 1 行で stderr へ出す。
//! `scripts/bench/framework-compare/run_ab_thread_policy_macmini_cpu.sh`（`AB_GATE=small-cap`）が
//! before／after 各 checkout で実行する（判定規則の正は
//! `docs/perf/logs/cpu-thread-policy-macmini-3178/RULE.txt`）。
//!
//! 環境変数 `EXPECT_SMALL_SHAPE_CAP` で検査を切り替える。
//! - 未設定: 出力のみ（Mac mini 以外の smoke・手動確認用。assert しない）。
//! - `0`: before 腕（既定 OFF）。`pool_active == false` を要求する。
//! - `1`: after 腕（`SMALL_SHAPE_CAP_ENABLED = true` と brand 追加の patch 適用済み）。
//!   `eligible`・`!env_override`・`current > cap_threads`・`pool_active` を要求する。
//! - それ以外: panic で拒否する（fail-closed）。
//!
//! 実機依存のため `#[ignore]`。after checkout では `--lib` を実行せず、
//! `--test small_shape_cap_report` で単独実行すること。

use fandhe_ai_backend_cpu::{gemm_blis_parallel, small_shape_cap_report};

#[test]
#[ignore = "Mac mini 実機依存（イシュー #3178）。run_ab_thread_policy_macmini_cpu.sh が --ignored で実行する"]
fn small_shape_cap_report_dump() {
    // 学習形状（cap 対象）。出力値は検査しない（発火の有無だけが目的）。
    let (m, n, k) = (64usize, 256usize, 784usize);
    let a = vec![0.5f32; m * k];
    let b = vec![0.25f32; k * n];
    let mut c = vec![0.0f32; m * n];
    gemm_blis_parallel(&a, &b, &mut c, m, n, k).expect("学習形状の GEMM が失敗した");

    let r = small_shape_cap_report();
    eprintln!(
        "small_shape_cap_report current={} eligible={} env_override={} cap_threads={} max_work={} pool_active={}",
        r.current, r.eligible, r.env_override, r.cap_threads, r.max_work, r.pool_active
    );

    match std::env::var("EXPECT_SMALL_SHAPE_CAP").ok().as_deref() {
        None => {}
        Some("0") => {
            assert!(
                !r.pool_active,
                "before 腕なのに専用プールが有効（SMALL_SHAPE_CAP_ENABLED が true）"
            );
        }
        Some("1") => {
            assert!(
                r.eligible,
                "自機判定が不成立（ELIGIBLE_BRANDS に実測 brand を追加した patch が未適用）"
            );
            assert!(
                !r.env_override,
                "RAYON_NUM_THREADS により機構が無効化されている"
            );
            assert!(
                r.current > r.cap_threads,
                "グローバルプールが専用プール以下のスレッド数で、cap の意味がない"
            );
            assert!(r.pool_active, "専用プールが発火していない");
        }
        Some(other) => panic!("EXPECT_SMALL_SHAPE_CAP は 0 か 1 のみ（got: {other}）"),
    }
}
