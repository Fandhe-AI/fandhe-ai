//! Metal の「同期 1 回あたりの固定費」を空 dispatch の壁時間で測る record-only 診断
//! （イシュー #3180。規則の正は `docs/perf/logs/train-infer-phases-macmini-3180/RULE.txt`）。
//!
//! # 役割と呼び出し文脈
//!
//! `MetalContext::dispatch_sync`（encode → commit → `waitUntilCompleted`）へ中身が空の
//! encoder を渡し、コマンドバッファ 1 本の生成・commit・完了待ちだけの壁時間を取る。
//! 結果は `docs/perf/train-step-phase-breakdown.md` §18 で、同期回数（11/7/7 の wait 回数）×
//! 固定費による「同期で説明できる時間」の見積りに使う（後続 #3195・#3208・#3217 の判断材料）。
//! 実カーネルを伴う同期を代表しない可能性があるため、`infer_fixed_cost_diag`
//! （wall − GPU busy）と併記して読む。
//!
//! # 契約
//!
//! - ライブラリ本体・既定値・feature は変えない（公開・safe な API のみ。`unsafe` なし）。
//! - 壁時間の大小は assert しない（record-only）。assert するのは「反復ごとに
//!   `command_buffers`・`wait_until_completed` が自インスタンスのカウンタでちょうど +1」という
//!   構造だけ（1 回の同期になっていない計測は固定費として不採用。fail-closed）。
//!
//! 実機実行（`--release` 必須・`--test-threads=1`）:
//!
//! ```sh
//! cargo test --release -p fandhe-ai-backend-metal --test sync_fixed_cost_diag -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use std::time::Instant;

use bench_harness::median_q1_q3;
use fandhe_ai_backend_metal::MetalContext;

const WARMUP: usize = 20;
const TRIALS: usize = 200;

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存の record-only 診断。CI では実行しない"]
fn sync_fixed_cost_empty_dispatch() {
    let ctx = MetalContext::new().expect("Metal コンテキストの生成に失敗");
    for _ in 0..WARMUP {
        ctx.dispatch_sync(|_enc| {}).expect("warmup の空 dispatch");
    }
    let mut samples = Vec::with_capacity(TRIALS);
    for i in 0..TRIALS {
        let before = ctx.diagnostic_batch_counters();
        let t0 = Instant::now();
        ctx.dispatch_sync(|_enc| {}).expect("空 dispatch");
        let wall = t0.elapsed().as_secs_f64();
        let after = ctx.diagnostic_batch_counters();
        assert_eq!(
            after.command_buffers - before.command_buffers,
            1,
            "反復 {i}: コマンドバッファがちょうど 1 本ではない"
        );
        assert_eq!(
            after.wait_until_completed - before.wait_until_completed,
            1,
            "反復 {i}: waitUntilCompleted がちょうど 1 回ではない"
        );
        samples.push(wall);
    }
    let q = median_q1_q3(&samples).expect("サンプルは空でない");
    println!(
        "SYNC_FIXED_COST_EMPTY_DISPATCH trials={TRIALS} warmup={WARMUP} median_us={:.3} q1_us={:.3} q3_us={:.3} structure=cb+1,wait+1",
        q.median * 1e6,
        q.q1 * 1e6,
        q.q3 * 1e6
    );
}
