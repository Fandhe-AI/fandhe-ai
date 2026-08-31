//! イシュー #1015（親ツリー）・#1017（実装）・`docs/backend-metal-
//! command-batching-design.md` §4「期待効果と実機計測計画」の Mac
//! セッション記入用マイクロベンチ。
//!
//! `crates/backend-metal/tests/command_batching.rs` が正しさ（数値一致・
//! 実行順序）を検証するのに対し、本ファイルは **#1017 が変更した唯一の
//! 経路（`BackendOps::sgd_step_device_tracked`）の性能を、変更前の同期
//! 契約（`BackendOps::sgd_step_device`。呼び出しごとに `encode` +
//! `ctx.synchronize()` を行う。`ops.rs::MetalBackendOps::
//! sgd_step_device_tracked` doc「デフォルト実装は `sgd_step_device` へ
//! 委譲」）と直接比較する**ことで、#1017 の変更を他のイシュー
//! （#1013・#1023・#1028 等）の効果と混同せず単独に隔離して計測する
//! （設計文書 §4 は `bench-fandhe --task train --mode reuse` の 5 回計測
//! 中央値を基準点に挙げるが、その基準点〈crates.io 0.4.0 ピン〉は本
//! ワークツリーの改善前コードを含まないため #1017 単独の delta を測れ
//! ない。本ファイルは同一 HEAD 上で「バッチ化した経路」と「バッチ化して
//! いない経路」を横並びで測ることで、その制約を回避する）。
//!
//! `cfg(target_os = "macos")` ＋ 各 `#[test]` の理由付き `#[ignore]` は
//! `command_batching.rs` と同じ方針。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture
//! ```
#![cfg(target_os = "macos")]

use std::time::Instant;

use bench_harness::median_q1_q3;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::{BackendOps, DispatchFailureCell, SgdStepConfig, Tensor};

/// 更新対象の要素数。固定費（コマンドバッファ生成・
/// `waitUntilCompleted`）が支配的になる範囲であることを意図した値
/// （`docs/perf/metal-fixed-overhead-diagnosis.md` §1 の「サイズに依らず
/// 約 5 ms に張り付く」観測と同じ着眼点）。
const NUMEL: usize = 1024;
/// 5 回計測中央値方針（`.claude/rules/coding-rust.md`）。
const TRIALS: usize = 5;
/// 1 trial あたりの連続更新回数。
const STEPS: usize = 100;

fn sgd_config() -> SgdStepConfig {
    SgdStepConfig {
        lr: 0.001,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        is_first_step: true,
    }
}

/// 非バッチ経路（`BackendOps::sgd_step_device`）で `steps` 回連続更新した
/// 所要秒を返す。呼び出しごとに `encode` + `ctx.synchronize()`
/// （`sgd.rs::MetalSgd::run` の `token: None` 分岐）を行うため、GPU
/// 実行完了をホストが毎回ブロッキング待機する（#1017 以前の
/// `sgd_step_device_tracked` デフォルト委譲と同一の同期契約）。
fn run_untracked(ops: &MetalBackendOps, steps: usize) -> f64 {
    let mem = ops
        .memory_ops()
        .expect("MetalBackendOps must implement MemoryOps");
    let mut param = mem
        .upload(&Tensor::new(vec![0.0f32; NUMEL], &[NUMEL]).unwrap())
        .unwrap();
    let grad = mem
        .upload(&Tensor::new(vec![0.01f32; NUMEL], &[NUMEL]).unwrap())
        .unwrap();
    let config = sgd_config();

    let start = Instant::now();
    for _ in 0..steps {
        ops.sgd_step_device(&mut param, &grad, None, &config)
            .expect("sgd_step_device must succeed on real hardware");
    }
    start.elapsed().as_secs_f64()
}

/// バッチ経路（`BackendOps::sgd_step_device_tracked`。#1017）で `steps`
/// 回連続更新した所要秒を返す。呼び出しごとには待たず（`encode` のみ）、
/// `steps` 回分をまとめて 1 回の `download`（唯一の同期点。設計文書
/// §3.5）で完了を確定させる。
fn run_tracked(ops: &MetalBackendOps, steps: usize) -> f64 {
    let mem = ops
        .memory_ops()
        .expect("MetalBackendOps must implement MemoryOps");
    let mut param = mem
        .upload(&Tensor::new(vec![0.0f32; NUMEL], &[NUMEL]).unwrap())
        .unwrap();
    let grad = mem
        .upload(&Tensor::new(vec![0.01f32; NUMEL], &[NUMEL]).unwrap())
        .unwrap();
    let config = sgd_config();
    let token = DispatchFailureCell::new();

    let start = Instant::now();
    for _ in 0..steps {
        ops.sgd_step_device_tracked(&mut param, &grad, None, &config, &token)
            .expect("sgd_step_device_tracked must succeed on real hardware");
    }
    let _ = mem
        .download(&param)
        .expect("download must synchronize the batched steps");
    let elapsed = start.elapsed().as_secs_f64();
    assert!(
        !token.is_set(),
        "no runtime error expected on real hardware"
    );
    elapsed
}

/// #1017 の効果を隔離するマイクロベンチ本体（record only。hard assert
/// はしない。`crates/facade/tests/device_param_store_bench.rs` と同じ
/// 「GPU クロック挙動等の環境揺らぎを hard assert に持ち込まない」方針）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn command_batching_micro_bench_untracked_vs_tracked() {
    let ops = MetalBackendOps::new();

    // warmup: パイプライン初回コンパイル（`MetalSgd::new` の実行時
    // コンパイル）コストを本計測から除く。
    let _ = run_untracked(&ops, 5);
    let _ = run_tracked(&ops, 5);

    let mut untracked_secs = Vec::with_capacity(TRIALS);
    let mut tracked_secs = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        untracked_secs.push(run_untracked(&ops, STEPS));
        tracked_secs.push(run_tracked(&ops, STEPS));
    }

    let untracked_q = median_q1_q3(&untracked_secs).expect("TRIALS 個の non-NaN サンプルのはず");
    let tracked_q = median_q1_q3(&tracked_secs).expect("TRIALS 個の non-NaN サンプルのはず");

    println!(
        "[command_batching_micro_bench] steps={STEPS} numel={NUMEL} \
         untracked_median_s={:.6} (q1={:.6}, q3={:.6}) \
         tracked_median_s={:.6} (q1={:.6}, q3={:.6}) \
         speedup_x={:.3} tracked_faster={} \
         — record only, non-gating（本ファイル冒頭コメント参照）",
        untracked_q.median,
        untracked_q.q1,
        untracked_q.q3,
        tracked_q.median,
        tracked_q.q1,
        tracked_q.q3,
        untracked_q.median / tracked_q.median.max(f64::EPSILON),
        tracked_q.median < untracked_q.median,
    );
}
