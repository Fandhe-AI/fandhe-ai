//! infer の GPU 起動・同期固定費を wall 時間と GPU busy（`GPUStartTime`／`GPUEndTime`）の差から
//! 切り分ける record-only 診断（イシュー #2114。設計・判定規則は
//! `docs/perf/metal-tape-build-infer-fixedcost.md`・`docs/perf/logs/metal-tape-build-infer-phase-2114/RULE.txt` の H4）。
//!
//! # 配置理由
//!
//! `MetalContext::synchronize_with_gpu_timestamps` は既定ビルドで `pub(crate)` のため、
//! `gemm_reuse_phase_diag_tests` と同じく crate 内部の兄弟モジュールに置く（`Cargo.toml`・feature は
//! 変更しない）。本番経路（`synchronize`）にはフックしない。
//!
//! # 内容
//!
//! infer（784→256→ReLU→10・batch 64）の GPU 部分だけを `MetalBackendOps` 経由で再現する:
//! upload → `linear_forward_device` × 2（encode のみ）→ `synchronize_with_gpu_timestamps`（commit + wait）
//! → download。区間ごとの wall・GPU busy・両者の差（ホスト側の起動・同期固定費）を出力する。
//! `linear_forward_device` の途中で暗黙同期が起きた分はタイムスタンプを取れないため、取得できた
//! バッチ数を `wait_until_completed` の差分と併記する。値の大小は assert しない（record-only。
//! 環境揺らぎによる flaky 化防止。`gemm_reuse_phase_diag_tests` と同じ方針）。
//!
//! 実機実行: `cargo test --release -p fandhe-ai-backend-metal infer_fixed_cost_diag -- --ignored --nocapture --test-threads=1`

use std::time::Instant;

use bench_harness::{median_q1_q3, rng::Xorshift64Star};
use fandhe_ai_tensor_core::buffer::DeviceBufferView;
use fandhe_ai_tensor_core::{Activation, BackendOps, Tensor};

use crate::context_cache::cached_context;
use crate::ops::MetalBackendOps;

const BATCH: usize = 64;
const IN_FEATURES: usize = 784;
const HIDDEN: usize = 256;
const OUT_FEATURES: usize = 10;
const WARMUP: usize = 20;
const TRIALS: usize = 40;

fn tensor(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(Xorshift64Star::new(seed).fill_vec(n), shape).expect("形状とデータ長は一致")
}

/// 1 反復の区間（秒）。
struct Sample {
    upload: f64,
    encode: f64,
    commit_wait: f64,
    download: f64,
    wall: f64,
    gpu_busy: f64,
    batches_with_timestamps: usize,
    /// 反復内でコミットされたバッチ総数（タイムスタンプ取得数との突合用）。
    batches_total: usize,
    wait_calls: usize,
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存の record-only 診断。CI では実行しない"]
fn infer_fixed_cost_diag_wall_vs_gpu_busy() {
    let ops = MetalBackendOps::new();
    let mem = ops.memory_ops().expect("Metal は MemoryOps を実装");
    let ctx = cached_context().expect("Metal context");
    let x = tensor(1, &[BATCH, IN_FEATURES]);
    let w1 = tensor(2, &[IN_FEATURES, HIDDEN]);
    let b1 = tensor(3, &[HIDDEN]);
    let w2 = tensor(4, &[HIDDEN, OUT_FEATURES]);
    let b2 = tensor(5, &[OUT_FEATURES]);
    let (w1d, b1d) = (mem.upload(&w1).expect("w1"), mem.upload(&b1).expect("b1"));
    let (w2d, b2d) = (mem.upload(&w2).expect("w2"), mem.upload(&b2).expect("b2"));
    let (s_w1, s_b1) = ([IN_FEATURES, HIDDEN], [HIDDEN]);
    let (s_w2, s_b2) = ([HIDDEN, OUT_FEATURES], [OUT_FEATURES]);

    let mut samples = Vec::new();
    for i in 0..(WARMUP + TRIALS) {
        let wait0 = crate::__diagnostic_batch_counters_snapshot()
            .expect("counters")
            .wait_until_completed;
        let t_all = Instant::now();
        let t = Instant::now();
        let a = mem.upload(&x).expect("upload");
        let upload = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let h = ops
            .linear_forward_device(
                &a,
                DeviceBufferView::new(&w1d, 0, &s_w1).expect("view"),
                Some(DeviceBufferView::new(&b1d, 0, &s_b1).expect("view")),
                Activation::Relu,
            )
            .expect("L1");
        let y = ops
            .linear_forward_device(
                &h,
                DeviceBufferView::new(&w2d, 0, &s_w2).expect("view"),
                Some(DeviceBufferView::new(&b2d, 0, &s_b2).expect("view")),
                Activation::None,
            )
            .expect("L2");
        let encode = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let batches = ctx
            .synchronize_with_gpu_timestamps()
            .expect("commit + wait");
        let commit_wait = t.elapsed().as_secs_f64();
        // 反復開始から最終同期直後までの waitUntilCompleted 回数。
        // download 側の同期は gpu_busy の対象外のため、download の前に読む。
        let wait1 = crate::__diagnostic_batch_counters_snapshot()
            .expect("counters")
            .wait_until_completed;

        let t = Instant::now();
        let out = mem.download(&y).expect("download");
        let download = t.elapsed().as_secs_f64();
        let wall = t_all.elapsed().as_secs_f64();
        std::hint::black_box(&out);

        let gpu_busy: f64 = batches.iter().filter_map(|b| b.kernel_gpu_secs()).sum();
        let with_ts = batches
            .iter()
            .filter(|b| b.kernel_gpu_secs().is_some())
            .count();
        if i >= WARMUP {
            samples.push(Sample {
                upload,
                encode,
                commit_wait,
                download,
                wall,
                gpu_busy,
                batches_with_timestamps: with_ts,
                batches_total: batches.len(),
                wait_calls: wait1.saturating_sub(wait0),
            });
        }
    }

    let med = |f: fn(&Sample) -> f64| -> f64 {
        let v: Vec<f64> = samples.iter().map(f).collect();
        median_q1_q3(&v).expect("非空・非 NaN").median
    };
    let wall = med(|s| s.wall);
    let gpu = med(|s| s.gpu_busy);
    println!("infer_fixed_cost_diag (record-only, n={}):", samples.len());
    println!("  upload      median={:.1} us", med(|s| s.upload) * 1e6);
    println!("  encode      median={:.1} us", med(|s| s.encode) * 1e6);
    println!(
        "  commit_wait median={:.1} us",
        med(|s| s.commit_wait) * 1e6
    );
    println!("  download    median={:.1} us", med(|s| s.download) * 1e6);
    println!("  wall        median={:.1} us", wall * 1e6);
    println!("  gpu_busy    median={:.1} us", gpu * 1e6);
    // 全バッチ・全反復で GPU タイムスタンプが取れた場合のみ gpu_busy を有効とする。
    // 欠落時は filter_map の sum が 0 になり (wall-gpu_busy)/wall が約 100% と
    // 表示されて H4（ホスト固定費主体）を誤支持するため、判定不能として扱う。
    // 加えて、反復内の wait 回数が最終同期の返却バッチ数と一致しない反復
    // （linear_forward_device 途中の暗黙同期で一部バッチの GPU 時間が
    // 返却分から欠落）が 1 つでもあれば、ホスト固定費を過大計上しうるため
    // 判定不能とする。
    let ts_complete = samples.iter().all(|s| {
        s.batches_total > 0
            && s.batches_with_timestamps == s.batches_total
            && s.wait_calls == s.batches_total
    });
    if ts_complete && wall > 0.0 {
        // wall - gpu_busy は upload/download（転送とその待機）を含む GPU 非稼働時間。
        // H4（ホスト固定費）の判定には転送区間を除いた値を使う。
        let non_gpu = wall - gpu;
        let non_gpu_share = non_gpu / wall * 100.0;
        println!(
            "  gpu_idle (wall - gpu_busy) median={:.1} us ({non_gpu_share:.0}% of wall。転送を含むため H4 判定には使わない)",
            non_gpu * 1e6
        );
        let transfer = med(|s| s.upload) + med(|s| s.download);
        let host_fixed = non_gpu - transfer;
        let share = host_fixed / wall * 100.0;
        println!(
            "  host_fixed (wall - gpu_busy - upload - download) median={:.1} us ({share:.0}% of wall)",
            host_fixed * 1e6
        );
    } else {
        println!(
            "  host_fixed: GPU タイムスタンプ取得数が期待数未満、または途中同期が疑われる（wait 回数≠返却バッチ数）ため H4 は判定不能（割合は算出しない）"
        );
    }
    let ts_min = samples
        .iter()
        .map(|s| s.batches_with_timestamps)
        .min()
        .unwrap_or(0);
    let ts_max = samples
        .iter()
        .map(|s| s.batches_with_timestamps)
        .max()
        .unwrap_or(0);
    let w_min = samples.iter().map(|s| s.wait_calls).min().unwrap_or(0);
    let w_max = samples.iter().map(|s| s.wait_calls).max().unwrap_or(0);
    println!(
        "  batches_with_timestamps/iter min={ts_min} max={ts_max}; wait_until_completed/iter min={w_min} max={w_max}"
    );
}

// ---------------------------------------------------------------------------
// イシュー #3183: Mac mini の infer Metal 二峰性の腕 G（1 反復 1 行の GPU タイムスタンプ付き時系列）
// ---------------------------------------------------------------------------

const BIMODAL_MEASURED: usize = 500;

/// 腕 G: 上の `infer_fixed_cost_diag_wall_vs_gpu_busy` と同じ列を、中央値ではなく
/// **反復ごと**に `DIAG_JSON ` 接頭辞の 1 行 JSON で出力する（集計・モード検出は
/// `docs/perf/logs/metal-infer-bimodal-macmini-3183/aggregate.py`、規則は同 `RULE.txt`）。
///
/// 本番の `predict_device_chain` は `linear_forward_device_tracked` を呼ぶため、ここでも
/// `DispatchFailureCell` 付きの tracked 版を使う。ただし facade 層（tape・snapshot・
/// legacy 読み出し）は含まないため、R・D 腕との差は「チェーン外の層」と読む。
/// 返却バッチ数と wait 回数が合わない反復・タイムスタンプ欠落の反復は `gpu_busy` を
/// null にする（#2114 の判定不能規則と同じ）。値の大小は assert しない。
/// 本 crate の dev-dependency に `serde_json` を足さないため、JSON は手組みする。
#[test]
#[ignore = "Mac mini 実機計測用の record-only 診断（RULE.txt・orchestrate.sh 参照）"]
fn infer_bimodal_diag_per_iter_gpu_timestamps() {
    use fandhe_ai_tensor_core::DispatchFailureCell;

    let ops = MetalBackendOps::new();
    let mem = ops.memory_ops().expect("Metal は MemoryOps を実装");
    let ctx = cached_context().expect("Metal context");
    let x = tensor(1, &[BATCH, IN_FEATURES]);
    let w1 = tensor(2, &[IN_FEATURES, HIDDEN]);
    let b1 = tensor(3, &[HIDDEN]);
    let w2 = tensor(4, &[HIDDEN, OUT_FEATURES]);
    let b2 = tensor(5, &[OUT_FEATURES]);
    let (w1d, b1d) = (mem.upload(&w1).expect("w1"), mem.upload(&b1).expect("b1"));
    let (w2d, b2d) = (mem.upload(&w2).expect("w2"), mem.upload(&b2).expect("b2"));
    let (s_w1, s_b1) = ([IN_FEATURES, HIDDEN], [HIDDEN]);
    let (s_w2, s_b2) = ([HIDDEN, OUT_FEATURES], [OUT_FEATURES]);
    let token = DispatchFailureCell::new();

    let counters = || {
        let c = crate::__diagnostic_batch_counters_snapshot().expect("counters");
        (c.encode_calls, c.command_buffers, c.wait_until_completed)
    };
    let t0 = Instant::now();
    let mut rows: Vec<String> = Vec::with_capacity(BIMODAL_MEASURED);
    let mut first_bits: Option<Vec<u32>> = None;
    for i in 0..(WARMUP + BIMODAL_MEASURED) {
        let t_start = t0.elapsed().as_secs_f64();
        let b0 = counters();
        let t_all = Instant::now();
        let t = Instant::now();
        let a = mem.upload(&x).expect("upload");
        let upload = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let h = ops
            .linear_forward_device_tracked(
                &a,
                DeviceBufferView::new(&w1d, 0, &s_w1).expect("view"),
                Some(DeviceBufferView::new(&b1d, 0, &s_b1).expect("view")),
                Activation::Relu,
                &token,
            )
            .expect("L1");
        let y = ops
            .linear_forward_device_tracked(
                &h,
                DeviceBufferView::new(&w2d, 0, &s_w2).expect("view"),
                Some(DeviceBufferView::new(&b2d, 0, &s_b2).expect("view")),
                Activation::None,
                &token,
            )
            .expect("L2");
        let encode = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let batches = ctx
            .synchronize_with_gpu_timestamps()
            .expect("commit + wait");
        let commit_wait = t.elapsed().as_secs_f64();
        let b1_ = counters();

        let t = Instant::now();
        let out = mem.download(&y).expect("download");
        let download = t.elapsed().as_secs_f64();
        let wall = t_all.elapsed().as_secs_f64();
        let b2_ = counters();

        let with_ts = batches
            .iter()
            .filter(|b| b.kernel_gpu_secs().is_some())
            .count();
        let wait_calls = b1_.2.saturating_sub(b0.2);
        let ts_complete =
            !batches.is_empty() && with_ts == batches.len() && wait_calls == batches.len();
        let gpu_busy = if ts_complete {
            format!(
                "{:e}",
                batches
                    .iter()
                    .filter_map(|b| b.kernel_gpu_secs())
                    .sum::<f64>()
            )
        } else {
            "null".to_string()
        };

        // 出力の反復間 bit 一致（warmup 後の全反復で同一であること）と FNV-1a ダイジェスト。
        let bits: Vec<u32> = out
            .as_slice()
            .expect("download は contiguous")
            .iter()
            .map(|v| v.to_bits())
            .collect();
        if i < WARMUP {
            continue;
        }
        match &first_bits {
            None => first_bits = Some(bits.clone()),
            Some(prev) => assert_eq!(prev, &bits, "G: 反復間で出力が不一致"),
        }
        let mut fnv: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in (bits.len() as u64).to_le_bytes() {
            fnv = (fnv ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        for byte in bits.iter().flat_map(|v| v.to_le_bytes()) {
            fnv = (fnv ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        rows.push(format!(
            "{{\"issue\":3183,\"arm\":\"G\",\"iter\":{},\"t_since_start_s\":{:e},\
\"spans\":{{\"upload\":{:e},\"encode\":{:e},\"commit_wait\":{:e},\"download\":{:e},\
\"wall\":{:e},\"iter_total\":{:e},\"gpu_busy\":{}}},\
\"counters\":{{\"encode\":{},\"command_buffers\":{},\"wait\":{},\
\"batches_returned\":{},\"batches_with_timestamps\":{}}},\
\"checksum_bits\":\"{:016x}\",\"target_arch\":\"{}\",\"target_os\":\"{}\"}}",
            i - WARMUP,
            t_start,
            upload,
            encode,
            commit_wait,
            download,
            wall,
            wall,
            gpu_busy,
            b1_.0.saturating_sub(b0.0),
            b1_.1.saturating_sub(b0.1),
            b2_.2.saturating_sub(b0.2),
            batches.len(),
            with_ts,
            fnv,
            std::env::consts::ARCH,
            std::env::consts::OS,
        ));
    }
    for r in rows {
        println!("DIAG_JSON {r}");
    }
}
