//! イシュー #3612: Mac mini の Metal で常駐チェーン（train・infer）の 1 step の時間を区間別に測る
//! record-only 診断（`docs/compat-resident-train-step-decision.md` §17.6・事前登録手順
//! `docs/perf/logs/resident-chain-metal-breakdown-3612/RULE.txt`）。
//!
//! # 配置理由
//!
//! `MetalContext::synchronize_with_gpu_timestamps` は既定ビルドで `pub(crate)`、同期カウンタは
//! `__diagnostic_batch_counters_snapshot` 経由のため、`infer_fixed_cost_diag_tests` と同じく crate 内部の
//! 兄弟モジュールに置く（`Cargo.toml`・feature は変更しない。`#[cfg(test)]`・`#[ignore]` なので lib
//! には入らず、本番経路は変わらない）。facade は dev-dependency にしない（別コピーができて
//! singleton の `cached_context` が分かれ、カウンタとタイムスタンプが読めなくなるため）。facade の
//! `train_step_resident`／`ResidentLoss::read`／`predict_resident_input` と同じ呼び出し列
//! （`train_step_chain_diag_tests_3252` と同じ再現方式）を 1 スレッドで再現する。
//!
//! # 腕（RULE.txt）
//!
//! - **B0**（本番の同期位置）: train は prep（tape 作成・`snapshot_resident_params`・
//!   `validate_train_step_device_chain`）／encode（`train_step_device_chain`）／read（新しい tape で
//!   `read_train_step_loss`）の wall。infer は `predict_device_chain_resident` の wall。
//!   定常 step で `wait_until_completed` の差分が 1 であることを assert する（Tape 経路への
//!   フォールバックなら約 7）。
//! - **B1**（GPU 時間の分離）: train の encode 直後・read の前に `synchronize_with_gpu_timestamps`
//!   を明示的に呼び、`commit_wait`・`gpu_busy`（バッチの End−Start の和）・`host_fixed`
//!   （`commit_wait − gpu_busy`）・`readout` を分ける。infer は `predict_device_chain_once` と同じ部品列
//!   （`linear_forward_device` を層ごとに積む。Metal は `_into` 非対応で確保経路）を再現し、
//!   出力が B0 の出力と bit 一致することを assert する。
//!
//! 値の大小は assert しない（record-only）。hard 条件（wait の数・bit 一致・有限）だけ assert する。
//! 1 run 分の JSONL を環境変数 `BREAKDOWN_3612_OUT` のパスへ追記する（未設定なら stdout のみ）。
//!
//! 実機実行: `cargo test --release -p fandhe-ai-backend-metal --lib resident_chain_breakdown_diag_tests_3612 -- --ignored --nocapture --test-threads=1`

use std::io::Write;
use std::time::Instant;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::optim::resident_input::ResidentInput;
use fandhe_ai_autodiff::optim::{DeviceParamStore, ResidentLeaf, SgdConfig};
use fandhe_ai_tensor_core::buffer::DeviceBufferView;
use fandhe_ai_tensor_core::{Activation, BackendOps, Tensor};

use crate::context_cache::cached_context;
use crate::ops::MetalBackendOps;

const BATCH: usize = 64;
const DIMS: [usize; 3] = [784, 256, 10];
const TRAIN_STEPS: usize = 100;
const TRAIN_WARMUP: usize = 20;
const INFER_ITERS: usize = 100;
const INFER_WARMUP: usize = 20;

fn pattern(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + salt * 17) % 23) as f32 - 11.0) * 0.011)
        .collect()
}

fn tensor(salt: usize, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(pattern(n, salt), shape).expect("形状とデータ長は一致")
}

fn config() -> SgdConfig {
    SgdConfig {
        lr: 0.01,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
    }
}

fn new_tape() -> Tape {
    Tape::new_with_ops(Box::new(MetalBackendOps::new()))
}

fn counters() -> crate::context::BatchCountersSnapshot {
    crate::__diagnostic_batch_counters_snapshot().expect("counters")
}

/// 784-256-10 の store と常駐入力 x・y を作る（窓の外）。
fn setup() -> (DeviceParamStore, ResidentInput, ResidentInput) {
    let tape = new_tape();
    let params = [
        tensor(1, &[DIMS[0], DIMS[1]]),
        tensor(2, &[DIMS[1]]),
        tensor(3, &[DIMS[1], DIMS[2]]),
        tensor(4, &[DIMS[2]]),
    ];
    let refs: Vec<&Tensor<f32>> = params.iter().collect();
    let store = DeviceParamStore::new(&tape, &refs).expect("store");
    let x = store
        .upload_resident_input(&tape, &tensor(5, &[BATCH, DIMS[0]]))
        .expect("x");
    let y = store
        .upload_resident_input(&tape, &tensor(6, &[BATCH, DIMS[2]]))
        .expect("y");
    (store, x, y)
}

fn med(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("非 NaN"));
    s[s.len() / 2]
}

fn emit(out: &mut Option<std::fs::File>, line: &str) {
    println!("{line}");
    if let Some(f) = out.as_mut() {
        writeln!(f, "{line}").expect("JSONL 書き込み");
    }
}

fn open_out() -> Option<std::fs::File> {
    std::env::var("BREAKDOWN_3612_OUT").ok().map(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .expect("出力 JSONL を開けない")
    })
}

/// 1 step の区間（秒）と同期回数。
struct TrainSample {
    prep: f64,
    encode: f64,
    commit_wait: f64,
    gpu_busy: f64,
    readout: f64,
    total: f64,
    waits: usize,
    command_buffers: usize,
    encode_calls: usize,
    loss: f32,
}

/// 1 step。`explicit_sync` が true なら B1（encode 直後に GPU タイムスタンプ付きで同期）。
fn train_step(
    store: &mut DeviceParamStore,
    x: &ResidentInput,
    y: &ResidentInput,
    explicit_sync: bool,
) -> TrainSample {
    let ctx = cached_context().expect("Metal context");
    let c0 = counters();
    let t_all = Instant::now();

    // prep: facade の tape_for・snapshot・validate に相当（ホスト側の記帳）。
    let t = Instant::now();
    let tape = new_tape();
    let leaves: Vec<ResidentLeaf<'_>> = store.snapshot_resident_params(&tape).expect("leaves");
    let steps = [
        (&leaves[0], Some(&leaves[1]), Activation::Relu),
        (&leaves[2], Some(&leaves[3]), Activation::None),
    ];
    store
        .validate_train_step_device_chain(&tape, x, y, &steps, &config())
        .expect("validate");
    let prep = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let handle = store
        .train_step_device_chain(&tape, x, y, &steps, &config())
        .expect("train_step_device_chain");
    let encode = t.elapsed().as_secs_f64();
    drop(leaves);

    let (mut commit_wait, mut gpu_busy) = (0.0, 0.0);
    if explicit_sync {
        let t = Instant::now();
        let batches = ctx
            .synchronize_with_gpu_timestamps()
            .expect("commit + wait");
        commit_wait = t.elapsed().as_secs_f64();
        // タイムスタンプ欠落は gpu_busy を過小にするため hard 条件にする。
        gpu_busy = batches
            .iter()
            .map(|b| b.kernel_gpu_secs().expect("GPU タイムスタンプが取れない"))
            .sum();
    }

    let t = Instant::now();
    let read_tape = new_tape();
    let loss = store
        .read_train_step_loss(&read_tape, &handle)
        .expect("read_train_step_loss");
    let readout = t.elapsed().as_secs_f64();
    let total = t_all.elapsed().as_secs_f64();
    let c1 = counters();
    TrainSample {
        prep,
        encode,
        commit_wait,
        gpu_busy,
        readout,
        total,
        waits: c1.wait_until_completed - c0.wait_until_completed,
        command_buffers: c1.command_buffers - c0.command_buffers,
        encode_calls: c1.encode_calls - c0.encode_calls,
        loss,
    }
}

fn run_train(arm: &str, explicit_sync: bool) {
    let mut out = open_out();
    let (mut store, x, y) = setup();
    let mut samples = Vec::new();
    for i in 0..(TRAIN_WARMUP + TRAIN_STEPS) {
        let s = train_step(&mut store, &x, &y, explicit_sync);
        assert!(s.loss.is_finite(), "loss={}", s.loss);
        if i >= TRAIN_WARMUP {
            // 定常 step で同期は 1 step あたりちょうど 1 回（フォールバック検出を兼ねる）。
            assert_eq!(
                s.waits, 1,
                "arm={arm} step={i}: wait_until_completed の差分"
            );
            samples.push(s);
        }
    }
    let m = |f: fn(&TrainSample) -> f64| med(&samples.iter().map(f).collect::<Vec<_>>());
    let host_fixed = if explicit_sync {
        med(&samples
            .iter()
            .map(|s| s.commit_wait - s.gpu_busy)
            .collect::<Vec<_>>())
    } else {
        0.0
    };
    let wmax = samples.iter().map(|s| s.waits).max().unwrap_or(0);
    let cbmax = samples.iter().map(|s| s.command_buffers).max().unwrap_or(0);
    let ecmax = samples.iter().map(|s| s.encode_calls).max().unwrap_or(0);
    emit(
        &mut out,
        &format!(
            "{{\"task\":\"train\",\"arm\":\"{arm}\",\"n\":{},\"prep_us\":{:.3},\"encode_us\":{:.3},\"commit_wait_us\":{:.3},\"gpu_busy_us\":{:.3},\"host_fixed_us\":{:.3},\"readout_us\":{:.3},\"total_us\":{:.3},\"wait_max\":{wmax},\"command_buffers_max\":{cbmax},\"encode_calls_max\":{ecmax},\"loss_last\":{:.9}}}",
            samples.len(),
            m(|s| s.prep) * 1e6,
            m(|s| s.encode) * 1e6,
            m(|s| s.commit_wait) * 1e6,
            m(|s| s.gpu_busy) * 1e6,
            host_fixed * 1e6,
            m(|s| s.readout) * 1e6,
            m(|s| s.total) * 1e6,
            samples.last().expect("非空").loss,
        ),
    );
}

#[test]
#[ignore = "Metal 実機（Mac mini）依存の record-only 診断。CI では実行しない"]
fn resident_chain_breakdown_3612_train_b0() {
    run_train("b0", false);
}

#[test]
#[ignore = "Metal 実機（Mac mini）依存の record-only 診断。CI では実行しない"]
fn resident_chain_breakdown_3612_train_b1() {
    run_train("b1", true);
}

/// infer の B0（本番の `predict_device_chain_resident`）と B1（部品列の再現＋明示同期）。
#[test]
#[ignore = "Metal 実機（Mac mini）依存の record-only 診断。CI では実行しない"]
fn resident_chain_breakdown_3612_infer() {
    let mut out = open_out();
    let ctx = cached_context().expect("Metal context");
    let (store, x, _y) = setup();
    let tape = new_tape();
    let leaves: Vec<ResidentLeaf<'_>> = store.snapshot_resident_params(&tape).expect("leaves");
    let steps = [
        (&leaves[0], Some(&leaves[1]), Activation::Relu),
        (&leaves[2], Some(&leaves[3]), Activation::None),
    ];

    // B0: 本番経路の wall。定常で wait は 1 回。
    // RULE.txt: 奇数 round は B0 先・偶数 round は B1 先。ランナーが偶数 round で
    // `BREAKDOWN_3612_B1_FIRST=1` を渡し、計測順だけを反転する（出力の bit 一致検査は順序に依らない）。
    let run_b0 = || {
        let (mut b0_wall, mut b0_ref) = (Vec::new(), None);
        let (mut b0_wait_max, mut b0_cb_max) = (0, 0);
        for i in 0..(INFER_WARMUP + INFER_ITERS) {
            let c0 = counters();
            let t = Instant::now();
            let o = store
                .predict_device_chain_resident(&tape, &x, &steps)
                .expect("predict_device_chain_resident");
            let wall = t.elapsed().as_secs_f64();
            let c1 = counters();
            let waits = c1.wait_until_completed - c0.wait_until_completed;
            let cbs = c1.command_buffers - c0.command_buffers;
            if i >= INFER_WARMUP {
                assert_eq!(waits, 1, "B0 infer: wait の差分（iter={i}）");
                b0_wait_max = b0_wait_max.max(waits);
                b0_cb_max = b0_cb_max.max(cbs);
                b0_wall.push(wall);
            }
            b0_ref = Some(o);
        }
        (b0_ref.expect("非空"), b0_wall, b0_wait_max, b0_cb_max)
    };

    // B1: 同じ部品列を own buffer で再現（store と同じ初期値を upload）。
    let run_b1 = || {
        let ops = MetalBackendOps::new();
        let mem = ops.memory_ops().expect("Metal は MemoryOps を実装");
        let w1 = mem.upload(&tensor(1, &[DIMS[0], DIMS[1]])).expect("w1");
        let b1 = mem.upload(&tensor(2, &[DIMS[1]])).expect("b1");
        let w2 = mem.upload(&tensor(3, &[DIMS[1], DIMS[2]])).expect("w2");
        let b2 = mem.upload(&tensor(4, &[DIMS[2]])).expect("b2");
        let xin = mem.upload(&tensor(5, &[BATCH, DIMS[0]])).expect("x");
        let (s_w1, s_b1, s_w2, s_b2) =
            ([DIMS[0], DIMS[1]], [DIMS[1]], [DIMS[1], DIMS[2]], [DIMS[2]]);
        let (mut enc, mut cw, mut gb, mut ro, mut tot) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut last = None;
        let (mut wait_max, mut cb_max) = (0, 0);
        for i in 0..(INFER_WARMUP + INFER_ITERS) {
            let c0 = counters();
            let t_all = Instant::now();
            let t = Instant::now();
            let h = ops
                .linear_forward_device(
                    &xin,
                    DeviceBufferView::new(&w1, 0, &s_w1).expect("view"),
                    Some(DeviceBufferView::new(&b1, 0, &s_b1).expect("view")),
                    Activation::Relu,
                )
                .expect("L1");
            let o = ops
                .linear_forward_device(
                    &h,
                    DeviceBufferView::new(&w2, 0, &s_w2).expect("view"),
                    Some(DeviceBufferView::new(&b2, 0, &s_b2).expect("view")),
                    Activation::None,
                )
                .expect("L2");
            let e = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let batches = ctx
                .synchronize_with_gpu_timestamps()
                .expect("commit + wait");
            let c = t.elapsed().as_secs_f64();
            let g: f64 = batches
                .iter()
                .map(|b| b.kernel_gpu_secs().expect("GPU タイムスタンプが取れない"))
                .sum();
            let t = Instant::now();
            let res = mem.download(&o).expect("download");
            let r = t.elapsed().as_secs_f64();
            let total = t_all.elapsed().as_secs_f64();
            let c1 = counters();
            let waits = c1.wait_until_completed - c0.wait_until_completed;
            let cbs = c1.command_buffers - c0.command_buffers;
            if i >= INFER_WARMUP {
                // train・B0 と同じく定常区間の各反復で同期 1 回を hard 条件にする。
                assert_eq!(waits, 1, "B1 infer: wait の差分（iter={i}）");
                wait_max = wait_max.max(waits);
                cb_max = cb_max.max(cbs);
                enc.push(e);
                cw.push(c);
                gb.push(g);
                ro.push(r);
                tot.push(total);
            }
            last = Some(res);
        }
        (last.expect("非空"), enc, cw, gb, ro, tot, wait_max, cb_max)
    };

    let b1_first = std::env::var("BREAKDOWN_3612_B1_FIRST").is_ok_and(|v| v == "1");
    let ((b0_ref, b0_wall, b0_wait_max, b0_cb_max), (last, enc, cw, gb, ro, tot, wait_max, cb_max)) =
        if b1_first {
            let b1r = run_b1();
            let b0r = run_b0();
            (b0r, b1r)
        } else {
            let b0r = run_b0();
            let b1r = run_b1();
            (b0r, b1r)
        };
    let sum0: f64 = b0_ref
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|v| f64::from(*v))
        .sum();
    emit(
        &mut out,
        &format!(
            "{{\"task\":\"infer\",\"arm\":\"b0\",\"n\":{},\"total_us\":{:.3},\"wait_max\":{b0_wait_max},\"command_buffers_max\":{b0_cb_max},\"checksum\":{:.9}}}",
            b0_wall.len(),
            med(&b0_wall) * 1e6,
            sum0
        ),
    );
    // hard 条件: B1 の再現出力は B0 と bit 一致（窓の外で 1 回検査）。
    let (a, b) = (
        last.as_slice().expect("contiguous"),
        b0_ref.as_slice().expect("contiguous"),
    );
    let same = a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(same, "B1 infer の出力が B0 と bit 一致しない");
    let hf: Vec<f64> = cw.iter().zip(&gb).map(|(c, g)| c - g).collect();
    emit(
        &mut out,
        &format!(
            "{{\"task\":\"infer\",\"arm\":\"b1\",\"n\":{},\"encode_us\":{:.3},\"commit_wait_us\":{:.3},\"gpu_busy_us\":{:.3},\"host_fixed_us\":{:.3},\"readout_us\":{:.3},\"total_us\":{:.3},\"wait_max\":{wait_max},\"command_buffers_max\":{cb_max},\"bit_identical_to_b0\":true}}",
            tot.len(),
            med(&enc) * 1e6,
            med(&cw) * 1e6,
            med(&gb) * 1e6,
            med(&hf) * 1e6,
            med(&ro) * 1e6,
            med(&tot) * 1e6,
        ),
    );
}
