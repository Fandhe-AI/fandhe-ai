//! イシュー #3183（`docs/perf/metal-infer-bimodal-macmini.md`）: Mac mini の infer Metal reuse
//! （784→256→ReLU→10・batch 64・`Sequential::predict_resident`）に見られる 1 反復時間の
//! 二峰性を、**1 反復 1 行**の時系列で記録する record-only 診断。
//!
//! `metal_infer_tape_build_fixedcost_diag.rs`（#2114）はセル単位の集計値しか出さないため、
//! 高い側のモードがどの区間・どのカウンタと同時に現れるかを分けられない。本ファイルは
//! 同じ区間分解を反復ごとに出力し、モード検出・帰属は `aggregate.py`（事前登録規則
//! `docs/perf/logs/metal-infer-bimodal-macmini-3183/RULE.txt`）が行う。
//!
//! - 腕 R（`metal_infer_bimodal_arm_r`）: `predict_resident` → bench-fandhe と同じ legacy 読み出し
//!   （`contiguous().as_slice().to_vec()`）→ checksum
//! - 腕 D（`metal_infer_bimodal_arm_d`）: provider_select → tape_new → snapshot → chain →
//!   host_copy → checksum（デバイス存在確認キャッシュ OFF = 既定）
//! - 腕 Don（`metal_infer_bimodal_arm_d_on`）: D と同じ列をキャッシュ ON のスコープ内で実行
//!   （診断専用。既定値は変えない）
//! - 腕 G（GPU タイムスタンプ）は `MetalContext` の内部 API が必要なため
//!   `crates/backend-metal/src/infer_fixed_cost_diag_tests.rs` に置く
//!
//! 起動順は腕ごとに別 `#[test]` とし、`orchestrate.sh` が `--exact` で run ごとにローテーションする。
//! 値の大小は assert しない（flaky 化防止）。checksum の反復間一致のみ hard assert する。
//! `MetalContext` singleton のカウンタを共有するため `METAL_SINGLETON_LOCK` で直列化する。
//!
//! 本番コード・tolerance・baseline は変更しない。
#![cfg(target_os = "macos")]

use std::sync::Mutex;
use std::time::Instant;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tensor};
use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::optim::DeviceParamStore;
use fandhe_ai_backend_metal::fixed_cost_diag::{
    __diagnostic_fixed_cost_counters_snapshot, override_device_verify_cache_for_scope,
    verify_device_cached,
};
use fandhe_ai_backend_metal::{
    __diagnostic_batch_counters_snapshot, MetalBackendOps, MetalDeviceProvider,
};
use fandhe_ai_tensor_core::{Activation, DeviceProvider};

static METAL_SINGLETON_LOCK: Mutex<()> = Mutex::new(());

const BATCH: usize = 64;
const IN_FEATURES: usize = 784;
const HIDDEN: usize = 256;
const OUT_FEATURES: usize = 10;
const WARMUP: usize = 20;
const MEASURED: usize = 500;
/// 標準出力の 1 行 1 反復 JSON の接頭辞（`orchestrate.sh` が抽出）。
const PREFIX: &str = "DIAG_JSON ";

fn make_input() -> Tensor<f32> {
    let data = Xorshift64Star::new(0x9000).fill_vec(BATCH * IN_FEATURES);
    Tensor::new(data, &[BATCH, IN_FEATURES]).expect("形状とデータ長は一致させている")
}

fn make_model() -> Sequential {
    Sequential::new()
        .add_linear(IN_FEATURES, HIDDEN, 42)
        .expect("Linear 構築")
        .add_relu()
        .add_linear(HIDDEN, OUT_FEATURES, 43)
        .expect("Linear 構築")
}

/// bench-fandhe の Metal 既定（legacy 読み出し）と同じホスト側コピー。
fn host_copy(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous 直後は as_slice が Some")
        .to_vec()
}

/// 出力の順序込み bit ダイジェスト（FNV-1a 64bit。#2114 の診断と同一定義）。
fn checksum(v: &[f32]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for byte in (v.len() as u64).to_le_bytes() {
        h = (h ^ u64::from(byte)).wrapping_mul(PRIME);
    }
    for x in v {
        for byte in x.to_bits().to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(PRIME);
        }
    }
    h
}

fn provider_select_cached() {
    verify_device_cached(|| {
        let provider = MetalDeviceProvider::new();
        provider.select(Device::Metal).map(|_| ())
    })
    .expect("Metal デバイス");
}

fn batch_counters() -> (usize, usize, usize) {
    let c = __diagnostic_batch_counters_snapshot().expect("Metal context");
    (c.encode_calls, c.command_buffers, c.wait_until_completed)
}

struct Ctx {
    model: Sequential,
    store: DeviceParamStore,
    input: Tensor<f32>,
}

fn make_ctx() -> Ctx {
    let model = make_model();
    let init_tape = fandhe_ai::tape_for(Device::Metal).expect("tape_for");
    let store = model.init_device_param_store(&init_tape).expect("store");
    drop(init_tape);
    Ctx {
        model,
        store,
        input: make_input(),
    }
}

/// 1 反復の結果。`spans` は (区間名, 秒)、`sum` は出力の checksum。
struct Iter {
    spans: Vec<(&'static str, f64)>,
    sum: u64,
}

fn timed<T>(spans: &mut Vec<(&'static str, f64)>, name: &'static str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let out = f();
    spans.push((name, t.elapsed().as_secs_f64()));
    out
}

fn iter_r(c: &Ctx) -> Iter {
    let mut spans = Vec::new();
    let t_all = Instant::now();
    let out = timed(&mut spans, "predict_resident", || {
        c.model
            .predict_resident(&c.store, &c.input)
            .expect("resident")
    });
    let v = timed(&mut spans, "host_copy", || host_copy(&out));
    let sum = timed(&mut spans, "checksum", || checksum(&v));
    spans.push(("iter_total", t_all.elapsed().as_secs_f64()));
    Iter { spans, sum }
}

fn iter_d(c: &Ctx) -> Iter {
    let mut spans = Vec::new();
    let t_all = Instant::now();
    timed(&mut spans, "provider_select", provider_select_cached);
    let tape = timed(&mut spans, "tape_new", || {
        Tape::new_with_ops(Box::new(MetalBackendOps::new()))
    });
    let leaves = timed(&mut spans, "snapshot", || {
        c.store.snapshot_resident_params(&tape).expect("snapshot")
    });
    let steps = vec![
        (&leaves[0], Some(&leaves[1]), Activation::Relu),
        (&leaves[2], Some(&leaves[3]), Activation::None),
    ];
    let out = timed(&mut spans, "chain", || {
        c.store
            .predict_device_chain(&tape, &c.input, &steps)
            .expect("chain")
    });
    let v = timed(&mut spans, "host_copy", || host_copy(&out));
    let sum = timed(&mut spans, "checksum", || checksum(&v));
    spans.push(("iter_total", t_all.elapsed().as_secs_f64()));
    Iter { spans, sum }
}

/// 腕 1 本分の長い系列（warmup 20 + 計測 500）を回し、計測分だけ 1 反復 1 行で出力する。
/// 出力は計測の外側（系列の最後）にまとめて行い、反復間に標準出力の I/O を挟まない。
fn run_series(arm: &'static str, cache_on: bool, body: impl Fn(&Ctx) -> Iter) {
    let _guard = METAL_SINGLETON_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let c = make_ctx();
    let _cache = override_device_verify_cache_for_scope(cache_on);
    let t0 = Instant::now();
    let mut rows = Vec::with_capacity(MEASURED);
    let mut first_sum: Option<u64> = None;
    for i in 0..(WARMUP + MEASURED) {
        let (b0, f0) = (
            batch_counters(),
            __diagnostic_fixed_cost_counters_snapshot(),
        );
        let t_start = t0.elapsed().as_secs_f64();
        let it = body(&c);
        let (b1, f1) = (
            batch_counters(),
            __diagnostic_fixed_cost_counters_snapshot(),
        );
        if i < WARMUP {
            continue;
        }
        match first_sum {
            None => first_sum = Some(it.sum),
            Some(prev) => assert_eq!(
                prev, it.sum,
                "arm {arm}: 反復間で checksum が不一致（{prev:016x} vs {:016x}）",
                it.sum
            ),
        }
        let d = f1.delta_since(&f0);
        let spans: serde_json::Map<String, serde_json::Value> = it
            .spans
            .iter()
            .map(|(k, v)| ((*k).to_string(), serde_json::json!(v)))
            .collect();
        rows.push(serde_json::json!({
            "issue": 3183,
            "arm": arm,
            "iter": i - WARMUP,
            "t_since_start_s": t_start,
            "spans": spans,
            "counters": {
                "encode": b1.0.saturating_sub(b0.0),
                "command_buffers": b1.1.saturating_sub(b0.1),
                "wait": b1.2.saturating_sub(b0.2),
                "verify_probe": d.verify_probe_calls,
                "verify_hit": d.verify_cache_hits,
                "uploads": d.host_uploads,
                "downloads": d.host_downloads,
            },
            "checksum_bits": format!("{:016x}", it.sum),
            "target_arch": std::env::consts::ARCH,
            "target_os": std::env::consts::OS,
        }));
    }
    for r in rows {
        println!("{PREFIX}{r}");
    }
}

#[test]
#[ignore = "Mac mini 実機計測用の record-only 診断（RULE.txt・orchestrate.sh 参照）"]
fn metal_infer_bimodal_arm_r() {
    run_series("R", false, iter_r);
}

#[test]
#[ignore = "Mac mini 実機計測用の record-only 診断（RULE.txt・orchestrate.sh 参照）"]
fn metal_infer_bimodal_arm_d() {
    run_series("D", false, iter_d);
}

#[test]
#[ignore = "Mac mini 実機計測用の record-only 診断（RULE.txt・orchestrate.sh 参照）"]
fn metal_infer_bimodal_arm_d_on() {
    run_series("Don", true, iter_d);
}
