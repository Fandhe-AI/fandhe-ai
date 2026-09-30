//! イシュー #2114（`docs/perf/metal-tape-build-infer-fixedcost.md`）: Metal の `tape_build` と
//! infer（fresh／reuse）のフェーズ分解診断。
//!
//! `docs/perf/train-step-phase-breakdown.md` §17.3・§17.4 は Metal train の `tape_build` が
//! 14.9〜15.7 µs（CPU 0.1 µs）と報告し、`docs/perf/infer-reuse-phase-breakdown.md` §10.3・§10.4 は
//! Metal infer reuse が CPU の約 2.1 倍で `predict_resident` が単一区間のため分解できないと報告している。
//! 本ファイルは `tape_for(Device::Metal)`（`facade::resolve_ops` の Metal 分岐 =
//! `MetalDeviceProvider::select` → IOKit probe + `MTLCopyAllDevices`）と `Tape::new_with_ops` を
//! 個別に計測し、デバイス存在確認キャッシュ（opt-in・既定 OFF。`fandhe_ai_backend_metal::
//! fixed_cost_diag`）の OFF／ON を同一プロセス内で比較する。
//!
//! - テスト 1（`#[ignore]`・Metal 実機）: 分解した写しが公開 API と bit 一致し、キャッシュ OFF／ON で
//!   出力が bit 同一であることを hard assert する（分解帰属・最適化の前提）
//! - テスト 2（`#[ignore]`・record-only）: 腕（fresh／reuse／reuse_decomposed × キャッシュ off／on と
//!   tape_build_micro）を 20 反復 × 4 ラウンド（腕順をラウンドごとに反転）で計測し、
//!   `DIAG_JSON ` 接頭辞の 1 行 1 JSON を出力する。判定規則は
//!   `docs/perf/logs/metal-tape-build-infer-phase-2114/RULE.txt`（実測前に固定）、集計は
//!   `orchestrate.sh`／`aggregate.py`
//! - fresh の `tape_build` は計測窓の**内側**に入れる。bench-fandhe（#1217 の D4: `make_tape` は
//!   計測窓の外）とは窓の定義が違うため、`infer-reuse-phase-breakdown.md` §10.2 の数値とは
//!   直接比較できない
//! - `facade::Tape` の内部は `pub(crate)` のため、reuse の分解は `fandhe_ai_autodiff::Tape::
//!   new_with_ops(MetalBackendOps)`（`resolve_ops(Device::Metal)` と同じ構成）で写す
//! - `MetalContext` singleton のカウンタを共有するため、`METAL_SINGLETON_LOCK` でファイル内の
//!   テストを直列化する（`infer_device_chain_metal.rs` と同型）
//!
//! 本番コード・tolerance・baseline は変更しない。
#![cfg(target_os = "macos")]

use std::sync::Mutex;
use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tensor};
use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::optim::DeviceParamStore;
use fandhe_ai_backend_metal::fixed_cost_diag::{
    __diagnostic_fixed_cost_counters_snapshot, FixedCostCountersSnapshot,
    override_device_verify_cache_for_scope, verify_device_cached,
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
const ROUNDS: usize = 4;
const ITERS_PER_ROUND: usize = 20;
/// 標準出力の 1 行 1 フェーズ JSON の接頭辞（`orchestrate.sh` が抽出）。
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

fn host_copy(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous 直後は as_slice が Some")
        .to_vec()
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    host_copy(t).into_iter().map(f32::to_bits).collect()
}

fn checksum(v: &[f32]) -> f64 {
    v.iter().map(|x| f64::from(*x)).sum()
}

/// `resolve_ops(Device::Metal)` の存在確認部分と同じ呼び出し（キャッシュ越し）。
fn provider_select_cached() {
    verify_device_cached(|| {
        let provider = MetalDeviceProvider::new();
        provider.select(Device::Metal).map(|_| ())
    })
    .expect("Metal デバイス");
}

/// autodiff レベルの写し: `resolve_ops` 相当の存在確認 → `Tape::new_with_ops` → snapshot →
/// 手組み steps → `predict_device_chain`（`Sequential::predict_resident` の呼び出し列）。
fn autodiff_chain(store: &DeviceParamStore, input: &Tensor<f32>) -> Tensor<f32> {
    provider_select_cached();
    let tape = Tape::new_with_ops(Box::new(MetalBackendOps::new()));
    let leaves = store.snapshot_resident_params(&tape).expect("snapshot");
    let steps = vec![
        (&leaves[0], Some(&leaves[1]), Activation::Relu),
        (&leaves[2], Some(&leaves[3]), Activation::None),
    ];
    store
        .predict_device_chain(&tape, input, &steps)
        .expect("chain")
}

fn fresh_public(model: &Sequential, input: &Tensor<f32>) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(Device::Metal).expect("tape_for");
    let x = tape.var(input);
    model.forward(&tape, &x).expect("forward").to_tensor()
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_infer_decomposition_matches_public_api_bit_exact() {
    let _guard = METAL_SINGLETON_LOCK.lock().unwrap();
    let model = make_model();
    let input = make_input();
    let init_tape = fandhe_ai::tape_for(Device::Metal).expect("tape_for");
    let store = model.init_device_param_store(&init_tape).expect("store");
    drop(init_tape);

    let public_off = model.predict_resident(&store, &input).expect("public off");
    let decomposed_off = autodiff_chain(&store, &input);
    let fresh_off = fresh_public(&model, &input);
    let (public_on, decomposed_on, fresh_on) = {
        let _on = override_device_verify_cache_for_scope(true);
        // 1 回目（probe 実行）と 2 回目（キャッシュヒット）の両方で bit 同一を確認する。
        let first = model.predict_resident(&store, &input).expect("public on 1");
        let second = model.predict_resident(&store, &input).expect("public on 2");
        assert_eq!(
            bits(&first),
            bits(&second),
            "キャッシュ ON の 2 回目が不一致"
        );
        (
            second,
            autodiff_chain(&store, &input),
            fresh_public(&model, &input),
        )
    };

    assert_eq!(
        bits(&public_off),
        bits(&decomposed_off),
        "公開 predict_resident と autodiff レベルの写しが bit 不一致"
    );
    assert_eq!(
        bits(&public_off),
        bits(&public_on),
        "キャッシュ OFF と ON で predict_resident が bit 不一致"
    );
    assert_eq!(
        bits(&decomposed_off),
        bits(&decomposed_on),
        "キャッシュ OFF と ON で分解が bit 不一致"
    );
    assert_eq!(
        bits(&fresh_off),
        bits(&fresh_on),
        "キャッシュ OFF と ON で fresh が bit 不一致"
    );
}

#[derive(Default)]
struct Cell {
    secs: Vec<f64>,
    checksum: Option<u64>,
}

/// 1 反復あたりに換算するカウンタ差分の蓄積（arm ごと）。
#[derive(Default, Clone, Copy)]
struct CounterSum {
    encode: f64,
    command_buffers: f64,
    wait: f64,
    fixed: FixedCostCountersSnapshot,
    n: u64,
}

#[derive(Default)]
struct Rec {
    cells: Vec<((&'static str, &'static str), Cell)>,
    counters: Vec<(&'static str, CounterSum)>,
    on: bool,
}

impl Rec {
    fn cell(&mut self, arm: &'static str, phase: &'static str) -> &mut Cell {
        let i = match self.cells.iter().position(|(k, _)| *k == (arm, phase)) {
            Some(i) => i,
            None => {
                self.cells.push(((arm, phase), Cell::default()));
                self.cells.len() - 1
            }
        };
        &mut self.cells[i].1
    }

    fn push(&mut self, arm: &'static str, phase: &'static str, dt: f64) {
        if self.on {
            self.cell(arm, phase).secs.push(dt);
        }
    }

    fn timed<T>(&mut self, arm: &'static str, phase: &'static str, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        self.push(arm, phase, t.elapsed().as_secs_f64());
        out
    }

    /// 同一 arm の全反復で checksum の bit 一致を要求する（不一致なら panic）。
    fn record_checksum(&mut self, arm: &'static str, s: f64) {
        if !self.on {
            return;
        }
        let bits = s.to_bits();
        let c = self.cell(arm, "iter_total");
        match c.checksum {
            None => c.checksum = Some(bits),
            Some(prev) => assert_eq!(
                prev, bits,
                "arm {arm}: 反復間で checksum が不一致（{prev:016x} vs {bits:016x}）"
            ),
        }
    }

    fn add_counters(
        &mut self,
        arm: &'static str,
        b0: (usize, usize, usize),
        b1: (usize, usize, usize),
        f0: &FixedCostCountersSnapshot,
        f1: &FixedCostCountersSnapshot,
    ) {
        if !self.on {
            return;
        }
        let d = f1.delta_since(f0);
        let i = match self.counters.iter().position(|(k, _)| *k == arm) {
            Some(i) => i,
            None => {
                self.counters.push((arm, CounterSum::default()));
                self.counters.len() - 1
            }
        };
        let s = &mut self.counters[i].1;
        s.encode += b1.0.saturating_sub(b0.0) as f64;
        s.command_buffers += b1.1.saturating_sub(b0.1) as f64;
        s.wait += b1.2.saturating_sub(b0.2) as f64;
        s.fixed.verify_probe_calls += d.verify_probe_calls;
        s.fixed.verify_cache_hits += d.verify_cache_hits;
        s.fixed.host_uploads += d.host_uploads;
        s.fixed.upload_bytes += d.upload_bytes;
        s.fixed.host_downloads += d.host_downloads;
        s.fixed.download_bytes += d.download_bytes;
        s.n += 1;
    }
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

/// 腕の定義。`(名前, キャッシュ ON か)`。`micro` はキャッシュと無関係。
const ARMS: [(&str, bool); 7] = [
    ("fresh_off", false),
    ("fresh_on", true),
    ("reuse_off", false),
    ("reuse_on", true),
    ("reuse_decomposed_off", false),
    ("reuse_decomposed_on", true),
    ("tape_build_micro", false),
];

fn run_fresh(c: &Ctx, rec: &mut Rec, arm: &'static str) {
    let (b0, f0) = (
        batch_counters(),
        __diagnostic_fixed_cost_counters_snapshot(),
    );
    let t_all = Instant::now();
    let tape = rec.timed(arm, "tape_build", || {
        fandhe_ai::tape_for(Device::Metal).expect("tape_for")
    });
    let x = rec.timed(arm, "leaf_register", || tape.var(&c.input));
    let y = rec.timed(arm, "forward", || {
        c.model.forward(&tape, &x).expect("forward")
    });
    let out = rec.timed(arm, "to_tensor", || y.to_tensor());
    let v = rec.timed(arm, "host_copy", || host_copy(&out));
    let s = rec.timed(arm, "checksum", || checksum(&v));
    rec.push(arm, "iter_total", t_all.elapsed().as_secs_f64());
    rec.record_checksum(arm, s);
    rec.add_counters(
        arm,
        b0,
        batch_counters(),
        &f0,
        &__diagnostic_fixed_cost_counters_snapshot(),
    );
}

fn run_reuse(c: &Ctx, rec: &mut Rec, arm: &'static str) {
    let (b0, f0) = (
        batch_counters(),
        __diagnostic_fixed_cost_counters_snapshot(),
    );
    let t_all = Instant::now();
    let out = rec.timed(arm, "predict_resident", || {
        c.model
            .predict_resident(&c.store, &c.input)
            .expect("resident")
    });
    let v = rec.timed(arm, "host_copy", || host_copy(&out));
    let s = rec.timed(arm, "checksum", || checksum(&v));
    rec.push(arm, "iter_total", t_all.elapsed().as_secs_f64());
    rec.record_checksum(arm, s);
    rec.add_counters(
        arm,
        b0,
        batch_counters(),
        &f0,
        &__diagnostic_fixed_cost_counters_snapshot(),
    );
}

fn run_reuse_decomposed(c: &Ctx, rec: &mut Rec, arm: &'static str) {
    let (b0, f0) = (
        batch_counters(),
        __diagnostic_fixed_cost_counters_snapshot(),
    );
    let t_all = Instant::now();
    let t_b = Instant::now();
    rec.timed(arm, "provider_select", provider_select_cached);
    let tape = rec.timed(arm, "tape_new", || {
        Tape::new_with_ops(Box::new(MetalBackendOps::new()))
    });
    rec.push(arm, "tape_build", t_b.elapsed().as_secs_f64());
    let leaves = rec.timed(arm, "snapshot", || {
        c.store.snapshot_resident_params(&tape).expect("snapshot")
    });
    let steps = vec![
        (&leaves[0], Some(&leaves[1]), Activation::Relu),
        (&leaves[2], Some(&leaves[3]), Activation::None),
    ];
    let out = rec.timed(arm, "chain", || {
        c.store
            .predict_device_chain(&tape, &c.input, &steps)
            .expect("chain")
    });
    let v = rec.timed(arm, "host_copy", || host_copy(&out));
    let s = rec.timed(arm, "checksum", || checksum(&v));
    rec.push(arm, "iter_total", t_all.elapsed().as_secs_f64());
    rec.record_checksum(arm, s);
    rec.add_counters(
        arm,
        b0,
        batch_counters(),
        &f0,
        &__diagnostic_fixed_cost_counters_snapshot(),
    );
}

/// `tape_build` の内訳を個別に計測する（常にキャッシュ OFF。IOKit 単独・存在確認全体・
/// `Tape::new_with_ops` 単独・`tape_for` 全体）。`MTLCopyAllDevices` と `name` の分は
/// `provider_select - probe_gpu_core_count` として導出する（facade テストから objc2-metal を
/// 直接呼ばないため）。
fn run_micro(rec: &mut Rec, arm: &'static str) {
    rec.timed(arm, "probe_gpu_core_count", || {
        std::hint::black_box(fandhe_ai_backend_metal::device::probe_gpu_core_count())
    });
    rec.timed(arm, "provider_select", || {
        let provider = MetalDeviceProvider::new();
        std::hint::black_box(provider.select(Device::Metal).expect("select"))
    });
    rec.timed(arm, "tape_new", || {
        std::hint::black_box(Tape::new_with_ops(Box::new(MetalBackendOps::new())))
    });
    rec.timed(arm, "tape_for", || {
        std::hint::black_box(fandhe_ai::tape_for(Device::Metal).expect("tape_for"))
    });
}

fn run_arm(c: &Ctx, rec: &mut Rec, arm: (&'static str, bool)) {
    let _cache = override_device_verify_cache_for_scope(arm.1);
    match arm.0 {
        "fresh_off" | "fresh_on" => run_fresh(c, rec, arm.0),
        "reuse_off" | "reuse_on" => run_reuse(c, rec, arm.0),
        "reuse_decomposed_off" | "reuse_decomposed_on" => run_reuse_decomposed(c, rec, arm.0),
        _ => run_micro(rec, arm.0),
    }
}

/// 実機用 record-only 診断（`orchestrate.sh` が独立プロセスで 5 回起動）。
#[test]
#[ignore = "Metal 実機計測用の record-only 診断（RULE.txt・orchestrate.sh 参照）"]
fn metal_infer_tape_build_fixedcost_phases() {
    let _guard = METAL_SINGLETON_LOCK.lock().unwrap();
    let model = make_model();
    let init_tape = fandhe_ai::tape_for(Device::Metal).expect("tape_for");
    let store = model.init_device_param_store(&init_tape).expect("store");
    drop(init_tape);
    let c = Ctx {
        model,
        store,
        input: make_input(),
    };

    let mut rec = Rec::default();
    for _ in 0..WARMUP {
        for arm in ARMS {
            run_arm(&c, &mut rec, arm);
        }
    }
    rec.on = true;
    for round in 0..ROUNDS {
        // 順序効果を打ち消すためラウンドごとに腕順を反転する。
        let order: Vec<(&'static str, bool)> = if round % 2 == 0 {
            ARMS.to_vec()
        } else {
            ARMS.iter().rev().copied().collect()
        };
        for _ in 0..ITERS_PER_ROUND {
            for arm in &order {
                run_arm(&c, &mut rec, *arm);
            }
        }
    }

    for ((arm, phase), cell) in &rec.cells {
        let q = median_q1_q3(&cell.secs).expect("サンプルは非空・非 NaN");
        let mut sorted = cell.secs.clone();
        sorted.sort_by(f64::total_cmp);
        let line = serde_json::json!({
            "issue": 2114,
            "arm": arm,
            "phase": phase,
            "median_s": q.median,
            "q1_s": q.q1,
            "q3_s": q.q3,
            "min_s": sorted.first(),
            "max_s": sorted.last(),
            "n": cell.secs.len(),
            "checksum_bits": cell.checksum.map(|b| format!("{b:016x}")),
            "target_arch": std::env::consts::ARCH,
            "target_os": std::env::consts::OS,
        });
        println!("{PREFIX}{line}");
    }
    // 1 反復あたりのカウンタ（encode／command buffer／wait・存在確認 probe／ヒット・転送）。
    for (arm, s) in &rec.counters {
        let n = s.n.max(1) as f64;
        let line = serde_json::json!({
            "issue": 2114,
            "arm": arm,
            "phase": "counters",
            "n": s.n,
            "encode_per_iter": s.encode / n,
            "command_buffers_per_iter": s.command_buffers / n,
            "wait_per_iter": s.wait / n,
            "verify_probe_per_iter": s.fixed.verify_probe_calls as f64 / n,
            "verify_hit_per_iter": s.fixed.verify_cache_hits as f64 / n,
            "uploads_per_iter": s.fixed.host_uploads as f64 / n,
            "upload_bytes_per_iter": s.fixed.upload_bytes as f64 / n,
            "downloads_per_iter": s.fixed.host_downloads as f64 / n,
            "download_bytes_per_iter": s.fixed.download_bytes as f64 / n,
        });
        println!("{PREFIX}{line}");
    }
}
