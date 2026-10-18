//! イシュー #3194: CPU の resident 分岐で使われない `flat_grad`
//! （`DeviceParamStore::step*` 内の `Vec::with_capacity(total_numel)`）を
//! 確保しないことを `TrackingAllocator` の確保ピークで証明する統合テスト。
//!
//! 時間計測ではなく確保量の実測である。MLP 784→256→10 は total_numel =
//! 203,530 要素（f32 で 814,120 B。`docs/perf/cpu-reuse-device-update.md` §2）で、
//! 修正前は 1 step ごとにこの量が無条件に確保されていた。修正後の 2 step 目
//! 以降のピークは bias 勾配の clone や簿記の `Vec`（数 KiB）だけで、閾値は
//! その半分（約 407 KB）とする。
//!
//! `harness = false`（`Cargo.toml` の `[[test]]`）: `TrackingAllocator` の
//! 静的カウンタをプロセス内で共有するため、`crates/autodiff/tests/
//! view_zero_alloc.rs` と同じくスレッドを作らず `main` から順に検査する。

use bench_harness::alloc_tracker::{TrackingAllocator, measure};
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::AdamConfig;
use fandhe_ai::{DeviceParamStore, SgdConfig, Tape, Tensor};

#[global_allocator]
static GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator;

const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const TOTAL_NUMEL: usize = D_IN * D_HIDDEN + D_HIDDEN + D_HIDDEN * D_OUT + D_OUT;
const FLAT_GRAD_BYTES: u64 = (TOTAL_NUMEL * std::mem::size_of::<f32>()) as u64;

fn make_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, 42)
        .expect("Linear 構築")
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, 43)
        .expect("Linear 構築")
}

fn make_data() -> (Tensor<f32>, Tensor<f32>) {
    let x = Xorshift64Star::new(0x3194).fill_vec(BATCH * D_IN);
    let y = Xorshift64Star::new(0x3195).fill_vec(BATCH * D_OUT);
    (
        Tensor::new(x, &[BATCH, D_IN]).expect("形状一致"),
        Tensor::new(y, &[BATCH, D_OUT]).expect("形状一致"),
    )
}

fn new_store(model: &Sequential) -> DeviceParamStore {
    let tape = fandhe_ai::tape();
    let store = model.init_device_param_store(&tape).expect("store");
    let _ = tape.sync_device_param_store_to_host(&store).expect("sync");
    store
}

/// forward → backward（未計測）。勾配と、それを作った tape を返す。
fn fwd_bwd(
    model: &Sequential,
    store: &mut DeviceParamStore,
    x: &Tensor<f32>,
    y: &Tensor<f32>,
) -> (Tape, fandhe_ai::Gradients) {
    let tape = fandhe_ai::tape();
    let xv = tape.var(x);
    let yv = tape.var(y);
    let pred = model.forward_resident(&tape, &xv, store).expect("fwd");
    let loss = pred.mse_loss(&yv).expect("mse");
    let grads = tape.backward_device_param_store(&loss, store).expect("bwd");
    (tape, grads)
}

/// 正のプローブ: 計測器が 814,120 B 級の確保を検出できること。
fn check_probe_detects_flat_grad_sized_alloc() {
    let (v, peak) = measure(|| Vec::<f32>::with_capacity(TOTAL_NUMEL));
    drop(v);
    let peak = peak.expect("TrackingAllocator が有効でピークが取れること");
    assert!(
        peak >= FLAT_GRAD_BYTES,
        "プローブが flat_grad 相当の確保を検出できない: peak={peak}"
    );
}

fn run_step(
    opt: &str,
    tape: &Tape,
    store: &mut DeviceParamStore,
    g: &fandhe_ai::Gradients,
) -> Result<(), fandhe_ai::BackendError> {
    match opt {
        "sgd" => tape.step_device_param_store(store, g, &SgdConfig::new(0.01)),
        _ => tape.step_device_param_store_adam(store, g, &AdamConfig::default()),
    }
}

fn check_resident_step_does_not_alloc_flat_grad(opt: &str) {
    let model = make_model();
    let (x, y) = make_data();
    let mut store = new_store(&model);
    // 初回に遅れて確保される状態（staging・momentum・m/v）を先に確保させる。
    for _ in 0..2 {
        let (tape, grads) = fwd_bwd(&model, &mut store, &x, &y);
        // 前提固定: CPU は weight=Some・bias=None（any_resident == true の経路）。
        let some: Vec<bool> = tape
            .resident_grads_to_host(&store, &grads)
            .expect("resident")
            .iter()
            .map(Option::is_some)
            .collect();
        assert_eq!(some, vec![true, false, true, false], "resident slot 構成");
        run_step(opt, &tape, &mut store, &grads).expect("warmup step");
    }
    let (tape, grads) = fwd_bwd(&model, &mut store, &x, &y);
    let (r, peak) = measure(|| run_step(opt, &tape, &mut store, &grads));
    r.expect("step");
    let peak = peak.expect("確保ピークが取れること（fail-closed）");
    println!("{opt}: step 確保ピーク = {peak} B（flat_grad = {FLAT_GRAD_BYTES} B）");
    assert!(
        peak < FLAT_GRAD_BYTES / 2,
        "{opt}: resident 分岐で flat_grad 相当の確保が起きている: peak={peak}"
    );
}

fn main() {
    check_probe_detects_flat_grad_sized_alloc();
    check_resident_step_does_not_alloc_flat_grad("sgd");
    check_resident_step_does_not_alloc_flat_grad("adam");
    println!("device_param_store_resident_no_flat_grad_alloc: ok");
}
