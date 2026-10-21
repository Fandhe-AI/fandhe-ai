//! イシュー #3189（`docs/perf/infer-reuse-phase-breakdown.md` §12）: GB10 の CUDA infer reuse
//! （`Sequential::predict_resident`）の上位固定費の分解診断（上位層）。
//!
//! `cpu_predict_resident_fixedcost_diag.rs`（CPU）と同型で、`predict_resident` の呼び出し経路
//! （tape 生成 → `snapshot_resident_params` → チェーン steps 構築 → `predict_device_chain`）を
//! facade の外側から写して区間に分ける。backend 呼び出し部分（H2D・カーネル発行・同期 D2H）の
//! 内訳は `crates/backend-cuda/tests/infer_chain_phase_diag_3189.rs`（下位層）が担当する。
//!
//! - `public` arm: `Sequential::predict_resident` ＋ host 読み出し（bench-fandhe reuse と同じ呼び出し列）。
//! - `decomposed` arm: `tape` → `snapshot` → `steps` → `chain`（`predict_device_chain` 全体）→ `readout`。
//! - `paired/residual` = 反復ごとに `public − Σ decomposed` を取ってから run 内の中央値を取る
//!   （両 arm を同一反復内で実行し、ラウンドごとに順序を反転する。2116 の上位診断と同じ方式）。
//!
//! 構成: CPU の bit 一致テストは `#[ignore]` なしで CI 実行し、CUDA の計測は `#[ignore]`。
//! 両 arm の出力は計測窓の外で `f32::to_bits` 比較する（hard assert）。出力は `DIAG_JSON` 行
//! （1 行 1 区間。`orchestrate_gb10.sh` が抽出し `aggregate.py` が集計する）。
//! 診断のみ。本番コード・tolerance・baseline は変更しない。実行は `--test-threads=1`。

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, DeviceParamStore, Tensor};
use fandhe_ai_autodiff::Tape as InnerTape;
use fandhe_ai_tensor_core::Activation;

const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const WARMUP: usize = 20;
const ROUNDS: usize = 4;
const ITERS_PER_ROUND: usize = 20;
const PREFIX: &str = "DIAG_JSON ";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Cpu,
    Cuda,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Cpu => "cpu",
            Kind::Cuda => "cuda",
        }
    }

    fn device(self) -> Device {
        match self {
            Kind::Cpu => Device::Cpu,
            Kind::Cuda => Device::Cuda(0),
        }
    }

    /// facade の resolve_ops と同じ構成の autodiff tape（`decomposed` arm 用）。
    fn inner_tape(self) -> InnerTape {
        match self {
            Kind::Cpu => {
                InnerTape::new_with_ops(Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()))
            }
            Kind::Cuda => {
                InnerTape::new_with_ops(Box::new(fandhe_ai_backend_cuda::CudaBackendOps::new(0)))
            }
        }
    }
}

fn make_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, 0x1111_1111)
        .expect("Linear 構築")
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, 0x2222_2222)
        .expect("Linear 構築")
}

fn make_input() -> Tensor<f32> {
    Tensor::new(
        Xorshift64Star::new(0x9000).fill_vec(BATCH * D_IN),
        &[BATCH, D_IN],
    )
    .expect("形状とデータ長は一致させている")
}

fn host_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous 直後は as_slice が Some")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn bits_hash(bits: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bits {
        for byte in b.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let out = f();
    (out, t.elapsed().as_secs_f64())
}

fn new_store(kind: Kind, model: &Sequential) -> DeviceParamStore {
    let tape = fandhe_ai::tape_for(kind.device()).expect("tape_for");
    model.init_device_param_store(&tape).expect("store")
}

/// `public` arm の 1 反復。`(出力, 全体秒)`。
fn public_iter(
    model: &Sequential,
    store: &DeviceParamStore,
    input: &Tensor<f32>,
) -> (Vec<u32>, f64) {
    let (out, dt) = timed(|| {
        let y = model.predict_resident(store, input).expect("public");
        host_bits(&y)
    });
    (out, dt)
}

/// `decomposed` arm の 1 反復の区間。
#[derive(Default)]
struct Parts {
    tape: f64,
    snapshot: f64,
    steps: f64,
    chain: f64,
    readout: f64,
}

impl Parts {
    fn sum(&self) -> f64 {
        self.tape + self.snapshot + self.steps + self.chain + self.readout
    }
}

fn decomposed_iter(kind: Kind, store: &DeviceParamStore, input: &Tensor<f32>) -> (Vec<u32>, Parts) {
    let mut p = Parts::default();
    let (tape, dt) = timed(|| kind.inner_tape());
    p.tape = dt;
    let (leaves, dt) = timed(|| store.snapshot_resident_params(&tape).expect("snapshot"));
    p.snapshot = dt;
    let (steps, dt) = timed(|| {
        vec![
            (&leaves[0], Some(&leaves[1]), Activation::Relu),
            (&leaves[2], Some(&leaves[3]), Activation::None),
        ]
    });
    p.steps = dt;
    let (y, dt) = timed(|| {
        store
            .predict_device_chain(&tape, input, &steps)
            .expect("chain")
    });
    p.chain = dt;
    let (bits, dt) = timed(|| host_bits(&y));
    p.readout = dt;
    (bits, p)
}

#[derive(Default)]
struct Series {
    public: Vec<f64>,
    tape: Vec<f64>,
    snapshot: Vec<f64>,
    steps: Vec<f64>,
    chain: Vec<f64>,
    readout: Vec<f64>,
    residual: Vec<f64>,
}

fn run_series(kind: Kind, warmup: usize, rounds: usize, iters: usize) -> (Series, u64) {
    let model = make_model();
    let input = make_input();
    let store = new_store(kind, &model);
    let mut s = Series::default();
    let mut reference: Option<Vec<u32>> = None;
    let total = warmup + rounds * iters;
    for it in 0..total {
        let record = it >= warmup;
        let round = if record { (it - warmup) / iters } else { 0 };
        // ラウンドごとに arm の順序を反転する。
        let (pub_out, pub_dt, dec_out, parts);
        if round % 2 == 0 {
            let (a, b) = public_iter(&model, &store, &input);
            let (c, d) = decomposed_iter(kind, &store, &input);
            (pub_out, pub_dt, dec_out, parts) = (a, b, c, d);
        } else {
            let (c, d) = decomposed_iter(kind, &store, &input);
            let (a, b) = public_iter(&model, &store, &input);
            (pub_out, pub_dt, dec_out, parts) = (a, b, c, d);
        }
        // 計測窓の外で bit 一致を hard assert する。
        assert_eq!(pub_out, dec_out, "public と decomposed が bit 不一致");
        match &reference {
            None => reference = Some(pub_out),
            Some(r) => assert_eq!(r, &pub_out, "反復間で出力が bit 不一致"),
        }
        if record {
            s.public.push(pub_dt);
            s.tape.push(parts.tape);
            s.snapshot.push(parts.snapshot);
            s.steps.push(parts.steps);
            s.chain.push(parts.chain);
            s.readout.push(parts.readout);
            s.residual.push(pub_dt - parts.sum());
        }
    }
    let checksum = bits_hash(&reference.expect("反復 1 回以上"));
    (s, checksum)
}

fn emit(kind: Kind, arm: &str, phase: &str, samples: &[f64]) {
    let q = median_q1_q3(samples).expect("標本は非 NaN");
    println!(
        "{PREFIX}{{\"backend\":\"{}\",\"arm\":\"{arm}\",\"phase\":\"{phase}\",\"n\":{},\"median_s\":{:.9},\"q1_s\":{:.9},\"q3_s\":{:.9}}}",
        kind.label(),
        samples.len(),
        q.median,
        q.q1,
        q.q3
    );
}

#[test]
fn cuda_infer_reuse_decomposition_bit_identical_cpu() {
    // CI 実行可能な CPU で写しの bit 一致と系列の健全性を確認する。
    let (s, checksum) = run_series(Kind::Cpu, 2, 2, 2);
    assert_eq!(s.public.len(), 4);
    assert_ne!(checksum, 0);
}

#[test]
#[ignore = "CUDA 実機計測用の record-only 診断（RULE.txt・orchestrate_gb10.sh 参照）"]
fn cuda_infer_reuse_phases() {
    // 未設定または cuda のみ受け付ける（CPU 混入の拒否）。
    match std::env::var("INFER_DIAG_KIND").as_deref() {
        Err(_) | Ok("cuda") => {}
        Ok(other) => panic!("INFER_DIAG_KIND={other} は不可（CUDA 実測のみ）"),
    }
    let kind = Kind::Cuda;
    let (s, checksum) = run_series(kind, WARMUP, ROUNDS, ITERS_PER_ROUND);
    emit(kind, "public", "predict_resident", &s.public);
    emit(kind, "decomposed", "tape", &s.tape);
    emit(kind, "decomposed", "snapshot", &s.snapshot);
    emit(kind, "decomposed", "steps", &s.steps);
    emit(kind, "decomposed", "chain", &s.chain);
    emit(kind, "decomposed", "readout", &s.readout);
    emit(kind, "paired", "residual", &s.residual);
    println!(
        "{PREFIX}{{\"backend\":\"{}\",\"arm\":\"meta\",\"phase\":\"checksum\",\"checksum\":{checksum}}}",
        kind.label()
    );
}
