//! candle (candle-core) benchmark binary.
//!
//! Protocol: warmup 20 -> measure 20 (train: 100 SGD steps, first 20 warmup).
//! The measured region ends with host materialization (`to_vec2()` /
//! `to_scalar()`) so asynchronous Metal execution cannot leak out of the
//! timing window.
//!
//! `--mode reuse`（イシュー #925）: candle は本ハーネスのプロトコル上でも
//! `Device` を一度だけ構築して使い回す設計（`make_device` は毎回呼ぶが
//! 内部状態を持たない値を返すのみ）のため、fandhe-ai 向けの reuse モード
//! （デバイス/tape 初期化コストの分離計測）は対象外。`--mode reuse` は
//! MEASURE_ERROR で fail-fast する（run_all.sh のスイープでは skipped.log
//! に理由付きで記録される）。

use bench_common::*;
use candle_core::{DType, Device, Tensor, Var};
use std::time::{Duration, Instant};

const FRAMEWORK: &str = "candle";
const VERSION: &str = "0.11.0";

const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const TRAIN_STEPS: usize = 100;
const TRAIN_WARMUP: usize = 20;
const LR: f64 = 0.01;

fn make_device(device: &str) -> Result<Device, Box<dyn std::error::Error>> {
    match device {
        "cpu" => Ok(Device::Cpu),
        "metal" => Device::new_metal(0)
            .map_err(|e| format!("MEASURE_ERROR: candle Device::new_metal(0) failed: {e}").into()),
        // new_cuda exists unconditionally; without the `cuda` cargo feature it
        // returns NotCompiledWithCudaSupport (recorded, never fabricated).
        "cuda" => Device::new_cuda(0)
            .map_err(|e| format!("MEASURE_ERROR: candle Device::new_cuda(0) failed: {e}").into()),
        other => Err(format!("MEASURE_ERROR: unknown device '{other}'").into()),
    }
}

/// Host-materialize a 2D tensor and return a checksum (forces sync).
fn checksum2(t: &Tensor) -> Result<f64, Box<dyn std::error::Error>> {
    let rows = t.to_vec2::<f32>()?;
    Ok(rows.iter().flat_map(|r| r.iter()).map(|&x| x as f64).sum())
}

fn run_gemm(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let n = cli.size;
    let dev = make_device(&cli.device)?;
    let a_host = Xorshift64Star::new(SEED_A).fill_vec(n * n);
    let b_host = Xorshift64Star::new(SEED_B).fill_vec(n * n);
    let a = Tensor::from_vec(a_host, (n, n), &dev)?;
    let b = Tensor::from_vec(b_host, (n, n), &dev)?;
    let mut checksum = 0.0;

    let one = |sync_checksum: &mut f64| -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let c = a.matmul(&b)?;
        *sync_checksum = checksum2(&c)?;
        Ok(start.elapsed())
    };

    for _ in 0..WARMUP_ITERS {
        one(&mut checksum)?;
    }
    let mut durations = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        durations.push(one(&mut checksum)?);
    }
    let st = stats(&durations)?;
    Record {
        framework: FRAMEWORK,
        framework_version: VERSION,
        task: "gemm",
        device: &cli.device,
        size: n,
        stats: st,
        gflops: Some(gemm_gflops(n, st.median_s)),
        throughput_per_s: None,
        checksum,
        warmup: WARMUP_ITERS,
        iters: MEASURE_ITERS,
        mode: "fresh",
        init_s: None,
    }
    .emit(&cli.out)?;
    Ok(())
}

struct Mlp {
    w1: Var,
    b1: Var,
    w2: Var,
    b2: Var,
}

impl Mlp {
    /// Deterministic init from the shared RNG (uniform in [-0.5, 0.5)).
    fn new(dev: &Device) -> Result<Self, Box<dyn std::error::Error>> {
        let mut r1 = Xorshift64Star::new(SEED_L1);
        let w1 = Var::from_tensor(&Tensor::from_vec(
            r1.fill_vec(D_IN * D_HIDDEN),
            (D_IN, D_HIDDEN),
            dev,
        )?)?;
        let b1 = Var::from_tensor(&Tensor::zeros((D_HIDDEN,), DType::F32, dev)?)?;
        let mut r2 = Xorshift64Star::new(SEED_L2);
        let w2 = Var::from_tensor(&Tensor::from_vec(
            r2.fill_vec(D_HIDDEN * D_OUT),
            (D_HIDDEN, D_OUT),
            dev,
        )?)?;
        let b2 = Var::from_tensor(&Tensor::zeros((D_OUT,), DType::F32, dev)?)?;
        Ok(Self { w1, b1, w2, b2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let h = x
            .matmul(self.w1.as_tensor())?
            .broadcast_add(self.b1.as_tensor())?
            .relu()?;
        Ok(h.matmul(self.w2.as_tensor())?
            .broadcast_add(self.b2.as_tensor())?)
    }
}

fn mlp_inputs(dev: &Device) -> Result<(Tensor, Tensor), Box<dyn std::error::Error>> {
    let x = Xorshift64Star::new(SEED_X).fill_vec(BATCH * D_IN);
    let y = Xorshift64Star::new(SEED_Y).fill_vec(BATCH * D_OUT);
    Ok((
        Tensor::from_vec(x, (BATCH, D_IN), dev)?,
        Tensor::from_vec(y, (BATCH, D_OUT), dev)?,
    ))
}

fn run_train(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let dev = make_device(&cli.device)?;
    let model = Mlp::new(&dev)?;
    let (x, y) = mlp_inputs(&dev)?;
    let mut durations = Vec::with_capacity(TRAIN_STEPS);
    let mut last_loss = 0.0f32;

    for _ in 0..TRAIN_STEPS {
        let start = Instant::now();
        let pred = model.forward(&x)?;
        let loss = (pred - &y)?.sqr()?.mean_all()?;
        let grads = loss.backward()?;
        for var in [&model.w1, &model.b1, &model.w2, &model.b2] {
            let grad = grads
                .get(var.as_tensor())
                .ok_or("missing gradient for parameter")?;
            var.set(&(var.as_tensor() - (grad * LR)?)?)?;
        }
        // host readout of the loss (sync point ending the step)
        last_loss = loss.to_scalar::<f32>()?;
        durations.push(start.elapsed());
    }

    if !last_loss.is_finite() {
        return Err(format!("MEASURE_ERROR: final loss not finite: {last_loss}").into());
    }
    let measured = &durations[TRAIN_WARMUP..];
    let st = stats(measured)?;
    Record {
        framework: FRAMEWORK,
        framework_version: VERSION,
        task: "train",
        device: &cli.device,
        size: BATCH,
        stats: st,
        gflops: None,
        throughput_per_s: None,
        checksum: last_loss as f64,
        warmup: TRAIN_WARMUP,
        iters: measured.len(),
        mode: "fresh",
        init_s: None,
    }
    .emit(&cli.out)?;
    Ok(())
}

fn run_infer(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let dev = make_device(&cli.device)?;
    let model = Mlp::new(&dev)?;
    let (x, _) = mlp_inputs(&dev)?;
    let mut checksum = 0.0;

    let one = |sync_checksum: &mut f64| -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let out = model.forward(&x)?;
        *sync_checksum = checksum2(&out)?;
        Ok(start.elapsed())
    };

    for _ in 0..WARMUP_ITERS {
        one(&mut checksum)?;
    }
    let mut durations = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        durations.push(one(&mut checksum)?);
    }
    let st = stats(&durations)?;
    Record {
        framework: FRAMEWORK,
        framework_version: VERSION,
        task: "infer",
        device: &cli.device,
        size: BATCH,
        stats: st,
        gflops: None,
        throughput_per_s: Some(1.0 / st.median_s),
        checksum,
        warmup: WARMUP_ITERS,
        iters: MEASURE_ITERS,
        mode: "fresh",
        init_s: None,
    }
    .emit(&cli.out)?;
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

/// `--mode reuse` は対象外（モジュールコメント参照。イシュー #925）。
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_cli()?;
    if cli.mode == "reuse" {
        return Err(
            "MEASURE_ERROR: --mode reuse is not applicable to candle (device reuse is already \
             the default API design; issue #925)"
                .into(),
        );
    }
    match cli.task.as_str() {
        "gemm" => run_gemm(&cli),
        "train" => run_train(&cli),
        "infer" => run_infer(&cli),
        other => Err(format!("MEASURE_ERROR: unknown task '{other}'").into()),
    }
}
