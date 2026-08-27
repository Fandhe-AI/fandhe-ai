//! fandhe-ai benchmark binary.
//!
//! Protocol: warmup 20 -> measure 20 (train: 100 SGD steps total, first 20
//! treated as warmup, stats over the remaining 80). Each measured iteration
//! builds a fresh tape. The measured region ends with host materialization
//! (`to_tensor()` + element readout) so asynchronous device execution cannot
//! leak out of the timing window.

use bench_common::*;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tape, Tensor};
use std::time::{Duration, Instant};

const FRAMEWORK: &str = "fandhe-ai";
const VERSION: &str = "0.3.0";

const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const TRAIN_STEPS: usize = 100;
const TRAIN_WARMUP: usize = 20;
const LR: f32 = 0.01;

fn make_tape(device: &str) -> Result<Tape, Box<dyn std::error::Error>> {
    match device {
        "cpu" => Ok(fandhe_ai::tape()),
        // Device::Metal exists only on macOS (cfg-gated in fandhe-ai).
        #[cfg(target_os = "macos")]
        "metal" => fandhe_ai::tape_for(Device::Metal).map_err(|e| {
            format!("MEASURE_ERROR: fandhe-ai tape_for(Device::Metal) failed: {e}").into()
        }),
        #[cfg(not(target_os = "macos"))]
        "metal" => Err("MEASURE_ERROR: Device::Metal is macOS-only".into()),
        // fandhe-ai selects backends via cfg + runtime probing (cudarc dynamic
        // load), not cargo features: fail-fast with BackendError when absent.
        "cuda" => fandhe_ai::tape_for(Device::Cuda(0)).map_err(|e| {
            format!("MEASURE_ERROR: fandhe-ai tape_for(Device::Cuda(0)) failed: {e}").into()
        }),
        other => Err(format!("MEASURE_ERROR: unknown device '{other}'").into()),
    }
}

/// Host-materialize a Var result and return a checksum (forces sync).
fn checksum_var(v: &fandhe_ai::Var) -> Result<f64, Box<dyn std::error::Error>> {
    let t = v.to_tensor();
    let slice = t
        .contiguous()
        .as_slice()
        .ok_or("as_slice() returned None after contiguous()")?
        .to_vec();
    Ok(slice.iter().map(|&x| x as f64).sum())
}

fn checksum_tensor(t: &Tensor<f32>) -> Result<f64, Box<dyn std::error::Error>> {
    let slice = t
        .contiguous()
        .as_slice()
        .ok_or("as_slice() returned None after contiguous()")?
        .to_vec();
    Ok(slice.iter().map(|&x| x as f64).sum())
}

fn gemm_inputs(n: usize) -> Result<(Tensor<f32>, Tensor<f32>), Box<dyn std::error::Error>> {
    let a = Xorshift64Star::new(SEED_A).fill_vec(n * n);
    let b = Xorshift64Star::new(SEED_B).fill_vec(n * n);
    Ok((Tensor::new(a, &[n, n])?, Tensor::new(b, &[n, n])?))
}

fn run_gemm(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let n = cli.size;
    let (a_data, b_data) = gemm_inputs(n)?;
    let mut checksum = 0.0;

    let one = |sync_checksum: &mut f64| -> Result<Duration, Box<dyn std::error::Error>> {
        // fresh tape per measurement (no accumulated graph)
        let tape = make_tape(&cli.device)?;
        let a = tape.var(&a_data);
        let b = tape.var(&b_data);
        let start = Instant::now();
        let c = a.matmul(&b)?;
        // sync: materialize result on host and read elements
        *sync_checksum = checksum_var(&c)?;
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
    }
    .emit(&cli.out)?;
    Ok(())
}

fn mlp_data() -> Result<(Tensor<f32>, Tensor<f32>), Box<dyn std::error::Error>> {
    let x = Xorshift64Star::new(SEED_X).fill_vec(BATCH * D_IN);
    let y = Xorshift64Star::new(SEED_Y).fill_vec(BATCH * D_OUT);
    Ok((
        Tensor::new(x, &[BATCH, D_IN])?,
        Tensor::new(y, &[BATCH, D_OUT])?,
    ))
}

fn build_model() -> Result<Sequential, Box<dyn std::error::Error>> {
    Ok(Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)?
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)?)
}

fn run_train(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let mut model = build_model()?;
    let (x_data, y_data) = mlp_data()?;
    let mut durations = Vec::with_capacity(TRAIN_STEPS);
    let mut last_loss = 0.0f32;

    for _ in 0..TRAIN_STEPS {
        let start = Instant::now();
        let updated: Vec<Tensor<f32>> = {
            // fresh tape per step
            let tape = make_tape(&cli.device)?;
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);
            let pred = bound.forward(&tape, &x)?;
            let loss = pred.mse_loss(&y)?;
            // host readout of the loss (sync point inside the step)
            last_loss = loss
                .to_tensor()
                .get(&[])
                .ok_or("loss should be a scalar with shape []")?;
            let grads = tape.backward(&loss)?;
            let grad_refs = bound.trainable_grads(&grads)?;
            let param_refs = model.trainable_parameters();
            let mut next = Vec::with_capacity(param_refs.len());
            for (param, grad) in param_refs.iter().zip(grad_refs.iter()) {
                let p = param
                    .contiguous()
                    .as_slice()
                    .ok_or("param as_slice None")?
                    .to_vec();
                let g = grad
                    .contiguous()
                    .as_slice()
                    .ok_or("grad as_slice None")?
                    .to_vec();
                let upd: Vec<f32> = p.iter().zip(g.iter()).map(|(p, g)| p - LR * g).collect();
                next.push(Tensor::from_slice(&upd, param.shape())?);
            }
            next
        };
        model.apply_parameters(updated)?;
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
    }
    .emit(&cli.out)?;
    Ok(())
}

fn run_infer(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let model = build_model()?;
    let (x_data, _) = mlp_data()?;
    let mut checksum = 0.0;

    let one = |sync_checksum: &mut f64| -> Result<Duration, Box<dyn std::error::Error>> {
        match cli.device.as_str() {
            "cpu" => {
                // predict() builds an internal default (CPU) tape
                let start = Instant::now();
                let out = model.predict(&x_data)?;
                *sync_checksum = checksum_tensor(&out)?;
                Ok(start.elapsed())
            }
            _ => {
                // explicit tape on the requested device + forward + host sync
                let tape = make_tape(&cli.device)?;
                let start = Instant::now();
                let x = tape.var(&x_data);
                let out = model.forward(&tape, &x)?;
                *sync_checksum = checksum_var(&out)?;
                Ok(start.elapsed())
            }
        }
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

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_cli()?;
    match cli.task.as_str() {
        "gemm" => run_gemm(&cli),
        "train" => run_train(&cli),
        "infer" => run_infer(&cli),
        other => Err(format!("MEASURE_ERROR: unknown task '{other}'").into()),
    }
}
