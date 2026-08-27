//! Burn benchmark binary. CPU = NdArray backend, "metal" = Wgpu backend
//! (Metal via wgpu on macOS).
//!
//! Protocol: warmup 20 -> measure 20 (train: 100 SGD steps, first 20 warmup).
//! The measured region ends with host materialization (`into_data()` +
//! element readout) so asynchronous GPU execution cannot leak out of the
//! timing window.

use bench_common::*;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
#[cfg(feature = "cuda")]
use burn::backend::{Cuda, cuda::CudaDevice};
#[cfg(feature = "metal")]
use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn::tensor::activation::relu;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Tensor, TensorData};
use std::time::{Duration, Instant};

const FRAMEWORK: &str = "burn";
const VERSION: &str = "0.21.0";

const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const TRAIN_STEPS: usize = 100;
const TRAIN_WARMUP: usize = 20;
const LR: f32 = 0.01;

fn tensor2<B: Backend>(data: Vec<f32>, shape: [usize; 2], dev: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(data, shape), dev)
}

/// Host-materialize and checksum (forces sync on async backends).
/// Conversion failures are propagated as errors, not panics.
fn checksum<B: Backend, const D: usize>(
    t: Tensor<B, D>,
) -> Result<f64, Box<dyn std::error::Error>> {
    Ok(t.into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("MEASURE_ERROR: into_data to_vec failed: {e:?}"))?
        .iter()
        .map(|&x| x as f64)
        .sum())
}

fn run_gemm<B: Backend>(cli: &Cli, dev: &B::Device) -> Result<(), Box<dyn std::error::Error>> {
    let n = cli.size;
    let a = tensor2::<B>(Xorshift64Star::new(SEED_A).fill_vec(n * n), [n, n], dev);
    let b = tensor2::<B>(Xorshift64Star::new(SEED_B).fill_vec(n * n), [n, n], dev);
    let mut cs = 0.0;

    let one = |cs: &mut f64| -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let c = a.clone().matmul(b.clone());
        *cs = checksum(c)?;
        Ok(start.elapsed())
    };

    for _ in 0..WARMUP_ITERS {
        one(&mut cs)?;
    }
    let mut durations = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        durations.push(one(&mut cs)?);
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
        checksum: cs,
        warmup: WARMUP_ITERS,
        iters: MEASURE_ITERS,
    }
    .emit(&cli.out)?;
    Ok(())
}

struct Mlp<B: Backend> {
    w1: Tensor<B, 2>,
    b1: Tensor<B, 2>,
    w2: Tensor<B, 2>,
    b2: Tensor<B, 2>,
}

impl<B: Backend> Mlp<B> {
    /// Deterministic init from the shared RNG (uniform in [-0.5, 0.5)).
    fn new(dev: &B::Device) -> Self {
        let mut r1 = Xorshift64Star::new(SEED_L1);
        let w1 = tensor2::<B>(r1.fill_vec(D_IN * D_HIDDEN), [D_IN, D_HIDDEN], dev);
        let b1 = tensor2::<B>(vec![0.0; D_HIDDEN], [1, D_HIDDEN], dev);
        let mut r2 = Xorshift64Star::new(SEED_L2);
        let w2 = tensor2::<B>(r2.fill_vec(D_HIDDEN * D_OUT), [D_HIDDEN, D_OUT], dev);
        let b2 = tensor2::<B>(vec![0.0; D_OUT], [1, D_OUT], dev);
        Self { w1, b1, w2, b2 }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(x.matmul(self.w1.clone()) + self.b1.clone());
        h.matmul(self.w2.clone()) + self.b2.clone()
    }
}

fn mlp_inputs<B: Backend>(dev: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
    (
        tensor2::<B>(
            Xorshift64Star::new(SEED_X).fill_vec(BATCH * D_IN),
            [BATCH, D_IN],
            dev,
        ),
        tensor2::<B>(
            Xorshift64Star::new(SEED_Y).fill_vec(BATCH * D_OUT),
            [BATCH, D_OUT],
            dev,
        ),
    )
}

fn run_train<B: AutodiffBackend>(
    cli: &Cli,
    dev: &B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let init = Mlp::<B>::new(dev);
    let mut model = Mlp::<B> {
        w1: init.w1.require_grad(),
        b1: init.b1.require_grad(),
        w2: init.w2.require_grad(),
        b2: init.b2.require_grad(),
    };
    let (x, y) = mlp_inputs::<B>(dev);
    let mut durations = Vec::with_capacity(TRAIN_STEPS);
    let mut last_loss = 0.0f32;

    for _ in 0..TRAIN_STEPS {
        let start = Instant::now();
        let pred = model.forward(x.clone());
        let diff = pred - y.clone();
        let loss = (diff.clone() * diff).mean();
        let grads = loss.backward();
        let step = |p: Tensor<B, 2>,
                    g: Option<Tensor<B::InnerBackend, 2>>|
         -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
            let g = g.ok_or("missing gradient for parameter")?;
            Ok(Tensor::from_inner(p.inner() - g.mul_scalar(LR)).require_grad())
        };
        let g1 = model.w1.grad(&grads);
        let gb1 = model.b1.grad(&grads);
        let g2 = model.w2.grad(&grads);
        let gb2 = model.b2.grad(&grads);
        model = Mlp {
            w1: step(model.w1, g1)?,
            b1: step(model.b1, gb1)?,
            w2: step(model.w2, g2)?,
            b2: step(model.b2, gb2)?,
        };
        // host readout of the loss (sync point ending the step)
        last_loss = loss.into_scalar().elem::<f32>();
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

fn run_infer<B: Backend>(cli: &Cli, dev: &B::Device) -> Result<(), Box<dyn std::error::Error>> {
    let model = Mlp::<B>::new(dev);
    let (x, _) = mlp_inputs::<B>(dev);
    let mut cs = 0.0;

    let one = |cs: &mut f64| -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let out = model.forward(x.clone());
        *cs = checksum(out)?;
        Ok(start.elapsed())
    };

    for _ in 0..WARMUP_ITERS {
        one(&mut cs)?;
    }
    let mut durations = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        durations.push(one(&mut cs)?);
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
        checksum: cs,
        warmup: WARMUP_ITERS,
        iters: MEASURE_ITERS,
    }
    .emit(&cli.out)?;
    Ok(())
}

fn dispatch<B: Backend>(cli: &Cli, dev: &B::Device) -> Result<(), Box<dyn std::error::Error>>
where
    Autodiff<B>: AutodiffBackend<InnerBackend = B> + Backend<Device = B::Device>,
{
    match cli.task.as_str() {
        "gemm" => run_gemm::<B>(cli, dev),
        "train" => run_train::<Autodiff<B>>(cli, dev),
        "infer" => run_infer::<B>(cli, dev),
        other => Err(format!("MEASURE_ERROR: unknown task '{other}'").into()),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_cli()?;
    match cli.device.as_str() {
        "cpu" => dispatch::<NdArray>(&cli, &NdArrayDevice::Cpu),
        #[cfg(feature = "metal")]
        "metal" => dispatch::<Wgpu>(&cli, &WgpuDevice::default()),
        #[cfg(feature = "cuda")]
        "cuda" => dispatch::<Cuda>(&cli, &CudaDevice { index: 0 }),
        other => Err(format!(
            "MEASURE_ERROR: device '{other}' unknown or not compiled in (features: metal={}, cuda={})",
            cfg!(feature = "metal"),
            cfg!(feature = "cuda"),
        )
        .into()),
    }
}
