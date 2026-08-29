//! Burn benchmark binary. CPU = NdArray backend, "metal" = Wgpu backend
//! (Metal via wgpu on macOS).
//!
//! Protocol: warmup 20 -> measure 20 (train: 100 SGD steps, first 20 warmup).
//! The measured region ends with host materialization (`into_data()` +
//! element readout) so asynchronous GPU execution cannot leak out of the
//! timing window.
//!
//! `--mode reuse`（イシュー #925）: Burn はバックエンドの `Device` を
//! 一度だけ構築して使い回す設計（`dispatch` の呼び出し元で構築した
//! `B::Device` をタスク関数へ借用で渡すのみ）のため、fandhe-ai 向けの
//! reuse モード（デバイス/tape 初期化コストの分離計測）は対象外。
//! `--mode reuse` は MEASURE_ERROR で fail-fast する。

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

/// Host-materialize and return the raw elements (forces sync). `run_gemm`
/// （イシュー #970）は checksum に加え要素単位の参照比較
/// （`GemmReference::verify`）が必要なため、`checksum` とは別に生の
/// `Vec<f32>` を返す readout を用意する（`checksum` は `run_infer` が
/// 引き続き使うためシグネチャを変更しない）。
fn readout<B: Backend, const D: usize>(
    t: Tensor<B, D>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    t.into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("MEASURE_ERROR: into_data to_vec failed: {e:?}").into())
}

fn run_gemm<B: Backend>(cli: &Cli, dev: &B::Device) -> Result<(), Box<dyn std::error::Error>> {
    let n = cli.size;
    // イシュー #970 codex-review 指摘（PR #978・P1）: `n * n`（要素数）を
    // 入力ベクタ生成前に `gemm_element_count` で検証する（`--size` は
    // 未検証な CLI 入力であり、無検証の乗算は debug で panic・release では
    // wrap した長さで確保・使用してしまう）。
    let len = gemm_element_count(n)?;
    let a_host = Xorshift64Star::new(SEED_A).fill_vec(len);
    let b_host = Xorshift64Star::new(SEED_B).fill_vec(len);
    // イシュー #970: 参照 GEMM は `tensor2` に渡す前の host Vec の clone から
    // 計算する（本体の FMA 契約と同じ参照。計測窓の外・warmup 前に 1 回だけ）。
    let reference = GemmReference::compute(n, &a_host, &b_host)?;
    let a = tensor2::<B>(a_host, [n, n], dev);
    let b = tensor2::<B>(b_host, [n, n], dev);
    let mut cs = 0.0;
    let mut parity: Option<ParityStats> = None;

    // イシュー #965 codex-review 指摘: cs は毎反復上書きされるため、ループ後に
    // 最後の値だけを検査すると途中反復が縮退（全ゼロ/非有限）していても
    // 最終反復が正常なら見逃す。壊れた計算の実行時間を性能値として記録しない
    // 契約（`.claude/rules/security.md` A08）を反復単位で満たすため、
    // checksum 計算直後・warmup を含む全反復で検証する。
    let one = |cs: &mut f64,
               parity: &mut Option<ParityStats>|
     -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let c = a.clone().matmul(b.clone());
        // sync: materialize result on host and read elements（従来どおり
        // 計測窓内）。
        let out = readout(c)?;
        *cs = out.iter().map(|&x| x as f64).sum();
        let elapsed = start.elapsed();
        // イシュー #965: Burn(wgpu) Metal 経路は N>=512 で結果テンソル全ゼロを
        // 返す upstream 既知バグを持つ（tracel-ai/burn#4966 →
        // tracel-ai/cubek#283。`docs/perf/burn-wgpu-metal-gemm-zero-result.md`）。
        validate_gemm_checksum(*cs)?;
        // イシュー #970: 要素単位検証は計測窓の外で行う（O(n^2)。GEMM 自体
        // の O(n^3) に対する比較コストの混入を避けるため）。
        let stats = reference.verify(&out)?;
        *parity = Some(match parity.take() {
            Some(prev) => prev.worst(stats),
            None => stats,
        });
        Ok(elapsed)
    };

    for _ in 0..WARMUP_ITERS {
        one(&mut cs, &mut parity)?;
    }
    let mut durations = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        durations.push(one(&mut cs, &mut parity)?);
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
        mode: "fresh",
        init_s: None,
        parity,
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
        mode: "fresh",
        init_s: None,
        parity: None,
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
        mode: "fresh",
        init_s: None,
        parity: None,
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

/// `--mode reuse` は対象外（モジュールコメント参照。イシュー #925）。
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_cli()?;
    if cli.mode == "reuse" {
        return Err(
            "MEASURE_ERROR: --mode reuse is not applicable to burn (device reuse is already \
             the default API design; issue #925)"
                .into(),
        );
    }
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
