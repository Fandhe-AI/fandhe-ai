//! fandhe-ai benchmark binary.
//!
//! Protocol: warmup 20 -> measure 20 (train: 100 SGD steps total, first 20
//! treated as warmup, stats over the remaining 80). Each measured iteration
//! builds a fresh tape. The measured region ends with host materialization
//! (`to_tensor()` + element readout) so asynchronous device execution cannot
//! leak out of the timing window.
//!
//! `--mode reuse`（イシュー #925。`gemm` タスクのみ）: 上記の毎回新規 tape
//! プロトコルは fandhe-ai の CUDA/Metal でタイル初期化コスト（CUDA コンテキ
//! スト作成・NVRTC カーネルコンパイル等）を毎計測に含めてしまい、デバイス・
//! グラフを使い回す candle / Burn との比較でフレームワーク間の不公平が生じる
//! （`results/summary.md` 環境 2 の備考）。reuse モードは tape を 1 回だけ
//! 構築し、その初期化コスト（init_s）を「カーネル実行時間」（中央値・Q1/Q3）
//! と分離して記録することで、初期化コストとカーネル実行を切り分けて比較可能
//! にする。

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

/// Host-materialize a Var result and return the raw elements (forces sync).
/// `run_gemm`/`run_gemm_reuse`（イシュー #970）は checksum（全要素和）に
/// 加え要素単位の参照比較（`GemmReference::verify`）が必要なため、
/// `checksum_var` とは別に生の `Vec<f32>` を返す readout を用意する
/// （`checksum_var`/`checksum_tensor` は `run_infer` が引き続き使うため
/// シグネチャを変更しない）。
fn readout_var(v: &fandhe_ai::Var) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let t = v.to_tensor();
    Ok(t.contiguous()
        .as_slice()
        .ok_or("as_slice() returned None after contiguous()")?
        .to_vec())
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
    // 要素単位検証（イシュー #970）の参照値。本体 `backend-cpu::parity::
    // matmul_reference_fma` と同じ FMA 契約（f32 mul_add・逐次 k 昇順）の
    // 参照 GEMM を計測窓の外（warmup 前）で 1 回だけ計算する。
    let a_host = a_data
        .contiguous()
        .as_slice()
        .ok_or("a_data as_slice() returned None")?
        .to_vec();
    let b_host = b_data
        .contiguous()
        .as_slice()
        .ok_or("b_data as_slice() returned None")?
        .to_vec();
    let reference = GemmReference::compute(n, &a_host, &b_host)?;

    let mut checksum = 0.0;
    let mut parity: Option<ParityStats> = None;

    let one = |sync_checksum: &mut f64,
               parity: &mut Option<ParityStats>|
     -> Result<Duration, Box<dyn std::error::Error>> {
        // fresh tape per measurement (no accumulated graph)。計時開始は
        // tape 構築より前に置く（イシュー #925 レビュー指摘）。「fresh は
        // tape/デバイス初期化コスト（CUDA コンテキスト作成・NVRTC カーネル
        // コンパイル等）を毎計測に含む」という上記モジュールコメント・reuse
        // モードとの対比説明が実際の計測範囲と一致するようにするため。
        let start = Instant::now();
        let tape = make_tape(&cli.device)?;
        let a = tape.var(&a_data);
        let b = tape.var(&b_data);
        let c = a.matmul(&b)?;
        // sync: materialize result on host and read elements（従来どおり
        // 計測窓内。checksum の計算コストも従来と変えない）。
        let out = readout_var(&c)?;
        *sync_checksum = out.iter().map(|&x| x as f64).sum();
        let elapsed = start.elapsed();
        // イシュー #965 codex-review 指摘: sync_checksum は毎反復上書きされる
        // ため、ループ後に最後の値だけを検査すると途中反復の縮退を見逃す。
        // burn 側の縮退 checksum 遮断（bench-burn/src/main.rs）と対称に、
        // fandhe-ai 側でも将来同種の不具合が出た場合に壊れた計算の実行時間を
        // 性能値として記録しないよう、checksum 計算直後・warmup を含む
        // 全反復で検査する。
        validate_gemm_checksum(*sync_checksum)?;
        // イシュー #970: 要素単位の複合判定（O(n^2)）は計測窓の外で行う
        // （GEMM 自体は O(n^3) だが、比較コストが計測時間へ混入するのを
        // 避けるため elapsed 取得後に実行する）。反復間の worst-case を
        // 保持し、途中反復の破損（要素の入れ替わり等、checksum では
        // 見逃しうる破損）も見逃さない。
        let stats = reference.verify(&out)?;
        *parity = Some(match parity.take() {
            Some(prev) => prev.worst(stats),
            None => stats,
        });
        Ok(elapsed)
    };

    for _ in 0..WARMUP_ITERS {
        one(&mut checksum, &mut parity)?;
    }
    let mut durations = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        durations.push(one(&mut checksum, &mut parity)?);
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
        parity,
    }
    .emit(&cli.out)?;
    Ok(())
}

/// `--mode reuse` の gemm 計測（イシュー #925）。tape/デバイスを 1 回だけ
/// 構築し、その構築 + 葉 Var 登録 + 初回 matmul + ホスト実体化までの経過を
/// init_s として分離記録したうえで、同一 tape 上で warmup 残り + 計測を回す。
/// 葉 Var（A・B）は 1 回だけ登録して使い回すが、matmul の結果ノードは呼ぶ
/// たびに tape へ蓄積される（N=2048 で約 16 MiB/回 × 40 回 ≒ 640 MiB。
/// N=4096 でも約 2.6 GiB で対象 GPU メモリ内に収まる。README 計測プロトコル
/// 節に明記）。
fn run_gemm_reuse(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let n = cli.size;
    let (a_data, b_data) = gemm_inputs(n)?;
    // イシュー #970: 参照 GEMM は init_s 計測（tape/デバイス初期化コスト）
    // を汚さないよう、init_start より前に計算する。
    let a_host = a_data
        .contiguous()
        .as_slice()
        .ok_or("a_data as_slice() returned None")?
        .to_vec();
    let b_host = b_data
        .contiguous()
        .as_slice()
        .ok_or("b_data as_slice() returned None")?
        .to_vec();
    let reference = GemmReference::compute(n, &a_host, &b_host)?;

    // init_s: tape 構築 + 葉 Var 登録 + 初回 matmul + ホスト実体化までの
    // 経過（CUDA コンテキスト作成・NVRTC コンパイル等の一度きりのコストを
    // すべて含む）。
    let init_start = Instant::now();
    let tape = make_tape(&cli.device)?;
    let a = tape.var(&a_data);
    let b = tape.var(&b_data);
    let c0 = a.matmul(&b)?;
    let out0 = readout_var(&c0)?;
    let mut checksum: f64 = out0.iter().map(|&x| x as f64).sum();
    let init_s = init_start.elapsed().as_secs_f64();
    // イシュー #965 codex-review 指摘: checksum は毎反復上書きされるため、
    // ループ後に最後の値だけを検査すると途中反復の縮退を見逃す。init 計測分
    // を含め reuse 経路（同一 tape を使い回す）でも fresh 経路と同様に
    // checksum 計算直後・全反復で検証する。
    validate_gemm_checksum(checksum)?;
    // イシュー #970: init 計測分の要素単位検証は init_s の外（elapsed 取得後）
    // で行う。以後の反復と worst-case で集約する。
    let mut parity = reference.verify(&out0)?;

    // 残り warmup（1 回は init 計測内で消費済み）+ 計測本体。同一 tape・同一
    // 葉 Var を使い回し、matmul のみを繰り返す。
    let mut one = || -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let c = a.matmul(&b)?;
        let out = readout_var(&c)?;
        checksum = out.iter().map(|&x| x as f64).sum();
        let elapsed = start.elapsed();
        validate_gemm_checksum(checksum)?;
        parity = parity.worst(reference.verify(&out)?);
        Ok(elapsed)
    };
    for _ in 0..WARMUP_ITERS.saturating_sub(1) {
        one()?;
    }
    let mut durations = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        durations.push(one()?);
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
        mode: "reuse",
        init_s: Some(init_s),
        parity: Some(parity),
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
        mode: "fresh",
        init_s: None,
        parity: None,
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
        mode: "fresh",
        init_s: None,
        parity: None,
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

/// task × mode の分岐。reuse モードは受け入れ条件の範囲（gemm のみ）に限定
/// し、train / infer × reuse は MEASURE_ERROR で fail-fast する（イシュー
/// #925 §2.1・§8 のスコープ境界。将来対応が必要な場合は別イシューで追跡）。
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_cli()?;
    match (cli.task.as_str(), cli.mode.as_str()) {
        ("gemm", "fresh") => run_gemm(&cli),
        ("gemm", "reuse") => run_gemm_reuse(&cli),
        ("train", "fresh") => run_train(&cli),
        ("infer", "fresh") => run_infer(&cli),
        (task, "reuse") => Err(format!(
            "MEASURE_ERROR: --mode reuse is not implemented for task '{task}' (gemm only; issue #925)"
        )
        .into()),
        (other, _) => Err(format!("MEASURE_ERROR: unknown task '{other}'").into()),
    }
}
