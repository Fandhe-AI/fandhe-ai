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

/// Host-materialize a 2D tensor and return the row-major `Vec<Vec<f32>>`
/// (forces sync). `run_gemm`（イシュー #970）は checksum に加え要素単位の
/// 参照比較（`GemmReference::verify`）が必要なため、`checksum2` とは別に
/// 生の行データを返す readout を用意する（`checksum2` は `run_infer` が
/// 引き続き使うためシグネチャを変更しない）。
///
/// イシュー #970 codex-review 指摘（PR #978）: 以前はここで
/// `rows.into_iter().flatten().collect::<Vec<f32>>()` により行データを
/// フラット化していたが、`to_vec2` が既に返す `Vec<Vec<f32>>` に加えて
/// 新規に O(n^2) の Vec を確保・コピーする分だけ、計測窓内（`elapsed`
/// 取得前）のコストが従来の `checksum2`（`to_vec2` の結果を
/// `flat_map`/`iter` で走査するのみ・追加確保なし）より増えてしまう。
/// フラット化（`GemmReference::verify` が要求する `&[f32]`）は `elapsed`
/// 取得後に行う（呼び出し元 `run_gemm` 参照）ことで、計測窓内のコストを
/// `checksum2` と同一に保つ。
fn readout2(t: &Tensor) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    Ok(t.to_vec2::<f32>()?)
}

fn run_gemm(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let n = cli.size;
    let dev = make_device(&cli.device)?;
    // イシュー #1042: `--tf32`（`dispatch` が事前に `task == "gemm" &&
    // device == "cuda"` を検証済み）は candle 0.11 の公開プロセスグローバル
    // スイッチ（`cuda_backend::set_gemm_reduced_precision_f32`。既定
    // `false` = FP32 厳密）を有効化する。burn の TF32 既定降格
    // （`burn-cuda-tf32.md`）との条件差を埋め、fandhe-ai opt-in 実装
    // （`docs/cuda-tf32-optin-api-decision.md`）との同条件比較を可能にする。
    // 本バイナリはタスク 1 回の計測ごとにプロセスを起動する設計（`run_all*.sh`
    // のスイープ方式）のため、有効化後に明示的に無効へ戻す必要はない。
    // `cuda` cargo feature が無効なビルド（既定 `metal`）では
    // `candle_core::cuda_backend` モジュール自体が存在しない
    // （candle-core `#[cfg(feature = "cuda")]`）ため、`--features cuda`
    // ビルドを要求する型付きエラーへ fail-closed する（`dispatch` の
    // task/device 検証を通過していても、`cuda` feature 自体が無効な
    // ビルドでは `make_device("cuda")` が `NotCompiledWithCudaSupport` を
    // 返すため通常この分岐には到達しないが、コンパイル可能性のため両方の
    // cfg 分岐を用意する）。
    if cli.tf32 {
        #[cfg(feature = "cuda")]
        {
            candle_core::cuda_backend::set_gemm_reduced_precision_f32(true);
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Err(
                "MEASURE_ERROR: --tf32 requires building bench-candle with --features cuda \
                 --no-default-features (issue #1042)"
                    .into(),
            );
        }
    }
    // イシュー #970 codex-review 指摘（PR #978・P1）: `n * n`（要素数）を
    // 入力ベクタ生成前に `gemm_element_count` で検証する（`--size` は
    // 未検証な CLI 入力であり、無検証の乗算は debug で panic・release では
    // wrap した長さで確保・使用してしまう）。
    let len = gemm_element_count(n)?;
    let a_host = Xorshift64Star::new(SEED_A).fill_vec(len);
    let b_host = Xorshift64Star::new(SEED_B).fill_vec(len);
    // イシュー #970: 参照 GEMM は `Tensor::from_vec` が host Vec を消費する
    // 前に、その clone から計算する（本体の FMA 契約と同じ参照。計測窓の
    // 外・warmup 前に 1 回だけ）。
    let reference = GemmReference::compute(n, &a_host, &b_host)?;
    let a = Tensor::from_vec(a_host, (n, n), &dev)?;
    let b = Tensor::from_vec(b_host, (n, n), &dev)?;
    let mut checksum = 0.0;
    let mut parity: Option<ParityStats> = None;

    // イシュー #965 codex-review 指摘: checksum は毎反復上書きされるため、
    // ループ後に最後の値だけを検査すると途中反復の縮退を見逃す。
    // burn 側の縮退 checksum 遮断（bench-burn/src/main.rs）と対称に、
    // candle 側でも将来同種の不具合が出た場合に壊れた計算の実行時間を
    // 性能値として記録しないよう、checksum 計算直後・warmup を含む
    // 全反復で検査する。
    let one = |sync_checksum: &mut f64,
               parity: &mut Option<ParityStats>|
     -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let c = a.matmul(&b)?;
        // sync: materialize result on host and read elements（従来どおり
        // 計測窓内）。checksum は `checksum2` と同じく `rows` を走査するのみ
        // で、新規の平坦化 Vec は確保しない（PR #978 codex-review 指摘）。
        let rows = readout2(&c)?;
        *sync_checksum = rows.iter().flat_map(|r| r.iter()).map(|&x| x as f64).sum();
        let elapsed = start.elapsed();
        validate_gemm_checksum(*sync_checksum)?;
        // イシュー #970: 要素単位検証は計測窓の外で行う（O(n^2)。GEMM 自体
        // の O(n^3) に対する比較コストの混入を避けるため）。行データの平坦化
        // （`GemmReference::verify` が要求する `&[f32]`）もここで行う。
        let out: Vec<f32> = rows.into_iter().flatten().collect();
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
        tf32: cli.tf32,
        managed: cli.managed,
    }
    .emit(&cli.out)?;
    Ok(())
}

/// イシュー #1103: `docs/perf/metal-gemm-bottleneck-rediagnosis.md` §7.1 が
/// 未確定のまま残した「candle 側の転送分離測定」を埋めるタスク。
///
/// fandhe 側（`crates/backend-metal/examples/gemm_bench.rs`）は
/// `dispatch_auto`（転送込み: 毎反復ホスト→デバイスアップロード＋カーネル
/// 実行＋readback）と `dispatch_tiled_prepared`（転送除外: バッファを
/// ループ外で 1 回だけアップロードし、計測対象をディスパッチ〈エンコード＋
/// コマンドバッファ完了待ち〉のみに絞る）の 2 境界を同一プロセス内で比較
/// している（同 doc §4.1・§4.3）。本関数はこの 2 境界を candle 側でも
/// 同一プロセス内で再現し、`gemm_transfer_incl`（転送込み）・
/// `gemm_transfer_excl`（転送除外）という 2 つの `task` 名の `Record` を
/// 1 回のプロセス実行で出力する（`run_gemm` の既存 `task: "gemm"` は
/// 変更しない。既存の候補比較・summarize.py・candle 比ゲート〈#1082〉は
/// 新 task 名を扱わないため影響しない）。
///
/// - **転送込み境界**: 計測クロージャ内で `Tensor::from_slice`（A・B の
///   毎反復アップロード。`&[f32]` 参照を渡すのみでホスト側 clone を
///   発生させない。fandhe 側 `dispatch_auto`/`dispatch_variant` も
///   `&[f32]` 参照のみを受け取るため、ここで `Tensor::from_vec` +
///   事前 `clone()` を使うと candle 側だけに余分な host メモリコピーが
///   計測窓へ混入し比較の対称性が崩れる。イシュー #1103 Review 指摘）
///   → `matmul` → `to_vec2` readback を行う。checksum 集計は readback
///   完了後（計測窓外）で行い、転送除外境界側の計測方針と揃える。
///   fandhe `dispatch_auto` と同じ境界。
/// - **転送除外境界**: A・B はループ外で 1 回だけデバイス転送する。
///   計測クロージャ内は `matmul` と `Device::synchronize()`（candle 0.11
///   の公開 API。`~/.cargo/registry/.../candle-core-0.11.0/src/device.rs:511`）
///   のみを計測し、readback は計測窓の外で 1 回行う（fandhe
///   `dispatch_tiled_prepared` が readback を計測対象に含めないのと同じ
///   境界）。checksum 検証（縮退遮断・`validate_gemm_checksum`）は両境界
///   とも実施するが、除外境界側は計測窓外の readback 結果に対して行う。
///
/// `--mode`/`--phases`/`--tf32` は `dispatch` の既存 fail-fast 分岐
/// （`--mode reuse` 拒否・`--phases` 拒否・`--tf32` は gemm×cuda 限定）を
/// そのまま適用する（本タスクは metal 限定の追加検証のため、それらの
/// 分岐を独自に緩めない）。
fn run_gemm_transfer_split(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    if cli.device != "metal" {
        return Err(format!(
            "MEASURE_ERROR: gemm-transfer-split は --device metal 限定（イシュー #1103。\
             docs/perf/metal-gemm-bottleneck-rediagnosis.md §7.1 の candle 転送分離測定は \
             Metal 頭打ち診断の追補のため、他デバイスへは拡張しない）: got device='{}'",
            cli.device
        )
        .into());
    }
    let n = cli.size;
    let dev = make_device(&cli.device)?;
    let len = gemm_element_count(n)?;

    // --- 転送込み境界（fandhe dispatch_auto と同一。run_gemm と同じ入力
    // 生成・checksum 縮退検査・参照比較を踏襲する） ---
    {
        let a_host = Xorshift64Star::new(SEED_A).fill_vec(len);
        let b_host = Xorshift64Star::new(SEED_B).fill_vec(len);
        let reference = GemmReference::compute(n, &a_host, &b_host)?;
        let mut checksum = 0.0;
        let mut parity: Option<ParityStats> = None;

        let one = |sync_checksum: &mut f64,
                   parity: &mut Option<ParityStats>|
         -> Result<Duration, Box<dyn std::error::Error>> {
            let start = Instant::now();
            // 毎反復ホスト→デバイス転送（fandhe dispatch_auto のアップロード
            // 込み境界と同一）。`Tensor::from_slice` は `&[f32]` 参照を
            // 受け取り device へアップロードする（`Tensor::from_vec` と違い
            // Vec を消費しないため呼び出し側の事前 `clone()` が不要）。
            // fandhe 側の `dispatch_auto`/`dispatch_variant` も `&[f32]`
            // 参照を受け取るのみでホスト側 clone を発生させないため、ここで
            // `from_vec(...clone())` を使うと candle 側だけに余分な host
            // メモリコピー（A・B 計 128MB @ N=4096）が計測窓内に混入し
            // 比較の対称性が崩れる（イシュー #1103 Review 指摘）。
            let a = Tensor::from_slice(&a_host, (n, n), &dev)?;
            let b = Tensor::from_slice(&b_host, (n, n), &dev)?;
            let c = a.matmul(&b)?;
            let rows = readout2(&c)?;
            // checksum 集計（O(n²) の `sum()`）は fandhe 側の計測境界
            // （アップロード＋カーネル実行＋readback のみ）に対応物がない
            // ホスト側後処理のため、readback 完了時点で `elapsed` を確定し
            // 集計は計測窓の外で行う（転送除外境界の checksum 計測方針との
            // 対称性も合わせて確保する。イシュー #1103 Review 指摘）。
            let elapsed = start.elapsed();
            *sync_checksum = rows.iter().flat_map(|r| r.iter()).map(|&x| x as f64).sum();
            validate_gemm_checksum(*sync_checksum)?;
            let out: Vec<f32> = rows.into_iter().flatten().collect();
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
            task: "gemm_transfer_incl",
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
            tf32: cli.tf32,
            managed: cli.managed,
        }
        .emit(&cli.out)?;
    }

    // --- 転送除外境界（fandhe dispatch_tiled_prepared と同一。A/B はループ
    // 外で 1 回だけデバイス転送し、計測窓は matmul + synchronize のみ） ---
    {
        let a_host = Xorshift64Star::new(SEED_A).fill_vec(len);
        let b_host = Xorshift64Star::new(SEED_B).fill_vec(len);
        let reference = GemmReference::compute(n, &a_host, &b_host)?;
        // ループ外の 1 回だけの転送（prepared 境界。以降の計測窓には
        // ホスト→デバイス転送を含めない）。
        let a = Tensor::from_vec(a_host, (n, n), &dev)?;
        let b = Tensor::from_vec(b_host, (n, n), &dev)?;

        let one = || -> Result<Duration, Box<dyn std::error::Error>> {
            let start = Instant::now();
            let c = a.matmul(&b)?;
            // readback を計測窓に含めない代わりに、デバイス側の実行完了を
            // 待つ（`dispatch_tiled_prepared` のコマンドバッファ完了待ちと
            // 同じ役割）。`c` は計測窓内では破棄され、checksum は計測ループ
            // 終了後に別途 1 回だけ readback して検証する。
            dev.synchronize()?;
            let elapsed = start.elapsed();
            drop(c);
            Ok(elapsed)
        };

        for _ in 0..WARMUP_ITERS {
            one()?;
        }
        let mut durations = Vec::with_capacity(MEASURE_ITERS);
        for _ in 0..MEASURE_ITERS {
            durations.push(one()?);
        }
        let st = stats(&durations)?;

        // checksum 検証は計測窓の外で 1 回だけ行う（fandhe prepared 系列と
        // 同じく、readback 自体を計測対象に含めないため）。
        let c = a.matmul(&b)?;
        let rows = readout2(&c)?;
        let checksum: f64 = rows.iter().flat_map(|r| r.iter()).map(|&x| x as f64).sum();
        validate_gemm_checksum(checksum)?;
        let out: Vec<f32> = rows.into_iter().flatten().collect();
        let parity = Some(reference.verify(&out)?);

        Record {
            framework: FRAMEWORK,
            framework_version: VERSION,
            task: "gemm_transfer_excl",
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
            tf32: cli.tf32,
            managed: cli.managed,
        }
        .emit(&cli.out)?;
    }

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
        parity: None,
        tf32: false,
        managed: cli.managed,
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
        parity: None,
        tf32: false,
        managed: cli.managed,
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
/// `--phases`（イシュー #1009）も `bench-fandhe` 専用であり、未知フラグを
/// 黙殺せず fail-fast する（README「train --phases」節参照）。`run()` から
/// [`dispatch`] を分離してあるのは `parse_cli()`（`std::env::args()` 依存）
/// を経由せずテストから直接分岐を検証できるようにするため
/// （`bench-fandhe::dispatch` と同じ構成方針）。
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_cli()?;
    dispatch(&cli)
}

/// task × `--tf32` の分岐（イシュー #1042）。`--tf32` は `--task gemm
/// --device cuda` 限定で候補精度（`candle_core::cuda_backend::
/// set_gemm_reduced_precision_f32`。candle 0.11 の公開 API）を有効化し、
/// それ以外の task/device 組合せは MEASURE_ERROR で fail-fast する
/// （`docs/cuda-tf32-optin-api-decision.md` C-1）。`cuda` cargo feature が
/// 無効なビルド（既定 `metal`）では `cuda_backend` モジュール自体が
/// コンパイル対象から外れる（candle-core `#[cfg(feature = "cuda")]`）ため、
/// `--tf32` 指定時は `device == "cuda"` を先に検査してから
/// `#[cfg(feature = "cuda")]` 分岐へ入る（feature 無効ビルドでは
/// `device == "cuda"` でも `Device::new_cuda` が
/// `NotCompiledWithCudaSupport` を返すため、実際にはこの分岐へ到達する
/// 前に `run_gemm` 内の `make_device` が失敗する）。
fn dispatch(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    if cli.mode == "reuse" {
        return Err(
            "MEASURE_ERROR: --mode reuse is not applicable to candle (device reuse is already \
             the default API design; issue #925)"
                .into(),
        );
    }
    if cli.phases {
        return Err(
            "MEASURE_ERROR: --phases is not supported by candle (bench-fandhe only; issue #1009)"
                .into(),
        );
    }
    if cli.tf32 && !(cli.task == "gemm" && cli.device == "cuda") {
        return Err(format!(
            "MEASURE_ERROR: --tf32 is only supported for --task gemm --device cuda on candle \
             (got task='{}' device='{}'; issue #1042)",
            cli.task, cli.device
        )
        .into());
    }
    // イシュー #1353: `--managed`（CUDA managed memory 配置）は fandhe-ai
    // 固有の opt-in API（`set_cuda_managed_memory_enabled`）を指す概念で
    // あり、candle には対応する公開 API が存在しない。`--phases`／
    // `--mode reuse` と同型の allowlist 方式で常に拒否する。
    if cli.managed {
        return Err(
            "MEASURE_ERROR: --managed is not supported by candle (fandhe-ai-only CUDA managed \
             memory opt-in; issue #1353)"
                .into(),
        );
    }
    match cli.task.as_str() {
        "gemm" => run_gemm(cli),
        "gemm-transfer-split" => run_gemm_transfer_split(cli),
        "train" => run_train(cli),
        "infer" => run_infer(cli),
        other => Err(format!("MEASURE_ERROR: unknown task '{other}'").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_cli(task: &str, device: &str, tf32: bool) -> Cli {
        Cli {
            task: task.to_string(),
            device: device.to_string(),
            size: 64,
            out: "/dev/null".to_string(),
            mode: "fresh".to_string(),
            phases: false,
            tf32,
            managed: false,
        }
    }

    #[test]
    fn tf32_with_non_gemm_task_is_measure_error() {
        let cli = base_cli("train", "cuda", true);
        let err = dispatch(&cli).expect_err("--tf32 with train task must be rejected");
        let msg = err.to_string();
        assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
        assert!(msg.contains("--tf32"), "msg={msg}");
    }

    #[test]
    fn tf32_with_non_cuda_device_is_measure_error() {
        let cli = base_cli("gemm", "cpu", true);
        let err = dispatch(&cli).expect_err("--tf32 with non-cuda device must be rejected");
        let msg = err.to_string();
        assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
        assert!(msg.contains("--tf32"), "msg={msg}");
    }

    /// イシュー #1103: `gemm-transfer-split` は Metal 頭打ち診断の追補
    /// （`docs/perf/metal-gemm-bottleneck-rediagnosis.md` §7.1）専用のため
    /// `--device metal` 以外を fail-closed で拒否する
    /// （`run_gemm_transfer_split` 冒頭のデバイス検証）。
    #[test]
    fn gemm_transfer_split_with_non_metal_device_is_measure_error() {
        let cli = base_cli("gemm-transfer-split", "cpu", false);
        let err =
            dispatch(&cli).expect_err("gemm-transfer-split with non-metal device must be rejected");
        let msg = err.to_string();
        assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
        assert!(msg.contains("gemm-transfer-split"), "msg={msg}");
    }

    /// `--mode reuse` は candle 全体で対象外（モジュール冒頭コメント・
    /// イシュー #925）。`gemm-transfer-split` も例外にしない（`dispatch`
    /// の既存 fail-fast 分岐がタスク種別を問わず先に検査するため）。
    #[test]
    fn gemm_transfer_split_with_reuse_mode_is_measure_error() {
        let mut cli = base_cli("gemm-transfer-split", "metal", false);
        cli.mode = "reuse".to_string();
        let err = dispatch(&cli).expect_err("--mode reuse must be rejected for candle");
        let msg = err.to_string();
        assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
        assert!(msg.contains("--mode reuse"), "msg={msg}");
    }

    /// イシュー #1353: `--managed` は fandhe-ai 固有の CUDA managed memory
    /// opt-in API を指す概念であり、candle には対応する公開 API がないため
    /// 常に MEASURE_ERROR で fail-fast する。
    #[test]
    fn managed_flag_is_always_measure_error() {
        let mut cli = base_cli("gemm", "cuda", false);
        cli.managed = true;
        let err = dispatch(&cli).expect_err("--managed must be rejected on bench-candle");
        let msg = err.to_string();
        assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
        assert!(msg.contains("--managed"), "msg={msg}");
    }
}
