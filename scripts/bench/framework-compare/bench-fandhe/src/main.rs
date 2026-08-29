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
//!
//! `train --mode reuse`（イシュー #958）: gemm の reuse（イシュー #925）と
//! 同じ「初期化コストとカーネル実行の分離」の考えを学習ループへ適用する。
//! fresh の `run_train` は各 step でホスト経由 SGD（勾配を download →
//! ホストで `p - lr*g` → `apply_parameters` で書き戻し）を行っており、
//! candle（`Var::set`）や Burn（デバイス上更新）と非対称なプロトコルに
//! なっている（#957 背景）。reuse は #954 で追加されたデバイス常駐パラ
//! メータ更新 API（`fandhe_ai::DeviceParamStore`）を使い、`p - lr*g` の
//! 更新自体をデバイス上で完結させる。
//!
//! 参照実装は `crates/facade/tests/device_param_store_train.rs` の
//! `train_with_device_param_store`（`init_device_param_store` で 1 回だけ
//! 全パラメータを H2D upload → 以後は同一 `DeviceParamStore` を使い回す）
//! であり、本関数（`run_train_reuse`）はその構造に揃える。tape 自体は
//! （gemm reuse と異なり）**step ごとに新規生成**する: `fandhe_ai_autodiff::
//! Tape` はノード列クリア API を持たず学習ループはステップごとに tape を
//! 生成・破棄する設計契約（`crates/autodiff/src/tape.rs`）であり、単一
//! tape を 100 step 使い回すと `Tape::backward` の逆順走査コストが step
//! 数に比例して増加し 1 step の計測時間が非定常になる。reuse で使い回す
//! のは tape ではなく `DeviceParamStore`（デバイス常駐バッファ・デバイス
//! を固定する側）であり、fresh/reuse の計時差は「ホスト経由 SGD vs デバ
//! イス常駐更新」に限定される。
//!
//! 既知の前提（改善量の解釈範囲）: `Sequential::forward_resident` が呼ぶ
//! `DeviceParamStore::register_resident_leaves` は forward 用に毎 step
//! パラメータを D2H download する（`crates/autodiff/src/optim/
//! device_store.rs`・`docs/device-resident-update-design.md` §3.3b・
//! イシュー #954 申し送り）。reuse が排除するのは「毎 step の再アップ
//! ロード（H2D）+ ホスト計算」であり、この D2H は reuse でも残存する。

use bench_common::*;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, SgdConfig, Tape, Tensor};
use std::time::{Duration, Instant};

const FRAMEWORK: &str = "fandhe-ai";
const VERSION: &str = "0.4.0";

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
    // イシュー #970 codex-review 指摘（PR #978・P1）: `n * n`（要素数）を
    // 入力ベクタ生成前に `gemm_element_count` で検証する（`--size` は
    // 未検証な CLI 入力であり、無検証の乗算は debug で panic・release では
    // wrap した長さで確保・使用してしまう）。`run_gemm`/`run_gemm_reuse`
    // 双方がこの関数を経由するため、両経路とも生成前検証が及ぶ。
    let len = gemm_element_count(n)?;
    let a = Xorshift64Star::new(SEED_A).fill_vec(len);
    let b = Xorshift64Star::new(SEED_B).fill_vec(len);
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

/// `--mode reuse` の train 計測（イシュー #958）。`DeviceParamStore` を
/// 1 回だけ構築し（`init_s` として初期化コストを分離記録）、以後の各 step
/// は新規 tape 上で `forward_resident` → `backward` →
/// `step_device_param_store`（デバイス上 SGD 更新）を行う。ホスト経由の
/// download/upload（fresh の `p - lr*g` 相当）はループ内で一切行わない。
/// モジュール doc「train --mode reuse」節に設計判断の詳細を記す。
fn run_train_reuse(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let model = build_model()?;
    let (x_data, y_data) = mlp_data()?;

    // init_s: 初回 tape 構築 + 全パラメータの 1 回限りの H2D upload
    // （`init_device_param_store`）までの経過。以後の DeviceParamStore は
    // このデバイス・バッファを固定して使い回す。
    //
    // codex-review PR #998 P2 是正: `init_device_param_store` の内部
    // 実装（`MemoryOps::upload`）は CUDA では `clone_htod` 等の非同期
    // H2D コピーを発行するのみで、発行元ストリーム上の完了を待たない
    // （`DeviceParamStore::new` doc・`CudaMemory::upload_inner` 参照）。
    // `elapsed()` を非同期発行直後に取得すると、転送完了待ちのコストが
    // `init_s` から漏れ、代わりに最初の `forward_resident` 内の
    // download 同期（暗黙のストリーム同期点）へ計上されてしまう。
    // ここでは `sync_to_host` の明示同期点（`DeviceParamStore::
    // sync_to_host` doc「明示同期点」参照）を `elapsed()` 取得前に挟み、
    // アップロードされた全パラメータの転送完了を保証したうえで `init_s`
    // を確定する（ダウンロードした内容自体は計測対象ではないため破棄）。
    let init_start = Instant::now();
    let init_tape = make_tape(&cli.device)?;
    let mut store = model.init_device_param_store(&init_tape)?;
    let _ = init_tape.sync_device_param_store_to_host(&store)?;
    drop(init_tape);
    let init_s = init_start.elapsed().as_secs_f64();

    let config = SgdConfig::new(LR);
    let mut durations = Vec::with_capacity(TRAIN_STEPS);
    let mut last_loss = 0.0f32;

    for _ in 0..TRAIN_STEPS {
        let start = Instant::now();
        // fresh tape per step（モジュール doc 参照: DeviceParamStore は
        // 使い回すが tape は使い回さない）。
        let tape = make_tape(&cli.device)?;
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = model.forward_resident(&tape, &x, &mut store)?;
        let loss = pred.mse_loss(&y)?;
        // host readout of the loss（fresh と同じくループ内の同期点）。
        last_loss = loss
            .to_tensor()
            .get(&[])
            .ok_or("loss should be a scalar with shape []")?;
        let grads = tape.backward(&loss)?;
        // デバイス上 SGD 更新（ホストへの download/upload を経由しない）。
        tape.step_device_param_store(&mut store, &grads, &config)?;
        durations.push(start.elapsed());
    }

    if !last_loss.is_finite() {
        return Err(format!("MEASURE_ERROR: final loss not finite: {last_loss}").into());
    }

    // 終端同期: 新規 tape から DeviceParamStore の内容をホストへ実体化する
    // （計測窓の外。gemm reuse の checksum 実体化と同じ位置づけ）。件数が
    // trainable_parameters() と不一致・非有限要素があれば MEASURE_ERROR
    // として記録を拒否する（A08: 破損した学習結果を性能値として残さない）。
    let final_tape = make_tape(&cli.device)?;
    let synced = final_tape.sync_device_param_store_to_host(&store)?;
    let expected_len = model.trainable_parameters().len();
    if synced.len() != expected_len {
        return Err(format!(
            "MEASURE_ERROR: sync_device_param_store_to_host returned {} tensors, expected {expected_len}",
            synced.len()
        )
        .into());
    }
    for t in &synced {
        let slice = t
            .contiguous()
            .as_slice()
            .ok_or("synced param as_slice() returned None")?
            .to_vec();
        if slice.iter().any(|v| !v.is_finite()) {
            return Err("MEASURE_ERROR: synced parameter contains non-finite element".into());
        }
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
        mode: "reuse",
        init_s: Some(init_s),
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

/// task × mode の分岐。reuse モードは受け入れ条件の範囲（gemm・train）に
/// 限定し、infer × reuse は MEASURE_ERROR で fail-fast する（gemm は
/// イシュー #925 §2.1・§8、train はイシュー #958。infer への拡張は将来
/// 対応が必要な場合は別イシューで追跡）。
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_cli()?;
    match (cli.task.as_str(), cli.mode.as_str()) {
        ("gemm", "fresh") => run_gemm(&cli),
        ("gemm", "reuse") => run_gemm_reuse(&cli),
        ("train", "fresh") => run_train(&cli),
        ("train", "reuse") => run_train_reuse(&cli),
        ("infer", "fresh") => run_infer(&cli),
        (task, "reuse") => Err(format!(
            "MEASURE_ERROR: --mode reuse is not implemented for task '{task}' (gemm / train only; issue #925 / #958)"
        )
        .into()),
        (other, _) => Err(format!("MEASURE_ERROR: unknown task '{other}'").into()),
    }
}

/// `run_train`（fresh）と `run_train_reuse`（reuse）の最終 loss（checksum）
/// が統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
/// `.claude/rules/coding-rust.md`）の範囲内で一致することを検証する
/// （受け入れ条件 5。イシュー #958）。cpu 経由・実機非依存のため `#[ignore]`
/// は付けない。`--release` 推奨（README 参照。debug では GEMM が遅い）。
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// テスト間で衝突しない一時 JSONL パスを作る（pid + カウンタで一意化。
    /// 並行テスト実行時の読み取り／削除の混入を防ぐ）。
    fn temp_out_path(tag: &str) -> std::path::PathBuf {
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "bench-fandhe-test-{tag}-{}-{n}.jsonl",
            std::process::id()
        ))
    }

    fn make_cli(task: &str, mode: &str, out: &std::path::Path) -> Cli {
        Cli {
            task: task.to_string(),
            device: "cpu".to_string(),
            size: 64,
            out: out.to_string_lossy().into_owned(),
            mode: mode.to_string(),
        }
    }

    /// JSONL の最終行から `"checksum":<value>` を取り出す最小パーサー
    /// （フル JSON デコーダを新規依存させないための最小実装。`Record::emit`
    /// の出力形式に依存する）。
    fn last_line_checksum(path: &std::path::Path) -> f64 {
        let content = std::fs::read_to_string(path).expect("test: JSONL 読み取り失敗");
        let last = content.lines().next_back().expect("test: JSONL に行がない");
        let key = "\"checksum\":";
        let start = last
            .find(key)
            .expect("test: checksum フィールドが見つからない")
            + key.len();
        let rest = &last[start..];
        let end = rest
            .find([',', '}'])
            .expect("test: checksum フィールドの終端が見つからない");
        rest[..end]
            .trim()
            .parse::<f64>()
            .expect("test: checksum を f64 としてパースできない")
    }

    #[test]
    fn train_reuse_matches_fresh_final_loss_within_composite_tolerance() {
        let fresh_path = temp_out_path("fresh");
        let reuse_path = temp_out_path("reuse");

        run_train(&make_cli("train", "fresh", &fresh_path)).expect("run_train (fresh) failed");
        run_train_reuse(&make_cli("train", "reuse", &reuse_path))
            .expect("run_train_reuse (reuse) failed");

        let fresh_checksum = last_line_checksum(&fresh_path);
        let reuse_checksum = last_line_checksum(&reuse_path);

        let _ = std::fs::remove_file(&fresh_path);
        let _ = std::fs::remove_file(&reuse_path);

        assert!(
            fresh_checksum.is_finite() && reuse_checksum.is_finite(),
            "checksum must be finite: fresh={fresh_checksum} reuse={reuse_checksum}"
        );
        let abs_diff = (fresh_checksum - reuse_checksum).abs();
        let rel_diff = abs_diff / fresh_checksum.abs().max(1e-12);
        assert!(
            abs_diff < 1e-5 || rel_diff < 1e-3,
            "fresh/reuse final loss mismatch: fresh={fresh_checksum} reuse={reuse_checksum} \
             abs_diff={abs_diff} rel_diff={rel_diff}"
        );
    }

    #[test]
    fn train_reuse_produces_expected_record_fields() {
        let out = temp_out_path("reuse-fields");
        run_train_reuse(&make_cli("train", "reuse", &out)).expect("run_train_reuse failed");
        let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&out);
        let last = content.lines().next_back().expect("test: JSONL に行がない");
        assert!(last.contains("\"task\":\"train\""), "line={last}");
        assert!(last.contains("\"mode\":\"reuse\""), "line={last}");
        assert!(last.contains("\"init_s\":"), "line={last}");
        assert!(!last.contains("\"init_s\":null"), "line={last}");
    }
}
