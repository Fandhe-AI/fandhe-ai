//! イシュー #3608: Graph capture 中のチェーン外の通常操作（`ResidentLoss::read` 相当の
//! `read_train_step_loss`・store 構築・入力 upload・`sync_to_host`・`wait_device`・Tape 演算）が、
//! 別スレッドの capture を待ってから成功する（待たずに `DeviceContextCaptureInProgress` で
//! 失敗しない）ことの GB10 実機回帰テスト（G1 型の並行テスト。
//! `docs/backend-cuda-graph-step-capture-design.md` §8.12.5・所有者確認 2026-10-27）。
//!
//! 役割: `context_cache::begin_driver_call_waiting`（本番の待ち上限 1000 ms）が、実 driver 上で
//! capture 区間と通常操作を競合させても通常操作側にエラーを出さないことを確かめる。待ち上限は
//! lib の `cfg(test)` では 30 ms になるため、`src/` 配下のテストではなく integration test として
//! 置く（本番値で走らせる）。
//!
//! **既存の 3260 ハーネス（`train_step_graph_diag_tests_3260.rs`）の `retry_busy` は使わない**。
//! あれは本件でなくすエラーそのものを握りつぶすため、ここでは B の各呼び出しを直接 `Result` で
//! 受け、`DeviceContextCaptureInProgress` が 1 件でも出たら失敗にする。
//!
//! 実行（GB10。`FANDHE_AI_CUDA_GRAPH_TRAIN=1` は起動時に stream 種別を決めるため必須）:
//! ```text
//! FANDHE_AI_CUDA_GRAPH_TRAIN=1 cargo test -p fandhe-ai-backend-cuda --release \
//!   --test train_graph_out_of_chain_wait_3608 -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! 合格条件:
//! - B に `DeviceContextCaptureInProgress`（を含むどのエラー）も出ない
//! - B の loss 列・最終パラメータ・Tape 演算の結果が、単独実行（A なし）の参照と bit 完全一致
//! - A・B とも poison されない（`wait_device` が成功する）
//! - `out_of_chain_capture_wait_count`（B 側の入口が foreign capture の待機分岐へ入った回数）が
//!   1 以上増えている（A の capture と B の通常操作が実際に重なった確認。B は最低 REPEATS 回に
//!   加え、A が終わるまで繰り返して重なりを確保する）
//! - `train_graph_stats` の `captured` が増えている（capture に届いた確認。届かないまま合格
//!   しないため必須）

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::optim::resident_input::{ResidentInput, TrainStepLoss};
use fandhe_ai_autodiff::optim::{DeviceParamStore, ResidentLeaf, SgdConfig};
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_backend_cuda::graph::{
    out_of_chain_capture_wait_count, override_train_graph_for_scope, train_graph_stats,
};
use fandhe_ai_tensor_core::{Activation, BackendError, Tensor};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

const BATCH: usize = 64;
const DIMS: [usize; 3] = [784, 256, 10];
const STEPS: usize = 20;
const REPEATS: usize = 30;

fn pattern(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + salt * 17) % 23) as f32 - 11.0) * 0.011)
        .collect()
}

fn tensor(salt: usize, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(pattern(n, salt), shape).expect("tensor")
}

fn config(lr: f32) -> SgdConfig {
    SgdConfig {
        lr,
        momentum: 0.9,
        dampening: 0.0,
        weight_decay: 1e-3,
        nesterov: false,
    }
}

/// stream 種別は起動時の環境変数で決まる。未指定だと capture に届かないまま bit 一致だけが通る。
fn require_opt_in_env() {
    assert_eq!(
        std::env::var("FANDHE_AI_CUDA_GRAPH_TRAIN").as_deref(),
        Ok("1"),
        "FANDHE_AI_CUDA_GRAPH_TRAIN=1 を付けて起動すること"
    );
}

struct Run {
    tape: Tape,
    store: DeviceParamStore,
    x: ResidentInput,
    y: ResidentInput,
}

/// 構築・upload は `?` で直接受ける（`retry_busy` なし）。
fn new_run() -> Result<Run, BackendError> {
    let tape = Tape::new_with_ops(Box::new(CudaBackendOps::new(0)));
    let params = [
        tensor(1, &[DIMS[0], DIMS[1]]),
        tensor(2, &[DIMS[1]]),
        tensor(3, &[DIMS[1], DIMS[2]]),
        tensor(4, &[DIMS[2]]),
    ];
    let refs: Vec<&Tensor<f32>> = params.iter().collect();
    let store = DeviceParamStore::new(&tape, &refs)?;
    let x = store.upload_resident_input(&tape, &tensor(5, &[BATCH, DIMS[0]]))?;
    let y = store.upload_resident_input(&tape, &tensor(6, &[BATCH, DIMS[2]]))?;
    Ok(Run { tape, store, x, y })
}

fn step(run: &mut Run, lr: f32) -> Result<TrainStepLoss, BackendError> {
    let leaves: Vec<ResidentLeaf<'_>> = run.store.snapshot_resident_params(&run.tape)?;
    let steps = [
        (&leaves[0], Some(&leaves[1]), Activation::Relu),
        (&leaves[2], Some(&leaves[3]), Activation::None),
    ];
    run.store
        .train_step_device_chain(&run.tape, &run.x, &run.y, &steps, &config(lr))
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// B の 1 回分の結果（loss 列・最終パラメータ・Tape 演算の結果）。
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    losses: Vec<u32>,
    params: Vec<Vec<u32>>,
    tape_out: Vec<u32>,
    tape_grad: Vec<u32>,
}

/// B の通常操作 1 回分。どの呼び出しも直接 `Result` で受けて、失敗したら panic にする。
fn out_of_chain_once() -> Outcome {
    let mut run = new_run().unwrap_or_else(|e| panic!("B: new_run: {e:?}"));
    let mut losses = Vec::with_capacity(STEPS);
    for i in 0..STEPS {
        let h = step(&mut run, 0.01).unwrap_or_else(|e| panic!("B: step {i}: {e:?}"));
        let loss = run
            .store
            .read_train_step_loss(&run.tape, &h)
            .unwrap_or_else(|e| panic!("B: read_train_step_loss {i}: {e:?}"));
        losses.push(loss.to_bits());
    }
    run.store
        .wait_device(&run.tape)
        .unwrap_or_else(|e| panic!("B: wait_device（poison を含む）: {e:?}"));
    let params = run
        .store
        .sync_to_host(&run.tape)
        .unwrap_or_else(|e| panic!("B: sync_to_host: {e:?}"))
        .iter()
        .map(bits)
        .collect();

    // Tape の通常演算（forward・backward）。
    let tape = Tape::new_with_ops(Box::new(CudaBackendOps::new(0)));
    let a = tape.var(&tensor(7, &[32, 48]));
    let w = tape.var(&tensor(8, &[48, 16]));
    let bias = tape.var(&tensor(9, &[32, 16]));
    let y = a
        .matmul(&w)
        .and_then(|m| m.add(&bias))
        .unwrap_or_else(|e| panic!("B: tape forward: {e:?}"));
    let loss = y.sum(None).unwrap_or_else(|e| panic!("B: tape sum: {e:?}"));
    let grads = tape
        .backward(&loss)
        .unwrap_or_else(|e| panic!("B: tape backward: {e:?}"));
    let tape_out = bits(&y.to_tensor());
    let tape_grad = bits(
        grads
            .get(&w)
            .unwrap_or_else(|e| panic!("B: grads.get: {e:?}"))
            .expect("w の勾配"),
    );
    Outcome {
        losses,
        params,
        tape_out,
        tape_grad,
    }
}

/// A: opt-in ON。繰り返しごとに新しい store を作って capture の窓を増やす。
fn capture_thread() -> usize {
    let _opt = override_train_graph_for_scope(true);
    for r in 0..REPEATS {
        let mut run = new_run().unwrap_or_else(|e| panic!("A: new_run {r}: {e:?}"));
        for i in 0..STEPS {
            let h = step(&mut run, 0.01).unwrap_or_else(|e| panic!("A: step {r}/{i}: {e:?}"));
            run.store
                .read_train_step_loss(&run.tape, &h)
                .unwrap_or_else(|e| panic!("A: read_train_step_loss {r}/{i}: {e:?}"));
        }
        run.store
            .wait_device(&run.tape)
            .unwrap_or_else(|e| panic!("A: wait_device（poison を含む）{r}: {e:?}"));
    }
    REPEATS
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。FANDHE_AI_CUDA_GRAPH_TRAIN=1 で単独実行"]
fn out_of_chain_ops_wait_for_foreign_capture_without_errors_and_stay_bit_exact() {
    require_opt_in_env();
    let before = train_graph_stats();
    let wait_before = out_of_chain_capture_wait_count();

    // 参照: A なしの単独実行（opt-in OFF）。
    let reference = {
        let _opt = override_train_graph_for_scope(false);
        out_of_chain_once()
    };

    let a_done = Arc::new(AtomicBool::new(false));
    let a_done_a = Arc::clone(&a_done);
    let a = std::thread::spawn(move || {
        let r = capture_thread();
        a_done_a.store(true, Ordering::SeqCst);
        r
    });
    let b = std::thread::spawn(move || {
        let _opt = override_train_graph_for_scope(false);
        // 最低 REPEATS 回、A が終わるまで（待機経路に届くまで）繰り返す。上限は暴走防止。
        let mut outs = Vec::new();
        while outs.len() < REPEATS
            || (!a_done.load(Ordering::SeqCst)
                && out_of_chain_capture_wait_count() == wait_before
                && outs.len() < REPEATS * 20)
        {
            outs.push(out_of_chain_once());
        }
        outs
    });
    let outs = b
        .join()
        .expect("B が panic しない（CaptureInProgress を含む失敗なし）");
    a.join().expect("A が panic しない");

    for (r, out) in outs.iter().enumerate() {
        assert_eq!(out, &reference, "B repeat {r} が単独実行と bit 一致しない");
    }
    let after = train_graph_stats();
    let waited = out_of_chain_capture_wait_count() - wait_before;
    assert!(
        waited >= 1,
        "B が foreign capture の待機分岐に一度も入っていない（A の capture と重なっていない。\
         待機処理が即時拒否へ戻っても通ってしまうため不合格とする）"
    );
    assert!(
        after.captured > before.captured,
        "capture に届いていない（captured {} -> {}）。stream 種別・opt-in を確認すること",
        before.captured,
        after.captured
    );
    println!(
        "OOC3608 waited={waited} captured={} replayed={} fallbacks={}",
        after.captured - before.captured,
        after.replayed - before.replayed,
        after.fallbacks - before.fallbacks
    );
}
