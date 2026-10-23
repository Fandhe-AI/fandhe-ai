//! イシュー #3252: 学習 1 step 全体（forward・MSE・backward・SGD）を同期なしで積むチェーン
//! （`DeviceParamStore::train_step_device_chain`。公開入口は #3254）が、定常状態の計測窓の中で
//! ホスト同期・D2H・H2D・デバイス確保を起こさず、loss を読む 1 回だけが同期になることを数える
//! GB10 実機診断テスト（`docs/compat-resident-train-step-decision.md` §12）。
//!
//! # 配置理由
//! 診断カウンタ（`gemm.rs` の `GemmLaunchDiagCounters`。`cfg(test)` で有効。`pub(crate)`）へ
//! 到達する必要があるため integration test ではなく `lib.rs` の兄弟モジュールとして置く
//! （`resident_input_diag_tests_3250` と同じ判断）。カウンタはスレッドローカルなので、
//! 計測区間の呼び出しは同じスレッドで行う。
//!
//! # 計数と判定
//! - 窓（チェーン区間）: `stream_syncs`・`d2h_calls`・`h2d_calls`・`mem_uploads`・`mem_allocs` が
//!   すべて 0
//! - `read_train_step_loss` の区間: `stream_syncs` と `d2h_calls` がそれぞれちょうど 1
//! - 対照: 初回 step は確保（`mem_allocs > 0`）が起きる。`download` で同期が数えられる
//!   （カウンタが常に 0 を返す壊れ方をしていないことの担保）
//! - カウンタの範囲外（cudarc の `CudaSlice::Drop` が `has_async_alloc == false` の環境で行う
//!   同期等）はコード監査で補う（決定記録 §12）
//!
//! 実行: `cargo test -p fandhe-ai-backend-cuda --release train_step_chain_diag_tests_3252 -- --ignored --test-threads=1`

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::optim::{DeviceParamStore, ResidentLeaf, SgdConfig};
use fandhe_ai_tensor_core::{Activation, Tensor};

use crate::gemm::{gemm_launch_diag_reset, gemm_launch_diag_snapshot};
use crate::ops::CudaBackendOps;

const BATCH: usize = 64;
const DIMS: [usize; 3] = [784, 256, 10];

fn pattern(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + salt * 17) % 23) as f32 - 11.0) * 0.011)
        .collect()
}

fn tensor(salt: usize, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(pattern(n, salt), shape).expect("tensor")
}

fn config() -> SgdConfig {
    SgdConfig {
        lr: 0.01,
        momentum: 0.9,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
    }
}

/// 784-256-10・batch 64 の MLP 用 store と常駐入力を作る（窓の外）。
fn setup() -> (
    Tape,
    DeviceParamStore,
    fandhe_ai_autodiff::optim::resident_input::ResidentInput,
    fandhe_ai_autodiff::optim::resident_input::ResidentInput,
) {
    let tape = Tape::new_with_ops(Box::new(CudaBackendOps::new(0)));
    let params = [
        tensor(1, &[DIMS[0], DIMS[1]]),
        tensor(2, &[DIMS[1]]),
        tensor(3, &[DIMS[1], DIMS[2]]),
        tensor(4, &[DIMS[2]]),
    ];
    let refs: Vec<&Tensor<f32>> = params.iter().collect();
    let store = DeviceParamStore::new(&tape, &refs).expect("store");
    let x = store
        .upload_resident_input(&tape, &tensor(5, &[BATCH, DIMS[0]]))
        .expect("x");
    let y = store
        .upload_resident_input(&tape, &tensor(6, &[BATCH, DIMS[2]]))
        .expect("y");
    (tape, store, x, y)
}

/// `chain` 1 step を積む（window の外で leaves を作る必要はない: スナップショットは device 側
/// のメタ情報のみで H2D・同期を起こさない）。
fn run_step(
    tape: &Tape,
    store: &mut DeviceParamStore,
    x: &fandhe_ai_autodiff::optim::resident_input::ResidentInput,
    y: &fandhe_ai_autodiff::optim::resident_input::ResidentInput,
) -> fandhe_ai_autodiff::optim::resident_input::TrainStepLoss {
    let leaves: Vec<ResidentLeaf<'_>> = store.snapshot_resident_params(tape).expect("leaves");
    let steps = [
        (&leaves[0], Some(&leaves[1]), Activation::Relu),
        (&leaves[2], Some(&leaves[3]), Activation::None),
    ];
    store
        .train_step_device_chain(tape, x, y, &steps, &config())
        .expect("train_step_device_chain")
}

/// 定常状態（3 step のウォームアップ後）の窓で同期・D2H・H2D・確保が 0、loss を読む区間で
/// 同期と D2H がそれぞれ 1 回。初回 step は確保が起きる（対照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn train_chain_window_has_no_sync_alloc_or_h2d_and_loss_read_is_one_sync() {
    let (tape, mut store, x, y) = setup();

    // 対照: 初回 step は確保（出力バッファ・勾配連結・velocity・ワークスペース）が起きる。
    gemm_launch_diag_reset();
    let before = gemm_launch_diag_snapshot();
    let first = run_step(&tape, &mut store, &x, &y);
    let d_first = gemm_launch_diag_snapshot().since(before);
    assert!(
        d_first.mem_allocs > 0,
        "初回 step は確保が数えられるはず: {d_first:?}"
    );
    assert_eq!(d_first.stream_syncs, 0, "チェーンは同期しない: {d_first:?}");
    let _ = store.read_train_step_loss(&tape, &first).expect("read");

    // ウォームアップ（NVRTC 構築・ワークスペースの grow・プール安定化）。
    for _ in 0..3 {
        let h = run_step(&tape, &mut store, &x, &y);
        let _ = store.read_train_step_loss(&tape, &h).expect("read");
    }

    // 計測窓: チェーン区間。
    gemm_launch_diag_reset();
    let before = gemm_launch_diag_snapshot();
    let handle = run_step(&tape, &mut store, &x, &y);
    let d = gemm_launch_diag_snapshot().since(before);
    assert_eq!(d.stream_syncs, 0, "窓内の stream 同期: {d:?}");
    assert_eq!(d.d2h_calls, 0, "窓内の D2H: {d:?}");
    assert_eq!(d.h2d_calls, 0, "窓内の H2D（upload_h2d_new）: {d:?}");
    assert_eq!(d.mem_uploads, 0, "窓内の H2D（MemoryOps::upload）: {d:?}");
    assert_eq!(d.mem_allocs, 0, "窓内のデバイス確保: {d:?}");

    // loss を読む区間: 同期 1 回・D2H 1 回。
    let before = gemm_launch_diag_snapshot();
    let loss = store.read_train_step_loss(&tape, &handle).expect("read");
    let d_read = gemm_launch_diag_snapshot().since(before);
    assert!(loss.is_finite(), "loss={loss}");
    assert_eq!(d_read.stream_syncs, 1, "loss 読み出しの同期: {d_read:?}");
    assert_eq!(d_read.d2h_calls, 1, "loss 読み出しの D2H: {d_read:?}");
    println!("DIAG window={d:?} read={d_read:?} first={d_first:?} loss={loss}");
}
