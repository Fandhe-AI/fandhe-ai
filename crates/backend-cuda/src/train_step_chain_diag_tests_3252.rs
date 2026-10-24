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
//! # イシュー #3256 の追記（公開入口と同じ呼び出し列の再現・TF32 観測）
//! 上の B1 は `train_step_device_chain` を直接呼ぶため、facade の公開入口
//! （`Sequential::train_step_resident`／`ResidentLoss::read`）が足す処理を通らない。facade からは
//! 診断カウンタへ届かない（`internal-diagnostics` を facade の dev-dependency にすると
//! `cargo test --workspace` の対象が増えるため採らない）ので、facade の `resolve_ops`・`tape_for`・
//! `train_step_resident_with`・`ResidentLoss::read` と同じ呼び出し列（存在検証 → tape 作成 →
//! `snapshot_resident_params` → `validate_train_step_device_chain` → チェーン、読み出しは新しい tape で
//! `read_train_step_loss`）を本モジュールで 1 スレッド上に再現して数える。facade 固有の残りの処理
//! （層の検査・`build_device_chain_steps`）はホスト側の純計算でデバイス呼び出しを持たないため、
//! カウンタの範囲外としてコード監査で補う。窓の `d2h_calls` が 0 であることは、第 2 段フォールバック
//! （D2H を伴う）に入っていないことの証拠にもなる。
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
/// facade の `resolve_ops`（CUDA 腕）＋ `tape_for` と同じ手順で tape を作る（存在検証つき）。
fn public_entry_tape() -> Tape {
    use fandhe_ai_tensor_core::{Device, DeviceProvider, select_from};
    let provider = crate::CudaDeviceProvider::new();
    let providers: [&dyn DeviceProvider; 1] = [&provider];
    select_from(&providers, Device::Cuda(0)).expect("select_from");
    Tape::new_with_ops(Box::new(CudaBackendOps::new(0)))
}

/// 公開入口 `train_step_resident` と同じ呼び出し列で 1 step 積む（tape は step ごとに作る）。
fn run_step_public_sequence(
    store: &mut DeviceParamStore,
    x: &fandhe_ai_autodiff::optim::resident_input::ResidentInput,
    y: &fandhe_ai_autodiff::optim::resident_input::ResidentInput,
) -> Result<
    fandhe_ai_autodiff::optim::resident_input::TrainStepLoss,
    fandhe_ai_tensor_core::BackendError,
> {
    let tape = public_entry_tape();
    let leaves: Vec<ResidentLeaf<'_>> = store.snapshot_resident_params(&tape).expect("leaves");
    let steps = [
        (&leaves[0], Some(&leaves[1]), Activation::Relu),
        (&leaves[2], Some(&leaves[3]), Activation::None),
    ];
    store
        .validate_train_step_device_chain(&tape, x, y, &steps, &config())
        .expect("validate");
    store.train_step_device_chain(&tape, x, y, &steps, &config())
}

/// #3256: 公開入口と同じ呼び出し列でも、窓の同期・D2H・H2D・確保が 0、`ResidentLoss::read` 相当の
/// 区間（新しい tape の作成を含む）で同期 1 回・D2H 1 回。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn public_entry_call_sequence_window_has_no_sync_and_read_is_one_sync() {
    let (tape, mut store, x, y) = setup();
    drop(tape);

    gemm_launch_diag_reset();
    let before = gemm_launch_diag_snapshot();
    let first = run_step_public_sequence(&mut store, &x, &y).expect("first");
    let d_first = gemm_launch_diag_snapshot().since(before);
    assert!(
        d_first.mem_allocs > 0,
        "初回 step は確保が数えられるはず: {d_first:?}"
    );
    let t = public_entry_tape();
    let _ = store.read_train_step_loss(&t, &first).expect("read");

    for _ in 0..3 {
        let h = run_step_public_sequence(&mut store, &x, &y).expect("warm");
        let t = public_entry_tape();
        let _ = store.read_train_step_loss(&t, &h).expect("read");
    }

    gemm_launch_diag_reset();
    let before = gemm_launch_diag_snapshot();
    let handle = run_step_public_sequence(&mut store, &x, &y).expect("step");
    let d = gemm_launch_diag_snapshot().since(before);
    assert_eq!(d.stream_syncs, 0, "窓内の stream 同期: {d:?}");
    assert_eq!(
        d.d2h_calls, 0,
        "窓内の D2H（フォールバックなら > 0）: {d:?}"
    );
    assert_eq!(d.h2d_calls, 0, "窓内の H2D（upload_h2d_new）: {d:?}");
    assert_eq!(d.mem_uploads, 0, "窓内の H2D（MemoryOps::upload）: {d:?}");
    assert_eq!(d.mem_allocs, 0, "窓内のデバイス確保: {d:?}");

    let before = gemm_launch_diag_snapshot();
    let t = public_entry_tape();
    let loss = store.read_train_step_loss(&t, &handle).expect("read");
    let d_read = gemm_launch_diag_snapshot().since(before);
    assert!(loss.is_finite(), "loss={loss}");
    assert_eq!(d_read.stream_syncs, 1, "読み出し区間の同期: {d_read:?}");
    assert_eq!(d_read.d2h_calls, 1, "読み出し区間の D2H: {d_read:?}");
    println!("DIAG3256 window={d:?} read={d_read:?} first={d_first:?} loss={loss}");
}

/// 精度モードを drop で元に戻す RAII ガード。
struct PrecisionRestore(crate::precision::CudaGemmPrecision);

impl Drop for PrecisionRestore {
    fn drop(&mut self) {
        crate::precision::set_gemm_precision(self.0);
    }
}

/// #3256 申し送り（#3255）の観測: TF32 モードで `train_step_device_chain` が成功するか
/// `Unsupported`（= facade が第 2 段の Tape 経路へフォールバック）になるかを記録する。
/// 新しい契約は作らないため、主張は panic しないことだけ。結果は標準出力の `TF32OBS` 行を見る。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。記録専用で合否判定しない"]
fn tf32_mode_chain_outcome_is_recorded() {
    let (tape, mut store, x, y) = setup();
    drop(tape);
    let _restore = PrecisionRestore(crate::precision::gemm_precision());
    crate::precision::set_gemm_precision(crate::precision::CudaGemmPrecision::Tf32);

    gemm_launch_diag_reset();
    let before = gemm_launch_diag_snapshot();
    let r = run_step_public_sequence(&mut store, &x, &y);
    let d = gemm_launch_diag_snapshot().since(before);
    let outcome = match &r {
        Ok(_) => "Ok（チェーンが TF32 モードでも成功）".to_string(),
        Err(fandhe_ai_tensor_core::BackendError::Unsupported(m)) => {
            format!("Err(Unsupported)（facade は第 2 段へフォールバック）: {m}")
        }
        Err(e) => format!("Err(その他): {e:?}"),
    };
    println!("TF32OBS outcome={outcome} counters={d:?}");
}
