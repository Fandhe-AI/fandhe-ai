//! Metal `gemm_fp32_strict_into` の NN フォールバック（イシュー #1555・
//! PR #1556 の codex-review 指摘是正）の facade 統合回帰テスト。
//!
//! `crates/backend-metal/src/ops.rs::MetalBackendOps::gemm_fp32_strict_into`
//! doc「`Unsupported` を返さない理由」が挙げる実例（`Linear(1, 8)` →
//! `ReLU` → `Linear(8, 4)`。前段の d_weight は `x_t` の strides が
//! `[1, 1]` になり NN 扱いで分類不能、後段は NT/TN で対応）そのものを
//! 再現する。旧実装（NN を `Unsupported` で返す）では、backward が
//! 後段（NT/TN・resident 成功）→ 前段（NN・`Unsupported`）の順で処理
//! されるため、前段の `Unsupported` が `DeviceParamStore::
//! resident_grad_capability` をストア全体で `Some(false)` へ倒し、
//! 既に resident へ直接書き込まれた後段の重み勾配（`Gradients` には
//! 載らない契約。`grad::vjp` doc 参照）が `param_grads_to_host` から
//! 一切読めず `MissingGradient` で失敗していた
//! （`crates/autodiff/src/optim/device_store.rs::param_grads_to_host`）。
//!
//! `crates/facade/tests/device_param_store_backend_parity.rs` と同型の
//! 構成（`tape_for` composition root・decisive gate は統一複合判定）だが、
//! 対応形状（後段 NT/TN）と非対応形状（前段 NN 縮退）が同一 backward に
//! 混在するケースに絞った回帰テストである点が異なる。
//!
//! **実機ゲーティング**: `cfg(target_os = "macos")` は非 macOS の CI での
//! コンパイル対象除外にしかならず実機の有無までは保証しないため
//! （`.claude/rules/ci.md`「実機依存」節）、`#[ignore]` で通常 CI から
//! 分離する。
//!
//! 実行コマンド（Apple Silicon 実機）:
//!
//! ```sh
//! cargo test -p fandhe-ai --test device_param_store_metal_mixed_shape_grad -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai::compat::Sequential;
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_tensor_core::Tensor;

const BATCH: usize = 4;
const D_IN: usize = 1;
const D_HIDDEN: usize = 8;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xFACE_1555;
const SEED_L1: u64 = 0xAAAA_AAAA;
const SEED_L2: u64 = 0xBBBB_BBBB;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// 統一複合判定（`.claude/rules/coding-rust.md`）。
fn assert_close(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

fn assert_tensors_match(actual: &Tensor<f32>, expected: &Tensor<f32>, ctx: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{ctx}: shape mismatch");
    let a = actual.contiguous();
    let e = expected.contiguous();
    let a_data = a.as_slice().unwrap();
    let e_data = e.as_slice().unwrap();
    for (i, (av, ev)) in a_data.iter().zip(e_data.iter()).enumerate() {
        assert_close(*av, *ev, &format!("{ctx}: element {i}"));
    }
}

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

/// `Linear(1, 8)` → `ReLU` → `Linear(8, 4)`。前段の in_features=1 が
/// `x_t` を strides=[1, 1] の NN 縮退へ追い込む codex 指摘の実例。
fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap()
}

/// Metal 実機（macOS）: 同じ backward に対応形状（後段 NT/TN）と非対応
/// 形状（前段 NN 縮退）が混在しても、`resident_grads_to_host` が両層の
/// weight slot を `Some` で返し、`param_grads_to_host` が
/// `MissingGradient` を起こさずホスト参照実装と一致することを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn param_grads_to_host_succeeds_when_backward_mixes_supported_and_fallback_shapes() {
    let device = Device::Metal;
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let init_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    // `pending`（`DeviceParamStore` 1 backward ぶんの登録）は `step()`
    // で消費するまで再登録を拒否する（`BackendError::
    // PendingForwardUnconsumed`）ため、本テストは単発の forward→backward
    // で codex 指摘の縮退を再現する（`assert_grad_readout_contract` と
    // 同型の単発構成）。
    let tape = fandhe_ai::tape_for(device).unwrap();
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
    let grads = tape.backward_device_param_store(&loss, &store).unwrap();

    // AC: `resident_grads_to_host` が `Unsupported`（旧実装のバグ挙動）
    // ではなく `Ok` を返し、両層の weight slot（build_model の層順で
    // index 0・2）が `Some`（resident 経由で充填済み）であること。
    // bias slot（index 1・3）は resident 経由で充填されないため `None`。
    let resident = tape.resident_grads_to_host(&store, &grads).unwrap();
    assert_eq!(resident.len(), 4, "2 層 Linear で計 4 パラメータのはず");
    for (i, slot) in resident.iter().enumerate() {
        let is_weight = i % 2 == 0;
        assert_eq!(
            slot.is_some(),
            is_weight,
            "param {i} の resident 充填状態が期待と異なる（is_weight={is_weight}）: {slot:?}"
        );
    }

    // AC（codex-review 指摘の本体）: `param_grads_to_host` が
    // `MissingGradient` を起こさず全 4 パラメータの勾配を返す。
    let device_grads = tape.param_grads_to_host(&store, &grads).unwrap();
    assert_eq!(device_grads.len(), 4);

    // ホスト参照実装（CPU タープ・`eval::matmul` 系の通常 VJP）と統一
    // 複合判定で一致することを確認する（`device_param_store_backend_
    // parity.rs::assert_grad_readout_contract` と同型）。
    let host_grads = {
        let host_tape = fandhe_ai::tape();
        let bound = model.bind(&host_tape);
        let hx = host_tape.var(&x_data);
        let hy = host_tape.var(&y_data);
        let hpred = bound.forward(&host_tape, &hx).unwrap();
        let hloss = MseLoss::new(Reduction::Mean).forward(&hpred, &hy).unwrap();
        let hgrads = host_tape.backward(&hloss).unwrap();
        bound
            .trainable_grads(&hgrads)
            .unwrap()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>()
    };
    assert_eq!(host_grads.len(), 4);
    for (i, (dg, hg)) in device_grads.iter().zip(host_grads.iter()).enumerate() {
        assert_tensors_match(dg, hg, &format!("param {i} grad (Metal vs host)"));
    }
}
