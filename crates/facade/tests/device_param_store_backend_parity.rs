//! イシュー #936（デバイス常駐更新の parity テストとベンチ非後退確認）の
//! 受け入れ条件 1（parity テスト追加）を、`crates/facade/tests/
//! device_param_store_train.rs` の CPU tape 限定テストでは検証できない
//! 実バックエンド（Metal／CUDA）横断で満たす統合テスト。
//!
//! `fandhe_ai::tape_for(Device::Metal)`／`tape_for(Device::Cuda(0))`
//! （composition root。`crates/facade/src/lib.rs::tape_for`）経由で
//! `Sequential::init_device_param_store`／`forward_resident`／
//! `Tape::step_device_param_store` を回し、CPU ホスト参照実装
//! （`fandhe_ai_autodiff::optim::Sgd::step` + `Sequential::
//! apply_parameters`）の最終パラメータと突合する。
//!
//! **判定方式**: 設計文書 `docs/device-resident-update-design.md` §5.3・
//! §7「#936 への引き渡し事項」が定める「決定的シードで 100 step 程度学習
//! し、最終パラメータのみを統一複合判定（相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満。`.claude/rules/coding-rust.md`）で比較する」方式
//! （中間ステップごとの判定は必須としない）。
//!
//! **実機ゲーティング**: `cfg(target_os = "macos")` は非 macOS の CI での
//! コンパイル対象除外にしかならず実機の有無までは保証しないため
//! （`.claude/rules/ci.md`「実機依存」節・`.claude/rules/coding-rust.md`
//! 「テスト・ベンチ」節）、Metal・CUDA いずれも理由付き `#[ignore]` で
//! 通常 CI（GitHub ホステッド ubuntu-latest）から分離する
//! （`crates/backend-metal/tests/sgd_device_parity.rs`・
//! `crates/backend-cuda/tests/sgd_device_real_device.rs` と同じ方針）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, SgdConfig as FacadeSgdConfig};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_autodiff::optim::{Sgd, SgdConfig};
use fandhe_ai_tensor_core::Tensor;

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

const STEPS: usize = 100;
const LR: f32 = 0.02;

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

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap()
}

/// `device` へ結線した `tape_for` 経由でデバイス常駐経路を `steps` 回
/// 回し、最終的にホストへ同期したパラメータ列を返す。momentum・
/// weight_decay・nesterov を有効化した構成で回す（`crates/backend-cpu/
/// tests/sgd_device_parity.rs::full_combo_matches_host_reference_across_100_steps`
/// と同じ意図で、全オプション経路を累積誤差の観点で保険的に検証する）。
fn train_device_resident(device: Device, steps: usize, lr: f32) -> Vec<Tensor<f32>> {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let init_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let config = FacadeSgdConfig::new(lr)
        .with_momentum(0.9)
        .with_weight_decay(0.01)
        .with_nesterov(true);

    for _ in 0..steps {
        let tape = fandhe_ai::tape_for(device).unwrap();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        tape.step_device_param_store(&mut store, &grads, &config)
            .unwrap();
    }

    let final_tape = fandhe_ai::tape_for(device).unwrap();
    final_tape.sync_device_param_store_to_host(&store).unwrap()
}

/// `Sgd::step` + `apply_parameters`（ホスト参照実装。CPU）で同一初期化・
/// 同一データ・同一ハイパーパラメータの学習ループを回し、最終パラメータ
/// 列を返す。
fn train_host_reference(steps: usize, lr: f32) -> Vec<Tensor<f32>> {
    let mut model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let sgd_config = SgdConfig::new(lr)
        .with_momentum(0.9)
        .with_weight_decay(0.01)
        .with_nesterov(true);
    let mut sgd = Sgd::new(sgd_config).unwrap();

    for _ in 0..steps {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);
            let pred = bound.forward(&tape, &x).unwrap();
            let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs).unwrap()
        };
        model.apply_parameters(updated).unwrap();
    }

    model.trainable_parameters().into_iter().cloned().collect()
}

fn assert_params_match(device_params: &[Tensor<f32>], host_params: &[Tensor<f32>], ctx: &str) {
    assert_eq!(device_params.len(), host_params.len());
    for (i, (d, h)) in device_params.iter().zip(host_params.iter()).enumerate() {
        assert_eq!(d.shape(), h.shape(), "{ctx}: param {i} shape mismatch");
        let d_slice = d.contiguous();
        let h_slice = h.contiguous();
        let d_data = d_slice.as_slice().unwrap();
        let h_data = h_slice.as_slice().unwrap();
        for (j, (dv, hv)) in d_data.iter().zip(h_data.iter()).enumerate() {
            assert_close(*dv, *hv, &format!("{ctx}: param {i} element {j}"));
        }
    }
}

/// Metal 実機（macOS）での 100 step 累積・最終値判定 parity。
///
/// ```sh
/// cargo test -p fandhe-ai --test device_param_store_backend_parity -- --ignored --nocapture
/// ```
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn device_resident_matches_host_sgd_on_metal_across_100_steps() {
    let device_params = train_device_resident(Device::Metal, STEPS, LR);
    let host_params = train_host_reference(STEPS, LR);
    assert_params_match(&device_params, &host_params, "Metal vs host (100 steps)");
}

/// CUDA 実機（DGX Spark GB10 等）での 100 step 累積・最終値判定 parity。
///
/// ```sh
/// cargo test -p fandhe-ai --test device_param_store_backend_parity -- --ignored --nocapture
/// ```
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn device_resident_matches_host_sgd_on_cuda_across_100_steps() {
    let device_params = train_device_resident(Device::Cuda(0), STEPS, LR);
    let host_params = train_host_reference(STEPS, LR);
    assert_params_match(&device_params, &host_params, "CUDA vs host (100 steps)");
}

/// イシュー #1479 の AC-R2（バックエンド差異）を実機で確認する共通
/// ヘルパ: `device` で 1 step だけ forward・backward し、`resident_
/// capable`（`BackendOps::gemm_fp32_strict_into` を実装しているか）に
/// 応じて strict 版（`resident_grads_to_host`）の期待挙動を切り替える。
///
/// - `resident_capable == false`（CUDA。`gemm_fp32_strict_into` 未実装の
///   ため resident 経路に到達しない）: `BackendError::Unsupported` を
///   返すこと（panic なし）を検証する。
/// - `resident_capable == true`（CPU・Metal〈イシュー #1555〉）: 各
///   `Linear` 層の weight slot（`build_model` の層順・層内 weight →
///   bias の順序契約。`Sequential::init_device_param_store` doc 参照）が
///   `Some`、bias slot は resident 経由で充填されない（`gemm_fp32_
///   strict_into` は d_weight のみを対象とし bias 勾配は reduction 経由
///   のまま）ため `None` であることを検証する。
///
/// unified 版（`param_grads_to_host`）はいずれの場合も全パラメータの
/// 勾配を `Ok` で返し、host-only 参照実装（`Sgd::step` が使うのと同じ
/// `bound.trainable_grads`）と統一複合判定で一致することを検証する
/// （backend 間でカーネルが異なりうるため bit 一致ではなく複合判定）。
///
/// `docs/perf/train-resident-grad-device-update.md` §4「追補: #1479」・
/// イシュー本文の受入基準 AC-R4 に対応。本エージェント実行環境には
/// CUDA 実機がないため、`resident_capable == false` 側は未実測（未実測は
/// 下記個別テストの doc に明記）。
fn assert_grad_readout_contract(device: Device, resident_capable: bool) {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let init_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let tape = fandhe_ai::tape_for(device).unwrap();
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
    let grads = tape.backward_device_param_store(&loss, &store).unwrap();

    if resident_capable {
        // strict 版: `build_model`（2 層 Linear。層順に weight → bias）の
        // weight slot（index 0・2）は resident 経由で新鮮に充填され
        // `Some`、bias slot（index 1・3）は resident 経由で充填されない
        // ため `None`（`docs/device-resident-update-design.md` 追補
        // 「gemm_fp32_strict_into は d_weight のみが対象」）。
        let strict = tape.resident_grads_to_host(&store, &grads).unwrap();
        assert_eq!(
            strict.len(),
            4,
            "build_model は 2 層 Linear（各 weight・bias）で計 4 パラメータ"
        );
        for (i, slot) in strict.iter().enumerate() {
            let is_weight = i % 2 == 0;
            assert_eq!(
                slot.is_some(),
                is_weight,
                "param {i}（{}）の resident 充填状態が期待と異なる: {slot:?}",
                if is_weight { "weight" } else { "bias" }
            );
        }
    } else {
        // strict 版: CUDA は `gemm_fp32_strict_into` 未実装のため
        // resident 経路に到達せず、必ず `Unsupported`（panic なし）。
        let strict_err = tape.resident_grads_to_host(&store, &grads).unwrap_err();
        assert!(
            matches!(strict_err, fandhe_ai::BackendError::Unsupported(_)),
            "resident 未対応バックエンドでは Unsupported を返すはず（panic なし）: {strict_err:?}"
        );
    }

    // unified 版: 全パラメータの勾配を `Ok` で返す（`grads.get(...)`
    // フォールバックのみ。CUDA は resident 未充填のため全 slot
    // がこの経路を通る）。
    let device_grads_host = tape.param_grads_to_host(&store, &grads).unwrap();

    // host-only 参照実装（同一初期化・同一データの 1 step 目）。
    let host_model = build_model();
    let host_grads_host = {
        let host_tape = fandhe_ai::tape();
        let bound = host_model.bind(&host_tape);
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
            .collect::<Vec<Tensor<f32>>>()
    };

    assert_eq!(device_grads_host.len(), host_grads_host.len());
    for (i, (d, h)) in device_grads_host
        .iter()
        .zip(host_grads_host.iter())
        .enumerate()
    {
        assert_eq!(d.shape(), h.shape(), "param {i} shape mismatch");
        let d_slice = d.contiguous();
        let h_slice = h.contiguous();
        for (j, (dv, hv)) in d_slice
            .as_slice()
            .unwrap()
            .iter()
            .zip(h_slice.as_slice().unwrap())
            .enumerate()
        {
            assert_close(
                *dv,
                *hv,
                &format!("param {i} element {j} (grad readout contract)"),
            );
        }
    }
}

/// Metal 実機での AC-R2 契約検証（`assert_grad_readout_contract` 参照）。
///
/// イシュー #1479 実装セッションでは本エージェント実行環境に Metal 実機が
/// なかったため未実測のまま引き継がれていたが、**イシュー #1555（Metal
/// `gemm_fp32_strict_into` の実装）で Metal は resident_capable
/// （`resident_capable = true`）側へ移った**うえで M4 Max 実機実測 pass
/// 済み: strict 版 `resident_grads_to_host` が weight slot（index 0・2）
/// を `Some`・bias slot（index 1・3）を `None` で返すこと（`gemm_fp32_
/// strict_into` は NT/TN 限定・d_weight のみ対象）・統合版
/// `param_grads_to_host` が返す全パラメータ勾配がホスト参照実装と統一
/// 複合判定内で一致することの両方を確認した。
///
/// ```sh
/// cargo test -p fandhe-ai --test device_param_store_backend_parity -- --ignored --nocapture
/// ```
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn grad_readout_contract_on_metal() {
    assert_grad_readout_contract(Device::Metal, true);
}

/// CUDA 実機での AC-R2 契約検証（`assert_grad_readout_contract` 参照）。
///
/// イシュー #1479 実装セッションでは本エージェント実行環境に CUDA 実機が
/// なかったため未実測のまま引き継がれていたが、**イシュー #1480 の GB10
/// 実機実測（2026-09-09）で pass 済み**（`docs/perf/logs/
/// cuda-graph-step-grad-1480/grad_readout_contract_on_cuda.log`）:
/// strict 版 `resident_grads_to_host` が `BackendError::Unsupported` を
/// 返すこと（CUDA は `gemm_fp32_strict_into` 未実装のため resident
/// staging 未到達）・統合版 `param_grads_to_host` が返す全パラメータ
/// 勾配がホスト参照実装（`eval::matmul` ベース）と統一複合判定内で一致
/// することの両方を確認した。
///
/// ```sh
/// cargo test -p fandhe-ai --test device_param_store_backend_parity -- --ignored --nocapture
/// ```
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn grad_readout_contract_on_cuda() {
    assert_grad_readout_contract(Device::Cuda(0), false);
}
