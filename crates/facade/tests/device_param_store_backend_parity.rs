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

/// strict 版 `resident_grads_to_host` が bias slot に何を返すかを
/// バックエンド別に表す（イシュー #1898）。
///
/// `gemm_fp32_strict_into_with_bias_reduce_tracked`（`BackendOps` の
/// trait 既定実装は bias を無視し常に `Ok(false)` を返す）を Metal
/// のみがオーバーライドし、NT/TN（encode-only カーネル）・NN/TT
/// フォールバック（ホスト縮約 → `upload_into`）のいずれの経路でも
/// bias 縮約を resident staging へ書く（イシュー #1566・PR #1659。
/// `crates/backend-metal/src/ops.rs::MetalBackendOps::gemm_fp32_
/// strict_into_with_bias_reduce_tracked` doc 参照）。CPU・CUDA は
/// この trait メソッドをオーバーライドしないため、bias 縮約は常に
/// 通常の `Gradients` 経由（`DeviceParamStore::fill_resident_weight_
/// grad` の `!outcome.bias_filled` 分岐）に留まる。
#[derive(Clone, Copy)]
enum StrictBiasExpectation {
    /// CPU・CUDA: bias slot は resident 経由で充填されないため `None`。
    HostRouted,
    /// Metal（#1566 以降）: bias slot も resident 経由で充填されるため
    /// `Some`。`grad_readout_contract_on_metal`
    /// （`#[cfg(target_os = "macos")]`）のみが構築するため、非 macOS
    /// ビルドでは未構築（`dead_code` lint 抑制。他バックエンド用の
    /// `#[cfg]` ゲート付きテストと同型。`device_param_store_grad_
    /// readout.rs::assert_var_type_is_reexported` 参照）。
    #[allow(dead_code)]
    Resident,
}

/// イシュー #1479 の AC-R2（バックエンド差異）を実機で確認する共通
/// ヘルパ: `device` で 1 step だけ forward・backward し、
/// `bias_expectation`（[`StrictBiasExpectation`]）に応じて strict 版
/// （`resident_grads_to_host`）の bias slot 期待挙動を切り替える。
///
/// weight slot（`build_model` の層順・層内 weight → bias の順序契約。
/// `Sequential::init_device_param_store` doc 参照）は CPU・Metal
/// 〈#1555〉・CUDA〈#1559〉のいずれも `gemm_fp32_strict_into` を実装済み
/// のため resident 経路に到達し常に `Some`。bias slot のみバックエンド
/// によって異なる（`StrictBiasExpectation` doc 参照）。
///
/// **`resident_grad_capability` が `Some(false)`（`Unsupported`）となる
/// 分岐**: 本ヘルパの呼び出し元（CPU・Metal・CUDA）はいずれも resident
/// 経路に到達するためこの分岐は再現しない。同分岐の回帰カバレッジは
/// `crates/autodiff/src/optim/device_store.rs` の単体テスト
/// `resident_grads_to_host_is_unsupported_on_backend_without_gemm_into`
/// （`MockDeviceOps::new()`。`gemm_fp32_strict_into` 未実装のモック）に
/// 残る。
///
/// unified 版（`param_grads_to_host`）はいずれの場合も全パラメータの
/// 勾配を `Ok` で返し、host-only 参照実装（`Sgd::step` が使うのと同じ
/// `bound.trainable_grads`）と統一複合判定で一致することを検証する
/// （backend 間でカーネルが異なりうるため bit 一致ではなく複合判定）。
///
/// `docs/perf/train-resident-grad-device-update.md` §4「追補: #1479」・
/// イシュー本文の受入基準 AC-R4 に対応。本エージェント実行環境には
/// CUDA・Metal 実機がないため、実機再実測は未実施のまま引き継ぐ
/// （未実測は下記個別テストの doc に明記）。
fn assert_grad_readout_contract(device: Device, bias_expectation: StrictBiasExpectation) {
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

    // strict 版: `build_model`（2 層 Linear。層順に weight → bias）の
    // weight slot（index 0・2）は resident 経由で新鮮に充填され常に
    // `Some`。bias slot（index 1・3）は `bias_expectation` に応じて
    // `Some`（Metal）／`None`（CPU・CUDA）となる
    // （`StrictBiasExpectation` doc 参照）。
    let strict = tape.resident_grads_to_host(&store, &grads).unwrap();
    assert_eq!(
        strict.len(),
        4,
        "build_model は 2 層 Linear（各 weight・bias）で計 4 パラメータ"
    );
    for (i, slot) in strict.iter().enumerate() {
        let is_weight = i % 2 == 0;
        let expected_some =
            is_weight || matches!(bias_expectation, StrictBiasExpectation::Resident);
        assert_eq!(
            slot.is_some(),
            expected_some,
            "param {i}（{}）の resident 充填状態が期待と異なる: {slot:?}",
            if is_weight { "weight" } else { "bias" }
        );
    }

    // strict 版の bias slot が `Some` の場合（Metal）、統合版
    // `param_grads_to_host` の同 index と bit 完全一致することを確認
    // する（同じ staging バッファからの download のはず。イシュー
    // #1898）。ホスト参照実装との比較は下記のとおり統一複合判定の
    // ままで、ここでは strict 版と統合版の内部整合のみを bit 単位で
    // 検証する。
    if matches!(bias_expectation, StrictBiasExpectation::Resident) {
        let unified_preview = tape.param_grads_to_host(&store, &grads).unwrap();
        for (i, slot) in strict.iter().enumerate() {
            if let Some(strict_tensor) = slot {
                let strict_bits: Vec<u32> = strict_tensor
                    .contiguous()
                    .as_slice()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                let unified_bits: Vec<u32> = unified_preview[i]
                    .contiguous()
                    .as_slice()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                assert_eq!(
                    strict_bits, unified_bits,
                    "param {i}: strict 版と統合版の resident 充填値が bit 一致しない"
                );
            }
        }
    }

    // unified 版: 全パラメータの勾配を `Ok` で返す（resident 未充填の
    // slot は `grads.get(...)` フォールバックへ回る）。
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
/// なかったため未実測のまま引き継がれていたが、イシュー #1555（Metal
/// `gemm_fp32_strict_into` の実装）で Metal は resident 経路（weight
/// slot は resident 経由で `Some`）へ移った時点で M4 Max 実機実測 pass
/// 済みだった。
///
/// **イシュー #1898 で是正（本テスト自体の期待を更新）**: その後イシュー
/// #1566（PR #1659。Metal `gemm_fp32_strict_into_with_bias_reduce_
/// tracked` オーバーライドの追加）により、Metal は bias slot（index
/// 1・3）も resident 経由で `Some` を返すようになった。本テストは
/// 従来「Metal も CPU・CUDA と同じく bias slot は `None`」という古い
/// 期待のまま据え置かれていたため、#1566 マージ後の main
/// （`e851e91a..565300e4`）で「param 1（bias）の resident 充填状態が
/// 期待と異なる」で FAIL していた（原因コミット `98c3c67e`）。
/// `assert_grad_readout_contract` の `bias_expectation` 引数を
/// `StrictBiasExpectation::Resident` へ更新し、strict 版
/// `resident_grads_to_host` が weight slot（index 0・2）・bias slot
/// （index 1・3）とも `Some` で返すこと（`gemm_fp32_strict_into_
/// with_bias_reduce_tracked` は NT/TN・NN/TT フォールバックいずれも
/// bias 縮約を resident staging へ書く）・その値が統合版
/// `param_grads_to_host` の同 index と bit 完全一致すること・統合版が
/// 返す全パラメータ勾配がホスト参照実装と統一複合判定内で一致すること
/// を検証する（M4 Max 実機実測は未実施のまま Mac セッションへ申し送り。
/// `docs/perf/logs/grad-readout-contract-1898/` 参照）。
///
/// ```sh
/// cargo test -p fandhe-ai --test device_param_store_backend_parity -- --ignored --nocapture
/// ```
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn grad_readout_contract_on_metal() {
    assert_grad_readout_contract(Device::Metal, StrictBiasExpectation::Resident);
}

/// CUDA 実機での AC-R2 契約検証（`assert_grad_readout_contract` 参照）。
///
/// イシュー #1479 実装セッションでは本エージェント実行環境に CUDA 実機が
/// なかったため未実測のまま引き継がれていたが、イシュー #1480 の GB10
/// 実機実測（2026-09-09）で pass 済みだった（`docs/perf/logs/
/// cuda-graph-step-grad-1480/grad_readout_contract_on_cuda.log`）。その後
/// イシュー #1559 で CUDA が `gemm_fp32_strict_into` を実装したことにより
/// `resident_grad_capability` が `Some(true)` へ確定し、weight slot は
/// resident 経由で `Some` を返す側へ移行した（`crates/backend-cuda/src/
/// ops.rs::CudaBackendOps::gemm_fp32_strict_into` doc 参照）。
///
/// **CUDA は bias slot は `None` のまま（イシュー #1898 で確認・不変）**:
/// CUDA は `gemm_fp32_strict_into_with_bias_reduce_tracked` を
/// オーバーライドしない（trait 既定実装のまま）ため、bias 縮約は
/// resident staging へ書かれず常に通常の `Gradients` 経由。Metal
/// （イシュー #1566 以降）とは異なりこの契約は変わっていない
/// （`StrictBiasExpectation::HostRouted`）。本エージェント実行環境には
/// CUDA 実機がないため、この契約下での実機再実測は未実施のまま
/// 引き継ぐ（`docs/perf/logs/grad-readout-contract-1898/` 参照）。
///
/// ```sh
/// cargo test -p fandhe-ai --test device_param_store_backend_parity -- --ignored --nocapture
/// ```
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn grad_readout_contract_on_cuda() {
    assert_grad_readout_contract(Device::Cuda(0), StrictBiasExpectation::HostRouted);
}
