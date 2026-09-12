//! イシュー #1577 受け入れ確認: CPU reuse 学習（`Op::LinearResident`・
//! `act == Relu`）の loss・各 step の重み勾配（`Tape::param_grads_to_host`）・
//! パラメータの bit 表現を標準出力へ出力する診断用テスト。
//!
//! `crates/facade/tests/metal_reuse_step_grad_bit_dump.rs`（同型テストの
//! Metal 版・#1555 で新設）を `Device::Cpu`（`cfg(target_os = "macos")`
//! 制約なし。CPU バックエンドは全 OS で利用可能）へ移植した CPU 版。
//! 目的は、`crates/autodiff/src/grad.rs::elementwise_mul_mask`（reuse
//! backward の ReLU マスク適用）を非連続 transpose view に stride 対応
//! させたイシュー #1577 の変更が、`Tape::param_grads_to_host` の戻り値・
//! loss・パラメータのいずれにも影響しない（bit 同一）ことを、
//! **本ファイルを origin/main と本ブランチの両方へコピーして実行**した
//! バイナリ出力の diff で機械的に確認すること。
//!
//! **`resident_grads_to_host`（strict 版）は比較対象に含めない**: CPU
//! バックエンドは resident staging（`gemm_fp32_strict_into`）へは
//! 到達せず常に `grads.get(...)` フォールバック経由になる
//! （`device_param_store_backend_parity.rs::assert_grad_readout_contract`
//! の `resident_capable = false` 分岐と同じ挙動）。本テストが検証したい
//! のは「`param_grads_to_host`（統合版。resident 未充填 slot は
//! `grads.get(...)` へフォールバックする設計のため常に `Ok` を返す）の
//! 戻り値が変更前後で bit 同一」という事実であり、strict 版の成否
//! （`Some`/`None`/`Unsupported` のいずれか）には依存させない。
//!
//! 実行コマンド（`#[ignore]` とするのは before/after ツリー間 diff 専用の
//! ためで、CPU 自体は実機不要。`--release` は速度のためで数値契約には
//! 影響しない）:
//!
//! ```sh
//! cargo test -p fandhe-ai --release --test cpu_reuse_step_grad_bit_dump \
//!   -- --ignored --nocapture
//! ```
//!
//! 出力行形式は `cuda_graph_step_common::print_bit_identity_report` と
//! 同一（`step[{i}].loss.bits`／`step[{s}].grad[{p}][{j}].bits`／
//! `step[{s}].param[{p}][{i}].bits`／`final.param[{p}][{i}].bits`）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, SgdConfig as FacadeSgdConfig};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_tensor_core::Tensor;

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;
const STEPS: usize = 10;
const LR: f32 = 0.05;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

/// bias あり 2 層 MLP（`Linear` → `ReLU` → `Linear`）。bias 勾配は
/// `gemm_fp32_strict_into`（d_weight 限定）の対象外のため常にホスト
/// フォールバック（`grads.get(...)`）経由になる。この経路が結線変更の
/// 前後で bit 同一であることも本テストの検証範囲に含めるため bias
/// ありの構成を使う（`train_on_cuda` doc コメントの `Op::LinearResident`
/// 契約と同じ理由）。
fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap()
}

/// [`train_on_cpu`] の戻り値型（clippy `type_complexity` 回避。
/// `(loss 列, 各 step の重み勾配〈`param_grads_to_host` の戻り値そのもの〉列,
/// 各 step 完了直後のパラメータ列, 最終パラメータ列)`）。
type TrainOnCpuResult = (
    Vec<f32>,
    Vec<Vec<Tensor<f32>>>,
    Vec<Vec<Tensor<f32>>>,
    Vec<Tensor<f32>>,
);

/// CPU reuse 学習ループを `STEPS` 回実行し、loss 列・各 step の重み
/// 勾配（`param_grads_to_host` の戻り値そのもの）列・各 step 完了直後の
/// パラメータ列・最終パラメータ列を返す。
fn train_on_cpu() -> TrainOnCpuResult {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let init_tape = fandhe_ai::tape_for(Device::Cpu).expect("CPU device must be available");
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let config = FacadeSgdConfig::new(LR);
    let mut log = Vec::with_capacity(STEPS);
    let mut per_step_grads = Vec::with_capacity(STEPS);
    let mut per_step_params = Vec::with_capacity(STEPS);

    for _ in 0..STEPS {
        let tape = fandhe_ai::tape_for(Device::Cpu).expect("CPU device must be available");
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);

        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        log.push(scalar(&loss.to_tensor()));

        let grads = tape.backward_device_param_store(&loss, &store).unwrap();

        // イシュー #1577 受け入れ確認の主眼: `elementwise_mul_mask` の
        // stride 対応が `param_grads_to_host`（resident 未充填 slot は
        // `grads.get(...)` フォールバック）の戻り値に影響しない
        // （変更前後で bit 同一である）こと。呼び出し窓は backward
        // 直後・`step_device_param_store` の前（`DeviceParamStore::
        // param_grads_to_host` doc の契約）。
        let step_grads = tape
            .param_grads_to_host(&store, &grads)
            .expect("param_grads_to_host: backward 直後・step 前の呼び出し窓内で呼んでいるはず");
        per_step_grads.push(step_grads);

        tape.step_device_param_store(&mut store, &grads, &config)
            .unwrap();

        let step_synced = tape.sync_device_param_store_to_host(&store).unwrap();
        per_step_params.push(step_synced);
    }

    let final_tape = fandhe_ai::tape_for(Device::Cpu).expect("CPU device must be available");
    let final_params = final_tape.sync_device_param_store_to_host(&store).unwrap();
    (log, per_step_grads, per_step_params, final_params)
}

/// loss 列・各 step の重み勾配列・各 step 完了直後のパラメータ列・最終
/// パラメータを `to_bits()` の 16 進表現で標準出力へ出す。行形式は
/// `cuda_graph_step_common::print_bit_identity_report` と揃えている
/// （プロセス間・ブランチ間の diff を単純化するため）。
fn print_bit_identity_report(
    log: &[f32],
    per_step_grads: &[Vec<Tensor<f32>>],
    per_step_params: &[Vec<Tensor<f32>>],
    final_params: &[Tensor<f32>],
) {
    println!("=== cpu_reuse_step_grad_bit_dump ===");
    for (i, loss) in log.iter().enumerate() {
        println!("step[{i}].loss.bits = {:#010x}", loss.to_bits());
    }
    for (step, grads) in per_step_grads.iter().enumerate() {
        for (p, t) in grads.iter().enumerate() {
            let contiguous = t.contiguous();
            let slice = contiguous.as_slice().unwrap_or(&[]);
            for (j, v) in slice.iter().enumerate() {
                println!("step[{step}].grad[{p}][{j}].bits = {:#010x}", v.to_bits());
            }
        }
    }
    for (step, params) in per_step_params.iter().enumerate() {
        for (p, t) in params.iter().enumerate() {
            let contiguous = t.contiguous();
            let slice = contiguous.as_slice().unwrap_or(&[]);
            for (i, v) in slice.iter().enumerate() {
                println!("step[{step}].param[{p}][{i}].bits = {:#010x}", v.to_bits());
            }
        }
    }
    for (p, t) in final_params.iter().enumerate() {
        let contiguous = t.contiguous();
        let slice = contiguous.as_slice().unwrap_or(&[]);
        for (i, v) in slice.iter().enumerate() {
            println!("final.param[{p}][{i}].bits = {:#010x}", v.to_bits());
        }
    }
}

#[test]
#[ignore = "before/after ツリー間 diff 専用（通常 CI では不要）。--nocapture 必須"]
fn dump_cpu_reuse_step_grad_bits() {
    let (log, per_step_grads, per_step_params, final_params) = train_on_cpu();
    print_bit_identity_report(&log, &per_step_grads, &per_step_params, &final_params);
}
