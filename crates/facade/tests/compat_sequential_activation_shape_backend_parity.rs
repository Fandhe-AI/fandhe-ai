//! `compat::Sequential::add_softmax`／`add_log_softmax`／`add_gelu`／
//! `add_gelu_tanh`／`add_softplus`／`add_flatten`（イシュー #2065・親
//! #2059）の CUDA／Metal parity テスト（`compat_sequential_layers_
//! backend_parity.rs` と同型）。
//!
//! CPU 側の正しさ検証（`predict`／`forward` bit 完全一致・遅延検査・
//! 常駐経路・無状態層の確認）は `compat_sequential_activation_shape.rs`
//! が Linux 実行可能な形で既に担う。本ファイルは新規カーネルを一切
//! 追加していない構成（`Softmax`／`LogSoftmax`／`Gelu`／`GeluTanh`／
//! `Softplus` の各バックエンドカーネルは #1594／#1713 で実装済み）を
//! 前提に、`compat::Sequential` 経由で組んだモデルの forward が CPU と
//! `assert_parity`（REQ-2 統一複合判定）で一致することのみを確認する。
//!
//! `Flatten` は算術を一切含まない view 演算（`Var::reshape` への委譲）
//! のため、5 活性化層と異なり `assert_parity`（許容誤差付き判定）では
//! なく **`assert_eq!`（dense ベクトルの厳密一致）** を要求する。
//!
//! 実機実測は本エージェントの実行環境に CUDA／Metal 実機への到達
//! 手段がないため未実施のまま Mac／GB10 セッションへ申し送る
//! （`docs/perf/logs/compat-sequential-activation-shape-2065/`）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tensor};
use fandhe_ai_backend_cpu::parity::assert_parity;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

const SEED1: u64 = 0x5555_6666;
const SEED2: u64 = 0x7777_8888;

/// `Flatten(1,2) → Linear → ReLU → Softmax(1) → Linear`（`compat_
/// sequential_activation_shape.rs::mixed_model` と同じ構成）の forward
/// を CPU と対象デバイスで比較する。`Flatten` を含むためカーネル
/// 実装なしで常駐経路も含めた全体（`predict`。tape 不要経路）を対象に
/// 数値のみ突合する（`assert_parity` は REQ-2 の許容誤差付き判定）。
fn run_flatten_softmax_parity(device: Device) {
    let model = Sequential::new()
        .add_flatten(1, 2)
        .add_linear(6, 4, SEED1)
        .unwrap()
        .add_relu()
        .add_softmax(1)
        .add_linear(4, 2, SEED2)
        .unwrap();
    let x = tensor(
        (0..2 * 2 * 3).map(|i| i as f32 * 0.1 - 0.7).collect(),
        &[2, 2, 3],
    );

    let cpu_out = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(&x);
    let device_out = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_parity(
        "compat::Sequential(Flatten→Linear→ReLU→Softmax→Linear) device vs CPU",
        &dense(&device_out),
        &dense(&cpu_out),
    );
}

/// `LogSoftmax`／`Gelu`／`GeluTanh`／`Softplus` を 1 モデルに混在させた
/// 構成の forward を CPU と対象デバイスで比較する。
fn run_log_softmax_gelu_softplus_parity(device: Device) {
    let model = Sequential::new()
        .add_linear(4, 4, SEED1)
        .unwrap()
        .add_gelu()
        .add_linear(4, 4, SEED2)
        .unwrap()
        .add_gelu_tanh()
        .add_softplus(1.0, 20.0)
        .unwrap()
        .add_log_softmax(1);
    let x = tensor((0..3 * 4).map(|i| i as f32 * 0.2 - 0.5).collect(), &[3, 4]);

    let cpu_out = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(&x);
    let device_out = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_parity(
        "compat::Sequential(Gelu→GeluTanh→Softplus→LogSoftmax) device vs CPU",
        &dense(&device_out),
        &dense(&cpu_out),
    );
}

/// `Flatten` 単独（算術を含まない view 演算）の forward が CPU と対象
/// デバイスで **bit 完全一致**することを確認する（`assert_parity` の
/// 許容誤差付き判定ではなく `assert_eq!` の厳密一致を要求する）。
fn run_flatten_only_bit_exact(device: Device) {
    let model = Sequential::new().add_flatten(1, 2);
    let x = tensor(
        (0..2 * 2 * 3).map(|i| i as f32 * 0.1 - 0.7).collect(),
        &[2, 2, 3],
    );

    let cpu_out = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(&x);
    let device_out = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_eq!(dense(&device_out), dense(&cpu_out));
}

// --- CUDA（本エージェント実行環境に実機なし。GB10 セッションへ申し送り） ---

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #2065 の申し送り先（GB10 セッション）\
            へ引き継ぐ"]
fn cuda_flatten_softmax_matches_cpu() {
    run_flatten_softmax_parity(Device::Cuda(0));
}

#[test]
#[ignore = "CUDA 実機必須"]
fn cuda_log_softmax_gelu_softplus_matches_cpu() {
    run_log_softmax_gelu_softplus_parity(Device::Cuda(0));
}

#[test]
#[ignore = "CUDA 実機必須"]
fn cuda_flatten_only_bit_exact() {
    run_flatten_only_bit_exact(Device::Cuda(0));
}

// --- Metal（本エージェント実行環境に実機なし。Mac セッションへ申し送り） ---

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実行は #2065 の申し送り先（Mac セッション）へ \
            引き継ぐ"]
fn metal_flatten_softmax_matches_cpu() {
    run_flatten_softmax_parity(Device::Metal);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機必須"]
fn metal_log_softmax_gelu_softplus_matches_cpu() {
    run_log_softmax_gelu_softplus_parity(Device::Metal);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機必須"]
fn metal_flatten_only_bit_exact() {
    run_flatten_only_bit_exact(Device::Metal);
}
