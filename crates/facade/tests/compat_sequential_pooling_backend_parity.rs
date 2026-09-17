//! `compat::Sequential::add_max_pool2d`／`add_max_pool1d`／
//! `add_avg_pool2d`／`add_avg_pool1d`／`add_adaptive_avg_pool2d`／
//! `add_adaptive_avg_pool1d`（イシュー #1957・親 #1618）の CUDA／Metal
//! parity テスト（`compat_sequential_layers_backend_parity.rs` と同型）。
//!
//! CPU 側の正しさ検証（`predict`／`forward` bit 完全一致・直接呼び出し
//! との突合・backward・学習ループ・常駐経路 fail-closed）は
//! `compat_sequential_pooling.rs` が Linux 実行可能な形で既に担う。
//! 本ファイルは新規カーネルを一切追加していない構成（MaxPool／
//! AvgPool／AdaptiveAvgPool の各バックエンドカーネルは #1729（CUDA）・
//! #1730（Metal）で実装済み・2026-09-16 実機実測済み〈#1902／#1903〉。
//! `docs/compat-api-scope.md` §1.2「Pooling」行参照）を前提に、
//! `compat::Sequential` 経由で組んだモデルの forward が CPU と
//! `assert_parity`（REQ-2 統一複合判定）で一致することのみを確認する。
//!
//! 実機実測は本エージェントの実行環境に CUDA／Metal 実機への到達
//! 手段がないため未実施のまま Mac／GB10 セッションへ申し送る
//! （`docs/compat-api-scope.md` §1.2「#1957」追記・PR 本文に明記）。

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

/// MaxPool2d→AvgPool2d→AdaptiveAvgPool2d を 1 モデルに混在させた
/// 構成の forward を CPU と対象デバイスで比較する（`[N, C, H, W]`）。
fn run_pool2d_chain_parity(device: Device) {
    let model = Sequential::new()
        .add_max_pool2d([2, 2], None, [0, 0], [1, 1])
        .unwrap()
        .add_avg_pool2d([2, 2], None, [0, 0], true)
        .unwrap()
        .add_adaptive_avg_pool2d([1, 1])
        .unwrap();
    let x = tensor(
        (0..2 * 3 * 8 * 8)
            .map(|i| (i as f32) * 0.01 - 0.3)
            .collect(),
        &[2, 3, 8, 8],
    );

    let cpu_out = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(&x);
    let device_out = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_parity(
        "compat::Sequential(MaxPool2d→AvgPool2d→AdaptiveAvgPool2d) device vs CPU",
        &dense(&device_out),
        &dense(&cpu_out),
    );
}

/// MaxPool1d→AvgPool1d→AdaptiveAvgPool1d を 1 モデルに混在させた
/// 構成の forward を CPU と対象デバイスで比較する（`[N, C, L]`）。
fn run_pool1d_chain_parity(device: Device) {
    let model = Sequential::new()
        .add_max_pool1d(2, None, 0, 1)
        .unwrap()
        .add_avg_pool1d(2, None, 0, true)
        .unwrap()
        .add_adaptive_avg_pool1d(1)
        .unwrap();
    let x = tensor(
        (0..2 * 3 * 8).map(|i| (i as f32) * 0.01 - 0.3).collect(),
        &[2, 3, 8],
    );

    let cpu_out = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(&x);
    let device_out = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_parity(
        "compat::Sequential(MaxPool1d→AvgPool1d→AdaptiveAvgPool1d) device vs CPU",
        &dense(&device_out),
        &dense(&cpu_out),
    );
}

// --- CUDA（本エージェント実行環境に実機なし。GB10 セッションへ申し送り） ---

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #1957 の申し送り先（GB10 セッション）\
            へ引き継ぐ"]
fn cuda_pool2d_chain_matches_cpu() {
    run_pool2d_chain_parity(Device::Cuda(0));
}

#[test]
#[ignore = "CUDA 実機必須"]
fn cuda_pool1d_chain_matches_cpu() {
    run_pool1d_chain_parity(Device::Cuda(0));
}

// --- Metal（本エージェント実行環境に実機なし。Mac セッションへ申し送り） ---

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実行は #1957 の申し送り先（Mac セッション）へ \
            引き継ぐ"]
fn metal_pool2d_chain_matches_cpu() {
    run_pool2d_chain_parity(Device::Metal);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機必須"]
fn metal_pool1d_chain_matches_cpu() {
    run_pool1d_chain_parity(Device::Metal);
}
