//! `fandhe_ai::Tape::transfer`（`Var::to_tape` の facade 到達経路。
//! イシュー #1614）のバックエンド横断テスト（`cast_backend_parity.rs`
//! と同型。`docs/facade-device-transfer-enumeration-design.md` §5
//! 「テスト」(C) 節）。
//!
//! 転送は算術を含まないホスト値の受け渡しのため bit 完全一致契約
//! （`crates/autodiff/tests/device_transfer.rs` の A4 と同じ検証方針）。
//!
//! - 属性なし: `fandhe_ai::tape()`（CPU）↔ `fandhe_ai_autodiff::
//!   Tape::new()`（naive CPU 参照実装）間で `Tape::transfer` の値が
//!   bit 一致することを確認する。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` から CPU tape へ
//!   `matmul` 結果を転送し、転送バイト自体が転送元 `to_tensor()` と
//!   bit 一致すること（純粋な値渡しのため）、および CPU 計算値とは
//!   REQ-2 統一複合判定で比較することを確認する。実機実測は本
//!   エージェント実行環境に CUDA／Metal 実機がないため未実施のまま
//!   Mac／GB10 セッションへ申し送る。

use fandhe_ai::Device;
use fandhe_ai_tensor_core::Tensor;

fn f32_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

// --- 属性なし: facade CPU tape <-> naive autodiff tape ---

/// `Tape::transfer` で naive tape 上の値を facade CPU tape へ転送すると
/// bit 完全一致する（`Tape::device()` が両者とも `Device::Cpu` になる
/// ことも併せて確認する）。
#[test]
fn transfer_from_naive_tape_to_facade_cpu_tape_is_bit_exact() {
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_data = Tensor::new(vec![1.0, -2.5, 0.0, 3.5], &[2, 2])
        .expect("test fixture: shape とデータ長は事前に一致させている");
    let x = naive_tape.var(&x_data);
    assert_eq!(
        x.device(),
        Device::Cpu,
        "naive tape も Device::Cpu を報告する"
    );

    let facade_tape = fandhe_ai::tape();
    assert_eq!(facade_tape.device(), Device::Cpu);

    let x_on_facade = facade_tape
        .transfer(&x)
        .expect("同じ Device::Cpu 間でも別 Tape のため実データ転送が行われる");
    assert_eq!(
        f32_bits(&x_on_facade.to_tensor()),
        f32_bits(&x.to_tensor()),
        "転送は算術を含まないため bit 完全一致するはず"
    );
}

/// 逆方向（facade CPU tape → naive tape）も同様に bit 完全一致する。
/// `lazy` な elementwise 連鎖（`add` の結果）を転送対象にして、
/// `to_tape` が転送前に実体化することも併せて確認する。
#[test]
fn transfer_from_facade_cpu_tape_to_naive_tape_materializes_and_is_bit_exact() {
    let facade_tape = fandhe_ai::tape();
    let a_data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("test fixture: shape とデータ長は事前に一致させている");
    let b_data = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])
        .expect("test fixture: shape とデータ長は事前に一致させている");
    let a = facade_tape.var(&a_data);
    let b = facade_tape.var(&b_data);
    let sum = a.add(&b).expect("同 shape の加算は失敗しない");
    let expected = sum.to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let sum_on_naive = sum
        .to_tape(&naive_tape)
        .expect("lazy チェーンでも実体化されて転送される");
    assert_eq!(f32_bits(&sum_on_naive.to_tensor()), f32_bits(&expected));
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

/// `device` に結線した tape 上で `matmul` を計算し、その結果を CPU
/// tape へ `Tape::transfer` した値を返す（転送値と、転送先で
/// `to_tensor()` した値が bit 一致することも合わせて検証してから
/// 戻す）。
fn matmul_transferred_to_cpu(device: Device) -> (Tensor<f32>, Tensor<f32>) {
    let source_tape = fandhe_ai::tape_for(device)
        .expect("実機が利用可能な前提のテストのため tape_for は成功するはず");
    let a_data = Tensor::new((0..16).map(|i| i as f32 * 0.5).collect(), &[4, 4])
        .expect("test fixture: shape とデータ長は事前に一致させている");
    let b_data = Tensor::new((0..16).map(|i| (16 - i) as f32 * 0.25).collect(), &[4, 4])
        .expect("test fixture: shape とデータ長は事前に一致させている");
    let a = source_tape.var(&a_data);
    let b = source_tape.var(&b_data);
    let out = a.matmul(&b).expect("4x4 同士の matmul は失敗しない");
    let source_value = out.to_tensor();

    let cpu_tape = fandhe_ai::tape();
    let out_on_cpu = cpu_tape
        .transfer(&out)
        .expect("実機から CPU tape への転送は実体化済みノードなら失敗しない");
    let transferred_value = out_on_cpu.to_tensor();
    assert_eq!(
        f32_bits(&transferred_value),
        f32_bits(&source_value),
        "転送バイト自体は転送元 to_tensor() と bit 一致するはず（純粋な値渡し）"
    );

    let cpu_reference_tape = fandhe_ai::tape();
    let a_cpu = cpu_reference_tape.var(&a_data);
    let b_cpu = cpu_reference_tape.var(&b_data);
    let cpu_native = a_cpu
        .matmul(&b_cpu)
        .expect("4x4 同士の matmul は失敗しない")
        .to_tensor();

    (transferred_value, cpu_native)
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`cast_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない。イシュー #1614"]
fn metal_matmul_transferred_to_cpu_matches_within_req2_tolerance() {
    let (transferred, cpu_native) = matmul_transferred_to_cpu(Device::Metal);
    fandhe_ai_backend_cpu::assert_parity(
        "metal_matmul_transferred_to_cpu",
        transferred.contiguous().host_slice().as_ref(),
        cpu_native.contiguous().host_slice().as_ref(),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。イシュー #1614"]
fn cuda_matmul_transferred_to_cpu_matches_within_req2_tolerance() {
    let (transferred, cpu_native) = matmul_transferred_to_cpu(Device::Cuda(0));
    fandhe_ai_backend_cpu::assert_parity(
        "cuda_matmul_transferred_to_cpu",
        transferred.contiguous().host_slice().as_ref(),
        cpu_native.contiguous().host_slice().as_ref(),
    );
}
