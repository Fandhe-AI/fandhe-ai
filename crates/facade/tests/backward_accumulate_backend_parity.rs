//! `Tape::backward_accumulate`（`fandhe_ai::Tape::backward_accumulate`
//! への薄い委譲）の facade 到達経路の parity テスト（イシュー #1749。
//! `no_grad_detach_backend_parity.rs` と同型）。
//!
//! フィクスチャは `mul`／`sum`（属性なし。CPU 本番 ops と naive 参照
//! 実装の bit 完全一致対象）と `add`（`#[ignore]` の Metal／CUDA
//! parity。`matmul` は CPU BLIS 経路が NaiveOps と bit 一致しないため
//! 使わない）のみを使う。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で、同一 loss を
//!   2 回蓄積した勾配（`2g`）が bit 完全一致することを確認する。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU tape
//!   と比較する。実機実測は本エージェント実行環境に到達手段が無い
//!   ため未実施（PR 本文に申し送り）。

use fandhe_ai::Device;
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn contiguous_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// `loss = sum(x * w)` を 2 回蓄積した `w` 勾配（`2x`）を CPU 本番 ops
/// （`fandhe_ai::tape()`）で計算する。
#[test]
fn cpu_backward_accumulate_weight_grad_matches_naive_reference() {
    let x_data = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let w_data = t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.var(&x_data);
    let w_cpu = cpu_tape.var(&w_data);
    let loss_cpu = x_cpu.mul(&w_cpu).expect("同 shape の要素積は失敗しない");
    let loss_cpu = loss_cpu.sum(None).expect("全軸縮約は失敗しない");
    let mut grads_cpu = cpu_tape
        .backward(&loss_cpu)
        .expect("w が追跡対象のため成功する");
    cpu_tape
        .backward_accumulate(&loss_cpu, &mut grads_cpu)
        .expect("同一グラフへの 2 回目の backward は成功する");
    let dw_cpu = grads_cpu
        .get(&w_cpu)
        .expect("w は requires_grad=true の葉")
        .expect("w は loss に到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.var(&x_data);
    let w_naive = naive_tape.var(&w_data);
    let loss_naive = x_naive
        .mul(&w_naive)
        .expect("同 shape の要素積は失敗しない");
    let loss_naive = loss_naive.sum(None).expect("全軸縮約は失敗しない");
    let mut grads_naive = naive_tape
        .backward(&loss_naive)
        .expect("w が追跡対象のため成功する");
    naive_tape
        .backward_accumulate(&loss_naive, &mut grads_naive)
        .expect("同一グラフへの 2 回目の backward は成功する");
    let dw_naive = grads_naive
        .get(&w_naive)
        .expect("w は requires_grad=true の葉")
        .expect("w は loss に到達する");

    assert_eq!(
        contiguous_bits(dw_cpu),
        contiguous_bits(dw_naive),
        "backward_accumulate の w 勾配は CPU 本番 ops と naive 参照実装で bit 一致するはず"
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---
//
// `no_grad_detach_backend_parity.rs` と同じ理由で `Device::Metal`
// variant を参照するテスト関数のみ `cfg(target_os = "macos")` で
// コンパイル自体を限定する。実機実測は本エージェント実行環境に到達
// 手段が無いため未実施（PR 本文に申し送り）。

fn backward_accumulate_weight_grad_bits_on(device: Device, seed_offset: f32) -> Vec<u32> {
    let x_data = t(vec![1.0 + seed_offset, 2.0, 3.0, 4.0], &[2, 2]);
    let w_data = t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.var(&x_data);
    let w = tape.var(&w_data);
    let loss = x.mul(&w).expect("同 shape の要素積は失敗しない");
    let loss = loss.sum(None).expect("全軸縮約は失敗しない");
    let mut grads = tape.backward(&loss).expect("w が追跡対象のため成功する");
    tape.backward_accumulate(&loss, &mut grads)
        .expect("同一グラフへの 2 回目の backward は成功する");
    let dw = grads
        .get(&w)
        .expect("w は requires_grad=true の葉")
        .expect("w は loss に到達する");
    contiguous_bits(dw)
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_backward_accumulate_weight_grad_matches_cpu() {
    let metal_bits = backward_accumulate_weight_grad_bits_on(Device::Metal, 0.0);
    let cpu_bits = backward_accumulate_weight_grad_bits_on(Device::Cpu, 0.0);
    assert_eq!(
        metal_bits, cpu_bits,
        "backward_accumulate の w 勾配は Metal と CPU で bit 一致するはず（mul の要素積のみ）"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_backward_accumulate_weight_grad_matches_cpu() {
    let cuda_bits = backward_accumulate_weight_grad_bits_on(Device::Cuda(0), 0.0);
    let cpu_bits = backward_accumulate_weight_grad_bits_on(Device::Cpu, 0.0);
    assert_eq!(
        cuda_bits, cpu_bits,
        "backward_accumulate の w 勾配は CUDA と CPU で bit 一致するはず（mul の要素積のみ）"
    );
}
