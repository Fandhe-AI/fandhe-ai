//! `Tape::var_no_grad`（`fandhe_ai::Tape::var_no_grad` への薄い委譲）・
//! `Var::detach`（既存 `Var` 再エクスポート経由）の facade 到達経路の
//! parity テスト（イシュー #1748。`shape_ops_backend_parity.rs` と
//! 同型）。
//!
//! 本演算は算術を一切伴わない（テープのメタ情報〈`requires_grad`〉の
//! みが対象）ため、CPU 本番 ops（`CpuBackendOps`）と naive 参照実装
//! （`NaiveOps`）の間で勾配値が **bit 完全一致**することを確認する
//! （`.claude/rules/coding-rust.md` の「数値一致は複合判定」の対象
//! 外——両者とも同じホスト演算〈`matmul`／`add`／`mul`〉を通した値の
//! 比較であり、本テストが検証するのは `requires_grad` フラグの伝播・
//! 蓄積スキップ・エラー型が両バックエンドで一致することそのもの）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で突き合わせる。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU tape
//!   と比較する。実機実測は本エージェント実行環境に到達手段が無い
//!   ため未実施（PR 本文に申し送り）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::AutodiffError;
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

/// `loss = sum(x.detach() * w)` の `w` 勾配（`x` の値そのもの）を
/// CPU 本番 ops（`fandhe_ai::tape()`）で計算する。
#[test]
fn cpu_detach_weight_grad_matches_naive_reference() {
    let x_data = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let w_data = t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.var(&x_data);
    let w_cpu = cpu_tape.var(&w_data);
    let x_cpu_detached = x_cpu.detach().expect("detach は失敗しない");
    let product_cpu = x_cpu_detached
        .mul(&w_cpu)
        .expect("同 shape の要素積は失敗しない");
    let loss_cpu = product_cpu.sum(None).expect("全軸縮約は失敗しない");
    let grads_cpu = cpu_tape
        .backward(&loss_cpu)
        .expect("w が追跡対象のため成功する");
    let dw_cpu = grads_cpu
        .get(&w_cpu)
        .expect("w は requires_grad=true の葉")
        .expect("w は loss に到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.var(&x_data);
    let w_naive = naive_tape.var(&w_data);
    let x_naive_detached = x_naive.detach().expect("detach は失敗しない");
    let product_naive = x_naive_detached
        .mul(&w_naive)
        .expect("同 shape の要素積は失敗しない");
    let loss_naive = product_naive.sum(None).expect("全軸縮約は失敗しない");
    let grads_naive = naive_tape
        .backward(&loss_naive)
        .expect("w が追跡対象のため成功する");
    let dw_naive = grads_naive
        .get(&w_naive)
        .expect("w は requires_grad=true の葉")
        .expect("w は loss に到達する");

    assert_eq!(
        contiguous_bits(dw_cpu),
        contiguous_bits(dw_naive),
        "detach の w 勾配は CPU 本番 ops と naive 参照実装で bit 一致するはず"
    );

    // `x_cpu_detached` は requires_grad=false の葉のためエラー型も一致。
    let err_cpu = grads_cpu.get(&x_cpu_detached).unwrap_err();
    let err_naive = grads_naive.get(&x_naive_detached).unwrap_err();
    assert!(matches!(err_cpu, AutodiffError::GradientTrackingDisabled));
    assert!(matches!(err_naive, AutodiffError::GradientTrackingDisabled));
}

/// `Tape::var_no_grad`（facade ラッパー）を通した入力の weight／bias
/// 勾配が CPU 本番 ops と naive 参照実装で bit 一致すること。
#[test]
fn cpu_var_no_grad_weight_grad_matches_naive_reference() {
    let x_data = t(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]);
    let w_data = t(vec![0.1, 0.2, -0.3, 0.4], &[2, 2]);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.var_no_grad(&x_data);
    let w_cpu = cpu_tape.var(&w_data);
    let out_cpu = x_cpu.matmul(&w_cpu).expect("matmul は形状適合");
    let loss_cpu = out_cpu.sum(None).expect("全軸縮約は失敗しない");
    let grads_cpu = cpu_tape
        .backward(&loss_cpu)
        .expect("w が追跡対象のため成功する");
    let dw_cpu = grads_cpu
        .get(&w_cpu)
        .expect("w は requires_grad=true の葉")
        .expect("w は loss に到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.var_no_grad(&x_data);
    let w_naive = naive_tape.var(&w_data);
    let out_naive = x_naive.matmul(&w_naive).expect("matmul は形状適合");
    let loss_naive = out_naive.sum(None).expect("全軸縮約は失敗しない");
    let grads_naive = naive_tape
        .backward(&loss_naive)
        .expect("w が追跡対象のため成功する");
    let dw_naive = grads_naive
        .get(&w_naive)
        .expect("w は requires_grad=true の葉")
        .expect("w は loss に到達する");

    assert_eq!(
        contiguous_bits(dw_cpu),
        contiguous_bits(dw_naive),
        "var_no_grad の w 勾配は CPU 本番 ops と naive 参照実装で bit 一致するはず"
    );

    let err_cpu = grads_cpu.get(&x_cpu).unwrap_err();
    let err_naive = grads_naive.get(&x_naive).unwrap_err();
    assert!(matches!(err_cpu, AutodiffError::GradientTrackingDisabled));
    assert!(matches!(err_naive, AutodiffError::GradientTrackingDisabled));
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---
//
// `unique_backend_parity.rs` と同じ理由で `Device::Metal` variant を
// 参照するテスト関数のみ `cfg(target_os = "macos")` でコンパイル自体を
// 限定する。実機実測は本エージェント実行環境に到達手段が無いため
// 未実施のまま（PR 本文に申し送り）。

fn detach_weight_grad_bits_on(device: Device, seed_offset: f32) -> Vec<u32> {
    let x_data = t(vec![1.0 + seed_offset, 2.0, 3.0, 4.0], &[2, 2]);
    let w_data = t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.var(&x_data);
    let w = tape.var(&w_data);
    let x_detached = x.detach().expect("detach は失敗しない");
    let product = x_detached.mul(&w).expect("同 shape の要素積は失敗しない");
    let loss = product.sum(None).expect("全軸縮約は失敗しない");
    let grads = tape.backward(&loss).expect("w が追跡対象のため成功する");
    let dw = grads
        .get(&w)
        .expect("w は requires_grad=true の葉")
        .expect("w は loss に到達する");
    contiguous_bits(dw)
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_detach_weight_grad_matches_cpu() {
    let metal_bits = detach_weight_grad_bits_on(Device::Metal, 0.0);
    let cpu_bits = detach_weight_grad_bits_on(Device::Cpu, 0.0);
    assert_eq!(
        metal_bits, cpu_bits,
        "detach の w 勾配は Metal と CPU で bit 一致するはず（算術を伴わない選択演算）"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_detach_weight_grad_matches_cpu() {
    let cuda_bits = detach_weight_grad_bits_on(Device::Cuda(0), 0.0);
    let cpu_bits = detach_weight_grad_bits_on(Device::Cpu, 0.0);
    assert_eq!(
        cuda_bits, cpu_bits,
        "detach の w 勾配は CUDA と CPU で bit 一致するはず（算術を伴わない選択演算）"
    );
}
