//! `compat::Sequential::add_mish`／`add_hardtanh`／`add_relu6`／`add_glu`／`add_prelu`
//! （イシュー #2529）の CUDA／Metal parity テスト（`compat_sequential_dropout_embedding_bag_backend_parity.rs`
//! と同型）。
//!
//! 5 層は既存演算の合成で新規カーネルはない。対象デバイスの `forward` 出力を CPU `predict` と比較する。
//! 判定方式は `activation_ops_backend_parity.rs` と同じ割り当てで、tolerance は変更せず新設もしない:
//! - bit 完全一致: `hardtanh`／`relu6`／`prelu`（forward）
//! - REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）: `mish`／`glu`（forward）と、
//!   `Linear → PRelu → Linear` の学習 1 ステップの勾配（`weight` 勾配は縮約を経由するため）
//!
//! CPU 側の正しさは `compat_sequential_activation_layers.rs` が Linux で担う。実機実測は未実施で、
//! GB10・M4 Max セッションへ申し送る
//! （`docs/perf/logs/compat-sequential-activation-layers-2529/README.md`）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tensor};

fn dense(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn input(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(
        (0..n).map(|i| ((i as f32) * 0.37).sin() * 4.0).collect(),
        shape,
    )
    .expect("test fixture: shape とデータ長は一致させている")
}

fn device_forward(model: &Sequential, x: &Tensor<f32>, device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(x);
    model.forward(&tape, &xv).unwrap().to_tensor()
}

/// 学習 1 ステップ分の勾配（`Linear → PRelu → Linear` の MSE。層順の全パラメータ）。
fn device_grads(model: &Sequential, device: Option<Device>) -> Vec<Tensor<f32>> {
    let tape = match device {
        Some(d) => fandhe_ai::tape_for(d).expect("実機必須"),
        None => fandhe_ai::tape(),
    };
    let bound = model.bind(&tape);
    let xv = tape.var(&input(&[5, 2]));
    let tv = tape.var(&input(&[5, 1]));
    let loss = bound.forward(&tape, &xv).unwrap().mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    bound
        .trainable_grads(&grads)
        .unwrap()
        .into_iter()
        .cloned()
        .collect()
}

fn run_parity(device: Device) {
    // bit 完全一致の 3 層。
    let x = input(&[3, 4]);
    for (name, m) in [
        (
            "hardtanh",
            Sequential::new().add_hardtanh(-1.0, 2.0).unwrap(),
        ),
        ("relu6", Sequential::new().add_relu6()),
        ("prelu", Sequential::new().add_prelu(4, 0.2).unwrap()),
    ] {
        let cpu = m.predict(&x).unwrap();
        let dev = device_forward(&m, &x, device);
        assert_eq!(dense(&dev), dense(&cpu), "{name}");
    }

    // REQ-2 統一複合判定の 2 層。
    for (name, m) in [
        ("mish", Sequential::new().add_mish()),
        ("glu", Sequential::new().add_glu(1)),
    ] {
        let cpu = m.predict(&x).unwrap();
        let dev = device_forward(&m, &x, device);
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("{name} forward: cpu vs {device:?}"),
            cpu.host_slice().as_ref(),
            dev.host_slice().as_ref(),
        );
    }

    // PRelu を含むモデルの学習 1 ステップの勾配（weight 勾配は縮約を含むため REQ-2 判定）。
    let model = Sequential::new()
        .add_linear(2, 3, 11)
        .unwrap()
        .add_prelu(3, 0.25)
        .unwrap()
        .add_linear(3, 1, 12)
        .unwrap();
    let cpu = device_grads(&model, None);
    let dev = device_grads(&model, Some(device));
    assert_eq!(cpu.len(), dev.len());
    for (i, (c, d)) in cpu.iter().zip(dev.iter()).enumerate() {
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("prelu model grad[{i}]: cpu vs {device:?}"),
            c.host_slice().as_ref(),
            d.host_slice().as_ref(),
        );
    }
}

#[test]
fn cpu_reference_runs_without_device() {
    assert_eq!(
        Sequential::new()
            .add_mish()
            .add_glu(1)
            .predict(&input(&[3, 4]))
            .unwrap()
            .shape(),
        &[3, 2]
    );
    let model = Sequential::new()
        .add_linear(2, 3, 11)
        .unwrap()
        .add_prelu(3, 0.25)
        .unwrap()
        .add_linear(3, 1, 12)
        .unwrap();
    assert_eq!(device_grads(&model, None).len(), 5);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #2529 の申し送り先（GB10 セッション）へ引き継ぐ"]
fn cuda_activation_layers_parity() {
    run_parity(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実行は #2529 の申し送り先（Mac セッション）へ引き継ぐ"]
fn metal_activation_layers_parity() {
    run_parity(Device::Metal);
}
