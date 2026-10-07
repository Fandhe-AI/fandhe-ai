//! `compat::Sequential` の活性化 9 層（`add_selu`／`add_celu`／`add_softsign`／`add_hardsigmoid`／
//! `add_log_sigmoid`／`add_softmin`／`add_tanhshrink`／`add_threshold`／`add_rrelu`。イシュー #2679）の
//! CUDA／Metal parity テスト（`compat_sequential_activation_layers_backend_parity.rs` と同型）。
//!
//! 9 層は既存演算の合成または内部実装の `scalar_unary`／ホスト参照へのフォールバックで、専用の GPU
//! カーネルを持たない。対象デバイスの `forward` 出力と、層を挟んだ `Linear → 層 → Linear` の学習 1 ステップの
//! 勾配を CPU と比較する。判定は REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）で、
//! tolerance は変更せず新設もしない。RReLU は推論モード（固定傾き）で比較する（学習時の傾きは乱数のため）。
//!
//! CPU 側の正しさは `compat_sequential_activation_scalar_layers.rs` が Linux で担う。実機実測は未実施で、
//! GB10・M4 Max セッションへ申し送る
//! （`docs/perf/logs/compat-sequential-activation-scalar-layers-2679/README.md`）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tensor};

fn input(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(
        (0..n).map(|i| ((i as f32) * 0.37).sin() * 4.0).collect(),
        shape,
    )
    .expect("test fixture: shape とデータ長は一致させている")
}

const LAYERS: [&str; 9] = [
    "selu",
    "celu",
    "softsign",
    "hardsigmoid",
    "log_sigmoid",
    "softmin",
    "tanhshrink",
    "threshold",
    "rrelu",
];

/// 1 層だけのモデル（RReLU は推論モードへ切り替える）。
fn single(layer: &str) -> Sequential {
    let mut m = match layer {
        "selu" => Sequential::new().add_selu(),
        "celu" => Sequential::new().add_celu(1.0).unwrap(),
        "softsign" => Sequential::new().add_softsign(),
        "hardsigmoid" => Sequential::new().add_hardsigmoid(),
        "log_sigmoid" => Sequential::new().add_log_sigmoid(),
        "softmin" => Sequential::new().add_softmin(1),
        "tanhshrink" => Sequential::new().add_tanhshrink(),
        "threshold" => Sequential::new().add_threshold(0.5, -2.0),
        "rrelu" => Sequential::new().add_rrelu(0.125, 0.3).unwrap(),
        other => panic!("未知の層: {other}"),
    };
    m.eval();
    m
}

/// 層を `Linear(2→3) → 層 → Linear(3→1)` に挟んだ学習モデル（RReLU は推論モードへ切り替える）。
fn sandwich(layer: &str) -> Sequential {
    let base = Sequential::new().add_linear(2, 3, 11).unwrap();
    let mid = match layer {
        "selu" => base.add_selu(),
        "celu" => base.add_celu(1.0).unwrap(),
        "softsign" => base.add_softsign(),
        "hardsigmoid" => base.add_hardsigmoid(),
        "log_sigmoid" => base.add_log_sigmoid(),
        "softmin" => base.add_softmin(1),
        "tanhshrink" => base.add_tanhshrink(),
        "threshold" => base.add_threshold(-10.0, 0.0),
        "rrelu" => base.add_rrelu(0.125, 0.3).unwrap(),
        other => panic!("未知の層: {other}"),
    };
    let mut m = mid.add_linear(3, 1, 12).unwrap();
    m.eval();
    m
}

fn device_forward(model: &Sequential, x: &Tensor<f32>, device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(x);
    model.forward(&tape, &xv).unwrap().to_tensor()
}

/// 学習 1 ステップ分の勾配（層順の全パラメータ）。
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
    let x = input(&[3, 4]);
    for layer in LAYERS {
        let m = single(layer);
        let cpu = m.predict(&x).unwrap();
        let dev = device_forward(&m, &x, device);
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("{layer} forward: cpu vs {device:?}"),
            cpu.host_slice().as_ref(),
            dev.host_slice().as_ref(),
        );

        let model = sandwich(layer);
        let cpu = device_grads(&model, None);
        let dev = device_grads(&model, Some(device));
        assert_eq!(cpu.len(), dev.len(), "{layer}");
        for (i, (c, d)) in cpu.iter().zip(dev.iter()).enumerate() {
            fandhe_ai_backend_cpu::parity::assert_parity(
                &format!("{layer} model grad[{i}]: cpu vs {device:?}"),
                c.host_slice().as_ref(),
                d.host_slice().as_ref(),
            );
        }
    }
}

#[test]
fn cpu_reference_runs_without_device() {
    for layer in LAYERS {
        let m = single(layer);
        let y = m.predict(&input(&[3, 4])).unwrap();
        assert_eq!(y.shape(), &[3, 4], "{layer}");
        assert_eq!(device_grads(&sandwich(layer), None).len(), 4, "{layer}");
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #2679 の申し送り先（GB10 セッション）へ引き継ぐ"]
fn cuda_activation_scalar_layers_parity() {
    run_parity(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実行は #2679 の申し送り先（Mac セッション）へ引き継ぐ"]
fn metal_activation_scalar_layers_parity() {
    run_parity(Device::Metal);
}
