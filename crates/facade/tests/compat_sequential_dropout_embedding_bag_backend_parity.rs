//! `compat::Sequential::add_dropout2d`／`add_alpha_dropout`／`add_embedding_bag`（イシュー #2528）の
//! CUDA／Metal parity テスト（`compat_sequential_adaptive_max_global_pool_backend_parity.rs`
//! と同型）。
//!
//! 3 層は既存 `Op::Dropout`・`Var::embedding`・縮約の合成で、GPU 側はホストフォールバック経由。
//! 対象デバイスと CPU を **bit 完全一致**で比較する（tolerance は変更しない）。dropout 系は
//! 各呼び出し直前に `manual_seed` を打ち直す。CPU 側の正しさは
//! `compat_sequential_dropout_embedding_bag.rs` が Linux で担う。実機実測は未実施で、GB10・
//! M4 Max セッションへ申し送る
//! （`docs/perf/logs/compat-sequential-dropout-embedding-bag-2528/README.md`）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, EmbeddingBagMode, Tensor};

fn dense(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn feature_input() -> Tensor<f32> {
    let shape = [2, 4, 5, 3];
    let n: usize = shape.iter().product();
    Tensor::new(
        (0..n).map(|i| ((i as f32) * 0.37).sin() + 1.0).collect(),
        &shape,
    )
    .expect("test fixture: shape とデータ長は一致させている")
}

fn id_input() -> Tensor<f32> {
    Tensor::new((0..12).map(|i| (i % 6) as f32).collect(), &[3, 4])
        .expect("test fixture: shape とデータ長は一致させている")
}

fn dropout_model() -> Sequential {
    Sequential::new()
        .add_dropout2d(0.5)
        .unwrap()
        .add_alpha_dropout(0.25)
        .unwrap()
}

fn bag_model(mode: EmbeddingBagMode) -> Sequential {
    Sequential::new()
        .add_embedding_bag(6, 4, mode, Some(0), 21)
        .unwrap()
}

fn device_forward(model: &Sequential, x: &Tensor<f32>, device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(x);
    model.forward(&tape, &xv).unwrap().to_tensor()
}

fn run_bit_exact(device: Device) {
    let m = dropout_model();
    let x = feature_input();
    fandhe_ai::manual_seed(77);
    let cpu = m.predict(&x).unwrap();
    fandhe_ai::manual_seed(77);
    let dev = device_forward(&m, &x, device);
    assert_eq!(dense(&dev), dense(&cpu), "dropout");

    for mode in [
        EmbeddingBagMode::Sum,
        EmbeddingBagMode::Mean,
        EmbeddingBagMode::Max,
    ] {
        let m = bag_model(mode);
        let ids = id_input();
        let cpu = m.predict(&ids).unwrap();
        let dev = device_forward(&m, &ids, device);
        assert_eq!(dense(&dev), dense(&cpu), "{mode:?}");
    }
}

#[test]
fn cpu_reference_runs_without_device() {
    let _ = fandhe_ai::manual_seed;
    assert_eq!(
        dropout_model().predict(&feature_input()).unwrap().shape(),
        &[2, 4, 5, 3]
    );
    assert_eq!(
        bag_model(EmbeddingBagMode::Mean)
            .predict(&id_input())
            .unwrap()
            .shape(),
        &[3, 4]
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #2528 の申し送り先（GB10 セッション）へ引き継ぐ"]
fn cuda_dropout_embedding_bag_bit_exact() {
    run_bit_exact(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実行は #2528 の申し送り先（Mac セッション）へ引き継ぐ"]
fn metal_dropout_embedding_bag_bit_exact() {
    run_bit_exact(Device::Metal);
}
