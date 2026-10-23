//! 常駐入力（`ResidentTensor`）の推論・行列積の受け入れ基準対応テスト
//! （イシュー #3250 P1〜P4・`docs/compat-resident-train-step-decision.md` §4・§10）。
//!
//! 常駐版（`Sequential::predict_resident_input`・`Tape::matmul_resident`）の出力が
//! ホスト入力版（`predict_resident`・`Var::matmul`）と bit 一致すること、別 store・別デバイス・
//! rank 不正・次元不一致・ホスト経路へ落とさない `Unsupported` を型付きエラーで確かめる。
//! CPU は CI で実行し、CUDA・Metal は同じ内容を `#[ignore]`（実機）で実行する。
//! 公開の仕上げ（正ガード・doctest）は #3254 で扱うため本ファイルでは足さない。

use fandhe_ai::{
    AutodiffError, BackendError, Device, ResidentTensor, Tape, Tensor, compat::Sequential, tape_for,
};

fn tensor(seed: u32, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let data = (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Tensor::new(data, shape).unwrap()
}

fn assert_bits_eq(a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape());
    let (a, b) = (a.contiguous(), b.contiguous());
    for (x, y) in a.as_slice().unwrap().iter().zip(b.as_slice().unwrap()) {
        assert_eq!(x.to_bits(), y.to_bits(), "shape {:?}", a.shape());
    }
}

fn model() -> Sequential {
    Sequential::new()
        .add_linear(6, 16, 11)
        .unwrap()
        .add_relu()
        .add_linear(16, 8, 12)
        .unwrap()
        .add_relu()
        .add_linear(8, 3, 13)
        .unwrap()
}

fn predict_matches_host(device: Device) {
    let tape = tape_for(device).unwrap();
    let m = model();
    let store = m.init_device_param_store(&tape).unwrap();
    for batch in [1usize, 5, 64] {
        let x = tensor(batch as u32, &[batch, 6]);
        let expect = m.predict_resident(&store, &x).unwrap();
        let handle = tape.upload_resident(&store, &x).unwrap();
        assert_eq!(handle.shape(), &[batch, 6]);
        assert_eq!(handle.device(), device);
        let got = m.predict_resident_input(&store, &handle).unwrap();
        assert_bits_eq(&expect, &got);
        // 同じハンドルを繰り返し使える（借用のみで消費・解放しない）。
        let again = m.predict_resident_input(&store, &handle).unwrap();
        assert_bits_eq(&expect, &again);
    }
}

fn matmul_matches_host(device: Device) {
    let tape = tape_for(device).unwrap();
    let store = Sequential::new().init_device_param_store(&tape).unwrap();
    // 8 の倍数の形状は全デバイス共通（Metal の `gemm_device` は 8 の倍数のみ対応）。
    // 倍数でない形状は CPU だけで確かめる（Metal は `Unsupported`）。
    let mut shapes = vec![(8, 8, 8), (16, 24, 32), (64, 784, 256), (256, 256, 256)];
    if device == Device::Cpu {
        shapes.extend([(1, 1, 1), (1, 5, 3), (7, 13, 5)]);
    }
    for &(m, k, n) in &shapes {
        let a = tensor(1, &[m, k]);
        let b = tensor(2, &[k, n]);
        let expect = tape.var(&a).matmul(&tape.var(&b)).unwrap().to_tensor();
        let ah = tape.upload_resident(&store, &a).unwrap();
        let bh = tape.upload_resident(&store, &b).unwrap();
        assert_bits_eq(&expect, &tape.matmul_resident(&ah, &bh).unwrap());
    }
    // Metal は 8 の倍数でない形状を黙って別経路で計算せず `Unsupported` にする。
    if matches!(device, Device::Metal) {
        let a = tape.upload_resident(&store, &tensor(1, &[7, 13])).unwrap();
        let b = tape.upload_resident(&store, &tensor(2, &[13, 5])).unwrap();
        let e = backend_err(tape.matmul_resident(&a, &b).unwrap_err());
        assert!(matches!(e, BackendError::Unsupported(_)), "{e:?}");
    }
    // 同じハンドルを a と b の両方に渡せる。
    let sq = tensor(3, &[16, 16]);
    let h = tape.upload_resident(&store, &sq).unwrap();
    let expect = tape.var(&sq).matmul(&tape.var(&sq)).unwrap().to_tensor();
    assert_bits_eq(&expect, &tape.matmul_resident(&h, &h).unwrap());
}

fn backend_err(e: AutodiffError) -> BackendError {
    match e {
        AutodiffError::Backend(b) => b,
        other => panic!("expected backend error, got {other:?}"),
    }
}

fn errors(device: Device) {
    let tape = tape_for(device).unwrap();
    let m = model();
    let store = m.init_device_param_store(&tape).unwrap();
    let other_store = m.init_device_param_store(&tape).unwrap();

    // 別 store のハンドル。
    let foreign: ResidentTensor = tape
        .upload_resident(&other_store, &tensor(1, &[2, 6]))
        .unwrap();
    let e = backend_err(m.predict_resident_input(&store, &foreign).unwrap_err());
    assert!(matches!(e, BackendError::InvalidArgument(_)), "{e:?}");

    // rank 不正・列数不一致。
    let rank3 = tape
        .upload_resident(&store, &tensor(2, &[2, 3, 6]))
        .unwrap();
    let e = backend_err(m.predict_resident_input(&store, &rank3).unwrap_err());
    assert!(matches!(e, BackendError::InvalidArgument(_)), "{e:?}");
    let wrong_k = tape.upload_resident(&store, &tensor(3, &[2, 5])).unwrap();
    let e = backend_err(m.predict_resident_input(&store, &wrong_k).unwrap_err());
    assert!(matches!(e, BackendError::ShapeMismatch(_)), "{e:?}");

    // チェーンを組めないモデル（Sigmoid 混在）はホスト経路へ落とさず Unsupported。
    let sig = Sequential::new()
        .add_linear(6, 4, 21)
        .unwrap()
        .add_sigmoid()
        .add_linear(4, 2, 22)
        .unwrap();
    let sig_store = sig.init_device_param_store(&tape).unwrap();
    let h = tape
        .upload_resident(&sig_store, &tensor(4, &[2, 6]))
        .unwrap();
    let e = backend_err(sig.predict_resident_input(&sig_store, &h).unwrap_err());
    assert!(matches!(e, BackendError::Unsupported(_)), "{e:?}");

    // 行列積: rank 不正・内側の次元不一致。
    let a = tape.upload_resident(&store, &tensor(5, &[2, 3])).unwrap();
    let b = tape.upload_resident(&store, &tensor(6, &[4, 5])).unwrap();
    let e = backend_err(tape.matmul_resident(&a, &b).unwrap_err());
    assert!(matches!(e, BackendError::ShapeMismatch(_)), "{e:?}");
    let v = tape.upload_resident(&store, &tensor(7, &[3])).unwrap();
    let e = backend_err(tape.matmul_resident(&v, &b).unwrap_err());
    assert!(matches!(e, BackendError::InvalidArgument(_)), "{e:?}");
}

#[test]
fn cpu_predict_resident_input_matches_host_bitwise() {
    predict_matches_host(Device::Cpu);
}

#[test]
fn cpu_matmul_resident_matches_var_matmul_bitwise() {
    matmul_matches_host(Device::Cpu);
}

#[test]
fn cpu_resident_input_errors_are_typed() {
    errors(Device::Cpu);
}

#[test]
fn cpu_predict_resident_input_leaves_store_usable() {
    // 拒否されたあとも store は壊れず、通常の推論が続けられる。
    let tape = tape_for(Device::Cpu).unwrap();
    let m = model();
    let store = m.init_device_param_store(&tape).unwrap();
    let bad = tape.upload_resident(&store, &tensor(9, &[2, 5])).unwrap();
    assert!(m.predict_resident_input(&store, &bad).is_err());
    let x = tensor(10, &[2, 6]);
    let h = tape.upload_resident(&store, &x).unwrap();
    assert_bits_eq(
        &m.predict_resident(&store, &x).unwrap(),
        &m.predict_resident_input(&store, &h).unwrap(),
    );
    let _: &Tape = &tape;
}

#[test]
#[ignore = "CUDA 実機（GB10）で実行"]
fn cuda_predict_resident_input_matches_host_bitwise() {
    predict_matches_host(Device::Cuda(0));
}

#[test]
#[ignore = "CUDA 実機（GB10）で実行"]
fn cuda_matmul_resident_matches_var_matmul_bitwise() {
    matmul_matches_host(Device::Cuda(0));
}

#[test]
#[ignore = "CUDA 実機（GB10）で実行"]
fn cuda_resident_input_errors_are_typed() {
    errors(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機で実行"]
fn metal_predict_resident_input_matches_host_bitwise() {
    predict_matches_host(Device::Metal);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機で実行"]
fn metal_matmul_resident_matches_var_matmul_bitwise() {
    matmul_matches_host(Device::Metal);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機で実行"]
fn metal_resident_input_errors_are_typed() {
    errors(Device::Metal);
}
