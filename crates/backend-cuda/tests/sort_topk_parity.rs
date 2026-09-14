//! イシュー #1741: `BackendOps::sort`／`topk`（`torch.sort`／
//! `torch.topk` 相当）の CPU-CUDA 数値一致検証。
//!
//! `unique_parity.rs` と同じ構成方針を踏襲する: 環境適応スモーク
//! （属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `BackendError::CudaUnavailable` を確認して panic しないことのみ
//! 検証）と、実機必須の形状網羅（`#[ignore]`。DGX Spark GB10 等）を
//! 分離する。
//!
//! **契約は bit 同一**（`values` は算術演算を含まない `input` の並べ
//! 替え・`index` は整数。`fandhe_ai_tensor_core::BackendOps::sort` doc
//! の順序契約 1〜4 を正とする）: CPU 参照実装
//! （[`fandhe_ai_backend_cpu::CpuBackendOps`]）と CUDA 実装の出力は
//! `values` の `to_bits()`・`index` が完全一致する契約。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test sort_topk_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn dense_f32_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn dense_i32(t: &Tensor<i32>) -> Vec<i32> {
    t.contiguous().as_slice().expect("contiguous").to_vec()
}

fn assert_sort_parity(seed: u64, shape: &[usize], dim: usize, descending: bool) {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let numel: usize = shape.iter().product();

    let mut data = Xorshift64Star::new(seed).fill_vec(numel);
    if numel >= 4 {
        data[0] = f32::NAN;
        data[1] = -f32::NAN;
        data[2] = 0.0;
        data[3] = -0.0;
    }
    let x = Tensor::new(data, shape).expect("valid tensor");

    let (cpu_values, cpu_index) = BackendOps::sort(&cpu, &x, dim, descending).expect("cpu sort");
    let (cuda_values, cuda_index) =
        BackendOps::sort(&cuda, &x, dim, descending).expect("cuda sort must succeed");

    assert_eq!(
        dense_f32_bits(&cuda_values),
        dense_f32_bits(&cpu_values),
        "sort values mismatch: shape={shape:?} dim={dim} descending={descending}"
    );
    assert_eq!(
        dense_i32(&cuda_index),
        dense_i32(&cpu_index),
        "sort index mismatch: shape={shape:?} dim={dim} descending={descending}"
    );

    // run-to-run 決定性。
    let (cuda_values2, cuda_index2) =
        BackendOps::sort(&cuda, &x, dim, descending).expect("cuda sort run2");
    assert_eq!(dense_f32_bits(&cuda_values2), dense_f32_bits(&cuda_values));
    assert_eq!(dense_i32(&cuda_index2), dense_i32(&cuda_index));
}

fn assert_topk_parity(seed: u64, shape: &[usize], dim: usize, k: usize, largest: bool) {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let numel: usize = shape.iter().product();

    let data = Xorshift64Star::new(seed).fill_vec(numel);
    let x = Tensor::new(data, shape).expect("valid tensor");

    let (cpu_values, cpu_index) = BackendOps::topk(&cpu, &x, dim, k, largest).expect("cpu topk");
    let (cuda_values, cuda_index) =
        BackendOps::topk(&cuda, &x, dim, k, largest).expect("cuda topk must succeed");

    assert_eq!(
        dense_f32_bits(&cuda_values),
        dense_f32_bits(&cpu_values),
        "topk values mismatch: shape={shape:?} dim={dim} k={k} largest={largest}"
    );
    assert_eq!(
        dense_i32(&cuda_index),
        dense_i32(&cpu_index),
        "topk index mismatch: shape={shape:?} dim={dim} k={k} largest={largest}"
    );
}

#[test]
fn sort_topk_parity_smoke_env_adaptive() {
    let cuda = CudaBackendOps::new(0);
    let x = Tensor::new(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]).expect("valid tensor");

    match BackendOps::sort(&cuda, &x, 1, false) {
        Ok(_) => {
            assert_sort_parity(40001, &[1, 4], 1, false);
            assert_sort_parity(40002, &[1, 4], 1, true);
            assert_sort_parity(40003, &[3, 4], 1, false);
            assert_sort_parity(40004, &[4, 3], 0, true);
            assert_sort_parity(40005, &[2, 3, 4], 1, false);
            assert_sort_parity(40006, &[1, 100], 1, false);

            assert_topk_parity(40011, &[1, 4], 1, 2, true);
            assert_topk_parity(40012, &[1, 4], 1, 2, false);
            assert_topk_parity(40013, &[3, 4], 1, 4, true); // k == dim_size（sort と同一のはず）
            assert_topk_parity(40014, &[1, 100], 1, 10, true);
        }
        Err(fandhe_ai_tensor_core::device::BackendError::CudaUnavailable(_)) => {
            // CUDA 非搭載環境（通常 CI）。panic せず終了する
            // （`unique_parity.rs` と同じ環境適応方針）。
        }
        Err(other) => panic!("unexpected error on CUDA-equipped runner: {other}"),
    }
}

/// サイズ網羅（2 のべき乗境界・非 2 のべき乗・大きめサイズ・rank 別
/// `dim`）を実機で確認する（DGX Spark GB10 等。`#[ignore]`）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn sort_matches_cpu_across_shapes() {
    let shapes: &[(&[usize], usize)] = &[
        (&[1, 1], 1),
        (&[1, 2], 1),
        (&[1, 3], 1),
        (&[1, 4], 1),
        (&[1, 7], 1),
        (&[1, 8], 1),
        (&[1, 255], 1),
        (&[1, 256], 1),
        (&[1, 257], 1),
        (&[1, 1025], 1),
        (&[2, 3, 4], 0),
        (&[2, 3, 4], 1),
        (&[2, 3, 4], 2),
        (&[5, 1], 0),
    ];
    for (i, &(shape, dim)) in shapes.iter().enumerate() {
        assert_sort_parity(50_000 + i as u64, shape, dim, false);
        assert_sort_parity(51_000 + i as u64, shape, dim, true);
    }
}

/// 非 contiguous（transpose 済み view）入力の parity を実機で確認する
/// （`#[ignore]`）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn sort_matches_cpu_for_non_contiguous_input() {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let data = Xorshift64Star::new(52_000).fill_vec(12);
    let x = Tensor::new(data, &[3, 4]).expect("valid tensor");
    let xt = x.transpose(0, 1).expect("transpose");

    let (cpu_values, cpu_index) = BackendOps::sort(&cpu, &xt, 1, false).expect("cpu sort");
    let (cuda_values, cuda_index) =
        BackendOps::sort(&cuda, &xt, 1, false).expect("cuda sort must succeed");
    assert_eq!(dense_f32_bits(&cuda_values), dense_f32_bits(&cpu_values));
    assert_eq!(dense_i32(&cuda_index), dense_i32(&cpu_index));
}

/// `topk` の `k` 網羅（境界: `k=1`・`k=dim_size/2`・`k=dim_size`）を
/// 実機で確認する（`#[ignore]`）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn topk_matches_cpu_across_k_and_largest() {
    let shape: &[usize] = &[1, 100];
    for &k in &[1usize, 50, 99, 100] {
        assert_topk_parity(53_000 + k as u64, shape, 1, k, true);
        assert_topk_parity(54_000 + k as u64, shape, 1, k, false);
    }
}
