//! イシュー #1741: `BackendOps::sort`／`topk`（`torch.sort`／
//! `torch.topk` 相当）の CPU-Metal 数値一致検証（CUDA 側
//! `sort_topk_parity.rs` の Metal 対応版）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`unique_parity.rs` と同方針。`#![cfg(target_os = "macos")]` により
//! Linux CI ではコンパイル対象外になり、`#[ignore]` により通常の
//! `cargo test` からも除外される）。
//!
//! **契約は bit 同一**（`fandhe_ai_tensor_core::BackendOps::sort` doc の
//! 順序契約 1〜4 を正とする）: CPU 参照実装
//! （[`fandhe_ai_backend_cpu::CpuBackendOps`]）と Metal 実装の出力は
//! `values` の `to_bits()`・`index` が完全一致する契約。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test sort_topk_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_metal::MetalBackendOps;
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
    let metal = MetalBackendOps::new();
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
    let (metal_values, metal_index) =
        BackendOps::sort(&metal, &x, dim, descending).expect("metal sort must succeed");

    assert_eq!(
        dense_f32_bits(&metal_values),
        dense_f32_bits(&cpu_values),
        "sort values mismatch: shape={shape:?} dim={dim} descending={descending}"
    );
    assert_eq!(
        dense_i32(&metal_index),
        dense_i32(&cpu_index),
        "sort index mismatch: shape={shape:?} dim={dim} descending={descending}"
    );

    // run-to-run 決定性。
    let (metal_values2, metal_index2) =
        BackendOps::sort(&metal, &x, dim, descending).expect("metal sort run2");
    assert_eq!(
        dense_f32_bits(&metal_values2),
        dense_f32_bits(&metal_values)
    );
    assert_eq!(dense_i32(&metal_index2), dense_i32(&metal_index));
}

fn assert_topk_parity(seed: u64, shape: &[usize], dim: usize, k: usize, largest: bool) {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();
    let numel: usize = shape.iter().product();

    let data = Xorshift64Star::new(seed).fill_vec(numel);
    let x = Tensor::new(data, shape).expect("valid tensor");

    let (cpu_values, cpu_index) = BackendOps::topk(&cpu, &x, dim, k, largest).expect("cpu topk");
    let (metal_values, metal_index) =
        BackendOps::topk(&metal, &x, dim, k, largest).expect("metal topk must succeed");

    assert_eq!(
        dense_f32_bits(&metal_values),
        dense_f32_bits(&cpu_values),
        "topk values mismatch: shape={shape:?} dim={dim} k={k} largest={largest}"
    );
    assert_eq!(
        dense_i32(&metal_index),
        dense_i32(&cpu_index),
        "topk index mismatch: shape={shape:?} dim={dim} k={k} largest={largest}"
    );
}

/// サイズ網羅（2 のべき乗境界・非 2 のべき乗・大きめサイズ・rank 別
/// `dim`）を実機で確認する（Apple Silicon）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
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
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn sort_matches_cpu_for_non_contiguous_input() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();
    let data = Xorshift64Star::new(52_000).fill_vec(12);
    let x = Tensor::new(data, &[3, 4]).expect("valid tensor");
    let xt = x.transpose(0, 1).expect("transpose");

    let (cpu_values, cpu_index) = BackendOps::sort(&cpu, &xt, 1, false).expect("cpu sort");
    let (metal_values, metal_index) =
        BackendOps::sort(&metal, &xt, 1, false).expect("metal sort must succeed");
    assert_eq!(dense_f32_bits(&metal_values), dense_f32_bits(&cpu_values));
    assert_eq!(dense_i32(&metal_index), dense_i32(&cpu_index));
}

/// `topk` の `k` 網羅（境界: `k=1`・`k=dim_size/2`・`k=dim_size`）を
/// 実機で確認する（`#[ignore]`）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn topk_matches_cpu_across_k_and_largest() {
    let shape: &[usize] = &[1, 100];
    for &k in &[1usize, 50, 99, 100] {
        assert_topk_parity(53_000 + k as u64, shape, 1, k, true);
        assert_topk_parity(54_000 + k as u64, shape, 1, k, false);
    }
}

/// `topk(k=0)` はバックエンド固有の早期 return（GPU dispatch なし）
/// 経路を通る（`sort.rs::MetalSort::run_sort_f32` の `total_out == 0`
/// 分岐）ことを空出力形状で確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn topk_k_zero_returns_empty_on_metal() {
    let metal = MetalBackendOps::new();
    let x = Tensor::new(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]).expect("valid tensor");
    let (values, index) = BackendOps::topk(&metal, &x, 1, 0, true).expect("metal topk(k=0)");
    assert_eq!(values.shape(), &[1, 0]);
    assert_eq!(index.shape(), &[1, 0]);
}
