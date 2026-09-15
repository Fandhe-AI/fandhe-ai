//! `pooling.rs::CudaPooling` の実機（DGX Spark GB10 等）検証（イシュー
//! #1729）。`poison_recovery_real_device_tests.rs`（`context_cache.rs`
//! から `#[path]` 登録）と同じ配置理由: `CudaPooling`・
//! `crate::pooling_model::*` はいずれも非公開（`mod pooling;`／
//! `mod pooling_model;` が private）のため `tests/`（統合テスト・
//! クレート外部扱い）からは到達できず、本ファイルを `pooling.rs`
//! 末尾から `#[cfg(test)] #[path] mod` として登録する。
//!
//! **`ops.rs::CudaBackendOps` への override 配線が無い**（`pooling.rs`
//! モジュール doc 参照）ため、本ファイルは `CudaPooling` を直接構築
//! して検証する（`fandhe_ai_backend_cuda::CudaBackendOps` 経由の
//! `tests/*_parity.rs` 群とは異なる経路）。
//!
//! オラクルは [`crate::pooling_model`]（GPU カーネルの逐語ホスト
//! モデル）で、3 演算とも **bit 完全一致**（MaxPool は値・索引とも）
//! を受入基準とする（設計 doc §5／§7・`.claude/rules/coding-rust.md`
//! 数値契約節）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`--ignored` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --lib \
//!     pooling::pooling_real_device_tests -- --ignored --nocapture
//! ```
//!
//! 属性なしテスト（[`pooling_smoke_environment_adaptive`]）は通常 CI
//! （CUDA 非搭載環境）でも実行され、`CudaDevice::new`／
//! `CudaPooling::new`（前者は `libcuda` の dlopen、後者は NVRTC
//! コンパイル＋`libnvrtc` の dlopen をそれぞれ要求するため、
//! `libcuda` のみ存在し `libnvrtc` が存在しない環境では前者が
//! `Ok` を返しつつ後者が `Err(NvrtcUnavailable)` を返す構成もあり得る。
//! 両呼び出しをそれぞれ個別に許容する）が `CudaError`（`Driver
//! Unavailable`／`NvrtcUnavailable` 等）を返すことのみを確認して
//! panic しないことを検証する（`im2col_col2im_parity.rs` 冒頭コメント
//! の「環境適応スモーク」と同じ方針）。

use super::CudaPooling;
use crate::device::CudaDevice;
use crate::pooling_model::{adaptive_avg_pool2d_model, avg_pool2d_model, max_pool2d_model};
use bench_harness::rng::Xorshift64Star;

/// `data` の全要素の `to_bits()` を FNV-1a 相当で fold した診断用
/// チェックサムを 1 行出力する（`tests/im2col_col2im_parity.rs::
/// print_fold_bits` と同一の FNV-1a fold。同じ理由で run-to-run
/// 決定性チェックの抽出対象を必ず 1 行以上作る）。
fn print_fold_bits(label: &str, data: &[f32]) {
    let mut acc: u64 = 0xcbf29ce484222325;
    for &v in data.iter() {
        let bits = v.to_bits() as u64;
        acc ^= bits;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    println!("{label}.fold_bits={acc:#018x}");
}

fn print_fold_bits_i32(label: &str, data: &[i32]) {
    let mut acc: u64 = 0xcbf29ce484222325;
    for &v in data.iter() {
        acc ^= v as u32 as u64;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    println!("{label}.fold_bits={acc:#018x}");
}

/// `f32` 値・NaN クラスの bit 完全一致検証（`assert_bits_eq` と同型の
/// 複製。理由は他 parity テストの同名関数 doc を参照）。
fn assert_bits_eq(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: 要素数が一致しない");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.is_nan() || e.is_nan() {
            assert!(
                a.is_nan() && e.is_nan(),
                "{label}: 要素 {i} が NaN クラス一致しない（actual={a}, expected={e}）"
            );
        } else {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}: 要素 {i} が bit 一致しない（actual={a:?}, expected={e:?}）"
            );
        }
    }
}

fn random_input(seed: u64, numel: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(numel)
}

/// CUDA 非搭載環境でも安全に終了することを確認する（通常 CI 実行対象。
/// モジュール doc 参照）。
#[test]
fn pooling_smoke_environment_adaptive() {
    match CudaDevice::new(0) {
        Err(_e) => {
            // CUDA driver/NVRTC 非搭載環境（GitHub ホステッド CI 等）。
            // panic せず安全に終了する契約のみを確認する。
            println!("pooling_smoke_environment_adaptive: CUDA unavailable, skipping");
        }
        Ok(device) => match CudaPooling::new(&device) {
            Err(_e) => {
                // `libcuda` は存在するが NVRTC コンパイル基盤
                // （`libnvrtc`）が欠けている環境（本エージェント実行
                // 環境で実際に観測済み）。同様に panic せず終了する。
                println!(
                    "pooling_smoke_environment_adaptive: CudaPooling::new unavailable, skipping"
                );
            }
            Ok(pooling) => {
                let input = vec![1.0f32, 2.0, 3.0, 4.0];
                let (values, indices, out_shape) = pooling
                    .run_max_pool2d_f32(&input, [1, 1, 2, 2], [2, 2], [2, 2], [0, 0], [1, 1])
                    .expect("max_pool2d smoke run must succeed on available device");
                assert_eq!(out_shape, [1, 1, 1, 1]);
                assert_eq!(values, vec![4.0]);
                assert_eq!(indices, vec![3]);
            }
        },
    }
}

/// 基本形 `k=2, s=2`（重なりなし）の値・索引 bit 完全一致。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn max_pool2d_basic_matches_model() {
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    let pooling = CudaPooling::new(&device).expect("CudaPooling::new");

    let in_shape = [2usize, 3, 8, 8];
    let numel: usize = in_shape.iter().product();
    let input = random_input(0xC0FFEE, numel);

    let (values, indices, out_shape) = pooling
        .run_max_pool2d_f32(&input, in_shape, [2, 2], [2, 2], [0, 0], [1, 1])
        .expect("max_pool2d basic");
    let (exp_values, exp_indices) =
        max_pool2d_model(&input, in_shape, out_shape, [2, 2], [2, 2], [0, 0], [1, 1]);
    assert_bits_eq("max_pool2d_basic.values", &values, &exp_values);
    assert_eq!(indices, exp_indices, "max_pool2d_basic: indices mismatch");
    print_fold_bits("max_pool2d_basic.values", &values);
    print_fold_bits_i32("max_pool2d_basic.indices", &indices);
}

/// 重なり窓（`stride < kernel`）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn max_pool2d_overlapping_window_matches_model() {
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    let pooling = CudaPooling::new(&device).expect("CudaPooling::new");

    let in_shape = [1usize, 2, 7, 7];
    let numel: usize = in_shape.iter().product();
    let input = random_input(0xBEEF, numel);

    let (values, indices, out_shape) = pooling
        .run_max_pool2d_f32(&input, in_shape, [3, 3], [1, 1], [1, 1], [1, 1])
        .expect("max_pool2d overlapping");
    let (exp_values, exp_indices) =
        max_pool2d_model(&input, in_shape, out_shape, [3, 3], [1, 1], [1, 1], [1, 1]);
    assert_bits_eq("max_pool2d_overlap.values", &values, &exp_values);
    assert_eq!(indices, exp_indices);
    print_fold_bits("max_pool2d_overlap.values", &values);
    print_fold_bits_i32("max_pool2d_overlap.indices", &indices);
}

/// `dilation=2`（境界: `in=2, k=2, s=1, p=1, d=2`。設計 doc §13）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn max_pool2d_dilation_boundary_matches_model() {
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    let pooling = CudaPooling::new(&device).expect("CudaPooling::new");

    let in_shape = [1usize, 1, 2, 2];
    let input = vec![1.0f32, 2.0, 3.0, 4.0];

    let (values, indices, out_shape) = pooling
        .run_max_pool2d_f32(&input, in_shape, [2, 2], [1, 1], [1, 1], [2, 2])
        .expect("max_pool2d dilation boundary");
    let (exp_values, exp_indices) =
        max_pool2d_model(&input, in_shape, out_shape, [2, 2], [1, 1], [1, 1], [2, 2]);
    assert_bits_eq("max_pool2d_dilation.values", &values, &exp_values);
    assert_eq!(indices, exp_indices);
    print_fold_bits("max_pool2d_dilation.values", &values);
    print_fold_bits_i32("max_pool2d_dilation.indices", &indices);
}

/// 1d 併合形状（`[N,C,1,L]`）でも 2d と同じ経路で bit 一致する
/// （設計 doc §2）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn max_pool2d_1d_merged_shape_matches_model() {
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    let pooling = CudaPooling::new(&device).expect("CudaPooling::new");

    let in_shape = [2usize, 3, 1, 17];
    let numel: usize = in_shape.iter().product();
    let input = random_input(0x1D_1D_1D, numel);

    let (values, indices, out_shape) = pooling
        .run_max_pool2d_f32(&input, in_shape, [1, 3], [1, 2], [0, 1], [1, 1])
        .expect("max_pool2d 1d merged");
    let (exp_values, exp_indices) =
        max_pool2d_model(&input, in_shape, out_shape, [1, 3], [1, 2], [0, 1], [1, 1]);
    assert_bits_eq("max_pool2d_1d.values", &values, &exp_values);
    assert_eq!(indices, exp_indices);
    print_fold_bits("max_pool2d_1d.values", &values);
    print_fold_bits_i32("max_pool2d_1d.indices", &indices);
}

/// 256 ブロック境界をまたぐ numel（設計 doc §13）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn max_pool2d_crosses_block_boundary_matches_model() {
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    let pooling = CudaPooling::new(&device).expect("CudaPooling::new");

    // out_shape numel = 1*1*16*17 = 272 > POOLING_BLOCK_DIM(256).
    let in_shape = [1usize, 1, 32, 34];
    let numel: usize = in_shape.iter().product();
    let input = random_input(0xABCD_EF01, numel);

    let (values, indices, out_shape) = pooling
        .run_max_pool2d_f32(&input, in_shape, [2, 2], [2, 2], [0, 0], [1, 1])
        .expect("max_pool2d block boundary");
    assert_eq!(out_shape[2] * out_shape[3], 16 * 17);
    let (exp_values, exp_indices) =
        max_pool2d_model(&input, in_shape, out_shape, [2, 2], [2, 2], [0, 0], [1, 1]);
    assert_bits_eq("max_pool2d_block_boundary.values", &values, &exp_values);
    assert_eq!(indices, exp_indices);
    print_fold_bits("max_pool2d_block_boundary.values", &values);
    print_fold_bits_i32("max_pool2d_block_boundary.indices", &indices);
}

/// NaN／±inf／−0.0 混在入力: 値は NaN クラス一致・NaN の索引は走査順
/// で最初に現れた位置（設計 doc §5）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn max_pool2d_nan_inf_negzero_matches_model() {
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    let pooling = CudaPooling::new(&device).expect("CudaPooling::new");

    let in_shape = [1usize, 1, 2, 2];
    let input = vec![f32::NAN, f32::NEG_INFINITY, -0.0f32, f32::INFINITY];

    let (values, indices, out_shape) = pooling
        .run_max_pool2d_f32(&input, in_shape, [2, 2], [2, 2], [0, 0], [1, 1])
        .expect("max_pool2d nan/inf");
    let (exp_values, exp_indices) =
        max_pool2d_model(&input, in_shape, out_shape, [2, 2], [2, 2], [0, 0], [1, 1]);
    assert_bits_eq("max_pool2d_nan_inf.values", &values, &exp_values);
    assert_eq!(indices, exp_indices);
}

/// 非 contiguous な走査（重なり窓・padding あり）の run-to-run bit 同一。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn max_pool2d_run_to_run_bit_identical() {
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    let pooling = CudaPooling::new(&device).expect("CudaPooling::new");

    let in_shape = [1usize, 2, 9, 9];
    let numel: usize = in_shape.iter().product();
    let input = random_input(0x5EED_5EED, numel);

    let (v1, i1, _) = pooling
        .run_max_pool2d_f32(&input, in_shape, [3, 3], [2, 2], [1, 1], [1, 1])
        .expect("run 1");
    let (v2, i2, _) = pooling
        .run_max_pool2d_f32(&input, in_shape, [3, 3], [2, 2], [1, 1], [1, 1])
        .expect("run 2");
    assert_bits_eq("max_pool2d_run_to_run", &v1, &v2);
    assert_eq!(i1, i2);
}

/// AvgPool `count_include_pad=true` の基本形。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn avg_pool2d_count_include_pad_true_matches_model() {
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    let pooling = CudaPooling::new(&device).expect("CudaPooling::new");

    let in_shape = [2usize, 3, 8, 8];
    let numel: usize = in_shape.iter().product();
    let input = random_input(0x7777_7777, numel);

    let (values, out_shape) = pooling
        .run_avg_pool2d_f32(&input, in_shape, [3, 3], [2, 2], [1, 1], true)
        .expect("avg_pool2d include_pad=true");
    let exp = avg_pool2d_model(&input, in_shape, out_shape, [3, 3], [2, 2], [1, 1], true);
    assert_bits_eq("avg_pool2d_include_pad_true.values", &values, &exp);
    print_fold_bits("avg_pool2d_include_pad_true.values", &values);
}

/// AvgPool `count_include_pad=false`。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn avg_pool2d_count_include_pad_false_matches_model() {
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    let pooling = CudaPooling::new(&device).expect("CudaPooling::new");

    let in_shape = [2usize, 3, 8, 8];
    let numel: usize = in_shape.iter().product();
    let input = random_input(0x8888_8888, numel);

    let (values, out_shape) = pooling
        .run_avg_pool2d_f32(&input, in_shape, [3, 3], [2, 2], [1, 1], false)
        .expect("avg_pool2d include_pad=false");
    let exp = avg_pool2d_model(&input, in_shape, out_shape, [3, 3], [2, 2], [1, 1], false);
    assert_bits_eq("avg_pool2d_include_pad_false.values", &values, &exp);
    print_fold_bits("avg_pool2d_include_pad_false.values", &values);
}

/// AdaptiveAvgPool `output_size=1`（global average）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn adaptive_avg_pool2d_global_average_matches_model() {
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    let pooling = CudaPooling::new(&device).expect("CudaPooling::new");

    let in_shape = [2usize, 4, 5, 7];
    let numel: usize = in_shape.iter().product();
    let input = random_input(0x9999_9999, numel);

    let (values, out_shape) = pooling
        .run_adaptive_avg_pool2d_f32(&input, in_shape, [1, 1])
        .expect("adaptive_avg_pool2d global");
    let exp = adaptive_avg_pool2d_model(&input, in_shape, out_shape);
    assert_bits_eq("adaptive_avg_pool2d_global.values", &values, &exp);
    print_fold_bits("adaptive_avg_pool2d_global.values", &values);
}

/// AdaptiveAvgPool の `out > in`（拡大側。窓が重なり合う）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn adaptive_avg_pool2d_upsampling_matches_model() {
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    let pooling = CudaPooling::new(&device).expect("CudaPooling::new");

    let in_shape = [1usize, 2, 3, 3];
    let numel: usize = in_shape.iter().product();
    let input = random_input(0xAAAA_AAAA, numel);

    let (values, out_shape) = pooling
        .run_adaptive_avg_pool2d_f32(&input, in_shape, [7, 5])
        .expect("adaptive_avg_pool2d upsample");
    let exp = adaptive_avg_pool2d_model(&input, in_shape, out_shape);
    assert_bits_eq("adaptive_avg_pool2d_upsample.values", &values, &exp);
    print_fold_bits("adaptive_avg_pool2d_upsample.values", &values);
}
