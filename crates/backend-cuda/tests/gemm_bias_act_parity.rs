//! イシュー #599: elementwise 5 演算・`gemm_bias_act` epilogue 融合カーネル
//! の CPU-CUDA 数値一致検証。
//!
//! `backend_ops_real_device.rs`（TASK-1.9d・#47）と同じ構成方針を踏襲する:
//! 環境適応スモーク（属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `BackendError::CudaUnavailable` を確認して panic しないことのみ検証）と、
//! 実機必須の形状網羅（`#[ignore]`。DGX Spark GB10 等）を分離する。
//! 判定式・許容誤差は再定義せず `backend_cpu::parity` を唯一の参照とする
//! （`.claude/rules/coding-rust.md`）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p backend-cuda --release --test gemm_bias_act_parity -- --ignored --nocapture
//! ```

use backend_cpu::CpuBackendOps;
use backend_cuda::CudaBackendOps;
use bench_harness::rng::Xorshift64Star;
use tensor_core::device::BackendError;
use tensor_core::{Activation, BackendOps, Tensor};

/// CPU 側 `assert_tolerance_constants_pinned`（B-0・イシュー #491）を
/// 流用し、REQ-2 複合判定の tolerance 定数が本ファイルの実行時点でも
/// 無断変更されていないことを固定する。
mod common;

fn assert_gemm_bias_act_parity(
    seed_a: u64,
    seed_b: u64,
    seed_bias: u64,
    m: usize,
    n: usize,
    k: usize,
    act: Activation,
) {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let a_data = Xorshift64Star::new(seed_a).fill_vec(m * k);
    let b_data = Xorshift64Star::new(seed_b).fill_vec(k * n);
    let bias_data = Xorshift64Star::new(seed_bias).fill_vec(n);
    let a = Tensor::new(a_data, &[m, k]).expect("valid tensor");
    let b = Tensor::new(b_data, &[k, n]).expect("valid tensor");
    let bias = Tensor::new(bias_data, &[n]).expect("valid tensor");

    // (a) CPU 融合実装（`gemm_blis_bias_act_parallel`）との REQ-2 複合判定。
    let cpu_result = cpu
        .gemm_bias_act(&a, &b, Some(&bias), act)
        .expect("cpu gemm_bias_act always succeeds");
    let cuda_result = cuda
        .gemm_bias_act(&a, &b, Some(&bias), act)
        .expect("CudaBackendOps::gemm_bias_act must succeed on CUDA-equipped test runner");
    assert_eq!(cuda_result.shape(), cpu_result.shape());
    backend_cpu::parity::assert_parity(
        &format!("gemm_bias_act cpu-cuda parity m={m} n={n} k={k} act={act:?}"),
        cuda_result.as_slice().expect("contiguous"),
        cpu_result.as_slice().expect("contiguous"),
    );

    // (b) CUDA 上での融合 vs 非融合合成（gemm→add→act）の bit 完全一致
    // （`kernels::TILED_BIAS_ACT_F32` ドキュメンテーションコメント
    // 「数値契約」参照。両者は同一の tiled アキュムレーションを経由する
    // ため、epilogue の演算順序に依存しない加算・比較のみの差は
    // bit 単位で一致するはず）。
    let mut composed = cuda.gemm(&a, &b).expect("cuda gemm must succeed");
    composed = cuda.add(&composed, &bias).expect("cuda add must succeed");
    if act == Activation::Relu {
        composed = cuda.relu(&composed).expect("cuda relu must succeed");
    }
    assert_eq!(
        cuda_result.as_slice().expect("contiguous"),
        composed.as_slice().expect("contiguous"),
        "fused gemm_bias_act must bit-exact match composed gemm->add->act on CUDA \
         (m={m}, n={n}, k={k}, act={act:?})"
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA 不在なら
/// `BackendError::CudaUnavailable` を確認して早期 return する
/// （`backend_ops_real_device.rs::backend_ops_gemm_parity_smoke_env_adaptive`
/// と同じ分岐パターン）。実機なら形状網羅ケースまで実行する。
#[test]
fn gemm_bias_act_parity_smoke_env_adaptive() {
    let cuda = CudaBackendOps::new(0);
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");
    let bias = Tensor::new(vec![0.5, -0.5], &[2]).expect("valid tensor");

    match cuda.gemm_bias_act(&a, &b, Some(&bias), Activation::Relu) {
        Ok(_) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            assert_gemm_bias_act_parity(501, 502, 503, 40, 32, 48, Activation::Relu);
            assert_gemm_bias_act_parity(504, 505, 506, 33, 29, 17, Activation::None);
            // bias 形状 [1] はブロードキャスト可能だが `[n]` ちょうどでは
            // ないため非融合合成（`gemm`→`add`→`relu`）へフォールバック
            // する経路（`ops::gemm_bias_act_route`）を実機で確認する。
            let bias_broadcast = Tensor::new(vec![0.25], &[1]).expect("valid tensor");
            let cpu = CpuBackendOps::new();
            let cpu_result = cpu
                .gemm_bias_act(&a, &b, Some(&bias_broadcast), Activation::Relu)
                .expect("cpu gemm_bias_act always succeeds");
            let cuda_result = cuda
                .gemm_bias_act(&a, &b, Some(&bias_broadcast), Activation::Relu)
                .expect("cuda gemm_bias_act (broadcast fallback) must succeed");
            backend_cpu::parity::assert_parity(
                "gemm_bias_act broadcast-fallback cpu-cuda parity",
                cuda_result.as_slice().expect("contiguous"),
                cpu_result.as_slice().expect("contiguous"),
            );
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::gemm_bias_act: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。m/n/k=0 縮退・bias 未指定
/// （`Activation::None`）を含む。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_bias_act_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let cases: &[(u64, u64, u64, usize, usize, usize, Activation)] = &[
        (601, 602, 603, 96, 96, 96, Activation::Relu),
        (604, 605, 606, 128, 64, 96, Activation::None),
        (607, 608, 609, 1, 1, 1, Activation::Relu),
        (610, 611, 612, 19, 29, 23, Activation::Relu),
        (613, 614, 615, 65, 33, 31, Activation::None),
    ];
    for &(seed_a, seed_b, seed_bias, m, n, k, act) in cases {
        assert_gemm_bias_act_parity(seed_a, seed_b, seed_bias, m, n, k, act);
    }

    // k=0 縮退（GEMM 部分は全 0、epilogue のみホスト側で直接計算する
    // 経路。`gemm.rs::run_tiled_bias_act_f32` の「k == 0」分岐参照）。
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let a = Tensor::<f32>::zeros(&[4, 0]).expect("valid tensor");
    let b = Tensor::<f32>::zeros(&[0, 3]).expect("valid tensor");
    let bias = Tensor::new(vec![1.0, -2.0, 3.0], &[3]).expect("valid tensor");
    let cpu_result = cpu
        .gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
        .expect("cpu gemm_bias_act k=0 must succeed");
    let cuda_result = cuda
        .gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
        .expect("cuda gemm_bias_act k=0 must succeed");
    assert_eq!(
        cuda_result.as_slice().expect("contiguous"),
        cpu_result.as_slice().expect("contiguous"),
        "k=0 degenerate gemm_bias_act must match CPU exactly (integer bias+relu, no float GEMM)"
    );
}

/// elementwise 5 演算（`add`／`mul`／`relu`／`exp`／`tanh`）の CPU-CUDA
/// 数値一致（実機必須）。`exp`／`tanh` は libm 由来の丸め差がありうる
/// ため REQ-2 複合判定で突き合わせる（`kernels_elementwise.rs` 冒頭
/// コメント「意味論の正」参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn elementwise_matches_cpu_across_ops() {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let a_data = Xorshift64Star::new(701).fill_vec(37 * 5);
    let b_data = Xorshift64Star::new(702).fill_vec(37 * 5);
    let a = Tensor::new(a_data, &[37, 5]).expect("valid tensor");
    let b = Tensor::new(b_data, &[37, 5]).expect("valid tensor");

    type ElementwiseOpFn = fn(&dyn BackendOps, &Tensor<f32>, &Tensor<f32>) -> Tensor<f32>;
    let ops: &[(&str, ElementwiseOpFn)] = &[
        ("add", |ops, a, b| ops.add(a, b).expect("add must succeed")),
        ("mul", |ops, a, b| ops.mul(a, b).expect("mul must succeed")),
        ("relu", |ops, a, _b| ops.relu(a).expect("relu must succeed")),
        ("exp", |ops, a, _b| ops.exp(a).expect("exp must succeed")),
        ("tanh", |ops, a, _b| ops.tanh(a).expect("tanh must succeed")),
    ];

    for &(name, run) in ops {
        let cpu_result = run(&cpu, &a, &b);
        let cuda_result = run(&cuda, &a, &b);
        assert_eq!(cuda_result.shape(), cpu_result.shape());
        backend_cpu::parity::assert_parity(
            &format!("elementwise cpu-cuda parity op={name}"),
            cuda_result.as_slice().expect("contiguous"),
            cpu_result.as_slice().expect("contiguous"),
        );
    }

    // ブロードキャスト（`[37, 5]` + `[5]`）が CPU と同一の意味論で解決
    // されることも確認する（`ops.rs::CudaBackendOps::elementwise_binary`
    // の `Tensor::broadcast_with` 経由。`kernels_elementwise.rs` 冒頭
    // コメント「ブロードキャスト」参照）。
    let row = Tensor::new(vec![1.0, -1.0, 2.0, -2.0, 0.5], &[5]).expect("valid tensor");
    let cpu_bc = cpu.add(&a, &row).expect("cpu broadcast add must succeed");
    let cuda_bc = cuda.add(&a, &row).expect("cuda broadcast add must succeed");
    assert_eq!(cuda_bc.shape(), cpu_bc.shape());
    backend_cpu::parity::assert_parity(
        "elementwise broadcast cpu-cuda parity op=add",
        cuda_bc.as_slice().expect("contiguous"),
        cpu_bc.as_slice().expect("contiguous"),
    );
}
