//! TASK-1.9d（#47）: CUDA 実機での `BackendOps` 経由数値一致検証。
//!
//! `cpu_cuda_parity.rs`（TASK-2.2b・#54）は `CudaGemm::run_naive_f32` を
//! 直接呼び出す形で CPU-CUDA ペアの数値一致を検証しているが、本ファイルは
//! 抽象層 `fandhe_ai_tensor_core::backend_ops::CudaBackendOps`（TASK-1.9c・#46）を
//! 経由した場合にも同じ複合判定（REQ-2）が成立することを固定する
//! （抽象層自体はカーネル呼び出し前後で shape 検証・contiguous 化・
//! エラー変換のみを行うため理論上は同値だが、回帰保護として独立に検証
//! する）。判定式・許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の
//! 参照とする（`.claude/rules/coding-rust.md`）。
//!
//! `cpu_cuda_parity.rs` と意図的に異なる形状を選び、K=4096 ストレスケース
//! （同ファイルが既に担う）を重複させない。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice, CudaError, CudaGemm};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// `CudaBackendOps::gemm` を CPU `BackendOps::gemm` と複合判定で突き合わせる。
fn assert_backend_ops_gemm_parity(seed_a: u64, seed_b: u64, m: usize, n: usize, k: usize) {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let a_data = Xorshift64Star::new(seed_a).fill_vec(m * k);
    let b_data = Xorshift64Star::new(seed_b).fill_vec(k * n);
    let a = Tensor::new(a_data, &[m, k]).expect("valid tensor");
    let b = Tensor::new(b_data, &[k, n]).expect("valid tensor");

    let cpu_result = cpu.gemm(&a, &b).expect("cpu gemm always succeeds");
    let cuda_result = cuda
        .gemm(&a, &b)
        .expect("CudaBackendOps::gemm must succeed on CUDA-equipped test runner");

    assert_eq!(cuda_result.shape(), cpu_result.shape());
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("BackendOps cpu-cuda gemm parity m={m} n={n} k={k}"),
        cuda_result.as_slice().expect("contiguous"),
        cpu_result.as_slice().expect("contiguous"),
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA 不在なら
/// `CudaBackendOps::gemm` の型付きエラー（`BackendError::CudaUnavailable`）
/// を確認して早期 return する（`cpu_cuda_parity.rs::naive_f32_parity_smoke_env_adaptive`
/// と同じ分岐パターン）。実機なら小型 gemm を BackendOps 経由で実行し
/// CPU 結果と一致することまで確認する。
#[test]
fn backend_ops_gemm_parity_smoke_env_adaptive() {
    let cuda = CudaBackendOps::new(0);
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

    match cuda.gemm(&a, &b) {
        Ok(_) => {
            // 実機（CUDA + NVRTC 搭載 CI ランナー）: 形状網羅ケースまで実行する。
            assert_backend_ops_gemm_parity(301, 302, 48, 40, 56);
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            // 非搭載環境（driver 不在・NVRTC 不在いずれも `CudaBackendOps::gemm`
            // 内部で `CudaUnavailable` へ変換される。`backend-cuda/src/ops.rs`
            // 参照）での期待経路（panic しない）。
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::gemm: {other}"),
    }
}

// elementwise・reduction の `Unsupported` 契約は driver 呼び出し前に確定する
// （`backend-cuda/src/ops.rs`。`device_handle()` を経由しない）ため実機を
// 要さず、`backend_ops_dispatch.rs::cuda_elementwise_and_reduction_return_unsupported_not_panic`
// が通常 CI で既に検証済みである。本ファイルでは重複させず、`#[ignore]` は
// 実機（driver 経由）のみが検証できる gemm 数値一致に限定する。

/// 実機必須の形状網羅（受け入れ条件の本体）。`cpu_cuda_parity.rs` の
/// 形状（128^3・512^3・64x96x128・1x1x1・17x23x19・33x31x65）とは意図的に
/// 異なる組を選び、直接の重複を避ける。K=4096 ストレスは
/// `cpu_cuda_parity.rs::naive_f32_k4096_stress_poc_v2_5` が既に担うためここでは
/// 含めない。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn backend_ops_gemm_matches_cpu_across_shapes() {
    let cases: &[(u64, u64, usize, usize, usize)] = &[
        (401, 402, 96, 96, 96),
        (403, 404, 256, 256, 256),
        (405, 406, 48, 80, 112),
        (407, 408, 1, 1, 1),
        (409, 410, 19, 29, 23),
        (411, 412, 65, 33, 31),
    ];
    for &(seed_a, seed_b, m, n, k) in cases {
        assert_backend_ops_gemm_parity(seed_a, seed_b, m, n, k);
    }
}

/// `CudaBackendOps` を介さず直接 `CudaDevice`／`CudaGemm` を初期化する
/// 経路も non-panicking であることを回帰させる（`device_handle` の
/// panic 回避ゲートが `BackendOps` 実装のみでなく、その内部委譲先
/// （`crate::device::CudaDevice::new`）でも保たれていることの確認。
/// `cpu_cuda_parity.rs::naive_f32_parity_smoke_env_adaptive` と同じ
/// 分岐パターンを踏襲）。
#[test]
fn cuda_device_initialization_does_not_panic() {
    match CudaDevice::new(0) {
        Ok(device) => {
            // NVRTC 未搭載環境（driver のみ）では `CudaGemm::new` が
            // `NvrtcUnavailable` を返すため、ここでは初期化成否のみ確認する。
            match CudaGemm::new(&device) {
                Ok(_) | Err(CudaError::NvrtcUnavailable { .. }) => {}
                Err(other) => panic!("unexpected CudaError variant from CudaGemm::new: {other}"),
            }
        }
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty());
        }
        Err(CudaError::Driver(_)) => {}
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    }
}

/// `ops_for`（`fandhe_ai_tensor_core::backend_ops`）を介したディスパッチでも同じ
/// 環境適応スモークが成立することを固定する（`Device::Cuda(0)` 選択の
/// 回帰保護）。
#[test]
fn ops_for_selects_cuda_backend_and_dispatch_does_not_panic() {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let ops: Vec<&dyn BackendOps> = vec![&cpu, &cuda];

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

    let selected =
        fandhe_ai_tensor_core::ops_for(&ops, Device::Cuda(0)).expect("cuda ops registered");
    match selected.gemm(&a, &b) {
        Ok(_) | Err(BackendError::CudaUnavailable(_)) => {}
        Err(other) => panic!("unexpected error variant for ops_for-dispatched gemm: {other}"),
    }
}
