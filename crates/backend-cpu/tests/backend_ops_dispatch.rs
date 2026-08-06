//! TASK-1.9c（#46）の受け入れ条件「同一コードで 3 バックエンド（CPU／
//! CUDA／Metal）のカーネルが呼び分けられる」を直接検証する統合テスト。
//!
//! `backend_cpu::CpuBackendOps`・`backend_cuda::CudaBackendOps`・
//! （macOS のみ）`backend_metal::MetalBackendOps` を `Vec<&dyn BackendOps>`
//! へ束ね、`tensor_core::ops_for` が単一の trait オブジェクト経由で
//! `Device` ごとに正しい実装へディスパッチできることを、同一の呼び出し
//! コード（[`run_gemm_through_dispatch`]）で検証する。
//!
//! - CPU: 実際に `gemm` を実行し既知値と一致することを確認する（CI で
//!   常時実行。CPU は無条件で有効なため実カーネル検証が可能）。
//! - CUDA/Metal: `ops_for` が正しい実装を選択すること、およびデバイス
//!   不在環境（本 CI 環境は CUDA 非搭載・非 macOS）では型付きエラー
//!   （`CudaUnavailable` 等）を返すこと（panic しない）を検証する。
//!   実機での実行検証（GPU 実カーネル一致）は `#[ignore]` 分離し、3
//!   バックエンド網羅の本格的な統合テストは TASK-1.9d（#47）が担う
//!   （本テストは受け入れ条件検証に必要な最小限に留める）。

use backend_cpu::CpuBackendOps;
use backend_cuda::CudaBackendOps;
use tensor_core::device::{BackendError, Device};
use tensor_core::{BackendOps, Tensor, ops_for};

#[cfg(target_os = "macos")]
use backend_metal::MetalBackendOps;

/// 「単一の計算記述から各バックエンドのカーネルへディスパッチする」受け
/// 入れ条件そのものを表す関数。呼び出し元は `Device` を変えるだけで CPU
/// ／CUDA／Metal いずれのカーネルも実行できる（本関数自体はどの
/// `BackendOps` 実装が渡されるか一切関知しない）。
fn run_gemm_through_dispatch(
    ops: &[&dyn BackendOps],
    device: Device,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Result<Tensor<f32>, BackendError> {
    let selected = ops_for(ops, device)?;
    selected.gemm(a, b)
}

#[test]
fn same_code_dispatches_gemm_to_cpu_backend_and_matches_known_values() {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let ops: Vec<&dyn BackendOps> = vec![&cpu, &cuda];

    // A = [[1, 2], [3, 4]] (2x2)、B = [[5, 6], [7, 8]] (2x2)
    // A @ B = [[19, 22], [43, 50]]（既知値）。
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

    let result =
        run_gemm_through_dispatch(&ops, Device::Cpu, &a, &b).expect("cpu gemm always succeeds");

    assert_eq!(result.shape(), &[2, 2]);
    let out = result.as_slice().expect("contiguous result");
    let expected = [19.0f32, 22.0, 43.0, 50.0];
    for (got, want) in out.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-5,
            "got {got}, want {want}（複合判定 絶対誤差 1e-5 未満。REQ-2）"
        );
    }
}

#[test]
fn same_code_dispatches_gemm_to_cuda_backend_or_returns_typed_error() {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let ops: Vec<&dyn BackendOps> = vec![&cpu, &cuda];

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

    // 同一の `run_gemm_through_dispatch` 呼び出しコードのまま `Device` を
    // `Cuda(0)` に変えるだけで CUDA 実装へディスパッチされる（受け入れ
    // 条件の核心）。CUDA ドライバ非搭載環境（本 CI）では panic せず
    // `BackendError::CudaUnavailable` を返す（fail-safe）。GPU 実機での
    // 数値一致検証は TASK-1.9d（#47）・実機 `#[ignore]` テストの担当。
    match run_gemm_through_dispatch(&ops, Device::Cuda(0), &a, &b) {
        Ok(_) => {
            // 実機（CUDA 搭載 CI ランナー）では実カーネルが成功しうる。
        }
        Err(BackendError::CudaUnavailable(_)) => {
            // 非搭載環境での期待経路（panic しない）。
        }
        Err(other) => panic!("unexpected error variant for CUDA gemm dispatch: {other}"),
    }
}

#[test]
fn ops_for_returns_device_unavailable_when_no_matching_backend_registered() {
    let cpu = CpuBackendOps::new();
    let ops: Vec<&dyn BackendOps> = vec![&cpu];

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

    // CUDA 実装が `ops` に登録されていない場合は `select_from`（1.9a）と
    // 同じ意味論で `DeviceUnavailable` を返す。
    let result = run_gemm_through_dispatch(&ops, Device::Cuda(0), &a, &b);
    assert!(matches!(result, Err(BackendError::DeviceUnavailable(_))));
}

#[cfg(target_os = "macos")]
#[test]
fn same_code_dispatches_gemm_to_metal_backend_or_returns_typed_error() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();
    let ops: Vec<&dyn BackendOps> = vec![&cpu, &metal];

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

    // macOS 実機（GitHub Actions macOS ランナー・self-hosted 実機）では
    // Metal デバイスが通常利用可能なため実カーネルが成功するはずだが、
    // Metal 非対応環境（GPU 無効化等）でも panic せず
    // `BackendError::DeviceAllocationFailed`／`KernelLaunchFailed` を
    // 返すことを許容する（fail-safe。`error.rs::MetalError::DeviceUnavailable`
    // 起因）。
    match run_gemm_through_dispatch(&ops, Device::Metal, &a, &b) {
        Ok(result) => {
            assert_eq!(result.shape(), &[2, 2]);
        }
        Err(BackendError::DeviceAllocationFailed(_)) | Err(BackendError::KernelLaunchFailed(_)) => {
            // Metal 非対応環境での期待経路（panic しない）。
        }
        Err(other) => panic!("unexpected error variant for Metal gemm dispatch: {other}"),
    }
}

/// 全 8 演算のうち GEMM 以外（elementwise・reduction）は CUDA/Metal 未
/// 実装のため `BackendError::Unsupported` を返すことを確認する
/// （fail-safe 実装の受け皿。CPU は全演算とも実装済みのため
/// `Unsupported` を返さない）。
#[test]
fn cuda_elementwise_and_reduction_return_unsupported_not_panic() {
    let cuda = CudaBackendOps::new(0);
    let a = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]).expect("valid tensor");
    let b = a.clone();

    assert!(matches!(
        cuda.add(&a, &b),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        cuda.mul(&a, &b),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(cuda.relu(&a), Err(BackendError::Unsupported(_))));
    assert!(matches!(cuda.exp(&a), Err(BackendError::Unsupported(_))));
    assert!(matches!(cuda.tanh(&a), Err(BackendError::Unsupported(_))));
    assert!(matches!(
        cuda.sum(&a, None),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        cuda.max(&a, None),
        Err(BackendError::Unsupported(_))
    ));
}

#[test]
fn cpu_elementwise_and_reduction_are_fully_implemented() {
    let cpu = CpuBackendOps::new();
    let a = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]).expect("valid tensor");
    let b = a.clone();

    // CPU は参照実装のため全演算が `Unsupported` を返さず実カーネルを
    // 実行する（受け入れ条件「同一コードで 3 バックエンドのカーネルが
    // 呼び分けられる」の CPU 側の裏付け）。
    assert!(cpu.add(&a, &b).is_ok());
    assert!(cpu.mul(&a, &b).is_ok());
    assert!(cpu.relu(&a).is_ok());
    assert!(cpu.exp(&a).is_ok());
    assert!(cpu.tanh(&a).is_ok());
    assert!(cpu.sum(&a, None).is_ok());
    assert!(cpu.max(&a, None).is_ok());
}
