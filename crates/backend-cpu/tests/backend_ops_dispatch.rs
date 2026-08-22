//! TASK-1.9c（#46）の受け入れ条件「同一コードで 3 バックエンド（CPU／
//! CUDA／Metal）のカーネルが呼び分けられる」を直接検証する統合テスト。
//!
//! `fandhe_ai_backend_cpu::CpuBackendOps`・`fandhe_ai_backend_cuda::CudaBackendOps`・
//! （macOS のみ）`fandhe_ai_backend_metal::MetalBackendOps` を `Vec<&dyn BackendOps>`
//! へ束ね、`fandhe_ai_tensor_core::ops_for` が単一の trait オブジェクト経由で
//! `Device` ごとに正しい実装へディスパッチできることを、同一の呼び出し
//! コード（[`run_gemm_through_dispatch`]）で検証する。
//!
//! - CPU: 実際に `gemm` を実行し既知値と一致することを確認する（CI で
//!   常時実行。CPU は無条件で有効なため実カーネル検証が可能）。
//! - CUDA/Metal: `ops_for` が正しい実装を選択すること、およびデバイス
//!   不在環境（本 CI 環境は CUDA 非搭載・非 macOS）では型付きエラー
//!   （`CudaUnavailable` 等）を返すこと（panic しない）を検証する。
//!   実機での実行検証（GPU 実カーネル一致）は `#[ignore]` 分離し、3
//!   バックエンド網羅の本格的な統合テストは TASK-1.9d（#47）が
//!   `backend_ops_integration.rs`（CPU 全 8 演算・非 contiguous・エラー
//!   経路・端点・3 バックエンド横断エンドツーエンド）・
//!   `backend-cuda/tests/backend_ops_real_device.rs`・
//!   `backend-metal/tests/backend_ops_real_device.rs`（実機 `#[ignore]`）で
//!   引き継いだ（本テストは受け入れ条件検証に必要な最小限に留める）。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, Tensor, ops_for};

#[cfg(target_os = "macos")]
use fandhe_ai_backend_metal::MetalBackendOps;

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
fn cpu_gemm_dispatch_handles_non_contiguous_transpose_input() {
    // `CpuBackendOps::gemm` は非 contiguous な入力を `contiguous()` で
    // 実体化してから `as_slice()` を呼ぶ経路（crates/backend-cpu/src/
    // ops.rs）を持つが、既存テストは常に新規生成した contiguous な
    // テンソルのみを検証していたため回帰保護がなかった（Review 指摘
    // 対応）。`transpose` で意図的に非 contiguous なビューを作り、同じ
    // ディスパッチ経由で正しい結果が得られることを確認する。
    let cpu = CpuBackendOps::new();
    let ops: Vec<&dyn BackendOps> = vec![&cpu];

    // A^T = [[1, 3], [2, 4]] を transpose で作る（A = [[1, 2], [3, 4]]）。
    // A^T @ B = [[1,3],[2,4]] @ [[5,6],[7,8]] = [[26, 30], [38, 44]]。
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let a_t = a.transpose(0, 1).expect("valid transpose");
    assert!(
        !a_t.is_contiguous(),
        "transpose した 2x2 テンソルは非 contiguous なはず"
    );
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

    let result = run_gemm_through_dispatch(&ops, Device::Cpu, &a_t, &b)
        .expect("non-contiguous cpu gemm must succeed via contiguous() realization");

    assert_eq!(result.shape(), &[2, 2]);
    let out = result.as_slice().expect("contiguous result");
    let expected = [26.0f32, 30.0, 38.0, 44.0];
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
    // `BackendError::DeviceUnavailable`／`DeviceAllocationFailed`／
    // `KernelLaunchFailed` を返すことを許容する（fail-safe）。
    // `MetalBackendOps::gemm` は `MetalContext::new` の失敗（デバイス
    // 不在。`MetalError::DeviceUnavailable`）を `map_metal_error`
    // （`backend-metal/src/memory.rs`）経由で
    // `BackendError::DeviceUnavailable` に統一している
    // （`backend-metal/src/ops.rs`）ため、この分岐も許容する必要がある
    // （PR #262 Bugbot 指摘対応）。
    match run_gemm_through_dispatch(&ops, Device::Metal, &a, &b) {
        Ok(result) => {
            assert_eq!(result.shape(), &[2, 2]);
        }
        Err(BackendError::DeviceUnavailable(_))
        | Err(BackendError::DeviceAllocationFailed(_))
        | Err(BackendError::KernelLaunchFailed(_)) => {
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
    // イシュー #599: elementwise（add/mul/relu/exp/tanh）は CUDA カーネル
    // 実装済みのため `Unsupported` を返さなくなった。CUDA 非搭載環境
    // （本 CI 環境）では `device_handle()` が driver 初期化時点で
    // `BackendError::CudaUnavailable` を返す（`backend-cuda/src/ops.rs`
    // 参照）ため、ここでは「panic しない」ことと「`Unsupported` ではなく
    // `CudaUnavailable` へ変換される」ことを検証する（環境適応。実機での
    // 実カーネル一致検証は `backend-cuda/tests/backend_ops_real_device.rs`
    // の `#[ignore]` テストが引き継ぐ）。reduction（sum/max）は本イシュー
    // 時点でも未実装のため引き続き `Unsupported` を検証する。
    let cuda = CudaBackendOps::new(0);
    let a = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]).expect("valid tensor");
    let b = a.clone();

    assert_cuda_elementwise_env_adaptive(cuda.add(&a, &b), "add");
    assert_cuda_elementwise_env_adaptive(cuda.mul(&a, &b), "mul");
    assert_cuda_elementwise_env_adaptive(cuda.relu(&a), "relu");
    assert_cuda_elementwise_env_adaptive(cuda.exp(&a), "exp");
    assert_cuda_elementwise_env_adaptive(cuda.tanh(&a), "tanh");

    assert!(matches!(
        cuda.sum(&a, None),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        cuda.max(&a, None),
        Err(BackendError::Unsupported(_))
    ));
}

/// `cuda_elementwise_and_reduction_return_unsupported_not_panic` の
/// elementwise 演算共通の環境適応アサーション（実機なら `Ok`、非搭載環境
/// なら `CudaUnavailable`。いずれでも panic しないことが目的）。
fn assert_cuda_elementwise_env_adaptive(result: Result<Tensor<f32>, BackendError>, op_name: &str) {
    match result {
        Ok(_) => {}
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::{op_name}: {other}"),
    }
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
