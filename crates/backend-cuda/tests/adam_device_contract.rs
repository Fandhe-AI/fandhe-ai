//! `CudaBackendOps::adam_step_device`（イシュー #2069・in-place デバイス
//! 常駐 Adam・AdamW 更新）の Linux 実行可能な契約テスト。CUDA driver の
//! 有無を前提にしない env-adaptive スモーク（`typed_ops_f64_contract.rs`
//! と同じ構成方針）に加え、driver に触れる前に確定する検証
//! （device／shape 不一致）を検査する。
//!
//! 実機（GB10 等）での数値正しさそのものは `adam_device_real_device.rs`
//! の `#[ignore]` テストへ引き継ぐ。

use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::buffer::{BufferHandle, DeviceBuffer};
use fandhe_ai_tensor_core::device::Device;
use fandhe_ai_tensor_core::{AdamStepConfig, AdamStepKind, BackendError, BackendOps, Tensor};

/// 実ストレージを持たないダミーハンドル。本ファイルのテストは shape／
/// device 不一致（driver に一切触れる前に確定するエラー分岐）を検証
/// する用途のみに使うため、実データへのアクセスが発生する前に必ず
/// `Err` へ抜けるユースケースに限って使う（`ops.rs::tests::EmptyHandle`
/// と同型だが、`tests/` ディレクトリは別コンパイル単位のため
/// `pub(crate)` を越えて到達できず、本ファイル用に複製する）。
#[derive(Debug)]
struct DummyHandle;

impl BufferHandle for DummyHandle {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn dummy_cuda_buffer(ordinal: usize, shape: &[usize]) -> DeviceBuffer<f32> {
    DeviceBuffer::<f32>::new(Device::Cuda(ordinal), shape.to_vec(), Box::new(DummyHandle))
}

fn adam_config() -> AdamStepConfig {
    AdamStepConfig {
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        decay_factor: 1.0,
        step_size: 0.001,
        bias_correction2_sqrt: 1.0,
        kind: AdamStepKind::Coupled,
    }
}

/// `grad` の shape が `param` と一致しない場合、driver に触れる前に
/// `ShapeMismatch` を返す。
#[test]
fn grad_shape_mismatch_rejected_before_touching_driver() {
    let ops = CudaBackendOps::new(0);
    let mut param = dummy_cuda_buffer(0, &[4]);
    let grad = dummy_cuda_buffer(0, &[3]);
    let mut m = dummy_cuda_buffer(0, &[4]);
    let mut v = dummy_cuda_buffer(0, &[4]);
    let err = ops
        .adam_step_device(&mut param, &grad, &mut m, &mut v, &adam_config())
        .unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

/// `m` の shape が `param` と一致しない場合、driver に触れる前に
/// `ShapeMismatch` を返す。
#[test]
fn m_shape_mismatch_rejected_before_touching_driver() {
    let ops = CudaBackendOps::new(0);
    let mut param = dummy_cuda_buffer(0, &[4]);
    let grad = dummy_cuda_buffer(0, &[4]);
    let mut m = dummy_cuda_buffer(0, &[3]);
    let mut v = dummy_cuda_buffer(0, &[4]);
    let err = ops
        .adam_step_device(&mut param, &grad, &mut m, &mut v, &adam_config())
        .unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

/// `v` の shape が `param` と一致しない場合、driver に触れる前に
/// `ShapeMismatch` を返す。
#[test]
fn v_shape_mismatch_rejected_before_touching_driver() {
    let ops = CudaBackendOps::new(0);
    let mut param = dummy_cuda_buffer(0, &[4]);
    let grad = dummy_cuda_buffer(0, &[4]);
    let mut m = dummy_cuda_buffer(0, &[4]);
    let mut v = dummy_cuda_buffer(0, &[3]);
    let err = ops
        .adam_step_device(&mut param, &grad, &mut m, &mut v, &adam_config())
        .unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

/// CPU 側 `DeviceBuffer`（`CpuBackendOps::alloc_zeroed`）を `grad` に
/// 混ぜると driver に触れる前に `DeviceMismatch` を返す。
#[test]
fn cpu_buffer_mixed_in_rejected_as_device_mismatch() {
    let ops = CudaBackendOps::new(0);
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let cpu_mem = cpu_ops
        .memory_ops()
        .expect("CpuBackendOps must implement MemoryOps");

    let mut param = dummy_cuda_buffer(0, &[4]);
    let grad = cpu_mem.alloc_zeroed(&[4]).unwrap();
    let mut m = dummy_cuda_buffer(0, &[4]);
    let mut v = dummy_cuda_buffer(0, &[4]);
    let err = ops
        .adam_step_device(&mut param, &grad, &mut m, &mut v, &adam_config())
        .unwrap_err();
    assert!(matches!(err, BackendError::DeviceMismatch));
}

/// 有効な shape・device の Coupled／Decoupled 呼び出しは、driver 不在
/// 環境では `CudaUnavailable`（`Unsupported` になったら override 結線の
/// 回帰を意味する）を、driver 搭載環境（実機）では `Ok` を返す
/// （env-adaptive。`typed_ops_f64_contract.rs` と同じ分岐パターン）。
///
/// `CudaBackendOps::memory_ops()` は driver 不在環境では `device_handle()`
/// が失敗するため `None` を返す fail-safe 契約（`ops.rs::memory_ops` doc
/// 参照）であり、`adam_step_device` 自体が返す `CudaUnavailable` と同じ
/// 「driver 不在」を意味する。この PR が追加した当初の実装は
/// `.expect()` で `memory_ops()` の `None` を握り潰していなかったため、
/// CUDA driver 非搭載の CI（GitHub ホステッド `ubuntu-latest`）で panic
/// していた（イシュー #2069 PR #2210 CI 実測）。よって `memory_ops()`
/// の欠如も env-adaptive 分岐の一部として早期 return する。
#[test]
fn valid_shape_coupled_and_decoupled_succeed_or_return_cuda_unavailable_env_adaptive() {
    let ops = CudaBackendOps::new(0);
    let Some(mem) = ops.memory_ops() else {
        return;
    };

    let init = Tensor::new(vec![1.0f32, -2.0, 0.5, 3.25], &[4]).unwrap();
    let grad = Tensor::new(vec![0.1f32, 0.2, 0.3, 0.4], &[4]).unwrap();

    for kind in [AdamStepKind::Coupled, AdamStepKind::Decoupled] {
        let upload_result = (|| -> Result<(), BackendError> {
            let mut param = mem.upload(&init)?;
            let grad_buf = mem.upload(&grad)?;
            let mut m = mem.alloc_zeroed(&[4])?;
            let mut v = mem.alloc_zeroed(&[4])?;
            let config = AdamStepConfig {
                kind,
                ..adam_config()
            };
            ops.adam_step_device(&mut param, &grad_buf, &mut m, &mut v, &config)
        })();

        match upload_result {
            Ok(()) => {}
            Err(BackendError::CudaUnavailable(_)) => {}
            other => panic!("kind={kind:?}: unexpected result {other:?}"),
        }
    }
}

/// `adam_step_device_tracked`（既定委譲）経由も同じ env-adaptive 契約
/// （CUDA は `DispatchFailureCell` を無視して既定実装へ委譲するのみで
/// 独自 override を持たないため、`adam_step_device` と同じ結果になる
/// はず）。`memory_ops()` の `None`（driver 不在）を早期 return で扱う
/// 理由は上記
/// `valid_shape_coupled_and_decoupled_succeed_or_return_cuda_unavailable_env_adaptive`
/// のコメントと同じ。
#[test]
fn tracked_variant_matches_default_delegation_env_adaptive() {
    use fandhe_ai_tensor_core::DispatchFailureCell;

    let ops = CudaBackendOps::new(0);
    let Some(mem) = ops.memory_ops() else {
        return;
    };

    let init = Tensor::new(vec![1.0f32, -2.0, 0.5, 3.25], &[4]).unwrap();
    let grad = Tensor::new(vec![0.1f32, 0.2, 0.3, 0.4], &[4]).unwrap();
    let token = DispatchFailureCell::new();

    let result = (|| -> Result<(), BackendError> {
        let mut param = mem.upload(&init)?;
        let grad_buf = mem.upload(&grad)?;
        let mut m = mem.alloc_zeroed(&[4])?;
        let mut v = mem.alloc_zeroed(&[4])?;
        ops.adam_step_device_tracked(
            &mut param,
            &grad_buf,
            &mut m,
            &mut v,
            &adam_config(),
            &token,
        )
    })();

    match result {
        Ok(()) => {}
        Err(BackendError::CudaUnavailable(_)) => {}
        other => panic!("unexpected result {other:?}"),
    }
}
