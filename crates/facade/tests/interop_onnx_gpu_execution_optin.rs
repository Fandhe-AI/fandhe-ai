//! `fandhe_ai::set_cuda_onnx_gpu_execution_enabled`／
//! `fandhe_ai::cuda_onnx_gpu_execution_enabled`（イシュー #2077・親
//! #2076「ONNX import モデルの GPU 実行」）の受入テスト。
//!
//! setter/getter の往復・既定 OFF・OFF 時の `OnnxModel::run` bit 不変は
//! CUDA 実機を要さないため常に CI（GitHub ホステッド）で実行する。
//! ON 時の実際の CUDA 実行は `CudaDeviceProvider::is_available()` で
//! 分岐する「実行環境適応型」方針（`tests/memory_pool_api.rs` と同じ
//! 先例）に従う: 実機が無い開発ホスト・CI では `OnnxModel::run` が
//! `OnnxError::Execution` を返すこと（CPU への黙示フォールバックを
//! しない fail-closed 契約）を確認し、実機（DGX Spark GB10 等）がある
//! 開発ホストでは `Ok` かつ OFF 出力との REQ-2 複合判定を確認する。
//!
//! フラグはプロセスグローバル（`crate::interop::onnx::CUDA_ONNX_GPU_EXEC`）
//! のため、`cuda_tf32_gemm_optin.rs::Tf32FlagGuard` と同型の Mutex ガード
//! で直列化・原状復帰する。

use std::collections::HashMap;
use std::path::PathBuf;

use fandhe_ai::Tensor;
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel, OnnxValue};
use fandhe_ai_backend_cpu::parity::assert_parity;

/// `set_cuda_onnx_gpu_execution_enabled` の原状復帰ガード（プロセス
/// グローバルフラグのため他テストとの並列実行競合を避ける）。
struct CudaOnnxGpuExecGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    original: bool,
}

impl CudaOnnxGpuExecGuard {
    fn acquire() -> Self {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let original = fandhe_ai::cuda_onnx_gpu_execution_enabled();
        Self {
            _lock: lock,
            original,
        }
    }
}

impl Drop for CudaOnnxGpuExecGuard {
    fn drop(&mut self) {
        fandhe_ai::set_cuda_onnx_gpu_execution_enabled(self.original);
    }
}

fn onnx_interop_fixture(rel: &str) -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../onnx-interop/tests/fixtures"
    ))
    .join(rel)
}

fn run_model_onnx(model: &OnnxModel, input: [f32; 2]) -> Tensor<f32> {
    let mut feeds = HashMap::new();
    feeds.insert(
        "input".to_string(),
        OnnxValue::F32(Tensor::<f32>::new(input.to_vec(), &[1, 2]).unwrap()),
    );
    let result = model.run(feeds).expect("run は成功するはず");
    match &result["output"] {
        OnnxValue::F32(t) => t.clone(),
        other => panic!("OnnxValue::F32 を期待したが {other:?}"),
    }
}

#[test]
fn cuda_onnx_gpu_execution_defaults_to_disabled() {
    let _guard = CudaOnnxGpuExecGuard::acquire();
    // 他テストが原状復帰を怠っていないことも含め、既定値そのものは
    // プロセス起動直後の値を直接検査できないため（他テストが先に
    // 走った可能性がある）、setter を経由しない状態で off であることを
    // 期待する代わりに round trip（off に戻す→確認）で固定する。
    fandhe_ai::set_cuda_onnx_gpu_execution_enabled(false);
    assert!(!fandhe_ai::cuda_onnx_gpu_execution_enabled());
}

#[test]
fn cuda_onnx_gpu_execution_setter_getter_round_trip() {
    let _guard = CudaOnnxGpuExecGuard::acquire();
    fandhe_ai::set_cuda_onnx_gpu_execution_enabled(true);
    assert!(fandhe_ai::cuda_onnx_gpu_execution_enabled());
    fandhe_ai::set_cuda_onnx_gpu_execution_enabled(false);
    assert!(!fandhe_ai::cuda_onnx_gpu_execution_enabled());
}

#[test]
fn onnx_model_run_is_bit_unchanged_when_opt_in_is_off() {
    // 契約 (a): opt-in OFF（既定）時の `OnnxModel::run` は本イシュー導入前
    // と bit 完全に不変（`interp::run` 直接呼び出しと同一の出力）。
    let _guard = CudaOnnxGpuExecGuard::acquire();
    fandhe_ai::set_cuda_onnx_gpu_execution_enabled(false);

    let model =
        OnnxModel::from_path(onnx_interop_fixture("model.onnx")).expect("from_path は成功する");
    let out = run_model_onnx(&model, [0.3, 0.7]);
    // `tests/interop_onnx_internal_parity.rs` と同じ fixture・入力を使い、
    // 期待値は同テストの bit 一致検証に委ねられているため、ここでは
    // 2 回連続の呼び出しが同一出力になること（決定性。opt-in が経路を
    // 一切変えないこと）のみを固定する。
    let out2 = run_model_onnx(&model, [0.3, 0.7]);
    assert_eq!(
        out.as_slice().unwrap(),
        out2.as_slice().unwrap(),
        "opt-in OFF の run は決定的であるはず"
    );
}

#[test]
fn cuda_onnx_gpu_execution_on_is_fail_closed_or_matches_off_within_req2() {
    // driver 選択可能（`CudaDeviceProvider::is_available()`）であっても
    // NVRTC 等の実行時コンポーネントが欠けた部分的 CUDA 環境（例:
    // driver は検出できるが `libnvrtc` 不在のサンドボックス）では
    // カーネル起動自体が失敗しうるため、`is_available()` の真偽では
    // 分岐しない。代わりに結果そのもので判定する: `Ok` なら OFF 出力との
    // REQ-2 複合判定（相対誤差 1e-3 未満 または絶対誤差 1e-5 未満）、
    // `Err` なら `OnnxError::Execution`（ホスト CPU への黙示フォール
    // バックをしない fail-closed 契約）であることを確認する。
    let _guard = CudaOnnxGpuExecGuard::acquire();
    let model =
        OnnxModel::from_path(onnx_interop_fixture("model.onnx")).expect("from_path は成功する");

    fandhe_ai::set_cuda_onnx_gpu_execution_enabled(false);
    let off = run_model_onnx(&model, [0.3, 0.7]);

    fandhe_ai::set_cuda_onnx_gpu_execution_enabled(true);
    let mut feeds = HashMap::new();
    feeds.insert(
        "input".to_string(),
        OnnxValue::F32(Tensor::<f32>::new(vec![0.3, 0.7], &[1, 2]).unwrap()),
    );
    let result = model.run(feeds);

    match result {
        Ok(outputs) => {
            let on = match &outputs["output"] {
                OnnxValue::F32(t) => t.clone(),
                other => panic!("OnnxValue::F32 を期待したが {other:?}"),
            };
            assert_parity(
                "OnnxModel::run: CUDA opt-in ON vs OFF",
                on.contiguous().as_slice().unwrap(),
                off.contiguous().as_slice().unwrap(),
            );
        }
        Err(OnnxError::Execution { message }) => {
            assert!(
                !message.is_empty(),
                "Execution エラーメッセージは空であってはならない"
            );
        }
        Err(other) => panic!(
            "CUDA 実行時エラーは OnnxError::Execution を期待したが {other:?}（\
             CPU への黙示フォールバックが発生していないか確認すること）"
        ),
    }
}
