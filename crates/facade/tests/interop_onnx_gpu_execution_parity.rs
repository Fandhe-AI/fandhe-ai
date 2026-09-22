//! `fandhe_ai::set_cuda_onnx_gpu_execution_enabled`／
//! `fandhe_ai::set_metal_onnx_gpu_execution_enabled`（イシュー #2077）の
//! 実機 parity テスト。`.claude/rules/coding-rust.md`「実機（DGX Spark
//! GB10・Metal 実機）依存テストは `#[ignore]` で分離」に従う。
//!
//! `tests/interop_onnx_gpu_execution_optin.rs` の非 `#[ignore]` テスト
//! （fail-closed／bit 不変の固定）とは異なり、本ファイルは
//! 「GPU 実行（ON）とホスト実行（OFF）が REQ-2 複合判定で一致する」
//! （契約 (c)）を実機で確認する。
//!
//! 実行例:
//! `cargo test -p fandhe-ai --release --test interop_onnx_gpu_execution_parity -- --ignored --nocapture`
//!
//! 未実測分の申し送りは `docs/perf/logs/onnx-gpu-execution-2077/README.md`
//! を参照。
//!
//! **フラグ排他制御（codex-review 指摘・PR #2222）**: `CUDA_ONNX_GPU_EXEC`／
//! `METAL_ONNX_GPU_EXEC` はいずれもプロセスグローバル（`OnnxModel::run`
//! は CUDA フラグを優先分岐するため `crates/facade/src/interop/onnx.rs`
//! 参照）で、両テストが同一プロセス内で並列実行されると CUDA 側の
//! ON 設定が Metal 側の parity 検証に割り込み、CUDA 優先分岐へ入って
//! 誤判定・失敗しうる（`#[ignore]` は実行のみをスキップしテスト関数
//! 自体は登録されるため、`--ignored` 一括実行時は既定でスレッド並列に
//! 実行される）。`tests/interop_onnx_gpu_execution_optin.rs::
//! CudaOnnxGpuExecGuard`・`tests/cuda_tf32_gemm_optin.rs::Tf32FlagGuard`
//! と同型の RAII ガードで両フラグをまとめて直列化・原状復帰する
//! （CUDA・Metal 双方の実機テストが同一ロックを取得するため、片方の
//! プラットフォームでしか実行されない環境でも安全）。

use std::collections::HashMap;
use std::path::PathBuf;

use fandhe_ai::Tensor;
use fandhe_ai::interop::onnx::{OnnxModel, OnnxValue};
use fandhe_ai_backend_cpu::parity::assert_parity;

/// `set_cuda_onnx_gpu_execution_enabled`／`set_metal_onnx_gpu_execution_
/// enabled` の原状復帰ガード。両フラグはプロセスグローバルかつ
/// `OnnxModel::run` の分岐で CUDA を優先するため、CUDA 版・Metal 版
/// 双方のテストが同一の `LOCK` を取得することで、並列実行時に一方の
/// フラグ変更がもう一方の parity 検証へ混入するのを防ぐ。
struct OnnxGpuExecGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    original_cuda: bool,
    // `set_metal_onnx_gpu_execution_enabled`／`metal_onnx_gpu_execution_
    // enabled` は macOS 限定 cfg のため非 macOS ビルドでは存在しない
    // （`crates/facade/src/lib.rs` 参照）。フィールド自体も cfg で
    // 分離し、非 macOS では CUDA フラグのみを対象に直列化・原状復帰する
    // （Metal テストは `#[cfg(target_os = "macos")]` で非 macOS では
    // コンパイル対象外のため、CUDA 単独の直列化で足りる）。
    #[cfg(target_os = "macos")]
    original_metal: bool,
}

impl OnnxGpuExecGuard {
    fn acquire() -> Self {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let original_cuda = fandhe_ai::cuda_onnx_gpu_execution_enabled();
        #[cfg(target_os = "macos")]
        let original_metal = fandhe_ai::metal_onnx_gpu_execution_enabled();
        Self {
            _lock: lock,
            original_cuda,
            #[cfg(target_os = "macos")]
            original_metal,
        }
    }
}

impl Drop for OnnxGpuExecGuard {
    fn drop(&mut self) {
        fandhe_ai::set_cuda_onnx_gpu_execution_enabled(self.original_cuda);
        #[cfg(target_os = "macos")]
        fandhe_ai::set_metal_onnx_gpu_execution_enabled(self.original_metal);
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
    let result = model.run(feeds).expect("run は成功するはず（実機前提）");
    match &result["output"] {
        OnnxValue::F32(t) => t.clone(),
        other => panic!("OnnxValue::F32 を期待したが {other:?}"),
    }
}

/// CUDA 実機（DGX Spark GB10 等）: opt-in ON（GPU 実行）と OFF（ホスト
/// CPU 実行）の出力が REQ-2 統一複合判定で一致することを確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_onnx_gpu_execution_on_matches_off_within_req2_on_real_device() {
    let _guard = OnnxGpuExecGuard::acquire();
    let model =
        OnnxModel::from_path(onnx_interop_fixture("model.onnx")).expect("from_path は成功する");

    fandhe_ai::set_cuda_onnx_gpu_execution_enabled(false);
    let off = run_model_onnx(&model, [0.3, 0.7]);

    fandhe_ai::set_cuda_onnx_gpu_execution_enabled(true);
    let on = run_model_onnx(&model, [0.3, 0.7]);
    fandhe_ai::set_cuda_onnx_gpu_execution_enabled(false);

    assert_parity(
        "OnnxModel::run: CUDA opt-in ON vs OFF（実機）",
        on.contiguous().as_slice().unwrap(),
        off.contiguous().as_slice().unwrap(),
    );
}

/// Metal 実機（Apple Silicon）: CUDA 版と同型の契約 (c) 検証。
///
/// **`#[cfg(target_os = "macos")]` 必須**（`Device::Metal` は cfg 限定
/// variant。`#[ignore]` は実行のみスキップしコンパイルはスキップしない
/// ため、非 macOS でもコンパイルが通る必要がある。
/// `crates/facade/tests/softmax_backend_parity.rs:145-152` と同じ注記）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_onnx_gpu_execution_on_matches_off_within_req2_on_real_device() {
    let _guard = OnnxGpuExecGuard::acquire();
    let model =
        OnnxModel::from_path(onnx_interop_fixture("model.onnx")).expect("from_path は成功する");

    fandhe_ai::set_metal_onnx_gpu_execution_enabled(false);
    let off = run_model_onnx(&model, [0.3, 0.7]);

    fandhe_ai::set_metal_onnx_gpu_execution_enabled(true);
    let on = run_model_onnx(&model, [0.3, 0.7]);
    fandhe_ai::set_metal_onnx_gpu_execution_enabled(false);

    assert_parity(
        "OnnxModel::run: Metal opt-in ON vs OFF（実機）",
        on.contiguous().as_slice().unwrap(),
        off.contiguous().as_slice().unwrap(),
    );
}
