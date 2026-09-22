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

use std::collections::HashMap;
use std::path::PathBuf;

use fandhe_ai::Tensor;
use fandhe_ai::interop::onnx::{OnnxModel, OnnxValue};
use fandhe_ai_backend_cpu::parity::assert_parity;

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
