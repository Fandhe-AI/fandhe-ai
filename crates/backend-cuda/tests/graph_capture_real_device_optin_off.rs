//! イシュー #1349（親 #1348・ルート #1341 → #1269）: 学習 step の update
//! 区間を対象とする CUDA Graph capture opt-in が **OFF**（step opt-in。既定）のときの
//! 非後退契約の実機検証。#3594 で学習 step 全体 capture が既定 ON になったため、stream 種別は
//! opt-in から導く（`FANDHE_AI_CUDA_GRAPH_TRAIN=0` で起動すれば legacy stream に戻ることの確認になる）（`docs/backend-cuda-graph-step-capture-design.md`）。
//!
//! **本ファイルを `graph_capture_real_device.rs`（opt-in ON 側）から分離
//! した理由（codex-review P2／Cursor Bugbot Low 指摘対応。PR #1390
//! 再修正）**: `context_cache::cached_device`（`ordinal` キーのプロセス内
//! singleflight キャッシュ）はプロセス生存期間中キャッシュされ、opt-in
//! フラグの値には関知しない。そのため、もし本テストが ON 側テストと
//! **同一プロセス**（＝同一 `cargo test` バイナリ）で実行されると、先に
//! 走った方が確保した `CudaDevice`（`StreamKind::Legacy`／`Created` の
//! いずれか）が後から走る側にもキャッシュ経由でそのまま見えてしまい、
//! opt-in フラグを変更しても実際に使われる `CudaDevice` の `StreamKind`
//! が追従しない——テスト実行順序に判定が依存する競合状態になる
//! （`GraphOptInGuard` のプロセスワイド `Mutex` はテストの**時間的な**
//! 直列化はできても、この**プロセス内キャッシュの共有**そのものは
//! 解消できない）。`cargo test` は `tests/` 配下のファイルをそれぞれ
//! 独立したバイナリ（＝独立プロセス）としてビルド・実行するため、別
//! ファイルへ分けることでキャッシュ共有を構造的に断つ。
//!
//! 実行方法は `graph_capture_real_device.rs` と同じ
//! （`cargo test -p fandhe-ai-backend-cuda --release --test \
//! graph_capture_real_device_optin_off -- --ignored --nocapture`。
//! 本ファイルは opt-in を一切変更しない単一テストのみを持つため
//! `--test-threads` は問わない）。

use fandhe_ai_backend_cuda::graph::{infer_graph_enabled, step_graph_enabled, train_graph_enabled};
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::BackendOps;
use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::buffer::DeviceBuffer;

/// step opt-in OFF では `captured_segment_key` が常に `Ok(None)` を返す。stream 種別は
/// 3 つの opt-in から導く（#3594 以降は学習 capture が既定 ON のため通常は created。
/// `FANDHE_AI_CUDA_GRAPH_TRAIN=0` 等で全 opt-in を OFF にすると legacy。`internal-diagnostics`
/// ビルドは常に legacy。design doc §4.1・§8.12）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn opt_in_off_keeps_legacy_stream_and_returns_none() {
    assert!(
        !step_graph_enabled(),
        "本ファイルは step opt-in を変更しないため常に OFF のはず"
    );
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    let expected_capturable =
        !cfg!(feature = "internal-diagnostics") && (train_graph_enabled() || infer_graph_enabled());
    assert_eq!(
        device.is_capturable_stream(),
        expected_capturable,
        "stream 種別は opt-in から導かれるはず（全 opt-in OFF なら legacy のまま）"
    );

    let cuda = CudaBackendOps::new(0);
    let mem = cuda.memory_ops().expect("memory_ops must be Some");
    let param = mem
        .upload(&Tensor::new(vec![1.0], &[1]).unwrap())
        .expect("upload succeeds");
    let refs: [&DeviceBuffer<f32>; 1] = [&param];
    let key = cuda
        .captured_segment_key(&refs, 0)
        .expect("captured_segment_key must not error when opt-in is off");
    assert!(key.is_none(), "step opt-in OFF では常に None のはず");
}
