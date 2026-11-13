//! イシュー #1349（親 #1348・ルート #1341 → #1269）: 学習 step の update
//! 区間を対象とする CUDA Graph capture opt-in が **OFF** のときの
//! 非後退契約の実機検証（`docs/backend-cuda-graph-step-capture-design.md`）。
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
//! 本ファイルは単一テストのみを持つため `--test-threads` は問わない）。
//!
//! **#4167 以降の前提**: 推論 Graph（`FANDHE_AI_CUDA_GRAPH_INFER`）は既定 ON
//! のため、本テストは「明示 OFF（#4167 以降は既定 ON）」で legacy stream を
//! 保つことを検証する。`CudaDevice::new` の前に
//! `graph::set_infer_graph_enabled(false)` を API 明示で呼ぶ（API 明示は
//! 環境変数より優先されるため環境変数は触らない）。

use fandhe_ai_backend_cuda::graph::{set_infer_graph_enabled, step_graph_enabled};
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::BackendOps;
use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::buffer::DeviceBuffer;

/// opt-in 明示 OFF（#4167 以降は既定 ON のため推論側を API で明示 OFF）では
/// `captured_segment_key` が常に `Ok(None)` を
/// 返し、ストリームは legacy のまま（本イシュー導入前と挙動不変）で
/// あることを確認する（design doc §4.1 の非後退契約）。
///
/// 本ファイルは opt-in を一度も `true` にせず推論側も明示 OFF にするため、
/// `context_cache::cached_device` が本プロセス内で最初に確保する
/// `CudaDevice` は必ず `StreamKind::Legacy` になる（ファイル冒頭コメント
/// 参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn opt_in_off_keeps_legacy_stream_and_returns_none() {
    assert!(
        !step_graph_enabled(),
        "本ファイルは STEP opt-in を変更しないため常に OFF のはず"
    );
    // #4167: 推論 Graph の既定 ON を明示 OFF にしてから device を確保する。
    set_infer_graph_enabled(false);
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    assert!(
        !device.is_capturable_stream(),
        "opt-in OFF では legacy stream のまま（本イシュー導入前と挙動不変）"
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
    assert!(key.is_none(), "opt-in OFF では常に None のはず");
}
