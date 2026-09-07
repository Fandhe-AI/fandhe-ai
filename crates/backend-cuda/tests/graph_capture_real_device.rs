//! イシュー #1349（親 #1348・ルート #1341 → #1269）: 学習 step の update
//! 区間を対象とする CUDA Graph capture／instantiate／launch 経路の実機
//! 検証（`docs/backend-cuda-graph-step-capture-design.md`）。
//!
//! **本ファイルの全テストは opt-in（`CudaGraph::set_step_graph_enabled`
//! 相当。実体は `fandhe_ai_backend_cuda::graph::set_step_graph_enabled`）
//! をプロセスグローバルに変更する。`CudaDevice` は ordinal ごとに 1 回
//! だけ構築されプロセス生存期間中キャッシュされないためテスト間の直接
//! 干渉は限定的だが、opt-in フラグ自体はプロセスワイドで即座に他テスト
//! （同一バイナリ内の並行実行スレッド）にも見える。fail-closed の判定
//! （`captured_segment_key` が「opt-in ON かつ capturable stream」を
//! 要求する分岐）を安定させるため、本ファイルは
//! `cargo test -p fandhe-ai-backend-cuda --release --test \
//! graph_capture_real_device -- --ignored --nocapture --test-threads=1`
//! （単一スレッド実行）で実行する契約とする（`docs/real-hardware-
//! verification-env.md` の実行手順に本コマンドを追記する）。
//!
//! 受け入れ条件 (a)（10 step の loss・勾配・パラメータが bit 同一）は
//! `crates/facade/tests/cuda_graph_step_bit_identity.rs`（facade 経由・
//! `DeviceParamStore::step` 結線を通す。design doc §10）で検証する。
//! 本ファイルは `backend-cuda` 単体（`BackendOps::captured_segment_key`／
//! `run_captured_sgd_step_segment`）の契約——capture 可否判定・再利用
//! （Replayed）・fail-closed（capture エラーの poison 伝播）——を検証
//! する。
//!
//! **codex-review P0 指摘対応（`run_captured_sgd_step_segment` への
//! 改称）で削除したテスト**: 旧稿は `run_captured_segment` が任意
//! クロージャ `body` を受け取る設計だったため、「capture 中に同期点
//! （`MemoryOps::download`）を `body` 内から呼ぶと driver に触れる前に
//! `Unsupported` で拒否される」ことを `body` へ注入して直接検証する
//! テストを持っていた。新シグネチャは `param`／`grad`／`velocity` の
//! 3 引数のみを受け取り、区間本体（SGD 更新）は実装が固定的に行う
//! ため、そもそも任意コード（同期点呼び出しを含む）を capture 区間へ
//! 注入する経路自体が型レベルで存在しなくなった——このテストが検証
//! していた性質は「テストで確認すべき仕様」から「型システムが保証する
//! 不変条件」へ格上げされた。同期点ガード自体
//! （`context_cache::begin_sync_point_call`）の状態機械としての正しさ
//! は実機非依存の `context_cache.rs`
//! `begin_sync_point_call_rejects_before_touching_driver_while_capturing`
//! で引き続き検証する。

use std::sync::Arc;

use fandhe_ai_backend_cuda::graph::{set_step_graph_enabled, step_graph_enabled};
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::buffer::DeviceBuffer;
use fandhe_ai_tensor_core::{BackendOps, DispatchFailureCell, SegmentRun, SgdStepConfig, Tensor};

/// opt-in を明示的に OFF/ON へ切り替え、テスト終了時に OFF へ戻す RAII
/// ガード（`crate::precision::tests::FlagGuard` と同型。本ファイルは
/// `--test-threads=1` 前提のため直列化ロックは持たない）。
struct GraphOptInGuard;

impl GraphOptInGuard {
    fn enable() -> Self {
        set_step_graph_enabled(true);
        Self
    }
}

impl Drop for GraphOptInGuard {
    fn drop(&mut self) {
        set_step_graph_enabled(false);
    }
}

/// SGD の update 区間を capture・再生できることを確認する（design doc
/// §4.1 の実機プローブを兼ねる）。
///
/// 手順: opt-in を有効化 → `CudaDevice::new(0)`（capturable stream で
/// 初期化される）→ 連結パラメータ・grad staging・velocity なしの最小
/// 構成で `captured_segment_key` → `Some` を確認 → 1 回目の
/// `run_captured_sgd_step_segment` が [`SegmentRun::Captured`] を返し、
/// param が期待どおり更新されることを確認 → 2 回目（同一 key）が
/// [`SegmentRun::Replayed`] を返し、region 本体（実装内部の SGD 更新）を
/// 再実行しなくても同じ更新が再生されることを param の値で確認する
/// （codex-review P0 指摘対応で `body` クロージャの呼び出し回数カウンタは
/// 廃止済み——「本体を呼んだか」は SGD 更新の副作用〈param の値〉でしか
/// 観測できない設計になった）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。--test-threads=1 で実行すること"]
fn sgd_update_segment_captures_then_replays_bit_identically() {
    let _guard = GraphOptInGuard::enable();
    assert!(step_graph_enabled());

    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    assert!(
        device.is_capturable_stream(),
        "opt-in が最初のデバイス初期化より前に設定されていれば capturable stream になるはず"
    );

    let cuda = CudaBackendOps::new(0);
    let mem = cuda
        .memory_ops()
        .expect("CudaBackendOps::memory_ops must be Some");

    let numel = 4usize;
    let mut param = mem
        .upload(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[numel]).unwrap())
        .expect("param upload succeeds");
    let grad = mem
        .upload(&Tensor::new(vec![0.1, 0.1, 0.1, 0.1], &[numel]).unwrap())
        .expect("grad upload succeeds");

    let config = SgdStepConfig {
        lr: 0.5,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        is_first_step: true,
    };
    let config_key = config.lr.to_bits() as u64;

    let key = {
        let refs: [&DeviceBuffer<f32>; 2] = [&param, &grad];
        cuda.captured_segment_key(&refs, config_key)
            .expect(
                "captured_segment_key must not error when opt-in is on and stream is \
                     capturable",
            )
            .expect("captured_segment_key must return Some when opt-in is on")
    };

    let token = DispatchFailureCell::new();

    let run1 = cuda
        .run_captured_sgd_step_segment(key.clone(), &mut param, &grad, None, &config, &token)
        .expect("first run_captured_sgd_step_segment call must succeed (capture)");
    assert_eq!(run1, SegmentRun::Captured);

    let after_first = mem.download(&param).expect("download succeeds");
    // lr=0.5, grad=0.1 → param -= 0.05 per element.
    fandhe_ai_backend_cpu::parity::assert_parity(
        "graph capture sgd step 1",
        after_first.as_slice().unwrap(),
        &[0.95, 1.95, 2.95, 3.95],
    );

    let run2 = cuda
        .run_captured_sgd_step_segment(key, &mut param, &grad, None, &config, &token)
        .expect("second run_captured_sgd_step_segment call must succeed (replay)");
    assert_eq!(run2, SegmentRun::Replayed);

    let after_second = mem.download(&param).expect("download succeeds");
    fandhe_ai_backend_cpu::parity::assert_parity(
        "graph capture sgd step 2 (replayed)",
        after_second.as_slice().unwrap(),
        &[0.90, 1.90, 2.90, 3.90],
    );
}

/// opt-in OFF（既定）では `captured_segment_key` が常に `Ok(None)` を
/// 返し、ストリームは legacy のまま（本イシュー導入前と挙動不変）で
/// あることを確認する（design doc §4.1 の非後退契約）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。--test-threads=1 で実行すること"]
fn opt_in_off_keeps_legacy_stream_and_returns_none() {
    assert!(
        !step_graph_enabled(),
        "他テストの opt-in が残留していないこと"
    );
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

/// capture 済み graph が触れるバッファ集合を [`SegmentKey`] で捉えている
/// ため、異なる config（例えば学習率変更）は別 key になり再 capture が
/// 促されることを確認する（design doc §4.5 の「設定変更検出」契約の
/// backend-cuda 側の裏付け）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。--test-threads=1 で実行すること"]
fn different_config_key_produces_a_different_segment_key() {
    let _guard = GraphOptInGuard::enable();
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    assert!(device.is_capturable_stream());

    let cuda = CudaBackendOps::new(0);
    let mem = cuda.memory_ops().expect("memory_ops must be Some");
    let param = mem
        .upload(&Tensor::new(vec![1.0], &[1]).unwrap())
        .expect("upload succeeds");
    let refs: [&DeviceBuffer<f32>; 1] = [&param];

    let key_a = cuda
        .captured_segment_key(&refs, 1)
        .unwrap()
        .expect("Some when opt-in is on");
    let key_b = cuda
        .captured_segment_key(&refs, 2)
        .unwrap()
        .expect("Some when opt-in is on");
    assert_ne!(
        key_a, key_b,
        "config_key が異なれば SegmentKey も異なるはず（再 capture の契機）"
    );

    // `Device` を握るだけの参照カウントの生存確認（`Arc<CudaStream>` の
    // 寿命が本テスト内で有効であることの簡易チェック）。
    let _stream: &Arc<cudarc::driver::CudaStream> = device.stream();
}
