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
//! `run_captured_segment`）の契約——capture 可否判定・再利用（Replayed）・
//! fail-closed（同期点拒否・capture エラーの poison 伝播）——を検証する。

use std::sync::Arc;

use fandhe_ai_backend_cuda::graph::{set_step_graph_enabled, step_graph_enabled};
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::buffer::DeviceBuffer;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, SegmentRun, SgdStepConfig, Tensor};

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
/// `run_captured_segment` が [`SegmentRun::Captured`] を返し、param が
/// 期待どおり更新されることを確認 → 2 回目（同一 key）が
/// [`SegmentRun::Replayed`] を返し、`body` を呼ばなくても同じ更新が
/// 再生されることを確認する（`body` に呼び出し回数カウンタを仕込み、
/// 2 回目は増えないことも検証する）。
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

    let mut call_count = 0usize;
    {
        let param_ptr: *mut DeviceBuffer<f32> = &mut param;
        let grad_ref = &grad;
        let mut body = || -> Result<(), BackendError> {
            call_count += 1;
            // SAFETY: `param_ptr` は同一スコープの `param` を指し、
            // capture 中は他に借用されない（`body` は capture 中に 1 回
            // だけ呼ばれる契約。`run_captured_segment` doc 参照）。
            let param_mut = unsafe { &mut *param_ptr };
            cuda.sgd_step_device(param_mut, grad_ref, None, &config)
        };
        let run1 = cuda
            .run_captured_segment(key.clone(), &mut body)
            .expect("first run_captured_segment call must succeed (capture)");
        assert_eq!(run1, SegmentRun::Captured);
    }
    assert_eq!(
        call_count, 1,
        "初回は capture のため body が 1 回呼ばれるはず"
    );

    let after_first = mem.download(&param).expect("download succeeds");
    // lr=0.5, grad=0.1 → param -= 0.05 per element.
    fandhe_ai_backend_cpu::parity::assert_parity(
        "graph capture sgd step 1",
        after_first.as_slice().unwrap(),
        &[0.95, 1.95, 2.95, 3.95],
    );

    {
        let mut body = || -> Result<(), BackendError> {
            call_count += 1;
            Ok(())
        };
        let run2 = cuda
            .run_captured_segment(key, &mut body)
            .expect("second run_captured_segment call must succeed (replay)");
        assert_eq!(run2, SegmentRun::Replayed);
    }
    assert_eq!(
        call_count, 1,
        "2 回目は再生のため body は呼ばれないはず（呼び出し回数は 1 のまま）"
    );

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

/// capture 中に同期点（`MemoryOps::download`）を呼ぶと、driver に
/// 触れる前に `Unsupported` で拒否され、ordinal は poison しない
/// （受け入れ条件 (b) 前半。design doc §4.7）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。--test-threads=1 で実行すること"]
fn sync_point_call_during_capture_is_rejected_without_poisoning() {
    let _guard = GraphOptInGuard::enable();
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    assert!(device.is_capturable_stream());

    let cuda = CudaBackendOps::new(0);
    let mem = cuda.memory_ops().expect("memory_ops must be Some");
    let param = mem
        .upload(&Tensor::new(vec![1.0], &[1]).unwrap())
        .expect("upload succeeds");
    let grad = mem
        .upload(&Tensor::new(vec![0.1], &[1]).unwrap())
        .expect("upload succeeds");
    let refs: [&DeviceBuffer<f32>; 2] = [&param, &grad];
    let key = cuda
        .captured_segment_key(&refs, 1)
        .expect("captured_segment_key must succeed")
        .expect("must be Some when opt-in is on");

    let mut body = || -> Result<(), BackendError> {
        // capture 中に同期点（download）を呼ぶ契約違反。
        mem.download(&param).map(|_| ())
    };
    let result = cuda.run_captured_segment(key, &mut body);
    assert!(
        matches!(result, Err(BackendError::Unsupported(_))),
        "capture 中の同期点呼び出しは Unsupported で拒否されるはず: {result:?}"
    );

    // ordinal は poison していないはず（同期点ガードは driver に触れる
    // 前に拒否するため sticky エラーを生じさせない）。次の通常操作が
    // 成功することで間接的に確認する。
    let after = mem
        .download(&param)
        .expect("ordinal must not be poisoned after a rejected sync-point-during-capture call");
    assert_eq!(after.as_slice().unwrap(), &[1.0]);
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
