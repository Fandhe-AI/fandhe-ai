//! H2D pinned staging（[`fandhe_ai_backend_cuda::set_pinned_h2d_enabled`]。
//! イシュー #1585・`crate::host_staging` モジュール「H2D 用ステージング」
//! 節）の実機必須テスト。
//!
//! codex-review P2 指摘（PR #1678）: 既存のクレート内テスト
//! （`host_staging.rs::h2d_staging_tests`）は `HostStagingKind::Pageable`
//! を使ったキャッシュ操作・フラグの on/off 切り替えのみを検証しており、
//! `PinnedHostSlice`（実 CUDA driver 確保・`unsafe` 経路）を実際に経由
//! する `upload_new`／`upload_into` の転送結果自体（連続 upload・部分
//! 更新・解放後の再確保・読み戻し）を bit 単位で検証するテストが存在
//! しなかった。本ファイルはその欠落を埋める（`.claude/rules/
//! coding-rust.md` の実機依存分離方針、`crates/backend-cuda/tests/
//! host_view_real_device.rs` と同じ構成）。実行するには CUDA 実機で
//! 以下を叩く:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --all-features \
//!     --test pinned_h2d_real_device -- --ignored --nocapture --test-threads=1
//! ```

use fandhe_ai_backend_cuda::{CudaDevice, CudaMemory, pinned_h2d_enabled, set_pinned_h2d_enabled};
use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::buffer::MemoryOps;
use fandhe_ai_tensor_core::device::Device;

/// `PINNED_H2D_ENABLED` はプロセスグローバル（`AtomicBool`）のため、
/// `cargo test` の既定並列実行下での相互干渉を避けて直列化・原状復帰
/// する RAII ガード（`host_view_real_device.rs::PlacementFlagGuard`・
/// `crates/backend-cuda/src/host_staging.rs::h2d_staging_tests::
/// FlagGuard` と同型。統合テストはクレート内 `#[cfg(test)]` 限定の
/// `pinned_h2d_test_support` へ到達できないため、本ファイル専用の
/// ロックを持つ）。
static PINNED_H2D_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct FlagGuard {
    original: bool,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl FlagGuard {
    fn acquire(enabled: bool) -> Self {
        let lock = PINNED_H2D_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let original = pinned_h2d_enabled();
        set_pinned_h2d_enabled(enabled);
        Self {
            original,
            _lock: lock,
        }
    }
}

impl Drop for FlagGuard {
    fn drop(&mut self) {
        set_pinned_h2d_enabled(self.original);
    }
}

/// `set_pinned_h2d_enabled(true)` 下での `upload`（`host_staging::
/// upload_new` の `Pinned` 経路）が、フラグ OFF（従来の
/// `clone_htod` 直呼び）と bit 完全一致の結果を返すことを確認する。
/// 同一 numel の A・B を連続 upload する（正方 GEMM の典型パターン。
/// `H2dStagingCache` が複数エントリを保持する設計の実機裏付け）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn pinned_h2d_sequential_uploads_match_plain_upload_bit_exact() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");

    let data_a: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.5 - 100.0).collect();
    let data_b: Vec<f32> = (0..4096).map(|i| -(i as f32) * 0.25 + 7.0).collect();
    let tensor_a = Tensor::<f32>::new(data_a.clone(), &[64, 64]).unwrap();
    let tensor_b = Tensor::<f32>::new(data_b.clone(), &[64, 64]).unwrap();

    // ベースライン: フラグ OFF（従来の `clone_htod` 直呼び経路）。
    let baseline_a;
    let baseline_b;
    {
        let _guard = FlagGuard::acquire(false);
        let mem = CudaMemory::new(&device);
        let buf_a = mem.upload(&tensor_a).expect("baseline upload A failed");
        let buf_b = mem.upload(&tensor_b).expect("baseline upload B failed");
        baseline_a = mem.download(&buf_a).expect("baseline download A failed");
        baseline_b = mem.download(&buf_b).expect("baseline download B failed");
    }

    // 対象: フラグ ON（`Pinned` staging 経由）。A・B を連続 upload し、
    // `H2dStagingCache` が同一 numel の複数エントリ（LIFO）を正しく
    // 扱うことを実機で確認する。
    let pinned_a;
    let pinned_b;
    {
        let _guard = FlagGuard::acquire(true);
        let mem = CudaMemory::new(&device);
        let buf_a = mem.upload(&tensor_a).expect("pinned upload A failed");
        let buf_b = mem.upload(&tensor_b).expect("pinned upload B failed");
        pinned_a = mem.download(&buf_a).expect("pinned download A failed");
        pinned_b = mem.download(&buf_b).expect("pinned download B failed");
    }

    for i in 0..64 {
        for j in 0..64 {
            assert_eq!(
                pinned_a.get(&[i, j]).unwrap().to_bits(),
                baseline_a.get(&[i, j]).unwrap().to_bits(),
                "A must be bit exact at [{i}, {j}]"
            );
            assert_eq!(
                pinned_b.get(&[i, j]).unwrap().to_bits(),
                baseline_b.get(&[i, j]).unwrap().to_bits(),
                "B must be bit exact at [{i}, {j}]"
            );
            assert_eq!(
                pinned_a.get(&[i, j]).unwrap().to_bits(),
                data_a[i * 64 + j].to_bits(),
                "A must match host source at [{i}, {j}]"
            );
            assert_eq!(
                pinned_b.get(&[i, j]).unwrap().to_bits(),
                data_b[i * 64 + j].to_bits(),
                "B must match host source at [{i}, {j}]"
            );
        }
    }
}

/// `set_pinned_h2d_enabled(true)` 下での `upload_into`（`host_staging::
/// upload_into` の `Pinned` 経路）による部分更新が、フラグ OFF と
/// bit 完全一致の結果を返すことを確認する（`DeviceParamStore::step` の
/// grad staging 書き込みパターンの実機裏付け）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn pinned_h2d_upload_into_partial_update_matches_plain_bit_exact() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");

    // dst は 256 要素、書き込むのは中央の 64 要素（部分更新）。
    let dst_numel = 256usize;
    let offset = 96usize;
    let patch: Vec<f32> = (0..64).map(|i| (i as f32) * 1.5 + 3.0).collect();
    let patch_tensor = Tensor::<f32>::new(patch.clone(), &[64]).unwrap();

    let run = |enabled: bool| -> Vec<f32> {
        let _guard = FlagGuard::acquire(enabled);
        let mem = CudaMemory::new(&device);
        let mut dst = mem
            .alloc_zeroed(&[dst_numel])
            .expect("alloc_zeroed for dst failed");
        assert_eq!(dst.device(), Device::Cuda(0));
        mem.upload_into(&patch_tensor, &mut dst, offset)
            .expect("upload_into failed");
        let back = mem.download(&dst).expect("download failed");
        (0..dst_numel).map(|i| back.get(&[i]).unwrap()).collect()
    };

    let baseline = run(false);
    let pinned = run(true);

    assert_eq!(baseline.len(), pinned.len());
    for i in 0..dst_numel {
        assert_eq!(
            pinned[i].to_bits(),
            baseline[i].to_bits(),
            "dst[{i}] must be bit exact between pinned and plain paths"
        );
        if (offset..offset + patch.len()).contains(&i) {
            assert_eq!(
                pinned[i].to_bits(),
                patch[i - offset].to_bits(),
                "written region dst[{i}] must match patch source"
            );
        } else {
            assert_eq!(pinned[i], 0.0, "untouched region dst[{i}] must stay zero");
        }
    }
}

/// `release_h2d_staging` でキャッシュを明示解放した直後に、同一 numel
/// で異なる内容の再 upload を行っても bit 完全一致の結果を返す
/// ことを確認する（「解放後読み戻し」。解放が新規確保〈`unsafe`
/// `alloc_pinned` の未初期化メモリ〉を強制するため、旧データの残留や
/// 未初期化ビットの露出がないことの直接検証）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn pinned_h2d_upload_after_release_staging_is_bit_exact() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let _guard = FlagGuard::acquire(true);
    let mem = CudaMemory::new(&device);

    let first: Vec<f32> = (0..1024).map(|i| (i as f32) * 3.0 - 17.0).collect();
    let tensor_first = Tensor::<f32>::new(first.clone(), &[1024]).unwrap();
    let buf_first = mem
        .upload(&tensor_first)
        .expect("first upload failed (should populate h2d_staging cache)");
    let back_first = mem.download(&buf_first).expect("first download failed");
    for (i, expected) in first.iter().enumerate() {
        assert_eq!(
            back_first.get(&[i]).unwrap().to_bits(),
            expected.to_bits(),
            "first upload must be bit exact at [{i}]"
        );
    }
    drop(buf_first);

    // キャッシュを明示解放（page-locked メモリの解放。以降の同一 numel
    // upload はキャッシュ miss となり `HostStaging::alloc` が新規に
    // `cuMemHostAlloc`〈未初期化〉する経路を通る）。
    let freed = mem.release_h2d_staging();
    assert!(
        freed > 0,
        "release_h2d_staging must report freed bytes after a cached pinned entry existed"
    );

    let second: Vec<f32> = (0..1024).map(|i| -(i as f32) * 0.75 + 5.0).collect();
    let tensor_second = Tensor::<f32>::new(second.clone(), &[1024]).unwrap();
    let buf_second = mem
        .upload(&tensor_second)
        .expect("second upload failed (post-release realloc)");
    let back_second = mem.download(&buf_second).expect("second download failed");
    for (i, expected) in second.iter().enumerate() {
        assert_eq!(
            back_second.get(&[i]).unwrap().to_bits(),
            expected.to_bits(),
            "post-release upload must be bit exact at [{i}] (no stale/uninitialized leakage)"
        );
    }
}
