//! `CudaMemory::with_host_view`（[`MemoryOps::with_host_view`] の CUDA
//! 実装。イシュー #1336）の実機必須テスト。
//!
//! CUDA 実機（DGX Spark GB10 等）でのみ意味を持つ肯定的検証（`download`／
//! `to_vec()` との bit 同一・managed 配置のコピーなし借用・
//! `PooledMemory` 経由のパススルー・キャッシュ再利用・
//! `release_host_staging` のリーク検出）を `#[ignore]` 分離のうえ本
//! ファイルに置く（`.claude/rules/coding-rust.md` の実機依存分離方針、
//! `crates/backend-cuda/tests/memory_real_device.rs` と同じ構成）。
//! 実行するには CUDA 実機で以下を叩く:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --all-features \
//!     --test host_view_real_device -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `host_staging_stats`（キャッシュ hit／miss 観測）は `internal-
//! diagnostics` feature 限定の公開 API のため `--all-features` が必須
//! （`crates/backend-cuda/src/memory.rs::CudaMemory::host_staging_stats`
//! ドキュメンテーションコメント参照）。

use fandhe_ai_backend_cuda::placement::{managed_placement_enabled, set_managed_placement_enabled};
use fandhe_ai_backend_cuda::{CudaDevice, CudaMemory, HostStagingKind};
use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::buffer::MemoryOps;
use fandhe_ai_tensor_core::device::Device;
use fandhe_ai_tensor_core::pool::{PoolConfig, PooledMemory};

/// `crate::placement` の opt-in フラグはプロセスグローバル（`AtomicBool`）
/// のため、`cargo test` の既定並列実行下での相互干渉を避けて直列化・
/// 原状復帰する RAII ガード（`managed_placement_real_device.rs::
/// PlacementFlagGuard` と同型。統合テストは `crate::placement::
/// test_support`〈`pub(crate)` 限定〉へ到達できないため、本ファイル専用の
/// ロックを持つ）。
static PLACEMENT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct PlacementFlagGuard {
    original: bool,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl PlacementFlagGuard {
    fn acquire(enabled: bool) -> Self {
        let lock = PLACEMENT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let original = managed_placement_enabled();
        set_managed_placement_enabled(enabled);
        Self {
            original,
            _lock: lock,
        }
    }
}

impl Drop for PlacementFlagGuard {
    fn drop(&mut self) {
        set_managed_placement_enabled(self.original);
    }
}

/// `with_host_view` が返す借用スライスが `download().as_slice()` および
/// `upload` に渡した元データと bit 完全一致することを確認する（R2 の
/// 中心的な受け入れ条件。tolerance は使わない）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn with_host_view_matches_download_and_upload_source_bit_exact() {
    // 既定並列度実行下では `with_host_view_matches_download_on_managed_
    // placement`（プロセスグローバル `crate::placement` フラグを一時的に
    // `true` へ切り替える）と並行実行されうる。本テストは `Device`
    // 配置（`host_staging` 経由）を前提とするため、`false` 固定で
    // 直列化する（codex-review 指摘: managed 配置テストのみがガードを
    // 取得しており、他テストが並行して managed 配置へ切り替わりうる
    // 競合を防ぐ）。
    let _guard = PlacementFlagGuard::acquire(false);
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let mem = CudaMemory::new(&device);

    let data: Vec<f32> = (0..4096)
        .map(|i| {
            // NaN／inf の伝播経路（`memcpy_dtoh` はビット列をそのまま
            // 転送するため丸めは発生しないはず）も含めて bit 一致を
            // 検証する。
            match i % 7 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => f32::MIN_POSITIVE,
                _ => (i as f32) * 0.5 - 100.0,
            }
        })
        .collect();
    let tensor = Tensor::<f32>::new(data.clone(), &[64, 64]).unwrap();

    let buf = mem
        .upload(&tensor)
        .expect("upload must succeed on real hardware");

    let downloaded = mem.download(&buf).expect("download must succeed");
    let download_bits: Vec<u32> = downloaded
        .as_slice()
        .expect("download returns a contiguous tensor")
        .iter()
        .map(|v| v.to_bits())
        .collect();

    let mut view_bits: Option<Vec<u32>> = None;
    mem.with_host_view(&buf, &mut |slice| {
        view_bits = Some(slice.iter().map(|v| v.to_bits()).collect());
    })
    .expect("with_host_view must succeed on real hardware");
    let view_bits = view_bits.expect("closure must be invoked exactly once");

    let source_bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        view_bits, source_bits,
        "with_host_view must match upload source bit-for-bit"
    );
    assert_eq!(
        view_bits, download_bits,
        "with_host_view must match download().as_slice() bit-for-bit"
    );
}

/// 同一形状（要素数）を複数回 `with_host_view` すると、2 回目以降は
/// ステージングキャッシュを再利用する（`host_staging_stats().hits` が
/// 増加する）ことを確認する。1 回目の miss は初回確保のため許容する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn with_host_view_reuses_staging_buffer_for_same_shape() {
    // `Device` 配置（`host_staging` 経由）前提のため `false` 固定で直列化
    // する（上記 `with_host_view_matches_download_and_upload_source_bit_
    // exact` と同じ理由）。
    let _guard = PlacementFlagGuard::acquire(false);
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let mem = CudaMemory::new(&device);

    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let tensor = Tensor::<f32>::new(data, &[1024]).unwrap();
    let buf = mem.upload(&tensor).expect("upload must succeed");

    let stats_before = mem.host_staging_stats();

    for _ in 0..3 {
        mem.with_host_view(&buf, &mut |_slice| {})
            .expect("with_host_view must succeed");
    }

    let stats_after = mem.host_staging_stats();
    assert!(
        stats_after.hits > stats_before.hits,
        "2 回目以降の呼び出しはキャッシュ hit を観測するはず: before={stats_before:?} after={stats_after:?}"
    );
}

/// `release_host_staging` がキャッシュ済みバッファを全て解放し、以降の
/// 呼び出しでは再確保（miss）から始まることを確認する（REQ-14
/// `release_cached` 系と同型の明示解放 API の受け入れ条件）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn release_host_staging_clears_cache_and_frees_reported_bytes() {
    // `Device` 配置（`host_staging` 経由）前提のため `false` 固定で直列化
    // する（`with_host_view_matches_download_and_upload_source_bit_exact`
    // と同じ理由）。
    let _guard = PlacementFlagGuard::acquire(false);
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let mem = CudaMemory::new(&device);

    let data: Vec<f32> = (0..2048).map(|i| i as f32).collect();
    let tensor = Tensor::<f32>::new(data, &[2048]).unwrap();
    let buf = mem.upload(&tensor).expect("upload must succeed");

    mem.with_host_view(&buf, &mut |_slice| {})
        .expect("first with_host_view must succeed");
    let stats_before_release = mem.host_staging_stats();
    assert!(
        stats_before_release.cached_bytes > 0,
        "使用後は少なくとも 1 エントリがキャッシュに戻っているはず"
    );

    let freed = mem.release_host_staging();
    assert!(freed > 0, "release_host_staging は解放バイト数を返すはず");

    let stats_after_release = mem.host_staging_stats();
    assert_eq!(
        stats_after_release.cached_bytes, 0,
        "release_host_staging 後はキャッシュが空になるはず"
    );

    // 解放後の再呼び出しは miss から始まる（新規確保が必要になる）。
    let misses_before = stats_after_release.misses;
    mem.with_host_view(&buf, &mut |_slice| {})
        .expect("release 後の with_host_view も成功するはず");
    let stats_final = mem.host_staging_stats();
    assert!(
        stats_final.misses > misses_before,
        "release 後の初回呼び出しは miss を観測するはず"
    );
}

/// `PooledMemory<CudaMemory>`（TASK-#201・REQ-14 14-3）が `with_host_view`
/// をそのまま `inner.with_host_view` へ転送し、`download` と同じ bit 同一
/// 結果を返すことを確認する（`tensor_core::pool::PooledMemory::
/// with_host_view` は透過ダウンキャストで動作する設計。`pool.rs` の
/// ドキュメンテーションコメント参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn pooled_memory_with_host_view_matches_download_bit_exact() {
    // `Device` 配置（`host_staging` 経由）前提のため `false` 固定で直列化
    // する（`with_host_view_matches_download_and_upload_source_bit_exact`
    // と同じ理由）。
    let _guard = PlacementFlagGuard::acquire(false);
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let inner = CudaMemory::new(&device);
    let mem = PooledMemory::new(inner, Device::Cuda(device.ordinal()), PoolConfig::default());

    let data: Vec<f32> = (0..512).map(|i| (i as f32) * 1.5).collect();
    let tensor = Tensor::<f32>::new(data.clone(), &[512]).unwrap();
    let buf = mem.upload(&tensor).expect("upload must succeed");

    let downloaded = mem.download(&buf).expect("download must succeed");
    let mut observed: Option<Vec<f32>> = None;
    mem.with_host_view(&buf, &mut |slice| observed = Some(slice.to_vec()))
        .expect("with_host_view must succeed through PooledMemory");
    let observed = observed.expect("closure must be invoked");

    assert_eq!(
        observed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        downloaded
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

/// managed 配置（イシュー #1352。`crate::placement::
/// managed_placement_enabled()` opt-in）の `with_host_view` がコピーなし
/// 借用で `download`（内部で `to_vec()` する経路）と bit 同一の結果を
/// 返すことを確認する。`managed_placement_real_device.rs` と同じ
/// `PlacementFlagGuard` 方式でフラグを直列化する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn with_host_view_matches_download_on_managed_placement() {
    let _guard = PlacementFlagGuard::acquire(true);
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    if !device.managed_memory_supported() {
        // managed memory 非対応デバイス（`CU_DEVICE_ATTRIBUTE_MANAGED_
        // MEMORY` 等）では `upload`/`alloc_zeroed` が `Unsupported` を
        // 返すため、`managed_placement_real_device.rs` と同じ判断で
        // スキップする。
        return;
    }
    let mem = CudaMemory::new(&device);

    let data: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.25 - 50.0).collect();
    let tensor = Tensor::<f32>::new(data.clone(), &[1024]).unwrap();
    let buf = mem
        .upload(&tensor)
        .expect("managed 配置の upload は成功するはず");

    let downloaded = mem.download(&buf).expect("download must succeed");
    let mut observed: Option<Vec<f32>> = None;
    mem.with_host_view(&buf, &mut |slice| observed = Some(slice.to_vec()))
        .expect("managed 配置の with_host_view は成功するはず");
    let observed = observed.expect("closure must be invoked");

    assert_eq!(
        observed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        downloaded
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "with_host_view (managed) must match download().as_slice() bit-for-bit"
    );
}

/// codex-review 指摘（イシュー #1336）: 本番既定 [`HostStagingKind::
/// Pageable`] のみが `with_host_view` の通常経路を通り、`Pinned`
/// （page-locked・WRITECOMBINED）経路は実機テスト・A/B 比較のいずれから
/// も到達できていなかった。本テストは `CudaMemory::
/// with_host_view_using_kind`（`internal-diagnostics` feature 限定の
/// 診断専用入口）を介して `Pinned` を明示的に選び、`download()` および
/// `Pageable` 経由の結果と bit 完全一致することを確認する（`Pinned`
/// 経路が実際に driver へ到達し、かつ両種別が同じ D2H 内容を返すことの
/// 直接検証。tolerance は使わない）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn with_host_view_using_kind_pinned_matches_pageable_and_download_bit_exact() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let mem = CudaMemory::new(&device);

    let data: Vec<f32> = (0..4096)
        .map(|i| match i % 7 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => f32::MIN_POSITIVE,
            _ => (i as f32) * 0.5 - 100.0,
        })
        .collect();
    let tensor = Tensor::<f32>::new(data, &[64, 64]).unwrap();
    let buf = mem
        .upload(&tensor)
        .expect("upload must succeed on real hardware");

    let downloaded = mem.download(&buf).expect("download must succeed");
    let download_bits: Vec<u32> = downloaded
        .as_slice()
        .expect("download returns a contiguous tensor")
        .iter()
        .map(|v| v.to_bits())
        .collect();

    let mut pageable_bits: Option<Vec<u32>> = None;
    mem.with_host_view_using_kind(&buf, HostStagingKind::Pageable, &mut |slice| {
        pageable_bits = Some(slice.iter().map(|v| v.to_bits()).collect());
    })
    .expect("with_host_view_using_kind(Pageable) must succeed on real hardware");
    let pageable_bits = pageable_bits.expect("closure must be invoked exactly once");

    let mut pinned_bits: Option<Vec<u32>> = None;
    mem.with_host_view_using_kind(&buf, HostStagingKind::Pinned, &mut |slice| {
        pinned_bits = Some(slice.iter().map(|v| v.to_bits()).collect());
    })
    .expect("with_host_view_using_kind(Pinned) must succeed on real hardware");
    let pinned_bits = pinned_bits.expect("closure must be invoked exactly once");

    assert_eq!(
        pinned_bits, download_bits,
        "Pinned 経路は download().as_slice() と bit 完全一致するはず"
    );
    assert_eq!(
        pinned_bits, pageable_bits,
        "Pinned 経路は Pageable 経路と bit 完全一致するはず（種別による内容差異は許容しない）"
    );
}
