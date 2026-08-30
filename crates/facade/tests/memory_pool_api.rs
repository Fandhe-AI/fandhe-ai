//! `fandhe_ai::release_cached_memory`／`fandhe_ai::memory_pool_stats`
//! （REQ-14 の明示解放 API。イシュー #1018 ツリー・#1020 CUDA・#1021
//! Metal）の受入テスト。
//!
//! CPU 経路（プールを持たない既定契約。`BackendOps` の既定メソッド。
//! `crates/tensor-core/src/backend_ops.rs`）は Linux CI でも実行できる
//! ため CI（GitHub ホステッド）で固定する。
//!
//! CUDA 経路は `tests/tape_construction.rs` と同じ「実行環境適応型」
//! 方針（`CudaDeviceProvider::is_available()` で選択可能デバイスの有無を
//! 判定してから分岐）に従い、実機がある場合のみ確保→drop→解放→
//! `cached_bytes == 0` を検証する。Metal 経路（実際のプール保持・解放の
//! 裏付け）は Apple Silicon 実機依存のため両方とも `#[ignore]` で分離
//! する（`.claude/rules/coding-rust.md`）。

use fandhe_ai::Device;

/// CPU バックエンドは `release_cached_device_memory`（`BackendOps` の
/// 既定実装）のまま常に `Ok(())` を返す（`docs/device-memory-pool-
/// design.md` §3.1「デフォルト実装（非破壊拡張）」）。
#[test]
fn release_cached_memory_on_cpu_is_ok() {
    fandhe_ai::release_cached_memory(Device::Cpu)
        .expect("CPU の release_cached_memory は常に成功するはず");
}

/// CPU バックエンドは `device_memory_pool_stats` の既定実装のまま
/// `Ok(None)` を返す（プールを持たないバックエンド。同 doc §3.1）。
#[test]
fn memory_pool_stats_on_cpu_is_ok_none() {
    let stats = fandhe_ai::memory_pool_stats(Device::Cpu)
        .expect("CPU の memory_pool_stats は常に成功するはず");
    assert!(stats.is_none(), "CPU はプールを持たないため None のはず");
}

/// Metal バックエンド: 確保 → drop → 解放 → `cached_bytes == 0` の
/// 一連の流れを facade 公開面のみで確認する（実機依存）。
///
/// facade は `MemoryOps`／`DeviceBuffer` を直接公開しないため（REQ-12。
/// `docs/compat-api-scope.md` §0）、本テストは確保自体を
/// `fandhe_ai::tape_for(Device::Metal)` 経由の演算（`Tape` の
/// `Var` 生成 API）で間接的に発生させることはせず、`release_cached_
/// memory`／`memory_pool_stats` の往復のみを確認する（確保・解放の
/// 直接検証は `crates/backend-metal/tests/pool_real_device.rs` が
/// `BackendOps` を直接使って担う）。
#[test]
#[cfg(target_os = "macos")]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn release_cached_memory_on_metal_reports_zero_cached_bytes_afterwards() {
    fandhe_ai::release_cached_memory(Device::Metal)
        .expect("Metal 実機で release_cached_memory は成功するはず");
    let stats = fandhe_ai::memory_pool_stats(Device::Metal)
        .expect("Metal 実機で memory_pool_stats は成功するはず")
        .expect("Metal はプールを持つため Some のはず");
    assert_eq!(
        stats.cached_bytes, 0,
        "release_cached_memory 直後は cached_bytes == 0 のはず"
    );
}

/// CUDA 実機（`libcuda` 到達可能かつ選択可能デバイス 1 台以上）がある
/// 場合のみ実行する。GEMM を 1 回実行してプールへ確保させたのち
/// `release_cached_memory` で解放し、`cached_bytes == 0` を確認する。
///
/// `#[ignore]`（実機依存。`.claude/rules/coding-rust.md`）。実行例:
/// `cargo test -p fandhe-ai --test memory_pool_api -- --ignored`
#[test]
#[ignore]
fn release_cached_memory_on_cuda_drains_pool_after_gemm() {
    use fandhe_ai_backend_cuda::CudaDeviceProvider;
    use fandhe_ai_tensor_core::Tensor;
    use fandhe_ai_tensor_core::device::DeviceProvider;

    if !CudaDeviceProvider::new().is_available() {
        eprintln!("CUDA デバイス未検出のためスキップ（実機依存テスト）");
        return;
    }

    let device = Device::Cuda(0);
    let tape = fandhe_ai::tape_for(device).expect("CUDA tape_for は実機があれば成功する");
    let a = Tensor::new(vec![1.0f32; 4], &[2, 2]).expect("shape 一致");
    let b = Tensor::new(vec![1.0f32; 4], &[2, 2]).expect("shape 一致");
    let a_var = tape.var(&a);
    let b_var = tape.var(&b);
    let _ = a_var
        .matmul(&b_var)
        .expect("GEMM は shape 一致で成功する（プールへ確保が発生する）");

    // `a_var`／`b_var`／matmul 結果はここで drop 済み（スコープ終端で
    // 明示的に処理する必要はない）。`release_cached_memory` はプールが
    // アイドル保持している分（返却済みバッファ）を実解放する。
    fandhe_ai::release_cached_memory(device).expect("release_cached_memory は成功する");

    let stats = fandhe_ai::memory_pool_stats(device)
        .expect("memory_pool_stats は成功する")
        .expect("CUDA は Some(PoolStats) を返す");
    assert_eq!(
        stats.cached_bytes, 0,
        "release_cached_memory 後はアイドル保持バイト数が 0 のはず"
    );
}
