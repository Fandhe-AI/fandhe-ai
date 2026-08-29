//! `fandhe_ai::release_cached_memory`／`fandhe_ai::memory_pool_stats`
//! （REQ-14 の明示解放 API。イシュー #1018 ツリー・#1021）の受入テスト。
//!
//! CPU 経路（プールを持たない既定契約）は Linux CI でも実行できる。
//! Metal 経路（実際のプール保持・解放の裏付け）は Apple Silicon 実機
//! 依存のため `#[ignore]` で分離する（`.claude/rules/coding-rust.md`）。

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
