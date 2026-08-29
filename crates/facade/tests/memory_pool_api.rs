//! `fandhe_ai::release_cached_memory`／`memory_pool_stats`（イシュー
//! #1020・REQ-14）の結線検証。
//!
//! CPU は常に「プールを持たないバックエンド」（`BackendOps` の既定
//! メソッド。`crates/tensor-core/src/backend_ops.rs`）であるため、
//! `release_cached_memory(Device::Cpu) == Ok(())`・
//! `memory_pool_stats(Device::Cpu) == Ok(None)` を CI（GitHub ホステッド。
//! CUDA 実機不要）で固定する。
//!
//! CUDA 経路は `tests/tape_construction.rs` と同じ「実行環境適応型」
//! 方針（`CudaDeviceProvider::is_available()` で選択可能デバイスの有無を
//! 判定してから分岐）に従い、実機がある場合のみ確保→drop→解放→
//! `cached_bytes == 0` を検証する `#[ignore]` テストとする（実機依存を
//! 通常 CI ジョブへ持ち込まない。`.claude/rules/coding-rust.md`）。

use fandhe_ai::Device;

/// CPU（プール未接続バックエンド）での `release_cached_memory` は常に
/// `Ok(())`（no-op）。
#[test]
fn release_cached_memory_on_cpu_is_ok_noop() {
    // `BackendError` は `#[non_exhaustive]`・`PartialEq` 非実装（拡張性
    // 優先の設計。`device.rs` 冒頭コメント参照）のため `assert_eq!` では
    // なく `assert!(matches!(..))` で判定する。
    assert!(matches!(
        fandhe_ai::release_cached_memory(Device::Cpu),
        Ok(())
    ));
}

/// CPU（プール未接続バックエンド）での `memory_pool_stats` は常に
/// `Ok(None)`（`BackendOps::device_memory_pool_stats` の既定実装）。
#[test]
fn memory_pool_stats_on_cpu_is_ok_none() {
    assert!(matches!(
        fandhe_ai::memory_pool_stats(Device::Cpu),
        Ok(None)
    ));
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
