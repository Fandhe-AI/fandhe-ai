//! 受け入れ条件「シード固定で結果が再現する」（イシュー #28）の統合テスト。
//!
//! CI（self-hosted・Linux）で実行可能な CPU 経路の決定性検証と、実機
//! （DGX Spark GB10・Metal 実機）に依存する `SyncPoint` 実装の疎通確認を
//! `#[ignore]` 分離で提供する（`.claude/rules/coding-rust.md`）。

use bench_harness::rng::Xorshift64Star;
use bench_harness::sync::{CpuSync, SyncPoint};

/// 同一シードから生成した入力で「小さな CPU 計算」を 2 回実行し、
/// 結果がビット一致することを確認する。ガードレール（REQ-4）の
/// フレーキーテスト対策・偽陽性防止の基盤となる決定性を担保する
/// （spec 根拠: PoC-2 発見事項 0、`docs/spec/04-requirements.md:117,290`）。
#[test]
fn same_seed_reproduces_identical_computation_result() {
    fn generate_and_dot(seed: u64) -> f32 {
        let mut rng = Xorshift64Star::new(seed);
        let a = rng.fill_vec(64);
        let b = rng.fill_vec(64);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x.mul_add(*y, 0.0))
            .sum()
    }

    let first = generate_and_dot(2026);
    let second = generate_and_dot(2026);
    assert_eq!(
        first.to_bits(),
        second.to_bits(),
        "同一シードから生成した入力の計算結果はビット一致するはずが不一致だった"
    );
}

/// CPU バックエンドは同期方式統一（REQ-8）上「該当なし」の no-op であることを
/// 統合テストとしても確認する（`sync.rs` のユニットテストと合わせた冗長確認）。
#[test]
fn cpu_sync_wait_idle_always_succeeds() {
    assert!(CpuSync.wait_idle().is_ok());
}

/// CUDA `stream.synchronize()` によるホスト転送を伴わない完了待ちを、
/// DGX Spark GB10 実機で疎通確認する（`make test-ignored` で実行）。
/// 通常 CI（self-hosted・Linux、CUDA 非搭載）では実行しない
/// （`.claude/rules/ci.md` 実機依存節）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。make test-ignored で実行する"]
fn cuda_stream_sync_wait_idle_succeeds_on_real_device() {
    use bench_harness::sync::CudaStreamSync;
    use std::sync::Arc;

    let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA context の生成に失敗しました");
    let stream = ctx.default_stream();
    let sync = CudaStreamSync::new(Arc::clone(&stream));
    assert!(sync.wait_idle().is_ok());
}
