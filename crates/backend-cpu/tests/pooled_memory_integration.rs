//! TASK-#201（REQ-14 14-3）の受け入れ条件「上限超過時に自動破棄され、
//! ピーク計測 API（TASK-14.1）に反映される」を `backend-cpu::CpuMemory` +
//! `tensor_core::pool::PooledMemory` の組合せで直接検証する統合テスト。
//!
//! `crates/tensor-core/src/pool.rs` の単体テストはモック `MemoryOps` で
//! プールコアのロジック（バケット分離・LRU 破棄・パススルー分岐）を検証
//! 済みだが、本テストは実バックエンド（`CpuMemory`）を通すことで、
//! `PoolZeroFill` 実装（`crates/backend-cpu/src/memory.rs`）と
//! `memory_stats::AllocationTracker` の計測反映（PR #359・TASK-14.1a）が
//! 実際に噛み合うことを確認する（受け入れ条件そのものの直接検証）。

use backend_cpu::CpuMemory;
use tensor_core::device::Device;
use tensor_core::memory_stats::MemoryStats;
use tensor_core::pool::{PoolConfig, PooledMemory};
use tensor_core::{DeviceBuffer, MemoryOps};

/// alloc → drop 後もプール保持分が `MemoryStats::allocated_bytes` に
/// 計上され続けることを検証する（design §2 point 8「計測反映」）。
#[test]
fn pooled_allocation_stays_counted_while_idle_in_pool() {
    let mem = PooledMemory::new(CpuMemory::new(), Device::Cpu, PoolConfig::default());

    assert_eq!(mem.allocated_bytes(), 0);
    let buf = mem.alloc_zeroed(&[1024]).unwrap(); // 4096 バイト
    assert_eq!(mem.allocated_bytes(), 4096);

    drop(buf); // プールへ返却: CpuBufferHandle（と内部の TrackedAllocation）は生存し続ける
    assert_eq!(
        mem.allocated_bytes(),
        4096,
        "プール保持中はハンドルが生存しているため allocated_bytes は減らないはず"
    );
    assert_eq!(mem.peak_allocated_bytes(), 4096);
}

/// #201 受け入れ条件そのものの直接検証: 上限超過時に自動破棄され、
/// `allocated_bytes`（ピーク計測 API）に反映される。
#[test]
fn exceeding_pool_limit_evicts_and_reduces_allocated_bytes() {
    // 4096 バイト（[1024] 要素）を 1 本だけ保持できる上限。
    let config = PoolConfig {
        max_pool_bytes: 4096,
    };
    let mem = PooledMemory::new(CpuMemory::new(), Device::Cpu, config);

    let buf1 = mem.alloc_zeroed(&[1024]).unwrap(); // 4096 バイト
    drop(buf1); // プールへ返却: pooled_bytes = 4096（上限ちょうど）
    assert_eq!(mem.pooled_bytes(), 4096);
    assert_eq!(
        mem.allocated_bytes(),
        4096,
        "上限ちょうどの間はまだ破棄されず生存しているはず"
    );

    // もう 1 本（別バケット: [2048] 要素 = 8192 バイト）を確保・返却すると
    // pooled_bytes 合計が 4096 + 8192 = 12288 > 4096 となり、
    // 最古（buf1 由来の 4096 バイトエントリ）が LRU 破棄される。
    let buf2 = mem.alloc_zeroed(&[2048]).unwrap();
    drop(buf2);

    assert!(
        mem.pooled_bytes() <= 4096,
        "上限超過後は pooled_bytes が max_pool_bytes 以下になるはず（実測 {}）",
        mem.pooled_bytes()
    );
    assert_eq!(
        mem.allocated_bytes(),
        4096,
        "LRU 破棄により最古エントリ（4096 バイト）の TrackedAllocation が drop され、\
         allocated_bytes へ即座に反映されるはず（#201 受け入れ条件の直接検証）"
    );
    assert_eq!(
        mem.peak_allocated_bytes(),
        12288,
        "peak は buf1（プール保持中）と buf2（確保直後）が同時生存した \
         瞬間の合計 4096 + 8192 を記録し、LRU 破棄後も過去最大値として \
         保持され続けるはず（`memory_stats::AllocationTracker` の契約）"
    );
}

/// 再利用されたバッファが全要素 0 で初期化されていることを検証する
/// （`alloc_zeroed` の「全要素 0」契約を再利用時にも維持する。
/// `PoolZeroFill for CpuMemory` の受け入れ条件）。
#[test]
fn reused_buffer_is_zero_filled() {
    let mem = PooledMemory::new(CpuMemory::new(), Device::Cpu, PoolConfig::default());

    // 1 本目を確保し、非ゼロ値を書き込んでからプールへ返却する。
    // `MemoryOps::download` を経由して内容を確認する（`upload` は
    // パススルーのためプール経由バッファには使わない。§2 point 7）。
    let buf1 = mem.alloc_zeroed(&[8]).unwrap();
    let downloaded = mem.download(&buf1).unwrap();
    assert_eq!(downloaded.as_slice().unwrap(), &[0.0f32; 8]);
    drop(buf1);

    // 同一サイズを再確保するとプールから再利用される。ゼロ初期化契約が
    // 維持されていることを `download` で確認する。
    let buf2 = mem.alloc_zeroed(&[8]).unwrap();
    let downloaded2 = mem.download(&buf2).unwrap();
    assert_eq!(
        downloaded2.as_slice().unwrap(),
        &[0.0f32; 8],
        "再利用バッファも zero_fill によって全要素 0 であるはず"
    );
    drop(buf2);
}

/// 透過ダウンキャスト（`PooledBufferHandle::as_any` が内部ハンドルへ転送
/// する設計。`pool.rs` モジュールコメント参照）が `CpuMemory::download` の
/// 具体的な `downcast_handle::<CpuBufferHandle>()` 呼び出しと噛み合う
/// ことを、`DeviceBuffer` 経由の往復（roundtrip）で検証する。
#[test]
fn pooled_buffer_roundtrips_through_download() {
    let mem = PooledMemory::new(CpuMemory::new(), Device::Cpu, PoolConfig::default());
    let buf: DeviceBuffer<f32> = mem.alloc_zeroed(&[4]).unwrap();
    let tensor = mem.download(&buf).unwrap();
    assert_eq!(tensor.shape(), &[4]);
    assert_eq!(tensor.as_slice().unwrap(), &[0.0f32; 4]);
}

/// #202 受け入れ条件そのものの直接検証:
/// 「解放 API 後にピークが理論値近傍へ戻る」（`docs/memory-pool-design.md`・
/// REQ-14 14-3）を `CpuMemory` 実バックエンドで確認する。
///
/// `peak_allocated_bytes` は `AllocationTracker` の契約上、単調増加の
/// high-water mark であり解放だけでは下がらない
/// （`crates/tensor-core/src/memory_stats.rs` の `MemoryStats::peak_allocated_bytes`
/// ドキュメンテーションコメント参照）。そのため検証は 2 段構成にする:
///
/// 1. **主張明（プールが実際に空になったことの直接証拠）**:
///    `release_all_pooled()` 直後の `allocated_bytes() == 0`・
///    `pooled_bytes() == 0` を主アサーションとする。これは
///    `reset_peak()` を経由しない、メモリが実際に戻ったことそのものの
///    証拠である。
/// 2. **副次的な実証（ピーク自体の回復）**: `peak_allocated_bytes` は解放
///    だけでは下がらないため、「ピークが理論値へ戻る」ことを観測するには
///    `reset_peak()`（peak を現在値へ再基準化。`memory_stats.rs` の
///    `reset_peak` 契約）を挟んだ新区間での再計測が構造的に必要となる。
///    再基準化後に代表ワークロード（A・B・C の 3 バッファ）を確保し、
///    ピークが理論最小ワーキングセット（3 バッファの合計）と一致し、
///    REQ-14 14-3 の係数上限（2 倍以内。緩和はユーザー承認必須。
///    `.claude/rules/coding-rust.md`）を満たすことを確認する。
#[test]
fn release_api_restores_peak_to_theoretical_working_set() {
    let mem = PooledMemory::new(CpuMemory::new(), Device::Cpu, PoolConfig::default());

    // 1. プール蓄積状態を作る: 異サイズの GEMM 様パターンを複数反復し、
    //    バケットが分かれてアイドル保持が理論ワーキングセットを大きく
    //    超えて蓄積することを事前に確認する。
    for k in 1..=4usize {
        let a = mem.alloc_zeroed(&[64 * k]).unwrap();
        let b = mem.alloc_zeroed(&[128 * k]).unwrap();
        let c = mem.alloc_zeroed(&[32 * k]).unwrap();
        drop(a);
        drop(b);
        drop(c);
    }
    let accumulated_pooled = mem.pooled_bytes();
    // 理論最小ワーキングセット（k=1 の A・B・C 合計。後段で使う代表値）。
    let theoretical_working_set: u64 = (64 + 128 + 32) * 4; // f32 4 バイト
    assert!(
        accumulated_pooled > theoretical_working_set * 2,
        "事前条件: プール蓄積が理論ワーキングセットの 2 倍を大きく超えている \
         はず（実測 pooled_bytes={accumulated_pooled}, 理論値={theoretical_working_set}）"
    );

    // 2. 主アサーション: release_all_pooled 後は実際にメモリが戻っている
    //    （reset_peak を経由しない直接証拠）。
    let freed = mem.release_all_pooled();
    assert_eq!(freed, accumulated_pooled);
    assert_eq!(
        mem.allocated_bytes(),
        0,
        "release_all_pooled 後は allocated_bytes が 0 になるはず"
    );
    assert_eq!(mem.pooled_bytes(), 0);

    // 3. 副次的な実証: reset_peak で新しい計測区間を区切り、代表ワーク
    //    ロード 1 回分（A・B・C）を確保してピークが理論値近傍へ戻ることを
    //    確認する（`peak_allocated_bytes` の単調増加契約により、reset を
    //    挟まないと過去の蓄積ピークが残ったままになるため必須の手順）。
    mem.reset_peak();
    assert_eq!(
        mem.peak_allocated_bytes(),
        0,
        "reset_peak 直後は allocated_bytes（解放済みで 0）へ再基準化されるはず"
    );

    let a = mem.alloc_zeroed(&[64]).unwrap();
    let b = mem.alloc_zeroed(&[128]).unwrap();
    let c = mem.alloc_zeroed(&[32]).unwrap();
    let peak_after = mem.peak_allocated_bytes();
    assert_eq!(
        peak_after, theoretical_working_set,
        "CPU 経路は決定的なため、代表ワークロード 1 回分のピークは理論値と \
         一致するはず（実測 {peak_after}, 理論値 {theoretical_working_set}）"
    );
    assert!(
        peak_after <= theoretical_working_set * 2,
        "REQ-14 14-3 の係数上限（2 倍以内）を満たすはず（実測 {peak_after}, \
         理論値 {theoretical_working_set}。この係数は緩和禁止＝ユーザー承認必須）"
    );
    drop(a);
    drop(b);
    drop(c);
}

/// 同一 shape の反復ワークロード（プール再利用が支配的なケース）で、
/// ピークが理論最小ワーキングセットの 2 倍以内に収まり続けることを検証する
/// （REQ-14 14-3 の係数維持回帰。`PoolConfig::default` の 128MiB 上限設計が
/// 意図どおり機能することの確認）。
#[test]
fn coefficient_stays_within_2x_for_repeated_same_shape_workload() {
    let mem = PooledMemory::new(CpuMemory::new(), Device::Cpu, PoolConfig::default());
    let theoretical_working_set: u64 = (64 + 128 + 32) * 4; // A・B・C 合計バイト数

    mem.reset_peak();
    for _ in 0..8 {
        let a = mem.alloc_zeroed(&[64]).unwrap();
        let b = mem.alloc_zeroed(&[128]).unwrap();
        let c = mem.alloc_zeroed(&[32]).unwrap();
        drop(a);
        drop(b);
        drop(c);
    }
    let peak = mem.peak_allocated_bytes();
    assert!(
        peak <= theoretical_working_set * 2,
        "同一 shape 反復では係数 2 倍以内に収まり続けるはず（実測 {peak}, \
         理論値 {theoretical_working_set}。係数は緩和禁止＝ユーザー承認必須）"
    );

    mem.release_all_pooled();
}
