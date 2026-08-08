//! `MetalMemory`（TASK-1.9b・#45）に対する実機ロードテスト。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。CI（self-hosted・Linux）では
//! `#![cfg(target_os = "macos")]` によりコンパイル対象外になり、`#[ignore]` により通常の
//! `cargo test` からも除外される（実機依存テストの分離。`.claude/rules/coding-rust.md`。
//! `tests/device_smoke.rs` と同じ構成）。実行するには macOS 実機で以下を叩く:
//!
//! ```sh
//! cargo test -p backend-metal --test memory_roundtrip -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use backend_metal::{MetalContext, MetalMemory};
use tensor_core::Tensor;
use tensor_core::buffer::MemoryOps;
use tensor_core::device::Device;
use tensor_core::memory_stats::MemoryStats;
use tensor_core::pool::{PoolConfig, PooledMemory};

/// upload → download の roundtrip が bit 完全一致することを確認する
/// （受け入れ条件「確保・転送がリークなく動作する」の数値面の裏付け。
/// tolerance は使わない。`.claude/rules/coding-rust.md`）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn upload_download_roundtrip_is_bit_exact() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let mem = MetalMemory::new(ctx);

    let data: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.25 - 50.0).collect();
    let tensor = Tensor::<f32>::new(data.clone(), &[32, 32]).unwrap();

    let buf = mem.upload(&tensor).expect("upload は成功するはず");
    let back = mem.download(&buf).expect("download は成功するはず");

    for i in 0..32 {
        for j in 0..32 {
            let expected = data[i * 32 + j];
            let actual = back.get(&[i, j]).unwrap();
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "roundtrip は bit 完全一致するはず（[{i}, {j}]）"
            );
        }
    }
}

/// `alloc_zeroed` が全 0 のバッファを返すことを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn alloc_zeroed_returns_all_zero() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let mem = MetalMemory::new(ctx);

    let buf = mem
        .alloc_zeroed(&[16, 16])
        .expect("alloc_zeroed は成功するはず");
    let tensor = mem.download(&buf).expect("download は成功するはず");
    for i in 0..16 {
        for j in 0..16 {
            assert_eq!(tensor.get(&[i, j]).unwrap(), 0.0);
        }
    }
}

/// 空テンソル（numel == 0）が FFI を経由せず roundtrip することを確認
/// する（`tensor_core::buffer` モジュールコメント「空テンソルの契約」。
/// `MetalBuffer::new_with_data`/`new_zeroed` は長さ 0 を
/// `MetalError::ZeroLengthAllocation` として拒否するため、`MetalMemory`
/// 側の早期 return が正しく機能していないとこのテストは Err で失敗する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn zero_numel_tensor_roundtrips_without_touching_ffi() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let mem = MetalMemory::new(ctx);

    let empty = Tensor::<f32>::zeros(&[0, 4]).unwrap();
    let buf = mem
        .upload(&empty)
        .expect("空テンソルの upload は FFI を経由せず成功するはず");
    assert!(buf.is_empty());

    let back = mem
        .download(&buf)
        .expect("空バッファの download は成功するはず");
    assert!(back.is_empty());
    assert_eq!(back.shape(), &[0, 4]);
}

/// 反復確保・解放（16MiB 級バッファ × 100 回）でメモリが枯渇しないことを
/// 確認する（受け入れ条件「リークなく動作する」の直接検証。`MetalBuffer`
/// が内部で保持する `Retained<MTLBuffer>` の `Drop` が正しく解放して
/// いれば、毎回同サイズの確保を繰り返しても失敗しない）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn repeated_alloc_drop_cycles_do_not_leak_device_memory() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let mem = MetalMemory::new(ctx);

    // 16MiB / 4 bytes(f32) = 4,194,304 要素。
    let numel = 4 * 1024 * 1024;

    for _ in 0..100 {
        let buf = mem
            .alloc_zeroed(&[numel])
            .expect("ループ内の alloc_zeroed は成功するはず");
        drop(buf);
    }

    let final_buf = mem
        .alloc_zeroed(&[numel])
        .expect("100 回の確保・解放サイクル後も確保が成功するはず（リークなし）");
    drop(final_buf);
}

/// TASK-#201（REQ-14 14-3）: `PooledMemory<MetalMemory>` 経由の確保・再利用が
/// 実機（`MetalBuffer::zero_fill` を含む `PoolZeroFill::zero_fill`）で
/// 正しく動作し、再利用バッファが全 0 で観測できることを確認する
/// （`PoolZeroFill for MetalMemory` の受け入れ条件）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn pooled_memory_reuses_buffer_and_zero_fills() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let inner = MetalMemory::new(ctx);
    let mem = PooledMemory::new(inner, Device::Metal, PoolConfig::default());

    let numel = 256;
    let buf1 = mem
        .alloc_zeroed(&[numel])
        .expect("1 本目の alloc_zeroed は成功するはず");
    drop(buf1); // プールへ返却

    // 同一サイズの再確保でプールから再利用される。ゼロ初期化契約が
    // `MetalBuffer::zero_fill` 経由で維持されていることを確認する。
    let buf2 = mem
        .alloc_zeroed(&[numel])
        .expect("再利用時の alloc_zeroed は成功するはず");
    let tensor = mem.download(&buf2).expect("download は成功するはず");
    for i in 0..numel {
        assert_eq!(
            tensor.get(&[i]).unwrap(),
            0.0,
            "再利用バッファは zero_fill によって全要素 0 であるはず（index {i}）"
        );
    }
    drop(buf2);
}

/// TASK-14.1b（#175）: 受け入れ条件「Metal バックエンドでピーク値が
/// 取得できる」の実機検証。既知サイズ（要素数 `32 * 32` の rank-1 shape
/// `[1024]`、f32 = 4,096 バイト）を複数同時確保し
/// `peak_allocated_bytes()` が期待合計と一致すること、
/// drop 後は `allocated_bytes()` が減少しつつ `peak` が過去最大値を保持
/// することを確認する（`backend-cuda::tests::memory_real_device` の同名
/// テストと同種のシナリオを実機で裏付ける）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn peak_allocated_bytes_matches_known_allocation_size() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let mem = MetalMemory::new(ctx);
    mem.reset_peak();

    let numel = 32 * 32;
    let expected_bytes = (numel * std::mem::size_of::<f32>()) as u64;

    let a = mem
        .alloc_zeroed(&[numel])
        .expect("1 本目の alloc_zeroed は成功するはず");
    let b = mem
        .alloc_zeroed(&[numel])
        .expect("2 本目の alloc_zeroed は成功するはず");

    assert_eq!(mem.allocated_bytes(), expected_bytes * 2);
    assert_eq!(mem.peak_allocated_bytes(), expected_bytes * 2);

    drop(a);
    assert_eq!(
        mem.allocated_bytes(),
        expected_bytes,
        "1 本解放後は current が半分に戻るはず"
    );
    assert_eq!(
        mem.peak_allocated_bytes(),
        expected_bytes * 2,
        "peak は解放後も同時生存時の合計を保持するはず"
    );

    drop(b);
}
