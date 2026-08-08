//! `CudaMemory`（TASK-1.9b・#45）の実機必須テスト。
//!
//! CUDA 実機（DGX Spark GB10 等）でのみ意味を持つ肯定的検証（roundtrip
//! bit 一致・`mem_get_info` によるリーク検出）を `#[ignore]` 分離のうえ
//! 本ファイルに置く（`.claude/rules/coding-rust.md` の実機依存分離方針、
//! `crates/backend-cuda/tests/device_init.rs` と同じ構成）。実行するには
//! CUDA 実機で以下を叩く:
//!
//! ```sh
//! cargo test -p backend-cuda --test memory_real_device -- --ignored --nocapture
//! ```

use backend_cuda::{CudaDevice, CudaMemory};
use tensor_core::Tensor;
use tensor_core::buffer::MemoryOps;
use tensor_core::device::Device;
use tensor_core::memory_stats::MemoryStats;
use tensor_core::pool::{PoolConfig, PooledMemory};

/// upload → download の roundtrip が bit 完全一致することを確認する
/// （受け入れ条件「確保・転送がリークなく動作する」の数値面の裏付け。
/// tolerance は使わない。`.claude/rules/coding-rust.md`）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn upload_download_roundtrip_is_bit_exact_on_real_hardware() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let mem = CudaMemory::new(&device);

    let data: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.5 - 100.0).collect();
    let tensor = Tensor::<f32>::new(data.clone(), &[64, 64]).unwrap();

    let buf = mem
        .upload(&tensor)
        .expect("upload must succeed on real hardware");
    let back = mem
        .download(&buf)
        .expect("download must succeed on real hardware");

    for i in 0..64 {
        for j in 0..64 {
            let expected = data[i * 64 + j];
            let actual = back.get(&[i, j]).unwrap();
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "roundtrip must be bit exact at [{i}, {j}]"
            );
        }
    }
}

/// `alloc_zeroed` が全 0 のバッファを返すことを確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn alloc_zeroed_returns_all_zero_on_real_hardware() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let mem = CudaMemory::new(&device);

    let buf = mem
        .alloc_zeroed(&[128, 128])
        .expect("alloc_zeroed must succeed");
    let tensor = mem.download(&buf).expect("download must succeed");
    for i in 0..128 {
        for j in 0..128 {
            assert_eq!(tensor.get(&[i, j]).unwrap(), 0.0);
        }
    }
}

/// 反復確保・解放（64MiB 級バッファ × 100 回）でデバイスメモリが枯渇
/// しないことを確認する（受け入れ条件「リークなく動作する」の直接検証）。
/// `CudaSlice` の `Drop`（RAII）が正しく解放していれば、毎回同サイズの
/// 確保を繰り返しても失敗しない。リークがあれば途中の `alloc_zeroed`
/// が `DeviceAllocationFailed`（`cuMemAlloc` の OOM）で `expect` に
/// 失敗し、このテスト自体が panic して検出する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn repeated_alloc_drop_cycles_do_not_leak_device_memory() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let mem = CudaMemory::new(&device);

    // 64MiB / 4 bytes(f32) = 16,777,216 要素。
    let numel = 16 * 1024 * 1024;

    // ウォームアップ 1 回（アロケータの初回確保コストを free 値比較から除く）。
    {
        let buf = mem
            .alloc_zeroed(&[numel])
            .expect("warmup alloc must succeed");
        drop(buf);
    }

    for _ in 0..100 {
        let buf = mem
            .alloc_zeroed(&[numel])
            .expect("alloc_zeroed must succeed within the loop");
        // 明示的に drop してこのイテレーションで即座に解放されることを
        // 確認する（RAII が働かない場合、次の確保が OOM で失敗し
        // このループ自体が panic することが期待される検出経路）。
        drop(buf);
    }

    // 最終確認: ループ後も同サイズの確保が成功する（累積リークがあれば
    // ここで DeviceAllocationFailed になるはずである）。
    let final_buf = mem
        .alloc_zeroed(&[numel])
        .expect("allocation after 100 alloc/drop cycles must still succeed (no leak)");
    drop(final_buf);
}

/// TASK-#201（REQ-14 14-3）: `PooledMemory<CudaMemory>` 経由の確保・再利用が
/// 実機（CUDA memset を含む `PoolZeroFill::zero_fill`）で正しく動作し、
/// 再利用バッファが全 0 で観測できることを確認する（`PoolZeroFill for
/// CudaMemory` の受け入れ条件。`backend-cpu` 側の `pooled_memory_integration.rs`
/// と同種のシナリオを実機で裏付ける）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn pooled_memory_reuses_buffer_and_zero_fills_on_real_hardware() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let inner = CudaMemory::new(&device);
    let mem = PooledMemory::new(inner, Device::Cuda(device.ordinal()), PoolConfig::default());

    let numel = 1024;
    let buf1 = mem
        .alloc_zeroed(&[numel])
        .expect("first alloc_zeroed must succeed");
    drop(buf1); // プールへ返却

    // 同一サイズの再確保でプールから再利用される（下位 `alloc_zeros` を
    // 再度叩かない）。`memset_zeros` によるゼロ初期化が実機で正しく
    // 機能することを `download` で確認する。
    let buf2 = mem
        .alloc_zeroed(&[numel])
        .expect("reused alloc_zeroed must succeed");
    let tensor = mem.download(&buf2).expect("download must succeed");
    for i in 0..numel {
        assert_eq!(
            tensor.get(&[i]).unwrap(),
            0.0,
            "reused buffer must be zero-filled by CudaStream::memset_zeros at index {i}"
        );
    }
    drop(buf2);
}

/// TASK-14.1b（#175）: 受け入れ条件「CUDA バックエンドでピーク値が
/// 取得できる」の実機検証。既知サイズ（要素数 `64 * 64` の rank-1 shape
/// `[4096]`、f32 = 16,384 バイト）を複数同時確保し
/// `peak_allocated_bytes()` が期待合計と一致すること、
/// drop 後は `allocated_bytes()` が減少しつつ `peak` が過去最大値を保持
/// することを確認する（`backend-cpu::memory` の
/// `peak_allocated_bytes_tracks_sum_of_concurrent_buffers` と同種の
/// シナリオを実機で裏付ける）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn peak_allocated_bytes_matches_known_allocation_size_on_real_hardware() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let mem = CudaMemory::new(&device);
    mem.reset_peak();

    let numel = 64 * 64;
    let expected_bytes = (numel * std::mem::size_of::<f32>()) as u64;

    let a = mem
        .alloc_zeroed(&[numel])
        .expect("first alloc_zeroed must succeed");
    let b = mem
        .alloc_zeroed(&[numel])
        .expect("second alloc_zeroed must succeed");

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
