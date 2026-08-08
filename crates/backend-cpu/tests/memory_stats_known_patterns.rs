//! TASK-14.1c（#176・REQ-14）: 「既知サイズのテンソルを N 個確保した場合の
//! 期待値」をバックエンド共通シグネチャ（`MemoryOps` + `MemoryStats`）経由
//! で検証する統合テスト。spec の受け入れ基準の文言（`docs/spec/05-tasks.md`
//! TASK-14.1）どおり、`tensor-core` 側（`memory_stats_known_patterns.rs`）が
//! トラッカー層を直接駆動するのに対し、本ファイルは `CpuMemory::alloc_zeroed`／
//! `upload` を経由した「テンソル確保」単位で期待値を検証する。
//!
//! 期待バイト数は `numel * size_of::<f32>()`（= numel * 4）から解析的に
//! 導出する（`crates/backend-cpu/src/memory.rs` の `checked_byte_len` と
//! 同じ換算式）。
//!
//! サイズクラス別プール（`tensor_core::pool::PooledMemory`）経由の計測
//! 反映は `pooled_memory_integration.rs`（#201）が検証済みのため、本ファイル
//! はプールを介さない素の `CpuMemory` 確保パターンに限定し重複実装しない。
//! CUDA/Metal（#175・TASK-14.1b は origin/main 未マージ）の実機経路は対象外
//! とし、テストは `MemoryOps + MemoryStats` を実装する型に対する汎用
//! ヘルパ関数として書く（#175 マージ後に同パターンを流用しやすくするため）。
//!
//! 各テストは独立した `CpuMemory::new()` インスタンスを使う（並列テスト間
//! の計数混線を防ぐ。`memory_stats.rs` モジュールコメント「トラッカーの
//! 共有範囲」と同じ設計意図）。

use std::mem::size_of;

use backend_cpu::CpuMemory;
use tensor_core::memory_stats::MemoryStats;
use tensor_core::{MemoryOps, Tensor};

/// shape から `alloc_zeroed` の期待バイト数（numel * size_of::<f32>()）を
/// 導出する（`backend-cpu::memory::checked_byte_len` と同じ換算式を
/// テスト側で独立に再現し、実装とテストの偶然の一致に依存しないようにする）。
fn expected_bytes_for_shape(shape: &[usize]) -> u64 {
    let numel: usize = shape.iter().product();
    (numel * size_of::<f32>()) as u64
}

/// `MemoryOps + MemoryStats` を実装する型に対して `shape` を N 回
/// `alloc_zeroed` し、生存させたまま Vec に集めて返す汎用ヘルパ。
/// #175（CUDA/Metal）マージ後にジェネリック境界を広げて再利用できるように、
/// 呼び出し側の型を型パラメータとして受け取る形にしてある。
fn alloc_n<M: MemoryOps>(
    mem: &M,
    shape: &[usize],
    n: usize,
) -> Vec<tensor_core::DeviceBuffer<f32>> {
    (0..n).map(|_| mem.alloc_zeroed(shape).unwrap()).collect()
}

/// テスト 1: `alloc_zeroed` で `[1024]`（4096 バイト）を N=8 本同時生存させ、
/// `peak == allocated == N * 4096` となることを検証する。drop 後は
/// `current == 0`・`peak` は維持される。
#[test]
fn alloc_zeroed_n_concurrent_tensors_track_exact_total() {
    const N: usize = 8;
    let shape = [1024usize];
    let per_tensor = expected_bytes_for_shape(&shape);
    assert_eq!(per_tensor, 4096);

    let mem = CpuMemory::new();
    let bufs = alloc_n(&mem, &shape, N);

    assert_eq!(mem.allocated_bytes(), per_tensor * N as u64);
    assert_eq!(mem.peak_allocated_bytes(), per_tensor * N as u64);

    drop(bufs);
    assert_eq!(mem.allocated_bytes(), 0, "全 drop 後は current が 0 に戻る");
    assert_eq!(
        mem.peak_allocated_bytes(),
        per_tensor * N as u64,
        "peak は解放後も過去最大値を維持する"
    );
}

/// テスト 2: 多次元 shape 混在確保の期待値一致（バイト換算式そのものの検証）。
/// `[16, 16, 4]`（numel=1024 → 4096 バイト）と `[3, 5, 7]`（numel=105 →
/// 420 バイト）を同時確保し、合計値が解析的期待値と一致することを確認する。
#[test]
fn multi_dimensional_shapes_byte_conversion_matches_numel_times_four() {
    let shape_a = [16usize, 16, 4];
    let shape_b = [3usize, 5, 7];
    let expected_a = expected_bytes_for_shape(&shape_a);
    let expected_b = expected_bytes_for_shape(&shape_b);
    assert_eq!(expected_a, 4096);
    assert_eq!(expected_b, 420);

    let mem = CpuMemory::new();
    let buf_a = mem.alloc_zeroed(&shape_a).unwrap();
    assert_eq!(mem.allocated_bytes(), expected_a);

    let buf_b = mem.alloc_zeroed(&shape_b).unwrap();
    assert_eq!(mem.allocated_bytes(), expected_a + expected_b);
    assert_eq!(mem.peak_allocated_bytes(), expected_a + expected_b);

    drop(buf_a);
    drop(buf_b);
    assert_eq!(mem.allocated_bytes(), 0);
    assert_eq!(mem.peak_allocated_bytes(), expected_a + expected_b);
}

/// テスト 3: `upload` 経路の計上検証: contiguous・非 contiguous（transpose 後）の
/// 両方について、既知長 `Tensor` を N 本 upload した合計が期待バイト数と
/// 一致することを確認する。非 contiguous 側は「実体化後の numel 基準」で
/// あることを確認する（`memory.rs` の upload 実装は `contiguous()` してから
/// 転送するため、転置しても numel は不変で期待値は変わらない）。
#[test]
fn upload_contiguous_and_non_contiguous_tensors_match_expected_bytes() {
    let mem = CpuMemory::new();

    // contiguous 側: [2, 3] を 3 本 upload → numel=6 * size_of::<f32>() = 24
    // バイトずつ、合計 72 バイト。
    let contiguous_shape = [2usize, 3];
    let per_contiguous = expected_bytes_for_shape(&contiguous_shape);
    assert_eq!(per_contiguous, 24);

    let mut bufs = Vec::new();
    for _ in 0..3 {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &contiguous_shape).unwrap();
        bufs.push(mem.upload(&t).unwrap());
    }
    assert_eq!(mem.allocated_bytes(), per_contiguous * 3);

    // 非 contiguous 側: [2, 3] を transpose して [3, 2] の非 contiguous view
    // にしてから upload。実体化後も numel=6 は不変のため期待値は同じ 24
    // バイト（転置してもバイト数は変化しない契約であることの検証）。
    let base = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
    let transposed = base.transpose(0, 1).unwrap();
    assert!(
        !transposed.is_contiguous(),
        "transpose 直後は非 contiguous のはず"
    );

    let before = mem.allocated_bytes();
    let non_contig_buf = mem.upload(&transposed).unwrap();
    assert_eq!(
        mem.allocated_bytes(),
        before + per_contiguous,
        "非 contiguous でも実体化後の numel（6）基準で 24 バイト計上されるはず"
    );

    drop(bufs);
    drop(non_contig_buf);
    assert_eq!(mem.allocated_bytes(), 0);
    assert_eq!(mem.peak_allocated_bytes(), per_contiguous * 4);
}

/// テスト 4: 鋸歯パターン（テンソル版）: `[2048]`（8192 バイト）を確保 → drop を
/// 繰り返し、`peak == 8192`（同時生存が常に 1 本のため N によらず一定）を
/// 検証する。
#[test]
fn sawtooth_tensor_allocations_keep_peak_at_single_tensor_size() {
    const N: usize = 20;
    let shape = [2048usize];
    let expected = expected_bytes_for_shape(&shape);
    assert_eq!(expected, 8192);

    let mem = CpuMemory::new();
    for _ in 0..N {
        let buf = mem.alloc_zeroed(&shape).unwrap();
        assert_eq!(mem.allocated_bytes(), expected);
        drop(buf);
        assert_eq!(mem.allocated_bytes(), 0);
    }
    assert_eq!(mem.peak_allocated_bytes(), expected);
}

/// テスト 5: `clone()` によるトラッカー共有: 2 つの `CpuMemory` インスタンス
/// （`clone()` で複製）から交互に確保すると、双方の `allocated_bytes`／
/// `peak_allocated_bytes` が合算値を返すことを確認する（「同一計測系列への
/// 参照複製」契約。`memory.rs` モジュールコメント参照）。
#[test]
fn cloned_instances_report_the_combined_total_from_either_handle() {
    let mem_a = CpuMemory::new();
    let mem_b = mem_a.clone();

    let buf1 = mem_a.alloc_zeroed(&[100]).unwrap(); // 400 バイト
    assert_eq!(mem_a.allocated_bytes(), 400);
    assert_eq!(
        mem_b.allocated_bytes(),
        400,
        "clone 先からも同じ計測系列が見える"
    );

    let buf2 = mem_b.alloc_zeroed(&[100]).unwrap(); // 400 バイト
    assert_eq!(mem_a.allocated_bytes(), 800, "clone 元からも合算値が見える");
    assert_eq!(mem_b.allocated_bytes(), 800);
    assert_eq!(mem_a.peak_allocated_bytes(), 800);
    assert_eq!(mem_b.peak_allocated_bytes(), 800);

    drop(buf1);
    drop(buf2);
    assert_eq!(mem_a.allocated_bytes(), 0);
    assert_eq!(mem_b.allocated_bytes(), 0);
}

/// テスト 6: 独立インスタンスの分離: `CpuMemory::new()` を 2 つ作り片方だけ確保
/// すると、もう片方は 0 のまま（計測系列の独立性）を確認する。
#[test]
fn independent_instances_do_not_share_counters() {
    let mem_a = CpuMemory::new();
    let mem_b = CpuMemory::new();

    let buf = mem_a.alloc_zeroed(&[512]).unwrap(); // 2048 バイト
    assert_eq!(mem_a.allocated_bytes(), 2048);
    assert_eq!(
        mem_b.allocated_bytes(),
        0,
        "new() で作った別インスタンスは独立した計測系列を持つはず"
    );
    assert_eq!(mem_b.peak_allocated_bytes(), 0);

    drop(buf);
    assert_eq!(mem_a.allocated_bytes(), 0);
}

/// テスト 7: 0 要素テンソルの確保・upload が計数 no-op であることを確認する
/// （`[0, 3]` shape。`memory_stats.rs` モジュールコメント「0 バイト加算は
/// no-op」参照）。
#[test]
fn zero_element_tensor_allocation_and_upload_are_no_ops_for_counters() {
    let mem = CpuMemory::new();

    let buf = mem.alloc_zeroed(&[0, 3]).unwrap();
    assert_eq!(mem.allocated_bytes(), 0);
    assert_eq!(mem.peak_allocated_bytes(), 0);
    drop(buf);

    let empty = Tensor::<f32>::zeros(&[0, 3]).unwrap();
    let uploaded = mem.upload(&empty).unwrap();
    assert_eq!(mem.allocated_bytes(), 0);
    assert_eq!(mem.peak_allocated_bytes(), 0);
    drop(uploaded);
}

/// テスト 8: `reset_peak` を挟むテンソル確保区間: バックエンド API（`CpuMemory`）
/// 越しでも、区間 1 のピーク形成後に `reset_peak()` した区間 2 のピークが
/// 「reset 基点 + 区間 2 の増分」の期待値に一致することを確認する。
#[test]
fn reset_peak_across_tensor_allocation_windows_matches_expected_values() {
    let mem = CpuMemory::new();

    // 区間 1: [1024]（4096）+ [1024]（4096）→ current=8192 でピーク形成。
    let a = mem.alloc_zeroed(&[1024]).unwrap();
    let b = mem.alloc_zeroed(&[1024]).unwrap();
    assert_eq!(mem.allocated_bytes(), 8192);
    assert_eq!(mem.peak_allocated_bytes(), 8192);

    // b を解放して基点を 4096 に落とす。
    drop(b);
    assert_eq!(mem.allocated_bytes(), 4096);

    mem.reset_peak();
    let baseline = mem.allocated_bytes();
    assert_eq!(
        mem.peak_allocated_bytes(),
        baseline,
        "reset 直後は peak == current"
    );

    // 区間 2: [128]（512 バイト）を追加確保 → 期待ピークは baseline + 512。
    let c = mem.alloc_zeroed(&[128]).unwrap();
    assert_eq!(
        mem.peak_allocated_bytes(),
        baseline + 512,
        "区間 2 のピークは reset 基点 + 区間 2 の増分と一致するはず"
    );

    drop(a);
    drop(c);
}
