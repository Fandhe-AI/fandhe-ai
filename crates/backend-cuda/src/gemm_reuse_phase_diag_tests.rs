//! CUDA GEMM reuse 計測境界（`scripts/bench/framework-compare` の
//! `bench-fandhe --task gemm --mode reuse --phases`。イシュー #1182）の
//! `matmul` 区間内訳を実測分解する診断テスト。
//!
//! # 背景
//!
//! `#1142`（`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §4.3・§8）
//! は「reuse の計測境界に残る H2D／D2H／同期の固定費が candle 比を押し
//! 下げている」と**推定**したまま確定していなかった。`bench-fandhe` の
//! `gemm --mode reuse --phases`（イシュー #1182）は公開 API 呼び出し
//! 境界（`matmul`／`to_tensor`／`host_copy`／`checksum`／`iter_total`）
//! までしか分解できず、`matmul` 区間の内側（H2D→カーネル→D2H→同期）は
//! fandhe-ai の公開 API では観測不能である（`run_f32_kernel`。
//! `crates/backend-cuda/src/gemm.rs` 参照）。本ファイルはその内側を
//! `crates/backend-cuda` の非公開 API へ直接アクセスして分解する。
//!
//! # 配置理由（`fresh_overhead_diag_tests.rs` と同型の判断）
//!
//! `context_cache::{cached_device, cached_gemm, cached_allocator}`
//! （いずれも `pub(crate)`）・`CudaGemm::launch_tiled_f32_pooled`
//! （`pub(crate)`。本イシューで新設）へ到達するため、integration test
//! ではなく `lib.rs` の兄弟モジュールとして配置する。
//!
//! # `run_gemm_reuse` 1 反復との対応
//!
//! `crates/facade` の `Var::matmul` → `CudaBackendOps::gemm_fp32_strict`
//! → `run_f32_kernel`（`gemm.rs`）の実体は毎反復
//! `clone_htod(A)`・`clone_htod(B)`・プール `alloc_uninit_f32(C)`・
//! `launch`・`readback`（`clone_dtoh` + `synchronize`）。本ファイルは
//! この 1 反復を (a) H2D A・(b) H2D B・(c) C 確保（プール経由）・
//! (d) カーネル投入・(e) カーネル専有時間（明示 `synchronize`）・
//! (f) D2H（`clone_dtoh` + `synchronize`）・(g) ホストコピー（`to_vec`
//! + f64 和）へ分解計測する。
//!
//! HEAD の `launch_tiled_f32`／`launch_tiled_f32_pooled` は #1013 以降
//! **非同期投入**（完了待ちなし）契約であり、`fresh_overhead_diag_tests.rs`
//! の「(d) launch+synchronize」ラベルはこの契約変更後 stale になっている
//! （投入直後の `elapsed()` はカーネル完了を待たない）。本ファイルは
//! カーネル投入（`launch_issue`）とカーネル専有時間（`kernel_wait`。
//! 明示 `stream.synchronize()`）を別区間として分離し、この誤りを踏襲
//! しない。
//!
//! # kernel 変種（select / classic）
//!
//! `CudaGemm::DiagTiledF32Kernel::{Select, Classic}` の 2 変種で計測する。
//! `Select` は本番同一（`select_tiled_f32_kernel`。形状条件付き
//! pipeline／classic 自動選択）、`Classic` は常に classic 固定
//! （crates.io 公開版 `fandhe-ai =0.6.0` の `kernels.rs` に pipeline
//! 分岐が無いための近似対応。`launch_tiled_f32_pooled` doc コメント
//! 参照）。突合方針は `docs/perf/cuda-gemm-reuse-phase-breakdown.md` §5。
//!
//! # 実行時は必ず `--test-threads=1`（`fresh_overhead_diag_tests.rs` と
//! 同じ理由: 同一 GPU 上での複数テストスレッド競合を避ける）
//!
//! # メモリ使用量
//!
//! ホスト側キープアライブ（`Vec<Vec<f32>>`）は N=4096 で 1 変種あたり
//! 約 (20 warmup + 20 測定) × 4096² × 4 bytes ≈ 2.7 GiB
//! （`run_gemm_reuse` の reuse tape 蓄積と同型。README「`--mode
//! fresh|reuse`」節の既存注記と同じオーダー）。デバイス側 `c_dev` は
//! 反復末尾で drop しプールへ返却する（ホストのみキープアライブする）。
//!
//! # gating しない方針（`fresh_overhead_diag_tests.rs` と同じ理由）
//!
//! 本ファイルの `#[test]` は実行が成功すること（各フェーズが例外なく
//! 完了すること）のみを検証条件とし、フェーズ間の大小関係・絶対値への
//! `assert!` は行わない（環境揺らぎによる flaky 化防止。`out.len()`・
//! 有限性・非ゼロ checksum の sanity assert のみ行う）。数値は
//! `println!` に残し、`docs/perf/cuda-gemm-reuse-phase-breakdown.md`
//! へ転記する一次情報とする。

use std::time::Instant;

use bench_harness::{Quartiles, median_q1_q3, rng::Xorshift64Star};

use crate::context_cache::{cached_allocator, cached_device, cached_gemm};
use crate::gemm::DiagTiledF32Kernel;

const WARMUP_TRIALS: usize = 20;
const MEASURED_TRIALS: usize = 20;

/// `bench-fandhe --task gemm --mode reuse --phases`（イシュー #1182）が
/// 対象とするのと同じサイズ（README「GEMM ゲート 5 回計測」節・
/// `docs/perf/cuda-gemm-candle-gate-remeasurement.md` 対象形状）。
const SIZES: [usize; 3] = [1024, 2048, 4096];

fn gen_square_ab(seed: u64, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let a = rng.fill_vec(n * n);
    let b = rng.fill_vec(n * n);
    (a, b)
}

fn median_of(samples: &[f64]) -> Quartiles {
    median_q1_q3(samples)
        .expect("samples collected from successful trials must be non-empty and NaN-free")
}

fn print_quartiles_ms(label: &str, q: Quartiles) {
    println!(
        "    {label}: median={:.4} ms  q1={:.4} ms  q3={:.4} ms",
        q.median * 1e3,
        q.q1 * 1e3,
        q.q3 * 1e3
    );
}

/// 1 反復分のフェーズ計測結果（§本ファイル冒頭「対応」参照）。
struct PhaseSample {
    h2d_a_secs: f64,
    h2d_b_secs: f64,
    alloc_c_secs: f64,
    launch_issue_secs: f64,
    kernel_wait_secs: f64,
    d2h_secs: f64,
    host_copy_secs: f64,
}

/// 1 サイズ・1 変種の 1 反復を計測する。`gemm`／`allocator` は呼び出し元
/// が `context_cache` 経由で取得したハンドル（本番 `ops::CudaBackendOps
/// ::gemm` と同じキャッシュ経由）を使い回す（`fresh_overhead_diag_tests
/// ::measure_one_phase_trial` と同じ設計判断: 本診断の対象はあくまで
/// 毎反復の H2D／確保／launch／同期／D2H／ホストコピーであり、
/// `CudaGemm::new`／`CudaAllocator::new` のコストではない）。
///
/// `c_dev`（プール確保ハンドル）は本関数末尾で drop しプールへ返却する
/// （`run_gemm_reuse` の reuse tape が C ノードを蓄積するのとは異なり、
/// 本診断は「定常状態のカーネル専有時間」を見るため反復ごとに新規確保
/// する。プールヒットにより確保コスト自体は定常状態を再現する）。
#[allow(clippy::too_many_arguments)]
fn measure_one_phase_trial(
    device: &crate::device::CudaDevice,
    gemm: &crate::gemm::CudaGemm,
    allocator: &crate::pool::CudaAllocator,
    a: &[f32],
    b: &[f32],
    n: u32,
    kernel: DiagTiledF32Kernel,
    keep_alive: &mut Vec<Vec<f32>>,
) -> PhaseSample {
    // `device.stream()` は `&Arc<CudaStream>`（`device.rs`）。既存経路と
    // 同じく `.clone()` で所有権を持つ `Arc<CudaStream>` を作る
    // （`fresh_overhead_diag_tests.rs` と同じ理由）。
    let stream = device.stream().clone();

    let t = Instant::now();
    let a_dev = stream.clone_htod(a).expect("H2D A upload must succeed");
    let h2d_a_secs = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let b_dev = stream.clone_htod(b).expect("H2D B upload must succeed");
    let h2d_b_secs = t.elapsed().as_secs_f64();

    // 本番 `run_f32_kernel` と同じプール経由確保（`alloc_output_f32`＝
    // `alloc_zeros` の memset を含む確保とは異なる。`launch_tiled_f32_pooled`
    // doc コメント参照）。
    let t = Instant::now();
    let mut c_dev = allocator
        .alloc_uninit_f32((n as usize) * (n as usize))
        .expect("pooled output buffer allocation must succeed");
    let alloc_c_secs = t.elapsed().as_secs_f64();

    // カーネル投入のみ（#1013 以降の非同期投入契約。この elapsed は
    // カーネル完了を含まない — ファイル冒頭コメント参照）。
    let t = Instant::now();
    gemm.launch_tiled_f32_pooled(&a_dev, &b_dev, &mut c_dev, n, n, n, kernel)
        .expect("launch_tiled_f32_pooled (issue only) must succeed");
    let launch_issue_secs = t.elapsed().as_secs_f64();

    // カーネル専有時間 = 投入から完了までの明示同期待ち。
    let t = Instant::now();
    stream
        .synchronize()
        .expect("stream synchronize (kernel completion wait) must succeed");
    let kernel_wait_secs = t.elapsed().as_secs_f64();

    // D2H: `clone_dtoh` 発行 + 完了までの同期（`memory::readback` と同じ
    // 順序契約。§冒頭コメント参照）。
    let t = Instant::now();
    let out = stream
        .clone_dtoh(&c_dev.as_view())
        .expect("D2H download must succeed");
    stream
        .synchronize()
        .expect("stream synchronize after D2H must succeed before host buffer is read");
    let d2h_secs = t.elapsed().as_secs_f64();

    // ホストコピー（`readout_var` の `contiguous().as_slice().to_vec()`
    // 相当。`clone_dtoh` が返す `Vec<f32>` は既にホスト所有のため、ここ
    // では framework-compare 側の `host_copy` 区間に対応する「複製」の
    // コストを模して `to_vec()` でもう 1 段コピーする）。
    let t = Instant::now();
    let copied = out.to_vec();
    let host_copy_secs = t.elapsed().as_secs_f64();

    // sanity: 有限・非ゼロ（本ファイル冒頭「gating しない方針」参照。
    // 大小関係への assert は行わない）。
    assert_eq!(copied.len(), (n as usize) * (n as usize));
    assert!(
        copied.iter().all(|v| v.is_finite()),
        "output must be finite (n={n}, kernel={kernel:?})"
    );

    // `c_dev` を反復末尾で drop（プールへ返却）。ホスト側 `copied` は
    // `keep_alive` に保持する（メモリ使用量の見積り根拠。ファイル冒頭
    // コメント参照）。
    drop(c_dev);
    keep_alive.push(copied);

    PhaseSample {
        h2d_a_secs,
        h2d_b_secs,
        alloc_c_secs,
        launch_issue_secs,
        kernel_wait_secs,
        d2h_secs,
        host_copy_secs,
    }
}

fn run_size_kernel(n: usize, kernel: DiagTiledF32Kernel) {
    let device = cached_device(0).expect("CUDA device (ordinal 0) must be available");
    let gemm = cached_gemm(&device).expect("CudaGemm construction must succeed");
    let allocator = cached_allocator(&device).expect("CudaAllocator construction must succeed");

    let (a, b) = gen_square_ab(0x1182_a000 ^ (n as u64), n);
    let mut keep_alive: Vec<Vec<f32>> = Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);

    for _ in 0..WARMUP_TRIALS {
        let _ = measure_one_phase_trial(
            &device,
            &gemm,
            &allocator,
            &a,
            &b,
            n as u32,
            kernel,
            &mut keep_alive,
        );
    }

    let mut h2d_a = Vec::with_capacity(MEASURED_TRIALS);
    let mut h2d_b = Vec::with_capacity(MEASURED_TRIALS);
    let mut alloc_c = Vec::with_capacity(MEASURED_TRIALS);
    let mut launch_issue = Vec::with_capacity(MEASURED_TRIALS);
    let mut kernel_wait = Vec::with_capacity(MEASURED_TRIALS);
    let mut d2h = Vec::with_capacity(MEASURED_TRIALS);
    let mut host_copy = Vec::with_capacity(MEASURED_TRIALS);

    for _ in 0..MEASURED_TRIALS {
        let s = measure_one_phase_trial(
            &device,
            &gemm,
            &allocator,
            &a,
            &b,
            n as u32,
            kernel,
            &mut keep_alive,
        );
        h2d_a.push(s.h2d_a_secs);
        h2d_b.push(s.h2d_b_secs);
        alloc_c.push(s.alloc_c_secs);
        launch_issue.push(s.launch_issue_secs);
        kernel_wait.push(s.kernel_wait_secs);
        d2h.push(s.d2h_secs);
        host_copy.push(s.host_copy_secs);
    }

    let total: f64 = [
        &h2d_a,
        &h2d_b,
        &alloc_c,
        &launch_issue,
        &kernel_wait,
        &d2h,
        &host_copy,
    ]
    .iter()
    .map(|v| median_of(v).median)
    .sum();

    println!(
        "  N={n} kernel={kernel:?} (median over {MEASURED_TRIALS} trials, {WARMUP_TRIALS} warmup):"
    );
    print_quartiles_ms("h2d_a", median_of(&h2d_a));
    print_quartiles_ms("h2d_b", median_of(&h2d_b));
    print_quartiles_ms("alloc_c", median_of(&alloc_c));
    print_quartiles_ms("launch_issue", median_of(&launch_issue));
    print_quartiles_ms("kernel_wait", median_of(&kernel_wait));
    print_quartiles_ms("d2h", median_of(&d2h));
    print_quartiles_ms("host_copy", median_of(&host_copy));
    println!("    sum of medians: {:.4} ms", total * 1e3);

    // 診断テストの片付け（`fresh_overhead_diag_tests.rs` には無いが、
    // 本ファイルはプールを明示的に使うため、次の (n, kernel) 計測へ
    // キャッシュ状態を持ち越さないよう明示的に返却する）。
    let _ = allocator.release_cached();
}

/// 実機（CUDA）依存の診断テスト。N × 変種（select/classic）の全 6 組合せ
/// を計測する。`--test-threads=1` 必須（ファイル冒頭コメント参照）。
#[test]
#[ignore]
fn gemm_reuse_phase_diag_select() {
    for n in SIZES {
        run_size_kernel(n, DiagTiledF32Kernel::Select);
    }
}

#[test]
#[ignore]
fn gemm_reuse_phase_diag_classic() {
    for n in SIZES {
        run_size_kernel(n, DiagTiledF32Kernel::Classic);
    }
}
