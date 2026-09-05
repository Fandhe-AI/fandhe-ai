//! Metal GEMM reuse 計測境界（`scripts/bench/framework-compare` の
//! `bench-fandhe --task gemm --device metal --mode reuse --phases`）の
//! `matmul` 区間内訳を実測分解する診断テスト（イシュー #1189。CUDA 側
//! 同型調査 `crates/backend-cuda/src/gemm_reuse_phase_diag_tests.rs`・
//! イシュー #1182 の Metal 版）。
//!
//! # 背景
//!
//! `#1147`（`docs/perf/metal-gemm-candle-gate-remeasurement.md` §8）・
//! `#1036`/`#1103`（`docs/perf/metal-gemm-bottleneck-rediagnosis.md`
//! §4.1〜4.3・§7.1b）は、fandhe-ai 自系列内で「転送（アップロード＋
//! readback）＋同期」が reuse 計測境界の end-to-end 時間の約 38〜46%
//! （N=4096）を占めることまでは確認したが、candle 比未達の主因を
//! 転送と確定できず、`bench-fandhe` の reuse 1 反復
//! （`matmul`／`to_tensor`／`host_copy`／`checksum`）のどこに固定費が
//! 乗っているかは未分解のままだった。本ファイルはその `matmul` 区間の
//! 内側を `crates/backend-metal` の非公開 API へ直接アクセスして分解
//! する。
//!
//! # 配置理由（CUDA 側 `gemm_reuse_phase_diag_tests.rs` と同型の判断）
//!
//! `context_cache::{cached_context, cached_gemm}`（いずれも
//! `pub(crate)`）・`MetalGemm::diag_encode_tiled_nn`（`#[cfg(test)]
//! pub(crate)`。本イシューで新設）へ到達するため、integration test
//! ではなく `lib.rs` の兄弟モジュールとして配置する。
//!
//! # `run_gemm_reuse` 1 反復との対応・Metal 固有の区間定義
//!
//! `crates/facade` の `Var::matmul` → `MetalBackendOps::gemm`
//! （`ops.rs`）→ `MetalGemm::dispatch_auto`（`gemm.rs`）→
//! `dispatch_variant` の実体は毎反復 `MetalBuffer::new_with_data(A)`・
//! `MetalBuffer::new_with_data(B)`・`alloc_uninit_pooled(C)`・
//! `ctx.dispatch_sync`（encode + commit + `waitUntilCompleted`）・
//! `read_to_vec`。本ファイルはこの 1 反復を次の区間へ分解計測する
//! （CUDA の H2D／D2H に相当する区間は Apple Silicon の統合メモリ上の
//! memcpy であり、GPU への明示転送ではない点が CUDA と異なる。
//! `docs/perf/metal-gemm-reuse-phase-breakdown.md` §2 参照）:
//!
//! | 区間名 | 実体 |
//! | --- | --- |
//! | `upload_a`／`upload_b` | `MetalBuffer::new_with_data`（`newBufferWithBytes_length_options`。統合メモリへの memcpy） |
//! | `alloc_c` | `MetalBuffer::alloc_uninit_pooled`（プール経由。本診断は必ず `cached_context()` を使うためプールヒットが定常化する） |
//! | `encode` | `MetalContext::encode`（`pub(crate)`。エンコーダ生成＋バッファ結線＋ディスパッチ記録。commit しない） |
//! | `commit_wait` | `MetalContext::synchronize`（`flush_locked`＝`endEncoding`＋`commit` → `waitUntilCompleted`） |
//! | `readback` | `MetalBuffer::read_to_vec`（`contents()` からの memcpy。D2H 固有の同期は存在しない。`synchronize` 済み） |
//! | `host_copy` | `Vec::to_vec()`（`bench-fandhe` の `readout_var` 二重コピーに対応） |
//!
//! `pad_a`／`pad_b`／`unpad` は対象サイズ（1024/2048/4096。いずれも 8
//! の倍数）では `pad::pad_matrix` が `Cow::Borrowed` の no-op になる
//! ため計測しない（`docs/perf/metal-gemm-reuse-phase-breakdown.md` §2
//! に明記）。
//!
//! # GPU タイムスタンプ変種（`kernel_gpu`）を実装しない判断
//!
//! 実装計画は `MTLCommandBuffer::GPUStartTime`/`GPUEndTime` を使う
//! 変種 B（`commit_wait` から純カーネル専有時間を分離する）を「実装
//! 可能なら必須」としていたが、`MetalContext::encode`/`synchronize`
//! はバッチング機構（イシュー #1017・`docs/backend-metal-command-
//! batching-design.md`）によりコマンドバッファを内部に閉じ込めており
//! 呼び出し元へ公開しない。変種 B は自前のコマンドキュー経路
//! （`ctx.queue()`。`pub`）を新設する必要があり、AC-2「既存の本番
//! 経路・既存テストを変更しない（読み取り計測のみ）」の安全側判断
//! （計画§7 リスク節が明示的に許容する縮退先）に従い、本イシューでは
//! 変種 A（`ProductionBatch`。`encode`＋`synchronize` を本番同一経路で
//! 個別計時）のみを実装する。`commit_wait` は「commit＋カーネル専有
//! ＋`waitUntilCompleted`」の合算値として扱い、sync 単体は分離しない
//! （`docs/perf/metal-gemm-reuse-phase-breakdown.md` §8 に明記）。
//!
//! # 実行時は必ず `--test-threads=1`
//!
//! CUDA 側と同じ理由（同一 GPU 上での複数テストスレッド競合を避ける）。
//!
//! # メモリ使用量
//!
//! ホスト側キープアライブ（`Vec<Vec<f32>>`）は N=4096 で
//! (20 warmup + 20 測定) × 4096² × 4 bytes ≈ 2.7 GiB（統合メモリ上。
//! CUDA 側と同じオーダー）。
//!
//! # gating しない方針（CUDA 側と同じ理由）
//!
//! 本ファイルの `#[test]` は実行が成功すること（各フェーズが例外なく
//! 完了すること）のみを検証条件とし、フェーズ間の大小関係・絶対値への
//! `assert!` は行わない（環境揺らぎによる flaky 化防止。`out.len()`・
//! 有限性の sanity assert のみ行う）。数値は `println!` に残し、
//! `docs/perf/metal-gemm-reuse-phase-breakdown.md` へ転記する一次情報
//! とする。
//!
//! # プロダクションコード不変（AC-2）
//!
//! 本ファイルは純新規追加であり、`gemm.rs` への変更も新規 `pub(crate)`
//! ヘルパ `diag_encode_tiled_nn`（`#[cfg(test)]`）の追加のみに限定する
//! （既存 `dispatch_auto`／`dispatch_variant`／`dispatch_tiled_prepared`
//! は無変更）。

use std::time::Instant;

use bench_harness::{Quartiles, median_q1_q3, rng::Xorshift64Star};

use crate::buffer::MetalBuffer;
use crate::context_cache::{cached_context, cached_gemm};
use crate::tile;

const WARMUP_TRIALS: usize = 20;
const MEASURED_TRIALS: usize = 20;

/// `bench-fandhe --task gemm --mode reuse --phases`（イシュー #1189）が
/// 対象とするのと同じサイズ（`docs/perf/metal-gemm-candle-gate-
/// remeasurement.md` 対象形状。CUDA 側 `SIZES` と同一）。
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

/// 1 反復分のフェーズ計測結果（ファイル冒頭「対応」表参照）。
struct PhaseSample {
    upload_a_secs: f64,
    upload_b_secs: f64,
    alloc_c_secs: f64,
    encode_secs: f64,
    commit_wait_secs: f64,
    readback_secs: f64,
    host_copy_secs: f64,
}

/// 1 サイズの 1 反復を計測する。`ctx`／`gemm` は呼び出し元が
/// `context_cache` 経由で取得したハンドル（本番 `ops::MetalBackendOps
/// ::gemm` と同じキャッシュ経由。プールヒットを定常化するため
/// `alloc_uninit_pooled` は `cached_context()` に一致する `ctx` でのみ
/// プール経由になる契約 — `buffer.rs::MetalBuffer::alloc_uninit_pooled`
/// ドキュメンテーションコメント参照）。
///
/// 選択構成 `cfg` は `dispatch_auto` と同一の `tile::select_for_device`
/// で呼び出し元が 1 度だけ解決し、全反復で使い回す（本番も `dispatch_
/// auto` 呼び出しのたびに再解決するため反復間で変わらない。診断側で
/// 反復ごとに解決し直しても結果は同じだが、解決コスト自体をフェーズへ
/// 混入させないためループ外で 1 回だけ呼ぶ）。
fn measure_one_phase_trial(
    ctx: &crate::context::MetalContext,
    gemm: &crate::gemm::MetalGemm,
    a: &[f32],
    b: &[f32],
    n: usize,
    cfg: tile::TileConfig,
    keep_alive: &mut Vec<Vec<f32>>,
) -> PhaseSample {
    let t = Instant::now();
    let a_buf =
        MetalBuffer::new_with_data(ctx, a).expect("A upload (host->device memcpy) must succeed");
    let upload_a_secs = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let b_buf =
        MetalBuffer::new_with_data(ctx, b).expect("B upload (host->device memcpy) must succeed");
    let upload_b_secs = t.elapsed().as_secs_f64();

    // 本番 `dispatch_variant` と同じプール経由確保（`buffer.rs::
    // MetalBuffer::alloc_uninit_pooled` ドキュメンテーションコメント
    // 参照）。
    let t = Instant::now();
    let c_buf = MetalBuffer::alloc_uninit_pooled(ctx, n * n)
        .expect("pooled output buffer allocation must succeed");
    let alloc_c_secs = t.elapsed().as_secs_f64();

    // エンコード（記録のみ・commit しない。ファイル冒頭「GPU タイム
    // スタンプ変種を実装しない判断」参照）。
    let t = Instant::now();
    gemm.diag_encode_tiled_nn(ctx, &a_buf, &b_buf, &c_buf, n, n, n, cfg)
        .expect("diag_encode_tiled_nn (record only) must succeed");
    let encode_secs = t.elapsed().as_secs_f64();

    // commit + GPU 完了待ち（`kernel_gpu` を分離しない合算値。ファイル
    // 冒頭コメント参照）。
    let t = Instant::now();
    ctx.synchronize()
        .expect("synchronize (commit + waitUntilCompleted) must succeed");
    let commit_wait_secs = t.elapsed().as_secs_f64();

    // readback: `contents()` からの memcpy（D2H 固有の同期は存在しない。
    // 直前の `synchronize` で書き込み完了済み）。
    let t = Instant::now();
    let out = c_buf.read_to_vec();
    let readback_secs = t.elapsed().as_secs_f64();

    // ホストコピー（`bench-fandhe` の `readout_var` 二重コピーに対応。
    // `read_to_vec` が返す `Vec<f32>` は既にホスト所有のため、ここでは
    // framework-compare 側の `host_copy` 区間に対応する「複製」の
    // コストを模して `to_vec()` でもう 1 段コピーする）。
    let t = Instant::now();
    let copied = out.to_vec();
    let host_copy_secs = t.elapsed().as_secs_f64();

    // sanity: 有限（本ファイル冒頭「gating しない方針」参照。大小関係
    // への assert は行わない）。
    assert_eq!(copied.len(), n * n);
    assert!(
        copied.iter().all(|v| v.is_finite()),
        "output must be finite (n={n})"
    );

    // `a_buf`／`b_buf`／`c_buf` はこの関数末尾で drop する（`c_buf` は
    // プールへ返却される。イシュー #1021）。ホスト側 `copied` のみ
    // `keep_alive` に保持する（メモリ使用量の見積り根拠。ファイル冒頭
    // コメント参照）。
    keep_alive.push(copied);

    PhaseSample {
        upload_a_secs,
        upload_b_secs,
        alloc_c_secs,
        encode_secs,
        commit_wait_secs,
        readback_secs,
        host_copy_secs,
    }
}

fn run_size(n: usize) {
    let ctx = cached_context().expect("Metal device (system default) must be available");
    let gemm = cached_gemm(&ctx).expect("MetalGemm construction must succeed");

    // `dispatch_auto` と同一の構成解決（ファイル冒頭「1 サイズの 1 反復」
    // ドキュメンテーションコメント参照）。
    let cfg = tile::select_for_device(n, n, n, ctx.verified_m4_max_gpu_core_count());

    let (a, b) = gen_square_ab(0x1189_a000 ^ (n as u64), n);
    let mut keep_alive: Vec<Vec<f32>> = Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);

    for _ in 0..WARMUP_TRIALS {
        let _ = measure_one_phase_trial(&ctx, &gemm, &a, &b, n, cfg, &mut keep_alive);
    }

    let mut upload_a = Vec::with_capacity(MEASURED_TRIALS);
    let mut upload_b = Vec::with_capacity(MEASURED_TRIALS);
    let mut alloc_c = Vec::with_capacity(MEASURED_TRIALS);
    let mut encode = Vec::with_capacity(MEASURED_TRIALS);
    let mut commit_wait = Vec::with_capacity(MEASURED_TRIALS);
    let mut readback = Vec::with_capacity(MEASURED_TRIALS);
    let mut host_copy = Vec::with_capacity(MEASURED_TRIALS);

    for _ in 0..MEASURED_TRIALS {
        let s = measure_one_phase_trial(&ctx, &gemm, &a, &b, n, cfg, &mut keep_alive);
        upload_a.push(s.upload_a_secs);
        upload_b.push(s.upload_b_secs);
        alloc_c.push(s.alloc_c_secs);
        encode.push(s.encode_secs);
        commit_wait.push(s.commit_wait_secs);
        readback.push(s.readback_secs);
        host_copy.push(s.host_copy_secs);
    }

    let total: f64 = [
        &upload_a,
        &upload_b,
        &alloc_c,
        &encode,
        &commit_wait,
        &readback,
        &host_copy,
    ]
    .iter()
    .map(|v| median_of(v).median)
    .sum();

    println!(
        "  N={n} resolved_tile={cfg:?} (median over {MEASURED_TRIALS} trials, {WARMUP_TRIALS} warmup):"
    );
    print_quartiles_ms("upload_a", median_of(&upload_a));
    print_quartiles_ms("upload_b", median_of(&upload_b));
    print_quartiles_ms("alloc_c", median_of(&alloc_c));
    print_quartiles_ms("encode", median_of(&encode));
    print_quartiles_ms("commit_wait", median_of(&commit_wait));
    print_quartiles_ms("readback", median_of(&readback));
    print_quartiles_ms("host_copy", median_of(&host_copy));
    println!("    sum of medians: {:.4} ms", total * 1e3);
}

/// 実機（Metal）依存の診断テスト。N=1024/2048/4096 の全サイズを計測
/// する。`--test-threads=1` 必須（ファイル冒頭コメント参照）。
#[test]
#[ignore]
fn gemm_reuse_phase_diag_production_batch() {
    for n in SIZES {
        run_size(n);
    }
}
