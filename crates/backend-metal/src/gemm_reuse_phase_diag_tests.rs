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
//! # GPU タイムスタンプ変種（`kernel_gpu`）の実装（イシュー #1276）
//!
//! `#1189`（本ファイル。旧版）は `MTLCommandBuffer::GPUStartTime`/
//! `GPUEndTime` を使う変種 B（`commit_wait` から純カーネル専有時間を
//! 分離する）を、`MetalContext::encode`/`synchronize` がバッチング
//! 機構（イシュー #1017・`docs/backend-metal-command-batching-
//! design.md`）によりコマンドバッファを内部に閉じ込めていることを
//! 理由に見送っていた。イシュー #1276 は「自前のコマンドキュー経路を
//! 新設する」のではなく「`MetalContext::synchronize` の内部（完了
//! バッチを `waitUntilCompleted` した直後・drop する前）にオブザーバ
//! を差し込む」方式（`context.rs::synchronize_observed`／
//! `#[cfg(test)] pub(crate) synchronize_with_gpu_timestamps`）でこれを
//! 解消した。本番 `synchronize()` は no-op オブザーバのままのため
//! ディスパッチ挙動・数値結果は不変（AC-2）。`commit_wait` は引き続き
//! 「commit＋カーネル専有＋`waitUntilCompleted`」の合算値として扱い、
//! `kernel_gpu`（GPUEnd−GPUStart）はその内訳の 1 項目として別途出力
//! する（`sum of medians` への二重計上はしない。§後述「フェーズ計測
//! 結果」参照）。N=1024/2048/4096 の 5 run 実測・`docs/perf/metal-
//! gemm-reuse-phase-breakdown.md` への数表転記は本イシューのスコープ
//! 外（親 #1275 配下の実測 sub-issue が担う）。
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
//! 完了すること）のみを検証条件とし、フェーズ間の**性能の**大小関係・
//! 絶対値への `assert!` は行わない（環境揺らぎによる flaky 化防止。
//! `out.len()`・有限性の sanity assert のみ行う）。数値は `println!` に
//! 残し、`docs/perf/metal-gemm-reuse-phase-breakdown.md` へ転記する
//! 一次情報とする。
//!
//! これは**正しさの不変条件**（バッチ構成・順序・タイムスタンプの
//! 内包関係）への assert とは別物である: `gpu_timestamps_within_
//! commit_wait_window`（イシュー #1276 で新設）は「返却バッチ数が
//! 1」「`labels` が想定どおり」「両タイムスタンプが `Some`」
//! 「`kernel_gpu >= 0`」「`kernel_gpu <= commit_wait`」を assert する
//! が、これらは実行結果の**論理的整合性**の検査であり「値がどれだけ
//! 速いか」という性能値の大小 gating ではないため、上記の gating しない
//! 方針とは矛盾しない。
//!
//! # プロダクションコード不変（AC-2）
//!
//! `gemm.rs` への変更は新規 `pub(crate)` ヘルパ `diag_encode_tiled_nn`
//! （`#[cfg(test)]`）の追加のみに限定する（既存 `dispatch_auto`／
//! `dispatch_variant`／`dispatch_tiled_prepared`は無変更）。
//! `context.rs::synchronize` はイシュー #1276 で内部を `synchronize_
//! observed`（オブザーバ引数を取る）へ切り出したが、本番
//! `synchronize()` は no-op オブザーバで呼ぶ薄いラッパーのため、
//! 挙動・エラー伝播・ロック区間・`pending_pool_returns` 合流順序は
//! 従来と完全に同一（追加の FFI 呼び出しもゼロ）。

use std::time::Instant;

use bench_harness::{Quartiles, median_q1_q3, rng::Xorshift64Star};

use crate::buffer::MetalBuffer;
use crate::context_cache::{cached_context, cached_gemm};
use crate::tile;

pub(crate) const WARMUP_TRIALS: usize = 20;
pub(crate) const MEASURED_TRIALS: usize = 20;

/// `bench-fandhe --task gemm --mode reuse --phases`（イシュー #1189）が
/// 対象とするのと同じサイズ（`docs/perf/metal-gemm-candle-gate-
/// remeasurement.md` 対象形状。CUDA 側 `SIZES` と同一）。
const SIZES: [usize; 3] = [1024, 2048, 4096];

pub(crate) fn gen_square_ab(seed: u64, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let a = rng.fill_vec(n * n);
    let b = rng.fill_vec(n * n);
    (a, b)
}

pub(crate) fn median_of(samples: &[f64]) -> Quartiles {
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
pub(crate) struct PhaseSample {
    /// `diag_encode_tiled_nn` が実際にディスパッチした構成（`pipeline_
    /// for_tile` のフォールバック解決後）。要求構成 `cfg`（呼び出し元が
    /// `tile::select_for_device` で解決した値）と一致するとは限らない
    /// （デバイス上限超過・パイプライン構築失敗時にフォールバックしうる。
    /// `pipeline_for_tile` ドキュメンテーションコメント参照）ため、
    /// 計測時間をどの構成の性能として記録したかを区別するために保持
    /// する（codex-review 指摘 #1189）。
    pub(crate) resolved_cfg: tile::TileConfig,
    pub(crate) upload_a_secs: f64,
    pub(crate) upload_b_secs: f64,
    pub(crate) alloc_c_secs: f64,
    pub(crate) encode_secs: f64,
    pub(crate) commit_wait_secs: f64,
    /// `commit_wait_secs` の内訳（イシュー #1276。`context.rs::
    /// synchronize_with_gpu_timestamps` が返す `GPUEndTime−GPUStartTime`）。
    /// `commit_wait_secs` へは二重計上しない（`run_size` の `sum of
    /// medians` は従来どおり 7 フェーズの合計のまま）。
    pub(crate) kernel_gpu_secs: f64,
    pub(crate) readback_secs: f64,
    pub(crate) host_copy_secs: f64,
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
pub(crate) fn measure_one_phase_trial(
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
    let resolved_cfg = gemm
        .diag_encode_tiled_nn(ctx, &a_buf, &b_buf, &c_buf, n, n, n, cfg)
        .expect("diag_encode_tiled_nn (record only) must succeed");
    let encode_secs = t.elapsed().as_secs_f64();

    // commit + GPU 完了待ち（壁時計は従来どおり合算値。`kernel_gpu` は
    // この区間の内訳としてイシュー #1276 で追加。GPU タイムスタンプは
    // 同一ホスト単調クロックではなく `CFTimeInterval`（別の時刻基準）
    // のため、`commit_wait_secs` という壁時計と直接比較するのではなく
    // 「この `Instant` 区間の内側に完全に収まるはず」という区間長の
    // 大小関係のみを診断テスト側で検証する。ファイル冒頭コメント
    // 「GPU タイムスタンプ変種」参照）。
    let t = Instant::now();
    let batches = ctx
        .synchronize_with_gpu_timestamps()
        .expect("synchronize (commit + waitUntilCompleted) must succeed");
    let commit_wait_secs = t.elapsed().as_secs_f64();
    // `diag_encode_tiled_nn` の 1 回の `encode` は 1 バッチにつき
    // ちょうど 1 ディスパッチのみを記録するため、この `synchronize` が
    // 完了させるバッチは常にちょうど 1 個のはず（`cached_context()` は
    // プロセスワイド singleton のため `--test-threads=1` が前提。
    // ファイル冒頭「実行時は必ず `--test-threads=1`」参照）。他の診断
    // テスト・並行スレッドからのディスパッチが紛れ込んでいないかを
    // ここで検出する。
    assert_eq!(
        batches.len(),
        1,
        "synchronize_with_gpu_timestamps must complete exactly one batch per diag_encode_tiled_nn call \
         (got {}; run with --test-threads=1)",
        batches.len()
    );
    let batch = &batches[0];
    assert_eq!(
        batch.labels(),
        ["diag_encode_tiled_nn"],
        "unexpected dispatch labels in the completed batch: {:?}",
        batch.labels()
    );
    let kernel_gpu_secs = batch.kernel_gpu_secs().unwrap_or_else(|| {
        panic!(
            "GPUStartTime/GPUEndTime must both be non-zero for a completed batch (labels={:?})",
            batch.labels()
        )
    });
    assert!(
        kernel_gpu_secs >= 0.0,
        "kernel_gpu_secs must be non-negative (GPUEndTime must not precede GPUStartTime): {kernel_gpu_secs}"
    );
    assert!(
        kernel_gpu_secs <= commit_wait_secs,
        "kernel_gpu_secs ({kernel_gpu_secs:.6}s) must fit inside the commit_wait wall-clock window \
         ({commit_wait_secs:.6}s) that fully encloses commit+kernel+waitUntilCompleted"
    );

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
        resolved_cfg,
        upload_a_secs,
        upload_b_secs,
        alloc_c_secs,
        encode_secs,
        commit_wait_secs,
        kernel_gpu_secs,
        readback_secs,
        host_copy_secs,
    }
}

/// 1 サイズ分の全反復（warmup + 測定）を実行し、フェーズ内訳を
/// `println!` する（`run_size` の本体。イシュー #1289 で `ctx`／`gemm`／
/// `label` を引数化し、`gemm_spec_source_diag_tests` が本番
/// `cached_gemm()` 以外の `MetalGemm` インスタンス（`new_with_source_
/// specialization` で構築した base/head）でも同じフェーズ分解を再利用
/// できるようにした。`run_size`（本番 `cached_gemm()` 固定）は本関数の
/// 薄いラッパーのまま出力形式・挙動とも不変）。`label` は出力の先頭に
/// 付与し、複数インスタンスの出力を `grep` で区別できるようにする。
pub(crate) fn run_size_with(
    ctx: &crate::context::MetalContext,
    gemm: &crate::gemm::MetalGemm,
    n: usize,
    label: &str,
) {
    // `dispatch_auto` と同一の構成解決（ファイル冒頭「1 サイズの 1 反復」
    // ドキュメンテーションコメント参照）。
    let cfg = tile::select_for_device(n, n, n, ctx.verified_m4_max_gpu_core_count());

    let (a, b) = gen_square_ab(0x1189_a000 ^ (n as u64), n);
    let mut keep_alive: Vec<Vec<f32>> = Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);

    for _ in 0..WARMUP_TRIALS {
        let _ = measure_one_phase_trial(ctx, gemm, &a, &b, n, cfg, &mut keep_alive);
    }

    let mut upload_a = Vec::with_capacity(MEASURED_TRIALS);
    let mut upload_b = Vec::with_capacity(MEASURED_TRIALS);
    let mut alloc_c = Vec::with_capacity(MEASURED_TRIALS);
    let mut encode = Vec::with_capacity(MEASURED_TRIALS);
    let mut commit_wait = Vec::with_capacity(MEASURED_TRIALS);
    // `commit_wait` の内訳（イシュー #1276）。`sum of medians` へは
    // 含めない（`commit_wait` 自身が既に合算値のため。二重計上防止）。
    let mut kernel_gpu = Vec::with_capacity(MEASURED_TRIALS);
    let mut readback = Vec::with_capacity(MEASURED_TRIALS);
    let mut host_copy = Vec::with_capacity(MEASURED_TRIALS);
    // 要求構成 `cfg` と実際にディスパッチされた構成（`pipeline_for_tile`
    // フォールバック解決後）が全測定反復で一致するかを記録する
    // （codex-review 指摘 #1189: フォールバック時に「実行していない
    // 構成」の性能として誤記録することを防ぐため、両者を区別して出力
    // する）。
    let mut resolved_cfgs: Vec<tile::TileConfig> = Vec::with_capacity(MEASURED_TRIALS);

    for _ in 0..MEASURED_TRIALS {
        let s = measure_one_phase_trial(ctx, gemm, &a, &b, n, cfg, &mut keep_alive);
        upload_a.push(s.upload_a_secs);
        upload_b.push(s.upload_b_secs);
        alloc_c.push(s.alloc_c_secs);
        encode.push(s.encode_secs);
        commit_wait.push(s.commit_wait_secs);
        kernel_gpu.push(s.kernel_gpu_secs);
        readback.push(s.readback_secs);
        host_copy.push(s.host_copy_secs);
        resolved_cfgs.push(s.resolved_cfg);
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

    // 測定反復全体で解決構成が一意なら単一値、フォールバックが発生し
    // 揺れていれば「一致しなかった」ことが分かる形で出力する（要求
    // 構成と実行構成を混同しない）。
    let resolved_tile_report: String = {
        let mut distinct: Vec<tile::TileConfig> = Vec::new();
        for &rc in &resolved_cfgs {
            if !distinct.contains(&rc) {
                distinct.push(rc);
            }
        }
        match distinct.as_slice() {
            [only] => format!("{only:?}"),
            other => format!("MIXED{other:?}"),
        }
    };
    let fallback_occurred = resolved_cfgs.iter().any(|&rc| rc != cfg);

    println!(
        "  [{label}] N={n} requested_tile={cfg:?} resolved_tile={resolved_tile_report} (median over {MEASURED_TRIALS} trials, {WARMUP_TRIALS} warmup):"
    );
    if fallback_occurred {
        println!(
            "    NOTE: resolved_tile diverged from requested_tile in at least one measured trial (pipeline_for_tile fallback) — below timings reflect the ACTUAL executed configuration(s), not the requested one."
        );
    }
    print_quartiles_ms("upload_a", median_of(&upload_a));
    print_quartiles_ms("upload_b", median_of(&upload_b));
    print_quartiles_ms("alloc_c", median_of(&alloc_c));
    print_quartiles_ms("encode", median_of(&encode));
    print_quartiles_ms("commit_wait", median_of(&commit_wait));
    // `kernel_gpu`（GPUEndTime−GPUStartTime）は `commit_wait` の内訳
    // （イシュー #1276）。`commit_wait_minus_kernel_gpu` はその残り
    // （commit・スケジューリング・`waitUntilCompleted` 復帰までの
    // オーバーヘッドに相当）。いずれも `sum of medians` には含めない
    // （二重計上防止。ファイル冒頭コメント参照）。
    let kernel_gpu_q = median_of(&kernel_gpu);
    print_quartiles_ms("commit_wait.kernel_gpu", kernel_gpu_q);
    // 各反復ごとに `commit_wait − kernel_gpu` の差を取ってから中央値を
    // 求める（`median(commit_wait) − median(kernel_gpu)` は反復ごとの
    // 揺らぎで中央値に対応する反復がずれるため一般には一致せず、GPU 外
    // オーバーヘッドを過大・過小評価しうる。PR #1371 レビュー指摘）。
    let commit_wait_minus_kernel_gpu: Vec<f64> = commit_wait
        .iter()
        .zip(kernel_gpu.iter())
        .map(|(&cw, &kg)| cw - kg)
        .collect();
    println!(
        "    commit_wait.commit_wait_minus_kernel_gpu: median={:.4} ms",
        median_of(&commit_wait_minus_kernel_gpu).median * 1e3
    );
    print_quartiles_ms("readback", median_of(&readback));
    print_quartiles_ms("host_copy", median_of(&host_copy));
    println!("    sum of medians: {:.4} ms", total * 1e3);
}

/// [`run_size_with`] の本番既定ラッパー（従来の `run_size`。出力形式・
/// 挙動とも従来どおり不変）: `cached_context()`／`cached_gemm()`（本番
/// `ops::MetalBackendOps::gemm` と同じプロセス内キャッシュ）を使う。
fn run_size(n: usize) {
    let ctx = cached_context().expect("Metal device (system default) must be available");
    let gemm = cached_gemm(&ctx).expect("MetalGemm construction must succeed");
    run_size_with(&ctx, &gemm, n, "production");
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

/// AC-3 の自己検証専用（イシュー #1276）: N=1024 のみ・少数反復で
/// `synchronize_with_gpu_timestamps` が返す正しさ不変条件（バッチ数・
/// ラベル・タイムスタンプの取得可否・`kernel_gpu` の非負性・
/// `commit_wait` 内包）を単独・短時間で検証する導線。フル 3 サイズ×
/// `MEASURED_TRIALS` 回の `gemm_reuse_phase_diag_production_batch` とは
/// 独立に実行できる（実行方法は `Makefile` `test-ignored-metal` 相当。
/// `--test-threads=1` 必須）。assert 自体は `measure_one_phase_trial`
/// 内（`commit_wait_secs` 計測ブロック）にあるため、本テストは同関数を
/// 少数回呼ぶだけでよい。
#[test]
#[ignore]
fn gpu_timestamps_within_commit_wait_window() {
    const N: usize = 1024;
    const TRIALS: usize = 5;

    let ctx = cached_context().expect("Metal device (system default) must be available");
    let gemm = cached_gemm(&ctx).expect("MetalGemm construction must succeed");
    let cfg = tile::select_for_device(N, N, N, ctx.verified_m4_max_gpu_core_count());
    let (a, b) = gen_square_ab(0x1276_a000, N);
    let mut keep_alive: Vec<Vec<f32>> = Vec::with_capacity(TRIALS);

    for i in 0..TRIALS {
        let s = measure_one_phase_trial(&ctx, &gemm, &a, &b, N, cfg, &mut keep_alive);
        println!(
            "  trial {i}: commit_wait={:.4}ms kernel_gpu={:.4}ms (commit_wait-kernel_gpu={:.4}ms)",
            s.commit_wait_secs * 1e3,
            s.kernel_gpu_secs * 1e3,
            (s.commit_wait_secs - s.kernel_gpu_secs) * 1e3
        );
    }
}
