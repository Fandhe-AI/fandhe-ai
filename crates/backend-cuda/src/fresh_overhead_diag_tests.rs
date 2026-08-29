//! CUDA GEMM fresh モードの N=2048 固有オーバーヘッド（イシュー #956。
//! #946〈`context_cache` プロセス内キャッシュ〉反映後の
//! `scripts/bench/framework-compare/` 実測で、fresh モード GEMM が
//! N=1024・N=4096 では reuse モードとほぼ一致する一方、N=2048 のみで
//! 約 166 ms の再現性ある固定コストを残す事象）のフェーズ分解診断。
//!
//! # 配置理由（`init_cost_diag_tests.rs` と同型の判断）
//!
//! `context_cache::cached_device`／`cached_gemm`（いずれも `pub(crate)`。
//! `context_cache.rs`）へ到達するため、integration test ではなく
//! `lib.rs` の兄弟モジュールとして配置する。
//!
//! # 帰属の明確化（イシュー #956 本文の解釈。実装計画 §2）
//!
//! `scripts/bench/framework-compare/bench-fandhe/src/main.rs::run_gemm`
//! （fresh）と `run_gemm_reuse`（reuse）の 1 イテレーションあたりの差分は
//! 次の 3 つに限られる:
//!
//! - (f1) `fandhe_ai::tape_for(Device::Cuda(0))`（`resolve_ops` →
//!   `CudaDeviceProvider::select` → `context_cache::cached_device` +
//!   `ctx.total_mem()` + `ctx.attribute(MULTIPROCESSOR_COUNT)`）
//! - (f2) `tape.var(&a_data)`／`tape.var(&b_data)`（`Tensor` の
//!   `Arc<Storage>` clone。深いコピーなし）
//! - (f3) イテレーション末尾の `Tape` drop。A・B は `Arc` 参照カウント減の
//!   みだが、**結果テンソル C（N=2048 で 16 MiB の `Vec<f32>`。
//!   `clone_dtoh` が確保）はここで解放される**。reuse では C ノードが
//!   tape 上に蓄積され解放されない
//!
//! fresh N=1024 の合計が約 1.9 ms（イシュー #956 本文）であるため、
//! サイズ非依存の (f1)(f2) は高々 1.9 ms に収まる。約 166 ms は
//! 「サイズ依存かつ fresh 限定」の要因、すなわち (f3) の C 解放と
//! その後続影響（次イテレーションの D2H 転送先バッファ確保が「直前に
//! 解放された同サイズ領域」を再利用する点）に帰属するはずである。本
//! ファイルはこの帰属を実機で検証するため、GEMM 1 回を (a) H2D A・
//! (b) H2D B・(c) C 確保・(d) launch+synchronize・(e) D2H・(f) ホスト
//! 出力バッファ解放へ分解計測する。
//!
//! # 判別対象の仮説（実装計画 §2.3）
//!
//! - H1: D2H 宛先（`Vec::with_capacity` + `set_len` の未タッチページ。
//!   `cudarc-0.19.8/src/driver/safe/core.rs` の `clone_dtoh`）への
//!   `cuMemcpyDtoHAsync` が、直前に解放・再確保された領域で著しく遅い
//! - H2: glibc の動的 mmap しきい値遷移（16 MiB 解放 → brk ヒープ化 →
//!   trim/再フォールト）の反復コスト
//! - H3: 解放（munmap／brk 縮小）自体が driver 側（MMU notifier 経由の
//!   GPU ページテーブル無効化等）で高コスト
//! - H4: (f1) の毎回プローブが主因（N=1024 の 1.9 ms 上界により事前に
//!   除外見込みだが、本ファイルは (g) として参考値を併記し確定する）
//!
//! D2H 宛先の変種（V0〜V3）で H1 を、`--test-threads=1` 実行下での
//! `strace`／`MALLOC_MMAP_THRESHOLD_` 等の外部条件変更（本ファイルでは
//! 制御しない。実行手順は `docs/perf/
//! cuda-fresh-gemm-n2048-overhead-diagnosis.md` §5 参照）で H2 を、
//! V1（保持して解放を抑止）との比較で H3 を判別する。
//!
//! # 実行時は必ず `--test-threads=1`（`init_cost_diag_tests.rs` と同じ
//! 理由: 同一 GPU 上での複数テストスレッド競合を避ける）
//!
//! # gating しない方針（`init_cost_diag_tests.rs`・`jit_cache_bench_tests.rs`
//! と同じ理由）
//!
//! 本ファイルの `#[test]` は実行が成功すること（H2D／確保／launch／
//! synchronize／D2H が例外なく完了すること）のみを検証条件とし、
//! フェーズ間の大小関係・絶対値への `assert!` は行わない（環境揺らぎに
//! よる flaky 化防止）。数値は `println!` に残し、`docs/perf/
//! cuda-fresh-gemm-n2048-overhead-diagnosis.md` へ転記する一次情報とする。

use std::time::Instant;

use cudarc::driver::CudaSlice;

use bench_harness::{Quartiles, median_q1_q3, rng::Xorshift64Star};

use crate::context_cache::{cached_device, cached_gemm};
use crate::device::CudaDevice;

const WARMUP_TRIALS: usize = 3;
const MEASURED_TRIALS: usize = 10;

/// イシュー #956 本文が固定コストを観測した対象サイズを中心に、
/// サイズ非依存であるはずの (f1)(f2) の上界確認（N=1024）・別サイズでの
/// 非再現確認（N=4096）を含めた 3 点。
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
        "  {label}: median={:.3} ms  q1={:.3} ms  q3={:.3} ms",
        q.median * 1e3,
        q.q1 * 1e3,
        q.q3 * 1e3
    );
}

/// 1 試行の H2D/launch/D2H フェーズ分解結果。
struct PhaseSample {
    h2d_a_secs: f64,
    h2d_b_secs: f64,
    alloc_c_secs: f64,
    launch_sync_secs: f64,
    d2h_secs: f64,
    /// ホスト出力バッファ（`Vec<f32>`）を明示的に drop するのに要した
    /// 時間（(f3) 相当。V0 でのみ意味を持つ。V1 は drop しないため常に
    /// 0.0）。
    free_host_secs: f64,
}

/// D2H 宛先の確保方式（H1 判別）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DownloadVariant {
    /// V0: 毎試行 `stream.clone_dtoh`（未タッチページの `Vec`。本番
    /// `CudaGemm::download_f32`／fresh モードの `run_tiled_f32` と同一
    /// 経路）で確保 + 転送し、試行末尾で即 drop する（fresh 相当）。
    Fresh,
    /// V1: 毎試行 `clone_dtoh` するが、結果を呼び出し元の `Vec<Vec<f32>>`
    /// に保持し試行間で drop しない（reuse 相当。H3 判別: 解放自体が
    /// 高コストなら V1 は V0 より速いはず）。
    KeepAlive,
    /// V2: 確保後に全要素へ明示書き込みでページを事前タッチしてから
    /// `memcpy_dtoh` する（未タッチページ由来のフォルトコストを事前に
    /// 支払わせて転送区間から除く。H1 判別）。
    PreTouched,
}

/// 1 サイズ・1 変種の 1 試行を計測する。`gemm` は呼び出し元が
/// `cached_gemm` で取得したハンドル（本番 `ops::CudaBackendOps::gemm` と
/// 同じキャッシュ経由）を使い回す — fresh モードでも #946 以降は
/// `CudaGemm` 自体はプロセス内キャッシュされるため、本診断の対象は
/// あくまで「毎試行の H2D/確保/launch/D2H/解放」であり、`CudaGemm::new`
/// のコストではない（イシュー #956 の帰属分析 (f1)(f2)(f3) 参照）。
fn measure_one_phase_trial(
    device: &CudaDevice,
    gemm: &crate::gemm::CudaGemm,
    a: &[f32],
    b: &[f32],
    n: u32,
    variant: DownloadVariant,
    keep_alive: &mut Vec<Vec<f32>>,
) -> PhaseSample {
    // `device.stream()` は `&Arc<CudaStream>` を返す（`device.rs`）。
    // `CudaStream` の各メソッドは `self: &Arc<Self>` を取るため、既存経路
    // （`gemm.rs`・`memory.rs` 等）と同じく `.clone()` で所有権を持つ
    // `Arc<CudaStream>` を作ってから呼ぶ。
    let stream = device.stream().clone();

    // A・B を個別にタイムスタンプするため `gemm.upload_f32`（両方まとめて
    // 1 メソッドで転送する公開ヘルパー。`gemm.rs` 参照）は使わず、同じ
    // `stream.clone_htod` を個別に 2 回呼ぶ（`upload_f32` の実装と同一の
    // 呼び出し。フェーズを分離するための呼び出し形だけの違い）。
    let t = Instant::now();
    let a_dev: CudaSlice<f32> = stream.clone_htod(a).expect("H2D A upload must succeed");
    let h2d_a_secs = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let b_dev: CudaSlice<f32> = stream.clone_htod(b).expect("H2D B upload must succeed");
    let h2d_b_secs = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut c_dev = gemm
        .alloc_output_f32(n, n)
        .expect("output device buffer allocation must succeed");
    let alloc_c_secs = t.elapsed().as_secs_f64();

    let t = Instant::now();
    gemm.launch_tiled_f32(&a_dev, &b_dev, &mut c_dev, n, n, n)
        .expect("launch_tiled_f32 (H2D-less GPU execution + synchronize) must succeed");
    let launch_sync_secs = t.elapsed().as_secs_f64();

    let (d2h_secs, free_host_secs, out_len) = match variant {
        DownloadVariant::Fresh => {
            let t = Instant::now();
            let out = stream
                .clone_dtoh(&c_dev)
                .expect("D2H download (untouched-page Vec, production clone_dtoh) must succeed");
            // `clone_dtoh`／`memcpy_dtoh` は `cuMemcpyDtoHAsync` を発行する
            // だけで返る（plain `Vec<T>` は `HostSlice::stream_synced_mut_
            // slice` が `SyncOnDrop::Sync(None)` を返すため drop 時の暗黙
            // sync も無い。cudarc-0.19.8 `src/driver/safe/core.rs`
            // `memcpy_dtoh`／`HostSlice for Vec<T>` 実装）。ここで
            // `stream.synchronize()` を挟まないと、直後の drop（このすぐ
            // 下）がホストバッファを解放する一方で GPU 側の書き込みが
            // 未完了のままになりうる（転送未完了のうちに解放される安全性
            // 問題。GB10 実機で顕在化しうる。PR #973 レビュー指摘）。
            // これにより d2h_secs は「転送発行のみ」ではなく「転送完了
            // まで」を計測する値になる（本診断の (e) D2H フェーズの定義
            // 〈ファイル冒頭コメント〉に合わせて正確化）。
            stream
                .synchronize()
                .expect("stream synchronize after D2H must succeed before host buffer is freed");
            let d2h_secs = t.elapsed().as_secs_f64();
            let out_len = out.len();
            // (f3) 相当: fresh モードのイテレーション末尾で C が drop
            // されるのと同じタイミングで、ここで明示的に drop し所要時間を
            // 計測する（H3 判別材料）。
            let t = Instant::now();
            drop(out);
            let free_host_secs = t.elapsed().as_secs_f64();
            (d2h_secs, free_host_secs, out_len)
        }
        DownloadVariant::KeepAlive => {
            let t = Instant::now();
            let out = stream
                .clone_dtoh(&c_dev)
                .expect("D2H download (untouched-page Vec, kept alive) must succeed");
            // Fresh 分岐と同じ理由（上記コメント）で、`keep_alive` へ退避
            // する前に転送完了を待つ。KeepAlive はこの試行内では drop
            // しないが、`keep_alive` は次サイズへ進む前に呼び出し元が
            // 明示的に drop するため（本ファイル内の使用箇所参照）、
            // その時点で転送未完了のまま解放されるのを防ぐには、ここで
            // 完了を確定させておく必要がある。
            stream
                .synchronize()
                .expect("stream synchronize after D2H must succeed before host buffer is freed");
            let d2h_secs = t.elapsed().as_secs_f64();
            let out_len = out.len();
            // reuse 相当: 保持して次試行以降も解放しない（H3 判別: 解放を
            // 抑止した場合に D2H 自体が速いままなら、166 ms は解放コスト
            // ではなく「直前に解放された領域への再アロケーション」由来
            // だと分かる）。
            keep_alive.push(out);
            (d2h_secs, 0.0, out_len)
        }
        DownloadVariant::PreTouched => {
            let len = c_dev.len();
            // ページ事前タッチ: `vec![0.0f32; len]` はゼロクリアであり
            // glibc の `calloc` 経路が mmap で既にゼロ化済みのページを
            // 返す最適化を行う可能性があるため、`black_box` 越しに
            // 非ゼロ書き込みでページフォールトを強制する（LLVM が
            // 書き込みループを消し去らないようにする）。
            let t_touch = Instant::now();
            let mut dst = vec![0.0f32; len];
            for x in dst.iter_mut() {
                *x = std::hint::black_box(1.0_f32);
            }
            let touch_secs = t_touch.elapsed().as_secs_f64();

            let t = Instant::now();
            stream
                .memcpy_dtoh(&c_dev, &mut dst)
                .expect("D2H download (pre-touched Vec) must succeed");
            // `memcpy_dtoh` は `cuMemcpyDtoHAsync` を発行するだけで返る
            // （Fresh 分岐と同じ理由。上記コメント参照）。直後の drop が
            // ホスト destination（`dst`）を解放する前に転送完了を確定
            // させないと、GB10 実機で転送未完了のうちに解放されうる
            // （V2 固有の安全性問題として PR #973 レビューで指摘）。
            stream
                .synchronize()
                .expect("stream synchronize after D2H must succeed before host buffer is freed");
            let d2h_secs = t.elapsed().as_secs_f64();
            let out_len = dst.len();
            let t = Instant::now();
            drop(dst);
            let free_host_secs = t.elapsed().as_secs_f64();
            // touch_secs はホスト側ページフォールトを事前に強制した分の
            // コストであり、`h2d_a_secs` 等と同列の GPU 転送区間ではない
            // ため個別の集計対象へは含めず、d2h_secs には計上しない
            // （転送そのものの純粋な所要時間を分離する目的）。free_host
            // 側に加算して「V2 全体の追加ホスト側コスト」として記録する。
            (d2h_secs, free_host_secs + touch_secs, out_len)
        }
    };
    assert_eq!(
        out_len,
        (n as usize) * (n as usize),
        "downloaded C length must equal N*N for a square GEMM"
    );

    PhaseSample {
        h2d_a_secs,
        h2d_b_secs,
        alloc_c_secs,
        launch_sync_secs,
        d2h_secs,
        free_host_secs,
    }
}

fn run_phase_breakdown_for_variant(label: &str, variant: DownloadVariant) {
    let device = cached_device(0)
        .expect("CUDA device must be available on the ignored diagnostic bench runner");
    let gemm =
        cached_gemm(&device).expect("CudaGemm construction (via context_cache) must succeed");

    println!("=== fresh GEMM D2H フェーズ分解: variant={label} (イシュー #956) ===");
    for &n in &SIZES {
        let (a, b) = gen_square_ab(0x956_0000 ^ (n as u64), n);
        let n_u32 = n as u32;
        let mut keep_alive: Vec<Vec<f32>> = Vec::new();

        for _ in 0..WARMUP_TRIALS {
            let _ =
                measure_one_phase_trial(&device, &gemm, &a, &b, n_u32, variant, &mut keep_alive);
        }

        let mut h2d_a = Vec::with_capacity(MEASURED_TRIALS);
        let mut h2d_b = Vec::with_capacity(MEASURED_TRIALS);
        let mut alloc_c = Vec::with_capacity(MEASURED_TRIALS);
        let mut launch_sync = Vec::with_capacity(MEASURED_TRIALS);
        let mut d2h = Vec::with_capacity(MEASURED_TRIALS);
        let mut free_host = Vec::with_capacity(MEASURED_TRIALS);
        let mut total = Vec::with_capacity(MEASURED_TRIALS);

        for _ in 0..MEASURED_TRIALS {
            let s =
                measure_one_phase_trial(&device, &gemm, &a, &b, n_u32, variant, &mut keep_alive);
            h2d_a.push(s.h2d_a_secs);
            h2d_b.push(s.h2d_b_secs);
            alloc_c.push(s.alloc_c_secs);
            launch_sync.push(s.launch_sync_secs);
            d2h.push(s.d2h_secs);
            free_host.push(s.free_host_secs);
            total.push(
                s.h2d_a_secs
                    + s.h2d_b_secs
                    + s.alloc_c_secs
                    + s.launch_sync_secs
                    + s.d2h_secs
                    + s.free_host_secs,
            );
        }

        println!(" -- N={n} --");
        print_quartiles_ms("(a) H2D A", median_of(&h2d_a));
        print_quartiles_ms("(b) H2D B", median_of(&h2d_b));
        print_quartiles_ms("(c) C 確保", median_of(&alloc_c));
        print_quartiles_ms("(d) launch+synchronize", median_of(&launch_sync));
        print_quartiles_ms("(e) D2H", median_of(&d2h));
        print_quartiles_ms(
            "(f) ホスト出力バッファ解放（+ V2 事前タッチ）",
            median_of(&free_host),
        );
        print_quartiles_ms("(a+b+c+d+e+f) 再構成合計", median_of(&total));

        // keep_alive（V1）は N=4096 でも `MEASURED_TRIALS + WARMUP_TRIALS`
        // 回 × 64 MiB ≒ 832 MiB に収まる（`.claude/rules/security.md`
        // 「資源枯渇」節: 診断テストのメモリ使用量を明記する方針）。
        drop(keep_alive);
    }
}

/// 受け入れ条件 1（内訳の特定）・H1 判別本体。
#[test]
#[ignore = "CUDA 実機（NVRTC 搭載・compute capability 8.0 以上。DGX Spark GB10 想定）必須。#956"]
fn fresh_overhead_diag_v0_fresh_drop_each_trial() {
    run_phase_breakdown_for_variant("V0-fresh(drop each trial)", DownloadVariant::Fresh);
}

/// H3 判別: 解放を抑止した場合に D2H 自体のコストが変わらないかを見る。
#[test]
#[ignore = "CUDA 実機（NVRTC 搭載・compute capability 8.0 以上。DGX Spark GB10 想定）必須。#956"]
fn fresh_overhead_diag_v1_keep_alive() {
    run_phase_breakdown_for_variant("V1-keep_alive(reuse相当)", DownloadVariant::KeepAlive);
}

/// H1 判別: 転送先ページを事前タッチしてからの D2H 時間を見る。
#[test]
#[ignore = "CUDA 実機（NVRTC 搭載・compute capability 8.0 以上。DGX Spark GB10 想定）必須。#956"]
fn fresh_overhead_diag_v2_pre_touched() {
    run_phase_breakdown_for_variant("V2-pre_touched", DownloadVariant::PreTouched);
}

/// H4 の除外確定用参考値: `CudaDeviceProvider::select` 相当
/// （`context_cache::cached_device` 経由。#946 以降は 2 回目以降ヒットする
/// ためこれ自体は軽量なはずだが、本イシューの帰属分析からサイズ非依存
/// である根拠として記録する）。
#[test]
#[ignore = "CUDA 実機（NVRTC 搭載・compute capability 8.0 以上。DGX Spark GB10 想定）必須。#956"]
fn fresh_overhead_diag_g_cached_device_select_reference() {
    // 事前にキャッシュを温める（プロセス内で最初の 1 回だけ cold）。
    let _ = cached_device(0).expect("warmup cached_device(0) must succeed");

    let mut samples = Vec::with_capacity(MEASURED_TRIALS);
    for _ in 0..MEASURED_TRIALS {
        let t = Instant::now();
        let _ = cached_device(0).expect("cached_device(0) must succeed on a warmed cache");
        samples.push(t.elapsed().as_secs_f64());
    }
    println!("=== (g) context_cache::cached_device(0) 参照値（イシュー #956 H4）===");
    print_quartiles_ms("cached_device(0) [warm]", median_of(&samples));
}

/// V3: (c) C 確保フェーズを `crate::pool::CudaAllocator`（イシュー #1020・
/// REQ-14 のサイズクラス別プール）経由へ差し替えた変種。
///
/// V0（`stream.alloc_zeros` 直接呼び出し・毎試行フルコスト）との対比で、
/// プールが (c) フェーズにどの程度効くかを実機で確認するための計測点
/// （1 試行目＝プールミス〈初回確保〉と、2 試行目以降＝プールヒット
/// 〈再利用〉を分離して記録する）。
///
/// **正直な記録（実装計画 AC-3）**: #956/#1025 の N=2048 固有 166 ms は
/// 主にホスト側 `Vec<f32>` 解放（(f3)。上記 `DownloadVariant::Fresh` 分岐
/// の `free_host_secs`）に帰属する仮説であり、本プールが直接効くのは
/// デバイス側 C 確保（(c)。`alloc_c_secs` 相当）のみである。V3 が (c) を
/// 大幅短縮しても、166 ms 全体の解消を意味しない点を実測時に区別して
/// 転記すること（`docs/perf/cuda-fresh-gemm-n2048-overhead-diagnosis.md`
/// §8 記入欄参照）。
#[test]
#[ignore = "CUDA 実機（NVRTC 搭載・compute capability 8.0 以上。DGX Spark GB10 想定）必須。#956/#1020"]
fn fresh_overhead_diag_v3_pooled_output() {
    let device = cached_device(0)
        .expect("CUDA device must be available on the ignored diagnostic bench runner");
    let gemm =
        cached_gemm(&device).expect("CudaGemm construction (via context_cache) must succeed");
    let allocator = crate::context_cache::cached_allocator(&device)
        .expect("CudaAllocator construction (via context_cache) must succeed");

    println!("=== fresh GEMM (c) C 確保フェーズ: V3-pooled（イシュー #1020）===");
    for &n in &SIZES {
        let (a, b) = gen_square_ab(0x1020_0000 ^ (n as u64), n);
        let numel = n * n;

        // ウォームアップ: 1 回目でプールへ確保させ（ミス）、その後
        // プールへ返却する。以後の `MEASURED_TRIALS` 回はプールヒット
        // （再利用）区間として (c) 確保フェーズのみを計測する
        // （GEMM 本体・H2D/D2H は V0〜V2 が既に計測済みのためここでは
        // 重複計測しない。`crate::gemm::CudaGemm::run_tiled_f32`〈本番
        // 経路。既に本イシューでプール接続済み〉を素通しで 1 回呼び、
        // 「プール経由の C 確保が GEMM 実行と組み合わせても壊れない」
        // ことを合わせて確認する）。
        let t = Instant::now();
        let _ = gemm
            .run_tiled_f32(&a, &b, n as u32, n as u32, n as u32)
            .expect("warmup run_tiled_f32 (pooled output, via production path) must succeed");
        let miss_secs = t.elapsed().as_secs_f64();

        let mut hit_samples = Vec::with_capacity(MEASURED_TRIALS);
        for _ in 0..MEASURED_TRIALS {
            // `alloc_zeroed_f32` 単体（(c) フェーズのみ）を計測する
            // （`run_tiled_f32` 内部の H2D／launch／D2H は計測対象外）。
            let t = Instant::now();
            let handle = allocator
                .alloc_zeroed_f32(numel)
                .expect("pooled allocation must succeed");
            let alloc_secs = t.elapsed().as_secs_f64();
            hit_samples.push(alloc_secs);
            // `handle` はここで drop されプールへ返却される（次イテレー
            // ションの `alloc_zeroed_f32` がヒットする）。
            drop(handle);
        }

        println!("--- N={n} ---");
        println!(
            "  run_tiled_f32 [pool miss on first C alloc, warmup]: {:.3} ms",
            miss_secs * 1000.0
        );
        print_quartiles_ms("  (c) alloc_zeroed_f32 [pool hit]", median_of(&hit_samples));
        println!("  pool stats after N={n}: {:?}", allocator.stats());
    }

    // 実機セッションでの後片付け（次の変種・次のテスト実行への持ち越しを
    // 避ける）。`release_cached` 自体の実測（フェーズ (i)〜(iv)）は
    // `crates/backend-cuda/src/pool.rs` の `#[cfg(test)]` フォールト注入
    // テストが GPU 非依存ロジックとして既にカバーしており、本箇所は
    // 実機での通し確認のみを目的とする。
    let freed_bytes = allocator
        .release_cached()
        .expect("release_cached must succeed after V3 measurement loop");
    println!("=== release_cached: freed_bytes={freed_bytes} ===");
}
