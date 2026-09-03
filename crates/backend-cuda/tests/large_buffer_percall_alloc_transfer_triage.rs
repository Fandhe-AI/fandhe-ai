//! 大容量バッファ per-call アロケーション＋転送の閾値サイズ・二峰性の
//! 発生条件を GB10 実機で特定する診断専用テスト（イシュー #1146。
//! `docs/perf/cuda-wmma-f16-perf-triage.md` §3.3〜§4.2（#1123）で観測した
//! dim4096（A/B/C いずれも 32 MiB・f16）の「転送のみ」計測が dim2048
//! （8 MiB）比で約 580 倍に跳ね、`CudaMmaGemm::run_f16` 合算計測も 1 回目
//! 0.0185 s／2 回目 0.260 s の二峰性を示した外れ値の切り分けを、#1130 から
//! 引き継ぐ）。
//!
//! ## 位置づけ
//!
//! `wmma_f16_opt_perf_triage.rs`（#1123）は「カーネル単体 vs 転送込み vs
//! 転送のみ」の 3 系統でカーネル起因か転送・アロケーション起因かを
//! 切り分けたが、後者と分かった時点で止まっている。本ファイルは転送・
//! アロケーション側をさらに **P0〜P6 のフェーズへ分解**し、**サイズを
//! 5 段階以上スイープ**して閾値サイズを特定する（受け入れ条件 1〜2）。
//!
//! - P0: 参照。`transfer_only_measurement`（`wmma_f16_opt_perf_triage.rs`
//!   と同一実装。clone_htod×2・alloc_zeros・synchronize・clone_dtoh の合算）
//! - P1: デバイス確保＋解放のみ（コピー・memset なし。H-B: `cuMemAllocAsync`
//!   プールのトリム／unified memory ページマッピング仮説）
//! - P2: `alloc_zeros` のみ（memset を含む。H-B）
//! - P3: H2D のみ（`clone_htod`。H-C: ページャブル転送のステージング仮説）
//! - P4: D2H のみ・宛先が毎回新規 `Vec`（`clone_dtoh`。H-A: 未タッチページ
//!   仮説。`docs/perf/cuda-fresh-gemm-n2048-overhead-diagnosis.md` §6.2 の
//!   「未タッチ `Vec` への D2H が 539 ms」の再現確認）
//! - P5: D2H のみ・宛先が事前タッチ済みの再利用 `Vec`（H-A の対照。同 §6.4
//!   の「事前タッチ済み `Vec` への D2H は 1.15 ms」の再現確認）
//! - P6: ホスト `Vec` 確保＋全ページタッチ＋解放のみ（GPU 非関与。H-A の
//!   純ホスト成分・glibc mmap しきい値の寄与を切り分ける）
//!
//! P4 と P5 の差が閾値を持てば H-A、P1／P2 が閾値を持てば H-B、P3 が
//! 閾値を持てば H-C、と判定できる。P0 は他フェーズの和と突き合わせ、
//! 合算計測（#1123 の観測値）が説明できるかを確認する。
//!
//! ## 二峰性の可視化
//!
//! 中央値のみでは 2 モードの混在を見落とすため、`min`／`median`／`q1`／
//! `q3`／`max` に加え `slow_count`（サンプルが `min` の [`SLOW_FACTOR`] 倍を
//! 超えた回数。[`count_slow_samples`] として純関数切り出し・非 ignore
//! 単体テスト付き）を記録する。各サイズにつき **同一プロセス内で連続 3 回**
//! （[`RUNS_PER_SIZE`]。受け入れ条件 2）計測し、ラン間の乖離（1 回目 vs
//! 2 回目以降）を突き合わせる。加えて `warmup` なしの cold 1 回目
//! （`Instant` 直接計測）を別途記録し、プロセス起動直後の値も残す。
//!
//! サイズ走査順は昇順・降順の 2 パスで行い、「直前に確保・解放した
//! サイズ」への順序依存の有無を確認する（受け入れ条件 3「呼び出し順序
//! 依存」）。総実行時間短縮のため、フル P0〜P6 は昇順パスのみで確保し、
//! 降順パスは P0（参照）・P4／P5（H-A 判定に必要な最小集合）・P6 に
//! 絞る（12 サイズ×2 順×3 ラン×7 フェーズ×40 iters の全組合せは、
//! dim>=32 MiB 帯の病的モード〈約 260 ms〉を踏むと数時間規模になるため。
//! 昇順パスの結果と突き合わせれば順序依存の有無は判定できる）。
//!
//! ## driver プール状態の事実記録
//!
//! `has_async_alloc` が真の環境では、スイープ前後で driver プール
//! （`cuMemAllocAsync` が内部的に使うプール）の `RESERVED_MEM_CURRENT`／
//! `USED_MEM_CURRENT` を `cuMemPoolGetAttribute` で読み取る（**読み取り
//! のみ**。`set_attribute` は呼ばない。閾値変更の A/B は #1149 のスコープ）。
//! `pool.rs::CudaAllocator::release_cached` の `cuMemPoolTrimTo` 呼び出しと
//! 同じ事前条件（`ctx.cu_device()` は `CudaContext::new` 済みの有効な
//! デバイスハンドル）に基づく。
//!
//! ## 対策コードを含まない
//!
//! 本ファイルは**計測・記録専用**（イシュー #1146 の受け入れ条件）。
//! `crates/backend-cuda/src/**` のプロダクションコード・ディスパッチ規則・
//! tolerance は変更しない。対策（release threshold 変更・プール経由化）は
//! #1149／#1153 のスコープ（`.claude/rules/coding-rust.md`「テスト・ベンチ」
//! 節・`docs/perf/` 診断ドキュメント群と同じ「診断は受け入れ判定に使わない」
//! 方針）。
//!
//! ## 実機前提
//!
//! `wmma_f16_opt_perf_triage.rs` と同様、通常 CI（GitHub ホステッド・CUDA
//! 実機なし）では実行されない `#[ignore]` 分離テスト。本ファイルは公開
//! API（`CudaDevice::stream`／`CudaDevice::context`・`cudarc` の公開
//! `driver::result` 関数群）のみを使うため `internal-diagnostics` feature
//! は不要（`Cargo.toml` の `[[test]]` 追加なしで `cargo test --workspace`
//! に自然に含まれる）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda \
//!     --test large_buffer_percall_alloc_transfer_triage \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` は同一 GPU 上での計測競合を避けるための前提
//! （`wmma_f16_opt_perf_triage.rs` 等、既存の実機診断テストと同じ規約）。

use std::mem::size_of;
use std::time::Instant;

use bench_harness::MeasurementConfig;
use bench_harness::rng::Xorshift64Star;
use cudarc::driver::CudaContext;
use fandhe_ai_backend_cuda::CudaDevice;
use half::f16;

/// スイープ対象のバッファ単体サイズ（MiB）。glibc の 64-bit 既定 mmap
/// しきい値（動的上限 32 MiB。`M_MMAP_THRESHOLD`）を跨ぐ境界を 1 MiB
/// 刻みで密に取り、受け入れ条件「5 段階以上」を満たす（12 段階）。
const SIZES_MIB: [u64; 12] = [8, 16, 24, 28, 30, 31, 32, 33, 36, 40, 48, 64];

/// 各サイズにつき同一プロセス内で連続実行するラン数（受け入れ条件 2）。
const RUNS_PER_SIZE: usize = 3;

/// `count_slow_samples` が「遅い」と判定する倍率（サンプル値が `min` の
/// この倍を超えたら遅いサンプルとしてカウントする）。二峰性の 2 モード
/// （#1123 の観測では約 580 倍の開き）を十分に捉えつつ、通常のシステム
/// ジッタ（数倍程度）を誤検出しない値として 10 倍を採用する。
const SLOW_FACTOR: f64 = 10.0;

/// サイズ走査順（受け入れ条件 3「呼び出し順序依存」の確認軸）。
#[derive(Debug, Clone, Copy)]
enum SweepOrder {
    Ascending,
    Descending,
}

impl SweepOrder {
    fn label(self) -> &'static str {
        match self {
            SweepOrder::Ascending => "asc",
            SweepOrder::Descending => "desc",
        }
    }

    /// `SIZES_MIB` を本順序で並べたベクタを返す。
    fn ordered_sizes(self) -> Vec<u64> {
        let mut sizes = SIZES_MIB.to_vec();
        if matches!(self, SweepOrder::Descending) {
            sizes.reverse();
        }
        sizes
    }
}

/// `mib` MiB を `f16` 要素数へ変換する。`mib * 1024 * 1024 /
/// size_of::<f16>()` を `checked_mul`／`checked_div` で導出し、
/// `usize` オーバーフロー時は `None` を返す（外部入力ではなく定数配列
/// からの導出だが、`memory.rs::checked_byte_len` と同じ fail-closed
/// 方針を踏襲する。`.claude/rules/security.md` A03 節）。
fn numel_for_mib(mib: u64) -> Option<usize> {
    let bytes = mib.checked_mul(1024)?.checked_mul(1024)?;
    let elem_size = size_of::<f16>() as u64;
    let numel_u64 = bytes.checked_div(elem_size)?;
    usize::try_from(numel_u64).ok()
}

/// `samples`（秒）のうち `min` の `factor` 倍を超えたサンプル数を返す。
/// 中央値だけでは埋もれる二峰性（一部サンプルだけが極端に遅いモード）を
/// 定量化するための純関数。
fn count_slow_samples(samples: &[f64], factor: f64) -> usize {
    let Some(&min) = samples
        .iter()
        .min_by(|a, b| a.partial_cmp(b).expect("samples must not be NaN"))
    else {
        return 0;
    };
    if min <= 0.0 {
        // min が 0 秒相当（計測不能な短時間）の場合は倍率判定が無意味な
        // ため保守的に 0 を返す（誤検出よりも「判定不能」を優先する）。
        return 0;
    }
    samples.iter().filter(|&&s| s > min * factor).count()
}

/// [`bench_harness::Measurement`] から本テストが出力する要約統計
/// （min／median／q1／q3／max／slow_count）を導出する。
struct Summary {
    min_secs: f64,
    median_secs: f64,
    q1_secs: f64,
    q3_secs: f64,
    max_secs: f64,
    slow_count: usize,
}

fn summarize(measurement: &bench_harness::Measurement) -> Summary {
    let min_secs = measurement
        .samples_secs
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_secs = measurement
        .samples_secs
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    Summary {
        min_secs,
        median_secs: measurement.median_secs,
        q1_secs: measurement.q1_secs,
        q3_secs: measurement.q3_secs,
        max_secs,
        slow_count: count_slow_samples(&measurement.samples_secs, SLOW_FACTOR),
    }
}

fn print_summary_row(
    phase: &str,
    size_mib: u64,
    order: &str,
    run_idx: usize,
    cold_ms: f64,
    s: &Summary,
) {
    println!(
        "{phase},{size_mib},{order},{run_idx},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
        cold_ms,
        s.min_secs * 1000.0,
        s.q1_secs * 1000.0,
        s.median_secs * 1000.0,
        s.q3_secs * 1000.0,
        s.max_secs * 1000.0,
        s.slow_count,
    );
}

/// driver プールの `RESERVED_MEM_CURRENT`／`USED_MEM_CURRENT`（`u64`。
/// `cuuint64_t`）を読み取る。`has_async_alloc` が偽の環境では driver
/// プール自体が存在しないため `None` を返す（`pool.rs::release_cached`
/// と同じ判断）。
fn read_mem_pool_usage(ctx: &CudaContext) -> Option<(u64, u64)> {
    if !ctx.has_async_alloc() {
        return None;
    }
    // SAFETY: `ctx.cu_device()` は `CudaContext::new` が既に構築済みの
    // 有効なデバイスハンドルを返す。`get_mem_pool` の戻り値をそのまま
    // `get_attribute` へ渡すため「`get` から得た有効な pool」という
    // 事前条件を満たす（`pool.rs::release_cached` の SAFETY コメントと
    // 同一根拠）。`get_attribute` は読み取り専用（`set_attribute` は
    // 呼ばない。閾値変更を伴わない診断専用のため）。
    unsafe {
        let mem_pool = cudarc::driver::result::device::get_mem_pool(ctx.cu_device()).ok()?;
        let mut reserved: u64 = 0;
        cudarc::driver::result::mem_pool::get_attribute(
            mem_pool,
            cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            (&mut reserved as *mut u64).cast(),
        )
        .ok()?;
        let mut used: u64 = 0;
        cudarc::driver::result::mem_pool::get_attribute(
            mem_pool,
            cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            (&mut used as *mut u64).cast(),
        )
        .ok()?;
        Some((reserved, used))
    }
}

fn print_pool_usage(label: &str, usage: Option<(u64, u64)>) {
    match usage {
        Some((reserved, used)) => {
            println!("mem_pool_usage,{label},reserved_bytes={reserved},used_bytes={used}")
        }
        None => println!("mem_pool_usage,{label},unavailable(has_async_alloc=false)"),
    }
}

/// `workload` を 1 回だけ `Instant` で計測してから（cold・warmup なしの
/// 文字どおり最初の 1 回）、同じクロージャを [`bench_harness::run`] に渡して
/// 20/20 warmup+計測を行う。`FnMut` はクロージャを消費しないため、cold の
/// 1 回目呼び出し後にそのまま渡し直せる（呼び出し 1 回分を余計に増やす
/// だけで、cold と warm 計測の二重ベンチマーク化を避ける。Review 指摘:
/// 以前の実装は cold 測定に `phase_pX` 関数〈内部で 40 回実行するフル
/// ベンチマーク〉をそのまま渡していたため `cold_ms` が「40 回分の壁時計
/// 時間」になっていた）。
///
/// 戻り値は `(cold_ms, Measurement)`。
fn measure_with_cold<F: FnMut()>(
    config: &MeasurementConfig,
    mut workload: F,
) -> (f64, bench_harness::Measurement) {
    let cold_start = Instant::now();
    workload();
    let cold_ms = cold_start.elapsed().as_secs_f64() * 1000.0;
    let measurement = bench_harness::run(config, workload)
        .expect("phase measurement must satisfy TASK-8.1 protocol");
    (cold_ms, measurement)
}

/// P0: 参照計測（転送のみ合算。`wmma_f16_opt_perf_triage.rs::
/// transfer_only_measurement` と同一実装）。
fn phase_p0_transfer_only(
    device: &CudaDevice,
    config: &MeasurementConfig,
    a: &[f16],
    b: &[f16],
    out_len: usize,
) -> (f64, bench_harness::Measurement) {
    measure_with_cold(config, || {
        let a_dev = device
            .stream()
            .clone_htod(a)
            .expect("clone_htod must succeed on CUDA-equipped test runner");
        let b_dev = device
            .stream()
            .clone_htod(b)
            .expect("clone_htod must succeed on CUDA-equipped test runner");
        let c_dev = device
            .stream()
            .alloc_zeros::<f16>(out_len)
            .expect("alloc_zeros must succeed on CUDA-equipped test runner");
        device
            .stream()
            .synchronize()
            .expect("synchronize must succeed on CUDA-equipped test runner");
        let _c_host = device
            .stream()
            .clone_dtoh(&c_dev)
            .expect("clone_dtoh must succeed on CUDA-equipped test runner");
        // P4／P5 と同じ理由（Cursor Bugbot 指摘 discussion_r3921118928）で、
        // `clone_dtoh` 完了前に `_c_host` が閉包末尾で暗黙 drop されるのを
        // 防ぐため明示的に synchronize する。
        device
            .stream()
            .synchronize()
            .expect("synchronize after D2H must succeed before host buffer is freed");
        drop(a_dev);
        drop(b_dev);
    })
}

/// P1: デバイス確保＋解放のみ（コピー・memset なし）。
///
/// codex-review 指摘（PR #1169 discussion_r3921097911）: `cuMemAllocAsync`
/// はストリームへ enqueue するだけで返る非同期 API のため、`synchronize`
/// なしで計測すると「ホスト側 enqueue 時間」しか測れず、H-B（デバイス側
/// 確保が原因）を棄却する根拠にならない。確保完了・解放完了それぞれを
/// 明示的に待ってから計測を終える（`fresh_overhead_diag_tests.rs` の
/// D2H フェーズと同じ「非同期 API の後には synchronize で完了を確定
/// させる」方針）。
fn phase_p1_alloc_only(
    device: &CudaDevice,
    config: &MeasurementConfig,
    numel: usize,
) -> (f64, bench_harness::Measurement) {
    measure_with_cold(config, || {
        // SAFETY: 確保した領域は読まず drop するのみ（未初期化領域への
        // アクセスなし）。`pool.rs::CudaAllocator::try_alloc_from_stream`
        // の `stream.alloc::<f32>` と同じ「境界検査なしの alloc」を、
        // 出力の書き込みなしで確保・解放コストのみ測る目的で使用する。
        let buf = unsafe {
            device
                .stream()
                .alloc::<f16>(numel)
                .expect("alloc must succeed on CUDA-equipped test runner")
        };
        device
            .stream()
            .synchronize()
            .expect("synchronize after alloc must succeed to measure allocation completion");
        drop(buf);
        // `cuMemFreeAsync` も非同期 enqueue のため、解放完了まで待つ
        // （alloc 側と対称に扱い、本フェーズの意味を「確保完了＋解放完了
        // の合算」に統一する）。
        device
            .stream()
            .synchronize()
            .expect("synchronize after free must succeed to measure free completion");
    })
}

/// P2: `alloc_zeros` のみ（memset を含む）。P1 と同じ理由（上記コメント）
/// で `alloc_zeros`（`cuMemAllocAsync` + memset の非同期 enqueue）・解放
/// それぞれの完了を待ってから計測を終える。
fn phase_p2_alloc_zeros(
    device: &CudaDevice,
    config: &MeasurementConfig,
    numel: usize,
) -> (f64, bench_harness::Measurement) {
    measure_with_cold(config, || {
        let buf = device
            .stream()
            .alloc_zeros::<f16>(numel)
            .expect("alloc_zeros must succeed on CUDA-equipped test runner");
        device
            .stream()
            .synchronize()
            .expect("synchronize after alloc_zeros must succeed to measure completion");
        drop(buf);
        device
            .stream()
            .synchronize()
            .expect("synchronize after free must succeed to measure free completion");
    })
}

/// P3: H2D のみ（`clone_htod`。ホスト側ソースは計測外で確保・書き込み
/// 済みの `host_src` を使う）。
fn phase_p3_h2d_only(
    device: &CudaDevice,
    config: &MeasurementConfig,
    host_src: &[f16],
) -> (f64, bench_harness::Measurement) {
    measure_with_cold(config, || {
        let dev = device
            .stream()
            .clone_htod(host_src)
            .expect("clone_htod must succeed on CUDA-equipped test runner");
        device
            .stream()
            .synchronize()
            .expect("synchronize must succeed on CUDA-equipped test runner");
        drop(dev);
    })
}

/// P4: D2H のみ・宛先が毎回新規 `Vec`（未タッチページ仮説の再現確認）。
/// `device_src` は常駐バッファとして計測外で確保する。
///
/// Cursor Bugbot 指摘（PR #1169 discussion_r3921118928）: `clone_dtoh` は
/// `cuMemcpyDtoHAsync` を発行するだけで返り、plain `Vec<T>` は
/// `HostSlice::stream_synced_mut_slice` が `SyncOnDrop::Sync(None)` を
/// 返すため drop 時の暗黙 sync も無い（`cudarc-0.19.8` 実装。
/// `fresh_overhead_diag_tests.rs` の Fresh 分岐と同じ根拠）。synchronize
/// なしで直後に drop すると、DMA 転送が進行中のままホスト宛先が解放
/// される競合になりうる（GB10 実機で顕在化しうる）。転送完了を待って
/// から drop する。
fn phase_p4_d2h_fresh_vec(
    device: &CudaDevice,
    config: &MeasurementConfig,
    device_src: &cudarc::driver::CudaSlice<f16>,
) -> (f64, bench_harness::Measurement) {
    measure_with_cold(config, || {
        let host = device
            .stream()
            .clone_dtoh(device_src)
            .expect("clone_dtoh must succeed on CUDA-equipped test runner");
        device
            .stream()
            .synchronize()
            .expect("synchronize after D2H must succeed before host buffer is freed");
        drop(host);
    })
}

/// P5: D2H のみ・宛先が事前タッチ済みの再利用 `Vec`（P4 の対照。
/// `docs/perf/cuda-fresh-gemm-n2048-overhead-diagnosis.md` §6.4 の V2 と
/// 同型）。`cudarc` の同期 D2H 相当を `memcpy_dtoh` で行う。
///
/// P4 と同じ理由（上記コメント）で `memcpy_dtoh` 後に synchronize する。
/// `host_buf` は次イテレーションでも再利用されるため、転送完了前に
/// 次回の `memcpy_dtoh` が同じバッファへ再突入するデータ競合を防ぐ
/// 意味でも同期が必須。
fn phase_p5_d2h_reused_vec(
    device: &CudaDevice,
    config: &MeasurementConfig,
    device_src: &cudarc::driver::CudaSlice<f16>,
    numel: usize,
) -> (f64, bench_harness::Measurement) {
    // 計測外でホスト側バッファを 1 回だけ確保・全ページタッチする
    // （P4 の「毎回新規 Vec」との差分がページタッチ由来であることを
    // 保証するため、確保・タッチ自体は計測ループの外に置く）。
    let mut host_buf: Vec<f16> = vec![f16::ZERO; numel];
    measure_with_cold(config, || {
        device
            .stream()
            .memcpy_dtoh(device_src, &mut host_buf)
            .expect("memcpy_dtoh must succeed on CUDA-equipped test runner");
        device
            .stream()
            .synchronize()
            .expect("synchronize after D2H must succeed before host buffer is reused");
    })
}

/// P6: ホスト `Vec` 確保＋全ページタッチ＋解放のみ（GPU 非関与）。
/// H-A の純ホスト成分（glibc mmap しきい値の寄与）を切り分ける。
fn phase_p6_host_vec_touch_only(
    config: &MeasurementConfig,
    numel: usize,
) -> (f64, bench_harness::Measurement) {
    measure_with_cold(config, || {
        let v = std::hint::black_box(vec![f16::ZERO; numel]);
        drop(v);
    })
}

/// 固定サイズ比較（合計転送サイズの単体閾値 vs 合計閾値の切り分け。
/// 受け入れ条件 3「発生条件」の一部）。`(16 MiB ×3 = 48 MiB 合計)` vs
/// `(48 MiB ×1)` vs `(24 MiB ×2)` を H2D（htod×N・synchronize）と D2H
/// （dtoh×N・synchronize）の両方向で比較する。
///
/// codex-review 指摘（PR #1169 discussion_r3921097920）: 以前の実装は
/// H2D（`clone_htod`）の繰り返しのみを計測しており、D2H（P0／P4 相当。
/// `alloc_zeros`／D2H）を含まないため、`docs/perf/
/// cuda-large-buffer-percall-alloc-transfer-threshold.md` の「閾値は
/// 単体バッファサイズに働き合計転送サイズには働かない」という結論
/// （§4.5・§6 の 2）の根拠に D2H 側が欠けていた。P4（`phase_p4_d2h_
/// fresh_vec`）と同じ「宛先が毎回新規 `Vec`」の未タッチページ方式で、
/// 新規 D2H 宛先を用いた対照実験を追加し、単体サイズのみ変えて D2H
/// 総量を固定する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 想定）必須。イシュー #1146 の診断専用テストで受け入れ判定には使わない"]
fn large_buffer_percall_fixed_total_size_comparison() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    println!(
        "environment: name={:?} compute_capability={:?} arch={:?}",
        device.name(),
        device.compute_capability(),
        device.arch()
    );

    let config = MeasurementConfig::new(20, 20).expect("20/20 must satisfy TASK-8.1 minimums");

    println!(
        "fixed_total_comparison,phase,label,buffer_count,per_buffer_mib,total_mib,min_ms,median_ms,max_ms,slow_count"
    );

    // 各ケースは複数バッファの H2D（`clone_htod`）または D2H
    // （`clone_dtoh`）を連続実行した合算を計測する。3 ケースとも合計
    // 転送量は 48 MiB で固定し、バッファの本数・単体サイズだけを変える
    // （「単体バッファサイズが閾値を支配するのか、合計転送量が支配する
    // のか」を切り分ける。Review 指摘: 以前の実装は `c_mib == 0` を
    // `a_numel.max(b_numel)` へフォールバックしていたため、`24x2` ケース
    // の実効合計が 72 MiB・`48x1` が 96 MiB になり、ラベルと実合計が
    // 食い違っていた）。
    let cases: [(&str, &[u64]); 3] = [
        ("16x3_total48", &[16, 16, 16]),
        ("24x2_total48", &[24, 24]),
        ("48x1_total48", &[48]),
    ];

    for (label, buffer_sizes_mib) in cases {
        let total_mib: u64 = buffer_sizes_mib.iter().sum();
        debug_assert_eq!(
            total_mib, 48,
            "fixed-total comparison must hold the total at 48 MiB"
        );

        let buffers: Vec<Vec<f16>> = buffer_sizes_mib
            .iter()
            .enumerate()
            .map(|(idx, &mib)| {
                let numel = numel_for_mib(mib).expect("buffer size must fit usize");
                let mut rng = Xorshift64Star::new(0xF000 ^ (mib << 8) ^ (idx as u64));
                rng.fill_vec_f16(numel)
            })
            .collect();

        // --- H2D 側（既存） ---
        let measurement_h2d = bench_harness::run(&config, || {
            let mut dev_bufs = Vec::with_capacity(buffers.len());
            for host_buf in &buffers {
                dev_bufs.push(
                    device
                        .stream()
                        .clone_htod(host_buf)
                        .expect("clone_htod must succeed on CUDA-equipped test runner"),
                );
            }
            device
                .stream()
                .synchronize()
                .expect("synchronize must succeed on CUDA-equipped test runner");
            drop(dev_bufs);
        })
        .expect("fixed-total-size H2D comparison measurement must satisfy TASK-8.1 protocol");

        let s_h2d = summarize(&measurement_h2d);
        println!(
            "fixed_total_comparison,h2d,{label},{},{},{total_mib},{:.4},{:.4},{:.4},{}",
            buffer_sizes_mib.len(),
            buffer_sizes_mib[0],
            s_h2d.min_secs * 1000.0,
            s_h2d.median_secs * 1000.0,
            s_h2d.max_secs * 1000.0,
            s_h2d.slow_count,
        );

        // --- D2H 側（新規。codex-review 指摘対応） ---
        // デバイス側ソースは計測外で確保する（H2D 転送時間を D2H 計測へ
        // 混入させないため。P4 の `device_src` 常駐方式と同じ）。
        let device_srcs: Vec<cudarc::driver::CudaSlice<f16>> = buffers
            .iter()
            .map(|host_buf| {
                device.stream().clone_htod(host_buf).expect(
                    "clone_htod (device_srcs setup) must succeed on CUDA-equipped test runner",
                )
            })
            .collect();
        device.stream().synchronize().expect(
            "synchronize after device_srcs setup must succeed on CUDA-equipped test runner",
        );

        let measurement_d2h = bench_harness::run(&config, || {
            let mut host_bufs = Vec::with_capacity(device_srcs.len());
            for dev_buf in &device_srcs {
                host_bufs.push(
                    device
                        .stream()
                        .clone_dtoh(dev_buf)
                        .expect("clone_dtoh must succeed on CUDA-equipped test runner"),
                );
            }
            // P4／P5 と同じ理由（Cursor Bugbot 指摘 discussion_r3921118928）
            // で、`clone_dtoh` 完了前に `host_bufs` が drop されるのを防ぐ
            // ため明示的に synchronize する。
            device
                .stream()
                .synchronize()
                .expect("synchronize after D2H must succeed before host buffers are freed");
            drop(host_bufs);
        })
        .expect("fixed-total-size D2H comparison measurement must satisfy TASK-8.1 protocol");

        let s_d2h = summarize(&measurement_d2h);
        println!(
            "fixed_total_comparison,d2h,{label},{},{},{total_mib},{:.4},{:.4},{:.4},{}",
            buffer_sizes_mib.len(),
            buffer_sizes_mib[0],
            s_d2h.min_secs * 1000.0,
            s_d2h.median_secs * 1000.0,
            s_d2h.max_secs * 1000.0,
            s_d2h.slow_count,
        );

        drop(device_srcs);
    }
}

/// 本体: サイズスイープ × フェーズ分解 × 走査順 × 連続 3 ラン。
///
/// 出力 1 行の書式:
/// `phase,size_mib,order,run_idx,cold_ms,min_ms,q1_ms,median_ms,q3_ms,max_ms,slow_count`
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 想定）必須。イシュー #1146 の診断専用テストで受け入れ判定には使わない"]
fn large_buffer_percall_alloc_transfer_triage_record() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    println!(
        "environment: name={:?} compute_capability={:?} arch={:?} has_async_alloc={}",
        device.name(),
        device.compute_capability(),
        device.arch(),
        device.context().has_async_alloc(),
    );

    print_pool_usage("before_sweep", read_mem_pool_usage(device.context()));

    let config = MeasurementConfig::new(20, 20).expect("20/20 must satisfy TASK-8.1 minimums");

    println!("phase,size_mib,order,run_idx,cold_ms,min_ms,q1_ms,median_ms,q3_ms,max_ms,slow_count");

    for order in [SweepOrder::Ascending, SweepOrder::Descending] {
        for size_mib in order.ordered_sizes() {
            let numel = numel_for_mib(size_mib)
                .unwrap_or_else(|| panic!("size {size_mib} MiB must fit usize"));

            let mut rng = Xorshift64Star::new(0xE000 ^ size_mib);
            let a: Vec<f16> = rng.fill_vec_f16(numel);
            let b: Vec<f16> = rng.fill_vec_f16(numel);

            // 降順パスは実行時間短縮のため P0／P4／P5／P6（H-A の判定に
            // 必要な最小集合＋参照計測）のみに絞る。順序依存の有無は
            // これらのフェーズだけで昇順パスと突き合わせられる。フル
            // フェーズ集合（P0〜P6）は昇順パスで確保する（advisor 指摘:
            // 12 サイズ×2 順×3 ラン×7 フェーズ×40 iters は dim>=32 MiB
            // 帯の病的モード〈約 260 ms〉を踏むと数時間規模になるため、
            // 受け入れ条件を満たしたまま総実行時間を縮小する）。
            let full_phase_set = matches!(order, SweepOrder::Ascending);

            for run_idx in 0..RUNS_PER_SIZE {
                // P0（参照・転送のみ合算）
                let (cold, m) = phase_p0_transfer_only(&device, &config, &a, &b, numel);
                print_summary_row(
                    "p0_transfer_only",
                    size_mib,
                    order.label(),
                    run_idx,
                    cold,
                    &summarize(&m),
                );

                if full_phase_set {
                    // P1（確保＋解放のみ）
                    let (cold, m) = phase_p1_alloc_only(&device, &config, numel);
                    print_summary_row(
                        "p1_alloc_only",
                        size_mib,
                        order.label(),
                        run_idx,
                        cold,
                        &summarize(&m),
                    );

                    // P2（alloc_zeros のみ）
                    let (cold, m) = phase_p2_alloc_zeros(&device, &config, numel);
                    print_summary_row(
                        "p2_alloc_zeros",
                        size_mib,
                        order.label(),
                        run_idx,
                        cold,
                        &summarize(&m),
                    );

                    // P3（H2D のみ）
                    let (cold, m) = phase_p3_h2d_only(&device, &config, &a);
                    print_summary_row(
                        "p3_h2d_only",
                        size_mib,
                        order.label(),
                        run_idx,
                        cold,
                        &summarize(&m),
                    );
                }

                // P4／P5 は常駐デバイスバッファを 1 本用意して使い回す
                // （計測外で確保。D2H 側の差分のみを見るため）。
                let device_src = device
                    .stream()
                    .clone_htod(&a)
                    .expect("clone_htod must succeed on CUDA-equipped test runner");
                device
                    .stream()
                    .synchronize()
                    .expect("synchronize must succeed on CUDA-equipped test runner");

                let (cold, m) = phase_p4_d2h_fresh_vec(&device, &config, &device_src);
                print_summary_row(
                    "p4_d2h_fresh_vec",
                    size_mib,
                    order.label(),
                    run_idx,
                    cold,
                    &summarize(&m),
                );

                let (cold, m) = phase_p5_d2h_reused_vec(&device, &config, &device_src, numel);
                print_summary_row(
                    "p5_d2h_reused_vec",
                    size_mib,
                    order.label(),
                    run_idx,
                    cold,
                    &summarize(&m),
                );

                drop(device_src);

                // P6（ホスト Vec タッチのみ・GPU 非関与）
                let (cold, m) = phase_p6_host_vec_touch_only(&config, numel);
                print_summary_row(
                    "p6_host_vec_touch_only",
                    size_mib,
                    order.label(),
                    run_idx,
                    cold,
                    &summarize(&m),
                );
            }
        }
    }

    print_pool_usage("after_sweep", read_mem_pool_usage(device.context()));
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn numel_for_mib_computes_f16_element_count() {
        // 1 MiB = 1_048_576 byte / 2 byte(f16) = 524_288 要素。
        assert_eq!(numel_for_mib(1), Some(524_288));
        assert_eq!(numel_for_mib(32), Some(524_288 * 32));
    }

    #[test]
    fn numel_for_mib_rejects_overflow() {
        assert_eq!(numel_for_mib(u64::MAX), None);
    }

    #[test]
    fn count_slow_samples_counts_values_above_factor_times_min() {
        // min=1.0 に対し factor=10 なら 10.0 超過のみカウント（境界の
        // 10.0 ちょうどは超過扱いしない: `>` 判定）。
        let samples = [1.0, 1.5, 2.0, 10.0, 10.1, 260.0];
        assert_eq!(count_slow_samples(&samples, 10.0), 2);
    }

    #[test]
    fn count_slow_samples_returns_zero_for_uniform_samples() {
        let samples = [5.0, 5.1, 4.9, 5.0];
        assert_eq!(count_slow_samples(&samples, 10.0), 0);
    }

    #[test]
    fn count_slow_samples_returns_zero_for_empty_input() {
        assert_eq!(count_slow_samples(&[], 10.0), 0);
    }

    #[test]
    fn sweep_order_descending_reverses_sizes() {
        let asc = SweepOrder::Ascending.ordered_sizes();
        let mut desc = SweepOrder::Descending.ordered_sizes();
        desc.reverse();
        assert_eq!(asc, desc);
        assert_eq!(asc, SIZES_MIB.to_vec());
    }

    #[test]
    fn sizes_mib_has_at_least_five_stages_and_is_sorted() {
        assert!(
            SIZES_MIB.len() >= 5,
            "受け入れ条件: 5 段階以上のバッファサイズを計測する"
        );
        assert!(SIZES_MIB.windows(2).all(|w| w[0] < w[1]));
    }
}
