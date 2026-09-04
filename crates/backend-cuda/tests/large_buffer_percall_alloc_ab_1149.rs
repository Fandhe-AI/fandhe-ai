//! cuMemPool release threshold の変更・`cuMemAlloc` 同期割当への切替に
//! よる per-call アロケーション＋転送病態への効果を GB10 実機で A/B 計測
//! する（イシュー #1149）。
//!
//! ## 位置づけ・引き継ぎ元
//!
//! `large_buffer_percall_alloc_transfer_triage.rs`（イシュー #1146）で
//! 32→33 MiB 帯にデバイス確保・`alloc_zeros`・H2D の約 2.2 倍の段差が
//! 新規観測され（`docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`
//! §4.3・§7）、原因が driver プール（`cuMemAllocAsync` 内部のサイズ
//! クラス・トリム挙動）にあるかは未特定のまま引き継がれた。本ファイルは
//! `docs/backend-cuda-pool-allocator-decision.md` §8 の保留事項
//! （`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` の調整可否）と合わせて、以下 2
//! つの対策候補を実機データで検証する:
//!
//! - **案 A**（[`ReleaseThresholdGuard`]）: driver プールの release
//!   threshold を既定（実測して記録。0 と仮定しない）から `u64::MAX` へ
//!   引き上げ、synchronize 毎の OS 返却（トリム）を抑制する
//! - **案 B**（[`SyncDeviceBuffer`]）: `stream.alloc`／`alloc_zeros` が
//!   内部で使う非同期プール経由の確保（`cuMemAllocAsync`。cudarc 0.19.8
//!   `CudaStream::alloc`〈`core.rs:1530`〉は `ctx.has_async_alloc()` が
//!   真なら常にこの経路を通る）を迂回し、`result::malloc_sync`
//!   （`cuMemAlloc`）＋ `result::free_sync`（`cuMemFree`）による同期割当
//!   に切り替える
//!
//! 条件は「なし（baseline）」「A」「B」「A+B」の 4 通り。各条件で
//! `large_buffer_percall_alloc_transfer_triage.rs` の P0〜P4 相当
//! フェーズを再実行し、32→33 MiB 段差・二峰性が条件間でどう変化するかを
//! `docs/perf/cuda-percall-alloc-pool-threshold-ab.md` の事前登録判定
//! 基準（解消／緩和／効果なし）に照らして判定する。さらに P7 として
//! `CudaMmaGemm`（`gemm_mma.rs`）の本番経路（`upload_f16`→
//! `alloc_output_f16`→`launch_f16`→`download_f16`）を dim4096 相当で
//! 直接レプリカ計測し、#1123 で観測された「転送のみ」約 261〜263 ms を
//! 条件別に比較する。
//!
//! ## 対策コードを含まない・本番非変更
//!
//! 本ファイルは**計測・記録専用**（#1146 と同じ位置づけ）。
//! `crates/backend-cuda/src/**` のプロダクションコードは一切変更しない。
//! 案 A の release threshold 変更はプロセス内の [`ReleaseThresholdGuard`]
//! のスコープに閉じ、Drop で必ず復元する。案 A／B いずれかが有効だった
//! 場合の本番結線判断は #1153 のスコープ（本ファイルでは行わない）。
//!
//! ## 実機前提
//!
//! `large_buffer_percall_alloc_transfer_triage.rs` と同様、通常 CI
//! （GitHub ホステッド・CUDA 実機なし）では実行されない `#[ignore]`
//! 分離テスト。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release \
//!     --test large_buffer_percall_alloc_ab_1149 \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--release` 必須（#1146 の記録値と比較可能な条件を揃えるため）。
//! `--test-threads=1` 必須（release threshold はプロセス全体の driver
//! プール状態を変更するため、他テストとの並行実行はプール状態の競合を
//! 招く）。

use std::mem::size_of;
use std::time::Instant;

use bench_harness::MeasurementConfig;
use bench_harness::rng::Xorshift64Star;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::result;
use cudarc::driver::sys::CUmemPool_attribute;
use fandhe_ai_backend_cuda::CudaDevice;
use fandhe_ai_backend_cuda::CudaMmaGemm;
use half::f16;

/// スイープ対象のバッファ単体サイズ（MiB）。#1146 で特定された 32 MiB
/// 前後の閾値・32→33 MiB 段差を中心に据える（`SIZES_MIB` は同ファイルの
/// 値を全て包含する超集合ではないが、判定に必要な帯を密に取る）。
const SIZES_MIB: [u64; 8] = [24, 28, 31, 32, 33, 36, 48, 64];

/// key サイズ（32・33 MiB）で採用するラン数（規約「5 回計測中央値」を
/// 適用する対象。`.claude/rules/coding-rust.md` テスト・ベンチ節）。
const KEY_RUNS: usize = 5;

/// key サイズ以外で採用するラン数（#1146 と同じ 3 ラン。総実行時間を
/// 抑えつつラン間の乖離は引き続き記録する）。
const OTHER_RUNS: usize = 3;

/// P7（本番経路レプリカ）で採用するラン数。key サイズと同じ扱い。
const P7_RUNS: usize = 5;

/// [`count_slow_samples`] が「遅い」と判定する倍率（#1146 と同一値・同一
/// 根拠。二峰性の 2 モードを捉えつつ通常ジッタを誤検出しない）。
const SLOW_FACTOR: f64 = 10.0;

/// P7 の GEMM 形状（M=N=K=4096。A/B/C いずれも f16 32 MiB。#1123 の
/// 「転送のみ」約 261〜263 ms 観測形状と一致させる）。
const P7_DIM: u32 = 4096;

/// A/B/A+B 計測対象の 4 条件（受け入れ条件: なし／A／B／A+B の全組合せ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Condition {
    Baseline,
    ReleaseThreshold,
    SyncAlloc,
    Both,
}

impl Condition {
    fn label(self) -> &'static str {
        match self {
            Condition::Baseline => "baseline",
            Condition::ReleaseThreshold => "release_threshold",
            Condition::SyncAlloc => "sync_alloc",
            Condition::Both => "both",
        }
    }

    /// 案 B（`cuMemAlloc` 同期割当）を使うかどうか。
    fn uses_sync_alloc(self) -> bool {
        matches!(self, Condition::SyncAlloc | Condition::Both)
    }

    /// 案 A（release threshold 引き上げ）を使うかどうか。
    fn raises_threshold(self) -> bool {
        matches!(self, Condition::ReleaseThreshold | Condition::Both)
    }
}

/// サイズ・フェーズごとの計測を `[Baseline, A, B, Both, Baseline]` の順で
/// 実行する（計画 §3.3「ブラケット方式」）。前後の baseline 差分を
/// 「ドリフト幅」として記録し、順序依存・プール状態の持ち越しと対策効果
/// を分離する。
fn bracket_conditions() -> [Condition; 5] {
    [
        Condition::Baseline,
        Condition::ReleaseThreshold,
        Condition::SyncAlloc,
        Condition::Both,
        Condition::Baseline,
    ]
}

/// `mib` MiB を `f16` 要素数へ変換する（`large_buffer_percall_alloc_
/// transfer_triage.rs::numel_for_mib` と同一実装・同一 fail-closed 方針。
/// `.claude/rules/security.md` A03 節）。
fn numel_for_mib(mib: u64) -> Option<usize> {
    let bytes = mib.checked_mul(1024)?.checked_mul(1024)?;
    let elem_size = size_of::<f16>() as u64;
    let numel_u64 = bytes.checked_div(elem_size)?;
    usize::try_from(numel_u64).ok()
}

/// `samples`（秒）のうち `min` の `factor` 倍を超えたサンプル数を返す
/// （#1146 と同一実装）。
fn count_slow_samples(samples: &[f64], factor: f64) -> usize {
    let Some(&min) = samples
        .iter()
        .min_by(|a, b| a.partial_cmp(b).expect("samples must not be NaN"))
    else {
        return 0;
    };
    if min <= 0.0 {
        return 0;
    }
    samples.iter().filter(|&&s| s > min * factor).count()
}

/// [`bench_harness::Measurement`] からの要約統計（#1146 と同一構造）。
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

/// 出力 1 行の書式:
/// `condition,phase,size_mib,order,run_idx,cold_ms,min_ms,q1_ms,median_ms,q3_ms,max_ms,slow_count`
#[allow(clippy::too_many_arguments)]
fn print_summary_row(
    condition: Condition,
    phase: &str,
    size_mib: u64,
    order: &str,
    run_idx: usize,
    cold_ms: f64,
    s: &Summary,
) {
    println!(
        "{},{phase},{size_mib},{order},{run_idx},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
        condition.label(),
        cold_ms,
        s.min_secs * 1000.0,
        s.q1_secs * 1000.0,
        s.median_secs * 1000.0,
        s.q3_secs * 1000.0,
        s.max_secs * 1000.0,
        s.slow_count,
    );
}

/// driver プールの属性読み取り結果。`has_async_alloc` が偽の非対応環境
/// （`Disabled`）と、API 呼び出し自体が失敗した異常（`Failed`）を区別
/// する（#1146 `PoolUsage` と同じ設計判断）。
enum PoolAttrs {
    Disabled,
    Available {
        reserved: u64,
        used: u64,
        release_threshold: u64,
    },
    Failed(String),
}

/// driver プールの `RESERVED_MEM_CURRENT`／`USED_MEM_CURRENT`／
/// `RELEASE_THRESHOLD` を読み取る（読み取りのみ。呼び出し元が
/// [`ReleaseThresholdGuard`] 経由で `set_attribute` を使う場合と経路を
/// 分離する）。
fn read_mem_pool_attrs(ctx: &CudaContext) -> PoolAttrs {
    if !ctx.has_async_alloc() {
        return PoolAttrs::Disabled;
    }
    // SAFETY: `ctx.cu_device()` は `CudaContext::new` 済みの有効な
    // デバイスハンドル（#1146 `read_mem_pool_usage` と同一根拠）。
    // `get_attribute` は読み取り専用で `set_attribute` は呼ばない。
    unsafe {
        let mem_pool = match result::device::get_mem_pool(ctx.cu_device()) {
            Ok(pool) => pool,
            Err(e) => return PoolAttrs::Failed(format!("get_mem_pool 失敗: {e:?}")),
        };
        let mut reserved: u64 = 0;
        if let Err(e) = result::mem_pool::get_attribute(
            mem_pool,
            CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            (&mut reserved as *mut u64).cast(),
        ) {
            return PoolAttrs::Failed(format!("get_attribute(RESERVED_MEM_CURRENT) 失敗: {e:?}"));
        }
        let mut used: u64 = 0;
        if let Err(e) = result::mem_pool::get_attribute(
            mem_pool,
            CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            (&mut used as *mut u64).cast(),
        ) {
            return PoolAttrs::Failed(format!("get_attribute(USED_MEM_CURRENT) 失敗: {e:?}"));
        }
        let mut release_threshold: u64 = 0;
        if let Err(e) = result::mem_pool::get_attribute(
            mem_pool,
            CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut release_threshold as *mut u64).cast(),
        ) {
            return PoolAttrs::Failed(format!("get_attribute(RELEASE_THRESHOLD) 失敗: {e:?}"));
        }
        PoolAttrs::Available {
            reserved,
            used,
            release_threshold,
        }
    }
}

/// `usage` を 1 行の CSV 風ログとして出力する。`Failed`（API 呼び出し
/// 自体の異常）は診断前提が崩れている可能性があるためテスト失敗として
/// 扱う（#1146 `print_pool_usage` と同一方針）。
fn print_pool_attrs(label: &str, attrs: &PoolAttrs) {
    match attrs {
        PoolAttrs::Available {
            reserved,
            used,
            release_threshold,
        } => println!(
            "mem_pool_attr,{label},reserved_bytes={reserved},used_bytes={used},release_threshold={release_threshold}"
        ),
        PoolAttrs::Disabled => {
            println!("mem_pool_attr,{label},disabled(has_async_alloc=false)")
        }
        PoolAttrs::Failed(err) => {
            println!("mem_pool_attr,{label},error({err})");
            panic!(
                "read_mem_pool_attrs({label}) が失敗した（has_async_alloc=true の環境での API 呼び出し異常）: {err}"
            );
        }
    }
}

/// driver プールの release threshold を `u64::MAX` へ引き上げる RAII
/// ガード（案 A）。構築時に現在値を読み取って保持し、`Drop` で必ず復元
/// する（プロセス全体の driver プール状態を変更するため、テスト内の
/// どの終了経路〈`expect` panic 含む〉でも復元されることが必須）。
///
/// `has_async_alloc` が偽の環境では driver プール自体が存在しないため
/// `new` は `None` を返し、呼び出し元は条件 A をスキップする。
struct ReleaseThresholdGuard<'a> {
    ctx: &'a CudaContext,
    mem_pool: cudarc::driver::sys::CUmemoryPool,
    original_threshold: u64,
}

impl<'a> ReleaseThresholdGuard<'a> {
    fn new(ctx: &'a CudaContext) -> Option<Self> {
        if !ctx.has_async_alloc() {
            println!(
                "release_threshold_guard,skipped,reason=has_async_alloc=false（driver プール非対応環境のため案 A をスキップ）"
            );
            return None;
        }
        // SAFETY: `ctx.cu_device()` は `CudaContext::new` 済みの有効な
        // デバイスハンドル。`get_mem_pool` から得た有効な pool を
        // `get_attribute`／`set_attribute` へそのまま渡す（#1146
        // `read_mem_pool_usage` と同一根拠）。`value` は `u64` 変数への
        // 生ポインタで `set_attribute` の「正しい型の領域を指す」契約を
        // 満たす（`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` は `cuuint64_t`）。
        let (mem_pool, original_threshold) = unsafe {
            let mem_pool = result::device::get_mem_pool(ctx.cu_device())
                .expect("get_mem_pool must succeed on has_async_alloc=true environment");
            let mut original: u64 = 0;
            result::mem_pool::get_attribute(
                mem_pool,
                CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut original as *mut u64).cast(),
            )
            .expect("get_attribute(RELEASE_THRESHOLD) must succeed before raising it");
            let mut new_threshold: u64 = u64::MAX;
            result::mem_pool::set_attribute(
                mem_pool,
                CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut new_threshold as *mut u64).cast(),
            )
            .expect("set_attribute(RELEASE_THRESHOLD, u64::MAX) must succeed（案 A の適用）");
            // 適用確認のため読み戻す（`set_attribute` が黙って無視される
            // 環境がないことを保証する。読み戻し値が期待どおりであるかは
            // 呼び出し元の `bracket_conditions` 末尾比較で確認する）。
            let mut applied: u64 = 0;
            result::mem_pool::get_attribute(
                mem_pool,
                CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut applied as *mut u64).cast(),
            )
            .expect("get_attribute(RELEASE_THRESHOLD) 読み戻しに失敗した");
            assert_eq!(
                applied,
                u64::MAX,
                "release threshold の適用が反映されていない（読み戻し値が u64::MAX と不一致）"
            );
            (mem_pool, original)
        };
        Some(ReleaseThresholdGuard {
            ctx,
            mem_pool,
            original_threshold,
        })
    }
}

impl Drop for ReleaseThresholdGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.mem_pool` は `new` で取得した有効な pool ハンドル
        // （`ctx` のライフタイムに束縛されており本ガードのスコープ中は
        // 有効）。`value` は `u64` 変数への生ポインタ。Drop 内で panic
        // すると二重パニックでプロセスが abort しうるため、失敗は
        // `eprintln!` に留め復元を試みるのみとする。
        let mut restore = self.original_threshold;
        let result = unsafe {
            result::mem_pool::set_attribute(
                self.mem_pool,
                CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut restore as *mut u64).cast(),
            )
        };
        if let Err(e) = result {
            eprintln!(
                "release_threshold_guard,restore_failed,original={},error={e:?}（driver プール状態が u64::MAX のまま残留している可能性）",
                self.original_threshold
            );
        }
        // `self.ctx` は Drop 内では使わないが、ガードが `ctx` のライフ
        // タイムより長生きしないことをコンパイラに保証させるために保持
        // する（use-after-free 系の誤用防止）。
        let _ = self.ctx;
    }
}

/// 案 B: `cuMemAlloc`（同期割当）で確保したデバイスバッファ。確保・
/// 解放 API のみを baseline（`cuMemAllocAsync`／`cuMemFreeAsync`）から
/// 差し替え、memset・H2D・D2H・カーネル起動は呼び出し側が baseline と
/// 同じ cudarc safe API（`upgrade_device_ptr` で得た `CudaSlice` に対する
/// `memset_zeros`／`memcpy_htod`／`clone_dtoh`／`launch_f16`）で行う
/// （計画 §3.1「差分を確保／解放 API の 1 変数に限定する」）。
struct SyncDeviceBuffer {
    stream: std::sync::Arc<CudaStream>,
    slice: Option<CudaSlice<f16>>,
}

impl SyncDeviceBuffer {
    /// `numel` 要素分の未初期化デバイス領域を `cuMemAlloc` で確保する。
    fn alloc(
        stream: &std::sync::Arc<CudaStream>,
        numel: usize,
    ) -> Result<Self, cudarc::driver::result::DriverError> {
        let bytes = numel
            .checked_mul(size_of::<f16>())
            .expect("SyncDeviceBuffer::alloc のバイト長計算は overflow しない前提のサイズのみ扱う");
        // SAFETY: `bytes` は `numel * size_of::<f16>()` を checked に
        // 導出した値（overflow なし）。`malloc_sync` の戻り値は
        // `upgrade_device_ptr` へそのまま渡し、`numel` 要素・`f16` 型分の
        // 領域として扱う（`malloc_sync` が確保したバイト長と完全一致）。
        // 確保直後は未初期化のため、呼び出し側が memset／H2D で書き込む
        // までは読み出さない契約（`upgrade_device_ptr` の Safety コメント
        // と同じ前提）。
        let ptr = unsafe { result::malloc_sync(bytes)? };
        // SAFETY: `ptr` は直前の `malloc_sync(bytes)` が返した有効な
        // アロケーションで、`bytes == numel * size_of::<f16>()`。
        let slice = unsafe { stream.upgrade_device_ptr::<f16>(ptr, numel) };
        Ok(SyncDeviceBuffer {
            stream: stream.clone(),
            slice: Some(slice),
        })
    }

    fn slice(&self) -> &CudaSlice<f16> {
        self.slice
            .as_ref()
            .expect("slice は Drop 前のみ None になる")
    }

    fn slice_mut(&mut self) -> &mut CudaSlice<f16> {
        self.slice
            .as_mut()
            .expect("slice は Drop 前のみ None になる")
    }
}

impl Drop for SyncDeviceBuffer {
    fn drop(&mut self) {
        // `cuMemAlloc` 由来ポインタへの `cuMemFreeAsync` 適用可否は
        // 未検証のため `CudaSlice` の通常 drop（`has_async_alloc=true`
        // 環境では `free_async` を発行する）には任せない。`leak()` で
        // 所有権を外し、`synchronize` で全保留操作の完了を確定してから
        // `free_sync`（`cuMemFree`）で明示解放する（二重解放防止のため
        // `free_sync` を呼ぶのは本 `Drop` のみ）。
        //
        // `cuMemFree` は暗黙のデバイス同期を伴う（計画 §3.1 の注記）ため
        // 事前の `synchronize` は厳密には冗長だが、`leak()` が内部で待つ
        // read/write イベントとは独立に「全ストリーム操作完了後に解放
        // する」という baseline との対称性を明示するため呼ぶ。
        if let Err(e) = self.stream.synchronize() {
            eprintln!("sync_device_buffer,drop,synchronize_failed,error={e:?}");
        }
        if let Some(slice) = self.slice.take() {
            let ptr = slice.leak();
            // SAFETY: `ptr` は本構造体の `alloc` で `malloc_sync` により
            // 確保され、他のどこからも解放されていない（`slice.take()`
            // による一度きりの消費で二重 `free_sync` を防止）。直前の
            // `synchronize` により全アクセスが完了している。
            if let Err(e) = unsafe { result::free_sync(ptr) } {
                eprintln!("sync_device_buffer,drop,free_sync_failed,error={e:?}");
            }
        }
    }
}

/// `workload` を 1 回だけ `Instant` で計測してから（cold）、同じ
/// クロージャを [`bench_harness::run`] へ渡して warmup+計測を行う
/// （#1146 `measure_with_cold` と同一実装）。
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

/// P1: デバイス確保＋解放のみ。`condition.uses_sync_alloc()` に応じて
/// baseline（`stream.alloc`。`cuMemAllocAsync` 経由）か案 B
/// （[`SyncDeviceBuffer::alloc`]。`cuMemAlloc` 経由）を切り替える。
fn phase_p1_alloc_only(
    device: &CudaDevice,
    config: &MeasurementConfig,
    numel: usize,
    condition: Condition,
) -> (f64, bench_harness::Measurement) {
    measure_with_cold(config, || {
        if condition.uses_sync_alloc() {
            let buf = SyncDeviceBuffer::alloc(device.stream(), numel)
                .expect("SyncDeviceBuffer::alloc must succeed on CUDA-equipped test runner");
            drop(buf);
        } else {
            // SAFETY: 確保領域は読まず drop するのみ（#1146
            // `phase_p1_alloc_only` と同一根拠）。
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
            device
                .stream()
                .synchronize()
                .expect("synchronize after free must succeed to measure free completion");
        }
    })
}

/// P2: ゼロ初期化確保＋解放。baseline は `alloc_zeros`、案 B は
/// [`SyncDeviceBuffer::alloc`] + `memset_zeros`（差分を確保／解放 API の
/// 1 変数に限定する契約。計画 §3.1）。
fn phase_p2_alloc_zeros(
    device: &CudaDevice,
    config: &MeasurementConfig,
    numel: usize,
    condition: Condition,
) -> (f64, bench_harness::Measurement) {
    measure_with_cold(config, || {
        if condition.uses_sync_alloc() {
            let mut buf = SyncDeviceBuffer::alloc(device.stream(), numel)
                .expect("SyncDeviceBuffer::alloc must succeed on CUDA-equipped test runner");
            device
                .stream()
                .memset_zeros(buf.slice_mut())
                .expect("memset_zeros must succeed on CUDA-equipped test runner");
            device
                .stream()
                .synchronize()
                .expect("synchronize after memset_zeros must succeed to measure completion");
            drop(buf);
        } else {
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
        }
    })
}

/// P3: H2D のみ。baseline は `clone_htod`、案 B は [`SyncDeviceBuffer::
/// alloc`] + `memcpy_htod`。
fn phase_p3_h2d_only(
    device: &CudaDevice,
    config: &MeasurementConfig,
    host_src: &[f16],
    condition: Condition,
) -> (f64, bench_harness::Measurement) {
    measure_with_cold(config, || {
        if condition.uses_sync_alloc() {
            let mut buf = SyncDeviceBuffer::alloc(device.stream(), host_src.len())
                .expect("SyncDeviceBuffer::alloc must succeed on CUDA-equipped test runner");
            device
                .stream()
                .memcpy_htod(host_src, buf.slice_mut())
                .expect("memcpy_htod must succeed on CUDA-equipped test runner");
            device
                .stream()
                .synchronize()
                .expect("synchronize must succeed on CUDA-equipped test runner");
            drop(buf);
        } else {
            let dev = device
                .stream()
                .clone_htod(host_src)
                .expect("clone_htod must succeed on CUDA-equipped test runner");
            device
                .stream()
                .synchronize()
                .expect("synchronize must succeed on CUDA-equipped test runner");
            drop(dev);
            device
                .stream()
                .synchronize()
                .expect("synchronize after free must succeed to measure free completion");
        }
    })
}

/// P4: D2H のみ・宛先が毎回新規 `Vec`（#1146 の二峰性発生箇所）。
/// `device_src` は常駐バッファとして計測外で確保する（baseline・案 B
/// 双方とも同じ `CudaSlice<f16>` として渡せるため、確保方式によらず
/// 共通実装で扱える）。
fn phase_p4_d2h_fresh_vec(
    device: &CudaDevice,
    config: &MeasurementConfig,
    device_src: &CudaSlice<f16>,
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

/// P0: 転送のみ合算（`large_buffer_percall_alloc_transfer_triage.rs::
/// phase_p0_transfer_only` と同型。H2D×2＋ゼロ初期化確保＋D2H）。
/// baseline／案 B の切替は各段（H2D 先の A/B バッファ・C 出力バッファ）
/// で [`SyncDeviceBuffer`] を使うかどうかに帰着する。
fn phase_p0_transfer_only(
    device: &CudaDevice,
    config: &MeasurementConfig,
    a: &[f16],
    b: &[f16],
    out_len: usize,
    condition: Condition,
) -> (f64, bench_harness::Measurement) {
    measure_with_cold(config, || {
        if condition.uses_sync_alloc() {
            let mut a_dev = SyncDeviceBuffer::alloc(device.stream(), a.len())
                .expect("SyncDeviceBuffer::alloc(a) must succeed on CUDA-equipped test runner");
            device
                .stream()
                .memcpy_htod(a, a_dev.slice_mut())
                .expect("memcpy_htod(a) must succeed on CUDA-equipped test runner");
            let mut b_dev = SyncDeviceBuffer::alloc(device.stream(), b.len())
                .expect("SyncDeviceBuffer::alloc(b) must succeed on CUDA-equipped test runner");
            device
                .stream()
                .memcpy_htod(b, b_dev.slice_mut())
                .expect("memcpy_htod(b) must succeed on CUDA-equipped test runner");
            let mut c_dev = SyncDeviceBuffer::alloc(device.stream(), out_len)
                .expect("SyncDeviceBuffer::alloc(c) must succeed on CUDA-equipped test runner");
            device
                .stream()
                .memset_zeros(c_dev.slice_mut())
                .expect("memset_zeros(c) must succeed on CUDA-equipped test runner");
            device
                .stream()
                .synchronize()
                .expect("synchronize must succeed on CUDA-equipped test runner");
            let _c_host = device
                .stream()
                .clone_dtoh(c_dev.slice())
                .expect("clone_dtoh must succeed on CUDA-equipped test runner");
            device
                .stream()
                .synchronize()
                .expect("synchronize after D2H must succeed before host buffer is freed");
            drop(a_dev);
            drop(b_dev);
            drop(c_dev);
        } else {
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
            device
                .stream()
                .synchronize()
                .expect("synchronize after D2H must succeed before host buffer is freed");
            drop(a_dev);
            drop(b_dev);
            drop(c_dev);
            device
                .stream()
                .synchronize()
                .expect("synchronize after free must succeed to measure free completion");
        }
    })
}

/// P7: 本番経路レプリカ（`CudaMmaGemm`）。`upload_f16`→
/// `alloc_output_f16`→`launch_f16`→`download_f16` を条件別に実行する。
/// baseline は本番と同一の呼び出し列、案 B は `alloc_output_f16` の
/// 代わりに [`SyncDeviceBuffer`] を使い `launch_f16` へその `slice_mut`
/// を渡す（A/B 入力は両条件とも `upload_f16`〈`cuMemAllocAsync` 経由の
/// H2D〉のまま。C 出力側の確保・解放 API のみを差し替え、
/// dim4096 相当での効果を見る）。checksum（f64 総和）を返し、経路差し
/// 替えが数値結果を変えていないことを条件間で検証する材料とする。
#[allow(clippy::too_many_arguments)]
fn phase_p7_gemm_replica(
    device: &CudaDevice,
    gemm: &CudaMmaGemm,
    a: &[f16],
    b: &[f16],
    m: u32,
    n: u32,
    k: u32,
    condition: Condition,
) -> (f64, bench_harness::Measurement, f64) {
    let mut checksum = 0.0f64;
    let out_len = (m as usize) * (n as usize);
    let (cold_ms, measurement) = measure_with_cold(
        &MeasurementConfig::new(P7_RUNS, P7_RUNS)
            .expect("P7_RUNS/P7_RUNS must satisfy TASK-8.1 minimums"),
        || {
            let (a_dev, b_dev) = gemm
                .upload_f16(a, b)
                .expect("upload_f16 must succeed on CUDA-equipped test runner");
            let c_host = if condition.uses_sync_alloc() {
                let mut c_dev = SyncDeviceBuffer::alloc(device.stream(), out_len)
                    .expect("SyncDeviceBuffer::alloc(c) must succeed on CUDA-equipped test runner");
                device
                    .stream()
                    .memset_zeros(c_dev.slice_mut())
                    .expect("memset_zeros(c) must succeed on CUDA-equipped test runner");
                gemm.launch_f16(&a_dev, &b_dev, c_dev.slice_mut(), m, n, k)
                    .expect("launch_f16 must succeed on CUDA-equipped test runner");
                let host = gemm
                    .download_f16(c_dev.slice())
                    .expect("download_f16 must succeed on CUDA-equipped test runner");
                drop(c_dev);
                host
            } else {
                let mut c_dev = gemm
                    .alloc_output_f16(m, n)
                    .expect("alloc_output_f16 must succeed on CUDA-equipped test runner");
                gemm.launch_f16(&a_dev, &b_dev, &mut c_dev, m, n, k)
                    .expect("launch_f16 must succeed on CUDA-equipped test runner");
                let host = gemm
                    .download_f16(&c_dev)
                    .expect("download_f16 must succeed on CUDA-equipped test runner");
                drop(c_dev);
                host
            };
            drop(a_dev);
            drop(b_dev);
            gemm.synchronize()
                .expect("synchronize after free must succeed to measure free completion");
            // 最終イテレーションの出力のみ checksum へ反映すれば十分
            // （全ラン同一入力・決定的カーネルのため各回同値になるはず）。
            checksum = c_host.iter().map(|v| f64::from(v.to_f32())).sum();
        },
    );
    (cold_ms, measurement, checksum)
}

/// サイズごとに何ラン計測するか（key サイズ 32・33 MiB は
/// [`KEY_RUNS`]、それ以外は [`OTHER_RUNS`]）。
fn runs_for_size(size_mib: u64) -> usize {
    if size_mib == 32 || size_mib == 33 {
        KEY_RUNS
    } else {
        OTHER_RUNS
    }
}

/// 本体: サイズスイープ × 条件ブラケット × フェーズ × ラン。
///
/// 出力 1 行の書式（フェーズ計測行）:
/// `condition,phase,size_mib,order,run_idx,cold_ms,min_ms,q1_ms,median_ms,q3_ms,max_ms,slow_count`
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 想定）必須。イシュー #1149 の A/B 計測専用テストで受け入れ判定には使わない"]
fn large_buffer_percall_alloc_ab_record() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    println!(
        "environment: name={:?} compute_capability={:?} arch={:?} has_async_alloc={}",
        device.name(),
        device.compute_capability(),
        device.arch(),
        device.context().has_async_alloc(),
    );

    print_pool_attrs("before_sweep", &read_mem_pool_attrs(device.context()));

    println!(
        "condition,phase,size_mib,order,run_idx,cold_ms,min_ms,q1_ms,median_ms,q3_ms,max_ms,slow_count"
    );

    // --- 昇順パス: 全フェーズ（P0〜P4） ---
    for &size_mib in SIZES_MIB.iter() {
        let numel =
            numel_for_mib(size_mib).unwrap_or_else(|| panic!("size {size_mib} MiB must fit usize"));
        let mut rng = Xorshift64Star::new(0xAB00 ^ size_mib);
        let a: Vec<f16> = rng.fill_vec_f16(numel);
        let b: Vec<f16> = rng.fill_vec_f16(numel);
        let runs = runs_for_size(size_mib);
        let config = MeasurementConfig::new(20, 20).expect("20/20 must satisfy TASK-8.1 minimums");

        for condition in bracket_conditions() {
            // 案 A（release threshold 引き上げ）はガードのスコープ内で
            // のみ有効。ブロック終端で自動的に元へ復元される
            // （`ReleaseThresholdGuard::drop`）。
            let _guard = if condition.raises_threshold() {
                ReleaseThresholdGuard::new(device.context())
            } else {
                None
            };

            for run_idx in 0..runs {
                let (cold, m) = phase_p1_alloc_only(&device, &config, numel, condition);
                print_summary_row(
                    condition,
                    "p1_alloc_only",
                    size_mib,
                    "asc",
                    run_idx,
                    cold,
                    &summarize(&m),
                );

                let (cold, m) = phase_p2_alloc_zeros(&device, &config, numel, condition);
                print_summary_row(
                    condition,
                    "p2_alloc_zeros",
                    size_mib,
                    "asc",
                    run_idx,
                    cold,
                    &summarize(&m),
                );

                let (cold, m) = phase_p3_h2d_only(&device, &config, &a, condition);
                print_summary_row(
                    condition,
                    "p3_h2d_only",
                    size_mib,
                    "asc",
                    run_idx,
                    cold,
                    &summarize(&m),
                );

                let (cold, m) = phase_p0_transfer_only(&device, &config, &a, &b, numel, condition);
                print_summary_row(
                    condition,
                    "p0_transfer_only",
                    size_mib,
                    "asc",
                    run_idx,
                    cold,
                    &summarize(&m),
                );

                // P4 は常駐デバイスバッファを 1 本用意して使い回す
                // （計測外で確保。D2H 側の差分のみを見るため。#1146 と
                // 同じ理由）。条件によらず baseline 相当の `clone_htod`
                // で確保する（P4 が見るのは D2H 側のみのため、確保方式は
                // baseline に固定して差分を D2H に絞る）。
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
                    condition,
                    "p4_d2h_fresh_vec",
                    size_mib,
                    "asc",
                    run_idx,
                    cold,
                    &summarize(&m),
                );

                drop(device_src);
                device
                    .stream()
                    .synchronize()
                    .expect("synchronize after free must succeed to measure free completion");
            }

            print_pool_attrs(
                &format!("after_{}_{}mib", condition.label(), size_mib),
                &read_mem_pool_attrs(device.context()),
            );
        }
    }

    // --- 降順パス: P0／P4 のみ（#1146 §4.4 の降順限定スパイクが条件別に
    // 変わるかを見る。総実行時間短縮のため全フェーズは回さない）。 ---
    for &size_mib in SIZES_MIB.iter().rev() {
        let numel =
            numel_for_mib(size_mib).unwrap_or_else(|| panic!("size {size_mib} MiB must fit usize"));
        let mut rng = Xorshift64Star::new(0xCD00 ^ size_mib);
        let a: Vec<f16> = rng.fill_vec_f16(numel);
        let b: Vec<f16> = rng.fill_vec_f16(numel);
        let runs = runs_for_size(size_mib);
        let config = MeasurementConfig::new(20, 20).expect("20/20 must satisfy TASK-8.1 minimums");

        for condition in bracket_conditions() {
            let _guard = if condition.raises_threshold() {
                ReleaseThresholdGuard::new(device.context())
            } else {
                None
            };

            for run_idx in 0..runs {
                let (cold, m) = phase_p0_transfer_only(&device, &config, &a, &b, numel, condition);
                print_summary_row(
                    condition,
                    "p0_transfer_only",
                    size_mib,
                    "desc",
                    run_idx,
                    cold,
                    &summarize(&m),
                );

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
                    condition,
                    "p4_d2h_fresh_vec",
                    size_mib,
                    "desc",
                    run_idx,
                    cold,
                    &summarize(&m),
                );

                drop(device_src);
                device
                    .stream()
                    .synchronize()
                    .expect("synchronize after free must succeed to measure free completion");
            }
        }
    }

    // --- P7: 本番経路レプリカ（dim4096 相当。#1123 症状の直接再現） ---
    println!(
        "condition,phase,size_mib,order,run_idx,cold_ms,min_ms,q1_ms,median_ms,q3_ms,max_ms,slow_count,checksum"
    );
    let gemm = CudaMmaGemm::new(&device)
        .expect("CudaMmaGemm::new must succeed on CUDA-equipped test runner");
    let numel = (P7_DIM as usize) * (P7_DIM as usize);
    let mut rng = Xorshift64Star::new(0xEF00);
    let a: Vec<f16> = rng.fill_vec_f16(numel);
    let b: Vec<f16> = rng.fill_vec_f16(numel);

    let mut checksums: Vec<(Condition, f64)> = Vec::new();
    for condition in bracket_conditions() {
        let _guard = if condition.raises_threshold() {
            ReleaseThresholdGuard::new(device.context())
        } else {
            None
        };
        let (cold, m, checksum) =
            phase_p7_gemm_replica(&device, &gemm, &a, &b, P7_DIM, P7_DIM, P7_DIM, condition);
        let s = summarize(&m);
        println!(
            "{},p7_gemm_replica,{},n/a,0,{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.6}",
            condition.label(),
            P7_DIM,
            cold,
            s.min_secs * 1000.0,
            s.q1_secs * 1000.0,
            s.median_secs * 1000.0,
            s.q3_secs * 1000.0,
            s.max_secs * 1000.0,
            s.slow_count,
            checksum,
        );
        checksums.push((condition, checksum));
    }

    // 経路差し替え（baseline vs 案 B）が数値結果を変えていないことの検査
    // （計画 §3.3「数値一致の確認として各条件の出力 checksum が条件間で
    // 一致することを assert」）。f16 → f64 総和は決定的カーネル・同一
    // 入力のため条件間で完全一致するはず（絶対誤差の複合判定は不要な
    // ほど厳密に同一の計算経路のため、僅かな浮動小数点差異の余地を見て
    // 相対誤差 1e-9 を許容する）。
    let reference = checksums[0].1;
    for (condition, checksum) in &checksums {
        let rel_err = if reference.abs() > 0.0 {
            (checksum - reference).abs() / reference.abs()
        } else {
            (checksum - reference).abs()
        };
        assert!(
            rel_err < 1e-9,
            "P7 checksum が条件間で乖離した（経路差し替えが数値結果を変えている疑い）: \
             condition={:?} checksum={checksum} reference={reference} rel_err={rel_err}",
            condition
        );
    }

    // release threshold が全ブラケット終了後に初期値へ復元済みであること
    // を確認する（ガードの Drop が正しく機能していることの最終検査）。
    print_pool_attrs("after_sweep", &read_mem_pool_attrs(device.context()));
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn numel_for_mib_computes_f16_element_count() {
        assert_eq!(numel_for_mib(1), Some(524_288));
        assert_eq!(numel_for_mib(32), Some(524_288 * 32));
    }

    #[test]
    fn numel_for_mib_rejects_overflow() {
        assert_eq!(numel_for_mib(u64::MAX), None);
    }

    #[test]
    fn count_slow_samples_counts_values_above_factor_times_min() {
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
    fn condition_label_and_flags_are_consistent() {
        assert_eq!(Condition::Baseline.label(), "baseline");
        assert!(!Condition::Baseline.uses_sync_alloc());
        assert!(!Condition::Baseline.raises_threshold());

        assert_eq!(Condition::ReleaseThreshold.label(), "release_threshold");
        assert!(!Condition::ReleaseThreshold.uses_sync_alloc());
        assert!(Condition::ReleaseThreshold.raises_threshold());

        assert_eq!(Condition::SyncAlloc.label(), "sync_alloc");
        assert!(Condition::SyncAlloc.uses_sync_alloc());
        assert!(!Condition::SyncAlloc.raises_threshold());

        assert_eq!(Condition::Both.label(), "both");
        assert!(Condition::Both.uses_sync_alloc());
        assert!(Condition::Both.raises_threshold());
    }

    #[test]
    fn bracket_conditions_follow_baseline_a_b_both_baseline_order() {
        let bracket = bracket_conditions();
        assert_eq!(
            bracket,
            [
                Condition::Baseline,
                Condition::ReleaseThreshold,
                Condition::SyncAlloc,
                Condition::Both,
                Condition::Baseline,
            ]
        );
    }

    #[test]
    fn runs_for_size_uses_key_runs_for_32_and_33_mib_only() {
        assert_eq!(runs_for_size(32), KEY_RUNS);
        assert_eq!(runs_for_size(33), KEY_RUNS);
        assert_eq!(runs_for_size(24), OTHER_RUNS);
        assert_eq!(runs_for_size(64), OTHER_RUNS);
    }

    #[test]
    fn sizes_mib_covers_the_32_to_33_mib_boundary_and_is_sorted() {
        assert!(SIZES_MIB.contains(&32));
        assert!(SIZES_MIB.contains(&33));
        assert!(SIZES_MIB.windows(2).all(|w| w[0] < w[1]));
    }
}
