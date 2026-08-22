//! GEMM 4096³ ピークメモリ計測ハーネス（TASK-14.2a・イシュー #178）。
//!
//! REQ-14（`docs/spec/04-requirements.md` 2026-08-05 再設計）は、内部計測 API
//! の必須提供と、GEMM（M=N=K=4096, f32）を代表ワークロードとした係数上限
//! （理論最小ワーキングセット A+B+C ≈ 192MiB の 2 倍以内 = 384MiB 以内）を
//! 求める。内部計測 API は TASK-14.1（#173〜#176。完了済み）で実装済みの
//! `fandhe_ai_tensor_core::memory_stats::{MemoryStats, AllocationTracker,
//! TrackedAllocation}` を `CpuMemory`／`CudaMemory`／`MetalMemory` の 3
//! バックエンドが同一シグネチャで実装する（`allocated_bytes`／
//! `peak_allocated_bytes`／`reset_peak`。`fandhe_ai_tensor_core::memory_stats` モジュール
//! コメント参照）。
//!
//! 本モジュール（#178・TASK-14.2a）はこの内部計測 API で GEMM 4096³ の
//! ピークメモリを 3 バックエンドで実測し、**バックエンド別ピーク値の実測
//! 記録を残す**ことを受け入れ条件とする。係数の確定・再調整は兄弟イシュー
//! #179（TASK-14.2b）、計測手段の環境差文書化は #180（TASK-14.3）のスコープ
//! であり、本モジュールはその入力データ（内部 API 値と外部参考値の乖離
//! データ）を生成するのみに留める。
//!
//! # 計測対象の粒度（計測境界。`fandhe_ai_tensor_core::memory_stats` モジュール
//! コメント「計測対象の粒度」参照）
//!
//! [`MemoryStats`] が計上するのは `MemoryOps`（`alloc_zeroed`／`upload`）
//! 経由のデバイスバッファ確保のみである。`BackendOps::gemm` 演算内部の
//! 一時確保（CPU: `CpuBackendOps::gemm` の出力 `Vec<f32>` と BLIS
//! パッキングバッファ、CUDA: `CudaGemm::run_tiled_f32` の stream 直接確保、
//! Metal: `MetalGemm` の直接確保）は計測対象外である。`BackendOps::gemm` は
//! `&Tensor<f32>` を直接受け取る API であり `DeviceBuffer`（`MemoryOps` の
//! 確保結果）を経由しないため（各バックエンドの `ops.rs` 参照）、本ハーネスの
//! 「代表ワーキングセット（A・B・C 各 1 バッファ）を `MemoryOps` で確保し
//! 保持したまま `gemm` を実行する」という手順は、GEMM 実行に必要な最小限の
//! デバイス常駐量を模した計測であり、GEMM カーネル内部の一時確保量そのもの
//! を計測するものではない。この分離自体が本イシューの計測境界である。
//!
//! 計測境界の外側にある実確保量を推定する参考値として、Linux では
//! `/proc/self/status` の `VmHWM`（プロセス全体のピーク常駐セットサイズ）を
//! `PeakMemoryTrial::vm_hwm_bytes` に採取する。macOS の `getrusage`
//! （`ru_maxrss`）相当は `libc` クレートの新規追加が必要になり許容依存 8
//! 区分外のため（`.claude/rules/deps-policy.md`）、本イシューでは実装せず
//! `None` を返す（スコープ外。#180 への申し送り）。
//!
//! `PeakMemoryTrial::gemm_alloc_peak_bytes`（PR #370 codex-review 指摘 P1
//! 対応。[`crate::alloc_tracker`]）は、上記「計測境界の外側」を CPU
//! バックエンド限定で埋める: `std::alloc::GlobalAlloc` フックにより
//! `BackendOps::gemm` 実行区間中の実ヒープ確保量（出力 `Vec<f32>`・BLIS
//! パッキングバッファを含む）を実測する。`peak_bytes`（`MemoryOps` 経由・
//! 理論値と一致する決定的な値）とは独立した指標であり、GEMM 実装の内部
//! パラメータ（ブロッキング係数等）が変わればコード変更なしに値も追従して
//! 変化する。CUDA（`cudarc` driver 確保）・Metal（`objc2-metal`
//! `MTLBuffer`）は Rust の `GlobalAlloc` を経由しないため引き続き対象外
//! （`None`。実 GPU デバイス確保量を測るバックエンド固有の代替手段は
//! 別イシューのスコープとする。`.claude/rules/out-of-scope-tracking.md`）。
//!
//! # 計測手順（1 trial）
//!
//! 1. バックエンド入口（`CpuMemory::new()`／`CudaMemory::new(&CudaDevice)`／
//!    `MetalMemory::new(MetalContext)`）を単一インスタンスで構築し
//!    `reset_peak()` を呼ぶ
//! 2. `MemoryOps` 経由で代表ワーキングセットを確保する: `upload(A)`
//!    （M×K f32）→ `upload(B)`（K×N f32）→ `alloc_zeroed(C)`（M×N f32）
//! 3. ワーキングセット保持中に `BackendOps::gemm(A, B)` を実行する（所要
//!    秒数を `gemm_secs` として記録する）。CPU バックエンドはこの区間を
//!    `alloc_tracker::measure()`（起点記録〜ピーク読み出しを
//!    `MEASUREMENT_LOCK` で直列化する唯一の公開計測 API。PR #370
//!    codex-review 指摘 P1 再指摘対応）で包み、`gemm_alloc_peak_bytes`
//!    を採取する
//! 4. `peak_allocated_bytes()`（`peak_bytes`）を採取する
//! 5. `upload`/`alloc_zeroed` で得た 3 バッファを drop し、
//!    `allocated_bytes()`（`allocated_after_drop_bytes`）が 0 に戻ることを
//!    リーク検査として記録する（[`PeakMemoryReport::validate`] が
//!    fail-closed で強制する）。[`PeakMemoryReport::write_to_file`] は
//!    さらに [`PeakMemoryReport::require_gemm_alloc_tracked`] を通し、
//!    CPU バックエンドの公式実測記録では `gemm_alloc_peak_bytes` が
//!    `Some`（かつ出力バッファ分以上）であることを fail-closed に強制する
//!    （PR #370 codex-review 指摘 P1 再指摘対応: 旧スキーマの実測データが
//!    現行スキーマの結果として黙って正常扱いされることを防ぐ）
//!
//! 試行回数は既定 [`DEFAULT_PEAK_MEMORY_TRIALS`]（5 回）・中央値を採用する
//! （`.claude/rules/coding-rust.md` ベンチ規約）。行列データは
//! [`crate::rng::Xorshift64Star`]（決定的シード）で trial ごとに異なる
//! （しかし再現可能な）系列から生成する。
//!
//! # セキュリティ（OWASP Top 10。`.claude/rules/security.md`）
//!
//! - CLI 引数は許可リスト方式（[`PeakMemoryBackend::parse`]）で検証する（A03）
//! - `--size` から確保バイト数を導出する計算はすべて `checked_mul`（`byte_len_for_square`）
//!   でオーバーフローを拒否する（`backend-cpu::memory::checked_byte_len` と同型の方針。A03）
//! - `--size`／`--trials` は [`MAX_GEMM_SIZE`]／[`MAX_PEAK_MEMORY_TRIALS`] で
//!   上限を設け、極端な値による過大なメモリ確保・`Vec::with_capacity` の
//!   リソース枯渇を fail-closed に拒否する（`startup::MAX_STARTUP_TRIALS` と
//!   同型の方針。PR #360 codex-review 指摘 P1 の先例を踏襲）
//! - [`PeakMemoryReport::to_json`]/[`PeakMemoryReport::from_json`] は
//!   書き出し前・読み込み後に必ず [`PeakMemoryReport::validate`] を通す
//!   （`startup::StartupReport` と同じ「検証済み DTO」方針。A08）

use std::fmt;
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::memory_stats::MemoryStats;
use fandhe_ai_tensor_core::{BackendOps, MemoryOps, Tensor};

use crate::rng::Xorshift64Star;
use crate::stats::{self, BenchError as StatsError};

/// [`PeakMemoryReport`] の JSON スキーマバージョン（`startup::STARTUP_SCHEMA_VERSION`
/// と同じ手法。未知バージョンは [`PeakMemoryReport::validate`] が fail-closed で拒否する）。
pub const PEAK_MEMORY_SCHEMA_VERSION: &str = "1";

/// 既定の試行回数（`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値」）。
pub const DEFAULT_PEAK_MEMORY_TRIALS: usize = 5;

/// 1 計測あたりの試行回数の許容上限（`startup::MAX_STARTUP_TRIALS` と同型の
/// 過大な `Vec::with_capacity` によるリソース枯渇防止。PR #360 codex-review 指摘 P1）。
pub const MAX_PEAK_MEMORY_TRIALS: usize = 10_000;

/// REQ-14 の代表ワークロード（GEMM M=N=K=4096, f32）の既定正方行列サイズ。
pub const DEFAULT_GEMM_SIZE: usize = 4096;

/// `--size` に許容する正方行列サイズの上限。
///
/// `DEFAULT_GEMM_SIZE`（4096）の 4 倍（16384）を上限とする。16384² × 4 バイト
/// （f32）× 3 バッファ（A・B・C）は約 3GiB であり、意図しない極端な値
/// （`usize::MAX` 等）による OOM・長時間ハングを防ぎつつ、代表ワークロード
/// 以外のサイズでの探索的計測（#179 の係数調整時の比較実測等）にも十分な
/// 余地を残す（`.claude/rules/security.md` A03。`startup::MAX_STARTUP_TRIALS`
/// と同型の防御的上限）。
pub const MAX_GEMM_SIZE: usize = DEFAULT_GEMM_SIZE * 4;

/// 決定的シードの基点（`bench_harness::rng`。`.claude/rules/coding-rust.md`
/// 「学習系回帰テストには決定的シード設定ユーティリティを使う」）。
/// trial ごとに `RNG_SEED_BASE + trial_index` を使い、系列を変えつつ再現性を保つ。
const RNG_SEED_BASE: u64 = 0x5045_414b_4d45_4d01; // "PEAKMEM" 由来の固定値

/// ピークメモリ計測ハーネス固有のエラー。
///
/// 本番経路で `unwrap`/`expect` を使わない方針（`.claude/rules/coding-rust.md`）に基づき、
/// 引数検証・デバイス初期化・演算失敗をすべて型付きエラーとして返す。
#[derive(Debug)]
pub enum PeakMemoryError {
    /// 引数検証（[`PeakMemoryBackend::parse`]・[`PeakMemoryConfig::new`]）の失敗。
    InvalidArgument(String),
    /// ファイル I/O（`--out` 書き込み等）の失敗。
    Io(String),
    /// デバイス初期化に失敗した（CUDA driver 非搭載・Metal 非対応 OS 等）。
    /// `CudaDevice::is_available() == false`／`CudaDevice::new`／
    /// `MetalContext::new` の失敗を fail-closed（panic せず型付きエラー）で
    /// 表す（`startup::run_cuda`／`run_metal` と同型の契約。
    /// `.claude/rules/coding-rust.md`）。
    DeviceUnavailable(String),
    /// テンソル構築・`MemoryOps`／`BackendOps` 呼び出しの失敗。
    Backend(String),
    /// リーク検査（drop 後 `allocated_bytes() == 0`）を含む、レポート内部の
    /// 不変条件違反（fail-closed 拒否。A08）。
    ProtocolViolation(String),
    /// 分位点計算（[`stats::median_q1_q3`]）の失敗。
    Stats(StatsError),
}

impl fmt::Display for PeakMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PeakMemoryError::InvalidArgument(msg) => write!(f, "引数不正: {msg}"),
            PeakMemoryError::Io(msg) => write!(f, "I/O エラー: {msg}"),
            PeakMemoryError::DeviceUnavailable(msg) => write!(f, "デバイス利用不可: {msg}"),
            PeakMemoryError::Backend(msg) => write!(f, "バックエンド呼び出し失敗: {msg}"),
            PeakMemoryError::ProtocolViolation(msg) => write!(f, "計測プロトコル違反: {msg}"),
            PeakMemoryError::Stats(e) => write!(f, "統計計算エラー: {e}"),
        }
    }
}

impl std::error::Error for PeakMemoryError {}

impl From<StatsError> for PeakMemoryError {
    fn from(e: StatsError) -> Self {
        PeakMemoryError::Stats(e)
    }
}

impl From<BackendError> for PeakMemoryError {
    fn from(e: BackendError) -> Self {
        PeakMemoryError::Backend(e.to_string())
    }
}

/// 計測対象バックエンド識別子（`startup::StartupBackend` と同型の設計）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeakMemoryBackend {
    Cpu,
    Cuda,
    /// `cfg(target_os = "macos")` 限定（`.claude/rules/deps-policy.md`）。
    /// 非 macOS でも列挙自体は許容し、[`run_peak_memory`] が実行時に
    /// 「このプラットフォームでは Metal 未対応」を型付きエラーで返す
    /// （`startup::run_metal` と同型の契約）。
    Metal,
}

impl PeakMemoryBackend {
    /// CLI 引数・レポート `backend` フィールドで用いる文字列表現。
    pub fn as_str(&self) -> &'static str {
        match self {
            PeakMemoryBackend::Cpu => "cpu",
            PeakMemoryBackend::Cuda => "cuda",
            PeakMemoryBackend::Metal => "metal",
        }
    }

    /// 許可リスト方式の引数検証（`.claude/rules/security.md` A03）。
    pub fn parse(s: &str) -> Result<Self, PeakMemoryError> {
        match s {
            "cpu" => Ok(PeakMemoryBackend::Cpu),
            "cuda" => Ok(PeakMemoryBackend::Cuda),
            "metal" => Ok(PeakMemoryBackend::Metal),
            other => Err(PeakMemoryError::InvalidArgument(format!(
                "未知のバックエンド指定 {other:?}（cpu / cuda / metal のいずれかを指定する）"
            ))),
        }
    }
}

/// `size × size` の正方行列 1 枚分の f32 確保バイト数を検査付きで計算する。
/// `fandhe_ai_backend_cpu::memory::checked_byte_len` と同型の方針
/// （`.claude/rules/security.md` A03: `--size` は外部入力に相当するため、
/// バイト数換算でもオーバーフローを型付きエラーとして拒否する）。
fn byte_len_for_square(size: usize) -> Result<u64, PeakMemoryError> {
    let numel = size.checked_mul(size).ok_or_else(|| {
        PeakMemoryError::InvalidArgument(format!("size の 2 乗が usize の範囲を超える: {size}"))
    })?;
    let bytes = numel
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            PeakMemoryError::InvalidArgument(format!(
                "size から算出したバイト数が usize の範囲を超える: {size}"
            ))
        })?;
    Ok(bytes as u64)
}

/// A・B・C（いずれも `size × size` の f32 正方行列）の理論最小ワーキング
/// セット（REQ-14 の係数上限判定の分母）を検査付きで計算する。
fn theoretical_min_bytes(size: usize) -> Result<u64, PeakMemoryError> {
    let one = byte_len_for_square(size)?;
    one.checked_mul(3).ok_or_else(|| {
        PeakMemoryError::InvalidArgument(format!(
            "理論最小ワーキングセットの合計バイト数が u64 の範囲を超える: size={size}"
        ))
    })
}

/// ピークメモリ計測の実行設定。
///
/// フィールドは非公開とし、[`PeakMemoryConfig::new`] の値域検証（`size`・
/// `trials` の上限・オーバーフロー検査）を必ず経由させる（PR #370
/// codex-review 指摘 P1 対応: 構造体リテラルによる直接構築を許すと
/// `run_peak_memory` 内の `Vec::with_capacity(trials)`・`size * size`
/// 計算が未検証値に到達し、リソース枯渇や整数オーバーフローを招く）。
#[derive(Debug, Clone, Copy)]
pub struct PeakMemoryConfig {
    backend: PeakMemoryBackend,
    /// GEMM の M=N=K（正方行列のみ対応。REQ-14 代表ワークロードが正方形の
    /// ため、本ハーネスも非正方形をサポートしない）。
    size: usize,
    /// 試行回数（[`DEFAULT_PEAK_MEMORY_TRIALS`] 推奨。`startup::StartupConfig`
    /// と同型の理由で `1..=MAX_PEAK_MEMORY_TRIALS` を要求する）。
    trials: usize,
}

impl PeakMemoryConfig {
    /// `size`・`trials` の値域を検証して構築する（`startup::StartupConfig::new`
    /// と同型の fail-closed 方針。0 は分位点・行列として無意味なため拒否する）。
    /// 本コンストラクタが `PeakMemoryConfig` を得る唯一の手段であり（フィールド
    /// 非公開）、`run_peak_memory` へ未検証値が渡る経路を持たない。
    pub fn new(
        backend: PeakMemoryBackend,
        size: usize,
        trials: usize,
    ) -> Result<Self, PeakMemoryError> {
        if size == 0 {
            return Err(PeakMemoryError::InvalidArgument(
                "size は 1 以上を指定する".to_string(),
            ));
        }
        if size > MAX_GEMM_SIZE {
            return Err(PeakMemoryError::InvalidArgument(format!(
                "size は {MAX_GEMM_SIZE} 以下を指定する（指定値: {size}）"
            )));
        }
        if trials == 0 {
            return Err(PeakMemoryError::InvalidArgument(
                "trials は 1 以上を指定する".to_string(),
            ));
        }
        if trials > MAX_PEAK_MEMORY_TRIALS {
            return Err(PeakMemoryError::InvalidArgument(format!(
                "trials は {MAX_PEAK_MEMORY_TRIALS} 以下を指定する（指定値: {trials}）"
            )));
        }
        // 早期にオーバーフロー検査を通しておく（構築時点で拒否し、
        // 計測実行後に判明する事態を避ける）。
        theoretical_min_bytes(size)?;
        Ok(Self {
            backend,
            size,
            trials,
        })
    }

    /// 計測対象バックエンド。
    pub fn backend(&self) -> PeakMemoryBackend {
        self.backend
    }

    /// GEMM の M=N=K（`1..=MAX_GEMM_SIZE` に検証済み）。
    pub fn size(&self) -> usize {
        self.size
    }

    /// 試行回数（`1..=MAX_PEAK_MEMORY_TRIALS` に検証済み）。
    pub fn trials(&self) -> usize {
        self.trials
    }
}

/// 1 試行分の計測結果。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PeakMemoryTrial {
    /// `MemoryOps`（`upload(A)`→`upload(B)`→`alloc_zeroed(C)`）経由の
    /// 確保のみを対象とした内部計測ピーク（`MemoryStats::peak_allocated_bytes`）。
    pub peak_bytes: u64,
    /// A・B・C の 3 バッファを drop した直後の `allocated_bytes()`。
    /// リーク検査（0 であるはず）を兼ねる。
    pub allocated_after_drop_bytes: u64,
    /// `BackendOps::gemm(A, B)` の所要秒数（ワークロード実行の証跡）。
    pub gemm_secs: f64,
    /// 外部参考値: `/proc/self/status` の `VmHWM`（Linux 限定。モジュール
    /// 冒頭「計測対象の粒度」参照）。取得不能・非対応 OS では `None`。
    pub vm_hwm_bytes: Option<u64>,
    /// `BackendOps::gemm(A, B)` 実行区間中に実際にヒープ確保された
    /// バイト数の純増分ピーク（[`crate::alloc_tracker`] 経由。PR #370
    /// codex-review 指摘 P1 対応）。`peak_bytes` が `MemoryOps` 経由の
    /// 代表ワーキングセット（A+B+C の理論値と一致する決定的な値）しか
    /// 計上しないのに対し、本フィールドは GEMM 実装が実際に確保する
    /// 出力バッファ・パッキングバッファ等を反映する（モジュール冒頭
    /// 「計測対象の粒度」参照）。CPU バックエンドのみ `Some`
    /// （`TrackingAllocator` が `System` の代わりにプロセスの
    /// `#[global_allocator]` として有効化されている場合に限る）。
    /// CUDA／Metal はデバイス確保が Rust の `GlobalAlloc` を経由しないため
    /// 常に `None`（`crate::alloc_tracker` モジュール冒頭「適用範囲」参照）。
    pub gemm_alloc_peak_bytes: Option<u64>,
}

/// バイト数の分位点（中央値・Q1・Q3）。`startup::QuartileSecs` と同型の
/// 出力専用 DTO（`stats::Quartiles` は `Serialize` を持たない設計方針。
/// `stats.rs` 冒頭コメント参照）。内部計算は f64 で行うが、確保バイト数は
/// 現実的な規模（GiB オーダー）で `f64` の 53bit 仮数部に収まるため、
/// `u64` との往復に丸め誤差は生じない。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuartileBytes {
    pub median: u64,
    pub q1: u64,
    pub q3: u64,
}

impl QuartileBytes {
    fn from_samples(samples: &[u64]) -> Result<Self, PeakMemoryError> {
        let as_f64: Vec<f64> = samples.iter().map(|&b| b as f64).collect();
        let q = stats::median_q1_q3(&as_f64)?;
        Ok(Self {
            median: q.median as u64,
            q1: q.q1 as u64,
            q3: q.q3 as u64,
        })
    }
}

/// 秒数の分位点（`startup::QuartileSecs` と同型）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuartileSecs {
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
}

impl From<stats::Quartiles> for QuartileSecs {
    fn from(q: stats::Quartiles) -> Self {
        Self {
            median: q.median,
            q1: q.q1,
            q3: q.q3,
        }
    }
}

/// GEMM ピークメモリ計測の構造化出力（1 バックエンド × 1 サイズ分）。
///
/// `startup::StartupReport` と同じ「検証済み DTO」方針を踏襲する:
/// [`Self::to_json`] は書き出し前に、[`Self::from_json`] は読み込み直後に
/// 必ず [`Self::validate`] を通す（`.claude/rules/security.md` A08）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeakMemoryReport {
    pub schema_version: String,
    pub backend: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// 本ハーネスは f32 のみ対応（REQ-14 代表ワークロードの dtype）。
    pub dtype: String,
    pub trials: usize,
    /// A+B+C の理論最小ワーキングセット（バイト）。
    pub theoretical_min_bytes: u64,
    pub peak_bytes: QuartileBytes,
    pub gemm_secs: QuartileSecs,
    /// 全試行の生データ（#179 での係数確定・再調整の入力データ）。
    pub samples: Vec<PeakMemoryTrial>,
}

impl PeakMemoryReport {
    /// `trials` 分の [`PeakMemoryTrial`] から分位点を集計して構築する。
    fn from_trials(
        backend: PeakMemoryBackend,
        size: usize,
        samples: Vec<PeakMemoryTrial>,
    ) -> Result<Self, PeakMemoryError> {
        let peak_bytes_samples: Vec<u64> = samples.iter().map(|t| t.peak_bytes).collect();
        let gemm_secs_samples: Vec<f64> = samples.iter().map(|t| t.gemm_secs).collect();

        let report = Self {
            schema_version: PEAK_MEMORY_SCHEMA_VERSION.to_string(),
            backend: backend.as_str().to_string(),
            m: size,
            n: size,
            k: size,
            dtype: "f32".to_string(),
            trials: samples.len(),
            theoretical_min_bytes: theoretical_min_bytes(size)?,
            peak_bytes: QuartileBytes::from_samples(&peak_bytes_samples)?,
            gemm_secs: stats::median_q1_q3(&gemm_secs_samples)?.into(),
            samples,
        };
        report.validate()?;
        Ok(report)
    }

    /// スキーマ・試行数整合・値の有限性・リーク検査・分位点の再計算一致を
    /// fail-closed で検証する（`startup::StartupReport::validate` と同型の
    /// 方針。詳細は同関数参照）。
    pub fn validate(&self) -> Result<(), PeakMemoryError> {
        if self.schema_version != PEAK_MEMORY_SCHEMA_VERSION {
            return Err(PeakMemoryError::ProtocolViolation(format!(
                "未知の schema_version: 期待値 {PEAK_MEMORY_SCHEMA_VERSION:?}, 実際 {:?}",
                self.schema_version
            )));
        }
        PeakMemoryBackend::parse(&self.backend).map_err(|e| {
            PeakMemoryError::ProtocolViolation(format!("backend が許可リスト外: {e}"))
        })?;
        if self.dtype != "f32" {
            return Err(PeakMemoryError::ProtocolViolation(format!(
                "未対応の dtype: {:?}（f32 のみ対応）",
                self.dtype
            )));
        }
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(PeakMemoryError::ProtocolViolation(
                "m/n/k はすべて 1 以上が必須".to_string(),
            ));
        }
        if self.m != self.n || self.n != self.k {
            return Err(PeakMemoryError::ProtocolViolation(format!(
                "本ハーネスは正方行列のみ対応（m={}, n={}, k={} が不一致）",
                self.m, self.n, self.k
            )));
        }
        let expected_theoretical = theoretical_min_bytes(self.m)?;
        if self.theoretical_min_bytes != expected_theoretical {
            return Err(PeakMemoryError::ProtocolViolation(format!(
                "theoretical_min_bytes が m/n/k から再計算した値と不一致: \
                 stored={}, recomputed={}",
                self.theoretical_min_bytes, expected_theoretical
            )));
        }
        if self.trials == 0 {
            return Err(PeakMemoryError::ProtocolViolation(
                "trials は 1 以上が必須".to_string(),
            ));
        }
        if self.samples.len() != self.trials {
            return Err(PeakMemoryError::ProtocolViolation(format!(
                "samples の要素数（{}）が trials（{}）と不一致",
                self.samples.len(),
                self.trials
            )));
        }

        for (i, t) in self.samples.iter().enumerate() {
            if !(t.gemm_secs.is_finite() && t.gemm_secs >= 0.0) {
                return Err(PeakMemoryError::ProtocolViolation(format!(
                    "samples[{i}] に非有限値または負の gemm_secs が含まれている: {}",
                    t.gemm_secs
                )));
            }
            if t.allocated_after_drop_bytes != 0 {
                return Err(PeakMemoryError::ProtocolViolation(format!(
                    "samples[{i}] のリーク検査に失敗: drop 後も allocated_after_drop_bytes = {} \
                     （0 であるべき。MemoryOps 経由の確保が解放されず残存している）",
                    t.allocated_after_drop_bytes
                )));
            }
            if t.peak_bytes < self.theoretical_min_bytes {
                return Err(PeakMemoryError::ProtocolViolation(format!(
                    "samples[{i}] の peak_bytes（{}）が理論最小ワーキングセット（{}）を \
                     下回っている（A・B・C の 3 バッファを同時保持する契約に反する）",
                    t.peak_bytes, self.theoretical_min_bytes
                )));
            }
        }

        let peak_bytes_samples: Vec<u64> = self.samples.iter().map(|t| t.peak_bytes).collect();
        let recomputed_peak = QuartileBytes::from_samples(&peak_bytes_samples)?;
        if recomputed_peak != self.peak_bytes {
            return Err(PeakMemoryError::ProtocolViolation(format!(
                "peak_bytes: samples から再計算した分位点と格納値が不一致（改ざんの疑い）: \
                 recomputed={recomputed_peak:?}, stored={:?}",
                self.peak_bytes
            )));
        }
        if !(self.peak_bytes.q1 <= self.peak_bytes.median
            && self.peak_bytes.median <= self.peak_bytes.q3)
        {
            return Err(PeakMemoryError::ProtocolViolation(format!(
                "peak_bytes: q1 <= median <= q3 を満たさない: {:?}",
                self.peak_bytes
            )));
        }

        let gemm_secs_samples: Vec<f64> = self.samples.iter().map(|t| t.gemm_secs).collect();
        let recomputed_gemm: QuartileSecs = stats::median_q1_q3(&gemm_secs_samples)?.into();
        if recomputed_gemm != self.gemm_secs {
            return Err(PeakMemoryError::ProtocolViolation(format!(
                "gemm_secs: samples から再計算した分位点と格納値が不一致（改ざんの疑い）: \
                 recomputed={recomputed_gemm:?}, stored={:?}",
                self.gemm_secs
            )));
        }
        if !(self.gemm_secs.q1 <= self.gemm_secs.median
            && self.gemm_secs.median <= self.gemm_secs.q3)
        {
            return Err(PeakMemoryError::ProtocolViolation(format!(
                "gemm_secs: q1 <= median <= q3 を満たさない: {:?}",
                self.gemm_secs
            )));
        }

        Ok(())
    }

    /// CPU バックエンドの `gemm_alloc_peak_bytes` が実測されていることを fail-closed に
    /// 検証する（PR #370 codex-review 指摘 P1 再指摘対応）。
    ///
    /// [`Self::validate`] とは別関数に分離する: `gemm_alloc_peak_bytes` は
    /// `TrackingAllocator` がプロセスの `#[global_allocator]` として実際に有効化されて
    /// いる場合に限り `Some` になる契約（[`PeakMemoryTrial::gemm_alloc_peak_bytes`] doc
    /// comment参照）であり、これは `peak_memory_bench` バイナリ（`src/bin/
    /// peak_memory_bench.rs`）だけが満たす。ライブラリ API を経由する他クレート・
    /// 統合テスト（`tests/peak_memory_smoke.rs` 等、`TrackingAllocator` を
    /// `#[global_allocator]` 宣言しないバイナリ）では意図的に `None` になりうるため、
    /// これを [`Self::validate`]（`from_json`／`to_json` が常に経由する汎用 DTO 検証）で
    /// 強制すると正当な呼び出し経路まで壊してしまう。本メソッドは「公式の実測記録として
    /// 永続化する直前」（[`Self::write_to_file`]）にのみ適用する、狭いスコープの
    /// fail-closed ゲートである。
    ///
    /// CPU は各 sample の `gemm_alloc_peak_bytes` が `Some` かつ出力バッファ分
    /// （`m * n * size_of::<f32>()`）以上であることを要求する（ゼロ・過小な観測値や、
    /// 本指標の実測実装前に採取した旧スキーマデータの紛れ込みを検出する）。CUDA／Metal は
    /// デバイス確保が `GlobalAlloc` を経由しないため常に `None` であることを要求する
    /// （契約からの逸脱＝実装ミスを検出する）。
    pub fn require_gemm_alloc_tracked(&self) -> Result<(), PeakMemoryError> {
        let output_buffer_bytes = self
            .m
            .checked_mul(self.n)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .map(|v| v as u64);

        for (i, t) in self.samples.iter().enumerate() {
            match (self.backend.as_str(), t.gemm_alloc_peak_bytes) {
                ("cpu", None) => {
                    return Err(PeakMemoryError::ProtocolViolation(format!(
                        "samples[{i}] の gemm_alloc_peak_bytes が None（CPU の公式実測記録は \
                         TrackingAllocator により常に Some のはず。旧スキーマの実測データが \
                         紛れ込んでいる可能性がある）"
                    )));
                }
                ("cpu", Some(observed)) => {
                    if let Some(min_expected) = output_buffer_bytes
                        && observed < min_expected
                    {
                        return Err(PeakMemoryError::ProtocolViolation(format!(
                            "samples[{i}] の gemm_alloc_peak_bytes（{observed}）が \
                             出力バッファ分の下限（{min_expected}）を下回っている \
                             （GEMM 実行区間の実ヒープ確保が観測できていない）"
                        )));
                    }
                }
                (backend, Some(_)) if backend != "cpu" => {
                    return Err(PeakMemoryError::ProtocolViolation(format!(
                        "samples[{i}] の gemm_alloc_peak_bytes が {backend} バックエンドで \
                         Some（GlobalAlloc を経由しないため常に None のはず）"
                    )));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// JSON へシリアライズする（書き出し前に [`Self::validate`] を実行する）。
    pub fn to_json(&self) -> Result<String, PeakMemoryError> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|e| PeakMemoryError::ProtocolViolation(format!("JSON エンコード失敗: {e}")))
    }

    /// JSON からデシリアライズし、[`Self::validate`] を通してから返す。
    pub fn from_json(json: &str) -> Result<Self, PeakMemoryError> {
        let report: Self = serde_json::from_str(json)
            .map_err(|e| PeakMemoryError::ProtocolViolation(format!("JSON デコード失敗: {e}")))?;
        report.validate()?;
        Ok(report)
    }

    /// `--out` 指定先へ書き出す（[`Self::to_json`] を経由。呼び出し元 CLI
    /// （`peak_memory_bench`）から共有するための薄いヘルパー）。書き出し前に
    /// [`Self::require_gemm_alloc_tracked`] を通し、公式の実測記録として永続化する
    /// レポートが `gemm_alloc_peak_bytes` 契約を満たしていることを fail-closed に
    /// 強制する（PR #370 codex-review 指摘 P1 再指摘対応。[`Self::
    /// require_gemm_alloc_tracked`] doc comment 参照）。
    pub fn write_to_file(&self, path: &Path) -> Result<(), PeakMemoryError> {
        self.require_gemm_alloc_tracked()?;
        let json = self.to_json()?;
        std::fs::write(path, json)
            .map_err(|e| PeakMemoryError::Io(format!("出力ファイル書き込み失敗（{path:?}）: {e}")))
    }
}

/// `/proc/self/status` の `VmHWM` 行（KiB 単位）をバイト数へ変換して返す。
/// Linux 限定（モジュール冒頭「計測対象の粒度」参照）。取得できない場合
/// （非 Linux・行不在・パース失敗）は `None` を返す（外部参考値のため、
/// 取得失敗を計測全体の失敗にしない: `.claude/rules/coding-rust.md` の
/// 「本番経路で panic しない」方針の延長として、参考値の欠落は fail-open
/// で許容する）。
#[cfg(target_os = "linux")]
fn read_vm_hwm_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kib_str = rest.trim().trim_end_matches(" kB").trim();
            let kib: u64 = kib_str.parse().ok()?;
            return kib.checked_mul(1024);
        }
    }
    None
}

/// 非 Linux（macOS 含む）では未実装（モジュール冒頭「計測対象の粒度」の
/// `libc` 非追加判断を参照）。
#[cfg(not(target_os = "linux"))]
fn read_vm_hwm_bytes() -> Option<u64> {
    None
}

/// trial 用の決定的入力（A・B、いずれも `size × size` の f32 正方行列）を生成する。
fn make_trial_inputs(
    size: usize,
    trial_index: usize,
) -> Result<(Tensor<f32>, Tensor<f32>), PeakMemoryError> {
    let seed = RNG_SEED_BASE.wrapping_add(trial_index as u64);
    let mut rng = Xorshift64Star::new(seed);
    let a = Tensor::new(rng.fill_vec(size * size), &[size, size])
        .map_err(|e| PeakMemoryError::Backend(format!("入力テンソル A 構築失敗: {e:?}")))?;
    let b = Tensor::new(rng.fill_vec(size * size), &[size, size])
        .map_err(|e| PeakMemoryError::Backend(format!("入力テンソル B 構築失敗: {e:?}")))?;
    Ok((a, b))
}

/// CPU バックエンド 1 trial 分の計測（モジュール冒頭「計測手順」参照）。
fn run_cpu_trial(size: usize, trial_index: usize) -> Result<PeakMemoryTrial, PeakMemoryError> {
    let (a, b) = make_trial_inputs(size, trial_index)?;
    let mem = fandhe_ai_backend_cpu::CpuMemory::new();
    mem.reset_peak();

    let buf_a = mem.upload(&a)?;
    let buf_b = mem.upload(&b)?;
    let buf_c = mem.alloc_zeroed(&[size, size])?;

    let ops = fandhe_ai_backend_cpu::CpuBackendOps::new();
    // `gemm` 実行区間だけを `alloc_tracker` の計測窓に切り出す（`upload`／
    // `alloc_zeroed` 自体のホスト側確保を GEMM 内部確保と混同しないため。
    // PR #370 codex-review 指摘 P1 対応）。`measure` が
    // `MEASUREMENT_LOCK`（`crate::alloc_tracker` モジュール冒頭
    // 「スレッド安全性」参照）で起点記録〜ピーク読み出しを直列化する
    // ため、並行する trial 実行同士でグローバル状態（`PEAK_BYTES`／
    // `BASELINE_BYTES`）を破壊し合わない（PR #370 codex-review 指摘 P1
    // 再指摘対応）。
    let start = Instant::now();
    let (gemm_result, gemm_alloc_peak_bytes) = crate::alloc_tracker::measure(|| ops.gemm(&a, &b));
    let gemm_secs = start.elapsed().as_secs_f64();
    gemm_result?;

    let peak_bytes = mem.peak_allocated_bytes();
    drop(buf_a);
    drop(buf_b);
    drop(buf_c);
    let allocated_after_drop_bytes = mem.allocated_bytes();

    Ok(PeakMemoryTrial {
        peak_bytes,
        allocated_after_drop_bytes,
        gemm_secs,
        vm_hwm_bytes: read_vm_hwm_bytes(),
        gemm_alloc_peak_bytes,
    })
}

/// CUDA バックエンド 1 trial 分の計測。`CudaDevice::is_available()` を経由し、
/// driver/NVRTC 不在時は panic せず [`PeakMemoryError::DeviceUnavailable`]
/// を返す（`startup::run_cuda` と同型の fail-closed 契約。
/// `backend-cuda/src/device.rs` の動的ロードゲート参照）。
fn run_cuda_trial(size: usize, trial_index: usize) -> Result<PeakMemoryTrial, PeakMemoryError> {
    if !fandhe_ai_backend_cuda::CudaDevice::is_available() {
        return Err(PeakMemoryError::DeviceUnavailable(
            "CUDA driver が利用不可（is_available() == false）".to_string(),
        ));
    }
    let device = fandhe_ai_backend_cuda::CudaDevice::new(0)
        .map_err(|e| PeakMemoryError::DeviceUnavailable(format!("CudaDevice::new 失敗: {e}")))?;

    let (a, b) = make_trial_inputs(size, trial_index)?;
    let mem = fandhe_ai_backend_cuda::CudaMemory::new(&device);
    mem.reset_peak();

    let buf_a = mem.upload(&a)?;
    let buf_b = mem.upload(&b)?;
    let buf_c = mem.alloc_zeroed(&[size, size])?;

    let ops = fandhe_ai_backend_cuda::CudaBackendOps::new(device.ordinal());
    let start = Instant::now();
    ops.gemm(&a, &b)?;
    let gemm_secs = start.elapsed().as_secs_f64();

    let peak_bytes = mem.peak_allocated_bytes();
    drop(buf_a);
    drop(buf_b);
    drop(buf_c);
    let allocated_after_drop_bytes = mem.allocated_bytes();

    Ok(PeakMemoryTrial {
        peak_bytes,
        allocated_after_drop_bytes,
        gemm_secs,
        vm_hwm_bytes: read_vm_hwm_bytes(),
        // CUDA driver 確保（`cudarc`）は Rust の `GlobalAlloc` を経由しない
        // ため `alloc_tracker` では観測できない（`crate::alloc_tracker`
        // モジュール冒頭「適用範囲」参照。PR #370 codex-review 指摘 P1）。
        gemm_alloc_peak_bytes: None,
    })
}

/// Metal バックエンド 1 trial 分の計測。`cfg(target_os = "macos")` 限定
/// （`startup::run_metal` と同型。非 macOS では `backend-metal` クレート
/// 自体がビルド対象に入らない。`crates/bench-harness/Cargo.toml` 参照）。
#[cfg(target_os = "macos")]
fn run_metal_trial(size: usize, trial_index: usize) -> Result<PeakMemoryTrial, PeakMemoryError> {
    let (a, b) = make_trial_inputs(size, trial_index)?;
    let context = fandhe_ai_backend_metal::MetalContext::new().map_err(|e| {
        PeakMemoryError::DeviceUnavailable(format!("MetalContext::new 失敗: {e:?}"))
    })?;
    let mem = fandhe_ai_backend_metal::MetalMemory::new(context);
    mem.reset_peak();

    let buf_a = mem.upload(&a)?;
    let buf_b = mem.upload(&b)?;
    let buf_c = mem.alloc_zeroed(&[size, size])?;

    let ops = fandhe_ai_backend_metal::MetalBackendOps::new();
    let start = Instant::now();
    ops.gemm(&a, &b)?;
    let gemm_secs = start.elapsed().as_secs_f64();

    let peak_bytes = mem.peak_allocated_bytes();
    drop(buf_a);
    drop(buf_b);
    drop(buf_c);
    let allocated_after_drop_bytes = mem.allocated_bytes();

    Ok(PeakMemoryTrial {
        peak_bytes,
        allocated_after_drop_bytes,
        gemm_secs,
        vm_hwm_bytes: read_vm_hwm_bytes(),
        // Metal（`objc2-metal` の `MTLBuffer`）確保は Rust の `GlobalAlloc`
        // を経由しないため `alloc_tracker` では観測できない
        // （`crate::alloc_tracker` モジュール冒頭「適用範囲」参照。
        // PR #370 codex-review 指摘 P1）。
        gemm_alloc_peak_bytes: None,
    })
}

/// 非 macOS では Metal 未対応を型付きエラーで返す（`startup::run_metal` の
/// `#[cfg(not(target_os = "macos"))]` 分岐と同型）。
#[cfg(not(target_os = "macos"))]
fn run_metal_trial(_size: usize, _trial_index: usize) -> Result<PeakMemoryTrial, PeakMemoryError> {
    Err(PeakMemoryError::DeviceUnavailable(
        "Metal バックエンドは macOS 限定（cfg(target_os = \"macos\")）のため本 OS では未対応"
            .to_string(),
    ))
}

/// 指定バックエンドの GEMM ピークメモリを計測する（本モジュールの公開入口）。
///
/// `peak_memory_bench` CLI（`src/bin/peak_memory_bench.rs`）・スモークテスト
/// （`tests/peak_memory_smoke.rs`）から呼ばれる。
pub fn run_peak_memory(config: &PeakMemoryConfig) -> Result<PeakMemoryReport, PeakMemoryError> {
    let mut samples = Vec::with_capacity(config.trials());
    for i in 0..config.trials() {
        let trial = match config.backend() {
            PeakMemoryBackend::Cpu => run_cpu_trial(config.size(), i)?,
            PeakMemoryBackend::Cuda => run_cuda_trial(config.size(), i)?,
            PeakMemoryBackend::Metal => run_metal_trial(config.size(), i)?,
        };
        samples.push(trial);
    }
    PeakMemoryReport::from_trials(config.backend(), config.size(), samples)
}

/// `peak_memory_bench` CLI（`src/bin/peak_memory_bench.rs`）の解析済み引数。
///
/// パース処理・その回帰テストをライブラリ側へ寄せる（イシュー #161 PR #357
/// codex-review 再指摘 P1 対応）: `peak_memory_bench.rs` は本モジュール冒頭
/// 「`gemm_alloc_peak_bytes` は `TrackingAllocator` がプロセスの
/// `#[global_allocator]` として有効化されている場合に限り `Some`」という契約を
/// 満たすため、`#[cfg(test)]` を付けず常時 `TrackingAllocator` を宣言する
/// （通常実行でも計測対象のため）。そのため同バイナリの `cargo test --bin`
/// プロセスは全 `#[test]` がこの `TrackingAllocator` を共有し、libtest の
/// 既定並列実行下では他テストのスレッド起動・終了に伴う確保・解放が
/// `gemm_alloc_peak_bytes` を計測する `measure()` 区間へ干渉しうる
/// （`alloc_tracker` モジュール冒頭「スレッド安全性」参照。テスト関数内
/// ロックでは `#[test]` 本体の外側で起こる干渉を防げない）。CLI の引数
/// パース・検証バイパス回帰は `TrackingAllocator` の実測値に依存しないため、
/// ロジックごと本モジュールへ移し `cargo test --lib`（`TrackingAllocator` を
/// 宣言しない通常のテストバイナリ）側でテストする。バイナリ本体は
/// [`parse_peak_memory_cli_args`]／[`emit_peak_memory_report`] を呼ぶだけの
/// 薄い `main()` になり、`cargo test --bin peak_memory_bench` は `#[test]` を
/// 一切持たなくなる（干渉源そのものを消す）。
#[derive(Debug)]
pub struct PeakMemoryCliArgs {
    pub backend: PeakMemoryBackend,
    pub size: usize,
    pub trials: usize,
    pub out: Option<std::path::PathBuf>,
}

/// `--backend <cpu|cuda|metal>`（必須）・`--size <N>`（既定 [`DEFAULT_GEMM_SIZE`]）・
/// `--trials <N>`（既定 [`DEFAULT_PEAK_MEMORY_TRIALS`]）・`--out <path>`
/// （省略時は標準出力）を許可リスト方式で解釈する（`startup::parse_args` と同型。
/// `.claude/rules/security.md` A03）。
pub fn parse_peak_memory_cli_args(raw: &[String]) -> Result<PeakMemoryCliArgs, String> {
    let mut backend: Option<PeakMemoryBackend> = None;
    let mut size = DEFAULT_GEMM_SIZE;
    let mut trials = DEFAULT_PEAK_MEMORY_TRIALS;
    let mut out: Option<std::path::PathBuf> = None;

    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--backend" => {
                let value = raw
                    .get(i + 1)
                    .ok_or_else(|| "--backend には値が必要".to_string())?;
                backend = Some(PeakMemoryBackend::parse(value).map_err(|e| e.to_string())?);
                i += 2;
            }
            "--size" => {
                let value = raw
                    .get(i + 1)
                    .ok_or_else(|| "--size には値が必要".to_string())?;
                size = value
                    .parse::<usize>()
                    .map_err(|e| format!("--size の値が不正（{value:?}）: {e}"))?;
                i += 2;
            }
            "--trials" => {
                let value = raw
                    .get(i + 1)
                    .ok_or_else(|| "--trials には値が必要".to_string())?;
                trials = value
                    .parse::<usize>()
                    .map_err(|e| format!("--trials の値が不正（{value:?}）: {e}"))?;
                i += 2;
            }
            "--out" => {
                let value = raw
                    .get(i + 1)
                    .ok_or_else(|| "--out には値が必要".to_string())?;
                out = Some(std::path::PathBuf::from(value));
                i += 2;
            }
            other => {
                return Err(format!(
                    "未知の引数 {other:?}（--backend / --size / --trials / --out のいずれかを指定）"
                ));
            }
        }
    }

    let backend = backend.ok_or_else(|| "--backend <cpu|cuda|metal> は必須".to_string())?;
    Ok(PeakMemoryCliArgs {
        backend,
        size,
        trials,
        out,
    })
}

/// `peak_memory_bench` CLI の公式出力（標準出力または `--out` 先ファイル）を
/// 確定する唯一の経路（`main()` から呼ばれる）。
///
/// 出力先の分岐（`out`）に関わらず、シリアライズより前に必ず
/// [`PeakMemoryReport::require_gemm_alloc_tracked`] を通す。`--out` 経路は
/// `write_to_file()` が内部で同メソッドを呼ぶため元々 fail-closed だったが、
/// 標準出力（`out: None`。`make peak-memory-bench` の既定経路）は `to_json()` の
/// みで検証をバイパスできてしまい、CPU の `gemm_alloc_peak_bytes` が `None` または
/// 過小でも「公式記録」として出力されてしまっていた（PR #370 codex-review 再指摘
/// P1 対応）。検証を `main()` の分岐の外側・本関数の入口に集約することで、将来
/// 出力経路を追加しても検証バイパスの抜け道にならない構造にする。
pub fn emit_peak_memory_report(
    report: &PeakMemoryReport,
    out: Option<&Path>,
) -> Result<(), PeakMemoryError> {
    report.require_gemm_alloc_tracked()?;
    match out {
        Some(path) => report.write_to_file(path),
        None => {
            let json = report.to_json()?;
            println!("{json}");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_known_backends() {
        assert_eq!(
            PeakMemoryBackend::parse("cpu").unwrap(),
            PeakMemoryBackend::Cpu
        );
        assert_eq!(
            PeakMemoryBackend::parse("cuda").unwrap(),
            PeakMemoryBackend::Cuda
        );
        assert_eq!(
            PeakMemoryBackend::parse("metal").unwrap(),
            PeakMemoryBackend::Metal
        );
    }

    #[test]
    fn parse_rejects_unknown_backend() {
        assert!(PeakMemoryBackend::parse("gpu").is_err());
    }

    #[test]
    fn theoretical_min_bytes_matches_known_4096_value() {
        // REQ-14 の代表ワークロード（M=N=K=4096, f32）の理論最小ワーキング
        // セットは 3 × 4096² × 4 バイト = 201,326,592 バイト（192MiB）。
        assert_eq!(theoretical_min_bytes(4096).unwrap(), 201_326_592);
    }

    #[test]
    fn config_rejects_zero_size() {
        assert!(PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 0, 5).is_err());
    }

    #[test]
    fn config_rejects_zero_trials() {
        assert!(PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 256, 0).is_err());
    }

    #[test]
    fn config_rejects_size_over_max() {
        assert!(PeakMemoryConfig::new(PeakMemoryBackend::Cpu, MAX_GEMM_SIZE + 1, 5).is_err());
    }

    #[test]
    fn config_rejects_trials_over_max() {
        assert!(
            PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 256, MAX_PEAK_MEMORY_TRIALS + 1).is_err()
        );
    }

    #[test]
    fn config_accepts_boundary_values() {
        assert!(PeakMemoryConfig::new(PeakMemoryBackend::Cpu, MAX_GEMM_SIZE, 1).is_ok());
        assert!(PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 1, MAX_PEAK_MEMORY_TRIALS).is_ok());
    }

    #[test]
    fn byte_len_for_square_overflow_is_rejected() {
        assert!(byte_len_for_square(usize::MAX).is_err());
    }

    /// 受け入れ条件の直接検証: CPU バックエンドで GEMM 256³ のピーク値が
    /// 理論最小ワーキングセットちょうど（`MemoryOps` 経由 3 バッファのみ
    /// 計上のため決定的）であり、drop 後にリークがないことを確認する。
    /// 4096³ はスモークテストとして実行するには重いため、CI 実行可能な
    /// 小サイズを用いる（`tests/peak_memory_smoke.rs` と同じ判断）。
    #[test]
    fn run_peak_memory_cpu_matches_theoretical_minimum_for_small_size() {
        let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 64, 3).unwrap();
        let report = run_peak_memory(&config).unwrap();

        let expected = theoretical_min_bytes(64).unwrap();
        assert_eq!(report.theoretical_min_bytes, expected);
        assert_eq!(report.trials, 3);
        assert_eq!(report.samples.len(), 3);
        for trial in &report.samples {
            assert_eq!(
                trial.peak_bytes, expected,
                "CPU の peak_bytes は MemoryOps 経由 3 バッファのみ計上のため決定的なはず"
            );
            assert_eq!(
                trial.allocated_after_drop_bytes, 0,
                "drop 後は allocated_bytes が 0 に戻るはず（リーク検査）"
            );
            assert!(trial.gemm_secs.is_finite() && trial.gemm_secs >= 0.0);
        }
    }

    // `run_peak_memory_cpu_gemm_alloc_peak_bytes_reflects_real_gemm_allocation`・
    // `require_gemm_alloc_tracked_accepts_cpu_report_with_tracking_active`
    // （実測 `gemm_alloc_peak_bytes` に依存するテスト）は、本テストバイナリ
    // （`cargo test --lib`）が `TrackingAllocator` を `#[global_allocator]`
    // 宣言しなくなったため（イシュー #161 PR #357 codex-review 再指摘 P1
    // 対応。`crate::alloc_tracker` モジュール冒頭「スレッド安全性」参照）、
    // `tests/alloc_tracker_serial.rs`（`harness = false` の専用プロセス）へ
    // 移設した。libtest の並列実行を経由しない唯一の場所であるため。

    #[test]
    fn report_json_roundtrip_preserves_validation() {
        let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 32, 2).unwrap();
        let report = run_peak_memory(&config).unwrap();
        let json = report.to_json().unwrap();
        let restored = PeakMemoryReport::from_json(&json).unwrap();
        assert_eq!(report, restored);
    }

    #[test]
    fn validate_rejects_leak() {
        let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 32, 1).unwrap();
        let mut report = run_peak_memory(&config).unwrap();
        report.samples[0].allocated_after_drop_bytes = 128;
        assert!(matches!(
            report.validate(),
            Err(PeakMemoryError::ProtocolViolation(_))
        ));
    }

    #[test]
    fn validate_rejects_tampered_quartiles() {
        let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 32, 1).unwrap();
        let mut report = run_peak_memory(&config).unwrap();
        report.peak_bytes.median += 1;
        assert!(matches!(
            report.validate(),
            Err(PeakMemoryError::ProtocolViolation(_))
        ));
    }

    /// [`PeakMemoryReport::require_gemm_alloc_tracked`] の fail-closed 挙動: 旧スキーマ
    /// （`gemm_alloc_peak_bytes` 実測前）の CPU レポートを模した `None` を fail-closed で
    /// 拒否することを確認する（本イシューの codex-review P1 指摘そのものの回帰テスト:
    /// 「旧スキーマの実測データが現行スキーマの結果として黙って正常扱いされる」ことを防ぐ）。
    /// `gemm_alloc_peak_bytes` を手動で `None` に上書きするため、`TrackingAllocator` が
    /// `#[global_allocator]` として有効化されているかに依存しない
    /// （`cargo test --lib` で実行可能）。
    #[test]
    fn require_gemm_alloc_tracked_rejects_none_for_cpu() {
        let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 32, 1).unwrap();
        let mut report = run_peak_memory(&config).unwrap();
        report.samples[0].gemm_alloc_peak_bytes = None;
        assert!(matches!(
            report.require_gemm_alloc_tracked(),
            Err(PeakMemoryError::ProtocolViolation(_))
        ));
    }

    /// [`PeakMemoryReport::require_gemm_alloc_tracked`] は出力バッファ分未満の過小な
    /// 観測値（GEMM 実行区間の実ヒープ確保を捕捉できていない状態）も拒否する。
    /// 上のテストと同様、手動上書きのため `TrackingAllocator` 有効化に依存しない。
    #[test]
    fn require_gemm_alloc_tracked_rejects_value_below_output_buffer() {
        let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 32, 1).unwrap();
        let mut report = run_peak_memory(&config).unwrap();
        report.samples[0].gemm_alloc_peak_bytes = Some(1);
        assert!(matches!(
            report.require_gemm_alloc_tracked(),
            Err(PeakMemoryError::ProtocolViolation(_))
        ));
    }

    /// `write_to_file` は素通しではなく必ず `require_gemm_alloc_tracked` を通す
    /// （直接呼び出しではなくファイル I/O 経由でも fail-closed が効くことの確認）。
    /// 手動上書きのため `TrackingAllocator` 有効化に依存しない。
    #[test]
    fn write_to_file_rejects_report_missing_gemm_alloc_tracking() {
        let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 32, 1).unwrap();
        let mut report = run_peak_memory(&config).unwrap();
        report.samples[0].gemm_alloc_peak_bytes = None;

        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "peak_memory_write_to_file_test_{}.json",
            std::process::id()
        ));
        let result = report.write_to_file(&path);
        // 失敗経路の確認が主目的のため、書き出しに成功してしまった場合に備えて
        // 後始末する（本番経路ではないテストコード限定の best-effort クリーンアップ）。
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(PeakMemoryError::ProtocolViolation(_))));
    }

    // --- 以下、`peak_memory_bench` CLI（`src/bin/peak_memory_bench.rs`）の
    // 引数パース・出力経路回帰テスト（イシュー #161 PR #357 codex-review
    // 再指摘 P1 対応で本モジュールへ移設。`parse_peak_memory_cli_args`
    // doc comment 参照）。

    #[test]
    fn parse_peak_memory_cli_args_requires_backend() {
        let err = parse_peak_memory_cli_args(&[]).unwrap_err();
        assert!(err.contains("--backend"));
    }

    #[test]
    fn parse_peak_memory_cli_args_accepts_full_form() {
        let raw: Vec<String> = [
            "--backend",
            "cpu",
            "--size",
            "128",
            "--trials",
            "3",
            "--out",
            "out.json",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let args = parse_peak_memory_cli_args(&raw).unwrap();
        assert_eq!(args.backend, PeakMemoryBackend::Cpu);
        assert_eq!(args.size, 128);
        assert_eq!(args.trials, 3);
        assert_eq!(args.out, Some(std::path::PathBuf::from("out.json")));
    }

    #[test]
    fn parse_peak_memory_cli_args_uses_defaults_when_omitted() {
        let raw: Vec<String> = ["--backend", "cpu"].into_iter().map(String::from).collect();
        let args = parse_peak_memory_cli_args(&raw).unwrap();
        assert_eq!(args.size, DEFAULT_GEMM_SIZE);
        assert_eq!(args.trials, DEFAULT_PEAK_MEMORY_TRIALS);
        assert_eq!(args.out, None);
    }

    #[test]
    fn parse_peak_memory_cli_args_rejects_unknown_flag() {
        let raw: Vec<String> = ["--bogus", "x"].into_iter().map(String::from).collect();
        assert!(parse_peak_memory_cli_args(&raw).is_err());
    }

    #[test]
    fn parse_peak_memory_cli_args_rejects_invalid_backend() {
        let raw: Vec<String> = ["--backend", "gpu"].into_iter().map(String::from).collect();
        assert!(parse_peak_memory_cli_args(&raw).is_err());
    }

    /// [`emit_peak_memory_report`] の標準出力経路（`out: None`）が
    /// `gemm_alloc_peak_bytes` 欠落を fail-closed で拒否することの回帰テスト
    /// （PR #370 codex-review 再指摘 P1: 「`--out` 省略時は `to_json()` のみが
    /// 呼ばれ検証をバイパスできていた」欠陥そのものの再現）。手動上書きのため
    /// `TrackingAllocator` 有効化に依存しない。
    #[test]
    fn emit_peak_memory_report_rejects_missing_gemm_alloc_tracking_via_stdout() {
        let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 32, 1).unwrap();
        let mut report = run_peak_memory(&config).unwrap();
        report.samples[0].gemm_alloc_peak_bytes = None;

        assert!(matches!(
            emit_peak_memory_report(&report, None),
            Err(PeakMemoryError::ProtocolViolation(_))
        ));
    }

    /// [`emit_peak_memory_report`] の `--out` 経路でも同じ欠陥形が fail-closed で
    /// 拒否されることを確認する（標準出力経路と対称であることの確認）。
    #[test]
    fn emit_peak_memory_report_rejects_missing_gemm_alloc_tracking_via_out_file() {
        let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 32, 1).unwrap();
        let mut report = run_peak_memory(&config).unwrap();
        report.samples[0].gemm_alloc_peak_bytes = None;

        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "emit_peak_memory_report_rejects_missing_gemm_alloc_tracking_via_out_file_{}.json",
            std::process::id()
        ));
        let result = emit_peak_memory_report(&report, Some(&path));

        assert!(matches!(result, Err(PeakMemoryError::ProtocolViolation(_))));
        assert!(!path.exists(), "検証失敗時にファイルを書き出してはならない");
    }
}
