//! 学習 step の CUDA Graph capture／instantiate／launch 経路（opt-in・
//! 既定 OFF。イシュー #1349・親 #1348・ルート #1341 → #1269）。
//!
//! # スコープ（設計は `docs/backend-cuda-graph-step-capture-design.md`）
//!
//! 本モジュールが capture できるのは学習 step のうち **update 区間
//! （`BackendOps::sgd_step_device_tracked`）のみ**であり、forward／
//! backward は対象外（既存データパスがホスト境界・pageable H2D を含み
//! capture の前提〈同一ストリーム上の driver 呼び出しのみで完結する
//! こと〉を満たさないため。同 design doc §1・§3.2）。`exec update`
//! （`cuGraphExecUpdate_v2`）は cudarc の安全 API に存在せず `unsafe`
//! 導入がユーザー承認事項のため本モジュールでは実装しない（同 doc
//! §3。構造変化時は再 capture + 再 instantiate で置き換える）。
//!
//! # 推論 forward チェーンの capture（イシュー #2115。別機構）
//!
//! 学習 step の update 区間（上記・`STEP_GRAPH_MODE`・#1349）とは**別の
//! opt-in**として、推論 forward チェーン（`DeviceParamStore::
//! predict_device_chain` の層カーネル列）を 1 本の graph へ capture し、
//! 2 回目以降は graph launch 1 回で再生する経路を持つ
//! （`run_captured_linear_chain`・opt-in は [`infer_graph_enabled`]・
//! 環境変数 `FANDHE_AI_CUDA_GRAPH_INFER`・既定 OFF）。層ごとの kernel
//! launch と出力バッファの `alloc_zeroed`・入力の `upload` 新規確保を
//! 除去することが目的で、graph・staging・中間バッファは thread-local の
//! `INFER_GRAPHS` が所有する。**stream 種別（created stream）の決定は
//! 学習側と共通**（`device.rs` が `step_graph_mode` と本 opt-in の OR で
//! 決める。ordinal ごとに sticky）。設計・判定規則は
//! `docs/perf/infer-chain-graphcapture-cuda-ab.md`・
//! `docs/inference-chain-single-sync-design.md` §11。
//!
//! # opt-in フラグと共有ストリーム（`device.rs` との契約）
//!
//! `STEP_GRAPH_MODE` は 3 値（`GraphMode::Off`／`GraphMode::StreamOnly`／
//! `GraphMode::On`）。`device.rs::CudaDevice::new` は
//! `step_graph_mode` が `StreamOnly` 以上のときのみ `ctx.new_stream()`
//! （capture 可能な非 legacy ストリーム）を保持し、`Off` のときは現行の
//! `default_stream()`（legacy NULL stream。capture 不可）を維持する。
//! `StreamOnly` は「created stream の event 管理コストのみを計測したい」
//! 診断用の中間状態（イシュー #1350 が「ストリーム種別の効果」と
//! 「capture の効果」を分離計測するために使う。design doc §9）で、
//! capture 自体は行わない（`captured_segment_key`（`ops.rs::CudaBackendOps::captured_segment_key`）相当の判定は `On`
//! のみ `Some` を返す）。
//!
//! **フラグは最初の CUDA デバイス初期化より前に設定する必要がある**
//! （`CudaDevice` は ordinal ごとに 1 回だけ `context_cache::cached_device`
//! で構築され、以後生存し続けるため。design doc §4.1）。
//!
//! # thread-local graph キャッシュ
//!
//! `CudaGraph`（`cudarc::driver::CudaGraph`）は `Send`/`Sync` を実装せず
//! （NVIDIA の規定でも graph オブジェクトはスレッド非安全）、プロセス
//! ワイド static へは `unsafe impl Send` なしに置けない。そのため
//! `STEP_GRAPHS` は `thread_local!` とし、[`SegmentKey`] をキーに
//! 最大 `MAX_CACHED_GRAPHS_PER_THREAD` 件まで保持する（超過時は挿入順
//! 最古を evict）。

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use cudarc::driver::CudaGraph;
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

use fandhe_ai_tensor_core::buffer::{DeviceBuffer, DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{
    BackendOps, DispatchFailureCell, SegmentKey, SegmentRun, SgdStepConfig, Tensor,
};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::memory::{CudaBufferHandle, CudaMemory, CudaStorage};
use crate::ops::CudaBackendOps;

/// CUDA Graph step capture の opt-in 状態（モジュール冒頭コメント参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GraphMode {
    /// 既定。legacy stream・capture なし（本イシュー導入前と挙動不変）。
    Off,
    /// created stream で初期化するが capture はしない（イシュー #1350
    /// の分離計測用診断状態）。
    StreamOnly,
    /// created stream で初期化し、update 区間を capture・再利用する。
    On,
}

impl GraphMode {
    fn as_u8(self) -> u8 {
        match self {
            GraphMode::Off => 0,
            GraphMode::StreamOnly => 1,
            GraphMode::On => 2,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => GraphMode::StreamOnly,
            2 => GraphMode::On,
            _ => GraphMode::Off,
        }
    }

    /// `device.rs::CudaDevice::new` が created stream を選ぶべきかどうか
    /// （`StreamOnly` 以上）。
    ///
    /// `internal-diagnostics` feature 有効時は未使用になる（codex-review
    /// P0 再々指摘対応・PR #1390 マージ時是正）: `device.rs::CudaDevice::
    /// new` は同 feature が有効な間、`unsafe { ctx.disable_event_tracking()
    /// }` を呼ぶ分岐自体を cfg でコンパイル対象から外し `StreamKind::
    /// Legacy` に固定するため、本メソッドを呼ぶ側の分岐が存在しなくなる
    /// （`device.rs::CudaDevice::new` の cfg 分岐コメント参照）。
    #[cfg_attr(feature = "internal-diagnostics", allow(dead_code))]
    pub(crate) fn requires_created_stream(self) -> bool {
        !matches!(self, GraphMode::Off)
    }
}

/// `0`（未設定＝環境変数フォールバックを使う）を挟むための 3 値
/// エンコーディング。API setter が呼ばれた時点でこの `OnceLock` 相当の
/// 「明示設定済みフラグ」を兼ねる（下記 `EXPLICIT` 参照）。
static STEP_GRAPH_MODE: AtomicU8 = AtomicU8::new(0);
/// API setter（[`set_step_graph_enabled`]）が一度でも呼ばれたかどうか。
/// 呼ばれていれば環境変数より API 設定を優先する（モジュール冒頭
/// コメント・facade 公開 API doc の契約）。
static EXPLICIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// イシュー #1350: `framework-compare` の `bench-fandhe`（crates.io
/// `fandhe-ai =0.7.0` ピンのため本モジュールの内部型を直接触れない）が
/// launch 固定費の実測（`step_total`／`device_update` の A/B と launch
/// 回数）を行うための、プロセスワイド診断カウンタ 3 種。`internal-
/// diagnostics` feature には載せない（`GraphMode::requires_created_stream`
/// 呼び出し側が同 feature 下で `StreamKind::Legacy` に固定され capture
/// 経路へ到達しないため、載せても常に 0 のまま無意味になる。B4）。
///
/// 値そのものは性能に影響しない `AtomicU64` カウントのみ（`Ordering::
/// Relaxed` で十分。診断用途で他メモリ操作との順序保証を必要としない）。
static CAPTURED_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REPLAYED_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GRAPH_LAUNCH_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// [`crate::sgd::CudaSgd::run`] の launch 成功時に加算される SGD カーネル
/// launch 回数（graph capture 中の 1 回・capture 後の replay では
/// カーネル自体は launch されず graph launch に置き換わる点に注意。
/// つまり ON では「capture 時のウォームアップ launch＋capture 内 1 回」
/// のみがここに計上され、以後の replay 分は `GRAPH_LAUNCH_COUNT` 側に
/// 計上される。OFF では毎 step 1 回ずつ加算される）。
static SGD_KERNEL_LAUNCH_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`crate::sgd`] から呼ばれる: SGD カーネル launch が成功した回数を
/// 加算する（graph capture 中の launch・capture 外の通常 launch の
/// いずれも計上する。ドキュメントは [`SGD_KERNEL_LAUNCH_COUNT`] 参照）。
pub(crate) fn record_sgd_kernel_launch() {
    SGD_KERNEL_LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// 公開 API（`facade::cuda_graph_step_mode` 経由で再公開）向けの
/// `GraphMode`（クレート内部限定）写像。内部 `GraphMode` をそのまま公開すると
/// `pub(crate)` の可視性契約が崩れるため、値が同じだけの独立した
/// 公開 enum を用意する（`crate::precision` の公開型と同型の設計）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepGraphMode {
    /// 既定。legacy stream・capture なし。
    Off,
    /// created stream で初期化するが capture はしない（イシュー #1350
    /// の診断状態。モジュール冒頭コメント参照）。
    StreamOnly,
    /// created stream で初期化し、update 区間を capture・再利用する。
    On,
}

impl From<GraphMode> for StepGraphMode {
    fn from(m: GraphMode) -> Self {
        match m {
            GraphMode::Off => StepGraphMode::Off,
            GraphMode::StreamOnly => StepGraphMode::StreamOnly,
            GraphMode::On => StepGraphMode::On,
        }
    }
}

/// 現在の opt-in モードを公開型で返す（`facade::cuda_graph_step_mode`
/// の実体。イシュー #1350: `bench-fandhe` が `--graph stream-only` 起動時
/// に環境変数が実際に反映されたかを確認するために使う。API setter
/// （[`set_step_graph_enabled`]）を呼ぶと `stream-only` は選べなくなる
/// 契約〈モジュール冒頭コメント〉があるため、この確認は環境変数経由の
/// 起動でのみ意味を持つ）。
pub fn step_graph_mode_public() -> StepGraphMode {
    step_graph_mode().into()
}

/// launch 固定費の診断用スナップショット（POD。イシュー #1350）。
/// `facade::cuda_graph_step_stats` の実体。値はプロセス起動からの累積
/// カウントで、計測ウィンドウの前後で差分を取ることを想定している
/// （bench 側は計測ループの前後で 2 回呼ぶのではなく、計測後に 1 回
/// だけ呼んで record に載せる設計〈計測時間そのものへの影響を避ける〉
/// ため、実際には「計測プロセス全体の累積値」がそのまま記録される。
/// bench-fandhe は 1 計測 = 1 プロセス起動のため、この累積値は当該
/// 計測分の合計と一致する）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepGraphStats {
    pub mode: StepGraphMode,
    /// [`SegmentRun::Captured`] を返した回数（新規 capture の回数）。
    pub captured: u64,
    /// [`SegmentRun::Replayed`] を返した回数（既存 graph の再生回数）。
    pub replayed: u64,
    /// `CudaGraph::launch()` が成功した回数（capture 直後の初回 launch
    /// ＋以後の replay launch の合計。`captured + replayed` と一致する）。
    pub graph_launches: u64,
    /// SGD カーネル自体の launch 成功回数（`record_sgd_kernel_launch`
    /// 〈クレート内部限定〉参照。OFF では毎 step 1・ON では capture 時のみ増える）。
    pub sgd_kernel_launches: u64,
}

/// [`StepGraphStats`] の現在値を返す（`facade::cuda_graph_step_stats`
/// の実体。イシュー #1350）。
pub fn step_graph_stats() -> StepGraphStats {
    StepGraphStats {
        mode: step_graph_mode().into(),
        captured: CAPTURED_COUNT.load(Ordering::Relaxed),
        replayed: REPLAYED_COUNT.load(Ordering::Relaxed),
        graph_launches: GRAPH_LAUNCH_COUNT.load(Ordering::Relaxed),
        sgd_kernel_launches: SGD_KERNEL_LAUNCH_COUNT.load(Ordering::Relaxed),
    }
}

/// 環境変数 `FANDHE_AI_CUDA_GRAPH_STEP` の初回参照結果（`OnceLock` で
/// プロセス生存期間中 1 回だけ読む。イシュー #1349 §4.6。framework-compare
/// の `bench-fandhe` は crates.io ピン版のためライブラリの新規公開 API
/// を呼べない〈#1350 が opt-in を切り替えるための代替経路〉）。
static ENV_MODE: OnceLock<GraphMode> = OnceLock::new();

/// 環境変数の値文字列を [`GraphMode`] へ解釈する（純粋関数。`std::env`
/// を読まないため、環境変数の設定順序に依存する `cargo test` の
/// 既定並列実行下でも安全に単体テストできる）。許容値は `1`／`true`
/// （[`GraphMode::On`]）・`stream-only`（[`GraphMode::StreamOnly`]）の
/// 完全一致のみ（OWASP A03: 未知の値は fail-closed で `Off` に倒す。
/// 値をログ・エラー文へエコーしない）。
fn parse_env_value(raw: &str) -> GraphMode {
    match raw {
        "1" | "true" => GraphMode::On,
        "stream-only" => GraphMode::StreamOnly,
        _ => GraphMode::Off,
    }
}

fn env_mode() -> GraphMode {
    *ENV_MODE.get_or_init(|| match std::env::var("FANDHE_AI_CUDA_GRAPH_STEP") {
        Ok(raw) => parse_env_value(&raw),
        Err(_) => GraphMode::Off,
    })
}

/// 現在の opt-in モードを返す（API 明示設定 > 環境変数 > 既定 OFF の
/// 優先順位。モジュール冒頭コメント参照）。
pub(crate) fn step_graph_mode() -> GraphMode {
    if EXPLICIT.load(Ordering::SeqCst) {
        GraphMode::from_u8(STEP_GRAPH_MODE.load(Ordering::SeqCst))
    } else {
        env_mode()
    }
}

/// `facade::set_cuda_graph_step_enabled` から委譲される opt-in スイッチ
/// （`crate::precision::set_tf32_gemm_enabled` と同型）。`true` で
/// `GraphMode::On`・`false` で `GraphMode::Off` を明示設定する
/// （`stream-only` は API からは選べない診断専用値。環境変数のみで
/// 選択する）。
pub fn set_step_graph_enabled(enabled: bool) {
    STEP_GRAPH_MODE.store(
        if enabled {
            GraphMode::On
        } else {
            GraphMode::Off
        }
        .as_u8(),
        Ordering::SeqCst,
    );
    EXPLICIT.store(true, Ordering::SeqCst);
}

/// 現在の opt-in 状態を返す（既定 `false`）。
pub fn step_graph_enabled() -> bool {
    matches!(step_graph_mode(), GraphMode::On)
}

/// [`CUgraphInstantiate_flags`] の既定選択（design doc §4.3・F5）。
/// mem alloc ノードを含まない本 graph では不活性だが、
/// `cuGraphInstantiateWithFlags` は必須引数のため明示する。
fn instantiate_flags() -> CUgraphInstantiate_flags {
    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
}

/// thread-local に保持する 1 個の capture 済み graph（挿入順 evict 用に
/// 単調増加のシーケンス番号を添える）。
struct CachedGraph {
    graph: CudaGraph,
    inserted_seq: u64,
}

/// 1 スレッドあたりの graph キャッシュ上限（design doc §4.3）。
const MAX_CACHED_GRAPHS_PER_THREAD: usize = 8;

thread_local! {
    // Cursor Bugbot Medium 指摘対応（PR #1390 マージ時是正）: 旧稿は
    // `SegmentKey`（`generation`・`config_key`・リソース `addr`／
    // `numel`）のみをキーとする単一 `HashMap` だったため、device
    // ordinal を一切区別しなかった。`generation` は ordinal ごとに 0
    // から始まる独立したカウンタ（`context_cache::current_generation`）
    // であり、`addr`（CUDA driver の仮想アドレス）も別デバイス上の
    // 別バッファが同じ値になりうる（driver の確保パターン次第で現実に
    // 起こりうる）ため、同一スレッドが複数 GPU 上で capture を行うと
    // 「別デバイスの graph を誤って replay する」「別デバイスの graph
    // キャッシュを誤って evict する」双方が起こりえた（opt-in は
    // プロセスワイドで全 CUDA device に適用されるため、multi-GPU 構成
    // では実際に到達しうる経路。design doc の前提「1 スレッドが同時に
    // 触るのは 1 ordinal」を型で保証していなかった）。
    //
    // 外側を ordinal でパーティションし、世代不一致の evict（下記
    // `put_cached_graph` の `retain`）も同一 ordinal のサブマップ内に
    // 閉じることで、別 ordinal の graph を replay・evict する経路を
    // 構造的になくす。
    static STEP_GRAPHS: RefCell<HashMap<usize, HashMap<SegmentKey, CachedGraph>>> =
        RefCell::new(HashMap::new());
    static NEXT_SEQ: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn next_seq() -> u64 {
    NEXT_SEQ.with(|c| {
        let v = c.get();
        c.set(v.wrapping_add(1));
        v
    })
}

/// キャッシュから `ordinal`／`key` に一致する graph を取り出す（ヒット
/// 時はエントリを一旦 map から取り除いた「所有」状態で返す。呼び出し
/// 元は launch 後に [`put_cached_graph`] で戻す。`RefCell` の borrow を
/// `body` 実行中に保持しないための take/put 方式。design doc §4.3）。
fn take_cached_graph(ordinal: usize, key: &SegmentKey) -> Option<CudaGraph> {
    STEP_GRAPHS.with(|cache| {
        cache
            .borrow_mut()
            .get_mut(&ordinal)
            .and_then(|per_ordinal| per_ordinal.remove(key))
            .map(|c| c.graph)
    })
}

/// capture・launch 済みの graph をキャッシュへ戻す（新規挿入・世代不一致
/// の陳腐化エントリの evict・上限超過時の最古 evict をまとめて行う。
/// いずれも `ordinal` に対応するサブマップ内に閉じる）。
fn put_cached_graph(ordinal: usize, key: SegmentKey, graph: CudaGraph) {
    STEP_GRAPHS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let cache = cache.entry(ordinal).or_default();
        // 世代不一致（`invalidate` による回復後の新世代）のエントリは
        // もう再利用されないため、ついでに掃除する（無制限増加の防止。
        // design doc §4.3）。同一 ordinal 内に閉じるため他デバイスの
        // エントリへは影響しない。
        cache.retain(|k, _| k.generation == key.generation);
        if cache.len() >= MAX_CACHED_GRAPHS_PER_THREAD
            && !cache.contains_key(&key)
            && let Some(oldest_key) = cache
                .iter()
                .min_by_key(|(_, v)| v.inserted_seq)
                .map(|(k, _)| k.clone())
        {
            cache.remove(&oldest_key);
        }
        cache.insert(
            key,
            CachedGraph {
                graph,
                inserted_seq: next_seq(),
            },
        );
    });
}

/// [`crate::ops::CudaBackendOps::run_captured_sgd_step_segment`] の実体
/// （イシュー #1349）。
///
/// **codex-review P0 指摘対応（旧稿からの 2 つの変更）**:
///
/// 1. **任意クロージャの撤廃**: 旧稿は `resources: &mut [&mut
///    DeviceBuffer<f32>]` と任意クロージャ `body` を受け取っていたが、
///    `body` が `resources` に含まれない外部 `DeviceBuffer<f32>` を
///    クロージャキャプチャ経由で直接触れる抜け道があった
///    （`fandhe_ai_tensor_core::backend_ops::BackendOps::
///    run_captured_sgd_step_segment` doc コメント参照）。本関数は
///    `param`／`grad`／`velocity`（SGD 更新区間が触れる全リソース）を
///    直接引数として受け取り、区間本体（capture 対象のカーネル起動）も
///    本関数が固定的に [`CudaBackendOps::sgd_step_device_tracked`] を
///    呼ぶことで行う。呼び出し元は任意コードを注入できない。
/// 2. **capture 開始前の in_flight ドレイン**: 旧稿は `begin_driver_call`
///    （呼び出しスレッド自身のトークン取得）を `begin_capture_session`
///    より前に呼んでいたため、他スレッドが capture 開始の**直前**に
///    `begin_driver_call` を通過済み（`in_flight` に計上済みだが実際の
///    driver 呼び出しはまだ）だった場合、その呼び出しが capture 開始後に
///    共有ストリームへカーネル起動を発行し、意図せず graph へ混入し
///    うる窓があった。本関数はキャッシュミスの分岐で
///    `context_cache::begin_capture_session`（他スレッドの `in_flight`
///    をドレインしてから返る）を**呼び出しスレッド自身のトークンを
///    1 つも保持していない状態で**呼び、その後に初めて
///    `begin_driver_call` を呼ぶ（`context_cache::begin_capture_session`
///    doc コメントの契約）。
///
/// 手順（モジュール冒頭コメント・design doc §4.3。PR #1390 再修正で
/// ウォームアップの位置を変更）:
/// 1. thread-local キャッシュに `key` があれば `begin_driver_call` で
///    poison／世代検査してから `graph.launch()` して
///    [`SegmentRun::Replayed`] を返す。
/// 2. なければ、まず独立した `begin_driver_call` 境界で SGD カーネルを
///    ウォームアップ（NVRTC コンパイル・モジュールロードを capture の
///    排他区間の外で済ませる。Bugbot Medium 指摘対応）→
///    `begin_capture_session`（in_flight ドレイン完了と同一ロック区間で
///    `capturing_active` を設定済み）→ `begin_driver_call` →
///    `stream.begin_capture` → SGD 更新 1 回 →
///    `stream.end_capture`（成功時は `graph.upload()` を 1 回）→
///    初回 `graph.launch()` → キャッシュへ格納して
///    [`SegmentRun::Captured`] を返す。
///
/// SGD 更新が `Err` を返した場合・`end_capture` 自体が失敗した場合は、
/// capture を安全に終了させたうえで `Err` を返す（graph はキャッシュに
/// 残さない）。空 graph（`end_capture` が `Ok(None)`）は fail-closed
/// エラーとする（design doc §4.3 手順 3）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_captured_sgd_step_segment(
    ordinal: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    key: SegmentKey,
    ops: &CudaBackendOps,
    param: &mut DeviceBuffer<f32>,
    grad: &DeviceBuffer<f32>,
    mut velocity: Option<&mut DeviceBuffer<f32>>,
    config: &SgdStepConfig,
    token: &DispatchFailureCell,
) -> Result<SegmentRun, BackendError> {
    // ① キャッシュヒット: 既存 graph を再生する。
    //
    // `begin_driver_call` は別スレッドの capture と競合しただけでも
    // 一過性の `BackendError::DeviceContextCaptureInProgress` を返しうる
    // （`context_cache::begin_driver_call` doc コメント参照）。この場合
    // graph 自体は破損しておらず再利用可能なため、素朴に `?` で早期
    // return すると take 済みの graph がどこにも戻らずキャッシュから
    // 失われ、次回呼び出しで再 capture・re-instantiate が必要になって
    // しまう（codex-review P2 指摘・PR #1390）。よってエラー発生時は
    // graph をキャッシュへ戻してからエラーを返す。
    if let Some(graph) = take_cached_graph(ordinal, &key) {
        let call_token = match context_cache::begin_driver_call(ordinal, &[key.generation]) {
            Ok(token) => token,
            Err(e) => {
                put_cached_graph(ordinal, key, graph);
                return Err(e);
            }
        };
        let launch_result = graph.launch();
        if let Err(e) = context_cache::observe_driver_result(ordinal, &call_token, launch_result) {
            put_cached_graph(ordinal, key, graph);
            return Err(crate::memory::map_cuda_error(CudaError::Driver(e)));
        }
        GRAPH_LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed);
        REPLAYED_COUNT.fetch_add(1, Ordering::Relaxed);
        put_cached_graph(ordinal, key, graph);
        return Ok(SegmentRun::Replayed);
    }

    // SGD カーネルを確実にコンパイル・ロード済みにする（Cursor Bugbot
    // Medium 指摘対応・PR #1390 再修正）: `sgd_step_device_tracked` は
    // 内部で `context_cache::cached_sgd`（`ordinal` キーの NVRTC
    // コンパイル済みカーネルの singleflight キャッシュ）を参照するが、
    // プロセス内でこの ordinal の SGD が一度も呼ばれていない場合、
    // キャッシュミスにより NVRTC コンパイル＋`cuModuleLoadDataEx`
    // （driver へのモジュールロード）が初回発生する。この初回発生が
    // `stream.begin_capture` 後（＝下記 `body_outcome` 内の
    // `sgd_step_device_tracked` 呼び出し時）まで遅延すると、capture
    // 領域内でモジュールロードという「ストリームに紐づかない driver
    // 操作」が走ることになり、`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` の
    // 下で未定義動作・capture 失敗（ordinal poison）につながりうる。
    //
    // **`begin_capture_session` より前で行う（Bugbot Medium 再指摘:
    // 旧稿は `begin_capture_session`／`begin_driver_call` の後でこの
    // ウォームアップを行っていたため、NVRTC コンパイルという遅い操作の
    // 間ずっと `state.capture` が設定済みのまま——他スレッドの
    // `begin_driver_call` を Unsupported で拒否し続けていた。コンパイル
    // 自体は capture の排他と無関係〈このスレッドの capture 意図さえ
    // 登録されていなければ、他スレッドの通常呼び出しを妨げる理由が
    // ない〉ため、専用の `begin_driver_call` 境界（poison／世代検査を
    // 保ったまま）で先に済ませ、`begin_capture_session` は実際に
    // capture を開始する直前まで呼ばない）**。
    {
        let warmup_token = context_cache::begin_driver_call(ordinal, &[key.generation])?;
        let warmup_result = context_cache::cached_device(ordinal)
            .and_then(|device| context_cache::cached_sgd(&device).map(|_| ()));
        // Cursor Bugbot Medium 指摘対応（PR #1390 マージ時是正）: 旧稿は
        // `warmup_token` を観測に使わず drop していたため、初回 NVRTC
        // コンパイル・モジュールロード（`cached_sgd` がキャッシュミスの
        // 場合に発生）が sticky な driver エラーで失敗しても、この
        // ordinal が poison されなかった（本クレート全体の「最初の
        // driver エラーを観測する」契約からの逸脱。`captured_segment_key`
        // の `context_cache::observe_cuda_result(self.ordinal, &token,
        // self.device_handle_raw())` と同じパターンをここでも適用する）。
        // `observe_cuda_result` は `&CallToken` を要求するため、token を
        // 消費（drop）する前に結果を通す。
        let warmup_result =
            context_cache::observe_cuda_result(ordinal, &warmup_token, warmup_result);
        drop(warmup_token);
        warmup_result.map_err(crate::memory::map_cuda_error)?;
    }

    // ② キャッシュミス: capture する。呼び出しスレッドはこの時点で
    // 当該 ordinal のトークンを 1 つも保持していない
    // （`begin_capture_session` の in_flight ドレイン契約。上記 doc
    // コメント参照）。
    let guard = context_cache::begin_capture_session(ordinal)?;
    let _token = match context_cache::begin_driver_call(ordinal, &[key.generation]) {
        Ok(t) => t,
        Err(e) => {
            drop(guard);
            return Err(e);
        }
    };

    // codex-review P0 再指摘対応（Cursor Bugbot High。PR #1390 再修正）:
    // `context_cache::begin_buffer_release`（`memory.rs::
    // CudaBufferHandle::Drop` が使う）が別スレッドからの解放を正しく
    // 駐機できるよう、`state.capturing_active` は `begin_capture_session`
    // が in_flight ドレイン完了と同一ロック区間内で既に立てている
    // （`begin_capture_session` doc コメント「P0 再修正」参照。以前は
    // 本関数が `stream.begin_capture()` 成功後に別関数
    // `mark_capture_active` を呼んで立てていたが、drain 完了からこの
    // 呼び出しまでの間に競合窓が生じるため撤廃した）。
    let begin_result =
        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    if let Err(e) = context_cache::observe_driver_result(ordinal, &_token, begin_result) {
        drop(guard);
        return Err(crate::memory::map_cuda_error(CudaError::Driver(e)));
    }

    // SGD 更新本体の実行を `catch_unwind` で包み、panic（unwind）した
    // 場合でも直後で必ず `end_capture` を呼んでから、panic を
    // `BackendError` へ変換して呼び出し元の `Result` 経由の失敗処理へ
    // 合流させる（codex-review P1・Cursor Bugbot 指摘: body が panic
    // すると driver 側の stream capture が終了されないまま残り、以後
    // その ストリームへの通常呼び出しが `CUDA_ERROR_STREAM_CAPTURE_*`
    // 系で恒久的に失敗しうる整合性違反になる。design doc §4.3 手順 3 の
    // 拡張）。
    //
    // codex-review P1 再指摘対応（PR #1390 再修正）: 旧稿は
    // `end_capture` 実行後に `std::panic::resume_unwind(payload)` で
    // panic をライブラリ境界外へ再送出していたが、これは
    // `.claude/rules/coding-rust.md`「本番経路で panic しない」規約
    // （AGENTS.md 同旨）に反し、呼び出し元（`fandhe_ai_autodiff::optim::
    // device_store::DeviceParamStore::step`）の `Result` ベースの
    // poison・pending 復元処理を丸ごと迂回してしまう。捕捉した panic は
    // 下記 `body_result` を `Err(BackendError::KernelLaunchFailed(..))`
    // にすることで、直後の「body 成功・グラフ空・end_capture 失敗」判定
    // ロジック（`body_err` 変数。このコメントの下）へそのまま合流させ、
    // panic 発生時も end_capture の失敗を観測したうえで `Result` として
    // 返す。
    let body_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ops.sgd_step_device_tracked(param, grad, velocity.as_deref_mut(), config, token)
    }));

    // capture は body の成否（正常終了・Err・panic のいずれか）に関わらず
    // 必ず終了させる（design doc §4.3 手順 3。driver 状態を capture 中の
    // まま残さない）。
    let end_result = stream.end_capture(instantiate_flags());
    drop(guard);

    let body_result = match body_outcome {
        Ok(r) => r,
        Err(payload) => {
            // panic ペイロードから可読なメッセージを抽出する
            // （`&str`／`String` のいずれかで panic した一般的なケースを
            // カバーする。それ以外の型で panic した場合は詳細不明の
            // 定型文にフォールバックする。`gemm.rs` の parity 回帰
            // ハーネス〈`downcast_ref::<String>()`／`downcast_ref::<&str>()`
            // の同型抽出〉と同じ設計判断だが、本経路は本番経路のため
            // `BackendError` として型付きで返す）。
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "panic payload of unknown type".to_string());
            Err(BackendError::KernelLaunchFailed(format!(
                "run_captured_sgd_step_segment: sgd_step_device_tracked panicked during CUDA \
                 Graph capture (end_capture has already been called to keep driver state \
                 consistent): {msg}"
            )))
        }
    };

    let body_err = body_result.err();

    let graph = match end_result {
        Ok(Some(graph)) => graph,
        Ok(None) => {
            // 空 graph は fail-closed（silent success 禁止。design doc
            // §4.3 手順 3）。body 自体は成功していても、capture 内で
            // 1 個も launch されていないことは契約違反として扱う。
            let msg =
                "run_captured_sgd_step_segment: end_capture produced an empty graph (no kernel \
                        launches were recorded during capture)"
                    .to_string();
            return Err(body_err.unwrap_or(BackendError::Unsupported(msg)));
        }
        Err(e) => {
            let _ = context_cache::observe_driver_result::<()>(ordinal, &_token, Err(e));
            let mapped = crate::memory::map_cuda_error(CudaError::Driver(e));
            return Err(body_err.unwrap_or(mapped));
        }
    };

    if let Some(e) = body_err {
        // capture 自体は正常終了したが body が失敗した区間の graph は
        // 使わずに破棄する（drop 済み `graph` が `Drop` で
        // `cuGraphExecDestroy`／`cuGraphDestroy` を行う）。
        return Err(e);
    }

    // instantiate 直後に 1 回 upload しておく（初回 launch の setup 費を
    // 前倒しする。design doc F5）。**upload 失敗は fail-closed で伝播する**
    // （codex-review P0 指摘: 以前は失敗を poison 化のみで握りつぶし、
    // 直後の `launch` が別途成功すると全体が `Ok(Captured)` 扱いになって
    // いた——「最初に失敗した driver エラーを伝播する」という本クレート
    // 全体の契約〈`context_cache.rs::observe_cuda_result` doc コメント
    // 参照〉に反する後退だった。graph はキャッシュへ入れず、poison 化は
    // 行ったうえでこのエラーをそのまま返す）。
    if let Err(e) = graph.upload() {
        context_cache::observe_cuda_error_ref(ordinal, &_token, &CudaError::Driver(e));
        return Err(crate::memory::map_cuda_error(CudaError::Driver(e)));
    }

    let launch_result = graph.launch();
    if let Err(e) = context_cache::observe_driver_result::<()>(ordinal, &_token, launch_result) {
        // 初回 launch が失敗した graph はキャッシュへ入れない。
        return Err(crate::memory::map_cuda_error(CudaError::Driver(e)));
    }

    GRAPH_LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed);
    CAPTURED_COUNT.fetch_add(1, Ordering::Relaxed);
    put_cached_graph(ordinal, key, graph);
    Ok(SegmentRun::Captured)
}

// ============================================================
// 推論 forward チェーンの capture（イシュー #2115。別機構・opt-in・既定 OFF）
// ============================================================

/// 推論チェーン capture の既定値（**既定 OFF**）。A/B の after 側・将来の
/// 既定反転（別 PR）で反転する単一ゲート。反転すると `device.rs` の
/// created stream 選択にも波及する（全 CUDA 利用者の既定ストリームが
/// created に変わる）ため、反転は GB10 での ADOPT 判定後に限る。
pub(crate) const INFER_GRAPH_DEFAULT_ENABLED: bool = false;

/// 環境変数 `FANDHE_AI_CUDA_GRAPH_INFER` の値と既定値から opt-in を決める
/// 純粋関数（`std::env` を読まないため並列テストで安全）。許容値は
/// `1`／`true` の完全一致のみで、それ以外の設定値は fail-closed で OFF
/// （OWASP A03。値をログ・エラー文へエコーしない）。未設定は `default`。
fn resolve_infer_enabled(env: Option<&str>, default: bool) -> bool {
    match env {
        Some(raw) => matches!(raw, "1" | "true"),
        None => default,
    }
}

static INFER_ENV_ENABLED: OnceLock<bool> = OnceLock::new();

/// override を含まない opt-in 判定（環境変数 > 既定 const。プロセス生存
/// 期間中 1 回だけ読む）。`device.rs::CudaDevice::new` が created stream を
/// 選ぶ判断に使う（stream 種別は ordinal ごとに sticky なため、スレッド
/// ローカルの test override をここへ混ぜない）。
///
/// `internal-diagnostics` feature 下では `device.rs` 側の分岐が cfg で
/// 除外され呼ばれなくなる（`GraphMode::requires_created_stream` と同じ）。
#[cfg_attr(feature = "internal-diagnostics", allow(dead_code))]
pub(crate) fn infer_graph_stream_requested() -> bool {
    *INFER_ENV_ENABLED.get_or_init(|| {
        resolve_infer_enabled(
            std::env::var("FANDHE_AI_CUDA_GRAPH_INFER").ok().as_deref(),
            INFER_GRAPH_DEFAULT_ENABLED,
        )
    })
}

thread_local! {
    /// テスト専用の thread-local override（[`override_infer_graph_for_scope`]）。
    static INFER_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// 推論チェーン capture が有効か（override > 環境変数 > 既定 const）。
/// `CudaBackendOps::linear_chain_forward_captured` の最初の判定。
pub fn infer_graph_enabled() -> bool {
    INFER_OVERRIDE
        .with(|c| c.get())
        .unwrap_or_else(infer_graph_stream_requested)
}

/// [`override_infer_graph_for_scope`] の RAII ガード。drop で直前の値へ
/// 戻す。thread-local を書き換えるため `!Send`
/// （`tensor-core::alloc` の同型ガードと同じ設計）。
#[must_use = "ガードを drop すると override が解除される"]
pub struct InferGraphOverrideGuard {
    prev: Option<bool>,
    _not_send: PhantomData<*const ()>,
}

impl Drop for InferGraphOverrideGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        let _ = INFER_OVERRIDE.try_with(|c| c.set(prev));
    }
}

/// 同一プロセス内で capture あり／なしを比較する実機テスト専用に、
/// 現スレッドの [`infer_graph_enabled`] を上書きする。**stream 種別は
/// 変えない**（created stream 化は環境変数で起動時に決まる）ため、
/// `true` を指定しても legacy stream 上では `Ok(None)`（不適用）になる。
#[doc(hidden)]
pub fn override_infer_graph_for_scope(enabled: bool) -> InferGraphOverrideGuard {
    let prev = INFER_OVERRIDE.with(|c| c.replace(Some(enabled)));
    InferGraphOverrideGuard {
        prev,
        _not_send: PhantomData,
    }
}

static INFER_CAPTURED_COUNT: AtomicU64 = AtomicU64::new(0);
static INFER_REPLAYED_COUNT: AtomicU64 = AtomicU64::new(0);
static INFER_GRAPH_LAUNCH_COUNT: AtomicU64 = AtomicU64::new(0);

/// 推論チェーン capture の診断カウンタ（POD）。学習側の
/// [`StepGraphStats`] とは独立（0.9.0 ピンの bench が読む公開型を変えない
/// ため別型）。値はプロセス起動からの累積で、前後の差分で使う。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferGraphStats {
    /// 新規 capture（キャッシュミス→capture→初回 launch 成功）の回数。
    pub captured: u64,
    /// 既存 graph の再生回数。
    pub replayed: u64,
    /// `CudaGraph::launch()` 成功回数（`captured + replayed` と一致する）。
    pub graph_launches: u64,
}

/// [`InferGraphStats`] の現在値を返す。
pub fn infer_graph_stats() -> InferGraphStats {
    InferGraphStats {
        captured: INFER_CAPTURED_COUNT.load(Ordering::Relaxed),
        replayed: INFER_REPLAYED_COUNT.load(Ordering::Relaxed),
        graph_launches: INFER_GRAPH_LAUNCH_COUNT.load(Ordering::Relaxed),
    }
}

/// thread-local に保持する 1 個の capture 済み推論チェーン。
///
/// **フィールド順が drop 順**: graph（exec）を先に破棄してから、graph が
/// 参照するバッファ（staging・中間出力）を解放する。TLS 破棄時は
/// `CudaBufferHandle::Drop` → `context_cache::begin_buffer_release` が
/// 走るため、そちらは `try_with` 化済み（`context_cache.rs`）。
struct CachedInferChain {
    graph: CudaGraph,
    input_staging: DeviceBuffer<f32>,
    /// 各層の出力。最終要素が最終出力（`download` 対象）。
    acts: Vec<DeviceBuffer<f32>>,
    inserted_seq: u64,
}

thread_local! {
    /// ordinal でパーティションした推論チェーンのキャッシュ（`STEP_GRAPHS`
    /// と同型。別 ordinal の graph を replay・evict しない）。
    static INFER_GRAPHS: RefCell<HashMap<usize, HashMap<SegmentKey, CachedInferChain>>> =
        RefCell::new(HashMap::new());
}

fn take_cached_chain(ordinal: usize, key: &SegmentKey) -> Option<CachedInferChain> {
    INFER_GRAPHS
        .try_with(|cache| {
            cache
                .borrow_mut()
                .get_mut(&ordinal)
                .and_then(|per_ordinal| per_ordinal.remove(key))
        })
        .ok()
        .flatten()
}

/// 世代不一致の掃除・上限超過時の最古 evict をまとめて行い、`entry` を
/// 戻す（[`put_cached_graph`] と同じ方針。同一 ordinal 内で閉じる）。
fn put_cached_chain(ordinal: usize, key: SegmentKey, mut entry: CachedInferChain) {
    let _ = INFER_GRAPHS.try_with(|cache| {
        let mut cache = cache.borrow_mut();
        let cache = cache.entry(ordinal).or_default();
        cache.retain(|k, _| k.generation == key.generation);
        if cache.len() >= MAX_CACHED_GRAPHS_PER_THREAD
            && !cache.contains_key(&key)
            && let Some(oldest_key) = cache
                .iter()
                .min_by_key(|(_, v)| v.inserted_seq)
                .map(|(k, _)| k.clone())
        {
            cache.remove(&oldest_key);
        }
        entry.inserted_seq = next_seq();
        cache.insert(key, entry);
    });
}

/// capture 対象 1 層分の記述（`CudaBackendOps::linear_chain_forward_captured`
/// が検査済みの値を渡す）。
pub(crate) struct ChainLayer<'a> {
    pub(crate) w: DeviceBufferView<'a>,
    pub(crate) bias: Option<DeviceBufferView<'a>>,
    pub(crate) relu: bool,
    pub(crate) k: usize,
    pub(crate) n: usize,
}

/// `buf` の CUDA 実体（`CudaStorage`）を取り出す。
fn storage_ref(buf: &DeviceBuffer<f32>) -> Result<&CudaStorage, BackendError> {
    let handle = buf
        .downcast_handle::<CudaBufferHandle>()
        .ok_or(BackendError::DeviceMismatch)?;
    handle.storage.as_ref().ok_or_else(|| {
        BackendError::DeviceAllocationFailed(
            "linear_chain capture: buffer has numel > 0 but no device allocation".to_string(),
        )
    })
}

/// 層カーネル列を `stream` へ積む（capture 区間の body。`launch_tiled_bias_act_
/// f32_resident` は `linear_forward_device` と同じ融合カーネルで、alloc・
/// upload・download・同期を含まないため capture 可能）。1 層目の入力は
/// `input_staging`、以降は直前層の出力。launch 失敗は `token` で観測
/// （sticky なら ordinal を poison）してから型付きエラーで返す。
fn launch_chain_layers(
    gemm: &crate::gemm::CudaGemm,
    ordinal: usize,
    token: &context_cache::CallToken,
    input_staging: &DeviceBuffer<f32>,
    acts: &mut [DeviceBuffer<f32>],
    layers: &[ChainLayer<'_>],
    m: usize,
) -> Result<(), BackendError> {
    for (i, layer) in layers.iter().enumerate() {
        let (done, rest) = acts.split_at_mut(i);
        let a_buf: &DeviceBuffer<f32> = if i == 0 { input_staging } else { &done[i - 1] };
        let c_buf = rest.first_mut().ok_or_else(|| {
            BackendError::InvalidArgument(
                "linear_chain capture: acts/layers length mismatch".into(),
            )
        })?;
        let a_arg = storage_ref(a_buf)?.as_arg();
        let w_full = storage_ref(layer.w.buffer())?;
        let w_view = w_full.view(layer.w.offset()..layer.w.offset() + layer.w.numel());
        let bias_view = match layer.bias {
            Some(b) => {
                let full = storage_ref(b.buffer())?;
                Some(full.view(b.offset()..b.offset() + b.numel()))
            }
            None => None,
        };
        let c_handle = c_buf
            .downcast_handle_mut::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let c_storage = c_handle.storage.as_mut().ok_or_else(|| {
            BackendError::DeviceAllocationFailed(
                "linear_chain capture: output buffer has numel > 0 but no device allocation"
                    .to_string(),
            )
        })?;
        let mut c_arg = c_storage.as_arg_mut();
        if let Err(e) = gemm.launch_tiled_bias_act_f32_resident(
            &a_arg,
            &w_view,
            bias_view.as_ref(),
            layer.relu,
            &mut c_arg,
            m as u32,
            layer.n as u32,
            layer.k as u32,
        ) {
            context_cache::observe_cuda_error_ref(ordinal, token, &e);
            return Err(crate::memory::map_cuda_error(e));
        }
    }
    Ok(())
}

/// キャッシュミス時に staging・中間バッファを確保し、層カーネル列を
/// capture して graph を作る（`run_captured_sgd_step_segment` の ②と同じ
/// 手順: warmup → `begin_capture_session` → `begin_driver_call` →
/// `begin_capture` → body〈`catch_unwind`〉→ 必ず `end_capture` →
/// `upload`）。**capture 中は alloc／upload／download をしない**
/// （`with_sync_point_call` が拒否するため。確保は全て capture の外）。
/// 呼び出し時点でスレッドは当該 ordinal の `CallToken` を保持していない
/// こと（`begin_capture_session` の in_flight ドレイン契約）。
///
/// 中間出力バッファを replay ごとに再利用できるのは、融合カーネル
/// （`kernels.rs::gemm_tiled_bias_act_f32`）が境界検査付きで `c[..] = v` と
/// **全要素を上書き**し、累積（`+=`）しないため（zeroed 前提に依存しない）。
fn capture_chain(
    ordinal: usize,
    device: &Arc<CudaDevice>,
    mem: &CudaMemory,
    key: &SegmentKey,
    m: usize,
    layers: &[ChainLayer<'_>],
) -> Result<CachedInferChain, BackendError> {
    let stream = device.stream();
    let k0 = layers.first().map(|l| l.k).ok_or_else(|| {
        BackendError::InvalidArgument("linear_chain capture: empty layers".to_string())
    })?;
    let input_staging = mem.alloc_zeroed(&[m, k0])?;
    let mut acts = Vec::with_capacity(layers.len());
    for l in layers {
        acts.push(mem.alloc_zeroed(&[m, l.n])?);
    }

    // NVRTC／モジュールロードを capture の排他区間の外で済ませる
    // （`run_captured_sgd_step_segment` の warmup と同じ理由）。さらに
    // 層カーネル列を capture の**外で 1 回実際に launch** して、CUDA の
    // lazy module loading（`CUDA_MODULE_LOADING=LAZY`）で初回 launch
    // 時に走る関数ロードを capture 区間へ持ち込まない（capture 中の
    // 暗黙ロードによる capture 失敗を避ける。staging はゼロ初期化済みで
    // 出力は捨てる。ストリーム順序により後続の graph launch より前に
    // 完了する）。結果は観測して sticky エラーなら ordinal を poison する。
    let gemm = {
        let warmup_token = context_cache::begin_driver_call(ordinal, &[key.generation])?;
        let gemm = match context_cache::observe_cuda_result(
            ordinal,
            &warmup_token,
            context_cache::cached_gemm(device),
        ) {
            Ok(g) => g,
            Err(e) => {
                drop(warmup_token);
                return Err(crate::memory::map_cuda_error(e));
            }
        };
        let warm = launch_chain_layers(
            &gemm,
            ordinal,
            &warmup_token,
            &input_staging,
            &mut acts,
            layers,
            m,
        );
        drop(warmup_token);
        warm?;
        gemm
    };

    let guard = context_cache::begin_capture_session(ordinal)?;
    let token = match context_cache::begin_driver_call(ordinal, &[key.generation]) {
        Ok(t) => t,
        Err(e) => {
            drop(guard);
            return Err(e);
        }
    };
    let begin_result =
        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    if let Err(e) = context_cache::observe_driver_result(ordinal, &token, begin_result) {
        drop(guard);
        return Err(crate::memory::map_cuda_error(CudaError::Driver(e)));
    }

    // panic しても必ず `end_capture` してから型付きエラーへ変換する
    // （driver 側の capture を残さない。`run_captured_sgd_step_segment` と
    // 同じ契約。本番経路で panic を再送出しない）。
    let body_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        launch_chain_layers(&gemm, ordinal, &token, &input_staging, &mut acts, layers, m)
    }));
    let end_result = stream.end_capture(instantiate_flags());
    drop(guard);

    let body_result = match body_outcome {
        Ok(r) => r,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "panic payload of unknown type".to_string());
            Err(BackendError::KernelLaunchFailed(format!(
                "capture_chain: layer launch panicked during CUDA Graph capture (end_capture \
                 has already been called): {msg}"
            )))
        }
    };
    let body_err = body_result.err();

    let graph = match end_result {
        Ok(Some(graph)) => graph,
        Ok(None) => {
            // 空 graph は fail-closed（silent success 禁止）。
            return Err(body_err.unwrap_or_else(|| {
                // `Unsupported` にしない: チェーン側の `Unsupported` は tape 経路
                // への黙示の全体フォールバック（決定 7）になり、毎回 capture を
                // 試みて最遅経路へ落ちるサイレント劣化を招くため、実失敗として
                // 顕在化させる。
                BackendError::KernelLaunchFailed(
                    "capture_chain: end_capture produced an empty graph (no kernel launches \
                     were recorded during capture)"
                        .to_string(),
                )
            }));
        }
        Err(e) => {
            let _ = context_cache::observe_driver_result::<()>(ordinal, &token, Err(e));
            let mapped = crate::memory::map_cuda_error(CudaError::Driver(e));
            return Err(body_err.unwrap_or(mapped));
        }
    };
    if let Some(e) = body_err {
        return Err(e);
    }
    if let Err(e) = graph.upload() {
        context_cache::observe_cuda_error_ref(ordinal, &token, &CudaError::Driver(e));
        return Err(crate::memory::map_cuda_error(CudaError::Driver(e)));
    }
    drop(token);
    Ok(CachedInferChain {
        graph,
        input_staging,
        acts,
        inserted_seq: 0,
    })
}

/// 共通の実行手順（キャッシュヒット・ミス直後とも）: 入力を staging へ
/// H2D（capture 外の同期点。既存チェーンの `upload` 相当）→ graph launch
/// 1 回 → 最終出力を `download`（最終同期点 1 回）。決定 3（設計文書
/// `docs/inference-chain-single-sync-design.md`）の「同期点は入力 1・
/// 出力 1」を守る。
fn execute_chain(
    ordinal: usize,
    mem: &CudaMemory,
    entry: &mut CachedInferChain,
    generation: u64,
    input: &Tensor<f32>,
    fresh: bool,
) -> Result<Tensor<f32>, BackendError> {
    mem.upload_into(input, &mut entry.input_staging, 0)?;
    {
        let token = context_cache::begin_driver_call(ordinal, &[generation])?;
        let launch_result = entry.graph.launch();
        context_cache::observe_driver_result(ordinal, &token, launch_result)
            .map_err(|e| crate::memory::map_cuda_error(CudaError::Driver(e)))?;
    }
    // 3 カウンタは launch 成功時に揃えて加算する（`InferGraphStats` の
    // `graph_launches == captured + replayed` の不変条件。以後の download
    // 失敗では崩れない）。
    INFER_GRAPH_LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed);
    if fresh {
        INFER_CAPTURED_COUNT.fetch_add(1, Ordering::Relaxed);
    } else {
        INFER_REPLAYED_COUNT.fetch_add(1, Ordering::Relaxed);
    }
    let last = entry.acts.last().ok_or_else(|| {
        BackendError::InvalidArgument("linear_chain capture: cached chain has no output".into())
    })?;
    mem.download(last)
}

/// [`crate::ops::CudaBackendOps::linear_chain_forward_captured`] の実体
/// （イシュー #2115）。`key` に対応する capture 済みチェーンがあれば
/// replay、なければ capture してから実行する。`key` は呼び出し元が同じ
/// 呼び出し内の live 借用から計算した値のため、ヒットした graph が焼き
/// 込んだ weight/bias アドレスは現在生存しているバッファと一致する
/// （#1349 §4.4 の再検証論法）。
///
/// 呼び出しスレッドは当該 ordinal の `CallToken` を保持していないこと。
pub(crate) fn run_captured_linear_chain(
    ordinal: usize,
    device: &Arc<CudaDevice>,
    mem: &CudaMemory,
    key: SegmentKey,
    input: &Tensor<f32>,
    m: usize,
    layers: &[ChainLayer<'_>],
) -> Result<Tensor<f32>, BackendError> {
    let k0 = layers.first().map(|l| l.k).ok_or_else(|| {
        BackendError::InvalidArgument("linear_chain capture: empty layers".to_string())
    })?;
    if input.shape() != [m, k0] {
        return Err(BackendError::InvalidArgument(
            "linear_chain capture: input shape does not match the captured chain".to_string(),
        ));
    }
    let generation = key.generation;
    let (mut entry, fresh) = match take_cached_chain(ordinal, &key) {
        Some(e) => (e, false),
        None => (capture_chain(ordinal, device, mem, &key, m, layers)?, true),
    };
    match execute_chain(ordinal, mem, &mut entry, generation, input, fresh) {
        Ok(t) => {
            put_cached_chain(ordinal, key, entry);
            Ok(t)
        }
        Err(e) => {
            // 既存 graph は破損していないため、失敗しても戻す（一過性の
            // `DeviceContextCaptureInProgress` 等で失わない。step 版と同型）。
            // 新規 capture 直後の失敗は再利用しない。
            if !fresh {
                put_cached_chain(ordinal, key, entry);
            }
            Err(e)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_env_value_accepts_only_allowlisted_values() {
        assert_eq!(parse_env_value("1"), GraphMode::On);
        assert_eq!(parse_env_value("true"), GraphMode::On);
        assert_eq!(parse_env_value("stream-only"), GraphMode::StreamOnly);
        assert_eq!(parse_env_value("0"), GraphMode::Off);
        assert_eq!(parse_env_value("false"), GraphMode::Off);
        assert_eq!(parse_env_value("TRUE"), GraphMode::Off);
        assert_eq!(parse_env_value(""), GraphMode::Off);
        assert_eq!(parse_env_value("; rm -rf /"), GraphMode::Off);
    }

    #[test]
    fn requires_created_stream_is_false_only_for_off() {
        assert!(!GraphMode::Off.requires_created_stream());
        assert!(GraphMode::StreamOnly.requires_created_stream());
        assert!(GraphMode::On.requires_created_stream());
    }

    /// `set_step_graph_enabled`／`step_graph_enabled` の往復を検証する
    /// RAII ガード付きテスト（`crate::precision` の `FlagGuard` と同型。
    /// プロセスグローバルな `STEP_GRAPH_MODE`／`EXPLICIT` を他テストと
    /// 直列化・原状復帰する）。
    struct FlagGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        original_mode: u8,
        original_explicit: bool,
    }

    impl FlagGuard {
        fn acquire() -> Self {
            static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
            let lock = LOCK.lock().unwrap_or_else(|p| p.into_inner());
            Self {
                _lock: lock,
                original_mode: STEP_GRAPH_MODE.load(Ordering::SeqCst),
                original_explicit: EXPLICIT.load(Ordering::SeqCst),
            }
        }
    }

    impl Drop for FlagGuard {
        fn drop(&mut self) {
            STEP_GRAPH_MODE.store(self.original_mode, Ordering::SeqCst);
            EXPLICIT.store(self.original_explicit, Ordering::SeqCst);
        }
    }

    #[test]
    fn set_step_graph_enabled_round_trips_and_overrides_env() {
        let _guard = FlagGuard::acquire();
        set_step_graph_enabled(true);
        assert!(step_graph_enabled());
        assert_eq!(step_graph_mode(), GraphMode::On);
        set_step_graph_enabled(false);
        assert!(!step_graph_enabled());
        assert_eq!(step_graph_mode(), GraphMode::Off);
    }

    /// イシュー #1350: `StepGraphMode`（公開型）が内部 `GraphMode` の
    /// 3 値すべてを取り違えなく写像することを検証する（`bench-fandhe`
    /// の `--graph stream-only` ゲートが `cuda_graph_step_mode() ==
    /// StreamOnly` を突き合わせに使うため、ここでの取り違えは静かに
    /// 誤った行を計測してしまう）。
    #[test]
    fn step_graph_mode_public_maps_all_three_variants() {
        let _guard = FlagGuard::acquire();
        set_step_graph_enabled(true);
        assert_eq!(step_graph_mode_public(), StepGraphMode::On);
        set_step_graph_enabled(false);
        assert_eq!(step_graph_mode_public(), StepGraphMode::Off);
        // `stream-only` は API から設定できない（モジュール冒頭コメント
        // の契約）ため、`From<GraphMode>` 単体の写像を直接検証する。
        assert_eq!(
            StepGraphMode::from(GraphMode::StreamOnly),
            StepGraphMode::StreamOnly
        );
    }

    /// イシュー #1350: `step_graph_stats()` がプロセスワイドカウンタの
    /// 現在値をそのまま反映することを検証する（実機 CUDA 呼び出しを
    /// 経由しないホストモデルテスト。`record_sgd_kernel_launch` と
    /// カウンタの直接操作のみで完結する）。GPU 実機側の
    /// `captured`／`replayed` の実加算は
    /// `tests/graph_capture_real_device.rs`（`#[ignore]`）が検証する。
    #[test]
    fn step_graph_stats_reflects_counters() {
        let _guard = FlagGuard::acquire();
        // 他テストと static カウンタを共有するため、差分（delta）で
        // 検証する（絶対値は cargo test の並列実行順に依存するため
        // 固定できない）。
        let before = step_graph_stats();
        record_sgd_kernel_launch();
        CAPTURED_COUNT.fetch_add(1, Ordering::Relaxed);
        REPLAYED_COUNT.fetch_add(1, Ordering::Relaxed);
        GRAPH_LAUNCH_COUNT.fetch_add(2, Ordering::Relaxed);
        let after = step_graph_stats();
        assert_eq!(after.sgd_kernel_launches - before.sgd_kernel_launches, 1);
        assert_eq!(after.captured - before.captured, 1);
        assert_eq!(after.replayed - before.replayed, 1);
        assert_eq!(after.graph_launches - before.graph_launches, 2);
    }

    /// イシュー #2115: 推論チェーン opt-in の環境変数解釈は `1`／`true` の
    /// 完全一致のみ ON（未設定は既定値・それ以外は fail-closed で OFF）。
    #[test]
    fn resolve_infer_enabled_accepts_only_allowlisted_values() {
        assert!(resolve_infer_enabled(Some("1"), false));
        assert!(resolve_infer_enabled(Some("true"), false));
        for bad in [
            "0",
            "false",
            "TRUE",
            "True",
            "",
            " 1",
            "; rm -rf /",
            "stream-only",
        ] {
            assert!(
                !resolve_infer_enabled(Some(bad), true),
                "{bad:?} は OFF のはず"
            );
        }
        assert!(!resolve_infer_enabled(None, false));
        assert!(resolve_infer_enabled(None, true));
    }

    /// 既定は OFF（受け入れ基準 1。反転は ADOPT 判定後の別 PR）。
    #[test]
    fn infer_graph_default_is_off() {
        const { assert!(!INFER_GRAPH_DEFAULT_ENABLED) };
    }

    /// override は現スレッド限定で、ガード drop で直前の値へ戻る。
    #[test]
    fn override_infer_graph_round_trips_and_nests() {
        let base = infer_graph_enabled();
        {
            let _g1 = override_infer_graph_for_scope(true);
            assert!(infer_graph_enabled());
            {
                let _g2 = override_infer_graph_for_scope(false);
                assert!(!infer_graph_enabled());
            }
            assert!(infer_graph_enabled());
            let other = std::thread::spawn(infer_graph_enabled).join().unwrap();
            assert_eq!(other, infer_graph_stream_requested());
        }
        assert_eq!(infer_graph_enabled(), base);
    }

    /// `infer_graph_stats` がカウンタ現在値を反映する（差分検証）。
    #[test]
    fn infer_graph_stats_reflects_counters() {
        let before = infer_graph_stats();
        INFER_CAPTURED_COUNT.fetch_add(1, Ordering::Relaxed);
        INFER_REPLAYED_COUNT.fetch_add(2, Ordering::Relaxed);
        INFER_GRAPH_LAUNCH_COUNT.fetch_add(3, Ordering::Relaxed);
        let after = infer_graph_stats();
        assert_eq!(after.captured - before.captured, 1);
        assert_eq!(after.replayed - before.replayed, 2);
        assert_eq!(after.graph_launches - before.graph_launches, 3);
    }

    /// TLS 破棄後を模擬: 別スレッドの終了時に `INFER_GRAPHS` を触っても
    /// （`try_with` 化により）abort しない。キャッシュ操作が破棄後に
    /// no-op へ縮退することを確認する。
    #[test]
    fn infer_cache_access_after_tls_destruction_does_not_panic() {
        struct TouchOnDrop;
        impl Drop for TouchOnDrop {
            fn drop(&mut self) {
                let key = SegmentKey {
                    generation: 0,
                    config_key: 0,
                    resources: Vec::new(),
                };
                assert!(take_cached_chain(usize::MAX, &key).is_none());
            }
        }
        thread_local! { static T: TouchOnDrop = const { TouchOnDrop }; }
        std::thread::spawn(|| {
            let _ = INFER_GRAPHS.with(|c| c.borrow().len());
            T.with(|_| {});
        })
        .join()
        .unwrap();
    }
}
