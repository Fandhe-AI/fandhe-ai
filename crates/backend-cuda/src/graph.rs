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

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

use cudarc::driver::CudaGraph;
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

use fandhe_ai_tensor_core::buffer::DeviceBuffer;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{
    BackendOps, DispatchFailureCell, SegmentKey, SegmentRun, SgdStepConfig,
};

use crate::context_cache;
use crate::error::CudaError;
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
    static STEP_GRAPHS: RefCell<HashMap<SegmentKey, CachedGraph>> = RefCell::new(HashMap::new());
    static NEXT_SEQ: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn next_seq() -> u64 {
    NEXT_SEQ.with(|c| {
        let v = c.get();
        c.set(v.wrapping_add(1));
        v
    })
}

/// キャッシュから `key` に一致する graph を取り出す（ヒット時は
/// エントリを一旦 map から取り除いた「所有」状態で返す。呼び出し元は
/// launch 後に [`put_cached_graph`] で戻す。`RefCell` の borrow を
/// `body` 実行中に保持しないための take/put 方式。design doc §4.3）。
fn take_cached_graph(key: &SegmentKey) -> Option<CudaGraph> {
    STEP_GRAPHS.with(|cache| cache.borrow_mut().remove(key).map(|c| c.graph))
}

/// capture・launch 済みの graph をキャッシュへ戻す（新規挿入・世代不一致
/// の陳腐化エントリの evict・上限超過時の最古 evict をまとめて行う）。
fn put_cached_graph(key: SegmentKey, graph: CudaGraph) {
    STEP_GRAPHS.with(|cache| {
        let mut cache = cache.borrow_mut();
        // 世代不一致（`invalidate` による回復後の新世代）のエントリは
        // もう再利用されないため、ついでに掃除する（無制限増加の防止。
        // design doc §4.3）。
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
/// 手順（モジュール冒頭コメント・design doc §4.3）:
/// 1. thread-local キャッシュに `key` があれば `begin_driver_call` で
///    poison／世代検査してから `graph.launch()` して
///    [`SegmentRun::Replayed`] を返す。
/// 2. なければ `begin_capture_session` → `begin_driver_call` →
///    `stream.begin_capture` → SGD 更新 1 回 → `stream.end_capture`
///    （成功時は `graph.upload()` を 1 回）→ 初回 `graph.launch()` →
///    キャッシュへ格納して [`SegmentRun::Captured`] を返す。
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
    if let Some(graph) = take_cached_graph(&key) {
        let call_token = context_cache::begin_driver_call(ordinal, &[key.generation])?;
        let launch_result = graph.launch();
        context_cache::observe_driver_result(ordinal, &call_token, launch_result)
            .map_err(|e| crate::memory::map_cuda_error(CudaError::Driver(e)))?;
        put_cached_graph(key, graph);
        return Ok(SegmentRun::Replayed);
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

    // capture 開始前に SGD カーネルを確実にコンパイル・ロード済みにする
    // （Cursor Bugbot 指摘対応。追記）: `sgd_step_device_tracked` は
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
    // そのため `cached_device`／`cached_sgd` を明示的に先行呼び出しし、
    // 2 回目以降はキャッシュヒットで実質無償になる契約（`cached_sgd`
    // doc コメント参照）を利用して capture 領域の外でウォームアップを
    // 完了させる（`_token` 取得済み＝poison／世代検査は通過済みの
    // driver 呼び出し境界内で行う）。
    if let Err(e) = context_cache::cached_device(ordinal)
        .and_then(|device| context_cache::cached_sgd(&device).map(|_| ()))
    {
        drop(guard);
        return Err(crate::memory::map_cuda_error(e));
    }

    let begin_result =
        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    if let Err(e) = context_cache::observe_driver_result(ordinal, &_token, begin_result) {
        drop(guard);
        return Err(crate::memory::map_cuda_error(CudaError::Driver(e)));
    }

    // SGD 更新本体の実行を `catch_unwind` で包み、panic（unwind）した
    // 場合でも直後で必ず `end_capture` を呼んでから panic を再送出する
    // （codex-review P1・Cursor Bugbot 指摘: body が panic すると
    // driver 側の stream capture が終了されないまま残り、以後その
    // ストリームへの通常呼び出しが `CUDA_ERROR_STREAM_CAPTURE_*` 系で
    // 恒久的に失敗しうる整合性違反になる。design doc §4.3 手順 3 の
    // 拡張）。
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
            // 直前で end_capture 済み（driver 側の capture 状態は整合
            // している）。end_capture 自体の成否は body の panic という
            // 一次情報の前では捨て、panic をそのまま呼び出し元へ伝播する。
            std::panic::resume_unwind(payload);
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

    put_cached_graph(key, graph);
    Ok(SegmentRun::Captured)
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
}
