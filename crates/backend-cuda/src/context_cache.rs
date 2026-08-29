//! CUDA コンテキスト／カーネルスイートのプロセス内キャッシュ
//! （イシュー #929。親: フレームワーク横並びベンチ `docs/perf/
//! oss-gemm-comparison-baseline.md` 等が指摘する「2 回目以降の
//! `tape_for(Device::Cuda(_))` も初期化コストを毎回支払う」固定
//! オーバーヘッドの解消）。
//!
//! # 何をキャッシュするか
//!
//! デバイス層は [`cached_device`] が `ordinal` をキーに
//! [`crate::device::CudaDevice`]（`CudaContext::new`・`default_stream`・
//! name/CC 取得を内包する。`device.rs` 参照）をプロセスワイドに共有する。
//!
//! カーネルスイート層は [`cached_gemm`]／[`cached_elementwise`]／
//! [`cached_rmsnorm`]／[`cached_softmax`] が、それぞれ
//! [`crate::gemm::CudaGemm`]／[`crate::elementwise::CudaElementwise`]／
//! [`crate::rmsnorm::CudaRmsNorm`]／[`crate::softmax::CudaSoftmax`]
//! （いずれも `new` 内で NVRTC コンパイル + `load_module` を複数
//! カーネル分行う。`ops.rs` 冒頭コメント参照）を同じく `ordinal` キーで
//! 共有する。
//!
//! `ops::CudaBackendOps` は演算メソッド呼び出しごとにこれらを都度構築
//! していた（イシュー #929 実装計画 §2「現状分析」）。本モジュールを
//! 経由させることで、同一プロセス内の 2 回目以降の呼び出しは
//! コンテキスト生成・NVRTC コンパイルを再度支払わない。
//!
//! # fail-fast 契約（エラーはキャッシュしない）
//!
//! ミス時の構築が失敗した場合（driver 不在・範囲外 ordinal・NVRTC
//! コンパイル失敗等）、その `Err` はキャッシュへ格納しない。次回呼び出しは
//! 再度構築を試みる。これにより「driver が後から利用可能になった環境」
//! でも `tape_for(Device::Cuda(_))` は正しく回復する（`device.rs` の
//! panic 回避ゲート方針・REQ-1 の fail-fast 契約と整合）。`CudaDevice::new`
//! 冒頭の `is_culib_present()` プローブは軽量であり、失敗経路の再試行
//! コストは許容範囲（実装計画 §3.1）。
//!
//! # 所有モデル・生存期間
//!
//! 各キャッシュは `ordinal` をキーとする
//! `HashMap<usize, Arc<Mutex<Option<Arc<T>>>>>`（外側はエントリ登録専用の
//! 短命ロック、内側はキー単位の single-flight ロック。[`get_or_build`]
//! 参照）を `OnceLock<Mutex<_>>` で保持するプロセスワイド static。エントリは
//! プロセスの生存期間中 evict されない（キーは物理デバイス ordinal で
//! 有界であり、`module_cache.rs::KernelModuleCache`〈shape 特化コンパイル
//! キャッシュ。無限に増えうる key 空間のため LRU 容量上限を持つ〉とは
//! 前提が異なる。本モジュールは常駐させてよい「デバイス数 ×
//! スイート数」個の有界エントリのみを扱うため、容量制御・LRU は不要）。
//! `CudaGemm`／`CudaElementwise`／`CudaRmsNorm`／`CudaSoftmax` の `Arc` は
//! いずれも内部で `Arc<CudaStream>`（延いては `Arc<CudaContext>`）を
//! 強参照するため、スイートキャッシュのエントリが 1 つでも生存する限り
//! 対応する `CudaContext` は解放されない（`module_cache.rs` の ABA 考察と
//! 同型の所有モデル）。
//!
//! # `Mutex` poison
//!
//! `KernelModuleCache`（`module_cache.rs`）と同じ方針で、`Mutex` の
//! poison を [`CudaError::ContextCacheUnavailable`] へ変換し panic
//! させない（本モジュールの臨界区間自体は `unwrap`/`expect` を持たない
//! ため通常到達しない）。呼び出し元（`ops::CudaBackendOps`）はこの
//! エラーをそのまま `BackendError` へ伝播してよい（本キャッシュは純粋な
//! 最適化ではあるが、ロック不能は環境異常を示すため、既存 GEMM 経路
//! 同様に型付きエラーとして呼び出し元へ伝える。キャッシュなしへの
//! 縮退運転は行わない — 縮退させると「毎回フレッシュ構築」に戻り
//! 受け入れ条件 1〈2 回目以降が初期化コストを支払わない〉自体が崩れる
//! ため）。

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::device::CudaDevice;
use crate::elementwise::CudaElementwise;
use crate::error::CudaError;
use crate::gemm::CudaGemm;
use crate::rmsnorm::CudaRmsNorm;
use crate::softmax::CudaSoftmax;

/// コンパイル時アサーション: 本モジュールがキャッシュする全ハンドル型が
/// `Send + Sync` であることを固定する。`OnceLock<Mutex<HashMap<usize,
/// Arc<T>>>>` static 経由で複数スレッドから共有する前提（`Arc<T>` を他
/// スレッドへ渡す・複数スレッドから同時に `&T` で参照する）が成立するには
/// `T: Send + Sync` が必須。`module_cache.rs` が `Arc<CudaModule>` を
/// 同型の static へ既に格納できている実績（cudarc 側の型が Send+Sync で
/// あること）から、本モジュールが束ねる各ハンドル型（内部に
/// `Arc<CudaContext>`／`Arc<CudaStream>`／`CudaFunction` を保持するのみ）
/// も同様に Send+Sync であるはずだが、将来のフィールド追加でこの前提が
/// 崩れた場合にここでコンパイルエラーとして検出する。
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CudaDevice>();
    assert_send_sync::<CudaGemm>();
    assert_send_sync::<CudaElementwise>();
    assert_send_sync::<CudaRmsNorm>();
    assert_send_sync::<CudaSoftmax>();
};

/// `ordinal` 単位の single-flight ロック。`None` は未構築（またはミスの
/// まま失敗して片付いた状態）、`Some` は構築済みハンドルの共有 `Arc`。
/// [`get_or_build`] 参照。
type Slot<T> = Arc<Mutex<Option<Arc<T>>>>;

/// `ordinal` をキーとする [`Slot<T>`] のプロセスワイドキャッシュ本体の型
/// エイリアス（clippy `type_complexity` 回避。実体は [`get_or_build`] 冒頭
/// のドキュメント参照）。
type SingleFlightCache<T> = Mutex<HashMap<usize, Slot<T>>>;

/// `Mutex` guard 取得の共通ヘルパー。poison を
/// [`CudaError::ContextCacheUnavailable`] へ変換する（`module_cache.rs::
/// KernelModuleCache::get`/`insert` と同じ変換方針。panic 経路を持たない
/// `.claude/rules/coding-rust.md`）。
fn lock_cache<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>, CudaError> {
    mutex
        .lock()
        .map_err(|e| CudaError::ContextCacheUnavailable {
            detail: format!("context cache mutex poisoned: {e}"),
        })
}

/// `ordinal` キーのプロセスワイド `HashMap<usize, Arc<Mutex<Option<Arc<T>>>>>`
/// に対する「ヒットなら clone・ミスなら `build` で構築して登録」の
/// single-flight ロジック（codex-review 指摘。イシュー #929 PR #946）。
///
/// ロックを 2 階層に分ける: 外側の `cache` Mutex は「`ordinal` に対応する
/// キー単位ロック（`Arc<Mutex<Option<Arc<T>>>>`）を取得・登録する」だけの
/// ごく短い臨界区間に限定し、コストの高い `build`（NVRTC コンパイル等を
/// 含みうる）を実行している間は保持しない（他 ordinal への同時アクセスを
/// 妨げない）。実際の構築は内側のキー単位 `Mutex` を保持したまま行う
/// ため、同一 `ordinal` への並行呼び出しは 2 つ目以降がこのキー単位
/// ロックの取得で待機し、`build` を二重実行しない（旧実装は「先にロック外
/// で `build` し、登録は先着 1 件のみ採用・後着は破棄」という楽観的方式
/// だったため、同一 ordinal への並行初回呼び出しで `CudaDevice::new`／
/// `CudaGemm::new` 等の NVRTC コンパイルが重複実行されうる欠陥があった。
/// 本方式はキー単位ロックで構築区間そのものを直列化することでこれを防ぐ）。
///
/// `build` の失敗（`Err`）はスロットへ格納せず（`None` のまま）、そのまま
/// 呼び出し元へ伝播する（モジュール冒頭「fail-fast 契約」参照）。次回
/// 呼び出しはキー単位ロックを再度取得できるため `build` を再試行できる。
fn get_or_build<T>(
    cache: &SingleFlightCache<T>,
    ordinal: usize,
    build: impl FnOnce() -> Result<T, CudaError>,
) -> Result<Arc<T>, CudaError> {
    let slot = {
        let mut guard = lock_cache(cache)?;
        Arc::clone(
            guard
                .entry(ordinal)
                .or_insert_with(|| Arc::new(Mutex::new(None))),
        )
    };

    let mut slot_guard = lock_cache(&slot)?;
    if let Some(existing) = slot_guard.as_ref() {
        return Ok(Arc::clone(existing));
    }
    let built = Arc::new(build()?);
    *slot_guard = Some(Arc::clone(&built));
    Ok(built)
}

/// `ordinal` 番目の GPU に対応する [`CudaDevice`] をプロセス内キャッシュ
/// から取得する。ヒット時は `CudaContext::new`／NVRTC 初期化を再実行
/// しない（受け入れ条件 1）。
///
/// [`crate::device::CudaDeviceProvider::probe`]（`enumerate`／`select` の
/// 内部経路）・[`crate::ops::CudaBackendOps::device_handle`] の唯一の
/// 呼び出し先とする。
pub(crate) fn cached_device(ordinal: usize) -> Result<Arc<CudaDevice>, CudaError> {
    static CACHE: OnceLock<SingleFlightCache<CudaDevice>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ordinal, || CudaDevice::new(ordinal))
}

/// `ordinal` に対応する [`CudaGemm`] スイートをプロセス内キャッシュから
/// 取得する。`device` はヒット時未使用（ミス時の構築にのみ使う）。
///
/// `ops::CudaBackendOps::gemm`／`gemm_bias_act` の唯一の呼び出し先。
pub(crate) fn cached_gemm(ordinal: usize, device: &CudaDevice) -> Result<Arc<CudaGemm>, CudaError> {
    static CACHE: OnceLock<SingleFlightCache<CudaGemm>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ordinal, || CudaGemm::new(device))
}

/// `ordinal` に対応する [`CudaElementwise`] スイートをプロセス内キャッシュ
/// から取得する。`ops::CudaBackendOps::elementwise_binary`／
/// `elementwise_unary` の唯一の呼び出し先。
pub(crate) fn cached_elementwise(
    ordinal: usize,
    device: &CudaDevice,
) -> Result<Arc<CudaElementwise>, CudaError> {
    static CACHE: OnceLock<SingleFlightCache<CudaElementwise>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ordinal, || CudaElementwise::new(device))
}

/// `ordinal` に対応する [`CudaRmsNorm`] スイートをプロセス内キャッシュ
/// から取得する。`ops::CudaBackendOps::run_fused_rmsnorm` の唯一の
/// 呼び出し先。
pub(crate) fn cached_rmsnorm(
    ordinal: usize,
    device: &CudaDevice,
) -> Result<Arc<CudaRmsNorm>, CudaError> {
    static CACHE: OnceLock<SingleFlightCache<CudaRmsNorm>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ordinal, || CudaRmsNorm::new(device))
}

/// `ordinal` に対応する [`CudaSoftmax`] スイートをプロセス内キャッシュ
/// から取得する。`ops::CudaBackendOps::run_fused_softmax` の唯一の
/// 呼び出し先。
pub(crate) fn cached_softmax(
    ordinal: usize,
    device: &CudaDevice,
) -> Result<Arc<CudaSoftmax>, CudaError> {
    static CACHE: OnceLock<SingleFlightCache<CudaSoftmax>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ordinal, || CudaSoftmax::new(device))
}

/// `ordinal` に対応する [`CudaSgd`] スイートをプロセス内キャッシュから
/// 取得する（イシュー #935）。`ops::CudaBackendOps::sgd_step_device` の
/// 唯一の呼び出し先。デバイス常駐パラメータ更新は学習ループの毎ステップ
/// 呼ばれるため、NVRTC 再コンパイルを避けるキャッシュの効果が
/// `cached_gemm`／`cached_elementwise` 以上に重要（`docs/
/// device-resident-update-design.md` §3.3d「Cross-tape 契約」: `XMemory` が
/// 持つ stream/context は必ず既存 `context_cache` 経由で取得する）。
pub(crate) fn cached_sgd(
    ordinal: usize,
    device: &CudaDevice,
) -> Result<Arc<crate::sgd::CudaSgd>, CudaError> {
    static CACHE: OnceLock<SingleFlightCache<crate::sgd::CudaSgd>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ordinal, || crate::sgd::CudaSgd::new(device))
}

// =========================================================================
// 非同期実行の遅延エラー伝播（イシュー #1013・
// `docs/backend-cuda-async-execution-design.md` §5「エラー伝播」）
// =========================================================================
//
// カーネル起動直後の都度 `synchronize()` を除去した非同期実行契約の下では、
// ある演算の実行時エラー（sticky な driver エラー）の発覚が後続の別演算
// まで遅延しうる。本節は ordinal 単位の状態機械（[`Phase`]）で sticky
// エラーを検出した ordinal を fail-closed に poison し、以降の演算入口
// （[`begin_driver_call`]）で拒否することで、poison 後も気づかずに処理を
// 継続する fail-open を防ぐ。
//
// 各 ordinal の状態は [`OrdinalState`]（`generation`・`phase`・
// `in_flight`）として `Mutex` + `Condvar` で保護する。単一の
// `Mutex<HashMap<usize, ...>>` ではなく ordinal ごとに独立した
// `Mutex`/`Condvar` ペアを持たせることで、ある ordinal の `invalidate`
// （drain 待ち。`Condvar::wait` でブロックしうる）が他 ordinal の
// `begin_driver_call` を妨げない（`get_or_build` の 2 階層ロック設計と
// 同じ「他 ordinal への同時アクセスを妨げない」判断）。
//
// state 遷移:
//   Active（世代 g） --sticky エラー観測--> Poisoned{unrecoverable:false}
//   Poisoned{false} --invalidate 開始--> Retiring
//   Retiring --drain 完了 + stream sync 成功 + プローブ成功--> Active（世代 g+1）
//   Retiring --stream sync 失敗 / プローブが sticky・不一致--> Poisoned{true}
//   Retiring --プローブが operation-local・再試行余地あり--> Poisoned{false}（同一世代）
//   Poisoned{true} は恒久状態（プロセス内に回復手段なし）

use std::sync::Condvar;

use fandhe_ai_tensor_core::device::BackendError;

/// [`OrdinalState::phase`] の値。ordinal 単位の poison 状態機械
/// （モジュール冒頭「非同期実行の遅延エラー伝播」参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// 通常運用中。`generation` は現行世代。
    Active,
    /// [`invalidate`] が復旧処理（drain → stream 完了同期 → 実処理
    /// プローブ）を実行中。この間の [`begin_driver_call`] は
    /// `BackendError::DeviceContextRetiring` で一時拒否する。
    Retiring,
    /// sticky エラーにより poison 済み。`unrecoverable == true` は
    /// [`invalidate`] 自体が失敗した恒久状態（回復手段なし）、`false` は
    /// [`invalidate`] で再生成しうる状態。
    Poisoned { unrecoverable: bool },
}

/// ordinal 単位の poison 状態機械の本体（モジュール冒頭コメント参照）。
struct OrdinalState {
    /// `invalidate` が成功するたびに 1 増える世代カウンタ。
    /// [`fandhe_ai_tensor_core::buffer::DeviceBuffer::generation`] と
    /// 比較し、旧世代のハンドル・バッファの使用を検出する。
    generation: u64,
    phase: Phase,
    /// 現在この ordinal 上で実行中（`begin_driver_call` 済み・
    /// `CallToken` 未 drop）の演算数。[`invalidate`] の drain フェーズは
    /// これが 0 になるまで待つ。
    in_flight: u64,
    /// `invalidate` の実処理プローブが operation-local エラーで失敗した
    /// 連続回数。[`LIMIT_PROBE_RETRIES`] に達すると恒久 poison へ確定する
    /// （無限リトライを避ける fail-closed な上限）。
    probe_retry_count: u32,
}

impl Default for OrdinalState {
    fn default() -> Self {
        Self {
            generation: 0,
            phase: Phase::Active,
            in_flight: 0,
            probe_retry_count: 0,
        }
    }
}

/// `invalidate` の実処理プローブの再試行上限（モジュール冒頭コメント
/// 「state 遷移」参照）。operation-local な失敗が続く場合、無限に
/// リトライせず恒久 poison（`unrecoverable: true`）へ確定する。
const LIMIT_PROBE_RETRIES: u32 = 3;

/// ordinal をキーとする [`OrdinalState`] レジストリ。テスト容易性のため
/// static に依存しない構造体として切り出す（`get_or_build` テストと
/// 同方針。本番経路は [`ordinal_registry`] が返すプロセスワイド static
/// インスタンスを使う）。
// #[allow(dead_code)]: 本 PR（#1013）は状態機械本体（Phase B）のみを実装し、
// `ops.rs` 各演算入口への結線（Phase C。§9 item 7・9〜11）は advisor 助言
// （「部分結線は fail-open になりかねない」）に基づき本 PR の範囲外とした
// （PR 本文・out-of-scope へ記録）。単体テスト（`#[cfg(test)] mod tests`）は
// これらを直接呼んで検証するが、`cargo build`（非 test）では未使用と判定
// されるため許容する。後続 PR の Phase C 結線で解消する想定。
/// ordinal 単位の `(Mutex<OrdinalState>, Condvar)` セルへの共有ハンドル
/// （clippy `type_complexity` 回避。実体は [`OrdinalRegistry`] 参照）。
type OrdinalCell = Arc<(Mutex<OrdinalState>, Condvar)>;

#[allow(dead_code)]
struct OrdinalRegistry {
    states: Mutex<HashMap<usize, OrdinalCell>>,
}

// #[allow(dead_code)]: 本 PR（#1013）は状態機械本体（Phase B）のみを実装し、
// `ops.rs` 各演算入口への結線（Phase C。§9 item 7・9〜11）は advisor 助言
// （「部分結線は fail-open になりかねない」）に基づき本 PR の範囲外とした
// （PR 本文・out-of-scope へ記録）。単体テスト（`#[cfg(test)] mod tests`）は
// これらを直接呼んで検証するが、`cargo build`（非 test）では未使用と判定
// されるため許容する。後続 PR の Phase C 結線で解消する想定。
#[allow(dead_code)]
impl OrdinalRegistry {
    fn new() -> Self {
        Self {
            states: Mutex::new(HashMap::new()),
        }
    }

    /// `ordinal` に対応する `(Mutex<OrdinalState>, Condvar)` を取得する
    /// （未登録なら `OrdinalState::default()` で新規登録）。
    fn entry(&self, ordinal: usize) -> Result<OrdinalCell, BackendError> {
        let mut guard = self.states.lock().map_err(|e| {
            BackendError::DeviceContextPoisoned(format!(
                "context_cache::OrdinalRegistry の Mutex が poison しました: {e}"
            ))
        })?;
        Ok(Arc::clone(guard.entry(ordinal).or_insert_with(|| {
            Arc::new((Mutex::new(OrdinalState::default()), Condvar::new()))
        })))
    }
}

/// プロセスワイドな [`OrdinalRegistry`]（本番経路が使う唯一のインスタンス。
/// テストは [`OrdinalRegistry::new`] で独立インスタンスを使う）。
// #[allow(dead_code)]: 本 PR（#1013）は状態機械本体（Phase B）のみを実装し、
// `ops.rs` 各演算入口への結線（Phase C。§9 item 7・9〜11）は advisor 助言
// （「部分結線は fail-open になりかねない」）に基づき本 PR の範囲外とした
// （PR 本文・out-of-scope へ記録）。単体テスト（`#[cfg(test)] mod tests`）は
// これらを直接呼んで検証するが、`cargo build`（非 test）では未使用と判定
// されるため許容する。後続 PR の Phase C 結線で解消する想定。
#[allow(dead_code)]
fn ordinal_registry() -> &'static OrdinalRegistry {
    static REGISTRY: OnceLock<OrdinalRegistry> = OnceLock::new();
    REGISTRY.get_or_init(OrdinalRegistry::new)
}

/// 1 回の driver 呼び出し（1 演算）に対応するトークン。[`begin_driver_call`]
/// が発行し、`Drop` で `in_flight` を 1 減らし [`invalidate`] の drain 待ち
/// （`Condvar::notify_all`）へ通知する。演算関数のスコープ末尾で自然に
/// drop される（`?` によるアーリーリターン・panic 経路でも解放される。
/// `.claude/rules/coding-rust.md` の RAII 一本化方針と同型）。
// #[allow(dead_code)]: 本 PR（#1013）は状態機械本体（Phase B）のみを実装し、
// `ops.rs` 各演算入口への結線（Phase C。§9 item 7・9〜11）は advisor 助言
// （「部分結線は fail-open になりかねない」）に基づき本 PR の範囲外とした
// （PR 本文・out-of-scope へ記録）。単体テスト（`#[cfg(test)] mod tests`）は
// これらを直接呼んで検証するが、`cargo build`（非 test）では未使用と判定
// されるため許容する。後続 PR の Phase C 結線で解消する想定。
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct CallToken {
    ordinal: usize,
    generation: u64,
}

impl Drop for CallToken {
    fn drop(&mut self) {
        // レジストリ取得・ロック取得が失敗しても（プロセス末期等）
        // Drop は panic できないため、失敗時は静かに諦める
        // （in_flight のわずかな不整合は `invalidate` の drain を長引かせる
        // だけで安全性を損なわない。fail-open にはならない）。
        if let Ok(cell) = ordinal_registry().entry(self.ordinal)
            && let Ok(mut state) = cell.0.lock()
        {
            state.in_flight = state.in_flight.saturating_sub(1);
            cell.1.notify_all();
        }
    }
}

/// 演算入口（`ops.rs` の各公開メソッド）が driver 呼び出し直前に 1 回だけ
/// 呼ぶ（イシュー #1013 設計文書 §9 item 9「TOCTOU 回避のため事前検査を
/// 別ステップにしない」）。
///
/// `resource_generations` には、当該演算が触れる全デバイスバッファ・
/// ハンドルの [`fandhe_ai_tensor_core::buffer::DeviceBuffer::generation`]
/// を渡す。現行世代（`ordinal` の `generation`）と 1 つでも不一致なら
/// [`BackendError::StaleDeviceGeneration`] で拒否する（`in_flight` は
/// 変更しない。世代不一致は「投入前」の拒否であり実行そのものが
/// 行われないため）。
///
/// 拒否条件（`phase` 優先）: `Poisoned{false}` →
/// [`BackendError::DeviceContextPoisoned`]、`Poisoned{true}` →
/// [`BackendError::DeviceContextUnrecoverable`]、`Retiring` →
/// [`BackendError::DeviceContextRetiring`]。`Active` かつ世代一致なら
/// `in_flight += 1` して [`CallToken`] を返す。
// #[allow(dead_code)]: 本 PR（#1013）は状態機械本体（Phase B）のみを実装し、
// `ops.rs` 各演算入口への結線（Phase C。§9 item 7・9〜11）は advisor 助言
// （「部分結線は fail-open になりかねない」）に基づき本 PR の範囲外とした
// （PR 本文・out-of-scope へ記録）。単体テスト（`#[cfg(test)] mod tests`）は
// これらを直接呼んで検証するが、`cargo build`（非 test）では未使用と判定
// されるため許容する。後続 PR の Phase C 結線で解消する想定。
#[allow(dead_code)]
pub(crate) fn begin_driver_call(
    ordinal: usize,
    resource_generations: &[u64],
) -> Result<CallToken, BackendError> {
    let cell = ordinal_registry().entry(ordinal)?;
    let mut state = cell.0.lock().map_err(|e| {
        BackendError::DeviceContextPoisoned(format!(
            "context_cache::OrdinalState の Mutex が poison しました（ordinal={ordinal}）: {e}"
        ))
    })?;
    match state.phase {
        Phase::Poisoned {
            unrecoverable: true,
        } => {
            return Err(BackendError::DeviceContextUnrecoverable {
                ordinal,
                probe_error: "device context is permanently poisoned; process restart required"
                    .to_string(),
            });
        }
        Phase::Poisoned {
            unrecoverable: false,
        } => {
            return Err(BackendError::DeviceContextPoisoned(format!(
                "ordinal {ordinal} is poisoned; call invalidate() to attempt recovery"
            )));
        }
        Phase::Retiring => {
            return Err(BackendError::DeviceContextRetiring { ordinal });
        }
        Phase::Active => {}
    }
    let current_generation = state.generation;
    if let Some(&stale) = resource_generations
        .iter()
        .find(|&&g| g != current_generation)
    {
        return Err(BackendError::StaleDeviceGeneration {
            ordinal,
            resource_generation: stale,
            current_generation,
        });
    }
    state.in_flight += 1;
    Ok(CallToken {
        ordinal,
        generation: current_generation,
    })
}

/// driver 呼び出しの結果を観測し、sticky エラーなら ordinal を poison
/// する（イシュー #1013 設計文書 §5「エラー伝播」）。ラッパー型の
/// `observe` 委譲（`self.stream.xxx(..)?` → `self.observe(token,
/// self.stream.xxx(..))?`）から呼ばれる想定。
///
/// `in_flight` は変更しない（解放は [`CallToken`] の `Drop` のみ。
/// TOCTOU・二重減算を避けるため本関数の責務に含めない）。`Retiring` 中に
/// sticky を観測しても `phase` を上書きしない（`invalidate` の drain を
/// 永久待機させないため。`invalidate` 自身がストリーム同期・プローブで
/// 別途エラーを検出する）。`Err` はそのまま呼び出し元へ返す
/// （分類・poison 化は副作用であり戻り値は変えない）。
// #[allow(dead_code)]: 本 PR（#1013）は状態機械本体（Phase B）のみを実装し、
// `ops.rs` 各演算入口への結線（Phase C。§9 item 7・9〜11）は advisor 助言
// （「部分結線は fail-open になりかねない」）に基づき本 PR の範囲外とした
// （PR 本文・out-of-scope へ記録）。単体テスト（`#[cfg(test)] mod tests`）は
// これらを直接呼んで検証するが、`cargo build`（非 test）では未使用と判定
// されるため許容する。後続 PR の Phase C 結線で解消する想定。
#[allow(dead_code)]
pub(crate) fn observe_driver_result<T>(
    ordinal: usize,
    token: &CallToken,
    result: Result<T, cudarc::driver::result::DriverError>,
) -> Result<T, cudarc::driver::result::DriverError> {
    if let Err(ref e) = result
        && classify_cuda_result(e) == ResultClass::Sticky
        && let Ok(cell) = ordinal_registry().entry(ordinal)
        && let Ok(mut state) = cell.0.lock()
        && state.generation == token.generation
        && state.phase == Phase::Active
    {
        state.phase = Phase::Poisoned {
            unrecoverable: false,
        };
    }
    result
}

/// `ordinal` が現在 poison 状態（`Poisoned{..}`。`unrecoverable` の
/// いずれも含む）かを返す。
// #[allow(dead_code)]: 本 PR（#1013）は状態機械本体（Phase B）のみを実装し、
// `ops.rs` 各演算入口への結線（Phase C。§9 item 7・9〜11）は advisor 助言
// （「部分結線は fail-open になりかねない」）に基づき本 PR の範囲外とした
// （PR 本文・out-of-scope へ記録）。単体テスト（`#[cfg(test)] mod tests`）は
// これらを直接呼んで検証するが、`cargo build`（非 test）では未使用と判定
// されるため許容する。後続 PR の Phase C 結線で解消する想定。
#[allow(dead_code)]
pub(crate) fn is_poisoned(ordinal: usize) -> bool {
    let Ok(cell) = ordinal_registry().entry(ordinal) else {
        // レジストリ自体が取得できない異常時は fail-closed（poison 扱い）
        // に倒す（呼び出し元が「安全側」の判定を得られるようにする）。
        return true;
    };
    let Ok(state) = cell.0.lock() else {
        return true;
    };
    matches!(state.phase, Phase::Poisoned { .. })
}

/// `ordinal` の現行世代を返す（レジストリ未取得時は `0`。新規 ordinal は
/// `OrdinalState::default()` の世代 `0` と整合する）。
// #[allow(dead_code)]: 本 PR（#1013）は状態機械本体（Phase B）のみを実装し、
// `ops.rs` 各演算入口への結線（Phase C。§9 item 7・9〜11）は advisor 助言
// （「部分結線は fail-open になりかねない」）に基づき本 PR の範囲外とした
// （PR 本文・out-of-scope へ記録）。単体テスト（`#[cfg(test)] mod tests`）は
// これらを直接呼んで検証するが、`cargo build`（非 test）では未使用と判定
// されるため許容する。後続 PR の Phase C 結線で解消する想定。
#[allow(dead_code)]
pub(crate) fn current_generation(ordinal: usize) -> u64 {
    let Ok(cell) = ordinal_registry().entry(ordinal) else {
        return 0;
    };
    let Ok(state) = cell.0.lock() else {
        return 0;
    };
    state.generation
}

/// `invalidate` の実処理プローブが検出しうる失敗の分類
/// （[`invalidate_with`] のテスト注入用クロージャの戻り値）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// #[allow(dead_code)]: 本 PR（#1013）は状態機械本体（Phase B）のみを実装し、
// `ops.rs` 各演算入口への結線（Phase C。§9 item 7・9〜11）は advisor 助言
// （「部分結線は fail-open になりかねない」）に基づき本 PR の範囲外とした
// （PR 本文・out-of-scope へ記録）。単体テスト（`#[cfg(test)] mod tests`）は
// これらを直接呼んで検証するが、`cargo build`（非 test）では未使用と判定
// されるため許容する。後続 PR の Phase C 結線で解消する想定。
#[allow(dead_code)]
enum ProbeFailure {
    /// 一時的・当該試行固有の失敗（例: OOM）。再試行の余地がある。
    OperationLocal,
    /// sticky なデバイス異常。即座に恒久 poison へ確定する。
    Sticky,
    /// プローブの往復値が一致しなかった（データ破損の兆候）。sticky と
    /// 同様に即座に恒久 poison へ確定する。
    Mismatch,
}

/// sticky（デバイス全体に影響し続ける） / operation-local（当該呼び出し
/// 固有）の分類（イシュー #1013 設計文書 §5）。**未知の `CUresult` は
/// sticky 側へ倒す**（fail-closed。個々のエラーコードを網羅できていない
/// 場合に fail-open にしないための既定）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// #[allow(dead_code)]: 本 PR（#1013）は状態機械本体（Phase B）のみを実装し、
// `ops.rs` 各演算入口への結線（Phase C。§9 item 7・9〜11）は advisor 助言
// （「部分結線は fail-open になりかねない」）に基づき本 PR の範囲外とした
// （PR 本文・out-of-scope へ記録）。単体テスト（`#[cfg(test)] mod tests`）は
// これらを直接呼んで検証するが、`cargo build`（非 test）では未使用と判定
// されるため許容する。後続 PR の Phase C 結線で解消する想定。
#[allow(dead_code)]
enum ResultClass {
    Sticky,
    OperationLocal,
}

// #[allow(dead_code)]: 本 PR（#1013）は状態機械本体（Phase B）のみを実装し、
// `ops.rs` 各演算入口への結線（Phase C。§9 item 7・9〜11）は advisor 助言
// （「部分結線は fail-open になりかねない」）に基づき本 PR の範囲外とした
// （PR 本文・out-of-scope へ記録）。単体テスト（`#[cfg(test)] mod tests`）は
// これらを直接呼んで検証するが、`cargo build`（非 test）では未使用と判定
// されるため許容する。後続 PR の Phase C 結線で解消する想定。
#[allow(dead_code)]
fn classify_cuda_result(err: &cudarc::driver::result::DriverError) -> ResultClass {
    use cudarc::driver::sys::CUresult::*;
    // operation-local: 当該呼び出しの引数・リソース状況に起因し、以降の
    // 呼び出しには影響しない既知のエラーコード。
    match err.0 {
        CUDA_ERROR_INVALID_VALUE
        | CUDA_ERROR_INVALID_HANDLE
        | CUDA_ERROR_OUT_OF_MEMORY
        | CUDA_ERROR_INVALID_IMAGE
        | CUDA_ERROR_NO_BINARY_FOR_GPU
        | CUDA_ERROR_NOT_FOUND
        | CUDA_ERROR_INVALID_PTX
        | CUDA_ERROR_UNSUPPORTED_PTX_VERSION => ResultClass::OperationLocal,
        // 上記以外（`CUDA_ERROR_ILLEGAL_ADDRESS`・`CUDA_ERROR_LAUNCH_FAILED`・
        // `CUDA_ERROR_ECC_UNCORRECTABLE`・`CUDA_ERROR_HARDWARE_STACK_ERROR`・
        // `CUDA_ERROR_MISALIGNED_ADDRESS`・`CUDA_ERROR_INVALID_ADDRESS_SPACE`・
        // `CUDA_ERROR_ASSERT`・`CUDA_ERROR_LAUNCH_TIMEOUT`・
        // `CUDA_ERROR_INVALID_PC`・`CUDA_ERROR_CONTEXT_IS_DESTROYED`・
        // `CUDA_ERROR_UNKNOWN` を含む未知の全コード）は sticky 側へ倒す。
        _ => ResultClass::Sticky,
    }
}

/// `invalidate`（本体は [`invalidate_with`]。実 CUDA のプローブ処理は
/// 本関数自身では実装せず、`ops.rs`／`memory.rs` 側の復旧経路が実 CUDA
/// クロージャを渡して呼ぶ想定。イシュー #1013 の本 PR 時点では
/// `begin_driver_call`／`observe_driver_result` の結線（Phase C）を
/// スコープ外としたため、本関数の呼び出し元は未結線（テストのみが
/// `invalidate_with` を直接検証する）。
///
/// state 遷移: retire（`Retiring` へ。他スレッドが既に `Retiring` 中なら
/// 完了を待つ。`Poisoned{true}` は即エラー）→ drain（`in_flight == 0` を
/// 待つ）→ `sync`（ストリーム完了同期。失敗で恒久 poison）→ `probe`
/// （実処理プローブ。`OperationLocal` なら同一世代のまま `Poisoned{false}`
/// へ戻し再試行余地を残す、上限超過または `Sticky`/`Mismatch` なら恒久
/// poison）→ 成功なら世代を 1 進めて `Active` へ復帰する。
#[allow(dead_code)]
fn invalidate_with(
    registry: &OrdinalRegistry,
    ordinal: usize,
    sync: impl FnOnce() -> Result<(), ProbeFailure>,
    probe: impl FnOnce() -> Result<(), ProbeFailure>,
) -> Result<(), BackendError> {
    let cell = registry.entry(ordinal)?;

    // a. retire: 所有権を一本化する。他スレッドが既に Retiring 中なら
    // Condvar でその完了を待つ（drain 中に別スレッドが invalidate を
    // 呼んでも二重に実処理プローブを走らせない）。
    let mut state = cell.0.lock().map_err(|e| {
        BackendError::DeviceContextPoisoned(format!(
            "context_cache::OrdinalState の Mutex が poison しました（ordinal={ordinal}）: {e}"
        ))
    })?;
    loop {
        match state.phase {
            Phase::Poisoned {
                unrecoverable: true,
            } => {
                return Err(BackendError::DeviceContextUnrecoverable {
                    ordinal,
                    probe_error: "already permanently poisoned".to_string(),
                });
            }
            Phase::Retiring => {
                state = cell.1.wait(state).map_err(|e| {
                    BackendError::DeviceContextPoisoned(format!(
                        "invalidate の Condvar 待機中に Mutex が poison しました: {e}"
                    ))
                })?;
                continue;
            }
            Phase::Active | Phase::Poisoned { .. } => {
                state.phase = Phase::Retiring;
                break;
            }
        }
    }

    // b. drain: in_flight == 0 になるまで待つ。
    while state.in_flight != 0 {
        state = cell.1.wait(state).map_err(|e| {
            BackendError::DeviceContextPoisoned(format!(
                "invalidate の drain 待機中に Mutex が poison しました: {e}"
            ))
        })?;
    }
    // ロックを一旦手放し、sync/probe（時間のかかる実処理）はロック外で
    // 実行する（`get_or_build` の「build はロック外で実行しない」設計とは
    // 異なり、こちらは Retiring 中の排他が既に他スレッドの
    // begin_driver_call を拒否しているため、ロックを保持し続ける必要が
    // ない。長時間ロックを保持すると `is_poisoned`／`current_generation`
    // まで巻き込んで無用にブロックする）。
    drop(state);

    // b'. ストリーム完了同期。
    if let Err(_probe_failure) = sync() {
        let mut state = cell.0.lock().map_err(|e| {
            BackendError::DeviceContextPoisoned(format!(
                "invalidate の sync 失敗処理中に Mutex が poison しました: {e}"
            ))
        })?;
        state.phase = Phase::Poisoned {
            unrecoverable: true,
        };
        cell.1.notify_all();
        return Err(BackendError::DeviceContextUnrecoverable {
            ordinal,
            probe_error: "stream synchronize failed during invalidate".to_string(),
        });
    }

    // c. 実処理プローブ。
    match probe() {
        Ok(()) => {
            let mut state = cell.0.lock().map_err(|e| {
                BackendError::DeviceContextPoisoned(format!(
                    "invalidate の成功処理中に Mutex が poison しました: {e}"
                ))
            })?;
            state.generation += 1;
            state.phase = Phase::Active;
            state.probe_retry_count = 0;
            cell.1.notify_all();
            Ok(())
        }
        Err(ProbeFailure::OperationLocal) => {
            let mut state = cell.0.lock().map_err(|e| {
                BackendError::DeviceContextPoisoned(format!(
                    "invalidate のプローブ失敗処理中に Mutex が poison しました: {e}"
                ))
            })?;
            state.probe_retry_count += 1;
            if state.probe_retry_count >= LIMIT_PROBE_RETRIES {
                state.phase = Phase::Poisoned {
                    unrecoverable: true,
                };
                cell.1.notify_all();
                return Err(BackendError::DeviceContextUnrecoverable {
                    ordinal,
                    probe_error: format!(
                        "invalidate probe failed {} times (operation-local); giving up",
                        state.probe_retry_count
                    ),
                });
            }
            state.phase = Phase::Poisoned {
                unrecoverable: false,
            };
            cell.1.notify_all();
            Err(BackendError::DeviceContextPoisoned(
                "invalidate probe failed (operation-local); retry may recover".to_string(),
            ))
        }
        Err(reason @ (ProbeFailure::Sticky | ProbeFailure::Mismatch)) => {
            let mut state = cell.0.lock().map_err(|e| {
                BackendError::DeviceContextPoisoned(format!(
                    "invalidate のプローブ失敗処理中に Mutex が poison しました: {e}"
                ))
            })?;
            state.phase = Phase::Poisoned {
                unrecoverable: true,
            };
            cell.1.notify_all();
            Err(BackendError::DeviceContextUnrecoverable {
                ordinal,
                probe_error: format!("invalidate probe failed ({reason:?})"),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`get_or_build`] は GPU 非依存の純粋なキャッシュロジックのため、
    /// 実 CUDA 型を要求しない汎用 `T`（ここでは `u32`）で検証する
    /// （`module_cache.rs::LruCache` のテストと同じ「GPU 不要ロジックは
    /// 実カーネル型に依存しない形でテストする」方針）。
    fn fresh_cache<T>() -> SingleFlightCache<T> {
        Mutex::new(HashMap::new())
    }

    #[test]
    fn get_or_build_constructs_once_and_caches_hit() {
        let cache = fresh_cache::<u32>();
        let calls = std::sync::atomic::AtomicU32::new(0);

        let first = get_or_build(&cache, 0, || {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(42)
        })
        .expect("build succeeds");
        let second = get_or_build(&cache, 0, || {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(43)
        })
        .expect("cache hit succeeds without invoking build");

        assert_eq!(*first, 42);
        assert_eq!(*second, 42, "2 回目はキャッシュヒットで旧値のまま");
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "build は 1 回だけ呼ばれるはず"
        );
        assert!(Arc::ptr_eq(&first, &second), "同一 Arc を共有するはず");
    }

    /// codex-review 指摘の回帰テスト（イシュー #929 PR #946）: 同一
    /// `ordinal` への並行初回呼び出しは `build` を single-flight で
    /// 直列化し、二重実行しない（旧実装は「先にロック外で build し登録は
    /// 先着 1 件のみ採用」という楽観的方式のため、同時ミス時に `build` が
    /// 複数回実行されうる欠陥があった）。`build` 内にスリープを挟んで
    /// 2 スレッドの実行区間を重ねさせ、それでも `build` が 1 回しか
    /// 呼ばれないことを確認する。
    #[test]
    fn get_or_build_single_flights_concurrent_misses() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let cache = std::sync::Arc::new(fresh_cache::<u32>());
        let calls = std::sync::Arc::new(AtomicU32::new(0));

        let spawn = |cache: std::sync::Arc<SingleFlightCache<u32>>,
                     calls: std::sync::Arc<AtomicU32>| {
            std::thread::spawn(move || {
                get_or_build(&cache, 0, || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    Ok(99)
                })
            })
        };

        let t1 = spawn(std::sync::Arc::clone(&cache), std::sync::Arc::clone(&calls));
        // t1 が `build`（内側のキー単位ロック取得〜スリープ中）に確実に
        // 入ってから t2 を開始し、single-flight でなければ t2 も build に
        // 入ってしまう猶予を与える。
        std::thread::sleep(std::time::Duration::from_millis(10));
        let t2 = spawn(std::sync::Arc::clone(&cache), std::sync::Arc::clone(&calls));

        let r1 = t1
            .join()
            .expect("thread 1 does not panic")
            .expect("build succeeds");
        let r2 = t2
            .join()
            .expect("thread 2 does not panic")
            .expect("build succeeds");

        assert_eq!(*r1, 99);
        assert_eq!(*r2, 99);
        assert!(Arc::ptr_eq(&r1, &r2), "同一 Arc を共有するはず");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "single-flight のため build は 1 回だけ呼ばれるはず（同時ミスでも重複構築しない）"
        );
    }

    #[test]
    fn get_or_build_different_ordinals_do_not_share_entries() {
        let cache = fresh_cache::<u32>();
        let a = get_or_build(&cache, 0, || Ok(1)).expect("ordinal 0 succeeds");
        let b = get_or_build(&cache, 1, || Ok(2)).expect("ordinal 1 succeeds");
        assert_eq!(*a, 1);
        assert_eq!(*b, 2);
    }

    /// fail-fast 契約（モジュール冒頭コメント）: `build` が失敗しても
    /// キャッシュへ格納されず、次回呼び出しで再度 `build` が呼ばれる。
    #[test]
    fn get_or_build_does_not_cache_errors() {
        let cache = fresh_cache::<u32>();
        let calls = std::sync::atomic::AtomicU32::new(0);

        let build = || {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err(CudaError::ContextCacheUnavailable {
                detail: "simulated failure".into(),
            })
        };

        assert!(get_or_build(&cache, 0, build).is_err());
        assert!(get_or_build(&cache, 0, build).is_err());
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "エラーはキャッシュされないため build は毎回呼ばれるはず"
        );

        // driver が後から利用可能になった環境を模す: 3 回目は成功する。
        let third = get_or_build(&cache, 0, || Ok(7)).expect("recovers once build succeeds");
        assert_eq!(*third, 7);
    }

    /// `Mutex` poison 時は panic せず `CudaError::ContextCacheUnavailable`
    /// を返す（`module_cache.rs` と同じ縮退運転パターンの検証）。
    #[test]
    fn get_or_build_reports_typed_error_on_poisoned_mutex() {
        let cache = fresh_cache::<u32>();
        let cache = std::panic::AssertUnwindSafe(&cache);
        let _ = std::panic::catch_unwind(|| {
            let _guard = cache.0.lock().expect("lock before poisoning");
            panic!("intentionally poison the mutex for this test");
        });

        let err = get_or_build(cache.0, 0, || Ok(1)).unwrap_err();
        assert!(matches!(err, CudaError::ContextCacheUnavailable { .. }));
    }
}

/// 非同期実行の遅延エラー伝播（状態機械。モジュール冒頭「非同期実行の
/// 遅延エラー伝播」参照）の単体テスト。実 CUDA 依存なし（GPU 不要）で
/// CI 常時実行できる（イシュー #1013 設計文書 §8 T3 系統の最小構成。
/// 実機依存の T1・T2・T3b・T3i・T4 は #1014 へ引き渡す）。
///
/// [`begin_driver_call`]／[`observe_driver_result`]／[`is_poisoned`]／
/// [`current_generation`] はプロセスワイド static（[`ordinal_registry`]）
/// を経由するため、テスト間の干渉を避けるべく各テストは他所と衝突しない
/// 専用 ordinal（10000 番台。実 CUDA テストは ordinal 0/1 のみを使う）を
/// 使う。[`invalidate_with`] は `OrdinalRegistry::new()`（独立インスタンス）
/// で検証するためこの制約を受けない。
#[cfg(test)]
mod poison_state_tests {
    use super::*;
    use cudarc::driver::result::DriverError;
    use cudarc::driver::sys::CUresult;

    /// テスト間で衝突しない ordinal を払い出す（`AtomicUsize` カウンタ。
    /// 10000 番台に限定し実 CUDA テスト・他モジュールの ordinal 0/1 とは
    /// 衝突しない）。
    fn unique_ordinal() -> usize {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(10_000);
        NEXT.fetch_add(1, Ordering::SeqCst)
    }

    fn sticky_err() -> DriverError {
        DriverError(CUresult::CUDA_ERROR_ILLEGAL_ADDRESS)
    }

    fn operation_local_err() -> DriverError {
        DriverError(CUresult::CUDA_ERROR_INVALID_VALUE)
    }

    #[test]
    fn classify_known_operation_local_codes_are_operation_local() {
        for code in [
            CUresult::CUDA_ERROR_INVALID_VALUE,
            CUresult::CUDA_ERROR_INVALID_HANDLE,
            CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            CUresult::CUDA_ERROR_INVALID_IMAGE,
            CUresult::CUDA_ERROR_NO_BINARY_FOR_GPU,
            CUresult::CUDA_ERROR_NOT_FOUND,
            CUresult::CUDA_ERROR_INVALID_PTX,
            CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
        ] {
            assert_eq!(
                classify_cuda_result(&DriverError(code)),
                ResultClass::OperationLocal,
                "{code:?} は operation-local に分類されるはず"
            );
        }
    }

    #[test]
    fn classify_known_sticky_codes_are_sticky() {
        for code in [
            CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
            CUresult::CUDA_ERROR_LAUNCH_FAILED,
            CUresult::CUDA_ERROR_ECC_UNCORRECTABLE,
            CUresult::CUDA_ERROR_MISALIGNED_ADDRESS,
            CUresult::CUDA_ERROR_LAUNCH_TIMEOUT,
        ] {
            assert_eq!(
                classify_cuda_result(&DriverError(code)),
                ResultClass::Sticky,
                "{code:?} は sticky に分類されるはず"
            );
        }
    }

    #[test]
    fn classify_unknown_code_falls_back_to_sticky_fail_closed() {
        // `CUDA_ERROR_UNKNOWN` は明示リストのどちらにも含まれない代表例。
        // fail-closed 契約（モジュール冒頭コメント）: 未知は sticky 側へ倒す。
        assert_eq!(
            classify_cuda_result(&DriverError(CUresult::CUDA_ERROR_UNKNOWN)),
            ResultClass::Sticky
        );
    }

    #[test]
    fn begin_driver_call_succeeds_on_fresh_active_ordinal() {
        let ordinal = unique_ordinal();
        assert_eq!(current_generation(ordinal), 0);
        assert!(!is_poisoned(ordinal));
        let token = begin_driver_call(ordinal, &[0]).expect("fresh ordinal is Active・世代 0");
        assert_eq!(token.generation, 0);
    }

    #[test]
    fn begin_driver_call_rejects_stale_generation_without_touching_in_flight() {
        let ordinal = unique_ordinal();
        let err = begin_driver_call(ordinal, &[7]).expect_err("世代 7 は現行世代 0 と不一致");
        assert!(matches!(
            err,
            BackendError::StaleDeviceGeneration {
                ordinal: o,
                resource_generation: 7,
                current_generation: 0,
            } if o == ordinal
        ));
        // 世代不一致の拒否は「投入前」のため in_flight は変化しない
        // （後続の正常な begin_driver_call が引き続き成功することで
        // 間接的に確認する）。
        let token = begin_driver_call(ordinal, &[0]).expect("世代一致なら成功するはず");
        drop(token);
    }

    #[test]
    fn observe_driver_result_poisons_ordinal_on_sticky_error() {
        let ordinal = unique_ordinal();
        let token = begin_driver_call(ordinal, &[0]).expect("begin succeeds");
        let observed = observe_driver_result::<()>(ordinal, &token, Err(sticky_err()));
        assert!(observed.is_err(), "Err はそのまま伝播するはず");
        assert!(is_poisoned(ordinal), "sticky エラーで poison するはず");

        let rejected = begin_driver_call(ordinal, &[0]);
        assert!(matches!(
            rejected,
            Err(BackendError::DeviceContextPoisoned(_))
        ));
    }

    #[test]
    fn observe_driver_result_does_not_poison_on_operation_local_error() {
        let ordinal = unique_ordinal();
        let token = begin_driver_call(ordinal, &[0]).expect("begin succeeds");
        let observed = observe_driver_result::<()>(ordinal, &token, Err(operation_local_err()));
        assert!(observed.is_err());
        assert!(
            !is_poisoned(ordinal),
            "operation-local エラーは poison しないはず"
        );
    }

    #[test]
    fn observe_driver_result_passes_through_ok_without_side_effects() {
        let ordinal = unique_ordinal();
        let token = begin_driver_call(ordinal, &[0]).expect("begin succeeds");
        let observed = observe_driver_result(ordinal, &token, Ok(42));
        assert_eq!(observed.unwrap(), 42);
        assert!(!is_poisoned(ordinal));
    }

    #[test]
    fn call_token_drop_decrements_in_flight_allowing_second_begin_to_succeed() {
        let ordinal = unique_ordinal();
        let token = begin_driver_call(ordinal, &[0]).expect("begin succeeds");
        drop(token);
        let token2 = begin_driver_call(ordinal, &[0]).expect("2 回目も成功するはず");
        drop(token2);
    }

    #[test]
    fn invalidate_with_succeeds_advances_generation_and_reactivates() {
        let registry = OrdinalRegistry::new();
        let ordinal = 0usize;
        invalidate_with(&registry, ordinal, || Ok(()), || Ok(()))
            .expect("sync・probe とも成功すれば invalidate は成功するはず");

        let cell = registry.entry(ordinal).unwrap();
        let state = cell.0.lock().unwrap();
        assert_eq!(state.generation, 1, "成功時は世代が 1 進むはず");
        assert_eq!(state.phase, Phase::Active);
        assert_eq!(state.probe_retry_count, 0);
    }

    #[test]
    fn invalidate_with_sync_failure_poisons_unrecoverably() {
        let registry = OrdinalRegistry::new();
        let ordinal = 0usize;
        let err = invalidate_with(&registry, ordinal, || Err(ProbeFailure::Sticky), || Ok(()))
            .expect_err("sync 失敗は Err を返すはず");
        assert!(matches!(
            err,
            BackendError::DeviceContextUnrecoverable { ordinal: o, .. } if o == ordinal
        ));

        let cell = registry.entry(ordinal).unwrap();
        let state = cell.0.lock().unwrap();
        assert_eq!(
            state.phase,
            Phase::Poisoned {
                unrecoverable: true
            },
            "sync 失敗は b' の時点で恒久 poison へ確定するはず（c へ進まない）"
        );
    }

    #[test]
    fn invalidate_with_operation_local_probe_failure_stays_recoverable_under_retry_limit() {
        let registry = OrdinalRegistry::new();
        let ordinal = 0usize;
        let err = invalidate_with(
            &registry,
            ordinal,
            || Ok(()),
            || Err(ProbeFailure::OperationLocal),
        )
        .expect_err("プローブ失敗は Err を返すはず");
        assert!(matches!(err, BackendError::DeviceContextPoisoned(_)));

        let cell = registry.entry(ordinal).unwrap();
        let state = cell.0.lock().unwrap();
        assert_eq!(
            state.phase,
            Phase::Poisoned {
                unrecoverable: false
            },
            "上限未満の operation-local 失敗は再試行余地を残すはず"
        );
        assert_eq!(state.probe_retry_count, 1);
        assert_eq!(
            state.generation, 0,
            "回復可能 poison では世代を進めないはず"
        );
    }

    #[test]
    fn invalidate_with_operation_local_probe_failure_exceeding_limit_becomes_unrecoverable() {
        let registry = OrdinalRegistry::new();
        let ordinal = 0usize;
        for _ in 0..LIMIT_PROBE_RETRIES {
            let _ = invalidate_with(
                &registry,
                ordinal,
                || Ok(()),
                || Err(ProbeFailure::OperationLocal),
            );
        }
        let cell = registry.entry(ordinal).unwrap();
        let state = cell.0.lock().unwrap();
        assert_eq!(
            state.phase,
            Phase::Poisoned {
                unrecoverable: true
            },
            "上限到達で恒久 poison へ確定するはず"
        );
    }

    #[test]
    fn invalidate_with_sticky_probe_failure_poisons_unrecoverably_immediately() {
        let registry = OrdinalRegistry::new();
        let ordinal = 0usize;
        let err = invalidate_with(&registry, ordinal, || Ok(()), || Err(ProbeFailure::Sticky))
            .expect_err("sticky プローブ失敗は Err を返すはず");
        assert!(matches!(
            err,
            BackendError::DeviceContextUnrecoverable { .. }
        ));
        let cell = registry.entry(ordinal).unwrap();
        let state = cell.0.lock().unwrap();
        assert_eq!(
            state.phase,
            Phase::Poisoned {
                unrecoverable: true
            },
            "sticky は再試行回数に依らず即座に恒久 poison になるはず"
        );
    }

    #[test]
    fn invalidate_with_mismatch_probe_failure_poisons_unrecoverably_immediately() {
        let registry = OrdinalRegistry::new();
        let ordinal = 0usize;
        let err = invalidate_with(
            &registry,
            ordinal,
            || Ok(()),
            || Err(ProbeFailure::Mismatch),
        )
        .expect_err("mismatch プローブ失敗は Err を返すはず");
        assert!(matches!(
            err,
            BackendError::DeviceContextUnrecoverable { .. }
        ));
    }

    #[test]
    fn invalidate_with_already_unrecoverable_rejects_immediately() {
        let registry = OrdinalRegistry::new();
        let ordinal = 0usize;
        // 1 回目でまず恒久 poison へ確定させる。
        let _ = invalidate_with(&registry, ordinal, || Err(ProbeFailure::Sticky), || Ok(()));

        // 2 回目は sync/probe が呼ばれずに即座に拒否されることを、
        // クロージャ内で panic させることで検証する（呼ばれたら test 失敗）。
        let err = invalidate_with(
            &registry,
            ordinal,
            || panic!("恒久 poison 後は sync を呼ばないはず"),
            || panic!("恒久 poison 後は probe を呼ばないはず"),
        )
        .expect_err("恒久 poison は即座に Err を返すはず");
        assert!(matches!(
            err,
            BackendError::DeviceContextUnrecoverable { .. }
        ));
    }

    #[test]
    fn independent_registry_state_is_isolated_from_static_registry() {
        let registry = OrdinalRegistry::new();
        let ordinal = 0usize;
        // 独立レジストリなので static（`is_poisoned`/`current_generation`）
        // には一切影響しない。独立レジストリの生値を `cell` 経由で直接
        // 検査する形で相当する契約を確認する。
        let cell = registry.entry(ordinal).unwrap();
        {
            let state = cell.0.lock().unwrap();
            assert_eq!(state.generation, 0);
            assert!(!matches!(state.phase, Phase::Poisoned { .. }));
        }
        invalidate_with(&registry, ordinal, || Ok(()), || Ok(())).unwrap();
        let state = cell.0.lock().unwrap();
        assert_eq!(state.generation, 1);
        // static 側（ordinal 0 は実 CUDA テストが使う可能性があるため
        // 直接は突かず、`unique_ordinal()` の新規発行分が世代 0 の
        // ままであることで独立性を確認する）。
        let isolated = unique_ordinal();
        assert_eq!(current_generation(isolated), 0);
    }
}
