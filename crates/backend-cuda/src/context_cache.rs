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
//! カーネル分行う。`ops.rs` 冒頭コメント参照）を [`ContextKey`]
//! （ordinal + `CudaContext` の同一性）キーで共有する（`cached_allocator`
//! も同様。単純な `ordinal` キーでは異なる `CudaContext` 間で allocator／
//! スイートを誤って共有しうる欠陥があった。理由は [`ContextKey`] の
//! ドキュメントコメント参照。codex-review 指摘。イシュー #1020 PR
//! #1061）。
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
//! 各キャッシュは `cached_device` では `ordinal`（`usize`）、それ以外
//! （`cached_gemm`／`cached_elementwise`／`cached_rmsnorm`／
//! `cached_softmax`／`cached_sgd`／`cached_allocator`）では
//! [`ContextKey`]（ordinal + `CudaContext` の同一性。理由は
//! [`ContextKey`] のドキュメントコメント参照）をキーとする
//! `HashMap<K, Arc<Mutex<Option<Arc<T>>>>>`（外側はエントリ登録専用の
//! 短命ロック、内側はキー単位の single-flight ロック。[`get_or_build`]
//! 参照）を `OnceLock<Mutex<_>>` で保持するプロセスワイド static。エントリは
//! プロセスの生存期間中 evict されない（キーは物理デバイス ordinal
//! （＋実務上ほぼ 1 個に収まる `CudaContext` インスタンス数）で
//! 有界であり、`module_cache.rs::KernelModuleCache`〈shape 特化コンパイル
//! キャッシュ。無限に増えうる key 空間のため LRU 容量上限を持つ〉とは
//! 前提が異なる。本モジュールは常駐させてよい「デバイス数 ×
//! スイート数」個程度の有界エントリのみを扱うため、容量制御・LRU は
//! 不要）。
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
use crate::pool::CudaAllocator;
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
    assert_send_sync::<CudaAllocator>();
};

/// キャッシュキー単位の single-flight ロック。`None` は未構築（または
/// ミスのまま失敗して片付いた状態）、`Some` は構築済みハンドルの共有
/// `Arc`。[`get_or_build`] 参照。
type Slot<T> = Arc<Mutex<Option<Arc<T>>>>;

/// キー `K` をキーとする [`Slot<T>`] のプロセスワイドキャッシュ本体の型
/// エイリアス（clippy `type_complexity` 回避。実体は [`get_or_build`] 冒頭
/// のドキュメント参照）。
///
/// `K` は [`cached_device`] では `usize`（物理デバイス ordinal。この
/// 関数はまだ `CudaContext` が存在しない段階で呼ばれるため ordinal しか
/// キーにできない）、それ以外の `cached_gemm`／`cached_elementwise`／
/// `cached_rmsnorm`／`cached_softmax`／`cached_sgd`／`cached_allocator`
/// では [`ContextKey`]（ordinal + context の同一性）を使う（codex-review
/// 指摘。イシュー #1020 PR #1061。理由は [`ContextKey`] のドキュメント
/// コメント参照）。
type SingleFlightCache<K, T> = Mutex<HashMap<K, Slot<T>>>;

/// `device` を受け取る各 `cached_*` 関数のキャッシュキー。
///
/// `CudaDevice::new`／`CudaGemm::new` 等は `pub` な公開関数であり、
/// `context_cache::cached_device` を経由しない呼び出し元は同一
/// `ordinal` に対して複数の `CudaContext` を生成できてしまう
/// （`ops::CudaBackendOps` は内部で必ず `cached_device` 経由の単一
/// `CudaContext` を使うが、それはこのモジュールの利用者に強制できる
/// 制約ではない）。ordinal のみをキーにすると、後から作られた
/// `CudaContext`／`CudaStream` に基づく演算（カーネル起動）へ、
/// 別の（より先に構築された）`CudaContext` の `CudaAllocator` が
/// 確保した出力バッファを渡してしまい、CUDA context 不一致による
/// 起動失敗・invalid device pointer を招きうる（codex-review 指摘。
/// イシュー #1020 PR #1061 の `context_cache.rs:251` 相当）。
///
/// これを防ぐため、ordinal に加えて `Arc<CudaContext>` の同一性
/// （`Arc::as_ptr` によるポインタ比較。`CudaContext` 自体は `Eq` を
/// 実装しないため、確保元 `Arc` の生存期間中は不変なアドレスを識別子
/// として使う）をキーへ含める。同一 `CudaContext`（＝同一 `Arc` の
/// clone）を共有する呼び出しはヒットし、異なる `CudaContext`（同じ
/// ordinal でも別インスタンス）はミスとして扱われ、それぞれ自分の
/// context に紐づく allocator／スイートを個別に持つ。
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ContextKey {
    ordinal: usize,
    context_ptr: usize,
}

impl ContextKey {
    fn from_device(device: &CudaDevice) -> Self {
        Self {
            ordinal: device.ordinal(),
            context_ptr: Arc::as_ptr(device.context()) as usize,
        }
    }
}

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

/// `K` キーのプロセスワイド `HashMap<K, Arc<Mutex<Option<Arc<T>>>>>`
/// に対する「ヒットなら clone・ミスなら `build` で構築して登録」の
/// single-flight ロジック（codex-review 指摘。イシュー #929 PR #946）。
/// `K` は呼び出し元により `usize`（ordinal。[`cached_device`]）または
/// [`ContextKey`]（ordinal + context の同一性。それ以外の `cached_*`）
/// を渡す（イシュー #1020 PR #1061 で `usize` 固定からジェネリック化）。
///
/// ロックを 2 階層に分ける: 外側の `cache` Mutex は「`key` に対応する
/// キー単位ロック（`Arc<Mutex<Option<Arc<T>>>>`）を取得・登録する」だけの
/// ごく短い臨界区間に限定し、コストの高い `build`（NVRTC コンパイル等を
/// 含みうる）を実行している間は保持しない（他キーへの同時アクセスを
/// 妨げない）。実際の構築は内側のキー単位 `Mutex` を保持したまま行う
/// ため、同一キーへの並行呼び出しは 2 つ目以降がこのキー単位
/// ロックの取得で待機し、`build` を二重実行しない（旧実装は「先にロック外
/// で `build` し、登録は先着 1 件のみ採用・後着は破棄」という楽観的方式
/// だったため、同一キーへの並行初回呼び出しで `CudaDevice::new`／
/// `CudaGemm::new` 等の NVRTC コンパイルが重複実行されうる欠陥があった。
/// 本方式はキー単位ロックで構築区間そのものを直列化することでこれを防ぐ）。
///
/// `build` の失敗（`Err`）はスロットへ格納せず（`None` のまま）、そのまま
/// 呼び出し元へ伝播する（モジュール冒頭「fail-fast 契約」参照）。次回
/// 呼び出しはキー単位ロックを再度取得できるため `build` を再試行できる。
fn get_or_build<K, T>(
    cache: &SingleFlightCache<K, T>,
    key: K,
    build: impl FnOnce() -> Result<T, CudaError>,
) -> Result<Arc<T>, CudaError>
where
    K: Eq + std::hash::Hash,
{
    let slot = {
        let mut guard = lock_cache(cache)?;
        Arc::clone(
            guard
                .entry(key)
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
    static CACHE: OnceLock<SingleFlightCache<usize, CudaDevice>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ordinal, || CudaDevice::new(ordinal))
}

/// `device` の `CudaContext` に対応する [`CudaGemm`] スイートを
/// プロセス内キャッシュから取得する。キーは [`ContextKey::from_device`]
/// （ordinal + context の同一性）であり、同じ ordinal でも別
/// `CudaContext` の `device` を渡せば別スイートを構築する
/// （[`ContextKey`] のドキュメントコメント参照）。
///
/// `ops::CudaBackendOps::gemm`／`gemm_bias_act` の唯一の呼び出し先。
pub(crate) fn cached_gemm(device: &CudaDevice) -> Result<Arc<CudaGemm>, CudaError> {
    static CACHE: OnceLock<SingleFlightCache<ContextKey, CudaGemm>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ContextKey::from_device(device), || {
        CudaGemm::new(device)
    })
}

/// `device` の `CudaContext` に対応する [`CudaElementwise`] スイートを
/// プロセス内キャッシュから取得する（キーは [`ContextKey`]。
/// `cached_gemm` 冒頭コメント参照）。`ops::CudaBackendOps::
/// elementwise_binary`／`elementwise_unary` の唯一の呼び出し先。
pub(crate) fn cached_elementwise(device: &CudaDevice) -> Result<Arc<CudaElementwise>, CudaError> {
    static CACHE: OnceLock<SingleFlightCache<ContextKey, CudaElementwise>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ContextKey::from_device(device), || {
        CudaElementwise::new(device)
    })
}

/// `device` の `CudaContext` に対応する [`CudaRmsNorm`] スイートを
/// プロセス内キャッシュから取得する（キーは [`ContextKey`]。
/// `cached_gemm` 冒頭コメント参照）。`ops::CudaBackendOps::
/// run_fused_rmsnorm` の唯一の呼び出し先。
pub(crate) fn cached_rmsnorm(device: &CudaDevice) -> Result<Arc<CudaRmsNorm>, CudaError> {
    static CACHE: OnceLock<SingleFlightCache<ContextKey, CudaRmsNorm>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ContextKey::from_device(device), || {
        CudaRmsNorm::new(device)
    })
}

/// `device` の `CudaContext` に対応する [`CudaSoftmax`] スイートを
/// プロセス内キャッシュから取得する（キーは [`ContextKey`]。
/// `cached_gemm` 冒頭コメント参照）。`ops::CudaBackendOps::
/// run_fused_softmax` の唯一の呼び出し先。
pub(crate) fn cached_softmax(device: &CudaDevice) -> Result<Arc<CudaSoftmax>, CudaError> {
    static CACHE: OnceLock<SingleFlightCache<ContextKey, CudaSoftmax>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ContextKey::from_device(device), || {
        CudaSoftmax::new(device)
    })
}

/// `device` の `CudaContext` に対応する [`CudaSgd`] スイートを
/// プロセス内キャッシュから取得する（イシュー #935。キーは
/// [`ContextKey`]。`cached_gemm` 冒頭コメント参照）。
/// `ops::CudaBackendOps::sgd_step_device` の唯一の呼び出し先。
/// デバイス常駐パラメータ更新は学習ループの毎ステップ呼ばれるため、
/// NVRTC 再コンパイルを避けるキャッシュの効果が `cached_gemm`／
/// `cached_elementwise` 以上に重要（`docs/
/// device-resident-update-design.md` §3.3d「Cross-tape 契約」: `XMemory` が
/// 持つ stream/context は必ず既存 `context_cache` 経由で取得する）。
pub(crate) fn cached_sgd(device: &CudaDevice) -> Result<Arc<crate::sgd::CudaSgd>, CudaError> {
    static CACHE: OnceLock<SingleFlightCache<ContextKey, crate::sgd::CudaSgd>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ContextKey::from_device(device), || {
        crate::sgd::CudaSgd::new(device)
    })
}

/// `device` の `CudaContext` に対応する [`CudaAllocator`]（出力バッファの
/// サイズクラス別プール。イシュー #1020・REQ-14）をプロセス内キャッシュ
/// から取得する。キーは [`ContextKey`]（ordinal + context の同一性）
/// であり、同じ ordinal でも異なる `CudaContext`（`cached_device` を
/// 経由しない `CudaDevice::new` の直接呼び出し等）には別々の
/// `CudaAllocator`（＝別々の `stream`）を割り当てる。ordinal のみを
/// キーにすると、後から構築された `CudaContext`／`CudaStream` 上の
/// 演算へ、別 `CudaContext` の `CudaAllocator` が確保した出力バッファを
/// 渡してしまい、CUDA context 不一致による起動失敗・invalid device
/// pointer を招く欠陥があった（codex-review 指摘。イシュー #1020
/// PR #1061。[`ContextKey`] のドキュメントコメント参照）。
///
/// `crate::ops::CudaBackendOps::release_cached_device_memory`／
/// `device_memory_pool_stats` の唯一の呼び出し先であり、`gemm.rs`
/// （`CudaGemm::new` が本関数を呼び、`allocator` フィールドとして
/// 保持する）からも参照される。`CudaAllocator::new` はカーネル
/// コンパイルを伴わず失敗しない（`Result` を返さない）ため、
/// `get_or_build` の `build` クロージャは常に `Ok` を返す。
pub(crate) fn cached_allocator(device: &CudaDevice) -> Result<Arc<CudaAllocator>, CudaError> {
    static CACHE: OnceLock<SingleFlightCache<ContextKey, CudaAllocator>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ContextKey::from_device(device), || {
        Ok(CudaAllocator::new(device))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`get_or_build`] は GPU 非依存の純粋なキャッシュロジックのため、
    /// 実 CUDA 型を要求しない汎用 `T`（ここでは `u32`）・`K = usize` の
    /// 単純な整数キーで検証する（`module_cache.rs::LruCache` のテストと
    /// 同じ「GPU 不要ロジックは実カーネル型に依存しない形でテストする」
    /// 方針。実際の `cached_*` 群が使う `usize`／[`ContextKey`] の
    /// どちらでも `get_or_build` 自体の検証の本質は変わらない）。
    fn fresh_cache<T>() -> SingleFlightCache<usize, T> {
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

        let spawn = |cache: std::sync::Arc<SingleFlightCache<usize, u32>>,
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

    /// codex-review 指摘の回帰テスト（イシュー #1020 PR #1061。
    /// `context_cache.rs:251` 相当）: `CudaDevice::new`／`CudaGemm::new`
    /// は `pub` であり、`cached_device` を経由しない呼び出し元は同一
    /// `ordinal` に対して複数の `CudaContext` を生成できる。ordinal のみ
    /// をキーにしていた旧実装では、後から作られた `CudaContext`（デバイス
    /// `B`）の `cached_allocator` 呼び出しが先に作られた `CudaContext`
    /// （デバイス `A`）の `CudaAllocator` をヒットしてしまい、`A` の
    /// `stream`／`ctx` を `B` の演算へ渡す context 不一致を招いていた。
    /// [`ContextKey`] 導入後は同じ ordinal でも異なる `CudaContext` は
    /// 別エントリになる（＝それぞれ自分の `CudaContext` に紐づく
    /// `CudaAllocator` を持つ）ことを実 CUDA デバイスで検証する。
    ///
    /// 実機（CUDA トールキット・GPU）依存のため通常 CI では実行しない
    /// （`.claude/rules/coding-rust.md`「実機依存テストは `#[ignore]` で
    /// 分離」）。
    #[test]
    #[ignore = "実 CUDA デバイスが必要（CI では実行しない）"]
    fn cached_allocator_keeps_distinct_contexts_on_same_ordinal_separate() {
        // `cached_device` を経由せず直接 `CudaDevice::new` を呼ぶことで、
        // 同一 ordinal（0）に対して独立した 2 つの `CudaContext` を作る
        // （公開 API の誤用パターンの再現）。
        let device_a =
            CudaDevice::new(0).expect("CUDA device 0 must be available on the ignored runner");
        let device_b =
            CudaDevice::new(0).expect("CUDA device 0 must be available on the ignored runner");
        assert!(
            !Arc::ptr_eq(device_a.context(), device_b.context()),
            "CudaDevice::new を 2 回直接呼べば別 CudaContext になるはず（前提の確認）"
        );

        let allocator_a =
            cached_allocator(&device_a).expect("cached_allocator must succeed for device_a");
        let allocator_b =
            cached_allocator(&device_b).expect("cached_allocator must succeed for device_b");

        assert!(
            !Arc::ptr_eq(&allocator_a, &allocator_b),
            "異なる CudaContext は別々の CudaAllocator を持つはず（旧実装は ordinal のみキーのため \
             ここで同一 Arc を返し、device_b の演算へ device_a の context に紐づく allocator を \
             渡す不整合を起こしていた）"
        );
        assert!(
            Arc::ptr_eq(allocator_a.context(), device_a.context()),
            "allocator_a の CudaContext は device_a のものと一致するはず"
        );
        assert!(
            Arc::ptr_eq(allocator_b.context(), device_b.context()),
            "allocator_b の CudaContext は device_b のものと一致するはず"
        );

        // 同一 device への再呼び出しはヒットし、同一 Arc を返すはず
        // （ContextKey が ordinal だけでなく context ポインタも一致する
        // 限りキャッシュを再利用する既存の性質を維持していることの確認）。
        let allocator_a_again =
            cached_allocator(&device_a).expect("cached_allocator must succeed for device_a again");
        assert!(
            Arc::ptr_eq(&allocator_a, &allocator_a_again),
            "同一 device への再呼び出しはキャッシュヒットで同一 Arc を返すはず"
        );
    }
}
