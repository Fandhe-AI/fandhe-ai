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
//! `cached_device` は `ordinal`（`usize`）を、それ以外の `cached_gemm`／
//! `cached_elementwise`／`cached_rmsnorm`／`cached_softmax`／`cached_sgd`／
//! `cached_allocator` は [`ContextKey`]（ordinal + `CudaContext` の同一性。
//! 理由は [`ContextKey`] のドキュメントコメント参照）をキーとする。
//! `ops::CudaBackendOps` は演算メソッドごとに `Self::device_handle()`
//! （`cached_device` 経由）を毎回取得し直し、呼び出し終了時にローカル
//! 変数を drop する（`ops.rs` 参照）。つまり `cached_device`／
//! `cached_gemm` 等の値を呼び出しをまたいで生かし続けているのは
//! **このキャッシュ自身の `Arc` 保持だけ**であり、これは #929 受け入れ
//! 条件 1（2 回目以降の呼び出しが `CudaContext::new`／NVRTC 初期化を
//! 再度支払わない）を成立させるための意図した設計である。
//!
//! ## eternal 組（`cached_device`／`cached_gemm`／`cached_elementwise`／
//! `cached_rmsnorm`／`cached_softmax`／`cached_sgd`）
//!
//! `HashMap<K, Arc<Mutex<Option<Arc<T>>>>>`（外側はエントリ登録専用の
//! 短命ロック、内側はキー単位の single-flight ロック。[`get_or_build`]
//! 参照）を `OnceLock<Mutex<_>>` で保持するプロセスワイド static。エントリは
//! プロセスの生存期間中 evict されない（キーは物理デバイス ordinal
//! （＋実務上ほぼ 1 個に収まる `CudaContext` インスタンス数）で
//! 有界であり、`module_cache.rs::KernelModuleCache`〈shape 特化コンパイル
//! キャッシュ。無限に増えうる key 空間のため LRU 容量上限を持つ〉とは
//! 前提が異なる。本モジュールは常駐させてよい「デバイス数 ×
//! スイート数」個程度の有界エントリのみを扱うため、容量制御・LRU は
//! 不要）。`CudaGemm`／`CudaElementwise`／`CudaRmsNorm`／`CudaSoftmax`／
//! `CudaSgd` の `Arc` はいずれも内部で `Arc<CudaStream>`（延いては
//! `Arc<CudaContext>`）を強参照するため、スイートキャッシュのエントリが
//! 1 つでも生存する限り対応する `CudaContext` は解放されない
//! （`module_cache.rs` の ABA 考察と同型の所有モデル）。この eternal な
//! 保持は「`cached_device` を経由する正準な呼び出し（`ops.rs` が
//! 唯一の呼び出し元）」を前提にしており、その前提の下では有界である。
//!
//! ## `cached_allocator` のみ Weak 参照＋刈り取り
//!
//! [`CudaAllocator`] は上記 eternal 組と異なり、`pub(crate)` な
//! `CudaGemm::new`／`CudaElementwise::new` 等（`gemm.rs`・`elementwise.rs`
//! 参照）の**内部**から `device` を受け取って構築される。これらのコンス
//! トラクタ自体は `context_cache` を経由しない直接呼び出しにも開かれて
//! いるため（`ContextKey` ドキュメントコメントが指す「公開関数の誤用
//! パターン」）、`cached_device` に anchor されない `CudaContext`（＝
//! `CudaDevice::new` を直接呼んで得た使い捨ての context）に対しても
//! `cached_allocator` は呼ばれうる。旧実装は他の eternal 組と同じく
//! `Arc<CudaAllocator>` を永久保持していたため、利用者が
//! `CudaDevice::new`／`CudaGemm::new` を直接繰り返し呼ぶたびに
//! `ContextKey`（ordinal + context ポインタ）が毎回変わり、
//! エントリが際限なく積み上がって当該 `CudaContext`・stream・プール内
//! GPU メモリ（既定最大 128 MiB。`pool.rs::PoolConfig::default()`）が
//! 全ハンドル drop 後も解放されない欠陥があった（codex-review 指摘。
//! イシュー #1020 PR #1061）。
//!
//! このため `cached_allocator` のみ [`get_or_build_weak`] を使い、
//! `HashMap<ContextKey, Arc<Mutex<Option<Weak<CudaAllocator>>>>>` として
//! **`Weak` を保持する**（[`get_or_build`] の `Arc` 保持版とはキャッシュ
//! 本体の型が異なる。[`WeakSingleFlightCache`] 参照）。呼び出し元
//! （`CudaGemm`／`CudaElementwise`／`CudaSoftmax` の `allocator`
//! フィールド〈`CudaRmsNorm`／`CudaSgd` は `allocator` を持たず本キャッシュ
//! を呼ばない。`gemm.rs`／`elementwise.rs`／`softmax.rs` 参照〉、または
//! `ops::CudaBackendOps::
//! release_cached_device_memory`／`device_memory_pool_stats` のローカル
//! 変数）が `Arc<CudaAllocator>` を保持し続ける限りキャッシュヒットし
//! 続ける（正準経路では eternal 組の `CudaGemm` 等が `allocator` を
//! 内部保持し続けるため、実質的に従来同様キャッシュが効き続ける）。
//! 一方、`Arc<CudaAllocator>` を保持する全ハンドルが drop されれば
//! `Weak::upgrade()` が失敗し、次回 `get_or_build_weak` 呼び出し時に
//! 当該エントリを刈り取る（`try_lock` で他スレッドが構築中のエントリを
//! 誤って刈り取らない。[`get_or_build_weak`] 参照）。これにより
//! 「利用者が `CudaAllocator` への全参照を drop すれば対応する
//! `CudaContext`・プールメモリも解放される」ライフサイクル契約になる。
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
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

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
/// `cached_rmsnorm`／`cached_softmax`／`cached_sgd` では [`ContextKey`]
/// （ordinal + context の同一性）を使う（codex-review 指摘。イシュー
/// #1020 PR #1061。理由は [`ContextKey`] のドキュメントコメント参照）。
/// `cached_allocator` のみ本型ではなく [`WeakSingleFlightCache`] を使う
/// （モジュール冒頭「所有モデル・生存期間」参照）。
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

/// [`get_or_build_weak`] 用のキー単位 single-flight スロット。`None` は
/// 未構築、`Some` は「直近に構築したハンドルへの `Weak`」（[`get_or_build`]
/// の `Slot<T>` と異なり `Arc` ではなく `Weak` を保持する。モジュール冒頭
/// 「`cached_allocator` のみ Weak 参照＋刈り取り」参照）。
type WeakSlot<T> = Arc<Mutex<Option<Weak<T>>>>;

/// [`cached_allocator`] 専用のプロセスワイドキャッシュ本体の型エイリアス
/// （`SingleFlightCache<K, T>` の `Weak` 保持版）。他の `cached_*` が使う
/// eternal な [`SingleFlightCache`] とは異なり、値を強参照しない
/// （モジュール冒頭「所有モデル・生存期間」参照）。
type WeakSingleFlightCache<K, T> = Mutex<HashMap<K, WeakSlot<T>>>;

/// `cached_allocator` 専用の「ヒットなら `Weak::upgrade`・ミスまたは
/// 死んだエントリなら `build` で再構築」の single-flight ロジック。
/// [`get_or_build`] と 2 階層ロック構成（外側は登録専用の短命ロック、
/// 内側はキー単位ロックで構築区間を直列化）は同一だが、以下 2 点が
/// 異なる:
///
/// - スロットが保持するのは `Arc<T>` ではなく `Weak<T>`（[`WeakSlot`]）
///   であり、呼び出し元が返り値の `Arc<T>`（および内部で `Arc<T>` を
///   保持する `CudaGemm` 等）をすべて drop すれば `T` はキャッシュとは
///   無関係に破棄される。
/// - 外側ロック取得時に「死んだエントリ」（`Weak::upgrade()` が失敗する
///   エントリ）を刈り取る。刈り取りは `try_lock` のみで行い、他スレッドが
///   `build` 実行中で内側ロックを保持しているエントリ（`try_lock` 失敗）
///   は無条件に残す（構築中のエントリを誤って刈り取ると、その構築の
///   `single-flight` 契約〈`get_or_build_single_flights_concurrent_misses`
///   と同型〉が壊れ、並行呼び出しが `build` を二重実行しうる）。走査対象は
///   このキャッシュが実際に持つエントリ数のみであり、モジュール冒頭
///   「所有モデル・生存期間」の想定どおり「デバイス数 × 直接構築の
///   バイパス呼び出し延べ回数」に閉じる（正準経路ではエントリは 1 個の
///   まま増えない）。
///
/// `None`（未構築 or 構築失敗直後）のエントリは刈り取り対象にしない
/// （[`get_or_build`] の同種コメント参照: 外側ロックと内側ロックの間の
/// 短い区間でスロットが `None` のまま他スレッドから観測されうるため、
/// ここで刈り取ると two-thread が同時に `build` を実行する退行になる）。
fn get_or_build_weak<K, T>(
    cache: &WeakSingleFlightCache<K, T>,
    key: K,
    build: impl FnOnce() -> Result<T, CudaError>,
) -> Result<Arc<T>, CudaError>
where
    K: Eq + std::hash::Hash,
{
    let slot = {
        let mut guard = lock_cache(cache)?;
        guard.retain(|_, slot| match slot.try_lock() {
            Ok(inner) => inner
                .as_ref()
                .map(|weak| weak.strong_count() > 0)
                .unwrap_or(true),
            Err(_) => true,
        });
        Arc::clone(
            guard
                .entry(key)
                .or_insert_with(|| Arc::new(Mutex::new(None))),
        )
    };

    let mut slot_guard = lock_cache(&slot)?;
    if let Some(existing) = slot_guard.as_ref().and_then(Weak::upgrade) {
        return Ok(existing);
    }
    let built = Arc::new(build()?);
    *slot_guard = Some(Arc::downgrade(&built));
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
/// `device_memory_pool_stats` の唯一の呼び出し先であり、`gemm.rs`・
/// `elementwise.rs`・`softmax.rs`（各 `CudaXxx::new` が本関数を呼び、
/// `allocator` フィールドとして保持する。`CudaRmsNorm`／`CudaSgd` は
/// `allocator` を持たず本関数を呼ばない）からも参照される。
/// `CudaAllocator::new` はカーネルコンパイルを伴わず失敗しない
/// （`Result` を返さない）ため、`build` クロージャは常に `Ok` を返す。
///
/// 他の `cached_*` と異なり [`get_or_build_weak`]（`Weak` 保持・死んだ
/// エントリの刈り取りあり）を使う。理由はモジュール冒頭「所有モデル・
/// 生存期間」§「`cached_allocator` のみ Weak 参照＋刈り取り」を参照
/// （codex-review 指摘。イシュー #1020 PR #1061）。正準経路（`ops.rs`
/// 経由）では `CudaGemm` 等の eternal キャッシュが `allocator` を内部
/// 保持し続けるため、キャッシュ効果（2 回目以降が確保をやり直さない）は
/// 従来と変わらない。
pub(crate) fn cached_allocator(device: &CudaDevice) -> Result<Arc<CudaAllocator>, CudaError> {
    static CACHE: OnceLock<WeakSingleFlightCache<ContextKey, CudaAllocator>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build_weak(cache, ContextKey::from_device(device), || {
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

    /// [`get_or_build_weak`] 用の空キャッシュを構築する（`fresh_cache`
    /// の `Weak` 版）。
    fn fresh_weak_cache<T>() -> WeakSingleFlightCache<usize, T> {
        Mutex::new(HashMap::new())
    }

    /// codex-review 指摘の回帰テスト（イシュー #1020 PR #1061。
    /// `context_cache.rs:146` 付近）: [`get_or_build_weak`] はキャッシュ
    /// 自体が値を強参照しないため、呼び出し元が返り値の `Arc` を drop
    /// すれば値は破棄される（＝「利用者が全ハンドルを drop すれば
    /// context・プールも解放される」ライフサイクル契約）。`get_or_build`
    /// （`Arc` 保持版）ではこの `Arc::downgrade` した `Weak` は破棄後に
    /// 必ず `upgrade` が `None` になる。
    #[test]
    fn get_or_build_weak_does_not_keep_value_alive() {
        let cache = fresh_weak_cache::<u32>();

        let built = get_or_build_weak(&cache, 0, || Ok(42)).expect("build succeeds");
        let weak = Arc::downgrade(&built);
        assert!(weak.upgrade().is_some(), "drop 前は生存しているはず");

        drop(built);
        assert!(
            weak.upgrade().is_none(),
            "呼び出し元の Arc を drop すればキャッシュが強参照していない限り値は破棄されるはず"
        );
    }

    /// 上記テストの続き: 値が破棄された後に同じキーで再度呼び出すと、
    /// 死んだエントリを刈り取ったうえで `build` を再実行し、新しい値を
    /// 返す（ABA: 同じキーが新しい値へ差し替わるケースの確認）。
    #[test]
    fn get_or_build_weak_rebuilds_after_value_is_dropped() {
        let cache = fresh_weak_cache::<u32>();
        let calls = std::sync::atomic::AtomicU32::new(0);

        // 注意: `calls` の加算は `get_or_build_weak` が渡された `build`
        // クロージャを実際に呼び出した瞬間のみに置く（クロージャの
        // 生成〈評価〉タイミングで加算すると、2 回目がキャッシュヒットで
        // build 自体は呼ばれない場合でも `calls == 2` になってしまい、
        // 「再構築された」ことの検証にならない）。
        let first = get_or_build_weak(&cache, 0, || {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(1)
        })
        .expect("first build succeeds");
        drop(first);

        let second = get_or_build_weak(&cache, 0, || {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(2)
        })
        .expect("rebuild succeeds");
        assert_eq!(
            *second, 2,
            "値が破棄された後の呼び出しは build を再実行して新しい値を返すはず"
        );
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "1 回目・2 回目それぞれで build が呼ばれるはず（死んだエントリはキャッシュヒットにならない）"
        );
    }

    /// [`get_or_build_weak`] はヒット中（値が生存している間）は
    /// `get_or_build`（`Arc` 保持版）と同じく `build` を再実行しない
    /// （正準経路〈`ops.rs` 経由・`CudaGemm` 等が `allocator` を内部保持
    /// し続ける〉でキャッシュ効果が従来と変わらないことの確認）。
    #[test]
    fn get_or_build_weak_caches_hit_while_value_is_alive() {
        let cache = fresh_weak_cache::<u32>();
        let calls = std::sync::atomic::AtomicU32::new(0);

        let first = get_or_build_weak(&cache, 0, || {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(7)
        })
        .expect("build succeeds");
        let second = get_or_build_weak(&cache, 0, || {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(8)
        })
        .expect("cache hit succeeds without invoking build");

        assert!(
            Arc::ptr_eq(&first, &second),
            "値が生存している間は同一 Arc を共有するはず"
        );
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "値が生存している間は build が 1 回だけ呼ばれるはず"
        );
    }

    /// 死んだエントリの刈り取り（codex-review 指摘。イシュー #1020
    /// PR #1061）: 異なるキーのエントリが死んでも生存中のエントリには
    /// 影響しない。かつ、別キーへの呼び出しをきっかけに死んだエントリが
    /// 外側 `HashMap` から取り除かれる（マップサイズの回帰確認。
    /// [`get_or_build_weak`] 冒頭コメント「外側ロック取得時に...刈り取る」
    /// 参照）。
    #[test]
    fn get_or_build_weak_prunes_dead_entries_on_other_key_access() {
        let cache = fresh_weak_cache::<u32>();

        let dead = get_or_build_weak(&cache, 0, || Ok(100)).expect("key 0 builds");
        drop(dead);
        let alive = get_or_build_weak(&cache, 1, || Ok(200)).expect("key 1 builds");

        {
            let guard = cache.lock().expect("cache lock");
            assert_eq!(
                guard.len(),
                1,
                "key 0 の死んだエントリは key 1 へのアクセスをきっかけに刈り取られるはず"
            );
            assert!(guard.contains_key(&1), "key 1 の生存中エントリは残るはず");
        }
        assert_eq!(*alive, 200);
    }

    /// `Mutex` poison 時は panic せず `CudaError::ContextCacheUnavailable`
    /// を返す（`get_or_build_reports_typed_error_on_poisoned_mutex` の
    /// `Weak` 版）。
    #[test]
    fn get_or_build_weak_reports_typed_error_on_poisoned_mutex() {
        let cache = fresh_weak_cache::<u32>();
        let cache = std::panic::AssertUnwindSafe(&cache);
        let _ = std::panic::catch_unwind(|| {
            let _guard = cache.0.lock().expect("lock before poisoning");
            panic!("intentionally poison the mutex for this test");
        });

        let err = get_or_build_weak(cache.0, 0, || Ok(1)).unwrap_err();
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
