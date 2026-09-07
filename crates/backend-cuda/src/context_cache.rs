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
//! GPU メモリ（既定最大 128 MiB。`fandhe_ai_tensor_core::pool_core::SizeClassPoolConfig::default()`）が
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

/// `device` の `CudaContext` に対応する [`crate::mse::CudaMse`] スイートを
/// プロセス内キャッシュから取得する（イシュー #1045。キーは
/// [`ContextKey`]。`cached_gemm` 冒頭コメント参照）。
/// `ops::CudaBackendOps::mse_loss`／`mse_loss_backward` の唯一の呼び出し
/// 先。
pub(crate) fn cached_mse(device: &CudaDevice) -> Result<Arc<crate::mse::CudaMse>, CudaError> {
    static CACHE: OnceLock<SingleFlightCache<ContextKey, crate::mse::CudaMse>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ContextKey::from_device(device), || {
        crate::mse::CudaMse::new(device)
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
    /// 現在 `invalidate_with` の retire ループで `Phase::Retiring` を
    /// 観測し `Condvar::wait` により駐機中（parked）のスレッド数。
    /// `Condvar` 自体は「現在待機中のスレッド数」を問い合わせる API を
    /// 持たないため、`invalidate_with` 内で明示的にカウントする
    /// （`became_waiter` の判定と対で 1 回だけ増減する）。並行 `invalidate`
    /// の統合テスト（`invalidate_with_concurrent_waiter_does_not_become_a_new_owner`）
    /// が「waiter が実際に `Condvar::wait` で駐機した後で初めて owner を
    /// 完了させる」という決定的な同期点を得るために使う（この値が 0 から
    /// 1 になった時点を観測すれば、waiter 自身の `invalidate_with` 呼び
    /// 出しが `Phase::Retiring` を観測し `wait` へ入ったことを保証できる。
    /// テストの外側から `Phase::Retiring` を別途ポーリングするだけでは
    /// 「waiter がまだ呼び出しさえ始めていない」状態と区別できず、
    /// owner を早期に完了させてしまうレースが起きうる）。将来的な
    /// 診断（「現在この ordinal の回復待ちで停止しているスレッド数」）
    /// にも転用できる。
    retiring_waiters: u64,
    /// CUDA Graph stream capture 中の区間（イシュー #1349・`docs/
    /// backend-cuda-graph-step-capture-design.md` §4.2）。`Some` の間は
    /// `begin_capture_session` が同一 ordinal への再入・別スレッドからの
    /// capture 開始を拒否し、`is_capturing_on_current_thread` が
    /// 同期点呼び出し（`memory.rs`／`ops.rs` の各ガード。`gemm.rs::
    /// synchronize()` は診断専用・本番ディスパッチから呼ばれないため
    /// 個別のガードを追加していない。`docs/backend-cuda-graph-step-
    /// capture-design.md` §3.2）の判定に使う。**さらに `begin_driver_call`
    /// 自身も本フィールドを検査し（codex-review P0 指摘対応）、capture
    /// を開始したスレッド以外からの driver 呼び出し全般を一律拒否する**
    /// （`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` は「capture を乱しうる
    /// driver API 呼び出し」の判定をスレッドローカルにする目的のモード
    /// であり、別スレッドが共有ストリームへ直接カーネル起動すること
    /// 自体を driver 側が防いでくれる保証ではないため、ホスト側で
    /// 明示的に排他する。`begin_driver_call` doc コメント参照）。
    capture: Option<std::thread::ThreadId>,
    /// `capture` が `Some` の区間のうち、**in_flight ドレインが完了した
    /// 後**だけ `true` になるサブフラグ（Cursor Bugbot High 指摘対応・
    /// PR #1390 再修正）。[`begin_capture_session`] がドレイン完了と
    /// **同一ロック区間内**で立てる（ロックを一旦手放してから別関数で
    /// 立てる設計は、drain 完了から本フラグ設定までの間に他スレッドの
    /// `begin_buffer_release` が素通りしてしまう競合窓を生むため不採用。
    /// `begin_capture_session` doc コメント「P0 再修正」参照）。
    ///
    /// **この区別が必要な理由（デッドロック回避）**: [`begin_buffer_release`]
    /// （`memory.rs::CudaBufferHandle::Drop` が使う）は「別スレッドが
    /// 実際に driver capture 中」の間だけ駐機し、`capture` が
    /// `Some`（ドレイン待機中を含む）というだけでは駐機しない。もし
    /// ドレイン待機中も駐機してしまうと、あるスレッド B が既に
    /// [`CallToken`] を保持したまま（＝ `in_flight` に計上済みのまま）
    /// 一時バッファを drop し、その `Drop` が `begin_buffer_release` 経由で
    /// 別スレッド A の capture 完了を待つ一方、A は
    /// `begin_capture_session` の drain で B の `in_flight` が 0 になるのを
    /// 待ち続ける——という相互待機（AB-BA デッドロック）が生じる。
    /// ドレイン中（`capturing_active == false`）は driver 側で実際に
    /// capture が始まっていないため、この間に他スレッドの
    /// `cuMemFreeAsync` が共有ストリームへ発行されても捕獲対象の graph を
    /// 汚染しない（捕獲はまだ始まっていない）。よってこの間は
    /// [`begin_buffer_release`] を通し、B の `in_flight` が速やかに
    /// 減ってドレインが完了できるようにする。真に危険な区間
    /// （driver capture が実際に進行中）でのみ [`begin_buffer_release`]
    /// を駐機させれば、`memory.rs` P0 の脆弱性（capture 中の共有
    /// ストリームへの無関係な解放混入）は防げる。
    capturing_active: bool,
}

impl Default for OrdinalState {
    fn default() -> Self {
        Self {
            generation: 0,
            phase: Phase::Active,
            in_flight: 0,
            probe_retry_count: 0,
            retiring_waiters: 0,
            capture: None,
            capturing_active: false,
        }
    }
}

/// CUDA Graph stream capture の区間を表す RAII ガード（イシュー #1349）。
/// `Drop` で `capture` を必ず解放する（`body()` が `?` による早期
/// return・panic のいずれで抜けても、capture フラグが残留して以降の
/// 全 driver 呼び出しが `begin_sync_point_call` に拒否され続ける事態を
/// 防ぐ。`CallToken` と同じ RAII 一本化方針。`.claude/rules/
/// coding-rust.md`）。
#[derive(Debug)]
pub(crate) struct CaptureGuard {
    ordinal: usize,
}

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        // レジストリ取得・ロック取得が失敗しても（プロセス末期等）
        // panic できないため、失敗時は静かに諦める（`CallToken::drop`
        // と同じ方針）。
        if let Ok(cell) = ordinal_registry().entry(self.ordinal)
            && let Ok(mut state) = cell.0.lock()
        {
            state.capture = None;
            // codex-review P0 再指摘対応（PR #1390 再修正）: `capturing_active`
            // も必ず false へ戻す（`begin_capture_session` が drain 完了と
            // 同一ロック区間で立てる。`OrdinalState::capturing_active` doc
            // コメント参照。`stream.begin_capture` 自体が失敗した早期
            // return 経路でも `capturing_active` は既に true になって
            // いるため、ここで確実に戻す必要がある）。
            state.capturing_active = false;
            // `begin_buffer_release`（`memory.rs::CudaBufferHandle::Drop`
            // が使う）が `capturing_active` の解除を待って駐機している
            // 可能性がある（`begin_buffer_release` doc コメント参照）ため、
            // `in_flight` の drain 待ちと同じ `Condvar` で起こす。
            cell.1.notify_all();
        }
    }
}

/// `graph::run_captured_sgd_step_segment` が stream capture 開始直前に 1 回だけ
/// 呼ぶ（イシュー #1349・codex-review P0 指摘対応）。同一 ordinal で既に
/// capture 中（別スレッド・同一スレッドいずれも）なら
/// [`BackendError::InvalidArgument`] で再入を拒否する
/// （`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` は同一 ordinal・同一
/// ストリームへの多重 capture を想定しない）。`Active` 以外の `phase`
/// （`Poisoned`／`Retiring`）でも同様に拒否する（fail-closed。capture
/// 開始を許してしまうと、直後の driver 呼び出しが健全でないコンテキスト
/// 上で行われる）。
///
/// **`in_flight` ドレイン（codex-review P0 指摘対応。`invalidate_with`
/// の retire→drain と同型のパターン）**: 本関数を呼ぶ時点で**呼び出し
/// スレッド自身は当該 ordinal の [`CallToken`] を 1 つも保持していない
/// 契約**とする（`graph::run_captured_sgd_step_segment` は cache miss を確認した
/// 直後・`begin_driver_call` を呼ぶ**前**に本関数を呼ぶよう構成する）。
/// この契約により、本関数がロック中に観測する `state.capture` を
/// `Some(current thread)` へ設定した瞬間から、以後の他スレッドの
/// [`begin_driver_call`] 呼び出しはすべて拒否される（新規呼び出しの
/// 遮断）一方、**このロック取得より前に既に [`begin_driver_call`] を
/// 通過し `in_flight` へ計上済みの呼び出し**（他スレッドが capture 開始
/// より前にカーネル起動を予約済みだが、実際の driver 呼び出しは
/// まだこれから行う場合）は、そのまま共有ストリームへ driver 呼び出しを
/// 発行しうる。これを許すと、capture 開始直後に「capture 対象外の
/// つもりだった」他スレッドのカーネル起動が graph へ意図せず混入する
/// （`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` は「別スレッドから同一
/// ストリームへの発行」自体を必ず防ぐ保証ではない。`begin_driver_call`
/// doc コメント参照）。そのため `state.capture` 設定後、`state.in_flight
/// == 0` になるまで [`Condvar`] で待ってから返す（[`CallToken::drop`]
/// が `notify_all` する。呼び出しスレッド自身は上記契約によりこの時点で
/// 0 件のため、待つのは常に他スレッド分のみで自己デッドロックしない）。
///
/// 返した [`CaptureGuard`] が drop されるまで
/// [`is_capturing_on_current_thread`] はこのスレッドに対して `true` を
/// 返す。
pub(crate) fn begin_capture_session(ordinal: usize) -> Result<CaptureGuard, BackendError> {
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
    if state.capture.is_some() {
        return Err(BackendError::InvalidArgument(format!(
            "begin_capture_session: ordinal {ordinal} is already capturing on this or another \
             thread (CUDA Graph capture does not support re-entrant capture on a single stream)"
        )));
    }
    // 新規呼び出しの遮断はここで確定する（以後の他スレッド
    // `begin_driver_call` は `state.capture.is_some() && owner != self`
    // により拒否される）。
    state.capture = Some(std::thread::current().id());
    // 既存の別スレッド呼び出し（このロック取得より前に `begin_driver_call`
    // を通過済み）の完了を待つ（ドレイン）。
    while state.in_flight != 0 {
        state = cell.1.wait(state).map_err(|e| {
            BackendError::DeviceContextPoisoned(format!(
                "begin_capture_session の in_flight ドレイン待機中に Mutex が poison しました: \
                 {e}"
            ))
        })?;
    }
    // codex-review P0 再指摘対応（PR #1390 再修正。旧稿は
    // `graph::run_captured_sgd_step_segment` が実際の
    // `stream.begin_capture()` 成功後に別関数 `mark_capture_active` を
    // 呼んで立てていたが、drain 完了（ここ）から `stream.begin_capture()`
    // 呼び出しまでの間、ロックを一旦手放す窓が生じ、その間に別スレッドの
    // `begin_buffer_release`（`state.capturing_active` 未設定のため
    // 素通り）が `in_flight` を増減させて素通り、実際の driver capture
    // 開始と競合しうる〈`begin_buffer_release` doc コメント「P0 再修正」
    // の窓と同型〉）: **drain 完了と同一ロック区間内**で `capturing_active`
    // を立てる。`in_flight == 0` をこのロックの下で確認した直後のため、
    // この時点で存命の `BufferReleaseToken`（他スレッド分）は存在せず、
    // 以後の `begin_buffer_release` 呼び出しは（capture 開始スレッド
    // 自身を除き）必ずここで立てた `capturing_active` を見て駐機する
    // （`OrdinalState::capturing_active` doc コメント参照）。
    state.capturing_active = true;
    Ok(CaptureGuard { ordinal })
}

/// `ordinal` が現在このスレッド上で CUDA Graph capture 中かどうかを返す
/// （イシュー #1349）。`memory.rs`／`ops.rs` の同期点ガード
/// （`begin_sync_point_call`）が driver に触れる前の判定に使う（`gemm.rs::
/// synchronize()` は診断専用・本番ディスパッチから呼ばれないため対象外。
/// `docs/backend-cuda-graph-step-capture-design.md` §3.2）。
/// レジストリ自体が取得できない異常時は fail-closed（capture 中扱い＝
/// 同期点呼び出しを拒否する側）に倒す（`is_poisoned` と同じ方針）。
pub(crate) fn is_capturing_on_current_thread(ordinal: usize) -> bool {
    let Ok(cell) = ordinal_registry().entry(ordinal) else {
        return true;
    };
    let Ok(state) = cell.0.lock() else {
        return true;
    };
    state.capture == Some(std::thread::current().id())
}

/// [`begin_buffer_release`] が返す RAII トークン（codex-review P0 再指摘
/// 対応・PR #1390）。`memory.rs::CudaBufferHandle::Drop` はこのトークンを
/// 実際の `cudarc::driver::CudaSlice::drop`（`cuMemFreeAsync`/
/// `cuMemFree` の発行そのもの）が完了するまで保持し続ける契約とする
/// （[`begin_buffer_release`] doc コメント「P0 再修正」参照）。
#[derive(Debug)]
pub(crate) struct BufferReleaseToken {
    ordinal: usize,
    /// [`begin_buffer_release`] がレジストリ・ロック取得に成功し
    /// `state.in_flight` を実際に 1 増やせた場合のみ `true`。取得自体が
    /// 失敗した異常系（fail-open）では `false` のままとし、`Drop` で
    /// 対応する減算を行わない（増減の非対称による `in_flight` 破損を
    /// 防ぐ）。
    registered: bool,
}

impl Drop for BufferReleaseToken {
    fn drop(&mut self) {
        if !self.registered {
            return;
        }
        // レジストリ・ロック取得が失敗しても（プロセス末期等）`Drop` は
        // panic できないため、失敗時は静かに諦める（`CallToken::drop`
        // と同じ方針。稀な異常系でのみ `in_flight` がわずかに残留し
        // `invalidate` の drain を長引かせるだけで安全性は損なわない）。
        if let Ok(cell) = ordinal_registry().entry(self.ordinal)
            && let Ok(mut state) = cell.0.lock()
        {
            state.in_flight = state.in_flight.saturating_sub(1);
            cell.1.notify_all();
        }
    }
}

/// `memory.rs::CudaBufferHandle` の `Drop` 実装（codex-review P1 指摘
/// 対応。PR #1390）が、実際に `cudarc::driver::CudaSlice::drop`
/// （`cuMemFreeAsync`/`cuMemFree` をこのハンドルの `stream` へ直接
/// 発行する。`memory.rs` モジュール冒頭コメント「解放は `CudaSlice` の
/// `Drop` に一本化する」参照）を走らせる**前**に呼び、返した
/// [`BufferReleaseToken`] を実際の解放が終わるまで保持する契約とする。
///
/// **背景**: `CudaSlice::drop` は `begin_driver_call`／
/// `begin_capture_session` の排他機構を一切経由しない（`cudarc` 側の
/// 実装であり本クレートが介入できない）。そのため、あるスレッドが
/// `run_captured_sgd_step_segment`（`graph.rs`）で当該 ordinal を実際に
/// driver capture 中に、**別スレッド**が同じ ordinal 上の
/// `DeviceBuffer`（SGD 更新区間とは無関係な、例えば古い活性化テンソル）
/// を drop すると、その `cuMemFreeAsync` が capture 中の共有ストリームへ
/// 直接記録されてしまい、capture 対象の graph に「本来含めるつもりの
/// ない解放操作」が混入する（再生のたびに既に無効なポインタへ
/// `cuMemFreeAsync` を発行する二重解放相当の破損）。
///
/// **P0 再修正（codex-review 指摘。旧稿は「待ってから戻るだけ」だった）**:
/// 旧稿の `wait_until_not_capturing` は駐機から戻った直後にロックを
/// 手放していたため、戻った瞬間から実際の `cuMemFreeAsync` 発行までの
/// 間に**別の**capture セッションが開始・実際の driver capture まで
/// 進んでしまう競合窓が残っていた。本関数はこの窓を閉じるため、
/// 駐機が終わったら（ロックを保持したまま）[`CallToken`] と同じ
/// `state.in_flight` カウンタへ登録してから返す。以後
/// [`begin_capture_session`] が新たに呼ばれても、そのドレイン
/// （`state.in_flight == 0` 待ち）は本トークンが drop される（＝実際の
/// 解放が完了する）までブロックされるため、「駐機解除後・解放発行前」
/// の間に新しい capture が実際に始まることはない。
///
/// **駐機条件は「実際に driver capture が進行中」（[`OrdinalState::
/// capturing_active`]）のみ（Cursor Bugbot High 指摘対応・PR #1390
/// 再修正）**: `state.capture` が `Some`（`begin_capture_session` の
/// in_flight ドレイン待機中を含む）というだけでは駐機しない。`capture`
/// だけを条件にすると、既に [`CallToken`] を保持したまま（＝
/// `in_flight` に計上済みのまま）一時バッファを drop する別スレッドが
/// 本関数で駐機し、一方 capture 側は当のスレッドの `in_flight` が
/// 減るのを待ち続けるという相互待機（Cursor Bugbot 指摘のデッドロック）
/// が生じる。`capturing_active` のみを条件にすることで、ドレイン中
/// （driver 側はまだ capture していない）の解放は素通しし、実際に
/// capture が進行中の区間だけを対象にする（`capturing_active` doc
/// コメント参照）。
///
/// **capture 中のスレッド自身**からの呼び出しは待たずに即座に登録・
/// 返す（`state.capture == Some(current thread)` の場合。同一スレッドが
/// 自身の capture 中に意図してバッファを解放するケース〈capture 本体が
/// テンポラリを確保・解放する場合等〉を自己デッドロックさせないため。
/// この場合の解放は「このスレッド自身が構成している capture の一部」
/// として cudarc の `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` 契約の範囲内
/// であり、本関数の対象外＝呼び出し元〈`graph.rs`〉の責務とする）。
///
/// `Drop::drop` から呼ばれる想定のため `Result` を返さない。レジストリ・
/// ロック取得の失敗時は他の `Drop` 実装（[`CaptureGuard::drop`]・
/// [`CallToken::drop`]）と同じ方針で静かに諦め、`registered: false` の
/// トークンを返す（fail-open。これ以上安全にできることがない）。
pub(crate) fn begin_buffer_release(ordinal: usize) -> BufferReleaseToken {
    let Ok(cell) = ordinal_registry().entry(ordinal) else {
        return BufferReleaseToken {
            ordinal,
            registered: false,
        };
    };
    let current = std::thread::current().id();
    let Ok(mut state) = cell.0.lock() else {
        return BufferReleaseToken {
            ordinal,
            registered: false,
        };
    };
    while state.capturing_active && state.capture != Some(current) {
        state = match cell.1.wait(state) {
            Ok(s) => s,
            Err(_) => {
                return BufferReleaseToken {
                    ordinal,
                    registered: false,
                };
            }
        };
    }
    state.in_flight += 1;
    BufferReleaseToken {
        ordinal,
        registered: true,
    }
}

/// 同期点（ホスト⇔デバイス転送・確保・解放・明示同期）となる driver
/// 呼び出しの入口が使う（イシュー #1349・`docs/backend-cuda-graph-step-
/// capture-design.md` §4.2）。`begin_driver_call` と同じ排他区間で
/// `ordinal` の capture 状態を検査し、現在のスレッドが capture 中なら
/// **driver に一切触れる前に** [`BackendError::Unsupported`] で拒否する
/// （capture 中の同期呼び出しは `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`
/// 等で driver 自身も拒否するが、それを待たずアプリケーション層で
/// 早期に検出することで、driver 側のエラーを sticky 分類して意図せず
/// ordinal を poison してしまう事態を避ける——同期点呼び出しの誤りは
/// 呼び出し元の契約違反であり、デバイスコンテキスト自体は健全な
/// ままにしてよいため）。`what` は診断メッセージ用の呼び出し名
/// （`"upload"`／`"download"`／`"alloc_zeroed"`／`"synchronize"` 等）。
///
/// capture 中でなければ [`begin_driver_call`] へそのまま委譲する。
pub(crate) fn begin_sync_point_call(
    ordinal: usize,
    resource_generations: &[u64],
    what: &'static str,
) -> Result<CallToken, BackendError> {
    if is_capturing_on_current_thread(ordinal) {
        return Err(BackendError::Unsupported(format!(
            "cuda graph capture: {what} is a host synchronization point and cannot be captured"
        )));
    }
    begin_driver_call(ordinal, resource_generations)
}

/// `invalidate` の実処理プローブの再試行上限（モジュール冒頭コメント
/// 「state 遷移」参照）。operation-local な失敗が続く場合、無限に
/// リトライせず恒久 poison（`unrecoverable: true`）へ確定する。
const LIMIT_PROBE_RETRIES: u32 = 3;

/// ordinal をキーとする [`OrdinalState`] レジストリ。テスト容易性のため
/// static に依存しない構造体として切り出す（`get_or_build` テストと
/// 同方針。本番経路は [`ordinal_registry`] が返すプロセスワイド static
/// インスタンスを使う）。
/// ordinal 単位の `(Mutex<OrdinalState>, Condvar)` セルへの共有ハンドル
/// （clippy `type_complexity` 回避。実体は [`OrdinalRegistry`] 参照）。
type OrdinalCell = Arc<(Mutex<OrdinalState>, Condvar)>;

// 本 PR（#1064）で `ops.rs`／`memory.rs` の BackendOps／MemoryOps 実装
// 入口（`with_driver_call` ヘルパー経由）から `begin_driver_call` を
// 呼ぶよう結線した（Phase C。旧コメント「Phase C は #1062 へ引き継ぐ」は
// 解消済み）ため、以降の `OrdinalRegistry`・`ordinal_registry`・
// `CallToken`・`begin_driver_call`・`observe_driver_result`・
// `current_generation`・`ResultClass`・`classify_cuda_result` は本番経路
// から実際に呼ばれる（`#[allow(dead_code)]` 不要）。
struct OrdinalRegistry {
    states: Mutex<HashMap<usize, OrdinalCell>>,
}

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
fn ordinal_registry() -> &'static OrdinalRegistry {
    static REGISTRY: OnceLock<OrdinalRegistry> = OnceLock::new();
    REGISTRY.get_or_init(OrdinalRegistry::new)
}

/// 1 回の driver 呼び出し（1 演算）に対応するトークン。[`begin_driver_call`]
/// が発行し、`Drop` で `in_flight` を 1 減らし [`invalidate`] の drain 待ち
/// （`Condvar::notify_all`）へ通知する。演算関数のスコープ末尾で自然に
/// drop される（`?` によるアーリーリターン・panic 経路でも解放される。
/// `.claude/rules/coding-rust.md` の RAII 一本化方針と同型）。
#[derive(Debug)]
pub(crate) struct CallToken {
    ordinal: usize,
    generation: u64,
}

impl CallToken {
    /// `begin_driver_call` がこのトークンを発行した時点の世代
    /// （イシュー #1349 `ops.rs::CudaBackendOps::captured_segment_key`／
    /// `run_captured_sgd_step_segment` が `SegmentKey::generation` の
    /// 導出に使う。`current_generation` を別途再ロックする代わりに
    /// トークン発行時点の値をそのまま使うことで、`begin_driver_call` の
    /// poison／世代検査と `SegmentKey` の世代が常に同一ロック区間の値に
    /// 揃う）。
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }
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
///
/// **CUDA Graph capture 中の排他制御（イシュー #1349・codex-review P0
/// 指摘・`docs/backend-cuda-graph-step-capture-design.md` §4.2）**:
/// `state.capture`（capture 開始スレッドの [`std::thread::ThreadId`]）が
/// `Some` の間、**capture を開始したスレッド以外**からの本関数呼び出しは
/// [`BackendError::Unsupported`] で拒否する（driver に一切触れる前。
/// `begin_sync_point_call` と同じ「早期拒否」方針）。`CU_STREAM_
/// CAPTURE_MODE_THREAD_LOCAL` は「capture を乱しうる driver API 呼び
/// 出し」を capture 開始スレッドに限定する目的のモードであり、
/// **別スレッドが同じ共有ストリームへ直接カーネル起動すること自体を
/// driver 側が防いでくれる保証ではない**（起動する演算・driver バージョン
/// 依存の未定義動作になりうる）。そのため本関数がホスト側で明示的に
/// 排他する: 別スレッドの呼び出しは capture 完了（[`CaptureGuard`] の
/// drop）まで一律拒否し、SGD 更新等が capture 中の graph へ意図せず
/// 混入する事態を防ぐ。**capture を開始したスレッド自身**の呼び出しは
/// 通す（`graph::run_captured_sgd_step_segment` の `body()` 実行自体が同一
/// スレッドから `begin_driver_call` 経由で実カーネルを起動する契約の
/// ため。再入拒否ではなく同一スレッドの正常フロー）。
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
    if let Some(owner) = state.capture
        && owner != std::thread::current().id()
    {
        return Err(BackendError::Unsupported(format!(
            "cuda graph capture: ordinal {ordinal} is currently being captured by another              thread; concurrent driver calls on the shared stream during capture are not              supported"
        )));
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

/// [`observe_driver_result`] の `CudaError` 版（イシュー #1013 設計文書
/// §9 item 7・本 PR #1064 の Phase C 結線）。
///
/// `ops.rs`／`memory.rs` の演算入口は、内部の `gemm.rs`／`elementwise.rs`／
/// `softmax.rs`／`rmsnorm.rs`／`sgd.rs` の起動 API から
/// `Result<T, CudaError>` を受け取る（これらは `?` により
/// `cudarc::driver::result::DriverError` を必ず [`CudaError::Driver`] へ
/// 変換する契約。`error.rs` の `impl From<DriverError> for CudaError`
/// 参照）。本関数はその境界で 1 回だけ呼び、`CudaError::Driver(e)` のみを
/// [`observe_driver_result`] へ委譲して分類・poison 化する
/// （`CudaError::Driver` 以外の variant は driver 呼び出し由来ではない
/// ため素通しする）。
///
/// `?` チェーンは最初に失敗した 1 回の driver 呼び出しで早期 return する
/// ため（本クレート全体の契約。個々の `.launch()`／`clone_htod`／
/// `clone_dtoh`／`synchronize()` はいずれも `?` で直結している）、演算
/// 内部で複数回の driver 呼び出しが起きても、呼び出し元まで伝播する
/// `CudaError` は常にその「最初に失敗した 1 回」を表す。したがって
/// 演算入口で 1 個の [`CallToken`]（`begin_driver_call` で 1 回だけ取得）
/// に対して本関数を 1 回呼ぶだけで、個々の内部呼び出しをすべて
/// `observe_driver_result` でラップした場合と同じ分類結果が得られる
/// （成功した先行呼び出しは `Ok` のため観測しても副作用がない）。
pub(crate) fn observe_cuda_result<T>(
    ordinal: usize,
    token: &CallToken,
    result: Result<T, CudaError>,
) -> Result<T, CudaError> {
    match result {
        Err(CudaError::Driver(e)) => match observe_driver_result::<()>(ordinal, token, Err(e)) {
            Err(e) => Err(CudaError::Driver(e)),
            Ok(()) => unreachable!(
                "observe_driver_result passes Err(_) through unchanged; it cannot turn an \
                 Err input into Ok"
            ),
        },
        other => other,
    }
}

/// [`observe_cuda_result`] の「値を消費しない」版（codex-review P0
/// 指摘・PR #1064 追補）。
///
/// `pool.rs::ReleaseCacheError` のように、driver 呼び出しの失敗を
/// `CudaError` 単体ではなく他のコンテキスト情報（`ReleasePhase` 等）と
/// 組にして保持する独自エラー型が呼び出し元に存在する場合、
/// `observe_cuda_result` のように `Result<T, CudaError>` を消費・再構築
/// させると呼び出し元の型情報（phase 等）が失われる。本関数は
/// `&CudaError`（借用）のみを受け取り、sticky エラーなら ordinal を
/// poison する副作用だけを行う（戻り値なし）。呼び出し元は自身の
/// エラー型（`ReleaseCacheError` 等）から `&CudaError` を取り出して
/// 本関数へ渡し、その後は自身のエラー型のまま `BackendError` へ変換
/// してよい（`ops.rs::CudaBackendOps::release_cached_device_memory`
/// 参照）。
pub(crate) fn observe_cuda_error_ref(ordinal: usize, token: &CallToken, err: &CudaError) {
    if let CudaError::Driver(e) = err {
        // `observe_driver_result` は `Result` を消費・再構築する設計だが、
        // 分類・poison化の副作用だけが必要なため `Err` を渡して結果を
        // 捨てる（`observe_driver_result` は `Err(_)` を無変更で返す契約
        // のため、ここで戻り値を調べる必要はない）。
        let _ = observe_driver_result::<()>(ordinal, token, Err(*e));
    }
}

/// `ordinal` が現在 poison 状態（`Poisoned{..}`。`unrecoverable` の
/// いずれも含む）かを返す。
// 本番経路では `begin_driver_call` が同等の poison 判定を内包する
// ため個別に呼ぶ必要がなく、`#[cfg(test)] mod poison_state_tests`
// からのみ参照される。テスト・将来の診断 CLI（`docs/
// guardrail-self-repair-cli.md` 系の拡張）用の公開面として残す。
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
// 本 PR（#1064）で検出側（`begin_driver_call`／`observe_driver_result`・
// `observe_cuda_result`）は `ops.rs`／`memory.rs` へ結線済みだが、回復側
// （`invalidate_with` を実 CUDA のストリーム同期・プローブクロージャで
// 呼び出す経路）はスコープ外のまま残す（poison 後の自動再生成は本 PR の
// 対象外。呼び出し元は #1062 へ引き継ぐ）。`ProbeFailure` は
// `invalidate_with` のテスト注入用クロージャの戻り値としてのみ使われる
// ため、本番経路では未使用のまま。
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
enum ResultClass {
    Sticky,
    OperationLocal,
}

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
        // CUDA Graph capture 系エラー（イシュー #1349）: capture の失敗は
        // driver 内部の capture 状態機械そのものを壊しうる（NVIDIA の
        // 規定では capture 失敗後のストリームは `end_capture` するまで
        // 使用不能）ため sticky 側へ明示的に固定する（`_` の wildcard に
        // 依存せず意図をコード化する。`docs/backend-cuda-graph-step-
        // capture-design.md` §4.2 F6）。
        CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED
        | CUDA_ERROR_STREAM_CAPTURE_INVALIDATED
        | CUDA_ERROR_STREAM_CAPTURE_MERGE
        | CUDA_ERROR_STREAM_CAPTURE_UNMATCHED
        | CUDA_ERROR_STREAM_CAPTURE_UNJOINED
        | CUDA_ERROR_STREAM_CAPTURE_ISOLATION
        | CUDA_ERROR_STREAM_CAPTURE_IMPLICIT
        | CUDA_ERROR_CAPTURED_EVENT
        | CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD => ResultClass::Sticky,
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
/// クロージャを渡して呼ぶ想定）。本 PR（#1064）で検出側
/// （`begin_driver_call`／`observe_driver_result`／`observe_cuda_result`）
/// は `ops.rs`／`memory.rs` へ結線し、sticky エラー観測後は ordinal を
/// fail-closed に poison するようになったが、poison からの**回復**
/// （本関数を実 CUDA クロージャ付きで呼び出す経路）は本 PR のスコープ外
/// のまま残す。呼び出し元の実装（poison 検出後にどのタイミングで
/// `invalidate` を試みるか）は #1062 へ引き継ぐ（テストのみが本関数を
/// 直接検証する）。
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
    //
    // Cursor Bugbot 指摘（PR #1064）: 待機側が `Retiring` の完了を待った
    // 後、単純にループの先頭へ戻って `state.phase` を再マッチすると、
    // 先行呼び出しが回復に成功して `Active` へ戻った直後の状態を
    // 「新規の Active」と誤認し、待機側自身が③（新たな retire の所有者）
    // へ遷移してしまう。これは 1 回の `invalidate` 要求ストームに対して
    // 世代を 2 回進めてしまい、直後（1 回目の回復で正しくなったはず）の
    // ハンドル・バッファを再び stale 世代にしてしまう欠陥だった。設計文書
    // `docs/backend-cuda-async-execution-design.md` §5 item 4a②の契約
    // 「待機側は先行呼び出しが確定させた結果をそのまま返して終了する
    // （③へ遷移して所有権を新たに得ることはない）」に従い、`Retiring` を
    // 一度でも観測した呼び出しは `became_waiter` を立てて待機専用の経路へ
    // 固定し、ループを抜けたあとは新規 retire の所有者にはならず、先行
    // 呼び出しが残した最終状態をそのまま Ok/Err に変換して return する。
    let mut became_waiter = false;
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
                if !became_waiter {
                    // 1 回だけ増やす（同じスレッドが spurious wakeup で
                    // 複数回このアームへ戻ってきても二重加算しない）。
                    became_waiter = true;
                    state.retiring_waiters += 1;
                }
                state = cell.1.wait(state).map_err(|e| {
                    BackendError::DeviceContextPoisoned(format!(
                        "invalidate の Condvar 待機中に Mutex が poison しました: {e}"
                    ))
                })?;
                continue;
            }
            Phase::Active | Phase::Poisoned { .. } => {
                break;
            }
        }
    }

    if became_waiter {
        // 駐機終了（`retiring_waiters` を対で減らす。モジュール内
        // `OrdinalState::retiring_waiters` ドキュメンテーションコメント
        // 参照）。
        state.retiring_waiters -= 1;
        // 先行呼び出し（所有者）が retire を確定させた最終状態を、
        // このスレッド自身の結果としてそのまま返す（新たな所有権は
        // 取らない。sync/probe を再実行しない）。
        return match state.phase {
            Phase::Active => Ok(()),
            Phase::Poisoned {
                unrecoverable: true,
            } => Err(BackendError::DeviceContextUnrecoverable {
                ordinal,
                probe_error: "concurrent invalidate() call observed a permanently poisoned \
                    result"
                    .to_string(),
            }),
            Phase::Poisoned {
                unrecoverable: false,
            } => Err(BackendError::DeviceContextPoisoned(format!(
                "ordinal {ordinal} is poisoned; call invalidate() to attempt recovery"
            ))),
            Phase::Retiring => unreachable!(
                "wait loop only exits when phase is no longer Retiring (see loop above)"
            ),
        };
    }

    // ここに到達するのは `became_waiter == false` の場合のみ、すなわち
    // このスレッドが最初に `Active`／`Poisoned{false}` を観測して自ら
    // 所有者になった場合に限る。
    state.phase = Phase::Retiring;

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

/// 非同期実行の遅延エラー伝播（状態機械。モジュール冒頭「非同期実行の
/// 遅延エラー伝播」参照）の単体テスト。実 CUDA 依存なし（GPU 不要）で
/// CI 常時実行できる（イシュー #1013 設計文書 §8 T3 系統の最小構成。
/// 実機依存の T1・T2・T4 は `crates/backend-cuda/tests/
/// async_ordering_real_device.rs`、T3b は `ops.rs` の `#[cfg(test)]`
/// モジュール、T3i は本ファイル末尾の `async_ordering_poison_tests`
/// 子モジュールへ、それぞれイシュー #1014 で実装済み）。
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
    fn observe_cuda_result_poisons_ordinal_on_sticky_driver_error() {
        let ordinal = unique_ordinal();
        let token = begin_driver_call(ordinal, &[0]).expect("begin succeeds");
        let observed =
            observe_cuda_result::<()>(ordinal, &token, Err(CudaError::Driver(sticky_err())));
        assert!(observed.is_err(), "Err はそのまま伝播するはず");
        assert!(
            is_poisoned(ordinal),
            "CudaError::Driver に包まれた sticky エラーも poison するはず \
             （ops.rs／memory.rs の境界結線が観測する形そのもの）"
        );

        // 演算入口の P0 契約: poison 後の次回呼び出しは `begin_driver_call`
        // が fail-closed に拒否する（PR #1064・`sgd.rs:180` codex-review
        // P0 指摘への対応そのもの）。
        let rejected = begin_driver_call(ordinal, &[0]);
        assert!(matches!(
            rejected,
            Err(BackendError::DeviceContextPoisoned(_))
        ));
    }

    /// [`observe_cuda_error_ref`] の回帰テスト（codex-review P0 指摘・
    /// `ops.rs:1117` 相当・PR #1064 追補）: `pool.rs::ReleaseCacheError`
    /// のように `CudaError` を他の情報（フェーズ識別子等）と組にして
    /// 保持する独自エラー型からでも、`&CudaError` を取り出して渡せば
    /// 同じ分類・poison 化が行われることを確認する（値を消費しないため
    /// 呼び出し元は独自エラー型を失わずに済む）。
    #[test]
    fn observe_cuda_error_ref_poisons_ordinal_on_sticky_driver_error() {
        let ordinal = unique_ordinal();
        let token = begin_driver_call(ordinal, &[0]).expect("begin succeeds");

        let driver_error = CudaError::Driver(sticky_err());
        observe_cuda_error_ref(ordinal, &token, &driver_error);

        assert!(
            is_poisoned(ordinal),
            "&CudaError 経由でも sticky エラーは poison するはず"
        );
        let rejected = begin_driver_call(ordinal, &[0]);
        assert!(matches!(
            rejected,
            Err(BackendError::DeviceContextPoisoned(_))
        ));
    }

    #[test]
    fn observe_cuda_error_ref_does_not_poison_on_operation_local_driver_error() {
        let ordinal = unique_ordinal();
        let token = begin_driver_call(ordinal, &[0]).expect("begin succeeds");

        let driver_error = CudaError::Driver(operation_local_err());
        observe_cuda_error_ref(ordinal, &token, &driver_error);

        assert!(
            !is_poisoned(ordinal),
            "operation-local エラーは &CudaError 経由でも poison しないはず"
        );
    }

    #[test]
    fn observe_cuda_error_ref_does_not_poison_on_non_driver_cuda_error() {
        let ordinal = unique_ordinal();
        let token = begin_driver_call(ordinal, &[0]).expect("begin succeeds");

        let non_driver = CudaError::InvalidShape {
            detail: "host-side validation failure, not a driver call".to_string(),
        };
        observe_cuda_error_ref(ordinal, &token, &non_driver);

        assert!(!is_poisoned(ordinal));
    }

    #[test]
    fn observe_cuda_result_does_not_poison_on_non_driver_cuda_error() {
        let ordinal = unique_ordinal();
        let token = begin_driver_call(ordinal, &[0]).expect("begin succeeds");
        // `CudaError::InvalidShape` はホスト側の事前検証失敗であり driver
        // 呼び出し由来ではないため、素通しして poison しない。
        let observed = observe_cuda_result::<()>(
            ordinal,
            &token,
            Err(CudaError::InvalidShape {
                detail: "host-side validation failure, not a driver call".to_string(),
            }),
        );
        assert!(observed.is_err());
        assert!(
            !is_poisoned(ordinal),
            "driver 呼び出し由来でないエラーは poison しないはず"
        );
    }

    #[test]
    fn observe_cuda_result_does_not_poison_on_operation_local_driver_error() {
        let ordinal = unique_ordinal();
        let token = begin_driver_call(ordinal, &[0]).expect("begin succeeds");
        let observed = observe_cuda_result::<()>(
            ordinal,
            &token,
            Err(CudaError::Driver(operation_local_err())),
        );
        assert!(observed.is_err());
        assert!(!is_poisoned(ordinal));
        // operation-local エラーでは以降の呼び出しを拒否しない
        // （sticky エラーとの非対称性の検証。P0 対応の裏取り）。
        drop(token);
        let next = begin_driver_call(ordinal, &[0]);
        assert!(
            next.is_ok(),
            "operation-local エラーの観測後も begin_driver_call は成功し続けるはず"
        );
    }

    #[test]
    fn observe_cuda_result_passes_through_ok_without_side_effects() {
        let ordinal = unique_ordinal();
        let token = begin_driver_call(ordinal, &[0]).expect("begin succeeds");
        let observed = observe_cuda_result(ordinal, &token, Ok::<_, CudaError>(7));
        assert_eq!(observed.unwrap(), 7);
        assert!(!is_poisoned(ordinal));
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

    /// Cursor Bugbot 指摘（PR #1064・`context_cache.rs:679` 相当）の
    /// 回帰テスト: 並行 `invalidate` 要求のうち、先に `Retiring` を観測
    /// して待機した側（waiter）は、所有者（owner）が回復を確定させた後に
    /// 自ら新たな所有者へ遷移して 2 回目の retire/probe を走らせては
    /// ならない。これが起きると 1 回の要求ストームで `generation` が
    /// 2 回進んでしまい、1 回目の回復で正しくなったはずのハンドルを
    /// 再び stale 世代にしてしまう。
    ///
    /// `std::thread::scope` で owner／waiter を同時に起動する。owner の
    /// `sync` クロージャ内でチャネル通知した後、`OrdinalState::
    /// retiring_waiters`（waiter 自身の `invalidate_with` 呼び出しが
    /// `Phase::Retiring` を観測し `Condvar::wait` で実際に駐機した
    /// ことを示す明示カウンタ）が 1 になるまでポーリングしてから owner
    /// の sync を完了させる。
    ///
    /// CI 実測で間欠的に FAILED（イシュー #1064 追補）: 旧実装は
    /// waiter 側で `Phase::Retiring` を別途ポーリングしてから
    /// `invalidate_with` を呼んでいたが、「ポーリングが `Retiring` を
    /// 観測する」タイミングと「waiter 自身の `invalidate_with` 呼び出しが
    /// 実際に `Phase::Retiring` を観測して `wait` に入る」タイミングは
    /// 別々のロック取得であり、両者の間に owner が sync/probe を完了
    /// させてしまうと waiter の `invalidate_with` 呼び出しは
    /// `Phase::Active` を最初から観測し、became_waiter とはならず
    /// **自ら新たな所有者になってしまう**（`Condvar` は「現在待機中の
    /// スレッド数」を問い合わせる API を持たないため、外側からの
    /// ポーリングだけでは「waiter の呼び出しが `wait` に入った」ことを
    /// 判定できなかった）。`retiring_waiters` は waiter の
    /// `invalidate_with` 呼び出し自身が `wait` へ入る直前に増分する
    /// ため、これをポーリングすれば「waiter は必ず `Phase::Retiring` を
    /// 観測して `wait` に入った」ことを取りこぼしなく判定できる
    /// （`OrdinalState::retiring_waiters` ドキュメンテーションコメント
    /// 参照）。
    #[test]
    fn invalidate_with_concurrent_waiter_does_not_become_a_new_owner() {
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        let registry = OrdinalRegistry::new();
        let ordinal = 0usize;
        let (owner_entered_tx, owner_entered_rx) = mpsc::channel::<()>();
        let (release_owner_tx, release_owner_rx) = mpsc::channel::<()>();

        std::thread::scope(|scope| {
            let registry_ref = &registry;
            let owner = scope.spawn(move || {
                invalidate_with(
                    registry_ref,
                    ordinal,
                    move || {
                        // owner が Retiring へ遷移しロックを手放した直後
                        // （sync クロージャ呼び出し時点）に waiter を起動
                        // してよいことを知らせ、waiter が実際に `wait` へ
                        // 入るまで待ってから sync を完了させる。
                        owner_entered_tx.send(()).unwrap();
                        release_owner_rx.recv().unwrap();
                        Ok(())
                    },
                    || Ok(()),
                )
            });

            // owner が Retiring へ遷移するのを待つ。
            owner_entered_rx.recv().unwrap();

            let waiter =
                scope.spawn(move || invalidate_with(registry_ref, ordinal, || Ok(()), || Ok(())));

            // waiter 自身の `invalidate_with` 呼び出しが `Phase::Retiring`
            // を観測し `Condvar::wait` で実際に駐機するまで待つ
            // （`retiring_waiters` が 1 になった時点が「waiter は必ず
            // 待機側の経路に入った」ことを意味する決定的な同期点）。
            let cell = registry_ref.entry(ordinal).unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                if cell.0.lock().unwrap().retiring_waiters >= 1 {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "waiter が 5 秒以内に Condvar::wait へ入らなかった（テスト環境の                      スレッドスケジューリング異常の疑い）"
                );
                std::thread::yield_now();
            }
            release_owner_tx.send(()).unwrap();

            let owner_result = owner.join().expect("owner thread does not panic");
            let waiter_result = waiter.join().expect("waiter thread does not panic");

            assert!(owner_result.is_ok(), "owner の retire は成功するはず");
            assert!(
                waiter_result.is_ok(),
                "waiter は owner が確定させた Active 状態をそのまま Ok として返すはず \
                 （新たな所有者にはならない）: {waiter_result:?}"
            );
        });

        let cell = registry.entry(ordinal).unwrap();
        let state = cell.0.lock().unwrap();
        assert_eq!(
            state.generation, 1,
            "1 回の要求ストーム（owner 1 回 + waiter 1 回）に対して generation は \
             1 回だけ進むはず（waiter が 2 回目の retire/probe を走らせていたら 2 に \
             なってしまう）"
        );
        assert_eq!(state.phase, Phase::Active);
        assert_eq!(
            state.retiring_waiters, 0,
            "テスト終了時点で駐機中のスレッドは残らないはず"
        );
    }

    // -----------------------------------------------------------------
    // CUDA Graph capture 状態機械（イシュー #1349）の GPU 非依存な
    // ホストモデルテスト。実 CUDA driver・`CudaStream` を要さない
    // `begin_capture_session`／`is_capturing_on_current_thread`／
    // `begin_sync_point_call` の制御フローのみを検証する。
    // -----------------------------------------------------------------

    #[test]
    fn begin_capture_session_succeeds_and_marks_current_thread_capturing() {
        let ordinal = unique_ordinal();
        assert!(!is_capturing_on_current_thread(ordinal));
        let guard = begin_capture_session(ordinal).expect("first capture session succeeds");
        assert!(is_capturing_on_current_thread(ordinal));
        drop(guard);
        assert!(
            !is_capturing_on_current_thread(ordinal),
            "CaptureGuard の Drop で capture 状態が解除されるはず"
        );
    }

    #[test]
    fn begin_capture_session_rejects_reentrant_capture_on_same_ordinal() {
        let ordinal = unique_ordinal();
        let _guard = begin_capture_session(ordinal).expect("first capture session succeeds");
        let second = begin_capture_session(ordinal);
        assert!(
            matches!(second, Err(BackendError::InvalidArgument(_))),
            "同一 ordinal への 2 重 capture 開始は拒否されるはず: {second:?}"
        );
    }

    #[test]
    fn begin_capture_session_allows_reuse_after_guard_is_dropped() {
        let ordinal = unique_ordinal();
        let guard = begin_capture_session(ordinal).expect("first capture session succeeds");
        drop(guard);
        let second = begin_capture_session(ordinal);
        assert!(
            second.is_ok(),
            "先行 capture が終了（guard drop）していれば再度開始できるはず: {second:?}"
        );
    }

    #[test]
    fn begin_sync_point_call_rejects_before_touching_driver_while_capturing() {
        let ordinal = unique_ordinal();
        let _guard = begin_capture_session(ordinal).expect("capture session succeeds");
        let result = begin_sync_point_call(ordinal, &[], "upload");
        assert!(
            matches!(&result, Err(BackendError::Unsupported(msg)) if msg.contains("upload")),
            "capture 中の同期点呼び出しは Unsupported で拒否されるはず: {result:?}"
        );
        // 拒否は `begin_driver_call` に到達する前（in_flight を変更しない）
        // ことを確認する。
        let cell = ordinal_registry().entry(ordinal).unwrap();
        let state = cell.0.lock().unwrap();
        assert_eq!(
            state.in_flight, 0,
            "同期点ガードの拒否は begin_driver_call の in_flight を変更しないはず"
        );
    }

    #[test]
    fn begin_sync_point_call_delegates_to_begin_driver_call_when_not_capturing() {
        let ordinal = unique_ordinal();
        let token = begin_sync_point_call(ordinal, &[], "upload");
        assert!(
            token.is_ok(),
            "capture 中でなければ begin_driver_call と同じ挙動になるはず: {token:?}"
        );
    }

    #[test]
    fn is_capturing_on_current_thread_is_false_on_a_fresh_ordinal() {
        let ordinal = unique_ordinal();
        assert!(!is_capturing_on_current_thread(ordinal));
    }

    /// capture モード（`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`）は「この
    /// スレッドが開始した capture」のみを検出する契約であり、別スレッド
    /// から見た `is_capturing_on_current_thread` は capture 中でも
    /// `false` を返す——これは `begin_sync_point_call`（同期点の早期
    /// 拒否判定）が別スレッドを対象にしない、という**この関数専用**の
    /// 契約であり、別スレッドの driver 呼び出し一般が野放しという意味
    /// ではない（codex-review P0 指摘対応で `begin_driver_call` 自体が
    /// 別途、capture 中は capture 開始スレッド以外を一律拒否するように
    /// なった。上記 `begin_driver_call_rejects_other_thread_while_
    /// capturing` 参照。design doc §4.2 追記）。
    #[test]
    fn is_capturing_on_current_thread_is_thread_local() {
        let ordinal = unique_ordinal();
        let guard = begin_capture_session(ordinal).expect("capture session succeeds");
        let observed_from_other_thread =
            std::thread::spawn(move || is_capturing_on_current_thread(ordinal))
                .join()
                .expect("spawned thread does not panic");
        assert!(
            !observed_from_other_thread,
            "別スレッドからは capture 中と観測されないはず（THREAD_LOCAL 契約）"
        );
        drop(guard);
    }

    /// capture 系 CUresult（900〜908）が sticky に分類されることを固定する
    /// （イシュー #1349・design doc §4.2 F6）。
    #[test]
    fn classify_capture_related_codes_are_sticky() {
        for code in [
            CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
            CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
            CUresult::CUDA_ERROR_STREAM_CAPTURE_MERGE,
            CUresult::CUDA_ERROR_STREAM_CAPTURE_UNMATCHED,
            CUresult::CUDA_ERROR_STREAM_CAPTURE_UNJOINED,
            CUresult::CUDA_ERROR_STREAM_CAPTURE_ISOLATION,
            CUresult::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT,
            CUresult::CUDA_ERROR_CAPTURED_EVENT,
            CUresult::CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD,
        ] {
            assert_eq!(
                classify_cuda_result(&DriverError(code)),
                ResultClass::Sticky,
                "{code:?} は sticky に分類されるはず"
            );
        }
    }

    /// codex-review P0 指摘（イシュー #1349）: capture 中は capture を
    /// 開始したスレッド以外からの `begin_driver_call` を拒否し、共有
    /// ストリームへの意図しないカーネル起動混入を防ぐ（`design doc`
    /// §4.2 追記）。
    #[test]
    fn begin_driver_call_rejects_other_thread_while_capturing() {
        let ordinal = unique_ordinal();
        let guard = begin_capture_session(ordinal).expect("capture session succeeds");
        let other_thread_result =
            std::thread::spawn(move || begin_driver_call(ordinal, &[0]).map(|_| ()))
                .join()
                .expect("spawned thread does not panic");
        assert!(
            matches!(other_thread_result, Err(BackendError::Unsupported(_))),
            "capture 中の別スレッドからの driver 呼び出しは Unsupported で拒否されるはず:              {other_thread_result:?}"
        );
        drop(guard);
    }

    /// capture を開始した**同一スレッド**の `begin_driver_call` は通す
    /// （`graph::run_captured_sgd_step_segment` の `body()` 実行契約。上記テストの
    /// 対照ケース）。
    #[test]
    fn begin_driver_call_allows_capturing_thread_itself() {
        let ordinal = unique_ordinal();
        let guard = begin_capture_session(ordinal).expect("capture session succeeds");
        let token = begin_driver_call(ordinal, &[0]);
        assert!(
            token.is_ok(),
            "capture を開始した本人のスレッドは引き続き driver 呼び出しできるはず: {token:?}"
        );
        drop(token);
        drop(guard);
    }

    // =====================================================================
    // `begin_buffer_release`／`capturing_active` の状態機械（codex-review
    // P0 再指摘・Cursor Bugbot High 指摘対応・PR #1390 再修正）。
    // `memory.rs::CudaBufferHandle::Drop` の排他契約をホストモデルで検証
    // する（`.claude/rules/coding-rust.md` テスト規約: 受け入れ基準に
    // 対応するテストを同一 PR に含める）。
    // =====================================================================

    /// (a) `begin_capture_session` の in_flight ドレイン中（`capturing_active`
    /// がまだ立っていない区間）は、別スレッドの `begin_buffer_release` は
    /// 駐機せず即座に登録される（`begin_buffer_release` doc コメント
    /// 「P0 再修正」の「ドレイン中は素通し」契約）。ドレイン完了と
    /// 同一ロック区間内で `capturing_active` が立つこと（`begin_capture_
    /// session` doc コメント「P0 再修正」）・`CaptureGuard` drop で
    /// 解除されること（`OrdinalState::capturing_active` doc コメント）も
    /// あわせて確認する。
    #[test]
    fn begin_buffer_release_passes_through_during_capture_session_drain_then_capturing_active_is_set_after_drain()
     {
        let ordinal = unique_ordinal();
        // in-flight を模した CallToken（`begin_capture_session` の drain
        // 対象になる）。
        let in_flight_token = begin_driver_call(ordinal, &[0]).expect("token 1 取得");

        let handle = std::thread::spawn(move || begin_capture_session(ordinal));

        // 別スレッドが `state.capture` を設定（drain 待機に入った）まで
        // ポーリングする。この時点では `in_flight_token` をまだ保持して
        // いるため drain は完了しておらず、`capturing_active` は
        // 立っていないはず。
        let cell = ordinal_registry().entry(ordinal).expect("registry entry");
        loop {
            let state = cell.0.lock().unwrap_or_else(|e| e.into_inner());
            if state.capture.is_some() {
                assert!(
                    !state.capturing_active,
                    "in_flight ドレイン中は capturing_active がまだ立っていないはず"
                );
                break;
            }
            drop(state);
            std::thread::yield_now();
        }

        // ドレイン中（`capturing_active == false`）の別スレッドからの
        // `begin_buffer_release` は駐機せず即座に登録されるはず。もし
        // 誤って `state.capture.is_some()` だけを条件に駐機してしまうと、
        // ここで `begin_capture_session` 側の drain（`in_flight_token` の
        // drop 待ち）と本呼び出しの `Condvar::wait` が互いを待ち合い、
        // このテスト自体がハングして CI タイムアウトで検出される
        // （Cursor Bugbot High 指摘のデッドロックそのもの）。
        let release_token = begin_buffer_release(ordinal);
        drop(release_token);

        // drain を完了させる。
        drop(in_flight_token);
        let guard = handle
            .join()
            .expect("spawned thread does not panic")
            .expect("capture session succeeds after drain");

        {
            let state = cell.0.lock().unwrap_or_else(|e| e.into_inner());
            assert!(
                state.capturing_active,
                "drain 完了後（begin_capture_session が返った後）は capturing_active が \
                 立っているはず"
            );
        }

        drop(guard);
        let state_after = cell.0.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            !state_after.capturing_active,
            "CaptureGuard の Drop で capturing_active も解除されるはず"
        );
    }

    /// (b) `capturing_active` が立った後（実際の driver capture 進行中に
    /// 相当する区間）は、別スレッドの `begin_buffer_release` は
    /// `CaptureGuard` の drop まで駐機する（`begin_buffer_release` doc
    /// コメント「P0 再修正」の本題）。
    #[test]
    fn begin_buffer_release_blocks_other_thread_while_capturing_active_until_guard_drops() {
        let ordinal = unique_ordinal();
        // in_flight が 0 のため drain は即完了し、`begin_capture_session`
        // が返った時点で `capturing_active` は既に true。
        let guard = begin_capture_session(ordinal).expect("capture session succeeds");
        {
            let cell = ordinal_registry().entry(ordinal).unwrap();
            assert!(
                cell.0
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .capturing_active
            );
        }

        let (unblock_tx, unblock_rx) = std::sync::mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            let token = begin_buffer_release(ordinal);
            // 駐機が解除され登録できた時点で通知する。
            unblock_tx.send(()).unwrap();
            drop(token);
        });

        // `capturing_active` の間は別スレッドの `begin_buffer_release` が
        // 駐機し続け、短いタイムアウト内には通知が届かないはず
        // （フレーク回避のための現実的なタイムアウト。真の無限待機の
        // 証明ではないが、駐機せず即座に登録されてしまう回帰は確実に
        // 検出できる）。
        assert!(
            unblock_rx
                .recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "capturing_active の間は別スレッドの begin_buffer_release は駐機し続けるはず"
        );

        drop(guard);
        unblock_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect(
                "CaptureGuard の drop 後は begin_buffer_release の駐機が解除され \
                 token を得られるはず",
            );
        handle.join().expect("spawned thread does not panic");
    }

    /// (c) capture 中の**スレッド自身**からの `begin_buffer_release` は
    /// 駐機せず即座に登録される（`begin_buffer_release` doc コメント
    /// 「capture 中のスレッド自身」参照。自己デッドロック回避）。
    #[test]
    fn begin_buffer_release_does_not_block_the_capturing_thread_itself() {
        let ordinal = unique_ordinal();
        let guard = begin_capture_session(ordinal).expect("capture session succeeds");
        // ここでハングすればテストがタイムアウトし回帰を検出する。
        let release_token = begin_buffer_release(ordinal);
        drop(release_token);
        drop(guard);
    }
}

/// イシュー #1014（設計文書 §8 T3i）: `invalidate_with` を実 CUDA
/// クロージャで検証する実機依存テスト（`#[ignore]`）。`OrdinalRegistry`・
/// `invalidate_with`・`ProbeFailure` がモジュール非公開のため、統合テスト
/// （`tests/` 配下）ではなく子モジュールとして配置する
/// （`jit_cache_regression_tests.rs` と同じ配置方式）。
#[cfg(test)]
#[path = "async_ordering_poison_tests.rs"]
mod async_ordering_poison_tests;

/// イシュー #1084: `invalidate_with` の poison 回復経路（実 CUDA の
/// `sync`／実処理プローブ）を実機で検証する追加テスト（T3i の隣接拡張。
/// 上記 `async_ordering_poison_tests` と同じ配置理由・同じ非公開
/// アイテムへのアクセス方針）。
#[cfg(test)]
#[path = "poison_recovery_real_device_tests.rs"]
mod poison_recovery_real_device_tests;
