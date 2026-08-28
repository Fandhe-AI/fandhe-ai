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
//! 各キャッシュは `ordinal` をキーとする `HashMap<usize, Arc<T>>` を
//! `OnceLock<Mutex<_>>` で保持するプロセスワイド static。エントリは
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

/// `ordinal` キーのプロセスワイド `HashMap<usize, Arc<T>>` に対する
/// 「ヒットなら clone・ミスなら `build` で構築して登録」の共通ロジック。
///
/// ロック区間を 2 段に分ける（先に読み取り専用でヒット判定、ミス時のみ
/// `build` をロック外で実行してから再度ロックして登録）ことで、
/// コストの高い `build`（NVRTC コンパイル等を含みうる）をプロセス全体の
/// `Mutex` 保持中に実行しない（`module_cache.rs::KernelModuleCache::insert`
/// の「evict をロック外で drop する」判断と同じ「重い処理をロック外に
/// 追い出す」方針）。2 スレッドが同時にミスした場合は両方が `build` を
/// 実行しうるが、登録は `entry().or_insert_with` で先着 1 件のみが採用
/// され、後着側の構築結果は呼び出し元に返した後 drop されるだけであり
/// 数値的な誤りにはつながらない（許容する冗長構築。呼び出し頻度に対し
/// 発生確率は低く、複雑な二重チェックロックを導入するほどではないと
/// 判断した。実装計画 §3.1）。
///
/// `build` の失敗（`Err`）はキャッシュへ格納せず、そのまま呼び出し元へ
/// 伝播する（モジュール冒頭「fail-fast 契約」参照）。
fn get_or_build<T>(
    cache: &Mutex<HashMap<usize, Arc<T>>>,
    ordinal: usize,
    build: impl FnOnce() -> Result<T, CudaError>,
) -> Result<Arc<T>, CudaError> {
    {
        let guard = lock_cache(cache)?;
        if let Some(existing) = guard.get(&ordinal) {
            return Ok(Arc::clone(existing));
        }
    }
    let built = Arc::new(build()?);
    let mut guard = lock_cache(cache)?;
    Ok(Arc::clone(guard.entry(ordinal).or_insert(built)))
}

/// `ordinal` 番目の GPU に対応する [`CudaDevice`] をプロセス内キャッシュ
/// から取得する。ヒット時は `CudaContext::new`／NVRTC 初期化を再実行
/// しない（受け入れ条件 1）。
///
/// [`crate::device::CudaDeviceProvider::probe`]（`enumerate`／`select` の
/// 内部経路）・[`crate::ops::CudaBackendOps::device_handle`] の唯一の
/// 呼び出し先とする。
pub(crate) fn cached_device(ordinal: usize) -> Result<Arc<CudaDevice>, CudaError> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<CudaDevice>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ordinal, || CudaDevice::new(ordinal))
}

/// `ordinal` に対応する [`CudaGemm`] スイートをプロセス内キャッシュから
/// 取得する。`device` はヒット時未使用（ミス時の構築にのみ使う）。
///
/// `ops::CudaBackendOps::gemm`／`gemm_bias_act` の唯一の呼び出し先。
pub(crate) fn cached_gemm(ordinal: usize, device: &CudaDevice) -> Result<Arc<CudaGemm>, CudaError> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<CudaGemm>>>> = OnceLock::new();
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
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<CudaElementwise>>>> = OnceLock::new();
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
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<CudaRmsNorm>>>> = OnceLock::new();
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
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<CudaSoftmax>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_build(cache, ordinal, || CudaSoftmax::new(device))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`get_or_build`] は GPU 非依存の純粋なキャッシュロジックのため、
    /// 実 CUDA 型を要求しない汎用 `T`（ここでは `u32`）で検証する
    /// （`module_cache.rs::LruCache` のテストと同じ「GPU 不要ロジックは
    /// 実カーネル型に依存しない形でテストする」方針）。
    fn fresh_cache<T>() -> Mutex<HashMap<usize, Arc<T>>> {
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
