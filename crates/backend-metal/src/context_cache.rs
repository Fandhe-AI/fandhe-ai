//! Metal コンテキスト／カーネルスイートのプロセス内キャッシュ
//! （イシュー #930。診断 #927・`docs/perf/metal-fixed-overhead-diagnosis.md`
//! が特定した「演算メソッド呼び出しごとの Metal 資源都度構築」固定
//! オーバーヘッド〈約 5 ms・N 非依存〉の解消）。
//!
//! CUDA 側 [`crate::context_cache`] 相当のプロセスワイド static キャッシュ
//! （`crates/backend-cuda/src/context_cache.rs`。イシュー #929）と同型の
//! 設計を Metal 側に揃える。ただし CUDA 側は `Device::Cuda(ordinal)` が
//! 複数 GPU を ordinal で区別するのに対し、`Device::Metal` は ordinal を
//! 持たない単一 variant（`docs/public-api-design.md` §4.1・
//! `crate::device::MetalDeviceProvider` と同じ位置付け）のため、本モジュールは
//! `HashMap<usize, Arc<T>>` ではなく型ごとに単一エントリを持つ
//! `OnceLock<Mutex<Option<Arc<T>>>>` を使う（システムデフォルトの Metal
//! デバイス 1 台のみを扱う前提。`crate::ops::MetalBackendOps` の
//! ドキュメンテーションコメント参照）。
//!
//! # 何をキャッシュするか
//!
//! デバイス層は [`cached_context`] が [`crate::context::MetalContext`]
//! （`MTLCreateSystemDefaultDevice` + `newCommandQueue` + `supportsFamily` +
//! IOKit occupancy probe を内包する。`context.rs` 参照）をプロセスワイドに
//! 共有する。
//!
//! カーネルスイート層は [`cached_gemm`]／[`cached_elementwise`]／
//! [`cached_rmsnorm`]／[`cached_softmax`] が、それぞれ
//! [`crate::gemm::MetalGemm`]（本番既定構成〈`MetalGemm::new` 相当〉のみを
//! キャッシュし、A/B ベンチ用 `new_with_swizzle`／`new_with_fine_barrier`
//! 入口はキャッシュ対象外で従来どおり直接構築する）／
//! [`crate::elementwise::MetalElementwise`]／[`crate::rmsnorm::MetalRmsNorm`]／
//! [`crate::softmax::MetalSoftmax`]（いずれも `new` 内で MSL 実行時
//! コンパイル + 複数パイプライン構築を行う。`ops.rs` 冒頭コメント参照）を
//! 共有する。
//!
//! `ops::MetalBackendOps` は演算メソッド呼び出しごとにこれらを都度構築
//! していた（イシュー #930 実装計画 §3.4）。本モジュールを経由させる
//! ことで、同一プロセス内の 2 回目以降の呼び出しはデバイス取得・MSL
//! コンパイルを再度支払わない。
//!
//! # fail-fast 契約（エラーはキャッシュしない）
//!
//! ミス時の構築が失敗した場合（Metal デバイス不在・MSL コンパイル失敗等）、
//! その `Err` はキャッシュへ格納しない。次回呼び出しは再度構築を試みる
//! （CUDA 側 `context_cache.rs` と同じ fail-fast 契約。`ops.rs::gemm` の
//! `DeviceUnavailable` 分類契約〈PR #262 レビュー対応〉を不変に保つ）。
//!
//! # 生存期間
//!
//! 各キャッシュはプロセスの生存期間中 evict されない（デバイス 1 台 ×
//! スイート数個の有界エントリのみを扱うため、`crate::gemm::MetalGemm::
//! tiled_cache`〈shape 特化コンパイルキャッシュ〉と異なり容量制御・LRU は
//! 不要。CUDA 側 `context_cache.rs` と同じ判断）。`MetalGemm`／
//! `MetalElementwise`／`MetalRmsNorm`／`MetalSoftmax` の `Arc` はいずれも
//! 内部で `Retained<MtlDevice>`／`Retained<MtlQueue>`（`MetalContext`
//! 経由）を強参照するため、スイートキャッシュのエントリが 1 つでも生存
//! する限り対応する `MetalContext` は解放されない。
//!
//! # `Mutex` poison
//!
//! `gemm.rs::lock_tile_cache` と同じ方針で、`Mutex` の poison を
//! [`MetalError::ContextCacheUnavailable`] へ変換し panic させない
//! （本モジュールの臨界区間自体は `unwrap`/`expect` を持たないため通常
//! 到達しない）。呼び出し元（`ops::MetalBackendOps`）はこのエラーを
//! そのまま `BackendError` へ伝播してよい（本キャッシュは純粋な最適化
//! ではあるが、ロック不能は環境異常を示すため型付きエラーとして呼び出し
//! 元へ伝える。キャッシュなしへの縮退運転は行わない — 縮退させると
//! 「毎回フレッシュ構築」に戻り受け入れ条件 1〈2 回目以降が構築費を
//! 支払わない〉自体が崩れるため）。

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::context::MetalContext;
use crate::elementwise::MetalElementwise;
use crate::error::MetalError;
use crate::gemm::MetalGemm;
use crate::rmsnorm::MetalRmsNorm;
use crate::softmax::MetalSoftmax;

/// コンパイル時アサーション: 本モジュールがキャッシュする全ハンドル型が
/// `Send + Sync` であることを固定する。`OnceLock<Mutex<Option<Arc<T>>>>`
/// static 経由で複数スレッドから共有する前提（`Arc<T>` を他スレッドへ渡す・
/// 複数スレッドから同時に `&T` で参照する）が成立するには `T: Send + Sync`
/// が必須。objc2-metal 0.3.2 では `MTLDevice`／`MTLCommandQueue`／
/// `MTLLibrary`／`MTLComputePipelineState` の各 protocol が `Send + Sync`
/// を supertrait に持つため、それらを `Retained<ProtocolObject<dyn _>>`
/// で保持するのみの各ハンドル型は成立するはずだが、将来のフィールド
/// 追加でこの前提が崩れた場合にここでコンパイルエラーとして検出する
/// （CUDA 側 `context_cache.rs` の同名アサーションと同じ設計判断）。
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MetalContext>();
    assert_send_sync::<MetalGemm>();
    assert_send_sync::<MetalElementwise>();
    assert_send_sync::<MetalRmsNorm>();
    assert_send_sync::<MetalSoftmax>();
};

/// `Mutex` guard 取得の共通ヘルパー。poison を
/// [`MetalError::ContextCacheUnavailable`] へ変換する（`gemm.rs::
/// lock_tile_cache` と同じ変換方針。panic 経路を持たない
/// `.claude/rules/coding-rust.md`）。
fn lock_cache<T>(
    mutex: &Mutex<Option<Arc<T>>>,
) -> Result<MutexGuard<'_, Option<Arc<T>>>, MetalError> {
    mutex
        .lock()
        .map_err(|e| MetalError::ContextCacheUnavailable {
            detail: format!("context cache mutex poisoned: {e}"),
        })
}

/// 単一エントリの `OnceLock<Mutex<Option<Arc<T>>>>` に対する
/// 「ヒットなら clone・ミスなら `build` で構築して登録」の共通ロジック。
/// `Device::Metal` が ordinal を持たない単一 variant であるため、
/// CUDA 側 `context_cache.rs::get_or_build`（`ordinal` キーの
/// `HashMap`）と異なりキーを持たない。
///
/// ロック区間を 2 段に分ける（先に読み取り専用でヒット判定、ミス時のみ
/// `build` をロック外で実行してから再度ロックして登録）ことで、コストの
/// 高い `build`（MSL コンパイル等を含みうる）をプロセス全体の `Mutex`
/// 保持中に実行しない（CUDA 側 `get_or_build` と同じ「重い処理をロック外
/// に追い出す」方針）。2 スレッドが同時にミスした場合は両方が `build` を
/// 実行しうるが、登録は `get_or_insert_with` で先着 1 件のみが採用され、
/// 後着側の構築結果は呼び出し元に返した後 drop されるだけであり数値的な
/// 誤りにはつながらない（許容する冗長構築。実装計画 §3.1）。
///
/// `build` の失敗（`Err`）はキャッシュへ格納せず、そのまま呼び出し元へ
/// 伝播する（モジュール冒頭「fail-fast 契約」参照）。
fn get_or_build<T>(
    cache: &Mutex<Option<Arc<T>>>,
    build: impl FnOnce() -> Result<T, MetalError>,
) -> Result<Arc<T>, MetalError> {
    {
        let guard = lock_cache(cache)?;
        if let Some(existing) = guard.as_ref() {
            return Ok(Arc::clone(existing));
        }
    }
    let built = Arc::new(build()?);
    let mut guard = lock_cache(cache)?;
    Ok(Arc::clone(guard.get_or_insert(built)))
}

/// システムデフォルトの Metal デバイスに対応する [`MetalContext`] を
/// プロセス内キャッシュから取得する。ヒット時は `MTLCreateSystemDefaultDevice`
/// ／`newCommandQueue`／occupancy probe を再実行しない（受け入れ条件 1）。
///
/// `ops::MetalBackendOps` の各演算メソッドの唯一の呼び出し先とする。
pub(crate) fn cached_context() -> Result<Arc<MetalContext>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalContext>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, MetalContext::new)
}

/// 本番既定構成の [`MetalGemm`]（`MetalGemm::new` 相当。threadgroup ID
/// スウィズル・simdgroup 細粒度同期はいずれも既定値 `false`）を
/// プロセス内キャッシュから取得する。
///
/// A/B ベンチ用 `MetalGemm::new_with_swizzle`／`new_with_fine_barrier`
/// はキャッシュ対象外（呼び出し元がそれぞれ明示的に直接構築する。
/// 実装計画 §3.1）。`ctx` はヒット時未使用（ミス時の構築にのみ使う）。
///
/// `ops::MetalBackendOps::gemm`／`gemm_bias_act` の唯一の呼び出し先。
pub(crate) fn cached_gemm(ctx: &Arc<MetalContext>) -> Result<Arc<MetalGemm>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalGemm>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, || MetalGemm::new(ctx))
}

/// [`MetalElementwise`] スイートをプロセス内キャッシュから取得する。
/// `ops::MetalBackendOps::elementwise_binary`／`elementwise_unary` の
/// 唯一の呼び出し先。
pub(crate) fn cached_elementwise(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<MetalElementwise>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalElementwise>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, || MetalElementwise::new(ctx))
}

/// [`MetalRmsNorm`] スイートをプロセス内キャッシュから取得する。
/// `ops::MetalBackendOps::run_fused_rmsnorm` の唯一の呼び出し先。
pub(crate) fn cached_rmsnorm(ctx: &Arc<MetalContext>) -> Result<Arc<MetalRmsNorm>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalRmsNorm>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, || MetalRmsNorm::new(ctx))
}

/// [`MetalSoftmax`] スイートをプロセス内キャッシュから取得する。
/// `ops::MetalBackendOps::run_fused_softmax` の唯一の呼び出し先。
pub(crate) fn cached_softmax(ctx: &Arc<MetalContext>) -> Result<Arc<MetalSoftmax>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalSoftmax>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, || MetalSoftmax::new(ctx))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`get_or_build`] は Metal 非依存の純粋なキャッシュロジックのため、
    /// 実 Metal 型を要求しない汎用 `T`（ここでは `u32`）で検証する
    /// （CUDA 側 `context_cache.rs` のテストと同じ「GPU 不要ロジックは
    /// 実カーネル型に依存しない形でテストする」方針。本テストは Mac 実機
    /// 非依存で CI〈Linux〉でも実行される。ただし本モジュール全体は
    /// `cfg(target_os = "macos")` 限定〈`lib.rs`〉のため、実際に走るのは
    /// Mac 実機セッションのみ）。
    fn fresh_cache<T>() -> Mutex<Option<Arc<T>>> {
        Mutex::new(None)
    }

    #[test]
    fn get_or_build_constructs_once_and_caches_hit() {
        let cache = fresh_cache::<u32>();
        let calls = std::sync::atomic::AtomicU32::new(0);

        let first = get_or_build(&cache, || {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(42)
        })
        .expect("build succeeds");
        let second = get_or_build(&cache, || {
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

    /// fail-fast 契約（モジュール冒頭コメント）: `build` が失敗しても
    /// キャッシュへ格納されず、次回呼び出しで再度 `build` が呼ばれる。
    #[test]
    fn get_or_build_does_not_cache_errors() {
        let cache = fresh_cache::<u32>();
        let calls = std::sync::atomic::AtomicU32::new(0);

        let build = || {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err(MetalError::ContextCacheUnavailable {
                detail: "simulated failure".into(),
            })
        };

        assert!(get_or_build(&cache, build).is_err());
        assert!(get_or_build(&cache, build).is_err());
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "エラーはキャッシュされないため build は毎回呼ばれるはず"
        );

        // デバイスが後から利用可能になった環境を模す: 3 回目は成功する。
        let third = get_or_build(&cache, || Ok(7)).expect("recovers once build succeeds");
        assert_eq!(*third, 7);
    }

    /// `Mutex` poison 時は panic せず
    /// `MetalError::ContextCacheUnavailable` を返す。
    #[test]
    fn get_or_build_reports_typed_error_on_poisoned_mutex() {
        let cache = fresh_cache::<u32>();
        let cache = std::panic::AssertUnwindSafe(&cache);
        let _ = std::panic::catch_unwind(|| {
            let _guard = cache.0.lock().expect("lock before poisoning");
            panic!("intentionally poison the mutex for this test");
        });

        let err = get_or_build(cache.0, || Ok(1)).unwrap_err();
        assert!(matches!(err, MetalError::ContextCacheUnavailable { .. }));
    }

    /// [`cached_context`] を 2 回呼ぶと `Arc::ptr_eq` で同一インスタンスが
    /// 返る（実 Metal デバイスが必要なため Mac 実機でのみ実行される。
    /// `tests/device_smoke.rs` と同じ前提）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn cached_context_returns_same_instance_across_calls() {
        let first = cached_context().expect("Metal context available on test host");
        let second = cached_context().expect("cache hit");
        assert!(
            Arc::ptr_eq(&first, &second),
            "2 回目の呼び出しは同一 Arc<MetalContext> を返すはず"
        );
    }

    /// [`cached_gemm`] も同様に同一インスタンスを返す。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn cached_gemm_returns_same_instance_across_calls() {
        let ctx = cached_context().expect("Metal context available on test host");
        let first = cached_gemm(&ctx).expect("gemm suite builds");
        let second = cached_gemm(&ctx).expect("cache hit");
        assert!(
            Arc::ptr_eq(&first, &second),
            "2 回目の呼び出しは同一 Arc<MetalGemm> を返すはず"
        );
    }
}
