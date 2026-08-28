//! `Mutex<Option<Arc<T>>>` に対する「ヒットなら clone・ミスなら構築して
//! 登録する」汎用キャッシュロジック（イシュー #930 codex-review 対応）。
//!
//! `context_cache.rs`（`cfg(target_os = "macos")` 限定・イシュー #930）の
//! `cached_context`／`cached_gemm` 等が使うコア判定ロジックをここへ切り出す。
//! 本モジュールは `objc2` 系 FFI・`MetalError` 等 macOS 固有の具体型に
//! 一切触れない純粋ロジックのみで構成するため `cfg(target_os = "macos")`
//! を付けず、`pad.rs`／`tile.rs`／`row_kernel.rs` と同じ設計判断で Linux
//! （CI・本実装環境）の `cargo test -p fandhe-ai-backend-metal` でも
//! 単体テストが回るようにしてある。
//!
//! `context_cache.rs` 側は Metal 固有の poison エラー変換
//! （`MetalError::ContextCacheUnavailable` への変換）のみをクロージャで
//! 注入し、それ以外のロック区間分割・登録判定ロジックは本モジュールへ
//! 完全に委譲する（ロジックの二重管理を避ける）。
//!
//! 本番からの唯一の呼び出し元 `context_cache.rs` は `cfg(target_os =
//! "macos")` 限定（`lib.rs`）のため、非 macOS ビルド（Linux 単体ビルド・
//! `cargo build`／`cargo clippy` の非テストパス）では本モジュールの関数が
//! 「クレート内から到達不能」と判定され dead_code lint が誤検知する
//! （`row_kernel.rs` 冒頭コメントと同じ状況・同じ対処方針）。`pub` へ広げず
//! `cfg_attr` で対象を非 macOS ビルドに限定して抑制する。
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use std::sync::{Arc, Mutex, MutexGuard};

/// `Mutex` guard 取得の共通ヘルパー。poison した場合は `on_poison` で
/// 呼び出し元固有のエラー型 `E` へ変換する（呼び出し元は panic させず
/// 型付きエラーとして伝播できる。`.claude/rules/coding-rust.md`
/// 「本番経路で unwrap/expect を使わない」）。
pub(crate) fn lock_cache<T, E>(
    mutex: &Mutex<Option<Arc<T>>>,
    on_poison: impl FnOnce(String) -> E,
) -> Result<MutexGuard<'_, Option<Arc<T>>>, E> {
    mutex
        .lock()
        .map_err(|e| on_poison(format!("context cache mutex poisoned: {e}")))
}

/// 単一エントリの `Mutex<Option<Arc<T>>>` に対する
/// 「ヒットなら clone・ミスなら `build` で構築して登録」の共通ロジック。
///
/// ロック区間を 2 段に分ける（先に読み取り専用でヒット判定、ミス時のみ
/// `build` をロック外で実行してから再度ロックして登録）ことで、コストの
/// 高い `build`（MSL コンパイル等を含みうる）をプロセス全体の `Mutex`
/// 保持中に実行しない。2 スレッドが同時にミスした場合は両方が `build` を
/// 実行しうるが、登録は `get_or_insert_with` で先着 1 件のみが採用され、
/// 後着側の構築結果は呼び出し元に返した後 drop されるだけであり数値的な
/// 誤りにはつながらない（許容する冗長構築。イシュー #930 実装計画 §3.1）。
///
/// `build` の失敗（`Err`）はキャッシュへ格納せず、そのまま呼び出し元へ
/// 伝播する（fail-fast 契約。`context_cache.rs` モジュール冒頭参照）。
pub(crate) fn get_or_build<T, E>(
    cache: &Mutex<Option<Arc<T>>>,
    on_poison: impl Fn(String) -> E,
    build: impl FnOnce() -> Result<T, E>,
) -> Result<Arc<T>, E> {
    {
        let guard = lock_cache(cache, &on_poison)?;
        if let Some(existing) = guard.as_ref() {
            return Ok(Arc::clone(existing));
        }
    }
    let built = Arc::new(build()?);
    let mut guard = lock_cache(cache, &on_poison)?;
    Ok(Arc::clone(guard.get_or_insert(built)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト専用の poison エラー型。`MetalError` 等 macOS 固有型を
    /// 要求せず、本モジュールが macOS 非依存であることをテスト自体でも
    /// 保証する。
    #[derive(Debug, PartialEq, Eq)]
    struct TestCacheError(String);

    fn on_poison(detail: String) -> TestCacheError {
        TestCacheError(detail)
    }

    fn fresh_cache<T>() -> Mutex<Option<Arc<T>>> {
        Mutex::new(None)
    }

    #[test]
    fn get_or_build_constructs_once_and_caches_hit() {
        let cache = fresh_cache::<u32>();
        let calls = std::sync::atomic::AtomicU32::new(0);

        let first = get_or_build(&cache, on_poison, || {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok::<_, TestCacheError>(42)
        })
        .expect("build succeeds");
        let second = get_or_build(&cache, on_poison, || {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok::<_, TestCacheError>(43)
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

    /// fail-fast 契約: `build` が失敗してもキャッシュへ格納されず、
    /// 次回呼び出しで再度 `build` が呼ばれる。
    #[test]
    fn get_or_build_does_not_cache_errors() {
        let cache = fresh_cache::<u32>();
        let calls = std::sync::atomic::AtomicU32::new(0);

        let build = || {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err(TestCacheError("simulated failure".into()))
        };

        assert!(get_or_build(&cache, on_poison, build).is_err());
        assert!(get_or_build(&cache, on_poison, build).is_err());
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "エラーはキャッシュされないため build は毎回呼ばれるはず"
        );

        // デバイスが後から利用可能になった環境を模す: 3 回目は成功する。
        let third = get_or_build(&cache, on_poison, || Ok::<_, TestCacheError>(7))
            .expect("recovers once build succeeds");
        assert_eq!(*third, 7);
    }

    /// `Mutex` poison 時は panic せず `on_poison` で変換したエラーを返す。
    #[test]
    fn get_or_build_reports_typed_error_on_poisoned_mutex() {
        let cache = fresh_cache::<u32>();
        let cache = std::panic::AssertUnwindSafe(&cache);
        let _ = std::panic::catch_unwind(|| {
            let _guard = cache.0.lock().expect("lock before poisoning");
            panic!("intentionally poison the mutex for this test");
        });

        let err = get_or_build(cache.0, on_poison, || Ok::<_, TestCacheError>(1)).unwrap_err();
        assert!(matches!(err, TestCacheError(_)));
    }
}
