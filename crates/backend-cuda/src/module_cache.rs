//! プロセス内 LRU カーネルモジュールキャッシュ（Phase C-4・イシュー #511）。
//!
//! 親イシュー #503（Phase C: CUDA JIT shape 特化・コンパイルキャッシュ・
//! 静的タイル選定）の最終タスク。C-1〜C-3・C-5（[`crate::nvrtc`] の
//! `CudaKernelCacheKey`／ディスクキャッシュ `store_cache_entry`／
//! `load_cache_entry`）は実装済みだが GEMM 経路へ未結線であり、繰り返し
//! 形状での NVRTC 再コンパイル・`load_module` 再ロードのコストが残って
//! いた（`kernels_mma.rs::RenderedMmaKernel::compile` からの結線で解消）。
//!
//! # 所有モデル（cudarc 0.19.8）
//!
//! `CudaContext::load_module` は `Arc<CudaModule>` を返し（`Drop` で
//! `cuModuleUnload`）、`CudaModule::load_function` が返す `CudaFunction` は
//! ロード元 `Arc<CudaModule>` を内部に clone 保持する（cudarc-0.19.8
//! `src/driver/safe/core.rs:2152-2226`）。したがって本キャッシュが
//! エビクション時に保持していた `Arc<CudaModule>` を drop しても、未解放の
//! `CudaFunction` が別途生存していれば実際のアンロードはその最後の参照が
//! 落ちるまで自然に遅延するだけで、ダングリングにはならない
//! （`KernelModuleCache` ドキュメンテーションコメント参照）。
//!
//! # 縮退方針
//!
//! 本キャッシュは純粋な最適化であり、ロック取得に失敗した場合
//! （[`CudaError::ModuleCacheUnavailable`]。`Mutex` poison。本モジュールの
//! 臨界区間自体は `unwrap`/`expect` を持たないため通常到達しない）でも
//! 呼び出し元（`kernels_mma.rs::RenderedMmaKernel::compile`）はキャッシュ
//! なしの直接コンパイルへフォールバックすればよい。数値正しさはディスク
//! キャッシュ側のソース全文照合（`nvrtc.rs::load_cache_entry`）・NVRTC
//! 直コンパイルのいずれの経路でも独立に保たれるため、本キャッシュの可用性
//! 低下が誤った PTX の実行につながることはない。

use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaModule};

use crate::error::CudaError;
use crate::nvrtc::CudaKernelCacheKey;

/// 容量上限つき LRU の既定容量（未設定時）。
///
/// カーネル種別（現状 `mma_f16` のみ）×特化形状数（`CompiledDims` の
/// 組合せ×実運用で登場する shape 数）の実用上界に対する余裕値として
/// 32 を選んだ。DeepGEMM 参照実装（`csrc/jit/cache.hpp`）は容量無制限
/// （TODO コメントあり）だが、本クレートは GPU リソース（ロード済み
/// モジュールが保持する device メモリ・コンテキスト参照）の無制限滞留を
/// 避けるため容量上限つき LRU 化する（実装計画 §1）。
const DEFAULT_CAPACITY: NonZeroUsize = match NonZeroUsize::new(32) {
    Some(v) => v,
    None => panic!("DEFAULT_CAPACITY: 32 must be non-zero"),
};

/// 容量上限の天井（過大設定による資源枯渇 DoS 耐性。実装計画 §7）。
const MAX_CAPACITY: usize = 1024;

/// 容量を上書きする環境変数名（`RUST_AI_CUDA_CACHE_DIR` と同系の命名。
/// `nvrtc.rs` のディスクキャッシュ環境変数と対称的に扱う）。
const CAPACITY_ENV_VAR: &str = "RUST_AI_CUDA_MODULE_CACHE_CAPACITY";

/// tick 方式のジェネリック LRU（std のみ・依存追加なし）。
///
/// `HashMap<K, (V, tick)>` を単調増加 `u64` tick で使用順管理する。
/// ヒットで tick 更新、満杯時は最小 tick のエントリを線形スキャンして
/// evict する（O(capacity)。既定容量が小さいため十分・決定的）。`lru`
/// クレート等は許容依存 8 区分（`.claude/rules/deps-policy.md`）外のため
/// 自作する。
///
/// evict／同一キー再挿入で外れた値は [`Self::insert`] の戻り値として
/// 呼び出し元へ返す（呼び出し元がその場で drop することで、解放
/// タイミングを構造で保証する。本モジュール冒頭ドキュメンテーション
/// コメント「所有モデル」参照）。
///
/// crate 外へは公開しない（`module_cache` 自体が非公開 `mod`。`lib.rs`
/// 参照）。GPU 非依存のためユニットテストは本ファイル末尾の
/// `#[cfg(test)]` で行う。
struct LruCache<K, V> {
    capacity: NonZeroUsize,
    entries: HashMap<K, (V, u64)>,
    tick: u64,
}

impl<K: Eq + Hash + Clone, V> LruCache<K, V> {
    fn new(capacity: NonZeroUsize) -> Self {
        Self {
            capacity,
            entries: HashMap::new(),
            tick: 0,
        }
    }

    /// 単調増加する次の tick を発行する。`get`／`insert` の双方が呼び、
    /// 「直近にアクセスしたエントリほど大きい tick を持つ」不変条件を
    /// 維持する。
    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// キーを検索し、ヒットしたら使用順（tick）を更新して値への参照を
    /// 返す。ミスなら `None`（呼び出し元のヒット/ミスカウンタ更新は
    /// [`KernelModuleCache`] 側の責務）。
    fn get(&mut self, key: &K) -> Option<&V> {
        let tick = self.next_tick();
        match self.entries.get_mut(key) {
            Some(entry) => {
                entry.1 = tick;
                Some(&entry.0)
            }
            None => None,
        }
    }

    /// `key`/`value` を挿入する。
    ///
    /// - 既存キーへの再挿入: 旧値を `Some` で返す（容量チェックは不要。
    ///   要素数が変わらないため）。
    /// - 新規キーで容量超過: 挿入後に最小 tick のエントリを 1 件 evict し
    ///   その値を `Some` で返す（挿入直後のエントリは最大 tick のため
    ///   evict 対象になり得ない）。
    /// - 新規キーで容量内: `None`。
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        let tick = self.next_tick();
        if let Some((old_value, _)) = self.entries.insert(key, (value, tick)) {
            return Some(old_value);
        }
        if self.entries.len() > self.capacity.get() {
            let evict_key = self
                .entries
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| k.clone());
            if let Some(evict_key) = evict_key
                && let Some((evicted_value, _)) = self.entries.remove(&evict_key)
            {
                return Some(evicted_value);
            }
        }
        None
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// 環境変数からプロセス内 LRU モジュールキャッシュの容量を解決する純関数
/// （`nvrtc.rs::resolve_cache_root` と同じ「注入で決定化」パターン。実
/// 環境変数は [`module_cache_capacity`] のみが読む）。
///
/// - `raw` が `None`（未設定）→ 既定値 [`DEFAULT_CAPACITY`]
/// - 10 進整数文字列（`1..=1024`）のみ許容。空文字列・非数値・`0`・
///   範囲外・非 UTF-8 は `CudaError::InvalidModuleCacheCapacity` で
///   fail-closed に拒否する（A03 対策。`resolve_cache_root` が不正値を
///   黙殺せず拒否する既存方針と整合。`.claude/rules/security.md`）
///
/// 容量は呼び出し元（[`KernelModuleCache::global`]）が `OnceLock` 経由で
/// プロセス内 1 回だけ解決し以降固定する契約であり、実行途中の環境変数
/// 変更は反映されない（`KernelModuleCache::global` ドキュメンテーション
/// コメント参照）。
fn resolve_module_cache_capacity(raw: Option<&std::ffi::OsStr>) -> Result<NonZeroUsize, CudaError> {
    let raw = match raw {
        None => return Ok(DEFAULT_CAPACITY),
        Some(raw) => raw,
    };
    let text = raw
        .to_str()
        .ok_or_else(|| CudaError::InvalidModuleCacheCapacity {
            detail: format!("{CAPACITY_ENV_VAR} must be valid UTF-8, got {raw:?}"),
        })?;
    if text.is_empty() {
        return Err(CudaError::InvalidModuleCacheCapacity {
            detail: format!("{CAPACITY_ENV_VAR} is set but empty"),
        });
    }
    let value: usize = text
        .parse()
        .map_err(|e| CudaError::InvalidModuleCacheCapacity {
            detail: format!("{CAPACITY_ENV_VAR} must be a decimal integer, got {text:?} ({e})"),
        })?;
    if value == 0 {
        return Err(CudaError::InvalidModuleCacheCapacity {
            detail: format!("{CAPACITY_ENV_VAR} must be non-zero, got {value}"),
        });
    }
    if value > MAX_CAPACITY {
        return Err(CudaError::InvalidModuleCacheCapacity {
            detail: format!("{CAPACITY_ENV_VAR} must be <= {MAX_CAPACITY}, got {value}"),
        });
    }
    // 上の `value == 0` 検査により `NonZeroUsize::new` は必ず `Some` を
    // 返す（構築失敗は型として到達不能）。
    NonZeroUsize::new(value).ok_or_else(|| CudaError::InvalidModuleCacheCapacity {
        detail: format!("{CAPACITY_ENV_VAR} must be non-zero, got {value}"),
    })
}

/// [`resolve_module_cache_capacity`] の crate 内ラッパー。実プロセス
/// 環境変数 `RUST_AI_CUDA_MODULE_CACHE_CAPACITY` を読んで委譲する。
///
/// 唯一の呼び出し元 [`KernelModuleCache::global`] が
/// `kernels_mma.rs::RenderedMmaKernel::compile`（`internal-diagnostics`
/// feature ゲート経由でのみ到達。`kernels_mma.rs` 該当ドキュメンテーション
/// コメント参照）経由でのみ呼ばれるため、既定ビルドでは crate 内呼び出し
/// 元が実質存在せず dead-code 解析が誤検知する。テストは実環境変数への
/// 依存を避けるため注入可能な [`resolve_module_cache_capacity`] を直接
/// 呼ぶ（本関数と同じ理由）。
#[allow(dead_code)]
fn module_cache_capacity() -> Result<NonZeroUsize, CudaError> {
    resolve_module_cache_capacity(std::env::var_os(CAPACITY_ENV_VAR).as_deref())
}

/// ロード済み CUDA モジュールのプロセス内 LRU 再利用キャッシュ（CUDA
/// 特化ラッパー。イシュー #511・Phase C-4）。
///
/// キーは `(ctx_id, CudaKernelCacheKey)`。`ctx_id` は要求元
/// `Arc<CudaContext>` のポインタ識別（`Arc::as_ptr`）であり、別
/// `CudaContext` にロードしたモジュールを誤って共有しないための境界
/// （`CudaFunction`／`CudaModule` はロード元 context に紐付く。cudarc の
/// 型では context 同一性が保証されないため、本キャッシュのキーレベルで
/// 遮断する。`gemm_auto.rs::SpecializedMmaKernelHandle` が `stream`
/// 〈延いては context〉をハンドル内に固定して起動時の外部入力から外す
/// のと同型の判断）。
///
/// **ABA 耐性**: `ctx_id` はポインタアドレスのため、`CudaContext` が
/// 解放され同一アドレスへ別 context が再割当されると理論上 ABA が起こり
/// うる。しかし本キャッシュが保持する `Arc<CudaModule>` は内部に
/// `Arc<CudaContext>`（cudarc `CudaModule::ctx` フィールド）を強参照して
/// 保持するため、「当該 ctx のエントリが本キャッシュに 1 つでも残る限り
/// 元の `CudaContext` は解放されない」。したがって同一アドレスへの
/// 再割当は当該 ctx の全エントリが本キャッシュから evict された後にしか
/// 起こり得ず、構造的に ABA は発生しない。
///
/// hit/miss カウンタ（[`Self::hit_count`]／[`Self::miss_count`]）を持ち、
/// 実機テストでの再利用検証・将来の C-12（#534）性能計測の観測点とする。
///
/// 唯一の crate 内呼び出し元 `kernels_mma.rs::RenderedMmaKernel::compile`
/// が `internal-diagnostics` feature（既定 off）ゲート経由でのみ到達する
/// （同ファイルの `#[allow(dead_code)]` 群と同じ理由）ため、既定ビルドでは
/// 本 struct・以下の全メソッドが未参照のまま dead-code 解析が誤検知する。
#[allow(dead_code)]
pub(crate) struct KernelModuleCache {
    inner: Mutex<LruCache<(usize, CudaKernelCacheKey), Arc<CudaModule>>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

// `#[allow(dead_code)]` の理由は `KernelModuleCache` 本体のドキュメン
// テーションコメントと同じ（既定ビルドでは `internal-diagnostics`
// feature 経由でのみ到達するため）。impl 全体を覆うことで各メソッド
// 個別への重複付与を避ける。
#[allow(dead_code)]
impl KernelModuleCache {
    fn new(capacity: NonZeroUsize) -> Self {
        Self {
            inner: Mutex::new(LruCache::new(capacity)),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// プロセスワイドの唯一のインスタンスを返す（`static` +
    /// `OnceLock`）。
    ///
    /// 容量は初回呼び出し時に [`module_cache_capacity`]（環境変数
    /// `RUST_AI_CUDA_MODULE_CACHE_CAPACITY`）で 1 回だけ解決しプロセス内で
    /// 一貫させる契約とし、実行途中の環境変数変更は反映されない（実装
    /// 計画 §3.3。テスト容易性のため容量注入が必要な場合は
    /// [`KernelModuleCache::new`]（本関数を経由しない直接構築）を使う。
    /// GPU 不要のユニットテストは本ファイル末尾の `#[cfg(test)]` を参照）。
    ///
    /// 容量解決自体が失敗した場合（不正な環境変数値）は `Err` を返し、
    /// 呼び出し元（`kernels_mma.rs::RenderedMmaKernel::compile`）は
    /// キャッシュなしの直接コンパイルへ縮退する。
    pub(crate) fn global() -> Result<&'static KernelModuleCache, CudaError> {
        static CACHE: OnceLock<KernelModuleCache> = OnceLock::new();
        if let Some(cache) = CACHE.get() {
            return Ok(cache);
        }
        let capacity = module_cache_capacity()?;
        Ok(CACHE.get_or_init(|| KernelModuleCache::new(capacity)))
    }

    /// `ctx`（要求元 `Arc<CudaContext>`）・`key` に対応するロード済み
    /// モジュールを検索する。ヒットなら `Arc<CudaModule>` を clone して
    /// 返す（LRU の使用順も更新）。
    ///
    /// `Mutex` poison 時は `CudaError::ModuleCacheUnavailable` を返す
    /// （panic 経路を持たない。`.claude/rules/coding-rust.md`）。
    pub(crate) fn get(
        &self,
        ctx: &Arc<CudaContext>,
        key: &CudaKernelCacheKey,
    ) -> Result<Option<Arc<CudaModule>>, CudaError> {
        let ctx_id = Arc::as_ptr(ctx) as usize;
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| CudaError::ModuleCacheUnavailable {
                detail: format!("module cache mutex poisoned: {e}"),
            })?;
        let hit = guard.get(&(ctx_id, key.clone())).cloned();
        if hit.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        Ok(hit)
    }

    /// `ctx`・`key` に対応するロード済み `module` を登録する。容量超過で
    /// evict されたエントリの旧 `Arc<CudaModule>` はこのメソッド呼び出し内
    /// でそのまま drop される（未解放の `CudaFunction` が別途生存していな
    /// い限り即時 `cuModuleUnload`。本モジュール冒頭ドキュメンテーション
    /// コメント「所有モデル」参照）。
    ///
    /// `Mutex` poison 時は `CudaError::ModuleCacheUnavailable` を返す。
    /// 呼び出し元はこの場合キャッシュへの登録をスキップしてよい
    /// （登録できなくても `module` 自体は呼び出し元が既に保持しており、
    /// 数値経路・当該呼び出し自体には影響しない。単に次回以降のヒットを
    /// 逃すだけ）。
    pub(crate) fn insert(
        &self,
        ctx: &Arc<CudaContext>,
        key: CudaKernelCacheKey,
        module: Arc<CudaModule>,
    ) -> Result<(), CudaError> {
        let ctx_id = Arc::as_ptr(ctx) as usize;
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| CudaError::ModuleCacheUnavailable {
                detail: format!("module cache mutex poisoned: {e}"),
            })?;
        // 戻り値（evict／置換で外れた旧 `Arc<CudaModule>`）はこの文の終端で
        // drop される。呼び出し元へは返さない（本メソッドの責務は登録の
        // みであり、解放タイミングの構造的保証は `LruCache::insert` の
        // ドキュメンテーションコメント参照）。
        let _evicted = guard.insert((ctx_id, key), module);
        Ok(())
    }

    /// ヒット件数（実機テストでの再利用検証・C-12〈#534〉計測用）。
    pub(crate) fn hit_count(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// ミス件数。
    pub(crate) fn miss_count(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // LruCache（GPU 非依存）
    // ------------------------------------------------------------------

    fn cap(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("test capacity must be non-zero")
    }

    #[test]
    fn insert_within_capacity_does_not_evict() {
        let mut lru: LruCache<&str, i32> = LruCache::new(cap(2));
        assert_eq!(lru.insert("a", 1), None);
        assert_eq!(lru.insert("b", 2), None);
        assert_eq!(lru.len(), 2);
    }

    #[test]
    fn insert_over_capacity_evicts_least_recently_used() {
        let mut lru: LruCache<&str, i32> = LruCache::new(cap(2));
        lru.insert("a", 1);
        lru.insert("b", 2);
        // "a" にアクセスして使用順を更新し、"b" を最古にする。
        assert_eq!(lru.get(&"a"), Some(&1));
        let evicted = lru.insert("c", 3);
        assert_eq!(evicted, Some(2)); // "b" が evict される
        assert_eq!(lru.len(), 2);
        assert_eq!(lru.get(&"a"), Some(&1));
        assert_eq!(lru.get(&"c"), Some(&3));
    }

    #[test]
    fn insert_same_key_replaces_and_returns_old_value_without_evicting() {
        let mut lru: LruCache<&str, i32> = LruCache::new(cap(1));
        assert_eq!(lru.insert("a", 1), None);
        assert_eq!(lru.insert("a", 2), Some(1));
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.get(&"a"), Some(&2));
    }

    #[test]
    fn get_miss_returns_none() {
        let mut lru: LruCache<&str, i32> = LruCache::new(cap(2));
        assert_eq!(lru.get(&"missing"), None);
    }

    #[test]
    fn capacity_one_evicts_previous_entry_on_new_key() {
        let mut lru: LruCache<&str, i32> = LruCache::new(cap(1));
        lru.insert("a", 1);
        let evicted = lru.insert("b", 2);
        assert_eq!(evicted, Some(1));
        assert_eq!(lru.get(&"a"), None);
        assert_eq!(lru.get(&"b"), Some(&2));
    }

    /// リーク検査（受入基準）: evict されたエントリの値が実際に drop
    /// されることを `Weak::upgrade() == None` で構造的に確認する
    /// （`Arc<CudaModule>` の場合は drop = `cuModuleUnload` 実行に相当。
    /// 実機なしで検証可能な形へ一般化するため `Arc<()>` で代用する）。
    #[test]
    fn evicted_value_is_actually_dropped() {
        let mut lru: LruCache<&str, Arc<()>> = LruCache::new(cap(1));
        let value_a = Arc::new(());
        let weak_a = Arc::downgrade(&value_a);
        lru.insert("a", value_a);

        let evicted = lru.insert("b", Arc::new(()));
        assert!(evicted.is_some());
        drop(evicted); // `LruCache::insert` の戻り値を呼び出し元が drop する契約を模擬する

        assert!(
            weak_a.upgrade().is_none(),
            "evicted value must be dropped once the caller drops the returned Option"
        );
    }

    // ------------------------------------------------------------------
    // resolve_module_cache_capacity（`resolve_cache_root` と同型の網羅）
    // ------------------------------------------------------------------

    #[test]
    fn resolve_capacity_unset_uses_default() {
        assert_eq!(
            resolve_module_cache_capacity(None).unwrap(),
            DEFAULT_CAPACITY
        );
    }

    #[test]
    fn resolve_capacity_valid_value() {
        let raw = std::ffi::OsString::from("64");
        assert_eq!(resolve_module_cache_capacity(Some(&raw)).unwrap().get(), 64);
    }

    #[test]
    fn resolve_capacity_empty_is_rejected() {
        let raw = std::ffi::OsString::from("");
        assert!(matches!(
            resolve_module_cache_capacity(Some(&raw)),
            Err(CudaError::InvalidModuleCacheCapacity { .. })
        ));
    }

    #[test]
    fn resolve_capacity_non_numeric_is_rejected() {
        let raw = std::ffi::OsString::from("not-a-number");
        assert!(matches!(
            resolve_module_cache_capacity(Some(&raw)),
            Err(CudaError::InvalidModuleCacheCapacity { .. })
        ));
    }

    #[test]
    fn resolve_capacity_zero_is_rejected() {
        let raw = std::ffi::OsString::from("0");
        assert!(matches!(
            resolve_module_cache_capacity(Some(&raw)),
            Err(CudaError::InvalidModuleCacheCapacity { .. })
        ));
    }

    #[test]
    fn resolve_capacity_out_of_range_is_rejected() {
        let raw = std::ffi::OsString::from("1025");
        assert!(matches!(
            resolve_module_cache_capacity(Some(&raw)),
            Err(CudaError::InvalidModuleCacheCapacity { .. })
        ));
    }

    #[test]
    fn resolve_capacity_max_boundary_is_accepted() {
        let raw = std::ffi::OsString::from("1024");
        assert_eq!(
            resolve_module_cache_capacity(Some(&raw)).unwrap().get(),
            1024
        );
    }

    #[test]
    fn resolve_capacity_negative_is_rejected() {
        let raw = std::ffi::OsString::from("-1");
        assert!(matches!(
            resolve_module_cache_capacity(Some(&raw)),
            Err(CudaError::InvalidModuleCacheCapacity { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn resolve_capacity_non_utf8_is_rejected() {
        use std::os::unix::ffi::OsStrExt;
        let raw = std::ffi::OsStr::from_bytes(&[0xFF, 0xFE]).to_os_string();
        assert!(matches!(
            resolve_module_cache_capacity(Some(&raw)),
            Err(CudaError::InvalidModuleCacheCapacity { .. })
        ));
    }
}
