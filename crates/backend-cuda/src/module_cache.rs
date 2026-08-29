//! プロセス内 LRU カーネルモジュールキャッシュ（Phase C-4・イシュー #511）。
//!
//! 親イシュー #503（Phase C: CUDA JIT shape 特化・コンパイルキャッシュ・
//! 静的タイル選定）の最終タスク。C-1〜C-3・C-5（[`crate::nvrtc`] の
//! `CudaKernelCacheKey`／ディスクキャッシュ `store_cache_entry`／
//! `load_cache_entry`）は実装済みだが GEMM 経路へ未結線であり、繰り返し
//! 形状での NVRTC 再コンパイル・`load_module` 再ロードのコストが残って
//! いた（`kernels_mma.rs::RenderedMmaKernel::compile` からの結線で解消、
//! さらにイシュー #1024 で `gemm.rs::CudaGemm::new`（f32 本番経路の 9
//! カーネル）へも結線した）。
//!
//! [`load_function_cached`] が 3 段フォールバック（プロセス内 LRU →
//! ディスクキャッシュ照合 → NVRTC 直コンパイル）の単一実装であり、
//! `kernels_mma.rs::RenderedMmaKernel::compile` と `gemm.rs::CudaGemm::new`
//! の双方がこのヘルパーを呼ぶ（イシュー #1024。ロジックの二重管理を
//! 避けるための共通化）。
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

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::nvrtc::{CudaKernelCacheKey, CudaKernelDescriptor};

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
/// 唯一の呼び出し元 [`KernelModuleCache::global`] は [`load_function_cached`]
/// （`kernels_mma.rs::RenderedMmaKernel::compile` と `gemm.rs::CudaGemm::new`
/// の双方が既定ビルドで到達する共通ヘルパー。イシュー #1024）経由で呼ばれる。
/// テストは実環境変数への依存を避けるため注入可能な
/// [`resolve_module_cache_capacity`] を直接呼ぶ。
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
/// 呼び出し元は [`load_function_cached`] 経由の
/// `kernels_mma.rs::RenderedMmaKernel::compile`（`internal-diagnostics`
/// feature 限定）と `gemm.rs::CudaGemm::new`（f32 本番経路。イシュー
/// #1024 で結線）の 2 経路。後者は既定ビルドでも到達するため、
/// `#[allow(dead_code)]` は不要（結線前は前者のみだったため付与して
/// いた）。
pub(crate) struct KernelModuleCache {
    inner: Mutex<LruCache<(usize, CudaKernelCacheKey), Arc<CudaModule>>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

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
        // `Mutex` guard をブロックへ閉じ込め、evict／置換で外れた旧
        // `Arc<CudaModule>`（`evicted`）をブロック外へ持ち出してから
        // drop する（Bugbot 指摘〈Evict drops under cache lock〉対応）。
        // ローカル変数は宣言の逆順で drop されるため、旧実装のように
        // `guard` を先に・`_evicted` を後に宣言すると `_evicted` が
        // `guard` より先に drop され、evict された `Arc<CudaModule>` の
        // `cuModuleUnload`（本モジュール冒頭ドキュメンテーションコメント
        // 「所有モデル」参照）がこのプロセス全体の `Mutex` を保持した
        // ままドライバへ発行されてしまう。並行する他 `get`／`insert`
        // 呼び出し元がその unload の完了を待つ間ロックが占有され続け、
        // モジュールキャッシュ全体が不必要に足止めされる。`guard` を
        // ブロックの終端で drop してからロック外で `evicted` を drop
        // することで、unload はロック解放後に実行される。
        let evicted = {
            let mut guard = self
                .inner
                .lock()
                .map_err(|e| CudaError::ModuleCacheUnavailable {
                    detail: format!("module cache mutex poisoned: {e}"),
                })?;
            guard.insert((ctx_id, key), module)
        };
        drop(evicted);
        Ok(())
    }

    /// ヒット件数（実機テストでの再利用検証・C-12〈#534〉計測用）。
    ///
    /// 呼び出し元は `lib.rs::diagnostics::module_cache_hit_count`
    /// （`internal-diagnostics` feature 限定）と
    /// `module_cache_wiring_tests.rs`（`#[cfg(test)]`）のみのため、
    /// 既定ビルド（feature なし・非 test）では未参照のまま dead-code
    /// 解析が誤検知する。
    #[allow(dead_code)]
    pub(crate) fn hit_count(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// ミス件数。`#[allow(dead_code)]` の理由は [`Self::hit_count`] と同じ。
    #[allow(dead_code)]
    pub(crate) fn miss_count(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
}

/// `descriptor`／`source`（最終レンダー済みカーネルソース全体）を鍵に、
/// 3 段フォールバック（プロセス内 LRU → ディスクキャッシュ照合 → NVRTC
/// 直コンパイル）で `func_name` の [`CudaFunction`] を取得する共通ヘルパー
/// （イシュー #1024）。
///
/// 元々は `kernels_mma.rs::RenderedMmaKernel::compile` に個別実装されて
/// いた 3 段ロジック（イシュー #511・C-4）をそのまま抽出したもの。
/// `gemm.rs::CudaGemm::new`（f32 GEMM 本番経路の 9 カーネル）も本関数を
/// 呼ぶことで、2 箇所が独立にフォールバック段数・縮退方針を実装して
/// 乖離するのを防ぐ（単一の真実源化）。
///
/// # 3 段フォールバック
///
/// 1. **プロセス内 LRU**（[`KernelModuleCache`]）: 同一 `CudaContext`・
///    同一キャッシュキーでロード済みの `Arc<CudaModule>` があれば
///    `cuModuleGetFunction`（軽量）のみで済ませる。
/// 2. **ディスクキャッシュ**（[`crate::nvrtc::load_cache_entry`]。
///    C-3・#509）: ソース全文のバイト単位照合込みでヒット判定のみ
///    行う。**ヒットしてもディスク上の `kernel.ptx` は実行入力として
///    使わない**（イシュー #511 PR #703 codex-review P0 対応の踏襲。
///    理由は `kernels_mma.rs::RenderedMmaKernel::compile` のドキュメン
///    テーションコメント「2 段目」節を正とし、ここでは重複記載しない）。
///    ヒット／ミスいずれの場合も 3 段目へ進む。
/// 3. **NVRTC 直コンパイル**（2 段目のヒット／ミスを問わず必ず実行）:
///    `compile_ptx` 実行後、ディスクに当該キーのエントリがまだ
///    なければ [`crate::nvrtc::store_cache_entry`] でディスクへ保存する。
///
/// いずれの段でロードした `Arc<CudaModule>` も、最終的に
/// [`KernelModuleCache::insert`] へ登録し、次回以降のプロセス内再利用に
/// 備える。
///
/// # 縮退方針（fail-safe）
///
/// プロセス内 LRU（容量設定不正・`Mutex` poison 等）・ディスクキャッシュ
/// （`workspace_root` 解決不能・fs I/O 失敗）いずれの失敗もコンパイル
/// 失敗にせず、直後の段（最終的には NVRTC 直コンパイル）へ静かにフォール
/// バックする（`module_cache.rs` 冒頭ドキュメンテーションコメント
/// 「縮退方針」節と同じ判断）。
pub(crate) fn load_function_cached(
    device: &CudaDevice,
    descriptor: CudaKernelDescriptor,
    source: &str,
    func_name: &str,
) -> Result<CudaFunction, CudaError> {
    let ctx = device.context();
    let compile_flags = vec![format!("--gpu-architecture={}", device.arch())];
    let key =
        CudaKernelCacheKey::from_device(descriptor, device, compile_flags, source.to_owned())?;

    // 1 段目: プロセス内 LRU。キャッシュ自体が利用不能（容量設定不正・
    // poison）でもフォールバックし続ける（縮退方針）。
    let module_cache = KernelModuleCache::global().ok();
    if let Some(cache) = module_cache
        && let Ok(Some(module)) = cache.get(ctx, &key)
    {
        return Ok(module.load_function(func_name)?);
    }

    // ディスクキャッシュの読み書きに使う `workspace_root`。解決失敗も
    // 縮退運転（ディスクキャッシュなし）へ倒す。
    let workspace_root = crate::nvrtc::runtime_workspace_root().ok();

    // 2 段目: ディスクキャッシュ（ソース全文のバイト照合込み。実行入力
    // としては使わない。上記ドキュメンテーションコメント参照）。
    let disk_hit = workspace_root.as_ref().and_then(|root| {
        crate::nvrtc::load_cache_entry(root, &key, source)
            .ok()
            .flatten()
    });

    // 3 段目: NVRTC 直コンパイル。hit／miss いずれの場合もこのプロセス内
    // で NVRTC を実行して得た PTX のみをロードする。
    let ptx = crate::nvrtc::compile_ptx(source, device.arch())?;
    if disk_hit.is_none()
        && let Some(root) = workspace_root.as_ref()
    {
        // 保存失敗はコンパイル結果自体には影響しない（縮退方針）ため
        // 戻り値は無視する。
        let _ = crate::nvrtc::store_cache_entry(root, &key, source, &ptx.to_src());
    }
    let module = ctx.load_module(ptx)?;

    // ロード済みモジュールをプロセス内 LRU へ登録する（挿入失敗＝ poison
    // も縮退方針でコンパイル結果自体は返す）。
    if let Some(cache) = module_cache {
        let _ = cache.insert(ctx, key, Arc::clone(&module));
    }

    Ok(module.load_function(func_name)?)
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
