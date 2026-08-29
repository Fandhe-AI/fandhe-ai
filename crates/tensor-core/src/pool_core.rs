//! サイズクラス別・ハンドル型非依存のデバイスメモリプール本体
//! （イシュー #1021・#1020。設計は `docs/device-memory-pool-design.md`
//! §3.1〜3.6）。
//!
//! # 位置付け（既存 [`crate::pool`] との関係）
//!
//! [`crate::pool::PooledMemory`]（TASK-#201・REQ-14 14-3）は「バイトサイズ
//! 完全一致バケット」の opt-in デコレータであり、`MemoryOps` 実装をラップ
//! する形で既に存在する。本モジュールはそれとは**別モジュール**として
//! 共存し、サイズクラス丸め（jemalloc 型のオクターブ分割。§3.2）・
//! `capacity != 論理長` の分離・Metal の GPU 完了待ち返却
//! （`pending_return_bytes`）という新しい契約を持つ。既存 `PooledMemory`・
//! `PoolZeroFill` の公開 API は本モジュールの追加によって一切変更しない
//! （非破壊拡張。設計文書 §3.1「既存 `PooledMemory` との関係」）。
//!
//! # `tensor-core` に置く理由・置かないもの
//!
//! 本モジュールが公開するのは POD 型（[`SizeClassPoolConfig`]・
//! [`PoolStats`]）とハンドル型非依存の [`SizeClassPool<H>`] のみである。
//! `H` に一切 trait 境界を課さない（`Send` のみ。§3.5）ため、`tensor-core`
//! 自身は `H` が具体的に何であるか（`CudaSliceHandle`・
//! `crate::pool::PooledBufferHandle`（`pool.rs` 内部の非公開型）とは
//! 異なる新設の生ハンドル型。
//! `backend-cuda`／`backend-metal` それぞれの `pub(crate)` モジュール）を
//! 一切知らない。低水準 trait（`DeviceAllocator` 相当）・`BufferHandle`
//! を実装するアダプタ型はここには置かない（codex-review PR #1056 の
//! 2 度の P1 是正を経て確定した設計。§3.1 冒頭「レビュー履歴」参照）。
//!
//! # 命名の差異（イシュー #1021 実装確定・PR 本文に記録）
//!
//! 設計文書は新設定型を `PoolConfig` と例示しているが、[`crate::pool::
//! PoolConfig`]（`PooledMemory` 用。クレート root へ `pub use` 済み）と
//! 同名になり既存公開 API と衝突するため、本実装は
//! [`SizeClassPoolConfig`] を採用する（クレート root へは [`PoolStats`]
//! のみ再エクスポートし、本設定型は `pool_core::SizeClassPoolConfig` の
//! パス経由でのみ到達可能とする）。#1020（CUDA 実装）はこの命名へ揃える。

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::BackendError;

/// [`SizeClassPool`] のサイズクラス境界・総量上限の設定値（POD。
/// 設計文書 §3.2 の帯・丸め粒度に対応するフィールドを持つ）。
///
/// `Default` は設計文書が確定した既定値（`max_pool_bytes` 128 MiB・
/// 小帯上限 1 MiB・大帯粒度 2 MiB・巨大帯下限 64 MiB）を返す。閾値の
/// 見直しは #1010 の内訳実測後にユーザー承認を経て行う（設計文書 §3.2
/// 末尾）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeClassPoolConfig {
    /// アイドル保持（`cached_bytes + pending_return_bytes`）の総量上限。
    /// 超過時はグローバル LRU で古いエントリから破棄する（§3.4）。
    /// `0` はプール無効（全パススルー）を意味する（既存 `PoolConfig` と
    /// 同じ契約）。
    pub max_pool_bytes: u64,
    /// 小帯（オクターブ 4 段丸め）の上限バイト数。これ以下は §3.2 の
    /// `1x`/`1.25x`/`1.5x`/`1.75x` 丸めを適用する。
    pub small_max_bytes: u64,
    /// 大帯（1 バッファ 1 エントリの exclusive プール）の丸め粒度。
    pub large_granule_bytes: u64,
    /// 巨大帯（完全一致・1 エントリ／クラス保持）の下限バイト数。
    pub huge_min_bytes: u64,
    /// 巨大帯 1 クラスあたりの保持上限エントリ数（設計文書は「1
    /// エントリ／クラス」を確定済み。将来の見直しに備え設定値として
    /// 持たせる）。
    ///
    /// `SizeClassPool::put` が記帳直後に本フィールドを参照し、
    /// `class_bytes >= huge_min_bytes`（巨大帯）のエントリ数がこの上限を
    /// 超える間は最も古い同一クラスのエントリから追い出す
    /// （codex-review P1 是正。旧稿は総量上限＋グローバル LRU〈§3.4〉
    /// のみに委ねクラス単位の強制をしていなかったため、既定
    /// `max_pool_bytes` 128 MiB を上回るまで同一巨大クラスを複数保持
    /// できてしまっていた）。
    pub huge_entries_per_class: usize,
}

impl Default for SizeClassPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_bytes: 128 * 1024 * 1024,
            small_max_bytes: 1024 * 1024,
            large_granule_bytes: 2 * 1024 * 1024,
            huge_min_bytes: 64 * 1024 * 1024,
            huge_entries_per_class: 1,
        }
    }
}

/// [`SizeClassPool`] の統計スナップショット（POD。ハンドル・ポインタ・
/// trait object を一切含まない。設計文書 §3.1「フィールド更新契約」表が
/// 各フィールドの増減規則の正）。
///
/// `stats()` 単発の呼び出し内で全フィールドが厳密に同一時刻の値である
/// ことは要求しない（`Mutex` 保護フィールドと `AtomicU64`
/// （`pending_return_bytes`）を別々に読むため。診断用スナップショットと
/// しての利用に留める。§3.1）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolStats {
    /// 新規物理確保（[`SizeClassPool::record_allocation`]）の累積回数。
    pub alloc_count: u64,
    /// フリーリストからの再利用ヒット（[`SizeClassPool::take`] が
    /// `Some` を返した回数）の累積。
    pub reuse_count: u64,
    /// フリーリストが現在保持する総バイト数
    /// （`Σ(フリーリスト中の各エントリの class_bytes)`）。
    pub cached_bytes: u64,
    /// Metal 専用: GPU 完了待ちでフリーリストへ未合流の返却待ちバイト数
    /// （CUDA・CPU では常に `0`）。
    pub pending_return_bytes: u64,
    /// 現在貸出中の全ハンドルについての内部断片化ストック量
    /// （`Σ(class_bytes − logical_bytes)`。累積ではなくリアルタイム値）。
    pub capacity_waste_bytes: u64,
    /// `release_cached()` が個別解放に成功した累積バイト数（診断用）。
    pub released_bytes: u64,
}

/// `bytes` を §3.2 のサイズクラス表へ丸めた `class_bytes` を返す。
///
/// - `bytes == 0`: プール非経由（呼び出し元が空ハンドルを扱う契約。
///   本関数自体は `bytes == 0` を渡されても `Ok(0)` を返すのみで、
///   「経由しない」という制御自体は呼び出し元が担う）。
/// - `1..=255`: 256 B（小帯の最小クラス）へ切り上げる。
/// - `256..=small_max_bytes`: オクターブを `1x`/`1.25x`/`1.5x`/`1.75x`
///   の 4 段に分割し、切り上げ先の最小クラスへ丸める（内部断片化の
///   理論上限 25%）。
/// - `small_max_bytes` 超 〜 `huge_min_bytes` 未満: `large_granule_bytes`
///   単位へ切り上げる。
/// - `huge_min_bytes` 以上: 完全一致（丸めなし）。
///
/// `checked_mul`/`checked_add`（`u64`）で計算し、オーバーフロー時は
/// [`BackendError::DeviceAllocationFailed`] を返す（設計文書 §3.2 末尾。
/// 呼び出し元の shape 由来の巨大な確保要求に対する前段検証。OWASP A03）。
pub fn size_class_for(bytes: u64, cfg: &SizeClassPoolConfig) -> Result<u64, BackendError> {
    if bytes == 0 {
        return Ok(0);
    }
    if bytes >= cfg.huge_min_bytes {
        return Ok(bytes);
    }
    if bytes > cfg.small_max_bytes {
        // 大帯: `large_granule_bytes` の倍数へ切り上げ。
        let granule = cfg.large_granule_bytes.max(1);
        let steps = bytes
            .checked_add(granule - 1)
            .ok_or_else(|| overflow_err(bytes))?
            / granule;
        return steps
            .checked_mul(granule)
            .ok_or_else(|| overflow_err(bytes));
    }
    if bytes <= 255 {
        return Ok(256);
    }
    // 小帯: 2 の冪ごとのオクターブを 1x/1.25x/1.5x/1.75x の 4 段に分割。
    // `octave` は `bytes` が属するオクターブの下端の 2 冪指数
    // （`2^octave <= bytes < 2^(octave+1)`）。
    let octave = 63 - bytes.leading_zeros();
    let base: u64 = 1u64
        .checked_shl(octave)
        .ok_or_else(|| overflow_err(bytes))?;
    // 4 段の刻み幅は `base / 4`（= `base * 0.25`）。`base >= 256` なので
    // `base / 4 >= 64` であり整数除算での精度損失はない。
    let step = base / 4;
    for mult in 4..=7u64 {
        let class = base
            .checked_add(
                step.checked_mul(mult - 4)
                    .ok_or_else(|| overflow_err(bytes))?,
            )
            .ok_or_else(|| overflow_err(bytes))?;
        if class >= bytes {
            return Ok(class);
        }
    }
    // `mult == 7`（`base * 1.75`）でも `class >= bytes` を満たさない場合
    // （`bytes` が `(base * 1.75, base * 2)` の範囲。Cursor Bugbot 指摘。
    // 旧稿はここを「到達不能」と誤って前提していたが、`octave` の不変
    // 条件 `bytes < 2^(octave+1) == base * 2` はこの区間を排除しない
    // ため実際に到達しうる）は、次オクターブの base（`base * 2` ==
    // `2^(octave+1)`）へ切り上げる。設計文書 §3.2 が「巨大帯下限
    // 未満はオクターブ内 4 段丸め、それ以上は次段階の丸め」を要求して
    // おり、`2p` ちょうどまでの切り上げが小帯の契約（内部断片化の理論
    // 上限 25%）と整合する。
    base.checked_mul(2).ok_or_else(|| overflow_err(bytes))
}

fn overflow_err(bytes: u64) -> BackendError {
    BackendError::DeviceAllocationFailed(format!(
        "size_class_for: size-class rounding overflowed u64 for bytes={bytes}"
    ))
}

/// `class_bytes` をキーとするフリーリストの 1 バケット。
type FreeListEntry<H> = (u64, H);

/// `Mutex` で保護する内部状態（フリーリスト・統計。`pending_return_bytes`
/// は別途 `AtomicU64` で持つため含めない。§3.1「統計専用メソッドの
/// 検証」）。
struct PoolCore<H> {
    /// 挿入順を保つフリーリスト（LRU 破棄は先頭から。`take`/
    /// `take_one_for_release` は末尾からではなく該当クラス優先で探す
    /// ため単純な `Vec` で十分。設計文書は「グローバル LRU」を要求する
    /// が、挿入順管理自体は `Vec` の順序で表現し、破棄は
    /// 先頭〈最も古い〉から行う）。
    free: Vec<FreeListEntry<H>>,
    alloc_count: u64,
    reuse_count: u64,
    cached_bytes: u64,
    capacity_waste_bytes: u64,
    released_bytes: u64,
}

impl<H> Default for PoolCore<H> {
    fn default() -> Self {
        Self {
            free: Vec::new(),
            alloc_count: 0,
            reuse_count: 0,
            cached_bytes: 0,
            capacity_waste_bytes: 0,
            released_bytes: 0,
        }
    }
}

/// ハンドル型非依存のサイズクラス別プール本体（設計文書 §3.1）。
///
/// `H` は各バックエンドの生ハンドル型（`tensor-core` はその中身を
/// 一切知らない。`H: Send` のみを要求）。所有権の不変条件（設計文書
/// §3.1「不変条件」）: 同一ハンドルはフリーリストと貸出中のいずれか
/// 一方にのみ存在する。新規確保直後のハンドルは `take`/`record_
/// allocation` を経由せず直接呼び出し元の RAII ラッパーが排他所有する
/// （`SizeClassPool` はハンドルの custody を一切持たない。統計のみ
/// `record_allocation` で関知する）。
pub struct SizeClassPool<H> {
    config: SizeClassPoolConfig,
    core: Mutex<PoolCore<H>>,
    /// Metal 専用の返却待ちバイト数（lock-free。§3.1「統計専用メソッドの
    /// 検証」・§3.5「ロック順序規則」）。CUDA・CPU では常に `0` のまま
    /// （`record_pending_return`/`record_pending_merge` を呼ばないため）。
    pending_return_bytes: AtomicU64,
}

// `SizeClassPool<H>: Send + Sync where H: Send`（設計文書 §3.5）。
// `Mutex<PoolCore<H>>` は `H: Send` であれば `Sync` になる（`Mutex<T>:
// Sync where T: Send`）ため、`H: Send` のみで両方が自動導出される。
// 下記のコンパイル時 assert で固定する（§3.1 末尾）。

impl<H> SizeClassPool<H>
where
    H: Send,
{
    /// 新規のプールを構築する。
    pub fn new(config: SizeClassPoolConfig) -> Self {
        Self {
            config,
            core: Mutex::new(PoolCore::default()),
            pending_return_bytes: AtomicU64::new(0),
        }
    }

    /// 設定値を返す（丸め計算の再利用等、呼び出し元〈具体アロケータ〉が
    /// `size_class_for` を自ら呼ぶ際に使う）。
    pub fn config(&self) -> SizeClassPoolConfig {
        self.config
    }

    /// `Mutex` guard を取得する共通ヘルパー。poison 時は `into_inner` で
    /// 継続する（既存 `crate::pool::PoolCore` と同じ「panic させない」
    /// 方針。§3.5）。
    fn lock(&self) -> std::sync::MutexGuard<'_, PoolCore<H>> {
        self.core
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// フリーリストから `bytes`（呼び出し元が計算済みの `class_bytes`）に
    /// 一致するエントリを 1 件取り出す。取り出した時点で所有権は完全に
    /// 呼び出し元へ移り、フリーリストからは消える。無ければ `None`
    /// （呼び出し元が新規確保 FFI を呼び `record_allocation` で統計を
    /// 更新する契約。設計文書 §3.1）。
    ///
    /// **`reuse_count`／`capacity_waste_bytes` の更新はここでは行わない**
    /// （codex-review・Cursor Bugbot 指摘対応。旧稿は本メソッドが
    /// `reuse_count` のみを直接加算し、`capacity_waste_bytes`
    /// への加算〈設計文書 §3.1 契約表「`take` 成功で
    /// `+(class_bytes − bytes)`」〉が漏れていたため、再利用貸出後に
    /// `record_loan_end` が減算しても対応する加算が存在せず
    /// `saturating_sub` で過小表示・`debug_assert!` 失敗を招いていた）。
    /// 呼び出し元が `Some` を受け取った直後に必ず [`Self::record_reuse`]
    /// を呼び、両フィールドをまとめて更新する契約とする。
    pub fn take(&self, class_bytes: u64) -> Option<H> {
        let mut core = self.lock();
        let idx = core.free.iter().position(|(c, _)| *c == class_bytes)?;
        let (c, handle) = core.free.remove(idx);
        core.cached_bytes = core.cached_bytes.saturating_sub(c);
        Some(handle)
    }

    /// 新規物理確保が発生したことを記録する統計専用メソッド（ハンドルを
    /// 一切受け取らない。`take` が `None` を返した直後に具体アロケータが
    /// 呼ぶ。§3.1）。
    pub fn record_allocation(&self, logical_bytes: u64, class_bytes: u64) {
        let mut core = self.lock();
        core.alloc_count += 1;
        core.capacity_waste_bytes = core
            .capacity_waste_bytes
            .saturating_add(class_bytes.saturating_sub(logical_bytes));
        debug_assert!(
            class_bytes >= logical_bytes,
            "size class must never round below the requested logical size"
        );
    }

    /// フリーリストからの再利用貸出が成功したことを記録する統計専用
    /// メソッド（ハンドルを一切受け取らない。`take` が `Some` を返した
    /// 直後に具体アロケータが呼ぶ。§3.1 契約表・§8.2「具体メソッドの
    /// 引数形状の微調整」の範囲での新設）。
    ///
    /// `reuse_count` の `+1`（旧稿は `take` 自身が加算していた）と
    /// `capacity_waste_bytes` の `+(class_bytes − logical_bytes)`
    /// （codex-review P2・Cursor Bugbot 指摘対応。新規確保
    /// （`record_allocation`）と同様、再利用貸出でも `capacity_bytes −
    /// 論理バイト数` の内部断片化ストックを計上しなければ、対応する
    /// `record_loan_end` の減算と対にならない）を単一のロック区間で
    /// まとめて行う。`logical_bytes` はこの貸出が終わる際に呼ばれる
    /// `record_loan_end` へ渡す値と同一でなければならない
    /// （RAII ラッパーが自身のフィールドに保持し続けることで担保する。
    /// `record_allocation` と同じ契約）。
    pub fn record_reuse(&self, logical_bytes: u64, class_bytes: u64) {
        let mut core = self.lock();
        core.reuse_count += 1;
        core.capacity_waste_bytes = core
            .capacity_waste_bytes
            .saturating_add(class_bytes.saturating_sub(logical_bytes));
        debug_assert!(
            class_bytes >= logical_bytes,
            "size class must never round below the requested logical size"
        );
    }

    /// 貸出中のハンドルをフリーリストへ返却する（RAII ラッパーの
    /// `Drop` から、または `release_cached()` の個別解放失敗時の再挿入
    /// から呼ばれる。§3.1「不変条件」）。フリーリストへの記帳
    /// （`cached_bytes += class_bytes`）のみを行い、内部断片化の増減は
    /// 関知しない（`record_loan_end` が別途担う。§3.1）。
    ///
    /// **総量上限・グローバル LRU（設計文書 §3.4。OWASP A04 資源枯渇
    /// 対策）**: 記帳後、`cached_bytes + pending_return_bytes`
    /// （`max_pool_bytes`／LRU 判定の対象。§3.4「`max_pool_bytes`／LRU
    /// 判定の対象」）が `config().max_pool_bytes` を超える間、挿入順が
    /// 最も古いエントリ（`free` の先頭。`Vec` の push は末尾へ追加する
    /// ため先頭が最古）から追い出す。呼び出し元はこの戻り値
    /// （追い出されたエントリ）を `Mutex` 解放後に drop する契約とする
    /// （ロック保持中に `H` の drop〈FFI 解放を伴いうる〉を行わない。
    /// §3.5「プール本体は `Mutex<PoolCore>`」の「ロックを保持したまま
    /// FFI 呼び出しを行わない」方針と同じ理由）。`max_pool_bytes == 0`
    /// はプール無効（全パススルー）契約のため、この場合は挿入した
    /// エントリ自身を含め全件が追い出される。
    ///
    /// **巨大帯クラス別保持上限（`SizeClassPoolConfig::
    /// huge_entries_per_class`。設計文書 §3.2「巨大 | 64 MiB 以上 |
    /// 完全一致のみ・保持上限 1 エントリ／クラス」・codex-review P1
    /// 指摘対応）**: `class_bytes >= huge_min_bytes`（巨大帯）の場合、
    /// 記帳直後にフリーリスト中の同一 `class_bytes` のエントリ数を数え、
    /// `huge_entries_per_class` を超える間は最も古い同一クラスの
    /// エントリから追い出す（今回挿入した分は必ず最新のため対象外。
    /// 巨大帯は完全一致クラスのため「同一クラス」は「同一物理サイズ」
    /// と同義）。追い出したエントリは上記のグローバル LRU 追い出しと
    /// 同じ戻り値へ合流させ、呼び出し元が `Mutex` 解放後に drop する
    /// 契約を共有する。
    #[must_use = "追い出されたエントリは呼び出し元が Mutex 解放後に drop すること"]
    pub fn put(&self, class_bytes: u64, handle: H) -> Vec<(u64, H)> {
        let mut core = self.lock();
        core.free.push((class_bytes, handle));
        core.cached_bytes = core.cached_bytes.saturating_add(class_bytes);
        let mut evicted = self.evict_over_huge_class_limit(&mut core, class_bytes);
        evicted.extend(self.evict_over_capacity(&mut core));
        evicted
    }

    /// [`Self::put`] が記帳直後に呼ぶ巨大帯クラス別保持上限の強制本体。
    /// `class_bytes` が巨大帯（`>= huge_min_bytes`）でなければ何もしない
    /// （小帯・大帯はクラス別ではなく総量上限＋グローバル LRU
    /// （[`Self::evict_over_capacity`]）のみで制御する設計文書 §3.2 の
    /// 契約どおり）。
    fn evict_over_huge_class_limit(
        &self,
        core: &mut PoolCore<H>,
        class_bytes: u64,
    ) -> Vec<(u64, H)> {
        let mut evicted = Vec::new();
        if class_bytes < self.config.huge_min_bytes {
            return evicted;
        }
        loop {
            let count = core.free.iter().filter(|(c, _)| *c == class_bytes).count();
            if count <= self.config.huge_entries_per_class {
                break;
            }
            // 同一クラスのうち最も古い（`free` 内で最も先頭に近い）
            // エントリから追い出す（グローバル LRU と同じ「挿入順が
            // 最古から」の方針。今回挿入した分は末尾にあるため、他に
            // 同一クラスのエントリが残っている限りそちらが先に選ばれる）。
            let Some(idx) = core.free.iter().position(|(c, _)| *c == class_bytes) else {
                break;
            };
            let (bytes, handle) = core.free.remove(idx);
            core.cached_bytes = core.cached_bytes.saturating_sub(bytes);
            evicted.push((bytes, handle));
        }
        evicted
    }

    /// [`Self::put`] が記帳直後に呼ぶ LRU 追い出しの本体。`core` は
    /// 呼び出し元が既にロック取得済みであることが前提（`&mut
    /// PoolCore<H>` を直接受け取る。`flush_locked`〈`context.rs`〉と
    /// 同型のパターン）。
    fn evict_over_capacity(&self, core: &mut PoolCore<H>) -> Vec<(u64, H)> {
        let mut evicted = Vec::new();
        loop {
            let idle = core
                .cached_bytes
                .saturating_add(self.pending_return_bytes.load(Ordering::Relaxed));
            if idle <= self.config.max_pool_bytes || core.free.is_empty() {
                break;
            }
            // 先頭（挿入順が最も古いエントリ）から追い出す（グローバル
            // LRU。設計文書 §3.4「総量上限＋グローバル LRU」）。
            let (bytes, handle) = core.free.remove(0);
            core.cached_bytes = core.cached_bytes.saturating_sub(bytes);
            evicted.push((bytes, handle));
        }
        evicted
    }

    /// 貸出の終了（RAII ラッパーの `Drop`）を記録する統計専用メソッド。
    /// `put` 自体の呼び出しタイミングとは独立に、`Drop` の時点で常に
    /// 即座に呼ぶ（CUDA・Metal 共通。§3.1「フィールド更新契約」
    /// `capacity_waste_bytes` 行）。
    pub fn record_loan_end(&self, logical_bytes: u64, class_bytes: u64) {
        let mut core = self.lock();
        let waste = class_bytes.saturating_sub(logical_bytes);
        debug_assert!(
            core.capacity_waste_bytes >= waste,
            "capacity_waste_bytes underflow: loan end waste exceeds outstanding stock"
        );
        core.capacity_waste_bytes = core.capacity_waste_bytes.saturating_sub(waste);
    }

    /// 解放処理専用: フリーリストから 1 エントリだけ取り出す
    /// （`release_cached()` の解放トランザクションで使う。一括 `drain`
    /// は設けない。§3.1「解放時の所有権遷移」）。
    pub fn take_one_for_release(&self) -> Option<(u64, H)> {
        let mut core = self.lock();
        let entry = core.free.pop()?;
        core.cached_bytes = core.cached_bytes.saturating_sub(entry.0);
        Some(entry)
    }

    /// `take_one_for_release` で取り出したエントリの個別解放が成功した
    /// ことを記録する統計専用メソッド（`released_bytes` の加算のみ。
    /// `cached_bytes` は `take_one_for_release` が既に減算済み。§3.1）。
    pub fn record_release(&self, class_bytes: u64) {
        let mut core = self.lock();
        core.released_bytes = core.released_bytes.saturating_add(class_bytes);
    }

    /// Metal 専用: `pending_pool_returns` へ返却を委譲したことを記録する
    /// lock-free な統計専用メソッド（`AtomicU64::fetch_add`。`Mutex<
    /// PoolCore<H>>` を一切取らない。`BatchSlots` のロックを保持したまま
    /// 呼んでもデッドロックしない。§3.1・§3.5「ロック順序規則」）。
    pub fn record_pending_return(&self, class_bytes: u64) {
        self.pending_return_bytes
            .fetch_add(class_bytes, Ordering::Relaxed);
    }

    /// Metal 専用: `pending_pool_returns` からフリーリストへ合流させる
    /// ことを記録する lock-free な統計専用メソッド（`AtomicU64::
    /// fetch_sub`。`record_pending_return` と対になる。§3.1・§3.5）。
    ///
    /// 設計文書が保証するとおり、この減算は対応する加算より先に発生し
    /// 得ない構造（push/`take` と同一の `BatchSlots` クリティカル
    /// セクション内で対になって呼ばれる契約）であるため、`Mutex` 系
    /// 統計メソッドと異なり `debug_assert!`/`saturating_sub` の防御は
    /// 適用しない設計判断とする（§3.1「統計専用メソッドの検証」）。
    pub fn record_pending_merge(&self, class_bytes: u64) {
        self.pending_return_bytes
            .fetch_sub(class_bytes, Ordering::Relaxed);
    }

    /// 統計スナップショットを返す（診断用。§3.1「`stats`」）。
    pub fn stats(&self) -> PoolStats {
        let core = self.lock();
        PoolStats {
            alloc_count: core.alloc_count,
            reuse_count: core.reuse_count,
            cached_bytes: core.cached_bytes,
            pending_return_bytes: self.pending_return_bytes.load(Ordering::Relaxed),
            capacity_waste_bytes: core.capacity_waste_bytes,
            released_bytes: core.released_bytes,
        }
    }

    /// `max_pool_bytes`／LRU 判定の対象量（`cached_bytes +
    /// pending_return_bytes`。§3.4「`max_pool_bytes`／LRU 判定の対象」）。
    pub fn idle_bytes(&self) -> u64 {
        let cached = self.lock().cached_bytes;
        cached.saturating_add(self.pending_return_bytes.load(Ordering::Relaxed))
    }
}

// コンパイル時アサーション: `SizeClassPool<H>: Send + Sync where H: Send`
// を固定する（設計文書 §3.5・codex-review PR #1056 の合意事項）。
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    fn check<H: Send>() {
        assert_send_sync::<SizeClassPool<H>>();
    }
    let _ = check::<u32>;
};

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SizeClassPoolConfig {
        SizeClassPoolConfig::default()
    }

    // --- サイズクラス丸め（設計文書 §3.2 の 6 例 + 境界） ---

    #[test]
    fn size_class_representative_workload_examples() {
        let c = cfg();
        assert_eq!(size_class_for(200_704, &c).unwrap(), 229_376);
        assert_eq!(size_class_for(802_816, &c).unwrap(), 917_504);
        assert_eq!(size_class_for(65_536, &c).unwrap(), 65_536);
        assert_eq!(size_class_for(10_240, &c).unwrap(), 10_240);
        assert_eq!(size_class_for(2_560, &c).unwrap(), 2_560);
        assert_eq!(
            size_class_for(64 * 1024 * 1024, &c).unwrap(),
            64 * 1024 * 1024
        );
    }

    #[test]
    fn size_class_zero_bytes_passthrough() {
        assert_eq!(size_class_for(0, &cfg()).unwrap(), 0);
    }

    #[test]
    fn size_class_tiny_rounds_up_to_256() {
        assert_eq!(size_class_for(1, &cfg()).unwrap(), 256);
        assert_eq!(size_class_for(255, &cfg()).unwrap(), 256);
        assert_eq!(size_class_for(256, &cfg()).unwrap(), 256);
    }

    #[test]
    fn size_class_small_band_upper_boundary() {
        let c = cfg();
        assert_eq!(size_class_for(1024 * 1024, &c).unwrap(), 1024 * 1024);
    }

    #[test]
    fn size_class_large_band_rounds_to_granule() {
        let c = cfg();
        // 1 MiB + 1 は大帯へ入り、2 MiB 単位へ切り上げられる。
        assert_eq!(
            size_class_for(1024 * 1024 + 1, &c).unwrap(),
            2 * 1024 * 1024
        );
    }

    #[test]
    fn size_class_huge_lower_boundary_exact_match() {
        let c = cfg();
        // 64 MiB - 1 は大帯（2 MiB 単位切り上げ）に属する。
        assert_eq!(
            size_class_for(64 * 1024 * 1024 - 1, &c).unwrap(),
            64 * 1024 * 1024
        );
        // 64 MiB ちょうどは巨大帯（完全一致）。
        assert_eq!(
            size_class_for(64 * 1024 * 1024, &c).unwrap(),
            64 * 1024 * 1024
        );
    }

    #[test]
    fn size_class_octave_1_75x_boundary_rounds_within_octave() {
        // Cursor Bugbot 指摘の境界（`1.75p` ちょうど）: オクターブ内
        // 4 段丸めの最終段（`base * 1.75`）自身に一致する場合はその
        // クラスへ丸める（次オクターブへの繰り上げは発生しない）。
        // オクターブ 10（`base = 1024`）で検証する。
        let c = cfg();
        let base = 1024u64;
        assert_eq!(size_class_for(base + base * 3 / 4, &c).unwrap(), 1792);
    }

    #[test]
    fn size_class_between_1_75x_and_2x_rounds_up_to_next_octave() {
        // Cursor Bugbot High 指摘の回帰テスト: `(1.75p, 2p)` の範囲は
        // 旧稿では `size_class_for` が「到達不能」と誤って前提していた
        // 分岐に入り `AllocationSizeOverflow` 相当のエラーになっていた。
        // 設計文書 §3.2 は `2p` までの切り上げを要求するため、次
        // オクターブの base（`2p`）へ切り上げる。
        let c = cfg();
        assert_eq!(
            size_class_for(1793, &c).unwrap(),
            2048,
            "1.75p + 1 は 2p へ切り上げ"
        );
    }

    #[test]
    fn size_class_exact_2x_is_next_octave_base() {
        // `2p` ちょうどは（`octave` の再計算により）次オクターブの
        // 1x（そのもの）として既に正しく扱われていたことの確認
        // （回帰防止。上記 2 テストと合わせて `1.75p`／`1.75p+1`／`2p`
        // の 3 点を境界として押さえる）。
        let c = cfg();
        assert_eq!(size_class_for(2048, &c).unwrap(), 2048);
    }

    #[test]
    fn size_class_overflow_rejected() {
        // `bytes == u64::MAX` は既定設定では巨大帯（完全一致・丸め計算
        // なし）に該当し `Ok` を返してしまうため、大帯（`checked_add`
        // による切り上げ演算を経由する帯）でオーバーフローが起きる
        // よう `large_granule_bytes` を意図的に巨大化した設定を使う。
        let overflow_cfg = SizeClassPoolConfig {
            small_max_bytes: 0,
            large_granule_bytes: u64::MAX - 1,
            huge_min_bytes: u64::MAX,
            ..cfg()
        };
        let err = size_class_for(u64::MAX - 1, &overflow_cfg).unwrap_err();
        assert!(matches!(err, BackendError::DeviceAllocationFailed(_)));
    }

    // --- 所有権・統計契約 ---

    #[test]
    fn take_on_empty_pool_returns_none() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        assert!(pool.take(256).is_none());
    }

    #[test]
    fn put_then_take_reuses_handle_and_updates_stats() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        let _ = pool.put(256, 42);
        assert_eq!(pool.stats().cached_bytes, 256);

        let handle = pool.take(256).expect("reuse hit");
        assert_eq!(handle, 42);
        // `take` 自体は `reuse_count`／`capacity_waste_bytes` を更新
        // しない契約（`record_reuse` 呼び出し前は据え置き）。
        let stats = pool.stats();
        assert_eq!(stats.cached_bytes, 0);
        assert_eq!(stats.reuse_count, 0);

        pool.record_reuse(200, 256);
        let stats = pool.stats();
        assert_eq!(stats.reuse_count, 1);
        assert_eq!(stats.capacity_waste_bytes, 56);
    }

    #[test]
    fn record_reuse_waste_is_reversed_by_record_loan_end() {
        // codex P2・Cursor Bugbot 指摘対応の回帰テスト: 再利用貸出
        // （`take` 成功 → `record_reuse`）でも新規確保
        // （`record_allocation`）と同様に `capacity_waste_bytes` が
        // 加算され、対応する `record_loan_end` で過不足なく相殺される
        // ことを検証する（設計文書 §3.1 契約表「`take` 成功で
        // `+(class_bytes − bytes)`」）。
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        let _ = pool.put(256, 1);
        let handle = pool.take(256).expect("reuse hit");
        pool.record_reuse(200, 256);
        assert_eq!(pool.stats().capacity_waste_bytes, 56);

        pool.record_loan_end(200, 256);
        assert_eq!(
            pool.stats().capacity_waste_bytes,
            0,
            "再利用貸出で加算した waste は record_loan_end で過不足なく相殺されるべき"
        );

        // 相殺後、返却して次の貸出（新規確保経路）でも整合することを
        // 確認する（複数貸出経路をまたいだ加算・減算の対称性）。
        let _ = pool.put(256, handle);
        pool.record_allocation(180, 256);
        assert_eq!(pool.stats().capacity_waste_bytes, 76);
        pool.record_loan_end(180, 256);
        assert_eq!(pool.stats().capacity_waste_bytes, 0);
    }

    #[test]
    fn record_allocation_updates_alloc_count_and_waste() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        pool.record_allocation(200, 256);
        let stats = pool.stats();
        assert_eq!(stats.alloc_count, 1);
        assert_eq!(stats.capacity_waste_bytes, 56);
    }

    #[test]
    fn record_loan_end_reverses_capacity_waste() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        pool.record_allocation(200, 256);
        pool.record_loan_end(200, 256);
        assert_eq!(pool.stats().capacity_waste_bytes, 0);
    }

    #[test]
    fn take_one_for_release_transaction_removes_and_updates_cached_bytes() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        let _ = pool.put(256, 1);
        let _ = pool.put(512, 2);
        assert_eq!(pool.stats().cached_bytes, 768);

        let (class_bytes, handle) = pool.take_one_for_release().expect("entry present");
        assert_eq!(pool.stats().cached_bytes, 768 - class_bytes);
        // どちらの順で pop されても handle は投入済みの値のいずれか。
        assert!(handle == 1 || handle == 2);

        pool.record_release(class_bytes);
        assert_eq!(pool.stats().released_bytes, class_bytes);

        // 失敗を模した再挿入: `put` はハンドルをフリーリストへ戻す
        // （§3.1「解放時の所有権遷移」フェーズ (ii) 失敗時の再挿入）。
        let _ = pool.put(class_bytes, handle);
        assert_eq!(pool.stats().cached_bytes, 768);
    }

    #[test]
    fn take_one_for_release_on_empty_pool_returns_none() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        assert!(pool.take_one_for_release().is_none());
    }

    #[test]
    fn pending_return_and_merge_round_trip_to_zero() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        pool.record_pending_return(256);
        assert_eq!(pool.stats().pending_return_bytes, 256);
        assert_eq!(pool.idle_bytes(), 256);

        pool.record_pending_merge(256);
        assert_eq!(pool.stats().pending_return_bytes, 0);
        assert_eq!(pool.idle_bytes(), 0);
    }

    #[test]
    fn idle_bytes_combines_cached_and_pending() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        let _ = pool.put(256, 1);
        pool.record_pending_return(512);
        assert_eq!(pool.idle_bytes(), 768);
    }

    #[test]
    fn put_evicts_oldest_entries_when_over_max_pool_bytes() {
        let cfg = SizeClassPoolConfig {
            max_pool_bytes: 300,
            ..SizeClassPoolConfig::default()
        };
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg);

        // 1 件目（最古）は上限内。
        let evicted = pool.put(100, 1);
        assert!(evicted.is_empty(), "1 件目は上限を超えないため追い出しなし");

        // 2 件目で 200 バイトとなり依然上限（300）以内。
        let evicted = pool.put(100, 2);
        assert!(evicted.is_empty(), "2 件目まででは上限を超えない");

        // 3 件目で 300 バイトとなりちょうど上限（超過ではない）。
        let evicted = pool.put(100, 3);
        assert!(
            evicted.is_empty(),
            "ちょうど上限と一致する場合は追い出さない（`<=` 判定）"
        );

        // 4 件目で上限超過。最古（1 件目）から追い出される。
        let evicted = pool.put(100, 4);
        assert_eq!(
            evicted,
            vec![(100, 1)],
            "最も古いエントリ（挿入順1件目）から追い出されるはず"
        );
        assert_eq!(pool.stats().cached_bytes, 300, "追い出し後は上限以内に戻る");
    }

    #[test]
    fn put_with_zero_max_pool_bytes_evicts_everything() {
        // `max_pool_bytes == 0` はプール無効（全パススルー）契約。
        let cfg = SizeClassPoolConfig {
            max_pool_bytes: 0,
            ..SizeClassPoolConfig::default()
        };
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg);
        let evicted = pool.put(256, 1);
        assert_eq!(
            evicted,
            vec![(256, 1)],
            "max_pool_bytes == 0 は挿入したエントリ自身も即座に追い出す"
        );
        assert_eq!(pool.stats().cached_bytes, 0);
    }

    #[test]
    fn put_eviction_accounts_for_pending_return_bytes() {
        let cfg = SizeClassPoolConfig {
            max_pool_bytes: 300,
            ..SizeClassPoolConfig::default()
        };
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg);
        // 返却待ち分（pending_return_bytes）も判定対象に含む（§3.4）。
        pool.record_pending_return(250);
        let evicted = pool.put(100, 1);
        assert_eq!(
            evicted,
            vec![(100, 1)],
            "pending_return_bytes 込みで上限超過するため挿入直後に追い出される"
        );
    }

    #[test]
    fn size_class_pool_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SizeClassPool<u32>>();
    }

    // --- 巨大帯クラス別保持上限（codex-review P1 是正の回帰テスト） ---

    #[test]
    fn put_evicts_oldest_same_huge_class_entry_over_per_class_limit() {
        // 既定 `huge_entries_per_class == 1` のまま、`max_pool_bytes` を
        // 十分大きくして総量上限＋グローバル LRU（`evict_over_capacity`）
        // が介入しない条件で、巨大帯クラス別上限のみで追い出されることを
        // 確認する。
        let huge = 64 * 1024 * 1024u64;
        let cfg = SizeClassPoolConfig {
            max_pool_bytes: huge * 10,
            ..SizeClassPoolConfig::default()
        };
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg);

        let evicted = pool.put(huge, 1);
        assert!(evicted.is_empty(), "1 件目は上限（1 件／クラス）以内");

        // 同一クラスへ 2 件目を挿入すると上限超過し、最古（1 件目）が
        // 追い出される。
        let evicted = pool.put(huge, 2);
        assert_eq!(
            evicted,
            vec![(huge, 1)],
            "同一巨大クラスの最古エントリが追い出されるはず"
        );
        assert_eq!(pool.stats().cached_bytes, huge, "残るのは 2 件目の分のみ");
    }

    #[test]
    fn put_huge_class_limit_does_not_affect_other_classes() {
        // 巨大帯クラス別上限は「同一 `class_bytes`」単位。異なる巨大
        // クラス（完全一致丸めのため異なるバイト数 = 異なるクラス）は
        // 互いに干渉しない。
        let huge_a = 64 * 1024 * 1024u64;
        let huge_b = 65 * 1024 * 1024u64;
        let cfg = SizeClassPoolConfig {
            max_pool_bytes: huge_a + huge_b + 1,
            ..SizeClassPoolConfig::default()
        };
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg);

        assert!(pool.put(huge_a, 1).is_empty());
        assert!(
            pool.put(huge_b, 2).is_empty(),
            "異なる巨大クラスは互いの上限に影響しない"
        );
        assert_eq!(pool.stats().cached_bytes, huge_a + huge_b);
    }

    #[test]
    fn put_huge_class_limit_does_not_apply_below_huge_min_bytes() {
        // 巨大帯クラス別上限は `class_bytes >= huge_min_bytes` の場合の
        // み適用する。小帯・大帯は従来どおり総量上限＋グローバル LRU の
        // みで制御される（設計文書 §3.2）。
        let cfg = SizeClassPoolConfig {
            max_pool_bytes: u64::MAX,
            ..SizeClassPoolConfig::default()
        };
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg);

        // 同一クラス（256 バイト。小帯）を複数保持しても追い出されない。
        assert!(pool.put(256, 1).is_empty());
        assert!(pool.put(256, 2).is_empty());
        assert!(pool.put(256, 3).is_empty());
        assert_eq!(pool.stats().cached_bytes, 256 * 3);
    }

    #[test]
    fn put_huge_class_limit_respects_configured_entries_per_class() {
        // `huge_entries_per_class` を 2 に緩めた場合、3 件目の挿入で
        // 初めて追い出しが発生することを確認する（上限値そのものが
        // 参照されていることの直接検証）。
        let huge = 64 * 1024 * 1024u64;
        let cfg = SizeClassPoolConfig {
            max_pool_bytes: huge * 10,
            huge_entries_per_class: 2,
            ..SizeClassPoolConfig::default()
        };
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg);

        assert!(pool.put(huge, 1).is_empty());
        assert!(
            pool.put(huge, 2).is_empty(),
            "2 件目までは上限（2 件／クラス）以内"
        );
        let evicted = pool.put(huge, 3);
        assert_eq!(
            evicted,
            vec![(huge, 1)],
            "3 件目で最古（1 件目）が追い出される"
        );
    }
}
