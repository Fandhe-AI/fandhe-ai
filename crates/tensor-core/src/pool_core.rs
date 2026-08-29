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
    /// **未実装（イシュー #1021 実装時点。out-of-scope 記録）**:
    /// `SizeClassPool::put`（[`Self::default`] が持つ既定値 128 MiB の
    /// `max_pool_bytes`・グローバル LRU〈§3.4〉のみを実装しており、
    /// クラス単位の本フィールドは値を保持するのみで参照・強制していない
    /// （巨大帯 1 エントリ = 数十 MiB のため、既定 `max_pool_bytes`
    /// 128 MiB の下では実質的に少数エントリしか同時保持できず、
    /// グローバル LRU が代替的な安全弁として機能する。設計文書 §0
    /// 不変事項〈REQ-14 係数上限〉を侵さない範囲での意図的な簡略化）。
    /// クラス単位の厳密な強制が必要と判明した場合は別途実装する
    /// （`docs/device-memory-pool-design.md` §9 実装記録参照）。
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
    // 到達不能（`mult == 7` は `base * 1.75` であり、`bytes < 2^(octave+1)
    // == base * 2` の不変条件から必ず `class >= bytes` を満たす分岐に
    // 入る）。`.claude/rules/coding-rust.md`「本番経路で panic しない」に
    // 従い、万一到達した場合も型付きエラーで返す（防御的経路）。
    Err(BackendError::DeviceAllocationFailed(format!(
        "size_class_for: unreachable rounding failure for bytes={bytes}"
    )))
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
    pub fn take(&self, class_bytes: u64) -> Option<H> {
        let mut core = self.lock();
        let idx = core.free.iter().position(|(c, _)| *c == class_bytes)?;
        let (c, handle) = core.free.remove(idx);
        core.cached_bytes = core.cached_bytes.saturating_sub(c);
        core.reuse_count += 1;
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
    #[must_use = "追い出されたエントリは呼び出し元が Mutex 解放後に drop すること"]
    pub fn put(&self, class_bytes: u64, handle: H) -> Vec<(u64, H)> {
        let mut core = self.lock();
        core.free.push((class_bytes, handle));
        core.cached_bytes = core.cached_bytes.saturating_add(class_bytes);
        self.evict_over_capacity(&mut core)
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
        let stats = pool.stats();
        assert_eq!(stats.cached_bytes, 0);
        assert_eq!(stats.reuse_count, 1);
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
}
