//! `CudaMemory::with_host_view`（[`MemoryOps::with_host_view`] の CUDA
//! 実装。イシュー #1336）が使うホストステージングバッファのキャッシュ。
//!
//! ## 背景
//!
//! `MemoryOps::with_host_view` の既定実装（`tensor_core::buffer` モジュール
//! コメント）は毎回 `download`（`readback` 経由の `clone_dtoh` が**都度新規
//! `Vec<f32>` を確保**）→ `Tensor::new` → `as_slice()` という経路を辿る。
//! `docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`
//! （§4.2・§6）は、D2H の宛先を毎回新規確保する構成（P4）のみが 31→32 MiB
//! で 24〜33 倍の段差（glibc `M_MMAP_THRESHOLD` 動的上限）を示す一方、
//! **事前タッチ済み再利用 `Vec`（P5）は線形・段差なし**であることを実測
//! 済みである。本モジュールは形状（要素数）ごとに再利用するホスト
//! ステージングバッファへ `memcpy_dtoh` 1 回で D2H し、そのスライスを
//! `with_host_view` の呼び出し元クロージャへ借用として渡すことで、この
//! 段差を機構として回避する。
//!
//! ## 種別（[`HostStagingKind`]）
//!
//! `Pinned`（cudarc `CudaContext::alloc_pinned`。page-locked・
//! `CU_MEMHOSTALLOC_WRITECOMBINED` 固定・**unsafe 1 箇所**）と
//! `Pageable`（事前タッチ済み `Vec<f32>`。unsafe なし）の 2 種を実装し、
//! 実測（`docs/perf/cuda-host-view-staging-readout.md`）で本番種を選ぶ
//! 設計とした。WRITECOMBINED メモリは GPU 側の DMA 書き込み先としては
//! 有利だが CPU 側の読み出しが著しく遅いことが知られており（cudarc-0.19.8
//! `core.rs:1406-1427`）、D2H 自体は速くても「D2H＋ホスト読み出し」の
//! 合計では `Pinned` が `Pageable` に劣る可能性がある（決め打ちしない）。
//!
//! **GB10 実機実測（2026-09-08。`docs/perf/cuda-host-view-staging-readout.md`
//! §5）は当初、計測対象クロージャの戻り値を `black_box` で保護しない
//! バイナリで取得した参考値だったが、is-optimized-away 懸念を是正した
//! うえでの再計測（同 doc §5.3・イシュー #1438）により `Pinned` が全
//! 計測形状（N=1024/2048/4096）で `Pageable` をさらに上回る（約 6〜21%
//! 高速）ことが確定した。この確定値とユーザー承認（2026-09-09・
//! イシュー #1478）に基づき、既定 `HOST_STAGING_KIND` は `Pinned`
//! へ切り替えている**（同 doc §8。`Pageable` は `new_with_host_
//! staging_kind` 経由で A/B 比較用に明示選択できる対照腕として残す）。
//! `cuMemHostAlloc`（`alloc_pinned`）が失敗した場合は `CudaError` として
//! 呼び出し元へ fail-closed に伝播し、`Pageable` へのサイレント
//! フォールバックは行わない（意図的な契約。`docs/perf/cuda-host-view-
//! staging-readout.md` §8 参照）。
//!
//! ## 同期契約
//!
//! `MemoryOps::with_host_view` の呼び出し元契約（`tensor_core::buffer`
//! モジュールの同トレイトのドキュメンテーションコメント）と同一:
//! 「復帰時点でデバイス側の書き込みが完了していること」。実際の
//! `memcpy_dtoh`／`synchronize` 呼び出しは `memory.rs::CudaMemory::
//! with_host_view` が `with_driver_call`（poison／世代状態機械）経由で
//! 行う。本モジュール自身は driver を直接呼ばない（`HostStaging::alloc`
//! の `Pinned` 確保を除く）。
//!
//! ## ロック方針（take/put 方式）
//!
//! [`HostStagingCache`] の `Mutex` は `take`／`put` の間（データ構造操作
//! のみ）だけ保持し、`memcpy_dtoh`・`synchronize`・呼び出し元クロージャ
//! `f` の実行中は保持しない（`memory.rs::CudaMemory::with_host_view` の
//! 実装参照）。これにより `f` の内部で別バッファの `with_host_view` を
//! 再入しても deadlock しない（同一 numel の再入は新規確保で対応する）。

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtrMut, HostSlice, PinnedHostSlice,
};

use crate::error::CudaError;

/// 本番経路が使うステージング種別。GB10 実機実測（`docs/perf/cuda-host-
/// view-staging-readout.md` §5.2・§5.3。イシュー #1438 で is-optimized-
/// away 懸念を是正した確定値）で `Pinned` が全計測形状で `Pageable` を
/// 上回ることを確認したうえ、unsafe 経路の既定化についてユーザー承認
/// （2026-09-09・イシュー #1478）を得て `Pinned` へ切り替えた（同 doc
/// §8）。確保失敗（`cuMemHostAlloc`）は `CudaError` として fail-closed に
/// 伝播し、`Pageable` への暗黙フォールバックはしない。`default_host_
/// staging_kind_is_pinned`（本ファイル下部）が将来の意図しない差し戻し
/// を検知する drift ガード。
pub(crate) const HOST_STAGING_KIND: HostStagingKind = HostStagingKind::Pinned;

/// [`HostStagingCache`] が保持するホストバッファ合計の確保上限
/// （バイト）。超過分は使用後に破棄する（`.claude/rules/security.md`
/// 「リソース枯渇（DoS）」対策。page-locked メモリはホスト RAM を固定
/// するため特に重要）。4096×4096 の f32（64 MiB）を複数形状ぶん保持
/// できる程度の余裕を持たせた既定値。
pub(crate) const HOST_STAGING_CAP_BYTES: u64 = 256 * 1024 * 1024;

/// ホストステージングバッファの実装種別。`Pageable`・`Pinned` いずれも
/// `pub`（`pub(crate)` から変更。codex-review 指摘の経緯: 当時の本番既定
/// （`HOST_STAGING_KIND`）は `Pageable` に固定されており、`Pinned`
/// 経路を実機で検証する・両者を A/B 比較する手段が診断入口から
/// 提供されていなかった）で、`internal-diagnostics` feature（既定
/// off）限定の `memory::CudaMemory::with_host_view_using_kind`
/// （`lib.rs` の `pub use host_staging::HostStagingKind` re-export と
/// 対）から crate 外部（実機 `#[ignore]` テスト・A/B ハーネス）が
/// 明示的に選べる。現在は本番既定が `Pinned`（#1478）であるため、この
/// 診断入口は主に `Pageable`（対照腕）を明示選択する用途で使う。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostStagingKind {
    /// cudarc `alloc_pinned`（page-locked・WRITECOMBINED）。unsafe 1 箇所。
    /// **本番既定**（`HOST_STAGING_KIND`。イシュー #1478・2026-09-09
    /// ユーザー承認）。
    ///
    /// GB10 実機実測（`docs/perf/cuda-host-view-staging-readout.md`
    /// §5.2・§5.3・2026-09-08。イシュー #1438 で is-optimized-away 懸念を
    /// 是正した確定値）では本 variant が全計測形状（N=1024/2048/4096）で
    /// `Pageable` を一貫して上回ることを確認済み（約 6〜21% 高速）。
    /// `memory::CudaMemory::new_with_host_staging_kind`（`internal-
    /// diagnostics` feature 限定）を介して `Pageable` を明示選択すれば
    /// 切替前の既定（#1336〜#1438）と同じ経路を A/B 比較用の対照腕として
    /// 使える。
    Pinned,
    /// 事前タッチ済み pageable `Vec<f32>`（unsafe なし）。切替前の既定
    /// （#1336〜#1438）。`internal-diagnostics` feature 限定の
    /// `new_with_host_staging_kind` 経由で A/B 比較用の対照腕として
    /// 明示選択できる。
    Pageable,
}

/// ホストステージングバッファ本体（種別ごとの実データを保持する）。
pub(crate) enum HostStaging {
    Pinned(PinnedHostSlice<f32>),
    Pageable(Vec<f32>),
}

impl HostStaging {
    fn len(&self) -> usize {
        match self {
            HostStaging::Pinned(p) => p.len(),
            HostStaging::Pageable(v) => v.len(),
        }
    }

    fn byte_len(&self) -> u64 {
        (self.len() * std::mem::size_of::<f32>()) as u64
    }

    /// `CudaStream::memcpy_dtoh` の `dst: &mut Dst where Dst: HostSlice<T>
    /// + ?Sized` へそのまま渡せるトレイトオブジェクト参照（両 variant を
    /// 統一的に扱う。`cudarc::driver::HostSlice` は object-safe な設計
    /// のため `dyn HostSlice<f32>` として扱える）。
    pub(crate) fn as_host_slice_mut(&mut self) -> &mut dyn HostSlice<f32> {
        match self {
            HostStaging::Pinned(p) => p,
            HostStaging::Pageable(v) => v,
        }
    }

    /// 転送後のホスト側読み出し（`with_host_view` の呼び出し元クロージャ
    /// へ渡すスライス）。呼び出し元は `memcpy_dtoh` 直後に必ず
    /// `stream.synchronize()` を経由済みであること（`memory.rs::
    /// CudaMemory::with_host_view` の実装が保証する呼び出し順）。`Pinned`
    /// 側の `as_slice()` 内部 `event.synchronize()` は、その `synchronize`
    /// 後は既に完了済みの event を待つだけの no-op に近い二重待ちであり
    /// 追加コストは無視できる。
    pub(crate) fn as_slice(&self) -> Result<&[f32], CudaError> {
        match self {
            HostStaging::Pinned(p) => Ok(p.as_slice()?),
            HostStaging::Pageable(v) => Ok(v.as_slice()),
        }
    }

    /// H2D pinned staging（イシュー #1585・下部「H2D 用ステージング」節）
    /// が呼び出し元データを書き込む先として使う可変ビュー。`Pinned`
    /// 側の `as_mut_slice()` 内部 `event.synchronize()` は、直前の D2H
    /// （このバッファを別用途で使っていた場合）や前回の H2D 発行が
    /// 完了するまで待つ（`take` がキャッシュから取り出す＝前回の
    /// `clone_htod`／`memcpy_htod` 発行が終わっていない可能性がある
    /// ため、ここで待ってから上書きする契約。`PinnedHostSlice` の
    /// event 追跡により、上書き前に前回の非同期 DMA 読み取りが完了
    /// していることが保証される）。
    pub(crate) fn as_mut_slice(&mut self) -> Result<&mut [f32], CudaError> {
        match self {
            HostStaging::Pinned(p) => Ok(p.as_mut_slice()?),
            HostStaging::Pageable(v) => Ok(v.as_mut_slice()),
        }
    }

    /// `kind` に応じて `numel` 要素ぶんのステージングバッファを新規確保
    /// する（キャッシュ miss 時。`HostStagingCache::take` が `None` を
    /// 返した場合に `memory.rs` から呼ばれる）。
    pub(crate) fn alloc(
        kind: HostStagingKind,
        ctx: &Arc<CudaContext>,
        numel: usize,
    ) -> Result<Self, CudaError> {
        match kind {
            HostStagingKind::Pageable => {
                // 事前タッチ（ゼロ初期化確保）: #1146 P5「事前タッチ済み
                // 再利用 Vec」相当。確保直後に全要素へ書き込むため、初回
                // ページフォールトを本関数内で消化してから返す。
                Ok(HostStaging::Pageable(vec![0.0f32; numel]))
            }
            HostStagingKind::Pinned => {
                // SAFETY: `CudaContext::alloc_pinned` が unsafe な唯一の
                // 理由（cudarc-0.19.8 `core.rs:1410-1411` "この呼び出しの
                // 後、メモリは未初期化"）は、返す `PinnedHostSlice` の
                // 内容が確保直後は不定であること。本関数はキャッシュ
                // miss 時にのみ呼ばれ、呼び出し元
                // （`memory.rs::CudaMemory::with_host_view`）は返った
                // バッファを `f`（呼び出し元クロージャ）へ渡す前に必ず
                // `CudaStream::memcpy_dtoh` で要素数ぶん全域を D2H 上書き
                // してから `as_slice()` で読み出す（未初期化内容が外部へ
                // 露出する経路は存在しない）。要素型は `f32` であり
                // 全ビットパターンが有効な浮動小数点表現になるため
                // （`cudarc::driver::ValidAsZeroBits` が `f32` に実装
                // 済み。`memory.rs::alloc_zeroed_inner` の managed 確保
                // 分岐と同種の安全性根拠）、未初期化ビット列を無効値と
                // して解釈する余地もない。
                let pinned = unsafe { ctx.alloc_pinned::<f32>(numel)? };
                Ok(HostStaging::Pinned(pinned))
            }
        }
    }
}

/// [`HostStagingCache::take`]／[`put`](HostStagingCache::put) の呼び出し
/// 回数・キャッシュ状態を集計する診断用スナップショット（テスト・
/// 実機診断向け。`pub` だが値の生成は本モジュール限定）。
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HostStagingStats {
    /// `take` が既存エントリを返せた回数。
    pub hits: u64,
    /// `take` が新規確保を要求した回数（世代・長さ不一致による破棄も
    /// 含む。`fandhe_ai_backend_cuda` クレート外からは判別しない）。
    pub misses: u64,
    /// 世代・長さ不一致またはキャッシュ上限超過により破棄したエントリ数。
    pub evicted: u64,
    /// 現在キャッシュが保持しているバイト数の合計。
    pub cached_bytes: u64,
}

/// キャッシュが保持する 1 エントリ（バッファ本体＋確保時点の ordinal 世代）。
struct StagingEntry {
    buf: HostStaging,
    generation: u64,
}

/// 形状（要素数）ごとに再利用するホストステージングバッファのキャッシュ
/// 本体。`CudaMemory::host_staging`（`Arc<Mutex<Self>>`。モジュール冒頭
/// 「ロック方針」節）が保持し、`with_host_view` から `take`／`put` される。
pub(crate) struct HostStagingCache {
    entries: HashMap<usize, StagingEntry>,
    cached_bytes: u64,
    cap_bytes: u64,
    kind: HostStagingKind,
    stats: HostStagingStats,
}

impl HostStagingCache {
    pub(crate) fn new(kind: HostStagingKind) -> Self {
        Self {
            entries: HashMap::new(),
            cached_bytes: 0,
            cap_bytes: HOST_STAGING_CAP_BYTES,
            kind,
            stats: HostStagingStats::default(),
        }
    }

    #[cfg(test)]
    fn with_cap(kind: HostStagingKind, cap_bytes: u64) -> Self {
        Self {
            cap_bytes,
            ..Self::new(kind)
        }
    }

    pub(crate) fn kind(&self) -> HostStagingKind {
        self.kind
    }

    /// 統計スナップショットを返す。`cached_bytes` は `take`／`put` の
    /// たびに `self.stats.cached_bytes` を個別更新するのではなく、
    /// 読み出し時に `self.cached_bytes`（唯一の真実源）から合成する
    /// （2 箇所を独立更新して drift させない設計）。
    pub(crate) fn stats(&self) -> HostStagingStats {
        HostStagingStats {
            cached_bytes: self.cached_bytes,
            ..self.stats
        }
    }

    /// `numel`・`generation` に一致する既存エントリを取り出す
    /// （世代不一致・長さ不一致は fail-closed に破棄して miss 扱いに
    /// する。`context_cache::invalidate` はストリームの世代を進めるが
    /// `CudaMemory` 自体は再生成されないため、旧世代のステージング
    /// バッファを新世代の D2H へ誤って使い回すことを防ぐ。実装計画
    /// 「スコープ上の重要事実」item 7）。
    pub(crate) fn take(&mut self, numel: usize, generation: u64) -> Option<HostStaging> {
        match self.entries.remove(&numel) {
            Some(entry) if entry.generation == generation && entry.buf.len() == numel => {
                self.cached_bytes = self.cached_bytes.saturating_sub(entry.buf.byte_len());
                self.stats.hits += 1;
                Some(entry.buf)
            }
            Some(stale) => {
                self.cached_bytes = self.cached_bytes.saturating_sub(stale.buf.byte_len());
                self.stats.misses += 1;
                self.stats.evicted += 1;
                None
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// 使用済みバッファをキャッシュへ返却する。`cap_bytes` を超える
    /// 場合は登録せず破棄する（DoS 対策。モジュール冒頭コメント参照）。
    ///
    /// 同一 `numel` キーへ既存エントリがある状態（再入・並行
    /// `with_host_view` により同一形状のバッファが `take` されずに
    /// 複数回 `put` されうる）で `HashMap::insert` が既存エントリを
    /// 置換する場合、置換前の既存分バイト数を `cached_bytes` から
    /// 差し引いてから新規分を加算する。差し引かずに加算のみを行うと
    /// `cached_bytes` が実際の保持量（`entries` の総バイト数）より
    /// 過大計上され、以降の上限判定（本メソッド冒頭の cap 比較）が
    /// 不当に返却を拒否し、`release_all`（`CudaMemory::
    /// release_host_staging` 経由）の返却値も過大になる
    /// （codex-review 指摘 P2・Cursor Bugbot Medium 指摘。両者は
    /// 同一箇所・同一問題）。
    fn put(&mut self, numel: usize, generation: u64, buf: HostStaging) {
        let bytes = buf.byte_len();
        let existing_bytes = self
            .entries
            .get(&numel)
            .map(|entry| entry.buf.byte_len())
            .unwrap_or(0);
        let projected_cached_bytes = self
            .cached_bytes
            .saturating_sub(existing_bytes)
            .saturating_add(bytes);
        if projected_cached_bytes > self.cap_bytes {
            self.stats.evicted += 1;
            return;
        }
        self.cached_bytes = projected_cached_bytes;
        self.entries.insert(numel, StagingEntry { buf, generation });
    }

    /// 全エントリを破棄し解放したバイト数を返す（REQ-14
    /// `release_cached` 系と同型の明示解放 API。
    /// `CudaMemory::release_host_staging` から呼ばれる）。
    pub(crate) fn release_all(&mut self) -> u64 {
        let bytes = self.cached_bytes;
        self.entries.clear();
        self.cached_bytes = 0;
        bytes
    }
}

/// 使用済みバッファをキャッシュへ返却する（`memory.rs::CudaMemory::
/// return_staging` から呼ばれる。poison 後も `into_inner` で返却を試み
/// panic しない一方、以降の新規取得〈`memory.rs::CudaMemory::
/// take_cached_staging` が直接 `lock()` する経路〉は通常どおり poison
/// エラーとして fail-closed に拒否される）。
pub(crate) fn put_back(
    cache: &std::sync::Mutex<HostStagingCache>,
    numel: usize,
    generation: u64,
    buf: HostStaging,
) {
    let mut guard = match cache.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.put(numel, generation, buf);
}

// ---------------------------------------------------------------------
// H2D 用ステージング（イシュー #1585。低レイヤー診断
// `docs/perf/lowlayer-diagnosis-2026-09-12.md` §7 表の B-3 行で起票された
// 候補）。
//
// 上記の [`HostStagingCache`]・[`put_back`] は
// `MemoryOps::with_host_view`（D2H 側。イシュー #1336/#1478）専用の
// キャッシュだった。本節はホスト→デバイス方向（H2D）向けの対称な
// 機構を、既定 OFF・明示 opt-in で追加する。
//
// 現行の CUDA H2D（`memory.rs::upload_inner`／`upload_into`・`gemm.rs`
// の各 `run_*` 系・`ops.rs` の転置 NT 分岐）は、すべて pageable ホスト
// メモリ（`&[f32]`）からの `stream.clone_htod`／`memcpy_htod` を直接
// 発行している。cudarc-0.19.8 は内部で `cuMemcpyHtoDAsync` を発行する
// が、pageable ソースでは driver が同期的に一時 pinned バッファへ
// ステージングするため、呼び出し元が明示的に pinned メモリを用意すれば
// この暗黙ステージングを避けられる（`docs/backend-cuda-async-execution-
// design.md` §3 不変条件 I3 の再確認・追補も参照）。
//
// **既定 OFF・明示 opt-in**（`set_pinned_h2d_enabled`）。フラグ OFF・
// `data` が空の場合は導入前と経路・出力とも bit 同一（`upload_new`／
// `upload_into` 冒頭の早期分岐）。ON 時も、pinned バッファへ同期コピー
// してから `clone_htod`／`memcpy_htod` へ渡すだけであり、DMA 転送
// 対象の内容自体は変わらないため出力は bit 同一（数値契約は変えない。
// `.claude/rules/coding-rust.md` の FMA 契約とは独立の軸）。
//
// **unsafe は追加しない**: 新規確保は [`HostStaging::alloc`]
// （`HostStagingKind::Pinned` 分岐。D2H 側と共有する既存の唯一の
// `unsafe` ブロック）を再利用する。

/// H2D pinned staging の既定値の単一情報源（イシュー #1585・codex-review
/// P2 是正。`docs/backend-metal-splitk-decision.md` §5「定数ゲートの
/// 撤去（#1547）」で確立した `split_k_runtime::SPLIT_K_DEFAULT_ENABLED`
/// と同型のパターン）。既定 `false`（従来の pageable 直接転送経路）。
///
/// `PINNED_H2D_ENABLED` の初期値をこの定数から seed し、drift ガード
/// テスト（[`tests::default_h2d_enabled_matches_declared_default`]）は
/// `set_pinned_h2d_enabled` を一切呼ばずにこの定数自体を直接検査する。
/// 旧実装は `assert!` の直前で `set_pinned_h2d_enabled(false)` を呼んで
/// いたため、`static` の初期値そのものが誤って `true` へ差し戻されても
/// テストが偽陽性で pass してしまい検知できない欠陥があった（codex-
/// review 指摘）。
pub(crate) const PINNED_H2D_DEFAULT_ENABLED: bool = false;

/// H2D pinned staging の opt-in フラグ本体。[`PINNED_H2D_DEFAULT_ENABLED`]
/// で初期化するプロセスワイド `AtomicBool`（`crate::placement::
/// MANAGED_PLACEMENT_ENABLED` と同型。`Ordering::SeqCst`。頻度の低い
/// 設定変更のため緩い順序による最適化は不要）。
static PINNED_H2D_ENABLED: AtomicBool = AtomicBool::new(PINNED_H2D_DEFAULT_ENABLED);

/// H2D pinned staging を有効化・無効化する（`facade::
/// set_cuda_pinned_h2d_enabled` から委譲される。プロセスワイドな設定
/// であり、以降の全スレッド・全 `CudaMemory`／`CudaGemm` インスタンス
/// の H2D 発行に反映される）。
pub fn set_pinned_h2d_enabled(enabled: bool) {
    PINNED_H2D_ENABLED.store(enabled, Ordering::SeqCst);
}

/// 現在の H2D pinned staging opt-in 状態を返す（既定 `false`）。
pub fn pinned_h2d_enabled() -> bool {
    PINNED_H2D_ENABLED.load(Ordering::SeqCst)
}

/// `PINNED_H2D_ENABLED` を操作するテスト間で共有する直列化ロック
/// （`cfg(test)` 限定。`crate::placement::test_support` と同じ理由で
/// 本フラグ専用の別ロックとして用意する）。
#[cfg(test)]
pub(crate) mod pinned_h2d_test_support {
    use std::sync::Mutex as StdMutex;

    pub(crate) fn pinned_h2d_flag_test_lock() -> &'static StdMutex<()> {
        static LOCK: StdMutex<()> = StdMutex::new(());
        &LOCK
    }
}

/// [`H2dStagingCache`] が保持する 1 エントリ（`numel` ごとに複数
/// エントリを許す設計上の理由は [`H2dStagingCache`] のドキュメンテー
/// ションコメント参照）。
struct H2dStagingEntry {
    buf: HostStaging,
    generation: u64,
}

/// H2D 用ホストステージングバッファのキャッシュ本体（イシュー #1585）。
///
/// [`HostStagingCache`]（D2H 側）と異なり、**同一 `numel` に対して
/// 複数エントリ**（`HashMap<usize, Vec<H2dStagingEntry>>`）を保持する。
/// 正方 GEMM は同じ要素数の A・B を連続して upload するため（例:
/// `gemm.rs::run_f32_kernel` の `a_dev`／`b_dev`）、1 エントリ方式だと
/// 2 個目（B）が毎回キャッシュ miss となり `cuMemHostAlloc`（ミリ秒級。
/// `docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`）
/// を毎回発行してしまい、機構と無関係な性能後退（REJECT 判定）を
/// 招く。`take` は LIFO（`Vec::pop`）で取り出し、世代・長さ不一致の
/// エントリは fail-closed に破棄する（[`HostStagingCache::take`] と同じ
/// 契約。`context_cache::invalidate` 後の旧世代バッファ誤使用を防ぐ）。
///
/// `CudaMemory`／`CudaGemm` それぞれがインスタンス単位で本キャッシュを
/// 持つため（`memory.rs`／`gemm.rs` のフィールド参照）、プロセス全体の
/// pinned 常駐量は理論上 `cap_bytes` の複数倍になりうる点に注意
/// （既定化判断〈本イシューのスコープ外〉で再検討する）。
pub(crate) struct H2dStagingCache {
    entries: HashMap<usize, Vec<H2dStagingEntry>>,
    cached_bytes: u64,
    cap_bytes: u64,
    stats: HostStagingStats,
}

impl H2dStagingCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            cached_bytes: 0,
            cap_bytes: HOST_STAGING_CAP_BYTES,
            stats: HostStagingStats::default(),
        }
    }

    #[cfg(test)]
    fn with_cap(cap_bytes: u64) -> Self {
        Self {
            cap_bytes,
            ..Self::new()
        }
    }

    /// 統計スナップショット（[`HostStagingCache::stats`] と同じ
    /// 「`cached_bytes` は唯一の真実源から合成する」設計）。
    pub(crate) fn stats(&self) -> HostStagingStats {
        HostStagingStats {
            cached_bytes: self.cached_bytes,
            ..self.stats
        }
    }

    /// `numel` キーに対応する `Vec` の末尾（LIFO）から、`generation` に
    /// 一致するエントリを探して取り出す。世代・長さ不一致のエントリは
    /// 経路上で破棄する（fail-closed。[`HostStagingCache::take`] と同じ
    /// 契約）。
    pub(crate) fn take(&mut self, numel: usize, generation: u64) -> Option<HostStaging> {
        let Some(vec) = self.entries.get_mut(&numel) else {
            self.stats.misses += 1;
            return None;
        };
        while let Some(entry) = vec.pop() {
            self.cached_bytes = self.cached_bytes.saturating_sub(entry.buf.byte_len());
            if entry.generation == generation && entry.buf.len() == numel {
                self.stats.hits += 1;
                if vec.is_empty() {
                    self.entries.remove(&numel);
                }
                return Some(entry.buf);
            }
            self.stats.evicted += 1;
        }
        self.entries.remove(&numel);
        self.stats.misses += 1;
        None
    }

    /// 使用済みバッファをキャッシュへ返却する。`cap_bytes` を超える
    /// 場合は登録せず破棄する（DoS 対策。page-locked メモリはホスト
    /// RAM を固定するため特に重要。`.claude/rules/security.md`）。
    pub(crate) fn put(&mut self, numel: usize, generation: u64, buf: HostStaging) {
        let bytes = buf.byte_len();
        let projected = self.cached_bytes.saturating_add(bytes);
        if projected > self.cap_bytes {
            self.stats.evicted += 1;
            return;
        }
        self.cached_bytes = projected;
        self.entries
            .entry(numel)
            .or_default()
            .push(H2dStagingEntry { buf, generation });
    }

    /// 全エントリを破棄し解放したバイト数を返す（REQ-14
    /// `release_cached` 系と同型の明示解放 API。
    /// `CudaMemory::release_h2d_staging`／`CudaGemm::release_h2d_staging`
    /// から呼ばれる）。
    pub(crate) fn release_all(&mut self) -> u64 {
        let bytes = self.cached_bytes;
        self.entries.clear();
        self.cached_bytes = 0;
        bytes
    }
}

/// [`H2dStagingCache::take`] の miss 時に [`HostStaging::alloc`]
/// （`HostStagingKind::Pinned`。既存 `unsafe` 1 箇所を再利用）で新規
/// 確保する（poison 時も `into_inner` で回復し panic しない。
/// [`put_back`] と同じ方針）。
fn take_or_alloc_h2d(
    cache: &Mutex<H2dStagingCache>,
    ctx: &Arc<CudaContext>,
    numel: usize,
    generation: u64,
) -> Result<HostStaging, CudaError> {
    let existing = {
        let mut guard = match cache.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.take(numel, generation)
    };
    match existing {
        Some(buf) => Ok(buf),
        None => HostStaging::alloc(HostStagingKind::Pinned, ctx, numel),
    }
}

/// 使用済み H2D ステージングバッファをキャッシュへ返却する
/// （[`put_back`] の H2D 版）。
fn put_back_h2d(cache: &Mutex<H2dStagingCache>, numel: usize, generation: u64, buf: HostStaging) {
    let mut guard = match cache.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.put(numel, generation, buf);
}

/// **新規デバイスバッファへの H2D**（`stream.clone_htod(data)` の
/// pinned staging 版）。`memory.rs::upload_inner`（`Device` 分岐）・
/// `gemm.rs` の各 `run_*` 系（`a_dev`／`b_dev`／`bias_dev` 等）・
/// `ops.rs` の転置 NT 分岐（`bt_dev`）から呼ばれる。
///
/// フラグ OFF または `data` が空の場合は `stream.clone_htod(data)` を
/// そのまま呼ぶ（経路・出力とも導入前と bit 同一。0 要素で
/// `cuMemHostAlloc` を発行しない）。ON の場合は `cache` から
/// pinned バッファを取得（miss なら新規確保）し、`data` を同期コピー
/// してから **`PinnedHostSlice` 自身**を `clone_htod` へ渡す
/// （`as_slice()` で得た `&[f32]` を渡すと `[T]` 側の `HostSlice` 実装が
/// event を記録せず、非同期 DMA と次回のホスト書き込みが競合しうる
/// ため、必ずステージング型そのものを渡す契約）。
pub(crate) fn upload_new(
    stream: &Arc<CudaStream>,
    cache: &Mutex<H2dStagingCache>,
    ctx: &Arc<CudaContext>,
    generation: u64,
    data: &[f32],
) -> Result<CudaSlice<f32>, CudaError> {
    if !pinned_h2d_enabled() || data.is_empty() {
        return Ok(stream.clone_htod(data)?);
    }
    let numel = data.len();
    let mut staging = take_or_alloc_h2d(cache, ctx, numel, generation)?;
    staging.as_mut_slice()?.copy_from_slice(data);
    let result = match &staging {
        HostStaging::Pinned(p) => stream.clone_htod(p)?,
        HostStaging::Pageable(v) => stream.clone_htod(v.as_slice())?,
    };
    put_back_h2d(cache, numel, generation, staging);
    Ok(result)
}

/// **既存デバイスバッファへの H2D**（`stream.memcpy_htod(data, dst)` の
/// pinned staging 版）。`memory.rs::upload_into`（`Device` 分岐）から
/// 呼ばれる（`DeviceParamStore::step` の毎 step grad staging 書き込み・
/// `register_resident_leaves` の初期化）。
///
/// フラグ・空データの扱いは [`upload_new`] と同一（フラグ OFF・空
/// データ時は `stream.memcpy_htod(data, dst)` をそのまま呼ぶ）。
pub(crate) fn upload_into<Dst: DevicePtrMut<f32>>(
    stream: &Arc<CudaStream>,
    cache: &Mutex<H2dStagingCache>,
    ctx: &Arc<CudaContext>,
    generation: u64,
    data: &[f32],
    dst: &mut Dst,
) -> Result<(), CudaError> {
    if !pinned_h2d_enabled() || data.is_empty() {
        stream.memcpy_htod(data, dst)?;
        return Ok(());
    }
    let numel = data.len();
    let mut staging = take_or_alloc_h2d(cache, ctx, numel, generation)?;
    staging.as_mut_slice()?.copy_from_slice(data);
    match &staging {
        HostStaging::Pinned(p) => stream.memcpy_htod(p, dst)?,
        HostStaging::Pageable(v) => stream.memcpy_htod(v.as_slice(), dst)?,
    }
    put_back_h2d(cache, numel, generation, staging);
    Ok(())
}

#[cfg(test)]
mod h2d_staging_tests {
    use super::*;

    /// フラグはプロセスグローバルのため、他のテストとの競合を避けて
    /// 直列化・原状復帰する RAII ガード（`crate::placement::tests::
    /// FlagGuard` と同型）。
    struct FlagGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        original: bool,
    }

    impl FlagGuard {
        fn acquire() -> Self {
            let lock = pinned_h2d_test_support::pinned_h2d_flag_test_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let original = pinned_h2d_enabled();
            Self {
                _lock: lock,
                original,
            }
        }
    }

    impl Drop for FlagGuard {
        fn drop(&mut self) {
            set_pinned_h2d_enabled(self.original);
        }
    }

    /// drift ガード: `PINNED_H2D_DEFAULT_ENABLED`（`PINNED_H2D_ENABLED`
    /// の唯一の初期値供給源）の宣言値が意図せず `true` へ差し戻されて
    /// いないかを機械的に検知する（opt-in 契約。イシュー #1585・
    /// codex-review P2 是正）。
    ///
    /// `set_pinned_h2d_enabled`／`pinned_h2d_enabled` を一切呼ばず定数
    /// 自体を直接検査する点が旧実装との違い: 旧実装は `assert!` 直前で
    /// `set_pinned_h2d_enabled(false)` を呼んでいたため、たとえ
    /// `static` の初期値（コンパイル時定数）が `true` へ書き換えられて
    /// いても、実行時に明示的な `false` 上書きでテストが偽陽性 pass
    /// してしまい drift を検知できなかった。本テストは他テストの実行
    /// 順序・`FlagGuard` の有無に関わらず常に同じ結果になる（プロセス
    /// グローバルな `AtomicBool` の実行時状態を一切参照しないため）。
    #[test]
    fn default_h2d_enabled_matches_declared_default() {
        // `std::hint::black_box` は clippy `assertions_on_constants`
        // （定数評価済みの assert は無意味という lint）を、意図どおり
        // 「コンパイル時定数の値そのもの」を検査対象にしたまま回避する
        // （`crate::split_k_runtime::SPLIT_K_DEFAULT_ENABLED` の drift
        // ガードテストと同じ手当て。定数畳み込みを抑止するだけで、
        // 検査対象の値自体は変えない）。
        assert!(!std::hint::black_box(PINNED_H2D_DEFAULT_ENABLED));
    }

    #[test]
    fn set_true_then_false_round_trips() {
        let _guard = FlagGuard::acquire();
        set_pinned_h2d_enabled(true);
        assert!(pinned_h2d_enabled());
        set_pinned_h2d_enabled(false);
        assert!(!pinned_h2d_enabled());
    }

    #[test]
    fn cache_take_miss_then_put_then_take_hit() {
        let mut cache = H2dStagingCache::new();
        assert!(cache.take(4, 0).is_none());
        assert_eq!(cache.stats().misses, 1);

        cache.put(4, 0, HostStaging::Pageable(vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(cache.stats().cached_bytes, 16);

        let hit = cache.take(4, 0).expect("同一 numel・世代は hit するはず");
        assert_eq!(hit.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().cached_bytes, 0);
    }

    /// 同一 numel に対して A・B（正方 GEMM の 2 入力）を連続して `put`
    /// した場合、2 個とも独立にキャッシュされ、後続の 2 回の `take` が
    /// いずれも hit することを検証する（`H2dStagingCache` が
    /// `HostStagingCache` と異なり複数エントリを保持する設計そのものの
    /// 回帰テスト。正方 GEMM の A/B が同一 numel の場合に 2 個目が毎回
    /// miss してしまう欠陥を防ぐ）。
    #[test]
    fn cache_holds_multiple_entries_for_same_numel() {
        let mut cache = H2dStagingCache::new();
        cache.put(4, 0, HostStaging::Pageable(vec![1.0; 4]));
        cache.put(4, 0, HostStaging::Pageable(vec![2.0; 4]));
        assert_eq!(cache.stats().cached_bytes, 32);

        let first = cache.take(4, 0).expect("2 個目（LIFO で後入れ）は hit");
        assert_eq!(first.as_slice().unwrap(), &[2.0; 4]);
        let second = cache.take(4, 0).expect("1 個目も hit");
        assert_eq!(second.as_slice().unwrap(), &[1.0; 4]);
        assert!(cache.take(4, 0).is_none(), "3 回目は miss");
        assert_eq!(cache.stats().hits, 2);
        assert_eq!(cache.stats().cached_bytes, 0);
    }

    #[test]
    fn cache_take_discards_entry_with_mismatched_generation() {
        let mut cache = H2dStagingCache::new();
        cache.put(4, 0, HostStaging::Pageable(vec![0.0; 4]));

        let result = cache.take(4, 1);
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().evicted, 1);
        assert_eq!(cache.stats().cached_bytes, 0);
        assert!(cache.take(4, 0).is_none());
    }

    #[test]
    fn cache_put_discards_entries_exceeding_cap() {
        let mut cache = H2dStagingCache::with_cap(16);
        cache.put(4, 0, HostStaging::Pageable(vec![0.0; 4]));
        assert_eq!(cache.stats().cached_bytes, 16);

        cache.put(2, 0, HostStaging::Pageable(vec![0.0; 2]));
        assert_eq!(cache.stats().cached_bytes, 16, "cap 超過分は登録されない");
        assert_eq!(cache.stats().evicted, 1);
        assert!(cache.take(2, 0).is_none());
    }

    #[test]
    fn release_all_clears_cache_and_returns_freed_bytes() {
        let mut cache = H2dStagingCache::new();
        cache.put(4, 0, HostStaging::Pageable(vec![0.0; 4]));
        cache.put(8, 0, HostStaging::Pageable(vec![0.0; 8]));
        assert_eq!(cache.stats().cached_bytes, 48);

        let freed = cache.release_all();
        assert_eq!(freed, 48);
        assert_eq!(cache.stats().cached_bytes, 0);
        assert!(cache.take(4, 0).is_none());
        assert!(cache.take(8, 0).is_none());
    }

    #[test]
    fn take_or_alloc_and_put_back_roundtrip_via_mutex_poison_recovery() {
        // `Pinned` 種の alloc（miss 経路）は実 `CudaContext` を要求する
        // ため実機テスト側で検証する（`take_or_alloc_h2d` 自体の
        // poison 回復・`put_back_h2d` の往復は `H2dStagingCache` の
        // `take`／`put`（上記テスト群）で検証済み）。ここでは poison
        // 後も `put_back_h2d` が panic しないことのみを確認する。
        let cache = Mutex::new(H2dStagingCache::new());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = cache.lock().unwrap();
            panic!("deliberately poison the mutex");
        }));
        assert!(result.is_err());
        assert!(cache.is_poisoned());

        put_back_h2d(&cache, 4, 0, HostStaging::Pageable(vec![1.0; 4]));
        assert!(cache.lock().is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `HostStaging::alloc(Pinned, ..)` の miss→alloc 経路そのものは
    // `CudaContext`（実 driver 初期化）を要求するため CUDA 実機テスト側
    // （`tests/host_view_real_device.rs`）で検証する。本モジュールの
    // GPU 非依存テストは `HostStaging::Pageable` の直接構築・
    // `HostStagingCache` の `take`／`put` ロジック・poison 処理に限定
    // する。

    #[test]
    fn host_staging_kind_variants_are_distinct() {
        // `HostStagingKind::Pinned` という enum 値自体は `HOST_STAGING_
        // KIND`（本番既定・#1478）としても構築されるため GPU 非依存の
        // 通常ビルドでも複数箇所で構築されるが、`HostStaging::alloc`
        // が実際に `CudaContext::alloc_pinned` を呼ぶ経路は実機
        // （`tests/host_view_real_device.rs`）でしか検証できない。
        // 本テストは `Eq`／`Clone` 実装そのものの契約検証を兼ねる。
        assert_ne!(HostStagingKind::Pinned, HostStagingKind::Pageable);
        assert_eq!(HostStagingKind::Pinned, HostStagingKind::Pinned.clone());
    }

    /// drift ガード: `HOST_STAGING_KIND` の本番既定が意図せず
    /// `Pageable` へ差し戻されていないかを機械的に検知する（イシュー
    /// #1478・2026-09-09 ユーザー承認。`docs/perf/cuda-host-view-
    /// staging-readout.md` §8 実測記録）。既定を変更する場合は本
    /// テストの期待値・上記 doc・モジュール冒頭コメントを合わせて
    /// 更新すること。
    #[test]
    fn default_host_staging_kind_is_pinned() {
        assert_eq!(HOST_STAGING_KIND, HostStagingKind::Pinned);
    }

    #[test]
    fn pageable_staging_is_zero_filled_and_correct_length() {
        let staging = HostStaging::Pageable(vec![0.0f32; 8]);
        assert_eq!(staging.len(), 8);
        assert_eq!(staging.byte_len(), 32);
        assert_eq!(staging.as_slice().unwrap(), &[0.0f32; 8]);
    }

    #[test]
    fn cache_take_miss_then_put_then_take_hit() {
        let mut cache = HostStagingCache::new(HostStagingKind::Pageable);
        assert!(cache.take(4, 0).is_none());
        assert_eq!(cache.stats().misses, 1);

        cache.put(4, 0, HostStaging::Pageable(vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(cache.stats().cached_bytes, 16);

        let hit = cache.take(4, 0).expect("同一 numel・世代は hit するはず");
        assert_eq!(hit.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(
            cache.stats().cached_bytes,
            0,
            "take 後は cached_bytes から差し引かれる"
        );
    }

    #[test]
    fn cache_take_discards_entry_with_mismatched_generation() {
        let mut cache = HostStagingCache::new(HostStagingKind::Pageable);
        cache.put(4, 0, HostStaging::Pageable(vec![0.0; 4]));

        // 世代 1 で取り出そうとすると、格納時の世代 0 と不一致のため
        // 破棄されて miss 扱いになる（fail-closed。`invalidate` 後の
        // 旧世代バッファ誤使用防止）。
        let result = cache.take(4, 1);
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().evicted, 1);
        assert_eq!(cache.stats().cached_bytes, 0);

        // 破棄済みのため、同じ numel を世代 0 で再度取り出しても miss。
        assert!(cache.take(4, 0).is_none());
    }

    #[test]
    fn cache_take_discards_entry_with_mismatched_length() {
        // `HashMap` のキーは numel のため通常長さは一致するが、内部
        // 不整合（本来到達しない防御的経路）を模して直接検証する。
        let mut cache = HostStagingCache::new(HostStagingKind::Pageable);
        cache.entries.insert(
            4,
            StagingEntry {
                buf: HostStaging::Pageable(vec![0.0; 3]),
                generation: 0,
            },
        );
        assert!(cache.take(4, 0).is_none());
        assert_eq!(cache.stats().evicted, 1);
    }

    #[test]
    fn cache_put_discards_entries_exceeding_cap() {
        let mut cache = HostStagingCache::with_cap(HostStagingKind::Pageable, 16);
        // 16 バイト（numel=4）はちょうど収まる。
        cache.put(4, 0, HostStaging::Pageable(vec![0.0; 4]));
        assert_eq!(cache.stats().cached_bytes, 16);

        // 追加で 8 バイト（numel=2）を put すると cap（16 バイト）を
        // 超過するため破棄される（既存エントリは維持）。
        cache.put(2, 0, HostStaging::Pageable(vec![0.0; 2]));
        assert_eq!(cache.stats().cached_bytes, 16, "cap 超過分は登録されない");
        assert_eq!(cache.stats().evicted, 1);
        assert!(cache.take(2, 0).is_none(), "cap 超過で破棄されたため miss");
    }

    #[test]
    fn cache_put_same_numel_twice_does_not_overcount_cached_bytes() {
        // 回帰テスト（codex-review P2 指摘・Cursor Bugbot Medium 指摘。
        // 同一箇所・同一問題）: 同一 numel キーへ `take` を挟まず 2 回
        // `put` すると、`HashMap::insert` が既存エントリを置換する際に
        // 旧エントリ分のバイト数を差し引かずに新規分だけ加算していた
        // ため `cached_bytes` が実体（`entries` の総バイト数）より過大
        // 計上されていた。修正後は置換時に旧分を差し引くため、2 回目の
        // `put` 後も `cached_bytes` は最新エントリの実バイト数と一致
        // する。
        let mut cache = HostStagingCache::new(HostStagingKind::Pageable);
        cache.put(4, 0, HostStaging::Pageable(vec![0.0; 4]));
        assert_eq!(cache.stats().cached_bytes, 16);

        // take を挟まず同一 numel へ再入で put（並行 with_host_view を
        // 模す）。旧エントリが破棄され新エントリに置換される。
        cache.put(4, 1, HostStaging::Pageable(vec![1.0; 4]));
        assert_eq!(
            cache.stats().cached_bytes,
            16,
            "同一 numel の置換では合計バイト数は変わらないはず（過大計上の回帰検知）"
        );

        // 実体との整合を release_all の返却値でも確認する。
        let freed = cache.release_all();
        assert_eq!(freed, 16, "release_all の返却値も実体と一致するはず");
    }

    #[test]
    fn cache_put_same_numel_twice_respects_cap_with_replacement() {
        // 置換時に旧分を差し引いたうえで cap 判定することを検証する
        // （旧分を差し引かずに加算のみだと、実際には cap 内に収まる
        // 置換でも不当に拒否されうる）。
        let mut cache = HostStagingCache::with_cap(HostStagingKind::Pageable, 16);
        cache.put(4, 0, HostStaging::Pageable(vec![0.0; 4]));
        assert_eq!(cache.stats().cached_bytes, 16);

        // 同一 numel（同じ 16 バイト）への置換は cap ちょうどのため
        // 受理されるはず（旧分を差し引かず加算のみだと 32 > 16 で
        // 不当に拒否されていた）。
        cache.put(4, 1, HostStaging::Pageable(vec![1.0; 4]));
        assert_eq!(
            cache.stats().cached_bytes,
            16,
            "置換は cap 内で受理されるはず"
        );
        assert_eq!(
            cache.stats().evicted,
            0,
            "置換は cap 超過ではないため evicted は増えない"
        );

        let hit = cache.take(4, 1).expect("置換後の世代 1 で take できるはず");
        assert_eq!(hit.as_slice().unwrap(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn release_all_clears_cache_and_returns_freed_bytes() {
        let mut cache = HostStagingCache::new(HostStagingKind::Pageable);
        cache.put(4, 0, HostStaging::Pageable(vec![0.0; 4]));
        cache.put(8, 0, HostStaging::Pageable(vec![0.0; 8]));
        assert_eq!(cache.stats().cached_bytes, 48);

        let freed = cache.release_all();
        assert_eq!(freed, 48);
        assert_eq!(cache.stats().cached_bytes, 0);
        assert!(cache.take(4, 0).is_none());
        assert!(cache.take(8, 0).is_none());
    }

    #[test]
    fn take_or_alloc_and_put_back_roundtrip_via_mutex() {
        let cache = std::sync::Mutex::new(HostStagingCache::new(HostStagingKind::Pageable));

        // `Pageable` 種は `ctx` を参照しないため、実機なしでも
        // `take_or_alloc` の miss→alloc 経路を検証できる。`CudaContext`
        // を構築せずダミーの `Arc` を渡すことはできない（型が要求する）
        // ため、本経路の GPU 非依存検証は `HostStagingCache` 単体の
        // `take`／`put`（上記テスト群）で行い、`take_or_alloc`／
        // `put_back` の poison 処理のみを別途検証する。
        {
            let mut guard = cache.lock().unwrap();
            assert!(guard.take(4, 0).is_none());
            guard.put(4, 0, HostStaging::Pageable(vec![9.0; 4]));
        }
        {
            let mut guard = cache.lock().unwrap();
            let hit = guard.take(4, 0).unwrap();
            assert_eq!(hit.as_slice().unwrap(), &[9.0f32; 4]);
        }
    }

    #[test]
    fn put_back_recovers_from_poisoned_mutex() {
        // Mutex poison 後も `into_inner` で内部データへアクセスし続け、
        // 呼び出し元（`memory.rs`）の `lock()` 直接呼び出しが poison を
        // 検出して `BackendError` を返す一方、`put_back`／`take_or_alloc`
        // 自体は panic しないことを確認する（`.claude/rules/coding-rust.md`
        // 「本番経路で unwrap/expect を使わない」の対偶: poison からの
        // 復旧経路が panic しないこと自体は許容される設計上の緩和策）。
        let cache = std::sync::Mutex::new(HostStagingCache::new(HostStagingKind::Pageable));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = cache.lock().unwrap();
            panic!("deliberately poison the mutex");
        }));
        assert!(result.is_err());
        assert!(cache.is_poisoned());

        // poison 後でも put_back は panic せずに完了する。
        put_back(&cache, 4, 0, HostStaging::Pageable(vec![1.0; 4]));

        // 直接 `lock()` を呼ぶ経路（`memory.rs` が実際に使う経路）は
        // `Err(PoisonError)` を観測できる（呼び出し元が
        // `BackendError::DeviceUnavailable` へ変換する契約）。
        assert!(cache.lock().is_err());
    }
}
