//! デバイス常駐 KV（K-3）の CPU 参照実装（イシュー #3110・親 #3108。
//! 設計正本 `docs/device-resident-kv-paged-attention-design.md` §2・§3・§5・§7・§8）。
//!
//! # 役割と責務境界
//!
//! K-1 の [`super::KvCache`]（ホスト `Tensor<f32>` を `Var::cat` で毎回
//! 再確保して連結する）に対し、K-3 は K／V をブロック（ページ）単位の
//! プールへ置き、追記で再確保しない構造にする。本モジュールはそのページ
//! 構造を **CPU メモリ上で再現する参照実装**であり、後続のデバイス版
//! `PagedKvCache`（CUDA・Metal。#3111）・facade 公開（#3112）・実機
//! parity テスト（#3113）の期待値生成器を兼ねる。`BackendOps`／
//! `MemoryOps`・`KvCache`・facade 公開面には触れない。
//!
//! - [`PagedKvCacheConfig`]: `block_size`／`max_blocks`／`embed_dim` の検査つき設定
//!   （デバイス版と共有できる形。設計 §6）
//! - `BlockTable`（非公開）: ページテーブルと空きブロック管理。バックエンドに
//!   依存しないホストロジックなので #3111 のデバイス版が再利用する
//! - [`PagedKvCacheReference`]: 1 層ぶんのホスト参照キャッシュ
//!   （追記・読み出し・巻き戻し）。多層は `Vec<PagedKvCacheReference>`
//! - [`paged_attention_reference`]: ブロックを論理順に走査する attention
//!
//! # レイアウト
//!
//! K／V プールとも `[max_blocks, block_size, E]` の行優先 `Vec<f32>`
//! （設計 §3）。論理位置 `p` の行は `table[p / block_size]` 番ブロックの
//! `p % block_size` 行目にあり、K-1 の `[B, S, E]` と行の並びが揃う。
//!
//! # 縮約順序と判定方式（設計 §3・§5）
//!
//! [`paged_attention_reference`] は各 query 行について key 位置を論理順に
//! たどる**単一の連続ループ**で score・softmax・加重和を求める。したがって
//! REQ-2 の「結合順序が単一の連続 K ループと異なるカーネル」には当たらず、
//! K-1 との比較は統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5
//! 未満）をそのまま適用する。flash-decoding 型の split を導入する場合は
//! baseline 非後退方式の対象になる。FMA 契約は `f32::mul_add`
//! （`coding-rust.md`）。score の Dh 方向累積と加重和は `mul_add` で行い、
//! K-1（`sdpa_compose`）に合わせて query 側へ先に scale を掛ける。
//!
//! # 安全性（OWASP A03・A04、REQ-8）
//!
//! 新規 `unsafe` を持たない（下の `forbid`）。確保サイズは `checked_mul` と
//! `isize::MAX` で検査し、`try_reserve` で OOM を `Err` にする。添字アクセスは
//! すべて `get`／`get_mut` で行い、範囲外は panic ではなく `Err`。変更操作は
//! 先にすべて検証してから書き込み、失敗時は状態を変えない。

#![forbid(unsafe_code)]

use crate::error::AutodiffError;
use fandhe_ai_tensor_core::Tensor;

/// `block_size` の既定値（設計 §6。vLLM 既定と同じ 16）。
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// `block_size` の上限（設計 §6。2 のべき乗で 1〜256）。
const MAX_BLOCK_SIZE: usize = 256;

pub(crate) fn invalid(msg: impl Into<String>) -> AutodiffError {
    AutodiffError::InvalidArgument(format!("paged_kv: {}", msg.into()))
}

pub(crate) fn reserve_err(what: &str) -> AutodiffError {
    invalid(format!("{what} の確保に失敗した（メモリ不足）"))
}

/// ページ化 KV キャッシュの設定（設計 §6）。[`PagedKvCacheConfig::new`] が
/// 唯一の構築経路で、検査を通った値のみを保持する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvCacheConfig {
    block_size: usize,
    max_blocks: usize,
    embed_dim: usize,
}

impl PagedKvCacheConfig {
    /// 設定を検査して作る。
    ///
    /// - `block_size`: 2 のべき乗で 1〜256
    /// - `max_blocks`: 1 以上 `u32::MAX` 以下（ブロック ID が `u32`。0・無制限は不可）
    /// - `embed_dim`: 1 以上
    /// - K／V 2 本ぶんのプールのバイト数が `isize::MAX` 以下（`checked_mul` で検査）
    ///
    /// 検査だけを行い、確保はしない。
    pub fn new(
        block_size: usize,
        max_blocks: usize,
        embed_dim: usize,
    ) -> Result<PagedKvCacheConfig, AutodiffError> {
        if block_size == 0 || block_size > MAX_BLOCK_SIZE || !block_size.is_power_of_two() {
            return Err(invalid(format!(
                "block_size は 2 のべき乗で 1〜{MAX_BLOCK_SIZE} でなければならない（実際: {block_size}）"
            )));
        }
        if max_blocks == 0 || u64::try_from(max_blocks).map_or(true, |m| m > u64::from(u32::MAX)) {
            return Err(invalid(format!(
                "max_blocks は 1 以上 u32::MAX 以下でなければならない（実際: {max_blocks}）"
            )));
        }
        if embed_dim == 0 {
            return Err(invalid("embed_dim は 1 以上でなければならない"));
        }
        let cfg = PagedKvCacheConfig {
            block_size,
            max_blocks,
            embed_dim,
        };
        cfg.total_bytes()?;
        Ok(cfg)
    }

    /// ブロックあたりのトークン数。
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// プール全体のブロック数。
    pub fn max_blocks(&self) -> usize {
        self.max_blocks
    }

    /// 埋め込み次元 `E`。
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// 片側（K または V）のプールの要素数。
    pub(crate) fn pool_elems(&self) -> Result<usize, AutodiffError> {
        self.max_blocks
            .checked_mul(self.block_size)
            .and_then(|n| n.checked_mul(self.embed_dim))
            .ok_or_else(|| invalid("プールの要素数が usize を超える"))
    }

    /// K と V の合計バイト数（`isize::MAX` 超は `Err`）。
    fn total_bytes(&self) -> Result<usize, AutodiffError> {
        let bytes = self
            .pool_elems()?
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| invalid("プールのバイト数が usize を超える"))?;
        if bytes > isize::MAX as usize {
            return Err(invalid("プールのバイト数が isize::MAX を超える"));
        }
        Ok(bytes)
    }
}

/// ページテーブルと空きブロック管理（バックエンド非依存のホストロジック）。
///
/// 空きブロックは「返却済みスタック」と「未使用の連番」の 2 つで表し、
/// 確保は返却済みを優先して後入れ先出し、なければ連番（0 から）を使う。
/// 順序は決定的で、`max_blocks` が大きくても空きリストを eager に作らない。
///
/// デバイス版 `PagedKvCache`（`paged_kv_device.rs`。#3111）も同じ型を再利用する
/// （`pub(crate)`）。デバイス版は追記のたびに本型を複製して
/// [`BlockTable::extend_for_append`] を適用し、バックエンドの書き込みが成功した
/// 後にだけ複製を確定する（失敗時の論理状態不変。設計 §4・OWASP A08）。
#[derive(Debug, Clone, Default)]
pub(crate) struct BlockTable {
    pub(crate) returned: Vec<u32>,
    pub(crate) next_unused: usize,
    pub(crate) tables: Vec<Vec<u32>>,
    pub(crate) seq_lens: Vec<usize>,
    pub(crate) batch: Option<usize>,
}

impl BlockTable {
    pub(crate) fn free_blocks(&self, max_blocks: usize) -> usize {
        self.returned.len() + max_blocks.saturating_sub(self.next_unused)
    }

    /// 空きブロックを 1 つ確保する。空きがなければ `None`（呼び出し元が事前に検査済み）。
    pub(crate) fn alloc(&mut self, max_blocks: usize) -> Option<u32> {
        if let Some(b) = self.returned.pop() {
            return Some(b);
        }
        if self.next_unused < max_blocks {
            let id = u32::try_from(self.next_unused).ok()?;
            self.next_unused += 1;
            return Some(id);
        }
        None
    }

    pub(crate) fn reset(&mut self) {
        self.returned.clear();
        self.next_unused = 0;
        self.tables = Vec::new();
        self.seq_lens = Vec::new();
        self.batch = None;
    }
}

impl BlockTable {
    /// 現在の有効系列長（空なら 0。全行同長の不変条件）。
    pub(crate) fn seq_len(&self) -> usize {
        self.seq_lens.first().copied().unwrap_or(0)
    }

    /// `B` 行・各 `l_new` トークンの追記ぶんのブロックを確保し、行優先
    /// （行 `b`・トークン `i` の順）のスロット写像
    /// （`block_id · block_size + pos % block_size`）を返す。`self` を直接変えるので、
    /// 呼び出し元は**複製に対して**呼び、書き込み成功後に確定すること。
    ///
    /// 検査: `B` の一貫性・系列長／必要ブロック数のオーバーフロー・空きブロック数・
    /// スロットが `u32` に収まること。`l_new == 0` は何もせず空の写像を返す。
    pub(crate) fn extend_for_append(
        &mut self,
        max_blocks: usize,
        block_size: usize,
        b: usize,
        l_new: usize,
    ) -> Result<Vec<u32>, AutodiffError> {
        if b == 0 {
            return Err(invalid("バッチサイズ 0 は追記できない"));
        }
        if let Some(cb) = self.batch
            && cb != b
        {
            return Err(invalid(format!(
                "バッチ {b} がキャッシュの {cb} と一致しない"
            )));
        }
        if l_new == 0 {
            return Ok(Vec::new());
        }
        let s_prev = self.seq_len();
        let s_total = s_prev
            .checked_add(l_new)
            .ok_or_else(|| invalid("系列長がオーバーフローした"))?;
        let held = s_prev.div_ceil(block_size);
        let extra_per_row = s_total.div_ceil(block_size) - held;
        let extra_total = extra_per_row
            .checked_mul(b)
            .ok_or_else(|| invalid("必要ブロック数がオーバーフローした"))?;
        if extra_total > self.free_blocks(max_blocks) {
            return Err(invalid(format!(
                "空きブロック不足（必要 {extra_total}・空き {}）",
                self.free_blocks(max_blocks)
            )));
        }
        let rows = b
            .checked_mul(l_new)
            .ok_or_else(|| invalid("追記行数がオーバーフローした"))?;
        let mut slots: Vec<u32> = Vec::new();
        slots
            .try_reserve_exact(rows)
            .map_err(|_| reserve_err("スロット写像"))?;
        if self.batch.is_none() {
            self.tables = Vec::new();
            self.seq_lens = Vec::new();
            self.tables
                .try_reserve_exact(b)
                .map_err(|_| reserve_err("行テーブル"))?;
            self.seq_lens
                .try_reserve_exact(b)
                .map_err(|_| reserve_err("行長テーブル"))?;
            for _ in 0..b {
                self.tables.push(Vec::new());
                self.seq_lens.push(0);
            }
            self.batch = Some(b);
        }
        for row in self.tables.iter_mut() {
            row.try_reserve(extra_per_row)
                .map_err(|_| reserve_err("行ブロックテーブル"))?;
        }
        for row in 0..b {
            for i in 0..l_new {
                let pos = s_prev + i;
                if pos.is_multiple_of(block_size) {
                    let id = self
                        .alloc(max_blocks)
                        .ok_or_else(|| invalid("内部不整合: 検査済みなのに空きがない"))?;
                    self.tables
                        .get_mut(row)
                        .ok_or_else(|| invalid("内部不整合: 行がない"))?
                        .push(id);
                }
                let blk = *self
                    .tables
                    .get(row)
                    .and_then(|t| t.get(pos / block_size))
                    .ok_or_else(|| invalid("ブロックテーブルの添字が範囲外"))?;
                let slot = usize::try_from(blk)
                    .ok()
                    .and_then(|n| n.checked_mul(block_size))
                    .and_then(|n| n.checked_add(pos % block_size))
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| invalid("スロット番号が u32 に収まらない"))?;
                slots.push(slot);
            }
        }
        for len in self.seq_lens.iter_mut() {
            *len = s_total;
        }
        Ok(slots)
    }

    /// `[B, max_blocks_per_seq]` に平坦化したブロックテーブル（未使用枠は番兵
    /// `u32::MAX`）。`B` 未確定なら空。サイズは `checked_mul` と `try_reserve`。
    pub(crate) fn flatten_tables(
        &self,
        max_blocks_per_seq: usize,
    ) -> Result<Vec<u32>, AutodiffError> {
        let n = self
            .tables
            .len()
            .checked_mul(max_blocks_per_seq)
            .ok_or_else(|| invalid("ブロックテーブルの要素数がオーバーフローした"))?;
        let mut out: Vec<u32> = Vec::new();
        out.try_reserve_exact(n)
            .map_err(|_| reserve_err("平坦化ブロックテーブル"))?;
        for row in &self.tables {
            if row.len() > max_blocks_per_seq {
                return Err(invalid("行のブロック数が max_blocks_per_seq を超える"));
            }
            out.extend_from_slice(row);
            out.resize(out.len() + (max_blocks_per_seq - row.len()), u32::MAX);
        }
        Ok(out)
    }

    /// 有効長を `new_len` へ縮め、余る末尾ブロックを空きへ返す
    /// （[`PagedKvCacheReference::rewind_to`] と同じ意味。失敗時は状態不変）。
    pub(crate) fn rewind_to(
        &mut self,
        new_len: usize,
        block_size: usize,
    ) -> Result<(), AutodiffError> {
        let cur = self.seq_len();
        if new_len > cur {
            return Err(invalid(format!(
                "巻き戻し先 {new_len} が有効長 {cur} を超える"
            )));
        }
        if self.batch.is_none() || new_len == cur {
            return Ok(());
        }
        if new_len == 0 {
            self.reset();
            return Ok(());
        }
        let keep = new_len.div_ceil(block_size);
        let release: usize = self
            .tables
            .iter()
            .map(|t| t.len().saturating_sub(keep))
            .sum();
        self.returned
            .try_reserve(release)
            .map_err(|_| reserve_err("空きブロック返却用スタック"))?;
        for row in self.tables.iter_mut() {
            while row.len() > keep {
                if let Some(id) = row.pop() {
                    self.returned.push(id);
                }
            }
        }
        for len in self.seq_lens.iter_mut() {
            *len = new_len;
        }
        Ok(())
    }
}

/// 1 層ぶんの CPU 参照ページ化 KV キャッシュ（K-3。設計 §2）。
///
/// デバイス版 `PagedKvCache`（#3111）の期待値を作るホスト実装。名前を
/// `…Reference` にして、デバイス版の名前は #3111／#3112 に残している。
/// `Clone` は全ページをコピーする（O(S)）。K-1 の案 (i)（スナップショット
/// 退避）と同じ使い方ができる。
///
/// 巻き戻し [`PagedKvCacheReference::rewind_to`] の facade への公開形は
/// #3112 が決める。`KvCache` の巻き戻しは #3095／#3096 の決定どおり非公開の
/// `kv_rewind` のままで、本型はそれに干渉しない。
#[derive(Debug, Clone)]
pub struct PagedKvCacheReference {
    config: PagedKvCacheConfig,
    k_pages: Vec<f32>,
    v_pages: Vec<f32>,
    table: BlockTable,
}

impl PagedKvCacheReference {
    /// プールを確保して空のキャッシュを作る。メモリ不足は abort でなく `Err`。
    pub fn new(config: PagedKvCacheConfig) -> Result<PagedKvCacheReference, AutodiffError> {
        let n = config.pool_elems()?;
        let mut k_pages: Vec<f32> = Vec::new();
        k_pages
            .try_reserve_exact(n)
            .map_err(|_| reserve_err("K プール"))?;
        k_pages.resize(n, 0.0);
        let mut v_pages: Vec<f32> = Vec::new();
        v_pages
            .try_reserve_exact(n)
            .map_err(|_| reserve_err("V プール"))?;
        v_pages.resize(n, 0.0);
        Ok(PagedKvCacheReference {
            config,
            k_pages,
            v_pages,
            table: BlockTable::default(),
        })
    }

    /// 設定。
    pub fn config(&self) -> &PagedKvCacheConfig {
        &self.config
    }

    /// ブロックあたりのトークン数。
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// 埋め込み次元 `E`。
    pub fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    /// 有効系列長（空なら 0）。現行は全行の長さが等しい不変条件。
    pub fn seq_len(&self) -> usize {
        self.table.seq_lens.first().copied().unwrap_or(0)
    }

    /// バッチサイズ `B`（最初の追記で確定。空なら `None`）。
    pub fn batch(&self) -> Option<usize> {
        self.table.batch
    }

    /// 使用中のブロック総数（全行の合計）。
    pub fn num_used_blocks(&self) -> usize {
        self.table.tables.iter().map(Vec::len).sum()
    }

    /// 空きブロック数。
    pub fn num_free_blocks(&self) -> usize {
        self.table.free_blocks(self.config.max_blocks)
    }

    /// 行 `row` のブロックテーブル（論理順のブロック ID）。
    pub fn block_table(&self, row: usize) -> Option<&[u32]> {
        self.table.tables.get(row).map(Vec::as_slice)
    }

    /// 論理位置 `pos`（行 `row`）の K／V プール内の先頭要素オフセット。
    /// 位置が有効長以上・ブロック ID が範囲外なら `Err`（手動境界検査）。
    fn row_offset(&self, row: usize, pos: usize) -> Result<usize, AutodiffError> {
        let len = *self
            .table
            .seq_lens
            .get(row)
            .ok_or_else(|| invalid("行番号が範囲外"))?;
        if pos >= len {
            return Err(invalid("位置が有効長以上"));
        }
        let bs = self.config.block_size;
        let blk = *self
            .table
            .tables
            .get(row)
            .and_then(|t| t.get(pos / bs))
            .ok_or_else(|| invalid("ブロックテーブルの添字が範囲外"))? as usize;
        if blk >= self.config.max_blocks {
            return Err(invalid("ブロック ID が max_blocks 以上"));
        }
        blk.checked_mul(bs)
            .and_then(|n| n.checked_add(pos % bs))
            .and_then(|n| n.checked_mul(self.config.embed_dim))
            .ok_or_else(|| invalid("オフセットがオーバーフローした"))
    }

    /// 新トークンの K／V（射影済み `[B, L_new, E]`）を末尾へ追記する。
    ///
    /// 検証（rank・shape 一致・`E`・`B`・長さオーバーフロー・空きブロック数・
    /// 行テーブルの確保）をすべて通してから書き込む。`Err` のとき状態は変わらない。
    /// `L_new == 0` は検証だけして `Ok`（`B` も確定しない）。
    pub fn append(
        &mut self,
        k_new: &Tensor<f32>,
        v_new: &Tensor<f32>,
    ) -> Result<(), AutodiffError> {
        let shape = k_new.shape().to_vec();
        if shape.len() != 3 {
            return Err(invalid(format!(
                "k_new は rank 3 が必要（実際: {}）",
                shape.len()
            )));
        }
        if v_new.shape() != shape.as_slice() {
            return Err(invalid("k_new と v_new の shape が一致しない"));
        }
        let (b, l_new, e) = (shape[0], shape[1], shape[2]);
        if e != self.config.embed_dim {
            return Err(invalid(format!(
                "最終軸 {e} が embed_dim {} と一致しない",
                self.config.embed_dim
            )));
        }
        if b == 0 {
            return Err(invalid("バッチサイズ 0 は追記できない"));
        }
        if let Some(cb) = self.table.batch
            && cb != b
        {
            return Err(invalid(format!(
                "バッチ {b} がキャッシュの {cb} と一致しない"
            )));
        }
        if l_new == 0 {
            return Ok(());
        }
        let bs = self.config.block_size;
        let s_prev = self.seq_len();
        let s_total = s_prev
            .checked_add(l_new)
            .ok_or_else(|| invalid("系列長がオーバーフローした"))?;
        let held = s_prev.div_ceil(bs);
        let extra_per_row = s_total.div_ceil(bs) - held;
        let extra_total = extra_per_row
            .checked_mul(b)
            .ok_or_else(|| invalid("必要ブロック数がオーバーフローした"))?;
        if extra_total > self.num_free_blocks() {
            return Err(invalid(format!(
                "空きブロック不足（必要 {extra_total}・空き {}）",
                self.num_free_blocks()
            )));
        }

        // 行テーブルの確保など、失敗しうる処理をすべて書き込み前に済ませる。
        let mut fresh_tables: Vec<Vec<u32>> = Vec::new();
        let mut fresh_lens: Vec<usize> = Vec::new();
        if self.table.batch.is_none() {
            fresh_tables
                .try_reserve_exact(b)
                .map_err(|_| reserve_err("行テーブル"))?;
            fresh_lens
                .try_reserve_exact(b)
                .map_err(|_| reserve_err("行長テーブル"))?;
            for _ in 0..b {
                fresh_tables.push(Vec::new());
                fresh_lens.push(0);
            }
            for row in fresh_tables.iter_mut() {
                row.try_reserve_exact(extra_per_row)
                    .map_err(|_| reserve_err("行ブロックテーブル"))?;
            }
        } else {
            for row in self.table.tables.iter_mut() {
                row.try_reserve(extra_per_row)
                    .map_err(|_| reserve_err("行ブロックテーブル"))?;
            }
        }
        let kc = k_new.contiguous();
        let vc = v_new.contiguous();
        let (ks, vs) = match (kc.as_slice(), vc.as_slice()) {
            (Some(k), Some(v)) => (k, v),
            _ => return Err(invalid("入力を contiguous にできなかった")),
        };

        // ここから先は失敗しない（事前検査済み）。
        if self.table.batch.is_none() {
            self.table.tables = fresh_tables;
            self.table.seq_lens = fresh_lens;
            self.table.batch = Some(b);
        }
        let max_blocks = self.config.max_blocks;
        for row in 0..b {
            for i in 0..l_new {
                let pos = s_prev + i;
                if pos.is_multiple_of(bs) {
                    let id = self
                        .table
                        .alloc(max_blocks)
                        .ok_or_else(|| invalid("内部不整合: 検査済みなのに空きがない"))?;
                    self.table
                        .tables
                        .get_mut(row)
                        .ok_or_else(|| invalid("内部不整合: 行がない"))?
                        .push(id);
                }
                // `row_offset` は有効長を見るため、書き込み中は先に長さを進める。
                if let Some(len) = self.table.seq_lens.get_mut(row) {
                    *len = pos + 1;
                }
                let dst = self.row_offset(row, pos)?;
                let src = (row * l_new + i) * e;
                let (src_k, src_v) = (
                    ks.get(src..src + e)
                        .ok_or_else(|| invalid("K 入力の範囲外"))?,
                    vs.get(src..src + e)
                        .ok_or_else(|| invalid("V 入力の範囲外"))?,
                );
                self.k_pages
                    .get_mut(dst..dst + e)
                    .ok_or_else(|| invalid("K プールの範囲外"))?
                    .copy_from_slice(src_k);
                self.v_pages
                    .get_mut(dst..dst + e)
                    .ok_or_else(|| invalid("V プールの範囲外"))?
                    .copy_from_slice(src_v);
            }
        }
        Ok(())
    }

    /// 論理順に並べた K／V（それぞれ `[B, S, E]`）を返す（ビットコピー）。
    /// 空キャッシュ（`B` 未確定または `S == 0`）は `Err`。
    pub fn gather(&self) -> Result<(Tensor<f32>, Tensor<f32>), AutodiffError> {
        let b = self
            .table
            .batch
            .ok_or_else(|| invalid("空キャッシュは gather できない"))?;
        let s = self.seq_len();
        let e = self.config.embed_dim;
        let n = b
            .checked_mul(s)
            .and_then(|n| n.checked_mul(e))
            .ok_or_else(|| invalid("gather の要素数がオーバーフローした"))?;
        let mut kd: Vec<f32> = Vec::new();
        let mut vd: Vec<f32> = Vec::new();
        kd.try_reserve_exact(n)
            .map_err(|_| reserve_err("gather K"))?;
        vd.try_reserve_exact(n)
            .map_err(|_| reserve_err("gather V"))?;
        for row in 0..b {
            for pos in 0..s {
                let off = self.row_offset(row, pos)?;
                kd.extend_from_slice(
                    self.k_pages
                        .get(off..off + e)
                        .ok_or_else(|| invalid("K プールの範囲外"))?,
                );
                vd.extend_from_slice(
                    self.v_pages
                        .get(off..off + e)
                        .ok_or_else(|| invalid("V プールの範囲外"))?,
                );
            }
        }
        let shape = [b, s, e];
        Ok((
            Tensor::new(kd, &shape).map_err(AutodiffError::Shape)?,
            Tensor::new(vd, &shape).map_err(AutodiffError::Shape)?,
        ))
    }

    /// 有効長を `new_len` へ縮め、余る末尾ブロックを空きへ返す。
    /// `new_len > seq_len` は `Err`（状態不変）。`new_len == 0` は `B` の確定も
    /// 解除して [`PagedKvCacheReference::clear`] と同じ状態にする。
    pub fn rewind_to(&mut self, new_len: usize) -> Result<(), AutodiffError> {
        let cur = self.seq_len();
        if new_len > cur {
            return Err(invalid(format!(
                "巻き戻し先 {new_len} が有効長 {cur} を超える"
            )));
        }
        if self.table.batch.is_none() || new_len == cur {
            return Ok(());
        }
        if new_len == 0 {
            self.clear();
            return Ok(());
        }
        let keep = new_len.div_ceil(self.config.block_size);
        let release: usize = self
            .table
            .tables
            .iter()
            .map(|t| t.len().saturating_sub(keep))
            .sum();
        self.table
            .returned
            .try_reserve(release)
            .map_err(|_| reserve_err("空きブロック返却用スタック"))?;
        for row in self.table.tables.iter_mut() {
            while row.len() > keep {
                if let Some(id) = row.pop() {
                    self.table.returned.push(id);
                }
            }
        }
        for len in self.table.seq_lens.iter_mut() {
            *len = new_len;
        }
        Ok(())
    }

    /// すべてのブロックを空きへ戻し、`B` を未確定に戻す。
    pub fn clear(&mut self) {
        self.table.reset();
    }
}

/// ページ化キャッシュに対する attention の CPU 参照（期待値生成ユーティリティ）。
///
/// `q` は射影済み `[B, L_new, E]`、戻り値は head を結合した `[B, L_new, E]`
/// （`out_proj` 適用前）。呼び出し側は新トークンの K／V を
/// [`PagedKvCacheReference::append`] した**後**に呼ぶ（`s_prev = seq_len − L_new`）。
/// mask は「位置 `j` に attend できるのは `j <= s_prev + i`」の 1 規則で、
/// K-1 の (a) prefill（causal）・(b) `L_new == 1`・(c) 複数トークン追記を
/// すべて再現する。`L_new == 0` は `[B, 0, E]` を返す。
///
/// 数値順序: query に先に `scale` を掛け、score は Dh 方向に `f32::mul_add` で
/// 累積、位置を論理順にたどる単一の連続ループで max → `exp(s − max)` の和 →
/// 正規化重みの `mul_add` 加重和を求める。結合順序が単一の連続 K ループと
/// 異ならないので統一複合判定をそのまま適用できる（モジュール doc 参照）。
///
/// 違反（rank・`B`／`E` 不一致・`num_heads`・`scale` 非有限・`L_new > seq_len`・
/// ブロックテーブル異常）はすべて `Err`で、panic しない。実機 parity
/// テスト（#3113）は autodiff の `nn` から本関数を呼べる。backend-cuda に
/// autodiff の dev-dependency を足すかは #3111／#3113 が判断する。
pub fn paged_attention_reference(
    q: &Tensor<f32>,
    cache: &PagedKvCacheReference,
    num_heads: usize,
    scale: f32,
) -> Result<Tensor<f32>, AutodiffError> {
    let shape = q.shape().to_vec();
    if shape.len() != 3 {
        return Err(invalid(format!(
            "q は rank 3 が必要（実際: {}）",
            shape.len()
        )));
    }
    let (b, l_new, e) = (shape[0], shape[1], shape[2]);
    if e != cache.config.embed_dim {
        return Err(invalid("q の最終軸がキャッシュの embed_dim と一致しない"));
    }
    if num_heads == 0 || e % num_heads != 0 {
        return Err(invalid(
            "num_heads は 1 以上で embed_dim を割り切る必要がある",
        ));
    }
    if !scale.is_finite() {
        return Err(invalid("scale は有限値でなければならない"));
    }
    if let Some(cb) = cache.table.batch
        && cb != b
    {
        return Err(invalid("q のバッチがキャッシュと一致しない"));
    }
    let s_total = cache.seq_len();
    if l_new > s_total {
        return Err(invalid(format!(
            "L_new {l_new} が有効長 {s_total} を超える（先に append すること）"
        )));
    }
    let n_out = b
        .checked_mul(l_new)
        .and_then(|n| n.checked_mul(e))
        .ok_or_else(|| invalid("出力の要素数がオーバーフローした"))?;
    if l_new == 0 || b == 0 {
        return Tensor::new(Vec::new(), &[b, l_new, e]).map_err(AutodiffError::Shape);
    }
    let qc = q.contiguous();
    let qs = qc
        .as_slice()
        .ok_or_else(|| invalid("q を contiguous にできなかった"))?;
    let dh = e / num_heads;
    let s_prev = s_total - l_new;

    let mut out: Vec<f32> = Vec::new();
    out.try_reserve_exact(n_out)
        .map_err(|_| reserve_err("出力"))?;
    out.resize(n_out, 0.0);
    let mut weights: Vec<f32> = Vec::new();
    weights
        .try_reserve_exact(s_total)
        .map_err(|_| reserve_err("score バッファ"))?;
    weights.resize(s_total, 0.0);

    for bi in 0..b {
        for h in 0..num_heads {
            for i in 0..l_new {
                let q_off = (bi * l_new + i) * e + h * dh;
                let q_row = qs
                    .get(q_off..q_off + dh)
                    .ok_or_else(|| invalid("q の範囲外"))?;
                let limit = s_prev + i + 1;
                // 第 1 パス: score と最大値（論理順）。
                let mut max = f32::NEG_INFINITY;
                for j in 0..limit {
                    let off = cache.row_offset(bi, j)? + h * dh;
                    let k_row = cache
                        .k_pages
                        .get(off..off + dh)
                        .ok_or_else(|| invalid("K プールの範囲外"))?;
                    let mut acc = 0.0f32;
                    for (&qv, &kv) in q_row.iter().zip(k_row) {
                        acc = (qv * scale).mul_add(kv, acc);
                    }
                    if let Some(w) = weights.get_mut(j) {
                        *w = acc;
                    }
                    if acc > max {
                        max = acc;
                    }
                }
                // 第 2 パス: exp(s - max) と和。
                let mut sum = 0.0f32;
                for w in weights.iter_mut().take(limit) {
                    *w = (*w - max).exp();
                    sum += *w;
                }
                // 加重和（正規化した重み）。
                let o_off = (bi * l_new + i) * e + h * dh;
                for j in 0..limit {
                    let p = weights.get(j).copied().unwrap_or(0.0) / sum;
                    let off = cache.row_offset(bi, j)? + h * dh;
                    let v_row = cache
                        .v_pages
                        .get(off..off + dh)
                        .ok_or_else(|| invalid("V プールの範囲外"))?;
                    let o_row = out
                        .get_mut(o_off..o_off + dh)
                        .ok_or_else(|| invalid("出力の範囲外"))?;
                    for (o, &vv) in o_row.iter_mut().zip(v_row) {
                        *o = p.mul_add(vv, *o);
                    }
                }
            }
        }
    }
    Tensor::new(out, &[b, l_new, e]).map_err(AutodiffError::Shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_bad_block_size_and_overflow() {
        assert!(PagedKvCacheConfig::new(0, 4, 8).is_err());
        assert!(PagedKvCacheConfig::new(3, 4, 8).is_err());
        assert!(PagedKvCacheConfig::new(512, 4, 8).is_err());
        assert!(PagedKvCacheConfig::new(256, 4, 8).is_ok());
        assert!(PagedKvCacheConfig::new(4, 0, 8).is_err());
        assert!(PagedKvCacheConfig::new(4, 4, 0).is_err());
        assert!(PagedKvCacheConfig::new(256, u32::MAX as usize, usize::MAX / 4).is_err());
    }

    #[test]
    fn config_byte_bound_is_checked() {
        // 2 * 4 byte * max_blocks * bs * E が isize::MAX を超える。
        let e = (isize::MAX as usize) / (2 * 4 * 256) + 1;
        assert!(PagedKvCacheConfig::new(256, 1, e).is_err());
    }

    #[test]
    fn block_table_alloc_order_is_deterministic() {
        let mut t = BlockTable::default();
        assert_eq!(t.alloc(3), Some(0));
        assert_eq!(t.alloc(3), Some(1));
        t.returned.push(0);
        assert_eq!(t.free_blocks(3), 2);
        assert_eq!(t.alloc(3), Some(0));
        assert_eq!(t.alloc(3), Some(2));
        assert_eq!(t.alloc(3), None);
    }

    #[test]
    fn ceil_boundaries_use_expected_block_counts() {
        let cfg = PagedKvCacheConfig::new(4, 8, 2).unwrap();
        for (len, blocks) in [(1, 1), (3, 1), (4, 1), (5, 2), (13, 4)] {
            let mut c = PagedKvCacheReference::new(cfg).unwrap();
            let x = Tensor::new(vec![0.5f32; len * 2], &[1, len, 2]).unwrap();
            c.append(&x, &x).unwrap();
            assert_eq!(c.num_used_blocks(), blocks, "len={len}");
        }
    }
}
