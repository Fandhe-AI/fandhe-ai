//! デバイス常駐 KV（K-3）の引数型と、3 バックエンド共通の検証関数
//! （イシュー #3111・親 #3108。設計正本
//! `docs/device-resident-kv-paged-attention-design.md` §2・§3・§4・§7）。
//!
//! # 役割と責務境界
//!
//! [`crate::BackendOps::paged_kv_append`]・[`crate::BackendOps::paged_attention_forward`]
//! （およびその `_tracked` 版）が受ける引数の型（[`PagedKvLayout`]・
//! [`PagedAttentionArgs`]）と、FFI（CUDA launch・Metal encode）や CPU 参照の
//! 計算に入る**前**に必ず通す pure 検証関数
//! （[`validate_paged_kv_append`]・[`validate_paged_attention`]）を置く。
//! `backend-cpu`・`backend-cuda`・`backend-metal` の 3 実装はすべて同じ関数を
//! 呼ぶので、検査の抜けがバックエンド間でずれない（OWASP A03・A04）。
//! ページテーブルそのもの（空きブロック管理・巻き戻し）はホストの
//! `autodiff::nn::paged_kv::BlockTable` が持ち、本モジュールは関与しない。
//!
//! # プールとスロット写像
//!
//! K／V プールとも `[num_blocks, block_size, E]` の行優先 `f32`。論理位置の行は
//! 「トークン行番号 = `block_id · block_size + pos % block_size`」で表し、追記は
//! この行番号の列（スロット写像 `slot_mapping`）で書き先を指す。
//!
//! # 安全性
//!
//! 本モジュールは `unsafe` を持たない。すべての算術は `checked_*` で行い、超過は
//! panic でなく [`BackendError::InvalidArgument`] を返す。

use crate::device::BackendError;

/// `block_size` の上限（`autodiff::nn::PagedKvCacheConfig` と同値。2 のべき乗で 1〜256）。
pub const PAGED_KV_MAX_BLOCK_SIZE: usize = 256;

/// attention 作業バッファ（score 行列）の要素数の上限（`1 << 30` 要素 = 4 GiB。
/// 設計 §4・OWASP A04）。超えたら [`BackendError::InvalidArgument`]。
pub const PAGED_ATTENTION_MAX_WORKSPACE_ELEMS: usize = 1 << 30;

fn invalid(msg: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument(format!("paged_kv: {}", msg.into()))
}

/// K／V プールのレイアウト（`[num_blocks, block_size, embed_dim]`）。
///
/// [`PagedKvLayout::new`] が唯一の構築経路で、検査を通った値だけを保持する
/// （`#[non_exhaustive]` なので他クレートは struct リテラルで作れない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PagedKvLayout {
    /// プール全体のブロック数（1 以上 `u32::MAX` 未満。ブロック ID は `u32`）。
    pub num_blocks: usize,
    /// ブロックあたりのトークン数（2 のべき乗で 1〜256）。
    pub block_size: usize,
    /// 埋め込み次元 `E`（1 以上 `i32::MAX` 以下。カーネル引数が `int`）。
    pub embed_dim: usize,
}

impl PagedKvLayout {
    /// 検査つき構築。片側プールの要素数・K＋V のバイト数が `isize::MAX` を超える
    /// 組み合わせも拒否する。
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        embed_dim: usize,
    ) -> Result<PagedKvLayout, BackendError> {
        if block_size == 0 || block_size > PAGED_KV_MAX_BLOCK_SIZE || !block_size.is_power_of_two()
        {
            return Err(invalid(format!(
                "block_size は 2 のべき乗で 1〜{PAGED_KV_MAX_BLOCK_SIZE}（実際: {block_size}）"
            )));
        }
        if num_blocks == 0 || u64::try_from(num_blocks).map_or(true, |n| n >= u64::from(u32::MAX)) {
            return Err(invalid(format!(
                "num_blocks は 1 以上 u32::MAX 未満（実際: {num_blocks}）"
            )));
        }
        if embed_dim == 0 || i32::try_from(embed_dim).is_err() {
            return Err(invalid(format!(
                "embed_dim は 1 以上 i32::MAX 以下（実際: {embed_dim}）"
            )));
        }
        let layout = PagedKvLayout {
            num_blocks,
            block_size,
            embed_dim,
        };
        let elems = layout.pool_elems()?;
        let bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| invalid("プールのバイト数が usize を超える"))?;
        if bytes > isize::MAX as usize {
            return Err(invalid("プールのバイト数が isize::MAX を超える"));
        }
        Ok(layout)
    }

    /// 片側（K または V）のプールの要素数。
    pub fn pool_elems(&self) -> Result<usize, BackendError> {
        self.num_blocks
            .checked_mul(self.block_size)
            .and_then(|n| n.checked_mul(self.embed_dim))
            .ok_or_else(|| invalid("プールの要素数が usize を超える"))
    }

    /// プール内のトークン行の総数（`num_blocks · block_size`）。
    pub fn pool_rows(&self) -> Result<usize, BackendError> {
        self.num_blocks
            .checked_mul(self.block_size)
            .ok_or_else(|| invalid("プールの行数が usize を超える"))
    }
}

/// [`crate::BackendOps::paged_attention_forward`] の引数（形状・head 分割・scale）。
///
/// [`PagedAttentionArgs::new`] が唯一の構築経路。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PagedAttentionArgs {
    /// K／V プールのレイアウト。
    pub layout: PagedKvLayout,
    /// バッチ `B`（`i32::MAX` 以下）。
    pub batch: usize,
    /// 今回の query 行数 `L_new`（`i32::MAX` 以下）。
    pub l_new: usize,
    /// head 数（1 以上で `embed_dim` を割り切る）。
    pub num_heads: usize,
    /// ブロックテーブルの 1 行あたりの枠数（`i32::MAX` 以下）。
    pub max_blocks_per_seq: usize,
    /// query に先に掛ける scale（有限値）。
    pub scale: f32,
}

impl PagedAttentionArgs {
    /// 検査つき構築（カーネル引数が `int` に収まること・`num_heads` の割り切り・
    /// `scale` の有限性）。
    pub fn new(
        layout: PagedKvLayout,
        batch: usize,
        l_new: usize,
        num_heads: usize,
        max_blocks_per_seq: usize,
        scale: f32,
    ) -> Result<PagedAttentionArgs, BackendError> {
        for (name, v) in [
            ("batch", batch),
            ("l_new", l_new),
            ("max_blocks_per_seq", max_blocks_per_seq),
        ] {
            if i32::try_from(v).is_err() {
                return Err(invalid(format!("{name} が i32::MAX を超える（{v}）")));
            }
        }
        if num_heads == 0 || !layout.embed_dim.is_multiple_of(num_heads) {
            return Err(invalid(
                "num_heads は 1 以上で embed_dim を割り切る必要がある",
            ));
        }
        if !scale.is_finite() {
            return Err(invalid("scale は有限値でなければならない"));
        }
        Ok(PagedAttentionArgs {
            layout,
            batch,
            l_new,
            num_heads,
            max_blocks_per_seq,
            scale,
        })
    }

    /// head あたりの次元 `Dh`。
    pub fn head_dim(&self) -> usize {
        self.layout.embed_dim / self.num_heads
    }
}

/// [`validate_paged_attention`] が返す実行計画（出力サイズ・作業バッファ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedAttentionPlan {
    /// 出力 `[B, L_new, E]` の要素数。
    pub out_numel: usize,
    /// 使用する系列長の最大値（`max(seq_lens)`。score 行列の 1 行の幅）。
    pub s_max: usize,
    /// query 行の総数 `B · H · L_new`（カーネルの grid 幅）。
    pub query_rows: usize,
    /// score 作業バッファの要素数（`query_rows · s_max`）。
    pub workspace_elems: usize,
}

/// `paged_kv_append` の引数検証（FFI・参照計算の前に 3 バックエンドが呼ぶ）。
///
/// 検査: K／V 入力の要素数が `slot_mapping.len() · E` に等しい／K／V プールの要素数が
/// レイアウトと一致する／スロットがすべて `num_blocks · block_size` 未満／重複スロットが
/// ない（カーネルの書き込みが競合しない前提）。成功時は追記する行数を返す。
pub fn validate_paged_kv_append(
    k_new_numel: usize,
    v_new_numel: usize,
    k_pool_numel: usize,
    v_pool_numel: usize,
    slot_mapping: &[u32],
    layout: &PagedKvLayout,
) -> Result<usize, BackendError> {
    let rows = slot_mapping.len();
    let expect = rows
        .checked_mul(layout.embed_dim)
        .ok_or_else(|| invalid("追記要素数が usize を超える"))?;
    if k_new_numel != expect || v_new_numel != expect {
        return Err(invalid(format!(
            "k_new／v_new の要素数が slot_mapping.len() · E と一致しない（k={k_new_numel}・v={v_new_numel}・期待={expect}）"
        )));
    }
    // カーネルの行番号は `int`。
    if i32::try_from(rows).is_err() {
        return Err(invalid("追記行数が i32::MAX を超える"));
    }
    let pool = layout.pool_elems()?;
    if k_pool_numel != pool || v_pool_numel != pool {
        return Err(invalid(format!(
            "プールの要素数がレイアウトと一致しない（k={k_pool_numel}・v={v_pool_numel}・期待={pool}）"
        )));
    }
    let pool_rows = layout.pool_rows()?;
    let mut sorted: Vec<u32> = Vec::new();
    sorted
        .try_reserve_exact(rows)
        .map_err(|_| invalid("スロット検査用バッファの確保に失敗した"))?;
    for &s in slot_mapping {
        let su = usize::try_from(s).map_err(|_| invalid("スロットが usize に収まらない"))?;
        if su >= pool_rows {
            return Err(invalid(format!(
                "スロット {s} がプール行数 {pool_rows} 以上"
            )));
        }
        sorted.push(s);
    }
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(invalid("slot_mapping に重複スロットがある"));
    }
    Ok(rows)
}

/// `paged_attention_forward` の引数検証。成功時は [`PagedAttentionPlan`] を返す。
///
/// 検査: q・K／V プールの要素数／`block_tables.len() == B · max_blocks_per_seq`・
/// `seq_lens.len() == B`／各行 `L_new <= seq_len <= max_blocks_per_seq · block_size`／
/// 使用範囲（`ceil(seq_len / block_size)` 枠）のブロック ID が `num_blocks` 未満
/// （番兵 `u32::MAX` は `num_blocks` 未満にならないので使用範囲では拒否される）／
/// 作業バッファ・grid 幅の上限。
pub fn validate_paged_attention(
    q_numel: usize,
    k_pool_numel: usize,
    v_pool_numel: usize,
    block_tables: &[u32],
    seq_lens: &[u32],
    args: &PagedAttentionArgs,
) -> Result<PagedAttentionPlan, BackendError> {
    let layout = &args.layout;
    let e = layout.embed_dim;
    let out_numel = args
        .batch
        .checked_mul(args.l_new)
        .and_then(|n| n.checked_mul(e))
        .ok_or_else(|| invalid("出力の要素数が usize を超える"))?;
    if q_numel != out_numel {
        return Err(invalid(format!(
            "q の要素数が B · L_new · E と一致しない（実際 {q_numel}・期待 {out_numel}）"
        )));
    }
    let pool = layout.pool_elems()?;
    if k_pool_numel != pool || v_pool_numel != pool {
        return Err(invalid("プールの要素数がレイアウトと一致しない"));
    }
    let tables_len = args
        .batch
        .checked_mul(args.max_blocks_per_seq)
        .ok_or_else(|| invalid("block_tables の要素数が usize を超える"))?;
    if block_tables.len() != tables_len {
        return Err(invalid(format!(
            "block_tables.len() が B · max_blocks_per_seq と一致しない（実際 {}・期待 {tables_len}）",
            block_tables.len()
        )));
    }
    if seq_lens.len() != args.batch {
        return Err(invalid(format!(
            "seq_lens.len() が B と一致しない（実際 {}・期待 {}）",
            seq_lens.len(),
            args.batch
        )));
    }
    let max_seq = args
        .max_blocks_per_seq
        .checked_mul(layout.block_size)
        .ok_or_else(|| invalid("max_blocks_per_seq · block_size が usize を超える"))?;
    let mut s_max = 0usize;
    for (b, &sl) in seq_lens.iter().enumerate() {
        let sl = usize::try_from(sl).map_err(|_| invalid("seq_len が usize に収まらない"))?;
        if sl < args.l_new || sl > max_seq {
            return Err(invalid(format!(
                "行 {b} の seq_len {sl} が範囲外（L_new={}・上限={max_seq}）",
                args.l_new
            )));
        }
        s_max = s_max.max(sl);
        let used = sl.div_ceil(layout.block_size);
        let row = block_tables
            .get(b * args.max_blocks_per_seq..)
            .and_then(|r| r.get(..args.max_blocks_per_seq))
            .ok_or_else(|| invalid("block_tables の行範囲が範囲外"))?;
        for (slot, &id) in row.iter().take(used).enumerate() {
            let idu =
                usize::try_from(id).map_err(|_| invalid("ブロック ID が usize に収まらない"))?;
            if idu >= layout.num_blocks {
                return Err(invalid(format!(
                    "行 {b} の枠 {slot} のブロック ID {id} が num_blocks {} 以上（番兵を含む）",
                    layout.num_blocks
                )));
            }
        }
    }
    let query_rows = args
        .batch
        .checked_mul(args.num_heads)
        .and_then(|n| n.checked_mul(args.l_new))
        .ok_or_else(|| invalid("query 行数が usize を超える"))?;
    if i32::try_from(query_rows).is_err() {
        return Err(invalid("query 行数 B · H · L_new が i32::MAX を超える"));
    }
    if i32::try_from(s_max).is_err() {
        return Err(invalid("系列長が i32::MAX を超える"));
    }
    let workspace_elems = query_rows
        .checked_mul(s_max)
        .ok_or_else(|| invalid("作業バッファの要素数が usize を超える"))?;
    if workspace_elems > PAGED_ATTENTION_MAX_WORKSPACE_ELEMS {
        return Err(invalid(format!(
            "作業バッファ {workspace_elems} 要素が上限 {PAGED_ATTENTION_MAX_WORKSPACE_ELEMS} を超える"
        )));
    }
    Ok(PagedAttentionPlan {
        out_numel,
        s_max,
        query_rows,
        workspace_elems,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> PagedKvLayout {
        PagedKvLayout::new(4, 2, 4).unwrap()
    }

    fn args(batch: usize, l_new: usize, max_blocks: usize) -> PagedAttentionArgs {
        PagedAttentionArgs::new(layout(), batch, l_new, 2, max_blocks, 0.5).unwrap()
    }

    #[test]
    fn layout_rejects_invalid_values() {
        assert!(PagedKvLayout::new(0, 2, 4).is_err());
        assert!(PagedKvLayout::new(4, 3, 4).is_err());
        assert!(PagedKvLayout::new(4, 512, 4).is_err());
        assert!(PagedKvLayout::new(4, 2, 0).is_err());
        assert!(PagedKvLayout::new(u32::MAX as usize, 2, 4).is_err());
        assert!(PagedKvLayout::new(4, 2, usize::MAX).is_err());
        assert!(PagedKvLayout::new(1 << 20, 256, (isize::MAX as usize) / (8 * 256) + 1).is_err());
        assert!(PagedKvLayout::new(4, 256, 4).is_ok());
    }

    #[test]
    fn args_reject_bad_heads_scale_and_sizes() {
        let l = layout();
        assert!(PagedAttentionArgs::new(l, 1, 1, 0, 2, 1.0).is_err());
        assert!(PagedAttentionArgs::new(l, 1, 1, 3, 2, 1.0).is_err());
        assert!(PagedAttentionArgs::new(l, 1, 1, 2, 2, f32::NAN).is_err());
        assert!(PagedAttentionArgs::new(l, 1, 1, 2, 2, f32::INFINITY).is_err());
        assert!(PagedAttentionArgs::new(l, usize::MAX, 1, 2, 2, 1.0).is_err());
        assert_eq!(args(1, 1, 2).head_dim(), 2);
    }

    #[test]
    fn append_accepts_valid_and_rejects_violations() {
        let l = layout();
        let pool = 4 * 2 * 4;
        assert_eq!(
            validate_paged_kv_append(8, 8, pool, pool, &[0, 7], &l).unwrap(),
            2
        );
        // 要素数不一致。
        assert!(validate_paged_kv_append(7, 8, pool, pool, &[0, 7], &l).is_err());
        assert!(validate_paged_kv_append(8, 4, pool, pool, &[0, 7], &l).is_err());
        // プール不一致。
        assert!(validate_paged_kv_append(8, 8, pool - 1, pool, &[0, 7], &l).is_err());
        // 範囲外スロット（pool_rows = 8）。
        assert!(validate_paged_kv_append(8, 8, pool, pool, &[0, 8], &l).is_err());
        assert!(validate_paged_kv_append(4, 4, pool, pool, &[u32::MAX], &l).is_err());
        // 重複スロット。
        assert!(validate_paged_kv_append(8, 8, pool, pool, &[3, 3], &l).is_err());
        // 空は成功（0 行）。
        assert_eq!(
            validate_paged_kv_append(0, 0, pool, pool, &[], &l).unwrap(),
            0
        );
    }

    #[test]
    fn attention_accepts_valid_and_reports_plan() {
        let a = args(2, 1, 2);
        let pool = 32;
        // B=2・各行 seq_len=3（2 ブロック使用）。
        let tables = [0u32, 1, 2, 3];
        let plan = validate_paged_attention(8, pool, pool, &tables, &[3, 3], &a).unwrap();
        assert_eq!(plan.out_numel, 8);
        assert_eq!(plan.s_max, 3);
        assert_eq!(plan.query_rows, 4);
        assert_eq!(plan.workspace_elems, 12);
    }

    #[test]
    fn attention_rejects_violations() {
        let a = args(1, 2, 2);
        let pool = 32;
        let ok_tables = [0u32, 1];
        assert!(validate_paged_attention(8, pool, pool, &ok_tables, &[3], &a).is_ok());
        // q 要素数。
        assert!(validate_paged_attention(7, pool, pool, &ok_tables, &[3], &a).is_err());
        // プール。
        assert!(validate_paged_attention(8, pool - 1, pool, &ok_tables, &[3], &a).is_err());
        // 長さ不一致。
        assert!(validate_paged_attention(8, pool, pool, &[0u32], &[3], &a).is_err());
        assert!(validate_paged_attention(8, pool, pool, &ok_tables, &[3, 3], &a).is_err());
        // seq_len < L_new・> 上限（2 枠 · block_size 2 = 4）。
        assert!(validate_paged_attention(8, pool, pool, &ok_tables, &[1], &a).is_err());
        assert!(validate_paged_attention(8, pool, pool, &ok_tables, &[5], &a).is_err());
        // ブロック ID 範囲外・番兵が使用範囲に入る。
        assert!(validate_paged_attention(8, pool, pool, &[0u32, 4], &[3], &a).is_err());
        assert!(validate_paged_attention(8, pool, pool, &[0u32, u32::MAX], &[3], &a).is_err());
        // 使用範囲外の番兵は許可（seq_len=2 なら 1 枠のみ使用）。
        assert!(validate_paged_attention(8, pool, pool, &[0u32, u32::MAX], &[2], &a).is_ok());
    }

    #[test]
    fn attention_rejects_workspace_overflow() {
        let l = PagedKvLayout::new(4, 2, 4).unwrap();
        let a = PagedAttentionArgs::new(l, 1, 1, 1, 2, 1.0).unwrap();
        // 正常。上限超過は query_rows · s_max の積が大きい構成で確認する。
        let big = PagedAttentionArgs {
            batch: 1 << 20,
            l_new: 1 << 10,
            num_heads: 1,
            ..a
        };
        let r = validate_paged_attention(
            (1usize << 20) * (1 << 10) * 4,
            32,
            32,
            &vec![0u32; (1 << 20) * 2],
            &vec![2u32; 1 << 20],
            &big,
        );
        assert!(r.is_err());
    }

    #[test]
    fn empty_batch_and_empty_l_new_are_valid() {
        let a = args(0, 1, 2);
        let plan = validate_paged_attention(0, 32, 32, &[], &[], &a).unwrap();
        assert_eq!(plan.out_numel, 0);
        assert_eq!(plan.workspace_elems, 0);
        let a = args(1, 0, 2);
        let plan = validate_paged_attention(0, 32, 32, &[0, 1], &[3], &a).unwrap();
        assert_eq!(plan.out_numel, 0);
        assert_eq!(plan.workspace_elems, 0);
    }
}
