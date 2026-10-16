//! デバイス常駐 KV（K-3）の CPU 参照実装（イシュー #3111。設計
//! `docs/device-resident-kv-paged-attention-design.md` §4・§5）。
//!
//! # 役割と責務境界
//!
//! `ops.rs::CpuBackendOps` の `paged_kv_append`・`paged_attention_forward` が呼ぶ
//! ホスト slice 上の計算本体。CUDA・Metal のカーネルと同じ入力契約（スロット写像・
//! ブロックテーブル・`seq_lens`）を持つので、実機 parity テスト（#3113）の期待値生成器に
//! なる。引数検証は `tensor-core` の共通関数
//! （[`fandhe_ai_tensor_core::validate_paged_kv_append`]・
//! [`fandhe_ai_tensor_core::validate_paged_attention`]）を `ops.rs` が先に通す。
//!
//! # 縮約順序（`autodiff::nn::paged_attention_reference` と同一）
//!
//! query へ先に `scale` を掛け、score は `Dh` 方向に `f32::mul_add` で逐次累積する。
//! 系列方向は論理位置 `j = 0..` の単一の連続ループで max → `exp(s − max)` の和 →
//! 正規化重みの `mul_add` 加重和を求める。結合順序が単一の連続 K ループと異ならない
//! ので統一複合判定をそのまま適用できる（REQ-2）。
//!
//! 本ファイルは `unsafe` を持たず、添字アクセスはすべて `get`／`get_mut`（範囲外は
//! panic でなく `Err`。REQ-8）。

#![forbid(unsafe_code)]

use fandhe_ai_tensor_core::{BackendError, PagedAttentionArgs, PagedAttentionPlan, PagedKvLayout};

fn invalid(msg: &str) -> BackendError {
    BackendError::InvalidArgument(format!("paged_kv: {msg}"))
}

/// スロット写像どおりに K／V 行をプールへコピーする（検証済み入力が前提。
/// ただし範囲外は `Err` で防御する）。
pub(crate) fn append_rows(
    k_new: &[f32],
    v_new: &[f32],
    k_pool: &mut [f32],
    v_pool: &mut [f32],
    slot_mapping: &[u32],
    layout: &PagedKvLayout,
) -> Result<(), BackendError> {
    let e = layout.embed_dim;
    for (r, &slot) in slot_mapping.iter().enumerate() {
        let dst = usize::try_from(slot)
            .ok()
            .and_then(|s| s.checked_mul(e))
            .ok_or_else(|| invalid("スロットのオフセットがオーバーフローした"))?;
        let src = r
            .checked_mul(e)
            .ok_or_else(|| invalid("入力オフセットがオーバーフローした"))?;
        let k_src = k_new
            .get(src..src + e)
            .ok_or_else(|| invalid("k_new の範囲外"))?;
        let v_src = v_new
            .get(src..src + e)
            .ok_or_else(|| invalid("v_new の範囲外"))?;
        k_pool
            .get_mut(dst..dst + e)
            .ok_or_else(|| invalid("K プールの範囲外"))?
            .copy_from_slice(k_src);
        v_pool
            .get_mut(dst..dst + e)
            .ok_or_else(|| invalid("V プールの範囲外"))?
            .copy_from_slice(v_src);
    }
    Ok(())
}

/// ブロックテーブル経由で論理位置 `pos`（行 `b`）のプール内オフセット（要素単位）を求める。
fn row_offset(
    block_tables: &[u32],
    args: &PagedAttentionArgs,
    b: usize,
    pos: usize,
) -> Result<usize, BackendError> {
    let bs = args.layout.block_size;
    let idx = b
        .checked_mul(args.max_blocks_per_seq)
        .and_then(|n| n.checked_add(pos / bs))
        .ok_or_else(|| invalid("ブロックテーブルの添字がオーバーフローした"))?;
    let blk = usize::try_from(
        *block_tables
            .get(idx)
            .ok_or_else(|| invalid("ブロックテーブルの添字が範囲外"))?,
    )
    .map_err(|_| invalid("ブロック ID が usize に収まらない"))?;
    if blk >= args.layout.num_blocks {
        return Err(invalid("ブロック ID が num_blocks 以上"));
    }
    blk.checked_mul(bs)
        .and_then(|n| n.checked_add(pos % bs))
        .and_then(|n| n.checked_mul(args.layout.embed_dim))
        .ok_or_else(|| invalid("オフセットがオーバーフローした"))
}

/// ブロックを論理順に走査する attention（head 結合後 `[B, L_new, E]` の平坦 `Vec`）。
pub(crate) fn attention(
    q: &[f32],
    k_pool: &[f32],
    v_pool: &[f32],
    block_tables: &[u32],
    seq_lens: &[u32],
    args: &PagedAttentionArgs,
    plan: &PagedAttentionPlan,
) -> Result<Vec<f32>, BackendError> {
    let mut out: Vec<f32> = Vec::new();
    out.try_reserve_exact(plan.out_numel).map_err(|_| {
        BackendError::DeviceAllocationFailed("paged_kv: 出力の確保に失敗した".into())
    })?;
    out.resize(plan.out_numel, 0.0);
    if plan.out_numel == 0 {
        return Ok(out);
    }
    let e = args.layout.embed_dim;
    let dh = args.head_dim();
    let mut weights: Vec<f32> = Vec::new();
    weights.try_reserve_exact(plan.s_max).map_err(|_| {
        BackendError::DeviceAllocationFailed("paged_kv: score の確保に失敗した".into())
    })?;
    weights.resize(plan.s_max, 0.0);

    for b in 0..args.batch {
        let s_total = usize::try_from(
            *seq_lens
                .get(b)
                .ok_or_else(|| invalid("seq_lens の範囲外"))?,
        )
        .map_err(|_| invalid("seq_len が usize に収まらない"))?;
        let s_prev = s_total
            .checked_sub(args.l_new)
            .ok_or_else(|| invalid("seq_len が L_new 未満"))?;
        for h in 0..args.num_heads {
            for i in 0..args.l_new {
                let q_off = (b * args.l_new + i) * e + h * dh;
                let q_row = q
                    .get(q_off..q_off + dh)
                    .ok_or_else(|| invalid("q の範囲外"))?;
                let limit = s_prev + i + 1;
                let mut max = f32::NEG_INFINITY;
                for j in 0..limit {
                    let off = row_offset(block_tables, args, b, j)? + h * dh;
                    let k_row = k_pool
                        .get(off..off + dh)
                        .ok_or_else(|| invalid("K プールの範囲外"))?;
                    let mut acc = 0.0f32;
                    for (&qv, &kv) in q_row.iter().zip(k_row) {
                        acc = (qv * args.scale).mul_add(kv, acc);
                    }
                    if let Some(w) = weights.get_mut(j) {
                        *w = acc;
                    }
                    if acc > max {
                        max = acc;
                    }
                }
                let mut sum = 0.0f32;
                for w in weights.iter_mut().take(limit) {
                    *w = (*w - max).exp();
                    sum += *w;
                }
                let o_off = q_off;
                for j in 0..limit {
                    let p = weights.get(j).copied().unwrap_or(0.0) / sum;
                    let off = row_offset(block_tables, args, b, j)? + h * dh;
                    let v_row = v_pool
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
    Ok(out)
}
