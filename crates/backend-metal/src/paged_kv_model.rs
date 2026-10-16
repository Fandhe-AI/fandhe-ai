//! デバイス常駐 KV（K-3）の Metal 起動パラメータ計算とシェーダソース不変条件
//! （イシュー #3111。設計 `docs/device-resident-kv-paged-attention-design.md` §3〜§5）。
//!
//! # 役割と責務境界
//!
//! `crate::paged_kv`（macOS 限定の起動 API。`objc2` 系 FFI に触れる）が使う、GPU 非依存の
//! 純関数（dispatch サイズ・カーネル引数の `u32` 収容検査）と、`shaders/paged_kv.metal` の
//! **ソース不変条件テスト**を置く。`objc2` に触れないので `cfg(target_os = "macos")` を
//! 付けず、Linux（開発環境・CI）でも単体テストが回る（`batch_norm_model.rs`・
//! `adam_model.rs` と同じ設計判断）。
//!
//! MSL が実際にコンパイルできるか・実機で CPU 参照と一致するかは Linux では検証できない。
//! ここでは「手動境界チェックが残っていること」「縮約順序を変える構文
//! （`simd_sum` 等）が入っていないこと」をソース文字列で固定し、実コンパイルと実機一致は
//! Mac 実機（#3172。`tests/paged_kv_real_device.rs` の `#[ignore]`）へ申し送る。

use fandhe_ai_tensor_core::{BackendError, PagedAttentionArgs, PagedAttentionPlan, PagedKvLayout};

/// `shaders/paged_kv.metal` のソース（`crate::paged_kv` もこの定数を使う）。
pub const PAGED_KV_MSL_SRC: &str = include_str!("shaders/paged_kv.metal");

/// append の 1 スレッドグループあたりのスレッド数（x 方向 = 成分軸）。
pub const PAGED_APPEND_THREADGROUP_WIDTH: usize = 256;

/// attention の 1 スレッドグループあたりのスレッド数（`kernels_paged_kv.rs` の
/// `PAGED_ATTENTION_BLOCK_DIM` と同値）。
pub const PAGED_ATTENTION_THREADGROUP_WIDTH: usize = 128;

fn invalid(msg: String) -> BackendError {
    BackendError::InvalidArgument(format!("paged_kv(metal): {msg}"))
}

fn to_u32(name: &str, v: usize) -> Result<u32, BackendError> {
    u32::try_from(v).map_err(|_| invalid(format!("{name} が u32::MAX を超える（{v}）")))
}

/// append の dispatch 計算結果（x = ceil(E / 256) グループ・y = 行数グループ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppendDispatch {
    /// 行数（`constant uint& rows`）。
    pub rows: u32,
    /// 埋め込み次元（`constant uint& embed_dim`）。
    pub embed_dim: u32,
    /// プール行数（`constant ulong& pool_rows`）。
    pub pool_rows: u64,
    /// x 方向のスレッドグループ数。
    pub groups_x: usize,
    /// y 方向のスレッドグループ数（= 行数）。
    pub groups_y: usize,
}

/// append のカーネル引数と dispatch を、バッファ長の整合も含めて検査して求める。
pub fn append_dispatch(
    k_new_len: usize,
    v_new_len: usize,
    k_pool_len: usize,
    v_pool_len: usize,
    rows: usize,
    layout: &PagedKvLayout,
) -> Result<AppendDispatch, BackendError> {
    let rows_u = to_u32("rows", rows)?;
    let e_u = to_u32("embed_dim", layout.embed_dim)?;
    let total = rows
        .checked_mul(layout.embed_dim)
        .ok_or_else(|| invalid("追記要素数が usize を超える".into()))?;
    let pool = layout.pool_elems()?;
    if k_new_len != total || v_new_len != total || k_pool_len != pool || v_pool_len != pool {
        return Err(invalid(format!(
            "バッファ長が一致しない（k_new={k_new_len}・v_new={v_new_len}・k_pool={k_pool_len}・\
             v_pool={v_pool_len}・期待 入力={total} プール={pool}）"
        )));
    }
    let pool_rows = u64::try_from(layout.pool_rows()?)
        .map_err(|_| invalid("pool_rows が u64 に収まらない".into()))?;
    Ok(AppendDispatch {
        rows: rows_u,
        embed_dim: e_u,
        pool_rows,
        groups_x: layout
            .embed_dim
            .div_ceil(PAGED_APPEND_THREADGROUP_WIDTH)
            .max(1),
        groups_y: rows,
    })
}

/// attention のカーネル引数（すべて `uint`／`float`）と dispatch 幅。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttentionDispatch {
    pub batch: u32,
    pub l_new: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub embed_dim: u32,
    pub block_size: u32,
    pub num_blocks: u32,
    pub max_blocks_per_seq: u32,
    pub s_stride: u32,
    pub scale: f32,
    /// スレッドグループ数（= query 行数 `B · H · L_new`）。
    pub groups: usize,
}

/// attention のカーネル引数を `u32` 収容検査つきで求める。
pub fn attention_dispatch(
    args: &PagedAttentionArgs,
    plan: &PagedAttentionPlan,
) -> Result<AttentionDispatch, BackendError> {
    Ok(AttentionDispatch {
        batch: to_u32("batch", args.batch)?,
        l_new: to_u32("l_new", args.l_new)?,
        num_heads: to_u32("num_heads", args.num_heads)?,
        head_dim: to_u32("head_dim", args.head_dim())?,
        embed_dim: to_u32("embed_dim", args.layout.embed_dim)?,
        block_size: to_u32("block_size", args.layout.block_size)?,
        num_blocks: to_u32("num_blocks", args.layout.num_blocks)?,
        max_blocks_per_seq: to_u32("max_blocks_per_seq", args.max_blocks_per_seq)?,
        s_stride: to_u32("s_max", plan.s_max)?,
        scale: args.scale,
        groups: {
            to_u32("query 行数", plan.query_rows)?;
            plan.query_rows
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::validate_paged_attention;

    fn layout() -> PagedKvLayout {
        PagedKvLayout::new(4, 2, 4).unwrap()
    }

    #[test]
    fn append_dispatch_accepts_consistent_lengths() {
        let d = append_dispatch(8, 8, 32, 32, 2, &layout()).unwrap();
        assert_eq!((d.rows, d.embed_dim, d.pool_rows), (2, 4, 8));
        assert_eq!((d.groups_x, d.groups_y), (1, 2));
        // E が 256 を超えると x 方向のグループが増える。
        let big = PagedKvLayout::new(2, 2, 600).unwrap();
        let d = append_dispatch(600, 600, 2400, 2400, 1, &big).unwrap();
        assert_eq!(d.groups_x, 3);
    }

    #[test]
    fn append_dispatch_rejects_mismatch_and_overflow() {
        let l = layout();
        assert!(append_dispatch(7, 8, 32, 32, 2, &l).is_err());
        assert!(append_dispatch(8, 8, 31, 32, 2, &l).is_err());
        assert!(append_dispatch(8, 8, 32, 32, usize::MAX, &l).is_err());
        assert!(append_dispatch(0, 0, 32, 32, u32::MAX as usize + 1, &l).is_err());
    }

    #[test]
    fn attention_dispatch_follows_plan() {
        let args = PagedAttentionArgs::new(layout(), 2, 1, 2, 2, 0.5).unwrap();
        let plan = validate_paged_attention(8, 32, 32, &[0, 1, 2, 3], &[3, 3], &args).unwrap();
        let d = attention_dispatch(&args, &plan).unwrap();
        assert_eq!((d.batch, d.l_new, d.num_heads, d.head_dim), (2, 1, 2, 2));
        assert_eq!((d.s_stride, d.groups, d.num_blocks), (3, 4, 4));
        assert_eq!(d.scale, 0.5);
    }

    // --- シェーダソース不変条件（REQ-8・縮約順序。実コンパイルは Mac 実機へ申し送り） ---

    #[test]
    fn shader_keeps_manual_bound_checks() {
        let s = PAGED_KV_MSL_SRC;
        // append。
        assert!(s.contains("if (r >= rows || e >= embed_dim)"));
        assert!(s.contains("if (slot >= pool_rows)"));
        assert!(s.contains("ulong dst = slot * (ulong)embed_dim + (ulong)e;"));
        // attention: ブロックテーブル添字・ブロック ID（番兵拒否）・seq_len の範囲・batch。
        assert!(s.contains("slot_idx >= max_blocks_per_seq"));
        assert!(s.contains("id >= num_blocks"));
        assert!(s.contains("s_prev < 0"));
        assert!(s.contains("(uint)limit_i > s_stride"));
        assert!(s.contains("b_ll >= (ulong)batch"));
        // 違反時は NaN を書く（黙ってゼロにしない）。
        assert!(s.contains("= PAGED_QNAN;"));
    }

    #[test]
    fn shader_fixes_reduction_order() {
        let s = PAGED_KV_MSL_SRC;
        // score・加重和は fma で逐次累積し、q * scale は先に作る。
        assert!(s.contains("acc = fma(q[q_off + d] * scale, k_pool[k_off + d], acc);"));
        assert!(s.contains("o = fma(p, v_pool[v_off], o);"));
        // 和は thread 0 が論理順に逐次。
        assert!(s.contains("sum += score_row[j];"));
        // j 軸の並列リダクションや、結合順序を変える構文は入れない。
        for banned in [
            "simd_sum",
            "simd_shuffle",
            "simd_max",
            "simdgroup_",
            "atomic_",
            "fast::",
            "reduce_add",
        ] {
            assert!(!s.contains(banned), "{banned} must not appear");
        }
    }

    #[test]
    fn shader_uses_64bit_offsets_and_bit_pattern_constants() {
        let s = PAGED_KV_MSL_SRC;
        for needle in [
            "ulong q_off",
            "ulong k_off",
            "ulong v_off",
            "qrow * (ulong)s_stride",
        ] {
            assert!(s.contains(needle), "{needle}");
        }
        assert!(s.contains("as_type<float>(0xFF800000u)"));
        assert!(s.contains("as_type<float>(0x7FC00000u)"));
        assert!(!s.contains("INFINITY"));
    }

    #[test]
    fn shader_declares_both_kernels_with_expected_buffer_layout() {
        let s = PAGED_KV_MSL_SRC;
        assert!(s.contains("kernel void paged_kv_append_f32("));
        assert!(s.contains("kernel void paged_attention_f32("));
        // append は buffer 0..7、attention は buffer 0..16（`crate::paged_kv` の結線と一致）。
        assert!(s.contains("constant ulong& pool_rows [[buffer(7)]]"));
        assert!(s.contains("constant float& scale [[buffer(16)]]"));
    }
}
