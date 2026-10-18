//! デバイス常駐 KV（K-3）の ROCm 実行入口（`backend-cuda::paged_kv` の移植。イシュー #3443・分解案 R12。
//! 設計 `docs/device-resident-kv-paged-attention-design.md` §3〜§5）。
//!
//! 役割: HIP/HIPRTC をロードして静的ソース（`kernels_paged_kv`）をコンパイルし、デバイス常駐バッファへ
//! カーネルを積む実行型 [`RocmPagedKv`] を提供する。`BackendOps` への結線（ハンドルの downcast・
//! tensor-core 共通検証・空入力の早期リターン・engine 分岐・`DeviceBuffer` の組み立て）は
//! `ops/paged_kv.rs` の責務で、本ファイルは launch 直前のバッファ長・引数型の再検証（多層防御。A03）と
//! 起動のみを持つ。CUDA 版と異なりスロット写像・ブロックテーブル等の `u32` 配列は呼び出しごとに upload
//! する（`TypedDeviceBuffer<u32>`）。
//!
//! 公開面はスライス入出力の [`RocmPagedKv::append`]／[`RocmPagedKv::attention`]（実機テストと crate 外の
//! 直接利用向け。入口で tensor-core の共通検証を通す）。`TypedDeviceBuffer` は crate 内部型のため、
//! デバイス常駐版（`append_resident`／`attention_resident`）は `pub(crate)`。
//!
//! 起動は `KernelHost` に集約され、呼び出しごとに同期する。**新しい `unsafe` はない**: 再利用している
//! 監査対象は `KernelHost::run_geometry`（監査行 K1）と `TypedDeviceBuffer`（T1〜T3）のみ。
//! `RocmPagedKv` は `KernelHost`（生ポインタを持つ）を含むため `!Send`・`!Sync` のままで、`unsafe impl` は
//! 足さない。

use fandhe_ai_tensor_core::{
    PagedAttentionArgs, PagedAttentionPlan, PagedKvLayout, validate_paged_attention,
    validate_paged_kv_append,
};

use crate::elementwise::fit_i32;
use crate::error::RocmError;
use crate::kernels_paged_kv::{
    PAGED_APPEND_MAX_BLOCKS, PAGED_ATTENTION_BLOCK_DIM, PAGED_ATTENTION_SPEC, PAGED_KV_APPEND_SPEC,
    render_paged_kv_source,
};
use crate::launch::{Arg, INDEX_OPS_BLOCK_DIM, KernelHost};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

const OP_APPEND: &str = "paged_kv_append";
const OP_ATTENTION: &str = "paged_attention";

fn invalid(op: &'static str, detail: String) -> RocmError {
    RocmError::InvalidOpArgs { op, detail }
}

/// tensor-core の検証エラー（`BackendError`）を `RocmError` へ写す（公開スライス版の入口用）。
fn from_validation(op: &'static str, e: fandhe_ai_tensor_core::BackendError) -> RocmError {
    invalid(op, e.to_string())
}

/// append の grid 幅（`total` 要素を `INDEX_OPS_BLOCK_DIM` スレッドで grid-stride）。
/// `1..=PAGED_APPEND_MAX_BLOCKS` に clamp する。GPU 不要の純関数。
pub(crate) fn append_grid_blocks(total: usize) -> u32 {
    let blocks = total.div_ceil(INDEX_OPS_BLOCK_DIM as usize);
    u32::try_from(blocks)
        .unwrap_or(PAGED_APPEND_MAX_BLOCKS)
        .clamp(1, PAGED_APPEND_MAX_BLOCKS)
}

/// append のカーネル引数（`rows`・`embed_dim`・`pool_rows`）と grid を launch 前に型検査して求める。
/// 要素数の整合（入力 `rows · E`・プール `num_blocks · block_size · E`）も確認する。
pub(crate) fn append_launch_params(
    k_new_len: usize,
    v_new_len: usize,
    k_pool_len: usize,
    v_pool_len: usize,
    rows: usize,
    layout: &PagedKvLayout,
) -> Result<(i32, i32, i64, u32), RocmError> {
    let op = OP_APPEND;
    let rows_i = fit_i32(op, rows, "rows")?;
    let e_i = fit_i32(op, layout.embed_dim, "embed_dim")?;
    let total = rows
        .checked_mul(layout.embed_dim)
        .ok_or_else(|| invalid(op, "追記要素数が usize を超える".into()))?;
    let pool = layout.pool_elems().map_err(|e| from_validation(op, e))?;
    if k_new_len != total || v_new_len != total || k_pool_len != pool || v_pool_len != pool {
        return Err(invalid(
            op,
            format!(
                "バッファ長が一致しない（k_new={k_new_len}・v_new={v_new_len}・k_pool={k_pool_len}・\
                 v_pool={v_pool_len}・期待 入力={total} プール={pool}）"
            ),
        ));
    }
    let pool_rows = layout.pool_rows().map_err(|e| from_validation(op, e))?;
    let pool_rows_i = i64::try_from(pool_rows)
        .map_err(|_| invalid(op, "pool_rows が i64 に収まらない".into()))?;
    Ok((rows_i, e_i, pool_rows_i, append_grid_blocks(total)))
}

/// attention のカーネル引数をまとめた値。`num_blocks` は u32 全域を受けるため `i64`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AttentionLaunchParams {
    pub(crate) batch: i32,
    pub(crate) l_new: i32,
    pub(crate) num_heads: i32,
    pub(crate) head_dim: i32,
    pub(crate) embed_dim: i32,
    pub(crate) block_size: i32,
    pub(crate) num_blocks: i64,
    pub(crate) max_blocks_per_seq: i32,
    pub(crate) s_stride: i32,
    pub(crate) scale: f32,
    pub(crate) grid_blocks: u32,
}

/// attention のカーネル引数を launch 前に型検査して求める（GPU 不要）。
pub(crate) fn attention_launch_params(
    args: &PagedAttentionArgs,
    plan: &PagedAttentionPlan,
) -> Result<AttentionLaunchParams, RocmError> {
    let op = OP_ATTENTION;
    Ok(AttentionLaunchParams {
        batch: fit_i32(op, args.batch, "batch")?,
        l_new: fit_i32(op, args.l_new, "l_new")?,
        num_heads: fit_i32(op, args.num_heads, "num_heads")?,
        head_dim: fit_i32(op, args.head_dim(), "head_dim")?,
        embed_dim: fit_i32(op, args.layout.embed_dim, "embed_dim")?,
        block_size: fit_i32(op, args.layout.block_size, "block_size")?,
        num_blocks: i64::try_from(args.layout.num_blocks)
            .map_err(|_| invalid(op, "num_blocks が i64 に収まらない".into()))?,
        max_blocks_per_seq: fit_i32(op, args.max_blocks_per_seq, "max_blocks_per_seq")?,
        s_stride: fit_i32(op, plan.s_max, "s_max")?,
        scale: args.scale,
        grid_blocks: u32::try_from(plan.query_rows)
            .map_err(|_| invalid(op, "query 行数が u32 に収まらない".into()))?,
    })
}

/// attention の入出力・補助配列の長さを検査する（launch 直前の多層防御。GPU 不要）。
pub(crate) fn check_attention_lens(
    q_len: usize,
    k_pool_len: usize,
    v_pool_len: usize,
    block_tables_len: usize,
    seq_lens_len: usize,
    args: &PagedAttentionArgs,
    plan: &PagedAttentionPlan,
) -> Result<(), RocmError> {
    let op = OP_ATTENTION;
    let pool = args
        .layout
        .pool_elems()
        .map_err(|e| from_validation(op, e))?;
    let tables_len = args
        .batch
        .checked_mul(args.max_blocks_per_seq)
        .ok_or_else(|| invalid(op, "block_tables の要素数が usize を超える".into()))?;
    if q_len != plan.out_numel
        || k_pool_len != pool
        || v_pool_len != pool
        || block_tables_len != tables_len
        || seq_lens_len != args.batch
    {
        return Err(invalid(
            op,
            format!(
                "バッファ長が一致しない（q={q_len}・k_pool={k_pool_len}・v_pool={v_pool_len}・\
                 tables={block_tables_len}・seq_lens={seq_lens_len}）"
            ),
        ));
    }
    Ok(())
}

/// コンパイル済みの paged KV 2 カーネル（append・attention。いずれも f32）を保持し、実行する。
pub struct RocmPagedKv {
    host: KernelHost,
}

impl RocmPagedKv {
    /// HIP/HIPRTC をロードし 2 カーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_paged_kv_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// スロット写像どおりに `k_new`／`v_new` の行を K／V プールへ書く（デバイス常駐版）。
    ///
    /// 重複・範囲外スロットの検証（`validate_paged_kv_append`）は呼び出し元が済ませている前提で、
    /// ここでは launch 前にバッファ長・引数型を再検証する。行数 0 は HIP に触れず成功する。
    pub(crate) fn append_resident(
        &self,
        k_new: &TypedDeviceBuffer<f32>,
        v_new: &TypedDeviceBuffer<f32>,
        k_pool: &mut TypedDeviceBuffer<f32>,
        v_pool: &mut TypedDeviceBuffer<f32>,
        slot_mapping: &[u32],
        layout: &PagedKvLayout,
    ) -> Result<(), RocmError> {
        let rows = slot_mapping.len();
        if rows == 0 {
            return Ok(());
        }
        let (rows_i, e_i, pool_rows, blocks) = append_launch_params(
            k_new.len(),
            v_new.len(),
            k_pool.len(),
            v_pool.len(),
            rows,
            layout,
        )?;
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let d_slots = TypedDeviceBuffer::<u32>::from_slice(rt, slot_mapping)?;
        let args = [
            Arg::Ptr(k_new.as_device_ptr()),
            Arg::Ptr(v_new.as_device_ptr()),
            Arg::Ptr(k_pool.as_device_ptr()),
            Arg::Ptr(v_pool.as_device_ptr()),
            Arg::Ptr(d_slots.as_device_ptr()),
            Arg::I32(rows_i),
            Arg::I32(e_i),
            Arg::I64(pool_rows),
        ];
        self.host
            .run_blocks(OP_APPEND, &PAGED_KV_APPEND_SPEC, blocks, &args)
    }

    /// ブロックを論理順に走査する attention（デバイス常駐版）。出力 `[B, L_new, E]` を新規に確保して返す。
    ///
    /// 引数検証と `plan` の算出（`validate_paged_attention`）は呼び出し元が済ませている前提で、ここでは
    /// launch 前にバッファ長・引数型を再検証する。score の作業バッファは 0 初期化する（未初期化読みを
    /// 作らない多層防御。数値契約には影響しない）。呼び出し元は `plan.out_numel == 0` を先に処理する。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_resident(
        &self,
        q: &TypedDeviceBuffer<f32>,
        k_pool: &TypedDeviceBuffer<f32>,
        v_pool: &TypedDeviceBuffer<f32>,
        block_tables: &[u32],
        seq_lens: &[u32],
        args: &PagedAttentionArgs,
        plan: &PagedAttentionPlan,
    ) -> Result<TypedDeviceBuffer<f32>, RocmError> {
        if plan.query_rows == 0 || plan.out_numel == 0 {
            return Err(invalid(
                OP_ATTENTION,
                "空の出力は呼び出し元で処理する契約です".into(),
            ));
        }
        let p = attention_launch_params(args, plan)?;
        check_attention_lens(
            q.len(),
            k_pool.len(),
            v_pool.len(),
            block_tables.len(),
            seq_lens.len(),
            args,
            plan,
        )?;
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let d_tables = TypedDeviceBuffer::<u32>::from_slice(rt.clone(), block_tables)?;
        let d_seq = TypedDeviceBuffer::<u32>::from_slice(rt.clone(), seq_lens)?;
        // スコア workspace は最大 1<<30 要素になり得るため、host の Vec を経由するゼロ初期化はしない
        // （host abort・数 GiB の H2D 転送の回避。CUDA 側と同じ未初期化確保）。カーネルは各行で
        // `j < limit` の要素を書いてから読み（違反時は読まずに return）、出力は grid = query_rows の
        // 全ブロックが各自の `head_dim` 要素を全経路で書くため、未初期化領域は読まれない。
        let d_scores = TypedDeviceBuffer::<f32>::new(rt.clone(), plan.workspace_elems)?;
        let d_out = TypedDeviceBuffer::<f32>::new(rt, plan.out_numel)?;
        let kargs = [
            Arg::Ptr(q.as_device_ptr()),
            Arg::Ptr(k_pool.as_device_ptr()),
            Arg::Ptr(v_pool.as_device_ptr()),
            Arg::Ptr(d_tables.as_device_ptr()),
            Arg::Ptr(d_seq.as_device_ptr()),
            Arg::Ptr(d_scores.as_device_ptr()),
            Arg::Ptr(d_out.as_device_ptr()),
            Arg::I32(p.batch),
            Arg::I32(p.l_new),
            Arg::I32(p.num_heads),
            Arg::I32(p.head_dim),
            Arg::I32(p.embed_dim),
            Arg::I32(p.block_size),
            Arg::I64(p.num_blocks),
            Arg::I32(p.max_blocks_per_seq),
            Arg::I32(p.s_stride),
            Arg::F32(p.scale),
        ];
        self.host.run_geometry(
            OP_ATTENTION,
            &PAGED_ATTENTION_SPEC,
            (p.grid_blocks, 1, 1),
            (PAGED_ATTENTION_BLOCK_DIM, 1, 1),
            &kargs,
        )?;
        Ok(d_out)
    }

    /// スライス入出力の append。`k_pool`／`v_pool` は `[num_blocks, block_size, E]` の平坦配列で、
    /// 行を書いた結果を同じスライスへ書き戻す。入口で tensor-core の共通検証を通す。
    pub fn append(
        &self,
        k_new: &[f32],
        v_new: &[f32],
        k_pool: &mut [f32],
        v_pool: &mut [f32],
        slot_mapping: &[u32],
        layout: &PagedKvLayout,
    ) -> Result<(), RocmError> {
        validate_paged_kv_append(
            k_new.len(),
            v_new.len(),
            k_pool.len(),
            v_pool.len(),
            slot_mapping,
            layout,
        )
        .map_err(|e| from_validation(OP_APPEND, e))?;
        if slot_mapping.is_empty() {
            return Ok(());
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let d_kn = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), k_new)?;
        let d_vn = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), v_new)?;
        let mut d_kp = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), k_pool)?;
        let mut d_vp = TypedDeviceBuffer::<f32>::from_slice(rt, v_pool)?;
        self.append_resident(&d_kn, &d_vn, &mut d_kp, &mut d_vp, slot_mapping, layout)?;
        d_kp.download(k_pool)?;
        d_vp.download(v_pool)
    }

    /// スライス入出力の attention。出力 `[B, L_new, E]` を平坦な `Vec` で返す。入口で tensor-core の
    /// 共通検証（`validate_paged_attention`）を通す。
    pub fn attention(
        &self,
        q: &[f32],
        k_pool: &[f32],
        v_pool: &[f32],
        block_tables: &[u32],
        seq_lens: &[u32],
        args: &PagedAttentionArgs,
    ) -> Result<Vec<f32>, RocmError> {
        let plan = validate_paged_attention(
            q.len(),
            k_pool.len(),
            v_pool.len(),
            block_tables,
            seq_lens,
            args,
        )
        .map_err(|e| from_validation(OP_ATTENTION, e))?;
        if plan.out_numel == 0 || plan.query_rows == 0 {
            return Ok(vec![0.0; plan.out_numel]);
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let d_q = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), q)?;
        let d_kp = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), k_pool)?;
        let d_vp = TypedDeviceBuffer::<f32>::from_slice(rt, v_pool)?;
        let out =
            self.attention_resident(&d_q, &d_kp, &d_vp, block_tables, seq_lens, args, &plan)?;
        out.to_vec()
    }
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
    fn append_grid_is_clamped_to_one_through_max() {
        assert_eq!(append_grid_blocks(0), 1);
        assert_eq!(append_grid_blocks(1), 1);
        assert_eq!(append_grid_blocks(256), 1);
        assert_eq!(append_grid_blocks(257), 2);
        assert_eq!(
            append_grid_blocks(256 * 65_535 + 1),
            PAGED_APPEND_MAX_BLOCKS
        );
        assert_eq!(append_grid_blocks(usize::MAX), PAGED_APPEND_MAX_BLOCKS);
    }

    #[test]
    fn append_params_accept_matching_lengths() {
        let l = layout();
        let pool = 4 * 2 * 4;
        let (rows, e, pool_rows, blocks) = append_launch_params(8, 8, pool, pool, 2, &l).unwrap();
        assert_eq!((rows, e, pool_rows, blocks), (2, 4, 8, 1));
    }

    #[test]
    fn append_params_reject_length_mismatch_and_overflow() {
        let l = layout();
        let pool = 4 * 2 * 4;
        for (kn, vn, kp, vp) in [
            (7, 8, pool, pool),
            (8, 7, pool, pool),
            (8, 8, pool - 1, pool),
            (8, 8, pool, pool + 1),
        ] {
            assert!(matches!(
                append_launch_params(kn, vn, kp, vp, 2, &l),
                Err(RocmError::InvalidOpArgs { .. })
            ));
        }
        // rows が i32 を超える。
        assert!(matches!(
            append_launch_params(0, 0, pool, pool, i32::MAX as usize + 1, &l),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }

    #[test]
    fn attention_params_widen_num_blocks_to_i64() {
        let a = args(2, 3, 5);
        let plan = PagedAttentionPlan {
            out_numel: 2 * 3 * 4,
            s_max: 7,
            query_rows: 2 * 2 * 3,
            workspace_elems: 2 * 2 * 3 * 7,
        };
        let p = attention_launch_params(&a, &plan).unwrap();
        assert_eq!(p.num_blocks, 4);
        assert_eq!((p.batch, p.l_new, p.num_heads, p.head_dim), (2, 3, 2, 2));
        assert_eq!((p.embed_dim, p.block_size), (4, 2));
        assert_eq!(
            (p.max_blocks_per_seq, p.s_stride, p.grid_blocks),
            (5, 7, 12)
        );
        assert_eq!(p.scale, 0.5);
        // num_blocks は i32 を超える値（u32::MAX 未満）でも i64 で表せる。
        let mut big = a;
        big.layout.num_blocks = u32::MAX as usize - 1;
        assert_eq!(
            attention_launch_params(&big, &plan).unwrap().num_blocks,
            i64::from(u32::MAX) - 1
        );
    }

    #[test]
    fn attention_params_reject_oversized_values() {
        let a = args(1, 1, 2);
        let mut plan = PagedAttentionPlan {
            out_numel: 4,
            s_max: 2,
            query_rows: 2,
            workspace_elems: 4,
        };
        plan.s_max = i32::MAX as usize + 1;
        assert!(attention_launch_params(&a, &plan).is_err());
        plan.s_max = 2;
        plan.query_rows = u32::MAX as usize + 1;
        assert!(attention_launch_params(&a, &plan).is_err());
    }

    #[test]
    fn attention_lens_are_checked() {
        let a = args(2, 1, 3);
        let plan = PagedAttentionPlan {
            out_numel: 2 * 4,
            s_max: 4,
            query_rows: 4,
            workspace_elems: 16,
        };
        let pool = 4 * 2 * 4;
        check_attention_lens(8, pool, pool, 6, 2, &a, &plan).unwrap();
        for (q, k, v, t, s) in [
            (7, pool, pool, 6, 2),
            (8, pool - 1, pool, 6, 2),
            (8, pool, pool + 1, 6, 2),
            (8, pool, pool, 5, 2),
            (8, pool, pool, 6, 1),
        ] {
            assert!(matches!(
                check_attention_lens(q, k, v, t, s, &a, &plan),
                Err(RocmError::InvalidOpArgs { .. })
            ));
        }
    }
}
