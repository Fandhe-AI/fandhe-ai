//! デバイス常駐 KV（K-3）の CUDA 起動 API（NVRTC コンパイル・保持・起動。イシュー #3111・
//! 親 #3108。設計 `docs/device-resident-kv-paged-attention-design.md` §3〜§5）。
//!
//! # 役割と責務境界
//!
//! [`CudaPagedKv::new`] が `CudaDevice` から 2 カーネル（`kernels_paged_kv.rs`）を NVRTC
//! コンパイルして保持し、[`CudaPagedKv::launch_append`]・[`CudaPagedKv::launch_attention`]
//! がデバイス常駐の入出力（`CudaArg`／`CudaArgMut`）へカーネルを積む（`kl_div.rs`・`bce.rs`
//! と同型構成。取得は `context_cache::cached_paged_kv`）。`ops.rs::CudaBackendOps` の
//! `paged_kv_append`／`paged_attention_forward` が唯一の呼び出し元で、**poison・世代検査・
//! 引数検証（`tensor-core::validate_paged_*`）は呼び出し元が済ませる**。本ファイルは
//! launch 直前にバッファ長とカーネル引数型（`int`）への収まりを再検証する（多層防御。
//! OWASP A03）。
//!
//! # ストリーム順序
//!
//! 起動は同一ストリームに積むだけで同期しない（同期点は呼び出し元の `download`。
//! `docs/backend-cuda-async-execution-design.md`）。`slot_mapping`・`block_tables`・
//! `seq_lens`・作業バッファは `clone_htod`／プール確保のストリーム順序付きデバイス
//! メモリで、カーネル完了後にストリーム順序で解放される（`CudaSlice` の drop は
//! `cuMemFreeAsync`）。
//!
//! # unsafe
//!
//! `launch_builder(..).launch(..)` の 2 箇所のみ（FFI 境界）。各ブロックに `// SAFETY:` で
//! バッファ長・カーネル内の境界検査・生存期間の根拠を書く。監査記録は
//! `docs/device-resident-kv-unsafe-audit.md`。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use fandhe_ai_tensor_core::{PagedAttentionArgs, PagedAttentionPlan, PagedKvLayout};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_paged_kv::{
    PAGED_APPEND_BLOCK_DIM, PAGED_APPEND_MAX_BLOCKS, PAGED_ATTENTION_BLOCK_DIM,
    PAGED_ATTENTION_F32, PAGED_KV_APPEND_F32,
};
use crate::memory::{CudaArg, CudaArgMut};
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

fn shape_err(detail: String) -> CudaError {
    CudaError::InvalidElementwiseShape { detail }
}

fn to_i32(name: &str, v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| shape_err(format!("paged_kv: {name} が i32::MAX を超える（{v}）")))
}

/// append の grid 幅（`total` 要素を `PAGED_APPEND_BLOCK_DIM` スレッドで grid-stride）。
/// GPU を使わない単体テストのため pure 関数として切り出す。
pub(crate) fn append_grid_blocks(total: usize) -> u32 {
    let blocks = total.div_ceil(PAGED_APPEND_BLOCK_DIM as usize);
    u32::try_from(blocks)
        .unwrap_or(PAGED_APPEND_MAX_BLOCKS)
        .clamp(1, PAGED_APPEND_MAX_BLOCKS)
}

/// append のカーネル引数（`rows`・`embed_dim`・`pool_rows`）と grid を、launch 前に型検査して
/// 求める。要素数の整合（入力 `rows · E`・プール `num_blocks · block_size · E`）も確認する。
pub(crate) fn append_launch_params(
    k_new_len: usize,
    v_new_len: usize,
    k_pool_len: usize,
    v_pool_len: usize,
    rows: usize,
    layout: &PagedKvLayout,
) -> Result<(i32, i32, i64, u32), CudaError> {
    let rows_i = to_i32("rows", rows)?;
    let e_i = to_i32("embed_dim", layout.embed_dim)?;
    let total = rows
        .checked_mul(layout.embed_dim)
        .ok_or_else(|| shape_err("paged_kv: 追記要素数が usize を超える".into()))?;
    let pool = layout
        .pool_elems()
        .map_err(|e| shape_err(format!("paged_kv: {e}")))?;
    if k_new_len != total || v_new_len != total || k_pool_len != pool || v_pool_len != pool {
        return Err(shape_err(format!(
            "paged_kv: バッファ長が一致しない（k_new={k_new_len}・v_new={v_new_len}・\
             k_pool={k_pool_len}・v_pool={v_pool_len}・期待 入力={total} プール={pool}）"
        )));
    }
    let pool_rows = layout
        .pool_rows()
        .map_err(|e| shape_err(format!("paged_kv: {e}")))?;
    let pool_rows_i = i64::try_from(pool_rows)
        .map_err(|_| shape_err("paged_kv: pool_rows が i64 に収まらない".into()))?;
    Ok((rows_i, e_i, pool_rows_i, append_grid_blocks(total)))
}

/// attention のカーネル引数（`int`／`unsigned`）をまとめた値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AttentionLaunchParams {
    pub(crate) batch: i32,
    pub(crate) l_new: i32,
    pub(crate) num_heads: i32,
    pub(crate) head_dim: i32,
    pub(crate) embed_dim: i32,
    pub(crate) block_size: i32,
    pub(crate) num_blocks: u32,
    pub(crate) max_blocks_per_seq: i32,
    pub(crate) s_stride: i32,
    pub(crate) scale: f32,
    pub(crate) grid_blocks: u32,
}

/// attention のカーネル引数を launch 前に型検査して求める（GPU 不要）。
pub(crate) fn attention_launch_params(
    args: &PagedAttentionArgs,
    plan: &PagedAttentionPlan,
) -> Result<AttentionLaunchParams, CudaError> {
    Ok(AttentionLaunchParams {
        batch: to_i32("batch", args.batch)?,
        l_new: to_i32("l_new", args.l_new)?,
        num_heads: to_i32("num_heads", args.num_heads)?,
        head_dim: to_i32("head_dim", args.head_dim())?,
        embed_dim: to_i32("embed_dim", args.layout.embed_dim)?,
        block_size: to_i32("block_size", args.layout.block_size)?,
        num_blocks: u32::try_from(args.layout.num_blocks)
            .map_err(|_| shape_err("paged_kv: num_blocks が u32 に収まらない".into()))?,
        max_blocks_per_seq: to_i32("max_blocks_per_seq", args.max_blocks_per_seq)?,
        s_stride: to_i32("s_max", plan.s_max)?,
        scale: args.scale,
        grid_blocks: u32::try_from(plan.query_rows)
            .map_err(|_| shape_err("paged_kv: query 行数が u32 に収まらない".into()))?,
    })
}

/// paged KV 2 カーネル（append・attention。いずれも f32）のコンパイル済みハンドル。
pub struct CudaPagedKv {
    stream: Arc<CudaStream>,
    allocator: Arc<CudaAllocator>,
    append_f32: CudaFunction,
    attention_f32: CudaFunction,
}

impl CudaPagedKv {
    /// `device` 上で 2 カーネルを NVRTC コンパイルして保持する（`kl_div.rs::CudaKlDiv::new`
    /// と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();
        let append_ptx = compile_ptx(PAGED_KV_APPEND_F32, arch)?;
        let attention_ptx = compile_ptx(PAGED_ATTENTION_F32, arch)?;
        let append_f32 = device
            .context()
            .load_module(append_ptx)?
            .load_function("paged_kv_append_f32")?;
        let attention_f32 = device
            .context()
            .load_module(attention_ptx)?
            .load_function("paged_attention_f32")?;
        let allocator = context_cache::cached_allocator(device)?;
        Ok(Self {
            stream: device.stream().clone(),
            allocator,
            append_f32,
            attention_f32,
        })
    }

    /// スロット写像どおりに `k_new`／`v_new` の行を K／V プールへ書くカーネルを積む。
    /// 同期しない。`slot_mapping` の検証（範囲・重複）は呼び出し元が済ませている前提。
    pub(crate) fn launch_append(
        &self,
        k_new: &CudaArg<'_>,
        v_new: &CudaArg<'_>,
        k_pool: &mut CudaArgMut<'_>,
        v_pool: &mut CudaArgMut<'_>,
        slot_mapping: &[u32],
        layout: &PagedKvLayout,
    ) -> Result<(), CudaError> {
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
        let slots_dev = self.stream.clone_htod(slot_mapping)?;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (PAGED_APPEND_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `k_new`／`v_new` は `rows · embed_dim` 要素、`k_pool`／`v_pool` は
        // `num_blocks · block_size · embed_dim` 要素であることを `append_launch_params` が
        // launch 直前に検証済み。`slots_dev` は `rows` 要素の H2D 済みバッファで、カーネルは
        // `r < rows`・`e < embed_dim`・`slot < pool_rows`（= プール行数）を手動検査してから
        // `dst = slot · embed_dim + e`（64bit）へ書く（REQ-8）ため OOB 書き込みは起きない。
        // 重複スロットは呼び出し元（`validate_paged_kv_append`）が拒否済みで書き込みは競合
        // しない。`slots_dev` は同一ストリーム上でカーネルより後に drop される（ストリーム
        // 順序付き解放）ので、カーネル実行中に解放されない。引数は `launch` 完了まで
        // ローカルで生存する。
        unsafe {
            let mut builder = self.stream.launch_builder(&self.append_f32);
            k_new.push(&mut builder);
            v_new.push(&mut builder);
            k_pool.push(&mut builder);
            v_pool.push(&mut builder);
            builder
                .arg(&slots_dev)
                .arg(&rows_i)
                .arg(&e_i)
                .arg(&pool_rows)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// ブロックを論理順に走査する attention カーネルを積む（縮約順序は
    /// `kernels_paged_kv.rs` のモジュール doc）。同期しない。引数検証と `plan` の算出は
    /// 呼び出し元（`validate_paged_attention`）が済ませている前提。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_attention(
        &self,
        q: &CudaArg<'_>,
        k_pool: &CudaArg<'_>,
        v_pool: &CudaArg<'_>,
        block_tables: &[u32],
        seq_lens: &[u32],
        out: &mut CudaArgMut<'_>,
        args: &PagedAttentionArgs,
        plan: &PagedAttentionPlan,
    ) -> Result<(), CudaError> {
        if plan.query_rows == 0 || plan.out_numel == 0 {
            return Ok(());
        }
        let p = attention_launch_params(args, plan)?;
        let pool = args
            .layout
            .pool_elems()
            .map_err(|e| shape_err(format!("paged_kv: {e}")))?;
        let tables_len = args
            .batch
            .checked_mul(args.max_blocks_per_seq)
            .ok_or_else(|| shape_err("paged_kv: block_tables の要素数が usize を超える".into()))?;
        if q.len() != plan.out_numel
            || out.len() != plan.out_numel
            || k_pool.len() != pool
            || v_pool.len() != pool
            || block_tables.len() != tables_len
            || seq_lens.len() != args.batch
        {
            return Err(shape_err(format!(
                "paged_kv: attention のバッファ長が一致しない（q={}・out={}・k_pool={}・\
                 v_pool={}・tables={}・seq_lens={}）",
                q.len(),
                out.len(),
                k_pool.len(),
                v_pool.len(),
                block_tables.len(),
                seq_lens.len()
            )));
        }
        let tables_dev = self.stream.clone_htod(block_tables)?;
        let seq_lens_dev = self.stream.clone_htod(seq_lens)?;
        // score 作業バッファ。各 query 行ブロックが `[0, limit)` を書いてから読む
        // （書く前に読む経路はない。違反時は NaN を出力して抜ける）ので未初期化確保でよい。
        let mut scores_dev = self.allocator.alloc_uninit_f32(plan.workspace_elems)?;
        let mut scores_view = scores_dev.as_view_mut();
        let cfg = LaunchConfig {
            grid_dim: (p.grid_blocks, 1, 1),
            block_dim: (PAGED_ATTENTION_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `q`／`out` は `B · L_new · E`、`k_pool`／`v_pool` は
        // `num_blocks · block_size · E` 要素であることを上で検証済み（要素数は
        // `validate_paged_attention` の `plan` と一致）。`tables_dev`・`seq_lens_dev` は
        // それぞれ `B · max_blocks_per_seq`・`B` 要素の H2D 済みバッファ。`scores_dev` は
        // `plan.workspace_elems = (B · H · L_new) · s_max` 要素で、grid は `B · H · L_new`
        // ブロック、各ブロックは自分の行 `[qrow · s_stride, qrow · s_stride + limit)`
        // （`limit <= s_stride` をカーネルが検査）だけを書く。カーネルは `b < batch`・
        // ブロックテーブル添字 `< max_blocks_per_seq`・ブロック ID `< num_blocks`・
        // `seq_len` の範囲を手動検査し、違反時は NaN を書いて抜ける（REQ-8）。オフセットは
        // 64bit。ローカルの各デバイスバッファは同一ストリーム上でカーネルより後に drop
        // される（ストリーム順序付き解放）ので実行中に解放されない。`out` は呼び出し元が
        // 排他的に借用している。
        unsafe {
            let mut builder = self.stream.launch_builder(&self.attention_f32);
            q.push(&mut builder);
            k_pool.push(&mut builder);
            v_pool.push(&mut builder);
            builder
                .arg(&tables_dev)
                .arg(&seq_lens_dev)
                .arg(&mut scores_view);
            out.push(&mut builder);
            builder
                .arg(&p.batch)
                .arg(&p.l_new)
                .arg(&p.num_heads)
                .arg(&p.head_dim)
                .arg(&p.embed_dim)
                .arg(&p.block_size)
                .arg(&p.num_blocks)
                .arg(&p.max_blocks_per_seq)
                .arg(&p.s_stride)
                .arg(&p.scale)
                .launch(cfg)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::{validate_paged_attention, validate_paged_kv_append};

    fn layout() -> PagedKvLayout {
        PagedKvLayout::new(4, 2, 4).unwrap()
    }

    #[test]
    fn append_params_accept_consistent_lengths() {
        let l = layout();
        let (rows_i, e_i, pool_rows, blocks) = append_launch_params(8, 8, 32, 32, 2, &l).unwrap();
        assert_eq!((rows_i, e_i, pool_rows, blocks), (2, 4, 8, 1));
        // 検証関数と整合する入力。
        assert_eq!(
            validate_paged_kv_append(8, 8, 32, 32, &[0, 7], &l).unwrap(),
            2
        );
    }

    #[test]
    fn append_params_reject_mismatched_lengths() {
        let l = layout();
        assert!(append_launch_params(7, 8, 32, 32, 2, &l).is_err());
        assert!(append_launch_params(8, 8, 31, 32, 2, &l).is_err());
        assert!(append_launch_params(8, 8, 32, 32, usize::MAX, &l).is_err());
        // rows が i32::MAX を超える。
        assert!(append_launch_params(0, 0, 32, 32, i32::MAX as usize + 1, &l).is_err());
    }

    #[test]
    fn append_grid_is_clamped() {
        assert_eq!(append_grid_blocks(1), 1);
        assert_eq!(append_grid_blocks(256), 1);
        assert_eq!(append_grid_blocks(257), 2);
        assert_eq!(append_grid_blocks(usize::MAX), PAGED_APPEND_MAX_BLOCKS);
    }

    #[test]
    fn attention_params_follow_plan() {
        let l = layout();
        let args = PagedAttentionArgs::new(l, 2, 1, 2, 2, 0.5).unwrap();
        let plan = validate_paged_attention(8, 32, 32, &[0, 1, 2, 3], &[3, 3], &args).unwrap();
        let p = attention_launch_params(&args, &plan).unwrap();
        assert_eq!(p.batch, 2);
        assert_eq!(p.head_dim, 2);
        assert_eq!(p.s_stride, 3);
        assert_eq!(p.grid_blocks, 4);
        assert_eq!(p.num_blocks, 4);
        assert_eq!(p.scale, 0.5);
    }
}
