//! デバイス常駐 KV（K-3）の Metal 起動 API（実行時 MSL コンパイル・パイプライン保持・
//! encode。イシュー #3111・親 #3108。設計 `docs/device-resident-kv-paged-attention-
//! design.md` §3〜§5。CUDA 側 `backend-cuda::paged_kv` の Metal 対応版）。
//!
//! # 役割と責務境界
//!
//! [`MetalPagedKv::new`] が `shaders/paged_kv.metal` を実行時コンパイルして 2 パイプライン
//! （append・attention）を保持し、[`MetalPagedKv::run_append`]・
//! [`MetalPagedKv::run_attention`] がデバイス常駐の `MetalBuffer` へ encode する
//! （`sgd.rs::MetalSgd::run` と同型）。`ops.rs::MetalBackendOps` の `paged_kv_append*`／
//! `paged_attention_forward*` が唯一の呼び出し元で、引数検証
//! （`tensor-core::validate_paged_*`）は呼び出し元が済ませる。dispatch サイズと
//! `u32` 収容検査は cfg なしの `crate::paged_kv_model`（Linux CI でもテストされる）。
//!
//! # 同期契約（`token` の有無）
//!
//! - `token == None`（非 tracked 版）: encode 直後に `ctx.synchronize()` まで行い、復帰時点で
//!   GPU 実行の完了・成否を返す（「1 呼び出し = 完了保証」）
//! - `token == Some`（tracked 版）: 待たずに返す。実行時エラーは `ctx.encode` と同一ロック区間で
//!   バッチへ登録した `token` に、後続の `synchronize`（ホスト実体化 = `download`）が検出した
//!   とき書き込まれる。呼び出し元（`PagedKvCache`）は `download` 後に token を再検査して
//!   poison する
//!
//! # unsafe
//!
//! encode 関数内の `setBuffer_offset_atIndex`／`setBytes_length_atIndex` に限る（FFI 境界。
//! 既存 encode パターンと同じ範囲）。各ブロックに `// SAFETY:` を付ける。監査記録は
//! `docs/device-resident-kv-unsafe-audit.md`。MSL の実コンパイルと実機一致は Mac 実機
//! （#3172）で確認する（Linux では未検証）。

use fandhe_ai_tensor_core::{
    DispatchFailureCell, PagedAttentionArgs, PagedAttentionPlan, PagedKvLayout,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::index_buffer::MetalIndexBuffer;
use crate::paged_kv_model::{
    self, AppendDispatch, AttentionDispatch, PAGED_APPEND_THREADGROUP_WIDTH,
    PAGED_ATTENTION_THREADGROUP_WIDTH, PAGED_KV_MSL_SRC,
};
use crate::pipeline::{self, MtlPipeline};

fn shape_err(e: fandhe_ai_tensor_core::BackendError) -> MetalError {
    MetalError::InvalidElementwiseShape {
        detail: e.to_string(),
    }
}

/// paged KV 2 カーネルのコンパイル済みパイプライン。
pub struct MetalPagedKv {
    append_f32: Retained<MtlPipeline>,
    attention_f32: Retained<MtlPipeline>,
}

impl MetalPagedKv {
    /// `ctx` のデバイス上で `shaders/paged_kv.metal` を実行時コンパイルしてパイプラインを
    /// 構築する（`sgd.rs::MetalSgd::new` と同一手順）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(PAGED_KV_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;
        let append_f32 = pipeline::make_pipeline(ctx.device(), &library, "paged_kv_append_f32")?;
        let attention_f32 = pipeline::make_pipeline(ctx.device(), &library, "paged_attention_f32")?;
        Ok(Self {
            append_f32,
            attention_f32,
        })
    }

    /// スロット写像どおりに `k_new`／`v_new` の行を K／V プールへ書く encode。
    /// `slot_mapping` の検証（範囲・重複）は呼び出し元が済ませている前提。0 行は何もしない。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_append(
        &self,
        ctx: &MetalContext,
        k_new: &MetalBuffer,
        v_new: &MetalBuffer,
        k_pool: &MetalBuffer,
        v_pool: &MetalBuffer,
        slot_mapping: &[u32],
        layout: &PagedKvLayout,
        token: Option<&DispatchFailureCell>,
    ) -> Result<(), MetalError> {
        let rows = slot_mapping.len();
        if rows == 0 {
            return Ok(());
        }
        let dispatch = paged_kv_model::append_dispatch(
            k_new.len(),
            v_new.len(),
            k_pool.len(),
            v_pool.len(),
            rows,
            layout,
        )
        .map_err(shape_err)?;
        let slots = MetalIndexBuffer::new_with_u32(ctx, slot_mapping)?;

        // `ctx.encode` の `resources` に全バッファを渡し、`Batch::in_flight` へ retain させる
        // （`synchronize` 完了まで実体が生存する。`sgd.rs::MetalSgd::run` と同じ契約）。
        ctx.encode(
            "paged_kv_append_f32",
            &[
                k_new.raw(),
                v_new.raw(),
                k_pool.raw(),
                v_pool.raw(),
                slots.raw(),
            ],
            token,
            |encoder| {
                encode_append(
                    encoder,
                    &self.append_f32,
                    [k_new, v_new, k_pool, v_pool],
                    &slots,
                    &dispatch,
                );
            },
        )?;
        match token {
            None => ctx.synchronize(),
            Some(_) => Ok(()),
        }
    }

    /// ブロックを論理順に走査する attention の encode（縮約順序は `shaders/paged_kv.metal`
    /// 冒頭コメント）。引数検証と `plan` の算出は呼び出し元（`validate_paged_attention`）が
    /// 済ませている前提。`q`・`out` は `B · L_new · E`、K／V プールは
    /// `num_blocks · block_size · E` 要素。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_attention(
        &self,
        ctx: &MetalContext,
        q: &MetalBuffer,
        k_pool: &MetalBuffer,
        v_pool: &MetalBuffer,
        block_tables: &[u32],
        seq_lens: &[u32],
        out: &MetalBuffer,
        args: &PagedAttentionArgs,
        plan: &PagedAttentionPlan,
        token: Option<&DispatchFailureCell>,
    ) -> Result<(), MetalError> {
        if plan.query_rows == 0 || plan.out_numel == 0 {
            return Ok(());
        }
        let dispatch = paged_kv_model::attention_dispatch(args, plan).map_err(shape_err)?;
        let pool = args.layout.pool_elems().map_err(shape_err)?;
        let tables_len = args
            .batch
            .checked_mul(args.max_blocks_per_seq)
            .ok_or_else(|| MetalError::InvalidElementwiseShape {
                detail: "paged_kv: block_tables の要素数が usize を超える".into(),
            })?;
        if q.len() != plan.out_numel
            || out.len() != plan.out_numel
            || k_pool.len() != pool
            || v_pool.len() != pool
            || block_tables.len() != tables_len
            || seq_lens.len() != args.batch
        {
            return Err(MetalError::InvalidElementwiseShape {
                detail: format!(
                    "paged_kv: attention のバッファ長が一致しない（q={}・out={}・k_pool={}・\
                     v_pool={}・tables={}・seq_lens={}）",
                    q.len(),
                    out.len(),
                    k_pool.len(),
                    v_pool.len(),
                    block_tables.len(),
                    seq_lens.len()
                ),
            });
        }
        let tables = MetalIndexBuffer::new_with_u32(ctx, block_tables)?;
        let lens = MetalIndexBuffer::new_with_u32(ctx, seq_lens)?;
        // score 作業バッファ。各 query 行グループが `[0, limit)` を書いてから読む。
        // 非同期 encode でも `resources` へ渡して GPU 完了まで保持する。
        let scores = MetalBuffer::new_zeroed(ctx, plan.workspace_elems)?;

        ctx.encode(
            "paged_attention_f32",
            &[
                q.raw(),
                k_pool.raw(),
                v_pool.raw(),
                tables.raw(),
                lens.raw(),
                scores.raw(),
                out.raw(),
            ],
            token,
            |encoder| {
                encode_attention(
                    encoder,
                    &self.attention_f32,
                    [q, k_pool, v_pool],
                    [&tables, &lens],
                    [&scores, out],
                    &dispatch,
                );
            },
        )?;
        match token {
            None => ctx.synchronize(),
            Some(_) => Ok(()),
        }
    }
}

/// append のバッファ結線（index 0〜4）・スカラー（index 5〜7）・dispatch。
fn encode_append(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    bufs: [&MetalBuffer; 4],
    slots: &MetalIndexBuffer,
    d: &AppendDispatch,
) {
    encoder.setComputePipelineState(pipeline);
    // SAFETY: FFI 境界 1/2。`setBuffer_offset_atIndex` は生存中の `MTLBuffer` への参照を
    // 保持するのみで即座に読み書きしない。`bufs`・`slots` は `MetalPagedKv::run_append` が
    // `ctx.encode` の `resources` へ渡し `Batch::in_flight` へ retain させるため、
    // `ctx.synchronize()` 完了まで生存する。バッファ長は `append_dispatch` が検証済みで、
    // カーネルも `r < rows`・`e < embed_dim`・`slot < pool_rows` を手動検査する（REQ-8）。
    unsafe {
        for (i, b) in bufs.iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(b.raw()), 0, i);
        }
        encoder.setBuffer_offset_atIndex(Some(slots.raw()), 0, 4);
    }
    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタから指定バイト数を
    // 即座に複製する。各ローカルはこの呼び出し中生存し、長さは `constant` 宣言の型
    // （`uint` = 4・`ulong` = 8。`shaders/paged_kv.metal` 参照）と揃えている。
    unsafe {
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&d.rows).cast(), 4, 5);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&d.embed_dim).cast(), 4, 6);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&d.pool_rows).cast(), 8, 7);
    }
    let threads_per_tg = MTLSize {
        width: PAGED_APPEND_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: d.groups_x,
        height: d.groups_y,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// attention のバッファ結線（index 0〜6）・スカラー（index 7〜16）・dispatch。
fn encode_attention(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    f32_in: [&MetalBuffer; 3],
    idx_in: [&MetalIndexBuffer; 2],
    f32_work_out: [&MetalBuffer; 2],
    d: &AttentionDispatch,
) {
    encoder.setComputePipelineState(pipeline);
    // SAFETY: FFI 境界 1/2。`setBuffer_offset_atIndex` は生存中の `MTLBuffer` への参照を
    // 保持するのみで即座に読み書きしない。全バッファは `MetalPagedKv::run_attention` が
    // `ctx.encode` の `resources` へ渡し `Batch::in_flight` へ retain させるため
    // `ctx.synchronize()` 完了まで生存する。要素数は `run_attention` が検証済みで、
    // カーネルも `b < batch`・ブロックテーブル添字・ブロック ID・`seq_len` の範囲を手動検査し、
    // 違反時は NaN を書いて抜ける（REQ-8）。`scores` は `(B · H · L_new) · s_stride` 要素で
    // 各グループは自分の行だけを書く。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(f32_in[0].raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(f32_in[1].raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(f32_in[2].raw()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(idx_in[0].raw()), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(idx_in[1].raw()), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(f32_work_out[0].raw()), 0, 5);
        encoder.setBuffer_offset_atIndex(Some(f32_work_out[1].raw()), 0, 6);
    }
    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタから指定バイト数を
    // 即座に複製する。各ローカルはこの呼び出し中生存し、長さは `constant` 宣言の型
    // （`uint`／`float` = 4。`shaders/paged_kv.metal` 参照）と揃えている。
    unsafe {
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&d.batch).cast(), 4, 7);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&d.l_new).cast(), 4, 8);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&d.num_heads).cast(), 4, 9);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&d.head_dim).cast(), 4, 10);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&d.embed_dim).cast(), 4, 11);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&d.block_size).cast(), 4, 12);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&d.num_blocks).cast(), 4, 13);
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&d.max_blocks_per_seq).cast(),
            4,
            14,
        );
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&d.s_stride).cast(), 4, 15);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&d.scale).cast(), 4, 16);
    }
    let threads_per_tg = MTLSize {
        width: PAGED_ATTENTION_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: d.groups,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}
