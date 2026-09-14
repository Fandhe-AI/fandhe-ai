//! `unique`（`torch.unique(input, sorted=True)` の values のみ。
//! イシュー #1734）の起動 API（実行時コンパイル・パイプライン保持・
//! 実行）。
//!
//! `gather_scatter.rs::MetalGatherScatter` と同じ構成方針を踏襲する:
//! [`MetalUnique::new`] が `shaders/unique.metal`（`bitonic_step_u32`）
//! を実行時コンパイルしてパイプラインを保持し、[`MetalUnique::
//! run_unique_f32`] へホスト側スライスを渡すだけでバッファ確保・
//! ディスパッチ・readback を内部で完結できる。`ops.rs::
//! MetalBackendOps::unique` から呼ばれる。
//!
//! # アルゴリズム
//!
//! `crates/backend-cuda/src/unique.rs` モジュール doc と同一（`n == 0`／
//! `n == 1` の早期処理・totalOrder キー変換・パディング・ビットニック
//! ソート・先頭 `n` 要素の復元・ホスト側 `dedup_by(==)`）。GPU 側の
//! 違いは、ホストと同一エンコーダ（`ctx.dispatch_sync` が生成する
//! serial encoder）へ全ステップを連続してエンコードし 1 回の
//! `dispatch_sync`（＝1 回の同期）で完結させる点（CUDA の同一ストリーム
//! 上への繰り返し `launch` と同じ「複数ステップを 1 回の同期区間へ
//! まとめる」設計）。エンコーダは `MetalContext::encode` が生成する
//! serial encoder（`computeCommandEncoder`）であり、同一エンコーダ内の
//! 連続 dispatch は投入順に直列実行される契約
//! （`docs/backend-metal-command-batching-design.md`）。
//! `memoryBarrierWithScope` の明示挿入は serial encoder では
//! 「許可されるが無視される」（objc2-metal 生成コードの doc comment
//! 参照）ため厳密には意味を持たないが、将来 concurrent encoder へ
//! 変更された場合の安全側の防御として残す。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBarrierScope, MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::context::MetalContext;
use crate::error::MetalError;
use crate::index_buffer::MetalIndexBuffer;
use crate::pipeline::{self, MtlPipeline};
use crate::unique_model::{UniquePrepareError, checked_padded_len, key_to_f32, total_order_key};

/// `shaders/unique.metal` のソース。
const UNIQUE_MSL_SRC: &str = include_str!("shaders/unique.metal");

/// 1 スレッドグループあたりのスレッド数（`gather_scatter.rs::
/// GS_THREADGROUP_WIDTH` と同じ値・同じ判断根拠）。
const UNIQUE_THREADGROUP_WIDTH: usize = 256;

/// `bitonic_step_u32` カーネルのコンパイル済みパイプラインを保持する
/// ハンドル。
pub struct MetalUnique {
    bitonic_step_u32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalUnique {
    /// `ctx` のデバイス上で `bitonic_step_u32` を実行時コンパイルし
    /// パイプラインを構築する（`gather_scatter.rs::MetalGatherScatter::
    /// new` と同型）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(UNIQUE_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let bitonic_step_u32 = pipeline::make_pipeline(ctx.device(), &library, "bitonic_step_u32")?;

        Ok(Self { bitonic_step_u32 })
    }

    /// `torch.unique(input, sorted=True)` の values（本ファイル冒頭
    /// コメント「アルゴリズム」節参照）。
    ///
    /// `n == 0`／`n == 1` は `MetalIndexBuffer` が 0 バイト確保を拒否する
    /// ため、バッファ確保・`dispatch_sync` に入る前に早期 return する
    /// （`gather_scatter.rs::run_scatter_f32` の空出力早期リターンと
    /// 同じ理由）。`padded` の検証（`checked_padded_len`）・キー生成・
    /// パディングはすべて `dispatch_sync` のクロージャに入る前に完了
    /// させる（クロージャは `Result` を返せないため。`ops.rs` は
    /// `UniquePrepareError::SizeLimitExceeded` を検出してから本メソッドを
    /// 呼ぶ契約——実際には `ops.rs` 側で事前にサイズ検査を行い、本メソッド
    /// は `n < 2` 早期リターン経路以外で `checked_padded_len` が失敗
    /// しない前提の入力のみを受け取る。念のため本メソッド自身も検査
    /// する）。
    pub fn run_unique_f32(&self, ctx: &MetalContext, x: &[f32]) -> Result<Vec<f32>, MetalError> {
        let n = x.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n == 1 {
            return Ok(vec![x[0]]);
        }

        let padded = checked_padded_len(n).map_err(|e: UniquePrepareError| {
            MetalError::InvalidGatherScatterShape {
                detail: e.to_string(),
            }
        })?;

        let mut keys: Vec<u32> = x.iter().map(|&v| total_order_key(v)).collect();
        keys.resize(padded, u32::MAX);

        let keys_buf = MetalIndexBuffer::new_with_u32(ctx, &keys)?;
        let padded_u = padded as u32;

        ctx.dispatch_sync(|encoder| {
            encoder.setComputePipelineState(&self.bitonic_step_u32);

            let mut k = 2usize;
            while k <= padded {
                let mut j = k / 2;
                while j >= 1 {
                    let j_u = j as u32;
                    let k_u = k as u32;
                    encode_bitonic_step(encoder, &keys_buf, j_u, k_u, padded_u);
                    // serial encoder では barrier は許可されるが無視
                    // される（本ファイル冒頭コメント参照）。将来
                    // concurrent encoder へ変更された場合の安全側の
                    // 防御として残す。
                    encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                    j /= 2;
                }
                k *= 2;
            }
        })?;

        let sorted_keys = keys_buf.read_to_vec_u32();
        let mut out: Vec<f32> = sorted_keys[..n].iter().map(|&k| key_to_f32(k)).collect();
        out.dedup_by(|cur, prev| *cur == *prev);
        Ok(out)
    }
}

/// `bitonic_step_u32` の 1 ステップをエンコードする（バッファ index 0・
/// スカラー index 1〜3。`shaders/unique.metal::bitonic_step_u32` の
/// バッファ宣言と一致させる）。
fn encode_bitonic_step(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    keys_buf: &MetalIndexBuffer,
    j: u32,
    k: u32,
    n: u32,
) {
    // SAFETY: FFI 境界 1/2（`gather_scatter.rs::encode_gather_dispatch`
    // と同じ契約）。`setBuffer_offset_atIndex` は生存中の `MTLBuffer`
    // への参照を保持するのみで即座に読み書きしない。`keys_buf` は
    // 呼び出し元 `ctx.dispatch_sync` が完了するまで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(keys_buf.raw()), 0, 0);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタから
    // 指定バイト数を即座に複製する。各ローカル変数は本呼び出し中生存し、
    // 型・バイト数は `shaders/unique.metal::bitonic_step_u32` の
    // `constant uint&` 宣言と一致させている。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&j).cast(),
            std::mem::size_of::<u32>(),
            1,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&k).cast(),
            std::mem::size_of::<u32>(),
            2,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&n).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
    }

    let threads_per_tg = MTLSize {
        width: UNIQUE_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (n as usize).div_ceil(UNIQUE_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}
