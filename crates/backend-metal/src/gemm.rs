//! GEMM 公開入口（naive: TASK-1.8b・#39／tiled・simdgroup: TASK-1.8c・#40）。
//!
//! [`crate::pipeline::compile_gemm_library`]・[`crate::pipeline::make_pipeline`]
//! でビルドした `gemm_naive`/`gemm_tiled`/`gemm_simdgroup` の 3 パイプライン
//! を保持する [`MetalGemm`] を介して [`crate::context::MetalContext::dispatch_sync`]
//! へエンコーダ結線を委ね、[`crate::buffer::MetalBuffer`] で A・B・C を
//! 確保・readback する。`GemmVariant::Simdgroup` は [`crate::pad`] で
//! 8 の倍数へパディングした実効次元でディスパッチし、readback 後に
//! 元の m×n 形状へ切り出す（呼び出し元へパディングを隠蔽する）。
//! `backend_cpu::parity::matmul_reference_fma`（本クレートの `dev-dependencies`
//! 経由）との数値一致（REQ-2 統一複合判定）は `tests/gemm_naive_parity.rs`・
//! `tests/gemm_simdgroup_parity.rs` で検証する。
//!
//! **移植元**: `docs/spec/03-poc/poc-v2-4-metal-gemm/code/rust/src/metal_gemm.rs`
//! の `MetalGemm::prepare`/`GemmCase::dispatch`。PoC の `unsafe`・`expect`
//! 呼び出しは維持しつつ、確保直前の形状検証を追加し型付きエラー化した
//! （coding-rust.md）。
//!
//! `MetalGemm` は `pipeline_naive`/`pipeline_tiled`/`pipeline_simdgroup` の
//! 3 パイプラインを並べて保持する（`docs/spec/.../metal_gemm.rs` の
//! `MetalGemm { pipeline_naive, pipeline_tiled, pipeline_simdgroup, .. }`
//! と同型の設計。#39 時点は naive のみを productize していた）。

use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pad::{pad_matrix, pad8, unpad_matrix};
use crate::pipeline::{self, MtlPipeline};

/// threadgroup サイズ（16×16）。`Naive`/`Tiled` 用（PoC-v2-4 実測構成。
/// `metal_gemm.rs` の naive/tiled 両段で採用）を踏襲する。grid は
/// `div_ceil(16)` で切り上げ、はみ出す末尾スレッドは `shaders/gemm.metal`
/// 側の手動境界チェック（REQ-8）で無視される。
const THREADGROUP_SIDE: usize = 16;

/// `Simdgroup` 用の threadgroup 幅（32 スレッド = 1 simdgroup。高さは 1）。
/// grid は実効次元（8 の倍数に padding 済み）を 8 で割った値になり、
/// `div_ceil` は不要（`shaders/gemm.metal` の `gemm_simdgroup` コメント
/// 参照。1 threadgroup が C の 8×8 タイルを 1 つ担当する）。
const SIMDGROUP_THREADGROUP_WIDTH: usize = 32;

/// `shaders/gemm.metal` の 3 段カーネルのどれを使うかを表す。
///
/// [`MetalGemm::dispatch_variant`] が本 enum で選択したパイプラインへ
/// ディスパッチする（`docs/spec/.../metal_gemm.rs` の `GemmVariant` と
/// 同型）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmVariant {
    /// 素朴な 3 重ループ（タイル化なし）。
    Naive,
    /// threadgroup 共有メモリによるタイル化。
    Tiled,
    /// `simdgroup_matrix`（8x8 ハードウェア行列演算命令）。
    /// m・n・k を 8 の倍数にパディングして実行する。
    Simdgroup,
}

impl GemmVariant {
    /// `shaders/gemm.metal` 内の対応する `kernel` 関数名。
    fn function_name(self) -> &'static str {
        match self {
            GemmVariant::Naive => "gemm_naive",
            GemmVariant::Tiled => "gemm_tiled",
            GemmVariant::Simdgroup => "gemm_simdgroup",
        }
    }
}

/// `shaders/gemm.metal` の `Dims` 構造体とレイアウトを一致させる
/// （`repr(C)`・12 バイト）。`crate::gemm::naive` が形状検証後にここへ
/// キャストして `setBytes_length_atIndex` で渡す。
// `Debug` は `tests` の `Result<Dims, MetalError>` に対する `unwrap_err()`
// が `T: Debug` を要求するため（`buffer.rs::MetalBuffer` の `Debug` 導出
// 理由と同種。clippy 経由でも要求される）。
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Dims {
    m: u32,
    n: u32,
    k: u32,
}

/// naive・tiled・simdgroup の 3 パイプラインを保持するハンドル。
///
/// [`MetalContext`] とは別に保持する理由: パイプライン構築（MSL コンパイル
/// 込み）は比較的重い処理であり、`MetalGemm::new` を 1 回呼んで使い回す
/// ことを想定する（`MetalContext` はデバイス・キューのみの軽量ハンドル。
/// TASK-1.8a・#38 の責務分離を維持する）。
pub struct MetalGemm {
    pipeline_naive: objc2::rc::Retained<MtlPipeline>,
    pipeline_tiled: objc2::rc::Retained<MtlPipeline>,
    pipeline_simdgroup: objc2::rc::Retained<MtlPipeline>,
}

impl MetalGemm {
    /// `shaders/gemm.metal` を `ctx` のデバイス上でコンパイルし、
    /// `gemm_naive`/`gemm_tiled`/`gemm_simdgroup` の 3 パイプラインを
    /// 構築する。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let library = pipeline::compile_gemm_library(ctx.device())?;
        let pipeline_naive =
            pipeline::make_pipeline(ctx.device(), &library, GemmVariant::Naive.function_name())?;
        let pipeline_tiled =
            pipeline::make_pipeline(ctx.device(), &library, GemmVariant::Tiled.function_name())?;
        let pipeline_simdgroup = pipeline::make_pipeline(
            ctx.device(),
            &library,
            GemmVariant::Simdgroup.function_name(),
        )?;
        Ok(Self {
            pipeline_naive,
            pipeline_tiled,
            pipeline_simdgroup,
        })
    }

    fn pipeline_for(&self, variant: GemmVariant) -> &MtlPipeline {
        match variant {
            GemmVariant::Naive => &self.pipeline_naive,
            GemmVariant::Tiled => &self.pipeline_tiled,
            GemmVariant::Simdgroup => &self.pipeline_simdgroup,
        }
    }

    /// naive GEMM（`C = A @ B`。ゼロ初期化した C へのディスパッチ 1 回のみ、
    /// 蓄積なし）を実行し、結果をホストへ読み出す。[`GemmVariant::Naive`]
    /// で [`Self::dispatch_variant`] へ委譲する薄いラッパー（#39 時点の
    /// 既存呼び出し元・テストを壊さないため温存する。TASK-1.8c・#40）。
    pub fn dispatch(
        &self,
        ctx: &MetalContext,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, MetalError> {
        self.dispatch_variant(ctx, GemmVariant::Naive, a, b, m, n, k)
    }

    /// `variant` で選択した GEMM カーネルをディスパッチし、結果をホストへ
    /// 読み出す（TASK-1.8c・#40）。
    ///
    /// 形状検証（`m/n/k == 0` 拒否・`a.len() == m*k`・`b.len() == k*n`・
    /// `checked_mul` によるオーバーフロー検出・`u32::MAX` 超過検出）を
    /// FFI 呼び出し前に行う（OWASP A03 観点。`.claude/rules/security.md`。
    /// 将来 safetensors/ONNX 由来の形状がこの経路に流入する前提の前段
    /// 検証）。[`GemmVariant::Simdgroup`] の場合はさらに [`pad8`] で
    /// 実効次元（8 の倍数）を算出し、パディングにより増える積（m_eff*k_eff
    /// 等）にも同じオーバーフロー・`u32::MAX` 検証を [`validate_effective_dims`]
    /// で通す（元 shape の検証だけでは実効次元側の桁あふれを見逃すため）。
    /// [`pad_matrix`] で A・B を実効次元へ 0 パディングしてから
    /// [`MetalBuffer::new_with_data`]・[`MetalBuffer::new_zeroed`] で
    /// バッファを確保し、[`MetalContext::dispatch_sync`] のクロージャ内で
    /// パイプライン・バッファ（index 0〜2）・`Dims`（index 3）を結線して
    /// ディスパッチする。readback 後は [`unpad_matrix`] で元の m×n 形状へ
    /// 切り出す（`Naive`/`Tiled` は実効次元 = 元次元のため実質無変換）。
    ///
    /// `#[allow(clippy::too_many_arguments)]`: `variant`・`a`・`b`・`m`・
    /// `n`・`k` を個別引数として持つことで呼び出し側の意図が明確になる
    /// ため、構造体へのまとめ込みは行わない（`backend_cpu::gemm::kernel_block`
    /// と同じ判断根拠。理由コメント必須のルール `.claude/rules/coding-rust.md`
    /// に対応）。
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_variant(
        &self,
        ctx: &MetalContext,
        variant: GemmVariant,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, MetalError> {
        validate_dims(a, b, m, n, k)?;

        let (m_eff, n_eff, k_eff) = if variant == GemmVariant::Simdgroup {
            (pad8(m), pad8(n), pad8(k))
        } else {
            (m, n, k)
        };
        let dims = validate_effective_dims(m_eff, n_eff, k_eff)?;

        let a_padded = pad_matrix(a, m, k, m_eff, k_eff);
        let b_padded = pad_matrix(b, k, n, k_eff, n_eff);

        // `MetalBuffer` の確保エラーはそのまま `MetalError`
        // （`crate::error::MetalError`）であり、本関数の戻り値の
        // `Result<_, MetalError>` と型が一致するため変換不要（`?` で
        // そのまま伝播する）。
        let a_buf = MetalBuffer::new_with_data(ctx, &a_padded)?;
        let b_buf = MetalBuffer::new_with_data(ctx, &b_padded)?;
        let c_buf = MetalBuffer::new_zeroed(ctx, m_eff * n_eff)?;

        let pipeline = self.pipeline_for(variant);
        ctx.dispatch_sync(|encoder| {
            encode_dispatch(encoder, pipeline, &a_buf, &b_buf, &c_buf, dims, variant);
        })?;

        let padded_c = c_buf.read_to_vec();
        Ok(unpad_matrix(&padded_c, m_eff, n_eff, m, n))
    }
}

/// `m/n/k == 0` 拒否・長さ一致検証・`checked_mul` によるオーバーフロー
/// 検出・`u32::MAX` 超過検出（`Dims` への cast 前検証）を行う。
///
/// `backend_cpu::gemm::validate_dims`（`crates/backend-cpu/src/gemm.rs`）
/// と同種の検証を Metal 側の型（[`MetalError`]・`u32` cast 制約）に
/// 合わせて独立実装する（クレートをまたいだ検証ロジック共有は本イシュー
/// のスコープ外。#40 以降で共通化が必要になった場合に判断する）。
fn validate_dims(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Dims, MetalError> {
    if m == 0 || n == 0 || k == 0 {
        return Err(MetalError::ZeroDimension { m, n, k });
    }

    let mk = m.checked_mul(k).ok_or(MetalError::DimProductOverflow)?;
    let kn = k.checked_mul(n).ok_or(MetalError::DimProductOverflow)?;
    // m*n は C バッファの要素数算出（`MetalBuffer::new_zeroed`）に使われ、
    // その内部で改めて `checked_mul` 相当の検証（`checked_byte_len`）が
    // 走るが、`Dims` の妥当性確認としてここでも算出しオーバーフローを
    // 早期検出する。
    m.checked_mul(n).ok_or(MetalError::DimProductOverflow)?;

    // `u32` 範囲チェックを長さ一致検証より先に行う理由: このチェックが
    // 対象とする「巨大な m/n/k」は `a`/`b` の実長さも数十億要素級になり、
    // 単体テスト（`validate_dims_rejects_dimension_exceeding_u32`）が
    // 実際にそのサイズのベクタを確保するとメモリを圧迫する。長さ検証より
    // 先に軽量な数値比較のみで拒否できるようにし、テストは小さな `a`/`b`
    // （長さ不一致）のまま `DimensionExceedsU32` を観測できるようにする。
    if m > u32::MAX as usize || n > u32::MAX as usize || k > u32::MAX as usize {
        return Err(MetalError::DimensionExceedsU32 { m, n, k });
    }

    if a.len() != mk {
        return Err(MetalError::ALenMismatch {
            expected: mk,
            actual: a.len(),
        });
    }
    if b.len() != kn {
        return Err(MetalError::BLenMismatch {
            expected: kn,
            actual: b.len(),
        });
    }

    Ok(Dims {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    })
}

/// `GemmVariant::Simdgroup` の実効次元（[`pad8`] で 8 の倍数に切り上げた
/// m_eff/n_eff/k_eff）に対する `checked_mul` オーバーフロー検出・
/// `u32::MAX` 超過検出を行う（`Dims` への cast 前検証）。
///
/// [`validate_dims`] は元 shape（`a`/`b` の実長さ）に対する検証であり、
/// パディングにより増加する積（m_eff*k_eff 等）はここで別途検証する
/// 必要がある（[`MetalGemm::dispatch_variant`] は `validate_dims` 通過後に
/// 本関数を呼ぶ）。`Naive`/`Tiled` は m_eff/n_eff/k_eff がそのまま
/// m/n/k と一致するため、`validate_dims` で検証済みの範囲を再確認する
/// だけで実質的にオーバーフローしない。
fn validate_effective_dims(m_eff: usize, n_eff: usize, k_eff: usize) -> Result<Dims, MetalError> {
    m_eff
        .checked_mul(k_eff)
        .ok_or(MetalError::DimProductOverflow)?;
    k_eff
        .checked_mul(n_eff)
        .ok_or(MetalError::DimProductOverflow)?;
    m_eff
        .checked_mul(n_eff)
        .ok_or(MetalError::DimProductOverflow)?;

    if m_eff > u32::MAX as usize || n_eff > u32::MAX as usize || k_eff > u32::MAX as usize {
        return Err(MetalError::DimensionExceedsU32 {
            m: m_eff,
            n: n_eff,
            k: k_eff,
        });
    }

    Ok(Dims {
        m: m_eff as u32,
        n: n_eff as u32,
        k: k_eff as u32,
    })
}

/// パイプライン設定・バッファ結線（index 0〜2）・`Dims`（index 3）の
/// `setBytes`・`dispatchThreadgroups_threadsPerThreadgroup` を行う。
/// [`MetalGemm::dispatch_variant`] が [`MetalContext::dispatch_sync`]
/// のクロージャから呼ぶ。`variant` によって threadgroup サイズ・grid
/// の計算方式が異なる（`Naive`/`Tiled` は 16×16・`div_ceil(16)`、
/// `Simdgroup` は 32×1・`n_eff/8 × m_eff/8`。`shaders/gemm.metal` の
/// 各カーネルのディスパッチ形状契約と一致させる）。
fn encode_dispatch(
    encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    a_buf: &MetalBuffer,
    b_buf: &MetalBuffer,
    c_buf: &MetalBuffer,
    dims: Dims,
    variant: GemmVariant,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY（FFI 境界 1/2）: `setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きはしない
    // （実際のアクセスは `dispatchThreadgroups` 後、GPU 側の非同期実行で
    // 発生する）。`a_buf`/`b_buf`/`c_buf` は本関数の呼び出し元
    // （`MetalGemm::dispatch`）のスタックフレームで `ctx.dispatch_sync`
    // が完了するまで生存しており（`dispatch_sync` は同期実行で
    // `waitUntilCompleted()` まで戻らない）、エンコード中に破棄されない。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(c_buf.raw()), 0, 2);
    }

    // SAFETY（FFI 境界 2/2）: `setBytes_length_atIndex` は指定ポインタから
    // `size_of::<Dims>()` バイトを即座に複製する（PoC-v2-4 `metal_gemm.rs`
    // と同じ呼び出し形）。`dims` はローカル変数でありポインタは本呼び出し
    // 中生存し、長さは `size_of::<Dims>()` と正確に一致する。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&dims).cast(),
            std::mem::size_of::<Dims>(),
            3,
        );
    }

    // `Simdgroup` は 1 threadgroup（32 スレッド = 1 simdgroup）が C の 8×8
    // タイルを 1 つ担当するため、grid は実効次元（`dispatch_variant` が
    // `pad8` で 8 の倍数に揃え済み）をちょうど 8 で割った値になる
    // （`div_ceil` 不要。`shaders/gemm.metal` の `gemm_simdgroup` と同じ
    // ディスパッチ形状契約）。`Naive`/`Tiled` は 16×16 threadgroup・
    // `div_ceil(16)` grid（既存構成を維持）。
    let (tg_w, tg_h, grid_w, grid_h) = match variant {
        GemmVariant::Simdgroup => (
            SIMDGROUP_THREADGROUP_WIDTH,
            1,
            (dims.n as usize) / 8,
            (dims.m as usize) / 8,
        ),
        GemmVariant::Naive | GemmVariant::Tiled => (
            THREADGROUP_SIDE,
            THREADGROUP_SIDE,
            (dims.n as usize).div_ceil(THREADGROUP_SIDE),
            (dims.m as usize).div_ceil(THREADGROUP_SIDE),
        ),
    };
    let threads_per_tg = MTLSize {
        width: tg_w,
        height: tg_h,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: grid_w,
        height: grid_h,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- validate_dims（pure・実機不要） ---

    #[test]
    fn validate_dims_accepts_valid_shape() {
        let a = vec![0.0f32; 6]; // m=2, k=3
        let b = vec![0.0f32; 12]; // k=3, n=4
        let dims = validate_dims(&a, &b, 2, 4, 3).unwrap();
        assert_eq!((dims.m, dims.n, dims.k), (2, 4, 3));
    }

    #[test]
    fn validate_dims_rejects_zero_m() {
        let err = validate_dims(&[], &[], 0, 4, 3).unwrap_err();
        assert!(matches!(
            err,
            MetalError::ZeroDimension { m: 0, n: 4, k: 3 }
        ));
    }

    #[test]
    fn validate_dims_rejects_zero_n() {
        let a = vec![0.0f32; 6];
        let err = validate_dims(&a, &[], 2, 0, 3).unwrap_err();
        assert!(matches!(
            err,
            MetalError::ZeroDimension { m: 2, n: 0, k: 3 }
        ));
    }

    #[test]
    fn validate_dims_rejects_a_len_mismatch() {
        let a = vec![0.0f32; 5]; // m*k=6 を期待
        let b = vec![0.0f32; 12];
        let err = validate_dims(&a, &b, 2, 4, 3).unwrap_err();
        assert!(matches!(
            err,
            MetalError::ALenMismatch {
                expected: 6,
                actual: 5
            }
        ));
    }

    #[test]
    fn validate_dims_rejects_b_len_mismatch() {
        let a = vec![0.0f32; 6];
        let b = vec![0.0f32; 11]; // k*n=12 を期待
        let err = validate_dims(&a, &b, 2, 4, 3).unwrap_err();
        assert!(matches!(
            err,
            MetalError::BLenMismatch {
                expected: 12,
                actual: 11
            }
        ));
    }

    #[test]
    fn validate_dims_rejects_dim_product_overflow() {
        let huge = usize::MAX / 2 + 1;
        let err = validate_dims(&[], &[], huge, huge, 2).unwrap_err();
        assert!(matches!(err, MetalError::DimProductOverflow));
    }

    #[test]
    fn validate_dims_rejects_dimension_exceeding_u32() {
        let over_u32 = u32::MAX as usize + 1;
        // u32 範囲チェックは長さ一致検証より先に行う（実装コメント参照）
        // ため、`a`/`b` は実サイズ（数十億要素）を確保せず空スライスの
        // まま `DimensionExceedsU32` を観測できる。
        let err = validate_dims(&[], &[], over_u32, 1, 1).unwrap_err();
        assert!(matches!(
            err,
            MetalError::DimensionExceedsU32 { m, n: 1, k: 1 } if m == over_u32
        ));
    }

    // --- validate_effective_dims（pure・実機不要） ---

    #[test]
    fn validate_effective_dims_accepts_padded_shape() {
        // pad8(37)=40, pad8(41)=48, pad8(53)=56 相当の実効次元。
        let dims = validate_effective_dims(40, 48, 56).unwrap();
        assert_eq!((dims.m, dims.n, dims.k), (40, 48, 56));
    }

    #[test]
    fn validate_effective_dims_rejects_dim_product_overflow() {
        let huge = usize::MAX / 2 + 1;
        let err = validate_effective_dims(huge, huge, 2).unwrap_err();
        assert!(matches!(err, MetalError::DimProductOverflow));
    }

    #[test]
    fn validate_effective_dims_rejects_dimension_exceeding_u32() {
        let over_u32 = u32::MAX as usize + 1;
        let err = validate_effective_dims(over_u32, 8, 8).unwrap_err();
        assert!(matches!(
            err,
            MetalError::DimensionExceedsU32 { m, n: 8, k: 8 } if m == over_u32
        ));
    }

    // --- GemmVariant ---

    #[test]
    fn gemm_variant_function_names_match_shader_kernel_names() {
        assert_eq!(GemmVariant::Naive.function_name(), "gemm_naive");
        assert_eq!(GemmVariant::Tiled.function_name(), "gemm_tiled");
        assert_eq!(GemmVariant::Simdgroup.function_name(), "gemm_simdgroup");
    }
}
