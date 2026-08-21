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

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use objc2::rc::Retained;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::half_buffer::MetalHalfBuffer;
use crate::pad::{pad_matrix, pad_matrix_f16, pad8, unpad_matrix, unpad_matrix_f16};
use crate::pipeline::{self, MtlLibrary, MtlPipeline};
use crate::tile::{self, TileConfig};

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

thread_local! {
    /// [`MetalGemm::run_tiled_bias_act_f32`] が実際に GPU カーネルを起動した
    /// 回数（イシュー #605。CUDA 側
    /// `backend_cuda::gemm::BIAS_ACT_FUSED_LAUNCH_COUNT`〈#599〉と同じ
    /// 役割・同じスレッドローカル化の理由）。
    ///
    /// `ops.rs::MetalBackendOps::gemm_bias_act` の経路選択（融合 vs
    /// `tensor_core::backend_ops::BackendOps::gemm_bias_act` デフォルト実装の
    /// 非融合 3 段合成）が実際に融合カーネルへ到達しているかを、実機なしの
    /// 単体テスト（`ops.rs` 内 `#[cfg(test)]`）が検証するための可観測点。
    /// テスト専用の計測であり公開 API の意味論・数値契約には一切影響しない。
    /// `m == 0 || n == 0`（no-op）・`k == 0`（ホスト側で直接 epilogue のみ
    /// 計算し GPU 起動を回避する分岐）の場合はカーネルを起動しないため
    /// カウントしない。バッファ確保・`dispatch_sync` の**成功後**にのみ
    /// 増加させる（codex-review 指摘・PR #717: 確保／dispatch 失敗時に
    /// 「起動済み」として誤記録すると経路検証テスト・診断が偽陽性になる）。
    ///
    /// **スレッドローカルにする理由**（CUDA 側 `gemm.rs` の該当コメントと
    /// 同一の論拠）: `static AtomicU64`（プロセス全体共有）だと `cargo test`
    /// 既定の並列実行下で「`before` 読み取り〜`gemm_bias_act` 呼び出し〜
    /// `after` 読み取り」の間に別スレッドで実行中の別テストが同じ融合
    /// カーネルを起動すると偽陽性になりうる。Rust の既定テストハーネストは
    /// 各テスト関数の実行を単一スレッド内で完結させるため、カウンタを
    /// スレッドローカルにすれば呼び出し元スレッドが実際に起動した回数のみを
    /// 観測できる。
    pub(crate) static BIAS_ACT_FUSED_LAUNCH_COUNT: Cell<u64> = const { Cell::new(0) };
}

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
    /// 動的タイル選択（TASK-1.8f・#188）。`gemm_simdgroup_tiled` を
    /// 保持する [`TileConfig`] の MSL function constant（BM/BN/BK/WM/WN・
    /// `USE_TGP_STAGING`）で特殊化してディスパッチする。行列サイズから
    /// 構成を自動選択する入口は [`MetalGemm::dispatch_auto`]（[`tile::select`]
    /// を使う。`tile::select_with_occupancy` による occupancy 縮退はイシュー
    /// #542 で実装済みだが、`dispatch_auto` への適用はイシュー #747 で
    /// **不採用確定**（#744 是正により、実測帯域〈512/1024/2048/4096〉では
    /// 段 1〈形状判定〉が occupancy 縮退を経ず `select()` と同一結果へ収束
    /// するため。`docs/perf/metal-gemm-occupancy-select.md`「#747 判断」節）。
    /// `m`・`n`・`k` は
    /// `Simdgroup` と同じく 8 の倍数へパディングして実行する
    /// （[`MetalGemm::dispatch_variant`] 参照）。
    SimdgroupTiled(TileConfig),
}

impl GemmVariant {
    /// `shaders/gemm.metal` 内の対応する `kernel` 関数名。
    fn function_name(self) -> &'static str {
        match self {
            GemmVariant::Naive => "gemm_naive",
            GemmVariant::Tiled => "gemm_tiled",
            GemmVariant::Simdgroup => "gemm_simdgroup",
            GemmVariant::SimdgroupTiled(_) => "gemm_simdgroup_tiled",
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
    /// `gemm_simdgroup_f16`（TASK-8.3b・#156）のパイプライン。
    /// [`Self::dispatch_f16_unverified`] からのみ参照する（`GemmVariant` には含めない。
    /// f16 経路は `dispatch_variant` の既存分岐に統合しない設計判断。
    /// `crate::lib` クレートコメント参照）。
    pipeline_simdgroup_f16: objc2::rc::Retained<MtlPipeline>,
    /// `gemm_tiled_bias_act`（イシュー #605）のパイプライン。GEMM epilogue
    /// （bias 加算・activation）を融合した tiled GEMM。`ops.rs::
    /// MetalBackendOps::gemm_bias_act` から
    /// [`Self::run_tiled_bias_act_f32`] 経由でのみ参照する（`GemmVariant`
    /// には含めない。CUDA 側 `tiled_bias_act_f32` フィールドと同じ設計
    /// 判断: `#include` を使わず全 GPU family で成立するため `Option` 化
    /// しない）。
    pipeline_tiled_bias_act: objc2::rc::Retained<MtlPipeline>,
    /// `gemm_simdgroup_tiled` のコンパイル済みライブラリ（`crate::pipeline::
    /// make_pipeline_with_constants` が構成キーごとにパイプラインを特殊化
    /// する際に再利用する。TASK-1.8f・#188）。
    library: objc2::rc::Retained<MtlLibrary>,
    /// 構成キー（[`TileConfig`]）→ パイプラインの遅延キャッシュ
    /// （[`Self::pipeline_for_tile`]）。候補構成は有限個のため、初回
    /// ディスパッチ時に構築したパイプラインを以降のディスパッチで使い回す
    /// （`MTLFunctionConstantValues` を介したコンパイルは比較的重い処理。
    /// イシュー #188 計画「パイプライン管理」節）。`&self` の非可変参照から
    /// 書き込むため `RefCell` で内部可変性を持たせる（`Retained` は
    /// 参照カウント型でありスレッド境界を跨がない前提。本クレートの
    /// ディスパッチは呼び出しごとに同期完了するため並行アクセスは想定しない）。
    tiled_cache: RefCell<HashMap<TileConfig, objc2::rc::Retained<MtlPipeline>>>,
    /// `gemm_simdgroup_tiled_f16`（イシュー #796）の構成キー
    /// （[`TileConfig`]）→ パイプラインの遅延キャッシュ（[`Self::pipeline_for_tile_f16`]）。
    /// f32 版 [`Self::tiled_cache`] と同じ設計判断（候補構成は有限個のため
    /// 初回ディスパッチ時に構築したパイプラインを使い回す）だが、
    /// `gemm_simdgroup_tiled`（f32）と `gemm_simdgroup_tiled_f16`
    /// は関数名が異なる別パイプラインのため、同一 `TileConfig` キーで
    /// あってもキャッシュを混在させない（f32 版キャッシュとの取り違えは
    /// 誤った関数へのディスパッチに直結するため独立フィールドとして持つ）。
    tiled_f16_cache: RefCell<HashMap<TileConfig, objc2::rc::Retained<MtlPipeline>>>,
    /// threadgroup ID スウィズル（イシュー #540）をこのインスタンスの
    /// `SimdgroupTiled` 経路（[`Self::pipeline_for_tile`]・
    /// `encode_dispatch_tiled`）で有効化するかどうか。イシュー #746 で
    /// クレート定数 `tile::SWIZZLE_ENABLED` の直接参照から instance
    /// フィールドへ格上げした: 同一プロセス内で base（`false`）/head
    /// （`true`）の 2 `MetalGemm` を構築し interleaved に A/B 計測する運用
    /// （`docs/perf/metal-gemm-tgid-swizzle-ab.md`）のため、CUDA 側
    /// `CudaMmaGemm`（`crates/backend-cuda/src/gemm.rs`）と同型の設計に揃えた。
    /// `MetalGemm::new` は本番既定 `tile::SWIZZLE_ENABLED`（`false`）を渡すため
    /// 既定挙動は不変。
    swizzle_enabled: bool,
    /// simdgroup 細粒度同期（イシュー #809）をこのインスタンスの
    /// `SimdgroupTiled`（f32/f16 とも）経路（[`Self::pipeline_for_tile`]・
    /// [`Self::pipeline_for_tile_f16`]）で有効化するかどうか。
    /// `swizzle_enabled` と同じ設計判断（instance フィールド化により
    /// base（`false`）/head（`true`）の 2 `MetalGemm` を同一プロセス内に
    /// 構築して interleaved A/B 計測できるようにする）。`MetalGemm::new` は
    /// 本番既定 `tile::FINE_BARRIER_ENABLED`（`false`）を渡すため既定挙動は
    /// 不変。
    fine_barrier_enabled: bool,
}

impl MetalGemm {
    /// `shaders/gemm.metal` を `ctx` のデバイス上でコンパイルし、
    /// `gemm_naive`/`gemm_tiled`/`gemm_simdgroup` の 3 パイプラインを
    /// 構築する。`gemm_simdgroup_tiled`（TASK-1.8f・#188）はコンパイル済み
    /// ライブラリのみ保持し、構成別パイプラインは [`Self::pipeline_for_tile`]
    /// が初回ディスパッチ時に遅延構築する（候補が有限個でも全構成を
    /// 前もって構築すると起動コストが増えるため）。
    ///
    /// threadgroup ID スウィズル（イシュー #540）は本番既定
    /// `tile::SWIZZLE_ENABLED`（`false`）で [`Self::new_with_swizzle`] へ
    /// 委譲する薄いラッパー（同フィールドドキュメンテーションコメント参照）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        Self::new_with_swizzle(ctx, tile::SWIZZLE_ENABLED)
    }

    /// [`Self::new`] と同じ構築を行うが、simdgroup 細粒度同期
    /// （イシュー #809）の有効・無効を明示的な `fine_barrier_enabled` 引数で
    /// 指定する。ベンチ用途専用の入口: 同一プロセス内で base
    /// （`fine_barrier_enabled=false`）/head（`fine_barrier_enabled=true`）
    /// の 2 インスタンスを構築し、interleaved に A/B 計測する
    /// （`docs/perf/metal-gemm-fine-barrier-ab.md`・
    /// `crates/backend-metal/examples/gemm_fine_barrier_ab_bench.rs`）。
    /// [`Self::new_with_swizzle`] と同型の設計（threadgroup ID スウィズルは
    /// 本番既定 `tile::SWIZZLE_ENABLED` のまま据え置く）。本番経路
    /// （[`Self::new`]）は常に `tile::FINE_BARRIER_ENABLED`（`false`）を渡す
    /// ため、本関数の追加自体は既定挙動を変えない。
    pub fn new_with_fine_barrier(
        ctx: &MetalContext,
        fine_barrier_enabled: bool,
    ) -> Result<Self, MetalError> {
        Self::new_with_swizzle_and_fine_barrier(ctx, tile::SWIZZLE_ENABLED, fine_barrier_enabled)
    }

    /// [`Self::new`] と同じ構築を行うが、threadgroup ID スウィズル
    /// （イシュー #540）の有効・無効を明示的な `swizzle_enabled` 引数で
    /// 指定する（イシュー #746）。ベンチ用途専用の入口: 同一プロセス内で
    /// base（`swizzle_enabled=false`）/head（`swizzle_enabled=true`）の
    /// 2 インスタンスを構築し、interleaved に A/B 計測することで
    /// checkout 切替方式（base/head 計測が時間的に分離されサーマル
    /// ドリフトが系統誤差になる）を回避する狙い
    /// （`docs/perf/metal-gemm-tgid-swizzle-ab.md`・
    /// `crates/backend-metal/examples/gemm_swizzle_ab_bench.rs`）。
    /// CUDA 側 `CudaMmaGemm::new_with_swizzle`
    /// （`crates/backend-cuda/examples/gemm_mma_swizzle_bench.rs`）と同型の
    /// 設計。本番経路（`Self::new`）は常に `tile::SWIZZLE_ENABLED`（`false`）
    /// を渡すため、本関数の追加自体は既定挙動を変えない。
    pub fn new_with_swizzle(ctx: &MetalContext, swizzle_enabled: bool) -> Result<Self, MetalError> {
        Self::new_with_swizzle_and_fine_barrier(ctx, swizzle_enabled, tile::FINE_BARRIER_ENABLED)
    }

    /// [`Self::new_with_swizzle`]・[`Self::new_with_fine_barrier`] が共に
    /// 委譲する共通実装（イシュー #809）。threadgroup ID スウィズル
    /// （イシュー #540）と simdgroup 細粒度同期（イシュー #809）はいずれも
    /// `crate::pipeline::make_pipeline_with_constants` の function constant
    /// 特殊化（index 7／8）であり、A/B 計測の対象軸が異なるだけで構築手順
    /// 自体は独立のため、両フラグを引数に持つ 1 実装へ集約する（各専用入口
    /// が個別に構築ロジックを複製すると、パイプライン構築手順の変更時に
    /// 複数箇所を同期させる必要が生じるため）。
    fn new_with_swizzle_and_fine_barrier(
        ctx: &MetalContext,
        swizzle_enabled: bool,
        fine_barrier_enabled: bool,
    ) -> Result<Self, MetalError> {
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
        // TASK-8.3b（#156）: `gemm_simdgroup_f16` は `GemmVariant` に含めず
        // 独立したパイプラインとして保持する（`Self::dispatch_f16_unverified` からのみ
        // 参照。上記フィールドコメント参照）。
        let pipeline_simdgroup_f16 =
            pipeline::make_pipeline(ctx.device(), &library, "gemm_simdgroup_f16")?;
        let pipeline_tiled_bias_act =
            pipeline::make_pipeline(ctx.device(), &library, "gemm_tiled_bias_act")?;
        Ok(Self {
            pipeline_naive,
            pipeline_tiled,
            pipeline_simdgroup,
            pipeline_simdgroup_f16,
            pipeline_tiled_bias_act,
            library,
            tiled_cache: RefCell::new(HashMap::new()),
            tiled_f16_cache: RefCell::new(HashMap::new()),
            swizzle_enabled,
            fine_barrier_enabled,
        })
    }

    /// `Naive`/`Tiled`/`Simdgroup`（構成を持たない固定パイプライン）用。
    /// `SimdgroupTiled` は構成別に [`Self::pipeline_for_tile`] を使う
    /// （呼び出し元の [`Self::dispatch_variant`] が variant で分岐するため、
    /// 本関数へ `SimdgroupTiled` が渡ることはない）。
    fn pipeline_for(&self, variant: GemmVariant) -> &MtlPipeline {
        match variant {
            GemmVariant::Naive => &self.pipeline_naive,
            GemmVariant::Tiled => &self.pipeline_tiled,
            GemmVariant::Simdgroup => &self.pipeline_simdgroup,
            GemmVariant::SimdgroupTiled(_) => unreachable!(
                "SimdgroupTiled は dispatch_variant が pipeline_for_tile へ振り分ける（呼び出し元の内部不変条件）"
            ),
        }
    }

    /// [`TileConfig`] 候補に対するパイプラインをキャッシュから取得、
    /// 無ければ構築してキャッシュする（TASK-1.8f・#188）。
    ///
    /// `cfg` 自体、または `MTLFunctionConstantValues` によるコンパイル・
    /// パイプライン構築が失敗した場合（デバイス上限超過等）は
    /// `crate::tile::fallback_chain` の次候補（最終的には常に妥当な
    /// [`TileConfig::SINGLE_SIMDGROUP_8X8`]）へ fail-closed でフォール
    /// バックする（計画「パイプライン管理」節。ガードレール閾値の変更
    /// ではなく構成選択のフォールバックであり `.claude/rules/security.md`
    /// の対象外）。返り値は実際に使用する構成（フォールバック後）を含み、
    /// [`Self::dispatch_variant`] がエンコード時の grid・threadgroup 計算に
    /// 使う。
    fn pipeline_for_tile(
        &self,
        ctx: &MetalContext,
        cfg: TileConfig,
    ) -> Result<(Retained<MtlPipeline>, TileConfig), MetalError> {
        let mut last_err: Option<MetalError> = None;

        for candidate in tile::fallback_chain(cfg) {
            if let Some(pipeline) = self.tiled_cache.borrow().get(&candidate) {
                return Ok((Retained::clone(pipeline), candidate));
            }

            // デバイスの threadgroup メモリ上限（`maxThreadgroupMemoryLength`）
            // で事前検証する。スレッド数上限（`maxTotalThreadsPerThreadgroup`）
            // は `MTLComputePipelineState` 構築後にしか取得できないため、
            // 事前検証は Apple GPU の一般的な上限（1024）を仮定し、構築後に
            // パイプライン実測値で再検証する（下記）。
            let max_shared_mem_bytes = ctx.device().maxThreadgroupMemoryLength() as u32;
            if candidate.validate(1024, max_shared_mem_bytes).is_err() {
                continue;
            }

            match pipeline::make_pipeline_with_constants(
                ctx.device(),
                &self.library,
                GemmVariant::SimdgroupTiled(candidate).function_name(),
                candidate,
                self.swizzle_enabled,
                self.fine_barrier_enabled,
            ) {
                Ok(pipeline) => {
                    let actual_max_threads = pipeline.maxTotalThreadsPerThreadgroup() as u32;
                    if candidate.thread_count() > actual_max_threads {
                        continue;
                    }
                    self.tiled_cache
                        .borrow_mut()
                        .insert(candidate, Retained::clone(&pipeline));
                    return Ok((pipeline, candidate));
                }
                Err(err) => {
                    last_err = Some(err);
                    continue;
                }
            }
        }

        Err(last_err.unwrap_or(MetalError::PipelineCreation {
            message: "no tile configuration in fallback chain was accepted".to_string(),
        }))
    }

    /// [`TileConfig`] 候補に対する `gemm_simdgroup_tiled_f16`
    /// （イシュー #796）パイプラインをキャッシュから取得、無ければ構築して
    /// キャッシュする。[`Self::pipeline_for_tile`]（f32 版）と同じ
    /// フォールバック戦略（`crate::tile::fallback_chain`。デバイス上限超過
    /// 等で次候補・最終的には常に妥当な [`TileConfig::SINGLE_SIMDGROUP_8X8`]
    /// へ fail-closed でフォールバック）を踏襲するが、以下 2 点が異なる:
    ///
    /// 1. `self.tiled_f16_cache`（f32 版とは独立したキャッシュ。構造体
    ///    フィールドのドキュメンテーションコメント参照）を使う。
    /// 2. デバイス上限の事前検証を `candidate.validate(1024,
    ///    max_shared_mem_bytes)`（f32 単位の `shared_mem_bytes` を見る）に
    ///    加えて `candidate.shared_mem_bytes_f16() <= max_shared_mem_bytes`
    ///    でも行う。f16 版はエピローグ staging 領域（f32。`bm*bn*4`
    ///    バイト。イシュー #797 でタイル粒度へ拡大）を常に追加確保するため、
    ///    `staged=false`（f32 版
    ///    `shared_mem_bytes()` は 0 を返し `validate` を素通りする）構成
    ///    でも f16 版は非 0 バイトを要求する（`TileConfig::
    ///    shared_mem_bytes_f16` ドキュメントコメント参照）。この追加検査を
    ///    怠ると、f32 版 `validate` だけを通過した構成が実際には f16 版の
    ///    デバイス上限を超過したまま `setThreadgroupMemoryLength_atIndex`
    ///    （`encode_dispatch_tiled_f16`）へ渡ってしまう。
    fn pipeline_for_tile_f16(
        &self,
        ctx: &MetalContext,
        cfg: TileConfig,
    ) -> Result<(Retained<MtlPipeline>, TileConfig), MetalError> {
        let mut last_err: Option<MetalError> = None;

        for candidate in tile::fallback_chain(cfg) {
            if let Some(pipeline) = self.tiled_f16_cache.borrow().get(&candidate) {
                return Ok((Retained::clone(pipeline), candidate));
            }

            let max_shared_mem_bytes = ctx.device().maxThreadgroupMemoryLength() as u32;
            if candidate.validate(1024, max_shared_mem_bytes).is_err() {
                continue;
            }
            // f16 版専用の追加検査（上記ドキュメントコメント 2 点目）。
            if candidate.shared_mem_bytes_f16() > max_shared_mem_bytes {
                continue;
            }

            match pipeline::make_pipeline_with_constants(
                ctx.device(),
                &self.library,
                "gemm_simdgroup_tiled_f16",
                candidate,
                self.swizzle_enabled,
                self.fine_barrier_enabled,
            ) {
                Ok(pipeline) => {
                    let actual_max_threads = pipeline.maxTotalThreadsPerThreadgroup() as u32;
                    if candidate.thread_count() > actual_max_threads {
                        continue;
                    }
                    self.tiled_f16_cache
                        .borrow_mut()
                        .insert(candidate, Retained::clone(&pipeline));
                    return Ok((pipeline, candidate));
                }
                Err(err) => {
                    last_err = Some(err);
                    continue;
                }
            }
        }

        Err(last_err.unwrap_or(MetalError::PipelineCreation {
            message: "no tile configuration in fallback chain was accepted (f16)".to_string(),
        }))
    }

    /// [`Self::pipeline_for_tile`] が実際に採用した [`TileConfig`]（フォール
    /// バック後の構成）を検証する（イシュー #532・PR #651 codex-review 指摘
    /// 対応。P2/P3）。
    ///
    /// `dispatch_variant`（`SimdgroupTiled`）は `pipeline_for_tile` が
    /// `crate::tile::fallback_chain` で構成失敗時に
    /// `TileConfig::SINGLE_SIMDGROUP_8X8` へサイレントにフォールバックして
    /// も戻り値の `Vec<f32>` だけを見ると成功にしか見えず、指定した構成が
    /// 実際にコンパイル・パイプライン構築（実デバイスの
    /// `maxThreadgroupMemoryLength`・パイプライン構築後実測の
    /// `maxTotalThreadsPerThreadgroup` を含む）まで通ったかを外側から検証
    /// できない問題があった。`crate::tile` モジュール末尾の実機依存テスト、
    /// および `tests/gemm_dynamic_tile_parity.rs`（別コンパイル単位の統合
    /// テスト）が本メソッドで `resolve_tile_config(cfg) == cfg` を確認した
    /// うえで初めて実際のディスパッチへ進む契約にする。
    ///
    /// `pub(crate)`（PR #651 codex-review 再指摘・P1）: 当初は統合テスト
    /// （`tests/` 配下・クレート境界の外）から参照するため `#[doc(hidden)]
    /// pub` としていたが、`doc(hidden)` はドキュメント表示を抑えるだけで
    /// 可視性・semver 契約は変更されず、パイプライン構築・フォールバック
    /// というバックエンド内部実装が外部から呼び出し可能な公開 API に
    /// なってしまう問題があった。実セット（`crate::tile::CANDIDATES`）を
    /// 検証するテストはクレート内テスト（本ファイル末尾ではなく
    /// `crate::tile` の `#[cfg(test)] mod tests`。同モジュールは
    /// クレート境界の内側のため `pub(crate)` で届く）へ集約し、統合テスト
    /// （`tests/gemm_dynamic_tile_parity.rs`）側は本メソッドを呼ばず
    /// `dispatch_variant` の数値一致確認に限定した（フォールバック検知は
    /// クレート内テストが担う）。
    ///
    /// 呼び出し元は `crate::tile` の `#[cfg(test)] mod tests`（実機依存・
    /// `#[ignore]`）のみで、本番ディスパッチ経路（[`Self::dispatch_auto`]・
    /// [`Self::dispatch_variant`]）からは呼ばれない。よってテストを含まない
    /// 通常の `lib` ターゲットビルドでは到達不能になる（dead_code 判定は
    /// ターゲットごとのため）ため `#[cfg(test)]` を付ける。
    #[cfg(test)]
    pub(crate) fn resolve_tile_config(
        &self,
        ctx: &MetalContext,
        cfg: TileConfig,
    ) -> Result<TileConfig, MetalError> {
        self.pipeline_for_tile(ctx, cfg)
            .map(|(_, resolved)| resolved)
    }

    /// [`Self::resolve_tile_config`] の f16 版（イシュー #796）。
    /// [`Self::pipeline_for_tile_f16`] が実際に採用した [`TileConfig`] を
    /// 検証する。`crate::tile` の `#[cfg(test)] mod tests`（クレート境界の
    /// 内側のため `pub(crate)` で届く）が `CANDIDATES` を巡回して
    /// `gemm_simdgroup_tiled_f16` のフォールバック（デバイス上限超過等での
    /// サイレントな `TileConfig::SINGLE_SIMDGROUP_8X8` 縮退）を検知する
    /// ために使う（`resolve_tile_config` と同じ判断根拠）。
    #[cfg(test)]
    pub(crate) fn resolve_tile_config_f16(
        &self,
        ctx: &MetalContext,
        cfg: TileConfig,
    ) -> Result<TileConfig, MetalError> {
        self.pipeline_for_tile_f16(ctx, cfg)
            .map(|(_, resolved)| resolved)
    }

    /// 動的タイル選択（TASK-1.8f・#188）の自動入口。`(m, n, k)` から
    /// [`tile::select`] で [`TileConfig`] を選び、[`GemmVariant::SimdgroupTiled`]
    /// で [`Self::dispatch_variant`] へ委譲する。バックエンド抽象層からの
    /// accelerated/tiled 経路選択（#67/#68）とはレイヤが異なる（本関数は
    /// 「Metal GEMM を実行すると決まった後」のタイル構成選択のみを担う。
    /// イシュー #188 計画「スコープ外」節）。
    ///
    /// **occupancy 判定（イシュー #542・[`tile::select_with_occupancy`]）は
    /// 不採用確定（イシュー #747）**: `ctx.occupancy_params()` は実機値
    /// （GPU コア数・threadgroup memory 上限）からキャッシュされるが、
    /// #744 是正（段 1 の正方立方形状判定是正）により実測帯域
    /// 〈512/1024/2048/4096〉では occupancy 縮退の適用対象が実質消滅し
    /// `select()` と常に同一結果になることを確認したため、本番ディスパッチ
    /// へは組み込まない（`docs/perf/metal-gemm-occupancy-select.md`
    /// 「#747 判断」節・`crate::tile::select_with_occupancy` ドキュメンテー
    /// ションコメント参照）。
    pub fn dispatch_auto(
        &self,
        ctx: &MetalContext,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let cfg = tile::select(m, n, k);
        self.dispatch_variant(ctx, GemmVariant::SimdgroupTiled(cfg), a, b, m, n, k)
    }

    /// f16 動的タイル選択（イシュー #798）の自動入口。[`Self::dispatch_auto`]
    /// （f32 版）と同じ「Metal GEMM を実行すると決まった後のタイル構成選択」
    /// 責務を f16 でも提供する。`(m, n, k)` から [`tile::select`] で
    /// [`TileConfig`] を選び、[`Self::dispatch_f16_tiled_unverified`]
    /// （`gemm_simdgroup_tiled_f16`。イシュー #796）へ委譲する。
    ///
    /// `GemmVariant` enum への f16 統合は行わない（`dispatch_variant` が
    /// `&[f32]` 型に閉じている既存設計判断は維持。[`Self::pipeline_simdgroup_f16`]
    /// フィールドコメント参照）。動的タイル選択機構そのもの（[`tile::select`]・
    /// fallback chain・パイプラインキャッシュ）は f32 と完全共有するため、
    /// f16 専用の選択ロジックを別途持たない。
    ///
    /// **後方互換方針（イシュー #798 受け入れ条件 3）**: `gemm_simdgroup_f16`
    /// （非タイル 8x8 カーネル）・[`Self::dispatch_f16_unverified`] 系の明示
    /// 入口は削除・置換せず維持する。理由は (a)
    /// `crates/backend-metal/examples/gemm_f16_bench.rs`（REQ-8 実測境界）・
    /// `tests/cpu_metal_f16_parity.rs` が参照する計測・回帰基線であること、
    /// (b) 新旧カーネルの相互照合（`tests/cpu_metal_f16_tiled_parity.rs`）の
    /// 対向として必要なこと。本関数（自動経路）は `gemm_simdgroup_tiled_f16`
    /// のみを使い、微小形状・フォールバック時も同カーネルの
    /// `TileConfig::SINGLE_SIMDGROUP_8X8` 構成（旧カーネルと同一の
    /// 1 threadgroup = 1 simdgroup = 8x8 構造）で賄う——すなわち旧カーネル
    /// 自体は自動経路の縮退先ではなく、明示入口専用の計測・回帰基線として
    /// 存置する。
    ///
    /// # 精度検証状況・`_unverified` suffix（PR #819 codex-review P1 指摘対応）
    ///
    /// REQ-2 統一複合判定（相対誤差 1e-3 未満または絶対誤差 1e-5 未満）の
    /// 検証は `tests/gemm_f16_auto_parity.rs`（Metal 実機依存・`#[ignore]`）
    /// の契約だが、本 PR 時点では実機（M4 Max 等）での実行が未完了であり、
    /// 実機実行は #799 実機セッションで実施する（#796 の
    /// `tests/cpu_metal_f16_tiled_parity.rs` 引き継ぎと同一様式）。すなわち
    /// 本関数が委譲する `gemm_simdgroup_tiled_f16`（[`Self::dispatch_f16_tiled_unverified`]。
    /// `_unverified` suffix・`#[doc(hidden)]`）自体が精度未検証カーネルであり、
    /// 未検証カーネルを検証済み production 入口へ直接結線しない既存の安全境界
    /// （[`Self::dispatch_f16_unverified`] ドキュメントコメント参照）に従い、
    /// 本関数自身も `_unverified` suffix・`#[doc(hidden)]` とする。#799 の
    /// 実機検証結果をこの経路へ反映し `tests/gemm_f16_auto_parity.rs` が
    /// green になった時点で、suffix・`#[doc(hidden)]` の解除を別イシューで
    /// 検討する（[`Self::dispatch_f16_unverified`] と同型の解除条件）。
    #[doc(hidden)]
    pub fn dispatch_f16_auto_unverified(
        &self,
        ctx: &MetalContext,
        a: &[half::f16],
        b: &[half::f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<half::f16>, MetalError> {
        let cfg = tile::select(m, n, k);
        self.dispatch_f16_tiled_unverified(ctx, a, b, m, n, k, cfg)
    }

    /// バックエンド抽象層からの GEMM 自動経路選択入口（TASK-11.2b・#68）。
    ///
    /// `ctx.caps()`（`MetalContext::new` 時にキャッシュした
    /// `MTLDevice::supportsFamily(MTLGPUFamily::Apple7)` 判定）と `(m, n, k)`
    /// から `tensor_core::dispatch::select_gemm_kernel` を呼び、その結果
    /// （[`tensor_core::dispatch::KernelKind`]）を [`GemmVariant`] へ写像
    /// する（`docs/dispatch-rules-design.md` §5.3 決定表の Metal 側行）:
    ///
    /// - `MatrixUnit` → [`Self::dispatch_auto`]（[`tile::select`] による
    ///   動的タイル選択。occupancy 判定〈イシュー #542〉はイシュー #747 で
    ///   不採用確定のため未適用〈[`Self::dispatch_auto`] ドキュメンテー
    ///   ションコメント参照〉。「Metal GEMM を実行すると決まった後のタイル構成選択」と
    ///   いう別レイヤの責務は [`Self::dispatch_auto`] のドキュメンテー
    ///   ションコメントどおり変更しない。実装計画 §3.3）
    /// - `Tiled` → [`GemmVariant::Tiled`]
    /// - `Naive` → [`GemmVariant::Naive`]（現決定表では Metal 行から到達
    ///   しないが、`select_gemm_kernel` の将来変更に対する fail-safe と
    ///   して網羅する）
    ///
    /// `dtype` は現時点で `f32` 固定（`BackendOps` v1 が f32 専用のため。
    /// `docs/public-api-design.md:469`、設計文書 §5.3 注記）。`m`／`n`／`k`
    /// は `u32` へ飽和変換して形状判定に使う（`u32::MAX` を超える巨大な
    /// 形状は後段 [`Self::dispatch_variant`] の `validate_dims` が
    /// `DimensionExceedsU32` として確実に拒否するため、選択段階での
    /// 変換は判定用途に限定され安全性に影響しない）。
    pub fn dispatch_backend_auto(
        &self,
        ctx: &MetalContext,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let shape = tensor_core::dispatch::GemmShape::new(
            u32::try_from(m).unwrap_or(u32::MAX),
            u32::try_from(n).unwrap_or(u32::MAX),
            u32::try_from(k).unwrap_or(u32::MAX),
        );
        let kernel = tensor_core::dispatch::select_gemm_kernel(
            &ctx.caps(),
            shape,
            tensor_core::dispatch::DType::F32,
        );
        match kernel {
            tensor_core::dispatch::KernelKind::MatrixUnit => self.dispatch_auto(ctx, a, b, m, n, k),
            tensor_core::dispatch::KernelKind::Tiled => {
                self.dispatch_variant(ctx, GemmVariant::Tiled, a, b, m, n, k)
            }
            tensor_core::dispatch::KernelKind::Naive => {
                self.dispatch_variant(ctx, GemmVariant::Naive, a, b, m, n, k)
            }
        }
    }

    /// `SimdgroupTiled`（f32）の §4 準拠 prepared 入口（イシュー #572・
    /// Phase F-2）。[`Self::dispatch_f16_prepared_unverified`] と同型の
    /// 計測境界（エンコード＋コマンドバッファ完了待ちのみ）を f32 側にも
    /// 提供する目的で追加した（`docs/perf/gemm-optimization-baseline.md`
    /// §2 が「f16 と同型の §4 準拠 prepared ディスパッチ入口を f32 側にも
    /// 用意したうえでの f32 再計測は Phase F の #572 のスコープ」と明記した
    /// 対応）。`crates/backend-metal/examples/gemm_f32_prepared_bench.rs` が
    /// `scripts/bench/gemm_bench_torch_mps_f32.py`（`torch.mm` +
    /// `torch.mps.synchronize()` のみ計測）と同一の同期境界で計測するために
    /// 本関数を使う。
    ///
    /// [`Self::dispatch_variant`]（`SimdgroupTiled`）はパディング・バッファ
    /// 確保／アップロード・ディスパッチ・readback／アンパディングを一括で
    /// 行うのに対し、本関数は呼び出し元が事前に [`pad8`] で実効次元へ
    /// パディング・確保・アップロード済みの [`MetalBuffer`] に対して
    /// [`MetalContext::dispatch_sync`] のクロージャ結線のみを行う。呼び出し
    /// 元が `pad8` を経由せず任意の `m_eff`/`n_eff`/`k_eff`・バッファ長を
    /// 渡せるため、[`validate_prepared_inputs_f32`] で 8 の倍数・バッファ長
    /// 一致をエンコード前に検証する（f16 側 `validate_prepared_inputs`・
    /// PR #346 codex-review P1-1 指摘と同水準の検証）。
    ///
    /// `cfg` は呼び出し元が選んだ候補構成（`tile::select(m, n, k)` 等）だが、
    /// [`Self::pipeline_for_tile`] がデバイス上限超過等でサイレントに
    /// `TileConfig::SINGLE_SIMDGROUP_8X8` へフォールバックしうるため
    /// （フォールバック透明性は [`Self::pipeline_for_tile`] ドキュメント
    /// コメント参照）、戻り値として実際に採用された構成（resolved）を返す。
    /// 呼び出し元（ベンチ入口）は戻り値のラベルで実測対象構成を確定できる。
    ///
    /// `SimdgroupTiled` は `pad8` パディング契約が必須（[`Self::dispatch_variant`]
    /// 参照）。実効次元は呼び出し元が [`pad8`] で 8 の倍数へ揃えたうえで
    /// [`pad_matrix`] 済みのデータを [`MetalBuffer`] へアップロードしておく
    /// こと（パディング・バッファ確保／アップロードは計測ループ外で行う
    /// 想定。呼び出し元コメント参照）。
    ///
    /// `#[allow(clippy::too_many_arguments)]`: [`Self::dispatch_f16_prepared_unverified`]
    /// と同じ判断根拠（個別引数で呼び出し側の意図が明確になるため構造体へ
    /// まとめ込まない。理由コメント必須のルール `.claude/rules/coding-rust.md`
    /// に対応）。
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_tiled_prepared(
        &self,
        ctx: &MetalContext,
        a_buf: &MetalBuffer,
        b_buf: &MetalBuffer,
        c_buf: &MetalBuffer,
        m_eff: usize,
        n_eff: usize,
        k_eff: usize,
        cfg: TileConfig,
    ) -> Result<TileConfig, MetalError> {
        let dims = validate_effective_dims(m_eff, n_eff, k_eff)?;
        validate_prepared_inputs_f32(a_buf, b_buf, c_buf, m_eff, n_eff, k_eff)?;

        let (pipeline, resolved_cfg) = self.pipeline_for_tile(ctx, cfg)?;
        ctx.dispatch_sync(|encoder| {
            encode_dispatch_tiled(
                encoder,
                &pipeline,
                a_buf,
                b_buf,
                c_buf,
                dims,
                resolved_cfg,
                self.swizzle_enabled,
            );
        })?;

        Ok(resolved_cfg)
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

    /// GEMM epilogue（bias 加算・activation）を融合した tiled GEMM（f32。
    /// イシュー #605）を実行し、結果をホストへ読み出す。
    /// `ops.rs::MetalBackendOps::gemm_bias_act` の融合経路
    /// （[`crate::ops::gemm_bias_act_route`]）からのみ呼ばれる。
    ///
    /// `bias` が `Some` の場合は `bias.len() == n` を fail-closed に検証
    /// する（`shaders/gemm.metal::gemm_tiled_bias_act` の `bias[col]`
    /// epilogue が `n` 要素を前提とするため）。
    ///
    /// `m == 0 || n == 0` は no-op（空の結果）、`k == 0` は CPU 参照実装
    /// （`backend_cpu::gemm_blis::gemm_blis_bias_act_parallel`）・CUDA 側
    /// `run_tiled_bias_act_f32` と同じ契約で epilogue のみホスト側で計算し
    /// GPU 起動を回避する（[`BIAS_ACT_FUSED_LAUNCH_COUNT`] は増加させない。
    /// `validate_dims`〈`m/n/k == 0` を一律 `ZeroDimension` として拒否〉を
    /// 経由せず、本関数専用の [`validate_bias_act_dims`] で検証してから
    /// この縮退分岐を判定する）。
    ///
    /// `bias` が `None` の場合は `n` 要素のゼロ初期化バッファを渡す
    /// （1 要素ダミーではない。`shaders/gemm.metal::gemm_tiled_bias_act`
    /// 冒頭コメント「`bias` が `None`」参照: select 化の可能性がある
    /// Metal コンパイラの最適化戦略に依存しない fail-closed な対策）。
    #[allow(clippy::too_many_arguments)]
    pub fn run_tiled_bias_act_f32(
        &self,
        ctx: &MetalContext,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        act_relu: bool,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, MetalError> {
        validate_bias_act_dims(a.len(), b.len(), m, n, k)?;
        if let Some(bias) = bias
            && bias.len() != n
        {
            return Err(MetalError::InvalidElementwiseShape {
                detail: format!(
                    "bias length mismatch: expected {n} (n), actual {}",
                    bias.len()
                ),
            });
        }

        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }

        if k == 0 {
            // 上記ドキュメンテーションコメント「k == 0」参照: CPU 参照
            // 実装・CUDA 側と同じく epilogue はホスト側で直接適用し、GPU
            // 起動は行わない（BIAS_ACT_FUSED_LAUNCH_COUNT は増加させない）。
            let mut out = vec![0.0f32; m * n];
            if let Some(bias) = bias {
                for row in out.chunks_mut(n) {
                    for (x, bv) in row.iter_mut().zip(bias.iter()) {
                        *x += *bv;
                    }
                }
            }
            if act_relu {
                for x in out.iter_mut() {
                    *x = x.max(0.0);
                }
            }
            return Ok(out);
        }

        let a_buf = MetalBuffer::new_with_data(ctx, a)?;
        let b_buf = MetalBuffer::new_with_data(ctx, b)?;
        // `bias` が `None` の場合は `n` 要素のゼロ初期化バッファを渡す
        // （`shaders/gemm.metal::gemm_tiled_bias_act` 冒頭コメント参照）。
        let (bias_buf, has_bias): (MetalBuffer, i32) = match bias {
            Some(bias) => (MetalBuffer::new_with_data(ctx, bias)?, 1),
            None => (MetalBuffer::new_zeroed(ctx, n)?, 0),
        };
        let c_buf = MetalBuffer::new_zeroed(ctx, m * n)?;

        let dims = Dims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
        };
        let act_i: i32 = if act_relu { 1 } else { 0 };

        ctx.dispatch_sync(|encoder| {
            encode_dispatch_bias_act(
                encoder,
                &self.pipeline_tiled_bias_act,
                &a_buf,
                &b_buf,
                &bias_buf,
                &c_buf,
                dims,
                has_bias,
                act_i,
            );
        })?;

        // バッファ確保（`MetalBuffer::new_with_data`／`new_zeroed`）・
        // `ctx.dispatch_sync` がすべて成功した後にのみ増加させる（codex-review
        // 指摘・PR #717。確保／dispatch 失敗時に「起動済み」として誤記録
        // すると、経路検証テスト・診断が偽陽性になるため）。
        BIAS_ACT_FUSED_LAUNCH_COUNT.with(|c| c.set(c.get() + 1));

        Ok(c_buf.read_to_vec())
    }

    /// f16 GEMM（`C = A @ B`。TASK-8.3b・#156）を実行し、結果をホストへ
    /// 読み出す。`gemm_simdgroup_f16`（`shaders/gemm.metal`）のみを対象と
    /// する明示ディスパッチ入口であり、[`Self::dispatch_auto`]／
    /// `dispatch_backend_auto`（f32 専用の自動経路選択）とは独立している
    /// （`crate::lib` クレートコメント「f16 の自動ディスパッチ統合は
    /// 本 TASK のスコープ外」）。
    ///
    /// # 精度検証状況（イシュー #380 で実機検証済み。`_unverified` suffix は維持）
    ///
    /// `gemm_simdgroup_f16` は A・B に `simdgroup_half8x8`、アキュムレータに
    /// `simdgroup_float8x8`（f32 累算）を使う（`shaders/gemm.metal::
    /// gemm_simdgroup_f16` 冒頭コメント「累算精度契約」参照。実装計画時点の
    /// half 統一判断はイシュー #380 の実機 spike で `simdgroup_float8x8`
    /// アキュムレータが実在すると判明し変更済み）。CUDA 側 WMMA f16
    /// （`f32.f16.f16.f32`。f32 累算）と精度契約が整合しており、Apple
    /// Silicon 実機（M4 Max・macOS 26.6）での `cpu_metal_f16_parity.rs`
    /// 6 件が REQ-2 複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）
    /// で green であることを実測確認済み（`docs/backend-metal-real-device-
    /// testing.md`）。`_unverified` suffix・`#[doc(hidden)]` は当面維持する
    /// （本関数は `dispatch_auto`／`dispatch_backend_auto`（production 経路）
    /// へまだ統合されていないため。統合可否・suffix 見直しは別イシューの
    /// スコープとする）。
    ///
    /// `Simdgroup`（f32 版）と同じく [`pad8`] で実効次元（8 の倍数）を
    /// 算出し、[`pad_matrix_f16`] で A・B を 0 パディングしてから
    /// [`MetalHalfBuffer`] を確保・ディスパッチする。readback 後は
    /// [`unpad_matrix_f16`] で元の m×n 形状へ切り出す。
    #[doc(hidden)]
    pub fn dispatch_f16_unverified(
        &self,
        ctx: &MetalContext,
        a: &[half::f16],
        b: &[half::f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<half::f16>, MetalError> {
        validate_dims_f16(a, b, m, n, k)?;

        let (m_eff, n_eff, k_eff) = (pad8(m), pad8(n), pad8(k));
        // `pad_matrix_f16`／`m_eff * n_eff`（バッファ確保サイズ算出）より
        // 前に実効次元のオーバーフロー／`u32` 範囲検証を行う（f32 側の
        // `dispatch_variant` と同じ順序。PR #346 Bugbot 指摘: 本チェックを
        // `dispatch_f16_prepared_unverified` 内のみに委ねると、大きな
        // padding 後の shape に対してこのパディング処理・確保サイズ計算が
        // 検証前に実行されてしまう）。戻り値の `Dims` は
        // `dispatch_f16_prepared_unverified` 側で改めて算出するため破棄する。
        validate_effective_dims(m_eff, n_eff, k_eff)?;

        let a_padded = pad_matrix_f16(a, m, k, m_eff, k_eff);
        let b_padded = pad_matrix_f16(b, k, n, k_eff, n_eff);

        let a_buf = MetalHalfBuffer::new_with_data(ctx, &a_padded)?;
        let b_buf = MetalHalfBuffer::new_with_data(ctx, &b_padded)?;
        let c_buf = MetalHalfBuffer::new_zeroed(ctx, m_eff * n_eff)?;

        self.dispatch_f16_prepared_unverified(ctx, &a_buf, &b_buf, &c_buf, m_eff, n_eff, k_eff)?;

        let padded_c = c_buf.read_to_vec();
        Ok(unpad_matrix_f16(padded_c, m_eff, n_eff, m, n))
    }

    /// `gemm_simdgroup_f16` のディスパッチ（エンコード＋コマンドバッファ
    /// 完了待ち）のみを行う入口（TASK-8.3b・#156 ベンチ計測境界修正。
    /// PR #346 Bugbot 指摘 2）。
    ///
    /// 精度未検証である理由・`_unverified` suffix・`#[doc(hidden)]` の
    /// 判断根拠は [`Self::dispatch_f16_unverified`] のドキュメント
    /// コメント（PR #346 codex-review P1-2 指摘）を参照。
    ///
    /// [`Self::dispatch_f16_unverified`] はパディング・バッファ確保／
    /// アップロード・ディスパッチ・readback／アンパディングを一括で行うのに
    /// 対し、本関数は呼び出し元が事前に確保・アップロード済みの
    /// `MetalHalfBuffer`（実効次元 `m_eff`/`n_eff`/`k_eff` でパディング
    /// 済みである前提。[`pad8`]／[`crate::pad::pad_matrix_f16`] 参照）に
    /// 対して [`MetalContext::dispatch_sync`] のクロージャ結線のみを行う。
    /// 呼び出し元が `pad8` を経由せず任意の `m_eff`/`n_eff`/`k_eff`・
    /// バッファ長を渡せるため、[`validate_prepared_inputs`] で 8 の倍数・
    /// バッファ長一致をエンコード前に検証する（PR #346 codex-review
    /// P1-1 指摘）。
    ///
    /// `crates/backend-metal/examples/gemm_f16_bench.rs` が
    /// `scripts/bench/gemm_bench_torch_mps_f16.py`（オンデバイス `matmul`＋
    /// `torch.mps.synchronize()` のみを計測）と同一の同期境界で計測する
    /// ために本関数を使う。バッファ確保・転送・パディング処理は計測外
    /// （ウォームアップ側）で行う想定（呼び出し元コメント参照）。
    ///
    /// `#[allow(clippy::too_many_arguments)]`: `ctx`・`a_buf`・`b_buf`・
    /// `c_buf`・`m_eff`・`n_eff`・`k_eff` を個別引数として持つことで
    /// 呼び出し側の意図が明確になるため、構造体へのまとめ込みは行わない
    /// （`dispatch_variant` と同じ判断根拠。理由コメント必須のルール
    /// `.claude/rules/coding-rust.md` に対応）。
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    pub fn dispatch_f16_prepared_unverified(
        &self,
        ctx: &MetalContext,
        a_buf: &MetalHalfBuffer,
        b_buf: &MetalHalfBuffer,
        c_buf: &MetalHalfBuffer,
        m_eff: usize,
        n_eff: usize,
        k_eff: usize,
    ) -> Result<(), MetalError> {
        let dims = validate_effective_dims(m_eff, n_eff, k_eff)?;
        validate_prepared_inputs(a_buf, b_buf, c_buf, m_eff, n_eff, k_eff)?;

        ctx.dispatch_sync(|encoder| {
            encode_dispatch_f16(
                encoder,
                &self.pipeline_simdgroup_f16,
                a_buf,
                b_buf,
                c_buf,
                dims,
            );
        })
    }

    /// `gemm_simdgroup_tiled_f16`（イシュー #796。BM/BN/BK/WM/WN・協調ロード
    /// をスカラーロードで実装した half タイル化カーネル）の §4 準拠 prepared
    /// 入口。[`Self::dispatch_tiled_prepared`]（f32 版）・
    /// [`Self::dispatch_f16_prepared_unverified`]（非タイル f16 版）と同型の
    /// 計測境界（エンコード＋コマンドバッファ完了待ちのみ）を提供する。
    ///
    /// `#[doc(hidden)]`・`_unverified` suffix の判断根拠は
    /// [`Self::dispatch_f16_unverified`] のドキュメントコメントと同一
    /// （REQ-2 複合判定を満たすかは `tests/cpu_metal_f16_tiled_parity.rs`・
    /// `tests/gemm_f16_auto_parity.rs`〈いずれも Metal 実機依存・`#[ignore]`〉
    /// で検証する契約）。動的タイル選択入口
    /// [`Self::dispatch_f16_auto_unverified`]（イシュー #798）が本関数を
    /// 呼ぶ形で結線済みだが、同関数自体も精度未検証（`_unverified`
    /// suffix・`#[doc(hidden)]`。PR #819 codex-review P1 指摘対応）であり、
    /// `ops::MetalBackendOps`・`dispatch_backend_auto` 等の検証済み
    /// production 経路へはまだ統合されていない（本関数自体は
    /// `#[doc(hidden)]` の明示入口として維持）。
    ///
    /// 呼び出し元は事前に [`pad8`] で実効次元へパディング・確保・
    /// アップロード済みの [`MetalHalfBuffer`] を渡す契約（`dispatch_variant`
    /// の `SimdgroupTiled` と同じ `pad8` 契約。[`Self::dispatch_tiled_prepared`]
    /// ドキュメントコメント参照）。[`validate_prepared_inputs`]（f16 版・
    /// 非タイルカーネルと共有）で 8 の倍数・バッファ長一致をエンコード前に
    /// 検証する。
    ///
    /// `cfg` は呼び出し元が選んだ候補構成だが、[`Self::pipeline_for_tile_f16`]
    /// がデバイス上限超過等でサイレントに `TileConfig::SINGLE_SIMDGROUP_8X8`
    /// へフォールバックしうる（f32 版 `pipeline_for_tile` と同じフォール
    /// バック透明性の設計）ため、戻り値として実際に採用された構成
    /// （resolved）を返す。
    ///
    /// `#[allow(clippy::too_many_arguments)]`: [`Self::dispatch_tiled_prepared`]
    /// と同じ判断根拠（個別引数で呼び出し側の意図が明確になるため構造体へ
    /// まとめ込まない）。
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    pub fn dispatch_f16_tiled_prepared_unverified(
        &self,
        ctx: &MetalContext,
        a_buf: &MetalHalfBuffer,
        b_buf: &MetalHalfBuffer,
        c_buf: &MetalHalfBuffer,
        m_eff: usize,
        n_eff: usize,
        k_eff: usize,
        cfg: TileConfig,
    ) -> Result<TileConfig, MetalError> {
        let dims = validate_effective_dims(m_eff, n_eff, k_eff)?;
        validate_prepared_inputs(a_buf, b_buf, c_buf, m_eff, n_eff, k_eff)?;

        let (pipeline, resolved_cfg) = self.pipeline_for_tile_f16(ctx, cfg)?;
        ctx.dispatch_sync(|encoder| {
            encode_dispatch_tiled_f16(
                encoder,
                &pipeline,
                a_buf,
                b_buf,
                c_buf,
                dims,
                resolved_cfg,
                self.swizzle_enabled,
            );
        })?;

        Ok(resolved_cfg)
    }

    /// `gemm_simdgroup_tiled_f16`（イシュー #796）の全部入り入口:
    /// パディング・バッファ確保／アップロード・ディスパッチ・readback／
    /// アンパディングを一括で行う。[`Self::dispatch_f16_unverified`]
    /// （非タイル f16 版）と同じ全部入り構造に、明示 `cfg` を追加した形
    /// （実装計画 §3.2「明示 `TileConfig` 指定の単体ディスパッチ入口」）。
    ///
    /// `#[doc(hidden)]`・`_unverified` suffix の判断根拠は
    /// [`Self::dispatch_f16_unverified`] のドキュメントコメントと同一。
    /// 動的タイル選択入口（[`Self::dispatch_f16_auto_unverified`]）は
    /// イシュー #798 で本関数を `tile::select` が選んだ [`TileConfig`]
    /// 付きで呼ぶ形で結線済み（本関数自体は明示 `cfg` 指定用の入口として
    /// 維持。dtype 汎化は `GemmVariant` へ統合しない設計判断のため
    /// 行わない）。
    ///
    /// `#[allow(clippy::too_many_arguments)]`: [`Self::dispatch_f16_unverified`]
    /// と同じ判断根拠に `cfg` が加わったのみ。
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    pub fn dispatch_f16_tiled_unverified(
        &self,
        ctx: &MetalContext,
        a: &[half::f16],
        b: &[half::f16],
        m: usize,
        n: usize,
        k: usize,
        cfg: TileConfig,
    ) -> Result<Vec<half::f16>, MetalError> {
        validate_dims_f16(a, b, m, n, k)?;

        let (m_eff, n_eff, k_eff) = (pad8(m), pad8(n), pad8(k));
        // `dispatch_f16_unverified` と同じ順序判断（コメント参照）:
        // `pad_matrix_f16`／`m_eff * n_eff`（バッファ確保サイズ算出）より
        // 前に実効次元のオーバーフロー／`u32` 範囲検証を行う。
        validate_effective_dims(m_eff, n_eff, k_eff)?;

        let a_padded = pad_matrix_f16(a, m, k, m_eff, k_eff);
        let b_padded = pad_matrix_f16(b, k, n, k_eff, n_eff);

        let a_buf = MetalHalfBuffer::new_with_data(ctx, &a_padded)?;
        let b_buf = MetalHalfBuffer::new_with_data(ctx, &b_padded)?;
        let c_buf = MetalHalfBuffer::new_zeroed(ctx, m_eff * n_eff)?;

        self.dispatch_f16_tiled_prepared_unverified(
            ctx, &a_buf, &b_buf, &c_buf, m_eff, n_eff, k_eff, cfg,
        )?;

        let padded_c = c_buf.read_to_vec();
        Ok(unpad_matrix_f16(padded_c, m_eff, n_eff, m, n))
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
        // 戻り値の `Dims`（パディング前の実効次元）は使わず検証のためだけに
        // 呼ぶ。実際に後段で使うのはパディング後の `validate_effective_dims`
        // が返す `Dims`（`dims`）であるため意図的に破棄する。
        validate_dims(a, b, m, n, k)?;

        // `SimdgroupTiled` も `Simdgroup` と同じく `simdgroup_load`/`_store`
        // が 8x8 タイル単位でしかアクセスできないため、実効次元を 8 の倍数
        // へパディングする契約は共通（`shaders/gemm.metal` の
        // `gemm_simdgroup_tiled` 直接ロード経路のコメント参照）。
        let (m_eff, n_eff, k_eff) = match variant {
            GemmVariant::Simdgroup | GemmVariant::SimdgroupTiled(_) => (pad8(m), pad8(n), pad8(k)),
            GemmVariant::Naive | GemmVariant::Tiled => (m, n, k),
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

        match variant {
            GemmVariant::SimdgroupTiled(cfg) => {
                let (pipeline, resolved_cfg) = self.pipeline_for_tile(ctx, cfg)?;
                ctx.dispatch_sync(|encoder| {
                    encode_dispatch_tiled(
                        encoder,
                        &pipeline,
                        &a_buf,
                        &b_buf,
                        &c_buf,
                        dims,
                        resolved_cfg,
                        self.swizzle_enabled,
                    );
                })?;
            }
            fixed_variant => {
                let pipeline = self.pipeline_for(fixed_variant);
                ctx.dispatch_sync(|encoder| {
                    encode_dispatch(
                        encoder,
                        pipeline,
                        &a_buf,
                        &b_buf,
                        &c_buf,
                        dims,
                        fixed_variant,
                    );
                })?;
            }
        }

        let padded_c = c_buf.read_to_vec();
        Ok(unpad_matrix(padded_c, m_eff, n_eff, m, n))
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

/// [`MetalGemm::run_tiled_bias_act_f32`] 専用の形状検証（イシュー #605）。
///
/// [`validate_dims`] と異なり `m/n/k == 0` を一律拒否しない
/// （`run_tiled_bias_act_f32` は `m == 0 || n == 0` を no-op・`k == 0` を
/// ホスト側 epilogue のみの縮退経路として受理する契約のため。CUDA 側
/// `gemm.rs::validate_gemm_dims` と同じ「ゼロ次元を許容し、呼び出し元が
/// 縮退分岐を判定する」設計）。長さ一致検証・`checked_mul` オーバーフロー
/// 検出・`u32::MAX` 超過検出（`Dims` への cast 前検証）を行う。
fn validate_bias_act_dims(
    a_len: usize,
    b_len: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Dims, MetalError> {
    let mk = m.checked_mul(k).ok_or(MetalError::DimProductOverflow)?;
    let kn = k.checked_mul(n).ok_or(MetalError::DimProductOverflow)?;
    m.checked_mul(n).ok_or(MetalError::DimProductOverflow)?;

    if m > u32::MAX as usize || n > u32::MAX as usize || k > u32::MAX as usize {
        return Err(MetalError::DimensionExceedsU32 { m, n, k });
    }

    if a_len != mk {
        return Err(MetalError::ALenMismatch {
            expected: mk,
            actual: a_len,
        });
    }
    if b_len != kn {
        return Err(MetalError::BLenMismatch {
            expected: kn,
            actual: b_len,
        });
    }

    Ok(Dims {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    })
}

/// [`validate_dims`] の f16 版（TASK-8.3b・#156）。判定ロジック自体は
/// 要素数（`.len()`）にのみ依存し dtype に非依存のため中身は同一だが、
/// `MetalGemm::dispatch_f16_unverified` の入力型（`&[half::f16]`）に合わせて独立実装
/// する（`validate_dims`（f32 版）と同じ判断根拠: クレートをまたいだ検証
/// ロジック共有はスコープ外という既存方針を型違いの同クレート内複製にも
/// 適用する）。
fn validate_dims_f16(
    a: &[half::f16],
    b: &[half::f16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Dims, MetalError> {
    if m == 0 || n == 0 || k == 0 {
        return Err(MetalError::ZeroDimension { m, n, k });
    }

    let mk = m.checked_mul(k).ok_or(MetalError::DimProductOverflow)?;
    let kn = k.checked_mul(n).ok_or(MetalError::DimProductOverflow)?;
    m.checked_mul(n).ok_or(MetalError::DimProductOverflow)?;

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
/// だけで実質的にオーバーフローしない（`Naive`/`Tiled` は 8 の倍数ではない
/// 任意の m/n/k を許容する必要があるため、8 の倍数検証は本関数には含めない。
/// `Simdgroup`（f32）は [`pad8`] 済みの実効次元を渡すため常に 8 の倍数だが、
/// `gemm_simdgroup_f16_unverified` 専用の [`validate_prepared_inputs`] は
/// `pad8` を経由しない呼び出し元を想定し、8 の倍数検証を別途持つ）。
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

/// [`MetalGemm::dispatch_f16_prepared_unverified`] の入力検証（PR #346
/// codex-review P1-1 指摘）。
///
/// `dispatch_f16_prepared_unverified` は公開入口であり、呼び出し元が
/// [`MetalHalfBuffer::new_with_data`]／[`MetalHalfBuffer::new_zeroed`]
/// （いずれも公開コンストラクタ）で任意長のバッファを渡せ、`m_eff`/
/// `n_eff`/`k_eff` も [`pad8`] を経由せず直接渡せる。[`validate_effective_dims`]
/// は積のオーバーフロー・`u32::MAX` 超過のみを検証するため、本関数は
/// エンコード（`encode_dispatch_f16`）前に以下 2 点を追加検証する:
///
/// 1. `m_eff`/`n_eff`/`k_eff` がいずれも 8 の倍数であること
///    （`gemm_simdgroup_f16_unverified` は 1 threadgroup = C の 8×8 タイル
///    1 つを前提に grid を `n_eff/8 × m_eff/8` で算出する。`encode_dispatch_f16`
///    参照。非 8 倍数では末尾タイルを黙って計算しない）
/// 2. `a_buf.len() == m_eff*k_eff`・`b_buf.len() == k_eff*n_eff`・
///    `c_buf.len() == m_eff*n_eff`（不一致のまま進むと短いバッファでは
///    Metal カーネルが範囲外アクセスしうる。shader 側の手動境界チェック
///    〈REQ-8〉はスレッド数の範囲は守るが、バッファ自体の確保長不足までは
///    検出できない）
///
/// [`validate_effective_dims`] 通過後にのみ呼ばれる前提のため、
/// `m_eff*k_eff` 等の積は `usize` の範囲でオーバーフローしないことが
/// 保証されている（`checked_mul` を再度呼ぶ必要はない）。
fn validate_prepared_inputs(
    a_buf: &MetalHalfBuffer,
    b_buf: &MetalHalfBuffer,
    c_buf: &MetalHalfBuffer,
    m_eff: usize,
    n_eff: usize,
    k_eff: usize,
) -> Result<(), MetalError> {
    if !m_eff.is_multiple_of(8) || !n_eff.is_multiple_of(8) || !k_eff.is_multiple_of(8) {
        return Err(MetalError::NotEightAligned {
            m_eff,
            n_eff,
            k_eff,
        });
    }

    let mk = m_eff * k_eff;
    let kn = k_eff * n_eff;
    let mn = m_eff * n_eff;

    if a_buf.len() != mk {
        return Err(MetalError::ALenMismatch {
            expected: mk,
            actual: a_buf.len(),
        });
    }
    if b_buf.len() != kn {
        return Err(MetalError::BLenMismatch {
            expected: kn,
            actual: b_buf.len(),
        });
    }
    if c_buf.len() != mn {
        return Err(MetalError::CLenMismatch {
            expected: mn,
            actual: c_buf.len(),
        });
    }

    Ok(())
}

/// [`MetalGemm::dispatch_tiled_prepared`]（イシュー #572）の入力検証。
///
/// [`validate_prepared_inputs`]（f16 版。PR #346 codex-review P1-1 指摘）と
/// 判定ロジックは同一だが、引数型が [`MetalBuffer`]（f32）のため独立実装
/// する（本ファイル既存の `validate_dims`/`validate_dims_f16` と同じ
/// 「クレートをまたいだ検証ロジック共有はスコープ外」という既存方針を
/// 型違いの同クレート内複製にも適用する判断）。
///
/// [`validate_effective_dims`] 通過後にのみ呼ばれる前提のため、
/// `m_eff*k_eff` 等の積は `usize` の範囲でオーバーフローしないことが
/// 保証されている（`checked_mul` を再度呼ぶ必要はない）。
fn validate_prepared_inputs_f32(
    a_buf: &MetalBuffer,
    b_buf: &MetalBuffer,
    c_buf: &MetalBuffer,
    m_eff: usize,
    n_eff: usize,
    k_eff: usize,
) -> Result<(), MetalError> {
    if !m_eff.is_multiple_of(8) || !n_eff.is_multiple_of(8) || !k_eff.is_multiple_of(8) {
        return Err(MetalError::NotEightAligned {
            m_eff,
            n_eff,
            k_eff,
        });
    }

    let mk = m_eff * k_eff;
    let kn = k_eff * n_eff;
    let mn = m_eff * n_eff;

    if a_buf.len() != mk {
        return Err(MetalError::ALenMismatch {
            expected: mk,
            actual: a_buf.len(),
        });
    }
    if b_buf.len() != kn {
        return Err(MetalError::BLenMismatch {
            expected: kn,
            actual: b_buf.len(),
        });
    }
    if c_buf.len() != mn {
        return Err(MetalError::CLenMismatch {
            expected: mn,
            actual: c_buf.len(),
        });
    }

    Ok(())
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

    // SAFETY: FFI 境界 1/2。`setBuffer_offset_atIndex` は生存中の
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

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタから
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
        GemmVariant::SimdgroupTiled(_) => unreachable!(
            "SimdgroupTiled は encode_dispatch_tiled を使う（呼び出し元 dispatch_variant の内部不変条件）"
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

/// `gemm_tiled_bias_act`（イシュー #605）用のパイプライン設定・バッファ
/// 結線（index 0〜3）・`Dims`（index 4）・`has_bias`（index 5）・`act`
/// （index 6）の `setBytes`・ディスパッチを行う。
/// [`MetalGemm::run_tiled_bias_act_f32`] が [`MetalContext::dispatch_sync`]
/// のクロージャから呼ぶ。threadgroup・grid 計算は `Tiled` variant と同一
/// （16×16・`div_ceil(16)`。`gemm_tiled_bias_act` は `gemm_tiled` と同じ
/// タイリング形状のため）。
#[allow(clippy::too_many_arguments)]
fn encode_dispatch_bias_act(
    encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    a_buf: &MetalBuffer,
    b_buf: &MetalBuffer,
    bias_buf: &MetalBuffer,
    c_buf: &MetalBuffer,
    dims: Dims,
    has_bias: i32,
    act: i32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2。`encode_dispatch` の同種コメントと同一の
    // 契約（`a_buf`/`b_buf`/`bias_buf`/`c_buf` は `dispatch_sync` の同期
    // 完了まで呼び出し元スタックフレームで生存する）。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(bias_buf.raw()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(c_buf.raw()), 0, 3);
    }

    // SAFETY: FFI 境界 2/2。`encode_dispatch` の同種コメントと同一の
    // 契約（各ローカル変数は本呼び出し中生存し、型・バイト数は
    // `shaders/gemm.metal::gemm_tiled_bias_act` の
    // `constant Dims&`/`constant int&` 宣言と一致させている）。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&dims).cast(),
            std::mem::size_of::<Dims>(),
            4,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&has_bias).cast(),
            std::mem::size_of::<i32>(),
            5,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&act).cast(),
            std::mem::size_of::<i32>(),
            6,
        );
    }

    let threads_per_tg = MTLSize {
        width: THREADGROUP_SIDE,
        height: THREADGROUP_SIDE,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: (dims.n as usize).div_ceil(THREADGROUP_SIDE),
        height: (dims.m as usize).div_ceil(THREADGROUP_SIDE),
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// `gemm_simdgroup_f16`（TASK-8.3b・#156）用のパイプライン設定・
/// バッファ結線・ディスパッチ。[`MetalGemm::dispatch_f16_prepared_unverified`] が
/// [`MetalContext::dispatch_sync`] のクロージャから呼ぶ。threadgroup・
/// grid の計算方式は `encode_dispatch` の `GemmVariant::Simdgroup` 分岐と
/// 同一（1 threadgroup = 1 simdgroup = C の 8×8 タイル 1 つ。`dims` は
/// 呼び出し元が [`pad8`] で 8 の倍数へ揃え済みの実効次元）。
fn encode_dispatch_f16(
    encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    a_buf: &MetalHalfBuffer,
    b_buf: &MetalHalfBuffer,
    c_buf: &MetalHalfBuffer,
    dims: Dims,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_dispatch` の SAFETY コメント（FFI 境界 1/2）と同一の
    // 契約（`a_buf`/`b_buf`/`c_buf` は `dispatch_sync` の同期完了まで
    // 呼び出し元スタックフレームで生存する）。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(c_buf.raw()), 0, 2);
    }

    // SAFETY: `encode_dispatch` の SAFETY コメント（FFI 境界 2/2）と同一の
    // 契約（`dims` はローカル変数、長さは `size_of::<Dims>()` と一致）。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&dims).cast(),
            std::mem::size_of::<Dims>(),
            3,
        );
    }

    let threads_per_tg = MTLSize {
        width: SIMDGROUP_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: (dims.n as usize) / 8,
        height: (dims.m as usize) / 8,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// [`GemmVariant::SimdgroupTiled`] 用のパイプライン設定・バッファ結線・
/// threadgroup 共有メモリ長設定・ディスパッチ（TASK-1.8f・#188）。
/// [`MetalGemm::dispatch_variant`] が [`MetalContext::dispatch_sync`] の
/// クロージャから呼ぶ。`cfg` は [`MetalGemm::pipeline_for_tile`] が
/// フォールバック解決した後の実際の構成（`resolved_cfg`）。
///
/// grid は素朴には `div_ceil(dims.n, cfg.bn) × div_ceil(dims.m, cfg.bm)`
/// （threadgroup 単位のブロック分割数）だが、`swizzle_enabled`
/// （呼び出し元 [`MetalGemm`] インスタンスが保持する固定値。既定は本番経路
/// `tile::SWIZZLE_ENABLED` = `false`。イシュー #540・実験的機構）が `true`
/// の場合のみ threadgroup ID スウィズル（`swizzle_log` 相当）を適用するため
/// `tile::tiled_dispatch_grid_with` へ委譲する（`shaders/gemm.metal` の
/// `gemm_simdgroup_tiled` 冒頭の tgid 変換と 1:1 対応する契約。採否は
/// `docs/perf/metal-gemm-tgid-swizzle-ab.md` の A/B 計測で判断する）。
/// `swizzle_enabled=false` では素朴な `(tiles_n, tiles_m)` grid をそのまま
/// 使う。シェーダ側は同じ `swizzle_enabled` 値から決まる `SWIZZLE_ENABLED`
/// function constant（`crate::pipeline::make_pipeline_with_constants`。
/// 呼び出し元が同一の `MetalGemm::swizzle_enabled` を両呼び出しへ渡す責務を
/// 負う）で同期し、恒等変換（`tid_y = tgid.y`・`tid_x = tgid.x`）になる。
/// threadgroup スレッド数は `cfg.thread_count()`（`wm*wn*32`）。
#[allow(clippy::too_many_arguments)]
fn encode_dispatch_tiled(
    encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    a_buf: &MetalBuffer,
    b_buf: &MetalBuffer,
    c_buf: &MetalBuffer,
    dims: Dims,
    cfg: TileConfig,
    swizzle_enabled: bool,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_dispatch` の SAFETY コメント（FFI 境界 1/2）と同一の
    // 契約。`a_buf`/`b_buf`/`c_buf` は `dispatch_sync` の同期完了まで
    // 呼び出し元スタックフレームで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(c_buf.raw()), 0, 2);
    }

    // SAFETY: `encode_dispatch` の SAFETY コメント（FFI 境界 2/2）と同一の
    // 契約（`dims` はローカル変数、長さは `size_of::<Dims>()` と一致）。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&dims).cast(),
            std::mem::size_of::<Dims>(),
            3,
        );
    }

    // SAFETY: `setThreadgroupMemoryLength_atIndex` は
    // `dispatchThreadgroups` 時に GPU 側が確保する threadgroup 共有メモリの
    // バイト長を指定するだけで、即座のメモリアクセスは発生しない。
    // `shaders/gemm.metal` の `gemm_simdgroup_tiled` は `threadgroup(0)` に
    // `TileConfig::shared_mem_bytes()`（A タイル BM×(BK+pad) ＋ B タイル
    // BK×(BN+pad)。`pad` はイシュー #538 の threadgroup memory パディング
    // 幅）分のバイト数を要求する契約。`staged=false`（直接ロード経路。
    // small-shape 構成・最終フォールバック `SINGLE_SIMDGROUP_8X8` を含む）
    // では共有メモリを使わず 0 バイトを返すが、`setThreadgroupMemoryLength`
    // へ 0 を渡すと未定義動作になりうるため非 0 の下限値が必要になる
    // （カーネル側は `USE_TGP_STAGING=false` の分岐でこの領域へアクセス
    // しない）。Metal は `setThreadgroupMemoryLength` の長さに 16 バイト
    // 境界整合を要求するため、下限値は 4 ではなく 16 バイト境界の最小値
    // （16）にする。`staged=true` 側は `bm`/`bk`/`bn` が常に 8 の倍数・
    // `pad`（[`TileConfig::pad`]）が常に 4 の倍数（`staged` からの導出値
    // 自体が保証する構成不変条件。イシュー #538 codex-review 指摘 P1 再指摘
    // 対応・PR #673 で `TileConfig::validate` による実行時検証ではなく型の
    // 設計自体で保証する方式へ変更した。`crate::tile` の `TileConfig`
    // ドキュメント「破壊的変更を伴わない導入設計」節参照）のため、
    // `(bm*(bk+pad) + bk*(bn+pad))*4` は常に 256 以上かつ 16 の倍数になり
    // 非 staged 側の下限を上書きしない（bugbot 指摘。#253 レビュー）。
    let shared_mem_bytes = cfg.shared_mem_bytes().max(16) as usize;
    debug_assert!(
        shared_mem_bytes.is_multiple_of(16),
        "Metal は setThreadgroupMemoryLength に 16 バイト境界整合を要求する"
    );
    unsafe {
        encoder.setThreadgroupMemoryLength_atIndex(shared_mem_bytes, 0);
    }

    let threads_per_tg = MTLSize {
        width: cfg.thread_count() as usize,
        height: 1,
        depth: 1,
    };
    let tiles_n = (dims.n as usize).div_ceil(cfg.bn as usize);
    let tiles_m = (dims.m as usize).div_ceil(cfg.bm as usize);
    let (grid_w, grid_h) = crate::tile::tiled_dispatch_grid_with(tiles_n, tiles_m, swizzle_enabled);
    let threadgroups = MTLSize {
        width: grid_w,
        height: grid_h,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// `gemm_simdgroup_tiled_f16`（イシュー #796）用のパイプライン設定・
/// バッファ結線・threadgroup 共有メモリ長設定・ディスパッチ。
/// [`MetalGemm::dispatch_f16_tiled_prepared_unverified`] が
/// [`MetalContext::dispatch_sync`] のクロージャから呼ぶ。`encode_dispatch_tiled`
/// （f32 版）と構造は同一（`cfg` は [`MetalGemm::pipeline_for_tile_f16`] が
/// フォールバック解決した後の実際の構成・grid 計算は `tile::
/// tiled_dispatch_grid_with` へ同じく委譲）だが、以下 2 点が異なる:
///
/// 1. バッファ型が [`MetalHalfBuffer`]（f16）。
/// 2. `setThreadgroupMemoryLength_atIndex` へ渡す長さは
///    [`TileConfig::shared_mem_bytes_f16`]（`staged` を問わずエピローグ
///    staging 領域を含む。f32 版 `shared_mem_bytes` とは異なる計算式。
///    `crate::tile::TileConfig::shared_mem_bytes_f16` ドキュメントコメント
///    参照）を使う。
#[allow(clippy::too_many_arguments)]
fn encode_dispatch_tiled_f16(
    encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    a_buf: &MetalHalfBuffer,
    b_buf: &MetalHalfBuffer,
    c_buf: &MetalHalfBuffer,
    dims: Dims,
    cfg: TileConfig,
    swizzle_enabled: bool,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_dispatch` の SAFETY コメント（FFI 境界 1/2）と同一の
    // 契約。`a_buf`/`b_buf`/`c_buf` は `dispatch_sync` の同期完了まで
    // 呼び出し元スタックフレームで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(c_buf.raw()), 0, 2);
    }

    // SAFETY: `encode_dispatch` の SAFETY コメント（FFI 境界 2/2）と同一の
    // 契約（`dims` はローカル変数、長さは `size_of::<Dims>()` と一致）。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&dims).cast(),
            std::mem::size_of::<Dims>(),
            3,
        );
    }

    // SAFETY: `encode_dispatch_tiled`（f32 版）の同名 SAFETY コメントと同一
    // の契約。`TileConfig::shared_mem_bytes_f16` は `staged` の有無を問わず
    // エピローグ staging 領域を含むため常に非 0（`SINGLE_SIMDGROUP_8X8` でも
    // 256 バイト）であり `.max(16)` は事実上 no-op だが、f32 版との対称性・
    // 将来の構成変更に対する fail-safe として同じ下限適用を維持する。
    let shared_mem_bytes = cfg.shared_mem_bytes_f16().max(16) as usize;
    debug_assert!(
        shared_mem_bytes.is_multiple_of(16),
        "Metal は setThreadgroupMemoryLength に 16 バイト境界整合を要求する"
    );
    unsafe {
        encoder.setThreadgroupMemoryLength_atIndex(shared_mem_bytes, 0);
    }

    let threads_per_tg = MTLSize {
        width: cfg.thread_count() as usize,
        height: 1,
        depth: 1,
    };
    let tiles_n = (dims.n as usize).div_ceil(cfg.bn as usize);
    let tiles_m = (dims.m as usize).div_ceil(cfg.bm as usize);
    let (grid_w, grid_h) = crate::tile::tiled_dispatch_grid_with(tiles_n, tiles_m, swizzle_enabled);
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

    // `validate_prepared_inputs`（PR #346 codex-review P1-1 指摘の入力検証）
    // は引数に `&MetalHalfBuffer` を取るため Linux 上の pure 単体テストが
    // 書けない（`MetalHalfBuffer` の確保に Metal デバイスが必要）。8 の倍数・
    // バッファ長不一致の両方の拒否は `tests/cpu_metal_f16_parity.rs::
    // f16_dispatch_prepared_rejects_undersized_and_misaligned_inputs`
    // （`#[ignore]`・Metal 実機依存）で検証する。
    //
    // `validate_prepared_inputs_f32`（イシュー #572・`dispatch_tiled_prepared`
    // の入力検証）も同じ理由（`&MetalBuffer` の確保に Metal デバイスが
    // 必要）で Linux 上の pure 単体テストが書けない。8 の倍数・バッファ長
    // 不一致の拒否は `tests/gemm_dynamic_tile_parity.rs::
    // dispatch_tiled_prepared_rejects_undersized_and_misaligned_inputs`
    // （`#[ignore]`・Metal 実機依存）で検証する。

    // --- validate_dims_f16（pure・実機不要。TASK-8.3b・#156） ---

    #[test]
    fn validate_dims_f16_accepts_valid_shape() {
        let a = vec![half::f16::from_f32(0.0); 6]; // m=2, k=3
        let b = vec![half::f16::from_f32(0.0); 12]; // k=3, n=4
        let dims = validate_dims_f16(&a, &b, 2, 4, 3).unwrap();
        assert_eq!((dims.m, dims.n, dims.k), (2, 4, 3));
    }

    #[test]
    fn validate_dims_f16_rejects_zero_k() {
        let err = validate_dims_f16(&[], &[], 2, 4, 0).unwrap_err();
        assert!(matches!(
            err,
            MetalError::ZeroDimension { m: 2, n: 4, k: 0 }
        ));
    }

    #[test]
    fn validate_dims_f16_rejects_a_len_mismatch() {
        let a = vec![half::f16::from_f32(0.0); 5]; // m*k=6 を期待
        let b = vec![half::f16::from_f32(0.0); 12];
        let err = validate_dims_f16(&a, &b, 2, 4, 3).unwrap_err();
        assert!(matches!(
            err,
            MetalError::ALenMismatch {
                expected: 6,
                actual: 5
            }
        ));
    }

    #[test]
    fn validate_dims_f16_rejects_dimension_exceeding_u32() {
        let over_u32 = u32::MAX as usize + 1;
        let err = validate_dims_f16(&[], &[], over_u32, 1, 1).unwrap_err();
        assert!(matches!(
            err,
            MetalError::DimensionExceedsU32 { m, n: 1, k: 1 } if m == over_u32
        ));
    }

    // --- validate_bias_act_dims（pure・実機不要。イシュー #605） ---

    #[test]
    fn validate_bias_act_dims_accepts_valid_shape() {
        let a = [0.0f32; 6]; // m=2, k=3
        let b = [0.0f32; 12]; // k=3, n=4
        let dims = validate_bias_act_dims(a.len(), b.len(), 2, 4, 3).unwrap();
        assert_eq!((dims.m, dims.n, dims.k), (2, 4, 3));
    }

    #[test]
    fn validate_bias_act_dims_accepts_zero_m_n_k_unlike_validate_dims() {
        // `run_tiled_bias_act_f32` は m/n/k==0 を縮退分岐として受理する契約
        // であり、`validate_dims`（`ZeroDimension` で一律拒否）とは異なる。
        // 長さ検証は `a_len == m*k`・`b_len == k*n` を要求するため（m/n/k
        // のいずれかが 0 でも `a_len`／`b_len` は対応する非ゼロの積を渡す
        // 必要がある。Cursor Bugbot 指摘・PR #717 レビュースレッド:
        // 旧テストは 3 ケース全てで a_len=0, b_len=0 を渡しており、
        // (m=0,n=4,k=3) は kn=12 との不一致で `BLenMismatch`、
        // (m=2,n=0,k=3) は mk=6 との不一致で `ALenMismatch` を返し、
        // 意図した成功ケースになっていなかった。k=0 のケースのみ
        // mk=kn=0 になり元のまま成功する）。
        assert!(validate_bias_act_dims(0, 12, 0, 4, 3).is_ok());
        assert!(validate_bias_act_dims(6, 0, 2, 0, 3).is_ok());
        assert!(validate_bias_act_dims(0, 0, 2, 4, 0).is_ok());
    }

    #[test]
    fn validate_bias_act_dims_rejects_a_len_mismatch() {
        let err = validate_bias_act_dims(5, 12, 2, 4, 3).unwrap_err();
        assert!(matches!(
            err,
            MetalError::ALenMismatch {
                expected: 6,
                actual: 5
            }
        ));
    }

    #[test]
    fn validate_bias_act_dims_rejects_b_len_mismatch() {
        let err = validate_bias_act_dims(6, 11, 2, 4, 3).unwrap_err();
        assert!(matches!(
            err,
            MetalError::BLenMismatch {
                expected: 12,
                actual: 11
            }
        ));
    }

    #[test]
    fn validate_bias_act_dims_rejects_dimension_exceeding_u32() {
        let over_u32 = u32::MAX as usize + 1;
        let err = validate_bias_act_dims(0, 0, over_u32, 1, 1).unwrap_err();
        assert!(matches!(
            err,
            MetalError::DimensionExceedsU32 { m, n: 1, k: 1 } if m == over_u32
        ));
    }

    // --- GemmVariant ---

    #[test]
    fn gemm_variant_function_names_match_shader_kernel_names() {
        assert_eq!(GemmVariant::Naive.function_name(), "gemm_naive");
        assert_eq!(GemmVariant::Tiled.function_name(), "gemm_tiled");
        assert_eq!(GemmVariant::Simdgroup.function_name(), "gemm_simdgroup");
        assert_eq!(
            GemmVariant::SimdgroupTiled(TileConfig::SINGLE_SIMDGROUP_8X8).function_name(),
            "gemm_simdgroup_tiled"
        );
    }

    // --- SIMDGROUP_THREADGROUP_WIDTH と gemm.metal のハードコード結合 ---

    /// `gemm.metal` の `include_str!` 済みソース全文。`gemm_simdgroup_f16`
    /// エピローグ書き戻しループ（`for (uint i = tid; i < 64; i += 32u)`）が
    /// `SIMDGROUP_THREADGROUP_WIDTH` と一致していることを機械検証するために
    /// 使う（このモジュール自体はビルド対象ではなく文字列検査のみ）。
    const GEMM_METAL_SOURCE_FOR_WIDTH_CHECK: &str = include_str!("shaders/gemm.metal");

    /// レビュー指摘（イシュー #383）: `gemm_simdgroup_f16` のエピローグ
    /// 書き戻しループの刻み幅 `32u` は `encode_dispatch_f16` が dispatch する
    /// threadgroup 幅 `SIMDGROUP_THREADGROUP_WIDTH` とコメントで結合を明示
    /// しているだけで、両者の一致を機械的にロックする専用テストがなかった
    /// （Rust 側定数を変更しても shader 側の `32u` が追随せずサイレントに
    /// 欠落／過剰書き込みが起きうる）。本テストは `SIMDGROUP_THREADGROUP_WIDTH`
    /// の実値から生成した `{n}u` リテラルが `gemm.metal` のエピローグ
    /// ループ刻み幅として実在することを contains 検査でロックし、値の乖離を
    /// 検出可能にする（`crates/backend-metal/tests/shader_source_evidence.rs`
    /// と同じ静的検査方式。Linux CI で GPU 実機不要）。
    // --- run_tiled_bias_act_f32 融合カーネル起動カウンタ（イシュー #605。
    //     Metal 実機依存。`BIAS_ACT_FUSED_LAUNCH_COUNT` は `pub(crate)`
    //     のため、クレート境界外の `tests/` からは参照できない
    //     〈`resolve_tile_config` ドキュメンテーションコメントが記録する
    //     codex-review 是正と同じ理由で `#[doc(hidden)] pub` 化を避ける〉。
    //     「フォールバック非経由」の確認は本クレート内テストに閉じる）。

    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn run_tiled_bias_act_f32_increments_fused_launch_counter() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        let a = vec![1.0f32; 4 * 4];
        let b = vec![1.0f32; 4 * 4];
        let bias = vec![0.0f32; 4];

        let before = BIAS_ACT_FUSED_LAUNCH_COUNT.with(|c| c.get());
        gemm.run_tiled_bias_act_f32(&ctx, &a, &b, Some(&bias), true, 4, 4, 4)
            .expect("run_tiled_bias_act_f32 must succeed on Metal-equipped test runner");
        let after = BIAS_ACT_FUSED_LAUNCH_COUNT.with(|c| c.get());

        assert_eq!(
            after,
            before + 1,
            "bias=[n] は融合カーネルへ進み BIAS_ACT_FUSED_LAUNCH_COUNT を \
             1 増加させる契約"
        );
    }

    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn run_tiled_bias_act_f32_k_zero_does_not_increment_fused_launch_counter() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        let bias = vec![0.0f32; 4];

        let before = BIAS_ACT_FUSED_LAUNCH_COUNT.with(|c| c.get());
        let out = gemm
            .run_tiled_bias_act_f32(&ctx, &[], &[], Some(&bias), false, 3, 4, 0)
            .expect("run_tiled_bias_act_f32 (k=0) must succeed without touching the GPU");
        let after = BIAS_ACT_FUSED_LAUNCH_COUNT.with(|c| c.get());

        assert_eq!(out, vec![0.0f32; 3 * 4]);
        assert_eq!(
            after, before,
            "k=0 はホスト側 epilogue のみで完結し GPU 起動を回避する契約 \
             （BIAS_ACT_FUSED_LAUNCH_COUNT は増加しない）"
        );
    }

    #[test]
    fn simdgroup_f16_epilogue_stride_matches_threadgroup_width_constant() {
        // イシュー #796 で追加した `gemm_simdgroup_tiled_f16` のエピローグも
        // `i += 32u`（`simd_lane` の刻み幅。1 simdgroup = 32 レーン）という
        // 別の不変条件で偶然同じリテラルを使うため、`GEMM_METAL_SOURCE_
        // FOR_WIDTH_CHECK`（ファイル全文）への `contains` のままだと
        // `gemm_simdgroup_f16` 側の刻み幅を `SIMDGROUP_THREADGROUP_WIDTH` と
        // 無関係な値へ書き換えても、新カーネル側の `i += 32u` に偶然一致し
        // 検出できない偽陰性が生じる（`gemm_simdgroup_tiled_f16` の刻み幅は
        // `simd_lane`／simdgroup レーン数に結合したハードウェア定数であり、
        // `gemm_simdgroup_f16` の刻み幅が結合する Rust 側 dispatch 幅
        // `SIMDGROUP_THREADGROUP_WIDTH` とは別の不変条件のため、同じ needle
        // を新カーネルへ拡張しない）。検査範囲を `gemm_simdgroup_f16`
        // カーネル本体（`typedef simdgroup_half8x8 MM_T;` から
        // `gemm_simdgroup_tiled(` 開始位置まで。
        // `tests/shader_source_evidence.rs::gemm_simdgroup_f16_kernel_body`
        // と同じ境界の取り方）に限定する。
        let kernel_start = GEMM_METAL_SOURCE_FOR_WIDTH_CHECK
            .find("typedef simdgroup_half8x8 MM_T;")
            .expect("gemm_simdgroup_f16 の MM_T typedef が見つかりません");
        let next_kernel_start = GEMM_METAL_SOURCE_FOR_WIDTH_CHECK[kernel_start..]
            .find("kernel void gemm_simdgroup_tiled(")
            .map(|offset| kernel_start + offset)
            .expect(
                "gemm_simdgroup_tiled カーネル本体が見つかりません（次カーネル境界の特定に失敗）",
            );
        let kernel_body = &GEMM_METAL_SOURCE_FOR_WIDTH_CHECK[kernel_start..next_kernel_start];

        let expected_stride = format!("i += {SIMDGROUP_THREADGROUP_WIDTH}u");
        assert!(
            kernel_body.contains(&expected_stride),
            "gemm.metal の gemm_simdgroup_f16 エピローグ刻み幅が \
             SIMDGROUP_THREADGROUP_WIDTH（{SIMDGROUP_THREADGROUP_WIDTH}）と \
             一致しません。`{expected_stride}` が見つかりませんでした"
        );
    }
}
