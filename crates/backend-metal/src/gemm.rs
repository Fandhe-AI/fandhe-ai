//! GEMM 公開入口（naive: TASK-1.8b・#39／tiled・simdgroup: TASK-1.8c・#40）。
//!
//! `crate::pipeline::compile_gemm_library`・`crate::pipeline::make_pipeline`
//! でビルドした `gemm_naive`/`gemm_tiled`/`gemm_simdgroup` の 3 パイプライン
//! を保持する [`MetalGemm`] を介して [`crate::context::MetalContext::dispatch_sync`]
//! へエンコーダ結線を委ね、[`crate::buffer::MetalBuffer`] で A・B・C を
//! 確保・readback する。`GemmVariant::Simdgroup` は [`crate::pad`] で
//! 8 の倍数へパディングした実効次元でディスパッチし、readback 後に
//! 元の m×n 形状へ切り出す（呼び出し元へパディングを隠蔽する）。
//! `fandhe_ai_backend_cpu::parity::matmul_reference_fma`（本クレートの `dev-dependencies`
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

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::half_buffer::MetalHalfBuffer;
use crate::layout::{self, MatrixLayout, TransposePattern};
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
    /// `fandhe_ai_backend_cuda::gemm::BIAS_ACT_FUSED_LAUNCH_COUNT`〈#599〉と同じ
    /// 役割・同じスレッドローカル化の理由）。
    ///
    /// `ops.rs::MetalBackendOps::gemm_bias_act` の経路選択（融合 vs
    /// `fandhe_ai_tensor_core::backend_ops::BackendOps::gemm_bias_act` デフォルト実装の
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

    /// [`MetalGemm::dispatch_strided_tiled_prepared`] が実際に呼ばれた回数
    /// （イシュー #1138）。`BIAS_ACT_FUSED_LAUNCH_COUNT` と同じ設計判断
    /// （スレッドローカル化の理由も同一。上記コメント参照）。
    /// `dispatch_strided_bias_act_prepared` の tiled 経路ルーティングが
    /// 実際に発火しているかをクレート内テストで確認するための可観測点。
    pub(crate) static STRIDED_TILED_ROUTE_COUNT: Cell<u64> = const { Cell::new(0) };

    /// [`MetalGemm::encode_tiled_by_class`]（`TileClassMode::Split`）が
    /// 内部タイル（Interior）領域を実際に dispatch した回数（イシュー
    /// #1327）。`STRIDED_TILED_ROUTE_COUNT` と同じ設計判断（スレッド
    /// ローカル化の理由も同一）。実機テストが「Split 経路で内部クラスが
    /// 空振りせず実際に起動した」ことを assert するための可観測点。
    pub(crate) static TILE_CLASS_INTERIOR_DISPATCH_COUNT: Cell<u64> = const { Cell::new(0) };

    /// [`MetalGemm::encode_tiled_by_class`]（`TileClassMode::Split`）が
    /// 端タイル（Edge）領域を実際に dispatch した回数（イシュー #1327）。
    pub(crate) static TILE_CLASS_EDGE_DISPATCH_COUNT: Cell<u64> = const { Cell::new(0) };

    /// [`MetalGemm::encode_tiled_by_class`]（`TileClassMode::Split`）が
    /// Interior/Edge の解決構成不一致により Legacy 単一 dispatch へ
    /// フォールバックした回数（イシュー #1327）。
    pub(crate) static TILE_CLASS_SPLIT_FALLBACK_COUNT: Cell<u64> = const { Cell::new(0) };
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
    /// 構成を自動選択する入口は [`MetalGemm::dispatch_auto`]（[`tile::select_for_device`]
    /// を使う。`tile::select_with_occupancy_for_device` による occupancy 縮退はイシュー
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

/// `shaders/gemm.metal` の `GemmStrides` 構造体とレイアウトを一致させる
/// （`repr(C)`・4 × u32 = 16 バイト。イシュー #1040）。
/// [`MetalGemm::dispatch_strided_bias_act_prepared`] が
/// `crate::layout::MatrixLayout` から構築し `setBytes_length_atIndex`
/// （buffer index 7）で渡す。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GemmStrides {
    lda: u32,
    ldb: u32,
    trans_a: u32,
    trans_b: u32,
}

impl GemmStrides {
    /// NN（両オペランドとも行優先 contiguous）構成。`lda == k`・
    /// `ldb == n`・転置フラグ両方 0。既存 [`MetalGemm::
    /// dispatch_bias_act_prepared`]（後方互換入口）が使う。
    fn nn(k: u32, n: u32) -> Self {
        GemmStrides {
            lda: k,
            ldb: n,
            trans_a: 0,
            trans_b: 0,
        }
    }

    /// [`crate::layout::MatrixLayout`] の組から構築する。
    fn from_layouts(a: &MatrixLayout, b: &MatrixLayout) -> Result<Self, MetalError> {
        let lda = u32::try_from(a.ld).map_err(|_| MetalError::DimensionExceedsU32 {
            m: a.ld,
            n: 0,
            k: 0,
        })?;
        let ldb = u32::try_from(b.ld).map_err(|_| MetalError::DimensionExceedsU32 {
            m: 0,
            n: b.ld,
            k: 0,
        })?;
        Ok(GemmStrides {
            lda,
            ldb,
            trans_a: u32::from(a.transposed),
            trans_b: u32::from(b.transposed),
        })
    }
}

/// `shaders/gemm.metal` の `TileClassRegion` 構造体とレイアウトを一致させる
/// （`repr(C)`・4 × u32 = 16 バイト。イシュー #1327・E6 試作）。
/// タイルクラス分割時に threadgroup dispatch grid 上のタイル座標矩形
/// （内部タイル／端タイル領域）を表す。`crate::tile::TileClassRegion`
/// （純粋な `TileClassPlan` 計算専用の型）とはフィールド構成が同一だが、
/// FFI 境界（`setBytes_length_atIndex`。buffer index 5）に直接触れる型は
/// 本モジュール側で独立定義する契約（`crate::tile` 側 doc comment 参照）。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TileClassRegion {
    row_off: u32,
    col_off: u32,
    rows: u32,
    cols: u32,
}

impl TileClassRegion {
    /// dispatch grid 全体を覆う恒等領域（`TileClassMode::Legacy` が渡す
    /// 値。原点 0・`tiles_m`×`tiles_n` 全体）。
    fn full_grid(tiles_m: u32, tiles_n: u32) -> Self {
        TileClassRegion {
            row_off: 0,
            col_off: 0,
            rows: tiles_m,
            cols: tiles_n,
        }
    }
}

impl From<tile::TileClassRegion> for TileClassRegion {
    fn from(r: tile::TileClassRegion) -> Self {
        TileClassRegion {
            row_off: r.row_off,
            col_off: r.col_off,
            rows: r.rows,
            cols: r.cols,
        }
    }
}

/// [`MetalGemm::plan_tiled_by_class`] が求める 1 回分の
/// `dispatchThreadgroups` 仕様（イシュー #1328。診断専用の
/// `MetalGemm::diag_encode_tiled_nn` と本番 `MetalGemm::
/// encode_tiled_by_class` が計画フェーズを共有するための中間表現）。
struct TiledDispatchSpec {
    /// この dispatch が使うパイプライン（`TileClass::Legacy`/`Interior`/
    /// `Edge` のいずれかで解決済み）。
    pipeline: objc2::rc::Retained<MtlPipeline>,
    /// この dispatch のスレッドグループ形状・staging 有無を決める構成
    /// （`TileClassMode::Split` では Interior/Edge で異なりうる）。
    cfg: TileConfig,
    /// dispatch grid 上の担当領域（`shaders/gemm.metal::TileClassRegion`
    /// へ `setBytes` する値）。
    region: TileClassRegion,
    /// `setThreadgroupMemoryLength` の算出に使う `TileClass`
    /// （`encode_dispatch_tiled` ドキュメンテーションコメント参照）。
    tile_class: tile::TileClass,
    tgp_pad_elems: u32,
}

/// [`MetalGemm::plan_tiled_by_class`] の戻り値。`resolved_cfg` は
/// フォールバック解決後に実際に採用した構成（`TileClassMode::Split` の
/// 場合は Edge/Interior 双方が一致した構成）、`dispatches` は 1〜3 件の
/// dispatch 仕様（Legacy／フォールバック時は 1 件、Split は Interior
/// （存在すれば）＋端ストリップ最大 2 件）。
struct TiledClassPlan {
    resolved_cfg: TileConfig,
    dispatches: Vec<TiledDispatchSpec>,
}

/// naive・tiled・simdgroup の 3 パイプラインを保持するハンドル。
///
/// [`MetalContext`] とは別に保持する理由: パイプライン構築（MSL コンパイル
/// 込み）は比較的重い処理であり、`MetalGemm::new` を 1 回呼んで使い回す
/// ことを想定する（`MetalContext` はデバイス・キューのみの軽量ハンドル。
/// TASK-1.8a・#38 の責務分離を維持する）。イシュー #930 で
/// `ops::MetalBackendOps::gemm`／`gemm_bias_act` は演算呼び出しごとの
/// 都度構築をやめ、`crate::context_cache::cached_gemm` 経由で
/// `Arc<MetalGemm>` をプロセス内キャッシュから取得する本番経路へ移行した
/// （A/B 計測・自己検証専用の `new_with_swizzle`／`new_with_fine_barrier`／
/// `new_with_source_specialization`〈イシュー #1288〉入口はキャッシュ
/// 対象外のまま従来どおり直接構築する）。
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
    /// 書き込むため内部可変性を持たせる。イシュー #930（プロセス内
    /// コンテキストキャッシュ化）で `MetalGemm` を `Arc` 経由の複数スレッド
    /// 共有ハンドル（`crate::context_cache::cached_gemm`）にしたため、
    /// `!Sync` な `RefCell` から `Mutex` へ変更した（`RefCell` のままだと
    /// `MetalGemm: !Sync` となり `Arc<MetalGemm>` を跨スレッド共有できず
    /// `context_cache` の `Send + Sync` コンパイル時アサーションが破綻する）。
    /// イシュー #1138: キーへ [`TransposePattern`] を加えた
    /// `(TileConfig, TransposePattern)`。`gemm_simdgroup_tiled` が
    /// `TRANS_A`/`TRANS_B` function constant で NT/TN/TT へも特殊化される
    /// ようになったため、同一 `TileConfig` でもパターンが異なれば別
    /// パイプライン（MSL コンパイル結果）になる。NN（`TransposePattern::Nn`）
    /// のみを渡す既存呼び出し元（`dispatch_variant`・`dispatch_tiled_prepared`）
    /// の挙動・キャッシュヒット率は変わらない。
    tiled_cache: Mutex<
        HashMap<(TileConfig, TransposePattern, tile::TileClass), objc2::rc::Retained<MtlPipeline>>,
    >,
    /// `gemm_simdgroup_tiled_f16`（イシュー #796）の構成キー
    /// （[`TileConfig`]）→ パイプラインの遅延キャッシュ（[`Self::pipeline_for_tile_f16`]）。
    /// f32 版 [`Self::tiled_cache`] と同じ設計判断（候補構成は有限個のため
    /// 初回ディスパッチ時に構築したパイプラインを使い回す）だが、
    /// `gemm_simdgroup_tiled`（f32）と `gemm_simdgroup_tiled_f16`
    /// は関数名が異なる別パイプラインのため、同一 `TileConfig` キーで
    /// あってもキャッシュを混在させない（f32 版キャッシュとの取り違えは
    /// 誤った関数へのディスパッチに直結するため独立フィールドとして持つ）。
    /// `tiled_cache` と同じ理由（イシュー #930）で `RefCell` から `Mutex`
    /// へ変更した。
    tiled_f16_cache: Mutex<HashMap<TileConfig, objc2::rc::Retained<MtlPipeline>>>,
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
    /// `SimdgroupTiled` **f32 経路**（[`Self::pipeline_for_tile`]）で
    /// 有効化するかどうか。`shaders/gemm.metal` の `gemm_simdgroup_tiled`
    /// staged 経路のみが `FINE_BARRIER_ENABLED` を参照する。
    /// [`Self::pipeline_for_tile_f16`] にも同じ値を伝播するが、
    /// `gemm_simdgroup_tiled_f16` は `FINE_BARRIER_ENABLED` を参照しないため
    /// 値の特殊化自体は無害な no-op（`crate::pipeline::
    /// make_pipeline_with_constants` の同名引数ドキュメンテーションコメント
    /// 参照）で、f16 経路の挙動は本フィールドの値に関わらず不変（f16 側へ
    /// 細粒度同期を実装する場合は本コメントと合わせて更新すること）。
    /// `swizzle_enabled` と同じ設計判断（instance フィールド化により
    /// base（`false`）/head（`true`）の 2 `MetalGemm` を同一プロセス内に
    /// 構築して interleaved A/B 計測できるようにする）。`MetalGemm::new` は
    /// 本番既定 `tile::FINE_BARRIER_ENABLED`（`false`）を渡すため既定挙動は
    /// 不変。
    fine_barrier_enabled: bool,
    /// `gemm_simdgroup_tiled` のアキュムレータ系ループへの条件付き
    /// loop unroll（イシュー #1282）をこのインスタンスで opt-in するか
    /// どうか。`swizzle_enabled`/`fine_barrier_enabled` と同じ設計判断
    /// （instance フィールド化により base（`false`）/head（`true`）の
    /// 2 `MetalGemm` を同一プロセス内に構築して interleaved A/B 計測できる
    /// ようにする。性能実測・本番既定切替は兄弟イシュー #1284）。実際に
    /// unroll 版へ切り替わるかは本フラグと候補の acc 積閾値の AND
    /// （`crate::tile::unroll_acc_loops_for`）で決まる。`MetalGemm::new` は
    /// 本番既定 `tile::UNROLL_ACC_ENABLED`（`false`）を渡すため既定挙動は
    /// 不変。
    unroll_acc_enabled: bool,
    /// ソーステキスト特殊化経路（イシュー #1288。E2 試作）をこのインスタンス
    /// の `SimdgroupTiled` **f32 経路**（[`Self::pipeline_for_tile`]）で
    /// 有効化するかどうか。`swizzle_enabled`/`fine_barrier_enabled`/
    /// `unroll_acc_enabled` と同じ設計判断（instance フィールド化により
    /// base（`false`）/head（`true`）の 2 `MetalGemm` を同一プロセス内に
    /// 構築して bit 一致を自己検証できるようにする）。`MetalGemm::new` は
    /// 本番既定 `tile::SOURCE_SPECIALIZATION_ENABLED`（`false`）を渡すため
    /// 既定挙動は不変。性能実測・本番既定切替は行わない（後続イシュー
    /// #1289／#1302 のスコープ。`docs/perf/metal-gemm-n4096-kernel-gap.md`
    /// §8）。`true` の場合、[`Self::pipeline_for_tile`] は
    /// `self.tiled_cache`（function constant 経路のキャッシュ）を一切
    /// 更新せず [`Self::tiled_spec_cache`] のみを使う（両経路の取り違えを
    /// 構造的に防ぐため、`tiled_f16_cache` と同じく独立フィールドとする）。
    source_specialized: bool,
    /// [`Self::source_specialized`] が `true` のときに使う、ソーステキスト
    /// 特殊化経路専用のパイプラインキャッシュ（[`Self::tiled_cache`] とは
    /// 独立。キーは `tiled_cache` と同じ `(TileConfig, TransposePattern)`）。
    /// `#[cfg(test)]` の [`Self::source_specialized_cache_len`]／
    /// [`Self::function_constant_cache_len`] が「実際にどちらの経路が
    /// 走ったか」を検証するために参照する。
    tiled_spec_cache: Mutex<
        HashMap<(TileConfig, TransposePattern, tile::TileClass), objc2::rc::Retained<MtlPipeline>>,
    >,
    /// フラグメントロード方式候補（イシュー #1293）をこのインスタンスの
    /// `SimdgroupTiled` **f32 経路**（[`Self::pipeline_for_tile`]）で
    /// どう有効化するか。`swizzle_enabled`/`fine_barrier_enabled`/
    /// `unroll_acc_enabled`/`source_specialized` と同じ設計判断（instance
    /// フィールド化により base（`tile::FRAG_LOAD_CONFIG`。既定）/head
    /// （任意の [`tile::FragLoadConfig`]）の 2 `MetalGemm` を同一プロセス
    /// 内に構築して bit 一致を自己検証できるようにする）。
    /// `crate::pipeline::GemmGateConstants::frag_load_device_hoisted`/
    /// `frag_load_ksteps` として `pipeline_for_tile` 経由で
    /// `shaders/gemm.metal` の `FRAG_LOAD_DEVICE_HOISTED`（index 12）/
    /// `FRAG_LOAD_KSTEPS`（index 13）へ畳み込まれる。`MetalGemm::new` は
    /// 本番既定 `tile::FRAG_LOAD_CONFIG`（`device_hoisted=false`・
    /// `ksteps=One`）を渡すため既定挙動は不変。[`Self::pipeline_for_tile_f16`]
    /// にも同じ値を伝播するが `gemm_simdgroup_tiled_f16` はいずれの定数も
    /// 参照しないため無害な no-op（他ゲートと同じ扱い）。性能実測・
    /// `tile::select` への組み込み判断は兄弟イシュー #1295 のスコープ。
    frag_load: tile::FragLoadConfig,
    /// 協調ロードレイアウト候補（イシュー #1298）をこのインスタンスの
    /// `SimdgroupTiled` **f32 staged 経路**（[`Self::pipeline_for_tile`]）
    /// でどう有効化するか。`swizzle_enabled`/`fine_barrier_enabled`/
    /// `unroll_acc_enabled`/`source_specialized`/`frag_load` と同じ設計
    /// 判断（instance フィールド化により base（`tile::COOP_LOAD_CONFIG`。
    /// 既定）/head（任意の [`tile::CoopLoadConfig`]）の 2 `MetalGemm` を
    /// 同一プロセス内に構築して bit 一致を自己検証できるようにする）。
    /// `crate::pipeline::GemmGateConstants::tgp_pad_elems`/
    /// `coop_load_layout` として `pipeline_for_tile` 経由で
    /// `shaders/gemm.metal` の `TGP_PAD`（index 6）/`COOP_LOAD_LAYOUT`
    /// （index 14）へ畳み込まれる。`MetalGemm::new` は本番既定
    /// `tile::COOP_LOAD_CONFIG`（`RowLinear`・`Four`）を渡すため既定挙動は
    /// 不変。[`Self::pipeline_for_tile_f16`] には既定値（`cfg.pad()`／`0`）
    /// のみを渡す no-op 契約（呼び出し側コメント参照）。**本 sub-issue
    /// （#1298）は機構の実装と bit 一致の自己検証のみを行い、性能実測・
    /// `tile::select` への組み込み判断は行わない**（後続イシュー #1300／
    /// #1302／#1304 のスコープ）。
    coop_load: tile::CoopLoadConfig,
    /// タイルクラス分割（イシュー #1327・E6 試作）をこのインスタンスの
    /// `SimdgroupTiled` **f32 経路**（[`Self::pipeline_for_tile`]・
    /// [`Self::encode_tiled_by_class`]）で有効化するかどうか。
    /// `swizzle_enabled`/`fine_barrier_enabled`/`unroll_acc_enabled`/
    /// `source_specialized`/`frag_load`/`coop_load` と同じ設計判断
    /// （instance フィールド化により base（`TileClassMode::Legacy`。既定）/
    /// head（`TileClassMode::Split`）の 2 `MetalGemm` を同一プロセス内に
    /// 構築して bit 一致を自己検証できるようにする）。`MetalGemm::new` は
    /// 本番既定 `tile::TILE_CLASS_MODE`（`Legacy`）を渡すため既定挙動は
    /// 不変。[`Self::pipeline_for_tile_f16`] には常に `TileClass::Legacy`
    /// のみを渡す no-op 契約（呼び出し側コメント参照）。本 sub-issue
    /// （#1327）は機構の実装と bit 一致の自己検証のみを行い、性能実測・
    /// `tile::select` への組み込み判断は行わない（兄弟イシュー #1328 の
    /// スコープ）。
    tile_class_mode: tile::TileClassMode,
}

/// `tiled_cache`／`tiled_f16_cache`（`Mutex` 化。イシュー #930）の共通
/// ロックヘルパー。poison を [`MetalError::ContextCacheUnavailable`] へ
/// 変換し panic させない（`crate::context_cache::lock_cache` と同じ変換
/// 方針。`.claude/rules/coding-rust.md`「本番経路で unwrap/expect を
/// 使わない」）。
fn lock_tile_cache<K: std::hash::Hash + Eq, T>(
    mutex: &Mutex<HashMap<K, T>>,
) -> Result<std::sync::MutexGuard<'_, HashMap<K, T>>, MetalError> {
    mutex
        .lock()
        .map_err(|e| MetalError::ContextCacheUnavailable {
            detail: format!("tile pipeline cache mutex poisoned: {e}"),
        })
}

/// [`MetalGemm::diag_tile_pipeline_reflection`] の戻り値（イシュー
/// #1289）。`MTLComputePipelineState` 構築後にのみ取得できる 3 反射値
/// （`docs/perf/metal-gemm-n4096-kernel-gap.md` §2 の H1 仮説検証と同じ
/// 3 値）に加え、要求構成・解決構成（`pipeline_for_tile` フォールバック
/// 後）・要求スレッド数を並べて持つ（フォールバックが起きていないかを
/// 反射値取得側でも独立に確認できるようにするため。`gemm_reuse_phase_
/// diag_tests.rs::PhaseSample::resolved_cfg` と同じ設計判断）。テスト
/// モジュール外へ漏出させない `#[cfg(test)] pub(crate)`。
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct TilePipelineReflectionDiag {
    pub(crate) requested_cfg: TileConfig,
    pub(crate) resolved_cfg: TileConfig,
    pub(crate) requested_thread_count: u32,
    pub(crate) max_total_threads_per_threadgroup: u32,
    pub(crate) thread_execution_width: u32,
    pub(crate) static_threadgroup_memory_length: u32,
}

impl MetalGemm {
    /// `shaders/gemm.metal` を `ctx` のデバイス上でコンパイルし、
    /// `gemm_naive`/`gemm_tiled`/`gemm_simdgroup` の 3 パイプラインを
    /// 構築する。`gemm_simdgroup_tiled`（TASK-1.8f・#188）はコンパイル済み
    /// ライブラリのみ保持し、構成別パイプラインは `Self::pipeline_for_tile`
    /// が初回ディスパッチ時に遅延構築する（候補が有限個でも全構成を
    /// 前もって構築すると起動コストが増えるため）。
    ///
    /// threadgroup ID スウィズル（イシュー #540）は本番既定
    /// `tile::SWIZZLE_ENABLED`（`false`）で [`Self::new_with_swizzle`] へ
    /// 委譲する薄いラッパー（同フィールドドキュメンテーションコメント参照）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        Self::new_with_swizzle(ctx, tile::SWIZZLE_ENABLED)
    }

    /// [`Self::new`] と同じ構築を行うが、`gemm_simdgroup_tiled` の
    /// アキュムレータ系ループへの条件付き loop unroll（イシュー #1282）の
    /// 有効・無効を明示的な `unroll_acc_enabled` 引数で指定する。ベンチ・
    /// 実機 `#[ignore]` bit 一致テスト用の入口: 同一プロセス内で base
    /// （`unroll_acc_enabled=false`）/head（`unroll_acc_enabled=true`）の
    /// 2 インスタンスを構築して比較する（[`Self::new_with_swizzle`]・
    /// [`Self::new_with_fine_barrier`] と同型の設計）。他 2 フラグ
    /// （threadgroup ID スウィズル・simdgroup 細粒度同期）は本番既定
    /// （`tile::SWIZZLE_ENABLED`／`tile::FINE_BARRIER_ENABLED`。いずれも
    /// `false`）のまま据え置く。本番経路（[`Self::new`]）は常に
    /// `tile::UNROLL_ACC_ENABLED`（`false`）を渡すため、本関数の追加自体は
    /// 既定挙動を変えない。性能実測・本番既定の `true` への切替判断は
    /// 兄弟イシュー #1284 のスコープ。
    pub fn new_with_unroll_acc(
        ctx: &MetalContext,
        unroll_acc_enabled: bool,
    ) -> Result<Self, MetalError> {
        Self::new_with_gates(
            ctx,
            tile::SWIZZLE_ENABLED,
            tile::FINE_BARRIER_ENABLED,
            unroll_acc_enabled,
            tile::SOURCE_SPECIALIZATION_ENABLED,
            tile::FRAG_LOAD_CONFIG,
            tile::COOP_LOAD_CONFIG,
            tile::TILE_CLASS_MODE,
        )
    }

    /// [`Self::new`] と同じ構築を行うが、`gemm_simdgroup_tiled` の
    /// フラグメントロード方式候補（イシュー #1293）を明示的な `frag_load`
    /// 引数で指定する。実機 `#[ignore]` bit 一致自己検証テスト・#1295 の
    /// A/B 計測専用の入口: 同一プロセス内で base（`tile::FRAG_LOAD_CONFIG`。
    /// 既定）/head（任意の [`tile::FragLoadConfig`]）の 2 インスタンスを
    /// 構築して比較する（[`Self::new_with_unroll_acc`]・
    /// [`Self::new_with_source_specialization`] と同型の設計）。他 4 フラグ
    /// （threadgroup ID スウィズル・simdgroup 細粒度同期・条件付き loop
    /// unroll・ソーステキスト特殊化）は本番既定のまま据え置く。本番経路
    /// （[`Self::new`]）は常に `tile::FRAG_LOAD_CONFIG`（`device_hoisted=false`・
    /// `ksteps=One`）を渡すため、本関数の追加自体は既定挙動を変えない。
    /// 性能実測・`tile::select` への組み込み判断は行わない（兄弟イシュー
    /// #1295 のスコープ）。
    ///
    /// `pub`（`tile::FragLoadConfig`/`FragLoadKSteps` を `pub` にしている
    /// 理由も同じ）にする理由: `pub(crate)` のまま `#[cfg(test)]` を
    /// 付けない場合、他クレートからの通常の依存ビルド（`cfg(test)` が
    /// 付かない）で「crate 内に呼び出し元が無い」dead_code 検査に抵触する
    /// （`tile::FragLoadKSteps` doc comment 参照）。`new_with_swizzle`・
    /// `new_with_fine_barrier`・`new_with_unroll_acc`・
    /// `new_with_source_specialization` と同型の設計（実機 `#[ignore]`
    /// bit 一致自己検証テスト・兄弟イシュー #1295 の A/B 計測 example の
    /// 両方から呼べる公開入口）。
    pub fn new_with_frag_load(
        ctx: &MetalContext,
        frag_load: tile::FragLoadConfig,
    ) -> Result<Self, MetalError> {
        Self::new_with_gates(
            ctx,
            tile::SWIZZLE_ENABLED,
            tile::FINE_BARRIER_ENABLED,
            tile::UNROLL_ACC_ENABLED,
            tile::SOURCE_SPECIALIZATION_ENABLED,
            frag_load,
            tile::COOP_LOAD_CONFIG,
            tile::TILE_CLASS_MODE,
        )
    }

    /// [`Self::new`] と同じ構築を行うが、`gemm_simdgroup_tiled` の
    /// 協調ロードレイアウト候補（イシュー #1298）を明示的な `coop_load`
    /// 引数で指定する。実機 `#[ignore]` bit 一致自己検証テスト・後続
    /// イシュー #1300 の A/B 計測専用の入口: 同一プロセス内で base
    /// （`tile::COOP_LOAD_CONFIG`。既定）/head（任意の
    /// [`tile::CoopLoadConfig`]）の 2 インスタンスを構築して比較する
    /// （[`Self::new_with_frag_load`] と同型の設計）。他 5 フラグ
    /// （threadgroup ID スウィズル・simdgroup 細粒度同期・条件付き loop
    /// unroll・ソーステキスト特殊化・フラグメントロード方式候補）は本番
    /// 既定のまま据え置く。本番経路（[`Self::new`]）は常に
    /// `tile::COOP_LOAD_CONFIG`（`RowLinear`・`Four`）を渡すため、本関数の
    /// 追加自体は既定挙動を変えない。性能実測・`tile::select` への組み込み
    /// 判断は行わない（後続イシュー #1300／#1302／#1304 のスコープ）。
    ///
    /// `pub` にする理由は [`Self::new_with_frag_load`] doc comment と同じ
    /// （`tile::CoopLoadConfig`/`CoopLoadLayout`/`TgpPad` を `pub` にして
    /// いる以上、本関数も少なくとも同じ可視性が必要。`pub(crate)` のまま
    /// `#[cfg(test)]` を付けない場合の dead_code 検査抵触も同型）。
    pub fn new_with_coop_load(
        ctx: &MetalContext,
        coop_load: tile::CoopLoadConfig,
    ) -> Result<Self, MetalError> {
        Self::new_with_gates(
            ctx,
            tile::SWIZZLE_ENABLED,
            tile::FINE_BARRIER_ENABLED,
            tile::UNROLL_ACC_ENABLED,
            tile::SOURCE_SPECIALIZATION_ENABLED,
            tile::FRAG_LOAD_CONFIG,
            coop_load,
            tile::TILE_CLASS_MODE,
        )
    }

    /// テスト専用: このインスタンスが保持する [`tile::CoopLoadConfig`] を
    /// 取得する（実機 `#[ignore]` テストが base/head インスタンスの構成を
    /// 突き合わせる用途。イシュー #1298）。
    #[cfg(test)]
    pub(crate) fn coop_load(&self) -> tile::CoopLoadConfig {
        self.coop_load
    }

    /// [`Self::new`] と同じ構築を行うが、`gemm_simdgroup_tiled` の 1
    /// dispatch を内部タイル／端タイルの 2 クラスへ分割する機構
    /// （イシュー #1327・E6 試作）を明示的な `tile_class_mode` 引数で
    /// 指定する。実機 `#[ignore]` bit 一致自己検証テスト・兄弟イシュー
    /// #1328 の性能実測専用の入口: 同一プロセス内で base
    /// （`tile::TILE_CLASS_MODE`＝`Legacy`。既定）/head（`TileClassMode::
    /// Split`）の 2 インスタンスを構築して比較する（[`Self::
    /// new_with_coop_load`] と同型の設計）。他 6 フラグ（threadgroup ID
    /// スウィズル・simdgroup 細粒度同期・条件付き loop unroll・ソース
    /// テキスト特殊化・フラグメントロード方式候補・協調ロードレイアウト
    /// 候補）は本番既定のまま据え置く。本番経路（[`Self::new`]）は常に
    /// `tile::TILE_CLASS_MODE`（`Legacy`）を渡すため、本関数の追加自体は
    /// 既定挙動を変えない。性能実測・`tile::select` への組み込み判断は
    /// 行わない（兄弟イシュー #1328 のスコープ）。
    ///
    /// `pub` にする理由は [`Self::new_with_coop_load`] doc comment と同じ
    /// （`tile::TileClassMode` を `pub` にしている以上、本関数も少なくとも
    /// 同じ可視性が必要。`pub(crate)` のまま `#[cfg(test)]` を付けない
    /// 場合の dead_code 検査抵触も同型）。
    pub fn new_with_tile_class(
        ctx: &MetalContext,
        tile_class_mode: tile::TileClassMode,
    ) -> Result<Self, MetalError> {
        Self::new_with_gates(
            ctx,
            tile::SWIZZLE_ENABLED,
            tile::FINE_BARRIER_ENABLED,
            tile::UNROLL_ACC_ENABLED,
            tile::SOURCE_SPECIALIZATION_ENABLED,
            tile::FRAG_LOAD_CONFIG,
            tile::COOP_LOAD_CONFIG,
            tile_class_mode,
        )
    }

    /// テスト専用: このインスタンスが保持する [`tile::TileClassMode`] を
    /// 取得する（実機 `#[ignore]` テストが base/head インスタンスの構成を
    /// 突き合わせる用途。イシュー #1327）。
    #[cfg(test)]
    pub(crate) fn tile_class_mode(&self) -> tile::TileClassMode {
        self.tile_class_mode
    }

    /// [`Self::new`] と同じ構築を行うが、`gemm_simdgroup_tiled` の
    /// ソーステキスト特殊化経路（イシュー #1288。E2 試作）の有効・無効を
    /// 明示的な `source_specialized` 引数で指定する。実機 `#[ignore]`
    /// bit 一致自己検証テスト専用の入口: 同一プロセス内で base
    /// （`source_specialized=false`。function constant 経路）/head
    /// （`source_specialized=true`。ソーステキスト特殊化経路）の 2
    /// インスタンスを構築して比較する（[`Self::new_with_unroll_acc`] と
    /// 同型の設計）。他 3 フラグ（threadgroup ID スウィズル・simdgroup
    /// 細粒度同期・条件付き loop unroll）は本番既定のまま据え置く。本番
    /// 経路（[`Self::new`]）は常に `tile::SOURCE_SPECIALIZATION_ENABLED`
    /// （`false`）を渡すため、本関数の追加自体は既定挙動を変えない。
    /// 性能実測・本番既定の `true` への切替判断は行わない（後続イシュー
    /// #1289／#1302 のスコープ）。
    pub fn new_with_source_specialization(
        ctx: &MetalContext,
        source_specialized: bool,
    ) -> Result<Self, MetalError> {
        Self::new_with_gates(
            ctx,
            tile::SWIZZLE_ENABLED,
            tile::FINE_BARRIER_ENABLED,
            tile::UNROLL_ACC_ENABLED,
            source_specialized,
            tile::FRAG_LOAD_CONFIG,
            tile::COOP_LOAD_CONFIG,
            tile::TILE_CLASS_MODE,
        )
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
        Self::new_with_gates(
            ctx,
            tile::SWIZZLE_ENABLED,
            fine_barrier_enabled,
            tile::UNROLL_ACC_ENABLED,
            tile::SOURCE_SPECIALIZATION_ENABLED,
            tile::FRAG_LOAD_CONFIG,
            tile::COOP_LOAD_CONFIG,
            tile::TILE_CLASS_MODE,
        )
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
        Self::new_with_gates(
            ctx,
            swizzle_enabled,
            tile::FINE_BARRIER_ENABLED,
            tile::UNROLL_ACC_ENABLED,
            tile::SOURCE_SPECIALIZATION_ENABLED,
            tile::FRAG_LOAD_CONFIG,
            tile::COOP_LOAD_CONFIG,
            tile::TILE_CLASS_MODE,
        )
    }

    /// [`Self::new_with_swizzle`]・[`Self::new_with_fine_barrier`]・
    /// [`Self::new_with_unroll_acc`] が共に委譲する共通実装（イシュー
    /// #809・#1282）。threadgroup ID スウィズル（イシュー #540）・
    /// simdgroup 細粒度同期（イシュー #809）・条件付き loop unroll
    /// （イシュー #1282）はいずれも `crate::pipeline::
    /// make_pipeline_with_constants` の function constant 特殊化
    /// （index 7／8／11。`crate::pipeline::GemmGateConstants`）であり、
    /// A/B 計測の対象軸が異なるだけで構築手順自体は独立のため、3 フラグを
    /// 引数に持つ 1 実装へ集約する（各専用入口が個別に構築ロジックを
    /// 複製すると、パイプライン構築手順の変更時に複数箇所を同期させる
    /// 必要が生じるため）。イシュー #1327 で `tile_class_mode` 引数
    /// （`TILE_CLASS`。index 15）を追加したため 8 引数となり
    /// `clippy::too_many_arguments` を明示的に許容する（既存の設計判断
    /// 〈構築手順を単一実装へ集約する〉を優先し、構造体化はしない。
    /// `GemmGateConstants`/`SpecializationParams` 側は既に構造体化済み）。
    #[allow(clippy::too_many_arguments)]
    fn new_with_gates(
        ctx: &MetalContext,
        swizzle_enabled: bool,
        fine_barrier_enabled: bool,
        unroll_acc_enabled: bool,
        source_specialized: bool,
        frag_load: tile::FragLoadConfig,
        coop_load: tile::CoopLoadConfig,
        tile_class_mode: tile::TileClassMode,
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
            tiled_cache: Mutex::new(HashMap::new()),
            tiled_f16_cache: Mutex::new(HashMap::new()),
            swizzle_enabled,
            fine_barrier_enabled,
            unroll_acc_enabled,
            source_specialized,
            tiled_spec_cache: Mutex::new(HashMap::new()),
            frag_load,
            coop_load,
            tile_class_mode,
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
    ///
    /// `pattern`（イシュー #1138）は `gemm_simdgroup_tiled` の `TRANS_A`/
    /// `TRANS_B` function constant 特殊化を選ぶ。デバイス上限の事前検証は
    /// `TileConfig::shared_mem_bytes_for(pattern)`（NN 以外は物理タイル
    /// 配置が変わるため確保量も変わりうる）で行う。既存呼び出し元
    /// （`dispatch_variant`・`dispatch_tiled_prepared`）は常に
    /// `TransposePattern::Nn` を渡すため、キャッシュキー・検証結果は
    /// 本イシュー以前と変わらない。
    fn pipeline_for_tile(
        &self,
        ctx: &MetalContext,
        cfg: TileConfig,
        pattern: TransposePattern,
        tile_class: tile::TileClass,
    ) -> Result<(Retained<MtlPipeline>, TileConfig), MetalError> {
        let mut last_err: Option<MetalError> = None;

        // イシュー #1288: `source_specialized` に応じてキャッシュ・構築関数
        // を丸ごと切り替える（`tiled_cache`/`make_pipeline_with_constants`
        // 〈function constant 経路〉対 `tiled_spec_cache`/
        // `make_pipeline_source_specialized`〈ソーステキスト特殊化経路〉）。
        // 事前検証（`validate`・`shared_mem_bytes_for`）・ゲート導出
        // （`unroll_acc_loops_for`）・事後検証（`maxTotalThreadsPerThreadgroup`）
        // は両経路で完全に同一の式を使う（base/head を同条件で比較する
        // ための不変条件。計画「設計」節）。
        let cache = if self.source_specialized {
            &self.tiled_spec_cache
        } else {
            &self.tiled_cache
        };

        for candidate in tile::fallback_chain(cfg) {
            if let Some(pipeline) = lock_tile_cache(cache)?.get(&(candidate, pattern, tile_class)) {
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
            // イシュー #1138: NN 専用 `validate` の shared-mem 検査（内部で
            // `shared_mem_bytes()`＝NN 固定式を使う）は転置パターンでは
            // 過小評価しうるため、パターン込みの実際の確保量でも上限検査
            // する。イシュー #1298: パディング幅も `cfg.pad()` 固定ではなく
            // `self.coop_load.pad_elems(candidate)`（協調ロードパディング
            // 候補。0/4/8）を渡す（本番既定では `cfg.pad()` と一致するため
            // 挙動は無変更。`encode_dispatch_tiled` への
            // `setThreadgroupMemoryLength` 呼び出しも同じ式を使う契約
            // — 下記 `tgp_pad_elems` 変数参照）。
            let tgp_pad_elems = self.coop_load.pad_elems(candidate);
            // イシュー #1327 codex-review／Bugbot 指摘（PR #1388）: `TILE_CLASS
            // == 2`（Edge）は `cfg.staged`（`USE_TGP_STAGING`）の値に関わらず
            // 常に staged ロード経路を強制する（`shaders/gemm.metal` の
            // `staging_active` 定義）。事前検証もこの実効値
            // （`TileConfig::shared_mem_bytes_for_class`）で行い、
            // `cfg.staged == false` の構成が Edge クラスとして実際に
            // アクセスする共有メモリ量を過小評価しないようにする
            // （`encode_dispatch_tiled` の実確保量計算と同じ式を使う
            // fail-closed 契約）。
            if candidate.shared_mem_bytes_for_class(pattern, tgp_pad_elems, tile_class)
                > max_shared_mem_bytes
            {
                continue;
            }

            // イシュー #1282: unroll ゲートは要求 `cfg` ではなく、フォール
            // バック chain 巡回中の `candidate` 自身から導出する（要求構成の
            // acc 積を引きずったまま誤った unroll 判定になることを避ける。
            // `crate::tile::unroll_acc_loops_for` doc comment 参照）。
            let gates = pipeline::GemmGateConstants {
                swizzle_enabled: self.swizzle_enabled,
                fine_barrier_enabled: self.fine_barrier_enabled,
                unroll_acc_enabled: tile::unroll_acc_loops_for(candidate, self.unroll_acc_enabled),
                frag_load_device_hoisted: self.frag_load.device_hoisted,
                frag_load_ksteps: self.frag_load.ksteps.as_u32(),
                tgp_pad_elems,
                coop_load_layout: self.coop_load.layout.as_u32(),
                tile_class: tile_class.as_u32(),
            };
            let function_name = GemmVariant::SimdgroupTiled(candidate).function_name();
            let build_result = if self.source_specialized {
                pipeline::make_pipeline_source_specialized(
                    ctx.device(),
                    function_name,
                    candidate,
                    gates,
                    pattern,
                )
            } else {
                pipeline::make_pipeline_with_constants(
                    ctx.device(),
                    &self.library,
                    function_name,
                    candidate,
                    gates,
                    pattern,
                )
            };
            match build_result {
                Ok(pipeline) => {
                    let actual_max_threads = pipeline.maxTotalThreadsPerThreadgroup() as u32;
                    if candidate.thread_count() > actual_max_threads {
                        continue;
                    }
                    lock_tile_cache(cache)?
                        .insert((candidate, pattern, tile_class), Retained::clone(&pipeline));
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

    /// [`Self::encode_tiled_by_class`]・[`Self::diag_encode_tiled_nn`]
    /// （イシュー #1328）が共有する、タイルクラス分割の「どのパイプライン・
    /// 領域・構成で何回 dispatch するか」を決める計画フェーズ（副作用なし。
    /// GPU への実際の記録は行わない）。本番 3 入口（`dispatch_tiled_
    /// prepared`・`dispatch_strided_tiled_prepared`・`dispatch_variant`）は
    /// `self.tile_class_mode == TileClassMode::Legacy`（既定）のため常に
    /// 1 要素の `dispatches` を返す。
    ///
    /// `self.tile_class_mode` で分岐する:
    ///
    /// 1. [`tile::TileClassMode::Legacy`]（本番既定）: [`Self::
    ///    pipeline_for_tile`] を [`tile::TileClass::Legacy`] で 1 回呼び、
    ///    恒等領域（grid 全体）を 1 件返す（既存挙動と完全に同一）。
    /// 2. [`tile::TileClassMode::Split`]: まず Edge クラスでパイプラインを
    ///    解決し（`resolved_cfg` を確定）、続けてその `resolved_cfg` を
    ///    起点に Interior クラスを要求する（`tile::fallback_chain` は
    ///    渡した構成自身を先頭に持つため、両者が同じデバイス制約下にある
    ///    限り同一候補で解決される契約）。**両者の解決構成が一致しない
    ///    場合は fail-closed に [`tile::TileClassMode::Legacy`] 単一
    ///    dispatch へフォールバックする**（`TILE_CLASS_SPLIT_FALLBACK_COUNT`
    ///    で可観測。エラーにはしない）。一致した場合は
    ///    `tile::tile_class_plan` が求めた領域（互いに素・grid 全体を
    ///    過不足なく被覆する。`tile.rs` の網羅テスト参照）ごとに、
    ///    Interior（存在すれば）→ Edge（右ストリップ・下ストリップの順）
    ///    の dispatch 仕様を積む。領域は互いに素なので dispatch順序は
    ///    出力に影響しない。
    ///
    /// `TILE_CLASS_INTERIOR_DISPATCH_COUNT`／`TILE_CLASS_EDGE_DISPATCH_
    /// COUNT`／`TILE_CLASS_SPLIT_FALLBACK_COUNT` の加算は本関数（plan
    /// フェーズ）で行う。旧実装は encode クロージャ内（実際の dispatch
    /// 直前）で加算していたが、`plan_tiled_by_class` は常に直後に
    /// [`Self::encode_tiled_plan`] へ渡されて即座に消費される（呼び出し元
    /// が計画だけ作って握り潰すことはない）ため、可観測な回数は不変。
    fn plan_tiled_by_class(
        &self,
        ctx: &MetalContext,
        cfg: TileConfig,
        dims: Dims,
        pattern: TransposePattern,
    ) -> Result<TiledClassPlan, MetalError> {
        if self.tile_class_mode == tile::TileClassMode::Legacy {
            let (pipeline, resolved_cfg) =
                self.pipeline_for_tile(ctx, cfg, pattern, tile::TileClass::Legacy)?;
            let tiles_m = (dims.m as usize).div_ceil(resolved_cfg.bm as usize) as u32;
            let tiles_n = (dims.n as usize).div_ceil(resolved_cfg.bn as usize) as u32;
            let region = TileClassRegion::full_grid(tiles_m, tiles_n);
            let tgp_pad_elems = self.coop_load.pad_elems(resolved_cfg);
            return Ok(TiledClassPlan {
                resolved_cfg,
                dispatches: vec![TiledDispatchSpec {
                    pipeline,
                    cfg: resolved_cfg,
                    region,
                    tile_class: tile::TileClass::Legacy,
                    tgp_pad_elems,
                }],
            });
        }

        // `TileClassMode::Split`: Edge クラスを先に解決し、その解決構成を
        // 起点に Interior クラスを要求する。
        let (edge_pipeline, edge_cfg) =
            self.pipeline_for_tile(ctx, cfg, pattern, tile::TileClass::Edge)?;
        let (interior_pipeline, interior_cfg) =
            self.pipeline_for_tile(ctx, edge_cfg, pattern, tile::TileClass::Interior)?;

        if interior_cfg != edge_cfg {
            // 解決構成が食い違う場合は Legacy 単一 dispatch へ
            // フォールバックする（fail-closed。設計上ほぼ起こらない
            // ケース——`TileClass` はデバイス制約〈`validate`／共有メモリ
            // 上限〉に影響しないため——だが、将来の拡張余地として明示的に
            // 扱う）。
            TILE_CLASS_SPLIT_FALLBACK_COUNT.with(|c| c.set(c.get() + 1));
            let (pipeline, resolved_cfg) =
                self.pipeline_for_tile(ctx, cfg, pattern, tile::TileClass::Legacy)?;
            let tiles_m = (dims.m as usize).div_ceil(resolved_cfg.bm as usize) as u32;
            let tiles_n = (dims.n as usize).div_ceil(resolved_cfg.bn as usize) as u32;
            let region = TileClassRegion::full_grid(tiles_m, tiles_n);
            let tgp_pad_elems = self.coop_load.pad_elems(resolved_cfg);
            return Ok(TiledClassPlan {
                resolved_cfg,
                dispatches: vec![TiledDispatchSpec {
                    pipeline,
                    cfg: resolved_cfg,
                    region,
                    tile_class: tile::TileClass::Legacy,
                    tgp_pad_elems,
                }],
            });
        }

        let resolved_cfg = edge_cfg;
        let plan = tile::tile_class_plan(dims.m, dims.n, dims.k, resolved_cfg);
        let tgp_pad_interior = self.coop_load.pad_elems(interior_cfg);
        let tgp_pad_edge = self.coop_load.pad_elems(edge_cfg);

        let mut dispatches = Vec::with_capacity(3);
        if let Some(interior) = plan.interior
            && !interior.is_empty()
        {
            TILE_CLASS_INTERIOR_DISPATCH_COUNT.with(|c| c.set(c.get() + 1));
            dispatches.push(TiledDispatchSpec {
                pipeline: Retained::clone(&interior_pipeline),
                cfg: interior_cfg,
                region: interior.into(),
                tile_class: tile::TileClass::Interior,
                tgp_pad_elems: tgp_pad_interior,
            });
        }
        for edge in plan.edges.iter().flatten() {
            if !edge.is_empty() {
                TILE_CLASS_EDGE_DISPATCH_COUNT.with(|c| c.set(c.get() + 1));
                dispatches.push(TiledDispatchSpec {
                    pipeline: Retained::clone(&edge_pipeline),
                    cfg: edge_cfg,
                    region: (*edge).into(),
                    tile_class: tile::TileClass::Edge,
                    tgp_pad_elems: tgp_pad_edge,
                });
            }
        }

        Ok(TiledClassPlan {
            resolved_cfg,
            dispatches,
        })
    }

    /// [`Self::plan_tiled_by_class`] が求めた dispatch 仕様列を、渡された
    /// `encoder` へ順に記録する（GPU への実際の記録・副作用はここで初めて
    /// 発生する）。呼び出し元（[`Self::encode_tiled_by_class`]の
    /// `ctx.dispatch_sync`・[`Self::diag_encode_tiled_nn`]の `ctx.encode`）
    /// が同じ `encoder` を使って 1〜3 回の `dispatchThreadgroups` を記録
    /// する。1 回の `MetalContext::encode`／`dispatch_sync` 呼び出しに
    /// つき 1 回だけラベルが記録される契約（`context.rs`）のため、Split で
    /// 複数回 dispatch しても計測境界（バッチ数・ラベル）は不変。
    #[allow(clippy::too_many_arguments)]
    fn encode_tiled_plan(
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        plan: &TiledClassPlan,
        a_buf: &MetalBuffer,
        a_offset: usize,
        b_buf: &MetalBuffer,
        b_offset: usize,
        c_buf: &MetalBuffer,
        dims: Dims,
        swizzle_enabled: bool,
        strides: GemmStrides,
        pattern: TransposePattern,
    ) {
        for dispatch in &plan.dispatches {
            encode_dispatch_tiled(
                encoder,
                &dispatch.pipeline,
                a_buf,
                a_offset,
                b_buf,
                b_offset,
                c_buf,
                dims,
                dispatch.cfg,
                swizzle_enabled,
                strides,
                pattern,
                dispatch.tgp_pad_elems,
                dispatch.region,
                dispatch.tile_class,
            );
        }
    }

    /// タイルクラス分割（イシュー #1327・E6 試作）を適用した NN/NT/TN/TT
    /// 共通の tiled GEMM dispatch。[`Self::dispatch_tiled_prepared`]・
    /// [`Self::dispatch_strided_tiled_prepared`]・[`Self::dispatch_variant`]
    /// （`SimdgroupTiled` 分岐）の 3 入口が共有するヘルパー（重複実装を
    /// 避ける）。挙動の分岐規則は [`Self::plan_tiled_by_class`] を参照
    /// （本関数は `plan_tiled_by_class` → `ctx.dispatch_sync`（`encode_
    /// tiled_plan`）に薄く委譲するのみ。イシュー #1328 で診断専用の
    /// [`Self::diag_encode_tiled_nn`] と計画フェーズを共有するために分離
    /// した。本番 3 入口の呼び出し・引数・戻り値は非後退）。
    #[allow(clippy::too_many_arguments)]
    fn encode_tiled_by_class(
        &self,
        ctx: &MetalContext,
        a_buf: &MetalBuffer,
        a_offset: usize,
        b_buf: &MetalBuffer,
        b_offset: usize,
        c_buf: &MetalBuffer,
        dims: Dims,
        cfg: TileConfig,
        strides: GemmStrides,
        pattern: TransposePattern,
    ) -> Result<TileConfig, MetalError> {
        let plan = self.plan_tiled_by_class(ctx, cfg, dims, pattern)?;
        let resolved_cfg = plan.resolved_cfg;
        let swizzle_enabled = self.swizzle_enabled;
        ctx.dispatch_sync(|encoder| {
            Self::encode_tiled_plan(
                encoder,
                &plan,
                a_buf,
                a_offset,
                b_buf,
                b_offset,
                c_buf,
                dims,
                swizzle_enabled,
                strides,
                pattern,
            );
        })?;
        Ok(resolved_cfg)
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
            if let Some(pipeline) = lock_tile_cache(&self.tiled_f16_cache)?.get(&candidate) {
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

            // `gemm_simdgroup_tiled_f16` は TRANS_A/TRANS_B（index 9/10）・
            // UNROLL_ACC_ENABLED（index 11）・FRAG_LOAD_DEVICE_HOISTED/
            // FRAG_LOAD_KSTEPS（index 12/13）のいずれも参照しないため、
            // 値の設定自体は無害な no-op（`swizzle_enabled`/
            // `fine_barrier_enabled` と同じ扱い。`unroll_acc_enabled`/
            // `frag_load_device_hoisted` は常に `false` 固定・
            // `frag_load_ksteps` は既定値固定で f16 経路を明示的に不変と
            // 宣言する。`crate::pipeline::make_pipeline_with_constants` の
            // `pattern` 引数ドキュメントコメント参照。イシュー #1138・
            // #1282・#1293）。
            // イシュー #1298: f16 経路は協調ロードレイアウト候補
            // （`COOP_LOAD_LAYOUT`。index 14）を一切参照しないため常に `0`
            // （`CoopLoadLayout::RowLinear`）を渡す no-op 契約
            // （`swizzle_enabled`/`fine_barrier_enabled` と同じ扱い）。ただし
            // `TGP_PAD`（index 6）は f16 版も共有メモリレイアウトの導出に
            // 使うため、従来どおり `candidate.pad()`（`crate::tile::
            // CoopLoadConfig::pad_elems` を経由しない固定 2 値）を渡す。
            let gates = pipeline::GemmGateConstants {
                swizzle_enabled: self.swizzle_enabled,
                fine_barrier_enabled: self.fine_barrier_enabled,
                unroll_acc_enabled: false,
                frag_load_device_hoisted: false,
                frag_load_ksteps: tile::FragLoadConfig::DEFAULT.ksteps.as_u32(),
                tgp_pad_elems: candidate.pad(),
                coop_load_layout: 0,
                // イシュー #1327: f16 経路はタイルクラス分割
                // （`TILE_CLASS`）を一切参照しないため常に `0`
                // （`TileClass::Legacy`）を渡す no-op 契約（他ゲートと
                // 同じ扱い）。
                tile_class: 0,
            };
            match pipeline::make_pipeline_with_constants(
                ctx.device(),
                &self.library,
                "gemm_simdgroup_tiled_f16",
                candidate,
                gates,
                TransposePattern::Nn,
            ) {
                Ok(pipeline) => {
                    let actual_max_threads = pipeline.maxTotalThreadsPerThreadgroup() as u32;
                    if candidate.thread_count() > actual_max_threads {
                        continue;
                    }
                    lock_tile_cache(&self.tiled_f16_cache)?
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
        self.pipeline_for_tile(ctx, cfg, TransposePattern::Nn, tile::TileClass::Legacy)
            .map(|(_, resolved)| resolved)
    }

    /// [`Self::resolve_tile_config`] の転置パターン版（イシュー #1138）。
    /// `tile.rs` の `#[cfg(test)] mod tests` が `CANDIDATES` を NT/TN/TT
    /// 込みで巡回し、フォールバックの有無をパターン別に検証するために使う。
    #[cfg(test)]
    pub(crate) fn resolve_tile_config_for_pattern(
        &self,
        ctx: &MetalContext,
        cfg: TileConfig,
        pattern: TransposePattern,
    ) -> Result<TileConfig, MetalError> {
        self.pipeline_for_tile(ctx, cfg, pattern, tile::TileClass::Legacy)
            .map(|(_, resolved)| resolved)
    }

    /// [`Self::pipeline_for_tile`] が構築・キャッシュした
    /// `MTLComputePipelineState` の反射値（`maxTotalThreadsPerThreadgroup`／
    /// `threadExecutionWidth`／`staticThreadgroupMemoryLength`）を取得する
    /// （イシュー #1289。§8.3「#1289 への引き継ぎ」の指示どおり `#[cfg(test)]
    /// pub(crate)` の薄いアクセサとし、`TilePipelineReflection` 相当の型を
    /// 公開 API 面へ出さない — `docs/perf/metal-gemm-n4096-kernel-gap.md`
    /// §2 の削除記録・PR #1168 codex-review 指摘〈P1〉と同じ判断）。
    /// `self.source_specialized` の値に応じて `pipeline_for_tile` が
    /// function constant 経路／ソーステキスト特殊化経路のどちらを構築するか
    /// が自動的に切り替わるため、呼び出し元は base（`source_specialized=
    /// false`）/head（`true`）の 2 インスタンスへ同じ `cfg`／`pattern` を
    /// 渡すだけで両経路の反射値を取得できる（`gemm_spec_source_diag_tests`
    /// から呼ばれる）。ディスパッチを伴わないため秒未満で完了する。
    #[cfg(test)]
    pub(crate) fn diag_tile_pipeline_reflection(
        &self,
        ctx: &MetalContext,
        cfg: TileConfig,
        pattern: TransposePattern,
    ) -> Result<TilePipelineReflectionDiag, MetalError> {
        let (pipeline, resolved_cfg) =
            self.pipeline_for_tile(ctx, cfg, pattern, tile::TileClass::Legacy)?;
        Ok(TilePipelineReflectionDiag {
            requested_cfg: cfg,
            resolved_cfg,
            requested_thread_count: cfg.thread_count(),
            max_total_threads_per_threadgroup: pipeline.maxTotalThreadsPerThreadgroup() as u32,
            thread_execution_width: pipeline.threadExecutionWidth() as u32,
            static_threadgroup_memory_length: pipeline.staticThreadgroupMemoryLength() as u32,
        })
    }

    /// ソーステキスト特殊化経路（イシュー #1288）のキャッシュ
    /// （[`Self::tiled_spec_cache`]）に登録済みのパイプライン数を返す。
    /// [`Self::function_constant_cache_len`] と対にして、実機テストが
    /// 「head（`source_specialized=true`）は特殊化キャッシュのみが増え、
    /// function constant キャッシュは空のまま」「base はその逆」を
    /// assert するために使う（出力一致だけでは両者が同じ経路へ倒れた
    /// false-green を検出できないため、新経路が実際に走ったことを
    /// 独立に証明する）。
    #[cfg(test)]
    pub(crate) fn source_specialized_cache_len(&self) -> Result<usize, MetalError> {
        Ok(lock_tile_cache(&self.tiled_spec_cache)?.len())
    }

    /// function constant 経路（[`Self::tiled_cache`]）に登録済みの
    /// パイプライン数を返す。[`Self::source_specialized_cache_len`] の
    /// doc comment 参照。
    #[cfg(test)]
    pub(crate) fn function_constant_cache_len(&self) -> Result<usize, MetalError> {
        Ok(lock_tile_cache(&self.tiled_cache)?.len())
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

    /// このインスタンスが `cfg`（フォールバック解決前の要求構成。呼び出し元
    /// は事前に [`Self::resolve_tile_config`] で解決済みの構成を渡す想定）に
    /// 対して実際に使う `UNROLL_ACC_ENABLED` 実効値を返す（イシュー
    /// #1282）。実機 `#[ignore]` bit 一致テスト（`gemm_unroll_acc_bit_match_tests`）
    /// が head インスタンスで「候補 0/4/8 が unroll 版を選んだこと」を
    /// assert するために使う（`Self::pipeline_for_tile` 内部の導出式
    /// `crate::tile::unroll_acc_loops_for` をテストから直接呼べるようにする
    /// 薄いラッパー）。
    #[cfg(test)]
    pub(crate) fn unroll_acc_effective(&self, cfg: TileConfig) -> bool {
        tile::unroll_acc_loops_for(cfg, self.unroll_acc_enabled)
    }

    /// このインスタンスが `pipeline_for_tile` へ渡すフラグメントロード
    /// 方式候補（イシュー #1293）の実効値を返す（テストからの実効値確認
    /// 用。`unroll_acc_effective` と同型の薄いラッパー）。
    #[cfg(test)]
    pub(crate) fn frag_load(&self) -> tile::FragLoadConfig {
        self.frag_load
    }

    /// 動的タイル選択（TASK-1.8f・#188）の自動入口。`(m, n, k)` から
    /// [`tile::select`] で [`TileConfig`] を選び、[`GemmVariant::SimdgroupTiled`]
    /// で [`Self::dispatch_variant`] へ委譲する。バックエンド抽象層からの
    /// accelerated/tiled 経路選択（#67/#68）とはレイヤが異なる（本関数は
    /// 「Metal GEMM を実行すると決まった後」のタイル構成選択のみを担う。
    /// イシュー #188 計画「スコープ外」節）。
    ///
    /// **occupancy 判定（イシュー #542・[`tile::select_with_occupancy_for_device`]）
    /// は不採用確定（イシュー #747）**: `ctx.occupancy_params()` は実機値
    /// （GPU コア数・threadgroup memory 上限）からキャッシュされるが、
    /// #744 是正（段 1 の正方立方形状判定是正）により実測帯域
    /// 〈512/1024/2048/4096〉では occupancy 縮退の適用対象が実質消滅し
    /// [`tile::select_for_device`] と常に同一結果になることを確認したため、
    /// 本番ディスパッチへは組み込まない（`docs/perf/
    /// metal-gemm-occupancy-select.md`「#747 判断」節・`crate::tile::
    /// select_with_occupancy_for_device` ドキュメンテーションコメント参照）。
    /// **`ctx.verified_m4_max_gpu_core_count()` は occupancy 縮退の有効化
    /// とは別に、イシュー #1039 の M4 Max 実測厳密一致テーブル
    /// （`tile::select_with_occupancy_for_device` 内の `exact_match_cfg`）
    /// の機種ゲートとして [`tile::select_for_device`] へ渡す**（P1・
    /// codex-review 指摘・PR #1108 レビュー: 実測機種以外へ無条件適用され
    /// ないようにするため。GPU コア数だけでは機種〈M3 Max との 40 コア
    /// 構成の混同〉を一意に識別できないため、[`MetalContext::
    /// verified_m4_max_gpu_core_count`] が SoC ブランド文字列と組み合わせて
    /// 検証済みの値を渡す。`crate::tile` モジュール `verify_m4_max` 参照）。
    pub fn dispatch_auto(
        &self,
        ctx: &MetalContext,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let cfg = tile::select_for_device(m, n, k, ctx.verified_m4_max_gpu_core_count());
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
        let cfg = tile::select_for_device(m, n, k, ctx.verified_m4_max_gpu_core_count());
        self.dispatch_f16_tiled_unverified(ctx, a, b, m, n, k, cfg)
    }

    /// バックエンド抽象層からの GEMM 自動経路選択入口（TASK-11.2b・#68）。
    ///
    /// `ctx.caps()`（`MetalContext::new` 時にキャッシュした
    /// `MTLDevice::supportsFamily(MTLGPUFamily::Apple7)` 判定）と `(m, n, k)`
    /// から `fandhe_ai_tensor_core::dispatch::select_gemm_kernel` を呼び、その結果
    /// （[`fandhe_ai_tensor_core::dispatch::KernelKind`]）を [`GemmVariant`] へ写像
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
        let shape = fandhe_ai_tensor_core::dispatch::GemmShape::new(
            u32::try_from(m).unwrap_or(u32::MAX),
            u32::try_from(n).unwrap_or(u32::MAX),
            u32::try_from(k).unwrap_or(u32::MAX),
        );
        let kernel = fandhe_ai_tensor_core::dispatch::select_gemm_kernel(
            &ctx.caps(),
            shape,
            fandhe_ai_tensor_core::dispatch::DType::F32,
        );
        match kernel {
            fandhe_ai_tensor_core::dispatch::KernelKind::MatrixUnit => {
                self.dispatch_auto(ctx, a, b, m, n, k)
            }
            fandhe_ai_tensor_core::dispatch::KernelKind::Tiled => {
                self.dispatch_variant(ctx, GemmVariant::Tiled, a, b, m, n, k)
            }
            fandhe_ai_tensor_core::dispatch::KernelKind::Naive => {
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
    /// 渡せるため、`validate_prepared_inputs_f32` で 8 の倍数・バッファ長
    /// 一致をエンコード前に検証する（f16 側 `validate_prepared_inputs`・
    /// PR #346 codex-review P1-1 指摘と同水準の検証）。
    ///
    /// `cfg` は呼び出し元が選んだ候補構成（`tile::select(m, n, k)` 等）だが、
    /// `Self::pipeline_for_tile` がデバイス上限超過等でサイレントに
    /// `TileConfig::SINGLE_SIMDGROUP_8X8` へフォールバックしうるため
    /// （フォールバック透明性は `Self::pipeline_for_tile` ドキュメント
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

        // イシュー #1327: タイルクラス分割（`self.tile_class_mode`）を
        // 適用した共通ヘルパーへ委譲する。`TileClassMode::Legacy`（本番
        // 既定）では従来の 1 回 dispatch と完全に同一の挙動になる。
        self.encode_tiled_by_class(
            ctx,
            a_buf,
            0,
            b_buf,
            0,
            c_buf,
            dims,
            cfg,
            GemmStrides::nn(dims.k, dims.n),
            TransposePattern::Nn,
        )
    }

    /// [`Self::dispatch_tiled_prepared`] と同じ NN 経路（`plan_tiled_by_
    /// class` → `encode_tiled_plan`）を、`encode`（記録のみ）と
    /// `synchronize`（commit + `waitUntilCompleted`）へ分離した診断専用
    /// ヘルパ（イシュー #1189）。
    ///
    /// # 追加理由
    ///
    /// [`MetalContext::dispatch_sync`] は「encode → synchronize」を 1 回で
    /// 行う薄いラッパーであり（`dispatch_sync` ドキュメントコメント参照）、
    /// 公開 API からは encode（コマンドバッファへの記録）と synchronize
    /// （commit + GPU 完了待ち）を個別に計時できない。`crates/backend-metal/
    /// src/gemm_reuse_phase_diag_tests.rs`（reuse 計測境界の transfer／
    /// sync／kernel 内訳一次測定）はこの 2 段の境界を区別する必要がある
    /// ため、本メソッドを新設した。既存の [`Self::dispatch_tiled_prepared`]
    /// 自体は無変更（本メソッドは新規追加のみで、既存関数を書き換えない）。
    ///
    /// # `self.tile_class_mode` を尊重する（イシュー #1328）
    ///
    /// 当初は `pipeline_for_tile(..., TileClass::Legacy)` を直接呼ぶ実装で
    /// `self.tile_class_mode` を無視していたため、`MetalGemm::
    /// new_with_tile_class(Split)` で構築したインスタンスでも常に Legacy
    /// 経路しか測れなかった（イシュー #1328 診断の前提）。[`Self::
    /// plan_tiled_by_class`] へ委譲することで本番 3 入口
    /// （[`Self::encode_tiled_by_class`]）と計画フェーズを共有し、
    /// `tile_class_mode` に応じた Legacy／Split 双方の経路を正しく測れる
    /// ようにした。`tile_class_mode == Legacy`（既定）のときの出力
    /// （パイプライン解決・領域・ラベル・`coop_load.pad_elems`）は従来
    /// 実装と完全に同一（`plan_tiled_by_class` の Legacy 分岐は旧実装の
    /// ロジックをそのまま移したもの）であり、既存の E2〜E4 診断テストの
    /// 前提（`measure_one_phase_trial` の「1 バッチ・1 ラベル」不変条件）
    /// を壊さない。
    ///
    /// `#[cfg(test)]` 限定（本体ビルドには含まれない。AC-2「既存の本番
    /// 経路・既存テストを変更しない（読み取り計測のみ）」に対応）。
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn diag_encode_tiled_nn(
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

        let plan = self.plan_tiled_by_class(ctx, cfg, dims, TransposePattern::Nn)?;
        let resolved_cfg = plan.resolved_cfg;
        let swizzle_enabled = self.swizzle_enabled;
        ctx.encode(
            "diag_encode_tiled_nn",
            &[a_buf.raw(), b_buf.raw(), c_buf.raw()],
            None,
            |encoder| {
                Self::encode_tiled_plan(
                    encoder,
                    &plan,
                    a_buf,
                    0,
                    b_buf,
                    0,
                    c_buf,
                    dims,
                    swizzle_enabled,
                    GemmStrides::nn(dims.k, dims.n),
                    TransposePattern::Nn,
                );
            },
        )?;

        Ok(resolved_cfg)
    }

    /// `gemm_simdgroup_tiled`（`TRANS_A`/`TRANS_B` 拡張。イシュー #1138）への
    /// strided 明示入口。bias/act エピローグは持たない（[`Self::
    /// dispatch_tiled_prepared`] と同じ「GPU 実行のみ」計測境界。呼び出し元
    /// がバイアス無し・活性化無しの場合のみ使う契約）。`a_layout`/
    /// `b_layout`（[`crate::layout::MatrixLayout`]）を介して NT/TN/TT・
    /// offset 付き view をそのまま受け取り、`strided_tiled_eligibility`
    /// を通過した入力のみ `cfg`（フォールバック解決込み）へディスパッチする。
    /// 戻り値は fallback 解決後に実際に採用した構成。
    ///
    /// [`Self::dispatch_strided_bias_act_prepared`] のルーティング判断
    /// （bias/act 無しかつ適格な入力を本関数へ委譲するか）は実測
    /// （`docs/perf/metal-gemm-transpose-tiled.md`）に基づき別途行う（本
    /// 関数自体は常に利用可能な明示入口として提供する）。
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_strided_tiled_prepared(
        &self,
        ctx: &MetalContext,
        a_buf: &MetalBuffer,
        a_offset: usize,
        a_layout: MatrixLayout,
        b_buf: &MetalBuffer,
        b_offset: usize,
        b_layout: MatrixLayout,
        c_buf: &MetalBuffer,
        m: usize,
        n: usize,
        k: usize,
        cfg: TileConfig,
    ) -> Result<TileConfig, MetalError> {
        let (dims, strides) = validate_strided_dims(
            a_buf.len(),
            a_offset,
            a_layout,
            b_buf.len(),
            b_offset,
            b_layout,
            c_buf.len(),
            m,
            n,
            k,
        )?;
        strided_tiled_eligibility(m, n, k, a_layout, a_offset, b_layout, b_offset)?;

        let pattern = TransposePattern::from_flags(a_layout.transposed, b_layout.transposed);
        // イシュー #1327: タイルクラス分割（`self.tile_class_mode`）を
        // 適用した共通ヘルパーへ委譲する（`dispatch_tiled_prepared` と
        // 同じ設計判断）。
        let resolved_cfg = self.encode_tiled_by_class(
            ctx, a_buf, a_offset, b_buf, b_offset, c_buf, dims, cfg, strides, pattern,
        )?;

        STRIDED_TILED_ROUTE_COUNT.with(|c| c.set(c.get() + 1));

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
    /// （`crate::ops::gemm_bias_act_route`）からのみ呼ばれる。
    ///
    /// `bias` が `Some` の場合は `bias.len() == n` を fail-closed に検証
    /// する（`shaders/gemm.metal::gemm_tiled_bias_act` の `bias[col]`
    /// epilogue が `n` 要素を前提とするため）。
    ///
    /// `m == 0 || n == 0` は no-op（空の結果）、`k == 0` は CPU 参照実装
    /// （`fandhe_ai_backend_cpu::gemm_blis::gemm_blis_bias_act_parallel`）・CUDA 側
    /// `run_tiled_bias_act_f32` と同じ契約で epilogue のみホスト側で計算し
    /// GPU 起動を回避する（`BIAS_ACT_FUSED_LAUNCH_COUNT` は増加させない。
    /// `validate_dims`〈`m/n/k == 0` を一律 `ZeroDimension` として拒否〉を
    /// 経由せず、本関数専用の `validate_bias_act_dims` で検証してから
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
        // イシュー #1021: ゼロ初期化が必要なため `alloc_zeroed_pooled`
        // を使う（設計文書 §3.1「ホットパスへの接続点」）。
        let (bias_buf, has_bias): (MetalBuffer, i32) = match bias {
            Some(bias) => (MetalBuffer::new_with_data(ctx, bias)?, 1),
            None => (MetalBuffer::alloc_zeroed_pooled(ctx, n)?, 0),
        };
        // イシュー #1021: C は `encode_dispatch_bias_act` が `m * n`
        // 全要素を書き切る出力専用バッファのため `alloc_uninit_pooled`
        // を使う（`dispatch_variant` と同じ判断。設計文書 §6「A02」）。
        let c_buf = MetalBuffer::alloc_uninit_pooled(ctx, m * n)?;

        let dims = Dims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
        };
        let act_i: i32 = if act_relu { 1 } else { 0 };
        // NN（両オペランドとも行優先 contiguous。`run_tiled_bias_act_f32`
        // は常にホストスライスを密行優先として受け取る契約）。
        let strides = GemmStrides::nn(dims.k, dims.n);

        ctx.dispatch_sync(|encoder| {
            encode_dispatch_bias_act(
                encoder,
                &self.pipeline_tiled_bias_act,
                &a_buf,
                0,
                &b_buf,
                0,
                &bias_buf,
                0,
                &c_buf,
                dims,
                has_bias,
                act_i,
                strides,
            );
        })?;

        // バッファ確保（`MetalBuffer::new_with_data`／`new_zeroed`）・
        // `ctx.dispatch_sync` がすべて成功した後にのみ増加させる（codex-review
        // 指摘・PR #717。確保／dispatch 失敗時に「起動済み」として誤記録
        // すると、経路検証テスト・診断が偽陽性になるため）。
        BIAS_ACT_FUSED_LAUNCH_COUNT.with(|c| c.set(c.get() + 1));

        Ok(c_buf.read_to_vec())
    }

    /// デバイス常駐済みの A/B/(bias)/C バッファに対して bias 加算・
    /// activation 融合 tiled GEMM を実行する（イシュー #1022）。
    /// [`Self::run_tiled_bias_act_f32`] がホストスライスから
    /// `MetalBuffer::new_with_data` で毎回アップロードするのに対し、本
    /// 関数は呼び出し元が既に確保・アップロード済みの [`MetalBuffer`]
    /// をそのまま結線する「GPU 実行のみ」契約（`crate::gemm::MetalGemm::
    /// dispatch_tiled_prepared` と同じ設計方針）。`ops::MetalBackendOps::
    /// gemm_resident_rhs` が `w`（デバイス常駐 weight）をホストへ
    /// download せずに forward するために使う。
    ///
    /// `bias_buf` が `None` の場合は `n` 要素のゼロ初期化バッファを内部で
    /// 確保して渡す（`run_tiled_bias_act_f32` と同じ理由。`shaders/
    /// gemm.metal::gemm_tiled_bias_act` 冒頭コメント「`bias` が `None`」
    /// 参照: 1 要素ダミーでは Metal コンパイラの select 化最適化次第で
    /// 範囲外アクセスになりうる fail-closed 対策）。
    ///
    /// `m == 0 || n == 0` は no-op（呼び出し元が空の `c_buf` を用意する
    /// 前提。#39 系「デバイス常駐済みバッファに対する GPU 実行のみ」契約
    /// と同様、shape 0 の縮退はここでは扱わない——呼び出し元
    /// （`ops::MetalBackendOps`）が事前に検査する）。`k == 0` は
    /// `validate_bias_act_dims` が `a_len == 0`／`b_len == 0` を正しい
    /// 積として受理するため通常どおりディスパッチする（GPU 側で `acc`
    /// が 0 のまま epilogue のみ適用される。ホスト側 early-return は
    /// 行わない——`k == 0` はどのみち `Linear::new` の `in_features > 0`
    /// 制約により実運用では到達しない）。
    /// `a_offset`／`b_offset`／`bias`（`Some` の場合の offset）は要素
    /// 単位のオフセット（イシュー #1023「R3: 要素オフセット付き常駐
    /// ビュー」設計。`docs/device-resident-update-design.md` 追補
    /// 参照）。`#1023` のパラメータ横断連結バッファ化後、`ops::
    /// MetalBackendOps::gemm_resident_rhs`／`gemm_resident_lhs` の
    /// `w`／`bias` は単一の連結 `MetalBuffer` 内の部分範囲としてしか
    /// 表現できないため、`a_buf`／`b_buf` それぞれの物理バッファ全体が
    /// ちょうど `m*k`／`k*n` であることを要求していた旧
    /// `validate_bias_act_dims` 直接呼び出しをやめ、`offset + numel <=
    /// buf.len()` の範囲検査へ置き換える（Apple Silicon の UMA・
    /// `StorageModeShared` のため offset はバイト単位へ変換して
    /// `setBuffer:offset:` へそのまま渡せる。CPU 側配列のオフセット
    /// スライスと同じ発想）。既存の全体バッファ呼び出し（`run_tiled_
    /// bias_act_f32`）は `a_offset = b_offset = bias offset = 0` を渡す
    /// （offset 0 の場合は従来の「バッファ全体 = m*k/k*n」契約と等価）。
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_bias_act_prepared(
        &self,
        ctx: &MetalContext,
        a_buf: &MetalBuffer,
        a_offset: usize,
        b_buf: &MetalBuffer,
        b_offset: usize,
        bias: Option<(&MetalBuffer, usize)>,
        act_relu: bool,
        c_buf: &MetalBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), MetalError> {
        // NN（両オペランドとも行優先 contiguous）委譲。イシュー #1040 で
        // 転置パターン対応 [`Self::dispatch_strided_bias_act_prepared`]
        // を追加した際、本関数はその後方互換の薄いラッパーへ変更した
        // （`lda == k`・`ldb == n`・転置フラグ両方 0 で従来と完全に
        // 同一の添字・数値結果になる。`tests` のビット同一回帰参照）。
        let a_layout = MatrixLayout {
            rows: m,
            cols: k,
            ld: k,
            transposed: false,
        };
        let b_layout = MatrixLayout {
            rows: k,
            cols: n,
            ld: n,
            transposed: false,
        };
        self.dispatch_strided_bias_act_prepared(
            ctx, a_buf, a_offset, a_layout, b_buf, b_offset, b_layout, bias, act_relu, c_buf, m, n,
            k,
        )
    }

    /// [`Self::dispatch_bias_act_prepared`] の転置パターン・stride 対応版
    /// （イシュー #1040）。`a_layout`/`b_layout`
    /// （[`crate::layout::MatrixLayout`]。`crate::layout::classify_2d`／
    /// `crate::layout::collapse_leading_dims` が導出する）を介して、
    /// 転置 view（NT/TN/TT）や先頭次元 collapse 後の view を
    /// `Tensor::contiguous()`（ホスト側転置コピー）を経由せずそのまま
    /// GPU カーネルへ渡す。`ops::MetalBackendOps::gemm_resident_lhs`／
    /// `gemm_resident_rhs` が転置 view を検出した場合にこちらを直接
    /// 呼ぶ。NN（`dispatch_bias_act_prepared` の委譲先としての利用）と
    /// 転置経路の双方をこの 1 関数に集約することで、`GemmStrides` の
    /// 構築・検証ロジックの重複を避ける。`ops::MetalBackendOps::
    /// gemm`（片側のみ転置の NT/TN 判定分岐。イシュー #1215）も同じ
    /// 入口を呼ぶ——`Op::MatMul`／`Op::LinearAct`／`Op::LinearResident`
    /// の VJP が `BackendOps::gemm_fp32_strict`（既定実装が `gemm` へ
    /// 委譲）経由で到達する呼び出し元。
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_strided_bias_act_prepared(
        &self,
        ctx: &MetalContext,
        a_buf: &MetalBuffer,
        a_offset: usize,
        a_layout: MatrixLayout,
        b_buf: &MetalBuffer,
        b_offset: usize,
        b_layout: MatrixLayout,
        bias: Option<(&MetalBuffer, usize)>,
        act_relu: bool,
        c_buf: &MetalBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), MetalError> {
        self.encode_strided_bias_act_prepared(
            ctx, a_buf, a_offset, a_layout, b_buf, b_offset, b_layout, bias, act_relu, c_buf, m, n,
            k,
        )?;
        ctx.synchronize()
    }

    /// [`Self::dispatch_strided_bias_act_prepared`] の encode-only 版
    /// （イシュー #1216・codex-review 指摘対応。`ctx.dispatch_sync` では
    /// なく待たない `MetalContext::encode` でバッチへ積むのみで復帰
    /// する）。`ops::MetalBackendOps::linear_forward_device` が多層 MLP
    /// 推論チェーンの各層をこの経路で連鎖させることで、層ごとの
    /// `waitUntilCompleted` を発生させず最終 `download` の 1 回へ
    /// 同期点を集約する（`ops.rs::linear_forward_device` doc「同期
    /// 契約」参照。`sgd.rs::MetalSgd::run` の `token: Some` 経路と同型の
    /// 「`ctx.encode` に resources を渡して生存を保証する」パターン）。
    ///
    /// `bias` が `None` の場合に確保する一時 `zero_bias`（プール経由）は
    /// 本メソッド復帰時に Rust 側スコープを抜けて drop されるが、
    /// `resources` に `raw()` を渡しているため、その裏の `MTLBuffer` は
    /// `Batch::in_flight` へ retain 済みであり、かつ
    /// `PooledMetalHandle::drop` が `ctx.defer_pool_return` で
    /// 「バッチが in-flight の間はプールへ返却しない」ことを保証する
    /// （`context.rs::defer_pool_return` 参照）ため、GPU 実行完了前に
    /// 実体が再利用されることはない。
    #[allow(clippy::too_many_arguments)]
    pub fn encode_strided_bias_act_prepared(
        &self,
        ctx: &MetalContext,
        a_buf: &MetalBuffer,
        a_offset: usize,
        a_layout: MatrixLayout,
        b_buf: &MetalBuffer,
        b_offset: usize,
        b_layout: MatrixLayout,
        bias: Option<(&MetalBuffer, usize)>,
        act_relu: bool,
        c_buf: &MetalBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), MetalError> {
        let (dims, strides) = validate_strided_dims(
            a_buf.len(),
            a_offset,
            a_layout,
            b_buf.len(),
            b_offset,
            b_layout,
            c_buf.len(),
            m,
            n,
            k,
        )?;
        if let Some((b, offset)) = bias {
            let end = offset
                .checked_add(n)
                .ok_or(MetalError::DimProductOverflow)?;
            if end > b.len() {
                return Err(MetalError::InvalidElementwiseShape {
                    detail: format!(
                        "bias view range [{offset}, {end}) exceeds backing buffer length ({})",
                        b.len()
                    ),
                });
            }
        }

        let zero_bias;
        let (bias_ref, bias_offset, has_bias): (&MetalBuffer, usize, i32) = match bias {
            Some((b, offset)) => (b, offset, 1),
            None => {
                // イシュー #1021: `alloc_zeroed_pooled`（プール経由・
                // ゼロ初期化契約維持）。
                zero_bias = MetalBuffer::alloc_zeroed_pooled(ctx, n)?;
                (&zero_bias, 0, 0)
            }
        };
        let act_i: i32 = if act_relu { 1 } else { 0 };

        // `resources` へ 4 本すべて（`a_buf`／`b_buf`／`bias_ref`／
        // `c_buf`）を渡し、`ctx.encode` 復帰後も `Batch::in_flight` の
        // retain によって GPU 完了（`ctx.synchronize()`）まで実体を
        // 生存させる（本メソッド doc 参照）。
        ctx.encode(
            "gemm_bias_act_strided",
            &[a_buf.raw(), b_buf.raw(), bias_ref.raw(), c_buf.raw()],
            None,
            |encoder| {
                encode_dispatch_bias_act(
                    encoder,
                    &self.pipeline_tiled_bias_act,
                    a_buf,
                    a_offset,
                    b_buf,
                    b_offset,
                    bias_ref,
                    bias_offset,
                    c_buf,
                    dims,
                    has_bias,
                    act_i,
                    strides,
                );
            },
        )?;

        BIAS_ACT_FUSED_LAUNCH_COUNT.with(|c| c.set(c.get() + 1));

        Ok(())
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
    /// 等）にも同じオーバーフロー・`u32::MAX` 検証を `validate_effective_dims`
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
    /// ため、構造体へのまとめ込みは行わない（`fandhe_ai_backend_cpu::gemm::kernel_block`
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
        // イシュー #1021: C は `encode_dispatch`／`encode_dispatch_tiled`
        // が `m_eff * n_eff` 全要素を書き切る出力専用バッファ（下記
        // `match` の両分岐がその全域を dispatch する）のため、ゼロ初期化
        // を経由しないプール確保（`alloc_uninit_pooled`）を使う（設計
        // 文書 §6「A02」: `alloc_uninit` は「カーネルが全要素を書き切る
        // 出力専用」に限定する契約）。
        let c_buf = MetalBuffer::alloc_uninit_pooled(ctx, m_eff * n_eff)?;

        match variant {
            GemmVariant::SimdgroupTiled(cfg) => {
                // イシュー #1327: タイルクラス分割（`self.tile_class_mode`）
                // を適用した共通ヘルパーへ委譲する
                // （`dispatch_tiled_prepared` と同じ設計判断）。
                self.encode_tiled_by_class(
                    ctx,
                    &a_buf,
                    0,
                    &b_buf,
                    0,
                    &c_buf,
                    dims,
                    cfg,
                    GemmStrides::nn(dims.k, dims.n),
                    TransposePattern::Nn,
                )?;
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
/// `fandhe_ai_backend_cpu::gemm::validate_dims`（`crates/backend-cpu/src/gemm.rs`）
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

/// [`MetalGemm::dispatch_strided_bias_act_prepared`] 専用の検証
/// （イシュー #1040）。`validate_bias_act_dims`（全体バッファ =
/// `m*k`/`k*n` を前提とする密行優先専用の検証）と異なり、
/// `a_layout`/`b_layout`（転置・stride 付き view）が論理形状
/// `(m,k)`/`(k,n)` と整合すること、および `offset +
/// required_span(layout) <= buf_len` を fail-closed に検証する。
/// `m == 0 || n == 0`（呼び出し元が no-op として扱う縮退）は
/// `validate_bias_act_dims` と同じく一律拒否しない。
#[allow(clippy::too_many_arguments)]
fn validate_strided_dims(
    a_buf_len: usize,
    a_offset: usize,
    a_layout: MatrixLayout,
    b_buf_len: usize,
    b_offset: usize,
    b_layout: MatrixLayout,
    c_len: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(Dims, GemmStrides), MetalError> {
    if a_layout.rows != m || a_layout.cols != k {
        return Err(MetalError::ShapeMismatch {
            detail: format!(
                "a_layout logical shape [{}, {}] does not match (m, k) = ({m}, {k})",
                a_layout.rows, a_layout.cols
            ),
        });
    }
    if b_layout.rows != k || b_layout.cols != n {
        return Err(MetalError::ShapeMismatch {
            detail: format!(
                "b_layout logical shape [{}, {}] does not match (k, n) = ({k}, {n})",
                b_layout.rows, b_layout.cols
            ),
        });
    }

    m.checked_mul(k).ok_or(MetalError::DimProductOverflow)?;
    k.checked_mul(n).ok_or(MetalError::DimProductOverflow)?;
    m.checked_mul(n).ok_or(MetalError::DimProductOverflow)?;

    if m > u32::MAX as usize || n > u32::MAX as usize || k > u32::MAX as usize {
        return Err(MetalError::DimensionExceedsU32 { m, n, k });
    }

    let a_span = layout::required_span(&a_layout).ok_or(MetalError::DimProductOverflow)?;
    let a_end = a_offset
        .checked_add(a_span)
        .ok_or(MetalError::DimProductOverflow)?;
    if a_end > a_buf_len {
        return Err(MetalError::ALenMismatch {
            expected: a_end,
            actual: a_buf_len,
        });
    }

    let b_span = layout::required_span(&b_layout).ok_or(MetalError::DimProductOverflow)?;
    let b_end = b_offset
        .checked_add(b_span)
        .ok_or(MetalError::DimProductOverflow)?;
    if b_end > b_buf_len {
        return Err(MetalError::BLenMismatch {
            expected: b_end,
            actual: b_buf_len,
        });
    }

    if c_len != m * n {
        return Err(MetalError::CLenMismatch {
            expected: m * n,
            actual: c_len,
        });
    }

    let dims = Dims {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let strides = GemmStrides::from_layouts(&a_layout, &b_layout)?;
    Ok((dims, strides))
}

/// [`MetalGemm::dispatch_strided_tiled_prepared`] の適格性ゲート（純粋
/// 関数。GPU 非依存で Linux 上でも検証可能。イシュー #1138）。
/// `gemm_simdgroup_tiled` の float4 ベクトルロード（staged 経路）・8x8
/// direct-load（`simdgroup_load`/`_store`）は以下の前提に依存するため、
/// 成立しない入力は classic strided（`gemm_tiled_bias_act`）へ
/// フォールバックさせる（fail-closed。`.claude/rules/coding-rust.md`
/// 「カーネル実装の境界検査」）:
///
/// - `m`/`n`/`k` がいずれも非 0・8 の倍数（`simdgroup_store` の 8x8
///   直接ストア・direct-load 経路の pad8 契約前提。`gemm_simdgroup_tiled`
///   自体は 8 の倍数でない実効次元も境界チェック込みで受理できるが、
///   strided 入口は呼び出し元がパディングを行わないため、ここで明示的に
///   要求する）
/// - `a_layout.ld`/`b_layout.ld` が [`TileConfig::VEC_WIDTH`]（4）の倍数、
///   かつ `a_offset`/`b_offset`（要素単位）も同じく 4 の倍数
///   （float4 `reinterpret_cast` の 16 バイト境界。`setBuffer:offset:`
///   後の device 先頭ポインタと合わせて成立させる必要があるため offset も
///   検査する）
fn strided_tiled_eligibility(
    m: usize,
    n: usize,
    k: usize,
    a_layout: MatrixLayout,
    a_offset: usize,
    b_layout: MatrixLayout,
    b_offset: usize,
) -> Result<(), MetalError> {
    if m == 0 || n == 0 || k == 0 {
        return Err(MetalError::StridedTiledIneligible {
            detail: format!("m/n/k must be nonzero (m={m}, n={n}, k={k})"),
        });
    }
    if !m.is_multiple_of(8) || !n.is_multiple_of(8) || !k.is_multiple_of(8) {
        return Err(MetalError::StridedTiledIneligible {
            detail: format!("m/n/k must all be multiples of 8 (m={m}, n={n}, k={k})"),
        });
    }
    let vec_width = TileConfig::VEC_WIDTH as usize;
    if !a_layout.ld.is_multiple_of(vec_width) || !b_layout.ld.is_multiple_of(vec_width) {
        return Err(MetalError::StridedTiledIneligible {
            detail: format!(
                "a_layout.ld={} / b_layout.ld={} must both be multiples of {vec_width}",
                a_layout.ld, b_layout.ld
            ),
        });
    }
    if !a_offset.is_multiple_of(vec_width) || !b_offset.is_multiple_of(vec_width) {
        return Err(MetalError::StridedTiledIneligible {
            detail: format!(
                "a_offset={a_offset} / b_offset={b_offset} must both be multiples of {vec_width}"
            ),
        });
    }
    Ok(())
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

/// `gemm_tiled_bias_act`（イシュー #605。イシュー #1040 で `GemmStrides`
/// 引数を追加）用のパイプライン設定・バッファ結線（index 0〜3）・
/// `Dims`（index 4）・`has_bias`（index 5）・`act`（index 6）・
/// `GemmStrides`（index 7）の `setBytes`・ディスパッチを行う。
/// [`MetalGemm::run_tiled_bias_act_f32`] が [`MetalContext::dispatch_sync`]
/// のクロージャから呼ぶ。threadgroup・grid 計算は `Tiled` variant と同一
/// （16×16・`div_ceil(16)`。`gemm_tiled_bias_act` は `gemm_tiled` と同じ
/// タイリング形状のため）。
#[allow(clippy::too_many_arguments)]
fn encode_dispatch_bias_act(
    encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    a_buf: &MetalBuffer,
    a_offset: usize,
    b_buf: &MetalBuffer,
    b_offset: usize,
    bias_buf: &MetalBuffer,
    bias_offset: usize,
    c_buf: &MetalBuffer,
    dims: Dims,
    has_bias: i32,
    act: i32,
    strides: GemmStrides,
) {
    encoder.setComputePipelineState(pipeline);

    // イシュー #1023「R3」: `*_offset` は要素単位のオフセットであり、
    // Metal の `setBuffer:offset:atIndex:` はバイト単位を要求するため
    // `size_of::<f32>()` を掛けて変換する（呼び出し元
    // `dispatch_bias_act_prepared` が offset+numel の範囲検査を済ませて
    // いるため、ここでの追加検査は不要）。
    let a_byte_offset = a_offset * std::mem::size_of::<f32>();
    let b_byte_offset = b_offset * std::mem::size_of::<f32>();
    let bias_byte_offset = bias_offset * std::mem::size_of::<f32>();

    // SAFETY: FFI 境界 1/2。`encode_dispatch` の同種コメントと同一の
    // 契約（`a_buf`/`b_buf`/`bias_buf`/`c_buf` は `dispatch_sync` の同期
    // 完了まで呼び出し元スタックフレームで生存する）。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a_buf.raw()), a_byte_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(b_buf.raw()), b_byte_offset, 1);
        encoder.setBuffer_offset_atIndex(Some(bias_buf.raw()), bias_byte_offset, 2);
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
        // イシュー #1040: `shaders/gemm.metal::gemm_tiled_bias_act` の
        // `constant GemmStrides& st [[buffer(7)]]` に対応する。この
        // `setBytes` を追加したことで buffer index 7 は
        // `gemm_tiled_bias_act` を起動する全経路（本関数のみ）で必須と
        // なった——本関数は唯一のディスパッチ入口であり、他のパイプライン
        // 構築・エンコードコードパスは存在しない（`grep -rn
        // 'gemm_tiled_bias_act\b'` で確認済み）。
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&strides).cast(),
            std::mem::size_of::<GemmStrides>(),
            7,
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
///
/// イシュー #1138: `a_offset`/`b_offset`（要素単位。`encode_dispatch_bias_act`
/// と同じ規約でバイト単位へ変換して `setBuffer:offset:` へ渡す）・
/// `strides`（`GemmStrides`。`gemm_simdgroup_tiled` の `constant
/// GemmStrides& st [[buffer(4)]]` へ `setBytes`）・`pattern`（threadgroup
/// 共有メモリ長を `TileConfig::shared_mem_bytes_for(pattern)` で計算する
/// ため）を追加した。既存呼び出し元（`dispatch_variant`・
/// `dispatch_tiled_prepared`）は `a_offset = b_offset = 0`・
/// `GemmStrides::nn(dims.k, dims.n)`・`TransposePattern::Nn` を渡すため
/// 挙動は非後退（`pattern == Nn` では `shared_mem_bytes_for` は
/// `shared_mem_bytes()` と同値。`tile.rs` 末尾テスト参照）。
#[allow(clippy::too_many_arguments)]
fn encode_dispatch_tiled(
    encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    a_buf: &MetalBuffer,
    a_offset: usize,
    b_buf: &MetalBuffer,
    b_offset: usize,
    c_buf: &MetalBuffer,
    dims: Dims,
    cfg: TileConfig,
    swizzle_enabled: bool,
    strides: GemmStrides,
    pattern: TransposePattern,
    tgp_pad_elems: u32,
    // イシュー #1327（E6 試作）: タイルクラス分割時の担当領域（threadgroup
    // dispatch grid 上のタイル座標矩形）。`region` に応じて grid を
    // `region.cols`×`region.rows`（`tiles_n`/`tiles_m` 全体ではなく）で
    // 張り、`shaders/gemm.metal::TileClassRegion`（buffer(5)）へ常に
    // `setBytes` する（`TILE_CLASS==0`＝Legacy でも未バインド参照を作らない
    // ため。カーネル側は Legacy では region を一切参照しない no-op）。
    region: TileClassRegion,
    // codex-review／Bugbot 指摘（PR #1388）: 共有メモリ確保量
    // （`setThreadgroupMemoryLength`）を実効ロード方式と一致させるため、
    // 呼び出し元がこの dispatch に使う `pipeline` を解決したのと同じ
    // `tile::TileClass` を渡す（`pipeline_for_tile` の事前検証
    // （`TileConfig::shared_mem_bytes_for_class`）と同一の式を使う契約）。
    tile_class: tile::TileClass,
) {
    encoder.setComputePipelineState(pipeline);

    let a_byte_offset = a_offset * std::mem::size_of::<f32>();
    let b_byte_offset = b_offset * std::mem::size_of::<f32>();

    // SAFETY: `encode_dispatch` の SAFETY コメント（FFI 境界 1/2）と同一の
    // 契約。`a_buf`/`b_buf`/`c_buf` は `dispatch_sync` の同期完了まで
    // 呼び出し元スタックフレームで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a_buf.raw()), a_byte_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(b_buf.raw()), b_byte_offset, 1);
        encoder.setBuffer_offset_atIndex(Some(c_buf.raw()), 0, 2);
    }

    // SAFETY: `encode_dispatch` の SAFETY コメント（FFI 境界 2/2）と同一の
    // 契約（`dims`/`strides` はローカル変数、長さは各々の `size_of` と
    // 一致）。`GemmStrides` は `gemm_tiled_bias_act` と共用のレイアウト
    // 一致テスト対象（本ファイル `#[cfg(test)]` 参照）。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&dims).cast(),
            std::mem::size_of::<Dims>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&strides).cast(),
            std::mem::size_of::<GemmStrides>(),
            4,
        );
        // イシュー #1327: `region` は `TILE_CLASS==0`（Legacy）でも常に
        // バインドする（上記関数ドキュメンテーションコメント参照）。
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&region).cast(),
            std::mem::size_of::<TileClassRegion>(),
            5,
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
    // イシュー #1298: パディング幅は `cfg.pad()` 固定ではなく呼び出し元
    // （`MetalGemm::pipeline_for_tile` と同一の
    // `self.coop_load.pad_elems(candidate)`）が渡す実効値を使う。
    // 事前検証（`pipeline_for_tile`）・共有メモリ確保の両方が同じ
    // `tgp_pad_elems` を参照することで、確保量とカーネルが実際に
    // アクセスする範囲を一致させる fail-closed 契約を維持する（本番既定
    // では `cfg.pad()` と一致するため挙動は無変更）。
    let shared_mem_bytes = cfg
        .shared_mem_bytes_for_class(pattern, tgp_pad_elems, tile_class)
        .max(16) as usize;
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
    // イシュー #1327: grid は `dims`/`cfg` から導出する全体タイル数では
    // なく `region.cols`/`region.rows`（Legacy では常に全体と一致する
    // 恒等領域）で張る。カーネル側のスウィズル変換は region ローカル
    // 座標系（`[0, region.rows) × [0, region.cols)`）で行われ、その後
    // `row_off`/`col_off` を加算する契約（`shaders/gemm.metal` の
    // `TILE_CLASS` ガードのコメント参照）。
    let (grid_w, grid_h) = crate::tile::tiled_dispatch_grid_with(
        region.cols as usize,
        region.rows as usize,
        swizzle_enabled,
    );
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

    // --- GemmStrides / validate_strided_dims（イシュー #1040。pure・実機不要） ---

    /// `shaders/gemm.metal::GemmStrides`（4 × uint32）とのレイアウト一致
    /// （`repr(C)`・16 バイト）を Linux 上で確認する（advisor 指摘:
    /// MSL の実際のフィールドオフセットは Mac 実機コンパイルでしか検証
    /// できないため、少なくとも Rust 側の `repr(C)` サイズ・アライン
    /// メントが「4 個の連続する u32」という前提から外れていないことを
    /// 機械的に固定する）。
    #[test]
    fn gemm_strides_repr_c_layout_matches_msl_struct() {
        assert_eq!(std::mem::size_of::<GemmStrides>(), 16);
        assert_eq!(std::mem::align_of::<GemmStrides>(), 4);
    }

    #[test]
    fn gemm_strides_nn_has_zero_transpose_flags() {
        let s = GemmStrides::nn(4, 5);
        assert_eq!(
            s,
            GemmStrides {
                lda: 4,
                ldb: 5,
                trans_a: 0,
                trans_b: 0,
            }
        );
    }

    #[test]
    fn gemm_strides_from_layouts_encodes_transpose_flags() {
        let a = MatrixLayout {
            rows: 2,
            cols: 3,
            ld: 3,
            transposed: false,
        };
        let b = MatrixLayout {
            rows: 3,
            cols: 4,
            ld: 3,
            transposed: true,
        };
        let s = GemmStrides::from_layouts(&a, &b).unwrap();
        assert_eq!(
            s,
            GemmStrides {
                lda: 3,
                ldb: 3,
                trans_a: 0,
                trans_b: 1,
            }
        );
    }

    #[test]
    fn validate_strided_dims_accepts_nn_and_matches_validate_bias_act_dims() {
        let a_layout = MatrixLayout {
            rows: 2,
            cols: 3,
            ld: 3,
            transposed: false,
        };
        let b_layout = MatrixLayout {
            rows: 3,
            cols: 4,
            ld: 4,
            transposed: false,
        };
        let (dims, strides) =
            validate_strided_dims(6, 0, a_layout, 12, 0, b_layout, 8, 2, 4, 3).unwrap();
        assert_eq!((dims.m, dims.n, dims.k), (2, 4, 3));
        assert_eq!(strides, GemmStrides::nn(3, 4));
    }

    #[test]
    fn validate_strided_dims_accepts_transposed_layout() {
        // 転置 view: 元 [3,2] 行優先バッファ（12 要素…以下略）を A=[2,3] の
        // 転置 view として読む（strides 相当 = ld: 2, transposed: true）。
        let a_layout = MatrixLayout {
            rows: 2,
            cols: 3,
            ld: 2,
            transposed: true,
        };
        let b_layout = MatrixLayout {
            rows: 3,
            cols: 4,
            ld: 4,
            transposed: false,
        };
        // required_span(transposed): (cols-1)*ld + rows = (3-1)*2+2 = 6
        let (_, strides) =
            validate_strided_dims(6, 0, a_layout, 12, 0, b_layout, 8, 2, 4, 3).unwrap();
        assert_eq!(strides.trans_a, 1);
        assert_eq!(strides.trans_b, 0);
    }

    #[test]
    fn validate_strided_dims_rejects_shape_mismatch() {
        let a_layout = MatrixLayout {
            rows: 99,
            cols: 3,
            ld: 3,
            transposed: false,
        };
        let b_layout = MatrixLayout {
            rows: 3,
            cols: 4,
            ld: 4,
            transposed: false,
        };
        let err = validate_strided_dims(300, 0, a_layout, 12, 0, b_layout, 8, 2, 4, 3).unwrap_err();
        assert!(matches!(err, MetalError::ShapeMismatch { .. }));
    }

    #[test]
    fn validate_strided_dims_rejects_span_exceeding_buffer() {
        let a_layout = MatrixLayout {
            rows: 2,
            cols: 3,
            ld: 3,
            transposed: false,
        };
        let b_layout = MatrixLayout {
            rows: 3,
            cols: 4,
            ld: 4,
            transposed: false,
        };
        // a_buf_len = 5 だが required_span = (2-1)*3+3 = 6 のため不足。
        let err = validate_strided_dims(5, 0, a_layout, 12, 0, b_layout, 8, 2, 4, 3).unwrap_err();
        assert!(matches!(
            err,
            MetalError::ALenMismatch {
                expected: 6,
                actual: 5
            }
        ));
    }

    #[test]
    fn validate_strided_dims_rejects_offset_pushing_span_out_of_bounds() {
        let a_layout = MatrixLayout {
            rows: 2,
            cols: 3,
            ld: 3,
            transposed: false,
        };
        let b_layout = MatrixLayout {
            rows: 3,
            cols: 4,
            ld: 4,
            transposed: false,
        };
        // a_offset=2・required_span=6 → end=8 > a_buf_len=6。
        let err = validate_strided_dims(6, 2, a_layout, 12, 0, b_layout, 8, 2, 4, 3).unwrap_err();
        assert!(matches!(
            err,
            MetalError::ALenMismatch {
                expected: 8,
                actual: 6
            }
        ));
    }

    #[test]
    fn validate_strided_dims_rejects_c_len_mismatch() {
        let a_layout = MatrixLayout {
            rows: 2,
            cols: 3,
            ld: 3,
            transposed: false,
        };
        let b_layout = MatrixLayout {
            rows: 3,
            cols: 4,
            ld: 4,
            transposed: false,
        };
        let err = validate_strided_dims(6, 0, a_layout, 12, 0, b_layout, 7, 2, 4, 3).unwrap_err();
        assert!(matches!(
            err,
            MetalError::CLenMismatch {
                expected: 8,
                actual: 7
            }
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

    // --- strided_tiled_eligibility（イシュー #1138。pure・実機不要） ---

    fn nn_layout(rows: usize, cols: usize, ld: usize) -> MatrixLayout {
        MatrixLayout {
            rows,
            cols,
            ld,
            transposed: false,
        }
    }

    #[test]
    fn strided_tiled_eligibility_accepts_eight_divisible_nn_shape() {
        let a = nn_layout(16, 32, 32);
        let b = nn_layout(32, 24, 24);
        assert!(strided_tiled_eligibility(16, 24, 32, a, 0, b, 0).is_ok());
    }

    #[test]
    fn strided_tiled_eligibility_rejects_zero_dims() {
        let a = nn_layout(0, 32, 32);
        let b = nn_layout(32, 24, 24);
        let err = strided_tiled_eligibility(0, 24, 32, a, 0, b, 0).unwrap_err();
        assert!(matches!(err, MetalError::StridedTiledIneligible { .. }));
    }

    #[test]
    fn strided_tiled_eligibility_rejects_non_eight_divisible_m() {
        let a = nn_layout(15, 32, 32);
        let b = nn_layout(32, 24, 24);
        let err = strided_tiled_eligibility(15, 24, 32, a, 0, b, 0).unwrap_err();
        assert!(matches!(err, MetalError::StridedTiledIneligible { .. }));
    }

    #[test]
    fn strided_tiled_eligibility_rejects_non_four_divisible_ld() {
        // `ld` が 8 整除の `rows`/`cols` とは独立に非 4 整除でありうる
        // （view の元バッファが余分な列を持つ場合等）。
        let a = nn_layout(16, 32, 33);
        let b = nn_layout(32, 24, 24);
        let err = strided_tiled_eligibility(16, 24, 32, a, 0, b, 0).unwrap_err();
        assert!(matches!(err, MetalError::StridedTiledIneligible { .. }));
    }

    #[test]
    fn strided_tiled_eligibility_rejects_non_four_divisible_offset() {
        let a = nn_layout(16, 32, 32);
        let b = nn_layout(32, 24, 24);
        let err = strided_tiled_eligibility(16, 24, 32, a, 1, b, 0).unwrap_err();
        assert!(matches!(err, MetalError::StridedTiledIneligible { .. }));
    }

    #[test]
    fn strided_tiled_eligibility_accepts_four_divisible_offset() {
        let a = nn_layout(16, 32, 32);
        let b = nn_layout(32, 24, 24);
        assert!(strided_tiled_eligibility(16, 24, 32, a, 4, b, 8).is_ok());
    }

    // --- 条件付き loop unroll（イシュー #1282）の自己検証 ---
    //
    // `UNROLL_ACC_ENABLED` function constant（`shaders/gemm.metal` index
    // 11）が候補ごとに正しく畳み込まれ、かつ有無で出力が bit 単位で
    // 一致することを実機で確認する（AC-1・AC-2）。`crate::tile::CANDIDATES`
    // は `pub(crate)` のためクレート境界外の `tests/` からは参照できず、
    // 本クレート内テストに閉じる（`resolve_tile_config`・`BIAS_ACT_FUSED_
    // LAUNCH_COUNT` 系の既存テストと同じ配置判断）。

    /// `MetalGemm::unroll_acc_effective` が [`tile::CANDIDATES`] の
    /// acc 積が閾値以上の候補（index 0/4/8）でのみ head インスタンス
    /// （`unroll_acc_enabled=true`）で `true` を返すことを確認する
    /// （イシュー #1282 AC-1）。`crate::tile::unroll_acc_loops_for` の
    /// 単体テスト（`tile.rs` 側）は純粋関数レベルの検証だが、本テストは
    /// 実際の `MetalGemm` インスタンス経由でも同じ結果になることを実機で
    /// 確認する（base インスタンスは常に `false` を返すことも合わせて
    /// 検証）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn unroll_acc_effective_matches_candidate_acc_product_threshold() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_unroll_acc(&ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_unroll_acc(&ctx, true)
            .expect("head GEMM パイプラインの構築に失敗した");

        // イシュー #1329 で index 9（64,64,32,2,2）を追加した後も
        // acc 積 >= 16（`CANDIDATES[0]` と同じ acc_rows=4, acc_cols=4）の
        // ため対象へ入る（`tile.rs::unroll_acc_candidates_are_exactly_
        // acc_product_ge_16` と同じ判定根拠）。
        let expected_unroll_indices = [0usize, 4, 8, 9];
        for (i, cfg) in tile::CANDIDATES.iter().enumerate() {
            assert!(
                !base_gemm.unroll_acc_effective(*cfg),
                "index={i}: base（unroll_acc_enabled=false）は常に false であるべき"
            );
            let expected = expected_unroll_indices.contains(&i);
            assert_eq!(
                head_gemm.unroll_acc_effective(*cfg),
                expected,
                "index={i}: head（unroll_acc_enabled=true）の実効値が acc 積 >= 16 の\
                 候補判定と一致しない"
            );
        }
    }

    /// [`tile::CANDIDATES`] 全 10 候補 × N=512/1024/2048/4096 で、
    /// base（`unroll_acc_enabled=false`）/head（`unroll_acc_enabled=true`）
    /// の `dispatch_tiled_prepared` 出力が bit 単位で一致することを確認する
    /// （イシュー #1282 AC-2）。`resolve_tile_config`（`#[cfg(test)]`）で
    /// フォールバック非経由（候補がサイレントに `SINGLE_SIMDGROUP_8X8` 等へ
    /// 縮退していないこと）も合わせて確認し、検証が空振りしないようにする
    /// （`tests/gemm_fine_barrier_bit_match.rs` と同じ設計判断）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn unroll_acc_on_off_bit_match_all_candidates() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_unroll_acc(&ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_unroll_acc(&ctx, true)
            .expect("head GEMM パイプラインの構築に失敗した");

        const SEED: u64 = 0xC0FFEE;

        for (i, cfg) in tile::CANDIDATES.iter().copied().enumerate() {
            for size in [512usize, 1024, 2048, 4096] {
                let base_resolved = base_gemm
                    .resolve_tile_config(&ctx, cfg)
                    .expect("base 構成の解決に失敗した");
                assert_eq!(
                    base_resolved, cfg,
                    "index={i} size={size}: base 側でフォールバックが発生した（検証が空振りする）"
                );
                let head_resolved = head_gemm
                    .resolve_tile_config(&ctx, cfg)
                    .expect("head 構成の解決に失敗した");
                assert_eq!(
                    head_resolved, cfg,
                    "index={i} size={size}: head 側でフォールバックが発生した（検証が空振りする）"
                );

                let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
                let a = rng.fill_vec(size * size);
                let b = rng.fill_vec(size * size);

                let a_buf = MetalBuffer::new_with_data(&ctx, &a)
                    .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
                let b_buf = MetalBuffer::new_with_data(&ctx, &b)
                    .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
                let base_c_buf = MetalBuffer::new_zeroed(&ctx, size * size)
                    .expect("base C バッファの確保に失敗した（実機でのみ実行する前提）");
                let head_c_buf = MetalBuffer::new_zeroed(&ctx, size * size)
                    .expect("head C バッファの確保に失敗した（実機でのみ実行する前提）");

                base_gemm
                    .dispatch_tiled_prepared(
                        &ctx,
                        &a_buf,
                        &b_buf,
                        &base_c_buf,
                        size,
                        size,
                        size,
                        cfg,
                    )
                    .expect(
                        "base GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                    );
                head_gemm
                    .dispatch_tiled_prepared(
                        &ctx,
                        &a_buf,
                        &b_buf,
                        &head_c_buf,
                        size,
                        size,
                        size,
                        cfg,
                    )
                    .expect(
                        "head GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                    );

                let base_out = base_c_buf.read_to_vec();
                let head_out = head_c_buf.read_to_vec();
                let base_bits: Vec<u32> = base_out.iter().map(|v| v.to_bits()).collect();
                let head_bits: Vec<u32> = head_out.iter().map(|v| v.to_bits()).collect();
                assert_eq!(
                    base_bits, head_bits,
                    "index={i} size={size}: UNROLL_ACC_ENABLED の有無により出力がビット単位で\
                     一致しなかった。演算オペランド列が変わっている疑いがあるため、\
                     shaders/gemm.metal の UNROLL_ACC_ENABLED 分岐箇所を確認すること。"
                );
            }
        }
    }

    /// 本番自動選択経路（`dispatch_auto`）でも base/head の出力が bit 単位で
    /// 一致することを size 512/1024/2048/4096 で確認する（イシュー #1282
    /// AC-2。`tests/gemm_fine_barrier_bit_match.rs::
    /// fine_barrier_on_off_bit_match_dispatch_auto` と同型）。`select_for_
    /// device` が選ぶ候補は現行実測範囲では acc 積 >= 16（index 0/4/8）に
    /// 一致しないため unroll 版ループは通らないが、`UNROLL_ACC_ENABLED`
    /// function constant 特殊化自体が既存の自動選択経路の挙動を変えない
    /// ことを確認する目的（性能実測・本番結線判断は #1284）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn unroll_acc_on_off_bit_match_dispatch_auto() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_unroll_acc(&ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_unroll_acc(&ctx, true)
            .expect("head GEMM パイプラインの構築に失敗した");

        const SEED: u64 = 0xC0FFEE;

        for size in [512usize, 1024, 2048, 4096] {
            let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
            let a = rng.fill_vec(size * size);
            let b = rng.fill_vec(size * size);

            let base_out = base_gemm
                .dispatch_auto(&ctx, &a, &b, size, size, size)
                .expect("base GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
            let head_out = head_gemm
                .dispatch_auto(&ctx, &a, &b, size, size, size)
                .expect("head GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");

            let base_bits: Vec<u32> = base_out.iter().map(|v| v.to_bits()).collect();
            let head_bits: Vec<u32> = head_out.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                base_bits, head_bits,
                "size={size}: dispatch_auto で UNROLL_ACC_ENABLED の有無により出力が\
                 ビット単位で一致しなかった。"
            );
        }
    }

    /// [`tile::CANDIDATES`] 全 10 候補 × N=512/1024/2048/4096 で、
    /// base（function constant 経路。`source_specialized=false`）/head
    /// （ソーステキスト特殊化経路。`source_specialized=true`）の
    /// `dispatch_tiled_prepared` 出力が bit 単位で一致することを確認する
    /// （イシュー #1288 R-3）。`resolve_tile_config`（`#[cfg(test)]`）で
    /// フォールバック非経由（候補がサイレントに `SINGLE_SIMDGROUP_8X8` 等へ
    /// 縮退していないこと）も合わせて確認する
    /// （`unroll_acc_on_off_bit_match_all_candidates` と同型の設計）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn source_specialized_on_off_bit_match_all_candidates() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_source_specialization(&ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_source_specialization(&ctx, true)
            .expect("head GEMM パイプラインの構築に失敗した");

        const SEED: u64 = 0xC0FFEE;

        for (i, cfg) in tile::CANDIDATES.iter().copied().enumerate() {
            for size in [512usize, 1024, 2048, 4096] {
                let base_resolved = base_gemm
                    .resolve_tile_config(&ctx, cfg)
                    .expect("base 構成の解決に失敗した");
                assert_eq!(
                    base_resolved, cfg,
                    "index={i} size={size}: base 側でフォールバックが発生した（検証が空振りする）"
                );
                let head_resolved = head_gemm
                    .resolve_tile_config(&ctx, cfg)
                    .expect("head 構成の解決に失敗した");
                assert_eq!(
                    head_resolved, cfg,
                    "index={i} size={size}: head 側でフォールバックが発生した（検証が空振りする）"
                );

                let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
                let a = rng.fill_vec(size * size);
                let b = rng.fill_vec(size * size);

                let a_buf = MetalBuffer::new_with_data(&ctx, &a)
                    .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
                let b_buf = MetalBuffer::new_with_data(&ctx, &b)
                    .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
                let base_c_buf = MetalBuffer::new_zeroed(&ctx, size * size)
                    .expect("base C バッファの確保に失敗した（実機でのみ実行する前提）");
                let head_c_buf = MetalBuffer::new_zeroed(&ctx, size * size)
                    .expect("head C バッファの確保に失敗した（実機でのみ実行する前提）");

                base_gemm
                    .dispatch_tiled_prepared(
                        &ctx,
                        &a_buf,
                        &b_buf,
                        &base_c_buf,
                        size,
                        size,
                        size,
                        cfg,
                    )
                    .expect(
                        "base GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                    );
                head_gemm
                    .dispatch_tiled_prepared(
                        &ctx,
                        &a_buf,
                        &b_buf,
                        &head_c_buf,
                        size,
                        size,
                        size,
                        cfg,
                    )
                    .expect(
                        "head GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                    );

                let base_out = base_c_buf.read_to_vec();
                let head_out = head_c_buf.read_to_vec();
                let base_bits: Vec<u32> = base_out.iter().map(|v| v.to_bits()).collect();
                let head_bits: Vec<u32> = head_out.iter().map(|v| v.to_bits()).collect();
                assert_eq!(
                    base_bits, head_bits,
                    "index={i} size={size}: ソーステキスト特殊化の有無により出力がビット単位で\
                     一致しなかった。演算オペランド列が変わっている疑いがあるため、\
                     shaders/gemm.metal の GEMM_SPEC_ENABLED 分岐箇所・\
                     crate::spec_source::specialized_gemm_source を確認すること。"
                );
            }
        }
    }

    /// [`Self::source_specialized_on_off_bit_match_all_candidates`] の
    /// 実行後、head（`source_specialized=true`）は
    /// [`MetalGemm::source_specialized_cache_len`] のみが増え
    /// [`MetalGemm::function_constant_cache_len`] は 0 のまま、base は
    /// その逆であることを確認する（イシュー #1288。新経路が実際に走った
    /// ことを出力一致とは独立に証明する。出力一致だけでは両者が同じ経路へ
    /// 倒れた false-green を検出できないため）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn source_specialized_route_populates_only_spec_cache() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_source_specialization(&ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_source_specialization(&ctx, true)
            .expect("head GEMM パイプラインの構築に失敗した");

        assert_eq!(base_gemm.source_specialized_cache_len().unwrap(), 0);
        assert_eq!(base_gemm.function_constant_cache_len().unwrap(), 0);
        assert_eq!(head_gemm.source_specialized_cache_len().unwrap(), 0);
        assert_eq!(head_gemm.function_constant_cache_len().unwrap(), 0);

        let cfg = tile::CANDIDATES[3];
        base_gemm
            .resolve_tile_config(&ctx, cfg)
            .expect("base 構成の解決に失敗した");
        head_gemm
            .resolve_tile_config(&ctx, cfg)
            .expect("head 構成の解決に失敗した");

        assert_eq!(
            base_gemm.source_specialized_cache_len().unwrap(),
            0,
            "base（function constant 経路）が特殊化キャッシュを誤って更新した"
        );
        assert_eq!(
            base_gemm.function_constant_cache_len().unwrap(),
            1,
            "base が function constant キャッシュを更新していない（経路が走っていない疑い）"
        );
        assert_eq!(
            head_gemm.function_constant_cache_len().unwrap(),
            0,
            "head（ソーステキスト特殊化経路）が function constant キャッシュを誤って更新した"
        );
        assert_eq!(
            head_gemm.source_specialized_cache_len().unwrap(),
            1,
            "head が特殊化キャッシュを更新していない（経路が走っていない疑い）"
        );
    }

    /// 本番自動選択経路（`dispatch_auto`）でも base/head の出力が bit 単位で
    /// 一致することを size 512/1024/2048/4096 で確認する（イシュー #1288
    /// R-3。`unroll_acc_on_off_bit_match_dispatch_auto` と同型）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn source_specialized_on_off_bit_match_dispatch_auto() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_source_specialization(&ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_source_specialization(&ctx, true)
            .expect("head GEMM パイプラインの構築に失敗した");

        const SEED: u64 = 0xC0FFEE;

        for size in [512usize, 1024, 2048, 4096] {
            let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
            let a = rng.fill_vec(size * size);
            let b = rng.fill_vec(size * size);

            let base_out = base_gemm
                .dispatch_auto(&ctx, &a, &b, size, size, size)
                .expect("base GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
            let head_out = head_gemm
                .dispatch_auto(&ctx, &a, &b, size, size, size)
                .expect("head GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");

            let base_bits: Vec<u32> = base_out.iter().map(|v| v.to_bits()).collect();
            let head_bits: Vec<u32> = head_out.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                base_bits, head_bits,
                "size={size}: dispatch_auto でソーステキスト特殊化の有無により出力が\
                 ビット単位で一致しなかった。"
            );
        }
    }

    /// [`tile::CANDIDATES`] 全 10 候補 × N=512/1024/2048/4096 で、base
    /// （`tile::FRAG_LOAD_CONFIG`。既定）と head ∈
    /// {tgp-k2〈`{false, Two}`〉, device-hoisted-k1〈`{true, One}`〉,
    /// device-hoisted-k2〈`{true, Two}`〉} の `dispatch_tiled_prepared`
    /// 出力が bit 単位で一致することを確認する（イシュー #1293 AC-2 T1。
    /// `docs/perf/metal-gemm-frag-load-candidates.md` §3.1 候補表）。
    /// staged 候補では `device_hoisted` が no-op であることも含め検証する。
    /// `resolve_tile_config`（`#[cfg(test)]`）でフォールバック非経由
    /// （候補がサイレントに `SINGLE_SIMDGROUP_8X8` 等へ縮退していないこと）
    /// も合わせて確認する（`unroll_acc_on_off_bit_match_all_candidates` と
    /// 同型の設計）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn frag_load_on_off_bit_match_all_candidates() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_frag_load(&ctx, tile::FRAG_LOAD_CONFIG)
            .expect("base GEMM パイプラインの構築に失敗した");
        assert_eq!(base_gemm.frag_load(), tile::FRAG_LOAD_CONFIG);

        let head_configs = [
            tile::FragLoadConfig {
                device_hoisted: false,
                ksteps: tile::FragLoadKSteps::Two,
            },
            tile::FragLoadConfig {
                device_hoisted: true,
                ksteps: tile::FragLoadKSteps::One,
            },
            tile::FragLoadConfig {
                device_hoisted: true,
                ksteps: tile::FragLoadKSteps::Two,
            },
        ];

        const SEED: u64 = 0xC0FFEE;

        for head_cfg in head_configs {
            let head_gemm = MetalGemm::new_with_frag_load(&ctx, head_cfg)
                .expect("head GEMM パイプラインの構築に失敗した");

            for (i, cfg) in tile::CANDIDATES.iter().copied().enumerate() {
                for size in [512usize, 1024, 2048, 4096] {
                    let base_resolved = base_gemm
                        .resolve_tile_config(&ctx, cfg)
                        .expect("base 構成の解決に失敗した");
                    assert_eq!(
                        base_resolved, cfg,
                        "head={head_cfg:?} index={i} size={size}: base 側でフォールバックが\
                         発生した（検証が空振りする）"
                    );
                    let head_resolved = head_gemm
                        .resolve_tile_config(&ctx, cfg)
                        .expect("head 構成の解決に失敗した");
                    assert_eq!(
                        head_resolved, cfg,
                        "head={head_cfg:?} index={i} size={size}: head 側でフォールバックが\
                         発生した（検証が空振りする）"
                    );

                    let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
                    let a = rng.fill_vec(size * size);
                    let b = rng.fill_vec(size * size);

                    let a_buf = MetalBuffer::new_with_data(&ctx, &a)
                        .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
                    let b_buf = MetalBuffer::new_with_data(&ctx, &b)
                        .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
                    let base_c_buf = MetalBuffer::new_zeroed(&ctx, size * size)
                        .expect("base C バッファの確保に失敗した（実機でのみ実行する前提）");
                    let head_c_buf = MetalBuffer::new_zeroed(&ctx, size * size)
                        .expect("head C バッファの確保に失敗した（実機でのみ実行する前提）");

                    base_gemm
                        .dispatch_tiled_prepared(
                            &ctx,
                            &a_buf,
                            &b_buf,
                            &base_c_buf,
                            size,
                            size,
                            size,
                            cfg,
                        )
                        .expect(
                            "base GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                        );
                    head_gemm
                        .dispatch_tiled_prepared(
                            &ctx,
                            &a_buf,
                            &b_buf,
                            &head_c_buf,
                            size,
                            size,
                            size,
                            cfg,
                        )
                        .expect(
                            "head GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                        );

                    let base_out = base_c_buf.read_to_vec();
                    let head_out = head_c_buf.read_to_vec();
                    let base_bits: Vec<u32> = base_out.iter().map(|v| v.to_bits()).collect();
                    let head_bits: Vec<u32> = head_out.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(
                        base_bits, head_bits,
                        "head={head_cfg:?} index={i} size={size}: フラグメントロード方式候補の\
                         違いにより出力がビット単位で一致しなかった。演算オペランド列が\
                         変わっている疑いがあるため、shaders/gemm.metal の\
                         FRAG_LOAD_DEVICE_HOISTED/FRAG_LOAD_KSTEPS 分岐箇所を確認すること。"
                    );
                }
            }
        }
    }

    /// 各 staged 候補（`cfg.staged == true`）について、base 出力
    /// （staged 現行。`tile::FRAG_LOAD_CONFIG`）と `TileConfig { staged:
    /// false, ..cfg }` twin（device-legacy／device-hoisted-k1／
    /// device-hoisted-k2 の 3 head）の出力を同一タイル形状で比較する
    /// （イシュー #1293 AC-2 T2。「threadgroup 経由／device 直接」の
    /// 同一形状対比。兄弟イシュー #1295 が実際に性能比較する軸）。
    /// `validate` が通らない twin（`staged=false` では device 側の
    /// shared_mem/thread 制約が異なりうる）は `continue` し、検証した
    /// 候補数が 0 にならないことをアサートして空振りを検出する。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn frag_load_tgp_vs_device_same_shape_bit_match() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_frag_load(&ctx, tile::FRAG_LOAD_CONFIG)
            .expect("base GEMM パイプラインの構築に失敗した");

        let head_device_configs = [
            tile::FragLoadConfig {
                device_hoisted: false,
                ksteps: tile::FragLoadKSteps::One,
            },
            tile::FragLoadConfig {
                device_hoisted: true,
                ksteps: tile::FragLoadKSteps::One,
            },
            tile::FragLoadConfig {
                device_hoisted: true,
                ksteps: tile::FragLoadKSteps::Two,
            },
        ];

        const SEED: u64 = 0xC0FFEE;
        const SIZE: usize = 1024;

        let max_shared_mem_bytes = ctx.device().maxThreadgroupMemoryLength() as u32;
        // head_cfg ごとの実比較回数（`head_resolved != twin` でスキップされた
        // 分は含めない）。各 head_cfg で最低 1 件は実比較が行われたことを
        // 検査することで、outer cfg ループが空振りしなかった（=候補が
        // 存在した）ことだけでなく、個々の head_cfg も一度も比較されずに
        // 素通りしていないことを保証する（codex-review 指摘対応）。
        let mut verified_per_head = [0usize; 3];

        for cfg in tile::CANDIDATES.iter().copied().filter(|c| c.staged) {
            let twin = TileConfig {
                staged: false,
                ..cfg
            };
            if twin.validate(1024, max_shared_mem_bytes).is_err() {
                continue;
            }

            let base_resolved = base_gemm
                .resolve_tile_config(&ctx, cfg)
                .expect("staged 候補の解決に失敗した");
            if base_resolved != cfg {
                // フォールバックが発生した staged 候補は twin 比較の対象外
                // （空振り防止。他候補で検証を続ける）。
                continue;
            }

            let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
            let a = rng.fill_vec(SIZE * SIZE);
            let b = rng.fill_vec(SIZE * SIZE);
            let a_buf = MetalBuffer::new_with_data(&ctx, &a)
                .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
            let b_buf = MetalBuffer::new_with_data(&ctx, &b)
                .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
            let base_c_buf = MetalBuffer::new_zeroed(&ctx, SIZE * SIZE)
                .expect("base C バッファの確保に失敗した（実機でのみ実行する前提）");
            base_gemm
                .dispatch_tiled_prepared(&ctx, &a_buf, &b_buf, &base_c_buf, SIZE, SIZE, SIZE, cfg)
                .expect("base GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");
            let base_out = base_c_buf.read_to_vec();
            let base_bits: Vec<u32> = base_out.iter().map(|v| v.to_bits()).collect();

            for (head_idx, head_cfg) in head_device_configs.into_iter().enumerate() {
                let head_gemm = MetalGemm::new_with_frag_load(&ctx, head_cfg)
                    .expect("head GEMM パイプラインの構築に失敗した");
                let head_resolved = head_gemm
                    .resolve_tile_config(&ctx, twin)
                    .expect("twin 構成の解決に失敗した");
                if head_resolved != twin {
                    continue;
                }
                let head_c_buf = MetalBuffer::new_zeroed(&ctx, SIZE * SIZE)
                    .expect("head C バッファの確保に失敗した（実機でのみ実行する前提）");
                head_gemm
                    .dispatch_tiled_prepared(
                        &ctx,
                        &a_buf,
                        &b_buf,
                        &head_c_buf,
                        SIZE,
                        SIZE,
                        SIZE,
                        twin,
                    )
                    .expect(
                        "head GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                    );
                let head_out = head_c_buf.read_to_vec();
                let head_bits: Vec<u32> = head_out.iter().map(|v| v.to_bits()).collect();
                assert_eq!(
                    base_bits, head_bits,
                    "cfg={cfg:?} head={head_cfg:?}: staged/device 同一形状対比で出力が\
                     ビット単位で一致しなかった。"
                );
                // head_resolved == twin で実際に比較を行った場合のみ加算する
                // （スキップされたケースでの無条件加算は「比較が一度も
                // 行われなくても検査を通過する」空振りを許してしまうため
                // 不可。codex-review 指摘対応）。
                verified_per_head[head_idx] += 1;
            }
        }

        for (head_idx, head_cfg) in head_device_configs.iter().enumerate() {
            assert!(
                verified_per_head[head_idx] > 0,
                "head_cfg={head_cfg:?}: staged/device 同一形状対比を検証した候補が\
                 0 件だった（検証が空振りした）"
            );
        }
    }

    /// 本番自動選択経路（`dispatch_auto`）でも base/head の出力が bit 単位で
    /// 一致することを N=512/1024/2048/4096 で確認する（イシュー #1293
    /// AC-2 T3。`unroll_acc_on_off_bit_match_dispatch_auto` と同型）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn frag_load_on_off_bit_match_dispatch_auto() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_frag_load(&ctx, tile::FRAG_LOAD_CONFIG)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_frag_load(
            &ctx,
            tile::FragLoadConfig {
                device_hoisted: true,
                ksteps: tile::FragLoadKSteps::Two,
            },
        )
        .expect("head GEMM パイプラインの構築に失敗した");

        const SEED: u64 = 0xC0FFEE;

        for size in [512usize, 1024, 2048, 4096] {
            let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
            let a = rng.fill_vec(size * size);
            let b = rng.fill_vec(size * size);

            let base_out = base_gemm
                .dispatch_auto(&ctx, &a, &b, size, size, size)
                .expect("base GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
            let head_out = head_gemm
                .dispatch_auto(&ctx, &a, &b, size, size, size)
                .expect("head GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");

            let base_bits: Vec<u32> = base_out.iter().map(|v| v.to_bits()).collect();
            let head_bits: Vec<u32> = head_out.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                base_bits, head_bits,
                "size={size}: dispatch_auto でフラグメントロード方式候補の違いにより出力が\
                 ビット単位で一致しなかった。"
            );
        }
    }

    /// NT/TN/TT（`dispatch_strided_tiled_prepared`）を N=1024 で staged
    /// 候補 1 つ以上 + device twin 1 つで比較する（イシュー #1293 AC-2 T4。
    /// `TRANS_A`/`TRANS_B` 分岐を含む新ブロック〈staged k2・direct-load
    /// hoisted〉が転置ロードでも bit 一致することを確認する必須ケース）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn frag_load_transposed_bit_match() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_frag_load(&ctx, tile::FRAG_LOAD_CONFIG)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_frag_load(
            &ctx,
            tile::FragLoadConfig {
                device_hoisted: false,
                ksteps: tile::FragLoadKSteps::Two,
            },
        )
        .expect("head GEMM パイプラインの構築に失敗した");
        let head_device_gemm = MetalGemm::new_with_frag_load(
            &ctx,
            tile::FragLoadConfig {
                device_hoisted: true,
                ksteps: tile::FragLoadKSteps::Two,
            },
        )
        .expect("head（device-hoisted）GEMM パイプラインの構築に失敗した");

        const SEED: u64 = 0xC0FFEE;
        const SIZE: usize = 1024;
        let cfg = tile::CANDIDATES[3];

        for pattern in [
            TransposePattern::Nt,
            TransposePattern::Tn,
            TransposePattern::Tt,
        ] {
            let (trans_a, trans_b) = match pattern {
                TransposePattern::Nt => (false, true),
                TransposePattern::Tn => (true, false),
                TransposePattern::Tt => (true, true),
                TransposePattern::Nn => unreachable!("NN はこのループ対象外"),
            };
            let a_layout = MatrixLayout {
                rows: SIZE,
                cols: SIZE,
                ld: SIZE,
                transposed: trans_a,
            };
            let b_layout = MatrixLayout {
                rows: SIZE,
                cols: SIZE,
                ld: SIZE,
                transposed: trans_b,
            };

            let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
            let a = rng.fill_vec(SIZE * SIZE);
            let b = rng.fill_vec(SIZE * SIZE);
            let a_buf = MetalBuffer::new_with_data(&ctx, &a)
                .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
            let b_buf = MetalBuffer::new_with_data(&ctx, &b)
                .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
            let base_c_buf = MetalBuffer::new_zeroed(&ctx, SIZE * SIZE)
                .expect("base C バッファの確保に失敗した（実機でのみ実行する前提）");
            let head_c_buf = MetalBuffer::new_zeroed(&ctx, SIZE * SIZE)
                .expect("head C バッファの確保に失敗した（実機でのみ実行する前提）");
            let head_device_c_buf = MetalBuffer::new_zeroed(&ctx, SIZE * SIZE).expect(
                "head（device-hoisted）C バッファの確保に失敗した（実機でのみ実行する前提）",
            );

            base_gemm
                .dispatch_strided_tiled_prepared(
                    &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &base_c_buf, SIZE, SIZE, SIZE,
                    cfg,
                )
                .expect(
                    "base GEMM dispatch_strided_tiled_prepared に失敗した（実機でのみ実行する前提）",
                );
            head_gemm
                .dispatch_strided_tiled_prepared(
                    &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &head_c_buf, SIZE, SIZE, SIZE,
                    cfg,
                )
                .expect(
                    "head GEMM dispatch_strided_tiled_prepared に失敗した（実機でのみ実行する前提）",
                );
            let device_twin = TileConfig {
                staged: false,
                ..cfg
            };
            head_device_gemm
                .dispatch_strided_tiled_prepared(
                    &ctx,
                    &a_buf,
                    0,
                    a_layout,
                    &b_buf,
                    0,
                    b_layout,
                    &head_device_c_buf,
                    SIZE,
                    SIZE,
                    SIZE,
                    device_twin,
                )
                .expect(
                    "head（device-hoisted）GEMM dispatch_strided_tiled_prepared に失敗した\
                     （実機でのみ実行する前提）",
                );

            let base_bits: Vec<u32> = base_c_buf
                .read_to_vec()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let head_bits: Vec<u32> = head_c_buf
                .read_to_vec()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let head_device_bits: Vec<u32> = head_device_c_buf
                .read_to_vec()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(
                base_bits, head_bits,
                "pattern={pattern:?}: staged K=2 ブロックの転置ロードで出力がビット単位で\
                 一致しなかった。"
            );
            assert_eq!(
                base_bits, head_device_bits,
                "pattern={pattern:?}: device-hoisted ブロックの転置ロードで出力がビット単位で\
                 一致しなかった。"
            );
        }
    }

    // --- イシュー #1298: 協調ロードレイアウト候補（`tile::CoopLoadConfig`）の
    //     bit 一致自己検証（T1〜T6。計画「2.5 実機 #[ignore] テスト」節） ---

    /// T1〜T5 が対象とする必須 5 候補（`docs/perf/
    /// metal-gemm-coop-load-candidates.md` §1 候補表。base
    /// `tile::COOP_LOAD_CONFIG`＝L0-P4 は別途保持する）。
    fn coop_load_required_heads() -> [tile::CoopLoadConfig; 5] {
        [
            tile::CoopLoadConfig {
                layout: tile::CoopLoadLayout::RowLinear,
                pad: tile::TgpPad::Zero,
            }, // L0-P0
            tile::CoopLoadConfig {
                layout: tile::CoopLoadLayout::RowLinear,
                pad: tile::TgpPad::Eight,
            }, // L0-P8
            tile::CoopLoadConfig {
                layout: tile::CoopLoadLayout::RowStrided,
                pad: tile::TgpPad::Zero,
            }, // L1-P0
            tile::CoopLoadConfig {
                layout: tile::CoopLoadLayout::RowStrided,
                pad: tile::TgpPad::Four,
            }, // L1-P4
            tile::CoopLoadConfig {
                layout: tile::CoopLoadLayout::RowStrided,
                pad: tile::TgpPad::Eight,
            }, // L1-P8
        ]
    }

    /// T1: [`tile::CANDIDATES`] 全 10 候補 × N∈{512,1024,2048,4096} ×
    /// 必須 5 head で `dispatch_tiled_prepared` 出力が bit 単位で一致する
    /// ことを確認する（イシュー #1298。`frag_load_on_off_bit_match_all_
    /// candidates` と同型の設計）。`resolve_tile_config` でフォールバック
    /// 非経由（検証の空振り防止）も base/head 双方で確認する。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn coop_load_bit_match_all_candidates() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_coop_load(&ctx, tile::COOP_LOAD_CONFIG)
            .expect("base GEMM パイプラインの構築に失敗した");
        assert_eq!(base_gemm.coop_load(), tile::COOP_LOAD_CONFIG);

        const SEED: u64 = 0xC0FFEE;

        for head_cfg in coop_load_required_heads() {
            let head_gemm = MetalGemm::new_with_coop_load(&ctx, head_cfg)
                .expect("head GEMM パイプラインの構築に失敗した");

            for (i, cfg) in tile::CANDIDATES.iter().copied().enumerate() {
                for size in [512usize, 1024, 2048, 4096] {
                    let base_resolved = base_gemm
                        .resolve_tile_config(&ctx, cfg)
                        .expect("base 構成の解決に失敗した");
                    assert_eq!(
                        base_resolved, cfg,
                        "head={head_cfg:?} index={i} size={size}: base 側でフォールバックが\
                         発生した（検証が空振りする）"
                    );
                    let head_resolved = head_gemm
                        .resolve_tile_config(&ctx, cfg)
                        .expect("head 構成の解決に失敗した");
                    assert_eq!(
                        head_resolved, cfg,
                        "head={head_cfg:?} index={i} size={size}: head 側でフォールバックが\
                         発生した（検証が空振りする）"
                    );

                    let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
                    let a = rng.fill_vec(size * size);
                    let b = rng.fill_vec(size * size);

                    let a_buf = MetalBuffer::new_with_data(&ctx, &a)
                        .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
                    let b_buf = MetalBuffer::new_with_data(&ctx, &b)
                        .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
                    let base_c_buf = MetalBuffer::new_zeroed(&ctx, size * size)
                        .expect("base C バッファの確保に失敗した（実機でのみ実行する前提）");
                    let head_c_buf = MetalBuffer::new_zeroed(&ctx, size * size)
                        .expect("head C バッファの確保に失敗した（実機でのみ実行する前提）");

                    base_gemm
                        .dispatch_tiled_prepared(
                            &ctx,
                            &a_buf,
                            &b_buf,
                            &base_c_buf,
                            size,
                            size,
                            size,
                            cfg,
                        )
                        .expect(
                            "base GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                        );
                    head_gemm
                        .dispatch_tiled_prepared(
                            &ctx,
                            &a_buf,
                            &b_buf,
                            &head_c_buf,
                            size,
                            size,
                            size,
                            cfg,
                        )
                        .expect(
                            "head GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                        );

                    let base_out = base_c_buf.read_to_vec();
                    let head_out = head_c_buf.read_to_vec();
                    let base_bits: Vec<u32> = base_out.iter().map(|v| v.to_bits()).collect();
                    let head_bits: Vec<u32> = head_out.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(
                        base_bits, head_bits,
                        "head={head_cfg:?} index={i} size={size}: 協調ロードレイアウト候補の\
                         違いにより出力がビット単位で一致しなかった。演算オペランド列が\
                         変わっている疑いがあるため、shaders/gemm.metal の\
                         COOP_LOAD_LAYOUT/TGP_PAD 分岐箇所を確認すること。"
                    );
                }
            }
        }
    }

    /// T2: 本番自動選択経路 `dispatch_auto` で N=512〜4096 × 必須 5 head の
    /// bit 一致を確認する（イシュー #1298）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn coop_load_bit_match_dispatch_auto() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_coop_load(&ctx, tile::COOP_LOAD_CONFIG)
            .expect("base GEMM パイプラインの構築に失敗した");

        const SEED: u64 = 0xC0FFEE;

        for head_cfg in coop_load_required_heads() {
            let head_gemm = MetalGemm::new_with_coop_load(&ctx, head_cfg)
                .expect("head GEMM パイプラインの構築に失敗した");

            for size in [512usize, 1024, 2048, 4096] {
                let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
                let a = rng.fill_vec(size * size);
                let b = rng.fill_vec(size * size);

                let base_out = base_gemm
                    .dispatch_auto(&ctx, &a, &b, size, size, size)
                    .expect("base GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
                let head_out = head_gemm
                    .dispatch_auto(&ctx, &a, &b, size, size, size)
                    .expect("head GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");

                let base_bits: Vec<u32> = base_out.iter().map(|v| v.to_bits()).collect();
                let head_bits: Vec<u32> = head_out.iter().map(|v| v.to_bits()).collect();
                assert_eq!(
                    base_bits, head_bits,
                    "head={head_cfg:?} size={size}: dispatch_auto で協調ロードレイアウト候補の\
                     違いにより出力がビット単位で一致しなかった。"
                );
            }
        }
    }

    /// T3: NT/TN/TT（`dispatch_strided_tiled_prepared`）を N=1024・
    /// `CANDIDATES[3]`／`CANDIDATES[5]`（bk=32）× 必須 5 head で確認する
    /// （イシュー #1298。`TRANS_A`/`TRANS_B` 分岐は `(rows,row_len)` が
    /// 入れ替わり `lda = BM+TGP_PAD` も変わるため必須。
    /// `frag_load_transposed_bit_match` と同型の設計）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn coop_load_transposed_bit_match() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_coop_load(&ctx, tile::COOP_LOAD_CONFIG)
            .expect("base GEMM パイプラインの構築に失敗した");

        const SEED: u64 = 0xC0FFEE;
        const SIZE: usize = 1024;

        for cfg in [tile::CANDIDATES[3], tile::CANDIDATES[5]] {
            for head_cfg in coop_load_required_heads() {
                let head_gemm = MetalGemm::new_with_coop_load(&ctx, head_cfg)
                    .expect("head GEMM パイプラインの構築に失敗した");

                for pattern in [
                    TransposePattern::Nt,
                    TransposePattern::Tn,
                    TransposePattern::Tt,
                ] {
                    let (trans_a, trans_b) = match pattern {
                        TransposePattern::Nt => (false, true),
                        TransposePattern::Tn => (true, false),
                        TransposePattern::Tt => (true, true),
                        TransposePattern::Nn => unreachable!("NN はこのループ対象外"),
                    };
                    let a_layout = MatrixLayout {
                        rows: SIZE,
                        cols: SIZE,
                        ld: SIZE,
                        transposed: trans_a,
                    };
                    let b_layout = MatrixLayout {
                        rows: SIZE,
                        cols: SIZE,
                        ld: SIZE,
                        transposed: trans_b,
                    };

                    let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
                    let a = rng.fill_vec(SIZE * SIZE);
                    let b = rng.fill_vec(SIZE * SIZE);
                    let a_buf = MetalBuffer::new_with_data(&ctx, &a)
                        .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
                    let b_buf = MetalBuffer::new_with_data(&ctx, &b)
                        .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
                    let base_c_buf = MetalBuffer::new_zeroed(&ctx, SIZE * SIZE)
                        .expect("base C バッファの確保に失敗した（実機でのみ実行する前提）");
                    let head_c_buf = MetalBuffer::new_zeroed(&ctx, SIZE * SIZE)
                        .expect("head C バッファの確保に失敗した（実機でのみ実行する前提）");

                    base_gemm
                        .dispatch_strided_tiled_prepared(
                            &ctx,
                            &a_buf,
                            0,
                            a_layout,
                            &b_buf,
                            0,
                            b_layout,
                            &base_c_buf,
                            SIZE,
                            SIZE,
                            SIZE,
                            cfg,
                        )
                        .expect(
                            "base GEMM dispatch_strided_tiled_prepared に失敗した\
                             （実機でのみ実行する前提）",
                        );
                    head_gemm
                        .dispatch_strided_tiled_prepared(
                            &ctx,
                            &a_buf,
                            0,
                            a_layout,
                            &b_buf,
                            0,
                            b_layout,
                            &head_c_buf,
                            SIZE,
                            SIZE,
                            SIZE,
                            cfg,
                        )
                        .expect(
                            "head GEMM dispatch_strided_tiled_prepared に失敗した\
                             （実機でのみ実行する前提）",
                        );

                    let base_bits: Vec<u32> = base_c_buf
                        .read_to_vec()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect();
                    let head_bits: Vec<u32> = head_c_buf
                        .read_to_vec()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect();
                    assert_eq!(
                        base_bits, head_bits,
                        "cfg={cfg:?} head={head_cfg:?} pattern={pattern:?}: 転置ロードで協調\
                         ロードレイアウト候補の違いにより出力がビット単位で一致しなかった。"
                    );
                }
            }
        }
    }

    /// T4: 端数形状（M=1032・N=1048・K=1032。8 の倍数だが `BM`/`BN`/`BK`
    /// の倍数でないため末尾タイル・部分ブロックのスカラーフォールバック
    /// 経路を通る。正方 N=512〜4096 では到達しない経路）で
    /// `dispatch_tiled_prepared` × 全候補 × 必須 5 head の bit 一致を
    /// 確認する（イシュー #1298）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn coop_load_bit_match_boundary_shape() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_coop_load(&ctx, tile::COOP_LOAD_CONFIG)
            .expect("base GEMM パイプラインの構築に失敗した");

        const SEED: u64 = 0xC0FFEE;
        const M: usize = 1032;
        const N: usize = 1048;
        const K: usize = 1032;

        for head_cfg in coop_load_required_heads() {
            let head_gemm = MetalGemm::new_with_coop_load(&ctx, head_cfg)
                .expect("head GEMM パイプラインの構築に失敗した");

            for cfg in tile::CANDIDATES.iter().copied() {
                let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
                let a = rng.fill_vec(M * K);
                let b = rng.fill_vec(K * N);

                let a_buf = MetalBuffer::new_with_data(&ctx, &a)
                    .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
                let b_buf = MetalBuffer::new_with_data(&ctx, &b)
                    .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
                let base_c_buf = MetalBuffer::new_zeroed(&ctx, M * N)
                    .expect("base C バッファの確保に失敗した（実機でのみ実行する前提）");
                let head_c_buf = MetalBuffer::new_zeroed(&ctx, M * N)
                    .expect("head C バッファの確保に失敗した（実機でのみ実行する前提）");

                base_gemm
                    .dispatch_tiled_prepared(&ctx, &a_buf, &b_buf, &base_c_buf, M, N, K, cfg)
                    .expect(
                        "base GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                    );
                head_gemm
                    .dispatch_tiled_prepared(&ctx, &a_buf, &b_buf, &head_c_buf, M, N, K, cfg)
                    .expect(
                        "head GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                    );

                let base_bits: Vec<u32> = base_c_buf
                    .read_to_vec()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                let head_bits: Vec<u32> = head_c_buf
                    .read_to_vec()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                assert_eq!(
                    base_bits, head_bits,
                    "cfg={cfg:?} head={head_cfg:?}: 端数形状（M={M} N={N} K={K}）で協調ロード\
                     レイアウト候補の違いにより出力がビット単位で一致しなかった。"
                );
            }
        }
    }

    /// T5: `gemm_simdgroup_tiled_f16` は `COOP_LOAD_LAYOUT` を参照しない
    /// no-op 契約であることを、base（`tile::COOP_LOAD_CONFIG`）と head
    /// （L1-P8）が `dispatch_f16_tiled_prepared_unverified` で bit 一致
    /// することにより実機証明する（イシュー #1298。`pipeline_for_tile_f16`
    /// が常に `coop_load_layout: 0` を渡す契約〈`gemm.rs` 呼び出し側
    /// コメント参照〉の裏付け）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn coop_load_f16_path_is_noop() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_coop_load(&ctx, tile::COOP_LOAD_CONFIG)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_coop_load(
            &ctx,
            tile::CoopLoadConfig {
                layout: tile::CoopLoadLayout::RowStrided,
                pad: tile::TgpPad::Eight,
            },
        )
        .expect("head GEMM パイプラインの構築に失敗した");

        const SEED: u64 = 0xC0FFEE;
        const SIZE: usize = 1024;
        let cfg = tile::CANDIDATES[3];

        let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
        let a: Vec<half::f16> = rng.fill_vec_f16(SIZE * SIZE);
        let b: Vec<half::f16> = rng.fill_vec_f16(SIZE * SIZE);

        let a_buf = MetalHalfBuffer::new_with_data(&ctx, &a)
            .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
        let b_buf = MetalHalfBuffer::new_with_data(&ctx, &b)
            .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
        let base_c_buf = MetalHalfBuffer::new_zeroed(&ctx, SIZE * SIZE)
            .expect("base C バッファの確保に失敗した（実機でのみ実行する前提）");
        let head_c_buf = MetalHalfBuffer::new_zeroed(&ctx, SIZE * SIZE)
            .expect("head C バッファの確保に失敗した（実機でのみ実行する前提）");

        base_gemm
            .dispatch_f16_tiled_prepared_unverified(
                &ctx,
                &a_buf,
                &b_buf,
                &base_c_buf,
                SIZE,
                SIZE,
                SIZE,
                cfg,
            )
            .expect(
                "base GEMM dispatch_f16_tiled_prepared_unverified に失敗した\
                 （実機でのみ実行する前提）",
            );
        head_gemm
            .dispatch_f16_tiled_prepared_unverified(
                &ctx,
                &a_buf,
                &b_buf,
                &head_c_buf,
                SIZE,
                SIZE,
                SIZE,
                cfg,
            )
            .expect(
                "head GEMM dispatch_f16_tiled_prepared_unverified に失敗した\
                 （実機でのみ実行する前提）",
            );

        let base_bits: Vec<u16> = base_c_buf
            .read_to_vec()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let head_bits: Vec<u16> = head_c_buf
            .read_to_vec()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(
            base_bits, head_bits,
            "f16 経路（gemm_simdgroup_tiled_f16）が COOP_LOAD_LAYOUT を参照してしまっている\
             疑いがある（no-op 契約違反）。"
        );
    }

    /// T6: [`tile::COOP_LOAD_CONFIG`]（本番既定）が
    /// [`tile::CoopLoadConfig::DEFAULT`]（`RowLinear`・`Four`）と一致し、
    /// `MetalGemm::new` が構築するインスタンスの `coop_load()` も同じ値で
    /// あることを実機上で固定する（イシュー #1298。Linux 実行可能な部分は
    /// `tile::tests::coop_load_config_default_is_current_path`）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn coop_load_default_matches_production_constants() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");
        assert_eq!(gemm.coop_load(), tile::COOP_LOAD_CONFIG);
        assert_eq!(tile::COOP_LOAD_CONFIG, tile::CoopLoadConfig::DEFAULT);
    }

    // --- タイルクラス分割（イシュー #1327・E6 試作） ---

    /// T1: [`tile::CANDIDATES`] 全 10 候補 × N∈{512,1024,2048,4096} で
    /// base（`TileClassMode::Legacy`）/head（`TileClassMode::Split`）の
    /// `dispatch_tiled_prepared` 出力が bit 単位で一致することを確認する
    /// （`coop_load_bit_match_all_candidates` と同型の設計）。staged 候補
    /// （`CANDIDATES[7]`＝`SINGLE_SIMDGROUP_8X8`。`staged=false`）は
    /// Interior/Edge どちらも direct-load へ縮退し `tile_class_plan` の
    /// interior が構造上 0 面積になりうる形状もあるため、bit 一致のみを
    /// 検証し空振り検査（インターカウンタ増加の assert）は他候補に限定
    /// する。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn tile_class_split_bit_match_all_candidates() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_tile_class(&ctx, tile::TileClassMode::Legacy)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_tile_class(&ctx, tile::TileClassMode::Split)
            .expect("head GEMM パイプラインの構築に失敗した");
        assert_eq!(base_gemm.tile_class_mode(), tile::TileClassMode::Legacy);
        assert_eq!(head_gemm.tile_class_mode(), tile::TileClassMode::Split);

        const SEED: u64 = 0x1327C1A55;

        for (i, cfg) in tile::CANDIDATES.iter().copied().enumerate() {
            for size in [512usize, 1024, 2048, 4096] {
                let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
                let a = rng.fill_vec(size * size);
                let b = rng.fill_vec(size * size);

                let a_buf = MetalBuffer::new_with_data(&ctx, &a)
                    .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
                let b_buf = MetalBuffer::new_with_data(&ctx, &b)
                    .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
                let base_c_buf = MetalBuffer::new_zeroed(&ctx, size * size)
                    .expect("base C バッファの確保に失敗した（実機でのみ実行する前提）");
                let head_c_buf = MetalBuffer::new_zeroed(&ctx, size * size)
                    .expect("head C バッファの確保に失敗した（実機でのみ実行する前提）");

                let interior_before = TILE_CLASS_INTERIOR_DISPATCH_COUNT.with(|c| c.get());
                let fallback_before = TILE_CLASS_SPLIT_FALLBACK_COUNT.with(|c| c.get());

                base_gemm
                    .dispatch_tiled_prepared(
                        &ctx,
                        &a_buf,
                        &b_buf,
                        &base_c_buf,
                        size,
                        size,
                        size,
                        cfg,
                    )
                    .expect(
                        "base GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                    );
                head_gemm
                    .dispatch_tiled_prepared(
                        &ctx,
                        &a_buf,
                        &b_buf,
                        &head_c_buf,
                        size,
                        size,
                        size,
                        cfg,
                    )
                    .expect(
                        "head GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                    );

                let fallback_after = TILE_CLASS_SPLIT_FALLBACK_COUNT.with(|c| c.get());
                assert_eq!(
                    fallback_before, fallback_after,
                    "index={i} size={size}: Split 経路が Legacy フォールバックへ\
                     縮退した（Interior/Edge の解決構成が食い違っている疑い）"
                );

                if cfg.staged {
                    let interior_after = TILE_CLASS_INTERIOR_DISPATCH_COUNT.with(|c| c.get());
                    assert!(
                        interior_after > interior_before,
                        "index={i} size={size}: staged 候補で内部タイル（Interior）dispatch\
                         が 1 回も起動しなかった（空振り検査）"
                    );
                }

                let base_bits: Vec<u32> = base_c_buf
                    .read_to_vec()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                let head_bits: Vec<u32> = head_c_buf
                    .read_to_vec()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                assert_eq!(
                    base_bits, head_bits,
                    "index={i} size={size}: タイルクラス分割（Split）の有無により出力が\
                     ビット単位で一致しなかった。演算オペランド列が変わっている疑いが\
                     あるため、shaders/gemm.metal の TILE_CLASS 分岐箇所を確認すること。"
                );
            }
        }
    }

    /// T2: 端あり形状（8 の倍数だが `BM`/`BN`/`BK` の倍数でない、または
    /// `K` が `BK` の倍数でない）× 全候補で base/head の bit 一致を
    /// 確認する（イシュー #1327）。形状ごとの領域構成（interior 有無・
    /// edge 数）を `tile::tile_class_plan` から事前に求め、実際の
    /// dispatch カウンタ差分と整合することも確認する（空振り検査）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn tile_class_split_bit_match_edge_shapes() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_tile_class(&ctx, tile::TileClassMode::Legacy)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_tile_class(&ctx, tile::TileClassMode::Split)
            .expect("head GEMM パイプラインの構築に失敗した");

        const SEED: u64 = 0x1327ED6E;
        let shapes: &[(usize, usize, usize)] = &[
            (1032, 1048, 1032),
            (1040, 1056, 1024),
            (1032, 1024, 1024),
            (1024, 1032, 1024),
            (8, 8, 8),
            (64, 64, 4096),
        ];

        for (i, cfg) in tile::CANDIDATES.iter().copied().enumerate() {
            for &(m, n, k) in shapes {
                let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
                let a = rng.fill_vec(m * k);
                let b = rng.fill_vec(k * n);

                let a_buf = MetalBuffer::new_with_data(&ctx, &a)
                    .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
                let b_buf = MetalBuffer::new_with_data(&ctx, &b)
                    .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
                let base_c_buf = MetalBuffer::new_zeroed(&ctx, m * n)
                    .expect("base C バッファの確保に失敗した（実機でのみ実行する前提）");
                let head_c_buf = MetalBuffer::new_zeroed(&ctx, m * n)
                    .expect("head C バッファの確保に失敗した（実機でのみ実行する前提）");

                let base_resolved = base_gemm
                    .dispatch_tiled_prepared(&ctx, &a_buf, &b_buf, &base_c_buf, m, n, k, cfg)
                    .expect(
                        "base GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                    );
                head_gemm
                    .dispatch_tiled_prepared(&ctx, &a_buf, &b_buf, &head_c_buf, m, n, k, cfg)
                    .expect(
                        "head GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）",
                    );

                let plan = tile::tile_class_plan(m as u32, n as u32, k as u32, base_resolved);
                if plan.interior.is_none() {
                    assert!(
                        m % (base_resolved.bm as usize) != 0
                            || n % (base_resolved.bn as usize) != 0
                            || k % (base_resolved.bk as usize) != 0,
                        "index={i} shape=({m},{n},{k}): interior が None なのに完全整列\
                         している（tile_class_plan の分割規則が想定と食い違う）"
                    );
                }

                let base_bits: Vec<u32> = base_c_buf
                    .read_to_vec()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                let head_bits: Vec<u32> = head_c_buf
                    .read_to_vec()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                assert_eq!(
                    base_bits, head_bits,
                    "index={i} shape=({m},{n},{k}): 端あり形状でタイルクラス分割の有無に\
                     より出力がビット単位で一致しなかった。"
                );
            }
        }
    }

    /// T3: 本番自動選択経路 `dispatch_auto` で N=512〜4096 の base/head
    /// bit 一致を確認する（イシュー #1327）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn tile_class_split_bit_match_dispatch_auto() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let base_gemm = MetalGemm::new_with_tile_class(&ctx, tile::TileClassMode::Legacy)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_tile_class(&ctx, tile::TileClassMode::Split)
            .expect("head GEMM パイプラインの構築に失敗した");

        const SEED: u64 = 0x1327A0A0;

        for size in [512usize, 1024, 2048, 4096] {
            let mut rng = bench_harness::rng::Xorshift64Star::new(SEED);
            let a = rng.fill_vec(size * size);
            let b = rng.fill_vec(size * size);

            let base_out = base_gemm
                .dispatch_auto(&ctx, &a, &b, size, size, size)
                .expect("base GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
            let head_out = head_gemm
                .dispatch_auto(&ctx, &a, &b, size, size, size)
                .expect("head GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");

            let base_bits: Vec<u32> = base_out.iter().map(|v| v.to_bits()).collect();
            let head_bits: Vec<u32> = head_out.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                base_bits, head_bits,
                "size={size}: dispatch_auto 経路でタイルクラス分割の有無により出力が\
                 ビット単位で一致しなかった。"
            );
        }
    }

    /// T6: f16 経路（`gemm_simdgroup_tiled_f16`）が `TileClassMode` に
    /// 関わらず bit 一致する（no-op 契約の裏付け）ことと、
    /// [`tile::TILE_CLASS_MODE`]（本番既定）が `Legacy` であり
    /// `MetalGemm::new` が構築するインスタンスの `tile_class_mode()` も
    /// 同じ値であることを実機上で固定する（イシュー #1327）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn tile_class_default_matches_production_constants() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");
        assert_eq!(gemm.tile_class_mode(), tile::TILE_CLASS_MODE);
        assert_eq!(tile::TILE_CLASS_MODE, tile::TileClassMode::Legacy);
    }
}
