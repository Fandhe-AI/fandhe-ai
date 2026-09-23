//! Metal コンテキスト／カーネルスイートのプロセス内キャッシュ
//! （イシュー #930。診断 #927・`docs/perf/metal-fixed-overhead-diagnosis.md`
//! が特定した「演算メソッド呼び出しごとの Metal 資源都度構築」固定
//! オーバーヘッド〈約 5 ms・N 非依存〉の解消）。
//!
//! CUDA 側 [`crate::context_cache`] 相当のプロセスワイド static キャッシュ
//! （`crates/backend-cuda/src/context_cache.rs`。イシュー #929）と同型の
//! 設計を Metal 側に揃える。ただし CUDA 側は `Device::Cuda(ordinal)` が
//! 複数 GPU を ordinal で区別するのに対し、`Device::Metal` は ordinal を
//! 持たない単一 variant（`docs/public-api-design.md` §4.1・
//! `crate::device::MetalDeviceProvider` と同じ位置付け）のため、本モジュールは
//! `HashMap<usize, Arc<T>>` ではなく型ごとに単一エントリを持つ
//! `OnceLock<Mutex<Option<Arc<T>>>>` を使う（システムデフォルトの Metal
//! デバイス 1 台のみを扱う前提。`crate::ops::MetalBackendOps` の
//! ドキュメンテーションコメント参照）。
//!
//! # 何をキャッシュするか
//!
//! デバイス層は [`cached_context`] が [`crate::context::MetalContext`]
//! （`MTLCreateSystemDefaultDevice` + `newCommandQueue` + `supportsFamily` +
//! IOKit occupancy probe を内包する。`context.rs` 参照）をプロセスワイドに
//! 共有する。
//!
//! カーネルスイート層は [`cached_gemm`]／[`cached_elementwise`]／
//! [`cached_rmsnorm`]／[`cached_softmax`] が、それぞれ
//! [`crate::gemm::MetalGemm`]（本番既定構成〈`MetalGemm::new` 相当〉のみを
//! キャッシュし、A/B ベンチ用 `new_with_swizzle`／`new_with_fine_barrier`
//! 入口はキャッシュ対象外で従来どおり直接構築する）／
//! [`crate::elementwise::MetalElementwise`]／[`crate::rmsnorm::MetalRmsNorm`]／
//! [`crate::softmax::MetalSoftmax`]（いずれも `new` 内で MSL 実行時
//! コンパイル + 複数パイプライン構築を行う。`ops.rs` 冒頭コメント参照）を
//! 共有する。
//!
//! `ops::MetalBackendOps` は演算メソッド呼び出しごとにこれらを都度構築
//! していた（イシュー #930 実装計画 §3.4）。本モジュールを経由させる
//! ことで、同一プロセス内の 2 回目以降の呼び出しはデバイス取得・MSL
//! コンパイルを再度支払わない。
//!
//! # fail-fast 契約（エラーはキャッシュしない）
//!
//! ミス時の構築が失敗した場合（Metal デバイス不在・MSL コンパイル失敗等）、
//! その `Err` はキャッシュへ格納しない。次回呼び出しは再度構築を試みる
//! （CUDA 側 `context_cache.rs` と同じ fail-fast 契約。`ops.rs::gemm` の
//! `DeviceUnavailable` 分類契約〈PR #262 レビュー対応〉を不変に保つ）。
//!
//! # 生存期間
//!
//! 各キャッシュはプロセスの生存期間中 evict されない（デバイス 1 台 ×
//! スイート数個の有界エントリのみを扱うため、`crate::gemm::MetalGemm::
//! tiled_cache`〈shape 特化コンパイルキャッシュ〉と異なり容量制御・LRU は
//! 不要。CUDA 側 `context_cache.rs` と同じ判断）。`MetalGemm`／
//! `MetalElementwise`／`MetalRmsNorm`／`MetalSoftmax` の `Arc` はいずれも
//! 内部で `Retained<MtlDevice>`／`Retained<MtlQueue>`（`MetalContext`
//! 経由）を強参照するため、スイートキャッシュのエントリが 1 つでも生存
//! する限り対応する `MetalContext` は解放されない。
//!
//! # `Mutex` poison
//!
//! `gemm.rs::lock_tile_cache` と同じ方針で、`Mutex` の poison を
//! [`MetalError::ContextCacheUnavailable`] へ変換し panic させない
//! （本モジュールの臨界区間自体は `unwrap`/`expect` を持たないため通常
//! 到達しない）。呼び出し元（`ops::MetalBackendOps`）はこのエラーを
//! そのまま `BackendError` へ伝播してよい（本キャッシュは純粋な最適化
//! ではあるが、ロック不能は環境異常を示すため型付きエラーとして呼び出し
//! 元へ伝える。キャッシュなしへの縮退運転は行わない — 縮退させると
//! 「毎回フレッシュ構築」に戻り受け入れ条件 1〈2 回目以降が構築費を
//! 支払わない〉自体が崩れるため）。

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use fandhe_ai_tensor_core::{ScalarBinaryOp, ScalarOpKind, ScalarUnaryOp};

use crate::adam::MetalAdam;
use crate::batch_norm::MetalBatchNorm;
use crate::constant_pad::MetalConstantPad;
use crate::context::MetalContext;
use crate::elementwise::MetalElementwise;
use crate::error::MetalError;
use crate::gather_scatter::MetalGatherScatter;
use crate::gemm::MetalGemm;
use crate::generic_cache::{get_or_build, get_or_build_keyed};
use crate::im2col::MetalIm2col;
use crate::interpolate::MetalInterpolate;
use crate::layer_norm::MetalLayerNorm;
use crate::norm_backward::MetalNormBackward;
use crate::pipeline::{self, MtlPipeline};
use crate::pool::MetalAllocator;
use crate::rmsnorm::MetalRmsNorm;
use crate::sgd::MetalSgd;
use crate::softmax::MetalSoftmax;

/// poison 時のエラー変換（`crate::generic_cache::get_or_build` へ注入する
/// Metal 固有クロージャ）。`gemm.rs::lock_tile_cache` と同じ変換方針
/// （panic 経路を持たない `.claude/rules/coding-rust.md`）。
fn on_poison(detail: String) -> MetalError {
    MetalError::ContextCacheUnavailable { detail }
}

/// コンパイル時アサーション: 本モジュールがキャッシュする全ハンドル型が
/// `Send + Sync` であることを固定する。`OnceLock<Mutex<Option<Arc<T>>>>`
/// static 経由で複数スレッドから共有する前提（`Arc<T>` を他スレッドへ渡す・
/// 複数スレッドから同時に `&T` で参照する）が成立するには `T: Send + Sync`
/// が必須。objc2-metal 0.3.2 では `MTLDevice`／`MTLCommandQueue`／
/// `MTLLibrary`／`MTLComputePipelineState` の各 protocol が `Send + Sync`
/// を supertrait に持つため、それらを `Retained<ProtocolObject<dyn _>>`
/// で保持するのみの各ハンドル型は成立するはずだが、将来のフィールド
/// 追加でこの前提が崩れた場合にここでコンパイルエラーとして検出する
/// （CUDA 側 `context_cache.rs` の同名アサーションと同じ設計判断）。
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MetalContext>();
    assert_send_sync::<MetalAdam>();
    assert_send_sync::<MetalGemm>();
    assert_send_sync::<MetalElementwise>();
    assert_send_sync::<MetalRmsNorm>();
    assert_send_sync::<MetalLayerNorm>();
    assert_send_sync::<MetalBatchNorm>();
    assert_send_sync::<MetalSoftmax>();
    assert_send_sync::<MetalAllocator>();
    assert_send_sync::<MetalGatherScatter>();
    assert_send_sync::<MetalInterpolate>();
    assert_send_sync::<MetalConstantPad>();
    // イシュー #1707: `cached_scalar_unary_pipeline`／
    // `cached_scalar_binary_pipeline` が `Mutex<HashMap<&'static str,
    // Retained<MtlPipeline>>>` として直接キャッシュする値の Send/Sync
    // をコンパイル時に固定する（既存エントリと同じ意図。CUDA 側
    // `context_cache.rs` の `CudaFunction` アサーションと同型。
    // `MtlPipeline`＝`ProtocolObject<dyn MTLComputePipelineState>` は
    // 非 `Sized` のため `Retained<MtlPipeline>`〈参照カウントハンドル。
    // キャッシュが実際に保持する値の型〉で検査する）。
    assert_send_sync::<objc2::rc::Retained<MtlPipeline>>();
};

/// システムデフォルトの Metal デバイスに対応する [`MetalContext`] を
/// プロセス内キャッシュから取得する。ヒット時は `MTLCreateSystemDefaultDevice`
/// ／`newCommandQueue`／occupancy probe を再実行しない（受け入れ条件 1）。
///
/// `ops::MetalBackendOps` の各演算メソッドの唯一の呼び出し先とする。
pub(crate) fn cached_context() -> Result<Arc<MetalContext>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalContext>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, MetalContext::new)
}

/// 本番既定構成の [`MetalGemm`]（`MetalGemm::new` 相当。threadgroup ID
/// スウィズル・simdgroup 細粒度同期はいずれも既定値 `false`）を
/// プロセス内キャッシュから取得する。
///
/// A/B ベンチ用 `MetalGemm::new_with_swizzle`／`new_with_fine_barrier`
/// はキャッシュ対象外（呼び出し元がそれぞれ明示的に直接構築する。
/// 実装計画 §3.1）。`ctx` はヒット時未使用（ミス時の構築にのみ使う）。
///
/// `ops::MetalBackendOps::gemm`／`gemm_bias_act` の唯一の呼び出し先。
pub(crate) fn cached_gemm(ctx: &Arc<MetalContext>) -> Result<Arc<MetalGemm>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalGemm>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || MetalGemm::new(ctx))
}

/// [`MetalElementwise`] スイートをプロセス内キャッシュから取得する。
/// `ops::MetalBackendOps::elementwise_binary`／`elementwise_unary` の
/// 唯一の呼び出し先。
pub(crate) fn cached_elementwise(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<MetalElementwise>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalElementwise>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || MetalElementwise::new(ctx))
}

/// [`MetalRmsNorm`] スイートをプロセス内キャッシュから取得する。
/// `ops::MetalBackendOps::run_fused_rmsnorm` の唯一の呼び出し先。
pub(crate) fn cached_rmsnorm(ctx: &Arc<MetalContext>) -> Result<Arc<MetalRmsNorm>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalRmsNorm>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || MetalRmsNorm::new(ctx))
}

/// [`MetalLayerNorm`] スイートをプロセス内キャッシュから取得する
/// （イシュー #1596）。`ops::MetalBackendOps::layer_norm` の唯一の
/// 呼び出し先。
pub(crate) fn cached_layer_norm(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<MetalLayerNorm>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalLayerNorm>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || MetalLayerNorm::new(ctx))
}

/// [`MetalNormBackward`] スイートをプロセス内キャッシュから取得する
/// （イシュー #1953。`ops::MetalBackendOps::rmsnorm_backward`／
/// `layer_norm_backward` の唯一の呼び出し先）。
pub(crate) fn cached_norm_backward(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<MetalNormBackward>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalNormBackward>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || MetalNormBackward::new(ctx))
}

/// [`MetalBatchNorm`] スイートをプロセス内キャッシュから取得する
/// （イシュー #1736）。`ops::MetalBackendOps::batch_norm_train`／
/// `batch_norm_infer` の唯一の呼び出し先。
pub(crate) fn cached_batch_norm(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<MetalBatchNorm>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalBatchNorm>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || MetalBatchNorm::new(ctx))
}

/// [`MetalSoftmax`] スイートをプロセス内キャッシュから取得する。
/// `ops::MetalBackendOps::run_fused_softmax` の唯一の呼び出し先。
pub(crate) fn cached_softmax(ctx: &Arc<MetalContext>) -> Result<Arc<MetalSoftmax>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalSoftmax>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || MetalSoftmax::new(ctx))
}

/// [`MetalSgd`] スイートをプロセス内キャッシュから取得する（イシュー
/// #935）。`ops::MetalBackendOps::sgd_step_device` の唯一の呼び出し先。
/// デバイス常駐パラメータ更新は学習ループの毎ステップ呼ばれるため、
/// MSL 再コンパイルを避けるキャッシュの効果が他スイート以上に重要
/// （`docs/device-resident-update-design.md` §3.3d「Cross-tape 契約」）。
pub(crate) fn cached_sgd(ctx: &Arc<MetalContext>) -> Result<Arc<MetalSgd>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalSgd>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || MetalSgd::new(ctx))
}

/// [`MetalAdam`] スイートをプロセス内キャッシュから取得する（イシュー
/// #2070）。`ops::MetalBackendOps::adam_step_device_impl` の唯一の
/// 呼び出し先。デバイス常駐パラメータ更新は学習ループの毎ステップ
/// 呼ばれるため、MSL 再コンパイルを避けるキャッシュの効果が他スイート
/// 以上に重要（`cached_sgd` と同じ理由。`docs/device-resident-update-
/// design.md` §3.3d「Cross-tape 契約」）。
pub(crate) fn cached_adam(ctx: &Arc<MetalContext>) -> Result<Arc<MetalAdam>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalAdam>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || MetalAdam::new(ctx))
}

/// [`crate::mse::MetalMse`] スイートをプロセス内キャッシュから取得する
/// （イシュー #1045）。`ops::MetalBackendOps::mse_loss`／
/// `mse_loss_backward` の唯一の呼び出し先。
pub(crate) fn cached_mse(ctx: &Arc<MetalContext>) -> Result<Arc<crate::mse::MetalMse>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<crate::mse::MetalMse>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || crate::mse::MetalMse::new(ctx))
}

/// [`crate::nll::MetalNll`] スイートをプロセス内キャッシュから取得する
/// （イシュー #1738。`cached_mse` と同型）。`ops::MetalBackendOps::
/// nll_loss`／`nll_loss_backward` の唯一の呼び出し先。
pub(crate) fn cached_nll(ctx: &Arc<MetalContext>) -> Result<Arc<crate::nll::MetalNll>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<crate::nll::MetalNll>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || crate::nll::MetalNll::new(ctx))
}

/// [`crate::kl_div::MetalKlDiv`] スイートをプロセス内キャッシュから
/// 取得する（イシュー #1738。`cached_mse` と同型）。`ops::
/// MetalBackendOps::kl_div_loss`／`kl_div_loss_backward` の唯一の呼び
/// 出し先。
pub(crate) fn cached_kl_div(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<crate::kl_div::MetalKlDiv>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<crate::kl_div::MetalKlDiv>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || crate::kl_div::MetalKlDiv::new(ctx))
}

/// [`crate::huber::MetalHuber`] スイートをプロセス内キャッシュから取得
/// する（イシュー #1739。`cached_mse` と同型）。
/// `ops::MetalBackendOps::huber_loss`／`huber_loss_backward` の唯一の
/// 呼び出し先。
pub(crate) fn cached_huber(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<crate::huber::MetalHuber>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<crate::huber::MetalHuber>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || crate::huber::MetalHuber::new(ctx))
}

/// [`crate::bce::MetalBce`] スイートをプロセス内キャッシュから取得する
/// （イシュー #1737。`cached_mse` と同型）。`ops::MetalBackendOps::
/// bce_loss`／`bce_loss_backward` の唯一の呼び出し先。
pub(crate) fn cached_bce(ctx: &Arc<MetalContext>) -> Result<Arc<crate::bce::MetalBce>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<crate::bce::MetalBce>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || crate::bce::MetalBce::new(ctx))
}

/// [`crate::rnn_cell::MetalRnnCell`] スイートをプロセス内キャッシュから
/// 取得する（イシュー #1647）。`ops::MetalBackendOps::{lstm_pointwise,
/// lstm_hidden_backward, lstm_cell_backward, gru_pointwise, gru_backward}`
/// の唯一の呼び出し先。
pub(crate) fn cached_rnn_cell(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<crate::rnn_cell::MetalRnnCell>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<crate::rnn_cell::MetalRnnCell>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || crate::rnn_cell::MetalRnnCell::new(ctx))
}

/// device 単位のプロセスワイド singleton [`MetalAllocator`]（イシュー
/// #1021・設計文書 §3.1「プールは device 単位のプロセスワイド singleton
/// とする」・§3.5）をプロセス内キャッシュから取得する。
///
/// `cached_gemm` と同じく `ctx` はヒット時未使用（ミス時の構築にのみ
/// 使う）。プロセスに Metal デバイスは 1 台のみ（`context_cache.rs`
/// モジュール冒頭コメント「システムデフォルトの Metal デバイス 1 台
/// のみを扱う前提」）のため、本番経路（`crate::gemm`／`elementwise`／
/// `softmax`／`rmsnorm`／`memory` がいずれも `context_cache::
/// cached_context()` 由来の同一 `MetalContext` を参照する。`ops.rs` の
/// 各演算メソッド参照）では初回構築時に渡した `ctx` が終始一貫する。
///
/// `crate::buffer::MetalBuffer::alloc_zeroed_pooled`／
/// `alloc_uninit_pooled`・`crate::ops::MetalBackendOps::
/// release_cached_device_memory`／`device_memory_pool_stats` の唯一の
/// 呼び出し先。
/// [`MetalGatherScatter`] スイートをプロセス内キャッシュから取得する
/// （イシュー #1778）。`ops::MetalBackendOps::gather`／`scatter` の
/// 唯一の呼び出し先。
pub(crate) fn cached_gather_scatter(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<MetalGatherScatter>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalGatherScatter>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || MetalGatherScatter::new(ctx))
}

/// [`MetalConstantPad`] をプロセス内キャッシュから取得する（イシュー
/// #1756）。`ops::MetalBackendOps::pad` の唯一の呼び出し先
/// （`cached_gather_scatter` と同型）。
pub(crate) fn cached_constant_pad(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<MetalConstantPad>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalConstantPad>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || MetalConstantPad::new(ctx))
}

/// [`MetalIm2col`] をプロセス内キャッシュから取得する（イシュー
/// #1768）。`ops::MetalBackendOps::im2col`／`col2im` の唯一の呼び出し
/// 先（`cached_constant_pad` と同型）。
pub(crate) fn cached_im2col(ctx: &Arc<MetalContext>) -> Result<Arc<MetalIm2col>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalIm2col>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || MetalIm2col::new(ctx))
}

/// [`crate::pooling::MetalPooling`] をプロセス内キャッシュから取得
/// する（イシュー #1730・追従イシュー。`cached_im2col` と同型）。
/// `ops::MetalBackendOps::max_pool2d`／`avg_pool2d`／
/// `adaptive_avg_pool2d` の唯一の呼び出し先。
pub(crate) fn cached_pooling(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<crate::pooling::MetalPooling>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<crate::pooling::MetalPooling>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || crate::pooling::MetalPooling::new(ctx))
}

/// `unique` カーネル（`unique.rs::MetalUnique`）のコンパイル済み
/// パイプラインをプロセス内キャッシュから取得する（イシュー #1734。
/// `cached_gather_scatter` と同型）。
pub(crate) fn cached_unique(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<crate::unique::MetalUnique>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<crate::unique::MetalUnique>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || crate::unique::MetalUnique::new(ctx))
}

/// `cast` カーネルスイート（`fandhe_ai_tensor_core::cast::CastOps` の
/// Metal 実装本体。`cast.rs::MetalCast`）のコンパイル済みパイプライン
/// をプロセス内キャッシュから取得する（イシュー #1751。
/// `cached_unique` と同型）。
pub(crate) fn cached_cast(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<crate::cast::MetalCast>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<crate::cast::MetalCast>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || crate::cast::MetalCast::new(ctx))
}

/// [`crate::interpolate::MetalInterpolate`] をプロセス内キャッシュから
/// 取得する（イシュー #1757）。`ops::MetalBackendOps::interpolate` の
/// 唯一の呼び出し先（`cached_gather_scatter` と同型）。
pub(crate) fn cached_interpolate(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<crate::interpolate::MetalInterpolate>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<crate::interpolate::MetalInterpolate>>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || {
        crate::interpolate::MetalInterpolate::new(ctx)
    })
}

/// scan（累積和／累積積。`scan.rs::MetalScan`）のコンパイル済み
/// パイプラインをプロセス内キャッシュから取得する（イシュー #1740。
/// `cached_unique` と同型）。
pub(crate) fn cached_scan(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<crate::scan::MetalScan>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<crate::scan::MetalScan>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || crate::scan::MetalScan::new(ctx))
}

/// `sort`／`topk` カーネル（`sort.rs::MetalSort`）のコンパイル済み
/// パイプラインをプロセス内キャッシュから取得する（イシュー #1741。
/// `cached_unique` と同型）。
pub(crate) fn cached_sort(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<crate::sort::MetalSort>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<crate::sort::MetalSort>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || crate::sort::MetalSort::new(ctx))
}

/// f32 `sum` reduction（全要素・単一軸。`reduce.rs::MetalReduce`）の
/// コンパイル済みパイプラインをプロセス内キャッシュから取得する
/// （イシュー #1896。`cached_unique` と同型）。
/// `ops::MetalBackendOps::sum` の唯一の呼び出し先。
pub(crate) fn cached_reduce(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<crate::reduce::MetalReduce>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<crate::reduce::MetalReduce>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || crate::reduce::MetalReduce::new(ctx))
}

/// `log_softmax` backward（2 カーネル。`log_softmax_backward.rs::
/// MetalLogSoftmaxBackward`）のコンパイル済みパイプラインをプロセス内
/// キャッシュから取得する（イシュー #1952。`cached_reduce` と同型）。
/// `ops::MetalBackendOps::log_softmax_backward` の唯一の呼び出し先。
pub(crate) fn cached_log_softmax_backward(
    ctx: &Arc<MetalContext>,
) -> Result<Arc<crate::log_softmax_backward::MetalLogSoftmaxBackward>, MetalError> {
    static CACHE: OnceLock<
        Mutex<Option<Arc<crate::log_softmax_backward::MetalLogSoftmaxBackward>>>,
    > = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || {
        crate::log_softmax_backward::MetalLogSoftmaxBackward::new(ctx)
    })
}

pub(crate) fn cached_allocator(ctx: &Arc<MetalContext>) -> Result<Arc<MetalAllocator>, MetalError> {
    static CACHE: OnceLock<Mutex<Option<Arc<MetalAllocator>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    get_or_build(cache, on_poison, || {
        Ok(MetalAllocator::new(Arc::clone(ctx)))
    })
}

/// `ctx` のデバイス上で `op`（[`ScalarUnaryOp`]）のテンプレート生成
/// パイプラインをプロセス内キャッシュから取得する（イシュー #1707。
/// CUDA 側 `context_cache::cached_scalar_unary_kernel`〈イシュー #1700〉
/// の Metal 対応版）。
///
/// Metal はシステムデフォルトデバイス 1 台のみを扱う前提（本モジュール
/// 冒頭コメント）のため、CUDA 側と異なりキーは `op.kind_name()`
/// （[`ScalarOpKind::kind_name`]。ペイロード値を含まない安定文字列）
/// のみを使う（`ContextKey` 相当の区別は不要）。`f32` ペイロード
/// （[`ScalarUnaryOp::Clamp`]。イシュー #1709 実装済み。`crate::
/// scalar_op_source` モジュール doc「ペイロード seam」参照）はキャッシュ
/// キーに含めない契約は CUDA 側と同一（`cached_scalar_unary_pipeline_
/// second_call_reuses_cache_for_clamp_regardless_of_payload` 参照）。
///
/// [`crate::scalar_op_source::unary_kernel_source`] が `None`（未実装
/// kind）を返す場合はキャッシュへ触れずに `Ok(None)` を返す（呼び出し元
/// `ops::MetalBackendOps::scalar_unary_dispatch` が `BackendError::
/// Unsupported` へ変換しホスト参照実装へフォールバックする）。
pub(crate) fn cached_scalar_unary_pipeline(
    ctx: &Arc<MetalContext>,
    op: ScalarUnaryOp,
) -> Result<Option<objc2::rc::Retained<MtlPipeline>>, MetalError> {
    let Some(source) = crate::scalar_op_source::unary_kernel_source(op) else {
        return Ok(None);
    };
    static CACHE: OnceLock<Mutex<HashMap<&'static str, objc2::rc::Retained<MtlPipeline>>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = op.kind_name();
    let built = get_or_build_keyed(cache, key, on_poison, || {
        let library = pipeline::compile_source(ctx.device(), &source)?;
        let name = crate::scalar_op_source::unary_function_name(op);
        // `name` は `format!` が返す `String`（`kind_name()` から動的生成）
        // だが `pipeline::make_pipeline` は `function_name: &'static str`
        // を要求する。`Box::leak`（プロセス生存期間中に有限個〈kind 数〉
        // だけ発生するプロセスワイドキャッシュのため許容するリーク。
        // `ops::MetalBackendOps` の他キャッシュと同じ生存期間契約
        // 〈モジュール冒頭「生存期間」〉）で `'static` へ変換する。
        let name: &'static str = Box::leak(name.into_boxed_str());
        pipeline::make_pipeline(ctx.device(), &library, name)
    })?;
    Ok(Some(built))
}

/// [`cached_scalar_unary_pipeline`] の 2 項版（[`ScalarBinaryOp`]。
/// イシュー #1707）。
pub(crate) fn cached_scalar_binary_pipeline(
    ctx: &Arc<MetalContext>,
    op: ScalarBinaryOp,
) -> Result<Option<objc2::rc::Retained<MtlPipeline>>, MetalError> {
    let Some(source) = crate::scalar_op_source::binary_kernel_source(op) else {
        return Ok(None);
    };
    static CACHE: OnceLock<Mutex<HashMap<&'static str, objc2::rc::Retained<MtlPipeline>>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = op.kind_name();
    let built = get_or_build_keyed(cache, key, on_poison, || {
        let library = pipeline::compile_source(ctx.device(), &source)?;
        let name = crate::scalar_op_source::binary_function_name(op);
        // `cached_scalar_unary_pipeline` と同じ理由で `'static` 化する。
        let name: &'static str = Box::leak(name.into_boxed_str());
        pipeline::make_pipeline(ctx.device(), &library, name)
    })?;
    Ok(Some(built))
}

/// [`cached_fused_elementwise_pipeline`] のプロセス内キャッシュ上限
/// （区分 B-1・イシュー #2085。CUDA 側 `context_cache::
/// FUSED_ELEMENTWISE_CACHE_CAP` と同一値・同一理由）。
const FUSED_ELEMENTWISE_CACHE_CAP: usize = 256;

/// `ctx` のデバイス上で、GPU `run_fused` の elementwise allowlist 融合
/// カーネル（[`crate::fused_elementwise::ElementwiseProgram`] 単位で
/// 動的生成・コンパイルされる。区分 B-1・イシュー #2085）のコンパイル
/// 済みパイプラインをプロセス内キャッシュから取得する。
///
/// `cached_scalar_unary_pipeline`／`cached_scalar_binary_pipeline` と
/// 異なり `get_or_build_keyed`（`K: Copy` 制約）を使わない——本関数の
/// キー（[`crate::fused_elementwise_source::cache_key`] が融合
/// プランの op 列から都度導出する `String`）は `Copy` ではないため。
/// 代わりに `Mutex<HashMap<String, Arc<Mutex<Option<_>>>>>`（CUDA 側
/// `context_cache::cached_fused_elementwise_kernel` の `SingleFlightCache`
/// と同型のキー単位 2 階層ロック）を直接操作し、同じ**キャッシュ上限**
/// （[`FUSED_ELEMENTWISE_CACHE_CAP`]。上限到達時は新規コンパイルを行わ
/// ず `Ok(None)`。呼び出し元 `ops::MetalBackendOps::run_fused` の融合
/// 分岐が `BackendError::Unsupported` へ変換し per-op フォールバックへ
/// 委ねる。fail-closed。OWASP A04・`.claude/rules/security.md`）を課す。
/// 関数名は固定リテラル（`FUSED_EW_FUNCTION_NAME`）のため
/// `cached_scalar_unary_pipeline` と異なり `Box::leak` は不要。
///
/// 上限チェックとスロット予約（外側 `HashMap` へのキー登録）は同一の
/// 外側ロック区間内で原子的に行う（codex-review P2 是正・PR #2232。
/// 旧実装は「チェック → 解放 → 別ロックで登録」の 2 段構成だったため、
/// チェックと登録の間に異なるキーへの並行呼び出しが割り込むと、複数の
/// 呼び出しが同時に「上限未満」を観測してそれぞれ新規エントリを登録
/// でき、並行呼び出し数に応じて上限を任意に超過しうる欠陥があった。
/// CUDA 側 `cached_fused_elementwise_kernel` と同じ是正）。コンパイル
/// （`pipeline::compile_source`／`make_pipeline`）自体はキー単位の内側
/// ロックを保持したまま行うため、同一キーへの並行呼び出しは 2 回目
/// 以降が内側ロックの取得で待機し構築を二重実行しない（副次的に、旧
/// 実装が許容していた同一キーの冗長構築も解消される）。
///
/// コンパイル失敗時はスロットの登録自体を取り消す（codex-review P2 是正・
/// PR #2232。CUDA 側 `context_cache::cached_fused_elementwise_kernel` と
/// 同じ是正）。以前の実装は空スロットを外側 `HashMap` へ登録した後に
/// `pipeline::compile_source`／`make_pipeline` を呼んでいたが、いずれかが
/// 失敗しても登録済みスロットを削除しなかったため、一時的なコンパイル
/// 失敗が累積すると `guard.len()` が [`FUSED_ELEMENTWISE_CACHE_CAP`] へ
/// 達し、以後の正常な融合プランまで恒久的に `Ok(None)`（呼び出し元
/// フォールバック）へ落ちる欠陥があった。本関数は構築に失敗した場合、
/// 他スレッドが同じキーで構築に成功していない（スロットが `None` の
/// まま）ことを確認したうえでエントリを削除し、次回呼び出しが枠を
/// 再利用して再試行できるようにする。
pub(crate) fn cached_fused_elementwise_pipeline(
    ctx: &Arc<MetalContext>,
    key: &str,
    source: &str,
    function_name: &'static str,
) -> Result<Option<objc2::rc::Retained<MtlPipeline>>, MetalError> {
    static CACHE: OnceLock<
        Mutex<HashMap<String, Arc<Mutex<Option<objc2::rc::Retained<MtlPipeline>>>>>>,
    > = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    let slot = {
        let mut guard = cache
            .lock()
            .map_err(|e| on_poison(format!("fused elementwise pipeline cache poisoned: {e}")))?;
        if !guard.contains_key(key) && guard.len() >= FUSED_ELEMENTWISE_CACHE_CAP {
            return Ok(None);
        }
        Arc::clone(
            guard
                .entry(key.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(None))),
        )
    };

    let mut slot_guard = slot
        .lock()
        .map_err(|e| on_poison(format!("fused elementwise pipeline cache poisoned: {e}")))?;
    if let Some(existing) = slot_guard.as_ref() {
        return Ok(Some(existing.clone()));
    }
    let build_result: Result<objc2::rc::Retained<MtlPipeline>, MetalError> = (|| {
        let library = pipeline::compile_source(ctx.device(), source)?;
        pipeline::make_pipeline(ctx.device(), &library, function_name)
    })();
    match build_result {
        Ok(built) => {
            *slot_guard = Some(built.clone());
            Ok(Some(built))
        }
        Err(err) => {
            // コンパイル失敗（`compile_source`／`make_pipeline` のいずれか）
            // 時は、枠だけ確保して中身が空のまま残るスロットが
            // `FUSED_ELEMENTWISE_CACHE_CAP` の枠を恒久的に消費するのを
            // 防ぐため、外側 `HashMap` からエントリ自体を取り消す
            // （codex-review 指摘・PR #2232。本関数内のロック順序は常に
            // 外側 → 内側のため、内側 `slot_guard` を保持したまま外側
            // ロックを取ると逆順になり他呼び出しの通常経路とデッドロック
            // しうる。よって一旦内側ロックを解放してから外側ロックを取得
            // し、削除直前に内側ロックを再取得して「まだ未構築のまま」
            // であることを確認した上で削除する（この間に他スレッドが
            // 同じキーで構築に成功していれば、削除せずそのエントリを
            // 温存する）。
            drop(slot_guard);
            if let Ok(mut guard) = cache.lock()
                && guard
                    .get(key)
                    .is_some_and(|existing| Arc::ptr_eq(existing, &slot))
            {
                let still_unbuilt = slot.lock().map(|inner| inner.is_none()).unwrap_or(false);
                if still_unbuilt {
                    guard.remove(key);
                }
            }
            Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 汎用キャッシュ契約（ヒットの clone・ビルド失敗の非キャッシュ・
    /// poison 時 fail-closed）のテスト本体は `crate::generic_cache` へ
    /// 移設済み（Linux CI でも実行される。イシュー #930 codex-review
    /// 対応: 本モジュールは `cfg(target_os = "macos")` 限定のため、
    /// ここに置いたままでは Linux CI で全く実行されず GPU 非依存の
    /// キャッシュ契約が未検証のまま埋もれてしまう）。
    ///
    /// 本テストは Metal 固有の配線（poison 変換クロージャ [`on_poison`]
    /// が実際に [`MetalError::ContextCacheUnavailable`] を返すこと）のみ
    /// を確認する。
    #[test]
    fn on_poison_produces_context_cache_unavailable() {
        let err = on_poison("simulated poison".into());
        assert!(matches!(err, MetalError::ContextCacheUnavailable { .. }));
    }

    /// [`cached_context`] を 2 回呼ぶと `Arc::ptr_eq` で同一インスタンスが
    /// 返る（実 Metal デバイスが必要なため Mac 実機でのみ実行される。
    /// `tests/device_smoke.rs` と同じ前提）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn cached_context_returns_same_instance_across_calls() {
        let first = cached_context().expect("Metal context available on test host");
        let second = cached_context().expect("cache hit");
        assert!(
            Arc::ptr_eq(&first, &second),
            "2 回目の呼び出しは同一 Arc<MetalContext> を返すはず"
        );
    }

    /// [`cached_gemm`] も同様に同一インスタンスを返す。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn cached_gemm_returns_same_instance_across_calls() {
        let ctx = cached_context().expect("Metal context available on test host");
        let first = cached_gemm(&ctx).expect("gemm suite builds");
        let second = cached_gemm(&ctx).expect("cache hit");
        assert!(
            Arc::ptr_eq(&first, &second),
            "2 回目の呼び出しは同一 Arc<MetalGemm> を返すはず"
        );
    }

    /// [`cached_scalar_unary_pipeline`] を超越関数系 kind（`Log`。イシュー
    /// #1708 で実装した 8 kind のうち代表 1 つ）で 2 回呼ぶと同一
    /// パイプラインを返す（CUDA 側
    /// `scalar_op_cache_wiring_tests::
    /// cached_scalar_unary_kernel_second_call_reuses_cache_for_transcendental_kind`
    /// と同型。キャッシュ機構が kind 非依存の汎用機構であることの
    /// 追加確認）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn cached_scalar_unary_pipeline_second_call_reuses_cache_for_transcendental_kind() {
        let ctx = cached_context().expect("Metal context available on test host");
        let first = cached_scalar_unary_pipeline(&ctx, ScalarUnaryOp::Log)
            .expect("Log is implemented")
            .expect("Log must return Some(pipeline)");
        let second = cached_scalar_unary_pipeline(&ctx, ScalarUnaryOp::Log)
            .expect("2nd call must succeed given the 1st succeeded")
            .expect("Log must return Some(pipeline)");
        // `Retained<MtlPipeline>` は `ptr_eq` を持たないため
        // `Retained::as_ptr`（生ポインタ抽出。所有権は移動しない）で
        // 比較する（`cached_context_returns_same_instance_across_calls`
        // の `Arc::ptr_eq` と同じ意図の Retained 版）。
        assert!(
            std::ptr::eq(
                objc2::rc::Retained::as_ptr(&first),
                objc2::rc::Retained::as_ptr(&second)
            ),
            "2 回目の cached_scalar_unary_pipeline(Log) 呼び出しは同一パイプラインを返すはず"
        );
    }

    /// [`cached_scalar_unary_pipeline`] を [`ScalarUnaryOp::Clamp`]
    /// （イシュー #1709。初のペイロードあり unary kind）で異なる
    /// `min`／`max` 値により 2 回呼んでも同一パイプラインを返す
    /// ことを確認する（キャッシュキーが `kind_name()` のみに依存し
    /// payload 非依存であることの直接検証。モジュール doc「ペイロード
    /// seam」参照）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn cached_scalar_unary_pipeline_second_call_reuses_cache_for_clamp_regardless_of_payload() {
        let ctx = cached_context().expect("Metal context available on test host");
        let first = cached_scalar_unary_pipeline(&ctx, ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 })
            .expect("Clamp is implemented")
            .expect("Clamp must return Some(pipeline)");
        let second = cached_scalar_unary_pipeline(
            &ctx,
            ScalarUnaryOp::Clamp {
                min: -5.0,
                max: 5.0,
            },
        )
        .expect("2nd call must succeed given the 1st succeeded")
        .expect("Clamp must return Some(pipeline)");
        assert!(
            std::ptr::eq(
                objc2::rc::Retained::as_ptr(&first),
                objc2::rc::Retained::as_ptr(&second)
            ),
            "異なる payload 値でも 2 回目の cached_scalar_unary_pipeline(Clamp) 呼び出しは同一パイプラインを返すはず"
        );
    }
}
