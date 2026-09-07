//! CUDA デバイス初期化・メタデータ取得、および
//! `fandhe_ai_tensor_core::device::DeviceProvider` の CUDA 実装（TASK-1.7a・#32 と
//! TASK-1.9a・#44 を統合）。
//!
//! PoC-v2-3 の `CudaGemm::new`（`docs/spec/03-poc/poc-v2-3-cuda-gemm/code/rust/src/cuda/mod.rs:119-162`）
//! からデバイス初期化・メタデータ部分のみを productize した [`CudaDevice`]
//! を土台とし、その上に薄いラッパーとして [`CudaDeviceProvider`]（3
//! バックエンド共通 trait 実装）を構築する構成とする。カーネル保持・
//! `run_*`（GEMM 実行）は #33（naive GEMM）・#34（tiled GEMM）が
//! `CudaDevice` の上に載せる。
//!
//! # 動的ロード panic 回避ゲート（受け入れ条件の核心）
//!
//! `cudarc` 0.19.8 の `dynamic-loading` feature は、`libcuda` が
//! `dlopen` できない環境で driver API（`CudaContext::new` 等）を直接
//! 呼ぶと `Err` ではなく **panic** する（`culib()` が
//! `panic_no_lib_found` を呼ぶ。cudarc-0.19.8/src/driver/sys/mod.rs:16119-16129）。
//! PoC-v2-3 のコメント「`CudaContext::new` の時点で `Err` を返す」は
//! `libcuda` が存在し `cuInit` 等が失敗するケースのみ正しく、
//! `libcuda` 不在（CUDA 非搭載環境）では不正確である。
//!
//! そのため `CudaDevice::new`／`device_count` は driver API を呼ぶ前に
//! 必ず `cudarc::driver::sys::is_culib_present()`（non-panicking な
//! 存在プローブ）でゲートし、不在なら `CudaError::DriverUnavailable`
//! を返してから抜ける。これにより「CUDA 非搭載環境で実行時に型付き
//! エラーが返る（panic しない）」という #32 の受け入れ条件を満たす。
//!
//! [`CudaDeviceProvider`]（`fandhe_ai_tensor_core::device::DeviceProvider` 実装）は
//! `enumerate`／`select` の内部で必ずこの `CudaDevice` 経由の初期化
//! パスを通す。`CudaContext::new`／`CudaContext::device_count` を
//! `CudaDevice` を経由せず直接呼ぶと上記の panic 回避ゲートを迂回して
//! しまうため、`BackendError::CudaUnavailable`／`DeviceUnavailable` へ
//! 変換する前に必ず `CudaDevice::is_available()`／`device_count()`／
//! `new()` を経由する。

use std::sync::Arc;

// `capturable_stream_for`（本ファイル下部）専用の import。同関数・その
// static キャッシュと同じ `#[cfg(not(feature = "internal-diagnostics"))]`
// で揃える（理由は `CAPTURABLE_STREAM_CACHE` doc コメント参照）。
#[cfg(not(feature = "internal-diagnostics"))]
use std::collections::HashMap;
#[cfg(not(feature = "internal-diagnostics"))]
use std::sync::{Mutex, OnceLock};

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaStream};

use crate::error::CudaError;
use fandhe_ai_tensor_core::device::{BackendError, Device, DeviceInfo, DeviceProvider};

/// `ordinal` ごとに、その ordinal で最初に確定した [`StreamKind`] の
/// 決定を、以後の `requires_created_stream()` の値に関わらず**恒久的に
/// 固定**（sticky）するためのキャッシュ（codex-review P0 再指摘対応・
/// PR #1390 再々修正）。
///
/// # 背景（なぜ「capture 可能ストリームだけの共有」では閉じないか）
///
/// [`CudaDevice::new`] は `crate::context_cache::cached_device`（内部限定
/// キャッシュ）の構築経路からも、`pub fn new` として crate 外から**直接**
/// 呼ばれる経路（`CudaMemory::new(&CudaDevice::new(ordinal)?)` 等。crate の
/// 公開 API）からも到達する。同一 ordinal に対する複数回の `new` 呼び出し
/// の**間**で opt-in フラグ（`crate::graph::step_graph_mode()`）が切り替わる
/// と（例: ON で 1 回目を構築した後、OFF に切り替えて同じ ordinal を再構築）、
/// 1 回目は `capturable_stream_for` 経由の capture 可能ストリーム
/// （`StreamKind::Created`。イベント追跡無効化済み）、2 回目は
/// `ctx.default_stream()`（`StreamKind::Legacy`。新規 `CudaContext` の
/// イベント追跡は既定で有効）となり、**同一 ordinal 上で Created と
/// Legacy のストリーム種別が共存**しうる。この状態で片方の `CudaDevice`
/// で確保したバッファをもう片方が駆動する演算へ渡すと、確保元ストリーム
/// でのイベント追跡状態が異なるため cross-stream の自動同期
/// （`cuStreamWaitEvent`）が一貫せず、進行中カーネル・転送との読み書き
/// 競合・解放後アクセスが起こりうる（`ordinal`・`generation` の一致検査は
/// いずれも通過するため、これらの検査だけでは検出できない。codex-review
/// 指摘・PR #1390）。
///
/// # 対策
///
/// **ordinal ごとの `StreamKind` の決定を、その ordinal で最初に
/// `CudaDevice::new` が呼ばれた時点の `requires_created_stream()` の値で
/// 固定し、以後は現在のフラグ値を無視してキャッシュ済みの決定を使う**
/// （`ResolvedStreamKind`。Created の場合はその 1 本の `Arc<CudaStream>`
/// をキャッシュに保持し以後の呼び出しへそのまま共有、Legacy の場合は
/// 決定のみを記録し呼び出し元は毎回 `ctx.default_stream()` を呼ぶ——
/// legacy ストリームはプロセス内で単一の NULL stream `cu_stream: null`
/// を指す〈cudarc-0.19.8 `default_stream` 実装参照〉ため Arc 共有は
/// 不要）。これにより「この ordinal では `StreamKind` は全プロセスで
/// 常に 1 種類だけ」という不変条件を、直接構築・キャッシュ経由いずれの
/// 呼び出し経路でも、かつ構築順序やフラグの後からの切り替えとも独立に
/// 機械的に保証する（`docs/backend-cuda-graph-step-capture-design.md`
/// §4.1 の「単一ストリーム」契約をインスタンス単位ではなく ordinal 単位の
/// 恒久的な不変条件へ強化）。opt-in を Legacy 固定後に ON へ切り替えても
/// この ordinal は Legacy のまま——`is_capturable_stream() == false` と
/// なり、`ops.rs::CudaBackendOps::captured_segment_key` の既存の
/// `!is_capturable_stream()` fail-closed 経路（design doc §4.7「opt-in が
/// 最初のデバイス初期化に間に合わなかった」契約）がそのまま機能する。
///
/// `internal-diagnostics` feature が有効な間は `StreamKind::Created` 分岐
/// 自体が到達不能（`new` 冒頭の cfg 分岐コメント参照）のため、本 static・
/// 直後の [`resolve_stream_kind_for`] も同じ `#[cfg(not(feature =
/// "internal-diagnostics"))]` で揃える（`--all-features` ビルドで未使用
/// 扱いになり `-D warnings` を壊すのを避けるため）。
#[cfg(not(feature = "internal-diagnostics"))]
static STREAM_KIND_CACHE: OnceLock<Mutex<HashMap<usize, ResolvedStreamKind>>> = OnceLock::new();

/// [`STREAM_KIND_CACHE`] に記録する、ordinal ごとに固定された解決済みの
/// ストリーム種別（`StreamKind` そのものと異なり、`Created` の場合は共有
/// すべき `Arc<CudaStream>` 本体を保持する）。
#[cfg(not(feature = "internal-diagnostics"))]
#[derive(Clone)]
enum ResolvedStreamKind {
    Legacy,
    // Bugbot 指摘対応（PR #1390 再修正・「Cached stream paired with a new
    // context」）: ストリームだけでなく、それを生成した `CudaContext`
    // 自身も一緒に保持・共有する。`resolve_stream_kind_for` doc コメント
    // 「# 対策」節参照。
    Created(Arc<CudaContext>, Arc<CudaStream>),
}

/// [`STREAM_KIND_CACHE`] 経由で `ordinal` の `(ctx, stream, StreamKind)` を
/// 解決する。`ctx`（`CudaContext`）自体の生成もこの関数が引き受ける
/// （Bugbot 指摘対応。PR #1390 再修正。旧稿は呼び出し元が先に
/// `CudaContext::new(ordinal)` して `&ctx` を渡していたため、`Created`
/// 決定が既にキャッシュ済みの ordinal に対する 2 回目以降の呼び出しで
/// 返る `stream`〈1 回目に作った `CudaContext` 由来〉と、呼び出し元が
/// 新規に作った `ctx`〈今回限りの別 `CudaContext`〉が食い違っていた）。
/// ordinal が未キャッシュの場合のみ `requires_created` の現在値で決定し
/// （`Created` なら新規 `CudaContext::new(ordinal)` の上で `new_stream()`
/// と `disable_event_tracking()` を実行）、以後はキャッシュ済みの決定を
/// `requires_created` の値に関わらずそのまま返す（sticky。doc コメント
/// 「# 対策」節参照）。`Created` の場合は `ctx`／`stream` 双方を 1 回目の
/// `Arc` のまま共有して返す（`Legacy` は `ctx.default_stream()` がどの
/// `CudaContext` から呼んでも安全な単一 NULL stream のため、`ctx` は
/// 呼び出しのたびに独立した新規 `CudaContext::new(ordinal)` でよい）。
///
/// # Safety
///
/// 呼び出し元（[`CudaDevice::new`]）は、本関数が返した `(ctx, stream,
/// StreamKind::Created)` の組を `CudaDevice` の `ctx`／`stream` フィールド
/// としてそのまま保持する契約とする（この ordinal に対して本関数以外の
/// 経路で追加の `new_stream()` を呼ばない）。この契約が守られる限り、
/// `unsafe { ctx.disable_event_tracking() }` の安全性根拠（この ordinal
/// では本キャッシュ内で唯一のストリームしか作らない）が成立する。
#[cfg(not(feature = "internal-diagnostics"))]
fn resolve_stream_kind_for(
    ordinal: usize,
    requires_created: bool,
) -> Result<(Arc<CudaContext>, Arc<CudaStream>, StreamKind), CudaError> {
    let cache = STREAM_KIND_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(existing) = guard.get(&ordinal) {
        return Ok(match existing {
            // Legacy は呼び出しのたびに独立した `CudaContext` を作ってよい
            // （`ctx.default_stream()` はどの `CudaContext` から呼んでも
            // プロセス内で単一の NULL stream を指すため、`ctx` と `stream`
            // が別インスタンスでも不整合は生じない。`ResolvedStreamKind`
            // doc コメント参照）。
            ResolvedStreamKind::Legacy => {
                let ctx = CudaContext::new(ordinal)?;
                let stream = ctx.default_stream();
                (ctx, stream, StreamKind::Legacy)
            }
            // Created は 1 回目の呼び出しで作った `CudaContext`／
            // `CudaStream` をそのまま共有する（Bugbot 指摘対応。
            // `CudaDevice::new` 呼び出し元コメント参照）: `ctx` を新規に
            // 作り直すと `device.context()` と `device.stream()` が別々の
            // `CudaContext` を指してしまい、カーネルロード先（`ctx`）と
            // 起動先ストリーム（`stream`）の context が食い違う。
            ResolvedStreamKind::Created(ctx, stream) => {
                (Arc::clone(ctx), Arc::clone(stream), StreamKind::Created)
            }
        });
    }
    if requires_created {
        let ctx = CudaContext::new(ordinal)?;
        let created = ctx.new_stream()?;
        // SAFETY: 上記関数 doc コメント「# Safety」節参照。本関数はミューテックス
        // 保持下で「未キャッシュの場合にのみ」`new_stream()` を呼ぶため、同一
        // ordinal に対して複数回この分岐へ到達することはなく（Mutex により
        // 直列化済み）、この ordinal で作られるストリームはこの 1 本のみとなる。
        unsafe {
            ctx.disable_event_tracking();
        }
        guard.insert(
            ordinal,
            ResolvedStreamKind::Created(Arc::clone(&ctx), Arc::clone(&created)),
        );
        Ok((ctx, created, StreamKind::Created))
    } else {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        guard.insert(ordinal, ResolvedStreamKind::Legacy);
        Ok((ctx, stream, StreamKind::Legacy))
    }
}

/// GPU 1 台分のハンドル・メタデータ。
///
/// `ctx`／`stream` は `Arc` で共有し、#33/#34 のカーネルモジュールが
/// クローンして保持する契約とする（PoC-v2-3 と同じく `CudaContext`／
/// `CudaStream` 自体が内部で `Arc` 前提の API 設計になっているため）。
/// `arch` は NVRTC の `--gpu-architecture` にそのまま渡せる
/// `compute_XY` 形式の文字列（`nvrtc::compile_ptx` の呼び出し契約）。
pub struct CudaDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    ordinal: usize,
    name: String,
    compute_capability: (i32, i32),
    arch: String,
    stream_kind: StreamKind,
    /// managed memory（`cuMemAllocManaged`）を安全に使える構成かどうか
    /// （イシュー #1352。`crate::placement`「配置非依存の管理」参照）。
    ///
    /// `CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY`（デバイスが managed memory
    /// をサポートするか）と `CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS`
    /// （ホスト・デバイス双方から同時アクセス可能か。GB10 のような
    /// 物理統合メモリ環境で該当）の両方が非 0 の場合にのみ `true`。
    /// `new` の中で 1 回だけ照会してキャッシュする（`cached_device`
    /// 経由で ordinal ごとに再利用されるため、per-op のコストにはならない）。
    ///
    /// 事前照会が必要な理由: `cudarc::driver::safe::unified_memory::
    /// CudaContext::alloc_unified` は自身も `MANAGED_MEMORY` 属性を検査し
    /// 非対応なら `DriverError(CUDA_ERROR_NOT_PERMITTED)` を返すが、この
    /// エラーは `context_cache::classify_cuda_result` の operation-local
    /// 一覧に含まれず sticky（ordinal を poison する）扱いになる。
    /// `memory.rs` の managed 確保経路は driver 呼び出しの**前**に本
    /// フィールドを検査し、非対応デバイスでは `CudaError::
    /// ManagedMemoryUnsupported`（非 `Driver` variant。`observe_cuda_result`
    /// を素通りし poison を起こさない）で fail-closed に拒否する。
    managed_memory_supported: bool,
}

/// `CudaDevice::stream` の種別（イシュー #1349・`docs/backend-cuda-graph-
/// step-capture-design.md` §4.1）。
///
/// `cudarc` の legacy NULL stream（`default_stream()`）は
/// `cuStreamBeginCapture` の対象にできない（driver が
/// `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` を返す）ため、CUDA Graph
/// capture opt-in（`crate::graph::step_graph_mode` が
/// `crate::graph::GraphMode::StreamOnly` 以上）が最初の CUDA デバイス
/// 初期化より前に設定されている場合のみ `ctx.new_stream()`
/// （非 legacy・capture 可能なストリーム）を保持する。opt-in OFF
/// （既定）では常に [`StreamKind::Legacy`] のまま、本イシュー導入前と
/// ストリーム挙動・性能とも bit・timing とも不変（`new_stream()` は
/// `cudarc` の「multi-stream mode」を有効化し、以降の全 `device_ptr`
/// 呼び出しに `cuStreamWaitEvent` を自動挿入するため、無条件に切り替える
/// と opt-in OFF 時の既存経路まで変えてしまう。design doc §4.1）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamKind {
    /// `ctx.default_stream()`（legacy NULL stream）。capture 不可。
    Legacy,
    /// `ctx.new_stream()`（capture 可能な非 legacy ストリーム）。
    Created,
}

impl CudaDevice {
    /// `libcuda` が動的リンカから解決可能かを、ライブラリ初期化を
    /// 伴わずに確認する。
    ///
    /// # Safety
    ///
    /// `cudarc::driver::sys::is_culib_present()` は `unsafe fn` だが、
    /// 内部で行うのは cudarc が生成する標準ライブラリ名候補
    /// （`libcuda.so` 等）に対する `dlopen` 試行のみであり、事前条件
    /// （初期化順序・排他制御等）は要求しない。dlopen はライブラリの
    /// 初期化コード（コンストラクタ関数）を実行しうるが、探索対象は
    /// 動的リンカの標準探索パス（`LD_LIBRARY_PATH` 含む）上の CUDA
    /// 公式ライブラリのみであり、動的リンカの標準信頼モデルの範囲内
    /// である（`.claude/rules/security.md` の unsafe 方針）。
    pub fn is_available() -> bool {
        // SAFETY: 上記ドキュメンテーションコメント（# Safety 節）参照。
        // `cudarc::driver::sys::is_culib_present()` は事前条件を要求せず、
        // 動的リンカの標準探索パス上の CUDA 公式ライブラリに対する
        // `dlopen` 試行のみを行う non-panicking なプローブである。
        unsafe { cudarc::driver::sys::is_culib_present() }
    }

    /// `ordinal` 番目の GPU を初期化し、ハンドル・メタデータを構築する。
    ///
    /// 手順: (1) `is_culib_present()` プローブで `libcuda` の存在を
    /// 確認（不在なら panic 回避のためここで `Err` を返す）、
    /// (2) `CudaContext::new(ordinal)`、(3) `default_stream()`、
    /// (4) `name()`/`compute_capability()` 取得、(5) `arch` 文字列構築。
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        if !Self::is_available() {
            return Err(CudaError::DriverUnavailable {
                detail: "libcuda dynamic library not found (dlopen failed); \
                         CUDA driver is not installed or not on the library search path"
                    .to_string(),
            });
        }

        // イシュー #1349: CUDA Graph capture opt-in が最初のデバイス
        // 初期化より前に設定されている場合のみ、capture 可能な
        // `new_stream()` を保持する（`StreamKind` doc コメント参照）。
        // opt-in OFF（既定）では現行どおり `default_stream()`
        // （legacy NULL stream）のまま、本イシュー導入前と挙動不変。
        //
        // **`internal-diagnostics` feature が有効な間は常に
        // `StreamKind::Legacy` に固定する（codex-review P0 再々指摘
        // 対応。PR #1390）**: 下記 `unsafe { ctx.disable_event_tracking()
        // }` の安全性根拠は「この `ctx` 上で `CudaDevice` が唯一の
        // ストリームしか作らない」という本クレート内部の運用契約であり、
        // `context()`/`stream()` の可視性ゲート（`internal-diagnostics`
        // feature 有効時のみ `pub`。本ファイル下部 doc コメント参照）
        // だけでは閉じ切れない。`internal-diagnostics` と graph capture
        // opt-in は同一プロセス内で同時に有効化できる（CI の `cargo test
        // --workspace --all-features` が両方を有効化する）ため、その
        // 組合せ下では利用者が `context().clone().new_stream()` で第 2
        // の非追跡ストリームを作れてしまい、`ctx` 上でイベント追跡が
        // 無効化された状態と矛盾する（`AGENTS.md` の unsafe 不変条件
        // 保証の要件違反）。そこで「イベント追跡を無効化する分岐」と
        // 「raw context／stream を安全な公開 API として返す分岐」を
        // **同一ビルドで両立させない**よう、`internal-diagnostics`
        // feature が有効な間はこの分岐自体を到達不能にする（機構としては
        // `unsafe { ctx.disable_event_tracking() }` を呼ぶコード自体が
        // コンパイルされないため、実行時チェックに頼らず型システムの
        // 外側〈cfg〉で不変条件を保証する）。この場合、graph capture
        // opt-in が有効でも `stream_kind` は `Legacy` のままとなり、
        // `ops.rs::CudaBackendOps::captured_segment_key` の
        // `!is_capturable_stream()` 検査が `BackendError::Unsupported`
        // を返して fail-closed にフォールバックする（design doc §4.7 の
        // 既存の「opt-in が最初のデバイス初期化に間に合わなかった」経路
        // と同じ扱い。`internal-diagnostics` feature は crates.io の
        // 通常利用者が有効化しない開発者・CI 専用 feature のため、この
        // フォールバックが本番経路の性能・挙動へ影響することはない）。
        #[cfg(feature = "internal-diagnostics")]
        let (ctx, stream, stream_kind) = {
            let ctx = CudaContext::new(ordinal)?;
            let stream = ctx.default_stream();
            (ctx, stream, StreamKind::Legacy)
        };
        //
        // **codex-review P0 再指摘対応（PR #1390 再々修正）・Bugbot 指摘
        // 対応（PR #1390 再修正。「Cached stream paired with a new
        // context」）**: `StreamKind` の決定を ordinal ごとに恒久固定
        // （sticky）する `resolve_stream_kind_for`（本ファイル冒頭の
        // `STREAM_KIND_CACHE`）を経由する。理由は同関数の doc コメント
        // 「# 背景」「# 対策」節を参照——公開 `CudaDevice::new` は
        // `context_cache::cached_device` の構築経路とは独立に何度でも
        // 呼び出せるため、同一 ordinal に対する複数回の呼び出しの間で
        // opt-in フラグ（`crate::graph::step_graph_mode()`）が切り替わると、
        // `Created` ストリームと `Legacy` ストリームが同一 ordinal 上で
        // 共存しうる。`resolve_stream_kind_for` は ordinal ごとに最初の
        // 呼び出し時点のフラグ値で `StreamKind` を確定し、以後は現在の
        // フラグ値を無視してその決定を返すことで、この共存を構造的に防ぐ。
        //
        // **`ctx` 自体も `resolve_stream_kind_for` から受け取る（Bugbot
        // 指摘の是正）**: 旧稿は `CudaDevice::new` が呼び出しのたびに
        // 無条件で新規 `CudaContext::new(ordinal)` を作り、その `ctx` を
        // 使って `resolve_stream_kind_for` へ渡していた。`StreamKind::
        // Created` が既にキャッシュ済みの ordinal に対して 2 回目以降の
        // `CudaDevice::new` を呼ぶと、返る `stream`（1 回目の呼び出しで
        // 作られた `Arc<CudaStream>`。内部に自身の生成元 `CudaContext` を
        // 保持する）は 1 回目の `CudaContext` に紐づいたままなのに対し、
        // `self.ctx` は今回新規に作った別の `CudaContext` になり、
        // `device.context()` と `device.stream()` が異なる CUDA context を
        // 指す不整合が生じていた（`gemm_mma_tf32x3.rs::CudaMmaTf32x3Gemm
        // ::new` 等が `device.context().load_module(ptx)` でカーネルを
        // 今回の新規 `ctx` へロードしつつ `device.stream().clone()` で
        // 1 回目の `ctx` に紐づくストリームへ起動する形になり、モジュール
        // 読み込み先と起動先ストリームの context が食い違ってカーネル
        // 起動が失敗しうる）。`resolve_stream_kind_for` が `ctx` の生成
        // 自体も引き受け、`Created` 決定済みの ordinal では 1 回目に
        // 作った `Arc<CudaContext>` をそのまま共有して返すことで、
        // `device.context()`／`device.stream()` は常に同一 `CudaContext`
        // を指す（`Legacy` 決定の ordinal では従来どおり呼び出しのたびに
        // 独立した `CudaContext::new(ordinal)` を作る。単一の NULL
        // stream を返す `ctx.default_stream()` は元々どの `ctx` から
        // 呼んでも安全なため挙動不変）。
        #[cfg(not(feature = "internal-diagnostics"))]
        let (ctx, stream, stream_kind) = resolve_stream_kind_for(
            ordinal,
            crate::graph::step_graph_mode().requires_created_stream(),
        )?;
        let name = ctx.name()?;
        let compute_capability = ctx.compute_capability()?;
        // nvrtc の --gpu-architecture は仮想アーキテクチャ（compute_XY）を
        // 受け付ける。実機の compute capability をそのまま使うことで、
        // sm 番号のハードコードが新しい GPU 世代で無効化される事態を
        // 避ける（PoC-v2-3 の方針を踏襲。cuda/mod.rs:129-132）。
        let arch = format!("compute_{}{}", compute_capability.0, compute_capability.1);
        // managed memory 対応可否を 1 回だけ照会する（フィールド doc
        // コメント参照）。属性取得自体が失敗した場合は非対応として扱う
        // （fail-soft ではなく fail-closed: `total_memory_bytes`／
        // `compute_units` と異なり、この値は opt-in 経路の安全性判定に
        // 使うため、取得失敗を「対応している」側へ倒さない）。
        let managed_memory_supported = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)
            .unwrap_or(0)
            != 0
            && ctx
                .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS)
                .unwrap_or(0)
                != 0;

        Ok(Self {
            ctx,
            stream,
            ordinal,
            name,
            compute_capability,
            arch,
            stream_kind,
            managed_memory_supported,
        })
    }

    /// システムに存在する CUDA デバイス数を返す。
    ///
    /// `new` と同じ理由で `is_culib_present()` によるプローブゲートを
    /// 先行させる（panic 回避。DGX Spark GB10 は単一 GPU 構成のため
    /// 呼び出し元は通常 `new(0)` のみで足りるが、複数 GPU 環境向けに
    /// 提供する）。
    pub fn device_count() -> Result<usize, CudaError> {
        if !Self::is_available() {
            return Err(CudaError::DriverUnavailable {
                detail: "libcuda dynamic library not found (dlopen failed); \
                         CUDA driver is not installed or not on the library search path"
                    .to_string(),
            });
        }
        let count = CudaContext::device_count()?;
        Ok(count as usize)
    }

    /// #33/#34 のカーネルロード・起動が使う `CudaContext` 共有ハンドル。
    ///
    /// **可視性を `internal-diagnostics` feature でゲートし、かつ
    /// `unsafe { ctx.disable_event_tracking() }` 分岐自体を同 feature
    /// 下では到達不能にする（codex-review P0 再々指摘対応・PR #1390。
    /// 旧稿は feature ゲートのみで `internal-diagnostics` と graph
    /// capture opt-in の同時有効化を防げていなかった）**:
    /// `crate::graph`（イシュー #1349）が opt-in ON 時この `ctx` 上に
    /// `disable_event_tracking()` を適用する根拠は「この `CudaDevice`
    /// はこの `ctx` 上で唯一のストリーム（`self.stream`）しか作らない」
    /// という**本クレート内部の**運用契約であり、`cudarc` 自身が強制
    /// する不変条件ではない。`CudaDevice` 自体は本クレート（crates.io
    /// 公開クレート `fandhe-ai-backend-cuda`）から `pub use` で再公開
    /// されているため、本メソッドが無条件 `pub` のままだと、クレート外
    /// の利用者が安全な公開 API の組み合わせだけで `context().clone()`
    /// → `.new_stream()` と呼んで**この `ctx` 上に第 2 のストリーム**を
    /// 作れてしまう（`AGENTS.md` の unsafe 不変条件保証の要件に違反）。
    /// 第 2 ストリームが作られると、イベント追跡を無効化済みの `ctx` の
    /// 下で 2 本のストリーム間の読み書き順序を保証する手段がなくなり、
    /// バッファをまたいだ競合が起こりうる。
    ///
    /// 単なる可視性ゲートだけでは不十分だった理由: `internal-diagnostics`
    /// と graph capture opt-in は同一プロセス内で**同時に有効化できる**
    /// （CI の `cargo test --workspace --all-features` が両方を有効化
    /// する）。そのため `new`（本ファイル上部）は `internal-diagnostics`
    /// feature が有効な間は `stream_kind` を常に [`StreamKind::Legacy`]
    /// に固定し、`disable_event_tracking()` を呼ぶコード自体をコンパイル
    /// 対象から外す（`new` 内の cfg 分岐コメント参照）。これにより
    /// 「イベント追跡が無効化された `ctx`」と「raw context／stream を
    /// 安全な公開 API として返す」の 2 条件が**同一ビルドで両立しない**
    /// ことが型システムの外側（cfg。コンパイル時）で保証される
    /// （実行時チェックには依存しない）。この場合 graph capture opt-in
    /// は事実上無効化され、`ops.rs::CudaBackendOps::captured_segment_key`
    /// の `!is_capturable_stream()` 検査により `BackendError::
    /// Unsupported` を返す fail-closed 経路へフォールバックする
    /// （design doc §4.7）。`internal-diagnostics` は crates.io の通常
    /// 利用者が有効化しない開発者・CI 専用 feature のため、本番経路
    /// （同 feature 無効）の性能・挙動には影響しない。
    ///
    /// 本クレート自身の実機診断テスト・ベンチ（`tests/`・`examples/`
    /// 配下。`large_buffer_percall_alloc_ab_1149.rs`・
    /// `tma_probe_real_device.rs`・`device_attributes_dump.rs` 等）は、
    /// 上記のとおり opt-in が無効化された状態（`ctx` のイベント追跡は
    /// 無効化されていない）でのみ `context()`/`stream()` へ直接
    /// アクセスして driver 属性・pool 状態を読む正当な既存用途を持つ
    /// （`Cargo.toml` の `internal-diagnostics` feature コメント「内部
    /// 診断専用ツールの可視性制御」参照）。そのため既定ビルド
    /// （`internal-diagnostics` feature 無効。crates.io の通常利用者は
    /// この feature を有効化しない）では `pub(crate)` に絞る一方、同
    /// feature 有効時（本クレート自身の `tests/`／`examples/` が
    /// `required-features` 経由でのみ要求する。CI の `cargo test
    /// --workspace --all-features` は常にこの feature を含む）だけ
    /// `pub` へ戻す。既定ビルドの公開 API 面からは変わらず除外される
    /// ため、`AGENTS.md` が要求する不変条件保証は成立する
    /// （`docs/backend-cuda-graph-step-capture-design.md` §4.1）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// [`Self::context`] doc コメント参照。既定ビルド（`internal-diagnostics`
    /// feature 無効）ではクレート内部限定に絞る。
    #[cfg(not(feature = "internal-diagnostics"))]
    pub(crate) fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// #33/#34 のカーネル起動・メモリ転送が使う既定ストリーム。
    ///
    /// `internal-diagnostics` feature によるゲート（codex-review P0
    /// 再々指摘対応・PR #1390）: [`Self::context`] と同じ理由。本
    /// feature 有効時は `new` が `stream_kind` を常に
    /// [`StreamKind::Legacy`] に固定するため、本メソッドが返す
    /// `&Arc<CudaStream>` は capture 可能な非 legacy ストリームには
    /// なり得ない。
    #[cfg(feature = "internal-diagnostics")]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// [`Self::stream`] doc コメント参照。既定ビルド（`internal-diagnostics`
    /// feature 無効）ではクレート内部限定に絞る。
    #[cfg(not(feature = "internal-diagnostics"))]
    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// このデバイスのストリームが CUDA Graph capture 可能（`StreamKind::
    /// Created`）かどうかを返す（イシュー #1349）。`ops.rs::
    /// CudaBackendOps::captured_segment_key` が opt-in ON でも実際に
    /// capture 可能なストリームで初期化済みかを判定するために使う
    /// （opt-in が最初のデバイス初期化より遅れて ON にされた場合、
    /// このデバイスは `StreamKind::Legacy` のまま残り、本メソッドは
    /// `false` を返す。`StreamKind` doc コメント参照）。
    pub fn is_capturable_stream(&self) -> bool {
        matches!(self.stream_kind, StreamKind::Created)
    }

    /// `new` に渡した GPU の ordinal（デバイス番号）。
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// GPU 名（README 等の実施環境節に転記する情報。PoC-v2-3 の
    /// `device_name` 相当）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `cudaDeviceProp` 相当から取得した compute capability（major, minor）。
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    /// NVRTC の `--gpu-architecture` にそのまま渡せる `compute_XY` 形式の
    /// アーキテクチャ文字列（`nvrtc::compile_ptx` の呼び出し契約）。
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// デバイスの総メモリ容量（バイト）。取得失敗はプロパティ欠損として
    /// `None` に落とす（[`CudaDeviceProvider::probe`] から呼ばれる。
    /// デバイス自体の検出成功〈`new` の成否〉を主判定材料とするため）。
    fn total_memory_bytes(&self) -> Option<u64> {
        self.ctx.total_mem().ok().map(|bytes| bytes as u64)
    }

    /// SM（マルチプロセッサ）数。取得失敗時は `None`（`total_memory_bytes`
    /// と同じ fail-soft 方針）。
    fn compute_units(&self) -> Option<u32> {
        self.ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .ok()
            .and_then(|count| u32::try_from(count).ok())
    }

    /// SM（マルチプロセッサ）数の公開アクセサ（イシュー #499）。
    ///
    /// `compute_units` と同一の取得ロジック・
    /// fail-soft 方針（取得失敗時 `None`）をそのまま公開する薄いラッパー。
    /// `swizzle::select_swizzle_group_width`（`swizzle.rs`）・
    /// `gemm_mma.rs::CudaMmaGemm::new_with_swizzle`・
    /// `examples/gemm_mma_swizzle_bench.rs` が、グルーピング幅の動的選択に
    /// 使う SM 数をここから取得する。`DeviceInfo::compute_units`
    /// （`CudaDeviceProvider::probe`）は既に同じ値を crate 外へ公開して
    /// いるため（`fandhe_ai_tensor_core::device::DeviceInfo` 経由）、本アクセサは
    /// 新規の公開面を作るものではなく、`CudaDevice` から直接取得する経路を
    /// 追加するのみ。
    pub fn multiprocessor_count(&self) -> Option<u32> {
        self.compute_units()
    }

    /// ブロックあたり opt-in 可能な共有メモリの上限バイト数
    /// （`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`。
    /// イシュー #742）。
    ///
    /// `cudaFuncAttributeMaxDynamicSharedMemorySize`（driver API では
    /// `cuFuncSetAttribute` の同名属性）で 1 ブロックへ割り当て可能な
    /// **動的**共有メモリの実効上限で、既定の
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`（48KiB。static
    /// `__shared__` 宣言の実効上限と同一値。
    /// `crate::kernels_mma::MMA_STATIC_SMEM_LIMIT_BYTES`）より大きい
    /// （sm_121 GB10 実測 101,376B。`docs/perf/sm121-device-attributes.md`
    /// §「SMEM 実効帯域」参照）。[`multiprocessor_count`](Self::multiprocessor_count)
    /// と同じ fail-soft 方針（取得失敗時 `None`）。呼び出し元は
    /// `internal-diagnostics` feature 配下の TF32 staged 段数スイープ
    /// example（`examples/gemm_wmma_tf32_staged_stages_bench.rs`）で、
    /// 動的共有メモリ変種カーネルの opt-in 予算検査に使う。
    pub fn shared_memory_per_block_optin(&self) -> Option<u32> {
        self.ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
            .ok()
            .and_then(|bytes| u32::try_from(bytes).ok())
    }

    /// managed memory（`cuMemAllocManaged`）を安全に使える構成かどうか
    /// （`new` で照会・キャッシュ済み。フィールド doc コメント参照）。
    /// `memory.rs` の managed 確保経路（`crate::placement::
    /// managed_placement_enabled()` が `true` の場合のみ参照される）が、
    /// driver 呼び出し前の fail-closed 事前検査として使う。
    pub fn managed_memory_supported(&self) -> bool {
        self.managed_memory_supported
    }

    /// SM（マルチプロセッサ）1 個あたりの共有メモリ上限バイト数
    /// （`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`。
    /// イシュー #742）。
    ///
    /// [`shared_memory_per_block_optin`](Self::shared_memory_per_block_optin)
    /// と同じ呼び出し元が、段数ごとの occupancy 上限
    /// （`floor(この値 / ブロックあたり SMEM 所要)`）算出に使う。
    pub fn shared_memory_per_multiprocessor(&self) -> Option<u32> {
        self.ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
            .ok()
            .and_then(|bytes| u32::try_from(bytes).ok())
    }
}

/// CUDA バックエンドの `DeviceProvider` 実装（TASK-1.9a・#44）。
///
/// `fandhe_ai_tensor_core::device::DeviceProvider` の CUDA 実装。`cudarc` は無条件
/// 依存＋動的ロード方式であるため（`.claude/rules/deps-policy.md`）、CUDA
/// toolkit・ドライバが非搭載の環境でも本クレートはビルドが成立する。この
/// 契約の実行時側の受け皿として、本 provider はドライバ不在時に
/// `panic!`／`unwrap()` せず `is_available() == false`・
/// `enumerate() == Ok(vec![])` を返す（REQ-1・`docs/public-api-design.md`
/// §4.4 `BackendError::CudaUnavailable` のコメント参照）。
///
/// `enumerate`／`select` の呼び出しごとに [`CudaDevice::device_count`]／
/// `Self::probe`（内部で `crate::context_cache::cached_device` を経由。
/// イシュー #929）を経由してプローブする。いずれも内部で必ず
/// `is_culib_present()` の panic 回避ゲートを通すため（モジュール冒頭
/// コメント参照）、本 provider が `CudaContext` を直接呼ぶことはない。
/// コンテキストの常駐・再利用（`ordinal` キーのプロセス内キャッシュ）は
/// `crate::context_cache` が一元的に担い、`BackendOps` 結線
/// （`ops::CudaBackendOps`）も同じキャッシュを参照するため、2 回目以降の
/// `select`（`facade::tape_for(Device::Cuda(_))` の存在検証経路）は
/// `CudaContext::new` を再実行しない。
#[derive(Debug, Default, Clone, Copy)]
pub struct CudaDeviceProvider;

impl CudaDeviceProvider {
    /// 新規 provider を構築する。CUDA ドライバの検出自体は
    /// `is_available`／`enumerate`／`select` 呼び出し時に遅延して行う
    /// （構築時点ではプローブしない）。
    pub fn new() -> Self {
        Self
    }

    /// 指定 ordinal のデバイス情報を取得する。`CudaDevice::new` が
    /// 失敗した場合（ドライバ不在・範囲外 ordinal 等）は `CudaError`
    /// をそのまま呼び出し元へ伝播し、`CudaUnavailable`／
    /// `DeviceUnavailable` への変換は呼び出し元（`enumerate`／`select`）
    /// が文脈に応じて行う。
    ///
    /// イシュー #929: `CudaDevice::new` を直接呼ばず
    /// `crate::context_cache::cached_device` 経由にする。2 回目以降の
    /// `select`（`facade::tape_for(Device::Cuda(_))` の存在検証経路）が
    /// `CudaContext::new` を再実行しないため（受け入れ条件 1）。失敗は
    /// キャッシュされず毎回再試行される（`context_cache` モジュール冒頭
    /// コメント「fail-fast 契約」参照）ため、本関数の fail-fast セマン
    /// ティクス（範囲外 ordinal・driver 不在の区別。`enumerate`/`select`
    /// ドキュメンテーションコメント参照）は変更しない。
    fn probe(ordinal: usize) -> Result<DeviceInfo, CudaError> {
        let device = crate::context_cache::cached_device(ordinal)?;
        let total_memory_bytes = device.total_memory_bytes();
        let compute_units = device.compute_units();
        Ok(DeviceInfo::new(
            Device::Cuda(ordinal),
            device.name().to_string(),
            total_memory_bytes,
            compute_units,
        ))
    }
}

impl DeviceProvider for CudaDeviceProvider {
    fn backend_name(&self) -> &'static str {
        "cuda"
    }

    fn is_available(&self) -> bool {
        // `device_count() > 0` だけでは「デバイス数は正だが `enumerate` は
        // コンテキスト初期化失敗等で全滅させ 0 件を返す」という矛盾が
        // 生じうる（Bugbot #237 指摘 2）。`enumerate` と同じ探索・除外
        // ロジック（`probe` 成功のみ数える）を通し、実際に選択可能な
        // デバイスが 1 件以上あることを条件にする。
        matches!(self.enumerate(), Ok(devices) if !devices.is_empty())
    }

    fn enumerate(&self) -> Result<Vec<DeviceInfo>, BackendError> {
        // ドライバ不在（toolkit 非搭載環境等）は `Err` ではなく空列挙を
        // 返す。呼び出し元（`enumerate_all`）が「1 バックエンドの不在で
        // 全体の列挙が止まらない」ことを前提にできるようにするため
        // （モジュール冒頭コメント参照）。
        let count = match CudaDevice::device_count() {
            Ok(count) => count,
            Err(_) => return Ok(vec![]),
        };
        let devices = (0..count)
            .filter_map(|ordinal| Self::probe(ordinal).ok())
            .collect();
        Ok(devices)
    }

    fn select(&self, device: Device) -> Result<DeviceInfo, BackendError> {
        let ordinal = match device {
            Device::Cuda(ordinal) => ordinal,
            other => {
                return Err(BackendError::DeviceUnavailable(format!(
                    "CudaDeviceProvider cannot select {other:?}"
                )));
            }
        };
        // 範囲外 ordinal（不正なリクエスト）と CUDA バックエンド自体が
        // 利用不可（ドライバ不在等）を呼び出し側が区別できるよう、
        // `probe` 前に `device_count()` で ordinal を検証する
        // （Bugbot #237 指摘 1）。`device_count()` 自体の失敗は
        // ドライバ不在を意味するため `CudaUnavailable` のまま維持する。
        let count = match CudaDevice::device_count() {
            Ok(count) => count,
            Err(err) => return Err(BackendError::CudaUnavailable(format!("{err}"))),
        };
        if ordinal >= count {
            return Err(BackendError::DeviceUnavailable(format!(
                "ordinal {ordinal} out of range (found {count} CUDA device(s))"
            )));
        }
        Self::probe(ordinal).map_err(|err| BackendError::CudaUnavailable(format!("{err}")))
    }
}
