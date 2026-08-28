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

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaStream};

use crate::error::CudaError;
use fandhe_ai_tensor_core::device::{BackendError, Device, DeviceInfo, DeviceProvider};

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

        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let name = ctx.name()?;
        let compute_capability = ctx.compute_capability()?;
        // nvrtc の --gpu-architecture は仮想アーキテクチャ（compute_XY）を
        // 受け付ける。実機の compute capability をそのまま使うことで、
        // sm 番号のハードコードが新しい GPU 世代で無効化される事態を
        // 避ける（PoC-v2-3 の方針を踏襲。cuda/mod.rs:129-132）。
        let arch = format!("compute_{}{}", compute_capability.0, compute_capability.1);

        Ok(Self {
            ctx,
            stream,
            ordinal,
            name,
            compute_capability,
            arch,
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
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// #33/#34 のカーネル起動・メモリ転送が使う既定ストリーム。
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
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
/// [`Self::probe`]（内部で `crate::context_cache::cached_device` を経由。
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
