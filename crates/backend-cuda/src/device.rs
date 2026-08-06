//! CUDA デバイス初期化・メタデータ取得。
//!
//! PoC-v2-3 の `CudaGemm::new`（`docs/spec/03-poc/poc-v2-3-cuda-gemm/code/rust/src/cuda/mod.rs:119-162`）
//! からデバイス初期化・メタデータ部分のみを productize したもの。
//! カーネル保持・`run_*`（GEMM 実行）は #33（naive GEMM）・#34（tiled GEMM）が
//! 本モジュールの `CudaDevice` の上に載せる。
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
//! エラーが返る（panic しない）」という本イシュー（#32）の受け入れ
//! 条件を満たす。

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};

use crate::error::CudaError;

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
}
