//! `backend-cuda` の型付きエラー。
//!
//! `device.rs`（デバイス初期化）・`nvrtc.rs`（NVRTC コンパイル）が返す
//! 失敗をすべて `CudaError` に集約する。TASK-1.9（#43/#44）で
//! `BackendOps`/`BackendError`（`docs/public-api-design.md` §4.4）が
//! 導入された際は、呼び出し元（backend 抽象層）が本型を
//! `BackendError::CudaUnavailable(String)` 等へマップする想定である
//! （本イシューではマップ側は実装しない）。

use std::fmt;

/// CUDA バックエンドの初期化・コンパイル失敗を表す。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、#33/#34（カーネル起動・
/// メモリ転送）で variant が増えても呼び出し側の網羅的 match を
/// 破壊しないため。
#[non_exhaustive]
#[derive(Debug)]
pub enum CudaError {
    /// `libcuda` が動的リンカから見つからず `dlopen` できない
    /// （CUDA 非搭載環境。受け入れ条件「CUDA 非搭載環境で実行時に
    /// 型付きエラーが返る（panic しない）」の主役）。
    ///
    /// `cudarc` 0.19.8 の `dynamic-loading` feature は、この状態で
    /// driver API（`CudaContext::new` 等）を直接呼ぶと `Err` ではなく
    /// panic する（`culib()` が `panic_no_lib_found` を呼ぶ。
    /// cudarc-0.19.8/src/driver/sys/mod.rs:16119-16129、src/lib.rs:199-200）。
    /// `CudaDevice::new`/`device_count` は本 variant を返す前に必ず
    /// `cudarc::driver::sys::is_culib_present()` で存在確認し、
    /// 不在ならこの panic 経路に入る前に `Err` を返す（device.rs 参照）。
    DriverUnavailable { detail: String },

    /// `libnvrtc` が動的リンカから見つからず `dlopen` できない。
    ///
    /// driver 側と同様、NVRTC 側にも `is_culib_present` による
    /// panic 回避プローブが用意されている
    /// （cudarc-0.19.8/src/nvrtc/sys/mod.rs:529、540、550）。
    /// `compile_ptx`（nvrtc.rs）はコンパイル呼び出し前に必ずこれを
    /// 確認する。
    NvrtcUnavailable { detail: String },

    /// `cuInit`／コンテキスト生成／デバイス問い合わせ等、driver API
    /// 呼び出し自体が `Err` を返したケース（`libcuda` は存在するが
    /// 実行時エラーが発生した場合。例: ドライババージョン不一致）。
    Driver(cudarc::driver::result::DriverError),

    /// NVRTC コンパイル失敗（構文エラー・`--include-path` 未解決等）。
    Compile(cudarc::nvrtc::CompileError),

    /// GEMM 起動 API（`gemm.rs`）のホスト側形状検証が拒否した入力。
    ///
    /// `run_naive_f32`／`run_naive_f16` は GPU へバッファを転送しカーネルを
    /// 起動する前に、スライス長が `m*k`／`k*n` と一致するか・`m`/`n`/`k` の
    /// 積が `usize`/`i32` の範囲でオーバーフローしないかを検証する
    /// （`gemm.rs::validate_gemm_dims` 参照）。この検証は GPU 起動前の
    /// A03 インジェクション対策（外部由来の形状値を境界チェックなしに
    /// カーネル引数へ渡さない。`.claude/rules/security.md`）を兼ねる。
    InvalidShape { detail: String },

    /// elementwise 演算（`elementwise.rs::CudaElementwise`）のホスト側形状
    /// 検証、および GEMM epilogue の bias 長検証（`gemm.rs::run_tiled_bias_act_f32`）
    /// が拒否した入力。
    ///
    /// `InvalidShape` の `Display` 実装は "invalid GEMM shape" 接頭辞を
    /// 固定で付けるため、elementwise（`add`／`mul`／`relu`／`exp`／`tanh`）や
    /// bias 長不一致の失敗を `InvalidShape` で表すと `to_string()` 経由の
    /// エラーマッピングが GEMM 形状エラーと誤表示してしまう
    /// （codex-review 指摘・イシュー #599 PR #688 レビュー）。GEMM 本体の
    /// 形状検証（`m*k`／`k*n` 一致・オーバーフロー等）とは別 variant に
    /// 分離し、`Display` メッセージも専用文言にする。
    InvalidElementwiseShape { detail: String },

    /// 転置カーネル起動 API（`transpose.rs::CudaTranspose`）のホスト側形状
    /// 検証が拒否した入力。
    ///
    /// `validate_transpose_dims`／`validate_transpose_output_len`（素朴・
    /// smem 転置の `src.len()==m*n`／`dst.len()==out_m*out_n` 検証）と
    /// `validate_tiled_transposed_gemm_dims`（GEMM epilogue 融合転置の
    /// `a.len()==m*k`／`b.len()==k*n` 検証）が対象。転置は GEMM 本体
    /// （`InvalidShape`）でも elementwise／bias（`InvalidElementwiseShape`）
    /// でもないため、`Display` メッセージの誤表示（Cursor Bugbot 指摘・
    /// PR #690）を避けて独立 variant に分離する。GEMM epilogue 融合転置
    /// （`m`/`n`/`k` を持つ）が `InvalidShape` と紛らわしい構造を持つ点は
    /// 承知のうえで、あくまで転置 API（`transpose.rs`）の検証であることを
    /// 優先し本 variant に統一する。
    InvalidTransposeShape { detail: String },

    /// f16 WMMA GEMM（`gemm_wmma.rs::CudaWmmaGemm`）が、Tensor Core（WMMA）
    /// の要件を満たさないデバイス上で要求された。
    ///
    /// WMMA f16 経路は compute capability 7.0 以降を要求する（設計メモ
    /// `docs/cuda-tensor-core-design.md` 7 節「ディスパッチ規則への引き渡し
    /// 事項」）。`CudaWmmaGemm::new` は NVRTC コンパイルを試みる前に
    /// `CudaDevice::compute_capability()` でこの下限を検査し、満たさない
    /// 場合は本 variant を返す（対応する tiled 経路へのフォールバック判断は
    /// TASK-11.2／#66 のディスパッチ規則側の責務であり、本クレートでは
    /// 判定のみを行う）。
    TensorCoreUnsupported { detail: String },

    /// WMMA(TF32) GEMM カーネル（`kernels::WMMA_TF32_F32`）が `CudaGemm::new`
    /// 時点でコンパイル・ロードに失敗しており、`run_wmma_tf32` を呼べない
    /// 状態であることを表す。
    ///
    /// `WMMA_TF32_F32` は `#include <mma.h>`（NVRTC の include パス解決が
    /// 必要）と compute capability 8.0 以降を要求するため、naive/tiled の
    /// 4 カーネル（`#include` を使わず全 compute capability で成立）より
    /// 失敗しうる環境が広い（レビュー指摘 #62。実機での `<mma.h>` 解決は
    /// 未検証）。`CudaGemm::new` はこの失敗を `Err` として早期 return せず
    /// 本 variant の detail として保持し、naive/tiled 4 カーネルの可用性を
    /// 道連れにしない（`gemm.rs::CudaGemm::new` ドキュメンテーションコメント
    /// 参照）。
    WmmaUnavailable { detail: String },

    /// カーネルソースのテンプレート展開（`kernels_mma::render_mma_f16`・
    /// `kernels_wmma_opt::render_wmma_tf32_opt`／`render_wmma_f16_opt`。
    /// イシュー #516）に渡された shape／タイル／段数の構成値が、境界検査・
    /// 共有メモリ予算・整列制約等の不変条件を満たさないケースを表す。
    ///
    /// レンダラは文字列組み立てより前にこれらの不変条件を検査し、違反時は
    /// 実際に NVRTC へ渡すことなく本 variant で早期拒否する（A03 対策。
    /// 不正な構成値がそのままカーネルソースへ焼き込まれるのを防ぐ。
    /// `.claude/rules/security.md`）。既定 config はコンパイル時 const
    /// アサーションで別途保証されるため、本 variant は非既定 config
    /// （後続 #519 の次元別静的化選択・#521 の段数逆算等が生成する構成）
    /// を検証する経路でのみ返る。
    InvalidKernelConfig { detail: String },

    /// カーネル特化パラメータ記述子（[`crate::CudaKernelDescriptor`]）の
    /// 構築時、ブロックタイル寸法（BM/BN/BK）・パイプライン段数が
    /// ゼロ値で渡された。
    ///
    /// イシュー #504（Phase C-1）: metal-flash-attention の
    /// `Optional + fatalError` 方式（未確定パラメータのまま descriptor を
    /// 使うと実行時に fatal error）を、Rust では「非ゼロを要求する
    /// コンストラクタ + 型付きエラー」で置き換える。`NonZeroU32::new` が
    /// `None` を返した場合にのみこの variant を返し、`unwrap`/`expect` に
    /// よる panic 経路は持たない（`.claude/rules/coding-rust.md`
    /// 「本番経路で `unwrap()`/`expect()` を使わない」）。
    InvalidKernelDescriptor { detail: String },

    /// コンパイルキャッシュのルートディレクトリ（`nvrtc.rs::cache_root`）が
    /// 解決できない。
    ///
    /// イシュー #506（Phase C-2）: `RUST_AI_CUDA_CACHE_DIR` / `XDG_CACHE_HOME`
    /// / `HOME` のいずれからもキャッシュルートを導けない場合（全環境変数
    /// 欠落）、`RUST_AI_CUDA_CACHE_DIR` に空文字列・相対パスが指定された
    /// 場合、または解決結果（3 分岐いずれも）がコンパイル時ワークスペース
    /// ルート配下に字句上収まる場合（PR #659 codex-review P0 指摘。
    /// `resolve_cache_root`／`path_lexically_within` 参照）に返る。相対
    /// パスを許すとカレントディレクトリ（リポジトリツリー内でありうる）
    /// 配下にキャッシュが作られ、「キャッシュルートはリポジトリツリー外」
    /// 要件（security.md・runner workspace に成果物を残さない方針）に
    /// 反するため fail-closed で拒否する（panic 経路は持たない）。
    /// containment 検証はシンボリックリンク非対応の字句比較に留まり、
    /// シンボリックリンク対応の `canonicalize` 再検証は C-3（#509）が
    /// 実ディレクトリ作成・オープン時点で担う。
    CacheDirUnavailable { detail: String },
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::DriverUnavailable { detail } => {
                write!(f, "CUDA driver library unavailable: {detail}")
            }
            CudaError::NvrtcUnavailable { detail } => {
                write!(f, "CUDA NVRTC library unavailable: {detail}")
            }
            // `cudarc::driver::result::DriverError`／`cudarc::nvrtc::CompileError`
            // は本クレートが依存する `cudarc` の feature 構成（driver／nvrtc／
            // dynamic-loading／f16。deps-policy.md 記載の 4 feature に限定し
            // `std` feature は有効化しない）では `std::fmt::Display`／
            // `std::error::Error` を実装しない（`std` feature でのみ
            // `#[cfg(feature = "std")]` 実装が有効になる。cudarc-0.19.8
            // src/driver/result.rs:73-81）。そのため `Debug` 表示に委ねる
            // （PoC-v2-3 の `CudaGemmError` と同じ方針。cuda/mod.rs:48・54）。
            CudaError::Driver(e) => write!(f, "cuda driver error: {e:?}"),
            CudaError::Compile(e) => write!(f, "nvrtc compile error: {e:?}"),
            CudaError::InvalidShape { detail } => {
                write!(f, "invalid GEMM shape: {detail}")
            }
            CudaError::InvalidElementwiseShape { detail } => {
                write!(f, "invalid elementwise/bias shape: {detail}")
            }
            CudaError::InvalidTransposeShape { detail } => {
                write!(f, "invalid transpose shape: {detail}")
            }
            CudaError::TensorCoreUnsupported { detail } => {
                write!(f, "tensor core (WMMA) unsupported on this device: {detail}")
            }
            CudaError::WmmaUnavailable { detail } => {
                write!(f, "WMMA(TF32) GEMM kernel unavailable: {detail}")
            }
            CudaError::InvalidKernelConfig { detail } => {
                write!(f, "invalid kernel template config: {detail}")
            }
            CudaError::InvalidKernelDescriptor { detail } => {
                write!(f, "invalid CUDA kernel descriptor: {detail}")
            }
            CudaError::CacheDirUnavailable { detail } => {
                write!(
                    f,
                    "CUDA kernel compile cache directory unavailable: {detail}"
                )
            }
        }
    }
}

// 上記と同じ理由（`cudarc` を `std` feature なしで利用）で
// `DriverError`／`CompileError` は `std::error::Error` を実装しないため
// `source()` で辿ることができない。`Display` に整形済みメッセージを
// 含めているため、既定の `source() -> None` のままで足りる。
impl std::error::Error for CudaError {}

// `?` で `device.rs`／`nvrtc.rs` から `CudaError` へそのまま伝播できるよう、
// cudarc が返す 2 種類のエラー型（driver API 由来・NVRTC コンパイル由来）
// それぞれから個別に変換する（PoC-v2-3 の CudaGemmError と同じ方針。
// `std::error::Error` 全体へのブランケット実装は標準ライブラリの
// `impl<T> From<T> for T` と衝突するため採らない）。
impl From<cudarc::driver::result::DriverError> for CudaError {
    fn from(e: cudarc::driver::result::DriverError) -> Self {
        CudaError::Driver(e)
    }
}

impl From<cudarc::nvrtc::CompileError> for CudaError {
    fn from(e: cudarc::nvrtc::CompileError) -> Self {
        CudaError::Compile(e)
    }
}
