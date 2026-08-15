//! MSL 実行時コンパイル・パイプライン構築（TASK-1.8b・#39）。
//!
//! [`crate::gemm`] のディスパッチ入口が呼ぶ土台層。`shaders/gemm.metal`
//! を `include_str!` で埋め込み、[`crate::context::MetalContext`] が保持
//! するデバイス上で `newLibraryWithSource_options_error` によりコンパイル
//! し、関数名から `MTLComputePipelineState` を構築する。#40（TASK-1.8c）
//! で [`crate::gemm::MetalGemm::new`] が本モジュールの [`compile_gemm_library`]・
//! [`make_pipeline`] を 3 回呼んで `gemm_tiled`・`gemm_simdgroup` の
//! パイプラインも併設した（naive 版と同じ関数を再利用）。
//!
//! **移植元**: `docs/spec/03-poc/poc-v2-4-metal-gemm/code/rust/src/metal_gemm.rs`
//! の `MetalGemm::new`（コンパイル・パイプライン構築部分）。PoC は
//! `Option` 返し（`?` で `None` へ潰す）だったが、本実装は診断文字列を
//! 保持した [`MetalError`] を返す（coding-rust.md「本番経路で
//! unwrap/expect を使わない」）。
//!
//! **コンパイルオプション**（`MTLCompileOptions`）: `docs/spec/03-poc/poc-v2-5-backend-numeric-parity/code/rust/src/metal_backend.rs:70-80`
//! と同一の `MathMode::Safe` + `MathFloatingPointFunctions::Precise` を
//! 明示する。既定値（`MathMode` 既定は `Fast`・`MathFloatingPointFunctions`
//! 既定は `Fast`）に依存すると Metal のバージョン・デバイスによって
//! 丸め・関数ディスパッチ先が変わりうるため、CPU 参照実装（`f32::mul_add`）
//! ・CUDA 側の標準精度指定と数値一致条件を確実に揃える（REQ-2・
//! PoC-v2-5 実測確認済み）。

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
    MTLMathFloatingPointFunctions, MTLMathMode,
};

use crate::context::MtlDevice;
use crate::error::MetalError;
use crate::tile::TileConfig;

/// `shaders/gemm.metal` のソース（naive・tiled・simdgroup の 3 段カーネルを含む）。
const GEMM_MSL_SRC: &str = include_str!("shaders/gemm.metal");

pub(crate) type MtlLibrary = ProtocolObject<dyn MTLLibrary>;
pub(crate) type MtlPipeline = ProtocolObject<dyn MTLComputePipelineState>;

/// `shaders/gemm.metal` を実行時コンパイルして `MTLLibrary` を返す。
///
/// [`crate::gemm::MetalGemm::new`] から、パイプライン構築（[`make_pipeline`]）の
/// 前段として呼ばれる。エラー時の `message` は `NSError` の
/// `localizedDescription`（構文エラー等の診断文字列）を保持する。
pub(crate) fn compile_gemm_library(device: &MtlDevice) -> Result<Retained<MtlLibrary>, MetalError> {
    let src = NSString::from_str(GEMM_MSL_SRC);
    let options = compile_options();

    device
        .newLibraryWithSource_options_error(&src, Some(&options))
        .map_err(|err| MetalError::LibraryCompilation {
            message: err.localizedDescription().to_string(),
        })
}

/// `MathMode::Safe` + `MathFloatingPointFunctions::Precise` を明示した
/// `MTLCompileOptions` を構築する（設定理由は本ファイル冒頭のコメント参照。
/// PoC-v2-5 実測構成・REQ-2 前提 (b)）。
///
/// [`compile_gemm_library`] から呼ばれるほか、`#[cfg(test)]` の
/// `compile_options_pins_precise_math_mode`（本ファイル末尾）が返却値の
/// getter を検査することで、「Precise 明示」の設定漏れ・fast-math への
/// 退行を GPU デバイス無しでも機械検出できるようにする（TASK-2.2c・#55。
/// PR #38 Bugbot 指摘: `mathMode=Safe` だけでは transcendental 関数が
/// fast 経路に落ちる余地が残るため、両フィールドを独立に固定する）。
pub(crate) fn compile_options() -> Retained<objc2_metal::MTLCompileOptions> {
    let options = objc2_metal::MTLCompileOptions::new();
    options.setMathMode(MTLMathMode::Safe);
    options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
    options
}

/// `library` から `function_name` の関数を取得し `MTLComputePipelineState`
/// を構築する。[`crate::gemm::MetalGemm::new`] から `GemmVariant::function_name()`
/// （`"gemm_naive"`/`"gemm_tiled"`/`"gemm_simdgroup"`）を指定して 3 回呼ばれる。
pub(crate) fn make_pipeline(
    device: &MtlDevice,
    library: &MtlLibrary,
    function_name: &'static str,
) -> Result<Retained<MtlPipeline>, MetalError> {
    let name = NSString::from_str(function_name);
    let func = library
        .newFunctionWithName(&name)
        .ok_or(MetalError::FunctionNotFound {
            name: function_name,
        })?;
    device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|err| MetalError::PipelineCreation {
            message: err.localizedDescription().to_string(),
        })
}

/// `gemm_simdgroup_tiled`（`shaders/gemm.metal`。TASK-1.8f・#188）を
/// `cfg`（[`TileConfig`]）の MSL function constant（`BM`/`BN`/`BK`/`WM`/
/// `WN`/`USE_TGP_STAGING`/`TGP_PAD`。index 0〜6。`TGP_PAD` はイシュー #538
/// で追加した threadgroup memory パディング幅）を畳み込んだ状態でコンパイル・
/// パイプライン化する。[`crate::gemm::MetalGemm`] の構成キー → パイプライン
/// 遅延キャッシュから、構成ごとに 1 回だけ呼ばれる想定
/// （`newFunctionWithName_constantValues_error` は MSL コンパイラを呼ぶ
/// 比較的重い処理）。
///
/// `library` は [`compile_gemm_library`] が返す `shaders/gemm.metal` 全体の
/// ライブラリを再利用する（function constant はコンパイル済みライブラリ
/// から関数を「特殊化」する API であり、ソース側の再コンパイルは不要）。
pub(crate) fn make_pipeline_with_constants(
    device: &MtlDevice,
    library: &MtlLibrary,
    function_name: &'static str,
    cfg: TileConfig,
) -> Result<Retained<MtlPipeline>, MetalError> {
    let name = NSString::from_str(function_name);
    let constants = MTLFunctionConstantValues::new();

    // SAFETY: `setConstantValue_type_atIndex` は指定ポインタから
    // `type` が示すバイト数（ここでは `u32`/`bool` の 4/1 バイト）を
    // 即座に複製する（`crate::gemm::encode_dispatch` の
    // `setBytes_length_atIndex` と同じ「即時複製」契約。
    // objc2-metal 0.3.2 ドキュメント参照）。各ローカル変数は本呼び出し中
    // 生存しており、型・バイト数は `shaders/gemm.metal` の
    // `[[function_constant(n)]]` 宣言（`constant uint`/`constant bool`）と
    // 一致させている。index 6（`pad`）はイシュー #538 で追加した threadgroup
    // memory パディング幅（`TGP_PAD`。u32）で、既存 index 0〜5 と同じ即時
    // 複製契約に従う。
    unsafe {
        let bm = cfg.bm;
        constants.setConstantValue_type_atIndex(
            std::ptr::NonNull::from(&bm).cast(),
            MTLDataType::UInt,
            0,
        );
        let bn = cfg.bn;
        constants.setConstantValue_type_atIndex(
            std::ptr::NonNull::from(&bn).cast(),
            MTLDataType::UInt,
            1,
        );
        let bk = cfg.bk;
        constants.setConstantValue_type_atIndex(
            std::ptr::NonNull::from(&bk).cast(),
            MTLDataType::UInt,
            2,
        );
        let wm = cfg.wm;
        constants.setConstantValue_type_atIndex(
            std::ptr::NonNull::from(&wm).cast(),
            MTLDataType::UInt,
            3,
        );
        let wn = cfg.wn;
        constants.setConstantValue_type_atIndex(
            std::ptr::NonNull::from(&wn).cast(),
            MTLDataType::UInt,
            4,
        );
        let staged = cfg.staged;
        constants.setConstantValue_type_atIndex(
            std::ptr::NonNull::from(&staged).cast(),
            MTLDataType::Bool,
            5,
        );
        let pad = cfg.pad;
        constants.setConstantValue_type_atIndex(
            std::ptr::NonNull::from(&pad).cast(),
            MTLDataType::UInt,
            6,
        );
    }

    let func = library
        .newFunctionWithName_constantValues_error(&name, &constants)
        .map_err(|err| MetalError::LibraryCompilation {
            message: err.localizedDescription().to_string(),
        })?;
    device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|err| MetalError::PipelineCreation {
            message: err.localizedDescription().to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-2 前提 (b)「Metal は `mathFloatingPointFunctions=Precise` 明示」
    /// の契約テスト（TASK-2.2c・#55）。`MTLCompileOptions::new()` は GPU
    /// デバイスを介さない純粋なオブジェクト生成のため `#[ignore]` は付けず、
    /// macOS 実機上の `cargo test -p backend-metal`（`--ignored` なし）で
    /// 実行する。
    ///
    /// **CI（self-hosted・Linux）では実行されない**点に注意: 本ファイル
    /// （`pipeline` モジュール）は `lib.rs` の `#[cfg(target_os = "macos")]`
    /// によりそもそも Linux 上ではコンパイル対象外になる。CI の唯一の
    /// macOS 経路である `make build-cross`（TASK-2.1b）は
    /// `cargo build --workspace --locked --target aarch64-apple-darwin`
    /// であり `--all-targets` を含まないため、test ターゲット自体をビルド
    /// しない。実機 CI 整備（TASK-1.8e・#42）までは、既定値
    /// （`MathMode`/`MathFloatingPointFunctions` とも既定 `Fast`）への
    /// 意図しない退行を検出できるのは macOS 実機での手動実行時のみ。
    #[test]
    fn compile_options_pins_safe_math_mode_and_precise_functions() {
        let options = compile_options();
        assert_eq!(options.mathMode(), MTLMathMode::Safe);
        assert_eq!(
            options.mathFloatingPointFunctions(),
            MTLMathFloatingPointFunctions::Precise
        );
    }
}
