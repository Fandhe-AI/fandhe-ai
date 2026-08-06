//! MSL 実行時コンパイル・パイプライン構築（TASK-1.8b・#39）。
//!
//! [`crate::gemm`] のディスパッチ入口が呼ぶ土台層。`shaders/gemm.metal`
//! を `include_str!` で埋め込み、[`crate::context::MetalContext`] が保持
//! するデバイス上で `newLibraryWithSource_options_error` によりコンパイル
//! し、関数名から `MTLComputePipelineState` を構築する。#40（TASK-1.8c）
//! で simdgroup 版パイプラインを併設する際も本モジュールの
//! [`compile_library`]・[`make_pipeline`] を再利用できるようにする。
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
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLMathFloatingPointFunctions, MTLMathMode,
};

use crate::context::MtlDevice;
use crate::error::MetalError;

/// `shaders/gemm.metal` のソース（naive GEMM カーネルを含む）。
const GEMM_MSL_SRC: &str = include_str!("shaders/gemm.metal");

pub(crate) type MtlLibrary = ProtocolObject<dyn MTLLibrary>;
pub(crate) type MtlPipeline = ProtocolObject<dyn MTLComputePipelineState>;

/// `shaders/gemm.metal` を実行時コンパイルして `MTLLibrary` を返す。
///
/// [`crate::gemm::naive`] から、パイプライン構築（[`make_pipeline`]）の
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
/// を構築する。[`crate::gemm::naive`] から `"gemm_naive"` を指定して呼ば
/// れる。#40（TASK-1.8c）が `"gemm_simdgroup"` で再利用する想定。
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
