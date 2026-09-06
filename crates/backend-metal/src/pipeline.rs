//! MSL 実行時コンパイル・パイプライン構築（TASK-1.8b・#39）。
//!
//! [`crate::gemm`] のディスパッチ入口が呼ぶ土台層。`shaders/gemm.metal`
//! を `include_str!` で埋め込み、[`crate::context::MetalContext`] が保持
//! するデバイス上で `newLibraryWithSource_options_error` によりコンパイル
//! し、関数名から `MTLComputePipelineState` を構築する。#40（TASK-1.8c）
//! で [`crate::gemm::MetalGemm::new`] が本モジュールの `compile_gemm_library`・
//! `make_pipeline` を 3 回呼んで `gemm_tiled`・`gemm_simdgroup` の
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
use crate::layout::TransposePattern;
use crate::spec_source::GEMM_MSL_SRC;
use crate::tile::TileConfig;

pub(crate) type MtlLibrary = ProtocolObject<dyn MTLLibrary>;
pub(crate) type MtlPipeline = ProtocolObject<dyn MTLComputePipelineState>;

/// `gemm_simdgroup_tiled`（[`make_pipeline_with_constants`]）の実験的
/// bool ゲート 3 個（`SWIZZLE_ENABLED`〈index 7・#540〉・
/// `FINE_BARRIER_ENABLED`〈index 8・#809〉・`UNROLL_ACC_ENABLED`
/// 〈index 11・#1282〉）を束ねる構造体。3 個目（`unroll_acc_enabled`）を
/// 追加する際、[`make_pipeline_with_constants`] へ素朴に引数を 1 個
/// 増やすと `clippy::too_many_arguments`（`-D warnings`）に抵触するため、
/// `#[allow]` を安易に追加せず構造体へ束ねる方式で回避する
/// （`.claude/rules/coding-rust.md`「`#[allow]` の安易な追加で黙らせない」）。
/// 各フィールドの意味・A/B 計測用途は [`make_pipeline_with_constants`] の
/// 同名引数ドキュメンテーションコメント（以下）を参照。
#[derive(Clone, Copy, Debug)]
pub(crate) struct GemmGateConstants {
    pub(crate) swizzle_enabled: bool,
    pub(crate) fine_barrier_enabled: bool,
    /// `gemm_simdgroup_tiled` のアキュムレータ系ループへの条件付き
    /// loop unroll（イシュー #1282）ゲート。呼び出し元
    /// [`crate::gemm::MetalGemm::pipeline_for_tile`] が候補ごとに
    /// `crate::tile::unroll_acc_loops_for(candidate, instance_flag)` で
    /// 導出した実効値を渡す契約（インスタンスの opt-in フラグと候補の
    /// acc 積閾値の AND。`crate::tile::unroll_acc_loops_for` doc comment
    /// 参照）。`gemm_simdgroup_tiled_f16` は index 11 を参照しないため、
    /// `pipeline_for_tile_f16` が渡す値は無害な no-op（`swizzle_enabled`／
    /// `fine_barrier_enabled` と同じ扱い）。
    pub(crate) unroll_acc_enabled: bool,
}

/// `shaders/gemm.metal` を実行時コンパイルして `MTLLibrary` を返す。
///
/// [`crate::gemm::MetalGemm::new`] から、パイプライン構築（[`make_pipeline`]）の
/// 前段として呼ばれる。エラー時の `message` は `NSError` の
/// `localizedDescription`（構文エラー等の診断文字列）を保持する。
pub(crate) fn compile_gemm_library(device: &MtlDevice) -> Result<Retained<MtlLibrary>, MetalError> {
    compile_source(device, GEMM_MSL_SRC)
}

/// `src`（MSL ソーステキスト）を `device` 上でコンパイルし `MTLLibrary` を
/// 返す共通実装（イシュー #1288）。[`compile_gemm_library`]（本番既定の
/// function constant 経路。`GEMM_MSL_SRC` をそのまま渡す薄いラッパー）と
/// `crate::spec_source::specialized_gemm_source` が生成した候補固有
/// ソース（[`make_pipeline_source_specialized`]）の**両方**がこの関数を
/// 経由する契約とすることで、`compile_options()`（`MathMode::Safe` +
/// `MathFloatingPointFunctions::Precise`。本ファイル冒頭コメント参照）を
/// 2 経路で確実に同一適用する（ここが分岐すると丸め方針が経路依存になり
/// REQ-2 の複合判定が黙って壊れるため、単一の関数へ集約する設計判断）。
fn compile_source(device: &MtlDevice, src: &str) -> Result<Retained<MtlLibrary>, MetalError> {
    let ns_src = NSString::from_str(src);
    let options = compile_options();

    device
        .newLibraryWithSource_options_error(&ns_src, Some(&options))
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
/// で追加した threadgroup memory パディング幅）と `crate::tile::SWIZZLE_ENABLED`
/// （`SWIZZLE_ENABLED`。index 7。イシュー #540）を畳み込んだ状態でコンパイル・
/// パイプライン化する。[`crate::gemm::MetalGemm`] の構成キー → パイプライン
/// 遅延キャッシュから、構成ごとに 1 回だけ呼ばれる想定
/// （`newFunctionWithName_constantValues_error` は MSL コンパイラを呼ぶ
/// 比較的重い処理）。
///
/// `library` は [`compile_gemm_library`] が返す `shaders/gemm.metal` 全体の
/// ライブラリを再利用する（function constant はコンパイル済みライブラリ
/// から関数を「特殊化」する API であり、ソース側の再コンパイルは不要）。
///
/// `swizzle_enabled`（index 7 の function constant 値。イシュー #746）は
/// 呼び出し元 [`crate::gemm::MetalGemm`] インスタンスが保持する固定値
/// （`MetalGemm::new` は `crate::tile::SWIZZLE_ENABLED`〈既定 `false`〉を
/// 渡し、`MetalGemm::new_with_swizzle` はベンチ用途で任意値を渡せる。
/// `crate::gemm::gemm.rs` 冒頭のフィールドコメント参照）をそのまま特殊化に
/// 使う。以前はここで `crate::tile::SWIZZLE_ENABLED` を直接参照していたが、
/// 同一プロセス内で base（off）/head（on）の 2 インスタンスを構築し
/// interleaved に A/B 計測する運用（`docs/perf/metal-gemm-tgid-swizzle-ab.md`）
/// のため引数化した（CUDA 側 `CudaMmaGemm::new_with_swizzle` と同型の設計。
/// `crates/backend-cuda/examples/gemm_mma_swizzle_bench.rs` 参照）。
///
/// `fine_barrier_enabled`（index 8 の function constant 値。イシュー #809）は
/// `swizzle_enabled` と同じ設計判断で `crate::gemm::MetalGemm` インスタンスが
/// 保持する固定値を伝播する（`MetalGemm::new` は `crate::tile::
/// FINE_BARRIER_ENABLED`〈既定 `false`〉を渡し、`MetalGemm::new_with_fine_barrier`
/// はベンチ用途で任意値を渡せる）。
///
/// `unroll_acc_enabled`（index 11 の function constant 値。イシュー
/// #1282）は `gemm_simdgroup_tiled` のアキュムレータ系ループへの条件付き
/// loop unroll ゲート。呼び出し元 `crate::gemm::MetalGemm::pipeline_for_tile`
/// が候補（フォールバック chain 巡回中の実際の `TileConfig`）ごとに
/// `crate::tile::unroll_acc_loops_for` で導出した実効値を渡す。`swizzle_enabled`／
/// `fine_barrier_enabled`／`unroll_acc_enabled` の 3 個は
/// [`GemmGateConstants`] へ束ねて渡す（`clippy::too_many_arguments`
/// 回避。同構造体のドキュメンテーションコメント参照）。
///
/// `pattern`（index 9/10 の `TRANS_A`/`TRANS_B`。イシュー #1138）は
/// `gemm_simdgroup_tiled`（NT/TN/TT 転置ロード拡張）のみが参照する。
/// `gemm_simdgroup_tiled_f16` など未参照の関数へ特殊化する際も
/// `swizzle_enabled`/`fine_barrier_enabled` と同じ理由で無害な no-op
/// となるため、[`crate::gemm::MetalGemm::pipeline_for_tile_f16`] は常に
/// [`TransposePattern::Nn`] を渡す契約（呼び出し側コメント参照）。
pub(crate) fn make_pipeline_with_constants(
    device: &MtlDevice,
    library: &MtlLibrary,
    function_name: &'static str,
    cfg: TileConfig,
    gates: GemmGateConstants,
    pattern: TransposePattern,
) -> Result<Retained<MtlPipeline>, MetalError> {
    let GemmGateConstants {
        swizzle_enabled,
        fine_barrier_enabled,
        unroll_acc_enabled,
    } = gates;
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
        let pad = cfg.pad();
        constants.setConstantValue_type_atIndex(
            std::ptr::NonNull::from(&pad).cast(),
            MTLDataType::UInt,
            6,
        );
        // threadgroup ID スウィズル（イシュー #540）のゲート。本番経路
        // （`MetalGemm::new`）は既定 `false`（`crate::tile::SWIZZLE_ENABLED`）
        // で、実機未検証の間は `shaders/gemm.metal` 側を恒等変換のまま動作
        // させる（PR #661 codex-review 指摘: 未検証のまま本番経路へ無条件
        // 適用しない）。イシュー #746 で本関数の引数へ格上げしたのは、A/B
        // 計測用の `MetalGemm::new_with_swizzle` インスタンスが `true` を
        // 渡せるようにするため（上記関数ドキュメンテーションコメント参照）。
        // `crate::gemm::encode_dispatch_tiled` の grid 計算も同じ
        // `swizzle_enabled` 値で分岐させ、シェーダ側の tgid 変換と grid
        // 形状を同期させる契約（`MetalGemm` が両呼び出しへ同一フィールド値
        // を伝播する）。index 7（`TGP_PAD`〈#538・index 6〉の直後）を
        // 割り当てる。
        constants.setConstantValue_type_atIndex(
            std::ptr::NonNull::from(&swizzle_enabled).cast(),
            MTLDataType::Bool,
            7,
        );
        // simdgroup 細粒度同期（イシュー #809）のゲート。`gemm_simdgroup_tiled`
        // の staged 経路のみが参照する定数だが、`gemm_simdgroup_tiled_f16`
        // など未参照の関数へ特殊化する際も値の設定自体は無害（Metal は
        // 関数が実際に参照しない function constant への値設定を許容する）。
        // index は SWIZZLE_ENABLED（index 7）の直後の index 8。
        let fine_barrier = fine_barrier_enabled;
        constants.setConstantValue_type_atIndex(
            std::ptr::NonNull::from(&fine_barrier).cast(),
            MTLDataType::Bool,
            8,
        );
        // 転置ロードゲート（イシュー #1138）。index は FINE_BARRIER_ENABLED
        // （index 8）の直後の 9/10（`shaders/gemm.metal` 冒頭 TRANS_A/TRANS_B
        // 宣言と 1:1 対応。`tests/shader_source_evidence.rs` が index を
        // 含めて固定する）。
        let trans_a = matches!(pattern, TransposePattern::Tn | TransposePattern::Tt);
        constants.setConstantValue_type_atIndex(
            std::ptr::NonNull::from(&trans_a).cast(),
            MTLDataType::Bool,
            9,
        );
        let trans_b = matches!(pattern, TransposePattern::Nt | TransposePattern::Tt);
        constants.setConstantValue_type_atIndex(
            std::ptr::NonNull::from(&trans_b).cast(),
            MTLDataType::Bool,
            10,
        );
        // 条件付き loop unroll ゲート（イシュー #1282）。`gemm_simdgroup_tiled`
        // のアキュムレータ系ループ 6 ブロックが `if (UNROLL_ACC_ENABLED)` で
        // 参照する。`gemm_simdgroup_tiled_f16` は参照しないため、
        // `pipeline_for_tile_f16` からの呼び出しでは無害な no-op
        // （`swizzle_enabled`/`fine_barrier_enabled` と同じ扱い）。index は
        // TRANS_B（index 10）の直後の 11（`shaders/gemm.metal` 冒頭
        // UNROLL_ACC_ENABLED 宣言と 1:1 対応。`tests/shader_source_evidence.rs`
        // が index を含めて固定する）。
        let unroll_acc = unroll_acc_enabled;
        constants.setConstantValue_type_atIndex(
            std::ptr::NonNull::from(&unroll_acc).cast(),
            MTLDataType::Bool,
            11,
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

/// `gemm_simdgroup_tiled` のソーステキスト特殊化経路（イシュー #1288。
/// E2 試作）。[`make_pipeline_with_constants`]（function constant 経路。
/// 本番既定）に対し、本関数は `crate::spec_source::specialized_gemm_source`
/// が候補固有の `#define GEMM_SPEC_*` を前置したソースを**候補ごとに
/// 再コンパイル**する（`newFunctionWithName_constantValues_error` の
/// function constant 値設定を経由しないため `unsafe` を一切使わない）。
///
/// `library` を再利用せず毎回 [`compile_source`] を呼ぶ点が
/// [`make_pipeline_with_constants`] との構造上の違い（function constant は
/// 既存ライブラリから関数を「特殊化」する API だが、ソーステキスト特殊化は
/// プリプロセッサマクロの展開結果が候補ごとに異なる別ソースになるため、
/// ライブラリ自体を候補ごとに構築する必要がある）。呼び出し元
/// [`crate::gemm::MetalGemm::pipeline_for_tile`] は候補（フォールバック
/// chain 巡回中の `TileConfig`）ごとに構築したパイプラインを
/// `tiled_spec_cache`（function constant 経路の `tiled_cache` とは独立の
/// フィールド）へキャッシュする契約のため、本関数はコンパイル済み
/// `MtlLibrary` を保持しない（呼び出し元がパイプラインのみキャッシュする
/// 設計。ライブラリの重複コンパイルは候補が有限個〈`tile::CANDIDATES`〉の
/// 遅延構築 1 回限りのコストとして許容する。性能実測は行わない
/// `docs/perf/metal-gemm-n4096-kernel-gap.md` §8）。
///
/// `gates`/`pattern` の意味は [`make_pipeline_with_constants`] の同名
/// 引数と同一（`crate::spec_source::SpecializationParams::new` が
/// `pattern` から `trans_a`/`trans_b` を導出する）。
pub(crate) fn make_pipeline_source_specialized(
    device: &MtlDevice,
    function_name: &'static str,
    cfg: TileConfig,
    gates: GemmGateConstants,
    pattern: TransposePattern,
) -> Result<Retained<MtlPipeline>, MetalError> {
    let GemmGateConstants {
        swizzle_enabled,
        fine_barrier_enabled,
        unroll_acc_enabled,
    } = gates;
    let params = crate::spec_source::SpecializationParams::new(
        cfg,
        swizzle_enabled,
        fine_barrier_enabled,
        unroll_acc_enabled,
        pattern,
    );
    let src = crate::spec_source::specialized_gemm_source(&params);
    let library = compile_source(device, &src)?;
    make_pipeline(device, &library, function_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-2 前提 (b)「Metal は `mathFloatingPointFunctions=Precise` 明示」
    /// の契約テスト（TASK-2.2c・#55）。`MTLCompileOptions::new()` は GPU
    /// デバイスを介さない純粋なオブジェクト生成のため `#[ignore]` は付けず、
    /// macOS 実機上の `cargo test -p fandhe-ai-backend-metal`（`--ignored` なし）で
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
