//! `ScalarUnaryOp`／`ScalarBinaryOp` の MSL カーネルソース生成
//! （式テンプレート。イシュー #1707・親 #1636・祖 #1592）。
//!
//! CUDA 側 `backend-cuda::kernels_scalar_op`（イシュー #1700）の Metal
//! 対応版。`elementwise.rs::ELEMENTWISE_MSL_SRC`（固定 7 カーネルの
//! 静的文字列）とは異なり、本モジュールは `ScalarUnaryOp`／
//! `ScalarBinaryOp`（`tensor-core::scalar_op`）の任意 kind から
//! `crate::pipeline::compile_source` に渡す MSL ソースを実行時に生成
//! する。1 kind = 1 独立コンパイル単位（関数名は
//! `scalar_unary_<kind_name>`／`scalar_binary_<kind_name>`。`kind_name`
//! は [`ScalarOpKind::kind_name`]（ペイロード値を含まない安定文字列）
//! を使う。キャッシュキー・カーネル関数名がこの `kind_name` のみに
//! 依存する契約は CUDA 側と同一（`kernels_scalar_op.rs` モジュール doc
//! 「#1635 への申し送り」参照）。
//!
//! # forward 数式の正
//!
//! `tensor-core::scalar_op::{ScalarUnaryOp, ScalarBinaryOp}::apply`
//! （`scalar_op.rs` モジュール doc「forward 数式の単一情報源」参照）が
//! forward 数式の単一情報源であり、本モジュールの `unary_expr`／
//! `binary_expr` は crate 境界（Rust と MSL という別言語）のため
//! やむを得ずその MSL 版の意図的複製である
//! （`docs/scalar-op-dispatch-design.md` §9・CUDA 側
//! `kernels_scalar_op.rs` と同型の事情）。
//!
//! # コンパイルオプションと数値契約
//!
//! [`crate::pipeline::compile_source`] は必ず `MathMode::Safe` +
//! `MathFloatingPointFunctions::Precise`（[`crate::pipeline::
//! compile_options`]）を適用する唯一の関数であり、本モジュールが生成する
//! ソースもこの関数だけを経由してコンパイルする契約とする（迂回禁止。
//! `pipeline.rs` モジュール冒頭コメント「2 経路で確実に同一適用」参照）。
//! この既定下で `+ - * /` は correctly rounded（IEEE 754 準拠）であり、
//! `Sub`（`-`）・`Div`（`/`）はホスト `f32` 演算と bit 同一になる想定。
//! `sqrt` は `metal::precise::sqrt` を明示使用する（`fast::` 名前空間・
//! `rsqrt` 系近似 intrinsic は使わない）ことで correctly rounded を保証し、
//! `Sqrt` もホスト `f32::sqrt` と bit 同一になる想定（`elementwise.metal`
//! の `metal::precise::exp`／`metal::precise::tanh` と同じ「コンパイル
//! オプションだけに委ねず呼び出し側でも明示する」方針）。`Pow` は
//! `metal::precise::pow` を使うが、MSL 数値準拠仕様上 `pow` は
//! correctly rounded を保証されない（最大 16 ulp。CUDA 側 `powf` と同じ
//! 扱い）ため bit 同一を主張せず REQ-2 複合判定のみで検証する。
//!
//! `Neg`（`-x`）・`Abs`（`metal::fabs(x)`）は算術・選択のみのため
//! ホスト `f32` 演算と bit 同一になる想定（`NaN` は payload が処理系
//! 依存のためクラス一致で検証する）。超越関数系（`Log`／`Log2`／
//! `Log10`／`Sin`／`Cos`／`Tan`）は `metal::precise::` 名前空間を明示
//! 使用する（`fast::` 近似 intrinsic は使わない）が、MSL 数値準拠仕様
//! （`.claude/skills/apple-silicon/references/msl/
//! numerical-compliance.md` Table 8.1）は precise 変種を最大 4 ulp と
//! 規定し correctly rounded を保証しないため、`Pow` と同様 bit 同一を
//! 主張せず REQ-2 複合判定のみで検証する（ホスト libm との ulp 一致は
//! 保証されない）。
//!
//! MSL 仕様は subnormal（非正規化数）の flush-to-zero を許容するため、
//! `Neg`／`Abs` の bit 同一検証に使う乱数入力は `[-1, 1)`（subnormal
//! 非到達域）とする。将来 Mac 実機で subnormal 起因の bit 差異が判明
//! した場合の対処は「subnormal 限定でクラス一致へ切り替える」であり、
//! tolerance 定数の変更ではない（`.claude/rules/coding-rust.md` の
//! 許容誤差はユーザー承認必須のポリシー除外対象）。
//!
//! # ペイロード seam（#1709 `Clamp` 向け申し送り。本イシューでは実装しない）
//!
//! CUDA 側 `kernels_scalar_op::UnaryPayload`（`Clamp` の `min`/`max` を
//! カーネル起動引数として渡す設計）と同型の拡張余地を残す:
//! ペイロードを追加する場合は `masked_fill`（`elementwise.rs::
//! run_binary_scalar`・`shaders/elementwise.metal::ew_masked_fill_f32`）
//! と同様に `numel` の後ろへ `setBytes_length_atIndex` で渡し（`constant
//! float& p0 [[buffer(3)]]` 等）、ソース文字列・キャッシュキー・関数名
//! には値を埋め込まない。本イシュー（#1707）が対象とする 4 kind
//! （`Sub`／`Div`／`Pow`／`Sqrt`）はいずれもペイロードを持たないため、
//! この seam は未使用のまま将来（#1709）へ引き継ぐ。
//!
//! # スコープ
//!
//! [`ScalarBinaryOp::Sub`]／[`ScalarBinaryOp::Div`]／[`ScalarBinaryOp::Pow`]
//! ・[`ScalarUnaryOp::Sqrt`]（#1707）に加え、[`ScalarUnaryOp::Neg`]／
//! [`ScalarUnaryOp::Abs`]／[`ScalarUnaryOp::Log`]／[`ScalarUnaryOp::Log2`]／
//! [`ScalarUnaryOp::Log10`]／[`ScalarUnaryOp::Sin`]／[`ScalarUnaryOp::Cos`]／
//! [`ScalarUnaryOp::Tan`]（超越関数系 8 kind。#1708）を実装する。
//! 比較 6 種＋`Clamp`（ペイロード実装）は #1709 の担当。他 kind は
//! `None`（未実装。呼び出し元 `ops::MetalBackendOps::scalar_unary`／
//! `scalar_binary` が `BackendError::Unsupported` を返しホスト参照実装
//! （`ScalarUnaryOp::apply`／`ScalarBinaryOp::apply`）へフォールバックする
//! 既存契約。`fandhe_ai_autodiff::grad::scalar_unary_with_fallback`／
//! `scalar_binary_with_fallback` 参照）。
//!
//! # cfg 方針
//!
//! `objc2` 系 FFI に一切触れない純粋な文字列生成ロジックのみで構成する
//! ため、`crate::generic_cache`／`crate::row_kernel` と同じ設計判断で
//! モジュール自体には `cfg(target_os = "macos")` を付けず、Linux（CI・
//! 本実装環境）でも `cargo test -p fandhe-ai-backend-metal` で生成結果を
//! 単体テストできるようにしてある。本番からの唯一の呼び出し元
//! （`context_cache.rs`・`ops.rs`）は `cfg(target_os = "macos")` 限定
//! （`lib.rs`）のため、非 macOS ビルド（`cargo build`／`cargo clippy` の
//! 非テストパス）では本モジュールの関数が「クレート内から到達不能」と
//! 判定され dead_code lint が誤検知する。`pub` へ広げず `cfg_attr` で
//! 対象を非 macOS ビルドに限定して抑制する（`row_kernel.rs`・
//! `generic_cache.rs` と同じ対処方針）。
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use fandhe_ai_tensor_core::{ScalarBinaryOp, ScalarOpKind, ScalarUnaryOp};

/// [`ScalarUnaryOp`] の MSL 式（変数名は `x`）。未実装 kind は `None`
/// （呼び出し元がホスト参照実装へフォールバックする）。
fn unary_expr(op: ScalarUnaryOp) -> Option<&'static str> {
    match op {
        // `metal::precise::sqrt` は correctly rounded（モジュール doc
        // 「コンパイルオプションと数値契約」参照）。`fast::sqrt`／
        // `rsqrt` 系近似 intrinsic は使わない。
        ScalarUnaryOp::Sqrt => Some("metal::precise::sqrt(x)"),
        // `Neg`／`Abs` は算術・選択のみでホスト `f32` 演算（`-x`／
        // `f32::abs`）と bit 同一になる想定（モジュール doc「コンパイル
        // オプションと数値契約」参照）。`Abs` は `metal::fabs` を明示
        // 使用する（`abs` は整数オーバーロードとの曖昧性を避けるため。
        // CUDA 側 `kernels_scalar_op.rs::unary_expr` の `fabsf` と同じ
        // 判断）。
        ScalarUnaryOp::Neg => Some("-x"),
        ScalarUnaryOp::Abs => Some("metal::fabs(x)"),
        // 超越関数系（`Log`／`Log2`／`Log10`／`Sin`／`Cos`／`Tan`）は
        // `metal::precise::` 名前空間を明示使用する（`fast::` 近似
        // intrinsic は使わない。モジュール doc「コンパイルオプションと
        // 数値契約」参照）。MSL 数値準拠仕様（`.claude/skills/
        // apple-silicon/references/msl/numerical-compliance.md` Table
        // 8.1）は precise 変種を最大 4 ulp と規定し correctly rounded を
        // 保証しないため、`Pow` と同じく bit 同一を主張せず REQ-2
        // 複合判定のみで検証する。
        ScalarUnaryOp::Log => Some("metal::precise::log(x)"),
        ScalarUnaryOp::Log2 => Some("metal::precise::log2(x)"),
        ScalarUnaryOp::Log10 => Some("metal::precise::log10(x)"),
        ScalarUnaryOp::Sin => Some("metal::precise::sin(x)"),
        ScalarUnaryOp::Cos => Some("metal::precise::cos(x)"),
        ScalarUnaryOp::Tan => Some("metal::precise::tan(x)"),
        _ => None,
    }
}

/// [`ScalarBinaryOp`] の MSL 式（変数名は `a_v`／`b_v`）。未実装 kind は
/// `None`。
fn binary_expr(op: ScalarBinaryOp) -> Option<&'static str> {
    match op {
        ScalarBinaryOp::Sub => Some("a_v - b_v"),
        ScalarBinaryOp::Div => Some("a_v / b_v"),
        // `metal::precise::pow` は超越関数（最大 16 ulp。correctly
        // rounded を保証しない）。REQ-2 複合判定のみで検証する
        // （モジュール doc「コンパイルオプションと数値契約」参照）。
        ScalarBinaryOp::Pow => Some("metal::precise::pow(a_v, b_v)"),
        _ => None,
    }
}

/// カーネル関数名（`compile_source`／`make_pipeline`／キャッシュキーへ
/// 渡す。`kind_name()` のみに依存しペイロード値を含まない。モジュール
/// doc「ペイロード seam」参照）。
pub(crate) fn unary_function_name(op: ScalarUnaryOp) -> String {
    format!("scalar_unary_{}", op.kind_name())
}

/// [`unary_function_name`] の 2 項版。
pub(crate) fn binary_function_name(op: ScalarBinaryOp) -> String {
    format!("scalar_binary_{}", op.kind_name())
}

/// `op` の単項カーネルソースを生成する（未実装 kind は `None`）。
///
/// バッファ index は `shaders/elementwise.metal::ew_relu_f32` 等の
/// 単項カーネルと完全一致させる（`a=0, out=1, numel=2`）ことで、
/// `elementwise.rs::encode_unary_dispatch`（既存の単項ディスパッチ
/// エンコーダ）をそのまま再利用できるようにする。
///
/// REQ-8（`.claude/rules/coding-rust.md`）: `if (idx < numel)` の手動
/// 境界チェックを維持する（`shaders/elementwise.metal` と同じ理由。1
/// スレッド = 1 要素の 1 次元グリッドで末尾スレッドが `numel` を
/// 超えうるため）。
pub(crate) fn unary_kernel_source(op: ScalarUnaryOp) -> Option<String> {
    let expr = unary_expr(op)?;
    let name = unary_function_name(op);
    Some(format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {{
    if (idx < numel) {{
        float x = a[idx];
        out[idx] = {expr};
    }}
}}
"#
    ))
}

/// `op` の 2 項カーネルソースを生成する（[`unary_kernel_source`] と同型。
/// バッファ index は `ew_add_f32` 等と完全一致（`a=0, b=1, out=2,
/// numel=3`）。未実装 kind は `None`）。
pub(crate) fn binary_kernel_source(op: ScalarBinaryOp) -> Option<String> {
    let expr = binary_expr(op)?;
    let name = binary_function_name(op);
    Some(format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {{
    if (idx < numel) {{
        float a_v = a[idx];
        float b_v = b[idx];
        out[idx] = {expr};
    }}
}}
"#
    ))
}

#[cfg(test)]
mod tests {
    //! 実機不要: 生成ソースが REQ-8 の境界チェック・buffer index 契約を
    //! 満たし、未実装 kind は `None` を返すことをホスト側のみで固定する
    //! 回帰テスト（CUDA 側 `kernels_scalar_op.rs` の「静的ソース内容
    //! 検査」と同型）。

    use super::*;

    #[test]
    fn sqrt_source_includes_bounds_check_and_precise_sqrt() {
        let src = unary_kernel_source(ScalarUnaryOp::Sqrt).expect("Sqrt must be implemented");
        assert!(src.contains("if (idx < numel)"));
        assert!(src.contains("kernel void scalar_unary_sqrt("));
        assert!(src.contains("metal::precise::sqrt("));
        assert!(!src.contains("fast::sqrt("));
        assert!(!src.contains("rsqrt("));
    }

    #[test]
    fn sqrt_source_declares_expected_buffer_indices() {
        let src = unary_kernel_source(ScalarUnaryOp::Sqrt).expect("Sqrt must be implemented");
        assert!(src.contains("[[buffer(0)]]"));
        assert!(src.contains("[[buffer(1)]]"));
        assert!(src.contains("[[buffer(2)]]"));
        assert!(!src.contains("[[buffer(3)]]"));
    }

    #[test]
    fn implemented_binary_kinds_include_bounds_check_and_expected_buffer_indices() {
        for op in [
            ScalarBinaryOp::Sub,
            ScalarBinaryOp::Div,
            ScalarBinaryOp::Pow,
        ] {
            let src = binary_kernel_source(op).expect("must be implemented");
            assert!(src.contains("if (idx < numel)"));
            assert!(src.contains(&format!("kernel void {}(", binary_function_name(op))));
            assert!(src.contains("[[buffer(0)]]"));
            assert!(src.contains("[[buffer(1)]]"));
            assert!(src.contains("[[buffer(2)]]"));
            assert!(src.contains("[[buffer(3)]]"));
        }
    }

    #[test]
    fn pow_source_uses_precise_pow() {
        let src = binary_kernel_source(ScalarBinaryOp::Pow).expect("Pow must be implemented");
        assert!(src.contains("metal::precise::pow("));
    }

    #[test]
    fn sub_and_div_sources_use_plain_operators() {
        let sub_src = binary_kernel_source(ScalarBinaryOp::Sub).expect("Sub must be implemented");
        assert!(sub_src.contains("a_v - b_v"));
        let div_src = binary_kernel_source(ScalarBinaryOp::Div).expect("Div must be implemented");
        assert!(div_src.contains("a_v / b_v"));
    }

    #[test]
    fn unimplemented_unary_kinds_return_none() {
        // `Sqrt`（#1707）・超越関数系 8 kind（`Neg`／`Abs`／`Log`／
        // `Log2`／`Log10`／`Sin`／`Cos`／`Tan`。#1708）は実装済みになった
        // ため、番兵 kind を未実装のまま残る kind（`Relu`〈活性化系。
        // いずれの sub issue にも含まれない〉・`Clamp`〈#1709 が担当〉）
        // へ付け替える（残すと未実装 kind への `None` フォールバック
        // 契約の検証が消えてしまう）。
        assert!(unary_kernel_source(ScalarUnaryOp::Relu).is_none());
        assert!(unary_kernel_source(ScalarUnaryOp::Sigmoid).is_none());
        assert!(unary_kernel_source(ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 }).is_none());
    }

    /// 超越関数系 8 kind すべてが REQ-8 境界チェック・buffer index
    /// 契約・関数名を満たし、6 超越関数（`Log`／`Log2`／`Log10`／
    /// `Sin`／`Cos`／`Tan`）が `metal::precise::` 名前空間のみを使い
    /// `fast::` を使わないことを固定する（CUDA 側
    /// `kernels_scalar_op.rs` の「静的ソース内容検査」と同型）。
    ///
    /// `metal::precise::log(x)` 自体が部分文字列 `"log("` を含むため、
    /// 「裸の呼び出しがない」検査は先頭スペース付きパターン
    /// （`" log("` 等）で行う（`!contains("log(")` では `precise::log(`
    /// にも誤反応してしまう）。
    #[test]
    fn transcendental_unary_kinds_include_bounds_check_and_use_precise_msl() {
        let unary_kinds = [
            ScalarUnaryOp::Neg,
            ScalarUnaryOp::Abs,
            ScalarUnaryOp::Log,
            ScalarUnaryOp::Log2,
            ScalarUnaryOp::Log10,
            ScalarUnaryOp::Sin,
            ScalarUnaryOp::Cos,
            ScalarUnaryOp::Tan,
        ];
        for op in unary_kinds {
            let src =
                unary_kernel_source(op).unwrap_or_else(|| panic!("{op:?} must be implemented"));
            assert!(src.contains("if (idx < numel)"));
            assert!(src.contains(&format!("kernel void {}(", unary_function_name(op))));
            assert!(src.contains("[[buffer(0)]]"));
            assert!(src.contains("[[buffer(1)]]"));
            assert!(src.contains("[[buffer(2)]]"));
            assert!(!src.contains("[[buffer(3)]]"));
        }

        let neg_src = unary_kernel_source(ScalarUnaryOp::Neg).expect("Neg implemented");
        assert!(neg_src.contains("-x"));
        let abs_src = unary_kernel_source(ScalarUnaryOp::Abs).expect("Abs implemented");
        assert!(abs_src.contains("metal::fabs("));

        for (op, fn_name) in [
            (ScalarUnaryOp::Log, "log"),
            (ScalarUnaryOp::Log2, "log2"),
            (ScalarUnaryOp::Log10, "log10"),
            (ScalarUnaryOp::Sin, "sin"),
            (ScalarUnaryOp::Cos, "cos"),
            (ScalarUnaryOp::Tan, "tan"),
        ] {
            let src =
                unary_kernel_source(op).unwrap_or_else(|| panic!("{op:?} must be implemented"));
            assert!(
                src.contains(&format!("metal::precise::{fn_name}(")),
                "{op:?} source must call metal::precise::{fn_name}(): {src}"
            );
            assert!(
                !src.contains(&format!("fast::{fn_name}(")),
                "{op:?} source must not call fast::{fn_name}()"
            );
            // 先頭スペース付きパターンで裸呼び出し（`metal::` 修飾なし）
            // が無いことを確認する（`precise::log(` 自体が `"log("` を
            // 含むため `!contains("log(")` では書けない）。
            assert!(
                !src.contains(&format!(" {fn_name}(")),
                "{op:?} source must not call bare {fn_name}() without a namespace qualifier"
            );
        }
    }

    /// `Neg`／`Abs` のホスト参照値の符号規約を固定する（CUDA 側
    /// `kernels_scalar_op.rs` 相当の bit 契約テスト。`+0.0 == -0.0` が
    /// 真になる `assert_eq!` を避け `to_bits()` で比較する）。
    #[test]
    fn neg_and_abs_host_reference_matches_documented_bit_contract() {
        assert_eq!((-0.0f32).to_bits(), (-(0.0f32)).to_bits());
        assert_eq!((0.0f32).to_bits(), (-(-0.0f32)).to_bits());
        assert!((-f32::NAN).is_nan());
        assert_eq!((0.0f32).to_bits(), (-0.0f32).abs().to_bits());
        assert!(f32::NAN.abs().is_nan());
    }

    #[test]
    fn unimplemented_binary_kinds_return_none() {
        assert!(binary_kernel_source(ScalarBinaryOp::Add).is_none());
        assert!(binary_kernel_source(ScalarBinaryOp::Maximum).is_none());
        assert!(binary_kernel_source(ScalarBinaryOp::Gt).is_none());
    }

    #[test]
    fn function_names_are_kind_name_derived_and_stable() {
        assert_eq!(
            unary_function_name(ScalarUnaryOp::Sqrt),
            "scalar_unary_sqrt"
        );
        assert_eq!(
            binary_function_name(ScalarBinaryOp::Sub),
            "scalar_binary_sub"
        );
        assert_eq!(
            binary_function_name(ScalarBinaryOp::Div),
            "scalar_binary_div"
        );
        assert_eq!(
            binary_function_name(ScalarBinaryOp::Pow),
            "scalar_binary_pow"
        );
        assert_eq!(unary_function_name(ScalarUnaryOp::Neg), "scalar_unary_neg");
        assert_eq!(unary_function_name(ScalarUnaryOp::Abs), "scalar_unary_abs");
        assert_eq!(unary_function_name(ScalarUnaryOp::Log), "scalar_unary_log");
        assert_eq!(
            unary_function_name(ScalarUnaryOp::Log2),
            "scalar_unary_log2"
        );
        assert_eq!(
            unary_function_name(ScalarUnaryOp::Log10),
            "scalar_unary_log10"
        );
        assert_eq!(unary_function_name(ScalarUnaryOp::Sin), "scalar_unary_sin");
        assert_eq!(unary_function_name(ScalarUnaryOp::Cos), "scalar_unary_cos");
        assert_eq!(unary_function_name(ScalarUnaryOp::Tan), "scalar_unary_tan");
    }

    /// 生成ソースが payload 値を一切含まないこと（モジュール doc
    /// 「ペイロード seam」の前提: 本イシューの対象 4 kind はいずれも
    /// ペイロードを持たないため、`unary_kernel_source`／
    /// `binary_kernel_source` の戻り値は `op` に依存しないはず）。
    #[test]
    fn sqrt_source_is_payload_independent() {
        let src_a = unary_kernel_source(ScalarUnaryOp::Sqrt).expect("Sqrt implemented");
        let src_b = unary_kernel_source(ScalarUnaryOp::Sqrt).expect("Sqrt implemented");
        assert_eq!(src_a, src_b);
    }
}
