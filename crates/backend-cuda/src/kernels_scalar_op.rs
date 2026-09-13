//! `ScalarUnaryOp`／`ScalarBinaryOp` の CUDA C カーネルソース生成
//! （式テンプレート。イシュー #1700・親 #1635・祖 #1592）。
//!
//! `kernels_elementwise.rs`（固定 7 カーネルの静的文字列）とは異なり、
//! 本モジュールは `ScalarUnaryOp`／`ScalarBinaryOp`（`tensor-core::
//! scalar_op`）の任意 kind から NVRTC 実行時コンパイル用のカーネル
//! ソースを実行時に生成する。1 kind = 1 独立コンパイル単位（関数名は
//! `scalar_unary_<kind_name>`／`scalar_binary_<kind_name>`。`kind_name`
//! は [`ScalarOpKind::kind_name`]（ペイロード値を含まない安定文字列）
//! を使う。`scalar_op.rs` モジュール doc「#1635 への申し送り」の契約
//! どおり NVRTC キャッシュキー・カーネル関数名はこの `kind_name` のみに
//! 依存し、`f32` ペイロード（将来 `Clamp` 等が追加する場合）はソース
//! 文字列へ埋め込まずカーネル起動引数として渡す設計とする）。
//!
//! # forward 数式の正
//!
//! `tensor-core::scalar_op::{ScalarUnaryOp, ScalarBinaryOp}::apply`
//! （`scalar_op.rs` モジュール doc「forward 数式の単一情報源」参照）が
//! forward 数式の単一情報源であり、本モジュールの `unary_expr`／
//! `binary_expr` は crate 境界（Rust と CUDA C という別言語）のため
//! やむを得ずその CUDA C 版の意図的複製である
//! （`docs/scalar-op-dispatch-design.md` §9 と同型の事情）。
//!
//! # NVRTC 既定オプションと数値契約
//!
//! `nvrtc::compile_ptx` は `prec-div`／`prec-sqrt` が既定 true
//! （fast-math 未指定）で IEEE 754 丸めになる（`kernels_rmsnorm.rs`
//! 冒頭コメント「近似 intrinsic `rsqrtf` を使わない理由」で実機確認済み）。
//! よって `Sub`（`-`）・`Div`（`/`）・`Sqrt`（`sqrtf`。`rsqrtf` は使わない）・
//! `Neg`（`-x`。符号ビット反転）・`Abs`（`fabsf`。符号ビットクリア）
//! はホスト `f32` 演算と bit 同一になる想定（`elementwise_matches_cpu_
//! across_ops` の `add`／`mul`／`relu` と同じ扱い。`NaN` 入力は payload
//! が処理系依存のためクラス一致で検証する）。`Pow`（`powf`）・
//! `Log`／`Log2`／`Log10`（`logf`／`log2f`／`log10f`）・`Sin`／`Cos`／
//! `Tan`（`sinf`／`cosf`／`tanf`）は超越関数（CUDA libm の ulp 誤差が
//! ホスト側 glibc libm と一致する保証がない）のため bit 同一を主張せず
//! REQ-2 複合判定のみで検証する（既存 `exp`／`tanh` と同じ扱い）。
//! 近似 intrinsic（`__logf`／`__sinf` 等の `__` プレフィックス付き）・
//! `double` 版（`log`／`sin` 等への暗黙昇格）は使わない。
//!
//! # スコープ
//!
//! [`ScalarBinaryOp::Sub`]／[`ScalarBinaryOp::Div`]／[`ScalarBinaryOp::Pow`]
//! （#1700）に加え、[`ScalarUnaryOp::Sqrt`]（#1700）・[`ScalarUnaryOp::Neg`]／
//! [`ScalarUnaryOp::Abs`]／[`ScalarUnaryOp::Log`]／[`ScalarUnaryOp::Log2`]／
//! [`ScalarUnaryOp::Log10`]／[`ScalarUnaryOp::Sin`]／[`ScalarUnaryOp::Cos`]／
//! [`ScalarUnaryOp::Tan`]（#1701）を実装する。他 kind は `None`（未実装。
//! 呼び出し元 `ops::CudaBackendOps::scalar_unary`／`scalar_binary` が
//! `BackendError::Unsupported` を返しホスト参照実装（`ScalarUnaryOp::apply`／
//! `ScalarBinaryOp::apply`）へフォールバックする既存契約。
//! `fandhe_ai_autodiff::grad::scalar_unary_with_fallback`／
//! `scalar_binary_with_fallback` 参照）。比較演算（`Gt`／`Ge`／`Lt`／
//! `Le`／`Eq`／`Ne`）＋`Clamp` は #1702 が同じ関数へ match arm を追加
//! する形で拡張する。`LeakyRelu`／`Elu`／`Softplus`／`PowScalar`
//! （ペイロードあり unary kind）・活性化 unary kind（`Relu`／`Exp`／
//! `Tanh`／`Sigmoid`／`Gelu`／`GeluTanh`／`Silu`／`Hardswish`）・
//! `Add`／`Mul`／`Maximum`／`Minimum`（比較・算術以外の残り binary kind）
//! はいずれの sub issue にも含まれず対象外のまま残る（`.claude/rules/
//! out-of-scope-tracking.md` 対象。必要なら別イシューで追跡）。

use fandhe_ai_tensor_core::{ScalarBinaryOp, ScalarOpKind, ScalarUnaryOp};

/// [`ScalarUnaryOp`] の CUDA C 式（変数名は `x`）。未実装 kind は
/// `None`（呼び出し元がホスト参照実装へフォールバックする）。
fn unary_expr(op: ScalarUnaryOp) -> Option<&'static str> {
    match op {
        ScalarUnaryOp::Sqrt => Some("sqrtf(x)"),
        // `Neg`／`Abs` はホスト `f32` 演算（`-x`／`f32::abs`）と bit
        // 同一（モジュール doc「NVRTC 既定オプションと数値契約」参照）。
        ScalarUnaryOp::Neg => Some("-x"),
        ScalarUnaryOp::Abs => Some("fabsf(x)"),
        // 超越関数系（`Log`／`Log2`／`Log10`／`Sin`／`Cos`／`Tan`）は
        // 単精度 libm 名のみを使う（`__` プレフィックス付き近似
        // intrinsic・`double` 版への暗黙昇格は使わない）。
        ScalarUnaryOp::Log => Some("logf(x)"),
        ScalarUnaryOp::Log2 => Some("log2f(x)"),
        ScalarUnaryOp::Log10 => Some("log10f(x)"),
        ScalarUnaryOp::Sin => Some("sinf(x)"),
        ScalarUnaryOp::Cos => Some("cosf(x)"),
        ScalarUnaryOp::Tan => Some("tanf(x)"),
        _ => None,
    }
}

/// [`ScalarBinaryOp`] の CUDA C 式（変数名は `a_v`／`b_v`）。未実装 kind
/// は `None`。
fn binary_expr(op: ScalarBinaryOp) -> Option<&'static str> {
    match op {
        ScalarBinaryOp::Sub => Some("a_v - b_v"),
        ScalarBinaryOp::Div => Some("a_v / b_v"),
        ScalarBinaryOp::Pow => Some("powf(a_v, b_v)"),
        _ => None,
    }
}

/// カーネル起動関数名（`load_function`／NVRTC キャッシュキーへ渡す。
/// `kind_name()` のみに依存しペイロード値を含まない。モジュール doc
/// 「#1635 への申し送り」参照）。
pub(crate) fn unary_function_name(op: ScalarUnaryOp) -> String {
    format!("scalar_unary_{}", op.kind_name())
}

/// [`unary_function_name`] の 2 項版。
pub(crate) fn binary_function_name(op: ScalarBinaryOp) -> String {
    format!("scalar_binary_{}", op.kind_name())
}

/// `op` の単項カーネルソースを生成する（未実装 kind は `None`）。
///
/// REQ-8（`.claude/rules/coding-rust.md`）: `if (idx < numel)` の手動
/// 境界チェックを維持する（`kernels_elementwise.rs` と同じ理由。1
/// スレッド = 1 要素でグリッドを `div_ceil` により切り上げ生成するため、
/// 末尾ブロックでは `idx` が `numel` を超えるスレッドが必ず発生する）。
pub(crate) fn unary_kernel_source(op: ScalarUnaryOp) -> Option<String> {
    let expr = unary_expr(op)?;
    let name = unary_function_name(op);
    Some(format!(
        r#"
extern "C" __global__ void {name}(
    const float* __restrict__ a,
    float* __restrict__ out,
    int numel)
{{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        float x = a[idx];
        out[idx] = {expr};
    }}
}}
"#
    ))
}

/// `op` の 2 項カーネルソースを生成する（[`unary_kernel_source`] と同型。
/// 未実装 kind は `None`）。
pub(crate) fn binary_kernel_source(op: ScalarBinaryOp) -> Option<String> {
    let expr = binary_expr(op)?;
    let name = binary_function_name(op);
    Some(format!(
        r#"
extern "C" __global__ void {name}(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int numel)
{{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
    //! 実機不要：生成ソースが REQ-8 の境界チェックを含み、未実装 kind は
    //! `None` を返すことをホスト側のみで固定する回帰テスト
    //! （`kernels_rmsnorm.rs::forward_kernels_do_not_use_approximate_rsqrtf`
    //! と同型の「静的ソース内容検査」）。

    use super::*;

    #[test]
    fn implemented_unary_kind_includes_bounds_check() {
        let src = unary_kernel_source(ScalarUnaryOp::Sqrt).expect("Sqrt must be implemented");
        assert!(src.contains("if (idx < numel)"));
        assert!(src.contains("scalar_unary_sqrt"));
        assert!(!src.contains("rsqrtf("));
    }

    #[test]
    fn implemented_binary_kinds_include_bounds_check() {
        for op in [
            ScalarBinaryOp::Sub,
            ScalarBinaryOp::Div,
            ScalarBinaryOp::Pow,
        ] {
            let src = binary_kernel_source(op).expect("must be implemented");
            assert!(src.contains("if (idx < numel)"));
            assert!(src.contains(&binary_function_name(op)));
        }
    }

    /// #1701 の対象 8 kind が REQ-8 境界チェックを含み、単精度 libm のみ
    /// を使う（近似 intrinsic・`double` 版への暗黙昇格を含まない）ことを
    /// 静的ソース検査で固定する（`kernels_rmsnorm.rs` の同型契約）。
    #[test]
    fn transcendental_unary_kinds_include_bounds_check_and_use_precise_libm() {
        for op in [
            ScalarUnaryOp::Neg,
            ScalarUnaryOp::Abs,
            ScalarUnaryOp::Log,
            ScalarUnaryOp::Log2,
            ScalarUnaryOp::Log10,
            ScalarUnaryOp::Sin,
            ScalarUnaryOp::Cos,
            ScalarUnaryOp::Tan,
        ] {
            let src = unary_kernel_source(op)
                .unwrap_or_else(|| panic!("{} must be implemented by #1701", op.kind_name()));
            assert!(src.contains("if (idx < numel)"));
            assert!(src.contains(&unary_function_name(op)));
            // 近似 intrinsic（`__` プレフィックス）は使わない。
            assert!(!src.contains("__logf("));
            assert!(!src.contains("__log2f("));
            assert!(!src.contains("__log10f("));
            assert!(!src.contains("__sinf("));
            assert!(!src.contains("__cosf("));
            assert!(!src.contains("__tanf("));
            // `double` 版（`logf` ではなく `log` 等）への暗黙昇格も
            // 使わない（`sinf(` 等の単精度名のみを許容）。
            assert!(!src.contains(" log("));
            assert!(!src.contains(" log2("));
            assert!(!src.contains(" log10("));
            assert!(!src.contains(" sin("));
            assert!(!src.contains(" cos("));
            assert!(!src.contains(" tan("));
            assert!(!src.contains(" fabs("));
        }
    }

    #[test]
    fn unimplemented_kinds_return_none() {
        // `Log` は #1701 で実装済みになったため、番兵 kind を未実装のまま
        // 残る kind（#1702 が担当する比較演算・活性化系）へ付け替える
        // （残すと未実装 kind への `None` フォールバック契約の検証が
        // 消えてしまう）。
        assert!(unary_kernel_source(ScalarUnaryOp::Relu).is_none());
        assert!(unary_kernel_source(ScalarUnaryOp::Sigmoid).is_none());
        assert!(binary_kernel_source(ScalarBinaryOp::Add).is_none());
        assert!(binary_kernel_source(ScalarBinaryOp::Maximum).is_none());
    }

    #[test]
    fn function_names_are_kind_name_derived_and_stable() {
        assert_eq!(
            unary_function_name(ScalarUnaryOp::Sqrt),
            "scalar_unary_sqrt"
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
    }

    /// `Neg`（`-x`）／`Abs`（`fabsf`）はホスト `f32` 演算と bit 同一に
    /// なる想定（モジュール doc 参照）。特殊値（`-0.0`／`NaN`）でも
    /// ホスト側の符号規約と一致することをここで明示しておく（NVRTC 実機
    /// 経由の parity テストは `tests/scalar_op_parity.rs` を参照）。
    #[test]
    fn neg_and_abs_host_reference_matches_documented_bit_contract() {
        assert_eq!(
            ScalarUnaryOp::Neg.apply(0.0_f32).to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(
            ScalarUnaryOp::Neg.apply(-0.0_f32).to_bits(),
            (0.0_f32).to_bits()
        );
        assert_eq!(
            ScalarUnaryOp::Abs.apply(-0.0_f32).to_bits(),
            (0.0_f32).to_bits()
        );
        assert!(ScalarUnaryOp::Neg.apply(f32::NAN).is_nan());
        assert!(ScalarUnaryOp::Abs.apply(f32::NAN).is_nan());
    }
}
