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
//! [`ScalarUnaryOp::Tan`]（#1701）・比較演算（[`ScalarBinaryOp::Gt`]／
//! [`Ge`]／[`Lt`]／[`Le`]／[`Eq`]／[`Ne`]）＋[`ScalarUnaryOp::Clamp`]
//! （#1702）を実装する。他 kind は `None`（未実装。呼び出し元
//! `ops::CudaBackendOps::scalar_unary`／`scalar_binary` が
//! `BackendError::Unsupported` を返しホスト参照実装（`ScalarUnaryOp::apply`／
//! `ScalarBinaryOp::apply`）へフォールバックする既存契約。
//! `fandhe_ai_autodiff::grad::scalar_unary_with_fallback`／
//! `scalar_binary_with_fallback` 参照）。[`ScalarUnaryOp::Gelu`]／
//! [`GeluTanh`](ScalarUnaryOp::GeluTanh)／[`Softplus`](ScalarUnaryOp::Softplus)
//! はイシュー #1713 で実装済み。`LeakyRelu`／`Elu`／`PowScalar`（他の
//! ペイロードあり unary kind）・`Relu`／`Exp`／`Tanh`／`Sigmoid`／
//! `Silu`／`Hardswish`（残る活性化 unary kind）・`Add`／`Mul`／
//! `Maximum`／`Minimum`（比較・算術以外の残り binary kind）はいずれの
//! sub issue にも含まれず対象外のまま残る（`.claude/rules/
//! out-of-scope-tracking.md` 対象。必要なら別イシューで追跡）。
//!
//! イシュー #1713 で GELU（誤差関数版・tanh 近似版）・Softplus を実装
//! した。`Gelu`／`GeluTanh` は超越関数（`erff`／`tanhf`）のためホスト
//! `f32` 演算・NVRTC 単精度 libm いずれも bit 同一は主張せず REQ-2
//! 複合判定のみで検証する（`Pow`／`Log` 系と同じ扱い）。`Softplus` は
//! `Clamp` と同型の 2 引数ペイロード（`beta`・`threshold`）を持つ最初の
//! 超越関数系 kind で、ホスト `ScalarUnaryOp::apply` の `ln_1p`
//! （桁落ち回避）形を CUDA C の `log1pf`／`expf` へ逐語で写す。
//!
//! `Clamp` は本モジュールで初めて `f32` ペイロードを持つ unary kind
//! （[`UnaryPayload`] 参照）。モジュール doc冒頭「#1635 への申し送り」の
//! 契約どおり、NVRTC キャッシュキー・カーネル関数名は `kind_name()`
//! （ペイロード非依存）のみに依存させ、ペイロード値はソース文字列へ
//! 埋め込まずカーネル起動引数として渡す（`unary_kernel_source` が
//! `numel` の後ろへ `float p0, float p1` を宣言し、
//! `elementwise.rs::run_unary`／`run_scalar_unary_f32` が
//! [`unary_payload`] の返す値をその順序で追加起動引数として渡す）。

use fandhe_ai_tensor_core::{ScalarBinaryOp, ScalarOpKind, ScalarUnaryOp};

/// [`ScalarUnaryOp`] の CUDA C 式（変数名は `x`。ペイロードあり kind は
/// `p0`／`p1` も使う。[`unary_payload`] 参照）。未実装 kind は
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
        // `ScalarUnaryOp::apply` の `Clamp` 分岐（`scalar_op.rs`）を
        // 逐語で写す（`isnan(x)` → `NaN` を伝播 / `p0 > p1`（min > max）
        // → 常に `p1`（max） / `x < p0` → `p0` / `x > p1` → `p1` /
        // それ以外 → `x`。`fminf`／`fmaxf`〈IEEE minNum/maxNum は非 NaN
        // 側を優先し `is_nan` 明示分岐と異なる〉は使わない。モジュール
        // doc「forward 数式の正」参照）。
        ScalarUnaryOp::Clamp { .. } => {
            Some("isnan(x) ? x : (p0 > p1 ? p1 : (x < p0 ? p0 : (x > p1 ? p1 : x)))")
        }
        // GELU（誤差関数版）: `ScalarUnaryOp::apply` の `Gelu` 分岐
        // （`gelu_erf` → `apply` が `f64` 精度の自作 `erf_f64` 近似を
        // 呼ぶ）とは異なり、CUDA 側は組み込み単精度 intrinsic `erff`
        // を使う（モジュール doc「forward 数式の正」参照。超越関数の
        // ため bit 同一は主張しない）。`1/sqrt(2)` は定数畳み込み。
        ScalarUnaryOp::Gelu => Some("0.5f * x * (1.0f + erff(x * 0.70710678118654752f))"),
        // GELU（tanh 近似版）: `gelu_tanh_approx`（`scalar_op.rs`）と
        // 同じ演算順序（`x + 0.044715*x^3` → `0.7978846*(...)` →
        // `tanhf` → `0.5*x*(1+...)`）を単精度 `tanhf` で逐語再現する。
        ScalarUnaryOp::GeluTanh => {
            Some("0.5f * x * (1.0f + tanhf(0.7978846f * (x + 0.044715f * x * x * x)))")
        }
        // Softplus: `ScalarUnaryOp::apply` の `Softplus` 分岐
        // （`x*beta > threshold` で恒等・それ以外は `ln_1p(exp(beta*x))
        // / beta`）を `log1pf`／`expf` で逐語再現する（`f32::ln_1p` の
        // 桁落ち回避特性〈PR #1686 是正〉を維持）。`p0`＝`beta`・
        // `p1`＝`threshold`（`unary_payload` 参照）。
        ScalarUnaryOp::Softplus { .. } => Some("(x * p0 > p1) ? x : (log1pf(expf(p0 * x)) / p0)"),
        _ => None,
    }
}

/// [`ScalarBinaryOp`] の CUDA C 式（変数名は `a_v`／`b_v`）。比較演算は
/// `bool_to_f32`（`scalar_op.rs`）と同じ `0.0f`／`1.0f` を返す（`docs/
/// scalar-op-dispatch-design.md` §3.2「bool 出力」契約）。未実装 kind
/// は `None`。
fn binary_expr(op: ScalarBinaryOp) -> Option<&'static str> {
    match op {
        ScalarBinaryOp::Sub => Some("a_v - b_v"),
        ScalarBinaryOp::Div => Some("a_v / b_v"),
        ScalarBinaryOp::Pow => Some("powf(a_v, b_v)"),
        ScalarBinaryOp::Gt => Some("(a_v > b_v) ? 1.0f : 0.0f"),
        ScalarBinaryOp::Ge => Some("(a_v >= b_v) ? 1.0f : 0.0f"),
        ScalarBinaryOp::Lt => Some("(a_v < b_v) ? 1.0f : 0.0f"),
        ScalarBinaryOp::Le => Some("(a_v <= b_v) ? 1.0f : 0.0f"),
        ScalarBinaryOp::Eq => Some("(a_v == b_v) ? 1.0f : 0.0f"),
        ScalarBinaryOp::Ne => Some("(a_v != b_v) ? 1.0f : 0.0f"),
        _ => None,
    }
}

/// [`ScalarUnaryOp`] のカーネル起動引数として渡す `f32` ペイロード
/// （ソース文字列へは埋め込まない。モジュール doc「スコープ」参照）。
/// `None` はペイロードなし kind（起動引数列は既存 `numel` までで不変＝
/// bit 同一）、`Two([p0, p1])` は 2 引数ペイロード kind（現状 `Clamp`
/// のみ）。
pub(crate) enum UnaryPayload {
    None,
    Two([f32; 2]),
}

impl UnaryPayload {
    /// カーネル起動引数として `numel` の後ろへ渡す順序どおりのスライス
    /// （空スライスは追加引数なし＝既存 kind と同じ起動引数列）。
    pub(crate) fn as_slice(&self) -> &[f32] {
        match self {
            Self::None => &[],
            Self::Two(v) => v,
        }
    }
}

/// `op` のカーネル起動ペイロードを返す（[`unary_kernel_source`] が
/// 宣言する `p0`／`p1` パラメータへ対応する値。呼び出し順は
/// `elementwise.rs::run_scalar_unary_f32` → `run_unary` が
/// `as_slice()` の順序で `numel` の後ろへ追加起動引数として渡す）。
pub(crate) fn unary_payload(op: ScalarUnaryOp) -> UnaryPayload {
    match op {
        ScalarUnaryOp::Clamp { min, max } => UnaryPayload::Two([min, max]),
        ScalarUnaryOp::Softplus { beta, threshold } => UnaryPayload::Two([beta, threshold]),
        _ => UnaryPayload::None,
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
    // ペイロードあり kind（`Clamp` 等）は `numel` の後ろへ `float p0,
    // float p1` を追加宣言する（モジュール doc「スコープ」参照）。
    // ペイロード値自体はここへ埋め込まず、常に固定パラメータ名
    // （`p0`／`p1`）のみを使うため、`kind_name()` が同じ限り payload
    // 値が異なってもソース文字列は完全一致する（NVRTC キャッシュキーが
    // payload 非依存であることの根拠。単体テスト
    // `clamp_source_declares_payload_params_and_omits_values` 参照）。
    let payload_params = match unary_payload(op) {
        UnaryPayload::None => String::new(),
        UnaryPayload::Two(_) => ", float p0, float p1".to_string(),
    };
    Some(format!(
        r#"
extern "C" __global__ void {name}(
    const float* __restrict__ a,
    float* __restrict__ out,
    int numel{payload_params})
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
            ScalarBinaryOp::Gt,
            ScalarBinaryOp::Ge,
            ScalarBinaryOp::Lt,
            ScalarBinaryOp::Le,
            ScalarBinaryOp::Eq,
            ScalarBinaryOp::Ne,
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
    fn comparison_ops_return_zero_or_one_literals() {
        // 比較 6 種は `bool_to_f32`（`scalar_op.rs`）と同じ `0.0f`／
        // `1.0f` を返す（`docs/scalar-op-dispatch-design.md` §3.2「bool
        // 出力」契約）。式に浮動小数点数以外の値（整数 `1`／`0` 等）が
        // 紛れていないことをソース内容で固定する。
        for op in [
            ScalarBinaryOp::Gt,
            ScalarBinaryOp::Ge,
            ScalarBinaryOp::Lt,
            ScalarBinaryOp::Le,
            ScalarBinaryOp::Eq,
            ScalarBinaryOp::Ne,
        ] {
            let src = binary_kernel_source(op).expect("must be implemented");
            assert!(src.contains("1.0f"), "{op:?} source must contain 1.0f");
            assert!(src.contains("0.0f"), "{op:?} source must contain 0.0f");
        }
    }

    #[test]
    fn clamp_source_declares_payload_params_and_omits_values() {
        let src = unary_kernel_source(ScalarUnaryOp::Clamp {
            min: 0.123,
            max: 4.567,
        })
        .expect("Clamp must be implemented");
        assert!(src.contains("float p0, float p1"));
        assert!(!src.contains("0.123"));
        assert!(!src.contains("4.567"));
        assert!(src.contains("if (idx < numel)"));
        assert!(src.contains("scalar_unary_clamp"));
    }

    #[test]
    fn clamp_source_is_payload_value_independent() {
        // NVRTC キャッシュキーが `kind_name()` のみに依存する契約
        // （`context_cache::cached_scalar_unary_kernel` doc 参照）の
        // 前提: 異なる payload 値でも生成ソースは完全一致する。
        let src_a = unary_kernel_source(ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 })
            .expect("Clamp must be implemented");
        let src_b = unary_kernel_source(ScalarUnaryOp::Clamp {
            min: -5.0,
            max: 5.0,
        })
        .expect("Clamp must be implemented");
        assert_eq!(src_a, src_b);
    }

    #[test]
    fn clamp_source_does_not_use_fminf_fmaxf() {
        // `fminf`／`fmaxf`（IEEE minNum/maxNum。非 NaN 側を優先）は
        // `ScalarUnaryOp::apply` の `Clamp` 分岐（明示 `is_nan` 分岐）と
        // 数値契約が異なるため使わない（`kernels_rmsnorm.rs` の
        // `rsqrtf` 不在テストと同型）。
        let src = unary_kernel_source(ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 })
            .expect("Clamp must be implemented");
        assert!(!src.contains("fminf("));
        assert!(!src.contains("fmaxf("));
    }

    #[test]
    fn payload_param_count_matches_source_for_all_implemented_unary_kinds() {
        // 実装済み unary kind すべてについて、ソース中の `float p`
        // パラメータ宣言数が `unary_payload(op).as_slice().len()` と
        // 一致することを固定する（引数個数の不一致はカーネル起動時
        // にしか露見しないため事前にホスト側で検出する）。
        for op in [
            ScalarUnaryOp::Sqrt,
            ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 },
            ScalarUnaryOp::Gelu,
            ScalarUnaryOp::GeluTanh,
            ScalarUnaryOp::Softplus {
                beta: 1.0,
                threshold: 20.0,
            },
        ] {
            let src = unary_kernel_source(op).expect("must be implemented");
            let declared = src.matches("float p").count();
            let expected = unary_payload(op).as_slice().len();
            assert_eq!(
                declared, expected,
                "{op:?}: declared payload params ({declared}) != unary_payload len ({expected})"
            );
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
            unary_function_name(ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 }),
            "scalar_unary_clamp"
        );
        assert_eq!(
            unary_function_name(ScalarUnaryOp::Gelu),
            "scalar_unary_gelu"
        );
        assert_eq!(
            unary_function_name(ScalarUnaryOp::GeluTanh),
            "scalar_unary_gelu_tanh"
        );
        assert_eq!(
            unary_function_name(ScalarUnaryOp::Softplus {
                beta: 1.0,
                threshold: 20.0
            }),
            "scalar_unary_softplus"
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
        assert_eq!(binary_function_name(ScalarBinaryOp::Gt), "scalar_binary_gt");
        assert_eq!(binary_function_name(ScalarBinaryOp::Ge), "scalar_binary_ge");
        assert_eq!(binary_function_name(ScalarBinaryOp::Lt), "scalar_binary_lt");
        assert_eq!(binary_function_name(ScalarBinaryOp::Le), "scalar_binary_le");
        assert_eq!(binary_function_name(ScalarBinaryOp::Eq), "scalar_binary_eq");
        assert_eq!(binary_function_name(ScalarBinaryOp::Ne), "scalar_binary_ne");
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

    /// GELU（誤差関数版・tanh 近似版）のソースが REQ-8 境界チェックを
    /// 含み、単精度 intrinsic（`erff`／`tanhf`）のみを使うことを固定
    /// する（イシュー #1713）。
    #[test]
    fn gelu_kinds_include_bounds_check_and_use_single_precision_intrinsics() {
        let erf_src = unary_kernel_source(ScalarUnaryOp::Gelu).expect("Gelu must be implemented");
        assert!(erf_src.contains("if (idx < numel)"));
        assert!(erf_src.contains("scalar_unary_gelu("));
        assert!(erf_src.contains("erff("));
        assert!(!erf_src.contains("erf("));

        let tanh_src =
            unary_kernel_source(ScalarUnaryOp::GeluTanh).expect("GeluTanh must be implemented");
        assert!(tanh_src.contains("if (idx < numel)"));
        assert!(tanh_src.contains("scalar_unary_gelu_tanh("));
        assert!(tanh_src.contains("tanhf("));
    }

    /// GELU 2 kind はペイロードを持たないため生成ソースは `op` に
    /// 依存しない（`sqrt_source_is_payload_independent` と同型）。
    #[test]
    fn gelu_sources_are_payload_independent() {
        let a = unary_kernel_source(ScalarUnaryOp::Gelu).expect("Gelu implemented");
        let b = unary_kernel_source(ScalarUnaryOp::Gelu).expect("Gelu implemented");
        assert_eq!(a, b);
        assert!(!a.contains("float p"));
    }

    /// Softplus のソースが REQ-8 境界チェック・`p0`／`p1` ペイロード
    /// 宣言（値は埋め込まない）・`log1pf`／`expf` を含むことを固定する
    /// （`clamp_source_declares_payload_params_and_omits_values` と同型。
    /// イシュー #1713）。
    #[test]
    fn softplus_source_declares_payload_params_and_uses_log1pf_expf() {
        let src = unary_kernel_source(ScalarUnaryOp::Softplus {
            beta: 0.123,
            threshold: 4.567,
        })
        .expect("Softplus must be implemented");
        assert!(src.contains("float p0, float p1"));
        assert!(!src.contains("0.123"));
        assert!(!src.contains("4.567"));
        assert!(src.contains("if (idx < numel)"));
        assert!(src.contains("scalar_unary_softplus("));
        assert!(src.contains("log1pf("));
        assert!(src.contains("expf("));
    }

    #[test]
    fn softplus_source_is_payload_value_independent() {
        let src_a = unary_kernel_source(ScalarUnaryOp::Softplus {
            beta: 1.0,
            threshold: 20.0,
        })
        .expect("Softplus must be implemented");
        let src_b = unary_kernel_source(ScalarUnaryOp::Softplus {
            beta: 2.0,
            threshold: 1.0,
        })
        .expect("Softplus must be implemented");
        assert_eq!(src_a, src_b);
    }
}
