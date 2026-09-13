//! `ScalarUnaryOp`／`ScalarBinaryOp`: 1 対の dispatch メソッド
//! （[`crate::BackendOps::scalar_unary`]／[`crate::BackendOps::
//! scalar_binary`]）へスカラー elementwise 演算を集約する enum
//! （イシュー #1634・親 #1592）。
//!
//! # 動機
//!
//! `docs/compat-feature-gap.md` §2.4 が指摘するとおり、`BackendOps` の
//! elementwise 面は `add`／`mul`／`relu`／`exp`／`tanh` の 5 演算固定で、
//! 演算を 1 つ足すごとに trait メソッド追加＋3 バックエンド実装＋VJP
//! 追加が必要になる構造だった。本モジュールは演算種別を
//! `#[non_exhaustive]` enum で表し、1 対の dispatch メソッド（既定
//! [`crate::device::BackendError::Unsupported`]）へ集約することで、
//! 新しい演算の追加を「enum に variant を足す」だけで済む形にする。
//!
//! # 既存 `BinaryElementwiseOp`／`UnaryElementwiseOp`（イシュー #1584）
//! との違い
//!
//! 既存の 2 enum は [`crate::BackendOps::binary_elementwise_device`]／
//! [`crate::BackendOps::unary_elementwise_device`]（[`crate::DeviceBuffer`]
//! 常駐 dispatch）専用で、「ホスト版（`add`／`mul`／`relu`／`exp`／
//! `tanh`）と同一カーネルにより bit 同一」「shape 完全一致限定
//! （broadcast 非対応）」という狭い契約を持つ。本 enum はホスト
//! `Tensor` 入出力・`add`／`mul` と同じ broadcast 意味論・超越関数
//! （REQ-2 統一複合判定の対象）を含み契約が異なるため、既存 2 enum への
//! variant 追加ではなく別の enum として新設する。
//!
//! [`crate::DeviceBuffer`] 常駐版の ScalarOp dispatch は本イシューの
//! スコープ外（`docs/scalar-op-dispatch-design.md` §8 スコープ外事項。
//! 必要なら `.claude/rules/out-of-scope-tracking.md` に従い別イシュー）。
//!
//! # forward 数式の単一情報源
//!
//! [`ScalarUnaryOp::apply`]／[`ScalarBinaryOp::apply`] が forward 数式の
//! 単一情報源（`.claude/rules/code-comment-style.md`「数式の実体を
//! 二重管理しない」）である。CPU 参照実装（`backend-cpu::
//! scalar_elementwise`）・`autodiff` のホストフォールバック（`autodiff::
//! eval::scalar`）・本モジュールの単体テストがいずれもここへ委譲する。
//!
//! # スコープ（親 #1592 の分担）
//!
//! 本イシュー（#1634）は enum 定義・`BackendOps` dispatch メソッド
//! （既定 `Unsupported`）・CPU 参照実装・`autodiff` の VJP 接続まで。
//! CUDA／Metal のカーネル実装は #1635／#1636 の担当で、既定の
//! `Unsupported` のまま残る。`Var` の公開メソッド（`sub`／`div`／
//! `pow`／…）は #1593、活性化（GELU／SiLU／…）は #1595 の担当。
//! 設計判断の全体は `docs/scalar-op-dispatch-design.md` を参照。

/// [`ScalarUnaryOp`]／[`ScalarBinaryOp`] の NVRTC キャッシュキー等に使う
/// 安定な判別子文字列を返す（`Debug` 出力はペイロード値を含むため
/// キャッシュキーには使わない。**#1635 への申し送り**: CUDA 側 NVRTC
/// キャッシュキーは本メソッドが返す `kind_name()` のみを使い、`f32`
/// ペイロード値はカーネル引数として渡すこと）。
pub trait ScalarOpKind {
    /// variant 判別子（ペイロード値を含まない安定文字列）。
    fn kind_name(&self) -> &'static str;
}

/// [`crate::BackendOps::scalar_unary`] が適用する単項スカラー演算の
/// 種別（イシュー #1634）。
///
/// `#[non_exhaustive]`: 公開 API 非破壊（ガードレール条件・
/// `.claude/rules/security.md`）を保つため（`Activation`／
/// `BinaryElementwiseOp` と同方針）。将来 variant を追加しても
/// 呼び出し側の網羅的 match を破壊しない。
///
/// `derive` は `Debug, Clone, Copy, PartialEq`（`Eq` は付けない）:
/// `LeakyRelu`／`Elu`／`Softplus`／`Clamp`／`PowScalar` が `f32`
/// ペイロードを持つため（`f32` は全順序を持たず `Eq` を実装できない）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarUnaryOp {
    /// `-x`。
    Neg,
    /// `|x|`。劣勾配は `x == 0` で `0`（`sign(0) = 0`）。
    Abs,
    /// `sqrt(x)`。定義域外（`x < 0`）は IEEE のまま（`NaN`）。
    Sqrt,
    /// `ln(x)`（自然対数）。定義域外は IEEE のまま。
    Log,
    /// `log2(x)`。
    Log2,
    /// `log10(x)`。
    Log10,
    Sin,
    Cos,
    Tan,
    /// `max(x, 0)`。**`BackendOps::relu`（既存 5 演算の 1 つ）とは
    /// 数値契約が異なる**: `BackendOps::relu`（`backend-cpu::
    /// elementwise::relu`）は `f32::max` を用い `NaN` を伝播しない
    /// （`relu(NaN) == 0.0`。同モジュール冒頭コメント参照）が、本
    /// variant は PyTorch の `torch.relu` に合わせ `NaN` を伝播する
    /// （§3.4 数値規約。`x.is_nan()` 時は `NaN` を返す明示分岐）。
    /// この差異により CPU 参照実装（`backend-cpu::scalar_elementwise`）
    /// は既存 `elementwise::relu` へ委譲せず、本 variant 専用の
    /// `NaN` 伝播分岐を持つ（`docs/scalar-op-dispatch-design.md`
    /// §3.6 参照）。劣勾配は `x == 0` で `0`。
    Relu,
    /// `exp(x)`。既存 `BackendOps::exp` と同一定義（CPU 参照実装は
    /// `elementwise::exp` へ委譲し bit 同一）。
    Exp,
    /// `tanh(x)`。既存 `BackendOps::tanh` と同一定義（CPU 参照実装は
    /// `elementwise::tanh` へ委譲し bit 同一）。
    Tanh,
    /// `1 / (1 + exp(-x))`（数値安定形。`autodiff::eval::sigmoid` と
    /// 同一の分岐構造）。
    Sigmoid,
    /// GELU（誤差関数版）: `0.5 * x * (1 + erf(x / sqrt(2)))`。`erf`
    /// は依存追加不可（`.claude/rules/deps-policy.md`）のため `f64`
    /// 精度の自作近似（Abramowitz–Stegun 7.1.26）を使う（`erf_f64`
    /// doc 参照）。
    Gelu,
    /// GELU（tanh 近似版）:
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`。
    GeluTanh,
    /// SiLU／Swish: `x * sigmoid(x)`。
    Silu,
    /// Hardswish: `x * clamp(x + 3, 0, 6) / 6`。
    Hardswish,
    /// Leaky ReLU: `x >= 0` なら `x`、それ以外は `negative_slope * x`。
    LeakyRelu {
        negative_slope: f32,
    },
    /// ELU: `x > 0` なら `x`、それ以外は `alpha * (exp(x) - 1)`。
    Elu {
        alpha: f32,
    },
    /// Softplus: `x * beta > threshold` なら恒等（`x`）、それ以外は
    /// `(1 / beta) * ln(1 + exp(beta * x))`（PyTorch 準拠のオーバー
    /// フロー回避閾値）。
    Softplus {
        beta: f32,
        threshold: f32,
    },
    /// `x` を `[min, max]` へクランプする。`min > max` は `x` に関わらず
    /// 常に `max` を返す（panic しない。§3.4 数値規約）。`NaN` 入力は
    /// `NaN` を伝播する。
    Clamp {
        min: f32,
        max: f32,
    },
    /// `x.powf(exponent)`（定数指数へのスカラー累乗）。
    PowScalar {
        exponent: f32,
    },
}

impl ScalarOpKind for ScalarUnaryOp {
    fn kind_name(&self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Abs => "abs",
            Self::Sqrt => "sqrt",
            Self::Log => "log",
            Self::Log2 => "log2",
            Self::Log10 => "log10",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Relu => "relu",
            Self::Exp => "exp",
            Self::Tanh => "tanh",
            Self::Sigmoid => "sigmoid",
            Self::Gelu => "gelu",
            Self::GeluTanh => "gelu_tanh",
            Self::Silu => "silu",
            Self::Hardswish => "hardswish",
            Self::LeakyRelu { .. } => "leaky_relu",
            Self::Elu { .. } => "elu",
            Self::Softplus { .. } => "softplus",
            Self::Clamp { .. } => "clamp",
            Self::PowScalar { .. } => "pow_scalar",
        }
    }
}

impl ScalarUnaryOp {
    /// forward 数式の単一情報源（モジュール doc「forward 数式の単一
    /// 情報源」参照）。`NaN`／`inf` 入力は各分岐のコメントに従う
    /// （明示分岐がない限り IEEE 754 のまま伝播）。
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::Neg => -x,
            Self::Abs => x.abs(),
            Self::Sqrt => x.sqrt(),
            Self::Log => x.ln(),
            Self::Log2 => x.log2(),
            Self::Log10 => x.log10(),
            Self::Sin => x.sin(),
            Self::Cos => x.cos(),
            Self::Tan => x.tan(),
            Self::Relu => {
                if x.is_nan() {
                    f32::NAN
                } else {
                    x.max(0.0)
                }
            }
            Self::Exp => x.exp(),
            Self::Tanh => x.tanh(),
            Self::Sigmoid => sigmoid_stable(x),
            Self::Gelu => gelu_erf(x),
            Self::GeluTanh => gelu_tanh_approx(x),
            Self::Silu => x * sigmoid_stable(x),
            Self::Hardswish => x * relu6(x + 3.0) / 6.0,
            Self::LeakyRelu { negative_slope } => {
                if x >= 0.0 {
                    x
                } else {
                    negative_slope * x
                }
            }
            Self::Elu { alpha } => {
                if x > 0.0 {
                    x
                } else {
                    alpha * (x.exp() - 1.0)
                }
            }
            Self::Softplus { beta, threshold } => {
                if x * beta > threshold {
                    x
                } else {
                    (1.0 / beta) * (1.0 + (beta * x).exp()).ln()
                }
            }
            Self::Clamp { min, max } => {
                if x.is_nan() {
                    f32::NAN
                } else if min > max {
                    // §3.4 数値規約: `min > max` は入力に関わらず常に
                    // `max` を返す（panic しない。`f32::clamp` は
                    // `min <= max` を要求し違反時に panic するため
                    // 使わず、手書きの 3 分岐にする）。
                    max
                } else if x < min {
                    min
                } else if x > max {
                    max
                } else {
                    x
                }
            }
            Self::PowScalar { exponent } => x.powf(exponent),
        }
    }
}

/// [`crate::BackendOps::scalar_binary`] が適用する 2 項スカラー演算の
/// 種別（イシュー #1634）。`derive` は [`ScalarUnaryOp`] と同じ理由で
/// `Debug, Clone, Copy, PartialEq`（`f32` ペイロードなしのため `Eq` も
/// 実装可能だが、`ScalarUnaryOp` との対称性のため揃えて付けない）。
///
/// `#[non_exhaustive]`: [`ScalarUnaryOp`] と同方針。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarBinaryOp {
    /// `a + b`（`BackendOps::add` と同一の定義）。
    Add,
    /// `a - b`。
    Sub,
    /// `a * b`（`BackendOps::mul` と同一の定義）。
    Mul,
    /// `a / b`。IEEE 754 のまま（`0.0` 除算・`NaN` を panic せず返す）。
    Div,
    /// `a.powf(b)`。
    Pow,
    /// `NaN` 伝播 2 項最大値（`autodiff::eval::nan_propagating_max` と
    /// 同一の明示分岐）。
    Maximum,
    /// `NaN` 伝播 2 項最小値。
    Minimum,
    /// `a > b` を `1.0`／`0.0` で返す（比較演算。`NaN` を含む場合は
    /// IEEE 754 の順序付けにより常に `0.0`）。
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

impl ScalarOpKind for ScalarBinaryOp {
    fn kind_name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Pow => "pow",
            Self::Maximum => "maximum",
            Self::Minimum => "minimum",
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Eq => "eq",
            Self::Ne => "ne",
        }
    }
}

impl ScalarBinaryOp {
    /// forward 数式の単一情報源（[`ScalarUnaryOp::apply`] と同方針）。
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
            Self::Div => a / b,
            Self::Pow => a.powf(b),
            Self::Maximum => nan_propagating_max(a, b),
            Self::Minimum => nan_propagating_min(a, b),
            Self::Gt => bool_to_f32(a > b),
            Self::Ge => bool_to_f32(a >= b),
            Self::Lt => bool_to_f32(a < b),
            Self::Le => bool_to_f32(a <= b),
            Self::Eq => bool_to_f32(a == b),
            Self::Ne => bool_to_f32(a != b),
        }
    }
}

#[inline]
fn bool_to_f32(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}

/// `NaN` 伝播する 2 項最大値。`f32::max` は非 `NaN` 側を返すため
/// `autodiff::eval::nan_propagating_max` と同じ明示分岐で置き換える
/// （`tensor-core` → `autodiff` の逆依存は作れないため独立実装。
/// 数式は同一で 2 重管理ではなく crate 境界による意図的な複製）。
#[inline]
fn nan_propagating_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

/// `NaN` 伝播する 2 項最小値（[`nan_propagating_max`] の双対）。
#[inline]
fn nan_propagating_min(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.min(b)
    }
}

/// `clamp(x, 0, 6)`（Hardswish の内部で使う relu6）。
#[inline]
fn relu6(x: f32) -> f32 {
    x.clamp(0.0, 6.0)
}

/// 数値安定形のシグモイド。`autodiff::eval::sigmoid_scalar` と同一の
/// 分岐構造（`x >= 0` は `1/(1+exp(-x))`、`x < 0` は `exp(x)/(1+exp(x))`
/// を使い分け、大きな負値入力での `exp` オーバーフローを回避する）が、
/// `tensor-core` → `autodiff` の逆依存は作れないため独立実装として
/// ここに複製する（数式は同一。crate 境界による意図的な複製）。
#[inline]
fn sigmoid_stable(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// 誤差関数 `erf(x)` の `f64` 精度近似（Abramowitz–Stegun 7.1.26。
/// 最大絶対誤差 `1.5e-7`）。`libm` クレートは依存禁止リスト対象外だが
/// 許容依存 8 区分に含まれないため追加できず（`.claude/rules/
/// deps-policy.md`）、標準ライブラリのみで自作する。GPU 側の `erff`
/// （CUDA `erff`／Metal `precise::erf`）との差は REQ-2 統一複合判定
/// （絶対誤差 1e-5 未満）に収まる想定（GPU カーネル実装は #1635／
/// #1636 のスコープ）。
fn erf_f64(x: f64) -> f64 {
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    let y = 1.0 - poly * (-ax * ax).exp();
    sign * y
}

/// GELU（誤差関数版）を `f64` 精度で計算してから `f32` へ 1 回だけ
/// downcast する（`erf_f64` の丸め誤差を最小化するため）。
fn gelu_erf(x: f32) -> f32 {
    let xf = x as f64;
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let cdf = 0.5 * (1.0 + erf_f64(xf * INV_SQRT2));
    (xf * cdf) as f32
}

/// GELU（tanh 近似版）。`f32` のまま計算する（`erf` を経由しないため
/// `f64` 昇格は不要）。
fn gelu_tanh_approx(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2 / pi)
    let u = C * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + u.tanh())
}

/// GELU（誤差関数版）の導関数 `Φ(x) + x·φ(x)`（`Φ` = 標準正規分布の
/// CDF、`φ` = PDF）。`autodiff::eval::scalar`（VJP 側）から呼ばれる
/// （`pub(crate)` ではなく `pub` なのは `apply` と同様、crate 境界を
/// 越えて `autodiff` から使うため）。
pub fn gelu_erf_grad(x: f32) -> f32 {
    let xf = x as f64;
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1 / sqrt(2*pi)
    let cdf = 0.5 * (1.0 + erf_f64(xf * INV_SQRT2));
    let pdf = INV_SQRT_2PI * (-0.5 * xf * xf).exp();
    (cdf + xf * pdf) as f32
}

/// GELU（tanh 近似版）の導関数。`u = sqrt(2/pi)·(x + 0.044715·x^3)`
/// として `d/dx[0.5·x·(1+tanh(u))] = 0.5·(1+tanh(u)) +
/// 0.5·x·(1-tanh(u)^2)·du/dx`。
pub fn gelu_tanh_grad(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2 / pi)
    let u = C * (x + 0.044_715 * x * x * x);
    let tanh_u = u.tanh();
    let du_dx = C * (1.0 + 3.0 * 0.044_715 * x * x);
    0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - tanh_u * tanh_u) * du_dx
}

/// SiLU（`x * sigmoid(x)`）の導関数 `s + x·s·(1-s)`（`s = sigmoid(x)`）。
pub fn silu_grad(x: f32) -> f32 {
    let s = sigmoid_stable(x);
    s + x * s * (1.0 - s)
}

/// Hardswish（`x·clamp(x+3,0,6)/6`）の区分導関数。
pub fn hardswish_grad(x: f32) -> f32 {
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        1.0
    } else {
        (2.0 * x + 3.0) / 6.0
    }
}

/// `sigmoid_stable` を `autodiff::eval::scalar`（Softplus の導関数）
/// から使うための `pub` 再公開。
pub fn sigmoid_scalar(x: f32) -> f32 {
    sigmoid_stable(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn assert_close(actual: f32, expected: f32, tol: f32, label: &str) {
        assert!(
            (actual - expected).abs() <= tol,
            "{label}: actual={actual} expected={expected} tol={tol}"
        );
    }

    // --- ScalarUnaryOp::apply 既知値 ---

    #[test]
    fn unary_known_values() {
        assert_close(ScalarUnaryOp::Neg.apply(3.0), -3.0, EPS, "neg");
        assert_close(ScalarUnaryOp::Abs.apply(-3.0), 3.0, EPS, "abs");
        assert_close(ScalarUnaryOp::Sqrt.apply(4.0), 2.0, EPS, "sqrt");
        assert_close(ScalarUnaryOp::Log.apply(1.0), 0.0, EPS, "log");
        assert_close(ScalarUnaryOp::Log2.apply(8.0), 3.0, EPS, "log2");
        assert_close(ScalarUnaryOp::Log10.apply(1000.0), 3.0, 1e-4, "log10");
        assert_close(ScalarUnaryOp::Sin.apply(0.0), 0.0, EPS, "sin");
        assert_close(ScalarUnaryOp::Cos.apply(0.0), 1.0, EPS, "cos");
        assert_close(ScalarUnaryOp::Tan.apply(0.0), 0.0, EPS, "tan");
        assert_close(ScalarUnaryOp::Relu.apply(-1.0), 0.0, EPS, "relu neg");
        assert_close(ScalarUnaryOp::Relu.apply(2.0), 2.0, EPS, "relu pos");
        assert_close(ScalarUnaryOp::Exp.apply(0.0), 1.0, EPS, "exp");
        assert_close(ScalarUnaryOp::Tanh.apply(0.0), 0.0, EPS, "tanh");
        assert_close(ScalarUnaryOp::Sigmoid.apply(0.0), 0.5, EPS, "sigmoid");
        assert_close(ScalarUnaryOp::Silu.apply(0.0), 0.0, EPS, "silu");
        assert_close(
            ScalarUnaryOp::Hardswish.apply(3.0),
            3.0,
            EPS,
            "hardswish sat",
        );
        assert_close(
            ScalarUnaryOp::Hardswish.apply(-3.0),
            0.0,
            EPS,
            "hardswish zero",
        );
        assert_close(
            ScalarUnaryOp::LeakyRelu {
                negative_slope: 0.1,
            }
            .apply(-2.0),
            -0.2,
            EPS,
            "leaky_relu",
        );
        assert_close(
            ScalarUnaryOp::Elu { alpha: 1.0 }.apply(0.0),
            0.0,
            EPS,
            "elu at 0",
        );
        assert_close(
            ScalarUnaryOp::PowScalar { exponent: 2.0 }.apply(3.0),
            9.0,
            EPS,
            "pow_scalar",
        );
    }

    #[test]
    fn gelu_matches_known_reference_values() {
        // gelu(0) = 0
        assert_close(ScalarUnaryOp::Gelu.apply(0.0), 0.0, EPS, "gelu(0)");
        // gelu(1) = 0.5*(1+erf(1/sqrt(2))) ≈ 0.8413447（既知参照値。
        // `erf(1/sqrt2) ≈ 0.6826895`）。
        assert_close(ScalarUnaryOp::Gelu.apply(1.0), 0.841_344_7, 1e-5, "gelu(1)");
        // gelu(-1) = 1 - gelu(1)（erf の奇関数性: gelu(x) = x - gelu(-x)
        // は一般に成り立たないが、gelu(-x) = x*erf(x/√2)/... と分けず、
        // 直接 `0.5*(-1)*(1+erf(-1/√2)) = 0.5*(-1)*(1-erf(1/√2))`
        // = -0.5*(1-0.6826895) ≈ -0.1586553 を既知値として検証する）。
        assert_close(
            ScalarUnaryOp::Gelu.apply(-1.0),
            -0.158_655_3,
            1e-5,
            "gelu(-1)",
        );
    }

    #[test]
    fn gelu_tanh_matches_known_value_at_zero() {
        assert_close(ScalarUnaryOp::GeluTanh.apply(0.0), 0.0, EPS, "gelu_tanh(0)");
    }

    #[test]
    fn softplus_matches_identity_above_threshold() {
        let op = ScalarUnaryOp::Softplus {
            beta: 1.0,
            threshold: 20.0,
        };
        assert_close(op.apply(30.0), 30.0, EPS, "softplus identity branch");
        // 通常域は softplus(0) = ln(2)
        assert_close(op.apply(0.0), std::f32::consts::LN_2, 1e-5, "softplus(0)");
    }

    #[test]
    fn clamp_min_greater_than_max_always_returns_max() {
        let op = ScalarUnaryOp::Clamp { min: 5.0, max: 1.0 };
        assert_close(op.apply(-100.0), 1.0, EPS, "clamp min>max low");
        assert_close(op.apply(0.0), 1.0, EPS, "clamp min>max mid");
        assert_close(op.apply(100.0), 1.0, EPS, "clamp min>max high");
    }

    #[test]
    fn clamp_normal_range_passes_through_and_bounds() {
        let op = ScalarUnaryOp::Clamp {
            min: -1.0,
            max: 1.0,
        };
        assert_close(op.apply(0.5), 0.5, EPS, "clamp in range");
        assert_close(op.apply(-5.0), -1.0, EPS, "clamp below min");
        assert_close(op.apply(5.0), 1.0, EPS, "clamp above max");
    }

    #[test]
    fn unary_nan_propagation() {
        for op in [
            ScalarUnaryOp::Relu,
            ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 },
        ] {
            assert!(op.apply(f32::NAN).is_nan(), "{op:?}: NaN must propagate");
        }
        // Exp/Tanh/Sigmoid/Sqrt/Log は自然に NaN を伝播する（明示分岐
        // 不要）。
        assert!(ScalarUnaryOp::Exp.apply(f32::NAN).is_nan());
        assert!(ScalarUnaryOp::Tanh.apply(f32::NAN).is_nan());
        assert!(ScalarUnaryOp::Sqrt.apply(-1.0).is_nan());
    }

    #[test]
    fn abs_zero_is_nonneg_zero() {
        assert_eq!(ScalarUnaryOp::Abs.apply(-0.0), 0.0);
        assert_eq!(ScalarUnaryOp::Abs.apply(0.0), 0.0);
    }

    // --- ScalarBinaryOp::apply 既知値 ---

    #[test]
    fn binary_known_values() {
        assert_close(ScalarBinaryOp::Add.apply(1.0, 2.0), 3.0, EPS, "add");
        assert_close(ScalarBinaryOp::Sub.apply(1.0, 2.0), -1.0, EPS, "sub");
        assert_close(ScalarBinaryOp::Mul.apply(3.0, 4.0), 12.0, EPS, "mul");
        assert_close(ScalarBinaryOp::Div.apply(6.0, 3.0), 2.0, EPS, "div");
        assert_close(ScalarBinaryOp::Pow.apply(2.0, 3.0), 8.0, EPS, "pow");
        assert_close(ScalarBinaryOp::Maximum.apply(1.0, 2.0), 2.0, EPS, "max");
        assert_close(ScalarBinaryOp::Minimum.apply(1.0, 2.0), 1.0, EPS, "min");
        assert_close(ScalarBinaryOp::Gt.apply(2.0, 1.0), 1.0, EPS, "gt true");
        assert_close(ScalarBinaryOp::Gt.apply(1.0, 2.0), 0.0, EPS, "gt false");
        assert_close(ScalarBinaryOp::Eq.apply(1.0, 1.0), 1.0, EPS, "eq true");
        assert_close(ScalarBinaryOp::Ne.apply(1.0, 1.0), 0.0, EPS, "ne false");
    }

    #[test]
    fn div_by_zero_is_ieee_not_panic() {
        assert!(ScalarBinaryOp::Div.apply(1.0, 0.0).is_infinite());
        assert!(ScalarBinaryOp::Div.apply(0.0, 0.0).is_nan());
    }

    #[test]
    fn maximum_minimum_nan_propagation() {
        assert!(ScalarBinaryOp::Maximum.apply(f32::NAN, 1.0).is_nan());
        assert!(ScalarBinaryOp::Maximum.apply(1.0, f32::NAN).is_nan());
        assert!(ScalarBinaryOp::Minimum.apply(f32::NAN, 1.0).is_nan());
    }

    #[test]
    fn comparison_nan_is_always_false() {
        for op in [
            ScalarBinaryOp::Gt,
            ScalarBinaryOp::Ge,
            ScalarBinaryOp::Lt,
            ScalarBinaryOp::Le,
            ScalarBinaryOp::Eq,
        ] {
            assert_eq!(op.apply(f32::NAN, 1.0), 0.0, "{op:?} with NaN lhs");
        }
        // Ne は NaN を含む比較で true（IEEE 754: NaN != x は常に真）。
        assert_eq!(ScalarBinaryOp::Ne.apply(f32::NAN, 1.0), 1.0);
    }

    #[test]
    fn erf_matches_known_reference_values_within_tolerance() {
        // 既知参照値（`scipy.special.erf` 相当。Abramowitz–Stegun
        // 7.1.26 は最大絶対誤差 1.5e-7）。
        assert_close(erf_f64(0.0) as f32, 0.0, 1e-6, "erf(0)");
        assert_close(erf_f64(1.0) as f32, 0.842_700_8, 2e-6, "erf(1)");
        assert_close(erf_f64(-1.0) as f32, -0.842_700_8, 2e-6, "erf(-1)");
        assert_close(erf_f64(2.0) as f32, 0.995_322_3, 2e-6, "erf(2)");
    }

    #[test]
    fn kind_name_is_stable_and_payload_independent() {
        assert_eq!(
            ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 }.kind_name(),
            ScalarUnaryOp::Clamp {
                min: -9.0,
                max: 9.0
            }
            .kind_name(),
        );
        assert_eq!(ScalarUnaryOp::Relu.kind_name(), "relu");
        assert_eq!(ScalarBinaryOp::Add.kind_name(), "add");
    }
}
