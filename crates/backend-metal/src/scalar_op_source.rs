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
//! # ペイロード seam（#1709 で実装済み）
//!
//! CUDA 側 `kernels_scalar_op::UnaryPayload`（`Clamp` の `min`/`max` を
//! カーネル起動引数として渡す設計）と同型の拡張を [`UnaryPayload`]／
//! [`unary_payload`] として実装した。ペイロードは `masked_fill`
//! （`elementwise.rs::run_binary_scalar`・`shaders/elementwise.metal::
//! ew_masked_fill_f32`）と同様に `numel` の後ろへ `setBytes_length_atIndex`
//! で渡し（`constant float& p0 [[buffer(3)]]`／`p1 [[buffer(4)]]`）、
//! ソース文字列・キャッシュキー・関数名には値を埋め込まない（現状
//! ペイロードを持つ unary kind は [`ScalarUnaryOp::Clamp`] のみ。#1707
//! が対象とした 4 kind（`Sub`／`Div`／`Pow`／`Sqrt`）はいずれもペイロード
//! を持たないため、生成ソースは payload なしのまま不変＝bit 同一
//! 〈非後退契約〉）。
//!
//! # スコープ
//!
//! [`ScalarBinaryOp::Sub`]／[`ScalarBinaryOp::Div`]／[`ScalarBinaryOp::Pow`]
//! ・[`ScalarUnaryOp::Sqrt`]（#1707）、[`ScalarUnaryOp::Neg`]／
//! [`ScalarUnaryOp::Abs`]／[`ScalarUnaryOp::Log`]／[`ScalarUnaryOp::Log2`]／
//! [`ScalarUnaryOp::Log10`]／[`ScalarUnaryOp::Sin`]／[`ScalarUnaryOp::Cos`]／
//! [`ScalarUnaryOp::Tan`]（超越関数系 8 kind。#1708）に加え、比較演算
//! 6 種（[`ScalarBinaryOp::Gt`]／[`Ge`](ScalarBinaryOp::Ge)／
//! [`Lt`](ScalarBinaryOp::Lt)／[`Le`](ScalarBinaryOp::Le)／
//! [`Eq`](ScalarBinaryOp::Eq)／[`Ne`](ScalarBinaryOp::Ne)）と
//! [`ScalarUnaryOp::Clamp`]（#1709）を実装する。比較 6 種・`Clamp` は
//! 算術を含まない純粋な比較・選択のみのため、ホスト `f32` 演算と bit
//! 同一になる想定（`NaN` は payload が処理系依存のためクラス一致で
//! 検証する。CUDA 側 `kernels_scalar_op.rs` モジュール doc「NVRTC 既定
//! オプションと数値契約」と同型の扱い）。[`ScalarUnaryOp::Gelu`]／
//! [`GeluTanh`](ScalarUnaryOp::GeluTanh)／
//! [`Softplus`](ScalarUnaryOp::Softplus)（イシュー #1713）は超越関数
//! （`scalar_erf_f32`〈自作〉／`metal::precise::tanh`／
//! `scalar_log1p_f32`〈自作〉／`metal::precise::exp`）のため bit 同一
//! を主張せず REQ-2 複合判定のみで検証する（`unary_prelude` 参照）。
//!
//! [`ScalarUnaryOp::Silu`]／[`ScalarUnaryOp::Hardswish`]／
//! [`ScalarUnaryOp::LeakyRelu`]／[`ScalarUnaryOp::Elu`]（イシュー
//! #1714）も実装する。`LeakyRelu`／`Elu` は本モジュールで初めて 1
//! 引数ペイロード（[`UnaryPayload::One`]）を持つ unary kind（`Clamp` の
//! [`UnaryPayload::Two`] と同型の拡張）。`Elu` は MSL に `expm1`
//! 相当が存在しないため、`exp`／`log` から桁落ちなく再構成する
//! `fai_expm1_f32` ヘルパー関数（下記「`Elu` の `expm1` 非対応」節
//! 参照）を [`unary_preamble`] で kernel 本体の前に挿入して使う
//! （PR #1825 codex-review P1 是正: 単純な `exp(x) - 1.0f` は
//! `x=-1e-8, alpha=1e8` のようなゼロ近傍・大 `alpha` の入力で
//! `exp(x)` が `1.0` へ丸まり桁落ちし、CPU/CUDA の `expm1` 相当と
//! REQ-2 統一複合判定を満たさなかった）。
//!
//! 残 kind（`Add`／`Mul`／`Maximum`／`Minimum`・活性化系〈`Relu`／
//! `Exp`／`Tanh`／`Sigmoid`〉・`PowScalar`）はいずれの sub issue にも
//! 含まれず `None`（未実装のまま。
//! 呼び出し元 `ops::MetalBackendOps::scalar_unary`／`scalar_binary` が
//! `BackendError::Unsupported` を返しホスト参照実装
//! （`ScalarUnaryOp::apply`／`ScalarBinaryOp::apply`）へフォールバック
//! する既存契約。`fandhe_ai_autodiff::grad::scalar_unary_with_fallback`／
//! `scalar_binary_with_fallback` 参照）。
//! `.claude/rules/out-of-scope-tracking.md` の追跡対象。
//!
//! # `Elu` の `expm1` 非対応（イシュー #1714・PR #1825 codex-review P1 是正）
//!
//! ホスト参照実装（`scalar_op.rs::ScalarUnaryOp::apply` の `Elu` 分岐）
//! は `f32::exp_m1`（`exp(x) - 1` を `x` が `0` に近くても桁落ちなく
//! 計算する標準ライブラリ関数）を使うが、MSL には `expm1` 相当の
//! 組み込み関数が存在しない（`.claude/skills/apple-silicon/references/
//! msl/` に該当なしを確認済み）。当初は `metal::precise::exp(x) -
//! 1.0f` で代替していたが、`|x|` が `0` に近く `alpha` が大きい入力
//! （例: `x=-1e-8, alpha=1e8`）で `exp(x)` が `1.0f` へ丸まって
//! `expm1(x)` が `0.0` になり（正しくは `x` 自身にほぼ等しい
//! `-1e-8`）、`alpha` 倍された結果が CPU/CUDA の `expm1` 相当と
//! REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5
//! 未満）を満たさなかった（`tensor-core::scalar_op` の
//! `elu_tiny_negative_x_with_large_alpha_avoids_exp_cancellation` が
//! 同じ入力で CPU 側の桁落ちを既に回帰検査済み）。
//!
//! 本モジュールは Kahan/Goldberg の再構成式（`u = exp(x)` を計算した
//! うえで `u == 1.0` なら `expm1(x) ≈ x` を返し、それ以外は
//! `(u - 1) * x / log(u)` で桁落ちなく再構成する）を [`unary_preamble`]
//! が返す `fai_expm1_f32` ヘルパー関数として `unary_kernel_source` の
//! kernel 本体前へ挿入し、`Elu` の `expr`（[`unary_expr`]）から呼ぶ。
//! `x` が十分小さく `exp(x)` が `f32` の最小非正規化数未満へ
//! underflow して `u == 0.0` になる領域（`x` がおよそ `-104` 未満）は
//! 上記の式だと `log(0) = -inf` で `(u-1)*x/log(u)` が `0`（正しくは
//! `expm1(x) ≈ -1`）に潰れるため、`u == 0.0` を追加で分岐し `-1.0f`
//! を直接返す。近似ではなく `exp`／`log` の 2 回呼び出しへ分解した
//! 再構成のため、ホストの `f32::exp_m1`（正確な libm 実装）とは
//! 一般に bit 同一にならず、超越関数系と同じく REQ-2 統一複合判定の
//! みで検証する（tolerance 定数自体は不変。`.claude/rules/
//! coding-rust.md`）。
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
        // `ScalarUnaryOp::apply` の `Clamp` 分岐（`scalar_op.rs`）を
        // CUDA 側 `kernels_scalar_op.rs::unary_expr` と同一の式で逐語
        // 複製する（`isnan(x)` → NaN を伝播 / `p0 > p1`（min > max）→
        // 常に `p1`（max）/ `x < p0` → `p0` / `x > p1` → `p1` /
        // それ以外 → `x`）。`metal::clamp`／`fmin`／`fmax`（IEEE
        // minNum/maxNum は非 NaN 側を優先し明示 `isnan` 分岐と異なる）
        // は使わない（モジュール doc「forward 数式の正」参照）。
        // `isnan` は `metal_stdlib`（`using namespace metal;` 下）の
        // 関数でそのまま呼べる。
        ScalarUnaryOp::Clamp { .. } => {
            Some("isnan(x) ? x : (p0 > p1 ? p1 : (x < p0 ? p0 : (x > p1 ? p1 : x)))")
        }
        // `ScalarUnaryOp::apply` の `LeakyRelu` 分岐（`scalar_op.rs`）を
        // CUDA 側 `kernels_scalar_op.rs::unary_expr` と同一の式で逐語
        // 複製する（選択と乗算のみでホスト `f32` 演算と bit 同一になる
        // 想定。モジュール doc「コンパイルオプションと数値契約」参照）。
        ScalarUnaryOp::LeakyRelu { .. } => Some("(x >= 0.0f) ? x : (p0 * x)"),
        // `ScalarUnaryOp::apply` の `Hardswish` 分岐を CUDA 側と同一の
        // 演算順序（`relu6(x + 3.0) / 6.0` を先に評価してから `x` へ
        // 乗算する）で複製する。`fmin`／`fmax`／`clamp`（IEEE
        // minNum/maxNum は非 NaN 側を優先し `is_nan` 明示分岐と異なる）
        // は使わない。
        ScalarUnaryOp::Hardswish => {
            Some("x * ((x + 3.0f < 0.0f) ? 0.0f : ((x + 3.0f > 6.0f) ? 6.0f : (x + 3.0f)) / 6.0f)")
        }
        // `ScalarUnaryOp::apply` の `Silu` 分岐（`x * sigmoid_stable(x)`）
        // を `metal::precise::exp` で複製する（`fast::exp` は使わない。
        // 超越関数のため REQ-2 複合判定のみで検証する）。
        ScalarUnaryOp::Silu => Some(
            "(x >= 0.0f) ? (x * (1.0f / (1.0f + metal::precise::exp(-x)))) : (x * \
             (metal::precise::exp(x) / (1.0f + metal::precise::exp(x))))",
        ),
        // `ScalarUnaryOp::apply` の `Elu` 分岐。`fai_expm1_f32`
        // （[`unary_preamble`] が kernel 本体の前へ挿入するヘルパー
        // 関数。モジュール doc「`Elu` の `expm1` 非対応」参照）を使い
        // ゼロ近傍の桁落ちを避ける。REQ-2 複合判定のみで検証する。
        ScalarUnaryOp::Elu { .. } => Some("(x > 0.0f) ? x : (p0 * fai_expm1_f32(x))"),
        // GELU（誤差関数版）: MSL には `erf` が存在しないため
        // `unary_prelude` が定義する自作ヘルパ `scalar_erf_f32`
        // （A-S 7.1.26 近似・`float` 計算）を呼ぶ（超越関数のため
        // bit 同一は主張しない。モジュール doc「forward 数式の正」
        // 参照）。
        ScalarUnaryOp::Gelu => Some("0.5f * x * (1.0f + scalar_erf_f32(x * 0.70710678f))"),
        // GELU（tanh 近似版）: CUDA 側 `tanhf` に対応する
        // `metal::precise::tanh` を使う（既存 kind と同じ `precise::`
        // 方針）。
        ScalarUnaryOp::GeluTanh => Some(
            "0.5f * x * (1.0f + metal::precise::tanh(0.7978846f * (x + 0.044715f * x * x * x)))",
        ),
        // Softplus: MSL には `log1p` が存在しないため `unary_prelude`
        // が定義する自作ヘルパ `scalar_log1p_f32`（Kahan の補正式）を
        // 呼ぶ。`p0`＝`beta`・`p1`＝`threshold`（CUDA 側 `unary_expr`
        // の `Softplus` 分岐と同一構造）。
        ScalarUnaryOp::Softplus { .. } => {
            Some("(x * p0 > p1) ? x : (scalar_log1p_f32(metal::precise::exp(p0 * x)) / p0)")
        }
        _ => None,
    }
}

/// `op` の [`unary_expr`] が参照する device 側ヘルパー関数の宣言（kernel
/// 本体より前に挿入する。既存 kind との後方互換のため既定は空文字列＝
/// 挿入なし＝生成ソース不変）。`Elu` のみが `fai_expm1_f32`
/// （モジュール doc「`Elu` の `expm1` 非対応」参照。PR #1825
/// codex-review P1 是正）を必要とする。
///
/// `static inline` を使うのは `elementwise.metal` に既存のヘルパー
/// 関数（`ew_*` の前段に置かれるものはないが、MSL の関数はデフォルトで
/// 内部リンケージのため `static` は必須ではない。`shaders/` 系との
/// 記法統一のため明示する）。
fn unary_preamble(op: ScalarUnaryOp) -> &'static str {
    match op {
        // `expm1(x) = exp(x) - 1` を桁落ちなく計算する Kahan/Goldberg
        // の再構成式。`u == 1.0f`（`x` が `0` に非常に近く `exp(x)` が
        // `1.0` へ丸まる領域）では `expm1(x) ≈ x` を直接返す。
        // `u == 0.0f`（`exp(x)` が最小非正規化数未満へ underflow する
        // 領域。`x` がおよそ `-104` 未満）では `log(u) = -inf` になり
        // 一般式が `0` に潰れてしまう（正しくは `expm1(x) ≈ -1`）ため
        // 追加で分岐し `-1.0f` を直接返す。それ以外は
        // `(u - 1) * x / log(u)` で再構成する（`log(u) ≈ x` となる
        // ため桁落ちしない）。
        ScalarUnaryOp::Elu { .. } => {
            "static inline float fai_expm1_f32(float x) {\n    float u = metal::precise::exp(x);\n    if (u == 1.0f) {\n        return x;\n    }\n    if (u == 0.0f) {\n        return -1.0f;\n    }\n    return (u - 1.0f) * x / metal::precise::log(u);\n}\n\n"
        }
        _ => "",
    }
}

/// unary kind 別の MSL ヘルパ関数 prelude（`using namespace metal;` と
/// `kernel void` の間に挿入する文字列。既存 kind は空文字列のため生成
/// ソースは不変＝bit 同一の非後退契約を保つ。イシュー #1713）。
///
/// MSL には `erf`／`log1p` が存在しない（`.claude/skills/apple-silicon/
/// references/msl/math-functions.md` に記載なし）ため、[`ScalarUnaryOp::
/// Gelu`]（誤差関数版 GELU）・[`Softplus`](ScalarUnaryOp::Softplus) は
/// ここで自作する。ヘルパ引数はすべて `float` 値渡しとする（`constant
/// float&` ではないため `payload_param_count_matches_source_for_all_
/// implemented_unary_kinds` の「`p0`/`p1` 宣言数＝payload 数」検査を
/// 阻害しない）。
fn unary_prelude(op: ScalarUnaryOp) -> &'static str {
    match op {
        // `tensor-core::scalar_op::erf_f64`（Abramowitz–Stegun 7.1.26。
        // 最大絶対誤差 `1.5e-7`）と同一の多項式係数を `float` 精度で
        // 計算する（crate 境界〈Rust／MSL〉による意図的複製。設計 §9
        // と同型。ホストは `f64` 中間計算・本ヘルパは `float` のみの
        // ため厳密には別実装だが、超越関数として REQ-2 複合判定のみを
        // 主張するため許容する）。
        ScalarUnaryOp::Gelu => {
            "inline float scalar_erf_f32(float x) {\n\
             \x20   float sign = (x < 0.0f) ? -1.0f : 1.0f;\n\
             \x20   float ax = metal::precise::fabs(x);\n\
             \x20   float t = 1.0f / (1.0f + 0.3275911f * ax);\n\
             \x20   float poly = ((((1.061405429f * t + -1.453152027f) * t + \
             1.421413741f) * t + -0.284496736f) * t + 0.254829592f) * t;\n\
             \x20   float y = 1.0f - poly * metal::precise::exp(-ax * ax);\n\
             \x20   return sign * y;\n\
             }\n\n"
        }
        // `log1p(y) = ln(1+y)` を桁落ちなく計算する Kahan の補正式
        // （`u == 1.0` すなわち `y` が `f32` の ulp 未満のときは `y`
        // 自体を返す。`ScalarUnaryOp::apply` の `Softplus` 分岐が使う
        // ホスト `f32::ln_1p` の意図的複製）。`y` は
        // `metal::precise::exp(p0 * x)` の結果で非負だが、`x` が
        // `f32::exp` の飽和域（約 88.7 超）でも `x*beta <= threshold`
        // となりうるため `y` 自体が `+inf` を取りうる（CPU/CUDA の
        // `ln_1p`/`log1pf` は `log1p(inf) = inf` を返す）。この場合
        // `u = 1+y` も `+inf` になり補正式が `inf/inf` を計算して
        // `NaN` になってしまうため、先頭で明示的に `+inf` を伝播する。
        // 有限入力側も `log(u) * y` を先に評価すると（`y` が大きい
        // ほど）`f32` 上限を超えて誤って `inf` になりうるため、桁落ち
        // しない比 `y / (u - 1.0f)`（`u` が大きいほど 1 に近づき
        // オーバーフローしない）を先に評価してから `log(u)` を掛ける
        // 順序へ変更する（積の評価順序のみの変更で数式・契約は不変）。
        ScalarUnaryOp::Softplus { .. } => {
            "inline float scalar_log1p_f32(float y) {\n\
             \x20   if (isinf(y)) {\n\
             \x20       return y;\n\
             \x20   }\n\
             \x20   float u = 1.0f + y;\n\
             \x20   return (u == 1.0f) ? y : metal::precise::log(u) * (y / (u - 1.0f));\n\
             }\n\n"
        }
        _ => "",
    }
}

/// [`ScalarBinaryOp`] の MSL 式（変数名は `a_v`／`b_v`）。比較演算は
/// `bool_to_f32`（`scalar_op.rs`）と同じ `0.0f`／`1.0f` を返す（`docs/
/// scalar-op-dispatch-design.md` §3.2「bool 出力」契約）。未実装 kind
/// は `None`。
fn binary_expr(op: ScalarBinaryOp) -> Option<&'static str> {
    match op {
        ScalarBinaryOp::Sub => Some("a_v - b_v"),
        ScalarBinaryOp::Div => Some("a_v / b_v"),
        // `metal::precise::pow` は超越関数（最大 16 ulp。correctly
        // rounded を保証しない）。REQ-2 複合判定のみで検証する
        // （モジュール doc「コンパイルオプションと数値契約」参照）。
        ScalarBinaryOp::Pow => Some("metal::precise::pow(a_v, b_v)"),
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
/// （ソース文字列へは埋め込まない。モジュール doc「ペイロード seam」
/// 参照）。`None` はペイロードなし kind（起動引数列は既存 `numel`
/// までで不変＝bit 同一）、`Two([p0, p1])` は 2 引数ペイロード kind
/// （現状 `Clamp` のみ。CUDA 側 `kernels_scalar_op::UnaryPayload` と
/// 同型）。
pub(crate) enum UnaryPayload {
    None,
    /// 1 引数ペイロード（`LeakyRelu { negative_slope }`／
    /// `Elu { alpha }`。イシュー #1714。CUDA 側
    /// `kernels_scalar_op::UnaryPayload::One` と同型）。
    One([f32; 1]),
    Two([f32; 2]),
}

impl UnaryPayload {
    /// カーネル起動引数として `numel` の後ろへ渡す順序どおりのスライス
    /// （空スライスは追加引数なし＝既存 kind と同じ起動引数列）。
    pub(crate) fn as_slice(&self) -> &[f32] {
        match self {
            Self::None => &[],
            Self::One(v) => v,
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
        ScalarUnaryOp::LeakyRelu { negative_slope } => UnaryPayload::One([negative_slope]),
        ScalarUnaryOp::Elu { alpha } => UnaryPayload::One([alpha]),
        ScalarUnaryOp::Softplus { beta, threshold } => UnaryPayload::Two([beta, threshold]),
        _ => UnaryPayload::None,
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
    // ペイロードあり kind（`Clamp` 等）は `numel` の後ろへ `constant
    // float& p0 [[buffer(3)]]`／`p1 [[buffer(4)]]` を追加宣言する
    // （モジュール doc「ペイロード seam」参照。`elementwise.metal::
    // ew_masked_fill_f32` の `constant float& value` と同じ渡し方）。
    // ペイロード値自体はここへ埋め込まず、常に固定パラメータ名
    // （`p0`／`p1`）のみを使うため、`kind_name()` が同じ限り payload
    // 値が異なってもソース文字列は完全一致する（キャッシュキーが
    // payload 非依存であることの根拠。単体テスト
    // `clamp_source_declares_payload_params_and_omits_values` 参照）。
    let payload_params = match unary_payload(op) {
        UnaryPayload::None => String::new(),
        UnaryPayload::One(_) => ",\n    constant float& p0 [[buffer(3)]]".to_string(),
        UnaryPayload::Two(_) => {
            ",\n    constant float& p0 [[buffer(3)]],\n    constant float& p1 [[buffer(4)]]"
                .to_string()
        }
    };
    // 既存 kind（`preamble == "" && prelude == ""`）は挿入前と生成
    // ソース完全一致（`kind_name()` 依存のキャッシュキー契約を壊さない。
    // `sqrt_source_is_payload_independent` 等の bit 同一契約に抵触しない。
    // `Elu` は `unary_preamble` が、`Gelu`／`GeluTanh`／`Softplus` は
    // `unary_prelude` が非空文字列を返す（互いに排他的な kind 集合の
    // ためどちらか一方のみ非空。イシュー #1713／#1714）。
    let preamble = unary_preamble(op);
    let prelude = unary_prelude(op);
    Some(format!(
        r#"#include <metal_stdlib>
using namespace metal;

{preamble}{prelude}kernel void {name}(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& numel [[buffer(2)]]{payload_params},
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
        // `Log2`／`Log10`／`Sin`／`Cos`／`Tan`。#1708）・`Clamp`（#1709）・
        // `Silu`／`Hardswish`／`LeakyRelu`／`Elu`（#1714）は実装済みに
        // なったため、番兵 kind を未実装のまま残る kind（`Relu`〈活性化
        // 系〉・`PowScalar`〈他のペイロードあり unary kind〉。いずれも
        // sub issue に含まれない）へ付け替える（残すと未実装 kind への
        // `None` フォールバック契約の検証が消えてしまう）。
        assert!(unary_kernel_source(ScalarUnaryOp::Relu).is_none());
        assert!(unary_kernel_source(ScalarUnaryOp::Sigmoid).is_none());
        assert!(unary_kernel_source(ScalarUnaryOp::PowScalar { exponent: 2.0 }).is_none());
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
        // `Gt` は #1709 で実装済みになったため番兵を `Mul` へ付け替える
        // （`Add`／`Maximum` は維持）。
        assert!(binary_kernel_source(ScalarBinaryOp::Add).is_none());
        assert!(binary_kernel_source(ScalarBinaryOp::Maximum).is_none());
        assert!(binary_kernel_source(ScalarBinaryOp::Mul).is_none());
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
        assert_eq!(binary_function_name(ScalarBinaryOp::Gt), "scalar_binary_gt");
        assert_eq!(binary_function_name(ScalarBinaryOp::Ge), "scalar_binary_ge");
        assert_eq!(binary_function_name(ScalarBinaryOp::Lt), "scalar_binary_lt");
        assert_eq!(binary_function_name(ScalarBinaryOp::Le), "scalar_binary_le");
        assert_eq!(binary_function_name(ScalarBinaryOp::Eq), "scalar_binary_eq");
        assert_eq!(binary_function_name(ScalarBinaryOp::Ne), "scalar_binary_ne");
    }

    /// 生成ソースが payload 値を一切含まないこと（モジュール doc
    /// 「ペイロード seam」の前提: #1707 の対象 4 kind はいずれも
    /// ペイロードを持たないため、`unary_kernel_source`／
    /// `binary_kernel_source` の戻り値は `op` に依存しないはず）。
    #[test]
    fn sqrt_source_is_payload_independent() {
        let src_a = unary_kernel_source(ScalarUnaryOp::Sqrt).expect("Sqrt implemented");
        let src_b = unary_kernel_source(ScalarUnaryOp::Sqrt).expect("Sqrt implemented");
        assert_eq!(src_a, src_b);
    }

    /// 比較 6 種のソースが REQ-8 境界チェック・関数名を満たし、
    /// `1.0f`／`0.0f` リテラルを含むこと（CUDA 側
    /// `kernels_scalar_op.rs::comparison_ops_return_zero_or_one_literals`
    /// と同型）。
    #[test]
    fn comparison_kinds_include_bounds_check_and_zero_one_literals() {
        for op in [
            ScalarBinaryOp::Gt,
            ScalarBinaryOp::Ge,
            ScalarBinaryOp::Lt,
            ScalarBinaryOp::Le,
            ScalarBinaryOp::Eq,
            ScalarBinaryOp::Ne,
        ] {
            let src = binary_kernel_source(op).expect("must be implemented");
            assert!(src.contains("if (idx < numel)"));
            assert!(src.contains(&format!("kernel void {}(", binary_function_name(op))));
            assert!(src.contains("[[buffer(0)]]"));
            assert!(src.contains("[[buffer(1)]]"));
            assert!(src.contains("[[buffer(2)]]"));
            assert!(src.contains("[[buffer(3)]]"));
            assert!(src.contains("1.0f"), "{op:?} source must contain 1.0f");
            assert!(src.contains("0.0f"), "{op:?} source must contain 0.0f");
        }
    }

    /// `Clamp` ソースが `numel` の後ろへ `p0`／`p1` を
    /// `constant float&` として宣言し、payload 値自体は埋め込まず、
    /// 異なる payload 値でも生成ソースが完全一致すること（キャッシュ
    /// キーが `kind_name()` のみに依存する契約の根拠）を固定する。
    #[test]
    fn clamp_source_declares_payload_params_and_omits_values() {
        let src = unary_kernel_source(ScalarUnaryOp::Clamp {
            min: 0.123,
            max: 4.567,
        })
        .expect("Clamp must be implemented");
        assert!(src.contains("constant float& p0 [[buffer(3)]]"));
        assert!(src.contains("constant float& p1 [[buffer(4)]]"));
        assert!(!src.contains("0.123"));
        assert!(!src.contains("4.567"));
        assert!(src.contains("if (idx < numel)"));
        assert!(src.contains("kernel void scalar_unary_clamp("));
    }

    #[test]
    fn clamp_source_is_payload_value_independent() {
        let src_a = unary_kernel_source(ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 })
            .expect("Clamp must be implemented");
        let src_b = unary_kernel_source(ScalarUnaryOp::Clamp {
            min: -5.0,
            max: 5.0,
        })
        .expect("Clamp must be implemented");
        assert_eq!(src_a, src_b);
    }

    /// `metal::clamp`／`fmin`／`fmax`（IEEE minNum/maxNum は非 NaN 側を
    /// 優先し `ScalarUnaryOp::apply` の明示 `is_nan` 分岐と数値契約が
    /// 異なる）は使わない（CUDA 側
    /// `clamp_source_does_not_use_fminf_fmaxf` と同型）。
    ///
    /// 関数名 `scalar_unary_clamp(` 自体が部分文字列 `"clamp("` を含む
    /// ため、`!contains("clamp(")` は書けない。`transcendental_*`
    /// テストと同じ「先頭スペース付きパターン」で `metal::clamp(` の
    /// 裸呼び出しがないことを確認する。
    #[test]
    fn clamp_source_does_not_use_fmin_fmax_or_metal_clamp() {
        let src = unary_kernel_source(ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 })
            .expect("Clamp must be implemented");
        assert!(!src.contains("fmin("));
        assert!(!src.contains("fmax("));
        assert!(!src.contains(" clamp("));
        assert!(!src.contains("metal::clamp("));
    }

    /// 実装済み unary kind すべてについて、ソース中の宣言済み
    /// `constant float&` パラメータ数（`p0`／`p1`）が
    /// `unary_payload(op).as_slice().len()` と一致することを固定する
    /// （引数個数の不一致はカーネル起動時にしか露見しないため事前に
    /// ホスト側で検出する。CUDA 側
    /// `payload_param_count_matches_source_for_all_implemented_unary_kinds`
    /// と同型）。
    #[test]
    fn payload_param_count_matches_source_for_all_implemented_unary_kinds() {
        for op in [
            ScalarUnaryOp::Sqrt,
            ScalarUnaryOp::Neg,
            ScalarUnaryOp::Abs,
            ScalarUnaryOp::Log,
            ScalarUnaryOp::Log2,
            ScalarUnaryOp::Log10,
            ScalarUnaryOp::Sin,
            ScalarUnaryOp::Cos,
            ScalarUnaryOp::Tan,
            ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 },
            ScalarUnaryOp::LeakyRelu {
                negative_slope: 0.1,
            },
            ScalarUnaryOp::Elu { alpha: 1.0 },
            ScalarUnaryOp::Gelu,
            ScalarUnaryOp::GeluTanh,
            ScalarUnaryOp::Softplus {
                beta: 1.0,
                threshold: 20.0,
            },
        ] {
            let src = unary_kernel_source(op).expect("must be implemented");
            let declared = src.matches("constant float&").count();
            let expected = unary_payload(op).as_slice().len();
            assert_eq!(
                declared, expected,
                "{op:?}: declared payload params ({declared}) != unary_payload len ({expected})"
            );
        }
    }

    /// 比較演算・`Clamp` を含む [`unary_kernel_source`]／
    /// [`binary_kernel_source`] が `#1709` 追加後もそれぞれ番兵 kind と
    /// 混同なく識別できることの回帰（`function_names_are_kind_name_
    /// derived_and_stable` と重複しない追加確認: 番兵付け替えの安全性
    /// テスト）。
    #[test]
    fn unimplemented_kinds_remain_none_after_1709_additions() {
        assert!(unary_kernel_source(ScalarUnaryOp::Relu).is_none());
        assert!(binary_kernel_source(ScalarBinaryOp::Add).is_none());
        assert!(binary_kernel_source(ScalarBinaryOp::Maximum).is_none());
        assert!(binary_kernel_source(ScalarBinaryOp::Minimum).is_none());
    }

    /// イシュー #1714 の対象 4 kind（`Silu`／`Hardswish`／`LeakyRelu`／
    /// `Elu`）が REQ-8 境界チェック・関数名・buffer index 契約を含む
    /// ことを固定する（CUDA 側「静的ソース内容検査」の方針を踏襲）。
    #[test]
    fn silu_hardswish_leaky_relu_elu_include_bounds_check_and_buffer_indices() {
        let no_payload_kinds = [ScalarUnaryOp::Silu, ScalarUnaryOp::Hardswish];
        for op in no_payload_kinds {
            let src = unary_kernel_source(op)
                .unwrap_or_else(|| panic!("{op:?} must be implemented by #1714"));
            assert!(src.contains("if (idx < numel)"));
            assert!(src.contains(&format!("kernel void {}(", unary_function_name(op))));
            assert!(src.contains("[[buffer(0)]]"));
            assert!(src.contains("[[buffer(1)]]"));
            assert!(src.contains("[[buffer(2)]]"));
            assert!(!src.contains("[[buffer(3)]]"));
        }

        let payload_kinds = [
            ScalarUnaryOp::LeakyRelu {
                negative_slope: 0.1,
            },
            ScalarUnaryOp::Elu { alpha: 1.0 },
        ];
        for op in payload_kinds {
            let src = unary_kernel_source(op)
                .unwrap_or_else(|| panic!("{op:?} must be implemented by #1714"));
            assert!(src.contains("if (idx < numel)"));
            assert!(src.contains(&format!("kernel void {}(", unary_function_name(op))));
            assert!(src.contains("[[buffer(3)]]"));
            assert!(!src.contains("[[buffer(4)]]"));
        }
    }

    /// `LeakyRelu`／`Elu` はペイロード値をソース文字列へ埋め込まない
    /// （キャッシュキーが `kind_name()` のみに依存する契約。
    /// `clamp_source_is_payload_value_independent` と同型）。
    #[test]
    fn leaky_relu_and_elu_sources_are_payload_value_independent() {
        let lr_a = unary_kernel_source(ScalarUnaryOp::LeakyRelu {
            negative_slope: 0.01,
        })
        .expect("LeakyRelu must be implemented");
        let lr_b = unary_kernel_source(ScalarUnaryOp::LeakyRelu {
            negative_slope: 0.5,
        })
        .expect("LeakyRelu must be implemented");
        assert_eq!(lr_a, lr_b);
        assert!(!lr_a.contains("0.01"));
        assert!(!lr_a.contains("0.5"));

        let elu_a = unary_kernel_source(ScalarUnaryOp::Elu { alpha: 1.0 })
            .expect("Elu must be implemented");
        let elu_b = unary_kernel_source(ScalarUnaryOp::Elu { alpha: 2.5 })
            .expect("Elu must be implemented");
        assert_eq!(elu_a, elu_b);
        assert!(!elu_a.contains("2.5"));
    }

    /// `Silu`／`Elu` は `metal::precise::exp` のみを使い `fast::exp` を
    /// 使わない（超越関数系テストと同方針）。
    #[test]
    fn silu_and_elu_use_precise_exp_only() {
        let silu_src = unary_kernel_source(ScalarUnaryOp::Silu).expect("Silu must be implemented");
        assert!(silu_src.contains("metal::precise::exp("));
        assert!(!silu_src.contains("fast::exp("));

        let elu_src = unary_kernel_source(ScalarUnaryOp::Elu { alpha: 1.0 })
            .expect("Elu must be implemented");
        assert!(elu_src.contains("metal::precise::exp("));
        assert!(!elu_src.contains("fast::exp("));
    }

    /// PR #1825 codex-review P1 是正の回帰テスト: `Elu` の生成ソースが
    /// `metal::precise::exp(x) - 1.0f` という単純差分（ゼロ近傍で
    /// 桁落ちする式。`x=-1e-8, alpha=1e8` で `0.0` を返し CPU/CUDA の
    /// `expm1` 相当〈約 `-1`〉と REQ-2 統一複合判定を満たさなかった）
    /// を含まないこと、代わりに `fai_expm1_f32` ヘルパー（[`unary_
    /// preamble`]）を kernel 本体より前で宣言し `Elu` の式から呼ぶ
    /// ことを固定する。
    #[test]
    fn elu_source_uses_expm1_helper_not_naive_exp_minus_one() {
        let elu_src = unary_kernel_source(ScalarUnaryOp::Elu { alpha: 1.0 })
            .expect("Elu must be implemented");
        assert!(
            !elu_src.contains("metal::precise::exp(x) - 1.0f"),
            "ゼロ近傍で桁落ちする単純差分が残っている: {elu_src}"
        );
        assert!(elu_src.contains("fai_expm1_f32"));
        assert!(elu_src.contains("float fai_expm1_f32(float x)"));
        // ヘルパー宣言は kernel 本体（`kernel void`）より前に来ること。
        let helper_pos = elu_src
            .find("float fai_expm1_f32(float x)")
            .expect("helper must be declared");
        let kernel_pos = elu_src.find("kernel void").expect("kernel must exist");
        assert!(helper_pos < kernel_pos);
        // underflow（`u == 0.0f`）・ゼロ近傍（`u == 1.0f`）双方の分岐を
        // 含むこと（モジュール doc「`Elu` の `expm1` 非対応」参照）。
        assert!(elu_src.contains("u == 1.0f"));
        assert!(elu_src.contains("u == 0.0f"));
    }

    /// 他 kind（`Silu` 等ペイロードなし）は `unary_preamble` が空文字列
    /// を返すため、preamble 挿入前後でソースが完全不変である
    /// （既存キャッシュキー契約〈`kind_name()` のみへ依存〉を壊さない
    /// ことの回帰確認）。
    #[test]
    fn preamble_is_empty_for_kinds_without_expm1_helper() {
        let silu_src = unary_kernel_source(ScalarUnaryOp::Silu).expect("Silu must be implemented");
        assert!(!silu_src.contains("fai_expm1_f32"));
        assert!(silu_src.trim_start().starts_with("#include <metal_stdlib>"));

        let hs_src =
            unary_kernel_source(ScalarUnaryOp::Hardswish).expect("Hardswish must be implemented");
        assert!(!hs_src.contains("fai_expm1_f32"));
    }

    /// `fai_expm1_f32` の再構成式（`u == 1.0 ? x : (u == 0.0 ? -1.0 :
    /// (u - 1) * x / log(u))`）をホスト側 `f64` で忠実に再現し、
    /// codex-review 指摘の入力（`x=-1e-8`）・underflow 域
    /// （`x=-150.0`）・通常域（`x=-1.0`）で `f32::exp_m1` 相当の
    /// 正しい `expm1(x)` に近い値を返すことを検証する（Metal 実機
    /// なしで式そのものの正しさを Linux 上で確認する diagnostic）。
    #[test]
    fn expm1_reconstruction_formula_matches_exp_m1_reference() {
        fn expm1_reconstruction(x: f32) -> f32 {
            let u = x.exp();
            if u == 1.0 {
                x
            } else if u == 0.0 {
                -1.0
            } else {
                (u - 1.0) * x / u.ln()
            }
        }

        // codex-review 指摘の入力: ゼロ近傍・大 alpha で顕在化した
        // ケース。単純差分は `0.0` を返すが正しくは `x` にほぼ等しい。
        let x = -1e-8_f32;
        let got = expm1_reconstruction(x);
        let want = x.exp_m1();
        assert!(
            (got - want).abs() < 1e-5,
            "got={got}, want={want} (naive exp(x)-1 would give 0.0)"
        );
        assert_ne!(got, 0.0, "expm1(-1e-8) が 0.0 に桁落ちしてはならない");

        // underflow 域（`exp(x)` が `f32` 最小非正規化数未満）: 単純式
        // だけでなく `log(0)=-inf` 経路も `-1.0` へ正しく落ちること。
        let x = -150.0_f32;
        let got = expm1_reconstruction(x);
        let want = x.exp_m1();
        assert!(
            (got - want).abs() < 1e-5,
            "got={got}, want={want} (underflow branch)"
        );

        // 通常域（桁落ちが問題にならない典型値）。
        let x = -1.0_f32;
        let got = expm1_reconstruction(x);
        let want = x.exp_m1();
        assert!(
            (got - want).abs() < 1e-5,
            "got={got}, want={want} (normal range)"
        );
    }

    /// `Hardswish`／`LeakyRelu` は選択・算術のみで `fmin`／`fmax`／
    /// `metal::clamp`（`ScalarUnaryOp::apply` の明示分岐と数値契約が
    /// 異なる）を使わない（`clamp_source_does_not_use_fmin_fmax_or_
    /// metal_clamp` と同方針）。
    #[test]
    fn hardswish_and_leaky_relu_do_not_use_fmin_fmax_or_metal_clamp() {
        let hs_src =
            unary_kernel_source(ScalarUnaryOp::Hardswish).expect("Hardswish must be implemented");
        assert!(!hs_src.contains("fmin("));
        assert!(!hs_src.contains("fmax("));
        assert!(!hs_src.contains("metal::clamp("));

        let lr_src = unary_kernel_source(ScalarUnaryOp::LeakyRelu {
            negative_slope: 0.1,
        })
        .expect("LeakyRelu must be implemented");
        assert!(!lr_src.contains("fmin("));
        assert!(!lr_src.contains("fmax("));
        assert!(!lr_src.contains("metal::clamp("));
    }

    /// GELU（誤差関数版）のソースが REQ-8 境界チェック・prelude ヘルパ
    /// （`scalar_erf_f32`）を含み、`metal::precise::` のみを使うことを
    /// 固定する（イシュー #1713）。
    #[test]
    fn gelu_source_includes_bounds_check_and_erf_prelude() {
        let src = unary_kernel_source(ScalarUnaryOp::Gelu).expect("Gelu must be implemented");
        assert!(src.contains("if (idx < numel)"));
        assert!(src.contains("kernel void scalar_unary_gelu("));
        assert!(src.contains("inline float scalar_erf_f32("));
        assert!(src.contains("scalar_erf_f32(x * 0.70710678f)"));
        assert!(src.contains("metal::precise::exp("));
        assert!(!src.contains("fast::"));
    }

    /// GELU（tanh 近似版）のソースが REQ-8 境界チェック・
    /// `metal::precise::tanh` を含み、prelude ヘルパを持たない（helper
    /// 不要）ことを固定する。
    #[test]
    fn gelu_tanh_source_includes_bounds_check_and_precise_tanh_without_prelude() {
        let src =
            unary_kernel_source(ScalarUnaryOp::GeluTanh).expect("GeluTanh must be implemented");
        assert!(src.contains("if (idx < numel)"));
        assert!(src.contains("kernel void scalar_unary_gelu_tanh("));
        assert!(src.contains("metal::precise::tanh("));
        assert!(!src.contains("inline float"));
    }

    /// GELU 2 kind はペイロードを持たないため `op` に依存しない
    /// （`sqrt_source_is_payload_independent` と同型）。
    #[test]
    fn gelu_sources_are_payload_independent() {
        assert_eq!(
            unary_kernel_source(ScalarUnaryOp::Gelu),
            unary_kernel_source(ScalarUnaryOp::Gelu)
        );
        assert_eq!(
            unary_kernel_source(ScalarUnaryOp::GeluTanh),
            unary_kernel_source(ScalarUnaryOp::GeluTanh)
        );
    }

    /// Softplus のソースが REQ-8 境界チェック・`p0`／`p1` payload
    /// 宣言（値は埋め込まない）・prelude ヘルパ（`scalar_log1p_f32`）を
    /// 含むことを固定する（`clamp_source_declares_payload_params_and_
    /// omits_values` と同型）。
    #[test]
    fn softplus_source_declares_payload_params_and_log1p_prelude() {
        let src = unary_kernel_source(ScalarUnaryOp::Softplus {
            beta: 0.123,
            threshold: 4.567,
        })
        .expect("Softplus must be implemented");
        assert!(src.contains("constant float& p0 [[buffer(3)]]"));
        assert!(src.contains("constant float& p1 [[buffer(4)]]"));
        assert!(!src.contains("0.123"));
        assert!(!src.contains("4.567"));
        assert!(src.contains("if (idx < numel)"));
        assert!(src.contains("kernel void scalar_unary_softplus("));
        assert!(src.contains("inline float scalar_log1p_f32("));
        assert!(src.contains("metal::precise::exp("));
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

    /// prelude は該当 kind 限定で挿入され、既存 kind（`Sqrt`）の生成
    /// ソースには prelude ヘルパが一切含まれない（既存生成ソース不変＝
    /// bit 同一の非後退契約。モジュール doc「ペイロード seam」と同型）
    /// ことを固定する。
    #[test]
    fn prelude_is_empty_for_kinds_that_do_not_need_it() {
        let sqrt_src = unary_kernel_source(ScalarUnaryOp::Sqrt).expect("Sqrt implemented");
        assert!(!sqrt_src.contains("inline float"));
        let clamp_src = unary_kernel_source(ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 })
            .expect("Clamp implemented");
        assert!(!clamp_src.contains("inline float"));
    }
}
