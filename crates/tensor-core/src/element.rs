//! テンソルの要素型抽象化。
//!
//! `tensor.rs` の生成系 API（`zeros`/`ones`/`full`）がジェネリックな
//! `Tensor<T>` を返すには、`T` 自身が加法単位元・乗法単位元を生成できる
//! 必要がある。`Element` はこの capability を型境界として表現する
//! （spec 根拠: `docs/public-api-design.md` §2.3）。
//!
//! 実装対象は `f32`/`f64`/`i32`/`half::f16`（GPU バックエンド CUDA/Metal
//! が使用する半精度浮動小数点型）に加え、`i64`（ONNX `INT64` テンソル・
//! `onnx-interop` の shape／インデックス系値を型安全に表現するため。イシュー
//! #274）・`bool`（ONNX `BOOL` テンソルのデコード表現用。同じくイシュー #274）。
//! `i64`/`bool` はいずれも `onnx-interop` の形状系オペ（コピーのみで算術を伴わない）
//! でのみ用いる想定であり、`backend_ops`／`dispatch` の算術カーネル dispatch 対象には
//! 含めない（算術・backend dispatch 対応は本イシューのスコープ外。#274 実装計画 §7）。

use half::f16;

/// テンソルが扱える要素型の最小抽象化。
///
/// `Copy + Send + Sync + Debug + PartialEq + 'static` に加え、`zero()`/
/// `one()` を追加境界として要求する（`docs/public-api-design.md` §2.3
/// で「具体的なシグネチャは TASK-1.4 productize 時に確定する」とされた
/// 箇所を本イシュー（TASK-1.4a）で確定する）。
pub trait Element: Copy + Send + Sync + std::fmt::Debug + PartialEq + 'static {
    /// 加法単位元（`zeros` が使用する）。
    fn zero() -> Self;
    /// 乗法単位元（`ones` が使用する）。
    fn one() -> Self;
}

impl Element for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl Element for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl Element for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

impl Element for f16 {
    fn zero() -> Self {
        f16::ZERO
    }
    fn one() -> Self {
        f16::ONE
    }
}

impl Element for i64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

impl Element for bool {
    fn zero() -> Self {
        false
    }
    fn one() -> Self {
        true
    }
}
