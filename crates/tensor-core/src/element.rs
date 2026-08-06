//! テンソルの要素型抽象化。
//!
//! `tensor.rs` の生成系 API（`zeros`/`ones`/`full`）がジェネリックな
//! `Tensor<T>` を返すには、`T` 自身が加法単位元・乗法単位元を生成できる
//! 必要がある。`Element` はこの capability を型境界として表現する
//! （spec 根拠: `docs/public-api-design.md` §2.3）。
//!
//! 実装対象は `f32`/`f64`/`i32`/`half::f16`（GPU バックエンド CUDA/Metal
//! が使用する半精度浮動小数点型）。

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
