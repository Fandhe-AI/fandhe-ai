//! ユーザー定義 forward／backward プラグイン機構（`Op::Custom`）。
//!
//! 案 B（trait object variant。`docs/autodiff-custom-function-decision.md`
//! §12.4 で確定）の実装本体。`tape::Op` は `pub(crate)` のクローズド
//! enum（約 69 variant）だが、利用者が独自の forward／backward 対
//! （straight-through estimator・gradient reversal・独自安定式等）を
//! [`crate::Tape::custom`] 経由でグラフへ登録できるようにするための
//! 唯一の拡張口をここに設ける。
//!
//! **内部クレート限定の公開範囲（§12.5 (a)）**: 本 trait・
//! [`crate::Tape::custom`] は `fandhe_ai_autodiff` クレート内の `pub`
//! API であり crates.io 公開クレートの一部になるが、facade（唯一の
//! サポート対象公開面。`docs/compat-api-scope.md` §0）は再エクスポート
//! しない（`crates/facade/tests/api_surface.rs` の否定ガードで機械
//! 固定する）。facade 公開（§12.5 (b)）は本イシューの対象外で、別途
//! ユーザー承認を得てから着手する。

use std::fmt;
use std::sync::Arc;

use fandhe_ai_tensor_core::Tensor;

use crate::error::AutodiffError;

/// ユーザー定義の forward／backward 対（`docs/autodiff-custom-function-
/// decision.md` §12.4）。[`crate::Tape::custom`]（`tape.rs`）経由で
/// テープへ登録する唯一の実装口。
///
/// **契約**（§12.4 表・§6「数値契約」）:
/// - `forward`／`backward` は host `Tensor<f32>` のみを受け渡す
///   （`BackendOps` 非露出。REQ-12 §7 読み (i) に整合。デバイス側
///   カーネルへの直接アクセスは与えない）。
/// - 決定的・副作用なしの純関数として実装すること。同一ノードの
///   `backward` は `Tape::backward_accumulate`（#1749）・
///   `retain_graph` 常時保持契約（グラフを `reset`／drop まで保持する
///   契約）により複数回呼ばれうる。
/// - `'static` 境界により `&Tape`・`Var<'t>` を捕捉できない
///   （backward 中に `Tape` を再入すると `RefCell` 二重可変借用の
///   panic になるため、型で構造的に禁止する。§3 項 7）。
/// - 数値一致の複合判定（REQ-2）の対象外——host 実行のみで、丸め
///   契約・バックエンド間一致はユーザー実装の責任とする。
/// - GPU 上に構築した `Tape` でも常に host 実行になる（性能は非保証。
///   §9 のスコープ外整理）。
pub trait CustomFunction: Send + Sync + 'static {
    /// ログ・エラーメッセージ表示用の識別名。`Op::Custom` の手書き
    /// `Debug` 実装（`CustomFn`）が参照する。
    fn name(&self) -> &str;

    /// 入力 shape 列から出力 shape を宣言する。`forward` の実出力
    /// shape との不一致は `Tape::custom` が fail-closed
    /// （`AutodiffError::Shape`）で検出する（§6「shape 検証」）。
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError>;

    /// forward 計算。`inputs` は `Tape::custom` 呼び出し時点で層 1
    /// （`materialize_fallible`）により実体化済みの値。
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError>;

    /// backward 計算。戻り値は `inputs` と同じ長さ・順序。
    ///
    /// `requires_grad` は `inputs` と同じ長さで、各要素は呼び出し元
    /// （`grad::vjp`）が該当入力について判断済みの要否
    /// （`TapeNode::requires_grad` 前方伝播の結果。イシュー #1748・
    /// `docs/autodiff-nograd-leaf-dinput-skip-decision.md`）を渡す。
    /// `requires_grad[i] == false` の入力は計算を省略し `None` を
    /// 返してよい（`Some` を返すこと自体は禁止しない——`false` の入力
    /// への寄与は呼び出し元〈`backward_impl`〉が破棄する）。
    /// `requires_grad[i] == true` の入力に対して `None` を返した場合は
    /// fail-closed で `Err`（勾配欠落をユーザー実装の不備として
    /// 早期検出するため）。`Some` の要素は対応する入力と同じ shape で
    /// なければならず、不一致も fail-closed で拒否する。
    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError>;
}

/// [`crate::tape::Op::Custom`] が保持する newtype（`tape.rs`）。
///
/// `Op` 全体は `#[derive(Debug, Clone)]`（`grad::vjp` が `op.clone()`
/// して網羅 match するため `Clone` は必須）を維持したまま `Op::Custom`
/// 腕だけをこの型で表現する。`Arc<dyn CustomFunction>` は `Debug` を
/// 実装できない（trait object・ユーザー実装に `Debug` を要求しない
/// 設計）ため、`derive(Debug)` をそのまま `Op` へ適用できず、本型に
/// 手書き `Debug` を与えることで `Op` 全体の `derive(Debug)` を崩さず
/// 済ませている（§12.4「`Op: Debug` との整合」）。`Clone` は
/// `Arc::clone`（参照カウント複製のみ）で安価なため `derive` のまま
/// でよい。
#[derive(Clone)]
pub(crate) struct CustomFn(pub(crate) Arc<dyn CustomFunction>);

impl fmt::Debug for CustomFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomFn")
            .field("name", &self.0.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::Tensor;

    struct NamedFn;

    impl CustomFunction for NamedFn {
        fn name(&self) -> &str {
            "my_custom_op"
        }
        fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
            Ok(input_shapes[0].to_vec())
        }
        fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
            Ok((*inputs[0]).clone())
        }
        fn backward(
            &self,
            _inputs: &[&Tensor<f32>],
            _out_value: &Tensor<f32>,
            upstream: &Tensor<f32>,
            _requires_grad: &[bool],
        ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
            Ok(vec![Some(upstream.clone())])
        }
    }

    /// `Op` 全体の `#[derive(Debug)]` が `Op::Custom` 腕を表示する際、
    /// ユーザー実装自体に `Debug` を要求せず `.name()` のみを出力する
    /// （§12.4「`Op: Debug` との整合」）。
    #[test]
    fn custom_fn_debug_shows_name_only() {
        let f = CustomFn(Arc::new(NamedFn));
        let debug_str = format!("{f:?}");
        assert_eq!(debug_str, r#"CustomFn { name: "my_custom_op" }"#);
    }
}
