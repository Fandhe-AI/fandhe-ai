//! 共通 `Module` trait（TASK-9.2a・#95）。
//!
//! `Sequential`（当時 `fandhe_ai_autodiff::compat::sequential`。TASK-9.4・#411 で
//! `fandhe_ai::compat::sequential` へ移設）がレイヤーの列を
//! `Vec<Box<dyn Module>>` として保持し、種類の異なる `nn` の部品
//! （`Linear`・活性化関数）を統一シグネチャで呼べるようにするための
//! 最小 trait。`nn/mod.rs` が「共通 `Module` trait の定義は
//! `compat::Sequential` 設計時に確定する」としていた点を本イシューで
//! 確定する。
//!
//! シグネチャに `tape: &'t Tape` を含める理由: `Linear` は
//! `Linear::bind(&tape)` でそのステップの葉ノードを毎回登録してから
//! でないと forward できない（`nn/linear.rs` の `Tape` ライフサイクル
//! 節参照）。活性化関数側は `tape` を使わないが、`Box<dyn Module>` を
//! 均一に扱うため同じ引数を受け取る。この「毎呼び出しで葉ノードを
//! 登録し直す」契約は推論・1 ステップ forward 用である。学習（勾配
//! 取得・パラメータ更新）は #294（`compat::Sequential::bind`・
//! `compat/sequential.rs` の `SequentialVars`）で対応済み: 下記
//! `as_linear`/`as_linear_mut` が `Sequential` から学習可能パラメータ
//! （`Linear`）を層順に取り出すためのダウンキャストフックを提供する。

use crate::error::AutodiffError;
use crate::eval;
use crate::nn::activation::{Relu, Sigmoid, Tanh};
use crate::nn::linear::Linear;
use crate::tape::Tape;
use crate::var::Var;
use fandhe_ai_tensor_core::{BackendError, BackendOps, Tensor, broadcast_shape, matmul_out_shape};

/// `nn` の部品（層・活性化関数）に共通の forward シグネチャ。
pub trait Module {
    /// このステップの `tape` 上で 1 回分の forward を計算する。
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError>;

    /// `tape`（葉ノード登録・演算記録）を経由せず、`ops` を直接呼んで
    /// 1 回分の forward をホスト常駐 `Tensor` で計算する（イシュー
    /// #1028・`docs/inference-forward-fixed-cost-design.md` §3.1「段階
    /// A」）。推論専用の tape 不要経路（`compat::Sequential::predict`
    /// が呼ぶ）が、[`Self::forward`] と同じ演算列・同じ丸め（bit
    /// 完全一致）を保ちつつ、`Tape::var` の葉クローン（`Linear` の
    /// `weight`/`bias` を毎呼び出しで clone する固定費。`nn/linear.rs`
    /// の `Linear::bind` 参照）とノード記録のアロケーションを回避する
    /// ために追加した。
    ///
    /// # デフォルト実装
    ///
    /// `fandhe-ai-autodiff` は crates.io 公開クレートであり
    /// （`docs/crates-io-naming-decision.md`）、本メソッドは非破壊拡張
    /// （デフォルトメソッド追加。外部実装者の既存 `impl Module` を壊さ
    /// ない）とする。既定は [`BackendError::Unsupported`] を返す
    /// fail-safe（本クレート内 4 実装〈`Linear`・`Relu`・`Sigmoid`・
    /// `Tanh`〉はいずれもこのデフォルトをオーバーライドする。呼び出し元
    /// が独自の `Module` 実装をこの経路で使う場合、`Unsupported` を
    /// フォールバックの合図として扱うこと）。
    fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        _input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Err(AutodiffError::Backend(BackendError::Unsupported(
            "Module::forward_host: default fail-safe (tape-free forward not implemented for \
             this Module)"
                .into(),
        )))
    }

    /// 学習可能パラメータを持つ層（現状 `Linear` のみ）への読み取り
    /// アクセスフック。既定実装は `None`（活性化関数など無状態の層は
    /// オーバーライドしない）。`std::any::Any` による動的ダウンキャスト
    /// ではなくこの明示的フックを選ぶ理由: `compat` 層が対象とするレイヤー
    /// 集合は `docs/compat-api-scope.md` §1 で 3 種（Linear・
    /// ReLU/Sigmoid/Tanh）に閉じており、種類を増やすたびに `Any` の
    /// ダウンキャスト先を推測する曖昧さを避け、対応する層がここに
    /// 列挙されているかどうかで閉集合であることをコードとして保つため
    /// （#294。呼び出し元は `compat::Sequential::bind`/
    /// `trainable_parameters`/`apply_parameters`）。
    fn as_linear(&self) -> Option<&Linear> {
        None
    }

    /// [`Module::as_linear`] の可変版。`compat::Sequential::apply_parameters`
    /// が optimizer 更新後の `Tensor<f32>` を層へ書き戻す入口として使う。
    fn as_linear_mut(&mut self) -> Option<&mut Linear> {
        None
    }

    /// この層が `ReLU` かどうか（イシュー #1044・`docs/kernel-fusion.md`
    /// §2.2「学習経路への結線」）。`as_linear` と同じ明示列挙方式
    /// （`docs/compat-api-scope.md` §1 の閉集合維持。`Any` ダウンキャスト
    /// は使わない）で、`fandhe_ai_facade::compat::sequential::Sequential`
    /// が forward 中に「次層が `ReLU` か」を先読みし、`Linear` 層を
    /// `LinearVars::forward_with_activation(input, Activation::Relu)`
    /// （1 ノード・1 カーネル起動）へ結線して `ReLU` 層自体をスキップ
    /// するかどうかを判定する。既定は `false`（`Linear`／
    /// `Sigmoid`／`Tanh` はオーバーライドしない。`Sigmoid`／`Tanh` は
    /// `BackendOps::gemm_bias_act` の `Activation` に対応する variant を
    /// 持たないため融合対象外）。
    fn as_relu(&self) -> bool {
        false
    }
}

/// `Linear::bind(tape)` で当該ステップの葉ノードを登録してから
/// `LinearVars::forward` を呼ぶ（`nn/linear.rs` 参照）。
impl Module for Linear {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input)
    }

    fn as_linear(&self) -> Option<&Linear> {
        Some(self)
    }

    fn as_linear_mut(&mut self) -> Option<&mut Linear> {
        Some(self)
    }

    /// [`Module::forward`]（`Linear::bind(tape).forward(input)`。
    /// `LinearVars::forward` が `input.matmul(&weight)` → `.add(&bias)`
    /// と非融合合成する）と **同一の演算列**（`ops.gemm` → `ops.add`）を
    /// 直接呼ぶ。融合カーネル（`ops.gemm_bias_act`）を使わない理由は
    /// bit-exactness 契約（`Module::forward_host` doc 参照）: 融合
    /// epilogue はカーネル内 tiling 次第で加算順序が変わりうるため、
    /// 旧経路と厳密に同じ累積順序を保証できるのは非融合合成のみ。
    ///
    /// **エラー型の一致契約（review 指摘）**: `Var::matmul`/`add`（tape
    /// 経路。`var.rs`）は shape 不整合を `matmul_out_shape`/
    /// `broadcast_shape` で `ops.gemm`/`ops.add` 呼び出し**前**に検査し
    /// `AutodiffError::Shape` として返す。本メソッド（tape 不要経路）が
    /// この事前検査を省いて `ops.gemm`/`ops.add` の `?` に任せると、同じ
    /// shape 不整合が `BackendError::ShapeMismatch` 経由の
    /// `AutodiffError::Backend` として返り、`compat::Sequential::predict`
    /// のフォールバック判定対象外の経路で `AutodiffError` の variant が
    /// 呼び出し元から見て変わってしまう（旧経路と新経路で同じ入力が
    /// 異なるエラー variant を返す）。それを避けるため、tape 経路と
    /// 同じ関数で同じ順序に事前検査してから `ops` を呼ぶ。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        matmul_out_shape(input.shape(), self.weight().shape())?;
        let y = ops.gemm(input, self.weight())?;
        match self.bias() {
            Some(bias) => {
                broadcast_shape(y.shape(), bias.shape())?;
                Ok(ops.add(&y, bias)?)
            }
            None => Ok(y),
        }
    }
}

/// `Relu::forward`（shape 不変の単項演算のため構造的に失敗しえない）を
/// `Result` へ包むだけの委譲。`tape` は使わない（`nn/activation.rs`
/// 参照）。
impl Module for Relu {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(Relu::forward(self, input))
    }

    fn as_relu(&self) -> bool {
        true
    }

    /// `Var::relu()`（`nn::activation::Relu::forward`）が呼ぶ
    /// `tape.ops().relu(...)` と同一のディスパッチ（`ops.relu`）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Ok(ops.relu(input)?)
    }
}

/// `Sigmoid::forward` への委譲。`Relu` の実装と同じ理由で `tape` は
/// 使わない。
impl Module for Sigmoid {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(Sigmoid::forward(self, input))
    }

    /// `Var::sigmoid()` は `BackendOps` ディスパッチを経由せず
    /// `eval::sigmoid`（ホスト直接計算）を呼ぶ（`var.rs` 参照）。
    /// bit-exactness のため同じ `eval::sigmoid` を直接呼ぶ。
    fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Ok(eval::sigmoid(input))
    }
}

/// `Tanh::forward` への委譲。`Relu` の実装と同じ理由で `tape` は
/// 使わない。
impl Module for Tanh {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(Tanh::forward(self, input))
    }

    /// `Var::tanh()` も `Sigmoid` と同様 `eval::tanh`（ホスト直接計算）を
    /// 呼ぶ。bit-exactness のため同じ関数を直接呼ぶ。
    fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Ok(eval::tanh(input))
    }
}

#[cfg(test)]
mod tests {
    //! `Module::forward` が既存の直接呼び出し（`Linear::bind().forward()`・
    //! `Relu::forward()` 等）と同一の値・テープ記録を返すことを検証する
    //! （「薄いラッパー性」の担保）。

    use super::*;
    use crate::eval::dense_vec;
    use fandhe_ai_tensor_core::Tensor;

    #[test]
    fn linear_module_forward_matches_bind_forward() {
        let linear = Linear::new(3, 2, true, 42).expect("seed=42 は有効な構築引数");
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap());

        let via_module = <Linear as Module>::forward(&linear, &tape, &x).unwrap();
        let via_direct = linear.bind(&tape).forward(&x).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn relu_module_forward_matches_direct_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());

        let via_module = <Relu as Module>::forward(&Relu, &tape, &x).unwrap();
        let via_direct = Relu.forward(&x);

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn sigmoid_module_forward_matches_direct_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());

        let via_module = <Sigmoid as Module>::forward(&Sigmoid, &tape, &x).unwrap();
        let via_direct = Sigmoid.forward(&x);

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn tanh_module_forward_matches_direct_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());

        let via_module = <Tanh as Module>::forward(&Tanh, &tape, &x).unwrap();
        let via_direct = Tanh.forward(&x);

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }
}
