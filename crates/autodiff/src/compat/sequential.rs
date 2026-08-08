//! Keras `Sequential` 慣習のレイヤー積み上げビルダー（TASK-9.2a・
//! #95）。数値ロジックは一切持たず、`nn::Module` 実装
//! （`Linear`・`Relu`・`Sigmoid`・`Tanh`）をメソッドチェーンで積み上げ、
//! `forward`/`predict` で `nn::Module::forward` へ委譲するだけの薄い
//! ビルダー（REQ-9）。対象レイヤーは `docs/compat-api-scope.md` §1 の
//! 3 種限定（Linear・ReLU/Sigmoid/Tanh。Softmax・GELU・Conv 等は範囲
//! 拡張の手続き〈同 §5〉を経ずに追加しない）。
//!
//! **学習（勾配取得・パラメータ更新）は対象外**: `add_linear` が内部で
//! 保持する `Linear` は `LinearVars`（勾配取得の入口。`Tape::backward`
//! 後に `Gradients::get(&vars.weight)` する経路）を `Sequential` の外へ
//! 公開しないため、`Sequential` 経由でパラメータ・勾配へアクセスする
//! 手段が構造的にない。受け入れ条件（#95）は「Sequential でのモデル
//! 構築・推論が動作する」のみであり、学習 API（optimizer 接続）は
//! 本イシューのスコープ外として PR 本文に記録する
//! （`.claude/rules/out-of-scope-tracking.md`）。

use tensor_core::{BackendOps, Tensor};

use crate::error::AutodiffError;
use crate::nn::activation::{Relu, Sigmoid, Tanh};
use crate::nn::{Linear, Module};
use crate::tape::Tape;
use crate::var::Var;

/// Keras `Sequential` 慣習のレイヤー積み上げビルダー。`add_*` はメソッド
/// チェーン（`self` を消費し `Self` を返す）で層を追加し、`predict` で
/// 推論を実行する。層は `nn::Module`（`crate::nn::module`）実装として
/// `Vec<Box<dyn Module>>` に格納するため、種類の異なる層（`Linear` と
/// 活性化関数）を同じ列で扱える。
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    /// 全結合層を追加する（bias あり既定。PyTorch `nn.Linear` の既定
    /// `bias=True` と揃える）。`Linear::new` が `Result` を返す
    /// （`in_features == 0` を拒否する。`nn/linear.rs` 参照）ため、
    /// 本メソッドも `Result<Self, AutodiffError>` を返し `?` で連鎖
    /// できるようにする。
    pub fn add_linear(
        mut self,
        in_features: usize,
        out_features: usize,
        seed: u64,
    ) -> Result<Self, AutodiffError> {
        let linear = Linear::new(in_features, out_features, true, seed)?;
        self.layers.push(Box::new(linear));
        Ok(self)
    }

    /// ReLU 層を追加する（`nn::activation::Relu`。shape 不変の演算のため
    /// 構造的に失敗しえず `Result` を返さない）。
    pub fn add_relu(mut self) -> Self {
        self.layers.push(Box::new(Relu));
        self
    }

    /// シグモイド層を追加する（`nn::activation::Sigmoid`）。
    pub fn add_sigmoid(mut self) -> Self {
        self.layers.push(Box::new(Sigmoid));
        self
    }

    /// 双曲線正接層を追加する（`nn::activation::Tanh`）。
    pub fn add_tanh(mut self) -> Self {
        self.layers.push(Box::new(Tanh));
        self
    }

    /// 積み上げた層を先頭から順に `Module::forward` へ委譲する。
    /// 呼び出し元が用意した `tape` 上で 1 回分の forward を計算する
    /// （`Linear::bind` がステップごとに葉ノードを登録し直す契約に従う。
    /// `nn/module.rs` 参照）。外部 `Tape` 上で呼ぶことで
    /// `Tape::backward` までグラフ記録がつながる（推論だけでなく
    /// grad check 等の用途にも使える）。
    pub fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let mut current = *input;
        for layer in &self.layers {
            current = layer.forward(tape, &current)?;
        }
        Ok(current)
    }

    /// 推論の入口（受け入れ条件「Sequential でのモデル構築・推論が
    /// 動作する」を満たす API）。内部で 1 ステップ分の `Tape` を生成し
    /// `forward` を呼んだ後 `to_tensor()` で追跡を外した
    /// `Tensor<f32>` を返す（`Tape` はこの呼び出しのスコープ内で破棄
    /// される。`nn/linear.rs` の「`Tape` はステップごとに生成・破棄
    /// される前提」と同じ運用）。
    ///
    /// **TASK-12.1d（#164）**: `Tape::new(ops)` の破壊的変更に伴い `ops`
    /// を引数で受け取る（`autodiff` は具体バックエンドへ依存しないため
    /// `backend-cpu` 等の解決は呼び出し元の責務。`docs/
    /// fusion-graph-design.md` §3.4「`Device` → 具体 `BackendOps` の
    /// 構築・結線は `facade` クレート（TASK-9.3）が担う」。`facade` 未
    /// 実装の現時点では呼び出し元がテスト用フィクスチャ等を直接渡す）。
    /// 無引数版が必要な場合は [`Sequential::predict`] を使う（codex-review
    /// 第 19〜21 波・PR #403 の P1 是正で本メソッドから改名した compat 経路）。
    pub fn predict_with_ops(
        &self,
        input: &Tensor<f32>,
        ops: Box<dyn BackendOps + Send>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let tape = Tape::new(ops);
        let input_var = tape.var(input);
        let output = self.forward(&tape, &input_var)?;
        Ok(output.to_tensor())
    }

    /// 推論の入口（無引数 `ops` 版。codex-review 第 19〜21 波・PR #403 の
    /// P1 是正で追加した compat 経路）。`default_ops::naive_ops()`（`eval.rs`
    /// へ委譲する naive CPU 参照実装。具体バックエンドクレートには
    /// 依存しない）を `ops` に使い [`Sequential::predict_with_ops`] へ
    /// 委譲する。性能が必要な呼び出し元は `predict_with_ops` へ最適化済み
    /// `BackendOps` を明示的に渡すこと（`crate::default_ops` モジュール
    /// 冒頭コメント参照）。
    pub fn predict(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
        self.predict_with_ops(input, crate::default_ops::naive_ops())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::dense_vec;

    const SEED1: u64 = 1001;
    const SEED2: u64 = 2002;

    #[test]
    fn sequential_builds_and_predicts_two_layer_mlp() {
        // 受け入れ条件（#95）: Sequential でのモデル構築・推論が動作する。
        let model = Sequential::new()
            .add_linear(8, 16, SEED1)
            .unwrap()
            .add_relu()
            .add_linear(16, 4, SEED2)
            .unwrap();

        let batch = 3;
        let input = Tensor::new(vec![0.1_f32; batch * 8], &[batch, 8]).unwrap();
        let output = model
            .predict_with_ops(&input, crate::test_support::test_ops())
            .unwrap();

        assert_eq!(output.shape(), &[batch, 4]);
    }

    #[test]
    fn sequential_forward_matches_manual_forward_bit_exact() {
        // 同一シード・同一演算列で組んだ手動 forward（nn::Linear +
        // nn::activation を直接呼ぶ経路）と Sequential::forward の出力が
        // ビット一致することを確認する（同一演算列のため tolerance は
        // 新設しない。coding-rust.md「許容誤差を単独で緩和しない」の
        // 趣旨に沿い、ここでは緩和ではなく完全一致で判定する）。
        let linear1 = Linear::new(8, 16, true, SEED1).unwrap();
        let linear2 = Linear::new(16, 4, true, SEED2).unwrap();

        let batch = 2;
        let input_data: Vec<f32> = (0..batch * 8).map(|i| i as f32 * 0.05).collect();
        let input_tensor = Tensor::new(input_data, &[batch, 8]).unwrap();

        // 手動経路: Linear -> ReLU -> Linear を直接組み立てる。
        let manual_tape = Tape::new(crate::test_support::test_ops());
        let manual_input = manual_tape.var(&input_tensor);
        let h = linear1.bind(&manual_tape).forward(&manual_input).unwrap();
        let h = h.relu();
        let manual_output = linear2.bind(&manual_tape).forward(&h).unwrap();

        // Sequential 経路: 同じ Linear インスタンスを Module として積む。
        let model = Sequential {
            layers: vec![Box::new(linear1), Box::new(Relu), Box::new(linear2)],
        };
        let seq_tape = Tape::new(crate::test_support::test_ops());
        let seq_input = seq_tape.var(&input_tensor);
        let seq_output = model.forward(&seq_tape, &seq_input).unwrap();

        assert_eq!(
            dense_vec(&manual_output.to_tensor()),
            dense_vec(&seq_output.to_tensor())
        );
    }

    #[test]
    fn sequential_forward_on_external_tape_supports_backward() {
        // グラフ記録の整合確認: Sequential::forward を外部 Tape 上で
        // 実行し、Tape::backward まで通ることを検証する。
        let model = Sequential::new()
            .add_linear(4, 3, SEED1)
            .unwrap()
            .add_sigmoid();

        let tape = Tape::new(crate::test_support::test_ops());
        let input = tape.var(&Tensor::new(vec![0.1_f32, 0.2, 0.3, 0.4], &[1, 4]).unwrap());
        let output = model.forward(&tape, &input).unwrap();
        let loss = output.sum(None).unwrap();

        let grads = tape.backward(&loss).unwrap();
        // 入力自身の勾配が取得できること（グラフが入力ノードまで
        // つながっていることの確認）。
        let input_grad = grads
            .get(&input)
            .unwrap()
            .expect("入力は loss に寄与している");
        assert_eq!(input_grad.shape(), input.to_tensor().shape());
    }

    #[test]
    fn add_linear_propagates_invalid_argument() {
        // in_features == 0 は Linear::new が拒否する
        // （1/sqrt(in_features) が非有限になるため。nn/linear.rs 参照）。
        // add_linear はこれをそのまま Result で伝播する。
        // `Sequential` は `Box<dyn Module>` を保持し `Debug` を実装しない
        // ため、`unwrap_err()`（`Ok` 側にも `Debug` を要求する）は使わず
        // `match` で値を取り出す。
        let result = Sequential::new().add_linear(0, 4, SEED1);
        match result {
            Err(err) => assert!(matches!(err, AutodiffError::InvalidArgument(_))),
            Ok(_) => panic!("in_features == 0 は拒否されるはず"),
        }
    }
}
