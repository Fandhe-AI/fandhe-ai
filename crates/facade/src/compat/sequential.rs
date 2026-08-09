//! Keras `Sequential` 慣習のレイヤー積み上げビルダー（TASK-9.2a・
//! #95。TASK-9.4・#411 で `autodiff::compat` から本クレートへ移設）。
//! 数値ロジックは一切持たず、`autodiff::nn::Module` 実装
//! （`Linear`・`Relu`・`Sigmoid`・`Tanh`）をメソッドチェーンで積み上げ、
//! `forward`/`predict` で `nn::Module::forward` へ委譲するだけの薄い
//! ビルダー（REQ-9）。対象レイヤーは `docs/compat-api-scope.md` §1 の
//! 3 種限定（Linear・ReLU/Sigmoid/Tanh。Softmax・GELU・Conv 等は範囲
//! 拡張の手続き〈同 §5〉を経ずに追加しない）。
//!
//! **学習（勾配取得・パラメータ更新。#294 で対応済み）**: [`Sequential::bind`]
//! が返す [`SequentialVars`] を経由して `LinearVars`（勾配取得の入口。
//! `Tape::backward` 後に `Gradients::get(&vars.weight)` する経路）へ
//! アクセスできる。[`Sequential::trainable_parameters`]/
//! [`Sequential::apply_parameters`] と組み合わせ、`autodiff::optim::Sgd`・
//! `autodiff::nn::optim::AdamW` の位置対応契約にそのまま渡せる。
//!
//! **`predict` の既定結線（TASK-9.4・#411）**: `predict` は本クレートの
//! composition root（[`crate::tape`]。既定 CPU・`CpuBackendOps`・融合
//! 有効）で `Tape` を構築して forward する。旧 `autodiff::compat` 版が
//! 依存していた naive 参照実装（`autodiff::default_ops::naive_ops()`。
//! クレート非公開）は facade から到達できないため、この結線先の変更に
//! 伴い旧 `predict_with_ops`（任意 `BackendOps` 注入経路）は公開面から
//! 撤去した（REQ-12「任意 `BackendOps` 実装を注入できる公開 API を
//! 設けない」・`crates/facade/tests/api_surface.rs` の機械検査と整合。
//! `compat/mod.rs` モジュール doc 参照）。ops を明示的に選びたい内部用途
//! は [`Sequential::forward`]（`&autodiff::Tape` を受け取るだけで
//! `BackendOps` は受け取らない）へ、呼び出し元が任意に構築した `Tape` を
//! 渡せば足りる。

use autodiff::nn::activation::{Relu, Sigmoid, Tanh};
use autodiff::nn::{Linear, LinearVars, Module};
use autodiff::{AutodiffError, Gradients, Tape, Var};
use tensor_core::Tensor;

/// Keras `Sequential` 慣習のレイヤー積み上げビルダー。`add_*` はメソッド
/// チェーン（`self` を消費し `Self` を返す）で層を追加し、`predict` で
/// 推論を実行する。層は `nn::Module`（`autodiff::nn::module`）実装として
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
    /// （`in_features == 0` を拒否する。`autodiff::nn::linear` 参照）ため、
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
    /// `autodiff::nn::module` 参照）。外部 `Tape` 上で呼ぶことで
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
    /// 動作する」を満たす API）。内部で [`crate::tape`]（本クレートの
    /// composition root。既定 CPU・`CpuBackendOps`・融合有効）で 1
    /// ステップ分の `Tape` を生成し `forward` を呼んだ後 `to_tensor()`
    /// で追跡を外した `Tensor<f32>` を返す（`Tape` はこの呼び出しの
    /// スコープ内で破棄される。`autodiff::nn::linear` の「`Tape` は
    /// ステップごとに生成・破棄される前提」と同じ運用）。
    ///
    /// **TASK-9.4（#411）**: 旧 `autodiff::compat` 版が持っていた
    /// `predict_with_ops`（任意 `BackendOps` 注入経路）は本移設で公開面
    /// から撤去した（モジュール doc 参照）。
    pub fn predict(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
        let tape = crate::tape();
        let input_var = tape.var(input);
        let output = self.forward(&tape, &input_var)?;
        Ok(output.to_tensor())
    }

    /// 学習ステップの入口（#294）。このステップの `tape` へ全 `Linear`
    /// 層の `weight`/`bias` を葉ノードとして登録し（`Linear::bind` を
    /// 層順に呼ぶ）、学習用 forward・勾配取得の入口となる
    /// [`SequentialVars`] を返す。
    ///
    /// 返る `SequentialVars` は `&'m Sequential`（`self`）と `&'t Tape`
    /// の両方を借用する。このハンドルが生きている間は
    /// [`Sequential::apply_parameters`]（`&mut self` を要求）を呼べない
    /// （ステップ途中でのパラメータ書き換えを借用検査で静的に防ぐ設計。
    /// 呼び出し元は `bind` 以降の一連の処理をブロックスコープで囲み、
    /// スコープを抜けてから `apply_parameters` を呼ぶ運用とする。
    /// `crates/facade/tests/compat_sequential_train.rs` に実例がある）。
    pub fn bind<'m, 't>(&'m self, tape: &'t Tape) -> SequentialVars<'m, 't> {
        let linears = self
            .layers
            .iter()
            .filter_map(|layer| layer.as_linear())
            .map(|linear| linear.bind(tape))
            .collect();
        SequentialVars {
            model: self,
            linears,
        }
    }

    /// 学習可能パラメータ（`Linear` 層の `weight`/`bias`）への参照列を
    /// 層の追加順・各層内は weight → bias（`Some` の場合のみ）の順で
    /// 返す。`autodiff::optim::Sgd::step`/`autodiff::nn::optim::AdamW::step`
    /// の位置対応契約にそのまま渡せる。この順序契約は
    /// [`SequentialVars::trainable_vars`]/[`SequentialVars::trainable_grads`]/
    /// [`Sequential::apply_parameters`] と共通（#294 の設計不変条件）。
    pub fn trainable_parameters(&self) -> Vec<&Tensor<f32>> {
        let mut out = Vec::new();
        for layer in &self.layers {
            if let Some(linear) = layer.as_linear() {
                out.push(linear.weight());
                if let Some(bias) = linear.bias() {
                    out.push(bias);
                }
            }
        }
        out
    }

    /// optimizer（`Sgd::step`/`AdamW::step`）が返した更新後テンソル列を
    /// [`Sequential::trainable_parameters`] と同じ順序契約で各 `Linear`
    /// 層へ書き戻す。内部で `Linear::from_parameters`（`autodiff::nn::linear`）
    /// により層を再構築するため、compat 層自身は shape 検証ロジックを
    /// 重複実装しない（REQ-9「薄いラッパーに徹する」）。
    ///
    /// **`Linear::from_parameters` が検証する範囲（それ以上は検証しない）**:
    /// 検証は新しい `weight`/`bias` 単体の内部整合性（`weight` が rank 2・
    /// `weight.shape()[0] > 0`・`bias` を渡す場合は `bias.shape() ==
    /// [weight.shape()[1]]`）に限られ、**置換前の層の shape や隣接層との
    /// 整合性とは比較しない**（`autodiff::nn::linear::Linear::from_parameters`
    /// docstring 参照）。したがって例えば `in_features`/`out_features` を
    /// 変えてしまう更新（optimizer 側のバグ等）は `apply_parameters` 自身
    /// ではエラーにならず、後続の `forward`（`matmul` の shape 不整合）で
    /// 初めて顕在化する。
    ///
    /// # エラー
    /// - `updated` の要素数が `trainable_parameters()` の件数と不一致
    ///   → `AutodiffError::InvalidArgument`（fail-closed。位置対応契約が
    ///   崩れたまま一部の層だけ更新して黙って続行しない。`.claude/rules/
    ///   security.md` A03）
    /// - 個々のテンソルが上記の内部整合性検証に反する（例: bias の shape
    ///   が対応する weight の out_features と食い違う）→
    ///   `Linear::from_parameters` 由来の `AutodiffError::Shape`
    ///
    /// **アトミック性（two-pass）**: `updated` は 1 パス目で全層分を検証込みで
    /// `Linear::from_parameters` に通し、実際の代入（`self.layers` への
    /// 書き戻し）は全件の検証が成功した後の 2 パス目でまとめて行う。層ごとに
    /// 検証と代入を同時に行うと、途中の層（例: 2 層目）で shape 不整合エラーが
    /// 起きた際に、既に代入済みの前段の層は新パラメータのまま・未処理の後続層は
    /// 旧パラメータのままという不整合な混在状態がモデルに残ってしまう
    /// （呼び出し側は `Err` を受け取るにもかかわらず一部だけ更新されて
    /// 見える。fail-closed 違反。`.claude/rules/security.md` A03。
    /// review 指摘 #294）。two-pass にすることで、エラー時は呼び出し前の
    /// 状態を完全に維持する。
    pub fn apply_parameters(&mut self, updated: Vec<Tensor<f32>>) -> Result<(), AutodiffError> {
        // trainable_parameters() と同じ順序（層の追加順）で学習可能層への
        // 可変参照を先に集める。まだ何も書き換えない。
        let linears: Vec<&mut Linear> = self
            .layers
            .iter_mut()
            .filter_map(|layer| layer.as_linear_mut())
            .collect();

        let mut updated = updated.into_iter();
        // 1 パス目: 検証込みで新しい Linear を全層分構築する（代入は未実施）。
        let mut rebuilt = Vec::with_capacity(linears.len());
        for linear in &linears {
            let has_bias = linear.bias().is_some();
            let new_weight = updated.next().ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "Sequential::apply_parameters: updated has fewer elements than \
                     trainable_parameters() (weight missing)"
                        .to_string(),
                )
            })?;
            let new_bias = if has_bias {
                Some(updated.next().ok_or_else(|| {
                    AutodiffError::InvalidArgument(
                        "Sequential::apply_parameters: updated has fewer elements than \
                         trainable_parameters() (bias missing)"
                            .to_string(),
                    )
                })?)
            } else {
                None
            };
            rebuilt.push(Linear::from_parameters(new_weight, new_bias)?);
        }
        if updated.next().is_some() {
            return Err(AutodiffError::InvalidArgument(
                "Sequential::apply_parameters: updated has more elements than \
                 trainable_parameters()"
                    .to_string(),
            ));
        }
        // 2 パス目: ここに到達した時点で件数・shape 検証は全件完了しているため、
        // 代入自体は失敗し得ない。
        for (linear, new_linear) in linears.into_iter().zip(rebuilt) {
            *linear = new_linear;
        }
        Ok(())
    }
}

/// [`Sequential::bind`] が返す、1 学習ステップ分のテープ登録済み
/// ハンドル。`model`（`&'m Sequential`）は layer の走査順（活性化層と
/// `Linear` 層の混在列）を再現するため、`linears`（`Linear` 層のみを
/// 層順に抽出した `LinearVars` 列）は `Vec<Box<dyn Module>>` の添字とは
/// 別に独立して保持する。
pub struct SequentialVars<'m, 't> {
    model: &'m Sequential,
    linears: Vec<LinearVars<'t>>,
}

impl<'m, 't> SequentialVars<'m, 't> {
    /// 学習用 forward。`Linear` 層は [`Sequential::bind`] で既に
    /// 登録済みの `LinearVars`（`self.linears`。層出現順の添字対応）を
    /// 使い、活性化層は `Module::forward` へ委譲する。
    ///
    /// **`Linear::bind` を再度呼ばない理由**: `Module::forward`
    /// （`autodiff::nn::module`）の `Linear` 実装は呼び出しのたびに
    /// `self.bind(tape)` して新しい葉ノードを作る（推論用の使い捨て
    /// 契約）。学習用 forward がこの経路を使うと、`bind` 時点で
    /// 取得した `LinearVars`（[`SequentialVars::trainable_vars`]/
    /// [`SequentialVars::trainable_grads`] が参照する `Var`）が forward
    /// のテープ記録に含まれず、`Tape::backward` 後の `Gradients::get`
    /// が到達不能（`Ok(None)`）を返してしまう。そのため学習用 forward
    /// は必ず `self.linears` に保持済みの `LinearVars::forward` を使う。
    pub fn forward(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let mut current = *input;
        // `self.linears` は `model.layers` から `Linear` 層のみを同じ順序で
        // 抽出したもの（`Sequential::bind` 参照）のため、`Linear` 層に
        // 出会うたびにイテレータから 1 件ずつ消費すれば `model.layers`
        // 側の走査順と対応が取れる。添字アクセス（`self.linears[i]`）では
        // なくイテレータの `next()` を使う理由: 両者の対応は `bind`/
        // `forward` が同じ `as_linear().is_some()` 述語で層を数えている
        // ことに依存する不変条件であり、この不変条件自体は `self.model`
        // が `bind` 時点の `&'m Sequential` を保持し続ける（借用が生きた
        // ままの `SequentialVars` からは `model.layers` を書き換えられない）
        // ため現状は構造的に破れない。したがってこの `ok_or_else` 分岐は
        // 現状のコード構成では到達しない防御コードである。将来この関数・
        // `bind` のいずれかを書き換えて上記の対応関係が崩れた場合に、
        // 添字アクセスが本番経路で panic するのを避け fail-closed
        // （`InvalidArgument`）にするための保険として残す
        // （`.claude/rules/coding-rust.md` 本番経路 panic 禁止方針）。
        let mut linears = self.linears.iter();
        for layer in &self.model.layers {
            if layer.as_linear().is_some() {
                let vars = linears.next().ok_or_else(|| {
                    AutodiffError::InvalidArgument(
                        "SequentialVars::forward: bind 済み LinearVars が model.layers の \
                         Linear 層数より少ない（bind/forward 間の Linear 層数対応が崩れた）"
                            .to_string(),
                    )
                })?;
                current = vars.forward(&current)?;
            } else {
                current = layer.forward(tape, &current)?;
            }
        }
        Ok(current)
    }

    /// bind 済み `LinearVars` 列（層順）。受け入れ条件 1
    /// （「Sequential から学習可能パラメータを取得する API」）の本体。
    pub fn linears(&self) -> &[LinearVars<'t>] {
        &self.linears
    }

    /// [`Sequential::trainable_parameters`] と同一の順序契約（層順に
    /// weight → bias〈`Some` の場合のみ〉）で `Var` 参照列を返す。
    pub fn trainable_vars(&self) -> Vec<&Var<'t>> {
        let mut out = Vec::new();
        for vars in &self.linears {
            out.push(&vars.weight);
            if let Some(bias) = &vars.bias {
                out.push(bias);
            }
        }
        out
    }

    /// `Tape::backward` の結果 `grads` から、同じ順序契約
    /// （`trainable_vars`/`trainable_parameters` と共通）で勾配参照列を
    /// 収集する。
    ///
    /// 勾配が存在しないパラメータ（loss へ未到達。`Gradients::get` が
    /// `Ok(None)` を返す場合）は黙って除外せず `InvalidArgument` にする
    /// （fail-closed）: 除外してしまうと戻り値の件数が
    /// `trainable_parameters()` の件数より少なくなり、
    /// `Sgd::step`/`AdamW::step` の位置対応契約（`params[i]` ↔ `grads[i]`）
    /// が呼び出し元の意図と無関係にずれて、誤ったパラメータへ誤った
    /// 勾配を適用しかねないため（`.claude/rules/security.md` A03）。
    pub fn trainable_grads<'g>(
        &self,
        grads: &'g Gradients,
    ) -> Result<Vec<&'g Tensor<f32>>, AutodiffError> {
        let mut out = Vec::new();
        for vars in &self.linears {
            let weight_grad = grads.get(&vars.weight)?.ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "SequentialVars::trainable_grads: weight に到達する勾配がない \
                     (loss へ未到達)"
                        .to_string(),
                )
            })?;
            out.push(weight_grad);
            if let Some(bias) = &vars.bias {
                let bias_grad = grads.get(bias)?.ok_or_else(|| {
                    AutodiffError::InvalidArgument(
                        "SequentialVars::trainable_grads: bias に到達する勾配がない \
                         (loss へ未到達)"
                            .to_string(),
                    )
                })?;
                out.push(bias_grad);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED1: u64 = 1001;
    const SEED2: u64 = 2002;

    /// テスト専用の連続化ヘルパー（`tensor_core::Tensor` の `pub` API
    /// のみを使用。旧 `autodiff::eval::dense_vec`〈クレート非公開〉の
    /// 代替。`compat/array.rs` のテストヘルパーと同型）。
    fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
        t.contiguous()
            .as_slice()
            .expect("contiguous() 直後は必ず as_slice() が Some を返す")
            .to_vec()
    }

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
        let output = model.predict(&input).unwrap();

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
        let manual_tape = crate::tape();
        let manual_input = manual_tape.var(&input_tensor);
        let h = linear1.bind(&manual_tape).forward(&manual_input).unwrap();
        let h = h.relu();
        let manual_output = linear2.bind(&manual_tape).forward(&h).unwrap();

        // Sequential 経路: 同じ Linear インスタンスを Module として積む。
        let model = Sequential {
            layers: vec![Box::new(linear1), Box::new(Relu), Box::new(linear2)],
        };
        let seq_tape = crate::tape();
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

        let tape = crate::tape();
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
        // （1/sqrt(in_features) が非有限になるため。autodiff::nn::linear 参照）。
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

    // =================================================================
    // #294: 学習 API（optimizer 接続）
    // =================================================================

    #[test]
    fn bind_returns_linear_vars_in_layer_order_with_mixed_bias() {
        // bias 有無混在（第 1 層 bias あり・第 2 層 bias なし）で bind が
        // 層順どおり LinearVars を返すことを確認する。
        let l1 = Linear::new(4, 3, true, SEED1).unwrap();
        let l2 = Linear::new(3, 2, false, SEED2).unwrap();
        let model = Sequential {
            layers: vec![Box::new(l1), Box::new(Relu), Box::new(l2)],
        };

        let tape = crate::tape();
        let bound = model.bind(&tape);

        assert_eq!(bound.linears().len(), 2);
        assert!(bound.linears()[0].bias.is_some());
        assert!(bound.linears()[1].bias.is_none());
    }

    #[test]
    fn sequential_vars_forward_matches_manual_forward_bit_exact() {
        // SequentialVars::forward の出力が既存 Sequential::forward
        // （Module 経路）とビット一致することを検証する（tolerance は
        // 新設しない。既存 `sequential_forward_matches_manual_forward_bit_exact`
        // と同じ判定様式）。
        let batch = 2;
        let input_data: Vec<f32> = (0..batch * 8).map(|i| i as f32 * 0.05).collect();
        let input_tensor = Tensor::new(input_data, &[batch, 8]).unwrap();

        // Module 経路（既存の推論用 forward）。
        let module_model = Sequential {
            layers: vec![
                Box::new(Linear::new(8, 16, true, SEED1).unwrap()),
                Box::new(Relu),
                Box::new(Linear::new(16, 4, true, SEED2).unwrap()),
            ],
        };
        let module_tape = crate::tape();
        let module_input = module_tape.var(&input_tensor);
        let module_output = module_model.forward(&module_tape, &module_input).unwrap();

        // SequentialVars 経路（学習用 forward）。
        let train_model = Sequential {
            layers: vec![
                Box::new(Linear::new(8, 16, true, SEED1).unwrap()),
                Box::new(Relu),
                Box::new(Linear::new(16, 4, true, SEED2).unwrap()),
            ],
        };
        let train_tape = crate::tape();
        let bound = train_model.bind(&train_tape);
        let train_input = train_tape.var(&input_tensor);
        let train_output = bound.forward(&train_tape, &train_input).unwrap();

        assert_eq!(
            dense_vec(&module_output.to_tensor()),
            dense_vec(&train_output.to_tensor())
        );
    }

    #[test]
    fn trainable_parameters_order_contract_weight_then_bias_per_layer() {
        let model = Sequential::new()
            .add_linear(4, 3, SEED1) // bias あり
            .unwrap()
            .add_relu()
            .add_linear(3, 2, SEED2) // bias あり
            .unwrap();

        let params = model.trainable_parameters();
        // 層 1: weight, bias / 層 2: weight, bias の 4 件。
        assert_eq!(params.len(), 4);
        assert_eq!(params[0].shape(), &[4, 3]); // l1.weight
        assert_eq!(params[1].shape(), &[3]); // l1.bias
        assert_eq!(params[2].shape(), &[3, 2]); // l2.weight
        assert_eq!(params[3].shape(), &[2]); // l2.bias
    }

    #[test]
    fn apply_parameters_rejects_element_count_mismatch() {
        let mut model = Sequential::new().add_linear(4, 3, SEED1).unwrap();
        // trainable_parameters() は [weight, bias] の 2 件を要求するが
        // 1 件しか渡さない。
        let updated = vec![Tensor::new(vec![0.0f32; 12], &[4, 3]).unwrap()];
        let err = model.apply_parameters(updated).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn apply_parameters_propagates_bias_shape_mismatch() {
        let mut model = Sequential::new().add_linear(4, 3, SEED1).unwrap();
        // weight は rank 2 の妥当な shape（[4, 3]）だが、bias の shape が
        // weight.shape()[1]（out_features=3）と食い違う（[5]）ケース。
        // `Linear::from_parameters` の bias 検証（`autodiff::nn::linear`）
        // がこれを拒否することを確認する（`apply_parameters` は
        // 検証ロジックを重複実装せず委譲するだけ、という設計の裏付け）。
        let weight = Tensor::new(vec![0.0f32; 12], &[4, 3]).unwrap();
        let bad_bias = Tensor::new(vec![0.0f32; 5], &[5]).unwrap();
        let err = model.apply_parameters(vec![weight, bad_bias]).unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));
    }

    #[test]
    fn apply_parameters_leaves_model_unchanged_on_excess_element_count() {
        // two-pass 化の回帰テスト（review 指摘 #294 の Medium 項目、後半）:
        // 修正前は `if updated.next().is_some()` の超過件数チェックが
        // 全層への代入完了「後」に走っていたため、この分岐で `Err` を
        // 返すケースでも実際には全層が新パラメータへ更新済みという
        // 不整合（呼び出し側はエラー扱いなのにモデルは書き換わっている）
        // があった。two-pass では代入前に検証（超過チェックを含む）が
        // 完了するため、この場合もモデルは元の値のまま残ることを確認する。
        let mut model = Sequential::new().add_linear(2, 1, SEED1).unwrap();
        let original_params: Vec<Tensor<f32>> =
            model.trainable_parameters().into_iter().cloned().collect();

        // trainable_parameters() は [weight, bias] の 2 件を要求するが
        // 3 件目（余剰）を渡す。
        let weight = Tensor::new(vec![9.0f32, 9.0], &[2, 1]).unwrap();
        let bias = Tensor::new(vec![9.0f32], &[1]).unwrap();
        let extra = Tensor::new(vec![9.0f32], &[1]).unwrap();
        let err = model
            .apply_parameters(vec![weight, bias, extra])
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        let after_params = model.trainable_parameters();
        assert_eq!(original_params.len(), after_params.len());
        for (before, after) in original_params.iter().zip(after_params.iter()) {
            assert_eq!(dense_vec(before), dense_vec(after));
        }
    }

    #[test]
    fn apply_parameters_updates_weights_used_by_subsequent_predict() {
        // apply_parameters で書き戻した値が実際に以後の forward に
        // 反映されることを確認する（薄いラッパー性の実証: compat 層は
        // Linear::from_parameters への委譲のみで独自状態を持たない）。
        let mut model = Sequential::new().add_linear(2, 1, SEED1).unwrap();
        let zero_weight = Tensor::new(vec![0.0f32, 0.0], &[2, 1]).unwrap();
        let zero_bias = Tensor::new(vec![7.0f32], &[1]).unwrap();
        model
            .apply_parameters(vec![zero_weight, zero_bias])
            .unwrap();

        let input = Tensor::new(vec![1.0f32, 2.0], &[1, 2]).unwrap();
        let output = model.predict(&input).unwrap();
        // weight=0 のため matmul 項は 0、bias=7.0 のみが出力される。
        assert!((output.get(&[0, 0]).unwrap() - 7.0).abs() < 1e-6);
    }

    #[test]
    fn apply_parameters_updates_bias_less_layer() {
        // `apply_parameters` の has_bias == false 分岐（bias を渡さず
        // weight のみ消費する経路）を直接カバーする。`Sequential::add_linear`
        // は常に bias ありの層しか組み立てられないため（公開 API からは
        // 到達不能。review 指摘 #294）、bias なし `Linear::new` を内部
        // コンストラクタ（`Sequential { layers: ... }`。同一クレート内の
        // テストのみ使用可能）で直接組み込んで検証する。
        let linear = Linear::new(2, 1, false, SEED1).unwrap();
        let mut model = Sequential {
            layers: vec![Box::new(linear)],
        };

        // has_bias == false のため trainable_parameters()/apply_parameters()
        // は weight 1 件のみを要求する。
        assert_eq!(model.trainable_parameters().len(), 1);

        let zero_weight = Tensor::new(vec![0.0f32, 0.0], &[2, 1]).unwrap();
        model.apply_parameters(vec![zero_weight]).unwrap();

        let input = Tensor::new(vec![1.0f32, 2.0], &[1, 2]).unwrap();
        let output = model.predict(&input).unwrap();
        // weight=0・bias なしのため出力は 0.0。
        assert!((output.get(&[0, 0]).unwrap()).abs() < 1e-6);
    }

    #[test]
    fn apply_parameters_leaves_model_unchanged_on_mid_sequence_shape_error() {
        // two-pass 化（review 指摘 #294 の Medium 項目）の回帰テスト:
        // 2 層目で shape 不整合エラーが起きても、検証は代入前に全件完了する
        // ため 1 層目（先に走査される層）はエラー前の値のまま保持される
        // （部分適用によるモデルの不整合な混在状態を防ぐ。
        // `.claude/rules/security.md` A03）。
        let mut model = Sequential::new()
            .add_linear(2, 3, SEED1)
            .unwrap()
            .add_relu()
            .add_linear(3, 1, SEED2)
            .unwrap();

        let original_params: Vec<Tensor<f32>> =
            model.trainable_parameters().into_iter().cloned().collect();

        // 1 層目（weight, bias）は妥当な shape、2 層目の bias は
        // out_features=1 と食い違う shape（[5]）にして拒否させる。
        let l1_weight = Tensor::new(vec![9.0f32; 6], &[2, 3]).unwrap();
        let l1_bias = Tensor::new(vec![9.0f32; 3], &[3]).unwrap();
        let l2_weight = Tensor::new(vec![9.0f32; 3], &[3, 1]).unwrap();
        let l2_bad_bias = Tensor::new(vec![9.0f32; 5], &[5]).unwrap();

        let err = model
            .apply_parameters(vec![l1_weight, l1_bias, l2_weight, l2_bad_bias])
            .unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));

        // 1 層目を含め全パラメータがエラー前の値のまま残っていること。
        let after_params = model.trainable_parameters();
        assert_eq!(original_params.len(), after_params.len());
        for (before, after) in original_params.iter().zip(after_params.iter()) {
            assert_eq!(dense_vec(before), dense_vec(after));
        }
    }
}
