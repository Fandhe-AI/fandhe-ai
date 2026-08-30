//! Keras `Sequential` 慣習のレイヤー積み上げビルダー（TASK-9.2a・
//! #95。TASK-9.4・#411 で `fandhe_ai_autodiff::compat` から本クレートへ移設）。
//! 数値ロジックは一切持たず、`fandhe_ai_autodiff::nn::Module` 実装
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
//! [`Sequential::apply_parameters`] と組み合わせ、[`crate::optim::Sgd`]・
//! [`crate::optim::AdamW`]（`fandhe_ai::optim`。facade 公開面。イシュー #961）
//! の位置対応契約にそのまま渡せる。`facade` が唯一のサポートされる公開
//! API 面であり（`docs/compat-api-scope.md` §0）、利用者は内部クレート
//! `fandhe_ai_autodiff` へ直接依存する必要はない。適用順序契約
//! （`backward → clip → optimizer step`）の正は [`crate::optim`]
//! モジュール doc とする（イシュー #963）。
//!
//! ```
//! use fandhe_ai::Tensor;
//! use fandhe_ai::compat::Sequential;
//! use fandhe_ai::optim::{Sgd, SgdConfig, clip_grad_norm};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let x = Tensor::new(vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 4])?;
//! let y = Tensor::new(vec![0.0_f32, 1.0, 1.0, 0.0], &[2, 2])?;
//!
//! let mut model = Sequential::new()
//!     .add_linear(4, 8, /* seed = */ 42)?
//!     .add_relu()
//!     .add_linear(8, 2, /* seed = */ 43)?;
//!
//! let mut sgd = Sgd::new(SgdConfig::new(0.01))?;
//!
//! // `SequentialVars`（`bound`）は `&model`／`&tape` を借用するため、
//! // `apply_parameters`（`&mut model`）を呼ぶ前にブロックを抜けて
//! // 借用を解放する。
//! let updated = {
//!     let tape = fandhe_ai::tape();
//!     let bound = model.bind(&tape);
//!     let x_var = tape.var(&x);
//!     let y_var = tape.var(&y);
//!
//!     let pred = bound.forward(&tape, &x_var)?;
//!     let loss = pred.mse_loss(&y_var)?;
//!
//!     let grads = tape.backward(&loss)?;
//!     let grad_refs = bound.trainable_grads(&grads)?;
//!     // 適用順序契約: backward → clip → optimizer step。
//!     let clipped = clip_grad_norm(&grad_refs, /* max_norm = */ 1.0)?;
//!     let clipped_grad_refs: Vec<&Tensor<f32>> = clipped.grads.iter().collect();
//!     let param_refs = model.trainable_parameters();
//!     sgd.step(&param_refs, &clipped_grad_refs)?
//! };
//! model.apply_parameters(updated)?;
//! # Ok(())
//! # }
//! ```
//!
//! **`predict` の既定結線（TASK-9.4・#411）**: `predict` は本クレートの
//! composition root（[`crate::tape`]。既定 CPU・`CpuBackendOps`・融合
//! 有効）で `Tape` を構築して forward する。旧 `fandhe_ai_autodiff::compat` 版が
//! 依存していた naive 参照実装（`fandhe_ai_autodiff::default_ops::naive_ops()`。
//! クレート非公開）は facade から到達できないため、この結線先の変更に
//! 伴い旧 `predict_with_ops`（任意 `BackendOps` 注入経路）は公開面から
//! 撤去した（REQ-12「任意 `BackendOps` 実装を注入できる公開 API を
//! 設けない」・`crates/facade/tests/api_surface.rs` の機械検査と整合。
//! `compat/mod.rs` モジュール doc 参照）。ops を明示的に選びたい内部用途
//! は [`Sequential::forward`]（[`crate::Tape`] を受け取るだけで
//! `BackendOps` は受け取らない）へ、呼び出し元が任意に構築した `Tape` を
//! 渡せば足りる。
//!
//! **公開シグネチャの型（codex-review PR #424 P1 是正）**: `forward`／
//! `bind`／`predict` 等は `fandhe_ai_autodiff::Tape` を直接引数に取らず、本クレート
//! 所有の newtype [`crate::Tape`] を取る（`crate::lib.rs` モジュール doc
//! 「`Tape`（composition root が構築する値）の扱い」参照）。`Var`・
//! `Gradients`・`AutodiffError`・`LinearVars`・`Tensor` は `crate::`
//! 経由（facade の正式な再エクスポート）で参照する。

use crate::{
    AutodiffError, BackendError, DeviceParamStore, Gradients, LinearVars, ResidentLeaf, Tape,
    Tensor, Var,
};
use fandhe_ai_autodiff::nn::activation::{Relu, Sigmoid, Tanh};
use fandhe_ai_autodiff::nn::{Linear, Module};
use fandhe_ai_tensor_core::Activation;

/// Keras `Sequential` 慣習のレイヤー積み上げビルダー。`add_*` はメソッド
/// チェーン（`self` を消費し `Self` を返す）で層を追加し、`predict` で
/// 推論を実行する。層は `nn::Module`（`fandhe_ai_autodiff::nn::module`）実装として
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
    /// （`in_features == 0` を拒否する。`fandhe_ai_autodiff::nn::linear` 参照）ため、
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
    /// `fandhe_ai_autodiff::nn::module` 参照）。外部 `Tape` 上で呼ぶことで
    /// `Tape::backward` までグラフ記録がつながる（推論だけでなく
    /// grad check 等の用途にも使える）。
    pub fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let mut current = *input;
        // イシュー #1044（`docs/kernel-fusion.md` §2.2「学習経路への
        // 結線」）: 単純な `for layer in &self.layers` 逐次委譲だと
        // `Linear` → `Add`（bias）→ `Relu` が別ノード・別カーネル起動に
        // なる（`Module::forward` は多態 dispatch のため層間の関係を
        // 知らない）。ここでインデックス走査へ変え、`Linear` 層に出会う
        // たび「次層が `ReLU` か」（`Module::as_relu`）を先読みして
        // `LinearVars::forward_with_activation` へ結線し、`ReLU` 層自体を
        // スキップする（`ReLU` が続かない末尾 `Linear` 等は bias のみ
        // 融合の `Activation::None`）。それ以外の層（`Sigmoid`／`Tanh`
        // 等。`BackendOps::gemm_bias_act` の `Activation` に対応する
        // variant を持たない）は従来どおり `Module::forward` へ委譲する。
        let mut i = 0;
        while i < self.layers.len() {
            let layer = &self.layers[i];
            if let Some(linear) = layer.as_linear() {
                let act = if self.layers.get(i + 1).is_some_and(|next| next.as_relu()) {
                    Activation::Relu
                } else {
                    Activation::None
                };
                // `nn::Module::forward` と同じく `tape.0`（`pub(crate)`
                // フィールド）経由で内部の生 `Tape` を取り出す（本ファイル
                // 冒頭 doc「公開シグネチャの型」参照）。
                current = linear
                    .bind(&tape.0)
                    .forward_with_activation(&current, act)?;
                i += if matches!(act, Activation::Relu) {
                    2
                } else {
                    1
                };
            } else {
                current = layer.forward(&tape.0, &current)?;
                i += 1;
            }
        }
        Ok(current)
    }

    /// 推論の入口（受け入れ条件「Sequential でのモデル構築・推論が
    /// 動作する」を満たす API）。内部で [`crate::tape`]（本クレートの
    /// composition root。既定 CPU・`CpuBackendOps`・融合有効）で 1
    /// ステップ分の `Tape` を生成し `forward` を呼んだ後 `to_tensor()`
    /// で追跡を外した `Tensor<f32>` を返す（`Tape` はこの呼び出しの
    /// スコープ内で破棄される。`fandhe_ai_autodiff::nn::linear` の「`Tape` は
    /// ステップごとに生成・破棄される前提」と同じ運用）。
    ///
    /// **TASK-9.4（#411）**: 旧 `fandhe_ai_autodiff::compat` 版が持っていた
    /// `predict_with_ops`（任意 `BackendOps` 注入経路）は本移設で公開面
    /// から撤去した（モジュール doc 参照）。
    ///
    /// **イシュー #1028（`docs/inference-forward-fixed-cost-design.md`
    /// §3.1「段階 A」）**: 内部実装は tape 不要経路
    /// （`predict_tape_free`。非公開の内部メソッドのためリンクなし）
    /// を優先し、層構成が `Module::forward_host` 未対応
    /// （`BackendError::Unsupported`）の場合のみ旧経路
    /// （`predict_via_tape`。同じく非公開。`Tape::var` の葉
    /// クローン・ノード記録を伴う）へ fail-closed にフォールバックする。
    /// `docs/compat-api-scope.md` §1 の 3 種（Linear・ReLU/Sigmoid/Tanh）
    /// はいずれも tape 不要経路に対応済みのため（`nn/module.rs` の
    /// 各 `forward_host` オーバーライド）、通常はフォールバックへ
    /// 到達しない。公開シグネチャ・戻り値・数値結果は変更しない
    /// （新旧経路の bit 完全一致は `sequential_predict_tape_free_matches_via_tape_bit_exact`
    /// で検証）。
    pub fn predict(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
        match self.predict_tape_free(input) {
            Err(AutodiffError::Backend(BackendError::Unsupported(_))) => {
                self.predict_via_tape(input)
            }
            other => other,
        }
    }

    /// 旧経路（`Tape` 上で `forward` を実行し `to_tensor()` で実体化する。
    /// TASK-9.4 当初実装）。[`Self::predict`] のフォールバック先。
    fn predict_via_tape(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
        let tape = crate::tape();
        let input_var = tape.var(input);
        let output = self.forward(&tape, &input_var)?;
        Ok(output.to_tensor())
    }

    /// tape 不要経路（イシュー #1028）。`Tape`／`Var` を一切構築せず、
    /// 層ごとに `Module::forward_host` を直接呼ぶ（CPU の
    /// `fandhe_ai_backend_cpu::CpuBackendOps` に固定。[`crate::tape`] が
    /// 常に CPU バックエンドで `Tape` を構築するのと同じ構成であり、
    /// `predict` の既存の「CPU 固定」契約を変えない）。1 層でも
    /// `Unsupported` を返した場合は途中結果を捨てて `Unsupported` を
    /// そのまま伝播し、[`Self::predict`] が旧経路へ全体フォールバック
    /// する（部分的にホストフォールバックしない。§3.2 のフォール
    /// バック契約と同じ「黙示のフォールバックをしない」設計）。
    fn predict_tape_free(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
        let ops = fandhe_ai_backend_cpu::CpuBackendOps::new();
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.forward_host(&ops, &current)?;
        }
        Ok(current)
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
        // `Linear::bind` も `&fandhe_ai_autodiff::Tape` を要求する（`forward` と
        // 同じ理由。`tape.0` 経由で取り出す）。
        let linears = self
            .layers
            .iter()
            .filter_map(|layer| layer.as_linear())
            .map(|linear| linear.bind(&tape.0))
            .collect();
        SequentialVars {
            model: self,
            linears,
        }
    }

    /// 学習可能パラメータ（`Linear` 層の `weight`/`bias`）への参照列を
    /// 層の追加順・各層内は weight → bias（`Some` の場合のみ）の順で
    /// 返す。[`crate::optim::Sgd::step`]／[`crate::optim::AdamW::step`]
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

    /// optimizer（[`crate::optim::Sgd::step`]／[`crate::optim::AdamW::step`]）
    /// が返した更新後テンソル列を
    /// [`Sequential::trainable_parameters`] と同じ順序契約で各 `Linear`
    /// 層へ書き戻す。内部で `Linear::from_parameters`（`fandhe_ai_autodiff::nn::linear`）
    /// により層を再構築するため、compat 層自身は新パラメータ単体の内部
    /// 整合性検証ロジックを重複実装しない（REQ-9「薄いラッパーに徹する」）。
    /// ただし**置換前の層 shape との一致検証**（次段落）は
    /// `Linear::from_parameters` の責務範囲外（層の存在を知らない）なので、
    /// `apply_parameters` 自身の契約として本メソッドが担う（#426）。
    ///
    /// **置換前 shape との完全一致検証（#426。codex-review PR #420 P2 是正）**:
    /// 1 パス目で `Linear::from_parameters` を呼ぶ前に、各層の新
    /// `weight.shape()` が置換前の `weight.shape()` と完全一致することを
    /// 検査する（不一致なら層序数・expected/actual を含む
    /// `AutodiffError::InvalidArgument`）。bias 側の置換前一致は独立検査
    /// せず、`Linear::from_parameters` の bias 検証（`bias.shape() ==
    /// [weight.shape()[1]]`）に委譲する: weight が置換前と完全一致した
    /// 状態でこの bias 検証が通れば、bias の out_features 側も自動的に
    /// 置換前と一致するため（REQ-9「薄いラッパーに徹する」。独立実装は
    /// しない）。
    ///
    /// **層間整合は本検証に内包される**: 層 i の out_features と層 i+1 の
    /// in_features の整合は「置換前モデルが満たしていた水準」を shape 非
    /// 変更の更新が破り得ないことにより、置換前 shape との一致検証へ
    /// 帰着する。したがって隣接層比較を独立実装しない。**この検証により
    /// `apply_parameters` は shape を変えない optimizer 更新専用の契約と
    /// なる**: 層幅を変える意図的な「リサイズ」用途は本メソッドのサポート
    /// 対象外（そのような用途が必要になった場合は別 API として設計する）。
    ///
    /// # エラー
    /// - `updated` の要素数が `trainable_parameters()` の件数と不一致
    ///   → `AutodiffError::InvalidArgument`（fail-closed。位置対応契約が
    ///   崩れたまま一部の層だけ更新して黙って続行しない。`.claude/rules/
    ///   security.md` A03）
    /// - 新しい `weight` の shape が置換前の同一層の `weight` の shape と
    ///   不一致 → `AutodiffError::InvalidArgument`（#426。上記参照）
    /// - 個々のテンソルが `Linear::from_parameters` の内部整合性検証に
    ///   反する（例: bias の shape が対応する weight の out_features と
    ///   食い違う）→ `Linear::from_parameters` 由来の `AutodiffError::Shape`
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
    /// 状態を完全に維持する。#426 の shape 検証も同じ 1 パス目（代入前）に
    /// 置くため、この不変条件は変わらない。
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
        for (layer_index, linear) in linears.iter().enumerate() {
            let has_bias = linear.bias().is_some();
            let new_weight = updated.next().ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "Sequential::apply_parameters: updated has fewer elements than \
                     trainable_parameters() (weight missing)"
                        .to_string(),
                )
            })?;
            // #426: 置換前 shape との完全一致検証（層間整合を内包する設計。
            // メソッド doc 参照）。`Linear::from_parameters` 呼び出し前に
            // 行い、shape が変わる更新を fail-closed で拒否する。
            let expected_shape = linear.weight().shape();
            if new_weight.shape() != expected_shape {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Sequential::apply_parameters: layer {layer_index} weight shape changed \
                     from {expected_shape:?} to {:?} (apply_parameters only supports \
                     shape-preserving updates; #426)",
                    new_weight.shape()
                )));
            }
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

    /// デバイス上パラメータ更新（イシュー #935・
    /// `docs/device-resident-update-design.md`）の入口: `tape` のバック
    /// エンドへ全 `Linear` 層の `weight`／`bias` を [`Sequential::
    /// trainable_parameters`] と同一の順序契約（層順に weight → bias。
    /// `Some` の場合のみ）で 1 回だけアップロードし [`DeviceParamStore`]
    /// を構築する。
    ///
    /// 返る `DeviceParamStore` は `tape` とは独立した寿命を持つ（`Tape`
    /// はステップごとに使い捨てる運用のため。`fandhe_ai_autodiff::tape`
    /// モジュール doc「学習ループでの運用」参照）。以後の学習ステップは
    /// 新しい `tape`（`crate::tape_for` 等で構築）を [`Sequential::
    /// forward_resident`]／[`Tape::step_device_param_store`] へ渡しつつ、
    /// 同じ `DeviceParamStore` インスタンスを使い回す。
    pub fn init_device_param_store(&self, tape: &Tape) -> Result<DeviceParamStore, BackendError> {
        let params = self.trainable_parameters();
        DeviceParamStore::new(&tape.0, &params)
    }

    /// デバイス常駐パラメータでの学習用 forward（イシュー #935・
    /// #1022 でパラメータの毎 step D2H を排除）。
    ///
    /// `store.register_resident_params(&tape.0)` を 1 回呼び（forward 用
    /// 葉ノード登録。#1022 でホストへの download を撤去済み——
    /// `DeviceParamStore::register_resident_params` のドキュメント
    /// 参照）、返る [`ResidentLeaf`] 列から [`Sequential::forward`]
    /// （`SequentialVars::forward`）と同一の層イテレーション（`Linear`
    /// 層は `store.linear_forward`〈デバイス常駐 weight のまま forward〉・
    /// 活性化層は `Module::forward` 多態 dispatch）を辿る。`mem`／`ops`
    /// （`BackendOps`／`MemoryOps`）は一切表面に出さない（REQ-12）。
    ///
    /// 呼び出し元は本メソッドの戻り値（`loss` 計算に使う `Var`）から
    /// `tape.backward_device_param_store(...)`（素の `tape.backward` は
    /// `Op::LinearResident` を含むグラフに対し型付きエラーを返す。
    /// `DeviceParamStore::backward` doc 参照）→
    /// `tape.step_device_param_store(&mut store, &grads, &config)` と
    /// 繋げる（`crates/facade/tests/device_param_store_train.rs` に
    /// 実例がある）。
    ///
    /// **forward 失敗時の pending ロールバック（codex-review PR #954 P2
    /// 是正）**: `register_resident_params` で `store` を pending 状態へ
    /// 遷移させた後、`forward_from_flat_leaves`（層数不一致・バックエンド
    /// 演算エラー等）が失敗した場合、そのまま `Err` を返すと `store` は
    /// pending のまま残り、次回呼び出しの `register_resident_params` が
    /// `BackendError::PendingForwardUnconsumed` で拒否される
    /// （`DeviceParamStore` モジュール doc「状態機械」参照）。呼び出し元は
    /// forward の失敗を理由に `store` を破棄する義務を負わないため、公開
    /// ラッパーである本メソッド内で `store.abandon_pending_forward()` に
    /// より pending をロールバックしてから元のエラーを返す
    /// （`register_resident_params` 自体が失敗した場合は pending 状態へ
    /// 遷移していないため、`?` でそのまま伝播してよい）。
    pub fn forward_resident<'t>(
        &self,
        tape: &'t Tape,
        input: &Var<'t>,
        store: &mut DeviceParamStore,
    ) -> Result<Var<'t>, AutodiffError> {
        let leaves = store.register_resident_params(&tape.0)?;
        match self.forward_from_flat_leaves(&tape.0, input, &leaves, store) {
            Ok(output) => Ok(output),
            Err(e) => {
                store.abandon_pending_forward();
                Err(e)
            }
        }
    }

    /// デバイス常駐パラメータでの推論（イシュー #935・#1022 でパラメータの
    /// download を撤去）。`store.device()` へ結線した新規 [`Tape`]
    /// （`crate::tape_for`）を内部で構築し、`store.
    /// snapshot_resident_params`（読み取り専用版。`pending` 状態を
    /// 変化させない。#1022 で download を撤去済み）で得た [`ResidentLeaf`]
    /// 列から forward する。`Tape` はこの呼び出しのスコープ内で破棄される
    /// （`Sequential::predict` と同じ運用。`fandhe_ai_autodiff::nn::linear`
    /// 「`Tape` はステップごとに生成・破棄される前提」参照）。既存
    /// `ActivationKind`／`BackendOps::sigmoid` 等を新設せず
    /// `forward_from_flat_leaves`（`forward_resident` と共有する同一の
    /// 層イテレーション。private ヘルパーのためインドキュメントリンクは
    /// 張らずコードスパン表記とする）を再利用することで、`predict` との
    /// parity を構造的に保証する（設計文書 §3.3c）。
    pub fn predict_resident(
        &self,
        store: &DeviceParamStore,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let tape = crate::tape_for(store.device())?;
        let leaves = store.snapshot_resident_params(&tape.0)?;
        let input_var = tape.var(input);
        let output = self.forward_from_flat_leaves(&tape.0, &input_var, &leaves, store)?;
        Ok(output.to_tensor())
    }

    /// `tape`（`self.layers` を辿る forward 用）と `leaves`（`self.layers`
    /// のうち `Linear` 層に対応する、層順に weight → bias〈`Some` の
    /// 場合のみ〉のフラットな [`ResidentLeaf`] 列。[`Sequential::
    /// trainable_parameters`] と同一の順序契約）・`store`（`leaves` の
    /// 発行元であり `linear_forward` の実行主体）から forward する共通
    /// ロジック。[`Sequential::forward_resident`]／[`Sequential::
    /// predict_resident`] の両方から呼ばれる（層イテレーションの二重実装
    /// を避ける。設計文書 §3.3c）。
    ///
    /// **#1022 による変更**: 旧実装は `Var` 列から [`LinearVars`] を
    /// 組み立てて `LinearVars::forward`（ホスト常駐 weight 前提の
    /// `matmul`／`add`）を呼んでいた。`register_resident_params`／
    /// `snapshot_resident_params` の戻り型が不透明な `ResidentLeaf`
    /// （ホスト値を持たない）へ変わったため、`Linear` 層は
    /// `store.linear_forward`（デバイス常駐のまま `BackendOps::
    /// gemm_resident_rhs` へ委譲）を呼ぶ形へ置き換える。
    fn forward_from_flat_leaves<'t>(
        &self,
        tape: &'t fandhe_ai_autodiff::Tape,
        input: &Var<'t>,
        leaves: &[ResidentLeaf<'t>],
        store: &DeviceParamStore,
    ) -> Result<Var<'t>, AutodiffError> {
        let mut current = *input;
        let mut cursor = leaves.iter();
        // `SequentialVars::forward`（`Sequential::forward` 経由）と同じ
        // 「次層が `ReLU` かを先読みして `Linear` 層へ結線する」方式
        // （イシュー #1044）。デバイス常駐版は `store.
        // linear_forward_with_activation`（`BackendOps::
        // gemm_resident_rhs_act` を経由）を使う。
        let mut i = 0;
        while i < self.layers.len() {
            let layer = &self.layers[i];
            if let Some(linear) = layer.as_linear() {
                let weight = cursor.next().ok_or_else(|| {
                    AutodiffError::InvalidArgument(
                        "Sequential::forward_from_flat_leaves: leaves has fewer elements than \
                         trainable_parameters() (weight missing)"
                            .to_string(),
                    )
                })?;
                let bias = if linear.bias().is_some() {
                    Some(cursor.next().ok_or_else(|| {
                        AutodiffError::InvalidArgument(
                            "Sequential::forward_from_flat_leaves: leaves has fewer elements \
                             than trainable_parameters() (bias missing)"
                                .to_string(),
                        )
                    })?)
                } else {
                    None
                };
                let act = if self.layers.get(i + 1).is_some_and(|next| next.as_relu()) {
                    Activation::Relu
                } else {
                    Activation::None
                };
                current = store
                    .linear_forward_with_activation(tape, &current, weight, bias, act)
                    .map_err(AutodiffError::Backend)?;
                i += if matches!(act, Activation::Relu) {
                    2
                } else {
                    1
                };
            } else {
                current = layer.forward(tape, &current)?;
                i += 1;
            }
        }
        // `leaves` が `self.layers` の要求件数より多い（より大きなモデルの
        // `DeviceParamStore` を誤って渡した等）場合、余剰要素を無視した
        // まま黙って forward が成功すると、位置対応契約（層順に
        // weight → bias）が崩れているにもかかわらず誤った推論結果を
        // 返してしまう（Review 指摘）。`cursor` を消費し切ったことを
        // ここで検査し、余剰があれば fail-closed に拒否する
        // （`.claude/rules/security.md` A03）。
        if cursor.next().is_some() {
            return Err(AutodiffError::InvalidArgument(
                "Sequential::forward_from_flat_leaves: leaves has more elements than \
                 trainable_parameters() requires (extra weight/bias ignored)"
                    .to_string(),
            ));
        }
        Ok(current)
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
    /// （`fandhe_ai_autodiff::nn::module`）の `Linear` 実装は呼び出しのたびに
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
        // イシュー #1044（`docs/kernel-fusion.md` §2.2「学習経路への
        // 結線」）: `Sequential::forward`（推論・非学習経路）と同じ
        // 「次層が `ReLU` かを先読みして `Linear` 層へ結線する」方式。
        // `self.linears` の消費は従来どおりイテレータ（`linears.next()`）
        // に任せる（`Linear` 層に出会うたびに 1 件ずつ。ReLU をスキップ
        // しても `linears` 自体には ReLU 分の要素がないため対応関係は
        // 崩れない）。インデックス走査へ変えたのは `self.model.layers`
        // 側で「次層」を覗く必要があるため。
        let mut linears = self.linears.iter();
        let layers = &self.model.layers;
        let mut i = 0;
        while i < layers.len() {
            let layer = &layers[i];
            if layer.as_linear().is_some() {
                let vars = linears.next().ok_or_else(|| {
                    AutodiffError::InvalidArgument(
                        "SequentialVars::forward: bind 済み LinearVars が model.layers の \
                         Linear 層数より少ない（bind/forward 間の Linear 層数対応が崩れた）"
                            .to_string(),
                    )
                })?;
                let act = if layers.get(i + 1).is_some_and(|next| next.as_relu()) {
                    Activation::Relu
                } else {
                    Activation::None
                };
                current = vars.forward_with_activation(&current, act)?;
                i += if matches!(act, Activation::Relu) {
                    2
                } else {
                    1
                };
            } else {
                // 活性化層は `nn::Module::forward` へ委譲する（`&fandhe_ai_autodiff::Tape`
                // が必要。`Sequential::forward` と同じ理由で `tape.0` 経由）。
                current = layer.forward(&tape.0, &current)?;
                i += 1;
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
    /// [`crate::optim::Sgd::step`]／[`crate::optim::AdamW::step`] の
    /// 位置対応契約（`params[i]` ↔ `grads[i]`）
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

    /// テスト専用の連続化ヘルパー（`fandhe_ai_tensor_core::Tensor` の `pub` API
    /// のみを使用。旧 `fandhe_ai_autodiff::eval::dense_vec`〈クレート非公開〉の
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
        // `Linear::bind` は `&fandhe_ai_autodiff::Tape` を要求するため `manual_tape.0`
        // （同一クレート内なので `pub(crate)` フィールドへアクセス可能）
        // 経由で内部の生 Tape を渡す。
        let manual_tape = crate::tape();
        let manual_input = manual_tape.var(&input_tensor);
        let h = linear1.bind(&manual_tape.0).forward(&manual_input).unwrap();
        let h = h.relu();
        let manual_output = linear2.bind(&manual_tape.0).forward(&h).unwrap();

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
        // （1/sqrt(in_features) が非有限になるため。fandhe_ai_autodiff::nn::linear 参照）。
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
    // イシュー #1028: 推論 forward の固定費削減（tape 不要経路）
    // =================================================================

    /// [`Sequential::predict`] の新経路（tape 不要。`predict_tape_free`）が
    /// 旧経路（`predict_via_tape`）と bit 完全一致することを、
    /// Linear・ReLU・Sigmoid・Tanh を混在させた層構成・複数バッチ・
    /// bias 有無混在で確認する（`docs/inference-forward-fixed-cost-
    /// design.md` §3.3 (b) の bit-exactness 契約）。
    #[test]
    fn sequential_predict_tape_free_matches_via_tape_bit_exact() {
        for batch in [1usize, 3, 8] {
            let model = Sequential {
                layers: vec![
                    Box::new(Linear::new(8, 16, true, SEED1).unwrap()),
                    Box::new(Relu),
                    Box::new(Linear::new(16, 12, false, SEED2).unwrap()),
                    Box::new(Sigmoid),
                    Box::new(Linear::new(12, 4, true, SEED1 + SEED2).unwrap()),
                    Box::new(Tanh),
                ],
            };
            let input_data: Vec<f32> = (0..batch * 8).map(|i| (i as f32) * 0.05 - 0.3).collect();
            let input = Tensor::new(input_data, &[batch, 8]).unwrap();

            let via_tape_fast = model.predict(&input).unwrap();
            let via_tape_slow = model.predict_via_tape(&input).unwrap();

            assert_eq!(
                via_tape_fast.shape(),
                via_tape_slow.shape(),
                "batch={batch}"
            );
            assert_eq!(
                dense_vec(&via_tape_fast),
                dense_vec(&via_tape_slow),
                "predict（tape 不要経路）は predict_via_tape（旧経路）と \
                 bit 完全一致するはず（batch={batch}）"
            );
        }
    }

    /// [`Sequential::predict`] は `add_linear`（bias あり既定）で構築した
    /// 通常の `Sequential::new()` 経由モデルでも新旧経路が一致すること
    /// （`layers` を直接構築するテスト専用経路だけでなく、公開 API の
    /// 通常経路でも成立することの確認）。
    #[test]
    fn sequential_predict_public_builder_tape_free_matches_via_tape() {
        let model = Sequential::new()
            .add_linear(4, 6, SEED1)
            .unwrap()
            .add_relu()
            .add_linear(6, 2, SEED2)
            .unwrap();
        let input = Tensor::new(vec![0.1_f32, -0.2, 0.3, -0.4], &[1, 4]).unwrap();

        let fast = model.predict(&input).unwrap();
        let slow = model.predict_via_tape(&input).unwrap();

        assert_eq!(dense_vec(&fast), dense_vec(&slow));
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
        // `Linear::from_parameters` の bias 検証（`fandhe_ai_autodiff::nn::linear`）
        // がこれを拒否することを確認する（`apply_parameters` は新パラメータ
        // 単体の内部整合性検証を重複実装せず委譲する、という設計の裏付け。
        // 置換前 shape との一致検証〈#426〉は本テストの weight 側では
        // 通過するため、ここで拒否されるのは委譲先の bias 検証由来）。
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

    // =================================================================
    // #426: apply_parameters の置換前 shape 一致検証
    // =================================================================

    #[test]
    fn apply_parameters_rejects_weight_shape_change() {
        // 新 weight（[3, 3]）は `Linear::from_parameters` の単体検証
        // （rank 2・in_features > 0・bias 整合）は通過するが、置換前の
        // weight shape（[4, 3]）とは一致しない。#426 の検証がこれを
        // 拒否し、モデルは不変のまま残ることを確認する。
        let mut model = Sequential::new().add_linear(4, 3, SEED1).unwrap();
        let original_params: Vec<Tensor<f32>> =
            model.trainable_parameters().into_iter().cloned().collect();

        let new_weight = Tensor::new(vec![0.0f32; 9], &[3, 3]).unwrap();
        let new_bias = Tensor::new(vec![0.0f32; 3], &[3]).unwrap();
        let err = model
            .apply_parameters(vec![new_weight, new_bias])
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        let after_params = model.trainable_parameters();
        assert_eq!(original_params.len(), after_params.len());
        for (before, after) in original_params.iter().zip(after_params.iter()) {
            assert_eq!(dense_vec(before), dense_vec(after));
        }
    }

    #[test]
    fn apply_parameters_rejects_interlayer_consistent_resize() {
        // 層間整合（層 1 の out_features == 層 2 の in_features）は
        // 保たれているが、置換前の層幅（2→3→1）自体を変える更新
        // （2→4→1）を渡すケース。#426 の設計判断
        // （「層間整合は置換前 shape 一致に内包される。リサイズは非
        // サポート」）どおり拒否されることを固定する。
        let mut model = Sequential::new()
            .add_linear(2, 3, SEED1)
            .unwrap()
            .add_relu()
            .add_linear(3, 1, SEED2)
            .unwrap();

        let l1_weight = Tensor::new(vec![0.0f32; 8], &[2, 4]).unwrap();
        let l1_bias = Tensor::new(vec![0.0f32; 4], &[4]).unwrap();
        let l2_weight = Tensor::new(vec![0.0f32; 4], &[4, 1]).unwrap();
        let l2_bias = Tensor::new(vec![0.0f32; 1], &[1]).unwrap();

        let err = model
            .apply_parameters(vec![l1_weight, l1_bias, l2_weight, l2_bias])
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn apply_parameters_leaves_model_unchanged_on_mid_sequence_weight_shape_error() {
        // #426 の two-pass 非退行確認: 2 層目の weight shape のみ不一致に
        // した場合でも、1 層目（先に走査される層）を含む全パラメータが
        // エラー前の値のまま保持されることを確認する
        // （既存 `..._on_mid_sequence_shape_error` と同型。#294 の
        // 設計不変条件が #426 の検証にも及ぶことの回帰テスト）。
        let mut model = Sequential::new()
            .add_linear(2, 3, SEED1)
            .unwrap()
            .add_relu()
            .add_linear(3, 1, SEED2)
            .unwrap();

        let original_params: Vec<Tensor<f32>> =
            model.trainable_parameters().into_iter().cloned().collect();

        // 1 層目（weight, bias）は妥当な shape、2 層目の weight のみ
        // 置換前 shape（[3, 1]）と食い違う shape（[3, 2]）にする。
        let l1_weight = Tensor::new(vec![9.0f32; 6], &[2, 3]).unwrap();
        let l1_bias = Tensor::new(vec![9.0f32; 3], &[3]).unwrap();
        let l2_weight = Tensor::new(vec![9.0f32; 6], &[3, 2]).unwrap();
        let l2_bias = Tensor::new(vec![9.0f32; 2], &[2]).unwrap();

        let err = model
            .apply_parameters(vec![l1_weight, l1_bias, l2_weight, l2_bias])
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        let after_params = model.trainable_parameters();
        assert_eq!(original_params.len(), after_params.len());
        for (before, after) in original_params.iter().zip(after_params.iter()) {
            assert_eq!(dense_vec(before), dense_vec(after));
        }
    }

    /// イシュー #1044「Linear の epilogue 融合（bias + ReLU）が学習
    /// forward / backward 経路で適用されることを検証・修正する」の
    /// 機械検証本体。`fandhe_ai_backend_cpu::CpuBackendOps` を包み、
    /// `BackendOps` の各メソッド呼び出し回数を `AtomicUsize` で数える
    /// （`gemm_bias_act`／`gemm_resident_rhs_act` が実際に呼ばれ、
    /// 非融合合成〈`gemm`／`add`〉や別ノードの `relu` が呼ばれていない
    /// ことを検証する。値そのものは `CpuBackendOps` へ委譲するため
    /// 数値は変わらない）。`Tape::new_with_ops`（`pub(crate)`）を直接
    /// 呼べるのはこのクレート内（本ファイル）に限るため、この
    /// カウンタ検証は facade の統合テスト（`tests/*.rs`。別クレート
    /// コンパイル単位）ではなくここに置く。
    /// `CountingOps` の呼び出し回数カウンタ。`CountingOps` 自体は
    /// `Box<dyn BackendOps>` として `Tape` の所有物になり `Tape::ops()`
    /// は `&dyn BackendOps`（ダウンキャスト不能。`BackendOps` は
    /// `std::any::Any` を要求しない）しか返さないため、`Tape` 構築前に
    /// `Arc<AtomicUsize>` の複製をこちら側に残しておき、forward 後は
    /// この複製から読む。
    #[derive(Clone, Default)]
    struct Counters {
        gemm: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        add: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        relu: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        gemm_bias_act: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        run_fused: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        gemm_resident_rhs: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        gemm_resident_rhs_act: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Counters {
        fn get(counter: &std::sync::atomic::AtomicUsize) -> usize {
            counter.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    struct CountingOps {
        inner: fandhe_ai_backend_cpu::CpuBackendOps,
        counters: Counters,
    }

    impl CountingOps {
        fn new(counters: Counters) -> Self {
            Self {
                inner: fandhe_ai_backend_cpu::CpuBackendOps::new(),
                counters,
            }
        }
    }

    impl fandhe_ai_tensor_core::BackendOps for CountingOps {
        fn device(&self) -> fandhe_ai_tensor_core::device::Device {
            self.inner.device()
        }

        fn gemm(
            &self,
            a: &Tensor<f32>,
            b: &Tensor<f32>,
        ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
            self.counters
                .gemm
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.gemm(a, b)
        }

        fn add(
            &self,
            a: &Tensor<f32>,
            b: &Tensor<f32>,
        ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
            self.counters
                .add
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.add(a, b)
        }

        fn mul(
            &self,
            a: &Tensor<f32>,
            b: &Tensor<f32>,
        ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
            self.inner.mul(a, b)
        }

        fn relu(
            &self,
            a: &Tensor<f32>,
        ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
            self.counters
                .relu
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.relu(a)
        }

        fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
            self.inner.exp(a)
        }

        fn tanh(
            &self,
            a: &Tensor<f32>,
        ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
            self.inner.tanh(a)
        }

        fn sum(
            &self,
            a: &Tensor<f32>,
            dim: Option<usize>,
        ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
            self.inner.sum(a, dim)
        }

        fn max(
            &self,
            a: &Tensor<f32>,
            dim: Option<usize>,
        ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
            self.inner.max(a, dim)
        }

        fn gemm_bias_act(
            &self,
            a: &Tensor<f32>,
            b: &Tensor<f32>,
            bias: Option<&Tensor<f32>>,
            act: Activation,
        ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
            self.counters
                .gemm_bias_act
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.gemm_bias_act(a, b, bias, act)
        }

        fn gemm_resident_rhs(
            &self,
            a: &Tensor<f32>,
            w: fandhe_ai_tensor_core::DeviceBufferView<'_>,
            bias: Option<fandhe_ai_tensor_core::DeviceBufferView<'_>>,
        ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
            self.counters
                .gemm_resident_rhs
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.gemm_resident_rhs(a, w, bias)
        }

        fn gemm_resident_rhs_act(
            &self,
            a: &Tensor<f32>,
            w: fandhe_ai_tensor_core::DeviceBufferView<'_>,
            bias: Option<fandhe_ai_tensor_core::DeviceBufferView<'_>>,
            act: Activation,
        ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
            self.counters
                .gemm_resident_rhs_act
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.gemm_resident_rhs_act(a, w, bias, act)
        }

        fn run_fused(
            &self,
            plan: &fandhe_ai_tensor_core::FusionPlan,
            leaves: &[&Tensor<f32>],
        ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::BackendError> {
            self.counters
                .run_fused
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.run_fused(plan, leaves)
        }
    }

    #[test]
    fn sequential_forward_fuses_linear_relu_into_single_gemm_bias_act_launch() {
        // 受け入れ条件 1（イシュー #1044）: Linear -> ReLU -> Linear の
        // 層あたりカーネル起動数が 1（`gemm_bias_act`）になり、非融合
        // 合成（`gemm`／`add`）や別ノードの `relu` が発生しないことを
        // 機械検証する。
        let model = Sequential::new()
            .add_linear(4, 8, SEED1)
            .unwrap()
            .add_relu()
            .add_linear(8, 2, SEED2)
            .unwrap();

        // `counters`（`Arc` 共有）は `CountingOps` を `Box<dyn BackendOps>`
        // として `Tape` へ move した後もこちら側から読める（`Tape::ops()`
        // は `&dyn BackendOps` しか返さずダウンキャスト手段を持たない
        // ため、move 前に複製を残しておく設計）。
        let counters = Counters::default();
        let ops = CountingOps::new(counters.clone());
        let tape = Tape(fandhe_ai_autodiff::Tape::new_with_ops(Box::new(ops)));
        let input = Tensor::new(vec![0.25_f32; 3 * 4], &[3, 4]).unwrap();
        let input_var = tape.var(&input);

        let vars = model.bind(&tape);
        vars.forward(&tape, &input_var).unwrap();

        assert_eq!(
            Counters::get(&counters.gemm_bias_act),
            2,
            "Linear 層 2 個（1 個目は ReLU 融合、2 個目は bias のみ融合）でそれぞれ \
             gemm_bias_act が 1 回ずつ呼ばれるはず"
        );
        assert_eq!(
            Counters::get(&counters.gemm),
            0,
            "非融合 gemm 合成は発生しないはず"
        );
        assert_eq!(
            Counters::get(&counters.add),
            0,
            "非融合 add 合成は発生しないはず"
        );
        assert_eq!(
            Counters::get(&counters.relu),
            0,
            "ReLU は Linear へ融合され、別ノードとしての relu 呼び出しは発生しないはず"
        );
        assert_eq!(
            Counters::get(&counters.run_fused),
            0,
            "本経路は run_fused（elementwise 融合）を経由しないはず"
        );
    }

    /// イシュー #1044 の backward 側検証: `Op::LinearAct` の VJP
    /// （`grad.rs`）が非融合合成（`Op::MatMul` → `Op::Add` → `Op::Relu`）
    /// の VJP と同じ勾配をパラメータへ返すことを、`Tape::backward` を
    /// 通した実測でビット一致検証する（`out_value` からの ReLU マスク
    /// 復元が既存 `Op::Relu` の劣勾配規約〈`x > 0`〉と一致することの
    /// 統合確認。単体の規約一致自体は `grad.rs` 側で `out_value > 0` と
    /// `Op::Relu` の `v > 0.0` が同値であることをコード上の理由として
    /// 既に記録している）。
    #[test]
    fn sequential_vars_forward_with_activation_grad_matches_manual_composition() {
        let linear1 = Linear::new(4, 8, true, SEED1).unwrap();
        let linear2 = Linear::new(8, 2, true, SEED2).unwrap();

        let batch = 3;
        let input_data: Vec<f32> = (0..batch * 4).map(|i| i as f32 * 0.07 - 0.5).collect();
        let input_tensor = Tensor::new(input_data, &[batch, 4]).unwrap();

        // 融合経路: `LinearVars::forward_with_activation` を直接呼ぶ
        // （`Sequential::bind().forward()` が内部で辿るのと同じ 1 ノード
        // 経路）。
        let fused_tape = crate::tape();
        let fused_input = fused_tape.var(&input_tensor);
        let fused_vars1 = linear1.bind(&fused_tape.0);
        let fused_vars2 = linear2.bind(&fused_tape.0);
        let h = fused_vars1
            .forward_with_activation(&fused_input, Activation::Relu)
            .unwrap();
        let out = fused_vars2
            .forward_with_activation(&h, Activation::None)
            .unwrap();
        let loss = out.sum(None).unwrap();
        let fused_grads = fused_tape.backward(&loss).unwrap();

        // 非融合経路: `matmul` → `add` → `relu` を明示的に合成する
        // （`LinearVars::forward` + `Var::relu`）。
        let manual_tape = crate::tape();
        let manual_input = manual_tape.var(&input_tensor);
        let manual_vars1 = linear1.bind(&manual_tape.0);
        let manual_vars2 = linear2.bind(&manual_tape.0);
        let h = manual_vars1.forward(&manual_input).unwrap();
        let h = h.relu();
        let out = manual_vars2.forward(&h).unwrap();
        let loss = out.sum(None).unwrap();
        let manual_grads = manual_tape.backward(&loss).unwrap();

        for (label, fused_var, manual_var) in [
            ("linear1.weight", &fused_vars1.weight, &manual_vars1.weight),
            ("linear2.weight", &fused_vars2.weight, &manual_vars2.weight),
        ] {
            let fused_g = fused_grads.get(fused_var).unwrap().unwrap();
            let manual_g = manual_grads.get(manual_var).unwrap().unwrap();
            assert_eq!(
                dense_vec(fused_g),
                dense_vec(manual_g),
                "{label}: 融合経路（Op::LinearAct）と非融合合成の勾配がビット一致しない"
            );
        }
        for (label, fused_var, manual_var) in [
            (
                "linear1.bias",
                fused_vars1.bias.as_ref().unwrap(),
                manual_vars1.bias.as_ref().unwrap(),
            ),
            (
                "linear2.bias",
                fused_vars2.bias.as_ref().unwrap(),
                manual_vars2.bias.as_ref().unwrap(),
            ),
        ] {
            let fused_g = fused_grads.get(fused_var).unwrap().unwrap();
            let manual_g = manual_grads.get(manual_var).unwrap().unwrap();
            assert_eq!(
                dense_vec(fused_g),
                dense_vec(manual_g),
                "{label}: 融合経路（Op::LinearAct）と非融合合成の勾配がビット一致しない"
            );
        }
    }
}
