//! Functional モデルの学習（`bind`・パラメータ収集・`compile`・`fit`・`evaluate`。イシュー #2667・
//! 親 #2663・ルート #2499 Phase 4）。
//!
//! 役割: `compat::functional`（グラフ構築と forward。#2665・#2666）に、`compat::Sequential` の
//! Keras 風学習（`compile`／`fit`／`evaluate`）と同じ演算列を多入力・多出力グラフへ載せる。設計の正は
//! `docs/facade-functional-api-decision.md` §7（学習）・§18（#2667 実装記録）。公開形は**未承認**
//! （承認依頼は #2677。公開は承認後の #2679）のため、`functional.rs` と同じく `#[cfg(test)]` 隔離の
//! `pub(crate)` に留め、facade の公開面へは出さない。
//!
//! # 再利用と新設
//!
//! - optimizer 状態・loss・History は `compat::training` の既存型（`OptimizerState`・`Compiled`・
//!   `Optimizer`・`Loss`・`FitConfig`・`History`・`FitTarget`）を再利用する。**optimizer 状態の型を別に
//!   新設しない**（`training.rs` の可視性を crate 内部だけ `compat` へ広げた。公開面は不変）。
//! - 1 バッチの演算列は `Sequential::run_fit` の非 AMP・`accumulate_steps == 1` の分岐と同一
//!   （`crate::tape()` → `bind` → forward → `T::loss_for` → `to_tensor().get(&[])` → `backward` →
//!   `trainable_grads` → `optimizer.step` → `apply_parameters`）。集計も同じ（`f64` でサンプル数重み付き
//!   合計し epoch 末に 1 回 `f32` へ）。単一入力・単一ブロック・単一出力のグラフでは `Sequential::fit` と
//!   bit 一致する（回帰テストで固定。`fit_tests.rs`）。
//! - 多出力の損失は各出力へ `T::loss_for` を適用し、出力の指定順に `Var::add` で左畳み込む
//!   （Keras の単一 loss 指定時の既定。出力 1 件なら加算しない）。
//!
//! # 第 1 段で受け付けないもの（fail-closed）
//!
//! `Optimizer::Lbfgs`（closure 駆動で `&mut` モデルへの trial 書き込みと失敗時復元の別経路が要る）・
//! `accumulate_steps != 1`・AMP・callbacks・validation・metrics・`train_step` フック・常駐経路は入口を
//! 設けない。`compile` が `Lbfgs` を、`fit` が `accumulate_steps != 1` を `InvalidArgument` で拒否する
//! （後から許可するのは非破壊・逆は破壊的なため安全側）。
//!
//! # エラー時の不変条件
//!
//! `fit`／`evaluate` は引数検査（未 compile・入力／目標件数・`epochs`・`accumulate_steps`・
//! `add_module` のパラメータ持ち層・`batch_first=false` の MHA）をモード変更・データ構築より前に行い、
//! どの経路でも `compiled` を書き戻し、モードを呼び出し前へ復元する（`training()` は導出値のため
//! ブロック間でモードが混在していた場合は一方へ揃う）。呼び出し元: `fit_tests.rs`・
//! `model_io::functional_io` の往復テスト。

use fandhe_ai_tensor_core::Element;
use fandhe_ai_tensor_core::data::{
    DataError, DataLoader, DataLoaderConfig, Dataset, TensorDataset,
};

use super::{FunctionalModel, FunctionalVars, NodeDef, invalid};
use crate::compat::training::{
    Compiled, FitConfig, FitTarget, History, Loss, Optimizer, OptimizerState,
};
use crate::{AutodiffError, Gradients, Tape, Tensor, Var};

/// 未 compile のモデルへ `fit`／`evaluate` を呼んだ場合の共通エラー（`Sequential` 版は
/// `Sequential::` を名乗るため Functional 用に別に持つ）。
fn not_compiled(method: &str) -> AutodiffError {
    invalid(format!(
        "FunctionalModel::{method}: compile() が呼ばれていない（optimizer／loss が未設定）"
    ))
}

/// n 入力・m 目標を同じ添字列で切り出すデータセット（`(TensorDataset, TensorDataset)` の n 入力 m 目標版）。
///
/// `len()` は先頭入力の長さ、`validate` が全成分の長さ一致を `DataError::LengthMismatch` で検査し、
/// `batch` は同じ添字列を全成分へ適用する。`DataLoader` のシャッフル RNG の消費は成分数に依らないため、
/// 単一入力・単一目標では `Sequential::fit` が使うタプル版と同じ順序のバッチを返す。
struct GraphDataset<T: Element> {
    xs: Vec<TensorDataset<f32>>,
    ys: Vec<TensorDataset<T>>,
}

impl<T: Element> Dataset for GraphDataset<T> {
    type Batch = (Vec<Tensor<f32>>, Vec<Tensor<T>>);

    fn len(&self) -> usize {
        self.xs.first().map_or(0, Dataset::len)
    }

    fn validate(&self) -> Result<(), DataError> {
        for d in &self.xs {
            d.validate()?;
        }
        for d in &self.ys {
            d.validate()?;
        }
        let expected = self.len();
        let lens = self
            .xs
            .iter()
            .map(Dataset::len)
            .chain(self.ys.iter().map(Dataset::len));
        for (component, found) in lens.enumerate() {
            if found != expected {
                return Err(DataError::LengthMismatch {
                    expected,
                    found,
                    component,
                });
            }
        }
        Ok(())
    }

    fn batch(&self, indices: &[usize]) -> Result<Self::Batch, DataError> {
        let mut xs = Vec::new();
        xs.try_reserve_exact(self.xs.len()).map_err(|_| {
            DataError::Shape(fandhe_ai_tensor_core::ShapeError::ElementCountOverflow)
        })?;
        for d in &self.xs {
            xs.push(d.batch(indices)?);
        }
        let mut ys = Vec::new();
        ys.try_reserve_exact(self.ys.len()).map_err(|_| {
            DataError::Shape(fandhe_ai_tensor_core::ShapeError::ElementCountOverflow)
        })?;
        for d in &self.ys {
            ys.push(d.batch(indices)?);
        }
        Ok((xs, ys))
    }
}

/// `xs`／`ys` から [`GraphDataset`] の `DataLoader` を作る。成分ごとの `TensorDataset::new` が rank 0 を、
/// `DataLoader::new` が成分間の長さ不一致・`batch_size == 0` を拒否する（いずれも `InvalidArgument`）。
fn build_loader<T: FitTarget>(
    method: &str,
    xs: &[&Tensor<f32>],
    ys: &[&Tensor<T>],
    config: DataLoaderConfig,
) -> Result<DataLoader<GraphDataset<T>>, AutodiffError> {
    let to_invalid = |e: DataError| invalid(format!("FunctionalModel::{method}: {e}"));
    let mut x_sets = Vec::new();
    x_sets
        .try_reserve_exact(xs.len())
        .map_err(|_| super::super::alloc_failed())?;
    for x in xs {
        x_sets.push(TensorDataset::new((*x).clone()).map_err(to_invalid)?);
    }
    let mut y_sets = Vec::new();
    y_sets
        .try_reserve_exact(ys.len())
        .map_err(|_| super::super::alloc_failed())?;
    for y in ys {
        y_sets.push(TensorDataset::new((*y).clone()).map_err(to_invalid)?);
    }
    DataLoader::new(
        GraphDataset {
            xs: x_sets,
            ys: y_sets,
        },
        config,
    )
    .map_err(to_invalid)
}

/// 各出力へ `T::loss_for` を適用し、出力の指定順に `Var::add` で左畳み込んだ損失を返す
/// （出力 1 件なら加算なし。`Sequential::fit` と演算列が一致する）。
fn combined_loss<'t, T: FitTarget>(
    loss: Loss,
    tape: &'t Tape,
    preds: &[Var<'t>],
    targets: &[Tensor<T>],
) -> Result<Var<'t>, AutodiffError> {
    let mut total: Option<Var<'t>> = None;
    for (pred, target) in preds.iter().zip(targets) {
        let l = T::loss_for(loss, tape, pred, target)?;
        total = Some(match total {
            None => l,
            Some(acc) => acc.add(&l)?,
        });
    }
    total.ok_or_else(|| invalid("FunctionalModel: 出力が 0 件で損失を作れない".to_string()))
}

impl FunctionalModel {
    /// 全ブロックを挿入順に `Sequential::bind` し、1 学習ステップ分のハンドルを返す
    /// （同一 tape 上の複数 `bind`）。
    pub(crate) fn bind<'m, 't>(&'m self, tape: &'t Tape) -> FunctionalVars<'m, 't> {
        FunctionalVars {
            model: self,
            blocks: self.blocks().map(|(i, b)| (i, b.bind(tape))).collect(),
        }
    }

    /// 訓練対象パラメータ。ブロックの挿入順・各ブロック内は `Sequential::trainable_parameters` と
    /// 同じ順（`named_parameters` の通し番号順と一致。結合ノードは寄与しない）。
    pub(crate) fn trainable_parameters(&self) -> Vec<&Tensor<f32>> {
        self.blocks()
            .flat_map(|(_, b)| b.trainable_parameters())
            .collect()
    }

    /// [`Self::trainable_parameters`] と同じ並びの更新後パラメータを各ブロックへ書き戻す。
    ///
    /// 総数・各 shape の不一致は**何も変更せず** `InvalidArgument`。検査後の適用でブロックが失敗した
    /// 場合（独自層の `set_parameter` 失敗等）は適用済みブロックと失敗ブロックを適用前の値へ巻き戻し、
    /// 元のエラーを返す。巻き戻しも失敗した場合は部分適用の可能性を明示した `InvalidArgument` を返す
    /// （`load_state_dict` と同じ契約）。
    pub(crate) fn apply_parameters(
        &mut self,
        updated: Vec<Tensor<f32>>,
    ) -> Result<(), AutodiffError> {
        let mut counts: Vec<(usize, usize)> = Vec::new();
        counts
            .try_reserve_exact(self.nodes.len())
            .map_err(|_| super::super::alloc_failed())?;
        for (index, block) in self.blocks() {
            counts.push((index, block.trainable_parameters().len()));
        }
        let total: usize = counts.iter().map(|(_, n)| *n).sum();
        if updated.len() != total {
            return Err(invalid(format!(
                "FunctionalModel::apply_parameters: 更新後パラメータ数 {} が訓練対象パラメータ数 {total} と一致しない",
                updated.len()
            )));
        }
        let current = self.trainable_parameters();
        for (slot, (cur, new)) in current.iter().zip(&updated).enumerate() {
            if cur.shape() != new.shape() {
                return Err(invalid(format!(
                    "FunctionalModel::apply_parameters: パラメータ {slot} の shape {:?} が現在の {:?} と一致しない",
                    new.shape(),
                    cur.shape()
                )));
            }
        }
        let mut previous: Vec<Vec<Tensor<f32>>> = Vec::new();
        previous
            .try_reserve_exact(counts.len())
            .map_err(|_| super::super::alloc_failed())?;
        for (_, block) in self.blocks() {
            previous.push(block.trainable_parameters().into_iter().cloned().collect());
        }
        let mut rest = updated.into_iter();
        for (applied, (index, n)) in counts.iter().enumerate() {
            let chunk: Vec<Tensor<f32>> = rest.by_ref().take(*n).collect();
            let result = match self.nodes.get_mut(*index) {
                Some(NodeDef::Block { block, .. }) => block.apply_parameters(chunk),
                _ => Err(invalid(format!(
                    "内部不整合: ノード {index} がブロックでない"
                ))),
            };
            if let Err(err) = result {
                let mut failures: Vec<String> = Vec::new();
                for ((done, _), snapshot) in counts.iter().zip(previous).take(applied + 1) {
                    let restored = match self.nodes.get_mut(*done) {
                        Some(NodeDef::Block { block, .. }) => block.apply_parameters(snapshot),
                        _ => Err(invalid("ブロックでないため復元不能".to_string())),
                    };
                    if let Err(rb) = restored {
                        failures.push(format!("ノード {done}: {rb}"));
                    }
                }
                if !failures.is_empty() {
                    return Err(invalid(format!(
                        "FunctionalModel::apply_parameters: ノード {index} の適用に失敗（{err}）し、巻き戻しにも失敗した（{}）。モデルが部分適用のまま残っている可能性がある",
                        failures.join("; ")
                    )));
                }
                return Err(err);
            }
        }
        Ok(())
    }

    /// optimizer／loss を設定する（Keras `compile`）。再 compile は optimizer 状態を破棄して置換する。
    ///
    /// `Optimizer::Lbfgs` は `InvalidArgument`（第 1 段の対象外。モジュール doc）。`OptimizerState::new` が
    /// 成功してから代入する（construct-before-assign。失敗時は `compiled` を変更しない）。
    pub(crate) fn compile(
        &mut self,
        optimizer: Optimizer,
        loss: Loss,
    ) -> Result<(), AutodiffError> {
        if matches!(optimizer, Optimizer::Lbfgs(_)) {
            return Err(invalid(
                "FunctionalModel::compile: Optimizer::Lbfgs は第 1 段で未対応（closure 駆動の更新経路が要るため）"
                    .to_string(),
            ));
        }
        let state = OptimizerState::new(optimizer)?;
        self.compiled = Some(Compiled {
            optimizer: state,
            loss,
            amp: None,
        });
        Ok(())
    }

    /// `compile` 済みなら `true`。
    pub(crate) fn is_compiled(&self) -> bool {
        self.compiled.is_some()
    }

    /// `fit`／`evaluate` の共通引数検査（モード変更・データ構築より前に呼ぶ）。
    fn check_io_counts(
        &self,
        method: &str,
        x_count: usize,
        y_count: usize,
    ) -> Result<(), AutodiffError> {
        if x_count != self.inputs.len() {
            return Err(invalid(format!(
                "FunctionalModel::{method}: 入力テンソル数 {x_count} がモデルの入力数 {} と一致しない",
                self.inputs.len()
            )));
        }
        if y_count != self.outputs.len() {
            return Err(invalid(format!(
                "FunctionalModel::{method}: 目標テンソル数 {y_count} がモデルの出力数 {} と一致しない",
                self.outputs.len()
            )));
        }
        for (_, block) in self.blocks() {
            block.reject_seq_first_layer_for_fit(&format!("FunctionalModel::{method}"))?;
        }
        Ok(())
    }

    /// 学習（Keras `fit`）。`xs`／`ys` は `build` に渡した `inputs`／`outputs` の順。多出力の損失は
    /// 各出力の損失の和（モジュール doc）。`History` は `loss`・`lr` のみ埋める。
    ///
    /// 拒否（いずれもモード変更・データ構築より前。`compiled` は保持）: 未 compile・入力／目標件数の
    /// 不一致・`epochs == 0`・`accumulate_steps != 1`・パラメータ持ちの `add_module` 層・
    /// `batch_first=false` の MHA。サンプル数不一致・`batch_size == 0`・loss と目標 dtype の不整合は
    /// データ構築時／最初のバッチで `InvalidArgument`。`drop_last` で全バッチが落ちた epoch も
    /// `InvalidArgument`。
    pub(crate) fn fit<T: FitTarget>(
        &mut self,
        xs: &[&Tensor<f32>],
        ys: &[&Tensor<T>],
        config: FitConfig,
    ) -> Result<History, AutodiffError> {
        let mut compiled = self.compiled.take().ok_or_else(|| not_compiled("fit"))?;
        let prepared = self.prepare_fit(xs, ys, config);
        let loader = match prepared {
            Ok(loader) => loader,
            Err(e) => {
                self.compiled = Some(compiled);
                return Err(e);
            }
        };
        let prev_training = self.training();
        self.set_training(true);
        let result = self.run_fit(&mut compiled, &loader, config);
        self.set_training(prev_training);
        self.compiled = Some(compiled);
        result
    }

    /// `fit` の引数検査とデータローダ構築（副作用なし）。
    fn prepare_fit<T: FitTarget>(
        &self,
        xs: &[&Tensor<f32>],
        ys: &[&Tensor<T>],
        config: FitConfig,
    ) -> Result<DataLoader<GraphDataset<T>>, AutodiffError> {
        if config.epochs == 0 {
            return Err(invalid("FunctionalModel::fit: epochs == 0".to_string()));
        }
        if config.accumulate_steps != 1 {
            return Err(invalid(format!(
                "FunctionalModel::fit: accumulate_steps は 1 のみ対応（指定 {}。第 1 段の対象外）",
                config.accumulate_steps
            )));
        }
        self.check_io_counts("fit", xs.len(), ys.len())?;
        for (_, block) in self.blocks() {
            block.reject_untracked_parametric_layer("FunctionalModel::fit")?;
        }
        build_loader(
            "fit",
            xs,
            ys,
            DataLoaderConfig::new(config.batch_size)
                .shuffle(config.shuffle)
                .drop_last(config.drop_last),
        )
    }

    /// `fit` の本体（`compiled` は取り外し済み。`self` の `bind`〈`&self`〉と `compiled.optimizer.step`
    /// 〈`&mut compiled`〉を同時に生かすため、`Sequential::run_fit` と同じ構造にする）。
    fn run_fit<T: FitTarget>(
        &mut self,
        compiled: &mut Compiled,
        loader: &DataLoader<GraphDataset<T>>,
        config: FitConfig,
    ) -> Result<History, AutodiffError> {
        let mut loss = Vec::new();
        loss.try_reserve_exact(config.epochs)
            .map_err(|_| super::super::alloc_failed())?;
        let mut lr = Vec::new();
        lr.try_reserve_exact(config.epochs)
            .map_err(|_| super::super::alloc_failed())?;
        for _ in 0..config.epochs {
            lr.push(compiled.optimizer.lr());
            let mut weighted_sum = 0.0f64;
            let mut count = 0usize;
            for batch in loader {
                let (x_batch, y_batch) =
                    batch.map_err(|e| invalid(format!("FunctionalModel::fit: {e}")))?;
                let n_batch = x_batch
                    .first()
                    .and_then(|t| t.shape().first().copied())
                    .unwrap_or(0);
                let stepped = {
                    let tape = crate::tape();
                    let bound = self.bind(&tape);
                    let x_vars: Vec<Var<'_>> = x_batch.iter().map(|b| tape.var(b)).collect();
                    let preds = bound.forward(&tape, &x_vars)?;
                    let loss_var = combined_loss::<T>(compiled.loss, &tape, &preds, &y_batch)?;
                    let loss_scalar = loss_var.to_tensor().get(&[]).ok_or_else(|| {
                        invalid(
                            "FunctionalModel::fit: loss の shape が [] ではない（loss 演算の契約違反）"
                                .to_string(),
                        )
                    })?;
                    weighted_sum += loss_scalar as f64 * n_batch as f64;
                    count += n_batch;
                    let grads = tape.backward(&loss_var)?;
                    let grad_refs = bound.trainable_grads(&grads)?;
                    let param_refs = self.trainable_parameters();
                    compiled.optimizer.step(&param_refs, &grad_refs)?
                };
                self.apply_parameters(stepped)?;
            }
            if count == 0 {
                // `drop_last = true` で全バッチが落ちた場合（0 除算を黙って NaN にしない）。
                return Err(invalid(
                    "FunctionalModel::fit: この epoch で処理されたサンプルが 0 件（drop_last により全バッチが切り捨てられた可能性がある）"
                        .to_string(),
                ));
            }
            loss.push((weighted_sum / count as f64) as f32);
        }
        Ok(History {
            loss,
            val_loss: Vec::new(),
            lr,
            val_metrics: Vec::new(),
        })
    }

    /// 評価（Keras `evaluate`）。eval モードで `forward`（`bind` なし）し、`fit` と同じ損失合計・同じ
    /// 集計でサンプル数重み付き平均を返す。モードは呼び出し前へ復元する。
    pub(crate) fn evaluate<T: FitTarget>(
        &mut self,
        xs: &[&Tensor<f32>],
        ys: &[&Tensor<T>],
        batch_size: usize,
    ) -> Result<f32, AutodiffError> {
        let loss = self
            .compiled
            .as_ref()
            .ok_or_else(|| not_compiled("evaluate"))?
            .loss;
        self.check_io_counts("evaluate", xs.len(), ys.len())?;
        let loader = build_loader("evaluate", xs, ys, DataLoaderConfig::new(batch_size))?;
        let prev_training = self.training();
        self.set_training(false);
        let result = self.run_evaluate(&loader, loss);
        self.set_training(prev_training);
        result
    }

    /// `evaluate` の本体。
    fn run_evaluate<T: FitTarget>(
        &self,
        loader: &DataLoader<GraphDataset<T>>,
        loss: Loss,
    ) -> Result<f32, AutodiffError> {
        let mut weighted_sum = 0.0f64;
        let mut count = 0usize;
        for batch in loader {
            let (x_batch, y_batch) =
                batch.map_err(|e| invalid(format!("FunctionalModel::evaluate: {e}")))?;
            let n_batch = x_batch
                .first()
                .and_then(|t| t.shape().first().copied())
                .unwrap_or(0);
            let tape = crate::tape();
            let x_vars: Vec<Var<'_>> = x_batch.iter().map(|b| tape.var(b)).collect();
            let preds = self.forward(&tape, &x_vars)?;
            let loss_var = combined_loss::<T>(loss, &tape, &preds, &y_batch)?;
            let loss_scalar = loss_var.to_tensor().get(&[]).ok_or_else(|| {
                invalid(
                    "FunctionalModel::evaluate: loss の shape が [] ではない（loss 演算の契約違反）"
                        .to_string(),
                )
            })?;
            weighted_sum += loss_scalar as f64 * n_batch as f64;
            count += n_batch;
        }
        if count == 0 {
            return Err(invalid(
                "FunctionalModel::evaluate: 処理されたサンプルが 0 件".to_string(),
            ));
        }
        Ok((weighted_sum / count as f64) as f32)
    }
}

impl<'m, 't> FunctionalVars<'m, 't> {
    /// 学習用 forward。ブロックは `bind` 済みの `SequentialVars::forward` で評価し（葉を再登録しない。
    /// `SequentialVars::forward` doc の理由）、評価順・結合演算は `FunctionalModel::forward` と共通の
    /// 評価器を使う。
    pub(crate) fn forward(
        &self,
        tape: &'t Tape,
        inputs: &[Var<'t>],
    ) -> Result<Vec<Var<'t>>, AutodiffError> {
        let mut cursor = self.blocks.iter();
        self.model
            .eval_graph(tape, inputs, &mut |index, _block, x| match cursor.next() {
                Some((bound_index, vars)) if *bound_index == index => vars.forward(tape, x),
                _ => Err(invalid(format!(
                    "内部不整合: ブロックノード {index} の bind 結果が見つからない"
                ))),
            })
    }

    /// 訓練対象パラメータの `Var` 参照列（`FunctionalModel::trainable_parameters` と同じ並び）。
    pub(crate) fn trainable_vars(&self) -> Vec<&Var<'t>> {
        self.blocks
            .iter()
            .flat_map(|(_, vars)| vars.trainable_vars())
            .collect()
    }

    /// `grads` から訓練対象パラメータの勾配を同じ並びで取り出す（欠落は型付きエラー。
    /// `SequentialVars::trainable_grads` が各ブロックで検査する）。
    pub(crate) fn trainable_grads<'g>(
        &self,
        grads: &'g Gradients,
    ) -> Result<Vec<&'g Tensor<f32>>, AutodiffError> {
        let mut out = Vec::new();
        for (_, vars) in &self.blocks {
            let part = vars.trainable_grads(grads)?;
            out.try_reserve(part.len())
                .map_err(|_| super::super::alloc_failed())?;
            out.extend(part);
        }
        Ok(out)
    }
}
