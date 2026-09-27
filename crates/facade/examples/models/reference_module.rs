//! examples 限定の共通 trait `ReferenceModule`／`Trainable`（イシュー
//! #2202・親 #2190。AC3「各層が `nn::Module` trait を実装すること」の
//! 代替実装）。
//!
//! `facade` の `nn::Module` は #2133 で公開保留中のため到達できない
//! （`crates/facade/src/lib.rs` の `NnModuleHoldDoctestGuard`・
//! `docs/facade-nn-module-exposure-decision.md` §12）。examples から
//! 内部の `fandhe_ai_autodiff::nn::Module` を impl することもしない
//! （公開パス限定の方針）。本ファイルはその代わりに、PyTorch
//! `nn.Module` に似せた examples 限定の利用者コードとして
//! `ReferenceModule` trait を定義し、`resnet.rs`／`transformer.rs` の
//! 各型（ブロック・モデル本体の両方）に実装させることで層積層の
//! trait 化を示す。
//!
//! `resnet.rs`・`transformer.rs`・`crates/facade/examples/main.rs`・
//! `crates/facade/tests/example_resnet_cifar10.rs`・
//! `crates/facade/tests/example_transformer_cifar10.rs` の 5 箇所から
//! `#[path]` で直接取り込まれる単独完結ファイル（`mlp.rs`／`lenet.rs`
//! と同じ方針）。取り込み側は `use super::reference_module::...;` で
//! 参照する（`docs/reference-models-decision.md` #2202 節「取り込み
//! 方」参照）。

use fandhe_ai::optim::Adam;
use fandhe_ai::{AutodiffError, Tape, Tensor, Var};

/// PyTorch `nn.Module` に似せた最小限の trait（AC3 の代替）。
///
/// `forward`（推論用。`compat::Sequential::forward` への委譲チェーン）・
/// `named_parameters`（階層名付きパラメータ一覧）・`set_training`
/// （BatchNorm の train/eval 切替の伝播）の 3 メソッドに限定する。
/// 学習ステップ（backward・optimizer 適用）は複合モデルの構造ごとに
/// 演算列が異なる（`ResNetBlock` 単体には自然な loss がない）ため、
/// 本 trait には含めず [`Trainable`]（モデル本体限定）に分離する。
pub trait ReferenceModule {
    /// 外部 `Tape` 上で推論用 forward を計算する。
    fn forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError>;

    /// PyTorch 風の階層名（例: `"layer1.0.main.0.weight"`）付きの
    /// パラメータ一覧を返す。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)>;

    /// 内部の全 `compat::Sequential` 部品へ train/eval モードを伝える
    /// （`BatchNorm` の running stats 更新可否を切り替える。
    /// `compat::Sequential::train`／`eval` への委譲）。
    fn set_training(&mut self, training: bool);
}

/// [`ReferenceModule`] に学習ステップを追加した trait。モデル本体
/// （`ResNet`・`Transformer`）のみが実装し、部品（`ResNetBlock`）は
/// 実装しない。
pub trait Trainable: ReferenceModule {
    /// 1 学習ステップ（tape 構築 → forward → cross entropy → backward →
    /// 勾配抽出 → optimizer 適用 → パラメータ書き戻し）を実行し、
    /// ホスト側 loss 値を返す。
    fn train_step(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<i32>,
        opt: &mut Adam,
    ) -> Result<f32, AutodiffError>;
}

/// mean cross-entropy loss（`-mean(sum(onehot(y) * log_softmax(logits))))`）。
///
/// `facade` は `Reduction`（`Var::cross_entropy_loss` の引数）を
/// 再エクスポートしていないため（`docs/compat-api-scope.md`）、
/// 公開パスだけで呼べる形へ書き下ろす。`onehot_scaled` は正解位置が
/// `1/N`・それ以外が `0` の `[N, C]` 定数（`tape.var` で tape に載せる
/// だけの非学習対象。`Trainable::train_step` はモデル内部パラメータ
/// だけを `trainable_grads` で抽出するため、この定数への勾配は無視
/// される）とすることで、スカラー乗算 op を使わずに mean 相当を
/// 実現する（`docs/reference-models-decision.md` #2202 節参照）。
pub fn cross_entropy_mean<'t>(
    tape: &'t Tape,
    logits: &Var<'t>,
    targets: &Tensor<i32>,
    num_classes: usize,
) -> Result<Var<'t>, AutodiffError> {
    if num_classes == 0 {
        return Err(AutodiffError::InvalidArgument(
            "cross_entropy_mean: num_classes は 0 より大きい必要がある".to_string(),
        ));
    }
    let shape = targets.shape();
    if shape.len() != 1 {
        return Err(AutodiffError::InvalidArgument(format!(
            "cross_entropy_mean: targets の rank は 1 である必要がある（実際: {shape:?}）"
        )));
    }
    let n = shape[0];
    if n == 0 {
        return Err(AutodiffError::InvalidArgument(
            "cross_entropy_mean: targets は空であってはならない".to_string(),
        ));
    }

    let mut onehot = vec![0.0f32; n * num_classes];
    for i in 0..n {
        let t = targets.get(&[i]).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "cross_entropy_mean: targets[{i}] の読み出しに失敗した"
            ))
        })?;
        if t < 0 || (t as usize) >= num_classes {
            return Err(AutodiffError::InvalidArgument(format!(
                "cross_entropy_mean: targets[{i}]={t} が [0, {num_classes}) の範囲外"
            )));
        }
        onehot[i * num_classes + t as usize] = 1.0 / n as f32;
    }
    let onehot_tensor = Tensor::new(onehot, &[n, num_classes]).map_err(|e| {
        AutodiffError::InvalidArgument(format!(
            "cross_entropy_mean: onehot テンソル構築に失敗: {e}"
        ))
    })?;
    let onehot_var = tape.var(&onehot_tensor);

    let log_probs = logits.log_softmax(1)?;
    let weighted = log_probs.mul(&onehot_var)?;
    let summed = weighted.sum(None)?;
    summed.neg()
}

/// スカラー shape `[]` の `Tensor<f32>` から値を取り出す
/// （`Var::mse_loss` 系の既存テストと同じ `.get(&[])` パターン）。
pub fn scalar_of(t: &Tensor<f32>) -> Result<f32, AutodiffError> {
    t.get(&[]).ok_or_else(|| {
        AutodiffError::InvalidArgument(
            "scalar_of: loss テンソルの shape が [] ではない".to_string(),
        )
    })
}

/// `x`（`[N, ...]`）の先頭軸から `[start, start+len)` 行を切り出した
/// `Tensor<f32>` を作る（ミニバッチ抽出用。`Var`/`Tensor` に行選択 API
/// が無いため、`contiguous().as_slice()` で取り出した生データを
/// 直接スライスして作り直す）。
pub fn sub_tensor_f32(
    x: &Tensor<f32>,
    start: usize,
    len: usize,
) -> Result<Tensor<f32>, AutodiffError> {
    let full_shape = x.shape();
    let n = full_shape[0];
    if start + len > n {
        return Err(AutodiffError::InvalidArgument(format!(
            "sub_tensor_f32: [{start}, {}) が先頭軸の長さ {n} を超える",
            start + len
        )));
    }
    let item_len: usize = full_shape[1..].iter().product();
    let contiguous = x.contiguous();
    let flat = contiguous.as_slice().ok_or_else(|| {
        AutodiffError::InvalidArgument("sub_tensor_f32: contiguous() 直後は必ず Some".to_string())
    })?;
    let begin = start * item_len;
    let end = begin + len * item_len;
    let mut shape = full_shape.to_vec();
    shape[0] = len;
    Tensor::new(flat[begin..end].to_vec(), &shape).map_err(|e| {
        AutodiffError::InvalidArgument(format!("sub_tensor_f32: テンソル再構築に失敗: {e}"))
    })
}

/// [`sub_tensor_f32`] の `Tensor<i32>`（ラベル。常に rank 1）版。
pub fn sub_tensor_i32(
    y: &Tensor<i32>,
    start: usize,
    len: usize,
) -> Result<Tensor<i32>, AutodiffError> {
    let n = y.shape()[0];
    if start + len > n {
        return Err(AutodiffError::InvalidArgument(format!(
            "sub_tensor_i32: [{start}, {}) が先頭軸の長さ {n} を超える",
            start + len
        )));
    }
    let contiguous = y.contiguous();
    let flat = contiguous.as_slice().ok_or_else(|| {
        AutodiffError::InvalidArgument("sub_tensor_i32: contiguous() 直後は必ず Some".to_string())
    })?;
    Tensor::new(flat[start..start + len].to_vec(), &[len]).map_err(|e| {
        AutodiffError::InvalidArgument(format!("sub_tensor_i32: テンソル再構築に失敗: {e}"))
    })
}

/// 固定順（shuffle しない）のミニバッチで `epochs` 回学習し、epoch
/// ごとの平均 loss を返す（`.claude/rules/coding-rust.md`「ベンチは
/// 5 回計測の中央値」とは別軸——学習系は決定的シードのみを要件と
/// する）。
pub fn fit_epochs<M: Trainable>(
    model: &mut M,
    x: &Tensor<f32>,
    y: &Tensor<i32>,
    opt: &mut Adam,
    epochs: usize,
    batch_size: usize,
) -> Result<Vec<f32>, AutodiffError> {
    if epochs == 0 || batch_size == 0 {
        return Err(AutodiffError::InvalidArgument(
            "fit_epochs: epochs・batch_size はいずれも 0 より大きい必要がある".to_string(),
        ));
    }
    let n = x.shape()[0];
    if y.shape() != [n] {
        return Err(AutodiffError::InvalidArgument(format!(
            "fit_epochs: x の先頭軸長 {n} と y の shape {:?} が一致しない",
            y.shape()
        )));
    }

    model.set_training(true);
    let mut history = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        let mut total = 0.0f32;
        let mut count = 0usize;
        let mut start = 0;
        while start < n {
            let len = batch_size.min(n - start);
            let xb = sub_tensor_f32(x, start, len)?;
            let yb = sub_tensor_i32(y, start, len)?;
            let loss = model.train_step(&xb, &yb, opt)?;
            total += loss;
            count += 1;
            start += len;
        }
        history.push(total / count as f32);
    }
    Ok(history)
}

/// eval モードで `x`／`y` 全件に対する正解率を計算する。
pub fn accuracy<M: ReferenceModule>(
    model: &mut M,
    x: &Tensor<f32>,
    y: &Tensor<i32>,
    batch_size: usize,
    num_classes: usize,
) -> Result<f32, AutodiffError> {
    if batch_size == 0 {
        return Err(AutodiffError::InvalidArgument(
            "accuracy: batch_size は 0 より大きい必要がある".to_string(),
        ));
    }
    let n = x.shape()[0];
    if n == 0 {
        return Err(AutodiffError::InvalidArgument(
            "accuracy: x は空であってはならない".to_string(),
        ));
    }

    model.set_training(false);
    let mut correct = 0usize;
    let mut start = 0;
    while start < n {
        let len = batch_size.min(n - start);
        let xb = sub_tensor_f32(x, start, len)?;
        let yb = sub_tensor_i32(y, start, len)?;

        let tape = fandhe_ai::tape();
        let xv = tape.var(&xb);
        let logits = model.forward(&tape, &xv)?;
        let out = logits.to_tensor();
        let contiguous = out.contiguous();
        let flat = contiguous.as_slice().ok_or_else(|| {
            AutodiffError::InvalidArgument("accuracy: contiguous() 直後は必ず Some".to_string())
        })?;
        if flat.len() != len * num_classes {
            return Err(AutodiffError::InvalidArgument(format!(
                "accuracy: logits 要素数（{}）が期待値（{}）と一致しない",
                flat.len(),
                len * num_classes
            )));
        }
        for i in 0..len {
            let row = &flat[i * num_classes..(i + 1) * num_classes];
            let mut best_idx = 0usize;
            let mut best_val = row[0];
            for (idx, &v) in row.iter().enumerate().skip(1) {
                if v > best_val {
                    best_val = v;
                    best_idx = idx;
                }
            }
            let label = yb.get(&[i]).ok_or_else(|| {
                AutodiffError::InvalidArgument(format!("accuracy: targets[{i}] の読み出しに失敗"))
            })?;
            if best_idx as i32 == label {
                correct += 1;
            }
        }
        start += len;
    }
    Ok(correct as f32 / n as f32)
}
