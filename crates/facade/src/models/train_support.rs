//! `ResNet`／`TransformerClassifier` の `train_step` 専用の非公開ヘルパー（イシュー #2975・
//! 親 #2541 の Phase 11-2）。
//!
//! `examples/models/reference_module.rs` の `cross_entropy_mean`／`scalar_of` と演算列・
//! 入力検証・エラーメッセージを同一にした複製。公開面には出さない（`pub(super)` のみ。
//! 承認範囲は `docs/reference-models-decision.md` §11.8）。facade は `Reduction` を
//! 再エクスポートしていないため、`Var::gather` で正解クラスの log-probability だけを選ぶ
//! 方式で書く（one-hot 要素積は `0 * -inf = NaN` の汚染を招くため採らない。同 §10.4）。
//! examples 側のコピーは評価用ヘルパー `heldout_loss` が引き続き使う。

use crate::{AutodiffError, Tape, Tensor, Var};

/// mean cross-entropy loss（`-mean(gather(log_softmax(logits), targets))`）。
/// `targets` は rank 1 の `[N]`、`logits` は `[N, num_classes]` に完全一致する必要がある。
pub(super) fn cross_entropy_mean<'t>(
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

    let mut target_idx = vec![0i32; n];
    for (i, slot) in target_idx.iter_mut().enumerate() {
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
        *slot = t;
    }
    let target_idx_tensor = Tensor::new(target_idx, &[n, 1]).map_err(|e| {
        AutodiffError::InvalidArgument(format!(
            "cross_entropy_mean: target_idx テンソル構築に失敗: {e}"
        ))
    })?;

    let weight = vec![1.0f32 / n as f32; n];
    let weight_tensor = Tensor::new(weight, &[n, 1]).map_err(|e| {
        AutodiffError::InvalidArgument(format!(
            "cross_entropy_mean: weight テンソル構築に失敗: {e}"
        ))
    })?;
    let weight_var = tape.var(&weight_tensor);

    // 列数の異なる logits でも in-range な target_idx なら gather が通ってしまうため、
    // shape を事前に完全一致検証する。
    let logits_shape = logits.to_tensor().shape().to_vec();
    if logits_shape != [n, num_classes] {
        return Err(AutodiffError::InvalidArgument(format!(
            "cross_entropy_mean: logits の shape は [{n}, {num_classes}] である必要がある \
             （実際: {logits_shape:?}）"
        )));
    }

    let log_probs = logits.log_softmax(1)?;
    let selected = log_probs.gather(1, &target_idx_tensor)?;
    let weighted = selected.mul(&weight_var)?;
    let summed = weighted.sum(None)?;
    summed.neg()
}

/// スカラー shape `[]` の `Tensor<f32>` から値を取り出す。
pub(super) fn scalar_of(t: &Tensor<f32>) -> Result<f32, AutodiffError> {
    t.get(&[]).ok_or_else(|| {
        AutodiffError::InvalidArgument(
            "scalar_of: loss テンソルの shape が [] ではない".to_string(),
        )
    })
}

/// 更新後パラメータ列 `updated` から `[*idx, *idx + n)` を切り出して `idx` を進める
/// （範囲外は panic ではなく型付きエラー）。`train_step` の書き戻し専用。
pub(super) fn take_updated(
    updated: &[Tensor<f32>],
    idx: &mut usize,
    n: usize,
) -> Result<Vec<Tensor<f32>>, AutodiffError> {
    let end = idx.checked_add(n).ok_or_else(|| {
        AutodiffError::InvalidArgument("train_step: パラメータ位置が usize の範囲を超える".into())
    })?;
    let slice = updated.get(*idx..end).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "train_step: 更新後パラメータ数（{}）が不足している（必要: {end}）",
            updated.len()
        ))
    })?;
    *idx = end;
    Ok(slice.to_vec())
}

/// コンストラクタが確保するパラメータ・定数テンソルの総要素数の上限（`f32` で 4 GiB 相当）。
///
/// `ResNet::new`／`TransformerClassifier::new` は公開引数が任意の `usize` を取りうるため、
/// 巨大引数で `vec!`／`Vec::with_capacity` が capacity overflow で panic したり OOM で
/// abort したりするのを、確保前の上限検証で `AutodiffError::InvalidArgument` に変換する
/// （本番経路の panic 禁止。`.claude/rules/coding-rust.md`）。
pub(super) const MAX_MODEL_ELEMS: usize = 1 << 30;

/// 見積もった総要素数（呼び出し側が `saturating_*` で求めた値）が上限以内か検証する。
pub(super) fn check_model_size(ctx: &str, estimated_elems: usize) -> Result<(), AutodiffError> {
    if estimated_elems > MAX_MODEL_ELEMS {
        return Err(AutodiffError::InvalidArgument(format!(
            "{ctx}: 確保するパラメータ要素数の見積もり（{estimated_elems}）が上限 \
             {MAX_MODEL_ELEMS} を超える"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits_var(tape: &Tape, rows: usize, cols: usize) -> Var<'_> {
        tape.var(&Tensor::<f32>::new(vec![0.1; rows * cols], &[rows, cols]).unwrap())
    }

    #[test]
    fn rejects_logits_column_mismatch() {
        let tape = crate::tape();
        let logits = logits_var(&tape, 2, 3);
        let targets = Tensor::<i32>::new(vec![0, 1], &[2]).unwrap();
        assert!(matches!(
            cross_entropy_mean(&tape, &logits, &targets, 4),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_out_of_range_and_empty_targets() {
        let tape = crate::tape();
        let logits = logits_var(&tape, 2, 3);
        let bad = Tensor::<i32>::new(vec![0, 3], &[2]).unwrap();
        assert!(cross_entropy_mean(&tape, &logits, &bad, 3).is_err());
        let neg = Tensor::<i32>::new(vec![0, -1], &[2]).unwrap();
        assert!(cross_entropy_mean(&tape, &logits, &neg, 3).is_err());
        let empty = Tensor::<i32>::new(vec![], &[0]).unwrap();
        assert!(cross_entropy_mean(&tape, &logits, &empty, 3).is_err());
    }

    #[test]
    fn take_updated_rejects_shortage() {
        let updated = vec![Tensor::<f32>::new(vec![0.0], &[1]).unwrap()];
        let mut idx = 0;
        assert!(take_updated(&updated, &mut idx, 2).is_err());
        assert!(take_updated(&updated, &mut idx, usize::MAX).is_err());
        assert_eq!(take_updated(&updated, &mut idx, 1).unwrap().len(), 1);
        assert_eq!(idx, 1);
    }
}
