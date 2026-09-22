//! `fit()` の分類 metrics 対応（accuracy・precision・recall・F1・
//! confusion matrix。イシュー #2072・親 #2059）。
//!
//! [`super::training::Sequential::fit_with_metrics`] が validation
//! バッチごとの予測クラス（[`crate::Var::argmax`]。非微分・非追跡）と
//! 正解クラス添字（`Tensor<i32>`）を [`ConfusionAccumulator`] へ集計し、
//! epoch 末に [`MetricsResult`] へ確定して
//! [`super::training::History::val_metrics`] へ積む（REQ-9「互換 API
//! 層は自作コアの上の薄いラッパーに徹する」——新規 `Op`／`BackendOps`／
//! VJP／カーネルは一切追加しない。集計はホスト側の整数カウント＋`f64`
//! 導出のみ）。
//!
//! # バックエンド非依存性（受入基準 F）
//!
//! [`MetricsResult::compute`]・[`ConfusionAccumulator`] が扱う入力は
//! 予測クラス・正解クラスという整数添字のみであり、算術はホスト側
//! `f64`（最後に 1 回だけ `f32` へ downcast）で行う。バックエンド依存性は
//! forward 出力（`logits`。`Var::argmax` の入力）のみに閉じるため、
//! metrics 算術自体は CPU／CUDA／Metal で bit 同一になる契約
//! （`.claude/rules/coding-rust.md` REQ-2 の複合判定ではなく
//! `assert_eq!` 契約。`crates/facade/tests/
//! compat_sequential_metrics_backend_parity.rs` 参照）。
//!
//! # 平均方式（macro 平均。マイクロ／重み付き平均は対象外）
//!
//! precision／recall はクラスごとの `tp/(tp+fp)`／`tp/(tp+fn)`（分母 0
//! は sklearn `zero_division=0` と同じく 0 として寄与）を算術平均した
//! macro 平均。F1 は **クラスごとの F1（`2PcRc/(Pc+Rc)`。分母 0 は 0）を
//! 算術平均した macro F1** であり、macro precision／recall の調和平均
//! ではない（両者は一般に異なる値になる）。

use crate::{AutodiffError, Tensor};

/// `fit_with_metrics`／[`MetricsResult::compute`] が計算する指標の種別
/// （Keras `fit(metrics=[...])` 相当。イシュー #2072）。
///
/// `#[non_exhaustive]`: [`super::callbacks::Monitor`] が `Copy, Eq` を
/// 導出するための直接の型パラメータであり、後続の指標追加
/// （`docs/compat-metrics-design.md` 対象外節）が既存呼び出し元の
/// 非網羅的 `match` を破壊しないようにする。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metrics {
    /// 正解率（`trace(confusion) / total`）。
    Accuracy,
    /// macro 平均 precision（本モジュール doc「平均方式」節）。
    Precision,
    /// macro 平均 recall（本モジュール doc「平均方式」節）。
    Recall,
    /// macro F1（クラス別 F1 の算術平均。本モジュール doc「平均方式」
    /// 節）。
    F1,
    /// 混同行列（行優先 `[num_classes * num_classes]`。`row` = 正解
    /// クラス・`col` = 予測クラス）。`Monitor::ValMetric` で監視する
    /// と非スカラーのため `Sequential::fit_with_metrics` が
    /// `InvalidArgument` で拒否する（`training.rs` 事前検査節）。
    ConfusionMatrix,
}

/// [`MetricsResult::compute`]／`fit_with_metrics` の validation 集計
/// 結果（1 epoch 分。イシュー #2072）。
///
/// `#[non_exhaustive]`: 後続フィールド追加時に既存の struct 更新構文
/// 未使用の呼び出し元を壊さない（`History` と同じ方針）。要求した
/// [`Metrics`] のみ対応フィールドが `Some` になる（要求していない
/// 指標は `None` のまま——余分な計算コストをかけない）。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct MetricsResult {
    /// logits の クラス数 `C`（`confusion_matrix` の行・列サイズ）。
    pub num_classes: usize,
    /// [`Metrics::Accuracy`] を要求した場合のみ `Some`。
    pub accuracy: Option<f32>,
    /// [`Metrics::Precision`]（macro 平均）を要求した場合のみ `Some`。
    pub precision: Option<f32>,
    /// [`Metrics::Recall`]（macro 平均）を要求した場合のみ `Some`。
    pub recall: Option<f32>,
    /// [`Metrics::F1`]（macro F1）を要求した場合のみ `Some`。
    pub f1: Option<f32>,
    /// [`Metrics::ConfusionMatrix`] を要求した場合のみ `Some`
    /// （行優先 `[num_classes * num_classes]`。`row` = 正解クラス・
    /// `col` = 予測クラス）。
    pub confusion_matrix: Option<Vec<u64>>,
}

impl MetricsResult {
    /// `logits`（`[N, C]`。forward 出力）と `target`（`[N]`。正解クラス
    /// 添字）から `metrics` が要求する指標を計算する（単体入口。
    /// `Sequential::fit_with_metrics` の validation 経路も内部で同じ
    /// 混同行列アキュムレータを使う）。
    ///
    /// 予測クラスは `crate::tape()` 上で `tape.var_no_grad(logits)
    /// .argmax(Some(1))`（`Var::argmax` doc の非微分・タイは先頭添字・
    /// NaN 無視契約をそのまま継承）で求める。
    ///
    /// # Errors
    ///
    /// `logits` が rank 2 でない・`target` の shape が `[N]` でない・
    /// `target` の添字が `[0, C)` の範囲外・`N == 0`（総サンプル 0）の
    /// いずれかで `AutodiffError::InvalidArgument`（fail-closed。
    /// `.claude/rules/security.md` A03 の精神——本番経路で `unwrap`／
    /// `expect` を使わない）。
    pub fn compute(
        metrics: &[Metrics],
        logits: &Tensor<f32>,
        target: &Tensor<i32>,
    ) -> Result<MetricsResult, AutodiffError> {
        let logits_shape = logits.shape();
        if logits_shape.len() != 2 {
            return Err(AutodiffError::InvalidArgument(format!(
                "MetricsResult::compute: logits の shape は rank 2 [N, C] である必要がある \
                 (実際は {logits_shape:?})"
            )));
        }
        let n = logits_shape[0];
        let num_classes = logits_shape[1];
        let target_shape = target.shape();
        if target_shape != [n] {
            return Err(AutodiffError::InvalidArgument(format!(
                "MetricsResult::compute: target の shape は [{n}] である必要がある \
                 (実際は {target_shape:?})"
            )));
        }
        if n == 0 {
            return Err(AutodiffError::InvalidArgument(
                "MetricsResult::compute: logits/target のサンプル数が 0".to_string(),
            ));
        }

        let tape = crate::tape();
        let logits_var = tape.var_no_grad(logits);
        let pred_classes = logits_var.argmax(Some(1))?;

        let mut acc = ConfusionAccumulator::new(num_classes)?;
        acc.observe(&pred_classes, target)?;
        acc.finish(metrics)
    }
}

/// 混同行列をバッチ横断で蓄積する非公開アキュムレータ
/// （`u64` カウント。`fit_with_metrics` の validation ループが複数
/// バッチにまたがって [`Self::observe`] を呼び、epoch 末に
/// [`Self::finish`] で [`MetricsResult`] を確定する）。
pub(super) struct ConfusionAccumulator {
    num_classes: usize,
    /// 行優先 `[num_classes * num_classes]`（`row` = 正解クラス・
    /// `col` = 予測クラス）。
    counts: Vec<u64>,
    total: u64,
}

impl ConfusionAccumulator {
    /// `num_classes == 0` または `num_classes * num_classes` が
    /// `usize` で表現できない場合は `InvalidArgument`（`checked_mul`。
    /// 巨大クラス数での panic／OOM を避ける fail-closed 検査）。
    pub(super) fn new(num_classes: usize) -> Result<Self, AutodiffError> {
        if num_classes == 0 {
            return Err(AutodiffError::InvalidArgument(
                "MetricsResult: logits のクラス数（shape[1]）が 0".to_string(),
            ));
        }
        let cells = num_classes.checked_mul(num_classes).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "MetricsResult: num_classes={num_classes} の混同行列サイズが usize で表現できない"
            ))
        })?;
        let mut counts = Vec::new();
        counts.try_reserve_exact(cells).map_err(|e| {
            AutodiffError::InvalidArgument(format!(
                "MetricsResult: 混同行列（cells={cells}）用の確保に失敗した: {e}"
            ))
        })?;
        counts.resize(cells, 0u64);
        Ok(ConfusionAccumulator {
            num_classes,
            counts,
            total: 0,
        })
    }

    /// 1 バッチ分の予測クラス（`pred_classes`。`Var::argmax` 出力）と
    /// 正解クラス（`targets`）を混同行列へ加算する。両者の要素数が
    /// 一致しない、または `targets` の添字が `[0, num_classes)` を
    /// 外れる場合は `InvalidArgument`（fail-closed。`.claude/rules/
    /// coding-rust.md` REQ-8「境界検査を省略しない」の精神を validation
    /// 集計にも適用する）。
    pub(super) fn observe(
        &mut self,
        pred_classes: &Tensor<i32>,
        targets: &Tensor<i32>,
    ) -> Result<(), AutodiffError> {
        let pred_dense = pred_classes.contiguous();
        let target_dense = targets.contiguous();
        let pred_slice = pred_dense.as_slice().ok_or_else(|| {
            AutodiffError::InvalidArgument(
                "MetricsResult: 予測クラステンソルの実体化に失敗した（contiguous() 直後）"
                    .to_string(),
            )
        })?;
        let target_slice = target_dense.as_slice().ok_or_else(|| {
            AutodiffError::InvalidArgument(
                "MetricsResult: target テンソルの実体化に失敗した（contiguous() 直後）".to_string(),
            )
        })?;
        if pred_slice.len() != target_slice.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "MetricsResult: 予測クラス件数 ({}) != target 件数 ({})",
                pred_slice.len(),
                target_slice.len()
            )));
        }
        for (&pred, &tgt) in pred_slice.iter().zip(target_slice.iter()) {
            // `pred`（`Var::argmax` 出力）は縮約 shape `[0, num_classes)`
            // に収まる契約だが、`targets` は呼び出し元由来の外部入力
            // のため明示的に範囲検査する（A03 の精神）。
            if tgt < 0 || (tgt as usize) >= self.num_classes {
                return Err(AutodiffError::InvalidArgument(format!(
                    "MetricsResult: target 添字 {tgt} が範囲 [0, {}) を外れている",
                    self.num_classes
                )));
            }
            if pred < 0 || (pred as usize) >= self.num_classes {
                return Err(AutodiffError::InvalidArgument(format!(
                    "MetricsResult: 予測クラス添字 {pred} が範囲 [0, {}) を外れている\
                     （Var::argmax の契約違反）",
                    self.num_classes
                )));
            }
            let row = tgt as usize;
            let col = pred as usize;
            self.counts[row * self.num_classes + col] += 1;
            self.total += 1;
        }
        Ok(())
    }

    /// 蓄積済みの混同行列から `metrics` が要求する指標を確定する
    /// （`self.total == 0` は呼び出し元——`MetricsResult::compute` の
    /// `n == 0` 検査／`fit_with_metrics` の validation ループが必ず
    /// 1 バッチ以上を処理する契約——により到達しない）。
    pub(super) fn finish(&self, metrics: &[Metrics]) -> Result<MetricsResult, AutodiffError> {
        let want = |m: Metrics| metrics.contains(&m);
        let c = self.num_classes;

        // クラスごとの tp／fp／fn（`f64` で確定。導出は本モジュール
        // doc「バックエンド非依存性」節のとおりホスト側整数演算のみ）。
        let mut tp = vec![0f64; c];
        let mut fp = vec![0f64; c];
        let mut fn_ = vec![0f64; c];
        let mut correct = 0f64;
        for row in 0..c {
            for (col, fp_col) in fp.iter_mut().enumerate() {
                let count = self.counts[row * c + col] as f64;
                if row == col {
                    tp[row] += count;
                    correct += count;
                } else {
                    fn_[row] += count;
                    *fp_col += count;
                }
            }
        }

        let accuracy = want(Metrics::Accuracy).then(|| {
            let total = self.total.max(1) as f64;
            (correct / total) as f32
        });

        let need_precision = want(Metrics::Precision);
        let need_recall = want(Metrics::Recall);
        let need_f1 = want(Metrics::F1);

        let mut precision_sum = 0f64;
        let mut recall_sum = 0f64;
        let mut f1_sum = 0f64;
        if need_precision || need_recall || need_f1 {
            for k in 0..c {
                let p_denom = tp[k] + fp[k];
                let p_k = if p_denom > 0.0 { tp[k] / p_denom } else { 0.0 };
                let r_denom = tp[k] + fn_[k];
                let r_k = if r_denom > 0.0 { tp[k] / r_denom } else { 0.0 };
                precision_sum += p_k;
                recall_sum += r_k;
                let f1_denom = p_k + r_k;
                f1_sum += if f1_denom > 0.0 {
                    2.0 * p_k * r_k / f1_denom
                } else {
                    0.0
                };
            }
        }
        let classes = c.max(1) as f64;
        let precision = need_precision.then(|| (precision_sum / classes) as f32);
        let recall = need_recall.then(|| (recall_sum / classes) as f32);
        let f1 = need_f1.then(|| (f1_sum / classes) as f32);

        let confusion_matrix = want(Metrics::ConfusionMatrix).then(|| self.counts.clone());

        Ok(MetricsResult {
            num_classes: c,
            accuracy,
            precision,
            recall,
            f1,
            confusion_matrix,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 3 クラス confusion matrix（行=正解・列=予測）:
    //       pred0 pred1 pred2
    // true0   5     1     0    (6)
    // true1   2     3     0    (5)
    // true2   0     0     0    (0, このクラスは正解にも予測にも現れない)
    //
    // tp = [5, 3, 0] / fp = [2, 1, 0] / fn = [1, 2, 0]
    // P0 = 5/7, R0 = 5/6, F1_0 = 2*(5/7)*(5/6)/((5/7)+(5/6))
    // P1 = 3/4, R1 = 3/5, F1_1 = 2*(3/4)*(3/5)/((3/4)+(3/5))
    // P2 = 0 (分母0), R2 = 0 (分母0), F1_2 = 0
    fn build_fixture() -> ConfusionAccumulator {
        let preds = crate::Tensor::new(vec![0i32, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1], &[11]).unwrap();
        let targets = crate::Tensor::new(vec![0i32, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], &[11]).unwrap();
        let mut acc = ConfusionAccumulator::new(3).unwrap();
        acc.observe(&preds, &targets).unwrap();
        acc
    }

    #[test]
    fn accuracy_matches_hand_computation() {
        let acc = build_fixture();
        let result = acc
            .finish(&[Metrics::Accuracy])
            .expect("test fixture: finish は成功するはず");
        assert_eq!(result.accuracy, Some(8.0 / 11.0));
        assert_eq!(result.precision, None);
        assert_eq!(result.recall, None);
        assert_eq!(result.f1, None);
        assert_eq!(result.confusion_matrix, None);
    }

    #[test]
    fn macro_precision_recall_match_hand_computation() {
        let acc = build_fixture();
        let result = acc
            .finish(&[Metrics::Precision, Metrics::Recall])
            .expect("test fixture: finish は成功するはず");
        let expected_precision = ((5.0 / 7.0) + (3.0 / 4.0) + 0.0) / 3.0;
        let expected_recall = ((5.0 / 6.0) + (3.0 / 5.0) + 0.0) / 3.0;
        assert!((result.precision.unwrap() - expected_precision as f32).abs() < 1e-6);
        assert!((result.recall.unwrap() - expected_recall as f32).abs() < 1e-6);
    }

    /// macro F1（クラス別 F1 の算術平均）が macro precision／recall の
    /// 調和平均とは異なる値になることを確認する（モジュール doc
    /// 「平均方式」節の判別）。
    #[test]
    fn macro_f1_is_mean_of_per_class_f1_not_harmonic_mean_of_macro_pr() {
        let acc = build_fixture();
        let result = acc
            .finish(&[Metrics::Precision, Metrics::Recall, Metrics::F1])
            .expect("test fixture: finish は成功するはず");
        let p0 = 5.0 / 7.0_f64;
        let r0 = 5.0 / 6.0_f64;
        let f1_0 = 2.0 * p0 * r0 / (p0 + r0);
        let p1 = 3.0 / 4.0_f64;
        let r1 = 3.0 / 5.0_f64;
        let f1_1 = 2.0 * p1 * r1 / (p1 + r1);
        let expected_macro_f1 = (f1_0 + f1_1 + 0.0) / 3.0;

        let macro_p = result.precision.unwrap() as f64;
        let macro_r = result.recall.unwrap() as f64;
        let harmonic_of_macro = 2.0 * macro_p * macro_r / (macro_p + macro_r);

        assert!((result.f1.unwrap() as f64 - expected_macro_f1).abs() < 1e-6);
        assert!((expected_macro_f1 - harmonic_of_macro).abs() > 1e-4);
    }

    #[test]
    fn confusion_matrix_layout_is_row_true_col_pred_and_only_requested_when_asked() {
        let acc = build_fixture();
        let result = acc
            .finish(&[Metrics::ConfusionMatrix])
            .expect("test fixture: finish は成功するはず");
        let cm = result.confusion_matrix.expect("要求したので Some のはず");
        assert_eq!(cm, vec![5, 1, 0, 2, 3, 0, 0, 0, 0]);
        assert_eq!(result.accuracy, None);
    }

    #[test]
    fn compute_rejects_non_rank2_logits() {
        let logits = crate::Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let target = crate::Tensor::<i32>::new(vec![0], &[1]).unwrap();
        let err = MetricsResult::compute(&[Metrics::Accuracy], &logits, &target).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn compute_rejects_target_shape_mismatch() {
        let logits = crate::Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let target = crate::Tensor::<i32>::new(vec![0, 1, 0], &[3]).unwrap();
        let err = MetricsResult::compute(&[Metrics::Accuracy], &logits, &target).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn compute_rejects_out_of_range_target_index() {
        let logits = crate::Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let target = crate::Tensor::<i32>::new(vec![0, 2], &[2]).unwrap();
        let err = MetricsResult::compute(&[Metrics::Accuracy], &logits, &target).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn compute_rejects_zero_samples() {
        let logits = crate::Tensor::<f32>::new(Vec::<f32>::new(), &[0, 2]).unwrap();
        let target = crate::Tensor::<i32>::new(Vec::<i32>::new(), &[0]).unwrap();
        let err = MetricsResult::compute(&[Metrics::Accuracy], &logits, &target).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    /// タイ（先頭添字）・argmax 契約は `Var::argmax` 側の既存テストで
    /// 検証済み（本モジュールは `Var::argmax` 出力をそのまま受け取る
    /// だけで独自の tie-break ロジックを持たない）ため、ここでは
    /// `compute` が argmax 経由で妥当な結果を返すことのみ確認する。
    #[test]
    fn compute_end_to_end_matches_manual_argmax_accuracy() {
        // logits: サンプル 0 は class 1 が最大・サンプル 1 は class 0 が最大。
        let logits = crate::Tensor::<f32>::new(vec![0.1, 0.9, 0.8, 0.2], &[2, 2]).unwrap();
        let target = crate::Tensor::<i32>::new(vec![1, 0], &[2]).unwrap();
        let result = MetricsResult::compute(&[Metrics::Accuracy], &logits, &target)
            .expect("test fixture: compute は成功するはず");
        assert_eq!(result.accuracy, Some(1.0));
    }
}
