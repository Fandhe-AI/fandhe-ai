//! 損失関数群（親イシュー #189「損失関数（MSE・CrossEntropy）の実装」）。
//!
//! `nn::activation`（`nn/activation.rs`）と同じ設計方針を踏襲する:
//! 各構造体は `Var`（`crate::var`）の対応メソッドを呼ぶだけの薄い
//! ラッパーに徹し（REQ-9「互換 API 層は自作コアの上の薄いラッパーに
//! 徹する」の精神を `nn` モジュールにも適用。`nn/mod.rs` の境界説明
//! 参照）、共通 `Module` trait は未定義のため個別に `forward` を公開する
//! （trait 統一は #94/#95 側で設計する）。
//!
//! - #190（TASK-9.1c 相当）で `MseLoss`（`Var::mse_loss_with` の
//!   ラッパー）を追加した。
//! - #191 で `CrossEntropyLoss`（`Var::cross_entropy_loss` の
//!   ラッパー）を追加した。log-softmax → NLL を個別オペ合成せず、
//!   `Var::cross_entropy_loss`（`crate::var`）側で 1 個の融合オペ
//!   （`tape::Op::CrossEntropyLoss`）として実装する（実装計画 §3.1）。
//!   **Softmax 単体の公開活性化は本イシューのスコープ外**のまま維持する
//!   （`nn/activation.rs` の「Softmax は CE と密結合のため対象外」判断を
//!   踏襲）。
//! - #1737（親イシュー #1609）で `BceLoss`（`Var::bce_loss` の
//!   ラッパー。PyTorch `nn.BCELoss` 相当）・`BceWithLogitsLoss`
//!   （`Var::bce_with_logits_loss` の薄いラッパー。PyTorch
//!   `nn.BCEWithLogitsLoss` 相当）を追加した。`MseLoss` と同じ
//!   「`BackendOps` の融合カーネル優先・`Unsupported` のみホスト
//!   参照実装へフォールバック」パターンの実体は `Var` 側にあり、
//!   ここでは呼ぶだけ。
//! - #2167（親イシュー #2131）で距離ベースの損失 3 種
//!   （`CosineEmbeddingLoss`・`MarginRankingLoss`・`TripletMarginLoss`）
//!   と `PoissonNllLoss` を追加した。`L1Loss` と同じく `crate::loss_ops`
//!   の自由関数（`Var` に委譲メソッドを持たない。facade 非公開）を
//!   呼ぶだけの薄いラッパー。
//!
//! `Reduction`（mean/sum 縮約）は MSE・CrossEntropy の両損失で共有する
//! ため `crate::var::Reduction`（#190 が定義）をそのまま再利用し、
//! `nn::loss` 側には重複定義を置かない。

use fandhe_ai_tensor_core::Tensor;

use crate::error::AutodiffError;
use crate::loss_ops;
pub use crate::loss_ops::{CrossEntropyOptions, PoissonNllOptions, TripletMarginOptions};
pub use crate::var::Reduction;
use crate::var::Var;

/// 平均二乗誤差損失。`Var::mse_loss_with` の薄いラッパー
/// （PyTorch `nn.MSELoss` 相当）。`Default` は `Reduction::Mean`
/// （PyTorch `nn.MSELoss` の既定 `reduction='mean'` と一致）。
#[derive(Debug, Clone, Copy)]
pub struct MseLoss {
    reduction: Reduction,
}

impl Default for MseLoss {
    fn default() -> Self {
        MseLoss {
            reduction: Reduction::Mean,
        }
    }
}

impl MseLoss {
    /// 縮約種別を指定して構築する。
    pub fn new(reduction: Reduction) -> Self {
        MseLoss { reduction }
    }

    /// `pred`（予測値）・`target`（正解値）から損失を計算する。
    /// shape 不一致・クロステープは `Var::mse_loss_with` の検査
    /// （`AutodiffError`）をそのまま返す。
    pub fn forward<'t>(&self, pred: &Var<'t>, target: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        pred.mse_loss_with(target, self.reduction)
    }
}

/// L1 損失（`|pred − target|` の縮約。PyTorch `nn.L1Loss` 相当。
/// イシュー #2166・親イシュー #2131）。`crate::loss_ops::l1_loss` の
/// 薄いラッパー（`Var` に委譲メソッドを持たないため直接自由関数を
/// 呼ぶ。`crate::loss_ops` モジュール doc「facade 非公開（意図的）」
/// 参照）。`Default` は `Reduction::Mean`（PyTorch `nn.L1Loss` の
/// 既定 `reduction='mean'` と一致）。
#[derive(Debug, Clone, Copy)]
pub struct L1Loss {
    reduction: Reduction,
}

impl Default for L1Loss {
    fn default() -> Self {
        L1Loss {
            reduction: Reduction::Mean,
        }
    }
}

impl L1Loss {
    /// 縮約種別を指定して構築する。
    pub fn new(reduction: Reduction) -> Self {
        L1Loss { reduction }
    }

    /// `pred`（予測値）・`target`（正解値）から損失を計算する。
    /// shape 不一致・クロステープは `crate::loss_ops::l1_loss` の検査
    /// （`AutodiffError`）をそのまま返す。
    pub fn forward<'t>(&self, pred: &Var<'t>, target: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        loss_ops::l1_loss(pred, target, self.reduction)
    }
}

/// Huber 損失（`|d| < delta` で `0.5·d²`、それ以外で
/// `delta·(|d| − 0.5·delta)`。`d = pred − target`。PyTorch
/// `nn.HuberLoss(delta)` 相当。イシュー #1739）。`Var::huber_loss` の
/// 薄いラッパー。`Default` は `delta = 1.0`・`Reduction::Mean`
/// （PyTorch `nn.HuberLoss` の既定と一致）。
#[derive(Debug, Clone, Copy)]
pub struct HuberLoss {
    delta: f32,
    reduction: Reduction,
}

impl Default for HuberLoss {
    fn default() -> Self {
        HuberLoss {
            delta: 1.0,
            reduction: Reduction::Mean,
        }
    }
}

impl HuberLoss {
    /// `delta`（境界点）・縮約種別を指定して構築する。`delta` の検査
    /// （有限かつ `> 0`）は `forward`（`Var::huber_loss` 経由）が行う。
    pub fn new(delta: f32, reduction: Reduction) -> Self {
        HuberLoss { delta, reduction }
    }

    /// `pred`（予測値）・`target`（正解値）から損失を計算する。
    /// shape 不一致・クロステープ・`delta` 検査違反は `Var::huber_loss`
    /// の検査（`AutodiffError`）をそのまま返す。
    pub fn forward<'t>(&self, pred: &Var<'t>, target: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        pred.huber_loss(target, self.delta, self.reduction)
    }
}

/// SmoothL1 損失（[`HuberLoss`] と同じ折れ点構造だが二次分岐が
/// `0.5·d²/beta` に `beta` でスケールされる点のみ異なる。`beta = 1.0`
/// のとき両者は一致する。PyTorch `nn.SmoothL1Loss(beta)` 相当。
/// イシュー #1739）。`Var::smooth_l1_loss` の薄いラッパー。`Default`
/// は `beta = 1.0`・`Reduction::Mean`（PyTorch `nn.SmoothL1Loss` の
/// 既定と一致）。
#[derive(Debug, Clone, Copy)]
pub struct SmoothL1Loss {
    beta: f32,
    reduction: Reduction,
}

impl Default for SmoothL1Loss {
    fn default() -> Self {
        SmoothL1Loss {
            beta: 1.0,
            reduction: Reduction::Mean,
        }
    }
}

impl SmoothL1Loss {
    /// `beta`（境界点）・縮約種別を指定して構築する。`beta` の検査
    /// （有限かつ `> 0`。`beta = 0` は PyTorch では `nn.L1Loss` 相当に
    /// 退化するが本実装の対象外——他の値と同じ検査で拒否される）は
    /// `forward`（`Var::smooth_l1_loss` 経由）が行う。
    pub fn new(beta: f32, reduction: Reduction) -> Self {
        SmoothL1Loss { beta, reduction }
    }

    /// `pred`（予測値）・`target`（正解値）から損失を計算する。
    /// shape 不一致・クロステープ・`beta` 検査違反は
    /// `Var::smooth_l1_loss` の検査（`AutodiffError`）をそのまま返す。
    pub fn forward<'t>(&self, pred: &Var<'t>, target: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        pred.smooth_l1_loss(target, self.beta, self.reduction)
    }
}

/// 二値交差エントロピー損失（確率入力）。`Var::bce_loss` の薄い
/// ラッパー（PyTorch `nn.BCELoss` 相当。イシュー #1737）。`Default` は
/// `Reduction::Mean`（`MseLoss` と同じ既定）。
#[derive(Debug, Clone, Copy)]
pub struct BceLoss {
    reduction: Reduction,
}

impl Default for BceLoss {
    fn default() -> Self {
        BceLoss {
            reduction: Reduction::Mean,
        }
    }
}

impl BceLoss {
    /// 縮約種別を指定して構築する。
    pub fn new(reduction: Reduction) -> Self {
        BceLoss { reduction }
    }

    /// `input`（予測確率 `[0, 1]`）・`target`（正解ラベル）から損失を
    /// 計算する。範囲検査・shape 不一致・クロステープは
    /// `Var::bce_loss` の検査（`AutodiffError`）をそのまま返す。
    pub fn forward<'t>(&self, input: &Var<'t>, target: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.bce_loss(target, self.reduction)
    }
}

/// 二値交差エントロピー損失（logits 入力）。`Var::bce_with_logits_loss`
/// の薄いラッパー（PyTorch `nn.BCEWithLogitsLoss` 相当。イシュー
/// #1737）。`Default` は `Reduction::Mean`。
#[derive(Debug, Clone, Copy)]
pub struct BceWithLogitsLoss {
    reduction: Reduction,
}

impl Default for BceWithLogitsLoss {
    fn default() -> Self {
        BceWithLogitsLoss {
            reduction: Reduction::Mean,
        }
    }
}

impl BceWithLogitsLoss {
    /// 縮約種別を指定して構築する。
    pub fn new(reduction: Reduction) -> Self {
        BceWithLogitsLoss { reduction }
    }

    /// `input`（未正規化の logits）・`target`（正解ラベル）から損失を
    /// 計算する。shape 不一致・クロステープは
    /// `Var::bce_with_logits_loss` の検査（`AutodiffError`）をそのまま
    /// 返す。
    pub fn forward<'t>(&self, input: &Var<'t>, target: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.bce_with_logits_loss(target, self.reduction)
    }
}

/// CrossEntropy 損失（log-sum-exp 安定化・クラス次元指定。#191）。
/// `Var::cross_entropy_loss`（`crate::var`）の薄いラッパー。
#[derive(Debug, Clone, Copy)]
pub struct CrossEntropyLoss {
    /// クラス次元（PyTorch の `[N, C, d1..]` 形状は `class_dim = 1` に
    /// 相当する）。
    pub class_dim: usize,
    pub reduction: Reduction,
}

impl CrossEntropyLoss {
    /// `logits`（予測値・追跡対象）と `targets`（正解クラス添字・
    /// 非追跡）から損失を計算する。検査・数値安定化の実体は
    /// `Var::cross_entropy_loss` 側にあり、ここでは呼び出すだけ
    /// （「薄いラッパー性」は `tests/nn_cross_entropy.rs` で検証する）。
    pub fn forward<'t>(
        &self,
        logits: &Var<'t>,
        targets: &Tensor<i32>,
    ) -> Result<Var<'t>, AutodiffError> {
        logits.cross_entropy_loss(targets, self.class_dim, self.reduction)
    }

    /// `logits`・`targets` に加え、label_smoothing・ignore_index・
    /// class_weight（[`CrossEntropyOptions`]。イシュー #2166）を指定
    /// して損失を計算する。`options` が既定値のときは [`Self::forward`]
    /// と同じ数値経路（既存 `Var::cross_entropy_loss`。bit 完全一致）
    /// になる（`crate::loss_ops::cross_entropy_loss_with` doc 参照）。
    /// 検査・数値安定化の実体は `crate::loss_ops::cross_entropy_loss_with`
    /// 側にあり、ここでは呼び出すだけ（「薄いラッパー性」は
    /// `tests/loss_ops.rs` で検証する）。
    pub fn forward_with<'t>(
        &self,
        logits: &Var<'t>,
        targets: &Tensor<i32>,
        options: &CrossEntropyOptions,
    ) -> Result<Var<'t>, AutodiffError> {
        loss_ops::cross_entropy_loss_with(logits, targets, self.class_dim, self.reduction, options)
    }
}

/// 負対数尤度損失。`Var::nll_loss` の薄いラッパー（PyTorch
/// `nn.NLLLoss` 相当。イシュー #1738・親イシュー #1609）。`Default` は
/// `class_dim = 1`（PyTorch の `[N, C, d1..]` 形状規約）・
/// `Reduction::Mean`。
#[derive(Debug, Clone, Copy)]
pub struct NllLoss {
    /// クラス次元（`CrossEntropyLoss::class_dim` と同じ規約）。
    pub class_dim: usize,
    pub reduction: Reduction,
}

impl Default for NllLoss {
    fn default() -> Self {
        NllLoss {
            class_dim: 1,
            reduction: Reduction::Mean,
        }
    }
}

impl NllLoss {
    /// クラス次元・縮約種別を指定して構築する。
    pub fn new(class_dim: usize, reduction: Reduction) -> Self {
        NllLoss {
            class_dim,
            reduction,
        }
    }

    /// `input`（log 確率・追跡対象）と `targets`（正解クラス添字・
    /// 非追跡）から損失を計算する。検査の実体は `Var::nll_loss` 側に
    /// あり、ここでは呼び出すだけ（「薄いラッパー性」は
    /// `tests/nn_nll_kl_div_loss.rs` で検証する）。
    pub fn forward<'t>(
        &self,
        input: &Var<'t>,
        targets: &Tensor<i32>,
    ) -> Result<Var<'t>, AutodiffError> {
        input.nll_loss(targets, self.class_dim, self.reduction)
    }
}

/// Kullback-Leibler ダイバージェンス損失。`Var::kl_div_loss`／
/// `kl_div_loss_with_log_target` の薄いラッパー（PyTorch `nn.KLDivLoss`
/// 相当。イシュー #1738）。`Default` は `Reduction::Mean`・
/// `log_target = false`（PyTorch 既定と一致）。
#[derive(Debug, Clone, Copy)]
pub struct KlDivLoss {
    pub reduction: Reduction,
    /// `true` のとき `target` を対数確率として扱う（PyTorch
    /// `log_target=True` 相当。`Var::kl_div_loss_with_log_target` へ
    /// 委譲）。
    pub log_target: bool,
}

impl Default for KlDivLoss {
    fn default() -> Self {
        KlDivLoss {
            reduction: Reduction::Mean,
            log_target: false,
        }
    }
}

impl KlDivLoss {
    /// 縮約種別を指定して構築する（`log_target = false`）。
    pub fn new(reduction: Reduction) -> Self {
        KlDivLoss {
            reduction,
            log_target: false,
        }
    }

    /// 縮約種別・`log_target` を指定して構築する。
    pub fn new_with_log_target(reduction: Reduction, log_target: bool) -> Self {
        KlDivLoss {
            reduction,
            log_target,
        }
    }

    /// `input`（log 確率・追跡対象）・`target`（`log_target` に応じ
    /// 確率または対数確率・追跡対象）から損失を計算する。検査の実体は
    /// `Var::kl_div_loss`／`kl_div_loss_with_log_target` 側にあり、
    /// ここでは呼び出すだけ（「薄いラッパー性」は
    /// `tests/nn_nll_kl_div_loss.rs` で検証する）。
    pub fn forward<'t>(&self, input: &Var<'t>, target: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        if self.log_target {
            input.kl_div_loss_with_log_target(target, self.reduction)
        } else {
            input.kl_div_loss(target, self.reduction)
        }
    }
}

/// Cosine 類似度に基づく埋め込み損失。`crate::loss_ops::
/// cosine_embedding_loss` の薄いラッパー（PyTorch
/// `nn.CosineEmbeddingLoss` 相当。イシュー #2167）。`Default` は
/// `margin = 0.0`・`Reduction::Mean`（PyTorch 既定と一致）。
#[derive(Debug, Clone, Copy)]
pub struct CosineEmbeddingLoss {
    pub margin: f32,
    pub reduction: Reduction,
}

impl Default for CosineEmbeddingLoss {
    fn default() -> Self {
        CosineEmbeddingLoss {
            margin: 0.0,
            reduction: Reduction::Mean,
        }
    }
}

impl CosineEmbeddingLoss {
    /// マージン・縮約種別を指定して構築する。
    pub fn new(margin: f32, reduction: Reduction) -> Self {
        CosineEmbeddingLoss { margin, reduction }
    }

    /// `x1`・`x2`（追跡対象）と `y`（`+1`／`-1` ラベル・非追跡）から
    /// 損失を計算する。検査の実体は `crate::loss_ops::
    /// cosine_embedding_loss` 側にあり、ここでは呼び出すだけ。
    pub fn forward<'t>(
        &self,
        x1: &Var<'t>,
        x2: &Var<'t>,
        y: &Tensor<f32>,
    ) -> Result<Var<'t>, AutodiffError> {
        loss_ops::cosine_embedding_loss(x1, x2, y, self.margin, self.reduction)
    }
}

/// マージンランキング損失。`crate::loss_ops::margin_ranking_loss` の
/// 薄いラッパー（PyTorch `nn.MarginRankingLoss` 相当。イシュー
/// #2167）。`Default` は `margin = 0.0`・`Reduction::Mean`（PyTorch
/// 既定と一致）。
#[derive(Debug, Clone, Copy)]
pub struct MarginRankingLoss {
    pub margin: f32,
    pub reduction: Reduction,
}

impl Default for MarginRankingLoss {
    fn default() -> Self {
        MarginRankingLoss {
            margin: 0.0,
            reduction: Reduction::Mean,
        }
    }
}

impl MarginRankingLoss {
    /// マージン・縮約種別を指定して構築する。
    pub fn new(margin: f32, reduction: Reduction) -> Self {
        MarginRankingLoss { margin, reduction }
    }

    /// `x1`・`x2`（追跡対象）と `y`（`+1`／`-1` ラベル・非追跡）から
    /// 損失を計算する。検査の実体は `crate::loss_ops::
    /// margin_ranking_loss` 側にあり、ここでは呼び出すだけ。
    pub fn forward<'t>(
        &self,
        x1: &Var<'t>,
        x2: &Var<'t>,
        y: &Tensor<f32>,
    ) -> Result<Var<'t>, AutodiffError> {
        loss_ops::margin_ranking_loss(x1, x2, y, self.margin, self.reduction)
    }
}

/// トリプレットマージン損失。`crate::loss_ops::triplet_margin_loss` の
/// 薄いラッパー（PyTorch `nn.TripletMarginLoss` 相当。イシュー
/// #2167）。`Default` は [`TripletMarginOptions::default`]（PyTorch
/// 既定: `margin=1.0`・`p=2.0`・`eps=1e-6`・`swap=false`）・
/// `Reduction::Mean`。
#[derive(Debug, Clone)]
pub struct TripletMarginLoss {
    options: TripletMarginOptions,
    reduction: Reduction,
}

impl Default for TripletMarginLoss {
    fn default() -> Self {
        TripletMarginLoss {
            options: TripletMarginOptions::default(),
            reduction: Reduction::Mean,
        }
    }
}

impl TripletMarginLoss {
    /// オプション・縮約種別を指定して構築する。
    pub fn new(options: TripletMarginOptions, reduction: Reduction) -> Self {
        TripletMarginLoss { options, reduction }
    }

    /// `anchor`・`positive`・`negative`（いずれも追跡対象）から損失を
    /// 計算する。検査の実体は `crate::loss_ops::triplet_margin_loss`
    /// 側にあり、ここでは呼び出すだけ。
    pub fn forward<'t>(
        &self,
        anchor: &Var<'t>,
        positive: &Var<'t>,
        negative: &Var<'t>,
    ) -> Result<Var<'t>, AutodiffError> {
        loss_ops::triplet_margin_loss(anchor, positive, negative, &self.options, self.reduction)
    }
}

/// ポアソン負対数尤度損失。`crate::loss_ops::poisson_nll_loss` の薄い
/// ラッパー（PyTorch `nn.PoissonNLLLoss` 相当。イシュー #2167）。
/// `Default` は [`PoissonNllOptions::default`]（PyTorch 既定:
/// `log_input=true`・`full=false`・`eps=1e-8`）・`Reduction::Mean`。
#[derive(Debug, Clone)]
pub struct PoissonNllLoss {
    options: PoissonNllOptions,
    reduction: Reduction,
}

impl Default for PoissonNllLoss {
    fn default() -> Self {
        PoissonNllLoss {
            options: PoissonNllOptions::default(),
            reduction: Reduction::Mean,
        }
    }
}

impl PoissonNllLoss {
    /// オプション・縮約種別を指定して構築する。
    pub fn new(options: PoissonNllOptions, reduction: Reduction) -> Self {
        PoissonNllLoss { options, reduction }
    }

    /// `input`・`target`（いずれも追跡対象）から損失を計算する。
    /// 検査の実体は `crate::loss_ops::poisson_nll_loss` 側にあり、
    /// ここでは呼び出すだけ。
    pub fn forward<'t>(&self, input: &Var<'t>, target: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        loss_ops::poisson_nll_loss(input, target, &self.options, self.reduction)
    }
}

#[cfg(test)]
mod tests {
    //! `nn::loss::MseLoss::forward` が、対応する `Var` メソッド直接呼び
    //! 出しと同一の値・テープ記録を返すことを検証する（「薄いラッパー
    //! 性」の担保。`nn::activation` のテスト方針と同型。#190）。
    //! `CrossEntropyLoss::forward` の同種検証は `tests/nn_cross_entropy.rs`
    //! に含む（#191）。

    use super::*;
    use crate::eval::dense_vec;
    use crate::tape::Tape;

    #[test]
    fn default_reduction_is_mean() {
        assert_eq!(MseLoss::default().reduction, Reduction::Mean);
    }

    #[test]
    fn forward_mean_matches_var_mse_loss() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let pred = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]).unwrap());
        let target = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![0.5, -1.0, 2.5, 1.0], &[2, 2]).unwrap());
        let before = tape.len();

        let via_module = MseLoss::default().forward(&pred, &target).unwrap();
        let via_var = pred.mse_loss(&target).unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn forward_sum_matches_var_mse_loss_with() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let pred = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]).unwrap());
        let target = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![0.5, -1.0, 2.5, 1.0], &[2, 2]).unwrap());

        let via_module = MseLoss::new(Reduction::Sum)
            .forward(&pred, &target)
            .unwrap();
        let via_var = pred.mse_loss_with(&target, Reduction::Sum).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn forward_propagates_shape_mismatch_error() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let pred = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0, 2.0], &[2]).unwrap());
        let target =
            tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap());

        let err = MseLoss::default().forward(&pred, &target).unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));
    }

    #[test]
    fn huber_default_delta_and_reduction() {
        assert_eq!(HuberLoss::default().delta, 1.0);
        assert_eq!(HuberLoss::default().reduction, Reduction::Mean);
    }

    #[test]
    fn huber_forward_matches_var_huber_loss() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let pred = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]).unwrap());
        let target = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![0.5, -1.0, 2.5, 1.0], &[2, 2]).unwrap());
        let before = tape.len();

        let via_module = HuberLoss::new(1.0, Reduction::Mean)
            .forward(&pred, &target)
            .unwrap();
        let via_var = pred.huber_loss(&target, 1.0, Reduction::Mean).unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn huber_forward_propagates_shape_mismatch_error() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let pred = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0, 2.0], &[2]).unwrap());
        let target =
            tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap());

        let err = HuberLoss::default().forward(&pred, &target).unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));
    }

    #[test]
    fn huber_forward_propagates_invalid_delta_error() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let pred = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0, 2.0], &[2]).unwrap());
        let target = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![0.0, 0.0], &[2]).unwrap());

        let err = HuberLoss::new(0.0, Reduction::Mean)
            .forward(&pred, &target)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn smooth_l1_default_beta_and_reduction() {
        assert_eq!(SmoothL1Loss::default().beta, 1.0);
        assert_eq!(SmoothL1Loss::default().reduction, Reduction::Mean);
    }

    #[test]
    fn smooth_l1_forward_matches_var_smooth_l1_loss() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let pred = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]).unwrap());
        let target = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![0.5, -1.0, 2.5, 1.0], &[2, 2]).unwrap());

        let via_module = SmoothL1Loss::new(2.0, Reduction::Sum)
            .forward(&pred, &target)
            .unwrap();
        let via_var = pred.smooth_l1_loss(&target, 2.0, Reduction::Sum).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    /// `nn::loss::BceLoss::forward` が `Var::bce_loss` 直接呼び出しと
    /// 同一の値を返すことを確認する（「薄いラッパー性」の担保。
    /// `forward_mean_matches_var_mse_loss` と同型。イシュー #1737）。
    #[test]
    fn bce_loss_forward_mean_matches_var_bce_loss() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let input = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![0.2, 0.8, 0.5, 0.9], &[2, 2]).unwrap());
        let target = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![0.0, 1.0, 1.0, 0.0], &[2, 2]).unwrap());

        let via_module = BceLoss::default().forward(&input, &target).unwrap();
        let via_var = input.bce_loss(&target, Reduction::Mean).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    /// `delta = 1.0` のとき `HuberLoss` と `SmoothL1Loss` は一致する
    /// （実装計画 §2.1「`beta = 1.0` のとき両者は一致する」の担保）。
    #[test]
    fn huber_and_smooth_l1_agree_at_delta_one() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let pred = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![1.5, -3.0, 0.25, 0.0], &[4]).unwrap());
        let target = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![0.0; 4], &[4]).unwrap());

        let huber = HuberLoss::new(1.0, Reduction::Sum)
            .forward(&pred, &target)
            .unwrap();
        let smooth_l1 = SmoothL1Loss::new(1.0, Reduction::Sum)
            .forward(&pred, &target)
            .unwrap();

        assert_eq!(
            dense_vec(&huber.to_tensor()),
            dense_vec(&smooth_l1.to_tensor())
        );
    }

    /// `nn::loss::BceWithLogitsLoss::forward` が
    /// `Var::bce_with_logits_loss` 直接呼び出しと同一の値を返すことを
    /// 確認する（イシュー #1737）。
    #[test]
    fn bce_with_logits_loss_forward_sum_matches_var() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let input = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![-2.0, 1.5, 0.0, 3.0], &[2, 2]).unwrap());
        let target = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![0.0, 1.0, 1.0, 0.0], &[2, 2]).unwrap());

        let via_module = BceWithLogitsLoss::new(Reduction::Sum)
            .forward(&input, &target)
            .unwrap();
        let via_var = input.bce_with_logits_loss(&target, Reduction::Sum).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    /// `nn::loss::NllLoss::forward` が `Var::nll_loss` 直接呼び出しと
    /// 同一の値を返すことを確認する（「薄いラッパー性」の担保。
    /// `forward_mean_matches_var_mse_loss` と同型。イシュー #1738）。
    #[test]
    fn nll_loss_forward_mean_matches_var_nll_loss() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let input = tape.var(
            &fandhe_ai_tensor_core::Tensor::new(vec![-0.1, -2.0, -1.5, -0.3], &[2, 2]).unwrap(),
        );
        let targets = fandhe_ai_tensor_core::Tensor::new(vec![0, 1], &[2]).unwrap();

        let via_module = NllLoss::default().forward(&input, &targets).unwrap();
        let via_var = input.nll_loss(&targets, 1, Reduction::Mean).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    /// `nn::loss::KlDivLoss::forward` が `Var::kl_div_loss` 直接呼び出し
    /// と同一の値を返すことを確認する（イシュー #1738）。
    #[test]
    fn kl_div_loss_forward_sum_matches_var() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let input = tape.var(
            &fandhe_ai_tensor_core::Tensor::new(vec![-2.0, -0.5, -1.2, -0.1], &[2, 2]).unwrap(),
        );
        let target = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![0.2, 0.8, 0.5, 0.5], &[2, 2]).unwrap());

        let via_module = KlDivLoss::new(Reduction::Sum)
            .forward(&input, &target)
            .unwrap();
        let via_var = input.kl_div_loss(&target, Reduction::Sum).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    /// `nn::loss::KlDivLoss::forward`（`log_target = true`）が
    /// `Var::kl_div_loss_with_log_target` 直接呼び出しと同一の値を返す
    /// ことを確認する（イシュー #1738）。
    #[test]
    fn kl_div_loss_forward_log_target_matches_var() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let input = tape.var(
            &fandhe_ai_tensor_core::Tensor::new(vec![-2.0, -0.5, -1.2, -0.1], &[2, 2]).unwrap(),
        );
        let target = tape.var(
            &fandhe_ai_tensor_core::Tensor::new(vec![-1.6, -0.2, -0.7, -0.7], &[2, 2]).unwrap(),
        );

        let via_module = KlDivLoss::new_with_log_target(Reduction::Mean, true)
            .forward(&input, &target)
            .unwrap();
        let via_var = input
            .kl_div_loss_with_log_target(&target, Reduction::Mean)
            .unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    /// [`CosineEmbeddingLoss::default`]・[`MarginRankingLoss::default`]・
    /// [`TripletMarginLoss::default`]・[`PoissonNllLoss::default`] が
    /// PyTorch 既定値と一致することを固定する（イシュー #2167）。
    #[test]
    fn distance_and_poisson_defaults_match_pytorch() {
        let cos = CosineEmbeddingLoss::default();
        assert_eq!(cos.margin, 0.0);
        assert_eq!(cos.reduction, Reduction::Mean);

        let mr = MarginRankingLoss::default();
        assert_eq!(mr.margin, 0.0);
        assert_eq!(mr.reduction, Reduction::Mean);

        let triplet = TripletMarginLoss::default();
        assert_eq!(triplet.reduction, Reduction::Mean);
        assert_eq!(triplet.options.margin_value(), 1.0);
        assert_eq!(triplet.options.p_value(), 2.0);
        assert_eq!(triplet.options.eps_value(), 1e-6);
        assert!(!triplet.options.swap_value());

        let poisson = PoissonNllLoss::default();
        assert_eq!(poisson.reduction, Reduction::Mean);
        assert!(poisson.options.log_input_value());
        assert!(!poisson.options.full_value());
        assert_eq!(poisson.options.eps_value(), 1e-8);
    }

    /// `nn::loss::CosineEmbeddingLoss::forward` が `crate::loss_ops::
    /// cosine_embedding_loss` 直接呼び出しと同一の値を返すことを確認
    /// する（「薄いラッパー性」の担保。イシュー #2167）。
    #[test]
    fn cosine_embedding_loss_forward_matches_free_fn() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x1 = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0, 0.0], &[2]).unwrap());
        let x2 = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![0.0, 1.0], &[2]).unwrap());
        let y = fandhe_ai_tensor_core::Tensor::new(vec![1.0], &[]).unwrap();

        let via_module = CosineEmbeddingLoss::default()
            .forward(&x1, &x2, &y)
            .unwrap();
        let via_fn =
            crate::loss_ops::cosine_embedding_loss(&x1, &x2, &y, 0.0, Reduction::Mean).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_fn.to_tensor())
        );
    }

    /// `nn::loss::MarginRankingLoss::forward` が `crate::loss_ops::
    /// margin_ranking_loss` 直接呼び出しと同一の値を返すことを確認
    /// する（イシュー #2167）。
    #[test]
    fn margin_ranking_loss_forward_matches_free_fn() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x1 = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0, -1.0], &[2]).unwrap());
        let x2 = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![0.0, 0.0], &[2]).unwrap());
        let y = fandhe_ai_tensor_core::Tensor::new(vec![1.0, -1.0], &[2]).unwrap();

        let via_module = MarginRankingLoss::new(0.5, Reduction::Sum)
            .forward(&x1, &x2, &y)
            .unwrap();
        let via_fn =
            crate::loss_ops::margin_ranking_loss(&x1, &x2, &y, 0.5, Reduction::Sum).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_fn.to_tensor())
        );
    }

    /// `nn::loss::TripletMarginLoss::forward` が `crate::loss_ops::
    /// triplet_margin_loss` 直接呼び出しと同一の値を返すことを確認
    /// する（イシュー #2167）。
    #[test]
    fn triplet_margin_loss_forward_matches_free_fn() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let a = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![0.0, 0.0], &[2]).unwrap());
        let p = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0, 0.0], &[2]).unwrap());
        let n = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![0.0, 3.0], &[2]).unwrap());
        let options = TripletMarginOptions::default();

        let via_module = TripletMarginLoss::new(TripletMarginOptions::default(), Reduction::Sum)
            .forward(&a, &p, &n)
            .unwrap();
        let via_fn =
            crate::loss_ops::triplet_margin_loss(&a, &p, &n, &options, Reduction::Sum).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_fn.to_tensor())
        );
    }

    /// `nn::loss::PoissonNllLoss::forward` が `crate::loss_ops::
    /// poisson_nll_loss` 直接呼び出しと同一の値を返すことを確認する
    /// （イシュー #2167）。
    #[test]
    fn poisson_nll_loss_forward_matches_free_fn() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let input = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![0.1, -0.2, 0.5, 0.0], &[2, 2]).unwrap());
        let target = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0, 2.0, 0.0, 3.0], &[2, 2]).unwrap());
        let options = PoissonNllOptions::default();

        let via_module = PoissonNllLoss::new(PoissonNllOptions::default(), Reduction::Sum)
            .forward(&input, &target)
            .unwrap();
        let via_fn =
            crate::loss_ops::poisson_nll_loss(&input, &target, &options, Reduction::Sum).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_fn.to_tensor())
        );
    }
}
