//! L1 損失（`l1_loss`）と、label_smoothing・ignore_index・class_weight
//! 付き CrossEntropy 損失（`cross_entropy_loss_with`）の自由関数（イシュー
//! #2166・親イシュー #2131「PyTorch／TF 置き換えの API 網羅」）。
//!
//! **facade 非公開（意図的）**: `crates/autodiff/src/reduce_ops.rs`
//! モジュール doc と同じ理由・同じ判断枠組みによる。`Var` は facade
//! （`fandhe_ai` クレート）から直接再エクスポートされるため、`Var` への
//! inherent メソッド追加は即座に facade 公開面へ出てしまう。イシュー
//! #2166 実装計画は facade 公開面（`Var::l1_loss`／`Var::
//! cross_entropy_loss_with` の委譲メソッド追加）を承認事項として明示し、
//! 承認が取れるまでは自由関数として `Var` の外に置き到達不能にする
//! （`docs/autodiff-loss-ops-decision.md` §5「承認事項」）。承認後は
//! `Var` への薄い委譲メソッドを追加し、facade 側の保留ガード
//! （`crates/facade/src/lib.rs::LossOpsHoldDoctestGuard`）を撤去する。
//!
//! **既存 API との関係（R3・後方互換）**: 既存 `Var::cross_entropy_loss`
//! （`var.rs`）のシグネチャ・数値経路は本モジュールの追加によって
//! 一切変更しない。[`cross_entropy_loss_with`] は `options` が既定値
//! （[`CrossEntropyOptions::default`]。`is_default()`）のとき
//! `Var::cross_entropy_loss` をそのまま呼んで返す「丸ごと委譲」経路を
//! 取る（`crate::tape::Op::CrossEntropyLossWithOptions` は非既定
//! オプション時にのみ構築される）ため、既存経路（`eval::
//! cross_entropy_loss`・`grad.rs::cross_entropy_loss_vjp`）には一切
//! 手を入れない。この委譲は `crates/facade/tests/loss_ops_backend_
//! parity.rs` で bit 完全一致を検証する。
//!
//! **PyTorch 相当**: [`l1_loss`] は `nn.L1Loss` 相当（`|pred − target|`
//! の mean／sum 縮約）。[`cross_entropy_loss_with`] は
//! `nn.CrossEntropyLoss(label_smoothing=, ignore_index=, weight=)`
//! 相当（`aten/src/ATen/native/LossNLL.cpp` の label smoothing 実装に
//! 準拠。数式は [`CrossEntropyOptions`] doc・
//! `crate::eval::cross_entropy_loss_with_options_forward` doc 参照）。
//!
//! **新規 `Op`**: [`l1_loss`] は `crate::tape::Op::L1Loss`、非既定
//! オプションの [`cross_entropy_loss_with`] は `crate::tape::Op::
//! CrossEntropyLossWithOptions` を新設する。いずれも既存 `Op::
//! MseLoss`／`Op::CrossEntropyLoss` と同じく `BackendOps` に対応
//! メソッドを持たない融合対象外の演算のため、常にホスト参照実装
//! （`crate::eval`）で計算し `push_eager`（実体化済み）でテープへ
//! 記録する。GPU 専用カーネル（`BackendOps::l1_loss` 等）は本イシューの
//! スコープ外（`docs/autodiff-loss-ops-decision.md` §7「スコープ外」）。
//!
//! **数値契約**:
//! - [`l1_loss`]: `eval::l1_loss_forward` が `f64` アキュムレータで
//!   `Σ|pred − target|` を蓄積し 1 回だけ `f32` へ downcast する。
//!   `numel == 0` は mean／sum とも `0.0`（既存 `mse_loss` と同じ
//!   規約）。勾配は `dPred = scale·sign(pred − target)`
//!   （`sign(0) = 0`・`NaN` は `NaN` を伝播。`grad.rs::l1_grad_sign`）・
//!   `dTarget = −dPred`（既存 `Op::MseLoss` と同じ符号反転パターン）。
//! - [`cross_entropy_loss_with`]（非既定オプション経路）: サンプル
//!   `s`・クラス `c` について `lp_c = x_c − lse`（既存 `cross_entropy_
//!   loss` と同じ max シフト安定化）、`w_c` はクラス重み
//!   （[`CrossEntropyOptions::class_weight`] 未指定は全クラス `1.0`）、
//!   `W = Σ_{非 ignore} w[t_s]`。
//!   `L_s = (1−ε)·w[t_s]·(−lp_{t_s}) + (ε/C)·Σ_c w_c·(−lp_c)`
//!   （ignore されたサンプルは寄与 0・`W` にも含めない）。`Mean` は
//!   `(Σ_s L_s)/W`、`Sum` は `Σ_s L_s`。`W == 0`（全サンプル ignore、
//!   または重み和が 0）は損失 `0.0`・勾配 `0` を返す（既存 `mse_loss`
//!   の `n == 0 → 0.0` 規約と同型）。蓄積は `f64`・index 順で行い
//!   最後に 1 回だけ `f32` へ downcast する（詳細は `crate::eval::
//!   cross_entropy_loss_with_options_forward`／`crate::grad::
//!   cross_entropy_loss_with_options_vjp` doc 参照）。
//!
//! **PyTorch との差分**（`docs/autodiff-loss-ops-decision.md` §6）:
//! - `ignore_index` は PyTorch の既定 `-100` を採用せず、既定
//!   `None`（無効）とする。明示指定時のみ有効になる。
//! - 全サンプル ignore、または `class_weight` の重み和が 0 のとき、
//!   PyTorch は `NaN` を返すが、本実装は `0.0`（損失・勾配とも）を
//!   返す（`Op::MseLoss` の空バッチ規約を踏襲する安全側の判断）。
//! - `class_weight` は非負値のみ許容する（PyTorch は負値も受け付ける
//!   が、負の重みは損失の単調性が崩れるため fail-closed に拒否する
//!   安全側の判断）。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: 両公開
//! 入口は、確保を伴う経路（`materialize_fallible`／`dense_vec`／
//! 出力 `Vec` 確保）よりも前に `checked_bytes_for::<f32>` で入力 shape
//! の確保前バイト数上限を検査する（`reduce_ops.rs` の規律を踏襲）。
//! [`cross_entropy_loss_with`] は加えて targets 添字範囲（`ignore_index`
//! に一致する場合を除き `0 <= t < C`）・`label_smoothing` の有限性と
//! `[0, 1]` 範囲・`class_weight` の shape（厳密に `[C]`）と有限性・
//! 非負性を、いずれも実体化・forward 計算より前に検査する
//! （本番経路で `unwrap()`／`expect()` を使わない）。

use fandhe_ai_tensor_core::{Tensor, reduce_out_shape, require_same_shape};

use crate::bool_ops::checked_bytes_for;
use crate::error::AutodiffError;
use crate::eval;
use crate::tape::{Op, materialize_fallible};
use crate::var::{Reduction, Var};

/// [`cross_entropy_loss_with`] のオプション（label_smoothing・
/// ignore_index・class_weight。イシュー #2166）。フィールドは非公開
/// （builder メソッドでのみ構築する）。`Default` は PyTorch
/// `nn.CrossEntropyLoss()` の既定と同じ「オプションなし」
/// （`label_smoothing = 0.0`・`ignore_index = None`・
/// `class_weight = None`）で、`is_default()` が
/// `true` を返すこの既定値は [`cross_entropy_loss_with`] を既存
/// `Var::cross_entropy_loss` へ丸ごと委譲させる契約の起点になる
/// （モジュール doc「既存 API との関係」参照）。
///
/// `#[non_exhaustive]` とする理由: 将来 PyTorch の他オプション（例:
/// `reduction='none'`）を追加しうるため、追加時に呼び出し側の
/// 構築コードを破壊しないようにする（フィールドは既に非公開のため
/// 構造体リテラル構築は元々できないが、意図を明示する）。
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CrossEntropyOptions {
    label_smoothing: f32,
    ignore_index: Option<i32>,
    class_weight: Option<Tensor<f32>>,
}

impl CrossEntropyOptions {
    /// label smoothing 係数 `ε`（`[0, 1]`。PyTorch
    /// `nn.CrossEntropyLoss(label_smoothing=ε)` 相当）を設定する。
    /// 有限性・範囲検査は [`cross_entropy_loss_with`] が実行前に行う
    /// （fail-closed。ここでは検査しない——builder は純粋な値保持のみ）。
    pub fn label_smoothing(mut self, eps: f32) -> Self {
        self.label_smoothing = eps;
        self
    }

    /// 無視するクラス添字（PyTorch `nn.CrossEntropyLoss(ignore_
    /// index=idx)` 相当。既定 `None` は PyTorch の既定値 `-100` を
    /// 採用しない——モジュール doc「PyTorch との差分」参照）を設定
    /// する。
    pub fn ignore_index(mut self, index: i32) -> Self {
        self.ignore_index = Some(index);
        self
    }

    /// クラス重み `[C]`（PyTorch `nn.CrossEntropyLoss(weight=w)`
    /// 相当）を設定する。shape・有限性・非負性の検査は
    /// [`cross_entropy_loss_with`] が実行前に行う。
    pub fn class_weight(mut self, weight: Tensor<f32>) -> Self {
        self.class_weight = Some(weight);
        self
    }

    /// 全フィールドが既定値（オプションなし）かどうか。
    /// [`cross_entropy_loss_with`] が既存 `Var::cross_entropy_loss` への
    /// 丸ごと委譲を選ぶかどうかの判定に使う（モジュール doc 参照）。
    pub(crate) fn is_default(&self) -> bool {
        self.label_smoothing == 0.0 && self.ignore_index.is_none() && self.class_weight.is_none()
    }

    pub(crate) fn label_smoothing_value(&self) -> f32 {
        self.label_smoothing
    }

    pub(crate) fn ignore_index_value(&self) -> Option<i32> {
        self.ignore_index
    }

    pub(crate) fn class_weight_value(&self) -> Option<&Tensor<f32>> {
        self.class_weight.as_ref()
    }
}

/// `pred`／`target` を層 1 で実体化した `Tensor<f32>` の組を返す
/// （`bool_ops::materialize_pair` と同じ「`nodes` の `RefCell` 借用を
/// このブロック内に閉じ込め、返す前に解放する」パターン。
/// `bool_ops::materialize_pair` は非公開関数のためここで複製する）。
fn materialize_pair<'t>(
    a: &Var<'t>,
    b: &Var<'t>,
) -> Result<(Tensor<f32>, Tensor<f32>), AutodiffError> {
    let nodes = a.tape().nodes.borrow();
    let ops = a.tape().ops();
    let a_val = materialize_fallible(&nodes, ops, a.node_id())?.clone();
    let b_val = materialize_fallible(&nodes, ops, b.node_id())?.clone();
    Ok((a_val, b_val))
}

/// L1 損失（`pred`・`target` はいずれも追跡対象。PyTorch `nn.L1Loss`
/// 相当。イシュー #2166）。`|pred − target|` の `reduction` 縮約
/// （数値契約はモジュール doc 参照）。
///
/// 検査順序（`Var::huber_loss_impl` と同じ演算メソッド規律）:
/// ①`check_same_tape`（テープ不一致は `TapeMismatch`）→ ②shape 一致
/// （`require_same_shape`）→ ③確保前のバイト数上限検査
/// （`checked_bytes_for::<f32>`。REQ-8）→ ④実体化（層 1）→ ⑤forward
/// 値計算（`eval::l1_loss_forward`。`BackendOps` に対応メソッドが
/// ないため融合対象外）→ ⑥ノード記録。
pub fn l1_loss<'t>(
    pred: &Var<'t>,
    target: &Var<'t>,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError> {
    pred.check_same_tape(target)?;
    let pred_shape = pred.shape();
    let target_shape = target.shape();
    require_same_shape(&pred_shape, &target_shape)?;
    checked_bytes_for::<f32>(&pred_shape)?;

    let (pred_val, target_val) = materialize_pair(pred, target)?;
    let value = eval::l1_loss_forward(&pred_val, &target_val, reduction);
    let id = pred.tape().push_eager(
        Op::L1Loss {
            pred: pred.node_id(),
            target: target.node_id(),
            reduction,
        },
        value,
    );
    Ok(Var::from_raw(pred.tape(), id))
}

/// label_smoothing・ignore_index・class_weight 付き CrossEntropy 損失
/// （`logits` = 予測値・追跡対象、`targets` = 正解クラス添字・非追跡。
/// PyTorch `nn.CrossEntropyLoss(label_smoothing=, ignore_index=,
/// weight=)` 相当。イシュー #2166）。`options` が既定値
/// （`CrossEntropyOptions::is_default()`）のときは既存
/// `Var::cross_entropy_loss(targets, class_dim, reduction)` をそのまま
/// 呼んで返す（R3・モジュール doc「既存 API との関係」）。
///
/// 検査順序（既定オプション経路は `Var::cross_entropy_loss` の検査
/// 順序へ委譲。非既定オプション経路の検査順序）: ①`class_dim` 範囲・
/// targets shape 一致（`reduce_out_shape` 再利用）→ ②確保前のバイト数
/// 上限検査（`checked_bytes_for::<f32>`。REQ-8）→ ③`label_smoothing`
/// が有限かつ `[0, 1]`（違反は `AutodiffError::InvalidArgument`）→
/// ④`class_weight` が shape `[C]`・全要素有限かつ非負（違反は
/// `ShapeError::ShapeMismatch`／`InvalidArgument`）→ ⑤targets 全添字が
/// `ignore_index` と一致するか `0 <= t < C`（違反は
/// `InvalidArgument`）→ ⑥実体化（層 1）→ ⑦forward 値計算（`eval::
/// cross_entropy_loss_with_options_forward`）→ ⑧ノード記録。
pub fn cross_entropy_loss_with<'t>(
    logits: &Var<'t>,
    targets: &Tensor<i32>,
    class_dim: usize,
    reduction: Reduction,
    options: &CrossEntropyOptions,
) -> Result<Var<'t>, AutodiffError> {
    if options.is_default() {
        // 既定オプション（`options.is_default()`）は既存経路へ丸ごと
        // 委譲する（R3。モジュール doc 参照）。`Op::
        // CrossEntropyLossWithOptions` は構築されない。
        return logits.cross_entropy_loss(targets, class_dim, reduction);
    }

    let logits_shape = logits.shape();
    let expected_targets_shape = reduce_out_shape(&logits_shape, Some(class_dim))?;
    require_same_shape(targets.shape(), &expected_targets_shape)?;
    checked_bytes_for::<f32>(&logits_shape)?;

    // `reduce_out_shape` が成功した時点で `class_dim < logits_shape.len()`
    // が保証されるため、この添字アクセスは安全（`Var::cross_entropy_
    // loss` と同型の理由。REQ-8「検査済みの添字のみでアクセスする」）。
    let num_classes = logits_shape[class_dim];

    let eps = options.label_smoothing_value();
    if !eps.is_finite() || !(0.0..=1.0).contains(&eps) {
        return Err(AutodiffError::InvalidArgument(format!(
            "cross_entropy_loss_with: label_smoothing は有限かつ [0, 1] の範囲でなければならない（got {eps}）"
        )));
    }

    if let Some(class_weight) = options.class_weight_value() {
        require_same_shape(class_weight.shape(), &[num_classes])?;
        for w in eval::dense_vec(class_weight) {
            if !w.is_finite() || w < 0.0 {
                return Err(AutodiffError::InvalidArgument(format!(
                    "cross_entropy_loss_with: class_weight は有限かつ非負でなければならない（got {w}）"
                )));
            }
        }
    }

    let ignore_index = options.ignore_index_value();
    for t in eval::dense_vec_i32(targets) {
        if ignore_index == Some(t) {
            continue;
        }
        if t < 0 || (t as usize) >= num_classes {
            return Err(AutodiffError::InvalidArgument(format!(
                "cross_entropy_loss_with: target 添字 {t} が範囲 [0, {num_classes}) を外れている（ignore_index との一致でもない）"
            )));
        }
    }

    let logits_val = {
        let nodes = logits.tape().nodes.borrow();
        materialize_fallible(&nodes, logits.tape().ops(), logits.node_id())?.clone()
    };
    let value = eval::cross_entropy_loss_with_options_forward(
        &logits_val,
        targets,
        class_dim,
        reduction,
        options,
    );
    let id = logits.tape().push_eager(
        Op::CrossEntropyLossWithOptions {
            logits: logits.node_id(),
            targets: targets.clone(),
            class_dim,
            reduction,
            options: options.clone(),
        },
        value,
    );
    Ok(Var::from_raw(logits.tape(), id))
}
