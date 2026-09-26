//! L1 損失（`l1_loss`）と、label_smoothing・ignore_index・class_weight
//! 付き CrossEntropy 損失（`cross_entropy_loss_with`）の自由関数（イシュー
//! #2166・親イシュー #2131「PyTorch／TF 置き換えの API 網羅」）。
//!
//! イシュー #2167（親 #2131）で、距離ベースの損失 3 種
//! （[`cosine_embedding_loss`]・[`margin_ranking_loss`]・
//! [`triplet_margin_loss`]）と [`poisson_nll_loss`] を同じ設計枠組み
//! （facade 非公開・`Op` 融合対象外・ホスト参照実装）で追加した。
//! これら 4 損失の数式・境界規約・検査順序は各関数 doc に記す。
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
//!   `(Σ_s L_s)/W`、`Sum` は `Σ_s L_s`。全サンプル ignore、または
//!   `class_weight` の全クラス重み和（`Σ_c w_c`）が 0 のときは
//!   `Mean`／`Sum` いずれも損失 `0.0`・勾配 `0`（既存 `mse_loss` の
//!   `n == 0 → 0.0` 規約と同型）。`W == 0` だが `Σ_c w_c != 0`
//!   （target クラスの重みのみ 0）のときは `Mean`／`Sum` で扱いが
//!   異なる: `Sum` は `Σ_s L_s`（smoothing 項の非ゼロ寄与を反映）、
//!   `Mean` は `Σ_s L_s / 0` が未定義になるため `0.0`・勾配 `0`
//!   （2026-09-26 是正・codex-review 指摘・PR #2283。詳細は
//!   `docs/autodiff-loss-ops-decision.md` §2.2）。蓄積は `f64`・
//!   index 順で行い最後に 1 回だけ `f32` へ downcast する（詳細は
//!   `crate::eval::cross_entropy_loss_with_options_forward`／
//!   `crate::grad::cross_entropy_loss_with_options_vjp` doc 参照）。
//!
//! **PyTorch との差分**（`docs/autodiff-loss-ops-decision.md` §6）:
//! - `ignore_index` は PyTorch の既定 `-100` を採用せず、既定
//!   `None`（無効）とする。明示指定時のみ有効になる。
//! - 全サンプル ignore、または `class_weight` の全クラス重み和が
//!   0 のとき、PyTorch は `NaN` を返すが、本実装は `0.0`（損失・
//!   勾配とも）を返す（`Op::MseLoss` の空バッチ規約を踏襲する
//!   安全側の判断）。`W` のみが 0（他クラスは非ゼロ）の場合は上記
//!   のとおり `Mean`／`Sum` で扱いが分かれる。
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
//!
//! イシュー #2168（親 #2131）で [`ctc_loss`]（CTC 損失。PyTorch
//! `nn.CTCLoss` 相当）を同じ枠組みで追加した。`log_probs`（`[T, N, C]`。
//! 追跡対象）以外の入力（`targets`・`input_lengths`・`target_lengths`・
//! `options`）は非追跡データのため `Op::CtcLoss` payload に直接
//! 埋め込む（勾配は `log_probs` の 1 系統のみ）。CPU 上の `f64`
//! アキュムレータ参照実装のみを持ち、CUDA／Metal からはホスト計算への
//! フォールバックで到達する（GPU 専用カーネルはスコープ外）。
//! 数式・境界規約・VJP の設計判断（PyTorch 勾配規約との差分を含む）は
//! `docs/autodiff-ctc-design.md` を正とし、要旨のみ [`ctc_loss`] doc に
//! 記す。

use fandhe_ai_tensor_core::{Tensor, reduce_out_shape, require_same_shape};

use crate::bool_ops::checked_bytes_for;
use crate::error::AutodiffError;
use crate::eval;
use crate::tape::{Op, materialize_fallible};
use crate::var::{Reduction, Var};

/// [`ctc_loss`] のオプション（`blank`・`zero_infinity`。イシュー #2168）。
/// フィールドは非公開（builder メソッドでのみ構築する）。`Default` は
/// PyTorch `nn.CTCLoss()` の既定（`blank = 0`・`zero_infinity = false`）
/// と一致する。
///
/// `#[non_exhaustive]` とする理由: 他損失オプション（[`PoissonNllOptions`]
/// 等）と同じく将来のオプション追加で呼び出し側の構築コードを
/// 破壊しないようにするため。
/// `blank`（`usize` の既定 `0`）・`zero_infinity`（`bool` の既定
/// `false`）はいずれも Rust の型既定値が PyTorch 既定と一致するため
/// `#[derive(Default)]` で導出する（`CrossEntropyOptions` と同じ理由。
/// clippy `derivable_impls` 指摘）。
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CtcLossOptions {
    blank: usize,
    zero_infinity: bool,
}

impl CtcLossOptions {
    /// blank ラベルのクラス添字（PyTorch `blank=` 相当）を設定する。
    /// `0 <= blank < C` の検査は [`ctc_loss`] が実行前に行う。
    pub fn blank(mut self, blank: usize) -> Self {
        self.blank = blank;
        self
    }

    /// `zero_infinity`（PyTorch `zero_infinity=` 相当。`true` なら
    /// 整列不能サンプルの損失・勾配を `0` として扱う）を設定する。
    pub fn zero_infinity(mut self, zero_infinity: bool) -> Self {
        self.zero_infinity = zero_infinity;
        self
    }

    pub(crate) fn blank_value(&self) -> usize {
        self.blank
    }

    pub(crate) fn zero_infinity_value(&self) -> bool {
        self.zero_infinity
    }
}

/// [`triplet_margin_loss`] のオプション（`margin`・`p`・`eps`・`swap`。
/// イシュー #2167）。フィールドは非公開（builder メソッドでのみ構築
/// する）。`Default` は PyTorch `nn.TripletMarginLoss()` の既定
/// （`margin = 1.0`・`p = 2.0`・`eps = 1e-6`・`swap = false`）と一致する。
///
/// `#[non_exhaustive]` とする理由: [`CrossEntropyOptions`] と同じく
/// 将来のオプション追加（例: PyTorch の `reduce`／`size_average` の
/// 遺物系）で呼び出し側の構築コードを破壊しないようにするため。
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TripletMarginOptions {
    margin: f32,
    p: f32,
    eps: f32,
    swap: bool,
}

impl Default for TripletMarginOptions {
    fn default() -> Self {
        TripletMarginOptions {
            margin: 1.0,
            p: 2.0,
            eps: 1e-6,
            swap: false,
        }
    }
}

impl TripletMarginOptions {
    /// マージン（PyTorch `margin=` 相当）を設定する。有限性検査は
    /// [`triplet_margin_loss`] が実行前に行う。
    pub fn margin(mut self, margin: f32) -> Self {
        self.margin = margin;
        self
    }

    /// ノルムの次数 `p`（PyTorch `p=` 相当）を設定する。有限かつ
    /// `>= 1` の検査は [`triplet_margin_loss`] が実行前に行う。
    pub fn p(mut self, p: f32) -> Self {
        self.p = p;
        self
    }

    /// 距離差へ加える安定化項（PyTorch `eps=` 相当。`aten/src/ATen/
    /// native/Distance.cpp::pairwise_distance` と同じく**差にノルムを
    /// 取る前**に加える）。有限かつ `>= 0` の検査は
    /// [`triplet_margin_loss`] が実行前に行う。
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// `swap`（PyTorch `swap=` 相当。`true` のとき負例距離を
    /// `min(d(a,n), d(p,n))` に取り替える）を設定する。
    pub fn swap(mut self, swap: bool) -> Self {
        self.swap = swap;
        self
    }

    pub(crate) fn margin_value(&self) -> f32 {
        self.margin
    }

    pub(crate) fn p_value(&self) -> f32 {
        self.p
    }

    pub(crate) fn eps_value(&self) -> f32 {
        self.eps
    }

    pub(crate) fn swap_value(&self) -> bool {
        self.swap
    }
}

/// [`poisson_nll_loss`] のオプション（`log_input`・`full`・`eps`。
/// イシュー #2167）。フィールドは非公開（builder メソッドでのみ構築
/// する）。`Default` は PyTorch `nn.PoissonNLLLoss()` の既定
/// （`log_input = true`・`full = false`・`eps = 1e-8`）と一致する。
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PoissonNllOptions {
    log_input: bool,
    full: bool,
    eps: f32,
}

impl Default for PoissonNllOptions {
    fn default() -> Self {
        PoissonNllOptions {
            log_input: true,
            full: false,
            eps: 1e-8,
        }
    }
}

impl PoissonNllOptions {
    /// `log_input`（PyTorch `log_input=` 相当。`true` なら
    /// `input` を対数レートとして扱う）を設定する。
    pub fn log_input(mut self, log_input: bool) -> Self {
        self.log_input = log_input;
        self
    }

    /// `full`（PyTorch `full=` 相当。`true` なら Stirling 近似項を
    /// `target > 1` の要素に加える）を設定する。
    pub fn full(mut self, full: bool) -> Self {
        self.full = full;
        self
    }

    /// `log_input = false` のときの `log(input + eps)` 安定化項
    /// （PyTorch `eps=` 相当）を設定する。有限かつ `>= 0` の検査は
    /// [`poisson_nll_loss`] が実行前に行う。
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub(crate) fn log_input_value(&self) -> bool {
        self.log_input
    }

    pub(crate) fn full_value(&self) -> bool {
        self.full
    }

    pub(crate) fn eps_value(&self) -> f32 {
        self.eps
    }
}

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

/// `materialize_triple` の戻り値型（clippy `type_complexity` 回避の
/// ための別名）。
type TripleTensors = (Tensor<f32>, Tensor<f32>, Tensor<f32>);

/// `a`／`b`／`c` を層 1 で実体化した `Tensor<f32>` の組を返す
/// （[`materialize_pair`] の 3 入力版。`triplet_margin_loss` 用）。
fn materialize_triple<'t>(
    a: &Var<'t>,
    b: &Var<'t>,
    c: &Var<'t>,
) -> Result<TripleTensors, AutodiffError> {
    let nodes = a.tape().nodes.borrow();
    let ops = a.tape().ops();
    let a_val = materialize_fallible(&nodes, ops, a.node_id())?.clone();
    let b_val = materialize_fallible(&nodes, ops, b.node_id())?.clone();
    let c_val = materialize_fallible(&nodes, ops, c.node_id())?.clone();
    Ok((a_val, b_val, c_val))
}

/// `y`（`CosineEmbeddingLoss`／`MarginRankingLoss` のラベル）の全要素が
/// 厳密に `1.0` か `-1.0` であることを検査する（違反は
/// `AutodiffError::InvalidArgument`。PyTorch は任意の `y` を許容するが
/// 本実装はより厳しい制約を課す——モジュール doc「PyTorch との差分」
/// 参照）。
fn check_pm_one_labels(y: &Tensor<f32>, fn_name: &str) -> Result<(), AutodiffError> {
    for v in eval::dense_vec(y) {
        if v != 1.0 && v != -1.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "{fn_name}: y の要素は厳密に 1.0 か -1.0 でなければならない（got {v}）"
            )));
        }
    }
    Ok(())
}

/// Cosine 類似度に基づく埋め込み損失（`x1`・`x2` は追跡対象、`y`
/// （`+1`／`-1` ラベル）は非追跡。PyTorch `nn.CosineEmbeddingLoss`
/// 相当。イシュー #2167・親イシュー #2131）。
///
/// **shape 契約**: `x1`・`x2` は同一 shape で rank 1（`[D]`。この場合
/// `y` は 0 次元 `[]`）または rank 2（`[N, D]`。この場合 `y` は
/// `[N]`）のいずれか。
///
/// **数式**（サンプル `n`。`.claude/rules/coding-rust.md` の勾配長軸
/// 縮約契約に従い `m1`・`m2`・内積は要素を先に `f64` へ昇格してから
/// 二乗・積算する）: `m1 = Σ_d x1[n,d]² + EPSILON`・
/// `m2 = Σ_d x2[n,d]² + EPSILON`（`EPSILON = 1e-12`。PyTorch
/// `aten/src/ATen/native/Loss.cpp` の `cosine_embedding_loss` 実装と
/// 同じ値）・`dot = Σ_d x1[n,d]·x2[n,d]`・`cos = dot / sqrt(m1·m2)`。
/// `y[n] == 1.0` なら `L_n = 1 − cos`、`y[n] == -1.0` なら
/// `L_n = max(0, cos − margin)`。`Mean` は `(Σ_n L_n) / N`、`Sum` は
/// `Σ_n L_n`（`N == 0` はいずれも損失 `0.0`）。
///
/// **勾配**: `y=1` 分岐は `dx1 = −s·d(cos)/d(x1)`（符号反転）、`y=-1`
/// 分岐は hinge が有効（`cos − margin >= 0`。`clamp_min` の VJP
/// 契約——境界ちょうど 0 でも勾配を通す）なときのみ
/// `dx1 = s·d(cos)/d(x1)` を流す。`d(cos)/d(x1)[d] = x2[n,d]/sqrt(m1·m2)
/// − cos·x1[n,d]/m1`（`x2` 側は対称）。
///
/// **検査順序**（`l1_loss` と同じ演算メソッド規律）: ①`check_same_tape`
/// → ②`x1`／`x2` の shape 一致（`require_same_shape`）・rank 1 or 2 の
/// 検証・`y` の shape 一致検証 → ③確保前のバイト数上限検査
/// （`checked_bytes_for::<f32>`。REQ-8） → ④`margin` の有限性検査 →
/// ⑤`y` の全要素が `±1` であること（`check_pm_one_labels`） →
/// ⑥実体化（層 1） → ⑦forward 値計算（`eval::cosine_embedding_loss_
/// forward`。`BackendOps` に対応メソッドがないため融合対象外） →
/// ⑧ノード記録。
pub fn cosine_embedding_loss<'t>(
    x1: &Var<'t>,
    x2: &Var<'t>,
    y: &Tensor<f32>,
    margin: f32,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError> {
    x1.check_same_tape(x2)?;
    let x1_shape = x1.shape();
    let x2_shape = x2.shape();
    require_same_shape(&x1_shape, &x2_shape)?;
    let expected_y_shape: Vec<usize> = match x1_shape.len() {
        1 => vec![],
        2 => vec![x1_shape[0]],
        _ => {
            return Err(AutodiffError::InvalidArgument(format!(
                "cosine_embedding_loss: x1/x2 は rank 1（[D]）または rank 2（[N, D]）で\
                 なければならない（got rank {}）",
                x1_shape.len()
            )));
        }
    };
    require_same_shape(y.shape(), &expected_y_shape)?;
    checked_bytes_for::<f32>(&x1_shape)?;

    if !margin.is_finite() {
        return Err(AutodiffError::InvalidArgument(format!(
            "cosine_embedding_loss: margin は有限でなければならない（got {margin}）"
        )));
    }
    check_pm_one_labels(y, "cosine_embedding_loss")?;

    let (x1_val, x2_val) = materialize_pair(x1, x2)?;
    let value = eval::cosine_embedding_loss_forward(&x1_val, &x2_val, y, margin, reduction);
    let id = x1.tape().push_eager(
        Op::CosineEmbeddingLoss {
            x1: x1.node_id(),
            x2: x2.node_id(),
            y: y.clone(),
            margin,
            reduction,
        },
        value,
    );
    Ok(Var::from_raw(x1.tape(), id))
}

/// マージンランキング損失（`x1`・`x2` は追跡対象、`y`（`+1`／`-1`
/// ラベル）は非追跡。全入力は同一 shape の要素ごとの演算。PyTorch
/// `nn.MarginRankingLoss` 相当。イシュー #2167）。
///
/// **数式**: `L_i = max(0, −y_i·(x1_i − x2_i) + margin)`。`Mean` は
/// `numel` で除算、`Sum` はそのまま加算（`numel == 0` はいずれも
/// 損失 `0.0`）。
///
/// **勾配**: hinge が有効（`raw = −y_i·(x1_i−x2_i)+margin >= 0`。
/// `clamp_min` の VJP 契約——境界ちょうど 0 でも勾配を通す）なときのみ
/// `dx1_i = −s·y_i`・`dx2_i = +s·y_i` を流す。
///
/// **検査順序**: ①`check_same_tape` → ②`x1`／`x2`／`y` の shape 一致
/// （`require_same_shape`） → ③確保前のバイト数上限検査
/// （`checked_bytes_for::<f32>`。REQ-8） → ④`margin` の有限性検査 →
/// ⑤`y` の全要素が `±1` であること → ⑥実体化（層 1） → ⑦forward 値
/// 計算（`eval::margin_ranking_loss_forward`） → ⑧ノード記録。
pub fn margin_ranking_loss<'t>(
    x1: &Var<'t>,
    x2: &Var<'t>,
    y: &Tensor<f32>,
    margin: f32,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError> {
    x1.check_same_tape(x2)?;
    let x1_shape = x1.shape();
    let x2_shape = x2.shape();
    require_same_shape(&x1_shape, &x2_shape)?;
    require_same_shape(y.shape(), &x1_shape)?;
    checked_bytes_for::<f32>(&x1_shape)?;

    if !margin.is_finite() {
        return Err(AutodiffError::InvalidArgument(format!(
            "margin_ranking_loss: margin は有限でなければならない（got {margin}）"
        )));
    }
    check_pm_one_labels(y, "margin_ranking_loss")?;

    let (x1_val, x2_val) = materialize_pair(x1, x2)?;
    let value = eval::margin_ranking_loss_forward(&x1_val, &x2_val, y, margin, reduction);
    let id = x1.tape().push_eager(
        Op::MarginRankingLoss {
            x1: x1.node_id(),
            x2: x2.node_id(),
            y: y.clone(),
            margin,
            reduction,
        },
        value,
    );
    Ok(Var::from_raw(x1.tape(), id))
}

/// トリプレットマージン損失（`anchor`・`positive`・`negative` はいずれも
/// 追跡対象で同一 shape。PyTorch `nn.TripletMarginLoss` 相当。イシュー
/// #2167）。
///
/// **shape 契約**: 3 入力は同一 shape で rank 1（`[D]`。単一サンプル）
/// または rank 2（`[N, D]`）のいずれか。
///
/// **数式**（[`TripletMarginOptions`] doc・モジュール doc §2.3
/// 参照。距離は `eps` を差へ**先に**加えてから `p` ノルムを取る
/// ——`aten/src/ATen/native/Distance.cpp::pairwise_distance` に準拠）:
/// `d(u, v) = ‖u − v + eps‖_p`。`d_ap = d(anchor, positive)`、
/// `d_an = d(anchor, negative)`。`swap = true` のとき
/// `d_pn = d(positive, negative)`・`d_neg = min(d_an, d_pn)`、それ
/// 以外は `d_neg = d_an`。`L_n = max(0, d_ap − d_neg + margin)`。
/// `Mean` は `N` で除算（rank 1 は `N=1`）、`Sum` はそのまま加算
/// （`N == 0` はいずれも損失 `0.0`）。
///
/// **勾配**: hinge が有効（`clamp_min` 契約）なときのみ
/// `d_ap` の寄与 `+1`・`d_neg` の寄与 `−1` を、各距離のノルム勾配
/// （`p_norm_grad`。`‖v‖_p == 0` は勾配 `0`）経由で `anchor`・
/// `positive`・`negative` へ配分する。`swap` で `d_an == d_pn`
/// （同値）のときは PyTorch `min.other` の VJP 契約と同じく
/// `d_an`・`d_pn` へ寄与を等分（`1/2` ずつ）する。
///
/// **検査順序**: ①`anchor.check_same_tape(positive)`・
/// `anchor.check_same_tape(negative)` → ②3 入力の shape 一致・rank 1
/// or 2 の検証 → ③確保前のバイト数上限検査（REQ-8） → ④`options.margin`
/// の有限性・`options.p >= 1` かつ有限・`options.eps >= 0` かつ有限の
/// 検査 → ⑤実体化（層 1） → ⑥forward 値計算（`eval::triplet_margin_
/// loss_forward`） → ⑦ノード記録。
pub fn triplet_margin_loss<'t>(
    anchor: &Var<'t>,
    positive: &Var<'t>,
    negative: &Var<'t>,
    options: &TripletMarginOptions,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError> {
    anchor.check_same_tape(positive)?;
    anchor.check_same_tape(negative)?;
    let shape = anchor.shape();
    require_same_shape(&shape, &positive.shape())?;
    require_same_shape(&shape, &negative.shape())?;
    if shape.len() != 1 && shape.len() != 2 {
        return Err(AutodiffError::InvalidArgument(format!(
            "triplet_margin_loss: anchor/positive/negative は rank 1（[D]）または\
             rank 2（[N, D]）でなければならない（got rank {}）",
            shape.len()
        )));
    }
    checked_bytes_for::<f32>(&shape)?;

    let margin = options.margin_value();
    if !margin.is_finite() {
        return Err(AutodiffError::InvalidArgument(format!(
            "triplet_margin_loss: margin は有限でなければならない（got {margin}）"
        )));
    }
    let p = options.p_value();
    if !p.is_finite() || p < 1.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "triplet_margin_loss: p は有限かつ 1.0 以上でなければならない（got {p}）"
        )));
    }
    let eps = options.eps_value();
    if !eps.is_finite() || eps < 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "triplet_margin_loss: eps は有限かつ非負でなければならない（got {eps}）"
        )));
    }

    let (anchor_val, positive_val, negative_val) = materialize_triple(anchor, positive, negative)?;
    let value = eval::triplet_margin_loss_forward(
        &anchor_val,
        &positive_val,
        &negative_val,
        options,
        reduction,
    );
    let id = anchor.tape().push_eager(
        Op::TripletMarginLoss {
            anchor: anchor.node_id(),
            positive: positive.node_id(),
            negative: negative.node_id(),
            options: options.clone(),
            reduction,
        },
        value,
    );
    Ok(Var::from_raw(anchor.tape(), id))
}

/// ポアソン負対数尤度損失（`input`・`target` はいずれも追跡対象で
/// 同一 shape。PyTorch `nn.PoissonNLLLoss` 相当。イシュー #2167）。
///
/// **数式**（[`PoissonNllOptions`] doc・モジュール doc §2.3 参照）:
/// `log_input = true` なら `L = exp(x) − t·x`、`false` なら
/// `L = x − t·log(x + eps)`。`full = true` のとき、`t > 1` の要素にのみ
/// Stirling 近似項 `t·log(t) − t + 0.5·log(2π·t)` を加える（`t <= 1`
/// は寄与 0 のままスキップし、`t·log(t)` を計算してからマスクする
/// 経路は取らない——`t = 0` で `NaN` を生まないため）。`Mean` は
/// `numel` で除算、`Sum` はそのまま加算（`numel == 0` はいずれも
/// 損失 `0.0`）。
///
/// **勾配**: `log_input = true` なら `dx = exp(x) − t`・`dt = −x`、
/// `false` なら `dx = 1 − t/(x+eps)`・`dt = −log(x+eps)`。`full = true`
/// かつ `t > 1` の要素は `dt` に `log(t) + 0.5/t` を加える。
///
/// **検査順序**: ①`check_same_tape` → ②shape 一致
/// （`require_same_shape`） → ③確保前のバイト数上限検査（REQ-8） →
/// ④`options.eps` の有限性・非負性検査 → ⑤実体化（層 1） → ⑥forward
/// 値計算（`eval::poisson_nll_loss_forward`） → ⑦ノード記録。
pub fn poisson_nll_loss<'t>(
    input: &Var<'t>,
    target: &Var<'t>,
    options: &PoissonNllOptions,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError> {
    input.check_same_tape(target)?;
    let input_shape = input.shape();
    require_same_shape(&input_shape, &target.shape())?;
    checked_bytes_for::<f32>(&input_shape)?;

    let eps = options.eps_value();
    if !eps.is_finite() || eps < 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "poisson_nll_loss: eps は有限かつ非負でなければならない（got {eps}）"
        )));
    }

    let (input_val, target_val) = materialize_pair(input, target)?;
    let value = eval::poisson_nll_loss_forward(&input_val, &target_val, options, reduction);
    let id = input.tape().push_eager(
        Op::PoissonNllLoss {
            input: input.node_id(),
            target: target.node_id(),
            options: options.clone(),
            reduction,
        },
        value,
    );
    Ok(Var::from_raw(input.tape(), id))
}

/// CTC（Connectionist Temporal Classification）損失（`log_probs` は
/// 追跡対象、`targets`・`input_lengths`・`target_lengths` は非追跡。
/// PyTorch `nn.CTCLoss` 相当。イシュー #2168・親イシュー #2131）。CPU 上の
/// `f64` アキュムレータ参照実装のみを持ち、CUDA／Metal からは
/// ホスト計算へのフォールバックで到達する（GPU 専用カーネルは
/// スコープ外）。数式・VJP の設計判断は `docs/autodiff-ctc-design.md`
/// を正とする。
///
/// **shape 契約**: `log_probs` は rank 3（`[T, N, C]`。unbatched
/// `[T, C]` は非対応——モジュール doc「PyTorch との差分」参照）。
/// `targets` はパディング形式（rank 2・厳密に `[N, S]`）または連結形式
/// （rank 1・長さ `Σ target_lengths`）のいずれか。
///
/// **数式**（サンプル `n`。拡張ラベル列 `l' = [blank, l_1, blank, …,
/// blank]`・`L' = 2·target_lengths[n] + 1`）: 前向き再帰 α を対数空間・
/// `f64` で計算し `nll_n = −logaddexp(α_{T_n−1}(L'−1), α_{T_n−1}(L'−2))`
/// （`L'=1` は最後の 1 項のみ）。`T_n=0 ∧ tl_n=0` は `nll_n=0`、
/// `T_n=0 ∧ tl_n>0` は `nll_n=+∞`。`zero_infinity=true` のとき
/// `nll_n=+∞` のサンプルは損失 `0` として扱う。`Mean` は
/// `(1/N)·Σ_n nll_n/max(tl_n,1)`、`Sum` は `Σ_n nll_n`（`N==0` は
/// いずれも `0.0`）。
///
/// **VJP**: 後ろ向き再帰 β は**現フレームの emission を含めない**規約
/// （標準 Graves 規約は α・β 双方が現フレームを含み `lcab` から
/// `lp` を 1 回引くが、`lp=−∞` のとき `−∞−(−∞)=NaN` になるため
/// 不採用。`docs/autodiff-ctc-design.md` §2.4）で計算し、
/// `lcab_{t,k} = logsumexp_{s: l'_s=k}(α_t(s)+β_t(s))` から
/// `γ_{t,k} = exp(lcab_{t,k} + nll_n)`（`nll_n = −log P` なので
/// `log P = −nll_n`、`γ = exp(lcab − log P)`）・
/// `∂nll_n/∂lp[t,n,k] = −γ_{t,k}`
/// を得る（`log_softmax` 合成後は `softmax − γ` に一致し PyTorch
/// 勾配と揃う。テストで固定）。`t >= T_n` の勾配は `0`。
/// `zero_infinity=true` かつ `nll_n=+∞` のサンプルは勾配も全 `0`。
/// `zero_infinity=false` かつ `nll_n=+∞` のとき、有効フレームの勾配は
/// `NaN`（数学的に未定義。PyTorch も `NaN` を返す規約に合わせる）。
///
/// **検査順序**（確保より前に全て終える。REQ-8・A03。本番経路で
/// `unwrap()`／`expect()` は使わない）: ①`log_probs` が rank 3・`C>=1`
/// → ②確保前のバイト数上限検査（`checked_bytes_for::<f32>`） →
/// ③`options.blank < C` → ④`input_lengths.len()==N`・
/// `target_lengths.len()==N` → ⑤各 `n` で `input_lengths[n] <= T` →
/// ⑥`targets` の形式検査（パディング形式は shape `[N, S]`・
/// `target_lengths[n] <= S`、連結形式は長さが `Σ target_lengths`
/// と一致。`checked_add` で計算） → ⑦使われる範囲の全 target 値が
/// `0 <= t < C` かつ `t != blank` → ⑧各 `n` で `L'_n = 2·tl_n+1` を
/// `checked_mul`／`checked_add` で計算 → ⑨α／β バッファの確保前上限
/// 検査（`checked_bytes_for::<f64>`） → ⑩実体化（層 1） → ⑪forward
/// 値計算（`eval::ctc_loss_forward`。`BackendOps` に対応メソッドが
/// ないため融合対象外） → ⑫ノード記録。
#[allow(clippy::too_many_arguments)]
pub fn ctc_loss<'t>(
    log_probs: &Var<'t>,
    targets: &Tensor<i32>,
    input_lengths: &[usize],
    target_lengths: &[usize],
    options: &CtcLossOptions,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError> {
    let log_probs_shape = log_probs.shape();
    if log_probs_shape.len() != 3 {
        return Err(AutodiffError::InvalidArgument(format!(
            "ctc_loss: log_probs は rank 3（[T, N, C]）でなければならない\
             （got rank {}。unbatched [T, C] は非対応）",
            log_probs_shape.len()
        )));
    }
    let (t_max, n, c) = (log_probs_shape[0], log_probs_shape[1], log_probs_shape[2]);
    if c == 0 {
        return Err(AutodiffError::InvalidArgument(
            "ctc_loss: log_probs のクラス数 C は 1 以上でなければならない（got 0）".to_string(),
        ));
    }
    checked_bytes_for::<f32>(&log_probs_shape)?;

    let blank = options.blank_value();
    if blank >= c {
        return Err(AutodiffError::InvalidArgument(format!(
            "ctc_loss: blank（{blank}）は C（{c}）未満でなければならない"
        )));
    }

    if input_lengths.len() != n {
        return Err(AutodiffError::InvalidArgument(format!(
            "ctc_loss: input_lengths の長さ（{}）が N（{n}）と一致しない",
            input_lengths.len()
        )));
    }
    if target_lengths.len() != n {
        return Err(AutodiffError::InvalidArgument(format!(
            "ctc_loss: target_lengths の長さ（{}）が N（{n}）と一致しない",
            target_lengths.len()
        )));
    }
    for (i, &t_n) in input_lengths.iter().enumerate() {
        if t_n > t_max {
            return Err(AutodiffError::InvalidArgument(format!(
                "ctc_loss: input_lengths[{i}]（{t_n}）が T（{t_max}）を超えている"
            )));
        }
    }

    let targets_shape = targets.shape();
    let targets_data = eval::dense_vec_i32(targets);
    match targets_shape.len() {
        2 => {
            if targets_shape[0] != n {
                return Err(AutodiffError::InvalidArgument(format!(
                    "ctc_loss: targets（パディング形式）の shape[0]（{}）が N（{n}）と\
                     一致しない",
                    targets_shape[0]
                )));
            }
            let s = targets_shape[1];
            for (i, &tl) in target_lengths.iter().enumerate() {
                if tl > s {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "ctc_loss: target_lengths[{i}]（{tl}）が S（{s}）を超えている\
                         （パディング形式）"
                    )));
                }
            }
        }
        1 => {
            let expected_len: usize = target_lengths.iter().try_fold(0usize, |acc, &tl| {
                acc.checked_add(tl).ok_or_else(|| {
                    AutodiffError::InvalidArgument(
                        "ctc_loss: target_lengths の総和が usize を overflow する".to_string(),
                    )
                })
            })?;
            if targets_data.len() != expected_len {
                return Err(AutodiffError::InvalidArgument(format!(
                    "ctc_loss: targets（連結形式）の長さ（{}）が target_lengths の総和\
                     （{expected_len}）と一致しない",
                    targets_data.len()
                )));
            }
        }
        _ => {
            return Err(AutodiffError::InvalidArgument(format!(
                "ctc_loss: targets は rank 2（[N, S]。パディング形式）または rank 1\
                 （[Σ target_lengths]。連結形式）でなければならない（got rank {}）",
                targets_shape.len()
            )));
        }
    }

    let sample_targets = eval::ctc_sample_targets(targets, target_lengths);
    for (n_idx, sample) in sample_targets.iter().enumerate() {
        for &t in sample {
            if t < 0 || (t as usize) >= c || (t as usize) == blank {
                return Err(AutodiffError::InvalidArgument(format!(
                    "ctc_loss: targets[{n_idx}] の要素 {t} が範囲 [0, {c}) を外れている、\
                     または blank（{blank}）と一致している"
                )));
            }
        }
    }

    let mut l_prime_max = 0usize;
    for (i, &tl) in target_lengths.iter().enumerate() {
        let l_prime = tl
            .checked_mul(2)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| {
                AutodiffError::InvalidArgument(format!(
                    "ctc_loss: target_lengths[{i}]（{tl}）から拡張ラベル長 2·tl+1 の\
                     計算が overflow する"
                ))
            })?;
        l_prime_max = l_prime_max.max(l_prime);
    }
    checked_bytes_for::<f64>(&[t_max, l_prime_max])?;

    let log_probs_val = {
        let nodes = log_probs.tape().nodes.borrow();
        materialize_fallible(&nodes, log_probs.tape().ops(), log_probs.node_id())?.clone()
    };
    let value = eval::ctc_loss_forward(
        &log_probs_val,
        targets,
        input_lengths,
        target_lengths,
        options,
        reduction,
    );
    let id = log_probs.tape().push_eager(
        Op::CtcLoss {
            log_probs: log_probs.node_id(),
            targets: targets.clone(),
            input_lengths: input_lengths.to_vec(),
            target_lengths: target_lengths.to_vec(),
            options: options.clone(),
            reduction,
        },
        value,
    );
    Ok(Var::from_raw(log_probs.tape(), id))
}
