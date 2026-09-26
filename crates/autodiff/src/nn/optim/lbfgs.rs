//! L-BFGS（closure・strong Wolfe line search。イシュー #2197・親 #2172）。
//!
//! `torch.optim.LBFGS.step(closure)` 相当を値型で提供する。既存
//! optimizer（[`super::AdamW`] 等）は「`(param, grad)` の参照列 → 更新後
//! `Tensor<f32>` 列」の値型・純関数で 1 step あたり勾配評価 1 回を前提
//! とするが、L-BFGS は line search 中に目的関数・勾配を複数回評価する
//! 必要があり、その形では表現できない。本モジュールは呼び出し元が
//! 用意する **closure**（パラメータ列 → `(損失, 勾配列)`）を内部で複数回
//! 評価する `step` メソッドを提供する（`nn/optim/mod.rs` の「呼び出し元
//! が層を再構築する」不変更新パターンとは異なり、closure 自体が
//! 呼び出し元側で `Tape::backward` 等を駆動して勾配を作る責務を持つ）。
//!
//! # PyTorch との対応・意図的な逸脱（実装計画イシュー #2197 §2 参照）
//!
//! - **戻り値**: 受け入れ条件の字面は `Vec<Tensor<f32>>` 直接返却だが、
//!   本クレートの全 optimizer は `Result<_, AutodiffError>` を返す
//!   契約（`.claude/rules/coding-rust.md` の `unwrap`/`expect` 禁止）の
//!   ため、本 optimizer も `Result` を返す。closure の可失敗性に合わせ、
//!   可失敗 closure を受ける中核メソッド [`Lbfgs::try_step_closure`] と、
//!   不失敗 closure 向けの薄いラッパー [`Lbfgs::step_closure`] の両方を
//!   提供する。
//! - **line search**: `line_search_fn` に相当する [`LbfgsLineSearch`] は
//!   `None`（固定ステップ）・`StrongWolfe`（`torch/optim/lbfgs.py`
//!   `_strong_wolfe`/`_cubic_interpolate` の逐語移植）の 2 択。
//! - **空パラメータ列**: PyTorch は closure を評価して `orig_loss` を
//!   返すが、本実装は LBFGS 状態がフラット化ベクトル全体に対して
//!   定義されるため closure を呼ばずに `InvalidArgument` を返す。
//! - **closure 戻り値の検証**: 勾配数・shape の不一致、非有限の損失・
//!   勾配は毎評価で `InvalidArgument`/`Shape`（fail-closed。
//!   `.claude/rules/security.md` A03）。PyTorch は無検査。
//! - **エラー時の状態不変**: `try_step_closure` 1 回の呼び出し内の
//!   全状態更新はローカル作業コピー上で行い、正常終了時にのみ `self`
//!   へコミットする。closure が途中で `Err` を返しても
//!   `n_iter`/`func_evals`/履歴/`d`/`t`/`prev_flat_grad` は呼び出し前
//!   のまま残る。
//! - 対象外: `maximize`・複素数パラメータ・parameter group・sparse
//!   勾配。facade（`fandhe_ai::optim`）への公開・`compile()` 統合は
//!   別イシュー #2198（ユーザー承認を要する facade 公開面拡張）。
//!
//! # 数値型の方針
//!
//! フラットベクトル上の縮約（`g·d`/`y·s`/`y·y`/`s·q`/`y·r`/`‖g‖₁`）は
//! `f64` アキュムレータで index 順に蓄積し最後に 1 回 `f32` へ丸める
//! （`.claude/rules/coding-rust.md` の縮約方針。他の縮約と混在させ
//! ない）。スカラー演算（`ro`/`H_diag`/`al`/`t`/`gtd`/cubic 補間/Wolfe
//! 条件比較）は PyTorch の f32 テンソル演算に合わせ `f32`。
//! `|loss - prev_loss| < tolerance_change` のみ PyTorch が Python
//! float（f64）で計算するため `loss`/`prev_loss` を `f64` で保持する。
//! ベクトル更新（`q -= al·y` 等）は `f32::mul_add`（FMA 契約）を使う。

use std::collections::VecDeque;

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_ref;

/// line search の方式。`#[non_exhaustive]` は将来の backtracking 追加
/// 等を非破壊にするため（`AutodiffError` と同じ設計判断）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LbfgsLineSearch {
    /// 固定ステップ（`torch.optim.LBFGS(line_search_fn=None)` 相当）。
    None,
    /// strong Wolfe 条件による line search（`line_search_fn="strong_wolfe"`
    /// 相当）。
    StrongWolfe,
}

/// `torch.optim.LBFGS` と同一の既定値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LbfgsConfig {
    pub lr: f32,
    pub max_iter: usize,
    /// `None` の場合 `max_iter * 5 / 4`（PyTorch と同じ整数除算）へ
    /// 解決する。
    pub max_eval: Option<usize>,
    pub tolerance_grad: f32,
    pub tolerance_change: f32,
    pub history_size: usize,
    pub line_search: LbfgsLineSearch,
    /// strong Wolfe 1 回あたりの試行上限の追加キャップ。実効値は
    /// `min(line_search_steps, max_eval - current_evals)`
    /// （PyTorch は後者のみを渡す）。`line_search_steps >= max_eval`
    /// であれば PyTorch 側の項が常に支配し PyTorch と同一になる
    /// （実装計画 §2.3）。
    pub line_search_steps: usize,
}

impl Default for LbfgsConfig {
    fn default() -> LbfgsConfig {
        LbfgsConfig {
            lr: 1.0,
            max_iter: 20,
            max_eval: None,
            tolerance_grad: 1e-7,
            tolerance_change: 1e-9,
            history_size: 100,
            line_search: LbfgsLineSearch::None,
            line_search_steps: 25,
        }
    }
}

impl LbfgsConfig {
    fn resolved_max_eval(&self) -> usize {
        self.max_eval.unwrap_or(self.max_iter * 5 / 4)
    }
}

/// L-BFGS optimizer 本体。[`LbfgsConfig`] と、フラット化ベクトル全体に
/// 対して定義される大域状態（`step()` 呼び出しをまたいで持ち越す）を
/// 保持する。
pub struct Lbfgs {
    config: LbfgsConfig,
    /// 初回 `try_step_closure` 呼び出しで確定するパラメータスロットの
    /// shape 列。以後の呼び出しでスロット数・shape の一致を検査する
    /// （`AdamW` の `SlotState.shape` と同じ規律）。
    slot_shapes: Vec<Vec<usize>>,
    /// PyTorch `state["n_iter"]`（**大域**カウンタ。`step()` 呼び出しを
    /// またいで増加し続ける）。
    n_iter: u64,
    /// PyTorch `state["func_evals"]`。
    func_evals: u64,
    d: Vec<f32>,
    t: f32,
    old_dirs: VecDeque<Vec<f32>>,
    old_stps: VecDeque<Vec<f32>>,
    ro: VecDeque<f32>,
    h_diag: f32,
    prev_flat_grad: Option<Vec<f32>>,
    /// 直近 `step` 呼び出しの初回評価損失（PyTorch `step()` の戻り値
    /// `orig_loss` に相当）。
    last_loss: Option<f32>,
}

impl Lbfgs {
    /// ハイパーパラメータを検証して構築する。
    pub fn new(config: LbfgsConfig) -> Result<Lbfgs, AutodiffError> {
        if !(config.lr.is_finite() && config.lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lbfgs::new: lr must be finite and >= 0.0, got {}",
                config.lr
            )));
        }
        if config.max_iter < 1 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lbfgs::new: max_iter must be >= 1, got {}",
                config.max_iter
            )));
        }
        if config.resolved_max_eval() < 1 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lbfgs::new: max_eval must resolve to >= 1, got {}",
                config.resolved_max_eval()
            )));
        }
        if !(config.tolerance_grad.is_finite() && config.tolerance_grad >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lbfgs::new: tolerance_grad must be finite and >= 0.0, got {}",
                config.tolerance_grad
            )));
        }
        if !(config.tolerance_change.is_finite() && config.tolerance_change >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lbfgs::new: tolerance_change must be finite and >= 0.0, got {}",
                config.tolerance_change
            )));
        }
        if config.history_size < 1 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lbfgs::new: history_size must be >= 1, got {}",
                config.history_size
            )));
        }
        if config.line_search_steps < 1 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lbfgs::new: line_search_steps must be >= 1, got {}",
                config.line_search_steps
            )));
        }
        Ok(Lbfgs {
            config,
            slot_shapes: Vec::new(),
            n_iter: 0,
            func_evals: 0,
            d: Vec::new(),
            t: 0.0,
            old_dirs: VecDeque::new(),
            old_stps: VecDeque::new(),
            ro: VecDeque::new(),
            h_diag: 1.0,
            prev_flat_grad: None,
            last_loss: None,
        })
    }

    pub fn config(&self) -> &LbfgsConfig {
        &self.config
    }

    /// 学習率のみを書き換える（`AdamW::set_lr` と同一の意味論・検証。
    /// 状態〈`n_iter`/履歴/`d`/`t` 等〉は一切リセットしない）。
    pub fn set_lr(&mut self, new_lr: f32) -> Result<(), AutodiffError> {
        if !(new_lr.is_finite() && new_lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lbfgs::set_lr: lr must be finite and >= 0.0, got {new_lr}"
            )));
        }
        self.config.lr = new_lr;
        Ok(())
    }

    /// 直近 `step` の初回評価損失（PyTorch `step()` の戻り値
    /// `orig_loss` 相当）。`step` 未実行なら `None`。
    pub fn last_loss(&self) -> Option<f32> {
        self.last_loss
    }

    /// 累積反復数（PyTorch `state["n_iter"]`）。
    pub fn n_iter(&self) -> u64 {
        self.n_iter
    }

    /// 累積 closure 評価回数（PyTorch `state["func_evals"]`）。
    pub fn func_evals(&self) -> u64 {
        self.func_evals
    }

    /// `params` と同順で更新後の `Tensor<f32>` 列を返す（不失敗
    /// closure 版）。[`Lbfgs::try_step_closure`] の薄いラッパー。
    pub fn step_closure<F>(
        &mut self,
        params: &[Tensor<f32>],
        mut closure: F,
    ) -> Result<Vec<Tensor<f32>>, AutodiffError>
    where
        F: FnMut(&[Tensor<f32>]) -> (f32, Vec<Tensor<f32>>),
    {
        self.try_step_closure(params, |p| Ok(closure(p)))
    }

    /// `params` と同順で更新後の `Tensor<f32>` 列を返す（中核メソッド。
    /// 可失敗 closure を受ける）。
    ///
    /// closure は現在の試行パラメータ列を受け取り `(損失, 勾配列)` を
    /// 返す。line search 中は本メソッドが closure を複数回呼び出す
    /// （`torch.optim.LBFGS.step(closure)` と同じ契約）。
    ///
    /// # Errors
    ///
    /// - `params` が空の場合 closure を呼ばず `InvalidArgument`
    /// - 2 回目以降の呼び出しでスロット数・shape が変化した場合
    ///   `InvalidArgument`/`Shape`
    /// - closure が返す勾配の数・shape が不一致、または損失・勾配に
    ///   非有限値を含む場合 `InvalidArgument`（line search 中の各試行
    ///   評価を含む）
    /// - closure 自体が `Err` を返した場合はそれをそのまま伝播する
    ///
    /// いずれのエラー経路でも `self` の内部状態（`n_iter`/
    /// `func_evals`/履歴/`d`/`t`/`prev_flat_grad` 等）は呼び出し前の
    /// まま変化しない（ローカル作業コピー上で反復し、成功時にのみ
    /// `self` へコミットする）。
    pub fn try_step_closure<F>(
        &mut self,
        params: &[Tensor<f32>],
        mut closure: F,
    ) -> Result<Vec<Tensor<f32>>, AutodiffError>
    where
        F: FnMut(&[Tensor<f32>]) -> Result<(f32, Vec<Tensor<f32>>), AutodiffError>,
    {
        if params.is_empty() {
            return Err(AutodiffError::InvalidArgument(
                "Lbfgs::try_step_closure: params must not be empty".to_string(),
            ));
        }

        let slot_shapes: Vec<Vec<usize>> = if self.slot_shapes.is_empty() {
            params.iter().map(|p| p.shape().to_vec()).collect()
        } else {
            self.slot_shapes.clone()
        };
        if params.len() != slot_shapes.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lbfgs::try_step_closure: slot count changed across calls \
                 (expected {}, got {}); Lbfgs state is keyed by call-order \
                 slot index and cannot be resized after the first step()",
                slot_shapes.len(),
                params.len()
            )));
        }
        for (param, shape) in params.iter().zip(slot_shapes.iter()) {
            if param.shape() != shape.as_slice() {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: param.shape().to_vec(),
                    rhs: shape.clone(),
                }));
            }
        }

        // 初回 closure 評価（現在のパラメータそのものに対して評価する
        // ので x = flatten(params) を再構築する必要はない）。
        let mut func_evals = self.func_evals;
        let (loss0, grads0) = closure(params)?;
        func_evals += 1;
        validate_closure_output(&slot_shapes, loss0, &grads0)?;
        let mut flat_grad = flatten_tensors(&grads0);
        let orig_loss = loss0;

        let opt_cond_initial = abs_max(&flat_grad) <= self.config.tolerance_grad;
        if opt_cond_initial {
            // PyTorch: 最適条件が初回で満たされる場合、closure を 1 回
            // 消費した状態でパラメータ不変のまま return する（`n_iter`
            // は進めない）。
            self.func_evals = func_evals;
            self.slot_shapes = slot_shapes;
            self.last_loss = Some(orig_loss);
            return Ok(params.to_vec());
        }

        // ここから先はローカル作業コピー上で状態を進め、ループを抜けた
        // 後にのみ `self` へコミットする（closure が `Err` を返した
        // 場合の状態不変契約）。
        let mut n_iter_global = self.n_iter;
        let mut d = self.d.clone();
        let mut t = self.t;
        let mut old_dirs = self.old_dirs.clone();
        let mut old_stps = self.old_stps.clone();
        let mut ro = self.ro.clone();
        let mut h_diag = self.h_diag;
        let mut prev_flat_grad = self.prev_flat_grad.clone();
        // ループ内で毎反復上書きしてから同一反復内でのみ読む作業用
        // スカラー（`prev_flat_grad` と異なり `step()` 呼び出しをまたいで
        // 読まれることはないため `self` へは保持しない）。
        let mut prev_loss: f64;

        // `x`: 現在の実パラメータ値のフラット化。line search・固定
        // ステップの各反復末尾で実際に更新される（PyTorch
        // `self._add_grad(t, d)` に相当）。
        let mut x = flatten_tensors(params);

        let max_eval = self.config.resolved_max_eval();
        let mut current_evals = 1usize;
        let mut n_iter_local = 0usize;
        let mut loss = f64::from(orig_loss);
        let mut opt_cond = opt_cond_initial;

        loop {
            if n_iter_local >= self.config.max_iter {
                break;
            }
            n_iter_local += 1;
            n_iter_global += 1;

            if n_iter_global == 1 {
                d = flat_grad.iter().map(|&g| -g).collect();
                old_dirs.clear();
                old_stps.clear();
                ro.clear();
                h_diag = 1.0;
            } else {
                let prev_g = prev_flat_grad.as_ref().ok_or_else(|| {
                    AutodiffError::InvalidArgument(
                        "Lbfgs::try_step_closure: n_iter_global > 1 の時点で \
                         prev_flat_grad が None（直前の反復で必ず設定される \
                         内部不変条件違反。ロジック変更時の退行検出用）"
                            .to_string(),
                    )
                })?;
                let y: Vec<f32> = flat_grad
                    .iter()
                    .zip(prev_g.iter())
                    .map(|(&g, &pg)| g - pg)
                    .collect();
                let s: Vec<f32> = d.iter().map(|&di| di * t).collect();
                let ys = dot_f64(&y, &s);
                if ys > 1e-10 {
                    if old_dirs.len() == self.config.history_size {
                        old_dirs.pop_front();
                        old_stps.pop_front();
                        ro.pop_front();
                    }
                    let yy = dot_f64(&y, &y);
                    old_dirs.push_back(y);
                    old_stps.push_back(s);
                    ro.push_back(1.0 / ys);
                    h_diag = ys / yy;
                }

                let num_old = old_dirs.len();
                let mut al = vec![0f32; num_old];
                let mut q: Vec<f32> = flat_grad.iter().map(|&g| -g).collect();
                for i in (0..num_old).rev() {
                    let al_i = dot_f64(&old_stps[i], &q) * ro[i];
                    al[i] = al_i;
                    for k in 0..q.len() {
                        q[k] = f32::mul_add(-al_i, old_dirs[i][k], q[k]);
                    }
                }
                let mut r: Vec<f32> = q.iter().map(|&qi| qi * h_diag).collect();
                for i in 0..num_old {
                    let be_i = dot_f64(&old_dirs[i], &r) * ro[i];
                    let coeff = al[i] - be_i;
                    for k in 0..r.len() {
                        r[k] = f32::mul_add(coeff, old_stps[i][k], r[k]);
                    }
                }
                d = r;
            }

            prev_flat_grad = Some(flat_grad.clone());
            prev_loss = loss;

            t = if n_iter_global == 1 {
                let sum_abs = abs_sum_f64(&flat_grad) as f32;
                1.0f32.min(1.0 / sum_abs) * self.config.lr
            } else {
                self.config.lr
            };

            let gtd = dot_f64(&flat_grad, &d);
            if gtd > -self.config.tolerance_change {
                break;
            }

            let mut ls_func_evals = 0usize;
            match self.config.line_search {
                LbfgsLineSearch::StrongWolfe => {
                    let max_ls = self.config.line_search_steps.min(max_eval - current_evals);
                    let (new_f, new_g, new_t, ls_evals) = strong_wolfe(
                        &mut closure,
                        &slot_shapes,
                        &x,
                        t,
                        &d,
                        loss as f32,
                        &flat_grad,
                        gtd,
                        max_ls,
                    )?;
                    for k in 0..x.len() {
                        x[k] = f32::mul_add(new_t, d[k], x[k]);
                    }
                    t = new_t;
                    loss = f64::from(new_f);
                    flat_grad = new_g;
                    ls_func_evals = ls_evals;
                    opt_cond = abs_max(&flat_grad) <= self.config.tolerance_grad;
                }
                LbfgsLineSearch::None => {
                    for k in 0..x.len() {
                        x[k] = f32::mul_add(t, d[k], x[k]);
                    }
                    if n_iter_local != self.config.max_iter {
                        let trial_params = unflatten_tensors(&x, &slot_shapes)?;
                        let (new_f, new_grads) = closure(&trial_params)?;
                        validate_closure_output(&slot_shapes, new_f, &new_grads)?;
                        loss = f64::from(new_f);
                        flat_grad = flatten_tensors(&new_grads);
                        opt_cond = abs_max(&flat_grad) <= self.config.tolerance_grad;
                        ls_func_evals = 1;
                    }
                }
            }

            current_evals += ls_func_evals;
            func_evals += ls_func_evals as u64;

            let dt_max = d.iter().fold(0f32, |m, &di| m.max((di * t).abs()));
            let loss_diff = (loss - prev_loss).abs();
            if n_iter_local == self.config.max_iter {
                break;
            }
            if current_evals >= max_eval {
                break;
            }
            if opt_cond {
                break;
            }
            if dt_max <= self.config.tolerance_change {
                break;
            }
            if loss_diff < f64::from(self.config.tolerance_change) {
                break;
            }
        }

        let out = unflatten_tensors(&x, &slot_shapes)?;

        self.slot_shapes = slot_shapes;
        self.n_iter = n_iter_global;
        self.func_evals = func_evals;
        self.d = d;
        self.t = t;
        self.old_dirs = old_dirs;
        self.old_stps = old_stps;
        self.ro = ro;
        self.h_diag = h_diag;
        self.prev_flat_grad = prev_flat_grad;
        self.last_loss = Some(orig_loss);

        Ok(out)
    }
}

/// `params`/closure が返した `grads` を評価直後に検証する
/// （`.claude/rules/security.md` A03）。損失・勾配のいずれかに非有限値
/// を含む、または勾配の数・shape が `slot_shapes` と不一致なら
/// `InvalidArgument`/`Shape` を返す。
fn validate_closure_output(
    slot_shapes: &[Vec<usize>],
    loss: f32,
    grads: &[Tensor<f32>],
) -> Result<(), AutodiffError> {
    if !loss.is_finite() {
        return Err(AutodiffError::InvalidArgument(format!(
            "Lbfgs: closure returned non-finite loss {loss}"
        )));
    }
    if grads.len() != slot_shapes.len() {
        return Err(AutodiffError::InvalidArgument(format!(
            "Lbfgs: closure returned {} gradients, expected {} (one per parameter slot)",
            grads.len(),
            slot_shapes.len()
        )));
    }
    for (grad, shape) in grads.iter().zip(slot_shapes.iter()) {
        if grad.shape() != shape.as_slice() {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: grad.shape().to_vec(),
                rhs: shape.clone(),
            }));
        }
        if dense_vec_ref(grad).iter().any(|v| !v.is_finite()) {
            return Err(AutodiffError::InvalidArgument(
                "Lbfgs: closure returned a non-finite gradient element".to_string(),
            ));
        }
    }
    Ok(())
}

fn flatten_tensors(tensors: &[Tensor<f32>]) -> Vec<f32> {
    let mut out = Vec::new();
    for t in tensors {
        out.extend_from_slice(&dense_vec_ref(t));
    }
    out
}

fn unflatten_tensors(
    flat: &[f32],
    shapes: &[Vec<usize>],
) -> Result<Vec<Tensor<f32>>, AutodiffError> {
    let mut out = Vec::with_capacity(shapes.len());
    let mut offset = 0usize;
    for shape in shapes {
        let numel: usize = shape.iter().product();
        let slice = &flat[offset..offset + numel];
        out.push(Tensor::new(slice.to_vec(), shape)?);
        offset += numel;
    }
    Ok(out)
}

/// `a·b` を `f64` アキュムレータで index 順に蓄積し最後に 1 回 `f32`
/// へ丸める（縮約方針。冒頭 doc 参照）。
fn dot_f64(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0f64;
    for i in 0..a.len() {
        acc = f64::from(a[i]).mul_add(f64::from(b[i]), acc);
    }
    acc as f32
}

/// `Σ|a_i|` を `f64` アキュムレータで蓄積する（縮約方針）。
fn abs_sum_f64(a: &[f32]) -> f64 {
    let mut acc = 0f64;
    for &x in a {
        acc += f64::from(x.abs());
    }
    acc
}

fn abs_max(a: &[f32]) -> f32 {
    a.iter().fold(0f32, |m, &x| m.max(x.abs()))
}

/// `_cubic_interpolate`（`torch/optim/lbfgs.py`）の逐語移植。
fn cubic_interpolate(
    x1: f32,
    f1: f32,
    g1: f32,
    x2: f32,
    f2: f32,
    g2: f32,
    bounds: Option<(f32, f32)>,
) -> f32 {
    let (xmin_bound, xmax_bound) = match bounds {
        Some((lo, hi)) => (lo, hi),
        None => {
            if x1 <= x2 {
                (x1, x2)
            } else {
                (x2, x1)
            }
        }
    };
    let d1 = g1 + g2 - 3.0 * (f1 - f2) / (x1 - x2);
    let d2_square = d1 * d1 - g1 * g2;
    if d2_square >= 0.0 {
        let d2 = d2_square.sqrt();
        let min_pos = if x1 <= x2 {
            x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2.0 * d2))
        } else {
            x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2.0 * d2))
        };
        min_pos.max(xmin_bound).min(xmax_bound)
    } else {
        (xmin_bound + xmax_bound) / 2.0
    }
}

/// `x + t·d` におけるパラメータで closure を評価する
/// （`torch.optim.LBFGS._directional_evaluate` 相当）。`x` 自体は
/// 変更しない（試行パラメータを新しく構築して渡すだけなので、
/// PyTorch の「add → 評価 → restore」と等価）。
fn directional_evaluate<F>(
    closure: &mut F,
    slot_shapes: &[Vec<usize>],
    x: &[f32],
    t: f32,
    d: &[f32],
) -> Result<(f32, Vec<f32>), AutodiffError>
where
    F: FnMut(&[Tensor<f32>]) -> Result<(f32, Vec<Tensor<f32>>), AutodiffError>,
{
    let mut trial = x.to_vec();
    for k in 0..trial.len() {
        trial[k] = f32::mul_add(t, d[k], trial[k]);
    }
    let trial_params = unflatten_tensors(&trial, slot_shapes)?;
    let (loss, grads) = closure(&trial_params)?;
    validate_closure_output(slot_shapes, loss, &grads)?;
    Ok((loss, flatten_tensors(&grads)))
}

/// `_strong_wolfe`（`torch/optim/lbfgs.py`）の逐語移植。
///
/// `c1`/`c2`/`tolerance_change` は PyTorch の `_strong_wolfe` 既定値
/// （`1e-4`/`0.9`/`1e-9`）で固定する。**`LbfgsConfig::tolerance_change`
/// とは独立**（実装計画イシュー #2197 §3。line search 内部の停止
/// 判定は常に PyTorch 既定値を使う）。
///
/// 戻り値は `(損失, 勾配, 採用した step size, ls_func_evals)`。
// `torch/optim/lbfgs.py::_strong_wolfe` の逐語移植（本ファイル冒頭 doc
// 参照）のため、引数を構造体へまとめず PyTorch 側と 1 対 1 対応させる。
#[allow(clippy::too_many_arguments)]
fn strong_wolfe<F>(
    closure: &mut F,
    slot_shapes: &[Vec<usize>],
    x: &[f32],
    t0: f32,
    d: &[f32],
    f0: f32,
    g0: &[f32],
    gtd0: f32,
    max_ls: usize,
) -> Result<(f32, Vec<f32>, f32, usize), AutodiffError>
where
    F: FnMut(&[Tensor<f32>]) -> Result<(f32, Vec<Tensor<f32>>), AutodiffError>,
{
    const C1: f32 = 1e-4;
    const C2: f32 = 0.9;
    const TOLERANCE_CHANGE: f32 = 1e-9;

    let d_norm = abs_max(d);
    let mut t = t0;

    let (mut f_new, mut g_new) = directional_evaluate(closure, slot_shapes, x, t, d)?;
    let mut ls_func_evals = 1usize;
    let mut gtd_new = dot_f64(&g_new, d);

    let mut t_prev = 0f32;
    let mut f_prev = f0;
    let mut g_prev = g0.to_vec();
    let mut gtd_prev = gtd0;

    let mut done = false;
    let mut ls_iter = 0usize;

    // bracket は要素数 1（`done` 判定で確定した単一点）または 2
    // （区間）で表現する。PyTorch のリスト `bracket`/`bracket_f`/
    // `bracket_g`/`bracket_gtd` を固定長 2 の配列 + 有効長で表現する。
    let mut bracket_t = [0f32; 2];
    let mut bracket_f = [0f32; 2];
    let mut bracket_g: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    let mut bracket_gtd = [0f32; 2];
    let mut bracket_len = 0usize;

    while ls_iter < max_ls {
        if f_new > f0 + C1 * t * gtd0 || (ls_iter > 1 && f_new >= f_prev) {
            bracket_t = [t_prev, t];
            bracket_f = [f_prev, f_new];
            bracket_g = [g_prev.clone(), g_new.clone()];
            bracket_gtd = [gtd_prev, gtd_new];
            bracket_len = 2;
            break;
        }
        if gtd_new.abs() <= -C2 * gtd0 {
            bracket_t[0] = t;
            bracket_f[0] = f_new;
            bracket_g[0] = g_new.clone();
            bracket_len = 1;
            done = true;
            break;
        }
        if gtd_new >= 0.0 {
            bracket_t = [t_prev, t];
            bracket_f = [f_prev, f_new];
            bracket_g = [g_prev.clone(), g_new.clone()];
            bracket_gtd = [gtd_prev, gtd_new];
            bracket_len = 2;
            break;
        }

        let min_step = t + 0.01 * (t - t_prev);
        let max_step = t * 10.0;
        let tmp = t;
        t = cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            Some((min_step, max_step)),
        );

        t_prev = tmp;
        f_prev = f_new;
        g_prev = g_new.clone();
        gtd_prev = gtd_new;
        let (fe, ge) = directional_evaluate(closure, slot_shapes, x, t, d)?;
        f_new = fe;
        g_new = ge;
        ls_func_evals += 1;
        gtd_new = dot_f64(&g_new, d);
        ls_iter += 1;
    }

    if ls_iter == max_ls {
        bracket_t = [0.0, t];
        bracket_f = [f0, f_new];
        bracket_g = [g0.to_vec(), g_new.clone()];
        // PyTorch はこの分岐で bracket_gtd を設定しないため未使用の
        // まま残る（zoom フェーズはこの直後 `abs(bracket[1] -
        // bracket[0]) * d_norm < tolerance_change` で即 break しうる
        // 想定。読まれる場合に不定値を使わないよう安全な初期値を置く）。
        bracket_gtd = [gtd0, gtd_new];
        bracket_len = 2;
    }

    if bracket_len == 1 {
        // 単一点で Wolfe 条件成立（`done == true`）。zoom フェーズは
        // 実行せずそのまま返す。
        return Ok((
            bracket_f[0],
            bracket_g[0].clone(),
            bracket_t[0],
            ls_func_evals,
        ));
    }

    let (mut low_pos, mut high_pos) = if bracket_f[0] <= bracket_f[1] {
        (0usize, 1usize)
    } else {
        (1usize, 0usize)
    };
    let mut insuf_progress = false;

    while !done && ls_iter < max_ls {
        if (bracket_t[1] - bracket_t[0]).abs() * d_norm < TOLERANCE_CHANGE {
            break;
        }

        t = cubic_interpolate(
            bracket_t[0],
            bracket_f[0],
            bracket_gtd[0],
            bracket_t[1],
            bracket_f[1],
            bracket_gtd[1],
            None,
        );

        let bmax = bracket_t[0].max(bracket_t[1]);
        let bmin = bracket_t[0].min(bracket_t[1]);
        let eps = 0.1 * (bmax - bmin);
        if (bmax - t).min(t - bmin) < eps {
            if insuf_progress || t >= bmax || t <= bmin {
                if (t - bmax).abs() < (t - bmin).abs() {
                    t = bmax - eps;
                } else {
                    t = bmin + eps;
                }
                insuf_progress = false;
            } else {
                insuf_progress = true;
            }
        } else {
            insuf_progress = false;
        }

        let (fe, ge) = directional_evaluate(closure, slot_shapes, x, t, d)?;
        f_new = fe;
        g_new = ge;
        ls_func_evals += 1;
        gtd_new = dot_f64(&g_new, d);
        ls_iter += 1;

        if f_new > f0 + C1 * t * gtd0 || f_new >= bracket_f[low_pos] {
            bracket_t[high_pos] = t;
            bracket_f[high_pos] = f_new;
            bracket_g[high_pos] = g_new.clone();
            bracket_gtd[high_pos] = gtd_new;
            if bracket_f[0] <= bracket_f[1] {
                low_pos = 0;
                high_pos = 1;
            } else {
                low_pos = 1;
                high_pos = 0;
            }
        } else {
            if gtd_new.abs() <= -C2 * gtd0 {
                done = true;
            } else if gtd_new * (bracket_t[high_pos] - bracket_t[low_pos]) >= 0.0 {
                bracket_t[high_pos] = bracket_t[low_pos];
                bracket_f[high_pos] = bracket_f[low_pos];
                bracket_g[high_pos] = bracket_g[low_pos].clone();
                bracket_gtd[high_pos] = bracket_gtd[low_pos];
            }
            bracket_t[low_pos] = t;
            bracket_f[low_pos] = f_new;
            bracket_g[low_pos] = g_new.clone();
            bracket_gtd[low_pos] = gtd_new;
        }
    }

    Ok((
        bracket_f[low_pos],
        bracket_g[low_pos].clone(),
        bracket_t[low_pos],
        ls_func_evals,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
    }

    #[test]
    fn rejects_negative_lr() {
        let cfg = LbfgsConfig {
            lr: -1.0,
            ..LbfgsConfig::default()
        };
        assert!(matches!(
            Lbfgs::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_nan_lr() {
        let cfg = LbfgsConfig {
            lr: f32::NAN,
            ..LbfgsConfig::default()
        };
        assert!(matches!(
            Lbfgs::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_zero_max_iter() {
        let cfg = LbfgsConfig {
            max_iter: 0,
            ..LbfgsConfig::default()
        };
        assert!(matches!(
            Lbfgs::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_zero_history_size() {
        let cfg = LbfgsConfig {
            history_size: 0,
            ..LbfgsConfig::default()
        };
        assert!(matches!(
            Lbfgs::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_zero_line_search_steps() {
        let cfg = LbfgsConfig {
            line_search_steps: 0,
            ..LbfgsConfig::default()
        };
        assert!(matches!(
            Lbfgs::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_negative_tolerance() {
        let cfg = LbfgsConfig {
            tolerance_grad: -1.0,
            ..LbfgsConfig::default()
        };
        assert!(matches!(
            Lbfgs::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
        let cfg = LbfgsConfig {
            tolerance_change: -1.0,
            ..LbfgsConfig::default()
        };
        assert!(matches!(
            Lbfgs::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_empty_params_without_calling_closure() {
        let mut opt = Lbfgs::new(LbfgsConfig::default()).unwrap();
        let mut called = false;
        let result = opt.step_closure(&[], |_p| {
            called = true;
            (0.0, vec![])
        });
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
        assert!(!called, "空パラメータ列では closure を呼んではならない");
    }

    #[test]
    fn rejects_slot_count_change_after_first_step() {
        let mut opt = Lbfgs::new(LbfgsConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        opt.step_closure(std::slice::from_ref(&param), |p| {
            let g = t(vec![2.0 * p[0].get(&[0]).unwrap()], &[1]);
            (p[0].get(&[0]).unwrap().powi(2), vec![g])
        })
        .unwrap();

        let result = opt.step_closure(&[], |_p| (0.0, vec![]));
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
    }

    #[test]
    fn rejects_slot_shape_change_after_first_step() {
        let mut opt = Lbfgs::new(LbfgsConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        opt.step_closure(&[param], |p| {
            let v = p[0].get(&[0]).unwrap();
            (v * v, vec![t(vec![2.0 * v], &[1])])
        })
        .unwrap();

        let param2 = t(vec![1.0, 2.0], &[2]);
        let result = opt.step_closure(&[param2], |p| {
            let v0 = p[0].get(&[0]).unwrap();
            let v1 = p[0].get(&[1]).unwrap();
            (v0 * v0 + v1 * v1, vec![t(vec![2.0 * v0, 2.0 * v1], &[2])])
        });
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    #[test]
    fn rejects_closure_gradient_count_mismatch() {
        let mut opt = Lbfgs::new(LbfgsConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let result = opt.step_closure(&[param], |_p| (1.0, vec![]));
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
    }

    #[test]
    fn rejects_closure_gradient_shape_mismatch() {
        let mut opt = Lbfgs::new(LbfgsConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let result = opt.step_closure(&[param], |_p| (1.0, vec![t(vec![1.0, 2.0], &[2])]));
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    #[test]
    fn rejects_non_finite_loss() {
        let mut opt = Lbfgs::new(LbfgsConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let result = opt.step_closure(&[param], |_p| (f32::NAN, vec![t(vec![1.0], &[1])]));
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
    }

    #[test]
    fn rejects_non_finite_gradient() {
        let mut opt = Lbfgs::new(LbfgsConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let result = opt.step_closure(&[param], |_p| (1.0, vec![t(vec![f32::INFINITY], &[1])]));
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
    }

    /// closure が `Err` を返した直後の状態が呼び出し前と一致すること
    /// を、その後の正常な step が「エラーが起きなかった場合の対応する
    /// step」と bit 一致する結果を返すことで間接的に確認する
    /// （`AdamW` の `state_not_mutated_after_failed_step` と同型）。
    #[test]
    fn state_not_mutated_after_failed_closure() {
        fn quadratic_closure(p: &[Tensor<f32>]) -> Result<(f32, Vec<Tensor<f32>>), AutodiffError> {
            let v = p[0].get(&[0]).unwrap();
            Ok((v * v, vec![t(vec![2.0 * v], &[1])]))
        }

        let cfg = LbfgsConfig {
            line_search: LbfgsLineSearch::StrongWolfe,
            ..LbfgsConfig::default()
        };
        let mut opt = Lbfgs::new(cfg).unwrap();
        let param0 = t(vec![3.0], &[1]);
        let after_step1 = opt.try_step_closure(&[param0], quadratic_closure).unwrap();
        let n_iter_before = opt.n_iter();
        let func_evals_before = opt.func_evals();

        // 2 回目呼び出しで closure が Err を返す。
        let failing_result = opt.try_step_closure(&after_step1, |_p| {
            Err(AutodiffError::Backward("boom".into()))
        });
        assert!(matches!(failing_result, Err(AutodiffError::Backward(_))));
        assert_eq!(opt.n_iter(), n_iter_before);
        assert_eq!(opt.func_evals(), func_evals_before);

        // 参照 optimizer（同一初期状態）で「失敗しなかった場合の
        // 2 回目 step」を実行し、失敗後の 3 回目 step と一致すること
        // を確認する。
        let mut opt_ref = Lbfgs::new(LbfgsConfig {
            line_search: LbfgsLineSearch::StrongWolfe,
            ..LbfgsConfig::default()
        })
        .unwrap();
        let param0_ref = t(vec![3.0], &[1]);
        let after_ref_step1 = opt_ref
            .try_step_closure(&[param0_ref], quadratic_closure)
            .unwrap();
        let after_ref_step2 = opt_ref
            .try_step_closure(&after_ref_step1, quadratic_closure)
            .unwrap();

        let after_step2 = opt
            .try_step_closure(&after_step1, quadratic_closure)
            .unwrap();

        assert_eq!(
            after_step2[0].get(&[0]).unwrap(),
            after_ref_step2[0].get(&[0]).unwrap(),
            "failed closure の直後の状態が破損している"
        );
    }

    #[test]
    fn set_lr_does_not_reset_state() {
        fn quadratic_closure(p: &[Tensor<f32>]) -> (f32, Vec<Tensor<f32>>) {
            let v = p[0].get(&[0]).unwrap();
            (v * v, vec![t(vec![2.0 * v], &[1])])
        }

        let mut opt = Lbfgs::new(LbfgsConfig::default()).unwrap();
        let param = t(vec![3.0], &[1]);
        opt.step_closure(&[param], quadratic_closure).unwrap();
        let n_iter_before = opt.n_iter();
        opt.set_lr(0.5).unwrap();
        assert_eq!(opt.n_iter(), n_iter_before);
        assert_eq!(opt.config().lr, 0.5);
    }

    #[test]
    fn set_lr_rejects_negative() {
        let mut opt = Lbfgs::new(LbfgsConfig::default()).unwrap();
        assert!(matches!(
            opt.set_lr(-1.0),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    /// `tolerance_grad_early_return` ケースの単体版: 初回勾配が閾値
    /// 以下ならパラメータ不変・`func_evals == 1`・`n_iter == 0`。
    #[test]
    fn tolerance_grad_triggers_immediate_return() {
        let cfg = LbfgsConfig {
            tolerance_grad: 100.0,
            ..LbfgsConfig::default()
        };
        let mut opt = Lbfgs::new(cfg).unwrap();
        let param = t(vec![3.0], &[1]);
        let out = opt
            .step_closure(std::slice::from_ref(&param), |p| {
                let v = p[0].get(&[0]).unwrap();
                (v * v, vec![t(vec![2.0 * v], &[1])])
            })
            .unwrap();
        assert_eq!(out[0].get(&[0]).unwrap(), param.get(&[0]).unwrap());
        assert_eq!(opt.func_evals(), 1);
        assert_eq!(opt.n_iter(), 0);
        assert_eq!(opt.last_loss(), Some(9.0));
    }

    /// `line_search_steps` が実際に strong Wolfe の試行回数を制限する
    /// こと（`max_eval` 側の項が十分大きい設定にして本キャップのみが
    /// 効くようにする）。
    #[test]
    fn line_search_steps_cap_limits_trials() {
        // 振動的な非凸関数で line search を長引かせ、`line_search_steps`
        // が 1 のときは 1 回の直線探索評価で打ち切られることを確認する。
        fn wavy_closure(p: &[Tensor<f32>]) -> (f32, Vec<Tensor<f32>>) {
            let x = p[0].get(&[0]).unwrap();
            let loss = x.sin() + 0.1 * x * x;
            let grad = x.cos() + 0.2 * x;
            (loss, vec![t(vec![grad], &[1])])
        }

        let mut call_count = std::cell::Cell::new(0usize);
        let cfg = LbfgsConfig {
            line_search: LbfgsLineSearch::StrongWolfe,
            line_search_steps: 1,
            max_iter: 1,
            max_eval: Some(1000),
            ..LbfgsConfig::default()
        };
        let mut opt = Lbfgs::new(cfg).unwrap();
        let param = t(vec![5.0], &[1]);
        opt.step_closure(&[param], |p| {
            *call_count.get_mut() += 1;
            wavy_closure(p)
        })
        .unwrap();
        // 初回評価 1 回 + line search 最大 1 回（`line_search_steps` の
        // キャップ）で高々 2 回。
        assert!(
            *call_count.get_mut() <= 2,
            "line_search_steps キャップが効いていない: calls={}",
            call_count.get_mut()
        );
    }
}
