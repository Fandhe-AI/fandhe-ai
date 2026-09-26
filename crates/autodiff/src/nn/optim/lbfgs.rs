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
//! - **`max_eval` の解決**: `max_eval: None` の場合 `max_iter * 5 / 4`
//!   （PyTorch と同じ整数除算）へ解決するが、`max_iter` に極端に大きい
//!   値（例 `usize::MAX`）が渡されると乗算が overflow しうるため
//!   `checked_mul` で検出し、overflow する場合は `Lbfgs::new` が
//!   `InvalidArgument` を返す（本番経路 panic 禁止・
//!   `.claude/rules/coding-rust.md`。イシュー #2197 レビュー是正・
//!   discussion_r4110471294）。
//! - **line search 予算・parity の finite 性検証**: [`LbfgsConfig::line_search_steps`]
//!   の doc に記載のとおり、strong Wolfe の 1 回あたり closure 呼び出し
//!   総数は初回評価を含めて `min(line_search_steps, max_eval -
//!   current_evals)` を超えない。また、`LbfgsLineSearch::None`
//!   （固定ステップ）・`StrongWolfe` いずれの経路でも、`x += t·d` で
//!   更新した直後のパラメータに非有限値（overflow 由来の `inf`/`NaN`
//!   等）が含まれる場合は closure を呼ぶ・呼ばないに関わらず
//!   `InvalidArgument` を返し、`self` の状態は変更しない（イシュー
//!   #2197 レビュー是正・discussion_r4110451838/r4110471293）。
//! - **入力・内部縮約の finite 性検証の拡張**（PR #2295 レビュー是正・
//!   discussion_r4110471294/r4110528423 系）: 上記の「更新後の `x`」
//!   検査に加え、次の各点も同じ `ensure_finite_slice`／`dot_f64` の
//!   fail-closed 契約で検査する: (1) 呼び出し元が渡す入力 `params`
//!   自体（closure 呼び出し前・shape 検証の直後）、(2) `f64`
//!   アキュムレータを `f32` へ丸める全縮約 `dot_f64`（`ys`/`yy`/
//!   two-loop の `al_i`/`be_i`/`gtd`/`gtd_new` の全呼び出し箇所。丸め
//!   自体が `f32::MAX` を超えて `inf`/`-inf` になりうるため）、
//!   (3) `h_diag = ys / yy` の除算結果、(4) 二段ループ recursion 後の
//!   探索方向 `d`、(5) 初回ステップ幅の分母 `sum_abs =
//!   Σ|flat_grad|` を `f32` へキャストした結果（overflow して `inf`
//!   になると `1.0/inf == 0.0` で `t = 0` の no-op ステップとして
//!   異常が握り潰され、そのまま成功終了してしまうマスキングを防ぐ）。
//!   いずれも検出時は `self` の状態を変更せず `InvalidArgument` を
//!   返す。**この拡張が変えるのは「異常系（従来は非有限値を検出
//!   できず握り潰していた入力）の扱いのみ」であり、有限な入力に
//!   対する `dot_f64` の `f32` 丸め自体・正常系の出力は変えない**
//!   （既存の PyTorch 参照テストの期待値・許容誤差は不変）。
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
    /// （実装計画 §2.3）。**この予算は `strong_wolfe` 内の初回評価
    /// （PyTorch の逐語移植では無条件に 1 回発生する）も含む**（イシュー
    /// #2197 レビュー是正・discussion_r4110471290/r4110451852）。つまり
    /// 1 回の `step`/`step_closure` 呼び出しにおける line search 中の
    /// closure 呼び出し総数は `min(line_search_steps, max_eval -
    /// current_evals)` を超えない。この実効値が 0 の場合は初回評価も
    /// 行わず closure を呼ばない（`max_eval: Some(1)` 等で予算が
    /// 尽きている場合の fail-closed 契約）。
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
    /// `max_eval` を解決する。`None` の場合 `max_iter * 5 / 4`
    /// （PyTorch と同じ整数除算）を `checked_mul` 経由で計算し、
    /// overflow する場合は `None` を返す（`Lbfgs::new` が検出して
    /// `InvalidArgument` にする。呼び出し元は本関数を検証目的でのみ
    /// 使い、検証済み値は `Lbfgs::resolved_max_eval` フィールドに
    /// キャッシュして再計算しない）。
    fn resolved_max_eval(&self) -> Option<usize> {
        match self.max_eval {
            Some(v) => Some(v),
            None => self.max_iter.checked_mul(5).map(|v| v / 4),
        }
    }
}

/// L-BFGS optimizer 本体。[`LbfgsConfig`] と、フラット化ベクトル全体に
/// 対して定義される大域状態（`step()` 呼び出しをまたいで持ち越す）を
/// 保持する。
pub struct Lbfgs {
    config: LbfgsConfig,
    /// `config.resolved_max_eval()` を `new()` で 1 度だけ検証・確定した
    /// 値（`checked_mul` の overflow 検出込み）。`config.max_iter`/
    /// `config.max_eval` は構築後に変更する手段がない（`set_lr` は
    /// `lr` のみ書き換える）ため、以後の呼び出しは本フィールドを
    /// 再計算なしで安全に使える（overflow 再検査・`unwrap`/`expect`
    /// 不要。イシュー #2197 レビュー是正・discussion_r4110471294）。
    resolved_max_eval: usize,
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
        let resolved_max_eval = config.resolved_max_eval().ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "Lbfgs::new: max_iter * 5 overflows usize (max_iter={}); \
                 pass config.max_eval explicitly to avoid the derived \
                 `max_iter * 5 / 4` computation",
                config.max_iter
            ))
        })?;
        if resolved_max_eval < 1 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lbfgs::new: max_eval must resolve to >= 1, got {resolved_max_eval}"
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
            resolved_max_eval,
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

        // 入力 `params` 自体の非有限値を closure 呼び出し前に検査する
        // （codex-review 指摘・PR #2295 discussion_r4110451838 系の
        // 横展開: 呼び出し元が既に非有限な値〈NaN/inf〉を渡した場合、
        // これまでは無検査で closure に渡っていた）。ここで得た
        // フラット化済み `x` はそのまま後続の作業コピー初期値として
        // 再利用し、`flatten_tensors(params)` の再計算を避ける。
        let mut x = flatten_tensors(params);
        ensure_finite_slice("Lbfgs::try_step_closure: params", &x)?;

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

        // `x`（現在の実パラメータ値のフラット化。line search・固定
        // ステップの各反復末尾で実際に更新される。PyTorch
        // `self._add_grad(t, d)` に相当）は関数冒頭で検証済みの値を
        // そのまま使う（上記 `ensure_finite_slice` 参照）。

        let max_eval = self.resolved_max_eval;
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
                let ys = dot_f64("Lbfgs::try_step_closure: y·s (ys)", &y, &s)?;
                if ys > 1e-10 {
                    if old_dirs.len() == self.config.history_size {
                        old_dirs.pop_front();
                        old_stps.pop_front();
                        ro.pop_front();
                    }
                    let yy = dot_f64("Lbfgs::try_step_closure: y·y (yy)", &y, &y)?;
                    old_dirs.push_back(y);
                    old_stps.push_back(s);
                    ro.push_back(1.0 / ys);
                    h_diag = ys / yy;
                    // `ys`/`yy` は dot_f64 の f32 overflow 検査を通過済みだが
                    // 除算 `ys/yy` 自体が非有限になりうる（`yy` が極小の
                    // 場合等）ため個別に検査する（P2 是正の横展開。codex-review
                    // 指摘・PR #2295 discussion_r4110471294 系）。
                    ensure_finite_slice(
                        "Lbfgs::try_step_closure: h_diag (ys/yy)",
                        std::slice::from_ref(&h_diag),
                    )?;
                }

                let num_old = old_dirs.len();
                let mut al = vec![0f32; num_old];
                let mut q: Vec<f32> = flat_grad.iter().map(|&g| -g).collect();
                for i in (0..num_old).rev() {
                    let al_i = dot_f64("Lbfgs::try_step_closure: two-loop al_i", &old_stps[i], &q)?
                        * ro[i];
                    al[i] = al_i;
                    for k in 0..q.len() {
                        q[k] = f32::mul_add(-al_i, old_dirs[i][k], q[k]);
                    }
                }
                let mut r: Vec<f32> = q.iter().map(|&qi| qi * h_diag).collect();
                for i in 0..num_old {
                    let be_i = dot_f64("Lbfgs::try_step_closure: two-loop be_i", &old_dirs[i], &r)?
                        * ro[i];
                    let coeff = al[i] - be_i;
                    for k in 0..r.len() {
                        r[k] = f32::mul_add(coeff, old_stps[i][k], r[k]);
                    }
                }
                d = r;
                // 二段ループ（two-loop recursion）で求めた探索方向 `d` を
                // 検査する（`al`/`be`/`h_diag` いずれかの経路経由で非有限が
                // 混入していないことの最終確認。P2 是正の横展開）。
                ensure_finite_slice("Lbfgs::try_step_closure: search direction d", &d)?;
            }

            prev_flat_grad = Some(flat_grad.clone());
            prev_loss = loss;

            t = if n_iter_global == 1 {
                let sum_abs = abs_sum_f64(&flat_grad) as f32;
                // `Σ|flat_grad|` が f32 overflow して `inf` になる場合、
                // 検査せずに進むと `1.0/inf == 0.0` により `t = 0` の
                // no-op ステップとして異常を隠蔽したまま成功終了して
                // しまう（マスキング防止。P2 是正の横展開・codex-review
                // 指摘・PR #2295 discussion_r4110471294 系）。
                ensure_finite_slice(
                    "Lbfgs::try_step_closure: sum_abs(flat_grad) for initial step size",
                    std::slice::from_ref(&sum_abs),
                )?;
                1.0f32.min(1.0 / sum_abs) * self.config.lr
            } else {
                self.config.lr
            };

            let gtd = dot_f64("Lbfgs::try_step_closure: g·d (gtd)", &flat_grad, &d)?;
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
                    ensure_finite_slice("Lbfgs::try_step_closure: x += t*d (strong Wolfe)", &x)?;
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
                    // 固定ステップ更新直後に非有限値混入を検査する
                    // （closure 再評価の有無に関わらず。イシュー #2197
                    // レビュー是正・discussion_r4110451838/r4110471293:
                    // 最終反復〈`n_iter_local == max_iter`〉では closure
                    // を再評価せず `x` をそのまま返していたため、有限
                    // 入力でも overflow で `inf` な `x` を成功結果として
                    // 返し得た）。異常時は `self` の状態を変更せず
                    // `Err` を返す（本関数はここまでローカル作業コピー
                    // のみを変更しており `self.*` へのコミットは関数末尾
                    // でのみ行うため、ここで早期 return しても状態不変
                    // 契約を満たす）。
                    ensure_finite_slice("Lbfgs::try_step_closure: x += t*d (fixed step)", &x)?;
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
/// へ丸める（縮約方針。冒頭 doc 参照）。`f64` の縮約自体は事実上
/// overflow しないが、最後の `f32` への丸めは overflow しうる（例:
/// 巨大な勾配要素の内積 `g·d` が `f32::MAX` を超え `-inf`/`inf` に
/// なる）。呼び出し元はいずれも縮約結果をスカラーとして以後の
/// 判定・演算に使うため、ここで検査せずに非有限値を通すと
/// `NaN > 閾値` は常に `false` になる、あるいは `inf` が後続の
/// 演算へ伝播するといった形で異常が握り潰される（P2 是正・
/// codex-review 指摘・PR #2295 discussion_r4110471294 系）。
fn dot_f64(label: &str, a: &[f32], b: &[f32]) -> Result<f32, AutodiffError> {
    let mut acc = 0f64;
    for i in 0..a.len() {
        acc = f64::from(a[i]).mul_add(f64::from(b[i]), acc);
    }
    let v = acc as f32;
    if !v.is_finite() {
        return Err(AutodiffError::InvalidArgument(format!(
            "Lbfgs: {label} overflowed to a non-finite f32 value when \
             rounding the f64 accumulator (acc={acc}); state left unchanged"
        )));
    }
    Ok(v)
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

/// フラット化ベクトルに非有限値（overflow 由来の `inf`／`NaN`）が
/// 含まれないか検査する汎用ヘルパー（`.claude/rules/security.md`
/// A03。イシュー #2197 レビュー是正・
/// discussion_r4110451838/r4110471293・discussion_r4110528423。PR #2295
/// レビュー是正で `ensure_finite_params` から汎用化し検査対象を
/// 拡張: 入力 `params`・`x += t·d` 更新後の `x`・二段ループ後の探索
/// 方向 `d`・`h_diag`・初回ステップ計算の `sum_abs` 等の単一要素
/// スカラーにも同じヘルパーを使う）。呼び出し元
/// [`Lbfgs::try_step_closure`]・[`directional_evaluate`] はいずれも
/// ローカル作業コピー上で反復するため、ここで `Err` を返しても
/// `self` の状態は変更されない。`label` はエラーメッセージに検査
/// 対象を記録し、複数の検査点を区別できるようにする。
fn ensure_finite_slice(label: &str, x: &[f32]) -> Result<(), AutodiffError> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(AutodiffError::InvalidArgument(format!(
            "Lbfgs: {label} contains a non-finite value (likely overflow); \
             state left unchanged"
        )));
    }
    Ok(())
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
    // `x + t·d` の overflow により生じうる非有限値を closure へ渡す前に検査する
    // （codex-review 指摘・PR #2295 discussion_r4110528423。有限な初期パラメータ・
    // 勾配・学習率でも t が大きい場合は overflow しうるため、closure 呼び出し前に
    // 検証する必要がある）。
    ensure_finite_slice("Lbfgs::directional_evaluate: trial = x + t*d", &trial)?;
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

    // `max_ls` は本関数内で行う closure 呼び出し総数の予算（呼び出し元
    // `min(line_search_steps, max_eval - current_evals)`。以下の初回
    // 評価も含む。イシュー #2197 レビュー是正・
    // discussion_r4110471290/r4110451852）。PyTorch の逐語移植は初回
    // 評価を無条件に行うが、それでは予算 0（`max_eval` 到達済み）でも
    // closure を呼んでしまい、`max_ls == 1` でも「初回 + bracket/zoom
    // ループ 1 回」の計 2 回評価してしまう。予算を超えないよう、初回
    // 評価も 1 回分として `max_ls` から差し引く。
    if max_ls == 0 {
        // 予算 0: closure を 1 回も呼ばず、ステップ未適用（`t = 0`）
        // として現在値をそのまま返す。呼び出し元はこの `t = 0` を
        // `x += 0·d` として適用するため実質的に no-op であり、直後の
        // `current_evals >= max_eval` チェックで安全に反復を終える。
        return Ok((f0, g0.to_vec(), 0.0, 0));
    }
    // 初回評価の 1 回分を差し引いた、bracket/zoom ループ側の残り予算。
    let max_extra_ls = max_ls - 1;

    let d_norm = abs_max(d);
    let mut t = t0;

    let (mut f_new, mut g_new) = directional_evaluate(closure, slot_shapes, x, t, d)?;
    let mut ls_func_evals = 1usize;
    let mut gtd_new = dot_f64("Lbfgs::strong_wolfe: g_new·d (gtd_new, initial)", &g_new, d)?;

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

    while ls_iter < max_extra_ls {
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
        gtd_new = dot_f64(
            "Lbfgs::strong_wolfe: g_new·d (gtd_new, bracket search)",
            &g_new,
            d,
        )?;
        ls_iter += 1;
    }

    if ls_iter == max_extra_ls {
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

    while !done && ls_iter < max_extra_ls {
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
        gtd_new = dot_f64("Lbfgs::strong_wolfe: g_new·d (gtd_new, zoom)", &g_new, d)?;
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

    /// `max_iter` に `usize::MAX`・`max_eval: None` を渡すと
    /// `max_iter * 5` の内部計算が overflow するため `Lbfgs::new` が
    /// `InvalidArgument` を返すこと（`checked_mul` による fail-closed
    /// 検出。イシュー #2197 レビュー是正・discussion_r4110471294）。
    #[test]
    fn rejects_max_iter_overflow_when_max_eval_unset() {
        let cfg = LbfgsConfig {
            max_iter: usize::MAX,
            max_eval: None,
            ..LbfgsConfig::default()
        };
        assert!(matches!(
            Lbfgs::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    /// `max_eval` を明示指定すれば `max_iter * 5` の派生計算自体を
    /// 経由しないため、`max_iter` が極端に大きくても構築できること
    /// （overflow 検出が派生パス限定であることの確認）。
    #[test]
    fn accepts_max_iter_overflow_prone_value_when_max_eval_set() {
        let cfg = LbfgsConfig {
            max_iter: usize::MAX,
            max_eval: Some(5),
            ..LbfgsConfig::default()
        };
        assert!(Lbfgs::new(cfg).is_ok());
    }

    /// `max_eval: Some(1)` の場合、strong Wolfe line search 側の予算が
    /// 0 になり closure を追加評価しない（初回評価の 1 回のみで
    /// 終える）こと（イシュー #2197 レビュー是正・
    /// discussion_r4110471290/r4110451852）。
    #[test]
    fn strong_wolfe_respects_max_eval_one() {
        let cfg = LbfgsConfig {
            line_search: LbfgsLineSearch::StrongWolfe,
            max_eval: Some(1),
            ..LbfgsConfig::default()
        };
        let mut opt = Lbfgs::new(cfg).unwrap();
        let param = t(vec![3.0], &[1]);
        let call_count = std::cell::Cell::new(0usize);
        let out = opt
            .step_closure(&[param], |p| {
                call_count.set(call_count.get() + 1);
                let v = p[0].get(&[0]).unwrap();
                (v * v, vec![t(vec![2.0 * v], &[1])])
            })
            .unwrap();
        assert_eq!(call_count.get(), 1, "max_eval: Some(1) は初回評価のみ許す");
        assert_eq!(opt.func_evals(), 1);
        assert_eq!(
            out[0].get(&[0]).unwrap(),
            3.0,
            "予算 0 で closure は呼ばれず更新も適用されない"
        );
    }

    /// `LbfgsLineSearch::None`（固定ステップ）で `x += t*d` が overflow
    /// して `inf` になる場合、closure 再評価の有無に関わらず `Err` を
    /// 返し `self` の状態（`n_iter`/`func_evals`）が変化しないこと
    /// （イシュー #2197 レビュー是正・
    /// discussion_r4110451838/r4110471293）。`max_iter: 1` により
    /// 最終反復（closure 再評価を省略する分岐）でのみ検出されることを
    /// 確認する。
    #[test]
    fn rejects_non_finite_x_after_fixed_step_update() {
        let cfg = LbfgsConfig {
            line_search: LbfgsLineSearch::None,
            lr: 3e38,
            max_iter: 1,
            ..LbfgsConfig::default()
        };
        let mut opt = Lbfgs::new(cfg).unwrap();
        let param = t(vec![-3e38], &[1]);
        // 勾配を定数 1.0 にして `d = -1.0`・`t = lr = 3e38` とし、
        // `x = fma(3e38, -1.0, -3e38) = -inf` を発生させる。
        let result = opt.step_closure(&[param], |_p| (0.0, vec![t(vec![1.0], &[1])]));
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
        assert_eq!(opt.n_iter(), 0, "エラー時は n_iter が呼び出し前のまま");
        assert_eq!(
            opt.func_evals(),
            0,
            "エラー時は func_evals が呼び出し前のまま"
        );
    }

    #[test]
    fn directional_evaluate_rejects_non_finite_trial_without_calling_closure() {
        // `trial = x + t·d` の overflow で非有限値が生じる場合、
        // `unflatten_tensors`／`closure` 呼び出し前に `ensure_finite_slice`
        // で弾くことを直接検証する（codex-review 指摘・PR #2295
        // discussion_r4110528423）。`directional_evaluate` は同一モジュール
        // 内 private のためテストから直接呼び出せる。
        let mut called = false;
        let x = [f32::MAX];
        let d = [f32::MAX];
        let slot_shapes = vec![vec![1]];
        let result = directional_evaluate(
            &mut |_p: &[Tensor<f32>]| -> Result<(f32, Vec<Tensor<f32>>), AutodiffError> {
                called = true;
                Ok((0.0, vec![t(vec![1.0], &[1])]))
            },
            &slot_shapes,
            &x,
            2.0,
            &d,
        );
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
        assert!(!called, "非有限な trial は closure 呼び出し前に弾かれる");
    }

    /// P1 是正: 呼び出し元が渡す入力 `params` 自体に非有限値（NaN）が
    /// 含まれる場合、closure を 1 度も呼ばず `InvalidArgument` を返す
    /// こと（codex-review 指摘・PR #2295 discussion。lbfgs.rs:345 付近）。
    #[test]
    fn rejects_non_finite_params_without_calling_closure() {
        let mut opt = Lbfgs::new(LbfgsConfig::default()).unwrap();
        let mut called = false;
        let param = t(vec![f32::NAN], &[1]);
        let result = opt.step_closure(&[param], |_p| {
            called = true;
            (0.0, vec![t(vec![1.0], &[1])])
        });
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
        assert!(!called, "非有限な params では closure を呼んではならない");
        assert_eq!(opt.n_iter(), 0);
        assert_eq!(opt.func_evals(), 0);
    }

    /// P2 是正: `gtd = dot_f64(&flat_grad, &d)` の `f64 → f32` 丸めが
    /// overflow して `-inf` になる場合を検出すること（lbfgs.rs:468
    /// 付近）。初回反復は `d = -flat_grad` なので
    /// `gtd = -Σ flat_grad_i²`。勾配要素を `3e19` にすると
    /// `Σ flat_grad_i² = 9e38` は `f64` としては有限だが、符号反転後
    /// `f32` へ丸めると `f32::MAX`（約 `3.4e38`）を超え `-inf` になる。
    /// 是正前は `gtd.is_finite()` を検査しないため `NaN`/`inf` でも
    /// `gtd > -tolerance_change` が `false` のまま line search へ進み
    /// うる（`NaN` の場合は比較が常に `false` になり `break` が効かない
    /// マスキング）。
    #[test]
    fn rejects_gtd_overflow_from_dot_f64_rounding() {
        let mut opt = Lbfgs::new(LbfgsConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let result = opt.step_closure(&[param], |_p| (0.0, vec![t(vec![3e19], &[1])]));
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
        assert_eq!(opt.n_iter(), 0, "エラー時は n_iter が呼び出し前のまま");
        assert_eq!(
            opt.func_evals(),
            0,
            "エラー時は func_evals が呼び出し前のまま（初回 closure 評価も \
             ローカル作業コピー確定前のため未コミット）"
        );
    }

    /// `dot_f64` 単体で `f64` アキュムレータの `f32` への丸めが
    /// overflow する場合に `Err` を返すこと（P2 是正の縮約ヘルパー
    /// 本体の直接検証。private のため同一モジュール内テストから
    /// 呼び出す）。
    #[test]
    fn dot_f64_rejects_f32_rounding_overflow() {
        let a = [3e19f32];
        let b = [3e19f32];
        let result = dot_f64("test: a·b", &a, &b);
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
    }

    #[test]
    fn dot_f64_accepts_finite_result() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 32（`f64` 縮約→`f32` 丸めが有限入力の正常系
        // 出力を変えないことの確認。既存の縮約方針・丸め自体は不変）。
        assert_eq!(dot_f64("test: a·b", &a, &b).unwrap(), 32.0);
    }

    /// P2 是正の横展開: 初回ステップ幅の分母 `sum_abs =
    /// Σ|flat_grad|` を `f32` へキャストした結果が overflow する場合を
    /// 検出すること。単一要素の勾配自体は `f32::MAX` 未満に収める
    /// 必要があるため、`f64` 縮約後の合計が `f32::MAX` を超えるよう
    /// 複数要素（各 `2e38`）の和で構成する（`h_diag`/`d` の非有限
    /// ケースは、正常系での history 更新〈`ys > 1e-10`〉を経由しつつ
    /// 二段ループの al/be/h_diag 経路のみを overflow させる入力の
    /// 構成が複雑なため見送る。dot_f64／sum_abs の直接検査で縮約
    /// overflow の検出経路自体は担保できている）。
    #[test]
    fn rejects_sum_abs_overflow_for_initial_step_size() {
        let mut opt = Lbfgs::new(LbfgsConfig::default()).unwrap();
        let param = t(vec![1.0, 1.0], &[2]);
        let result = opt.step_closure(&[param], |_p| (0.0, vec![t(vec![2e38, 2e38], &[2])]));
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
        assert_eq!(opt.n_iter(), 0);
        assert_eq!(opt.func_evals(), 0);
    }
}
