//! `gradcheck`（有限差分との勾配突合。イシュー #2671・親 #2668。契約の正は
//! `docs/autodiff-jacobian-hessian-gradcheck-decision.md` §3.4）。
//!
//! 解析ヤコビアン（既存 [`crate::jacobian_ops::jacobian`] = `Tape::backward` の要素ごとの
//! 繰り返し）と、中心差分の数値ヤコビアンを全要素で突合する検証ユーティリティ。新規 `Op`・
//! VJP・`AutodiffError` variant・`BackendOps` メソッドはない。
//!
//! **判定式**（#223 承認済みの grad-check テスト判定と同式。REQ-2 の tolerance ではない）:
//! `abs = |解析 − 数値|`、`rel = abs / max(|解析|, |数値|, tau)`、`rel <= rtol` または
//! `abs <= atol` で合格。どちらかが非有限なら不合格。PyTorch `gradcheck` の `allclose` 形は
//! 採らない。`eps`・`atol`・`rtol`・`tau` は利用者が [`GradcheckOptions::new`] で明示する
//! アルゴリズム引数で、`Default` を実装しない（既定の閾値を暗黙に持たせない）。
//!
//! **適用範囲（利用者責任）**: 評価回数は `1 + 2·Σn_k`（`n_k` は入力 k の要素数）。
//! キンク（`relu` の 0 付近等）・タイ（`max` 等）の近傍、`f32` の丸めが支配的な低精度 forward は
//! 適用外（`eps` を跨ぐ不連続で偽陽性になる）。`VarF64` 版・複数出力・gradgradcheck は対象外。
//! 数値側は `f(x ± eps)` を入力の実際の `f32` 丸め後の差で割る。解析側は backward を
//! `出力要素数 × 入力数` 回行い `m×n` の領域を確保するため、大きな形状は呼び出し側の責任で避ける。
//!
//! **テープの扱い**: 評価ごとに `make_tape` で新しいテープを作る（`Tape::reset` は葉
//! プレフィックスを保持するため再利用しない）。利用者の既存テープには触れない。
//!
//! **公開状況（イシュー #2847）**: facade へは `Tape::gradcheck` と `GradcheckOptions`／`GradcheckReport`
//! だけを公開した（決定記録 `docs/autodiff-jacobian-hessian-gradcheck-decision.md` §11・§12。
//! 承認は #2499 の issuecomment-6052732061）。本モジュール自体と裸の自由関数 `gradcheck` は
//! 内部クレートの面に留め、facade へは出さない。facade の `Tape::gradcheck` は `make_tape` で
//! `tape_for(device)` を呼ぶため、テープ生成が失敗しうる（`make_tape` が `Result` を返す理由）。
//! 以前の「公開形は未承認（保留）」の記述（#2677・#2678 時点）は #2847 で解消済み。

use crate::error::AutodiffError;
use crate::jacobian_ops::jacobian;
use crate::tape::Tape;
use crate::var::Var;
use fandhe_ai_tensor_core::{ShapeError, Tensor};

/// [`gradcheck`] のアルゴリズム引数。`Default` は実装しない（閾値を暗黙に決めない）。
/// フィールドは非公開で、構築は [`GradcheckOptions::new`]（有限かつ正を fail-closed 検査）のみ。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradcheckOptions {
    eps: f64,
    atol: f64,
    rtol: f64,
    tau: f64,
}

impl GradcheckOptions {
    /// 4 引数すべてが有限かつ正であることを検査して構築する（違反は
    /// `Err(InvalidArgument)`）。`eps` は中心差分の摂動幅、`atol`／`rtol` は絶対・相対の
    /// 合格閾値、`tau` は相対誤差の分母の下限。
    pub fn new(eps: f64, atol: f64, rtol: f64, tau: f64) -> Result<Self, AutodiffError> {
        for (name, v) in [("eps", eps), ("atol", atol), ("rtol", rtol), ("tau", tau)] {
            if !v.is_finite() || v <= 0.0 {
                return Err(AutodiffError::InvalidArgument(format!(
                    "GradcheckOptions: {name} は有限かつ正である必要がある（{v}）"
                )));
            }
        }
        Ok(Self {
            eps,
            atol,
            rtol,
            tau,
        })
    }

    /// 中心差分の摂動幅。
    pub fn eps(&self) -> f64 {
        self.eps
    }
    /// 絶対誤差の合格閾値。
    pub fn atol(&self) -> f64 {
        self.atol
    }
    /// 相対誤差の合格閾値。
    pub fn rtol(&self) -> f64 {
        self.rtol
    }
    /// 相対誤差の分母の下限。
    pub fn tau(&self) -> f64 {
        self.tau
    }
}

/// [`gradcheck`] の結果。不一致は `Err` ではなく `passed() == false` の `Ok` で返す。
/// 最悪要素は絶対誤差が最大の要素（非有限は無限大として扱い、同値なら最初のもの）。
#[derive(Debug, Clone, PartialEq)]
pub struct GradcheckReport {
    passed: bool,
    max_abs_error: f64,
    max_rel_error: f64,
    worst_input: usize,
    worst_output_index: usize,
    worst_input_index: usize,
    checked_elements: usize,
}

impl GradcheckReport {
    /// 全要素が判定式を満たしたか。
    pub fn passed(&self) -> bool {
        self.passed
    }
    /// 最大絶対誤差（非有限の要素があれば `inf`）。
    pub fn max_abs_error(&self) -> f64 {
        self.max_abs_error
    }
    /// 最大相対誤差（分母は `max(|解析|, |数値|, tau)`。非有限の要素があれば `inf`）。
    pub fn max_rel_error(&self) -> f64 {
        self.max_rel_error
    }
    /// 最悪要素の位置 `(入力番号, 出力の平坦添字, 入力の平坦添字)`。
    pub fn worst_location(&self) -> (usize, usize, usize) {
        (
            self.worst_input,
            self.worst_output_index,
            self.worst_input_index,
        )
    }
    /// 突合した要素数（`出力要素数 × Σ入力要素数`）。
    pub fn checked_elements(&self) -> usize {
        self.checked_elements
    }
}

fn checked_numel(shape: &[usize]) -> Result<usize, AutodiffError> {
    shape.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    })
}

/// 1 要素の突合結果 `(合格, 絶対誤差, 相対誤差)`。非有限は不合格で誤差は `inf`。
fn judge(analytic: f64, numeric: f64, opts: &GradcheckOptions) -> (bool, f64, f64) {
    if !analytic.is_finite() || !numeric.is_finite() {
        return (false, f64::INFINITY, f64::INFINITY);
    }
    let abs = (analytic - numeric).abs();
    let rel = abs / analytic.abs().max(numeric.abs()).max(opts.tau);
    (rel <= opts.rtol || abs <= opts.atol, abs, rel)
}

/// 新しいテープで `f(inputs)` を評価し、出力の（shape, f64 昇格した値）を返す。
fn evaluate<M, F>(
    make_tape: &M,
    f: &F,
    inputs: &[Tensor<f32>],
) -> Result<(Vec<usize>, Vec<f64>), AutodiffError>
where
    M: Fn() -> Result<Tape, AutodiffError>,
    F: for<'a> Fn(&'a Tape, &[Var<'a>]) -> Result<Var<'a>, AutodiffError>,
{
    let tape = make_tape()?;
    let vars: Vec<Var<'_>> = inputs.iter().map(|t| tape.var(t)).collect();
    let out = f(&tape, &vars)?;
    if out.tape_id() != tape.id || out.tape_epoch() != tape.epoch() {
        return Err(AutodiffError::TapeMismatch);
    }
    let value = out.to_tensor();
    let data = value.host_slice().iter().map(|&v| f64::from(v)).collect();
    Ok((value.shape().to_vec(), data))
}

/// 解析勾配（`Tape::backward` の合成）と中心差分の数値勾配を全要素で突合する。
///
/// `make_tape` は評価ごとに空の新しいテープを返す関数（解析 1 回＋数値 `2·Σn_k` 回、
/// 計 `1 + 2·Σn_k` 回呼ばれる）。`f` は入力 `Var` 列から**単一の出力** `Var` を作る関数で、
/// 呼ばれるたびに渡されたテープ上へ記録する。
///
/// **入口検査（評価前・順序固定）**: `inputs` が空 → `Err(InvalidArgument)`／各入力の要素数を
/// 検査付きで算出（オーバーフローは `Err(Shape(ElementCountOverflow))`）。続く評価で、
/// 出力が渡されたテープに属さなければ `Err(TapeMismatch)`、出力の要素数が 0・摂動点で出力
/// shape が変わる・`eps` が入力値に対して小さすぎて摂動が `f32` で潰れる場合は
/// `Err(InvalidArgument)`（空虚な合格を返さない）。`f` と `make_tape` のエラー（テープ生成の失敗を含む）は入口検査の後、評価時にそのまま伝播する。
///
/// 判定式・適用範囲・計算量はモジュール doc を参照。
pub fn gradcheck<M, F>(
    make_tape: M,
    f: F,
    inputs: &[Tensor<f32>],
    options: &GradcheckOptions,
) -> Result<GradcheckReport, AutodiffError>
where
    M: Fn() -> Result<Tape, AutodiffError>,
    F: for<'a> Fn(&'a Tape, &[Var<'a>]) -> Result<Var<'a>, AutodiffError>,
{
    if inputs.is_empty() {
        return Err(AutodiffError::InvalidArgument(
            "gradcheck: inputs は 1 つ以上必要".into(),
        ));
    }
    let in_numels = inputs
        .iter()
        .map(|t| checked_numel(t.shape()))
        .collect::<Result<Vec<_>, _>>()?;

    // 解析側: 1 本のテープで出力を作り、入力ごとに jacobian（行 = 出力要素）を取る。
    let tape = make_tape()?;
    let vars: Vec<Var<'_>> = inputs.iter().map(|t| tape.var(t)).collect();
    let out = f(&tape, &vars)?;
    if out.tape_id() != tape.id || out.tape_epoch() != tape.epoch() {
        return Err(AutodiffError::TapeMismatch);
    }
    let out_shape = out.shape();
    let m = checked_numel(&out_shape)?;
    if m == 0 {
        return Err(AutodiffError::InvalidArgument(
            "gradcheck: 出力の要素数が 0（検査対象がない）".into(),
        ));
    }
    let mut analytic: Vec<Tensor<f32>> = Vec::with_capacity(inputs.len());
    for v in &vars {
        analytic.push(jacobian(&tape, &out, v)?);
    }

    // 数値側: 入力 k の要素 j を ±eps だけ動かし、新しいテープで評価する。
    let mut best = GradcheckReport {
        passed: true,
        max_abs_error: 0.0,
        max_rel_error: 0.0,
        worst_input: 0,
        worst_output_index: 0,
        worst_input_index: 0,
        checked_elements: 0,
    };
    let mut worst_abs = -1.0f64;
    for (k, &n) in in_numels.iter().enumerate() {
        let base = inputs[k].host_slice().into_owned();
        let jac = analytic[k].host_slice().into_owned();
        if jac.len() != m.saturating_mul(n) {
            return Err(AutodiffError::Backward(format!(
                "gradcheck: 解析ヤコビアンの要素数（{}）が期待値（{}）と一致しない",
                jac.len(),
                m.saturating_mul(n)
            )));
        }
        for j in 0..n {
            let x = f64::from(base[j]);
            let (xp, xm) = ((x + options.eps) as f32, (x - options.eps) as f32);
            let step = f64::from(xp) - f64::from(xm);
            // f32 変換で xp/xm が ±inf になり得る（例: 大きな入力）。非有限値を f に渡さない。
            // NaN は `step <= 0.0` を素通りするため有限性を先に検査する。
            if !(xp.is_finite() && xm.is_finite() && step.is_finite()) {
                return Err(AutodiffError::InvalidArgument(format!(
                    "gradcheck: 入力 {k} の要素 {j}（{x}）に eps={} を加減した摂動値が f32 で非有限になる",
                    options.eps
                )));
            }
            // 片側だけ潰れる場合（例: x=1.0f32 で eps が半 ULP 程度 → xp==x かつ xm<x）は
            // 中心差分でなく片側差分になるため、step の正値検査だけでなく両側を個別に確認する。
            let base_x = base[j];
            if step <= 0.0 || xp <= base_x || xm >= base_x {
                return Err(AutodiffError::InvalidArgument(format!(
                    "gradcheck: eps={} が入力 {k} の要素 {j}（{x}）に対して小さすぎ、f32 で摂動が潰れる",
                    options.eps
                )));
            }
            let eval_at = |v: f32| -> Result<Vec<f64>, AutodiffError> {
                let mut perturbed = inputs.to_vec();
                let mut data = base.clone();
                data[j] = v;
                perturbed[k] =
                    Tensor::new(data, inputs[k].shape()).map_err(AutodiffError::Shape)?;
                let (shape, vals) = evaluate(&make_tape, &f, &perturbed)?;
                if shape != out_shape {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "gradcheck: 摂動点で出力 shape が変わった（{out_shape:?} → {shape:?}）"
                    )));
                }
                Ok(vals)
            };
            let plus = eval_at(xp)?;
            let minus = eval_at(xm)?;
            for i in 0..m {
                let numeric = (plus[i] - minus[i]) / step;
                let a = f64::from(jac[i * n + j]);
                let (ok, abs, rel) = judge(a, numeric, options);
                best.checked_elements += 1;
                best.passed &= ok;
                if rel > best.max_rel_error {
                    best.max_rel_error = rel;
                }
                if abs > best.max_abs_error {
                    best.max_abs_error = abs;
                }
                if abs > worst_abs {
                    worst_abs = abs;
                    best.worst_input = k;
                    best.worst_output_index = i;
                    best.worst_input_index = j;
                }
            }
        }
    }
    if best.checked_elements == 0 {
        return Err(AutodiffError::InvalidArgument(
            "gradcheck: 入力の要素数がすべて 0（検査対象がない）".into(),
        ));
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> GradcheckOptions {
        GradcheckOptions::new(1e-3, 1e-3, 1e-2, 1e-4).unwrap()
    }

    #[test]
    fn options_reject_non_finite_and_non_positive() {
        assert!(GradcheckOptions::new(1e-3, 1e-3, 1e-2, 1e-4).is_ok());
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(GradcheckOptions::new(bad, 1.0, 1.0, 1.0).is_err());
            assert!(GradcheckOptions::new(1.0, bad, 1.0, 1.0).is_err());
            assert!(GradcheckOptions::new(1.0, 1.0, bad, 1.0).is_err());
            assert!(GradcheckOptions::new(1.0, 1.0, 1.0, bad).is_err());
        }
    }

    #[test]
    fn judge_passes_by_relative_or_absolute() {
        let o = opts();
        // 相対 0.5%（rtol 1% 以内）。
        assert!(judge(100.0, 100.5, &o).0);
        // 小さい値では絶対 1e-4 が atol 以内。
        assert!(judge(1e-5, 1.1e-4, &o).0);
        // 相対 50% かつ絶対 1 超。
        assert!(!judge(2.0, 3.0, &o).0);
    }

    #[test]
    fn judge_rejects_non_finite() {
        let o = opts();
        assert!(!judge(f64::NAN, 0.0, &o).0);
        assert!(!judge(0.0, f64::INFINITY, &o).0);
        assert_eq!(judge(f64::NAN, 0.0, &o).1, f64::INFINITY);
    }

    #[test]
    fn checked_numel_detects_overflow() {
        assert_eq!(checked_numel(&[]).ok(), Some(1));
        assert_eq!(checked_numel(&[2, 3]).ok(), Some(6));
        assert!(matches!(
            checked_numel(&[usize::MAX, 2]),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
    }
}
