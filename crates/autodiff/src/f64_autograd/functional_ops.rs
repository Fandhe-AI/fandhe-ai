//! `TapeF64` の `jvp`・`jacfwd`（イシュー #3787・親 #3719。設計は
//! `docs/autodiff-functional-transforms-design.md` §36、実装記録は同 §39）。
//!
//! # 役割と呼び出し文脈
//!
//! f32 の `crate::functional_ops::{jvp, jacfwd}`（double-VJP 法）を f64 の独立グラフへ
//! 写したもの。新しい `OpF64` variant は足さず、既存の `VarF64` 演算と
//! [`TapeF64::backward_create_graph`]（#3785・#3786）の合成だけで閉じる（§36.10）。
//! 呼び出し元は #3788 の facade 公開層（本イシューでは内部クレート限定）。
//! 対象外 Op（`max` 等）を含むグラフは `backward_create_graph` の fail-closed がそのまま伝播する。
//!
//! # 数値契約
//!
//! 結果は REQ-2 の統一複合判定でヤコビアン由来の `J·v` と一致し、bit 同一は契約にしない。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use super::{TapeF64, VarF64, checked_bytes_for_f64};
use crate::error::AutodiffError;
use crate::jacobian_ops::checked_numel;

/// 入口検査（`jvp`・`jacfwd` 共通）の結果。
struct Entry {
    out_shape: Vec<usize>,
    n: usize,
    m: usize,
}

/// double-VJP 前段の結果（子テープ上の `g = Jᵀu` と `u` の写し）。
type Stage<'c, 't> = Option<(VarF64<'c, 't>, VarF64<'c, 't>)>;

impl<'t> TapeF64<'t> {
    /// 別テープ → 追跡なしの入力、の順で検査し、要素数を求める。
    /// 戻り値の `bool` は `output` が追跡ありか。
    fn check_entry(
        &self,
        output: &VarF64<'_, 't>,
        input: &VarF64<'_, 't>,
    ) -> Result<(Entry, bool), AutodiffError> {
        if output.graph.id != self.id || input.graph.id != self.id {
            return Err(AutodiffError::TapeMismatch);
        }
        if !input.value_and_requires_grad().1 {
            return Err(AutodiffError::GradientTrackingDisabled);
        }
        let out_tracked = output.value_and_requires_grad().1;
        let out_shape = output.shape();
        let n = checked_numel(&input.shape())?;
        let m = checked_numel(&out_shape)?;
        // 結果（jvp は出力 shape、jacfwd は更に拡張した shape）の確保前に f64 のバイト数を検査し、
        // Vec の上限超過による capacity overflow の panic を防ぐ（§36.3）。
        checked_bytes_for_f64(&out_shape).map_err(AutodiffError::Shape)?;
        Ok((Entry { out_shape, n, m }, out_tracked))
    }

    /// double-VJP 前段。親へ 2 ノード（`u` と `mul`）が残る。`J ≡ 0` は `Ok(None)`。
    fn double_vjp_stage<'c>(
        &self,
        output: &VarF64<'_, 't>,
        input: &VarF64<'_, 't>,
        out_shape: &[usize],
        child: &'c TapeF64<'t>,
    ) -> Result<Stage<'c, 't>, AutodiffError> {
        let u = self.var(&Tensor::ones(out_shape).map_err(AutodiffError::Shape)?);
        let s = output.mul(&u)?;
        let cg = self.backward_create_graph(&s, child)?;
        let Some(cu) = cg.child_var(&u)? else {
            return Err(AutodiffError::Backward(
                "double-VJP: 余接葉 u の子テープ写しが存在しない（不変条件違反）".to_string(),
            ));
        };
        match cg.grad(input)? {
            Some(g) if g.value_and_requires_grad().1 => Ok(Some((g, cu))),
            _ => Ok(None),
        }
    }

    /// ヤコビアン・ベクトル積 `J·v`（`J = ∂output/∂input`）を double-VJP 法で求める
    /// （f32 の `functional_ops::jvp` の f64 版）。戻り値は shape が `output.shape()` の非微分値。
    /// `child` は空の子テープ（[`TapeF64::backward_create_graph`] と同じ契約）で、呼び出し後は再利用しない。
    ///
    /// **入口検査（順序固定。失敗時は親・子とも無変更）**: 別テープ → `TapeMismatch`、
    /// 追跡なしの入力 → `GradientTrackingDisabled`、接ベクトル shape 不一致 →
    /// `Shape(ShapeMismatch)`、要素数 → `Shape(ElementCountOverflow)`。要素数 0・追跡なし出力は
    /// テープに触れず全ゼロ（`child` も検査しない）。以降は `backward_create_graph` の拒否
    /// （`max` 含有・同一/非空/別デバイスの子）が `Err` で伝播する。
    ///
    /// **副作用**: 前段以降の失敗でも親へちょうど 2 ノードが残る。成功時の子には写しが残る。
    /// 数値は REQ-2 の統一複合判定の範囲で `J·v` と一致（bit 同一は主張しない）。
    /// 計算量は親 backward 1 回＋子 backward 1 回、メモリは full mirror 分。
    ///
    /// ```
    /// # use fandhe_ai_autodiff::{Tape, f64_autograd::TapeF64};
    /// # use fandhe_ai_tensor_core::Tensor;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let base = Tape::new();
    /// let (parent, child) = (TapeF64::new(&base), TapeF64::new(&base));
    /// let x = parent.var(&Tensor::new(vec![1.0f64, 2.0, 3.0], &[3])?);
    /// let y = x.mul(&x)?;
    /// let v = Tensor::new(vec![1.0f64, 0.5, 2.0], &[3])?;
    /// let out = parent.jvp(&y, &x, &v, &child)?;
    /// for (g, w) in out.host_slice().iter().zip([2.0f64, 2.0, 12.0]) {
    ///     assert!((g - w).abs() < 1e-5);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn jvp(
        &self,
        output: &VarF64<'_, 't>,
        input: &VarF64<'_, 't>,
        tangent: &Tensor<f64>,
        child: &TapeF64<'t>,
    ) -> Result<Tensor<f64>, AutodiffError> {
        if output.graph.id != self.id || input.graph.id != self.id {
            return Err(AutodiffError::TapeMismatch);
        }
        if !input.value_and_requires_grad().1 {
            return Err(AutodiffError::GradientTrackingDisabled);
        }
        let in_shape = input.shape();
        if tangent.shape() != in_shape.as_slice() {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: in_shape,
                rhs: tangent.shape().to_vec(),
            }));
        }
        let (e, out_tracked) = self.check_entry(output, input)?;
        if e.n == 0 || e.m == 0 || !out_tracked {
            return Tensor::zeros(&e.out_shape).map_err(AutodiffError::Shape);
        }
        let mut data = vec![0.0f64; e.m];
        if let Some((g, cu)) = self.double_vjp_stage(output, input, &e.out_shape, child)? {
            let prod = g.mul(&child.var_no_grad(tangent))?;
            let grads = child.backward(&prod)?;
            if let Some(h) = grads.get(&cu)? {
                copy_grad_row_f64(h, &mut data)?;
            }
        }
        Tensor::new(data, &e.out_shape).map_err(AutodiffError::Shape)
    }

    /// ヤコビアン全体 `J = ∂output/∂input` を double-VJP 法で列ごとに求める（J1 案。§36.10）。
    /// 戻り値は shape が `output.shape ++ input.shape` の非微分値。`child` の契約・入口検査・
    /// 副作用・数値契約は [`TapeF64::jvp`] と同じ（接ベクトル検査を除く）。
    ///
    /// 前段 `g = Jᵀu` を 1 回作り、列 `k` ごとに `g ⊙ e_k` を子で `backward` して
    /// `∂/∂u = J[:, k]` を得る。f32 の `FlatElements` に当たる `narrow`／`reshape` が `VarF64` に
    /// 無く、新 variant（J3）は範囲変更になるため one-hot 定数との `mul` で代える。
    /// 計算量は親 backward 1 回＋子 backward `n` 回、結果は `m × n` 要素（確保前に
    /// 要素数・バイト数を検査し `ElementCountOverflow` で拒否）。子テープは列ごとに rollback して成長を抑える。
    ///
    /// ```
    /// # use fandhe_ai_autodiff::{Tape, f64_autograd::TapeF64};
    /// # use fandhe_ai_tensor_core::Tensor;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let base = Tape::new();
    /// let (parent, child) = (TapeF64::new(&base), TapeF64::new(&base));
    /// let x = parent.var(&Tensor::new(vec![1.0f64, 2.0], &[2])?);
    /// let y = x.mul(&x)?;
    /// let j = parent.jacfwd(&y, &x, &child)?;
    /// for (g, w) in j.host_slice().iter().zip([2.0f64, 0.0, 0.0, 4.0]) {
    ///     assert!((g - w).abs() < 1e-5);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn jacfwd(
        &self,
        output: &VarF64<'_, 't>,
        input: &VarF64<'_, 't>,
        child: &TapeF64<'t>,
    ) -> Result<Tensor<f64>, AutodiffError> {
        let (e, out_tracked) = self.check_entry(output, input)?;
        let total =
            e.m.checked_mul(e.n)
                .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
        let in_shape = input.shape();
        let mut result_shape = e.out_shape.clone();
        result_shape.extend_from_slice(&in_shape);
        checked_bytes_for_f64(&result_shape).map_err(AutodiffError::Shape)?;
        if total == 0 || !out_tracked {
            return Tensor::zeros(&result_shape).map_err(AutodiffError::Shape);
        }
        let mut data = vec![0.0f64; total];
        if let Some((g, cu)) = self.double_vjp_stage(output, input, &e.out_shape, child)? {
            // 列ごとに ek・p の 2 ノードだけが増える。ループ内で消えるため rollback_to の前提を満たす。
            let point = child.rollback_point();
            for k in 0..e.n {
                let r = jacfwd_column(child, &g, &cu, &in_shape, e.n, k, &mut data);
                child.rollback_to(point);
                r?;
            }
        }
        Tensor::new(data, &result_shape).map_err(AutodiffError::Shape)
    }
}

/// 第 `k` 列 `J[:, k]` を求めて `data` へ書く（`jacfwd` の列ループ本体）。
fn jacfwd_column<'c, 't>(
    child: &'c TapeF64<'t>,
    g: &VarF64<'c, 't>,
    cu: &VarF64<'c, 't>,
    in_shape: &[usize],
    n: usize,
    k: usize,
    data: &mut [f64],
) -> Result<(), AutodiffError> {
    let mut hot = vec![0.0f64; n];
    if let Some(slot) = hot.get_mut(k) {
        *slot = 1.0;
    }
    let ek = child.var_no_grad(&Tensor::new(hot, in_shape).map_err(AutodiffError::Shape)?);
    let p = g.mul(&ek)?;
    let grads = child.backward(&p)?;
    if let Some(col) = grads.get(cu)? {
        write_column_f64(data, n, k, col)?;
    }
    Ok(())
}

/// 勾配 `g` を行バッファへ複写する。長さ不一致は型付きエラー。
fn copy_grad_row_f64(g: &Tensor<f64>, dst: &mut [f64]) -> Result<(), AutodiffError> {
    let g = g.contiguous();
    let host = g.host_slice();
    if host.len() != dst.len() {
        return Err(AutodiffError::Backward(format!(
            "jvp: 勾配の要素数（{}）が出力の要素数（{}）と一致しない",
            host.len(),
            dst.len()
        )));
    }
    dst.copy_from_slice(&host);
    Ok(())
}

/// 列 `col`（長さ `m`）を行優先の結果（`m × n`）の第 `k` 列へ書く。
fn write_column_f64(
    data: &mut [f64],
    n: usize,
    k: usize,
    col: &Tensor<f64>,
) -> Result<(), AutodiffError> {
    let col = col.contiguous();
    let host = col.host_slice();
    let rows = data.len().checked_div(n).unwrap_or(0);
    if host.len() != rows || k >= n {
        return Err(AutodiffError::Backward(format!(
            "jacfwd: 列の要素数（{}）が出力の要素数（{rows}）と一致しない、または列添字 {k} が範囲外",
            host.len()
        )));
    }
    for (i, v) in host.iter().enumerate() {
        data[i * n + k] = *v;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_ops::naive_ops;
    use crate::tape::Tape;

    fn t(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
    }

    #[test]
    fn entry_failures_leave_both_tapes_untouched() {
        let base = Tape::new_with_ops(naive_ops());
        let (p, c, other) = (
            TapeF64::new(&base),
            TapeF64::new(&base),
            TapeF64::new(&base),
        );
        let x = p.var(&t(vec![1.0, 2.0], &[2]));
        let y = x.mul(&x).expect("mul");
        let o = other.var(&t(vec![1.0, 2.0], &[2]));
        let v = t(vec![1.0, 1.0], &[2]);
        let nog = p.var_no_grad(&t(vec![1.0, 2.0], &[2]));
        let point = p.rollback_point();
        assert!(matches!(
            p.jvp(&y, &o, &v, &c),
            Err(AutodiffError::TapeMismatch)
        ));
        assert!(matches!(
            p.jvp(&y, &nog, &v, &c),
            Err(AutodiffError::GradientTrackingDisabled)
        ));
        assert!(matches!(
            p.jvp(&y, &x, &t(vec![1.0], &[1]), &c),
            Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
        ));
        assert!(matches!(
            p.jacfwd(&y, &nog, &c),
            Err(AutodiffError::GradientTrackingDisabled)
        ));
        assert_eq!(p.rollback_point(), point);
        assert!(c.is_empty());
    }

    #[test]
    fn rejected_max_adds_two_parent_nodes_and_child_stays_empty() {
        let base = Tape::new_with_ops(naive_ops());
        let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
        let x = p.var(&t(vec![1.0, 2.0], &[2]));
        let bad = x.max(None).expect("max");
        let before = p.rollback_point();
        let v = t(vec![1.0, 1.0], &[2]);
        assert!(matches!(
            p.jvp(&bad, &x, &v, &c),
            Err(AutodiffError::Backward(_))
        ));
        assert_eq!(p.rollback_point(), before + 2);
        assert!(c.is_empty());
    }

    #[test]
    fn jacfwd_rolls_back_child_per_column() {
        let base = Tape::new_with_ops(naive_ops());
        let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
        let x = p.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let y = x.mul(&x).expect("mul");
        // 前段のみを別の親子で実行して、前段終了時の子ノード数を得る。
        let (p2, c2) = (TapeF64::new(&base), TapeF64::new(&base));
        let x2 = p2.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let y2 = x2.mul(&x2).expect("mul");
        let stage = p2.double_vjp_stage(&y2, &x2, &[3], &c2).expect("stage");
        assert!(stage.is_some());
        let expected = c2.rollback_point();
        let j = p.jacfwd(&y, &x, &c).expect("jacfwd");
        assert_eq!(j.shape(), &[3, 3]);
        assert_eq!(c.rollback_point(), expected);
    }

    #[test]
    fn jvp_rejects_output_whose_byte_size_overflows_before_allocation() {
        let base = Tape::new_with_ops(naive_ops());
        let (p, c) = (TapeF64::new(&base), TapeF64::new(&base));
        let x = p.var(&t(vec![1.0, 2.0], &[2]));
        let view = t(vec![1.0], &[1])
            .broadcast_to(&[1usize << 61])
            .expect("view");
        let y = p.var(&view);
        let v = t(vec![1.0, 1.0], &[2]);
        assert!(matches!(
            p.jvp(&y, &x, &v, &c),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
        assert!(matches!(
            p.jacfwd(&y, &x, &c),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
    }

    #[test]
    fn byte_size_check_rejects_overflowing_shapes() {
        assert!(checked_bytes_for_f64(&[usize::MAX, 2]).is_err());
        assert!(checked_bytes_for_f64(&[1usize << 62]).is_err());
        assert!(checked_bytes_for_f64(&[3, 3]).is_ok());
    }
}
