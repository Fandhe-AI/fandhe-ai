//! `TapeF64` の `jvp`・`jacfwd`（イシュー #3787・親 #3719。設計は
//! `docs/autodiff-functional-transforms-design.md` §36、実装記録は同 §39）。
//!
//! # 役割と呼び出し文脈
//!
//! f32 の `crate::functional_ops::{jvp, jacfwd}`（double-VJP 法）を f64 の独立グラフへ
//! 写したもの。新しい `OpF64` variant は足さず、既存の `VarF64` 演算と
//! [`TapeF64::backward_create_graph`]（#3785・#3786）の合成だけで閉じる（§36.10）。
//! `jvp_multi`・`jvp_create_graph`・結果型 [`JvpCreateGraphResultF64`] も同じ前段
//! （`double_vjp_front`）を共用する（イシュー #3788・設計記録 §40）。
//! 呼び出し元は facade の `TapeF64` 公開層（#3788。4 関数を 1 式委譲で公開）。
//! 対象外 Op（`max` 等）を含むグラフは `backward_create_graph` の fail-closed がそのまま伝播する。
//!
//! # 数値契約
//!
//! 結果は REQ-2 の統一複合判定でヤコビアン由来の `J·v` と一致し、bit 同一は契約にしない。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use super::{CreateGraphResultF64, TapeF64, TapeIdF64, VarF64, checked_bytes_for_f64};
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

/// `double_vjp_front` の結果（親の `backward_create_graph` 結果と余接葉 `u` の子テープ写し）。
type Front<'c, 't> = (CreateGraphResultF64<'c, 't>, VarF64<'c, 't>);

impl<'t> TapeF64<'t> {
    /// 別テープ → 追跡なしの入力、の順で検査し、要素数を求める。
    /// 戻り値の `bool` は `output` が追跡ありか。
    fn check_entry(
        &self,
        output: &VarF64<'_, 't>,
        input: &VarF64<'_, 't>,
        jacobian: bool,
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
        // jacfwd は入力が空だと結果が 0 要素になり、出力側の検査（要素数・バイト数）に先んじて
        // 零返却経路へ入る（出力が巨大でも拒否しない。結果 shape の検査は jacfwd 側が行う）。
        if jacobian && n == 0 {
            return Ok((Entry { out_shape, n, m: 0 }, out_tracked));
        }
        let m = checked_numel(&out_shape)?;
        // 結果（jvp は出力 shape、jacfwd は更に拡張した shape）の確保前に f64 のバイト数を検査し、
        // Vec の上限超過による capacity overflow の panic を防ぐ（§36.3）。
        checked_bytes_for_f64(&out_shape).map_err(AutodiffError::Shape)?;
        Ok((Entry { out_shape, n, m }, out_tracked))
    }

    /// double-VJP 前段の共通部（`double_vjp_stage`・`jvp_multi`・`jvp_create_graph` が呼ぶ）。
    /// 親へ 2 ノード（`u` と `mul`）を足し、`backward_create_graph` の結果と
    /// 余接葉 `u` の子テープ写しを返す。
    fn double_vjp_front<'c>(
        &self,
        output: &VarF64<'_, 't>,
        out_shape: &[usize],
        child: &'c TapeF64<'t>,
    ) -> Result<Front<'c, 't>, AutodiffError> {
        let u = self.var(&Tensor::ones(out_shape).map_err(AutodiffError::Shape)?);
        let s = output.mul(&u)?;
        let cg = self.backward_create_graph(&s, child)?;
        let Some(cu) = cg.child_var(&u)? else {
            return Err(AutodiffError::Backward(
                "double-VJP: 余接葉 u の子テープ写しが存在しない（不変条件違反）".to_string(),
            ));
        };
        Ok((cg, cu))
    }

    /// double-VJP 前段。親へ 2 ノード（`u` と `mul`）が残る。`J ≡ 0` は `Ok(None)`。
    fn double_vjp_stage<'c>(
        &self,
        output: &VarF64<'_, 't>,
        input: &VarF64<'_, 't>,
        out_shape: &[usize],
        child: &'c TapeF64<'t>,
    ) -> Result<Stage<'c, 't>, AutodiffError> {
        let (cg, cu) = self.double_vjp_front(output, out_shape, child)?;
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
        let (e, out_tracked) = self.check_entry(output, input, false)?;
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
        let (e, out_tracked) = self.check_entry(output, input, true)?;
        let total =
            e.m.checked_mul(e.n)
                .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
        let in_shape = input.shape();
        let mut result_shape = e.out_shape.clone();
        result_shape.extend_from_slice(&in_shape);
        if total == 0 || !out_tracked {
            // 零返却経路。確保の検査は `Tensor::zeros` 自身が行う。
            return Tensor::zeros(&result_shape).map_err(AutodiffError::Shape);
        }
        checked_bytes_for_f64(&result_shape).map_err(AutodiffError::Shape)?;
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

    /// 複数入力のヤコビアン・ベクトル積 `J·v = Σ_i J_i·v_i`（`J_i = ∂output/∂inputs[i]`）を
    /// double-VJP 法で求める（f32 の `functional_ops::jvp_multi` の f64 版。イシュー #3788・
    /// 設計記録 §36・§40。facade の `TapeF64::jvp_multi` から 1 行委譲で呼ばれる）。
    /// 戻り値は shape が `output.shape()` の非微分値。`child` の契約は [`TapeF64::jvp`] と同じ。
    ///
    /// **入口検査（順序固定。失敗時は親・子とも無変更）**:
    /// 1. 長さ不一致 → `InvalidArgument`、2. 空 → `InvalidArgument`、
    /// 3. `output`・`inputs`（添字順）が別テープ → `TapeMismatch`、
    /// 4. 追跡なしの入力 → `GradientTrackingDisabled`、5. 同一ノードの重複 → `InvalidArgument`、
    /// 6. 接ベクトル shape 不一致 → `Shape(ShapeMismatch)`、
    /// 7. 要素数 → `Shape(ElementCountOverflow)`、8. 出力のバイト数 → `Shape(ElementCountOverflow)`、
    /// 9. 出力・全入力の要素数 0 または追跡なし出力 → テープに触れず全ゼロ（`child` も検査しない）。
    ///
    /// **算法**: 親 backward は 1 回。入力ごとの `p_i = Σ(g_i ⊙ v_i)` を `add` で畳んだ合計を
    /// 子で 1 回 backward する。寄与する入力が無ければ全ゼロ。
    /// **副作用**: 入口検査後の失敗でも親へちょうど 2 ノードが残る。数値は REQ-2 の統一複合判定の
    /// 範囲で `Σ J_i·v_i` と一致（bit 同一は主張しない）。メモリは full mirror 分。
    ///
    /// ```
    /// # use fandhe_ai_autodiff::{Tape, f64_autograd::TapeF64};
    /// # use fandhe_ai_tensor_core::Tensor;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let base = Tape::new();
    /// let (parent, child) = (TapeF64::new(&base), TapeF64::new(&base));
    /// let a = parent.var(&Tensor::new(vec![1.0f64, 2.0], &[2])?);
    /// let b = parent.var(&Tensor::new(vec![3.0f64, 4.0], &[2])?);
    /// let y = a.mul(&b)?;
    /// let va = Tensor::new(vec![1.0f64, 1.0], &[2])?;
    /// let vb = Tensor::new(vec![0.5f64, 2.0], &[2])?;
    /// let out = parent.jvp_multi(&y, &[&a, &b], &[&va, &vb], &child)?;
    /// // d(a*b) = b*va + a*vb
    /// for (g, w) in out.host_slice().iter().zip([3.5f64, 8.0]) {
    ///     assert!((g - w).abs() < 1e-5);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn jvp_multi(
        &self,
        output: &VarF64<'_, 't>,
        inputs: &[&VarF64<'_, 't>],
        tangents: &[&Tensor<f64>],
        child: &TapeF64<'t>,
    ) -> Result<Tensor<f64>, AutodiffError> {
        if inputs.len() != tangents.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "jvp_multi: inputs の数（{}）と tangents の数（{}）が一致しない",
                inputs.len(),
                tangents.len()
            )));
        }
        if inputs.is_empty() {
            return Err(AutodiffError::InvalidArgument(
                "jvp_multi: inputs が空".to_string(),
            ));
        }
        if output.graph.id != self.id || inputs.iter().any(|v| v.graph.id != self.id) {
            return Err(AutodiffError::TapeMismatch);
        }
        if inputs.iter().any(|v| !v.value_and_requires_grad().1) {
            return Err(AutodiffError::GradientTrackingDisabled);
        }
        for (i, a) in inputs.iter().enumerate() {
            if inputs[..i].iter().any(|b| b.id == a.id) {
                return Err(AutodiffError::InvalidArgument(format!(
                    "jvp_multi: inputs[{i}] が重複指定されている"
                )));
            }
        }
        for (input, tangent) in inputs.iter().zip(tangents) {
            let in_shape = input.shape();
            if tangent.shape() != in_shape.as_slice() {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: in_shape,
                    rhs: tangent.shape().to_vec(),
                }));
            }
        }
        let out_shape = output.shape();
        let m = checked_numel(&out_shape)?;
        let mut all_empty = true;
        for input in inputs {
            if checked_numel(&input.shape())? != 0 {
                all_empty = false;
            }
        }
        // 結果の確保前に f64 のバイト数を検査する（A03。`jvp` の `check_entry` と同じ）。
        checked_bytes_for_f64(&out_shape).map_err(AutodiffError::Shape)?;
        if m == 0 || all_empty || !output.value_and_requires_grad().1 {
            return Tensor::zeros(&out_shape).map_err(AutodiffError::Shape);
        }

        let mut data = vec![0.0f64; m];
        let (cg, cu) = self.double_vjp_front(output, &out_shape, child)?;
        let mut total: Option<VarF64<'_, 't>> = None;
        for (input, tangent) in inputs.iter().zip(tangents) {
            if checked_numel(tangent.shape())? == 0 {
                continue;
            }
            let Some(g) = cg.grad(input)? else { continue };
            if !g.value_and_requires_grad().1 {
                continue;
            }
            let p = g.mul(&child.var_no_grad(tangent))?.sum(None)?;
            total = Some(match total {
                Some(acc) => acc.add(&p)?,
                None => p,
            });
        }
        if let Some(total) = total {
            let grads = child.backward(&total)?;
            if let Some(h) = grads.get(&cu)? {
                copy_grad_row_f64(h, &mut data)?;
            }
        }
        Tensor::new(data, &out_shape).map_err(AutodiffError::Shape)
    }

    /// 追跡された `VarF64` を返す微分可能な `jvp`（f32 の `functional_ops::jvp_create_graph` の
    /// f64 版。イシュー #3788・設計記録 §36・§40。facade の `TapeF64::jvp_create_graph` から 1 行委譲）。
    /// 前段は [`TapeF64::jvp`] と同じ double-VJP 法で、2 段目の VJP だけ
    /// `child.backward_create_graph` で孫テープ `grandchild` に記録する。結果は孫テープ上の
    /// 追跡対象値で、元の入力への微分は [`JvpCreateGraphResultF64::child_var`] の写し
    /// （親→子→孫の 2 段合成）を通して得る。
    ///
    /// **入口検査（順序固定。ここまでの失敗では 3 枚のテープとも無変更）**:
    /// 1. 別テープ → `TapeMismatch`、2. 追跡なしの入力 → `GradientTrackingDisabled`、
    /// 3. 接ベクトル shape 不一致 → `Shape(ShapeMismatch)`、4. 要素数・バイト数 → `Shape(ElementCountOverflow)`、
    /// 5. テープ関係（子が親と同一・孫が親か子と同一 → `Backward`、子・孫のデバイス違い →
    ///    `DeviceMismatch`、子か孫が空でない → `Backward`）。
    ///
    /// 追跡なし出力・要素数 0 は 5 の後に、孫へ全ゼロの葉を 1 個置いて返す（写しは空）。
    ///
    /// 以降は `backward_create_graph` の拒否（`max` 含有等）が `Backward` で伝播する。親へは前段で
    /// ちょうど 2 ノードが残る。`J ≡ 0` も全ゼロの葉を返す。数値は REQ-2 の範囲で `jvp` と一致する。
    /// メモリは親→子の full mirror に加え、孫へ 2 段目の逆伝播グラフが残る分。
    ///
    /// ```
    /// # use fandhe_ai_autodiff::{Tape, f64_autograd::TapeF64};
    /// # use fandhe_ai_tensor_core::Tensor;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let base = Tape::new();
    /// let (p, c, g) = (TapeF64::new(&base), TapeF64::new(&base), TapeF64::new(&base));
    /// let x = p.var(&Tensor::new(vec![1.0f64, 2.0], &[2])?);
    /// let y = x.mul(&x)?;
    /// let v = Tensor::new(vec![1.0f64, 0.5], &[2])?;
    /// let res = p.jvp_create_graph(&y, &x, &v, &c, &g)?;
    /// let cx = res.child_var(&x)?.ok_or("x の写しがある")?;
    /// let grads = g.backward(&res.output().sum(None)?)?;
    /// let d = grads.get(&cx)?.ok_or("勾配が届く")?;
    /// // J·v = 2 x v なので d/dx Σ(J·v) = 2 v
    /// for (a, w) in d.host_slice().iter().zip([2.0f64, 1.0]) {
    ///     assert!((a - w).abs() < 1e-5);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn jvp_create_graph<'g>(
        &self,
        output: &VarF64<'_, 't>,
        input: &VarF64<'_, 't>,
        tangent: &Tensor<f64>,
        child: &TapeF64<'t>,
        grandchild: &'g TapeF64<'t>,
    ) -> Result<JvpCreateGraphResultF64<'g, 't>, AutodiffError> {
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
        let out_shape = output.shape();
        let n = checked_numel(&in_shape)?;
        let m = checked_numel(&out_shape)?;
        checked_bytes_for_f64(&out_shape).map_err(AutodiffError::Shape)?;

        if child.id == self.id {
            return Err(AutodiffError::Backward(
                "jvp_create_graph: 子テープに親と同一の TapeF64 は指定できない".into(),
            ));
        }
        if grandchild.id == self.id || grandchild.id == child.id {
            return Err(AutodiffError::Backward(
                "jvp_create_graph: 孫テープは親・子と異なる TapeF64 である必要がある".into(),
            ));
        }
        for other in [child, grandchild] {
            if other.tape.device() != self.tape.device() {
                return Err(AutodiffError::DeviceMismatch {
                    requested: other.tape.device(),
                    actual: self.tape.device(),
                });
            }
        }
        if !child.is_empty() || !grandchild.is_empty() {
            return Err(AutodiffError::Backward(
                "jvp_create_graph: 子テープ・孫テープは空である必要がある".into(),
            ));
        }
        if n == 0 || m == 0 || !output.value_and_requires_grad().1 {
            return self.zero_jvp_result(grandchild, &out_shape);
        }

        let (cg1, cu) = self.double_vjp_front(output, &out_shape, child)?;
        let g = match cg1.grad(input)? {
            Some(g) if g.value_and_requires_grad().1 => g,
            _ => return self.zero_jvp_result(grandchild, &out_shape),
        };
        let prod = g.mul(&child.var_no_grad(tangent))?;
        let cg2 = child.backward_create_graph(&prod, grandchild)?;
        let Some(r) = cg2.grad(&cu)? else {
            return Err(AutodiffError::Backward(
                "jvp_create_graph: 余接葉の 2 段目の勾配が存在しない（不変条件違反）".to_string(),
            ));
        };

        // 追跡なしの親ノードの写しは `var_no_grad` の葉で `child_var` が Err になるため、追跡ありだけ合成する。
        let mut mirror: Vec<Option<VarF64<'g, 't>>> = vec![None; cg1.mirror().len()];
        for (slot, cv) in mirror.iter_mut().zip(cg1.mirror()) {
            if let Some(cv) = cv
                && cv.graph.nodes.borrow()[cv.id.0].requires_grad
            {
                *slot = cg2.child_var(cv)?;
            }
        }
        Ok(JvpCreateGraphResultF64 {
            output: r,
            parent_id: self.id,
            mirror,
        })
    }

    /// 全ゼロの `J·v` を孫テープの追跡なし葉として返すゼロ経路（写しは空）。
    fn zero_jvp_result<'g>(
        &self,
        grandchild: &'g TapeF64<'t>,
        out_shape: &[usize],
    ) -> Result<JvpCreateGraphResultF64<'g, 't>, AutodiffError> {
        let zeros = Tensor::zeros(out_shape).map_err(AutodiffError::Shape)?;
        Ok(JvpCreateGraphResultF64 {
            output: grandchild.var_no_grad(&zeros),
            parent_id: self.id,
            mirror: Vec::new(),
        })
    }
}

/// [`TapeF64::jvp_create_graph`] の結果（f32 の `JvpCreateGraphResult` の f64 版。イシュー #3788）。
/// 孫テープ上の追跡された `J·v` と、親の `VarF64` から孫テープ上の写し（親→子→孫の 2 段合成）を
/// 引く対応表を持つ。`TapeF64` のノード列は追記専用で世代を持たないため epoch は要らない。
/// facade の newtype `JvpCreateGraphResultF64` が本型を 1 つだけ所有する。
pub struct JvpCreateGraphResultF64<'g, 't> {
    output: VarF64<'g, 't>,
    parent_id: TapeIdF64,
    /// 親 `NodeIdF64` を添字とする孫テープ上の写し（ゼロ経路では空）。
    mirror: Vec<Option<VarF64<'g, 't>>>,
}

impl<'g, 't> JvpCreateGraphResultF64<'g, 't> {
    /// `J·v`（shape は元の `output` と同じ）を孫テープ上の `VarF64` として返す。
    /// 出力が入力に依存しない場合は追跡なしになりうる。
    pub fn output(&self) -> VarF64<'g, 't> {
        self.output
    }

    /// 親の `var` の孫テープ上の写しを返す。別テープ → `TapeMismatch`、追跡なし →
    /// `GradientTrackingDisabled`、`Ok(None)` は `J·v` がその入力に恒等的に依存しないこと。
    pub fn child_var(&self, var: &VarF64<'_, 't>) -> Result<Option<VarF64<'g, 't>>, AutodiffError> {
        if var.graph.id != self.parent_id {
            return Err(AutodiffError::TapeMismatch);
        }
        if !var.graph.nodes.borrow()[var.id.0].requires_grad {
            return Err(AutodiffError::GradientTrackingDisabled);
        }
        Ok(self.mirror.get(var.id.0).copied().flatten())
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
    fn multi_and_create_graph_entry_failures_leave_all_tapes_untouched() {
        let base = Tape::new_with_ops(naive_ops());
        let (p, c, g, other) = (
            TapeF64::new(&base),
            TapeF64::new(&base),
            TapeF64::new(&base),
            TapeF64::new(&base),
        );
        let x = p.var(&t(vec![1.0, 2.0], &[2]));
        let y = x.mul(&x).expect("mul");
        let o = other.var(&t(vec![1.0, 2.0], &[2]));
        let nog = p.var_no_grad(&t(vec![1.0, 2.0], &[2]));
        let v = t(vec![1.0, 1.0], &[2]);
        let point = p.rollback_point();
        assert!(matches!(
            p.jvp_multi(&y, &[&x], &[], &c),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            p.jvp_multi(&y, &[], &[], &c),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            p.jvp_multi(&y, &[&o], &[&v], &c),
            Err(AutodiffError::TapeMismatch)
        ));
        assert!(matches!(
            p.jvp_multi(&y, &[&nog], &[&v], &c),
            Err(AutodiffError::GradientTrackingDisabled)
        ));
        assert!(matches!(
            p.jvp_multi(&y, &[&x, &x], &[&v, &v], &c),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            p.jvp_multi(&y, &[&x], &[&t(vec![1.0], &[1])], &c),
            Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
        ));
        assert!(matches!(
            p.jvp_create_graph(&y, &o, &v, &c, &g),
            Err(AutodiffError::TapeMismatch)
        ));
        assert!(matches!(
            p.jvp_create_graph(&y, &nog, &v, &c, &g),
            Err(AutodiffError::GradientTrackingDisabled)
        ));
        assert!(matches!(
            p.jvp_create_graph(&y, &x, &v, &c, &c),
            Err(AutodiffError::Backward(_))
        ));
        assert_eq!(p.rollback_point(), point);
        assert!(c.is_empty() && g.is_empty());
    }

    #[test]
    fn rejected_max_in_multi_and_create_graph_adds_two_parent_nodes_only() {
        let base = Tape::new_with_ops(naive_ops());
        let (p, c, g) = (
            TapeF64::new(&base),
            TapeF64::new(&base),
            TapeF64::new(&base),
        );
        let x = p.var(&t(vec![1.0, 2.0], &[2]));
        let bad = x.max(None).expect("max");
        let v = t(vec![1.0, 1.0], &[2]);
        let before = p.rollback_point();
        assert!(matches!(
            p.jvp_multi(&bad, &[&x], &[&v], &c),
            Err(AutodiffError::Backward(_))
        ));
        assert_eq!(p.rollback_point(), before + 2);
        let before = p.rollback_point();
        assert!(matches!(
            p.jvp_create_graph(&bad, &x, &v, &c, &g),
            Err(AutodiffError::Backward(_))
        ));
        assert_eq!(p.rollback_point(), before + 2);
        assert!(c.is_empty() && g.is_empty());
    }

    #[test]
    fn multi_and_create_graph_reject_output_whose_byte_size_overflows() {
        let base = Tape::new_with_ops(naive_ops());
        let (p, c, g) = (
            TapeF64::new(&base),
            TapeF64::new(&base),
            TapeF64::new(&base),
        );
        let x = p.var(&t(vec![1.0, 2.0], &[2]));
        let view = t(vec![1.0], &[1])
            .broadcast_to(&[1usize << 61])
            .expect("view");
        let y = p.var(&view);
        let v = t(vec![1.0, 1.0], &[2]);
        assert!(matches!(
            p.jvp_multi(&y, &[&x], &[&v], &c),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
        assert!(matches!(
            p.jvp_create_graph(&y, &x, &v, &c, &g),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
        assert!(c.is_empty() && g.is_empty());
    }

    #[test]
    fn byte_size_check_rejects_overflowing_shapes() {
        assert!(checked_bytes_for_f64(&[usize::MAX, 2]).is_err());
        assert!(checked_bytes_for_f64(&[1usize << 62]).is_err());
        assert!(checked_bytes_for_f64(&[3, 3]).is_ok());
    }
}
