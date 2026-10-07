//! `jacobian`・`hessian`（イシュー #2670・親 #2668。契約の正は
//! `docs/autodiff-jacobian-hessian-gradcheck-decision.md` §3.2・§3.3）。
//!
//! **新規 `Op`・`BackendOps` メソッド・VJP はゼロ**: 既存の
//! [`Tape::backward`]（呼び出しごとに独立した `Gradients` を返し、テープを
//! 消費しない）と [`Tape::backward_create_graph`]（子テープ方式の二階微分）を
//! 要素ごとに繰り返し呼ぶだけの reverse-mode 合成（PyTorch
//! `torch.autograd.functional.jacobian`／`hessian` の `vectorize=False`・
//! `create_graph=False` 相当）。CUDA／Metal は既存 Op の合成で到達するため
//! `Unsupported` フォールバックの追加対象がない。
//!
//! **公開形は未承認（保留）**: facade（`fandhe_ai`）へは公開しない。推奨案は
//! `Tape` の委譲メソッド（本モジュールへ 1 行委譲）で、承認依頼は #2677・
//! 公開は承認後の #2678。保留は facade の `JacobianHessianHoldDoctestGuard` と
//! `tests/api_surface.rs` の否定ガードで機械固定している。
//!
//! **共通の契約**:
//! - 検査はすべてテープ（子テープ）へノードを足す前に済ませる。順序は各関数の
//!   doc に固定する。
//! - テープへ補助ノードが増える（出力・勾配の平坦化は非 contiguous のときのみ
//!   `contiguous` 1、`reshape` 1、要素の取り出し `narrow` が要素数ぶん。rank 0 は
//!   平坦化を省く）。値は変えない。途中の `Err` 時も補助ノードは残る。
//! - 計算量は backward を `m`（jacobian は出力要素数、hessian は入力要素数）回。
//!   結果は `m × n` 個の `f32` を確保するため、大きな形状は呼び出し側の責任で避ける。
//! - 単一入力のみ。resident・fused 経路・checkpoint は既存 backward の挙動に従う。
//! - 結果は非微分のホスト値（`Tensor<f32>`）。

use crate::error::AutodiffError;
use crate::tape::Tape;
use crate::var::Var;
use fandhe_ai_tensor_core::{ShapeError, Tensor};

/// shape の要素数を `checked_mul` で求める（オーバーフロー時は型付きエラー）。
fn checked_numel(shape: &[usize]) -> Result<usize, AutodiffError> {
    shape.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    })
}

/// 結果要素数 `rows × cols` を検査付きで求める（`m×n` のオーバーフロー検査）。
fn checked_result_len(rows: usize, cols: usize) -> Result<usize, AutodiffError> {
    rows.checked_mul(cols)
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))
}

/// `a ++ b`（結果 shape の連結）。
fn concat_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut s = Vec::with_capacity(a.len() + b.len());
    s.extend_from_slice(a);
    s.extend_from_slice(b);
    s
}

/// `v` がテープ `tape` の現世代に属することを検査する（クロステープ・
/// `Tape::reset` 世代違いはどちらも `TapeMismatch`）。
fn check_on_tape(tape: &Tape, v: &Var<'_>) -> Result<(), AutodiffError> {
    if v.tape_id() != tape.id || v.tape_epoch() != tape.epoch() {
        return Err(AutodiffError::TapeMismatch);
    }
    Ok(())
}

/// 要素ごとの取り出し元。rank 0 は `v` 自身を 1 要素として使い、それ以外は
/// contiguous 化したうえで `[numel]` へ平坦化し `narrow` で 1 要素ずつ取り出す
/// （`Var::flatten` は reshape 委譲で非 contiguous 入力に制約があるため、
/// `pub(crate)` の `Var::contiguous` を先に挟む）。jacobian・hessian で共用し、
/// 後続の gradcheck（#2671）も同じ取り出しを使う想定。
struct FlatElements<'t> {
    flat: Var<'t>,
    scalar: bool,
}

impl<'t> FlatElements<'t> {
    fn new(v: &Var<'t>) -> Result<FlatElements<'t>, AutodiffError> {
        let shape = v.shape();
        if shape.is_empty() {
            return Ok(FlatElements {
                flat: *v,
                scalar: true,
            });
        }
        let numel = checked_numel(&shape)?;
        let flat = v.contiguous()?.reshape(&[numel])?;
        Ok(FlatElements {
            flat,
            scalar: false,
        })
    }

    /// 平坦添字 `i` の要素（rank 0 は自身、それ以外は shape `[1]`）。
    /// shape `[1]` の非スカラーでも `Tape::backward` は全要素 1 のシード
    /// （暗黙の総和）なので `sum` ノードは足さない。
    fn element(&self, i: usize) -> Result<Var<'t>, AutodiffError> {
        if self.scalar {
            Ok(self.flat)
        } else {
            self.flat.narrow(0, i, 1)
        }
    }
}

/// 勾配テンソル `g` を論理順のホスト値として `dst`（長さ `n`）へコピーする。
/// 長さが `n` でなければ型付きエラー（panic しない）。
fn copy_grad_row(g: &Tensor<f32>, dst: &mut [f32]) -> Result<(), AutodiffError> {
    let host = g.host_slice();
    if host.len() != dst.len() {
        return Err(AutodiffError::Backward(format!(
            "勾配の要素数（{}）が入力の要素数（{}）と一致しない",
            host.len(),
            dst.len()
        )));
    }
    dst.copy_from_slice(&host);
    Ok(())
}

/// ヤコビアン `∂output/∂input`（PyTorch `torch.autograd.functional.jacobian`
/// の reverse-mode・`vectorize=False` 相当）。戻り値は shape
/// `output.shape ++ input.shape` の非微分ホスト値で、行 `i`（`output` の
/// 平坦添字）は `output[i]` の `input` に関する勾配。
///
/// **入口検査（テープへノードを足す前。順序固定）**:
/// 1. `output`／`input` が `tape` の現世代に属さない → `Err(TapeMismatch)`。
/// 2. `input.requires_grad() == false` → `Err(GradientTrackingDisabled)`。
/// 3. `m = numel(output)`・`n = numel(input)`・`m × n` を検査付き乗算で算出
///    （オーバーフローは `Err(Shape(ElementCountOverflow))`）。
/// 4. `output` が `input` に構造的に依存しない（`output.requires_grad() ==
///    false`）→ backward を呼ばず全ゼロを返す（素の `backward` は追跡なしの
///    loss を `Err` にするため先に分岐する）。要素数 0 の `output`／`input` も
///    空（要素数 0）のテンソルを返す。
///
/// 本体は `output` の要素ごとに `tape.backward` を `m` 回呼び、`input` へ
/// 勾配が届かない行（`Gradients::get` が `Ok(None)`）は全ゼロとする。
/// `backward.rs`／`grad.rs` は変更しない。補助ノードの増加・計算量・単一入力・
/// resident／fused／checkpoint の扱いはモジュール doc を参照。
pub fn jacobian(
    tape: &Tape,
    output: &Var<'_>,
    input: &Var<'_>,
) -> Result<Tensor<f32>, AutodiffError> {
    check_on_tape(tape, output)?;
    check_on_tape(tape, input)?;
    if !input.requires_grad() {
        return Err(AutodiffError::GradientTrackingDisabled);
    }
    let out_shape = output.shape();
    let in_shape = input.shape();
    let m = checked_numel(&out_shape)?;
    let n = checked_numel(&in_shape)?;
    let total = checked_result_len(m, n)?;
    let result_shape = concat_shapes(&out_shape, &in_shape);
    if total == 0 || !output.requires_grad() {
        return Tensor::zeros(&result_shape).map_err(AutodiffError::Shape);
    }

    let elements = FlatElements::new(output)?;
    let mut data = vec![0.0f32; total];
    for (i, row) in data.chunks_exact_mut(n).enumerate() {
        let element = elements.element(i)?;
        let grads = tape.backward(&element)?;
        if let Some(g) = grads.get(input)? {
            copy_grad_row(g, row)?;
        }
    }
    Tensor::new(data, &result_shape).map_err(AutodiffError::Shape)
}

/// ヘッセ行列 `∂²loss/∂input²`（PyTorch `torch.autograd.functional.hessian`
/// の reverse-mode・`vectorize=False` 相当）。戻り値は shape
/// `input.shape ++ input.shape` の非微分ホスト値。`child` は呼び出し側が
/// [`Tape::new_with_ops`] 等で構築した空の子テープ（[`Tape::backward_create_graph`]
/// と同じ契約）。
///
/// **入口検査（`backward_create_graph` を呼ぶ前。失敗時 `child` は無変更。
/// 順序固定）**:
/// 1. `loss`／`input` が `tape` の現世代に属さない → `Err(TapeMismatch)`。
/// 2. `input.requires_grad() == false` → `Err(GradientTrackingDisabled)`。
/// 3. `loss` の要素数が 1 でない → `Err(InvalidArgument)`（shape `[]`・`[1]`・
///    `[1, 1]` は可。後から緩めるのは非破壊だが逆は破壊的なため厳しい側を
///    初期値にする）。
/// 4. `n × n` を検査付き乗算で算出（オーバーフローは
///    `Err(Shape(ElementCountOverflow))`）。`input` の要素数 0 は空のテンソルを
///    返す（この場合 `child` は検査されない）。
///
/// 以降は [`Tape::backward_create_graph`] の既存検査（対象外 Op・非空の子テープ・
/// デバイス不一致・checkpoint 済み親・追跡なし loss の拒否）をそのまま伝播する
/// （新しい検査・エラー variant は足さない）。対象は `Op::supports_create_graph()`
/// の範囲のみ。1 階勾配が `input` へ届かない場合、1 階勾配が定数
/// （`requires_grad == false`。例: 入力に線形な loss）の行、子テープの backward で
/// 二階勾配が `input` に届かない行は全ゼロとする。
///
/// 呼び出し後の `child` には記録が残る（再利用前に呼び出し側が作り直す）。
/// 子テープ上の数値方式は 1 階 VJP と bit 同一を主張しない（`create_graph` の
/// 既存契約）。HVP 専用 API は含めない。
pub fn hessian(
    tape: &Tape,
    loss: &Var<'_>,
    input: &Var<'_>,
    child: &Tape,
) -> Result<Tensor<f32>, AutodiffError> {
    check_on_tape(tape, loss)?;
    check_on_tape(tape, input)?;
    if !input.requires_grad() {
        return Err(AutodiffError::GradientTrackingDisabled);
    }
    let loss_shape = loss.shape();
    if checked_numel(&loss_shape)? != 1 {
        return Err(AutodiffError::InvalidArgument(format!(
            "hessian: loss の要素数は 1 である必要がある（shape {loss_shape:?}）"
        )));
    }
    let in_shape = input.shape();
    let n = checked_numel(&in_shape)?;
    let total = checked_result_len(n, n)?;
    let result_shape = concat_shapes(&in_shape, &in_shape);
    if total == 0 {
        return Tensor::zeros(&result_shape).map_err(AutodiffError::Shape);
    }

    let cg = tape.backward_create_graph(loss, child)?;
    let mut data = vec![0.0f32; total];
    let (Some(g), Some(cx)) = (cg.grad(input)?, cg.child_var(input)?) else {
        return Tensor::new(data, &result_shape).map_err(AutodiffError::Shape);
    };
    if !g.requires_grad() {
        return Tensor::new(data, &result_shape).map_err(AutodiffError::Shape);
    }

    let elements = FlatElements::new(&g)?;
    for (j, row) in data.chunks_exact_mut(n).enumerate() {
        let g_j = elements.element(j)?;
        if !g_j.requires_grad() {
            continue;
        }
        let grads = child.backward(&g_j)?;
        if let Some(h) = grads.get(&cx)? {
            copy_grad_row(h, row)?;
        }
    }
    Tensor::new(data, &result_shape).map_err(AutodiffError::Shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_numel_detects_overflow() {
        assert_eq!(checked_numel(&[]).ok(), Some(1));
        assert_eq!(checked_numel(&[2, 3]).ok(), Some(6));
        assert!(matches!(
            checked_numel(&[usize::MAX, 2]),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
        // ゼロ長軸が後ろにあっても部分積のオーバーフローを見逃さない。
        assert!(matches!(
            checked_numel(&[usize::MAX, 2, 0]),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
    }

    #[test]
    fn checked_result_len_detects_overflow() {
        assert_eq!(checked_result_len(3, 4).ok(), Some(12));
        assert!(matches!(
            checked_result_len(usize::MAX, 2),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
    }
}
