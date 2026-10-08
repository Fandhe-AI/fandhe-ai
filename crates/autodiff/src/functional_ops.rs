//! 関数型 AD ラッパー `vjp`・`hvp`（イシュー #2874／#2875・親 #2841。契約の正は
//! `docs/autodiff-functional-transforms-design.md` §3〜§7・§10）。
//!
//! **新規 `Op`・`BackendOps` メソッド・VJP・`AutodiffError` variant はゼロ**:
//! 勾配追跡なしの余接定数葉 `u` と `output.mul(u)` を 1 本足し、既存の
//! [`Tape::backward`] を 1 回呼ぶだけの reverse-mode 合成（PyTorch
//! `torch.autograd.grad(outputs, inputs, grad_outputs=u)` 相当）。`Tape::backward` は
//! 非スカラー loss に全要素 1 のシードを使い、これが暗黙の総和射影になるため
//! `sum` ノードは足さない（`jacobian_ops` の `FlatElements::element` と同じ理由）。
//! 既存の Op ごとの VJP ディスパッチャ（`grad.rs` の `pub(crate) fn vjp`）とは
//! 別物で、本モジュールは「出力と余接ベクトルの組」を受ける利用者向けの合成。
//!
//! **公開形は未承認（保留）**: facade（`fandhe_ai`）へは公開しない。保留は facade の
//! `FunctionalTransformsHoldDoctestGuard` と `tests/api_surface.rs` の否定ガードで
//! 機械固定している。`hvp`（#2875）は `backward_create_graph` で子テープへ 1 階勾配を
//! 写し、子テープ上で `g ⊙ v` を `child.backward` する合成（既存の手組み HVP と同じ形）。
//! ループ版 `vmap` も本モジュールへ入る予定で、検査ヘルパー（`jacobian_ops` の `checked_numel`・`check_on_tape`・`copy_grad_row`）を共用する。
//!
//! **共通の契約**: 単一入力・f32 の [`Tape`] のみ（`VarF64` は対象外）。結果は非微分の
//! ホスト値。resident・fused 経路・checkpoint・`DeviceMismatch` は既存 `mul`／`backward`
//! の挙動をそのまま伝播する。

use crate::error::AutodiffError;
use crate::jacobian_ops::{check_on_tape, checked_numel, copy_grad_row};
use crate::tape::Tape;
use crate::var::Var;
use fandhe_ai_tensor_core::{ShapeError, Tensor};

/// ベクトル・ヤコビアン積 `Σ_i cotangent[i] · ∂output[i]/∂input`（転置ヤコビアン積
/// `Jᵀu`）。戻り値は shape が `input.shape()` の非微分ホスト値。
///
/// **入口検査（テープへノードを足す前。順序固定）**:
/// 1. `output`／`input` が `tape` の現世代に属さない → `Err(TapeMismatch)`。
/// 2. `input.requires_grad() == false` → `Err(GradientTrackingDisabled)`。
/// 3. `cotangent.shape() != output.shape()` → `Err(Shape(ShapeMismatch))`
///    （ブロードキャストは許さない。`[1]` が `[3]` に黙って通るのを防ぐ）。
/// 4. 要素数を `checked_numel` で検査（オーバーフローは `Err(Shape(ElementCountOverflow))`）。
/// 5. 要素数 0、または `output` が `input` に構造的に依存しない
///    （`output.requires_grad() == false`）→ テープに触れず全ゼロを返す
///    （素の `backward` は追跡なしの loss を `Err` にするため先に分岐する）。
///
/// 本体はテープへちょうど 2 ノード（余接の定数葉と `mul`）を足し、`backward` を 1 回呼ぶ。
/// `backward` 等が途中で `Err` を返してもこの 2 ノードは残る（既存ノードの値は不変）。
/// 勾配が `input` へ届かない場合は全ゼロ。`cotangent` の非有限値は検査せずそのまま伝播する。
pub fn vjp(
    tape: &Tape,
    output: &Var<'_>,
    input: &Var<'_>,
    cotangent: &Tensor<f32>,
) -> Result<Tensor<f32>, AutodiffError> {
    check_on_tape(tape, output)?;
    check_on_tape(tape, input)?;
    if !input.requires_grad() {
        return Err(AutodiffError::GradientTrackingDisabled);
    }
    let out_shape = output.shape();
    if cotangent.shape() != out_shape.as_slice() {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: out_shape,
            rhs: cotangent.shape().to_vec(),
        }));
    }
    let in_shape = input.shape();
    let n = checked_numel(&in_shape)?;
    let m = checked_numel(&out_shape)?;
    if n == 0 || m == 0 || !output.requires_grad() {
        return Tensor::zeros(&in_shape).map_err(AutodiffError::Shape);
    }

    let u = tape.var_no_grad(cotangent);
    let weighted = output.mul(&u)?;
    let grads = tape.backward(&weighted)?;
    let mut data = vec![0.0f32; n];
    if let Some(g) = grads.get(input)? {
        copy_grad_row(g, &mut data)?;
    }
    Tensor::new(data, &in_shape).map_err(AutodiffError::Shape)
}

/// Hessian-vector product `Σ_j v_j · ∂g_j/∂input`（`g = ∂loss/∂input`。C² の範囲では
/// `H·v`）。戻り値は shape が `input.shape()` の非微分ホスト値。`child` は呼び出し側が
/// [`Tape::new_with_ops`] 等で構築した空の子テープ（[`Tape::backward_create_graph`] と
/// 同じ契約）。`jacobian_ops::hessian` と `v` の積と REQ-2 統一複合判定で一致する。
///
/// **入口検査（`backward_create_graph` を呼ぶ前。失敗時は親・子テープとも無変更。順序固定）**:
/// 1. `loss`／`input` が `tape` の現世代に属さない → `Err(TapeMismatch)`。
/// 2. `input.requires_grad() == false` → `Err(GradientTrackingDisabled)`。
/// 3. `loss` の要素数が 1 でない → `Err(InvalidArgument)`（`[]`・`[1]`・`[1, 1]` は可）。
/// 4. `vector.shape() != input.shape()` → `Err(Shape(ShapeMismatch))`（ブロードキャスト不可）。
/// 5. `input` の要素数 0 → 空テンソル（`child` は検査も変更もしない）。
///
/// 以降は `backward_create_graph` の既存検査（`supports_create_graph() == false` の Op・
/// rank 3 以上の `MatMul`・非空の子テープ・デバイス不一致・追跡なし `loss` 等）をそのまま
/// 伝播する（新しい検査・variant は足さない）。追跡なしの `loss` は `vjp` が全ゼロを返すのと
/// 異なり `hessian` と同じく `Err` になる。1 階勾配が `input` へ届かない、または定数
/// （`input` に線形な `loss`）の場合は全ゼロ。
///
/// 親テープへノードは足さない。子テープには写し・1 階勾配に加えてちょうど 2 ノード
/// （`vector` の定数葉と `mul`）が残り、途中で `Err` でもそれまでの分は残る。呼び出し後の
/// `child` は再利用せず作り直す。`mul` 結果は非スカラーだが `Tape::backward` が全要素 1 の
/// シードを使うため暗黙に総和され、`sum` ノードは足さない（`vjp` と同じ理由）。
/// `vector` の非有限値は検査せず伝播する。子テープ上の数値方式は 1 階 VJP と bit 同一を
/// 主張しない（`create_graph` の既存契約）。
pub fn hvp(
    tape: &Tape,
    loss: &Var<'_>,
    input: &Var<'_>,
    vector: &Tensor<f32>,
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
            "hvp: loss の要素数は 1 である必要がある（shape {loss_shape:?}）"
        )));
    }
    let in_shape = input.shape();
    if vector.shape() != in_shape.as_slice() {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: in_shape,
            rhs: vector.shape().to_vec(),
        }));
    }
    let n = checked_numel(&in_shape)?;
    if n == 0 {
        return Tensor::zeros(&in_shape).map_err(AutodiffError::Shape);
    }

    let cg = tape.backward_create_graph(loss, child)?;
    let (Some(g), Some(cx)) = (cg.grad(input)?, cg.child_var(input)?) else {
        return Tensor::zeros(&in_shape).map_err(AutodiffError::Shape);
    };
    if !g.requires_grad() {
        return Tensor::zeros(&in_shape).map_err(AutodiffError::Shape);
    }
    let vc = child.var_no_grad(vector);
    let prod = g.mul(&vc)?;
    let grads = child.backward(&prod)?;
    let mut data = vec![0.0f32; n];
    if let Some(h) = grads.get(&cx)? {
        copy_grad_row(h, &mut data)?;
    }
    Tensor::new(data, &in_shape).map_err(AutodiffError::Shape)
}
