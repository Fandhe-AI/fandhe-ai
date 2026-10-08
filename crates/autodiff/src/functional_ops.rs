//! 関数型 AD ラッパー `vjp`・`hvp`・`vmap`（イシュー #2874／#2875／#2876・親 #2841。契約の正は
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
//! **公開状況**: facade では `Tape::vjp`／`Tape::hvp`／`Tape::vmap` として公開済み（イシュー #2931・
//! 設計記録 `docs/autodiff-functional-transforms-design.md` §23〜§24。本モジュールの自由関数への薄い委譲）。
//! モジュール自体の再エクスポート・裸の自由関数・`Var`／`Tensor<f32>` 上の同名メソッドは引き続き未承認で、
//! facade の `FunctionalTransformsHoldDoctestGuard` と `tests/api_surface.rs` のガードが機械固定している。
//! `hvp`（#2875）は `backward_create_graph` で子テープへ 1 階勾配を
//! 写し、子テープ上で `g ⊙ v` を `child.backward` する合成（既存の手組み HVP と同じ形）。
//! ループ版 `vmap`（#2876）は `unbind`→クロージャ適用→`contiguous`→`stack` の合成で、
//! 検査ヘルパー（`jacobian_ops` の `checked_numel`・`check_on_tape`・`copy_grad_row`）を共用する。
//!
//! **共通の契約**: 単一入力・f32 の [`Tape`] のみ（`VarF64` は対象外）。`vjp`／`hvp` の結果は
//! 非微分のホスト値（`vmap` のみ同じテープ上の微分可能な `Var`）。resident・fused 経路・checkpoint・`DeviceMismatch` は既存 `mul`／`backward`
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

/// ループ版 `vmap`（`torch.func.vmap` 相当の意味論）。`input` を `in_dim` 軸で `unbind` し、
/// 各スライスへクロージャ `f` を順に適用して、結果を先頭軸（dim 0）へ `stack` した
/// 同じテープ上の微分可能な [`Var`] を返す。新規 `Op`・VJP・variant は足さず、既存の
/// `unbind`・`contiguous`・`stack` の合成のみ。性能は保証しない（ループ実行）。
///
/// 前提: `f` は副作用のない関数であること。バッチなしで実行した結果との一致は REQ-2 の
/// 統一複合判定で見る（bit 一致は契約にしない。設計 §11-2）。`vmap(grad)`
/// （クロージャ内で `backward` を呼ぶ per-sample gradient）は契約外で、値だけが必要なら
/// 呼び出し側が明示ループで `backward` を回す（設計 §5・§11-6）。
///
/// **入口検査（順序固定）**
///
/// Phase A（テープ無変更。失敗しても `tape.len()` は不変）:
/// 1. `input` が `tape` の現世代に属さない → `Err(TapeMismatch)`。
/// 2. `in_dim >= rank`（rank 0 を含む）→ `Err(Shape(AxisOutOfRange))`。
/// 3. `shape[in_dim] == 0`（空バッチ）→ `Err(InvalidArgument)`。結果形状は推定しない。
///    `unbind` は零長軸に `Ok(vec![])` を返すため、`unbind` より前に自前で検査する。
///
/// Phase B（ノードが積まれる段階。`vmap` 自身の後処理〈`contiguous`・`stack`〉の前に全検査を終える）:
/// 4. 各スライスについて `f` を呼び、`Err` はそのまま伝播する。
/// 5. 各戻り値が `tape` の現世代に属さない → `Err(TapeMismatch)`（形状検査より先）。
/// 6. 戻り値の shape が先頭出力と異なる → `Err(Shape(ShapeMismatch))`。
/// 7. 結果の要素数 `B × numel(out_0)` を検査（`Err(Shape(ElementCountOverflow))`）。
///
/// **テープに残るノード**: Phase A の失敗では何も残らない。Phase B で失敗した場合
/// （`f` の `Err`・別テープの出力・形状不一致・`contiguous`／`stack` の失敗〈`DeviceMismatch` 等の伝播〉）、
/// `unbind` が積んだノード（スライスあたり最大 3）、それまでに `f` が積んだノード、
/// `contiguous` 化の途中までのノードが残る。既存ノードの値は変えない。
///
/// 非 contiguous なクロージャ出力は `contiguous` で materialize してから `stack` する
/// （`stack` は非 contiguous 要素を `NonContiguousReshape` で拒否するため）。
/// 出力軸は先頭固定で `out_dim`・複数入力は扱わない（設計 §11-1）。
pub fn vmap<'t, F>(
    tape: &'t Tape,
    input: &Var<'t>,
    in_dim: usize,
    mut f: F,
) -> Result<Var<'t>, AutodiffError>
where
    F: FnMut(&Var<'t>) -> Result<Var<'t>, AutodiffError>,
{
    // Phase A
    check_on_tape(tape, input)?;
    let in_shape = input.shape();
    let rank = in_shape.len();
    if in_dim >= rank {
        return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            axis: in_dim,
            rank,
        }));
    }
    let batch = in_shape[in_dim];
    if batch == 0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "vmap: in_dim={in_dim} の軸長が 0（空バッチは結果形状を推定できないため拒否。shape {in_shape:?}）"
        )));
    }

    // Phase B
    let slices = input.unbind(in_dim)?;
    let mut outs: Vec<Var<'t>> = Vec::with_capacity(slices.len());
    for s in &slices {
        let out = f(s)?;
        check_on_tape(tape, &out)?;
        if let Some(first) = outs.first() {
            let (lhs, rhs) = (first.shape(), out.shape());
            if lhs != rhs {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch { lhs, rhs }));
            }
        }
        outs.push(out);
    }
    let out_numel = checked_numel(&outs[0].shape())?;
    out_numel
        .checked_mul(outs.len())
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
    let contiguous: Vec<Var<'t>> = outs
        .iter()
        .map(|o| o.contiguous())
        .collect::<Result<_, _>>()?;
    Var::stack(&contiguous, 0)
}
