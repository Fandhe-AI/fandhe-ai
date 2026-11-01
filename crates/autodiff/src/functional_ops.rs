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
//! **double-VJP 法の `jvp`／`jacfwd`（イシュー #2940・親 #2939。設計 §8・§19）**:
//! forward-mode の新規実装は持たず、`u` を追跡ありの葉とした `s = output ⊙ u` の 1 階勾配
//! `g = ∂s/∂input = Jᵀu`（`u` について線形）を `backward_create_graph` で子テープへ写し、
//! 第 2 段の VJP `∂(g ⊙ v)/∂u = J·v` を取る。facade の `Tape::jvp`／`Tape::jacfwd` として
//! 公開済み（イシュー #2956・設計記録 §28）。facade からは薄い 1 行委譲で呼ばれる。
//! `Tape` 以外の経路（モジュール再エクスポート・裸の自由関数・`Var`／`Tensor<f32>` 上の同名メソッド）は
//! 引き続き未承認で、facade の保留ガードが機械固定している。
//!
//! 複数入力版 `jvp_multi`（イシュー #3063・設計記録 §30）は同じ前段（`double_vjp_front`）を共用し、
//! 入力ごとの `g_i` を子テープ上でスカラー和へ畳んで `child.backward` を 1 回だけ呼ぶ。facade の
//! `Tape::jvp_multi` から 1 行委譲で呼ばれる。
//!
//! 微分可能な `jvp_create_graph`（イシュー #3067・設計記録 §31）は同じ前段に加え、2 段目の VJP を
//! `child.backward_create_graph` で孫テープへ `Var` 演算として記録し、結果 `J·v` を追跡された `Var` で返す。
//! facade への公開は #3068 で扱う（本モジュールは autodiff 内部と crate ルート再エクスポートまで）。
//!
//! **共通の契約**: `jvp_multi` 以外は単一入力・f32 の [`Tape`] のみ（`VarF64` は対象外）。`vjp`／`hvp` の結果は
//! 非微分のホスト値（`vmap` のみ同じテープ上の微分可能な `Var`。`vmap_grad`〈#3071〉は子テープで勾配を取る非微分の `Tensor<f32>`）。resident・fused 経路・checkpoint・`DeviceMismatch` は既存 `mul`／`backward`
//! の挙動をそのまま伝播する。

use crate::create_graph::CreateGraphResult;
use crate::error::AutodiffError;
use crate::jacobian_ops::{FlatElements, check_on_tape, checked_numel, copy_grad_row};
use crate::tape::{Tape, TapeId};
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
/// （クロージャ内で `backward` を呼ぶ per-sample gradient）は本関数の契約外で、子テープで勾配を取る
/// 別関数 [`vmap_grad`]（#3071・設計 §32）を使う。
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

/// 子テープで勾配を取るクロージャ型の `vmap`（`torch.func` の `vmap(grad(f))` 相当のループ版。
/// イシュー #3071・設計記録 `docs/autodiff-functional-transforms-design.md` §32）。結果は非微分。
///
/// `input` を `in_dim` 軸でサンプルへ分け、サンプルごとに呼び出し側が渡した空の子テープ `child` へ
/// スライスを追跡ありの葉として登録し、クロージャが返した 1 要素の損失を内部で [`Tape::backward`] して
/// スライス葉の勾配を dim 0 へ積んだ `Tensor<f32>`（shape は `[B] ++ slice_shape`）を返す。
/// 新規 `Op`・VJP・variant は足さない。facade への委譲は #3072。
///
/// **子テープの契約**: 呼び出し時に空であること。サンプルごとに `reset` のあと入口の葉プレフィックス状態へ
/// 戻すので、戻り時（成功・失敗とも）の `child` は空で、`reset` はサンプル数（失敗時は開始済みの数）だけ
/// 世代を進める。ピークメモリはサンプル 1 個分のグラフ。クロージャが panic した場合の状態は契約外
/// （[`vmap`] と同じ）。HRTB のためクロージャはサンプルをまたいで `Var` を保持できず、定数は
/// サンプルごとに `t.var_no_grad` 等で登録し直す。
///
/// **入口検査（順序固定）**
///
/// Phase A（子テープ無変更）:
/// 1. `in_dim >= rank`（rank 0 を含む）→ `Err(Shape(AxisOutOfRange))`。
/// 2. `shape[in_dim] == 0` → `Err(InvalidArgument)`。
/// 3. 結果要素数のオーバーフロー → `Err(Shape(ElementCountOverflow))`。
/// 4. `child` が空でない → `Err(Backward)`。
///
/// Phase B（サンプル順。最初の違反で止める）:
/// 5. クロージャの `Err` はそのまま伝播。
/// 6. 損失が `child` の現世代に属さない → `Err(TapeMismatch)`（要素数検査より先）。
/// 7. 損失の要素数が 1 でない → `Err(InvalidArgument)`（素の `backward` は非スカラーを暗黙に総和するため先に拒否）。
/// 8. 損失が追跡なし、またはスライスへ勾配が届かない → その行は全ゼロ。
/// 9. `backward`・バックエンド・hook の `Err` はそのまま伝播。
///
/// 数値は明示ループ参照と REQ-2 統一複合判定で一致する（bit 一致は契約にしない）。経路上の Op は 1 階
/// `backward` が対応する範囲。非有限値は検査せず伝播する。微分可能版は含めない（§32.5）。共有パラメータに
/// ついてのサンプルごとの勾配は返さない（§32.8-1）。
pub fn vmap_grad<F>(
    input: &Tensor<f32>,
    in_dim: usize,
    child: &mut Tape,
    mut f: F,
) -> Result<Tensor<f32>, AutodiffError>
where
    F: for<'c> FnMut(&'c Tape, &Var<'c>) -> Result<Var<'c>, AutodiffError>,
{
    // Phase A
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
            "vmap_grad: in_dim={in_dim} の軸長が 0（空バッチは結果形状を推定できないため拒否。shape {in_shape:?}）"
        )));
    }
    let slice_shape: Vec<usize> = in_shape
        .iter()
        .enumerate()
        .filter(|&(a, _)| a != in_dim)
        .map(|(_, &d)| d)
        .collect();
    let slice_numel = checked_numel(&slice_shape)?;
    let total = slice_numel
        .checked_mul(batch)
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
    if !child.is_empty() {
        return Err(AutodiffError::Backward(format!(
            "vmap_grad: 子テープは空である必要がある（ノード数 {}）",
            child.len()
        )));
    }
    let start = child.rollback_point();

    // Phase B
    let mut data = vec![0.0f32; total];
    for i in 0..batch {
        let lo = i * slice_numel;
        let r = match data.get_mut(lo..lo + slice_numel) {
            Some(row) => vmap_grad_sample(&*child, input, in_dim, i, &slice_shape, &mut f, row),
            None => Err(AutodiffError::Backward(
                "vmap_grad: 結果バッファの行範囲が不正".to_string(),
            )),
        };
        // 成否に関わらず次サンプル・呼び出し元へ空の子テープを渡す（hook・世代・葉の持ち越しを断つ）。
        child.reset();
        child.rollback_to(start);
        r?;
    }
    let mut out_shape = Vec::with_capacity(slice_shape.len() + 1);
    out_shape.push(batch);
    out_shape.extend_from_slice(&slice_shape);
    Tensor::new(data, &out_shape).map_err(AutodiffError::Shape)
}

/// `vmap_grad` のサンプル 1 個分（スライス登録→クロージャ→検査→`backward`→行コピー）。戻り値が寿命を
/// 持たないので、呼び出し側は直後に `child.reset()`（`&mut`）を呼べる。
fn vmap_grad_sample<F>(
    child: &Tape,
    input: &Tensor<f32>,
    in_dim: usize,
    i: usize,
    slice_shape: &[usize],
    f: &mut F,
    row: &mut [f32],
) -> Result<(), AutodiffError>
where
    F: for<'c> FnMut(&'c Tape, &Var<'c>) -> Result<Var<'c>, AutodiffError>,
{
    let slice = input
        .narrow(in_dim, i, 1)
        .map_err(AutodiffError::Shape)?
        .contiguous()
        .reshape(slice_shape)
        .map_err(AutodiffError::Shape)?;
    let s = child.var(&slice);
    let loss = f(child, &s)?;
    check_on_tape(child, &loss)?;
    let numel = checked_numel(&loss.shape())?;
    if numel != 1 {
        return Err(AutodiffError::InvalidArgument(format!(
            "vmap_grad: 損失は 1 要素である必要がある（shape {:?}）",
            loss.shape()
        )));
    }
    if !loss.requires_grad() {
        return Ok(());
    }
    let grads = child.backward(&loss)?;
    if let Some(g) = grads.get(&s)? {
        copy_grad_row(g, row)?;
    }
    Ok(())
}

/// double-VJP 法の前段（`jvp`／`jacfwd` 共通。#2940）。親テープへ追跡ありの葉 `u`（全要素 1）と
/// `s = output ⊙ u` を足し、`backward_create_graph` で `g = Jᵀu` を子テープへ写して
/// `(g, child 上の u)` を返す。`g` は `u` について線形なので `u` の値は結果に影響しない
/// （全要素 1 で固定）。`J ≡ 0`（`input` へ勾配が届かない・`g` が定数）は `Ok(None)`。
/// `child_var(u)` が `None` になるのは構造上の不変条件違反で、ゼロへ丸めず `Err(Backward)` にする。
/// 親テープへは `backward_create_graph` の検査より前にちょうど 2 ノードが積まれる（失敗時も残る）。
fn double_vjp_stage<'c>(
    tape: &Tape,
    output: &Var<'_>,
    input: &Var<'_>,
    child: &'c Tape,
) -> Result<Option<(Var<'c>, Var<'c>)>, AutodiffError> {
    let (cg, cu) = double_vjp_front(tape, output, child)?;
    match cg.grad(input)? {
        Some(g) if g.requires_grad() => Ok(Some((g, cu))),
        _ => Ok(None),
    }
}

/// double-VJP 法の前段の本体（`double_vjp_stage`・`jvp_multi` 共通。#3063）。親テープへ追跡ありの葉 `u` と
/// `s = output ⊙ u` を足して `backward_create_graph` を 1 回だけ呼び、`(CreateGraphResult, child 上の u)` を返す。
/// 入力ごとの `g_i = J_iᵀu` の取り出しは呼び出し側が行う。親テープへの操作列は単一入力版と同一。
fn double_vjp_front<'c>(
    tape: &Tape,
    output: &Var<'_>,
    child: &'c Tape,
) -> Result<(CreateGraphResult<'c>, Var<'c>), AutodiffError> {
    let u = tape.var(&Tensor::ones(&output.shape()).map_err(AutodiffError::Shape)?);
    let s = output.mul(&u)?;
    let cg = tape.backward_create_graph(&s, child)?;
    let Some(cu) = cg.child_var(&u)? else {
        return Err(AutodiffError::Backward(
            "double-VJP: 余接葉 u の子テープ写しが存在しない（不変条件違反）".to_string(),
        ));
    };
    Ok((cg, cu))
}

/// 列 `col`（長さ `m`）を行優先の結果 `data`（`m × n`）の第 `k` 列へ書く。長さ不一致は型付きエラー。
fn write_column(
    data: &mut [f32],
    n: usize,
    k: usize,
    col: &Tensor<f32>,
) -> Result<(), AutodiffError> {
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

/// ヤコビアン・ベクトル積 `J·v`（`J = ∂output/∂input`）を double-VJP 法で求める
/// （イシュー #2940。`jacobian_ops::jacobian` の `J·v` と REQ-2 統一複合判定で一致する）。
/// 戻り値は shape が `output.shape()` の非微分ホスト値。`child` は呼び出し側が
/// [`Tape::new_with_ops`] 等で構築した空の子テープ（[`Tape::backward_create_graph`] と同じ契約）で、
/// 呼び出し後は再利用しない。
///
/// **入口検査（ノードを足す前。順序固定。失敗時は親・子テープとも無変更）**:
/// 1. `output`／`input` が `tape` の現世代に属さない → `Err(TapeMismatch)`。
/// 2. `input.requires_grad() == false` → `Err(GradientTrackingDisabled)`。
/// 3. `tangent.shape() != input.shape()` → `Err(Shape(ShapeMismatch))`（ブロードキャスト不可）。
/// 4. 要素数を `checked_numel` で検査（`Err(Shape(ElementCountOverflow))`）。
/// 5. 要素数 0、または `output.requires_grad() == false` → テープに触れず全ゼロ（`child` も検査しない。
///    `jacobian`／`vjp` と同じ意味論で、追跡なし loss を `Err` にする `hvp` とは非対称）。
///
/// 以降は `backward_create_graph` の既存検査（`supports_create_graph() == false` の Op・rank 3 以上の
/// `MatMul`・非空の子テープ・同一テープ・デバイス不一致・checkpoint 登録済み）をそのまま `Err(Backward)`
/// 等で伝播する（新しい拒否ロジック・variant は足さない）。拒否時 `child` は空のまま。
///
/// **副作用**: 前段以降の失敗でも親テープへちょうど 2 ノード（全要素 1 の追跡あり葉 `u` と `mul`）が残る
/// （既存ノードの値は不変）。成功時の子テープには写し・1 階勾配・第 2 段の補助ノードが残る。
/// 子テープ上の数値は 1 階 VJP や `jacobian` と bit 同一を主張しない。非有限値は検査せず伝播する。
/// 計算量は親 backward 1 回＋子 backward 1 回。f32 の [`Tape`] 用（f64 は `f64_autograd::TapeF64::jvp`、f16 は対象外）。
pub fn jvp(
    tape: &Tape,
    output: &Var<'_>,
    input: &Var<'_>,
    tangent: &Tensor<f32>,
    child: &Tape,
) -> Result<Tensor<f32>, AutodiffError> {
    check_on_tape(tape, output)?;
    check_on_tape(tape, input)?;
    if !input.requires_grad() {
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
    if n == 0 || m == 0 || !output.requires_grad() {
        return Tensor::zeros(&out_shape).map_err(AutodiffError::Shape);
    }

    let mut data = vec![0.0f32; m];
    if let Some((g, cu)) = double_vjp_stage(tape, output, input, child)? {
        let prod = g.mul(&child.var_no_grad(tangent))?;
        let grads = child.backward(&prod)?;
        if let Some(h) = grads.get(&cu)? {
            copy_grad_row(h, &mut data)?;
        }
    }
    Tensor::new(data, &out_shape).map_err(AutodiffError::Shape)
}

/// ヤコビアン全体 `J = ∂output/∂input` を double-VJP 法で列ごとに求める（イシュー #2940。
/// `jacobian_ops::jacobian` と REQ-2 統一複合判定で一致する）。戻り値は shape が
/// `output.shape ++ input.shape` の非微分ホスト値。`child` の契約は [`jvp`] と同じ。
///
/// 前段（`g = Jᵀu`）を 1 回だけ作り、子テープ上で `g` の要素 `g_k = ∂(uᵀ output)/∂input_k` ごとに
/// `child.backward` を回すと `∂g_k/∂u = J[:, k]`（第 2 段の VJP を列ごとに適用）が得られる。
/// `input` の one-hot 接ベクトルで `jvp` を `n` 回呼ぶ方式と数学的に同じだが、子テープが 1 本で済み
/// 親テープのノードも 2 個で済むため採らない。計算量は親 backward 1 回＋子 backward `n` 回。
///
/// **入口検査**は [`jvp`] と同じ（接ベクトルの shape 検査を除く）に加え、`m × n` を検査付き乗算で求める。
/// 追跡なし `output`・要素数 0 はテープに触れず全ゼロ。副作用・拒否時の契約も [`jvp`] と同じ。
pub fn jacfwd(
    tape: &Tape,
    output: &Var<'_>,
    input: &Var<'_>,
    child: &Tape,
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
    let total = m
        .checked_mul(n)
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
    let mut result_shape = out_shape;
    result_shape.extend_from_slice(&in_shape);
    if total == 0 || !output.requires_grad() {
        return Tensor::zeros(&result_shape).map_err(AutodiffError::Shape);
    }

    let mut data = vec![0.0f32; total];
    if let Some((g, cu)) = double_vjp_stage(tape, output, input, child)? {
        let elements = FlatElements::new(&g)?;
        for k in 0..n {
            let g_k = elements.element(k)?;
            if !g_k.requires_grad() {
                continue;
            }
            let grads = child.backward(&g_k)?;
            if let Some(col) = grads.get(&cu)? {
                write_column(&mut data, n, k, col)?;
            }
        }
    }
    Tensor::new(data, &result_shape).map_err(AutodiffError::Shape)
}

/// 複数入力のヤコビアン・ベクトル積 `J·v = Σ_i J_i·v_i`（`J_i = ∂output/∂inputs[i]`）を double-VJP 法で求める
/// （イシュー #3063・親 #3062。設計記録 §30。facade の `Tape::jvp_multi` から 1 行委譲で呼ばれる。
/// `torch.func.jvp` に複数の primals／tangents を渡す形に相当）。戻り値は shape が `output.shape()` の
/// 非微分ホスト値。`child` の契約は [`jvp`] と同じ（空の子テープ。呼び出し後は再利用しない）。
///
/// **入口検査（ノードを足す前。順序固定。違反が複数あれば最初のものを返す。失敗時は親・子テープとも無変更）**:
/// 1. `inputs.len() != tangents.len()` → `Err(InvalidArgument)`。
/// 2. `inputs` が空 → `Err(InvalidArgument)`。
/// 3. `output`、続いて `inputs` を添字順に現世代検査 → `Err(TapeMismatch)`。
/// 4. 追跡なしの入力（添字順）→ `Err(GradientTrackingDisabled)`。
/// 5. 同一 `Var` の重複指定 → `Err(InvalidArgument)`（二重計上の誤用を防ぐ fail-closed）。
/// 6. `tangents[i].shape() != inputs[i].shape()`（添字順。ブロードキャスト不可）→ `Err(Shape(ShapeMismatch))`。
/// 7. 要素数を `checked_numel` で検査（`Err(Shape(ElementCountOverflow))`）。
/// 8. 出力の要素数 0・`output.requires_grad() == false`・全入力の要素数 0 → テープに触れず全ゼロ
///    （`child` も検査しない）。
///
/// 入力が 1 つのときはエラーの種類・順序とも [`jvp`] と一致する。
///
/// **算法**: 親 backward は `backward_create_graph` 1 回のみ。`g_i = J_iᵀu` を入力ごとに取り出し、
/// 子テープ上で `p_i = Σ(g_i ⊙ v_i)`（スカラー）を `add` で畳み込んだ合計を `child.backward` 1 回に渡す
/// （`∂/∂u = Σ J_i·v_i`）。計算量は入力数に依らず親 1 回＋子 1 回。寄与する入力が無ければ全ゼロ。
///
/// **副作用**: 入口検査後の失敗でも親テープへちょうど 2 ノード（`u` と `mul`）が残る（入力数に依らない）。
/// 成功時の子テープには写し・1 階勾配・第 2 段の補助ノードが残る。結果は REQ-2 統一複合判定で
/// `jacobian` 由来の `Σ J_i·v_i` と一致し、入力 1 つでも [`jvp`] と bit 同一は主張しない。
/// 非有限値は検査せず伝播する。f32 の [`Tape`] 用（f64 は `f64_autograd::TapeF64::jvp_multi`）。
pub fn jvp_multi(
    tape: &Tape,
    output: &Var<'_>,
    inputs: &[&Var<'_>],
    tangents: &[&Tensor<f32>],
    child: &Tape,
) -> Result<Tensor<f32>, AutodiffError> {
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
    check_on_tape(tape, output)?;
    for input in inputs {
        check_on_tape(tape, input)?;
    }
    if inputs.iter().any(|v| !v.requires_grad()) {
        return Err(AutodiffError::GradientTrackingDisabled);
    }
    for (i, a) in inputs.iter().enumerate() {
        if inputs[..i].iter().any(|b| b.node_id() == a.node_id()) {
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
    if m == 0 || all_empty || !output.requires_grad() {
        return Tensor::zeros(&out_shape).map_err(AutodiffError::Shape);
    }

    let mut data = vec![0.0f32; m];
    let (cg, cu) = double_vjp_front(tape, output, child)?;
    let mut total: Option<Var<'_>> = None;
    for (input, tangent) in inputs.iter().zip(tangents) {
        if tangent.shape().iter().product::<usize>() == 0 {
            continue;
        }
        let Some(g) = cg.grad(input)? else { continue };
        if !g.requires_grad() {
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
            copy_grad_row(h, &mut data)?;
        }
    }
    Tensor::new(data, &out_shape).map_err(AutodiffError::Shape)
}
/// [`jvp_create_graph`] の結果（イシュー #3067・設計記録 §31.1）。孫テープ上の追跡された `J·v` と、
/// 元テープの `Var` から孫テープ上の写し（親→子→孫の 2 段合成）を引く対応表を持つ。
/// 公開型に出るライフタイムは孫テープの `'g` だけ（1 段目の子テープは関数内に閉じる）。
#[derive(Debug)]
pub struct JvpCreateGraphResult<'g> {
    output: Var<'g>,
    parent_id: TapeId,
    parent_epoch: u64,
    /// 親 `NodeId.0` を添字とする孫テープ上の写し（`J ≡ 0` 等のゼロ経路では空）。
    mirror: Vec<Option<Var<'g>>>,
    /// テスト専用: 2 段目の余接葉 `u` の 1 階勾配（`jvp` との bit 同一性検査用。公開しない）。
    #[cfg(test)]
    stage2_cotangent_grad: Option<Tensor<f32>>,
}

impl<'g> JvpCreateGraphResult<'g> {
    /// `J·v`（shape は元の `output` と同じ）を孫テープ上の `Var` として返す。
    /// 写しの入力へ `grandchild.backward` で微分できる。出力が入力に依存しない場合
    /// （線形・`var_no_grad` の係数のみ等）は追跡なし（`requires_grad() == false`）になりうる。
    pub fn output(&self) -> Var<'g> {
        self.output
    }

    /// 元テープ上の `parent_var` の孫テープ上の写しを返す。`output()` を `grandchild.backward` で
    /// 微分した勾配はこの写しをキーに読む（元の `x` とは直接つながらない。§31.2）。
    ///
    /// - テープ・世代の不一致 → `Err(TapeMismatch)`、追跡なし → `Err(GradientTrackingDisabled)`
    ///   （[`CreateGraphResult::child_var`] と同じ検査順）。
    /// - `Ok(None)` は `J·v` がその入力に恒等的に依存しない（勾配ゼロ）ことを意味する。
    pub fn child_var(&self, parent_var: &Var<'_>) -> Result<Option<Var<'g>>, AutodiffError> {
        if parent_var.tape_id() != self.parent_id || parent_var.tape_epoch() != self.parent_epoch {
            return Err(AutodiffError::TapeMismatch);
        }
        if !parent_var.requires_grad() {
            return Err(AutodiffError::GradientTrackingDisabled);
        }
        Ok(self.mirror.get(parent_var.node_id().0).copied().flatten())
    }

    #[cfg(test)]
    pub(crate) fn stage2_cotangent_grad_host(&self) -> Option<&Tensor<f32>> {
        self.stage2_cotangent_grad.as_ref()
    }
}

/// 全ゼロの `J·v` を孫テープの追跡なし葉として返すゼロ経路（写しは空）。
fn zero_jvp_result<'g>(
    tape: &Tape,
    grandchild: &'g Tape,
    out_shape: &[usize],
) -> Result<JvpCreateGraphResult<'g>, AutodiffError> {
    let zeros = Tensor::zeros(out_shape).map_err(AutodiffError::Shape)?;
    Ok(JvpCreateGraphResult {
        output: grandchild.var_no_grad(&zeros),
        parent_id: tape.id,
        parent_epoch: tape.epoch(),
        mirror: Vec::new(),
        #[cfg(test)]
        stage2_cotangent_grad: None,
    })
}

/// 追跡された `Var` を返す微分可能な `jvp`（イシュー #3067・設計記録 §31.2〜§31.4・§31.6）。
/// 前段は [`jvp`] と同じ double-VJP 法で、2 段目の VJP だけ `child.backward_create_graph` で
/// 孫テープ `grandchild` に記録する。結果 `J·v` は孫テープ上の `Var` で、元の入力への微分は
/// [`JvpCreateGraphResult::child_var`] の写し（親→子→孫の 2 段合成）を通して得る。
/// facade の `Tape::jvp_create_graph`（#3068）から 1 行委譲で呼ばれる予定。
///
/// 入口検査（ノードを足す前。順序固定）: 1〜4 は [`jvp`] と同じ（テープ所属・追跡あり入力・
/// 接ベクトル shape・要素数）。5 は 3 本のテープ関係（`child`・`grandchild` は互いに別で `tape` とも別、
/// 同デバイス、どちらも空）で、違反は `Err(Backward)`／`Err(DeviceMismatch)`。
/// 6 は追跡なし `output`・要素数 0 のゼロ返却（親・`child` は無変更、`grandchild` に全ゼロの葉が 1 個）。
/// [`jvp`] のゼロ経路は `child` を検査しないが、本関数は 5 をゼロ返却より先に行う。
///
/// 以降は [`jvp`] と同じ拒否（非対象 Op・rank 3 以上の `matmul`・checkpoint 等）が 1 段目に、
/// `backward_create_graph` の既存検査が 2 段目に `Err(Backward)` で伝播する。親テープへは前段で
/// ちょうど 2 ノードが残る。`J ≡ 0` は全ゼロの追跡なし葉を返す。非有限値は検査せず伝播する。
/// f32 の [`Tape`] 用の単一入力版（f64 は `f64_autograd::TapeF64::jvp_create_graph`、f16 は対象外）。
pub fn jvp_create_graph<'g>(
    tape: &Tape,
    output: &Var<'_>,
    input: &Var<'_>,
    tangent: &Tensor<f32>,
    child: &Tape,
    grandchild: &'g Tape,
) -> Result<JvpCreateGraphResult<'g>, AutodiffError> {
    check_on_tape(tape, output)?;
    check_on_tape(tape, input)?;
    if !input.requires_grad() {
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

    if child.id == tape.id {
        return Err(AutodiffError::Backward(
            "jvp_create_graph: 子テープに親と同一の Tape は指定できない".into(),
        ));
    }
    if grandchild.id == tape.id || grandchild.id == child.id {
        return Err(AutodiffError::Backward(
            "jvp_create_graph: 孫テープは親・子と異なる Tape である必要がある".into(),
        ));
    }
    for other in [child, grandchild] {
        if other.device() != tape.device() {
            return Err(AutodiffError::DeviceMismatch {
                requested: other.device(),
                actual: tape.device(),
            });
        }
    }
    if !child.is_empty() || !grandchild.is_empty() {
        return Err(AutodiffError::Backward(
            "jvp_create_graph: 子テープ・孫テープは空である必要がある".into(),
        ));
    }
    if n == 0 || m == 0 || !output.requires_grad() {
        return zero_jvp_result(tape, grandchild, &out_shape);
    }

    let (cg1, cu) = double_vjp_front(tape, output, child)?;
    let g = match cg1.grad(input)? {
        Some(g) if g.requires_grad() => g,
        _ => return zero_jvp_result(tape, grandchild, &out_shape),
    };
    let prod = g.mul(&child.var_no_grad(tangent))?;
    let cg2 = child.backward_create_graph(&prod, grandchild)?;
    let Some(r) = cg2.grad(&cu)? else {
        return Err(AutodiffError::Backward(
            "jvp_create_graph: 余接葉の 2 段目の勾配が存在しない（不変条件違反）".to_string(),
        ));
    };

    // 追跡なしの親ノードの写しは `var_no_grad` の葉で `child_var` が Err になるため、追跡ありだけ合成する。
    let mut mirror: Vec<Option<Var<'g>>> = vec![None; cg1.mirror().len()];
    for (slot, cv) in mirror.iter_mut().zip(cg1.mirror()) {
        if let Some(cv) = cv
            && cv.requires_grad()
        {
            *slot = cg2.child_var(cv)?;
        }
    }

    #[cfg(test)]
    let stage2 = {
        let mut data = vec![0.0f32; m];
        if let Some(h) = cg2.first_order().get(&cu)? {
            copy_grad_row(h, &mut data)?;
        }
        Tensor::new(data, &out_shape).map_err(AutodiffError::Shape)?
    };
    Ok(JvpCreateGraphResult {
        output: r,
        parent_id: tape.id,
        parent_epoch: tape.epoch(),
        mirror,
        #[cfg(test)]
        stage2_cotangent_grad: Some(stage2),
    })
}
