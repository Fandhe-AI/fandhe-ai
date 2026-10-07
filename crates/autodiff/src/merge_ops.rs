//! 結合 4 演算 `merge_concatenate`・`merge_add`・`merge_multiply`・`merge_average`
//! （イシュー #2666・親 #2663・ルート #2499「Phase 4」）。
//!
//! Keras Functional API の結合層（`Concatenate`／`Add`／`Multiply`／`Average`）に対応する
//! 「複数の `Var` を 1 つへ合流させる」数値本体。呼び出し元は `crates/facade` の
//! Functional グラフ（`compat::functional`。`#[cfg(test)]` 隔離の内部実装）の結合ノードで、
//! compat 層へ数値ロジックを持ち込まない（REQ-9 の「薄いラッパー」。
//! `docs/facade-functional-api-decision.md` §6・§11）ためここへ置く。
//!
//! **新規 `Op`・`BackendOps` メソッド・VJP はゼロ**: 既存の `Var::cat`（`Op::Concat`）・
//! `Var::add`・`Var::mul`・`Var::div` の合成だけで構成する。CUDA／Metal へは既存演算の
//! 既定フォールバック経由で到達でき、GPU 専用カーネルは持たない。
//!
//! **合成表**
//! - `merge_concatenate`: `Var::cat(inputs, dim)` 1 回。
//! - `merge_add`: `Var::add` の index 順の左畳み込み（`((x0 + x1) + x2) + …`）。
//! - `merge_multiply`: `Var::mul` の index 順の左畳み込み。
//! - `merge_average`: `merge_add` の後に入力数 `n` の定数で 1 回 `Var::div`。定数は
//!   `Tape::var_no_grad` の葉で shape は `[1; rank]`（rank 0 なら `[]`）とし、出力 rank・
//!   shape を入力と同一に保つ。`stack` + `mean` は非 contiguous 入力を拒否するため使わない。
//!
//! **検証順（tape へノードを積む前にすべて完了し、引数起因のエラーで孤児ノードを残さない）**
//! 1. 入力 2 件以上（4 種共通。違反は `InvalidArgument`。メッセージは件数のみ）
//! 2. 先頭基準の同一 tape 検査（`TapeMismatch`）
//! 3. Add／Multiply／Average は全入力 shape の完全一致（`ShapeError::ShapeMismatch`）。
//!    `Var::add`／`Var::mul` は broadcast を受理するため、最初の演算より前に検査して
//!    broadcast を黙って通さない。Concatenate の shape 検査は `Var::cat` に委譲する
//! 4. Average は `n <= 2^24`（`n as f32` が厳密に表せる範囲。違反は `InvalidArgument`）
//!
//! 同一 `Var` の重複（`merge_add(&[x, x])`）は数式として自然なため本層では拒否しない
//! （結線の検証は facade のグラフビルダーの責務）。
//!
//! **公開形（未承認・保留）**: facade（`fandhe_ai`）への公開形は未承認で、承認依頼は #2677・
//! 公開自体は承認後の #2679。推奨案は `docs/facade-functional-api-decision.md` §17（推奨案の
//! 記録であり承認記録ではない）。保留中は `MergeOpsHoldDoctestGuard`
//! （`crates/facade/src/lib.rs`）と `crates/facade/tests/api_surface.rs` の否定ガードが
//! facade への漏出を拒否する。
//!
//! **数値契約**: 高々数個の要素ごと演算の合成で、長軸縮約（f64 アキュムレータ契約）の対象外。
//! FMA 契約・REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）は不変で、
//! PyTorch 2.14.0 の実行値 fixture（`tests/fixtures/merge-ops-pytorch-reference/`）と突合する。
//!
//! **PyTorch／Keras との意図した差分**: broadcast 拒否（shape 完全一致のみ）・入力 1 件拒否・
//! 軸は非負の `usize` のみ（負の `dim` 非対応）。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03・A04）**: 本番経路に
//! `unwrap`／`expect`／添字 panic を置かず、評価は反復のみで再帰しない。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::var::Var;

/// Average の入力数上限。`n as f32` が厳密に表せる最大の連続整数 `2^24`。
const MAX_AVERAGE_INPUTS: usize = 1 << 24;

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::InvalidArgument(msg)
}

/// 共通検証（件数 2 以上・同一 tape）。先頭要素のコピーを返す（`Var` は `Copy`）。
fn validate_common<'t>(op: &str, inputs: &[Var<'t>]) -> Result<Var<'t>, AutodiffError> {
    if inputs.len() < 2 {
        return Err(invalid(format!(
            "{op}: 入力は 2 件以上が必要（件数 {}）",
            inputs.len()
        )));
    }
    let Some(first) = inputs.first().copied() else {
        return Err(invalid(format!("{op}: 入力が空")));
    };
    for v in inputs.iter().skip(1) {
        first.check_same_tape(v)?;
    }
    Ok(first)
}

/// 全入力 shape が先頭と完全一致すること（broadcast を通さない）。
fn validate_same_shape(first: &Var<'_>, inputs: &[Var<'_>]) -> Result<(), AutodiffError> {
    let expected = first.shape();
    for v in inputs.iter().skip(1) {
        let shape = v.shape();
        if shape != expected {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: expected,
                rhs: shape,
            }));
        }
    }
    Ok(())
}

/// 入力を軸 `dim` で連結する（`torch.cat` 相当・2 件以上）。
///
/// shape 検査（rank 不一致・`dim` 範囲外・非連結軸の不一致）は `Var::cat` に委譲する。
pub fn merge_concatenate<'t>(inputs: &[Var<'t>], dim: usize) -> Result<Var<'t>, AutodiffError> {
    validate_common("merge_concatenate", inputs)?;
    Var::cat(inputs, dim)
}

/// 同 shape の入力を要素ごとに加算する（`x0 + x1 + …`。index 順の左畳み込み・2 件以上）。
pub fn merge_add<'t>(inputs: &[Var<'t>]) -> Result<Var<'t>, AutodiffError> {
    let first = validate_common("merge_add", inputs)?;
    validate_same_shape(&first, inputs)?;
    fold_left(inputs, Var::add)
}

/// 同 shape の入力を要素ごとに乗算する（`x0 * x1 * …`。index 順の左畳み込み・2 件以上）。
pub fn merge_multiply<'t>(inputs: &[Var<'t>]) -> Result<Var<'t>, AutodiffError> {
    let first = validate_common("merge_multiply", inputs)?;
    validate_same_shape(&first, inputs)?;
    fold_left(inputs, Var::mul)
}

/// 同 shape の入力の要素ごとの平均を取る（`(x0 + x1 + …) / n`・2 件以上・`n <= 2^24`）。
pub fn merge_average<'t>(inputs: &[Var<'t>]) -> Result<Var<'t>, AutodiffError> {
    let first = validate_common("merge_average", inputs)?;
    validate_same_shape(&first, inputs)?;
    let n = inputs.len();
    if n > MAX_AVERAGE_INPUTS {
        return Err(invalid(format!(
            "merge_average: 入力数 {n} が上限 {MAX_AVERAGE_INPUTS} を超える"
        )));
    }
    let rank = first.shape().len();
    let divisor = if rank == 0 {
        Tensor::scalar(n as f32)
    } else {
        Tensor::new(vec![n as f32], &vec![1usize; rank]).map_err(AutodiffError::Shape)?
    };
    let sum = fold_left(inputs, Var::add)?;
    let n_var = first.tape().var_no_grad(&divisor);
    sum.div(&n_var)
}

/// `inputs` を先頭から二項演算 `f` で左畳み込みする（検証済みで 2 件以上が前提）。
fn fold_left<'t>(
    inputs: &[Var<'t>],
    f: impl Fn(&Var<'t>, &Var<'t>) -> Result<Var<'t>, AutodiffError>,
) -> Result<Var<'t>, AutodiffError> {
    let mut iter = inputs.iter();
    let Some(first) = iter.next() else {
        return Err(invalid("内部不整合: 畳み込み入力が空".to_string()));
    };
    let mut acc = *first;
    for v in iter {
        acc = f(&acc, v)?;
    }
    Ok(acc)
}
