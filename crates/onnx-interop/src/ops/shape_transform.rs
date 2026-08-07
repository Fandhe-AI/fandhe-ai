//! ONNX `Reshape`／`Squeeze`／`Transpose` オペ（TASK-7.3b・#83）。
//!
//! `ops/shape_ops.rs` の `Shape`／`Unsqueeze`（#79）と同じ「入力テンソル＋属性 →
//! 出力テンソル」の純粋関数方針に従う。`shape`/`axes`/`perm` は ONNX proto デコード層
//! （TASK-7.2a・decode 実装は別イシュー）に依存しないプレーンなスライスで受け取る。
//! `transformer.onnx` は `ir_version=8`（PyTorch エクスポート・opset 13 以降相当）の
//! ため、`Squeeze` の `axes` は属性ではなく本来は第 2 入力テンソルだが、本モジュールは
//! decode 層に関知しない単体演算のみを扱う方針（`ops/mod.rs`）に従い、呼び出し元が
//! 入力テンソルから取り出した `&[i64]` をそのまま受け取る。

use tensor_core::Tensor;

use super::error::OpError;
use super::normalize_axis;

/// `Reshape(data, shape, allowzero=0)`: `data` を `shape` へ再解釈する。
///
/// - `shape` の要素 `0` は `allowzero=false`（既定）のとき対応する入力次元をそのまま
///   コピーする（ONNX Reshape-13 仕様。`allowzero=true` の場合は `0` を文字どおり
///   サイズ 0 として扱うため本関数では特別扱いしない）
/// - `shape` の要素 `-1` は 1 箇所のみ許容し、他の次元から要素数を逆算する
/// - 要素数の積は `checked_mul` でオーバーフローを検査してから確保する
///   （外部フォーマット由来の shape 値を信頼しない。OWASP A03。`.claude/rules/security.md`）
pub fn reshape(x: &Tensor<f32>, shape: &[i64], allowzero: bool) -> Result<Tensor<f32>, OpError> {
    let resolved = resolve_reshape_target(shape, x.shape(), allowzero, x.numel())?;
    x.contiguous().reshape(&resolved).map_err(OpError::from)
}

/// `shape`（`0`/`-1` を含みうる ONNX Reshape 表記）を確定した非負 `usize` shape へ解決する。
/// `reshape` から分離しているのは、shape 解決ロジックを純粋にテスト可能にするため。
fn resolve_reshape_target(
    shape: &[i64],
    input_shape: &[usize],
    allowzero: bool,
    input_numel: usize,
) -> Result<Vec<usize>, OpError> {
    let mut resolved = Vec::with_capacity(shape.len());
    let mut infer_at: Option<usize> = None;
    let mut known_product: usize = 1;

    for (i, &dim) in shape.iter().enumerate() {
        if dim == -1 {
            if infer_at.is_some() {
                return Err(OpError::InvalidReshapeSpec {
                    reason: "shape に -1 を複数指定できない",
                });
            }
            infer_at = Some(i);
            resolved.push(0); // 後で上書きするプレースホルダ
        } else if dim == 0 && !allowzero {
            let copied = *input_shape.get(i).ok_or(OpError::InvalidReshapeSpec {
                reason: "0 が指す入力次元が rank を超えている",
            })?;
            known_product =
                known_product
                    .checked_mul(copied)
                    .ok_or(OpError::InvalidReshapeSpec {
                        reason: "shape 要素数の積が usize 範囲を超える",
                    })?;
            resolved.push(copied);
        } else if dim < 0 {
            return Err(OpError::InvalidReshapeSpec {
                reason: "shape に -1 以外の負値は指定できない",
            });
        } else {
            let d = dim as usize;
            known_product = known_product
                .checked_mul(d)
                .ok_or(OpError::InvalidReshapeSpec {
                    reason: "shape 要素数の積が usize 範囲を超える",
                })?;
            resolved.push(d);
        }
    }

    if let Some(idx) = infer_at {
        if known_product == 0 || !input_numel.is_multiple_of(known_product) {
            return Err(OpError::InvalidReshapeSpec {
                reason: "-1 推論に必要な要素数が入力要素数を割り切れない",
            });
        }
        resolved[idx] = input_numel / known_product;
    }

    Ok(resolved)
}

/// `Squeeze(data, axes=None)`: `axes` が指す size-1 次元を削除する。
/// `axes` 省略時（`None`）は size 1 の全次元を削除する（opset 13 系仕様）。
/// 負軸は `x.rank()` に対して正規化する。対象次元が size 1 でない場合・範囲外・重複は
/// 型付きエラーを返す（データは不変・shape 再計算のみのため `reshape` に委譲する）。
pub fn squeeze(x: &Tensor<f32>, axes: Option<&[i64]>) -> Result<Tensor<f32>, OpError> {
    let rank = x.rank();
    let shape = x.shape();

    let mut targets: Vec<usize> = match axes {
        Some(axes) => {
            let mut normalized = Vec::with_capacity(axes.len());
            for &axis in axes {
                let n = normalize_axis(axis, rank).ok_or(OpError::AxisOutOfRange {
                    op: "Squeeze",
                    axis,
                    rank,
                })?;
                if shape[n] != 1 {
                    return Err(OpError::InvalidReshapeSpec {
                        reason: "Squeeze の対象軸は size 1 である必要がある",
                    });
                }
                normalized.push(n);
            }
            normalized
        }
        None => shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if d == 1 { Some(i) } else { None })
            .collect(),
    };

    targets.sort_unstable();
    for pair in targets.windows(2) {
        if pair[0] == pair[1] {
            return Err(OpError::DuplicateAxis {
                op: "Squeeze",
                axis: pair[0],
            });
        }
    }

    let new_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| {
            if targets.binary_search(&i).is_ok() {
                None
            } else {
                Some(d)
            }
        })
        .collect();

    x.contiguous().reshape(&new_shape).map_err(OpError::from)
}

/// `Transpose(data, perm=None)`: `perm` が指す軸順へ並べ替える。
/// `perm` 省略時は軸順を逆転する（ONNX Transpose 仕様の既定動作）。
/// `tensor-core::Tensor::transpose` は 2 軸交換のみ対応のため（`crates/tensor-core/src/tensor.rs`）、
/// N 階一般対応は本モジュール内の [`permute_copy`] で行う（tensor-core への汎用 `permute`
/// 追加はスコープ外。PR 本文に切り出し候補として記録する。`.claude/rules/out-of-scope-tracking.md`）。
pub fn transpose(x: &Tensor<f32>, perm: Option<&[i64]>) -> Result<Tensor<f32>, OpError> {
    let rank = x.rank();
    let resolved_perm: Vec<usize> = match perm {
        Some(perm) => validate_perm(perm, rank)?,
        None => (0..rank).rev().collect(),
    };
    permute_copy(x, &resolved_perm)
}

/// `perm` が `0..rank` の順列であることを検証し、正規化済み `usize` 列を返す。
fn validate_perm(perm: &[i64], rank: usize) -> Result<Vec<usize>, OpError> {
    if perm.len() != rank {
        return Err(OpError::LengthMismatch {
            op: "Transpose",
            name: "perm",
            expected: rank,
            actual: perm.len(),
        });
    }
    let mut seen = vec![false; rank];
    let mut resolved = Vec::with_capacity(rank);
    for &axis in perm {
        let n = normalize_axis(axis, rank).ok_or(OpError::AxisOutOfRange {
            op: "Transpose",
            axis,
            rank,
        })?;
        if seen[n] {
            return Err(OpError::DuplicateAxis {
                op: "Transpose",
                axis: n,
            });
        }
        seen[n] = true;
        resolved.push(n);
    }
    Ok(resolved)
}

/// `perm` に従い出力 shape を並べ替え、多重インデックス走査で contiguous な出力バッファへ
/// コピーする（N 階一般対応。`transpose(dim0, dim1)` の 2 軸限定を補うヘルパ）。
/// `x` を先に `contiguous()` してから読むため、`get` は常に `Some` を返す不変条件を持つ
/// （到達しない `None` 分岐は `NonContiguousInternal` として扱う。coding-rust.md の
/// panic 回避方針に従い `unwrap`/`expect` は使わない）。
fn permute_copy(x: &Tensor<f32>, perm: &[usize]) -> Result<Tensor<f32>, OpError> {
    let src = x.contiguous();
    let in_shape = src.shape().to_vec();
    let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();
    let numel = src.numel();

    let mut data = Vec::with_capacity(numel);
    let mut out_index = vec![0usize; out_shape.len()];
    for _ in 0..numel {
        // out_index[k] は perm[k] 番目の入力軸に対応するため、入力側インデックスは
        // perm の逆写像で組み立てる（in_index[perm[k]] = out_index[k]）。
        let mut in_index = vec![0usize; in_shape.len()];
        for (k, &p) in perm.iter().enumerate() {
            in_index[p] = out_index[k];
        }
        let value = src
            .get(&in_index)
            .ok_or(OpError::NonContiguousInternal("Transpose"))?;
        data.push(value);

        for axis in (0..out_shape.len()).rev() {
            out_index[axis] += 1;
            if out_index[axis] < out_shape[axis] {
                break;
            }
            out_index[axis] = 0;
        }
    }

    Tensor::new(data, &out_shape).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reshape_basic() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let y = reshape(&t, &[3, 2], false).unwrap();
        assert_eq!(y.shape(), &[3, 2]);
        assert_eq!(y.as_slice().unwrap(), t.as_slice().unwrap());
    }

    #[test]
    fn reshape_infers_minus_one() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let y = reshape(&t, &[-1, 2], false).unwrap();
        assert_eq!(y.shape(), &[3, 2]);
    }

    #[test]
    fn reshape_zero_copies_input_dim() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let y = reshape(&t, &[0, -1], false).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
    }

    #[test]
    fn reshape_element_count_mismatch_rejected() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let err = reshape(&t, &[4, 2], false).unwrap_err();
        assert!(matches!(err, OpError::Shape(_)));
    }

    #[test]
    fn reshape_multiple_minus_one_rejected() {
        let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let err = reshape(&t, &[-1, -1], false).unwrap_err();
        assert!(matches!(err, OpError::InvalidReshapeSpec { .. }));
    }

    #[test]
    fn squeeze_with_axes() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[1, 2, 3, 1]).unwrap();
        let y = squeeze(&t, Some(&[0, 3])).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
    }

    #[test]
    fn squeeze_negative_axis() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[1, 2, 3]).unwrap();
        let y = squeeze(&t, Some(&[-3])).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
    }

    #[test]
    fn squeeze_without_axes_drops_all_size_one() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[1, 2, 1, 3]).unwrap();
        let y = squeeze(&t, None).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
    }

    #[test]
    fn squeeze_non_size_one_axis_rejected() {
        let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let err = squeeze(&t, Some(&[0])).unwrap_err();
        assert!(matches!(err, OpError::InvalidReshapeSpec { .. }));
    }

    #[test]
    fn transpose_default_reverses_axes() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let y = transpose(&t, None).unwrap();
        assert_eq!(y.shape(), &[3, 2]);
        assert_eq!(y.get(&[1, 0]).unwrap(), t.get(&[0, 1]).unwrap());
        assert_eq!(y.get(&[2, 1]).unwrap(), t.get(&[1, 2]).unwrap());
    }

    #[test]
    fn transpose_explicit_perm_rank3() {
        // shape [2,3,4]、perm [2,0,1] → 出力 shape [4,2,3]
        let numel = 2 * 3 * 4;
        let t = Tensor::<f32>::new((0..numel).map(|v| v as f32).collect(), &[2, 3, 4]).unwrap();
        let y = transpose(&t, Some(&[2, 0, 1])).unwrap();
        assert_eq!(y.shape(), &[4, 2, 3]);
        // y[c, a, b] == t[a, b, c] （perm[k] は出力軸 k が指す入力軸）
        for a in 0..2 {
            for b in 0..3 {
                for c in 0..4 {
                    assert_eq!(
                        y.get(&[c, a, b]).unwrap(),
                        t.get(&[a, b, c]).unwrap(),
                        "mismatch at a={a} b={b} c={c}"
                    );
                }
            }
        }
    }

    #[test]
    fn transpose_perm_length_mismatch_rejected() {
        let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let err = transpose(&t, Some(&[0])).unwrap_err();
        assert!(matches!(
            err,
            OpError::LengthMismatch {
                op: "Transpose",
                ..
            }
        ));
    }

    #[test]
    fn transpose_perm_duplicate_rejected() {
        let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let err = transpose(&t, Some(&[0, 0])).unwrap_err();
        assert!(matches!(
            err,
            OpError::DuplicateAxis {
                op: "Transpose",
                ..
            }
        ));
    }

    #[test]
    fn transpose_perm_out_of_range_rejected() {
        let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let err = transpose(&t, Some(&[0, 5])).unwrap_err();
        assert!(matches!(
            err,
            OpError::AxisOutOfRange {
                op: "Transpose",
                ..
            }
        ));
    }
}
