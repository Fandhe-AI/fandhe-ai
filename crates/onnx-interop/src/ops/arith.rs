//! ONNX `Add`／`Mul`／`Div`／`Mod`／`Sqrt` MVP 算術オペ（TASK-7.3a・#82。
//! `i64` 版 `Add`／`Mul`／`Div`／`Mod` はイシュー #87 残作業〈TASK-7.4a〉で追加）。
//!
//! `Add`／`Mul`／`Div`／`Mod` は ONNX 仕様上 multidirectional broadcasting
//! （NumPy 互換）に対応する二項演算であり、`tensor_core::Tensor::broadcast_with`
//! （`broadcast_shape` 委譲。`crates/tensor-core/src/tensor.rs`）で出力 shape へ
//! 揃えた view を得てから要素ごとに計算する。`Sqrt` は `activation.rs` の
//! `Relu`/`Sigmoid` と同じ単項要素ごと写像パターンに従う。
//!
//! `i64` 版は `transformer.onnx` の `Shape -> Gather -> Div` 等（head_dim 算出。
//! `onnx/interp.rs` 冒頭コメント参照）が要求する。`f32` 版と異なり `i64` は
//! `checked_*` 系で厳密にオーバーフロー・0 除算を検査し、`f32` の暗黙の
//! `inf`/`NaN` 透過とは違って型付きエラーで拒否する（外部モデル由来の shape
//! 値を無条件に信頼しないため。`.claude/rules/security.md` A03）。
//! `Sqrt` は ONNX 仕様上 float 型のみを受け付ける演算であり `i64` 対応は行わない
//! （`transformer.onnx` でも `Sqrt` の直前には常に `Cast(to=FLOAT)` が挟まる。
//! イシュー #87 実測確認済み）。

use tensor_core::Tensor;

use super::error::OpError;

/// 二項要素ごと演算の共通実装。`lhs`／`rhs` を `broadcast_with` で共通 shape の
/// view へ揃え、`contiguous()` で実体化してから `f` を適用する（`activation.rs`
/// の `map_elementwise` と同様、非 contiguous view をそのまま走査しない）。
fn map_binary_elementwise(
    op: &'static str,
    lhs: &Tensor<f32>,
    rhs: &Tensor<f32>,
    f: impl Fn(f32, f32) -> f32,
) -> Result<Tensor<f32>, OpError> {
    let (l, r) = lhs.broadcast_with(rhs)?;
    let lc = l.contiguous();
    let rc = r.contiguous();
    let l_slice = lc.as_slice().ok_or(OpError::NonContiguousInternal(op))?;
    let r_slice = rc.as_slice().ok_or(OpError::NonContiguousInternal(op))?;
    let data: Vec<f32> = l_slice
        .iter()
        .zip(r_slice.iter())
        .map(|(&a, &b)| f(a, b))
        .collect();
    Tensor::new(data, lc.shape()).map_err(OpError::from)
}

/// `Add(a, b) = a + b`（multidirectional broadcasting）。
pub fn add(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map_binary_elementwise("Add", a, b, |x, y| x + y)
}

/// `Mul(a, b) = a * b`（multidirectional broadcasting）。
pub fn mul(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map_binary_elementwise("Mul", a, b, |x, y| x * y)
}

/// `Div(a, b) = a / b`（multidirectional broadcasting）。IEEE 754 の除算規則を
/// そのまま透過し（0 除算は `inf`／`-inf`／`NaN`）、事前にゼロ検査でエラーには
/// しない。ONNX の `Div` 自体も floating point 入力に対する 0 除算の扱いを
/// IEEE 754 に委ねている。
pub fn div(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map_binary_elementwise("Div", a, b, |x, y| x / y)
}

/// `Mod(a, b)`。ONNX `Mod-13` 仕様の `fmod` 属性を明示的に受け取る:
/// `fmod=1`（`fmod: true`）は C `fmod` 相当（結果の符号は被除数 `a` に従う。
/// Rust の `%` 演算子と同一）で、浮動小数点入力に対して仕様上有効な唯一の
/// モードである。`fmod=0`（`fmod: false`）は ONNX 仕様上「整数入力のみ」
/// 有効な Python 風モード（結果の符号は除数 `b` に従う）であり、`f32` 入力に
/// 対して要求された場合は黙って `rem_euclid` 等で代替せず
/// [`OpError::UnsupportedFmodMode`] を返す（誤った数値を静かに返さない。
/// `.claude/rules/security.md` A03 の「外部入力の検証」に準ずる）。
pub fn modulo(a: &Tensor<f32>, b: &Tensor<f32>, fmod: bool) -> Result<Tensor<f32>, OpError> {
    if !fmod {
        return Err(OpError::UnsupportedFmodMode { op: "Mod" });
    }
    map_binary_elementwise("Mod", a, b, |x, y| x % y)
}

/// `Sqrt(x) = sqrt(x)`。負値入力は ONNX 仕様上未定義域であり、`f32::sqrt` の
/// IEEE 754 準拠動作（`NaN`）をそのまま透過する（`Div` の 0 除算と同じ方針）。
pub fn sqrt(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    let xc = x.contiguous();
    let slice = xc
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("Sqrt"))?;
    let data: Vec<f32> = slice.iter().map(|&v| v.sqrt()).collect();
    Tensor::new(data, xc.shape()).map_err(OpError::from)
}

/// 二項要素ごと演算の `i64` 版共通実装。`f32` 版の [`map_binary_elementwise`] と
/// 異なり、`f` は `checked_*` 系の結果（`Option<i64>`）をそのまま返せるよう
/// フォールブルにする（オーバーフロー・0 除算を要素単位で検出して型付き
/// エラーに変換するため）。`f32` 版は非フォールブルのまま維持し（既存の
/// 9 テストへの回帰を避ける・無条件の `Ok` ラップで clippy を汚さない）、
/// 共通化はしない。
fn map_binary_elementwise_i64(
    op: &'static str,
    lhs: &Tensor<i64>,
    rhs: &Tensor<i64>,
    mk_err: impl Fn() -> OpError,
    f: impl Fn(i64, i64) -> Option<i64>,
) -> Result<Tensor<i64>, OpError> {
    let (l, r) = lhs.broadcast_with(rhs)?;
    let lc = l.contiguous();
    let rc = r.contiguous();
    let l_slice = lc.as_slice().ok_or(OpError::NonContiguousInternal(op))?;
    let r_slice = rc.as_slice().ok_or(OpError::NonContiguousInternal(op))?;
    let mut data: Vec<i64> = Vec::with_capacity(l_slice.len());
    for (&a, &b) in l_slice.iter().zip(r_slice.iter()) {
        data.push(f(a, b).ok_or_else(&mk_err)?);
    }
    Tensor::new(data, lc.shape()).map_err(OpError::from)
}

/// `Add(a, b) = a + b`（`i64` 版。multidirectional broadcasting）。ONNX の
/// `Shape -> Gather` 由来の次元値同士の加算等で使う。`checked_add` でオーバー
/// フローを検査し、`f32` 版のような暗黙の wrap は行わない（モジュール冒頭
/// `//!` 参照）。
pub fn add_i64(a: &Tensor<i64>, b: &Tensor<i64>) -> Result<Tensor<i64>, OpError> {
    map_binary_elementwise_i64(
        "Add",
        a,
        b,
        || OpError::IntegerOverflow { op: "Add" },
        |x, y| x.checked_add(y),
    )
}

/// `Mul(a, b) = a * b`（`i64` 版。multidirectional broadcasting）。
pub fn mul_i64(a: &Tensor<i64>, b: &Tensor<i64>) -> Result<Tensor<i64>, OpError> {
    map_binary_elementwise_i64(
        "Mul",
        a,
        b,
        || OpError::IntegerOverflow { op: "Mul" },
        |x, y| x.checked_mul(y),
    )
}

/// `Div(a, b) = a / b`（`i64` 版。0 方向切り捨て。Rust の `/` と同じ ONNX
/// 整数除算仕様）。`checked_div` は「除数 0」と「`i64::MIN / -1`」の両方を
/// `None` で表すため、[`OpError::IntegerDivisionFailed`] で両ケースをまとめて
/// 拒否する（`error.rs` の variant コメント参照）。
pub fn div_i64(a: &Tensor<i64>, b: &Tensor<i64>) -> Result<Tensor<i64>, OpError> {
    map_binary_elementwise_i64(
        "Div",
        a,
        b,
        || OpError::IntegerDivisionFailed { op: "Div" },
        |x, y| x.checked_div(y),
    )
}

/// `Mod(a, b)`（`i64` 版）。`f32` 版と異なり ONNX `Mod-13` 仕様の `fmod` 属性は
/// 整数入力に対して両モードとも有効: `fmod=1`（`fmod: true`）は C 風（結果の
/// 符号は被除数 `a` に従う。Rust の `%` と同一で `checked_rem` がそのまま
/// 対応する）、`fmod=0`（既定）は Python 風（結果の符号は除数 `b` に従う）。
/// 後者は `checked_rem` の結果を除数の符号に合わせて補正する
/// （`rem_euclid` は「常に非負」であり被除数側の符号情報を失うため使わない。
/// 除数が負の場合に Python の `%` と異なる結果になる）。
pub fn mod_i64(a: &Tensor<i64>, b: &Tensor<i64>, fmod: bool) -> Result<Tensor<i64>, OpError> {
    map_binary_elementwise_i64(
        "Mod",
        a,
        b,
        || OpError::IntegerDivisionFailed { op: "Mod" },
        move |x, y| {
            let r = x.checked_rem(y)?;
            if fmod {
                Some(r)
            } else if r != 0 && (r < 0) != (y < 0) {
                // Python 風: 剰余が 0 でなく被除数側の符号（checked_rem の結果）が
                // 除数の符号と異なる場合、除数を 1 回加えて符号を揃える。
                r.checked_add(y)
            } else {
                Some(r)
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_same_shape() {
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::<f32>::new(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let y = add(&a, &b).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 11.0);
        assert_eq!(y.get(&[1, 1]).unwrap(), 44.0);
    }

    #[test]
    fn add_broadcast_row_and_column() {
        // [3,1] + [1,4] -> [3,4]（NumPy 互換ブロードキャスト。ops_shape.rs の
        // elementwise_broadcast_row_and_column と同じ代表例）。
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let b = Tensor::<f32>::new(vec![10.0, 20.0, 30.0, 40.0], &[1, 4]).unwrap();
        let y = add(&a, &b).unwrap();
        assert_eq!(y.shape(), &[3, 4]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 11.0);
        assert_eq!(y.get(&[2, 3]).unwrap(), 43.0);
    }

    #[test]
    fn add_broadcast_incompatible_rejected() {
        let a = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let b = Tensor::<f32>::zeros(&[4]).unwrap();
        let err = add(&a, &b).unwrap_err();
        assert!(matches!(err, OpError::Shape(_)));
    }

    #[test]
    fn mul_broadcast_scalar() {
        // rank 差分（[2,3] と [3]）の暗黙先頭軸補完。
        let a = Tensor::<f32>::new((1..=6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let b = Tensor::<f32>::new(vec![2.0, 2.0, 2.0], &[3]).unwrap();
        let y = mul(&a, &b).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(y.get(&[1, 2]).unwrap(), 12.0);
    }

    #[test]
    fn div_known_values() {
        let a = Tensor::<f32>::new(vec![10.0, 9.0, 8.0, 7.0], &[4]).unwrap();
        let b = Tensor::<f32>::new(vec![2.0, 3.0, 4.0, 7.0], &[4]).unwrap();
        let y = div(&a, &b).unwrap();
        assert_eq!(y.get(&[0]).unwrap(), 5.0);
        assert_eq!(y.get(&[1]).unwrap(), 3.0);
        assert_eq!(y.get(&[2]).unwrap(), 2.0);
        assert_eq!(y.get(&[3]).unwrap(), 1.0);
    }

    #[test]
    fn div_by_zero_follows_ieee754() {
        let a = Tensor::<f32>::new(vec![1.0, -1.0, 0.0], &[3]).unwrap();
        let b = Tensor::<f32>::new(vec![0.0, 0.0, 0.0], &[3]).unwrap();
        let y = div(&a, &b).unwrap();
        assert!(y.get(&[0]).unwrap().is_infinite() && y.get(&[0]).unwrap() > 0.0);
        assert!(y.get(&[1]).unwrap().is_infinite() && y.get(&[1]).unwrap() < 0.0);
        assert!(y.get(&[2]).unwrap().is_nan());
    }

    #[test]
    fn modulo_fmod_matches_c_fmod_semantics() {
        // fmod=1: 結果の符号は被除数（a）に従う（ONNX Mod-13 仕様）。
        let a = Tensor::<f32>::new(vec![7.0, -7.0, 7.0, -7.0], &[4]).unwrap();
        let b = Tensor::<f32>::new(vec![3.0, 3.0, -3.0, -3.0], &[4]).unwrap();
        let y = modulo(&a, &b, true).unwrap();
        assert_eq!(y.get(&[0]).unwrap(), 1.0);
        assert_eq!(y.get(&[1]).unwrap(), -1.0);
        assert_eq!(y.get(&[2]).unwrap(), 1.0);
        assert_eq!(y.get(&[3]).unwrap(), -1.0);
    }

    #[test]
    fn modulo_python_style_rejected_for_float_input() {
        let a = Tensor::<f32>::new(vec![7.0], &[1]).unwrap();
        let b = Tensor::<f32>::new(vec![3.0], &[1]).unwrap();
        let err = modulo(&a, &b, false).unwrap_err();
        assert!(matches!(err, OpError::UnsupportedFmodMode { op: "Mod" }));
    }

    #[test]
    fn sqrt_known_values() {
        let x = Tensor::<f32>::new(vec![0.0, 1.0, 4.0, 9.0], &[4]).unwrap();
        let y = sqrt(&x).unwrap();
        assert_eq!(y.get(&[0]).unwrap(), 0.0);
        assert_eq!(y.get(&[1]).unwrap(), 1.0);
        assert_eq!(y.get(&[2]).unwrap(), 2.0);
        assert_eq!(y.get(&[3]).unwrap(), 3.0);
    }

    #[test]
    fn sqrt_negative_is_nan() {
        let x = Tensor::<f32>::new(vec![-1.0], &[1]).unwrap();
        let y = sqrt(&x).unwrap();
        assert!(y.get(&[0]).unwrap().is_nan());
    }

    // ---- i64 版（イシュー #87 残作業: transformer.onnx の head_dim 算出等）----

    #[test]
    fn add_i64_known_values() {
        let a = Tensor::<i64>::new(vec![1, 2, 3, 4], &[4]).unwrap();
        let b = Tensor::<i64>::new(vec![10, 20, 30, 40], &[4]).unwrap();
        let y = add_i64(&a, &b).unwrap();
        assert_eq!(y.get(&[0]).unwrap(), 11);
        assert_eq!(y.get(&[3]).unwrap(), 44);
    }

    #[test]
    fn add_i64_overflow_rejected() {
        let a = Tensor::<i64>::new(vec![i64::MAX], &[1]).unwrap();
        let b = Tensor::<i64>::new(vec![1], &[1]).unwrap();
        let err = add_i64(&a, &b).unwrap_err();
        assert!(matches!(err, OpError::IntegerOverflow { op: "Add" }));
    }

    #[test]
    fn mul_i64_known_values() {
        let a = Tensor::<i64>::new(vec![2, 3], &[2]).unwrap();
        let b = Tensor::<i64>::new(vec![4, 5], &[2]).unwrap();
        let y = mul_i64(&a, &b).unwrap();
        assert_eq!(y.get(&[0]).unwrap(), 8);
        assert_eq!(y.get(&[1]).unwrap(), 15);
    }

    #[test]
    fn mul_i64_overflow_rejected() {
        let a = Tensor::<i64>::new(vec![i64::MAX], &[1]).unwrap();
        let b = Tensor::<i64>::new(vec![2], &[1]).unwrap();
        let err = mul_i64(&a, &b).unwrap_err();
        assert!(matches!(err, OpError::IntegerOverflow { op: "Mul" }));
    }

    #[test]
    fn div_i64_truncates_toward_zero() {
        // ONNX 整数 Div は 0 方向切り捨て（Rust `/` と同一）。
        let a = Tensor::<i64>::new(vec![7, -7, 7, -7], &[4]).unwrap();
        let b = Tensor::<i64>::new(vec![2, 2, -2, -2], &[4]).unwrap();
        let y = div_i64(&a, &b).unwrap();
        assert_eq!(y.get(&[0]).unwrap(), 3);
        assert_eq!(y.get(&[1]).unwrap(), -3);
        assert_eq!(y.get(&[2]).unwrap(), -3);
        assert_eq!(y.get(&[3]).unwrap(), 3);
    }

    #[test]
    fn div_i64_by_zero_rejected() {
        let a = Tensor::<i64>::new(vec![1], &[1]).unwrap();
        let b = Tensor::<i64>::new(vec![0], &[1]).unwrap();
        let err = div_i64(&a, &b).unwrap_err();
        assert!(matches!(err, OpError::IntegerDivisionFailed { op: "Div" }));
    }

    #[test]
    fn div_i64_min_by_neg_one_rejected() {
        // i64::MIN / -1 は i64::MAX を超えるためオーバーフロー
        // （checked_div は 0 除算と同じ None を返す。error.rs 参照）。
        let a = Tensor::<i64>::new(vec![i64::MIN], &[1]).unwrap();
        let b = Tensor::<i64>::new(vec![-1], &[1]).unwrap();
        let err = div_i64(&a, &b).unwrap_err();
        assert!(matches!(err, OpError::IntegerDivisionFailed { op: "Div" }));
    }

    #[test]
    fn mod_i64_fmod_true_sign_follows_dividend() {
        let a = Tensor::<i64>::new(vec![7, -7, 7, -7], &[4]).unwrap();
        let b = Tensor::<i64>::new(vec![3, 3, -3, -3], &[4]).unwrap();
        let y = mod_i64(&a, &b, true).unwrap();
        assert_eq!(y.get(&[0]).unwrap(), 1);
        assert_eq!(y.get(&[1]).unwrap(), -1);
        assert_eq!(y.get(&[2]).unwrap(), 1);
        assert_eq!(y.get(&[3]).unwrap(), -1);
    }

    #[test]
    fn mod_i64_fmod_false_sign_follows_divisor() {
        // ONNX Mod-13 の既定（fmod=0）は Python 風: 結果の符号は除数に従う。
        let a = Tensor::<i64>::new(vec![7, -7, 7, -7], &[4]).unwrap();
        let b = Tensor::<i64>::new(vec![3, 3, -3, -3], &[4]).unwrap();
        let y = mod_i64(&a, &b, false).unwrap();
        assert_eq!(y.get(&[0]).unwrap(), 1);
        assert_eq!(y.get(&[1]).unwrap(), 2);
        assert_eq!(y.get(&[2]).unwrap(), -2);
        assert_eq!(y.get(&[3]).unwrap(), -1);
    }

    #[test]
    fn mod_i64_by_zero_rejected() {
        let a = Tensor::<i64>::new(vec![1], &[1]).unwrap();
        let b = Tensor::<i64>::new(vec![0], &[1]).unwrap();
        let err = mod_i64(&a, &b, true).unwrap_err();
        assert!(matches!(err, OpError::IntegerDivisionFailed { op: "Mod" }));
    }

    #[test]
    fn add_i64_broadcast_incompatible_rejected() {
        let a = Tensor::<i64>::new(vec![1, 2, 3], &[3]).unwrap();
        let b = Tensor::<i64>::new(vec![1, 2], &[2]).unwrap();
        let err = add_i64(&a, &b).unwrap_err();
        assert!(matches!(err, OpError::Shape(_)));
    }
}
