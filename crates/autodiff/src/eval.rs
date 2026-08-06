//! naive CPU の forward 値計算（クレート非公開・暫定参照実装）。
//!
//! `Var`（`var.rs`）の各演算メソッドが `tensor-core::Tensor<f32>` の
//! 値を実際に計算するために呼ぶ。`backend-cpu`（TASK-1.6・#20 以降）が
//! まだ未完のため、TASK-1.9（バックエンド抽象層への接続）で backend
//! 経由の実行に置き換えるまでの暫定実装である（PoC-v2-2 の
//! `docs/spec/03-poc/poc-v2-2-autodiff/` 構成に合わせ、テープ機構と
//! 値計算を分離しておくことで差し替えの影響範囲をこのファイルに限定
//! する）。
//!
//! **FMA 契約**: `matmul` の内積蓄積は `f32::mul_add` を用いる
//! （`.claude/rules/coding-rust.md`「CPU 参照実装は `f32::mul_add` を
//! 用い、GPU 側の既定 FMA 契約と揃える」。PoC-v2-5 の K=4096 ストレス
//! ケースで実測確認済みの丸め方針）。
//!
//! shape の事前検査（`matmul_out_shape`/`broadcast_shape`/
//! `require_same_shape`/`reduce_out_shape`）は呼び出し元（`var.rs`）が
//! 済ませてから本モジュールを呼ぶ契約とする。本モジュールの関数は
//! shape が既に整合していることを前提とし、`ShapeError` を返さない
//! （`tensor-core::Tensor` 側 API のエラーも本番経路の
//! `unwrap()`/`expect()` は使わず `debug_assert!` 経由のフォールバックで
//! 吸収する。`.claude/rules/coding-rust.md`）。

use tensor_core::Tensor;

/// テンソルを行優先連続バッファへ実体化し `Vec<f32>` として取り出す。
///
/// `contiguous()` は非 contiguous な入力（transpose・stride 0
/// ブロードキャスト view 等）を実体化するため、その結果に対する
/// `as_slice()` は理論上必ず `Some` を返す。それでも本番経路で
/// `unwrap()`/`expect()` は使わない方針（`.claude/rules/coding-rust.md`）
/// のため、`None` 経路は多次元インデックス走査によるコピーへ
/// フォールバックする（到達すれば `contiguous()`/`is_contiguous()` の
/// 契約違反であり、`debug_assert!` で検知可能にする）。
fn dense_vec(tensor: &Tensor<f32>) -> Vec<f32> {
    let contiguous = tensor.contiguous();
    if let Some(slice) = contiguous.as_slice() {
        return slice.to_vec();
    }
    debug_assert!(
        false,
        "dense_vec: contiguous() 後の as_slice() が None を返した（契約違反）"
    );
    let shape = contiguous.shape().to_vec();
    let numel = contiguous.numel();
    let mut out = Vec::with_capacity(numel);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..numel {
        out.push(contiguous.get(&index).unwrap_or(0.0));
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    out
}

/// `Tensor::new` は shape とデータ長が一致する限り失敗しない。呼び出し
/// 元（本モジュール内）はすべて事前に shape 検査済みの出力を組み立てる
/// ため、`ShapeError` は理論上発生しない。それでも `unwrap()`/`expect()`
/// を使わない方針のため、失敗時は空テンソルへ安全側フォールスルーする
/// （`debug_assert!` で契約違反を検知可能にする）。
fn build_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    match Tensor::new(data, shape) {
        Ok(t) => t,
        Err(_) => {
            debug_assert!(
                false,
                "build_tensor: shape 検査済みのはずのデータ構築が失敗した（契約違反）"
            );
            // 契約違反時の安全側フォールバック: 空データ・shape [0] は
            // 要素数積が 0 で一致するため `Tensor::new` が失敗する条件
            // （`ElementCountMismatch`/`ElementCountOverflow`）をいずれも
            // 満たさず、構造的に失敗しえない。到達すれば呼び出し元の
            // shape 計算ロジックにバグがある（`unwrap_or_else` の分岐は
            // 型を合わせるためだけの到達不能パスであり、本番経路の
            // 「失敗しうる入力に対する unwrap」には該当しない）。
            Tensor::new(Vec::new(), &[0])
                .unwrap_or_else(|_| unreachable!("shape [0] construction cannot fail"))
        }
    }
}

/// 2 次元 `matmul`（`lhs: [m,k]` × `rhs: [k,n]` → `[m,n]`）。
/// shape 検査（`matmul_out_shape`）は呼び出し元が済ませている前提。
pub(crate) fn matmul(lhs: &Tensor<f32>, rhs: &Tensor<f32>) -> Tensor<f32> {
    let m = lhs.shape()[0];
    let k = lhs.shape()[1];
    let n = rhs.shape()[1];
    let lhs_data = dense_vec(lhs);
    let rhs_data = dense_vec(rhs);
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                // FMA 契約統一（コメント冒頭参照）: 積和を `mul_add` で行う。
                acc = lhs_data[i * k + p].mul_add(rhs_data[p * n + j], acc);
            }
            out[i * n + j] = acc;
        }
    }
    build_tensor(out, &[m, n])
}

/// ブロードキャスト付き要素ごとの二項演算（`add`/`mul` 共通実装）。
/// shape 検査（`broadcast_shape`）は呼び出し元が済ませている前提。
/// `tensor-core::Tensor::broadcast_with` で両者を共通 shape の view へ
/// 揃えたうえで要素ごとに `op` を適用する。
fn broadcast_binary(
    lhs: &Tensor<f32>,
    rhs: &Tensor<f32>,
    op: impl Fn(f32, f32) -> f32,
) -> Tensor<f32> {
    let (blhs, brhs) = match lhs.broadcast_with(rhs) {
        Ok(pair) => pair,
        Err(_) => {
            debug_assert!(
                false,
                "broadcast_binary: 呼び出し元の broadcast_shape 検査済み前提が崩れた"
            );
            return lhs.clone();
        }
    };
    let shape = blhs.shape().to_vec();
    let lhs_data = dense_vec(&blhs);
    let rhs_data = dense_vec(&brhs);
    let out: Vec<f32> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(&a, &b)| op(a, b))
        .collect();
    build_tensor(out, &shape)
}

/// bias broadcast を含む要素ごとの加算（`docs/public-api-design.md` §3.2）。
pub(crate) fn add(lhs: &Tensor<f32>, rhs: &Tensor<f32>) -> Tensor<f32> {
    broadcast_binary(lhs, rhs, |a, b| a + b)
}

/// ブロードキャスト付き要素ごとの乗算。
pub(crate) fn mul(lhs: &Tensor<f32>, rhs: &Tensor<f32>) -> Tensor<f32> {
    broadcast_binary(lhs, rhs, |a, b| a * b)
}

/// shape 不変の要素ごとの単項演算（`relu`/`exp`/`tanh` 共通実装）。
fn unary(input: &Tensor<f32>, op: impl Fn(f32) -> f32) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    let data = dense_vec(input);
    let out: Vec<f32> = data.into_iter().map(op).collect();
    build_tensor(out, &shape)
}

pub(crate) fn relu(input: &Tensor<f32>) -> Tensor<f32> {
    unary(input, |v| v.max(0.0))
}

pub(crate) fn exp(input: &Tensor<f32>) -> Tensor<f32> {
    unary(input, f32::exp)
}

pub(crate) fn tanh(input: &Tensor<f32>) -> Tensor<f32> {
    unary(input, f32::tanh)
}

/// `dim` に沿った reduction（`sum`/`max` 共通の走査ロジック）。
/// `input` は行優先連続データとして走査し、`axis` を
/// 「外側（outer）× 走査軸（axis_len）× 内側（inner）」の 3 段に分解
/// することで任意軸の縮約を単一ループ構造で表現する
/// （`dim: None` の全軸縮約は呼び出し元がスカラー特別扱いする）。
fn reduce_axis(
    input: &Tensor<f32>,
    axis: usize,
    init: f32,
    op: impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    let shape = input.shape();
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let data = dense_vec(input);
    let mut out = vec![init; outer * inner];
    for o in 0..outer {
        for a in 0..axis_len {
            for i in 0..inner {
                let src = (o * axis_len + a) * inner + i;
                let dst = o * inner + i;
                out[dst] = op(out[dst], data[src]);
            }
        }
    }
    out
}

/// `sum(dim)`。`dim: None` は全要素の総和をスカラー（shape `[]`）で返す。
pub(crate) fn sum(input: &Tensor<f32>, dim: Option<usize>, out_shape: &[usize]) -> Tensor<f32> {
    match dim {
        None => {
            let total: f32 = dense_vec(input).into_iter().sum();
            build_tensor(vec![total], out_shape)
        }
        Some(axis) => build_tensor(reduce_axis(input, axis, 0.0, |a, b| a + b), out_shape),
    }
}

/// `max(dim)`。`dim: None` は全要素中の最大値をスカラー（shape `[]`）で
/// 返す。空テンソル（`numel() == 0`）は呼び出し元の `reduce_out_shape`
/// 検査を通過しうるが、そのケースでは `f32::NEG_INFINITY` を返す
/// （`fold` の初期値のまま。NumPy の `max` は空配列でエラーにするのが
/// 慣習だが、本イシューでは shape 検査のみをスコープとし数値的な特殊
/// ケースの扱いは #19（回帰テスト・数値突合）で確定する）。
pub(crate) fn max(input: &Tensor<f32>, dim: Option<usize>, out_shape: &[usize]) -> Tensor<f32> {
    match dim {
        None => {
            let m = dense_vec(input)
                .into_iter()
                .fold(f32::NEG_INFINITY, f32::max);
            build_tensor(vec![m], out_shape)
        }
        Some(axis) => build_tensor(
            reduce_axis(input, axis, f32::NEG_INFINITY, f32::max),
            out_shape,
        ),
    }
}

/// 平均二乗誤差（スカラー出力）。shape 一致検査（`require_same_shape`）
/// は呼び出し元が済ませている前提。
pub(crate) fn mse_loss(pred: &Tensor<f32>, target: &Tensor<f32>) -> Tensor<f32> {
    let pred_data = dense_vec(pred);
    let target_data = dense_vec(target);
    let numel = pred_data.len();
    let sum_sq: f32 = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &t)| {
            let diff = p - t;
            diff * diff
        })
        .sum();
    let mean = if numel == 0 {
        0.0
    } else {
        sum_sq / numel as f32
    };
    build_tensor(vec![mean], &[])
}
