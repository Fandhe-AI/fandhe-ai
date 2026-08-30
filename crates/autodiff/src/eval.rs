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

use std::borrow::Cow;

use fandhe_ai_tensor_core::Tensor;

use crate::layout;
use crate::var::Reduction;

std::thread_local! {
    /// `matmul`（下記）が転置 view（`grad.rs::transpose2d` が作る
    /// zero-copy view）を `layout::classify_2d` で分類できず、
    /// `dense_vec`（`contiguous()` 経由のホスト側転置コピー）へ
    /// フォールバックした回数（イシュー #1046。`backend-metal::ops::
    /// RESIDENT_HOST_REPACK_COUNT` と同型の可観測点）。matmul VJP
    /// （`grad.rs::matmul_vjp`・`Op::LinearResident` の `d_weight`）が
    /// 転置 view を渡しても本カウンタが増えないことをテストで確認する
    /// ことで、「転置コピー 0 回」（受け入れ条件 (a)）を機械検証する。
    pub(crate) static MATMUL_HOST_REPACK_COUNT: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

/// テンソルを行優先連続バッファへ実体化し `Vec<f32>` として取り出す。
///
/// `contiguous()` は非 contiguous な入力（transpose・stride 0
/// ブロードキャスト view 等）を実体化するため、その結果に対する
/// `as_slice()` は理論上必ず `Some` を返す。それでも本番経路で
/// `unwrap()`/`expect()` は使わない方針（`.claude/rules/coding-rust.md`）
/// のため、`None` 経路は多次元インデックス走査によるコピーへ
/// フォールバックする（到達すれば `contiguous()`/`is_contiguous()` の
/// 契約違反であり、`debug_assert!` で検知可能にする）。
///
/// `pub(crate)`: `grad.rs`（TASK-1.5b・#17）が各演算の VJP 計算・
/// 数値微分突合テストで forward と同じ稠密化ロジックを再利用する
/// （数式の実体を 2 か所に別実装しない方針。PoC-v2-2 準拠）。
pub(crate) fn dense_vec(tensor: &Tensor<f32>) -> Vec<f32> {
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

/// `dense_vec` の読み取り専用・コピー回避版（イシュー #1026・
/// `perf(backend-cpu): 学習ループのホスト側コピー・再構築を除去する`）。
///
/// `Sgd::step`／`AdamW::step`（`crates/autodiff/src/optim/sgd.rs`・
/// `crates/autodiff/src/nn/optim/adamw.rs`）は各 step で `param`／`grad`／
/// momentum バッファを走査するだけで書き換えない（更新後の値は別の
/// 新規 `Vec` へ積んで `Tensor::new` で構築し直す）。この読み取り専用の
/// 用途では `dense_vec` の `slice.to_vec()`（ヒープ確保 + 全要素コピー）
/// は不要であり、既に contiguous な入力（`Linear::weight`/`bias`・
/// `Gradients` 出力はいずれも密なバッファ）に対しては `tensor.as_slice()`
/// が直接借用スライスを返す（`contiguous()` を経由しない）ため、それを
/// そのまま返せば呼び出し元の走査は成立する。
///
/// 戻り値を `Cow<[f32]>` にしているのは、非 contiguous な入力
/// （transpose 済み view 等）では `contiguous()` が新しい `Tensor` を
/// 実体化する必要があり、その結果は本関数のローカル変数になるため
/// スライスを呼び出し元へ借用として返せない（ダングリング参照になり
/// コンパイルエラーになる）ためである。この場合のみ `dense_vec`
/// （所有権を持つ `Vec` を返す既存の稠密化ロジック。二重実装しない）
/// へフォールバックし `Cow::Owned` として返す。
///
/// `pub(crate)`: `dense_vec` と同じ可視性（optimizer モジュールから
/// 呼ばれるための最小限の公開範囲）。
pub(crate) fn dense_vec_ref(tensor: &Tensor<f32>) -> Cow<'_, [f32]> {
    match tensor.as_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => Cow::Owned(dense_vec(tensor)),
    }
}

/// shape とデータ長の一致を型で保証する非 panic 構築（TASK-12.1d・
/// #164。`docs/fusion-graph-design.md` §2.5「eval.rs 非 panic 化の設計
/// 方針」）。`Tensor::from_shape_fill`（`tensor-core` 側の総コンスト
/// ラクタ。`pub` + `#[doc(hidden)]`）は `shape` から `numel` を導出し
/// `fill` で埋める。呼び出し元（本モジュール内）はすべて事前に shape
/// 検査済みの出力を組み立てるため、実運用では `data.len()` と `shape`
/// は必ず一致し要素数積のオーバーフローも起こらない
/// （`debug_assert_eq!` で契約違反を検知可能にする。不一致時は
/// `get(i).copied().unwrap_or(0.0)` により欠落分を `0.0` で安全側に
/// 埋める）。
///
/// **`from_shape_fill` は shape の要素数積を `checked_numel` で検査する
/// `Result` を返す（PR #403 codex-review P1 是正。`tensor.rs` の該当
/// コメント参照）**: `materialize_non_fallible`〈`tape.rs`〉が要求する
/// 「構造的に失敗しない」契約（`docs/fusion-graph-design.md` §3.5.3
/// (iii)）を保つため、本関数自体は引き続き必ず値を返す非 panic 関数の
/// ままとする——`Err`（理論上到達しない契約違反）は `debug_assert!` で
/// 検知しつつ [`fandhe_ai_tensor_core::Tensor::scalar`]（真に infallible）による
/// 安全側フォールバックへ吸収する。
pub(crate) fn build_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    debug_assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "build_tensor: shape 検査済みのはずのデータ長が一致しない（契約違反）"
    );
    Tensor::from_shape_fill(shape, |i| data.get(i).copied().unwrap_or(0.0)).unwrap_or_else(|_| {
        debug_assert!(
            false,
            "build_tensor: shape の要素数積がオーバーフローした（契約違反）"
        );
        Tensor::scalar(0.0)
    })
}

/// クラス添字テンソル（`Tensor<i32>`）の稠密化。`dense_vec`（上記）の
/// `i32` 版で、`cross_entropy_loss`（下記）・`Var::cross_entropy_loss`
/// （`var.rs`。targets 添字の範囲検査）が読み出し専用で使う。
pub(crate) fn dense_vec_i32(tensor: &Tensor<i32>) -> Vec<i32> {
    let contiguous = tensor.contiguous();
    if let Some(slice) = contiguous.as_slice() {
        return slice.to_vec();
    }
    debug_assert!(
        false,
        "dense_vec_i32: contiguous() 後の as_slice() が None を返した（契約違反）"
    );
    let shape = contiguous.shape().to_vec();
    let numel = contiguous.numel();
    let mut out = Vec::with_capacity(numel);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..numel {
        out.push(contiguous.get(&index).unwrap_or(0));
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

/// `matmul`（下記）のオペランド 1 個を、ホスト側転置コピーなしで
/// 読み出せる形へ変換する（イシュー #1046）。
///
/// `layout::classify_2d`（`crate::layout`。`backend-metal::layout` と
/// 同一規則の双子モジュール。PR #1077 で `tensor-core` からクレート内
/// 非公開モジュールへ差し戻した。詳細は `crate::layout` のクレート
/// ドキュメント参照）が行優先 contiguous・転置 view（`grad.rs::transpose2d` が作る
/// `strides == [1, ld]` の zero-copy view）のいずれかに分類できる場合、
/// `Tensor::as_view_slice`（借用）をそのまま返し `MATMUL_HOST_REPACK_COUNT`
/// を増やさない。分類できない形状（stride 0 のブロードキャスト等）
/// のみ、従来どおり `dense_vec`（`contiguous()` 経由のホスト側コピー）
/// へフォールバックしカウンタを増やす。
fn matmul_operand(tensor: &Tensor<f32>) -> (Cow<'_, [f32]>, layout::MatrixLayout) {
    if let Some(matrix_layout) = layout::classify_2d(tensor.shape(), tensor.strides())
        && let Some(slice) = tensor.as_view_slice()
    {
        return (Cow::Borrowed(slice), matrix_layout);
    }
    MATMUL_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
    let (rows, cols) = (tensor.shape()[0], tensor.shape()[1]);
    (
        Cow::Owned(dense_vec(tensor)),
        layout::MatrixLayout {
            rows,
            cols,
            ld: cols,
            transposed: false,
        },
    )
}

/// 2 次元 `matmul`（`lhs: [m,k]` × `rhs: [k,n]` → `[m,n]`）。
/// shape 検査（`matmul_out_shape`）は呼び出し元が済ませている前提。
///
/// イシュー #1046: `matmul_vjp`（`grad.rs`）が `transpose2d`（zero-copy
/// view）で作った転置オペランドをそのまま渡しても、`matmul_operand` が
/// `layout::MatrixLayout` の添字式（`transposed` フラグで行優先／列優先
/// を切替）で読み出すためホスト側転置コピーが発生しない。行優先
/// contiguous 入力（従来からの主経路）では `ld == cols` となり、
/// 添字式は変更前の `lhs_data[i * k + p]`／`rhs_data[p * n + j]` と
/// 完全に一致する（k ループの反復順・`mul_add` 呼び出しも不変のため
/// 既存の bit 完全一致テストを崩さない）。
pub(crate) fn matmul(lhs: &Tensor<f32>, rhs: &Tensor<f32>) -> Tensor<f32> {
    let m = lhs.shape()[0];
    let k = lhs.shape()[1];
    let n = rhs.shape()[1];
    let (lhs_data, lhs_layout) = matmul_operand(lhs);
    let (rhs_data, rhs_layout) = matmul_operand(rhs);
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                let a = if lhs_layout.transposed {
                    lhs_data[p * lhs_layout.ld + i]
                } else {
                    lhs_data[i * lhs_layout.ld + p]
                };
                let b = if rhs_layout.transposed {
                    rhs_data[j * rhs_layout.ld + p]
                } else {
                    rhs_data[p * rhs_layout.ld + j]
                };
                // FMA 契約統一（コメント冒頭参照）: 積和を `mul_add` で行う。
                acc = a.mul_add(b, acc);
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

/// NaN 伝播する 2 項最大値（IEEE 754 `maximum` セマンティクス相当）。
///
/// `f32::max` は非 `NaN` 側のオペランドを返すため、上流で発生した
/// `NaN` が `relu`/`max` reduction を通過すると forward 値から消え、
/// テープに記録される数値のデバッグやバックエンド間数値一致検証
/// （`.claude/rules/coding-rust.md`「相対誤差 1e-3 未満 または絶対誤差
/// 1e-5 未満」）に影響しうる（Cursor Bugbot 指摘。PR #221）。
/// いずれかが `NaN` なら `NaN` を返し、伝播を保つ。
fn nan_propagating_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

pub(crate) fn relu(input: &Tensor<f32>) -> Tensor<f32> {
    unary(input, |v| nan_propagating_max(v, 0.0))
}

pub(crate) fn exp(input: &Tensor<f32>) -> Tensor<f32> {
    unary(input, f32::exp)
}

pub(crate) fn tanh(input: &Tensor<f32>) -> Tensor<f32> {
    unary(input, f32::tanh)
}

/// 数値安定形のシグモイド。`x >= 0` は `1/(1+exp(-x))`、`x < 0` は
/// `exp(x)/(1+exp(x))` を使い分け、大きな負値入力での `exp` オーバー
/// フロー（`exp(-x)` が `+inf` に発散する経路）を回避する
/// （TASK-9.1b・#92。`nn::activation::Sigmoid` の forward 実体）。
/// `NaN` 入力はいずれの分岐も `NaN` を伝播する（`is_sign_negative` は
/// `NaN` に対して符号ビットで分岐するが、後続の演算が `NaN` を保つため
/// 結果は変わらない）。
fn sigmoid_scalar(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub(crate) fn sigmoid(input: &Tensor<f32>) -> Tensor<f32> {
    unary(input, sigmoid_scalar)
}

/// `dim` に沿った reduction（`sum`/`max` 共通の走査ロジック）。
/// `input` は行優先連続データとして走査し、`axis` を
/// 「外側（outer）× 走査軸（axis_len）× 内側（inner）」の 3 段に分解
/// することで任意軸の縮約を単一ループ構造で表現する
/// （`dim: None` の全軸縮約は呼び出し元がスカラー特別扱いする）。
///
/// **TASK-12.1d（#164）**: `Var::sum`/`Var::max`（`var.rs`）の実行は
/// `eval.rs` 直接呼び出しから `self.tape.ops().sum`/`max`（`BackendOps`
/// 経由）へ置き換えたため、本関数（および `sum`/`max`。下記）は
/// `Var::sum`/`max` の本番経路では呼ばれなくなった。ただし
/// **codex-review 第 19〜21 波・PR #403 の P1 是正（2026-08-08 追記）**
/// で `default_ops::NaiveOps`（`Tape::default()`／
/// `compat::Sequential::predict` 無引数版が使う compat 用
/// `BackendOps` 実装）がこの `sum`/`max` に委譲するようになったため、
/// `#[cfg(test)]` は外し本番ビルドにも含める。統合テストの数値微分
/// 突合（`test_support.rs`・`grad.rs` の VJP テスト）も引き続き同じ
/// 実装を使う（数式の実体を二重管理しない）。
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
/// `Var::sum` の本番経路（`BackendOps::sum`）からは呼ばれないが、
/// `default_ops::NaiveOps::sum`（compat 経路）とテスト（数値微分突合）が
/// 使う（上記 `reduce_axis` コメント参照。TASK-12.1d・#164）。
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
/// `Var::max` の本番経路（`BackendOps::max`）からは呼ばれないが、
/// `default_ops::NaiveOps::max`（compat 経路）とテスト（数値微分突合）が
/// 使う（`reduce_axis` コメント参照。TASK-12.1d・#164）。
pub(crate) fn max(input: &Tensor<f32>, dim: Option<usize>, out_shape: &[usize]) -> Tensor<f32> {
    match dim {
        None => {
            let m = dense_vec(input)
                .into_iter()
                .fold(f32::NEG_INFINITY, nan_propagating_max);
            build_tensor(vec![m], out_shape)
        }
        Some(axis) => build_tensor(
            reduce_axis(input, axis, f32::NEG_INFINITY, nan_propagating_max),
            out_shape,
        ),
    }
}

/// 二乗誤差の縮約（スカラー出力）。shape 一致検査
/// （`require_same_shape`）は呼び出し元が済ませている前提。`reduction`
/// で mean（全要素平均）/sum（全要素総和）を切り替える（#190。
/// `Var::mse_loss_with`（`var.rs`）から呼ばれる）。`numel == 0` は
/// mean・sum とも 0.0 を返す（mean 側はゼロ除算回避、sum 側は空和が
/// 数学的に 0 のため元々の定義と一致）。
pub(crate) fn mse_loss(
    pred: &Tensor<f32>,
    target: &Tensor<f32>,
    reduction: crate::var::Reduction,
) -> Tensor<f32> {
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
    let out = match reduction {
        crate::var::Reduction::Mean => {
            if numel == 0 {
                0.0
            } else {
                sum_sq / numel as f32
            }
        }
        crate::var::Reduction::Sum => sum_sq,
    };
    build_tensor(vec![out], &[])
}

/// `axis` に沿った数値安定形 softmax（シフト → `exp` → 正規化）。
/// `cross_entropy_loss`（forward。下記）の log-sum-exp 計算と
/// `grad.rs::cross_entropy_loss_vjp`（`softmax(x) − onehot(t)`）が同じ
/// 「シフトして exp・正規化する」実体を共有する（数式の実体を
/// forward/backward で二重実装しない方針。`grad.rs` 冒頭 doc）。
/// `pub(crate)`: `grad.rs` が VJP 計算で再利用する。
pub(crate) fn softmax_along(input: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let data = dense_vec(input);
    let mut out = vec![0f32; data.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut m = f32::NEG_INFINITY;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                m = nan_propagating_max(m, data[idx]);
            }
            let mut sum_exp = 0f32;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                let e = (data[idx] - m).exp();
                out[idx] = e;
                sum_exp += e;
            }
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                out[idx] /= sum_exp;
            }
        }
    }
    build_tensor(out, &shape)
}

/// CrossEntropy 損失（log-sum-exp 安定化。クラス次元 `class_dim` 指定。
/// #191・親イシュー #189）。shape 検査（`class_dim` 範囲・targets
/// shape 一致・targets 添字範囲）は呼び出し元（`var.rs::
/// Var::cross_entropy_loss`）が済ませている前提。
///
/// `class_dim` を除いた添字の組（サンプル）ごとに
/// `loss = log_sum_exp(logits) − logits[target]`
/// （`= −log_softmax(logits)[target]`。オーバーフロー回避のシフト量
/// `m = max_c logits[c]` を経由するため大振幅入力でも有限値を保つ）を
/// 計算し、`reduction` で集約する。
///
/// 空バッチ（サンプル数 `N == 0`）は `mse_loss`（上記）の先例に合わせ
/// 0.0 を返す（PyTorch は `NaN`。差異は許容: #191 実装計画 §3.3）。
pub(crate) fn cross_entropy_loss(
    logits: &Tensor<f32>,
    targets: &Tensor<i32>,
    class_dim: usize,
    reduction: Reduction,
) -> Tensor<f32> {
    let shape = logits.shape().to_vec();
    let outer: usize = shape[..class_dim].iter().product();
    let axis_len = shape[class_dim];
    let inner: usize = shape[class_dim + 1..].iter().product();
    let data = dense_vec(logits);
    let target_data = dense_vec_i32(targets);
    let n = outer * inner;

    let mut total = 0f32;
    for o in 0..outer {
        for i in 0..inner {
            let mut m = f32::NEG_INFINITY;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                m = nan_propagating_max(m, data[idx]);
            }
            let mut sum_exp = 0f32;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                sum_exp += (data[idx] - m).exp();
            }
            let lse = m + sum_exp.ln();
            let t = target_data[o * inner + i];
            // 呼び出し元（`var.rs::Var::cross_entropy_loss`）が
            // `0 <= t < axis_len` を検査済みの前提。範囲外は契約違反で
            // あり `unwrap()`/`expect()` を使わず `debug_assert!` で
            // 検知しつつ安全側（loss 寄与 0）へフォールバックする
            // （`.claude/rules/coding-rust.md` 本番経路 panic 禁止方針）。
            let target_logit = if t >= 0 && (t as usize) < axis_len {
                data[(o * axis_len + t as usize) * inner + i]
            } else {
                debug_assert!(false, "cross_entropy_loss: target 添字が範囲外（契約違反）");
                lse
            };
            total += lse - target_logit;
        }
    }

    let loss = match reduction {
        Reduction::Mean if n > 0 => total / n as f32,
        Reduction::Mean => 0.0,
        Reduction::Sum => total,
    };
    build_tensor(vec![loss], &[])
}

#[cfg(test)]
mod dense_vec_ref_tests {
    use super::*;

    // イシュー #1026「学習ループのホスト側コピー・再構築を除去する」の
    // 機械的な回帰検証（advisor 助言: `dense_vec_ref` は `MemoryOps`
    // 境界を持たないため `AllocationTracker` ではコピー回数を数えられ
    // ない。ここでは「返した `Cow` が呼び出し元の `Tensor` のバッファを
    // 直接指している（ポインタ一致）」ことを確認することで、
    // `slice.to_vec()` によるヒープコピーが発生していないことを機械的に
    // 検証する）。

    #[test]
    fn contiguous_input_borrows_without_copy() {
        let tensor = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let borrowed = dense_vec_ref(&tensor);

        assert!(
            matches!(borrowed, std::borrow::Cow::Borrowed(_)),
            "contiguous な入力は Cow::Borrowed（コピーなし）を返す契約"
        );
        // ポインタ一致で「元の `Tensor` のバッファをそのまま指している」
        // ことを確認する（`to_vec()` していれば別のヒープ確保になり
        // ポインタが一致しない）。
        let original_ptr = tensor
            .as_slice()
            .expect("test fixture: contiguous")
            .as_ptr();
        assert_eq!(borrowed.as_ptr(), original_ptr);
        assert_eq!(&*borrowed, &[1.0, 2.0, 3.0, 4.0][..]);
    }

    #[test]
    fn non_contiguous_input_falls_back_to_owned_dense_vec() {
        // transpose 済み view は非 contiguous になるため `as_slice()` が
        // `None` を返す（`tensor.rs` doc）。`dense_vec_ref` は `dense_vec`
        // へフォールバックし、値は一致するが所有権を持つ `Cow::Owned` を
        // 返す契約。
        let tensor = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let transposed = tensor
            .transpose(0, 1)
            .expect("test fixture: 2 次元 tensor の transpose(0, 1) は常に成功する");
        assert!(
            transposed.as_slice().is_none(),
            "test fixture: transpose 後は非 contiguous であることが前提"
        );

        let owned = dense_vec_ref(&transposed);
        assert!(
            matches!(owned, std::borrow::Cow::Owned(_)),
            "非 contiguous な入力は Cow::Owned（dense_vec フォールバック）を返す契約"
        );
        assert_eq!(&*owned, &dense_vec(&transposed)[..]);
    }
}
