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

use fandhe_ai_tensor_core::{
    BceKind, GruBackwardOutput, GruPointwiseOutput, HuberKind, KlDivTarget, LstmPointwiseOutput,
    Pool2dParams, ScatterReduce, ShapeError, Tensor, VectorNormOrd, adaptive_window,
    bilinear_blend, bilinear_scale, bilinear_src_coord,
};

use crate::layout;
use crate::var::Reduction;

/// 線形代数（inv／solve／det／qr／cholesky／svd）・matrix_norm のホスト
/// 参照実装（イシュー #1621）。行数が大きいため子モジュールへ分ける
/// （モジュール冒頭コメント参照）。
pub(crate) mod linalg;
pub(crate) mod scalar;

std::thread_local! {
    /// `matmul`（下記）が転置 view（`grad.rs::transpose2d` が作る
    /// zero-copy view）を `layout::classify_2d` で分類できず、
    /// `dense_vec`（`contiguous()` 経由のホスト側転置コピー）へ
    /// フォールバックした回数（イシュー #1046。`backend-metal::ops::
    /// RESIDENT_HOST_REPACK_COUNT` と同型の可観測点）。
    ///
    /// イシュー #1211 以降、本番の matmul VJP（`grad.rs::matmul_vjp`・
    /// `Op::LinearResident` の `d_weight`）は `eval::matmul` ではなく
    /// `BackendOps::gemm` を経由するため、本カウンタが観測するのは
    /// `NaiveOps`／`TestOps`（compat・テストの forward 参照実装経路）
    /// 経由の呼び出しに限られる（`grad.rs` の `matmul_vjp_does_not_
    /// repack_transposed_operands` テストは `test_ops()` 越しにこの
    /// compat 経路のゼロコピーを検証している。本番 `CpuBackendOps::
    /// gemm` 経路の転置再パックは、片側転置〈NT/TN〉かつ dense な転置
    /// 格納を判定できる場合、CPU（イシュー #1213）・CUDA（イシュー
    /// #1214）・Metal（イシュー #1215）とも解消済み
    /// （`backend-cpu::ops::GEMM_HOST_REPACK_COUNT`／`backend-cuda::
    /// ops::GEMM_HOST_REPACK_COUNT`／`backend-metal::ops::
    /// GEMM_HOST_REPACK_COUNT`〈いずれも本カウンタとは別の crate 内部
    /// カウンタ〉で可観測。Metal は既存 NN 経路とは別カーネルを通る
    /// ため数値契約が bit 一致ではなく REQ-2 複合判定である点が
    /// CPU／CUDA と異なる）。`docs/matmul-vjp-zero-copy-decision.md`
    /// §4・§4.2・§4.3・§4.4 追補）。
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

/// `g: [m, n]`（rank-2）の**行方向の和**（列ごとに `sum_{row=0}^{m-1}
/// g[row, col]`）を計算し、長さ `n` の `Vec<f32>` を返す（イシュー
/// #1566）。
///
/// **数値方式（2026-09-12 ユーザー承認 A・PR #1659 codex-review P1
/// 是正）**: `.claude/rules/coding-rust.md` の「勾配の長軸縮約は `f64`
/// アキュムレータで統一する」規約（イシュー #1102・PR #1120）に従い、
/// 各列は `f64` アキュムレータへ `dense_vec` で稠密化した行を `f64` へ
/// 昇格して蓄積し、最後に 1 回だけ `f32` へ downcast する（単純な `f32`
/// 逐次 `+=` は `[1e8, 1.0, -1e8]` のような相殺パターンで寄与が丸め
/// 落ちして消える。回帰テスト `reduce_bias_grad_rows_tests::
/// preserves_cancelling_contribution_via_f64_accumulator` 参照）。
/// この変更により `grad::reduce_to_shape(g, &[n])`（同じ縮約を行う
/// 汎用パス。`f32` 逐次和のまま**変更しない**——weight 勾配・`reduce_
/// to_shape` 自体の bit 同一契約は本イシューのスコープ外）との bit
/// 完全一致は失われる（意図的な乖離。両者の使い分けは呼び出し元 doc
/// 「resident 経路のみ」を参照）。
///
/// **`m == 1` の特殊扱い**（PR #1659 codex-review P2 是正。f64 化後も
/// 不変）: `m == 1` は加算を経由せず入力を直接コピーするため、`f64`
/// 昇格・downcast のラウンドトリップでも符号付きゼロ（`-0.0`）を保持
/// する（`f32` → `f64` → `f32` は値を変えない可逆変換）。
///
/// `pub(crate)`: `grad.rs`（`Op::LinearResident` の非 resident bias
/// フォールバック。呼び出しは変更しない——既存の `reduce_to_shape` 経路
/// を維持し、本関数は resident 経路のみで使う）・`optim::device_store`
/// （weight tying 発生時の bias 勾配 tie 累積。`ResidentResolver::
/// fill_resident_weight_grad` doc「bias tie」参照）から呼ばれる。
///
/// `g.shape()` が `[m, n]`（rank-2）でない呼び出しは契約違反
/// （`debug_assert!` で検知。本番経路は空 `Vec` を返す安全側フォール
/// バックとし panic しない。`.claude/rules/coding-rust.md`「本番経路で
/// `unwrap()`/`expect()` を使わない」）。
pub(crate) fn reduce_bias_grad_rows(g: &Tensor<f32>) -> Vec<f32> {
    let shape = g.shape();
    if shape.len() != 2 {
        debug_assert!(
            false,
            "reduce_bias_grad_rows: g は rank-2 のはず（契約違反）"
        );
        return Vec::new();
    }
    let (m, n) = (shape[0], shape[1]);
    let data = dense_vec(g);
    if m == 1 {
        // `reduce_to_shape` は `m == 1` の軸を縮約しないため、直接
        // コピーして `-0.0` 等の符号付きゼロを保持する（上記 doc 参照）。
        return data[0..n].to_vec();
    }
    let mut acc = vec![0f64; n];
    for row in 0..m {
        for (col, a) in acc.iter_mut().enumerate() {
            *a += f64::from(data[row * n + col]);
        }
    }
    acc.into_iter().map(|v| v as f32).collect()
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
///
/// `pub(crate)`: `eval::scalar`（イシュー #1634。`ScalarBinaryOp` の
/// ホストフォールバック forward）が同じ broadcast 走査ロジックを再利用
/// する（`add`/`mul` と数式は異なるが走査部分の二重管理を避ける）。
pub(crate) fn broadcast_binary(
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

/// `outer`／`axis_len`／`inner` を `dim` から決定する（`Var::var`／
/// `Var::norm_l1`／`norm_l2` 共通のヘルパ。イシュー #1723）。`dim=None`
/// は「テンソル全体を単一の縮約軸として扱う」（`outer=1`・
/// `axis_len=numel`・`inner=1`）ことで、[`var_along`]／
/// [`vector_norm_along`] を軸指定・全縮約の両方で共通のループへ
/// 統一する（`softmax_vjp_along` 等〈`grad.rs`〉の 3 段走査と同型の
/// 分解だが、`dim=None` の場合分けをここへ集約する点が異なる）。
fn reduce_outer_axis_inner(shape: &[usize], dim: Option<usize>) -> (usize, usize, usize) {
    match dim {
        None => (1, shape.iter().product(), 1),
        Some(axis) => {
            let outer: usize = shape[..axis].iter().product();
            let axis_len = shape[axis];
            let inner: usize = shape[axis + 1..].iter().product();
            (outer, axis_len, inner)
        }
    }
}

/// `Op::Var`／`Op::Std` 共通の `f64` 分散計算コア（イシュー #1723
/// レビュー是正で `var_along`／`std_along` の重複を統合）。出力要素
/// ごとに ①`f64` で平均 ②`f64` で二乗和 の 2 パスを計算し、`f64` の
/// まま分散を返す（`f32` へ downcast しない——呼び出し元が `var_along`
/// のように分散をそのまま返すか、`std_along` のように `sqrt` してから
/// 返すかを選べるようにするため。`n == 0`／`n <= correction` の検査は
/// 呼び出し元〈`Var::var`／`Var::std`〉が事前に済ませている前提——本
/// 関数は shape が既に整合していることを前提とする契約〈モジュール
/// 冒頭コメント〉に従い検査しない）。戻り値は `outer * inner` 要素の
/// `f64` ベクタ（`out_shape` への詰め直しは呼び出し元が行う）。
fn var_f64_along(input: &Tensor<f32>, dim: Option<usize>, correction: usize) -> Vec<f64> {
    let shape = input.shape();
    let (outer, axis_len, inner) = reduce_outer_axis_inner(shape, dim);
    let data = dense_vec(input);
    let n = axis_len as f64;
    let denom = n - correction as f64;
    let mut out = vec![0f64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut mean_acc = 0.0f64;
            for a in 0..axis_len {
                let src = (o * axis_len + a) * inner + i;
                mean_acc += data[src] as f64;
            }
            let mean = mean_acc / n;
            let mut sq_acc = 0.0f64;
            for a in 0..axis_len {
                let src = (o * axis_len + a) * inner + i;
                let d = data[src] as f64 - mean;
                sq_acc += d * d;
            }
            out[o * inner + i] = sq_acc / denom;
        }
    }
    out
}

/// `Op::Var`（`Var::var`）のホスト参照実装（`BackendOps::var` が
/// `Unsupported` を返した場合のフォールバック。イシュー #1723）。
/// [`var_f64_along`]（`f64` 分散）を計算し、最後に 1 回だけ `f32` へ
/// downcast する（`.claude/rules/coding-rust.md`「正規化統計は要素を
/// 先に `f64` へ昇格してから二乗し、最後に 1 回だけ `f32` へ downcast
/// する」契約）。
pub(crate) fn var_along(
    input: &Tensor<f32>,
    dim: Option<usize>,
    correction: usize,
    out_shape: &[usize],
) -> Tensor<f32> {
    let out: Vec<f32> = var_f64_along(input, dim, correction)
        .into_iter()
        .map(|v| v as f32)
        .collect();
    build_tensor(out, out_shape)
}

/// `Op::Std`（`Var::std`）のホスト参照実装（イシュー #1723 レビュー
/// 是正。codex-review P2 指摘: 当初 `Var::var(..).sqrt()` の合成
/// （`Op::Var` なし）で実装していたため、`Op::Var` が分散を `f32` へ
/// downcast してから `Op::Sqrt` へ渡しており、真の分散が `f32` の
/// 範囲（有限最大値 約 `3.4e38`）を超える極端な入力（例
/// `[-1e20, 1e20]`。分散 `≈1e40` が overflow）で `std` 自体は `f32`
/// で表現可能（`≈1.41e20`）にもかかわらず `inf` になる問題があった）。
/// 本関数は [`var_f64_along`] が返す `f64` 分散に対し `sqrt` も `f64`
/// で計算してから、最後に 1 回だけ `f32` へ downcast する——`sqrt` を
/// 挟むことで分散段階の overflow を回避できる（`std` は分散よりも
/// 小さい値域に収まるため）。
pub(crate) fn std_along(
    input: &Tensor<f32>,
    dim: Option<usize>,
    correction: usize,
    out_shape: &[usize],
) -> Tensor<f32> {
    let out: Vec<f32> = var_f64_along(input, dim, correction)
        .into_iter()
        .map(|v| v.sqrt() as f32)
        .collect();
    build_tensor(out, out_shape)
}

/// `Op::VectorNorm`（`Var::norm_l1`／`norm_l2`）のホスト参照実装
/// （`BackendOps::vector_norm` が `Unsupported` を返した場合の
/// フォールバック。イシュー #1723）。[`var_along`] と同じ
/// `outer`／`axis_len`／`inner` 分解で、出力要素ごとに `f64` 累積し
/// L2 のみ最後に `sqrt` してから 1 回だけ `f32` へ downcast する
/// （`n == 0` の検査は呼び出し元 `Var::norm_l1`／`norm_l2` が済ませて
/// いる前提）。
pub(crate) fn vector_norm_along(
    input: &Tensor<f32>,
    ord: VectorNormOrd,
    dim: Option<usize>,
    out_shape: &[usize],
) -> Tensor<f32> {
    let shape = input.shape();
    let (outer, axis_len, inner) = reduce_outer_axis_inner(shape, dim);
    let data = dense_vec(input);
    let mut out = vec![0f32; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0.0f64;
            for a in 0..axis_len {
                let src = (o * axis_len + a) * inner + i;
                let v = data[src] as f64;
                acc += match ord {
                    VectorNormOrd::L1 => v.abs(),
                    VectorNormOrd::L2 => v * v,
                    // `VectorNormOrd` は `#[non_exhaustive]`
                    // （`tensor-core`）のため将来 variant に備えた
                    // `_` 分岐を持つ（`eval::linalg::matrix_norm` の
                    // `MatrixNormOrd` 拒否と同方針）。呼び出し元
                    // `Var::norm`（`var.rs`）が `norm_l1`／`norm_l2`
                    // 限定の `pub` 入口からのみ本関数を呼ぶ現状の
                    // 契約上、未知 variant は到達しない想定だが、
                    // 安全側フォールバックとして `0.0`（寄与なし）を
                    // 返す。
                    _ => 0.0,
                };
            }
            out[o * inner + i] = match ord {
                VectorNormOrd::L1 => acc as f32,
                VectorNormOrd::L2 => acc.sqrt() as f32,
                _ => 0.0,
            };
        }
    }
    build_tensor(out, out_shape)
}
/// `min(dim)`。`dim: None` は全要素中の最小値をスカラー（shape `[]`）で
/// 返す。[`max`] と異なり **NaN 非伝播**（`f32::min`。`fminf` と同じ）
/// を用いる——`BackendOps::min`（`crate::backend_ops` 経由）の CPU
/// 参照実装（`backend-cpu::reduction::min`）と本番フォールバック経路
/// （[`crate::grad::min_with_fallback`]）の意味論を一致させるため
/// （[`max`] の `nan_propagating_max` との不一致は既存の事実であり
/// 本関数の対象外。イシュー #1720 実装計画 §7「スコープ外」）。
/// 空縮約（要素数 0）の検査は呼び出し元（[`crate::grad::
/// min_with_fallback`]）が行う契約で、本関数自身は `dim: None` かつ
/// 空入力の場合 `f32::INFINITY`（`fold` の初期値のまま）を返す。
pub(crate) fn min(input: &Tensor<f32>, dim: Option<usize>, out_shape: &[usize]) -> Tensor<f32> {
    match dim {
        None => {
            let m = dense_vec(input).into_iter().fold(f32::INFINITY, f32::min);
            build_tensor(vec![m], out_shape)
        }
        Some(axis) => build_tensor(reduce_axis(input, axis, f32::INFINITY, f32::min), out_shape),
    }
}

/// `dim` 軸に沿った最大値／最小値の添字を求める共通走査
/// （[`argmax`]／[`argmin`] が使う。イシュー #1720）。`better(v, best)`
/// が `true` を返したときだけ現在の best を `v` へ更新する
/// （`argmax` は `v > best`・`argmin` は `v < best`）ため、同値
/// （`==`）では更新されず**最初の**添字が残る（タイ先勝ち契約）。
/// `best` が NaN の間は次に来た非 NaN 値で無条件に置換し、`v` が
/// NaN の間は無視する（[`min`]／[`max`] の NaN 規約と整合させる。
/// 全要素 NaN の場合は添字 0 のまま）。`i32::MAX` を超える添字は
/// [`sort`] と同じ理由で `ShapeError::IndexRangeOverflow` を返す。
fn arg_extremum(
    input: &Tensor<f32>,
    dim: Option<usize>,
    out_shape: &[usize],
    better: impl Fn(f32, f32) -> bool,
) -> Result<Tensor<i32>, ShapeError> {
    let data = dense_vec(input);
    let indices = match dim {
        None => {
            let mut best_idx = 0usize;
            let mut best_val = f32::NAN;
            for (idx, &v) in data.iter().enumerate() {
                if v.is_nan() {
                    continue;
                }
                if best_val.is_nan() || better(v, best_val) {
                    best_val = v;
                    best_idx = idx;
                }
            }
            vec![best_idx]
        }
        Some(axis) => {
            let shape = input.shape();
            let outer: usize = shape[..axis].iter().product();
            let axis_len = shape[axis];
            let inner: usize = shape[axis + 1..].iter().product();
            let mut out = vec![0usize; outer * inner];
            for o in 0..outer {
                for i in 0..inner {
                    let dst = o * inner + i;
                    let mut best_idx = 0usize;
                    let mut best_val = f32::NAN;
                    for a in 0..axis_len {
                        let src = (o * axis_len + a) * inner + i;
                        let v = data[src];
                        if v.is_nan() {
                            continue;
                        }
                        if best_val.is_nan() || better(v, best_val) {
                            best_val = v;
                            best_idx = a;
                        }
                    }
                    out[dst] = best_idx;
                }
            }
            out
        }
    };
    let mut out_idx = Vec::with_capacity(indices.len());
    for idx in indices {
        out_idx
            .push(i32::try_from(idx).map_err(|_| ShapeError::IndexRangeOverflow { index: idx })?);
    }
    Ok(build_index_tensor(out_idx, out_shape))
}

/// `dim` 軸に沿った最大値の添字のホスト参照実装（`torch.argmax(dim)`
/// 相当。イシュー #1720）。`BackendOps::argmax` が `Unsupported` を
/// 返したときのみ `Var::argmax`（`crate::grad::argext_with_fallback`
/// 経由）から呼ばれる。空縮約の検査は呼び出し元の責務（[`arg_extremum`]
/// doc 参照）。
pub(crate) fn argmax(
    input: &Tensor<f32>,
    dim: Option<usize>,
    out_shape: &[usize],
) -> Result<Tensor<i32>, ShapeError> {
    arg_extremum(input, dim, out_shape, |v, best| v > best)
}

/// `dim` 軸に沿った最小値の添字のホスト参照実装（`torch.argmin(dim)`
/// 相当。[`argmax`] の最小値版。イシュー #1720）。
pub(crate) fn argmin(
    input: &Tensor<f32>,
    dim: Option<usize>,
    out_shape: &[usize],
) -> Result<Tensor<i32>, ShapeError> {
    arg_extremum(input, dim, out_shape, |v, best| v < best)
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

/// [`huber_loss`]（ホストフォールバック）が縮約に使う固定チャンク
/// サイズ（イシュー #1739）。`backend-cpu::mse::CHUNK`（融合カーネル
/// 側）と同値とし、フォールバック経路が融合カーネル経路と異なる
/// 加算順序で丸め誤差を蓄積し REQ-2 判定を超えて乖離することを避ける
/// （フォールバックは `BackendOps::huber_loss` が `Unsupported` を
/// 返したときのみ経由する`.claude/rules/coding-rust.md`「バックエンド間
/// 数値一致は統一複合判定」の趣旨を先回りして踏襲）。
const HUBER_HOST_FALLBACK_CHUNK: usize = 4096;

/// Huber／SmoothL1 の要素損失 `l(d)`（`d = pred − target`。イシュー
/// #1739）。PyTorch `nn.HuberLoss(delta)`／`nn.SmoothL1Loss(beta)` の
/// 意味論の正（`grad.rs::huber_loss_vjp`・CPU 融合カーネルの CPU 参照
/// 実装〈parity テスト〉双方がこの定義に一致することを検証する）。
///
/// | kind | `\|d\| < delta` | それ以外 |
/// |---|---|---|
/// | `Huber` | `0.5·d²` | `delta·(\|d\| − 0.5·delta)` |
/// | `SmoothL1` | `0.5·d²/delta` | `\|d\| − 0.5·delta` |
///
/// 二次分岐の判定は PyTorch と同じ `<`（等号は線形分岐。両分岐は
/// 境界 `\|d\| == delta` で連続）。`delta` は呼び出し元
/// （`Var::huber_loss_impl`）が有限かつ `> 0` を検証済みの値。
/// `#[non_exhaustive]` な `HuberKind` の未知 variant は `Huber` 意味論へ
/// 安全側フォールバックする（`debug_assert!` でテスト時のみ検知。
/// `bce_elem_loss` と同型の判断）。
pub(crate) fn huber_elem_loss(d: f32, kind: HuberKind, delta: f32) -> f32 {
    let abs_d = d.abs();
    match kind {
        HuberKind::SmoothL1 => {
            if abs_d < delta {
                // `d*d` を先に計算すると delta・d が巨大な有限値の
                // ときに中間積が overflow しうる。`abs_d < delta`
                // 分岐内では `|d/delta| < 1` が保証されるため、先に
                // delta で割ってから d を掛けることで中間値を `|d|`
                // 以下に抑える（`backend-cpu::huber::elem_loss`・
                // CUDA `kernels_huber.rs`・Metal `shaders/huber.metal`
                // と同じ演算順序で揃える）。
                0.5 * (d / delta) * d
            } else {
                abs_d - 0.5 * delta
            }
        }
        // `Huber`。未知 variant はここへ安全側フォールバック。
        _ => {
            debug_assert!(
                matches!(kind, HuberKind::Huber),
                "huber_elem_loss: unknown HuberKind variant {kind:?}; falling back to Huber \
                 semantics"
            );
            if abs_d < delta {
                0.5 * d * d
            } else {
                delta * (abs_d - 0.5 * delta)
            }
        }
    }
}

/// ログクランプ下限（PyTorch `BCELoss` の実装と同じ `-100`。`p` が
/// `0`／`1` に極めて近い場合の `ln` の `-inf` 発散を避ける）。
const BCE_LOG_CLAMP_MIN: f32 = -100.0;

/// [`BceKind::Probabilities`]／[`BceKind::Logits`] 共通の要素損失
/// （`bce_loss`／`bce_loss_vjp`〈`grad.rs`〉の双方から呼ばれる意味論の
/// 正。`docs/compat-api-scope.md` §1.2「損失」節参照）。
///
/// - `Probabilities`: `l = −( y·max(ln p, −100) + (1−y)·max(ln(1−p),
///   −100) )`（PyTorch `BCELoss` と同じログクランプ）。
/// - `Logits`: `x>=0` は `l = (1−y)·x + ln(1+exp(−x))`、`x<0` は
///   `l = −y·x + ln(1+exp(x))`（`max(x,0) − x·y + ln(1+exp(−|x|))` と
///   数式として等価だが、`x` が大きく `y` が 1 に近いとき `x − x·y` が
///   桁落ちしバックエンド間 FMA 契約差で乖離しうるため、減算ではなく
///   `(1−y)·x` の乗算のみで打ち消し量を先に求める形へ書き換えている
///   〈codex-review 指摘・#1737 PR #1848〉。`exp` の引数は常に非正の
///   ため overflow しない。PyTorch `BCEWithLogitsLoss` の内部式と同型）。
pub(crate) fn bce_elem_loss(input: f32, target: f32, kind: BceKind) -> f32 {
    match kind {
        BceKind::Probabilities => {
            let log_p = input.ln().max(BCE_LOG_CLAMP_MIN);
            let log_1mp = (1.0 - input).ln().max(BCE_LOG_CLAMP_MIN);
            -(target * log_p + (1.0 - target) * log_1mp)
        }
        // `Logits`、および `BceKind`（`#[non_exhaustive]`。`tensor-core`
        // 側で将来 variant を追加しうる）の未知 variant は同じ
        // `Logits` 意味論へ安全側フォールバックする（`eval::scatter`
        // の `ScatterReduce` 未知 variant 処理と同型。本関数は
        // infallible 契約のため `Result` を返せない。未知 variant への
        // 到達は契約違反として `debug_assert!` で検知するのみに留める。
        // `.claude/rules/coding-rust.md` 本番経路 panic 禁止方針）。
        kind => {
            debug_assert!(
                matches!(kind, BceKind::Logits),
                "eval::bce_elem_loss: 未知の BceKind variant へフォールバックした（契約違反）"
            );
            if input >= 0.0 {
                (1.0 - target) * input + (-input).exp().ln_1p()
            } else {
                -target * input + input.exp().ln_1p()
            }
        }
    }
}

/// [`huber_elem_loss`] の `pred` に対する要素勾配 `∂l/∂pred`（`scale`
/// 乗算前。イシュー #1739）。`sign(d)` は 3 バックエンドとも
/// `copysign`（本関数は `f32::copysign`）で統一する
/// （`.claude/rules/coding-rust.md` の丸め方針統一と同じ理由で
/// `±0`／符号の扱いをバックエンド間で一致させる）。
///
/// | kind | `\|d\| < delta` | それ以外 |
/// |---|---|---|
/// | `Huber` | `d` | `copysign(delta, d)` |
/// | `SmoothL1` | `d/delta` | `copysign(1, d)` |
pub(crate) fn huber_elem_grad(d: f32, kind: HuberKind, delta: f32) -> f32 {
    // `d` が NaN（`pred`／`target` のいずれかが NaN）のとき、
    // `abs_d < delta` は NaN 比較の規約により常に false となり
    // else 分岐（`copysign` 系）へ落ちて有限な勾配（±1／±delta）を
    // 返してしまう。forward（`huber_elem_loss`）は同じ分岐構造でも
    // else 分岐の結果が `NaN - 0.5*delta = NaN` となり自然に NaN を
    // 返すため、forward と backward で NaN 伝播の有無が食い違う
    // （イシュー #1739 レビュー指摘）。ここで明示的に NaN を伝播する
    // （`backend-cpu::huber::elem_grad`・CUDA `kernels_huber.rs`・
    // Metal `shaders/huber.metal` と同じ方針で揃える）。
    if d.is_nan() {
        return d;
    }
    let abs_d = d.abs();
    match kind {
        HuberKind::SmoothL1 => {
            if abs_d < delta {
                d / delta
            } else {
                1.0f32.copysign(d)
            }
        }
        _ => {
            debug_assert!(
                matches!(kind, HuberKind::Huber),
                "huber_elem_grad: unknown HuberKind variant {kind:?}; falling back to Huber \
                 semantics"
            );
            if abs_d < delta { d } else { delta.copysign(d) }
        }
    }
}

/// `bce_elem_loss` の `dInput`（`scale` を乗じる前の要素勾配。`grad.rs`
/// が上流勾配由来の `scale` を別途掛ける）。
///
/// - `Probabilities`: `(p − y) / max(p·(1−p), 1e−12)`（PyTorch と同じ
///   `eps` によるゼロ除算回避）。
/// - `Logits`: `sigmoid(x) − y`（`sigmoid_scalar` の数値安定形を使う）。
pub(crate) fn bce_elem_grad_input(input: f32, target: f32, kind: BceKind) -> f32 {
    match kind {
        BceKind::Probabilities => {
            let denom = (input * (1.0 - input)).max(1e-12);
            (input - target) / denom
        }
        // `bce_elem_loss` と同じ未知 variant フォールバック規律。
        kind => {
            debug_assert!(
                matches!(kind, BceKind::Logits),
                "eval::bce_elem_grad_input: 未知の BceKind variant へフォールバックした（契約違反）"
            );
            sigmoid_scalar(input) - target
        }
    }
}

/// Huber／SmoothL1 損失の縮約（スカラー出力）のホスト参照実装
/// （イシュー #1739）。`BackendOps::huber_loss` が `Unsupported` を
/// 返したときのみ `Var::huber_loss_impl`（`var.rs`）から呼ばれる。
/// shape 一致検査は呼び出し元が済ませている前提。`numel == 0` は
/// [`mse_loss`] と同じく mean・sum とも `0.0` を返す。
///
/// **縮約方式**: [`HUBER_HOST_FALLBACK_CHUNK`] 固定チャンクで分割し、
/// チャンク内は逐次加算・チャンク間はチャンク番号順に結合する
/// （`backend-cpu::mse::mse_sum_sq_f32` と同じ決定的縮約方式のホスト側
/// ミラー。素朴な先頭からの逐次和のみだと大規模入力で融合カーネル
/// 経路と加算順序が乖離しうるため）。
pub(crate) fn huber_loss(
    pred: &Tensor<f32>,
    target: &Tensor<f32>,
    kind: HuberKind,
    delta: f32,
    reduction: crate::var::Reduction,
) -> Tensor<f32> {
    let pred_data = dense_vec(pred);
    let target_data = dense_vec(target);
    let numel = pred_data.len();
    let sum: f32 = pred_data
        .chunks(HUBER_HOST_FALLBACK_CHUNK)
        .zip(target_data.chunks(HUBER_HOST_FALLBACK_CHUNK))
        .map(|(p_chunk, t_chunk)| {
            p_chunk
                .iter()
                .zip(t_chunk.iter())
                .fold(0.0f32, |acc, (&p, &t)| {
                    acc + huber_elem_loss(p - t, kind, delta)
                })
        })
        .fold(0.0f32, |acc, v| acc + v);
    let out = match reduction {
        crate::var::Reduction::Mean => {
            if numel == 0 {
                0.0
            } else {
                sum / numel as f32
            }
        }
        crate::var::Reduction::Sum => sum,
    };
    build_tensor(vec![out], &[])
}

/// `bce_elem_loss` の `dTarget`（`scale` を乗じる前）。
///
/// - `Probabilities`: forward のクランプ済み式の厳密な導関数
///   `−( max(ln p, −100) − max(ln(1−p), −100) )`（`p` が
///   `(e^−100, 1−e^−100)` の外にあるときのみクランプが効き、PyTorch の
///   無クランプ `−logit(p)` と差が生じる。`docs/compat-api-scope.md`
///   §1.2 参照）。
/// - `Logits`: `−x`。
pub(crate) fn bce_elem_grad_target(input: f32, _target: f32, kind: BceKind) -> f32 {
    match kind {
        BceKind::Probabilities => {
            let log_p = input.ln().max(BCE_LOG_CLAMP_MIN);
            let log_1mp = (1.0 - input).ln().max(BCE_LOG_CLAMP_MIN);
            -(log_p - log_1mp)
        }
        // `bce_elem_loss` と同じ未知 variant フォールバック規律。
        kind => {
            debug_assert!(
                matches!(kind, BceKind::Logits),
                "eval::bce_elem_grad_target: 未知の BceKind variant へフォールバックした（契約違反）"
            );
            -input
        }
    }
}

/// [`bce_loss`] の縮約チャンクサイズ（`backend-cpu::bce::CHUNK` と
/// 同値。ホストフォールバック〈本関数〉と CPU 融合カーネル
/// （`Unsupported` でないときに呼ばれる経路）とで縮約精度を揃える
/// ための意図的複製。チャンク内は逐次 `f32` 加算・チャンク間は
/// チャンク番号順に逐次結合し、単純な全要素逐次加算より丸め誤差を
/// 抑える。codex-review 指摘（PR #1848）: フォールバック側が全要素
/// 逐次加算のままだと、`BackendOps::bce_loss` が `Unsupported` の
/// バックエンドへ切り替わった際に損失値が
/// 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を
/// 超えて乖離しうる（大きな入力での実測差: 相対差 約 0.00648・
/// 絶対差 約 0.00449）。
const BCE_HOST_FALLBACK_CHUNK: usize = 4096;

/// 二値交差エントロピー損失のホスト参照実装（`Var::bce_loss`／
/// `bce_with_logits_loss`〈`var.rs`〉から `BackendOps::bce_loss` が
/// `Unsupported` のときのみ呼ばれる。イシュー #1737）。`mse_loss`
/// （直上）と同じ「shape 一致検査は呼び出し元が済ませている・
/// `numel == 0` は `Mean`／`Sum` とも `0.0`」契約。
///
/// 縮約精度は `backend-cpu::bce::bce_sum_f32`（CPU 融合カーネル）と
/// 同じ固定チャンク（[`BCE_HOST_FALLBACK_CHUNK`]）縮約方式を用いる
/// （`.claude/rules/coding-rust.md`「正規化統計・勾配の長軸縮約は
/// `f64` アキュムレータで統一する」と同種の、縮約経路間の数値契約
/// 統一。BCE の場合は `f64` ではなくチャンク分割方式で backend-cpu
/// 側と揃える。PR #1848 codex-review 指摘）。
pub(crate) fn bce_loss(
    input: &Tensor<f32>,
    target: &Tensor<f32>,
    kind: BceKind,
    reduction: crate::var::Reduction,
) -> Tensor<f32> {
    let input_data = dense_vec(input);
    let target_data = dense_vec(target);
    let numel = input_data.len();
    let sum_loss: f32 = input_data
        .chunks(BCE_HOST_FALLBACK_CHUNK)
        .zip(target_data.chunks(BCE_HOST_FALLBACK_CHUNK))
        .map(|(i_chunk, t_chunk)| {
            i_chunk
                .iter()
                .zip(t_chunk.iter())
                .fold(0.0f32, |acc, (&p, &y)| acc + bce_elem_loss(p, y, kind))
        })
        .fold(0.0f32, |acc, v| acc + v);
    let out = match reduction {
        crate::var::Reduction::Mean => {
            if numel == 0 {
                0.0
            } else {
                sum_loss / numel as f32
            }
        }
        crate::var::Reduction::Sum => sum_loss,
    };
    build_tensor(vec![out], &[])
}

/// 負対数尤度損失（`NLLLoss`）のホスト参照実装（`Var::nll_loss`
/// （`var.rs`）から `BackendOps::nll_loss` が `Unsupported` のときのみ
/// 呼ばれる。イシュー #1738・親イシュー #1609）。`class_dim` 範囲・
/// `targets` shape 一致・`0 <= t < C` 範囲検査は呼び出し元が済ませている
/// 前提（`cross_entropy_loss`〈下記〉と同じ検査配置規律）。
///
/// `class_dim` を除いた添字の組（サンプル `s = o·inner + i`）ごとに
/// `l_s = −input[(o·C + t_s)·inner + i]` を計算し、`reduction` で
/// 集約する（`N = outer·inner` はサンプル数）。空バッチ（`N == 0`）は
/// `mse_loss`（上記）の先例に合わせ `Mean`／`Sum` とも `0.0`（PyTorch
/// は `NaN`。差異は既存の `mse_loss`／`cross_entropy_loss` と同じ許容
/// 方針）。
///
/// **`ShapeError::ElementCountOverflow` を返しうる**（PR #1850
/// codex-review P1 是正: `outer`／`inner` を `class_dim` の前後で分割
/// して個別に `.iter().product()` するため、`input.shape()` 全体の
/// 要素数積は `0`（例: 先頭軸が `0`）でも、分割後の片側の部分積だけを
/// 見ると `0` を含まず `usize` オーバーフローしうる〈例:
/// `shape=[0,1,usize::MAX,2]`・`class_dim=1` は `outer=0` で全体は
/// 空だが `inner`（`shape[2..]=[usize::MAX,2]`）の積は単独で
/// オーバーフローする〉。`shape` がどこかの軸で `0` を含む場合は
/// `outer`／`inner` を計算せずに早期リターンする（イシュー #1834 の
/// `interpolate_nearest` 是正と同じ「全体が空要素であることが既知なら
/// 部分積計算を回避する」考え方）。`shape` に `0` を含まない実際に
/// 実体化済みの `Tensor` に対しては、`outer`／`inner` いずれも全体の
/// 要素数積（`Tensor::new` 側で確保済み）以下であるため
/// `checked_mul` が失敗することはない契約だが、本番経路 panic 禁止
/// 方針（`.claude/rules/coding-rust.md`）に従い防御的に `checked_mul`
/// を使う。
pub(crate) fn nll_loss(
    input: &Tensor<f32>,
    targets: &Tensor<i32>,
    class_dim: usize,
    reduction: crate::var::Reduction,
) -> Result<Tensor<f32>, ShapeError> {
    let shape = input.shape().to_vec();
    if shape.contains(&0) {
        // 空バッチ契約（関数冒頭 doc 参照）。`outer`／`inner` の部分積
        // 計算そのものを回避する。
        return Ok(build_tensor(vec![0.0], &[]));
    }
    let outer: usize = shape[..class_dim]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)?;
    let axis_len = shape[class_dim];
    let inner: usize = shape[class_dim + 1..]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)?;
    let data = dense_vec(input);
    let target_data = dense_vec_i32(targets);
    let n = outer
        .checked_mul(inner)
        .ok_or(ShapeError::ElementCountOverflow)?;

    let mut total = 0f32;
    for o in 0..outer {
        for i in 0..inner {
            let t = target_data[o * inner + i];
            // 呼び出し元（`var.rs::Var::nll_loss`）が
            // `0 <= t < axis_len` を検査済みの前提（`cross_entropy_loss`
            // 直上の同型コメント参照）。範囲外は契約違反であり
            // `unwrap()`/`expect()` を使わず `debug_assert!` で検知しつつ
            // 安全側（loss 寄与 0）へフォールバックする（`.claude/rules/
            // coding-rust.md` 本番経路 panic 禁止方針）。
            if t >= 0 && (t as usize) < axis_len {
                let idx = (o * axis_len + t as usize) * inner + i;
                total -= data[idx];
            } else {
                debug_assert!(false, "nll_loss: target 添字が範囲外（契約違反）");
            }
        }
    }

    let loss = match reduction {
        crate::var::Reduction::Mean if n > 0 => total / n as f32,
        crate::var::Reduction::Mean => 0.0,
        crate::var::Reduction::Sum => total,
    };
    Ok(build_tensor(vec![loss], &[]))
}

#[cfg(test)]
mod nll_loss_empty_shape_overflow_tests {
    use super::*;

    // PR #1850 codex-review P1 是正の回帰: `input.shape()=[0,1,
    // usize::MAX,2]`・`class_dim=1` は `Tensor::new`／`Var::nll_loss`
    // の事前検査（`checked_numel`。先頭の `0` が後続の積を吸収する）を
    // 通過するが、`nll_loss` 自身が `outer`／`inner` を `class_dim` の
    // 前後で分割して個別に `.iter().product()` していたため、`inner`
    // （`shape[2..]=[usize::MAX,2]`。ゼロを含まない部分積）が単独で
    // overflow していた（`softmax_along`／`concat` と同型の bug。上記
    // `softmax_empty_tensor_overflow_tests`／
    // `concat_empty_out_shape_overflow_tests` 参照）。`shape` に `0`
    // を含む場合は部分積を計算する前に早期 return し、`Mean`／`Sum`
    // いずれも `0.0`（空バッチ契約）を返すことを確認する。
    #[test]
    fn nll_loss_empty_shape_with_overflow_prone_inner_does_not_panic_mean() {
        let shape = [0usize, 1, usize::MAX, 2];
        let input = Tensor::<f32>::new(Vec::new(), &shape)
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let targets = Tensor::<i32>::new(Vec::new(), &[0usize, usize::MAX, 2])
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let out = nll_loss(&input, &targets, 1, crate::var::Reduction::Mean)
            .expect("空バッチは checked_mul 経路でも成功する契約");
        assert_eq!(out.shape(), &[] as &[usize]);
        assert_eq!(out.get(&[]).unwrap(), 0.0);
    }

    #[test]
    fn nll_loss_empty_shape_with_overflow_prone_inner_does_not_panic_sum() {
        let shape = [0usize, 1, usize::MAX, 2];
        let input = Tensor::<f32>::new(Vec::new(), &shape)
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let targets = Tensor::<i32>::new(Vec::new(), &[0usize, usize::MAX, 2])
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let out = nll_loss(&input, &targets, 1, crate::var::Reduction::Sum)
            .expect("空バッチは checked_mul 経路でも成功する契約");
        assert_eq!(out.shape(), &[] as &[usize]);
        assert_eq!(out.get(&[]).unwrap(), 0.0);
    }
}

/// `KlDivTarget::Probabilities`／`LogProbabilities` 共通の要素損失
/// （`kl_div_loss`／`kl_div_loss_vjp`〈`grad.rs`〉の双方から呼ばれる
/// 意味論の正。`docs/compat-api-scope.md` §1.2「損失」節参照）。
///
/// - `Probabilities`: `l = 0`（`t == 0`）／`t·(ln t − x)`（それ以外。
///   `xlogy` 規約。負・NaN の `t` は PyTorch と同じく NaN を伝播）。
/// - `LogProbabilities`: `l = exp(t)·(t − x)`。
pub(crate) fn kl_div_elem_loss(input: f32, target: f32, kind: KlDivTarget) -> f32 {
    match kind {
        KlDivTarget::Probabilities => {
            if target == 0.0 {
                0.0
            } else {
                target * (target.ln() - input)
            }
        }
        // `LogProbabilities`、および `KlDivTarget`（`#[non_exhaustive]`。
        // `tensor-core` 側で将来 variant を追加しうる）の未知 variant は
        // 同じ `LogProbabilities` 意味論へ安全側フォールバックする
        // （`eval::bce_elem_loss`〈PR #1848・イシュー #1737〉の未知
        // variant 処理と同型。本関数は infallible 契約のため `Result`
        // を返せない。未知 variant への到達は契約違反として
        // `debug_assert!` で検知するのみに留める。`.claude/rules/
        // coding-rust.md` 本番経路 panic 禁止方針）。
        kind => {
            debug_assert!(
                matches!(kind, KlDivTarget::LogProbabilities),
                "eval::kl_div_elem_loss: 未知の KlDivTarget variant へフォールバックした（契約違反）"
            );
            target.exp() * (target - input)
        }
    }
}

/// `kl_div_elem_loss` の `dInput`（`scale` を乗じる前の要素勾配。
/// `grad.rs` が上流勾配由来の `scale` を別途掛ける）。
///
/// - `Probabilities`: `−t`。
/// - `LogProbabilities`: `−exp(t)`。
pub(crate) fn kl_div_elem_grad_input(_input: f32, target: f32, kind: KlDivTarget) -> f32 {
    match kind {
        KlDivTarget::Probabilities => -target,
        // `kl_div_elem_loss` と同じ未知 variant フォールバック規律。
        kind => {
            debug_assert!(
                matches!(kind, KlDivTarget::LogProbabilities),
                "eval::kl_div_elem_grad_input: 未知の KlDivTarget variant へフォールバックした（契約違反）"
            );
            -target.exp()
        }
    }
}

/// `kl_div_elem_loss` の `dTarget`（`scale` を乗じる前）。
///
/// - `Probabilities`: `t == 0` のとき `0`（forward の `l = 0` 分岐と
///   整合。それ以外は `ln t + 1 − x`）。
/// - `LogProbabilities`: `exp(t)·(t − x + 1)`。
pub(crate) fn kl_div_elem_grad_target(input: f32, target: f32, kind: KlDivTarget) -> f32 {
    match kind {
        KlDivTarget::Probabilities => {
            if target == 0.0 {
                0.0
            } else {
                target.ln() + 1.0 - input
            }
        }
        // `kl_div_elem_loss` と同じ未知 variant フォールバック規律。
        kind => {
            debug_assert!(
                matches!(kind, KlDivTarget::LogProbabilities),
                "eval::kl_div_elem_grad_target: 未知の KlDivTarget variant へフォールバックした（契約違反）"
            );
            target.exp() * (target - input + 1.0)
        }
    }
}

/// Kullback-Leibler ダイバージェンス損失のホスト参照実装
/// （`Var::kl_div_loss`／`kl_div_loss_with_log_target`〈`var.rs`〉から
/// `BackendOps::kl_div_loss` が `Unsupported` のときのみ呼ばれる。
/// イシュー #1738）。`mse_loss`（上記）と同じ「shape 一致検査は
/// 呼び出し元が済ませている・`numel == 0` は `Mean`／`Sum` とも `0.0`」
/// 契約。
pub(crate) fn kl_div_loss(
    input: &Tensor<f32>,
    target: &Tensor<f32>,
    kind: KlDivTarget,
    reduction: crate::var::Reduction,
) -> Tensor<f32> {
    let input_data = dense_vec(input);
    let target_data = dense_vec(target);
    let numel = input_data.len();
    let sum_loss: f32 = input_data
        .iter()
        .zip(target_data.iter())
        .map(|(&x, &t)| kl_div_elem_loss(x, t, kind))
        .sum();
    let out = match reduction {
        crate::var::Reduction::Mean => {
            if numel == 0 {
                0.0
            } else {
                sum_loss / numel as f32
            }
        }
        crate::var::Reduction::Sum => sum_loss,
    };
    build_tensor(vec![out], &[])
}
/// RMSNorm（`x · rsqrt(mean(x²) + eps) · w`。`w` が `None` の場合は乗算を
/// スキップ）の行内統計（`mean`・`rstd`）を `f64` で計算する
/// （イシュー #1596）。`x_row` は 1 行分（長さ `hidden`）。
///
/// **縮約精度契約**（`.claude/rules/coding-rust.md`「正規化統計の二乗和
/// は要素を先に `f64` へ昇格してから二乗する」）: 二乗和は要素を
/// `f64` へ昇格してから `f64::mul_add` で二乗・蓄積し、`rstd` へ代入
/// する 1 回だけ `f32` へ downcast する（`backend-cpu::rmsnorm::
/// rmsnorm_row_scalar` と同じ縮約方式のホスト参照実装ミラー）。
/// `hidden == 0` は呼び出し元（[`rmsnorm_rows`]）が空出力として
/// 早期処理する契約のため、本関数は `hidden >= 1` を前提とする
/// （`inv_n` は呼び出し元が `1/hidden` を渡す）。
pub(crate) fn row_rms_stats(x_row: &[f32], eps: f32, inv_n: f64) -> f32 {
    let mut acc = 0.0f64;
    for &v in x_row {
        let v = v as f64;
        acc = v.mul_add(v, acc);
    }
    (1.0f64 / acc.mul_add(inv_n, eps as f64).sqrt()) as f32
}

/// LayerNorm（`(x − mean(x)) · rsqrt(var(x) + eps) · w + b`。分散は
/// biased ÷N）の行内統計（`mean`・`rstd`）を `f64` で計算する
/// （イシュー #1596）。`Var::layer_norm` のホスト参照実装
/// （`Unsupported` フォールバック）・`grad::vjp` の `Op::LayerNorm`
/// 逆伝播（統計再計算）の双方から呼ばれる。
///
/// 平均・分散とも [`warp_reduce_f64`] により GPU（CUDA／Metal）の
/// warp／simdgroup butterfly 縮約と同一の加算順序で蓄積する（単純な
/// 先頭からの逐次和ではない。PR #1671 codex-review P1 是正:
/// 相殺を含む入力で加算順序により結果が乖離するため。
/// `backend-cpu::layer_norm::warp_reduce_f64` doc comment 参照）。
/// [`row_rms_stats`] とは異なり二乗和ではなく「二パス分散」
/// （`Σ(x−μ)²`。`E[x²]−μ²` は使わない。実装計画 §3-3）のため専用
/// 関数とする。`hidden >= 1` を前提とする。
///
/// **`mean`／`var` は `sum`／`sq_acc` を `hidden` で直接除算して求める
/// （事前丸めした逆数 `1/hidden` との積ではない。codex-review 指摘:
/// `x=[1e30f32;49]` のような一様行で `sum * (1/hidden)` は 2 回の
/// 丸め〈逆数の丸め・乗算の丸め〉が複合し、本来 0 であるべき偏差
/// `x−mean` が巨大な非ゼロ値になり出力を歪める。IEEE 754 の
/// 除算は単一の正しく丸められた演算のため、`sum` が `hidden` 個の
/// 同一値の和である場合に丸め誤差を持ち込まない）。
///
/// **`rstd` も `f64` のまま返す**（`row_rms_stats` は `f32` downcast
/// 済みだが、LayerNorm は呼び出し元が `(x − mean)` を `f64` のまま
/// 減算する必要があり、早期に `mean`／`rstd` いずれかを `f32` へ丸める
/// と偏差計算の精度が損なわれる。codex-review 指摘: `x` の値域が
/// `f32` 仮数精度限界〈`2^24` 付近〉に達する入力で `mean` の
/// 早期丸めが出力を大きく歪める）。呼び出し元は `x̂` を書き出す
/// 直前の 1 回だけ `f32` へ downcast する。
pub(crate) fn row_ln_stats(x_row: &[f32], eps: f32, hidden: usize) -> (f64, f64) {
    let n = hidden as f64;
    let sum = warp_reduce_f64(hidden, |idx, acc| acc + x_row[idx] as f64);
    let mean = sum / n;
    let sq_acc = warp_reduce_f64(hidden, |idx, acc| {
        let d = x_row[idx] as f64 - mean;
        d.mul_add(d, acc)
    });
    let var = sq_acc / n;
    let rstd = 1.0f64 / (var + eps as f64).sqrt();
    (mean, rstd)
}

/// GPU（CUDA `kernels_layer_norm.rs`::`__shfl_xor_sync`／Metal
/// `layer_norm.metal`::`simd_shuffle_xor`）の warp／simdgroup 縮約と
/// **同一の演算順序**を再現する（`backend-cpu::layer_norm::
/// warp_reduce_f64` のホスト参照実装ミラー。PR #1671 codex-review P1
/// 指摘・イシュー #1596 是正）。詳細な背景（なぜ単純な逐次和ではなく
/// この順序へ揃えるか）は `backend-cpu::layer_norm` モジュール doc
/// comment・同名関数の doc comment を正本とし、ここでは二重管理しない。
///
/// [`row_ln_stats`] に加え、[`crate::grad::rmsnorm_vjp_rows`]／
/// [`crate::grad::layer_norm_vjp_rows`] の `dot`／`sum_dxhat`（VJP の
/// 行内縮約。符号付き項の相殺が起こりうる）からも使う（イシュー
/// #1950・PR #1995 codex-review P1 是正: `kernels_norm_backward.rs`
/// の CUDA backward カーネルは同じレーンストライド＋butterfly 順序で
/// これらを縮約するため、ホスト参照実装側をこの縮約順序へ揃えないと
/// 極端な相殺入力〈例: `dy = [1e20, 1, -1e20, 0, ...]`〉で単純逐次和
/// との差が REQ-2 統一複合判定〈相対誤差 1e-3 未満 または絶対誤差
/// 1e-5 未満〉を外れる。[`row_rms_stats`] の二乗和〈符号なし項のみで
/// 相殺が生じない〉は対象外のまま単純逐次和を維持する）。
pub(crate) fn warp_reduce_f64(hidden: usize, mut contribute: impl FnMut(usize, f64) -> f64) -> f64 {
    const LANES: usize = 32;
    let mut lanes = [0.0f64; LANES];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut idx = lane;
        while idx < hidden {
            *slot = contribute(idx, *slot);
            idx += LANES;
        }
    }
    let mut offset = 16usize;
    while offset > 0 {
        let snapshot = lanes;
        for (lane, slot) in lanes.iter_mut().enumerate() {
            *slot = snapshot[lane] + snapshot[lane ^ offset];
        }
        offset >>= 1;
    }
    lanes[0]
}

/// RMSNorm のホスト参照実装（`BackendOps::rmsnorm` が
/// `Err(BackendError::Unsupported(_))` を返したときのみ `Var::rms_norm`
/// がフォールバックする。イシュー #1596。`docs/norm-ops-design.md`）。
///
/// `x` は `[rows, hidden]` の行優先 1 次元化済みテンソル、`w` を渡す
/// 場合は長さ `hidden` を要求する（呼び出し元 `var.rs::Var::rms_norm`
/// が shape 検査済み）。`rows == 0` または `hidden == 0` は空出力を
/// 返す（`backend-cpu::rmsnorm::run_rmsnorm_f32_raw` と同じ早期
/// return 契約）。
pub(crate) fn rmsnorm_rows(
    x: &Tensor<f32>,
    w: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
) -> Tensor<f32> {
    let shape = x.shape().to_vec();
    if rows == 0 || hidden == 0 {
        return build_tensor(Vec::new(), &shape);
    }
    let data = dense_vec(x);
    let inv_n = 1.0f64 / hidden as f64;
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        let row = &data[r * hidden..(r + 1) * hidden];
        let out_row = &mut out[r * hidden..(r + 1) * hidden];
        let rstd = row_rms_stats(row, eps, inv_n);
        match w {
            Some(w) => {
                for ((o, &v), &wv) in out_row.iter_mut().zip(row.iter()).zip(w.iter()) {
                    *o = v * rstd * wv;
                }
            }
            None => {
                for (o, &v) in out_row.iter_mut().zip(row.iter()) {
                    *o = v * rstd;
                }
            }
        }
    }
    build_tensor(out, &shape)
}

/// LayerNorm のホスト参照実装（`BackendOps::layer_norm` が
/// `Err(BackendError::Unsupported(_))` を返したときのみ
/// `Var::layer_norm` がフォールバックする。イシュー #1596。
/// `docs/norm-ops-design.md`）。[`rmsnorm_rows`] と同じ shape 契約・
/// 早期 return 契約を持つ。`bias` は `w` と独立に `None` を取りうる
/// （`elementwise_affine=false` の `LayerNorm::without_affine` から
/// 呼ばれる場合等）。
pub(crate) fn layer_norm_rows(
    x: &Tensor<f32>,
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
) -> Tensor<f32> {
    let shape = x.shape().to_vec();
    if rows == 0 || hidden == 0 {
        return build_tensor(Vec::new(), &shape);
    }
    let data = dense_vec(x);
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        let row = &data[r * hidden..(r + 1) * hidden];
        let out_row = &mut out[r * hidden..(r + 1) * hidden];
        let (mean, rstd) = row_ln_stats(row, eps, hidden);
        // `mean`／`rstd` を `f64` のまま偏差計算まで保持し、`x̂` を書き出す
        // 直前の 1 回だけ `f32` へ downcast する（codex-review 指摘。
        // [`row_ln_stats`] doc 参照）。affine は CUDA カーネルの既定 FMA
        // contraction と揃えるため `w`／`b` がともに指定された場合のみ
        // `f32::mul_add` で明示的に融合する（`.claude/rules/coding-rust.md`
        // の FMA 契約統一）。`w`／`b` が `None` の演算は従来どおりスキップ
        // する（`-0.0` 等の符号付きゼロを不要な `+0.0` 加算で変えない）。
        match (w, b) {
            (Some(w), Some(b)) => {
                for (i, &v) in row.iter().enumerate() {
                    let xhat = ((v as f64 - mean) * rstd) as f32;
                    out_row[i] = xhat.mul_add(w[i], b[i]);
                }
            }
            (Some(w), None) => {
                for (i, &v) in row.iter().enumerate() {
                    let xhat = ((v as f64 - mean) * rstd) as f32;
                    out_row[i] = xhat * w[i];
                }
            }
            (None, Some(b)) => {
                for (i, &v) in row.iter().enumerate() {
                    let xhat = ((v as f64 - mean) * rstd) as f32;
                    out_row[i] = xhat + b[i];
                }
            }
            (None, None) => {
                for (i, &v) in row.iter().enumerate() {
                    out_row[i] = ((v as f64 - mean) * rstd) as f32;
                }
            }
        }
    }
    build_tensor(out, &shape)
}

/// `axis` に沿った数値安定形 softmax（シフト → `exp` → 正規化）。
/// `cross_entropy_loss`（forward。下記）の log-sum-exp 計算と
/// `grad.rs::cross_entropy_loss_vjp`（`softmax(x) − onehot(t)`）が同じ
/// 「シフトして exp・正規化する」実体を共有する（数式の実体を
/// forward/backward で二重実装しない方針。`grad.rs` 冒頭 doc）。
/// `pub(crate)`: `grad.rs` が VJP 計算で再利用する。
/// BatchNorm1d／2d のチャネルごと統計（`mean: f64`・`rstd: f64`・
/// `var: f64`〈biased ÷M〉。イシュー #1732・親 #1608）。[`row_ln_stats`]
/// と同じ二パス縮約契約（平均は `Σx/M` の直接除算、分散は `Σ(x−μ)²/M`。
/// `E[x²]−μ²` は使わない）・[`warp_reduce_f64`] による GPU butterfly
/// 縮約順序の再現契約をチャネル方向へ拡張したもの。
///
/// `x`（`[N, C, spatial]` 相当の行優先平坦化データ）のうちチャネル
/// `c` に属する `M = n*spatial` 要素（局所添字 `i in 0..M` を
/// `batch = i/spatial`・`sp = i%spatial` へ分解し、実データ添字
/// `batch*(c_total*spatial) + c*spatial + sp` へマップする、NCHW／NCL
/// 平坦化データに対するチャネル方向ストライドアクセス）を縮約する。
/// `m >= 1` を前提とする（`m == 0` は呼び出し元が早期 return する。
/// [`batch_norm_train_channels`]／[`batch_norm_infer_channels`] 参照）。
pub(crate) fn channel_bn_stats(
    x: &[f32],
    c: usize,
    c_total: usize,
    spatial: usize,
    m: usize,
    eps: f32,
) -> (f64, f64, f64) {
    let idx_of = |i: usize| -> usize {
        let batch = i / spatial;
        let sp = i % spatial;
        batch * (c_total * spatial) + c * spatial + sp
    };
    let mm = m as f64;
    let sum = warp_reduce_f64(m, |i, acc| acc + x[idx_of(i)] as f64);
    let mean = sum / mm;
    let sq_acc = warp_reduce_f64(m, |i, acc| {
        let d = x[idx_of(i)] as f64 - mean;
        d.mul_add(d, acc)
    });
    let var = sq_acc / mm;
    let rstd = 1.0f64 / (var + eps as f64).sqrt();
    (mean, rstd, var)
}

/// affine（`w`／`b`）適用の共通 4 分岐（[`layer_norm_rows`] と同じ
/// 「`w`／`b` がともに指定された場合のみ `f32::mul_add` で明示的に
/// 融合する」FMA 契約統一。`.claude/rules/coding-rust.md`）。
#[inline]
fn apply_affine(xhat: f32, w: Option<f32>, b: Option<f32>) -> f32 {
    match (w, b) {
        (Some(w), Some(b)) => xhat.mul_add(w, b),
        (Some(w), None) => xhat * w,
        (None, Some(b)) => xhat + b,
        (None, None) => xhat,
    }
}

/// BatchNorm1d／2d train モード（バッチ統計）のホスト参照実装
/// （`BackendOps::batch_norm_train` が `Err(BackendError::
/// Unsupported(_))` を返したときのみ `Var::batch_norm_with_batch_stats`
/// がフォールバックする。イシュー #1732・親 #1608・
/// `docs/batch-norm-ops-design.md`）。
///
/// `x` は `[n, c, spatial]` 相当の行優先平坦化済みテンソル、`w`／`b`
/// を渡す場合はそれぞれ長さ `c` を要求する（呼び出し元
/// `var.rs::Var::batch_norm_with_batch_stats` が shape 検査済み）。
/// `n == 0`／`c == 0`／`spatial == 0` は全ゼロ出力・全ゼロ統計を返す
/// （[`layer_norm_rows`] の `rows == 0 || hidden == 0` 早期 return と
/// 同型の契約。呼び出し元は train モードの `m <= 1` 拒否を別途行う）。
///
/// 戻り値は `(output, batch_mean, batch_var)`。`batch_mean`／
/// `batch_var`（biased ÷M）は shape `[c]` で running stats 更新専用。
pub(crate) fn batch_norm_train_channels(
    x: &Tensor<f32>,
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    n: usize,
    c: usize,
    spatial: usize,
) -> (Tensor<f32>, Vec<f32>, Vec<f32>) {
    let shape = x.shape().to_vec();
    // `n == 0 || c == 0 || spatial == 0` の判定を `m = n * spatial` の
    // 計算より**先に**行う。`spatial` は呼び出し元
    // （`var.rs::Var::batch_norm_with_batch_stats`）の `batch_norm_layout`
    // が返す値で、`N`／`C`／空間軸のいずれかが 0 の空テンソルでは
    // `spatial=0` として計算済み（`ops_shape.rs::batch_norm_layout`
    // 参照）だが、本関数はその前提を信頼せず自前でも空テンソルを
    // 積の計算前に弾く（呼び出し元契約が将来変わった場合の
    // 多層防御・本番経路 panic 禁止規約 `.claude/rules/coding-rust.md`。
    // Cursor Bugbot 指摘・イシュー #1732・PR #1874 fix ループ）。
    // 早期 return の中身自体は `shape.iter().product()` を使わず
    // `Vec::new()` を直接使い積の計算自体を避ける（`softmax_along` と
    // 同じ理由。既存是正済み）。
    if n == 0 || c == 0 || spatial == 0 {
        return (
            build_tensor(Vec::new(), &shape),
            vec![0.0f32; c],
            vec![0.0f32; c],
        );
    }
    // ここに到達した時点で `n`／`c`／`spatial` はすべて非ゼロ。
    let m = n * spatial;
    let data = dense_vec(x);
    let mut out = vec![0.0f32; data.len()];
    let mut means = vec![0.0f32; c];
    let mut vars = vec![0.0f32; c];
    for ch in 0..c {
        let (mean, rstd, var) = channel_bn_stats(&data, ch, c, spatial, m, eps);
        means[ch] = mean as f32;
        vars[ch] = var as f32;
        let wv = w.map(|w| w[ch]);
        let bv = b.map(|b| b[ch]);
        for batch in 0..n {
            for sp in 0..spatial {
                let idx = batch * (c * spatial) + ch * spatial + sp;
                let xhat = ((data[idx] as f64 - mean) * rstd) as f32;
                out[idx] = apply_affine(xhat, wv, bv);
            }
        }
    }
    (build_tensor(out, &shape), means, vars)
}

/// BatchNorm1d／2d eval モード（固定統計）のホスト参照実装
/// （`BackendOps::batch_norm_infer` が `Err(BackendError::
/// Unsupported(_))` を返したときのみ `Var::batch_norm_infer` が
/// フォールバックする。イシュー #1732・親 #1608）。[`batch_norm_train_
/// channels`] と同じ shape 契約・早期 return 契約を持つが、`mean`／
/// `var`（`nn::BatchNorm1d`／`BatchNorm2d` が保持する running stats）
/// をバッチから計算し直さずそのまま使う。
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_norm_infer_channels(
    x: &Tensor<f32>,
    mean: &[f32],
    var: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    n: usize,
    c: usize,
    spatial: usize,
) -> Tensor<f32> {
    let shape = x.shape().to_vec();
    if n == 0 || c == 0 || spatial == 0 {
        // `batch_norm_train_channels` と同じ理由（softmax_along 型の
        // 部分積オーバーフロー回避）で `shape.iter().product()` を
        // 使わず `Vec::new()` を直接返す（PR #1874 Cursor Bugbot 是正）。
        return build_tensor(Vec::new(), &shape);
    }
    let data = dense_vec(x);
    let mut out = vec![0.0f32; data.len()];
    for ch in 0..c {
        let mean_c = mean[ch] as f64;
        let rstd = 1.0f64 / (var[ch] as f64 + eps as f64).sqrt();
        let wv = w.map(|w| w[ch]);
        let bv = b.map(|b| b[ch]);
        for batch in 0..n {
            for sp in 0..spatial {
                let idx = batch * (c * spatial) + ch * spatial + sp;
                let xhat = ((data[idx] as f64 - mean_c) * rstd) as f32;
                out[idx] = apply_affine(xhat, wv, bv);
            }
        }
    }
    build_tensor(out, &shape)
}

pub(crate) fn softmax_along(input: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    // 要素数ゼロ（shape のいずれかの次元が 0）のとき、`shape[..axis]`／
    // `shape[axis+1..]` の部分積は数学的には無関係な次元（例:
    // `usize::MAX`）を含みうり、`checked_numel`（`Tensor::new` 側）が
    // 通した shape でも部分積単体では usize オーバーフローしうる
    // （全体積は途中の 0 で吸収されるが部分積はそれを経由しない）。
    // 本番経路 panic 禁止規約（`.claude/rules/coding-rust.md`）に従い、
    // outer/axis_len/inner を計算する前に空出力へ早期 return する。
    if shape.contains(&0) {
        return build_tensor(Vec::new(), &shape);
    }
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

/// `axis` に沿った数値安定形 log_softmax（`x − m − ln(Σexp(x−m))`）。
/// `softmax_along`（直上）と同じ「シフト → exp → 縮約」走査構造を共有
/// するが、`ln(softmax_along(...))` へ委譲しない（`BackendOps::
/// log_softmax` doc「`ln(softmax(x))` にしない理由」参照: softmax が
/// アンダーフローで `0.0` になった要素の `ln(0.0) = -inf` を経由すると
/// 数値精度を落とすため、解析形で直接計算する）。`pub(crate)`:
/// `grad.rs` が VJP で・`var.rs` がホストフォールバックで再利用する。
pub(crate) fn log_softmax_along(input: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    // `softmax_along` 直上と同じ早期 return（部分積オーバーフロー回避）。
    if shape.contains(&0) {
        return build_tensor(Vec::new(), &shape);
    }
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
                sum_exp += (data[idx] - m).exp();
            }
            // `m + ln(sum_exp)` を先に加算してから `data[idx]` から引くと、
            // `m` が大きい（かつ `data[idx]` と近い）場合に `m` 自身の丸め
            // 精度で `ln(sum_exp)` の寄与が失われる（例: 全要素 1e8 のとき
            // `m + ln(sum_exp)` は `1e8` に丸まり `ln(2)` 分が消え、
            // `log_softmax` が `0.0`〈期待値 `-ln(2)`〉になる）。
            // `data[idx] - m` は Sterbenz の補題により丸め誤差なしで計算
            // できるため、先にこちらを計算してから `ln(sum_exp)` を引く
            // 順序（`(x - m) - ln(sum_exp)`）で丸め落ちを避ける。
            let ln_sum_exp = sum_exp.ln();
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                out[idx] = (data[idx] - m) - ln_sum_exp;
            }
        }
    }
    build_tensor(out, &shape)
}

/// `axis` に沿った累積和のホスト参照実装（`torch.cumsum` 相当。
/// イシュー #1731）。`softmax_along`／`log_softmax_along` と同じ
/// 「外側（outer）× 走査軸（axis_len）× 内側（inner）」の 3 段走査
/// だが、走査軸方向は独立な行ごとの縮約ではなく `dim` の添字昇順の
/// **逐次スキャン**（各ステップの出力が直前までの累積値に依存する）
/// である点が異なる。`BackendOps::cumsum` が `Unsupported` を返した
/// ときのみ `Var::cumsum` から呼ばれる（softmax と同じ「バックエンド
/// 実装 → フォールバック」の二段構成）。
///
/// **数値契約**（[`fandhe_ai_tensor_core::BackendOps::cumsum`] doc
/// と同一）: lane（`o`・`i` の組）ごとに `f64` アキュムレータを保持し、
/// `acc = acc + (x[idx] as f64)` を計算するたびにその時点の `acc` を
/// `f32` へ downcast したスナップショットを `out[idx]` に書き出す
/// （次ステップは downcast 後の `f32` を読み戻さない。`.claude/rules/
/// coding-rust.md` の f64 アキュムレータ方針を forward の scan へ
/// 拡張したもの。`backend-cpu::scan::cumsum` と同一アルゴリズムであり
/// 出力は `to_bits` で完全一致する契約）。
pub(crate) fn cumsum_along(input: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    // `softmax_along` 冒頭と同じ早期 return（部分積オーバーフロー回避。
    // `shape` のいずれかの次元が 0 なら空出力を返す）。
    if shape.contains(&0) {
        return build_tensor(Vec::new(), &shape);
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let data = dense_vec(input);
    let mut out = vec![0f32; data.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc: f64 = 0.0;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                acc += data[idx] as f64;
                out[idx] = acc as f32;
            }
        }
    }
    build_tensor(out, &shape)
}

/// `axis` に沿った累積積のホスト参照実装（`torch.cumprod` 相当。
/// イシュー #1731）。[`cumsum_along`] と同じ lane 構造・早期 return
/// だが、アキュムレータは `1.0` から開始し `acc = acc * (x[idx] as
/// f64)` を計算する（`BackendOps::cumprod` doc と同一契約。
/// `backend-cpu::scan::cumprod` と bit 完全一致する）。
pub(crate) fn cumprod_along(input: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    if shape.contains(&0) {
        return build_tensor(Vec::new(), &shape);
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let data = dense_vec(input);
    let mut out = vec![0f32; data.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc: f64 = 1.0;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                acc *= data[idx] as f64;
                out[idx] = acc as f32;
            }
        }
    }
    build_tensor(out, &shape)
}

/// `inputs` を `dim` 軸で連結するホスト参照実装（`torch.cat` 相当。
/// イシュー #1598）。`BackendOps::concat` が `Unsupported` を返した
/// ときのみ `grad::concat_with_fallback` から呼ばれる（`softmax_along`
/// と同じ「バックエンド実装 → フォールバック」の二段構成。判定迂回
/// 経路を作らない）。
///
/// `out_shape` は呼び出し元（`Var::cat`／`grad::concat_with_fallback`）
/// が [`fandhe_ai_tensor_core::concat_out_shape`] で検査・確定済みの
/// 出力 shape をそのまま渡す（本関数は shape 再検査を行わない前提）。
/// `inputs` は strided view（`dense_vec_ref` で稠密化してから読む）で
/// よい——`Op::Narrow` の VJP（`grad::concat_with_fallback` 経由）が
/// zero-pad テンソルを渡す際、そのテンソル自体は contiguous のため
/// 実害はないが、`Var::cat` の入力が transpose 直後の view でも
/// 正しく動く契約とする。
///
/// レイアウト分解: `outer = prod(out_shape[..dim])`・
/// `inner = prod(out_shape[dim+1..])`・出力の線形添字は
/// `(o * total + off_i + s) * inner + i`
/// （`o`: outer 添字・`s`: 入力 i 内の dim 添字・`i`: inner 添字・
/// `off_i`: 入力 i より前の dim 累積長・`total = out_shape[dim]`）。
pub(crate) fn concat(inputs: &[&Tensor<f32>], dim: usize, out_shape: &[usize]) -> Tensor<f32> {
    // 要素数ゼロ（`out_shape` のいずれかの次元が 0）のとき、
    // `out_shape[..dim]`／`out_shape[dim+1..]` の部分積は数学的には
    // 無関係な次元（例: `usize::MAX`）を含みうり、`concat_out_shape`
    // が通した shape でも部分積単体では usize オーバーフローしうる
    // （全体積は途中の 0 で吸収されるが部分積はそれを経由しない）。
    // `softmax_along`（本ファイル上部）と同じ理由・同じ対処で、
    // 本番経路 panic 禁止規約（`.claude/rules/coding-rust.md`）に従い
    // outer/inner/out_numel を計算する前に空出力へ早期 return する。
    if out_shape.contains(&0) {
        return build_tensor(Vec::new(), out_shape);
    }
    let outer: usize = out_shape[..dim].iter().product();
    let inner: usize = out_shape[dim + 1..].iter().product();
    let total = out_shape[dim];
    let out_numel: usize = out_shape.iter().product();
    let mut out = vec![0f32; out_numel];
    if outer == 0 || inner == 0 || total == 0 {
        return build_tensor(out, out_shape);
    }
    let mut off = 0usize;
    for input in inputs {
        let seg = input.shape()[dim];
        if seg == 0 {
            continue;
        }
        let data = dense_vec_ref(input);
        for o in 0..outer {
            for s in 0..seg {
                let src_row_start = (o * seg + s) * inner;
                let dst_row_start = (o * total + off + s) * inner;
                out[dst_row_start..dst_row_start + inner]
                    .copy_from_slice(&data[src_row_start..src_row_start + inner]);
            }
        }
        off += seg;
    }
    build_tensor(out, out_shape)
}

/// 条件テンソルによる要素選択のホスト参照実装（`torch.where` 相当。
/// イシュー #1637）。`BackendOps::where_cond` が `Unsupported` を
/// 返したときのみ `Var::where_cond` から呼ばれる（`concat` と同じ
/// 「バックエンド実装 → フォールバック」二段構成）。
///
/// `cond`／`a`／`b` はいずれも呼び出し元（`Var::where_cond`）が
/// `out_shape` へ broadcast 済み（`cond` は f32 マスクへ変換済み）で
/// あることを前提とし、本関数は shape 再検査を行わない。真偽判定は
/// [`fandhe_ai_tensor_core::BackendOps::where_cond`] と同じ
/// `c != 0.0` 契約。`dense_vec_ref` で稠密化してから読むため、
/// strided view（broadcast view 等）でも正しく動く。
pub(crate) fn where_cond(
    cond: &Tensor<f32>,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    out_shape: &[usize],
) -> Tensor<f32> {
    if out_shape.contains(&0) {
        return build_tensor(Vec::new(), out_shape);
    }
    let cond_data = dense_vec_ref(cond);
    let a_data = dense_vec_ref(a);
    let b_data = dense_vec_ref(b);
    let out: Vec<f32> = cond_data
        .iter()
        .zip(a_data.iter())
        .zip(b_data.iter())
        .map(|((&c, &av), &bv)| if c != 0.0 { av } else { bv })
        .collect();
    build_tensor(out, out_shape)
}

/// マスク位置を定数で置換するホスト参照実装（`torch.masked_fill`
/// 相当。イシュー #1637）。`BackendOps::masked_fill` が
/// `Unsupported` を返したときのみ `Var::masked_fill` から呼ばれる。
///
/// `x`／`mask` は呼び出し元が同一 shape（`mask` は broadcast 済み
/// f32 マスク）であることを保証済み。`mask != 0.0` の位置を `value`
/// に置換し、それ以外は `x` の値をそのまま返す。
pub(crate) fn masked_fill(x: &Tensor<f32>, mask: &Tensor<f32>, value: f32) -> Tensor<f32> {
    let shape = x.shape().to_vec();
    if shape.contains(&0) {
        return build_tensor(Vec::new(), &shape);
    }
    let x_data = dense_vec_ref(x);
    let mask_data = dense_vec_ref(mask);
    let out: Vec<f32> = x_data
        .iter()
        .zip(mask_data.iter())
        .map(|(&xv, &mv)| if mv != 0.0 { value } else { xv })
        .collect();
    build_tensor(out, &shape)
}

/// 定数パディングのホスト参照実装（`torch.nn.functional.pad
/// (mode='constant')` 相当。イシュー #1756）。`BackendOps::pad` が
/// `Unsupported` を返したときのみ `grad::pad_with_fallback` から
/// 呼ばれる。
///
/// `out_shape` は呼び出し元（`Var::pad`／`grad::pad_with_fallback`）が
/// [`fandhe_ai_tensor_core::pad_out_shape`] で検査・確定済みの出力
/// shape をそのまま渡す（本関数は shape 再検査を行わない前提）。
/// `input` は strided view（`dense_vec_ref` で稠密化してから読む）で
/// よい。算術を含まない純粋なコピー演算のため、`input` の要素は
/// bit そのままパディング先へ写す（`value` も同様に bit そのまま
/// 書き込む——バックエンド間 bit 完全一致契約。`.claude/rules/
/// coding-rust.md` 数値契約節参照）。
///
/// レイアウト分解: 出力を全域 `value` で初期化した後、`input` の
/// 各「行」（最終軸を除く多次元添字ごとの最内軸スライス）を
/// `before` オフセット分だけ平行移動した出力位置へ `copy_from_slice`
/// する（`concat` と同型の「最内軸は行単位でコピー」方針）。
pub(crate) fn pad(
    input: &Tensor<f32>,
    pads: &[(usize, usize)],
    value: f32,
    out_shape: &[usize],
) -> Tensor<f32> {
    // `out_shape` のいずれかの次元が 0 のとき、他の次元同士の部分積
    // （本関数は要素数積を一括計算するため対象は全体積そのものだが、
    // `concat`／`softmax_along` と同じ理由で）が `usize` オーバーフロー
    // しうる（例: `[0, usize::MAX, 2]` は全体積が 0 だが `usize::MAX * 2`
    // の時点で既にオーバーフローする）ため、積を計算する前に空出力へ
    // 早期 return する（本番経路 panic 禁止規約・`.claude/rules/
    // coding-rust.md`）。
    if out_shape.contains(&0) {
        return build_tensor(Vec::new(), out_shape);
    }
    let rank = out_shape.len();
    let out_numel: usize = out_shape.iter().product();
    let mut out = vec![value; out_numel];
    let in_shape = input.shape().to_vec();
    if in_shape.contains(&0) {
        // 入力が空でも出力は非空になりうる（全要素 `value`）。
        return build_tensor(out, out_shape);
    }
    let data = dense_vec_ref(input);
    if rank == 0 {
        // rank 0（スカラー）は pads が空のため恒等コピー。
        out[0] = data[0];
        return build_tensor(out, out_shape);
    }
    let out_strides = row_major_strides(out_shape);
    let inner = in_shape[rank - 1];
    let front_shape = &in_shape[..rank - 1];
    let outer: usize = front_shape.iter().product();
    let before_inner = pads[rank - 1].0;
    for o in 0..outer {
        let front_idx = unravel(o, front_shape);
        let mut out_offset = before_inner;
        for (axis, &fi) in front_idx.iter().enumerate() {
            out_offset += (fi + pads[axis].0) * out_strides[axis];
        }
        let src_start = o * inner;
        out[out_offset..out_offset + inner].copy_from_slice(&data[src_start..src_start + inner]);
    }
    build_tensor(out, out_shape)
}

/// Conv2d の im2col ホスト参照実装（`BackendOps::im2col` が
/// `Unsupported` を返したときのみ `grad::im2col_with_fallback` から
/// 呼ばれる。イシュー #1764・設計 `docs/conv-ops-design.md`）。
///
/// CPU 実装（`backend-cpu::im2col::im2col`）と数式・走査順が完全に
/// 同一のホスト側複製（クレート間依存を作らないため。`constant_pad.rs`
/// と `eval::pad` の関係と同型）。`out_shape` は呼び出し元が
/// [`fandhe_ai_tensor_core::im2col_out_shape`] で検査・確定済みの
/// `[N, G, Cin_g·kH·kW, Hout·Wout]` をそのまま渡す。
pub(crate) fn im2col(
    input: &Tensor<f32>,
    params: &fandhe_ai_tensor_core::Conv2dParams,
    out_shape: &[usize],
) -> Tensor<f32> {
    let out_numel: usize = out_shape.iter().product();
    if out_numel == 0 {
        return build_tensor(Vec::new(), out_shape);
    }
    let in_shape = input.shape();
    let (h_in, w_in) = (in_shape[2], in_shape[3]);
    let n_batch = out_shape[0];
    let groups = out_shape[1];
    let k_g = out_shape[2];
    let p = out_shape[3];
    let cin_g = in_shape[1] / groups.max(1);
    let [kh_k, kw_k] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();
    let w_out = conv2d_dim_out_len(w_in, kw_k, sw, pw, dw);

    let mut out = vec![0f32; out_numel];
    for n in 0..n_batch {
        for g in 0..groups {
            for k_idx in 0..k_g {
                let kw_ = k_idx % kw_k;
                let rest = k_idx / kw_k;
                let kh_ = rest % kh_k;
                let c_g = rest / kh_k;
                let c = g * cin_g + c_g;
                for p_idx in 0..p {
                    let ow = p_idx % w_out;
                    let oh = p_idx / w_out;
                    let h_pos = conv2d_window_input_pos(oh, sh, kh_, dh, ph);
                    let w_pos = conv2d_window_input_pos(ow, sw, kw_, dw, pw);
                    let value = match (h_pos, w_pos) {
                        (Some(h), Some(w)) if h < h_in && w < w_in => {
                            input.get(&[n, c, h, w]).unwrap_or(0.0)
                        }
                        _ => 0.0,
                    };
                    let out_idx = ((n * groups + g) * k_g + k_idx) * p + p_idx;
                    out[out_idx] = value;
                }
            }
        }
    }
    build_tensor(out, out_shape)
}

/// Conv2d の col2im ホスト参照実装（`BackendOps::col2im` が
/// `Unsupported` を返したときのみ `grad::col2im_with_fallback` から
/// 呼ばれる。[`im2col`] の随伴（転置畳み込み）。イシュー #1764）。
///
/// `backend-cpu::im2col::col2im` と数式・走査順・`f64` アキュムレータ
/// 契約が完全に同一のホスト側複製。
pub(crate) fn col2im(
    d_col: &Tensor<f32>,
    input_shape: &[usize],
    params: &fandhe_ai_tensor_core::Conv2dParams,
) -> Tensor<f32> {
    let out_numel: usize = input_shape.iter().product();
    if out_numel == 0 {
        return build_tensor(Vec::new(), input_shape);
    }
    let (n_batch, cin, h_in, w_in) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let d_col_shape = d_col.shape();
    let groups = d_col_shape[1];
    let p = d_col_shape[3];
    let cin_g = cin / groups.max(1);
    let [kh_k, kw_k] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();
    let h_out = conv2d_dim_out_len(h_in, kh_k, sh, ph, dh);
    let w_out = conv2d_dim_out_len(w_in, kw_k, sw, pw, dw);
    debug_assert_eq!(
        h_out.checked_mul(w_out),
        Some(p),
        "eval::col2im: d_col の P 軸が conv2d_dim_out_len から再計算した Hout*Wout と一致しない（契約違反）"
    );

    let mut out = vec![0f32; out_numel];
    for n in 0..n_batch {
        for c in 0..cin {
            let g = c / cin_g.max(1);
            let c_g = c % cin_g.max(1);
            for h in 0..h_in {
                for w in 0..w_in {
                    let mut acc: f64 = 0.0;
                    for kh_ in 0..kh_k {
                        let oh = match conv2d_window_out_idx(h, ph, kh_, dh, sh, h_out) {
                            Some(v) => v,
                            None => continue,
                        };
                        for kw_ in 0..kw_k {
                            let ow = match conv2d_window_out_idx(w, pw, kw_, dw, sw, w_out) {
                                Some(v) => v,
                                None => continue,
                            };
                            let k_idx = (c_g * kh_k + kh_) * kw_k + kw_;
                            let p_idx = oh * w_out + ow;
                            let v = d_col.get(&[n, g, k_idx, p_idx]).unwrap_or(0.0);
                            acc += f64::from(v);
                        }
                    }
                    let out_idx = ((n * cin + c) * h_in + h) * w_in + w;
                    out[out_idx] = acc as f32;
                }
            }
        }
    }
    build_tensor(out, input_shape)
}

/// Conv2d の forward ホスト参照実装（全ホスト im2col＋[`matmul`] の
/// per-`(n, g)` 合成。イシュー #1764・設計 `docs/conv-ops-design.md`
/// §5.3「本番フォールバックではなくテストオラクル／全ホスト参照専用」）。
///
/// `bias` は `[Cout]` を仮定し、GEMM 結果へ 1 回加算する（`Var::add`
/// の broadcast を経由せず直接計算するため、bias 軸誤加算の罠
/// 〈設計 doc §5.2「実装上の注意」〉が構造的に発生しない）。
#[cfg(test)]
pub(crate) fn conv2d(
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    params: &fandhe_ai_tensor_core::Conv2dParams,
    out_shape: &[usize],
) -> Tensor<f32> {
    let out_numel: usize = out_shape.iter().product();
    if out_numel == 0 {
        return build_tensor(Vec::new(), out_shape);
    }
    let im2col_out_shape = fandhe_ai_tensor_core::im2col_out_shape(input.shape(), params)
        .unwrap_or_else(|_| {
            debug_assert!(
                false,
                "eval::conv2d: im2col_out_shape 検査済みのはずが失敗した（契約違反）"
            );
            vec![0; 4]
        });
    let col = im2col(input, params, &im2col_out_shape);
    let (n_batch, groups, k_g, p) = (
        im2col_out_shape[0],
        im2col_out_shape[1],
        im2col_out_shape[2],
        im2col_out_shape[3],
    );
    let cout = out_shape[1];
    let cout_g = cout / groups.max(1);
    let bias_data = bias.map(dense_vec);
    let [kh_k, kw_k] = params.kernel_size();

    let mut out = vec![0f32; out_numel];
    for n in 0..n_batch {
        for g in 0..groups {
            // col の (n, g) スライス: [K_g, P]
            let col_ng: Vec<f32> = (0..k_g * p)
                .map(|i| {
                    let k_idx = i / p;
                    let p_idx = i % p;
                    col.get(&[n, g, k_idx, p_idx]).unwrap_or(0.0)
                })
                .collect();
            let col_mat = build_tensor(col_ng, &[k_g, p]);
            // weight の (g) スライス: [Cout_g, K_g]（K_g = Cin_g*kH*kW を
            // (c_in_g, kh, kw) の row-major で並べる。im2col と同順）。
            let w_ng: Vec<f32> = (0..cout_g * k_g)
                .map(|i| {
                    let co_g = i / k_g;
                    let k_idx = i % k_g;
                    let co = g * cout_g + co_g;
                    let kw_ = k_idx % kw_k;
                    let rest = k_idx / kw_k;
                    let kh_ = rest % kh_k;
                    let c_g = rest / kh_k;
                    weight.get(&[co, c_g, kh_, kw_]).unwrap_or(0.0)
                })
                .collect();
            let w_mat = build_tensor(w_ng, &[cout_g, k_g]);
            let out_ng = matmul(&w_mat, &col_mat); // [Cout_g, P]
            for co_g in 0..cout_g {
                let co = g * cout_g + co_g;
                let bias_v = bias_data
                    .as_ref()
                    .and_then(|d| d.get(co))
                    .copied()
                    .unwrap_or(0.0);
                for p_idx in 0..p {
                    let v = out_ng.get(&[co_g, p_idx]).unwrap_or(0.0) + bias_v;
                    let out_idx = ((n * cout + co) * p) + p_idx;
                    out[out_idx] = v;
                }
            }
        }
    }
    build_tensor(out, out_shape)
}

/// 直接畳み込みオラクル（`(c, kh, kw)` 昇順 `mul_add` 連鎖。padding
/// タップは非スキップ。CPU im2col＋GEMM との bit 完全一致テストオラクル
/// 専用。イシュー #1764・設計 `docs/conv-ops-design.md` §7）。
#[cfg(test)]
pub(crate) fn conv2d_direct(
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    params: &fandhe_ai_tensor_core::Conv2dParams,
    out_shape: &[usize],
) -> Tensor<f32> {
    let out_numel: usize = out_shape.iter().product();
    if out_numel == 0 {
        return build_tensor(Vec::new(), out_shape);
    }
    let in_shape = input.shape();
    let (cin, h_in, w_in) = (in_shape[1], in_shape[2], in_shape[3]);
    let (n_batch, cout, h_out, w_out) = (out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    let groups = params.groups();
    let cin_g = cin / groups.max(1);
    let cout_g = cout / groups.max(1);
    let [kh_k, kw_k] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();
    let bias_data = bias.map(dense_vec);

    let mut out = vec![0f32; out_numel];
    for n in 0..n_batch {
        for co in 0..cout {
            let g = co / cout_g.max(1);
            for oh in 0..h_out {
                for ow in 0..w_out {
                    // 設計 doc §7: `acc = 0.0` から (c, kh, kw) 昇順の
                    // `mul_add` 連鎖で積和を求め、bias は連鎖に混ぜず
                    // 最後に 1 回だけ加算する（im2col＋GEMM 側が
                    // 「GEMM 結果へ bias を後から加算する」という
                    // 独立した加算パスであることと一致させるため。
                    // bias をアキュムレータの初期値にすると FMA 連鎖の
                    // 丸め順序が変わり bit 不一致になる）。
                    let mut acc = 0f32;
                    for c_g in 0..cin_g {
                        let c = g * cin_g + c_g;
                        for kh_ in 0..kh_k {
                            for kw_ in 0..kw_k {
                                let h_pos = conv2d_window_input_pos(oh, sh, kh_, dh, ph);
                                let w_pos = conv2d_window_input_pos(ow, sw, kw_, dw, pw);
                                let x = match (h_pos, w_pos) {
                                    (Some(h), Some(w)) if h < h_in && w < w_in => {
                                        input.get(&[n, c, h, w]).unwrap_or(0.0)
                                    }
                                    _ => 0.0,
                                };
                                let w_v = weight.get(&[co, c_g, kh_, kw_]).unwrap_or(0.0);
                                acc = x.mul_add(w_v, acc);
                            }
                        }
                    }
                    let bias_v = bias_data
                        .as_ref()
                        .and_then(|d| d.get(co))
                        .copied()
                        .unwrap_or(0.0);
                    let out_idx = ((n * cout + co) * h_out + oh) * w_out + ow;
                    out[out_idx] = acc + bias_v;
                }
            }
        }
    }
    build_tensor(out, out_shape)
}

/// Conv2d の出力空間長（PyTorch `_conv_output_size` 相当。`eval` 内部
/// 用の非 fallible 複製——`fandhe_ai_tensor_core::conv_out_len` は
/// `Result` を返すが、`eval` の呼び出し元はいずれも呼び出し元が既に
/// shape 検査済みの契約〈モジュール doc「shape の事前検査は呼び出し元
/// が済ませてから本モジュールを呼ぶ契約」〉のため `debug_assert!` で
/// 契約違反を検知しつつ `0` で安全側に吸収する）。
fn conv2d_dim_out_len(in_len: usize, k: usize, s: usize, p: usize, d: usize) -> usize {
    match fandhe_ai_tensor_core::conv_out_len(in_len, k, s, p, d) {
        Ok(v) => v,
        Err(_) => {
            debug_assert!(
                false,
                "conv2d_dim_out_len: 呼び出し元の shape 検査済み契約が崩れた"
            );
            0
        }
    }
}

/// im2col の走査で使う「窓添字 → 入力座標」の符号安全な逆変換
/// （`backend-cpu::im2col::im2col_input_pos` と同型の独立実装）。
fn conv2d_window_input_pos(
    out_idx: usize,
    stride: usize,
    k: usize,
    dilation: usize,
    padding: usize,
) -> Option<usize> {
    let base = out_idx.checked_mul(stride)?;
    let offset = k.checked_mul(dilation)?;
    let sum = base.checked_add(offset)?;
    sum.checked_sub(padding)
}

/// col2im の走査で使う「入力座標 → 窓添字」の符号安全な逆変換
/// （`backend-cpu::im2col::col2im_out_idx` と同型の独立実装。設計
/// doc §6.2）。
fn conv2d_window_out_idx(
    pos: usize,
    padding: usize,
    k: usize,
    dilation: usize,
    stride: usize,
    out_len: usize,
) -> Option<usize> {
    let offset = k.checked_mul(dilation)?;
    let pos_plus_p = pos.checked_add(padding)?;
    let numerator = pos_plus_p.checked_sub(offset)?;
    if numerator % stride != 0 {
        return None;
    }
    let out_idx = numerator / stride;
    if out_idx < out_len {
        Some(out_idx)
    } else {
        None
    }
}

/// 行優先（C-order）ストライドを計算する（`Tensor::contiguous()` が
/// 実体化する順序と同一の走査順を、`gather`／`scatter`（下記）が
/// 独自に `input`／`index` の多次元添字から線形添字を導出するために
/// 使う。CPU バックエンドクレートの reduction モジュールにある同名の
/// 補助関数と対の関係にある独立実装——`autodiff` → 具体バックエンド
/// クレートへの依存は作れない〈`.claude/rules/coding-rust.md`／
/// `docs/fusion-graph-design.md` §3.4〉ため、ここでは重複実装する）。
pub(crate) fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// 線形添字（行優先）を `shape` の多次元添字へ展開する（上記
/// `row_major_strides` と対。CPU バックエンドクレートの reduction
/// モジュールにある同名の補助関数と同型の独立実装）。
pub(crate) fn unravel(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; shape.len()];
    for (axis, &d) in shape.iter().enumerate().rev() {
        if d == 0 {
            out[axis] = 0;
            continue;
        }
        out[axis] = idx % d;
        idx /= d;
    }
    out
}

/// `dim` 軸に沿った独立読み出しのホスト参照実装（`torch.gather`
/// 相当。イシュー #1776）。`BackendOps::gather` が `Unsupported` を
/// 返したときのみ `grad::gather_with_fallback` から呼ばれる。
///
/// `out_shape` は呼び出し元（`Var::gather`／
/// `grad::gather_with_fallback`）が
/// [`fandhe_ai_tensor_core::gather_out_shape`] で検査・確定済みの
/// 出力 shape（＝`index.shape()`）をそのまま渡す。`index` の値は
/// `[0, input.shape()[dim])` の範囲内であることを `Var::gather` が
/// forward 時点で検査済み（本関数は値検査を行わない前提）。
///
/// `input`／`index` は strided view でよい（`dense_vec_ref`／
/// `dense_vec_i32` で行優先稠密化してから読む）。出力位置同士の
/// 書き込み衝突がないため決定的集約順序の契約は不要
/// （[`fandhe_ai_tensor_core::ScatterReduce`] doc 参照）。
pub(crate) fn gather(
    input: &Tensor<f32>,
    dim: usize,
    index: &Tensor<i32>,
    out_shape: &[usize],
) -> Tensor<f32> {
    if out_shape.contains(&0) {
        return build_tensor(Vec::new(), out_shape);
    }
    let input_shape = input.shape().to_vec();
    let input_data = dense_vec_ref(input);
    let input_strides = row_major_strides(&input_shape);
    let index_data = dense_vec_i32(index);

    let numel: usize = out_shape.iter().product();
    let mut out = vec![0f32; numel];
    for (flat, out_val) in out.iter_mut().enumerate() {
        let coords = unravel(flat, out_shape);
        let dim_idx = index_data[flat] as usize;
        let mut pos = 0usize;
        for (axis, &stride) in input_strides.iter().enumerate() {
            let coord = if axis == dim { dim_idx } else { coords[axis] };
            pos += coord * stride;
        }
        *out_val = input_data[pos];
    }
    build_tensor(out, out_shape)
}

/// `dim` 軸に沿った書き込みのホスト参照実装（`torch.scatter`／
/// `torch.scatter_add` 相当。`reduce` で分岐。イシュー #1776）。
/// `BackendOps::scatter` が `Unsupported` を返したときのみ
/// `grad::scatter_with_fallback` から呼ばれる。出力 shape は
/// `input.shape()` と恒等（scatter は shape を変えない）。`index`／
/// `src` は同一 shape であることを呼び出し元
/// （[`fandhe_ai_tensor_core::scatter_out_shape`]）が検査済み。
/// `index` の値は `[0, input.shape()[dim])` の範囲内であることを
/// `Var::scatter`／`scatter_add` が forward 時点で検査済み。
///
/// **決定的集約契約（[`fandhe_ai_tensor_core::ScatterReduce`] doc を
/// 正とする）**: `index`／`src` を行優先（row-major）で走査し、同一
/// 出力位置への複数回書き込みは「この走査順で逐次処理」した結果と
/// する。`Overwrite` は最後に処理された値が残る単純代入、`Add` は
/// 出力位置ごとの `f64` アキュムレータへ逐次加算し走査完了後に
/// **1 回だけ** `f32` へ downcast する（未書き込み位置も
/// `input[pos] as f64` → `as f32` の往復のみを経る。`f32` は
/// `f64` へ丸めなしで昇格でき、加算がなければ最近接偶数丸めで元の
/// 値に戻るため実害はない）。単一スレッド逐次ループで実装する
/// （並列化する場合は本関数と同じ観測結果になる集約方式——出力位置
/// ごとの排他アキュムレータ・決定的な reduce 木——を選定すること。
/// `.claude/rules/out-of-scope-tracking.md` 対象・並列化自体は本
/// issue のスコープ外）。
pub(crate) fn scatter(
    input: &Tensor<f32>,
    dim: usize,
    index: &Tensor<i32>,
    src: &Tensor<f32>,
    reduce: ScatterReduce,
) -> Tensor<f32> {
    let out_shape = input.shape().to_vec();
    if out_shape.contains(&0) {
        return build_tensor(Vec::new(), &out_shape);
    }
    let input_data = dense_vec_ref(input);
    let input_strides = row_major_strides(&out_shape);
    let index_data = dense_vec_i32(index);
    let index_shape = index.shape().to_vec();
    let src_data = dense_vec_ref(src);
    let index_numel: usize = index_shape.iter().product();

    // `flat`（`index`／`src` の行優先線形添字）から書き込み先の
    // `input`（＝`out_shape`）行優先線形添字を導出する共通ロジック
    // （`Overwrite`／`Add` 両分岐・フォールバック分岐で共有し、
    // 重複実装によるドリフトを避ける）。
    let resolve_pos = |flat: usize| -> usize {
        let coords = unravel(flat, &index_shape);
        let dim_idx = index_data[flat] as usize;
        let mut pos = 0usize;
        for (axis, &stride) in input_strides.iter().enumerate() {
            let coord = if axis == dim { dim_idx } else { coords[axis] };
            pos += coord * stride;
        }
        pos
    };

    match reduce {
        ScatterReduce::Add => {
            let mut acc: Vec<f64> = input_data.iter().map(|&v| v as f64).collect();
            for flat in 0..index_numel {
                let pos = resolve_pos(flat);
                acc[pos] += src_data[flat] as f64;
            }
            let out: Vec<f32> = acc.iter().map(|&v| v as f32).collect();
            build_tensor(out, &out_shape)
        }
        // `Overwrite`、および `ScatterReduce`（`#[non_exhaustive]`。
        // `tensor-core` 側で将来 variant を追加しうる）の未知 variant
        // は同じ「上書き」意味論へ安全側フォールバックする（本関数は
        // infallible 契約のため `Result` を返せない。未知 variant への
        // 到達は契約違反として `debug_assert!` で検知するのみに留め、
        // release ビルドでは黙って `Overwrite` として振る舞う。
        // `.claude/rules/coding-rust.md` 本番経路 panic 禁止方針）。
        reduce => {
            debug_assert!(
                matches!(reduce, ScatterReduce::Overwrite),
                "eval::scatter: 未知の ScatterReduce variant へフォールバックした（契約違反）"
            );
            let mut out = input_data.to_vec();
            for flat in 0..index_numel {
                let pos = resolve_pos(flat);
                out[pos] = src_data[flat];
            }
            build_tensor(out, &out_shape)
        }
    }
}

/// `interpolate`（[`fandhe_ai_tensor_core::InterpolateMode::Nearest`]。`Var::interpolate`・
/// `grad::nearest_src_index_map` 双方が使う）の 1 軸単位の添字ヘルパー
/// （イシュー #1757）。**単一情報源**: forward のホスト参照実装
/// （下記 [`interpolate_nearest`]）と backward の VJP index 構築
/// （`grad::nearest_src_index_map`）が本関数を共有することで、両者が
/// 別々に添字式を書いて乖離するのを防ぐ（別々に書くと CPU
/// `ForceEvalFallback` テストでは forward／backward の乖離を検出
/// できないため。実装計画「設計判断」§3.2 参照）。
///
/// 添字式 `src = (dst * in_size) / out_size`（整数除算＝床）。
/// `dst < out_size` より数学的に `src < in_size` が自動的に成立するが、
/// REQ-8 の縦深防御として `min(src, in_size - 1)` を明示的に取る
/// （`.claude/rules/coding-rust.md`「境界検査を省略しない」）。
/// `out_size == 0` の場合は呼び出されない契約（呼び出し元が
/// `interpolate_out_shape` で事前に拒否済み）だが、防御的に `0` を
/// 返す（ゼロ除算 panic を避ける）。
///
/// 中間積 `dst * in_size` は `usize`（64bit 環境で最大約 1.8e19）を
/// 素朴な乗算で計算すると overflow しうる（例: `[1]` を
/// `broadcast_to` で `[1usize << 63]` へ拡張した view を `[3]` へ
/// 縮小する interpolate では `dst=2` の `dst * in_size` が `2^63 * 2`
/// を超える。backend-cpu クレートの `interpolate::nearest_src_coord`
/// と同型の overflow）。`u128`（最大 2^128 - 1）へ昇格して積・除算を
/// 行うことで、`dst`・`in_size` とも `usize::MAX` の場合でも `u128`
/// の範囲に収まり overflow しない（本関数は `grad::
/// nearest_src_index_map`〈backward の index 構築〉から forward
/// バックエンドの成否に関わらず常に呼ばれるため、本番経路 panic
/// 禁止規約に直結する。イシュー #1834 codex-review P1 是正）。
pub(crate) fn nearest_src_coord(dst: usize, in_size: usize, out_size: usize) -> usize {
    if out_size == 0 || in_size == 0 {
        return 0;
    }
    let src = (dst as u128 * in_size as u128) / out_size as u128;
    // `src < in_size <= usize::MAX` が `u128` 除算の結果として保証
    // されるため（`dst < out_size` より `src < in_size`）、`as usize`
    // への縮小は安全（真の値が `usize` の範囲を超えることはない）。
    (src as usize).min(in_size - 1)
}

/// `interpolate`（`torch.nn.functional.interpolate(mode='nearest')`
/// 相当）のホスト参照実装（イシュー #1757）。`BackendOps::interpolate`
/// が `Unsupported` を返したときのみ
/// `grad::interpolate_with_fallback` から呼ばれる。
///
/// `size` は末尾空間軸の出力サイズ（[`fandhe_ai_tensor_core::
/// interpolate_out_shape`] と同じ「末尾 `size.len()` 軸」規約。
/// `Var::interpolate` が事前に検査・確定済み）。出力 shape は
/// `input.shape()` の先頭軸をそのまま・末尾 `size.len()` 軸を `size`
/// で置き換えた形。
///
/// **算術を含まない純粋なコピー演算**のため、forward は 3 バックエンド
/// 間で構造的に **bit 完全一致**する（[`fandhe_ai_tensor_core::InterpolateMode::Nearest`]
/// doc 参照）。`input` は strided view でよい（`dense_vec_ref` で行
/// 優先稠密化してから読む）。座標ごとの src 添字導出は
/// [`nearest_src_coord`]（forward／backward の単一情報源）を使う。
pub(crate) fn interpolate_nearest(
    input: &Tensor<f32>,
    size: &[usize],
) -> Result<Tensor<f32>, ShapeError> {
    let in_shape = input.shape().to_vec();
    let rank = in_shape.len();
    let spatial_start = rank - size.len();
    let mut out_shape = in_shape.clone();
    out_shape[spatial_start..].copy_from_slice(size);

    // `interpolate_out_shape`（呼び出し元 `Var::interpolate_impl` が
    // 事前検査済み）は出力 shape のバイトサイズしか検査しないため、
    // 入力側（`dense_vec_ref` が非 contiguous な `input` を稠密化する
    // 際に `Vec::with_capacity` 相当を呼ぶ）を別途検査する必要がある。
    // 巨大な `broadcast_to` view（例: `[1]` を `[1usize << 63]` へ
    // 拡張した view）を小さい `size` へ縮小する interpolate では、
    // 出力は小さくても入力の稠密化が `f32` 換算で `isize::MAX` バイト
    // を超え capacity overflow panic しうる（本番経路 panic 禁止規約
    // `.claude/rules/coding-rust.md`。イシュー #1834 Cursor Bugbot
    // 指摘）。`Tensor::full`／`checked_numel_for::<f32>`
    // （`tensor-core` 側。`pub(crate)` のためここでは同型を独立複製）
    // と同じ検査を `dense_vec_ref` 呼び出し前に行う。
    let in_numel = in_shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)?;
    let in_bytes = in_numel
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(ShapeError::ElementCountOverflow)?;
    if in_bytes > isize::MAX as usize {
        return Err(ShapeError::ElementCountOverflow);
    }

    let numel: usize = out_shape.iter().product();
    if numel == 0 {
        return Ok(build_tensor(Vec::new(), &out_shape));
    }

    let input_data = dense_vec_ref(input);
    let input_strides = row_major_strides(&in_shape);

    let mut out = vec![0f32; numel];
    for (flat, out_val) in out.iter_mut().enumerate() {
        let coords = unravel(flat, &out_shape);
        let mut pos = 0usize;
        for (axis, &stride) in input_strides.iter().enumerate() {
            let coord = if axis >= spatial_start {
                nearest_src_coord(coords[axis], in_shape[axis], out_shape[axis])
            } else {
                coords[axis]
            };
            pos += coord * stride;
        }
        *out_val = input_data[pos];
    }
    Ok(build_tensor(out, &out_shape))
}

/// `interpolate`（[`fandhe_ai_tensor_core::InterpolateMode::
/// Bilinear`]）のホスト参照実装（イシュー #1762）。
/// `BackendOps::interpolate` が `Unsupported` を返したときのみ
/// `grad::interpolate_with_fallback` から呼ばれる。`interpolate_nearest`
/// と異なり算術（4 近傍の線形重み付け合成）を含むため、座標・重み・
/// ブレンド式はいずれも [`fandhe_ai_tensor_core::interpolate`]
/// （`bilinear_scale`／`bilinear_src_coord`／`bilinear_blend`）を単一
/// 情報源として使う（backward の VJP index 構築
/// `grad::bilinear_src_index_and_weight_map` も同じ関数を呼ぶ）。
///
/// `size` はちょうど 2 個の末尾空間軸サイズ（`(H, W)`。
/// `Var::interpolate` が `interpolate_out_shape_for_mode` で事前検査
/// 済み——本関数は `size.len() == 2` を前提とする。呼び出し元契約
/// 違反は `debug_assert!` でのみ検出する）。
pub(crate) fn interpolate_bilinear(
    input: &Tensor<f32>,
    size: &[usize],
    align_corners: bool,
) -> Result<Tensor<f32>, ShapeError> {
    debug_assert_eq!(
        size.len(),
        2,
        "interpolate_bilinear: caller must pre-validate size.len() == 2 via \
         interpolate_out_shape_for_mode"
    );
    let in_shape = input.shape().to_vec();
    let rank = in_shape.len();
    let spatial_start = rank - size.len();
    let mut out_shape = in_shape.clone();
    out_shape[spatial_start..].copy_from_slice(size);

    // `interpolate_nearest` と同じ理由: 出力側だけでなく入力側の
    // 稠密化コスト（`dense_vec_ref`）も別途検査する（巨大な
    // `broadcast_to` view からの縮小で入力の稠密化が capacity
    // overflow panic しうるため）。
    let in_numel = in_shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)?;
    let in_bytes = in_numel
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(ShapeError::ElementCountOverflow)?;
    if in_bytes > isize::MAX as usize {
        return Err(ShapeError::ElementCountOverflow);
    }

    let numel: usize = out_shape.iter().product();
    if numel == 0 {
        return Ok(build_tensor(Vec::new(), &out_shape));
    }

    let input_data = dense_vec_ref(input);
    let input_strides = row_major_strides(&in_shape);

    let h_axis = spatial_start;
    let w_axis = spatial_start + 1;
    let in_h = in_shape[h_axis];
    let in_w = in_shape[w_axis];
    let out_h = out_shape[h_axis];
    let out_w = out_shape[w_axis];
    let scale_h = bilinear_scale(in_h, out_h, align_corners);
    let scale_w = bilinear_scale(in_w, out_w, align_corners);

    let mut out = vec![0f32; numel];
    for (flat, out_val) in out.iter_mut().enumerate() {
        let coords = unravel(flat, &out_shape);
        let cy = bilinear_src_coord(coords[h_axis], in_h, scale_h, align_corners);
        let cx = bilinear_src_coord(coords[w_axis], in_w, scale_w, align_corners);

        // 空間軸以外（batch／channel 等）の素通し添字は共通の基底
        // オフセットへ畳み込んでおく（4 近傍の読み出しで毎回同じ計算
        // を繰り返さないため）。
        let mut base = 0usize;
        for (axis, &stride) in input_strides.iter().enumerate() {
            if axis != h_axis && axis != w_axis {
                base += coords[axis] * stride;
            }
        }
        let stride_h = input_strides[h_axis];
        let stride_w = input_strides[w_axis];
        let v00 = input_data[base + cy.i0 * stride_h + cx.i0 * stride_w];
        let v01 = input_data[base + cy.i0 * stride_h + cx.i1 * stride_w];
        let v10 = input_data[base + cy.i1 * stride_h + cx.i0 * stride_w];
        let v11 = input_data[base + cy.i1 * stride_h + cx.i1 * stride_w];
        *out_val = bilinear_blend(v00, v01, v10, v11, cx.lambda1, cy.lambda1);
    }
    Ok(build_tensor(out, &out_shape))
}

#[cfg(test)]
mod interpolate_nearest_host_fallback_tests {
    use super::*;

    /// Cursor Bugbot 指摘（イシュー #1834・PR レビュー）の回帰テスト。
    /// `[1]` を `broadcast_to([1usize << 63])` した巨大な非 contiguous
    /// view を小さい `size=[3]` へ縮小するホスト参照実装（`Var::
    /// interpolate` が `BackendOps::interpolate` の `Unsupported`
    /// フォールバックとして呼ぶ経路）は、出力側のバイトサイズは
    /// 小さいため呼び出し元 `interpolate_out_shape` の検査を通過する
    /// が、`dense_vec_ref`（内部で `Vec::with_capacity` 相当を呼ぶ）が
    /// 入力側の巨大な要素数で capacity overflow panic しうる（是正
    /// 前）。`dense_vec_ref` 呼び出し前の確保前検査（本関数冒頭）に
    /// より、パニックせず型付きエラーを返すことを確認する。
    #[test]
    fn rejects_huge_broadcast_view_input_without_panicking() {
        let base = Tensor::<f32>::new(vec![0.0f32], &[1usize]).unwrap();
        let huge = base.broadcast_to(&[1usize << 63]).unwrap();
        assert_eq!(huge.shape(), &[1usize << 63]);

        let err = interpolate_nearest(&huge, &[3])
            .expect_err("huge broadcast view の実体化は確保前に拒否されるはず");
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }
}

/// `i32` 版 `build_tensor`（上記）。sort／topk（下記）の `index` 出力
/// 構築に使う（`build_tensor` と同じ「shape 検査済みのはずのデータ長
/// 不一致は契約違反として `debug_assert!` で検知しつつ安全側
/// フォールバックする」infallible 契約）。`pub(crate)`: `grad.rs::
/// nearest_src_index_map`（イシュー #1757。`interpolate` VJP の
/// scatter_add index 構築）からも同じ契約で使う。
pub(crate) fn build_index_tensor(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    debug_assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "build_index_tensor: shape 検査済みのはずのデータ長が一致しない（契約違反）"
    );
    Tensor::new(data, shape).unwrap_or_else(|_| {
        debug_assert!(
            false,
            "build_index_tensor: shape の要素数積がオーバーフローした（契約違反）"
        );
        Tensor::scalar(0)
    })
}

/// `sort`／`topk`（下記）共通の全順序比較（`(値, 元添字)` タプル）。
/// [`fandhe_ai_tensor_core::BackendOps::sort`] doc の順序契約 1〜3 を
/// 実装する:
///
/// 1. **安定性**: 値side の比較（`value_cmp_ascending`）が同値
///    （`Equal`）を返した場合、添字（`.1`）の昇順で確定する
///    （`descending` の場合も同じ——後述のとおり値側の比較のみを
///    反転するため、添字側は常に昇順のまま）。
/// 2. **NaN**: NaN は任意の非 NaN より大きい・NaN 同士は同値
///    （`value_cmp_ascending`）。
/// 3. **±0**: `f32::partial_cmp` が `Equal` を返すため 1 の安定性
///    契約により同値扱い（添字順）となる。
///
/// `descending` は `value_cmp_ascending(b.0, a.0)`（オペランドを
/// 入れ替えて呼ぶ）で値側の大小関係のみを反転する。「昇順ソート
/// 結果を丸ごと `reverse()` する」実装は同値の添字順まで反転させて
/// しまうため意図的に避けている（`sort_by` はこの比較関数を直接
/// 使うため安定ソートの副作用に依存しない——`Vec::sort_by` 自体は
/// 安定ソートだが、本関数のタイブレークがどのような実装
/// （安定・非安定）のソートでも同じ結果になるよう明示的に定める）。
fn sort_cmp(a: (f32, usize), b: (f32, usize), descending: bool) -> std::cmp::Ordering {
    fn value_cmp_ascending(x: f32, y: f32) -> std::cmp::Ordering {
        match (x.is_nan(), y.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
        }
    }
    let primary = if descending {
        value_cmp_ascending(b.0, a.0)
    } else {
        value_cmp_ascending(a.0, b.0)
    };
    primary.then(a.1.cmp(&b.1))
}

/// `dim` 軸に沿った並べ替えのホスト参照実装（`torch.sort` 相当。
/// イシュー #1733）。`BackendOps::sort` が `Unsupported` を返した
/// ときのみ `Var::sort` から呼ばれる。
///
/// 出力 shape（`values`／`index` とも）は `input.shape()` と恒等
/// （[`fandhe_ai_tensor_core::sort_out_shape`] 参照）。順序契約は
/// [`fandhe_ai_tensor_core::BackendOps::sort`] doc の 1〜4 を正と
/// する（本関数は [`sort_cmp`] で契約 1〜3 を、単一スレッド逐次
/// 実装で契約 4〈決定性〉を満たす）。
///
/// 元添字（`orig_idx: usize`）は出力の `index` テンソルへ `i32` で
/// 書き戻すため、`i32::MAX` を超える添字は `i32::try_from` の失敗を
/// `ShapeError::IndexRangeOverflow` として伝播する（codex-review
/// 指摘・PR #1818。`crates/backend-cpu/src/sort_topk.rs::sort` が
/// 同条件で返す契約と一致させ、CUDA／Metal がこのホスト
/// フォールバックを経由してもバックエンド間の添字契約を崩さない）。
pub(crate) fn sort(
    input: &Tensor<f32>,
    dim: usize,
    descending: bool,
) -> Result<(Tensor<f32>, Tensor<i32>), ShapeError> {
    let shape = input.shape().to_vec();
    if shape.contains(&0) {
        return Ok((
            build_tensor(Vec::new(), &shape),
            build_index_tensor(Vec::new(), &shape),
        ));
    }
    let data = dense_vec_ref(input);
    let strides = row_major_strides(&shape);
    let dim_size = shape[dim];
    let numel = data.len();
    let mut out_vals = vec![0f32; numel];
    let mut out_idx = vec![0i32; numel];

    for flat in 0..numel {
        let coords = unravel(flat, &shape);
        // 各ライン（`dim` 軸以外の添字が共通の要素列）は、その先頭
        // （`dim` 軸添字 0）の位置に到達したときだけ 1 回処理する。
        if coords[dim] != 0 {
            continue;
        }
        let mut line: Vec<(f32, usize)> = Vec::with_capacity(dim_size);
        for idx in 0..dim_size {
            let mut pos = 0usize;
            for (axis, &stride) in strides.iter().enumerate() {
                let coord = if axis == dim { idx } else { coords[axis] };
                pos += coord * stride;
            }
            line.push((data[pos], idx));
        }
        line.sort_by(|&a, &b| sort_cmp(a, b, descending));
        for (out_pos, &(val, orig_idx)) in line.iter().enumerate() {
            let mut pos = 0usize;
            for (axis, &stride) in strides.iter().enumerate() {
                let coord = if axis == dim { out_pos } else { coords[axis] };
                pos += coord * stride;
            }
            out_vals[pos] = val;
            out_idx[pos] = i32::try_from(orig_idx)
                .map_err(|_| ShapeError::IndexRangeOverflow { index: orig_idx })?;
        }
    }
    Ok((
        build_tensor(out_vals, &shape),
        build_index_tensor(out_idx, &shape),
    ))
}

/// `dim` 軸に沿った上位（`largest=true`）／下位（`largest=false`）
/// `k` 個抽出のホスト参照実装（`torch.topk` 相当・`sorted=True`
/// 固定。イシュー #1733）。`BackendOps::topk` が `Unsupported` を
/// 返したときのみ `Var::topk` から呼ばれる。`largest` は [`sort`] の
/// `descending` へそのまま対応する（`largest=true` → 降順 sort の
/// 先頭 `k`）ため、[`sort_cmp`] を共有する。
///
/// `out_shape` は呼び出し元（`Var::topk`）が
/// [`fandhe_ai_tensor_core::topk_out_shape`] で検査・確定済みの
/// 出力 shape（`dim` 軸のみ `k` に置換）をそのまま渡す。
///
/// [`sort`] と同じ理由で、元添字が `i32::MAX` を超える場合は
/// `ShapeError::IndexRangeOverflow` を返す（codex-review 指摘・
/// PR #1818。`crates/backend-cpu/src/sort_topk.rs::topk` と契約を
/// 一致させる）。
pub(crate) fn topk(
    input: &Tensor<f32>,
    dim: usize,
    k: usize,
    largest: bool,
    out_shape: &[usize],
) -> Result<(Tensor<f32>, Tensor<i32>), ShapeError> {
    if out_shape.contains(&0) {
        return Ok((
            build_tensor(Vec::new(), out_shape),
            build_index_tensor(Vec::new(), out_shape),
        ));
    }
    let shape = input.shape().to_vec();
    let data = dense_vec_ref(input);
    let strides = row_major_strides(&shape);
    let out_strides = row_major_strides(out_shape);
    let dim_size = shape[dim];
    let numel = data.len();
    let out_numel: usize = out_shape.iter().product();
    let mut out_vals = vec![0f32; out_numel];
    let mut out_idx = vec![0i32; out_numel];

    for flat in 0..numel {
        let coords = unravel(flat, &shape);
        if coords[dim] != 0 {
            continue;
        }
        let mut line: Vec<(f32, usize)> = Vec::with_capacity(dim_size);
        for idx in 0..dim_size {
            let mut pos = 0usize;
            for (axis, &stride) in strides.iter().enumerate() {
                let coord = if axis == dim { idx } else { coords[axis] };
                pos += coord * stride;
            }
            line.push((data[pos], idx));
        }
        line.sort_by(|&a, &b| sort_cmp(a, b, largest));
        for (out_pos, &(val, orig_idx)) in line.iter().take(k).enumerate() {
            let mut pos = 0usize;
            for (axis, &stride) in out_strides.iter().enumerate() {
                let coord = if axis == dim { out_pos } else { coords[axis] };
                pos += coord * stride;
            }
            out_vals[pos] = val;
            out_idx[pos] = i32::try_from(orig_idx)
                .map_err(|_| ShapeError::IndexRangeOverflow { index: orig_idx })?;
        }
    }
    Ok((
        build_tensor(out_vals, out_shape),
        build_index_tensor(out_idx, out_shape),
    ))
}

/// `Var::one_hot` のホスト参照実装（`torch.nn.functional.one_hot`／
/// `tf.one_hot` 相当。イシュー #1755）。`BackendOps::one_hot` が
/// `Unsupported` を返したときのみ `grad::one_hot_with_fallback` から
/// 呼ばれる。`index`（値は `[0, num_classes)` の範囲内であることを
/// `Var::one_hot` が forward 時点で検査済み）の各要素 `c` に対し、
/// 出力の末尾軸（サイズ `num_classes`）へ `c` 番目だけ `1.0`・残りを
/// `0.0` とする one-hot 行を書く。出力 shape は `index.shape() ++
/// [num_classes]`（[`fandhe_ai_tensor_core::one_hot_out_shape`]
/// 参照）で、各出力位置は互いに独立（データ競合の心配がない単純な
/// 走査で bit 決定的）。
pub(crate) fn one_hot(index: &Tensor<i32>, num_classes: usize, out_shape: &[usize]) -> Tensor<f32> {
    if out_shape.contains(&0) {
        return build_tensor(Vec::new(), out_shape);
    }
    let index_data = dense_vec_i32(index);
    let numel: usize = out_shape.iter().product();
    let mut out = vec![0f32; numel];
    for (flat, out_val) in out.iter_mut().enumerate() {
        let row = flat / num_classes;
        let c = flat % num_classes;
        let dim_idx = index_data[row];
        if dim_idx >= 0 && (dim_idx as usize) == c {
            *out_val = 1.0;
        }
    }
    build_tensor(out, out_shape)
}

/// 窓添字から入力座標を符号安全に逆算する（`backend-cpu::pooling::
/// window_input_pos`／`grad.rs::pool_window_input_pos` と同型の式。
/// `eval` と CPU 実装の意図的複製方針〈`im2col`／`gather`／`scatter`
/// の先例〉に倣いクレートをまたいで複製する。イシュー #1728）。
fn pool_window_input_pos(
    out_idx: usize,
    stride: usize,
    k: usize,
    dilation: usize,
    padding: usize,
) -> Option<usize> {
    let base = out_idx.checked_mul(stride)?;
    let offset = k.checked_mul(dilation)?;
    let sum = base.checked_add(offset)?;
    sum.checked_sub(padding)
}

/// `Var::max_pool2d` のホスト参照実装（`torch.nn.functional.
/// max_pool2d` 相当。イシュー #1728・設計 `docs/pooling-ops-
/// design.md` §5）。`BackendOps::max_pool2d` が `Unsupported` を
/// 返したときのみ `grad::max_pool2d_with_fallback` から呼ばれる。
/// `backend-cpu::pooling::max_pool2d` と意図的に同一アルゴリズム
/// （先勝ちタイ規則・NaN 伝播・padding 走査除外）を複製する。
pub(crate) fn max_pool2d(
    input: &Tensor<f32>,
    params: &Pool2dParams,
    out_shape: &[usize],
) -> Result<(Tensor<f32>, Tensor<i32>), ShapeError> {
    let out_numel: usize = out_shape.iter().product();
    if out_numel == 0 {
        return Ok((
            build_tensor(Vec::new(), out_shape),
            build_index_tensor(Vec::new(), out_shape),
        ));
    }
    let in_shape = input.shape();
    let (h_in, w_in) = (in_shape[2], in_shape[3]);
    let (n_batch, c_ch, h_out, w_out) = (out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    let [kh, kw] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();

    let mut out_vals = vec![0f32; out_numel];
    let mut out_idx = vec![0i32; out_numel];
    for n in 0..n_batch {
        for c in 0..c_ch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut best: Option<(f32, usize)> = None;
                    for kh_ in 0..kh {
                        let Some(h) =
                            pool_window_input_pos(oh, sh, kh_, dh, ph).filter(|&h| h < h_in)
                        else {
                            continue;
                        };
                        for kw_ in 0..kw {
                            let Some(w) =
                                pool_window_input_pos(ow, sw, kw_, dw, pw).filter(|&w| w < w_in)
                            else {
                                continue;
                            };
                            let v = input.get(&[n, c, h, w]).ok_or_else(|| {
                                ShapeError::ShapeMismatch {
                                    lhs: vec![n, c, h, w],
                                    rhs: in_shape.to_vec(),
                                }
                            })?;
                            let flat = h * w_in + w;
                            best = Some(match best {
                                None => (v, flat),
                                Some((b, bi)) => {
                                    if v > b || (v.is_nan() && !b.is_nan()) {
                                        (v, flat)
                                    } else {
                                        (b, bi)
                                    }
                                }
                            });
                        }
                    }
                    let (v, idx) = best.ok_or(ShapeError::ElementCountOverflow)?;
                    let out_pos = ((n * c_ch + c) * h_out + oh) * w_out + ow;
                    out_vals[out_pos] = v;
                    out_idx[out_pos] = i32::try_from(idx)
                        .map_err(|_| ShapeError::IndexRangeOverflow { index: idx })?;
                }
            }
        }
    }
    Ok((
        build_tensor(out_vals, out_shape),
        build_index_tensor(out_idx, out_shape),
    ))
}

/// `Var::avg_pool2d` のホスト参照実装（イシュー #1728・設計 `docs/
/// pooling-ops-design.md` §7）。`BackendOps::avg_pool2d` が
/// `Unsupported` を返したときのみ `grad::avg_pool2d_with_fallback`
/// から呼ばれる。窓内を row-major で `f64` へ逐次加算し最後に 1 回
/// `f32` へ downcast する（`backend-cpu::pooling::avg_pool2d` と
/// 意図的に同一アルゴリズムを複製）。
pub(crate) fn avg_pool2d(
    input: &Tensor<f32>,
    params: &Pool2dParams,
    count_include_pad: bool,
    out_shape: &[usize],
) -> Result<Tensor<f32>, ShapeError> {
    let out_numel: usize = out_shape.iter().product();
    if out_numel == 0 {
        return Ok(build_tensor(Vec::new(), out_shape));
    }
    let in_shape = input.shape();
    let (h_in, w_in) = (in_shape[2], in_shape[3]);
    let (n_batch, c_ch, h_out, w_out) = (out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    let [kh, kw] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();

    let mut out = vec![0f32; out_numel];
    for n in 0..n_batch {
        for c in 0..c_ch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc: f64 = 0.0;
                    let mut count: usize = 0;
                    for kh_ in 0..kh {
                        let Some(h) =
                            pool_window_input_pos(oh, sh, kh_, dh, ph).filter(|&h| h < h_in)
                        else {
                            continue;
                        };
                        for kw_ in 0..kw {
                            let Some(w) =
                                pool_window_input_pos(ow, sw, kw_, dw, pw).filter(|&w| w < w_in)
                            else {
                                continue;
                            };
                            let v = input.get(&[n, c, h, w]).ok_or_else(|| {
                                ShapeError::ShapeMismatch {
                                    lhs: vec![n, c, h, w],
                                    rhs: in_shape.to_vec(),
                                }
                            })?;
                            acc += f64::from(v);
                            count += 1;
                        }
                    }
                    let divisor = if count_include_pad {
                        kh.checked_mul(kw).ok_or(ShapeError::ElementCountOverflow)?
                    } else {
                        count
                    };
                    if divisor == 0 {
                        return Err(ShapeError::ElementCountOverflow);
                    }
                    let v = (acc / divisor as f64) as f32;
                    let out_pos = ((n * c_ch + c) * h_out + oh) * w_out + ow;
                    out[out_pos] = v;
                }
            }
        }
    }
    Ok(build_tensor(out, out_shape))
}

/// `Var::adaptive_avg_pool2d` のホスト参照実装（イシュー #1728・
/// 設計 `docs/pooling-ops-design.md` §7）。`BackendOps::
/// adaptive_avg_pool2d` が `Unsupported` を返したときのみ
/// `grad::adaptive_avg_pool2d_with_fallback` から呼ばれる。窓は
/// [`adaptive_window`]（forward／VJP 共有の単一情報源）が定める。
/// 縮約の数値契約は [`avg_pool2d`] と同一（`backend-cpu::pooling::
/// adaptive_avg_pool2d` と意図的に同一アルゴリズムを複製）。
pub(crate) fn adaptive_avg_pool2d(
    input: &Tensor<f32>,
    out_shape: &[usize],
) -> Result<Tensor<f32>, ShapeError> {
    let out_numel: usize = out_shape.iter().product();
    if out_numel == 0 {
        return Ok(build_tensor(Vec::new(), out_shape));
    }
    let in_shape = input.shape();
    let (h_in, w_in) = (in_shape[2], in_shape[3]);
    let (n_batch, c_ch, h_out, w_out) = (out_shape[0], out_shape[1], out_shape[2], out_shape[3]);

    let mut out = vec![0f32; out_numel];
    for n in 0..n_batch {
        for c in 0..c_ch {
            for oh in 0..h_out {
                let (h_start, h_end) =
                    adaptive_window(oh, h_in, h_out).ok_or(ShapeError::ElementCountOverflow)?;
                for ow in 0..w_out {
                    let (w_start, w_end) =
                        adaptive_window(ow, w_in, w_out).ok_or(ShapeError::ElementCountOverflow)?;
                    let mut acc: f64 = 0.0;
                    let mut count: usize = 0;
                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            let v = input.get(&[n, c, h, w]).ok_or_else(|| {
                                ShapeError::ShapeMismatch {
                                    lhs: vec![n, c, h, w],
                                    rhs: in_shape.to_vec(),
                                }
                            })?;
                            acc += f64::from(v);
                            count += 1;
                        }
                    }
                    if count == 0 {
                        return Err(ShapeError::ElementCountOverflow);
                    }
                    let v = (acc / count as f64) as f32;
                    let out_pos = ((n * c_ch + c) * h_out + oh) * w_out + ow;
                    out[out_pos] = v;
                }
            }
        }
    }
    Ok(build_tensor(out, out_shape))
}

/// 平坦化・totalOrder ソート・隣接重複除去のホスト参照実装
/// （`torch.unique(input, sorted=True)` の values のみ。イシュー
/// #1734）。`BackendOps::unique` が `Unsupported` を返したときのみ
/// `grad::unique_with_fallback` から呼ばれる。`backend-cpu::unique`
/// と意図的に同一アルゴリズムを複製する（`eval` と CPU 実装の
/// 意図的複製方針。gather／scatter の先例に倣う）。
///
/// **順序キー**: `f32::total_cmp`（IEEE 754 totalOrder）。**重複判定**:
/// `==`（IEEE 比較。`-0.0`／`+0.0` は同一視され totalOrder で先頭側の
/// `-0.0` が代表として残り、`NaN` は `NaN != NaN` のためすべて保持
/// される）。`sort_unstable_by` は totalOrder で `Equal` と判定される
/// 要素同士が bit 単位で同一であることを前提に安定性を要求しない
/// （totalOrder は全順序でありタイは bit 同一の場合のみ発生する）。
///
/// **契約（PR #1828 codex-review P1 是正）**: 呼び出し元
/// `grad::unique_with_fallback` は本関数を呼ぶ前に要素数積のオーバー
/// フロー検査を済ませているため、本関数が実際にオーバーフロー形状で
/// 呼ばれることは契約上ない（`build_tensor` と同じ「呼び出し元が
/// 事前検査済み」契約・`docs/fusion-graph-design.md` §3.5.3 (iii)）。
/// それでも `input.numel()`（内部で無検査の `.iter().product()` を
/// 使い `overflow-checks` 有効ビルドで panic しうる）を直接呼ばず、
/// 事前に `checked_mul` で再検査してから読む（契約違反を `panic!` で
/// はなく `debug_assert!` で検知しつつ空集合へ安全側フォールバックする
/// 二重防御。本番経路 panic 禁止・`.claude/rules/coding-rust.md`）。
pub(crate) fn unique(input: &Tensor<f32>) -> Tensor<f32> {
    let checked_numel = input
        .shape()
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d));
    if checked_numel.is_none() {
        debug_assert!(
            false,
            "eval::unique: 呼び出し元が要素数積オーバーフローを事前検査済みのはずが違反した（契約違反）"
        );
        return build_tensor(Vec::new(), &[0]);
    }
    if input.numel() == 0 {
        return build_tensor(Vec::new(), &[0]);
    }
    let mut v = dense_vec(input);
    v.sort_unstable_by(f32::total_cmp);
    v.dedup_by(|cur, prev| *cur == *prev);
    let m = v.len();
    build_tensor(v, &[m])
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

// =====================================================================
// RNN／LSTM／GRU セル演算のホスト参照実装（イシュー #1647・設計
// `docs/autodiff-rnn-cell-tape-design.md` 決定 1・1b・1c・5・12）。
//
// `BackendOps::{lstm_pointwise,lstm_hidden_backward,lstm_cell_backward,
// gru_pointwise,gru_backward}`（`tensor-core::backend_ops`）が
// [`fandhe_ai_tensor_core::BackendError::Unsupported`] を返した場合の
// フォールバック（`var.rs::Var::{lstm_cell,gru_cell}` から呼ばれる。A08:
// `Unsupported` 以外のエラーは伝播し暗黙にはここへ来ない）。数式の正は
// 本モジュールであり、CPU／CUDA／Metal の各カーネル実装は同じ数式を
// バックエンド固有の並列化・FMA 契約で再実装する（`.claude/rules/
// coding-rust.md`）。
//
// ゲート配置（決定 5・PyTorch 準拠）: LSTM は列ブロック順 `i,f,g,o`
// （`pre: [B, 4H]`）、GRU は `r,z,n`（`pre_i`／`pre_h`: `[B, 3H]`）。
// =====================================================================

/// `pre: [B, G*H]` から `(b, gate, col)` の要素を読む（行優先連続データ
/// 前提。呼び出し元が `dense_vec` 済みのスライスを渡す）。
fn gate_elem(data: &[f32], gates: usize, hidden: usize, b: usize, gate: usize, j: usize) -> f32 {
    data[b * (gates * hidden) + gate * hidden + j]
}

/// LSTM セルの pointwise 段（決定 1・1b）参照実装。`pre: [B, 4H]`
/// （列ブロック順 `i,f,g,o`）・`c_prev: [B, H]` から `gates`（活性化後
/// `i,f,g,o`。`[B, 4H]`）・`c`（新セル状態。`[B, H]`）・`h`（新隠れ状態。
/// `[B, H]`）を計算する。`H` は `c_prev` の列数から導出する。
///
/// `c = f·c_prev + i·g`（`f32::mul_add` で FMA 契約統一）、
/// `h = o·tanh(c)`。
pub(crate) fn lstm_pointwise(pre: &Tensor<f32>, c_prev: &Tensor<f32>) -> LstmPointwiseOutput {
    let b_dim = c_prev.shape().first().copied().unwrap_or(0);
    let hidden = c_prev.shape().get(1).copied().unwrap_or(0);
    let pre_data = dense_vec(pre);
    let c_prev_data = dense_vec(c_prev);

    let mut gates = vec![0f32; b_dim * 4 * hidden];
    let mut c_out = vec![0f32; b_dim * hidden];
    let mut h_out = vec![0f32; b_dim * hidden];

    for b in 0..b_dim {
        for j in 0..hidden {
            let i_pre = gate_elem(&pre_data, 4, hidden, b, 0, j);
            let f_pre = gate_elem(&pre_data, 4, hidden, b, 1, j);
            let g_pre = gate_elem(&pre_data, 4, hidden, b, 2, j);
            let o_pre = gate_elem(&pre_data, 4, hidden, b, 3, j);

            let i_val = sigmoid_scalar(i_pre);
            let f_val = sigmoid_scalar(f_pre);
            let g_val = g_pre.tanh();
            let o_val = sigmoid_scalar(o_pre);

            let base = b * 4 * hidden;
            gates[base + j] = i_val;
            gates[base + hidden + j] = f_val;
            gates[base + 2 * hidden + j] = g_val;
            gates[base + 3 * hidden + j] = o_val;

            let c_prev_val = c_prev_data[b * hidden + j];
            let c_val = f_val.mul_add(c_prev_val, i_val * g_val);
            let h_val = o_val * c_val.tanh();
            c_out[b * hidden + j] = c_val;
            h_out[b * hidden + j] = h_val;
        }
    }

    LstmPointwiseOutput {
        gates: build_tensor(gates, &[b_dim, 4 * hidden]),
        c: build_tensor(c_out, &[b_dim, hidden]),
        h: build_tensor(h_out, &[b_dim, hidden]),
    }
}

/// [`Op::LstmHidden`] の VJP 補助（決定 1b・1b 追記）参照実装。
/// `d_pre_o = dh·tanh(c)·o·(1−o)`、`dc = dh·o·(1−tanh(c)²)`。
pub(crate) fn lstm_hidden_backward(
    c: &Tensor<f32>,
    gate_o: &Tensor<f32>,
    dh: &Tensor<f32>,
) -> (Tensor<f32>, Tensor<f32>) {
    let shape = c.shape().to_vec();
    let c_data = dense_vec(c);
    let o_data = dense_vec(gate_o);
    let dh_data = dense_vec(dh);

    let mut d_pre_o = vec![0f32; c_data.len()];
    let mut dc = vec![0f32; c_data.len()];
    for idx in 0..c_data.len() {
        let tanh_c = c_data[idx].tanh();
        let o_val = o_data[idx];
        let dh_val = dh_data[idx];
        d_pre_o[idx] = dh_val * tanh_c * o_val * (1.0 - o_val);
        dc[idx] = dh_val * o_val * (1.0 - tanh_c * tanh_c);
    }
    (build_tensor(d_pre_o, &shape), build_tensor(dc, &shape))
}

/// [`Op::LstmCell`] の VJP 補助（決定 1b）参照実装。`gates_ifg: [B, 3H]`
/// （活性化後の `i,f,g`）・`c_prev: [B, H]`・`dc: [B, H]` から
/// `d_pre_ifg: [B, 3H]`・`dc_prev: [B, H]` を計算する。
///
/// `d_pre_i = dc·g·i·(1−i)`、`d_pre_f = dc·c_prev·f·(1−f)`、
/// `d_pre_g = dc·i·(1−g²)`、`dc_prev = dc·f`。
pub(crate) fn lstm_cell_backward(
    gates_ifg: &Tensor<f32>,
    c_prev: &Tensor<f32>,
    dc: &Tensor<f32>,
) -> (Tensor<f32>, Tensor<f32>) {
    let b_dim = c_prev.shape().first().copied().unwrap_or(0);
    let hidden = c_prev.shape().get(1).copied().unwrap_or(0);
    let gates_data = dense_vec(gates_ifg);
    let c_prev_data = dense_vec(c_prev);
    let dc_data = dense_vec(dc);

    let mut d_pre_ifg = vec![0f32; b_dim * 3 * hidden];
    let mut dc_prev = vec![0f32; b_dim * hidden];
    for b in 0..b_dim {
        for j in 0..hidden {
            let i_val = gate_elem(&gates_data, 3, hidden, b, 0, j);
            let f_val = gate_elem(&gates_data, 3, hidden, b, 1, j);
            let g_val = gate_elem(&gates_data, 3, hidden, b, 2, j);
            let c_prev_val = c_prev_data[b * hidden + j];
            let dc_val = dc_data[b * hidden + j];

            let base = b * 3 * hidden;
            d_pre_ifg[base + j] = dc_val * g_val * i_val * (1.0 - i_val);
            d_pre_ifg[base + hidden + j] = dc_val * c_prev_val * f_val * (1.0 - f_val);
            d_pre_ifg[base + 2 * hidden + j] = dc_val * i_val * (1.0 - g_val * g_val);
            dc_prev[b * hidden + j] = dc_val * f_val;
        }
    }

    (
        build_tensor(d_pre_ifg, &[b_dim, 3 * hidden]),
        build_tensor(dc_prev, &[b_dim, hidden]),
    )
}

/// GRU セルの pointwise 段（決定 1c・5。`reset_after=True` 規約）参照
/// 実装。`pre_i`／`pre_h: [B, 3H]`（列ブロック順 `r,z,n`）・
/// `h_prev: [B, H]` から `gates`（活性化後 `r,z,n`。`[B, 3H]`）・`q`
/// （再帰側アフィン値 `pre_h` の n 列ブロック。`[B, H]`）・`h`（新隠れ
/// 状態。`[B, H]`）を計算する。
///
/// `r = σ(pre_i_r + pre_h_r)`、`z = σ(pre_i_z + pre_h_z)`、
/// `q = pre_h_n`、`n = tanh(r·q + pre_i_n)`、
/// `h = z·h_prev + (1−z)·n`。
pub(crate) fn gru_pointwise(
    pre_i: &Tensor<f32>,
    pre_h: &Tensor<f32>,
    h_prev: &Tensor<f32>,
) -> GruPointwiseOutput {
    let b_dim = h_prev.shape().first().copied().unwrap_or(0);
    let hidden = h_prev.shape().get(1).copied().unwrap_or(0);
    let pre_i_data = dense_vec(pre_i);
    let pre_h_data = dense_vec(pre_h);
    let h_prev_data = dense_vec(h_prev);

    let mut gates = vec![0f32; b_dim * 3 * hidden];
    let mut q_out = vec![0f32; b_dim * hidden];
    let mut h_out = vec![0f32; b_dim * hidden];

    for b in 0..b_dim {
        for j in 0..hidden {
            let r_pre = gate_elem(&pre_i_data, 3, hidden, b, 0, j)
                + gate_elem(&pre_h_data, 3, hidden, b, 0, j);
            let z_pre = gate_elem(&pre_i_data, 3, hidden, b, 1, j)
                + gate_elem(&pre_h_data, 3, hidden, b, 1, j);
            let q_val = gate_elem(&pre_h_data, 3, hidden, b, 2, j);
            let pre_i_n = gate_elem(&pre_i_data, 3, hidden, b, 2, j);

            let r_val = sigmoid_scalar(r_pre);
            let z_val = sigmoid_scalar(z_pre);
            let n_val = r_val.mul_add(q_val, pre_i_n).tanh();

            let base = b * 3 * hidden;
            gates[base + j] = r_val;
            gates[base + hidden + j] = z_val;
            gates[base + 2 * hidden + j] = n_val;
            q_out[b * hidden + j] = q_val;

            let h_prev_val = h_prev_data[b * hidden + j];
            h_out[b * hidden + j] = z_val.mul_add(h_prev_val, (1.0 - z_val) * n_val);
        }
    }

    GruPointwiseOutput {
        gates: build_tensor(gates, &[b_dim, 3 * hidden]),
        q: build_tensor(q_out, &[b_dim, hidden]),
        h: build_tensor(h_out, &[b_dim, hidden]),
    }
}

/// [`Op::GruCell`] の VJP 補助参照実装。`gates_rzn: [B, 3H]`（活性化後の
/// `r,z,n`）・`q: [B, H]`（決定 1c）・`h_prev: [B, H]`・`dh: [B, H]` から
/// `d_pre_i: [B, 3H]`・`d_pre_h: [B, 3H]`・`dh_prev_direct: [B, H]` を
/// 計算する。
///
/// `dn = dh·(1−z)`、`dz = dh·(h_prev−n)`、`dh_prev_direct = dh·z`、
/// `d_pre_n = dn·(1−n²)`、`dr = d_pre_n·q`、`d_pre_r = dr·r·(1−r)`、
/// `d_pre_z = dz·z·(1−z)`。`d_pre_i = [d_pre_r, d_pre_z, d_pre_n]`、
/// `d_pre_h = [d_pre_r, d_pre_z, d_pre_n·r]`（`n` の `q` に対する偏微分が
/// `r` であるため、`n` 列ブロックのみ追加で `r` を乗じる）。
pub(crate) fn gru_backward(
    gates_rzn: &Tensor<f32>,
    q: &Tensor<f32>,
    h_prev: &Tensor<f32>,
    dh: &Tensor<f32>,
) -> GruBackwardOutput {
    let b_dim = h_prev.shape().first().copied().unwrap_or(0);
    let hidden = h_prev.shape().get(1).copied().unwrap_or(0);
    let gates_data = dense_vec(gates_rzn);
    let q_data = dense_vec(q);
    let h_prev_data = dense_vec(h_prev);
    let dh_data = dense_vec(dh);

    let mut d_pre_i = vec![0f32; b_dim * 3 * hidden];
    let mut d_pre_h = vec![0f32; b_dim * 3 * hidden];
    let mut dh_prev_direct = vec![0f32; b_dim * hidden];

    for b in 0..b_dim {
        for j in 0..hidden {
            let r_val = gate_elem(&gates_data, 3, hidden, b, 0, j);
            let z_val = gate_elem(&gates_data, 3, hidden, b, 1, j);
            let n_val = gate_elem(&gates_data, 3, hidden, b, 2, j);
            let q_val = q_data[b * hidden + j];
            let h_prev_val = h_prev_data[b * hidden + j];
            let dh_val = dh_data[b * hidden + j];

            let dn = dh_val * (1.0 - z_val);
            let dz = dh_val * (h_prev_val - n_val);
            let d_pre_n = dn * (1.0 - n_val * n_val);
            let dr = d_pre_n * q_val;
            let d_pre_r = dr * r_val * (1.0 - r_val);
            let d_pre_z = dz * z_val * (1.0 - z_val);

            let base = b * 3 * hidden;
            d_pre_i[base + j] = d_pre_r;
            d_pre_i[base + hidden + j] = d_pre_z;
            d_pre_i[base + 2 * hidden + j] = d_pre_n;

            d_pre_h[base + j] = d_pre_r;
            d_pre_h[base + hidden + j] = d_pre_z;
            d_pre_h[base + 2 * hidden + j] = d_pre_n * r_val;

            dh_prev_direct[b * hidden + j] = dh_val * z_val;
        }
    }

    (
        build_tensor(d_pre_i, &[b_dim, 3 * hidden]),
        build_tensor(d_pre_h, &[b_dim, 3 * hidden]),
        build_tensor(dh_prev_direct, &[b_dim, hidden]),
    )
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

#[cfg(test)]
mod norm_rows_tests {
    //! [`rmsnorm_rows`]／[`layer_norm_rows`]（イシュー #1596）のホスト
    //! 参照実装単体テスト。`backend-cpu` の実機カーネルとの parity は
    //! `crates/backend-cpu/tests/{rmsnorm,layer_norm}_parity.rs` が担う
    //! （本モジュールは `eval.rs` 自体の正しさのみを検証する）。

    use super::*;

    #[test]
    fn rmsnorm_rows_basic_no_weight() {
        // hidden=4, x=[1,2,3,4] -> mean(x^2)=(1+4+9+16)/4=7.5
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let out = rmsnorm_rows(&x, None, 0.0, 1, 4);
        let rstd = 1.0f32 / 7.5f32.sqrt();
        for (o, v) in dense_vec(&out).iter().zip([1.0f32, 2.0, 3.0, 4.0].iter()) {
            assert!((o - v * rstd).abs() < 1e-5);
        }
    }

    #[test]
    fn rmsnorm_rows_applies_weight() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let w = [2.0f32, 1.0, 0.5, 1.0];
        let out = dense_vec(&rmsnorm_rows(&x, Some(&w), 0.0, 1, 4));
        let rstd = 1.0f32 / 7.5f32.sqrt();
        let expected = [1.0 * rstd * 2.0, 2.0 * rstd, 3.0 * rstd * 0.5, 4.0 * rstd];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5);
        }
    }

    #[test]
    fn rmsnorm_rows_empty_rows_or_hidden_is_empty_output() {
        let x0 = Tensor::new(Vec::<f32>::new(), &[0, 4]).unwrap();
        assert_eq!(
            dense_vec(&rmsnorm_rows(&x0, None, 1e-5, 0, 4)),
            Vec::<f32>::new()
        );
        let x1 = Tensor::new(Vec::<f32>::new(), &[3, 0]).unwrap();
        assert_eq!(
            dense_vec(&rmsnorm_rows(&x1, None, 1e-5, 3, 0)),
            Vec::<f32>::new()
        );
    }

    #[test]
    fn rmsnorm_rows_nan_propagates() {
        let x = Tensor::new(vec![f32::NAN, 1.0, 1.0, 1.0], &[1, 4]).unwrap();
        let out = dense_vec(&rmsnorm_rows(&x, None, 1e-5, 1, 4));
        assert!(out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn layer_norm_rows_matches_manual_computation() {
        // x = [1, 2, 3, 4] -> mean=2.5, var=Sigma(x-2.5)^2/4 = (2.25+0.25+0.25+2.25)/4 = 1.25
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let out = dense_vec(&layer_norm_rows(&x, None, None, 0.0, 1, 4));
        let rstd = 1.0f32 / 1.25f32.sqrt();
        let expected = [-1.5 * rstd, -0.5 * rstd, 0.5 * rstd, 1.5 * rstd];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "o={o} e={e}");
        }
    }

    /// PR #1671 codex-review P1 指摘（イシュー #1596）の反例を
    /// ホスト参照実装側で再現する回帰テスト。詳細は
    /// `backend-cpu::layer_norm::tests::
    /// run_layer_norm_f32_matches_gpu_butterfly_order_on_cancelling_row`
    /// の doc comment を参照（両実装は同じ `warp_reduce_f64` 順序を
    /// 使うため同じ期待値になる）。
    #[test]
    fn layer_norm_rows_matches_gpu_butterfly_order_on_cancelling_row() {
        let x = Tensor::new(vec![1e30f32, 1.0, -1e30, 0.0], &[1, 4]).unwrap();
        let w = [1.0f32, 1.0, 1.0, 1e30];
        let out = dense_vec(&layer_norm_rows(&x, Some(&w), None, 1e-5, 1, 4));

        let expected_out3 = -std::f64::consts::SQRT_2 / 4.0;
        assert!(
            (out[3] as f64 - expected_out3).abs() < 1e-3,
            "out[3]={} expected~={expected_out3}",
            out[3]
        );
        assert_ne!(out[3], 0.0, "単純な逐次和への後退の可能性がある");
    }

    #[test]
    fn layer_norm_rows_applies_weight_and_bias() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let w = [2.0f32, 1.0, 1.0, 0.5];
        let b = [1.0f32, 0.0, -1.0, 2.0];
        let out = dense_vec(&layer_norm_rows(&x, Some(&w), Some(&b), 0.0, 1, 4));
        let rstd = 1.0f32 / 1.25f32.sqrt();
        let xhat = [-1.5 * rstd, -0.5 * rstd, 0.5 * rstd, 1.5 * rstd];
        let expected = [
            xhat[0] * 2.0 + 1.0,
            xhat[1] * 1.0 + 0.0,
            xhat[2] * 1.0 - 1.0,
            xhat[3] * 0.5 + 2.0,
        ];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "o={o} e={e}");
        }
    }

    #[test]
    fn layer_norm_rows_empty_rows_or_hidden_is_empty_output() {
        let x0 = Tensor::new(Vec::<f32>::new(), &[0, 4]).unwrap();
        assert_eq!(
            dense_vec(&layer_norm_rows(&x0, None, None, 1e-5, 0, 4)),
            Vec::<f32>::new()
        );
        let x1 = Tensor::new(Vec::<f32>::new(), &[3, 0]).unwrap();
        assert_eq!(
            dense_vec(&layer_norm_rows(&x1, None, None, 1e-5, 3, 0)),
            Vec::<f32>::new()
        );
    }

    #[test]
    fn layer_norm_rows_nan_propagates() {
        let x = Tensor::new(vec![f32::NAN, 1.0, 1.0, 1.0], &[1, 4]).unwrap();
        let out = dense_vec(&layer_norm_rows(&x, None, None, 1e-5, 1, 4));
        assert!(out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn layer_norm_rows_extreme_scale_does_not_overflow_stats() {
        // f64 promotion before squaring avoids overflow at this scale
        // (coding-rust.md normalization-stat accumulator contract).
        let x = Tensor::new(vec![2e20f32, -2e20, 2e20, -2e20], &[1, 4]).unwrap();
        let out = dense_vec(&layer_norm_rows(&x, None, None, 1e-5, 1, 4));
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
    }
}

#[cfg(test)]
mod batch_norm_channels_tests {
    use super::*;

    /// N=1,C=2,spatial=2（`[1,2,2]`。BatchNorm1d の空間入力形状）で
    /// チャネルごとの平均・分散を手計算値と突き合わせる。
    #[test]
    fn batch_norm_train_matches_manual_computation() {
        // ch0: [1,2] mean=1.5 var=0.25; ch1: [3,4] mean=3.5 var=0.25
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();
        let (out, mean, var) = batch_norm_train_channels(&x, None, None, 0.0, 1, 2, 2);
        assert!((mean[0] as f64 - 1.5).abs() < 1e-9);
        assert!((mean[1] as f64 - 3.5).abs() < 1e-9);
        assert!((var[0] as f64 - 0.25).abs() < 1e-9);
        assert!((var[1] as f64 - 0.25).abs() < 1e-9);
        let out = dense_vec(&out);
        let rstd = 1.0f32 / 0.25f32.sqrt();
        let expected = [
            (1.0 - 1.5) * rstd,
            (2.0 - 1.5) * rstd,
            (3.0 - 3.5) * rstd,
            (4.0 - 3.5) * rstd,
        ];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "o={o} e={e}");
        }
    }

    #[test]
    fn batch_norm_train_reduces_over_batch_and_spatial_per_channel() {
        // N=2,C=1,spatial=2: ch0 の 4 要素すべてが 1 チャネルへ縮約される
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 1, 2]).unwrap();
        let (_, mean, var) = batch_norm_train_channels(&x, None, None, 0.0, 2, 1, 2);
        assert!((mean[0] as f64 - 2.5).abs() < 1e-9);
        // Σ(x-mean)^2/M = (2.25+0.25+0.25+2.25)/4 = 1.25
        assert!((var[0] as f64 - 1.25).abs() < 1e-9);
    }

    #[test]
    fn batch_norm_train_applies_weight_and_bias() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();
        let w = [2.0f32, 0.5];
        let b = [1.0f32, -1.0];
        let (out, _, _) = batch_norm_train_channels(&x, Some(&w), Some(&b), 0.0, 1, 2, 2);
        let out = dense_vec(&out);
        let rstd = 1.0f32 / 0.25f32.sqrt();
        let expected = [
            (1.0 - 1.5) * rstd * 2.0 + 1.0,
            (2.0 - 1.5) * rstd * 2.0 + 1.0,
            (3.0 - 3.5) * rstd * 0.5 - 1.0,
            (4.0 - 3.5) * rstd * 0.5 - 1.0,
        ];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "o={o} e={e}");
        }
    }

    #[test]
    fn batch_norm_train_empty_axes_are_empty_output() {
        let x0 = Tensor::new(Vec::<f32>::new(), &[0, 2, 2]).unwrap();
        let (out, mean, var) = batch_norm_train_channels(&x0, None, None, 1e-5, 0, 2, 2);
        assert_eq!(dense_vec(&out), Vec::<f32>::new());
        assert_eq!(mean, vec![0.0f32; 2]);
        assert_eq!(var, vec![0.0f32; 2]);
    }

    #[test]
    fn batch_norm_train_nan_propagates() {
        let x = Tensor::new(vec![f32::NAN, 1.0, 1.0, 1.0], &[1, 2, 2]).unwrap();
        let (out, mean, _) = batch_norm_train_channels(&x, None, None, 1e-5, 1, 2, 2);
        let out = dense_vec(&out);
        assert!(out[0].is_nan() && out[1].is_nan());
        assert!(mean[0].is_nan());
        assert!(!mean[1].is_nan());
    }

    #[test]
    fn batch_norm_train_extreme_scale_does_not_overflow_stats() {
        let x = Tensor::new(vec![2e20f32, -2e20, 2e20, -2e20], &[1, 2, 2]).unwrap();
        let (out, _, _) = batch_norm_train_channels(&x, None, None, 1e-5, 1, 2, 2);
        assert!(dense_vec(&out).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn batch_norm_infer_uses_fixed_stats_not_batch_stats() {
        let x = Tensor::new(vec![10.0f32, 20.0, 30.0, 40.0], &[1, 2, 2]).unwrap();
        let mean = [0.0f32, 0.0];
        let var = [1.0f32, 1.0];
        let out = dense_vec(&batch_norm_infer_channels(
            &x, &mean, &var, None, None, 0.0, 1, 2, 2,
        ));
        // with mean=0,var=1 output equals input unchanged
        assert_eq!(out, vec![10.0f32, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn batch_norm_infer_empty_axes_are_empty_output() {
        let x0 = Tensor::new(Vec::<f32>::new(), &[0, 2, 2]).unwrap();
        let mean = [0.0f32, 0.0];
        let var = [1.0f32, 1.0];
        let out = batch_norm_infer_channels(&x0, &mean, &var, None, None, 1e-5, 0, 2, 2);
        assert_eq!(dense_vec(&out), Vec::<f32>::new());
    }
}

#[cfg(test)]
mod reduce_bias_grad_rows_tests {
    use super::*;

    // イシュー #1566・PR #1659 codex-review P1 是正（2026-09-12 ユーザー
    // 承認 A）: `reduce_bias_grad_rows` は `f64` アキュムレータで列ごと
    // に蓄積するため、単純な `f32` 逐次 `+=` なら桁落ちで消える寄与
    // （`1e8 + 1.0 + (-1e8)` の `1.0`）が保持されることを確認する
    // （`.claude/rules/coding-rust.md` の勾配長軸縮約 f64 方針）。
    #[test]
    fn preserves_cancelling_contribution_via_f64_accumulator() {
        // 列 0: 1e8 + 1.0 + (-1e8) は f32 逐次和だと桁落ちで 1.0 の
        // 寄与が失われ 0.0 になる（以前の実装の回帰記録は git 履歴
        // 参照）が、f64 アキュムレータでは 1.0 が正しく残る。
        let g = Tensor::<f32>::new(vec![1.0e8, 10.0, 1.0, 20.0, -1.0e8, 30.0], &[3, 2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let got = reduce_bias_grad_rows(&g);

        // 参考: 素朴な f32 逐次和では 1.0 の寄与が失われることの確認
        // （contrast のための計算。got との比較には使わない）。
        let mut naive_f32_col0 = 0.0f32;
        naive_f32_col0 += 1.0e8;
        naive_f32_col0 += 1.0;
        naive_f32_col0 += -1.0e8;
        assert_eq!(
            naive_f32_col0, 0.0,
            "対照: f32 逐次和では桁落ちにより 1.0 の寄与が失われる"
        );

        assert_eq!(got.len(), 2);
        assert_eq!(
            got[0], 1.0,
            "f64 アキュムレータでは 1e8 + 1.0 + (-1e8) の 1.0 が保持されるはず"
        );
        assert_eq!(got[1], 60.0);
    }

    #[test]
    fn preserves_negative_zero_and_nan_and_inf() {
        let g = Tensor::<f32>::new(
            vec![-0.0, f32::NAN, f32::INFINITY, 1.0, -0.0, f32::NEG_INFINITY],
            &[3, 2],
        )
        .expect("test fixture: shape とデータ長は事前に一致させている");
        let got = reduce_bias_grad_rows(&g);
        assert_eq!(got.len(), 2);
        // row0=[-0.0, NaN]・row1=[+inf, 1.0]・row2=[-0.0, -inf]
        // （data は row-major: [row0col0, row0col1, row1col0, ...]）。
        // col0: -0.0 + (+inf) + -0.0 = +inf
        assert!(got[0].is_infinite() && got[0] > 0.0);
        // col1: NaN + 1.0 + -inf = NaN（NaN の伝播）
        assert!(got[1].is_nan());
    }

    #[test]
    fn single_row_returns_row_unchanged() {
        let g = Tensor::<f32>::new(vec![1.5, -2.5, 3.5], &[1, 3])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let got = reduce_bias_grad_rows(&g);
        assert_eq!(got, vec![1.5f32, -2.5, 3.5]);
    }

    // PR #1659 codex-review P2 是正の回帰テスト（`reduce_bias_grad_rows`
    // doc「`m == 1` の特殊扱い」）: `m == 1` の単純な `+=` 版
    // （`0.0f32 + (-0.0f32) == +0.0f32`）だと符号付きゼロが失われる
    // ことを直接検知する（`is_sign_negative` で `+0.0`/`-0.0` を区別）。
    #[test]
    fn single_row_preserves_negative_zero_sign() {
        let g = Tensor::<f32>::new(vec![-0.0f32, 0.0f32], &[1, 2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let got = reduce_bias_grad_rows(&g);
        assert_eq!(got.len(), 2);
        assert!(
            got[0].is_sign_negative(),
            "m == 1 では -0.0 の符号を保持するはず（reduce_to_shape との bit 完全一致契約）"
        );
        assert!(!got[1].is_sign_negative());
    }
}

#[cfg(test)]
mod log_softmax_along_precision_tests {
    use super::*;

    // codex-review 指摘（PR #1664）の回帰検証: `m + ln(sum_exp)` を
    // 先に加算してから `x` から引く実装では、`m` が大きい共通オフセット
    // を持つ入力で丸め落ちが発生し、`log_softmax([1e8, 1e8])` が
    // 期待値 `[-ln(2), -ln(2)]` ではなく `[0.0, 0.0]` になっていた
    // （`m + ln(2)` が `f32` の丸め精度で `m` そのものに丸まるため）。
    // `(x - m) - ln(sum_exp)` の順で計算することで `x - m` を Sterbenz
    // の補題により誤差なく求め、丸め落ちを避ける。
    #[test]
    fn large_common_offset_does_not_round_away_ln_sum_exp() {
        let input = Tensor::<f32>::new(vec![1e8, 1e8], &[1, 2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let out = log_softmax_along(&input, 1);
        let expected = -(2.0f32).ln();
        for c in 0..2 {
            let v = out.get(&[0, c]).unwrap();
            assert!(
                (v - expected).abs() < 1e-4,
                "log_softmax([1e8,1e8])[{c}] = {v}（期待値 {expected} 近傍）"
            );
        }
    }
}

#[cfg(test)]
mod softmax_empty_tensor_overflow_tests {
    use super::*;

    // codex-review 指摘（PR #1664）の回帰検証: `Tensor::new(vec![],
    // &[0, 0, usize::MAX, 2])` は `checked_numel` が要素数積を `0`
    // （先頭の `0` が後続の積を吸収する）と評価するため構築できるが、
    // `log_softmax_along(input, 1)` の `inner = shape[2..].iter()
    // .product()`（`= usize::MAX * 2`）はこの吸収を経由しない部分積
    // のため、overflow チェック有効時に本番経路の外で panic していた。
    // `softmax_along`／`log_softmax_along` 冒頭の早期 return
    // （`shape` がいずれかの次元 `0` を含めば空出力を返す）で、
    // 部分積を計算する前に安全側へ倒れることを確認する。
    #[test]
    fn log_softmax_along_empty_tensor_with_overflow_prone_inner_does_not_panic() {
        let shape = [0usize, 0, usize::MAX, 2];
        let input = Tensor::<f32>::new(Vec::new(), &shape)
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let out = log_softmax_along(&input, 1);
        assert_eq!(out.shape(), &shape);
        assert_eq!(out.numel(), 0);
    }

    #[test]
    fn softmax_along_empty_tensor_with_overflow_prone_inner_does_not_panic() {
        let shape = [0usize, 0, usize::MAX, 2];
        let input = Tensor::<f32>::new(Vec::new(), &shape)
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let out = softmax_along(&input, 1);
        assert_eq!(out.shape(), &shape);
        assert_eq!(out.numel(), 0);
    }
}

#[cfg(test)]
mod concat_empty_out_shape_overflow_tests {
    use super::*;

    // codex-review 指摘（PR #1680）の回帰検証: `concat_out_shape` が
    // 受理しうる有効な空 `out_shape`（先頭が `0` で後続次元の部分積が
    // overflow するケース）に対し、`concat` が `outer`／`inner` を
    // ゼロ軸チェックより先に `.iter().product()` で計算していたため、
    // overflow チェック有効時に本番経路の外（debug ビルド）で panic
    // していた（`softmax_along` と同型の bug。上記
    // `softmax_empty_tensor_overflow_tests` 参照）。`out_shape` に `0`
    // を含む場合は部分積を計算する前に空出力へ早期 return することを
    // 確認する（`dim=0` のとき `inner = out_shape[1..].iter().product()`
    // `= usize::MAX * 2` が旧実装で overflow していた）。
    #[test]
    fn concat_empty_out_shape_with_overflow_prone_inner_does_not_panic() {
        let out_shape = [0usize, usize::MAX, 2];
        let out = concat(&[], 0, &out_shape);
        assert_eq!(out.shape(), &out_shape);
        assert_eq!(out.numel(), 0);
    }

    // イシュー #1834 codex-review P1 是正: `nearest_src_coord`（`grad::
    // nearest_src_index_map`〈backward の index 構築。forward の
    // バックエンド成否に関わらず常に呼ばれる〉と `interpolate_nearest`
    // 〈forward のホスト参照実装〉の単一情報源）の中間積 `dst *
    // in_size` が `usize` を overflow しないことの回帰テスト。
    #[test]
    fn nearest_src_coord_does_not_overflow_for_huge_in_size() {
        // `dst=2`・`in_size=2^63`・`out_size=3` は素朴な `usize` 乗算
        // （`dst * in_size = 2^64`）が overflow する組み合わせ（debug
        // ビルドでは overflow panic・release ビルドでは wrap して誤った
        // 添字を返す）。backend-cpu クレートの `interpolate::nearest_src_coord` と同型の overflow・同型の是正。
        let in_size = 1usize << 63;
        let src = nearest_src_coord(2, in_size, 3);
        let expected = ((2u128 * in_size as u128) / 3) as usize;
        assert_eq!(src, expected);
        assert!(src < in_size);
    }
}

#[cfg(test)]
mod pad_tests {
    use super::*;
    use fandhe_ai_tensor_core::pad_out_shape;

    #[test]
    fn pad_1d_basic() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        let pads = [(1usize, 2usize)];
        let out_shape = pad_out_shape(x.shape(), &pads).unwrap();
        let out = pad(&x, &pads, 0.0, &out_shape);
        assert_eq!(out.shape(), &[6]);
        assert_eq!(
            out.contiguous().as_slice().unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
        );
    }

    #[test]
    fn pad_2d_both_axes_with_nonzero_value() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let pads = [(1usize, 0usize), (0usize, 1usize)];
        let out_shape = pad_out_shape(x.shape(), &pads).unwrap();
        let out = pad(&x, &pads, -1.0, &out_shape);
        assert_eq!(out.shape(), &[3, 3]);
        assert_eq!(
            out.contiguous().as_slice().unwrap(),
            &[-1.0, -1.0, -1.0, 1.0, 2.0, -1.0, 3.0, 4.0, -1.0]
        );
    }

    #[test]
    fn pad_empty_input_fills_all_value() {
        let x = Tensor::new(Vec::<f32>::new(), &[0, 3]).unwrap();
        let pads = [(1usize, 1usize), (0usize, 0usize)];
        let out_shape = pad_out_shape(x.shape(), &pads).unwrap();
        let out = pad(&x, &pads, 7.0, &out_shape);
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.contiguous().as_slice().unwrap(), &[7.0; 6]);
    }

    #[test]
    fn pad_noncontiguous_view_input() {
        // transpose 直後の view（非 contiguous）でも `dense_vec_ref`
        // 経由で正しく読めることを確認する（`Var::cat` の入力契約と
        // 同型）。
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        // transpose 後の論理 shape は [3, 2]:
        // [[1, 4], [2, 5], [3, 6]]
        let pads = [(1usize, 0usize), (0usize, 0usize)];
        let out_shape = pad_out_shape(x.shape(), &pads).unwrap();
        let out = pad(&x, &pads, 0.0, &out_shape);
        assert_eq!(out.shape(), &[4, 2]);
        assert_eq!(
            out.contiguous().as_slice().unwrap(),
            &[0.0, 0.0, 1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn pad_value_nan_and_neg_zero_bit_preserved() {
        let x = Tensor::new(vec![1.0f32], &[1]).unwrap();
        let pads = [(1usize, 1usize)];
        let out_shape = pad_out_shape(x.shape(), &pads).unwrap();
        let out = pad(&x, &pads, f32::NAN, &out_shape);
        let data = out.contiguous();
        let slice = data.as_slice().unwrap();
        assert!(slice[0].is_nan());
        assert_eq!(slice[1], 1.0);
        assert!(slice[2].is_nan());

        let out_neg_zero = pad(&x, &pads, -0.0, &out_shape);
        let data2 = out_neg_zero.contiguous();
        let slice2 = data2.as_slice().unwrap();
        assert_eq!(slice2[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(slice2[2].to_bits(), (-0.0f32).to_bits());
    }

    // codex-review 指摘（PR #1680）の回帰検証と同型（上記
    // `concat_empty_out_shape_with_overflow_prone_inner_does_not_panic`
    // 参照）: `pad_out_shape` が受理しうる有効な空 `out_shape`
    // （先頭が `0` で後続次元の部分積が overflow するケース）に対し
    // `pad` が空軸チェックより先に `.iter().product()` を計算すると
    // overflow する。空軸チェックを先に行うことを確認する。
    #[test]
    fn pad_empty_out_shape_with_overflow_prone_product_does_not_panic() {
        let x = Tensor::new(vec![1.0f32], &[1]).unwrap();
        let out_shape = [0usize, usize::MAX, 2];
        let pads = [(0usize, 0usize), (0usize, usize::MAX - 1), (0usize, 0usize)];
        let out = pad(&x, &pads, 0.0, &out_shape);
        assert_eq!(out.shape(), &out_shape);
        assert_eq!(out.numel(), 0);
    }

    #[test]
    fn pad_all_zero_pads_is_identity_copy() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let pads = [(0usize, 0usize), (0usize, 0usize)];
        let out_shape = pad_out_shape(x.shape(), &pads).unwrap();
        let out = pad(&x, &pads, 0.0, &out_shape);
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(
            out.contiguous().as_slice().unwrap(),
            x.contiguous().as_slice().unwrap()
        );
    }
}
