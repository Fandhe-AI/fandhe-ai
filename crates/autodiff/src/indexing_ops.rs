//! advanced indexing（複数軸整数配列索引の読み出し）・`index_put`・
//! `index_put_`（イシュー #2148・親 #2131「5-B 演算」）。
//!
//! **facade 非公開（意図的）**: `crates/autodiff/src/reduce_ops.rs`
//! モジュール doc と同じ理由・同じ判断枠組みによる。`Var` は facade
//! （`fandhe_ai` クレート）から直接再エクスポートされるため、`Var` への
//! inherent メソッド追加は即座に facade 公開面へ出てしまう。イシュー
//! #2148 本文は facade 公開面（`Var::advanced_indexing`／`index_put`／
//! `index_put_` の委譲メソッド）を承認事項として明示し、親 #2131 は
//! このツリーに限り「設計判断記録 → 承認 → 実装」の 2 段階を定めるため、
//! 承認が取れるまでは自由関数として `Var` の外に置き到達不能にする
//! （`docs/autodiff-indexing-inplace-design.md` §6）。承認後は
//! `Var::advanced_indexing` 等の薄い委譲メソッドを追加し、facade 側の
//! 保留ガード（`crates/facade/src/lib.rs::
//! VarIndexingOpsHoldDoctestGuard`）を撤去する。
//!
//! **「in-place」の解釈（不変値 API との整合）**: `Var<'t>` は `Copy` な
//! ハンドル、`Tensor<f32>` は `Arc` を共有する immutable 値、tape は
//! append-only であるため、バッファを書き換える本当の in-place は
//! データモデル上表現できない。[`index_put_`] は「`index_put` を呼んで
//! ローカルハンドルを再束縛するだけ」の糖衣であり、tape ノードも
//! `Tensor` も書き換えない。詳細・PyTorch との差異（同じノードを指す
//! 他の `Var` コピーが古い値のまま・旧ノードは tape に残り勾配もそのまま
//! 受け取る等）は `docs/autodiff-indexing-inplace-design.md` §2.1・§4 を
//! 正とする。
//!
//! **PyTorch 相当・出力**:
//!
//! | 演算 | PyTorch 相当 | 微分 |
//! |---|---|---|
//! | [`advanced_indexing`] | `x[i0, i1, …]`（先頭 k 軸の整数配列索引） | 可（`Op::Gather` の VJP） |
//! | [`index_put`] | `torch.index_put` | 可（`Op::Scatter` の VJP） |
//! | [`index_put_`] | `Tensor.index_put_`／`x[idx] = v`（再束縛による糖衣） | [`index_put`] と同じ |
//!
//! **新規 `Op` はゼロ**: 既存の `Op::Gather`（[`Var::index_select`] 経由）・
//! `Op::Scatter`（[`Var::scatter`]／[`Var::scatter_add`] 経由）と view 系
//! （`reshape`／`broadcast_to`／`contiguous`）の合成のみで forward・VJP
//! 双方の意味論を過不足なく表現できる（`docs/autodiff-indexing-inplace-
//! design.md` §2.3）。`BackendOps`／`crates/backend-*` は変更しない。
//!
//! **合成方式**（詳細は `docs/autodiff-indexing-inplace-design.md`
//! §2.4）。用語: `x.shape = [d0, …, d_{k-1}] ++ R`（`k = indices.len()`、
//! `R` は残り軸）、`B = broadcast_shape(indices[*].shape())`、
//! `M = numel(B)`、`P = d0 × … × d_{k-1}`。
//! - 共通前処理 `plan_flat_index`: `k` の範囲検査 → `B` の確定 →
//!   `P`／`M` の確保前検査（[`crate::rearrange_ops::
//!   checked_axis_len_as_i32`]／[`crate::rearrange_ops::
//!   checked_index_alloc_len`]）→ 各添字を `B` へ broadcast して
//!   行優先に走査し範囲検査（`0 <= v < d_j`。負値は拒否）→
//!   `flat = Σ v_j × stride_j` を `checked_mul`／`checked_add` で計算し
//!   `Tensor<i32>` `[M]` を返す。
//! - [`advanced_indexing`]: `x.contiguous() → reshape([P] ++ R) →
//!   index_select(0, flat) → reshape(B ++ R)`。
//! - [`index_put`]: `values` を `B ++ R` へ broadcast・実体化してから
//!   `[M] ++ R` へ reshape、`flat` を `[M, 1, …, 1] → broadcast_to([M]
//!   ++ R)` へ拡張し、`x` を `[P] ++ R` へ reshape したうえで
//!   `accumulate` に応じ `scatter`（`Overwrite`）／`scatter_add`
//!   （`Add`）を呼び、最後に `x.shape()` へ reshape する。
//!
//! **数値契約**（`docs/autodiff-indexing-inplace-design.md` §3）:
//! - [`advanced_indexing`] の forward・[`index_put`]（`accumulate:
//!   false`）の forward はコピーのみ（算術なし）のため 3 バックエンドで
//!   構造的に **bit 完全一致**する（`NaN` の payload も保たれる）。
//! - [`index_put`]（`accumulate: true`）の forward、重複添字を含む
//!   backward（`scatter_add`）は `ScatterReduce::Add` の `f64` 決定的
//!   集約契約（[`fandhe_ai_tensor_core::ScatterReduce`] doc）に従う。
//! - 重複の無い backward は各位置への寄与が高々 1 つで残りは厳密な
//!   `+0.0` のため bit 一致する。
//!
//! **PyTorch との差異**（`docs/autodiff-indexing-inplace-design.md`
//! §4）:
//! - 重複添字で `accumulate: false` の場合、PyTorch は未定義だが本実装は
//!   `ScatterReduce::Overwrite` の契約により B の行優先走査で
//!   **最後の書き手が勝つ決定的な結果**になる（PyTorch より強い契約）。
//! - 負の添字（wrap-around）は拒否する（`gather`／`scatter` の既存契約と
//!   揃え fail-closed にする）。
//! - 索引できるのは先頭の連続 `k` 軸のみ。スライス・`None` を挟む指定・
//!   bool マスク索引・int64 添字は対象外。
//! - [`index_put_`] はエイリアスが無いため、PyTorch が拒否する
//!   requires_grad な葉への in-place も許容する（`.claude/rules/
//!   security.md` の趣旨に反しない安全側の緩和。§2.1 参照）。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: 外部から来る
//! 添字は `plan_flat_index` が「バックエンドの呼び出し・確保前」に
//! すべて検査する（範囲・`k`・broadcast 可能性・確保サイズ）。
//! [`advanced_indexing`]／[`index_put`] の両公開入口は、`x`（および
//! `index_put` は `values`）の確保前サイズを [`crate::bool_ops::
//! checked_bytes_for`] で中間 shape ごとに検査してから合成へ進む。
//! 本番経路で `unwrap()`／`expect()` は使わない。

use fandhe_ai_tensor_core::{ShapeError, Tensor, broadcast_shape};

use crate::bool_ops::checked_bytes_for;
use crate::error::AutodiffError;
use crate::eval;
use crate::rearrange_ops::{checked_axis_len_as_i32, checked_index_alloc_len};
use crate::var::Var;

/// [`advanced_indexing`]・[`index_put`] が共有する前処理（モジュール doc
/// 「合成方式」参照）。`x_shape`（`x` の shape）と `indices`（`k` 本の
/// `Tensor<i32>`）から `B`（broadcast shape）・`P`（先頭 `k` 軸の要素数
/// 積）・`flat`（`Tensor<i32>` `[M]`。`M = numel(B)`）を求める。
///
/// **呼び出し規律**: `flat` の各値を計算する（バックエンドを呼ぶ・
/// 確保する）よりも前に、`k` の範囲・`P`／`M` の確保前サイズ
/// （[`checked_axis_len_as_i32`]／[`checked_index_alloc_len`]）・各添字の
/// 範囲（`0 <= v < d_j`）のすべてを検査する（REQ-8「境界検査を
/// バックエンド呼び出し・確保より前に行う」の趣旨。本番経路 panic
/// 禁止規約 `.claude/rules/coding-rust.md`）。
fn plan_flat_index(
    x_shape: &[usize],
    indices: &[Tensor<i32>],
) -> Result<(Vec<usize>, usize, Tensor<i32>), AutodiffError> {
    let rank = x_shape.len();
    let k = indices.len();
    if k == 0 || k > rank {
        return Err(AutodiffError::InvalidArgument(format!(
            "indexing_ops: indices.len() ({k}) は 1..={rank}（x の rank）の範囲でなければならない"
        )));
    }

    // B = 全添字テンソルの broadcast shape。
    let mut b_shape: Vec<usize> = indices[0].shape().to_vec();
    for idx in &indices[1..] {
        b_shape = broadcast_shape(&b_shape, idx.shape()).map_err(AutodiffError::Shape)?;
    }

    // P = d0 * ... * d_{k-1}（先頭 k 軸の要素数積）。添字値を i32 の
    // 平坦添字として使うため i32 範囲へ収まるかも検査する。
    let head = &x_shape[..k];
    let p = head
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)
        .map_err(AutodiffError::Shape)?;
    checked_axis_len_as_i32(p)?;

    // M = numel(B)。添字ベクタ Vec<i32>（長さ M）を確保する前に上限を
    // 検査する（`checked_index_alloc_len` は `usize` オーバーフローに
    // 加え実用上の確保上限〈既定 1 GiB〉も検査する）。
    let m = b_shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)
        .map_err(AutodiffError::Shape)?;
    checked_index_alloc_len(m)?;

    // head 空間（P 次元）の行優先ストライド。stride[k-1] = 1、
    // stride[j] = stride[j+1] * head[j+1]。head の各要素は checked_mul
    // 済みの P 以下のため i64 での積・和は安全に収まる。
    let mut strides = vec![1i64; k];
    for j in (0..k.saturating_sub(1)).rev() {
        strides[j] = strides[j + 1] * head[j + 1] as i64;
    }

    let mut flat_i64 = vec![0i64; m];
    for (j, idx) in indices.iter().enumerate() {
        let idx_bc = idx.broadcast_to(&b_shape).map_err(AutodiffError::Shape)?;
        let dim_size = head[j];
        let vals = eval::dense_vec_i32(&idx_bc);
        for (pos, &v) in vals.iter().enumerate() {
            if v < 0 || (v as usize) >= dim_size {
                return Err(AutodiffError::InvalidArgument(format!(
                    "indexing_ops: indices[{j}] の添字 {v} が範囲 [0, {dim_size}) を外れている"
                )));
            }
            let contrib = (v as i64)
                .checked_mul(strides[j])
                .ok_or(ShapeError::ElementCountOverflow)
                .map_err(AutodiffError::Shape)?;
            flat_i64[pos] = flat_i64[pos]
                .checked_add(contrib)
                .ok_or(ShapeError::ElementCountOverflow)
                .map_err(AutodiffError::Shape)?;
        }
    }

    // `checked_axis_len_as_i32(p)` を通過済みのため、`[0, p)` の範囲に
    // 収まる `flat_i64` の各値は必ず `i32` へ無損失変換できる。
    let flat_i32: Vec<i32> = flat_i64.into_iter().map(|v| v as i32).collect();
    let flat = Tensor::new(flat_i32, &[m]).map_err(AutodiffError::Shape)?;

    Ok((b_shape, p, flat))
}

/// 先頭 `k = indices.len()` 軸を複数軸整数配列索引で読み出す
/// （`x[i0, i1, …]` 相当。イシュー #2148）。`indices` の各要素は
/// [`Var::gather`]／[`Var::scatter`] の `index` と同じ非追跡データ
/// （`Op` payload に直接埋め込む設計。勾配は `indices` 側には流れない）。
///
/// 残り軸 `R = x.shape()[k..]` は素通しし、出力 shape は
/// `broadcast_shape(indices[*].shape()) ++ R`。重複添字の勾配は
/// [`Var::index_select`]（内部 `Op::Gather`）の決定的 `scatter_add` VJP
/// により加算される（PyTorch と同じ）。数値契約・合成方式はモジュール
/// doc を正とする。
pub fn advanced_indexing<'t>(
    x: &Var<'t>,
    indices: &[Tensor<i32>],
) -> Result<Var<'t>, AutodiffError> {
    let x_shape = x.shape();
    checked_bytes_for::<f32>(&x_shape)?;
    let (b_shape, p, flat) = plan_flat_index(&x_shape, indices)?;
    let k = indices.len();
    let r_shape = x_shape[k..].to_vec();

    let mut out_shape = b_shape;
    out_shape.extend_from_slice(&r_shape);
    checked_bytes_for::<f32>(&out_shape)?;

    let mut flat_shape = vec![p];
    flat_shape.extend_from_slice(&r_shape);
    checked_bytes_for::<f32>(&flat_shape)?;

    let x_flat = x.contiguous()?.reshape(&flat_shape)?;
    let gathered = x_flat.index_select(0, &flat)?;
    gathered.reshape(&out_shape)
}

/// 先頭 `k = indices.len()` 軸が指す位置へ `values` を書き込んだ
/// **非破壊**（新しい `Var` を返す）版を返す（`torch.index_put`
/// 〈アンダースコアなし〉相当。イシュー #2148）。`accumulate: false` は
/// 上書き（重複添字は B の行優先走査で最後の書き手が勝つ決定的な結果。
/// モジュール doc「PyTorch との差異」参照）、`accumulate: true` は加算
/// （`ScatterReduce::Add` の `f64` 決定的集約契約）。
///
/// `values` は `self` と同じ `Tape` 上の `Var` であること（違反は
/// [`AutodiffError::TapeMismatch`]。`Var::scatter_impl` と同じ検査）。
/// `values` は出力の索引形状 `B ++ R`（`B = broadcast_shape(indices[*]
/// .shape())`、`R = x.shape()[k..]`）へ broadcast できる必要がある
/// （broadcast 不能は [`AutodiffError::Shape`]）。
pub fn index_put<'t>(
    x: &Var<'t>,
    indices: &[Tensor<i32>],
    values: &Var<'t>,
    accumulate: bool,
) -> Result<Var<'t>, AutodiffError> {
    x.check_same_tape(values)?;
    let x_shape = x.shape();
    checked_bytes_for::<f32>(&x_shape)?;
    let (b_shape, p, flat) = plan_flat_index(&x_shape, indices)?;
    let k = indices.len();
    let r_shape = x_shape[k..].to_vec();
    let r_rank = r_shape.len();
    let m = flat.shape()[0];

    let mut bcast_shape = b_shape;
    bcast_shape.extend_from_slice(&r_shape);
    checked_bytes_for::<f32>(&bcast_shape)?;

    let mut m_r_shape = vec![m];
    m_r_shape.extend_from_slice(&r_shape);
    checked_bytes_for::<f32>(&m_r_shape)?;

    let mut p_r_shape = vec![p];
    p_r_shape.extend_from_slice(&r_shape);
    checked_bytes_for::<f32>(&p_r_shape)?;

    // `values` を出力索引形状 `B ++ R` へ broadcast・実体化してから
    // `[M] ++ R` へ reshape（`reshape` は contiguous 入力を要求するため
    // `broadcast_to`〈stride-0 view を作りうる〉の直後に実体化する。
    // `Var::index_select` 等、本クレート既存の view 拡張と同じ手順）。
    let src = values
        .broadcast_to(&bcast_shape)?
        .contiguous()?
        .reshape(&m_r_shape)?;

    // `flat`（`[M]`）を `scatter` が要求する `index.shape() == src.shape()`
    // （`[M] ++ R`）へ拡張する。`Tensor<i32>` の view 演算のため
    // `Var::scatter`（内部で `index.contiguous()` を通す）へそのまま
    // 渡せる。
    let mut idx_expand_shape = vec![m];
    idx_expand_shape.extend(std::iter::repeat_n(1usize, r_rank));
    let idx_reshaped = flat
        .reshape(&idx_expand_shape)
        .map_err(AutodiffError::Shape)?;
    let idx_bcast = idx_reshaped
        .broadcast_to(&m_r_shape)
        .map_err(AutodiffError::Shape)?;

    let x_flat = x.contiguous()?.reshape(&p_r_shape)?;
    let out_flat = if accumulate {
        x_flat.scatter_add(0, &idx_bcast, &src)?
    } else {
        x_flat.scatter(0, &idx_bcast, &src)?
    };
    out_flat.reshape(&x_shape)
}

/// [`index_put`] を呼んで `*x` を再束縛する糖衣（`Tensor.index_put_`／
/// `x[idx] = v` 相当の in-place 代入。イシュー #2148）。`Var<'t>` は
/// `Copy` なハンドルであり tape ノードも `Tensor` も書き換えないため、
/// 同じノードを指す他の `Var` コピーは古い値のまま（モジュール doc
/// 「『in-place』の解釈」・`docs/autodiff-indexing-inplace-design.md`
/// §2.1 参照）。
pub fn index_put_<'t>(
    x: &mut Var<'t>,
    indices: &[Tensor<i32>],
    values: &Var<'t>,
    accumulate: bool,
) -> Result<(), AutodiffError> {
    *x = index_put(x, indices, values, accumulate)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    // --- advanced_indexing ---

    #[test]
    fn advanced_indexing_k1_matches_index_select() {
        let tape = Tape::new();
        // [[1,2],[3,4],[5,6]] から行 [2, 0] を読む。
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));
        let idx = ti(vec![2, 0], &[2]);
        let out = advanced_indexing(&x, &[idx]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[2, 2]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![5.0, 6.0, 1.0, 2.0]
        );
    }

    #[test]
    fn advanced_indexing_k_equals_rank_no_remaining_axes() {
        let tape = Tape::new();
        // [[1,2],[3,4]] を (row, col) の組で読む: (0,1)->2, (1,0)->3
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let rows = ti(vec![0, 1], &[2]);
        let cols = ti(vec![1, 0], &[2]);
        let out = advanced_indexing(&x, &[rows, cols]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[2]);
        assert_eq!(out.to_tensor().host_slice().into_owned(), vec![2.0, 3.0]);
    }

    #[test]
    fn advanced_indexing_broadcasts_index_shapes() {
        let tape = Tape::new();
        // x: [4, 2]。rows: [2,1] broadcast cols: [3] -> B=[2,3]。
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]));
        let rows = ti(vec![0, 3], &[2, 1]);
        let cols = ti(vec![0, 1, 0], &[3]);
        let out = advanced_indexing(&x, &[rows, cols]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[2, 3]);
        // row 0: [1,2] -> cols [0,1,0] -> [1,2,1]
        // row 3: [7,8] -> cols [0,1,0] -> [7,8,7]
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 1.0, 7.0, 8.0, 7.0]
        );
    }

    #[test]
    fn advanced_indexing_gradient_matches_finite_difference() {
        let eps = 1e-3f32;
        let base = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let idx_shape = [2usize];
        let idx = ti(vec![2, 0], &idx_shape);
        let eval_fn = |data: &[f32]| -> Vec<f32> {
            let tape = Tape::new();
            let x = tape.var(&t(data.to_vec(), &[3, 2]));
            let y = advanced_indexing(&x, std::slice::from_ref(&idx)).unwrap();
            y.to_tensor().host_slice().into_owned()
        };
        let tape = Tape::new();
        let x = tape.var(&t(base.clone(), &[3, 2]));
        let y = advanced_indexing(&x, std::slice::from_ref(&idx)).unwrap();
        // 出力全要素の和にして単一スカラーの勾配を検算する。
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += eps;
            let mut minus = base.clone();
            minus[i] -= eps;
            let numeric = (eval_fn(&plus).iter().sum::<f32>()
                - eval_fn(&minus).iter().sum::<f32>())
                / (2.0 * eps);
            assert!(
                (numeric - dx[i]).abs() < 1e-2,
                "advanced_indexing 勾配の有限差分検算が乖離: i={i} numeric={numeric} analytic={}",
                dx[i]
            );
        }
    }

    #[test]
    fn advanced_indexing_duplicate_indices_accumulate_gradient() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let idx = ti(vec![0, 0, 1], &[3]);
        let y = advanced_indexing(&x, &[idx]).unwrap();
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        // index 0 は 2 回読まれるため勾配 2、index 1 は 1 回で勾配 1、
        // index 2 は未読で勾配 0。
        assert_eq!(dx, vec![2.0, 1.0, 0.0]);
    }

    #[test]
    fn advanced_indexing_rejects_k_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(
            advanced_indexing(&x, &[]),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn advanced_indexing_rejects_k_greater_than_rank() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let idx = ti(vec![0], &[1]);
        assert!(matches!(
            advanced_indexing(&x, &[idx.clone(), idx]),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn advanced_indexing_rejects_negative_index() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let idx = ti(vec![-1], &[1]);
        assert!(matches!(
            advanced_indexing(&x, &[idx]),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn advanced_indexing_rejects_out_of_range_index() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let idx = ti(vec![3], &[1]);
        assert!(matches!(
            advanced_indexing(&x, &[idx]),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn advanced_indexing_rejects_broadcast_incompatible_indices() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let rows = ti(vec![0, 1], &[2]);
        let cols = ti(vec![0, 1, 0], &[3]);
        assert!(matches!(
            advanced_indexing(&x, &[rows, cols]),
            Err(AutodiffError::Shape(_))
        ));
    }

    #[test]
    fn advanced_indexing_empty_index_returns_empty_output() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let idx = ti(vec![], &[0]);
        let out = advanced_indexing(&x, &[idx]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[0]);
    }

    // --- index_put ---

    #[test]
    fn index_put_overwrite_basic() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        let idx = ti(vec![1, 3], &[2]);
        let values = tape.var(&t(vec![20.0, 40.0], &[2]));
        let out = index_put(&x, &[idx], &values, false).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 20.0, 3.0, 40.0]
        );
    }

    #[test]
    fn index_put_overwrite_duplicate_last_writer_wins() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0, 0.0], &[2]));
        let idx = ti(vec![0, 0], &[2]);
        let values = tape.var(&t(vec![1.0, 2.0], &[2]));
        let out = index_put(&x, &[idx], &values, false).unwrap();
        // B の行優先走査で最後の書き手（values[1] = 2.0）が勝つ。
        assert_eq!(out.to_tensor().host_slice().into_owned(), vec![2.0, 0.0]);
    }

    #[test]
    fn index_put_accumulate_sums_duplicates() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![10.0, 0.0], &[2]));
        let idx = ti(vec![0, 0], &[2]);
        let values = tape.var(&t(vec![1.0, 2.0], &[2]));
        let out = index_put(&x, &[idx], &values, true).unwrap();
        assert_eq!(out.to_tensor().host_slice().into_owned(), vec![13.0, 0.0]);
    }

    #[test]
    fn index_put_with_remaining_axes_and_values_broadcast() {
        let tape = Tape::new();
        // x: [3, 2]。行 1 を [9, 9] へ上書き（values はスカラー相当の [1,2] broadcast）。
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));
        let idx = ti(vec![1], &[1]);
        let values = tape.var(&t(vec![9.0, 9.0], &[1, 2]));
        let out = index_put(&x, &[idx], &values, false).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 9.0, 9.0, 5.0, 6.0]
        );
    }

    #[test]
    fn index_put_overwrite_gradient_zeroes_overwritten_positions() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let idx = ti(vec![1], &[1]);
        let values = tape.var(&t(vec![5.0], &[1]));
        let out = index_put(&x, &[idx], &values, false).unwrap();
        let loss = out.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        let dv = grads
            .get(&values)
            .unwrap()
            .unwrap()
            .host_slice()
            .into_owned();
        // 上書きされた位置 1 の d_x は 0、それ以外は 1（Overwrite VJP）。
        assert_eq!(dx, vec![1.0, 0.0, 1.0]);
        assert_eq!(dv, vec![1.0]);
    }

    #[test]
    fn index_put_accumulate_gradient_matches_finite_difference() {
        let eps = 1e-3f32;
        let base = vec![1.0f32, 2.0, 3.0];
        let vbase = vec![0.5f32, -0.5];
        let idx = ti(vec![0, 2], &[2]);
        let eval_fn = |x_data: &[f32], v_data: &[f32]| -> f32 {
            let tape = Tape::new();
            let x = tape.var(&t(x_data.to_vec(), &[3]));
            let values = tape.var(&t(v_data.to_vec(), &[2]));
            let out = index_put(&x, std::slice::from_ref(&idx), &values, true).unwrap();
            out.sum(None).unwrap().to_tensor().host_slice()[0]
        };
        let tape = Tape::new();
        let x = tape.var(&t(base.clone(), &[3]));
        let values = tape.var(&t(vbase.clone(), &[2]));
        let out = index_put(&x, std::slice::from_ref(&idx), &values, true).unwrap();
        let loss = out.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += eps;
            let mut minus = base.clone();
            minus[i] -= eps;
            let numeric = (eval_fn(&plus, &vbase) - eval_fn(&minus, &vbase)) / (2.0 * eps);
            assert!(
                (numeric - dx[i]).abs() < 1e-2,
                "index_put(accumulate) の x 勾配の有限差分検算が乖離: i={i} numeric={numeric} analytic={}",
                dx[i]
            );
        }
    }

    #[test]
    fn index_put_rejects_tape_mismatch() {
        let tape_a = Tape::new();
        let tape_b = Tape::new();
        let x = tape_a.var(&t(vec![1.0, 2.0], &[2]));
        let values = tape_b.var(&t(vec![9.0], &[1]));
        let idx = ti(vec![0], &[1]);
        assert!(matches!(
            index_put(&x, &[idx], &values, false),
            Err(AutodiffError::TapeMismatch)
        ));
    }

    #[test]
    fn index_put_rejects_values_broadcast_incompatible() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let idx = ti(vec![0, 1], &[2]);
        // 索引形状 B ++ R = [2, 2] に対し values は [3] で broadcast 不能。
        let values = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        assert!(matches!(
            index_put(&x, &[idx], &values, false),
            Err(AutodiffError::Shape(_))
        ));
    }

    #[test]
    fn index_put_rejects_negative_index() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let idx = ti(vec![-1], &[1]);
        let values = tape.var(&t(vec![9.0], &[1]));
        assert!(matches!(
            index_put(&x, &[idx], &values, false),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn index_put_identity_when_index_empty() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let idx = ti(vec![], &[0]);
        let values = tape.var(&t(vec![], &[0]));
        let out = index_put(&x, &[idx], &values, false).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 3.0]
        );
    }

    // --- index_put_ ---

    #[test]
    fn index_put_in_place_rebinds_handle() {
        let tape = Tape::new();
        let mut x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let old = x;
        let idx = ti(vec![1], &[1]);
        let values = tape.var(&t(vec![9.0], &[1]));
        index_put_(&mut x, &[idx], &values, false).unwrap();
        assert_eq!(x.to_tensor().host_slice().into_owned(), vec![1.0, 9.0, 3.0]);
        // 旧ハンドルは古い値のまま（エイリアスが無いため）。
        assert_eq!(
            old.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn index_put_in_place_both_handles_receive_gradients() {
        let tape = Tape::new();
        let mut x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let old = x;
        let idx = ti(vec![1], &[1]);
        let values = tape.var(&t(vec![9.0], &[1]));
        index_put_(&mut x, &[idx], &values, false).unwrap();
        // 新ノード（x）を根に逆伝播すると、新旧いずれのハンドルでも
        // 対応するノードの勾配を取得できる（旧ノードは tape に残る）。
        let loss = x.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        assert!(grads.get(&x).unwrap().is_some());
        assert!(grads.get(&old).unwrap().is_some());
    }
}
