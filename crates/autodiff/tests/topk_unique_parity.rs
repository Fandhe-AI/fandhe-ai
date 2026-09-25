//! `fandhe_ai_autodiff::topk_unique_ops`（イシュー #2153・親 #2131）の
//! end-to-end 統合テスト（`NaiveOps` 上）。
//!
//! - topk: 負 dim（`-1`／`-rank`）が対応する正 dim と bit 同一である
//!   こと、範囲外 dim のエラー型、`sorted=false` の出力が「`sorted=
//!   true` 出力を index 昇順に並べ替えたもの」と一致すること、
//!   `sorted=true` が既存 `Var::topk` と bit 同一であること、
//!   `sorted=false` の勾配が `sorted=true` の勾配と bit 同一である
//!   こと（VJP は index 順序に依存しない scatter ベースのため）。
//! - unique: `dim=None` で values が既存 `Var::unique` と bit 同一で
//!   あること、`inverse` による再構成、`counts` の総和、`dim` 指定
//!   （dim=0／1／負）でのスライス単位重複除去、`unique_consecutive`
//!   の先頭出現代表・元順序保持。
//! - 共通: 呼び出し前後で tape ノード数が不変（unique 系。非微分・
//!   detached の契約固定）。

mod common;

use fandhe_ai_autodiff::topk_unique_ops::{
    TopkOptions, UniqueOptions, unique_consecutive, unique_with_options,
};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense_vec(tensor: &Tensor<f32>) -> Vec<f32> {
    let shape = tensor.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut out = Vec::with_capacity(numel);
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..numel {
        out.push(tensor.get(&idx).unwrap_or(0.0));
        for axis in (0..shape.len()).rev() {
            idx[axis] += 1;
            if idx[axis] < shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }
    out
}

fn dense_vec_i32(tensor: &Tensor<i32>) -> Vec<i32> {
    let shape = tensor.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut out = Vec::with_capacity(numel);
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..numel {
        out.push(tensor.get(&idx).unwrap_or(0));
        for axis in (0..shape.len()).rev() {
            idx[axis] += 1;
            if idx[axis] < shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }
    out
}

// ---------------------------------------------------------------------
// topk_with_options
// ---------------------------------------------------------------------

/// 負 dim（`-1`）が対応する正 dim（rank-1）と forward bit 同一。
#[test]
fn topk_negative_dim_matches_positive_dim() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]));
    let (out_neg, idx_neg) = fandhe_ai_autodiff::topk_unique_ops::topk_with_options(
        &x,
        2,
        TopkOptions::default().with_dim(-1),
    )
    .unwrap();
    let (out_pos, idx_pos) = fandhe_ai_autodiff::topk_unique_ops::topk_with_options(
        &x,
        2,
        TopkOptions::default().with_dim(1),
    )
    .unwrap();
    assert_eq!(
        dense_vec(&out_neg.to_tensor()),
        dense_vec(&out_pos.to_tensor())
    );
    assert_eq!(dense_vec_i32(&idx_neg), dense_vec_i32(&idx_pos));
}

/// `dim = -rank`（先頭軸）は正しく正規化される。
#[test]
fn topk_negative_dim_full_range() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 4.0, 2.0, 3.0], &[4, 1]));
    let (out, idx) = fandhe_ai_autodiff::topk_unique_ops::topk_with_options(
        &x,
        2,
        TopkOptions::default().with_dim(-2).with_largest(true),
    )
    .unwrap();
    assert_eq!(out.to_tensor().shape(), &[2, 1]);
    assert_eq!(dense_vec(&out.to_tensor()), vec![4.0, 3.0]);
    assert_eq!(dense_vec_i32(&idx), vec![1, 3]);
}

/// 範囲外 dim（`rank`・`-rank-1`）は型付きエラーになる。
#[test]
fn topk_dim_out_of_range_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));

    let err = fandhe_ai_autodiff::topk_unique_ops::topk_with_options(
        &x,
        1,
        TopkOptions::default().with_dim(1),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::AxisOutOfRange { axis: 1, rank: 1 })
    ));

    let err_neg = fandhe_ai_autodiff::topk_unique_ops::topk_with_options(
        &x,
        1,
        TopkOptions::default().with_dim(-2),
    )
    .unwrap_err();
    assert!(matches!(err_neg, AutodiffError::InvalidArgument(_)));
}

/// `sorted=true` は既存 `Var::topk` と forward・backward とも bit
/// 同一（委譲経路の固定）。
#[test]
fn topk_sorted_true_matches_existing_var_topk() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.6], &[1, 6]));
    let (out_opt, idx_opt) = fandhe_ai_autodiff::topk_unique_ops::topk_with_options(
        &x,
        3,
        TopkOptions::default().with_dim(1).with_largest(true),
    )
    .unwrap();
    let (out_plain, idx_plain) = x.topk(3, 1, true).unwrap();
    assert_eq!(
        dense_vec(&out_opt.to_tensor()),
        dense_vec(&out_plain.to_tensor())
    );
    assert_eq!(dense_vec_i32(&idx_opt), dense_vec_i32(&idx_plain));
}

/// `sorted=false` の出力は「`sorted=true`（既存 topk 相当）で選んだ
/// k 個を、`dim` 軸上の元添字の昇順に並べ替えた順」と一致する。
#[test]
fn topk_sorted_false_matches_index_ascending_reorder_of_sorted_true() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.6], &[1, 6]));
    let (sorted_out, sorted_idx) = x.topk(3, 1, true).unwrap();
    let mut expected: Vec<(i32, f32)> = dense_vec_i32(&sorted_idx)
        .into_iter()
        .zip(dense_vec(&sorted_out.to_tensor()))
        .collect();
    expected.sort_by_key(|&(i, _)| i);

    let (unsorted_out, unsorted_idx) = fandhe_ai_autodiff::topk_unique_ops::topk_with_options(
        &x,
        3,
        TopkOptions::default().with_dim(1).with_sorted(false),
    )
    .unwrap();
    let actual: Vec<(i32, f32)> = dense_vec_i32(&unsorted_idx)
        .into_iter()
        .zip(dense_vec(&unsorted_out.to_tensor()))
        .collect();
    assert_eq!(actual, expected);
    // index 昇順であることも直接確認する。
    let idx_vec = dense_vec_i32(&unsorted_idx);
    assert!(idx_vec.windows(2).all(|w| w[0] < w[1]));
}

/// `sorted=false` の勾配は `sorted=true` の勾配と bit 同一（`Op::Topk`
/// の VJP は scatter ベースで index 順序に依存しないため）。
#[test]
fn topk_sorted_false_gradient_matches_sorted_true() {
    let x0 = t(vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.6], &[1, 6]);

    let tape_sorted = Tape::new_with_ops(common::naive_ops());
    let xv_sorted = tape_sorted.var(&x0);
    let (out_sorted, _) = fandhe_ai_autodiff::topk_unique_ops::topk_with_options(
        &xv_sorted,
        3,
        TopkOptions::default().with_dim(1).with_sorted(true),
    )
    .unwrap();
    let loss_sorted = out_sorted.mul(&out_sorted).unwrap().sum(None).unwrap();
    let grads_sorted = tape_sorted.backward(&loss_sorted).unwrap();
    let dx_sorted = grads_sorted
        .get(&xv_sorted)
        .unwrap()
        .expect("x は loss に到達する");

    let tape_unsorted = Tape::new_with_ops(common::naive_ops());
    let xv_unsorted = tape_unsorted.var(&x0);
    let (out_unsorted, _) = fandhe_ai_autodiff::topk_unique_ops::topk_with_options(
        &xv_unsorted,
        3,
        TopkOptions::default().with_dim(1).with_sorted(false),
    )
    .unwrap();
    let loss_unsorted = out_unsorted.mul(&out_unsorted).unwrap().sum(None).unwrap();
    let grads_unsorted = tape_unsorted.backward(&loss_unsorted).unwrap();
    let dx_unsorted = grads_unsorted
        .get(&xv_unsorted)
        .unwrap()
        .expect("x は loss に到達する");

    assert_eq!(dense_vec(dx_sorted), dense_vec(dx_unsorted));
}

/// `k=0` は空出力で成功する（`sorted=false` 経路でも panic しない）。
#[test]
fn topk_sorted_false_k_zero_returns_empty() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]));
    let (out, idx) = fandhe_ai_autodiff::topk_unique_ops::topk_with_options(
        &x,
        0,
        TopkOptions::default().with_dim(1).with_sorted(false),
    )
    .unwrap();
    assert_eq!(out.to_tensor().shape(), &[1, 0]);
    assert_eq!(dense_vec_i32(&idx), Vec::<i32>::new());
}

// ---------------------------------------------------------------------
// unique_with_options / unique_consecutive
// ---------------------------------------------------------------------

/// `dim=None` の values は既存 `Var::unique` と bit 同一。
#[test]
fn unique_dim_none_values_match_existing_var_unique() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, 1.0, 2.0, 1.0, 3.0], &[5]));
    let plain = x.unique().unwrap();
    let out = unique_with_options(&x, UniqueOptions::default()).unwrap();
    assert_eq!(dense_vec(&out.values), dense_vec(&plain));
    assert!(out.inverse.is_none());
    assert!(out.counts.is_none());
}

/// `return_inverse`／`return_counts` を要求すると `inverse` で入力を
/// 再構成でき、`counts` の総和が要素数に一致する。
#[test]
fn unique_return_inverse_and_counts_reconstruct_and_sum() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x0 = t(vec![3.0, 1.0, 2.0, 1.0, 3.0], &[5]);
    let x = tape.var(&x0);
    let out = unique_with_options(
        &x,
        UniqueOptions::default()
            .with_return_inverse(true)
            .with_return_counts(true),
    )
    .unwrap();
    let values = dense_vec(&out.values);
    let inverse = dense_vec_i32(out.inverse.as_ref().unwrap());
    let counts = dense_vec_i32(out.counts.as_ref().unwrap());
    let x0_vec = dense_vec(&x0);
    for (i, &g) in inverse.iter().enumerate() {
        assert_eq!(values[g as usize], x0_vec[i]);
    }
    assert_eq!(counts.iter().sum::<i32>(), 5);
}

/// `unique_consecutive` は元順序のまま隣接要素のみ群化し、代表は各
/// 連続ランの先頭出現値になる。
#[test]
fn unique_consecutive_groups_adjacent_only_and_preserves_order() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 1.0, 2.0, 1.0, 1.0], &[5]));
    let out = unique_consecutive(
        &x,
        UniqueOptions::default()
            .with_return_inverse(true)
            .with_return_counts(true),
    )
    .unwrap();
    assert_eq!(dense_vec(&out.values), vec![1.0, 2.0, 1.0]);
    assert_eq!(
        dense_vec_i32(out.inverse.as_ref().unwrap()),
        vec![0, 0, 1, 2, 2]
    );
    assert_eq!(dense_vec_i32(out.counts.as_ref().unwrap()), vec![2, 1, 2]);
}

/// `dim` 指定（負値含む）でスライス単位の重複除去ができる。
#[test]
fn unique_with_dim_dedups_rows() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // shape [3, 2]: row0=[1,2] row1=[3,4] row2=[1,2]（row0 と row2 が重複）
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0], &[3, 2]));
    let out = unique_with_options(
        &x,
        UniqueOptions::default()
            .with_dim(0)
            .with_return_inverse(true),
    )
    .unwrap();
    assert_eq!(out.values.shape(), &[2, 2]);
    let inverse = dense_vec_i32(out.inverse.as_ref().unwrap());
    assert_eq!(inverse[0], inverse[2]);
    assert_ne!(inverse[0], inverse[1]);

    // 負 dim（-1 は末尾軸）でも同様に動作する。
    let out_neg = unique_with_options(&x, UniqueOptions::default().with_dim(-2)).unwrap();
    assert_eq!(out_neg.values.shape(), &[2, 2]);
}

/// unique 系呼び出し前後で tape ノード数が不変（非微分・detached の
/// 契約固定）。
#[test]
fn unique_does_not_record_tape_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, 1.0, 2.0, 1.0, 3.0], &[5]));
    let before = tape.len();
    let _ = unique_with_options(
        &x,
        UniqueOptions::default()
            .with_return_inverse(true)
            .with_return_counts(true),
    )
    .unwrap();
    let _ = unique_consecutive(&x, UniqueOptions::default()).unwrap();
    let after = tape.len();
    assert_eq!(before, after);
}

/// `dim` 範囲外は型付きエラーになる（unique 系も topk と同じ検査）。
#[test]
fn unique_dim_out_of_range_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let err = unique_with_options(&x, UniqueOptions::default().with_dim(1)).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::AxisOutOfRange { axis: 1, rank: 1 })
    ));
}

/// PR #2270 codex-review P0 是正の回帰テスト（`eval::unique_ext`
/// フォールバック経路。`common::naive_ops()` は `unique_ext` に
/// `Unsupported` を返すため必ずこの経路を通る）: `d` 自身は非 0 長軸
/// だが他軸が 0 長（`shape = [大軸長, 0]` 型・`slice_len == 0`）の
/// 場合、全スライスが等しく空であるため「1 群」に畳み込まれ、
/// `axis_len` に比例した行配列を構築せず `inverse`／`counts` が
/// 必要量だけ生成されることを確認する（codex 指摘）。
#[test]
fn unique_ext_dim_nonzero_axis_with_other_zero_axis_collapses_to_one_group() {
    let axis_len = 100_000usize;
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(Vec::new(), &[axis_len, 0]));
    let out = unique_with_options(
        &x,
        UniqueOptions::default()
            .with_dim(0)
            .with_return_inverse(true)
            .with_return_counts(true),
    )
    .unwrap();
    assert_eq!(out.values.shape(), &[1, 0]);
    let inverse = dense_vec_i32(out.inverse.as_ref().unwrap());
    assert_eq!(inverse.len(), axis_len);
    assert!(inverse.iter().all(|&g| g == 0));
    assert_eq!(
        dense_vec_i32(out.counts.as_ref().unwrap()),
        vec![axis_len as i32]
    );
}

/// PR #2270 codex-review Medium 是正の回帰テスト（`eval::unique_ext`
/// フォールバック経路）: `shape` の先頭が 0 長軸で、他軸が
/// `row_major_strides` の suffix 積で `usize` を溢れさせるほど巨大
/// （`[0, usize::MAX, 2]` 型。総積は 0 だが `usize::MAX * 2` の部分積は
/// overflow する）でも、`d` を 0 長軸自身に取れば早期 return で
/// strides 計算自体を回避でき panic しないことを確認する（Cursor
/// Bugbot 指摘）。
#[test]
fn unique_ext_dim_leading_zero_axis_with_overflow_prone_suffix_does_not_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(Vec::new(), &[0, usize::MAX, 2]));
    let out = unique_with_options(
        &x,
        UniqueOptions::default()
            .with_dim(0)
            .with_return_inverse(true)
            .with_return_counts(true),
    )
    .unwrap();
    assert_eq!(out.values.shape(), &[0, usize::MAX, 2]);
    assert_eq!(dense_vec_i32(out.inverse.as_ref().unwrap()), Vec::new());
    assert_eq!(dense_vec_i32(out.counts.as_ref().unwrap()), Vec::new());
}
