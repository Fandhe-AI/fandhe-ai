//! `fandhe_ai_autodiff::extremum_ops::{amax, amin}`（イシュー #2154・
//! 親 #2131「Phase 5」）の統合テスト。`common::NaiveOps` 経由で
//! forward の bit 同一性（`Var::max`／`min` との一致）・backward の
//! 均等分配 VJP・エラー系（`dim` 範囲外・空縮約・確保前検査）を検証
//! する（`cast.rs`／`reduce_ops.rs` 内部単体テストと同型の構成）。

mod common;

use fandhe_ai_autodiff::extremum_ops::{amax, amin};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

// --- forward: bit 同一性 ---

#[test]
fn amax_forward_matches_max_dim_none() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 5.0, 3.0, 5.0], &[4]));
    let a = amax(&x, None).unwrap();
    let b = x.max(None).unwrap();
    assert_eq!(
        a.to_tensor().host_slice()[0].to_bits(),
        b.to_tensor().host_slice()[0].to_bits()
    );
}

#[test]
fn amax_forward_matches_max_dim_some() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 5.0, 3.0, 2.0], &[2, 2]));
    let a = amax(&x, Some(1)).unwrap();
    let b = x.max(Some(1)).unwrap();
    assert_eq!(
        a.to_tensor().host_slice().into_owned(),
        b.to_tensor().host_slice().into_owned()
    );
}

#[test]
fn amin_forward_matches_min_dim_none() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -5.0, 3.0, -5.0], &[4]));
    let a = amin(&x, None).unwrap();
    let b = x.min(None).unwrap();
    assert_eq!(
        a.to_tensor().host_slice()[0].to_bits(),
        b.to_tensor().host_slice()[0].to_bits()
    );
}

#[test]
fn amin_forward_matches_min_dim_some() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -5.0, 3.0, -2.0], &[2, 2]));
    let a = amin(&x, Some(1)).unwrap();
    let b = x.min(Some(1)).unwrap();
    assert_eq!(
        a.to_tensor().host_slice().into_owned(),
        b.to_tensor().host_slice().into_owned()
    );
}

// --- backward: 均等分配の閉形式 ---

#[test]
fn amax_backward_distributes_evenly_across_ties() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 5.0, 3.0, 5.0], &[4]));
    let y = amax(&x, None).unwrap();
    let grads = tape.backward(&y).unwrap();
    let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
    assert_eq!(dx, vec![0.0, 0.5, 0.0, 0.5]);
}

#[test]
fn amin_backward_distributes_evenly_across_ties() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -5.0, 3.0, -5.0], &[4]));
    let y = amin(&x, None).unwrap();
    let grads = tape.backward(&y).unwrap();
    let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
    assert_eq!(dx, vec![0.0, 0.5, 0.0, 0.5]);
}

#[test]
fn amax_backward_dim_axis_per_lane_split() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // [[1,5],[5,5]] -> amax(dim=1) = [5, 5]（各行のタイ数は 1・2）
    let x = tape.var(&t(vec![1.0, 5.0, 5.0, 5.0], &[2, 2]));
    let y = amax(&x, Some(1)).unwrap();
    let grads = tape.backward(&y).unwrap();
    let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
    assert_eq!(dx, vec![0.0, 1.0, 0.5, 0.5]);
}

/// 既存 `Var::max`（先勝ち）の勾配が `amax`（均等分配）の追加によって
/// 変わっていないことを確かめる回帰。
#[test]
fn max_backward_still_first_match_after_amax_added() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 5.0, 3.0, 5.0], &[4]));
    let y = x.max(None).unwrap();
    let grads = tape.backward(&y).unwrap();
    let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
    assert_eq!(dx, vec![0.0, 1.0, 0.0, 0.0]);
}

// --- エラー系 ---

#[test]
fn amax_out_of_range_dim_is_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    assert!(amax(&x, Some(5)).is_err());
}

#[test]
fn amin_out_of_range_dim_is_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    assert!(amin(&x, Some(5)).is_err());
}

#[test]
fn amin_empty_reduction_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![], &[0]));
    assert!(matches!(
        amin(&x, None),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

/// `amax` の空縮約は `max` と同じエラーが伝播する（新しい規約を
/// 作らない。モジュール doc 参照）。
#[test]
fn amax_empty_reduction_matches_max_error_shape() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![], &[0]));
    let amax_err = amax(&x, None);
    let max_err = x.max(None);
    assert_eq!(amax_err.is_err(), max_err.is_err());
}
