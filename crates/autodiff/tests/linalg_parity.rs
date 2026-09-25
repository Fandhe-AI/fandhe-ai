//! `linalg_ops`（イシュー #2150・`eigh`・`slogdet`・`pinv`・
//! `matrix_rank`・`lstsq`）の `Tape`/`Var` を経由する end-to-end
//! 統合テスト（`tests/linalg_backward.rs` と同じ方針）。
//!
//! `common::naive_ops()`（`BackendOps::linalg_*` を一切オーバーライド
//! しない）を使うため、必ず `eval::linalg`（ホスト参照実装）への
//! フォールバック経路を通る。`crates/backend-cpu` 側の実装到達性は
//! `backend-cpu/tests/linalg_parity.rs` が、CPU 実装と Naive フォール
//! バックの 3 バックエンド parity は
//! `crates/facade/tests/linalg_ops_backend_parity.rs` がそれぞれ別途
//! 担当する（本ファイルは NaiveOps 経路の forward・backward の到達性・
//! 数値契約のみを対象とする）。

mod common;

use fandhe_ai_autodiff::linalg_ops::{eigh, lstsq, matrix_rank, pinv, slogdet};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

// --- eigh ---

#[test]
fn eigh_forward_and_backward_reach_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![2.0, 1.0, 1.0, 2.0], &[2, 2]));
    let out = eigh(&x).unwrap();
    let evals = out.eigenvalues.to_tensor().host_slice().into_owned();
    assert!((evals[0] - 1.0).abs() < 1e-4);
    assert!((evals[1] - 3.0).abs() < 1e-4);
    let grads = tape.backward(&out.eigenvalues).unwrap();
    assert!(grads.get(&x).unwrap().is_some());
}

#[test]
fn eigh_multi_output_gradient_accumulates_to_single_input() {
    // `eigenvalues`・`eigenvectors` 両方が損失へ寄与する場合、
    // `Tape::backward` が `input` ノードへ両方の部分寄与を合算すること
    // （`svd_multi_output_gradient_accumulates_to_single_input` と同型）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![2.0, 0.3, 0.3, 3.0], &[2, 2]));
    let out = eigh(&x).unwrap();
    let sum_evals = out.eigenvalues.sum(None).unwrap();
    let sum_evecs = out.eigenvectors.sum(None).unwrap();
    let loss = sum_evals.add(&sum_evecs).unwrap();
    let grads = tape.backward(&loss).unwrap();
    assert!(grads.get(&x).unwrap().is_some());
}

#[test]
fn eigh_non_square_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    assert!(matches!(eigh(&x), Err(AutodiffError::InvalidArgument(_))));
}

// --- slogdet ---

#[test]
fn slogdet_forward_and_backward_reach_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]));
    let out = slogdet(&x).unwrap();
    assert_eq!(out.sign.to_tensor().host_slice()[0], 1.0);
    let grads = tape.backward(&out.logabsdet).unwrap();
    assert!(grads.get(&x).unwrap().is_some());
}

#[test]
fn slogdet_sign_gradient_is_exactly_zero_but_path_retained() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]));
    let out = slogdet(&x).unwrap();
    let grads = tape.backward(&out.sign).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    assert_eq!(dx.host_slice().into_owned(), vec![0.0, 0.0, 0.0, 0.0]);
}

// --- pinv ---

#[test]
fn pinv_forward_and_backward_reach_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![4.0, 7.0, 2.0, 6.0], &[2, 2]));
    let p = pinv(&x, None).unwrap();
    let grads = tape.backward(&p.sum(None).unwrap()).unwrap();
    assert!(grads.get(&x).unwrap().is_some());
}

#[test]
fn pinv_rank_deficient_does_not_error() {
    // `svd_vjp` の無条件縮退判定を回避できていることの回帰
    // （`docs/autodiff-linalg-ops-decision.md` §2「判断 2」）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]));
    let p = pinv(&x, None).unwrap();
    let grads = tape.backward(&p.sum(None).unwrap()).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    for v in dx.host_slice().iter() {
        assert!(v.is_finite(), "rank 落ち入力で pinv 勾配が非有限: {dx:?}");
    }
}

// --- matrix_rank ---

#[test]
fn matrix_rank_gradient_is_exactly_zero_but_path_retained() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let r = matrix_rank(&x, None).unwrap();
    assert_eq!(r.to_tensor().host_slice()[0], 2.0);
    let grads = tape.backward(&r).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    assert_eq!(dx.host_slice().into_owned(), vec![0.0, 0.0, 0.0, 0.0]);
}

// --- lstsq ---

#[test]
fn lstsq_forward_and_backward_reach_both_inputs() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]));
    let b = tape.var(&t(vec![1.0, 2.0, 4.0], &[3, 1]));
    let x = lstsq(&a, &b, None).unwrap();
    let grads = tape.backward(&x.sum(None).unwrap()).unwrap();
    assert!(grads.get(&a).unwrap().is_some());
    assert!(grads.get(&b).unwrap().is_some());
}

#[test]
fn lstsq_row_mismatch_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let b = tape.var(&t(vec![1.0, 2.0, 3.0], &[3, 1]));
    assert!(matches!(
        lstsq(&a, &b, None),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

// --- 確保前検査: broadcast された巨大 view で panic しないこと ---

#[test]
fn broadcast_huge_view_returns_error_not_panic() {
    // 形状の要素数積が `usize` の乗算オーバーフローを起こす規模
    // （実メモリ確保を試みる前に型付きエラーで拒否する契約）。
    // `Var::broadcast_to` 自身が構築時点で `ShapeError::
    // ElementCountOverflow` を返す——`linalg_ops` の
    // `ensure_alloc_fits_f32` に到達する前段で既に fail-closed に
    // 拒否される。いずれにせよ panic しないことを確認する。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0], &[1, 1]));
    assert!(x.broadcast_to(&[1usize << 40, 1usize << 40]).is_err());
}
