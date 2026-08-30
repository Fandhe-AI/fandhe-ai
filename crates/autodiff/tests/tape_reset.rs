//! `Tape::reset`/`leaf_count`/`leaf`（イシュー #1048）の統合テスト。
//!
//! 学習ループ・reuse GEMM（framework-compare の `run_gemm_reuse`）で
//! 同一 `Tape` を毎ステップ再利用する運用を想定し、以下を検証する:
//! - 葉プレフィックス（演算前に登録した葉）のみが `reset` を跨いで
//!   保持されること（step 内で登録した葉は蓄積しない）
//! - `reset` 後に同じ演算列を再実行した結果が新規 `Tape` の結果と
//!   bit-exact に一致すること（forward・backward いずれも）
//! - `reset` 前に得た `Gradients` を `reset` 後の `Var` から読むと
//!   fail-closed に `Err(TapeMismatch)` になること

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// テスト 2 用の共通演算列（matmul → relu → mse_loss → backward）。
/// **クロージャではなく独立関数にする理由**: クロージャは呼び出しごとに
/// 独立したライフタイムを推論できず（高階トレイト境界がない限り単一の
/// 具体ライフタイムへ単一化される）、同じクロージャ値を異なる `Tape`
/// 借用へ複数回適用すると 1 回目の借用が 2 回目の呼び出しまで生き
/// 続けてしまい `&mut Tape`（`reset`）と衝突する。独立関数はジェネリック
/// ライフタイム `'t` を呼び出しごとに再束縛できるためこの問題がない。
fn run_matmul_relu_mse<'t>(
    tape: &'t Tape,
    x: fandhe_ai_autodiff::Var<'t>,
    w: fandhe_ai_autodiff::Var<'t>,
    target: fandhe_ai_autodiff::Var<'t>,
) -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
    let y = x.matmul(&w).unwrap();
    let y = y.relu();
    let loss = y.mse_loss(&target).unwrap();
    let loss_value = loss.to_tensor();
    let grads = tape.backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().unwrap().clone();
    let gw = grads.get(&w).unwrap().unwrap().clone();
    (loss_value, gx, gw)
}

/// 1. 葉 2 個 → 演算列 → `reset()` で、ノード数・葉数・葉の値が
///    保持されることを検証する（受け入れ条件の直接検証）。
#[test]
fn reset_truncates_to_leaf_prefix_and_keeps_leaf_values() {
    let mut tape = Tape::new_with_ops(common::naive_ops());

    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let w = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    assert_eq!(tape.leaf_count(), 2); // まだ演算前なので全葉が候補

    let y = x.matmul(&w).unwrap();
    let _ = y.relu();
    assert!(tape.len() > 2);

    tape.reset();
    assert_eq!(tape.len(), 2);
    assert_eq!(tape.leaf_count(), 2);

    let x0 = tape.leaf(0).expect("leaf 0 must survive reset");
    let w0 = tape.leaf(1).expect("leaf 1 must survive reset");
    assert_eq!(x0.to_tensor().shape(), &[2, 2]);
    assert_eq!(
        x0.to_tensor().contiguous().as_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        w0.to_tensor().contiguous().as_slice().unwrap(),
        &[1.0, 0.0, 0.0, 1.0]
    );
    assert!(tape.leaf(2).is_none());
}

/// 2. reset 後に同じ演算列を再実行した forward/backward の結果が、
///    新規 `Tape` で計算した結果と bit-exact に一致することを検証する
///    （reuse と fresh の数値一致契約。#1048 受け入れ基準）。
#[test]
fn reset_then_replay_matches_fresh_tape_bit_exact() {
    let mut reuse_tape = Tape::new_with_ops(common::naive_ops());
    let x0 = reuse_tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let w0 = reuse_tape.var(&t(vec![0.5, -1.0, 2.0, 0.25], &[2, 2]));
    let target0 = reuse_tape.var(&t(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]));

    let first = run_matmul_relu_mse(&reuse_tape, x0, w0, target0);

    // 1 回目の演算ノードを reset で切り詰め、同じ葉を再取得して同じ
    // 演算列を再実行する（reuse GEMM／学習ループの想定運用）。
    reuse_tape.reset();
    let x1 = reuse_tape.leaf(0).unwrap();
    let w1 = reuse_tape.leaf(1).unwrap();
    let target1 = reuse_tape.leaf(2).unwrap();
    let second = run_matmul_relu_mse(&reuse_tape, x1, w1, target1);

    // fresh tape（毎回新規生成。従来運用）で同じ演算列を計算した結果。
    let fresh_tape = Tape::new_with_ops(common::naive_ops());
    let xf = fresh_tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let wf = fresh_tape.var(&t(vec![0.5, -1.0, 2.0, 0.25], &[2, 2]));
    let targetf = fresh_tape.var(&t(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]));
    let fresh = run_matmul_relu_mse(&fresh_tape, xf, wf, targetf);

    assert_eq!(
        first.0.contiguous().as_slice(),
        second.0.contiguous().as_slice()
    );
    assert_eq!(
        second.0.contiguous().as_slice(),
        fresh.0.contiguous().as_slice()
    );
    assert_eq!(
        second.1.contiguous().as_slice(),
        fresh.1.contiguous().as_slice()
    );
    assert_eq!(
        second.2.contiguous().as_slice(),
        fresh.2.contiguous().as_slice()
    );
}

/// 3. 演算前に登録した葉 2 個 + 演算後に登録した葉 1 個（`tape.var` を
///    matmul の後に呼ぶ、`Sequential::bind` の per-step パラメータ葉に
///    相当するケース）→ reset を 100 回繰り返しても `len()` が一定で
///    あることを検証する（step ごとの葉の非蓄積。#1048 発端の
///    バッファ蓄積問題に対する回帰テスト）。
#[test]
fn reset_does_not_accumulate_leaves_registered_after_first_op() {
    let mut tape = Tape::new_with_ops(common::naive_ops());
    let _x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let _w = tape.var(&t(vec![1.0, 1.0], &[2]));

    for _ in 0..100 {
        // 演算前の葉プレフィックス（葉 0・1）は毎回 `leaf()` で
        // 再取得する（`reset()` は `&mut self` を要求するため、前回
        // ループの `Var`〈`&'t Tape` を借用〉をこのスコープへ持ち越すと
        // 借用検査で弾かれる——これが「reset 前の stale `Var` は
        // コンパイル時に排除される」設計の直接的な帰結）。
        let x = tape.leaf(0).unwrap();
        let w = tape.leaf(1).unwrap();
        let _y = x.add(&w).unwrap().to_tensor();
        // step 内で追加登録する葉（例: 毎 step のバッチ入力相当）。
        let _extra_leaf = tape.var(&t(vec![9.0, 9.0], &[2]));
        tape.reset();
        assert_eq!(tape.len(), 2, "reset 後は演算前の葉数のみ残るはず");
        assert_eq!(tape.leaf_count(), 2);
    }
}

/// 4. 演算を一切行わずに reset すると全葉が保持される（`retained_leaf_len`
///    未固定経路）。空 tape の reset は no-op。
#[test]
fn reset_before_any_op_keeps_all_leaves_and_empty_reset_is_noop() {
    let mut tape = Tape::new_with_ops(common::naive_ops());
    assert_eq!(tape.leaf_count(), 0);
    tape.reset(); // 空 tape の reset は no-op
    assert_eq!(tape.len(), 0);

    let _a = tape.var(&t(vec![1.0], &[1]));
    let _b = tape.var(&t(vec![2.0], &[1]));
    assert_eq!(tape.leaf_count(), 2);
    tape.reset(); // まだ演算していないので全葉保持
    assert_eq!(tape.len(), 2);
    assert!(tape.leaf(0).is_some());
    assert!(tape.leaf(1).is_some());
}

/// 5. `backward` で得た `Gradients` を保持したまま `reset` すると、
///    reset 後の `Var`（新世代）に対する `get` が `Err(TapeMismatch)`
///    になる（stale 勾配の fail-closed 拒否）。reset 前の `get` は
///    従来どおり成功する。
#[test]
fn gradients_from_before_reset_are_rejected_after_reset() {
    let mut tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![2.0, 3.0], &[2]));
    let y = x.relu();
    let loss = y.mse_loss(&tape.var(&t(vec![0.0, 0.0], &[2]))).unwrap();
    let grads = tape.backward(&loss).unwrap();
    // reset 前は成功する。
    assert!(grads.get(&x).unwrap().is_some());

    tape.reset();
    let x_new = tape.leaf(0).unwrap();
    let err = grads.get(&x_new).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

/// 6. 遅延 elementwise（未実体化ノード）を含む状態で reset しても
///    panic せず、再度 forward・`to_tensor` が正しく計算できることを
///    検証する（`push_lazy` 経路の葉プレフィックス固定・truncate の
///    健全性）。
#[test]
fn reset_with_unmaterialized_lazy_chain_does_not_panic() {
    let mut tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0], &[2]));
    let b = tape.var(&t(vec![3.0, 4.0], &[2]));
    // add は elementwise（遅延評価対象）。value を読まずに reset する。
    let _sum = a.add(&b).unwrap();

    tape.reset();
    assert_eq!(tape.len(), 2);

    let a2 = tape.leaf(0).unwrap();
    let b2 = tape.leaf(1).unwrap();
    let sum2 = a2.add(&b2).unwrap();
    assert_eq!(
        sum2.to_tensor().contiguous().as_slice().unwrap(),
        &[4.0, 6.0]
    );
}

/// 7. view ノード（reshape）を含む状態で reset しても panic せず、
///    再度演算できることを検証する。
#[test]
fn reset_with_view_node_does_not_panic() {
    let mut tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let _reshaped = a.reshape(&[4]).unwrap();

    tape.reset();
    assert_eq!(tape.len(), 1);

    let a2 = tape.leaf(0).unwrap();
    let reshaped2 = a2.reshape(&[4]).unwrap();
    assert_eq!(reshaped2.to_tensor().shape(), &[4]);
}
