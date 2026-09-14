//! `Tape::var_no_grad`（追跡なし葉）・`Var::detach`（追跡切り離し）の
//! 契約テスト（イシュー #1748）。設計は
//! `docs/autodiff-nograd-leaf-dinput-skip-decision.md` §5「案 B」。
//!
//! 数値微分は本 issue の検証には使わない——中央差分は `detach` した
//! 複製も同時に摂動してしまうため、`detach` した値を定数とみなした
//! 解析解との bit 一致で検証する（`common::naive_ops()` は
//! `f32::mul_add` の FMA 契約で決定的）。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// 1. `loss = sum(x.detach() * x)` は `x.detach()` を定数とみなした
///    `loss = sum(c * x)`（`c` は `x` の値）と等価であり、`d loss/d x`
///    は `x` の値そのもの（`2x` ではない——`x` 自身を 2 回使う
///    `sum(x * x)` の勾配 `2x` と区別できることを確認する）。
#[test]
fn detach_operand_receives_no_gradient_and_contributes_as_constant() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![2.0, 3.0, -1.0, 4.0], &[2, 2]));
    let x_detached = x
        .detach()
        .expect("detach: 実体化済みノードなので失敗しない");
    let product = x_detached.mul(&x).expect("同 shape の要素積は失敗しない");
    let loss = product.sum(None).expect("全軸縮約は失敗しない");

    let grads = tape
        .backward(&loss)
        .expect("追跡対象の祖先を持つため成功する");
    let dx = grads
        .get(&x)
        .expect("x は requires_grad=true の葉")
        .expect("x は loss に到達する");
    // d(sum(c * x))/dx = c = x の値そのもの（2x ではない）。
    assert_eq!(dx.as_slice().unwrap(), x.to_tensor().as_slice().unwrap());
}

/// 2. `loss = sum(x.detach() * w)` で `w`（追跡対象）は `d loss/dw = x`
///    の値を bit 一致で受け取り、`x`（未到達。detach された側の
///    「元」ノードは product の入力ではないため厳密には無関係だが
///    ここでは検証しない）と `x_detached` 自身は
///    `Err(GradientTrackingDisabled)` を返すことを確認する。
#[test]
fn detach_leaf_is_untracked_while_tracked_operand_receives_gradient() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let w = tape.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));
    let x_detached = x
        .detach()
        .expect("detach: 実体化済みノードなので失敗しない");
    let product = x_detached.mul(&w).expect("同 shape の要素積は失敗しない");
    let loss = product.sum(None).expect("全軸縮約は失敗しない");

    let grads = tape.backward(&loss).expect("w が追跡対象のため成功する");

    // d(sum(c * w))/dw = c = x の値そのもの。
    let dw = grads
        .get(&w)
        .expect("w は requires_grad=true の葉")
        .expect("w は loss に到達する");
    assert_eq!(dw.as_slice().unwrap(), x.to_tensor().as_slice().unwrap());

    // `x_detached` は requires_grad=false の葉のため型付きエラー。
    let err = grads
        .get(&x_detached)
        .expect_err("detach された葉は構造的に勾配を持たない");
    assert!(matches!(err, AutodiffError::GradientTrackingDisabled));
}

/// 3. `Tape::var_no_grad` で登録した入力 `x` を
///    `matmul → add(bias) → relu → matmul → mse_loss` の 2 層 MLP 相当へ
///    通しても、weight／bias の勾配は `Tape::var` 経由（すべて
///    requires_grad=true）の同じ計算グラフと bit 完全一致する
///    （`var_no_grad` は forward の値には一切影響しない——`requires_grad`
///    の初期値のみが違う）。`get(x)` は型付きエラーを返す。
#[test]
fn var_no_grad_input_forward_matches_tracked_forward_but_receives_no_gradient() {
    let x_data = t(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]);
    let w1_data = t(vec![0.1, 0.2, -0.3, 0.4, 0.5, -0.1], &[2, 3]);
    let b1_data = t(vec![0.01, -0.02, 0.03], &[3]);
    let w2_data = t(vec![0.2, -0.1, 0.05, 0.3, -0.2, 0.15], &[3, 2]);
    let target_data = t(vec![0.0, 1.0, 1.0, 0.0], &[2, 2]);

    fn forward<'t>(
        x: fandhe_ai_autodiff::Var<'t>,
        w1: fandhe_ai_autodiff::Var<'t>,
        b1: fandhe_ai_autodiff::Var<'t>,
        w2: fandhe_ai_autodiff::Var<'t>,
        target: fandhe_ai_autodiff::Var<'t>,
    ) -> fandhe_ai_autodiff::Var<'t> {
        let h = x.matmul(&w1).unwrap();
        let h = h.add(&b1).unwrap();
        let h = h.relu();
        let out = h.matmul(&w2).unwrap();
        out.mse_loss(&target).unwrap()
    }

    // 経路 A: すべて requires_grad=true（`Tape::var`）。
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let x_a = tape_a.var(&x_data);
    let w1_a = tape_a.var(&w1_data);
    let b1_a = tape_a.var(&b1_data);
    let w2_a = tape_a.var(&w2_data);
    let target_a = tape_a.var(&target_data);
    let loss_a = forward(x_a, w1_a, b1_a, w2_a, target_a);
    let grads_a = tape_a.backward(&loss_a).expect("経路 A は成功する");

    // 経路 B: x のみ `var_no_grad`。
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let x_b = tape_b.var_no_grad(&x_data);
    let w1_b = tape_b.var(&w1_data);
    let b1_b = tape_b.var(&b1_data);
    let w2_b = tape_b.var(&w2_data);
    let target_b = tape_b.var(&target_data);
    let loss_b = forward(x_b, w1_b, b1_b, w2_b, target_b);

    // forward の値自体は requires_grad と無関係に bit 一致する。
    assert_eq!(
        loss_a.to_tensor().as_slice().unwrap(),
        loss_b.to_tensor().as_slice().unwrap(),
        "var_no_grad は forward の値に影響しない"
    );

    let grads_b = tape_b
        .backward(&loss_b)
        .expect("w1/b1/w2 が追跡対象のため成功する");

    for (name, a, b) in [
        ("w1", &w1_a, &w1_b),
        ("b1", &b1_a, &b1_b),
        ("w2", &w2_a, &w2_b),
    ] {
        let ga = grads_a
            .get(a)
            .unwrap_or_else(|e| panic!("経路 A の {name} 勾配取得に失敗: {e:?}"))
            .unwrap_or_else(|| panic!("経路 A の {name} は loss に到達する"));
        let gb = grads_b
            .get(b)
            .unwrap_or_else(|e| panic!("経路 B の {name} 勾配取得に失敗: {e:?}"))
            .unwrap_or_else(|| panic!("経路 B の {name} は loss に到達する"));
        assert_eq!(
            ga.as_slice().unwrap(),
            gb.as_slice().unwrap(),
            "{name} の勾配が var_no_grad 有無で bit 一致しない"
        );
    }

    let err = grads_b
        .get(&x_b)
        .expect_err("var_no_grad の葉は構造的に勾配を持たない");
    assert!(matches!(err, AutodiffError::GradientTrackingDisabled));
}

/// 4. 葉がすべて追跡なし（`var_no_grad` のみ）の loss は
///    `Tape::backward` が `Err(Backward)` を返す（逆伝播できる追跡対象の
///    祖先を持たないため）。
#[test]
fn backward_on_loss_with_only_no_grad_leaves_via_var_no_grad_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var_no_grad(&t(vec![1.0, 2.0], &[2]));
    let b = tape.var_no_grad(&t(vec![3.0, 4.0], &[2]));
    let product = a.mul(&b).expect("同 shape の要素積は失敗しない");
    let loss = product.sum(None).expect("全軸縮約は失敗しない");

    let result = tape.backward(&loss);
    assert!(
        matches!(result, Err(AutodiffError::Backward(_))),
        "追跡対象の祖先を持たない loss は Err(Backward) を返す契約: {result:?}"
    );
}

/// 4b. 葉がすべて追跡なし（`detach` のみ）の loss も同様に
///    `Err(Backward)`。
#[test]
fn backward_on_loss_with_only_detached_leaves_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0], &[2]));
    let b = tape.var(&t(vec![3.0, 4.0], &[2]));
    let a_detached = a.detach().expect("detach は失敗しない");
    let b_detached = b.detach().expect("detach は失敗しない");
    let product = a_detached
        .mul(&b_detached)
        .expect("同 shape の要素積は失敗しない");
    let loss = product.sum(None).expect("全軸縮約は失敗しない");

    let result = tape.backward(&loss);
    assert!(
        matches!(result, Err(AutodiffError::Backward(_))),
        "追跡対象の祖先を持たない loss は Err(Backward) を返す契約: {result:?}"
    );
}

/// 5. 追跡なし葉のみを `reshape`／`transpose`／`relu`／`exp` 経由で
///    組んだ loss も `Err(Backward)`（`push_view`／`push_lazy` の
///    前方伝播が正しく機能することの確認）。追跡ありの重みを 1 つ
///    混ぜると成功する。
#[test]
fn view_and_lazy_chains_over_no_grad_leaves_propagate_untracked_forward() {
    // 追跡なしのみ: reshape → transpose → relu → exp を経由した loss は
    // 逆伝播できる祖先を持たない。
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var_no_grad(&t(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]));
    let reshaped = a.reshape(&[4]).expect("reshape は失敗しない");
    let transposed = reshaped
        .reshape(&[2, 2])
        .expect("reshape は失敗しない")
        .transpose(0, 1)
        .expect("transpose は失敗しない");
    let activated = transposed.relu().exp();
    let loss = activated.sum(None).expect("全軸縮約は失敗しない");
    let result = tape.backward(&loss);
    assert!(
        matches!(result, Err(AutodiffError::Backward(_))),
        "view／lazy 連鎖を経由しても追跡なし葉のみなら Err(Backward): {result:?}"
    );

    // 追跡ありの重みを 1 つ混ぜると view／lazy 連鎖越しでも追跡対象に
    // 転じ、backward が成功する。
    let tape2 = Tape::new_with_ops(common::naive_ops());
    let a2 = tape2.var_no_grad(&t(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]));
    let w2 = tape2.var(&t(vec![0.5, 0.5, 0.5, 0.5], &[2, 2]));
    let reshaped2 = a2.reshape(&[4]).expect("reshape は失敗しない");
    let transposed2 = reshaped2
        .reshape(&[2, 2])
        .expect("reshape は失敗しない")
        .transpose(0, 1)
        .expect("transpose は失敗しない");
    let activated2 = transposed2.relu().exp();
    let mixed = activated2.mul(&w2).expect("同 shape の要素積は失敗しない");
    let loss2 = mixed.sum(None).expect("全軸縮約は失敗しない");
    let grads2 = tape2
        .backward(&loss2)
        .expect("w2 が追跡対象のため view／lazy 連鎖越しでも成功する");
    let gw2 = grads2
        .get(&w2)
        .expect("w2 は requires_grad=true の葉")
        .expect("w2 は loss に到達する");
    assert_eq!(gw2.shape(), &[2usize, 2]);
}

/// 6. `Tape::reset()` 後に `leaf(index)` で再取得した葉について、
///    追跡フラグ（requires_grad）が保持されていることを確認する。
#[test]
fn requires_grad_flag_survives_tape_reset() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let tracked = tape.var(&t(vec![1.0, 2.0], &[2]));
    let untracked = tape.var_no_grad(&t(vec![3.0, 4.0], &[2]));
    let tracked_idx = 0usize;
    let untracked_idx = 1usize;
    // 葉のみが記録されている段階でのインデックス確認（`Tape::reset`
    // doc の「葉プレフィックス」契約に依存しない直接確認）。
    assert_eq!(tape.leaf_count(), 2);

    // 演算を 1 つ記録してから reset する（reset は葉プレフィックスまで
    // 切り詰める。#1048）。
    let _sum = tracked
        .add(&untracked)
        .expect("同 shape の加算は失敗しない");
    let mut tape = tape;
    tape.reset();
    assert_eq!(
        tape.leaf_count(),
        2,
        "reset 後も葉プレフィックスの 2 件は残る"
    );

    let tracked_after = tape
        .leaf(tracked_idx)
        .expect("reset 後も葉プレフィックス内のインデックスは有効");
    let untracked_after = tape
        .leaf(untracked_idx)
        .expect("reset 後も葉プレフィックス内のインデックスは有効");

    // 追跡ありの葉は引き続き勾配を持ちうる（loss として直接使う）。
    let grads = tape
        .backward(&tracked_after)
        .expect("tracked_after は requires_grad=true");
    assert!(grads.get(&tracked_after).unwrap().is_some());

    // 追跡なしの葉は reset をまたいでも requires_grad=false のまま。
    let result = tape.backward(&untracked_after);
    assert!(
        matches!(result, Err(AutodiffError::Backward(_))),
        "reset 後も untracked_after の requires_grad=false は保持される: {result:?}"
    );
}

/// 7. 演算前（葉プレフィックス固定前）に `detach` した葉は
///    `leaf_count` に含まれ、`Tape::reset()` を跨いで残る。
#[test]
fn detach_before_any_op_is_included_in_leaf_prefix_and_survives_reset() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    // まだ非葉演算を 1 つも記録していない段階で detach する
    // （detach 自体は葉として登録されるため、葉プレフィックスに
    // 含まれる）。
    let _x_detached = x.detach().expect("detach は失敗しない");
    assert_eq!(tape.leaf_count(), 2, "x・x_detached の 2 件とも葉");

    let mut tape = tape;
    tape.reset();
    assert_eq!(
        tape.leaf_count(),
        2,
        "detach で追加した葉も reset を跨いで葉プレフィックスに残る"
    );
    let x_detached_after = tape
        .leaf(1)
        .expect("reset 後も detach で追加した葉のインデックスは有効");
    let result = tape.backward(&x_detached_after);
    assert!(
        matches!(result, Err(AutodiffError::Backward(_))),
        "reset を跨いでも detach で登録した葉の requires_grad=false は保持される: {result:?}"
    );
}

/// 8. checkpoint 解放済みノードの再計算に失敗（poison）した値を
///    `detach` すると `Err` を返す（stale／不正な値を新しい葉へ
///    複製しない fail-closed 方針。`checkpoint_review_1624.rs` の
///    poison セットアップと同じ手法を使う）。
#[test]
fn detach_of_poisoned_checkpoint_node_returns_err() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use fandhe_ai_tensor_core::{BackendError, BackendOps, ChecksumReadout, Device, GemmChecksum};

    struct FailOnSecondGemm {
        inner: Box<dyn BackendOps + Send>,
        calls: Arc<AtomicUsize>,
    }

    impl BackendOps for FailOnSecondGemm {
        fn device(&self) -> Device {
            self.inner.device()
        }

        fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if n == 2 {
                return Err(BackendError::Unsupported(
                    "test: 2 回目の gemm を意図的に失敗させる".into(),
                ));
            }
            self.inner.gemm(a, b)
        }

        fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            self.inner.add(a, b)
        }
        fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            self.inner.mul(a, b)
        }
        fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            self.inner.relu(a)
        }
        fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            self.inner.exp(a)
        }
        fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            self.inner.tanh(a)
        }
        fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            self.inner.sum(a, dim)
        }
        fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            self.inner.max(a, dim)
        }
        fn gemm_checksum(
            &self,
            a: &Tensor<f32>,
            b: &Tensor<f32>,
            readout: ChecksumReadout,
        ) -> Result<GemmChecksum, BackendError> {
            self.inner.gemm_checksum(a, b, readout)
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let ops = FailOnSecondGemm {
        inner: common::naive_ops(),
        calls: Arc::clone(&calls),
    };
    let tape = Tape::new_with_ops(Box::new(ops));

    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));

    // 1 回目の gemm（forward）は成功する。
    let m = a.matmul(&b).expect("forward の 1 回目の gemm は成功する");
    let out = m.relu();
    let _checkpointed_out = out
        .checkpoint_from(&[&a, &b])
        .expect("checkpoint_from 自体は forward を再実行しないため成功する");

    // `m` は checkpoint 区間内部ノードとして解放済み。`detach` が
    // 内部で `materialize_fallible` を呼び、2 回目の gemm（再計算）が
    // 失敗するため `Err` を返すことを確認する。
    let result = m.detach();
    assert!(
        result.is_err(),
        "poison 済み（再計算失敗）のノードを detach すると Err を返す契約: {result:?}"
    );
}

/// 9. `detach` の出力値が元 `Var` の `to_tensor()` と bit 一致・
///    shape 一致すること（値共有の確認）。
#[test]
fn detach_output_value_matches_source_bit_exact() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.5, -2.5, 3.25, 0.0], &[2, 2]));
    let y = x.relu().exp();
    let y_detached = y.detach().expect("detach は失敗しない");
    assert_eq!(y_detached.to_tensor().shape(), &[2usize, 2]);
    assert_eq!(
        y_detached.to_tensor().as_slice().unwrap(),
        y.to_tensor().as_slice().unwrap()
    );
}

/// 10. `Tape::var_no_grad` の `Var` は forward に通常どおり参加し、
///     `value()`／`to_tensor()` で読める。
#[test]
fn var_no_grad_participates_in_forward_and_is_readable() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var_no_grad(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let y = x.relu().exp();
    assert_eq!(y.to_tensor().shape(), &[2usize, 2]);
    let expected: Vec<f32> = vec![1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| v.max(0.0).exp())
        .collect();
    assert_eq!(y.to_tensor().as_slice().unwrap(), expected.as_slice());
}
