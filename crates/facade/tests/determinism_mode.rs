//! facade の決定論モード公開（イシュー #2507・親 #2499）の利用例と
//! no-op 契約の end-to-end テスト。
//!
//! `crates/autodiff/tests/determinism_mode.rs` の facade 版で、
//! `fandhe_ai::{set_deterministic, is_deterministic}` と facade 公開の
//! `tape()`／`Var` だけで到達できることを示す。グローバル状態を扱うため
//! `#[test]` は 1 件に集約する（並列実行による競合の回避）。
//! 契約は `docs/autodiff-determinism-mode-design.md` §0・§3 を参照。

use fandhe_ai::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// matmul → relu → mse_loss の forward・backward を CPU Tape で実行し、
/// loss と `x` の勾配を bit 表現で返す。
fn run_graph() -> (u32, Vec<u32>) {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3]));
    let w = tape.var(&t(
        vec![0.5, -0.25, 0.125, 1.0, -1.0, 0.75, 0.25, -0.5, 1.5],
        &[3, 3],
    ));
    let target = tape.var(&t(vec![0.0; 6], &[2, 3]));
    let loss = x
        .matmul(&w)
        .expect("test fixture: shape 適合済み")
        .relu()
        .mse_loss(&target)
        .expect("test fixture: shape 適合済み");
    let loss_bits = loss
        .to_tensor()
        .get(&[])
        .expect("test fixture: mse_loss はスカラー")
        .to_bits();
    let grads = tape
        .backward(&loss)
        .expect("test fixture: backward は成功する");
    let dx = grads
        .get(&x)
        .expect("test fixture: 勾配取得は成功する")
        .expect("test fixture: x は loss へ到達する");
    let bits = dx
        .as_slice()
        .expect("test fixture: contiguous")
        .iter()
        .map(|v| v.to_bits())
        .collect();
    (loss_bits, bits)
}

#[test]
fn facade_set_deterministic_toggle_and_noop_contract() {
    assert!(!fandhe_ai::is_deterministic(), "既定値は false のはず");
    let off = run_graph();

    fandhe_ai::set_deterministic(true);
    fandhe_ai::set_deterministic(true);
    assert!(fandhe_ai::is_deterministic());
    let on = run_graph();
    assert_eq!(off, on, "ON/OFF で結果が bit 一致しない（no-op 契約違反）");

    fandhe_ai::set_deterministic(false);
    assert!(!fandhe_ai::is_deterministic());
}
