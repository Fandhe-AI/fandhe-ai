//! `Var::cumsum`／`cumprod`（`fandhe_ai_autodiff::var`。イシュー #1731）
//! の forward／backward を facade 経由（`fandhe_ai::tape_for`）で検証
//! する（`reduce_backend_parity.rs::cpu_sum_all_forward_and_backward_
//! match_analytic_values` と同型の (a) CPU 経路のみ）。
//!
//! `Var::cumsum`／`cumprod` は `facade` へ新規 `pub use`／`pub fn` を
//! 追加せず、既存の `Var` 再エクスポート経由でそのまま到達可能である
//! ことの機械的裏付けを兼ねる（`docs/compat-api-scope.md` §1.3）。
//! GPU（CUDA／Metal）は本イシュー時点で `Unsupported` フォール
//! バックのみのため対象外（`#[ignore]` 実機テストは追加しない）。

use fandhe_ai::{Device, Tensor, tape_for};

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().unwrap().to_vec()
}

/// CPU tape 上で `Var::cumsum` の forward／backward が既知の解析値
/// （`out[i] = Σ_{j<=i} x[j]`・`dx[i] = Σ_{j>=i} g[j]`）と一致する
/// ことを確認する。
#[test]
fn cpu_cumsum_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    let a = tape.var(&tensor(vec![1.0, 2.0, 3.0, 4.0], &[4]));

    let out = a.cumsum(0).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), vec![1.0, 3.0, 6.0, 10.0]);

    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    // loss = Σ_i cumsum(x)[i] = Σ_i Σ_{j<=i} x[j] なので
    // dx[j] = (n - j)（`x[j]` が寄与する出力位置の個数）。
    assert_eq!(dense_vec(da), vec![4.0, 3.0, 2.0, 1.0]);
}

/// CPU tape 上で `Var::cumprod` の forward／backward が既知の解析値
/// と一致することを確認する（`loss = sum(cumprod(x))` に対する
/// 解析的勾配を手計算し突合）。
#[test]
fn cpu_cumprod_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    let a = tape.var(&tensor(vec![2.0, 3.0, 0.5], &[3]));

    let out = a.cumprod(0).unwrap();
    // cumprod = [2, 6, 3]
    assert_eq!(dense_vec(&out.to_tensor()), vec![2.0, 6.0, 3.0]);

    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");

    // loss = out0 + out1 + out2 = x0 + x0*x1 + x0*x1*x2
    // dx0 = 1 + x1 + x1*x2 = 1 + 3 + 1.5 = 5.5
    // dx1 = x0 + x0*x2 = 2 + 1 = 3
    // dx2 = x0*x1 = 6
    assert_eq!(dense_vec(da), vec![5.5, 3.0, 6.0]);
}
