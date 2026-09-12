//! `Var::sum`／`Var::max`（`fandhe_ai_autodiff::var`）の forward／backward
//! を facade 横断で検証する（イシュー #1584・親イシュー #1571）。
//!
//! 属性なしのテストは CPU（`fandhe_ai::tape_for(Device::Cpu)`）のみを
//! 対象とし、既知の解析値との一致を確認する（CI で常時実行）。
//! `#[ignore]` テストは `tape_for(Device::Cuda(0))` の forward／backward
//! を CPU tape と REQ-2 統一複合判定（[`fandhe_ai_backend_cpu::
//! assert_parity`]）で突き合わせる（実機必須。Metal は本イシュー時点で
//! `sum`／`max` 未実装のため対象外）。
//!
//! ```sh
//! cargo test -p fandhe-ai --release --test reduce_backend_parity -- --ignored --nocapture
//! ```

use fandhe_ai::{Device, Tensor, tape_for};

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().unwrap().to_vec()
}

/// (a) CPU tape 上で `Var::sum(None)`（全軸）の forward／backward が
/// 既知の解析値と一致することを確認する（`Σ` の勾配は全要素 1）。
#[test]
fn cpu_sum_all_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    let a = tape.var(&tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let loss = a.sum(None).unwrap();
    assert_eq!(dense_vec(&loss.to_tensor()), vec![21.0]);

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    assert_eq!(dense_vec(da), vec![1.0; 6]);
}

/// (b) CPU tape 上で `Var::sum(Some(axis))`（単一軸）の forward／backward
/// が既知の解析値と一致することを確認する。非スカラー `loss` は
/// 「暗黙の総和射影」（`backward.rs::non_scalar_loss_seed_is_implicit_
/// sum_projection` と同じ契約）でシード全要素 1 として逆伝播される
/// ため、`Σ_axis` の勾配もやはり全要素 1 になる。
#[test]
fn cpu_sum_axis_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    let a = tape.var(&tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let loss = a.sum(Some(0)).unwrap();
    assert_eq!(dense_vec(&loss.to_tensor()), vec![5.0, 7.0, 9.0]);

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    assert_eq!(dense_vec(da), vec![1.0; 6]);
}

/// (c) CPU tape 上で `Var::max(None)`（全軸）の forward／backward が
/// 既知の解析値と一致することを確認する（勾配は argmax 位置のみ 1、
/// それ以外は 0。`grad.rs::max_vjp` の先勝ち決定的規約）。
#[test]
fn cpu_max_all_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    // 最大値 6.0 は唯一（index 5）。
    let a = tape.var(&tensor(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]));

    let loss = a.max(None).unwrap();
    assert_eq!(dense_vec(&loss.to_tensor()), vec![6.0]);

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    assert_eq!(dense_vec(da), vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
}

/// (d) CPU tape 上で `Var::max(Some(axis))`（単一軸）の forward／backward
/// が既知の解析値と一致することを確認する。
#[test]
fn cpu_max_axis_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    // shape [2, 3]、axis=0 の各列で最大値の行を確認する。
    // col0: max(1,4)=4 (row1) col1: max(5,2)=5 (row0) col2: max(3,6)=6 (row1)
    let a = tape.var(&tensor(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]));

    let loss = a.max(Some(0)).unwrap();
    assert_eq!(dense_vec(&loss.to_tensor()), vec![4.0, 5.0, 6.0]);

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    // row-major [2,3]: [row0_col0, row0_col1, row0_col2, row1_col0, row1_col1, row1_col2]
    assert_eq!(dense_vec(da), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
}

/// (e) `tape_for(Device::Cuda(0))` の `Var::sum`／`Var::max`（全軸・単一
/// 軸）forward／backward が CPU tape と REQ-2 統一複合判定で一致する
/// ことを確認する（実機必須）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_sum_and_max_forward_and_backward_match_cpu_tape_on_real_device() {
    let data = {
        // 決定的疑似乱数（Xorshift64Star。U[-0.5, 0.5)）。強い相殺を
        // 起こさない系列で `assert_parity`（REQ-2）の前提を満たす。
        let mut state = 0x1234_5678_9abc_def0u64;
        (0..24)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
            })
            .collect::<Vec<f32>>()
    };
    let shape = [4usize, 6];

    for dim in [None, Some(0usize), Some(1usize)] {
        let cpu_tape = tape_for(Device::Cpu).unwrap();
        let cpu_a = cpu_tape.var(&tensor(data.clone(), &shape));
        let cuda_tape = tape_for(Device::Cuda(0))
            .expect("CUDA device 0 must be available on ignored test runner");
        let cuda_a = cuda_tape.var(&tensor(data.clone(), &shape));

        // sum
        let cpu_sum = cpu_a.sum(dim).unwrap();
        let cuda_sum = cuda_a.sum(dim).unwrap();
        assert_parity_tensors(
            &cuda_sum.to_tensor(),
            &cpu_sum.to_tensor(),
            &format!("sum forward: dim={dim:?}"),
        );
        let cpu_grads = cpu_tape.backward(&cpu_sum).unwrap();
        let cuda_grads = cuda_tape.backward(&cuda_sum).unwrap();
        let cpu_da = cpu_grads
            .get(&cpu_a)
            .unwrap()
            .expect("a は loss に到達する");
        let cuda_da = cuda_grads
            .get(&cuda_a)
            .unwrap()
            .expect("a は loss に到達する");
        assert_parity_tensors(cuda_da, cpu_da, &format!("sum backward: dim={dim:?}"));

        // max
        let cpu_tape = tape_for(Device::Cpu).unwrap();
        let cpu_a = cpu_tape.var(&tensor(data.clone(), &shape));
        let cuda_tape = tape_for(Device::Cuda(0))
            .expect("CUDA device 0 must be available on ignored test runner");
        let cuda_a = cuda_tape.var(&tensor(data.clone(), &shape));

        let cpu_max = cpu_a.max(dim).unwrap();
        let cuda_max = cuda_a.max(dim).unwrap();
        assert_parity_tensors(
            &cuda_max.to_tensor(),
            &cpu_max.to_tensor(),
            &format!("max forward: dim={dim:?}"),
        );
        let cpu_grads = cpu_tape.backward(&cpu_max).unwrap();
        let cuda_grads = cuda_tape.backward(&cuda_max).unwrap();
        let cpu_da = cpu_grads
            .get(&cpu_a)
            .unwrap()
            .expect("a は loss に到達する");
        let cuda_da = cuda_grads
            .get(&cuda_a)
            .unwrap()
            .expect("a は loss に到達する");
        assert_parity_tensors(cuda_da, cpu_da, &format!("max backward: dim={dim:?}"));
    }
}

/// テンソル同士の統一複合判定（`linear_forward_device_real_device.rs`
/// と同じ方式。REQ-2 の唯一の実体である
/// [`fandhe_ai_backend_cpu::assert_parity`] へ委譲する）。
fn assert_parity_tensors(actual: &Tensor<f32>, expected: &Tensor<f32>, ctx: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{ctx}: shape mismatch");
    let a = actual.contiguous();
    let e = expected.contiguous();
    fandhe_ai_backend_cpu::assert_parity(ctx, a.as_slice().unwrap(), e.as_slice().unwrap());
}
