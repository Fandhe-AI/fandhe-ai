//! `Var::var`／`std`／`norm_l1`／`norm_l2`（`fandhe_ai_autodiff::var`。
//! イシュー #1723）の forward／backward を facade 横断で検証する
//! （`reduce_backend_parity.rs`〈イシュー #1584〉と同型）。
//!
//! 属性なしのテストは CPU（`fandhe_ai::tape_for(Device::Cpu)`）のみを
//! 対象とし、既知の解析値との一致を確認する（CI で常時実行）。
//! `#[ignore]` テストは CUDA／Metal の forward／backward を CPU tape と
//! REQ-2 統一複合判定（[`fandhe_ai_backend_cpu::assert_parity`]）で
//! 突き合わせる（実機必須）。`var`／`vector_norm` は
//! `BackendOps` 既定 `Unsupported` のため、CUDA／Metal はいずれも
//! ホストフォールバック（`eval::var_along`／`vector_norm_along`）経由で
//! 到達する（`matrix_norm` 等と同じ構図。CPU との数値差は生じない想定
//! だが、実機での動作確認自体に価値があるため対象に含める）。
//!
//! ```sh
//! cargo test -p fandhe-ai --release --test var_norm_backend_parity -- --ignored --nocapture
//! ```

use fandhe_ai::{Device, Tensor, tape_for};

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().unwrap().to_vec()
}

/// (a) CPU tape 上で `Var::var(None, correction=1)`（不偏分散・全軸）の
/// forward が既知の解析値と一致することを確認する。
#[test]
fn cpu_var_all_forward_matches_analytic_value() {
    let tape = tape_for(Device::Cpu).unwrap();
    let a = tape.var(&tensor(vec![1.0, 2.0, 3.0, 4.0], &[4]));

    let loss = a.var(None, 1).unwrap();
    let v = dense_vec(&loss.to_tensor());
    assert_eq!(v.len(), 1);
    assert!((v[0] - 5.0 / 3.0).abs() < 1e-6);
}

/// (b) CPU tape 上で `Var::var(Some(axis), correction=0)`（母分散・
/// 単一軸）の forward／backward が既知の解析値と一致することを確認
/// する。
#[test]
fn cpu_var_axis_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    // shape [2, 3]: 行 0 = [1,2,3]・行 1 = 定数 [4,4,4]（分散 0）
    let a = tape.var(&tensor(vec![1.0, 2.0, 3.0, 4.0, 4.0, 4.0], &[2, 3]));

    let loss = a.var(Some(1), 0).unwrap();
    let v = dense_vec(&loss.to_tensor());
    assert!((v[0] - (2.0 / 3.0)).abs() < 1e-6);
    assert!((v[1] - 0.0).abs() < 1e-6);

    let scalar_loss = loss.sum(None).unwrap();
    let grads = tape.backward(&scalar_loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    // 行 1（定数）の勾配は理論上 0。
    let da = dense_vec(da);
    assert!((da[3]).abs() < 1e-5);
    assert!((da[4]).abs() < 1e-5);
    assert!((da[5]).abs() < 1e-5);
}

/// (c) CPU tape 上で `Var::std(None, correction=0)` の forward が既知の
/// 解析値と一致することを確認する。
#[test]
fn cpu_std_all_forward_matches_analytic_value() {
    let tape = tape_for(Device::Cpu).unwrap();
    let a = tape.var(&tensor(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], &[8]));

    let loss = a.std(None, 0).unwrap();
    let v = dense_vec(&loss.to_tensor());
    assert!((v[0] - 2.0).abs() < 1e-5);
}

/// (d) CPU tape 上で `Var::norm_l1(None)` の forward／backward が既知の
/// 解析値と一致することを確認する（`sign` 劣勾配。イシュー #1723）。
#[test]
fn cpu_norm_l1_all_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    let a = tape.var(&tensor(vec![-1.0, 2.0, -3.0, 4.0], &[4]));

    let loss = a.norm_l1(None).unwrap();
    assert_eq!(dense_vec(&loss.to_tensor()), vec![10.0]);

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    assert_eq!(dense_vec(da), vec![-1.0, 1.0, -1.0, 1.0]);
}

/// (e) CPU tape 上で `Var::norm_l2(None)`（3-4-5 の直角三角形）の
/// forward／backward が既知の解析値と一致することを確認する。
#[test]
fn cpu_norm_l2_all_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    let a = tape.var(&tensor(vec![3.0, 4.0], &[2]));

    let loss = a.norm_l2(None).unwrap();
    assert_eq!(dense_vec(&loss.to_tensor()), vec![5.0]);

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    // d/dx ‖x‖ = x / ‖x‖ = [3/5, 4/5]
    let da = dense_vec(da);
    assert!((da[0] - 0.6).abs() < 1e-6);
    assert!((da[1] - 0.8).abs() < 1e-6);
}

/// (f) `tape_for(Device::Cuda(0))` の `Var::var`／`std`／`norm_l1`／
/// `norm_l2`（全軸・単一軸）forward／backward が CPU tape と REQ-2
/// 統一複合判定で一致することを確認する（実機必須）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_var_and_norm_forward_and_backward_match_cpu_tape_on_real_device() {
    run_device_parity(
        Device::Cuda(0),
        "CUDA device 0 must be available on ignored test runner",
    );
}

/// (g) `tape_for(Device::Metal)` 版（(f) と同型。実機必須）。
/// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
/// （`tensor-core::device::Device` doc 参照）のため、本テスト関数も
/// 同じ cfg で囲む（`einsum_backend_parity.rs` と同方針）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon 等）必須"]
fn metal_var_and_norm_forward_and_backward_match_cpu_tape_on_real_device() {
    run_device_parity(
        Device::Metal,
        "Metal device must be available on ignored test runner",
    );
}

fn run_device_parity(device: Device, unavailable_msg: &str) {
    let data = {
        // 決定的疑似乱数（Xorshift64Star。U[-0.5, 0.5)）。強い相殺を
        // 起こさない系列で `assert_parity`（REQ-2）の前提を満たす
        // （`reduce_backend_parity.rs` と同じ系列生成方式）。
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
        // var（correction=1）
        let cpu_tape = tape_for(Device::Cpu).unwrap();
        let cpu_a = cpu_tape.var(&tensor(data.clone(), &shape));
        let dev_tape = tape_for(device).expect(unavailable_msg);
        let dev_a = dev_tape.var(&tensor(data.clone(), &shape));

        let cpu_var = cpu_a.var(dim, 1).unwrap();
        let dev_var = dev_a.var(dim, 1).unwrap();
        assert_parity_tensors(
            &dev_var.to_tensor(),
            &cpu_var.to_tensor(),
            &format!("var forward: dim={dim:?}"),
        );
        let cpu_scalar = cpu_var.sum(None).unwrap();
        let dev_scalar = dev_var.sum(None).unwrap();
        let cpu_grads = cpu_tape.backward(&cpu_scalar).unwrap();
        let dev_grads = dev_tape.backward(&dev_scalar).unwrap();
        let cpu_da = cpu_grads
            .get(&cpu_a)
            .unwrap()
            .expect("a は loss に到達する");
        let dev_da = dev_grads
            .get(&dev_a)
            .unwrap()
            .expect("a は loss に到達する");
        assert_parity_tensors(dev_da, cpu_da, &format!("var backward: dim={dim:?}"));

        // norm_l1／norm_l2
        for (label, cpu_norm, dev_norm) in [
            (
                "norm_l1",
                cpu_a.norm_l1(dim).unwrap(),
                dev_a.norm_l1(dim).unwrap(),
            ),
            (
                "norm_l2",
                cpu_a.norm_l2(dim).unwrap(),
                dev_a.norm_l2(dim).unwrap(),
            ),
        ] {
            assert_parity_tensors(
                &dev_norm.to_tensor(),
                &cpu_norm.to_tensor(),
                &format!("{label} forward: dim={dim:?}"),
            );
            let cpu_scalar = cpu_norm.sum(None).unwrap();
            let dev_scalar = dev_norm.sum(None).unwrap();
            let cpu_grads = cpu_tape.backward(&cpu_scalar).unwrap();
            let dev_grads = dev_tape.backward(&dev_scalar).unwrap();
            let cpu_da = cpu_grads
                .get(&cpu_a)
                .unwrap()
                .expect("a は loss に到達する");
            let dev_da = dev_grads
                .get(&dev_a)
                .unwrap()
                .expect("a は loss に到達する");
            assert_parity_tensors(dev_da, cpu_da, &format!("{label} backward: dim={dim:?}"));
        }
    }
}

/// テンソル同士の統一複合判定（`reduce_backend_parity.rs` と同じ方式。
/// REQ-2 の唯一の実体である [`fandhe_ai_backend_cpu::assert_parity`]
/// へ委譲する）。
fn assert_parity_tensors(actual: &Tensor<f32>, expected: &Tensor<f32>, ctx: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{ctx}: shape mismatch");
    let a = actual.contiguous();
    let e = expected.contiguous();
    fandhe_ai_backend_cpu::assert_parity(ctx, a.as_slice().unwrap(), e.as_slice().unwrap());
}
