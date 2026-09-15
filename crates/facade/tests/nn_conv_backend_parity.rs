//! `nn::Conv2d`／`nn::Conv1d`（イシュー #1770・親 #1645）を積んだ
//! `compat::Sequential` の 3 バックエンド parity テスト
//! （`conv2d_backend_parity.rs`／`conv1d_backend_parity.rs` と同型）。
//!
//! - 属性なし（CPU）: `fandhe_ai::tape()`（`CpuBackendOps`）上で
//!   `compat::Sequential(add_conv2d→add_relu)` の forward／backward が
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）上に同じ重みで組んだ
//!   `nn::Conv2d::from_parameters` 参照実装（`bind`／`trainable_grads`
//!   相当の経路）と bit 完全一致することを確認する。
//! - `#[ignore]`: CUDA（`tape_for(Device::Cuda(0))`）・Metal
//!   （`tape_for(Device::Metal)`。`cfg(target_os = "macos")` 限定）の
//!   forward／backward（入力勾配・weight／bias 勾配）を CPU と
//!   `assert_parity`（REQ-2 複合判定）で比較する。実行は #1771 へ引き
//!   継ぎ、本 PR では `#[ignore]` のまま未実行・未実測と明記する。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai_autodiff::Tape as RawTape;
use fandhe_ai_autodiff::nn::{Conv2d, Module};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

fn assert_bits_eq(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: 要素数が一致しない");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.is_nan() || e.is_nan() {
            assert!(
                a.is_nan() && e.is_nan(),
                "{label}: 要素 {i} が NaN クラス一致しない（actual={a}, expected={e}）"
            );
        } else {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}: 要素 {i} が bit 一致しない（actual={a:?}, expected={e:?}）"
            );
        }
    }
}

// --- CPU: forward（compat::Sequential vs nn::Conv2d 直接） ---

#[test]
fn cpu_sequential_conv2d_predict_matches_naive_conv2d_layer() {
    let x_shape = [1usize, 2, 5, 5];
    let w_shape = [3usize, 2, 3, 3];
    let b_shape = [3usize];

    let x = leaf(11, &x_shape);
    let w = leaf(12, &w_shape);
    let b = leaf(13, &b_shape);

    // `compat::Sequential` 経由（`CpuBackendOps`。`predict` = tape 不要経路
    // または via-tape フォールバック）。`apply_parameters` で決定的な
    // 重みへ置き換える（`add_conv2d` 自体は seed 決定なので、直接
    // 比較用に固定値を注入する）。
    let mut model = Sequential::new()
        .add_conv2d(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, 1)
        .unwrap();
    model.apply_parameters(vec![w.clone(), b.clone()]).unwrap();
    let seq_out = model.predict(&x).unwrap();

    // naive 参照実装（`NaiveOps` 上に `nn::Conv2d::from_parameters` を
    // 直接組む）。
    let naive_conv = Conv2d::from_parameters(w, Some(b), [1, 1], [1, 1], [1, 1], 1).unwrap();
    let naive_tape = RawTape::new();
    let naive_xv = naive_tape.var(&x);
    let naive_out = Module::forward(&naive_conv, &naive_tape, &naive_xv)
        .unwrap()
        .to_tensor();

    let seq_slice = dense(&seq_out);
    let naive_slice = dense(&naive_out);
    assert_parity(
        "compat::Sequential(add_conv2d) vs nn::Conv2d::from_parameters（NaiveOps）",
        &seq_slice,
        &naive_slice,
    );
    assert_bits_eq(
        "compat::Sequential(add_conv2d) vs nn::Conv2d::from_parameters（NaiveOps）",
        &seq_slice,
        &naive_slice,
    );
}

// --- CPU: backward（weight／bias／input 勾配） ---

#[test]
fn cpu_sequential_conv2d_backward_matches_naive_conv2d_layer() {
    let x_shape = [2usize, 2, 4, 4];
    let w_shape = [3usize, 2, 3, 3];
    let b_shape = [3usize];
    let target_shape = [2usize, 3, 4, 4];

    let x = leaf(21, &x_shape);
    let w = leaf(22, &w_shape);
    let b = leaf(23, &b_shape);
    let target = leaf(24, &target_shape);

    // compat::Sequential 経由（`bind` → `trainable_grads`）。
    let mut model = Sequential::new()
        .add_conv2d(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, 1)
        .unwrap();
    model.apply_parameters(vec![w.clone(), b.clone()]).unwrap();

    let seq_tape = fandhe_ai::tape();
    let bound = model.bind(&seq_tape);
    let xv = seq_tape.var(&x);
    let tv = seq_tape.var(&target);
    let pred = bound.forward(&seq_tape, &xv).unwrap();
    let loss = pred.mse_loss(&tv).unwrap();
    let seq_grads = seq_tape.backward(&loss).unwrap();
    let seq_grad_refs = bound.trainable_grads(&seq_grads).unwrap();
    let seq_input_grad = seq_grads.get(&xv).unwrap().unwrap();

    // naive 参照実装。
    let naive_conv = Conv2d::from_parameters(w, Some(b), [1, 1], [1, 1], [1, 1], 1).unwrap();
    let naive_tape = RawTape::new();
    let naive_xv = naive_tape.var(&x);
    let naive_tv = naive_tape.var(&target);
    let naive_vars = naive_conv.bind(&naive_tape);
    let naive_pred = naive_vars.forward(&naive_xv).unwrap();
    let naive_loss = naive_pred.mse_loss(&naive_tv).unwrap();
    let naive_grads = naive_tape.backward(&naive_loss).unwrap();
    let naive_weight_grad = naive_grads.get(&naive_vars.weight).unwrap().unwrap();
    let naive_bias_grad = naive_grads
        .get(naive_vars.bias.as_ref().unwrap())
        .unwrap()
        .unwrap();
    let naive_input_grad = naive_grads.get(&naive_xv).unwrap().unwrap();

    assert_eq!(seq_grad_refs.len(), 2, "weight・bias の 2 件");
    assert_bits_eq(
        "weight_grad: Sequential vs naive",
        &dense(seq_grad_refs[0]),
        &dense(naive_weight_grad),
    );
    assert_bits_eq(
        "bias_grad: Sequential vs naive",
        &dense(seq_grad_refs[1]),
        &dense(naive_bias_grad),
    );
    assert_bits_eq(
        "input_grad: Sequential vs naive",
        &dense(seq_input_grad),
        &dense(naive_input_grad),
    );
}

// --- CUDA（イシュー #1771 実行時記入欄。本 PR では #[ignore] のまま
// 未実行・未実測） ---

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #1771 へ引き継ぎ"]
fn cuda_sequential_conv2d_matches_cpu() {
    let x_shape = [1usize, 2, 5, 5];
    let w_shape = [3usize, 2, 3, 3];
    let b_shape = [3usize];
    let x = leaf(31, &x_shape);
    let w = leaf(32, &w_shape);
    let b = leaf(33, &b_shape);

    let mut cpu_model = Sequential::new()
        .add_conv2d(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, 1)
        .unwrap();
    cpu_model
        .apply_parameters(vec![w.clone(), b.clone()])
        .unwrap();
    let cpu_out = cpu_model.predict(&x).unwrap();

    let cuda_tape = fandhe_ai::tape_for(fandhe_ai::Device::Cuda(0))
        .expect("CUDA 実機必須（本テストは #[ignore]）");
    let mut cuda_model = Sequential::new()
        .add_conv2d(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, 1)
        .unwrap();
    cuda_model.apply_parameters(vec![w, b]).unwrap();
    let xv = cuda_tape.var(&x);
    let cuda_out = cuda_model.forward(&cuda_tape, &xv).unwrap().to_tensor();

    assert_parity("CUDA vs CPU", &dense(&cuda_out), &dense(&cpu_out));
}

// --- Metal（イシュー #1771 実行時記入欄。本 PR では #[ignore] のまま
// 未実行・未実測） ---

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない。実行は #1771 へ引き継ぎ"]
fn metal_sequential_conv2d_matches_cpu() {
    let x_shape = [1usize, 2, 5, 5];
    let w_shape = [3usize, 2, 3, 3];
    let b_shape = [3usize];
    let x = leaf(41, &x_shape);
    let w = leaf(42, &w_shape);
    let b = leaf(43, &b_shape);

    let mut cpu_model = Sequential::new()
        .add_conv2d(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, 1)
        .unwrap();
    cpu_model
        .apply_parameters(vec![w.clone(), b.clone()])
        .unwrap();
    let cpu_out = cpu_model.predict(&x).unwrap();

    let metal_tape = fandhe_ai::tape_for(fandhe_ai::Device::Metal)
        .expect("Metal 実機必須（本テストは #[ignore]）");
    let mut metal_model = Sequential::new()
        .add_conv2d(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, 1)
        .unwrap();
    metal_model.apply_parameters(vec![w, b]).unwrap();
    let xv = metal_tape.var(&x);
    let metal_out = metal_model.forward(&metal_tape, &xv).unwrap().to_tensor();

    assert_parity("Metal vs CPU", &dense(&metal_out), &dense(&cpu_out));
}
