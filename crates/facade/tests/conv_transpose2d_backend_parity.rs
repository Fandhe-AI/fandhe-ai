//! `Var::conv_transpose2d`（col2im＋GEMM。イシュー #2067・設計
//! `docs/conv-ops-design.md` §15）の facade 到達経路（既存 `Var`
//! 再エクスポート経由。新規 `pub use`／`pub fn` は追加していない）の
//! 受け入れ条件対応テスト（`conv2d_backend_parity.rs` と同型）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`。`im2col`／`col2im`
//!   override 実装）と `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`。
//!   ホスト `eval::im2col`／`col2im` フォールバック経路）で forward／
//!   backward を bit 同一で突き合わせる。
//! - 同じく属性なし: 「実装計画」§0 で確認した CPU 側の bit 同一契約
//!   （`CpuBackendOps::gemm_batched` override は既定合成実装と bit
//!   同一・`gemm_fp32_strict` は override されず既定 `self.gemm` に
//!   委譲）に基づき、`CpuBackendOps` 上で `conv_transpose2d` forward
//!   と `Op::Conv2d` VJP の d_input・`Op::ConvTranspose2d` VJP の
//!   d_input と `conv2d` forward が bit 一致することを固定する
//!   （`grad.rs` unit test の CPU 版）。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の経路を CPU tape
//!   と `assert_parity`（REQ-2 複合判定）で比較する。実機未実測のまま
//!   出荷し記入欄を残す（`docs/perf/logs/conv-transpose2d-2067/
//!   README.md`）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

trait VarSource {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_>;
}

impl VarSource for fandhe_ai::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

impl VarSource for fandhe_ai_autodiff::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
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

fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

fn print_fold_bits(label: &str, data: &[f32]) {
    let mut acc: u64 = 0xcbf29ce484222325;
    for &v in data.iter() {
        let bits = v.to_bits() as u64;
        acc ^= bits;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    println!("{label}.fold_bits={acc:#018x}");
}

// --- forward（属性なし: CPU vs NaiveOps） ---

#[test]
fn cpu_conv_transpose2d_forward_matches_naive_reference() {
    let x_shape = [1usize, 2, 5, 5];
    let w_shape = [2usize, 3, 3, 3];
    let b_shape = [3usize];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(1, &x_shape));
    let w_cpu = cpu_tape.make_var(&leaf(2, &w_shape));
    let b_cpu = cpu_tape.make_var(&leaf(3, &b_shape));
    let out_cpu = x_cpu
        .conv_transpose2d(&w_cpu, Some(&b_cpu), [1, 1], [1, 1], [0, 0], [1, 1], 1)
        .expect("conv_transpose2d: 常に成功する形状")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(1, &x_shape));
    let w_naive = naive_tape.make_var(&leaf(2, &w_shape));
    let b_naive = naive_tape.make_var(&leaf(3, &b_shape));
    let out_naive = x_naive
        .conv_transpose2d(&w_naive, Some(&b_naive), [1, 1], [1, 1], [0, 0], [1, 1], 1)
        .expect("conv_transpose2d: 常に成功する形状")
        .to_tensor();

    let cpu_slice = contiguous_slice(&out_cpu);
    let naive_slice = contiguous_slice(&out_naive);
    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::im2col/col2im）vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
    assert_bits_eq(
        "fandhe_ai::tape()（CpuBackendOps::im2col/col2im）vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
}

// --- backward（属性なし: CPU vs NaiveOps） ---

#[test]
fn cpu_conv_transpose2d_backward_matches_naive_reference() {
    let x_shape = [1usize, 2, 5, 5];
    let w_shape = [2usize, 3, 3, 3];
    let b_shape = [3usize];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(6, &x_shape));
    let w_cpu = cpu_tape.make_var(&leaf(7, &w_shape));
    let b_cpu = cpu_tape.make_var(&leaf(8, &b_shape));
    let out_cpu = x_cpu
        .conv_transpose2d(&w_cpu, Some(&b_cpu), [1, 1], [1, 1], [0, 0], [1, 1], 1)
        .unwrap();
    let loss_cpu = out_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");
    let dw_cpu = grads_cpu.get(&w_cpu).unwrap().expect("到達する");
    let db_cpu = grads_cpu.get(&b_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(6, &x_shape));
    let w_naive = naive_tape.make_var(&leaf(7, &w_shape));
    let b_naive = naive_tape.make_var(&leaf(8, &b_shape));
    let out_naive = x_naive
        .conv_transpose2d(&w_naive, Some(&b_naive), [1, 1], [1, 1], [0, 0], [1, 1], 1)
        .unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");
    let dw_naive = grads_naive.get(&w_naive).unwrap().expect("到達する");
    let db_naive = grads_naive.get(&b_naive).unwrap().expect("到達する");

    assert_bits_eq(
        "conv_transpose2d backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dx_cpu),
        &contiguous_slice(dx_naive),
    );
    assert_bits_eq(
        "conv_transpose2d backward（dw）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dw_cpu),
        &contiguous_slice(dw_naive),
    );
    assert_bits_eq(
        "conv_transpose2d backward（db）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(db_cpu),
        &contiguous_slice(db_naive),
    );
}

// --- CPU 上での「転置畳み込みは通常畳み込みの随伴」構造 bit 一致 ---
// （実装計画「検証方法」§6.1 (a)/(b) の CPU バックエンド版。`grad.rs`
// unit test は `NaiveOps` 上でのみ検証しており、`CpuBackendOps` の
// override 経由での bit 一致は本テストが担う）。

#[test]
fn cpu_conv_transpose2d_forward_matches_conv2d_vjp_d_input() {
    // conv2d(input=[1,1,3,3], w=[1,1,2,2]) の出力は [1,1,2,2]。
    // upstream=x（[1,1,2,2]）としたときの d_input（shape [1,1,3,3]）が
    // conv_transpose2d(x, w, None) と一致するはず。
    let x = fandhe_ai_tensor_core::Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
    let w = fandhe_ai_tensor_core::Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]).unwrap();

    let tape = fandhe_ai::tape();
    let input_leaf =
        tape.make_var(&fandhe_ai_tensor_core::Tensor::new(vec![0.0; 9], &[1, 1, 3, 3]).unwrap());
    let weight_leaf = tape.make_var(&w);
    let conv_out = input_leaf
        .conv2d(&weight_leaf, None, [1, 1], [0, 0], [1, 1], 1)
        .unwrap();
    // upstream=x を conv_out に模した勾配として backward させるため、
    // `(conv_out * x).sum()` の d_input を使う（`grad.rs` unit test の
    // end-to-end 確認と同じ構成）。
    let x_var = tape.make_var(&x);
    let loss = conv_out.mul(&x_var).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let d_input = grads.get(&input_leaf).unwrap().expect("到達する");

    let ct_tape = fandhe_ai::tape();
    let x_ct = ct_tape.make_var(&x);
    let w_ct = ct_tape.make_var(&w);
    let ct_out = x_ct
        .conv_transpose2d(&w_ct, None, [1, 1], [0, 0], [0, 0], [1, 1], 1)
        .unwrap()
        .to_tensor();

    assert_bits_eq(
        "CpuBackendOps: conv_transpose2d(x,w,None) vs Op::Conv2d VJP d_input",
        &contiguous_slice(&ct_out),
        &contiguous_slice(d_input),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。REQ-2 複合判定） ---

fn conv_transpose2d_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[1, 2, 5, 5]));
    let w = tape.make_var(&leaf(2, &[2, 3, 3, 3]));
    let b = tape.make_var(&leaf(3, &[3]));
    x.conv_transpose2d(&w, Some(&b), [1, 1], [1, 1], [0, 0], [1, 1], 1)
        .expect("conv_transpose2d: 常に成功する形状")
        .to_tensor()
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_conv_transpose2d_forward_matches_cpu() {
    let metal_out = conv_transpose2d_forward_on(Device::Metal);
    let cpu_out = conv_transpose2d_forward_on(Device::Cpu);

    assert_parity(
        "conv_transpose2d forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
    print_fold_bits(
        "metal_conv_transpose2d_forward_matches_cpu[out]",
        &contiguous_slice(&metal_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_conv_transpose2d_forward_matches_cpu() {
    let cuda_out = conv_transpose2d_forward_on(Device::Cuda(0));
    let cpu_out = conv_transpose2d_forward_on(Device::Cpu);

    assert_parity(
        "conv_transpose2d forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
    print_fold_bits(
        "cuda_conv_transpose2d_forward_matches_cpu[out]",
        &contiguous_slice(&cuda_out),
    );
}

fn conv_transpose2d_backward_on(device: Device) -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[1, 2, 5, 5]));
    let w = tape.make_var(&leaf(2, &[2, 3, 3, 3]));
    let b = tape.make_var(&leaf(3, &[3]));
    let out = x
        .conv_transpose2d(&w, Some(&b), [1, 1], [1, 1], [0, 0], [1, 1], 1)
        .expect("conv_transpose2d: 常に成功する形状");
    let loss = out.sum(None).expect("sum: 常に成功する");
    let grads = tape.backward(&loss).expect("backward: 常に成功する形状");
    let dx = grads.get(&x).unwrap().expect("到達する").clone();
    let dw = grads.get(&w).unwrap().expect("到達する").clone();
    let db = grads.get(&b).unwrap().expect("到達する").clone();
    (dx, dw, db)
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_conv_transpose2d_backward_matches_cpu() {
    let (dx_cuda, dw_cuda, db_cuda) = conv_transpose2d_backward_on(Device::Cuda(0));
    let (dx_cpu, dw_cpu, db_cpu) = conv_transpose2d_backward_on(Device::Cpu);

    assert_parity(
        "conv_transpose2d backward（dx）: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&dx_cuda),
        &contiguous_slice(&dx_cpu),
    );
    assert_parity(
        "conv_transpose2d backward（dw）: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&dw_cuda),
        &contiguous_slice(&dw_cpu),
    );
    assert_parity(
        "conv_transpose2d backward（db）: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&db_cuda),
        &contiguous_slice(&db_cpu),
    );
    print_fold_bits(
        "cuda_conv_transpose2d_backward_matches_cpu[dx]",
        &contiguous_slice(&dx_cuda),
    );
    print_fold_bits(
        "cuda_conv_transpose2d_backward_matches_cpu[dw]",
        &contiguous_slice(&dw_cuda),
    );
    print_fold_bits(
        "cuda_conv_transpose2d_backward_matches_cpu[db]",
        &contiguous_slice(&db_cuda),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_conv_transpose2d_backward_matches_cpu() {
    let (dx_metal, dw_metal, db_metal) = conv_transpose2d_backward_on(Device::Metal);
    let (dx_cpu, dw_cpu, db_cpu) = conv_transpose2d_backward_on(Device::Cpu);

    assert_parity(
        "conv_transpose2d backward（dx）: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&dx_metal),
        &contiguous_slice(&dx_cpu),
    );
    assert_parity(
        "conv_transpose2d backward（dw）: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&dw_metal),
        &contiguous_slice(&dw_cpu),
    );
    assert_parity(
        "conv_transpose2d backward（db）: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&db_metal),
        &contiguous_slice(&db_cpu),
    );
    print_fold_bits(
        "metal_conv_transpose2d_backward_matches_cpu[dx]",
        &contiguous_slice(&dx_metal),
    );
    print_fold_bits(
        "metal_conv_transpose2d_backward_matches_cpu[dw]",
        &contiguous_slice(&dw_metal),
    );
    print_fold_bits(
        "metal_conv_transpose2d_backward_matches_cpu[db]",
        &contiguous_slice(&db_metal),
    );
}
