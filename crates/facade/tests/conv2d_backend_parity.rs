//! `Var::conv2d`（im2col＋GEMM。イシュー #1764・設計 `docs/conv-ops-
//! design.md`）の facade 到達経路（既存 `Var` 再エクスポート経由。
//! 新規 `pub use`／`pub fn` は追加していない）の受け入れ条件対応
//! テスト（`constant_pad_backend_parity.rs` と同型）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`。`im2col`／`col2im`
//!   override 実装）と `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`。
//!   ホスト `eval::im2col`／`col2im` フォールバック経路）で forward／
//!   backward を bit 同一で突き合わせる（im2col／col2im 自体は算術を
//!   含まないコピー・`f64` 逐次加算のいずれも bit 完全一致契約。設計
//!   doc §7）。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の経路を CPU tape
//!   と `assert_parity`（REQ-2 複合判定。GEMM 由来の差を許容）で比較
//!   する。CUDA（イシュー #1766）・Metal（イシュー #1768）とも
//!   `im2col`／`col2im` を override 済み（bit 完全一致契約は不変）だが
//!   `conv2d` 自身は override しないため「GPU im2col → GPU GEMM →
//!   GPU add」の段階的合成のうち GEMM 段のみが CPU 参照実装と
//!   異なりうる。im2col／col2im 自体は算術を含まないコピー・`f64`
//!   相当の逐次加算（Metal は binary64 ソフトウェアエミュレーション）
//!   のいずれも bit 完全一致契約のため、差分の発生源は両バックエンド
//!   とも GEMM 段のみである点は変わらない（設計 doc §7「GPU parity は
//!   REQ-2 複合判定・im2col／col2im 単体は bit 一致の 2 層構成」）。
//!   実機未実測のまま出荷し記入欄を残す。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`constant_pad_backend_
/// parity.rs::VarSource` と同じ理由・同じ構成）。
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

/// bit 完全一致（NaN 同士はクラス一致）の判定ヘルパー
/// （`constant_pad_backend_parity.rs::assert_bits_eq` と同型）。
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

// --- forward（属性なし: CPU vs NaiveOps） ---

#[test]
fn cpu_conv2d_forward_matches_naive_reference() {
    let x_shape = [1usize, 2, 5, 5];
    let w_shape = [3usize, 2, 3, 3];
    let b_shape = [3usize];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(1, &x_shape));
    let w_cpu = cpu_tape.make_var(&leaf(2, &w_shape));
    let b_cpu = cpu_tape.make_var(&leaf(3, &b_shape));
    let out_cpu = x_cpu
        .conv2d(&w_cpu, Some(&b_cpu), [1, 1], [1, 1], [1, 1], 1)
        .expect("conv2d: 常に成功する形状")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(1, &x_shape));
    let w_naive = naive_tape.make_var(&leaf(2, &w_shape));
    let b_naive = naive_tape.make_var(&leaf(3, &b_shape));
    let out_naive = x_naive
        .conv2d(&w_naive, Some(&b_naive), [1, 1], [1, 1], [1, 1], 1)
        .expect("conv2d: 常に成功する形状")
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

#[test]
fn cpu_conv2d_forward_matches_naive_reference_groups_no_bias() {
    let x_shape = [1usize, 4, 6, 6];
    let w_shape = [4usize, 1, 3, 3]; // depthwise（groups=4）

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(4, &x_shape));
    let w_cpu = cpu_tape.make_var(&leaf(5, &w_shape));
    let out_cpu = x_cpu
        .conv2d(&w_cpu, None, [1, 1], [1, 1], [1, 1], 4)
        .expect("conv2d: 常に成功する形状")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(4, &x_shape));
    let w_naive = naive_tape.make_var(&leaf(5, &w_shape));
    let out_naive = x_naive
        .conv2d(&w_naive, None, [1, 1], [1, 1], [1, 1], 4)
        .expect("conv2d: 常に成功する形状")
        .to_tensor();

    let cpu_slice = contiguous_slice(&out_cpu);
    let naive_slice = contiguous_slice(&out_naive);
    assert_bits_eq(
        "conv2d forward（groups=4・depthwise）: CpuBackendOps vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
}

// --- backward（属性なし: CPU vs NaiveOps） ---

#[test]
fn cpu_conv2d_backward_matches_naive_reference() {
    let x_shape = [1usize, 2, 5, 5];
    let w_shape = [3usize, 2, 3, 3];
    let b_shape = [3usize];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(6, &x_shape));
    let w_cpu = cpu_tape.make_var(&leaf(7, &w_shape));
    let b_cpu = cpu_tape.make_var(&leaf(8, &b_shape));
    let out_cpu = x_cpu
        .conv2d(&w_cpu, Some(&b_cpu), [1, 1], [1, 1], [1, 1], 1)
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
        .conv2d(&w_naive, Some(&b_naive), [1, 1], [1, 1], [1, 1], 1)
        .unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");
    let dw_naive = grads_naive.get(&w_naive).unwrap().expect("到達する");
    let db_naive = grads_naive.get(&b_naive).unwrap().expect("到達する");

    assert_bits_eq(
        "conv2d backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dx_cpu),
        &contiguous_slice(dx_naive),
    );
    assert_bits_eq(
        "conv2d backward（dw）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dw_cpu),
        &contiguous_slice(dw_naive),
    );
    assert_bits_eq(
        "conv2d backward（db）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(db_cpu),
        &contiguous_slice(db_naive),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。REQ-2 複合判定） ---

fn conv2d_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[1, 2, 5, 5]));
    let w = tape.make_var(&leaf(2, &[3, 2, 3, 3]));
    let b = tape.make_var(&leaf(3, &[3]));
    x.conv2d(&w, Some(&b), [1, 1], [1, 1], [1, 1], 1)
        .expect("conv2d: 常に成功する形状")
        .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`constant_pad_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_conv2d_forward_matches_cpu() {
    let metal_out = conv2d_forward_on(Device::Metal);
    let cpu_out = conv2d_forward_on(Device::Cpu);

    // GPU im2col（bit 一致。#1768）＋ GPU GEMM（REQ-2 複合判定）の
    // 合成のため、im2col／col2im 単体ではなく forward 全体を REQ-2
    // 複合判定で比較する（設計 doc §7「Conv 全体の 3 バックエンド
    // parity 判定は REQ-2 複合判定で行う」）。
    assert_parity(
        "conv2d forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_conv2d_forward_matches_cpu() {
    let cuda_out = conv2d_forward_on(Device::Cuda(0));
    let cpu_out = conv2d_forward_on(Device::Cpu);

    assert_parity(
        "conv2d forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

/// `device` 上で conv2d backward（forward → `sum(None)` → `backward`）を
/// 実行し `(dx, dw, db)` を返す（イシュー #1766）。CUDA は `im2col`／
/// `col2im` が本 issue で override されるが `conv2d` 自身は override
/// しないため、`conv2d_with_fallback` の段階的合成のうち im2col
/// （bit 一致）・GEMM（REQ-2）・col2im（bit 一致。d_input 側）を経由
/// する。d_weight／d_bias はホスト側縮約（設計 doc §6.3〜§6.4）のため
/// GEMM 段の差のみが REQ-2 判定対象となる。
fn conv2d_backward_on(device: Device) -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[1, 2, 5, 5]));
    let w = tape.make_var(&leaf(2, &[3, 2, 3, 3]));
    let b = tape.make_var(&leaf(3, &[3]));
    let out = x
        .conv2d(&w, Some(&b), [1, 1], [1, 1], [1, 1], 1)
        .expect("conv2d: 常に成功する形状");
    let loss = out.sum(None).expect("sum: 常に成功する");
    let grads = tape.backward(&loss).expect("backward: 常に成功する形状");
    let dx = grads.get(&x).unwrap().expect("到達する").clone();
    let dw = grads.get(&w).unwrap().expect("到達する").clone();
    let db = grads.get(&b).unwrap().expect("到達する").clone();
    (dx, dw, db)
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_conv2d_backward_matches_cpu() {
    let (dx_cuda, dw_cuda, db_cuda) = conv2d_backward_on(Device::Cuda(0));
    let (dx_cpu, dw_cpu, db_cpu) = conv2d_backward_on(Device::Cpu);

    assert_parity(
        "conv2d backward（dx）: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&dx_cuda),
        &contiguous_slice(&dx_cpu),
    );
    assert_parity(
        "conv2d backward（dw）: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&dw_cuda),
        &contiguous_slice(&dw_cpu),
    );
    assert_parity(
        "conv2d backward（db）: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&db_cuda),
        &contiguous_slice(&db_cpu),
    );
}

/// `metal_conv2d_forward_matches_cpu` と同じ理由で `cfg(target_os =
/// "macos")` 限定（イシュー #1768）。Metal は `im2col`／`col2im` を
/// override 済み（bit 完全一致契約は不変）だが `conv2d` 自身は
/// override しないため、`conv2d_with_fallback` の段階的合成のうち
/// im2col（bit 一致）・GEMM（REQ-2）・col2im（bit 一致。d_input 側）を
/// 経由する。d_weight／d_bias はホスト側縮約（設計 doc §6.3〜§6.4）の
/// ため GEMM 段の差のみが REQ-2 判定対象となる（`cuda_conv2d_backward_
/// matches_cpu` doc comment と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_conv2d_backward_matches_cpu() {
    let (dx_metal, dw_metal, db_metal) = conv2d_backward_on(Device::Metal);
    let (dx_cpu, dw_cpu, db_cpu) = conv2d_backward_on(Device::Cpu);

    assert_parity(
        "conv2d backward（dx）: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&dx_metal),
        &contiguous_slice(&dx_cpu),
    );
    assert_parity(
        "conv2d backward（dw）: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&dw_metal),
        &contiguous_slice(&dw_cpu),
    );
    assert_parity(
        "conv2d backward（db）: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&db_metal),
        &contiguous_slice(&db_cpu),
    );
}
