//! `fandhe_ai_autodiff::conv3d_ops::conv3d`（イシュー #2158・facade
//! 非公開の内部入口。`crates/autodiff/src/conv3d_ops.rs` モジュール
//! doc 参照）のバックエンド間 parity テスト（`conv2d_backend_parity.rs`
//! の空間 3 軸一般化）。
//!
//! `conv3d_ops` は facade から再エクスポートされないため、本テストは
//! `fandhe_ai_autodiff::conv3d_ops::conv3d` を直接 use する（facade の
//! dev 依存に `fandhe-ai-autodiff` が既に含まれている）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`。`im2col3d`／
//!   `col2im3d` override 実装）と `fandhe_ai_autodiff::Tape::new()`
//!   （`NaiveOps`。ホスト `eval::im2col3d`／`col2im3d` フォールバック
//!   経路）で forward／backward を bit 同一で突き合わせる（im2col3d／
//!   col2im3d 自体は算術を含まないコピー・`f64` 逐次加算のいずれも
//!   bit 完全一致契約。`docs/conv-ops-design.md` §16）。
//! - kD=1・D=1 の Conv3d が reshape 経由の Conv2d と bit 一致すること
//!   （`crates/autodiff/tests/conv3d.rs::
//!   kernel_depth_one_matches_conv2d_via_reshape` の facade 到達経路
//!   （CPU バックエンド）版）。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の経路を CPU tape
//!   と `assert_parity`（REQ-2 複合判定。GEMM 由来の差を許容）で比較
//!   する。CUDA／Metal とも `conv3d`／`im2col3d`／`col2im3d` を
//!   override しないため（既定 `Unsupported` のまま。設計 doc §16
//!   §2.3）、GPU 経路は `Op::Conv3d` の VJP 段が
//!   `im2col3d_with_fallback`（ホスト実装へフォールバック）→
//!   `ops.gemm_batched_fp32_strict`（GPU GEMM）の合成になる。実機未
//!   実測のまま出荷し記入欄を残す（`docs/perf/logs/conv3d-2158/
//!   README.md`）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::conv3d_ops::conv3d;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`conv2d_backend_parity.rs::
/// VarSource` と同じ理由・同じ構成）。
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
/// （`conv2d_backend_parity.rs::assert_bits_eq` と同型）。
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

/// `data` の全要素の `to_bits()` を FNV-1a 相当で fold した診断用
/// チェックサムを `<label>.fold_bits=<hex>` 形式で 1 行出力する
/// （`conv2d_backend_parity.rs::print_fold_bits` と同一の FNV-1a
/// fold）。
fn print_fold_bits(label: &str, data: &[f32]) {
    let mut acc: u64 = 0xcbf29ce484222325; // FNV-1a 相当の固定初期値（診断専用・暗号用途ではない）
    for &v in data.iter() {
        let bits = v.to_bits() as u64;
        acc ^= bits;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    println!("{label}.fold_bits={acc:#018x}");
}

// --- forward（属性なし: CPU vs NaiveOps） ---

#[test]
fn cpu_conv3d_forward_matches_naive_reference() {
    let x_shape = [1usize, 2, 3, 4, 4];
    let w_shape = [3usize, 2, 2, 3, 3];
    let b_shape = [3usize];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(1, &x_shape));
    let w_cpu = cpu_tape.make_var(&leaf(2, &w_shape));
    let b_cpu = cpu_tape.make_var(&leaf(3, &b_shape));
    let out_cpu = conv3d(
        &x_cpu,
        &w_cpu,
        Some(&b_cpu),
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        1,
    )
    .expect("conv3d: 常に成功する形状")
    .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(1, &x_shape));
    let w_naive = naive_tape.make_var(&leaf(2, &w_shape));
    let b_naive = naive_tape.make_var(&leaf(3, &b_shape));
    let out_naive = conv3d(
        &x_naive,
        &w_naive,
        Some(&b_naive),
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        1,
    )
    .expect("conv3d: 常に成功する形状")
    .to_tensor();

    let cpu_slice = contiguous_slice(&out_cpu);
    let naive_slice = contiguous_slice(&out_naive);
    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::im2col3d/col2im3d）vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
    assert_bits_eq(
        "fandhe_ai::tape()（CpuBackendOps::im2col3d/col2im3d）vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
}

#[test]
fn cpu_conv3d_forward_matches_naive_reference_groups_no_bias() {
    let x_shape = [1usize, 4, 4, 5, 5];
    let w_shape = [4usize, 1, 2, 3, 3]; // depthwise（groups=4）

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(4, &x_shape));
    let w_cpu = cpu_tape.make_var(&leaf(5, &w_shape));
    let out_cpu = conv3d(&x_cpu, &w_cpu, None, [1, 1, 1], [1, 1, 1], [1, 1, 1], 4)
        .expect("conv3d: 常に成功する形状")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(4, &x_shape));
    let w_naive = naive_tape.make_var(&leaf(5, &w_shape));
    let out_naive = conv3d(&x_naive, &w_naive, None, [1, 1, 1], [1, 1, 1], [1, 1, 1], 4)
        .expect("conv3d: 常に成功する形状")
        .to_tensor();

    let cpu_slice = contiguous_slice(&out_cpu);
    let naive_slice = contiguous_slice(&out_naive);
    assert_bits_eq(
        "conv3d forward（groups=4・depthwise）: CpuBackendOps vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
}

// --- backward（属性なし: CPU vs NaiveOps） ---

#[test]
fn cpu_conv3d_backward_matches_naive_reference() {
    let x_shape = [1usize, 2, 3, 4, 4];
    let w_shape = [3usize, 2, 2, 3, 3];
    let b_shape = [3usize];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(6, &x_shape));
    let w_cpu = cpu_tape.make_var(&leaf(7, &w_shape));
    let b_cpu = cpu_tape.make_var(&leaf(8, &b_shape));
    let out_cpu = conv3d(
        &x_cpu,
        &w_cpu,
        Some(&b_cpu),
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        1,
    )
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
    let out_naive = conv3d(
        &x_naive,
        &w_naive,
        Some(&b_naive),
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        1,
    )
    .unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");
    let dw_naive = grads_naive.get(&w_naive).unwrap().expect("到達する");
    let db_naive = grads_naive.get(&b_naive).unwrap().expect("到達する");

    assert_bits_eq(
        "conv3d backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dx_cpu),
        &contiguous_slice(dx_naive),
    );
    assert_bits_eq(
        "conv3d backward（dw）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dw_cpu),
        &contiguous_slice(dw_naive),
    );
    assert_bits_eq(
        "conv3d backward（db）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(db_cpu),
        &contiguous_slice(db_naive),
    );
}

// --- kD=1・D=1 が reshape 経由の Conv2d と bit 一致（CPU バックエンド） ---

#[test]
fn cpu_conv3d_kernel_depth_one_matches_conv2d_via_reshape() {
    let in_shape_3d = [1usize, 2, 1, 4, 4];
    let in_shape_2d = [1usize, 2, 4, 4];
    let w_shape_3d = [3usize, 2, 1, 2, 2];
    let w_shape_2d = [3usize, 2, 2, 2];

    let x_data = Xorshift64Star::new(9).fill_vec(in_shape_3d.iter().product());
    let w_data = Xorshift64Star::new(10).fill_vec(w_shape_3d.iter().product());
    let b_data = Xorshift64Star::new(11).fill_vec(3);

    let tape3 = fandhe_ai::tape();
    let x3 = tape3.make_var(&Tensor::new(x_data.clone(), &in_shape_3d).unwrap());
    let w3 = tape3.make_var(&Tensor::new(w_data.clone(), &w_shape_3d).unwrap());
    let b3 = tape3.make_var(&Tensor::new(b_data.clone(), &[3]).unwrap());
    let y3 = conv3d(&x3, &w3, Some(&b3), [1, 1, 1], [0, 0, 0], [1, 1, 1], 1).unwrap();

    let tape2 = fandhe_ai::tape();
    let x2 = tape2.make_var(&Tensor::new(x_data, &in_shape_2d).unwrap());
    let w2 = tape2.make_var(&Tensor::new(w_data, &w_shape_2d).unwrap());
    let b2 = tape2.make_var(&Tensor::new(b_data, &[3]).unwrap());
    let y2 = x2
        .conv2d(&w2, Some(&b2), [1, 1], [0, 0], [1, 1], 1)
        .unwrap();

    assert_bits_eq(
        "conv3d(kD=1) vs conv2d（CpuBackendOps）",
        &contiguous_slice(&y3.to_tensor()),
        &contiguous_slice(&y2.to_tensor()),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。REQ-2 複合判定） ---

fn conv3d_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[1, 2, 3, 4, 4]));
    let w = tape.make_var(&leaf(2, &[3, 2, 2, 3, 3]));
    let b = tape.make_var(&leaf(3, &[3]));
    conv3d(&x, &w, Some(&b), [1, 1, 1], [1, 1, 1], [1, 1, 1], 1)
        .expect("conv3d: 常に成功する形状")
        .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`conv2d_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_conv3d_forward_matches_cpu() {
    let metal_out = conv3d_forward_on(Device::Metal);
    let cpu_out = conv3d_forward_on(Device::Cpu);

    // GPU im2col3d（bit 一致想定。override 無しでホストへフォール
    // バックするため実質は CPU 参照実装と同型）＋ GPU GEMM（REQ-2
    // 複合判定）の合成のため、forward 全体を REQ-2 複合判定で比較する
    // （`conv2d_backend_parity.rs` と同じ判定方式）。
    assert_parity(
        "conv3d forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
    print_fold_bits(
        "metal_conv3d_forward_matches_cpu[out]",
        &contiguous_slice(&metal_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_conv3d_forward_matches_cpu() {
    let cuda_out = conv3d_forward_on(Device::Cuda(0));
    let cpu_out = conv3d_forward_on(Device::Cpu);

    assert_parity(
        "conv3d forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
    print_fold_bits(
        "cuda_conv3d_forward_matches_cpu[out]",
        &contiguous_slice(&cuda_out),
    );
}
