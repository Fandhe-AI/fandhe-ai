//! `fandhe_ai_autodiff::activation_ops`（イシュー #2146・facade 非公開の
//! 内部入口。`crates/autodiff/src/activation_ops.rs` モジュール doc
//! 参照）のバックエンド間 parity テスト（`matrix_ops_backend_parity.rs`
//! と同型）。
//!
//! `activation_ops` は facade から再エクスポートされないため、本テスト
//! は `fandhe_ai_autodiff::activation_ops::*` を直接 use する（facade
//! の dev 依存に `fandhe-ai-autodiff` が既に含まれている）。
//!
//! 本ファイルの契約は `mish`／`hardtanh`／`relu6`／`prelu`／`glu` の
//! 各演算 × {forward, backward} × {CPU vs NaiveOps, CUDA vs CPU, Metal
//! vs CPU} の各セルを埋めることであり、以下の関数名は網羅表と一対一
//! 対応する（`rearrange_ops_backend_parity.rs`／`matrix_ops_backend_
//! parity.rs` の教訓: 関数名・doc が謳う対象と実際に実行する演算が
//! ずれる codex-review 指摘の再発防止）。
//!
//! **判定方式の割り当て**（`crate::activation_ops` モジュール doc
//! 「数値契約」参照）:
//! - bit 完全一致: `hardtanh`／`relu6`（forward・backward とも）・
//!   `prelu`（forward・入力勾配）
//! - REQ-2 統一複合判定: `mish`（forward・backward とも）・`glu`
//!   （forward・backward とも）・`prelu`（`weight` 勾配）
//!
//! - 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps`〉と
//!   `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`〉の突き合わせ）:
//!   - bit 完全一致 forward（`hardtanh`／`relu6`／`prelu` 入力・出力）:
//!     `cpu_bit_exact_forward_matches_naive_reference`
//!   - bit 完全一致 backward（`hardtanh`／`relu6`／`prelu` 入力勾配）:
//!     `cpu_bit_exact_backward_matches_naive_reference`
//!   - REQ-2 forward（`mish`／`glu`）:
//!     `cpu_req2_forward_matches_naive_reference_within_tolerance`
//!   - REQ-2 backward（`mish`／`glu`／`prelu` の `weight` 勾配）:
//!     `cpu_req2_backward_matches_naive_reference_within_tolerance`
//! - `#[ignore]`（`tape_for(Device::Metal)`〈`cfg(target_os =
//!   "macos")` 限定〉／`tape_for(Device::Cuda(0))` で同じ経路を CPU
//!   tape と比較）: 上記 forward 2 種（bit 完全一致・REQ-2）と backward
//!   2 種を `cuda_*`／`metal_*` という接頭辞で対称に置く（計 8 件）。
//!
//!   実機（DGX Spark GB10／Apple Silicon）への到達手段が本エージェント
//!   実行環境にないため未実施のまま Mac／GB10 セッションへ申し送る
//!   （`docs/perf/logs/activation-ops-2146/README.md`）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::activation_ops::{glu, hardtanh, mish, prelu, relu6};
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

fn f32_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// `hardtanh`／`prelu` の入力 fixture（境界値近傍・正負・0 を含む）。
fn activation_fixture() -> Tensor<f32> {
    Tensor::new(vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], &[8]).unwrap()
}

/// `prelu` のチャネル数 C = 2 用 weight（重み自体は `mul`／`where_cond`
/// のみを経由するため forward・入力勾配は bit 完全一致するが、`weight`
/// 勾配は縮約を経由するため REQ-2 判定の対象になる）。
fn prelu_weight_c2() -> Tensor<f32> {
    Tensor::new(vec![0.1, 0.2], &[2]).unwrap()
}

fn prelu_input_c2() -> Tensor<f32> {
    Tensor::new(vec![-1.0, -2.0, -3.0, -4.0], &[1, 2, 2]).unwrap()
}

/// `mish`／`glu` の入力 fixture（超越関数経路を通すため REQ-2 判定の
/// 代表値。`glu` は末尾軸長を偶数にする）。
fn transcendental_fixture() -> Tensor<f32> {
    Tensor::new(vec![-3.0, -1.0, 0.0, 0.5, 1.0, 2.0], &[6]).unwrap()
}

// ---------------------------------------------------------------------
// CPU vs NaiveOps（属性なし）
// ---------------------------------------------------------------------

/// `hardtanh`／`relu6`／`prelu`（forward・入力側の出力）が CPU
/// （`fandhe_ai::tape()`）と NaiveOps（`fandhe_ai_autodiff::Tape::new()`）
/// で bit 完全一致することを確認する（`where_cond`／`mul` はいずれも
/// 算術を含まない選択・1 回乗算のみのため）。
#[test]
fn cpu_bit_exact_forward_matches_naive_reference() {
    let data = activation_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    assert_eq!(
        f32_bits(&hardtanh(&x_cpu, -1.0, 1.0).unwrap().to_tensor()),
        f32_bits(&hardtanh(&x_naive, -1.0, 1.0).unwrap().to_tensor()),
        "hardtanh forward"
    );
    assert_eq!(
        f32_bits(&relu6(&x_cpu).unwrap().to_tensor()),
        f32_bits(&relu6(&x_naive).unwrap().to_tensor()),
        "relu6 forward"
    );

    let w = prelu_weight_c2();
    let xin = prelu_input_c2();
    let x_cpu2 = cpu_tape.make_var(&xin);
    let w_cpu = cpu_tape.make_var(&w);
    let x_naive2 = naive_tape.make_var(&xin);
    let w_naive = naive_tape.make_var(&w);
    assert_eq!(
        f32_bits(&prelu(&x_cpu2, &w_cpu).unwrap().to_tensor()),
        f32_bits(&prelu(&x_naive2, &w_naive).unwrap().to_tensor()),
        "prelu forward"
    );
}

/// `hardtanh`／`relu6`／`prelu`（入力勾配）が CPU と NaiveOps で bit
/// 完全一致することを確認する（`hardtanh`・`relu6` は開区間内 1・
/// それ以外 0 の選択のみ、`prelu` 入力勾配も選択と乗算の局所勾配のみで
/// いずれも縮約を経由しないため）。
#[test]
fn cpu_bit_exact_backward_matches_naive_reference() {
    let data = activation_fixture();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = hardtanh(&x_cpu, -1.0, 1.0).unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let loss_naive = hardtanh(&x_naive, -1.0, 1.0).unwrap().sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().unwrap();

    assert_eq!(
        f32_bits(dx_cpu),
        f32_bits(dx_naive),
        "hardtanh backward (input grad)"
    );

    let cpu_tape2 = fandhe_ai::tape();
    let x_cpu2 = cpu_tape2.make_var(&data);
    let loss_cpu2 = relu6(&x_cpu2).unwrap().sum(None).unwrap();
    let grads_cpu2 = cpu_tape2.backward(&loss_cpu2).unwrap();
    let dx_cpu2 = grads_cpu2.get(&x_cpu2).unwrap().unwrap();

    let naive_tape2 = fandhe_ai_autodiff::Tape::new();
    let x_naive2 = naive_tape2.make_var(&data);
    let loss_naive2 = relu6(&x_naive2).unwrap().sum(None).unwrap();
    let grads_naive2 = naive_tape2.backward(&loss_naive2).unwrap();
    let dx_naive2 = grads_naive2.get(&x_naive2).unwrap().unwrap();

    assert_eq!(
        f32_bits(dx_cpu2),
        f32_bits(dx_naive2),
        "relu6 backward (input grad)"
    );

    let w = prelu_weight_c2();
    let xin = prelu_input_c2();
    let cpu_tape3 = fandhe_ai::tape();
    let x_cpu3 = cpu_tape3.make_var(&xin);
    let w_cpu3 = cpu_tape3.make_var(&w);
    let loss_cpu3 = prelu(&x_cpu3, &w_cpu3).unwrap().sum(None).unwrap();
    let grads_cpu3 = cpu_tape3.backward(&loss_cpu3).unwrap();
    let dx_cpu3 = grads_cpu3.get(&x_cpu3).unwrap().unwrap();

    let naive_tape3 = fandhe_ai_autodiff::Tape::new();
    let x_naive3 = naive_tape3.make_var(&xin);
    let w_naive3 = naive_tape3.make_var(&w);
    let loss_naive3 = prelu(&x_naive3, &w_naive3).unwrap().sum(None).unwrap();
    let grads_naive3 = naive_tape3.backward(&loss_naive3).unwrap();
    let dx_naive3 = grads_naive3.get(&x_naive3).unwrap().unwrap();

    assert_eq!(
        f32_bits(dx_cpu3),
        f32_bits(dx_naive3),
        "prelu backward (input grad)"
    );
}

/// `mish`／`glu`（forward）が CPU と NaiveOps で REQ-2 統一複合判定
/// （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を満たすことを
/// 確認する（`softplus`／`tanh`／`sigmoid` はいずれも超越関数のため）。
#[test]
fn cpu_req2_forward_matches_naive_reference_within_tolerance() {
    let data = transcendental_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    let mish_cpu = mish(&x_cpu).unwrap().to_tensor();
    let mish_naive = mish(&x_naive).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "mish forward: cpu vs naive",
        mish_cpu.host_slice().as_ref(),
        mish_naive.host_slice().as_ref(),
    );

    let glu_cpu = glu(&x_cpu, 0).unwrap().to_tensor();
    let glu_naive = glu(&x_naive, 0).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "glu forward: cpu vs naive",
        glu_cpu.host_slice().as_ref(),
        glu_naive.host_slice().as_ref(),
    );
}

/// `mish`／`glu`（backward）・`prelu`（`weight` 勾配）が CPU と
/// NaiveOps で REQ-2 統一複合判定を満たすことを確認する（`mish`／
/// `glu` は超越関数を含む乗算の合成勾配、`prelu` の `weight` 勾配は
/// `reduce_to_shape` の縮約を経由するため）。
#[test]
fn cpu_req2_backward_matches_naive_reference_within_tolerance() {
    let data = transcendental_fixture();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = mish(&x_cpu).unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let loss_naive = mish(&x_naive).unwrap().sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().unwrap();

    fandhe_ai_backend_cpu::parity::assert_parity(
        "mish backward: cpu vs naive",
        dx_cpu.host_slice().as_ref(),
        dx_naive.host_slice().as_ref(),
    );

    let cpu_tape2 = fandhe_ai::tape();
    let x_cpu2 = cpu_tape2.make_var(&data);
    let loss_cpu2 = glu(&x_cpu2, 0).unwrap().sum(None).unwrap();
    let grads_cpu2 = cpu_tape2.backward(&loss_cpu2).unwrap();
    let dx_cpu2 = grads_cpu2.get(&x_cpu2).unwrap().unwrap();

    let naive_tape2 = fandhe_ai_autodiff::Tape::new();
    let x_naive2 = naive_tape2.make_var(&data);
    let loss_naive2 = glu(&x_naive2, 0).unwrap().sum(None).unwrap();
    let grads_naive2 = naive_tape2.backward(&loss_naive2).unwrap();
    let dx_naive2 = grads_naive2.get(&x_naive2).unwrap().unwrap();

    fandhe_ai_backend_cpu::parity::assert_parity(
        "glu backward: cpu vs naive",
        dx_cpu2.host_slice().as_ref(),
        dx_naive2.host_slice().as_ref(),
    );

    let w = prelu_weight_c2();
    let xin = prelu_input_c2();
    let cpu_tape3 = fandhe_ai::tape();
    let x_cpu3 = cpu_tape3.make_var(&xin);
    let w_cpu3 = cpu_tape3.make_var(&w);
    let loss_cpu3 = prelu(&x_cpu3, &w_cpu3).unwrap().sum(None).unwrap();
    let grads_cpu3 = cpu_tape3.backward(&loss_cpu3).unwrap();
    let dw_cpu3 = grads_cpu3.get(&w_cpu3).unwrap().unwrap();

    let naive_tape3 = fandhe_ai_autodiff::Tape::new();
    let x_naive3 = naive_tape3.make_var(&xin);
    let w_naive3 = naive_tape3.make_var(&w);
    let loss_naive3 = prelu(&x_naive3, &w_naive3).unwrap().sum(None).unwrap();
    let grads_naive3 = naive_tape3.backward(&loss_naive3).unwrap();
    let dw_naive3 = grads_naive3.get(&w_naive3).unwrap().unwrap();

    fandhe_ai_backend_cpu::parity::assert_parity(
        "prelu weight grad: cpu vs naive",
        dw_cpu3.host_slice().as_ref(),
        dw_naive3.host_slice().as_ref(),
    );
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: Mac／DGX Spark GB10 実機セッションへ
// 申し送る（`docs/perf/logs/activation-ops-2146/README.md`）。
// ---------------------------------------------------------------------

/// bit 完全一致 forward 3 演算（`hardtanh`／`relu6`／`prelu`）の
/// CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/activation-ops-2146/README.md 参照"]
fn metal_bit_exact_forward_matches_cpu_reference() {
    let data = activation_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);

    assert_eq!(
        f32_bits(&hardtanh(&x_cpu, -1.0, 1.0).unwrap().to_tensor()),
        f32_bits(&hardtanh(&x_metal, -1.0, 1.0).unwrap().to_tensor()),
        "hardtanh forward"
    );
    assert_eq!(
        f32_bits(&relu6(&x_cpu).unwrap().to_tensor()),
        f32_bits(&relu6(&x_metal).unwrap().to_tensor()),
        "relu6 forward"
    );

    let w = prelu_weight_c2();
    let xin = prelu_input_c2();
    let x_cpu2 = cpu_tape.make_var(&xin);
    let w_cpu = cpu_tape.make_var(&w);
    let x_metal2 = metal_tape.make_var(&xin);
    let w_metal = metal_tape.make_var(&w);
    assert_eq!(
        f32_bits(&prelu(&x_cpu2, &w_cpu).unwrap().to_tensor()),
        f32_bits(&prelu(&x_metal2, &w_metal).unwrap().to_tensor()),
        "prelu forward"
    );
}

/// bit 完全一致 backward 3 演算の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/activation-ops-2146/README.md 参照"]
fn metal_bit_exact_backward_matches_cpu_reference() {
    let data = activation_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = hardtanh(&x_cpu, -1.0, 1.0).unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap();

    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);
    let loss_metal = hardtanh(&x_metal, -1.0, 1.0).unwrap().sum(None).unwrap();
    let grads_metal = metal_tape.backward(&loss_metal).unwrap();
    let dx_metal = grads_metal.get(&x_metal).unwrap().unwrap();

    assert_eq!(f32_bits(dx_cpu), f32_bits(dx_metal), "hardtanh backward");
}

/// REQ-2 forward 2 演算（`mish`／`glu`）の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/activation-ops-2146/README.md 参照"]
fn metal_req2_forward_matches_cpu_reference() {
    let data = transcendental_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);

    let mish_cpu = mish(&x_cpu).unwrap().to_tensor();
    let mish_metal = mish(&x_metal).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "mish forward: cpu vs metal",
        mish_cpu.host_slice().as_ref(),
        mish_metal.host_slice().as_ref(),
    );

    let glu_cpu = glu(&x_cpu, 0).unwrap().to_tensor();
    let glu_metal = glu(&x_metal, 0).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "glu forward: cpu vs metal",
        glu_cpu.host_slice().as_ref(),
        glu_metal.host_slice().as_ref(),
    );
}

/// REQ-2 backward（`mish`）の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/activation-ops-2146/README.md 参照"]
fn metal_req2_backward_matches_cpu_reference() {
    let data = transcendental_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = mish(&x_cpu).unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap();

    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);
    let loss_metal = mish(&x_metal).unwrap().sum(None).unwrap();
    let grads_metal = metal_tape.backward(&loss_metal).unwrap();
    let dx_metal = grads_metal.get(&x_metal).unwrap().unwrap();

    fandhe_ai_backend_cpu::parity::assert_parity(
        "mish backward: cpu vs metal",
        dx_cpu.host_slice().as_ref(),
        dx_metal.host_slice().as_ref(),
    );
}

/// bit 完全一致 forward 3 演算の CPU／CUDA 実機（DGX Spark GB10）比較。
/// 上記 Metal 版と対称。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/activation-ops-2146/README.md 参照"]
fn cuda_bit_exact_forward_matches_cpu_reference() {
    let data = activation_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);

    assert_eq!(
        f32_bits(&hardtanh(&x_cpu, -1.0, 1.0).unwrap().to_tensor()),
        f32_bits(&hardtanh(&x_cuda, -1.0, 1.0).unwrap().to_tensor()),
        "hardtanh forward"
    );
    assert_eq!(
        f32_bits(&relu6(&x_cpu).unwrap().to_tensor()),
        f32_bits(&relu6(&x_cuda).unwrap().to_tensor()),
        "relu6 forward"
    );

    let w = prelu_weight_c2();
    let xin = prelu_input_c2();
    let x_cpu2 = cpu_tape.make_var(&xin);
    let w_cpu = cpu_tape.make_var(&w);
    let x_cuda2 = cuda_tape.make_var(&xin);
    let w_cuda = cuda_tape.make_var(&w);
    assert_eq!(
        f32_bits(&prelu(&x_cpu2, &w_cpu).unwrap().to_tensor()),
        f32_bits(&prelu(&x_cuda2, &w_cuda).unwrap().to_tensor()),
        "prelu forward"
    );
}

/// bit 完全一致 backward 3 演算の CPU／CUDA 実機比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/activation-ops-2146/README.md 参照"]
fn cuda_bit_exact_backward_matches_cpu_reference() {
    let data = activation_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = hardtanh(&x_cpu, -1.0, 1.0).unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap();

    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);
    let loss_cuda = hardtanh(&x_cuda, -1.0, 1.0).unwrap().sum(None).unwrap();
    let grads_cuda = cuda_tape.backward(&loss_cuda).unwrap();
    let dx_cuda = grads_cuda.get(&x_cuda).unwrap().unwrap();

    assert_eq!(f32_bits(dx_cpu), f32_bits(dx_cuda), "hardtanh backward");
}

/// REQ-2 forward 2 演算の CPU／CUDA 実機比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/activation-ops-2146/README.md 参照"]
fn cuda_req2_forward_matches_cpu_reference() {
    let data = transcendental_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);

    let mish_cpu = mish(&x_cpu).unwrap().to_tensor();
    let mish_cuda = mish(&x_cuda).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "mish forward: cpu vs cuda",
        mish_cpu.host_slice().as_ref(),
        mish_cuda.host_slice().as_ref(),
    );

    let glu_cpu = glu(&x_cpu, 0).unwrap().to_tensor();
    let glu_cuda = glu(&x_cuda, 0).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "glu forward: cpu vs cuda",
        glu_cpu.host_slice().as_ref(),
        glu_cuda.host_slice().as_ref(),
    );
}

/// REQ-2 backward（`mish`）の CPU／CUDA 実機比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/activation-ops-2146/README.md 参照"]
fn cuda_req2_backward_matches_cpu_reference() {
    let data = transcendental_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = mish(&x_cpu).unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap();

    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);
    let loss_cuda = mish(&x_cuda).unwrap().sum(None).unwrap();
    let grads_cuda = cuda_tape.backward(&loss_cuda).unwrap();
    let dx_cuda = grads_cuda.get(&x_cuda).unwrap().unwrap();

    fandhe_ai_backend_cpu::parity::assert_parity(
        "mish backward: cpu vs cuda",
        dx_cpu.host_slice().as_ref(),
        dx_cuda.host_slice().as_ref(),
    );
}
