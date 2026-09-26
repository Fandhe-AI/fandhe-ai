//! `fandhe_ai_autodiff::loss_ops`（イシュー #2166・facade 非公開の
//! 内部入口。`crates/autodiff/src/loss_ops.rs` モジュール doc 参照）の
//! バックエンド間 parity テスト（`conv3d_backend_parity.rs` と同型の
//! 構成）。
//!
//! `loss_ops` は facade から再エクスポートされないため、本テストは
//! `fandhe_ai_autodiff::loss_ops` を直接 use する（facade の dev 依存に
//! `fandhe-ai-autodiff` が既に含まれている）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で L1・CE
//!   （既定オプション・非既定オプション）の forward／backward が bit
//!   完全一致すること（`Op::L1Loss`／`Op::CrossEntropyLossWithOptions`
//!   はいずれも `BackendOps` に対応メソッドを持たず常にホスト計算の
//!   ため）。既存 `Var::cross_entropy_loss`（既定オプションの委譲先）
//!   も同じ方法で固定する。
//! - `#[ignore]`: `Device::Cuda(0)`／`Device::Metal`（`cfg(target_os =
//!   "macos")` 限定）のテープを CPU と比較する。判定は
//!   `fandhe_ai_backend_cpu::parity::assert_parity`（REQ-2 統一複合
//!   判定。tolerance は変更しない）。ホスト計算のため bit 一致も
//!   期待できるが、主張は REQ-2 判定に留める（実機未実測のため
//!   `docs/perf/logs/loss-ops-2166/README.md` へ申し送る）。
//!
//! イシュー #2167（親 #2131）で距離ベースの損失 3 種
//! （`cosine_embedding_loss`・`margin_ranking_loss`・
//! `triplet_margin_loss`）と `poisson_nll_loss` の parity テストを
//! 追加した（L1・CE と同じ構成。実機未実測分は
//! `docs/perf/logs/loss-ops-2167/README.md` へ申し送る）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::loss_ops::{
    self, CrossEntropyOptions, PoissonNllOptions, TripletMarginOptions,
};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`conv3d_backend_parity.rs::
/// VarSource` と同型）。
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
/// （`conv3d_backend_parity.rs::assert_bits_eq` と同型）。
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

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

fn f32_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data.to_vec(), shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn i32_tensor(data: &[i32], shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data.to_vec(), shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

// --- L1 損失: 属性なし（CPU vs NaiveOps） ---

#[test]
fn cpu_l1_loss_forward_and_backward_match_naive_reference() {
    let pred_data = [1.0f32, -2.0, 3.0, 0.5];
    let target_data = [0.5f32, -1.0, 2.5, 1.0];

    let cpu_tape = fandhe_ai::tape();
    let pred_cpu = cpu_tape.make_var(&f32_tensor(&pred_data, &[2, 2]));
    let target_cpu = cpu_tape.make_var(&f32_tensor(&target_data, &[2, 2]));
    let loss_cpu =
        loss_ops::l1_loss(&pred_cpu, &target_cpu, fandhe_ai_autodiff::Reduction::Mean).unwrap();
    let grads_cpu = cpu_tape
        .backward(&loss_cpu)
        .expect("backward は成功するはず");
    let dpred_cpu = contiguous_slice(grads_cpu.get(&pred_cpu).unwrap().expect("到達する"));

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let pred_naive = naive_tape.make_var(&f32_tensor(&pred_data, &[2, 2]));
    let target_naive = naive_tape.make_var(&f32_tensor(&target_data, &[2, 2]));
    let loss_naive = loss_ops::l1_loss(
        &pred_naive,
        &target_naive,
        fandhe_ai_autodiff::Reduction::Mean,
    )
    .unwrap();
    let grads_naive = naive_tape
        .backward(&loss_naive)
        .expect("backward は成功するはず");
    let dpred_naive = contiguous_slice(grads_naive.get(&pred_naive).unwrap().expect("到達する"));

    assert_bits_eq(
        "l1_loss forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&loss_cpu.to_tensor()),
        &contiguous_slice(&loss_naive.to_tensor()),
    );
    assert_bits_eq(
        "l1_loss backward: CpuBackendOps vs NaiveOps",
        &dpred_cpu,
        &dpred_naive,
    );
}

// --- CrossEntropy（非既定オプション）: 属性なし（CPU vs NaiveOps） ---

#[test]
fn cpu_cross_entropy_loss_with_options_forward_and_backward_match_naive_reference() {
    let logits_data = [1.0f32, 2.0, 0.5, -1.0, 0.5, 2.0, 0.3, -0.2, 1.1];
    let targets = i32_tensor(&[2, 0, 1], &[3]);
    let options = CrossEntropyOptions::default()
        .label_smoothing(0.1)
        .ignore_index(1)
        .class_weight(f32_tensor(&[1.5, 0.5, 1.0], &[3]));

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&f32_tensor(&logits_data, &[3, 3]));
    let loss_cpu = loss_ops::cross_entropy_loss_with(
        &x_cpu,
        &targets,
        1,
        fandhe_ai_autodiff::Reduction::Mean,
        &options,
    )
    .unwrap();
    let grads_cpu = cpu_tape
        .backward(&loss_cpu)
        .expect("backward は成功するはず");
    let dx_cpu = contiguous_slice(grads_cpu.get(&x_cpu).unwrap().expect("到達する"));

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&f32_tensor(&logits_data, &[3, 3]));
    let loss_naive = loss_ops::cross_entropy_loss_with(
        &x_naive,
        &targets,
        1,
        fandhe_ai_autodiff::Reduction::Mean,
        &options,
    )
    .unwrap();
    let grads_naive = naive_tape
        .backward(&loss_naive)
        .expect("backward は成功するはず");
    let dx_naive = contiguous_slice(grads_naive.get(&x_naive).unwrap().expect("到達する"));

    assert_bits_eq(
        "cross_entropy_loss_with forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&loss_cpu.to_tensor()),
        &contiguous_slice(&loss_naive.to_tensor()),
    );
    assert_bits_eq(
        "cross_entropy_loss_with backward: CpuBackendOps vs NaiveOps",
        &dx_cpu,
        &dx_naive,
    );
}

// --- CrossEntropy（既定オプション）: 既存 Var::cross_entropy_loss との
//     bit 完全一致固定（R3。facade 到達経路版） ---

#[test]
fn cpu_cross_entropy_loss_with_default_options_matches_existing_method_bit_exact() {
    let logits_data = [1.0f32, 2.0, 0.5, -1.0, 0.5, 2.0];
    let targets = i32_tensor(&[2, 0], &[2]);

    let tape_a = fandhe_ai::tape();
    let x_a = tape_a.make_var(&f32_tensor(&logits_data, &[2, 3]));
    let via_existing = x_a
        .cross_entropy_loss(&targets, 1, fandhe_ai_autodiff::Reduction::Mean)
        .unwrap();

    let tape_b = fandhe_ai::tape();
    let x_b = tape_b.make_var(&f32_tensor(&logits_data, &[2, 3]));
    let via_with = loss_ops::cross_entropy_loss_with(
        &x_b,
        &targets,
        1,
        fandhe_ai_autodiff::Reduction::Mean,
        &CrossEntropyOptions::default(),
    )
    .unwrap();

    assert_bits_eq(
        "cross_entropy_loss_with(既定オプション) vs Var::cross_entropy_loss",
        &contiguous_slice(&via_with.to_tensor()),
        &contiguous_slice(&via_existing.to_tensor()),
    );
}

// --- CosineEmbedding 損失: 属性なし（CPU vs NaiveOps） ---

#[test]
fn cpu_cosine_embedding_loss_forward_and_backward_match_naive_reference() {
    let x1_data = [1.0f32, 2.0, -0.5, 0.3];
    let x2_data = [0.4f32, -0.6, 1.1, -0.9];
    let y = f32_tensor(&[1.0, -1.0], &[2]);

    let cpu_tape = fandhe_ai::tape();
    let x1_cpu = cpu_tape.make_var(&f32_tensor(&x1_data, &[2, 2]));
    let x2_cpu = cpu_tape.make_var(&f32_tensor(&x2_data, &[2, 2]));
    let loss_cpu = loss_ops::cosine_embedding_loss(
        &x1_cpu,
        &x2_cpu,
        &y,
        0.2,
        fandhe_ai_autodiff::Reduction::Mean,
    )
    .unwrap();
    let grads_cpu = cpu_tape
        .backward(&loss_cpu)
        .expect("backward は成功するはず");
    let dx1_cpu = contiguous_slice(grads_cpu.get(&x1_cpu).unwrap().expect("到達する"));

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x1_naive = naive_tape.make_var(&f32_tensor(&x1_data, &[2, 2]));
    let x2_naive = naive_tape.make_var(&f32_tensor(&x2_data, &[2, 2]));
    let loss_naive = loss_ops::cosine_embedding_loss(
        &x1_naive,
        &x2_naive,
        &y,
        0.2,
        fandhe_ai_autodiff::Reduction::Mean,
    )
    .unwrap();
    let grads_naive = naive_tape
        .backward(&loss_naive)
        .expect("backward は成功するはず");
    let dx1_naive = contiguous_slice(grads_naive.get(&x1_naive).unwrap().expect("到達する"));

    assert_bits_eq(
        "cosine_embedding_loss forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&loss_cpu.to_tensor()),
        &contiguous_slice(&loss_naive.to_tensor()),
    );
    assert_bits_eq(
        "cosine_embedding_loss backward: CpuBackendOps vs NaiveOps",
        &dx1_cpu,
        &dx1_naive,
    );
}

// --- MarginRanking 損失: 属性なし（CPU vs NaiveOps） ---

#[test]
fn cpu_margin_ranking_loss_forward_and_backward_match_naive_reference() {
    let x1_data = [2.0f32, 0.0, -1.0, 3.0];
    let x2_data = [0.5f32, 1.0, 1.5, -2.0];
    let y = f32_tensor(&[1.0, -1.0, 1.0, -1.0], &[4]);

    let cpu_tape = fandhe_ai::tape();
    let x1_cpu = cpu_tape.make_var(&f32_tensor(&x1_data, &[4]));
    let x2_cpu = cpu_tape.make_var(&f32_tensor(&x2_data, &[4]));
    let loss_cpu = loss_ops::margin_ranking_loss(
        &x1_cpu,
        &x2_cpu,
        &y,
        0.3,
        fandhe_ai_autodiff::Reduction::Mean,
    )
    .unwrap();
    let grads_cpu = cpu_tape
        .backward(&loss_cpu)
        .expect("backward は成功するはず");
    let dx1_cpu = contiguous_slice(grads_cpu.get(&x1_cpu).unwrap().expect("到達する"));

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x1_naive = naive_tape.make_var(&f32_tensor(&x1_data, &[4]));
    let x2_naive = naive_tape.make_var(&f32_tensor(&x2_data, &[4]));
    let loss_naive = loss_ops::margin_ranking_loss(
        &x1_naive,
        &x2_naive,
        &y,
        0.3,
        fandhe_ai_autodiff::Reduction::Mean,
    )
    .unwrap();
    let grads_naive = naive_tape
        .backward(&loss_naive)
        .expect("backward は成功するはず");
    let dx1_naive = contiguous_slice(grads_naive.get(&x1_naive).unwrap().expect("到達する"));

    assert_bits_eq(
        "margin_ranking_loss forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&loss_cpu.to_tensor()),
        &contiguous_slice(&loss_naive.to_tensor()),
    );
    assert_bits_eq(
        "margin_ranking_loss backward: CpuBackendOps vs NaiveOps",
        &dx1_cpu,
        &dx1_naive,
    );
}

// --- TripletMargin 損失: 属性なし（CPU vs NaiveOps） ---

#[test]
fn cpu_triplet_margin_loss_forward_and_backward_match_naive_reference() {
    let a_data = [0.2f32, -0.3, 1.1, 0.4];
    let p_data = [1.0f32, 0.5, -0.2, 0.1];
    let n_data = [-0.5f32, 1.2, 0.6, -0.9];
    let options = TripletMarginOptions::default().margin(0.5);

    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&f32_tensor(&a_data, &[2, 2]));
    let p_cpu = cpu_tape.make_var(&f32_tensor(&p_data, &[2, 2]));
    let n_cpu = cpu_tape.make_var(&f32_tensor(&n_data, &[2, 2]));
    let loss_cpu = loss_ops::triplet_margin_loss(
        &a_cpu,
        &p_cpu,
        &n_cpu,
        &options,
        fandhe_ai_autodiff::Reduction::Mean,
    )
    .unwrap();
    let grads_cpu = cpu_tape
        .backward(&loss_cpu)
        .expect("backward は成功するはず");
    let da_cpu = contiguous_slice(grads_cpu.get(&a_cpu).unwrap().expect("到達する"));

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a_naive = naive_tape.make_var(&f32_tensor(&a_data, &[2, 2]));
    let p_naive = naive_tape.make_var(&f32_tensor(&p_data, &[2, 2]));
    let n_naive = naive_tape.make_var(&f32_tensor(&n_data, &[2, 2]));
    let loss_naive = loss_ops::triplet_margin_loss(
        &a_naive,
        &p_naive,
        &n_naive,
        &options,
        fandhe_ai_autodiff::Reduction::Mean,
    )
    .unwrap();
    let grads_naive = naive_tape
        .backward(&loss_naive)
        .expect("backward は成功するはず");
    let da_naive = contiguous_slice(grads_naive.get(&a_naive).unwrap().expect("到達する"));

    assert_bits_eq(
        "triplet_margin_loss forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&loss_cpu.to_tensor()),
        &contiguous_slice(&loss_naive.to_tensor()),
    );
    assert_bits_eq(
        "triplet_margin_loss backward: CpuBackendOps vs NaiveOps",
        &da_cpu,
        &da_naive,
    );
}

// --- PoissonNLL 損失: 属性なし（CPU vs NaiveOps） ---

#[test]
fn cpu_poisson_nll_loss_forward_and_backward_match_naive_reference() {
    let input_data = [0.2f32, -0.3, 0.5, 0.1];
    let target_data = [2.5f32, 0.5, 3.0, 1.3];
    let options = PoissonNllOptions::default().full(true);

    let cpu_tape = fandhe_ai::tape();
    let input_cpu = cpu_tape.make_var(&f32_tensor(&input_data, &[4]));
    let target_cpu = cpu_tape.make_var(&f32_tensor(&target_data, &[4]));
    let loss_cpu = loss_ops::poisson_nll_loss(
        &input_cpu,
        &target_cpu,
        &options,
        fandhe_ai_autodiff::Reduction::Mean,
    )
    .unwrap();
    let grads_cpu = cpu_tape
        .backward(&loss_cpu)
        .expect("backward は成功するはず");
    let dinput_cpu = contiguous_slice(grads_cpu.get(&input_cpu).unwrap().expect("到達する"));

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let input_naive = naive_tape.make_var(&f32_tensor(&input_data, &[4]));
    let target_naive = naive_tape.make_var(&f32_tensor(&target_data, &[4]));
    let loss_naive = loss_ops::poisson_nll_loss(
        &input_naive,
        &target_naive,
        &options,
        fandhe_ai_autodiff::Reduction::Mean,
    )
    .unwrap();
    let grads_naive = naive_tape
        .backward(&loss_naive)
        .expect("backward は成功するはず");
    let dinput_naive = contiguous_slice(grads_naive.get(&input_naive).unwrap().expect("到達する"));

    assert_bits_eq(
        "poisson_nll_loss forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&loss_cpu.to_tensor()),
        &contiguous_slice(&loss_naive.to_tensor()),
    );
    assert_bits_eq(
        "poisson_nll_loss backward: CpuBackendOps vs NaiveOps",
        &dinput_cpu,
        &dinput_naive,
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。REQ-2 複合判定） ---

fn l1_loss_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let pred = tape.make_var(&f32_tensor(&[1.0, -2.0, 3.0, 0.5], &[2, 2]));
    let target = tape.make_var(&f32_tensor(&[0.5, -1.0, 2.5, 1.0], &[2, 2]));
    loss_ops::l1_loss(&pred, &target, fandhe_ai_autodiff::Reduction::Mean)
        .unwrap()
        .to_tensor()
}

fn cross_entropy_loss_with_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&f32_tensor(
        &[1.0, 2.0, 0.5, -1.0, 0.5, 2.0, 0.3, -0.2, 1.1],
        &[3, 3],
    ));
    let targets = i32_tensor(&[2, 0, 1], &[3]);
    let options = CrossEntropyOptions::default()
        .label_smoothing(0.1)
        .ignore_index(1)
        .class_weight(f32_tensor(&[1.5, 0.5, 1.0], &[3]));
    loss_ops::cross_entropy_loss_with(
        &x,
        &targets,
        1,
        fandhe_ai_autodiff::Reduction::Mean,
        &options,
    )
    .unwrap()
    .to_tensor()
}

fn cosine_embedding_loss_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x1 = tape.make_var(&f32_tensor(&[1.0, 2.0, -0.5, 0.3], &[2, 2]));
    let x2 = tape.make_var(&f32_tensor(&[0.4, -0.6, 1.1, -0.9], &[2, 2]));
    let y = f32_tensor(&[1.0, -1.0], &[2]);
    loss_ops::cosine_embedding_loss(&x1, &x2, &y, 0.2, fandhe_ai_autodiff::Reduction::Mean)
        .unwrap()
        .to_tensor()
}

fn margin_ranking_loss_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x1 = tape.make_var(&f32_tensor(&[2.0, 0.0, -1.0, 3.0], &[4]));
    let x2 = tape.make_var(&f32_tensor(&[0.5, 1.0, 1.5, -2.0], &[4]));
    let y = f32_tensor(&[1.0, -1.0, 1.0, -1.0], &[4]);
    loss_ops::margin_ranking_loss(&x1, &x2, &y, 0.3, fandhe_ai_autodiff::Reduction::Mean)
        .unwrap()
        .to_tensor()
}

fn triplet_margin_loss_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let a = tape.make_var(&f32_tensor(&[0.2, -0.3, 1.1, 0.4], &[2, 2]));
    let p = tape.make_var(&f32_tensor(&[1.0, 0.5, -0.2, 0.1], &[2, 2]));
    let n = tape.make_var(&f32_tensor(&[-0.5, 1.2, 0.6, -0.9], &[2, 2]));
    let options = TripletMarginOptions::default().margin(0.5);
    loss_ops::triplet_margin_loss(&a, &p, &n, &options, fandhe_ai_autodiff::Reduction::Mean)
        .unwrap()
        .to_tensor()
}

fn poisson_nll_loss_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let input = tape.make_var(&f32_tensor(&[0.2, -0.3, 0.5, 0.1], &[4]));
    let target = tape.make_var(&f32_tensor(&[2.5, 0.5, 3.0, 1.3], &[4]));
    let options = PoissonNllOptions::default().full(true);
    loss_ops::poisson_nll_loss(
        &input,
        &target,
        &options,
        fandhe_ai_autodiff::Reduction::Mean,
    )
    .unwrap()
    .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`conv3d_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_l1_loss_forward_matches_cpu() {
    let metal_out = l1_loss_forward_on(Device::Metal);
    let cpu_out = l1_loss_forward_on(Device::Cpu);
    assert_parity(
        "l1_loss forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_l1_loss_forward_matches_cpu() {
    let cuda_out = l1_loss_forward_on(Device::Cuda(0));
    let cpu_out = l1_loss_forward_on(Device::Cpu);
    assert_parity(
        "l1_loss forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_cross_entropy_loss_with_forward_matches_cpu() {
    let metal_out = cross_entropy_loss_with_forward_on(Device::Metal);
    let cpu_out = cross_entropy_loss_with_forward_on(Device::Cpu);
    assert_parity(
        "cross_entropy_loss_with forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_cross_entropy_loss_with_forward_matches_cpu() {
    let cuda_out = cross_entropy_loss_with_forward_on(Device::Cuda(0));
    let cpu_out = cross_entropy_loss_with_forward_on(Device::Cpu);
    assert_parity(
        "cross_entropy_loss_with forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_cosine_embedding_loss_forward_matches_cpu() {
    let metal_out = cosine_embedding_loss_forward_on(Device::Metal);
    let cpu_out = cosine_embedding_loss_forward_on(Device::Cpu);
    assert_parity(
        "cosine_embedding_loss forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_cosine_embedding_loss_forward_matches_cpu() {
    let cuda_out = cosine_embedding_loss_forward_on(Device::Cuda(0));
    let cpu_out = cosine_embedding_loss_forward_on(Device::Cpu);
    assert_parity(
        "cosine_embedding_loss forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_margin_ranking_loss_forward_matches_cpu() {
    let metal_out = margin_ranking_loss_forward_on(Device::Metal);
    let cpu_out = margin_ranking_loss_forward_on(Device::Cpu);
    assert_parity(
        "margin_ranking_loss forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_margin_ranking_loss_forward_matches_cpu() {
    let cuda_out = margin_ranking_loss_forward_on(Device::Cuda(0));
    let cpu_out = margin_ranking_loss_forward_on(Device::Cpu);
    assert_parity(
        "margin_ranking_loss forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_triplet_margin_loss_forward_matches_cpu() {
    let metal_out = triplet_margin_loss_forward_on(Device::Metal);
    let cpu_out = triplet_margin_loss_forward_on(Device::Cpu);
    assert_parity(
        "triplet_margin_loss forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_triplet_margin_loss_forward_matches_cpu() {
    let cuda_out = triplet_margin_loss_forward_on(Device::Cuda(0));
    let cpu_out = triplet_margin_loss_forward_on(Device::Cpu);
    assert_parity(
        "triplet_margin_loss forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_poisson_nll_loss_forward_matches_cpu() {
    let metal_out = poisson_nll_loss_forward_on(Device::Metal);
    let cpu_out = poisson_nll_loss_forward_on(Device::Cpu);
    assert_parity(
        "poisson_nll_loss forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_poisson_nll_loss_forward_matches_cpu() {
    let cuda_out = poisson_nll_loss_forward_on(Device::Cuda(0));
    let cpu_out = poisson_nll_loss_forward_on(Device::Cpu);
    assert_parity(
        "poisson_nll_loss forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}
