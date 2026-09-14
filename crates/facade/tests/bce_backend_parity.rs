//! `Var::bce_loss`／`bce_with_logits_loss`（イシュー #1737。親イシュー
//! #1609「損失関数の拡張」）の facade 到達経路（既存 `Var` 再エクス
//! ポート経由。`crates/facade/src/lib.rs` への新規 `pub use`／`pub fn`
//! は追加していない ── `docs/compat-api-scope.md` §5 範囲拡張手続きの
//! 対象外）の受け入れ条件対応テスト。`activation_gelu_softplus_
//! backend_parity.rs`（#1713）と同型の構成を踏襲する。
//!
//! `BackendOps::bce_loss`／`bce_loss_backward`（CPU／CUDA／Metal）・
//! `Op::BceLoss`・`grad::vjp` は本イシュー内で実装済みのため、本
//! ファイルは facade の `fandhe_ai::tape()`／`tape_for(Device::..)` から
//! 各バックエンドの融合カーネルへ実際に到達できることのみを検証する
//! （新規カーネル実装は含まない）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps::bce_loss`。融合
//!   カーネル）と `fandhe_ai_autodiff::Tape::new()`（`NaiveOps` →
//!   `eval::bce_loss` フォールバック）で forward／backward を突き合わ
//!   せる。CPU 融合カーネルは決定的固定チャンク累積・ホスト参照実装は
//!   単純逐次累積と丸め手順が異なるため（`mse_loss_fusion.rs` と同じ
//!   非主張）、REQ-2 複合判定（`assert_parity`）で検証する。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の forward／backward
//!   を CPU tape と REQ-2 複合判定で突き合わせる。本エージェント実行
//!   環境に実機への到達手段がないため未実測のまま Mac／GB10 セッション
//!   へ申し送る。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::{Reduction, Var};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`activation_gelu_softplus_
/// backend_parity.rs::VarSource` と同じ理由・同じ構成）。
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

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

/// `(1e-3, 1 - 1e-3)` 開区間の確率入力（`Probabilities` 用。`[0, 1]`
/// 範囲検査を必ず満たす）。
fn probability_leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = Xorshift64Star::new(seed)
        .fill_vec(numel)
        .into_iter()
        .map(|v| v.clamp(1e-3, 1.0 - 1e-3))
        .collect();
    Tensor::new(data, shape).expect("probability_leaf: shape 一致")
}

/// `[-4, 4)` の logits 入力（`Logits` 用。範囲制約なし）。
fn logits_leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = Xorshift64Star::new(seed)
        .fill_vec(numel)
        .into_iter()
        .map(|v| v * 8.0 - 4.0)
        .collect();
    Tensor::new(data, shape).expect("logits_leaf: shape 一致")
}

/// `{0, 1}` の 2 値ラベル（`target`）。
fn label_leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = Xorshift64Star::new(seed)
        .fill_vec(numel)
        .into_iter()
        .map(|v| if v < 0.5 { 0.0 } else { 1.0 })
        .collect();
    Tensor::new(data, shape).expect("label_leaf: shape 一致")
}

// --- forward/backward parity（属性なし: CPU 融合 vs NaiveOps フォール
// バック。REQ-2 複合判定） ---

#[test]
fn cpu_bce_loss_probabilities_matches_naive_reference() {
    let shape = [4usize, 3];
    let input_val = probability_leaf(41, &shape);
    let target_val = label_leaf(43, &shape);

    let cpu_tape = fandhe_ai::tape();
    let out_cpu = cpu_tape
        .make_var(&input_val)
        .bce_loss(&cpu_tape.make_var(&target_val), Reduction::Mean)
        .unwrap()
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let out_naive = naive_tape
        .make_var(&input_val)
        .bce_loss(&naive_tape.make_var(&target_val), Reduction::Mean)
        .unwrap()
        .to_tensor();

    assert_parity(
        "bce_loss(probabilities,mean) forward: cpu fused vs naive fallback",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

#[test]
fn cpu_bce_loss_probabilities_backward_matches_naive_reference() {
    let shape = [4usize, 3];
    let input_val = probability_leaf(45, &shape);
    let target_val = label_leaf(47, &shape);

    let cpu_tape = fandhe_ai::tape();
    let cpu_input = cpu_tape.make_var(&input_val);
    let cpu_target = cpu_tape.make_var(&target_val);
    let cpu_loss = cpu_input.bce_loss(&cpu_target, Reduction::Sum).unwrap();
    let cpu_grads = cpu_tape.backward(&cpu_loss).unwrap();
    let cpu_dinput = cpu_grads.get(&cpu_input).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_input = naive_tape.make_var(&input_val);
    let naive_target = naive_tape.make_var(&target_val);
    let naive_loss = naive_input.bce_loss(&naive_target, Reduction::Sum).unwrap();
    let naive_grads = naive_tape.backward(&naive_loss).unwrap();
    let naive_dinput = naive_grads.get(&naive_input).unwrap().expect("到達する");

    assert_parity(
        "bce_loss(probabilities,sum) backward dInput: cpu fused vs naive fallback",
        &contiguous_slice(cpu_dinput),
        &contiguous_slice(naive_dinput),
    );
}

#[test]
fn cpu_bce_with_logits_loss_matches_naive_reference() {
    let shape = [4usize, 3];
    let input_val = logits_leaf(49, &shape);
    let target_val = label_leaf(51, &shape);

    let cpu_tape = fandhe_ai::tape();
    let out_cpu = cpu_tape
        .make_var(&input_val)
        .bce_with_logits_loss(&cpu_tape.make_var(&target_val), Reduction::Mean)
        .unwrap()
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let out_naive = naive_tape
        .make_var(&input_val)
        .bce_with_logits_loss(&naive_tape.make_var(&target_val), Reduction::Mean)
        .unwrap()
        .to_tensor();

    assert_parity(
        "bce_with_logits_loss(mean) forward: cpu fused vs naive fallback",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

// --- 実機（Metal／CUDA）parity。本エージェント実行環境に実機への到達
// 手段がないため未実測のまま申し送る（`#[ignore]`）。 ---

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`activation_gelu_softplus_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_bce_loss_probabilities_matches_cpu() {
    let shape = [4usize, 3];
    let input_val = probability_leaf(53, &shape);
    let target_val = label_leaf(55, &shape);

    let cpu_tape = fandhe_ai::tape();
    let out_cpu = cpu_tape
        .make_var(&input_val)
        .bce_loss(&cpu_tape.make_var(&target_val), Reduction::Mean)
        .unwrap()
        .to_tensor();

    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let out_metal = metal_tape
        .make_var(&input_val)
        .bce_loss(&metal_tape.make_var(&target_val), Reduction::Mean)
        .unwrap()
        .to_tensor();

    assert_parity(
        "bce_loss(probabilities,mean) forward: cpu vs metal",
        &contiguous_slice(&out_metal),
        &contiguous_slice(&out_cpu),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）依存。CI では実行しない"]
fn cuda_bce_with_logits_loss_matches_cpu() {
    let shape = [4usize, 3];
    let input_val = logits_leaf(57, &shape);
    let target_val = label_leaf(59, &shape);

    let cpu_tape = fandhe_ai::tape();
    let out_cpu = cpu_tape
        .make_var(&input_val)
        .bce_with_logits_loss(&cpu_tape.make_var(&target_val), Reduction::Sum)
        .unwrap()
        .to_tensor();

    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let out_cuda = cuda_tape
        .make_var(&input_val)
        .bce_with_logits_loss(&cuda_tape.make_var(&target_val), Reduction::Sum)
        .unwrap()
        .to_tensor();

    assert_parity(
        "bce_with_logits_loss(sum) forward: cpu vs cuda",
        &contiguous_slice(&out_cuda),
        &contiguous_slice(&out_cpu),
    );
}
