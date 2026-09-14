//! `Var::nll_loss`／`kl_div_loss`／`kl_div_loss_with_log_target`（イシュー
//! #1738。親イシュー #1609「損失関数の拡張」）の facade 到達経路（既存
//! `Var` 再エクスポート経由。`crates/facade/src/lib.rs` への新規
//! `pub use`／`pub fn` は追加していない ── `docs/compat-api-scope.md`
//! §5 範囲拡張手続きの対象外）の受け入れ条件対応テスト。
//! `bce_backend_parity.rs`（イシュー #1737）と同型の構成を踏襲する。
//!
//! `BackendOps::nll_loss`／`nll_loss_backward`・`kl_div_loss`／
//! `kl_div_loss_backward`（CPU）・`Op::NllLoss`／`Op::KlDivLoss`・
//! `grad::vjp` は本イシュー内で実装済みのため、本ファイルは facade の
//! `fandhe_ai::tape()`／`tape_for(Device::..)` から各バックエンドの
//! 融合カーネルへ実際に到達できることのみを検証する（新規カーネル
//! 実装は含まない）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps::nll_loss`／
//!   `kl_div_loss`。融合カーネル）と `fandhe_ai_autodiff::Tape::new()`
//!   （`NaiveOps` → `eval::nll_loss`／`kl_div_loss` フォールバック）で
//!   forward／backward を突き合わせる。CPU 融合カーネルは決定的固定
//!   チャンク累積・ホスト参照実装は単純逐次累積と丸め手順が異なるため
//!   （`mse_loss_fusion.rs` と同じ非主張）、REQ-2 複合判定
//!   （`assert_parity`）で検証する。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の forward／backward
//!   を CPU tape と REQ-2 複合判定で突き合わせる。本エージェント実行
//!   環境に実機への到達手段がないため未実測のまま Mac／GB10 セッション
//!   へ申し送る。
//!
//! **facade 未再エクスポートの既存ギャップ**: `Reduction` は
//! `fandhe_ai::` から到達不能（MSE／CrossEntropy／BCE 共通の既存
//! ギャップ・`docs/compat-api-scope.md` 参照）。`bce_backend_parity.rs`
//! と同じく `fandhe_ai_autodiff::Reduction` を import する（本 issue
//! では是正しない）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::{Reduction, Var};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`bce_backend_parity.rs::
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

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

/// 負値中心の入力（log 確率想定。範囲検査は課されない）。
fn log_prob_leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = Xorshift64Star::new(seed)
        .fill_vec(numel)
        .into_iter()
        .map(|v| -(v * 3.0 + 0.05))
        .collect();
    Tensor::new(data, shape).expect("log_prob_leaf: shape 一致")
}

/// `(1e-3, 1 - 1e-3)` 開区間の確率入力（KLDiv `Probabilities` の
/// `target` 用。`t == 0` 分岐を避け中央差分・backward 双方を安定させる）。
fn probability_leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = Xorshift64Star::new(seed)
        .fill_vec(numel)
        .into_iter()
        .map(|v| v.clamp(1e-3, 1.0 - 1e-3))
        .collect();
    Tensor::new(data, shape).expect("probability_leaf: shape 一致")
}

/// `[0, num_classes)` の正解クラス添字（`targets`）。
fn targets_leaf(seed: u64, shape: &[usize], num_classes: usize) -> Tensor<i32> {
    let numel: usize = shape.iter().product();
    let data: Vec<i32> = Xorshift64Star::new(seed)
        .fill_vec(numel)
        .into_iter()
        .map(|v| ((v * num_classes as f32) as usize).min(num_classes - 1) as i32)
        .collect();
    Tensor::new(data, shape).expect("targets_leaf: shape 一致")
}

// --- forward/backward parity（属性なし: CPU 融合 vs NaiveOps フォール
// バック。REQ-2 複合判定） ---

#[test]
fn cpu_nll_loss_matches_naive_reference() {
    let input_shape = [4usize, 3];
    let num_classes = 3;
    let class_dim = 1;
    let input_val = log_prob_leaf(61, &input_shape);
    let targets_val = targets_leaf(63, &[4], num_classes);

    let cpu_tape = fandhe_ai::tape();
    let out_cpu = cpu_tape
        .make_var(&input_val)
        .nll_loss(&targets_val, class_dim, Reduction::Mean)
        .unwrap()
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let out_naive = naive_tape
        .make_var(&input_val)
        .nll_loss(&targets_val, class_dim, Reduction::Mean)
        .unwrap()
        .to_tensor();

    assert_parity(
        "nll_loss(mean) forward: cpu fused vs naive fallback",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

#[test]
fn cpu_nll_loss_backward_matches_naive_reference() {
    let input_shape = [4usize, 3];
    let num_classes = 3;
    let class_dim = 1;
    let input_val = log_prob_leaf(65, &input_shape);
    let targets_val = targets_leaf(67, &[4], num_classes);

    let cpu_tape = fandhe_ai::tape();
    let cpu_input = cpu_tape.make_var(&input_val);
    let cpu_loss = cpu_input
        .nll_loss(&targets_val, class_dim, Reduction::Sum)
        .unwrap();
    let cpu_grads = cpu_tape.backward(&cpu_loss).unwrap();
    let cpu_dinput = cpu_grads.get(&cpu_input).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_input = naive_tape.make_var(&input_val);
    let naive_loss = naive_input
        .nll_loss(&targets_val, class_dim, Reduction::Sum)
        .unwrap();
    let naive_grads = naive_tape.backward(&naive_loss).unwrap();
    let naive_dinput = naive_grads.get(&naive_input).unwrap().expect("到達する");

    assert_parity(
        "nll_loss(sum) backward dInput: cpu fused vs naive fallback",
        &contiguous_slice(cpu_dinput),
        &contiguous_slice(naive_dinput),
    );
}

#[test]
fn cpu_kl_div_loss_matches_naive_reference() {
    let shape = [4usize, 3];
    let input_val = log_prob_leaf(69, &shape);
    let target_val = probability_leaf(71, &shape);

    let cpu_tape = fandhe_ai::tape();
    let out_cpu = cpu_tape
        .make_var(&input_val)
        .kl_div_loss(&cpu_tape.make_var(&target_val), Reduction::Mean)
        .unwrap()
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let out_naive = naive_tape
        .make_var(&input_val)
        .kl_div_loss(&naive_tape.make_var(&target_val), Reduction::Mean)
        .unwrap()
        .to_tensor();

    assert_parity(
        "kl_div_loss(mean) forward: cpu fused vs naive fallback",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

#[test]
fn cpu_kl_div_loss_backward_matches_naive_reference() {
    let shape = [4usize, 3];
    let input_val = log_prob_leaf(73, &shape);
    let target_val = probability_leaf(75, &shape);

    let cpu_tape = fandhe_ai::tape();
    let cpu_input = cpu_tape.make_var(&input_val);
    let cpu_target = cpu_tape.make_var(&target_val);
    let cpu_loss = cpu_input.kl_div_loss(&cpu_target, Reduction::Sum).unwrap();
    let cpu_grads = cpu_tape.backward(&cpu_loss).unwrap();
    let cpu_dinput = cpu_grads.get(&cpu_input).unwrap().expect("到達する");
    let cpu_dtarget = cpu_grads.get(&cpu_target).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_input = naive_tape.make_var(&input_val);
    let naive_target = naive_tape.make_var(&target_val);
    let naive_loss = naive_input
        .kl_div_loss(&naive_target, Reduction::Sum)
        .unwrap();
    let naive_grads = naive_tape.backward(&naive_loss).unwrap();
    let naive_dinput = naive_grads.get(&naive_input).unwrap().expect("到達する");
    let naive_dtarget = naive_grads.get(&naive_target).unwrap().expect("到達する");

    assert_parity(
        "kl_div_loss(sum) backward dInput: cpu fused vs naive fallback",
        &contiguous_slice(cpu_dinput),
        &contiguous_slice(naive_dinput),
    );
    assert_parity(
        "kl_div_loss(sum) backward dTarget: cpu fused vs naive fallback",
        &contiguous_slice(cpu_dtarget),
        &contiguous_slice(naive_dtarget),
    );
}

#[test]
fn cpu_kl_div_loss_with_log_target_matches_naive_reference() {
    let shape = [4usize, 3];
    let input_val = log_prob_leaf(77, &shape);
    let target_val = log_prob_leaf(79, &shape);

    let cpu_tape = fandhe_ai::tape();
    let out_cpu = cpu_tape
        .make_var(&input_val)
        .kl_div_loss_with_log_target(&cpu_tape.make_var(&target_val), Reduction::Mean)
        .unwrap()
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let out_naive = naive_tape
        .make_var(&input_val)
        .kl_div_loss_with_log_target(&naive_tape.make_var(&target_val), Reduction::Mean)
        .unwrap()
        .to_tensor();

    assert_parity(
        "kl_div_loss_with_log_target(mean) forward: cpu fused vs naive fallback",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

// --- 実機（Metal／CUDA）parity。本エージェント実行環境に実機への到達
// 手段がないため未実測のまま申し送る（`#[ignore]`）。 ---

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`bce_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_nll_loss_matches_cpu() {
    let input_shape = [4usize, 3];
    let num_classes = 3;
    let class_dim = 1;
    let input_val = log_prob_leaf(81, &input_shape);
    let targets_val = targets_leaf(83, &[4], num_classes);

    let cpu_tape = fandhe_ai::tape();
    let out_cpu = cpu_tape
        .make_var(&input_val)
        .nll_loss(&targets_val, class_dim, Reduction::Mean)
        .unwrap()
        .to_tensor();

    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let out_metal = metal_tape
        .make_var(&input_val)
        .nll_loss(&targets_val, class_dim, Reduction::Mean)
        .unwrap()
        .to_tensor();

    assert_parity(
        "nll_loss(mean) forward: cpu vs metal",
        &contiguous_slice(&out_metal),
        &contiguous_slice(&out_cpu),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）依存。CI では実行しない"]
fn cuda_kl_div_loss_matches_cpu() {
    let shape = [4usize, 3];
    let input_val = log_prob_leaf(85, &shape);
    let target_val = probability_leaf(87, &shape);

    let cpu_tape = fandhe_ai::tape();
    let out_cpu = cpu_tape
        .make_var(&input_val)
        .kl_div_loss(&cpu_tape.make_var(&target_val), Reduction::Sum)
        .unwrap()
        .to_tensor();

    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let out_cuda = cuda_tape
        .make_var(&input_val)
        .kl_div_loss(&cuda_tape.make_var(&target_val), Reduction::Sum)
        .unwrap()
        .to_tensor();

    assert_parity(
        "kl_div_loss(sum) forward: cpu vs cuda",
        &contiguous_slice(&out_cuda),
        &contiguous_slice(&out_cpu),
    );
}
