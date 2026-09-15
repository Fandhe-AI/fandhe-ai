//! BatchNorm1d／2d（イシュー #1732・親 #1608）の facade 到達経路
//! （既存 `Var` 再エクスポート経由。`crates/facade/src/` は無変更——
//! `docs/compat-api-scope.md` §1.2「facade 到達経路は既存 `Var`
//! 再エクスポート経由」）の受け入れ条件対応テスト。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`。`BackendOps::
//!   batch_norm_train`／`batch_norm_infer` を実機カーネルでオーバー
//!   ライド済み）上の `Var::batch_norm`／`batch_norm_infer`
//!   forward・backward を、無引数 `fandhe_ai_autodiff::Tape::new()`
//!   （`NaiveOps` → ホスト参照実装 `eval::batch_norm_train_channels`／
//!   `batch_norm_infer_channels` へフォールバックする経路）と REQ-2
//!   統一複合判定で突き合わせる（`norm_backend_parity.rs`〈イシュー
//!   #1596〉と同型）。
//! - `#[ignore]`: `fandhe_ai::tape_for(Device::Metal)`／
//!   `tape_for(Device::Cuda(0))` 上の同経路を CPU tape と突き合わせる
//!   （実機必須。#1736／#1735 が実装する CUDA／Metal カーネル向けの
//!   スキャフォールドとして本ファイルへ追加する）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`norm_backend_parity.rs`
/// の `VarSource` と同じ理由・同じ構成）。
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

/// 決定的な固定値（`norm_backend_parity.rs` と同じ方針。facade へ
/// dev-dep を増やさないため乱数ユーティリティは使わない）。[N=4, C=3]
/// （rank 2。BatchNorm1d の非空間入力）。
fn leaf() -> Tensor<f32> {
    Tensor::new(
        vec![
            0.5, -1.2, 2.0, -0.3, 1.1, -0.7, 0.2, -2.1, 0.9, -0.4, 1.3, -0.6,
        ],
        &[4, 3],
    )
    .expect("leaf: shape 一致")
}

fn weight() -> Tensor<f32> {
    Tensor::new(vec![1.5, -0.8, 1.0], &[3]).expect("weight: shape 一致")
}

fn bias() -> Tensor<f32> {
    Tensor::new(vec![0.1, -0.2, 0.3], &[3]).expect("bias: shape 一致")
}

fn running_mean() -> Tensor<f32> {
    Tensor::new(vec![0.2, -0.1, 0.5], &[3]).expect("running_mean: shape 一致")
}

fn running_var() -> Tensor<f32> {
    Tensor::new(vec![1.2, 0.8, 2.0], &[3]).expect("running_var: shape 一致")
}

// --- train モード（バッチ統計） ---

/// `batch_norm(weight, bias, eps)`（train）forward の CPU（融合カーネル
/// 経由）と NaiveOps（ホスト参照実装フォールバック）の parity。
#[test]
fn cpu_batch_norm_train_forward_matches_naive_reference() {
    let cpu_tape = fandhe_ai::tape();
    let w_cpu = cpu_tape.make_var(&weight());
    let b_cpu = cpu_tape.make_var(&bias());
    let cpu_out = cpu_tape
        .make_var(&leaf())
        .batch_norm(Some(&w_cpu), Some(&b_cpu), 1e-5)
        .expect("batch_norm: weight/bias shape [3] は c=3 と一致")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let w_naive = naive_tape.make_var(&weight());
    let b_naive = naive_tape.make_var(&bias());
    let naive_out = naive_tape
        .make_var(&leaf())
        .batch_norm(Some(&w_naive), Some(&b_naive), 1e-5)
        .expect("batch_norm: weight/bias shape [3] は c=3 と一致")
        .to_tensor();

    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::batch_norm_train）vs NaiveOps フォールバック",
        cpu_out.as_slice().expect("contiguous"),
        naive_out.as_slice().expect("contiguous"),
    );
}

/// `matmul → batch_norm(train) → mse_loss` backward（`dW`〈batch_norm
/// weight〉）の CPU（`Op::BatchNorm` の VJP。融合カーネル forward +
/// ホスト VJP）と NaiveOps（forward・backward ともホスト参照実装）の
/// parity。
#[test]
fn cpu_batch_norm_train_backward_matches_naive_reference() {
    let w_lin = Tensor::new(
        vec![0.5, -0.3, 0.2, 0.7, -0.6, 0.1, 0.4, -0.2, 0.3],
        &[3, 3],
    )
    .expect("valid tensor");
    let target = Tensor::new(
        vec![
            0.2, 0.6, 0.3, 0.4, -0.1, 0.5, -0.3, 0.2, 0.1, 0.1, -0.2, 0.3,
        ],
        &[4, 3],
    )
    .expect("valid tensor");

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf());
    let w_lin_cpu = cpu_tape.make_var(&w_lin);
    let w_bn_cpu = cpu_tape.make_var(&weight());
    let b_bn_cpu = cpu_tape.make_var(&bias());
    let t_cpu = cpu_tape.make_var(&target);
    let y_cpu = x_cpu
        .matmul(&w_lin_cpu)
        .unwrap()
        .batch_norm(Some(&w_bn_cpu), Some(&b_bn_cpu), 1e-5)
        .unwrap();
    let loss_cpu = y_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dw_cpu = grads_cpu.get(&w_bn_cpu).unwrap().expect("到達する");
    let db_cpu = grads_cpu.get(&b_bn_cpu).unwrap().expect("到達する");
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf());
    let w_lin_naive = naive_tape.make_var(&w_lin);
    let w_bn_naive = naive_tape.make_var(&weight());
    let b_bn_naive = naive_tape.make_var(&bias());
    let t_naive = naive_tape.make_var(&target);
    let y_naive = x_naive
        .matmul(&w_lin_naive)
        .unwrap()
        .batch_norm(Some(&w_bn_naive), Some(&b_bn_naive), 1e-5)
        .unwrap();
    let loss_naive = y_naive.mse_loss(&t_naive).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dw_naive = grads_naive.get(&w_bn_naive).unwrap().expect("到達する");
    let db_naive = grads_naive.get(&b_bn_naive).unwrap().expect("到達する");
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

    assert_parity(
        "batch_norm train dW",
        dw_cpu.as_slice().expect("contiguous"),
        dw_naive.as_slice().expect("contiguous"),
    );
    assert_parity(
        "batch_norm train db",
        db_cpu.as_slice().expect("contiguous"),
        db_naive.as_slice().expect("contiguous"),
    );
    assert_parity(
        "batch_norm train dx",
        dx_cpu.as_slice().expect("contiguous"),
        dx_naive.as_slice().expect("contiguous"),
    );
}

// --- eval モード（固定統計） ---

/// `batch_norm_infer(weight, bias, running_mean, running_var, eps)`
/// forward の CPU と NaiveOps の parity。
#[test]
fn cpu_batch_norm_infer_forward_matches_naive_reference() {
    let cpu_tape = fandhe_ai::tape();
    let w_cpu = cpu_tape.make_var(&weight());
    let b_cpu = cpu_tape.make_var(&bias());
    let cpu_out = cpu_tape
        .make_var(&leaf())
        .batch_norm_infer(
            Some(&w_cpu),
            Some(&b_cpu),
            &running_mean(),
            &running_var(),
            1e-5,
        )
        .expect("batch_norm_infer: shape 一致")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let w_naive = naive_tape.make_var(&weight());
    let b_naive = naive_tape.make_var(&bias());
    let naive_out = naive_tape
        .make_var(&leaf())
        .batch_norm_infer(
            Some(&w_naive),
            Some(&b_naive),
            &running_mean(),
            &running_var(),
            1e-5,
        )
        .expect("batch_norm_infer: shape 一致")
        .to_tensor();

    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::batch_norm_infer）vs NaiveOps フォールバック",
        cpu_out.as_slice().expect("contiguous"),
        naive_out.as_slice().expect("contiguous"),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---
//
// `BackendOps::batch_norm_train`／`batch_norm_infer` は本 issue
// （#1732）時点では CPU のみが実装済みで、Metal／CUDA は既定
// `Unsupported` のままホスト参照実装へフォールバックする（`Var::
// batch_norm` の判定規律）。したがって以下は現時点では「同じホスト
// フォールバック経路同士の比較」に留まるが、#1736／#1735 が実機
// カーネルで `BackendOps::batch_norm_train`／`batch_norm_infer` を
// オーバーライドした後は本ファイルを変更せずそのまま実機 parity
// テストとして機能する（`norm_backend_parity.rs` と同じスキャフォー
// ルド方針）。

fn batch_norm_train_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let w = tape.make_var(&weight());
    let b = tape.make_var(&bias());
    tape.make_var(&leaf())
        .batch_norm(Some(&w), Some(&b), 1e-5)
        .expect("batch_norm: weight/bias shape [3] は c=3 と一致")
        .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`norm_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_batch_norm_train_forward_matches_cpu() {
    let metal_out = batch_norm_train_forward_on(Device::Metal);
    let cpu_out = batch_norm_train_forward_on(Device::Cpu);

    assert_parity(
        "batch_norm train forward: Metal tape_for vs CPU tape_for",
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_batch_norm_train_forward_matches_cpu() {
    let cuda_out = batch_norm_train_forward_on(Device::Cuda(0));
    let cpu_out = batch_norm_train_forward_on(Device::Cpu);

    assert_parity(
        "batch_norm train forward: CUDA tape_for vs CPU tape_for",
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}
