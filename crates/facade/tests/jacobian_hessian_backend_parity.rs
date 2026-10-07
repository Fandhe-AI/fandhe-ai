//! `fandhe_ai_autodiff::jacobian_ops::{jacobian, hessian}`（イシュー #2670。facade 非公開のため
//! `fandhe_ai_autodiff` を直接 use する。`crates/autodiff/src/jacobian_ops.rs` モジュール doc 参照）の
//! バックエンド間 parity テスト（`binning_ops_backend_parity.rs` と同型）。
//!
//! 属性なし: 実 `CpuBackendOps` を結線した tape と `Tape::new()`（`NaiveOps`）を突き合わせ、REQ-2
//! 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）で検証する。jacobian・hessian は
//! 新しい `BackendOps` メソッドを持たず既存 Op（matmul・tanh・sigmoid 等）の合成のみで到達するため、
//! ここで確認するのは「既存カーネルの新しい呼び出し形（backward の繰り返し・子テープ上の VJP）が
//! バックエンド間で一致すること」である。手計算の期待値も 1 件固定する。
//!
//! `#[ignore]`（CUDA／Metal〈`cfg(target_os = "macos")` 限定〉の `BackendOps` を結線した tape と CPU
//! tape の比較）: 実機への到達手段が本エージェント実行環境にないため未実施のまま GB10／Mac
//! セッションへ申し送る（`docs/perf/logs/jacobian-hessian-2670/README.md`）。形状は小さく、Metal
//! split-K が発動する形状は使わない。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::jacobian_ops::{hessian, jacobian};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

struct Outs {
    jac_shape: Vec<usize>,
    jac: Vec<f32>,
    hess_shape: Vec<usize>,
    hess: Vec<f32>,
}

fn compute(tape: &Tape, child: &Tape) -> Outs {
    let w1 = tape.var_no_grad(&t(
        (0..12).map(|i| 0.1 * (i as f32) - 0.5).collect(),
        &[3, 4],
    ));
    let w2 = tape.var_no_grad(&t(
        (0..8).map(|i| 0.2 - 0.05 * (i as f32)).collect(),
        &[4, 2],
    ));
    let x = tape.var(&t(vec![0.3, -0.7, 1.1, 0.5, 0.2, -0.4], &[2, 3]));
    let y = x.matmul(&w1).unwrap().tanh().matmul(&w2).unwrap();
    let jac = jacobian(tape, &y, &x).unwrap();

    let loss = y.sigmoid().sum(None).unwrap();
    let hess = hessian(tape, &loss, &x, child).unwrap();
    Outs {
        jac_shape: jac.shape().to_vec(),
        jac: jac.host_slice().into_owned(),
        hess_shape: hess.shape().to_vec(),
        hess: hess.host_slice().into_owned(),
    }
}

fn cpu_tape() -> Tape {
    Tape::new_with_ops(Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()))
}

fn assert_outs_match(label: &str, a: &Outs, b: &Outs) {
    assert_eq!(a.jac_shape, b.jac_shape, "{label}: jacobian shape");
    assert_eq!(a.hess_shape, b.hess_shape, "{label}: hessian shape");
    fandhe_ai_backend_cpu::parity::assert_parity(&format!("{label}: jacobian"), &a.jac, &b.jac);
    fandhe_ai_backend_cpu::parity::assert_parity(&format!("{label}: hessian"), &a.hess, &b.hess);
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = compute(&cpu_tape(), &cpu_tape());
    let naive = compute(&Tape::new(), &Tape::new());
    assert_eq!(cpu.jac_shape, vec![2, 2, 2, 3]);
    assert_eq!(cpu.hess_shape, vec![2, 3, 2, 3]);
    assert_outs_match("cpu vs naive", &cpu, &naive);
}

/// 手計算の期待値を 1 件固定する（両経路が同じ誤りで一致していないことの確認）。
/// `y = x ⊙ x`・`x = [1, 2]` の jacobian は `diag(2x) = [[2, 0], [0, 4]]`、
/// `loss = Σ x³`（`x ⊙ x ⊙ x` の総和）の hessian は `diag(6x) = [[6, 0], [0, 12]]`。
#[test]
fn cpu_matches_hand_computed_values() {
    let tape = cpu_tape();
    let child = cpu_tape();
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let sq = x.mul(&x).unwrap();
    let jac = jacobian(&tape, &sq, &x).unwrap();
    assert_eq!(jac.host_slice().as_ref(), &[2.0, 0.0, 0.0, 4.0]);
    let loss = sq.mul(&x).unwrap().sum(None).unwrap();
    let h = hessian(&tape, &loss, &x, &child).unwrap();
    assert_eq!(h.host_slice().as_ref(), &[6.0, 0.0, 0.0, 12.0]);
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/jacobian-hessian-2670/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(make_ops: impl Fn() -> Box<dyn BackendOps + Send>, label: &str) {
    let cpu = compute(&cpu_tape(), &cpu_tape());
    let dev = compute(
        &Tape::new_with_ops(make_ops()),
        &Tape::new_with_ops(make_ops()),
    );
    assert_outs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// jacobian・hessian の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/jacobian-hessian-2670/README.md 参照"]
fn metal_jacobian_hessian_match_cpu_reference() {
    assert_device_matches_cpu(
        || Box::new(fandhe_ai_backend_metal::MetalBackendOps::new()),
        "metal",
    );
}

/// jacobian・hessian の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/jacobian-hessian-2670/README.md 参照"]
fn cuda_jacobian_hessian_match_cpu_reference() {
    assert_device_matches_cpu(
        || Box::new(fandhe_ai_backend_cuda::CudaBackendOps::new(0)),
        "cuda",
    );
}
