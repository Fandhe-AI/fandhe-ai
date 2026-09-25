//! `fandhe_ai_autodiff::reduce_ops`（イシュー #2147・facade 非公開の
//! 内部入口。`crates/autodiff/src/reduce_ops.rs` モジュール doc 参照）
//! のバックエンド間 parity テスト（`matrix_ops_backend_parity.rs` と
//! 同型）。
//!
//! `reduce_ops` は facade から再エクスポートされないため、本テストは
//! `fandhe_ai_autodiff::reduce_ops::*` を直接 use する（facade の dev
//! 依存に `fandhe-ai-autodiff` が既に含まれている）。
//!
//! 本ファイルの契約は `prod`・`logsumexp`・`any`・`all`・`norm_p` の
//! 各演算 × {forward, backward} × {CPU vs NaiveOps, CUDA vs CPU, Metal
//! vs CPU} の各セルを埋めることであり、以下の関数名は網羅表と一対一
//! 対応する（`matrix_ops_backend_parity.rs` の教訓: 代表 1 演算で他を
//! 代替しない）。
//!
//! - 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps`〉と
//!   `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`〉の突き合わせ）:
//!   - `any`／`all` forward（出力が厳密に 0.0／1.0 のみのため bit 完全
//!     一致）: `cpu_any_all_forward_bit_matches_naive_reference`
//!   - `any`／`all` backward（`ne` の VJP がゼロを返すため bit 完全
//!     一致）: `cpu_any_all_gradient_is_bit_exact_zero`
//!   - `prod`・`logsumexp`・`norm_p` forward（REQ-2 統一複合判定）:
//!     `cpu_prod_logsumexp_norm_p_forward_matches_naive_reference_within_tolerance`
//!   - `prod`・`logsumexp`・`norm_p` backward（REQ-2 統一複合判定）:
//!     `cpu_prod_logsumexp_norm_p_backward_matches_naive_reference_within_tolerance`
//! - `#[ignore]`（`tape_for(Device::Metal)`〈`cfg(target_os =
//!   "macos")` 限定〉／`tape_for(Device::Cuda(0))` で同じ経路を CPU
//!   tape と比較）: 上記 4 種を `cuda_*`／`metal_*` という接頭辞で対称に
//!   置く（計 8 件）。
//!
//!   実機（DGX Spark GB10／Apple Silicon）への到達手段が本エージェント
//!   実行環境にないため未実施のまま Mac／GB10 セッションへ申し送る
//!   （`docs/perf/logs/reduce-ops-2147/README.md`）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::reduce_ops::{all, any, logsumexp, norm_p, prod};
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

/// `any`／`all` の fixture（0 と非 0 が混在する 2x2）。
fn any_all_fixture() -> Tensor<f32> {
    Tensor::new(vec![0.0, 1.0, -3.0, 0.0], &[2, 2]).expect("test fixture: shape 一致")
}

/// `prod`／`logsumexp`／`norm_p` の fixture（正負混在の 1-D）。
fn reduce_fixture() -> Tensor<f32> {
    Tensor::new(vec![1.5, -2.0, 3.0, -0.5], &[4]).expect("test fixture: shape 一致")
}

/// `any`／`all` forward が CPU（`fandhe_ai::tape()`）と NaiveOps
/// （`fandhe_ai_autodiff::Tape::new()`）で bit 完全一致することを
/// 確認する（出力が厳密に 0.0／1.0 のみで縮約順序に依存しないため）。
#[test]
fn cpu_any_all_forward_bit_matches_naive_reference() {
    let data = any_all_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    for dim in [None, Some(0), Some(1)] {
        assert_eq!(
            f32_bits(&any(&x_cpu, dim).unwrap().to_tensor()),
            f32_bits(&any(&x_naive, dim).unwrap().to_tensor()),
            "any dim={dim:?}"
        );
        assert_eq!(
            f32_bits(&all(&x_cpu, dim).unwrap().to_tensor()),
            f32_bits(&all(&x_naive, dim).unwrap().to_tensor()),
            "all dim={dim:?}"
        );
    }
}

/// `any`／`all` backward（`x.ne(&zero)` の VJP がゼロを返す合成）が
/// CPU と NaiveOps でいずれも厳密な `0.0` の bit 完全一致になることを
/// 確認する。
#[test]
fn cpu_any_all_gradient_is_bit_exact_zero() {
    let data = any_all_fixture();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = any(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let loss_naive = any(&x_naive, None).unwrap();
    let dx_naive = naive_tape
        .backward(&loss_naive)
        .unwrap()
        .get(&x_naive)
        .unwrap()
        .unwrap()
        .clone();
    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_naive), "any gradient");
    assert!(dx_cpu.host_slice().iter().all(|&v| v == 0.0));

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = all(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let loss_naive = all(&x_naive, None).unwrap();
    let dx_naive = naive_tape
        .backward(&loss_naive)
        .unwrap()
        .get(&x_naive)
        .unwrap()
        .unwrap()
        .clone();
    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_naive), "all gradient");
    assert!(dx_cpu.host_slice().iter().all(|&v| v == 0.0));
}

/// `prod`・`logsumexp`・`norm_p` forward が CPU と NaiveOps で REQ-2
/// 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を満たす
/// ことを確認する（`prod` は `cumprod` の `f64` アキュムレータを経由
/// するが縮約順序自体はバックエンドで異なりうるため）。
#[test]
fn cpu_prod_logsumexp_norm_p_forward_matches_naive_reference_within_tolerance() {
    let data = reduce_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    let prod_cpu = prod(&x_cpu, None).unwrap().to_tensor();
    let prod_naive = prod(&x_naive, None).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "prod forward: cpu vs naive",
        prod_cpu.host_slice().as_ref(),
        prod_naive.host_slice().as_ref(),
    );

    let lse_cpu = logsumexp(&x_cpu, None).unwrap().to_tensor();
    let lse_naive = logsumexp(&x_naive, None).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "logsumexp forward: cpu vs naive",
        lse_cpu.host_slice().as_ref(),
        lse_naive.host_slice().as_ref(),
    );

    let norm_cpu = norm_p(&x_cpu, 3.0, None).unwrap().to_tensor();
    let norm_naive = norm_p(&x_naive, 3.0, None).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "norm_p forward: cpu vs naive",
        norm_cpu.host_slice().as_ref(),
        norm_naive.host_slice().as_ref(),
    );
}

/// `prod`・`logsumexp`・`norm_p` backward が CPU と NaiveOps で REQ-2
/// 統一複合判定を満たすことを確認する。
#[test]
fn cpu_prod_logsumexp_norm_p_backward_matches_naive_reference_within_tolerance() {
    let data = reduce_fixture();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = prod(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let loss_naive = prod(&x_naive, None).unwrap();
    let dx_naive = naive_tape
        .backward(&loss_naive)
        .unwrap()
        .get(&x_naive)
        .unwrap()
        .unwrap()
        .clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "prod backward: cpu vs naive",
        dx_cpu.host_slice().as_ref(),
        dx_naive.host_slice().as_ref(),
    );

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = logsumexp(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let loss_naive = logsumexp(&x_naive, None).unwrap();
    let dx_naive = naive_tape
        .backward(&loss_naive)
        .unwrap()
        .get(&x_naive)
        .unwrap()
        .unwrap()
        .clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "logsumexp backward: cpu vs naive",
        dx_cpu.host_slice().as_ref(),
        dx_naive.host_slice().as_ref(),
    );

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = norm_p(&x_cpu, 3.0, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let loss_naive = norm_p(&x_naive, 3.0, None).unwrap();
    let dx_naive = naive_tape
        .backward(&loss_naive)
        .unwrap()
        .get(&x_naive)
        .unwrap()
        .unwrap()
        .clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "norm_p backward: cpu vs naive",
        dx_cpu.host_slice().as_ref(),
        dx_naive.host_slice().as_ref(),
    );
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: Mac／DGX Spark GB10 実機セッションへ
// 申し送る（`docs/perf/logs/reduce-ops-2147/README.md`）。
// ---------------------------------------------------------------------

/// `any`／`all` forward の CPU／Metal 実機比較。CPU 側の同型カバレッジ
/// は `cpu_any_all_forward_bit_matches_naive_reference`。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/reduce-ops-2147/README.md 参照"]
fn metal_any_all_forward_matches_cpu_reference() {
    let data = any_all_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);

    for dim in [None, Some(0), Some(1)] {
        assert_eq!(
            f32_bits(&any(&x_cpu, dim).unwrap().to_tensor()),
            f32_bits(&any(&x_metal, dim).unwrap().to_tensor()),
            "any dim={dim:?}"
        );
        assert_eq!(
            f32_bits(&all(&x_cpu, dim).unwrap().to_tensor()),
            f32_bits(&all(&x_metal, dim).unwrap().to_tensor()),
            "all dim={dim:?}"
        );
    }
}

/// `any`／`all` forward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/reduce-ops-2147/README.md 参照"]
fn cuda_any_all_forward_matches_cpu_reference() {
    let data = any_all_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);

    for dim in [None, Some(0), Some(1)] {
        assert_eq!(
            f32_bits(&any(&x_cpu, dim).unwrap().to_tensor()),
            f32_bits(&any(&x_cuda, dim).unwrap().to_tensor()),
            "any dim={dim:?}"
        );
        assert_eq!(
            f32_bits(&all(&x_cpu, dim).unwrap().to_tensor()),
            f32_bits(&all(&x_cuda, dim).unwrap().to_tensor()),
            "all dim={dim:?}"
        );
    }
}

/// `any`／`all` backward の CPU／Metal 実機比較（いずれも厳密 `0.0`）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/reduce-ops-2147/README.md 参照"]
fn metal_any_all_gradient_is_bit_exact_zero() {
    let data = any_all_fixture();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = any(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);
    let loss_metal = any(&x_metal, None).unwrap();
    let dx_metal = metal_tape
        .backward(&loss_metal)
        .unwrap()
        .get(&x_metal)
        .unwrap()
        .unwrap()
        .clone();
    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_metal), "any gradient");

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = all(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);
    let loss_metal = all(&x_metal, None).unwrap();
    let dx_metal = metal_tape
        .backward(&loss_metal)
        .unwrap()
        .get(&x_metal)
        .unwrap()
        .unwrap()
        .clone();
    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_metal), "all gradient");
}

/// `any`／`all` backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/reduce-ops-2147/README.md 参照"]
fn cuda_any_all_gradient_is_bit_exact_zero() {
    let data = any_all_fixture();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = any(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);
    let loss_cuda = any(&x_cuda, None).unwrap();
    let dx_cuda = cuda_tape
        .backward(&loss_cuda)
        .unwrap()
        .get(&x_cuda)
        .unwrap()
        .unwrap()
        .clone();
    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_cuda), "any gradient");

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = all(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);
    let loss_cuda = all(&x_cuda, None).unwrap();
    let dx_cuda = cuda_tape
        .backward(&loss_cuda)
        .unwrap()
        .get(&x_cuda)
        .unwrap()
        .unwrap()
        .clone();
    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_cuda), "all gradient");
}

/// `prod`・`logsumexp`・`norm_p` forward の CPU／Metal 実機比較（REQ-2
/// 統一複合判定）。CPU 側の同型カバレッジは `cpu_prod_logsumexp_
/// norm_p_forward_matches_naive_reference_within_tolerance`。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/reduce-ops-2147/README.md 参照"]
fn metal_prod_logsumexp_norm_p_forward_matches_cpu_reference() {
    let data = reduce_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);

    let prod_cpu = prod(&x_cpu, None).unwrap().to_tensor();
    let prod_metal = prod(&x_metal, None).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "prod forward: cpu vs metal",
        prod_cpu.host_slice().as_ref(),
        prod_metal.host_slice().as_ref(),
    );

    let lse_cpu = logsumexp(&x_cpu, None).unwrap().to_tensor();
    let lse_metal = logsumexp(&x_metal, None).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "logsumexp forward: cpu vs metal",
        lse_cpu.host_slice().as_ref(),
        lse_metal.host_slice().as_ref(),
    );

    let norm_cpu = norm_p(&x_cpu, 3.0, None).unwrap().to_tensor();
    let norm_metal = norm_p(&x_metal, 3.0, None).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "norm_p forward: cpu vs metal",
        norm_cpu.host_slice().as_ref(),
        norm_metal.host_slice().as_ref(),
    );
}

/// `prod`・`logsumexp`・`norm_p` forward の CPU／CUDA 実機（DGX Spark
/// GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/reduce-ops-2147/README.md 参照"]
fn cuda_prod_logsumexp_norm_p_forward_matches_cpu_reference() {
    let data = reduce_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);

    let prod_cpu = prod(&x_cpu, None).unwrap().to_tensor();
    let prod_cuda = prod(&x_cuda, None).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "prod forward: cpu vs cuda",
        prod_cpu.host_slice().as_ref(),
        prod_cuda.host_slice().as_ref(),
    );

    let lse_cpu = logsumexp(&x_cpu, None).unwrap().to_tensor();
    let lse_cuda = logsumexp(&x_cuda, None).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "logsumexp forward: cpu vs cuda",
        lse_cpu.host_slice().as_ref(),
        lse_cuda.host_slice().as_ref(),
    );

    let norm_cpu = norm_p(&x_cpu, 3.0, None).unwrap().to_tensor();
    let norm_cuda = norm_p(&x_cuda, 3.0, None).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "norm_p forward: cpu vs cuda",
        norm_cpu.host_slice().as_ref(),
        norm_cuda.host_slice().as_ref(),
    );
}

/// `prod`・`logsumexp`・`norm_p` backward の CPU／Metal 実機比較
/// （REQ-2 統一複合判定）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/reduce-ops-2147/README.md 参照"]
fn metal_prod_logsumexp_norm_p_backward_matches_cpu_reference() {
    let data = reduce_fixture();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = prod(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);
    let loss_metal = prod(&x_metal, None).unwrap();
    let dx_metal = metal_tape
        .backward(&loss_metal)
        .unwrap()
        .get(&x_metal)
        .unwrap()
        .unwrap()
        .clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "prod backward: cpu vs metal",
        dx_cpu.host_slice().as_ref(),
        dx_metal.host_slice().as_ref(),
    );

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = logsumexp(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);
    let loss_metal = logsumexp(&x_metal, None).unwrap();
    let dx_metal = metal_tape
        .backward(&loss_metal)
        .unwrap()
        .get(&x_metal)
        .unwrap()
        .unwrap()
        .clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "logsumexp backward: cpu vs metal",
        dx_cpu.host_slice().as_ref(),
        dx_metal.host_slice().as_ref(),
    );

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = norm_p(&x_cpu, 3.0, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);
    let loss_metal = norm_p(&x_metal, 3.0, None).unwrap();
    let dx_metal = metal_tape
        .backward(&loss_metal)
        .unwrap()
        .get(&x_metal)
        .unwrap()
        .unwrap()
        .clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "norm_p backward: cpu vs metal",
        dx_cpu.host_slice().as_ref(),
        dx_metal.host_slice().as_ref(),
    );
}

/// `prod`・`logsumexp`・`norm_p` backward の CPU／CUDA 実機（DGX Spark
/// GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/reduce-ops-2147/README.md 参照"]
fn cuda_prod_logsumexp_norm_p_backward_matches_cpu_reference() {
    let data = reduce_fixture();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = prod(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);
    let loss_cuda = prod(&x_cuda, None).unwrap();
    let dx_cuda = cuda_tape
        .backward(&loss_cuda)
        .unwrap()
        .get(&x_cuda)
        .unwrap()
        .unwrap()
        .clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "prod backward: cpu vs cuda",
        dx_cpu.host_slice().as_ref(),
        dx_cuda.host_slice().as_ref(),
    );

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = logsumexp(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);
    let loss_cuda = logsumexp(&x_cuda, None).unwrap();
    let dx_cuda = cuda_tape
        .backward(&loss_cuda)
        .unwrap()
        .get(&x_cuda)
        .unwrap()
        .unwrap()
        .clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "logsumexp backward: cpu vs cuda",
        dx_cpu.host_slice().as_ref(),
        dx_cuda.host_slice().as_ref(),
    );

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = norm_p(&x_cpu, 3.0, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);
    let loss_cuda = norm_p(&x_cuda, 3.0, None).unwrap();
    let dx_cuda = cuda_tape
        .backward(&loss_cuda)
        .unwrap()
        .get(&x_cuda)
        .unwrap()
        .unwrap()
        .clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "norm_p backward: cpu vs cuda",
        dx_cpu.host_slice().as_ref(),
        dx_cuda.host_slice().as_ref(),
    );
}
