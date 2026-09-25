//! `fandhe_ai_autodiff::linalg_ops`（イシュー #2150・facade 非公開の
//! 内部入口。`crates/autodiff/src/linalg_ops.rs` モジュール doc 参照）
//! のバックエンド間 parity テスト（`reduce_ops_backend_parity.rs` と
//! 同型）。
//!
//! `linalg_ops` は facade から再エクスポートされないため、本テストは
//! `fandhe_ai_autodiff::linalg_ops::*` を直接 use する。
//!
//! 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps`〉と
//! `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`〉の突き合わせ）:
//! `eigh`・`slogdet`・`pinv`・`matrix_rank`・`lstsq` の forward・
//! backward を REQ-2 統一複合判定（`matrix_rank`／`slogdet` の符号は
//! bit 一致）で検証する。CPU 本番経路の特異入力での
//! `AutodiffError::InvalidArgument` も確認する。
//!
//! `#[ignore]`（`tape_for(Device::Metal)`〈`cfg(target_os = "macos")`
//! 限定〉／`tape_for(Device::Cuda(0))` で同じ経路を CPU tape と比較）:
//! 実機（DGX Spark GB10／Apple Silicon）への到達手段が本エージェント
//! 実行環境にないため未実施のまま Mac／GB10 セッションへ申し送る
//! （`docs/perf/logs/linalg-ops-2150/README.md`）。CUDA／Metal はいずれも
//! `linalg_*` の GPU カーネルを持たないため（`BackendOps` 既定
//! `Unsupported`）、この比較は「ホストへのフォールバック経路が CPU
//! tape と同じ結果になること」を確認する（GPU カーネル自体の parity
//! ではない）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::linalg_ops::{eigh, lstsq, matrix_rank, pinv, slogdet};
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

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn assert_forward_parity(label: &str, cpu: &Tensor<f32>, naive: &Tensor<f32>) {
    fandhe_ai_backend_cpu::parity::assert_parity(
        label,
        cpu.host_slice().as_ref(),
        naive.host_slice().as_ref(),
    );
}

fn eigh_fixture() -> Tensor<f32> {
    t(vec![4.0, 1.0, 1.0, 3.0], &[2, 2])
}

fn slogdet_fixture() -> Tensor<f32> {
    t(vec![2.0, 0.5, 0.5, 3.0], &[2, 2])
}

fn pinv_fixture() -> Tensor<f32> {
    t(vec![4.0, 7.0, 2.0, 6.0], &[2, 2])
}

fn lstsq_fixture() -> (Tensor<f32>, Tensor<f32>) {
    (
        t(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]),
        t(vec![1.0, 2.0, 4.0], &[3, 1]),
    )
}

// --- eigh ---

#[test]
fn cpu_eigh_forward_matches_naive_reference_within_tolerance() {
    let data = eigh_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    let out_cpu = eigh(&x_cpu).unwrap();
    let out_naive = eigh(&x_naive).unwrap();
    assert_forward_parity(
        "eigh eigenvalues: cpu vs naive",
        &out_cpu.eigenvalues.to_tensor(),
        &out_naive.eigenvalues.to_tensor(),
    );
    assert_forward_parity(
        "eigh eigenvectors: cpu vs naive",
        &out_cpu.eigenvectors.to_tensor(),
        &out_naive.eigenvectors.to_tensor(),
    );
}

#[test]
fn cpu_eigh_backward_matches_naive_reference_within_tolerance() {
    let data = eigh_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = eigh(&x_cpu).unwrap().eigenvalues.sum(None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let loss_naive = eigh(&x_naive).unwrap().eigenvalues.sum(None).unwrap();
    let dx_naive = naive_tape
        .backward(&loss_naive)
        .unwrap()
        .get(&x_naive)
        .unwrap()
        .unwrap()
        .clone();
    assert_forward_parity("eigh backward: cpu vs naive", &dx_cpu, &dx_naive);
}

#[test]
fn cpu_eigh_singular_shape_is_invalid_argument() {
    let ops = fandhe_ai::tape();
    let x = ops.make_var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    assert!(matches!(eigh(&x), Err(AutodiffError::InvalidArgument(_))));
}

// --- slogdet ---

#[test]
fn cpu_slogdet_forward_matches_naive_reference_within_tolerance() {
    let data = slogdet_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    let out_cpu = slogdet(&x_cpu).unwrap();
    let out_naive = slogdet(&x_naive).unwrap();
    // 符号は区分定数のため bit 一致を要求する。
    assert_eq!(
        out_cpu.sign.to_tensor().host_slice()[0],
        out_naive.sign.to_tensor().host_slice()[0],
        "slogdet sign"
    );
    assert_forward_parity(
        "slogdet logabsdet: cpu vs naive",
        &out_cpu.logabsdet.to_tensor(),
        &out_naive.logabsdet.to_tensor(),
    );
}

#[test]
fn cpu_slogdet_backward_matches_naive_reference_within_tolerance() {
    let data = slogdet_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = slogdet(&x_cpu).unwrap().logabsdet;
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let loss_naive = slogdet(&x_naive).unwrap().logabsdet;
    let dx_naive = naive_tape
        .backward(&loss_naive)
        .unwrap()
        .get(&x_naive)
        .unwrap()
        .unwrap()
        .clone();
    assert_forward_parity("slogdet backward: cpu vs naive", &dx_cpu, &dx_naive);
}

#[test]
fn cpu_slogdet_singular_is_invalid_argument_on_backward() {
    let cpu_tape = fandhe_ai::tape();
    let x = cpu_tape.make_var(&t(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]));
    let out = slogdet(&x).unwrap();
    assert_eq!(out.sign.to_tensor().host_slice()[0], 0.0);
    assert!(matches!(
        cpu_tape.backward(&out.logabsdet),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

// --- pinv ---

#[test]
fn cpu_pinv_forward_matches_naive_reference_within_tolerance() {
    let data = pinv_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    let p_cpu = pinv(&x_cpu, None).unwrap().to_tensor();
    let p_naive = pinv(&x_naive, None).unwrap().to_tensor();
    assert_forward_parity("pinv: cpu vs naive", &p_cpu, &p_naive);
}

#[test]
fn cpu_pinv_backward_matches_naive_reference_within_tolerance() {
    let data = pinv_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = pinv(&x_cpu, None).unwrap().sum(None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let loss_naive = pinv(&x_naive, None).unwrap().sum(None).unwrap();
    let dx_naive = naive_tape
        .backward(&loss_naive)
        .unwrap()
        .get(&x_naive)
        .unwrap()
        .unwrap()
        .clone();
    assert_forward_parity("pinv backward: cpu vs naive", &dx_cpu, &dx_naive);
}

// --- matrix_rank ---

#[test]
fn cpu_matrix_rank_forward_is_bit_exact_vs_naive() {
    let data = t(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]);
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let r_cpu = matrix_rank(&x_cpu, None).unwrap().to_tensor();
    let r_naive = matrix_rank(&x_naive, None).unwrap().to_tensor();
    assert_eq!(
        r_cpu.host_slice()[0],
        r_naive.host_slice()[0],
        "matrix_rank cpu vs naive"
    );
}

#[test]
fn cpu_matrix_rank_gradient_is_bit_exact_zero() {
    let cpu_tape = fandhe_ai::tape();
    let x = cpu_tape.make_var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let r = matrix_rank(&x, None).unwrap();
    let dx = cpu_tape
        .backward(&r)
        .unwrap()
        .get(&x)
        .unwrap()
        .unwrap()
        .clone();
    assert!(dx.host_slice().iter().all(|&v| v == 0.0));
}

// --- lstsq ---

#[test]
fn cpu_lstsq_forward_matches_naive_reference_within_tolerance() {
    let (a_data, b_data) = lstsq_fixture();
    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&a_data);
    let b_cpu = cpu_tape.make_var(&b_data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a_naive = naive_tape.make_var(&a_data);
    let b_naive = naive_tape.make_var(&b_data);

    let x_cpu = lstsq(&a_cpu, &b_cpu, None).unwrap().to_tensor();
    let x_naive = lstsq(&a_naive, &b_naive, None).unwrap().to_tensor();
    assert_forward_parity("lstsq: cpu vs naive", &x_cpu, &x_naive);
}

#[test]
fn cpu_lstsq_backward_matches_naive_reference_within_tolerance() {
    let (a_data, b_data) = lstsq_fixture();
    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&a_data);
    let b_cpu = cpu_tape.make_var(&b_data);
    let loss_cpu = lstsq(&a_cpu, &b_cpu, None).unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let da_cpu = grads_cpu.get(&a_cpu).unwrap().unwrap().clone();
    let db_cpu = grads_cpu.get(&b_cpu).unwrap().unwrap().clone();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a_naive = naive_tape.make_var(&a_data);
    let b_naive = naive_tape.make_var(&b_data);
    let loss_naive = lstsq(&a_naive, &b_naive, None).unwrap().sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let da_naive = grads_naive.get(&a_naive).unwrap().unwrap().clone();
    let db_naive = grads_naive.get(&b_naive).unwrap().unwrap().clone();

    assert_forward_parity("lstsq backward dA: cpu vs naive", &da_cpu, &da_naive);
    assert_forward_parity("lstsq backward dB: cpu vs naive", &db_cpu, &db_naive);
}

#[test]
fn cpu_lstsq_row_mismatch_is_invalid_argument() {
    let cpu_tape = fandhe_ai::tape();
    let a = cpu_tape.make_var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let b = cpu_tape.make_var(&t(vec![1.0, 2.0, 3.0], &[3, 1]));
    assert!(matches!(
        lstsq(&a, &b, None),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: Mac／DGX Spark GB10 実機セッションへ
// 申し送る（`docs/perf/logs/linalg-ops-2150/README.md`）。
// ---------------------------------------------------------------------

/// `eigh` forward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/linalg-ops-2150/README.md 参照"]
fn metal_eigh_forward_matches_cpu_reference() {
    let data = eigh_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);

    let out_cpu = eigh(&x_cpu).unwrap();
    let out_metal = eigh(&x_metal).unwrap();
    assert_forward_parity(
        "eigh eigenvalues: cpu vs metal",
        &out_cpu.eigenvalues.to_tensor(),
        &out_metal.eigenvalues.to_tensor(),
    );
}

/// `eigh` forward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/linalg-ops-2150/README.md 参照"]
fn cuda_eigh_forward_matches_cpu_reference() {
    let data = eigh_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);

    let out_cpu = eigh(&x_cpu).unwrap();
    let out_cuda = eigh(&x_cuda).unwrap();
    assert_forward_parity(
        "eigh eigenvalues: cpu vs cuda",
        &out_cpu.eigenvalues.to_tensor(),
        &out_cuda.eigenvalues.to_tensor(),
    );
}

/// `pinv` forward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/linalg-ops-2150/README.md 参照"]
fn metal_pinv_forward_matches_cpu_reference() {
    let data = pinv_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);

    let p_cpu = pinv(&x_cpu, None).unwrap().to_tensor();
    let p_metal = pinv(&x_metal, None).unwrap().to_tensor();
    assert_forward_parity("pinv: cpu vs metal", &p_cpu, &p_metal);
}

/// `pinv` forward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/linalg-ops-2150/README.md 参照"]
fn cuda_pinv_forward_matches_cpu_reference() {
    let data = pinv_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);

    let p_cpu = pinv(&x_cpu, None).unwrap().to_tensor();
    let p_cuda = pinv(&x_cuda, None).unwrap().to_tensor();
    assert_forward_parity("pinv: cpu vs cuda", &p_cpu, &p_cuda);
}
