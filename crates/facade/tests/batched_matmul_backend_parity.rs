//! バッチ行列積（`Var::matmul` の rank≥2・バッチ次元 NumPy 互換
//! ブロードキャスト対応。イシュー #1715。親 #1600。spec REQ-9
//! 2026-09-12 追記 Tier 1「バッチ行列積」）の facade 到達経路（既存
//! `Var` 再エクスポート経由。新規 `pub use`／`pub fn` は追加していない）
//! の受け入れ条件対応テスト（`shape_ops_backend_parity.rs` と同型）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps::gemm_batched`
//!   オーバーライド経由）と `fandhe_ai_autodiff::Tape::new()`
//!   （`NaiveOps`。`BackendOps::gemm_batched` の既定合成実装経由）で
//!   rank≥3 `matmul` の forward／backward を REQ-2 統一複合判定で
//!   突き合わせる。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU tape
//!   と `assert_parity` で比較する。CUDA は #1716 で
//!   `CudaBackendOps::gemm_batched`／`gemm_batched_fp32_strict`
//!   専用オーバーライド（デバイス常駐バッチループ経路
//!   `CudaGemm::run_tiled_f32_batched`）を実装済み（forward・backward
//!   （rank≥3 `matmul_vjp` が呼ぶ `gemm_batched_fp32_strict` 経由）
//!   とも本ファイルの `#[ignore]` テストで検証する。数値契約・追加の
//!   形状網羅は `crates/backend-cuda/tests/gemm_batched_parity.rs`
//!   を参照）。Metal は本イシュー時点で `gemm_batched` を専用
//!   オーバーライドせず既定合成実装〈per-batch `gemm`〉のまま
//!   （専用バッチカーネルは後続イシュー #1717）。GEMM カーネルが
//!   異なるため bit 同一は主張しない。実機実測は本エージェント実行
//!   環境に CUDA／Apple Silicon 実機がないため未実施のまま該当
//!   セッションへ申し送る。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする
/// （`shape_ops_backend_parity.rs` の `VarSource` と同じ構成）。
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

// --- 属性なし: CPU vs NaiveOps ---

/// バッチ `matmul`（等バッチ形状）forward の CPU
/// （`CpuBackendOps::gemm_batched` オーバーライド）と NaiveOps
/// （`BackendOps::gemm_batched` 既定合成実装。per-batch `eval::matmul`）
/// の parity。
#[test]
fn cpu_batched_matmul_forward_matches_naive_reference() {
    let a_shape = [3usize, 2, 4];
    let b_shape = [3usize, 4, 5];

    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&leaf(1, &a_shape));
    let b_cpu = cpu_tape.make_var(&leaf(2, &b_shape));
    let out_cpu = a_cpu
        .matmul(&b_cpu)
        .expect("matmul: [3,2,4] x [3,4,5] は形状適合")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a_naive = naive_tape.make_var(&leaf(1, &a_shape));
    let b_naive = naive_tape.make_var(&leaf(2, &b_shape));
    let out_naive = a_naive
        .matmul(&b_naive)
        .expect("matmul: [3,2,4] x [3,4,5] は形状適合")
        .to_tensor();

    assert_eq!(out_cpu.shape(), &[3, 2, 5]);
    assert_eq!(out_cpu.shape(), out_naive.shape());
    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::gemm_batched 経由）vs NaiveOps（既定合成）",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

/// バッチ `matmul`（lhs バッチ次元 1 の broadcast）forward の parity。
#[test]
fn cpu_batched_matmul_forward_broadcast_matches_naive_reference() {
    let a_shape = [1usize, 3, 2];
    let b_shape = [4usize, 2, 3];

    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&leaf(3, &a_shape));
    let b_cpu = cpu_tape.make_var(&leaf(4, &b_shape));
    let out_cpu = a_cpu
        .matmul(&b_cpu)
        .expect("matmul: [1,3,2] x [4,2,3] は broadcast 適合")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a_naive = naive_tape.make_var(&leaf(3, &a_shape));
    let b_naive = naive_tape.make_var(&leaf(4, &b_shape));
    let out_naive = a_naive
        .matmul(&b_naive)
        .expect("matmul: [1,3,2] x [4,2,3] は broadcast 適合")
        .to_tensor();

    assert_eq!(out_cpu.shape(), &[4, 3, 3]);
    assert_parity(
        "fandhe_ai::tape()（broadcast batch）vs NaiveOps",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

/// バッチ `matmul` backward（`dA`／`dB`）の CPU と NaiveOps の parity。
#[test]
fn cpu_batched_matmul_backward_matches_naive_reference() {
    let a_shape = [2usize, 3, 4];
    let b_shape = [2usize, 4, 3];
    let target_shape = [2usize, 3, 3];

    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&leaf(5, &a_shape));
    let b_cpu = cpu_tape.make_var(&leaf(6, &b_shape));
    let t_cpu = cpu_tape.make_var(&leaf(7, &target_shape));
    let y_cpu = a_cpu.matmul(&b_cpu).unwrap();
    let loss_cpu = y_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let da_cpu = grads_cpu.get(&a_cpu).unwrap().expect("到達する");
    let db_cpu = grads_cpu.get(&b_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a_naive = naive_tape.make_var(&leaf(5, &a_shape));
    let b_naive = naive_tape.make_var(&leaf(6, &b_shape));
    let t_naive = naive_tape.make_var(&leaf(7, &target_shape));
    let y_naive = a_naive.matmul(&b_naive).unwrap();
    let loss_naive = y_naive.mse_loss(&t_naive).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let da_naive = grads_naive.get(&a_naive).unwrap().expect("到達する");
    let db_naive = grads_naive.get(&b_naive).unwrap().expect("到達する");

    assert_parity(
        "batched matmul backward dA: CpuBackendOps vs NaiveOps",
        &contiguous_slice(da_cpu),
        &contiguous_slice(da_naive),
    );
    assert_parity(
        "batched matmul backward dB: CpuBackendOps vs NaiveOps",
        &contiguous_slice(db_cpu),
        &contiguous_slice(db_naive),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn batched_matmul_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let a = tape.make_var(&leaf(1, &[3, 2, 4]));
    let b = tape.make_var(&leaf(2, &[3, 4, 5]));
    a.matmul(&b)
        .expect("matmul: [3,2,4] x [3,4,5] は形状適合")
        .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`shape_ops_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_batched_matmul_forward_matches_cpu() {
    let metal_out = batched_matmul_forward_on(Device::Metal);
    let cpu_out = batched_matmul_forward_on(Device::Cpu);

    assert_parity(
        "batched matmul forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_batched_matmul_forward_matches_cpu() {
    let cuda_out = batched_matmul_forward_on(Device::Cuda(0));
    let cpu_out = batched_matmul_forward_on(Device::Cpu);

    assert_parity(
        "batched matmul forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

/// バッチ `matmul` backward（`dA`／`dB`）の CUDA `tape_for` と CPU
/// `tape_for` の parity（イシュー #1716。`cpu_batched_matmul_backward_matches_naive_reference`
/// の実機横断版）。CUDA 側 backward は `matmul_vjp` の rank≥3 分岐が
/// 呼ぶ `CudaBackendOps::gemm_batched_fp32_strict`（#1716 で新設した
/// デバイス常駐バッチループ経路）を経由する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_batched_matmul_backward_matches_cpu() {
    let a_shape = [2usize, 3, 4];
    let b_shape = [2usize, 4, 3];
    let target_shape = [2usize, 3, 3];

    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let a_cuda = cuda_tape.make_var(&leaf(5, &a_shape));
    let b_cuda = cuda_tape.make_var(&leaf(6, &b_shape));
    let t_cuda = cuda_tape.make_var(&leaf(7, &target_shape));
    let y_cuda = a_cuda.matmul(&b_cuda).unwrap();
    let loss_cuda = y_cuda.mse_loss(&t_cuda).unwrap();
    let grads_cuda = cuda_tape.backward(&loss_cuda).unwrap();
    let da_cuda = grads_cuda.get(&a_cuda).unwrap().expect("到達する");
    let db_cuda = grads_cuda.get(&b_cuda).unwrap().expect("到達する");

    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&leaf(5, &a_shape));
    let b_cpu = cpu_tape.make_var(&leaf(6, &b_shape));
    let t_cpu = cpu_tape.make_var(&leaf(7, &target_shape));
    let y_cpu = a_cpu.matmul(&b_cpu).unwrap();
    let loss_cpu = y_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let da_cpu = grads_cpu.get(&a_cpu).unwrap().expect("到達する");
    let db_cpu = grads_cpu.get(&b_cpu).unwrap().expect("到達する");

    assert_parity(
        "batched matmul backward dA: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(da_cuda),
        &contiguous_slice(da_cpu),
    );
    assert_parity(
        "batched matmul backward dB: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(db_cuda),
        &contiguous_slice(db_cpu),
    );
}
