//! `fandhe_ai_autodiff::einsum_batch::einsum_batched`（イシュー #2149・
//! 親 #2131）の facade 到達経路の受け入れ条件対応テスト
//! （`einsum_backend_parity.rs` と同型）。facade（`fandhe_ai`）自体は
//! `einsum_batch` を再エクスポートしないため、`fandhe_ai_autodiff::
//! einsum_batch::einsum_batched` を直接呼び、`fandhe_ai::tape()`
//! （`CpuBackendOps`）が経由する `BackendOps::gemm_batched` の
//! parity を検証する（`einsum` 自体は新規カーネルを追加していない
//! ため「該当バックエンドすべてに実装」は rank≥3 `Var::matmul`
//! （イシュー #1715）の既存 parity 契約に帰着する）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で forward／backward
//!   を REQ-2 統一複合判定で突き合わせる（`ibj,bjk->bik`。batch 軸が
//!   先頭にない非恒等 permute 経路）。加えて `Var::einsum`（facade
//!   公開入口）が同じ spec で引き続き `InvalidArgument` を返すことを
//!   固定する保留ガード（`facade_var_einsum_still_rejects_batch_
//!   contraction`）を持つ。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU
//!   tape と `assert_parity` で比較する（GEMM カーネルが異なるため
//!   bit 同一は主張しない。GPU 側の GEMM は per-batch 経路と異なる
//!   結合順序を取りうるため REQ-2 統一複合判定で判定する）。実機実測
//!   の申し送りは `docs/perf/logs/einsum-batch-2149/README.md`。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::einsum_batch::einsum_batched;
use fandhe_ai_autodiff::{AutodiffError, Var};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`einsum_backend_parity.rs`
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

fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    // 決定的シード（`bench-harness::Xorshift64Star`。facade tests は
    // dev-dep として `bench-harness` を既に使用済み）。
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

/// spec：`"ibj,bjk->bik"`（batch 軸 b が a 側で先頭にないため非恒等
/// `permute` → `Op::Contiguous` を経由する GEMM 経路。`crate::einsum::
/// einsum_matmul_path` doc 参照）。形状は小さく取り（K<=64 程度）
/// Metal split-K・CUDA Tensor Core 経路の baseline 方式が要る形状を
/// 避け、既存の `batched_matmul_backend_parity.rs` と同じ判定方式で
/// 済むようにする。
const A_SHAPE: [usize; 3] = [3, 2, 4]; // [i, b, j]
const B_SHAPE: [usize; 3] = [2, 4, 5]; // [b, j, k]
const TARGET_SHAPE: [usize; 3] = [2, 3, 5]; // [b, i, k]

fn einsum_batch_forward<'t>(a: &Var<'t>, b: &Var<'t>) -> Tensor<f32> {
    einsum_batched("ibj,bjk->bik", &[a, b])
        .expect("einsum_batched: 形状適合（b/j 次元一致）")
        .to_tensor()
}

/// CPU（`CpuBackendOps` 経由 `gemm_batched`）と NaiveOps（ホスト参照
/// 実装）の forward parity。
#[test]
fn cpu_einsum_batch_forward_matches_naive_reference() {
    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&leaf(1, &A_SHAPE));
    let b_cpu = cpu_tape.make_var(&leaf(2, &B_SHAPE));
    let out_cpu = einsum_batch_forward(&a_cpu, &b_cpu);

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a_naive = naive_tape.make_var(&leaf(1, &A_SHAPE));
    let b_naive = naive_tape.make_var(&leaf(2, &B_SHAPE));
    let out_naive = einsum_batch_forward(&a_naive, &b_naive);

    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps 経由 einsum_batched \"ibj,bjk->bik\"）vs NaiveOps",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

/// 同経路 backward（`da`／`db`）の CPU と NaiveOps の parity。損失は
/// `mse_loss`（`Var::sum` の GPU 実装状況に依存しない。`docs/perf/
/// logs/einsum-batch-2149/README.md` の既知リスク節参照）。
#[test]
fn cpu_einsum_batch_backward_matches_naive_reference() {
    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&leaf(1, &A_SHAPE));
    let b_cpu = cpu_tape.make_var(&leaf(2, &B_SHAPE));
    let t_cpu = cpu_tape.make_var(&leaf(3, &TARGET_SHAPE));
    let y_cpu = einsum_batched("ibj,bjk->bik", &[&a_cpu, &b_cpu]).unwrap();
    let loss_cpu = y_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let da_cpu = grads_cpu.get(&a_cpu).unwrap().expect("到達する");
    let db_cpu = grads_cpu.get(&b_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a_naive = naive_tape.make_var(&leaf(1, &A_SHAPE));
    let b_naive = naive_tape.make_var(&leaf(2, &B_SHAPE));
    let t_naive = naive_tape.make_var(&leaf(3, &TARGET_SHAPE));
    let y_naive = einsum_batched("ibj,bjk->bik", &[&a_naive, &b_naive]).unwrap();
    let loss_naive = y_naive.mse_loss(&t_naive).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let da_naive = grads_naive.get(&a_naive).unwrap().expect("到達する");
    let db_naive = grads_naive.get(&b_naive).unwrap().expect("到達する");

    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps 経由 einsum_batched \"ibj,bjk->bik\"）da vs NaiveOps",
        &contiguous_slice(da_cpu),
        &contiguous_slice(da_naive),
    );
    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps 経由 einsum_batched \"ibj,bjk->bik\"）db vs NaiveOps",
        &contiguous_slice(db_cpu),
        &contiguous_slice(db_naive),
    );
}

/// facade の唯一の公開入口 `Var::einsum` は、`einsum_batch` 追加後も
/// batch 添字を伴う縮約を引き続き `InvalidArgument` で拒否すること
/// （facade 公開はイシュー #2149 の承認事項であり、承認前に
/// `Var::einsum` の facade から観測される挙動を変えない設計判断の
/// 実行時ガード。`VarEinsumBatchHoldDoctestGuard`〈`crates/facade/
/// src/lib.rs`〉と多層防御を成す）。
#[test]
fn facade_var_einsum_still_rejects_batch_contraction() {
    let tape = fandhe_ai::tape();
    let a = tape.make_var(&leaf(1, &A_SHAPE));
    let b = tape.make_var(&leaf(2, &B_SHAPE));
    let result = Var::einsum("ibj,bjk->bik", &[&a, &b]);
    assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn einsum_batch_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let a = tape.make_var(&leaf(1, &A_SHAPE));
    let b = tape.make_var(&leaf(2, &B_SHAPE));
    einsum_batch_forward(&a, &b)
}

fn einsum_batch_backward_da_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let a = tape.make_var(&leaf(1, &A_SHAPE));
    let b = tape.make_var(&leaf(2, &B_SHAPE));
    let t = tape.make_var(&leaf(3, &TARGET_SHAPE));
    let y = einsum_batched("ibj,bjk->bik", &[&a, &b]).unwrap();
    let loss = y.mse_loss(&t).unwrap();
    let grads = tape.backward(&loss).unwrap();
    grads.get(&a).unwrap().expect("到達する").clone()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`einsum_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_einsum_batch_forward_matches_cpu() {
    let metal_out = einsum_batch_forward_on(Device::Metal);
    let cpu_out = einsum_batch_forward_on(Device::Cpu);

    assert_parity(
        "einsum_batched \"ibj,bjk->bik\" forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_einsum_batch_backward_matches_cpu() {
    let metal_da = einsum_batch_backward_da_on(Device::Metal);
    let cpu_da = einsum_batch_backward_da_on(Device::Cpu);

    assert_parity(
        "einsum_batched \"ibj,bjk->bik\" backward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_da),
        &contiguous_slice(&cpu_da),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_einsum_batch_forward_matches_cpu() {
    let cuda_out = einsum_batch_forward_on(Device::Cuda(0));
    let cpu_out = einsum_batch_forward_on(Device::Cpu);

    assert_parity(
        "einsum_batched \"ibj,bjk->bik\" forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_einsum_batch_backward_matches_cpu() {
    let cuda_da = einsum_batch_backward_da_on(Device::Cuda(0));
    let cpu_da = einsum_batch_backward_da_on(Device::Cpu);

    assert_parity(
        "einsum_batched \"ibj,bjk->bik\" backward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_da),
        &contiguous_slice(&cpu_da),
    );
}
