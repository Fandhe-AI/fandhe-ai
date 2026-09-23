//! GPU elementwise allowlist 融合（区分 B-1・イシュー #2085）の勾配
//! bit 完全一致検証（Linux〈CI〉で実行可能。実装計画 §5.1 (d)）。
//!
//! 実機 CUDA／Metal デバイスを使わず、GPU 融合カーネルと**同じ発生順・
//! 同じ演算定義**で評価するホスト逐語モデル
//! （[`fandhe_ai_backend_cuda::fused_elementwise_model::eval_program_host`]。
//! `#[doc(hidden)] pub`・デバイス非依存の純 Rust 関数）を `BackendOps::
//! run_fused` として差し込んだ `Tape`（`fandhe_ai_autodiff::Tape::
//! new_with_ops`）で elementwise 連鎖の forward／backward を実行し、
//! 通常の `Tape::new()`（既定 CPU バックエンド。`backend-cpu::
//! fused_elementwise::run_fused_elementwise` による融合実行済み）と
//! 出力・全勾配を `to_bits()` で完全一致比較する。
//!
//! **契約の位置づけ**: この bit 完全一致は「GPU 融合カーネルの forward
//! 実体化値が変わらなければ、`Tape::backward` が読み出す VJP 入力も
//! 変わらない」ことの検証である（backward 自体〈VJP 式〉は本 PR の対象
//! 外のまま。`docs/fusion-graph-design.md` §3.3・決定記録 §3(3)
//! 「backward は融合対象外」は不変。`crates/backend-cuda/src/fused_
//! elementwise_model.rs` モジュール冒頭コメント参照）。
//!
//! フィクスチャの入力に `-0.0` を含めない（`eval_program_host` の
//! `Relu` 三項式 `x > 0.0 { x } else { 0.0 }` と CPU 融合カーネル
//! `eval_one` の `x.max(0.0)` は `-0.0` 入力でのみ符号ビットが異なり
//! うるため。`fused_elementwise_model.rs` モジュール冒頭コメント参照）。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::fused_elementwise_model::eval_program_host;
use fandhe_ai_tensor_core::{BackendError, BackendOps, Device, FusionPlan, Tensor};

/// [`CpuBackendOps`] へ全メソッドを委譲しつつ、`run_fused` のみ
/// GPU 融合カーネルのホスト逐語モデルへ差し替えるフィクスチャ
/// （`crates/autodiff/tests/fusion_backend_integration.rs::
/// CountingFusedOps` と同型の構成方針）。
struct HostModelFusedOps {
    inner: CpuBackendOps,
}

impl BackendOps for HostModelFusedOps {
    fn device(&self) -> Device {
        self.inner.device()
    }
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.gemm(a, b)
    }
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.add(a, b)
    }
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.mul(a, b)
    }
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.relu(a)
    }
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.exp(a)
    }
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.tanh(a)
    }
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.sum(a, dim)
    }
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.max(a, dim)
    }

    /// `plan.ops()` をそのまま [`eval_program_host`] へ渡す（GPU
    /// 融合カーネルの評価順序・演算定義をホスト側で模倣する）。
    /// leaf は `contiguous()` で密なバッファへ実体化してからスライスを
    /// 取る（`backend-cpu::fused_elementwise::run_fused_elementwise` の
    /// 呼び出し規約と同一）。
    fn run_fused(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
    ) -> Result<Tensor<f32>, BackendError> {
        let owned: Vec<Tensor<f32>> = leaves.iter().map(|t| t.contiguous()).collect();
        let mut slices: Vec<&[f32]> = Vec::with_capacity(owned.len());
        for (i, t) in owned.iter().enumerate() {
            let s = t.as_slice().ok_or_else(|| {
                BackendError::Unsupported(format!(
                    "HostModelFusedOps::run_fused: leaf {i} is non-contiguous after \
                     contiguous() (unexpected)"
                ))
            })?;
            slices.push(s);
        }
        let ops: Vec<_> = plan.ops().collect();
        let data = eval_program_host(&ops, &slices);
        Tensor::new(data, plan.output_shape()).map_err(BackendError::ShapeMismatch)
    }
}

/// 6 段の elementwise 連鎖（`add → mul → relu → exp → tanh → add`。
/// `crates/autodiff/tests/fusion_backend_integration.rs::build_chain`
/// と同一パターン）を構築する。fan-out（`bias` を 2 箇所で参照）を
/// 含む。`bias`・`scale` の `Var` も呼び出し元へ返す（本体の `x` に
/// 加え、fan-out する `bias` と乗算葉の `scale` の勾配も bit 完全
/// 一致検証の対象にするため。codex-review 指摘・PR #2232）。
fn build_chain<'t>(
    tape: &'t Tape,
    x: &fandhe_ai_autodiff::Var<'t>,
) -> (
    fandhe_ai_autodiff::Var<'t>,
    fandhe_ai_autodiff::Var<'t>,
    fandhe_ai_autodiff::Var<'t>,
) {
    let bias = tape.var(&Tensor::new(vec![0.1, 0.2, 0.3, 0.4], &[4]).unwrap());
    let scale = tape.var(&Tensor::new(vec![1.1, 0.9, 1.05, 0.95], &[4]).unwrap());
    let h1 = x.add(&bias).unwrap();
    let h2 = h1.mul(&scale).unwrap();
    let h3 = h2.relu();
    let h4 = h3.exp();
    let h5 = h4.tanh();
    let out = h5.add(&bias).unwrap();
    (out, bias, scale)
}

/// `-0.0` を含まない入力（モジュール冒頭コメント参照）。
fn fixture_x_data() -> Vec<f32> {
    vec![0.5, -0.5, 1.5, -1.5]
}

#[test]
fn forward_output_matches_bit_exact_between_host_model_and_cpu_fused_kernel() {
    let host_model_tape = Tape::new_with_ops(Box::new(HostModelFusedOps {
        inner: CpuBackendOps::new(),
    }));
    let x_hm = host_model_tape.var(&Tensor::new(fixture_x_data(), &[4]).unwrap());
    let (out_hm, _bias_hm, _scale_hm) = build_chain(&host_model_tape, &x_hm);
    let out_hm = out_hm.to_tensor();

    let cpu_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let x_cpu = cpu_tape.var(&Tensor::new(fixture_x_data(), &[4]).unwrap());
    let (out_cpu, _bias_cpu, _scale_cpu) = build_chain(&cpu_tape, &x_cpu);
    let out_cpu = out_cpu.to_tensor();

    let hm_slice = out_hm.as_slice().expect("contiguous");
    let cpu_slice = out_cpu.as_slice().expect("contiguous");
    assert_eq!(hm_slice.len(), cpu_slice.len());
    for (i, (&a, &b)) in hm_slice.iter().zip(cpu_slice.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "forward output element {i} differs: host_model={a} (bits={:#x}) cpu_fused={b} \
             (bits={:#x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

/// `to_bits()` による要素ごとの bit 完全一致を検証する共通アサーション
/// （どの葉の勾配かを `label` としてエラーメッセージへ含める）。
fn assert_grad_bit_exact(label: &str, hm: &Tensor<f32>, cpu: &Tensor<f32>) {
    let hm_slice = hm.as_slice().expect("contiguous");
    let cpu_slice = cpu.as_slice().expect("contiguous");
    assert_eq!(hm_slice.len(), cpu_slice.len());
    for (i, (&a, &b)) in hm_slice.iter().zip(cpu_slice.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label} element {i} differs: host_model={a} (bits={:#x}) cpu_fused={b} (bits={:#x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

/// 出力・全勾配（`x`・fan-out する `bias`・乗算葉の `scale`）の
/// bit 完全一致を検証する（codex-review 指摘・PR #2232: 従来は `x` の
/// 勾配のみを比較しており、`bias`・`scale` にだけ生じる回帰を検出
/// できなかった）。
#[test]
fn gradient_matches_bit_exact_between_host_model_and_cpu_fused_kernel() {
    let host_model_tape = Tape::new_with_ops(Box::new(HostModelFusedOps {
        inner: CpuBackendOps::new(),
    }));
    let x_hm = host_model_tape.var(&Tensor::new(fixture_x_data(), &[4]).unwrap());
    let (out_hm, bias_hm, scale_hm) = build_chain(&host_model_tape, &x_hm);
    let loss_hm = out_hm.sum(None).unwrap();
    let grads_hm = host_model_tape.backward(&loss_hm).unwrap();
    let grad_x_hm = grads_hm
        .get(&x_hm)
        .unwrap()
        .expect("x は loss に寄与しているはず");
    let grad_bias_hm = grads_hm
        .get(&bias_hm)
        .unwrap()
        .expect("bias は loss に寄与しているはず");
    let grad_scale_hm = grads_hm
        .get(&scale_hm)
        .unwrap()
        .expect("scale は loss に寄与しているはず");

    let cpu_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let x_cpu = cpu_tape.var(&Tensor::new(fixture_x_data(), &[4]).unwrap());
    let (out_cpu, bias_cpu, scale_cpu) = build_chain(&cpu_tape, &x_cpu);
    let loss_cpu = out_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let grad_x_cpu = grads_cpu
        .get(&x_cpu)
        .unwrap()
        .expect("x は loss に寄与しているはず");
    let grad_bias_cpu = grads_cpu
        .get(&bias_cpu)
        .unwrap()
        .expect("bias は loss に寄与しているはず");
    let grad_scale_cpu = grads_cpu
        .get(&scale_cpu)
        .unwrap()
        .expect("scale は loss に寄与しているはず");

    assert_grad_bit_exact("grad_x", grad_x_hm, grad_x_cpu);
    assert_grad_bit_exact("grad_bias", grad_bias_hm, grad_bias_cpu);
    assert_grad_bit_exact("grad_scale", grad_scale_hm, grad_scale_cpu);
}
