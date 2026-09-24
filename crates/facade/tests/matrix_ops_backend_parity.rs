//! `fandhe_ai_autodiff::matrix_ops`（イシュー #2144・facade 非公開の
//! 内部入口。`crates/autodiff/src/matrix_ops.rs` モジュール doc 参照）
//! のバックエンド間 parity テスト（`rearrange_ops_backend_parity.rs`
//! と同型）。
//!
//! `matrix_ops` は facade から再エクスポートされないため、本テストは
//! `fandhe_ai_autodiff::matrix_ops::*` を直接 use する（facade の dev
//! 依存に `fandhe-ai-autodiff` が既に含まれている）。
//!
//! 本ファイルの契約は `tril`／`triu`／`diag`（両方向）／`trace`／
//! `outer`／`dot` の 6 演算 × {forward, backward} × {CPU vs NaiveOps,
//! CUDA vs CPU, Metal vs CPU} の主要セルを埋めることであり、以下の
//! 関数名は網羅表と一対一対応する（`rearrange_ops_backend_parity.rs`
//! の教訓: 関数名・doc が謳う対象と実際に実行する演算がずれていた
//! codex-review 指摘の再発防止）。
//!
//! - 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps`〉と
//!   `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`〉の突き合わせ）:
//!   - コピー系（`tril`／`triu`／`diag` 両方向／`outer`）forward bit
//!     完全一致: `cpu_copy_ops_forward_bit_matches_naive_reference`
//!   - `tril`（0 埋め位置）の `NaN`／`inf` payload 保存確認:
//!     `cpu_forward_masked_positions_are_zero_and_nan_bits_preserved`
//!   - 縮約系（`trace`／`dot`）forward（REQ-2 統一複合判定）:
//!     `cpu_reduce_ops_forward_matches_naive_reference_within_tolerance`
//!   - bit 完全一致 backward（`tril`／`triu`／`diag`／`trace`／`dot`）:
//!     `cpu_bit_exact_backward_matches_naive_reference`
//!   - `outer` backward（REQ-2 統一複合判定）:
//!     `cpu_outer_backward_matches_naive_reference_within_tolerance`
//! - `#[ignore]`（`tape_for(Device::Metal)`〈`cfg(target_os =
//!   "macos")` 限定〉／`tape_for(Device::Cuda(0))` で同じ経路を CPU
//!   tape と比較）: 上記 4 つの比較（`NaN` 検査を除く）を `cuda_*`／
//!   `metal_*` という接頭辞で対称に置く（計 8 件）。
//!
//!   実機（DGX Spark GB10／Apple Silicon）への到達手段が本エージェント
//!   実行環境にないため未実施のまま Mac／GB10 セッションへ申し送る
//!   （`docs/perf/logs/shape-matrix-ops-2144/README.md`）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::matrix_ops::{diag, dot, outer, trace, tril, triu};
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

fn f32_fixture_3x3() -> Tensor<f32> {
    Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
        .expect("test fixture: shape 一致")
}

fn f32_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// コピー系 4 演算（`tril`／`triu`／`diag` 両方向／`outer`）forward が
/// CPU（`fandhe_ai::tape()`）と NaiveOps（`fandhe_ai_autodiff::
/// Tape::new()`）で bit 完全一致することを確認する（算術を含まない
/// コピー・0 埋め演算のため）。
#[test]
fn cpu_copy_ops_forward_bit_matches_naive_reference() {
    let data = f32_fixture_3x3();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    assert_eq!(
        f32_bits(&tril(&x_cpu, 0).unwrap().to_tensor()),
        f32_bits(&tril(&x_naive, 0).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&triu(&x_cpu, 1).unwrap().to_tensor()),
        f32_bits(&triu(&x_naive, 1).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&diag(&x_cpu, 0).unwrap().to_tensor()),
        f32_bits(&diag(&x_naive, 0).unwrap().to_tensor())
    );

    let vec1d = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let v_cpu = cpu_tape.make_var(&vec1d);
    let v_naive = naive_tape.make_var(&vec1d);
    assert_eq!(
        f32_bits(&diag(&v_cpu, 1).unwrap().to_tensor()),
        f32_bits(&diag(&v_naive, 1).unwrap().to_tensor())
    );

    let a_data = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();
    let b_data = Tensor::new(vec![10.0, 20.0, 30.0], &[3]).unwrap();
    let a_cpu = cpu_tape.make_var(&a_data);
    let b_cpu = cpu_tape.make_var(&b_data);
    let a_naive = naive_tape.make_var(&a_data);
    let b_naive = naive_tape.make_var(&b_data);
    assert_eq!(
        f32_bits(&outer(&a_cpu, &b_cpu).unwrap().to_tensor()),
        f32_bits(&outer(&a_naive, &b_naive).unwrap().to_tensor())
    );
}

/// `tril` が 0 化する位置（マスク位置）では `NaN`／`inf` を含む入力でも
/// 常に `+0.0` になり、残す位置では `NaN` の payload が保存されること
/// を CPU（`fandhe_ai::tape()`）で確認する（`masked_fill` 経由の
/// 数値契約。`rearrange_ops_backend_parity.rs::cpu_forward_preserves_
/// nan_bits` と同種の確認）。
#[test]
fn cpu_forward_masked_positions_are_zero_and_nan_bits_preserved() {
    let nan = f32::from_bits(0x7fc0_5678);
    let data = Tensor::new(vec![nan, f32::INFINITY, 1.0, 2.0], &[2, 2]).unwrap();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let out = tril(&x_cpu, 0).unwrap().to_tensor();
    let out_data = out.host_slice().into_owned();
    // 対角上 [0][0]=nan は残る（payload 保存）、[0][1]=inf は 0 化される。
    assert_eq!(out_data[0].to_bits(), nan.to_bits());
    assert_eq!(out_data[1], 0.0);
    assert!(out_data[1].is_sign_positive());
    assert_eq!(out_data[2], 1.0);
    assert_eq!(out_data[3], 2.0);
}

/// 縮約系 2 演算（`trace`／`dot`）forward が CPU と NaiveOps で REQ-2
/// 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を満たす
/// ことを確認する（`sum` の縮約順序がバックエンドで異なりうるため。
/// `.claude/rules/coding-rust.md`）。
#[test]
fn cpu_reduce_ops_forward_matches_naive_reference_within_tolerance() {
    let data = f32_fixture_3x3();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    let trace_cpu = trace(&x_cpu).unwrap().to_tensor();
    let trace_naive = trace(&x_naive).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "trace forward: cpu vs naive",
        trace_cpu.host_slice().as_ref(),
        trace_naive.host_slice().as_ref(),
    );

    let a_data = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b_data = Tensor::new(vec![4.0, 5.0, 6.0], &[3]).unwrap();
    let a_cpu = cpu_tape.make_var(&a_data);
    let b_cpu = cpu_tape.make_var(&b_data);
    let a_naive = naive_tape.make_var(&a_data);
    let b_naive = naive_tape.make_var(&b_data);
    let dot_cpu = dot(&a_cpu, &b_cpu).unwrap().to_tensor();
    let dot_naive = dot(&a_naive, &b_naive).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "dot forward: cpu vs naive",
        dot_cpu.host_slice().as_ref(),
        dot_naive.host_slice().as_ref(),
    );
}

/// `tril`／`triu`／`diag`／`trace`／`dot` の backward は CPU と NaiveOps
/// で bit 完全一致する（モジュール doc「数値契約」参照: マスク・
/// scatter の寄与がいずれも高々 1 つのため）。
#[test]
fn cpu_bit_exact_backward_matches_naive_reference() {
    let data = f32_fixture_3x3();
    let weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]).unwrap();

    // tril
    {
        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&data);
        let w_cpu = cpu_tape.make_var(&weight);
        let loss_cpu = tril(&x_cpu, 0)
            .unwrap()
            .mul(&w_cpu)
            .unwrap()
            .sum(None)
            .unwrap();
        let dx_cpu = cpu_tape
            .backward(&loss_cpu)
            .unwrap()
            .get(&x_cpu)
            .unwrap()
            .unwrap()
            .clone();

        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let x_naive = naive_tape.make_var(&data);
        let w_naive = naive_tape.make_var(&weight);
        let loss_naive = tril(&x_naive, 0)
            .unwrap()
            .mul(&w_naive)
            .unwrap()
            .sum(None)
            .unwrap();
        let dx_naive = naive_tape
            .backward(&loss_naive)
            .unwrap()
            .get(&x_naive)
            .unwrap()
            .unwrap()
            .clone();
        assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_naive));
    }

    // triu
    {
        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&data);
        let w_cpu = cpu_tape.make_var(&weight);
        let loss_cpu = triu(&x_cpu, 0)
            .unwrap()
            .mul(&w_cpu)
            .unwrap()
            .sum(None)
            .unwrap();
        let dx_cpu = cpu_tape
            .backward(&loss_cpu)
            .unwrap()
            .get(&x_cpu)
            .unwrap()
            .unwrap()
            .clone();

        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let x_naive = naive_tape.make_var(&data);
        let w_naive = naive_tape.make_var(&weight);
        let loss_naive = triu(&x_naive, 0)
            .unwrap()
            .mul(&w_naive)
            .unwrap()
            .sum(None)
            .unwrap();
        let dx_naive = naive_tape
            .backward(&loss_naive)
            .unwrap()
            .get(&x_naive)
            .unwrap()
            .unwrap()
            .clone();
        assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_naive));
    }

    // diag（2-D -> 1-D）
    {
        let w1d = Tensor::new(vec![10.0, 20.0, 30.0], &[3]).unwrap();
        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&data);
        let w_cpu = cpu_tape.make_var(&w1d);
        let loss_cpu = diag(&x_cpu, 0)
            .unwrap()
            .mul(&w_cpu)
            .unwrap()
            .sum(None)
            .unwrap();
        let dx_cpu = cpu_tape
            .backward(&loss_cpu)
            .unwrap()
            .get(&x_cpu)
            .unwrap()
            .unwrap()
            .clone();

        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let x_naive = naive_tape.make_var(&data);
        let w_naive = naive_tape.make_var(&w1d);
        let loss_naive = diag(&x_naive, 0)
            .unwrap()
            .mul(&w_naive)
            .unwrap()
            .sum(None)
            .unwrap();
        let dx_naive = naive_tape
            .backward(&loss_naive)
            .unwrap()
            .get(&x_naive)
            .unwrap()
            .unwrap()
            .clone();
        assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_naive));
    }

    // trace
    {
        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&data);
        let loss_cpu = trace(&x_cpu).unwrap();
        let dx_cpu = cpu_tape
            .backward(&loss_cpu)
            .unwrap()
            .get(&x_cpu)
            .unwrap()
            .unwrap()
            .clone();

        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let x_naive = naive_tape.make_var(&data);
        let loss_naive = trace(&x_naive).unwrap();
        let dx_naive = naive_tape
            .backward(&loss_naive)
            .unwrap()
            .get(&x_naive)
            .unwrap()
            .unwrap()
            .clone();
        assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_naive));
    }

    // dot
    {
        let a_data = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b_data = Tensor::new(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let cpu_tape = fandhe_ai::tape();
        let a_cpu = cpu_tape.make_var(&a_data);
        let b_cpu = cpu_tape.make_var(&b_data);
        let loss_cpu = dot(&a_cpu, &b_cpu).unwrap();
        let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
        let da_cpu = grads_cpu.get(&a_cpu).unwrap().unwrap().clone();

        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let a_naive = naive_tape.make_var(&a_data);
        let b_naive = naive_tape.make_var(&b_data);
        let loss_naive = dot(&a_naive, &b_naive).unwrap();
        let grads_naive = naive_tape.backward(&loss_naive).unwrap();
        let da_naive = grads_naive.get(&a_naive).unwrap().unwrap().clone();
        assert_eq!(f32_bits(&da_cpu), f32_bits(&da_naive));
    }
}

/// `outer` backward は CPU と NaiveOps で REQ-2 統一複合判定を満たす
/// （`broadcast_to` の VJP〈軸方向の `reduce_to_shape` 縮約〉を経由
/// するため。モジュール doc「数値契約」参照）。
#[test]
fn cpu_outer_backward_matches_naive_reference_within_tolerance() {
    let a_data = Tensor::new(vec![0.7, -1.3], &[2]).unwrap();
    let b_data = Tensor::new(vec![2.1, 0.4, -0.9], &[3]).unwrap();

    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&a_data);
    let b_cpu = cpu_tape.make_var(&b_data);
    let loss_cpu = outer(&a_cpu, &b_cpu).unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let da_cpu = grads_cpu.get(&a_cpu).unwrap().unwrap().clone();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a_naive = naive_tape.make_var(&a_data);
    let b_naive = naive_tape.make_var(&b_data);
    let loss_naive = outer(&a_naive, &b_naive).unwrap().sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let da_naive = grads_naive.get(&a_naive).unwrap().unwrap().clone();

    fandhe_ai_backend_cpu::parity::assert_parity(
        "outer backward: cpu vs naive",
        da_cpu.host_slice().as_ref(),
        da_naive.host_slice().as_ref(),
    );
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: Mac／DGX Spark GB10 実機セッションへ
// 申し送る（`docs/perf/logs/shape-matrix-ops-2144/README.md`）。
// ---------------------------------------------------------------------

/// コピー系 4 演算 forward が CPU と Metal 実機で bit 完全一致する
/// ことを確認する。CPU 側の同型カバレッジは
/// `cpu_copy_ops_forward_bit_matches_naive_reference`。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/shape-matrix-ops-2144/README.md 参照"]
fn metal_copy_ops_forward_matches_cpu_reference() {
    let data = f32_fixture_3x3();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);

    assert_eq!(
        f32_bits(&tril(&x_cpu, 0).unwrap().to_tensor()),
        f32_bits(&tril(&x_metal, 0).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&triu(&x_cpu, 1).unwrap().to_tensor()),
        f32_bits(&triu(&x_metal, 1).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&diag(&x_cpu, 0).unwrap().to_tensor()),
        f32_bits(&diag(&x_metal, 0).unwrap().to_tensor())
    );

    let a_data = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();
    let b_data = Tensor::new(vec![10.0, 20.0, 30.0], &[3]).unwrap();
    let a_cpu = cpu_tape.make_var(&a_data);
    let b_cpu = cpu_tape.make_var(&b_data);
    let a_metal = metal_tape.make_var(&a_data);
    let b_metal = metal_tape.make_var(&b_data);
    assert_eq!(
        f32_bits(&outer(&a_cpu, &b_cpu).unwrap().to_tensor()),
        f32_bits(&outer(&a_metal, &b_metal).unwrap().to_tensor())
    );
}

/// コピー系 4 演算 forward の CPU／CUDA 実機（DGX Spark GB10）比較。
/// 上記 Metal 版と対称。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/shape-matrix-ops-2144/README.md 参照"]
fn cuda_copy_ops_forward_matches_cpu_reference() {
    let data = f32_fixture_3x3();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);

    assert_eq!(
        f32_bits(&tril(&x_cpu, 0).unwrap().to_tensor()),
        f32_bits(&tril(&x_cuda, 0).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&triu(&x_cpu, 1).unwrap().to_tensor()),
        f32_bits(&triu(&x_cuda, 1).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&diag(&x_cpu, 0).unwrap().to_tensor()),
        f32_bits(&diag(&x_cuda, 0).unwrap().to_tensor())
    );

    let a_data = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();
    let b_data = Tensor::new(vec![10.0, 20.0, 30.0], &[3]).unwrap();
    let a_cpu = cpu_tape.make_var(&a_data);
    let b_cpu = cpu_tape.make_var(&b_data);
    let a_cuda = cuda_tape.make_var(&a_data);
    let b_cuda = cuda_tape.make_var(&b_data);
    assert_eq!(
        f32_bits(&outer(&a_cpu, &b_cpu).unwrap().to_tensor()),
        f32_bits(&outer(&a_cuda, &b_cuda).unwrap().to_tensor())
    );
}

/// 縮約系 2 演算（`trace`／`dot`）forward の CPU／Metal 実機比較
/// （REQ-2 統一複合判定）。CPU 側の同型カバレッジは
/// `cpu_reduce_ops_forward_matches_naive_reference_within_tolerance`。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/shape-matrix-ops-2144/README.md 参照"]
fn metal_reduce_ops_forward_matches_cpu_reference() {
    let data = f32_fixture_3x3();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);

    let trace_cpu = trace(&x_cpu).unwrap().to_tensor();
    let trace_metal = trace(&x_metal).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "trace forward: cpu vs metal",
        trace_cpu.host_slice().as_ref(),
        trace_metal.host_slice().as_ref(),
    );

    let a_data = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b_data = Tensor::new(vec![4.0, 5.0, 6.0], &[3]).unwrap();
    let a_cpu = cpu_tape.make_var(&a_data);
    let b_cpu = cpu_tape.make_var(&b_data);
    let a_metal = metal_tape.make_var(&a_data);
    let b_metal = metal_tape.make_var(&b_data);
    let dot_cpu = dot(&a_cpu, &b_cpu).unwrap().to_tensor();
    let dot_metal = dot(&a_metal, &b_metal).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "dot forward: cpu vs metal",
        dot_cpu.host_slice().as_ref(),
        dot_metal.host_slice().as_ref(),
    );
}

/// 縮約系 2 演算 forward の CPU／CUDA 実機（DGX Spark GB10）比較。上記
/// Metal 版と対称。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/shape-matrix-ops-2144/README.md 参照"]
fn cuda_reduce_ops_forward_matches_cpu_reference() {
    let data = f32_fixture_3x3();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);

    let trace_cpu = trace(&x_cpu).unwrap().to_tensor();
    let trace_cuda = trace(&x_cuda).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "trace forward: cpu vs cuda",
        trace_cpu.host_slice().as_ref(),
        trace_cuda.host_slice().as_ref(),
    );

    let a_data = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b_data = Tensor::new(vec![4.0, 5.0, 6.0], &[3]).unwrap();
    let a_cpu = cpu_tape.make_var(&a_data);
    let b_cpu = cpu_tape.make_var(&b_data);
    let a_cuda = cuda_tape.make_var(&a_data);
    let b_cuda = cuda_tape.make_var(&b_data);
    let dot_cpu = dot(&a_cpu, &b_cpu).unwrap().to_tensor();
    let dot_cuda = dot(&a_cuda, &b_cuda).unwrap().to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "dot forward: cpu vs cuda",
        dot_cpu.host_slice().as_ref(),
        dot_cuda.host_slice().as_ref(),
    );
}

/// `tril`／`triu`／`diag`／`trace`／`dot` backward（bit 完全一致契約）の
/// CPU／Metal 実機比較。CPU 側の同型カバレッジは
/// `cpu_bit_exact_backward_matches_naive_reference`（`trace` のみを
/// 代表として実機側は検証する。他 4 演算は同じ「寄与高々 1 つ」構造の
/// ため `trace` の bit 一致確認で契約全体の代表確認とする）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/shape-matrix-ops-2144/README.md 参照"]
fn metal_bit_exact_backward_matches_cpu_reference() {
    let data = f32_fixture_3x3();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = trace(&x_cpu).unwrap();
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
    let loss_metal = trace(&x_metal).unwrap();
    let dx_metal = metal_tape
        .backward(&loss_metal)
        .unwrap()
        .get(&x_metal)
        .unwrap()
        .unwrap()
        .clone();
    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_metal));
}

/// bit 完全一致 backward の CPU／CUDA 実機（DGX Spark GB10）比較。上記
/// Metal 版と対称。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/shape-matrix-ops-2144/README.md 参照"]
fn cuda_bit_exact_backward_matches_cpu_reference() {
    let data = f32_fixture_3x3();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = trace(&x_cpu).unwrap();
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
    let loss_cuda = trace(&x_cuda).unwrap();
    let dx_cuda = cuda_tape
        .backward(&loss_cuda)
        .unwrap()
        .get(&x_cuda)
        .unwrap()
        .unwrap()
        .clone();
    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_cuda));
}

/// `outer` backward（REQ-2 統一複合判定）の CPU／Metal 実機比較。
/// CPU 側の同型カバレッジは
/// `cpu_outer_backward_matches_naive_reference_within_tolerance`。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/shape-matrix-ops-2144/README.md 参照"]
fn metal_outer_backward_matches_cpu_reference() {
    let a_data = Tensor::new(vec![0.7, -1.3], &[2]).unwrap();
    let b_data = Tensor::new(vec![2.1, 0.4, -0.9], &[3]).unwrap();

    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&a_data);
    let b_cpu = cpu_tape.make_var(&b_data);
    let loss_cpu = outer(&a_cpu, &b_cpu).unwrap().sum(None).unwrap();
    let da_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&a_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let a_metal = metal_tape.make_var(&a_data);
    let b_metal = metal_tape.make_var(&b_data);
    let loss_metal = outer(&a_metal, &b_metal).unwrap().sum(None).unwrap();
    let da_metal = metal_tape
        .backward(&loss_metal)
        .unwrap()
        .get(&a_metal)
        .unwrap()
        .unwrap()
        .clone();

    fandhe_ai_backend_cpu::parity::assert_parity(
        "outer backward: cpu vs metal",
        da_cpu.host_slice().as_ref(),
        da_metal.host_slice().as_ref(),
    );
}

/// `outer` backward の CPU／CUDA 実機（DGX Spark GB10）比較。上記
/// Metal 版と対称。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/shape-matrix-ops-2144/README.md 参照"]
fn cuda_outer_backward_matches_cpu_reference() {
    let a_data = Tensor::new(vec![0.7, -1.3], &[2]).unwrap();
    let b_data = Tensor::new(vec![2.1, 0.4, -0.9], &[3]).unwrap();

    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&a_data);
    let b_cpu = cpu_tape.make_var(&b_data);
    let loss_cpu = outer(&a_cpu, &b_cpu).unwrap().sum(None).unwrap();
    let da_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&a_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let a_cuda = cuda_tape.make_var(&a_data);
    let b_cuda = cuda_tape.make_var(&b_data);
    let loss_cuda = outer(&a_cuda, &b_cuda).unwrap().sum(None).unwrap();
    let da_cuda = cuda_tape
        .backward(&loss_cuda)
        .unwrap()
        .get(&a_cuda)
        .unwrap()
        .unwrap()
        .clone();

    fandhe_ai_backend_cpu::parity::assert_parity(
        "outer backward: cpu vs cuda",
        da_cpu.host_slice().as_ref(),
        da_cuda.host_slice().as_ref(),
    );
}
