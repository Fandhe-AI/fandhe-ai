//! `fandhe_ai_autodiff::scalar_unary_ops`（イシュー #2145・facade 非公開
//! の内部入口。`crates/autodiff/src/scalar_unary_ops.rs` モジュール doc
//! 参照）のバックエンド間 parity テスト（`rearrange_ops_backend_parity.rs`
//! と同型）。
//!
//! `scalar_unary_ops` は facade から再エクスポートされないため、本テストは
//! `fandhe_ai_autodiff::scalar_unary_ops::*` を直接 use する（facade の
//! dev 依存に `fandhe-ai-autodiff` が既に含まれている）。
//!
//! 本ファイルの契約は `floor`／`ceil`／`round`／`sign`／`reciprocal`／
//! `rsqrt`／`erf`／`pow_scalar` の 8 演算 × {forward, backward} ×
//! {CPU vs NaiveOps, CUDA vs CPU, Metal vs CPU} の全セルを埋めることで
//! あり、以下の関数名は網羅表と一対一対応する。
//!
//! - 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps`〉と
//!   `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`〉の突き合わせ）:
//!   - forward 全 8 種 bit 完全一致: `cpu_forward_matches_naive_reference`
//!   - backward（floor/ceil/round/sign は恒等的に 0。reciprocal/rsqrt/
//!     erf/pow_scalar は `fandhe_ai_backend_cpu::parity::assert_parity`
//!     による REQ-2 統一複合判定）:
//!     `cpu_backward_matches_naive_reference`
//! - `#[ignore]`（`tape_for(Device::Metal)`〈`cfg(target_os =
//!   "macos")` 限定〉／`tape_for(Device::Cuda(0))` で同じ経路を CPU
//!   tape と比較）:
//!   - forward 全 8 種 bit 完全一致:
//!     `metal_forward_matches_cpu_reference`／
//!     `cuda_forward_matches_cpu_reference`
//!   - backward（同上判定方針）:
//!     `metal_backward_matches_cpu_reference`／
//!     `cuda_backward_matches_cpu_reference`
//!
//!   実機（DGX Spark GB10／Apple Silicon）への到達手段が本エージェント
//!   実行環境にないため未実施のまま Mac／GB10 セッションへ申し送る
//!   （`docs/perf/logs/scalar-unary-ops-2145/README.md`）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::scalar_unary_ops::{
    ceil, erf, floor, pow_scalar, reciprocal, round, rsqrt, sign,
};
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

/// `Floor`／`Ceil`／`Round`／`Sign`／`Reciprocal`／`Erf` に使える入力
/// （`Reciprocal` の 0 を避け、`Round` はタイブレークも含むよう
/// `0.5`／`2.5` を含む）。`Rsqrt`／`PowScalar`（分数指数）は定義域が
/// 正のため [`f32_positive_fixture`] を別途使う。
fn f32_fixture() -> Tensor<f32> {
    Tensor::new(vec![-2.7, -0.5, 0.5, 1.3, 2.5, 4.0], &[6]).expect("test fixture: shape 一致")
}

/// `Rsqrt`（`x < 0` は `NaN`）・`PowScalar`（分数指数。負の底は `NaN`）
/// のための正の入力。
fn f32_positive_fixture() -> Tensor<f32> {
    Tensor::new(vec![0.1, 0.5, 1.3, 2.5, 4.0, 6.7], &[6]).expect("test fixture: shape 一致")
}

/// `op_name` の定義域に合わせて [`f32_fixture`]／[`f32_positive_fixture`]
/// を選ぶ（`Rsqrt` のみ正の入力を要求。`Reciprocal` は 0 さえ避ければ
/// 負も可）。
fn fixture_for(op_name: &str) -> Tensor<f32> {
    if op_name == "rsqrt" {
        f32_positive_fixture()
    } else {
        f32_fixture()
    }
}

/// backward 突き合わせ用の重み（`mul` で non-trivial upstream を作る）。
fn f32_weight() -> Tensor<f32> {
    Tensor::new(vec![1.0, 0.5, -0.25, 2.0, 0.1, -1.0], &[6]).expect("test fixture: shape 一致")
}

fn f32_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

type UnaryFn = for<'t> fn(&Var<'t>) -> Result<Var<'t>, fandhe_ai_autodiff::AutodiffError>;

/// 網羅表と一対一対応させるための名前付き演算一覧（`pow_scalar` は
/// exponent 引数が必要なため別扱い）。
fn unary_ops() -> [(&'static str, UnaryFn); 7] {
    [
        ("floor", floor),
        ("ceil", ceil),
        ("round", round),
        ("sign", sign),
        ("reciprocal", reciprocal),
        ("rsqrt", rsqrt),
        ("erf", erf),
    ]
}

/// forward（8 種）が CPU（`fandhe_ai::tape()`）と NaiveOps
/// （`fandhe_ai_autodiff::Tape::new()`）で bit 完全一致することを確認
/// する（`ScalarUnaryOp::apply` が forward 数式の単一情報源のため、
/// バックエンド実装〈CPU〉とホスト参照実装〈NaiveOps フォールバック〉
/// が同じ `f32` 演算を辿り bit 一致する）。
#[test]
fn cpu_forward_matches_naive_reference() {
    for (name, f) in unary_ops() {
        let data = fixture_for(name);
        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&data);
        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let x_naive = naive_tape.make_var(&data);
        assert_eq!(
            f32_bits(&f(&x_cpu).unwrap().to_tensor()),
            f32_bits(&f(&x_naive).unwrap().to_tensor()),
            "{name}: forward が CPU と NaiveOps で bit 一致しない"
        );
    }

    let data = f32_positive_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    assert_eq!(
        f32_bits(&pow_scalar(&x_cpu, 2.5).unwrap().to_tensor()),
        f32_bits(&pow_scalar(&x_naive, 2.5).unwrap().to_tensor()),
        "pow_scalar: forward が CPU と NaiveOps で bit 一致しない"
    );
}

/// backward が CPU と NaiveOps で一致することを確認する。区分定数
/// （floor/ceil/round/sign）は恒等的に 0 のため bit 完全一致、
/// それ以外（reciprocal/rsqrt/erf/pow_scalar）は REQ-2 統一複合判定
/// （`fandhe_ai_backend_cpu::parity::assert_parity`）で比較する。
#[test]
fn cpu_backward_matches_naive_reference() {
    let weight = f32_weight();

    for (name, f) in unary_ops() {
        let data = fixture_for(name);
        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&data);
        let w_cpu = cpu_tape.make_var(&weight);
        let y_cpu = f(&x_cpu).unwrap();
        let loss_cpu = y_cpu.mul(&w_cpu).unwrap().sum(None).unwrap();
        let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
        let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();

        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let x_naive = naive_tape.make_var(&data);
        let w_naive = naive_tape.make_var(&weight);
        let y_naive = f(&x_naive).unwrap();
        let loss_naive = y_naive.mul(&w_naive).unwrap().sum(None).unwrap();
        let grads_naive = naive_tape.backward(&loss_naive).unwrap();
        let dx_naive = grads_naive.get(&x_naive).unwrap().unwrap().clone();

        if matches!(name, "floor" | "ceil" | "round" | "sign") {
            assert_eq!(
                f32_bits(&dx_cpu),
                f32_bits(&dx_naive),
                "{name}: backward（区分定数）が CPU と NaiveOps で bit 一致しない"
            );
        } else {
            fandhe_ai_backend_cpu::parity::assert_parity(
                &format!("{name} backward: cpu vs naive"),
                dx_cpu.host_slice().as_ref(),
                dx_naive.host_slice().as_ref(),
            );
        }
    }

    let data = f32_positive_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let w_cpu = cpu_tape.make_var(&weight);
    let loss_cpu = pow_scalar(&x_cpu, 2.5)
        .unwrap()
        .mul(&w_cpu)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let w_naive = naive_tape.make_var(&weight);
    let loss_naive = pow_scalar(&x_naive, 2.5)
        .unwrap()
        .mul(&w_naive)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().unwrap().clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "pow_scalar backward: cpu vs naive",
        dx_cpu.host_slice().as_ref(),
        dx_naive.host_slice().as_ref(),
    );
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: Mac／DGX Spark GB10 実機セッションへ
// 申し送る（`docs/perf/logs/scalar-unary-ops-2145/README.md`）。
// ---------------------------------------------------------------------

/// forward（8 種）が CPU と Metal 実機で bit 完全一致することを確認する。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/scalar-unary-ops-2145/README.md 参照"]
fn metal_forward_matches_cpu_reference() {
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");

    for (name, f) in unary_ops() {
        let data = fixture_for(name);
        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&data);
        let x_metal = metal_tape.make_var(&data);
        assert_eq!(
            f32_bits(&f(&x_cpu).unwrap().to_tensor()),
            f32_bits(&f(&x_metal).unwrap().to_tensor()),
            "{name}: forward が CPU と Metal で bit 一致しない"
        );
    }

    let data = f32_positive_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let x_metal = metal_tape.make_var(&data);
    assert_eq!(
        f32_bits(&pow_scalar(&x_cpu, 2.5).unwrap().to_tensor()),
        f32_bits(&pow_scalar(&x_metal, 2.5).unwrap().to_tensor())
    );
}

/// forward（8 種）が CPU と CUDA 実機（DGX Spark GB10）で bit 完全一致
/// することを確認する（上記 Metal 版と対称）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/scalar-unary-ops-2145/README.md 参照"]
fn cuda_forward_matches_cpu_reference() {
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");

    for (name, f) in unary_ops() {
        let data = fixture_for(name);
        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&data);
        let x_cuda = cuda_tape.make_var(&data);
        assert_eq!(
            f32_bits(&f(&x_cpu).unwrap().to_tensor()),
            f32_bits(&f(&x_cuda).unwrap().to_tensor()),
            "{name}: forward が CPU と CUDA で bit 一致しない"
        );
    }

    let data = f32_positive_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let x_cuda = cuda_tape.make_var(&data);
    assert_eq!(
        f32_bits(&pow_scalar(&x_cpu, 2.5).unwrap().to_tensor()),
        f32_bits(&pow_scalar(&x_cuda, 2.5).unwrap().to_tensor())
    );
}

/// backward（8 種）が CPU と Metal 実機で一致することを確認する
/// （判定方針は `cpu_backward_matches_naive_reference` と同じ）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/scalar-unary-ops-2145/README.md 参照"]
fn metal_backward_matches_cpu_reference() {
    let weight = f32_weight();
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");

    for (name, f) in unary_ops() {
        let data = fixture_for(name);
        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&data);
        let w_cpu = cpu_tape.make_var(&weight);
        let loss_cpu = f(&x_cpu).unwrap().mul(&w_cpu).unwrap().sum(None).unwrap();
        let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
        let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();

        let x_metal = metal_tape.make_var(&data);
        let w_metal = metal_tape.make_var(&weight);
        let loss_metal = f(&x_metal)
            .unwrap()
            .mul(&w_metal)
            .unwrap()
            .sum(None)
            .unwrap();
        let grads_metal = metal_tape.backward(&loss_metal).unwrap();
        let dx_metal = grads_metal.get(&x_metal).unwrap().unwrap().clone();

        if matches!(name, "floor" | "ceil" | "round" | "sign") {
            assert_eq!(
                f32_bits(&dx_cpu),
                f32_bits(&dx_metal),
                "{name}: backward mismatch"
            );
        } else {
            fandhe_ai_backend_cpu::parity::assert_parity(
                &format!("{name} backward: cpu vs metal"),
                dx_cpu.host_slice().as_ref(),
                dx_metal.host_slice().as_ref(),
            );
        }
    }

    let data = f32_positive_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let w_cpu = cpu_tape.make_var(&weight);
    let loss_cpu = pow_scalar(&x_cpu, 2.5)
        .unwrap()
        .mul(&w_cpu)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();

    let x_metal = metal_tape.make_var(&data);
    let w_metal = metal_tape.make_var(&weight);
    let loss_metal = pow_scalar(&x_metal, 2.5)
        .unwrap()
        .mul(&w_metal)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads_metal = metal_tape.backward(&loss_metal).unwrap();
    let dx_metal = grads_metal.get(&x_metal).unwrap().unwrap().clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "pow_scalar backward: cpu vs metal",
        dx_cpu.host_slice().as_ref(),
        dx_metal.host_slice().as_ref(),
    );
}

/// backward（8 種）が CPU と CUDA 実機（DGX Spark GB10）で一致することを
/// 確認する（上記 Metal 版と対称）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/scalar-unary-ops-2145/README.md 参照"]
fn cuda_backward_matches_cpu_reference() {
    let weight = f32_weight();
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");

    for (name, f) in unary_ops() {
        let data = fixture_for(name);
        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&data);
        let w_cpu = cpu_tape.make_var(&weight);
        let loss_cpu = f(&x_cpu).unwrap().mul(&w_cpu).unwrap().sum(None).unwrap();
        let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
        let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();

        let x_cuda = cuda_tape.make_var(&data);
        let w_cuda = cuda_tape.make_var(&weight);
        let loss_cuda = f(&x_cuda).unwrap().mul(&w_cuda).unwrap().sum(None).unwrap();
        let grads_cuda = cuda_tape.backward(&loss_cuda).unwrap();
        let dx_cuda = grads_cuda.get(&x_cuda).unwrap().unwrap().clone();

        if matches!(name, "floor" | "ceil" | "round" | "sign") {
            assert_eq!(
                f32_bits(&dx_cpu),
                f32_bits(&dx_cuda),
                "{name}: backward mismatch"
            );
        } else {
            fandhe_ai_backend_cpu::parity::assert_parity(
                &format!("{name} backward: cpu vs cuda"),
                dx_cpu.host_slice().as_ref(),
                dx_cuda.host_slice().as_ref(),
            );
        }
    }

    let data = f32_positive_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let w_cpu = cpu_tape.make_var(&weight);
    let loss_cpu = pow_scalar(&x_cpu, 2.5)
        .unwrap()
        .mul(&w_cpu)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();

    let x_cuda = cuda_tape.make_var(&data);
    let w_cuda = cuda_tape.make_var(&weight);
    let loss_cuda = pow_scalar(&x_cuda, 2.5)
        .unwrap()
        .mul(&w_cuda)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads_cuda = cuda_tape.backward(&loss_cuda).unwrap();
    let dx_cuda = grads_cuda.get(&x_cuda).unwrap().unwrap().clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "pow_scalar backward: cpu vs cuda",
        dx_cpu.host_slice().as_ref(),
        dx_cuda.host_slice().as_ref(),
    );
}
