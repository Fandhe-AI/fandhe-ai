//! `fandhe_ai_autodiff::low_precision_ops`（イシュー #2628。facade 非公開の
//! ため `fandhe_ai_autodiff::low_precision_ops::*` を直接 use する。
//! `crates/autodiff/src/low_precision_ops.rs` モジュール doc 参照）の
//! バックエンド parity テスト（`docs/autodiff-low-precision-op-extension-
//! decision.md` §3.5 の P1・P6）。
//!
//! 属性なし（実 CPU `fandhe_ai::tape()`）:
//! - **P1**: 6 Op × {F16, Bf16} の forward が、実 `CpuBackendOps` の f32 演算で
//!   組んだ丸めオラクル `round(f32_op(round(x)))` と bit 一致する
//!   （`TypedOps<f16/bf16>` の CPU 実装は「f32 昇格 → f32 カーネル → 1 回丸め」）。
//! - **非有限出力**: f16 の表現範囲超過は符号付き `inf`、NaN 入力は NaN へ
//!   伝播する（`compare` へは渡さない別テスト。`inf` は符号一致・NaN は
//!   クラス一致で突合）。
//!
//! `#[ignore]`（P6。`tape_for(Device::Metal)`〈`cfg(target_os = "macos")` 限定〉／
//! `tape_for(Device::Cuda(0))` と CPU tape の同一 dtype・有限出力の突合）:
//! 実機（DGX Spark GB10／Apple Silicon）へ到達できない環境では未実測のまま
//! `docs/perf/logs/low-precision-ops-2628/README.md` へ申し送る。**P6 の不一致は
//! 「判定不能」にせず通常の parity 失敗として扱う**（P4 の第三者比較対象と
//! 異なり同一 dtype 同士・自実装同士の比較のため。tolerance は不変）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::low_precision_ops::{
    add_low_precision, exp_low_precision, matmul_low_precision, mul_low_precision,
    relu_low_precision, tanh_low_precision,
};
use fandhe_ai_tensor_core::{BackendOps, ScalarDType, Tensor};

const DTYPES: [ScalarDType; 2] = [ScalarDType::F16, ScalarDType::Bf16];
const OPS: [&str; 6] = ["matmul", "add", "mul", "relu", "exp", "tanh"];

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data.to_vec(), shape).expect("test fixture: shape 一致")
}

fn round(dtype: ScalarDType, v: f32) -> f32 {
    match dtype {
        ScalarDType::F16 => fandhe_ai_tensor_core::f16::from_f32(v).to_f32(),
        _ => fandhe_ai_tensor_core::bf16::from_f32(v).to_f32(),
    }
}

fn round_t(dtype: ScalarDType, x: &Tensor<f32>) -> Tensor<f32> {
    let data: Vec<f32> = x.host_slice().iter().map(|&v| round(dtype, v)).collect();
    Tensor::new(data, x.shape()).unwrap()
}

/// 入力（op ごとの shape）。bias パターン `[2,3] + [3]` と broadcast を含む。
fn inputs(op: &str) -> Vec<Tensor<f32>> {
    let x = t(&[-1.5, -0.1, 0.0, 0.1, 0.7, 3.3], &[2, 3]);
    match op {
        "matmul" => vec![x, t(&[0.3, -0.2, 1.1, 0.05, -0.6, 0.9], &[3, 2])],
        "add" | "mul" => vec![x, t(&[0.1, -0.7, 2.3], &[3])],
        _ => vec![x],
    }
}

fn run<'t>(op: &str, xs: &[Var<'t>], d: ScalarDType) -> Var<'t> {
    match op {
        "matmul" => matmul_low_precision(&xs[0], &xs[1], d),
        "add" => add_low_precision(&xs[0], &xs[1], d),
        "mul" => mul_low_precision(&xs[0], &xs[1], d),
        "relu" => relu_low_precision(&xs[0], d),
        "exp" => exp_low_precision(&xs[0], d),
        "tanh" => tanh_low_precision(&xs[0], d),
        other => panic!("未知の op: {other}"),
    }
    .unwrap()
}

fn oracle(op: &str, d: ScalarDType, xs: &[Tensor<f32>]) -> Tensor<f32> {
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let r: Vec<Tensor<f32>> = xs.iter().map(|x| round_t(d, x)).collect();
    let y = match op {
        "matmul" => cpu.gemm(&r[0], &r[1]),
        "add" => cpu.add(&r[0], &r[1]),
        "mul" => cpu.mul(&r[0], &r[1]),
        "relu" => cpu.relu(&r[0]),
        "exp" => cpu.exp(&r[0]),
        "tanh" => cpu.tanh(&r[0]),
        other => panic!("未知の op: {other}"),
    }
    .unwrap();
    round_t(d, &y)
}

fn bits(x: &Tensor<f32>) -> Vec<u32> {
    x.host_slice().iter().map(|v| v.to_bits()).collect()
}

/// P1: 実 CPU の forward が丸めオラクルと bit 一致する。
#[test]
fn cpu_forward_matches_rounding_oracle_bit_exact() {
    for d in DTYPES {
        for op in OPS {
            let xs_t = inputs(op);
            let tape = fandhe_ai::tape();
            let xs: Vec<Var<'_>> = xs_t.iter().map(|x| tape.var(x)).collect();
            let y = run(op, &xs, d).to_tensor();
            assert_eq!(bits(&y), bits(&oracle(op, d, &xs_t)), "{op} {d:?}");
        }
    }
}

/// 非有限出力: f16 の表現範囲超過は符号付き `inf`、NaN は NaN へ伝播する。
#[test]
fn cpu_non_finite_outputs_follow_ieee_rounding() {
    let tape = fandhe_ai::tape();
    let d = ScalarDType::F16;
    // exp(12) ≈ 162754 > f16 最大 65504 → +inf。exp(-30) は 0 へ潰れる（有限）。
    let e = exp_low_precision(&tape.var(&t(&[12.0, 0.0, -30.0], &[3])), d)
        .unwrap()
        .to_tensor();
    let e = e.host_slice();
    assert_eq!(e[0], f32::INFINITY);
    assert_eq!(e[1], 1.0);
    assert_eq!(e[2], 0.0);
    // add: 60000 + 60000 → +inf、-60000 + -60000 → -inf（60000 は f16 で厳密）。
    let a = tape.var(&t(&[60000.0, -60000.0, 1.0], &[3]));
    let s = add_low_precision(&a, &a, d).unwrap().to_tensor();
    assert_eq!(
        s.host_slice().as_ref(),
        &[f32::INFINITY, f32::NEG_INFINITY, 2.0]
    );
    // mul: 300 * 300 = 90000 → +inf、300 * -300 → -inf。
    let l = tape.var(&t(&[300.0, 300.0], &[2]));
    let r = tape.var(&t(&[300.0, -300.0], &[2]));
    let m = mul_low_precision(&l, &r, d).unwrap().to_tensor();
    assert_eq!(m.host_slice().as_ref(), &[f32::INFINITY, f32::NEG_INFINITY]);
    // NaN 入力は NaN（payload 不問）。add／exp／tanh で確認。
    let n = tape.var(&t(&[f32::NAN, 1.0], &[2]));
    for out in [
        add_low_precision(&n, &n, d).unwrap(),
        exp_low_precision(&n, d).unwrap(),
        tanh_low_precision(&n, d).unwrap(),
    ] {
        let o = out.to_tensor();
        assert!(o.host_slice()[0].is_nan());
        assert!(o.host_slice()[1].is_finite());
    }
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: Mac／DGX Spark GB10 実機セッションへ
// 申し送る（`docs/perf/logs/low-precision-ops-2628/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    for d in DTYPES {
        for op in OPS {
            let xs_t = inputs(op);
            let cpu_tape = fandhe_ai::tape();
            let cpu_xs: Vec<Var<'_>> = xs_t.iter().map(|x| cpu_tape.var(x)).collect();
            let y_cpu = run(op, &cpu_xs, d).to_tensor();
            let dev_xs: Vec<Var<'_>> = xs_t.iter().map(|x| device_tape.var(x)).collect();
            let y_dev = run(op, &dev_xs, d).to_tensor();
            assert_eq!(y_cpu.shape(), y_dev.shape(), "{op} {d:?}: shape");
            fandhe_ai_backend_cpu::parity::assert_parity(
                &format!("{op} {d:?}: cpu vs {label}"),
                y_dev.host_slice().as_ref(),
                y_cpu.host_slice().as_ref(),
            );
        }
    }
}

/// 6 Op × {F16, Bf16} の CPU／Metal 実機比較（Metal の bf16 可用性は未検証。
/// accessor が `None` なら `Unsupported` で失敗し、それ自体が実測結果になる）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/low-precision-ops-2628/README.md 参照"]
fn metal_low_precision_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// 6 Op × {F16, Bf16} の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/low-precision-ops-2628/README.md 参照"]
fn cuda_low_precision_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
