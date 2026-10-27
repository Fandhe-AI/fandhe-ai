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
//!
//! `#[ignore]` の大形状診断（#2994。`cuda_low_precision_matmul_large_shape_report`）:
//! f16 の大形状 matmul が CUDA の Tensor Core 経路（`mma.sync`）に乗る形状を
//! CPU と突き合わせ、形状ごとの 4 指標と ulp 分布を記録する。**記録専用**で
//! 統一複合判定の fail 数は assert しない（tolerance 緩和ではない。扱いの提案は
//! README に書く）。
//!
//! `#[ignore]` の非後退ゲート（#3617。`cuda_low_precision_matmul_f16_nonzero_fail_shapes_do_not_regress`）:
//! 上記の診断は記録専用のため、f16 が `f16:Mma` 経路に乗り統一複合判定で fail≠0 となる
//! 3 形状（512³・1024³・256×256×1024）は後退しても検出できない。この 3 形状だけを
//! `crates/backend-cuda/tests/common/parity_baseline.rs` の
//! `ParityPath::FacadeLowPrecisionMatmulF16` 行（baseline 非後退方式。判定は
//! `assert_no_parity_regression` をそのまま使い複製しない）で検査する。
//! 厳密ゼロ fail が成立する 5 形状・bf16・tolerance 定数は不変。
//! 構造テスト `facade_f16_low_precision_baselines_cover_exactly_the_approved_shapes`
//! は GPU 不要で CI でも走り、fixture の形状集合が承認範囲と一致することを固定する。

// 共有 fixture をクレート外から取り込む（`crates/backend-cuda/src/gemm.rs` の
// 同型の `#[path]` 取り込みが前例。判定ロジックを facade 側へ複製しないため）。
// 注意: パッケージ化される facade の tarball には共有ファイルが入らないため、
// 下流が tarball から `cargo test` するとこのテストファイルはコンパイルできない
// （`cargo publish` の verify は lib のみのため影響なし。
// docs/cuda-tensor-core-parity-judgment-decision.md §7 に残存リスクとして記録）。
#[path = "../../backend-cuda/tests/common/parity_baseline.rs"]
mod parity_baseline;

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
/// 実機診断（#2629）: 12 組合せすべての CPU 対 実機の差分を 1 行ずつ出力し、
/// 全組合せを出し終えてから同じ統一複合判定（`compare`／`passes`。tolerance は
/// 新設せず不変）で集約 assert する。上の `assert_device_matches_cpu` は最初の
/// 失敗で panic し残りの組合せが見えなくなるため、実測記録
/// （`docs/perf/logs/low-precision-ops-2628/README.md`）用に別立てしている。
fn report_device_vs_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let mut failed = Vec::new();
    for d in DTYPES {
        for op in OPS {
            let xs_t = inputs(op);
            let cpu_tape = fandhe_ai::tape();
            let cpu_xs: Vec<Var<'_>> = xs_t.iter().map(|x| cpu_tape.var(x)).collect();
            let y_cpu = run(op, &cpu_xs, d).to_tensor();
            let dev_xs: Vec<Var<'_>> = xs_t.iter().map(|x| device_tape.var(x)).collect();
            let y_dev = run(op, &dev_xs, d).to_tensor();
            let (a, b) = (y_dev.host_slice(), y_cpu.host_slice());
            let bit_eq = a
                .iter()
                .zip(b.iter())
                .filter(|(p, q)| p.to_bits() == q.to_bits())
                .count();
            match fandhe_ai_backend_cpu::parity::compare(a.as_ref(), b.as_ref()) {
                Ok(r) => {
                    println!(
                        "REPORT {label} {op} {d:?}: total={} bit_eq={bit_eq} fail={} max_abs={:e} max_rel={:e} passes={}",
                        r.total,
                        r.fail_count,
                        r.max_abs_diff,
                        r.max_rel_err,
                        r.passes()
                    );
                    if !r.passes() {
                        failed.push(format!("{op} {d:?}"));
                    }
                }
                Err(e) => {
                    println!("REPORT {label} {op} {d:?}: error={e:?}");
                    failed.push(format!("{op} {d:?}"));
                }
            }
        }
    }
    assert!(failed.is_empty(), "parity 不合格の組合せ: {failed:?}");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。診断用（#2629）"]
fn metal_low_precision_ops_parity_report() {
    report_device_vs_cpu(Device::Metal, "metal");
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。診断用（#2629）"]
fn cuda_low_precision_ops_parity_report() {
    report_device_vs_cpu(Device::Cuda(0), "cuda");
}

// ---------------------------------------------------------------------
// 大形状診断（#2994）
// ---------------------------------------------------------------------

/// 大形状診断の形状表（m, n, k, seed）。lhs は `seed`、rhs は `seed + 1` の
/// `Xorshift64Star` で [-1, 1) を生成する（`amp_conv_mha_low_precision_backend_
/// parity.rs` と同じ生成法）。`crates/backend-cuda/tests/common/parity_baseline.rs`
/// の `ParityPath::MmaF16` 行とは入力生成経路が異なるため数値は直接比較できない。
/// 後ろ 2 形状は非整列（n%8≠0 または k%8≠0）で WMMA へフォールバックする想定。
const LARGE_SHAPES: &[(usize, usize, usize, u64)] = &[
    (64, 64, 64, 11),
    (128, 128, 128, 12),
    (256, 256, 256, 13),
    (512, 512, 512, 14),
    (1024, 1024, 1024, 15),
    (256, 256, 1024, 16),
    (128, 128, 100, 17),
    (65, 63, 33, 18),
];

/// f16／bf16 のビット列を「大小順に単調な整数」へ写す（ulp 差の算出用）。
fn ordered_bits(dtype: ScalarDType, v: f32) -> i32 {
    let raw = match dtype {
        ScalarDType::F16 => fandhe_ai_tensor_core::f16::from_f32(v).to_bits(),
        _ => fandhe_ai_tensor_core::bf16::from_f32(v).to_bits(),
    } as i32;
    if raw & 0x8000 != 0 {
        -(raw & 0x7fff)
    } else {
        raw
    }
}

/// 経路の**予測**（観測ではない）。f16 は `gemm_auto.rs::select_f16_matrix_unit_impl`
/// と同条件（n%8==0 && k%8==0 && m.div_ceil(64)<=65535 なら mma）を再現する。
/// bf16 は `typed_bf16.rs` が f32 昇格後 `gemm_fp32_strict` へ委譲するため
/// Tensor Core を通らない。
fn predicted_route(
    dtype: ScalarDType,
    mma_available: bool,
    (m, n, k): (usize, usize, usize),
) -> &'static str {
    if dtype != ScalarDType::F16 {
        return "fp32_strict(non-TC)";
    }
    if mma_available && n % 8 == 0 && k % 8 == 0 && m.div_ceil(64) <= 65535 {
        "f16:Mma"
    } else {
        "f16:Wmma-or-Tiled"
    }
}

/// 大形状診断・ゲート共通の入力生成（lhs は `seed`、rhs は `seed + 1`）。
/// 診断とゲートが同じ関数を使い、baseline の provenance（入力）を揃える。
fn large_shape_inputs(m: usize, n: usize, k: usize, seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut a = bench_harness::rng::Xorshift64Star::new(seed);
    let mut b = bench_harness::rng::Xorshift64Star::new(seed + 1);
    let lhs = Tensor::new(a.fill_vec(m * k), &[m, k]).unwrap();
    let rhs = Tensor::new(b.fill_vec(k * n), &[k, n]).unwrap();
    (lhs, rhs)
}

/// 大形状 matmul の CPU 対 CUDA 診断。assert は shape 一致・`compare` が `Ok`・
/// 全要素有限のみ。統一複合判定の fail 数は記録だけで panic しない
/// （Tensor Core 経路は結合順序が単一 K ループと異なり、決定記録
/// `docs/cuda-tensor-core-parity-judgment-decision.md` §1.1 のとおり fail 非ゼロが
/// 想定されるため。tolerance・baseline・`compare`／`passes` には触れない）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。診断用（#2994）"]
fn cuda_low_precision_matmul_large_shape_report() {
    let dev = fandhe_ai_backend_cuda::CudaDevice::new(0).expect("CUDA 実機が必要");
    let cc = dev.compute_capability();
    let auto = fandhe_ai_backend_cuda::CudaGemmAuto::new(&dev).expect("CudaGemmAuto 構築");
    let mma = auto.mma_available();
    println!(
        "ROUTE cuda cc={cc:?} mma_available={mma} reason={:?} (predicted)",
        auto.mma_unavailable_reason()
    );
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0)).expect("CUDA tape");
    let (mut zero, mut nonzero) = (Vec::new(), Vec::new());
    for d in DTYPES {
        for &(m, n, k, seed) in LARGE_SHAPES {
            let (lhs, rhs) = large_shape_inputs(m, n, k, seed);
            let cpu_tape = fandhe_ai::tape();
            let y_cpu = matmul_low_precision(&cpu_tape.var(&lhs), &cpu_tape.var(&rhs), d)
                .unwrap()
                .to_tensor();
            let y_dev = matmul_low_precision(&cuda_tape.var(&lhs), &cuda_tape.var(&rhs), d)
                .unwrap()
                .to_tensor();
            assert_eq!(y_cpu.shape(), y_dev.shape(), "{d:?} {m}x{n}x{k}: shape");
            let (dv, cv) = (y_dev.host_slice(), y_cpu.host_slice());
            assert!(
                dv.iter().all(|v| v.is_finite()),
                "{d:?} {m}x{n}x{k}: 非有限出力"
            );
            let bit_eq = dv
                .iter()
                .zip(cv.iter())
                .filter(|(p, q)| p.to_bits() == q.to_bits())
                .count();
            let (mut u0, mut u1, mut ugt1) = (0usize, 0usize, 0usize);
            for (p, q) in dv.iter().zip(cv.iter()) {
                match (ordered_bits(d, *p) - ordered_bits(d, *q)).abs() {
                    0 => u0 += 1,
                    1 => u1 += 1,
                    _ => ugt1 += 1,
                }
            }
            let r = fandhe_ai_backend_cpu::parity::compare(dv.as_ref(), cv.as_ref())
                .unwrap_or_else(|e| panic!("{d:?} {m}x{n}x{k}: compare Err {e:?}"));
            assert_eq!(r.total, m * n);
            println!(
                "REPORT cuda matmul-large {d:?} {m}x{n}x{k} seed={seed} route={} total={} bit_eq={bit_eq} fail={} mean_abs={:e} max_abs={:e} max_rel={:e} ulp0={u0} ulp1={u1} ulp_gt1={ugt1} strict_pass={}",
                predicted_route(d, mma, (m, n, k)),
                r.total,
                r.fail_count,
                r.mean_abs_diff,
                r.max_abs_diff,
                r.max_rel_err,
                r.passes()
            );
            let tag = format!("{d:?}@{m}x{n}x{k}");
            if r.fail_count == 0 {
                zero.push(tag)
            } else {
                nonzero.push(tag)
            }
        }
    }
    println!("SUMMARY strict_zero_fail={zero:?} nonzero_fail={nonzero:?}");
}

// ---------------------------------------------------------------------
// f16 Tensor Core 経路 fail≠0 3 形状の非後退ゲート（#3617）
// ---------------------------------------------------------------------

/// 所有者承認済み（#3617 コメント 6432022446）の対象形状 `(m, n, k, seed)`。
const F16_NONZERO_FAIL_SHAPES: [(usize, usize, usize, u64); 3] = [
    (512, 512, 512, 14),
    (1024, 1024, 1024, 15),
    (256, 256, 1024, 16),
];

/// 厳密ゼロ fail が成り立つ f16 形状（baseline 化の対象外。混入を禁じる）。
const F16_STRICT_ZERO_SHAPES: [(usize, usize, usize, u64); 5] = [
    (64, 64, 64, 11),
    (128, 128, 128, 12),
    (256, 256, 256, 13),
    (128, 128, 100, 17),
    (65, 63, 33, 18),
];

fn facade_f16_rows() -> Vec<&'static parity_baseline::ParityBaseline> {
    parity_baseline::BASELINES
        .iter()
        .filter(|b| b.path == parity_baseline::ParityPath::FacadeLowPrecisionMatmulF16)
        .collect()
}

/// GPU 不要の構造テスト。fixture の形状集合が承認範囲とちょうど一致し、
/// ceiling が全て確定済みで、tolerance 定数が固定されていることを確かめる。
#[test]
fn facade_f16_low_precision_baselines_cover_exactly_the_approved_shapes() {
    let rows = facade_f16_rows();
    assert_eq!(rows.len(), F16_NONZERO_FAIL_SHAPES.len(), "対象行は 3 件");
    let mut got: Vec<_> = rows
        .iter()
        .map(|b| (b.m as usize, b.n as usize, b.k as usize, b.seed))
        .collect();
    let mut want = F16_NONZERO_FAIL_SHAPES.to_vec();
    got.sort();
    want.sort();
    assert_eq!(got, want, "baseline 行の形状集合が承認範囲と不一致");
    for shape in &got {
        assert!(LARGE_SHAPES.contains(shape), "{shape:?} は LARGE_SHAPES 外");
        assert!(
            !F16_STRICT_ZERO_SHAPES.contains(shape),
            "{shape:?} は厳密ゼロ fail 形状（baseline 化対象外）"
        );
    }
    for b in rows {
        assert!(!b.baseline_provenance_unconfirmed, "{}", b.context);
        assert!(b.baseline_max_abs_diff_ceiling.is_some(), "{}", b.context);
        assert!(b.baseline_max_rel_err_ceiling.is_some(), "{}", b.context);
        assert_eq!(b.total, (b.m as usize) * (b.n as usize), "{}", b.context);
    }
    parity_baseline::assert_tolerance_constants_pinned();
}

/// f16 が `f16:Mma` に乗り fail≠0 となる 3 形状の非後退ゲート。facade の
/// `matmul_low_precision(F16)` を CUDA tape と CPU tape で実行し、
/// `assert_no_parity_regression` で baseline 行と突き合わせる
/// （判定式は複製しない。診断と同じ引数順 device, cpu）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。#3617 非後退ゲート"]
fn cuda_low_precision_matmul_f16_nonzero_fail_shapes_do_not_regress() {
    let rows = facade_f16_rows();
    assert!(!rows.is_empty(), "対象 baseline 行が 0 件（fail-closed）");
    let dev = fandhe_ai_backend_cuda::CudaDevice::new(0).expect("CUDA 実機が必要");
    let auto = fandhe_ai_backend_cuda::CudaGemmAuto::new(&dev).expect("CudaGemmAuto 構築");
    assert!(
        auto.mma_available(),
        "mma.sync が使えない環境では f16:Mma の baseline を判定できない: {:?}",
        auto.mma_unavailable_reason()
    );
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0)).expect("CUDA tape");
    let mut failures = Vec::new();
    for b in rows {
        let (m, n, k) = (b.m as usize, b.n as usize, b.k as usize);
        assert_eq!(
            predicted_route(ScalarDType::F16, true, (m, n, k)),
            "f16:Mma",
            "{}: Mma 経路に乗らない形状は本ゲートの対象外",
            b.context
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (lhs, rhs) = large_shape_inputs(m, n, k, b.seed);
            let cpu_tape = fandhe_ai::tape();
            let y_cpu =
                matmul_low_precision(&cpu_tape.var(&lhs), &cpu_tape.var(&rhs), ScalarDType::F16)
                    .unwrap()
                    .to_tensor();
            let y_dev =
                matmul_low_precision(&cuda_tape.var(&lhs), &cuda_tape.var(&rhs), ScalarDType::F16)
                    .unwrap()
                    .to_tensor();
            assert_eq!(y_cpu.shape(), y_dev.shape(), "{}: shape", b.context);
            let (dv, cv) = (y_dev.host_slice(), y_cpu.host_slice());
            assert!(
                dv.iter().all(|v| v.is_finite()),
                "{}: 非有限出力",
                b.context
            );
            let report = fandhe_ai_backend_cpu::parity::compare(dv.as_ref(), cv.as_ref())
                .unwrap_or_else(|e| panic!("{}: compare Err {e:?}", b.context));
            parity_baseline::assert_no_parity_regression(b.context, &report, b);
        }));
        if let Err(err) = result {
            let msg = err
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic (詳細不明)".to_string());
            failures.push(format!("{}: {msg}", b.context));
        }
    }
    assert!(
        failures.is_empty(),
        "f16 非後退ゲート FAIL（{} 件）:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
