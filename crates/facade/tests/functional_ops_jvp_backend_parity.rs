//! `fandhe_ai_autodiff::functional_ops::{jvp, jacfwd}`（イシュー #2940・親 #2939）の double-VJP 手順の
//! バックエンド間 parity テスト（イシュー #2942。`functional_ops_backend_parity.rs`〈#2881〉と同型）。
//!
//! **ミラー方針**: `jvp`／`jacfwd` は #2956 で facade `Tape::jvp`／`Tape::jacfwd` として公開済みだが、
//! 本ファイルは #2942 の実測対象（`docs/perf/logs/` の GB10 実測記録が指す）としてミラーを意図的に維持し、
//! ロジックは変えない。#2880 の `double_vjp_probe`
//! （`crates/autodiff/tests/double_vjp_feasibility.rs`）と同様に、公開 API だけで #2940 の手順
//! （追跡ありの葉 `u` = 全要素 1、`s = output ⊙ u`〈`sum` は足さない〉、`backward_create_graph`、
//! `jvp` は子テープ上で `g ⊙ v` を backward、`jacfwd` は `g` の要素ごとに子テープを backward）を
//! 逐語ミラーする。本物の API 呼び出しとの突き合わせは `functional_transforms_jvp_facade.rs`（CPU）が担う。
//! ヘルパー名に `jvp`／`jacfwd` の `fn` 名を使わないのは facade の保留ガード（`api_surface.rs`）に合わせるため。
//!
//! 属性なし: 実 `CpuBackendOps` tape と `Tape::new()`（`NaiveOps`）の突合、`jacobian_ops::jacobian`
//! との突合、手計算の閉形式。判定は REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）。
//!
//! `#[ignore]`: CUDA／Metal（`cfg(target_os = "macos")` 限定）対 CPU。実機に届かない環境では未実測のまま
//! `docs/perf/logs/functional-transforms-jvp-2942/README.md` へ申し送る。新規カーネルはなく、確認対象は
//! 既存カーネルの新しい呼び出し形（追跡あり余接葉を含む親 `backward_create_graph` と子テープ上の
//! 第 2 段 VJP）である。形状は小さく Metal split-K が発動する形状は使わない。

use fandhe_ai_autodiff::jacobian_ops::jacobian;
use fandhe_ai_autodiff::{Tape, Var};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn cpu_tape() -> Tape {
    Tape::new_with_ops(Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()))
}

fn seq(n: usize, a: f32, b: f32) -> Vec<f32> {
    (0..n).map(|i| a * (i as f32) + b).collect()
}

/// 二つのフィクスチャ（`supports_create_graph()` が真の Op のみ）。
/// 0: MLP 形 `y = (tanh(x W1 + b) W2) ⊙ sigmoid(x W3)`（`x: [2, 3]` → `y: [2, 2]`）。
/// 1: 要素ごと `y = narrow((x ⊙ exp(x))ᵀ)`（`x: [2, 3]` → `y: [2, 2]`、非 contiguous 経路を含む）。
fn build(tape: &Tape, case: usize) -> (Var<'_>, Var<'_>) {
    let x = tape.var(&t(vec![0.3, -0.7, 1.1, 0.5, 0.2, -0.4], &[2, 3]));
    let y = if case == 0 {
        let w1 = tape.var_no_grad(&t(seq(12, 0.1, -0.5), &[3, 4]));
        let b = tape.var_no_grad(&t(vec![0.1, -0.2, 0.05, 0.3], &[4]));
        let w2 = tape.var_no_grad(&t(seq(8, -0.05, 0.2), &[4, 2]));
        let w3 = tape.var_no_grad(&t(seq(6, -0.1, 0.3), &[3, 2]));
        let h = x.matmul(&w1).unwrap().add(&b).unwrap().tanh();
        h.matmul(&w2)
            .unwrap()
            .mul(&x.matmul(&w3).unwrap().sigmoid())
            .unwrap()
    } else {
        x.mul(&x.exp())
            .unwrap()
            .transpose(0, 1)
            .unwrap()
            .narrow(0, 1, 2)
            .unwrap()
    };
    (y, x)
}

/// `u`（全要素 1・追跡あり）と `s = output ⊙ u` を足し、`g = Jᵀu`（子テープ上）と `u` の子写しを返す
/// （#2940 `double_vjp_stage` のミラー）。
fn double_vjp_stage<'c>(
    tape: &Tape,
    output: &Var<'_>,
    input: &Var<'_>,
    child: &'c Tape,
) -> (Var<'c>, Var<'c>) {
    let u = tape.var(&Tensor::ones(&shape_of(output)).unwrap());
    let s = output.mul(&u).unwrap();
    let cg = tape.backward_create_graph(&s, child).unwrap();
    let cu = cg.child_var(&u).unwrap().expect("child_var(u) は Some");
    let g = cg.grad(input).unwrap().expect("grad(input) は Some");
    (g, cu)
}

/// `J·v` の double-VJP ミラー（`jvp` 相当）。結果 shape は `output` の shape。
fn double_vjp_jv(
    tape: &Tape,
    output: &Var<'_>,
    input: &Var<'_>,
    tangent: &Tensor<f32>,
    child: &Tape,
) -> (Vec<usize>, Vec<f32>) {
    let (g, cu) = double_vjp_stage(tape, output, input, child);
    let prod = g.mul(&child.var_no_grad(tangent)).unwrap();
    let grads = child.backward(&prod).unwrap();
    let h = grads.get(&cu).unwrap().expect("∂/∂u は Some");
    assert_eq!(h.shape(), shape_of(output).as_slice(), "jv の shape");
    (h.shape().to_vec(), h.host_slice().into_owned())
}

/// ヤコビアン全体の double-VJP ミラー（`jacfwd` 相当）。`g` を平坦化し要素ごとに子テープを backward する。
/// 結果は行優先 `m × n`（shape は `output.shape ++ input.shape`）。
fn double_vjp_jac(
    tape: &Tape,
    output: &Var<'_>,
    input: &Var<'_>,
    child: &Tape,
) -> (Vec<usize>, Vec<f32>) {
    let m: usize = shape_of(output).iter().product();
    let n: usize = shape_of(input).iter().product();
    let (g, cu) = double_vjp_stage(tape, output, input, child);
    let flat = g
        .reshape(&[n])
        .expect("g は contiguous でなければ reshape できない");
    let mut data = vec![0.0f32; m * n];
    for k in 0..n {
        let g_k = flat.narrow(0, k, 1).unwrap();
        let grads = child.backward(&g_k).unwrap();
        let col = grads.get(&cu).unwrap().expect("∂g_k/∂u は Some");
        let host = col.host_slice();
        assert_eq!(host.len(), m, "列の要素数");
        for (i, v) in host.iter().enumerate() {
            data[i * n + k] = *v;
        }
    }
    let mut shape = shape_of(output);
    shape.extend_from_slice(&shape_of(input));
    (shape, data)
}

/// ホスト f64 蓄積の `J·v`（最後に 1 回 `f32` へ）。閾値は持たない。
fn jacobian_times_vector_f64(jac: &[f32], m: usize, n: usize, v: &[f32]) -> Vec<f32> {
    assert_eq!(jac.len(), m * n);
    assert_eq!(v.len(), n);
    (0..m)
        .map(|i| {
            (0..n)
                .map(|k| f64::from(jac[i * n + k]) * f64::from(v[k]))
                .sum::<f64>() as f32
        })
        .collect()
}

struct Outs {
    jv_shape: Vec<usize>,
    jv: Vec<f32>,
    jac_shape: Vec<usize>,
    jac: Vec<f32>,
    ref_jac: Vec<f32>,
}

const TANGENT: [f32; 6] = [1.0, -0.5, 0.25, 0.0, 2.0, -1.5];

/// 親・子テープは同じバックエンドから作る（`backward_create_graph` がデバイス一致を検査する）。
/// ミラーごとに新しい親テープ・子テープを構築する（子テープは呼び出し後に再利用しない契約）。
fn compute(make_tape: &dyn Fn() -> Tape, case: usize) -> Outs {
    let tape = make_tape();
    let (y, x) = build(&tape, case);
    let (jv_shape, jv) = double_vjp_jv(&tape, &y, &x, &t(TANGENT.to_vec(), &[2, 3]), &make_tape());

    let tape = make_tape();
    let (y, x) = build(&tape, case);
    let (jac_shape, jac) = double_vjp_jac(&tape, &y, &x, &make_tape());

    let tape = make_tape();
    let (y, x) = build(&tape, case);
    let ref_jac = jacobian(&tape, &y, &x).unwrap().host_slice().into_owned();
    Outs {
        jv_shape,
        jv,
        jac_shape,
        jac,
        ref_jac,
    }
}

fn assert_internally_consistent(label: &str, o: &Outs) {
    let p = fandhe_ai_backend_cpu::parity::assert_parity;
    p(
        &format!("{label}: jacfwd 相当 vs jacobian"),
        &o.jac,
        &o.ref_jac,
    );
    let jv_ref = jacobian_times_vector_f64(&o.ref_jac, 4, 6, &TANGENT);
    p(&format!("{label}: jvp 相当 vs jacobian·v"), &o.jv, &jv_ref);
}

fn assert_outs_match(label: &str, a: &Outs, b: &Outs) {
    assert_eq!(a.jv_shape, b.jv_shape, "{label}: jv shape");
    assert_eq!(a.jac_shape, b.jac_shape, "{label}: jac shape");
    let p = fandhe_ai_backend_cpu::parity::assert_parity;
    p(&format!("{label}: jv"), &a.jv, &b.jv);
    p(&format!("{label}: jac"), &a.jac, &b.jac);
    p(&format!("{label}: jacobian"), &a.ref_jac, &b.ref_jac);
}

#[test]
fn cpu_double_vjp_matches_naive_reference() {
    for case in 0..2 {
        let cpu = compute(&cpu_tape, case);
        let naive = compute(&Tape::new, case);
        assert_eq!(cpu.jv_shape, vec![2, 2], "case {case}");
        assert_eq!(cpu.jac_shape, vec![2, 2, 2, 3], "case {case}");
        assert_outs_match(&format!("case {case} cpu vs naive"), &cpu, &naive);
    }
}

#[test]
fn cpu_double_vjp_matches_jacobian() {
    for case in 0..2 {
        let cpu = compute(&cpu_tape, case);
        assert_internally_consistent(&format!("case {case} cpu"), &cpu);
        // J ≢ 0 の確認（全ゼロ同士の自明な一致を排除する）。
        assert!(
            cpu.ref_jac.iter().any(|v| v.abs() > 1e-3),
            "case {case}: J ≡ 0"
        );
        // 列 k が接ベクトル e_k の jv と一致する（#2940 の J6 と同趣旨）。
        let (m, n) = (4usize, 6usize);
        for k in 0..n {
            let mut e = vec![0.0f32; n];
            e[k] = 1.0;
            let tape = cpu_tape();
            let (y, x) = build(&tape, case);
            let (_, col) = double_vjp_jv(&tape, &y, &x, &t(e, &[2, 3]), &cpu_tape());
            let want: Vec<f32> = (0..m).map(|i| cpu.jac[i * n + k]).collect();
            fandhe_ai_backend_cpu::parity::assert_parity(
                &format!("case {case}: 列 {k} vs jv(e_k)"),
                &col,
                &want,
            );
        }
    }
}

/// 手計算の期待値を固定する（両経路が同じ誤りで一致していないことの確認）。
/// `y = x ⊙ x`・`x = [1, 2]`・`v = [1, 1]` は `2x ⊙ v = [2, 4]`、ヤコビアンは `diag(2x) = [[2, 0], [0, 4]]`。
#[test]
fn cpu_double_vjp_matches_hand_computed_values() {
    let tape = cpu_tape();
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = x.mul(&x).unwrap();
    let (_, jv) = double_vjp_jv(&tape, &y, &x, &t(vec![1.0, 1.0], &[2]), &cpu_tape());
    fandhe_ai_backend_cpu::parity::assert_parity("手計算 jv", &jv, &[2.0, 4.0]);

    let tape = cpu_tape();
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = x.mul(&x).unwrap();
    let (shape, jac) = double_vjp_jac(&tape, &y, &x, &cpu_tape());
    assert_eq!(shape, vec![2, 2]);
    fandhe_ai_backend_cpu::parity::assert_parity("手計算 jac", &jac, &[2.0, 0.0, 0.0, 4.0]);
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/functional-transforms-jvp-2942/README.md`）。
// ---------------------------------------------------------------------

/// 実機 `BackendOps` を結線した tape（子テープも同一バックエンド）と CPU の `compute` 結果を比較する。
fn assert_device_matches_cpu(make_ops: impl Fn() -> Box<dyn BackendOps + Send>, label: &str) {
    let make_dev = || Tape::new_with_ops(make_ops());
    for case in 0..2 {
        let cpu = compute(&cpu_tape, case);
        let dev = compute(&make_dev, case);
        assert_internally_consistent(&format!("case {case} {label}"), &dev);
        assert_outs_match(&format!("case {case} cpu vs {label}"), &cpu, &dev);
    }
}

/// jvp／jacfwd 相当の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/functional-transforms-jvp-2942/README.md 参照"]
fn metal_double_vjp_jvp_jacfwd_match_cpu_reference() {
    assert_device_matches_cpu(
        || Box::new(fandhe_ai_backend_metal::MetalBackendOps::new()),
        "metal",
    );
}

/// jvp／jacfwd 相当の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/functional-transforms-jvp-2942/README.md 参照"]
fn cuda_double_vjp_jvp_jacfwd_match_cpu_reference() {
    assert_device_matches_cpu(
        || Box::new(fandhe_ai_backend_cuda::CudaBackendOps::new(0)),
        "cuda",
    );
}

/// `Var::shape` は crate 内限定のため `value()` 経由で取る（借用はこの式の中で閉じる）。
fn shape_of(v: &Var<'_>) -> Vec<usize> {
    v.value().shape().to_vec()
}
