//! 本物の facade `Tape::jvp`・`Tape::jacfwd`（イシュー #2956・決定記録
//! `docs/autodiff-functional-transforms-design.md` §27・§28）を `tape_for(device)` の実機 tape（CUDA／Metal）で呼び、
//! 同じ関数の `Device::Cpu` 結果と比べる `#[ignore]` parity テスト（イシュー #3167）。
//!
//! **ミラーとの関係**: `functional_ops_jvp_backend_parity.rs`（#2942。GB10 実測記録の対象）は置き換えずそのまま維持する。
//! 複数入力版は `functional_transforms_jvp_multi_backend_parity.rs` が担当し、本ファイルは単一入力だけを担当する。
//! CPU 版は `functional_transforms_jvp_facade.rs`（属性なし。フィクスチャの出典）。
//!
//! 判定は REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）。新しいカーネルはなく、確認対象は
//! 既存カーネルの新しい呼び出し形（追跡あり余接葉を含む親 `backward_create_graph` と子テープ上の第 2 段 VJP）である。
//! 形状は小さく、Metal split-K が発動する形状は使わない。
//!
//! 演算を基本グループと拡張グループ（#3087）に分けるのは、拡張側の Op が実機で型付きエラーや判定外になっても
//! 基本グループの結果が埋もれず、測定者がどの Op で失敗したかを記録できるようにするため。
//! 実測は #3168（GB10）・#3169（Mac mini）の担当で、本ファイルでは未実測のまま
//! `docs/perf/logs/functional-transforms-jvp-2942/README.md` へ申し送る。

use fandhe_ai::{Device, Tape, Tensor, Var};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn host(x: &Tensor<f32>) -> Vec<f32> {
    x.host_slice().into_owned()
}

/// `jacobian`（行優先 `m × n`）と接ベクトルの積を f64 で蓄積し最後に 1 回だけ f32 へ丸めた参照値。
fn reference_jv(jac: &[f32], v: &[f32]) -> Vec<f32> {
    let n = v.len();
    assert_eq!(jac.len() % n, 0);
    jac.chunks(n)
        .map(|row| {
            row.iter()
                .zip(v)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>() as f32
        })
        .collect()
}

/// 1 デバイス上の 1 ケースの結果（形状とホスト値）。
struct Outs {
    y_shape: Vec<usize>,
    jv_shape: Vec<usize>,
    jv: Vec<f32>,
    jv_ref: Vec<f32>,
    jac_shape: Vec<usize>,
    jac: Vec<f32>,
    ref_jac: Vec<f32>,
}

/// `device` 上で `jacobian`（同デバイスの reverse-mode 参照）・`jvp`・`jacfwd` を同一親テープで呼ぶ。
/// 子テープは単回使用の契約のため呼び出しごとに新しく作り、`backward_create_graph` がデバイス不一致を
/// 拒否するため親と同じデバイスで作る。
fn compute_on<F>(device: Device, x_data: &[f32], x_shape: &[usize], v: &[f32], build: &F) -> Outs
where
    F: for<'a> Fn(&'a Tape, &Var<'a>) -> Var<'a>,
{
    let tp = fandhe_ai::tape_for(device).expect("tape_for");
    let x = tp.var(&t(x_data.to_vec(), x_shape));
    let y = build(&tp, &x);
    let ref_jac_t = tp.jacobian(&y, &x).expect("jacobian");
    let ref_jac = host(&ref_jac_t);
    let tangent = t(v.to_vec(), x_shape);

    let c1 = fandhe_ai::tape_for(device).expect("tape_for(child jvp)");
    let jv = tp.jvp(&y, &x, &tangent, &c1).expect("jvp");
    let c2 = fandhe_ai::tape_for(device).expect("tape_for(child jacfwd)");
    let jf = tp.jacfwd(&y, &x, &c2).expect("jacfwd");

    Outs {
        y_shape: y.to_tensor().shape().to_vec(),
        jv_shape: jv.shape().to_vec(),
        jv: host(&jv),
        jv_ref: reference_jv(&ref_jac, v),
        jac_shape: jf.shape().to_vec(),
        jac: host(&jf),
        ref_jac,
    }
}

/// 1 ケースについて、デバイス結果をデバイス内の `jacobian` 参照と `Device::Cpu` 結果の両方に突き合わせる。
fn assert_case_matches_cpu<F>(
    device: Device,
    label: &str,
    case: &str,
    x_data: Vec<f32>,
    x_shape: &[usize],
    v: Vec<f32>,
    build: F,
) where
    F: for<'a> Fn(&'a Tape, &Var<'a>) -> Var<'a>,
{
    let dev = compute_on(device, &x_data, x_shape, &v, &build);
    let cpu = compute_on(Device::Cpu, &x_data, x_shape, &v, &build);
    let p = fandhe_ai_backend_cpu::parity::assert_parity;

    assert_eq!(dev.jv_shape, dev.y_shape, "{label} {case}: jvp shape");
    assert_eq!(
        dev.jv_shape, cpu.jv_shape,
        "{label} {case}: jvp shape vs CPU"
    );
    let mut want_jac_shape = dev.y_shape.clone();
    want_jac_shape.extend_from_slice(x_shape);
    assert_eq!(
        dev.jac_shape, want_jac_shape,
        "{label} {case}: jacfwd shape"
    );
    assert_eq!(
        dev.jac_shape, cpu.jac_shape,
        "{label} {case}: jacfwd shape vs CPU"
    );

    p(
        &format!("{label} {case}: jvp vs デバイス jacobian·v"),
        &dev.jv,
        &dev.jv_ref,
    );
    p(
        &format!("{label} {case}: jacfwd vs デバイス jacobian"),
        &dev.jac,
        &dev.ref_jac,
    );
    p(&format!("{label} {case}: jvp vs CPU jvp"), &dev.jv, &cpu.jv);
    p(
        &format!("{label} {case}: jacfwd vs CPU jacfwd"),
        &dev.jac,
        &cpu.jac,
    );
    p(
        &format!("{label} {case}: jacobian vs CPU jacobian"),
        &dev.ref_jac,
        &cpu.ref_jac,
    );
}

/// 基本グループ（`functional_transforms_jvp_facade.rs` の supported_ops と同じフィクスチャ）。
fn run_basic_ops(device: Device, label: &str) {
    assert_case_matches_cpu(
        device,
        label,
        "mul",
        vec![1.0, 2.0, 3.0],
        &[3],
        vec![1.0, 0.5, 2.0],
        |_, x| x.mul(x).unwrap(),
    );
    let w = (0..12).map(|i| 0.1 * (i as f32) - 0.5).collect::<Vec<_>>();
    assert_case_matches_cpu(
        device,
        label,
        "matmul_tanh",
        vec![0.3, -0.7, 1.1, 0.5, 0.2, -0.4],
        &[2, 3],
        vec![1.0, -0.5, 0.25, 0.0, 2.0, -1.5],
        move |tp, x| {
            let w = tp.var_no_grad(&t(w.clone(), &[3, 4]));
            x.matmul(&w).unwrap().tanh()
        },
    );
    assert_case_matches_cpu(
        device,
        label,
        "sum",
        vec![0.5, -1.0, 2.0, 0.25],
        &[2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
        |_, x| x.mul(x).unwrap().sum(None).unwrap(),
    );
    assert_case_matches_cpu(
        device,
        label,
        "broadcast_add",
        vec![0.1, -0.2, 0.3],
        &[3],
        vec![1.0, -1.0, 0.5],
        |tp, b| {
            let a = tp.var_no_grad(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
            a.add(b).unwrap().tanh()
        },
    );
    assert_case_matches_cpu(device, label, "rank0", vec![1.5], &[], vec![2.0], |_, x| {
        x.mul(x).unwrap()
    });
}

/// 拡張グループ（#3087 の代表のうち実機カーネルがある見込みのもの。rfft・solve・avg_pool2d は
/// 実機側の対応状況が不確かで `Unsupported` になりうるため対象外）。
fn run_extended_ops(device: Device, label: &str) {
    let x6 = vec![0.5_f32, 1.1, -0.6, 0.2, -1.3, 0.8];
    let v6 = vec![0.6_f32, -0.2, 0.3, 0.9, -0.5, 0.4];
    assert_case_matches_cpu(
        device,
        label,
        "softmax",
        x6.clone(),
        &[2, 3],
        v6.clone(),
        |tp, x| {
            let w = tp.var_no_grad(&t(vec![0.3, -1.2, 0.7, 2.0, -0.5, 1.1], &[2, 3]));
            x.softmax(1).unwrap().mul(&w).unwrap()
        },
    );
    assert_case_matches_cpu(
        device,
        label,
        "layer_norm",
        x6.clone(),
        &[2, 3],
        v6.clone(),
        |tp, x| {
            let w = tp.var_no_grad(&t(vec![1.2, 0.7, -0.4], &[3]));
            let b = tp.var_no_grad(&t(vec![0.1, -0.2, 0.3], &[3]));
            x.layer_norm(Some(&w), Some(&b), 1e-5).unwrap()
        },
    );
    assert_case_matches_cpu(
        device,
        label,
        "gelu_tanh",
        x6.clone(),
        &[2, 3],
        v6.clone(),
        |_, x| x.gelu_tanh().unwrap(),
    );
    assert_case_matches_cpu(
        device,
        label,
        "cumsum",
        x6.clone(),
        &[2, 3],
        v6.clone(),
        |_, x| x.cumsum(1).unwrap(),
    );
    assert_case_matches_cpu(device, label, "pad", x6, &[2, 3], v6, |_, x| {
        x.pad(&[(1, 0), (0, 1)], 0.0).unwrap()
    });
    assert_case_matches_cpu(
        device,
        label,
        "conv2d",
        vec![0.3, -0.8, 1.2, 0.1, -0.4, 0.9, 0.5, -0.2, 0.7],
        &[1, 1, 3, 3],
        vec![0.6, -0.2, 0.3, 0.9, -0.5, 0.4, 0.1, -0.7, 0.2],
        |tp, x| {
            let w = tp.var_no_grad(&t(vec![0.5, -0.3, 0.2, 0.8], &[1, 1, 2, 2]));
            x.conv2d(&w, None, [1, 1], [0, 0], [1, 1], 1).unwrap()
        },
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/functional-transforms-jvp-2942/README.md 参照"]
fn cuda_jvp_jacfwd_basic_ops_match_cpu_reference() {
    run_basic_ops(Device::Cuda(0), "cuda");
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/functional-transforms-jvp-2942/README.md 参照"]
fn cuda_jvp_jacfwd_extended_ops_match_cpu_reference() {
    run_extended_ops(Device::Cuda(0), "cuda");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/functional-transforms-jvp-2942/README.md 参照"]
fn metal_jvp_jacfwd_basic_ops_match_cpu_reference() {
    run_basic_ops(Device::Metal, "metal");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/functional-transforms-jvp-2942/README.md 参照"]
fn metal_jvp_jacfwd_extended_ops_match_cpu_reference() {
    run_extended_ops(Device::Metal, "metal");
}
