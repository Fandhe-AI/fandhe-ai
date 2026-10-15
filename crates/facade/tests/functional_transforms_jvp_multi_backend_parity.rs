//! 本物の facade `Tape::jvp_multi`（イシュー #3064・決定記録 `docs/autodiff-functional-transforms-design.md` §30）を
//! 実機 tape（CUDA／Metal）で呼び、同じ関数の CPU 結果と比べる `#[ignore]` parity テスト。
//!
//! **ミラーとの関係**: `functional_ops_jvp_backend_parity.rs`（#2942。§28 の GB10 実測記録の対象）は置き換えず
//! そのまま維持する。単一入力の本物の `jvp`／`jacfwd` の実機テストは #3167 が別ファイルで担当し、本ファイルは
//! 複数入力版だけを担当する（ファイル名の衝突回避のため `_multi_` を含める）。
//!
//! 判定は REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）。新しいカーネルはなく、確認対象は
//! 既存カーネルの新しい呼び出し形（複数の追跡あり入力を持つ `backward_create_graph` と子テープ上の合計 1 回の
//! backward）である。形状は小さく、Metal split-K が発動する形状は使わない。
//! 実機に届かない環境では未実測のまま `docs/perf/logs/functional-transforms-jvp-multi-3064/README.md` へ申し送る。

use fandhe_ai::{Device, Tape, Tensor, Var};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

/// フィクスチャ（2 入力とも追跡あり）。0: ブロードキャスト add → tanh（a `[2,3]`・b `[3]`）、
/// 1: matmul → tanh（x `[2,3]`・W `[3,4]`）。
type Fixture = (Vec<(Vec<f32>, Vec<usize>)>, Vec<Vec<f32>>);

fn fixture(case: usize) -> Fixture {
    if case == 0 {
        (
            vec![
                (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
                (vec![0.1, -0.2, 0.3], vec![3]),
            ],
            vec![vec![0.5, -1.0, 1.0, 2.0, 0.0, -0.5], vec![1.0, -1.0, 0.5]],
        )
    } else {
        (
            vec![
                (vec![0.3, -0.7, 1.1, 0.5, 0.2, -0.4], vec![2, 3]),
                (
                    (0..12).map(|i| 0.1 * (i as f32) - 0.5).collect(),
                    vec![3, 4],
                ),
            ],
            vec![
                vec![1.0, -0.5, 0.25, 0.0, 2.0, -1.5],
                (0..12).map(|i| 0.05 * (i as f32) - 0.3).collect(),
            ],
        )
    }
}

fn build<'a>(case: usize, l: &[Var<'a>]) -> Var<'a> {
    if case == 0 {
        l[0].add(&l[1]).unwrap().tanh()
    } else {
        l[0].matmul(&l[1]).unwrap().tanh()
    }
}

/// `device` 上で `jvp_multi` と、同デバイスの `jacobian` から作った `Σ J_i·v_i`（f64 一括合計）を返す。
/// 子テープも同じデバイスで作る（`backward_create_graph` がデバイス不一致を拒否するため）。
fn compute_on(device: Device, case: usize) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let (xs, vs) = fixture(case);
    let tp = fandhe_ai::tape_for(device).expect("tape_for");
    let leaves: Vec<Var<'_>> = xs.iter().map(|(d, s)| tp.var(&t(d.clone(), s))).collect();
    let y = build(case, &leaves);
    let tangents: Vec<Tensor<f32>> = xs
        .iter()
        .zip(&vs)
        .map(|((_, s), v)| t(v.clone(), s))
        .collect();
    let in_refs: Vec<&Var<'_>> = leaves.iter().collect();
    let tg_refs: Vec<&Tensor<f32>> = tangents.iter().collect();
    let child: Tape = fandhe_ai::tape_for(device).expect("tape_for(child)");
    let out = tp
        .jvp_multi(&y, &in_refs, &tg_refs, &child)
        .expect("jvp_multi");

    let m: usize = out.shape().iter().product();
    let mut reference = vec![0.0f64; m];
    for (x, v) in leaves.iter().zip(&vs) {
        let jac = tp
            .jacobian(&y, x)
            .expect("jacobian")
            .host_slice()
            .into_owned();
        let n = v.len();
        for (r, acc) in reference.iter_mut().enumerate() {
            for k in 0..n {
                *acc += f64::from(jac[r * n + k]) * f64::from(v[k]);
            }
        }
    }
    (
        out.shape().to_vec(),
        out.host_slice().into_owned(),
        reference.into_iter().map(|v| v as f32).collect(),
    )
}

fn assert_device_matches_cpu(device: Device, label: &str) {
    for case in 0..2 {
        let (shape, got, reference) = compute_on(device, case);
        let (cpu_shape, cpu_got, _) = compute_on(Device::Cpu, case);
        assert_eq!(shape, cpu_shape, "{label} case {case}: shape");
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("{label} case {case}: デバイス結果 vs デバイス参照"),
            &got,
            &reference,
        );
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("{label} case {case}: デバイス結果 vs CPU 結果"),
            &got,
            &cpu_got,
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/functional-transforms-jvp-multi-3064/README.md 参照"]
fn metal_jvp_multi_matches_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/functional-transforms-jvp-multi-3064/README.md 参照"]
fn cuda_jvp_multi_matches_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
