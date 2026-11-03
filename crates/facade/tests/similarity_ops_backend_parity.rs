//! `Var::cosine_similarity`／`Var::pairwise_distance`（イシュー #3821・親 #3813）のバックエンド間
//! parity テスト（`loss_ops_backend_parity.rs` と同型）。
//!
//! 2 演算とも `BackendOps` に対応メソッドを持たない host 計算（f64 アキュムレータ）の新 `Op` のため、
//! 入力を host へ実体化してから計算する。バックエンド間は REQ-2 より強い **bit 完全一致**で比べる
//! （NaN 同士はクラス一致）。tolerance・baseline は変更しない。
//!
//! - 属性なし: CPU（`fandhe_ai::tape()`）と NaiveOps（`fandhe_ai_autodiff::Tape::new()`）の bit 一致。
//!   あわせて独立した f64 参照実装（テスト内）と REQ-2 判定（`assert_parity`）でも突合する。
//! - `#[ignore]`: CUDA 実機（`cuda_*`）・Metal 実機（`metal_*`・macOS 限定）と CPU の bit 一致。
//!   実機への到達手段が実装環境にないため未実測のまま申し送る
//!   （`docs/perf/logs/similarity-ops-3821/README.md`。実測は Phase 10 の #3860・#3861）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

trait VarSource {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_>;
    fn run_backward(&self, loss: &Var<'_>) -> fandhe_ai_autodiff::Gradients;
}

impl VarSource for fandhe_ai::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
    fn run_backward(&self, loss: &Var<'_>) -> fandhe_ai_autodiff::Gradients {
        self.backward(loss).unwrap()
    }
}

impl VarSource for fandhe_ai_autodiff::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
    fn run_backward(&self, loss: &Var<'_>) -> fandhe_ai_autodiff::Gradients {
        self.backward(loss).unwrap()
    }
}

/// bit 完全一致（NaN 同士はクラス一致）。
fn assert_bits_eq(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: 要素数が一致しない");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if a.is_nan() || e.is_nan() {
            assert!(
                a.is_nan() && e.is_nan(),
                "{label}: 要素 {i} が NaN クラス不一致"
            );
        } else {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}: 要素 {i} が bit 一致しない（actual={a:?}, expected={e:?}）"
            );
        }
    }
}

const SHAPE: [usize; 3] = [2, 3, 4];
const DIM: usize = 1;

fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    Tensor::new(Xorshift64Star::new(seed).fill_vec(numel), shape).expect("leaf: shape 一致")
}

fn vec_of(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().host_slice().to_vec()
}

/// `(forward, dx1, dx2)`。損失は出力の重み付き和（勾配を非自明にする）。
fn run_cosine(tape: &impl VarSource) -> [Vec<f32>; 3] {
    let (x1, x2) = (
        tape.make_var(&leaf(21, &SHAPE)),
        tape.make_var(&leaf(22, &SHAPE)),
    );
    let y = x1
        .cosine_similarity(&x2, DIM, 1e-8)
        .expect("cosine: 形状適合");
    let out_shape = y.to_tensor().shape().to_vec();
    let out = vec_of(&y.to_tensor());
    let c = tape.make_var(&leaf(23, &out_shape));
    let loss = y.mul(&c).unwrap().sum(None).unwrap();
    let g = tape.run_backward(&loss);
    let grad = |v: &Var<'_>| vec_of(g.get(v).unwrap().expect("勾配が到達する"));
    [out, grad(&x1), grad(&x2)]
}

fn run_pairwise(tape: &impl VarSource, p: f32, keepdim: bool) -> [Vec<f32>; 3] {
    let (x1, x2) = (
        tape.make_var(&leaf(31, &SHAPE)),
        tape.make_var(&leaf(32, &SHAPE)),
    );
    let y = x1
        .pairwise_distance(&x2, p, 1e-6, keepdim)
        .expect("pairwise: 形状適合");
    let out_shape = y.to_tensor().shape().to_vec();
    let out = vec_of(&y.to_tensor());
    let c = tape.make_var(&leaf(33, &out_shape));
    let loss = y.mul(&c).unwrap().sum(None).unwrap();
    let g = tape.run_backward(&loss);
    let grad = |v: &Var<'_>| vec_of(g.get(v).unwrap().expect("勾配が到達する"));
    [out, grad(&x1), grad(&x2)]
}

const LABELS: [&str; 3] = ["forward", "dx1", "dx2"];

fn assert_all_bits(ctx: &str, a: &[Vec<f32>; 3], b: &[Vec<f32>; 3]) {
    for k in 0..3 {
        assert_bits_eq(&format!("{ctx} {}", LABELS[k]), &a[k], &b[k]);
    }
}

/// f64 の独立参照（cosine forward。`DIM` 軸を縮約）。
fn cosine_forward_ref() -> Vec<f32> {
    let (a, b) = (vec_of(&leaf(21, &SHAPE)), vec_of(&leaf(22, &SHAPE)));
    let (outer, len, inner) = (SHAPE[0], SHAPE[1], SHAPE[2]);
    let mut out = Vec::new();
    for o in 0..outer {
        for i in 0..inner {
            let at = |k: usize| o * len * inner + k * inner + i;
            let n1 = (0..len)
                .map(|k| (a[at(k)] as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let n2 = (0..len)
                .map(|k| (b[at(k)] as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let (c1, c2) = (n1.max(1e-8), n2.max(1e-8));
            let s: f64 = (0..len)
                .map(|k| (a[at(k)] as f64 / c1) * (b[at(k)] as f64 / c2))
                .sum();
            out.push(s as f32);
        }
    }
    out
}

fn pairwise_forward_ref(p: f64) -> Vec<f32> {
    let (a, b) = (vec_of(&leaf(31, &SHAPE)), vec_of(&leaf(32, &SHAPE)));
    let d = SHAPE[2];
    a.chunks(d)
        .zip(b.chunks(d))
        .map(|(u, v)| {
            u.iter()
                .zip(v)
                .map(|(x, y)| ((*x as f64) - (*y as f64) + 1e-6).abs().powf(p))
                .sum::<f64>()
                .powf(1.0 / p) as f32
        })
        .collect()
}

#[test]
fn cpu_bit_exact_forward_matches_naive_reference() {
    let cpu = fandhe_ai::tape();
    let naive = fandhe_ai_autodiff::Tape::new();
    let (a, b) = (run_cosine(&cpu), run_cosine(&naive));
    assert_bits_eq("cosine forward: cpu vs naive", &a[0], &b[0]);
    assert_parity(
        "cosine forward: cpu vs f64 ref",
        &a[0],
        &cosine_forward_ref(),
    );
    for p in [1.0f32, 2.0, 3.0] {
        for keepdim in [false, true] {
            let a = run_pairwise(&cpu, p, keepdim);
            let b = run_pairwise(&naive, p, keepdim);
            assert_bits_eq(
                &format!("pairwise p={p} kd={keepdim} forward"),
                &a[0],
                &b[0],
            );
            assert_parity(
                &format!("pairwise p={p} forward: cpu vs f64 ref"),
                &a[0],
                &pairwise_forward_ref(p as f64),
            );
        }
    }
}

#[test]
fn cpu_bit_exact_backward_matches_naive_reference() {
    let cpu = fandhe_ai::tape();
    let naive = fandhe_ai_autodiff::Tape::new();
    assert_all_bits(
        "cosine: cpu vs naive",
        &run_cosine(&cpu),
        &run_cosine(&naive),
    );
    for p in [1.0f32, 2.0, 3.0] {
        for keepdim in [false, true] {
            assert_all_bits(
                &format!("pairwise p={p} kd={keepdim}: cpu vs naive"),
                &run_pairwise(&cpu, p, keepdim),
                &run_pairwise(&naive, p, keepdim),
            );
        }
    }
}

fn cuda_tape() -> fandhe_ai::Tape {
    fandhe_ai::tape_for(Device::Cuda(0)).expect("実機が利用可能な前提のテストのため成功するはず")
}

fn compare_all(dev: &fandhe_ai::Tape, name: &str, forward_only: bool) {
    let cpu = fandhe_ai::tape();
    let (a, b) = (run_cosine(&cpu), run_cosine(dev));
    let (a_p, b_p) = (
        run_pairwise(&cpu, 2.0, false),
        run_pairwise(dev, 2.0, false),
    );
    let (a_p3, b_p3) = (run_pairwise(&cpu, 3.0, true), run_pairwise(dev, 3.0, true));
    let range = if forward_only { 0..1 } else { 1..3 };
    for k in range {
        assert_bits_eq(
            &format!("cosine {}: cpu vs {name}", LABELS[k]),
            &b[k],
            &a[k],
        );
        assert_bits_eq(
            &format!("pairwise {}: cpu vs {name}", LABELS[k]),
            &b_p[k],
            &a_p[k],
        );
        assert_bits_eq(
            &format!("pairwise3 {}: cpu vs {name}", LABELS[k]),
            &b_p3[k],
            &a_p3[k],
        );
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/similarity-ops-3821/README.md 参照"]
fn cuda_bit_exact_forward_matches_cpu_reference() {
    compare_all(&cuda_tape(), "cuda", true);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/similarity-ops-3821/README.md 参照"]
fn cuda_bit_exact_backward_matches_cpu_reference() {
    compare_all(&cuda_tape(), "cuda", false);
}

#[cfg(target_os = "macos")]
fn metal_tape() -> fandhe_ai::Tape {
    fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず")
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/similarity-ops-3821/README.md 参照"]
fn metal_bit_exact_forward_matches_cpu_reference() {
    compare_all(&metal_tape(), "metal", true);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/similarity-ops-3821/README.md 参照"]
fn metal_bit_exact_backward_matches_cpu_reference() {
    compare_all(&metal_tape(), "metal", false);
}
