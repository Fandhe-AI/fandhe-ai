//! `Var::bilinear`（イシュー #3818・親 #3813）のバックエンド間 parity テスト
//! （`batched_matmul_backend_parity.rs` と同型）。
//!
//! bilinear は `matmul`（FMA 契約・CUDA TF32 opt-in）と `sum`（f64 チャンク 2 段）を通る合成のため、
//! bit 一致は主張せず REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
//! `fandhe_ai_backend_cpu::parity::assert_parity`）で比較する。tolerance・baseline は変更しない。
//!
//! - 属性なし: CPU（`fandhe_ai::tape()`）と f64 ホスト参照（三重ループ）、および NaiveOps
//!   （`fandhe_ai_autodiff::Tape::new()`）との比較。
//! - `#[ignore]`: CUDA 実機（`cuda_*`）・Metal 実機（`metal_*`・macOS 限定）と CPU の比較。
//!   実機への到達手段が実装環境にないため未実測のまま申し送る
//!   （`docs/perf/logs/bilinear-ops-3818/README.md`。実測は Phase 10 の #3860・#3861）。

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

// 先頭の次元 [2, 3]（N = 6）、in1 = 5、in2 = 4、out = 3。
const LEAD: [usize; 2] = [2, 3];
const N: usize = 6;
const IN1: usize = 5;
const IN2: usize = 4;
const OUT: usize = 3;

fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    Tensor::new(Xorshift64Star::new(seed).fill_vec(numel), shape).expect("leaf: shape 一致")
}

fn vec_of(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().host_slice().to_vec()
}

struct Inputs {
    x1: Tensor<f32>,
    x2: Tensor<f32>,
    w: Tensor<f32>,
    b: Tensor<f32>,
}

fn inputs() -> Inputs {
    let shape1 = [LEAD[0], LEAD[1], IN1];
    let shape2 = [LEAD[0], LEAD[1], IN2];
    Inputs {
        x1: leaf(11, &shape1),
        x2: leaf(12, &shape2),
        w: leaf(13, &[OUT, IN1, IN2]),
        b: leaf(14, &[OUT]),
    }
}

/// `(forward 出力, dx1, dx2, dw, db)`。損失は出力の重み付き和（勾配を非自明にする）。
fn run(tape: &impl VarSource) -> [Vec<f32>; 5] {
    let i = inputs();
    let (x1, x2) = (tape.make_var(&i.x1), tape.make_var(&i.x2));
    let (w, b) = (tape.make_var(&i.w), tape.make_var(&i.b));
    let y = x1.bilinear(&x2, &w, Some(&b)).expect("bilinear: 形状適合");
    assert_eq!(y.to_tensor().shape(), [LEAD[0], LEAD[1], OUT]);
    let out = vec_of(&y.to_tensor());
    let c = tape.make_var(&leaf(15, &[LEAD[0], LEAD[1], OUT]));
    let loss = y.mul(&c).unwrap().sum(None).unwrap();
    let g = tape.run_backward(&loss);
    let grad = |v: &Var<'_>| vec_of(g.get(v).unwrap().expect("勾配が到達する"));
    [out, grad(&x1), grad(&x2), grad(&w), grad(&b)]
}

const LABELS: [&str; 5] = ["forward", "dx1", "dx2", "dw", "db"];

/// f64 のホスト参照（三重ループ）の forward。
fn host_forward(i: &Inputs) -> Vec<f32> {
    let (x1, x2, w, b) = (vec_of(&i.x1), vec_of(&i.x2), vec_of(&i.w), vec_of(&i.b));
    let mut y = vec![0.0f32; N * OUT];
    for r in 0..N {
        for o in 0..OUT {
            let mut acc = b[o] as f64;
            for p in 0..IN1 {
                for q in 0..IN2 {
                    acc += x1[r * IN1 + p] as f64
                        * w[(o * IN1 + p) * IN2 + q] as f64
                        * x2[r * IN2 + q] as f64;
                }
            }
            y[r * OUT + o] = acc as f32;
        }
    }
    y
}

/// f64 のホスト参照の勾配（損失 = Σ y ⊙ c）。
fn host_backward(i: &Inputs) -> [Vec<f32>; 4] {
    let (x1, x2, w) = (vec_of(&i.x1), vec_of(&i.x2), vec_of(&i.w));
    let c = vec_of(&leaf(15, &[LEAD[0], LEAD[1], OUT]));
    let (mut dx1, mut dx2) = (vec![0.0f64; N * IN1], vec![0.0f64; N * IN2]);
    let (mut dw, mut db) = (vec![0.0f64; OUT * IN1 * IN2], vec![0.0f64; OUT]);
    for r in 0..N {
        for o in 0..OUT {
            let g = c[r * OUT + o] as f64;
            db[o] += g;
            for p in 0..IN1 {
                for q in 0..IN2 {
                    let wv = w[(o * IN1 + p) * IN2 + q] as f64;
                    dx1[r * IN1 + p] += g * wv * x2[r * IN2 + q] as f64;
                    dx2[r * IN2 + q] += g * wv * x1[r * IN1 + p] as f64;
                    dw[(o * IN1 + p) * IN2 + q] +=
                        g * x1[r * IN1 + p] as f64 * x2[r * IN2 + q] as f64;
                }
            }
        }
    }
    let f = |v: Vec<f64>| v.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
    [f(dx1), f(dx2), f(dw), f(db)]
}

#[test]
fn cpu_bilinear_forward_matches_host_f64_reference() {
    let got = run(&fandhe_ai::tape());
    assert_parity(
        "bilinear forward: cpu vs host f64",
        &got[0],
        &host_forward(&inputs()),
    );
}

#[test]
fn cpu_bilinear_backward_matches_host_f64_reference() {
    let got = run(&fandhe_ai::tape());
    let want = host_backward(&inputs());
    for k in 0..4 {
        assert_parity(
            &format!("bilinear {}: cpu vs host f64", LABELS[k + 1]),
            &got[k + 1],
            &want[k],
        );
    }
}

#[test]
fn cpu_bilinear_forward_matches_naive_ops() {
    let a = run(&fandhe_ai::tape());
    let b = run(&fandhe_ai_autodiff::Tape::new());
    assert_parity("bilinear forward: cpu vs naive", &a[0], &b[0]);
}

#[test]
fn cpu_bilinear_backward_matches_naive_ops() {
    let a = run(&fandhe_ai::tape());
    let b = run(&fandhe_ai_autodiff::Tape::new());
    for k in 1..5 {
        assert_parity(
            &format!("bilinear {}: cpu vs naive", LABELS[k]),
            &a[k],
            &b[k],
        );
    }
}

fn cuda_tape() -> fandhe_ai::Tape {
    fandhe_ai::tape_for(Device::Cuda(0)).expect("実機が利用可能な前提のテストのため成功するはず")
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/bilinear-ops-3818/README.md 参照"]
fn cuda_bilinear_forward_matches_cpu_reference() {
    let a = run(&fandhe_ai::tape());
    let b = run(&cuda_tape());
    assert_parity("bilinear forward: cpu vs cuda", &b[0], &a[0]);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/bilinear-ops-3818/README.md 参照"]
fn cuda_bilinear_backward_matches_cpu_reference() {
    let a = run(&fandhe_ai::tape());
    let b = run(&cuda_tape());
    for k in 1..5 {
        assert_parity(
            &format!("bilinear {}: cpu vs cuda", LABELS[k]),
            &b[k],
            &a[k],
        );
    }
}

#[cfg(target_os = "macos")]
fn metal_tape() -> fandhe_ai::Tape {
    fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず")
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/bilinear-ops-3818/README.md 参照"]
fn metal_bilinear_forward_matches_cpu_reference() {
    let a = run(&fandhe_ai::tape());
    let b = run(&metal_tape());
    assert_parity("bilinear forward: cpu vs metal", &b[0], &a[0]);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/bilinear-ops-3818/README.md 参照"]
fn metal_bilinear_backward_matches_cpu_reference() {
    let a = run(&fandhe_ai::tape());
    let b = run(&metal_tape());
    for k in 1..5 {
        assert_parity(
            &format!("bilinear {}: cpu vs metal", LABELS[k]),
            &b[k],
            &a[k],
        );
    }
}
