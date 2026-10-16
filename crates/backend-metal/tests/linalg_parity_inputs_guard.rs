//! イシュー #3139: 実機 parity 入力集合の条件数ガード（cfg なし。Linux でも実行）。
//!
//! `docs/backend-gpu-linalg-decision.md` §5 は入力集合の κ₂ ≤ 1e3 を条件とする。`linalg_parity_inputs.rs`
//! の全入力についてテスト内の f64 Gauss-Jordan 逆行列から κ₂ の厳密な上界を求めて fail-closed に検査する
//! （正方: κ₂ ≤ √(κ₁κ∞)。長方形: 短辺側 Gram G の κ₂(A) = √κ₂(G) ≤ (κ₁(G)κ∞(G))^{1/4}）。
//! 倍率は κ を変えないので、倍率集合が縮まないこと（1e±20・`1e20 × I₂`）も固定する。外部依存なし。

#[path = "linalg_parity_inputs.rs"]
mod inputs;

use fandhe_ai_tensor_core::Tensor;
use inputs::*;

const KAPPA_MAX: f64 = 1e3;

fn to_f64(a: &Tensor<f32>) -> (Vec<f64>, usize, usize) {
    let sh = a.shape();
    (s(a).iter().map(|&v| v as f64).collect(), sh[0], sh[1])
}

/// 部分ピボット付き Gauss-Jordan 逆行列。ピボットが 0 なら None。
fn invert(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for c in 0..n {
        let p = (c..n).max_by(|&x, &y| m[x * n + c].abs().total_cmp(&m[y * n + c].abs()))?;
        if m[p * n + c] == 0.0 {
            return None;
        }
        for j in 0..n {
            m.swap(c * n + j, p * n + j);
            inv.swap(c * n + j, p * n + j);
        }
        let d = m[c * n + c];
        for j in 0..n {
            m[c * n + j] /= d;
            inv[c * n + j] /= d;
        }
        for r in 0..n {
            if r != c {
                let f = m[r * n + c];
                if f != 0.0 {
                    for j in 0..n {
                        m[r * n + j] -= f * m[c * n + j];
                        inv[r * n + j] -= f * inv[c * n + j];
                    }
                }
            }
        }
    }
    Some(inv)
}

fn norm1(a: &[f64], n: usize) -> f64 {
    (0..n)
        .map(|j| (0..n).map(|i| a[i * n + j].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn norm_inf(a: &[f64], n: usize) -> f64 {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// 正方行列 κ₂ の上界 √(κ₁κ∞)。
fn kappa2_upper_square(a: &[f64], n: usize) -> f64 {
    let inv = invert(a, n).expect("特異な入力は入力集合に含めない");
    (norm1(a, n) * norm1(&inv, n) * norm_inf(a, n) * norm_inf(&inv, n))
        .sqrt()
        .sqrt()
}

/// 長方形（または正方）行列 κ₂ の上界。長方形は短辺側 Gram の (κ₁κ∞)^{1/4}。
fn kappa2_upper(t: &Tensor<f32>) -> f64 {
    let (a, m, n) = to_f64(t);
    if m == n {
        return kappa2_upper_square(&a, n);
    }
    let k = m.min(n);
    let mut g = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            g[i * k + j] = if m > n {
                (0..m).map(|r| a[r * n + i] * a[r * n + j]).sum()
            } else {
                (0..n).map(|c| a[i * n + c] * a[j * n + c]).sum()
            };
        }
    }
    kappa2_upper_square(&g, k).sqrt()
}

fn check(name: &str, k: f64) {
    assert!(
        k.is_finite() && k <= KAPPA_MAX,
        "{name}: κ₂ 上界 {k:e} > {KAPPA_MAX:e}"
    );
}

#[test]
fn input_set_condition_numbers_are_bounded() {
    for n in SIZES {
        check(&format!("spd({n})"), kappa2_upper(&spd(n)));
        check(
            &format!("diag_dominant({n})"),
            kappa2_upper(&diag_dominant(n)),
        );
    }
    for (m, n) in QR_SHAPES {
        check(&format!("qr_input({m},{n})"), kappa2_upper(&qr_input(m, n)));
    }
    check("1e20*I2 (正規化後)", kappa2_upper(&eye2_scaled(1.0)));
}

#[test]
fn scale_set_is_not_silently_shrunk() {
    assert_eq!(SCALES, [1.0, 1e20, 1e-20]);
    assert_eq!(EYE2_SCALE, 1e20);
    assert_eq!(SIZES, [1, 2, 8, 17, 64]);
    assert_eq!(QR_SHAPES, [(8, 3), (3, 8), (17, 17)]);
    assert_eq!(DET_MAX_N, 17);
}
