//! イシュー #3139: 実機 parity 入力集合の条件数ガード（cfg なし。Linux でも実行）。
//!
//! `docs/backend-gpu-linalg-decision.md` §5 は入力集合の κ₂ ≤ 1e3 を条件とする。`linalg_parity_inputs.rs`
//! の全入力についてテスト内の f64 Gauss-Jordan 逆行列から κ₂ の厳密な上界を求めて fail-closed に検査する
//! （正方: κ₂ ≤ √(κ₁κ∞)。長方形: 短辺側 Gram G の κ₂(A) = √κ₂(G) ≤ (κ₁(G)κ∞(G))^{1/4}）。
//! 倍率は κ を変えないので、倍率集合が縮まないこと（1e±20・`1e20 × I₂`）も固定する。外部依存なし。

#[path = "linalg_parity_inputs.rs"]
mod inputs;

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::BackendOps;
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
    (norm1(a, n) * norm1(&inv, n) * norm_inf(a, n) * norm_inf(&inv, n)).sqrt()
}

/// 長方形（または正方）行列 κ₂ の上界。長方形は κ₂(A)=√κ₂(G) と Gram G の √(κ₁κ∞) 上界から (κ₁κ∞)^{1/4}。
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

/// 既知の条件数で上界計算を固定する（平方根を 2 回取る退行の検出）。diag(1,1e-4) は κ₂=1e4。
#[test]
fn kappa_upper_bound_detects_known_ill_conditioned() {
    let k = kappa2_upper_square(&[1.0, 0.0, 0.0, 1e-4], 2);
    assert!(k >= 1e4 * 0.999, "上界 {k:e}");
    // 長方形 diag(1,1e-4)（3x2）: κ₂=1e4 を下回らない
    let t = Tensor::<f32>::new(vec![1.0, 0.0, 0.0, 1e-4, 0.0, 0.0], &[3, 2]).unwrap();
    assert!(kappa2_upper(&t) >= 1e4 * 0.999);
}

#[test]
fn scale_set_is_not_silently_shrunk() {
    assert_eq!(SCALES, [1.0, 1e20, 1e-20]);
    assert_eq!(EYE2_SCALE, 1e20);
    assert_eq!(SIZES, [1, 2, 8, 17, 64]);
    assert_eq!(QR_SHAPES, [(8, 3), (3, 8), (17, 17)]);
    assert_eq!(DET_MAX_N, 17);
}

/// 厳密 0 構造の検出閾値（列ノルム比）。f64 の丸め残差（約 1e-17）と真の非零値（17×17 の最小でも約 5e-4）を
/// 分ける値で、f32 で解像できるかの境界ではない。これより大きくすると既に pass している 17×17 が落ちる。
const STRUCTURAL_ZERO_RATIO: f64 = 1e-10;

/// CPU 参照（f64）の R の上三角で `|R_ij| / ‖a_j‖₂` の最小値と位置を返す（i < min(m,n)、j ≥ i）。
fn min_upper_r_ratio(a: &Tensor<f32>) -> (f64, usize, usize) {
    let sh = a.shape().to_vec();
    let (m, n) = (sh[0], sh[1]);
    let av = s(a);
    let r = s(&CpuBackendOps::new().linalg_qr(a).expect("cpu qr").r);
    let mut best = (f64::INFINITY, 0, 0);
    for i in 0..m.min(n) {
        for j in i..n {
            let cn = (0..m)
                .map(|row| (av[row * n + j] as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let ratio = (r[i * n + j] as f64).abs() / cn;
            if ratio < best.0 {
                best = (ratio, i, j);
            }
        }
    }
    best
}

/// #2676: 偶発的な直交（R の厳密 0 要素）が 1e20 倍で f32 対 f64 の精度クラス差を相対誤差 1.0 として
/// 露出させた（旧 `qr_input(3,8)` の R[0,2]）。QR 入力に構造的 0 を再導入させない。
#[test]
fn qr_inputs_have_no_structural_zero_in_r() {
    for (m, n) in QR_SHAPES {
        let (ratio, i, j) = min_upper_r_ratio(&qr_input(m, n));
        println!("qr_input({m},{n}) min |R_ij|/|a_j| = {ratio:e} @[{i},{j}]");
        assert!(
            ratio >= STRUCTURAL_ZERO_RATIO,
            "qr_input({m},{n}): R[{i},{j}] が構造的 0（比 {ratio:e}）"
        );
    }
}

/// 旧 (3,8) 式（シフトなし）で構造的 0 を検出できること（ガード自身の退行検出）。
#[test]
fn structural_zero_guard_detects_legacy_wide_input() {
    let d: Vec<f32> = (0..24)
        .map(|i| (((i * 37 + 11) % 23) as f32 - 11.0) / 7.0)
        .collect();
    let (ratio, i, j) = min_upper_r_ratio(&t(d, &[3, 8]));
    assert!(
        ratio < STRUCTURAL_ZERO_RATIO,
        "旧入力で検出できない: {ratio:e}"
    );
    assert_eq!((i, j), (0, 2));
}
