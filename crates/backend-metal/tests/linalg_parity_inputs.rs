//! イシュー #3139: Metal 線形代数の実機 parity（`linalg_parity.rs`）と κ ガード
//! （`linalg_parity_inputs_guard.rs`）が共有する決定的入力集合の正本。
//!
//! `docs/backend-gpu-linalg-decision.md` §5 の入力集合（κ₂ ≤ 1e3・n ≤ 64・行列全体の 1e±20 倍・
//! `1e20 × I₂`）をここで固定する。`docs/perf/logs/metal-linalg-parity-3139/RULE.txt` の事前登録と
//! 1 対 1 に対応させる。実機結果を見た後に入力を変えない。cfg は付けない（Linux でも κ ガードが読む）。
//! 各テストは `#[path]` で include するため、使わない項目は `dead_code` を許容する。

#![allow(dead_code)]

use fandhe_ai_tensor_core::Tensor;

/// 行列全体にかける倍率（等倍・1e20 倍・1e-20 倍）。κ は倍率で変わらない。
pub const SCALES: [f32; 3] = [1.0, 1e20, 1e-20];
/// cholesky・LU 系で判定する行列サイズ。
pub const SIZES: [usize; 5] = [1, 2, 8, 17, 64];
/// det を判定する最大サイズ。n=64 は det が f32 範囲外（inf）になり、両側 inf で `inf - inf = NaN` となって
/// カーネルと無関係に fail するため n ≤ 17 に限る（`linalg_model_parity.rs` と同じ根拠）。
pub const DET_MAX_N: usize = 17;
/// 追加で判定する `1e20 × I₂` の倍率。
pub const EYE2_SCALE: f32 = 1e20;
/// QR の判定形状。
pub const QR_SHAPES: [(usize, usize); 3] = [(8, 3), (3, 8), (17, 17)];

pub fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("valid tensor")
}

pub fn s(x: &Tensor<f32>) -> Vec<f32> {
    x.contiguous().as_slice().expect("slice").to_vec()
}

pub fn spd(n: usize) -> Tensor<f32> {
    let mut d = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = if i == j {
                n as f32 + 1.0
            } else {
                1.0 / (1.0 + (i as f32 - j as f32).abs())
            };
        }
    }
    t(d, &[n, n])
}

pub fn diag_dominant(n: usize) -> Tensor<f32> {
    let mut d = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = if i == j {
                2.0 * n as f32
            } else {
                (((i * 7 + j * 3) % 5) as f32 - 2.0) / 2.0
            };
        }
    }
    t(d, &[n, n])
}

pub fn scaled(a: &Tensor<f32>, factor: f32) -> Tensor<f32> {
    let shape = a.shape().to_vec();
    t(s(a).iter().map(|v| v * factor).collect(), &shape)
}

/// `factor × I₂`。
pub fn eye2_scaled(factor: f32) -> Tensor<f32> {
    t(vec![factor, 0.0, 0.0, factor], &[2, 2])
}

/// QR 入力（m×n）。旧 17×17（`(((i*37+11)%23)-11)/7` のみ）は数値的に特異（κ₂ ≈ 3.5e16）で
/// §5 の κ ≤ 1e3 に反するため、17×17 のみ対角へ `2n` を加えて是正した（実機結果を見る前の決定。RULE.txt）。
/// (8,3)・(3,8) は旧式のまま（κ₂ ≈ 2.7・1.9）。
pub fn qr_input(m: usize, n: usize) -> Tensor<f32> {
    let shift = if m == n && n == 17 {
        2.0 * n as f32
    } else {
        0.0
    };
    let d: Vec<f32> = (0..m * n)
        .map(|i| {
            let base = (((i * 37 + 11) % 23) as f32 - 11.0) / 7.0;
            if shift != 0.0 && i / n == i % n {
                base + shift
            } else {
                base
            }
        })
        .collect();
    t(d, &[m, n])
}
/// #3139 の記録専用診断の共通出力（判定ではない）。`a`・`b` を `fandhe_ai_backend_cpu::compare`
/// （REQ-2 の統一複合判定。定数は不変）で突合し、集計と「判定を外れた要素」の添字・値を `println!` する。
/// 要素単位の可否は長さ 1 の `compare` を再利用して得る（判定式を書き直さない）。
/// `linalg_parity.rs`（実機）と `linalg_model_parity.rs`（Linux 可）の `#[ignore]` 診断から呼ばれる。
pub fn diag_compare(tag: &str, an: &str, a: &[f32], bn: &str, b: &[f32], cols: usize) {
    use fandhe_ai_backend_cpu::compare;
    let rep = compare(a, b).expect("same length");
    println!(
        "DIAG {tag} {an}-vs-{bn}: fail_count={}/{} max_abs_diff={:e} max_rel_err={:e}",
        rep.fail_count, rep.total, rep.max_abs_diff, rep.max_rel_err
    );
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let one = compare(&[x], &[y]).expect("len 1");
        if one.fail_count != 0 {
            println!(
                "DIAG {tag} {an}-vs-{bn} FAIL [{},{}]: {an}={x:e} {bn}={y:e}",
                i / cols.max(1),
                i % cols.max(1)
            );
        }
    }
}
