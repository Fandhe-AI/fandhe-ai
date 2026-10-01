//! ホスト側の参照モデル（fragment のレイアウトと期待値）。GPU を使わない純関数で、
//! `sm121_isa_probe_registry`（CI）が単体テストする。
//!
//! 方針（RULE.txt R-MMA）: 入力は f16／bf16／tf32 で正確に表せる小さな整数にし、
//! f32 累積でも結果が正確になるようにして**ビット一致**で判定する（tolerance の
//! 議論を持ち込まない。`fandhe_ai_backend_cpu::assert_parity` の複合判定は
//! 複数命令の累積誤差を前提にした閾値であり、ここへ複製しない）。
//!
//! fragment のレイアウトは PTX ISA の warp-level MMA 章（9.7.15。小節番号は
//! 要確認）に基づく。lane `l` は `g = l / 4`（行グループ）・`t = l % 4`（列
//! グループ）で表す。開発機（sm_86）で実行できる形状は RTX 3060 の実測で
//! 検証し、できない形状は `layout=unverified` として RULE.txt に登録する。

use half::{bf16, f16};

pub const LANES: usize = 32;

/// 行列 A（16 行）の小整数要素。範囲は -3..=3。
pub fn a_val(r: usize, c: usize) -> i32 {
    ((3 * r + 5 * c) % 7) as i32 - 3
}

/// 行列 B の小整数要素。範囲は -2..=2。
pub fn b_val(k: usize, n: usize) -> i32 {
    ((2 * k + 7 * n) % 5) as i32 - 2
}

/// 行列 C の小整数要素。範囲は -1..=2。
pub fn c_val(r: usize, n: usize) -> i32 {
    ((r * 8 + n) % 4) as i32 - 1
}

/// `D[r][n] = C[r][n] + sum_k A[r][k] * B[k][n]`（整数。小さい値のみなので厳密）。
pub fn dense_d(r: usize, n: usize, k_dim: usize) -> i32 {
    c_val(r, n) + (0..k_dim).map(|k| a_val(r, k) * b_val(k, n)).sum::<i32>()
}

fn half_bits(v: i32, bf: bool) -> u16 {
    if bf {
        bf16::from_f32(v as f32).to_bits()
    } else {
        f16::from_f32(v as f32).to_bits()
    }
}

fn pack2(lo: u16, hi: u16) -> u32 {
    u32::from(lo) | (u32::from(hi) << 16)
}

fn f32_bits(v: i32) -> u32 {
    (v as f32).to_bits()
}

// ---------------------------------------------------------------- f16/bf16 系

/// f16／bf16 系（`m16n8k16`・`m16n8k8`）の A fragment の第 `i` レジスタの要素座標。
/// 1 レジスタに 2 要素（下位 = 小さい列）。
pub fn f16_a_coords(l: usize, i: usize) -> [(usize, usize); 2] {
    let (g, t) = (l / 4, l % 4);
    let r = g + 8 * (i % 2);
    let c = 2 * t + 8 * (i / 2);
    [(r, c), (r, c + 1)]
}

/// B fragment の第 `j` レジスタの要素座標 `(k, n)`。
pub fn f16_b_coords(l: usize, j: usize) -> [(usize, usize); 2] {
    let (g, t) = (l / 4, l % 4);
    let k = 2 * t + 8 * j;
    [(k, g), (k + 1, g)]
}

/// f32 累積の C／D fragment の第 `i` 要素の座標 `(r, n)`。
pub fn acc32_coords(l: usize, i: usize) -> (usize, usize) {
    let (g, t) = (l / 4, l % 4);
    (g + 8 * (i / 2), 2 * t + i % 2)
}

/// f16 累積の C／D fragment の第 `i` レジスタ（2 要素）の座標。
pub fn acc16_coords(l: usize, i: usize) -> [(usize, usize); 2] {
    let (g, t) = (l / 4, l % 4);
    [(g + 8 * i, 2 * t), (g + 8 * i, 2 * t + 1)]
}

/// f16／bf16 系 mma の lane ごとの入力語（A, B, C の順に連結。lane 0 から並べる）。
pub fn f16_family_input(k: usize, bf: bool, acc_f16: bool) -> Vec<u32> {
    let mut words = Vec::new();
    for l in 0..LANES {
        for i in 0..k / 4 {
            let [(r0, c0), (r1, c1)] = f16_a_coords(l, i);
            words.push(pack2(
                half_bits(a_val(r0, c0), bf),
                half_bits(a_val(r1, c1), bf),
            ));
        }
        for j in 0..k / 8 {
            let [(k0, n0), (k1, n1)] = f16_b_coords(l, j);
            words.push(pack2(
                half_bits(b_val(k0, n0), bf),
                half_bits(b_val(k1, n1), bf),
            ));
        }
        if acc_f16 {
            for i in 0..2 {
                let [(r0, n0), (r1, n1)] = acc16_coords(l, i);
                words.push(pack2(
                    half_bits(c_val(r0, n0), bf),
                    half_bits(c_val(r1, n1), bf),
                ));
            }
        } else {
            for i in 0..4 {
                let (r, n) = acc32_coords(l, i);
                words.push(f32_bits(c_val(r, n)));
            }
        }
    }
    words
}

/// f16／bf16 系 mma の lane ごとの期待出力語。
pub fn f16_family_expected(k: usize, acc_f16: bool) -> Vec<u32> {
    let mut words = Vec::new();
    for l in 0..LANES {
        if acc_f16 {
            for i in 0..2 {
                let [(r0, n0), (r1, n1)] = acc16_coords(l, i);
                words.push(pack2(
                    half_bits(dense_d(r0, n0, k), false),
                    half_bits(dense_d(r1, n1, k), false),
                ));
            }
        } else {
            for i in 0..4 {
                let (r, n) = acc32_coords(l, i);
                words.push(f32_bits(dense_d(r, n, k)));
            }
        }
    }
    words
}

// ---------------------------------------------------------------- tf32 系

/// tf32 系（`m16n8k8`・`m16n8k4`）の A fragment の第 `i` レジスタの座標。
pub fn tf32_a_coords(l: usize, i: usize) -> (usize, usize) {
    let (g, t) = (l / 4, l % 4);
    (g + 8 * (i % 2), t + 4 * (i / 2))
}

/// tf32 系の B fragment の第 `j` レジスタの座標 `(k, n)`。
pub fn tf32_b_coords(l: usize, j: usize) -> (usize, usize) {
    let (g, t) = (l / 4, l % 4);
    (t + 4 * j, g)
}

pub fn tf32_input(k: usize) -> Vec<u32> {
    let mut words = Vec::new();
    for l in 0..LANES {
        for i in 0..k / 2 {
            let (r, c) = tf32_a_coords(l, i);
            words.push(f32_bits(a_val(r, c)));
        }
        for j in 0..k / 4 {
            let (kk, n) = tf32_b_coords(l, j);
            words.push(f32_bits(b_val(kk, n)));
        }
        for i in 0..4 {
            let (r, n) = acc32_coords(l, i);
            words.push(f32_bits(c_val(r, n)));
        }
    }
    words
}

pub fn tf32_expected(k: usize) -> Vec<u32> {
    let mut words = Vec::new();
    for l in 0..LANES {
        for i in 0..4 {
            let (r, n) = acc32_coords(l, i);
            words.push(f32_bits(dense_d(r, n, k)));
        }
    }
    words
}

// ---------------------------------------------------------------- f64 m8n8k4

fn push_f64(words: &mut Vec<u32>, v: f64) {
    let b = v.to_bits();
    words.push(b as u32);
    words.push((b >> 32) as u32);
}

/// f64 `m8n8k4`: A は `(g, t)`、B は `(k = t, n = g)`、C／D は `(g, 2t + i)`（i = 0, 1）。
pub fn f64_m8n8k4_input() -> Vec<u32> {
    let mut words = Vec::new();
    for l in 0..LANES {
        let (g, t) = (l / 4, l % 4);
        push_f64(&mut words, f64::from(a_val(g, t)));
        push_f64(&mut words, f64::from(b_val(t, g)));
        for i in 0..2 {
            push_f64(&mut words, f64::from(c_val(g, 2 * t + i)));
        }
    }
    words
}

pub fn f64_m8n8k4_expected() -> Vec<u32> {
    let mut words = Vec::new();
    for l in 0..LANES {
        let (g, t) = (l / 4, l % 4);
        for i in 0..2 {
            push_f64(&mut words, f64::from(dense_d(g, 2 * t + i, 4)));
        }
    }
    words
}

// ---------------------------------------------------------------- ldmatrix / stmatrix

/// ldmatrix／stmatrix の smem 画像（128 語 = 8x8 b16 行列 4 枚）。半語 `p`
/// （0..256）の値は `p + 1`（全要素が異なり、0 を含まない）。
pub fn ldmatrix_smem_input() -> Vec<u32> {
    (0..128u32)
        .map(|w| pack2((2 * w + 1) as u16, (2 * w + 2) as u16))
        .collect()
}

fn smem_hw(m: usize, r: usize, c: usize) -> u16 {
    (m * 64 + r * 8 + c + 1) as u16
}

/// `ldmatrix` の lane ごとの期待出力語（4 語/lane。使わないレジスタは 0）。
/// 非 trans は `M[m][g][2t..2t+2]`、trans は `M[m][2t..2t+2][g]`。
pub fn ldmatrix_expected(count: usize, trans: bool) -> Vec<u32> {
    let mut words = Vec::new();
    for l in 0..LANES {
        let (g, t) = (l / 4, l % 4);
        for m in 0..4 {
            if m >= count {
                words.push(0);
            } else if trans {
                words.push(pack2(smem_hw(m, 2 * t, g), smem_hw(m, 2 * t + 1, g)));
            } else {
                words.push(pack2(smem_hw(m, g, 2 * t), smem_hw(m, g, 2 * t + 1)));
            }
        }
    }
    words
}

/// `stmatrix x4` の lane ごとの入力語（4 語/lane）。
pub fn stmatrix_input() -> Vec<u32> {
    let mut words = Vec::new();
    for l in 0..LANES {
        for m in 0..4 {
            let id = (l * 4 + m) as u32;
            words.push((id << 16) | (0x8000 + id));
        }
    }
    words
}

/// `stmatrix x4` の期待 smem 画像（128 語）。行列 `m` の行 `r`・列対 `cp` の語は
/// lane `4r + cp` のレジスタ `m`。
pub fn stmatrix_expected(input: &[u32]) -> Vec<u32> {
    let mut words = vec![0u32; 128];
    for m in 0..4 {
        for r in 0..8 {
            for cp in 0..4 {
                words[m * 32 + r * 4 + cp] = input[(4 * r + cp) * 4 + m];
            }
        }
    }
    words
}

// ---------------------------------------------------------------- SIMT

fn lcg(seed: u32) -> u32 {
    seed.wrapping_mul(0x9E37_79B1).wrapping_add(0x7F4A_7C15)
}

/// `fma.rn.f32` の入力（lane ごとに a, b, c。丸めが生じる非整数値）。
pub fn fma_f32_input() -> Vec<u32> {
    let mut words = Vec::new();
    for l in 0..LANES {
        let lf = l as f32;
        words.push((1.0f32 + lf * 0.1).to_bits());
        words.push((3.3f32 - lf * 0.07).to_bits());
        words.push((-2.2f32 + lf * 0.05).to_bits());
    }
    words
}

pub fn fma_f32_expected(input: &[u32]) -> Vec<u32> {
    input
        .chunks(3)
        .map(|c| {
            f32::from_bits(c[0])
                .mul_add(f32::from_bits(c[1]), f32::from_bits(c[2]))
                .to_bits()
        })
        .collect()
}

pub fn fma_f64_input() -> Vec<u32> {
    let mut words = Vec::new();
    for l in 0..LANES {
        let lf = l as f64;
        push_f64(&mut words, 1.0 + lf * 0.1);
        push_f64(&mut words, 3.3 - lf * 0.07);
        push_f64(&mut words, -2.2 + lf * 0.05);
    }
    words
}

pub fn fma_f64_expected(input: &[u32]) -> Vec<u32> {
    let dbl = |lo: u32, hi: u32| f64::from_bits(u64::from(lo) | (u64::from(hi) << 32));
    let mut words = Vec::new();
    for c in input.chunks(6) {
        let d = dbl(c[0], c[1]).mul_add(dbl(c[2], c[3]), dbl(c[4], c[5]));
        push_f64(&mut words, d);
    }
    words
}

/// `fma.rn.f16x2`／`fma.rn.bf16x2` の入力（小整数。半精度で厳密）。
pub fn fma_half2_input(bf: bool) -> Vec<u32> {
    let mut words = Vec::new();
    for l in 0..LANES {
        let a = [(l % 7) as i32 + 1, ((l + 3) % 5) as i32 + 1];
        let b = [(l % 3) as i32 + 1, (l % 4) as i32 + 1];
        let c = [(l % 5) as i32 - 2, (l % 6) as i32 - 3];
        for v in [a, b, c] {
            words.push(pack2(half_bits(v[0], bf), half_bits(v[1], bf)));
        }
    }
    words
}

pub fn fma_half2_expected(bf: bool) -> Vec<u32> {
    let mut words = Vec::new();
    for l in 0..LANES {
        let a = [(l % 7) as i32 + 1, ((l + 3) % 5) as i32 + 1];
        let b = [(l % 3) as i32 + 1, (l % 4) as i32 + 1];
        let c = [(l % 5) as i32 - 2, (l % 6) as i32 - 3];
        let d = [a[0] * b[0] + c[0], a[1] * b[1] + c[1]];
        words.push(pack2(half_bits(d[0], bf), half_bits(d[1], bf)));
    }
    words
}

/// packed `f32x2` の入力。1 lane に a(2 語)・b(2 語)・（fma は c(2 語)）を並べる。
/// 下位語を第 1 要素とみなす（この仮定は `layout=unverified`）。
pub fn f32x2_input(with_c: bool) -> Vec<u32> {
    let mut words = Vec::new();
    for l in 0..LANES {
        let lf = l as f32;
        for base in [0.5f32, 2.25, -1.5].iter().take(if with_c { 3 } else { 2 }) {
            words.push((base + lf * 0.37).to_bits());
            words.push((base * 1.5 - lf * 0.11).to_bits());
        }
    }
    words
}

#[derive(Clone, Copy)]
pub enum F32x2Op {
    Add,
    Mul,
    Fma,
}

pub fn f32x2_expected(input: &[u32], op: F32x2Op) -> Vec<u32> {
    let stride = if matches!(op, F32x2Op::Fma) { 6 } else { 4 };
    let f = f32::from_bits;
    let mut words = Vec::new();
    for c in input.chunks(stride) {
        for e in 0..2 {
            let a = f(c[e]);
            let b = f(c[2 + e]);
            let d = match op {
                F32x2Op::Add => a + b,
                F32x2Op::Mul => a * b,
                F32x2Op::Fma => a.mul_add(b, f(c[4 + e])),
            };
            words.push(d.to_bits());
        }
    }
    words
}

/// `cvt.rna.tf32.f32` の入力（正の正規化数。下位 13 bit が非零）。
pub fn cvt_tf32_input() -> Vec<u32> {
    (0..LANES as u32)
        .map(|l| 0x3F80_0000 + l * 0x0001_2345 + 0x0FFF)
        .collect()
}

/// `cvt.rna`（最近接・tie は 0 から遠い側）の tf32 化。正の有限値に限り
/// `+0x1000` して下位 13 bit を落とす操作と一致する。
pub fn cvt_tf32_expected(input: &[u32]) -> Vec<u32> {
    input
        .iter()
        .map(|&b| b.wrapping_add(0x1000) & 0xFFFF_E000)
        .collect()
}

pub fn redux_u32_input() -> Vec<u32> {
    (0..LANES as u32).map(|l| lcg(l + 1) >> 3).collect()
}

/// lane ごとに `[add, min, max, or]`（全 lane 同値）。
pub fn redux_u32_expected(input: &[u32]) -> Vec<u32> {
    let sum = input.iter().fold(0u32, |s, &v| s.wrapping_add(v));
    let min = input.iter().copied().min().unwrap_or(0);
    let max = input.iter().copied().max().unwrap_or(0);
    let or = input.iter().fold(0u32, |s, &v| s | v);
    (0..LANES).flat_map(|_| [sum, min, max, or]).collect()
}
