//! `shaders/scan.metal`（累積和／累積積。イシュー #1740・親イシュー
//! #1731）のホスト側逐語モデル。`crate::soft_f64` の binary64
//! ソフトウェアエミュレーション（[`crate::soft_f64::widen_f32_bits`]／
//! [`crate::soft_f64::add_f64_bits`]／[`crate::soft_f64::mul_f64_bits`]／
//! [`crate::soft_f64::narrow_f64_bits`]）を `scan.metal::scan_f64_*`
//! と同じ演算列で呼び出すことで、GPU 側カーネルが正しい binary64
//! 逐次演算列を実行することを Mac 実機に到達できない環境
//! （本実装環境。Linux・CI）でも機械的に裏付ける（`unique_model.rs`・
//! `gather_scatter_model.rs` と同じ設計判断: `objc2` 系 FFI に触れない
//! ため `cfg(target_os = "macos")` を付けない）。
//!
//! # なぜこれが Metal 側の正しさの根拠になるか
//!
//! [`cumsum_lane_soft_f64`]／[`cumprod_lane_soft_f64`] は
//! `scan.metal::cumsum_f32`／`cumprod_f32` の 1 lane 分のループ本体
//! （`widen` → `add`／`mul` の逐次 → 各ステップ `narrow`）を Rust へ
//! 逐語移植したものである。本モジュールの単体テストはこの逐語モデルを
//! **native `f64` 逐次参照実装**（`cumsum_lane_native_f64`／
//! `cumprod_lane_native_f64`。`backend-cpu::scan::cumsum`／`cumprod` と
//! 同一アルゴリズム）と突き合わせ `to_bits` 完全一致（NaN はクラス
//! 一致）を確認する。この一致は「`soft_f64` の個々の演算が `f64` と
//! bit 一致する」ことに加えて「scan.metal のループ構造自体が正しい」
//! ことも検証する（個々の演算の正しさは `soft_f64.rs` 自身の単体
//! テストが既に広く確認済みだが、ループの組み方や `widen`／`narrow`
//! を呼ぶ位置を誤ると本モジュールのテストが失敗する）。

use crate::soft_f64::{add_f64_bits, mul_f64_bits, narrow_f64_bits, widen_f32_bits};

/// `scan.metal::cumsum_f32` の 1 lane 分のループ本体の逐語モデル
/// （`soft_f64` 経由）。`xs` は当該 lane の `axis_len` 要素（`dim` 添字
/// 昇順）。戻り値は各ステップの `out[idx]`（`f32` へ downcast した
/// スナップショット）を昇順に並べたもの。
pub fn cumsum_lane_soft_f64(xs: &[f32]) -> Vec<f32> {
    let mut acc: u64 = widen_f32_bits(0.0f32.to_bits());
    let mut out = Vec::with_capacity(xs.len());
    for &x in xs {
        acc = add_f64_bits(acc, widen_f32_bits(x.to_bits()));
        out.push(f32::from_bits(narrow_f64_bits(acc)));
    }
    out
}

/// `scan.metal::cumprod_f32` の 1 lane 分のループ本体の逐語モデル。
/// [`cumsum_lane_soft_f64`] と同じ構造だがアキュムレータは `1.0` から
/// 開始し [`mul_f64_bits`] で乗算する。
pub fn cumprod_lane_soft_f64(xs: &[f32]) -> Vec<f32> {
    let mut acc: u64 = widen_f32_bits(1.0f32.to_bits());
    let mut out = Vec::with_capacity(xs.len());
    for &x in xs {
        acc = mul_f64_bits(acc, widen_f32_bits(x.to_bits()));
        out.push(f32::from_bits(narrow_f64_bits(acc)));
    }
    out
}

/// 1 lane 分の native `f64` 逐次参照実装（`backend-cpu::scan::cumsum`
/// と同一アルゴリズム。[`cumsum_lane_soft_f64`] との bit 一致検証用の
/// 独立実装）。
#[cfg(test)]
pub fn cumsum_lane_native_f64(xs: &[f32]) -> Vec<f32> {
    let mut acc: f64 = 0.0;
    xs.iter()
        .map(|&x| {
            acc += x as f64;
            acc as f32
        })
        .collect()
}

/// 1 lane 分の native `f64` 逐次参照実装（`cumprod` 版。`backend-cpu::
/// scan::cumprod` と同一アルゴリズム）。
#[cfg(test)]
pub fn cumprod_lane_native_f64(xs: &[f32]) -> Vec<f32> {
    let mut acc: f64 = 1.0;
    xs.iter()
        .map(|&x| {
            acc *= x as f64;
            acc as f32
        })
        .collect()
}

/// `x`（`shape` 形状・`dim` 軸に沿って走査）全体へ [`cumsum_lane_soft_f64`]
/// を lane ごとに適用する（`backend-cpu::scan::cumsum` と同一の
/// `outer`／`axis_len`／`inner` 分解。テスト専用のため `#[cfg(test)]`
/// ——本体経路は `scan.rs::MetalScan::run_cumsum_f32` が GPU 側で直接
/// 計算するため、本関数は突合用の Rust 側参照実装としてのみ使う）。
#[cfg(test)]
pub fn cumsum_soft_f64(x: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
    scan_over_shape(x, shape, dim, cumsum_lane_soft_f64)
}

/// [`cumsum_soft_f64`] の `cumprod` 版。
#[cfg(test)]
pub fn cumprod_soft_f64(x: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
    scan_over_shape(x, shape, dim, cumprod_lane_soft_f64)
}

/// `x`（`shape` 形状・`dim` 軸）を `outer`／`axis_len`／`inner` へ分解
/// し、`lane_fn`（1 lane 分の `axis_len` 要素を受け取り同じ長さの出力を
/// 返す）を各 lane へ適用する（`backend-cpu::scan` と同じ添字計算）。
#[cfg(test)]
fn scan_over_shape(
    x: &[f32],
    shape: &[usize],
    dim: usize,
    lane_fn: impl Fn(&[f32]) -> Vec<f32>,
) -> Vec<f32> {
    let outer: usize = shape[..dim].iter().product();
    let axis_len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let mut out = vec![0f32; x.len()];
    for o in 0..outer {
        for i in 0..inner {
            let lane: Vec<f32> = (0..axis_len)
                .map(|a| x[(o * axis_len + a) * inner + i])
                .collect();
            let lane_out = lane_fn(&lane);
            for (a, &v) in lane_out.iter().enumerate() {
                out[(o * axis_len + a) * inner + i] = v;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits_or_nan_class(v: f32) -> Result<u32, ()> {
        if v.is_nan() { Err(()) } else { Ok(v.to_bits()) }
    }

    /// テスト専用の最小限 xorshift64* PRNG（`bench_harness` への
    /// 依存を避けるための自己完結実装。有限な `f32` のみを返す）。
    struct TestRng(u64);
    impl TestRng {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn next_f32(&mut self) -> f32 {
            let bits = (self.next_u64() >> 32) as u32;
            let sign = if bits & 1 == 0 { 1.0 } else { -1.0 };
            let v = f32::from_bits(bits & 0x7fff_ffff) * sign;
            if v.is_finite() { v } else { 1.0 }
        }
        fn next_len(&mut self, max_inclusive: usize) -> usize {
            1 + (self.next_u64() as usize % max_inclusive)
        }
    }

    /// [`cumsum_lane_soft_f64`] と [`cumsum_lane_native_f64`] が
    /// ランダム値で `to_bits` 完全一致（NaN はクラス一致）する
    /// （本モジュール冒頭コメント「なぜこれが Metal 側の正しさの根拠に
    /// なるか」参照）。
    #[test]
    fn cumsum_lane_soft_f64_matches_native_f64_random() {
        let mut rng = TestRng(0x1234_5678_9abc_def0u64);
        for _ in 0..200 {
            let n = rng.next_len(12);
            let xs: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
            let soft = cumsum_lane_soft_f64(&xs);
            let native = cumsum_lane_native_f64(&xs);
            for (i, (&a, &b)) in soft.iter().zip(native.iter()).enumerate() {
                match (bits_or_nan_class(a), bits_or_nan_class(b)) {
                    (Ok(ba), Ok(bb)) => {
                        assert_eq!(ba, bb, "cumsum lane mismatch at step {i}: xs={xs:?}")
                    }
                    (Err(()), Err(())) => {}
                    _ => {
                        panic!("cumsum lane NaN-class mismatch at step {i}: xs={xs:?} a={a} b={b}")
                    }
                }
            }
        }
    }

    /// [`cumprod_lane_soft_f64`] 版。
    #[test]
    fn cumprod_lane_soft_f64_matches_native_f64_random() {
        let mut rng = TestRng(0x0fed_cba9_8765_4321u64);
        for _ in 0..200 {
            let n = rng.next_len(12);
            let xs: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
            let soft = cumprod_lane_soft_f64(&xs);
            let native = cumprod_lane_native_f64(&xs);
            for (i, (&a, &b)) in soft.iter().zip(native.iter()).enumerate() {
                match (bits_or_nan_class(a), bits_or_nan_class(b)) {
                    (Ok(ba), Ok(bb)) => {
                        assert_eq!(ba, bb, "cumprod lane mismatch at step {i}: xs={xs:?}")
                    }
                    (Err(()), Err(())) => {}
                    _ => {
                        panic!("cumprod lane NaN-class mismatch at step {i}: xs={xs:?} a={a} b={b}")
                    }
                }
            }
        }
    }

    /// 特殊値: `-0.0` 先頭。
    #[test]
    fn cumsum_handles_negative_zero_leading() {
        // `acc` は `+0.0` から開始するため `+0.0 + (-0.0) == +0.0`
        // （IEEE 754: 異符号の加算のうち少なくとも一方が非ゼロで
        // なければ `+0.0`。round-to-nearest 既定モードでは `x + (-x)`
        // は常に `+0.0`）。`out[0].to_bits()` が `-0.0` ではなく `+0.0`
        // になることは native `f64` 参照実装（`acc: f64 = 0.0; acc +=
        // -0.0;`）と同じ挙動であり、`kernels_scan.rs` モジュール doc
        // 「`cumsum([-0.0]) == [+0.0]`」の記述と一致する。
        let xs = [-0.0f32, 1.0];
        let out = cumsum_lane_soft_f64(&xs);
        let native = cumsum_lane_native_f64(&xs);
        assert_eq!(out[0].to_bits(), native[0].to_bits());
        assert_eq!(out[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(out[1], 1.0);
    }

    /// 特殊値: subnormal を含む列。
    #[test]
    fn cumsum_handles_subnormal() {
        let subnormal = f32::from_bits(1);
        let xs = [subnormal, subnormal, subnormal];
        let soft = cumsum_lane_soft_f64(&xs);
        let native = cumsum_lane_native_f64(&xs);
        assert_eq!(
            soft.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            native.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// 契約証明ベクトル（`cumsum([1e8, 1.0, -1e8]) == [1e8, 1e8, 1.0]`）。
    #[test]
    fn cumsum_contract_vector_matches_f64_accumulator_semantics() {
        let out = cumsum_lane_soft_f64(&[1e8, 1.0, -1e8]);
        assert_eq!(out, vec![1e8, 1e8, 1.0]);
    }

    /// 契約証明ベクトル（`cumprod([1e-30, 1e-30, 1e30])[2] != 0.0`。
    /// `f32` 逐次アキュムレータなら underflow して `0.0` になるが
    /// `f64` 相当のアキュムレータなら非零のはず）。
    #[test]
    fn cumprod_contract_vector_avoids_underflow() {
        let out = cumprod_lane_soft_f64(&[1e-30, 1e-30, 1e30]);
        assert_ne!(out[2], 0.0);
    }

    /// inf + -inf → NaN。
    #[test]
    fn cumsum_inf_plus_neg_inf_is_nan() {
        let out = cumsum_lane_soft_f64(&[f32::INFINITY, f32::NEG_INFINITY]);
        assert!(out[1].is_nan());
    }

    /// 0 * inf → NaN。
    #[test]
    fn cumprod_zero_times_inf_is_nan() {
        let out = cumprod_lane_soft_f64(&[0.0, f32::INFINITY]);
        assert!(out[1].is_nan());
    }

    /// [`cumsum_soft_f64`]／[`cumprod_soft_f64`]（形状分解込み）が
    /// `backend-cpu::scan` 相当の `outer`／`axis_len`／`inner` 分解と
    /// 一致することを 2-D 形状で確認する。
    #[test]
    fn cumsum_soft_f64_respects_shape_dim() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2usize, 3usize];
        let out_dim1 = cumsum_soft_f64(&x, &shape, 1);
        // 行ごとに cumsum: [1,2,3] -> [1,3,6]、[4,5,6] -> [4,9,15]。
        assert_eq!(out_dim1, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);

        let out_dim0 = cumsum_soft_f64(&x, &shape, 0);
        // 列ごとに cumsum: col0 [1,4]->[1,5]、col1 [2,5]->[2,7]、
        // col2 [3,6]->[3,9]。
        assert_eq!(out_dim0, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
    }
}
