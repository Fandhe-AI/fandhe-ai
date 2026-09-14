//! `unique`（イシュー #1734）の GPU 非依存な純関数群（キー変換・
//! ビットニックソートのホストモデル）。Linux 実行可能な単体テストを
//! ここに同居させ、`kernels_unique.rs::BITONIC_STEP_U32` の逐語モデル
//! （[`bitonic_step_host`]）として GPU 実機がなくてもアルゴリズムの
//! 正しさを検証できるようにする（`backend-metal::gather_scatter_model`
//! と同じ「意図的複製」方針）。
//!
//! # totalOrder キー変換
//!
//! `f32` の IEEE 754 totalOrder（`f32::total_cmp` と同じ全順序）と
//! `u32` の通常の昇順比較が一致するよう、[`total_order_key`] で
//! `f32` の bit 表現を単調な `u32` へ変換する（標準的な
//! float-to-sortable-uint 変換）:
//! - 符号 bit が立っている（負数）場合: 全 bit を反転する（`!bits`）。
//!   これにより負数同士は絶対値が大きいほど小さいキーになり、かつ
//!   すべての負数キーは非負数キーより小さくなる。
//! - 符号 bit が立っていない（非負数）場合: 符号 bit だけを立てる
//!   （`bits | 0x8000_0000`）。これにより非負数キーはすべて負数キー
//!   より大きくなる。
//!
//! [`key_to_f32`] は逆変換（往復が恒等であることを本モジュールの
//! プロパティテストで検証する）。

/// `f32` の bit 表現を totalOrder 昇順の `u32` キーへ変換する
/// （本モジュール doc 参照）。
pub fn total_order_key(x: f32) -> u32 {
    let bits = x.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// [`total_order_key`] の逆変換。
pub fn key_to_f32(key: u32) -> f32 {
    let bits = if key >> 31 == 1 {
        key & 0x7FFF_FFFF
    } else {
        !key
    };
    f32::from_bits(bits)
}

/// [`crate::kernels_unique::BITONIC_STEP_U32`] の逐語ホストモデル
/// （GPU カーネル本文と同一の比較・スワップロジック。1 スレッド分の
/// 処理を `i` ごとに呼び出す形へ変換したもの）。`keys` は `padded`
/// （2 のべき乗）長で、`n` は `padded` と同値（カーネル引数と同じ
/// 意味）。テスト専用（`unique.rs::CudaUnique::run_unique_f32` は
/// GPU 側ループを直接持ち、本関数を呼ばない——`bitonic_sort_host` doc
/// 参照）のため `#[cfg(test)]`。
#[cfg(test)]
pub fn bitonic_step_host(keys: &mut [u32], j: usize, k: usize, n: usize) {
    for i in 0..n {
        let ixj = i ^ j;
        if ixj <= i || ixj >= n {
            continue;
        }
        let a = keys[i];
        let b = keys[ixj];
        let ascending = (i & k) == 0;
        let should_swap = if ascending { a > b } else { a < b };
        if should_swap {
            keys[i] = b;
            keys[ixj] = a;
        }
    }
}

/// `keys`（`padded`。2 のべき乗長）へ標準的なビットニックソート
/// （昇順）を適用する（`bitonic_step_host` をステップ順に繰り返す
/// ホストモデル。`unique.rs::CudaUnique::run_unique_f32` の GPU 側
/// ループ構造と同じ `k`／`j` の入れ子だが、GPU 側は本関数を呼ばず
/// 独立にループを持つ——両者の一致は
/// `bitonic_sort_host_matches_total_cmp_various_sizes` の期待値
/// 〈`total_cmp` によるソート結果〉と `run_unique_f32` の実機テストの
/// **両方**が同じ `total_cmp` 基準へ収束することで間接的に担保する）。
/// `padded` が 2 のべき乗でないと未定義の並びになる（呼び出し元が
/// 事前に保証する契約）。テスト専用のため `#[cfg(test)]`。
#[cfg(test)]
pub fn bitonic_sort_host(keys: &mut [u32]) {
    let n = keys.len();
    if n <= 1 {
        return;
    }
    let mut k = 2usize;
    while k <= n {
        let mut j = k / 2;
        while j >= 1 {
            bitonic_step_host(keys, j, k, n);
            j /= 2;
        }
        k *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_order_key_roundtrip_matches_bits() {
        let values = [
            0.0f32,
            -0.0f32,
            1.0f32,
            -1.0f32,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::NAN,
            -f32::NAN,
            f32::from_bits(f32::NAN.to_bits() | 1),
            1e-40f32, // 非正規化数
            -1e-40f32,
        ];
        for &v in &values {
            let key = total_order_key(v);
            let back = key_to_f32(key);
            assert_eq!(
                back.to_bits(),
                v.to_bits(),
                "roundtrip failed for {v} (bits={:#x})",
                v.to_bits()
            );
        }
    }

    /// `u32` の通常昇順比較が `f32::total_cmp` と一致することを
    /// 網羅的に確認する（代表値のペアワイズ比較）。
    #[test]
    fn total_order_key_ordering_matches_total_cmp() {
        let nan_pos = f32::NAN;
        let nan_pos2 = f32::from_bits(f32::NAN.to_bits() | 1);
        let nan_neg = -f32::NAN;
        let mut values = vec![
            f32::NEG_INFINITY,
            f32::MIN,
            -1.0,
            -1e-40,
            -0.0,
            0.0,
            1e-40,
            1.0,
            f32::MAX,
            f32::INFINITY,
            nan_neg,
            nan_pos,
            nan_pos2,
        ];
        // total_cmp によるソート結果を「真値」とする。
        values.sort_by(f32::total_cmp);
        let keys: Vec<u32> = values.iter().map(|&v| total_order_key(v)).collect();
        // 真値でソート済みの列から得たキー列は昇順（同値タイは
        // bit 同一のためキーも同一）でなければならない。
        for w in keys.windows(2) {
            assert!(
                w[0] <= w[1],
                "key ordering diverges from total_cmp ordering: {:#x} > {:#x}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn bitonic_sort_host_matches_total_cmp_various_sizes() {
        for &n in &[1usize, 2, 3, 4, 7, 8, 16, 33, 64, 1000, 4097] {
            let padded = n.next_power_of_two();
            // xorshift 的な決定的疑似乱数で入力を生成する（外部乱数
            // クレートへ依存しない。テスト専用の単純な生成器）。
            let mut seed = 0x1234_5678u64.wrapping_add(n as u64);
            let mut input: Vec<f32> = (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    // 上位 32bit を f32 の bit パターンとして使う
                    // （NaN／inf／非正規化数も含めて全 bit パターンを
                    // 母集団に含める）。
                    f32::from_bits((seed >> 32) as u32)
                })
                .collect();
            // NaN 混入時の total_cmp 基準列を先に確定する。
            let mut expected = input.clone();
            expected.sort_by(f32::total_cmp);

            let mut keys: Vec<u32> = input.iter().map(|&v| total_order_key(v)).collect();
            keys.resize(padded, u32::MAX);
            bitonic_sort_host(&mut keys);
            let sorted: Vec<f32> = keys[..n].iter().map(|&k| key_to_f32(k)).collect();

            for (i, (&a, &b)) in sorted.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "n={n}: mismatch at index {i} (got={a:?}, want={b:?})"
                );
            }
            // 入力を破壊していないことも確認（`bitonic_sort_host` は
            // `keys` のみを変更し `input` は不変）。
            let _ = &mut input;
        }
    }

    #[test]
    fn bitonic_sort_host_all_identical_elements() {
        let n: usize = 37;
        let padded = n.next_power_of_two();
        let mut keys: Vec<u32> = vec![total_order_key(5.0); n];
        keys.resize(padded, u32::MAX);
        bitonic_sort_host(&mut keys);
        for &k in &keys[..n] {
            assert_eq!(key_to_f32(k), 5.0);
        }
    }
}
