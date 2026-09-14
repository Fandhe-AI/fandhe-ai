//! `unique`（イシュー #1734）の GPU 非依存な純関数群（キー変換・
//! ビットニックソートのホストモデル・起動前検証）。`crate::
//! gather_scatter_model`・`crates/backend-cuda/src/unique_model.rs` と
//! 同じ「`objc2` 系 FFI に触れないため `cfg(target_os = "macos")` を
//! 付けず Linux（本実装環境・CI）でも単体テストが回る」設計判断を
//! 踏襲する（意図的複製。CUDA 側とアルゴリズムは同一だが、クレートを
//! 跨いで `pub(crate)` 関数を共有できないため独立実装する）。
//!
//! # totalOrder キー変換
//!
//! [`total_order_key`]／[`key_to_f32`] は `crates/backend-cuda/src/
//! unique_model.rs` と同一の変換（doc は同ファイルを参照）。

/// `f32` の bit 表現を totalOrder 昇順の `u32` キーへ変換する。
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

/// [`crate::shaders`]（`unique.metal::bitonic_step_u32`）の逐語
/// ホストモデル（`crates/backend-cuda/src/unique_model.rs::
/// bitonic_step_host` と同一）。テスト専用（`unique.rs::MetalUnique::
/// run_unique_f32` は GPU 側ループを直接持ち、本関数を呼ばない）の
/// ため `#[cfg(test)]`。
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
/// （昇順）を適用する（`crates/backend-cuda/src/unique_model.rs::
/// bitonic_sort_host` と同一）。テスト専用のため `#[cfg(test)]`。
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

/// `unique.rs::MetalUnique::run_unique_f32` が `dispatch_sync` の
/// クロージャに入る前に完了させる起動前検証（`dispatch_sync` の
/// クロージャは `Result` を返せないため、検証はすべてここで完結させる
/// 契約。`gather_scatter_model::validate_gather_launch` と同じ
/// 「エンコード関数へ渡す前にホスト側で完結させる」方針）。
///
/// `n == 0`／`n == 1` は呼び出し元（`unique.rs::MetalUnique::
/// run_unique_f32`）が GPU 起動なしで早期処理する契約のため、本関数は
/// 通常 `n >= 2` でのみ呼ばれる。ただし本関数自身は `pub fn`（`pub mod
/// unique_model`）であり、クレート外の利用者が `n < 2` を直接渡して
/// 呼ぶことも構文上可能なため、`debug_assert!` による契約違反検知
/// （debug ビルドでの panic）は行わない（本番経路の panic 禁止・
/// AGENTS.md。codex-review 指摘・PR #1828 是正）。`n < 2` でも
/// `checked_next_power_of_two`（`0`／`1` いずれも `Some(1)`）がそのまま
/// 有効な `padded` を返すため、明示的な早期分岐は不要で常に正常値
/// （またはサイズ超過時の型付きエラー）を返す。
///
/// 戻り値: `padded`（次の 2 のべき乗）。`i32::MAX` を超える場合は
/// `Err(UniquePrepareError::SizeLimitExceeded)`（呼び出し元 `ops.rs`
/// が `BackendError::Unsupported` へ写像し、ホストフォールバックへ
/// 委ねる。`GS_MAX_RANK` 超過時の `Unsupported` 返却と同じ設計判断）。
pub fn checked_padded_len(n: usize) -> Result<usize, UniquePrepareError> {
    let padded = n
        .checked_next_power_of_two()
        .ok_or(UniquePrepareError::SizeLimitExceeded {
            n,
            limit: usize::MAX,
        })?;
    if i32::try_from(padded).is_err() {
        return Err(UniquePrepareError::SizeLimitExceeded {
            n,
            limit: i32::MAX as usize,
        });
    }
    Ok(padded)
}

/// [`checked_padded_len`] の失敗理由。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniquePrepareError {
    /// `padded`（次の 2 のべき乗）がバックエンド固有上限
    /// （カーネル引数 `uint`／`int` の範囲）を超えた。
    SizeLimitExceeded { n: usize, limit: usize },
}

impl std::fmt::Display for UniquePrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UniquePrepareError::SizeLimitExceeded { n, limit } => {
                write!(f, "unique size limit exceeded: n={n} exceeds limit={limit}")
            }
        }
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
            f32::NAN,
            -f32::NAN,
            f32::from_bits(f32::NAN.to_bits() | 1),
            1e-40f32,
            -1e-40f32,
        ];
        for &v in &values {
            let key = total_order_key(v);
            let back = key_to_f32(key);
            assert_eq!(back.to_bits(), v.to_bits());
        }
    }

    #[test]
    fn bitonic_sort_host_matches_total_cmp_various_sizes() {
        for &n in &[1usize, 2, 3, 4, 7, 8, 16, 33, 64, 1000, 4097] {
            let padded = n.next_power_of_two();
            let mut seed = 0x8765_4321u64.wrapping_add(n as u64);
            let input: Vec<f32> = (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    f32::from_bits((seed >> 32) as u32)
                })
                .collect();
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
        }
    }

    #[test]
    fn checked_padded_len_rejects_i32_overflow() {
        let too_big = (i32::MAX as usize) + 1;
        let err = checked_padded_len(too_big).unwrap_err();
        assert!(matches!(err, UniquePrepareError::SizeLimitExceeded { .. }));
    }

    #[test]
    fn checked_padded_len_accepts_small_sizes() {
        assert_eq!(checked_padded_len(5).unwrap(), 8);
        assert_eq!(checked_padded_len(1000).unwrap(), 1024);
    }

    /// `checked_padded_len` は `pub fn`（`pub mod unique_model`）で
    /// あり、通常の呼び出し元（`unique.rs::MetalUnique::
    /// run_unique_f32`）は `n >= 2` のみで呼ぶ契約だが、クレート外の
    /// 利用者が `n < 2` を直接渡して呼ぶことも構文上可能である。
    /// debug ビルドでも panic せず正常値を返すことを確認する回帰
    /// テスト（codex-review 指摘・PR #1828 是正）。
    #[test]
    fn checked_padded_len_does_not_panic_on_small_input() {
        assert_eq!(checked_padded_len(0).unwrap(), 1);
        assert_eq!(checked_padded_len(1).unwrap(), 1);
    }
}
