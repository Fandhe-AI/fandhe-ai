//! unique の ROCm 実行入口（`backend-cuda::unique`・`unique_model` の移植。イシュー #3439・R8）。
//!
//! 役割: [`RocmUnique`] が HIP/HIPRTC をロードして `kernels_unique` のソースをコンパイルし、
//! `f32` のスライスを整列・重複除去した `Vec<f32>` を返す。`BackendOps::unique` への結線
//! （`Tensor` の平坦化・結果の `Tensor` 化）は `RocmBackendOps`（#3431・R0）側の責務。
//!
//! 処理: ホストで `total_order_key`（`f32` → 全順序 `u32`。NaN は payload を含め可逆）へ変換し、
//! 2 のべき乗長まで `u32::MAX` でパディングしてデバイスで昇順整列（ビットニック。整数演算のみ）、
//! 読み戻して `key_to_f32` で戻し、隣接重複を `==` で除く。`-0.0` と `0.0` は `==` で同値のため
//! 先に並ぶ `-0.0` が代表として残り、NaN は `==` が偽のためすべて保持される（CPU 参照実装と同一）。
//! 出典: `backend-cuda::unique_model::{total_order_key, key_to_f32}`（非公開 API のため複製）。

use crate::error::RocmError;
use crate::kernels_unique::{BITONIC_STEP_U32_SPEC, render_unique_source};
use crate::launch::{Arg, KernelHost};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

const OP: &str = "unique";

/// `f32` を全順序（`total_cmp` と同じ昇順）を保つ `u32` キーへ写す。可逆。
pub(crate) fn total_order_key(x: f32) -> u32 {
    let bits = x.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// [`total_order_key`] の逆変換。
pub(crate) fn key_to_f32(key: u32) -> f32 {
    let bits = if key >> 31 == 1 {
        key & 0x7FFF_FFFF
    } else {
        !key
    };
    f32::from_bits(bits)
}

/// パディング後の長さ（2 のべき乗）。`i32::MAX` を超える場合は `SizeLimitExceeded`。
pub(crate) fn padded_len(n: usize) -> Result<usize, RocmError> {
    let limit = i32::MAX as usize;
    let padded = n
        .checked_next_power_of_two()
        .ok_or(RocmError::SizeLimitExceeded { op: OP, n, limit })?;
    if padded > limit {
        return Err(RocmError::SizeLimitExceeded { op: OP, n, limit });
    }
    Ok(padded)
}

/// 整列済みキー先頭 `n` 個を `f32` へ戻して隣接重複（`==`）を除く。
pub(crate) fn finish_unique(sorted_keys: &[u32], n: usize) -> Vec<f32> {
    let mut out: Vec<f32> = sorted_keys[..n].iter().map(|&k| key_to_f32(k)).collect();
    out.dedup_by(|cur, prev| *cur == *prev);
    out
}

/// コンパイル済みの unique 用ソートカーネルを保持し、実行する。
pub struct RocmUnique {
    host: KernelHost,
}

impl RocmUnique {
    /// HIP/HIPRTC をロードしカーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_unique_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// `x` の重複を除いた昇順の値（NaN はすべて保持）。
    pub fn run_unique_f32(&self, x: &[f32]) -> Result<Vec<f32>, RocmError> {
        let n = x.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n == 1 {
            return Ok(vec![x[0]]);
        }
        let padded = padded_len(n)?;
        let mut keys: Vec<u32> = x.iter().map(|&v| total_order_key(v)).collect();
        keys.resize(padded, u32::MAX);

        let to_i64 = |v: usize| {
            i64::try_from(v).map_err(|_| RocmError::SizeLimitExceeded {
                op: OP,
                n: v,
                limit: i64::MAX as usize,
            })
        };
        let padded_i = to_i64(padded)?;

        let _guard = self.host.select()?;
        let dkeys = TypedDeviceBuffer::<u32>::from_slice(self.host.runtime(), &keys)?;
        let mut k = 2usize;
        while k <= padded {
            let mut j = k / 2;
            while j >= 1 {
                let args = [
                    Arg::Ptr(dkeys.as_device_ptr()),
                    Arg::I64(to_i64(j)?),
                    Arg::I64(to_i64(k)?),
                    Arg::I64(padded_i),
                ];
                self.host.run(OP, &BITONIC_STEP_U32_SPEC, padded, &args)?;
                j /= 2;
            }
            k *= 2;
        }
        let sorted = dkeys.to_vec()?;
        Ok(finish_unique(&sorted, n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_roundtrip_preserves_bits() {
        let values = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            1e-40,
            -1e-40,
            f32::NAN,
            -f32::NAN,
            f32::from_bits(f32::NAN.to_bits() | 1),
        ];
        for v in values {
            assert_eq!(key_to_f32(total_order_key(v)).to_bits(), v.to_bits());
        }
    }

    #[test]
    fn key_order_matches_total_cmp() {
        let mut values = [
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
            -f32::NAN,
            f32::NAN,
        ];
        values.sort_by(f32::total_cmp);
        let keys: Vec<u32> = values.iter().map(|&v| total_order_key(v)).collect();
        assert!(keys.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn padded_len_bounds() {
        assert_eq!(padded_len(2).unwrap(), 2);
        assert_eq!(padded_len(5).unwrap(), 8);
        assert_eq!(padded_len(1 << 30).unwrap(), 1 << 30);
        assert!(matches!(
            padded_len((1 << 30) + 1),
            Err(RocmError::SizeLimitExceeded { .. })
        ));
        assert!(padded_len(usize::MAX).is_err());
    }
}
