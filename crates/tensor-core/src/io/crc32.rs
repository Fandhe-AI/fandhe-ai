//! CRC-32（IEEE 802.3、多項式 0xEDB88320）の自作実装。
//!
//! `io::npz`（ZIP コンテナの各エントリの整合性検証・書き出し時の CRC
//! 計算）専用の `pub(crate)` ユーティリティ。依存追加なしの方針
//! （REQ-1・`.claude/rules/deps-policy.md`）に従い、256 エントリの
//! ルックアップテーブルを `const fn` でコンパイル時に構築する。

/// 反転多項式 `0xEDB88320`（ZIP／PNG 等が使う標準 CRC-32 多項式）。
const POLY: u32 = 0xEDB8_8320;

/// 256 エントリの CRC-32 ルックアップテーブル。`const fn` でビルド時に
/// 一度だけ計算し、実行時コストをゼロにする。
const TABLE: [u32; 256] = build_table();

const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// `data` の CRC-32（IEEE 802.3）を計算する。ZIP の CRC フィールドと
/// 同じ規約（初期値 `0xFFFFFFFF`・最終 XOR `0xFFFFFFFF`）。
pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ TABLE[idx];
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::crc32;

    /// CRC-32 の標準既知ベクタ（RFC 等で広く引用される値）。
    #[test]
    fn known_vector_123456789() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    /// 空入力の CRC-32 は 0。
    #[test]
    fn empty_input_is_zero() {
        assert_eq!(crc32(b""), 0);
    }
}
