//! RFC 1951（DEFLATE）伸長の自作実装（読み込み専用）。
//!
//! `io::npz`（`np.savez_compressed` が使う method 8／DEFLATE エントリの
//! 伸長）専用の `pub(crate)` ユーティリティ。依存追加なしの方針
//! （REQ-1・`.claude/rules/deps-policy.md`）に従い自作する。アルゴリズム
//! 構造は Mark Adler の public domain 参照実装 `puff.c` の Canonical
//! Huffman 復号方式（`construct`／`decode`）を踏襲しつつ、`panic` では
//! なくすべて `Result` で境界検査・整合性検査を行う
//! （`.claude/rules/security.md` A03/A04/A05。伸長爆弾・不正ストリーム
//! 対策）。
//!
//! 圧縮のみ（`io::npz::write_npz_bytes`）は STORED（無圧縮）に限定して
//! おり、本モジュールは npz **読み込み**専用である。

use super::NpyError;

/// Huffman 符号長の上限（DEFLATE 仕様上の最大）。
const MAX_BITS: usize = 15;

/// 長さ符号（257〜285）の基底長・追加ビット数（RFC 1951 §3.2.5）。
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// 距離符号（0〜29）の基底距離・追加ビット数。
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// 符号長の並び替え順（dynamic ブロックの code-length code、RFC 1951
/// §3.2.7）。
const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// DEFLATE ストリームの理論上の最大圧縮比（RFC 1951 の非圧縮ブロック
/// オーバーヘッドから導かれる下限の逆数に安全マージンを乗せたもの）。
/// `expected_len` がこの比を超えて大きい場合は伸長爆弾疑いとして確保
/// 前に拒否する（`.claude/rules/security.md` A04/A05）。
const MAX_COMPRESSION_RATIO: u64 = 1032;

/// `expected_len`（伸長後サイズ）の絶対上限。呼び出し元
/// （`io::npz::read_member_bytes`）は central directory の宣言値
/// （`entry.uncompressed_size`）を `io::npz::MAX_MEMBER_DECOMPRESSED_
/// BYTES` と既に突き合わせているが、本関数はそれに依存せず単独でも
/// 安全であるよう同じ絶対上限をここでも検査する（多層防御。将来 npz
/// 以外の呼び出し元が追加され、呼び出し前チェックを書き忘れても
/// 伸長爆弾を防げるようにする。`.claude/rules/security.md`
/// A04/A05。PR #2318 レビュー指摘・P0）。値は `io::MAX_FILE_READ_BYTES`
/// と同じ桁（新規閾値を持ち込まない）。
const MAX_INFLATE_OUTPUT_BYTES: u64 = super::MAX_FILE_READ_BYTES;

/// LSB ファーストでビット列を読み取るリーダ。範囲外アクセスは
/// `NpyError::InvalidDeflate` を返す（`panic` しない）。
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn get_bit(&mut self) -> Result<u32, NpyError> {
        let byte = *self
            .data
            .get(self.byte_pos)
            .ok_or(NpyError::InvalidDeflate("入力が途中で尽きた"))?;
        let bit = (byte as u32 >> self.bit_pos) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Ok(bit)
    }

    /// `n` ビットを LSB ファーストで読み、整数として組み立てる
    /// （`n <= 16` の呼び出しのみを想定）。
    fn get_bits(&mut self, n: u32) -> Result<u32, NpyError> {
        let mut value = 0u32;
        for i in 0..n {
            value |= self.get_bit()? << i;
        }
        Ok(value)
    }

    /// 現在位置をバイト境界へ切り上げる（stored ブロックの前に必要）。
    fn align_to_byte(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], NpyError> {
        let end = self
            .byte_pos
            .checked_add(n)
            .ok_or(NpyError::InvalidDeflate("stored ブロック長が不正"))?;
        let slice = self
            .data
            .get(self.byte_pos..end)
            .ok_or(NpyError::InvalidDeflate("stored ブロックが入力範囲外"))?;
        self.byte_pos = end;
        Ok(slice)
    }

    /// 現在位置までに消費したバイト数（ビット位置をバイト境界へ切り上げ）。
    /// `inflate` が `BFINAL` ブロック読み終わり後にストリーム終端検査
    /// （PR #2318 レビュー指摘・P2）で使う。最終バイトの未使用上位ビット
    /// （パディング）は RFC 1951 上ゼロが保証されないため、値そのものは
    /// 検査せず消費バイト数の一致のみを見る。
    fn consumed_bytes(&self) -> usize {
        self.byte_pos + usize::from(self.bit_pos != 0)
    }
}

/// Canonical Huffman 復号表（`puff.c::huffman` と同型）。
struct Huffman {
    /// 各符号長（1..=MAX_BITS）の符号数。`counts[0]` は長さ 0（未使用）
    /// のシンボル数。
    counts: [u16; MAX_BITS + 1],
    /// 符号長・長さ内シンボル順で並べたシンボル値。
    symbols: Vec<u16>,
}

/// 符号長配列（`lengths[symbol] = 0..=MAX_BITS`）から Canonical Huffman
/// 復号表を構築する。戻り値の `i32` は `puff.c::construct` と同じ
/// 「spare」値: 0 なら完全な符号集合、負なら over-subscribed（不正）、
/// 正なら incomplete（呼び出し側が許容条件を判定する）。
fn construct(lengths: &[u8]) -> Result<(Huffman, i32), NpyError> {
    let mut counts = [0u16; MAX_BITS + 1];
    for &len in lengths {
        let len = len as usize;
        if len > MAX_BITS {
            return Err(NpyError::InvalidDeflate("符号長が上限を超える"));
        }
        counts[len] += 1;
    }
    if counts[0] as usize == lengths.len() {
        // 有効な符号が 1 つもない（decode() は常に失敗する空表）。
        return Ok((
            Huffman {
                counts,
                symbols: Vec::new(),
            },
            0,
        ));
    }

    let mut left: i32 = 1;
    for len in 1..=MAX_BITS {
        left <<= 1;
        left -= counts[len] as i32;
        if left < 0 {
            // over-subscribed（複数シンボルが同じ符号空間を奪い合っている）。
            return Ok((
                Huffman {
                    counts,
                    symbols: Vec::new(),
                },
                left,
            ));
        }
    }

    let mut offsets = [0u16; MAX_BITS + 1];
    for len in 1..MAX_BITS {
        offsets[len + 1] = offsets[len] + counts[len];
    }
    let mut symbols = vec![0u16; lengths.len() - counts[0] as usize];
    for (symbol, &len) in lengths.iter().enumerate() {
        if len != 0 {
            let len = len as usize;
            symbols[offsets[len] as usize] = symbol as u16;
            offsets[len] += 1;
        }
    }

    Ok((Huffman { counts, symbols }, left))
}

/// Canonical Huffman 符号を 1 つ読み取り、対応するシンボルを返す
/// （`puff.c::decode` と同型のビット単位走査）。
fn decode(reader: &mut BitReader<'_>, huff: &Huffman) -> Result<u16, NpyError> {
    let mut code: i32 = 0;
    let mut first: i32 = 0;
    let mut index: i32 = 0;
    for len in 1..=MAX_BITS {
        code |= reader.get_bit()? as i32;
        let count = huff.counts[len] as i32;
        if code - first < count {
            let idx = (index + (code - first)) as usize;
            return huff
                .symbols
                .get(idx)
                .copied()
                .ok_or(NpyError::InvalidDeflate("Huffman シンボル添字が範囲外"));
        }
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    Err(NpyError::InvalidDeflate(
        "Huffman 符号が復号表に見つからない",
    ))
}

/// 固定 Huffman 符号（RFC 1951 §3.2.6）の literal/length・distance 復号
/// 表を構築する。
fn fixed_tables() -> Result<(Huffman, Huffman), NpyError> {
    let mut lit_lengths = [0u8; 288];
    for (i, l) in lit_lengths.iter_mut().enumerate() {
        *l = match i {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    let dist_lengths = [5u8; 30];
    let (lit, _) = construct(&lit_lengths)?;
    let (dist, _) = construct(&dist_lengths)?;
    Ok((lit, dist))
}

/// dynamic Huffman 符号（RFC 1951 §3.2.7）のブロックヘッダを読み、
/// literal/length・distance 復号表を構築する。
fn dynamic_tables(reader: &mut BitReader<'_>) -> Result<(Huffman, Huffman), NpyError> {
    let hlit = reader.get_bits(5)? as usize + 257;
    let hdist = reader.get_bits(5)? as usize + 1;
    let hclen = reader.get_bits(4)? as usize + 4;
    if hlit > 286 || hdist > 30 {
        return Err(NpyError::InvalidDeflate("HLIT/HDIST が範囲外"));
    }

    let mut cl_lengths = [0u8; 19];
    for &pos in CODE_LENGTH_ORDER.iter().take(hclen) {
        cl_lengths[pos] = reader.get_bits(3)? as u8;
    }
    let (cl_huff, cl_left) = construct(&cl_lengths)?;
    if cl_left != 0 {
        return Err(NpyError::InvalidDeflate("code-length 符号が不完全"));
    }

    let total = hlit + hdist;
    let mut lengths = vec![0u8; total];
    let mut index = 0usize;
    while index < total {
        let symbol = decode(reader, &cl_huff)?;
        match symbol {
            0..=15 => {
                lengths[index] = symbol as u8;
                index += 1;
            }
            16 => {
                if index == 0 {
                    return Err(NpyError::InvalidDeflate("符号長 16 の反復対象がない"));
                }
                let prev = lengths[index - 1];
                let repeat = 3 + reader.get_bits(2)? as usize;
                if index + repeat > total {
                    return Err(NpyError::InvalidDeflate("符号長の反復が範囲外"));
                }
                for _ in 0..repeat {
                    lengths[index] = prev;
                    index += 1;
                }
            }
            17 => {
                let repeat = 3 + reader.get_bits(3)? as usize;
                if index + repeat > total {
                    return Err(NpyError::InvalidDeflate("符号長の反復が範囲外"));
                }
                for _ in 0..repeat {
                    lengths[index] = 0;
                    index += 1;
                }
            }
            18 => {
                let repeat = 11 + reader.get_bits(7)? as usize;
                if index + repeat > total {
                    return Err(NpyError::InvalidDeflate("符号長の反復が範囲外"));
                }
                for _ in 0..repeat {
                    lengths[index] = 0;
                    index += 1;
                }
            }
            _ => return Err(NpyError::InvalidDeflate("code-length シンボルが範囲外")),
        }
    }
    if lengths[256] == 0 {
        return Err(NpyError::InvalidDeflate(
            "end-of-block 符号が定義されていない",
        ));
    }

    let (lit_lengths, dist_lengths) = lengths.split_at(hlit);
    let (lit, lit_left) = construct(lit_lengths)?;
    if lit_left != 0 && (lit_left < 0 || hlit != lit.counts[0] as usize + lit.counts[1] as usize) {
        // incomplete が許容されるのは「長さ 1 の符号が 1 個だけ」の
        // 退化ケースのみ（puff.c と同じ契約）。
        return Err(NpyError::InvalidDeflate("literal/length 符号が不完全"));
    }
    let (dist, dist_left) = construct(dist_lengths)?;
    if dist_left != 0
        && (dist_left < 0 || hdist != dist.counts[0] as usize + dist.counts[1] as usize)
    {
        return Err(NpyError::InvalidDeflate("distance 符号が不完全"));
    }

    Ok((lit, dist))
}

/// 1 ブロック分のシンボル列を復号し `out` へ書き出す。`expected_len` を
/// 超える書き込みは即時エラーにする（伸長爆弾対策）。
fn decode_block(
    reader: &mut BitReader<'_>,
    lit: &Huffman,
    dist: &Huffman,
    out: &mut Vec<u8>,
    expected_len: usize,
) -> Result<(), NpyError> {
    loop {
        let symbol = decode(reader, lit)?;
        if symbol < 256 {
            if out.len() >= expected_len {
                return Err(NpyError::InvalidDeflate("伸長後サイズが宣言値を超える"));
            }
            out.push(symbol as u8);
        } else if symbol == 256 {
            return Ok(());
        } else {
            let idx = (symbol - 257) as usize;
            if idx >= LENGTH_BASE.len() {
                return Err(NpyError::InvalidDeflate("長さ符号が範囲外（予約シンボル）"));
            }
            let len =
                LENGTH_BASE[idx] as usize + reader.get_bits(LENGTH_EXTRA[idx] as u32)? as usize;
            let dist_symbol = decode(reader, dist)? as usize;
            if dist_symbol >= DIST_BASE.len() {
                return Err(NpyError::InvalidDeflate("距離符号が範囲外（予約シンボル）"));
            }
            let distance = DIST_BASE[dist_symbol] as usize
                + reader.get_bits(DIST_EXTRA[dist_symbol] as u32)? as usize;
            if distance == 0 || distance > out.len() {
                return Err(NpyError::InvalidDeflate("距離が出力範囲を超える"));
            }
            if out.len() + len > expected_len {
                return Err(NpyError::InvalidDeflate("伸長後サイズが宣言値を超える"));
            }
            let start = out.len() - distance;
            for i in 0..len {
                // 重複区間（distance < len）を許すため 1 バイトずつ複写する。
                let byte = out[start + i];
                out.push(byte);
            }
        }
    }
}

/// `data`（raw DEFLATE ストリーム）を `expected_len` バイトちょうどまで
/// 伸長する。宣言サイズと異なる場合、不正な符号・距離・ブロック型を
/// 検出した場合はすべて `NpyError::InvalidDeflate` を返す。
///
/// 伸長爆弾対策として、確保前に 2 段の検査を行う（`.claude/rules/
/// security.md` A04/A05）:
/// 1. `expected_len` の絶対上限検査（[`MAX_INFLATE_OUTPUT_BYTES`]）。
///    呼び出し元の宣言値検査（`io::npz::MAX_MEMBER_DECOMPRESSED_BYTES`）
///    に依存しない独立の防御線
/// 2. `expected_len` が `data.len()` から導かれる理論上の圧縮比上限
///    （[`MAX_COMPRESSION_RATIO`]）を超えないかの検査。圧縮入力自体が
///    小さければ、1. の絶対上限を下回っていても不合理な伸長率は拒否する
///
/// ストリーム終端検査（PR #2318 レビュー指摘・P2）: `BFINAL` ブロックを
/// 読み終えた時点でビット位置をバイト境界へ切り上げた消費バイト数が
/// `data.len()`（ZIP の `compressed_size`＝raw DEFLATE ストリーム長。
/// `io::npz::read_member_bytes` が central directory の宣言値ちょうどに
/// 切り出して渡す）と一致することを要求する。出力長のみを照合すると、
/// 宣言された圧縮領域の末尾に付け足された任意の余剰バイト（CRC は伸長後
/// データのみが対象のため検出できない）を黙って受理してしまう。
pub(crate) fn inflate(data: &[u8], expected_len: usize) -> Result<Vec<u8>, NpyError> {
    if expected_len as u64 > MAX_INFLATE_OUTPUT_BYTES {
        return Err(NpyError::InvalidDeflate(
            "宣言された伸長後サイズが絶対上限を超える",
        ));
    }
    let max_plausible = (data.len() as u64)
        .saturating_mul(MAX_COMPRESSION_RATIO)
        .saturating_add(1024);
    if expected_len as u64 > max_plausible {
        return Err(NpyError::InvalidDeflate(
            "宣言された伸長後サイズが圧縮入力に対して理論上あり得ない大きさ",
        ));
    }

    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(expected_len)
        .map_err(|_| NpyError::InvalidDeflate("伸長後バッファの確保に失敗した"))?;

    let mut reader = BitReader::new(data);
    loop {
        let is_final = reader.get_bit()? != 0;
        let block_type = reader.get_bits(2)?;
        match block_type {
            0 => {
                // stored（無圧縮）ブロック: バイト境界へ揃えてから
                // LEN（u16）・NLEN（1 の補数）・実データの順に読む。
                reader.align_to_byte();
                let len_bytes = reader.read_bytes(4)?;
                let len = u16::from_le_bytes([len_bytes[0], len_bytes[1]]);
                let nlen = u16::from_le_bytes([len_bytes[2], len_bytes[3]]);
                if len != !nlen {
                    return Err(NpyError::InvalidDeflate(
                        "stored ブロックの LEN/NLEN が不整合",
                    ));
                }
                let payload = reader.read_bytes(len as usize)?;
                if out.len() + payload.len() > expected_len {
                    return Err(NpyError::InvalidDeflate("伸長後サイズが宣言値を超える"));
                }
                out.extend_from_slice(payload);
            }
            1 => {
                let (lit, dist) = fixed_tables()?;
                decode_block(&mut reader, &lit, &dist, &mut out, expected_len)?;
            }
            2 => {
                let (lit, dist) = dynamic_tables(&mut reader)?;
                decode_block(&mut reader, &lit, &dist, &mut out, expected_len)?;
            }
            _ => return Err(NpyError::InvalidDeflate("予約済みブロック型（3）")),
        }
        if is_final {
            break;
        }
    }

    if out.len() != expected_len {
        return Err(NpyError::InvalidDeflate("伸長後サイズが宣言値と一致しない"));
    }
    // ストリーム終端検査（P2）: `BFINAL` を読み終えた消費バイト数が
    // 入力全体と一致することを要求する。パディングビット自体の値は
    // RFC 1951 上ゼロが保証されないため検査しない（`consumed_bytes` の
    // doc comment参照）。宣言済み圧縮領域の末尾に余剰バイトを付加した
    // 細工ストリームを拒否する。
    if reader.consumed_bytes() != data.len() {
        return Err(NpyError::InvalidDeflate(
            "BFINAL ブロック後に入力の余剰バイトが残っている",
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// zlib/DEFLATE の stored ブロックを手組みして往復を確認する。
    #[test]
    fn stored_block_roundtrip() {
        let payload = b"hello, npz!";
        let len = payload.len() as u16;
        let nlen = !len;
        let mut bits = BitAccumulator::new();
        bits.push_bit(1); // final
        bits.push_bits(0, 2); // type 0 = stored
        bits.align_to_byte();
        bits.push_bytes(&len.to_le_bytes());
        bits.push_bytes(&nlen.to_le_bytes());
        bits.push_bytes(payload);
        let out = inflate(&bits.finish(), payload.len()).unwrap();
        assert_eq!(out, payload);
    }

    /// P2（PR #2318 レビュー指摘）: `BFINAL` ブロックを読み終えた後に
    /// 入力へ余剰バイトを付け足しても、出力長の一致だけでは検出できない
    /// （CRC は伸長後データのみが対象のため）。ストリーム終端検査
    /// （消費バイト数と入力長の一致）で拒否されることを確認する。
    #[test]
    fn rejects_trailing_bytes_after_bfinal() {
        let payload = b"hello, npz!";
        let len = payload.len() as u16;
        let nlen = !len;
        let mut bits = BitAccumulator::new();
        bits.push_bit(1); // final
        bits.push_bits(0, 2); // type 0 = stored
        bits.align_to_byte();
        bits.push_bytes(&len.to_le_bytes());
        bits.push_bytes(&nlen.to_le_bytes());
        bits.push_bytes(payload);
        let mut data = bits.finish();
        data.push(0xff); // 宣言された圧縮領域の末尾に余剰バイトを付加
        let err = inflate(&data, payload.len());
        assert!(
            matches!(err, Err(NpyError::InvalidDeflate(_))),
            "余剰バイト付き入力が拒否されなかった: {err:?}"
        );
    }

    /// 出力上限（`expected_len`）を超える stored ブロックは拒否する。
    #[test]
    fn stored_block_rejects_oversized_output() {
        let payload = b"0123456789";
        let len = payload.len() as u16;
        let nlen = !len;
        let mut bits = BitAccumulator::new();
        bits.push_bit(1);
        bits.push_bits(0, 2);
        bits.align_to_byte();
        bits.push_bytes(&len.to_le_bytes());
        bits.push_bytes(&nlen.to_le_bytes());
        bits.push_bytes(payload);
        let err = inflate(&bits.finish(), 5);
        assert!(matches!(err, Err(NpyError::InvalidDeflate(_))));
    }

    #[test]
    fn rejects_reserved_block_type() {
        let mut bits = BitAccumulator::new();
        bits.push_bit(1);
        bits.push_bits(3, 2); // block type 3 = reserved
        let err = inflate(&bits.finish(), 0);
        assert!(matches!(err, Err(NpyError::InvalidDeflate(_))));
    }

    #[test]
    fn rejects_truncated_input() {
        let mut bits = BitAccumulator::new();
        bits.push_bit(1);
        bits.push_bits(1, 2); // fixed huffman、本体データなし
        let err = inflate(&bits.finish(), 100);
        assert!(matches!(err, Err(NpyError::InvalidDeflate(_))));
    }

    /// fixed Huffman ブロックで既知の python `zlib.compressobj(wbits=-15)`
    /// 生 DEFLATE 出力（"aaaa" 4 バイト、fixed 符号）を伸長できることを
    /// 確認する（バイト列は `zlib` の実出力から採取・固定）。
    #[test]
    fn fixed_huffman_block_from_known_vector() {
        // python: zlib.compressobj(9, zlib.DEFLATED, -15).compress(b"aaaa") +
        //         .flush() -> bytes([75, 76, 76, 76, 4, 0])（fixed Huffman）
        let raw = [0x4b, 0x4c, 0x4c, 0x4c, 0x04, 0x00];
        let out = inflate(&raw, 4).unwrap();
        assert_eq!(out, b"aaaa");
    }

    /// dynamic Huffman ブロックの既知ベクタ（python
    /// `zlib.compressobj(6, zlib.DEFLATED, -15)` で非対称頻度データを
    /// 圧縮した実出力）を伸長できることを確認する。
    #[test]
    fn dynamic_huffman_block_from_known_vector() {
        // python:
        //   data = (b"a" * 50 + b"b" * 20 + b"c" * 5 + b"abcabcabc")
        //   zlib.compressobj(6, zlib.DEFLATED, -15).compress(data) + flush()
        let raw: &[u8] = &[75, 76, 36, 21, 36, 97, 1, 201, 32, 144, 152, 4, 69, 0];
        let expected = {
            let mut v = vec![b'a'; 50];
            v.extend(std::iter::repeat_n(b'b', 20));
            v.extend(std::iter::repeat_n(b'c', 5));
            v.extend_from_slice(b"abcabcabc");
            v
        };
        let out = inflate(raw, expected.len()).unwrap();
        assert_eq!(out, expected);
    }

    /// テスト専用のビット組み立てヘルパ（LSB ファースト、DEFLATE と同じ
    /// ビット順）。
    struct BitAccumulator {
        bytes: Vec<u8>,
        bit_pos: u32,
    }
    impl BitAccumulator {
        fn new() -> Self {
            BitAccumulator {
                bytes: vec![0],
                bit_pos: 0,
            }
        }
        fn push_bit(&mut self, bit: u32) {
            if self.bit_pos == 8 {
                self.bytes.push(0);
                self.bit_pos = 0;
            }
            let last = self.bytes.last_mut().unwrap();
            *last |= ((bit & 1) as u8) << self.bit_pos;
            self.bit_pos += 1;
        }
        fn push_bits(&mut self, value: u32, n: u32) {
            for i in 0..n {
                self.push_bit((value >> i) & 1);
            }
        }
        fn align_to_byte(&mut self) {
            // 現在の部分バイトはそのまま（上位ビットは初期値 0 のまま）
            // 確定させ、`bit_pos = 8` にして「満杯」を装う。こうすると
            // 次回の `push_bit` が新しいバイトを 1 つ push する（push_bit
            // の `bit_pos == 8` 分岐）ため、余分な空バイトを挿入せずに
            // バイト境界へ揃えられる。
            if self.bit_pos != 0 {
                self.bit_pos = 8;
            }
        }
        fn push_bytes(&mut self, bytes: &[u8]) {
            self.align_to_byte();
            self.bytes.extend_from_slice(bytes);
        }
        fn finish(self) -> Vec<u8> {
            self.bytes
        }
    }
}
