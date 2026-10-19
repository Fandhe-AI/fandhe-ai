//! 長さ前置きフレームのワイヤ形式（プロトコル版 1。すべて little-endian）。
//!
//! 役割: Hello・ランク表・データセグメントの符号化と、受信時の検証。`Read`／`Write` に対して
//! 汎用で、ソケットにも期限にも依存しない（期限は `transport::DeadlineStream` が与える）。
//! `rendezvous.rs`（Hello・ランク表）と後続 #3145（セグメント）が呼ぶ。
//! 根拠: `docs/facade-distributed-host-collective-decision.md` §4・§7（spec `:386-393`）。
//!
//! 受信側の不変条件（OWASP A03）: 本体長は**本体を読む前に**期待長との完全一致と絶対上限の
//! 両方で検証し、ワイヤ上の長さで `Vec` を確保しない。確保するのはスタック配列と呼び出し側の
//! バッファだけである。種別ごとの期待長はヘッダの kind と設定だけから決まる。
//!
//! #3145 への引き継ぎ: ヘッダ配置・kind 番号・セグメントヘッダの項目・上限値。変えるなら
//! [`PROTOCOL_VERSION`] を上げる。汎用 RPC は持たず、未知の kind は拒否する。

use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use super::error::{DistributedError, ProtocolViolation, Result};

/// フレーム先頭の magic（ASCII `"FDDP"` 相当）。
pub const MAGIC: u32 = 0x5044_4446;
/// プロトコル版。
pub const PROTOCOL_VERSION: u16 = 1;
/// フレームヘッダ長（バイト）。
pub const HEADER_LEN: usize = 12;
/// world_size の上限（記録 §3）。
pub const MAX_WORLD_SIZE: usize = 64;
/// 1 セグメントのペイロード上限（バイト。記録 §4・§7）。
pub const MAX_SEGMENT_BYTES: usize = 4 * 1024 * 1024;
/// Hello 本体長（固定）。
pub const HELLO_BODY_LEN: usize = 24;
/// ランク表 1 エントリの長さ。
pub const RANK_ENTRY_LEN: usize = 20;
/// ランク表本体の絶対上限（`4 + MAX_WORLD_SIZE * 20`）。
pub const MAX_RANK_TABLE_BODY_LEN: usize = 4 + MAX_WORLD_SIZE * RANK_ENTRY_LEN;
/// セグメントヘッダ長（本体の先頭）。
pub const SEGMENT_HEADER_LEN: usize = 32;

/// ペイロードを `f32` へ変換する際のスタック一時領域（バイト）。
const STAGE_BYTES: usize = 4096;

/// フレーム種別（固定集合）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameKind {
    /// ランク発見・リング確立の自己紹介。
    Hello = 1,
    /// ランク 0 が配るリング用アドレス表。
    RankTable = 2,
    /// reduce-scatter のセグメント（#3145）。
    ReduceScatterSegment = 3,
    /// all-gather のセグメント（#3145）。
    AllGatherSegment = 4,
}

impl FrameKind {
    fn from_u8(v: u8) -> Result<Self> {
        match v {
            1 => Ok(Self::Hello),
            2 => Ok(Self::RankTable),
            3 => Ok(Self::ReduceScatterSegment),
            4 => Ok(Self::AllGatherSegment),
            got => Err(ProtocolViolation::UnexpectedKind { got }.into()),
        }
    }
}

/// 検証済みのフレームヘッダ。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// フレーム種別。
    pub kind: FrameKind,
    /// 本体長（バイト。申告値で未検証。種別ごとの検証は各 `read_*` が行う）。
    pub body_len: u32,
}

/// Hello の内容。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hello {
    /// 送信側の world_size。
    pub world_size: u32,
    /// 送信側のランク。
    pub rank: u32,
    /// セッション識別子（混線防止用。認証ではない）。
    pub session_id: u64,
    /// 送信側のリング用待受ポート。
    pub ring_port: u16,
}

/// ランク表（添字がランク）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankTable {
    /// 各ランクのリング用アドレス。
    pub entries: Vec<SocketAddr>,
}

/// データセグメントのヘッダ（送信側）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentHeader {
    /// ステップ連番。
    pub step_seq: u64,
    /// 総要素数。
    pub total_elems: u64,
    /// チャンク番号。
    pub chunk: u32,
    /// チャンク内のセグメントのオフセット（要素数）。
    pub seg_offset: u64,
}

/// データセグメントの受信時に期待する値。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentExpect {
    /// 期待する種別（`ReduceScatterSegment` か `AllGatherSegment`）。
    pub kind: FrameKind,
    /// 期待するステップ連番。
    pub step_seq: u64,
    /// 期待する総要素数。
    pub total_elems: u64,
    /// 期待するチャンク番号。
    pub chunk: u32,
    /// 期待するセグメントのオフセット。
    pub seg_offset: u64,
}

fn put_header(buf: &mut [u8], kind: FrameKind, body_len: u32) {
    buf[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    buf[4..6].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    buf[6] = kind as u8;
    buf[7] = 0;
    buf[8..12].copy_from_slice(&body_len.to_le_bytes());
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(&b[o..o + 4]);
    u32::from_le_bytes(a)
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

/// ヘッダを読み、magic・版・reserved・kind を検証する。
pub fn read_header<R: Read>(r: &mut R) -> Result<FrameHeader> {
    let mut b = [0u8; HEADER_LEN];
    r.read_exact(&mut b)?;
    if u32_at(&b, 0) != MAGIC {
        return Err(ProtocolViolation::BadMagic.into());
    }
    let version = u16_at(&b, 4);
    if version != PROTOCOL_VERSION {
        return Err(ProtocolViolation::UnsupportedVersion { got: version }.into());
    }
    if b[7] != 0 {
        return Err(ProtocolViolation::ReservedNonZero.into());
    }
    Ok(FrameHeader {
        kind: FrameKind::from_u8(b[6])?,
        body_len: u32_at(&b, 8),
    })
}

fn expect_kind(h: &FrameHeader, kind: FrameKind) -> Result<()> {
    if h.kind == kind {
        Ok(())
    } else {
        Err(ProtocolViolation::UnexpectedKind { got: h.kind as u8 }.into())
    }
}

/// Hello を送る。
pub fn write_hello<W: Write>(w: &mut W, hello: &Hello) -> Result<()> {
    let mut b = [0u8; HEADER_LEN + HELLO_BODY_LEN];
    put_header(&mut b, FrameKind::Hello, HELLO_BODY_LEN as u32);
    b[12..16].copy_from_slice(&hello.world_size.to_le_bytes());
    b[16..20].copy_from_slice(&hello.rank.to_le_bytes());
    b[20..28].copy_from_slice(&hello.session_id.to_le_bytes());
    b[28..30].copy_from_slice(&hello.ring_port.to_le_bytes());
    w.write_all(&b)?;
    w.flush()?;
    Ok(())
}

/// Hello の本体を読む（`read_header` 済み）。本体長は 24 の完全一致を先に検証する。
pub fn read_hello<R: Read>(r: &mut R, h: &FrameHeader) -> Result<Hello> {
    expect_kind(h, FrameKind::Hello)?;
    if h.body_len as usize != HELLO_BODY_LEN {
        return Err(ProtocolViolation::LengthMismatch {
            expected: HELLO_BODY_LEN as u64,
            got: u64::from(h.body_len),
        }
        .into());
    }
    let mut b = [0u8; HELLO_BODY_LEN];
    r.read_exact(&mut b)?;
    if b[18..24].iter().any(|&x| x != 0) {
        return Err(ProtocolViolation::ReservedNonZero.into());
    }
    Ok(Hello {
        world_size: u32_at(&b, 0),
        rank: u32_at(&b, 4),
        session_id: u64_at(&b, 8),
        ring_port: u16_at(&b, 16),
    })
}

/// ランク表を送る。エントリ数は `MAX_WORLD_SIZE` 以下であること。
pub fn write_rank_table<W: Write>(w: &mut W, table: &RankTable) -> Result<()> {
    let n = table.entries.len();
    if n == 0 || n > MAX_WORLD_SIZE {
        return Err(ProtocolViolation::MalformedRankTable.into());
    }
    let body_len = 4 + n * RANK_ENTRY_LEN;
    let mut b = [0u8; HEADER_LEN + MAX_RANK_TABLE_BODY_LEN];
    put_header(&mut b, FrameKind::RankTable, body_len as u32);
    b[12..16].copy_from_slice(&(n as u32).to_le_bytes());
    for (i, a) in table.entries.iter().enumerate() {
        let o = 16 + i * RANK_ENTRY_LEN;
        match a.ip() {
            IpAddr::V4(v4) => {
                b[o] = 4;
                b[o + 4..o + 8].copy_from_slice(&v4.octets());
            }
            IpAddr::V6(v6) => {
                b[o] = 6;
                b[o + 4..o + 20].copy_from_slice(&v6.octets());
            }
        }
        b[o + 2..o + 4].copy_from_slice(&a.port().to_le_bytes());
    }
    w.write_all(&b[..HEADER_LEN + body_len])?;
    w.flush()?;
    Ok(())
}

/// ランク表の本体を読む（`read_header` 済み）。長さは本体を読む前に上限と
/// `4 + world_size * 20` の完全一致で検証し、`count == world_size`・family・reserved・port を検証する。
pub fn read_rank_table<R: Read>(
    r: &mut R,
    h: &FrameHeader,
    world_size: usize,
) -> Result<RankTable> {
    expect_kind(h, FrameKind::RankTable)?;
    let len = h.body_len as usize;
    if len > MAX_RANK_TABLE_BODY_LEN {
        return Err(ProtocolViolation::LengthTooLarge {
            len: u64::from(h.body_len),
            max: MAX_RANK_TABLE_BODY_LEN as u64,
        }
        .into());
    }
    let expected = world_size
        .checked_mul(RANK_ENTRY_LEN)
        .and_then(|x| x.checked_add(4))
        .ok_or(DistributedError::WorldSizeOutOfRange {
            world_size,
            max: MAX_WORLD_SIZE,
        })?;
    if len != expected {
        return Err(ProtocolViolation::LengthMismatch {
            expected: expected as u64,
            got: u64::from(h.body_len),
        }
        .into());
    }
    let mut b = [0u8; MAX_RANK_TABLE_BODY_LEN];
    r.read_exact(&mut b[..len])?;
    if u32_at(&b, 0) as usize != world_size {
        return Err(ProtocolViolation::MalformedRankTable.into());
    }
    let mut entries = Vec::with_capacity(world_size);
    for i in 0..world_size {
        let e = &b[4 + i * RANK_ENTRY_LEN..4 + (i + 1) * RANK_ENTRY_LEN];
        let port = u16_at(e, 2);
        if e[1] != 0 || port == 0 {
            return Err(ProtocolViolation::MalformedRankTable.into());
        }
        let ip = match e[0] {
            4 => {
                if e[8..20].iter().any(|&x| x != 0) {
                    return Err(ProtocolViolation::MalformedRankTable.into());
                }
                IpAddr::V4(Ipv4Addr::new(e[4], e[5], e[6], e[7]))
            }
            6 => {
                let mut o = [0u8; 16];
                o.copy_from_slice(&e[4..20]);
                IpAddr::V6(Ipv6Addr::from(o))
            }
            _ => return Err(ProtocolViolation::MalformedRankTable.into()),
        };
        entries.push(SocketAddr::new(ip, port));
    }
    Ok(RankTable { entries })
}

fn segment_kind_ok(kind: FrameKind) -> bool {
    matches!(
        kind,
        FrameKind::ReduceScatterSegment | FrameKind::AllGatherSegment
    )
}

/// データセグメントを送る。自分のセグメント長も受信側と同じ上限で検証し、超えるものは送らない。
pub fn send_segment<W: Write>(
    w: &mut W,
    kind: FrameKind,
    hdr: &SegmentHeader,
    data: &[f32],
) -> Result<()> {
    if !segment_kind_ok(kind) {
        return Err(ProtocolViolation::UnexpectedKind { got: kind as u8 }.into());
    }
    let payload = data
        .len()
        .checked_mul(4)
        .filter(|&p| p <= MAX_SEGMENT_BYTES)
        .ok_or(ProtocolViolation::LengthTooLarge {
            len: (data.len() as u64).saturating_mul(4),
            max: MAX_SEGMENT_BYTES as u64,
        })?;
    let mut b = [0u8; HEADER_LEN + SEGMENT_HEADER_LEN];
    put_header(&mut b, kind, (SEGMENT_HEADER_LEN + payload) as u32);
    b[12..20].copy_from_slice(&hdr.step_seq.to_le_bytes());
    b[20..28].copy_from_slice(&hdr.total_elems.to_le_bytes());
    b[28..32].copy_from_slice(&hdr.chunk.to_le_bytes());
    b[32..40].copy_from_slice(&hdr.seg_offset.to_le_bytes());
    b[40..44].copy_from_slice(&(data.len() as u32).to_le_bytes());
    w.write_all(&b)?;
    let mut stage = [0u8; STAGE_BYTES];
    for part in data.chunks(STAGE_BYTES / 4) {
        for (i, v) in part.iter().enumerate() {
            stage[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        w.write_all(&stage[..part.len() * 4])?;
    }
    w.flush()?;
    Ok(())
}

/// データセグメントを受け、`dst` へ直接書き込む。
///
/// 本体を読む前に、kind・本体長の上限・セグメントヘッダ（連番・総要素数・チャンク・オフセット）・
/// `seg_len == dst.len()`・`seg_len * 4` の上限・本体長の完全一致をすべて検証する。`Vec` は確保しない。
/// エラー時の `dst` の中身は不定（呼び出し側 #3145 が communicator を poisoned にする前提）。
pub fn recv_segment_into<R: Read>(
    r: &mut R,
    expect: &SegmentExpect,
    dst: &mut [f32],
) -> Result<()> {
    if !segment_kind_ok(expect.kind) {
        return Err(ProtocolViolation::UnexpectedKind {
            got: expect.kind as u8,
        }
        .into());
    }
    let h = read_header(r)?;
    expect_kind(&h, expect.kind)?;
    let max_body = SEGMENT_HEADER_LEN + MAX_SEGMENT_BYTES;
    let body_len = h.body_len as usize;
    if body_len > max_body {
        return Err(ProtocolViolation::LengthTooLarge {
            len: u64::from(h.body_len),
            max: max_body as u64,
        }
        .into());
    }
    if body_len < SEGMENT_HEADER_LEN {
        return Err(ProtocolViolation::LengthMismatch {
            expected: SEGMENT_HEADER_LEN as u64,
            got: u64::from(h.body_len),
        }
        .into());
    }
    let mut sh = [0u8; SEGMENT_HEADER_LEN];
    r.read_exact(&mut sh)?;
    if u64_at(&sh, 0) != expect.step_seq {
        return Err(ProtocolViolation::SequenceMismatch.into());
    }
    if u64_at(&sh, 8) != expect.total_elems {
        return Err(ProtocolViolation::TotalElementsMismatch.into());
    }
    if u32_at(&sh, 16) != expect.chunk {
        return Err(ProtocolViolation::ChunkMismatch.into());
    }
    if u64_at(&sh, 20) != expect.seg_offset {
        return Err(ProtocolViolation::OffsetMismatch.into());
    }
    let seg_len = u32_at(&sh, 28) as usize;
    let seg_bytes = seg_len
        .checked_mul(4)
        .filter(|&p| p <= MAX_SEGMENT_BYTES)
        .ok_or(ProtocolViolation::LengthTooLarge {
            len: (seg_len as u64).saturating_mul(4),
            max: MAX_SEGMENT_BYTES as u64,
        })?;
    if seg_len != dst.len() || body_len != SEGMENT_HEADER_LEN + seg_bytes {
        return Err(ProtocolViolation::LengthMismatch {
            expected: (dst.len() as u64).saturating_mul(4) + SEGMENT_HEADER_LEN as u64,
            got: u64::from(h.body_len),
        }
        .into());
    }
    let mut stage = [0u8; STAGE_BYTES];
    for part in dst.chunks_mut(STAGE_BYTES / 4) {
        let n = part.len() * 4;
        r.read_exact(&mut stage[..n])?;
        for (i, v) in part.iter_mut().enumerate() {
            *v = f32::from_le_bytes([
                stage[i * 4],
                stage[i * 4 + 1],
                stage[i * 4 + 2],
                stage[i * 4 + 3],
            ]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn hello() -> Hello {
        Hello {
            world_size: 4,
            rank: 2,
            session_id: 0xdead_beef_1234,
            ring_port: 4321,
        }
    }

    fn encode_hello() -> Vec<u8> {
        let mut v = Vec::new();
        write_hello(&mut v, &hello()).unwrap();
        v
    }

    fn table(n: usize) -> RankTable {
        RankTable {
            entries: (0..n)
                .map(|i| {
                    if i % 2 == 0 {
                        SocketAddr::new(Ipv4Addr::new(127, 0, 0, 1).into(), 1000 + i as u16)
                    } else {
                        SocketAddr::new(Ipv6Addr::LOCALHOST.into(), 1000 + i as u16)
                    }
                })
                .collect(),
        }
    }

    fn encode_table(n: usize) -> Vec<u8> {
        let mut v = Vec::new();
        write_rank_table(&mut v, &table(n)).unwrap();
        v
    }

    fn seg_hdr() -> SegmentHeader {
        SegmentHeader {
            step_seq: 7,
            total_elems: 100,
            chunk: 3,
            seg_offset: 25,
        }
    }

    fn seg_expect() -> SegmentExpect {
        SegmentExpect {
            kind: FrameKind::ReduceScatterSegment,
            step_seq: 7,
            total_elems: 100,
            chunk: 3,
            seg_offset: 25,
        }
    }

    fn encode_seg(data: &[f32]) -> Vec<u8> {
        let mut v = Vec::new();
        send_segment(&mut v, FrameKind::ReduceScatterSegment, &seg_hdr(), data).unwrap();
        v
    }

    fn recv(bytes: &[u8], dst: &mut [f32]) -> (Result<()>, u64) {
        let mut c = Cursor::new(bytes.to_vec());
        let r = recv_segment_into(&mut c, &seg_expect(), dst);
        (r, c.position())
    }

    fn proto(r: Result<impl std::fmt::Debug>) -> ProtocolViolation {
        match r {
            Err(DistributedError::Protocol(v)) => v,
            other => panic!("protocol violation expected, got {other:?}"),
        }
    }

    #[test]
    fn hello_round_trip() {
        let bytes = encode_hello();
        let mut c = Cursor::new(bytes);
        let h = read_header(&mut c).unwrap();
        assert_eq!(read_hello(&mut c, &h).unwrap(), hello());
    }

    #[test]
    fn rank_table_round_trip() {
        for n in [1, 2, 5, 64] {
            let mut c = Cursor::new(encode_table(n));
            let h = read_header(&mut c).unwrap();
            assert_eq!(read_rank_table(&mut c, &h, n).unwrap(), table(n));
        }
    }

    #[test]
    fn segment_round_trip_bit_exact() {
        let data = [
            1.5f32,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::MIN_POSITIVE,
            3.0,
        ];
        let mut dst = [0.0f32; 6];
        let (r, pos) = recv(&encode_seg(&data), &mut dst);
        r.unwrap();
        assert_eq!(pos as usize, HEADER_LEN + SEGMENT_HEADER_LEN + 24);
        for (a, b) in data.iter().zip(dst.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        // 1 度のステージ（4 KiB）を超える長さでも往復できる。
        let big: Vec<f32> = (0..3000).map(|i| i as f32 * 0.25).collect();
        let mut dst = vec![0.0f32; 3000];
        recv(&encode_seg(&big), &mut dst).0.unwrap();
        assert_eq!(big, dst);
        // 空セグメント。
        recv(&encode_seg(&[]), &mut []).0.unwrap();
    }

    #[test]
    fn rank_table_length_limits_reject_before_reading_body() {
        // 確保が起きない立証: 受信 API は呼び出し側のバッファとスタック配列にしか書かず、ワイヤ上の
        // 長さで確保しない。ここでは拒否時にヘッダ以降を 1 バイトも読んでいないことを確かめる。
        for bad in [u32::MAX, (MAX_RANK_TABLE_BODY_LEN + 1) as u32] {
            let mut b = encode_table(2);
            b[8..12].copy_from_slice(&bad.to_le_bytes());
            let mut c = Cursor::new(b);
            let h = read_header(&mut c).unwrap();
            let e = proto(read_rank_table(&mut c, &h, 2));
            assert!(matches!(e, ProtocolViolation::LengthTooLarge { .. }));
            assert_eq!(c.position() as usize, HEADER_LEN);
        }
        let mut b = encode_table(2);
        b[8..12].copy_from_slice(&((4 + 3 * RANK_ENTRY_LEN) as u32).to_le_bytes());
        let mut c = Cursor::new(b);
        let h = read_header(&mut c).unwrap();
        let e = proto(read_rank_table(&mut c, &h, 2));
        assert!(matches!(e, ProtocolViolation::LengthMismatch { .. }));
        assert_eq!(c.position() as usize, HEADER_LEN);
    }

    #[test]
    fn hello_length_mismatch_rejected() {
        let mut b = encode_hello();
        b[8..12].copy_from_slice(&(u32::MAX).to_le_bytes());
        let mut c = Cursor::new(b);
        let h = read_header(&mut c).unwrap();
        let e = proto(read_hello(&mut c, &h));
        assert!(matches!(e, ProtocolViolation::LengthMismatch { .. }));
        assert_eq!(c.position() as usize, HEADER_LEN);
    }

    #[test]
    fn segment_length_limits_reject_before_payload() {
        let hdr_and_seg = HEADER_LEN + SEGMENT_HEADER_LEN;
        let mut dst = [0.0f32; 4];
        // body_len が絶対上限超過。
        let mut b = encode_seg(&[0.0; 4]);
        b[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        let (r, pos) = recv(&b, &mut dst);
        assert!(matches!(proto(r), ProtocolViolation::LengthTooLarge { .. }));
        assert_eq!(pos as usize, HEADER_LEN);
        // seg_len が u32::MAX。
        let mut b = encode_seg(&[0.0; 4]);
        b[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
        let (r, pos) = recv(&b, &mut dst);
        assert!(matches!(proto(r), ProtocolViolation::LengthTooLarge { .. }));
        assert_eq!(pos as usize, hdr_and_seg);
        // seg_len が dst と不一致。
        let (r, pos) = recv(&encode_seg(&[0.0; 3]), &mut dst);
        assert!(matches!(proto(r), ProtocolViolation::LengthMismatch { .. }));
        assert_eq!(pos as usize, hdr_and_seg);
        // body_len が seg_len と整合しない。
        let mut b = encode_seg(&[0.0; 4]);
        b[8..12].copy_from_slice(&((SEGMENT_HEADER_LEN + 8) as u32).to_le_bytes());
        let (r, pos) = recv(&b, &mut dst);
        assert!(matches!(proto(r), ProtocolViolation::LengthMismatch { .. }));
        assert_eq!(pos as usize, hdr_and_seg);
        // 本体長がセグメントヘッダ未満。
        let mut b = encode_seg(&[]);
        b[8..12].copy_from_slice(&4u32.to_le_bytes());
        let (r, pos) = recv(&b, &mut dst);
        assert!(matches!(proto(r), ProtocolViolation::LengthMismatch { .. }));
        assert_eq!(pos as usize, HEADER_LEN);
    }

    #[test]
    fn send_rejects_oversized_segment() {
        let data = vec![0.0f32; MAX_SEGMENT_BYTES / 4 + 1];
        let mut out = Vec::new();
        let e = send_segment(&mut out, FrameKind::AllGatherSegment, &seg_hdr(), &data);
        assert!(matches!(proto(e), ProtocolViolation::LengthTooLarge { .. }));
        assert!(out.is_empty());
        // 上限ちょうどは送れる。
        let ok = vec![0.0f32; MAX_SEGMENT_BYTES / 4];
        send_segment(&mut out, FrameKind::AllGatherSegment, &seg_hdr(), &ok).unwrap();
    }

    #[test]
    fn truncated_frames_are_peer_disconnected() {
        let hello = encode_hello();
        let seg = encode_seg(&[1.0, 2.0, 3.0, 4.0]);
        // ヘッダの途中。
        assert!(matches!(
            read_header(&mut Cursor::new(hello[..5].to_vec())),
            Err(DistributedError::PeerDisconnected)
        ));
        // Hello 本体の途中。
        let mut c = Cursor::new(hello[..HEADER_LEN + 3].to_vec());
        let h = read_header(&mut c).unwrap();
        assert!(matches!(
            read_hello(&mut c, &h),
            Err(DistributedError::PeerDisconnected)
        ));
        // ペイロードの途中。
        let mut dst = [0.0f32; 4];
        let (r, _) = recv(&seg[..seg.len() - 3], &mut dst);
        assert!(matches!(r, Err(DistributedError::PeerDisconnected)));
        // セグメントヘッダの途中。
        let (r, _) = recv(&seg[..HEADER_LEN + 5], &mut dst);
        assert!(matches!(r, Err(DistributedError::PeerDisconnected)));
    }

    #[test]
    fn header_violations() {
        let base = encode_hello();
        let mut b = base.clone();
        b[0] ^= 0xff;
        assert_eq!(
            proto(read_header(&mut Cursor::new(b))),
            ProtocolViolation::BadMagic
        );
        let mut b = base.clone();
        b[4] = 9;
        assert!(matches!(
            proto(read_header(&mut Cursor::new(b))),
            ProtocolViolation::UnsupportedVersion { got: 9 }
        ));
        let mut b = base.clone();
        b[6] = 99;
        assert!(matches!(
            proto(read_header(&mut Cursor::new(b))),
            ProtocolViolation::UnexpectedKind { got: 99 }
        ));
        let mut b = base.clone();
        b[7] = 1;
        assert_eq!(
            proto(read_header(&mut Cursor::new(b))),
            ProtocolViolation::ReservedNonZero
        );
        // kind が期待と違う（Hello を RankTable として読む）。
        let mut c = Cursor::new(base.clone());
        let h = read_header(&mut c).unwrap();
        assert!(matches!(
            proto(read_rank_table(&mut c, &h, 4)),
            ProtocolViolation::UnexpectedKind { .. }
        ));
        // Hello の reserved。
        let mut b = base;
        b[HEADER_LEN + 20] = 1;
        let mut c = Cursor::new(b);
        let h = read_header(&mut c).unwrap();
        assert_eq!(
            proto(read_hello(&mut c, &h)),
            ProtocolViolation::ReservedNonZero
        );
    }

    #[test]
    fn segment_field_mismatches() {
        let seg = encode_seg(&[1.0, 2.0]);
        let mut dst = [0.0f32; 2];
        // 期待値側を変えて各項目の不一致を確認する。
        for (mutate, want) in [
            (
                (|e: &mut SegmentExpect| e.step_seq += 1) as fn(&mut SegmentExpect),
                ProtocolViolation::SequenceMismatch,
            ),
            (
                |e| e.total_elems += 1,
                ProtocolViolation::TotalElementsMismatch,
            ),
            (|e| e.chunk += 1, ProtocolViolation::ChunkMismatch),
            (|e| e.seg_offset += 1, ProtocolViolation::OffsetMismatch),
        ] {
            let mut e = seg_expect();
            mutate(&mut e);
            let r = recv_segment_into(&mut Cursor::new(seg.clone()), &e, &mut dst);
            assert_eq!(proto(r), want);
        }
        let mut e = seg_expect();
        e.kind = FrameKind::AllGatherSegment;
        let r = recv_segment_into(&mut Cursor::new(seg), &e, &mut dst);
        assert!(matches!(proto(r), ProtocolViolation::UnexpectedKind { .. }));
    }

    #[test]
    fn rank_table_content_violations() {
        let n = 2;
        let base = encode_table(n);
        let read = |b: Vec<u8>| {
            let mut c = Cursor::new(b);
            let h = read_header(&mut c).unwrap();
            read_rank_table(&mut c, &h, n)
        };
        let e0 = HEADER_LEN + 4;
        // count 不一致。
        let mut b = base.clone();
        b[HEADER_LEN] = 3;
        assert_eq!(proto(read(b)), ProtocolViolation::MalformedRankTable);
        // family 不正。
        let mut b = base.clone();
        b[e0] = 5;
        assert_eq!(proto(read(b)), ProtocolViolation::MalformedRankTable);
        // port = 0。
        let mut b = base.clone();
        b[e0 + 2] = 0;
        b[e0 + 3] = 0;
        assert_eq!(proto(read(b)), ProtocolViolation::MalformedRankTable);
        // IPv4 の余りバイト。
        let mut b = base.clone();
        b[e0 + 15] = 1;
        assert_eq!(proto(read(b)), ProtocolViolation::MalformedRankTable);
        // reserved。
        let mut b = base;
        b[e0 + 1] = 1;
        assert_eq!(proto(read(b)), ProtocolViolation::MalformedRankTable);
    }
}
