//! ring all-reduce（reduce-scatter → all-gather）。ホスト上の `f32` スライスが対象（イシュー #3145）。
//!
//! 役割: `rendezvous` が確立した [`RingLinks`] の上で、全ランクの勾配（平坦化バッファ）を合計または
//! 平均し、全ランクに bit 単位で同じ結果を返す。呼び出し元は後続の #3146（`DeviceParamStore` への
//! 結線）と #3147（facade 公開）。本モジュールは autodiff の内部 API で、サポートする公開面は facade のみ。
//! 根拠: `docs/facade-distributed-host-collective-decision.md` §4（加算順序）・§5（失敗時の poisoned）・
//! §6（数値契約）・§7（受信検証）・§14（本イシューでの追補）。spec: `docs/spec/04-requirements.md` `:386-393`。
//!
//! # 加算順序（決定論）
//!
//! チャンク `c` はランク `c, c+1, …, c+W-1 (mod W)` の順に左結合で加算する（記録 §4。チャンクごとに
//! 回転した固定順）。ランク数・要素数・入力が同じなら run 間で bit 一致する。[`reference_all_reduce`]
//! が同じ順序の逐次加算で、テストと #3148 の基準になる。
//!
//! # デッドロック回避と失敗時の挙動
//!
//! 各ステップは「次へ送る」と「前から受ける」を別スレッドで同時に行う（順番に行うと、セグメントが
//! ソケットの送信バッファを超えたとき全ランクが書き込みで止まる）。全送受信は絶対期限
//! （[`CollectiveOptions::op_timeout`]）で包むため、必ず戻る。失敗したら両ソケットを shutdown して
//! 隣へ伝え、communicator は poisoned になる（復旧・再接続は範囲外）。
//!
//! # セキュリティ上の前提
//!
//! 認証も暗号化も無い（信頼できるネットワーク限定。記録 §7）。受信長は `frame::recv_segment_into` が
//! 本体を読む前に自分の `(n, W, chunk, offset)` から決めた期待長と完全一致で検証する。

use std::io;
use std::net::TcpStream;
use std::ops::Range;
use std::sync::atomic::{AtomicU8, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use super::error::{DistributedError, Result};
use super::frame::{
    FrameKind, MAX_SEGMENT_BYTES, MAX_WORLD_SIZE, SegmentExpect, SegmentHeader, recv_segment_into,
    send_segment,
};
use super::rendezvous::RingLinks;
use super::transport::{DeadlineStream, shutdown_both};

/// 縮約の種類。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReduceOp {
    /// 合計。
    Sum,
    /// 合計を world_size で割った平均（`W >= 2` のとき所有チャンクで 1 回だけ割る）。
    Mean,
}

/// collective の期限設定。
///
/// `io_timeout` は 1 回の読み書きの上限、`op_timeout` は all-reduce 1 回の全体の絶対期限。
/// 少しずつ送るピアでも全体期限は延びない（OWASP A04）。#3147 が facade の設定型から組み立てる。
#[derive(Debug, Clone, Copy)]
pub struct CollectiveOptions {
    io_timeout: Duration,
    op_timeout: Duration,
}

impl Default for CollectiveOptions {
    /// io 30 秒（`RendezvousConfig` と同じ）・all-reduce 1 回 120 秒。
    fn default() -> Self {
        Self {
            io_timeout: Duration::from_secs(30),
            op_timeout: Duration::from_secs(120),
        }
    }
}

impl CollectiveOptions {
    /// 1 回の読み書きの上限を設定する。
    pub fn io_timeout(mut self, d: Duration) -> Self {
        self.io_timeout = d;
        self
    }

    /// all-reduce 1 回の全体期限を設定する。
    pub fn op_timeout(mut self, d: Duration) -> Self {
        self.op_timeout = d;
        self
    }

    /// 0 と `Instant` に加算できない巨大値を `InvalidConfig` で拒否する（panic しない）。
    pub fn validate(&self) -> Result<()> {
        if self.io_timeout.is_zero() || self.op_timeout.is_zero() {
            return Err(DistributedError::InvalidConfig {
                reason: "timeouts must be non-zero",
            });
        }
        let now = Instant::now();
        if now.checked_add(self.io_timeout).is_none() || now.checked_add(self.op_timeout).is_none()
        {
            return Err(DistributedError::InvalidConfig {
                reason: "timeout is too large",
            });
        }
        Ok(())
    }
}

/// `n` 要素を `world_size` 個に分ける。`base = n / W`・`rem = n % W` で先頭 `rem` 個が 1 要素多い。
/// 範囲は連続で昇順（チャンク番号順 = アドレス順）。`n < W` では末尾に空チャンクが出る。
pub fn chunk_ranges(n: usize, world_size: usize) -> Vec<Range<usize>> {
    let w = world_size.max(1);
    let base = n / w;
    let rem = n % w;
    let mut start = 0;
    (0..w)
        .map(|c| {
            let len = base + usize::from(c < rem);
            let r = start..start + len;
            start += len;
            r
        })
        .collect()
}

/// 全ランクの入力から、リングと同じ回転順序の逐次加算で結果を作る参照実装（bit 一致の基準）。
/// 入力が空、または長さがそろわなければ `InvalidConfig`。`W = 1` は入力をそのまま返す。
pub fn reference_all_reduce(inputs: &[&[f32]], op: ReduceOp) -> Result<Vec<f32>> {
    let w = inputs.len();
    if w == 0 {
        return Err(DistributedError::InvalidConfig {
            reason: "inputs must not be empty",
        });
    }
    let n = inputs[0].len();
    if inputs.iter().any(|x| x.len() != n) {
        return Err(DistributedError::InvalidConfig {
            reason: "input lengths differ",
        });
    }
    if w == 1 {
        return Ok(inputs[0].to_vec());
    }
    let mut out = vec![0.0f32; n];
    for (c, range) in chunk_ranges(n, w).into_iter().enumerate() {
        for i in range {
            let mut acc = inputs[c][i];
            for k in 1..w {
                acc += inputs[(c + k) % w][i];
            }
            if op == ReduceOp::Mean {
                acc /= w as f32;
            }
            out[i] = acc;
        }
    }
    Ok(out)
}

/// チャンク（長さ `len`）をセグメント `(オフセット, 長さ)` に分ける。空チャンクも長さ 0 のセグメントを
/// 1 個出す（ヘッダだけ送り、全ランクの連番とフレーム数をそろえる。記録 §4）。
fn segments(len: usize, max_seg: usize) -> impl Iterator<Item = (usize, usize)> {
    let count = if len == 0 { 1 } else { len.div_ceil(max_seg) };
    (0..count).map(move |i| {
        let o = i * max_seg;
        (o, (len - o).min(max_seg))
    })
}

/// 重ならない 2 つのチャンクを `(送信用の共有参照, 受信用の可変参照)` に分ける（`unsafe` 不要）。
/// チャンク番号の昇順がアドレス順なので、番号の大小で `split_at_mut` の向きを決める。
fn disjoint<'a>(
    buf: &'a mut [f32],
    ranges: &[Range<usize>],
    send_c: usize,
    recv_c: usize,
) -> (&'a [f32], &'a mut [f32]) {
    let (s, r) = (ranges[send_c].clone(), ranges[recv_c].clone());
    if send_c < recv_c {
        let (left, right) = buf.split_at_mut(r.start);
        (&left[s], &mut right[..r.len()])
    } else {
        let (left, right) = buf.split_at_mut(s.start);
        (&right[..s.len()], &mut left[r])
    }
}

/// 1 ステップの送受信に共通する文脈。
struct StepCtx<'a> {
    to_next: &'a TcpStream,
    from_prev: &'a TcpStream,
    deadline: Instant,
    io_timeout: Duration,
    step_seq: u64,
    total_elems: u64,
    max_seg: usize,
}

/// 1 ステップ: 送信スレッドで `send` を次へ送り、現在のスレッドで前から `recv` を受ける。
/// `accumulate` なら受けた部分和を左オペランドにして自分の値へ足す（reduce-scatter）。
/// そうでなければ受けた値で直接上書きする（all-gather。bit の複写）。
fn exchange(
    ctx: &StepCtx<'_>,
    kind: FrameKind,
    send: (u32, &[f32]),
    recv: (u32, &mut [f32]),
    scratch: &mut [f32],
    accumulate: bool,
) -> Result<()> {
    // 先に失敗した側の識別（1 = 送信・2 = 受信）。後から出る「相手を shutdown した結果の」
    // 二次エラーより根本原因を返すため。
    let first = AtomicU8::new(0);
    thread::scope(|sc| {
        // OS のスレッド数制限等で生成に失敗しても panic させず型付きエラーで返す（記録 §5）。
        let spawned = thread::Builder::new().spawn_scoped(sc, || {
            let mut w = DeadlineStream::new(ctx.to_next, ctx.deadline, ctx.io_timeout);
            let r = (|| -> Result<()> {
                for (off, len) in segments(send.1.len(), ctx.max_seg) {
                    let hdr = SegmentHeader {
                        step_seq: ctx.step_seq,
                        total_elems: ctx.total_elems,
                        chunk: send.0,
                        seg_offset: off as u64,
                    };
                    send_segment(&mut w, kind, &hdr, &send.1[off..off + len])?;
                }
                Ok(())
            })();
            if r.is_err() {
                let _ = first.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst);
                shutdown_both(ctx.from_prev);
            }
            r
        });
        let h = match spawned {
            Ok(h) => h,
            Err(e) => {
                // 受信側へ進まず両ソケットを閉じ、相手ランクを待たせず呼び出し元の poisoned 遷移へ流す。
                shutdown_both(ctx.to_next);
                shutdown_both(ctx.from_prev);
                return Err(DistributedError::Io(e));
            }
        };
        let (rc, dst) = recv;
        let mut rd = DeadlineStream::new(ctx.from_prev, ctx.deadline, ctx.io_timeout);
        let rr = (|| -> Result<()> {
            for (off, len) in segments(dst.len(), ctx.max_seg) {
                let expect = SegmentExpect {
                    kind,
                    step_seq: ctx.step_seq,
                    total_elems: ctx.total_elems,
                    chunk: rc,
                    seg_offset: off as u64,
                };
                let part = &mut dst[off..off + len];
                if accumulate {
                    let tmp = &mut scratch[..len];
                    recv_segment_into(&mut rd, &expect, tmp)?;
                    for (d, t) in part.iter_mut().zip(tmp.iter()) {
                        *d += *t; // f32 加算は可換のため `部分和 + 自分` と bit 一致
                    }
                } else {
                    recv_segment_into(&mut rd, &expect, part)?;
                }
            }
            Ok(())
        })();
        if rr.is_err() {
            let _ = first.compare_exchange(0, 2, Ordering::SeqCst, Ordering::SeqCst);
            shutdown_both(ctx.to_next);
        }
        let sr = h.join().unwrap_or_else(|_| {
            Err(DistributedError::Io(io::Error::other(
                "sender thread panicked",
            )))
        });
        match (sr, rr) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(e), Ok(())) | (Ok(()), Err(e)) => Err(e),
            (Err(se), Err(re)) => {
                if first.load(Ordering::SeqCst) == 1 {
                    Err(se)
                } else {
                    Err(re)
                }
            }
        }
    })
}

/// リング上の communicator。`RingLinks` を所有し、連番と poisoned 状態を持つ（記録 §5）。
#[derive(Debug)]
pub struct RingCommunicator {
    links: RingLinks,
    opts: CollectiveOptions,
    step_seq: u64,
    poisoned: bool,
    max_seg_elems: usize,
}

impl RingCommunicator {
    const DEFAULT_MAX_SEG_ELEMS: usize = MAX_SEGMENT_BYTES / 4;

    /// 期限とリングの整合を検証して communicator を作る。
    pub fn new(links: RingLinks, opts: CollectiveOptions) -> Result<Self> {
        opts.validate()?;
        if links.world_size == 0 || links.world_size > MAX_WORLD_SIZE {
            return Err(DistributedError::WorldSizeOutOfRange {
                world_size: links.world_size,
                max: MAX_WORLD_SIZE,
            });
        }
        if links.rank >= links.world_size {
            return Err(DistributedError::RankOutOfRange {
                rank: links.rank,
                world_size: links.world_size,
            });
        }
        if links.world_size >= 2 && (links.to_next.is_none() || links.from_prev.is_none()) {
            return Err(DistributedError::InvalidConfig {
                reason: "ring links are missing",
            });
        }
        Ok(Self {
            links,
            opts,
            step_seq: 0,
            poisoned: false,
            max_seg_elems: Self::DEFAULT_MAX_SEG_ELEMS,
        })
    }

    /// テスト用: セグメント長を小さくして多セグメントの経路を安価に通す（既定値が上限）。
    #[cfg(test)]
    fn with_max_seg_elems(mut self, n: usize) -> Self {
        self.max_seg_elems = n.clamp(1, Self::DEFAULT_MAX_SEG_ELEMS);
        self
    }

    /// 自ランク。
    pub fn rank(&self) -> usize {
        self.links.rank
    }

    /// world_size。
    pub fn world_size(&self) -> usize {
        self.links.world_size
    }

    /// 失敗後の poisoned 状態か。
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// `buf` を全ランクで合計（または平均）し、結果で上書きする。
    ///
    /// 契約: 全ランクが**同じ要素数・同じ `op`** で呼ぶこと（`op` はワイヤに載らないため検出しない。
    /// 要素数の不一致は型付きのプロトコル違反になる）。`Ok` なら `buf` は全ランクで bit 一致。
    /// `Err` のとき `buf` の中身は不定で、以後この communicator は `Poisoned`（呼び出し側 #3146 は
    /// `Ok` のときだけ結果を書き戻すこと）。`W = 1` は通信せず `buf` を変えない（`Mean` でも割らない）。
    pub fn all_reduce(&mut self, buf: &mut [f32], op: ReduceOp) -> Result<()> {
        if self.poisoned {
            return Err(DistributedError::Poisoned);
        }
        if self.links.world_size == 1 {
            return Ok(());
        }
        let r = self.run(buf, op);
        if r.is_err() {
            self.poisoned = true;
            for s in [&self.links.to_next, &self.links.from_prev]
                .into_iter()
                .flatten()
            {
                shutdown_both(s);
            }
        }
        r
    }

    fn run(&mut self, buf: &mut [f32], op: ReduceOp) -> Result<()> {
        let (w, r) = (self.links.world_size, self.links.rank);
        let missing = || DistributedError::InvalidConfig {
            reason: "ring links are missing",
        };
        let to_next = self.links.to_next.as_ref().ok_or_else(missing)?;
        let from_prev = self.links.from_prev.as_ref().ok_or_else(missing)?;
        let next_seq = self
            .step_seq
            .checked_add(1)
            .ok_or(DistributedError::InvalidConfig {
                reason: "step sequence exhausted",
            })?;
        let deadline = Instant::now().checked_add(self.opts.op_timeout).ok_or(
            DistributedError::InvalidConfig {
                reason: "timeout is too large",
            },
        )?;
        let n = buf.len();
        let ctx = StepCtx {
            to_next,
            from_prev,
            deadline,
            io_timeout: self.opts.io_timeout,
            step_seq: self.step_seq,
            total_elems: n as u64,
            max_seg: self.max_seg_elems,
        };
        let ranges = chunk_ranges(n, w);
        // 先頭チャンクが最長。確保量は自分の n だけで決まる（ピアの申告に左右されない）。
        let mut scratch = vec![0.0f32; ranges[0].len().min(self.max_seg_elems)];

        for s in 0..w - 1 {
            let (sc, rc) = ((r + w - s) % w, (r + 2 * w - s - 1) % w);
            let (send, recv) = disjoint(buf, &ranges, sc, rc);
            exchange(
                &ctx,
                FrameKind::ReduceScatterSegment,
                (sc as u32, send),
                (rc as u32, recv),
                &mut scratch,
                true,
            )?;
        }
        if op == ReduceOp::Mean {
            let wf = w as f32;
            for v in &mut buf[ranges[(r + 1) % w].clone()] {
                *v /= wf;
            }
        }
        for s in 0..w - 1 {
            let (sc, rc) = ((r + 1 + w - s) % w, (r + w - s) % w);
            let (send, recv) = disjoint(buf, &ranges, sc, rc);
            exchange(
                &ctx,
                FrameKind::AllGatherSegment,
                (sc as u32, send),
                (rc as u32, recv),
                &mut scratch,
                false,
            )?;
        }
        self.step_seq = next_seq;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::RendezvousConfig;
    use super::super::error::ProtocolViolation;
    use super::super::rendezvous::Rendezvous;
    use super::*;

    fn lo0() -> std::net::SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }

    fn cfg(w: usize, rank: usize, addr: std::net::SocketAddr) -> RendezvousConfig {
        RendezvousConfig::new(w, rank, addr)
            .unwrap()
            .session_id(7)
            .establish_timeout(Duration::from_secs(20))
            .io_timeout(Duration::from_secs(5))
    }

    fn establish_group(w: usize) -> Vec<RingLinks> {
        if w == 1 {
            return vec![
                Rendezvous::bind(cfg(1, 0, lo0()))
                    .unwrap()
                    .establish()
                    .unwrap(),
            ];
        }
        thread::scope(|s| {
            let r0 = Rendezvous::bind(cfg(w, 0, lo0())).unwrap();
            let addr = r0.rendezvous_local_addr().unwrap();
            let h0 = s.spawn(move || r0.establish());
            let hs: Vec<_> = (1..w)
                .map(|r| s.spawn(move || Rendezvous::bind(cfg(w, r, addr))?.establish()))
                .collect();
            let mut v = vec![h0.join().unwrap().unwrap()];
            for h in hs {
                v.push(h.join().unwrap().unwrap());
            }
            v
        })
    }

    fn comms(w: usize, opts: CollectiveOptions, max_seg: Option<usize>) -> Vec<RingCommunicator> {
        establish_group(w)
            .into_iter()
            .map(|l| {
                let c = RingCommunicator::new(l, opts).unwrap();
                match max_seg {
                    Some(m) => c.with_max_seg_elems(m),
                    None => c,
                }
            })
            .collect()
    }

    fn opts() -> CollectiveOptions {
        CollectiveOptions::default()
            .io_timeout(Duration::from_secs(5))
            .op_timeout(Duration::from_secs(30))
    }

    /// 桁の大きく違う正負の値（結合順序が結果に効く）の決定的な疑似乱数。
    fn gen_input(rank: usize, n: usize) -> Vec<f32> {
        let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ ((rank as u64 + 1) * 0x1234_5678_9ABC_DEF1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let mant = ((x >> 40) as i32 - (1 << 23)) as f32;
                let exp = ((x >> 8) % 21) as i32 - 10;
                mant * 2f32.powi(exp)
            })
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn run_all(cs: &mut [RingCommunicator], inputs: &[Vec<f32>], op: ReduceOp) -> Vec<Vec<f32>> {
        thread::scope(|s| {
            let hs: Vec<_> = cs
                .iter_mut()
                .zip(inputs)
                .map(|(c, inp)| {
                    let mut b = inp.clone();
                    s.spawn(move || {
                        c.all_reduce(&mut b, op).unwrap();
                        b
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        })
    }

    fn check(w: usize, n: usize, op: ReduceOp, max_seg: Option<usize>) {
        let inputs: Vec<Vec<f32>> = (0..w).map(|r| gen_input(r, n)).collect();
        let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let want = reference_all_reduce(&refs, op).unwrap();
        let mut cs = comms(w, opts(), max_seg);
        let outs = run_all(&mut cs, &inputs, op);
        for (r, o) in outs.iter().enumerate() {
            assert_eq!(bits(o), bits(&want), "w={w} n={n} op={op:?} rank={r}");
        }
    }

    #[test]
    fn chunk_ranges_cover_exactly() {
        for w in [1usize, 2, 3, 4, 64] {
            for n in [0usize, 1, 2, 5, 63, 64, 65, 1000] {
                let rs = chunk_ranges(n, w);
                assert_eq!(rs.len(), w);
                let mut pos = 0;
                let (base, rem) = (n / w, n % w);
                for (c, r) in rs.iter().enumerate() {
                    assert_eq!(r.start, pos);
                    assert_eq!(r.len(), base + usize::from(c < rem));
                    pos = r.end;
                }
                assert_eq!(pos, n);
            }
        }
    }

    #[test]
    fn reference_uses_rotated_order() {
        // n = 3, W = 3 なので要素 1 はチャンク 1: 加算順はランク 1, 2, 0。
        let a = [0.0f32, 1.0, 0.0];
        let b = [0.0f32, 1e8, 0.0];
        let c = [0.0f32, -1e8, 0.0];
        let got = reference_all_reduce(&[&a, &b, &c], ReduceOp::Sum).unwrap();
        // 回転順 (1e8 + -1e8) + 1 = 1。ランク番号順 (1 + 1e8) + -1e8 = 0 とは異なる。
        assert_eq!(got[1], 1.0);
        assert_eq!((1.0f32 + 1e8) + -1e8, 0.0);
    }

    #[test]
    fn reference_rejects_bad_inputs() {
        assert!(reference_all_reduce(&[], ReduceOp::Sum).is_err());
        assert!(reference_all_reduce(&[&[1.0], &[1.0, 2.0]], ReduceOp::Sum).is_err());
    }

    #[test]
    fn options_validate() {
        assert!(CollectiveOptions::default().validate().is_ok());
        let z = CollectiveOptions::default().io_timeout(Duration::ZERO);
        assert!(matches!(
            z.validate(),
            Err(DistributedError::InvalidConfig { .. })
        ));
        let m = CollectiveOptions::default().op_timeout(Duration::MAX);
        assert!(matches!(
            m.validate(),
            Err(DistributedError::InvalidConfig { .. })
        ));
        let m = CollectiveOptions::default().io_timeout(Duration::MAX);
        assert!(m.validate().is_err());
    }

    #[test]
    fn matches_reference_bitwise() {
        for w in 1..=4usize {
            for n in [0usize, 1, w.saturating_sub(1), 7, 1000] {
                for op in [ReduceOp::Sum, ReduceOp::Mean] {
                    check(w, n, op, None);
                }
            }
        }
    }

    #[test]
    fn multi_segment_and_empty_chunks() {
        for w in 2..=4usize {
            for n in [0usize, 1, 2, 5, 11, 50] {
                check(w, n, ReduceOp::Sum, Some(3));
                check(w, n, ReduceOp::Mean, Some(1));
            }
        }
    }

    #[test]
    fn large_exchange_does_not_deadlock() {
        let n = 2 * 1_048_576 + 5;
        let t = Instant::now();
        check(2, n, ReduceOp::Sum, None);
        assert!(t.elapsed() < Duration::from_secs(25));
    }

    #[test]
    fn repeated_calls_are_bit_identical() {
        let (w, n) = (3, 101);
        let inputs: Vec<Vec<f32>> = (0..w).map(|r| gen_input(r, n)).collect();
        let mut cs = comms(w, opts(), None);
        let first = run_all(&mut cs, &inputs, ReduceOp::Mean);
        for _ in 0..2 {
            let again = run_all(&mut cs, &inputs, ReduceOp::Mean);
            for (a, b) in first.iter().zip(&again) {
                assert_eq!(bits(a), bits(b));
            }
        }
        let mut cs2 = comms(w, opts(), None);
        let rebuilt = run_all(&mut cs2, &inputs, ReduceOp::Mean);
        assert_eq!(bits(&first[0]), bits(&rebuilt[2]));
    }

    #[test]
    fn nan_and_inf_propagate_like_reference() {
        let w = 3;
        let mut inputs: Vec<Vec<f32>> = (0..w).map(|r| gen_input(r, 9)).collect();
        inputs[0][0] = f32::NAN;
        inputs[1][1] = f32::INFINITY;
        inputs[2][1] = f32::NEG_INFINITY;
        inputs[2][4] = f32::INFINITY;
        let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let want = reference_all_reduce(&refs, ReduceOp::Sum).unwrap();
        let mut cs = comms(w, opts(), None);
        let outs = run_all(&mut cs, &inputs, ReduceOp::Sum);
        for o in &outs {
            for (g, e) in o.iter().zip(&want) {
                assert_eq!(g.is_nan(), e.is_nan());
                if !e.is_nan() {
                    assert_eq!(g.to_bits(), e.to_bits());
                }
            }
        }
        assert!(outs[0][0].is_nan() && outs[1][4].is_infinite());
    }

    #[test]
    fn world_size_one_is_a_no_op() {
        for op in [ReduceOp::Sum, ReduceOp::Mean] {
            let mut cs = comms(1, opts(), None);
            let snan = f32::from_bits(0x7fa0_0001);
            let mut b = vec![1.5, snan, -0.0];
            let before = bits(&b);
            cs[0].all_reduce(&mut b, op).unwrap();
            assert_eq!(bits(&b), before);
        }
    }

    #[test]
    fn disconnected_peer_returns_typed_error() {
        let mut cs = comms(3, opts().op_timeout(Duration::from_secs(8)), None);
        drop(cs.pop());
        let t = Instant::now();
        let res: Vec<Result<()>> = thread::scope(|s| {
            let hs: Vec<_> = cs
                .iter_mut()
                .map(|c| {
                    s.spawn(move || {
                        let mut b = gen_input(c.rank(), 64);
                        c.all_reduce(&mut b, ReduceOp::Sum)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(t.elapsed() < Duration::from_secs(10));
        for r in res {
            assert!(matches!(
                r,
                Err(DistributedError::PeerDisconnected | DistributedError::Timeout { .. })
            ));
        }
        assert!(cs.iter().all(|c| c.is_poisoned()));
    }

    #[test]
    fn silent_peer_times_out_and_poisons() {
        let mut cs = comms(2, opts().op_timeout(Duration::from_millis(500)), None);
        let t = Instant::now();
        let mut b = gen_input(0, 16);
        let e = cs[0].all_reduce(&mut b, ReduceOp::Sum).unwrap_err();
        assert!(matches!(e, DistributedError::Timeout { .. }), "{e:?}");
        assert!(t.elapsed() < Duration::from_secs(5));
        assert!(cs[0].is_poisoned());
        assert!(matches!(
            cs[0].all_reduce(&mut b, ReduceOp::Sum),
            Err(DistributedError::Poisoned)
        ));
        drop(cs);
    }

    #[test]
    fn mismatched_lengths_fail_without_hanging() {
        let mut cs = comms(2, opts().op_timeout(Duration::from_secs(8)), None);
        let t = Instant::now();
        let res: Vec<Result<()>> = thread::scope(|s| {
            let hs: Vec<_> = cs
                .iter_mut()
                .map(|c| {
                    s.spawn(move || {
                        let mut b = gen_input(c.rank(), 4 + 2 * c.rank());
                        c.all_reduce(&mut b, ReduceOp::Sum)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(t.elapsed() < Duration::from_secs(10));
        assert!(res.iter().all(|r| r.is_err()));
        assert!(
            res.iter()
                .any(|r| matches!(r, Err(DistributedError::Protocol(_))))
        );
    }

    #[test]
    fn fake_peer_wrong_sequence_and_length_are_rejected() {
        for bad_len in [false, true] {
            let mut links = establish_group(2);
            let peer = links.pop().unwrap();
            let mut c = RingCommunicator::new(links.pop().unwrap(), opts()).unwrap();
            // n = 4, W = 2: ランク 0 は reduce-scatter 1 ステップ目でチャンク 1（2 要素）を受ける。
            let hdr = SegmentHeader {
                step_seq: if bad_len { 0 } else { 99 },
                total_elems: 4,
                chunk: 1,
                seg_offset: 0,
            };
            let data: &[f32] = if bad_len { &[1.0; 5] } else { &[1.0; 2] };
            let mut s = peer.to_next.as_ref().unwrap();
            send_segment(&mut s, FrameKind::ReduceScatterSegment, &hdr, data).unwrap();
            let mut b = gen_input(0, 4);
            let e = c.all_reduce(&mut b, ReduceOp::Sum).unwrap_err();
            if bad_len {
                assert!(matches!(
                    e,
                    DistributedError::Protocol(ProtocolViolation::LengthMismatch { .. })
                ));
            } else {
                assert!(matches!(
                    e,
                    DistributedError::Protocol(ProtocolViolation::SequenceMismatch)
                ));
            }
            assert!(c.is_poisoned());
        }
    }
}
