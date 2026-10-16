//! FIPS 180-4 SHA-256 のストリーミング実装（イシュー #3125。`ModelRegistry::download` の pin 検証用）。
//!
//! `K`・`H0`・`rotr`・`compress` は `crates/self-repair/src/sha256.rs` から逐語移植した。
//! self-repair は非公開クレートで facade から依存できず、`sha2` は許容依存に無い
//! （`docs/model-download-design.md` §17.2）ため std のみで持つ。self-repair 側の
//! 一括 API と異なり、受信チャンクごとに `update` できる（全量をメモリに溜めない）。
//! 用途は整合性検証であり鍵付き MAC ではない。

/// 各ラウンドの加法定数 K（最初の 64 個の素数の立方根の小数部の先頭 32 bit）。
/// FIPS 180-4 4.2.2 節の定数表そのまま。
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// 初期ハッシュ値 H0（最初の 8 個の素数の平方根の小数部の先頭 32 bit）。
/// FIPS 180-4 5.3.3 節。
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// 32 bit 語の右巡回シフト（FIPS 180-4 3.2 節 `ROTR`）。
fn rotr(x: u32, n: u32) -> u32 {
    x.rotate_right(n)
}

/// 1 ブロック（64 バイト）を現在のハッシュ状態 `h` に圧縮する（FIPS 180-4
/// 6.2.2 節の圧縮関数）。
fn compress(h: &mut [u32; 8], block: &[u8]) {
    debug_assert_eq!(block.len(), 64);

    // メッセージスケジュール W[0..64]（6.2.2 手順 1）。
    let mut w = [0u32; 64];
    for (i, chunk) in block.as_chunks::<4>().0.iter().enumerate() {
        w[i] = u32::from_be_bytes(*chunk);
    }
    for i in 16..64 {
        let s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        let s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    // 作業変数の初期化（6.2.2 手順 2）。
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;

    // 64 ラウンドの圧縮（6.2.2 手順 3）。
    for i in 0..64 {
        let s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    // 中間ハッシュ値の更新（6.2.2 手順 4）。
    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
}

/// ストリーミング SHA-256。`download` が受信チャンクごとに `update` し、EOF で `finalize` する。
pub(super) struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

impl Sha256 {
    pub(super) fn new() -> Self {
        Self {
            state: H0,
            buf: [0; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    pub(super) fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                compress(&mut self.state, &block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            compress(&mut self.state, &data[..64]);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    pub(super) fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        let mut tail = [0u8; 128];
        tail[..self.buf_len].copy_from_slice(&self.buf[..self.buf_len]);
        tail[self.buf_len] = 0x80;
        let blocks = if self.buf_len < 56 { 1 } else { 2 };
        let end = blocks * 64;
        tail[end - 8..end].copy_from_slice(&bit_len.to_be_bytes());
        for b in 0..blocks {
            compress(&mut self.state, &tail[b * 64..(b + 1) * 64]);
        }
        let mut out = [0u8; 32];
        for (i, w) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(d: [u8; 32]) -> String {
        d.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn one_shot(data: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(data);
        hex(h.finalize())
    }

    #[test]
    fn nist_vectors() {
        assert_eq!(
            one_shot(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            one_shot(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            one_shot(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            one_shot(b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );
        let million = vec![b'a'; 1_000_000];
        assert_eq!(
            one_shot(&million),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn matches_reference_across_padding_boundaries_and_splits() {
        // self-repair の一括実装と同じ式（pad 方式）を参照とし、55/56/63/64/65 境界と
        // 分割投入が一括投入と一致することを確かめる。
        fn reference(message: &[u8]) -> [u8; 32] {
            let bit_len = (message.len() as u64).wrapping_mul(8);
            let mut padded = message.to_vec();
            padded.push(0x80);
            while padded.len() % 64 != 56 {
                padded.push(0);
            }
            padded.extend_from_slice(&bit_len.to_be_bytes());
            let mut h = H0;
            for block in padded.chunks(64) {
                compress(&mut h, block);
            }
            let mut out = [0u8; 32];
            for (i, w) in h.iter().enumerate() {
                out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
            }
            out
        }
        for n in [
            0usize, 1, 54, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 1000,
        ] {
            let data: Vec<u8> = (0..n).map(|i| (i * 7 + 3) as u8).collect();
            let want = reference(&data);
            let mut h = Sha256::new();
            h.update(&data);
            assert_eq!(h.finalize(), want, "一括 n={n}");
            for step in [1usize, 7, 64, 100] {
                let mut h = Sha256::new();
                for c in data.chunks(step) {
                    h.update(c);
                }
                assert_eq!(h.finalize(), want, "分割 n={n} step={step}");
            }
        }
    }
}
