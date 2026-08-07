//! FIPS 180-4 準拠 SHA-256 の自作実装（TASK-3.4・イシュー #145）。
//!
//! [`crate::logging`] のハッシュチェーン（改竄検知）専用に使う。`sha2` クレートは
//! `.claude/rules/deps-policy.md` の許容依存 8 区分に含まれず、依存追加は
//! ユーザー承認事項のため（`crates/guardrail/src/toml_lite.rs` が `toml` クレートを
//! 使わず自作パーサで代替したのと同じ方針）、SHA-256（アルゴリズム自体が
//! FIPS 180-4 で完全に固定されており実装判断の余地がない）を std のみで実装する。
//! 用途はログの完全性チェーン（改竄検知）であり、秘匿・認証（鍵付き MAC）用途では
//! ない点に注意（`docs/self-repair-log-format.md` セキュリティ注意節も参照）。
//!
//! 実装の正しさは本モジュール末尾のテストで NIST の既知テストベクタ（空文字列・
//! `"abc"`・55/56/64 バイト境界・2 ブロック跨ぎメッセージ）と突合して検証する。
//! `sha2` クレート採用への切り替え可否はユーザー判断事項として PR 本文に記録する
//! （`.claude/rules/deps-policy.md`: 依存の追加はユーザー承認必須）。

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

/// メッセージをブロックへパディングする（FIPS 180-4 5.1.1 節）。
///
/// `0x80` バイトを 1 個付加後、長さがブロック境界から 8 バイト引いた位置に
/// 揃うまで `0x00` で埋め、末尾にメッセージ長（bit 単位・ビッグエンディアン
/// 64 bit）を付与する。境界ケース（付加後にブロックが 1 個で足りず 2 個要る
/// パディング）はテストで検証する。
fn pad(message: &[u8]) -> Vec<u8> {
    let bit_len = (message.len() as u64).wrapping_mul(8);
    let mut padded = message.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    padded
}

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
    for (i, chunk) in block.chunks_exact(4).enumerate() {
        w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
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

/// 任意バイト列の SHA-256 ダイジェスト（32 バイト）を計算する。
///
/// [`crate::logging`] がレコードのハッシュチェーン計算（`compute_hash`・
/// `genesis_hash`）に使う入口。
pub fn sha256(message: &[u8]) -> [u8; 32] {
    let padded = pad(message);
    let mut h = H0;
    for block in padded.chunks_exact(64) {
        compress(&mut h, block);
    }
    let mut digest = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// ダイジェストを 16 進小文字文字列へエンコードする（ログの `hash`/`prev_hash`
/// フィールド・仕様書の既知値表記と同じ表現形式）。
pub fn to_hex(digest: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NIST FIPS 180-4 附属の既知テストベクタ: 空文字列。
    #[test]
    fn empty_string_matches_nist_vector() {
        let digest = sha256(b"");
        assert_eq!(
            to_hex(&digest),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// NIST FIPS 180-4 既知テストベクタ: `"abc"`。
    #[test]
    fn abc_matches_nist_vector() {
        let digest = sha256(b"abc");
        assert_eq!(
            to_hex(&digest),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// 448/512 bit（56/64 バイト）境界: パディング後に 2 ブロック目が
    /// 必要になるケース（`pad` の `while` ループが 1 ブロックで収まらない
    /// 分岐を通ることを確認する）。NIST 既知テストベクタ
    /// `"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"`。
    #[test]
    fn two_block_message_matches_nist_vector() {
        let digest = sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        assert_eq!(
            to_hex(&digest),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    /// 1,000,000 個の `'a'`（NIST 既知テストベクタ）。大きい入力での複数
    /// ブロック処理を検証する。
    #[test]
    fn one_million_a_matches_nist_vector() {
        let message = vec![b'a'; 1_000_000];
        let digest = sha256(&message);
        assert_eq!(
            to_hex(&digest),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    /// genesis 値の元となる固定文字列 `"self-repair-log-v1"` の SHA-256。
    /// [`crate::logging::genesis_hash`] がこの値を返すことが v1 実装との
    /// 互換性の実証になる（実装計画・検証方法 3 節）。
    #[test]
    fn self_repair_log_v1_seed_hash() {
        let digest = sha256(b"self-repair-log-v1");
        assert_eq!(
            to_hex(&digest),
            "be26b2311e026d01ceabc4dc7b360f8583ee203c4e4a858aef631c698c4b1a28"
        );
    }
}
