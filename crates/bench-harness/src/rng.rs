//! 決定的シードの自作 PRNG（xorshift64*）。
//!
//! `bench-harness`（本クレート）はベンチ入力・回帰テスト用の乱数生成を
//! `rand` 等の外部クレートに依存せず自作する（許容依存 8 区分外の追加を
//! 避けるため。`.claude/rules/deps-policy.md`）。シード未固定はフレーキー
//! テスト・AI 自律メンテナンスの偽陽性の原因となる（PoC-2 発見事項 0。
//! `docs/spec/04-requirements.md:117,290`）ため、「同一シード → 同一入力
//! 系列」を保証する本モジュールを、計測コア（TASK-8.1a）・回帰テスト
//! （TASK-8.1c）双方の入力生成の基盤として提供する。
//!
//! **移植元**: `docs/spec/03-poc/poc-v2-5-backend-numeric-parity/code/rust/src/rng.rs`
//! （PoC-v2-1/3/4/5 で共通利用され、CPU/CUDA/Metal 全バックエンド PoC の
//! 数値比較（PoC-v2-5）の前提を揃えた実装。差分なしで移植する）。
//!
//! **用途限定（重要）**: xorshift64* は暗号学的に安全な PRNG ではない。
//! 周期・分布はベンチマーク入力生成・回帰テストの決定性確保には十分だが、
//! 鍵・トークン生成やその他セキュリティ用途には使用しないこと
//! （OWASP A02 暗号化の失敗の観点。`.claude/rules/security.md`）。

/// xorshift64* 状態。
///
/// 呼び出し元（ベンチ計測コア・回帰テスト）はシード値のみを指定し、
/// 生成される数列は完全に決定的になる。
pub struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    /// シードが 0 だと xorshift の不動点（常に 0 を返す）に陥るため、
    /// 0 は非零値（黄金比由来の定数）に補正する。
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    /// 次の 64bit 乱数を返す。
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// `[-1.0, 1.0)` の範囲に収まる f32 を返す。
    ///
    /// ベンチ入力・回帰テストのテンソル要素として、極端な桁の値による
    /// 桁あふれ・打ち切り誤差の偏りを避けるため、絶対値 1 以下の範囲に
    /// 限定する（PoC-v2-1/3/5 実測方針を踏襲）。
    pub fn next_f32(&mut self) -> f32 {
        // 上位 24bit を仮数部として使い、[0, 1) の一様分布を作ってから [-1, 1) に写す。
        let bits = (self.next_u64() >> 40) as u32; // 24bit
        let unit = bits as f32 / (1u32 << 24) as f32; // [0, 1)
        unit * 2.0 - 1.0
    }

    /// 長さ `len` の f32 ベクトルを決定的に生成する。
    pub fn fill_vec(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next_f32()).collect()
    }

    /// 長さ `len` の f16 ベクトルを決定的に生成する。
    ///
    /// f32 と同じ `next_f32` 系列から `half::f16::from_f32` で丸めることで、
    /// f32/f16 の入力を「同じ乱数系列を丸めただけ」の関係に保ち、
    /// バックエンド間数値一致検証（REQ-2）の精度差起因を丸め誤差のみに
    /// 限定できるようにする（PRNG の系列自体を f16/f32 で分けない。
    /// PoC-v2-3 で確立した方針）。
    pub fn fill_vec_f16(&mut self, len: usize) -> Vec<half::f16> {
        (0..len)
            .map(|_| half::f16::from_f32(self.next_f32()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_produces_same_sequence() {
        let mut a = Xorshift64Star::new(42);
        let mut b = Xorshift64Star::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seed_diverges() {
        let mut a = Xorshift64Star::new(1);
        let mut b = Xorshift64Star::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn f32_in_range() {
        let mut r = Xorshift64Star::new(7);
        for _ in 0..1000 {
            let v = r.next_f32();
            assert!((-1.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn zero_seed_is_corrected() {
        // 0 シードでも不動点（常に 0 を返す）に陥らないことを確認する。
        let mut r = Xorshift64Star::new(0);
        assert_ne!(r.next_u64(), 0);
    }

    #[test]
    fn fill_vec_is_deterministic_and_sized() {
        let mut a = Xorshift64Star::new(123);
        let mut b = Xorshift64Star::new(123);
        let va = a.fill_vec(50);
        let vb = b.fill_vec(50);
        assert_eq!(va.len(), 50);
        assert_eq!(va, vb);
    }

    #[test]
    fn fill_vec_f16_matches_f32_sequence_rounding() {
        // f16/f32 が同一乱数系列を丸めただけの関係にあることを確認する
        // （PoC-v2-3 方針。`fill_vec_f16` のドキュメンテーションコメント参照）。
        let mut as_f32 = Xorshift64Star::new(99);
        let mut as_f16 = Xorshift64Star::new(99);
        let expected: Vec<half::f16> = as_f32
            .fill_vec(20)
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let actual = as_f16.fill_vec_f16(20);
        assert_eq!(expected, actual);
    }
}
