//! 決定的シードの重み初期化ヘルパー（`nn::Linear`・TASK-9.1a・#91）。
//!
//! xorshift64* PRNG は `bench-harness::rng::Xorshift64Star`
//! （`crates/bench-harness/src/rng.rs`）と**同一アルゴリズムを差分なしで
//! 再掲**する。`bench-harness` はベンチ計測クレートであり、`autodiff`
//! の本体コード（`src/`）から依存すると層構造上不適切（`autodiff` は
//! `crates/tensor-core` のみに依存する薄いコアであるべき。TASK-9.1a
//! 計画 §3.3）なので、`dev-dependencies` に留めたまま意図的に限定重複
//! させる。`autodiff/tests/poc_v2_2_parity.rs` は引き続き
//! `bench-harness` 版をテスト専用に使う（重複はここだけ）。
//!
//! **用途限定（重要）**: xorshift64* は暗号学的に安全な PRNG ではない。
//! 重み初期化・回帰テストの決定性確保には十分だが、鍵・トークン生成や
//! その他セキュリティ用途には使用しないこと
//! （OWASP A02 暗号化の失敗の観点。`.claude/rules/security.md`）。

/// xorshift64* 状態。`bench-harness::rng::Xorshift64Star` と同一実装
/// （移植元: `docs/spec/03-poc/poc-v2-5-backend-numeric-parity/code/rust/src/rng.rs`）。
struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    /// シードが 0 だと xorshift の不動点（常に 0 を返す）に陥るため、
    /// 0 は非零値（黄金比由来の定数）に補正する。
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    /// 次の 64bit 乱数を返す。
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// `[-1.0, 1.0)` の範囲に収まる f32 を返す。
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24bit
        let unit = bits as f32 / (1u32 << 24) as f32; // [0, 1)
        unit * 2.0 - 1.0
    }
}

/// `Linear::new`（`nn/linear.rs`）から呼ばれる重み初期化本体。
///
/// PyTorch `nn.Linear` の既定初期化（`U(-1/√in_features, 1/√in_features)`
/// の一様分布。`kaiming_uniform_` の `a=√5` 特殊ケースと同じ有効範囲）に
/// 整合させる（TASK-9.1a 計画 §3.3）。「同一シード → 同一重み」を保証し
/// （coding-rust.md の学習系回帰テスト向け決定的シード方針・#93 収束
/// テストの前提）、長さ `len` の f32 ベクトルを返す。
///
/// `in_features == 0` は呼び出し元（`Linear::new`）が shape 検証で
/// 事前に弾く契約とし、本関数は `bound` が有限の正値であることのみを
/// 前提とする。
pub(crate) fn uniform_init(len: usize, bound: f32, seed: u64) -> Vec<f32> {
    let mut rng = Xorshift64Star::new(seed);
    (0..len).map(|_| rng.next_f32() * bound).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_produces_same_weights() {
        let a = uniform_init(16, 0.5, 42);
        let b = uniform_init(16, 0.5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seed_diverges() {
        let a = uniform_init(16, 0.5, 1);
        let b = uniform_init(16, 0.5, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn values_are_within_bound() {
        let bound = 1.0 / (8f32).sqrt();
        let values = uniform_init(1000, bound, 7);
        for v in values {
            assert!(v.abs() <= bound, "out of bound: {v}");
        }
    }

    #[test]
    fn zero_seed_is_corrected() {
        // bench-harness 版と同じ補正契約を持つことを確認する
        // （0 シードでも不動点に陥らない）。
        let mut rng = Xorshift64Star::new(0);
        assert_ne!(rng.next_u64(), 0);
    }
}
