//! 決定的シードの重み初期化ヘルパー（`nn::Linear`・TASK-9.1a・#91）。
//!
//! xorshift64* PRNG コア（`Xorshift64Star`）は `bench-harness::rng::Xorshift64Star`
//! （`crates/bench-harness/src/rng.rs`）と**同一アルゴリズムを差分なしで
//! 再掲**する。`bench-harness` はベンチ計測クレートであり、`autodiff`
//! の本体コード（`src/`）から依存すると層構造上不適切（`autodiff` は
//! `crates/tensor-core` のみに依存する薄いコアであるべき。TASK-9.1a
//! 計画 §3.3）なので、`dev-dependencies` に留めたまま意図的に限定重複
//! させる。`autodiff/tests/poc_v2_2_parity.rs` は引き続き
//! `bench-harness` 版をテスト専用に使う（重複はここだけ）。
//!
//! PRNG コアの**上**に `derive_seed`（本ファイル下部）というシード導出層
//! を重ねている点は `bench-harness` 版との差分である。`Linear::new`
//! （`nn/linear.rs`）が weight・bias の 2 系統を 1 つの呼び出しシードから
//! 導出する際、単純な `seed`/`seed + 1` のような線形オフセットだと、
//! 複数層を連番シードで構築する自然な使い方（`Linear::new(.., 1)` →
//! `Linear::new(.., 2)` → ...）で「層 i の bias 系列」と「層 i+1 の
//! weight 系列」が同一の xorshift64* 生の乱数列を使い回し、スケール違い
//! なだけの完全相関列になりうる（review 指摘 #91 で実測確認）。
//! `derive_seed` は SplitMix64 の finalizer 相当のビットミキシングで
//! `(seed, salt)` を独立した 64bit 値へ拡散するため、salt が異なれば
//! 隣接する呼び出しシード同士でも衝突しない。
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
/// `in_features == 0` は呼び出し元（`Linear::new`）が構築前の引数検証
/// （`AutodiffError::InvalidArgument`。`error.rs` 参照）で事前に弾く契約
/// とし、本関数は `bound` が有限の正値であることのみを前提とする。
pub(crate) fn uniform_init(len: usize, bound: f32, seed: u64) -> Vec<f32> {
    let mut rng = Xorshift64Star::new(seed);
    (0..len).map(|_| rng.next_f32() * bound).collect()
}

/// `Linear::new` の呼び出しシード 1 個から weight・bias 用の独立した
/// シードを導出する（SplitMix64 の finalizer 相当のビットミキシング。
/// 参照実装: <https://prng.di.unimi.it/splitmix64.c> のアルゴリズムを
/// Rust へ移植）。単純な `seed + salt` のような線形オフセットでは、
/// 連番シードで複数層を構築する自然な使い方（層 i の bias salt=1 と
/// 層 i+1 の weight salt=0 呼び出しシードが 1 違いなだけ）で xorshift64*
/// の生の乱数列が丸ごと重複しうる（review 指摘 #91）。`salt` を乗算後に
/// 加算してから拡散するため、`seed` が隣接していても `salt` が異なれば
/// 独立した状態から系列が始まる。
pub(crate) const WEIGHT_SEED_SALT: u64 = 0;
pub(crate) const BIAS_SEED_SALT: u64 = 1;

pub(crate) fn derive_seed(seed: u64, salt: u64) -> u64 {
    let mut z = seed.wrapping_add(salt.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
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
    fn derive_seed_separates_adjacent_call_seeds_across_salts() {
        // review 指摘 #91 の再現条件: 連番呼び出しシード
        // 1, 2, 3, ... で「層 i の bias（salt=1）」と「層 i+1 の
        // weight（salt=0）」の導出後シードが衝突しないことを、
        // 実際に Linear::new が使う範囲（呼び出しシード 1..=8）で
        // 網羅的に確認する。
        for call_seed in 1u64..=8 {
            let bias_seed = derive_seed(call_seed, BIAS_SEED_SALT);
            let next_weight_seed = derive_seed(call_seed + 1, WEIGHT_SEED_SALT);
            assert_ne!(
                bias_seed, next_weight_seed,
                "call_seed={call_seed}: bias と次層 weight の導出後シードが衝突"
            );
        }
    }

    #[test]
    fn derive_seed_differs_between_weight_and_bias_salt_for_same_call_seed() {
        for call_seed in 0u64..=8 {
            assert_ne!(
                derive_seed(call_seed, WEIGHT_SEED_SALT),
                derive_seed(call_seed, BIAS_SEED_SALT),
                "call_seed={call_seed}: weight と bias の導出後シードが衝突"
            );
        }
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
