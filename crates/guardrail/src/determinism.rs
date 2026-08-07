//! 決定的シード設定ユーティリティ（TASK-4.4b・REQ-4）。
//!
//! 学習を伴う回帰テストはモデル初期化前に決定的シードを設定することを
//! `guardrail`・`self-repair` 双方に組み込む（`docs/spec/05-tasks.md` TASK-4.4）。
//! シード未固定の学習テストは非決定的に失敗し、`self-repair` の取り込み判断
//! （3 分岐判定。[`crate::decision`]）に対する偽陽性の原因となる
//! （PoC-2 発見事項 0・`docs/spec/04-requirements.md:117,290`）。本モジュールは
//! そのフレーキーテスト対策の共通入口であり、ガードレール判定ロジック自体には
//! 関与しない（判定閾値・許容誤差の変更ではない。`.claude/rules/security.md`）。
//!
//! PRNG 本体は再実装せず `bench-harness::rng::Xorshift64Star` を再利用する
//! （依存方向は `docs/guardrail-self-repair-cli.md` 1.4 節の既定どおり
//! `guardrail` → `bench-harness`）。`self-repair` 側は本モジュールを
//! `pub use guardrail::determinism;` で再輸出し、双方から同一の決定性契約を
//! 参照できるようにする（[`crate::self-repair`] 側の再輸出はそちらの
//! `lib.rs` を参照）。
//!
//! **用途限定（重要）**: 内部で用いる xorshift64* は暗号学的に安全な PRNG では
//! ない。学習系回帰テストの決定性確保には十分だが、鍵・トークン生成やその他
//! セキュリティ用途には使用しないこと（OWASP A02 暗号化の失敗の観点。
//! `bench_harness::rng` の用途限定コメントと同一方針）。

pub use bench_harness::rng::Xorshift64Star;

/// 学習系回帰テストの既定シード（`autodiff` 側の既存テスト慣行に合わせる。
/// 例: `crates/autodiff/src/nn/init.rs` のテストが用いる `42`）。
pub const DEFAULT_SEED: u64 = 42;

/// `(base, label)` から独立したシードを導出する。
///
/// `crates/autodiff/src/nn/init.rs` の `derive_seed`（review 指摘 #91: 単純な
/// 線形オフセット `seed + 1` 等では隣接呼び出しシード同士が系列相関を持ちうる
/// ことを実測確認済み）と同じ SplitMix64 finalizer 相当のビットミキシングを
/// 踏襲する。`label` は複数の学習系回帰テストが同一 `base`（例: [`DEFAULT_SEED`]）
/// から相互に独立したシードを導出するための識別子で、FNV-1a で 64bit に畳み込んで
/// からミキシングに合流させる（文字列 → 数値の決定的な変換であり、暗号学的な
/// ハッシュ強度は要求しない）。
pub fn derive_seed(base: u64, label: &str) -> u64 {
    // FNV-1a（64bit）: label の内容によって salt を決定的に分散させる。
    // オフセットバイアス・素数は FNV-1a の標準定数。
    let mut salt: u64 = 0xcbf29ce484222325;
    for byte in label.as_bytes() {
        salt ^= u64::from(*byte);
        salt = salt.wrapping_mul(0x100000001b3);
    }

    // SplitMix64 finalizer 相当のビットミキシング（autodiff/src/nn/init.rs の
    // derive_seed と同一定数・同一手順）。base と salt を独立した 64bit 値へ拡散し、
    // label が異なれば base が近接していても衝突しないようにする。
    let mut z = base.wrapping_add(salt.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `label` 用の決定的 RNG を生成する（[`DEFAULT_SEED`] と [`derive_seed`] の合成）。
///
/// 学習系回帰テストはモデル初期化前にこの関数で RNG を取得することで、
/// 「同一シード → 同一系列」（受け入れ条件）を満たす。
pub fn seeded_rng(label: &str) -> Xorshift64Star {
    Xorshift64Star::new(derive_seed(DEFAULT_SEED, label))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_rng_same_label_produces_same_sequence() {
        let mut a = seeded_rng("layer0.weight");
        let mut b = seeded_rng("layer0.weight");
        for _ in 0..50 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn seeded_rng_different_label_diverges() {
        let mut a = seeded_rng("layer0.weight");
        let mut b = seeded_rng("layer0.bias");
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn derive_seed_is_deterministic() {
        assert_eq!(
            derive_seed(DEFAULT_SEED, "layer0.weight"),
            derive_seed(DEFAULT_SEED, "layer0.weight")
        );
    }

    #[test]
    fn derive_seed_separates_adjacent_base_values_across_labels() {
        // autodiff/src/nn/init.rs の review #91 再現条件（連番シードの系列相関）を
        // label ベースの導出でも踏襲して確認する。
        for base in 1u64..=8 {
            let a = derive_seed(base, "a");
            let b = derive_seed(base + 1, "b");
            assert_ne!(a, b, "base={base}: label 違いの隣接 base が衝突");
        }
    }

    #[test]
    fn derive_seed_differs_between_labels_for_same_base() {
        for base in 0u64..=8 {
            assert_ne!(
                derive_seed(base, "weight"),
                derive_seed(base, "bias"),
                "base={base}: label 違いで導出後シードが衝突"
            );
        }
    }

    #[test]
    fn derive_seed_snapshot_is_stable() {
        // アルゴリズムの意図しない変更を検知するためのスナップショット固定。
        // 値そのものに意味はなく、変わったこと自体が「決定的シードの契約が
        // 変わった」ことの回帰検知シグナルとなる。
        assert_eq!(derive_seed(42, "layer0.weight"), 1214869257472169536);
    }

    #[test]
    fn zero_base_is_corrected_by_underlying_prng() {
        // derive_seed(0, ..) がたまたま 0 になった場合でも、
        // Xorshift64Star::new の 0 シード補正契約（bench_harness::rng 参照）が
        // 引き続き効くことを確認する。
        let mut rng = Xorshift64Star::new(0);
        assert_ne!(rng.next_u64(), 0);
    }
}
