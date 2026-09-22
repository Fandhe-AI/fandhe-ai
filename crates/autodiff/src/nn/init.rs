//! 決定的シードの重み初期化ヘルパー（`nn::Linear`・TASK-9.1a・#91）。
//!
//! xorshift64* PRNG コア（`Xorshift64Star`）は、以前は `bench-harness::
//! rng::Xorshift64Star` と同一アルゴリズムを本ファイルへ差分なしで
//! 再掲していたが、イシュー #1724（プロセスグローバル決定的 RNG 契約
//! `manual_seed` の新設）で共通コアを `tensor-core::rng::Xorshift64Star`
//! へ一本化したため、本ファイルはそこへ委譲する（`tensor-core` は
//! `autodiff` の依存先であり層構造上の逆転は生じない。`autodiff` 本体
//! コードが `bench-harness`〈ベンチ計測クレート〉に依存しない方針
//! 自体は不変。TASK-9.1a 計画 §3.3）。`bench-harness::rng::
//! Xorshift64Star` は同アルゴリズムのまま意図的に独立重複を続ける
//! （層構造上の理由は同ファイル冒頭コメント参照。統合しない）。
//! `autodiff/tests/poc_v2_2_parity.rs` は引き続き `bench-harness` 版を
//! テスト専用に使う。
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
//! **既存の個別シード API と `tensor-core::rng::manual_seed` の関係**:
//! 本ファイルの `Xorshift64Star` は `uniform_init`／`try_uniform_init`
//! の呼び出しごとに新規構築される一時的な状態であり、
//! `tensor-core::rng` が提供するプロセスグローバルな RNG 状態
//! （`manual_seed`）とは完全に独立する（`manual_seed` を何度呼んでも
//! `Linear::new(.., seed)` の出力は変わらない。イシュー #1724。設計は
//! `docs/rng-global-contract-design.md`）。
//!
//! **用途限定（重要）**: xorshift64* は暗号学的に安全な PRNG ではない。
//! 重み初期化・回帰テストの決定性確保には十分だが、鍵・トークン生成や
//! その他セキュリティ用途には使用しないこと
//! （OWASP A02 暗号化の失敗の観点。`.claude/rules/security.md`）。

use fandhe_ai_tensor_core::rng::Xorshift64Star;

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

/// `uniform_init` のフォールブル版。`(0..len).collect()` は `Vec<f32>`
/// の確保が総バイト数 `isize::MAX` を超えると capacity overflow で
/// panic する。`nn::rnn::build_gate_params` は `checked_mul` で
/// `usize` の乗算オーバーフローのみを検証済みの `len`（例:
/// `RnnCell::new(1usize << 61, 1, false, 0)` は `gh=1`・
/// `w_ih_len=1usize<<61` のようにすべての `checked_mul` を通過しつつ
/// `f32` 4 バイト換算で `isize::MAX` を超えうる）を渡しうるため、
/// `collect()` の代わりに `try_reserve_exact` で確保可否を先に検証し、
/// 確保不能時は panic させず `Err` を返す（本番経路 panic 禁止。
/// `.claude/rules/coding-rust.md`。イシュー #1647 codex-review P1
/// 指摘）。呼び出し元（`nn::rnn::build_gate_params`）が
/// `AutodiffError::InvalidArgument` へ変換する。
pub(crate) fn try_uniform_init(
    len: usize,
    bound: f32,
    seed: u64,
) -> Result<Vec<f32>, std::collections::TryReserveError> {
    let mut rng = Xorshift64Star::new(seed);
    let mut values = Vec::new();
    values.try_reserve_exact(len)?;
    for _ in 0..len {
        values.push(rng.next_f32() * bound);
    }
    Ok(values)
}

/// `Embedding::new`（`nn/embedding.rs`）から呼ばれる重み初期化本体
/// （イシュー #1604）。PyTorch `nn.Embedding` の既定初期化
/// （`N(0, 1)`。標準正規分布）に整合させる。Box–Muller 変換
/// （`f64` 中間計算）を使う点は [`fandhe_ai_tensor_core::rng::randn`]
/// と同一の変換式だが、本関数はプロセスグローバルな
/// [`fandhe_ai_tensor_core::rng::with_global_rng`] を経由せず、
/// [`uniform_init`] と同じく呼び出しごとに新規構築した一時的な
/// `Xorshift64Star` を使う（`nn::Linear::new` と同じ「グローバル
/// `manual_seed` 状態から独立」契約。モジュール冒頭コメント参照）。
///
/// 決定性の範囲は `randn` と同じ「同一プロセス・同一プラットフォーム
/// 内での再現」に限る（`ln`／`sin`／`cos` を経由するため。
/// `docs/rng-global-contract-design.md`）。`len == 0` は空 `Vec` を
/// 返す（`randn(&[0, D])` と同じ扱い。呼び出し元での境界検査は
/// 不要）。
///
/// `try_uniform_init` と同じ理由でフォールブルにする:
/// `Vec::with_capacity(len)` は総バイト数 `isize::MAX` を超えると
/// capacity overflow で panic するため、`try_reserve_exact` で確保
/// 可否を先に検証し、確保不能時は panic させず `Err` を返す（本番
/// 経路 panic 禁止。`.claude/rules/coding-rust.md`。イシュー #1604
/// codex-review P1 指摘: `Embedding::new` が `num_embeddings *
/// embedding_dim` の大きな `len` を渡しうる）。呼び出し元
/// （`nn::embedding::Embedding::new`）が `AutodiffError::
/// InvalidArgument` へ変換する。
pub(crate) fn try_normal_init(
    len: usize,
    seed: u64,
) -> Result<Vec<f32>, std::collections::TryReserveError> {
    let mut rng = Xorshift64Star::new(seed);
    let mut out = Vec::new();
    out.try_reserve_exact(len)?;
    let mut remaining = len;
    while remaining > 0 {
        // Box–Muller 変換。`u1` は `(0, 1]` に
        // 補正して `ln(0)`（負の無限大）を避ける。
        let u1 = 1.0 - rng.next_unit_f64();
        let u2 = rng.next_unit_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        let z0 = (r * theta.cos()) as f32;
        out.push(z0);
        remaining -= 1;
        if remaining == 0 {
            break;
        }
        let z1 = (r * theta.sin()) as f32;
        out.push(z1);
        remaining -= 1;
    }
    Ok(out)
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
/// `nn::rnn`（イシュー #1647）の `weight_hh` 導出用ソルト。`Linear` の
/// `WEIGHT_SEED_SALT`／`BIAS_SEED_SALT`（0・1）と衝突しない値を割り当て、
/// RNN／LSTM／GRU セルが `weight_ih`／`weight_hh`／`bias_ih`／`bias_hh`
/// の 4 系統を同一呼び出しシードから独立に導出できるようにする。
pub(crate) const WEIGHT_HH_SEED_SALT: u64 = 2;
/// `nn::rnn` の `bias_hh` 導出用ソルト（上記参照）。
pub(crate) const BIAS_HH_SEED_SALT: u64 = 3;
/// `nn::attention`（イシュー #1640。`MultiheadAttention`）の
/// `q_proj`／`k_proj`／`v_proj`／`out_proj` 導出用ソルト。既存の
/// `WEIGHT_SEED_SALT`〜`BIAS_HH_SEED_SALT`（0..=3）と衝突しない値
/// （4..=7）を割り当て、`MultiheadAttention::new` が単一の呼び出し
/// シードから 4 個の独立した `Linear::new` 呼び出しシードを導出できる
/// ようにする（`Linear::new` 自身がさらに weight／bias の 2 系統へ
/// `WEIGHT_SEED_SALT`／`BIAS_SEED_SALT` を再適用するため、2 段の
/// `derive_seed` 合成になる）。
pub(crate) const ATTN_Q_SEED_SALT: u64 = 4;
pub(crate) const ATTN_K_SEED_SALT: u64 = 5;
pub(crate) const ATTN_V_SEED_SALT: u64 = 6;
pub(crate) const ATTN_OUT_SEED_SALT: u64 = 7;

/// `nn::transformer_encoder_layer`（イシュー #2068）が単一の呼び出し
/// シードから self-attention・FFN 第 1 層・FFN 第 2 層の 3 系統を独立に
/// 導出するためのソルト。既存の `WEIGHT_SEED_SALT`〜`ATTN_OUT_SEED_SALT`
/// （0..=7）と衝突しない値（8..=10）を割り当てる。`ENC_ATTN_SEED_SALT`
/// で導出したシードは `MultiheadAttention::new` へさらに渡され、
/// そちら側で `ATTN_Q_SEED_SALT`〜`ATTN_OUT_SEED_SALT` を再適用する
/// （`ATTN_Q_SEED_SALT` 等と同じ「2 段の `derive_seed` 合成」構造）。
pub(crate) const ENC_ATTN_SEED_SALT: u64 = 8;
/// `nn::transformer_encoder_layer` の FFN 第 1 層（`linear1`）導出用
/// ソルト（上記参照）。
pub(crate) const ENC_LINEAR1_SEED_SALT: u64 = 9;
/// `nn::transformer_encoder_layer` の FFN 第 2 層（`linear2`）導出用
/// ソルト（上記参照）。
pub(crate) const ENC_LINEAR2_SEED_SALT: u64 = 10;

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

    // `try_normal_init`（イシュー #1604）の単体テスト。
    #[test]
    fn normal_init_same_seed_reproducible_within_process() {
        let a = try_normal_init(64, 42).unwrap();
        let b = try_normal_init(64, 42).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn normal_init_different_seed_diverges() {
        let a = try_normal_init(64, 1).unwrap();
        let b = try_normal_init(64, 2).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn normal_init_zero_len_returns_empty() {
        let values = try_normal_init(0, 42).unwrap();
        assert!(values.is_empty());
    }

    #[test]
    fn normal_init_odd_len_has_correct_length() {
        let values = try_normal_init(7, 42).unwrap();
        assert_eq!(values.len(), 7);
    }

    #[test]
    fn normal_init_large_sample_has_roughly_standard_normal_statistics() {
        // `randn` の同名テスト（`tensor-core::rng`）と同じ粗い検査:
        // 大標本で平均・分散が N(0, 1) から大きく外れないことのみ確認
        // する（厳密な統計検定ではない。決定的シードで再現可能）。
        let values = try_normal_init(20_000, 7).unwrap();
        let n = values.len() as f64;
        let mean: f64 = values.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var: f64 = values
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 0.05, "mean out of range: {mean}");
        assert!((var - 1.0).abs() < 0.1, "var out of range: {var}");
    }
}
