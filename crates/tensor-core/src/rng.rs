//! プロセス全体で共有されるグローバル決定的 RNG 契約
//! （PyTorch `torch.manual_seed` 相当。イシュー #1724。親 #1602）。
//!
//! # 位置づけ
//!
//! `Tensor::zeros`／`ones`／`full`（`tensor.rs`）と同じ「ホスト側だけで
//! 完結する生成系」レイヤーに属し、`BackendOps` を経由しない。将来の
//! `randn`／`rand`／`randint`（イシュー #1725）は本モジュールの
//! [`with_global_rng`] を通じて値を引く（ホスト生成 → デバイスへ
//! アップロードする方式。#1602 本文の設計方針）。CUDA／Metal のデバイス
//! 側乱数生成カーネルは対象外。
//!
//! # 既存の個別シード API との関係（独立した別機構）
//!
//! `autodiff::nn::Linear::new(.., seed: u64)`・`nn::rnn::RnnCell::new(..,
//! seed)` 等は、呼び出しごとに新規の一時的な PRNG 状態
//! （`autodiff::nn::init::Xorshift64Star`。本モジュールの
//! [`Xorshift64Star`] を再利用する）を構築するローカルなシード方式で
//! あり、本モジュールが提供するプロセスグローバルな状態
//! （[`manual_seed`]／[`with_global_rng`]）とは**完全に独立**する。
//! `manual_seed` を何度呼んでも `Linear::new(.., seed)` 等の出力は
//! 変わらない（`crates/autodiff/src/nn/init.rs` の
//! `linear_new_is_unaffected_by_global_manual_seed_state` で機構的に
//! 固定している）。設計判断の詳細は `docs/rng-global-contract-design.md`。
//!
//! # スレッド安全性・再現性の範囲
//!
//! グローバル状態は `Mutex` で直列化するため複数スレッドから安全に
//! 呼び出せるが、他スレッドと同時に [`manual_seed`]／[`with_global_rng`]
//! を呼ぶと「どのスレッドの呼び出しが先に消費するか」は決まらないため、
//! 単一スレッド内で完結する呼び出し列でない限り厳密な呼び出し順序の
//! 再現性は保証しない（PyTorch のグローバル generator も同種の制約を
//! 持つ。`docs/rng-global-contract-design.md` 参照）。
//!
//! # 用途限定（重要）
//!
//! xorshift64* は暗号学的に安全な PRNG ではない。重み初期化・回帰
//! テストの決定性確保・乱数テンソル生成には十分だが、鍵・トークン生成
//! やその他セキュリティ用途には使用しないこと（OWASP A02 暗号化の
//! 失敗の観点。`.claude/rules/security.md`）。

use std::sync::{Mutex, OnceLock};

/// xorshift64* 状態。`autodiff::nn::init::Xorshift64Star`（旧実装）・
/// `bench-harness::rng::Xorshift64Star` と同一アルゴリズム
/// （移植元: `docs/spec/03-poc/poc-v2-5-backend-numeric-parity/code/rust/src/rng.rs`）。
/// `bench-harness` 版は「ベンチ計測クレートに `autodiff`／`tensor-core`
/// 本体コードから依存しない」という層構造上の理由（`autodiff::nn::init`
/// モジュール冒頭コメント参照）で意図的に独立重複させたままとし、本
/// モジュールでは統合しない。
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
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24bit
        let unit = bits as f32 / (1u32 << 24) as f32; // [0, 1)
        unit * 2.0 - 1.0
    }
}

/// プロセスグローバルな RNG 状態。[`manual_seed`] を一度も呼ばない間は
/// 固定の既定シード（`0`）で遅延初期化する（「呼び出し前でも決定的」
/// という既存方針。PoC-2 発見事項 0・フレーキーテスト回避）。
fn global_state() -> &'static Mutex<Xorshift64Star> {
    static STATE: OnceLock<Mutex<Xorshift64Star>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(Xorshift64Star::new(0)))
}

/// PyTorch `torch.manual_seed` 相当。プロセス全体で共有されるグローバル
/// 決定的 RNG（[`with_global_rng`] 経由で将来の `randn`／`rand`／
/// `randint`〈#1725〉が消費する）の状態を `seed` からやり直す。
///
/// 既存の個別シード API（`nn::Linear::new(.., seed)` 等）とは独立した
/// 別機構であり、本関数を呼んでもそれらの挙動には一切影響しない
/// （モジュール冒頭コメント参照）。
pub fn manual_seed(seed: u64) {
    let mut guard = global_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard = Xorshift64Star::new(seed);
}

/// グローバル RNG 状態への排他アクセスを与える内部アクセサ
/// （`autodiff`／`facade` の乱数テンソル生成 API〈#1725〉が消費する）。
///
/// `f` の実行中はロックを保持し続けるため、複数値を連続して引く一括
/// 生成（例: `randn(shape)` が要素数分だけ値を引く操作）全体が他の
/// 並行呼び出しに割り込まれない単位になる（単純な `AtomicU64` による
/// 1 語ずつの CAS では、複数値をまとめて引く操作の原子性を保証できない
/// ため `Mutex` を選んだ設計判断。`docs/rng-global-contract-design.md`）。
///
/// 他スレッドがロック保持中に panic した場合の poison は
/// `into_inner()` で握り潰して継続する（本番経路で `unwrap()`／
/// `expect()` を使わない方針。`.claude/rules/coding-rust.md`。1 度の
/// panic が以後すべての RNG 呼び出しを恒久的に破壊しないための処方。
/// `backend-cuda::precision::FlagGuard::acquire` と同型）。
pub fn with_global_rng<R>(f: impl FnOnce(&mut Xorshift64Star) -> R) -> R {
    let mut guard = global_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f(&mut guard)
}

/// テスト直列化用ロック（`#[cfg(test)]` 限定・`pub(crate)`）。本モジュール
/// のグローバル状態を検証するテスト同士が `cargo test` の既定並列実行
/// で競合しないよう、`backend-cuda::precision::tf32_flag_test_lock`・
/// `backend-metal::split_k_runtime` と同型の直列化ロックを用意する。
#[cfg(test)]
pub(crate) fn global_rng_test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_seed_then_draws_are_deterministic() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(42);
        let a: Vec<u64> = (0..8).map(|_| with_global_rng(|r| r.next_u64())).collect();

        manual_seed(42);
        let b: Vec<u64> = (0..8).map(|_| with_global_rng(|r| r.next_u64())).collect();

        assert_eq!(a, b);
    }

    #[test]
    fn manual_seed_reseeds_regardless_of_prior_state() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // 事前に任意回数引いておき、その後の manual_seed が
        // 直前の消費履歴に依存しないことを確認する。
        manual_seed(1);
        for _ in 0..5 {
            with_global_rng(|r| r.next_u64());
        }

        manual_seed(7);
        let a: Vec<u64> = (0..8).map(|_| with_global_rng(|r| r.next_u64())).collect();

        manual_seed(7);
        let b: Vec<u64> = (0..8).map(|_| with_global_rng(|r| r.next_u64())).collect();

        assert_eq!(a, b);
    }

    #[test]
    fn zero_seed_is_corrected() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(0);
        let first = with_global_rng(|r| r.next_u64());
        assert_ne!(first, 0);
    }

    #[test]
    fn different_seed_diverges() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(1);
        let a = with_global_rng(|r| r.next_u64());
        manual_seed(2);
        let b = with_global_rng(|r| r.next_u64());
        assert_ne!(a, b);
    }
}
