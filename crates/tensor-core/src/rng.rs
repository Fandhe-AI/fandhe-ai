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
//! 側乱数生成カーネルは対象外。`randn`／`rand`／`randint` は本イシュー
//! （#1725）で実装済み（下記 §実装記録）。
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

use std::fmt;
use std::sync::{Mutex, OnceLock};

use crate::error::ShapeError;
use crate::tensor::{Tensor, checked_numel_for};

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

    /// `[0, 1)` の範囲に収まる f32 を返す（[`rand`] が使用する。24bit 精度
    /// で f32 の仮数部に厳密に収まる。`next_f32` の `[-1, 1)` 版とは別に
    /// 用意する: `rand` の一様分布契約は PyTorch `torch.rand` と同じ
    /// `[0, 1)` であり、符号反転を経由させたくないため）。
    pub fn next_unit_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24bit
        bits as f32 / (1u32 << 24) as f32
    }

    /// `[0, 1)` の範囲に収まる f64 を返す（[`randn`] の Box–Muller 変換が
    /// 使用する。53bit 精度で f64 の仮数部に厳密に収まる。`randn` は
    /// `ln`／`sin`／`cos` を経由するため f32 精度では丸め誤差が目立ち
    /// やすく、中間計算を f64 で行う）。
    pub fn next_unit_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // 53bit
        bits as f64 / (1u64 << 53) as f64
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

/// [`randint`] 専用のエラー型。`shape` 起因の不整合（要素数オーバー
/// フロー等）は `Tensor::zeros` 等と同じ [`ShapeError`] へ委譲し
/// （[`From<ShapeError>`] 実装）、`low >= high` という範囲自体の不正は
/// `ShapeError` の対象外（shape 不整合限定。`crates/tensor-core/src/error.rs`
/// の `ShapeError` variant 一覧参照）であるため本型で新設する
/// （イシュー #1725）。
///
/// `#[non_exhaustive]`: `ShapeError` と同じ理由（公開 API 非破壊。
/// `.claude/rules/security.md`）で後続の検査項目追加に備える。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RngError {
    /// shape 起因の不整合（要素数積のオーバーフロー等）。
    Shape(ShapeError),
    /// `randint(low, high, ..)` の `low >= high`（空区間・逆転区間）。
    InvalidRange { low: i32, high: i32 },
}

impl From<ShapeError> for RngError {
    fn from(err: ShapeError) -> Self {
        RngError::Shape(err)
    }
}

impl fmt::Display for RngError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RngError::Shape(err) => write!(f, "{err}"),
            RngError::InvalidRange { low, high } => {
                write!(f, "randint の範囲が不正: low={low} >= high={high}")
            }
        }
    }
}

impl std::error::Error for RngError {}

/// 標準正規分布 `N(0, 1)` に従う乱数テンソルを生成する（PyTorch
/// `torch.randn` 相当。イシュー #1725）。
///
/// [`with_global_rng`] を通じてホスト側だけで値を生成する
/// （`BackendOps` を経由しない。#1602 本文の設計方針。デバイスへの
/// 反映は `Tape::var` 等の既存アップロード経路が担う——本関数自体は
/// デバイスに触れない）。乱数は微分不能な葉値であるため `Op`／`VJP`
/// は追加しない（`torch.randn` にも勾配は無い）。
///
/// # アルゴリズム・決定性の範囲
///
/// Box–Muller 変換（`f64` 中間計算）で 2 値ずつ生成し、`numel` が
/// 奇数の場合は最後の組の 2 値目を生成せずに切り捨てる（1 組につき
/// `u1`／`u2` の抽選は必ず行うため、グローバル RNG が消費する抽選回数
/// は `randn(&[5])` と `randn(&[6])` とで同一——`ceil(numel / 2)` 組。
/// 差が出るのは出力へ書き出す要素数のみで、抽選後の状態は両者で一致
/// する）。整数演算のみの
/// [`rand`]／[`randint`] と異なり `ln`／`sin`／`cos`（libm 経由）を使う
/// ため、本関数が保証する決定性は「同一プロセス・同一プラットフォーム
/// 内での再現」に限る（クロスプラットフォームでの最下位 bit 一致は
/// 契約しない。`docs/rng-global-contract-design.md`）。
///
/// # 用途限定
///
/// xorshift64* は暗号学的に安全な PRNG ではない（モジュール冒頭
/// コメント参照。OWASP A02）。
pub fn randn(shape: &[usize]) -> Result<Tensor<f32>, ShapeError> {
    let numel = checked_numel_for::<f32>(shape)?;
    let data = with_global_rng(|rng| {
        let mut out = Vec::with_capacity(numel);
        let mut remaining = numel;
        while remaining > 0 {
            // u1 は `(0, 1]` に補正して `ln(0)` （負の無限大）を避ける。
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
        out
    });
    Tensor::new(data, shape)
}

/// `[0, 1)` の一様分布に従う乱数テンソルを生成する（PyTorch
/// `torch.rand` 相当。イシュー #1725）。設計方針は [`randn`] と同じ
/// （ホスト生成のみ・`Op`／`VJP` なし）。24bit 精度の整数演算のみで
/// 構成されるため、プラットフォーム横断で bit 同一の決定性を持つ
/// （[`randn`] と異なりゴールデン値による回帰が可能。`tests` 参照）。
pub fn rand(shape: &[usize]) -> Result<Tensor<f32>, ShapeError> {
    let numel = checked_numel_for::<f32>(shape)?;
    let data = with_global_rng(|rng| (0..numel).map(|_| rng.next_unit_f32()).collect());
    Tensor::new(data, shape)
}

/// `[low, high)` の一様分布に従う整数乱数テンソルを生成する（PyTorch
/// `torch.randint` 相当。イシュー #1725）。
///
/// dtype は `i32`（本リポの index／targets 型契約——`Var::gather`／
/// `index_select`／`cross_entropy` 等——に合わせる。PyTorch 既定の
/// int64 とは異なる意図的な差異。`docs/rng-global-contract-design.md`）。
///
/// `low >= high`（空・逆転区間）は [`RngError::InvalidRange`] を返す。
/// 剰余バイアスを避けるため rejection sampling（`zone` 未満の値のみ
/// 採用）で一様性を保証する。`shape` の要素数積オーバーフロー・
/// アロケーション不能なバイトサイズ（`i32` 換算で `Vec` の allocation
/// 上限を超える shape）は乱数を一切消費せず [`RngError::Shape`] を
/// 返す（クレート内共通の `checked_numel_for` によるロック取得前の
/// 事前検査。OWASP A03/A04 相当の入力検証。イシュー #1725・PR #1815
/// codex-review P1 是正）。
pub fn randint(low: i32, high: i32, shape: &[usize]) -> Result<Tensor<i32>, RngError> {
    if low >= high {
        return Err(RngError::InvalidRange { low, high });
    }
    let numel = checked_numel_for::<i32>(shape)?;
    // `range` は `i64` 経由で計算する: `high - low` が `u64` として
    // 求まればよく、`i32::MIN..i32::MAX` の最大区間でもオーバーフロー
    // しない（`i32` 同士の減算だと `i32::MAX - i32::MIN` は overflow
    // する）。
    let range = (high as i64 - low as i64) as u64;
    // rejection sampling: `u64::MAX` を `range` で割った商の倍数
    // （`zone`）未満に収まる `x` のみを採用することで、`range` が
    // `u64::MAX + 1` を割り切らない場合に生じる剰余バイアス
    // （小さい余りほど出現しやすくなる偏り）を排除する。
    let zone = range.wrapping_mul(u64::MAX / range);
    let data = with_global_rng(|rng| {
        (0..numel)
            .map(|_| {
                loop {
                    let x = rng.next_u64();
                    if x < zone {
                        return low as i64 + (x % range) as i64;
                    }
                }
            })
            .map(|v| v as i32)
            .collect::<Vec<i32>>()
    });
    Ok(Tensor::new(data, shape)?)
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

    #[test]
    fn rand_produces_correct_shape_and_range() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(1);
        let t = rand(&[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        for i in 0..2 {
            for j in 0..3 {
                let v = t.get(&[i, j]).unwrap();
                assert!((0.0..1.0).contains(&v), "value out of [0, 1): {v}");
            }
        }
    }

    #[test]
    fn rand_is_deterministic_after_manual_seed() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(123);
        let a = rand(&[4]).unwrap();
        manual_seed(123);
        let b = rand(&[4]).unwrap();
        for i in 0..4 {
            assert_eq!(a.get(&[i]), b.get(&[i]));
        }
    }

    #[test]
    fn rand_golden_values_after_manual_seed_42() {
        // 24bit 整数演算のみで構成されるため、プラットフォーム横断で
        // bit 同一の決定性を持つ（モジュール doc「決定性の範囲」参照）。
        // 期待値は `rand` を経由せず `Xorshift64Star` を独立に直接
        // 駆動して算出し、`rand` の実装（`with_global_rng` 経由の一括
        // 生成）が `next_unit_f32` を要素順に呼ぶ契約と一致することを
        // 固定する（実装の自己参照を避けるため `manual_seed` のグローバル
        // 状態は使わない）。
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(42);
        let t = rand(&[4]).unwrap();

        let mut independent = Xorshift64Star::new(42);
        let expected: Vec<f32> = (0..4).map(|_| independent.next_unit_f32()).collect();

        for (i, exp) in expected.into_iter().enumerate() {
            assert_eq!(t.get(&[i]).unwrap(), exp);
        }
    }

    #[test]
    fn rand_does_not_consume_rng_for_empty_shape() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(5);
        let before = with_global_rng(|r| r.next_u64());
        manual_seed(5);
        let _empty = rand(&[0, 3]).unwrap();
        let after = with_global_rng(|r| r.next_u64());
        assert_eq!(before, after, "空 shape は乱数を消費しないはず");
    }

    #[test]
    fn rand_rejects_element_count_overflow() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let err = rand(&[usize::MAX, 2]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
    }

    /// `shape = &[usize::MAX]` は要素数積（`usize` 積）としては
    /// `usize::MAX` そのものでオーバーフローしないが、`f32`（4 バイト）
    /// 換算では `numel * 4` が `Vec` の allocation 上限（`isize::MAX`
    /// バイト）を超えるため、`Vec::with_capacity`／`collect` に到達すると
    /// capacity overflow で panic する。事前の `checked_numel_for` が
    /// バイトサイズ側も検査し、乱数を一切消費せず型付きエラーで拒否する
    /// ことを確認する（イシュー #1725・PR #1815 codex-review P1 是正。
    /// panic しないこと自体が本テストの主眼であり、そのままパニックすれば
    /// `cargo test` がテストプロセスごと落ちて失敗として検出される）。
    #[test]
    fn rand_rejects_capacity_overflow_without_consuming_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(11);
        let before = with_global_rng(|r| r.next_u64());
        manual_seed(11);
        let err = rand(&[usize::MAX]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
        let after = with_global_rng(|r| r.next_u64());
        assert_eq!(before, after, "確保不能な shape は乱数を消費しないはず");
    }

    #[test]
    fn randn_produces_correct_shape_and_finite_values() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(1);
        let t = randn(&[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                let v = t.get(&[i, j]).unwrap();
                assert!(v.is_finite(), "randn produced non-finite value: {v}");
            }
        }
    }

    /// `rand_rejects_capacity_overflow_without_consuming_rng` と同じ理由
    /// （`f32` 4 バイト換算のバイトサイズ超過）で `randn` も
    /// `shape = &[usize::MAX]` を型付きエラーで拒否し、`Box–Muller` の
    /// `ln`／`sin`／`cos` に到達する前に乱数消費なしで弾くことを確認する。
    #[test]
    fn randn_rejects_capacity_overflow_without_consuming_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(12);
        let before = with_global_rng(|r| r.next_u64());
        manual_seed(12);
        let err = randn(&[usize::MAX]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
        let after = with_global_rng(|r| r.next_u64());
        assert_eq!(before, after, "確保不能な shape は乱数を消費しないはず");
    }

    #[test]
    fn randn_is_deterministic_within_same_process() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(99);
        let a = randn(&[6]).unwrap();
        manual_seed(99);
        let b = randn(&[6]).unwrap();
        for i in 0..6 {
            assert_eq!(a.get(&[i]), b.get(&[i]));
        }
    }

    #[test]
    fn randn_reshaped_matches_flat_generation() {
        // `randn(&[2,3])` と `randn(&[6])` が同一データになることを
        // 確認する（モジュール doc の契約。要素消費順が shape に依存
        // しないことの裏付け）。
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(7);
        let a = randn(&[2, 3]).unwrap();
        manual_seed(7);
        let b = randn(&[6]).unwrap();
        for i in 0..6 {
            let row = i / 3;
            let col = i % 3;
            assert_eq!(a.get(&[row, col]).unwrap(), b.get(&[i]).unwrap());
        }
    }

    #[test]
    fn randn_handles_odd_numel_without_extra_rng_consumption() {
        // 奇数 numel（[5]）は最後の組の 2 値目を生成せずに切り捨てる
        // 契約: 1 組につき u1/u2 の抽選は必ず行うため、`randn(&[5])` と
        // `randn(&[6])` は同数（3 組）の抽選を消費し、抽選後のグローバル
        // RNG 状態は完全に一致する（モジュール doc 参照）。
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(13);
        let five = randn(&[5]).unwrap();
        let state_after_five = with_global_rng(|r| r.next_u64());

        manual_seed(13);
        let six = randn(&[6]).unwrap();
        let state_after_six = with_global_rng(|r| r.next_u64());

        for i in 0..5 {
            assert_eq!(five.get(&[i]).unwrap(), six.get(&[i]).unwrap());
        }
        assert_eq!(
            state_after_five, state_after_six,
            "randn(&[5]) と randn(&[6]) は同数の抽選を消費し、抽選後の RNG 状態は一致するはず"
        );
    }

    #[test]
    fn randn_large_sample_has_roughly_standard_normal_statistics() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(2024);
        let n = 100_000usize;
        let t = randn(&[n]).unwrap();
        let sum: f64 = (0..n).map(|i| t.get(&[i]).unwrap() as f64).sum();
        let mean = sum / n as f64;
        let var: f64 = (0..n)
            .map(|i| {
                let d = t.get(&[i]).unwrap() as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        assert!(mean.abs() < 0.02, "mean out of range: {mean}");
        assert!((0.95..=1.05).contains(&var), "variance out of range: {var}");
    }

    #[test]
    fn randint_values_are_within_half_open_range() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(1);
        let t = randint(-3, 2, &[2000]).unwrap();
        for i in 0..2000 {
            let v = t.get(&[i]).unwrap();
            assert!((-3..2).contains(&v), "value out of range: {v}");
        }
    }

    #[test]
    fn randint_small_range_covers_all_values_given_enough_samples() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(2);
        let t = randint(-3, 2, &[2000]).unwrap();
        let mut seen = [false; 5];
        for i in 0..2000 {
            let v = t.get(&[i]).unwrap();
            seen[(v - (-3)) as usize] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "not all values in range appeared: {seen:?}"
        );
    }

    #[test]
    fn randint_rejects_empty_and_reversed_range() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        assert_eq!(
            randint(2, 2, &[1]).unwrap_err(),
            RngError::InvalidRange { low: 2, high: 2 }
        );
        assert_eq!(
            randint(5, 2, &[1]).unwrap_err(),
            RngError::InvalidRange { low: 5, high: 2 }
        );
    }

    #[test]
    fn randint_handles_full_i32_range_without_overflow() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(1);
        let t = randint(i32::MIN, i32::MAX, &[16]).unwrap();
        for i in 0..16 {
            let v = t.get(&[i]).unwrap();
            // `low` が `i32::MIN` のため下限検査は常に真（clippy
            // `absurd_extreme_comparisons` の対象）で意味を持たない。
            // 生成範囲の意味ある境界（`high=i32::MAX` は排他的上限）
            // のみを検査する。
            assert!(v < i32::MAX);
        }
    }

    #[test]
    fn randint_rejects_element_count_overflow_without_consuming_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(9);
        let before = with_global_rng(|r| r.next_u64());
        manual_seed(9);
        let err = randint(0, 10, &[usize::MAX, 2]).unwrap_err();
        assert!(matches!(
            err,
            RngError::Shape(ShapeError::ElementCountOverflow)
        ));
        let after = with_global_rng(|r| r.next_u64());
        assert_eq!(before, after);
    }

    /// `rand_rejects_capacity_overflow_without_consuming_rng` と同じ理由
    /// （`i32` 4 バイト換算のバイトサイズ超過）で `randint` も
    /// `shape = &[usize::MAX]` を型付きエラーで拒否することを確認する。
    #[test]
    fn randint_rejects_capacity_overflow_without_consuming_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(13);
        let before = with_global_rng(|r| r.next_u64());
        manual_seed(13);
        let err = randint(0, 10, &[usize::MAX]).unwrap_err();
        assert!(matches!(
            err,
            RngError::Shape(ShapeError::ElementCountOverflow)
        ));
        let after = with_global_rng(|r| r.next_u64());
        assert_eq!(before, after);
    }

    #[test]
    fn rng_error_from_shape_error_and_display() {
        let err: RngError = ShapeError::ElementCountOverflow.into();
        assert!(matches!(
            err,
            RngError::Shape(ShapeError::ElementCountOverflow)
        ));
        assert!(!err.to_string().is_empty());

        let range_err = RngError::InvalidRange { low: 3, high: 1 };
        assert!(range_err.to_string().contains("3"));
    }
}
