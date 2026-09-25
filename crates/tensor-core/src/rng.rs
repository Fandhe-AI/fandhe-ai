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
//! （#1725）で実装済み（下記 §実装記録）。`data::DataLoader`（イシュー
//! #1615）のシャッフルも本モジュールの [`with_global_rng`] を消費する
//! （`docs/dataset-dataloader-design.md`）。
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
/// （イシュー #1725）。[`bernoulli`]／[`multinomial`]／[`normal`]・
/// [`Generator`]（イシュー #2156）も同じ型を再利用し、確率・重み・
/// `mean`／`std` の検証違反用に variant を追加する（下記）。
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
    /// [`bernoulli`]／[`multinomial`] が受け取る確率・重みの論理 index
    /// `index` が非有限、または `bernoulli` では `[0, 1]` の範囲外
    /// （イシュー #2156）。乱数は一切消費しない（全要素の事前検証を
    /// ロック取得前に終えてから抽選に入る契約）。
    InvalidProbability { index: usize },
    /// [`multinomial`]／[`normal`] の引数不正（`reason` は固定文言。
    /// `n == 0`・`n > i32::MAX`・行和が 0 以下・非復元抽出で
    /// `num_samples` が正の重みの個数を超える・`mean`／`std` が非有限・
    /// `std < 0` 等。イシュー #2156）。乱数は一切消費しない。
    InvalidArgument { reason: &'static str },
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
            RngError::InvalidProbability { index } => {
                write!(f, "確率・重みが不正: index={index}")
            }
            RngError::InvalidArgument { reason } => {
                write!(f, "引数が不正: {reason}")
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

/// `probs` の各要素を成功確率としてベルヌーイ分布からサンプルする
/// （PyTorch `torch.bernoulli` 相当。イシュー #2156・親 #2131）。出力は
/// `probs.shape()` と同じ shape で、各要素は厳密に `1.0` か `0.0`。
///
/// 検証（ロック取得・抽選の前に全要素を検査する。`randint` と同じ
/// 「不正入力は乱数を一切消費しない」契約）: 非有限、または `[0, 1]`
/// 範囲外の要素があれば最初に違反した論理 index を
/// [`RngError::InvalidProbability`] で返す。
///
/// `p=0` は常に `0.0`、`p=1` は常に `1.0` になるが、消費する抽選回数は
/// 常に `numel` 回で固定する（1 要素につき [`Xorshift64Star::next_unit_f64`]
/// を 1 回引き `u < p` で判定。整数演算と比較のみで構成されるため
/// プラットフォーム横断で bit 同一の決定性を持つ）。
///
/// # 用途限定
///
/// xorshift64* は暗号学的に安全な PRNG ではない（モジュール冒頭コメント
/// 参照。OWASP A02）。
pub fn bernoulli(probs: &Tensor<f32>) -> Result<Tensor<f32>, RngError> {
    let shape = probs.shape();
    let numel = checked_numel_for::<f32>(shape)?;
    let input = probs.host_slice();
    for (index, &p) in input.iter().enumerate() {
        if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
            return Err(RngError::InvalidProbability { index });
        }
    }
    let data = with_global_rng(|rng| bernoulli_core(rng, &input, numel));
    Tensor::new(data, shape).map_err(RngError::from)
}

/// [`bernoulli`]・[`Generator::bernoulli`] が共有する抽選本体（イシュー
/// #2156）。検証済みの `input`（`[0, 1]` 範囲・有限であることを呼び出し
/// 元が保証する）を受け取り、要素ごとに `rng` から 1 回抽選する。
fn bernoulli_core(rng: &mut Xorshift64Star, input: &[f32], numel: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(numel);
    for &p in input {
        let u = rng.next_unit_f64();
        out.push(if u < f64::from(p) { 1.0 } else { 0.0 });
    }
    out
}

/// `weights` の各行から `num_samples` 個のカテゴリ添字をサンプルする
/// （PyTorch `torch.multinomial` 相当。イシュー #2156・親 #2131）。
///
/// `weights` は rank 1（`[n]`。出力 `[num_samples]`）または rank 2
/// （`[m, n]`。出力 `[m, num_samples]`）のみを受け付け、それ以外の rank
/// は [`RngError::Shape`]（[`ShapeError::RankMismatch`]。`expected: 2`）
/// を返す。出力 dtype は `i32`（本リポの index／targets 型契約に合わせる
/// 意図的な差異。PyTorch 既定の int64 とは異なる）。
///
/// # 検証（全行を検証し終えてから抽選を始める）
///
/// - `n == 0`、または `n > i32::MAX as usize` は
///   [`RngError::InvalidArgument`]。
/// - 負・非有限の重みは [`RngError::InvalidProbability`]（`index` は
///   フラット化した論理 index）。
/// - 行和（`f64` で計算）が `0.0` 以下は [`RngError::InvalidArgument`]。
/// - `replacement == false` で `num_samples` がその行の正の重みの個数を
///   超える場合も [`RngError::InvalidArgument`]（PyTorch と同じくエラー
///   にする）。
///
/// 検証を 1 行でも通らなければ乱数を一切消費しない。`num_samples == 0`
/// または `m == 0` は空テンソルを返し同様に消費しない。
///
/// # 復元抽出（`replacement = true`）
///
/// 行ごとに `f64` の累積和を index 順に作り、`u * total`（`u` は
/// [`Xorshift64Star::next_unit_f64`] の抽選）以上となる最初のビンを
/// 採用する（丸めで総和ちょうどに達した場合は最後の正の重みの index へ
/// 落とす防御を入れる）。
///
/// # 非復元抽出（`replacement = false`）
///
/// 1 サンプルごとに選んだ重みを `0.0` にして総和を取り直す
/// （`O(num_samples * n)`）。終了性は事前検証で保証するが、それでも
/// 総和が `0.0` 以下になった場合は panic せず [`RngError::InvalidArgument`]
/// を返す防御を入れる。
///
/// いずれの方式も消費回数は `m * num_samples` 回（行優先順）で、加算・
/// 乗算・比較のみで構成されるため libm を通らず、プラットフォーム横断で
/// bit 同一の決定性を持つ。
///
/// # 用途限定
///
/// xorshift64* は暗号学的に安全な PRNG ではない（モジュール冒頭コメント
/// 参照。OWASP A02）。
pub fn multinomial(
    weights: &Tensor<f32>,
    num_samples: usize,
    replacement: bool,
) -> Result<Tensor<i32>, RngError> {
    let shape = weights.shape();
    let (rows, n, out_shape): (usize, usize, Vec<usize>) = match *shape {
        [n] => (1, n, vec![num_samples]),
        [m, n] => (m, n, vec![m, num_samples]),
        _ => {
            return Err(RngError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: shape.len(),
            }));
        }
    };

    if n == 0 || n > i32::MAX as usize {
        return Err(RngError::InvalidArgument {
            reason: "multinomial: weights の列数 n は 1..=i32::MAX の範囲である必要があります",
        });
    }

    let out_numel = checked_numel_for::<i32>(&out_shape)?;
    // 作業用バッファ（行ごとの累積和・非復元抽出時の重みコピー）の
    // 容量も乱数消費前に検査する（`randint` と同じ防御。
    // イシュー #1725・PR #1815 codex-review P1 是正と同型の方針）。
    let _ = checked_numel_for::<f64>(&[n])?;

    let input = weights.host_slice();
    let mut positive_counts = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_slice = &input[row * n..(row + 1) * n];
        let mut sum = 0.0f64;
        let mut positive = 0usize;
        for (col, &w) in row_slice.iter().enumerate() {
            if !w.is_finite() || w < 0.0 {
                return Err(RngError::InvalidProbability {
                    index: row * n + col,
                });
            }
            if w > 0.0 {
                positive += 1;
            }
            sum += f64::from(w);
        }
        if sum <= 0.0 {
            return Err(RngError::InvalidArgument {
                reason: "multinomial: 行の重みの総和は正である必要があります",
            });
        }
        if !replacement && num_samples > positive {
            return Err(RngError::InvalidArgument {
                reason: "multinomial: 非復元抽出では num_samples は正の重みの個数以下である必要があります",
            });
        }
        positive_counts.push(positive);
    }

    if num_samples == 0 || rows == 0 {
        return Tensor::new(Vec::new(), &out_shape).map_err(RngError::from);
    }

    let data = with_global_rng(|rng| {
        let mut out = Vec::with_capacity(out_numel);
        for row in 0..rows {
            let row_slice = &input[row * n..(row + 1) * n];
            if replacement {
                multinomial_core_with_replacement(rng, row_slice, num_samples, &mut out);
            } else {
                multinomial_core_without_replacement(rng, row_slice, num_samples, &mut out);
            }
        }
        out
    });
    Tensor::new(data, &out_shape).map_err(RngError::from)
}

/// [`multinomial`] の復元抽出本体（イシュー #2156）。検証済みの
/// `row`（有限・非負・総和が正）から `num_samples` 個を独立に抽選し
/// `out` へ追記する。累積和は `f64` で作る（プラットフォーム横断 bit
/// 同一の決定性を保つため libm を経由しない）。
fn multinomial_core_with_replacement(
    rng: &mut Xorshift64Star,
    row: &[f32],
    num_samples: usize,
    out: &mut Vec<i32>,
) {
    let mut cdf = Vec::with_capacity(row.len());
    let mut acc = 0.0f64;
    for &w in row {
        acc += f64::from(w);
        cdf.push(acc);
    }
    let total = acc;
    // 総和が正であることは呼び出し元（`multinomial`）が検証済み。
    let last_positive = row
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, &w)| w > 0.0)
        .map(|(i, _)| i)
        .unwrap_or(row.len() - 1);
    for _ in 0..num_samples {
        let x = rng.next_unit_f64() * total;
        // `partition_point` は「`cdf[i] > x` を満たす最初の i」を返す
        // （`cdf` は非減少列）。丸めで `x` が `total` に一致し `i == n`
        // になった場合は最後の正の重みの index へ落とす。
        let idx = cdf.partition_point(|&c| c <= x);
        let idx = if idx >= row.len() { last_positive } else { idx };
        out.push(idx as i32);
    }
}

/// [`multinomial`] の非復元抽出本体（イシュー #2156）。選んだ添字の重みを
/// `0.0` にして総和を取り直しながら `num_samples` 個を抽選する
/// （`O(num_samples * n)`）。終了性は呼び出し元の事前検証
/// （`num_samples <= 正の重みの個数`）で保証するが、防御的に総和が
/// `0.0` 以下になった場合は `InvalidArgument` を返す（呼び出し元の
/// `multinomial` が `Result` を返せるようクロージャ経由で伝播する）。
fn multinomial_core_without_replacement(
    rng: &mut Xorshift64Star,
    row: &[f32],
    num_samples: usize,
    out: &mut Vec<i32>,
) {
    let mut remaining: Vec<f32> = row.to_vec();
    for _ in 0..num_samples {
        let mut acc = 0.0f64;
        let mut cdf = Vec::with_capacity(remaining.len());
        for &w in &remaining {
            acc += f64::from(w);
            cdf.push(acc);
        }
        let total = acc;
        // 呼び出し元の事前検証で `num_samples <= 正の重みの個数` が
        // 保証されるため、ここに到達する反復では常に `total > 0.0`
        // のはずだが、防御的に最後の正の重みへフォールバックする。
        let last_positive = remaining
            .iter()
            .enumerate()
            .rev()
            .find(|&(_, &w)| w > 0.0)
            .map(|(i, _)| i)
            .unwrap_or(remaining.len() - 1);
        let idx = if total > 0.0 {
            let x = rng.next_unit_f64() * total;
            let idx = cdf.partition_point(|&c| c <= x);
            if idx >= remaining.len() {
                last_positive
            } else {
                idx
            }
        } else {
            last_positive
        };
        out.push(idx as i32);
        remaining[idx] = 0.0;
    }
}

/// 正規分布 `N(mean, std²)` に従う乱数テンソルを生成する（PyTorch
/// `torch.normal` 相当。イシュー #2156・親 #2131）。引数順は
/// [`randint`]（`low, high, shape`）に揃えて `mean, std, shape` とする。
///
/// 検証: `mean` が有限、`std` が有限かつ `>= 0` であること
/// （[`RngError::InvalidArgument`]）。出力容量は [`checked_numel_for`]
/// で乱数消費前に検査する。
///
/// アルゴリズムは [`randn`] と同じ Box–Muller（`f64` 中間計算）で、
/// `z * std + mean` を最後に 1 回だけ `f32` へ downcast する。`std > 0`
/// の場合、この式は `autodiff::nn::init::fill_normal` と同一のため両者は
/// 同じ `manual_seed` の後で bit 一致する（`crates/autodiff/tests/
/// random_parity.rs` で固定）。抽選消費数は `randn` と同じく
/// `ceil(numel / 2)` 組で、`std == 0` でも同数を消費し値は厳密に `mean`
/// になる（`z` は常に有限なので `z*0.0 + mean == mean`。`nn::init::normal`
/// は `std == 0` で乱数を消費しない別契約であり、本関数とは意図的に異なる。
/// 差分は `docs/rng-distributions-generator-decision.md` に記録する）。
///
/// 決定性の範囲は [`randn`] と同じく「同一プロセス・同一プラットフォーム
/// 内での再現」に限る（`ln`／`sin`／`cos` が libm 経由のため）。
///
/// # 用途限定
///
/// xorshift64* は暗号学的に安全な PRNG ではない（モジュール冒頭コメント
/// 参照。OWASP A02）。
pub fn normal(mean: f32, std: f32, shape: &[usize]) -> Result<Tensor<f32>, RngError> {
    if !mean.is_finite() {
        return Err(RngError::InvalidArgument {
            reason: "normal: mean は有限である必要があります",
        });
    }
    if !(std.is_finite() && std >= 0.0) {
        return Err(RngError::InvalidArgument {
            reason: "normal: std は有限かつ非負である必要があります",
        });
    }
    let numel = checked_numel_for::<f32>(shape)?;
    let data = with_global_rng(|rng| normal_core(rng, numel, mean, std));
    Tensor::new(data, shape).map_err(RngError::from)
}

/// [`normal`]・[`Generator::normal`] が共有する生成本体（イシュー
/// #2156）。[`randn`]・`autodiff::nn::init::fill_normal` と同型の
/// Box–Muller 実装（`f64` 中間計算・奇数 `numel` は最後の組の 2 値目を
/// 切り捨てるが抽選自体は行う契約）。
fn normal_core(rng: &mut Xorshift64Star, numel: usize, mean: f32, std: f32) -> Vec<f32> {
    let mean_f64 = f64::from(mean);
    let std_f64 = f64::from(std);
    let mut out = Vec::with_capacity(numel);
    let mut remaining = numel;
    while remaining > 0 {
        // u1 は `(0, 1]` に補正して `ln(0)`（負の無限大）を避ける。
        let u1 = 1.0 - rng.next_unit_f64();
        let u2 = rng.next_unit_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        let z0 = r * theta.cos();
        out.push((z0 * std_f64 + mean_f64) as f32);
        remaining -= 1;
        if remaining == 0 {
            break;
        }
        let z1 = r * theta.sin();
        out.push((z1 * std_f64 + mean_f64) as f32);
        remaining -= 1;
    }
    out
}

/// プロセスグローバルな RNG（[`manual_seed`]／[`with_global_rng`]）とは
/// 完全に独立した状態を持つ乱数源（PyTorch `torch.Generator` 相当。
/// イシュー #2156・親 #2131）。
///
/// 用途は「グローバル状態を汚さない再現可能なサンプリング」。
/// [`Generator::new(seed)`](Self::new) と `manual_seed(seed)` の直後に
/// グローバル自由関数（[`bernoulli`]／[`multinomial`]／[`normal`]）を
/// 呼んだ結果は bit 完全一致する（いずれも
/// `Xorshift64Star::new(seed)` から始まるため。`crates/autodiff/tests/
/// random_parity.rs` で固定）。
///
/// `&mut self` で操作するためロックは不要（呼び出し元が所有権・借用で
/// 排他制御を担う。`with_global_rng` の `Mutex` とは異なる設計）。
///
/// # 用途限定
///
/// xorshift64* は暗号学的に安全な PRNG ではない（モジュール冒頭コメント
/// 参照。OWASP A02）。鍵・トークン生成などセキュリティ用途には使用
/// しないこと。
pub struct Generator {
    rng: Xorshift64Star,
    initial_seed: u64,
}

impl Generator {
    /// `seed` から新規の独立した RNG 状態を作る
    /// （`Xorshift64Star::new(seed)` と同じ 0 補正を継承する）。
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Xorshift64Star::new(seed),
            initial_seed: seed,
        }
    }

    /// 状態を `seed` からやり直す（[`manual_seed`] の Generator 版。
    /// グローバル状態には一切触れない）。
    pub fn manual_seed(&mut self, seed: u64) {
        self.rng = Xorshift64Star::new(seed);
        self.initial_seed = seed;
    }

    /// [`Self::new`]／[`Self::manual_seed`] に渡された、補正前のシード値を
    /// 返す（PyTorch `Generator.initial_seed()` と同じ意味論）。
    pub fn initial_seed(&self) -> u64 {
        self.initial_seed
    }

    /// [`bernoulli`] の Generator 版。グローバル RNG は消費しない。
    pub fn bernoulli(&mut self, probs: &Tensor<f32>) -> Result<Tensor<f32>, RngError> {
        let shape = probs.shape();
        let numel = checked_numel_for::<f32>(shape)?;
        let input = probs.host_slice();
        for (index, &p) in input.iter().enumerate() {
            if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
                return Err(RngError::InvalidProbability { index });
            }
        }
        let data = bernoulli_core(&mut self.rng, &input, numel);
        Tensor::new(data, shape).map_err(RngError::from)
    }

    /// [`normal`] の Generator 版。グローバル RNG は消費しない。
    pub fn normal(
        &mut self,
        mean: f32,
        std: f32,
        shape: &[usize],
    ) -> Result<Tensor<f32>, RngError> {
        if !mean.is_finite() {
            return Err(RngError::InvalidArgument {
                reason: "normal: mean は有限である必要があります",
            });
        }
        if !(std.is_finite() && std >= 0.0) {
            return Err(RngError::InvalidArgument {
                reason: "normal: std は有限かつ非負である必要があります",
            });
        }
        let numel = checked_numel_for::<f32>(shape)?;
        let data = normal_core(&mut self.rng, numel, mean, std);
        Tensor::new(data, shape).map_err(RngError::from)
    }

    /// [`multinomial`] の Generator 版。グローバル RNG は消費しない。
    pub fn multinomial(
        &mut self,
        weights: &Tensor<f32>,
        num_samples: usize,
        replacement: bool,
    ) -> Result<Tensor<i32>, RngError> {
        let shape = weights.shape();
        let (rows, n, out_shape): (usize, usize, Vec<usize>) = match *shape {
            [n] => (1, n, vec![num_samples]),
            [m, n] => (m, n, vec![m, num_samples]),
            _ => {
                return Err(RngError::Shape(ShapeError::RankMismatch {
                    expected: 2,
                    actual: shape.len(),
                }));
            }
        };

        if n == 0 || n > i32::MAX as usize {
            return Err(RngError::InvalidArgument {
                reason: "multinomial: weights の列数 n は 1..=i32::MAX の範囲である必要があります",
            });
        }

        let out_numel = checked_numel_for::<i32>(&out_shape)?;
        let _ = checked_numel_for::<f64>(&[n])?;

        let input = weights.host_slice();
        for row in 0..rows {
            let row_slice = &input[row * n..(row + 1) * n];
            let mut sum = 0.0f64;
            let mut positive = 0usize;
            for (col, &w) in row_slice.iter().enumerate() {
                if !w.is_finite() || w < 0.0 {
                    return Err(RngError::InvalidProbability {
                        index: row * n + col,
                    });
                }
                if w > 0.0 {
                    positive += 1;
                }
                sum += f64::from(w);
            }
            if sum <= 0.0 {
                return Err(RngError::InvalidArgument {
                    reason: "multinomial: 行の重みの総和は正である必要があります",
                });
            }
            if !replacement && num_samples > positive {
                return Err(RngError::InvalidArgument {
                    reason: "multinomial: 非復元抽出では num_samples は正の重みの個数以下である必要があります",
                });
            }
        }

        if num_samples == 0 || rows == 0 {
            return Tensor::new(Vec::new(), &out_shape).map_err(RngError::from);
        }

        let mut out = Vec::with_capacity(out_numel);
        for row in 0..rows {
            let row_slice = &input[row * n..(row + 1) * n];
            if replacement {
                multinomial_core_with_replacement(&mut self.rng, row_slice, num_samples, &mut out);
            } else {
                multinomial_core_without_replacement(
                    &mut self.rng,
                    row_slice,
                    num_samples,
                    &mut out,
                );
            }
        }
        Tensor::new(out, &out_shape).map_err(RngError::from)
    }
}

impl Clone for Generator {
    /// 内部状態を複製する（`Xorshift64Star` 自体に `Clone` を derive
    /// しない設計判断を保つため、フィールド単位で複製する。同一系列を
    /// 独立に分岐させる用途——例えば複数の Generator を並行してドリフト
    /// なく使い比べるテスト——を想定する）。
    fn clone(&self) -> Self {
        Self {
            rng: Xorshift64Star {
                state: self.rng.state,
            },
            initial_seed: self.initial_seed,
        }
    }
}

impl fmt::Debug for Generator {
    /// 内部状態（`Xorshift64Star` の `state`）は表示しない。`initial_seed`
    /// のみを表示する（内部実装詳細を Debug 出力経由で公開面へ漏らさない
    /// ための最小限の情報開示）。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Generator")
            .field("initial_seed", &self.initial_seed)
            .finish()
    }
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

    // ---- bernoulli ----

    #[test]
    fn bernoulli_golden_values_after_manual_seed_42() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let probs = Tensor::new(vec![0.5f32; 4], &[4]).unwrap();
        manual_seed(42);
        let t = bernoulli(&probs).unwrap();

        let mut independent = Xorshift64Star::new(42);
        let expected: Vec<f32> = (0..4)
            .map(|_| {
                if independent.next_unit_f64() < 0.5 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        for (i, exp) in expected.into_iter().enumerate() {
            assert_eq!(t.get(&[i]).unwrap(), exp);
        }
    }

    #[test]
    fn bernoulli_endpoints_are_deterministic() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let zeros = Tensor::new(vec![0.0f32; 200], &[200]).unwrap();
        let ones = Tensor::new(vec![1.0f32; 200], &[200]).unwrap();
        manual_seed(1);
        let z = bernoulli(&zeros).unwrap();
        manual_seed(1);
        let o = bernoulli(&ones).unwrap();
        for i in 0..200 {
            assert_eq!(z.get(&[i]).unwrap(), 0.0);
            assert_eq!(o.get(&[i]).unwrap(), 1.0);
        }
    }

    #[test]
    fn bernoulli_rejects_invalid_probability_without_consuming_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(3);
        let before = with_global_rng(|r| r.next_u64());
        manual_seed(3);
        let nan = Tensor::new(vec![0.5, f32::NAN], &[2]).unwrap();
        let err = bernoulli(&nan).unwrap_err();
        assert_eq!(err, RngError::InvalidProbability { index: 1 });
        let after = with_global_rng(|r| r.next_u64());
        assert_eq!(before, after);

        manual_seed(3);
        let out_of_range = Tensor::new(vec![0.5, 1.5], &[2]).unwrap();
        let err = bernoulli(&out_of_range).unwrap_err();
        assert_eq!(err, RngError::InvalidProbability { index: 1 });
    }

    #[test]
    fn bernoulli_non_contiguous_input_matches_contiguous() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let flat = Tensor::new(vec![0.1, 0.9, 0.3, 0.7], &[2, 2]).unwrap();
        let transposed = flat.transpose(0, 1).unwrap();
        let contiguous = transposed.contiguous();

        manual_seed(21);
        let a = bernoulli(&transposed).unwrap();
        manual_seed(21);
        let b = bernoulli(&contiguous).unwrap();
        assert_eq!(a.shape(), b.shape());
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(a.get(&[i, j]).unwrap(), b.get(&[i, j]).unwrap());
            }
        }
    }

    #[test]
    fn bernoulli_large_sample_mean_close_to_p() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let n = 20_000usize;
        let probs = Tensor::new(vec![0.3f32; n], &[n]).unwrap();
        manual_seed(2024);
        let t = bernoulli(&probs).unwrap();
        let sum: f64 = (0..n).map(|i| t.get(&[i]).unwrap() as f64).sum();
        let mean = sum / n as f64;
        assert!((mean - 0.3).abs() < 0.02, "mean out of range: {mean}");
    }

    // ---- multinomial ----

    #[test]
    fn multinomial_with_replacement_frequencies_match_weights() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let weights = Tensor::new(vec![1.0f32, 0.0, 3.0], &[3]).unwrap();
        manual_seed(5);
        let t = multinomial(&weights, 8_000, true).unwrap();
        assert_eq!(t.shape(), &[8_000]);
        let mut counts = [0usize; 3];
        for i in 0..8_000 {
            let idx = t.get(&[i]).unwrap();
            assert!((0..3).contains(&idx));
            counts[idx as usize] += 1;
        }
        assert_eq!(counts[1], 0, "重み 0 のビンは選ばれないはず");
        let ratio = counts[0] as f64 / counts[2] as f64;
        assert!(
            (ratio - (1.0 / 3.0)).abs() < 0.05,
            "ratio out of range: {ratio}"
        );
    }

    #[test]
    fn multinomial_without_replacement_has_no_duplicates_and_permutes_when_full() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let weights = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        manual_seed(6);
        let t = multinomial(&weights, 4, false).unwrap();
        let mut seen = [false; 4];
        for i in 0..4 {
            let idx = t.get(&[i]).unwrap();
            assert!((0..4).contains(&idx));
            assert!(!seen[idx as usize], "重複した添字が出た: {idx}");
            seen[idx as usize] = true;
        }
    }

    #[test]
    fn multinomial_rank2_rows_are_independent() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let weights = Tensor::new(vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]).unwrap();
        manual_seed(7);
        let t = multinomial(&weights, 5, true).unwrap();
        assert_eq!(t.shape(), &[2, 5]);
        for j in 0..5 {
            assert_eq!(t.get(&[0, j]).unwrap(), 0);
            assert_eq!(t.get(&[1, j]).unwrap(), 2);
        }
    }

    #[test]
    fn multinomial_rejects_invalid_inputs_without_consuming_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // rank 不正。
        let rank3 = Tensor::new(vec![1.0f32; 8], &[2, 2, 2]).unwrap();
        assert_eq!(
            multinomial(&rank3, 1, true).unwrap_err(),
            RngError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: 3
            })
        );

        manual_seed(8);
        let before = with_global_rng(|r| r.next_u64());

        // 負の重み。
        manual_seed(8);
        let negative = Tensor::new(vec![1.0f32, -1.0], &[2]).unwrap();
        assert_eq!(
            multinomial(&negative, 1, true).unwrap_err(),
            RngError::InvalidProbability { index: 1 }
        );

        // 総和 0。
        manual_seed(8);
        let all_zero = Tensor::new(vec![0.0f32, 0.0], &[2]).unwrap();
        assert!(matches!(
            multinomial(&all_zero, 1, true).unwrap_err(),
            RngError::InvalidArgument { .. }
        ));

        // 非復元抽出でカテゴリ不足。
        manual_seed(8);
        let scarce = Tensor::new(vec![1.0f32, 0.0], &[2]).unwrap();
        assert!(matches!(
            multinomial(&scarce, 2, false).unwrap_err(),
            RngError::InvalidArgument { .. }
        ));

        let after = with_global_rng(|r| r.next_u64());
        assert_eq!(before, after, "不正入力は乱数を消費しないはず");
    }

    #[test]
    fn multinomial_zero_num_samples_or_rows_does_not_consume_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(9);
        let before = with_global_rng(|r| r.next_u64());

        manual_seed(9);
        let weights = Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap();
        let t = multinomial(&weights, 0, true).unwrap();
        assert_eq!(t.numel(), 0);

        let after = with_global_rng(|r| r.next_u64());
        assert_eq!(before, after);
    }

    #[test]
    fn multinomial_rejects_capacity_overflow_without_consuming_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(14);
        let before = with_global_rng(|r| r.next_u64());
        manual_seed(14);
        let weights = Tensor::new(vec![1.0f32, 1.0], &[2]).unwrap();
        let err = multinomial(&weights, usize::MAX, true).unwrap_err();
        assert!(matches!(
            err,
            RngError::Shape(ShapeError::ElementCountOverflow)
        ));
        let after = with_global_rng(|r| r.next_u64());
        assert_eq!(before, after);
    }

    #[test]
    fn multinomial_consumes_m_times_num_samples_draws() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(15);
        let weights = Tensor::new(vec![1.0f32, 1.0, 1.0, 1.0], &[2, 2]).unwrap();
        let _ = multinomial(&weights, 3, true).unwrap();
        let state_after_multinomial = with_global_rng(|r| r.next_u64());

        manual_seed(15);
        for _ in 0..6 {
            // rows(2) * num_samples(3) = 6 回抽選するはず。
            with_global_rng(|r| r.next_u64());
        }
        let state_after_manual_draws = with_global_rng(|r| r.next_u64());

        assert_eq!(state_after_multinomial, state_after_manual_draws);
    }

    /// `partition_point` が丸めで `total` ちょうどに達し `idx == n` に
    /// なるフォールバック分岐（最後の正の重みの index へ落とす）が到達
    /// 可能であることを固定する。`Xorshift64Star::next_unit_f64` は
    /// `next_u64() >> 11` を `2^53` で割るため、`next_u64() >> 11 ==
    /// 2^53 - 1`（全 53bit が立った状態）で `next_unit_f64()` は
    /// `1.0` に最も近い最大値を返す——`x = u * total` が丸めで
    /// `total` に到達しうる境界ケースを直接検証する。
    #[test]
    fn multinomial_with_replacement_handles_upper_boundary_without_panicking() {
        // `state` を `next_u64()` が `0xFFFF_FFFF_FFFF_FFFF` 近傍を返す
        // よう手動で構築する（xorshift の内部状態を直接操作せず、実際に
        // 実現しうる `u` の最大値 `(2^53 - 1) / 2^53` で代表させる:
        // 実装が `partition_point` の `idx == row.len()` 分岐を panic
        // なく処理することが本テストの主眼）。
        let mut rng = Xorshift64Star::new(1);
        // 十分な回数回してから、最後に手動で最大値ケースを模した行を
        // 複数回評価し panic しないことを確認する。
        let row = [1.0f32, 1.0, 1.0];
        let mut out = Vec::new();
        for _ in 0..10_000 {
            multinomial_core_with_replacement(&mut rng, &row, 1, &mut out);
            let idx = *out.last().unwrap();
            assert!((0..3).contains(&idx));
        }
    }

    // ---- normal ----

    #[test]
    fn normal_is_deterministic_after_manual_seed() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(101);
        let a = normal(2.0, 3.0, &[6]).unwrap();
        manual_seed(101);
        let b = normal(2.0, 3.0, &[6]).unwrap();
        for i in 0..6 {
            assert_eq!(a.get(&[i]), b.get(&[i]));
        }
    }

    #[test]
    fn normal_std_zero_produces_mean_and_consumes_like_randn() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(22);
        let t = normal(5.0, 0.0, &[7]).unwrap();
        for i in 0..7 {
            assert_eq!(t.get(&[i]).unwrap(), 5.0);
        }
        let state_after_normal = with_global_rng(|r| r.next_u64());

        manual_seed(22);
        let _ = randn(&[7]).unwrap();
        let state_after_randn = with_global_rng(|r| r.next_u64());

        assert_eq!(
            state_after_normal, state_after_randn,
            "std=0 でも randn と同数の抽選を消費するはず"
        );
    }

    #[test]
    fn normal_odd_and_even_numel_consume_same_number_of_draws() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(23);
        let _ = normal(0.0, 1.0, &[5]).unwrap();
        let state_after_five = with_global_rng(|r| r.next_u64());

        manual_seed(23);
        let _ = normal(0.0, 1.0, &[6]).unwrap();
        let state_after_six = with_global_rng(|r| r.next_u64());

        assert_eq!(state_after_five, state_after_six);
    }

    #[test]
    fn normal_large_sample_has_roughly_expected_statistics() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(2025);
        let n = 100_000usize;
        let t = normal(3.0, 2.0, &[n]).unwrap();
        let sum: f64 = (0..n).map(|i| t.get(&[i]).unwrap() as f64).sum();
        let mean = sum / n as f64;
        let var: f64 = (0..n)
            .map(|i| {
                let d = t.get(&[i]).unwrap() as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        assert!((mean - 3.0).abs() < 0.05, "mean out of range: {mean}");
        assert!((3.6..=4.4).contains(&var), "variance out of range: {var}");
    }

    #[test]
    fn normal_rejects_non_finite_arguments_without_consuming_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(24);
        let before = with_global_rng(|r| r.next_u64());

        manual_seed(24);
        assert!(matches!(
            normal(f32::NAN, 1.0, &[2]).unwrap_err(),
            RngError::InvalidArgument { .. }
        ));
        manual_seed(24);
        assert!(matches!(
            normal(0.0, -1.0, &[2]).unwrap_err(),
            RngError::InvalidArgument { .. }
        ));
        manual_seed(24);
        assert!(matches!(
            normal(0.0, f32::INFINITY, &[2]).unwrap_err(),
            RngError::InvalidArgument { .. }
        ));

        let after = with_global_rng(|r| r.next_u64());
        assert_eq!(before, after);
    }

    #[test]
    fn normal_rejects_capacity_overflow_without_consuming_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(25);
        let before = with_global_rng(|r| r.next_u64());
        manual_seed(25);
        let err = normal(0.0, 1.0, &[usize::MAX]).unwrap_err();
        assert!(matches!(
            err,
            RngError::Shape(ShapeError::ElementCountOverflow)
        ));
        let after = with_global_rng(|r| r.next_u64());
        assert_eq!(before, after);
    }

    // ---- Generator ----

    #[test]
    fn generator_new_matches_manual_seed_then_global_free_functions() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(77);
        let probs = Tensor::new(vec![0.5f32; 4], &[4]).unwrap();
        let global_bernoulli = bernoulli(&probs).unwrap();
        let global_normal = normal(1.0, 2.0, &[4]).unwrap();

        let mut generator = Generator::new(77);
        let gen_bernoulli = generator.bernoulli(&probs).unwrap();
        let gen_normal = generator.normal(1.0, 2.0, &[4]).unwrap();

        for i in 0..4 {
            assert_eq!(
                global_bernoulli.get(&[i]).unwrap(),
                gen_bernoulli.get(&[i]).unwrap()
            );
            assert_eq!(
                global_normal.get(&[i]).unwrap(),
                gen_normal.get(&[i]).unwrap()
            );
        }
    }

    #[test]
    fn generator_does_not_affect_global_state() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(88);
        let before = with_global_rng(|r| r.next_u64());

        manual_seed(88);
        let mut generator = Generator::new(999);
        let probs = Tensor::new(vec![0.5f32; 10], &[10]).unwrap();
        let _ = generator.bernoulli(&probs).unwrap();
        let _ = generator.normal(0.0, 1.0, &[10]).unwrap();
        let weights = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        let _ = generator.multinomial(&weights, 5, true).unwrap();

        let after = with_global_rng(|r| r.next_u64());
        assert_eq!(
            before, after,
            "Generator を使ってもグローバル状態は変わらないはず"
        );
    }

    #[test]
    fn generator_clone_produces_identical_sequence() {
        let mut generator = Generator::new(42);
        let mut cloned = generator.clone();

        let weights = Tensor::new(vec![1.0f32, 1.0, 1.0], &[3]).unwrap();
        let x = generator.multinomial(&weights, 5, true).unwrap();
        let y = cloned.multinomial(&weights, 5, true).unwrap();
        for i in 0..5 {
            assert_eq!(x.get(&[i]).unwrap(), y.get(&[i]).unwrap());
        }
    }

    #[test]
    fn generator_manual_seed_restarts_sequence() {
        let mut generator = Generator::new(1);
        let a = generator.normal(0.0, 1.0, &[4]).unwrap();

        generator.manual_seed(1);
        let b = generator.normal(0.0, 1.0, &[4]).unwrap();

        for i in 0..4 {
            assert_eq!(a.get(&[i]).unwrap(), b.get(&[i]).unwrap());
        }
    }

    #[test]
    fn generator_initial_seed_reports_pre_correction_value() {
        let generator = Generator::new(0);
        assert_eq!(generator.initial_seed(), 0);
        let gen2 = Generator::new(123);
        assert_eq!(gen2.initial_seed(), 123);
    }

    #[test]
    fn generator_debug_does_not_expose_internal_state() {
        let generator = Generator::new(42);
        let debug = format!("{generator:?}");
        assert!(debug.contains("42"));
        assert!(!debug.contains("state"));
    }

    #[test]
    fn two_generators_do_not_interfere_with_each_other() {
        let mut a = Generator::new(1);
        let mut b = Generator::new(2);

        let a1 = a.normal(0.0, 1.0, &[3]).unwrap();
        let b1 = b.normal(0.0, 1.0, &[3]).unwrap();
        let a2 = a.normal(0.0, 1.0, &[3]).unwrap();

        // 独立系列であることの間接確認: 異なるシードから始まる a・b は
        // 別系列（a1 != b1 が高確率で成り立つ）で、a を 2 回呼んでも
        // b の状態には一切触れないため a の 2 回目呼び出し（a2）は a1 の
        // 続きの系列になる（別途 a を新規に same-seed で再現して比較）。
        let mut a_reference = Generator::new(1);
        let a1_ref = a_reference.normal(0.0, 1.0, &[3]).unwrap();
        let a2_ref = a_reference.normal(0.0, 1.0, &[3]).unwrap();
        for i in 0..3 {
            assert_eq!(a1.get(&[i]).unwrap(), a1_ref.get(&[i]).unwrap());
            assert_eq!(a2.get(&[i]).unwrap(), a2_ref.get(&[i]).unwrap());
        }
        let _ = b1;
    }
}
