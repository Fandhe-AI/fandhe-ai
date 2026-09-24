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
//!
//! # `nn::init`（PyTorch `torch.nn.init.*` 相当。イシュー #2140）
//!
//! 本モジュール下部の public 関数群（[`uniform`]／[`normal`]／
//! [`constant`]／[`xavier_uniform`]／[`xavier_normal`]／
//! [`kaiming_uniform`]／[`kaiming_normal`]／[`orthogonal`]／
//! [`trunc_normal`]）は、上記の個別シード方式（`uniform_init` 等）とは
//! 異なり、`tensor-core::rng::with_global_rng` を経由してプロセス
//! グローバルな決定的 RNG（[`fandhe_ai_tensor_core::rng::manual_seed`]）
//! に**従属する**（`tensor-core::rng::randn`／`rand` と同じ設計方針）。
//! `Linear::new(.., seed)` 等の既存個別シード API・`derive_seed`・
//! `uniform_init`／`try_uniform_init`／`try_normal_init`（本ファイル上部）
//! は一切変更せず、`manual_seed` を何度呼んでもそれらの出力は不変の
//! ままである（独立性はモジュール冒頭の契約どおり）。
//!
//! 呼び出し元は `Linear::from_parameters`／`Conv2d::from_parameters`／
//! `Embedding::from_parameters` 等の「明示的な重み・バイアスから構築
//! する」入口（safetensors ロード等と同じ位置づけ）へ、本モジュールの
//! 関数が返す [`Tensor<f32>`] を渡して層を組み立てる。各層の既定
//! コンストラクタ（`new(.., seed)`）自体はこの変更の対象外
//! （イシュー #2140 のスコープ外。本文参照）。
//!
//! facade（`fandhe_ai::nn::init`）への再エクスポートは別途ユーザー承認
//! （`docs/compat-api-scope.md` §5 経路 2）を要する公開面拡張であり、
//! 本イシュー時点では未承認のため `crates/facade/**` には反映しない
//! （`docs/facade-nn-init-exposure-decision.md` 参照）。

use fandhe_ai_tensor_core::rng::{Xorshift64Star, with_global_rng};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;

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

// ============================================================================
// `nn::init`（PyTorch `torch.nn.init.*` 相当。イシュー #2140）公開 API。
// 上のセクション（個別シード方式）とは独立に、プロセスグローバル RNG
// （モジュール冒頭コメント参照）に従属する関数群をここに実装する。
// ============================================================================

/// `nn::init` の入力検証エラーを組み立てる補助関数（`AutodiffError::
/// InvalidArgument` への集約。本番経路 `unwrap`／`expect` 禁止の方針
/// どおり、RNG ロック取得・アロケーション前に fail-closed で拒否する）。
///
/// **確保サイズ計算の失敗には使わない**（`alloc_failed` の doc 参照）:
/// 本関数は「引数の意味論が不正」（`gain` の符号・`low <= high`・`rank`
/// 等）を表す `String` メッセージ付きエラー専用とし、「shape が大きす
/// ぎて確保できない」系の失敗は非アロケーションな [`alloc_failed`] を
/// 使う。
fn invalid_argument(message: impl Into<String>) -> AutodiffError {
    AutodiffError::InvalidArgument(message.into())
}

/// 確保不能（要素数積の `usize` オーバーフロー、または
/// `Vec::try_reserve_exact` が実際に確保失敗を報告した）を表す
/// **非アロケーションな** `AutodiffError` を返す（codex-review 指摘。
/// PR #2239）。
///
/// `try_alloc`（本関数の直後）は以前 `format!("... 要素数 {len} の
/// 確保に失敗しました")` を経由して `AutodiffError::InvalidArgument
/// (String)` を構築していたが、これは確保失敗を検出した**直後**に
/// 新たな `String` 確保（`format!` はヒープ確保を伴う）を試みる構造
/// になっており、実メモリ枯渇による確保失敗時にはこのエラー文字列の
/// 構築自体が `handle_alloc_error`（`String`／`Vec` の非フォールブル
/// 確保 API 内部で呼ばれる）経由の abort を招きうる——「確保失敗は
/// panic／abort ではなく `Err` で伝播する」契約（イシュー #2140・
/// PR #2239 で `eval::linalg::qr` に確立した契約と同型）を、確保
/// 失敗の*報告*経路自体が破ってしまっていた。
///
/// **既存の非アロケーションなエラー型の再利用（新規 variant は追加
/// しない）**: `AutodiffError` は `#[non_exhaustive]` の公開型で
/// `facade`（`fandhe_ai::AutodiffError`）が再エクスポートするため、
/// 新規 variant の追加は公開面の変更としてユーザー承認が必要になる
/// （`.claude/rules/deps-policy.md` 相当の「公開 API 非破壊」ガード
/// レール）。調査の結果、`fandhe_ai_tensor_core::ShapeError::
/// ElementCountOverflow`（`tensor-core/src/error.rs`）が既に
/// 「shape の要素数積が `usize` の範囲でオーバーフローする、または
/// 要素型込みのバイトサイズが `Vec` の allocation 上限（`isize::MAX`
/// バイト）を超えアロケーション不能な shape」という**本件と全く同じ
/// 意味論**を持つ非データ（unit）variant として存在し（`Tensor::
/// zeros`／`ones`／`full` が既に同じ用途で使っている確立済みパターン
/// ——それらも確保失敗時に個別メッセージを持たない）、`AutodiffError`
/// は `From<ShapeError>` を既に実装しているため、新規 variant を
/// 追加せずそのまま再利用できる。`AutodiffError::Shape(...)` の
/// 構築・`ShapeError::ElementCountOverflow`（unit variant・`Clone`
/// ／`Eq` 導出のみで `String` 等のヒープ保持フィールドを持たない）の
/// 構築はいずれもスタック上のデータ移動のみで完結し、ヒープ確保を
/// 一切伴わない。
///
/// **適用範囲（類型化）**: 「確保対象のサイズ計算・確保そのものが
/// 失敗した」エラー（`checked_numel`・`try_alloc`・`calculate_fan_in_
/// and_fan_out` の `checked_mul` 系・`orthogonal` の shape 要素数
/// 計算・`eval::linalg::qr` 経路の `Mat::try_zeros`／`try_from_tensor`
/// ／`try_to_tensor`／`try_vec_zeroed`）にのみ使う。引数の**意味論**
/// が不正なエラー（`gain` の符号・`low <= high`・`std < 0`・`rank`
/// 不足・軸が 0 等）は、確保が一切絡まないため従来どおり
/// [`invalid_argument`]（`String` メッセージ付き）を使い続ける——
/// これらは確保失敗の最中に発生するものではなく、システムが実メモリ
/// 枯渇状態にあるとは限らないタイミングで発生するため、診断メッセ
/// ージを保持する価値の方が上回る。
fn alloc_failed() -> AutodiffError {
    AutodiffError::Shape(ShapeError::ElementCountOverflow)
}

/// `shape` の要素数積を `checked_mul` で求める（`tensor-core::
/// checked_numel_for` は `pub(crate)` で他クレートから到達不能なため、
/// `nn::init` 専用に同等の検査をここで再実装する。イシュー #1725／
/// #1726 の `randn`／`arange` 等と同じ「アロケーション前に要素数
/// オーバーフローを検出する」契約）。
fn checked_numel(shape: &[usize]) -> Result<usize, AutodiffError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(alloc_failed)
}

/// `len` 要素の `Vec<f32>` を、確保不能なら panic せず `Err` を返す形で
/// 事前予約する（`try_uniform_init`／`try_normal_init` と同じ理由。
/// `try_reserve_exact` は要素数換算のバイトサイズが `isize::MAX` を
/// 超えるアロケーション不能な `len` も検出する）。
fn try_alloc(len: usize) -> Result<Vec<f32>, AutodiffError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| alloc_failed())?;
    Ok(values)
}

/// `[low, high)` の一様分布で `len` 要素を埋める（グローバル RNG を
/// 1 回だけロックし、要素をまとめて引く。`tensor-core::rng::rand` と
/// 同じロック粒度の契約）。呼び出し元が `low <= high` を事前検証する。
///
/// **幅 `high - low` は `f64` で計算する**（codex-review 指摘。PR
/// #2239）: `low`・`high` はそれぞれ有限な `f32` として検査済みでも、
/// `low = -f32::MAX`・`high = f32::MAX` のように両端の差（幅）自体が
/// `f32` の表現範囲を超えるケースが存在する。旧実装は幅を `f32` で
/// 計算していたため、この幅が `inf` になり出力も `inf`（乱数値が 0 の
/// 要素は `0 * inf` で `NaN` にもなる）を返していた——呼び出し元が
/// 「両端とも有限」を保証していても出力が有限にならない欠陥だった。
/// `f64` は `f32` の全表現域の差を必ず有限で表せる（`f64::MAX ≈
/// 1.8e308` は `f32::MAX ≈ 3.4e38` の 2 倍を遥かに超える）ため、
/// `low + t·(high - low)` を `f64` で計算すれば結果は常に `[low, high]`
/// 区間内（両端は有限な `f32` として検証済み）に収まり、最後の 1 回の
/// `f32` ダウンキャストで overflow しない。`xavier_uniform`／
/// `kaiming_uniform` の `fill_uniform(numel, -bound, bound)` 呼び出し
/// （幅 `2·bound` が同じ overflow クラスに該当する）も本関数経由で
/// 同時に是正される。
fn fill_uniform(len: usize, low: f32, high: f32) -> Result<Vec<f32>, AutodiffError> {
    let mut out = try_alloc(len)?;
    let low_f64 = f64::from(low);
    let width_f64 = f64::from(high) - low_f64;
    with_global_rng(|rng| {
        for _ in 0..len {
            let v = low_f64 + f64::from(rng.next_unit_f32()) * width_f64;
            out.push(exclude_high(v as f32, low, high));
        }
    });
    Ok(out)
}

/// `[low, high)`（半開区間。上限 `high` を排他）を**最終出力型
/// （`f32`）で**保証する（codex-review 指摘。PR #2239）: `next_unit_f32`
/// 自体は `[0, 1)` だが、`low + t·(high - low)` を `f64` で計算した
/// 真の値が `high` 未満でも、`f32` へダウンキャストする際の最近接丸め
/// で `high` そのもの（`low`・`high` が隣接する `f32` で `t` が最大値
/// `1 - 2^-24` に近い場合等）へ丸め上がることがあり、`fill_uniform` が
/// 明記する `[low, high)` 契約を破る（`fill_uniform` の唯一の呼び出し
/// 元 `uniform`・および `-bound..bound` を渡す `xavier_uniform`／
/// `kaiming_uniform` に波及する）。
///
/// `v >= high` なら `high` の直前の有限 `f32`（[`f32::next_down`]。
/// Rust 1.86 で安定化済み）へ補正する。`low == high` の縮退ケース
/// （`uniform_low_equals_high_is_constant` が固定する「定数 `low` を
/// 返す」契約）では `width_f64 == 0.0` のため `v` は常に厳密に `low`
/// （`== high`）になり本関数の補正条件（`v >= high`）に必ず入るが、
/// `high.next_down()` は `low` 未満になってしまうため `.max(low)` で
/// 下限へクランプし、既存の「`low == high` は定数 `low` を返す」挙動を
/// 保つ。`v < low` へ丸め落ちる経路は存在しない（`t >= 0.0`・
/// `width_f64 >= 0.0` かつ `low` 自体が厳密に表現可能な `f32` である
/// ため、`v_f64 >= low_f64` が常に成り立ち、最近接丸めは `low` 未満には
/// ならない——`low` 自身が丸め候補の 1 つであるため）。
fn exclude_high(v: f32, low: f32, high: f32) -> f32 {
    if v >= high {
        high.next_down().max(low)
    } else {
        v
    }
}

/// `N(mean, std²)` で `len` 要素を埋める（Box–Muller 変換・`f64` 中間
/// 計算。`tensor-core::rng::randn`／本ファイル上部の `try_normal_init`
/// と同一のアルゴリズム。決定性の範囲は「同一プロセス・同一プラット
/// フォーム内」に限る〈`ln`／`sin`／`cos` を経由するため〉）。
///
/// **`z * std + mean` も `f64` で計算する**（codex-review 同類型点検。
/// PR #2239）: 旧実装は Box–Muller の `z`（`r * cos(theta)` 等）を
/// 直後に `f32` へダウンキャストしてから `f32` で `std` 倍・`mean` 加算
/// していたため、`z`・`std`・`mean` の個々の値は有限でも、
/// `z * std`（中間積）が `f32` の表現範囲を超えて `inf` になり、その後
/// `mean` を足しても `inf` のまま出力されうる不具合クラスが存在した
/// （`z ≈ 1.05`・`std ≈ 3.3e38`・`mean ≈ -3.3e38` の場合、真の値
/// `z * std + mean ≈ 1.65e37` は `f32` に十分収まるが、`f32` の
/// `z * std` 単体は `f32::MAX`（`≈3.4028e38`）を超えて `inf` になる）。
/// `z`（`r * theta.cos()`／`r * theta.sin()`）を `f32` へ早期変換せず
/// `f64` のまま `std`・`mean` を `f64` へ昇格して演算し、最後の 1 回だけ
/// `f32` へダウンキャストすることで、この中間 overflow を避ける
/// （`.claude/rules/coding-rust.md` の「勾配の長軸縮約は `f64` アキュム
/// レータで統一する」契約と同じ「最終書き出しのみ 1 回ダウンキャスト」
/// 方針を、ここでも適用した）。`xavier_normal`／`kaiming_normal` は
/// 本関数経由で同時に是正される。
fn fill_normal(len: usize, mean: f32, std: f32) -> Result<Vec<f32>, AutodiffError> {
    let mut out = try_alloc(len)?;
    let mean_f64 = f64::from(mean);
    let std_f64 = f64::from(std);
    with_global_rng(|rng| {
        let mut remaining = len;
        while remaining > 0 {
            // `u1` は `(0, 1]` に補正して `ln(0)`（負の無限大）を避ける。
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
    });
    Ok(out)
}

/// PyTorch `nn.init.calculate_gain` の対応表に準拠した非線形性 kind
/// （`#[non_exhaustive]`: 公開 API 非破壊のため後続の追加に備える。
/// `.claude/rules/security.md`）。`LeakyRelu` は負勾配を値として保持
/// する（`kaiming_uniform`／`kaiming_normal` の `a` 引数との関係は
/// [`calculate_gain`] の doc を参照）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Nonlinearity {
    Linear,
    Conv1d,
    Conv2d,
    Sigmoid,
    Tanh,
    Relu,
    /// 負勾配（PyTorch `leaky_relu` の `param` 相当）。
    LeakyRelu(f32),
    Selu,
}

/// [`calculate_fan_in_and_fan_out`] が返す `(fan_in, fan_out)` のどちら
/// を初期化スケールに使うか（PyTorch `nn.init._calculate_correct_fan`
/// の `mode` 相当）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FanMode {
    FanIn,
    FanOut,
}

/// PyTorch `nn.init.calculate_gain` 相当。非線形性ごとの推奨 gain 値を
/// 返す（単体で直接呼ぶ場合の他、[`kaiming_uniform`]／
/// [`kaiming_normal`] は内部の `kaiming_gain` 経由でこれを呼ぶ）。
///
/// `LeakyRelu(negative_slope)` は `negative_slope` を直接使って
/// `sqrt(2 / (1 + negative_slope²))` を返す。`kaiming_uniform`／
/// `kaiming_normal` から呼ぶ場合は `a` 引数（PyTorch シグネチャ互換の
/// `kaiming_uniform_(tensor, a=0, ...)` 相当）が `LeakyRelu` の負勾配
/// として優先される（`kaiming_gain` の doc 参照）。本関数を直接呼ぶ
/// 場合は列挙子に埋め込んだ `negative_slope` がそのまま使われる。
pub fn calculate_gain(nonlinearity: Nonlinearity) -> f32 {
    calculate_gain_f64(nonlinearity) as f32
}

/// [`calculate_gain`] の `f64` 内部実装（codex-review 指摘。PR #2239）:
/// `LeakyRelu(negative_slope)` の `negative_slope * negative_slope` を
/// `f32` のまま計算すると、`|negative_slope| > √f32::MAX ≈ 1.85e19`
/// という**有限**な入力で中間値が `inf` になり、`2.0 / inf == 0.0` の
/// `sqrt` で `calculate_gain` が誤って `0.0` を返してしまう（真の値は
/// `negative_slope = f32::MAX` で約 `4.2e-39`——`f32` の subnormal 域だが
/// 表現可能）。`kaiming_uniform`／`kaiming_normal` は `gain == 0.0` だと
/// `std == 0`／`bound == 0` になり重みが全て 0 で初期化されてしまう。
/// `negative_slope` を `f32` へ早期変換せず `f64` で二乗・除算・平方根
/// まで計算し、最後の 1 回だけ `f32` へダウンキャストする（`fill_*` 系
/// で確立した「最終書き出しのみ 1 回ダウンキャスト」方針と同型）。
///
/// **挙動変化（1 点）**: `LeakyRelu(f32::INFINITY)` は旧実装では
/// `inf * inf == inf`・`2.0 / (1.0 + inf) == 0.0`・`sqrt(0.0) == 0.0`
/// と経由するはずが実際には `1.0 + inf == inf` の除算で `NaN` を返して
/// いた（`f32` の `inf - inf`／`inf/inf` に相当する不定形が生じる中間
/// 経路が存在するため）。`f64` でも同じ極限（`negative_slope² → inf`・
/// `2/(1+inf) → 0`）を辿るため `0.0`（数学的な極限値と一致し、`NaN`
/// より意味のある結果）を返すようになる。`negative_slope` は有限値の
/// 想定（PyTorch の `leaky_relu` も有限のスロープを前提とする）であり、
/// 呼び出し元（`kaiming_uniform`／`kaiming_normal`）は `a`（`negative_
/// slope` の実引数）の有限性を事前検査するため実害はない。
fn calculate_gain_f64(nonlinearity: Nonlinearity) -> f64 {
    match nonlinearity {
        Nonlinearity::Linear | Nonlinearity::Conv1d | Nonlinearity::Conv2d => 1.0,
        Nonlinearity::Sigmoid => 1.0,
        Nonlinearity::Tanh => 5.0 / 3.0,
        Nonlinearity::Relu => std::f64::consts::SQRT_2,
        Nonlinearity::LeakyRelu(negative_slope) => {
            let slope = f64::from(negative_slope);
            (2.0 / (1.0 + slope * slope)).sqrt()
        }
        Nonlinearity::Selu => 3.0 / 4.0,
    }
}

/// PyTorch `nn.init._calculate_fan_in_and_fan_out` 相当。`shape` は
/// `[out_features_or_channels, in_features_or_channels, ..kernel_dims]`
/// （rank ≥ 2）を要求し、`shape[2..]` の積を receptive field size として
/// `fan_in = shape[1] * receptive_field_size`・
/// `fan_out = shape[0] * receptive_field_size` を返す。
///
/// **`nn::Linear` へ適用する際の注意**: PyTorch `nn.Linear.weight` は
/// `[out_features, in_features]` だが、`nn::Linear::weight`（`linear.rs`）
/// は `y = input.matmul(weight)` の合成のため転置の関係にある
/// `[in_features, out_features]` を持つ。本関数・[`xavier_uniform`] 等
/// を `nn::Linear::from_parameters` へそのまま渡す shape 引数として
/// 使う場合、`shape[0]` は物理的には `in_features` であり、fan の意味
/// （`fan_in`／`fan_out`）が PyTorch の直感とは入れ替わる。`Conv2d`／
/// `Embedding` の重みレイアウトは PyTorch と同じ `[out_channels,
/// in_channels, ..]` のため本注意は生じない（`docs/
/// facade-nn-init-exposure-decision.md` 参照）。
pub fn calculate_fan_in_and_fan_out(shape: &[usize]) -> Result<(usize, usize), AutodiffError> {
    if shape.len() < 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: shape.len(),
        }));
    }
    // `receptive_field_size`／`fan_in`／`fan_out` の `checked_mul` は
    // いずれも「shape 由来のサイズ計算が確保可能な範囲に収まるか」の
    // 検査であり、`alloc_failed`（非アロケーション）の適用対象（`alloc_
    // failed` の doc「類型化」参照。codex-review 同類型点検・PR #2239）。
    let receptive_field_size = shape[2..]
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(alloc_failed)?;
    let fan_in = shape[1]
        .checked_mul(receptive_field_size)
        .ok_or_else(alloc_failed)?;
    let fan_out = shape[0]
        .checked_mul(receptive_field_size)
        .ok_or_else(alloc_failed)?;
    Ok((fan_in, fan_out))
}

/// PyTorch `nn.init.uniform_` 相当。`shape` の全要素を `U(low, high)`
/// から独立にサンプルする。`low > high` または非有限値は
/// `AutodiffError::InvalidArgument` を返す（グローバル RNG は未消費）。
pub fn uniform(shape: &[usize], low: f32, high: f32) -> Result<Tensor<f32>, AutodiffError> {
    if !low.is_finite() || !high.is_finite() {
        return Err(invalid_argument(
            "nn::init::uniform: low・high は有限である必要があります",
        ));
    }
    if low > high {
        return Err(invalid_argument(
            "nn::init::uniform: low は high 以下である必要があります",
        ));
    }
    let numel = checked_numel(shape)?;
    let data = fill_uniform(numel, low, high)?;
    Tensor::new(data, shape).map_err(AutodiffError::from)
}

/// PyTorch `nn.init.normal_` 相当。`shape` の全要素を `N(mean, std²)`
/// から独立にサンプルする。`std == 0` は [`constant`]（`mean` の定数）
/// にフォールバックし RNG を消費しない（PyTorch も `std=0` を許容する
/// 挙動に合わせる）。`std < 0` または非有限値は
/// `AutodiffError::InvalidArgument` を返す。
pub fn normal(shape: &[usize], mean: f32, std: f32) -> Result<Tensor<f32>, AutodiffError> {
    if !mean.is_finite() {
        return Err(invalid_argument(
            "nn::init::normal: mean は有限である必要があります",
        ));
    }
    if !std.is_finite() || std < 0.0 {
        return Err(invalid_argument(
            "nn::init::normal: std は有限かつ非負である必要があります",
        ));
    }
    if std == 0.0 {
        return constant(shape, mean);
    }
    let numel = checked_numel(shape)?;
    let data = fill_normal(numel, mean, std)?;
    Tensor::new(data, shape).map_err(AutodiffError::from)
}

/// PyTorch `nn.init.constant_` 相当。`shape` の全要素を `value` で埋める
/// （RNG を消費しない）。
pub fn constant(shape: &[usize], value: f32) -> Result<Tensor<f32>, AutodiffError> {
    let numel = checked_numel(shape)?;
    let mut out = try_alloc(numel)?;
    out.resize(numel, value);
    Tensor::new(out, shape).map_err(AutodiffError::from)
}

/// PyTorch `nn.init.xavier_uniform_` 相当。
/// `bound = gain·√(6 / (fan_in + fan_out))` の `U(-bound, bound)`。
/// `fan_in + fan_out == 0`（`shape` の該当軸が 0）は
/// `AutodiffError::InvalidArgument` を返す。
/// 負の `gain` も拒否する（PyTorch の `uniform_` が `from > to` を拒否
/// するのと同じ意味論で、`bound` が非負区間の半幅であることを保証する。
/// `orthogonal` は単なるスケールとして負の gain を受理するため対象外）。
pub fn xavier_uniform(shape: &[usize], gain: f32) -> Result<Tensor<f32>, AutodiffError> {
    if !gain.is_finite() {
        return Err(invalid_argument(
            "nn::init::xavier_uniform: gain は有限である必要があります",
        ));
    }
    if gain < 0.0 {
        return Err(invalid_argument(
            "nn::init::xavier_uniform: gain は非負である必要があります",
        ));
    }
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(shape)?;
    let denom = fan_in
        .checked_add(fan_out)
        .filter(|&d| d > 0)
        .ok_or_else(|| {
            invalid_argument(
                "nn::init::xavier_uniform: fan_in + fan_out は 0 より大きい必要があります",
            )
        })?;
    // `bound` は `f64` で計算し最後の 1 回だけ `f32` へダウンキャスト
    // する（`calculate_gain_f64`／`kaiming_uniform` と同じ「最終書き出し
    // のみ 1 回ダウンキャスト」方針。codex-review 同類型点検・PR
    // #2239。`gain` 自体は既に有限な `f32` として検査済みのため、
    // `f64` へ昇格しても情報の欠落は生じない）。
    let bound_f64 = f64::from(gain) * (6.0 / denom as f64).sqrt();
    let bound = bound_f64 as f32;
    if !bound.is_finite() {
        return Err(invalid_argument(
            "nn::init::xavier_uniform: 計算された bound が有限ではありません",
        ));
    }
    let numel = checked_numel(shape)?;
    let data = fill_uniform(numel, -bound, bound)?;
    Tensor::new(data, shape).map_err(AutodiffError::from)
}

/// PyTorch `nn.init.xavier_normal_` 相当。
/// `std = gain·√(2 / (fan_in + fan_out))` の `N(0, std²)`。
/// 負の `gain` も拒否する（PyTorch の `normal_` が `std < 0` を拒否
/// するのと同じ意味論で、`std` が非負であることを保証する。
/// `orthogonal` は単なるスケールとして負の gain を受理するため対象外）。
pub fn xavier_normal(shape: &[usize], gain: f32) -> Result<Tensor<f32>, AutodiffError> {
    if !gain.is_finite() {
        return Err(invalid_argument(
            "nn::init::xavier_normal: gain は有限である必要があります",
        ));
    }
    if gain < 0.0 {
        return Err(invalid_argument(
            "nn::init::xavier_normal: gain は非負である必要があります",
        ));
    }
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(shape)?;
    let denom = fan_in
        .checked_add(fan_out)
        .filter(|&d| d > 0)
        .ok_or_else(|| {
            invalid_argument(
                "nn::init::xavier_normal: fan_in + fan_out は 0 より大きい必要があります",
            )
        })?;
    // `xavier_uniform` と同じ理由で `f64` 計算・最後の 1 回だけ
    // ダウンキャスト。
    let std_f64 = f64::from(gain) * (2.0 / denom as f64).sqrt();
    let std = std_f64 as f32;
    if !std.is_finite() {
        return Err(invalid_argument(
            "nn::init::xavier_normal: 計算された std が有限ではありません",
        ));
    }
    let numel = checked_numel(shape)?;
    let data = fill_normal(numel, 0.0, std)?;
    Tensor::new(data, shape).map_err(AutodiffError::from)
}

/// `mode` が選んだ fan（`FanIn`→`fan_in`／`FanOut`→`fan_out`）が 0 なら
/// `kaiming_uniform`／`kaiming_normal` の `std`／`bound` 計算が
/// `1/√0`（非有限）になるため、共通の事前検査としてここでまとめて
/// 拒否する。
fn select_fan(shape: &[usize], mode: FanMode) -> Result<usize, AutodiffError> {
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(shape)?;
    let fan = match mode {
        FanMode::FanIn => fan_in,
        FanMode::FanOut => fan_out,
    };
    if fan == 0 {
        return Err(invalid_argument(
            "nn::init::kaiming_*: 選択された fan（fan_in／fan_out）は 0 より大きい必要があります",
        ));
    }
    Ok(fan)
}

/// `kaiming_uniform`／`kaiming_normal` の `a` 引数を PyTorch
/// `kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')`
/// と同じ意味論で gain 計算へ反映する（PyTorch 内部実装は
/// `gain = calculate_gain(nonlinearity, param=a)` のように `a` を
/// `calculate_gain` の `param` として渡し、`nonlinearity` が
/// `'leaky_relu'` のときのみ `param` が負勾配として使われる）。
///
/// `nonlinearity` が `Nonlinearity::LeakyRelu(_)` の場合、埋め込まれた
/// 負勾配ではなく `a` を負勾配として採用する（`a` が唯一の情報源になる
/// よう `LeakyRelu` に埋め込まれた値は上書きする。呼び出し元が
/// `LeakyRelu(slope)` と `a` に異なる値を渡した場合の二重定義による
/// 曖昧さを避けるため）。`LeakyRelu` 以外の `nonlinearity` では `a` は
/// gain 計算に影響しない（PyTorch でも `nonlinearity != 'leaky_relu'`
/// のとき `param` は無視される）。以前は `a` を有限性検査にのみ使い
/// gain 計算から完全に除外していたため、PyTorch から移行する呼び出し元
/// が `a` を指定しても初期化分散に反映されない互換性の欠落があった
/// （codex-review 指摘。AGENTS.md「公開 API は PyTorch からの移行
/// 容易性を保つ」契約）。
/// `f64` で返す（呼び出し元 `kaiming_uniform`／`kaiming_normal` が
/// `std`／`bound` の計算も `f64` で行い、最後の 1 回だけ `f32` へ
/// ダウンキャストするため。`calculate_gain_f64` の doc 参照）。
fn kaiming_gain(nonlinearity: Nonlinearity, a: f32) -> f64 {
    let effective = match nonlinearity {
        Nonlinearity::LeakyRelu(_) => Nonlinearity::LeakyRelu(a),
        other => other,
    };
    calculate_gain_f64(effective)
}

/// PyTorch `nn.init.kaiming_uniform_` 相当。
/// `std = gain / √fan`・`bound = √3·std` の `U(-bound, bound)`。
///
/// `a`（負勾配。`nonlinearity` が `Nonlinearity::LeakyRelu(_)` のときの
/// み gain 計算に使う）の意味論は `kaiming_gain` の doc を参照。
pub fn kaiming_uniform(
    shape: &[usize],
    a: f32,
    mode: FanMode,
    nonlinearity: Nonlinearity,
) -> Result<Tensor<f32>, AutodiffError> {
    if !a.is_finite() {
        return Err(invalid_argument(
            "nn::init::kaiming_uniform: a は有限である必要があります",
        ));
    }
    let fan = select_fan(shape, mode)?;
    // `gain`／`std`／`bound` は `f64` で計算し、最後の 1 回だけ `f32` へ
    // ダウンキャストする（`calculate_gain_f64` の doc と同じ理由。
    // codex-review 同類型点検・PR #2239）。`fan as f64` は `fan` が
    // `usize`（典型的な shape では `2^53` を大きく下回る）である限り
    // 厳密変換になる。
    let gain = kaiming_gain(nonlinearity, a);
    let std_f64 = gain / (fan as f64).sqrt();
    let bound_f64 = std_f64 * 3f64.sqrt();
    let bound = bound_f64 as f32;
    if !bound.is_finite() {
        return Err(invalid_argument(
            "nn::init::kaiming_uniform: 計算された bound が有限ではありません",
        ));
    }
    let numel = checked_numel(shape)?;
    let data = fill_uniform(numel, -bound, bound)?;
    Tensor::new(data, shape).map_err(AutodiffError::from)
}

/// PyTorch `nn.init.kaiming_normal_` 相当。`std = gain / √fan` の
/// `N(0, std²)`。`a` の扱いは [`kaiming_uniform`]（`kaiming_gain` の
/// doc）と同じ。
pub fn kaiming_normal(
    shape: &[usize],
    a: f32,
    mode: FanMode,
    nonlinearity: Nonlinearity,
) -> Result<Tensor<f32>, AutodiffError> {
    if !a.is_finite() {
        return Err(invalid_argument(
            "nn::init::kaiming_normal: a は有限である必要があります",
        ));
    }
    let fan = select_fan(shape, mode)?;
    // `f64` 計算・最後の 1 回だけダウンキャスト（`kaiming_uniform` と
    // 同じ理由）。
    let gain = kaiming_gain(nonlinearity, a);
    let std_f64 = gain / (fan as f64).sqrt();
    let std = std_f64 as f32;
    if !std.is_finite() {
        return Err(invalid_argument(
            "nn::init::kaiming_normal: 計算された std が有限ではありません",
        ));
    }
    let numel = checked_numel(shape)?;
    let data = fill_normal(numel, 0.0, std)?;
    Tensor::new(data, shape).map_err(AutodiffError::from)
}

/// PyTorch `nn.init.orthogonal_` 相当。`shape`（rank ≥ 2）を
/// `[rows, cols]`（`rows = shape[0]`・`cols = Π shape[1..]`）へ平坦化し、
/// `N(0, 1)` 行列を `crate::eval::linalg::qr`（Householder reduced QR。
/// `R` の対角を非負に正規化済み——PyTorch の `q *= sign(diag(r))` と
/// 等価な一意化）で直交化してから `gain` 倍し `shape` へ書き戻す。
/// `rows < cols` の場合は PyTorch と同じく転置してから QR を取り、
/// 結果を転置し戻す（`rows >= cols` を要求する QR の制約を回避する
/// ため）。負の `gain` は（`xavier_uniform`／`xavier_normal` と異なり）
/// 単なるスケール係数として受理する（PyTorch `orthogonal_` も同様に
/// 符号チェックをしない。直交行列に負数を乗じても直交性は保たれる
/// ため意味論上の破綻がない）。
pub fn orthogonal(shape: &[usize], gain: f32) -> Result<Tensor<f32>, AutodiffError> {
    if !gain.is_finite() {
        return Err(invalid_argument(
            "nn::init::orthogonal: gain は有限である必要があります",
        ));
    }
    if shape.len() < 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: shape.len(),
        }));
    }
    let rows = shape[0];
    // `cols`／`gen_numel` の `checked_mul` は shape 由来のサイズ計算
    // なので `alloc_failed`（非アロケーション）の適用対象（`alloc_
    // failed` の doc「類型化」参照。codex-review 同類型点検・PR
    // #2239）。
    let cols = shape[1..]
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(alloc_failed)?;
    if rows == 0 || cols == 0 {
        return Err(invalid_argument(
            "nn::init::orthogonal: shape の各軸は 0 より大きい必要があります",
        ));
    }
    let (gen_rows, gen_cols, transposed) = if rows < cols {
        (cols, rows, true)
    } else {
        (rows, cols, false)
    };
    let gen_numel = gen_rows.checked_mul(gen_cols).ok_or_else(alloc_failed)?;
    let data = fill_normal(gen_numel, 0.0, 1.0)?;
    let m = Tensor::new(data, &[gen_rows, gen_cols])?;
    // `crate::eval::linalg::qr` は `Result` を返す（内部の `f64` 作業
    // 領域——`Mat::try_from_tensor`／`Mat::try_zeros`／Householder
    // ベクトルの一時確保——を全て `try_reserve_exact` 経由のフォール
    // ブル確保へ統一済み）。単発の probe-then-drop 検査（予約後すぐ
    // 解放するだけで、`qr` 内部が複数の `f64` 作業領域を同時に保持する
    // 実際の確保失敗を防げない）では不十分という codex-review 指摘
    // （PR #2239）を受け、確保そのものをフォールブル化して `?` で
    // そのまま伝播する形に是正した（実アロケータがオーバーコミット
    // 環境で成功を返した後に実メモリ不足で OOM killer が働く経路までは
    // 防げないが、我々のコード内の allocation panic／abort 経路は
    // 閉じる。`docs/facade-nn-init-exposure-decision.md` §6）。
    let (q, r) = crate::eval::linalg::qr(&m)?;
    // `R` は `orthogonal` では使わないため即座に破棄し（`Mat::try_zeros`
    // 分の `f64`／`f32` 作業領域を早期解放）、入力用の `m`（`fill_normal`
    // が確保した `f32` バッファ）も `q` の抽出後は不要になるため同様に
    // 破棄する。`scaled`（同程度のサイズの新規バッファ）を確保する前に
    // ピーク時の同時確保量を減らす目的（advisor 指摘。PR #2239）。
    drop(r);
    drop(m);
    // `q` の shape は `[gen_rows, gen_cols]`（`transposed` の場合は
    // 転置前の形状のまま）。`q.transpose_2d()?.contiguous()` を呼ぶと
    // `rows*cols` 要素の `f32` をもう 1 回非フォールブルに確保する
    // （`Tensor::contiguous` は `tensor-core` 全体で共有される既存の
    // 非フォールブル実装であり本 PR のスコープ外）ため、ここでは
    // 呼ばず `scaled`（[`try_alloc`] でフォールブル確保済み）へ
    // 転置とスケールを同時に書き込む（codex-review 指摘「テンソル
    // 変換」の是正。PR #2239）。
    // `q` は直前の `Mat::try_to_tensor`（`Tensor::new` で `offset=0`・
    // row-major strides の新規構築）が返した値のため必ず contiguous
    // であり、`as_slice()` は常に `Some`（追加確保なしの借用）を返す
    // 契約になる。`host_slice()`（非 contiguous 時に内部で非フォール
    // ブルな `Tensor::contiguous()` を呼ぶ）は使わない——`eval::linalg::
    // Mat::try_from_tensor` で同種の危険を除去した意図（codex-review
    // 再指摘。PR #2239）を呼び出し元側でも一貫させるため、ここでも
    // 明示的に contiguous 前提を検査してから `as_slice()` のみを使う。
    let q_slice = q.as_slice().ok_or_else(|| {
        invalid_argument(
            "nn::init::orthogonal: qr の出力が contiguous ではありません（内部契約違反）",
        )
    })?;
    let mut scaled = try_alloc(gen_numel)?;
    if transposed {
        // `out[i, j] = q[j, i] * gain`（`q` は行優先 `[cols, rows]`
        // 形状——`gen_rows = cols`・`gen_cols = rows`——のデータで、
        // `q[j, i]` は `q_slice[j * rows + i]` に対応する）。
        for i in 0..rows {
            for j in 0..cols {
                scaled.push(q_slice[j * rows + i] * gain);
            }
        }
    } else {
        scaled.extend(q_slice.iter().map(|&v| v * gain));
    }
    Tensor::new(scaled, shape).map_err(AutodiffError::from)
}

/// `trunc_normal` が無限ループへ陥らないための試行回数上限（要素あたり
/// 平均試行数の上限。DoS 対策——`[a, b]` が `N(mean, std²)` の裾に
/// ほとんど掛からない窓の場合、rejection sampling は原理的に停止しない
/// ため、実用上十分大きい値で fail-closed に打ち切る）。
const TRUNC_NORMAL_MAX_ATTEMPTS_PER_ELEMENT: usize = 10_000;

/// PyTorch `nn.init.trunc_normal_` 相当。`N(mean, std²)` を `[a, b]` へ
/// 切断した分布から `shape` の全要素をサンプルする。逆 CDF 法に必要な
/// erfinv を自作せず、rejection sampling（Box–Muller の各サンプルを
/// `[a, b]` 外なら棄却して引き直す）で実装する
/// （`docs/facade-nn-init-exposure-decision.md` 参照）。
///
/// `a >= b`（空・逆転区間）・非有限値・`std < 0` は
/// `AutodiffError::InvalidArgument` を返す。`std == 0` は `mean` が
/// `[a, b]` 内であることを検査したうえで [`constant`] にフォールバック
/// する。受理確率が極端に低い窓（試行上限
/// `TRUNC_NORMAL_MAX_ATTEMPTS_PER_ELEMENT` を要素平均で超過）も
/// `AutodiffError::InvalidArgument` で打ち切る。
pub fn trunc_normal(
    shape: &[usize],
    mean: f32,
    std: f32,
    a: f32,
    b: f32,
) -> Result<Tensor<f32>, AutodiffError> {
    if !mean.is_finite() || !a.is_finite() || !b.is_finite() {
        return Err(invalid_argument(
            "nn::init::trunc_normal: mean・a・b は有限である必要があります",
        ));
    }
    if !std.is_finite() || std < 0.0 {
        return Err(invalid_argument(
            "nn::init::trunc_normal: std は有限かつ非負である必要があります",
        ));
    }
    if a >= b {
        return Err(invalid_argument(
            "nn::init::trunc_normal: a は b 未満である必要があります",
        ));
    }
    let numel = checked_numel(shape)?;
    if std == 0.0 {
        if mean < a || mean > b {
            return Err(invalid_argument(
                "nn::init::trunc_normal: std == 0 のとき mean は [a, b] 内である必要があります",
            ));
        }
        return constant(shape, mean);
    }
    let mut out = try_alloc(numel)?;
    let attempt_budget = numel.saturating_mul(TRUNC_NORMAL_MAX_ATTEMPTS_PER_ELEMENT);
    // `mean`／`std`／窓比較を `f64` で行う（codex-review 同類型点検・
    // `fill_normal` と同じ理由。PR #2239）: `z * std` を `f32` のまま
    // 計算すると、`z`・`std` が個別に有限でも中間積が `f32` の表現範囲を
    // 超えて `inf` になりうる。窓 `[a, b]` の受理判定自体は「`inf` は
    // `<= b` を満たさず自動的に棄却される」ため誤った値が出力される
    // ことはなかったが、本来受理されるべき（真の値が `[a, b]` 内の）
    // サンプルまで中間 overflow のせいで誤って棄却され、`std` が極端に
    // 大きい設定では試行回数上限（`TRUNC_NORMAL_MAX_ATTEMPTS_PER_ELEMENT`）
    // を無駄に消費しうる欠陥だった。`mean`／`std`／`a`／`b` を `f64` へ
    // 昇格し、判定・出力とも `f64` で行ってから受理時にのみ 1 回
    // `f32` へダウンキャストする。
    let mean_f64 = f64::from(mean);
    let std_f64 = f64::from(std);
    let a_f64 = f64::from(a);
    let b_f64 = f64::from(b);
    let fill_result: Result<(), AutodiffError> = with_global_rng(|rng| {
        let mut generated = 0usize;
        let mut attempts = 0usize;
        while generated < numel {
            if attempts >= attempt_budget {
                return Err(invalid_argument(
                    "nn::init::trunc_normal: 試行回数上限に達しました（[a, b] の受理確率が極端に低い可能性があります）",
                ));
            }
            attempts += 1;
            // Box–Muller は本来 2 値ずつ生成するが、rejection sampling は
            // 値ごとに独立採否判定が必要なため、ここでは 1 回の変換で
            // 得られる `cos` 側の値のみを使う（`sin` 側は捨てる。
            // `fill_normal`〈#2140 の非切断版〉とは異なる消費契約になる
            // ことをこの関数のスコープに閉じる）。
            let u1 = 1.0 - rng.next_unit_f64();
            let u2 = rng.next_unit_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            let z = r * theta.cos();
            let value_f64 = z * std_f64 + mean_f64;
            if value_f64 >= a_f64 && value_f64 <= b_f64 {
                out.push(value_f64 as f32);
                generated += 1;
            }
        }
        Ok(())
    });
    fill_result?;
    Tensor::new(out, shape).map_err(AutodiffError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// codex-review 指摘の回帰（イシュー #2140・PR #2239）: `f64` 計算
    /// →`f32` ダウンキャストの最近接丸めで `v == high` になったケースが
    /// `high` の直前の有限 `f32` へ補正されることを、乱数を経由せず
    /// `exclude_high` を直接呼んで固定する（決定的）。
    #[test]
    fn exclude_high_corrects_value_rounded_up_to_high() {
        let low = 1.0f32;
        let high = f32::from_bits(low.to_bits() + 1); // `low` の次に表現可能な f32
        // `v == high`（最近接丸めで上限へ丸め上がった状況を模す）。
        let corrected = exclude_high(high, low, high);
        assert!(corrected < high, "high 未満へ補正されていない: {corrected}");
        assert!(corrected >= low, "low を下回って補正された: {corrected}");
        assert_eq!(
            corrected, low,
            "low と high が隣接する場合は low に一致するはず"
        );
    }

    #[test]
    fn exclude_high_leaves_values_strictly_below_high_unchanged() {
        assert_eq!(exclude_high(0.5, 0.0, 1.0), 0.5);
    }

    #[test]
    fn exclude_high_degenerate_low_equals_high_returns_low() {
        // `low == high` の縮退ケース（`uniform_low_equals_high_is_constant`
        // が固定する「定数 low を返す」契約）。
        let corrected = exclude_high(1.5, 1.5, 1.5);
        assert_eq!(corrected, 1.5);
    }

    // エンドツーエンドの決定的回帰（`low`・`high` を隣接する正の `f32`
    // に設定し、全出力が `high` 未満に収まることを確認する）は
    // `manual_seed`／並列テスト直列化用ロックを共有する
    // `crates/autodiff/tests/nn_init.rs::uniform_never_returns_high_for_
    // adjacent_f32_bounds` に置く（本モジュールの `#[cfg(test)]` は
    // プロセスグローバル RNG の直列化インフラ〈`test_lock`〉を持たない
    // ため）。

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
