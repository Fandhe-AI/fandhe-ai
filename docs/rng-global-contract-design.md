# グローバル RNG 契約（manual_seed）の設計

イシュー #1724（親 #1602）。PyTorch `torch.manual_seed` 相当の、プロセス全体で
共有される決定的 RNG 契約を新設する。親イシュー #1602 が求める
`manual_seed`／`randn`／`rand`／`randint`／`arange`／`linspace`／`eye`／
`zeros_like`／`ones_like` のうち、本イシューは**基盤機構である
`manual_seed` のみ**を対象とする。乱数テンソル生成本体（`randn` 等）は
#1725、形状ユーティリティ（`arange` 等）は #1726 が別途対応する。

**#1725 で乱数テンソル生成本体（`randn`／`rand`／`randint`）を実装済み**
（§10「実装記録（#1725）」参照）。**#1726 で決定的生成系（`arange`／
`linspace`／`eye`／`zeros_like`／`ones_like`）を実装済み**（§11「実装
記録（#1726）」参照）。

## 1. 背景・要件

`docs/compat-api-scope.md` §1.2（Tier 1・2026-09-12 改定）は「乱数生成と
RNG 契約」を実装対象範囲として列挙済みであり、前提の spec 改定イシュー
#1591 は CLOSED（クローズ済み）である。本イシューは以下を満たす:

1. 決定的シードに基づく**プロセス全体で共有されるグローバル RNG 状態**を
   持つ（`manual_seed(seed)` → 以後の乱数生成呼び出し列が再現可能になる、
   という PyTorch 同等の契約）。
2. 既存の個別シード API（`nn::Linear::new(.., seed: u64)`、
   `nn::rnn::RnnCell::new(.., seed)` 等）とは独立した別機構であり、
   個別シード API の挙動・シグネチャを変更しない。
3. 将来の `randn`／`rand`／`randint`（#1725）がこのグローバル状態から値を
   引けるようにする内部アクセサを用意する（ホスト生成のみ。`BackendOps`
   ／デバイスカーネルは対象外。#1602 本文の「ホスト生成→アップロード
   方式」）。
4. facade から `manual_seed` を利用者向けに公開する。

## 2. 配置場所

`autodiff` は「`tensor-core` のみに依存する薄いコア」という既存方針
（`crates/autodiff/src/nn/init.rs` 冒頭コメント）があるため、xorshift64*
コア＋グローバル状態は最下層の `tensor-core`（`crates/tensor-core/src/
rng.rs`）に置いた。`Tensor::zeros`／`ones`／`full`（`tensor.rs`）と同じ
「ホスト側だけで完結する生成系」レイヤーに属し、`BackendOps` を経由しない。

```
crates/tensor-core/src/rng.rs
  - pub struct Xorshift64Star { state: u64 }
  - impl Xorshift64Star { pub fn new / next_u64 / next_f32 }
  - グローバル状態: OnceLock<Mutex<Xorshift64Star>>（既定シード 0 で遅延初期化）
  - pub fn manual_seed(seed: u64)          … 状態を丸ごと差し替える
  - pub fn with_global_rng<R>(f: impl FnOnce(&mut Xorshift64Star) -> R) -> R
        … ロックを保持したままクロージャに &mut を渡す内部アクセサ（#1725 が消費する）
  - #[cfg(test)] pub(crate) fn global_rng_test_lock() -> &'static Mutex<()>
```

`crates/autodiff/src/nn/init.rs` は自前の `Xorshift64Star` 定義を削除し、
`fandhe_ai_tensor_core::rng::Xorshift64Star` を使うようリファクタした
（同一クレート内の下位クレート参照であり、既存の「`bench-harness` に
依存しない」という制約とは無関係。アルゴリズムは bit 同一のため既存
テストは無改修で通った）。`crates/bench-harness/src/rng.rs` は意図的な
独立重複のまま**変更しない**（既存方針の維持。層構造上の理由は
`bench-harness::rng` 冒頭コメント参照）。

`autodiff` クレート root（`lib.rs`）は `fandhe_ai_tensor_core::rng::
manual_seed` を素通しで再エクスポートし、`facade` クレート root
（`lib.rs`）が composition root としてさらに薄く委譲する（`tape_for`／
`resolve_ops` の直後、`release_cached_memory` の手前に配置）。

## 3. グローバル状態の実装方針

- `std::sync::{Mutex, OnceLock}` のみを使用（新規依存なし。deps-policy 上
  の承認は不要）。
- **`Mutex` を選んだ理由**（`AtomicU64` による lock-free CAS 案を検討した
  が不採用）: 将来の `randn(shape)`（#1725）は shape の要素数分だけ連続
  して値を引く必要があり、その「一括生成」全体を他スレッドの割り込み
  なしに完結させたい。`with_global_rng` がクロージャ全体でロックを保持
  する設計にすることで、1 回の生成呼び出し全体が他の並行呼び出しに
  割り込まれない単位になる。単純な `AtomicU64::fetch_update` による
  1 語ずつの CAS では、複数値をまとめて引く操作の原子性を保証できない。
- ミューテックスの poison（他スレッドがロック保持中に panic した場合）は
  `.lock().unwrap_or_else(|poisoned| poisoned.into_inner())` で握り潰して
  継続する（本番経路で `unwrap()`／`expect()` を使わない方針。
  `crates/backend-cuda/src/precision.rs::FlagGuard::acquire` と同じ処方）。
- 既定シード（`manual_seed` を一度も呼ばない状態）は固定値 `0`
  （`Xorshift64Star::new(0)` は既存実装がゴールデン比定数へ自動補正する
  ため不動点にならない）とし、「呼び出し前でも決定的」という既存方針
  （PoC-2 発見事項 0・フレーキーテスト回避）を満たす。

## 4. 既存の個別シード API との整合性

- `Linear::new(.., seed)` 等は本イシューの変更を一切受けない。グローバル
  RNG と個別シード RNG は完全に独立したテキスト・状態を持つ（前者は
  プロセスグローバルな `Mutex<Xorshift64Star>`、後者は呼び出しごとに
  新規生成される一時的な `Xorshift64Star` インスタンス）。
- テスト（`crates/autodiff/src/nn/linear.rs::tests::
  linear_new_is_unaffected_by_global_manual_seed_state`）で「`manual_seed(X)`
  を任意の値で複数回呼んでも `Linear::new(.., Y)` の weight／bias が
  変わらないこと」を明示的に確認し、独立性を機構的に固定している。

## 5. スレッド安全性・再現性の範囲

グローバル状態は `Mutex` で直列化するため複数スレッドから安全に呼び出せる
が、他スレッドと同時に `manual_seed`／`with_global_rng` を呼ぶと「どの
スレッドの呼び出しが先に消費するか」は決まらないため、単一スレッド内で
完結する呼び出し列でない限り厳密な呼び出し順序の再現性は保証しない
（PyTorch のグローバル generator も同種の制約を持つ。妥当なトレードオフ
として設計判断とする）。

## 6. facade 公開面

```rust
pub fn manual_seed(seed: u64) {
    fandhe_ai_autodiff::manual_seed(seed);
}
```

`crates/facade/tests/api_surface.rs` に以下を追加し機械検査している:

- `manual_seed_is_reachable_via_facade`: `fandhe_ai::manual_seed` が
  facade から呼び出し可能な `pub fn` であることの型検査。
- `facade_does_not_expose_rng_internal_types`: facade の公開面に
  `Xorshift64Star`／`with_global_rng`（将来 `randn` 等が消費する内部
  アクセサ）が一切露出していないことのソース走査。

## 7. セキュリティ考慮事項（OWASP Top 10）

- **A02 暗号化の失敗**: xorshift64* は暗号学的に安全な PRNG ではない。
  既存の `nn/init.rs`／`bench-harness/rng.rs` と同じ注記（「鍵・トークン
  生成等セキュリティ用途に使用しないこと」）を新規モジュールの doc
  コメントにも明記した。
- **A04 安全でない設計**: グローバル可変状態はマルチスレッド環境で
  「呼び出し順序に依存した再現性」というトレードオフを持つ（§5）。
  この制約は本ドキュメントに明記し、利用者に誤った再現性期待を持たせ
  ない設計とした。
- **A05 セキュリティ設定ミス／可用性**: Mutex poison 時に `unwrap()`／
  `expect()` で panic を連鎖させず `unwrap_or_else(|poisoned| poisoned
  .into_inner())` で復旧し、1 度のパニックが以後すべての RNG 呼び出しを
  恒久的に破壊しないようにした。
- **A06 脆弱・古いコンポーネント**: 新規外部依存を追加していない
  （`std::sync::{Mutex, OnceLock}` のみ）。
- **A08 ソフトウェア・データ整合性**: 既存の決定的シード契約（学習系
  回帰テスト・guardrail の determinism 検証が前提とする「同一シード→
  同一結果」）を壊さないことを、個別シード API との独立性テスト（§4）
  で機構的に担保した。
- **A03 インジェクション**: 外部入力のパースは行わない（`seed: u64` は
  型で閉じた値）ため、本カテゴリの新規リスクはない。

## 8. 対象外（out-of-scope）

- ~~`randn`／`rand`／`randint` の実際の生成関数（#1725 が担当。
  `with_global_rng` を消費する側）~~ → **#1725 で実装済み（§10）**。
- ~~`arange`／`linspace`／`eye`／`zeros_like`／`ones_like`（#1726 が担当。
  RNG 契約は不要）~~ → **#1726 で実装済み（§11）**。
- ~~`nn::Dropout`（#1603）等、将来グローバル RNG を利用する確率的演算。~~
  → **#1603 で実装済み**（`crate::grad::dropout_mask` が
  `fandhe_ai_tensor_core::rng::rand`〈本設計の `with_global_rng` を
  内部で消費する既存関数〉をそのまま呼び、`numel` 回の draw を一様
  乱数から `{0.0, scale}` マスクへ変換する。消費回数契約は
  「`dropout(shape)` は `rand(shape)` と同一の消費列になる」——独自に
  `with_global_rng` を叩き直さないことで機構として保証する。
  `docs/compat-api-scope.md` §1.2「Dropout」行参照）。
  `data::DataLoader` の shuffle は #1615 で実装済み（`with_global_rng`
  を消費するホスト側完結の機構。`docs/dataset-dataloader-design.md`）。
- `Linear::new`／`RnnCell::new` 等、既存の個別シード API のシグネチャ・
  挙動そのものの変更（本イシューは独立性の確認のみ）。
- スレッドローカル RNG・`Tape` 単位の RNG（将来必要になった場合は別途
  設計・ユーザー承認）。
- CUDA／Metal デバイス側の乱数生成カーネル（本設計はホスト生成のみ。
  #1602 本文の「ホスト生成→アップロード方式」の方針どおり）。
- `bench-harness::rng::Xorshift64Star` の重複解消（既存の意図的重複方針
  を変更しない）。

## 9. 依存関係

- 依存: なし（#1602 本文どおり）。
- 被依存: #1725・#1726（本イシュー完了後に着手可能）。

## 10. 実装記録（#1725）

`tensor-core::rng`（`randn`／`rand`／`randint`・`RngError`）を実装し、
`autodiff`（素通し）→ `facade`（`fandhe_ai::{randn, rand, randint,
RngError}`。委譲）まで結線した。`Tensor::zeros`／`ones`／`full` と同じ
「ホスト側だけで完結する生成系」レイヤーに属し、`Op`／`BackendOps`／
VJP は追加していない（生成結果は微分不能な葉値であり、`torch.randn` に
勾配が無いのと同じ）。

- **配置**: `crates/tensor-core/src/rng.rs`（`Xorshift64Star::next_unit_f32`
  ／`next_unit_f64` 補助メソッド新設・`RngError`〈`#[non_exhaustive]`〉
  ・`randn`／`rand`／`randint` 本体）。`crates/autodiff/src/lib.rs`・
  `crates/facade/src/lib.rs` は素通し／薄い委譲のみ。
- **アルゴリズム**:
  - `rand(shape)`: 24bit 整数演算のみ（`next_unit_f32`）で `[0, 1)` を
    生成。プラットフォーム横断で bit 同一の決定性を持つ（ゴールデン値
    回帰テストあり）。
  - `randn(shape)`: Box–Muller 変換（`f64` 中間計算。`u1 = 1.0 -
    next_unit_f64()` で `ln(0)` を回避）で 2 値ずつ生成。`numel` が奇数
    の場合は最後の組の 2 値目を**生成せずに**切り捨てる（1 組につき
    `u1`／`u2` の抽選は必ず行うため `randn(&[5])` と `randn(&[6])` は
    同数の抽選を消費し、抽選後のグローバル RNG 状態は一致する）。
    `ln`／`sin`／`cos`（libm 経由）を使うため、決定性の契約は「同一
    プロセス・同一プラットフォーム内での再現」に限り、クロス
    プラットフォームでの最下位 bit 一致は主張しない（ゴールデン定数
    テストは置かない）。
  - `randint(low, high, shape)`: `range = (high as i64 - low as i64) as
    u64` で `i32::MIN..i32::MAX` の最大区間でもオーバーフローしない。
    rejection sampling（`zone = range.wrapping_mul(u64::MAX / range)`
    未満の値のみ採用）で剰余バイアスを排除。`low >= high` は
    `RngError::InvalidRange` を返す（`shape` 起因の検査より先に判定し、
    範囲不正時は乱数を一切消費しない）。
- **dtype**: `randint` の出力は `i32`（PyTorch 既定の int64 とは異なる
  意図的な差異。本リポの index／targets 型契約——`Var::gather`／
  `index_select`／`cross_entropy` 等——に合わせた）。
- **`shape` の要素数オーバーフロー・アロケーション不能検査**:
  `checked_numel_for::<T>`（`tensor.rs` の `pub(crate)` 関数。要素数積の
  `usize` オーバーフローに加え、`T` 換算のバイトサイズが `Vec` の
  allocation 上限〈`isize::MAX` バイト〉を超えないかも検査する。当初は
  要素数積のみを見る `checked_numel` を使っていたが、`shape =
  &[usize::MAX]` のような shape では要素数積自体は `usize::MAX` の
  ままオーバーフローせず後続の `Vec::with_capacity` が capacity
  overflow で panic しうると PR #1815 の codex-review で指摘され是正
  した）を**グローバル RNG のロック取得前**に呼ぶため、確保不能な
  shape は乱数を一切消費せず `Err` を返す。
- **承認の扱い**: `docs/compat-api-scope.md` §5 は「Tier 1／Tier 2 に
  列挙済みの機能の実装は本節の再適用を要しない」と規定しており、本イ
  シューの対象（乱数生成と RNG 契約）は §1.2 Tier 1 に列挙済みのため、
  #1724 と同じ根拠で追加承認手続きなく着手した。
- **対象外**（変更なし・#1726 等へ引き継ぎ）: `arange`／`linspace`／
  `eye`／`zeros_like`／`ones_like`（#1726 で実装済み。§11）・
  `randn_like`／`rand_like`／`normal(mean, std)`／`uniform_(a, b)`／
  `bernoulli`／`multinomial`／`randperm`・`torch.Generator` 相当の非
  グローバル RNG・CUDA／Metal デバイス側乱数カーネル・`nn::Dropout`
  （#1603 で実装済み。上記「8. 対象外」節参照）・`randint` の int64
  版。`ShapeError` の facade 再エクスポート
  は本イシューで実装済み（`crates/facade/src/lib.rs::ShapeError` 再
  エクスポート。旧版の本節が「対象外」と誤記していた点を PR #1815
  codex-review 指摘〈P2〉により是正）。

## 11. 実装記録（#1726）

`tensor-core::creation`（`arange`／`linspace`／`eye`／`zeros_like`／
`ones_like`・`CreationError`）を実装し、`autodiff`（素通し）→ `facade`
（`fandhe_ai::{arange, linspace, eye, zeros_like, ones_like,
CreationError}`。委譲）まで結線した。`rng`（§10）の非乱数版カウンター
パートであり、同じく「ホスト側だけで完結する生成系」レイヤーに属する
（`Op`／`BackendOps`／VJP は追加しない。生成結果は微分不能な葉値のため）。

- **配置**: `crates/tensor-core/src/creation.rs`（`CreationError`
  〈`#[non_exhaustive]`〉・5 関数本体）。`crates/autodiff/src/lib.rs`・
  `crates/facade/src/lib.rs` は素通し／薄い委譲のみ（`eye`／
  `zeros_like`／`ones_like` は `tensor-core` 側でジェネリック
  〈`T: Element`〉だが、facade 公開面は他の生成系トップレベル関数と
  同じく `f32` 固定に薄く委譲する）。
- **受入基準テンプレートの非適用**: `#1602` 系列の他イシューが前提と
  する「`Op` 追加・`BackendOps` メソッド追加・`Var` メソッド追加・VJP
  実装・バックエンド間 parity テスト」という受入基準テンプレートは、
  `rng`（§10）と同じ理由で本モジュールには適用しない。「parity」は
  「ホスト生成 → CPU `Tape::var` アップロードの bit 完全一致」＋
  CUDA／Metal `#[ignore]` round-trip（`crates/facade/tests/
  creation_tensor_generation.rs`）で満たす——バックエンド別カーネルが
  存在しないため REQ-2 複合判定の対象自体が無い。
- **数値契約**:
  - `arange(start, end, step)`: 長さを `ceil((end − start) / step)` を
    `f64` で計算（PyTorch CPU の `accscalar_t = double` と同じ方式）。
    各要素は `(start as f64 + i as f64 * step as f64) as f32`。IEEE
    基本演算＋`ceil` のみで構成されるため、プラットフォーム横断で bit
    同一（`rand` と同じくゴールデン値テストで固定）。`start`／`end`
    の大小関係が `step` の符号と矛盾する場合は空テンソルを返す
    （`n <= 0` 判定）。`step == 0` または非有限値は `CreationError::
    InvalidStep` を返す。
  - `linspace(start, end, steps)`: PyTorch の 2 分割方式（前半
    `start + i·step`・後半 `end − (steps − 1 − i)·step`。`step =
    (end − start) / (steps − 1)` を `f64` で計算）を用い、先頭が
    `start`・末尾が `end` と bit 一致することを契約する。`steps == 0`
    は空、`steps == 1` は `[start]`。
  - `eye::<T>(n)`: `i / n == i % n` を満たす対角要素のみ `T::one()`、
    他は `T::zero()`。長方形版（`eye(rows, cols)`）は対象外。
  - `zeros_like`／`ones_like`: `like` の shape のみを引き継ぎ、strides
    （転置・broadcast view 由来の stride 0 等）は一切保存しない新規
    contiguous バッファを返す。broadcast view は shape の要素数が
    ストレージより大きくなりうるため infallible にはできず、`like.
    shape()` の要素数がアロケーション不能な場合は `ShapeError::
    ElementCountOverflow` を返す。
- **`shape` の要素数オーバーフロー・アロケーション不能検査**: `rng`
  （§10）と同じく `checked_numel_for::<T>` をアロケーション前に呼ぶ。
  `arange` は `f64` 中間計算で求めた長さが `usize` の範囲を超える場合
  `usize::MAX` へ丸めてから `checked_numel_for` へ渡し、必ず
  `ElementCountOverflow` として確保前に拒否する。
- **承認の扱い**: `docs/compat-api-scope.md` §5「Tier 1／Tier 2 に
  列挙済みの機能の実装は本節の再適用を要しない」＋ §1.2「乱数生成と
  RNG 契約」行が本イシューを名指しで列挙済み＋ #1602 コメント
  （2026-09-12 ユーザー承認）により、#1724／#1725 と同じ根拠で追加
  承認手続きなく着手した。
- **対象外**: 1 引数／2 引数の `arange(end)`／`arange(start, end)`
  便宜版・i32／i64 版 `arange`・長方形 `eye(rows, cols)`・`full_like`／
  `empty_like`／`randn_like`／`rand_like`・`logspace`・`Var::zeros_like`
  等の `Var` 側メソッド（`tape.var(&zeros_like(&v.to_tensor()))` で
  到達可能）・CUDA／Metal デバイス側の生成カーネル（#1602 のホスト
  生成→アップロード方針どおり）・`compat::array`／`Sequential` の
  変更・`torch.arange` の dtype 推論（整数引数→int64）との完全互換。
