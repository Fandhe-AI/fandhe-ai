# グローバル RNG 契約（manual_seed）の設計

イシュー #1724（親 #1602）。PyTorch `torch.manual_seed` 相当の、プロセス全体で
共有される決定的 RNG 契約を新設する。親イシュー #1602 が求める
`manual_seed`／`randn`／`rand`／`randint`／`arange`／`linspace`／`eye`／
`zeros_like`／`ones_like` のうち、本イシューは**基盤機構である
`manual_seed` のみ**を対象とする。乱数テンソル生成本体（`randn` 等）は
#1725、形状ユーティリティ（`arange` 等）は #1726 が別途対応する。

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

- `randn`／`rand`／`randint` の実際の生成関数（#1725 が担当。
  `with_global_rng` を消費する側）。
- `arange`／`linspace`／`eye`／`zeros_like`／`ones_like`（#1726 が担当。
  RNG 契約は不要）。
- `nn::Dropout`（#1603）等、将来グローバル RNG を利用する確率的演算。
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
