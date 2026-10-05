# `bernoulli`・`multinomial`・`normal` と独立 `Generator` の設計判断記録（#2156）

イシュー #2156「`bernoulli`・`multinomial`・`normal`・`Generator` 独立乱数生成」。親 #2131（Phase 5「PyTorch／TF 置き換えの API 網羅」5-A 基盤）。

本ドキュメントは `tensor-core::rng` 側実装（PyTorch `torch.bernoulli`／`torch.multinomial`／`torch.normal`／`torch.Generator` 相当）と、facade 公開面拡張（承認事項）の切り分けを記録する。`crates/facade/src/lib.rs`（保留 doctest 足場 `RngDistributionsHoldDoctestGuard` の追加を除く）・`tests/api_surface.rs`（否定ガード 4 件の追加を除く）・`CLAUDE.md`・`docs/spec/`（正本 submodule）・`Cargo.toml`／`Cargo.lock`・tolerance／baseline・ガードレール閾値・CI／hooks は変更しない。

## 0. 結論・段階

- **内部クレート側（`tensor-core::rng`）は実装済み**（本 PR）。`bernoulli`・`multinomial`・`normal`（自由関数）と `Generator`（`new`／`manual_seed`／`initial_seed`／`Clone`／`Debug`／3 メソッド）。`autodiff` は素通し再エクスポートのみ。
- **facade 公開（承認事項・`docs/compat-api-scope.md` §5 経路 2）は未承認のまま保留**。イシュー #2156 本文には「facade へ 3 関数と `Generator` 型を公開する」契約が明記されているが、**承認前の実施を禁じる**契約であり、#2156 に承認コメントは見当たらない（2026-09-25 確認）。親 #2131 も「設計判断を記録 → 承認 → 実装」の 2 段を求めている。公開形の候補比較と推奨案は §5.1（承認待ち。窓口は #2591〜#2593）。よって `crates/facade/src/**` は保留 doctest 足場（`RngDistributionsHoldDoctestGuard`）を追加するのみで、`bernoulli`／`multinomial`／`normal`／`Generator` は一切公開しない（兄弟イシュー #2144〈`docs/autodiff-matrix-ops-decision.md`〉・#2140〈`docs/facade-nn-init-exposure-decision.md`〉と同型の保留パターン）。
- **本 PR のマージで #2156 は COMPLETED とする**（前例と同じ「保留記録を残した PR のマージで issue をクローズし、承認が得られたら新規 issue か reopen で経路 2 を実施する」方針）。facade 公開面（`fandhe_ai::{bernoulli, multinomial, normal, Generator}`）の実装着手は、`docs/compat-api-scope.md` §5 経路 2 の承認取得後に別 issue／reopen・別 PR で行う。

## 1. API 設計（`crates/tensor-core/src/rng.rs`）

### 1.1 自由関数（グローバル RNG を消費）

- `pub fn bernoulli(probs: &Tensor<f32>) -> Result<Tensor<f32>, RngError>`: PyTorch `torch.bernoulli` 相当。出力は `probs.shape()` と同じ shape、各要素は厳密に `1.0` か `0.0`。全要素の検証（有限・`[0, 1]` 範囲）をロック取得・抽選の前に終えてから抽選に入る（不正入力は乱数を一切消費しない）。1 要素につき `next_unit_f64()` を 1 回引き `u < p` で判定するため整数演算・比較のみでプラットフォーム横断 bit 同一。
- `pub fn multinomial(weights: &Tensor<f32>, num_samples: usize, replacement: bool) -> Result<Tensor<i32>, RngError>`: PyTorch `torch.multinomial` 相当。rank 1（`[n]` → 出力 `[num_samples]`）・rank 2（`[m, n]` → 出力 `[m, num_samples]`）のみ受理し、それ以外は `RngError::Shape(ShapeError::RankMismatch { expected: 2, .. })`。出力 dtype は `i32`（本リポの index／targets 型契約に合わせる意図的な差異。PyTorch 既定の int64 とは異なる）。全行の検証（`n == 0`・`n > i32::MAX`・負／非有限の重み・行和 ≤ 0・非復元抽出でカテゴリ不足）を終えてから抽選に入る。復元抽出は行ごとの累積和（`f64`）と `u * total` の比較で選ぶ（丸めで `idx == n` に達した場合は最後の正の重みへフォールバック）。非復元抽出は選んだ添字の重みを 0 にして総和を取り直しながら `O(num_samples * n)` で抽選する。消費回数はいずれも `m * num_samples` 回固定（加算・乗算・比較のみで libm を通らずプラットフォーム横断 bit 同一）。
- `pub fn normal(mean: f32, std: f32, shape: &[usize]) -> Result<Tensor<f32>, RngError>`: PyTorch `torch.normal` 相当。引数順は `randint(low, high, shape)` に合わせ `mean, std, shape`。`mean` 非有限・`std` 非有限または負は `RngError::InvalidArgument`。`randn`・`nn::init::fill_normal` と同型の Box–Muller（`f64` 中間計算）で `z * std + mean` を最後に 1 回だけ `f32` へ downcast する。`std == 0` でも `randn` と同数（`ceil(numel/2)` 組）を消費し値は厳密に `mean` になる（`nn::init::normal` は `std == 0` で乱数を消費せず `constant` へフォールバックする別契約——§3 参照）。

### 1.2 `Generator`

PyTorch `torch.Generator` 相当。グローバル RNG（`manual_seed`／`with_global_rng`）とは完全に独立した状態を持つ乱数源。

- `pub fn new(seed: u64) -> Self`／`pub fn manual_seed(&mut self, seed: u64)`／`pub fn initial_seed(&self) -> u64`（補正前のシード値を返す）。
- `bernoulli`／`multinomial`／`normal` と同じシグネチャ（`self` は `&self` ではなく `&mut self`）の 3 メソッドを持つ。自由関数と private コア（`bernoulli_core`／`multinomial_core_with_replacement`／`multinomial_core_without_replacement`／`normal_core`）を共有するため、`manual_seed(s); bernoulli(p)` と `Generator::new(s).bernoulli(p)` は bit 完全一致する（両者とも `Xorshift64Star::new(s)` から始まるため。`crates/autodiff/tests/random_parity.rs` で固定）。
- `&mut self` で操作するためロックは不要（`with_global_rng` の `Mutex` とは異なる設計）。`Clone` はフィールド単位で複製する（`Xorshift64Star` 自体に `Clone` は derive しない）。`Debug` は `initial_seed` のみを表示し内部状態は出さない。
- `Tensor`／`Var` への inherent メソッド追加はしない（`Tensor` は facade が再エクスポートしているため、追加するとそれだけで facade へ漏れる）。

### 1.3 `RngError` への variant 追加

既存の `#[non_exhaustive]` enum（`randint` 用に新設。`docs/rng-global-contract-design.md` §10）へ非破壊追加する:

- `InvalidProbability { index: usize }`: `bernoulli`／`multinomial` の確率・重みが非有限、または `bernoulli` では `[0, 1]` 範囲外（最初に違反した論理 index）。
- `InvalidArgument { reason: &'static str }`: `multinomial`／`normal` の引数不正（固定文言）。

## 2. 対象ファイルと変更概要

| パス | 変更内容 |
|------|----------|
| `crates/tensor-core/src/rng.rs` | §1 の実装。unit tests（`global_rng_test_lock()` で直列化） |
| `crates/tensor-core/src/lib.rs` | `pub use rng::{..}` に `Generator, bernoulli, multinomial, normal` を追加。クレート doc に 1 文追記 |
| `crates/autodiff/src/lib.rs` | 素通し `pub use fandhe_ai_tensor_core::rng::{..}` に同じ 4 つを追加（facade 非公開の注記込み） |
| `crates/autodiff/tests/random_parity.rs`（新規） | グローバル経路の決定性・グローバルと `Generator` の bit 一致・`normal` と `nn::init::normal` の bit 一致・2 つの `Generator` の相互不干渉・エラー系の統合テスト |
| `crates/facade/src/lib.rs` | `RngDistributionsHoldDoctestGuard`（`#[cfg(doctest)]` 保留足場。正のプローブ 1 ブロック方式） |
| `crates/facade/tests/api_surface.rs` | `rng_distributions_hold_doctest_globs_all_pub_modules`・`rng_distributions_hold_doctest_probe_body_matches_fixed_contract`・`facade_does_not_reexport_or_declare_rng_distributions`・`workspace_declares_rng_distribution_names_only_in_allowed_locations` の 4 テスト |
| `docs/rng-distributions-generator-decision.md`（本ファイル） | 設計判断記録 |
| `docs/rng-global-contract-design.md` | §13「実装記録（#2156）」を新設 |
| `docs/compat-api-scope.md` | §1.2「乱数生成と RNG 契約」行に #2156 の実装状況・保留を追記 |

## 3. PyTorch との差分（意図的な差異）

- `multinomial` の出力 dtype は `i32`（PyTorch は int64）。本リポの index／targets 型契約（`gather`／`index_select`／`cross_entropy` 等）に合わせる。
- `normal(mean, std, shape)` の `std == 0` は乱数を **消費する**（値は厳密に `mean`）。一方 `nn::init::normal(shape, mean, std)`（イシュー #2140）は `std == 0` で `constant` へフォールバックし乱数を **消費しない**。両者は引数順（`mean, std, shape` 対 `shape, mean, std`）も異なる別 API であり、`std > 0` の場合のみ bit 一致する（`crates/autodiff/tests/random_parity.rs::normal_matches_nn_init_normal_bit_for_bit_when_std_positive`）。この差異は意図的なもので統一しない（`randn` 系と `nn::init` 系がもともと別の消費契約を持つため。`.claude/rules/coding-rust.md` の丸め方針とは独立の軸）。

## 4. 対象外（out-of-scope）

- facade 公開（経路 2。ユーザー承認待ち。現行窓口は #2591・#2592・#2593、形の推奨案は §5.1。旧窓口は #2156・#2131）
- GPU（CUDA／Metal）側の乱数生成（乱数は微分不能な葉値でホスト側だけで完結。`Op`／VJP は追加しない。デバイスへの反映は既存のアップロード経路が担う）
- テンソル値の `mean`／`std` を推定する `normal`（本 issue の `normal` はスカラー `mean`／`std` から新規生成する方のみ）
- `Generator` の状態 get/set・デバイス属性（PyTorch `Generator.get_state`／`set_state`／`device` 相当）
- rank 3 以上の `multinomial`
- 既存 `randn`／`rand`／`randint` の `Generator` 版（本 issue は新規 3 分布と `Generator` 型のみが対象）
- `nn::init` の `Generator` 対応

## 5. facade 保留と承認依頼用の事前設計（多層防御）

保留固定は `KvCacheHoldDoctestGuard`（#2084）・`VarMatrixOpsHoldDoctestGuard`（#2144）と同型の 3 層構成:

1. **正のプローブ doctest**（`crates/facade/src/lib.rs::RngDistributionsHoldDoctestGuard`）: facade の全 `pub mod` を glob import したスコープに、ローカル定義した `bernoulli`／`multinomial`／`normal`（自由関数）・`Generator`（型）と、`Tensor<f32>`／`Var` への同名トレイトメソッドを導入し実際に使う。facade がどの経路（再エクスポート・別名・独自宣言・inherent メソッド追加）でこれらの名前を公開しても、glob 衝突またはシグネチャ不一致でコンパイルが失敗する。
2. **ソース走査ガード**（`crates/facade/tests/api_surface.rs`）: glob 集合のドリフト検査 2 件・facade src 全体の再エクスポート／独自宣言の非存在検査 1 件・workspace 全体の宣言元インベントリ 1 件。
3. **インベントリの期待集合**: `crates/autodiff/src/nn/init.rs::normal`（既存の正規宣言。1 件）＋ `crates/tensor-core/src/rng.rs` の `bernoulli`／`multinomial`／`normal`（各 2 件——自由関数 1 件 + `Generator` の同名メソッド 1 件）。private コア（`*_core` 接尾辞）は別名のためインベントリに現れない。

**将来の干渉（解消済み・イシュー #2504）**: `nn::init` の facade 公開（`docs/facade-nn-init-exposure-decision.md`）が同じ `normal` という名前を持つため、`nn::init` が先に公開された。これに伴い、(1) `RngDistributionsHoldDoctestGuard` の自由関数 `normal` の衝突プローブを、`nn::init` を除く全 `pub mod` を glob した入れ子スコープ `__fandhe_rng_dist_normal_scope` へ分離し（ドリフトは `rng_distributions_normal_scope_globs_all_pub_modules_except_nn_init` が固定）、(2) ソース走査 `scan_rng_distributions_reexports_and_declarations` は `src/nn/init.rs`・接頭辞 `fandhe_ai_autodiff::nn::init::`・別名なしの `normal` に限り経路限定で許可した（それ以外の経路は従来どおり違反）。#2591 の保留（`tensor_core::rng::normal` を facade へ出さない）は弱めていない。

**承認後の切り替え手順**（形は §5.1 の承認結果で確定する。下記は #2156 時点の参考で、推奨案〈委譲 `pub fn`＋`Generator` の `pub use`〉とは手順 1・2 が異なる。実施しない。承認取得後の参考記録。#2591 で公開する際は、上記の分離した `normal` プローブ・入れ子スコープ・ソース走査の経路限定許可も合わせて撤去する）:

1. `RngDistributionsHoldDoctestGuard`・対応する否定ガード（`facade_does_not_reexport_or_declare_rng_distributions` 等）を削除する。
2. facade に `pub use fandhe_ai_tensor_core::rng::{bernoulli, multinomial, normal, Generator};` を 1 行追加する。
3. `crates/facade/tests/rng_tensor_generation.rs` 型の facade 到達性テストを追加する。
4. `docs/compat-api-scope.md` §5 の適用記録を更新する。

### 5.1 facade 公開形の候補比較・推奨案・承認依頼（#2592・承認待ち）

**本節は推奨案の記録であり、ユーザー承認の取得を意味しない。確定形ではない。** 親 #2591、実装は #2593（承認コメント確認後にのみ着手）。2026-10-04 の一括承認は「記録済みの推奨形」にのみ及ぶところ、§5 は配置・`nn::init::normal` との同名の扱い・委譲 `pub fn` か `pub use` かを決めていないため、一括承認の対象外である。

#### 5.1.1 着手時判定（基準: `origin/main` = `cb34712b`・2026-10-05）

| 項目 | 事実 | 出典 |
|------|------|------|
| 承認の有無 | #2591・#2592・#2593 にコメント 0 件（承認なし） | `gh issue view` |
| 内部 API | `bernoulli(&Tensor<f32>) -> Result<Tensor<f32>, RngError>`／`multinomial(&Tensor<f32>, usize, bool) -> Result<Tensor<i32>, RngError>`／`normal(f32, f32, &[usize]) -> Result<Tensor<f32>, RngError>`。`Generator` は `new`／`manual_seed`／`initial_seed`＋同名 3 メソッド（`&mut self`）・手書き `Clone`／`Debug` のみ。フィールド非公開・構築は `Generator::new(seed)` のみ | `crates/tensor-core/src/rng.rs` |
| 既存の facade RNG 面 | `manual_seed`／`randn`／`rand`／`randint` は crate 直下の委譲 `pub fn`、`RngError` は直下の `pub use fandhe_ai_tensor_core::RngError;`（`#[non_exhaustive]`） | `crates/facade/src/lib.rs` |
| 同名の既存公開 | `fandhe_ai::nn::init::normal(shape, mean, std) -> Result<Tensor<f32>, AutodiffError>`（純再エクスポート。引数順・`std == 0` の乱数消費・エラー型が `rng::normal` と異なる別機能。§3） | `crates/facade/src/nn/init.rs` |
| 出荷状況 | `rng` の 4 名は `fandhe-ai-tensor-core 0.10.0` に含まれる（`32ed8ce5` は `v0.10.0` の祖先）が内部クレートでサポート対象外。facade の `nn::init`（`3fa28ad5`）は `v0.10.0` に含まれない | `git merge-base --is-ancestor` |
| 決定性の範囲 | `bernoulli`／`multinomial` はプラットフォーム横断 bit 同一。`normal` は Box–Muller（libm の `ln`／`sin`／`cos`）のため同一プロセス・同一プラットフォーム内限定 | §1.1・`docs/rng-global-contract-design.md` |
| 保留ガード | `RngDistributionsHoldDoctestGuard` と `api_surface.rs` の rng 系テスト 6 件（`rng_distributions_hold_doctest_globs_all_pub_modules`／`..._probe_body_matches_fixed_contract`／`scan_rng_distributions_allows_only_approved_nn_init_normal`／`rng_distributions_normal_scope_globs_all_pub_modules_except_nn_init`／`facade_does_not_reexport_or_declare_rng_distributions`／`workspace_declares_rng_distribution_names_only_in_allowed_locations`）。`scan_rng_distributions_reexports_and_declarations` は共有ヘルパー | `crates/facade/tests/api_surface.rs` |
| 保留群の規模 | `*HoldDoctestGuard` 22 個・`*_globs_all_pub_modules*` テスト 23 件 | `grep -c` |
| 名前の非存在 | 保留 doctest が green であることが「`nn::init` 以外の facade モジュールは 4 名を公開していない」証拠（外側は全 `pub mod` を glob して裸の `bernoulli`／`multinomial` と `Generator` を使用。入れ子スコープは `nn::init` 以外を glob して裸の `normal` を使用） | `RngDistributionsHoldDoctestGuard` |
| 既存テストへの波及 | `crates/facade/tests/nn_init.rs` は `nn::init` を明示 import するため直下の `normal` 追加で曖昧にならない | `grep 'use fandhe_ai'` |

#### 5.1.2 候補比較

| 案 | 形 | 評価 |
|---|---|---|
| A: `Var`／`Tensor` 委譲 | `probs.bernoulli()` 等のメソッド | 不採用。3 関数は tape・勾配に触れない非微分のホスト生成で、`normal` は入力テンソルを取らず、`Generator` は型でメソッドに載らない。`Tensor` は facade が再エクスポート済みで inherent 追加が内部変更のまま公開面になる（§1.2 が退けた理由）。`autodiff`／`tensor-core` の変更を伴う |
| B-1: 直下の委譲 `pub fn` 3 件＋`Generator` の `pub use` | `fandhe_ai::{bernoulli, multinomial, normal}`（`fandhe_ai_autodiff::<name>(..)` への 1 式委譲）・`pub use fandhe_ai_tensor_core::Generator;` | **推奨**。`randn`／`rand`／`randint`（委譲 `pub fn`）と `RngError`（`pub use`）の既存の組み合わせと同型。PyTorch の `torch.bernoulli`／`torch.multinomial`／`torch.normal`／`torch.Generator` の配置に対応。新しい `pub mod`・型・trait が不要で変更が facade に閉じる。関数 rustdoc を facade 利用者向けに書ける |
| B-2: 直下の選択再エクスポート 4 名 | `pub use fandhe_ai_tensor_core::rng::{bernoulli, multinomial, normal, Generator};`（§5 の参考手順の形） | 利用側の形は B-1 と同じ。ただし `randn` 系（委譲 `pub fn`）と宣言形が割れ、内部クレート向け doc（イシュー番号・内部パス）が関数 rustdoc にそのまま出る。正ガードが `pub use` 1 行の完全一致で済む利点はある |
| C: 新モジュール | `fandhe_ai::random`（または `rng`／`distributions`）に 4 名 | 直下に `normal` を足さずに済むが、乱数 API が直下（`manual_seed`／`randn`／`rand`／`randint`）とモジュールに割れる。割れを避けて既存 4 関数も移すと入口が二重になる。`pub mod` 追加で保留 doctest 全件（22 ガード・23 glob 検査）の glob 一覧更新が要り、並列 PR と競合しやすい。`random::*` と `nn::init::*` を両方 glob する利用者には `normal` の曖昧性が残る |
| C': モジュール丸ごと再エクスポート | `pub use fandhe_ai_tensor_core::rng;` | 不採用。`Xorshift64Star`・`with_global_rng` まで公開され `randn` 系と入口が二重になる。内部モジュール名が公開名として固定される |
| D: 独自型 | facade に `Generator` の newtype／拡張 trait | 不採用。隠すべき内部型が無く（署名に出るのは `Tensor`・`RngError`・プリミティブのみ）、写像層と doc・テストの二重管理が増えるだけ。`Tensor`／`RngError` を再エクスポートで出す既存方針とも揃わない |
| E: `normal` の改名・別名 | `normal_sample` 等、または `pub use ... as` | 不採用。PyTorch 名との対応が崩れ、facade の「別名なし」慣行に反する。同名はパスで区別できる |

**名前衝突の整理**

- **facade 内**: 直下に 4 名と同名の既存項目は無い（根拠は現に green の保留 doctest。5.1.1 参照）。`normal` は `fandhe_ai::normal` と `fandhe_ai::nn::init::normal` の 2 つになるがパスが異なり定義の衝突ではない（PyTorch の `torch.normal` と `torch.nn.init.normal_` の関係と同じ）。両者は引数順（`mean, std, shape` 対 `shape, mean, std`）・`std == 0` の乱数消費・エラー型が異なる（§3）。引数の取り違えは `&[usize]` と `f32` の型不一致でコンパイル時に検出される。
- **既存の保留群**: 直下配置は `pub mod` を増やさないので glob 一覧の更新が不要。保留 doctest は `fandhe_ai::*` と `fandhe_ai::nn::init::*` を同時に glob するが、glob 同士の曖昧性は使用箇所でのみ生じ、裸の `normal` を使うのは rng ガード自身だけ（#2593 で撤去・書き換え対象）。他ガードのローカル名と 4 名の重なりは基準時点で見当たらない。#2593 で `cargo test -p fandhe-ai --doc` により実測する。
- **既存テスト**: `crates/facade/tests/nn_init.rs` は明示 import のため影響しない。
- **下流利用者**: `use fandhe_ai::*;` と別クレートの glob が同名を持ち、かつ裸で使う場合に限り曖昧性エラーになる（ローカル定義・明示 import は glob に優先）。`nn::init` は 0.10.0 未収録のため、0.10.0 向けコードが `nn::init::normal` との曖昧性で壊れる経路は無い。次版以降は両方を glob して裸の `normal` を使うと曖昧になる点を doc に明記する。公開項目の追加は通常 minor 変更の範囲だが、本記録は互換性の保証を断定しない。

#### 5.1.3 推奨案（承認待ち）

```rust
pub fn bernoulli(probs: &Tensor<f32>) -> Result<Tensor<f32>, RngError>
pub fn multinomial(weights: &Tensor<f32>, num_samples: usize, replacement: bool) -> Result<Tensor<i32>, RngError>
pub fn normal(mean: f32, std: f32, shape: &[usize]) -> Result<Tensor<f32>, RngError>
pub use fandhe_ai_tensor_core::Generator;
```

| 論点 | 推奨 | 比較した他案 |
|------|------|--------------|
| P1 形 | 関数 3 件は委譲 `pub fn`、`Generator` は `pub use` | B-2・D |
| P2 配置 | crate 直下（`manual_seed`／`randn`／`rand`／`randint` の隣） | C |
| P3 `normal` の同名 | 同名のまま直下に置き、`nn::init::normal` とはパスで区別。差異（引数順・`std == 0`・エラー型）を facade 側 `normal` の doc と `nn::init` モジュール doc に明記 | C・E |
| P4 `Var`／`Tensor` のメソッド | 追加しない（§1.2 を維持） | A |
| P5 エラー型 | `RngError` のまま（直下で公開済み・`#[non_exhaustive]`） | facade 独自エラー |
| P6 `Generator` の公開範囲 | 現行の inherent 一式（`new`／`manual_seed`／`initial_seed`／3 メソッド／`Clone`／`Debug`）のまま。`randn`／`rand`／`randint` の `Generator` 版が無い非対称（§4 の対象外）は既知の制限として doc に書き、追加は別 issue | 公開前に `Generator` 版を揃える |
| P7 ガード（#2593 で実施） | 部分反転。自由関数・型名の衝突プローブ、入れ子スコープ `__fandhe_rng_dist_normal_scope`、ソース走査の `nn::init` 経路限定許可は撤去し、「承認形だけを許す」正ガード（直下の `fn` 3 件が各 1 件で本体が 1 式委譲／`Generator` の `pub use` が丁度 1 行・別名なし／それ以外の経路は違反）へ置き換える。`Var`／`Tensor<f32>` への同名メソッド追加を検出するプローブは残す | 保留 doctest の全削除 |

根拠: 既存 RNG 面との一貫性／PyTorch 配置との対応／新しい `pub mod`・型・trait・`Op`・`BackendOps`・VJP・`unsafe`・依存が不要／変更が facade に閉じる／保留 doctest 群の glob 一覧に波及しない／意味論は内部自由関数と同一（検証を先に終えてから乱数を消費する契約を迂回しない 1 式委譲）。

#### 5.1.4 `fandhe-ai =0.10.0` 非破壊の確認

| 観点 | 評価 |
|------|------|
| 追加の種類 | 直下の `pub fn` 3 件と `pub use` 1 件のみ。既存項目の署名・意味論は不変。`FitConfig` 不変 |
| メソッド解決 | `Tensor`／`Var` にメソッドを足さないため、利用者の拡張 trait との解決順は変わらない |
| 公開後に固定されるもの | 3 関数の署名（`normal` の引数順とスカラー限定。テンソル値の `mean`／`std` 版は将来別名で足す必要がある）、`multinomial` の出力 dtype `i32`、`Generator` の inherent 署名（`&mut self`）と `Clone`／`Debug`、フィールド構成から導かれる auto trait。`PartialEq`／`Default` の後付けやメソッド追加は追加のみで行える |
| 決定性の範囲 | `bernoulli`／`multinomial` はプラットフォーム横断 bit 同一、`normal` は同一プロセス・同一プラットフォーム内限定。`manual_seed(s)` 後のグローバル経路と `Generator::new(s)` は bit 一致（`crates/autodiff/tests/random_parity.rs`）。版をまたぐ乱数列の安定性は既存 doc の範囲を超えて断定しない（#2593 で `docs/rng-global-contract-design.md` と文言を合わせる） |
| 依存・unsafe・基準値 | 依存追加なし・新規 `unsafe` なし・`Cargo.toml` 不変・tolerance／baseline／閾値／`docs/spec` 不変 |
| GPU | ホスト生成のみで新規 `Op`／カーネルなし。CUDA／Metal の新規 parity 実測は発生しない見込み（アップロード経路の `#[ignore]` テストは #2593 で `crates/facade/tests/rng_tensor_generation.rs` に倣って判断） |
| セキュリティ | xorshift64* は CSPRNG ではない。#2593 で委譲 `pub fn` 側の doc にも「暗号用途に使わない」注意を書くことを推奨案の条件とする。非復元 `multinomial` は `O(num_samples * n)` である点も doc に記す |

#### 5.1.5 ユーザーに決めてほしい事項

- (a) 公開形: A／B-1／B-2／C／C'／D／E。推奨は B-1
- (b) 配置: crate 直下か新しい `pub mod`（名前も）か。推奨は直下
- (c) `normal` の同名: `nn::init::normal` と同名共存／モジュール分離／改名。推奨は同名共存（doc で差異を明記）
- (d) `Generator` の公開範囲: 現行一式のままか、`randn`／`rand`／`randint` 版を揃えてから公開するか。推奨は現行のまま
- (e) 保留ガード: 全削除か部分反転か。推奨は部分反転（P7）
- (f) 公開せず保留のままにする選択肢
- (g) 承認の残し先: #2593 は自 issue 上の承認コメントを着手条件とするため、承認は **#2593**（または #2592 と #2593 の両方）に、(a)〜(e) の選択を名指しして残してほしい

#### 5.1.6 承認後の実装スケッチ（#2593。本記録では実施しない）

- `crates/facade/src/lib.rs`: `pub fn` 3 件＋`pub use` 1 行、facade のみを import した doctest、`normal` の doc に差異と用途限定注意。ガード doc を (e) に合わせて更新。
- `crates/facade/tests/api_surface.rs`: rng 系 6 テストを P7 の正ガードへ反転。`workspace_declares_rng_distribution_names_only_in_allowed_locations` の期待集合に facade 側宣言を追加。`pub mod` を選んだ場合のみ glob 検査全件を再計数。
- `crates/facade/tests/`: `rng_tensor_generation.rs` 型の到達性テスト（決定性・`Generator` との bit 一致・エラー系）。
- docs: 本記録への実装記録・`docs/compat-api-scope.md` §5 の適用記録・`docs/README.md`。`crates/facade/src/nn/init.rs` のモジュール doc に同名注意を 1 文。
- `Generator` を `pub use` すると tensor-core 側 rustdoc（intra-doc link を含む）が facade の doc にインライン化されるため、`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked` を #2593 で確認する。

#### 5.1.7 本節の位置づけ

承認は未取得。保留ガードとテストは維持し、`crates/`・`Cargo.*`・tolerance・`docs/spec` は不変。facade 公開・ガード反転・`compat-api-scope.md` の適用記録・`Generator` 版 `randn` 系・GPU 側乱数生成・Issue 起票は行わない。
