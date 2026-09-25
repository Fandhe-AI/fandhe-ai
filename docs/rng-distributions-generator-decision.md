# `bernoulli`・`multinomial`・`normal` と独立 `Generator` の設計判断記録（#2156）

イシュー #2156「`bernoulli`・`multinomial`・`normal`・`Generator` 独立乱数生成」。親 #2131（Phase 5「PyTorch／TF 置き換えの API 網羅」5-A 基盤）。

本ドキュメントは `tensor-core::rng` 側実装（PyTorch `torch.bernoulli`／`torch.multinomial`／`torch.normal`／`torch.Generator` 相当）と、facade 公開面拡張（承認事項）の切り分けを記録する。`crates/facade/src/lib.rs`（保留 doctest 足場 `RngDistributionsHoldDoctestGuard` の追加を除く）・`tests/api_surface.rs`（否定ガード 4 件の追加を除く）・`CLAUDE.md`・`docs/spec/`（正本 submodule）・`Cargo.toml`／`Cargo.lock`・tolerance／baseline・ガードレール閾値・CI／hooks は変更しない。

## 0. 結論・段階

- **内部クレート側（`tensor-core::rng`）は実装済み**（本 PR）。`bernoulli`・`multinomial`・`normal`（自由関数）と `Generator`（`new`／`manual_seed`／`initial_seed`／`Clone`／`Debug`／3 メソッド）。`autodiff` は素通し再エクスポートのみ。
- **facade 公開（承認事項・`docs/compat-api-scope.md` §5 経路 2）は未承認のまま保留**。イシュー #2156 本文には「facade へ 3 関数と `Generator` 型を公開する」契約が明記されているが、**承認前の実施を禁じる**契約であり、#2156 に承認コメントは見当たらない（2026-09-25 確認）。親 #2131 も「設計判断を記録 → 承認 → 実装」の 2 段を求めている。よって `crates/facade/src/**` は保留 doctest 足場（`RngDistributionsHoldDoctestGuard`）を追加するのみで、`bernoulli`／`multinomial`／`normal`／`Generator` は一切公開しない（兄弟イシュー #2144〈`docs/autodiff-matrix-ops-decision.md`〉・#2140〈`docs/facade-nn-init-exposure-decision.md`〉と同型の保留パターン）。
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

- facade 公開（経路 2。ユーザー承認待ち。窓口は #2156・#2131）
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

**将来の干渉**: `nn::init` の facade 公開も保留中（`docs/facade-nn-init-exposure-decision.md`）で、同じ `normal` という名前を持つ。どちらかが先に承認された場合、もう一方の doctest プローブ（特に `normal` を含む部分）を見直す必要がある。

**承認後の切り替え手順**（実施しない。承認取得後の参考記録）:

1. `RngDistributionsHoldDoctestGuard`・対応する否定ガード（`facade_does_not_reexport_or_declare_rng_distributions` 等）を削除する。
2. facade に `pub use fandhe_ai_tensor_core::rng::{bernoulli, multinomial, normal, Generator};` を 1 行追加する。
3. `crates/facade/tests/rng_tensor_generation.rs` 型の facade 到達性テストを追加する。
4. `docs/compat-api-scope.md` §5 の適用記録を更新する。
