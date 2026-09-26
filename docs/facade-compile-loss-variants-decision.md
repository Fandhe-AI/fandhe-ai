# compile() の Loss enum 拡張（BCE・BCEWithLogits・NLL・KLDiv・Huber・SmoothL1・L1）設計判断記録

イシュー #2169（親 #2131「PyTorch／TF 置き換えの API 網羅（対応表の
行内深掘り）」）。`docs/facade-nn-module-exposure-decision.md`・
`docs/autodiff-optimizer-adadelta-adamax-nadam-radam-decision.md` §8
と同型の記録。

## §0 結論

`compat::Loss`（`crates/facade/src/compat/training.rs`）への facade
公開面の拡張は**未承認のため保留**する。コード変更は保留ガード
（`crates/facade/src/lib.rs::CompileLossVariantsHoldDoctestGuard`＋
`crates/facade/tests/api_surface.rs` の否定ガード）のみで、`Loss`
enum・`FitTarget::loss_for` 本体（`crates/facade/src/compat/
training.rs`）は一切変更しない。

## §1 背景

- イシュー #2169 本文の「承認事項」節は「facade 公開面の拡張
  （`docs/compat-api-scope.md` §5 経路 2）: `Loss` enum への variant
  追加」を**承認待ちの事項として明記**している。イシューへのコメントは
  0 件で、承認は記録されていない（`gh issue view 2169 --json comments`
  で確認済み）
- 親 #2131 は、このツリーで facade 公開面を拡張する手順を「設計判断
  記録 → 承認 → 実装」の 2 段と定めている（`docs/compat-api-scope.md`
  §5）
- 同じツリーで本文に承認事項節を持つ他イシュー（#2133・#2140・#2146・
  #2173）は、いずれも facade 公開面を変更せず、保留ガードと決定記録で
  閉じている（`docs/facade-nn-module-exposure-decision.md` §12・
  `docs/autodiff-activation-ops-decision.md` §6・`docs/
  autodiff-param-groups-decision.md` 等）
- 例外的に実装まで進んだ #2065・#2068 の適用記録（`docs/
  compat-api-scope.md` §5）は、根拠として「イシュー本文が承認事項として
  列挙していない」ことを挙げている。#2169 はこの条件を満たさない
- `Loss::L1` は内部の `fandhe_ai_autodiff::loss_ops::l1_loss` を経由する
  設計になる（§2 参照）。これは別の保留事項である
  `LossOpsHoldDoctestGuard`（`docs/autodiff-loss-ops-decision.md` §5）の
  対象でもあり、#2169 の承認とは独立に `loss_ops::l1_loss` の facade
  到達自体も未承認のまま二重に保留されている
- #2170（`compat::Optimizer` enum への variant 追加）が実装されたのは、
  イシュー本文で「承認事項に該当しない」と明示されていたためであり、
  前提が #2169 とは異なる（`docs/compat-api-scope.md` 229 行）

### イシュー本文の不整合（記録のみ）

イシュー #2169 の受け入れ条件は 7 種を「BCE・BCEWithLogits・NLL・
KLDiv・Huber・SmoothL1・L1」と列挙する一方、承認事項節は「BCE・NLL・
KLDiv・Huber・SmoothL1・L1・CTC」と列挙しており、`BCEWithLogits` の
有無・`CTC` の有無が食い違っている。**本記録は受け入れ条件の集合
（BCEWithLogits を含み CTC を含まない 7 種）を正**として §2 以降を
書く。CTC を対象外とする理由は §3 で述べる。

## §2 承認後の実装仕様（承認後にそのまま実装へ移れる粒度）

### variant 名と対応する既存 API

`crates/autodiff/src/var.rs`・`crates/autodiff/src/loss_ops.rs` の
既存公開シグネチャを実測確認済み（下表）。

| `Loss` variant | 経由する既存 API | 備考 |
|---|---|---|
| `Bce` | `Var::bce_loss(&self, target: &Var, reduction: Reduction)` | `[0, 1]` 範囲検査は `bce_loss_impl` 内部で実施済み |
| `BceWithLogits` | `Var::bce_with_logits_loss(&self, target: &Var, reduction: Reduction)` | 範囲検査なし（logits 入力） |
| `Nll` | `Var::nll_loss(&self, targets: &Tensor<i32>, class_dim: usize, reduction: Reduction)` | `pred` は log 確率 `[N, C]`。`class_dim = 1` 固定 |
| `KlDiv` | `Var::kl_div_loss(&self, target: &Var, reduction: Reduction)` | `pred` は log 確率、`target` は確率（`log_target = false` 固定・`kl_div_loss_with_log_target` は使わない） |
| `Huber` | `Var::huber_loss(&self, target: &Var, delta: f32, reduction: Reduction)` | `delta = 1.0` 固定（§2.1「unit variant に限定する理由」参照） |
| `SmoothL1` | `Var::smooth_l1_loss(&self, target: &Var, beta: f32, reduction: Reduction)` | `beta = 1.0` 固定（PyTorch 既定値と同じ） |
| `L1` | `fandhe_ai_autodiff::loss_ops::l1_loss(pred: &Var, target: &Var, reduction: Reduction)` | facade 非公開の自由関数（`use` 経由）。§1 の二重保留に注意 |

いずれも既存の `Reduction::Mean` 固定方針（`Mse`／`CrossEntropy` と
同じ）を踏襲する。

### §2.1 unit variant に限定する理由

`Loss` は `#[derive(Debug, Clone, Copy, PartialEq, Eq)]` で `Eq` を
実装している。`Huber`／`SmoothL1` の `delta`／`beta` を `f32`
ペイロードとして持たせると `Eq` を外す必要があり、0.9.0 で公開済みの
`Loss` の `Eq` 実装（トレイト実装の削除は破壊的変更）を壊す。このため
本記録では両 variant とも unit variant のまま `delta = 1.0`／
`beta = 1.0`（PyTorch 既定値）に固定する案を推奨する。パラメータを
可変にしたい場合は `Eq` 互換のラッパー型（bit 表現比較。`f32` は
全順序を持たないため単純な `derive(Eq)` はできない）を別途設計し、
改めて承認を得る必要がある（§4 承認事項③）。`KlDiv` も同じ理由で
`log_target = false` 固定とする。

### §2.2 `FitTarget::loss_for` の dtype 写像

`crates/facade/src/compat/training.rs` の `impl FitTarget for f32`／
`impl FitTarget for i32` は現在 `Loss::Mse`／`Loss::CrossEntropy` を
match する非網羅（同一クレート内 exhaustive match。`#[non_exhaustive]`
は外部クレートにのみ効く）で書かれている。7 variant 追加時は両
`loss_for` の match を拡張する。

- `f32` target 側（`impl FitTarget for f32`）:
  `Bce`／`BceWithLogits`／`KlDiv`／`Huber`／`SmoothL1`／`L1` を
  `target_var = tape.var_no_grad(target_batch)` に対する各 `Var`
  メソッド呼び出しへ写像し、`Nll`／`CrossEntropy` は既存と同じ
  `InvalidArgument` で拒否する
- `i32` target 側（`impl FitTarget for i32`）:
  `Nll` を `pred.nll_loss(target_batch, 1, Reduction::Mean)` へ写像し、
  それ以外（`Bce`／`BceWithLogits`／`KlDiv`／`Huber`／`SmoothL1`／`L1`）
  は既存の `Loss::Mse` 分岐と同型の `InvalidArgument` で拒否する
- `require_metrics_support`（`f32` 実装のみ上書きして拒否）は変更不要
  ——分類 metrics は「クラス添字 target を使う loss」でのみ意味を持つ
  ため、`f32` 実装は既存のまま全 loss を拒否し続ける。ただし拒否文言
  「metrics は Loss::CrossEntropy（Tensor<i32> target）でのみ計算
  できる」は `Nll` も対象に含めるよう「クラス添字 target（`Tensor<i32>`:
  CrossEntropy／Nll）」へ更新する（`i32` 側で `Nll` を追加するため）
- `as_class_targets`（`i32` 実装のみ `Some` を返す）は変更不要

### §2.3 事前の内部結線を行わない理由

承認前に `Loss` へ variant を追加せず、`loss_for` の分岐も追加しない
（コード変更は保留ガードのみに限る）。承認前に到達不能なコードを
追加すると `cargo clippy --all-targets -D warnings` の `dead_code` に
抵触し、`.claude/rules/coding-rust.md`「`#[allow]` の安易な追加で
黙らせない」規約に反するため。

### §2.4 承認後に追加するテストの方針

- `crates/facade/tests/compat_sequential_fit_losses.rs`（仮）: 各
  variant で `fit` した損失値が、同じバッチに対し `Var` を直接呼んだ
  結果と CPU 上で bit 完全一致すること
- loss × dtype の不整合（例: `Nll` × `f32`、`Bce` × `i32`）が
  `AutodiffError::InvalidArgument` になることの網羅テスト
- `fit_types_are_reachable_via_facade_only`（既存）へ 7 variant の
  到達確認を追加する
- CUDA／Metal parity は `#[ignore]` で分離し、`docs/perf/logs/
  compile-loss-variants-2169/` へ実機実測を申し送る

## §3 CTC を `compile()` の対象外とする理由

イシュー本文の承認事項節は CTC を列挙しているが、受け入れ条件（§1 の
不整合参照）には含まれない。`fandhe_ai_autodiff::loss_ops::ctc_loss`
（#2168）は `target` に加え `input_lengths`／`target_lengths` の 2 本
の長さテンソルを追加引数として要求するため、`FitTarget::loss_for`
の「`pred`・`target_batch` の 2 引数のみ」という形に構造的に収まらない
（`Sequential::fit`／`evaluate` の `(x, y)` バッチ契約を破壊せずには
拡張できない）。本記録は受け入れ条件の集合を正としてこれを対象外と
確定し、新規 Issue は起票しない（承認なしに起票しない規約
`out-of-scope-tracking.md` に従う）。

## §4 承認事項

1. `Loss` への 7 unit variant（`Bce`／`BceWithLogits`／`Nll`／
   `KlDiv`／`Huber`／`SmoothL1`／`L1`）の追加（`docs/compat-api-scope.md`
   §5 経路 2）
2. `Loss::L1` の到達に伴う `loss_ops::l1_loss` の facade 到達
   （`docs/autodiff-loss-ops-decision.md` §5 の保留事項）との同時承認
3. `Huber`／`SmoothL1` のパラメータ（`delta`／`beta`）を可変にしたい
   場合の `Eq` 互換ペイロード型設計（§2.1 参照。本記録のスコープ外）

## §5 保留ガードの多層防御

`OptimizerExtHoldDoctestGuard`（`docs/autodiff-optimizer-adadelta-
adamax-nadam-radam-decision.md` §8）と同型の「正のプローブ 1 ブロック
方式」を採る。

1. **正のプローブ doctest**（`crates/facade/src/lib.rs::
   CompileLossVariantsHoldDoctestGuard`）: facade の全 `pub mod` を
   glob import したスコープに、`__FandheLossVariantHoldProbe` トレイト
   （関連 const `Bce`／`BceWithLogits`／`Nll`／`KlDiv`／`Huber`／
   `SmoothL1`／`L1` を持つ）を `compat::Loss` へ実装し、型相対パス
   `Loss::Bce` 等がこのトレイトの関連 const に解決されることを固定
   する。variant が追加されると型不一致または `ambiguous_associated_
   items` でコンパイルが失敗する（`compile_fail` 方式は stable
   rustdoc がエラーコードを照合しないため採らない）
2. **doctest ドリフト検査**（`crates/facade/tests/api_surface.rs`）:
   `compile_loss_variants_hold_doctest_globs_all_pub_modules`（glob
   import 集合の一致）・`compile_loss_variants_hold_doctest_probe_
   body_matches_fixed_contract`（本文の固定文言一致）
3. **variant inventory（最内層のソース走査ガード）**:
   `compat_loss_enum_variants_are_exactly_mse_and_cross_entropy`
   （`crates/facade/src/compat/training.rs` をトークン走査し
   `pub enum Loss { ... }` の variant 識別子集合が
   `["Mse", "CrossEntropy"]` と完全一致することを固定。プローブが
   正規表記のみを見る穴を塞ぐ主防御）

承認後は 3 層すべてを削除するか、正のガード（実際の 7 variant を
検証するテスト）へ置き換える。

## §6 スコープ外（out-of-scope-tracking.md）

- 承認なしに新規 Issue は起票しない
- CTC の `compile()` 統合（§3 参照）
- `Huber`／`SmoothL1` パラメータ可変化のためのペイロード型設計（§4③）
- CUDA／Metal 実機 parity（承認後の実装時に申し送る）
