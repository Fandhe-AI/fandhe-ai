# facade への `nn::loss` 損失構造体の公開形（決定記録・承認済み・#2602 で実装済み）

> **2026-10-07 承認済み（推奨案どおり）。** 根拠は ルート #2499 の所有者コメント https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965 の「#2600: 本書 §4・§7 の推奨案（`Reduction` は root に出さない）」。以下の本文は承認前に書かれた推奨案の記録で、実装結果は末尾「10. 実装記録（#2602）」を参照。

> **2026-10-08 追記: 損失のオプション型 5 つの公開パスと損失 5 本の `Var` 委譲は §11 に記録した（#2853。公開は #2854）。**

- イシュー: #2601（本記録の作成と承認依頼）・親 #2600・祖 #2542（Phase 3）・実装は #2602（承認後のみ）。§11 は #2853（親 #2852）・実装は #2854
- 調査基準: main `cb34712b`（2026-10-05）。以下の `file_path:line` はこの時点の実測で、#2602 着手時に再確認する
- 承認状況: 2026-10-07 承認（上記コメント）。#2602 で実装済み

## 0. 結論（承認済みの推奨）

**`fandhe_ai::nn::loss` サブモジュールを新設し、`fandhe_ai_autodiff::nn::loss` の次の 19 名を明示列挙（glob・別名なし）で純再エクスポートする案（b-1・19 名）を推奨する。**

- 損失構造体 14 種: `MseLoss`・`L1Loss`・`HuberLoss`・`SmoothL1Loss`・`BceLoss`・`BceWithLogitsLoss`・`CrossEntropyLoss`・`NllLoss`・`KlDivLoss`・`CosineEmbeddingLoss`・`MarginRankingLoss`・`TripletMarginLoss`・`PoissonNllLoss`・`CtcLoss`
- 引数型 5 種: `Reduction`・`CrossEntropyOptions`・`TripletMarginOptions`・`PoissonNllOptions`・`CtcLossOptions`

決め手は §2.2 の制約である。14 構造体だけを出すと、facade のみの利用者は構築手段を欠く。5 つの引数型は現状 facade から名指しできない。

## 1. 経緯

- #2600 は `crates/autodiff/src/nn/loss.rs` の損失構造体を facade（唯一のサポート公開面。`docs/compat-api-scope.md` §0）へ出すツリーである
- 既存の決定記録に公開形を名指しするものがなく、2026-10-04 の一括承認の対象外である。このため「記録作成（#2601）→ ユーザー承認 → 実装（#2602）」の 2 段で進める
- 先例: `docs/facade-nn-init-exposure-decision.md`（`nn::init`・#2504）、`docs/tensor-core-npy-npz-io-decision.md`・`docs/autodiff-bool-ops-exposure-decision.md` の推奨案記録

## 2. 確定済み内部 API との突合

### 2.1 14 構造体の形（`crates/autodiff/src/nn/loss.rs`）

| 構造体 | derive | フィールド | 構築手段 | `forward` の入力 |
|---|---|---|---|---|
| `MseLoss`・`L1Loss`・`BceLoss`・`BceWithLogitsLoss` | `Debug, Clone, Copy` | 非公開 | `new(Reduction)`・`Default` | `&Var, &Var` |
| `HuberLoss`・`SmoothL1Loss` | 同上 | 非公開 | `new(f32, Reduction)`・`Default` | `&Var, &Var` |
| `CrossEntropyLoss`（250 行付近） | 同上 | **pub**（`class_dim`・`reduction`） | **`new` も `Default` も無い。構造体リテラルのみ** | `&Var, &Tensor<i32>`（`forward_with` は加えて `&CrossEntropyOptions`） |
| `NllLoss`（293 行付近） | 同上 | **pub**（`class_dim`・`reduction`） | `new(usize, Reduction)`・`Default` | `&Var, &Tensor<i32>` |
| `KlDivLoss`（335 行付近） | 同上 | **pub**（`reduction`・`log_target`） | `new`・`new_with_log_target`・`Default` | `&Var, &Var` |
| `CosineEmbeddingLoss`（388 行付近） | 同上 | **pub**（`margin`・`reduction`） | `new(f32, Reduction)`・`Default` | `&Var, &Var, &Tensor<f32>` |
| `MarginRankingLoss`（426 行付近） | 同上 | **pub**（`margin`・`reduction`） | `new(f32, Reduction)`・`Default` | `&Var, &Var, &Tensor<f32>` |
| `TripletMarginLoss`・`PoissonNllLoss`・`CtcLoss` | `Debug, Clone`（`Copy` なし） | 非公開（`options`・`reduction`） | `new(<Options>, Reduction)`・`Default` | `Var` 3 本／`Var` 2 本／`&Var, &Tensor<i32>, &[usize], &[usize]` |

- 戻り値は全て `Result<Var<'t>, AutodiffError>`。14 件とも `#[non_exhaustive]` なし（`crates/autodiff/src/nn/loss.rs` に `non_exhaustive` は 0 件）・`PartialEq` なし
- どのシグネチャにも生の `Tape`・`BackendOps` が現れない（`nn::rnn` のような `Tape` 委譲メソッドは不要）
- 本体は `Var` メソッドまたは `loss_ops` 自由関数への 1 式委譲の薄いラッパーで、入力検査は委譲先が担う
- `nn/loss.rs:44-47` が 4 オプション型と `Reduction` を `pub use` している
- `v0.10.0` タグ時点で 14 構造体と同 `pub use` は `fandhe-ai-autodiff 0.10.0` に出荷済み（`git show v0.10.0:crates/autodiff/src/nn/loss.rs`）。ただし内部クレートはサポート対象外

### 2.2 決め手: 引数型 5 種が facade から名指しできない

- `Reduction`（`crates/autodiff/src/var.rs:53-61`。`#[non_exhaustive]`・`Debug, Clone, Copy, PartialEq, Eq`。**`Default` なし**）と、オプション 4 型（`crates/autodiff/src/loss_ops.rs`。`#[non_exhaustive]`）は facade で再エクスポートされていない。`crates/facade/src/` での `Reduction` の出現は `compat/training.rs:41` の非 `pub` な `use` のみ
- 既知ギャップとして `docs/autodiff-loss-ops-decision.md` の「#2538 追記（残る承認事項）」に記録済み。`api_surface.rs` の到達テスト（14945 行付近）もこの理由で `fandhe_ai_autodiff` から直接 import している
- 帰結 1: 14 構造体だけを出すと、facade のみの利用者は 13 種を `Default::default()` でしか構築できず、`CrossEntropyLoss` は構築手段が無い（リテラル構築には `Reduction` の名指しが要る）
- 帰結 2: pub フィールド `reduction` を持つ構造体を出すと、名指しできない型の値が公開面に現れる
- 帰結 3: `Reduction` を引数に取る公開済み `Var` メソッド（`mse_loss_with`・`huber_loss`・`smooth_l1_loss`・`bce_loss`・`bce_with_logits_loss`・`cross_entropy_loss`・`nll_loss`・`kl_div_loss_with_log_target`・`l1_loss` ほか。`var.rs` の 1419〜2373・4686〜4764 行付近）も、facade のみでは `Reduction` を作れず呼べない

### 2.3 既存の到達経路

- 14 損失は対応する `Var` メソッドで facade から既に到達できる（`mse_loss`／`mse_loss_with`・`huber_loss`・`smooth_l1_loss`・`bce_loss`・`bce_with_logits_loss`・`cross_entropy_loss`・`nll_loss`・`kl_div_loss`・`l1_loss`・`cross_entropy_loss_with`・`cosine_embedding_loss`・`margin_ranking_loss`・`triplet_margin_loss`・`poisson_nll_loss`・`ctc_loss`。後半 7 件は #2538〜#2540）
- `compat::Loss`（`crates/facade/src/compat/training.rs:183`。`compile()`／`fit` 用 enum）は別レイヤで、`Loss::Mse` と `MseLoss` は名前衝突しない
- 構造体の価値は、設定（reduction・margin 等）を保持するオブジェクト形（PyTorch `nn.MSELoss()` 相当）を `Var` を直接扱う手動ループへ与える点にある

## 3. 候補比較

| 案 | 内容 | 評価 |
|---|---|---|
| (a) `Var` 委譲のみ | 構造体は出さない（現状維持） | 非破壊だが、親 #2600 の目的（構造体の公開）を満たさない。5 型を名指しできない穴も残る |
| (a') 引数型 5 種のみ出す | `Reduction` とオプション 4 型だけ | 穴は最小の名前数で塞がる。ただし #2600 の目的を満たさない（参考案） |
| **(b-1) `nn::loss` に純再エクスポート** | `crates/facade/src/nn/loss.rs` を新設し `pub use fandhe_ai_autodiff::nn::loss::{…};` を明示列挙。`nn/mod.rs` に `pub mod loss;` | 内部パスと 1 対 1。`nn::init`（#2504）・`nn::rnn` と同型で、3 点セットの正ガードを鏡写しにできる。ロジック重複なし。**推奨** |
| (b-2) `nn/mod.rs` へ平らに `pub use` | `Transformer` 3 名の先例・`torch.nn.MSELoss` の平らな配置に対応 | `pub mod` が増えず glob 波及が無い。ただし `use fandhe_ai::nn::*` のスコープに最大 19 名が入り、`Reduction` が `nn` 直下に出る。`nn/mod.rs` の整理（層は `compat::Sequential::add_*` 経由）との関係説明が要る |
| (b-3) モジュール別名 `pub use fandhe_ai_autodiff::nn::loss;` | 1 行で済む | 集合を列挙で固定できず、内部側の追加が無審査で公開面へ流れる |
| (b-4) クレート root 直下 | `fandhe_ai::MseLoss` | `use fandhe_ai::*` のスコープに多数の名前が入る |
| (b-5) `compat` 配下 | | `compat` は numpy／Keras 形状の層で、`Var` を直接扱う手動ループ用の型とは層が違う |
| (c) facade 独自型 | ラッパー newtype・`#[non_exhaustive]`・ビルダー | 隠すべき内部型が無い。14 型の二重定義で doc・インベントリが二重管理になり、REQ-9「薄いラッパー」に反する方向。利点は pub フィールドと `Copy` の固定を避けられる点 |

(b) 系の再エクスポート集合は 3 水準がある。

| 水準 | 名数 | 帰結 |
|---|---|---|
| 14 名のみ | 14 | §2.2 の帰結 1〜3 が残る（`CrossEntropyLoss` は構築不能） |
| 14 + `Reduction` | 15 | 構築手段は揃う。オプション型を取る `new`／`forward_with` は既定値のみ使える |
| 14 + `Reduction` + オプション 4 型 | 19 | 全引数型を名指しできる。**推奨** |

## 4. 推奨案（承認済み）

**(b-1) で 19 名。**

- 根拠: §2.2 の帰結 1〜3 を同時に解消する。型の実体は autodiff と同一なので、既存 `Var` メソッドへそのまま渡せる。facade にロジックを置かない。新規 `Op`／`BackendOps`／VJP／`unsafe`／依存なし
- `Reduction` とオプション 4 型は `nn::loss` 経路 1 本のみとし、root 直下には出さない（配置の代替は §7 の付随事項）
- PyTorch の `torch.nn.MSELoss` 平置きに近いのは b-2 だが、(b-1) は内部クレートのパスと一致し、ガードを機械的に書ける利点を取る

## 5. `fandhe-ai =0.10.0` 非破壊の確認

| 観点 | 判定 |
|---|---|
| 追加物 | `pub mod loss` 1 件と `pub use` 19 名のみ。既存項目の署名・意味論は不変。`FitConfig` 不変 |
| 名前衝突 | root に名前を足さない。`use fandhe_ai::nn::*` 利用者のスコープに `loss`（型名前空間のモジュール）が増える。ローカル変数 `loss` は値名前空間で影響しない。glob 同士の曖昧さは使用箇所でのみ顕在化する。facade `src/` に 14 名の同名ローカル定義は無い（grep 実測）。互換性を保証するとは断定せず、#2602 で `cargo test -p fandhe-ai --doc` により実測する |
| 依存・`unsafe`・OS | 追加なし。std のみ・cfg 分岐なし |

公開後に固定されるもの（利用者に見える契約）:

- pub フィールドを持つ 5 構造体（`CrossEntropyLoss`・`NllLoss`・`KlDivLoss`・`CosineEmbeddingLoss`・`MarginRankingLoss`）は `#[non_exhaustive]` が無く、フィールド追加が破壊的になる
- `Copy` を持つ 11 構造体は `Copy` を外せない（将来 `weight: Tensor` 等を持たせられない）
- `new`／`forward` の署名
- `#[non_exhaustive]` の後付けは、出荷済み `fandhe-ai-autodiff 0.10.0` の構造体リテラル構築を壊すため提案しない。現状形のまま公開する

## 6. 既存保留群との整合

- 14 構造体名を名指しする保留ガード（`*HoldDoctestGuard`・否定テスト）は存在しない。`crates/facade/src/` に 14 名の出現は無い
- `fandhe_ai::nn` への追加を現に止めているのは次の 2 件である
  - `nn_mod_declares_only_init_and_rnn_submodules`（`crates/facade/tests/api_surface.rs:4737`）: `pub mod` は `init`・`rnn` の完全一致
  - `nn_mod_public_items_match_expected_set`（同 `:20914`）: `nn/mod.rs` の公開 item 集合の完全一致
- root や他モジュールへの `pub use fandhe_ai_autodiff::nn::loss::…` を止める専用ガードは、`nn::loss` を grep した範囲では見つからなかった。#2602 で承認形以外の経路を拒否する正ガードを新設する
- `facade_does_not_reexport_or_declare_loss_ops`（同 `:14768`）は、`pub use` 行が識別子 `loss_ops` を含む場合と、7 関数名の `fn` 宣言を拒否する。`nn::loss::…` 経路の `pub use` は `loss_ops` を含まないため抵触しない（テスト本体を読んで確認）
- `LossOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`）の doc は「オプション型の拒否は維持する」と述べるが、オプション型名を機械的に拒否するガードは見つからなかった（`api_surface.rs` での 5 型の出現は到達テストの import のみ）。**文言はあるが機械ガードは無い。本件の承認は、この文言を #2602 で上書きすることを含む**
- `pub mod` を 1 つ足す場合の波及（#2602 で実施。件数は兄弟 PR で増減するため再計数する）: `*HoldDoctestGuard` の glob 一覧、`*_globs_all_pub_modules` テスト群（調査時 22 件）、`GRAD_SCALER_PROBE_MODULES`（`api_surface.rs:21311`・10 モジュール）、上記 nn 系 2 テストの期待値、`RngDistributionsHoldDoctestGuard` の入れ子スコープ（#2593 で撤去済み）
- 同ツリーで `pub mod` を足しうる兄弟 issue と glob 一覧の編集が競合しうる。rebase 時は双方の行を残す
- `loss_ops` モジュール再エクスポートと `Tensor`／`Tape` 上の同名メソッドの拒否は維持する
- 入力検査（shape・クロステープ・値域）は委譲先の autodiff 側に残り、純再エクスポートのため facade に迂回経路を作らない

## 7. ユーザーに決めてほしい事項（承認依頼）

形を名指しして **#2602（または #2601 と #2602 の両方）** のコメントに残してほしい。#2602 は着手前に自 issue 上の承認を確認する。

- (A) 推奨どおり `fandhe_ai::nn::loss` に 19 名を純再エクスポート
- (B) 14 + `Reduction` の 15 名（オプション型は保留継続。`CrossEntropyLoss::forward_with` と距離・Poisson・CTC は既定オプションのみ）
- (C) 14 名のみ（§2.2 の制約を受容）
- (D) 別の形・配置を指定（b-2／root／独自型 等）
- (E) 保留のまま

付随事項:

- (f) `Reduction` を root にも出すか。推奨: 出さない
- (g) `CrossEntropyLoss` に `new`／`Default` を足すか。autodiff への非破壊追加になるため別 issue。推奨: 本ツリーではしない
- (h) pub フィールド・`Copy` の固定（§5）を受容するか
- (i) #2601 を承認前に本記録の PR マージで閉じてよいか

## 8. 承認後の実装スケッチ（#2602。本記録では実施しない）

- `crates/facade/src/nn/loss.rs` 新設（モジュール doc・facade のみ import の doctest）と `nn/mod.rs` への `pub mod loss;`
- §6 のガード更新、3 点セットの正ガード（`*_reexports_exactly_expected_surface`／`*_is_pure_reexport`／`*_reachable_via_facade_only`）、承認形以外の経路を拒否するソース走査
- 到達テスト（`api_surface.rs:14945` 付近）の import を facade 経路へ切り替えるかの判断
- `LossOpsHoldDoctestGuard` の doc と `docs/compat-api-scope.md` §5 の更新、本記録への実装記録
- 新規カーネルが無いため、CUDA／Metal の新規 `#[ignore]`・実測申し送りは不要

## 9. 本 PR で行わないこと

facade／autodiff のコード変更、ガードの新設・削除・反転、`docs/compat-api-scope.md` の更新、Issue 起票、spec 提案、tolerance・依存の変更。

## 10. 実装記録（#2602）

- 公開: `crates/facade/src/nn/loss.rs` を新設し、`fandhe_ai_autodiff::nn::loss` の 19 名（損失構造体 14・`Reduction`・オプション型 4）を明示列挙で純再エクスポートした。`nn/mod.rs` に `pub mod loss;` を追加。`Reduction` とオプション型の経路はこの 1 本のみ（crate root・`nn` 直下・`loss_ops` 経由には出していない）
- ガード（`crates/facade/tests/api_surface.rs`）: 期待値の拡張 3 件（`nn_mod_declares_only_approved_submodules`・`nn_mod_public_items_match_expected_set`・`GRAD_SCALER_PROBE_MODULES` と glob probe）。正ガードの新設 6 件（`facade_reexports_nn_loss_items_only_in_approved_shape` とその自己テスト・`nn_loss_module_reexports_exactly_expected_surface`・`nn_loss_module_is_pure_reexport`・`nn_loss_items_are_reachable_via_facade_only`）。既存の保留ガードに 14 構造体名を名指しするものは無かったため、反転ではなく期待値拡張と新設で対応した
- `LossOpsHoldDoctestGuard` の doc の「オプション型の拒否は維持する」を、`nn::loss` 経由のみ公開済みと書き換えた（doctest 本文は不変）。全 hold doctest の glob 一覧（43 か所）に `use fandhe_ai::nn::loss::*;` を追加した
- 到達テスト 3 件の import を `fandhe_ai::nn::loss` へ切り替え、facade のみで呼べることを検査する形にした。統合テスト `crates/facade/tests/nn_loss.rs` は 14 構造体の forward 値・勾配が対応する `Var` メソッドと bit 一致することを確認する
- 行わなかったこと: `CrossEntropyLoss` への `new`／`Default` 追加（§7 (g)）、`loss_ops` の再エクスポート、`Tensor`／`Tape` 上の同名メソッド、Phase 4 保留中の損失 3 本の公開、依存・tolerance・baseline・`docs/spec/` の変更、CUDA／Metal の `#[ignore]` テスト（新規カーネルなし）

## 11. オプション型 5 つの公開パスと損失 5 本の `Var` 委譲（#2853・承認済みの形）

### 11.0 結論と承認の根拠

- **オプション型 5 つは `fandhe_ai::nn::loss` の 1 経路だけで公開し、損失 5 本は `Var` の 1 行委譲メソッドにする。** 承認の根拠はルート #2499 の所有者コメント
  https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061（2026-10-08）の「オプション型は `fandhe_ai::nn::loss` へ再エクスポートし、損失は `Var` の 1 行委譲にする」。#2853 側の参照コメントは https://github.com/Fandhe-AI/fandhe-ai/issues/2853#issuecomment-6052737704。
  承認の範囲はこの方向と、それを本節に書いた形に限る。それ以上の承認は主張しない。
- イシュー: #2853（本節）・親 #2852・Phase 7 親 #2841・ルート #2499。公開（コード・ガード）は #2854 で行い、本 PR では何も公開しない。
- 調査基準: main `74171fb9`（2026-10-08）。`file_path:line` はこの時点の実測で、#2854 着手時に再確認する。

### 11.1 オプション型 5 つの型名と公開パス

| 型 | 定義（`crates/autodiff/src/`） | 公開パス |
|---|---|---|
| `BceWithLogitsOptions` | `elementwise_loss_ops.rs:62` | `fandhe_ai::nn::loss::BceWithLogitsOptions` |
| `GaussianNllOptions` | `elementwise_loss_ops.rs:82` | `fandhe_ai::nn::loss::GaussianNllOptions` |
| `MultiMarginOptions` | `margin_focal_loss_ops.rs:62` | `fandhe_ai::nn::loss::MultiMarginOptions` |
| `MultiLabelSoftMarginOptions` | `margin_focal_loss_ops.rs:114` | `fandhe_ai::nn::loss::MultiLabelSoftMarginOptions` |
| `SigmoidFocalLossOptions` | `margin_focal_loss_ops.rs`（`MultiLabelSoftMarginOptions` の直後） | `fandhe_ai::nn::loss::SigmoidFocalLossOptions` |

- 明示列挙・別名なし・glob なしの純再エクスポート。クレートルート・`nn` 直下・モジュール別名（`elementwise_loss_ops`／`margin_focal_loss_ops`）には出さない（#2600 の「`Reduction` を root に出さない」と同じ方針）。
- **`Reduction` は既存の `fandhe_ai::nn::loss::Reduction` の 1 経路のまま**で、経路を増やさない。
- `fandhe_ai::nn::loss` の公開名は 19 名から 24 名（構造体 14・`Reduction`・オプション型 9）になる。§0〜§10 の「19 名」は当時の記録として書き換えない。

### 11.2 内部の引き回し（既存 4 型と同じ方式）

公開形（利用者に見えるパス）は承認どおり 1 つで、内部の引き回しは既存 4 型の方式（`crates/autodiff/src/nn/loss.rs:44-46` が `loss_ops` から `pub use`）から一意に定まる。未承認の新しい選択肢ではない。

- 採用: 内部クレート `crates/autodiff/src/nn/loss.rs` に `pub use crate::elementwise_loss_ops::{BceWithLogitsOptions, GaussianNllOptions};` と `pub use crate::margin_focal_loss_ops::{MultiLabelSoftMarginOptions, MultiMarginOptions, SigmoidFocalLossOptions};` を足し、facade の `crates/facade/src/nn/loss.rs` は従来どおり `fandhe_ai_autodiff::nn::loss::` 接頭辞だけから 5 名を再エクスポートする。内部への追加は `pub use` 2 文の非破壊追加。
- 理由: 正ガード `nn_loss_module_reexports_exactly_expected_surface`（`crates/facade/tests/api_surface.rs`）は facade `nn/loss.rs` の全 `pub use` が接頭辞 `fandhe_ai_autodiff::nn::loss::` を持つことを要求する。この契約を緩めずに済む。
- 却下: facade が `fandhe_ai_autodiff::elementwise_loss_ops::…`／`margin_focal_loss_ops::…` から直接再エクスポートする案。接頭辞契約の緩和が要り、保留中のモジュール名が facade の `pub use` に現れて既存の否定ガード（`ELEMENTWISE_LOSS_IDENTS` 等）と衝突する。

### 11.3 損失 5 本の `Var` 委譲シグネチャ

導出規則: レシーバ = 自由関数の `input`、残りの引数は自由関数の順のまま、戻り値は `Result<Var<'t>, AutodiffError>`、本体は自由関数への 1 式委譲（PR #2837 の 3 本〈`crates/autodiff/src/var.rs` の `hinge_embedding_loss` ほか〉と同じ）。定義場所は `crates/autodiff/src/var.rs` の `impl<'t> Var<'t>`。

```rust
pub fn bce_with_logits_loss_with(
    &self,
    target: &Var<'t>,
    reduction: Reduction,
    options: &BceWithLogitsOptions,
) -> Result<Var<'t>, AutodiffError>
// 本体: crate::elementwise_loss_ops::bce_with_logits_loss_with(self, target, reduction, options)

pub fn gaussian_nll_loss(
    &self,
    target: &Var<'t>,
    var: &Var<'t>,
    options: &GaussianNllOptions,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError>
// 本体: crate::elementwise_loss_ops::gaussian_nll_loss(self, target, var, options, reduction)

pub fn multi_margin_loss(
    &self,
    target: &Tensor<i32>,
    options: &MultiMarginOptions,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError>
// 本体: crate::margin_focal_loss_ops::multi_margin_loss(self, target, options, reduction)

pub fn multilabel_soft_margin_loss(
    &self,
    target: &Tensor<f32>,
    options: &MultiLabelSoftMarginOptions,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError>
// 本体: crate::margin_focal_loss_ops::multilabel_soft_margin_loss(self, target, options, reduction)

pub fn sigmoid_focal_loss(
    &self,
    target: &Tensor<f32>,
    options: &SigmoidFocalLossOptions,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError>
// 本体: crate::margin_focal_loss_ops::sigmoid_focal_loss(self, target, options, reduction)
```

- 利用者から見た型名: オプション型は `fandhe_ai::nn::loss::<型名>`、`Reduction` は `fandhe_ai::nn::loss::Reduction`、`Tensor` は `fandhe_ai::Tensor`、`AutodiffError` は facade の既存公開名。
- **引数順の注意**: `bce_with_logits_loss_with` だけ `(target, reduction, options)` で、他 4 本は `(…, options, reduction)`。自由関数（`elementwise_loss_ops.rs:126`・`:240`、`margin_focal_loss_ops.rs:229`・`:338`・`:381`）の順をそのまま写した結果で、既存の `cross_entropy_loss_with`（reduction → options）と `triplet_margin_loss` 等（options → reduction）の先例に合う。順序を揃える変更は行わない（出荷済みの内部関数の順と食い違い、1 式委譲の固定とも合わなくなる）。

### 11.4 オプション型の構築方法の確認（facade 単独で使えるか）

結論: **5 つとも facade 単独で構築できる。「使えない場合は停止して列挙」の分岐には該当しない。**

| 型 | `Default`（既定値） | ビルダー（`self` 消費・`Self` 返し） |
|---|---|---|
| `BceWithLogitsOptions` | derive（`pos_weight = None`） | `pos_weight(Tensor<f32>)` |
| `GaussianNllOptions` | 手書き（`full = false`・`eps = 1e-6`） | `full(bool)`・`eps(f32)` |
| `MultiMarginOptions` | 手書き（`p = 1`・`margin = 1.0`・`weight = None`） | `p(u8)`・`margin(f32)`・`weight(Tensor<f32>)` |
| `MultiLabelSoftMarginOptions` | derive（`weight = None`） | `weight(Tensor<f32>)` |
| `SigmoidFocalLossOptions` | 手書き（`alpha = Some(0.25)`・`gamma = 2.0`） | `alpha(Option<f32>)`・`gamma(f32)` |

- 構築は `XOptions::default()` に続けてビルダーを連鎖する（例: `MultiMarginOptions::default().p(2).margin(0.5).weight(w)`、`SigmoidFocalLossOptions::default().alpha(None).gamma(0.0)`）。必要な型は `fandhe_ai::Tensor` と素のスカラー型だけで、facade から名指しできない型は引数に現れない。
- 5 つとも `#[non_exhaustive]` かつ全フィールド非公開・`new` なしのため、構造体リテラルでは構築できない（意図どおり）。`Debug + Clone` のみで `Copy`・`PartialEq`・値の読み出し（`pub(crate)` のみ）は提供しない。
- 公開後に契約として固定するもの: 型名、`Default` の既定値（上表）、ビルダーのメソッド名と引数型、`Debug + Clone`。`#[non_exhaustive]` と非公開フィールドにより将来のフィールド追加は非破壊。
- 値の妥当性検査（`p` は 1 か 2、`eps`・`weight`・`pos_weight` は有限かつ非負、`alpha` は `[0, 1]` 等）は構築時ではなく損失の呼び出し時に委譲先の autodiff が型付きエラーで行う。facade は純再エクスポートと 1 式委譲だけで、検査の迂回経路を作らない。

### 11.5 `fandhe-ai =0.10.0` 非破壊の確認

- 追加のみ: `fandhe_ai::nn::loss` に 5 名、`Var` に 5 メソッド。既存項目のシグネチャ・意味論・`FitConfig` は不変。依存・`unsafe`・新規 `Op`・新規カーネルなし。
- 名前衝突: 5 型名・5 メソッド名は 0.10.0 の facade 公開面に存在しない。`use fandhe_ai::nn::loss::*` の利用者のスコープに 5 名が増える点は、#2602 と同様に #2854 で `cargo test -p fandhe-ai --doc` により実測する（本記録では保証を断定しない）。

### 11.6 公開しないもの

クレートルート・`nn` 直下への再エクスポート、モジュール `elementwise_loss_ops`／`margin_focal_loss_ops` の再エクスポート、`Tensor`／`Tape` 上の同名メソッド、裸の自由関数、nn 構造体（`GaussianNllLoss` 等）、`BceWithLogitsLoss` の `pos_weight` 対応、`compat::Loss` への variant 追加、`reduction='none'`。いずれも保留またはスコープ外のまま。

### 11.7 #2854 の実装スケッチ（本記録では実施しない）

- `crates/autodiff/src/nn/loss.rs` の `pub use` 2 文、`crates/facade/src/nn/loss.rs` の `pub use` 5 名とモジュール doc（「19 名」の記述・doctest 追加）。
- `crates/autodiff/src/var.rs` に委譲 5 本（doc に `fandhe_ai::nn::loss::…` の経路を書く）。
- `crates/facade/src/lib.rs` の doc（オプション型 4 種の記述）と `ElementwiseLossOpsHoldDoctestGuard`・`MarginFocalLossOpsHoldDoctestGuard` の縮小（公開した型・メソッドのプローブを外し、モジュール名と `Tape`／`Tensor` 上の同名メソッドの拒否を残す）。
- `crates/facade/tests/api_surface.rs`: `NN_LOSS_NAMES` 19 → 24、到達テスト、`ELEMENTWISE_LOSS_*`／マージン側の同種定数、`PHASE4_VAR_EXPECTED_BODIES` に 5 件、宣言場所インベントリ。
- `crates/facade/tests/loss_var_delegates.rs` に 5 本（forward／backward が自由関数と bit 一致）。新規カーネルなしのため CUDA／Metal の新規 `#[ignore]`・実測申し送りは不要。

### 11.8 本 PR（#2853）で行わないこと

facade／autodiff のコード変更、ガードの新設・削除・反転、依存・tolerance・baseline・`docs/spec/` の変更、Issue 起票、ruleset・リポジトリ設定の変更。
