# pos_weight 付き BCEWithLogits・HingeEmbedding・SoftMargin・GaussianNLL の CPU 実装記録（イシュー #2652）

- 対象: PyTorch の `F.binary_cross_entropy_with_logits(pos_weight=)`／`F.hinge_embedding_loss`／`F.soft_margin_loss`／`F.gaussian_nll_loss` 相当
- 親: #2651
- 実装: `crates/autodiff/src/elementwise_loss_ops.rs`（自由関数 4 本・オプション型 2 つ）・`crates/autodiff/src/eval/elementwise_loss.rs`（forward／VJP カーネル）
- 範囲: **内部クレート限定**。facade 公開面の拡張は本イシューの範囲外（承認依頼 #2677・承認後の #2678）

## 0. 結論

- 新規 `Op` 4 種（`BceWithLogitsPosWeightLoss`・`HingeEmbeddingLoss`・`SoftMarginLoss`・`GaussianNllLoss`）とホスト参照実装で実装した（`Op::L1Loss`・`Op::PoissonNllLoss` と同型）。`BackendOps` は拡張せず、CPU／CUDA／Metal のどの tape からも同じホスト経路で到達する。
- `pos_weight` なしの BCE は既存 `Var::bce_with_logits_loss` へ丸ごと委譲し、出荷済み経路と bit 一致する（`Op::BceLoss`・`backend-cpu/src/bce.rs` は不変）。
- PyTorch 2.14.0 の実行値 fixture（`cases` 74 件・`edge_cases` 11 件）と REQ-2 統一複合判定で突合し、全件一致した。tolerance・baseline は変更していない。
- facade 非公開を保留ガード（`ElementwiseLossOpsHoldDoctestGuard` と `api_surface.rs` の否定ガード）で機械固定した。
- CUDA／Metal 実機の `#[ignore]` テストは未実測（§10）。

## 1. 着手時の判定

事実のみを記す。

- 着手時点（`origin/main` = `e146e129`）で #2652 にコメントはなく、#2677 は open。公開面の拡張は未承認として扱う。
- 同名の `fn` 宣言（4 名）は `crates/*/src` に 0 件。既存の `Var::bce_with_logits_loss`・`nn::BceWithLogitsLoss`（`Copy`）には触れない。
- 並列イシュー #2653 が `loss_ops.rs`／`nn/loss.rs` を変更しうるため、接触面を減らす目的で新規モジュールに置いた。

## 2. 実装方式

| 項目 | 内容 |
|---|---|
| 配置 | 新規 `elementwise_loss_ops.rs`（公開入口）と `eval/elementwise_loss.rs`（数値カーネル）。`loss_ops.rs` には追記せず、非公開ヘルパー 3 本（`materialize_pair`・`materialize_triple`・`check_pm_one_labels`）を可視性のみ `pub(crate)` にして再利用 |
| Op | 4 variant。`pos_weight` は `input` と同 shape へ展開済みの非追跡 Tensor を payload に持つ。`y`（±1）も非追跡 |
| 融合・checkpoint・create_graph | いずれも非対応側（`is_checkpoint_eligible`・`supports_create_graph` で fail-closed） |

### 却下した案

| 案 | 却下理由 |
|---|---|
| 既存 Op の合成 | GaussianNLL の「clamp は値だけ変え勾配を遮らない」意味論と hinge の境界勾配を合成では正確に制御しにくく、`f64` アキュムレータ契約も直接満たせない |
| `loss_ops.rs` への追記 | 公開・承認済みの 7 関数の inventory に未承認分が混ざる。並列 #2653 と衝突する |
| nn 構造体の追加 | 構造体形は #2600 ツリー（`docs/facade-nn-loss-structs-exposure-decision.md` の 19 名集合）の決着後に扱う |
| `Var` メソッドの先行追加 | 公開面の拡張は承認依頼 #2677 の範囲 |

## 3. 数値契約

共通: 要素を先に `f64` へ昇格して要素式を評価し、index 順に `f64` で蓄積して最後に 1 回だけ `f32` へ downcast する。`Mean` は `numel` で除算、`numel == 0` は損失 `0.0`・勾配は空（既存 `mse_loss` 規約。PyTorch の `Mean` は NaN）。VJP の `scale` は `Mean` で `g/n`、`Sum` で `g`。

| 損失 | 要素式 | 勾配（`s` = scale） |
|---|---|---|
| BCE + pos_weight | `l = (1−y)·x + (1+(p−1)·y)·softplus(−x)` | `dx = s·[(1−y) − lw·σ(−x)]`・`dy = s·[−x + (p−1)·softplus(−x)]` |
| HingeEmbedding | `y=1`: `x`、`y=−1`: `max(0, margin−x)`（NaN 伝播） | `y=1`: `s`、`y=−1`: `margin−x > 0` のとき `−s` |
| SoftMargin | `max(z,0) + ln_1p(exp(−|z|))`（`z = −y·x`） | `dx = s·(−y)·σ(z)` |
| GaussianNLL | `c = max(var,eps)`・`0.5·(ln c + (x−t)²/c)`（`full` で `+0.5·ln 2π`） | `dx = s·d/c`・`dt = −dx`・`dvar = s·0.5·(1/c − d²/c²)` |

判定方式: fixture 突合は REQ-2 統一複合判定（`common::req2_close`）。バックエンド間は新規 Op が bit 一致、委譲経路は REQ-2 判定（`crates/facade/tests/elementwise_loss_ops_backend_parity.rs`）。

### 実測で確定した点（fixture が正）

- **hinge の境界勾配は 0**: `y = −1` かつ `x == margin` の勾配は PyTorch 2.14.0 で 0 だった。計画段階の仮定（`clamp_min` 契約で境界を通す。`margin_ranking_loss` はこちら）は誤りで、本実装は `margin − x > 0` のときのみ通す。
- **GaussianNLL は `var < eps` でも `dvar` を遮らない**: `var = 0`・`eps = 1e-6` で `dvar` は clamp 後の値で評価される（`−1.25e11` 等）。

## 4. 境界検査（REQ-8・A03）

検査順序: ①`check_same_tape` → ②shape（`require_same_shape`・`pos_weight` の `broadcast_to` 可否）→ ③確保前バイト数上限（`checked_bytes_for::<f32>`）→ ④スカラー引数（`margin`・`eps` の有限性）・非追跡テンソルの値（`y` の ±1、`pos_weight` の有限・非負）→ ⑤実体化 → ⑥実体化後の値検査（`var` の非負・NaN 拒否）→ ⑦forward → ⑧`push_eager`。エラー時に tape へ孤児ノードを残さない（単体テストで `tape.len()` 不変を固定）。`pos_weight` の展開量は `input` の numel 以下。本番経路に `unwrap()`／`expect()`／添字 panic はない。

## 5. PyTorch 2.14.0 との差分

| 項目 | PyTorch | 本実装 |
|---|---|---|
| `y`（Hinge／SoftMargin） | 任意値 | 厳密に ±1（それ以外は `InvalidArgument`） |
| `pos_weight` | 負値も受ける | 有限かつ非負のみ。`input` shape へ右寄せ broadcast でき結果が `input` shape と一致する形のみ |
| GaussianNLL の `var` | `[..., 1]` 形・スカラー等を受ける。負値のみ拒否 | `input` と同 shape 限定（呼び出し側が `reshape`／`broadcast_to` で揃える）。負値と NaN を拒否 |
| SoftMargin の大振幅 | f32 の `log1p(exp(−y·x))` で `−y·x > 約 88.7` は `inf`（勾配 NaN） | 安定形で有限値。fixture の `soft_margin_large`（`diverges` 付き）で明示 assert |
| 空テンソルの Mean | NaN | `0.0`（既存 `mse_loss` 規約） |

## 6. テスト構成

- `crates/autodiff/src/elementwise_loss_ops.rs` 単体テスト: 手計算値、中心差分と VJP の一致（全追跡入力）、空テンソル、エラー経路と `tape.len()` 不変、既定オプションの委譲（bit 一致・1 ノード）、上流勾配スケール、`var < eps` の `dvar`。
- `crates/autodiff/tests/elementwise_loss_parity.rs`: fixture 全件の forward と全勾配（REQ-2）、`edge_cases` の値クラス、契約整合（`pos_weight` なしは bit 一致、`pos_weight = ones` は既存経路と REQ-2 一致）。
- `crates/facade/tests/elementwise_loss_ops_backend_parity.rs`: CPU 対 NaiveOps（属性なし）、CUDA／Metal は `#[ignore]`。
- fixture の出自: `crates/autodiff/tests/fixtures/elementwise-loss-pytorch-reference/README.md`。

## 7. facade 公開形の推奨案（推奨案の記録であり承認記録ではない）

- 推奨は `Var` の 1 行委譲メソッド 4 本（`bce_with_logits_loss_with`・`hinge_embedding_loss`・`soft_margin_loss`・`gaussian_nll_loss`）。既存 7 損失（#2538〜#2540）と同形で、追加のみ・非破壊。
- 推奨しない: モジュール再エクスポート、`Tensor`／`Tape` への配置、`Sequential::add_*`（損失は層ではない）、`compat::Loss` への variant 追加（GaussianNLL は 3 入力で `compile()` の形に合わない）。
- オプション型 2 つ（`BceWithLogitsOptions`・`GaussianNllOptions`）と `Reduction` の名指しは、既知ギャップ（`docs/autodiff-loss-ops-decision.md` の #2538 追記）と同じ論点で、#2600 ツリーの記録側で扱う（オプション型の公開形は §14〈#2853〉）。
- 承認依頼は #2677、公開は承認後の #2678。本イシューでは `var.rs` にメソッドを足さない。

## 8. スコープ外

- facade 公開（`Var` 委譲 4 本、保留ガードの正ガード反転）
- nn 構造体（`HingeEmbeddingLoss` 等）、`BceWithLogitsLoss` の `pos_weight` 対応、`compat::Loss` への追加、`model_io`／ONNX 対応
- BCEWithLogits の `weight`（要素重み）、`reduction='none'`
- GaussianNLL の `var` broadcast 形、`y` が ±1 以外のラベル、負の `pos_weight`
- GPU 専用カーネルと CUDA／Metal 実機実測、`create_graph`（高階微分）、activation checkpoint、f64／f16／bf16 経路

## 9. 多層防御

| 層 | 内容 |
|---|---|
| 1 | `ElementwiseLossOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`。全 `pub mod` を glob import した上でローカルのモジュール・型・同名メソッドをプローブ。同名の公開が現れると名前解決の曖昧性（E0659）で失敗する） |
| 2 | glob 集合の一致（`elementwise_loss_ops_hold_doctest_globs_all_pub_modules`） |
| 3 | プローブ本文の固定文言一致（`..._probe_body_matches_fixed_contract`） |
| 4 | ソース走査（`facade_does_not_reexport_or_declare_elementwise_loss_ops` と自己テスト）と workspace の `fn` 宣言 inventory（`workspace_declares_elementwise_loss_ops_fn_names_only_in_allowed_locations`） |

検出範囲は列挙した名前・型に限り、マクロ生成や別名経由のメソッドまでは保証しない。

**反証確認**: `crates/facade/src/lib.rs` に `pub use fandhe_ai_autodiff::elementwise_loss_ops;` を仮に足し、doctest が `E0659`（`elementwise_loss_ops` is ambiguous）で失敗し、ソース走査テスト `facade_does_not_reexport_or_declare_elementwise_loss_ops` が失敗することを確認して元に戻した。

## 10. 実機申し送り

CUDA／Metal の `#[ignore]` テスト（計 8 件）は本実行環境から実機に到達できないため未実測。手順と記入欄は `docs/perf/logs/elementwise-loss-ops-2652/README.md`。新規 GPU カーネルはなく、ホスト経路の bit 一致確認である。

## 11. 出典

- `docs/autodiff-loss-ops-decision.md`（既存 7 損失の設計・数値契約）
- `docs/autodiff-softmin-threshold-ops-decision.md`（保留ガードの様式）
- `.claude/rules/coding-rust.md`（`f64` アキュムレータ契約・境界検査）
- PyTorch 2.14.0 の実行値（`crates/autodiff/tests/fixtures/elementwise-loss-pytorch-reference/`）

## 12. #2678 での扱い（保留を維持）

- ルート #2499 の 2026-10-07 ユーザー承認コメント（issuecomment-6033824965）は、行 19・20 のうち `hinge_embedding_loss`・`soft_margin_loss`・`multilabel_margin_loss` の 3 本だけを承認し、`Reduction` を `fandhe_ai::nn::loss::Reduction` の 1 経路で名指しできることを前提にしていた。#2678 の着手時点で #2602（`nn::loss` の公開）は未マージで、`crates/facade/src/nn/mod.rs` に `loss` モジュールがない。承認条件（記録に形が書かれていない点・前提が満たされない点は実装せず止める）に従い、本記録の対象（`Var::hinge_embedding_loss`・`soft_margin_loss` を含む）は #2678 では公開せず、保留ガードと `api_surface.rs` の否定ガードを無変更のまま維持した。
- `Reduction` の公開経路を本イシューで作らない（公開面の拡大にあたるため）。#2602 のマージ後に、3 本を `Var` の 1 行委譲で公開する残作業がある。オプション型を引数に取る損失 5 本とオプション型 5 つは引き続き保留（承認の対象外）。
- 依存・tolerance・baseline・`docs/spec` は変更していない。

## 13. #2677 での公開（`hinge_embedding_loss`・`soft_margin_loss`）

- #2602（PR #2835）で `fandhe_ai::nn::loss::Reduction` が公開され、§12 の保留条件が満たされた。ルート #2499 の 2026-10-07 ユーザー承認コメント
  （https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965）に従い、§7 の推奨形（`Var` の 1 行委譲メソッド）で次の 2 本を公開した（`Refs #2677`）。
  - `Var::hinge_embedding_loss(&self, y: &Tensor<f32>, margin: f32, reduction: Reduction) -> Result<Var<'t>, AutodiffError>`
  - `Var::soft_margin_loss(&self, y: &Tensor<f32>, reduction: Reduction) -> Result<Var<'t>, AutodiffError>`
- シグネチャは §7 の「既存 7 損失と同形の `Var` 1 行委譲」と自由関数の引数列から一意に定まる（レシーバ = `input`）。`Reduction` は `fandhe_ai::nn::loss::Reduction` の 1 経路のみ（root には出さない）。
- **保留のまま**: `bce_with_logits_loss_with`・`gaussian_nll_loss` とオプション型 `BceWithLogitsOptions`・`GaussianNllOptions`（承認の対象外）。モジュール `elementwise_loss_ops` は再エクスポートしない。
- §9 の層 1〜2（`ElementwiseLossOpsHoldDoctestGuard` と glob 集合一致）は、受け手 `Var` のプローブから公開した 2 本を外した形へ縮小した（`Tape`／`Tensor<f32>` への配置は専用トレイトで拒否を維持）。層 3 は固定文言を更新した。
  層 4 のソース走査は facade src に対して 4 名のまま、宣言場所インベントリは `elementwise_loss_ops.rs` の自由関数 4 件に `var.rs` の 2 件を加えた。
- テスト: `crates/facade/tests/loss_var_delegates.rs`（forward／backward が自由関数と bit 一致）。実機（CUDA／Metal）は §10 のまま未実測（新規カーネルなし）。

## 14. 保留していた損失とオプション型の公開形（#2853）

- §13 で保留していた `bce_with_logits_loss_with`・`gaussian_nll_loss` とオプション型 `BceWithLogitsOptions`・`GaussianNllOptions` の公開形を、ルート #2499 の 2026-10-08 コメント（https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061）の承認範囲（オプション型は `fandhe_ai::nn::loss` へ再エクスポートし、損失は `Var` の 1 行委譲にする）に沿って決めた。
  パス・完全なシグネチャ・構築方法の確認は `docs/facade-nn-loss-structs-exposure-decision.md` §11 を正とし、本節では書き写さない。公開（コード・ガード）は #2854 で行い、**本節の時点では保留のまま**。
- モジュール `elementwise_loss_ops` は引き続き再エクスポートしない。`Reduction` は `fandhe_ai::nn::loss::Reduction` の 1 経路のまま。
