# `compile()` の `Optimizer` に Adadelta・Adamax・NAdam・RAdam を足す公開形の設計記録（0.14.0）

イシュー #3757（A3-g）。雛形 `docs/decision-record-template.md`（#3716）の 9 節に従う。ルートは #3711。承認は #3769 の論点 4、実装は #3839（Adadelta・Adamax）と #3840（NAdam・RAdam）で行う。

**承認根拠: なし（未承認・#3769 待ち）。本書は何も承認しない。** コード・公開面・tolerance・baseline・ガードレール閾値・依存・`docs/spec` は変えない（docs のみ）。

基準 sha は `02c0e5553`。本書の `path:line` はこの sha のもので、実装 PR の前に引き直す。

## §0 位置づけ・境界・推奨

- イシュー本文は「項目 24」と書くが、追跡表 `docs/compat-matrix-ari-tracker.md` では **行 #23（A3-g）** にあたる（1 ずれ。`docs/autodiff-param-groups-decision.md` §15 冒頭、`docs/compat-new-layers-io-export-decision.md` §0 と同じ注記）。追跡表の番号は書き換えない
- 経緯: `compat::Optimizer`（`crates/facade/src/compat/training.rs:238-240`、`#[non_exhaustive]`・`Debug, Clone, Copy, PartialEq`）は 7 種（Sgd・AdamW・Adam・RmsProp・Adagrad・Lamb・Lbfgs）。Adadelta・Adamax・NAdam・RAdam は内部クレートで実装済み（#2171）で、`fandhe_ai::optim` から再エクスポート済み（#2501。`crates/facade/src/optim.rs:515,520,538,548`）だが、`compile()`／`fit()` には入っておらず手動ループでしか使えない（`docs/autodiff-optimizer-adadelta-adamax-nadam-radam-decision.md` §9、`docs/autodiff-param-groups-decision.md` §12.6）。棚卸しは `docs/compat-unlisted-holds-inventory.md:53`（A3-g・保留）
- 境界（扱わない）: Rprop・ASGD・Adafactor・Lion の `Optimizer` 化、`DeviceParamStore` の常駐経路、`TrainStepOptimizer::set_lr` の公開、A3-a〜A3-d（#3752・#3754〜#3756）

### 調査で確定した事実

| 事実 | 根拠 |
|---|---|
| 4 種の `*Config` は facade から再エクスポート済み。新しい `pub use` は要らない | `optim.rs:515,520,538,548`。期待集合は `crates/facade/tests/api_surface.rs:372-381` |
| 4 種とも `new(config) -> Result<_, AutodiffError>`・`config()`・`set_lr`・`step(&[(&Tensor,&Tensor)])` を持つ | `crates/autodiff/src/nn/optim/{adadelta,adamax,nadam,radam}.rs` |
| 4 種とも `OptimizerStateDict` と `ParamGroupStep` を実装済み | 同上。#2298 |
| `*Config` は `Debug, Clone, Copy, PartialEq` で `#[non_exhaustive]` ではない。optimizer 本体は `Debug` を derive していない | 同ファイルの struct 定義 |
| `compile()` は `OptimizerState::new(optimizer)?` を `self.compiled` へ代入する前に評価する。検証失敗時に状態は変わらない | `training.rs:1894-1901` |
| `supports_lr_schedule` は否定形（`!matches!(RmsProp\|Adagrad\|Lamb)`）で、新 variant は自動で「対応」になる | `training.rs:1389-1394` |
| 組み合わせの拒否は Lbfgs 専用（AMP・`accumulate_steps>1`・フック・EMA・SWA・param groups・FunctionalModel）だけ。値型は一律に受理される | `training.rs:2001,2076,2780-2844`、`functional/train.rs:264` |
| model_io の manifest は kind の allowlist と kind ごとの固定キー集合で読む。kind 追加の先例（lbfgs・loss 9 種）は `format_version` 2 を据え置いた | `model_io/compiled.rs:152-160,179-245,364-449`、`docs/compat-model-io-decision.md:1492,1510,1530,1589` |
| variant 集合を固定する否定ガードはない。`compat_optimizer_enum_has_lbfgs_variant`（`api_surface.rs:19086`）は Lbfgs が 1 個あることだけを見るので、4 種を足しても green のまま | `api_surface.rs:19086-19118` |
| 集合と順序を固定する正ガードの先例: `compat_loss_enum_variants_are_exactly_approved_set`（`api_surface.rs:20709`。補助 `collect_top_level_enum_variant_idents`） | 同左 |

### 推奨: 実装する

1. 4 種は実装・再エクスポート・`set_lr`・`ParamGroupStep`・`OptimizerStateDict` がすべて済んでおり、残る差分は facade 内部の match の腕と `#[non_exhaustive]` enum への variant 追加だけ（追加のみ）
2. RmsProp・Adagrad・Lamb（#2170）と違って `set_lr` を持つので、`LrSchedule` を拒否する理由がない
3. Lbfgs と違い step 1 回あたり勾配評価 1 回の値型なので、既存の `step_dispatch` にそのまま乗る

| 案 | 内容 | 利点 | 欠点 |
|---|---|---|---|
| A（推奨） | 4 variant を追加する | `fit()` で使える。手動ループとの二重管理が消える | match の腕と model_io の kind が増える |
| B | 対象外（手動ループを正とする） | 変更なし | 4 種が `fit()`・callbacks・model_io から使えないまま残る |
| C | 一部だけ追加 | 変更量が小さい | 基準が曖昧で、残りの追跡が増える |

## §1 公開形

- 配置は `crates/facade/src/compat/training.rs` のまま。`Lbfgs(LbfgsConfig)` の**後ろ**に `Adadelta(AdadeltaConfig)`・`Adamax(AdamaxConfig)`・`NAdam(NAdamConfig)`・`RAdam(RAdamConfig)` を足す。名前は型名に合わせる（`AdamW(AdamWConfig)` 等と同じ規則）。新しい公開名（型・関数・`pub use`）は増えない。derive は保たれる（4 Config とも `Copy, PartialEq`）
- 実装 PR が足す網羅 match の腕（足し漏れはコンパイルエラーになる）:
  - `training.rs` の `OptimizerState`（:1186）: 手書き `Debug`（:1196。4 種は `Debug` を持たないので `debug_tuple("X").finish()`）・`new`（:1217。`*::new` へ委譲）・`lr`（:1233）・`set_lr`（:1259）・`step`（:1296）・`step_grouped`（:1346。`step_with_groups`）・`snapshot_of_compiled`（:1492）・`compiled_from_snapshot`（:1534）
  - `supports_lr_schedule`（:1389）: **網羅 match（肯定形）へ書き換える**ことを推奨する。否定形のままだと、将来 `set_lr` を持たない variant が足されたときに「対応」と誤判定される（fail-open）。内部の変更で公開面は変わらない（§12 論点 6）
  - `model_io/compiled.rs`: `optimizer_kind`（:152）・`config_fields`（:179）・`parse_non_lbfgs`（:364）・テスト用 `all_optimizers()`（:692）と use 文
  - 変えない: `functional/train.rs:261-270`・`functional_io.rs:352,1005`・`callbacks.rs` の拒否条件（Lbfgs と AMP だけを特別扱い）
- **ガード計画**: 反転する否定ガード（`*HoldDoctestGuard`・否定プローブ）は A3-g には**存在しない**。次を足す・直す
  - 新しい正ガード `compat_optimizer_enum_variants_are_exactly_approved_set`。`compat_loss_enum_variants_are_exactly_approved_set` と同形で、variant の集合と順序を完全一致で固定する。**#3839 で追加**し、期待列は `[Sgd, AdamW, Adam, RmsProp, Adagrad, Lamb, Lbfgs, Adadelta, Adamax]`（9 種）。**#3840 で** `NAdam, RAdam` を足して 11 種にする
  - `compat_optimizer_enum_has_lbfgs_variant`（:19086）とその自己テストは変えずに残す（表記揺れまで検出する Lbfgs 固有の検査）。新ガードとの役割の違いをコメントに書く
  - `fit_types_are_reachable_via_facade_only`（:4009）に、4 variant を facade だけの import で構築する行を足す
  - 否定プローブとして残す範囲: なし（承認外の variant は新ガードの完全一致で拒否される）
- ドリフトを直す doc（実装 PR）: `optim.rs:296`（「`compile()` にも未統合」）、`optim.rs:376`（「`Optimizer` enum にある 6 種」。#3840 後は 10 種）、`Optimizer` enum の doc、`OptimizerState` の doc（:1173-1185）、`compile()` の doc（:1887-1889）。doc の文言を逐語で照合するテストがないかを `rg` で確かめてから直す

## §2 rustdoc と doctest

- 各 variant に 1〜2 行の役割要約を書く（論文名・PyTorch の対応・`LrSchedule` に対応・param groups でも NAdam／RAdam の `decoupled_weight_decay` を group で上書きしないこと）
- doctest は `Optimizer` enum の doc に facade だけで動く CPU の例を 1 本置く（`Sequential::new()` → `compile(Optimizer::Adamax(AdamaxConfig::default()), Loss::Mse)` → `fit`）。GPU は使わないので `ignore`／`no_run` は不要
- 失敗系（検証違反）は `compile_fail` に頼らず、正のテスト（§4 の `compile_rejects_invalid_*`）で確かめる

## §3 エラー型

- 新しい variant は足さない。各 `*::new` の検証（`rho`・`beta*` は `[0,1)`、`eps > 0`、`lr`・`weight_decay` は有限かつ 0 以上、`momentum_decay` は 0 以上）が返す `AutodiffError::InvalidArgument` をそのまま `compile()` から返す
- 検証順: `OptimizerState::new` が `self.compiled` を書き換える前に `?` で失敗するので、既存の compile 状態は変わらない（fail-closed）。本番経路に `unwrap`／`expect` を入れない
- model_io: kind を allowlist で照合 → kind ごとの固定キー集合と完全一致で照合 → 正準 f32・bool をパース → 値の範囲は `*::new` に委ねる、という既存の順序を守る。未知の kind は `UnsupportedModel` で拒否する

## §4 テスト計画

実機テストはない（ホスト値型で GPU カーネルがないため。`docs/autodiff-optimizer-adadelta-adamax-nadam-radam-decision.md` §7）。`<cuda-node>`・`<metal-node>`・`<rocm-node>` はすべて該当なし。

| 受け入れ基準 | テスト（案）・置き場 | 区分 | 実機 |
|---|---|---|---|
| fit が手動ループと bit 完全一致（4 種） | `fit_matches_manual_loop_bit_exact_{adadelta,adamax,nadam,radam}`（`compat_sequential_fit_optimizers.rs`。手動ループは `compat_sequential_optim_ext.rs:67` の形を流用） | CI | 該当なし |
| fit 2 回 = epoch 倍の 1 回／再 compile で状態を捨てる | `fit_twice_equals_fit_once_with_double_epochs_*`・`recompile_discards_optimizer_state_*` | CI | 該当なし |
| 不正 config を compile が拒否し状態が変わらない | `compile_rejects_invalid_{adadelta,adamax,nadam,radam}_config` | CI | 該当なし |
| `LrSchedule` を受理し手動 `set_lr` ループと bit 一致 | `compat_sequential_callbacks.rs` に新規 | CI | 該当なし |
| `compile_with_param_groups` が手動 `step_with_groups` と bit 一致 | `compat_sequential_param_groups.rs` に新規 | CI | 該当なし |
| `compile_with_amp` を受理する | 既存 AMP テストに倣って新規 | CI | 該当なし |
| save_model／load_model で往復でき、改竄を拒否 | `compat_sequential_model_io_compiled.rs::optimizers()`（:54）に 4 種を足し、`every_optimizer_with_and_without_amp_resumes_bit_identically`・`tampered_kind_is_rejected`・`tampered_config_is_rejected` を通す。単体は `compiled.rs::all_optimizers` | CI | 該当なし |
| FunctionalModel の compile でも使える | `functional/fit_tests.rs` に 1 本 | CI | 該当なし |
| 公開面が承認形と一致 | 新正ガード・`fit_types_are_reachable_via_facade_only`・既存 Lbfgs ガード | CI | 該当なし |
| 既定経路が変わらない | 既存テストが無変更で通る | CI | 該当なし |

分担: #3839 が Adadelta・Adamax 分、#3840 が NAdam・RAdam 分。

## §5 数値契約

新しい GPU 演算・tolerance はなく、判定は bit 一致のみ。REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）、`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`、parity baseline、ガードレール閾値は変えない。PyTorch fixture（`crates/autodiff/tests/nn_optim_*.rs`）にも触れない。FMA 契約と f64 長軸縮約にも影響しない。baseline の追加・更新はない。

## §6 互換への影響

判定は**追加のみ**（手順の正は `docs/api-compat-0.14.0-decision.md` §8、表は同 §4 の行 23 と §1.4 の訂正〈`Optimizer` はすでに `#[non_exhaustive]`〉）。

- `Optimizer`: variant 追加。derive も保たれるので追加のみ
- `*Config`: 変更なし
- model_io の manifest: kind 追加で `format_version` 2 を据え置く（旧リーダーは未知の kind を fail-closed で拒否し、新リーダーは旧ファイルを読める）
- 移行ノート #3780 には「追加: `Optimizer` の 4 variant」とだけ書く想定（実装 PR が書く）

## §7 適用記録・索引・追跡表

- 公開状態: 未公開
- 適用記録の案（実装 PR が `docs/compat-api-scope.md` §5 と §1.2 の optimizer 行・`compile()` 行へ書く。承認 URL は #3769 の回答を待って埋める）: `**適用記録（イシュー #3839／#3840・親 #3757。承認: <URL>）**: compat::Optimizer へ Adadelta・Adamax・NAdam・RAdam を追加（既存の再エクスポート済み *Config を保持）。LrSchedule・param groups・AMP と併用可。model_io kind 4 種追加（format_version 2）。公開状態: 公開済み`
- 実装 PR があわせて追記するもの: `docs/compat-model-io-decision.md`（kind 追加）、`docs/autodiff-optimizer-adadelta-adamax-nadam-radam-decision.md` §9（「compile 統合」を解消済みへ。本文は書き換えず追記のみ）
- チェック: [x] `docs/README.md` 索引、[x] 追跡表の行 #23、[ ] `compat-api-scope`（実装 PR が行う）。最適化行の「未公開」「保留」が 0 になったことの `path:line` 根拠は #3840 が追跡表に書く

## §8 ベンチと性能への影響

既定経路（既存 7 variant）は match の腕が増えるだけで演算列は変わらない。前後比較は不要と判断する。計測していないので数値は書かない。

## §9 依存の追加の有無

依存の追加はない。新しい `unsafe` もない。許容 12 区分・禁止リストとの照合結果は変わらず、`docs/license-matrix.md` の更新も要らない。

## §10 組み合わせの対応表

| 組み合わせ | 4 種の扱い（推奨） | 根拠 |
|---|---|---|
| `Callback::LrSchedule` | 受理（`set_lr` へ委譲。moment や `step_count` はリセットしない） | `set_lr` がある |
| `compile_with_param_groups` | 受理（`step_with_groups`。NAdam／RAdam の `decoupled_weight_decay` は group で上書きしない） | #2298 |
| `compile_with_amp` | 受理（GradScaler の unscale → step という汎用経路） | 値型 |
| `accumulate_steps > 1`・`fit_with_train_step`・`Callback::Ema`・`Callback::Swa`・FunctionalModel | 受理（Lbfgs 専用の拒否は変えない） | 同上 |
| param groups と `LrSchedule` の同時指定 | 4 種固有の扱いは足さず A3-b（#3754）の決定に従う | `docs/autodiff-param-groups-decision.md` §15 |
| param groups と AMP の同時指定 | A3-a（#3752）の決定に従う | 同 §14 |
| save_model（groups が空のとき） | kind `adadelta`／`adamax`／`nadam`／`radam` で保存・復元。config キーは Adadelta が `[lr, rho, eps, weight_decay]`、Adamax が `ADAM_LIKE_KEYS`、NAdam が `+momentum_decay, decoupled_weight_decay`、RAdam が `+decoupled_weight_decay`（bool は `nesterov` と同じ描画） | 先例 lbfgs・loss |
| `DeviceParamStore` | 対象外（compile 経路と無関係） | 決定記録 §7 |

## §11 実装の分け方

- #3839: Adadelta・Adamax と新しい正ガード（9 種）
- #3840: NAdam・RAdam（ガードを 11 種へ更新）

## §12 所有者へ戻す論点（#3769 論点 4）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|---|---|---|---|
| 1. 採否 | 実装する／対象外 | 実装する | 実装へ進めない |
| 2. variant 名と位置 | 型名に合わせて Lbfgs の後ろ／別の名前・位置 | 前者 | 同上 |
| 3. `LrSchedule`・param groups・AMP など | 全部受理（§10）／一部を拒否 | 全部受理 | 同上 |
| 4. model_io | kind 4 種を追加し `format_version` 2 を据え置く／4 種の save を `InvalidArgument` で拒否 | 追加する | 同上 |
| 5. 分割 | #3839・#3840 の 2 本／1 本 | 2 本 | 同上 |
| 6. `supports_lr_schedule` の網羅 match 化 | 行う（内部の変更）／否定形のまま | 行う | 同上 |
| 7. Rprop・ASGD・Adafactor・Lion の compile 統合 | 対象外（起票しない）／別に起票 | 対象外（所有者が望むときだけ起票） | 起票しない |

## §13 セキュリティ・スコープ外・Closes／Refs

- A03: §3 の検証順（allowlist → 固定キー集合 → パース → `*::new`）を守り、未知の値は fail-closed で拒否する。A06: 依存の追加・更新なし。A08: tolerance・baseline・ガードレール閾値・自己修復の判定経路に触れない。秘密情報・ホスト名は書かない
- スコープ外は §0 の境界に書いたもの。無断で起票しない
- PR は `Closes #3757`（設計記録が完了条件）、`Refs #3769`・`Refs #3839`・`Refs #3840`
