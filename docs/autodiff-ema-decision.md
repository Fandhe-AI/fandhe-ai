# EMA（指数移動平均）設計判断記録

イシュー #2179（親 #2131「PyTorch／TF 置き換えの API 網羅」）。
`docs/autodiff-lbfgs-decision.md`（#2197／#2198）と同型の記録。本 doc は
`crates/autodiff/src/nn/ema.rs` のモジュール doc・実装を正として要約
したものであり、両者が食い違う場合は実装側を正とする。

## §0 結論

PyTorch `torch.optim.swa_utils.AveragedModel`（`avg_fn` に EMA 式を渡す
用法）／Keras 3 `EMAOverlay`（`ema_momentum`）相当を **内部クレート
限定**の型として実装した: `fandhe_ai_autodiff::nn::
ExponentialMovingAverage`（`crates/autodiff/src/nn/ema.rs`。
`nn/mod.rs` で再エクスポート）。

facade（`fandhe_ai::optim` 等への再エクスポート・`compat::FitConfig`
の `use_ema`／`ema_decay` 相当のオプション追加・`fit()` の各 step 後
自動更新）は**別途ユーザー承認**が必要な facade 公開面拡張（親 #2131
の「facade 公開面の拡張は設計判断記録 → 承認 → 実装の 2 段」規則）で
あり、本イシュー時点では未承認のため保留する（多層ガードで固定。
§4・§7）。#2559 の着手時判定と承認依頼は §10。**#2560 で §10 の推奨案を
facade へ公開した（§13 で解消）**。ガードの仕上げと適用記録は §14（#2561）。

## §1 使い方（内部クレート）

- 位置対応: `ExponentialMovingAverage::new(decay, &[&p0, &p1, ..])` →
  `update(&[&p0', &p1', ..])`。登録名は `"0"`, `"1"`, … の連番。
- 名前付き: `from_named(decay, vec![(name, &tensor), ..])` →
  `update_named(vec![(name, &tensor), ..])`。名前集合は構築時と完全
  一致していなければならない。
- `Module` trait 経由: `from_module(decay, &model)` →
  `update_from_module(&model)`（`model: &dyn
  fandhe_ai_autodiff::nn::Module`）。
- 推論・評価時の一時差し替え: `apply(&mut model)` が現在の
  `state_dict()` を退避しつつ shadow を書き込み、`restore(&mut model,
  backup)` で戻す（[`Module::load_state_dict`] への委譲。アトミック性
  契約は §3 参照）。
- facade `compat::Sequential` は内部 `Module` trait を実装しないため
  `from_module`／`apply`／`restore` は使えない。代わりに公開済みの
  `named_parameters()`／`state_dict()`／`load_state_dict()` と
  `from_named`／`update_named`／`shadow_state_dict` を結線する
  （`crates/facade/tests/compat_sequential_ema_manual.rs` 参照）。

## §2 数値契約

更新式: `shadow[i] = decay * shadow[i] + (1 - decay) * param[i]`
（`f32::mul_add(decay, shadow[i], one_minus_decay * param[i])`。
`nn/optim/rmsprop.rs::RmsProp::step` と同じ house style。
`.claude/rules/coding-rust.md` の CPU 参照実装 FMA 契約〈`f32::
mul_add`〉に整合）。Keras 3 `ema_momentum * average + (1 -
ema_momentum) * var` と同型だが、**PyTorch `AveragedModel`（既定
`avg_fn` は `lerp` 形 `averaged_param + (1 - decay) * (param -
averaged_param)`）とは丸めが異なるため bit 一致は主張しない**（数式
としては等価だが浮動小数演算順序が異なる）。

非有限値（`NaN`／`inf`）は特別扱いせず伝播させる。初期化は構築時
パラメータの `clone`（Keras／TF と同じ）。`num_updates` は呼び出し
回数のカウンタのみで、TensorFlow の `num_updates` による decay
ウォームアップ（`min(decay, (1 + n) / (10 + n))`）は本イシューでは
採用しない（§9「将来拡張候補」）。

## §3 検証・原子性

- `decay` は有限かつ `[0.0, 1.0]` を `new`／`from_named` が検証する
  （`AutodiffError::InvalidArgument`）。
- `update`／`update_named` は two-pass 検証（名前集合の完全一致・
  shape 一致を全件検査してから代入する）で、1 件でも失敗したら
  shadow を一切変更しない。
- `apply`／`restore` は `Module::load_state_dict` へ委譲するため、
  その「検証 two-pass ＋ベストエフォート・ロールバック」契約を
  そのまま継承する（`crates/autodiff/src/nn/module.rs::
  Module::load_state_dict` doc「アトミック性」節参照）。

## §4 承認事項（未承認）

> §13（#2560）で §10.2・§10.3 の推奨案が承認・実装された。以下は着手時点の判定記録。

**実装記録（#2560・#2561）**: 公開した名前は §13.2 の 3 件（`optim::ExponentialMovingAverage`・
`compat::EmaCallback`・`Callback::Ema`）。下記の保留対象 1 は承認形で公開、2（`FitConfig`／
`Sequential` への `use_ema`／`ema_decay` 追加）は §10.2 (d) で不採用のまま禁止経路として固定、
3（`fit` への結線）は `Callback::Ema` で実現した。現行のガード名と対応は §14.2。

以下は facade（crates.io 公開クレート）の新規公開面拡張に該当し、
ユーザー承認を要する。承認事項の分類根拠:

- 兄弟イシュー #2177（`FitConfig` へ 3 フィールド追加。PR #2306 で
  保留固定・出荷中）・#2180（`FitConfig::accumulate_steps` 追加）は
  いずれも同種変更（`FitConfig` へのオプション追加）を承認事項として
  列挙している。
- 親 #2131 は「facade 公開面拡張は設計判断記録 → 承認 → 実装の 2 段」
  を規則化している。
- #2170 は「`#[non_exhaustive]` enum への variant 追加は承認事項に
  該当しない」という個別の論証を持つ例外だが、本件（`FitConfig` への
  opt-in フラグ追加）には同じ論証が成立しない。

保留対象:

1. `fandhe_ai::optim::ExponentialMovingAverage`（または相当の型）の
   facade 再エクスポート。
2. `compat::FitConfig` への `use_ema`／`ema_decay` 相当のオプション
   追加（`fit(use_ema=true)` 相当）。
3. `Sequential::fit`／`fit_with_callbacks` の各 step 後の自動 EMA
   更新・evaluate 時の shadow 差し替え統合。

保留の機械的固定は `crates/facade/src/lib.rs::EmaHoldDoctestGuard`
（型名 glob 衝突プローブ＋ `FitConfig`／`Sequential` への inherent
メソッド衝突プローブの 2 系統）と、対応するソース走査ガード
（`crates/facade/tests/api_surface.rs::
ema_hold_doctest_globs_all_pub_modules`・
`ema_hold_doctest_probe_body_matches_fixed_contract`・
`facade_does_not_reexport_or_declare_ema_items`）の多層防御で行う。

（上記は着手時点の記録。ガードの実数は自己テストを含め 4 件だった。現行の名前は §14.2 を参照。）

## §5 承認後の facade 仕様案

> **実装記録（#2560・#2561）**: 案 A（`FitConfig::use_ema`）は不採用。案 B を整形した
> `Callback::Ema(EmaCallback)` を採用した（§10.2 (d)）。更新位置・評価時の shadow 差し替え・
> fit 終了時に重みを自動上書きしない点は §13、ガードと適用記録は §14。

`compat::FitConfig` は `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`
のため、`ema_decay: f32` フィールドをそのまま追加すると `Eq` が
（`f32` が `Eq` を実装しないため）外れ semver 破壊になる。選択肢:

- **案 A（推奨）**: `FitConfig::use_ema(bool)` のみを追加し、decay
  値は別 API（例: `Sequential::compile` の `Optimizer` 選択とは独立の
  `EmaDecay` newtype。`f32::to_bits` 比較で `Eq` を実装した非公開
  相当のラッパー）で保持する。`FitConfig` 自体の `Eq` 要件を壊さない。
- **案 B**: `fit_with_callbacks` の `Callback` variant として
  `Callback::Ema { decay: f32 }` を追加し、`FitConfig` 自体は変更
  しない（`FitConfig` の semver 制約を回避できる代わりに、`fit()`
  単純系からは EMA を使えなくなる）。

各 step 後の更新位置は `apply_parameters` 直後（本 doc §1 の手動結線と
同じ）。validation／evaluate 時は shadow へ一時差し替え →
評価 → 復帰の順（`apply`／`restore` と同じ契約）。
`restore_best_weights`（既存コールバック）との順序は「EMA 適用 →
評価 → best 判定 → 復帰」を想定（best 判定は EMA 適用後の重みで
行う）。`DeviceParamStore`（デバイス常駐経路）とは非互換
（ホスト側 shadow がデバイス常駐パラメータの更新を追跡しないため
stale 化する）。

## §6 検証結果（MNIST 規模手動ループ・定性確認）

`crates/facade/tests/compat_sequential_ema_manual.rs`（784 次元入力・
隠れ層 64・10 クラス分類・クラス依存の中心ベクトル＋ノイズ 0.3 の
合成データ、学習 256 サンプル・評価 128 サンプル、AdamW `lr=5e-3`・
12 epoch 全バッチ学習、`decay=0.9`）の実測:

```
raw_acc=1.0000 ema_acc=1.0000 decay=0.9 epochs=12 n_train=256 n_eval=128
```

学習が収束しやすい合成データのため両者とも 100% に達したが、AC R5
（定性確認）が求める「チャンスレベル（1/10）を明確に上回ること」・
「EMA 評価が学習状態を変化させない（退避／復帰の bit 完全一致）」は
いずれも満たしている。より緩やかな収束・ノイズの大きいデータでは
EMA が raw 重みより安定した accuracy を示す既知の一般的傾向（PyTorch
／TF の実績）はあるが、本イシューは定性確認の受け入れ基準のため
厳密な比較アサートは行っていない（flaky 化回避）。

## §7 実機・CUDA/Metal 固有経路

EMA はホスト `Tensor<f32>` に対する縮約を伴わない要素ごとの
`f32::mul_add` のみで実装しており、新規 `Op`／`BackendOps`／カーネル
は追加していない。CUDA／Metal 固有の数値経路を持たないため実機
parity テストは不要。デバイス常駐経路（`DeviceParamStore`）への結線
は §5 のとおり対象外。

## §8 スコープ

- 対象は `Module::named_parameters()`（学習可能パラメータ）のみ。
  `BatchNorm1d`／`BatchNorm2d` の `running_mean`／`running_var` は
  `RefCell` 越しの buffer であり `named_parameters()` に含まれない
  （`crates/autodiff/src/nn/batch_norm.rs` で確認。PyTorch
  `AveragedModel(use_buffers=False)` 既定・Keras（trainable
  variables のみ）と同じ）。
- SWA（Stochastic Weight Averaging）はスコープ外（等重み平均 `AveragedModel` と `SwaLr` は別型・別記録として #2658 で実装した。`docs/autodiff-swa-decision.md` 参照）。

## §9 将来拡張候補

- TensorFlow 方式の `num_updates` による decay ウォームアップ
  （`min(decay, (1 + n) / (10 + n))`）。
- `use_buffers=True` 相当（BatchNorm running stats も EMA 対象へ含める
  オプション）。

## §10 #2559 facade 公開形の着手時判定と承認依頼

イシュー #2559（親 #2558）。本節は docs のみの追記であり、facade コード・
保留ガード（`EmaHoldDoctestGuard`・`api_surface.rs` の EMA 否定ガード）・
`docs/compat-api-scope.md` は変更しない。**以下の推奨案は未承認**で、承認前は
兄弟 #2560（facade 公開の実装）・#2561（保留ガードの反転）に着手しない。

> §13（#2560）で承認の事実を記録し、本節の推奨案を実装した。

### 10.1 着手時判定

判定根拠は §4・§5 の本文（判定に使った `origin/main` は `4fa1b3e9`）。
そのまま実装できる確定した推奨形は無いと判断した。

1. §5 で唯一「推奨」の案 A（`FitConfig::use_ema(bool)`）は、値の保持に
   `FitConfig` へのフィールド追加を要する。#2558 の契約は `FitConfig` の
   `Copy + Eq` 固定のためフィールド追加を行わず非破壊な代替経路で結線すると
   定めており、案 A は採れない。decay の保持先（`EmaDecay` newtype）も配置未定。
   （実測: v0.10.0 の `FitConfig` は `#[derive(Debug, Clone, Copy, PartialEq,
   Eq)]`。契約が明示的に禁じるため採用しない）
2. 案 B（`Callback::Ema { decay: f32 }`）は「推奨」でなく、仕様が不足する。
   既存 variant（`EarlyStopping`・`ModelCheckpoint`・`LrSchedule`）は構築時検証
   型を包む tuple variant で形が揃わない。fit 後の shadow 取得経路、および
   `DeviceParamStore`・L-BFGS・`compile_with_amp`・`accumulate_steps > 1`・
   カスタム train_step フック・`ModelCheckpoint`／`EarlyStopping` との組合せが
   未記載。
3. §4 項目 1 は「（または相当の型）」で型の形が未確定。内部型の素の
   `pub use` は `from_module`／`update_from_module`／`apply`／`restore` の
   シグネチャに内部 `nn::Module` を露出し、facade 独自の `fandhe_ai::nn::Module`
   を持つ方針（`docs/facade-nn-module-exposure-decision.md` §1.3・REQ-12）に反する。

結論: 停止条項に従い、推奨案を記録して承認を依頼する。

### 10.2 推奨案（未承認）

- **(a) 配置**: `fandhe_ai::optim::ExponentialMovingAverage`。crate ルート・
  `nn` 配下には置かない。
- **(b) 型の形**: 内部型を 1 フィールドで保持する facade 独自の薄いラッパー。
  メソッドは内部の同名メソッドへの委譲で、数値契約（§2）・原子性（§3）は不変。
  `new(decay, &[&Tensor<f32>])`・`from_named`・`decay`・`num_updates`・`update`・
  `update_named`・`shadow`・`shadow_parameters`・`shadow_state_dict` は内部と同一
  シグネチャ。戻り値は個別に次のとおり（一律に `Result` ではない）:
  `new`／`from_named`／`update`／`update_named` は
  `Result<_, AutodiffError>`、`decay` は `f32`、`num_updates` は `u64`、
  `shadow` は `Option<&Tensor<f32>>`、`shadow_parameters` は
  `Vec<&Tensor<f32>>`、`shadow_state_dict` は
  `HashMap<String, Tensor<f32>>`。`from_module`／`update_from_module`／
  `apply`／`restore` は facade の `&dyn nn::Module` を受け、facade `Module` の
  `named_parameters`／`state_dict`／`load_state_dict` 経由で委譲する（初回公開面に
  含めるかは承認事項。代替: 初回は位置対応・名前付き API のみ）。
  `compat::Sequential` は facade `nn::Module` 非実装のため、既存公開 API と
  `from_named`／`update_named`／`shadow_state_dict` を結線する
  （`compat_sequential_ema_manual.rs` が先例）。不採用: 素の
  `pub use fandhe_ai_autodiff::nn::ExponentialMovingAverage;`（内部 trait 露出）。
- **(c) エラー型**: 既存 `AutodiffError::InvalidArgument` のみ。新 variant・新エラー
  型は作らない。検証は §3 と同一。
- **(d) fit への結線**: `FitConfig` は変更しない。案 B を整形した
  `Callback::Ema(EmaCallback)`（既存と同じ tuple variant）。
  `EmaCallback::new(decay: f32) -> Result<Self, AutodiffError>` で構築時に検証し、
  `Debug` を実装する。
- **(e) 更新位置**: §5 を踏襲（`apply_parameters` 直後に `update_named`、
  `accumulate_steps > 1` は実際に step した時のみ更新。評価は shadow 差し替え →
  評価 → best 判定 → 復帰。`ModelCheckpoint`／`EarlyStopping` は EMA 適用後の
  重みで判定）。
- **(f) 終了時挙動**: fit 終了時にモデル重みを自動上書きしない（既存意味論を
  変えない opt-in）。`EmaCallback` に `shadow_state_dict()` 等の accessor を設け、
  呼び出し元が `load_state_dict` で適用する。代替: 終了時に shadow で上書き
  （Keras `ema_overwrite_frequency=None` 相当）。
- **(g) fail-closed**: `fit_with_callbacks` は次を `InvalidArgument` で拒否する:
  `DeviceParamStore`／resident 経路（shadow が stale 化）・L-BFGS・カスタム
  train_step フックとの併用。`compile_with_amp` の skip step は「拒否」か
  「skip に追従して更新しない」かを論点とする。

### 10.3 記録に無かった追加論点（推奨はいずれも fail-closed）

- `Callback::Ema` の複数指定は拒否する。
- 名前集合の不一致は既存の two-pass 検証で拒否する。
- 非有限値は §2 のとおり伝播させる。
- `num_updates` ウォームアップ（§9）・BatchNorm buffer（§8）は対象外のまま。

### 10.4 `fandhe-ai =0.10.0` 公開 API の非破壊確認

| 論点 | 確認結果 |
|------|----------|
| (a)(b) | v0.10.0 の facade に EMA の実項目は無い（`git grep v0.10.0 -- crates/facade/src` の一致は保留ガードの doc コメントのみ）。追加のみで既存の名前・シグネチャは不変 |
| (c) | エラー型は不変 |
| (d) | v0.10.0 の `Callback` は `#[non_exhaustive]`＋`#[derive(Debug)]`（#2170 と同じ論証で variant 追加は非破壊）。payload に必要なのは `Debug` のみ |
| `FitConfig` | フィールド・derive（`Copy`・`Eq`）とも不変 |
| `compat::Sequential` | inherent メソッドを追加しない（`EmaHoldDoctestGuard` の衝突プローブの守る対象を維持） |
| 既存 API の意味論 | 不変（opt-in の callback のみ） |

### 10.5 承認を依頼する事項

1. (a) の配置。
2. (b) ラッパーか素の再エクスポートか、`Module` 系 4 メソッドを初回に含めるか。
3. (c) エラー型。
4. (d) `Callback::Ema(EmaCallback)` 形を採るか、fit への結線を後回しにするか。
5. (f) 終了時に上書きするか、accessor にするか。
6. (g)／10.3 の fail-closed 一覧（AMP skip の扱いを含む）。

承認された形だけが #2560／#2561 の実装範囲になる。承認が得られるまで両イシューは
blocked とする。

### 10.6 本節で行わないこと

`crates/facade/**` の変更、保留ガードの反転・`docs/compat-api-scope.md` §5 の更新
（#2561）、依存追加・`unsafe`・tolerance の変更。

## §11 #2560 着手時判定（§10 未承認のため停止）

イシュー #2560（親 #2558）。本節は docs のみの追記であり、facade コード・
保留ガード（`EmaHoldDoctestGuard`・`api_surface.rs` の EMA 否定ガード）・
`docs/compat-api-scope.md` は変更しない。

### 11.1 判定

判定に使った `origin/main` は `2ea128ba`（2026-10-05 確認）。

1. §4・§5 に、そのまま実装できる確定した推奨形は無い（根拠は §10.1。再掲しない）。
2. §10.2 の推奨案は §10 本文のとおり**未承認**である。
3. #2560・#2559・#2558・#2499 に承認を示すコメントは無い（コメント 0 件）。
4. ルート #2499 の一括承認は「記録が公開形を決めていない項目」に及ばず、
   §10.2 の推奨案を承認する記述も無い。

### 11.2 結論

停止条項に従い、facade 公開（`fandhe_ai::optim::ExponentialMovingAverage` の新設・
`Callback::Ema` の追加・fit への結線）は**実装せず停止した**。残る判断事項は
§10.5 の 1〜6（再掲しない）。#2560 の実装本体と #2561（保留ガードの反転・
`compat-api-scope.md` §5 の更新）は引き続き blocked とする。承認後に #2560 を
reopen するか新しい実装イシューを起票するかはユーザーが判断する。

### 11.3 本イシューで行わないこと

`crates/facade/**` の変更、保留ガードの撤去・反転、`compat-api-scope.md` §5 への
適用記録、追跡 Issue の起票（ユーザー承認が必要）、依存追加・`unsafe`・tolerance の変更。

## §12 #2561 着手時判定（§10 未承認・#2560 未実装のため停止）

イシュー #2561（親 #2558）。本節は docs のみの追記であり、停止の記録である。
承認を取得したことを意味しない。facade コード・保留ガード・
`docs/compat-api-scope.md` は変更しない。

### 12.1 判定

判定に使った `origin/main` は `6c966886`（2026-10-05 確認）。

1. #2560 は closed だが、facade 公開は実装されていない（§11 の停止 PR #2746 でクローズ）。
   `crates/facade/src` 内の `ExponentialMovingAverage`・`EmaCallback`・`Callback::Ema` の
   一致は `EmaHoldDoctestGuard` の doc コメントのみで、公開物は存在しない。
2. 正ガードは「承認・公開済みの形だけを許す」検査であり、公開物が無い現状では反転先が無い。
3. §10.2 の推奨案は §10 本文のとおり**未承認**である。#2561・#2560・#2559・#2558・#2499 に
   コメントは 0 件で、承認を示す記述は無い。#2499 の一括承認は §10.2 に及ばない（§11.1 と同じ判断）。

### 12.2 結論

停止条項に従い、`EmaHoldDoctestGuard` と `api_surface.rs` の次の 4 テストは
撤去も反転もせず現状維持する。

- `ema_hold_doctest_globs_all_pub_modules`
- `ema_hold_doctest_probe_body_matches_fixed_contract`
- `facade_does_not_reexport_or_declare_ema_items`
- `facade_does_not_reexport_or_declare_ema_items_detects_each_category`

解除の順序は、§10.5 の 1〜6 のユーザー承認 → #2560 の reopen または新しい実装イシューでの
facade 公開 → 保留ガードの反転（#2561 の受入条件）。承認だけでは解除されない。
受入条件 3 点（ガード反転・§4／§5 と `compat-api-scope.md` §5 の記録・facade 経由の利用例テスト）は
いずれも未達である。

なお §4 と `compat-api-scope.md` は「3 テスト」と書くが、実数は上記 4 件
（`…_detects_each_category` を含む）である。文言の修正は承認後の反転時に行う。

### 12.3 本イシューで行わないこと

`crates/facade/**` の変更、保留ガードの撤去・反転、`compat-api-scope.md` §5 への適用記録、
facade 経由の利用例テストの追加（対象 API が存在しないため）、追跡 Issue の起票・
#2560 の reopen（ユーザー承認が必要）、依存追加・`unsafe`・tolerance・baseline の変更。
承認事項の中身は §10.5 を参照する。

## OWASP Top 10 観点

- **A03 インジェクション／入力検証**: `decay`（有限・`[0, 1]`）・
  要素数・名前集合・shape を代入前に全件検証し型付きエラーで拒否。
  外部フォーマットのパース・シェル呼び出し・ファイル I/O は追加
  していない。
- **A04 安全でない設計**: 検証 two-pass で部分更新を残さない。
  `apply`／`restore` は既存 `load_state_dict` の原子性契約
  （ロールバック・失敗時の fail-closed エラー）を再利用し独自の
  書き戻し経路を作らない。
- **A05 設定不備**: facade 公開は §13 の承認形（`optim::ExponentialMovingAverage`・
  `compat::EmaCallback`・`Callback::Ema`）に限る。内部 `nn::Module` は公開シグネチャへ
  出さず、承認形以外の経路は `api_surface.rs` の正ガード（§14.2）と inherent メソッド衝突プローブで遮断する。
- **A06 脆弱なコンポーネント**: 依存追加なし（`Cargo.toml`／
  `Cargo.lock` 不変）。
- **A08 データ整合性**: 自己修復ループの判定経路・ガードレール
  閾値・tolerance／baseline に触れない。非有限値は隠蔽せず伝播させる。
- **資源枯渇（DoS 観点）**: shadow は params と同サイズの clone 1 組、
  `apply` は退避用 state_dict 1 組でピークメモリ約 2 倍
  （`load_state_dict` の既存コストと同等）。
- `unsafe` 追加なし・本番経路で `unwrap`/`expect` 不使用・秘密情報の
  扱いなし。

## §13 #2560 facade 公開の実装

イシュー #2560（親 #2558）。§10.2 の公開形と §10.3 の fail-closed 追加論点を実装した。

### 13.1 承認の根拠

- 根拠: ルート #2499 のコメント `issuecomment-6033824965`（2026-10-07、アカウント
  `aLiz-Nancy`。#2558 について §10 の推奨案を承認し、記録に形が無い点は実装せず止める旨）。
  #2558／#2560 の「Claude による承認の記録」コメントは補足であり承認の根拠ではない。
- 承認が及ぶ範囲は §10.2・§10.3 に書かれた形。記録に推奨が無い点は以下 13.4 のとおり実装せず
  fail-closed で止めた。

### 13.2 公開した名前

| 公開パス | 実体 |
|---|---|
| `fandhe_ai::optim::ExponentialMovingAverage` | `crates/facade/src/optim_ema.rs`（内部型を 1 フィールドで持つ薄いラッパー。`optim.rs` が `pub use crate::optim_ema::ExponentialMovingAverage;` の 1 文で公開。`mod optim_ema;` は private） |
| `fandhe_ai::compat::EmaCallback` | `crates/facade/src/compat/callbacks.rs`（`new(decay)`・`decay`・`num_updates`・`shadow_state_dict`） |
| `fandhe_ai::compat::Callback::Ema(EmaCallback)` | `#[non_exhaustive]` enum の末尾 variant。`FitConfig`・`Sequential` には何も足さない |

`from_module`／`update_from_module`／`apply`／`restore` は facade の `nn::Module` を受け、
`named_parameters`／`state_dict`／`load_state_dict` 経由で委譲する。新しいエラー型・variant は無い
（名前集合・個数の不一致と構築時の `decay` 検証は `InvalidArgument`。shape 不一致は既存の
`AutodiffError::Shape`）。

### 13.3 記録から導出した解釈点（新しい承認ではない）

1. `EmaCallback` は既存 payload 型と同じ `compat/callbacks.rs` に置き `compat` から再エクスポートする。
2. shadow は fit 終了後も保持し、次の fit で継続する（`ModelCheckpoint` と同じ継続型）。初回のみ
   最初の fit 開始時に `named_parameters()` から初期化する。別構成のモデルは既存検証で拒否する。
   リセット API は無い。
3. accessor は `decay`・`num_updates`・`shadow_state_dict`（未初期化の間は `None`）のみ。
4. validation が無い fit でも、epoch 末の callbacks（`ModelCheckpoint`／`EarlyStopping` の snapshot）
   は EMA 重みの下で実行する。したがって `restore_best_weights` が fit 終了時に書き戻すのは
   EMA 重みの snapshot になる。評価の差し替えは成功・`Err`・打ち切りのどの経路でも生の重みへ復帰する。
5. `DeviceParamStore`／常駐経路の検出機構は設けない。`fit` 経路は構造上常駐経路へ到達しない。
   手動で併用すると shadow が stale 化する（doc に明記）。
6. 記録に無い拒否（param groups・`LrSchedule`・`accumulate_steps > 1` 等との併用拒否）は追加しない。
   `accumulate_steps > 1` は実際に step した時（端数 flush を含む）だけ更新する。

### 13.4 未決のまま拒否した点（追加承認が必要）

§10.2 (g) の「`compile_with_amp` の skip step を拒否するか追従するか」は論点提示のみで推奨が無い。
実装は `Callback::Ema` と `compile_with_amp` の併用を fit 開始前に `InvalidArgument` で拒否する。
追従（skip step では更新しない等）へ緩めるには別途承認を要する。

同様に、`Callback::Ema` と `Monitor::Loss`（訓練損失監視）の `ModelCheckpoint`／`EarlyStopping` の
併用も記録に形が無い。`Monitor::Loss` は EMA 差し替え前の生の重みで計算した損失を指標にする一方、
snapshot は EMA 重みを保存するため、指標と保存重みが食い違う。承認された「記録に形が無い点は実装せず
止める」方針に従い fit 開始前に `InvalidArgument` で拒否した（`Monitor::ValLoss` 等は併用可）。
緩める（生の重みの損失でベスト判定して EMA 重みを保存する等）には別途承認を要する。

継続する `EmaCallback`（初期化済み）を別構成のモデルへ使い回した場合は、最初の step で
`update_named` が不一致を検出して拒否済みモデルに重み・optimizer 状態の変更が残らないよう、
fit 開始時に更新を伴わない名前集合・shape の照合で拒否する（名前集合・個数は `InvalidArgument`、
shape は `Shape`）。

### 13.5 ガードの変更と #2561 への残作業

- `EmaHoldDoctestGuard` は型名 glob 衝突プローブだけを削除し、`FitConfig`／`Sequential` への
  `use_ema`／`ema_decay` inherent メソッド衝突プローブを残した。プローブブロック数の下限
  （`MIN_KNOWN_PROBE_BLOCKS` = 16）は実測でも下回らないため変更していない。
- `api_surface.rs` は型名の否定ガードを正ガード `facade_exposes_ema_only_in_approved_shape`
  （承認形 = `optim.rs` の 1 行＋`optim_ema.rs` の宣言 1 件。内部型の素の再エクスポート・別名・
  重複・`use_ema`／`ema_decay` の `fn` 宣言は拒否）へ置き換え、`optim` の期待集合と
  `Callback` variant 集合（7 個）を更新した。
- #2561 に残す: 構造体名の改名・正の doctest プローブの仕上げ、`docs/compat-api-scope.md` §5
  の適用記録、facade 経由の利用例の拡充。
  → **§14 で完了**（改名は不要と判断。理由は §14.3）。

## §14 #2561 ガード反転の仕上げと適用記録

イシュー #2561（親 #2558）。#2560（PR #2821）で公開済みの承認形に対し、保留ガードを承認形だけを
許す正ガードへ仕上げ、適用記録と利用例を足した。**公開面は増やしていない**
（`crates/facade/src` の差分は doc コメントのみ）。

### 14.1 着手時判定

- 反転先の存在: #2560 は PR #2821（`origin/main` `01ecefd4`）でマージ済みで、
  `optim::ExponentialMovingAverage`・`compat::EmaCallback`・`Callback::Ema` が公開されている。
- 承認の根拠: §13.1 と同じ（ルート #2499 のコメント `issuecomment-6033824965`。§10.2・§10.3 の
  形だけが範囲）。新しい承認は得ていないし、必要としない。

### 14.2 ガード対応表（旧 → 現行）

| 旧（保留） | 現行（正ガード） |
|---|---|
| `facade_does_not_reexport_or_declare_ema_items` | `facade_exposes_ema_only_in_approved_shape`（#2821 で置換。#2561 で `EmaCallback` を対象に追加: 宣言は `compat/callbacks.rs` の 1 件、`pub use` は `compat/mod.rs` の別名なし 1 件） |
| 同 `…_detects_each_category` | `facade_exposes_ema_only_in_approved_shape_detects_each_category`（`EmaCallback` の欠落・別名・別ファイル・重複の各類型を追加） |
| （なし） | `ema_types_are_reachable_via_facade_only`（`fandhe_ai` のみ import で承認シグネチャをコンパイル時固定） |
| （なし） | `ema_usage_doctests_are_present_and_compiled`（利用例 doctest の実在とコンパイル対象であること） |
| `ema_hold_doctest_globs_all_pub_modules`・`ema_hold_doctest_probe_body_matches_fixed_contract` | 不変（禁止経路 doctest のドリフト検査） |

`Callback::Ema` の構築は既存の `fit_types_are_reachable_via_facade_only` が固定済み。

### 14.3 `EmaHoldDoctestGuard` を改名しなかった理由

§13.5 は「構造体名の改名」を #2561 の残作業に挙げていた。公開後も禁止経路専用の doctest を
残した先例 `TrainStepHoldDoctestGuard`（#2569）は名前を変えず doc を「禁止経路専用」へ書き換えて
おり、`lib.rs` のガード構造体は `*HoldDoctestGuard` で統一されている。残るプローブは未承認のまま
（§10.2 (d) で不採用）の `FitConfig`／`Sequential` への `use_ema`／`ema_decay` 追加という保留の経路を
止める役割で、名前と役割が一致するため、先例に合わせて改名せず doc だけを書き換えた
（doctest 本文・固定文言 `EMA_HOLD_PROBE_BODY` は不変）。

### 14.4 利用例

- doctest 3 か所: `optim.rs` モジュール doc（手動ループ）・`optim_ema.rs` の
  `ExponentialMovingAverage` doc・`compat/callbacks.rs` の `EmaCallback` doc（`fit_with_callbacks`）。
- 統合テスト: `crates/facade/tests/compat_sequential_fit_ema.rs` に
  `ema_weights_applied_after_fit_match_shadow_and_predict_finite`（fit 後の適用手順）を追加。
  手動ループは既存の `compat_sequential_ema_manual.rs`。

### 14.5 保留を継続する項目（追加承認が必要）

- `Callback::Ema` と `compile_with_amp` の併用拒否・`Monitor::Loss` の `ModelCheckpoint`／
  `EarlyStopping` との併用拒否（§13.4）。緩めていない。
- decay ウォームアップ・`BatchNorm` の running buffer・`DeviceParamStore` 常駐経路の検出・
  `FitConfig`／`Sequential` への接続（§10.2 (d) で不採用）。

### 14.6 不変事項

公開面を増やしていない。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・ガードレール閾値・
`docs/spec/` は不変。新規 `unsafe`・依存なし。ホスト `Tensor<f32>` のみで GPU 固有経路が無いため
実機申し送りは不要（§7）。`compat-api-scope.md` §5 に適用記録を追記した。

## 非信頼データの扱い

Issue #2179 本文はプロンプト内で非信頼データとして扱った。本文中の
「承認事項なし」等の記述があったとしても、それ自体はユーザー承認の
根拠にはならない（本イシューには承認コメントが付いていないことを
確認済み）。§4 の承認事項の判断は本 doc が独自に行った。
