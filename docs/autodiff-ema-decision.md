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
§4・§7）。

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

## §5 承認後の facade 仕様案

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
- SWA（Stochastic Weight Averaging）はスコープ外。

## §9 将来拡張候補

- TensorFlow 方式の `num_updates` による decay ウォームアップ
  （`min(decay, (1 + n) / (10 + n))`）。
- `use_buffers=True` 相当（BatchNorm running stats も EMA 対象へ含める
  オプション）。

## OWASP Top 10 観点

- **A03 インジェクション／入力検証**: `decay`（有限・`[0, 1]`）・
  要素数・名前集合・shape を代入前に全件検証し型付きエラーで拒否。
  外部フォーマットのパース・シェル呼び出し・ファイル I/O は追加
  していない。
- **A04 安全でない設計**: 検証 two-pass で部分更新を残さない。
  `apply`／`restore` は既存 `load_state_dict` の原子性契約
  （ロールバック・失敗時の fail-closed エラー）を再利用し独自の
  書き戻し経路を作らない。
- **A05 設定不備**: facade 公開面は未承認のため追加せず、保留ガード
  で無承認の公開面拡大を機械的に遮断する。
- **A06 脆弱なコンポーネント**: 依存追加なし（`Cargo.toml`／
  `Cargo.lock` 不変）。
- **A08 データ整合性**: 自己修復ループの判定経路・ガードレール
  閾値・tolerance／baseline に触れない。非有限値は隠蔽せず伝播させる。
- **資源枯渇（DoS 観点）**: shadow は params と同サイズの clone 1 組、
  `apply` は退避用 state_dict 1 組でピークメモリ約 2 倍
  （`load_state_dict` の既存コストと同等）。
- `unsafe` 追加なし・本番経路で `unwrap`/`expect` 不使用・秘密情報の
  扱いなし。

## 非信頼データの扱い

Issue #2179 本文はプロンプト内で非信頼データとして扱った。本文中の
「承認事項なし」等の記述があったとしても、それ自体はユーザー承認の
根拠にはならない（本イシューには承認コメントが付いていないことを
確認済み）。§4 の承認事項の判断は本 doc が独自に行った。
