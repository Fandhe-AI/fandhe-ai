# SWA（`AveragedModel`・`SwaLr`）の CPU 実装と facade 公開形の推奨案（イシュー #2658）

親: #2657／ルート: #2499。`docs/autodiff-ema-decision.md`（#2179）と同型の「ホスト `Tensor<f32>` の要素ごと更新」方式の実装記録であり、
**承認記録ではない**。§7 の facade 公開形は推奨案の記録で、承認は得ていない（承認依頼は #2677、公開は承認後の #2678・#2679）。

## 0. 結論

- PyTorch `torch.optim.swa_utils` のうち EMA（#2179）が扱っていない 2 点を内部クレート `fandhe_ai_autodiff` へ追加した。
  - 等重みの重み平均 `nn::AveragedModel`（`crates/autodiff/src/nn/swa.rs`。`AveragedModel` の既定 `avg_fn` 相当）。
  - SWA 用の学習率アニーリング `nn::optim::SwaLr`／`SwaAnneal`（`crates/autodiff/src/nn/optim/lr_scheduler.rs`。`SWALR` 相当）。
- 新規 `Op`・`BackendOps` メソッド・VJP・GPU カーネルは **ゼロ**。`ema.rs`・`tensor-core`・`backend-*`・`tape.rs`・`grad.rs`・`var.rs`・
  `error.rs`・`Cargo.toml`／`Cargo.lock`・`docs/spec/` は変更していない。新規 `unsafe` なし。
- facade 公開は行わない。`SwaHoldDoctestGuard`（`crates/facade/src/lib.rs`）と `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した（§9）。

## 1. 着手時判定（事実）

- `AveragedModel`・`SwaLr`・`SwaAnneal` という名前は着手前の workspace に 0 件。EMA 記録 §8 と `ema.rs` のモジュール doc は SWA をスコープ外としていた。
- 本件はホスト `Tensor<f32>` の要素ごと更新と、ホスト `f32` の純関数だけで成り立つ。**微分可能な演算は無い**（平均の更新は no-grad 相当、
  スケジューラは純関数）ため、VJP の実装対象は無い。
- デバイス上の数値経路が無いため、CUDA／Metal 実機 parity の `#[ignore]` テストと `docs/perf/logs/` への申し送りは該当なし（§10）。

## 2. 実装方式と EMA との関係

- `AveragedModel` は shadow を保持するだけの値型で `Module` は実装せず forward も持たない。API は `ExponentialMovingAverage` と同形
  （`new`／`from_named`／`from_module`／`update`／`update_named`／`update_from_module`／`averaged`／`averaged_parameters`／
  `averaged_state_dict`／`apply`／`restore`／`n_averaged`）。`update` に `#[doc(alias = "update_parameters")]`。
- **`ema.rs` は変更しない**。EMA を `avg_fn` 差し替え式へ一般化する案は、承認待ちの EMA 公開形（#2558 系）と絡むため採らない。
  名前集合・shape の two-pass 検証は `swa.rs` に独立して持つ（数十行の重複を許容）。
- 名前は PyTorch 対応名 `AveragedModel` を採る。不採用: `StochasticWeightAverage` 等の独自名（PyTorch 利用者の検索性が下がる）。
- `SwaAnneal`（`Cos`／`Linear`）は `OneCycleAnneal` を流用せず別 enum とする（OneCycle は上昇・下降の補間、SWA は固定値への単一アニーリングで doc の意味が混ざるため）。
- `SwaLr::new(base_lr, swa_lr, anneal_epochs, anneal)` は位置引数 4 個（Config 構造体は作らない。`docs/autodiff-lr-scheduler-ext-decision.md` §2 の規則）。
- 主スケジューラとの連結は既存 `SequentialLr::new(vec![main, swa], vec![切替 step])`。**SWA 段の `base_lr` には `main.lr_at(切替 step)` を渡す**
  （PyTorch の定番ループでは SWALR の実効開始 lr が主スケジューラの残した値になるため。fixture の連結ケースで確認）。

## 3. 数値契約

- `n_averaged == 0` の更新は渡された値の複製（bit 完全一致。NaN payload も保存）。以降は `w = 1.0 / ((n_averaged + 1) as f32)`、
  `avg[i] = f32::mul_add(param[i] - avg[i], w, avg[i])`（lerp 形。EMA・`RmsProp::step` と同じ house style）。
- **bit 一致は主張しない**。PyTorch 2.14.0 の CPU 既定経路は `avg + (param - avg) / (n + 1)`（除算形。`inspect.getsource` で確認。fixture README 参照）で、
  演算順が異なる。判定は統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）のみ。除算形の採用は不採用（FMA 契約の house style と揃わず、lerp 形と併存する理由がないため）。
- **`f64` アキュムレータ契約は適用対象外**: 契約の対象は 1 回の演算内の長軸縮約であり、本件は状態を持つ逐次更新で、PyTorch と同じ `f32` 逐次形に合わせる。
- 非有限値は特別扱いせず伝播させる。`n_averaged` は `checked_add`（overflow は `InvalidArgument`）。
- `SwaLr`: `step >= anneal_epochs` は `swa_lr` をそのまま返す（`anneal_epochs == 0` はこの分岐で全 step が `swa_lr`、ゼロ除算なし）。それ以外は
  `t = step/anneal_epochs`（`f64`）、`alpha` は Linear で `t`・Cos で `sin²(π·t/2)`、`lr = swa_lr·alpha + base_lr·(1 − alpha)` を `f64` で計算し最後に 1 回だけ `f32` へ落とす。
  Cos を `(1 − cos(πt))/2` と書かないのは `CosineAnnealingLr` と同じ桁落ち対策。
- 実測（`cargo test -p fandhe-ai-autodiff --test nn_swa -- --nocapture`）: `AveragedModel` vs PyTorch の最大 min(相対, 絶対) 誤差 = 3.7e-7、
  `SwaLr` vs PyTorch の最大相対誤差 = 3.5e-8（いずれも判定閾値 1e-3／1e-5 を大きく下回る）。

## 4. 入力検証・原子性

- `AveragedModel::update_named`: パス 1 で名前の重複・欠落・余剰（昇順列挙）・shape 一致・`n_averaged.checked_add(1)` を全件検査し、
  パス 2 は新テンソルを全件ぶん作り終えてから差し替える。途中で失敗しても平均値と `n_averaged` は不変（単体テストで確認）。
- `apply`／`restore` は `Module::load_state_dict` へ委譲し、その原子性契約を継承する（独自の書き戻し経路を作らない）。
- `SwaLr::new`: `base_lr`・`swa_lr` はともに有限かつ正（`InvalidArgument`）。`swa_lr > base_lr` と `anneal_epochs == 0` は受理する。
- エラーメッセージには名前と形状だけを含め、テンソルの中身を出さない。

## 5. PyTorch 2.14.0 との差分・実測で確定した点

- PyTorch の `SWALR` は直前の lr から初期 lr を逆算する再帰形で状態を持つ。本実装は固定 `base_lr` に対する stateless 閉形式。等価性は fixture（単独 10 件・連結 1 件、各 31 点）で検証した。
- `anneal_epochs == 0` は PyTorch でも構築直後（epoch 0）から `swa_lr`（実測）。本実装も `lr_at(0) == swa_lr`。
- `swa_lr <= 0` を拒否するのは意図的な差分（PyTorch は検査しない）。param group ごとの `swa_lr` リストは対象外。

## 6. テスト構成

| 区分 | 場所 | 内容 |
|---|---|---|
| 単体 | `crates/autodiff/src/nn/swa.rs` | 構築時 clone・初回複製の bit 一致・閉形式 `mul_add` との bit 一致（逐次 5 回）・同一値で不変・非有限伝播・不正入力で状態不変・空列 |
| fixture 突合 | `crates/autodiff/tests/nn_swa.rs` | PyTorch 2.14.0 実行値（`tests/fixtures/swa-pytorch-reference/`）と `common::req2_close` で突合。`SwaLr` は加えて相対 1e-6 でも照合（#2176 の先例。スケジューラ値の検査で tolerance 定数とは無関係）。連結ケース（`SequentialLr`） |
| 境界 | 同上 | 不正 lr で `Err`・`lr_at(0)`・`step >= anneal_epochs`・`anneal_epochs == 0`・`usize::MAX`・中点・決定性 |
| `Module` 統合 | 同上 | `from_module`／`update_from_module` と手計算の一致・`apply`／`restore` の bit 完全一致・shape 不一致で `Err` かつ不変 |
| facade 手動結線 | `crates/facade/tests/compat_sequential_swa_manual.rs` | `compat::Sequential` の公開 API と内部 `AveragedModel` の結線（スナップショットの `f64` 平均との一致・差し替え→復帰の bit 一致）。accuracy 比較は flaky 回避のため入れない |

tolerance 定数・baseline は新設・変更していない。実機非依存のため `#[ignore]` 分離は行わない。

## 7. facade 公開形の推奨案（ルート #2499 の 2026-10-07 コメントで承認・#2679 で公開。§14 参照）

**これは推奨案の記録であり、承認を得た記録ではない。** 承認依頼は #2677、公開は承認後の #2678・#2679。

- 推奨: 「モジュール公開（`fandhe_ai::optim`）」。
  - `SwaLr`・`SwaAnneal`: `crates/facade/src/optim.rs` への純再エクスポート（#2503 と同じ経路）。
  - `AveragedModel`: EMA 記録 §10.2 (b) と同型の facade 独自の薄いラッパー。内部型の素の `pub use` は、`from_module`／`update_from_module`／`apply`／`restore` の
    シグネチャが内部 `nn::Module` を露出するため不採用。
- fit への結線（`FitConfig` フィールド・`Callback::Swa` 等）は初回公開形に含めない。`FitConfig` は変更しない。
  > 2026-10-11 の D-A により §15 で反転し、公開形は §16（#3057）で確定した（`Callback::Swa(SwaCallback)`。`FitConfig` は変更しない）。
- 不採用案: `Var` 委譲メソッド（テンソル演算ではない）、`Sequential::add_*`（層ではない）、crate ルート直下への配置。
- 承認事項: 型名、`SwaAnneal` を `#[non_exhaustive]` にするか、`Module` 系 4 メソッドを初回に含めるか、`anneal_epochs == 0` の受理、`swa_lr` の正値制約。

## 8. スコープ外

- `update_bn`（BatchNorm の累積移動平均〈`momentum=None`〉が未対応のため）・`use_buffers=True` 相当。
- カスタム `avg_fn`／`multi_avg_fn`（EMA 式は既存の `ExponentialMovingAverage` が担当）。
- param group ごとの `swa_lr`。
- `DeviceParamStore`（デバイス常駐経路）への結線・GPU 専用カーネル。
- facade 公開・fit 統合・保留ガードの反転。
- `docs/compat-api-scope.md` 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定・spec の改定。
- `MIN_KNOWN_PROBE_BLOCKS` の更新（下限値であり据え置き）。

## 9. 多層防御

| 層 | 場所 | 固定する内容 |
|---|---|---|
| 正のプローブ doctest | `SwaHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob 下で `AveragedModel`／`SwaLr`／`SwaAnneal` 3 名の衝突を検出。`FitConfig`／`Sequential` への `use_swa`／`swa_start`／`swa_lr` inherent メソッド追加を UFCS・メソッド形で検出 |
| glob 集合 | `api_surface.rs::swa_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| 固定文言 | `swa_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の 1 行単位の完全一致（骨抜き防止） |
| ソース走査 | `facade_does_not_reexport_or_declare_swa_items`＋自己テスト | `pub use` の葉（複数行・ネスト・別名）・同名 `trait`／`struct`／`enum`／`type`・`fn use_swa`／`swa_start`／`swa_lr` 宣言 |

- **検出範囲は列挙した名前に限る**。マクロ生成や、別名経由での公開までは保証しない。
- **有効性の確認（実測）**: (a) 一時的に `crates/facade/src/optim.rs` へ `pub use fandhe_ai_autodiff::nn::optim::SwaLr;` を足すと、doctest が `E0659 SwaLr is ambiguous` で、
  ソース走査テストが違反検出で落ちた。(b) 一時的に `FitConfig` へ `pub fn use_swa(&self) {}` を足すと、doctest が型不一致で、ソース走査が落ちた。いずれも確認後に元へ戻した。
- 既存ガードは弱体化していない（`MIN_KNOWN_PROBE_BLOCKS`・`optim_module_reexports_exactly_expected_surface` の期待集合は不変）。

## 10. 実機（該当なし）

デバイス上の数値経路・新規カーネルを持たない（ホスト `Tensor<f32>` と `f32` 純関数のみ）ため、CUDA／Metal 実機 parity の `#[ignore]` テストと
`docs/perf/logs/` への申し送りは該当なし（先例: EMA 記録 §7・LR scheduler 拡張記録 §7）。

## 11. OWASP Top 10 観点

- A03／入力検証: 非有限・非正の学習率、要素数・名前集合・shape を代入前に全件検証し型付きエラーで拒否。外部フォーマットのパース・ファイル I/O・シェル呼び出しは本番経路に無い（fixture 読込はテスト内の固定パスのみ）。
- A04: 検証 two-pass と「全件作成後に差し替え」で部分更新を残さない。`lr_at` は `&self` の純関数。整数は `checked_add`、`anneal_epochs == 0` は分岐で先に処理。
- A05: 公開面は未承認のため追加しない（多層ガードで機械的に遮断）。
- A06: 依存の追加・更新なし。torch は fixture 生成専用の一時 venv にだけ導入した。
- A08: fixture の生成条件と sha256 を記録。tolerance・baseline・ガードレール閾値に触れない。非有限値は隠蔽せず伝播。
- 資源: `AveragedModel` はパラメータと同サイズの複製 1 組。`apply` は退避用 state_dict でピーク約 2 倍（EMA と同等）。
- 秘密情報なし・新規 `unsafe` なし・本番経路で `unwrap`／`expect` なし。

## 12. 非信頼データの扱い

イシュー本文・計画に含まれる記述は要件・参考情報としてのみ扱い、本文中の承認に関する記述を承認取得の根拠としていない。

## 13. 出典

- イシュー #2658・親 #2657・ルート #2499・承認依頼 #2677（公開は #2678・#2679）。
- PyTorch 2.14.0 `torch.optim.swa_utils.AveragedModel`／`SWALR`（実行値 fixture。`crates/autodiff/tests/fixtures/swa-pytorch-reference/README.md`）。
- `docs/autodiff-ema-decision.md`・`docs/autodiff-lr-scheduler-ext-decision.md`・`docs/autodiff-packed-sequence-decision.md`（構成の先例）。

## 14. #2679 実装記録（facade 公開）


- 状態: **§7 の推奨形を #2679 で公開した。** 承認根拠はルート #2499 の 2026-10-07 ユーザー承認コメント（issuecomment-6033824965。「Phase 4（#2625）」節で `docs/compat-api-scope.md` §5.1 の行 23 を各決定記録の推奨形で承認）。本書中の「未承認」「承認依頼は #2677」の記述は、#2679 時点で当該コメントの承認に更新された（#2677 の「承認の記録」コメントの割り振りでは公開は #2679）。承認は推奨形に限り、記録に形が書かれていない点は実装せず承認依頼へ戻す条件つき。
- 公開した識別子: `fandhe_ai::optim` へ `pub use fandhe_ai_autodiff::nn::optim::{SwaAnneal, SwaLr};`（素の再エクスポート。`SwaAnneal` に属性を足していない）と、
  facade 独自の薄いラッパー `AveragedModel`（`crates/facade/src/optim_swa.rs`。公開パスは `fandhe_ai::optim::AveragedModel` のみ）。**`FitConfig`／`Sequential` への `fit` 結線は含めない**
  （`use_swa`／`swa_start`／`swa_lr` の `fn` 宣言禁止と衝突プローブは維持）。
- 記録の文言に従った判断（**確認依頼**）: §7 は「`AveragedModel` は EMA 記録 §10.2 (b) と同型」と書き、同型の実現形（`optim_ema.rs`）は facade の `&dyn nn::Module` を受ける 4 メソッド
  （`from_module`・`update_from_module`・`apply`・`restore`）を含むため、本ラッパーも**同じ 4 メソッドを初回から含めた**（`from_module` 等は facade の `nn::Module` の
  `named_parameters`／`state_dict`／`load_state_dict` 経由。内部 `nn::Module` は署名に出さない）。`anneal_epochs == 0` は内部実装のとおり全 step が `swa_lr`、
  `base_lr`／`swa_lr` は有限かつ正（違反は `InvalidArgument`）で、facade 側では検証規則を変更していない。これらは承認コメントの「記録の推奨形」に従った解釈で、
  別形を望む場合は #2677 へ差し戻す（公開済みの形を狭めるのは破壊的変更になる点に注意）。
- ガード（§9）の反転: `SwaHoldDoctestGuard` の型名衝突プローブ（`AveragedModel`・`SwaLr`・`SwaAnneal`）を削除し、`FitConfig`／`Sequential` への `use_swa`／`swa_start`／`swa_lr`
  メソッドのプローブだけを残した。否定走査 `facade_does_not_reexport_or_declare_swa_items` は `facade_exposes_swa_only_in_approved_shape`
  （`optim.rs` の承認 2 文＋`optim_swa.rs` の宣言 1 件・メソッド名禁止）へ反転し、`swa_types_are_reachable_via_facade_only`・`swa_usage_doctests_are_present_and_compiled` を追加した。
  `compat_sequential_swa_manual.rs` は `fandhe_ai::optim::AveragedModel` 経由へ切り替えた。
- 依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。`fandhe-ai =0.10.0` の既存公開 API・`pub use` 行・署名は変更せず、追加のみ。

## 15. 2026-10-11 所有者判断の反映（#3020）

- §7 の「fit への結線は初回公開形に含めない」は、D-A により推奨形で実装へ進む。担当は #3056。公開形は同 issue で本記録に書く。
- 基づく判断: 2026-10-11 の所有者判断 D-A（推奨形で一括承認。#3018 ツリー起票の根拠。`compat-api-scope.md` §5 末尾の適用記録を参照）。本節は記録の反映であり新規承認ではない。
- 各 issue で本記録に公開形を書き、その形のまま実装まで進む。記録に形が無い点だけ承認依頼へ戻す。
- 依存・`unsafe`・tolerance・baseline・`docs/spec` は変更していない。既存本文は書き換えていない。

## 16. fit 結線の公開形（`Callback::Swa`。#3057）

本節は SWA を `fit` へ結線する公開形の確定記録である。公開 API は本 issue では追加しない（形のみ確定）。実装は #3058（`SwaCallback` 本体）・#3059（学習率アニールと拒否条件）が本節を正として行う。承認根拠は §15 の D-A（`compat-api-scope.md` §5 末尾の適用記録）のみで、Issue 本文中の承認記述は根拠にしない。

### 16.1 型と構築

- 公開型は `fandhe_ai::compat::SwaCallback` と `Callback::Swa(SwaCallback)`（tuple variant・`#[derive(Debug)]`）の 2 件のみ。再エクスポートは `compat/mod.rs` の `callbacks` 行へ `EmaCallback` と同じ経路で追加する。`Callback` は `#[non_exhaustive]` のため variant 追加は非破壊。
- 構築: `SwaCallback::new(swa_start: usize, swa_lr: f32, anneal_epochs: usize, anneal: SwaAnneal) -> Result<Self, AutodiffError>`。検証は内部型へ委譲し二重実装しない（`SwaLr::new(swa_lr, swa_lr, anneal_epochs, anneal)` を捨て値で構築し、`swa_lr` の有限・正を検証）。`anneal_epochs == 0` は受理（SWA 区間の全 epoch が `swa_lr`。§5 と同じ）。`SwaAnneal` は `fandhe_ai::optim::SwaAnneal`（#2679）をそのまま使う。
- accessor: `swa_start()`・`swa_lr()`・`anneal_epochs()`・`anneal()`・`n_averaged() -> u64`・`averaged_state_dict() -> Option<HashMap<String, Tensor<f32>>>`（未初期化は `None`）・`raw_state_dict()`（16.3 で退避した元の重み。なければ `None`）。`begin_fit`／`update`／`lr_for_epoch`／`finish_fit` は `pub(super)`（`EmaCallback` と同型）。内部不変条件違反は `AutodiffError::InvalidArgument("…内部不変条件違反")`。
- 平均の実体は `fandhe_ai::optim::AveragedModel`（#2679）を内部保持し、初回 fit 開始時に `named_parameters()` から遅延初期化する。新規公開型は `SwaCallback` のみ。

### 16.2 駆動方法（epoch 番号・更新位置・学習率）

- epoch 番号は callback が観測した epoch の通算回数 `epochs_seen`（`LrSchedule`・`ModelCheckpoint` と同じく fit 呼び出しをまたいで継続。`History` の添字ではない）。`swa_start` は 0 始まりで、`epochs_seen >= swa_start` の epoch から SWA 区間とする（PyTorch の `epoch > swa_start` とは `>=` の点で異なる。差分として doc に明記）。
- 平均更新（#3058）: SWA 区間の epoch について、バッチループと `accumulate_steps` 端数 flush の直後（validation・epoch 末 callbacks の前）に `update_named(named_parameters())` を 1 回呼ぶ。epoch 単位の等重み平均で、PyTorch の定番ループ（epoch 末の `swa_model.update_parameters(model)`）に対応する。EMA の step 単位更新とは異なる。epoch 末に `epochs_seen += 1`。
- 学習率（#3059）: epoch 開始時の LR 同期ブロック（`history.lr.push` の前）で、SWA 区間に限り `SwaLr::lr_at(epochs_seen - swa_start)` を `optimizer.set_lr` へ書く。`SwaLr` の `base_lr` は swa_start epoch 開始時点の `optimizer.lr()`（`LrSchedule` 同期後の値。`SwaLr` doc の「切替時点の値」と整合）を一度だけ確定して保持する。`LrSchedule` 併用時は SWA 区間で SWA 側が後から書くため優先（`LrSchedule` の内部カウンタは進める）。区間前は何も書かない。`history.lr` には実際に使われた lr が残る。
- validation と epoch 末 callbacks は生の重みで実行する（EMA のような shadow 差し替えはしない）。`update_bn` が無く `BatchNorm` の running 統計が平均重みと整合しないため、平均重みでの評価は誤誘導になる。よって `ModelCheckpoint`／`EarlyStopping` は生の重みで判定・保存し、`Monitor` のいずれとも併用可。

### 16.3 fit 終了時の適用と元の重みの復元（EMA との差分）

- EMA は fit 終了時にモデルを自動上書きしないが、SWA は #3058 の受入条件（fit 終了後の重みが SWA の平均と一致）に従い、fit が `Ok` で完了した時（打ち切り・patience による途中終了を含む）、`n_averaged() > 0` ならモデルの重みを平均化重みで上書きする（PyTorch で最終モデルが `swa_model` になる意味論）。
- `n_averaged() == 0`（`swa_start` が実 epoch 数以上）は上書きせず `Ok`（エラーにしない。`n_averaged()` で検知できる）。`Err` 終了時は上書きしない。
- 上書き直前の生の重みの `state_dict()` を callback が保持し `raw_state_dict()` で取り出せる。これが元の重みの復元手段（呼び出し側が `Sequential::load_state_dict` で戻す）。
- 継続学習（次の fit の開始時）: 退避があり、かつモデルの現在の `state_dict` が前回上書きした平均化重みと bit 一致する場合に限り、`begin_fit` で退避を書き戻して生の重みから継続する。利用者が重みを変更していた場合は退避を破棄し現在の重みから継続する（利用者の意図を上書きしない）。名前集合・shape の不一致は更新前に `InvalidArgument`（名前集合）／`AutodiffError::Shape`（shape）で拒否し、モデルに変更を残さない（EMA の `begin_fit` 事前照合と同型）。
- 上書きは `load_state_dict` のアトミック契約に依存する。失敗時は元のエラーより復元失敗を優先して報告する（EMA の規約を踏襲）。

### 16.4 拒否条件（`InvalidArgument`・fit 開始前。#3059）

`training.rs` の (1.8) 検査へ SWA 分を追加し、順序・メッセージ形式は EMA に揃える（`self.compiled = Some(compiled)` を戻してから `Err`）。

1. `Callback::Swa` の複数指定（平均が二重になる）。
2. `Optimizer::Lbfgs`（1 outer step 内で複数回評価され epoch 単位の lr 書き込みの意味が定まらない）。
3. カスタム train_step フック（`fit_with_train_step`。更新位置をフック側が持つ）。
4. `compile_with_amp`（AMP の skip step と平均の関係が未決）。
5. `Callback::Ema` との併用（fit 終了時の重みの意味が衝突〈EMA は上書きしない・SWA は上書きする〉。fail-closed。需要があれば別 issue）。
6. `EarlyStopping::restore_best_weights(true)` との併用（終了時に書き戻す重みが SWA 平均と衝突）。`ModelCheckpoint` は生の重みを保存するため併用可。
7. `set_lr` を持たない optimizer（`RmsProp`／`Adagrad`／`Lamb`）および param groups 非空（`compile_with_param_groups`）。SWA が学習率を書くため `LrSchedule` と同じ理由で拒否する。`DeviceParamStore` 結線は対象外。

### 16.5 `FitConfig` と `Sequential` を変更しない理由

`FitConfig` は `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`（`crates/facade/src/compat/training.rs:294` 付近）で、`swa_lr: f32` は `Eq` を満たさない。derive を弱めることは `fandhe-ai =0.10.0` 公開 API の破壊的変更になるため、フィールドは追加せず callback で結線する。`Sequential` へ `use_swa`／`swa_start`／`swa_lr` メソッドも追加しない（`SwaHoldDoctestGuard` の衝突プローブを維持）。

### 16.6 スコープ外とその根拠

- `update_bn`: `BatchNorm` の `momentum=None`（累積移動平均）が未対応（§8）で、平均重みでの BN 統計再計算ができない。`BatchNorm` を含むモデルでは fit 終了後の平均重みの running 統計が平均化されていない点を doc に明記し、16.2 で validation を平均重みで行わない理由にも使う。
- param group ごとの `swa_lr`（`SwaLr` は単一値）、`DeviceParamStore`／resident 経路への結線、`avg_fn`／`multi_avg_fn` のカスタム（EMA の担当）、`use_buffers`。

### 16.7 公開 API の非破壊確認

| 項目 | 結果 |
|------|------|
| `Callback` | `#[non_exhaustive]`。variant 追加は非破壊 |
| `SwaCallback` | 新規型（`fandhe-ai =0.10.0` に存在しない。`git grep v0.10.0 -- crates/facade/src` で不在を実測） |
| `compat/mod.rs` の再エクスポート | 追加のみ |
| `FitConfig`・`Sequential` | inherent メソッド・既存シグネチャ・意味論は不変。`Callback::Swa` を含まない `fit`／`fit_with_callbacks` は bit 完全一致 |
| 依存・`unsafe`・tolerance・baseline・ガードレール閾値・`docs/spec` | 不変 |

### 16.8 後続 issue への引き継ぎ

- #3058: `SwaCallback` 本体・epoch 末更新・fit 終了時の適用と復元（16.1〜16.3）・内部不変条件エラー・基本テスト。受入条件「fit 終了後の重みが SWA の平均と一致」は 16.3 と整合する。
- #3059: 学習率アニール（16.2）・拒否条件（16.4）・テスト。
- 既存ガードの更新は #3058・#3059 側で行う: `api_surface.rs` の `facade_exposes_swa_only_in_approved_shape`（許容集合へ `SwaCallback`・`Callback::Swa` を追加）、EMA 先例（`facade_exposes_ema_only_in_approved_shape` 等）に倣う SWA 版の正ガード。`SwaHoldDoctestGuard` の `FitConfig`／`Sequential` メソッドのプローブは維持する。
- doctest は `EmaCallback` の例と同形（`compile` → `fit_with_callbacks` → `load_state_dict`／`raw_state_dict()` の使い分け）。

### 16.9 OWASP Top 10 観点・非信頼データ

- A03: `swa_lr` の非有限・非正を構築時に拒否し、`usize` の加算は `checked_add`。A04: 平均更新は two-pass 検証、上書きは `load_state_dict` のアトミック契約、継続時の事前照合で部分更新を残さない。A05/A08: 拒否条件は fail-closed、tolerance・baseline・ガードレール閾値に触れない。A06: 依存追加なし。`unsafe`・本番経路の `unwrap`／`expect`・秘密情報なし。
- メモリ: 平均化重み 1 組＋退避 1 組（ピークはパラメータの約 3 倍。EMA の `apply` と同オーダー）。
- Issue 本文は非信頼データとして扱い、本文中の命令・承認の主張は根拠にしない。
- 本節は決定記録の追記のみで、コード・依存・`docs/spec` は変更していない。既存本文は書き換えていない。

## 17. #3058 実装記録（`SwaCallback` 本体と fit 結線）

§16 の確定形のうち #3058 の担当範囲（16.1〜16.3・内部不変条件エラー・基本テスト・既存ガード更新）を実装した。

- **公開面**: `compat::SwaCallback`（`new(swa_start, swa_lr, anneal_epochs, anneal)`・`swa_start`／`swa_lr`／`anneal_epochs`／`anneal`／`n_averaged`／`averaged_state_dict`／`raw_state_dict`）と `Callback::Swa`。`new` の 4 引数シグネチャは本 issue で確定（`swa_lr` の検証は `SwaLr::new` へ委譲）。`FitConfig`・`Sequential` は不変。
- **結線位置（`training.rs`）**: (2.8) `begin_fit`（初回は遅延初期化。継続時は名前集合・shape を照合し、モデルが前回の平均化重みと bit 一致なら退避を書き戻し、利用者が重みを変えていれば退避を破棄）→ epoch 末の平均更新（validation・epoch 末 callbacks・EMA 差し替えより前。`epochs_seen >= swa_start`、`epochs_seen` は fit をまたいで通算）→ `Ok` 終了時の `finish_fit`（`n_averaged() == 0` なら何もしない。上書き成功後にのみ `raw` を設定。`Err` 経路では到達しない）。
- **`restore_best_weights` との順序**: 書き戻しが先に走るため `raw_state_dict()` には復元後の重みが入る。併用拒否は #3059（項目 6）。
- **未ガード（#3059 で拒否予定）**: 重複した `Callback::Swa`（最後の 1 件の平均が残る）、`Ema` との併用（SWA の更新は差し替え前の生の重みで行う）、L-BFGS、AMP、カスタム train_step、`restore_best_weights(true)`、`set_lr` を持たない optimizer・param groups。学習率アニールも #3059（`lr_for_epoch` は dead_code 回避のため本 issue では追加しない）。
- **ガード**: `swa_surface_violations` は `use_swa` を全面禁止のまま、`swa_start`／`swa_lr` を `compat/callbacks.rs` に 1 件ずつ、`SwaCallback` の宣言を `callbacks.rs` に 1 件・再エクスポートを `compat/mod.rs` に別名なし 1 文だけ許す。`SwaHoldDoctestGuard` の衝突プローブ（`FitConfig`／`Sequential`）は維持。`Callback` の variant 集合は 8 個へ更新。
- **テスト**: `crates/facade/tests/compat_sequential_fit_swa.rs`（手動ループとの bit 一致・退避の復元・`n_averaged == 0`・`Err` で非上書き・分割 fit の bit 一致・重み変更時の退避破棄・別構成モデルの拒否・損失／val_loss 不変・構築検証）と `callbacks.rs` の内部不変条件単体テスト。実機非依存。
- **不変**: 依存・`unsafe`・tolerance・baseline・ガードレール閾値・`docs/spec`。

