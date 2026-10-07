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
