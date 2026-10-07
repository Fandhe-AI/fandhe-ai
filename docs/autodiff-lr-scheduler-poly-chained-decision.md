# PolynomialLR・ChainedScheduler の CPU 実装記録と facade 公開形の推奨案（イシュー #2659）

> 本書は実装記録と推奨案の記録であり、**承認記録ではない**。facade 公開形は未承認で、承認依頼は #2677、公開自体は承認後の #2679。

## 0. 結論

- `PolynomialLr`・`ChainedScheduler` を内部クレート（`fandhe_ai_autodiff::nn::optim::lr_scheduler`）に実装した。いずれも既存スケジューラと同型の stateless 閉形式（`LrScheduler::lr_at(&self, step) -> f32`）で、`f64` で計算し最後に 1 回だけ `f32` へ落とす。
- `LrScheduler` trait・`Op`・`BackendOps`・`Var`・VJP・カーネル・依存・`unsafe` は変更・追加していない。
- facade 公開面は追加していない（`LrSchedulerPolyChainedHoldDoctestGuard`＋`api_surface.rs` の否定ガードで固定）。
- PyTorch 2.14.0 実行値 fixture との最大相対誤差: `PolynomialLr` 4.7e-8、`ChainedScheduler` 8.6e-8（いずれも `f32` 出力丸め由来。統一複合判定と相対 1e-6 の両方を通過）。

## 1. 着手時判定（事実）

- 着手時の `origin/main` に `PolynomialLr`・`ChainedScheduler` の名前は 0 件。
- 既存の facade 公開済みスケジューラ（#2503）とは別に、SWA（#2658）と同じく「内部実装のみ・公開保留」の枠で扱う。

## 2. 型名とシグネチャ

- 名称は `PolynomialLr`・`ChainedScheduler`（既存命名は PyTorch 名の `LR` を `Lr` にするだけで、`Lr` を持たない名前は原語のまま。`CosineAnnealingWarmRestarts` が先例）。
- 不採用案: `ChainedLr`（PyTorch 利用者の検索性が下がる）。型名は #2677 の承認事項。
- `PolynomialLr::new(base_lr: f32, total_iters: usize, power: f32)`、`ChainedScheduler::new(base_lr: f32, schedulers: Vec<Box<dyn LrScheduler>>)`。位置引数のみで Config 構造体は作らない。

## 3. 数値仕様

- `PolynomialLr`: `lr(step) = base_lr · (1 − min(step, T)/T)^power`。PyTorch の再帰形との差は 30 step で最大相対 6.0e-16（計画時実測・6 設定）。`step >= T` かつ `power > 0` は厳密に 0、`power == 0` は全 step で `base_lr`（`0f64.powf(0.0) == 1.0` に依存。テストで名前を付けて固定）。巨大 `step`（`usize::MAX`）でも panic しない。
- `ChainedScheduler`: `lr(step) = base_lr · Π(s_i.lr_at(step) / base_lr)`。メンバー順に依存しない（逆順でも最大 3.5e-16）。単独メンバーは元のスケジューラと bit 一致。

## 4. `ChainedScheduler` の対応メンバー

`LrScheduler` trait から各メンバーの `base_lr` を取れないため、呼び出し側が全メンバーとチェーンへ同じ `base_lr` を渡す規約（`SequentialLr` と同じ）。

| 区分 | メンバー |
|---|---|
| PyTorch と一致（現在値への純粋な乗算） | `StepLr`・`MultiStepLr`・`ExponentialLr`・`LinearWarmupLr`（`LinearLR(end_factor=1)` 相当）・`PolynomialLr`・`ConstantLr`（係数 1 の恒等） |
| 一致しない（チェーンの外で使う） | `CosineAnnealingLr`・`CosineAnnealingWarmRestarts`・`CyclicLr`・`OneCycleLr`・`LambdaLr`・`SwaLr`・`SequentialLr`・`ReduceLrOnPlateau` |

- 一致しない理由: PyTorch 側の再帰形が加算項を持つか現在値を上書きするため、積の閉形式に落ちない。実測（fixture `non_multiplicative_probe`）で `CosineAnnealingLr(eta_min=0)` + `ExponentialLr` のチェーンは最大相対 0.74 乖離（計画時実測では `T_max` 超過後に約 0.65、`eta_min=0.01` で最大 0.95）。テスト `chain_with_non_multiplicative_member_diverges_from_pytorch_by_design` が乖離を固定しており、これを「直す」ことはしない。
- PyTorch の `ConstantLR(factor, total_iters)` は Rust の `ConstantLr`（常に `base_lr`）とは別物で、fixture のメンバーに使っていない。
- `Box<dyn LrScheduler>` では型でメンバー種別を弾けないため doc と本書で明示する（意図的な制限）。

## 5. 入力検証と PyTorch との意図的な差分

| 項目 | PyTorch | 本実装 | 理由 |
|---|---|---|---|
| `total_iters == 0` | 例外なし・常に `base_lr` | `InvalidArgument` | 閉形式は `0/0` で未定義。ゼロ除数を拒否する既存規則（`StepLr` 等）に揃える。`SwaLr` が 0 を受理するのは除算前の分岐で PyTorch と同値になるためで事情が異なる |
| `power < 0` | `step == total_iters` で `ZeroDivisionError`（遅延） | 構築時に `InvalidArgument` | fail-closed |
| `power == 0` | 全 step `base_lr` | 受理（同値） | — |
| `ChainedScheduler([])` | `ValueError` | `InvalidArgument` | 同じ |

- `base_lr` は両型とも有限かつ正。非有限値は特別扱いせず伝播させる（下流の `set_lr` が拒否する）。
- `set_lr` の下限基準: `Adam`／`AdamW`／`Adadelta`／`Adamax`／`NAdam`／`RAdam`／`Lbfgs`／`Rprop`／`Asgd`／`Adafactor`／`Lion` は grep で `new_lr.is_finite() && new_lr >= 0.0` を確認した。`PolynomialLr` が返す厳密な 0 は `LrSchedule::per_epoch` 経由でもエラーにならず学習が止まるだけ（PyTorch と同じ）。`set_lr` 自体は変更していない。
- `usize → f64` の丸めにより `total_iters > 2^53` では境界直前の step が早めに 0 になりうる（実用上到達しない）。

## 6. fixture とテスト構成

- fixture: `crates/autodiff/tests/fixtures/lr-scheduler-poly-chained-pytorch-reference/`（`gen_reference.py`・JSON・`README.md`。torch 2.14.0+cpu・Python 3.14.4。入力値は `f32` で厳密に表せる値のみ。再生成で同一 sha256）。`PolynomialLR` 6 件、`ChainedScheduler` 4 件、非乗算メンバーの乖離プローブ 1 件。
- `crates/autodiff/tests/nn_optim_lr_scheduler_poly_chained.rs`: fixture 突合（統一複合判定＋相対 1e-6。tolerance 定数は新設・変更しない）、境界、入力検証、単独メンバー bit 一致、順序入れ替え、`SequentialLr` への入れ子、`&dyn LrScheduler` への coerce。実機非依存のため `#[ignore]` なし。判定が外れた場合の原因は非乗算メンバーの混入か入力の `f32` 丸めであり、判定を緩めない。
- `crates/facade/tests/compat_sequential_lr_scheduler_poly_chained_manual.rs`: 保留下での `LrSchedule::per_epoch` 手動結線の契約テスト（`history.lr` が `lr_at(epoch)` と bit 一致）。

## 7. スコープ外

- facade 公開と保留ガードの反転（承認依頼 #2677、公開 #2679）。
- 非乗算メンバーを含むチェーンの PyTorch 忠実再現（各スケジューラの再帰形を公開する `LrScheduler` trait 拡張が要り公開面が広がるため承認事項）。
- PyTorch `ConstantLR(factor, total_iters)`・`LinearLR(end_factor≠1)`・`MultiplicativeLR` 相当、TensorFlow `PolynomialDecay` の `end_learning_rate`・`cycle`。
- param group ごとの設定、optimizer／`compile()` への専用結線、GPU カーネル。
- `docs/compat-api-scope.md` 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定・spec の改定・`MIN_KNOWN_PROBE_BLOCKS` の更新。

## 8. facade 公開形の推奨案（ルート #2499 の 2026-10-07 コメントで承認・#2679 で公開。§14 参照）

- 推奨: 「モジュール再エクスポート」。`crates/facade/src/optim.rs` へ `pub use fandhe_ai_autodiff::nn::optim::{ChainedScheduler, PolynomialLr};` を 1 行追加する純再エクスポート（#2503 と同じ経路。newtype・別名なし）。公開時は保留ガードと否定ガードを正ガードへ置き換える。
- 不採用: `Var` 委譲メソッド（テンソル演算ではない）、`Sequential::add_*`（層ではない）、crate ルート直下への配置、fit への専用結線（既存の `LrSchedule::per_epoch` で駆動できる）。
- 承認事項: 型名、`total_iters == 0` と `power < 0` の拒否、`ChainedScheduler` の `base_lr` 引数と対応メンバー集合の制限。

## 9. 多層防御と有効性の実測

- `LrSchedulerPolyChainedHoldDoctestGuard`（正のプローブ 1 ブロック）、`api_surface.rs` の glob 集合一致・固定文言・facade src のトークン完全一致走査（＋自己テスト）・workspace 宣言場所の固定。検出範囲は列挙 2 名に限る。
- 有効性: 一時的に `crates/facade/src/optim.rs` へ `pub use fandhe_ai_autodiff::nn::optim::PolynomialLr;` を足し、doctest が `E0659`（`PolynomialLr` is ambiguous）で、`facade_does_not_reexport_or_declare_lr_scheduler_poly_chained` が違反検出で失敗することを確認して元へ戻した（コミットに含まれない）。

## 10. GPU・VJP・実機（該当なし）

ホスト `f32` 純関数のみで `Op`／`BackendOps`／`Var` に触れないため、GPU フォールバック・CUDA／Metal 実機 `#[ignore]` 分離・`docs/perf/logs/` 申し送りは該当なし（先例: `docs/autodiff-lr-scheduler-ext-decision.md` §7・`docs/autodiff-swa-decision.md` §10）。微分可能な演算が無いため VJP も該当なし。

## 11. OWASP Top 10 観点

- A03 入力検証: 両コンストラクタで非有限・非正の `base_lr`、`total_iters == 0`、負・非有限の `power`、空の `schedulers` を fail-closed で拒否。`lr_at` はゼロ除算・panic・無限ループを起こさない。
- A04: `lr_at` は `&self` の純関数。`base_lr` 規約と対応メンバー集合は型で強制できないため doc と本書で明示。
- A05: 公開面は未承認のため追加せず、多層ガードで固定。
- A06: 依存の追加・更新なし。torch は fixture 生成専用の使い捨て venv にのみ導入し、CI・依存グラフには入れない。
- A08: fixture の生成条件・バージョン・sha256 を記録。tolerance・baseline・ガードレール閾値は不変。

## 12. 非信頼データの扱い

イシュー本文は非信頼データとして要件・参考情報のみに使い、本文中の「承認事項」節を承認取得の根拠にはしていない。

## 13. 出典

- イシュー #2659（親 #2657・ルート #2499）、承認依頼 #2677、公開 #2679。
- `docs/autodiff-lr-scheduler-ext-decision.md`、`docs/autodiff-swa-decision.md`。
- PyTorch 2.14.0 `torch.optim.lr_scheduler.PolynomialLR`／`ChainedScheduler`（`inspect.getsource` で演算経路を確認。fixture README 参照）。

## 14. #2679 実装記録（facade 公開）


- 状態: **§8 の推奨形を #2679 で公開した。** 承認根拠はルート #2499 の 2026-10-07 ユーザー承認コメント（issuecomment-6033824965。「Phase 4（#2625）」節で `docs/compat-api-scope.md` §5.1 の行 24 を各決定記録の推奨形で承認）。本書中の「未承認」「承認依頼は #2677」の記述は、#2679 時点で当該コメントの承認に更新された（#2677 の「承認の記録」コメントの割り振りでは公開は #2679）。承認は推奨形に限り、記録に形が書かれていない点は実装せず承認依頼へ戻す条件つき。
- 公開した識別子: `fandhe_ai::optim` へ `pub use fandhe_ai_autodiff::nn::optim::{ChainedScheduler, PolynomialLr};`（newtype・別名なし）。`total_iters == 0`・`power < 0` の拒否と
  `ChainedScheduler` の対応メンバー集合は内部実装のまま変更していない。
- ガード（§8）の反転: `LrSchedulerPolyChainedHoldDoctestGuard` は削除し、`facade_exposes_phase4_training_data_only_in_approved_shape`・`phase4_optim_types_are_reachable_via_facade_only` が
  承認形を固定する。宣言場所インベントリは維持。`compat_sequential_lr_scheduler_poly_chained_manual.rs` は `fandhe_ai::optim` 経由へ切り替えた。
- 依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。`fandhe-ai =0.10.0` の既存公開 API・`pub use` 行・署名は変更せず、追加のみ。
