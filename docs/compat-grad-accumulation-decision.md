# 勾配累積（`FitConfig::accumulate_steps`）設計判断記録

イシュー #2180（親 #2131「PyTorch／TF 置き換えの API 網羅（対応表の
行内深掘り）」）。`docs/autodiff-param-groups-decision.md`（#2173）・
`docs/autodiff-ema-decision.md`（#2179）と同型の記録。

## §0 結論

`fit()` 系（`Sequential::fit`／`fit_with_callbacks`／`fit_with_metrics`。
`crates/facade/src/compat/training.rs::run_fit`）へ、`accumulate_steps`
回の backward ごとに 1 回だけ `optimizer.step` → `apply_parameters` する
勾配累積（PyTorch の `.grad +=` を使うミニバッチ相当技法）を実装した。
`FitConfig` へ非公開フィールド `accumulate_steps: u32`（既定 `1`）を
追加し、`run_fit` のバッチループを「1 マイクロステップ目は clone・
2 マイクロステップ目以降は f32 逐次和で累積し、ウィンドウ境界
（`micro == accumulate_steps`）または epoch 末の端数フラッシュで
`optimizer.step` する」構造へ拡張した。`accumulate_steps == 1`
（既定）は既存の非累積経路と **bit 完全一致**する。

**公開ビルダーは追加していない（承認待ち）**: イシュー #2180・親 #2131
のいずれにも所有者の承認コメントはない（着手前に `gh issue view
--json comments` で確認済み）。加えてイシュー本文自身が
`compat::FitConfig` への `accumulate_steps` 追加を「facade 公開面の
拡張」（`docs/compat-api-scope.md` §5 経路 2）として承認事項に挙げて
いる。親 #2131 の「facade 公開面の拡張は設計判断記録 → 承認 → 実装の
2 段」規則（先例 #2171・#2173・#2176・#2178・#2179・#2198）に従い、
`FitConfig::accumulate_steps(n: u32)` 公開ビルダーは追加せず保留する。
現状は `#[cfg(test)] pub(crate) fn FitConfig::
with_accumulate_steps_for_test` 経由でのみ `accumulate_steps` を
変更でき、通常経路（公開 API のみ）では常に `1`（既存挙動）のまま
固定される。

**AMP（`compile_with_amp`）との併用は未実装**: `accumulate_steps > 1`
と AMP の組み合わせは、引数検査で `InvalidArgument`（fail-closed）に
拒否する（§3「AMP との関係」節）。

## §1 背景・目的

要件（Plan フェーズで構造化。原文はイシュー #2180 本文）:

- R1: `FitConfig` に `accumulate_steps: u32` を持たせる。既定値は `1`
  で、既存の挙動を変えない
- R2: fit ループは `accumulate_steps` 回の backward ごとに 1 回
  `optimizer.step` → `apply_parameters` を呼ぶ
- R3: `accumulate_steps = 1` は従来の fit と数値が同一（bit 完全一致）
  であること
- R4: 大バッチ相当の学習が、通常の小バッチより安定して収束する傾向を
  定性的に確認する

**契約（不変条件）**: tolerance・baseline・`Cargo.toml` 依存・ガード
レール閾値・`docs/spec/` は変更していない。`fandhe-ai =0.9.0` の公開
API を破壊していない。新規 `unsafe` は入れていない。新規 `Op`／
`BackendOps`／カーネルは追加していない（累積はホスト側の f32 加算の
みで、CPU／CUDA／Metal いずれのバックエンドにも一切触れない）。

**スコープ外**: 累積回数による勾配正規化（`N` で割る処理）・
`DeviceParamStore` 常駐経路（`step_device_param_store` 系）での勾配
累積・GPU 専用の累積カーネル・公開ビルダー（本 doc §5 参照）。

正規化を行わない代わりに呼び出し元が学習率を `1/N` にすることで大
バッチ相当に近づけられるのは、**SGD（momentum なし）かつ全ウィンドウ
（epoch 末の端数ウィンドウを含む）のマイクロバッチ数・サイズが揃う
構成に限られる**限定的な等価性であり、一般には成立しない（詳細と
不成立条件は §3 参照）。

## §2 設計

### 2.1 累積の数値形式

`crates/facade/src/compat/training.rs::accumulate_grads_into` が
`acc[i] += grads[i]` を要素ごとの **f32 逐次和**で行う（PyTorch
`.grad +=` の意味論・[`Tape::backward_accumulate`] が使う
`vjp_elementwise_add`〈f32 の `a + b`〉と同じ数値形式。
`docs/autodiff-retain-graph-accumulate-decision.md`）。正規化（`N` で
割る処理）は一切行わない。

`.claude/rules/coding-rust.md` の「勾配の長軸縮約は `f64` アキュムレー
タ」原則は、カーネル内の行方向縮約（dw・bias・rstd 等、1 要素あたり
多数の項を畳み込む縮約）が対象である。本機能は縮約済みの勾配テンソル
を高々 `accumulate_steps` 回（実用上小さい回数）だけ足すだけのホスト
側加算のため対象外——新たな `f64` アキュムレータ契約は導入していない。

1 マイクロステップ目は **clone のみ**（算術を一切行わない）にする
ことで、`accumulate_steps == 1` のとき常に 1 マイクロステップ目で
ウィンドウ境界に到達し、加算が一度も起きないため既存の非累積経路と
bit 完全一致する（R3）。累積結果は「新しい `Vec` を組み立て終えてから
`*acc` へ書き戻す」原子的な更新にしており、shape 不一致等の途中失敗で
`acc` が半端な状態のまま残らない。

### 2.2 ウィンドウ境界と epoch 末フラッシュ

`micro == accumulate_steps`（ウィンドウが埋まった）、または epoch 末で
`micro > 0`（`accumulate_steps` に満たない端数ウィンドウが残っている。
例: `N=3` でバッチ数 7 の最後の 1 バッチ）のいずれかで
`optimizer.step` → `apply_parameters` を実行し、`acc`／`micro` を
リセットする。

epoch 末に必ずフラッシュする理由は 2 つ:

1. validation・callbacks・`ModelCheckpoint`／`EarlyStopping` が毎
   epoch、更新済みのパラメータを見られるようにするため
2. `fit(1) + fit(1) == fit(2)`（既存契約）を維持するため——
   `acc`／`micro` は fit 呼び出しをまたいで持ち越さない

### 2.3 AMP との関係

推奨方式（窓単位で `scale_loss`／`unscale`／`scaler.update` を揃える）
は、`amp` フィールドの借用と累積バッファの可変借用が同時に必要になる
実装上の複雑さに対して本イシューの価値（累積 1 機能）が見合わないと
判断し、**代替のfail-closed方式**を採用した: `accumulate_steps > 1`
かつ `compiled.amp.is_some()` を引数検査段階（モード変更・パラメータ
更新より前）で `InvalidArgument` として拒否する
（`fit_with_callbacks_named` 内。既存の他引数検査と同じ位置）。

この結果、AMP 経路（`run_fit` 内の `if let Some(amp) = compiled.amp
...` 分岐）は本イシューで一切変更していない——`accumulate_steps == 1`
のときと常に同じ演算列（マイクロバッチごとに `scale_loss → backward →
unscale → step/skip → scaler.update`）のまま、既存の bit 完全一致
テスト（`compat_sequential_fit_amp.rs`）は無変更で green。

将来 AMP + 勾配累積を実装する場合は、窓の境界（フラッシュを含む）で
`unscale` → `should_skip_step` 判定 → 非 skip なら `optimizer.step`、
窓ごとにちょうど 1 回 `scaler.update` を呼ぶ設計（PyTorch と同じ順序）
を別イシューで検討する。

## §3 PyTorch との差分

- PyTorch は勾配累積を「ユーザーが `loss.backward()` を複数回呼び、
  `N` 回ごとに `optimizer.step()` する」手動パターンとして提供し、
  専用 API は持たない。本実装は Keras 風 `fit()` の内部でこのパターン
  を隠蔽する（`accumulate_steps` の 1 パラメータのみで完結する）
- PyTorch の一般的な実践は `loss = loss / N` で事前に正規化してから
  累積するが、本実装は正規化を行わない（§1「スコープ外」参照）。
  呼び出し元が学習率を `1/N` にすることで数学的に厳密に等価にできる
  のは **SGD（momentum なし）かつ全ウィンドウ（epoch 末の端数
  ウィンドウを含む）のマイクロバッチ数・サイズが揃う構成に限る**
  （§4 の R3 相当テストはこの条件を満たす構成〈`batch_size=B・
  accumulate_steps=N・lr=lr0/N` と `batch_size=N*B・accumulate_steps=1・
  lr=lr0` の比較〉のみを検証しており、バッチ数が `accumulate_steps`
  で割り切れず端数ウィンドウがそのまま flush される構成では、端数
  ウィンドウのマイクロバッチ数が他のウィンドウと異なるため学習率の
  単純な `1/N` 変換だけでは大バッチ相当と等価にならない）。
  この等価性は勾配が線形に合成される SGD（momentum なし）の場合に
  限られ、モーメント・二次モーメント等の状態を持つ optimizer
  （Adam・AdamW・RmsProp 等。`accumulate_steps` はこれらの optimizer
  でも受け付ける）では、累積ウィンドウごとに 1 回だけ `optimizer.step`
  が呼ばれる一方で大バッチ相当の構成でも同じ回数しか呼ばれないため
  勾配自体は一致しても内部状態の更新順序・丸めが異なり、学習率の
  変換だけでは一般に等価にならない

## §4 数値一致

- `accumulate_steps == 1`（既定）は既存の `fit`／`fit_with_callbacks`／
  `fit_with_metrics` と **bit 完全一致**（`crates/facade/src/compat/
  training.rs::accumulate_tests::
  accumulate_steps_one_matches_default_fit_bit_exact` で検証）
- `accumulate_steps == 3`（バッチ数 7・割り切れない構成）は独立に組んだ
  手動累積ループ（bind → forward → loss_for → backward →
  trainable_grads → f32 逐次和 → 境界 step → epoch 末 flush）と
  **bit 完全一致**（`accumulate_steps_three_matches_manual_window_
  loop_bit_exact`）
- **大バッチ等価**（R3 と別軸の数学的性質。SGD・momentum なし）:
  `batch_size=B・accumulate_steps=N・lr=lr0/N` の構成と
  `batch_size=N*B・accumulate_steps=1・lr=lr0` の構成は、`Reduction::
  Mean` の MSE 勾配の線形性から最終パラメータが数学的に厳密に一致する
  （§2.1 の f32 逐次和と大バッチ側の mean 縮約とで丸め順序が異なる
  のみ）。**既存の REQ-2 統一複合判定の定数**（`RELATIVE_TOLERANCE`＝
  `1e-3`（相対誤差）／`ABSOLUTE_RESCUE_THRESHOLD`＝`1e-5`（絶対誤差）。
  `fandhe_ai_backend_cpu::parity::assert_parity` が内部で使う既存定数を
  そのまま再利用するだけで新設・緩和ではない）で判定
  （`accumulate_matches_large_batch_equivalent_within_req2_tolerance`）
- `accumulate_steps == 0` は `InvalidArgument`（fail-closed）
  （`accumulate_steps_zero_is_rejected`）
- `accumulate_steps > 1` と AMP の併用は `InvalidArgument`
  （fail-closed）（`accumulate_steps_gt_one_rejected_with_amp`）

CUDA／Metal 固有の処理は追加していないため、実機 parity の
`#[ignore]` テストや申し送りは発生しない。

### R4（定性確認）: 大バッチ相当の学習の安定性

同一の合成回帰データ（64 サンプル・固定シード）・SGD（momentum なし）
30 epoch で、基準（`batch_size=1`・`accumulate_steps=1`・`lr=0.05`）と
累積（`batch_size=1`・`accumulate_steps=8`・`lr=0.05/8`）を比較した
（ローカル worktree の一時テストで測定・値のみ記録し、フレーキーな
閾値化はしない。§1 R4「定性的に確認」の要件どおり）。

| 構成 | epoch 間で loss が上がった回数（30 epoch 中） |
|---|---|
| 基準（`accumulate_steps=1`） | 2 |
| 累積（`accumulate_steps=8`） | 0 |

累積側は 30 epoch を通じて単調減少（1 度も loss が上がらない）であり、
基準側より安定して収束する傾向を定性的に確認できた（R4）。

## §5 承認事項（未承認のため保留）

facade（`compat::FitConfig`）公開面の拡張は次が未承認。
`crates/facade/src/lib.rs::GradAccumulationHoldDoctestGuard`（正の
プローブ doctest）・`crates/facade/tests/api_surface.rs` の 3 テスト
（`grad_accumulation_hold_doctest_globs_all_pub_modules`・
`grad_accumulation_hold_doctest_probe_body_matches_fixed_contract`・
`facade_does_not_declare_fit_config_accumulate_steps`）が機械的に
固定する:

1. `FitConfig::accumulate_steps(mut self, n: u32) -> Self`（ビルダー。
   `shuffle`／`drop_last` と同型）の公開
2. 承認後の作業手順: ビルダー追加・テストの `#[cfg(test)] mod
   accumulate_tests`（`crates/facade/src/compat/training.rs`）から
   `crates/facade/tests/compat_sequential_accumulate.rs`（外部統合
   テストクレート）への移設・`with_accumulate_steps_for_test` の削除・
   `GradAccumulationHoldDoctestGuard` と対応する 3 テストの削除（正の
   ガード・facade テストへ置き換え）

承認を得た日が来たら、上記を実施する。

## §6 スコープ外（out-of-scope-tracking）

- 公開ビルダー `FitConfig::accumulate_steps`（§5 参照）
- 累積回数による勾配正規化（`N` で割る処理）
- `DeviceParamStore` 常駐経路（`step_device_param_store` 系）での
  勾配累積
- GPU 専用の累積カーネル
- AMP（`compile_with_amp`）と `accumulate_steps > 1` の併用（§2.3）

新規 Issue の起票はユーザー承認後に行う。

## §7 検証コマンドと結果

```
cargo fmt --all -- --check
cargo clippy -p fandhe-ai --all-targets --no-deps -- -D warnings
cargo test -p fandhe-ai --lib compat::training::accumulate_tests
cargo test -p fandhe-ai --test compat_sequential_fit \
  --test compat_sequential_callbacks --test compat_sequential_metrics \
  --test compat_sequential_fit_amp --test compat_sequential_fit_optimizers
cargo test -p fandhe-ai --test api_surface accumulate_steps
cargo test -p fandhe-ai --doc
cargo test -p fandhe-ai
cargo build --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
```

全て green（2026-09-26 実測。ローカル worktree）。`cargo clippy -p
fandhe-ai --all-targets`（`--no-deps` なし・`--all-features` 付き）は
このローカル環境では `fandhe-ai-backend-cuda` の一部 `dead_code` lint
が fail する（origin/main でも再現することを確認済みの既知事象で本
PR の変更とは無関係）。原因（CUDA toolkit 非搭載固有か、他の環境差か）
は未特定のまま。`--no-deps` を付けると facade（`fandhe-ai`）自身は
依存クレートを素の `rustc` で扱い `-D warnings` の対象から外れるため
green になる。CI（`build-no-cuda-toolkit`／通常ホステッド環境）でも
同事象が再現するかは別途確認が必要（本イシューのスコープ外）。
