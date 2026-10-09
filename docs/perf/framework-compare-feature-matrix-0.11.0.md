# フレームワーク横並びスコアボード「役割・機能の対応表」v0.11.0 再監査

イシュー #2681（親 #2680・ルート #2499）。フレームワーク横並びスコアボード「役割・機能の対応表」の
fandhe-ai 列で 0.10.0 時点に「部分的」だった 9 行（dtype／自動微分／演算の範囲／NN 層・モデル構築／
最適化・学習ループ／バックエンド・ハード／相互運用／事前学習済みモデル／推論・サービング）を、
crates.io に公開済みの `fandhe-ai =0.11.0`（タグ `v0.11.0`）の facade 公開面基準で再判定し、
各行の根拠を「公開／未公開／対象外／保留／実機 parity」の 5 区分に分けて記録する。
0.9.0 版（`docs/perf/framework-compare-feature-matrix-0.9.0.md`・イシュー #1938）と同じ節構成である。

スコアボード（Artifact・`docs/perf/logs/framework-compare-0.10.0-remeasure/scoreboard/body_0100.html`）への
反映は後続の #2682 の担当で、本書は §3 の判定を確定内容として渡すだけである（Artifact・生成スクリプトは変更しない）。

## 1. 基準面の証明

判定基準は HEAD ではなく、crates.io に公開済みの `fandhe-ai =0.11.0`（= タグ `v0.11.0`）の
`crates/facade` の公開面とする。本書の `path:line` は、すべて `git show v0.11.0:<path>`（タグ時点の内容）で
再解決できる行番号である。

- タグ `v0.11.0` は `6b14fdb4`（`git rev-parse v0.11.0^{commit}`）。crates.io API の実測で
  `fandhe-ai 0.11.0` の `created_at` は 2026-10-09T13:01:08Z（yanked なし）。
- 本書作成時点の `origin/main`（公開記録の追記 #2960 を含む）とタグの間で、`crates/` 配下に差分はない
  （差分は `CLAUDE.md`・`README.md`・`docs/crates-io-publishing-order.md` のみ）。

```
$ git diff --stat v0.11.0..origin/main -- crates
（出力なし）
```

- 実機 parity の測定コミットはタグではない。§5 の 6 件の実測は main `8bbeb874` で行われており、
  `git diff --stat 8bbeb874..v0.11.0 -- crates` は 12 files changed（417 insertions・111 deletions）である。
  内訳は (a) `Cargo.toml` 6 本のバージョン更新（#2958）、(b) `jvp`／`jacfwd` の facade 公開（#2957）に伴う
  `crates/facade/src/lib.rs`・`crates/facade/tests/api_surface.rs`・新規 `crates/facade/tests/functional_transforms_jvp_facade.rs`
  と、`crates/autodiff/src/functional_ops.rs` の可視性変更（`pub(crate)` → `pub`。数値経路は不変）・
  テスト側の追従 2 本。したがって `jvp`／`jacfwd` の内部実装の実機 parity は測定済みだが、
  facade `Tape::jvp`／`Tape::jacfwd` の委譲そのものは実機未測定である（§3 の自動微分行）。

## 2. 判定規則

- **到達性**: 0.9.0 版 §2 を踏襲する。`pub use`／`pub mod` から到達できるものだけを公開扱いとし、
  `Var` は `fandhe_ai_autodiff::Var` の再エクスポート（`crates/facade/src/lib.rs:245`）なので `Var` の `pub fn` も公開扱いとする。
  内部クレートのみの機能は「未公開」とする。
- **行判定値**: ある／部分的／未公開（リポ内）／ない の 4 値。1 行内で項目により判定が割れる場合と、
  「未公開」「保留」が 1 つでも残る場合は「部分的」とする（機械的に決める）。「対象外」（承認済みの非目標・Won't）は
  行を下げない（0.9.0 版の「リポ内／対象外」の扱いと同じ）。
- **根拠セルの 5 区分**（必ずこの順に書き、空なら「なし」と書く。区分を混ぜない）
  - `公開:` Phase 1〜4・7〜9 で公開した識別子と実装イシュー（タグ時点の `path:line`）
  - `未公開:` 内部実装はあるが facade から到達できないもの（保留ガード名と `path:line`）
  - `対象外:` Phase 5 で承認された非目標・Won't とその決定記録（承認の出所は本節末尾のコメント 4 件）
  - `保留:` 所有者コメントで保留が続くもの（コメント ID を明記）
  - `実機 parity:` 下記 4 状態のいずれかと README パス
- **実機 parity の 4 状態**（`docs/perf/logs/*-<2500 以上>/README.md` の「実測記入欄」「結果記入欄」「記入欄」の行で判定する。見出しの語だけでは判定しない）
  - **実測済み**: 記入欄に日付と pass/fail が入っている。日付・README・測定コミットを書き、タグとの差分の有無を併記する（「タグで実測した」とは書かない）。
  - **実機未検証（#2683 未完了）**: #2629（f16／bf16 opt-in の実機 parity）・#2672（GPU 線形代数）・#2607（int8 量子化）に当たるものだけ。
  - **実機未実測（#2683 対象外）**: 記入欄が未実測のまま。README パスを書き、検証済みとは書かない。
  - **判定対象外**: README が REQ-2 の判定対象外と明記している（`crates/facade/src/lib.rs:1256` の `Tape::custom` は host 上の `Tensor<f32>` で完結する）。
- **0.10.0 時点列**: `body_0100.html` の fandhe-ai 0.10.0 列の記載を出典として要約する（原文は同ファイル）。
- **承認の記述**: 「承認」は #2499 のリポジトリ所有者コメント 4 件（`#issuecomment-6033824965`＝2026-10-07、
  `-6052732061`＝2026-10-08、`-6067263650`＝2026-10-08、`-6079384681`＝2026-10-09）と各イシュー番号の事実に限って書く。
  コメントの本文は転記せず要約する。

## 3. 対応表

セル内の `A§n` は `docs/compat-api-scope.md` の節、`D:` は `docs/` 配下の決定記録を指す。Phase 4 の公開形 30 行の全体は
`docs/compat-api-scope.md:554`（§5.1）を正とし、ここでは再掲しない。

| 行 | 0.10.0 時点（`body_0100.html`） | 0.11.0 判定 | 根拠（公開／未公開／対象外／保留／実機 parity） |
|---|---|---|---|
| dtype | 部分的（autograd は f32。f64／f16／bf16 は勾配なしの 8 演算。AMP は 3 層。TF32 は opt-in） | 部分的 | **公開:** f64 の自動微分 `TapeF64`／`VarF64`（演算は add／mul／div／pow／matmul／sum／mean／max。#2599・#2834。`crates/facade/src/lib.rs:1581`・`crates/facade/src/lib.rs:1612`）、低精度 forward の opt-in `Var::matmul_low_precision` ほか 6 本（#2628・#2678。`crates/autodiff/src/var.rs:3629`）、`CastDType`（`crates/facade/src/lib.rs:379`）・`TypedOps`（`crates/facade/src/lib.rs:391`）・`Tape::typed_ops_f64`（`crates/facade/src/lib.rs:1471`）・`compile_with_amp`（`crates/facade/src/compat/training.rs:2060`）。<br>**未公開:** f16／bf16 を値型とする `Tape`／`Var` は無い（低精度は選定 6 演算と AMP 対象 3 層の compute_dtype opt-in に限る）。`Var` の低精度委譲以外の経路は `VarLowPrecisionOpsHoldDoctestGuard`（`crates/facade/src/lib.rs:5315`）が固定する。<br>**対象外:** sparse／complex は #2616 案 (ii)（`#issuecomment-6033824965` で承認。「引き続き対象外」から外して除外事項〈Won't・条件付き〉へ移す spec 改定。実装は含めない。D: `tensor-core-sparse-complex-decision.md` §15.0・§15.3）。<br>**保留:** `jvp` の `VarF64`・f16 対応（`#issuecomment-6079384681`）。<br>**実機 parity:** f64 は実機未実測（`docs/perf/logs/f64-autograd-facade-2599/README.md`）。f16／bf16 opt-in は**実機未検証**（#2629 未完了・#2683。`docs/perf/logs/low-precision-ops-2628/README.md`）。int8 は**実機未検証**（#2607 未完了・#2608〜#2610 open。D: `backend-int8-quantization-decision.md` §3.1 は段階 0＝非対応） |
| 自動微分 | 部分的（動的テープ・1 階のみ。高階微分・カスタム VJP は内部実装のみで未公開） | 部分的 | **公開:** 高階微分 `Tape::backward_create_graph`（#2545。`crates/facade/src/lib.rs:737`）、`CustomFunction`（`crates/facade/src/lib.rs:252`）・`Tape::custom`（#2549。`crates/facade/src/lib.rs:1256`）、hooks `Tape::register_backward_hook`（#2587。`crates/facade/src/lib.rs:1314`）、`Tape::jacobian`／`hessian`（#2678。`crates/facade/src/lib.rs:820`・`crates/facade/src/lib.rs:850`）、`Tape::gradcheck`（#2847。`crates/facade/src/lib.rs:1166`）・`Tape::backward_detect_anomaly`（`crates/facade/src/lib.rs:1133`）、`Tape::vjp`／`hvp`／`vmap`（#2931。`crates/facade/src/lib.rs:900`・`crates/facade/src/lib.rs:952`・`crates/facade/src/lib.rs:1102`）、`Tape::jvp`／`jacfwd`（#2956。`crates/facade/src/lib.rs:1020`・`crates/facade/src/lib.rs:1057`）、決定論モード `set_deterministic`（`crates/facade/src/lib.rs:1787`）。0.10.0 から継続: `Tape::var_no_grad`（`crates/facade/src/lib.rs:440`）・`backward_accumulate`（`crates/facade/src/lib.rs:459`）・`Var::detach`（`crates/autodiff/src/var.rs:278`）・`Var::checkpoint_from`（`crates/autodiff/src/var.rs:2498`）。<br>**未公開:** `Var::custom`／`Sequential::add_custom`／`Tape::add_custom`（`VarCustomHoldDoctestGuard`。`crates/facade/src/lib.rs:2556`）、関数型 AD の `Tape` 以外の経路（`FunctionalTransformsHoldDoctestGuard`。`crates/facade/src/lib.rs:7855`）、gradcheck のモジュール名・裸の自由関数（`GradcheckAnomalyHoldDoctestGuard`。`crates/facade/src/lib.rs:7711`）。<br>**対象外:** ネイティブ forward-mode の JVP 規則とバッチ規則型 vmap（#2617 案 C の段階 2。`#issuecomment-6033824965` で案 C を承認。D: `autodiff-forward-mode-vmap-spec-proposal.md` §0・§3）。`jvp`／`jacfwd` は `Op::supports_create_graph()` が真の Op に限り、非対象 Op は型付きエラー（`#issuecomment-6079384681`）。<br>**保留:** 関数型 AD の論点 5（`supports_create_graph` の対象拡張）・論点 6（`vmap(grad)` の公開形）（`#issuecomment-6067263650`）、`jvp` の複数入力・`VarF64`・f16・微分可能な `jvp`（`#issuecomment-6079384681`）。<br>**実機 parity:** 実測済み（2026-10-09・main `8bbeb874`＝タグより前）: `Tape::vjp`／`hvp`／`vmap`（`docs/perf/logs/functional-transforms-2881/README.md`・`docs/perf/logs/functional-transforms-facade-2931/README.md`）、`jvp`／`jacfwd` の内部実装（`docs/perf/logs/functional-transforms-jvp-2942/README.md`）、`Tape::gradcheck`（`docs/perf/logs/tape-gradcheck-facade-2847/README.md`）。いずれも GB10・M4 Max とも pass。facade `Tape::jvp`／`jacfwd` の委譲はタグで追加され実機未測定。実機未実測: `create-graph-facade-2545`・`jacobian-hessian-2670`・`gradcheck-anomaly-2671`。判定対象外: `facade-custom-function-2549`（host 上で完結） |
| 演算の範囲 | 部分的（`Var` の演算約 110。線形代数は CPU のみ・CUDA／Metal は Unsupported） | 部分的 | **公開:** Phase 1 の `Var` 委譲（#2510〜#2519）に加え、Phase 4 の演算（A§5.1 行 1〜11・17〜20。#2678・#2854）: FFT（`Var::rfft`・`crates/autodiff/src/var.rs:4960`）、順序統計（`Var::median`・`crates/autodiff/src/var.rs:5204`）、テンソル積（`Var::kron`・`crates/autodiff/src/var.rs:5384`）、`Var::pad_with_mode`（`crates/autodiff/src/var.rs:5425`）、3D プーリング・ConvTranspose3d（`Var::max_pool3d`・`Var::conv_transpose3d`。#2850。`crates/autodiff/src/var.rs:5442`・`crates/autodiff/src/var.rs:5483`）、`Var::local_response_norm`（#2851。`crates/autodiff/src/var.rs:5605`）、オプション型を取る損失（`Var::bce_with_logits_loss_with`。#2854。`crates/autodiff/src/var.rs:4801`）、`Tape::bincount`（`crates/facade/src/lib.rs:762`）、logical 3 本（`logical_and`。`crates/facade/src/lib.rs:1988`）。<br>**未公開:** モジュール `elementwise_loss_ops`／`margin_focal_loss_ops` の再エクスポートと `Tape`／`Tensor` 上の同名メソッド（`crates/facade/src/lib.rs:7053`・`crates/facade/src/lib.rs:7182`）、`norm_except_dim`・`fold_ops` 等のモジュール再エクスポート（A§5.1。いずれも名前の配置の未承認で、演算そのものは `Var` から使える）。<br>**対象外:** sparse／complex の演算（dtype 行と同じ #2616 案 (ii)）。<br>**保留:** なし（演算の保留はない。層化は NN 層行）。<br>**実機 parity:** GPU 線形代数は**実機未検証**（#2672 open・#2673〜#2676 open・#2683。CUDA／Metal の `linalg_cholesky` は `Unsupported` を返す: `crates/backend-cpu/tests/backend_ops_dispatch.rs:264`・`crates/backend-cpu/tests/backend_ops_dispatch.rs:375`）。Phase 4 の演算 README は実機未実測（§5 の付録表） |
| NN 層・モデル構築 | 部分的（`Sequential` に 30 種の `add_*`＋`add_module`。Transformer デコーダ・GroupNorm 等は未公開） | 部分的 | **公開:** `compat::Sequential::add_*` は 31 → 65（`grep -cE '^\s*pub fn add_'` で v0.10.0 と v0.11.0 を比較）。追加分の例: `add_conv3d`（`crates/facade/src/compat/sequential.rs:982`）・`add_group_norm`（`crates/facade/src/compat/sequential.rs:1157`）・`add_transformer_encoder`（`crates/facade/src/compat/sequential.rs:1446`）・`add_transformer_decoder_layer`（`crates/facade/src/compat/sequential.rs:1500`）・`add_transformer`（`crates/facade/src/compat/sequential.rs:1562`）・`add_selu`（`crates/facade/src/compat/sequential.rs:667`）。Functional API `FunctionalBuilder`（#2665〜#2667・#2679。`crates/facade/src/compat/mod.rs:88`）、`nn::Transformer`（#2532・#2533。`crates/facade/src/nn/mod.rs:50`）、`nn::loss` の損失構造体（#2602。`crates/facade/src/nn/loss.rs:111`）、可変長系列 `nn::rnn`（#2679。`crates/facade/src/nn/rnn.rs:143`）、`KvCache`（`crates/facade/src/nn/kv_cache.rs:59`）、`nn::init`（`crates/facade/src/nn/mod.rs:32`）。<br>**未公開:** 3D プーリング・ConvTranspose3d・MaxUnpool・Fold／Unfold・LRN・weight_norm／spectral_norm の層型と `add_*`（`Pool3dOpsHoldDoctestGuard`〈`crates/facade/src/lib.rs:5945`〉・`ConvTranspose3dMaxUnpoolHoldDoctestGuard`〈`crates/facade/src/lib.rs:6118`〉・`FoldUnfoldHoldDoctestGuard`〈`crates/facade/src/lib.rs:6242`〉・`LrnWeightReparamHoldDoctestGuard`〈`crates/facade/src/lib.rs:6420`〉）、`Upsample`／`ZeroPad2d`／`Identity` の型名（`SpatialLayersHoldDoctestGuard`。`crates/facade/src/lib.rs:3193`）、参照モデル Mlp／LeNet／ResNet／Transformer は `crates/facade/examples/models/` の利用者コードのみ（公開経路は未決。D: `reference-models-decision.md` §11。#2541 は GitHub 上 closed だが facade に公開名は無い）、活性化 9 層は `save_model` が `UnsupportedModel` で拒否（A§5 の適用記録〈#2679〉）。<br>**対象外:** なし。<br>**保留:** §5.1 行 12〜15 の層化と行 15 の結線方式（`#issuecomment-6052732061`・`#issuecomment-6067263650`・`#issuecomment-6079384681`）。<br>**実機 parity:** 実機未実測: `compat-sequential-*`（9 件）・`mha-config-sequential-2530`・`transformer-decoder-sequential-2532`・`transformer-sequential-2533`・`functional-graph-2665`・`functional-fit-2667`・`merge-ops-2666`・`packed-sequence-2647`（付録表） |
| 最適化・学習ループ | 部分的（SGD・Adam 系・L-BFGS・LR scheduler 7 種・fit／callbacks・AMP。param groups・EMA・Adadelta／NAdam 等・マルチワーカー DataLoader は未公開） | 部分的 | **公開:** optimizer の追加 `Adadelta`（`crates/facade/src/optim.rs:515`）・`Adafactor`（`crates/facade/src/optim.rs:516`）・`Adamax`（`crates/facade/src/optim.rs:520`）・`Asgd`（`crates/facade/src/optim.rs:521`）・`Lion`（`crates/facade/src/optim.rs:529`）・`Rprop`（`crates/facade/src/optim.rs:530`）・`NAdam`（`crates/facade/src/optim.rs:538`）・`RAdam`（`crates/facade/src/optim.rs:548`）、scheduler `PolynomialLr`（`crates/facade/src/optim.rs:522`）・`CyclicLr`（`crates/facade/src/optim.rs:526`）、param groups `ParamGroup`（#2553。`crates/facade/src/optim.rs:546`）・`compile_with_param_groups`（`crates/facade/src/compat/training.rs:1993`）、`OptimizerStateDict`（#2556。`crates/facade/src/optim.rs:543`）、EMA `ExponentialMovingAverage`（#2560。`crates/facade/src/optim.rs:513`）・`Callback::Ema`（`crates/facade/src/compat/callbacks.rs:1342`）、SWA `AveragedModel`（#2679。`crates/facade/src/optim.rs:514`）、fit 拡張 `fit_with_weights`（`crates/facade/src/compat/training.rs:2631`）・`fit_with_train_step`（`crates/facade/src/compat/training.rs:2533`）、データ `PrefetchDataLoader`（#2603。`crates/facade/src/data.rs:121`）・`ConcatDataset`（`crates/facade/src/data.rs:117`）・`IterableDataset`（`crates/facade/src/data.rs:120`）・`random_split`（`crates/facade/src/data.rs:124`）。<br>**未公開:** SWA の `fit` 結線（`SwaHoldDoctestGuard`。`crates/facade/src/lib.rs:3982`）、`FitConfig`／`Sequential` へ EMA を足す形（`EmaHoldDoctestGuard`。`crates/facade/src/lib.rs:3887`）。<br>**対象外:** なし。<br>**保留:** なし（EMA と `Monitor::Loss` の併用は `#issuecomment-6052732061` 項 2 で承認済み・#2862 で実装）。<br>**実機 parity:** 実機未実測: `dataset-compose-2661`・`iterable-batch-sampler-2662`（host ユーティリティ。記入欄は round-trip の確認） |
| バックエンド・ハード | 部分的（CPU／CUDA／Metal を cfg 切替。単一デバイスのみ・分散なし・AMD なし） | 部分的 | **公開:** `tape_for`（`crates/facade/src/lib.rs:1696`）・`available_devices`（`crates/facade/src/lib.rs:1735`）・`set_cuda_gemm_precision`（`crates/facade/src/lib.rs:2217`）。0.10.0 から変化なし。<br>**未公開:** facade に `nccl`／DDP／all_reduce の公開面は無い（`git grep nccl v0.11.0 -- crates/facade/src Cargo.toml` は出力なし）。<br>**対象外:** FSDP・ZeRO 系・tensor／pipeline／model parallel は Won't（D: `ddp-grade-up-conditions.md` §3a）。DDP は #2612 の §5 項 1（§4 の格上げ条件表提案の spec への起票）を `#issuecomment-6033824965` で承認し、起票の実施は `#issuecomment-6052732061` 項 1 で指示された。条件表が定める実装は保留（同項 4）で、`nccl` feature の追加（#2613）も承認されていない。<br>**保留:** cudarc の `nccl` feature（#2613）・ROCm（#2614）（`#issuecomment-6033824965`・`-6067263650`・`-6079384681`）。<br>**実機 parity:** int8 量子化は**実機未検証**（#2607 未完了・#2683。コード側に量子化経路は無く、D: `backend-int8-quantization-decision.md` §3.1 が段階 0）。DDP・ROCm は実機検証の対象になっていない |
| 相互運用 | 部分的（ONNX import／export・safetensors・`save_model`。npy／npz は内部クレートのみで未公開） | 部分的 | **公開:** ONNX `OnnxModel`（`crates/facade/src/interop/onnx.rs:228`）・`from_bytes`（`crates/facade/src/interop/onnx.rs:237`）・`run`（`crates/facade/src/interop/onnx.rs:363`）・`to_bytes`（`crates/facade/src/interop/onnx.rs:402`）・`OnnxExportOptions`（`crates/facade/src/interop/onnx.rs:432`）、safetensors `load_safetensors_f32`（`crates/facade/src/interop/safetensors.rs:95`）・`save_safetensors_f32`（`crates/facade/src/interop/safetensors.rs:98`）、**npy／npz `load_npy`（`crates/facade/src/interop/npy.rs:72`）・`save_npz`（`crates/facade/src/interop/npy.rs:73`）が 0.10.0 の「未公開」から公開済みに変わった**（#2590・#2831。`#issuecomment-6033824965`）、`save_model`／`load_model`（`crates/facade/src/compat/model_io.rs:268`・`crates/facade/src/compat/model_io.rs:285`）、Functional モデルの保存（`crates/facade/src/compat/mod.rs:90`）。<br>**未公開:** ONNX export は import 済みモデルの往復用ラッパーに限る（`crates/facade/src/lib.rs:174` の `interop` の doc）、活性化 9 層の `save_model`（NN 層行）、`Tensor` への `load_npy` 等の inherent メソッド（`NpyIoHoldDoctestGuard`。`crates/facade/src/lib.rs:4258`）。<br>**対象外:** Python バインディングと TF 系形式（SavedModel・TFLite・Keras H5）は #2623 案 D（非目標を維持。`#issuecomment-6033824965` で承認。D: `python-binding-tf-format-non-target-spec-proposal.md` §0・§3b）。<br>**保留:** なし。<br>**実機 parity:** 該当なし（host 側 I/O） |
| 事前学習済みモデル | 部分的（同梱重み・ダウンロード・ハブはない。ONNX／safetensors の取り込みと読み取り専用の `ModelRegistry`） | 部分的 | **公開:** `ModelRegistry`（`crates/facade/src/model.rs:368`）・`ModelRegistry::load`（`crates/facade/src/model.rs:554`）。ローカル配置済みの重みを読む読み取り専用の入口で、0.10.0 から変化なし。取り込み手段は相互運用行（ONNX・safetensors・npy／npz）。<br>**未公開:** 参照モデル（Mlp／LeNet／ResNet／Transformer）の facade 公開（NN 層行と同じ。D: `reference-models-decision.md` §11）、重みの同梱・ダウンロード・ハブ連携。<br>**対象外:** なし。<br>**保留:** HTTP／TLS 依存（#2621。依存の承認を保留し、workspace 外での検証だけを承認・実施済み〈PR #2814〉）と Hugging Face Hub 連携（#2622）（`#issuecomment-6033824965`・`-6067263650`・`-6079384681`）。<br>**実機 parity:** 該当なし |
| 推論・サービング | 部分的（`predict`／`predict_resident`・ONNX の `run`。`generate()`・KV キャッシュは未公開。量子化・トークナイザ・グラフコンパイルなし） | 部分的 | **公開:** `Sequential::predict`（`crates/facade/src/compat/sequential.rs:1939`）・`predict_batches`（#2582。`crates/facade/src/compat/sequential.rs:2163`）・`predict_resident`（`crates/facade/src/compat/sequential.rs:3091`）、`PhaseMetrics`（`crates/facade/src/inference/mod.rs:231`）、`generate`／`AutoregressiveModel`（#2575。`crates/facade/src/inference/mod.rs:240`）、`KvCache`（#2579。`crates/facade/src/nn/kv_cache.rs:59`）、`BatchScheduler`（#2934。`crates/facade/src/inference/mod.rs:248`）・`generate_speculative`（greedy・B = 1。`crates/facade/src/inference/mod.rs:252`）、語彙 lookup 型のテキスト変換 `TextVectorization`（#2937。`crates/facade/src/text/mod.rs:111`）。<br>**未公開:** サンプリング版 speculative・`KvCache` の書き換え経路・`kv_rewind`・`inference` 配下モジュール自体の再エクスポート（否定プローブで固定。`docs/compat-api-scope.md` の #2934 適用記録）、内部状態保持型モデルは契約の対象外で、facade だけでは `caches` を正しく進めるモデルを組む経路が無い（D: `compat-feature-gap.md` の #2934 追記）、量子化（int8）は無い。<br>**対象外:** 汎用グラフコンパイル（`torch.compile` 相当）は #2615 の分岐 (i)＝対象外として確定（区分 B の実装は保留）、HTTP／API サーバ・量子化 KV・paged attention（条件付き）は #2624 案 C、サブワードトークナイザ（BPE 等）と Unicode 正規化表は #2618 の折衷案（語彙 lookup 型だけ対象内化）。いずれも `#issuecomment-6033824965` で承認（D: `graph-compile-scope-spec-proposal.md` §5・`facade-serving-infrastructure-spec-proposal.md` §0・`tokenizer-non-target-spec-proposal.md` §9）。<br>**保留:** speculative decoding・連続バッチングの論点 1・2・3・5・7・8、テキスト変換の論点 3 の Unicode 部分（小文字化・空白分割は ASCII 限定）と論点 6（出力モード・`StringLookup` 相当）（`#issuecomment-6067263650`・`-6079384681`）。<br>**実機 parity:** 実測済み（2026-10-09・main `8bbeb874`。タグとの `crates` 差分の 12 files に `inference`・`generate` 側のファイルは含まれない）: speculative・スケジューラの内部実装と facade 公開（`docs/perf/logs/speculative-batching-2890/README.md`・`docs/perf/logs/speculative-batching-facade-2934/README.md`。GB10・M4 Max とも pass）。`TextVectorization` は host 上のテキスト処理で GPU 経路が無い |

## 4. 0.10.0 列からの差分（タグで確認した事実のみ）

- `Sequential::add_*` は 31 → 65 に増えた（`add_conv3d`・`add_group_norm`・`add_transformer*`・活性化 9 種・Pool 系など）。
  0.10.0 の「Transformer デコーダ・GroupNorm 等は未公開」は解消した。
- `Adadelta`／`NAdam`／`Adamax`／`RAdam`／`Lion`／`Rprop` ほかの optimizer、param groups、EMA、SWA は公開済みになった。
  0.10.0 の「param groups・EMA・Adadelta／NAdam 等は未公開」は解消した。DataLoader は `PrefetchDataLoader` が加わったが、
  PyTorch のマルチワーカー DataLoader と同じ挙動かは本書では判定していない。
- 高階微分・カスタム VJP・`generate()`・KV キャッシュ・npy／npz・Adadelta は、0.10.0 のスコアボード注記で「未公開」の例に挙がっていたが、
  いずれも公開済みになった（上の各行）。
- 関数型 AD（`vjp`／`hvp`／`vmap`／`jvp`／`jacfwd`）・speculative decoding・連続バッチング第 1 段階・テキスト変換は 0.10.0 に無かった公開である。
- 0.10.0 で「部分的」だった 9 行は、いずれも 0.11.0 でも「部分的」のままである。理由は各行の「未公開」「保留」が空にならないことで、
  「ある」へ上げられる行は無い（判定は §2 の規則で機械的に決めた）。

## 5. 実機 parity の付録表

`docs/perf/logs/*-<2500 以上>/README.md` の 51 件を、タグ `v0.11.0` 時点の記入欄の内容で分類した。
実測済みは 6 件、判定対象外は 1 件、残り 44 件は記入欄が未実測である。
#2683（実機依存ツリー）の未完了分に当たるのは `low-precision-ops-2628`（#2629）のみで、#2607 と #2672 は 25xx 番台の README を持たない（§3 の各行に根拠を書いた）。
2500 未満の番号の README は本書の走査対象外である。

| README（`docs/perf/logs/` 配下） | 対象機能 | 対応表の行 | 状態（タグ時点の記入欄） | 測定コミット |
|---|---|---|---|---|
| `activation-scalar-ops-2649/README.md` | 活性化 5 種の `Var` 委譲 | 演算 | 実機未実測（#2683 対象外） | - |
| `binning-ops-2638/README.md` | histc／searchsorted／bucketize／bincount | 演算 | 実機未実測（#2683 対象外） | - |
| `compat-sequential-activation-layers-2529/README.md` | `Sequential` の活性化層 | NN 層 | 実機未実測（#2683 対象外） | - |
| `compat-sequential-activation-scalar-layers-2679/README.md` | `Sequential` の活性化 9 種の `add_*` | NN 層 | 実機未実測（#2683 対象外） | - |
| `compat-sequential-adaptive-max-global-pool-2527/README.md` | adaptive_max_pool・global_pool 層 | NN 層 | 実機未実測（#2683 対象外） | - |
| `compat-sequential-conv-transpose2d-2523/README.md` | `add_conv_transpose2d` | NN 層 | 実機未実測（#2683 対象外） | - |
| `compat-sequential-conv3d-2524/README.md` | `add_conv3d` | NN 層 | 実機未実測（#2683 対象外） | - |
| `compat-sequential-dropout-embedding-bag-2528/README.md` | Dropout2d／AlphaDropout／EmbeddingBag 層 | NN 層 | 実機未実測（#2683 対象外） | - |
| `compat-sequential-group-instance-norm-2525/README.md` | GroupNorm／InstanceNorm 層 | NN 層 | 実機未実測（#2683 対象外） | - |
| `compat-sequential-pixel-shuffle-2526/README.md` | PixelShuffle／PixelUnshuffle 層 | NN 層 | 実機未実測（#2683 対象外） | - |
| `compat-sequential-spatial-2522/README.md` | Upsample／ZeroPad2d／Identity 層 | NN 層 | 実機未実測（#2683 対象外） | - |
| `conv-transpose3d-max-unpool-2644/README.md` | ConvTranspose3d・MaxUnpool の `Var` 委譲 | 演算 | 実機未実測（#2683 対象外） | - |
| `create-graph-facade-2545/README.md` | `Tape::backward_create_graph` | 自動微分 | 実機未実測（#2683 対象外） | - |
| `cumulative-ops-2636/README.md` | cummax／cummin／logcumsumexp | 演算 | 実機未実測（#2683 対象外） | - |
| `dataset-compose-2661/README.md` | Subset／ConcatDataset／random_split | 最適化・学習ループ | 実機未実測（#2683 対象外） | - |
| `elementwise-loss-ops-2652/README.md` | 要素ごと損失 4 種 | 演算 | 実機未実測（#2683 対象外） | - |
| `f64-autograd-facade-2599/README.md` | `TapeF64`／`VarF64` | dtype | 実機未実測（#2683 対象外） | - |
| `facade-custom-function-2549/README.md` | `Tape::custom`／`CustomFunction` | 自動微分 | 判定対象外（README が REQ-2 の判定対象外と明記） | - |
| `fft-fft-ifft-2632/README.md` | fft／ifft | 演算 | 実機未実測（#2683 対象外） | - |
| `fft-rfft-irfft-2631/README.md` | rfft／irfft | 演算 | 実機未実測（#2683 対象外） | - |
| `fft-stft-istft-2633/README.md` | stft／istft | 演算 | 実機未実測（#2683 対象外） | - |
| `fold-unfold-2645/README.md` | Fold／Unfold の `Var` 委譲 | 演算 | 実機未実測（#2683 対象外） | - |
| `functional-fit-2667/README.md` | Functional モデルの fit | NN 層 | 実機未実測（#2683 対象外） | - |
| `functional-graph-2665/README.md` | Functional API のグラフ構築 | NN 層 | 実機未実測（#2683 対象外） | - |
| `functional-transforms-2881/README.md` | vjp／hvp／vmap の内部実装 | 自動微分 | **実測済み**（2026-10-09・GB10／M4 Max とも pass） | main `8bbeb874` |
| `functional-transforms-facade-2931/README.md` | `Tape::vjp`／`hvp`／`vmap`（facade） | 自動微分 | **実測済み**（2026-10-09・GB10／M4 Max とも pass） | main `8bbeb874` |
| `functional-transforms-jvp-2942/README.md` | jvp／jacfwd の内部実装（double-VJP） | 自動微分 | **実測済み**（2026-10-09・GB10／M4 Max とも pass） | main `8bbeb874` |
| `gradcheck-anomaly-2671/README.md` | gradcheck・anomaly detection（内部） | 自動微分 | 実機未実測（#2683 対象外） | - |
| `indexed-update-ops-2641/README.md` | scatter_reduce／index_add 等 | 演算 | 実機未実測（#2683 対象外） | - |
| `iterable-batch-sampler-2662/README.md` | IterableDataset／BatchSampler | 最適化・学習ループ | 実機未実測（#2683 対象外） | - |
| `jacobian-hessian-2670/README.md` | `Tape::jacobian`／`hessian` | 自動微分 | 実機未実測（#2683 対象外） | - |
| `low-precision-ops-2628/README.md` | 低精度 forward の compute_dtype opt-in | dtype | 実機未検証（#2683 未完了・#2629） | - |
| `lrn-weight-reparam-2646/README.md` | LRN・weight_norm・spectral_norm | 演算 | 実機未実測（#2683 対象外） | - |
| `margin-focal-loss-ops-2653/README.md` | マージン・focal 損失 4 種 | 演算 | 実機未実測（#2683 対象外） | - |
| `merge-ops-2666/README.md` | Functional の結合層 4 種 | NN 層 | 実機未実測（#2683 対象外） | - |
| `mha-config-sequential-2530/README.md` | `add_multihead_attention_with_config` | NN 層 | 実機未実測（#2683 対象外） | - |
| `nonfinite-ops-2635/README.md` | isnan／isinf／isfinite／nan_to_num | 演算 | 実機未実測（#2683 対象外） | - |
| `packed-sequence-2647/README.md` | PackedSequence | NN 層 | 実機未実測（#2683 対象外） | - |
| `pad-modes-2642/README.md` | pad の非定数モード | 演算 | 実機未実測（#2683 対象外） | - |
| `phase4-ops-autodiff-exposure-2678/README.md` | Phase 4 の演算・自動微分の facade 公開 | 演算 | 実機未実測（#2683 対象外） | - |
| `pool3d-ops-2643/README.md` | MaxPool3d／AvgPool3d | 演算 | 実機未実測（#2683 対象外） | - |
| `shape-view-ops-2639/README.md` | unbind／movedim／meshgrid 等 | 演算 | 実機未実測（#2683 対象外） | - |
| `softmin-threshold-ops-2650/README.md` | Softmin／Threshold／RReLU 等 | 演算 | 実機未実測（#2683 対象外） | - |
| `speculative-batching-2890/README.md` | speculative decoding・スケジューラの内部実装 | 推論・サービング | **実測済み**（2026-10-09・GB10／M4 Max とも pass） | main `8bbeb874` |
| `speculative-batching-facade-2934/README.md` | `inference` の speculative・`BatchScheduler`（facade） | 推論・サービング | **実測済み**（2026-10-09・GB10／M4 Max とも pass） | main `8bbeb874` |
| `stat-reduce-ops-2637/README.md` | median／kthvalue／quantile 等 | 演算 | 実機未実測（#2683 対象外） | - |
| `tape-gradcheck-facade-2847/README.md` | `Tape::gradcheck`（facade） | 自動微分 | **実測済み**（2026-10-09・GB10／M4 Max とも pass） | main `8bbeb874` |
| `tensor-product-ops-2640/README.md` | kron／tensordot／cdist／cross | 演算 | 実機未実測（#2683 対象外） | - |
| `transformer-decoder-sequential-2532/README.md` | `add_transformer_decoder_layer` | NN 層 | 実機未実測（#2683 対象外） | - |
| `transformer-sequential-2533/README.md` | `add_transformer` | NN 層 | 実機未実測（#2683 対象外） | - |
| `trig-ops-2634/README.md` | 逆三角・双曲線関数 9 種 | 演算 | 実機未実測（#2683 対象外） | - |

## 6. スコープ外・申し送り

- スコアボード（`body_0100.html`・Artifact）の対応表への反映は #2682 の担当である。
- #2683 の対象外で「実機未実測」のまま残る README 43 件には、追跡する open イシューが見当たらない。
  起票は承認が要るため本イシューでは行わず、申し送りとする（`.claude/rules/out-of-scope-tracking.md`）。
- #2541（参照モデルの facade 公開）は GitHub 上 closed だが、タグ時点の facade に公開名は無く、
  `docs/reference-models-decision.md` §11 は公開経路の判断待ちと記している。判定は公開面（タグのソース）を正とした。
  状態の食い違いの整理は別途判断が必要である。

## 7. 変更していないもの

`crates/` 全体・`docs/spec/`・tolerance／baseline・依存関係（`Cargo.toml`・`Cargo.lock`）・CI 設定・
`docs/compat-api-scope.md`／`docs/compat-feature-gap.md`・各申し送り README・0.9.0 版の記録・スコアボード関連ファイルは変更していない。
