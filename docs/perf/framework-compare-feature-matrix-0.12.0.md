# フレームワーク横並びスコアボード「役割・機能の対応表」v0.12.0 事前監査

イシュー #3269（親 #3018）。フレームワーク横並びスコアボード「役割・機能の対応表」の fandhe-ai 列で 0.11.0 時点に「部分的」だった 9 行
（dtype／自動微分／演算の範囲／NN 層・モデル構築／最適化・学習ループ／バックエンド・ハード／相互運用／事前学習済みモデル／推論・サービング）を、
0.12.0 の公開前に **main の `crates/facade` 公開面**で事前監査し、各行の根拠を「公開／未公開／対象外／保留／実機 parity」の 5 区分に分けて記録する。
0.11.0 版（`docs/perf/framework-compare-feature-matrix-0.11.0.md`・イシュー #2681）と同じ節構成で、その §8（#3023 の訂正 5 点）を入力にした。
**本書は crates.io 公開版ではなく main の事前監査である**。タグ `v0.12.0` との差分確認と、スコアボードへの反映は #3274 の担当で、
本書は §3 の判定を渡すだけである（Artifact・生成スクリプトは変更しない）。

## 1. 基準面の証明

判定基準は main の `crates/facade` の公開面とする。本書の `path:line` は、すべて `git show 1009825c9e9e954bf58748c5c1b42234bb12121d:<path>`（基準 sha 時点の内容）で再解決できる行番号である。

- **基準 sha**: `1009825c9e9e954bf58748c5c1b42234bb12121d`（`git rev-parse origin/main`。2026-10-27 に固定。以後 main が進んでも動かさない）。docs/spec の pin は `7534c9c`。
- 直近の公開版との差: `git diff --shortstat v0.11.0 1009825c9e9e954bf58748c5c1b42234bb12121d -- crates` は「487 files changed, 149852 insertions(+), 8800 deletions(-)」である（`v0.11.0` = `6b14fdb4`）。0.11.0 版 §1 の「差分なし」から大きく変わった。
- タグ `v0.12.0` は**未作成**で、版数の公開（publish）は別の明示指示を待つ。タグとの差分確認と再判定は #3274 が行う。
- 実機 parity の測定コミットは基準 sha と別である。各 README が記録する測定コミットと、基準 sha までの `crates/` 差分は §5 末尾の表に示す。
  測定コミットの多くは作業ブランチ上のコミットで main の履歴に残らない。したがって「測定済み」は「その測定コミットの内容が pass した」ことだけを意味し、基準 sha での pass を意味しない。
- 本書の調査は main 上のソース・決定記録・GitHub のコメントの読み取りだけで行い、コード・テスト・実機は動かしていない。

```
$ git diff --shortstat v0.11.0 1009825c9e9e954bf58748c5c1b42234bb12121d -- crates
487 files changed, 149852 insertions(+), 8800 deletions(-)
```

## 2. 判定規則

0.11.0 版 §2 を踏襲する（到達性・行判定値・根拠セルの 5 区分・実機 parity の状態・0.10.0 時点列の扱い）。変更点と追加点は次の 6 つである。

1. **件数表記**: 各区分は欄の先頭を `**公開（n 件）:**`・`**未公開（n 件）:**`・`**対象外（n 件）:**`・`**保留（n 件）:**` とし、項目は全角セミコロン「；」で区切る
   （項目の内部では読点「、」を使う）。0 件の区分は「なし（0 件）」と書く。区分の順序は公開→未公開→対象外→保留→実機 parity。
2. **保留ガードの扱い**: `*HoldDoctestGuard` の struct 名の有無は未公開の根拠にしない。ガードは公開後も「禁止経路専用」として名前を残す運用で、
   v0.11.0 と基準 sha で struct 名の集合は同じである（`lib.rs` から `^struct \w+HoldDoctestGuard` を抽出して比較し、43 件で同一）。判定は doctest 本文のプローブ（負）と `crates/facade/tests/api_surface.rs` の正ガードの内容で行う。
   別配置（`Tape`／`Tensor` 上の同名メソッド、モジュール再エクスポートなど）を拒むだけのガードは、0.11.0 版が項目に数えなかったものを今回も数えない
   （数えるのは 0.11.0 版 §3 が未公開に挙げた項目と、公開形を決めた機能の未公開部分）。
3. **facade 外クレートの扱い**: `hf-hub`・`backend-rocm` のように `publish = false` で facade の公開面から到達できないものは「公開」に数えない。
   行判定への効き方は該当行の項目に根拠を書く。
4. **承認の出所**: 0.11.0 版の #2499 のコメント 4 件（`#issuecomment-6033824965`＝2026-10-07、`-6052732061`＝2026-10-08、`-6067263650`＝2026-10-08、`-6079384681`＝2026-10-09）に、
   #3018 のコメント 3 件（`-6111643437`＝D-A／D-B／D-C／D-P／D-0〈2026-10-11〉、`-6111894131`＝D-1 追補〈2026-10-12〉、`-6207013704`＝K-3・paged attention〈2026-10-16〉）、
   #3021 のコメント（`-6111746880`＝項目 2〜10〈2026-10-12〉）、#3022 のコメント（`-6111750000`＝推奨形への包括承認〈2026-10-12〉）、
   #3140 のコメント（`-6227907703`〈2026-10-17〉）、#3114 のコメント（`-6213949713`〈2026-10-16〉）を加える。コメントの本文は転記せず要約する。
   「承認」はこれらのコメント ID と各イシュー番号の事実に限って書き、得ていない承認は書かない。
5. **実機 parity の「実機未検証（#2683 未完了）」**: #2683・#2629・#2672・#2607 はすべて CLOSED のため、この状態に当たるものは 0 件である（規則は残す）。
   ただし CLOSED でも README の記入欄が未実測のものがあり、それらは「実機未実測」として書く（§6）。
6. **保留の定義**: 0.11.0 版の「所有者コメントで保留が続くもの」に加え、「決定記録が所有者へ戻す論点として記録し、回答コメントが無いもの」も保留とする
   （ROCm の facade 公開・`TapeF64` の `jvp` 系・`create_graph` の残り。いずれも対応するコメントは無く、承認を得たとは書かない）。

## 3. 対応表

セル内の `D:` は `docs/` 配下の決定記録を指す。「0.11.0 時点」の列は 0.11.0 版 §3 の判定と §8 の訂正を要約したものである。

| 行 | 0.11.0 時点 | 0.12.0 事前監査判定 | 根拠（公開／未公開／対象外／保留／実機 parity） |
|---|---|---|---|
| dtype | 部分的（0.11.0 版 §3。保留・未公開が残る） | 部分的 | **公開（5 件）:** f64 の自動微分 `TapeF64`／`VarF64`（演算は add／mul／div／pow／matmul／sum／mean／max。#2599・#2834。`crates/facade/src/lib.rs:2075`・`crates/facade/src/lib.rs:2106`）；低精度 forward の opt-in `Var::matmul_low_precision` ほか 6 本（#2628・#2678。`crates/autodiff/src/var.rs:3634`）；dtype 変換と型付き演算 `CastDType`（`crates/facade/src/lib.rs:421`）・`TypedOps`（`crates/facade/src/lib.rs:433`）・`Tape::typed_ops_f64`（`crates/facade/src/lib.rs:1965`）；AMP `compile_with_amp`（`crates/facade/src/compat/training.rs:2060`）；CUDA GEMM の精度 opt-in `set_cuda_gemm_precision`（`crates/facade/src/lib.rs:2805`）。既定は FP32 のまま（#3262＝(A)。#3018 の `#issuecomment-6111894131`（D-1） の表の行）<br>**未公開（1 件）:** `Var` の低精度委譲以外の経路（`low_precision_ops` モジュール・裸の自由関数・`Tape` 上の同名メソッド）。承認形外の配置を `VarLowPrecisionOpsHoldDoctestGuard`（`crates/facade/src/lib.rs:5963`）が固定する。D-0 の除外名に含まれず、対象外へ移すには所有者確認が要る<br>**対象外（3 件）:** f16／bf16 を値型とする `Tape`／`Var`（#3021 の `#issuecomment-6111746880` の項目 2＝案 C〈opt-in の低精度 forward のみ〉を最終形とする。D: `autodiff-var-dtype-multiplexing-design.md` §0・§10 項 4）；sparse／complex（#2616 案 (ii)。#2499 の `#issuecomment-6033824965` で承認。D: `tensor-core-sparse-complex-decision.md` §15.0・§15.3）；int8 量子化（spec の除外事項「Won't・条件付き〈量子化 GEMM〉」。コード側に量子化経路は無い〈段階 0〉。D: `backend-int8-quantization-decision.md` §3.1。0.11.0 版 §8 訂正 1）<br>**保留（1 件）:** `TapeF64`／`VarF64` の `jvp` 系（`jvp`・`jacfwd`・`jvp_multi`・`jvp_create_graph`）。決定記録 `autodiff-functional-transforms-design.md` §33 は「f32 `Tape` 限定を契約として固定する」を推奨 (b) とするが、所有者の確認コメントは無い（#3074 は CLOSED でコメント 0 件）ため確定していない<br>**実機 parity:** f64 は実測済み（2026-10-19・CUDA `9e2f7ddcc`・Metal `bf894108a`・pass。`f64-autograd-facade-2599`・`var-f64-autograd-backend-parity-3168`）。f16／bf16 opt-in は実測済み（2026-10-11・`56b9d6ec`・CUDA／Metal とも pass。2026-10-19 に CUDA の大形状を `b640875f` で追加。`low-precision-ops-2628`）。int8 は性能判定対象外（`int8-mma-probe-2608`・`int8-quant-baseline-2609`。§5） |
| 自動微分 | 部分的（0.11.0 版 §3。保留 5 項目と未公開 3 項目） | 部分的 | **公開（12 件）:** 高階微分 `Tape::backward_create_graph`（#2545。`crates/facade/src/lib.rs:935`）。対象 Op は #3077〜#3086 で Softmax／LogSoftmax・Gelu・LayerNorm／RmsNorm・BatchNorm・Conv2d 系・損失 5 variant・索引／累積・Pad／MaskedFill・プーリング・FFT と一部の線形代数へ拡張し、facade 経由の確認は #3087；`CustomFunction`（`crates/facade/src/lib.rs:288`）・`Tape::custom`（#2549。`crates/facade/src/lib.rs:1686`）；hooks `Tape::register_backward_hook`（#2587。`crates/facade/src/lib.rs:1744`）；`Tape::jacobian`／`hessian`（#2678。`crates/facade/src/lib.rs:1018`・`crates/facade/src/lib.rs:1052`）；`Tape::gradcheck`（#2847。`crates/facade/src/lib.rs:1596`）・`Tape::backward_detect_anomaly`（`crates/facade/src/lib.rs:1563`）；関数型 AD `Tape::vjp`／`hvp`／`vmap`（#2931。`crates/facade/src/lib.rs:1102`・`crates/facade/src/lib.rs:1157`・`crates/facade/src/lib.rs:1466`）；`Tape::jvp`／`jacfwd`（#2956。`crates/facade/src/lib.rs:1228`・`crates/facade/src/lib.rs:1265`）；複数入力の `Tape::jvp_multi`（#3063。`crates/facade/src/lib.rs:1330`）；微分可能な `Tape::jvp_create_graph`（#3068。`crates/facade/src/lib.rs:1411`）；サンプルごとの勾配 `Tape::vmap_grad`（#3072。`crates/facade/src/lib.rs:1530`）；決定論モード `set_deterministic`（`crates/facade/src/lib.rs:2297`）；0.10.0 から継続: `Tape::var_no_grad`（`crates/facade/src/lib.rs:631`）・`backward_accumulate`（`crates/facade/src/lib.rs:650`）・`Var::detach`（`crates/autodiff/src/var.rs:278`）・`Var::checkpoint_from`（`crates/autodiff/src/var.rs:2501`）<br>**未公開（0 件）:** なし（0 件）<br>**対象外（4 件）:** D-0 の除外名: `Var::custom`／`Sequential::add_custom`／`Tape::add_custom`、関数型 AD の `Tape` 以外の経路（`functional_ops` の自由関数）、gradcheck のモジュール名と裸の自由関数。機能は `Tape::custom`・`Tape::vjp` 系・`Tape::gradcheck` で到達できる（#3018 の `#issuecomment-6111643437`（D-0）・#3021 の項目 1。D: `autodiff-custom-function-decision.md` §16.2・`autodiff-functional-transforms-design.md` §23.4）；ネイティブ forward-mode の JVP 規則とバッチ規則型 vmap（#2617 案 C の段階 2。#2499 の `#issuecomment-6033824965` で案 C を承認。D: `autodiff-forward-mode-vmap-spec-proposal.md` §0・§3）；`jvp` の f16 対応（#3021 の `#issuecomment-6111746880` の項目 3＝f32 `Tape` 限定を契約とする）；`create_graph` が構造的に定義できない Op（`ResidentLeaf`・`LinearResident`・`LinearAct`・`Custom`・`OneHot`・`CtcLoss`。決定記録 `autodiff-higher-order-grad-decision.md` §21.2 A 区分・§26.2。#3076 が D-A の「推奨形で一括承認」の下で書いた設計上の恒久除外で、D-A の出所は #3018 の `#issuecomment-6111643437`（D-0））<br>**保留（4 件）:** `create_graph` の対象 Op のうち §21.2 C 区分の 24 Op（選択縮約・統計縮約・Dropout／Interpolate／MaxUnpool・LocalResponseNorm／WeightNorm／SpectralNorm・RNN セル）。族に割り当てておらず、起票にはユーザー承認が要る（決定記録 §21.2）；`create_graph` が未実装で残る B 区分の Op（Conv3d／ConvTranspose3d〈§23.5 項 1〉、損失 15 variant〈§26.2〉、IndexedScatterReduce／Cumprod／Logcumsumexp〈`crates/autodiff/src/tape.rs` の `supports_create_graph` の doc〉、Stft／Istft・Cholesky・Pinv／Lstsq・MatrixNorm・QR／SVD／Eigh 系〈§31.1〉）。所有者へ戻す論点として記録されたまま確定していない；実装済み Op の拒否範囲と意味論（LayerNorm／RmsNorm／BatchNorm の S2〈§27.2・§28.5〉、Det／Inv／Solve／SlogdetLogAbsDet の S2／V4〈§31.5 項 1〉、Embedding の `padding_idx` の二階の意味論〈§25.3〉）。いずれも所有者判断として残っている；`TapeF64`／`VarF64` の `jvp` 系（dtype 行と同じ。#3074 の決定記録 §33 は推奨 (b) のまま所有者の確認コメントが無い）<br>**実機 parity:** 実測済み: `Tape::vjp`／`hvp`／`vmap`・`Tape::gradcheck`（2026-10-09・`8bbeb874c`・GB10／M4 Max とも pass。`functional-transforms-2881`・`functional-transforms-facade-2931`・`tape-gradcheck-facade-2847`）。`jvp`／`jacfwd` は double-VJP のミラー実装を 2026-10-09 に、本物の `functional_ops::{jvp,jacfwd}` を 2026-10-19 に測定（CUDA `9e2f7ddcc`・Metal `bf894108a`・pass。`functional-transforms-jvp-2942`・`cuda-autodiff-parity-3168`・`metal-autodiff-parity-3169`）。`jacobian`／`hessian`・gradcheck／anomaly は 2026-10-11（CUDA `784b4e79`・Metal `c8e1ceb1`。`jacobian-hessian-2670`・`gradcheck-anomaly-2671`）、`backward_create_graph` と拡張 Op は 2026-10-19（`create-graph-facade-2545`・`create-graph-extended-ops-3087`）、`jvp_multi`／`vmap_grad` は 2026-10-19（CUDA `b640875f`・Metal `66537019`。`functional-transforms-jvp-multi-3064`・`functional-transforms-vmap-grad-3073`）。判定対象外: `facade-custom-function-2549`（host 上の `Tensor<f32>` で完結する） |
| 演算の範囲 | 部分的（0.11.0 版 §3。未公開 2 項目） | 部分的 | **公開（7 件）:** Phase 1 の `Var` 委譲（#2510〜#2519）に加え、Phase 4 の演算（`docs/compat-api-scope.md` §5.1 行 1〜11・17〜20。#2678・#2854）；FFT `Var::rfft`（`crates/autodiff/src/var.rs:4966`）；順序統計 `Var::median`（`crates/autodiff/src/var.rs:5210`）、テンソル積 `Var::kron`（`crates/autodiff/src/var.rs:5390`）、`Var::pad_with_mode`（`crates/autodiff/src/var.rs:5431`）；3D プーリング・ConvTranspose3d・MaxUnpool の `Var` 委譲（#2850。`Var::max_pool3d` `crates/autodiff/src/var.rs:5448`・`Var::conv_transpose3d` `crates/autodiff/src/var.rs:5489`・`Var::max_unpool2d` `crates/autodiff/src/var.rs:5533`）；Fold／Unfold・LRN・weight_norm／spectral_norm の `Var` 委譲（#2851。`Var::fold` `crates/autodiff/src/var.rs:5593`・`Var::local_response_norm` `crates/autodiff/src/var.rs:5611`・`Var::weight_norm` `crates/autodiff/src/var.rs:5625`・`Var::spectral_norm` `crates/autodiff/src/var.rs:5632`）；オプション型を取る損失 `Var::bce_with_logits_loss_with`（#2854。`crates/autodiff/src/var.rs:4807`）；`Tape::bincount`（`crates/facade/src/lib.rs:960`）・logical 3 本（`logical_and`。`crates/facade/src/lib.rs:2498`）<br>**未公開（1 件）:** GPU 線形代数の opt-in 関数 `set_cuda_gpu_linalg_enabled`／`set_metal_gpu_linalg_enabled`。公開形は #3140 のコメント `#issuecomment-6227907703` で承認済みだが、前提 G2（#3138・#3139 の実機 parity）が未実測のため公開は #3405（OPEN）へ切り出された。公開面は否定ガード（`crates/facade/tests/api_surface.rs:41969`）が固定する。決定記録 `backend-gpu-linalg-decision.md` §11<br>**対象外（4 件）:** D-0 の除外名: `elementwise_loss_ops`／`margin_focal_loss_ops`／`fold_ops` のモジュール再エクスポートと `norm_except_dim`（演算は `Var` のメソッドから使える。#3021 の `#issuecomment-6111746880` の項目 1。`elementwise_loss_ops` のガードは `crates/facade/src/lib.rs:7685` が禁止経路専用として残る）；sparse／complex の演算（dtype 行と同じ #2616 案 (ii)）；Metal の eigh・SVD と CUDA の SVD（#3021 の `#issuecomment-6111746880` の項目 8。D: `backend-gpu-linalg-decision.md` §2・§9）；int／bool 演算・線形代数の網羅（#3021 の `#issuecomment-6111746880` の項目 4。REQ-9 Tier 2 の列挙を基準とし「数千 op 級」は注記として不算入）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** 実測済み: Phase 4 の演算は 2026-10-11（CUDA `784b4e79`〈#2992〉・Metal `c8e1ceb1`〈#2993〉・Mac mini。§5 の 2026-10-11 の行）。3D プーリング・ConvTranspose3d・Fold／Unfold・LRN・損失 2 群の 6 README は 2026-10-19（CUDA `b640875f`〈#3161〉・Metal `d79f4895`〈#3162〉）。Metal の `max` reduction は 2026-10-19（`metal-max-reduce-3475`）。GPU 線形代数は**実機未実測**（`cuda-linalg-lu-2674`・`cuda-linalg-chol-qr-eigh-2675`・`cuda-linalg-parity-3138`・`metal-linalg-lu-chol-qr-2676`・`metal-linalg-parity-3139`。#2672・#3138・#3139 は CLOSED だが記入欄は未記入のまま。§5・§6） |
| NN 層・モデル構築 | 部分的（0.11.0 版 §3。未公開 4 項目と保留 1 項目） | 部分的 | **公開（6 件）:** `compat::Sequential::add_*` は 65 → 71（`grep -E "^\s*pub fn add_"` で v0.11.0 と基準 sha を比較）。追加分: `add_max_pool3d`・`add_avg_pool3d`（`crates/facade/src/compat/sequential.rs:2150`。#3028）・`add_conv_transpose3d`（`crates/facade/src/compat/sequential.rs:1519`。#3032）・`add_fold`／`add_unfold`（`crates/facade/src/compat/sequential.rs:2214`。#3040）・`add_local_response_norm`（`crates/facade/src/compat/sequential.rs:2250`。#3044）；MaxUnpool の層経路 `FunctionalBuilder::max_unpool1d`／`2d`／`3d`（#3036。`crates/facade/src/compat/functional.rs:552`）。`Sequential::add_max_unpool*` は設けない形で確定（決定記録 `compat-sequential-3d-fold-lrn-layers-decision.md` §6 案 A）；weight_norm／spectral_norm の層への結線 `Sequential::parametrize_weight_norm`（`crates/facade/src/compat/sequential.rs:683`）・`parametrize_spectral_norm`（`crates/facade/src/compat/sequential.rs:748`）（#3046〜#3050）；`save_model`／`load_model` の kind 追加: 活性化 9 層（#3309・#3330 の PR）、`conv_transpose1d`／`unflatten`（PR #3310）、再パラメータ化 14 kind（#3050）、3D プーリング・ConvTranspose3d・MaxUnpool・Fold／Unfold・LRN（PR #3314・#3317・#3320・#3323・#3326。#3051 の系統）；参照モデル `fandhe_ai::models::{Mlp, LeNet, ResNet, ResNetBlock, TransformerClassifier, TransformerClassifierConfig}`（#2976・#2977。`crates/facade/src/lib.rs:241`。0.11.0 版 §8 訂正 4）；0.11.0 から継続: Functional API `FunctionalBuilder`、`nn::Transformer`、`nn::loss`、可変長系列 `nn::rnn`、`KvCache`、`nn::init`<br>**未公開（2 件）:** `add_upsample` の `scale_factor` 指定（型名ではなく機能の未公開。`UpsampleSize` の再エクスポートか別名メソッドが要り別承認。決定記録 `facade-nn-layer-type-exposure-decision.md` §6。現行の `crates/facade/src/compat/sequential.rs:2435` は `size` 指定のみ）；独自 `Module` から組み込み層を呼ぶ経路（`Sequential` への `nn::Module` 実装など新しい公開面が要り別承認。同 §6・`facade-nn-module-exposure-decision.md` §22.6）<br>**対象外（1 件）:** 層型名の公開（`ConvTranspose1d`・`Upsample`・`ZeroPad2d`・`Identity`・`Unflatten` の 5 型を含む `fandhe_ai::nn::*` の全層型）。D-0 の「層型名 5 件」＋#3022 の決定記録＝公開対象外（(i)+(iv)）。`SpatialLayersHoldDoctestGuard`（`crates/facade/src/lib.rs:3793`）は維持。#3022 の `#issuecomment-6111750000` は「推奨形への包括承認」であり (i)+(iv) への個別承認ではない（決定記録 §7）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** 実測済み: `Sequential` 層の旧 9 群は 2026-10-19（CUDA `b640875f`〈#3163〉・Metal は 1 回目 `b640875f` で 1 本 fail → #3475 の結線後に 2 回目 `b728b9cb` で 10 本 pass〈#3164〉）。3D 層・Fold／LRN・MaxUnpool の層経路は 2026-10-19（CUDA `b640875f`〈#3171〉・Metal `66537019`〈#3172〉）。Transformer／Functional／可変長系列の 7 群は 2026-10-19（CUDA `a115595c6`〈#3165〉・Metal `6570171a`〈#3166〉） |
| 最適化・学習ループ | 部分的（0.11.0 版 §3。未公開 2 項目） | 部分的 | **公開（7 件）:** optimizer: `Adadelta`（`crates/facade/src/optim.rs:515`）・`Adafactor`・`Adamax`・`Asgd`・`Lion`・`Rprop`・`NAdam`・`RAdam`、scheduler `PolynomialLr`（`crates/facade/src/optim.rs:522`）・`CyclicLr`（`crates/facade/src/optim.rs:526`）；param groups `ParamGroup`（#2553。`crates/facade/src/optim.rs:546`）・`compile_with_param_groups`（`crates/facade/src/compat/training.rs:1993`）・`OptimizerStateDict`（#2556。`crates/facade/src/optim.rs:543`）；EMA `ExponentialMovingAverage`（#2560。`crates/facade/src/optim.rs:513`）・`Callback::Ema`（`crates/facade/src/compat/callbacks.rs:1671`）；SWA `AveragedModel`（`crates/facade/src/optim.rs:514`）・`SwaLr`／`SwaAnneal`（`crates/facade/src/optim.rs:531`）と、fit への結線 `Callback::Swa(SwaCallback)`（#3056〜#3060。`crates/facade/src/compat/callbacks.rs:1673`・`crates/facade/src/compat/callbacks.rs:1403`。D-A の推奨形）；fit 拡張 `fit_with_weights`（`crates/facade/src/compat/training.rs:2650`）・`fit_with_train_step`（`crates/facade/src/compat/training.rs:2552`）；学習ステップ入口 `Sequential::train_step_resident` と loss ハンドル `ResidentLoss`（#3248・#3254。入力・ターゲットの常駐ハンドルと学習ステップ入口の公開は #3249＝案 A（#3018 の `#issuecomment-6111894131`（D-1） の表の行）。`crates/facade/src/compat/sequential.rs:3893`・`crates/facade/src/lib.rs:527`）；データ `PrefetchDataLoader`（#2603。`crates/facade/src/data.rs:121`）・`ConcatDataset`・`IterableDataset`・`random_split`（`crates/facade/src/data.rs:124`）<br>**未公開（1 件）:** `FitConfig`／`Sequential` への SWA 用メソッド（`use_swa`／`swa_start`／`swa_lr`）。決定記録 `autodiff-swa-decision.md` §16 の契約で、`SwaHoldDoctestGuard`（`crates/facade/src/lib.rs:4608`）が禁止経路として固定する。EMA の `FitConfig` 形は D-0 で対象外だが、D-0 の除外名と #3021 に SWA は無いため、同じ扱いにしてよいかは所有者確認要<br>**対象外（1 件）:** `FitConfig` への EMA（`use_ema`／`ema_decay`）。D-0 の除外名。機能は `Callback::Ema(EmaCallback)` で到達でき、`FitConfig` は `Copy + Eq` を公開契約とする（#3018 の `#issuecomment-6111643437`（D-0）。D: `autodiff-ema-decision.md` §10.2 (d)・§14.3。ガードは `crates/facade/src/lib.rs:4501` が禁止経路専用として残る）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** 実機未実測: `dataset-compose-2661`・`iterable-batch-sampler-2662`（host ユーティリティ。記入欄は round-trip の確認でいずれも未実測）。学習ステップ常駐チェーンは実測済み: Metal・CUDA とも Tape 経路と bit 一致（`resident-chain-parity-3256`。日付の記載なし）、2026-10-24 の `train-step-chain-cuda-3252`・`train-step-chain-metal-3251`、2026-10-23 の `train-step-resident-parts-parity-3224`（GB10 の初回は C5 が fail → 修正後の再計測で全 pass。§5） |
| バックエンド・ハード | 部分的（0.11.0 版 §3。未公開 1 項目と保留 1 項目。§8 訂正 1・2 を反映） | 部分的 | **公開（3 件）:** `tape_for`（`crates/facade/src/lib.rs:2190`）・`available_devices`（`crates/facade/src/lib.rs:2245`）・`set_cuda_gemm_precision`（`crates/facade/src/lib.rs:2805`）；デバイス間の値転送 `Tape::transfer`（`crates/facade/src/lib.rs:888`）。複数 ordinal の tape 作成も可能（0.11.0 版 §8 訂正 2）；分散学習の初期形 `fandhe_ai::distributed`（`crates/facade/src/lib.rs:248`）。`Communicator`・`Rendezvous`・`DistributedConfig`・`step_sgd`／`step_adam`／`step_adamw`（`crates/facade/src/distributed/mod.rs:65`）。std の TCP によるホスト側 ring all-reduce で、検証は単一ノード複数プロセス（#3144〜#3148。D: `facade-distributed-host-collective-decision.md` §0）<br>**未公開（0 件）:** なし（0 件）<br>**対象外（3 件）:** FSDP・ZeRO 系・tensor／pipeline parallel（Won't。D: `ddp-grade-up-conditions.md` §3a）；cudarc の `nccl` feature（#3021 の `#issuecomment-6111746880` の項目 10＝複数 GPU 機が無い間は条件付き対象外。初期形は #3142 のホスト側 collective）；int8 量子化（dtype 行と同じ。spec の除外事項「Won't・条件付き」。0.11.0 版 §8 訂正 1）<br>**保留（1 件）:** ROCm の facade 公開。`Device` は `#[non_exhaustive]` でなく、`Device::Rocm` の追加は破壊的変更で、`backend-rocm` は `publish = false`。決定記録 `backend-rocm-device-exposure-decision.md` §1・§6 の Q1〜Q3 は所有者判断待ちで承認コメントは無い（facade の doc も「ROCm は未公開」と書く。`crates/facade/src/lib.rs:2213`）。#3021 の `#issuecomment-6111746880` の項目 9 は表記を「条件付き」と定めただけで facade 公開の承認ではない。D-B（#3018 の `#issuecomment-6111643437`（D-B））は実装を進める承認で、公開の承認ではない<br>**実機 parity:** 実機未実測: `ddp-host-collective-gb10-3149`（GB10 の 2 プロセス分散学習。#3149 は CLOSED だが記入欄は未実測）・`rocm-parity-3159`（MI300X。#3159 は CLOSED だが「状態: 未実測」）。int8 は性能判定対象外（§5）。CUDA GEMM の既定精度は #3262 で FP32 のまま |
| 相互運用 | 部分的（0.11.0 版 §3。未公開 3 項目。§8 訂正 3 を反映） | ある | **公開（4 件）:** ONNX import `OnnxModel`（`crates/facade/src/interop/onnx.rs:228`）・`from_bytes`・`run`、export `OnnxModel::from_sequential`（`crates/facade/src/interop/onnx.rs:333`。対応層は `Linear`／`ReLU`／`Softmax`／`LayerNorm`／`GELU`／`Conv2d` の 6 種で、それ以外は `OnnxError::UnsupportedLayer`。0.11.0 版 §8 訂正 3）・`to_bytes`・`OnnxExportOptions`；safetensors `load_safetensors_f32`・`save_safetensors_f32`（`crates/facade/src/interop/safetensors.rs`）；npy／npz 自由関数 `load_npy`／`save_npy`／`load_npz`／`save_npz`（`crates/facade/src/interop/npy.rs`。#2590・#2831）；`save_model`／`load_model`（`crates/facade/src/compat/model_io.rs`）と Functional モデルの保存。kind 追加は NN 層行のとおり（活性化 9 層・再パラメータ化 14 kind・3D 層など）。基準 sha で `save_model` が `Unsupported { kind }` で拒否する層は 1 件だけで、`add_module` の利用者定義層（`crates/facade/src/compat/sequential.rs:2523`。設計上の拒否）である。`add_*` の 70 件（71 件から `add_module` を除く）はすべて固有の `LayerSpec` を積み、保存できる<br>**未公開（0 件）:** なし（0 件）<br>**対象外（2 件）:** Python バインディングと TF 系形式（SavedModel・TFLite・Keras H5）。#2623 案 D＝非目標を維持（#2499 の `#issuecomment-6033824965` で承認。D: `python-binding-tf-format-non-target-spec-proposal.md` §0・§3b）；`Tensor` への npy／npz の inherent メソッド（D-0 の除外名。facade から `Tensor` へ inherent を足せない〈E0116〉。機能は自由関数で到達できる。#3018 の `#issuecomment-6111643437`（D-0）。ガードは `crates/facade/src/lib.rs:4890` が禁止経路専用として残る）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** 該当なし（host 側 I/O） |
| 事前学習済みモデル | 部分的（0.11.0 版 §3。未公開 2 項目と保留 2 項目。§8 訂正 4 を反映） | 部分的 | **公開（4 件）:** `ModelRegistry`（`crates/facade/src/model.rs:684`）・`ModelRegistry::load`；ローカル配置の書き込み側 `ModelRegistry::save`（#3117・#3119。`crates/facade/src/model.rs:1017`）；HTTPS 取得 `ModelRegistry::download`（`crates/facade/src/model.rs:1234`）・`download_with`（`crates/facade/src/model.rs:1287`）。sha256 pin 検証とローカルキャッシュ配置を含む（#3121。HTTP／TLS は第 11 区分 `ureq`・`cfg(unix)`）；参照モデル `fandhe_ai::models`（アーキテクチャのみで重みは含まない。NN 層行と同じ。`crates/facade/src/lib.rs:241`）<br>**未公開（1 件）:** HF Hub 連携。実装は facade の外の補助クレート `crates/hf-hub`（`publish = false`・facade の安定性保証の外。公開 API は `HfHub::resolve_revision`／`list_files` で、`download` は crate 内限定）。crates.io 公開（2 段目）と `compat-api-scope.md` §0 との整合は所有者確認待ち（決定記録 `hf-hub-integration-design.md` §9.3・§9.9）。D-B は実装を進める承認で、公開の承認ではない<br>**対象外（1 件）:** 重みの同梱（#3021 の `#issuecomment-6111746880` の項目 5＝#1965 判断 2 を出所とし、配布は本ライブラリのスコープ外。`model-distribution-design.md` §5 は「汎用 URL ダウンロードはコア、特定ハブは別クレート」とする方針で、項目 5 と矛盾しない）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** 実測済み（HTTPS 実通信とビルドの実機確認。2026-10-19・`8eb15ab57`・GB10／Mac mini。`https-download-3173`。REQ-2 の数値 parity ではない）。重み読み込み自体は該当なし |
| 推論・サービング | 部分的（0.11.0 版 §3。未公開 3 項目と保留 2 項目。§8 訂正 5 を反映） | 部分的 | **公開（8 件）:** `Sequential::predict`（`crates/facade/src/compat/sequential.rs:2579`）・`predict_batches`・`predict_resident`（`crates/facade/src/compat/sequential.rs:3774`）・`predict_resident_input`（`crates/facade/src/compat/sequential.rs:3821`。PR #3562）；自己回帰生成 `generate`／`AutoregressiveModel`／`GenerateConfig`／`SamplingStrategy`（`crates/facade/src/inference/mod.rs:364`）。`caches` 形モデルを `generate` 系へつなぐ経路（#3094〜#3097。D: `facade-speculative-decoding-batching-design.md` §19）；`KvCache`（可変長・padding mask。#3105）と、デバイス常駐 KV の `PagedKvCache`・`Tape::new_paged_kv_cache`（`crates/facade/src/lib.rs:1919`）・`paged_attention_forward`（`crates/facade/src/lib.rs:1939`）（K-3・paged attention。#3112。承認は #3018 の `#issuecomment-6207013704` の項目 4。0.11.0 版 §8 訂正 5）；連続バッチング `BatchScheduler`（`crates/facade/src/inference/mod.rs:373`）とテンソル単位バッチの `step_batched`（#3106）；`generate_speculative`（`crates/facade/src/inference/mod.rs:377`）。greedy・B = 1 に加え、サンプリング版（#3099。判定契約は #3100 の選択肢 B、決定記録 §20）；テキスト変換 `TextVectorization`（Unicode の小文字化・空白分割〈#3089〉、出力モード multi_hot／count／tf_idf〈#3092〉）と `StringLookup`（#3093。`crates/facade/src/text/mod.rs:154`・`crates/facade/src/text/mod.rs:151`）；グラフコンパイル区分 B の opt-in setter `set_cuda_graph_infer_enabled`（`crates/facade/src/lib.rs:2716`。#3116）・`set_cuda_elementwise_fusion_enabled`（`crates/facade/src/lib.rs:2748`）・`set_metal_elementwise_fusion_enabled`（PR #3386。B-1 の新規 `unsafe` の承認は #3114 のコメント 6213949713）；`PhaseMetrics`（`crates/facade/src/inference/mod.rs:357`）<br>**未公開（1 件）:** `caches` で表せない内部状態（KV 以外。RNN の隠れ状態など）を持つモデルの speculative／連続バッチング。決定記録 `facade-speculative-decoding-batching-design.md` §19.10 は「保留・スコープ外。需要が記録になく、必要になれば案 B を出発点に別 issue」とする（所有者への承認依頼ではない）<br>**対象外（5 件）:** 汎用グラフコンパイル区分 A（`torch.compile` 相当）。#2615 の分岐 (i)＝対象外で確定（D: `graph-compile-scope-spec-proposal.md` §5。区分 B の opt-in は公開済み）；HTTP／API サーバと量子化 KV（#2624 案 C。D: `facade-serving-infrastructure-spec-proposal.md` §0）。paged attention は対象外ではなく移行条件を満たして公開済み（0.11.0 版 §8 訂正 5）；サブワードトークナイザ（BPE 等）と Unicode 正規化表（#2618 の折衷案。D: `tokenizer-non-target-spec-proposal.md` §9）；beam search・top-p・EOS 停止・repetition penalty・ストリーミング生成（#3021 の `#issuecomment-6111746880` の項目 7＝不算入。所有者の Won't ではなく、算入するなら別 issue）；量子化（int8）。spec の除外事項「Won't・条件付き」（dtype 行と同じ。0.11.0 版 §8 訂正 1）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** 実測済み: speculative とスケジューラは 2026-10-09（`8bbeb874c`・GB10／M4 Max とも pass。`speculative-batching-2890`・`speculative-batching-facade-2934`）。paged attention は 2026-10-19（CUDA `b640875f`・Metal `66537019`。`paged-attention-parity-3113`）。実機未実測: サンプリング版 speculative（`speculative-sampling-3102`。#3102 は CLOSED だが記入欄は未実測。決定記録 §27 は #3171／#3172 の測定対象に加えるかを所有者判断とする）。`TextVectorization` は host 上のテキスト処理で GPU 経路が無い。論点 2（greedy の token 列一致）は #3021 の `#issuecomment-6111746880` の項目 6 で閉じた |

## 4. 0.11.0 列からの差分

### 4.1 件数の集計と行判定

「ある」は未公開 0 件かつ保留 0 件の行だけで、機械的に決めた（§2 の規則）。件数は §3 の各欄の件数と一致する。

| 行 | 0.11.0 判定 | 公開 | 未公開 | 対象外 | 保留 | 0.12.0 事前監査判定 |
|---|---|---|---|---|---|---|
| dtype | 部分的 | 5 | 1 | 3 | 1 | 部分的 |
| 自動微分 | 部分的 | 12 | 0 | 4 | 4 | 部分的 |
| 演算の範囲 | 部分的 | 7 | 1 | 4 | 0 | 部分的 |
| NN 層・モデル構築 | 部分的 | 6 | 2 | 1 | 0 | 部分的 |
| 最適化・学習ループ | 部分的 | 7 | 1 | 1 | 0 | 部分的 |
| バックエンド・ハード | 部分的 | 3 | 0 | 3 | 1 | 部分的 |
| 相互運用 | 部分的 | 4 | 0 | 2 | 0 | ある |
| 事前学習済みモデル | 部分的 | 4 | 1 | 1 | 0 | 部分的 |
| 推論・サービング | 部分的 | 8 | 1 | 5 | 0 | 部分的 |

- 「ある」になった行: 相互運用。それ以外の 8 行は「部分的」のままである。
- 相互運用行は、0.11.0 版の未公開 3 項目が解消または再分類された結果である（ONNX export の記述は §8 訂正 3 で誤読と確定、活性化 9 層の `save_model` は kind 追加で公開、`Tensor` の npy inherent は D-0 で対象外）。
  ただし ONNX export の対応層が 6 種に限ること等は区分に当たらない機能範囲の制約であり、規則上は行を下げない（注記は §6）。

### 4.2 0.11.0 版から「公開」へ移った項目

- **NN 層・演算**: 3D プーリング・ConvTranspose3d・Fold／Unfold・LRN の層化（#3028・#3032・#3040・#3044）、MaxUnpool の層経路（#3036）、weight_norm／spectral_norm の結線（#3046〜#3050）、
  `save_model` の kind 追加（#3051〜#3055）、参照モデル（#2976・#2977）。
- **最適化・学習ループ**: SWA の fit 結線（#3056〜#3060）、学習ステップ入口と loss ハンドル（#3248・#3254）。
- **自動微分**: 複数入力の jvp（#3063）、微分可能な jvp（#3068）、`vmap(grad)`（#3072）、`create_graph` の対象 Op 拡張（#3075 配下の #3077〜#3087）。
- **推論・サービング**: Unicode（#3089）、出力モードと `StringLookup`（#3090〜#3093）、`caches` 形モデルの経路（#3094〜#3097）、サンプリング版 speculative（#3099）、
  テンソル単位バッチ（#3103〜#3106）、K-3・paged attention（#3108〜#3112）、区分 B の opt-in（#3114〜#3116 と PR #3386）。
- **事前学習済みモデル**: `ModelRegistry` の書き込み（#3117・#3119）、HTTPS ダウンロード（#3121）。
- **バックエンド・ハード**: 分散学習の初期形 `fandhe_ai::distributed`（#3144〜#3148 と PR #3414）。

### 4.3 0.11.0 版から「対象外」へ移った項目

- **D-0**（`#issuecomment-6111643437`）: `Var::custom` 系、`functional_ops` の自由関数、gradcheck の自由関数、`elementwise_loss_ops`／`margin_focal_loss_ops`／`fold_ops` の再エクスポート、
  `norm_except_dim`、`Tensor` の npy inherent、`FitConfig` への EMA、層型名 5 件（機能は別名で公開済み）。
- **#3021 の項目 2〜10**（`#issuecomment-6111746880`）: f16／bf16 を値型とする `Tape`／`Var`、`jvp` の f16、int／bool 演算の網羅（不算入）、同梱重み、speculative の greedy 仮説（閉じた）、
  beam search などの生成機能（不算入）、Metal の eigh・SVD と CUDA の SVD、ROCm の表記（条件付き）、`nccl` feature。
- **#3022**（`#issuecomment-6111750000`）: `fandhe_ai::nn::*` の全層型は公開対象外（(i)+(iv)）。
- **0.11.0 版 §8 訂正 1・5**: int8 は「Won't・条件付き」、paged attention は対象外ではなく条件付きで、移行条件を満たして公開済み。

### 4.4 未公開・保留のまま、または新たに記録した項目

- **未公開**: `low_precision_ops` 配置のガード、GPU 線形代数の opt-in 関数（#3405）、`add_upsample` の `scale_factor`、独自 `Module` から組み込み層を呼ぶ経路、
  SWA の `FitConfig` 形、HF Hub 連携（facade 外の補助クレート）、`caches` で表せない内部状態を持つモデル。
- **保留**: ROCm の facade 公開、`TapeF64`／`VarF64` の `jvp` 系、`create_graph` の残る Op・拒否範囲・意味論。
- 0.11.0 版にあった保留のうち、D-A（`#issuecomment-6111643437`）で推奨形のまま実装へ進んだ項目は §4.2 のとおり公開へ移った。残る保留は上の 3 件である。
  D-B の 4 件は、HTTPS ダウンロードが公開、DDP が初期形で公開、HF Hub が facade 外の補助クレートとして実装、ROCm は実装は進むが facade 公開が所有者判断待ちである。

## 5. 実機 parity の付録表

`docs/perf/logs/*-<2500 以上>/` の 120 件（基準 sha 時点。`git ls-tree -d --name-only 1009825c9e9e954bf58748c5c1b42234bb12121d docs/perf/logs/` から番号 2500 以上を数えた件数）を全件載せる。
README が無い 2 件（`infer-chain-graphcapture-rejudge-3205`・`train-step-graph-3260`）も行として載せ、黙って除外しない。
2500 未満の番号のディレクトリは走査対象外である。状態は記入欄の行で判定し、見出しの語だけでは判定しない（§2）。

- 実測済み 70 件・実機未実測 10 件・判定対象外 3 件・性能計測 37 件（合計 120 件）。
- 実機未検証（#2683 未完了）は 0 件（#2683・#2629・#2672・#2607 が CLOSED。§2 の 5）。
- 性能計測は対応表の行に属さず（「—」）、parity の記入欄を持たない性能診断である（`cpu-gemm-*`・`sme-*`・`metal-tile-*`・`opp-kernel-*`・`optin-ab-*`・`train-step-graph-*` など）。
- 判定対象外 3 件は、`facade-custom-function-2549`（README が REQ-2 の判定対象外と明記）と、int8 の性能判定 2 件（`int8-mma-probe-2608`・`int8-quant-baseline-2609`。int8 は Won't・条件付き）である。
- 0.11.0 版 §8 付随事項の #3006／#3008（14 機能・16 README）と、Phase 8（#3161〜#3173）の実測は、各 README の記入欄で確認して下表に反映した。

| README | 対象機能 | 対応表の行 | 状態（記入欄） | 実測日 | 測定コミット |
|---|---|---|---|---|---|
| `activation-scalar-ops-2649/README.md` | SELU・CELU・Softsign・Hardsigmoid・LogSigmoid の `Var` 委譲 | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `bench-fandhe-symmetry-ab-3266/README.md` | bench-fandhe 是正前後の併記計測 | — | 性能計測（parity 判定対象外） | - | - |
| `binning-ops-2638/README.md` | histc／searchsorted／bucketize／bincount | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `compat-sequential-activation-layers-2529/README.md` | `Sequential` の Mish／Hardtanh／ReLU6／GLU／PReLU | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-activation-scalar-layers-2679/README.md` | `Sequential` の活性化 9 種の `add_*` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-adaptive-max-global-pool-2527/README.md` | adaptive_max_pool・global_pool 層 | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-conv-transpose2d-2523/README.md` | `add_conv_transpose2d` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-conv-transpose3d-3033/README.md` | `add_conv_transpose3d` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `compat-sequential-conv3d-2524/README.md` | `add_conv3d` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-dropout-embedding-bag-2528/README.md` | Dropout2d／AlphaDropout／EmbeddingBag 層 | NN 層 | 実測済み（pass）。Metal の 1 回目は 1 本 fail（`MetalBackendOps::max` が `Unsupported`）→ #3475 の結線後の 2 回目で pass | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-fold-unfold-3041/README.md` | `add_fold`／`add_unfold` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `compat-sequential-group-instance-norm-2525/README.md` | GroupNorm／InstanceNorm 層 | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-local-response-norm-3045/README.md` | `add_local_response_norm` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `compat-sequential-max-unpool-3037/README.md` | `FunctionalBuilder` の MaxPool 索引・MaxUnpool | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `compat-sequential-pixel-shuffle-2526/README.md` | PixelShuffle／PixelUnshuffle 層 | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-pool3d-3029/README.md` | `add_max_pool3d`／`add_avg_pool3d` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `compat-sequential-spatial-2522/README.md` | Upsample・ZeroPad2d・Identity 層 | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `conv-transpose3d-max-unpool-2644/README.md` | ConvTranspose3d・MaxUnpool の `Var` 委譲 | 演算 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `d79f4895` |
| `cpu-gemm-reuse-fresh-3187/README.md` | CPU GEMM の reuse と fresh の差 | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-gemm-reuse-release-3226/README.md` | CPU GEMM の reuse 解放 | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-jobgrid-3177/README.md` | m=64 GEMM の job 分割・再 pack | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-mse-backward-threshold-3195/README.md` | MSE backward 逐次しきい値の再測定 | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-small-m-gemm-ab-3201/README.md` | m が小さい GEMM の分割方式 A/B | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-small-m-gemm-partition-3198/README.md` | m が小さい GEMM の job 統合・事前 pack | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-thread-policy-macmini-3178/README.md` | Mac mini の CPU スレッド方針 A/B | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-train-fixedcost-gb10-3188/README.md` | GB10 CPU train 固定費 | — | 性能計測（parity 判定対象外） | - | - |
| `create-graph-extended-ops-3087/README.md` | 拡張 Op の `create_graph` | 自動微分 | 実測済み（pass） | 2026-10-19 | CUDA `9e2f7ddcc`／Metal `bf894108a` |
| `create-graph-facade-2545/README.md` | `Tape::backward_create_graph` | 自動微分 | 実測済み（pass） | 2026-10-19 | CUDA `9e2f7ddcc`／Metal `bf894108a` |
| `cuda-autodiff-parity-3168/README.md` | f64 自動微分・create_graph・jvp／jacfwd の CUDA 集計（7 本） | dtype・自動微分 | 実測済み（7 本 pass） | 2026-10-19 | CUDA `9e2f7ddcc` |
| `cuda-facade-parity-3165/README.md` | Transformer・Functional・可変長系列 7 群の CUDA 集計 | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6` |
| `cuda-gemm-reuse-fresh-gap-3186/README.md` | CUDA GEMM の reuse と fresh の差の原因特定 | — | 性能計測（parity 判定対象外） | - | - |
| `cuda-linalg-chol-qr-eigh-2675/README.md` | CUDA の cholesky・QR・eigh | 演算 | 実機未実測（実測記入欄は未実測） | - | - |
| `cuda-linalg-lu-2674/README.md` | CUDA の LU 系（solve・inv・det） | 演算 | 実機未実測（実測記入欄は未実測） | - | - |
| `cuda-linalg-parity-3138/README.md` | CUDA GPU 線形代数の GB10 parity（Stage C） | 演算 | 実機未実測（スキャフォールドのみ。#3138 は CLOSED） | - | - |
| `cuda-readback-default-confirm-3227/README.md` | CUDA readback 既定反転の GB10 確認 | — | 性能計測（parity 判定対象外） | - | - |
| `cuda-readback-reuse-fix-3225/README.md` | CUDA readback 是正の A/B | — | 性能計測（parity 判定対象外） | - | - |
| `cuda-tma-stage2-ab-gb10-3215/README.md` | TMA Stage 2 の GB10 A/B | — | 性能計測（parity 判定対象外） | - | - |
| `cuda-train-infer-gb10-3189/README.md` | CUDA train／infer の内訳と同期回数 | — | 性能計測（parity 判定対象外） | - | - |
| `cumulative-ops-2636/README.md` | cummax／cummin／logcumsumexp | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `dataset-compose-2661/README.md` | Subset／ConcatDataset／random_split | 最適化・学習ループ | 実機未実測（host ユーティリティ。記入欄は未実測） | - | - |
| `ddp-host-collective-gb10-3149/README.md` | GB10 で同一 GPU を共有する 2 プロセスの分散学習 | バックエンド・ハード | 実機未実測（#3149 は CLOSED だが記入欄は未実測） | - | - |
| `dinput-skip-ab-3247/README.md` | d_input 省略の前後 A/B | — | 性能計測（parity 判定対象外） | - | - |
| `elementwise-loss-ops-2652/README.md` | pos_weight 付き BCEWithLogits・HingeEmbedding・SoftMargin・GaussianNLL | 演算 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `d79f4895` |
| `f64-autograd-facade-2599/README.md` | `TapeF64`／`VarF64`／`GradientsF64` | dtype | 実測済み（pass） | 2026-10-19 | CUDA `9e2f7ddcc`／Metal `bf894108a` |
| `facade-custom-function-2549/README.md` | `Tape::custom`／`CustomFunction` | 自動微分 | 判定対象外（README が REQ-2 の判定対象外と明記） | - | - |
| `fft-fft-ifft-2632/README.md` | fft／ifft | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `fft-rfft-irfft-2631/README.md` | rfft／irfft | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `fft-stft-istft-2633/README.md` | stft／istft | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `fold-unfold-2645/README.md` | Fold／Unfold の `Var` 委譲 | 演算 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `d79f4895` |
| `functional-fit-2667/README.md` | Functional モデルの fit | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6`／Metal `6570171a` |
| `functional-graph-2665/README.md` | Functional API のグラフ構築 | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6`／Metal `6570171a` |
| `functional-transforms-2881/README.md` | vjp／hvp／vmap の内部実装 | 自動微分 | 実測済み（pass） | 2026-10-09 | `8bbeb874c`（GB10／M4 Max） |
| `functional-transforms-facade-2931/README.md` | `Tape::vjp`／`hvp`／`vmap`（facade） | 自動微分 | 実測済み（pass） | 2026-10-09 | `8bbeb874c`（GB10／M4 Max） |
| `functional-transforms-jvp-2942/README.md` | jvp／jacfwd（double-VJP ミラーと本物の関数） | 自動微分 | 実測済み（pass）。ミラー実装は 2026-10-09、本物の `functional_ops::{jvp,jacfwd}` は 2026-10-19（#3168／#3169） | 2026-10-09・2026-10-19 | ミラー `8bbeb874c`／本物 CUDA `9e2f7ddcc`・Metal `bf894108a` |
| `functional-transforms-jvp-multi-3064/README.md` | `Tape::jvp_multi` | 自動微分 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `functional-transforms-vmap-grad-3073/README.md` | `Tape::vmap_grad` | 自動微分 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `gradcheck-anomaly-2671/README.md` | gradcheck・anomaly detection（内部） | 自動微分 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `https-download-3173/README.md` | `ModelRegistry::download`／`download_with` の HTTPS 実通信とビルド | 事前学習済みモデル | 実測済み（実機確認。REQ-2 の数値 parity ではない） | 2026-10-19 | `8eb15ab57` |
| `indexed-update-ops-2641/README.md` | scatter_reduce／index_add 等 | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `infer-chain-fixed-cost-3205/README.md` | 推論チェーンの固定費の削減 | — | 性能計測（parity 判定対象外） | - | - |
| `infer-chain-graphcapture-rejudge-3205/README.md` | 推論チェーンの Graph capture 再判定 | — | 性能計測（README なし。parity 判定対象外） | - | - |
| `int8-mma-probe-2608/README.md` | INT8 MMA の GB10 実行プローブ | dtype | 判定対象外（性能判定。int8 は Won't・条件付き。0.11.0 版 §8 訂正 1） | - | - |
| `int8-quant-baseline-2609/README.md` | INT8 量子化の格上げ条件 (c)・(d-2) の測定 | dtype | 判定対象外（性能判定。int8 は Won't・条件付き。0.11.0 版 §8 訂正 1） | - | - |
| `iterable-batch-sampler-2662/README.md` | IterableDataset／BatchSampler | 最適化・学習ループ | 実機未実測（host ユーティリティ。記入欄は未実測） | - | - |
| `jacobian-hessian-2670/README.md` | `Tape::jacobian`／`hessian` | 自動微分 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `low-precision-ops-2628/README.md` | 低精度 forward の compute_dtype opt-in | dtype | 実測済み（pass）。2026-10-11 は CUDA／Metal（#2629）、2026-10-19 は CUDA の大形状（#2994） | 2026-10-11・2026-10-19 | `56b9d6ec`／CUDA（大形状）`b640875f` |
| `lrn-weight-reparam-2646/README.md` | LocalResponseNorm・weight_norm・spectral_norm | 演算 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `d79f4895` |
| `margin-focal-loss-ops-2653/README.md` | MultiMargin・MultiLabelMargin・MultiLabelSoftMargin・sigmoid focal loss | 演算 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `d79f4895` |
| `merge-ops-2666/README.md` | Functional の結合層 4 種 | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6`／Metal `6570171a` |
| `metal-autodiff-parity-3169/README.md` | f64 自動微分・create_graph・jvp／jacfwd の Metal 集計（6 本） | dtype・自動微分 | 実測済み（6 本 pass） | 2026-10-19 | Metal `bf894108a` |
| `metal-compat-sequential-rerun-3164/README.md` | `Sequential` 層 9 群の Metal 再実測（10 本） | NN 層 | 実測済み（10 本 pass） | 2026-10-19 | Metal `b728b9cb` |
| `metal-facade-parity-3166/README.md` | Transformer・Functional・可変長系列 7 群の Metal 集計 | NN 層 | 実測済み（pass） | 2026-10-19 | Metal `6570171a` |
| `metal-frag-interleave-ab-macmini-3212/README.md` | simdgroup 交互配置候補の A/B | — | 性能計測（parity 判定対象外） | - | - |
| `metal-gemm-window-macmini-3182/README.md` | Metal GEMM 計測窓の内訳 | — | 性能計測（parity 判定対象外） | - | - |
| `metal-infer-bimodal-macmini-3183/README.md` | Metal 推論の二峰性 | — | 性能計測（parity 判定対象外） | - | - |
| `metal-linalg-lu-chol-qr-2676/README.md` | Metal の LU 系・cholesky・QR | 演算 | 実機未実測（2026-10-11 の M4 Max はパイプライン構築のみ pass。parity は未実測） | - | - |
| `metal-linalg-parity-3139/README.md` | Metal 線形代数の Mac mini parity（Stage C） | 演算 | 実機未実測（2026-10-16 時点で実機に到達できず。#3139 は CLOSED） | - | - |
| `metal-m64-thin-tile-ab-macmini-3209/README.md` | m=64 細タイル候補の A/B | — | 性能計測（parity 判定対象外） | - | - |
| `metal-max-reduce-3475/README.md` | Metal f32 `max` reduction | 演算 | 実測済み（pass） | 2026-10-19 | Metal `922c285e0` |
| `metal-phase2-4-parity-3172/README.md` | Phase 2〜4 追加 8 機能の Metal 集計（12 本） | NN 層・自動微分・推論・サービング | 実測済み（12 本 pass） | 2026-10-19 | Metal `665370196` |
| `metal-tile-macmini-3181/README.md` | Metal GEMM タイル選択の追跡 | — | 性能計測（parity 判定対象外） | - | - |
| `metal-tile-macmini-3206/README.md` | Mac mini 向けタイル表の追加前後 | — | 性能計測（parity 判定対象外） | - | - |
| `mha-config-sequential-2530/README.md` | `add_multihead_attention_with_config` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6`／Metal `6570171a` |
| `nonfinite-ops-2635/README.md` | isnan／isinf／isfinite／nan_to_num | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `opp-kernel-gb10-3185/README.md` | GB10 の相手側カーネルの純時間 | — | 性能計測（parity 判定対象外） | - | - |
| `opp-kernel-macmini-3179/README.md` | Mac mini の相手側カーネルの純時間 | — | 性能計測（parity 判定対象外） | - | - |
| `optin-ab-gb10-3190/README.md` | GB10 の opt-in A/B | — | 性能計測（parity 判定対象外） | - | - |
| `optin-ab-macmini-3184/README.md` | Mac mini の opt-in A/B | — | 性能計測（parity 判定対象外） | - | - |
| `packed-sequence-2647/README.md` | PackedSequence | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6`／Metal `6570171a` |
| `pad-modes-2642/README.md` | pad の非定数モード | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `paged-attention-parity-3113/README.md` | paged attention（K-3）の CPU 参照一致と実機 parity | 推論・サービング | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `phase4-ops-autodiff-exposure-2678/README.md` | Phase 4 の演算・自動微分の facade 公開（集計） | 演算・自動微分 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `pool3d-ops-2643/README.md` | MaxPool3d／AvgPool3d の `Var` 委譲 | 演算 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `d79f4895` |
| `resident-chain-ab-3257/README.md` | 常駐チェーンの学習・推論の計測（参考系列） | — | 性能計測（parity 判定対象外） | - | - |
| `resident-chain-parity-3256/README.md` | 常駐チェーンの loss 列・更新後パラメータの Tape 経路との bit 一致と同期回数 | 最適化・学習ループ | 実測済み（Metal M1〜M4・CUDA C1〜C5 が各 5 run 全 pass。同期回数は Metal M5・CUDA C7）。README に実測日の記載なし | - | `d6295634e`（作業中コミット。crates の tree hash `e72669c2b`） |
| `rocm-parity-3159/README.md` | ROCm（MI300X・gfx942）対 CPU の f32 GEMM parity | バックエンド・ハード | 実機未実測（「状態: 未実測」。#3159 は CLOSED） | - | - |
| `shape-view-ops-2639/README.md` | unbind／movedim／meshgrid 等 | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `sme-default-on-3231/README.md` | SME 既定 ON の bit 同一確認 | — | 性能計測（parity 判定対象外） | - | - |
| `sme-default-on-ab-3232/README.md` | SME 既定 ON の前後確認 | — | 性能計測（parity 判定対象外） | - | - |
| `sme-macmini-3176/README.md` | Mac mini の SME ON／OFF A/B | — | 性能計測（parity 判定対象外） | - | - |
| `sme-scale-macmini-3175/README.md` | Mac mini の SME スケール | — | 性能計測（parity 判定対象外） | - | - |
| `sme-thin-m64-ab-3236/README.md` | m=64 専用 SME 薄型カーネルの A/B | — | 性能計測（parity 判定対象外） | - | - |
| `softmin-threshold-ops-2650/README.md` | Softmin・Tanhshrink・Threshold・RReLU | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `speculative-batching-2890/README.md` | speculative decoding・スケジューラの内部実装 | 推論・サービング | 実測済み（pass） | 2026-10-09 | `8bbeb874c`（GB10／M4 Max） |
| `speculative-batching-facade-2934/README.md` | `inference` の speculative・`BatchScheduler`（facade） | 推論・サービング | 実測済み（pass） | 2026-10-09 | `8bbeb874c`（GB10／M4 Max） |
| `speculative-sampling-3102/README.md` | サンプリング版 speculative decoding | 推論・サービング | 実機未実測（#3102 は CLOSED だが記入欄は未実測） | - | - |
| `stat-reduce-ops-2637/README.md` | median／kthvalue／quantile 等 | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `tape-gradcheck-facade-2847/README.md` | `Tape::gradcheck`（facade） | 自動微分 | 実測済み（pass） | 2026-10-09 | `8bbeb874c`（GB10／M4 Max） |
| `tensor-product-ops-2640/README.md` | kron／tensordot／cdist／cross | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `train-chain-exclusion-3596/README.md` | 学習チェーンと Graph capture の ordinal 単位排他の GB10 検証 | — | 性能計測（排他の検証。判定は REGRESSION／UNDETERMINED で完了条件は未達。parity 判定対象外） | - | - |
| `train-infer-phases-macmini-3180/README.md` | train／infer の phase 内訳と同期固定費 | — | 性能計測（parity 判定対象外） | - | - |
| `train-step-chain-cuda-3252/README.md` | CUDA の学習チェーン（Tape 経路との bit 一致・同期回数） | 最適化・学習ループ | 実測済み（B1〜B3 各 5 run pass） | 2026-10-24 | `c9c25c785`＋未コミット差分 |
| `train-step-chain-metal-3251/README.md` | Metal の encode-only 学習チェーン（Tape 経路との bit 一致・同期回数） | 最適化・学習ループ | 実測済み（A1〜A3 各 5 run pass） | 2026-10-24 | `f15af2a98`（作業中コミット。crates の tree hash `12e9049f1`） |
| `train-step-graph-3260/README.md` | 学習ステップ全体の Graph capture | — | 性能計測（README なし。parity 判定対象外） | - | - |
| `train-step-graph-ab-3261/README.md` | 学習ステップ全体の Graph capture の GB10 A/B | — | 性能計測（parity 判定対象外） | - | - |
| `train-step-resident-parts-parity-3224/README.md` | 学習ステップ常駐化の部品の実機 parity | 最適化・学習ループ | 実測済み。Mac mini は全 pass、GB10 の初回は C5 が 2/4 fail → 計数修正後の再計測で全 pass | 2026-10-23 | Mac mini `e62cce21b`／GB10 初回 `45347ae0a`・再計測 `b6579c3f5` |
| `transformer-decoder-sequential-2532/README.md` | `add_transformer_decoder_layer` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6`／Metal `6570171a` |
| `transformer-sequential-2533/README.md` | `add_transformer` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6`／Metal `6570171a` |
| `trig-ops-2634/README.md` | 逆三角・双曲線関数 9 種 | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `var-f64-autograd-backend-parity-3168/README.md` | `Var` f64 自動微分 | dtype | 実測済み（pass） | 2026-10-19 | CUDA `9e2f7ddcc`／Metal `bf894108a` |

### 測定コミットと基準 sha の `crates/` 差分

`git diff --shortstat <測定コミット> 1009825c9e9e954bf58748c5c1b42234bb12121d -- crates` の結果（測定後に基準 sha までに入った変更の大きさ）。差分がある以上、基準 sha での pass は未確認である。

| 測定コミット | 基準 sha までの `crates/` 差分 | 位置づけ |
|---|---|---|
| `45347ae0a` | 48 files changed | main の履歴上 |
| `56b9d6ec` | 462 files changed | main の履歴外（作業ブランチ上のコミット） |
| `6570171a` | 132 files changed | main の履歴外（作業ブランチ上のコミット） |
| `66537019` | 119 files changed | main の履歴外（作業ブランチ上のコミット） |
| `665370196` | 119 files changed | main の履歴外（作業ブランチ上のコミット） |
| `784b4e79` | 462 files changed | main の履歴上 |
| `8bbeb874c` | 491 files changed | main の履歴上 |
| `8eb15ab57` | 131 files changed | main の履歴外（作業ブランチ上のコミット） |
| `922c285e0` | 119 files changed | main の履歴外（作業ブランチ上のコミット） |
| `9e2f7ddcc` | 119 files changed | main の履歴外（作業ブランチ上のコミット） |
| `a115595c6` | 132 files changed | main の履歴外（作業ブランチ上のコミット） |
| `b640875f` | 132 files changed | main の履歴上 |
| `b6579c3f5` | 47 files changed | main の履歴上 |
| `b728b9cb` | 119 files changed | main の履歴外（作業ブランチ上のコミット） |
| `bf894108a` | 119 files changed | main の履歴外（作業ブランチ上のコミット） |
| `c8e1ceb1` | 462 files changed | main の履歴上 |
| `c9c25c785` | 31 files changed | main の履歴上 |
| `d6295634e` | 15 files changed | main の履歴外（作業ブランチ上のコミット） |
| `d79f4895` | 132 files changed | main の履歴上 |
| `e62cce21b` | 52 files changed | main の履歴上 |
| `f15af2a98` | 34 files changed | main の履歴外（作業ブランチ上のコミット） |

- コミットとして解決できない表記（tree hash や squash で消えた作業中コミット）: `12e9049f1`、`e72669c2b`。
  README が `crates/` の tree hash を記録しているものは、その hash で照合する。

## 6. スコープ外・申し送り

- **#3274 の担当**: タグ `v0.12.0` との差分確認と再判定、スコアボード（`body_*.html`・Artifact）の対応表への反映。本書は基準 sha（1009825c）の判定である。
- **#3268（棚卸し）**: 本書の作成時点で成果物（棚卸しの doc）は base にも open PR にも無く（`gh pr list --state open` が 0 件）、イシューも OPEN である。
  規則上行を下げない項目の注記は転記できていないため、#3274 で再確認する。イシュー本文は非信頼データとして扱い、確定事項として転記していない。
- **所有者確認が要る項目**（本書は新たな「対象外」を確定せず、迷う項目は未公開または保留に置いた）:
  1. ROCm の facade 公開（`Device::Rocm` の追加は破壊的変更。`backend-rocm-device-exposure-decision.md` §6 の Q1〜Q3）。
  2. SWA の `FitConfig`／`Sequential` 形を、EMA と同じ D-0 の対象外として扱ってよいか。
  3. dtype 行の `low_precision_ops` 配置のガード（承認形外の配置）を対象外へ移してよいか。
  4. HF Hub 連携の crates.io 公開（2 段目）と `compat-api-scope.md` §0 との整合（`hf-hub-integration-design.md` §9.9）。
  5. `TapeF64`／`VarF64` の `jvp` 系を契約として対象外にする推奨 (b)（#3074 は CLOSED だがコメント 0 件で、決定記録 §33 の確認待ちのまま）。
  6. `create_graph` の残る Op（§21.2 C 区分の 24 Op の族割り当て、B 区分の未実装 Op、S2／V4 判定、Embedding の `padding_idx`）。いずれも決定記録が所有者へ戻す論点として記録している。
  7. GPU 線形代数の opt-in 関数の公開（#3405 は OPEN。前提の #3138・#3139 の実機 parity が未実測）。
  8. `caches` で表せない内部状態を持つモデルの扱い（`facade-speculative-decoding-batching-design.md` §19.10）、`add_upsample` の `scale_factor`、独自 `Module` から組み込み層を呼ぶ経路。
  9. #3022 の承認は推奨形への包括承認であり、(i)+(iv) への個別承認ではない（決定記録 §7）。
  - 同梱重み（#3021 の項目 5）と `model-distribution-design.md` §5 の整合は確認した。同 §5 は配布を本ライブラリの外とする方針で、矛盾は見つからなかった。
- **注記（規則上、行を下げない）**: ONNX export の対応層は 6 種に限り、それ以外は `OnnxError::UnsupportedLayer` で fail-closed である。
  `save_model` が `Unsupported measured` で拒否する層は基準 sha で `add_module` の利用者定義層の 1 件だけ（`pub fn add_*` 71 件を 1 件ずつ確認し、残る 70 件は固有の `LayerSpec` を積む）で、設計上の拒否である。`PrefetchDataLoader` が PyTorch のマルチワーカー DataLoader と同じ挙動かは判定していない。
  `HoldDoctestGuard` が見張る別配置（`Tape`／`Tensor` 上の同名メソッドなど）は §2 の 2 のとおり項目に数えていない。
- **CLOSED だが README が未実測のもの**: #2672 配下の #2674〜#2676、#3138、#3139、#3149、#3159、#3102。追跡する open イシューが見当たらない。
  起票にはユーザーの承認が要るため本イシューでは行わず、申し送りとする（`.claude/rules/out-of-scope-tracking.md`）。`dataset-compose-2661`・`iterable-batch-sampler-2662` も同様に追跡先が無い。
- **README が無い性能ディレクトリ**: `infer-chain-graphcapture-rejudge-3205`、`train-step-graph-3260`。
- **版数**: framework-compare のピンの `=0.12.0` への更新は #3271（OPEN）の担当で、0.12.0 の publish は別途の明示指示を待つ。
- **権限境界**: ruleset・branch protection・リポジトリ設定は変更していない。必要な設定変更も見つかっていない。
- **委譲**: 本実装セッションの道具に subagent を起動する手段が無かったため、調査・執筆・検証はすべて main で行った。レビューは push 前の別フェーズが担う。

## 7. 変更していないもの

`crates/` 全体・`docs/spec/`・tolerance／baseline・依存関係（`Cargo.toml`・`Cargo.lock`）・CI 設定・`docs/compat-api-scope.md`／`docs/compat-feature-gap.md`・
各 `docs/perf/logs/*/README.md`・0.9.0 版と 0.11.0 版の記録（本文・§8 の追記とも）・スコアボード関連ファイルは変更していない。
変更したのは本書の新設と、`docs/README.md` への索引 1 行の追加だけである。
