# 監査に未掲載の保留・torch.nn との差分・相互運用の拡張候補の棚卸し（イシュー #3268）

親: #3267（Phase 12）・ルート: #3018。本書は**棚卸し表と追加起票の提案**までで、コード・依存・tolerance・`docs/spec` は変えない。新規 issue の起票は所有者の承認後に人間が `create-issue` で行う（`.claude/rules/out-of-scope-tracking.md`）。本書の作成時点で新規 issue は起票していない。

## 1. 目的と基準面

- 対応表の監査（`docs/perf/framework-compare-feature-matrix-0.11.0.md`）の「未公開」「保留」欄に載らない項目を先に洗い出し、次回の監査基準で拾われる前に追跡先を決める。行の判定には影響しない。
- 基準面: `origin/main` の `1009825c9`（2026-10-27 確認）。出典はすべてこの commit の `path:line`。
- 入力: issue #3268 の列挙のみ。調査報告 B・C の原文（`feature-B-nn-optim-interop`・`feature-C-backend-pretrained-serving`）はリポ内に無い。報告 B・C が issue に列挙されていない項目を挙げていた場合、本書は拾えていない。
- issue 本文の数値は検証前の入力として扱い、下表の数値は HEAD で数え直した値だけを使う（§7 に照合結果）。

## 2. 判定規則

- 区分: 公開／未公開／保留／対象外。
- 「対象外」は 2 種類に分ける。(O-1) 所有者が確定したもの（コメント URL を出典にする）。(O-2) 決定記録に「その issue のスコープ外」と書かれているだけで、所有者の Won't ではないもの。O-2 は次回の監査で拾われうるため、本書では「未掲載の候補」として扱う。
- D-0（#3021 確認表・2026-10-11 確定）: 機能が別名で公開済みなら、名前だけの差は対象外へ分類し直す。層型名の公開方針は #3022 で扱う。
- 提案: 起票する／対象外に分類する／既存 issue でカバー済み。
- 必要な承認: 公開形（新しい公開名・シグネチャ）／数値契約／不要（記述の訂正）／spec 変更。

## 3. 本ツリー（#3018 配下）でカバー済みのため除いた項目

下表は #3018 の全子孫 276 件（Phase 1〜12。GraphQL で取得）のタイトルと照合した結果。

| 項目 | カバーする issue | 備考 |
|---|---|---|
| ストリーミング生成・beam search・top-p・EOS 停止・repetition penalty | #3021 項目 7（所有者回答 <https://github.com/Fandhe-AI/fandhe-ai/issues/3021#issuecomment-6111746880>、2026-10-12「不算入」確定） | 記録先は `docs/compat-api-scope.md:2757`（#3021 の確認表の項目 7）。出典食い違いは §7 R4 |
| f16／bf16 を値型とする `Tape`／`Var`・`jvp` の f16 | #3021 項目 2・3（同コメント） | safetensors ファイルの dtype（B5）は別軸で、未カバー |
| 生成のバッチ化・KV キャッシュを進めるモデル・speculative sampling・paged attention | #3103・#3105・#3094・#3099・#3108 | 報告 C の行外の保留のうち、これらは Phase 4 で対応済み |
| 3D プーリング層（MaxPool3d・AvgPool3d）・ConvTranspose3d・MaxUnpool・Fold／Unfold・LocalResponseNorm の層化 | #3025〜#3045（#3026・#3030・#3034・#3038・#3042） | AdaptiveAvg／MaxPool3d は含まない（C9） |
| `save_model` 非対応の 11 層 | #3051 | 層の追加ではなく kind の追加 |
| SWA の `fit` 結線 | #3056 | EMA とは別 |
| `Tape::jvp` の複数入力・微分可能 jvp・`vmap(grad)`・`create_graph` の対象 Op 拡大 | #3062・#3065・#3069・#3075（BatchNorm は #3079、Pad は #3084） | `BatchNorm3d` の層とは無関係 |
| 層型名の公開方針 | #3022 | 下表の C 群のうち層ラッパーは本方針に従属 |

最適化（EMA×AMP・warmup・param groups の保留）・ONNX autograd・ONNX export の層／op 拡張・safetensors の dtype・C 群の層は、#3018 配下のどの issue のタイトルにも現れない（照合語: ema・amp・param group・warmup・autograd・onnx・export・sigmoid・safetensors・dtype・shrink・bilinear・batchnorm・pad・cosine・pairwise・shuffle・softmax2d・lppool・fractional・3d・stream・lbfgs・DeviceParamStore）。

## 4. 棚卸し表

凡例: 区分は §2。「承認」は必要な承認の種類。

### 4.1 群 A: 最適化の保留

| 番号 | 項目 | 出典 | 区分 | 提案 | 承認 | カバー |
|---|---|---|---|---|---|---|
| A1 | `Callback::Ema` と `compile_with_amp` の併用拒否。AMP の skip step を「拒否のまま」にするか「skip に追従して更新しない」にするかが未決 | `crates/facade/src/compat/training.rs:2817`・`docs/autodiff-ema-decision.md:258-261`（§10.2 (g)） | 保留 | 起票する（決定記録 issue。A3 の「AMP 併用 API」と同じ論点なので 1 本にまとめる） | 公開形（併用時の挙動） | なし |
| A2 | EMA の decay ウォームアップ（TensorFlow 方式 `min(decay, (1 + n) / (10 + n))`）。`num_updates` は呼び出し回数のカウンタのみ | `docs/autodiff-ema-decision.md:60,183,268,515` | 保留（O-2） | 起票する（`EmaCallback` への任意設定として）。ただし利用者要望が無ければ対象外に分類する選択肢もある | 公開形（設定の追加）。`FitConfig` に足すと `Eq` が外れるため `EmaCallback` 側に限る（`docs/autodiff-ema-decision.md` §10.2 (d)、#3021 本文） | なし |
| A3-a | param groups: AMP 併用 API | `docs/compat-api-scope.md:1841-1842`・`docs/autodiff-param-groups-decision.md:486-490`（§12.6）・同 §13.5 | 保留 | A1 に統合して起票する | 公開形 | なし |
| A3-b | param groups: `LrSchedule` 併用（案 A） | 同上 | 保留 | 起票する（決定記録 issue） | 公開形 | なし |
| A3-c | param groups: groups の保存形式（`state_dict` への書き出し） | 同上・`docs/autodiff-param-groups-decision.md` §12.4（要判断事項） | 保留 | A3-b とまとめて起票する | 公開形（保存形式は互換性に関わる） | なし |
| A3-d | param groups: `DeviceParamStore` 常駐経路 | 同上 | 保留 | 起票する（GPU 常駐更新の経路。実機 parity が要る） | 公開形・数値契約（bit 一致の維持を確認） | なし |
| A3-e | param groups: `Lbfgs` 併用 | 同上 | 保留 | 対象外に分類する案を推奨（L-BFGS は単一のパラメータ集合を前提にする方式のため）。所有者が併用を望む場合のみ起票する | 公開形 | なし |
| A3-f | param groups: スロット添字の公開ヘルパー | `docs/compat-api-scope.md:1842`・`docs/autodiff-param-groups-decision.md` §13.4（`named_parameters()` の位置が契約。利用例テスト `named_parameters_position_is_slot_index_for_groups`） | 保留 | 対象外に分類する（契約で代替済み） | 不要（所有者の確認のみ） | なし |
| A3-g | `compile()` の `Optimizer` enum に無い 4 種（Adadelta・Adamax・NAdam・RAdam）の経路。issue 本文に無いが同じ decision doc に載る | `docs/autodiff-param-groups-decision.md` §12.6・§13.5。`Optimizer` enum は 7 種（Sgd・AdamW・Adam・RmsProp・Adagrad・Lamb・Lbfgs）（`crates/facade/src/compat/training.rs:240-`） | 保留 | 起票する | 公開形（`Optimizer` の variant 追加） | なし |

### 4.2 群 B: 相互運用の拡張候補

| 番号 | 項目 | 出典 | 区分 | 提案 | 承認 | カバー |
|---|---|---|---|---|---|---|
| B1 | ONNX autograd（`onnx::autograd` の `bind` ほか）の facade 公開。内部クレートでは実装済み（#2078 CLOSED）だが、facade の ONNX 面は「autograd 未接続」と明記 | `docs/onnx-autograd-decision.md:131-134`（§6 項 1）・`crates/onnx-interop/src/onnx/autograd.rs:226`・`crates/facade/src/interop/onnx.rs:8` | 未公開 | 起票する（公開形の決定記録 issue） | 公開形。あわせて §6 項 2（`Gather` ほか 8 op を autograd 未実装のままにする縮小）と項 3（`Erf`／`Softmax`／`LayerNormalization` は GPU 上でもホスト実行）の受容 | なし |
| B2 | ONNX export の層拡張。facade が export できる層は 6 種（`Linear`・`ReLU`・`Softmax`・`LayerNorm`・`GELU`〈erf 版〉・`Conv2d`）。`Tanh`・`GeluTanh`・`LogSoftmax`・`Conv1d` 等は `UnsupportedLayer` | `crates/facade/src/interop/onnx.rs:97-101,322-329`。`compat::Sequential` の `pub fn add_*` は 72 種（§7 R3） | 未公開 | 起票する（層ごとではなく、B4 の op 追加とセットで 1 本） | 数値契約（新しい層が既存の複合判定で検証できるかの確認）。公開面の増減は無い（`UnsupportedLayer` で拒否していた層が通るようになる） | なし |
| B3 | `Sigmoid` の export（数値契約が承認保留）。`predict` は `x >= 0` の 2 分岐、interp は全域 `1/(1+exp(-x))` で、負入力の丸めが異なり bit 一致を保証できない | `docs/facade-onnx-export-exposure-decision.md:239`（F9）・同 §15.7 項 5。exporter 側の `Sigmoid` op は対応済み（`crates/onnx-interop/src/onnx/export_ops.rs:202`） | 保留 | 起票する（選択肢 α〈既存の統一複合判定で検証〉の承認を得て解除。tolerance は変えない） | 数値契約 | なし |
| B4 | export allowlist と import の非対称。import 36 op に対し export は 26 op。差の 10 op は `AveragePool`・`Clip`・`Expand`・`Gelu`・`MaxPool`・`Pad`・`ReduceMean`・`Resize`・`Tanh`・`Where`（§7 R1 の機械照合） | `crates/onnx-interop/src/onnx/export_ops.rs:193-220`（`SUPPORTED_OP_TYPES`）・`crates/onnx-interop/src/onnx/interp.rs` の op アーム 36 個（`export_ops.rs:4` も 36 op と記述） | 未公開 | B2 に統合して起票する。export 側の op 実装が先、層の解禁（B2）が後 | 数値契約（各 op の bit 一致の可否） | なし |
| B5 | safetensors の F32 以外の dtype（f16 等）。`save`／`load` は `Tensor<f32>` 専用。`st_save.rs` が参照する追跡先 #274 は **CLOSED**（タイトルに「追加 dtype」を含むが close 済み）で、現在は追跡先が無い | `crates/onnx-interop/src/st_save.rs:19-20`・`docs/facade-safetensors-exposure-decision.md:66`（§7）・同 `:168-`（§11.3） | 未公開 | 起票する、または対象外に分類する。所有者の判断を仰ぐ（#3021 項目 2 は `Tape`／`Var` の値型の話で、ファイル dtype は含まない） | 公開形（API の型が `Tensor<f32>` に縛られているため、dtype 別の入口が要る） | なし |

### 4.3 群 C: `torch.nn` との差分

(a) autodiff に `Var`／`Op` がある。(b) `fandhe_ai::nn` から再エクスポートされている（facade の `nn` が出す層型は `Transformer` 系 3 型のみ。`crates/facade/src/nn/mod.rs:50`）。(c) `Sequential::add_*` がある。(d) 別名（D-0）で到達できる。

| 番号 | 項目 | (a) | (b) | (c) | (d) | 出典 | 区分 | 提案 | 承認 | カバー |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | `Hardshrink`・`Softshrink` | なし（`shrink` を含むのは `tanhshrink` のみ。`crates/autodiff/src/softmin_threshold_ops.rs:71`） | なし | なし（`add_tanhshrink` のみ。`crates/facade/src/compat/sequential.rs:1132`） | なし。`Var::threshold` や `where_cond` で合成はできるが、同名の機能ではない | grep（`crates` 全体で `Hardshrink`／`Softshrink` は 0 件） | 未公開 | 起票する（`Var` の演算 2 件＋`add_*`） | 公開形 | なし |
| C2 | `Bilinear`（`nn.Bilinear`。2 入力の双線形変換層） | なし | なし | なし | なし | `bilinear` を含むファイルは補間（`crates/tensor-core/src/interpolate.rs`・`InterpolateMode::Bilinear`）のみで、`nn.Bilinear` ではない | 未公開 | 起票する（優先度は低い） | 公開形 | なし |
| C3 | `BatchNorm3d` | 一部。`Var::batch_norm` は rank 2〜4（`crates/tensor-core/src/ops_shape.rs:798-806` の `batch_norm_layout`）。rank 5 は対象外と明記（`crates/autodiff/src/nn/batch_norm.rs:47`） | なし | なし（`add_batch_norm1d`／`2d` のみ） | なし | 同左。層の rank 制限は `batch_norm.rs:66-70` | 未公開（O-2。`docs/batch-norm-ops-design.md` §7 が「本 issue の対象外」とするのみ） | 起票する（`Var::batch_norm` の rank 5 対応が先） | 公開形・数値契約（rank 5 の縮約順序の確認） | なし |
| C4 | `ReflectionPad`／`ReplicationPad`／`CircularPad` の層 | あり。`Var::pad_with_mode`（`crates/autodiff/src/var.rs:5431`）と `PadMode`（`Reflect`／`Replicate`／`Circular`。`crates/facade/src/lib.rs:318` で再エクスポート済み。#2678 で公開形承認済み） | なし | なし（`add_zero_pad2d` のみ。`sequential.rs:2451`） | あり。機能は `Var::pad_with_mode` で到達できる | 同左 | 公開済み（機能）／層ラッパーは未公開 | 対象外に分類する（D-0。機能は到達可能）。`Sequential` 用の `add_*` は #3022 の方針が決まった後に判断する | 不要（所有者の確認のみ） | #3022 に従属 |
| C5 | `CosineSimilarity` | なし（`cosine_embedding_loss` のみ。`crates/autodiff/src/var.rs:4733`） | なし | なし | なし。`mul`／`sum`／`sqrt`／`div` で合成はできるが、同名の機能ではない | `cosine_similarity` は `crates` 全体で 0 件 | 未公開 | 起票する（`Var` の演算＋`add_*` は不要。`nn.CosineSimilarity` は 2 入力のため `Sequential` に載らない） | 公開形 | なし |
| C6 | `PairwiseDistance` | 内部のみ。`triplet_margin_loss` の内部計算（`crates/autodiff/src/loss_ops.rs:212,692`） | なし | なし | なし | 同左 | 未公開 | 起票する（C5 と同じ issue にまとめる） | 公開形 | なし |
| C7 | `ChannelShuffle` | なし | なし | なし | なし。`reshape`／`permute` で合成できる | 名前の grep は 0 件 | 未公開 | 起票する（優先度は低い。`add_pixel_shuffle` と同型） | 公開形 | なし |
| C8 | `Softmax2d` | あり（`softmax`） | なし | あり。`add_softmax(dim)`（`sequential.rs:939`）で、チャネル軸（4 次元入力なら 1）を指定すれば同じ | あり | 名前の grep は 0 件 | 公開済み（別名） | 対象外に分類する（D-0） | 不要（所有者の確認のみ） | #3021 項目 1 と同方針 |
| C9 | `LPPool`・`FractionalMaxPool`・`AdaptiveAvgPool3d`・`AdaptiveMaxPool3d` | なし（`adaptive_avg_pool1d`／`2d`・`adaptive_max_pool1d`／`2d` のみ。`var.rs:4126-4202`） | なし | なし | なし | `docs/autodiff-pool3d-ops-decision.md:170`（§8）が当該 issue のスコープ外とした項目。所有者の Won't ではない（O-2） | 未公開 | 起票する（優先度順: Adaptive 3D → LPPool → FractionalMaxPool〈乱数を伴い parity の設計が別途要る〉） | 公開形・数値契約 | なし（#3026 は MaxPool3d／AvgPool3d のみ） |

### 4.4 群 D: 報告 C の行外の保留

| 番号 | 項目 | 出典 | 区分 | 提案 | 承認 | カバー |
|---|---|---|---|---|---|---|
| D1 | ストリーミング生成 | `docs/compat-api-scope.md:2757` | 対象外（O-1） | 既存 issue でカバー済み（§3） | 不要 | #3021 項目 7 |
| D2 | 報告 C のその他の項目 | 報告 C の原文がリポに無いため、issue 本文に列挙された範囲だけを確認した | - | 追加の項目が見つかれば本書へ追記する | - | - |

## 5. 所有者への提示表

答え方は #3021 と同じ。「N 承認」／「N 変更: 理由」。全部同意なら「全て推奨どおり」。承認された項目だけ `create-issue` で起票する（親は #3018 配下の新しい Phase 親を `update-issue-tree` で提案する）。

### (i) 承認が要るもの（公開形・数値契約）

| 番号 | 項目 | 推奨 | 根拠 |
|---|---|---|---|
| 1 | A1＋A3-a: EMA×AMP の併用方針（拒否のまま／skip 追従） | 起票（決定記録）。推奨は skip 追従だが、決定記録の中で比較して決める | `training.rs:2817`・`docs/autodiff-ema-decision.md:258` |
| 2 | A2: EMA の decay ウォームアップ | 起票（`EmaCallback` の任意設定） | `docs/autodiff-ema-decision.md:60` |
| 3 | A3-b＋A3-c＋A3-d: param groups の `LrSchedule` 併用・保存形式・`DeviceParamStore` 常駐 | 起票（決定記録 1 本。実装は承認後に分割） | `docs/compat-api-scope.md:1841` |
| 4 | A3-e: param groups と `Lbfgs` の併用 | 対象外に分類 | L-BFGS は単一のパラメータ集合が前提 |
| 5 | A3-g: `Optimizer` enum への 4 種追加 | 起票 | `docs/autodiff-param-groups-decision.md` §12.6 |
| 6 | B1: ONNX autograd の facade 公開 | 起票（公開形の決定記録） | `docs/onnx-autograd-decision.md` §6 |
| 7 | B2＋B4: ONNX export の op・層拡張 | 起票（op 実装→層解禁の順） | `export_ops.rs:193-220`・`interop/onnx.rs:97-101` |
| 8 | B3: `Sigmoid` の export（選択肢 α＝既存の統一複合判定。tolerance は不変） | 起票 | `docs/facade-onnx-export-exposure-decision.md:239` |
| 9 | B5: safetensors の F32 以外の dtype | 起票するか対象外にするかを選ぶ（要望が無ければ対象外） | `st_save.rs:19-20`（追跡先 #274 は CLOSED） |
| 10 | C1: `Hardshrink`・`Softshrink` | 起票 | grep 0 件 |
| 11 | C2: `Bilinear` | 起票（低優先） | grep 0 件（補間のみ） |
| 12 | C3: `BatchNorm3d` | 起票 | `batch_norm.rs:47` |
| 13 | C5＋C6: `CosineSimilarity`・`PairwiseDistance` | 起票（1 本にまとめる） | `loss_ops.rs:212,692` |
| 14 | C7: `ChannelShuffle` | 起票（低優先） | grep 0 件 |
| 15 | C9: `AdaptiveAvgPool3d`・`AdaptiveMaxPool3d`・`LPPool`・`FractionalMaxPool` | 起票（Adaptive 3D を先に） | `docs/autodiff-pool3d-ops-decision.md:170` |
| 16 | C4: Reflection／Replication／Circular Pad の層 | 対象外に分類（機能は `Var::pad_with_mode` で到達。`add_*` は #3022 の後に判断） | D-0 |
| 17 | C8: `Softmax2d` | 対象外に分類（`add_softmax(dim)` で同じ） | D-0 |
| 18 | A3-f: スロット添字の公開ヘルパー | 対象外に分類（`named_parameters()` の位置を契約とする方針で代替済み） | `docs/autodiff-param-groups-decision.md` §13.4 |

項目 4・16・17・18 は「対象外に分類」の確認で、公開面は増えない。それ以外は、承認後に起票した issue の中で改めて公開形の承認を取る。

### (ii) 承認が要らないもの（記述の訂正。別 PR で扱える）

| 番号 | 内容 | 出典 |
|---|---|---|
| R1 | `export_ops.rs` の `SUPPORTED_OP_TYPES` の doc コメントが「`interp.rs` の import 対応 28 op のうち `MaxPool`／`AveragePool` を除いたもの」のまま。実際は import 36 op、差は 10 op（§4.2 B4）で、同ファイル冒頭（4 行目）の「36 op」と食い違う | `crates/onnx-interop/src/onnx/export_ops.rs:189-192` |
| R2 | `docs/facade-onnx-export-exposure-decision.md` §15.3 の「初期対応 3 種」は #2076 前の記録。現状は 6 種（`crates/facade/src/interop/onnx.rs:97`）。履歴として残す記述なので、追記するなら「現状は §14 以降を参照」の 1 行 | `docs/facade-onnx-export-exposure-decision.md:287-288` |
| R3 | issue #3268 本文の「`Sequential` 64 種」は HEAD と合わない。`crates/facade/src` の `pub fn add_*` は一意な名前で 72 個（`add_module`・`add_low_precision`・`add_multihead_attention_with_config` など層以外の補助を含む）。「何を層と数えるか」を決めてから数え直す | `grep -rhoE "pub fn add_[a-z0-9_]+" crates/facade/src` |
| R4 | `docs/compat-api-scope.md:2757` は項目 7 の出典に `docs/facade-generate-decision.md` §11 を挙げるが、同 doc に「ストリーミング」の語は無い（beam／top-p／EOS／repetition は §11 にある）。#3023 が `compat-api-scope.md` §5 を直すときに合わせて訂正する | `docs/compat-api-scope.md:2757` |

R1 は 1 行のコメント修正で足りる（コードの挙動は変わらない）。本 PR には含めない（本 issue はコードを変えない）。

### (iii) spec 側（`Fandhe-AI/fandhe-ai-spec`）への提案が要るもの

なし。B3 の選択肢 α は REQ-2 の既存の統一複合判定を使うので、判定式・tolerance 定数は変えない。要件・受け入れ基準の修正が必要になった項目は見つかっていない。

## 6. 限界と申し送り

- 報告 B・C の原文が無いため、issue に列挙されていない項目は拾えていない（§1）。
- C 群の「別名の有無」は名前・語幹の grep と `Sequential::add_*` 一覧、`fandhe_ai::nn` の再エクスポートで確認した。facade の `nn` が層型を再エクスポートしない方針は #3022 で決まるため、(b) 列はその結果で変わりうる。
- 0.12.0 の監査記録は #3269 が作る。本書は監査記録を書き換えない。
- 起票は承認後に人間が行う。親 issue の付け替えは `update-issue-tree` で提案する。

## 7. 数値・出典の照合結果（HEAD `1009825c9`）

| 項目 | issue 本文 | HEAD で数えた値 | 手順 |
|---|---|---|---|
| facade が export できる層 | 6 層 | 6 種（一致） | `crates/facade/src/interop/onnx.rs:97-99` |
| import 対応 op | 36 | 36（一致） | `interp.rs` の `"Op" =>` アームを抽出 |
| export allowlist | 26 | 26（一致） | `SUPPORTED_OP_TYPES` を抽出 |
| 非対称の op | 10 | 10（一致。`AveragePool`・`Clip`・`Expand`・`Gelu`・`MaxPool`・`Pad`・`ReduceMean`・`Resize`・`Tanh`・`Where`。export にあって import に無い op は 0） | 2 集合の差（`comm`） |
| `Sequential` の層数 | 64 | 72（一意な `pub fn add_*`）。数え方は R3 | grep |
| `export_ops.rs` のコメント | - | 「28 op」は陳腐化（R1） | `export_ops.rs:190` |
