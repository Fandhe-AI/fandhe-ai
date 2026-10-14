# 3D プーリング・ConvTranspose3d・MaxUnpool・Fold／Unfold・LRN の層化の公開形（決定記録）

イシュー #3025・親 #3019・ルート #3018。コード変更なしの決定記録。後続 issue（#3026〜#3055 の層化ツリー）は本記録を正として実装する。

## 0. 結論

- `Var` 委譲（#2850・#2851 で公開済み）に対し、層 struct・`Sequential::add_*`・`save_model` kind を追加する形を本書で確定する。2026-10-11 の所有者判断 D-A（推奨形で一括承認。記録に形が無い点だけ承認依頼へ戻す。`compat-api-scope.md` §5 末尾の適用記録）に基づく。本書は記録の確定であり新規承認ではない。
- 公開面の追加は **`Sequential::add_*` の 6 メソッド**と **`FunctionalBuilder` の 6 メソッド**のみ（追加のみ・既存シグネチャ不変）。`LayerSpec` は `pub(super)`（`crates/facade/src/compat/sequential.rs:192`）で公開型ではないため、variant 追加は公開 API に影響しない。
- 層型名は `fandhe_ai::nn::*` へ出さない（#3022・`facade-nn-layer-type-exposure-decision.md`）。
- **MaxUnpool は `Sequential` に層化せず、`FunctionalBuilder` 経由とする**（§6）。
- 保存形式は `format_version: 2` 据え置き（§9）。

## 1. 背景と先例

- `Module::forward` は単入力・単出力（`crates/autodiff/src/nn/module.rs:110`）。`FunctionalBuilder` のノード値は `Var<'t>`（f32）のみで、`Tensor<i32>` 索引は流せない（`crates/facade/src/compat/functional.rs` の `NodeDef`・`eval_graph`）。
- 常駐経路は `contains_resident_unsupported_layer`（`sequential.rs`）で拒否する。パラメータ持ち層は `first_untracked_parametric_layer` の `as_*` 列挙にも入れないと独自層扱いで fit が拒否される。
- manifest の JSON object は 1 つにつきキー 16 個まで（`MAX_OBJECT_KEYS = 16`・`crates/facade/src/compat/model_io.rs:147`）。`params` 内に配列・object は置けない（`MAX_JSON_DEPTH = 4`・同 136）。
- 先例: Conv3d（#2718）・PixelShuffle（#2720）・`format_version` 据え置き（`compat-model-io-decision.md` §4・追補群）。

## 2. 共通規約

- 固定値（manifest に書かない）: `ceil_mode=false`・`bias=true`・`seed`（`add_conv_transpose3d` のみ seed 引数あり。保存しない。重みは `load_state_dict` で上書き）。
- AMP 低精度は無視して f32 で forward する（`add_conv3d` と同じ）。
- tape 不要の推論経路は全新規層で `supports_forward_host() = false`（`Mish`・`Glu`・`PRelu` の先例。tape 経路との bit 一致を未検証のまま副作用を二重化しない安全側）。bit 一致を検証できた層のみ後続 issue が個別に有効化してよい。
- ONNX export は新規 6 層とも既存どおり `UnsupportedLayer`（fail-closed。拡張はスコープ外）。exhaustive match のコンパイルと拒否テストで確認する。
- Sequential の新 kind は 6 個（`max_pool3d`・`avg_pool3d`・`conv_transpose3d`・`unfold`・`fold`・`local_response_norm`）。kind allowlist の件数アサーション（`crates/facade/src/compat/model_io.rs` のテストの `assert_eq!(kinds.len(), 53, ...)`〈3861 行付近〉、および `crates/facade/tests/compat_sequential_model_io.rs`・`compat_sequential_model_io_layers.rs` の `ALL_KINDS: [&str; 53]`）は並行 issue で変動するため、件数でなく名前集合の包含で書く、または実装時点の main に合わせて更新することを後続へ申し送る。

## 3. グループ別公開形（1 行 1 層）

| 層 | autodiff 層 struct（置き場） | `add_*`（引数順） | `LayerSpec` variant | kind とキー数 | 常駐経路 | 反転するガード範囲 |
|---|---|---|---|---|---|---|
| MaxPool3d | `nn::MaxPool3d::new(kernel_size [usize;3], stride Option<[usize;3]>, padding, dilation)`（`nn/pooling.rs`。`Pool3dParams::new` の検査を前倒し） | `add_max_pool3d(kernel_size, stride, padding, dilation)`（`add_max_pool2d` と同流儀。値のみ返し索引は返さない） | `MaxPool3d` | `max_pool3d`: kernel_{d,h,w}・stride_{d,h,w}（3 つ全て null か全て整数）・padding_{d,h,w}・dilation_{d,h,w} = 12 | `is_pooling() = true` で拒否（既存 pooling 9 型と同じ） | `Pool3dOpsHoldDoctestGuard` の `__FandhePool3dHoldSequentialProbe` trait／impl と `Sequential::add_*` の UFCS 2 行のみ |
| AvgPool3d | `nn::AvgPool3d::new(kernel_size, stride, padding, count_include_pad)` | `add_avg_pool3d(kernel_size, stride, padding, count_include_pad)` | `AvgPool3d` | `avg_pool3d`: kernel×3・stride×3（null 規則同上）・padding×3・count_include_pad = 10 | 同上 | 同上 |
| ConvTranspose3d | `nn::ConvTranspose3d`（`nn/conv.rs`。`ConvTranspose2d` と同型。`Module` 実装＋`as_conv_transpose3d` フック） | `add_conv_transpose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation, groups, seed)`（`add_conv_transpose2d` の 3 軸版。`#[allow(clippy::too_many_arguments)]`） | `ConvTranspose3d` | `conv_transpose3d`: §4 のとおり 8 キー。tensor は `weight [in, out/groups, kD,kH,kW]`・`bias [out]`（`{i}.weight`／`{i}.bias`） | `as_conv_transpose3d` を `contains_resident_unsupported_layer` と `first_untracked_parametric_layer` の両方へ追加し拒否。`bind`・`trainable_parameters`・`apply_parameters` は Conv3d と同型に結線 | `ConvTranspose3dMaxUnpoolHoldDoctestGuard` の `add_conv_transpose3d` 1 行のみ |
| Unfold | `nn::Unfold::new(kernel_size [usize;2], stride, padding, dilation)`（新規 `nn/fold.rs`） | `add_unfold(kernel_size, stride, padding, dilation)`（引数順は `autodiff-fold-unfold-decision.md` §12.2 のとおり。PyTorch の dilation, padding, stride 順は採らない） | `Unfold` | `unfold`: kernel_{h,w}・stride_{h,w}・padding_{h,w}・dilation_{h,w} = 8 | §5 のとおり拒否 | `FoldUnfoldHoldDoctestGuard` の `add_unfold` 分 |
| Fold | `nn::Fold::new(output_size, kernel_size, stride, padding, dilation)` | `add_fold(output_size, kernel_size, stride, padding, dilation)` | `Fold` | `fold`: output_size_{h,w} + 上記 8 = 10 | 同上 | 同ガードの `add_fold` 分 |
| LocalResponseNorm | `nn::LocalResponseNorm::new(size, alpha, beta, k)`（新規 `nn/lrn.rs`。`LrnParams::new` の検査を前倒し。PyTorch の既定値は設けず全引数明示） | `add_local_response_norm(size, alpha, beta, k)` | `LocalResponseNorm` | `local_response_norm`: size・alpha・beta・k（f32 は既存の正準形。保存時に非有限は `UnsupportedModel`）= 4 | §5 のとおり拒否 | `LrnWeightReparamHoldDoctestGuard` の `add_local_response_norm` 1 行のみ。`add_weight_norm`／`add_spectral_norm` は #3046・#3047 まで保留維持 |
| MaxUnpool1d／2d／3d | `nn::MaxUnpool{1,2,3}d::new(kernel_size, stride, padding)`（新規 `nn/max_unpool.rs`。`Module` は実装しない。固有メソッド `forward(&self, input, indices, output_size)`） | `Sequential::add_max_unpool*` は設けない（§6） | なし | Sequential の kind なし。Functional manifest の op として §6 | Functional に常駐経路なし（該当なし） | `Sequential::add_max_unpool1d/2d/3d` の 3 プローブは維持（公開しない経路として確定）。正ガードは `FunctionalBuilder` 新メソッド分を `facade_exposes_functional_api_only_in_approved_shape` のインベントリへ追加 |

保留ガード運用: 反転するのは `Sequential::add_*` の UFCS 行と対応する `__Fandhe*SequentialProbe` trait／impl だけ。`Tape`／`Tensor<f32>` 上の同名メソッド・モジュール再エクスポート・層型名のプローブと `api_surface.rs` の否定ガードは維持し、公開した名前の正ガード（完全一致）を同時に足す。

## 4. conv_transpose3d の manifest キー数問題

- 平坦形で書くと 18 キー（in・out・kernel×3・stride×3・padding×3・output_padding×3・dilation×3・groups）となり、書けても読めない manifest になる（`conv3d` は 15 キー）。
- **決定（推奨）**: `conv_transpose3d` に限り、3 要素の引数を正準形の文字列 `"d,h,w"`（10 進・先頭ゼロなし・空白なし・要素ちょうど 3 個。それ以外は `Manifest` 拒否）で 1 キーへ畳む。キーは `in_channels`・`out_channels`・`kernel_size`・`stride`・`padding`・`output_padding`・`dilation`・`groups` の 8 個。`MAX_OBJECT_KEYS`・`MAX_JSON_DEPTH` は変更しない。load 側は文字列を厳格パースし、値の範囲は `add_conv_transpose3d` の構築時検査（0 次元・groups 割り切れ・`output_padding >= stride` 拒否）に委ねる。
- 却下案: (a) `MAX_OBJECT_KEYS` の引き上げは閾値変更で所有者の再承認が要る（`facade-functional-api-decision.md` §18 の「上限定数は変更も新設もしない」方針とも不一致）。(b) `dilation`／`output_padding` を固定値にする案は `add_conv_transpose2d` と非対称になり PyTorch 互換を損なう。

## 5. Fold／Unfold／LRN の常駐経路

- **決定**: 3 層とも常駐経路で `Unsupported`（fail-closed）とする。`PixelShuffle`／`Upsample` の素通し先例とは異なる扱いにする。
- 根拠: 3 演算とも GPU 専用カーネルを持たず BackendOps のホストフォールバックで動く（各決定記録 §8 スコープ外）。デバイス常駐中の中間値でのフォールバック挙動は未実測で、実機 parity は #3171・#3172 まで未測定。未実測経路を黙って通さない。緩和は実測後に追加のみの変更として行える。
- 実装形: `Module` トレイトへ新メソッドを足さず、`contains_resident_unsupported_layer` が `self.specs` の `Fold`／`Unfold`／`LocalResponseNorm` variant も見る形を推奨する（公開クレート `fandhe-ai-autodiff` の面を増やさないため）。

## 6. MaxUnpool の入力 2 つの扱い

| 案 | 内容 | 評価 |
|---|---|---|
| A（推奨） | `FunctionalBuilder` に索引を運ぶ専用ノードを追加。pool ノードは値を通常のノード値として返し、索引 `Tensor<i32>` はそのノードに付随する副スロットとして `eval_graph` が保持。unpool ノードは「特徴量ノード」と「索引元の pool ノード」の 2 入力を取る | `Block` 経路は不変。索引はノード値にならず隠れ状態がない。forward／fit／保存の全経路が `eval_graph` 共通本体を通るため bit 一致を保てる。変更は `functional.rs`・保存側・テストに限られ、Functional は別 format のため `load_model`／`Sequential` 保存に影響しない |
| B | 索引を保持する対の層（pool が最後の索引を `RefCell` に保存し unpool が参照） | 隠れ状態。`predict` の tape 不要経路が途中で `Unsupported` へ落ちて旧経路で再実行されると状態更新が二重になる（`Module::supports_forward_host` の既存注意と同種）。対応を load 時に層 index から再構築・検証する追加スキーマと `Module` への新フックが要る。ネスト pool（SegNet 型）で取り違えが起きうる |
| C | `Var` のみで層化しない | D-A の方針から外れるため、案 A の阻害が実装で判明した場合の退避先としてのみ記録 |

**結論は案 A**。公開形（`compat::FunctionalBuilder` への追加 6 メソッド。追加のみ）:

- `max_pool1d_with_indices(input: Node, kernel_size: usize, stride: Option<usize>, padding: usize, dilation: usize) -> Result<Node, AutodiffError>`、`max_pool2d_with_indices`（`[usize;2]`）、`max_pool3d_with_indices`（`[usize;3]`）。`ceil_mode=false` 固定。
- `max_unpool1d(input: Node, indices_from: Node, kernel_size, stride: Option<..>, padding, output_size: Option<..>) -> Result<Node, AutodiffError>`、`max_unpool2d`、`max_unpool3d`（`Var::max_unpool*` と引数型を一致。`seed` なし）。
- 検証（構築時・load 時の両方で fail-closed）: `indices_from` は自ノードより前の同次元 `max_pool*_with_indices` ノードのみ。他ビルダー由来・前方参照・次元不一致は `InvalidArgument`。索引値の範囲・shape は `Var::max_unpool*` が forward 時に検査する（pool との kernel／stride／padding 一致検査は設けない。`Var` 側と同じ契約）。
- Functional manifest（`fandhe-ai.compat.functional`・version 1 のまま）へ op を 6 個追加: `max_pool{1,2,3}d_indices`・`max_unpool{1,2,3}d`。`layer_len = 0`（Merge ノード同型）、`params` はスカラーのみの平坦キー（3d は pool・unpool とも 12 キー。1d／2d はそれ以下）。`inputs` は pool が 1 件、unpool が [特徴量, pool] の 2 件。旧リーダーは未知 op を fail-closed で拒否する（`facade-functional-api-decision.md` §8・§18）。
- 保存・学習: パラメータなしのため `train.rs`・`parameter_keys` は変化なし。重複索引の勾配規則は `autodiff-conv-transpose3d-max-unpool-decision.md` §12.2 (a)（last-writer の真の随伴）のまま。

## 7. 保留ガード反転範囲

§3 の表の最終列のとおり。ガードごとに反転するのは公開した `Sequential::add_*`（と MaxUnpool のみ `FunctionalBuilder`）の名前分だけで、他の否定プローブ・`api_surface.rs` の否定ガードは維持する。

## 8. 非破壊の確認

- `Sequential` への追加は `add_max_pool3d`・`add_avg_pool3d`・`add_conv_transpose3d`・`add_unfold`・`add_fold`・`add_local_response_norm` の 6 メソッド。`FunctionalBuilder` への追加は §6 の 6 メソッド。いずれも新メソッドのみで既存シグネチャ不変・`LayerSpec` 非公開・新規公開型なし。autodiff 側の層 struct は内部クレートへの追加のみ。`fandhe-ai =0.11.0` に対して追加のみ。
- 依存・`unsafe`・tolerance・baseline・`docs/spec` は変更しない。

## 9. format_version 据え置きと旧リーダー拒否

Sequential 保存は `format_version: 2` 据え置き。新 kind を足しても既存ファイルは読め、旧リーダーは未知 kind を fail-closed で拒否する（拒否時のエラー種別は実装確認後に `compat-model-io-decision.md` 追補で確定する）。Functional 保存も version 1 据え置き。他の記録・issue からは本節を参照する。

## 10. セキュリティ（OWASP A03・A08）

後続実装が守る load 側の厳格化を確定する: kind と op は文字列 allowlist・未知は fail-closed。`params` は厳密なキー集合。`stride`／`output_size` は「全て null か全て整数」。f32 は正準形で非有限は保存拒否。`conv_transpose3d` の三つ組文字列は正準形のみ受理。`indices_from` の参照先検証は safetensors に触れる前に完了する。ONNX export は fail-closed のまま。

## 11. 所有者確認点（確認が来るまで推奨で後続へ進める）

1. `conv_transpose3d` で `MAX_OBJECT_KEYS` 引き上げ案を望むか（推奨は文字列畳み込みで定数不変）。
2. MaxUnpool を案 A（`FunctionalBuilder` 限定）で確定してよいか。案 B を望む場合は `RefCell` 隠れ状態・`forward_host` 経路・load 時の対応再構築を含めて再設計が要る。
3. Fold／Unfold／LRN の常駐経路を実測前は拒否とする扱いでよいか（実機 parity 後に緩和）。

## 12. 後続 issue への申し送り

- 実装: pool3d は #3026 以降、ConvTranspose3d・Fold／Unfold・LRN・weight_norm／spectral_norm（#3046・#3047）は各 issue で本記録の形のまま実装する。実機 parity は #3161・#3162・#3171・#3172。
- **読み替え**: #3035 は `Module` impl と `as_*` フックを持たない固有メソッド型へ、#3036 は `Sequential::add_*` でなく `FunctionalBuilder` 追加メソッドへ、#3037 は Sequential の kind でなく Functional manifest の op 追加へ読み替える。issue 本文の更新は本 issue の範囲外（`update-issue-tree` での調整を提案）。

## 13. 出典

`autodiff-pool3d-ops-decision.md`・`autodiff-conv-transpose3d-max-unpool-decision.md`・`autodiff-fold-unfold-decision.md`・`autodiff-lrn-weight-reparam-decision.md`（各 §12・§14）、`facade-nn-layer-type-exposure-decision.md`、`facade-functional-api-decision.md`、`compat-model-io-decision.md`、`compat-api-scope.md` §5.1・§5 末尾。

## 14. #3028 実施記録（pool3d の `Sequential::add_*` 公開）

- `add_max_pool3d`／`add_avg_pool3d` を §3 の形で公開。`LayerSpec::Unsupported { kind: "max_pool3d" | "avg_pool3d" }` を採用し、`save_model` は `UnsupportedModel` で fail-closed（`celu` 等の先例と同型）。実 variant（12 キー／10 キー）への置換と保存・復元は後続 `-c` の範囲。
- 常駐経路は新規結線なしで `is_pooling()` 経由の `Unsupported`。学習経路（`fit`・`bind`）も無パラメータ層として新規結線不要。ONNX export は `UnsupportedLayer` のまま。
- 保留ガード `Pool3dOpsHoldDoctestGuard` は `Sequential` 受け手の分のみ反転。`nn::*` 層型名・`forward_host`・実機 parity は対象外。

## 15. #3029 実施記録（pool3d の保存対応）

- §14 の `LayerSpec::Unsupported` を実 variant（`MaxPool3d`／`AvgPool3d`）へ置換し、`save_model`／`load_model` を `max_pool3d`／`avg_pool3d` kind で結線した（スキーマは `compat-model-io-decision.md` の #3029 追補）。
- 層経路の実機 parity `#[ignore]` テストを追加。実測は #3171・#3172。

## 17. #3032 実施記録（`Sequential::add_conv_transpose3d` の公開）

- `add_conv_transpose3d` を §3 の形で公開。`LayerSpec::Unsupported { kind: "conv_transpose3d" }` を採用し、`save_model` は `UnsupportedModel` で fail-closed（#3028 の pool3d と同型）。実 variant（8 キー・`"d,h,w"` 正準形）への置換と保存・復元は後続 `-c` の範囲。
- 結線した箇所（`add_conv_transpose2d`／`add_conv3d` と同型）: `bind`・`SequentialVars::forward`・`trainable_vars`・`trainable_grads`・`first_untracked_parametric_layer`（独自層扱いの回避）・`contains_resident_unsupported_layer`（常駐は `BackendError::Unsupported`）。AMP 低精度は無視して f32 で forward。ONNX export は `UnsupportedLayer` のまま。
- 保留ガード `ConvTranspose3dMaxUnpoolHoldDoctestGuard` は `add_conv_transpose3d` の分のみ反転（残すと inherent メソッドと衝突するため）。`add_max_unpool*`・`Tape`／`Tensor<f32>`・モジュール・型名の否定は維持。`api_surface.rs` に承認シグネチャの正ガードを追加。
- `nn::*` 層型名の公開（#3022 方針）・`supports_forward_host` の見直し・実機 parity は対象外。依存・tolerance・baseline・`docs/spec` は不変。

## 18. #3033 実施記録（conv_transpose3d の保存対応）

- §17 の `LayerSpec::Unsupported { kind: "conv_transpose3d" }` を実 variant `ConvTranspose3d` へ置換し、`save_model`／`load_model` を `conv_transpose3d` kind（8 キー・三つ組は `"d,h,w"` 正準形）で結線した（スキーマは `compat-model-io-decision.md` の #3033 追補）。
- 層経路の実機 parity `#[ignore]` テストを追加。実測は #3171・#3172（`docs/perf/logs/compat-sequential-conv-transpose3d-3033/README.md`）。
- 公開 API 追加なし・`format_version` 2 据え置き・上限定数・tolerance・baseline・依存・`docs/spec` は不変。

## 19. #3036 実施記録（MaxUnpool の `FunctionalBuilder` 公開）

- Issue 本文は `Sequential` 向け雛形のままだが、本記録 §3・§6・§12 を正とし次の形で実施した。`LayerSpec` variant・`Sequential::add_max_unpool*` は作らない。
- `compat::FunctionalBuilder` に 6 メソッドを追加（追加のみ）: `max_pool{1,2,3}d_with_indices(input, kernel_size, stride, padding, dilation)`・`max_unpool{1,2,3}d(input, indices_from, kernel_size, stride, padding, output_size)`。いずれも `Result<Node, AutodiffError>`。`ceil_mode=false` 固定・`seed` なし・引数型は `Var::max_pool*`／`Var::max_unpool*` と一致。新規公開型・モジュールなし。
- ノードは `NodeDef::{MaxPoolIndices, MaxUnpool}`（非公開）。索引は pool ノードの副スロットとして `eval_graph` が保持し、unpool は `[特徴量, 索引元 pool]` の 2 入力。`forward`・`predict`・`fit`・`evaluate` は共通本体を通る。構築時に `indices_from` の種別・同次元・0 次元・`2*padding` オーバーフローを `InvalidArgument` で拒否し、索引の範囲・shape は forward 時に `Var` が検査する。1 つの pool の索引を複数の unpool が参照する fan-out と、特徴量＝pool 自身も許可する。
- 保存は #3036 時点では `save_functional_model` が `UnsupportedModel` で fail-closed（書き込み前に拒否）。manifest の op 追加と復元時の `indices_from` 検証は #3037（§12 の読み替え。実施は §20）。
- 常駐経路: Functional に入口が無く該当なし（§3）。`Sequential` に `add_max_unpool*` が無いことは保留プローブ `ConvTranspose3dMaxUnpoolHoldDoctestGuard`（プローブ本文は不変・doc のみ更新）と `api_surface.rs` が固定する。
- `api_surface.rs`: `max_unpool{1,2,3}d` の `fn` 宣言走査は `compat/functional.rs` に限り各 1 件を許可（他ファイル・2 件目・`add_max_unpool*` は従来どおり違反）。6 メソッドの承認シグネチャを正ガードで固定。
- 受け入れ条件との差分: 「保留ガードの反転」はプローブの受け手外のため該当なし（反転ではなく doc 更新と正ガード追加）、「常駐経路の `Unsupported` 結線」は該当なし。
- 依存・tolerance・baseline・`format_version`・`docs/spec` は不変。実機 parity（CUDA／Metal）は対象外。

## 20. #3037 実施記録（MaxUnpool の保存対応と実機 parity テスト）

- Issue 本文は `Sequential` の kind 追加という旧雛形のままだが、§6 案 A・§12 の読み替えどおり Functional manifest の op 追加として実施した（`Sequential` の `format_version` 2・kind 63 種は不変、Functional 形式の `format_version` は 1 据え置き）。スキーマ・検証規則は `docs/compat-model-io-decision.md` の #3037 追補。
- 実機 parity テストは `crates/facade/tests/compat_functional_max_unpool_backend_parity.rs`（Issue 本文のファイル名 `compat_sequential_*` は Functional 経路のため改めた）。ログ置き場は `docs/perf/logs/compat-sequential-max-unpool-3037/`。未実測（#3171・#3172）。
- 依存・tolerance・baseline・`docs/spec` は不変。
