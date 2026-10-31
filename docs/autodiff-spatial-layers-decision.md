# ConvTranspose1d・Upsample・ZeroPad2d・Identity・Unflatten の設計判断記録

イシュー #2159（親 #2131）。`docs/autodiff-activation-ops-decision.md`
（#2146）・`docs/conv-transpose2d-design.md` 系（#2067）と同型の記録。

## §0 結論

PyTorch の `nn.ConvTranspose1d`・`nn.Upsample`・`nn.ZeroPad2d`・
`nn.Identity`・`nn.Unflatten` に相当する 5 層を、`fandhe_ai_autodiff`
の `nn` モジュール（`crates/autodiff/src/nn/{conv,upsample,padding,
identity,unflatten}.rs`）へ追加した。いずれも既存 `Var` 演算
（`Var::conv_transpose2d`・`Var::interpolate`・`Var::pad`・
`Var::reshape`）の薄い合成であり、新規 `Op`・`BackendOps` メソッド・
VJP・GPU 専用カーネルは追加していない。`Var` に inherent の `pub fn`
は追加していない（`Var` は facade から再エクスポートされるため。
#2143・#2144・#2146 の先例）。facade 公開（`compat::Sequential::
add_*` 5 種・`Var::conv_transpose1d`／`Var::unflatten` の委譲メソッド）
は当初承認待ちのまま対象外とし、`crates/facade/src/lib.rs::
SpatialLayersHoldDoctestGuard`（正のプローブ doctest）と
`crates/facade/tests/api_surface.rs` のソース走査（2 テスト＋自己
テスト）で多層固定していた。**イシュー #2521 で `ConvTranspose1d`／
`Unflatten` の 2 層（`add_conv_transpose1d`／`add_unflatten`・
`Var::conv_transpose1d`／`Var::unflatten`）は公開済み**（§6 実装記録）。
残り 3 層（`Upsample`／`ZeroPad2d`／`Identity`）は #2522 で公開済み（§8）。

## §1 背景

イシュー #2159・親 #2131 のコメントはいずれも 0 件（2026-09-25 時点。
着手前に `gh issue view 2159/2131` で確認済み）。親 #2131 はこの
ツリーでの facade 公開面の拡張を「設計判断記録 → 承認 → 実装」の
2 段階と定めているため、本実装は内部クレート限定に倒す（同じ
ツリーの先例 #2146 §1 と同じ判断）。依存イシュー #2067
（ConvTranspose2d。PR #2209）はマージ済み・クローズ済みで従属条件は
満たしている。

## §2 設計判断

### §2.1 5 層の配置とハイパーパラメータ設計

| 層 | ファイル | 合成 | 数値・勾配の契約 |
|---|---|---|---|
| `ConvTranspose1d` | `nn/conv.rs` | `input`／`weight` を `[*, *, 1, *]` へ reshape し `Var::conv_transpose2d`（#2067）へ委譲 | `gemm_batched` を経由するため REQ-2 統一複合判定。同 seed の `ConvTranspose2d([1,k])` と重み・forward・勾配が bit 完全一致（構造的に同一のため） |
| `Upsample` | `nn/upsample.rs` | `UpsampleSize::ScaleFactor` は `interpolate_size_from_scale_factor`（#2152）で `size` を導出し `Var::interpolate` へ委譲 | `Nearest`／`NearestExact` は bit 完全一致、それ以外は REQ-2 |
| `ZeroPad2d` | `nn/padding.rs` | `padding: [left, right, top, bottom]` を先頭次元順 `pads` へ組み替え `Var::pad`（#1756）へ委譲 | 定数 0 埋めのみのため bit 完全一致 |
| `Identity` | `nn/identity.rs` | `Var` は `Copy` のため入力をそのまま返す（新規ノードなし） | forward・勾配とも恒等のため bit 完全一致 |
| `Unflatten` | `nn/unflatten.rs` | `Var::reshape` へ委譲（`Flatten` の逆変換） | reshape のみのため bit 完全一致 |

`ConvTranspose1d` は `Conv1d`（#1770）と同じ理由で `weight` を rank 3
のまま保持する（`weight()` が `&Tensor<f32>` を返す既存契約のため。
`forward`／`forward_host` の内部でのみ rank 4 へ一時的に reshape す
る）。`Var::reshape` は view ノードを tape へ push するため、
`ConvTranspose1dVars::forward` では x4/w4 への reshape より前に
`forward_host` と同じ検査順序（①rank → ②`Conv2dParams::new` →
③`output_padding < stride` → ④4 次元 `conv_transpose2d_out_shape` →
⑤bias shape）を純粋な shape 計算（形状値のみ・tape 操作なし）として
完了させ、`Err` 経路で孤児ノードを残さない（`Var::conv1d` と同じ
規律。イシュー #2159 レビュー指摘・codex／Cursor Bugbot 両方。PR
#2276）。`Var::conv_transpose2d` 内部でも同じ検査が再実行されるが、
それは reshape 後の 2 回目の検査であり `forward` 側のこの事前検査を
代替しない。

`Upsample` はサイズ指定方式を `enum UpsampleSize { Size(Vec<usize>),
ScaleFactor(Vec<f64>) }`（`#[non_exhaustive]`）で表現し、`size` と
`scale_factor` の両方指定・どちらも未指定という状態を型の上で
排除した。`scale_factor` 経路は `recompute_scale_factor=True` 相当の
座標系になる（`interpolate_size_from_scale_factor` doc・#2152 の
決定を継承する既知の PyTorch 非互換）。

`ZeroPad2d` は `usize` のため負パディング（クロップ）を表現できない
（`Var::narrow` を使うことを代替として doc に明記）。

`Unflatten` は PyTorch の `-1`（1 軸だけ自動推論）を `usize` では
表現できないため非対応とした。

### §2.2 facade 公開・compat::Sequential の add_* — 2 層は #2521、3 層は #2522 で公開

イシュー #2159 は「facade への 5 個の `add_*` メソッド」を承認事項
として挙げ「承認前に実施しない」と定めている。コメントでの承認も
ないため、`crates/facade/src/compat/sequential.rs` は変更していない。
`Var::conv_transpose1d`／`Var::unflatten`（forward 相当の inherent
メソッド追加。経路 1）も同様に未承認のため見送った。両者を
`SpatialLayersHoldDoctestGuard` の同一 doctest ブロックで併せて
保留固定している（`crates/facade/src/lib.rs` 参照）。

（#2521 更新）`ConvTranspose1d`／`Unflatten` の 2 層は §6 の承認
（ルート #2499 の一括承認）に基づき公開した。残り 3 層の `add_*` は
#2522 で公開済み（§8）。以上の本節前半は #2159 時点の当初判断の記録である。

### §2.3 `Module` trait への統合

`as_conv_transpose1d`／`as_conv_transpose1d_mut`（既定 `None`）を
`as_conv_transpose2d` の直後に追加した。5 層すべてに `forward`・
`forward_host`（bit 完全一致契約。`supports_forward_host` は既定
`true` のまま）を実装した。`ConvTranspose1d` は `named_parameters`
（`weight` → `bias`）・`set_parameter`・`set_requires_grad`・
`requires_grad` も `ConvTranspose2d`／`Conv1d` と同型で実装した。
`Identity` はパラメータを持たない ZST のため `named_modules` の
ZST 特例（`module.rs::collect_named_modules`）の対象になる。

## §3 契約（変更禁止・維持）

- tolerance・baseline・`Cargo.toml` の依存・ガードレール閾値・
  `docs/spec/` は変更していない
- 新規 `unsafe` なし・依存追加なし
- crates.io 公開済みの `fandhe-ai =0.9.0`・`fandhe-ai-autodiff` の
  公開 API は非破壊（新しい pub 型の追加と `Module` trait への
  既定実装付きメソッドの追加のみ）

## §4 テスト

- `crates/autodiff/src/nn/{conv,upsample,padding,identity,
  unflatten}.rs` の各 unit test（構築検査・数値・境界値・`forward_host`
  との bit 一致）
- `crates/autodiff/tests/nn_spatial_layers.rs`: `ConvTranspose1d`
  と `ConvTranspose2d([1,k])` の重み・forward・構造一致、PyTorch
  式の出力長、`output_padding >= stride`・rank 違反・bias shape
  不一致の拒否（孤児ノードを残さないこと込み）、`forward_host` と
  `forward` の bit 一致、5 層混在 `nn::Sequential` の統合
  （`named_parameters`／`parameter_count`／`set_requires_grad`／
  `summary`）
- `crates/facade/tests/spatial_layers_backend_parity.rs`: CPU
  （`CpuBackendOps`）対 `NaiveOps` の parity（`ZeroPad2d`・
  `Identity`・`Unflatten`・`Upsample(Nearest／NearestExact)` は
  bit 完全一致、`ConvTranspose1d`・`Upsample(Bilinear)` は REQ-2
  複合判定）と、`cuda_*`／`metal_*` の `#[ignore]` parity（実機未
  実測。`docs/perf/logs/spatial-layers-2159/README.md` 参照）
- `crates/facade/src/lib.rs::SpatialLayersHoldDoctestGuard`（正の
  プローブ doctest）・`crates/facade/tests/api_surface.rs` の
  `spatial_layers_hold_doctest_globs_all_pub_modules`・
  `spatial_layers_hold_doctest_probe_body_matches_fixed_contract`
  （`compat_sequential_does_not_expose_spatial_layer_add_methods` は #2521・#2522 で
  禁止対象が 0 件になったため削除）

## §5 スコープ外（`out-of-scope-tracking.md` に従う。Issue 起票は
ユーザー承認後）

- `compat::Sequential::add_*` 5 種・`Var::conv_transpose1d`／`Var::unflatten` の
  facade 公開は #2521（`add_conv_transpose1d`／`add_unflatten`・`Var` 2 メソッド）と
  #2522（`add_upsample`／`add_zero_pad2d`／`add_identity`。§8）で全件公開済み
- `save_model`／`load_model` での `conv_transpose1d`／`unflatten` の構成保存
  （manifest スキーマ拡張が必要。#2521 は型付きエラーで fail-closed）
- `nn::ConvTranspose1d`／`nn::Unflatten` の型の facade 再エクスポート
- GPU 専用カーネル（`ConvTranspose1d`・`Upsample` の新モード向け）
- `ZeroPad2d` の負パディング（クロップ）
- `Unflatten` の `-1` 推論
- `Upsample` のスカラー `scale_factor` の全空間軸への broadcast・
  `recompute_scale_factor=False` の座標系
- `output_padding >= stride` の PyTorch 互換化（col2im の `P` 軸
  契約の拡張が必要。#2067 から継承）
- ONNX export（`onnx-interop::export_nn`）での 5 層への対応

## §6 承認事項（未承認として列挙。ConvTranspose1d／Unflatten は #2521 で適用済み）

1. facade `compat::Sequential` への `add_conv_transpose1d`／
   `add_upsample`／`add_zero_pad2d`／`add_identity`／`add_unflatten`
   の追加（経路 2）と、学習経路（`bind`／`trainable_parameters`／
   `apply_parameters`／`trainable_vars`／`trainable_grads`／
   `contains_resident_unsupported_layer`）への結線
2. 上記に伴う `Var::conv_transpose1d`／`Var::unflatten` の追加
   （経路 1）と保留ガード（`SpatialLayersHoldDoctestGuard`・
   `api_surface.rs` の対応する否定ガード）の撤去

### §6 実装記録（イシュー #2521・親 #2520・ルート #2499 本文「承認範囲」節の一括承認）

上記 1・2 のうち `ConvTranspose1d`／`Unflatten` の 2 層分を、推奨形で実装した
（`fandhe-ai =0.10.0` の公開 API に対する追加のみで非破壊）。本節に具体シグネチャの
記載がなかったため、既存の内部 API（`ConvTranspose1d::new`・`Unflatten::new`・`Var::conv1d`・
`Var::flatten`・`add_conv1d`）から機械的に導いた。

| 公開面 | シグネチャ（導出元） |
|---|---|
| `compat::Sequential::add_conv_transpose1d` | `(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation, groups, seed: u64) -> Result<Self, AutodiffError>`（`add_conv1d` に `output_padding` を加えた形・bias あり固定） |
| `compat::Sequential::add_unflatten` | `(self, dim: usize, unflattened_size: Vec<usize>) -> Result<Self, AutodiffError>`（`Unflatten::new` が空 `sizes` を拒否するため `Result`） |
| `Var::conv_transpose1d` | `(&self, weight, bias: Option<&Var>, stride, padding, output_padding, dilation, groups: usize) -> Result<Var, AutodiffError>`（`Var::conv1d`／`conv_transpose2d` のスカラー版） |
| `Var::unflatten` | `(&self, dim: usize, sizes: &[usize]) -> Result<Var, AutodiffError>`（`Var::flatten` と対） |

- **forward 本体の一本化**: `nn/conv.rs::conv_transpose1d_forward`・`nn/unflatten.rs::unflatten_forward`
  （いずれも `pub(crate)`）へ集約し、層経路（`ConvTranspose1dVars::forward`／`Unflatten::forward`）と
  `Var` 経路が同じ検査順序を共有する（判定迂回経路を作らない）。shape 検査は tape 操作の前に完了し
  孤児ノードを残さない。`Var::unflatten` は空 `sizes` を `InvalidArgument` で拒否する。
- **`compat::Sequential` 結線**: `bind`／`SequentialVars::forward`／`trainable_vars`／`trainable_grads`／
  `first_untracked_parametric_layer`／`contains_resident_unsupported_layer`（`ConvTranspose1d` のみ。
  常駐経路は `BackendError::Unsupported`）へ結線した。`trainable_parameters`／`apply_parameters` は
  `named_parameters`（weight → bias）と `Module::set_parameter` による汎用分岐のまま処理する
  （層別 `requires_grad` を保持でき、兄弟 #2523 との競合面を減らせるため `Rebuilt` 方式を採らない）。
  `Unflatten` は `Flatten` と同じく無パラメータで、常駐経路も通過する。AMP 低精度は `Conv1d` と同じく無視し
  f32 で forward する。
- **保存は fail-closed**: `LayerSpec::Unsupported { kind: "conv_transpose1d" | "unflatten" }` とし、
  `save_model` は `dir` に触れる前に `ModelIoError::UnsupportedModel` を返す。`unflattened_size` は可変長で
  manifest の固定キー方式（`MAX_JSON_DEPTH = 4`）に乗らず、スキーマ拡張は本件の承認範囲を超えるため。
  ONNX export も既存どおり `OnnxError::UnsupportedLayer`。
- **保留ガードの反転（該当名のみ）**: `SpatialLayersHoldDoctestGuard` から 2 層分の trait 経由プローブ
  （`__FandheSpatialVarProbe` 全体・`__FandheSpatialAddProbe` の該当 2 メソッド）を削除し、
  `api_surface.rs::SPATIAL_LAYERS_HOLD_PROBE_BODY` と 1 行も違わず同期した。型名の衝突プローブと
  自由関数プローブは残し、型の再エクスポート・自由関数での公開は引き続き禁止する。
  `compat_sequential_does_not_expose_spatial_layer_add_methods` は #2522 との統合で禁止対象が 0 件になり削除し、正ガード
  （`workspace_declares_spatial_facade_fn_names_only_in_approved_locations`・
  `var_spatial_methods_are_thin_delegations`・`var_spatial_methods_are_reachable_via_facade_only`・
  `compat_sequential_spatial_add_methods_have_approved_signatures`）を新設した。
- **実機 parity**: 新規カーネルなし。CUDA／Metal は既存の `#[ignore]` テストと申し送り
  （`docs/perf/logs/spatial-layers-2159/README.md`）が有効で、本件では未実測。

## §7 実機実測の申し送り

CUDA（DGX Spark GB10）・Metal 実機は本エージェント実行環境に無いため
`#[ignore]` テストを未実行のまま出荷する。実行コマンド・記入欄は
`docs/perf/logs/spatial-layers-2159/README.md` を参照。

## §8 実装記録（イシュー #2522・ルート #2499 の 2026-10-04 一括承認）

§6 承認事項 1 のうち `add_upsample`／`add_zero_pad2d`／`add_identity` の 3 件を実装した
（`add_conv_transpose1d`／`add_unflatten` と `Var` 委譲メソッドは別イシュー〈#2521〉の対象で、そちらで公開済み。§6 実装記録）。

- **公開面**: `compat::Sequential` に `pub fn` 3 件のみ。型（`Upsample`／`ZeroPad2d`／`Identity`／`UpsampleSize`）の
  facade 再エクスポートはしない（§6 に記録がなく未承認）。
  - `add_upsample(self, size: Vec<usize>, mode: InterpolateMode) -> Result<Self, AutodiffError>`
    （`Upsample::with_size` への 1:1 委譲。**size 指定のみ**。`scale_factor` 経路は公開形が未承認のため保留）
  - `add_zero_pad2d(self, padding: [usize; 4]) -> Self`（`[left, right, top, bottom]`）
  - `add_identity(self) -> Self`
- **`as_*` フックは追加しない**: Issue 本文は `nn::Module` への `as_*` フック追加を例示するが、`save_model`／`load_model` は
  `LayerSpec`（`compat::sequential`）だけを根拠にするため不要であり、crates.io 公開済み `fandhe-ai-autodiff` の trait 面を
  広げない判断とした（Issue 本文の例示からの意図的な逸脱）。
- **学習経路**: 3 層はパラメータを持たない無状態層のため、`bind`／`trainable_parameters`／`apply_parameters`／
  `trainable_vars`／`trainable_grads` の既存走査をそのまま素通しする（`Flatten`〈#2065〉と同型・コード分岐の追加なし）。
- **常駐経路**: `Flatten` の先例どおり対応扱い（`contains_resident_unsupported_layer` に追加しない）。常駐非対応層
  （`Conv2d` 等）との混在は従来どおり `Unsupported` で拒否される（テストで固定）。
- **保存経路**: kind `upsample`・`zero_pad2d`・`identity` を追加（スキーマは `docs/compat-model-io-decision.md` §4）。
  `size` の軸数が 1..=3 外、または `InterpolateMode`（`#[non_exhaustive]`）の未知 variant は保存前に `UnsupportedModel`
  で拒否し `dir` へ副作用を出さない。`format_version` は不変。
- **保留ガードの反転**: `SpatialLayersHoldDoctestGuard` から 3 メソッドの自由関数・trait・呼び出しプローブを撤去し、
  型名プローブ（5 型）と `add_conv_transpose1d`／`add_unflatten`／`Var` 2 種のプローブは維持。`api_surface.rs` の否定テストの
  禁止リストは 2 件へ縮め、`compat_sequential_exposes_spatial_layer_add_methods_issue_2522`（3 メソッドが各ちょうど 1 件の
  `pub fn` であることを fail-closed で検査する正ガード＋自己テスト）を追加した。
- **残課題（要承認）**: `add_upsample` の `scale_factor` 指定（`UpsampleSize` 再エクスポートまたは別名メソッド）・型の再エクスポート・
  `ZeroPad2d::uniform`／負パディングの facade 公開・ONNX export 対応（`UnsupportedLayer` で fail-closed のまま）。
- **実機 parity**: `crates/facade/tests/compat_sequential_spatial_layers_backend_parity.rs`（`#[ignore]`）。
  未実測のまま `docs/perf/logs/compat-sequential-spatial-2522/README.md` で GB10／M4 Max へ申し送る。

## §9 保守・運用チェックリスト（イシュー #3740）

追跡表 `docs/compat-matrix-ari-tracker.md` の行 #7（`add_upsample` の `scale_factor` 指定。Issue 上は「項目 8」と呼ばれる同一項目）の
公開形を設計する。**設計のみ**で、コード・公開面・`docs/spec` は変えない。

- **承認根拠**: なし。**未承認（#3768 待ち）**。実装は承認後の #3814、実機 parity は #3860（`<cuda-node>`）・#3861（`<metal-node>`）、
  保存・export の横断方針は #3750（OPEN）。
- **現状**: facade の `compat::Sequential::add_upsample(size: Vec<usize>, mode)`（`crates/facade/src/compat/sequential.rs:2435`、
  基準 `30bd1d046`）は出力サイズ直指定のみ。内部クレートには倍率経路が実装済み
  （`autodiff::nn::Upsample::with_scale_factor`・`UpsampleSize::ScaleFactor(Vec<f64>)`〈`#[non_exhaustive]`、
  `crates/autodiff/src/nn/upsample.rs:21-31`〉・`tensor-core::interpolate_size_from_scale_factor`〈`interpolate.rs:426`〉）。
  §8 の残課題「`UpsampleSize` 再エクスポートまたは別名メソッド」を本節で比較する。

| 案 | 内容 | 利点 | 欠点・影響 |
|---|---|---|---|
| A（推奨） | 別名メソッド `compat::Sequential::add_upsample_scale_factor(self, scale_factor: Vec<f64>, mode: InterpolateMode) -> Result<Self, AutodiffError>` を新設（`Upsample::with_scale_factor` への 1:1 委譲。既存 `add_upsample` は不変） | 公開名は `pub fn` 1 件の追加のみ。新しい型を出さず `docs/facade-nn-layer-type-exposure-decision.md` の (i)+(iv) 方針に合う。`docs/api-compat-0.14.0-decision.md` §1.2 行 7 の「別名メソッドなら追加のみ」に当たる | メソッドが 2 つに分かれる。指定方式が増えるとメソッドも増える |
| B | `UpsampleSize` を再エクスポートし `add_upsample` の引数を `UpsampleSize` に変える | PyTorch の `size`／`scale_factor` 二択を型で表せる | 既存シグネチャ変更で**破壊的**（api-compat §8 手順 2）。§4 に承認済みの行がなく手順 3 により進められない。`#[non_exhaustive]` 型の variant 追加が公開面へ伝わる |
| C | `UpsampleSize` を再エクスポートし、それを受ける新メソッド（例 `add_upsample_with`）を足す | 追加のみ | 型名を名指ししないと使えない操作がなく (iv) 条件 (a) を満たさない。公開名が 2 件増える。`Size` が倍率も含み紛らわしい |

**推奨: 実装する（案 A）**。理由は、互換が追加のみで済むこと、型名公開方針 (i)+(iv) と矛盾しないこと、内部経路が実装済みで委譲 1 行で足りることの 3 点。

### §9.1 公開形とガード反転計画

- 配置: `crates/facade/src/compat/sequential.rs` の `add_upsample` 直後。`Upsample::with_scale_factor` で検査し、成功時のみ
  `inner.push` と `specs.push` を行う（検査前に状態を変えない）。
- `LayerSpec`（`pub(super)`、公開面の外）には新 variant `UpsampleScaleFactor { scale_factor: Vec<f64>, mode: InterpolateMode }` を足し、
  既存 `Upsample { size, mode }` は広げない。size 経路の forward・manifest のバイト列・既存テストは bit 単位で不変（#3814 の「既定経路が bit 同一」に合う）。
- ガード: 現在 `scale_factor` 用の否定ガードは存在しない（`api_surface.rs` に該当なし）。したがって「反転」は正ガードの新設を指す。
  - `SPATIAL_LAYER_APPROVED_ADD_METHODS`（`crates/facade/tests/api_surface.rs:14473`）を 4 名に拡張し、
    `compat_sequential_exposes_spatial_layer_add_methods_issue_2522`（`:14489`）と自己テスト `..._counts_declarations`（`:14508`）に 4 名目を足す。
  - `ADD_UPSAMPLE_SCALE_FACTOR_PARAMS` 定数を追加し、`compat_sequential_spatial_add_methods_have_approved_signatures`（`:14427`）で
    `sequential_spatial_add_signature_ok`（`:14401`）によりシグネチャを固定。すり替え検出の自己テスト（`:14451`）に `f32` 化・引数入れ替え・`Self` 返しを足す。
  - 否定プローブとして維持: `SpatialLayersHoldDoctestGuard`（`crates/facade/src/lib.rs:3793`）の 5 型名プローブと
    `add_conv_transpose1d`／`add_unflatten` の自由関数プローブ。案 A は `UpsampleSize` を出さないため、その型名プローブの追加を推奨する
    （追加時は `SPATIAL_LAYERS_HOLD_PROBE_BODY`〈`:14231`〉と 1 行も違わず同期し、`spatial_layers_hold_doctest_probe_body_matches_fixed_contract`〈`:14209`〉・
    `spatial_layers_hold_doctest_globs_all_pub_modules`〈`:14179`〉を通す）。

### §9.2 rustdoc と doctest

- 役割要約: 「Upsample 層を倍率指定で追加する（`nn::Upsample::with_scale_factor`。§9）」。丸め契約と座標系（§9.5）を doc に書く。
- doctest は facade だけで通り CPU で動く: `[1,1,3,3]` に `vec![2.0, 2.0]`・`Nearest` → `[1,1,6,6]`（`add_upsample(vec![6,6], Nearest)` の出力と `host_slice` 一致）、
  `vec![1.5, 1.5]` → `floor(3*1.5)=4` で `[1,1,4,4]`。GPU が要る例は置かない（置くなら `no_run`）。
- 失敗系は `compile_fail` に頼らず、CI の統合テストで `Err` の variant まで照合する。

### §9.3 エラー型

新 variant は追加しない。

- 構築時: `AutodiffError::InvalidArgument`（空・非有限・0 以下）。
- forward 時: rank 不足（入力 rank < `scale.len()`）は `AutodiffError::Shape(ShapeError::RankMismatch)`、オーバーフロー・出力 0 は
  `ScaleFactorError` を包んだ `AutodiffError::InvalidArgument`（`upsample.rs::resolve_size`）。`resolve_size` は入力の末尾 `scale.len()` 軸を切り出して
  ヘルパーへ渡すため、ヘルパー側の「2 スライスの長さ不一致」は到達不能で、契約にしない。倍率の個数と `mode` が要求する空間軸数の不一致は、
  導出 `size` を渡した後続の補間形状検証（`Var::interpolate`）が返す `AutodiffError::Shape` 系の Err で検出する（forward 時の遅延検査）。
- 本番経路で `unwrap`／`expect` を使わない。検証は確保（`Vec`・tape への push・`specs.push`）より前に行う。
- model_io 読込はキー集合の完全一致 → 型 → 正準表記 → 範囲の順に検査し `ModelIoError::Manifest` で fail-closed にする。

### §9.4 テスト計画

| 受け入れ基準 | テスト（置き場・名前案） | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|---|---|---|---|---|---|
| 構築時の拒否（空・0・負・NaN・±inf） | `crates/facade/tests/compat_sequential_spatial_layers.rs::add_upsample_scale_factor_rejects_invalid_scale` | CI | 該当なし | 該当なし | 該当なし |
| floor 丸め（1.5 倍で 3→4、0.5 倍で 4→2）と、出力 0・rank 不足（`Shape(RankMismatch)`）・倍率個数と `mode` の空間軸数の不一致（補間形状検証の `Shape` 系 `Err`）の forward 時 `Err` | 同 `add_upsample_scale_factor_floors_output_size`・`..._rejects_zero_output_and_rank_mismatch` | CI | 該当なし | 該当なし | 該当なし |
| 倍率経路と導出サイズの size 経路が CPU で bit 一致（Nearest・Bilinear） | 同 `add_upsample_scale_factor_matches_size_path_bit_exact` | CI | 該当なし | 該当なし | 該当なし |
| 学習経路（`fit`）と常駐経路の素通し | 同 `add_upsample_scale_factor_passes_train_and_resident_paths` | CI | 該当なし | 該当なし | 該当なし |
| 保存（§9.5 の論点 2 の結果に従う）。(ii) なら `save_model` が `dir` に触れる前に `UnsupportedModel`。(i) なら往復 bit 一致と、非正準・範囲外 manifest の拒否 | `crates/facade/tests/compat_sequential_model_io_layers.rs` に追加。非信頼入力は `model_io.rs` の既存表駆動テストに追加 | CI | 該当なし | 該当なし | 該当なし |
| ONNX export が `UnsupportedLayer` を返す | facade の ONNX export テストに 1 件追加 | CI | 該当なし | 該当なし | 該当なし |
| 公開形の正ガードと自己テスト | `api_surface.rs`（§9.1） | CI | 該当なし | 該当なし | 該当なし |
| 実機 parity（Nearest は bit 一致、Bilinear は REQ-2） | `crates/facade/tests/compat_sequential_spatial_layers_backend_parity.rs` に `cuda_upsample_scale_factor_nearest_bit_exact`・`cuda_upsample_scale_factor_bilinear_matches_cpu`・`metal_upsample_scale_factor_nearest_bit_exact`・`metal_upsample_scale_factor_bilinear_matches_cpu`（Metal は `cfg(target_os = "macos")`） | `#[ignore]` | 未実測（#3860） | 未実測（#3861） | 該当なし（ROCm は facade 未公開） |

### §9.5 数値契約

- **丸め**: `out[axis] = floor(in[axis] as f64 * scale[axis])`（`tensor-core::interpolate_size_from_scale_factor` の式。PyTorch と同じ）。
  非有限・0 以下・`usize` 超過（`as usize` の前に検査）・導出後 0 は fail-closed で拒否（個数不一致は rank 不足または後続の補間形状検証で拒否。§9.3）。縮小（`s < 1`）は受理。
  f64 の表現誤差（例 `0.29*100 = 28.999…` は 28）は `floor` 前に補正しない（PyTorch と同じ）。
- **座標系**: PyTorch の `recompute_scale_factor=True` 相当。導出 `size` を既存 `Var::interpolate` へ渡すため補間座標は `in/out`。
  PyTorch 既定（`None`、座標に `1/scale`）とは `in*s` が整数でないときだけ差が出うる（整数倍は一致）。これを契約として doc に明記し、切り替えは公開しない。
- **REQ-2**: 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）をそのまま適用。新カーネル・Op・VJP はなく既存 `Var::interpolate` の契約を継承する。
  `Nearest`／`NearestExact` は 3 バックエンドで bit 一致、`Bilinear` ほかは REQ-2。FMA 契約・f64 長軸縮約契約への影響なし。tolerance・baseline・ガードレール閾値は不変で baseline は追加しない。
- **model_io**: 倍率は f64 のまま bit 一致で往復する必要がある（`floor(in*s)` が値に依存）。現状の `model_io.rs` は `as_f32`・`put_real(f32)` のみ。
  - (i) 新 kind `upsample_scale_factor`: キー `mode`・`align_corners`・`scale_len`（1..=3）・`scale_0..2`（f64 最短往復表記 `{:?}` と正準表記検査、`as_f64` を新設。`MAX_NUMBER_LEXEME_LEN = 64`・キー上限 16 に収まる）。`format_version` 不変（#2522 の先例）。古い読み手は未知 kind を `UnsupportedModel` で拒否。
  - (ii) 当面 fail-closed: `dir` に触れる前に `UnsupportedModel`（§6 の `conv_transpose1d` の先例）。
  - 横断方針は #3750 が決めるため、(i) は設計案として記録し、#3814 は「#3750 が先に入っていればその規則、無ければ (ii)」を推奨する。
- **ONNX export**: `graph_from_layers` は 6 種のみ対応で size 指定 Upsample もすでに非対応。倍率指定も `OnnxError::UnsupportedLayer` で fail-closed。`Resize`（scales）対応は #3750 とスコープ外へ回す。

### §9.6 互換への影響

**追加のみ**（`docs/api-compat-0.14.0-decision.md` §8 手順 1〜4）。触れる公開型は `Sequential` の inherent `pub fn` 1 件の追加のみで、
`add_upsample` は不変、`UpsampleSize` は再エクスポートせず、`LayerSpec` は公開面の外、`ModelIoError`・`OnnxError` に variant 追加なし。
案 B を選ぶ場合は破壊的で承認要（移行ノート・`type(scope)!:`・`BREAKING CHANGE:` が必要で、§4 に承認済みの行もない）。

### §9.7 適用記録・索引・追跡表

- 適用記録の案（実装 PR #3814 で `docs/compat-api-scope.md` へ追記する。本 PR では追記しない）:
  `適用記録（イシュー #3814。承認: <#3768 の回答 URL>）: compat::Sequential::add_upsample_scale_factor(scale_factor: Vec<f64>, mode) を公開。UpsampleSize・Upsample 型は未公開のまま。保存は <(i)/(ii)>、ONNX export は UnsupportedLayer。公開状態: 公開済み`
- 本 PR の更新: `docs/README.md` の本書注釈に 1 文、追跡表 行 #7 の「設計記録」セル。`compat-api-scope.md` は #3814 で行う。公開状態は未公開。

### §9.8 ベンチと性能への影響

既定経路（size 指定）は不変。倍率経路で増えるのは forward ごとの O(空間軸数) の f64 乗算と `floor` のみで、補間カーネルは既存と同じ。ベンチは追加せず、前後比較も不要（未実測）。

### §9.9 依存

依存の追加なし（許容 12 区分・禁止リストの照合に変化なし）。新しい `unsafe` なし。

### 所有者へ戻す論点（#3768 の論点 1・4 へ渡す）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|---|---|---|---|
| 1. 公開形 | A 別名メソッド／B `UpsampleSize` に変えて `add_upsample` を変更（破壊的）／C 型の再エクスポートと新メソッド | A | 現状（公開面なし）を維持し、実装へ進めない |
| 2. 保存（model_io） | (i) 新 kind と f64 正準表記で保存／(ii) 当面 `UnsupportedModel` で fail-closed、#3750 に合わせ後で対応 | #3750 が先ならその規則、無ければ (ii) | (ii) 相当の fail-closed を維持 |
| 3. ONNX export | `UnsupportedLayer` のまま／`Resize`（scales）対応 | `UnsupportedLayer` のまま（#3750 に委ねる） | 同上 |
| 4. メソッド名 | `add_upsample_scale_factor`／`add_upsample_scale`／`add_upsample_with_scale_factor` | `add_upsample_scale_factor`（内部構築子と PyTorch 引数名に揃う） | 同上 |

### セキュリティ観点・スコープ外

- A03（manifest の非信頼入力）: f64 の正準表記・有限性検査、`scale_len` 上限、キー集合の完全一致を確保より前に行う。A04・A08: 構築と forward の判定を `resolve_size` 1 か所に寄せ迂回経路を作らない。`as usize` の飽和は前検査で防ぐ（既存実装）。
- スコープ外: `recompute_scale_factor=None` 相当の座標系の公開、ONNX `Resize` の export、`UpsampleSize`／`Upsample` 型の再エクスポート、`Var` への倍率メソッド追加。
