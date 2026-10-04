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
残り 3 層（`Upsample`／`ZeroPad2d`／`Identity`）は #2522 の担当で保留のまま。

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

### §2.2 facade 公開・compat::Sequential の add_* — 2 層は #2521 で公開、3 層は保留

イシュー #2159 は「facade への 5 個の `add_*` メソッド」を承認事項
として挙げ「承認前に実施しない」と定めている。コメントでの承認も
ないため、`crates/facade/src/compat/sequential.rs` は変更していない。
`Var::conv_transpose1d`／`Var::unflatten`（forward 相当の inherent
メソッド追加。経路 1）も同様に未承認のため見送った。両者を
`SpatialLayersHoldDoctestGuard` の同一 doctest ブロックで併せて
保留固定している（`crates/facade/src/lib.rs` 参照）。

（#2521 更新）`ConvTranspose1d`／`Unflatten` の 2 層に限り §6 の承認
（ルート #2499 の一括承認）に基づき公開した。残り 3 層の `add_*` は保留。

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
  `spatial_layers_hold_doctest_probe_body_matches_fixed_contract`・
  `compat_sequential_does_not_expose_spatial_layer_add_methods`
  （＋自己テスト）

## §5 スコープ外（`out-of-scope-tracking.md` に従う。Issue 起票は
ユーザー承認後）

- `compat::Sequential::add_upsample`／`add_zero_pad2d`／`add_identity` の
  facade 公開（#2522 の担当。保留）。`add_conv_transpose1d`／`add_unflatten`・
  `Var::conv_transpose1d`／`Var::unflatten` は #2521 で公開済み
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
  `compat_sequential_does_not_expose_spatial_layer_add_methods` は残り 3 種に縮小し、正ガード
  （`workspace_declares_spatial_facade_fn_names_only_in_approved_locations`・
  `var_spatial_methods_are_thin_delegations`・`var_spatial_methods_are_reachable_via_facade_only`・
  `compat_sequential_spatial_add_methods_have_approved_signatures`）を新設した。
- **実機 parity**: 新規カーネルなし。CUDA／Metal は既存の `#[ignore]` テストと申し送り
  （`docs/perf/logs/spatial-layers-2159/README.md`）が有効で、本件では未実測。

## §7 実機実測の申し送り

CUDA（DGX Spark GB10）・Metal 実機は本エージェント実行環境に無いため
`#[ignore]` テストを未実行のまま出荷する。実行コマンド・記入欄は
`docs/perf/logs/spatial-layers-2159/README.md` を参照。
