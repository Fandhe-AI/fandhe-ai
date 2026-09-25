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
は承認待ちのまま対象外とし、`crates/facade/src/lib.rs::
SpatialLayersHoldDoctestGuard`（正のプローブ doctest）と
`crates/facade/tests/api_surface.rs` のソース走査（2 テスト＋自己
テスト）で多層固定している。

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

### §2.2 facade 公開・compat::Sequential の add_* — 保留

イシュー #2159 は「facade への 5 個の `add_*` メソッド」を承認事項
として挙げ「承認前に実施しない」と定めている。コメントでの承認も
ないため、`crates/facade/src/compat/sequential.rs` は変更していない。
`Var::conv_transpose1d`／`Var::unflatten`（forward 相当の inherent
メソッド追加。経路 1）も同様に未承認のため見送った。両者を
`SpatialLayersHoldDoctestGuard` の同一 doctest ブロックで併せて
保留固定している（`crates/facade/src/lib.rs` 参照）。

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

- `compat::Sequential::add_*` 5 種・`Var::conv_transpose1d`／
  `Var::unflatten` の facade 公開（承認待ち）
- GPU 専用カーネル（`ConvTranspose1d`・`Upsample` の新モード向け）
- `ZeroPad2d` の負パディング（クロップ）
- `Unflatten` の `-1` 推論
- `Upsample` のスカラー `scale_factor` の全空間軸への broadcast・
  `recompute_scale_factor=False` の座標系
- `output_padding >= stride` の PyTorch 互換化（col2im の `P` 軸
  契約の拡張が必要。#2067 から継承）
- ONNX export（`onnx-interop::export_nn`）での 5 層への対応

## §6 承認事項（未承認として列挙）

1. facade `compat::Sequential` への `add_conv_transpose1d`／
   `add_upsample`／`add_zero_pad2d`／`add_identity`／`add_unflatten`
   の追加（経路 2）と、学習経路（`bind`／`trainable_parameters`／
   `apply_parameters`／`trainable_vars`／`trainable_grads`／
   `contains_resident_unsupported_layer`）への結線
2. 上記に伴う `Var::conv_transpose1d`／`Var::unflatten` の追加
   （経路 1）と保留ガード（`SpatialLayersHoldDoctestGuard`・
   `api_surface.rs` の対応する否定ガード）の撤去

## §7 実機実測の申し送り

CUDA（DGX Spark GB10）・Metal 実機は本エージェント実行環境に無いため
`#[ignore]` テストを未実行のまま出荷する。実行コマンド・記入欄は
`docs/perf/logs/spatial-layers-2159/README.md` を参照。
