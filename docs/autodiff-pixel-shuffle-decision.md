# PixelShuffle・PixelUnshuffle の設計判断記録

イシュー #2162（親 #2131）。`docs/autodiff-spatial-layers-decision.md`
（#2159）と同型の記録。

## §0 結論

PyTorch の `nn.PixelShuffle`・`nn.PixelUnshuffle` に相当する 2 層を、
`fandhe_ai_autodiff` の `nn` モジュール（`crates/autodiff/src/nn/
pixel_shuffle.rs`）へ追加した。いずれも既存 `Var` 演算
（`Var::reshape`・`Var::permute`・`Var::contiguous`）の薄い合成であり、
新規 `Op`・`BackendOps` メソッド・VJP・GPU 専用カーネルは追加して
いない。`Var` に inherent の `pub fn` は追加していない（`Var` は
facade から再エクスポートされるため。#2159 の先例）。facade 公開
（`compat::Sequential::add_pixel_shuffle`／`add_pixel_unshuffle`・
`Var::pixel_shuffle`／`Var::pixel_unshuffle` の委譲メソッド）は承認
待ちのまま対象外とし、`crates/facade/src/lib.rs::
PixelShuffleHoldDoctestGuard`（正のプローブ doctest）と
`crates/facade/tests/api_surface.rs` のソース走査（2 テスト＋自己
テスト）で多層固定している。

## §1 背景

イシュー #2162・親 #2131 のコメントはいずれも 0 件（2026-09-25 時点。
着手前に `gh issue view 2162/2131` で確認済み）。親 #2131 はこの
ツリーでの facade 公開面の拡張を「設計判断記録 → 承認 → 実装」の
2 段階と定めているため、本実装は内部クレート限定に倒す（#2159 §1 と
同じ判断）。

## §2 設計判断

### §2.1 演算の分解

PyTorch と同じ軸の並びを採用する:
`out[.., c, h*r+i, w*r+j] = in[.., c*r*r+i*r+j, h, w]`（`PixelShuffle`。
`PixelUnshuffle` はこの逆変換）。

| 層 | ファイル | 手順 |
|---|---|---|
| `PixelShuffle` | `nn/pixel_shuffle.rs` | `contiguous → reshape [B.., C, r, r, H, W] → permute [0..nb, nb, nb+3, nb+1, nb+4, nb+2] → contiguous → reshape [B.., C, H*r, W*r]` |
| `PixelUnshuffle` | `nn/pixel_shuffle.rs` | `contiguous → reshape [B.., C, H, r, W, r] → permute [0..nb, nb, nb+2, nb+4, nb+1, nb+3] → contiguous → reshape [B.., C*r*r, H, W]` |

先頭のバッチ軸（0 本以上。`nb = rank - 3`）はそのまま素通りする
（rank 3 以上の `[*, C, H, W]` を受理。rank 2 以下は拒否）。VJP は
reshape・permute（view）と `Op::Contiguous`（恒等パススルー）の既存
VJP 合成として自動的に成り立つため `grad.rs` は変更していない。

「CPU 先行・CUDA／Metal は `Unsupported` フォールバック」という
イシュー #2162 の契約との関係: `reshape`／`permute` は zero-copy の
view ノード、`Op::Contiguous` はホスト側で eager 実行されるノードの
ため、CUDA／Metal の tape でも追加実装なしで到達できる（#2159 の
`Unflatten`／`Identity` と同じ判断）。

shape 検査は `pixel_shuffle_out_shape`／`pixel_unshuffle_out_shape`
（`forward`／`forward_host` で共有。`.claude/rules/security.md` A08）
に集約し、`contiguous`／`reshape`／`permute` を積む前に rank・
`checked_mul` オーバーフロー・整除性を検査してから出力 shape を
確定する（`Err` 経路で孤児ノードを残さない。#2159 `ConvTranspose1d`
レビュー指摘と同じ規律）。

`PixelShuffle::new(upscale_factor)`／`PixelUnshuffle::new
(downscale_factor)` は `0` を [`AutodiffError::InvalidArgument`] で
拒否する（rank・整除性の検査は入力 shape に依存するため forward 時へ
遅延させる `Unflatten::new` と同じ契約）。

### §2.2 facade 公開・compat::Sequential の add_* — 保留

イシュー #2162 は「facade への 2 個の `add_*` メソッド」を承認事項
として挙げ「承認前に実施しない」と定めている。コメントでの承認も
ないため、`crates/facade/src/compat/sequential.rs` は変更していない。
`Var::pixel_shuffle`／`Var::pixel_unshuffle`（forward 相当の
inherent メソッド追加。経路 1）も同様に未承認のため見送った。両者を
`PixelShuffleHoldDoctestGuard` の同一 doctest ブロックで併せて保留
固定している（`crates/facade/src/lib.rs` 参照）。

### §2.3 `Module` trait への統合

`PixelShuffle`・`PixelUnshuffle` ともパラメータを持たない構造体の
ため `as_*` フックは追加していない（`Identity`・`Unflatten` と同型）。
`forward`・`forward_host`（bit 完全一致契約。`supports_forward_host`
は既定 `true` のまま）を実装した。

## §3 契約（変更禁止・維持）

- tolerance・baseline・`Cargo.toml` の依存・ガードレール閾値・
  `docs/spec/` は変更していない
- 新規 `unsafe` なし・依存追加なし
- crates.io 公開済みの `fandhe-ai =0.9.0`・`fandhe-ai-autodiff` の
  公開 API は非破壊（新しい pub 型の追加と `Module` trait への既定
  実装付きメソッドの追加のみ）

## §4 テスト

- `crates/autodiff/src/nn/pixel_shuffle.rs` の unit test（構築検査・
  r=2 手計算例・r=1 恒等・往復・rank3／rank5 の素朴な添字式参照実装
  との bit 一致・shape 関数の境界値・`forward_host` との bit 一致・
  非 contiguous 入力の受理）
- `crates/autodiff/tests/nn_pixel_shuffle.rs`: shuffle↔unshuffle 往復
  （両方向）の bit 完全一致、backward（`sum(shuffle(x)*g)` の勾配が
  `unshuffle(g)` と bit 一致・逆方向も対称確認）、非 contiguous 入力
  の受理、拒否経路（rank・非整除・オーバーフロー）で孤児ノードを
  残さないこと、`nn::Sequential`（Conv2d → PixelShuffle →
  PixelUnshuffle）への統合
- `crates/facade/tests/pixel_shuffle_backend_parity.rs`: CPU
  （`CpuBackendOps`）対 `NaiveOps` の forward／backward parity（純粋な
  コピー・view の合成のみのため bit 完全一致）と、`cuda_*`／
  `metal_*` の `#[ignore]` parity（実機未実測。`docs/perf/logs/
  pixel-shuffle-2162/README.md` 参照）
- `crates/facade/src/lib.rs::PixelShuffleHoldDoctestGuard`（正の
  プローブ doctest）・`crates/facade/tests/api_surface.rs` の
  `pixel_shuffle_hold_doctest_globs_all_pub_modules`・
  `pixel_shuffle_hold_doctest_probe_body_matches_fixed_contract`・
  `compat_sequential_does_not_expose_pixel_shuffle_add_methods`
  （＋自己テスト）

## §5 スコープ外（`out-of-scope-tracking.md` に従う。Issue 起票は
ユーザー承認後）

- `compat::Sequential::add_pixel_shuffle`／`add_pixel_unshuffle`・
  `Var::pixel_shuffle`／`Var::pixel_unshuffle` の facade 公開
  （承認待ち）
- GPU 専用の並べ替えカーネル
- ONNX `DepthToSpace`／`SpaceToDepth` との相互運用（`onnx-interop`
  の import／export）
- CUDA（DGX Spark GB10）・Metal 実機での `#[ignore]` parity の実測
  （申し送り README へ）

## §6 承認事項（未承認として列挙）

1. facade `compat::Sequential` への `add_pixel_shuffle`／
   `add_pixel_unshuffle` の追加（経路 2）と、学習経路（`bind`／
   `trainable_parameters`／`apply_parameters`／`trainable_vars`／
   `trainable_grads`／`contains_resident_unsupported_layer`）への
   結線
2. 上記に伴う `Var::pixel_shuffle`／`Var::pixel_unshuffle` の追加
   （経路 1）と保留ガード（`PixelShuffleHoldDoctestGuard`・
   `api_surface.rs` の対応する否定ガード）の撤去

## §7 実機実測の申し送り

CUDA（DGX Spark GB10）・Metal 実機は本エージェント実行環境に無いため
`#[ignore]` テストを未実行のまま出荷する。実行コマンド・記入欄は
`docs/perf/logs/pixel-shuffle-2162/README.md` を参照。
