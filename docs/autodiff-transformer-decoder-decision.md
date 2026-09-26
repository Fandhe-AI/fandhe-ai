# `TransformerDecoderLayer`・`Transformer` 設計判断記録

イシュー #2165（親 #2131「PyTorch／TF 置き換えの API 網羅」）。#2068
（`TransformerEncoderLayer`。2026-09-22 クローズ・PR #2211 マージ済み）の
対。

## 背景

#2068 で PyTorch `nn.TransformerEncoderLayer` 相当の 1 層
（`nn::TransformerEncoderLayer`）を実装済みだったが、decoder 側
（`nn.TransformerDecoderLayer`）と encoder-decoder 全体構築層
（`nn.Transformer`）は未実装だった。本イシューはこの 2 層を追加し、
既存の `nn::MultiheadAttention`・`nn::Linear`・`nn::LayerNorm`（いずれも
実装済み）の合成のみで seq2seq 型 Transformer を組めるようにする
（`docs/compat-api-scope.md` §1.2 Tier 1）。

## 前提の確認（実装着手時点）

- 自動運転モード（承認待ち不可）で実装したため、facade 公開面の拡張
  （`compat::Sequential` への `add_transformer_decoder_layer`／
  `add_transformer` の 2 メソッド追加）は実施していない。兄弟イシュー
  （#2158〜#2164）と同じ `*HoldDoctestGuard` 方式で保留を機械的に固定
  した（下記「承認事項」節）。
- base は `origin/main`（`b258df90`〈#2279・MHA オプション〉を含む）を
  使った。`TransformerEncoderLayer`（#2068）は decoder 側と同じ
  「`self_attn` が非既定 `MultiheadAttentionConfig`（`batch_first=false`・
  `kdim`/`vdim != d_model`）だと構築時に拒否する」fail-closed 化
  （イシュー #2163）を既に持っており、decoder 側もこの検証を
  `self_attn`・`multihead_attn` の両方へ適用する。
- ソルト定数（`nn/init.rs`）は実装時点で 0..=11 が使用済みだったため、
  `DEC_SELF_ATTN_SEED_SALT`〜`DEC_LINEAR2_SEED_SALT`（12..=15）・
  `TRANSFORMER_ENC_STACK_SEED_SALT`／`TRANSFORMER_DEC_STACK_SEED_SALT`
  （16・17）を新規採番した。全 18 個のソルトが互いに異なることは
  `nn::init::tests::all_seed_salts_are_pairwise_distinct` で固定する。

## 設計

### `TransformerDecoderLayer`

`crates/autodiff/src/nn/transformer_decoder_layer.rs`。フィールド名は
PyTorch と同じ: `self_attn`・`multihead_attn`（cross-attention）・
`linear1`・`linear2`・`norm1`・`norm2`・`norm3`（いずれも
`MultiheadAttention`／`Linear`／`LayerNorm`）・`activation`
（`TransformerEncoderLayer` の `FeedForwardActivation` を再利用）。

forward 順序は post-norm 固定（PyTorch `nn.TransformerDecoderLayer` の
既定 `norm_first=False` と同じ）:

1. `x1 = norm1(tgt + self_attn(tgt, tgt, tgt, tgt_mask, tgt_is_causal))`
2. `x2 = norm2(x1 + multihead_attn(x1, memory, memory, memory_mask, memory_is_causal))`
3. `y = norm3(x2 + linear2(act(linear1(x2))))`

`self_attn`・`multihead_attn` は両方とも `batch_first == true`・
`kdim == vdim == d_model` を構築時に要求する（イシュー #2163 の
fail-closed 方針を decoder 側へ横展開。`residual = x + attn(..)` が
`[B, L, E]` 同士の加算を前提とするため）。加えて 2 つの MHA の
`embed_dim` が一致することを検証する。

関連関数の引数が 8 個（self を含めない）になり clippy の既定閾値
（7）を超えるため、`#[allow(clippy::too_many_arguments)]` は使わず
子層をまとめる構造体（`TransformerDecoderLayerParts`／
`TransformerDecoderLayerVarsParts`。いずれも pub フィールド）を経由する
形にした。

mask 極性は本クレートの `MultiheadAttention` と同じ `true` = attend
規約（PyTorch の `True` = blocked とは逆）。`attn_mask` と `is_causal`
の同時指定は `MultiheadAttentionVars::forward` が拒否する。

シード導出は単一の呼び出し `seed` から `nn/init.rs` の 4 ソルト
（`DEC_SELF_ATTN_SEED_SALT`〜`DEC_LINEAR2_SEED_SALT`）で 4 系統の
独立した構築シードを導出する（`ENC_ATTN_SEED_SALT` 等と同じ「2 段の
`derive_seed` 合成」構造）。

### `Transformer`

`crates/autodiff/src/nn/transformer.rs`。PyTorch `nn.Transformer` と
同じ構成（**最終 LayerNorm を 2 つ含む**）: `encoder_layers:
Vec<TransformerEncoderLayer>` + `encoder_norm: LayerNorm` →
`decoder_layers: Vec<TransformerDecoderLayer>` + `decoder_norm:
LayerNorm`。中間コンテナ型（`TransformerEncoder`／`TransformerDecoder`
スタック）は公開しない（実装計画 §2.2「対象外」）。

構築は `TransformerConfig`（`#[non_exhaustive]`・ビルダー方式。
`MultiheadAttentionConfig` と同型）経由の `Transformer::new(&config,
seed)` のみ（引数 8 個の関連関数を避けるため）。既定値は PyTorch と
同じ `num_encoder_layers=6`・`num_decoder_layers=6`・
`dim_feedforward=2048`・`activation=Relu`・`eps=LAYER_NORM_DEFAULT_EPS`
（`1e-5`）。`num_encoder_layers == 0`／`num_decoder_layers == 0` は
拒否する。

層ごとのシードは `derive_seed(derive_seed(seed,
TRANSFORMER_*_STACK_SEED_SALT), i)`（`RNN_STACK_SEED_SALT` と同型の
2 段合成。`RNN_STACK_SEED_SALT` と異なり index 0 特例は設けない——
`Transformer` に単層との bit 一致契約はないため）。

forward: ①`src` を `encoder_layers` へ順に通す（`src_mask`・非
causal） → ②`encoder_norm` → `memory` → ③`tgt` を `decoder_layers` へ
順に通す（`tgt_mask`・`memory_mask`・`tgt_is_causal`・
`memory_is_causal = false` 固定） → ④`decoder_norm`。

`named_parameters`／`children`／`set_parameter` は
`encoder.layers.{i}.*` → `encoder.norm.*` → `decoder.layers.{i}.*` →
`decoder.norm.*` の接頭辞契約（PyTorch `state_dict` キー体系に揃える）。
`set_parameter` の index 部分は `ModuleList::set_parameter` と同型の
index-parse + range-check（`set_parameter_in_layers` ヘルパー。範囲外・
非数値は `AutodiffError::InvalidArgument`）。

### 単一入力の `Module::forward` の意味論

`TransformerDecoderLayer`・`Transformer` の `impl Module` の `forward`
（1 引数）は既存の慣習（`MultiheadAttention`・`TransformerEncoderLayer`
は `q = k = v = input`／`self-attn(input)`）に揃え、`tgt = memory =
input`（decoder）・`src = tgt = input`（Transformer）とする。mask は
全て `None`、非 causal。単体テスト
（`module_forward_matches_bind_forward_with_input_as_tgt_and_memory`・
`module_forward_matches_bind_forward_with_input_as_src_and_tgt`）で
`bind().forward(..)` との bit 一致を固定する。

## 承認事項（facade 公開は保留）

以下は本 PR では実施していない。承認まで
`crates/facade/src/lib.rs::TransformerDecoderHoldDoctestGuard`（正の
プローブ doctest）・`crates/facade/tests/api_surface.rs` の否定ガード
（ソース走査）で固定する。

1. `compat::Sequential` への `add_transformer_decoder_layer`／
   `add_transformer` の facade 公開
2. `TransformerDecoderLayer`／`Transformer`／`TransformerConfig` の
   facade 再エクスポート

承認後は本ガード（`TransformerDecoderHoldDoctestGuard`・対応する
否定ガード）を削除し、`Sequential::add_*` の結線（`bind`／
`SequentialVars::forward`／`trainable_vars`／`trainable_grads`／
`apply_parameters`／`contains_resident_unsupported_layer`）を別 PR で
行う。

## スコープ外

- pre-norm（`norm_first=True`）・Dropout 結線・`batch_first=False`
- `tgt_key_padding_mask`／`memory_key_padding_mask`／
  `src_key_padding_mask`・`src_is_causal`／`memory_is_causal`
  （`Transformer::forward` 引数として。明示 mask で代替できる）
- 独立した `TransformerEncoder`／`TransformerDecoder` スタック型の
  公開・カスタム encoder／decoder の注入
- `forward_host`（tape 不要推論経路）・AMP 低精度経路・KV キャッシュ
  付きの自己回帰デコード（#2083／#2084 系）
- GPU 専用カーネル（新規演算なし・既存 `Var` 演算の合成のみのため
  不要）
- CUDA（DGX Spark GB10）／Metal 実機での `#[ignore]` parity 実測
  （`docs/perf/logs/transformer-decoder-2165/README.md` へ申し送り）

## 検証

- `cargo test -p fandhe-ai-autodiff --lib nn::transformer`（51 件。
  `nn::transformer_decoder_layer`・`nn::transformer` の新規テスト）
- `cargo test -p fandhe-ai-autodiff --lib nn::init::tests::all_seed_salts_are_pairwise_distinct`
- `cargo test -p fandhe-ai-autodiff --test nn_transformer_decoder`
  （7 件。`nn::Sequential` への搭載・`state_dict`／`load_state_dict`
  往復・`freeze`）
- `cargo test -p fandhe-ai-autodiff --test nn_module_introspection`
  （35 件。decoder／Transformer の `children`／`named_modules` 回帰を
  追加。既存は無修正のまま green）
- `cargo test -p fandhe-ai --test transformer_decoder_backend_parity`
  （新設。`TransformerDecoderLayer`〈`S != L` の memory・causal あり〉・
  `Transformer`〈encoder 2 層・decoder 2 層〉の CPU vs NaiveOps
  forward・backward parity。CUDA／Metal は `#[ignore]`）
- `cargo test -p fandhe-ai --test api_surface`
  （`TransformerDecoderHoldDoctestGuard` 関連 4 件を追加。既存は無修正
  のまま green）
- `cargo test -p fandhe-ai --doc`（`TransformerDecoderHoldDoctestGuard`
  の正のプローブがコンパイルできること）
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo fmt --all -- --check`
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked`
- `git diff origin/main --stat -- Cargo.toml Cargo.lock docs/spec`
  が空であることを確認済み（依存・spec に変更なし）
