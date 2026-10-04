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

## 承認事項（facade 公開。#2532・#2533 で実施済み）

以下は #2165 の PR では実施せず保留していた（#2532 で decoder 1 層、#2533 で残りを公開済み。末尾の実装記録参照）。承認まで
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

## 実装記録（#2532・親 #2531）

承認事項のうち decoder 1 層の分を facade へ公開した（ルート #2499 の一括承認。`docs/compat-api-scope.md` §5 経路 2）。

- **公開名**: `fandhe_ai::nn::TransformerDecoderLayer`・`fandhe_ai::nn::TransformerConfig`（`crates/facade/src/nn/mod.rs` の独立した 1 文の `pub use`）と
  `compat::Sequential::add_transformer_decoder_layer(d_model, num_heads, dim_feedforward, seed)`。
- **配置の理由**: autodiff 側のパス（`nn::*`）の鏡写しにした。新しい `pub mod` は作らない（`nn_mod_declares_only_init_and_rnn_submodules` と、全 `*HoldDoctestGuard` の
  glob 一覧への波及を避けるため）。
- **Sequential 内の意味論**: `tgt = memory = 直前層の出力`・mask なし・非 causal・`relu`・eps は `LAYER_NORM_DEFAULT_EPS` 固定。
  `Module::forward` と同じ呼び出しなので `predict` と `bind().forward` は bit 一致する。
- **保存・復元**: kind `transformer_decoder_layer`（params は `d_model`・`num_heads`・`dim_feedforward`）を 31 種目以降として allowlist へ追加（合計 52 種）。
  `LayerSpec::Unsupported` は `add_module` 専用のため使わなかった。
- **ガードの縮小**: `TransformerDecoderHoldDoctestGuard` のプローブと `compat_sequential_does_not_expose_transformer_decoder_add_methods` は `Transformer`／
  `add_transformer` だけに縮めた（#2533 が残りを反転する）。正ガードは `api_surface.rs` の
  `facade_reexports_transformer_decoder_items_only_in_approved_shape`・`compat_sequential_declares_add_transformer_decoder_layer_exactly_once` ほか。
- **限界（後続候補）**: `TransformerDecoderLayer::new` は `FeedForwardActivation`、`from_parameters` は `MultiheadAttention`／`Linear`／`LayerNorm`、`bind` は生の `Tape` を取るため、
  facade 利用者は型名・アクセサには触れるが、単体での構築・forward は行えない（承認形の範囲外のため再エクスポート・委譲は追加していない。必要になれば本記録への追記と承認が要る）。
- **#2533 に残る範囲**: `Transformer`／`add_transformer`・その保存往復・残りの保留ガードの削除。
- 実機 parity（CUDA／Metal）は `docs/perf/logs/transformer-decoder-sequential-2532/README.md` へ申し送り。

## 実装記録（#2533・親 #2531）

承認事項の残り（`Transformer`／`add_transformer`・保存往復・保留ガードの撤去）を実装した（ルート #2499 の一括承認。`docs/compat-api-scope.md` §5 経路 2）。

- **公開名**: `fandhe_ai::nn::Transformer`（`crates/facade/src/nn/mod.rs` の 1 文の `pub use` に追加。`TransformerConfig`・`TransformerDecoderLayer` と合わせ 3 名形）と
  `compat::Sequential::add_transformer(config: TransformerConfig, seed: u64) -> Result<Self, AutodiffError>`。
- **シグネチャを config 方式にした理由**: 本記録の承認事項は「構築は `TransformerConfig` 経由の `Transformer::new(&config, seed)` のみ」を前提としており、
  位置引数にすると 7 個以上になりビルダーと重複する。`TransformerConfig` は `Copy` で、`add_multihead_attention_with_config(config, seed)` の前例に合わせて値渡しとした。
- **活性化**: `relu` のみ。`with_activation` で他を指定した config は `InvalidArgument`（`FeedForwardActivation` は facade から到達できず、manifest に活性化を持たせない。#2530 の kdim/vdim 拒否と同型）。
- **eps**: `with_eps` は facade から到達できるため `LayerSpec::Transformer` に持たせ、保存・復元する（往復の bit 一致に必要）。非有限値は保存時に `UnsupportedModel`。
- **Sequential 内の意味論**: `src = tgt = 直前層の出力`・mask なし・非 causal。`Module::forward` と同じ呼び出しのため `predict` と `bind().forward` は bit 一致する。
- **パラメータ数**: `16 * N_enc + 2 + 26 * N_dec + 2`。順序は `Transformer::named_parameters`（encoder 層 → `encoder.norm` → decoder 層 → `decoder.norm`）。
- **保存・復元**: kind `transformer`（params 6 キー）を allowlist へ追加（合計 53 種）。load 側は `layer_parameter_count`（checked 算術）で全層のキー数を実キー数と照合してから期待キー列を作り、
  0 層は `spec_from_kind` で早期拒否する。`MAX_ARRAY_LEN`・`MAX_OBJECT_KEYS`・`MAX_LAYERS` は変更しない。
- **リファクタ（挙動不変）**: `trainable_vars`／`trainable_grads` の encoder／decoder 分岐を Transformer 分岐と共有するヘルパー関数へ抽出。`model_io.rs` の TE／TD のキー列も関数化した。
- **ガードの反転**: `TransformerDecoderHoldDoctestGuard`（`lib.rs`）と、`api_surface.rs` の `transformer_decoder_hold_doctest_globs_all_pub_modules`・
  `transformer_decoder_hold_doctest_probe_body_matches_fixed_contract`・`compat_sequential_does_not_expose_transformer_decoder_add_methods`（および `_detects_offense`）・
  `TRANSFORMER_DECODER_HOLD_PROBE_BODY` を撤去。正ガードは `facade_reexports_transformer_decoder_items_only_in_approved_shape`（3 名形）・
  `compat_sequential_declares_add_transformer_exactly_once`・`transformer_decoder_types_are_reachable_via_facade_only`。
  `hold_doctest_probe_blocks_reference_every_glob_imported_item` の検出数下限は保留ガード 1 件の撤去に合わせて 21 → 20。
- **対象外**: 2 入力 API・mask／causal・pre-norm・Dropout・Gelu の選択・`FeedForwardActivation` の再エクスポート・`Tape` 委譲による単体 forward・GPU 専用カーネル・ONNX export 対応。
- 実機 parity（CUDA／Metal）は `docs/perf/logs/transformer-sequential-2533/README.md` へ申し送り。
