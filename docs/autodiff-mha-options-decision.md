# `nn::MultiheadAttention` のオプション（batch_first・kdim/vdim・key_padding_mask）設計判断記録

イシュー #2163（親 #2131「PyTorch／TF 置き換えの API 網羅」）。

## 背景

`nn::MultiheadAttention`（#1640・#1760・#2084）は「rank-3・batch_first
固定・q/k/v 同一次元 `E`・mask は `attn_mask`／`is_causal` のみ」という
最小契約で実装されていた（`crates/autodiff/src/nn/attention.rs` モジュール
doc・`docs/compat-feature-gap.md` の「#1640 の追補」節参照）。本イシューは
この対象外のうち `batch_first`・`kdim`/`vdim`・`key_padding_mask` の 3 点を
埋め、PyTorch `nn.MultiheadAttention` の主要オプションとの差分を縮める。

## 前提の確認（実装着手時点）

- Issue が「`MultiheadAttentionConfig` へ追加」と書いていたが、同名の
  構造体はリポジトリ内に存在しなかった。現状の構築入口は
  `MultiheadAttention::new(embed_dim, num_heads, bias, seed)`・
  `from_parameters(num_heads, q, k, v, out)`・
  `MultiheadAttentionVars::new(num_heads, q, k, v, out)` のみで、全
  フィールドは private だった。
- `key_padding_mask` は PyTorch でも forward 呼び出し時の引数であり、
  構築時設定（config）ではない。本実装では「構築時オプション
  （`batch_first`・`kdim`・`vdim`）は新設 `MultiheadAttentionConfig`」
  「呼び出し時オプション（`key_padding_mask`）は追加 forward メソッドの
  引数」に分けて扱う。

## 設計

### 構築時オプション: `MultiheadAttentionConfig`

`crates/autodiff/src/nn/attention.rs::MultiheadAttentionConfig`。フィール
ドは private（リテラル構築を最初から不可能にし、将来のオプション追加も
非破壊にする）。`MultiheadAttentionConfig::new(embed_dim, num_heads)` を
唯一のコンストラクタとし、`with_bias`／`with_batch_first`／`with_kdim`／
`with_vdim` のビルダーメソッドで上書きする。既定値は `bias=true`・
`batch_first=true`（= 現行挙動。PyTorch 既定 `False` とは異なる）・
`kdim`/`vdim=None`（`embed_dim` へ解決）。

`MultiheadAttention::new(embed_dim, num_heads, bias, seed)`（シグネチャ
不変）は内部で `from_config(&MultiheadAttentionConfig::new(e, h)
.with_bias(bias), seed)` へ委譲する。シード導出は変更前と同一のため、
4 層の初期値は **bit 同一**（単体テスト
`new_matches_from_config_bit_identical`）。

`MultiheadAttention::from_parameters`／`MultiheadAttentionVars::new` は
`validate_linear_projections`／`validate_projection_vars` を緩和し、
`k`/`v` の in_features（`kdim`/`vdim`）を weight 形状から推論するように
した（`q`/`out` は引き続き正方 `[E, E]` を要求）。`kdim = vdim = E` の
入力に対する既存のエラー値は変更していない（既存テスト
`from_parameters_rejects_*`・`new_rejects_*` を無修正のまま green で
確認済み）。

### 呼び出し時オプション: `key_padding_mask`

`MultiheadAttentionVars::forward_with_key_padding_mask`（既存
`forward` はここへ `key_padding_mask=None` で委譲する薄いラッパーへ
変更）。型・形状は `Tensor<bool> [B, S]`（broadcast 不可）。極性は
**`true` = attend**（本モジュールの `attn_mask` と同じ規約。PyTorch
`nn.MultiheadAttention` の `key_padding_mask`〈`True` = 無視〉とは
**逆**）。PyTorch 互換の極性反転フラグは追加しない（Issue の明示
スコープ外）。

`attn_mask`／`key_padding_mask`／`is_causal` の合成は private ヘルパー
`combine_masks_with_key_padding` が `[B, H, L, S]` の attend 極性
テンソルへ組み立て、`sdpa_compose(.., Some(&combined), is_causal=false,
..)` へ渡す。`key_padding_mask: [B, S]` は `Tensor::broadcast_to` が
右詰め整列するため `[B, H, L, S]` へ直接 broadcast すると `B` が `L`
に誤整列する。このため `attn_mask` のみ `broadcast_to` で読み出し、
`key_padding_mask`・`is_causal` は明示的な添字ループ（`B`／`S` 軸を
取り違えない）で合成する。`attn_mask + is_causal` の同時指定は従来
どおり事前拒否する。

### 既定経路の bit 同一保証

`batch_first == true`（既定）かつ `key_padding_mask.is_none()` のとき、
テープに積むノード列は本イシュー着手前の実装と完全に同一にする
（transpose を無条件に挟まない・mask を具体化しない。分岐で回避）。
`checkpoint_backend_bit_identity.rs`・`mha_backend_parity.rs`・
`kv_cache_backend_parity.rs` はこの既定経路を通り、無修正のまま
green を維持する（既存 CI テストが後退非退行の根拠）。

### 未対応経路の fail-closed 化

- `multihead_attention_forward_low_precision`（#2071）
- `MultiheadAttentionVars::forward_with_cache`（#2084。`cache` の更新前
  に拒否し原子性契約を保つ）
- `TransformerEncoderLayer::from_parameters`／
  `TransformerEncoderLayerVars::new`（#2068。`residual = x +
  self_attn(x)` の `[B, L, E]` 前提が崩れないよう構築時に拒否する）

いずれも非既定 config（`batch_first=false` または `kdim`/`vdim !=
embed_dim`）を渡すと `InvalidArgument` で拒否する（無言で batch-first
解釈しない）。

## 承認事項（#2530 で facade 公開済み）

#2163 の PR では実施せず、`MhaOptionsHoldDoctestGuard`（正のプローブ
doctest）・`api_surface.rs` の否定ガードで保留固定していた。ルート #2499 の
一括承認（2026-10-04）を受け、#2530 で実施した（下記「実装記録（#2530）」）。

1. `compat::Sequential` へのオプション付き MHA 追加メソッド（例:
   `add_multihead_attention_with_config(config, seed)`）の facade 公開
2. `MultiheadAttentionConfig` の facade 再エクスポート

承認後は本ガード（`MhaOptionsHoldDoctestGuard`・対応する否定ガード）を
削除し、正の実装へ置き換える。

## 実装記録（#2530・ルート #2499）

- **公開した名前と配置**: `fandhe_ai::compat::MultiheadAttentionConfig`
  （`compat/mod.rs` の `pub use fandhe_ai_autodiff::nn::MultiheadAttentionConfig;`
  1 行・別名なし）と `compat::Sequential::add_multihead_attention_with_config(self,
  config, seed) -> Result<Self, AutodiffError>`。承認形は 2 点のみで、再エクスポートの
  配置（モジュールパス）は記録に定めがなく本イシューで判断した。`nn/mod.rs` は
  `pub use` 集合が完全一致で固定され「nn 層は `Sequential::add_*` 経由で到達する契約」の
  ため避け、メソッドの引数型であり `FitConfig`／`AmpConfig` の先例がある `compat` に置いた。
- **kdim/vdim の拒否**: Sequential の `Module::forward` は self-attention
  （`q = k = v = input`）固定で、`k_proj: [kdim, E]` は `[.., E]` を受けられない。
  使えないのに保存できる層を作らないよう、`kdim`/`vdim != embed_dim` は追加時に
  `InvalidArgument` で拒否する。Sequential で選べるのは `bias`・`batch_first`。
  `key_padding_mask` は呼び出し時オプションで本 API の対象外。
- **manifest**: 新 kind `multihead_attention_config`（params は `embed_dim`・
  `num_heads`・`bias`・`batch_first` の固定 4 キー）を追加（kind 50 → 51 種）。既存 kind
  `multihead_attention` のスキーマ・`add_multihead_attention` の挙動は不変。
  `format_version` は #2714 の `compiled.loss` 拡張と同じく据え置き。
- **学習・推論経路**: 構築した層は `MultiheadAttention` そのもので既存の
  `as_multihead_attention` フックに結線済みのため、`bind`／`trainable_parameters`／
  `apply_parameters`／`SequentialVars::forward` は追加実装なしで `bias=false`・
  `batch_first=false` を保つ。facade の `nn::Module` trait にメソッドは追加しない。
  resident は既存どおり `Unsupported`。AMP（低精度 forward）は非既定 config を
  `InvalidArgument` で拒否する（`compile_with_amp` 後の `fit` が `Err`）。
  ONNX export は従来どおり MHA を `UnsupportedLayer` で拒否する。
- **ガードの反転（旧 → 新）**:
  - `MhaOptionsHoldDoctestGuard`（lib.rs）→ 撤去（`MIN_KNOWN_PROBE_BLOCKS` 22 → 21）
  - `mha_options_hold_doctest_*`・`MHA_OPTIONS_HOLD_PROBE_BODY` → 撤去
  - `compat_sequential_does_not_expose_mha_options_add_methods`（と自己テスト）→
    `compat_sequential_declares_mha_options_add_method_exactly_once`（と自己テスト）
  - `facade_does_not_reexport_multihead_attention_config` →
    `facade_reexports_multihead_attention_config_only_from_compat`（と自己テスト）・
    `multihead_attention_config_is_reachable_via_facade_only`
- **検証**: `cargo test -p fandhe-ai --test api_surface --test compat_sequential_mha_config
  --test compat_sequential_model_io --test compat_sequential_model_io_layers`、
  `cargo test -p fandhe-ai --lib compat::model_io`。CUDA／Metal 実機 parity は
  `#[ignore]` で未実測（`docs/perf/logs/mha-config-sequential-2530/README.md`）。

## スコープ外

- mask 極性の PyTorch 互換化（opt-in フラグ）— Issue の明示スコープ外
- `multihead_attention_forward_low_precision`・`forward_with_cache`／
  `StatefulAttention`・`TransformerEncoderLayer` のオプション対応
  （本 PR では非既定 config を fail-closed 拒否するのみ）
- float 型 `key_padding_mask`・unbatched（rank 2）入力・
  `need_weights`／attention weights 返却・`dropout`・`add_bias_kv`／
  `add_zero_attn`・packed `in_proj_weight`
- PyTorch 既定初期化（xavier_uniform）への整合・PyTorch `state_dict` 名
  （`kdim != E` 時の `q_proj_weight` 等）への対応付け
- GPU 専用カーネル（新規演算なし・既存 `Var` 演算の合成のみのため不要）
- `sdpa_compose` の `Var::scaled_dot_product_attention` への置換
  （既存の追跡事項。`attention.rs` モジュール doc 参照）
- CUDA（DGX Spark GB10）／Metal 実機での `#[ignore]` parity 実測
  （`docs/perf/logs/mha-options-2163/README.md` へ申し送り）

## 検証

- `cargo test -p fandhe-ai-autodiff --lib nn::attention`（45 件。既存
  26 件は無修正のまま green・新規 19 件を追加）
- `cargo test -p fandhe-ai-autodiff --lib nn::transformer_encoder_layer`
  （13 件。既存 13 件は無修正のまま green・fail-closed 化テスト 2 件を
  追加）
- `cargo test -p fandhe-ai-autodiff --test nn_attention --test
  nn_kv_cache`（既存 33 件は無修正のまま green・`nn_kv_cache.rs` へ
  fail-closed テスト 1 件を追加）
- `cargo test -p fandhe-ai --test mha_backend_parity --test
  kv_cache_backend_parity --test checkpoint_backend_bit_identity`
  （既定経路の bit 同一保証の根拠。無修正のまま green）
- `cargo test -p fandhe-ai --test mha_options_backend_parity`（新設。
  kdim/vdim 非対称・batch_first=false・key_padding_mask+is_causal の
  CPU vs NaiveOps parity。CUDA は `#[ignore]`）
- `cargo test -p fandhe-ai --test api_surface`（`MhaOptionsHoldDoctestGuard`
  関連 5 件を追加。既存は無修正のまま green）
- `cargo test --doc -p fandhe-ai`（`MhaOptionsHoldDoctestGuard` の正の
  プローブがコンパイルできること）
