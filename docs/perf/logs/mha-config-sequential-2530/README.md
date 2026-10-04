# MHA config 付き Sequential の実機 parity（イシュー #2530）

`compat::Sequential::add_multihead_attention_with_config`（`bias=false`・`batch_first=false`）の
forward が CPU と REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で一致するか。
新規カーネルはなく（`MultiheadAttention::from_config` は既存 `Op` の合成）、tolerance は不変。

## 測定コマンド

```sh
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_layers_backend_parity -- --ignored cuda_multihead_attention_config_matches_cpu
# Metal（M4 Max）
cargo test -p fandhe-ai --test compat_sequential_layers_backend_parity -- --ignored metal_multihead_attention_config_matches_cpu
```

## 結果

| 実機 | 結果 | 実測日 |
|------|------|--------|
| CUDA（GB10） | 未実測（実機到達手段なし。申し送り） | - |
| Metal（M4 Max） | 未実測（実機到達手段なし。申し送り） | - |
