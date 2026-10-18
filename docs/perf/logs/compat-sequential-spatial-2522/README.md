# compat::Sequential の Upsample・ZeroPad2d・Identity 実機 parity 申し送り（イシュー #2522）

本エージェントの実行環境には CUDA／Metal 実機への到達手段がないため、実機テストは未実測のまま
GB10・M4 Max セッションへ申し送る。CPU 側の正しさは Linux 実行可能な
`crates/facade/tests/compat_sequential_spatial_layers.rs` で検証済み。tolerance・baseline は変更しない。

## 判定契約

- `ZeroPad2d`・`Identity`・`Upsample(Nearest)`: 算術を含まない整数添字・コピー系のため CPU と **bit 完全一致**
- `Upsample(Bilinear)`: REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`assert_parity`）

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_spatial_layers_backend_parity -- --ignored cuda_
# Metal（M4 Max）
cargo test -p fandhe-ai --test compat_sequential_spatial_layers_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_spatial_layers_nearest_bit_exact | 未実測 | |
| GB10 | cuda_spatial_layers_bilinear_matches_cpu | 未実測 | |
| M4 Max | metal_spatial_layers_nearest_bit_exact | 未実測 | |
| M4 Max | metal_spatial_layers_bilinear_matches_cpu | 未実測 | |

## CUDA 実測手順（#3163・GB10 未到達のため未実測）

- 判定規則は事前登録の `cuda/RULE.txt` を正とする（実測前に固定）。GB10 行は未実測のまま。
- 実行: `cargo test -p fandhe-ai --test compat_sequential_spatial_layers_backend_parity -- --ignored --nocapture --test-threads=1 cuda_`
- 回収: ログ・`env_info.txt`・`.rev-stamp` をマスク（RULE.txt 記載）して `cuda/` に置き、上の結果記入欄の GB10 行と本節を実測値で更新する。
