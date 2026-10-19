# MHA config 付き Sequential の実機 parity（イシュー #2530）

`compat::Sequential::add_multihead_attention_with_config`（`bias=false`・`batch_first=false`）の
forward が CPU と REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で一致するか。
新規カーネルはなく（`MultiheadAttention::from_config` は既存 `Op` の合成）、tolerance は不変。

## 測定コマンド

```sh
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_layers_backend_parity -- --ignored cuda_multihead_attention_config_matches_cpu
# Metal（Mac mini・Apple M4）
cargo test -p fandhe-ai --test compat_sequential_layers_backend_parity -- --ignored metal_multihead_attention_config_matches_cpu
```

## 結果

| 実機 | 結果 | 実測日 |
|------|------|--------|
| CUDA（GB10） | pass（1/1。#3165・実行コミット a115595c6・`cuda/cuda_multihead_attention_config_matches_cpu.log`） | 2026-10-19 |
| Metal（Mac mini・Apple M4） | pass（1/1・0.05s） | 2026-10-19 |

## CUDA 実測（#3165）

- 日付: 2026-10-19。DGX Spark GB10（aarch64・sm_121）。判定は REQ-2 統一複合判定（tolerance・baseline は不変）。
- コマンド: `cargo test -p fandhe-ai --test compat_sequential_layers_backend_parity -- --ignored --exact --nocapture --test-threads=1 cuda_multihead_attention_config_matches_cpu`
- 結果: pass 1/1（`test result: ok. 1 passed; 0 failed`）。ログ: [`cuda/cuda_multihead_attention_config_matches_cpu.log`](./cuda/cuda_multihead_attention_config_matches_cpu.log)。
- 実行コミット: `a115595c6716b3614869fa157aaa7a3ad8378c76`（[`cuda/rev-stamp.txt`](./cuda/rev-stamp.txt)）。事前登録・全体結果は [`../cuda-facade-parity-3165/`](../cuda-facade-parity-3165/README.md)。
- 備考: GPU 経路かホストフォールバックかは確かめていない（pass は CUDA デバイス指定の実行結果が CPU 参照と判定内で一致したことのみを示す）。

- Metal の実測（#3166。2026-10-19・Mac mini〈Apple M4〉・実行コミット 6570171a・`metal/rev-stamp.txt` は dirty=0）
  - コマンド: `cargo test -p fandhe-ai --test compat_sequential_layers_backend_parity -- --ignored --exact --nocapture --test-threads=1 metal_multihead_attention_config_matches_cpu`（debug ビルド）
  - ログ: `metal/metal_multihead_attention_config_matches_cpu.log`。集計は `../metal-facade-parity-3166/README.md`
  - GPU カーネル経路か Unsupported によるホストフォールバックかはログから断定できず、確かめていない
