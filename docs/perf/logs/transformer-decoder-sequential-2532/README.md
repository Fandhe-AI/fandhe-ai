# TransformerDecoderLayer 付き Sequential の実機 parity（イシュー #2532）

> CUDA は #3165 で GB10 実測済み（2026-10-19・04e48feb・pass 1/1）。Metal は未実測（#3166 段 B）。

`compat::Sequential::add_transformer_decoder_layer`（`tgt = memory = 入力`・mask なし・非 causal 固定）の
forward が CPU と REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で一致するか。
新規カーネルはなく（`TransformerDecoderLayer::forward` は既存の `MultiheadAttention`／`Linear`／`LayerNorm`／`relu` の合成）、tolerance は不変。

## 測定コマンド

```sh
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_layers_backend_parity -- --ignored cuda_transformer_decoder_layer_matches_cpu
# Metal（Mac mini・Apple M4）
cargo test -p fandhe-ai --test compat_sequential_layers_backend_parity -- --ignored metal_transformer_decoder_layer_matches_cpu
```

## 結果

| 実機 | 結果 | 実測日 |
|------|------|--------|
| CUDA（GB10） | pass（1/1。`test result: ok. 1 passed; 0 failed`・0.51s。`cuda/cuda_transformer_decoder_layer_matches_cpu.log`） | 2026-10-19 |
| Metal（Mac mini・Apple M4） | 未実測（Mac 側へ申し送り。#3166・`../metal-facade-parity-3166/`） | - |

## CUDA 実測記入欄（#3165）

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-19 | DGX Spark GB10 | `cargo test -p fandhe-ai --test compat_sequential_layers_backend_parity -- --ignored --exact --nocapture --test-threads=1 cuda_transformer_decoder_layer_matches_cpu` | pass（1/1・0.51s） | GB10（aarch64・sm_121）・ドライバ 580.173.02・CUDA 13.0（nvcc）・rustc 1.97.0。実行コミット 04e48feb（`../cuda-facade-parity-3165/rev-stamp.txt`・`cuda/rev-stamp.txt` とも dirty=0。対象コミット b640875f との `crates/` 等の diff なしを確認）。展開先は専用ツリー（RULE.txt の意図的な逸脱）。ビルドは判定外。GPU 経路（GPU フック経由かホストフォールバックか）は確認していない |

判定条件（REQ-2 統一複合判定・tolerance 定数）は不変。事前登録は `../cuda-facade-parity-3165/RULE.txt`。
