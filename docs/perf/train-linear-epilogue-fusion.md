# 学習 forward の epilogue 融合（Linear + ReLU）— 起動数・CPU 実測

> 役割・参照元: イシュー #1044「Linear の epilogue 融合（bias + ReLU）が
> 学習 forward / backward 経路で適用されることを検証・修正する」の実測
> 記録。設計・結線内容は `docs/kernel-fusion.md` §2.2.1 を正本とし、本
> 文書は同節が参照する実測値・再現コマンドのみを保持する。

## 1. 背景

`docs/kernel-fusion.md` §2.2 の GEMM epilogue 融合（`BackendOps::
gemm_bias_act`）は CPU／CUDA／Metal のいずれも実装済みだったが、学習
forward（`fandhe_ai_facade::compat::sequential::Sequential::forward`／
`SequentialVars::forward`）からは呼ばれておらず、`Linear` → `ReLU` は
`matmul` → `add` → `relu` の 3 カーネル起動のままだった。#1044 で
`Sequential`（`Module::as_relu` による次層先読み）から `LinearVars::
forward_with_activation` へ結線した。

## 2. 起動数（before / after）

| 経路 | before（層 1 個あたり） | after |
|------|------------------------|-------|
| host 常駐 forward（`Sequential::forward`／`SequentialVars::forward`） | `matmul`（`Op::MatMul`）→ `add`（`Op::Add`。bias broadcast leaf `[n]` は `run_fused` 対象外のため per-op フォールバック）→ `relu`（`Op::Relu`）の **3 起動** | `Linear` → `ReLU` に融合される層は `gemm_bias_act`（`Op::LinearAct`）の **1 起動**。`ReLU` が続かない `Linear`（末尾 `Linear` や `Linear` → `Sigmoid`／`Tanh`）は bit-exactness 契約（`docs/inference-forward-fixed-cost-design.md` §3.1）を保つため融合せず従来どおり `matmul` → `add` のまま（レビュー指摘 #1079・PRRT_kwDOTuUCJc6dgIt- で融合対象を `Linear` → `ReLU` に限定） |
| デバイス常駐 forward（`forward_resident`／`DeviceParamStore::linear_forward_with_activation`） | `gemm_resident_rhs`（`Op::LinearResident`。bias 融合済み）→ `relu` の **2 起動** | `gemm_resident_rhs_act` の **1 起動**（CPU のみ。CUDA／Metal は §4 参照） |

機械検証（`BackendOps` 呼び出し回数の直接カウント）:

- `crates/facade/src/compat/sequential.rs::tests::
  sequential_forward_fuses_linear_relu_into_single_gemm_bias_act_launch`
  — `Sequential::new().add_linear(4, 8, _).add_relu().add_linear(8, 2, _)`
  の `bind().forward()` で `gemm_bias_act == 1`（`Linear` → `ReLU` に
  融合された 1 個目のみ）・`gemm == 1`（ReLU が続かない 2 個目の
  `Linear`）・`add == 0`（同 `Linear` の bias 加算は `Var::add` の
  遅延契約により本テストでは実体化されない）・`relu == 0`・
  `run_fused == 0`。

勾配の bit 一致検証:

- `crates/facade/src/compat/sequential.rs::tests::
  sequential_vars_forward_with_activation_grad_matches_manual_composition`
  （host 常駐・`Op::LinearAct` の VJP）
- `crates/autodiff/src/optim/device_store.rs::tests::
  linear_forward_with_activation_relu_grad_matches_manual_relu_composition`
  （デバイス常駐・`Op::LinearResident` の VJP）

いずれも融合経路と非融合合成（`matmul`→`add`→`relu` を手動で組んだ経路）
の forward 出力・パラメータ勾配が `assert_eq!`（bit 完全一致）で一致する
ことを検証する。

## 3. CPU 実測

**計測環境**: Linux x86_64（コンテナ環境。`model name` は
`QEMU Virtual CPU version 2.5+`・論理コア数 12。実行環境が仮想化された
コンテナのため絶対値の参考性は限定的だが、融合 vs 非融合の相対比較
〈本イシューの主眼〉には有効）。

**形状**: batch 64・784 → 256 → ReLU → 10（`docs/inference-forward-
fixed-cost-design.md` §1 が引用する framework-compare の推論プロトコル
と同一形状を学習 forward+backward へ転用）。warmup 20 回・本計測 20 回
の平均秒を 1 サンプルとし、5 回計測（`.claude/rules/coding-rust.md`
「ベンチは 5 回計測の中央値」）。

**計測区間**: forward（`bound.forward(...)` または旧経路の手動合成）→
`MseLoss`（mean）→ `Tape::backward` → `SequentialVars::trainable_grads`
の 1 ステップ（forward + backward のみ。パラメータ更新は含まない）。

**再現コマンド**:

```sh
cargo test -p fandhe-ai --release --test train_step_fusion_bench -- --nocapture
```

**実測値（5 回計測。各行が 1 回の `cargo test` 実行の中央値）**:

| trial | manual（非融合）中央値 [s] | fused（融合）中央値 [s] | speedup |
|------:|---------------------------:|-------------------------:|--------:|
| 1 | 0.044535514 | 0.042760590 | 1.042x |
| 2 | 0.045189459 | 0.042989685 | 1.051x |
| 3 | 0.043456268 | 0.041203250 | 1.055x |
| 4 | 0.043407048 | 0.041370650 | 1.049x |
| 5 | 0.042977951 | 0.041062862 | 1.047x |

- manual（非融合）中央値の中央値: **0.043456268 s**
- fused（融合）中央値の中央値: **0.041370650 s**
- **speedup ≈ 1.050 倍**（融合経路が全 5 trial で非融合合成を上回る）

`docs/kernel-fusion.md` §2.2 の単体 GEMM epilogue 融合実測（CPU
1.46〜2.56 倍）と比べて本ベンチの改善幅が小さいのは、本ベンチの計測
区間が forward だけでなく `Tape::backward`（`matmul_vjp` のホスト
`eval::matmul` 呼び出し。#1046 のスコープ）・`MseLoss`・grad 収集を
含むため、epilogue 融合で削減した割合（GEMM 1 回・elementwise 2 回分の
中間 `Tensor` 割当と再読み出し）が 1 ステップ全体に占める割合が
相対的に小さくなるため（`docs/kernel-fusion.md` §4「性能目標との関係」
の「複合ワークロードでは融合の効果を前提とした性能目標を設定しない」
方針と整合する）。

## 4. Metal / DGX Spark GB10 実機

**未実施**（本リポの開発環境が Linux コンテナのため）。CUDA／Metal は
`gemm_resident_rhs_act`（デバイス常駐 forward の融合入口）を
オーバーライドしておらず、`tensor-core::BackendOps` のデフォルト実装
（`gemm_resident_rhs` → `relu` の 2 起動合成。値は正しい）にフォール
バックする（`docs/kernel-fusion.md` §2.2.1「スコープ外」参照）。host
常駐 forward（`gemm_bias_act` 経由）は CUDA／Metal ともイシュー
#599／#605 で融合カーネルを実装済みのため、本イシューの結線
（`Sequential` の層対検出）は自動的に両バックエンドへも及ぶ——実機
計測（`train_step_fusion_bench.rs` の CUDA／Metal 版）は Mac ／ DGX
Spark セッションでの追実施が必要（`docs/real-hardware-verification-env.md`
の手順に従う）。
