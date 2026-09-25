# mish・hardtanh・relu6・prelu・glu（#2146）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-activation-ops-decision.md` §7・§8「実装記録」参照。
本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::activation_ops`（`mish`／`hardtanh`／`relu6`／
`prelu`／`glu`）の `crates/facade/tests/
activation_ops_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・
Metal（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする
**計 8 テスト**（CUDA・Metal 各 4 件で対称。bit 完全一致 forward 3 演算
〈`hardtanh`／`relu6`／`prelu` forward〉の
`cuda_bit_exact_forward_matches_cpu_reference`／
`metal_bit_exact_forward_matches_cpu_reference`、bit 完全一致
backward（`hardtanh`／`relu6`／`prelu` 入力勾配の 3 セルを個別に検証。
「代表 1 演算での省略」はしない。#2144 codex-review 指摘の横展開）の
`cuda_bit_exact_backward_matches_cpu_reference`／
`metal_bit_exact_backward_matches_cpu_reference`、REQ-2 統一複合判定
forward 2 演算〈`mish`／`glu`〉の
`cuda_req2_forward_matches_cpu_reference`／
`metal_req2_forward_matches_cpu_reference`、REQ-2 統一複合判定
backward（`mish`／`glu`・`prelu` の `weight` 勾配の 3 セルを個別に
検証）の
`cuda_req2_backward_matches_cpu_reference`／
`metal_req2_backward_matches_cpu_reference`）は `#[ignore]` のまま
未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test activation_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test activation_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト
（`cpu_bit_exact_forward_matches_naive_reference`・
`cpu_bit_exact_backward_matches_naive_reference`・
`cpu_req2_forward_matches_naive_reference_within_tolerance`・
`cpu_req2_backward_matches_naive_reference_within_tolerance`）で既に
検証済み（green）。

## 期待結果

`hardtanh`／`relu6`（`Var::clamp` と bit 完全一致。`activation_ops::
hardtanh` doc 参照）・`prelu`（forward・入力勾配。選択と乗算 1 回の
みで算術を含まない）forward・backward は CPU・CUDA・Metal 間で構造的
に bit 完全一致するはず（`where_cond`・`clamp`・`mul` はいずれも
既存カーネルの数値契約が全バックエンドで統一されているため）。

`mish`（`softplus`／`tanh` を経由）・`glu`（`sigmoid` を経由）
forward・backward、`prelu` の `weight` 勾配（`reduce_to_shape` の
縮約を経由）は超越関数または縮約順序の違いにより REQ-2 の統一複合
判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
`fandhe_ai_backend_cpu::parity::assert_parity`）で比較する。

この前提が崩れる場合（`clamp`／`where_cond`／`mul` の実装が変わった、
デバイス間で想定外の丸めが混入した等）は本 README の「期待結果」を
更新し、想定した契約を維持できない事実を型付き findings として PR へ
記録すること（tolerance の単独緩和は行わない。`.claude/rules/
coding-rust.md`）。
