# tril・triu・diag・trace・outer・dot（#2144）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-matrix-ops-decision.md` §7・§8「実装記録」参照。
本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::matrix_ops`（`tril`／`triu`／`diag`／`trace`／
`outer`／`dot`）の `crates/facade/tests/matrix_ops_backend_parity.rs`
のうち CUDA（`Device::Cuda(0)`）・Metal（`Device::Metal`。
`cfg(target_os = "macos")` 限定）を対象とする**計 8 テスト**（CUDA・
Metal 各 4 件で対称。コピー系 4 演算 forward bit 完全一致の
`cuda_copy_ops_forward_matches_cpu_reference`／
`metal_copy_ops_forward_matches_cpu_reference`、縮約系 2 演算 forward
の REQ-2 統一複合判定
`cuda_reduce_ops_forward_matches_cpu_reference`／
`metal_reduce_ops_forward_matches_cpu_reference`、bit 完全一致
backward（`trace` を代表として検証）の
`cuda_bit_exact_backward_matches_cpu_reference`／
`metal_bit_exact_backward_matches_cpu_reference`、`outer` backward の
REQ-2 統一複合判定
`cuda_outer_backward_matches_cpu_reference`／
`metal_outer_backward_matches_cpu_reference`）は `#[ignore]` のまま
未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test matrix_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test matrix_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト
（`cpu_copy_ops_forward_bit_matches_naive_reference`・
`cpu_forward_masked_positions_are_zero_and_nan_bits_preserved`・
`cpu_reduce_ops_forward_matches_naive_reference_within_tolerance`・
`cpu_bit_exact_backward_matches_naive_reference`・
`cpu_outer_backward_matches_naive_reference_within_tolerance`）で既に
検証済み（green）。

## 期待結果

`tril`／`triu`／`diag`（両方向）forward（`masked_fill`／`gather`／
`narrow`／`pad` の合成。いずれも値のコピーまたは定数 0 の埋め込みの
みで算術を含まない）は CPU・CUDA・Metal 間で構造的に bit 完全一致
するはず（`NaN`／`inf` の payload も含む）。backward（`Op::
MaskedFill`／`Op::Gather` の VJP。寄与が高々 1 つ）も同じ理由で
bit 一致するはず。

`trace`（`diag` → `sum`）・`dot`（`mul` → `sum`）forward は `sum` の
縮約順序がバックエンドで異なりうるため REQ-2 の統一複合判定（相対
誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`fandhe_ai_backend_cpu::
parity::assert_parity`）で比較する。`outer` backward（`broadcast_to`
の VJP。軸方向の `reduce_to_shape` 縮約）も同じ判定方式を適用する。

この前提が崩れる場合（`masked_fill`／`gather`／`pad` の実装が変わった、
デバイス間で想定外の丸めが混入した等）は本 README の「期待結果」を
更新し、想定した契約を維持できない事実を型付き findings として PR へ
記録すること（tolerance の単独緩和は行わない。`.claude/rules/
coding-rust.md`）。
