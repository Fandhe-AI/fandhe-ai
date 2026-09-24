# repeat・tile・flip・roll（#2143）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-rearrange-ops-decision.md` §7・§8「実装記録」参照。
本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::rearrange_ops`（`repeat`／`tile`／`flip`／`roll`）の
`crates/facade/tests/rearrange_ops_backend_parity.rs` のうち CUDA
（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os = "macos")`
限定）を対象とする 2 テストは `#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test rearrange_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test rearrange_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト
（`cpu_forward_matches_naive_reference`・`cpu_forward_preserves_nan_bits`・
`cpu_flip_roll_backward_bit_matches_naive_reference`・
`cpu_repeat_backward_matches_naive_reference_within_tolerance`）で既に
検証済み（green）。

## 期待結果

forward（`index_select`／`broadcast_to` の合成。いずれも値のコピーのみ
で算術を含まない）は CPU・CUDA・Metal 間で構造的に bit 完全一致する
はず（`NaN` の payload も含む）。backward（`Op::Gather` の VJP による
scatter-add）は `flip`／`roll` が各入力要素への寄与 1 つのため bit 一致
するはずだが、`repeat`／`tile` は `r` 個のコピーの勾配を合算するため
GPU の scatter 加算順序次第では厳密な bit 一致にならない可能性がある
——その場合は REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差
1e-5 未満。`fandhe_ai_backend_cpu::parity::assert_parity`）で比較する
（`crates/facade/tests/rearrange_ops_backend_parity.rs` の
`cpu_repeat_backward_matches_naive_reference_within_tolerance` と同じ
判定方式を CUDA／Metal にも適用する）。

この前提が崩れる場合（`gather` の実装が変わった、デバイス間で想定外の
丸めが混入した等）は本 README の「期待結果」を更新し、想定した契約を
維持できない事実を型付き findings として PR へ記録すること（tolerance
の単独緩和は行わない。`.claude/rules/coding-rust.md`）。
