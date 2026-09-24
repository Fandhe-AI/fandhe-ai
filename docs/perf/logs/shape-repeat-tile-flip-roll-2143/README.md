# repeat・tile・flip・roll（#2143）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-rearrange-ops-decision.md` §7・§8「実装記録」参照。
本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::rearrange_ops`（`repeat`／`tile`／`flip`／`roll`）の
`crates/facade/tests/rearrange_ops_backend_parity.rs` のうち CUDA
（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os = "macos")`
限定）を対象とする**計 6 テスト**（CUDA・Metal 各 3 件で対称。forward
全 4 種 bit 完全一致の `cuda_forward_matches_cpu_reference`／
`metal_forward_matches_cpu_reference`、`flip`／`roll` backward bit 完全
一致の `cuda_flip_roll_backward_matches_cpu_reference`／
`metal_flip_roll_backward_matches_cpu_reference`、`repeat`／`tile`
backward の REQ-2 統一複合判定
`cuda_repeat_tile_backward_matches_cpu_reference`／
`metal_repeat_tile_backward_matches_cpu_reference`。`flip`／`roll`
backward 2 件と `repeat`／`tile` backward 2 件はいずれもイシュー #2143
レビュー指摘を受けて追加し、下記「期待結果」節が要求するバックエンド
別の実機カバレッジを満たす）は `#[ignore]` のまま未実測である。

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
`cpu_repeat_tile_backward_matches_naive_reference_within_tolerance`）で既に
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
`cpu_repeat_tile_backward_matches_naive_reference_within_tolerance` と
同じ判定方式を `cuda_repeat_tile_backward_matches_cpu_reference`／
`metal_repeat_tile_backward_matches_cpu_reference` として CUDA／Metal
にも適用する）。

この前提が崩れる場合（`gather` の実装が変わった、デバイス間で想定外の
丸めが混入した等）は本 README の「期待結果」を更新し、想定した契約を
維持できない事実を型付き findings として PR へ記録すること（tolerance
の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 追記（PR #2256 codex-review 2 巡目の指摘対応）

当初の `#[ignore]` テストは `flip`／`roll` backward の実機カバレッジが
`roll` のみで `flip` を欠いていた（テスト名・doc が両演算を謳いながら
`flip` backward の実機比較を実行していなかった）。CPU 側の同型テスト
（`cpu_flip_roll_backward_bit_matches_naive_reference`）と対称になる
よう、`cuda_roll_backward_matches_cpu_reference`／
`metal_roll_backward_matches_cpu_reference` を
`cuda_flip_roll_backward_matches_cpu_reference`／
`metal_flip_roll_backward_matches_cpu_reference` へ改名し、`flip`
backward の比較を追加した（テスト数自体は変わらず各 3 件・計 6 件の
まま）。
