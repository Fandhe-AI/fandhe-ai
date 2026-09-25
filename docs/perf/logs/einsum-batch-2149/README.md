# einsum のバッチ添字縮約（#2149）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-einsum-batch-decision.md` §7「実機 parity の申し送り」
参照。本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::einsum_batch::einsum_batched` の `crates/facade/
tests/einsum_batch_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・
Metal（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする
**計 4 テスト**（forward／backward × CUDA／Metal）は `#[ignore]` のまま
未実測である。

- `metal_einsum_batch_forward_matches_cpu`
- `metal_einsum_batch_backward_matches_cpu`
- `cuda_einsum_batch_forward_matches_cpu`
- `cuda_einsum_batch_backward_matches_cpu`

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test einsum_batch_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test einsum_batch_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト
（`cpu_einsum_batch_forward_matches_naive_reference`・
`cpu_einsum_batch_backward_matches_naive_reference`・
`facade_var_einsum_still_rejects_batch_contraction`）で既に検証済み
（green）。CPU 版は `assert_parity`（REQ-2 統一複合判定）で NaiveOps
（ホスト参照実装）と突き合わせている。

## 期待結果

batch 添字を伴う縮約は rank≥3 `Var::matmul`（`gemm_batched`。イシュー
#1715）への分解として実装されており、既存の `batched_matmul_backend_
parity.rs`（同じ `gemm_batched` を直接呼ぶ既存テスト）が CUDA・Metal
それぞれで pass 済みの実測記録を持つ。GEMM カーネル自体は per-batch
経路と結合順序が異なりうるため（Metal split-K・CUDA Tensor Core 経路
等）bit 完全一致は主張せず、`assert_parity`（REQ-2 統一複合判定。相対
誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で pass することを期待する。

**既知リスク**: backward の損失計算には `mse_loss` を使い `Var::sum`
への依存を避けている（`docs/compat-feature-gap.md` の他の申し送り
README と同じ類型の既知リスク——`Var::sum` の GPU 実装状況によっては
backward が判定不能になりうるため。`ibj,bjk->bik`〈batch 軸が先頭に
ない・非恒等 permute を経由〉の形状で検証する）。

この前提が崩れる場合（`gemm_batched` の実装が変わった、デバイス間で
想定外の丸めが混入した等）は本 README の「期待結果」を更新し、想定
した契約を維持できない事実を型付き findings として PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。
