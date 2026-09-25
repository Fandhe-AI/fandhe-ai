# linalg_ops（#2150）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-linalg-ops-decision.md` §7「実装記録」参照。
本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::linalg_ops`（`eigh`／`slogdet`／`pinv`／
`matrix_rank`／`lstsq`）の
`crates/facade/tests/linalg_ops_backend_parity.rs` のうち
CUDA（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os =
"macos")` 限定）を対象とする 4 テスト（`eigh`／`pinv` forward ×
CUDA／Metal）は `#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test linalg_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test linalg_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（13 件。
`cpu_eigh_*`・`cpu_slogdet_*`・`cpu_pinv_*`・`cpu_matrix_rank_*`・
`cpu_lstsq_*`）で既に検証済み（green）。

## 期待結果

- 5 演算とも `BackendOps::linalg_eigh`／`linalg_slogdet`／`linalg_pinv`／
  `linalg_matrix_rank`／`linalg_lstsq` が CUDA／Metal では既定
  `Unsupported` を返すため（GPU カーネル未実装。`crates/tensor-core/
  src/backend_ops.rs` の既定実装）、実機テストは常にホスト参照実装
  （`eval::linalg::eigh` 等）へフォールバックする経路になる想定——
  CUDA／Metal 側に専用カーネルが無いため、CPU との差は「同じホスト
  参照実装をどのデバイスの `Tape` 経由で呼ぶか」の違いに留まり、
  数値的な乖離は生じにくいと予想される（REQ-2 統一複合判定を満たす
  見込み）。
- `matrix_rank`／`slogdet` の符号は区分定数のため bit 完全一致する
  見込み。

この前提が崩れる場合（想定外の丸めが混入した、フォールバック経路が
機能していない等）は本 README の「期待結果」を更新し、REQ-2 判定を
外れた事実を型付き findings として PR へ記録すること（tolerance の
単独緩和は行わない。`.claude/rules/coding-rust.md`）。
