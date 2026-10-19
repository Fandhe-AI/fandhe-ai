# `Var` f64 自動微分の CUDA／Metal 実機 parity（イシュー #3168・#3169）

決定記録 `docs/autodiff-var-dtype-multiplexing-design.md` §13「バックエンド別 dispatch（実測記録）」の CUDA 実機側を埋める記録。
対象は `crates/facade/tests/var_f64_autograd_backend_parity.rs` の CUDA 実機テスト 1 本。

## 判定契約

- **bit 完全一致**（`assert_eq!(bits(..))`）。CUDA ネイティブ経路（`typed_ops_f64()` が `Some`）の forward 値と 1 step backward 勾配を、
  ホスト経路（`RawTape::new()`・`NaiveOps`）と比べる。REQ-2 統一複合判定より厳しい。tolerance・baseline は変更しない
- bit 不一致で fail した場合は、テスト自身の bit 一致契約の fail として記録し、ただちに REQ-2 違反とはみなさない（RULE.txt）

## 実行コマンド

CUDA（DGX Spark GB10。#3168。全体の実行は [`../cuda-autodiff-parity-3168/`](../cuda-autodiff-parity-3168/README.md) の `run_cuda_parity.sh`）:

```
cargo test -p fandhe-ai --test var_f64_autograd_backend_parity -- --ignored --exact --nocapture --test-threads=1 cuda_f64_autograd_forward_and_grad_match_host_reference
```

Metal（Apple Silicon。#3169 の担当。MSL が `double` 非対応のため Metal は常にホスト計算で、このテストは Metal 側の dispatch 確認になる）:

```
cargo test -p fandhe-ai --test var_f64_autograd_backend_parity -- --ignored --exact --nocapture --test-threads=1 metal_f64_autograd_forward_and_grad_match_host_reference
```

## 実測記入欄

| 環境 | 実行日 | コミット | 結果 | fail 数 |
|------|--------|---------|------|---------|
| GB10（CUDA） | 2026-10-19 | 9e2f7ddcc（[`cuda/rev-stamp.txt`](./cuda/rev-stamp.txt)） | pass（1/1・0.28s。ログ [`cuda/cuda_f64_autograd_forward_and_grad_match_host_reference.log`](./cuda/cuda_f64_autograd_forward_and_grad_match_host_reference.log)）。GPU 経路かホストフォールバックかは確かめていない | 0 |
| Mac mini（Metal） | 未実施 | - | #3169 の担当 | - |

CPU 版（`cpu_f64_autograd_forward_and_grad_match_host_reference`）は CI で実行済み。
