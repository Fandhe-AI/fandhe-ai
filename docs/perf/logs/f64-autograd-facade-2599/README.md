# facade 公開 `TapeF64`／`VarF64`／`GradientsF64` の GPU 実機検証（イシュー #2599）

## 背景

- f64 専用の独立自動微分グラフ（`fandhe_ai_autodiff::f64_autograd`。#2195・#2196）を、決定記録 `docs/autodiff-var-dtype-multiplexing-design.md` §4.1・§10.1 の推奨案 D-2 で facade newtype として公開した（#2599）。新規カーネルはなく、既存の `TypedOps<f64>`（CPU・CUDA）とホスト参照実装（Metal は常にホスト計算）への 1 式委譲のみ
- 本環境（Linux・実機なし）では CPU のみ実行できた。CUDA（GB10）・Metal の parity は `#[ignore]` テストとして用意した。CUDA は #3168、Metal は #3169 で 2026-10-19 に Mac mini（Apple M4）で実測済み（下の記入欄。Metal は MSL が `double` 非対応でホスト計算になるため、ホスト経路が参照と一致することの確認）
- 判定は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。tolerance・baseline は変更しない
- 先行する内部クレート経由の実機手順は `docs/perf/logs/var-f64-gemm-reduction-2196/README.md`

## 実行コマンド

CUDA（DGX Spark GB10）:

```
cargo test -p fandhe-ai --test f64_autograd_facade -- --ignored f64_autograd_cuda_matches_cpu
```

Metal（Apple Silicon。`f64_autograd_metal_matches_cpu` は macOS のみ。`--ignored` 単独だと CUDA 専用テストも
実行され CUDA なし環境で失敗するため、テスト名で絞る）:

```
cargo test -p fandhe-ai --test f64_autograd_facade -- --ignored f64_autograd_metal_matches_cpu
```

## 記入欄

| 環境 | 実行日 | コミット | 結果 | fail 要素数 |
|------|--------|---------|------|-------------|
| GB10（CUDA） | 未実施 | - | - | - |
| Mac mini（Apple M4・Metal） | 2026-10-19 | bf894108a（[`metal/rev-stamp.txt`](./metal/rev-stamp.txt)） | `f64_autograd_metal_matches_cpu` pass（1/1・0.03s。`--exact --nocapture --test-threads=1`。ログ [`metal/f64_autograd_metal_matches_cpu.log`](./metal/f64_autograd_metal_matches_cpu.log)）。#3169（集計 [`../metal-autodiff-parity-3169/`](../metal-autodiff-parity-3169/README.md)）。GPU 経路かホストフォールバックかは確かめていない（MSL が `double` 非対応のため Metal は常にホスト計算） | 0 |
