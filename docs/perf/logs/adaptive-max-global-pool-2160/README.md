# イシュー #2160 実測記録先（AdaptiveMaxPool2d・GlobalPool 実機 parity）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10・
Apple Silicon（Metal）実機への到達手段がないため、**実測値は一切
含まれていない**。本ディレクトリは実行コマンド・保存すべきログ
一覧のみを提供し、実測は実機を持つセッションへ申し送る
（`docs/perf/logs/spatial-layers-2159/README.md` と同型の運用）。

## 目的

イシュー #2160「`nn::AdaptiveMaxPool2d`・`nn::AdaptiveMaxPool1d`・
`nn::GlobalPool`」の実機バックエンド（CUDA・Metal）間の正しさ検証
記録先。以下の `#[ignore]` テストを CUDA（DGX Spark GB10）・Metal
（Apple Silicon）実機で実行して green を確認することが目的。CUDA・
Metal とも専用カーネル未実装（`BackendOps::adaptive_max_pool2d` は
既定 `Unsupported` → ホストフォールバック）のため CPU と**構造的に
bit 完全一致**する契約であり、REQ-2 複合判定（相対誤差／絶対誤差）
ではなく厳密一致を主張する。CPU 側の bit 一致・数値微分突合は本 PR
の CI 内で検証済み（`crates/autodiff/tests/
nn_adaptive_max_global_pool.rs`・`crates/facade/tests/
adaptive_max_global_pool_backend_parity.rs` の属性なしテスト）。

## 実行コマンド

```bash
# facade の AdaptiveMaxPool2d／GlobalPool forward bit 完全一致判定
# （#[ignore] は CUDA／Metal 各 2 テスト）
cargo test -p fandhe-ai --release --test adaptive_max_global_pool_backend_parity -- --ignored --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）・Metal 実機
（`cfg(target_os = "macos")`）での相当コマンドにも上記が自動的に
含まれる。

## 保存すべきログ

- `adaptive_max_global_pool_backend_parity-ignored.log`: 上記コマンド
  の生出力
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **AdaptiveMaxPool2d／GlobalPool forward 実機一致**: CPU 参照実装
  （`fandhe_ai::tape_for(Device::Cpu)`）との **bit 完全一致**（ホスト
  フォールバック経由の構造的一致契約。REQ-2 複合判定の tolerance は
  適用対象外）
- 判定規則は本記録の追記時点で変更しない（変更する場合はユーザー
  承認が必要。`.claude/rules/coding-rust.md`）

## 記入欄（実機実測後に追記）

未実測（本 PR 時点）。実測後は `docs/autodiff-adaptive-max-global-
pool-decision.md` §7 節へ結果を追記する。
