# イシュー #2166 実測記録先（L1 損失・CE オプション付き損失 実機 parity）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10・
Apple Silicon（Metal）実機への到達手段（`docs/real-hardware-
verification-env.local.md`・`CUDA_NODE` 環境変数・`~/.ssh/config` の
いずれも確認できず）がないため、**実測値は一切含まれていない**。本
ディレクトリは実行コマンド・保存すべきログ一覧のみを提供し、実測は
実機を持つセッションへ申し送る（`docs/perf/logs/conv3d-2158/
README.md` と同型の運用）。

## 目的

イシュー #2166「L1 損失と CrossEntropy の label_smoothing・
ignore_index・class_weight」の実機正しさ検証記録先。以下の
`#[ignore]` テストを CUDA（DGX Spark GB10）・Metal（Apple Silicon）
実機で実行して green を確認することが目的。CPU 側の bit 一致・
数値微分突合は本 PR の CI 内で検証済み（`crates/autodiff/tests/
loss_ops.rs`・`crates/facade/tests/loss_ops_backend_parity.rs` の
属性なしテスト）。

`Op::L1Loss`・`Op::CrossEntropyLossWithOptions` はいずれもホスト
参照実装（`crate::eval`）のみで forward／backward を計算し
`BackendOps` を経由しない（`crates/autodiff/src/tape.rs::Op::L1Loss`／
`Op::CrossEntropyLossWithOptions` doc 参照）ため、CUDA／Metal 実機でも
**bit 完全一致**が期待される（`Op::CrossEntropyLoss`〈既存〉と同型の
性質）。ただし本 PR の受け入れ判定は REQ-2 の統一複合判定に留め、
bit 一致は実測後の付随的な確認とする。

## 実行コマンド

```bash
# facade の L1／CE-with-options forward REQ-2 複合判定
# （#[ignore] は CUDA 2 テスト・Metal 2 テスト〈macOS 限定〉）
cargo test -p fandhe-ai --release --test loss_ops_backend_parity -- --ignored --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）・Metal 実機
（`cfg(target_os = "macos")`）での相当コマンドにも上記が自動的に
含まれる。

## 保存すべきログ

- `loss_ops_backend_parity-ignored.log`: 上記コマンドの生出力
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **L1／CE-with-options forward REQ-2 parity**: CPU 参照実装
  （`fandhe_ai::tape()`）との複合判定（相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満）が全 fail 0 件であること
- ホスト計算のみのため上記に加え bit 完全一致（`to_bits()` 一致）も
  期待されるが、判定基準としては REQ-2 複合判定を正とする（本節冒頭
  「目的」参照）
- 判定規則・tolerance は本記録の追記時点で変更しない（変更する場合は
  ユーザー承認が必要。`.claude/rules/coding-rust.md`）

## 記入欄（実機実測後に追記）

未実測（本 PR 時点）。実測後は `docs/autodiff-loss-ops-decision.md`
へ結果を追記する。
