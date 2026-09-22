# イシュー #2067 実測記録先（ConvTranspose2d 実機 parity）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10・
Apple Silicon（Metal）実機への到達手段（`docs/real-hardware-
verification-env.local.md`・`CUDA_NODE` 環境変数・`~/.ssh/config` の
いずれも確認できず）がないため、**実測値は一切含まれていない**。本
ディレクトリは実行コマンド・保存すべきログ一覧のみを提供し、実測は
実機を持つセッションへ申し送る（`docs/perf/logs/cuda-conv2d-1766/
README.md` と同型の運用）。

## 目的

イシュー #2067「`Var::conv_transpose2d`・`Op::ConvTranspose2d`・
`nn::ConvTranspose2d`（転置畳み込み）」の実機正しさ検証記録先。以下の
`#[ignore]` テストを CUDA（DGX Spark GB10）・Metal（Apple Silicon）
実機で実行して green を確認することが目的。CPU 側の bit 一致・
数値微分突合は本 PR の CI 内で検証済み（`crates/autodiff/tests/
conv_transpose2d.rs`・`nn_conv.rs`・`crates/facade/tests/
conv_transpose2d_backend_parity.rs` の属性なしテスト。CPU 上での
「転置畳み込みは通常畳み込みの随伴」構造の bit 一致も併せて固定済み）。

## 実行コマンド

```bash
# facade の conv_transpose2d forward／backward REQ-2 複合判定
# （d_input／d_weight／d_bias。#[ignore] は CUDA／Metal 各 3 テスト）
cargo test -p fandhe-ai --release --test conv_transpose2d_backend_parity -- --ignored --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）・Metal 実機
（`cfg(target_os = "macos")`）での相当コマンドにも上記が自動的に
含まれる。

## 保存すべきログ

- `conv_transpose2d_backend_parity-ignored.log`: 上記コマンドの生出力
  （`fold_bits=` 行を含む。run-to-run 決定性チェック用）
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **conv_transpose2d forward／backward REQ-2 parity**: CPU 参照実装
  （`fandhe_ai::tape()`）との複合判定（相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満）が全 fail 0 件であること（GEMM 段由来の差の
  みが判定対象。im2col／col2im 自体は算術を含まないコピー・`f64`
  相当の逐次加算のいずれも bit 完全一致契約のため差の発生源にならない
  ——`docs/conv-ops-design.md` §7・§15「CPU bit 一致の実測根拠」節）
- **run-to-run 決定性**: 上記コマンドを 2 回起動し `fold_bits=` 行が
  2 起動間で bit 同一であること
- 判定規則・tolerance は本記録の追記時点で変更しない（変更する場合は
  ユーザー承認が必要。`.claude/rules/coding-rust.md`）

## 記入欄（実機実測後に追記）

未実測（本 PR 時点）。実測後は `docs/conv-ops-design.md` §15
「#2067」節へ結果を追記する。
