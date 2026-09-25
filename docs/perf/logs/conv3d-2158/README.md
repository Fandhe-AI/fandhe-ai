# イシュー #2158 実測記録先（Conv3d 実機 parity）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10・
Apple Silicon（Metal）実機への到達手段（`docs/real-hardware-
verification-env.local.md`・`CUDA_NODE` 環境変数・`~/.ssh/config` の
いずれも確認できず）がないため、**実測値は一切含まれていない**。本
ディレクトリは実行コマンド・保存すべきログ一覧のみを提供し、実測は
実機を持つセッションへ申し送る（`docs/perf/logs/conv-transpose2d-2067/
README.md` と同型の運用）。

## 目的

イシュー #2158「`conv3d_ops::conv3d`・`Op::Conv3d`・`nn::Conv3d`
（Conv3d。im2col の空間 3 軸一般化）」の実機正しさ検証記録先。以下の
`#[ignore]` テストを CUDA（DGX Spark GB10）・Metal（Apple Silicon）
実機で実行して green を確認することが目的。CPU 側の bit 一致・
数値微分突合は本 PR の CI 内で検証済み（`crates/autodiff/tests/
conv3d.rs`・`nn_conv3d.rs`・`crates/facade/tests/
conv3d_backend_parity.rs` の属性なしテスト。CPU 上での kD=1 の Conv3d
と reshape 経由の Conv2d の bit 一致も併せて固定済み）。

## 実行コマンド

```bash
# facade の conv3d forward REQ-2 複合判定
# （#[ignore] は CUDA 1 テスト・Metal 1 テスト〈macOS 限定〉）
cargo test -p fandhe-ai --release --test conv3d_backend_parity -- --ignored --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）・Metal 実機
（`cfg(target_os = "macos")`）での相当コマンドにも上記が自動的に
含まれる。

## 保存すべきログ

- `conv3d_backend_parity-ignored.log`: 上記コマンドの生出力
  （`fold_bits=` 行を含む。run-to-run 決定性チェック用）
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **conv3d forward REQ-2 parity**: CPU 参照実装（`fandhe_ai::tape()`）
  との複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）が全
  fail 0 件であること。CUDA／Metal とも `conv3d`／`im2col3d`／
  `col2im3d` を override しない（既定 `Unsupported`）ため、GPU 経路は
  `im2col3d_with_fallback`（ホスト実装へフォールバック。bit 完全
  一致契約）→ `ops.gemm_batched`（GPU GEMM）の合成になる。差の発生源
  は GEMM 段のみ（`docs/conv-ops-design.md` §16「16.3」参照）
- **run-to-run 決定性**: 上記コマンドを 2 回起動し `fold_bits=` 行が
  2 起動間で bit 同一であること
- 判定規則・tolerance は本記録の追記時点で変更しない（変更する場合は
  ユーザー承認が必要。`.claude/rules/coding-rust.md`）

## 記入欄（実機実測後に追記）

未実測（本 PR 時点）。実測後は `docs/conv-ops-design.md` §16「16.6」
節へ結果を追記する。
