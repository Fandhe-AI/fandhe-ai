# イシュー #2162 実測記録先（PixelShuffle・PixelUnshuffle 実機 parity）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10・
Apple Silicon（Metal）実機への到達手段がないため、**実測値は一切
含まれていない**。本ディレクトリは実行コマンド・保存すべきログ
一覧のみを提供し、実測は実機を持つセッションへ申し送る
（`docs/perf/logs/spatial-layers-2159/README.md` と同型の運用）。

## 目的

イシュー #2162「`nn::PixelShuffle`・`nn::PixelUnshuffle`」の実機
バックエンド（CUDA・Metal）間 parity 検証記録先。以下の `#[ignore]`
テストを CUDA（DGX Spark GB10）・Metal（Apple Silicon）実機で実行
して green を確認することが目的。両層は算術を含まない純粋なコピー・
view の合成のため 3 バックエンド間で構造的に bit 完全一致する見込み
だが、実機のメモリレイアウト・カーネル実装差を実測で確認する
（`docs/autodiff-pixel-shuffle-decision.md` §4 参照）。CPU 側の bit
一致は本 PR の CI 内で検証済み（`crates/autodiff/tests/
nn_pixel_shuffle.rs`・`crates/facade/tests/
pixel_shuffle_backend_parity.rs` の属性なしテスト）。

## 実行コマンド

```bash
# facade の PixelShuffle／PixelUnshuffle forward parity
# （#[ignore] は CUDA／Metal 各 2 テスト）
cargo test -p fandhe-ai --release --test pixel_shuffle_backend_parity -- --ignored --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）・Metal 実機
（`cfg(target_os = "macos")`）での相当コマンドにも上記が自動的に
含まれる。

## 保存すべきログ

- `pixel_shuffle_backend_parity-ignored.log`: 上記コマンドの生出力
  （`fold_bits=` 行を含む。run-to-run 決定性チェック用）
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **PixelShuffle／PixelUnshuffle forward parity**: CPU 参照実装
  （`fandhe_ai::tape_for(Device::Cpu)`）との複合判定（相対誤差
  1e-3 未満 または絶対誤差 1e-5 未満）が全 fail 0 件であること
  （両層とも純粋なコピー・view のため実際には bit 完全一致になる
  見込み）
- **run-to-run 決定性**: 上記コマンドを 2 回起動し `fold_bits=` 行が
  2 起動間で bit 同一であること
- 判定規則・tolerance は本記録の追記時点で変更しない（変更する場合は
  ユーザー承認が必要。`.claude/rules/coding-rust.md`）

## 記入欄（実機実測後に追記）

未実測（本 PR 時点）。実測後は `docs/autodiff-pixel-shuffle-
decision.md` §7 節へ結果を追記する。
