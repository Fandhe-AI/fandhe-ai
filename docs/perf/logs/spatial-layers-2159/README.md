# イシュー #2159 実測記録先（ConvTranspose1d・Upsample 実機 parity）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10・
Apple Silicon（Metal）実機への到達手段がないため、**実測値は一切
含まれていない**。本ディレクトリは実行コマンド・保存すべきログ
一覧のみを提供し、実測は実機を持つセッションへ申し送る
（`docs/perf/logs/conv-transpose2d-2067/README.md` と同型の運用）。

## 目的

イシュー #2159「`nn::ConvTranspose1d`・`nn::Upsample`・
`nn::ZeroPad2d`・`nn::Identity`・`nn::Unflatten`」のうち、実機
バックエンド（CUDA・Metal）間で REQ-2 複合判定を要する 2 層
（`ConvTranspose1d`・`Upsample(Bilinear)`）の実機正しさ検証記録先。
以下の `#[ignore]` テストを CUDA（DGX Spark GB10）・Metal（Apple
Silicon）実機で実行して green を確認することが目的。`ZeroPad2d`・
`Identity`・`Unflatten`・`Upsample(Nearest／NearestExact)` は算術を
含まないコピーまたは view のため 3 バックエンド間で構造的に bit
完全一致し、実機専用テストの対象としない（`docs/autodiff-spatial-
layers-decision.md` §4 参照）。CPU 側の bit 一致・数値微分突合は
本 PR の CI 内で検証済み（`crates/autodiff/tests/nn_spatial_layers.rs`・
`crates/facade/tests/spatial_layers_backend_parity.rs` の属性なし
テスト）。

## 実行コマンド

```bash
# facade の ConvTranspose1d／Upsample(Bilinear) forward REQ-2 複合判定
# （#[ignore] は CUDA／Metal 各 2 テスト）
cargo test -p fandhe-ai --release --test spatial_layers_backend_parity -- --ignored --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）・Metal 実機
（`cfg(target_os = "macos")`）での相当コマンドにも上記が自動的に
含まれる。

## 保存すべきログ

- `spatial_layers_backend_parity-ignored.log`: 上記コマンドの生出力
  （`fold_bits=` 行を含む。run-to-run 決定性チェック用）
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **ConvTranspose1d／Upsample(Bilinear) forward REQ-2 parity**: CPU
  参照実装（`raw_tape_for(Device::Cpu)`／`fandhe_ai::tape_for
  (Device::Cpu)`）との複合判定（相対誤差 1e-3 未満 または絶対誤差
  1e-5 未満）が全 fail 0 件であること
- **run-to-run 決定性**: 上記コマンドを 2 回起動し `fold_bits=` 行が
  2 起動間で bit 同一であること
- 判定規則・tolerance は本記録の追記時点で変更しない（変更する場合は
  ユーザー承認が必要。`.claude/rules/coding-rust.md`）

## 記入欄（実機実測後に追記）

未実測（本 PR 時点）。実測後は `docs/autodiff-spatial-layers-
decision.md` §7 節へ結果を追記する。
