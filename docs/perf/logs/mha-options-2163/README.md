# イシュー #2163 実測記録先（MultiheadAttention オプション実機 parity）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10・
Apple Silicon（Metal）実機への到達手段がないため、**実測値は一切
含まれていない**。本ディレクトリは実行コマンド・保存すべきログ
一覧のみを提供し、実測は実機を持つセッションへ申し送る
（`docs/perf/logs/spatial-layers-2159/README.md` と同型の運用）。

## 目的

イシュー #2163「`nn::MultiheadAttention` のオプション（`batch_first`・
`kdim`/`vdim`・`key_padding_mask`）」の実機バックエンド（CUDA・Metal）
間 forward parity の検証記録先。`kdim`/`vdim` 非対称の cross-attention
ケースを対象とする（`batch_first=false`・`key_padding_mask` は既存の
`Var::transpose`／`masked_fill` 経由に帰着し、これらの演算自体の実機
parity は既存テスト群で担保済みのため、本ケースを実機専用テストの
主対象とした）。CPU 側の bit 一致・数値微分突合・`batch_first=false`／
`key_padding_mask` の等価変換突合は本 PR の CI 内で検証済み
（`crates/autodiff/src/nn/attention.rs` の単体テスト・
`crates/facade/tests/mha_options_backend_parity.rs` の属性なしテスト）。

## 実行コマンド

```bash
# facade の MultiheadAttention（kdim/vdim 非対称）forward REQ-2 複合判定
# （#[ignore] は CUDA 1 テスト・Metal 1 テスト）
cargo test -p fandhe-ai --release --test mha_options_backend_parity -- --ignored --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）・Metal 実機
（`cfg(target_os = "macos")`）での相当コマンドにも上記が自動的に
含まれる。

## 保存すべきログ

- `mha_options_backend_parity-ignored.log`: 上記コマンドの生出力
  （`fold_bits=` 行を含む。run-to-run 決定性チェック用）
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **MultiheadAttention（kdim/vdim 非対称）forward REQ-2 parity**: CPU
  参照実装（`raw_tape_for(Device::Cpu)`／`fandhe_ai::tape_for
  (Device::Cpu)`）との複合判定（相対誤差 1e-3 未満 または絶対誤差
  1e-5 未満）が全 fail 0 件であること
- **run-to-run 決定性**: 上記コマンドを 2 回起動し `fold_bits=` 行が
  2 起動間で bit 同一であること
- 判定規則・tolerance は本記録の追記時点で変更しない（変更する場合は
  ユーザー承認が必要。`.claude/rules/coding-rust.md`）

## 記入欄（実機実測後に追記）

未実測（本 PR 時点）。実測後は `docs/autodiff-mha-options-decision.md`
へ結果を追記する。
