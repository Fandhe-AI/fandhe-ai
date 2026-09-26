# イシュー #2168 実測記録先（CTC 損失 実機 parity）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10・
Apple Silicon（Metal）実機への到達手段がないため、**実測値は一切
含まれていない**。本ディレクトリは実行コマンド・保存すべきログ
一覧のみを提供し、実測は実機を持つセッションへ申し送る
（`docs/perf/logs/loss-ops-2167/README.md` と同型の運用）。

## 目的

イシュー #2168「CTC 損失（CPU 参照実装）」の実機正しさ検証記録先。
以下の `#[ignore]` テストを CUDA（DGX Spark GB10）・Metal（Apple
Silicon）実機で実行して green を確認することが目的。CPU 側の bit
一致・数値微分突合・総当たり整列との forward 一致は本 PR の CI 内で
検証済み（`crates/autodiff/tests/loss_ops.rs`・`crates/facade/tests/
loss_ops_backend_parity.rs` の属性なしテスト）。

`Op::CtcLoss` はホスト参照実装（`crate::eval`／`crate::grad`）のみで
forward／backward を計算し `BackendOps` を経由しない
（`crates/autodiff/src/tape.rs` の `Op::CtcLoss` doc 参照）ため、
CUDA／Metal 実機でも **bit 完全一致**が期待される。ただし本 PR の
受け入れ判定は REQ-2 の統一複合判定に留め、bit 一致は実測後の
付随的な確認とする。

## 実行コマンド

```bash
# facade の ctc_loss forward REQ-2 複合判定
# （#[ignore] は CUDA 1 テスト・Metal 1 テスト〈macOS 限定〉）
cargo test -p fandhe-ai --release --test loss_ops_backend_parity -- --ignored ctc --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）・Metal 実機
（`cfg(target_os = "macos")`）での相当コマンドにも上記が自動的に
含まれる（`loss_ops_backend_parity.rs` は #2166／#2167／#2168 の
テストを同一ファイルに持つ）。

## 保存すべきログ

- `loss_ops_backend_parity-ignored-ctc.log`: 上記コマンドの生出力
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **CTC 損失 forward REQ-2 parity**: CPU 参照実装（`fandhe_ai::
  tape()`）との複合判定（相対誤差 1e-3 未満 または絶対誤差 1e-5
  未満）が全 fail 0 件であること
- ホスト計算のみのため上記に加え bit 完全一致（`to_bits()` 一致）も
  期待されるが、判定基準としては REQ-2 複合判定を正とする（本節冒頭
  「目的」参照）
- 判定規則・tolerance は本記録の追記時点で変更しない（変更する場合は
  ユーザー承認が必要。`.claude/rules/coding-rust.md`）

## 記入欄（実機実測後に追記）

未実測（本 PR 時点）。実測後は `docs/autodiff-ctc-design.md` §4 へ
結果を追記する。
