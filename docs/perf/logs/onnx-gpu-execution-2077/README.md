# イシュー #2077 実測記録先（ONNX import モデルの GPU 実行 実機 parity）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には Apple Silicon（Metal）
実機への到達手段がなく、CUDA は driver のみ検出可能で NVRTC 等の実行時
コンポーネントが欠けた部分的な環境（`libnvrtc` 不在。`crates/facade/
tests/interop_onnx_gpu_execution_optin.rs` の非 `#[ignore]` テストが
実際にこの環境で `OnnxError::Execution`（fail-closed）を実測している）
のため、**§6 契約 (c)（GPU 実行 ON とホスト実行 OFF の REQ-2 複合判定）
の実機実測値は一切含まれていない**。本ディレクトリは実行コマンド・
保存すべきログ一覧のみを提供し、実測は実機（DGX Spark GB10・Apple
Silicon）を持つセッションへ申し送る（`docs/perf/logs/conv-transpose2d-
2067/README.md` と同型の運用）。

## 目的

イシュー #2077「ONNX import モデルの GPU 実行（`OnnxModel::run` のホスト
CPU 限定解除）」の実機正しさ検証記録先。以下の `#[ignore]` テストを
CUDA（DGX Spark GB10）・Metal（Apple Silicon）実機で実行して green を
確認することが目的。opt-in の既定 OFF・fail-closed・bit 不変契約は
本 PR の CI 内で検証済み（`crates/facade/tests/
interop_onnx_gpu_execution_optin.rs`・`crates/onnx-interop/tests/
onnx_interp_backend_dispatch.rs` の非 `#[ignore]` テスト）。

## 実行コマンド

```bash
# facade の OnnxModel::run GPU 実行 opt-in ON/OFF の REQ-2 複合判定
# （#[ignore] は CUDA 1 テスト・Metal 1 テスト〈macOS cfg 限定〉）
cargo test -p fandhe-ai --release --test interop_onnx_gpu_execution_parity -- --ignored --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）・Metal 実機
（`cfg(target_os = "macos")`）での相当コマンドにも上記が自動的に含まれる。

## 保存すべきログ

- `interop_onnx_gpu_execution_parity-ignored.log`: 上記コマンドの生出力
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **契約 (c)（GPU 実行 ON vs ホスト実行 OFF）**: REQ-2 統一複合判定
  （相対誤差 1e-3 未満 または絶対誤差 1e-5 未満）が全 fail 0 件で
  あること（`fandhe_ai_backend_cpu::parity::assert_parity` による判定。
  `docs/onnx-gpu-execution-decision.md` §6）
- 判定規則・tolerance は本記録の追記時点で変更しない（変更する場合は
  ユーザー承認が必要。`.claude/rules/coding-rust.md`）

## 記入欄（実機実測後に追記）

未実測（本 PR 時点）。実測後は `docs/onnx-gpu-execution-decision.md`
§7「実機実測（申し送り）」節へ結果を追記する。
