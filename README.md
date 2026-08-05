# rust-ai-library

Rust 製 AI/ML ライブラリの実装リポジトリです。Burn 依存を排した完全自作コア（v2 方針）で実装します。

## 位置づけ

- **仕様・要件定義**: [rust-ai-library-spec](https://github.com/Fandhe-AI/rust-ai-library-spec)（`docs/spec` に submodule 参照）
- **旧実装（v1・Burn ベース）**: [rust-ai-library-v1](https://github.com/Fandhe-AI/rust-ai-library-v1)（アーカイブ済み。資産の引き継ぎ記録は [v1-assets-inventory.md](https://github.com/Fandhe-AI/rust-ai-library-spec/blob/main/v1-assets-inventory.md) を参照）
- **立ち上げ手順**: [v2-repo-migration.md](https://github.com/Fandhe-AI/rust-ai-library-spec/blob/main/v2-repo-migration.md)

## ステータス

M0（リポ基盤: workspace 骨格・依存禁止 CI 検査・ライセンス可否表）は未着手です。タスク定義は spec リポの [`05-tasks.md`](https://github.com/Fandhe-AI/rust-ai-library-spec/blob/main/05-tasks.md)（TASK-1.1〜1.3）、マイルストーンは [`06-roadmap.md`](https://github.com/Fandhe-AI/rust-ai-library-spec/blob/main/06-roadmap.md)（M0〜M5・全 51 タスク）を参照してください。

## 実装方針（要点）

- 想定クレート 9 個: `tensor-core`・`autodiff`・`backend-cpu`・`backend-cuda`・`backend-metal`・`onnx-interop`・`guardrail`・`self-repair`・`bench-harness`
- 許容依存 8 区分（`cudarc`／`objc2` 系／`safetensors`／`prost`／`serde` 系／`rayon`／`half`／`criterion`）を `=x.y.z` 完全固定で管理
- 依存禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・`ndarray`）を CI で機械検査（TASK-1.2）
