# 委譲原則（調査・設計フェーズ）

## 目的

main コンテキストの消費を抑えるため、調査・分析は subagent へ委譲し、main は判断と統合に専念する。

## 原則

- 2 ファイル以上の読み込みが必要な調査は main で行わず subagent へ委譲する
- main はサブエージェントの報告（結論＋根拠参照）のみを受け取り、生のファイル内容を抱え込まない
- 委譲時は「知りたいこと・報告してほしい形式」を明示して依頼する

## パスベース切り替え（調査）

| 対象パス・トピック | 委譲先 |
|------------------|--------|
| `crates/` の実装調査・影響範囲（workspace 作成後） | explorer |
| `docs/spec/`（REQ・TASK・ロードマップ・PoC-v2 結果）の調査 | explorer |
| CUDA（cudarc・NVRTC・DGX Spark GB10 / sm_121）の外部仕様 | reference-researcher |
| Metal（objc2 系・simdgroup_matrix・MSL）の外部仕様 | reference-researcher |
| safetensors / ONNX（prost）フォーマット・PyTorch との数値比較仕様 | reference-researcher |
| 許容依存クレートの API・ライセンス調査 | reference-researcher |

## main に残す判断

- アーキテクチャ・クレート分割の意思決定（必要に応じ opus / fable の subagent に設計させ、採否は main で判断）
- 依存の追加・更新の判断（deps-policy.md。ユーザー承認必須）
- ガードレール閾値・ポリシー除外リストの変更判断（ユーザー承認必須）
- ユーザーへの確認・承認フロー
- spec（正本 `docs/spec/`）との整合の最終判断
