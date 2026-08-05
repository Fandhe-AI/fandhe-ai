# セキュリティ規約

## 秘密情報の混入防止

- API キー・トークン・パスワード・`.env` をコミット・PR・hooks・workflow に含めない
- コミット前に staged 差分のシークレット検査を行う（create-commit スキルの検査に従う）
- `settings.json` の hooks・`lefthook.yml`・CI workflow にシークレットをハードコードしない

## OWASP Top 10 観点（本リポでの重点）

- **A03 インジェクション**: `guardrail` CLI・設定ファイル（TOML 等）のパース時に外部入力を検証する。シェル呼び出しでユーザー入力を直接展開しない。safetensors / ONNX（prost）の外部フォーマットパースは長さ・形状の検証を先に行う
- **A06 脆弱・古いコンポーネント**: 依存クレートは `=x.y.z` 完全固定（deps-policy.md）とし、追加時に `cargo deny check licenses sources` 相当の確認を行う
- **A08 ソフトウェア・データ整合性**: 自己修復ループが取り込む AI 生成変更はガードレール 3 分岐判定を必ず経由する。判定の迂回経路を作らない。CI の actions は SHA 固定とする

## 自己修復ループ固有のガードレール

- ガードレール閾値・ポリシー除外リスト・テスト許容誤差の変更は必ず人間（ユーザー）の承認を経る
- ループ試行ログは改竄検知可能な形式で記録し、取り込み判断の根拠を追跡可能にする

## unsafe

- `unsafe` は FFI 境界（cudarc・objc2 系）等の必要最小限に限る。使用時は理由コメント＋レビュー（security-auditor）必須

## レビュー体制

- 依存追加・ガードレール・CI/hooks 変更を含む PR は security-auditor の監査を並列で実施する
