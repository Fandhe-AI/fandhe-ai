# crates.io 公開クレート名: 空き確認結果と最終名の提案（#878）

イシュー #878「crates.io 名前空き確認と最終クレート名のユーザー承認」に対応する。
親ツリー #864（GitHub Pages / crates.io 公開トラッキング）→ Phase 2 親 #866 →
#877（命名確定と rename）の sub-issue。公開対象は `facade` を起点とする依存連鎖
6 クレート（`facade`・`tensor-core`・`autodiff`・`backend-cpu`・`backend-cuda`・
`backend-metal`）。命名方針（prefix 付与 rename）は #864 でユーザー確認済みだが、
**最終名の確定承認は本イシュー実装時に改めて取る**ことが #864 に明記されている。

## 承認ステータス: 承認済み（#878 は本承認を受け入れ条件として 2026-08-22 に COMPLETED クローズ）

本ドキュメントおよび #878 への issue コメントで提案した最終名 6 件は、#878 の
受け入れ条件（最終名 6 件のユーザー承認）を満たしたうえで同イシューが
COMPLETED としてクローズされたことをもって確定した。#879（rename 実施）は
本承認を受けて着手し、下表の rename を実装済みである（実装内容の詳細は
イシュー #879 の実装コミット・PR を参照）。

| ディレクトリ（不変） | 旧 `package.name` | 新 `package.name`（rename 済み） |
|---|---|---|
| `crates/facade` | `facade` | `fandhe-ai` |
| `crates/tensor-core` | `tensor-core` | `fandhe-ai-tensor-core` |
| `crates/autodiff` | `autodiff` | `fandhe-ai-autodiff` |
| `crates/backend-cpu` | `backend-cpu` | `fandhe-ai-backend-cpu` |
| `crates/backend-cuda` | `backend-cuda` | `fandhe-ai-backend-cuda` |
| `crates/backend-metal` | `backend-metal` | `fandhe-ai-backend-metal` |

## 確認方法

1. **crates.io API**: `GET https://crates.io/api/v1/crates/{name}` を候補名ごとに
   個別リクエスト（`curl -s -o <file> -w '%{http_code}'`、`User-Agent` に本リポジトリの
   URL を明示、リクエスト間 1.2 秒以上の間隔を空けて crates.io crawler policy に配慮）。
   HTTP 404 = 未登録（空き）、200 = 既存クレートあり。
2. **`cargo search`**: `cargo search fandhe-ai --limit 20` で prefix 周辺の既存クレートを
   俯瞰し、紛らわしい近接名の有無を確認する。
3. **アンダースコア正規化衝突確認**: crates.io は新規登録時に `-` と `_` を同一視するため
   （既存クレート名の正規化形と衝突する新規名は登録できない）、facade 相当の
   `fandhe_ai`（アンダースコア形）についても個別に確認する。

## 実測結果（確認日時: 2026-08-22 UTC。本イシュー実装時点の再実測）

| 対象クレート（`Cargo.toml` の現行 `name`） | 候補名 | crates.io API 応答 | 判定 |
|---|---|---|---|
| `facade` | `fandhe-ai` | HTTP 404 | 空き |
| `tensor-core` | `fandhe-ai-tensor-core` | HTTP 404 | 空き |
| `autodiff` | `fandhe-ai-autodiff` | HTTP 404 | 空き |
| `backend-cpu` | `fandhe-ai-backend-cpu` | HTTP 404 | 空き |
| `backend-cuda` | `fandhe-ai-backend-cuda` | HTTP 404 | 空き |
| `backend-metal` | `fandhe-ai-backend-metal` | HTTP 404 | 空き |
| （正規化衝突確認。facade 相当） | `fandhe_ai` | HTTP 404 | 衝突なし |

全 7 件（候補 6 件＋正規化衝突確認 1 件）が HTTP 404 であり、すべて空きであることを
確認した。

`cargo search fandhe-ai --limit 20` では `fandhe-ai` 前方一致の既存クレートはヒットせず、
近接する `fandhe-backend-*`（HTTP サーバフレームワーク。`fandhe-backend-core` 等）・
`fandhe-frontend-*`（フロントエンドレンダリングコア。`fandhe-frontend-core` 等）が
ヒットした。これらは本リポジトリ（`fandhe-ai`）とは別プロジェクトの既存クレートで
あり、名前の紛らわしさ・衝突のいずれも生じない（prefix `fandhe-ai` は `fandhe-backend`・
`fandhe-frontend` と語として明確に分離しており誤認の懸念は低いと判断する）。

crates.io の名前制約（ASCII 英数字・`-`・`_`、64 文字以内）は全候補が満たす
（最長は `fandhe-ai-backend-cuda` / `fandhe-ai-backend-metal` の 22 文字）。

## 最終名 6 件の提案

全候補が空きであったため、#864 でユーザー確認済みの例示スキームどおり以下を提案する。

| `Cargo.toml` 現行 `name` | 提案する公開クレート名（`package.name`） |
|---|---|
| `facade` | `fandhe-ai` |
| `tensor-core` | `fandhe-ai-tensor-core` |
| `autodiff` | `fandhe-ai-autodiff` |
| `backend-cpu` | `fandhe-ai-backend-cpu` |
| `backend-cuda` | `fandhe-ai-backend-cuda` |
| `backend-metal` | `fandhe-ai-backend-metal` |

代替案の検討は不要と判断する（全候補が空きのため、#864 の例示スキームをそのまま
採用できる）。

## #879（rename）への引き継ぎ条件

- 上表の最終名 6 件について #878 でユーザー承認を得ること
- 承認後、#879 側で `Cargo.toml` の各クレートの `package.name`・ワークスペース内
  相互参照（`path` 依存の `package` 指定等）・`docs/`／CI 記述中のクレート名言及箇所を
  更新する（本イシューの範囲外）
- 承認前に crates.io 上での名前の事前確保（空 publish 等の不可逆操作）は行わない

## スコープ外（本イシューで行わないこと）

- クレート `name` の変更・参照更新（#879 の範囲）
- publish フラグ・メタデータ整備（#880）、依存 version 付与（#881）
- crates.io 上での名前の事前確保（空 publish）。不可逆かつ承認前のため実施しない

**追記（PR #891）**: 依存 version 付与のうち「公開 6 クレート間 path 依存への
`version = "=0.3.0"`（workspace 公開バージョンと完全一致）併記」のみは、
PR #891 の codex-review P1 指摘（rename 後の path 依存が crates.io 公開要件を
欠くとの機械指摘）を受けて #879 側で前倒しして対応済み。#881 に残るスコープは
依存グラフの公開順序（トポロジカル順）の docs 記録・版数運用（`workspace.
package.version` 一括更新）方針の記録のみ。

## 出典

- crates.io API: `GET https://crates.io/api/v1/crates/{name}`（2026-08-22 UTC 実測。
  全 7 件 HTTP 404）
- `cargo search fandhe-ai --limit 20`（2026-08-22 実行。前方一致なし、近接クレートは
  別プロジェクト所属を確認）
- イシュー #878・親 #877・#866・ルート #864
- `Cargo.toml`（workspace members: `crates/facade`・`crates/tensor-core`・
  `crates/autodiff`・`crates/backend-cpu`・`crates/backend-cuda`・`crates/backend-metal`）
- `.claude/rules/security.md`「自己修復ループ固有のガードレール」（命名確定という
  不可逆性の高い判断への準用）
