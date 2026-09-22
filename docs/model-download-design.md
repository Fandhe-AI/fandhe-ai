# リモートモデル取得（URL ダウンロード）の設計記録（#2088）

イシュー #2088「feat(facade): リモートモデル取得（URL ダウンロード）の実装（依存追加の承認後）」に対応する。親: #2082（`docs/model-distribution-design.md`）。兄弟: #2087（`crates/facade/src/model.rs`・`ModelRegistry`・`ModelError`）。

本ドキュメントは**コード変更を伴わない設計記録**を成果物とする。`Cargo.toml`／`Cargo.lock`／`deny.toml`／`docs/license-matrix.md`／`docs/spec/`（正本 submodule）・`crates/**` はいずれも変更しない。

基準コミット: 本 PR ブランチ作成時点の `origin/main`（`c498389f0e936c147ce31aa75e21d9808c751db0`・2026-09-23）。

**結論**: 本イシューの受け入れ条件は「依存追加のユーザー承認**取得時**」と「**未取得時**」の 2 系統に分かれる。自動運転モード（ユーザーへの質問・承認待ち不可）では HTTP クライアント依存の追加承認・facade 公開面拡張の承認を得られないため、本ドキュメントは「承認未取得時」の系統として、承認判断に必要な材料（候補クレート・ライセンス実測・技術仕様案・承認後の実装手順・セキュリティ設計）を 1 箇所に確定させる。実装（`ModelRegistry::download` 本体・`api_surface.rs` 到達性テスト・進捗ログ機構）は承認後の別イシューへ引き継ぐ。

## 1. 背景

対応する他フレームワーク機能: `huggingface_hub.hf_hub_download`／`torch.hub.load_state_dict_from_url` 相当。親 #2082 は学習済み重みの配布方式（ローカルレジストリ・リモート取得・HF hub 連携）の設計記録を、兄弟 #2087 は `~/.fandhe-ai/models/<name>/<version>/model.safetensors` レイアウトのローカルレジストリ実装を担う。本 #2088 はその「リモート取得」部分で、HTTP(S) URL・HuggingFace hub から safetensors をダウンロードしてローカルレジストリへキャッシュする機構が対象である。

「承認未取得時」系統を採用した理由: 本セッションは自動運転モード（ユーザーへの質問・承認待ちが不可）で起動されており、依存追加は `.claude/rules/deps-policy.md`・CLAUDE.md Conventions により必ずユーザー承認を要する事項のため、この場では確定できない。

## 2. 現状のコード事実

| 事実 | 出典 |
|---|---|
| 本体 workspace の直接依存は許容 8 区分（CUDA／Metal／相互運用〈safetensors・prost〉／シリアライズ／CPU 並列／数値型／ベンチ／ベンチ比較対象）に限られ、HTTP クライアント・TLS スタックはいずれの区分にも属さない | `.claude/rules/deps-policy.md` 表 |
| `deny.toml` `[licenses].allow` は `MIT`・`Apache-2.0`・`Apache-2.0 WITH LLVM-exception`・`ISC`・`Zlib`・`Unicode-3.0`・`Unlicense`・`BSD-2-Clause` の 8 種。`[sources]` は `allow-registry = ["https://github.com/rust-lang/crates.io-index"]` で crates.io 限定（`unknown-registry`／`unknown-git` とも `deny`） | `deny.toml:43-52`,`deny.toml:55-60` |
| facade の safetensors 公開面（`load_safetensors_f32`／`load_safetensors_f32_from_bytes`／`require_keys`／`save_safetensors_f32`）は `onnx-interop::st_load`／`st_save` の純再エクスポートとして確定済み。REQ-7 契約（暗黙アダプタなし・無言 skip 禁止・F32 限定・決定的出力・一時ファイル + rename）を持つ | `crates/facade/src/interop/safetensors.rs:1-45`・`docs/facade-safetensors-exposure-decision.md` §11 |
| `crates/facade/tests/api_surface.rs` は facade の公開面・onnx-interop への依存形状（`path = "../onnx-interop"` 承認済み形状のみ）を機械的に固定する。新規 public 型・関数の追加は同テストへの到達性テスト追加とユーザー承認が前提 | `crates/facade/tests/api_surface.rs:974-1304` |
| `std` のみで実現できる範囲: `std::net::TcpStream` による平文 HTTP は可能だが、HF hub・実用上のモデル配布元は HTTPS 前提のため TLS なしでは要件を満たせない（依存追加が不可避である根拠） | `std::net` API 仕様（外部一般知識） |
| 親 #2082（`docs/model-distribution-design.md`）・兄弟 #2087（`crates/facade/src/model.rs`）は基準コミット時点で未マージ・PR 未作成 | 本 PR 作成時点の `find`／`git ls-remote`／`gh pr list` 実測（2026-09-23。§4） |

## 3. HTTP クライアント候補の比較

**ライセンスは推移的依存を含め実測が前提**（`docs/license-matrix.md` §1「feature 除外による回避を推定で記述しない」・旧 issue #2 の教訓）。本 PR は依存を一切追加しないため `cargo tree` 実測は実施していない（実測手順は §8 の承認後手順 1 で行う）。以下は候補の一次情報（公開されているライセンスメタデータ・アーキテクチャ）に基づく整理であり、推移的依存の allow 集合適合は**未実測**として扱う。

| 候補 | 同期／非同期 | TLS 選択肢 | 直接依存ライセンス | 推移的依存の allow 集合適合 | 備考 |
|---|---|---|---|---|---|
| `ureq` | 同期 | `rustls` feature | MIT OR Apache-2.0 | 未実測 | 小規模・依存ツリーが薄いとされる |
| `minreq` | 同期 | `https-rustls` feature | Apache-2.0 | 未実測 | 最小構成志向 |
| `attohttpc` | 同期 | `tls-rustls` feature | MIT OR Apache-2.0 | 未実測 | 同期専用設計 |
| `reqwest` | 非同期既定（`blocking` feature でも `tokio` 内包） | `rustls-tls`／`native-tls` | MIT OR Apache-2.0 | 未実測 | `tokio`／`hyper` ツリーを引き込むため依存ツリー肥大の懸念 |
| `std` のみ（TLS なし） | — | なし | — | — | 却下: HTTPS 不可 |
| 外部プロセス委譲（`curl`／`wget` を `std::process::Command` で起動） | — | — | — | — | 却下候補: 環境依存・OWASP A03（引数経由のインジェクション面）・エラー処理の不透明さ・ライブラリ利用者への暗黙の実行時要件 |

とくに実測が必要な点（承認後の実装手順 §8-1 で行う）:

- `rustls` 系スタックの証明書ストアクレート（`webpki-roots` と `rustls-native-certs` のどちらを選ぶ候補が使うかで、MPL-2.0 系ライセンスが allow 集合外になりうる）
- `ring`（`rustls` の暗号実装）のライセンス式
- `native-tls` 選択時の `openssl-sys` システムライブラリ依存（Linux で OpenSSL 実体を要求し、本リポの「環境非依存の開発コンテナ」方針〈README〉と相性を要検証）

推奨候補（**確定はユーザー承認事項。§7-1**）: facade は同期 API のみを公開する設計（`load_safetensors_f32` 等）であり、本リポの設計方針（feature フラグなし・cfg ベース・非同期ランタイム不使用）とも整合するため、同期専用の `ureq`／`minreq`／`attohttpc` のいずれかを推奨する。`reqwest` は非同期ランタイム（`tokio`）を推移的に引き込み設計方針と相性が悪いため非推奨とする。3 候補間の最終選定は §8-1 のライセンス実測結果で行う。

## 4. 依存の配置案（列挙のみ。確定は親 #2082 とユーザー承認）

| 案 | 内容 | 影響 |
|---|---|---|
| 案 A | `facade` の無条件依存 | crates.io 公開 `fandhe-ai` の依存ツリーが増え、全利用者に TLS スタックが載る |
| 案 B | cargo feature による opt-in | 本リポは「feature フラグなしの cfg ベース」方針（REQ-2・`.claude/rules/coding-rust.md`）で optional 依存の前例がなく、採否は方針判断を要する |
| 案 C | 非公開の別クレート（例: `model-hub`）へ切り出し | 公開クレート `facade` から非公開クレートへ通常依存できない構造問題（`docs/facade-onnx-import-exposure-decision.md` §3.1・`api_surface.rs` が監視する形状と同型）を再生産する。公開クレート 8 個目の新設とセットでないと成立しない |

各案の承認コスト・影響範囲は上記のとおりで、推奨は案 A（`onnx-interop` と同様、既存クレート構成に収まり構造問題を再生産しない）とするが、確定は親 #2082 の配布方式決定とユーザー承認による。

## 5. 技術仕様案（承認後にそのまま実装できる粒度）

- **公開 API**: `ModelRegistry::download(&self, url: &str, name: &str, version: &str) -> Result<(), ModelError>`（イシュー指定シグネチャ）。`&self` か関連関数かは兄弟 #2087 が確定する `ModelRegistry` の形状に追従する。
- **`ModelError` 拡張方針**: #2087 が定義する想定の `#[non_exhaustive]` enum へ追加バリアント（`Http { status: u16, url: String }`・`Network(io::Error 相当)`・`InvalidUrl`・`InvalidName`・`SizeLimitExceeded`・`StorageFull`・`Permission`・`CacheMetadata`・`Safetensors(LoadError)` 等）を追加する方針とする。#2087 が先にマージされた場合はその実際の定義に従う。
- **HF hub URL 変換**（要確認。§9 参照）: `https://huggingface.co/<owner>/<repo>` または `<owner>/<repo>` 形式 → `https://huggingface.co/<owner>/<repo>/resolve/<revision>/<file>`（既定 `revision=main`・`file=model.safetensors`）。この URL 形式・LFS ファイルの CDN リダイレクト・`ETag`／`X-Linked-Etag` ヘッダの挙動は外部仕様であり本 PR では未確認のまま「要確認」として記録する。
- **キャッシュ有効判定（etag）**: `<name>/<version>/download.json`（`serde_json`。許容区分内）に `url`・`etag`・`content_length`・`sha256`・`fetched_at` を保存する。再取得時は `If-None-Match` 条件付き GET（304 なら再利用）または HEAD の `ETag` 比較を行う。etag 非提供サーバは常に再取得する（安全側）。
- **書き込み**: 一時ファイル（同一ディレクトリ内）へストリーミング書き込み → 検証成功後に `rename` で `model.safetensors` へ公開する。REQ-7 の `st_save`（一時ファイル + rename）と同型の原子性契約とする。
- **検証方式の決定点**（doc に両案を記録し推奨のみ書く。確定は承認後の実装イシューで行う）:
  - (i) 「F32 ロード可能性」: 既存 `load_safetensors_f32_from_bytes` をそのまま流用（新規コードなしだが、REQ-7 の F32 限定契約により bf16／f16 の HF モデルはキャッシュ不可・全量をメモリへ二重に載せる）
  - (ii) 「フォーマット妥当性」: ヘッダ長（先頭 8 byte）＋ヘッダ JSON パース＋オフセット整合のみ検証（dtype 非依存・ストリーミング可）。facade は `safetensors` クレートへ直接依存できないため、`onnx-interop::st_load` 側へ小さなヘッダ検証関数を追加する必要がある
  - 推奨: (ii)（用途に dtype 制約を持ち込まない）。採否は承認後の実装イシューで確定する
- **エラー処理**: HTTP 非 2xx（3xx はリダイレクト上限付きで追従）・接続／タイムアウト・`io::ErrorKind::StorageFull`／`PermissionDenied`・`Content-Length` と実受信長の不一致・サイズ上限超過を型付きエラーで返し、本番経路で `unwrap`／`expect` を使わない（`.claude/rules/coding-rust.md`）。
- **進捗ログ**: `log`／`tracing` も許容区分外のため新規依存を作らない。呼び出し側が渡す `&mut dyn FnMut(DownloadProgress)` コールバック、または `Option<&mut dyn std::io::Write>` シンクに受信バイト数／総バイト数を書く設計とし、既定は無出力とする。
- **スコープ**: 同期・単一接続。並列ダウンロード・resume・認証トークンはイシュー明記のスコープ外（§10）。
- **タイムアウト・上限の既定値**: 接続／読み取りタイムアウト・最大サイズ・リダイレクト回数を承認後の実装イシューで提案・確定する（本 doc では確定しない）。

## 6. セキュリティ設計（OWASP Top 10。`.claude/rules/security.md`）

承認後の実装が満たすべき受け入れ条件として、以下をチェックリスト形式で記録する。

- [ ] **A01／A03（パストラバーサル・インジェクション）**: `name`・`version` はキャッシュディレクトリ配下のパス要素になるため、空文字・`.`／`..`・パス区切り（`/`・`\`）・絶対パス・NUL・制御文字を fail-closed で拒否する allowlist 検証（英数字・`-`・`_`・`.` の限定集合、先頭 `.` 禁止）を行う
- [ ] **A01／A03（キャッシュ書き込み時のシンボリックリンク脱出防御）**: 上記の文字列検証だけでは、`<name>`／`<version>` ディレクトリ自体が既にキャッシュルート外を指すシンボリックリンクである場合（キャッシュディレクトリは他プロセス・他ユーザーが書き込み可能な共有パスになりうる）、検証済みの一時ファイル・`model.safetensors`・`download.json` をキャッシュルート外の任意ディレクトリへ作成できてしまう。**文字列パスの再解決を前提にした「検査してから rename」方式は、検査（stat／canonicalize）と rename の間に `<name>`／`<version>` またはその親要素をシンボリックリンクへ差し替えられる TOCTOU を構造的に防げない**（最終 rename の直前に宛先パスを再検証しても、その再検証自体が新たな文字列パス解決であり競合窓口になる）。承認後の実装は次のディレクトリハンドル基点の契約とし、書き込み確定まで文字列パスを再解決しない: (a) キャッシュルートから対象パスまでの各パス要素（`<name>`・`<version>` を含む中間ディレクトリ）を、ルートから 1 段ずつ `openat`／`openat2`（`RESOLVE_NO_SYMLINKS | RESOLVE_BENEATH` 相当。Linux では `openat2` を用い、対応しない環境では `O_NOFOLLOW` を各段で明示指定した `openat` で代替する）でオープンし、以降の一時ファイル作成・`model.safetensors`・`download.json` の作成・書き込み・`renameat`／`renameat2` による公開はすべてこの検証済みディレクトリファイルディスクリプタ（dirfd）相対の操作のみで完結させる。生成した dirfd は次のパス要素をオープンするまで保持し、途中で文字列パスへ戻して再オープンしない。(b) 一時ファイルは検証済み dirfd に対して `openat(..., O_CREAT | O_EXCL | O_NOFOLLOW, ...)` で作成し、最終公開は同じ dirfd を親として `renameat`（Linux では `renameat2` に `RENAME_NOREPLACE` を付与できる場合はそれを用いる）で行う。rename の直前に文字列パスを再解決して宛先の型を再確認する手順は行わない（dirfd 相対操作である時点でパス再解決自体が発生しないため TOCTOU 窓口が存在しない）。(c) `openat2`／dirfd 相対 `renameat` 系 API を安全に提供できない OS・実行環境（該当 API 非対応のプラットフォーム）では、その環境向けの `download` 実装を fail-closed で非対応（明示エラーを返し書き込みを一切行わない）とし、文字列パス再解決＋事後検査による近似実装で代替しない。(d) 回帰テストとして、`<name>` または `<version>` に該当するディレクトリを事前にキャッシュルート外を指すシンボリックリンクへ差し替えたうえで `download` を呼び出し、キャッシュルート外への書き込みが発生せずエラーになることに加え、検査直後・rename 直前のタイミングでシンボリックリンクへ差し替える競合を模したケースでも同様にキャッシュルート外への書き込みが発生しないことを確認するテストを実装イシューの受け入れ基準に含める
- [ ] URL は `https` スキームのみ許可する（`http`・`file`・`ftp` は拒否）。リダイレクト先にも同じ検証を通す
- [ ] **A05（設定不備）／A02**: TLS 証明書検証を無効化するオプションは設けない
- [ ] **A08（整合性）**: `Content-Length` 上限・実受信長一致を検証する。任意の `sha256` ピンを指定できるようにし、指定時は不一致で拒否して一時ファイルを削除する。safetensors ヘッダ検証後にのみ `rename` する（検証前のファイルをキャッシュへ置かない）
- [ ] **A06（脆弱コンポーネント）**: 採用候補は `=x.y.z` 完全固定・`cargo deny check advisories bans licenses sources` 通過を採用条件に含める
- [ ] **A09（ログ）**: 進捗コールバックに URL のクエリ文字列・認証情報を流さない。URL に埋め込まれた資格情報（`user:pass@host` 形式）は拒否する
- [ ] **A10（SSRF）**: ライブラリ利用者が渡す URL をそのまま接続する性質上、内部ネットワーク宛先の判定・遮断はライブラリの責務としない（責務は呼び出し側）。`https` 限定と資格情報付き URL の拒否のみを担保範囲とすることを明記する
- [ ] 外部プロセス委譲案（`curl`／`wget` 起動）は A03 の観点で不採用とする（§3 と同一理由）

## 7. 承認事項（本イシュー時点ではいずれも未取得）

1. HTTP クライアント（＋ TLS スタック）を許容依存へ新規区分として追加すること。配置案（§4）の選択を含む
2. facade 公開面の拡張: `ModelRegistry::download`（および進捗コールバック型）の追加と `api_surface.rs` 到達性テストの追加
3. `docs/license-matrix.md` への行追加（依存追加とセット。承認前は行を増やさない）
4. 承認後の実装イシューの起票（`.claude/rules/out-of-scope-tracking.md` により起票自体もユーザー承認が必要なため本 PR では起票しない。§9 に起票草案を記録する）

## 8. 承認後の実装手順（順序付き）

1. `Cargo.toml` `[workspace.dependencies]` へ候補を `=x.y.z` 固定で追加（feature は §3 の実測構成どおり）・`Cargo.lock` 更新・`docs/license-matrix.md` 行追加（`cargo tree` 実測付き）・`cargo deny check` 通過確認・`scripts/check-forbidden-deps.sh` 通過確認
2. `crates/facade/Cargo.toml` へ結線（§4 の配置案に従う）
3. `crates/facade/src/model.rs`（#2087 成果物）へ `download` と `ModelError` 追加バリアントを実装。`name`／`version` 検証は #2087 の検証関数を再利用する
4. HF hub URL 変換・`download.json` メタデータ・条件付き GET・一時ファイル + rename・サイズ上限・進捗コールバックを実装する
5. `api_surface.rs` に到達性テストと「facade が HTTP クレートの型を公開面に漏らさない」テストを追加する
6. テスト: ローカル `std::net::TcpListener` による最小 HTTP モック（200／304／404／リダイレクト／`Content-Length` 不一致／サイズ超過）で CI 実行可能にする。実ネットワークを使うテスト・TLS 経路の実接続テストは `#[ignore]` 分離（`.claude/rules/coding-rust.md`）
7. docs: 本ドキュメントへ実装記録節を追記・`docs/README.md` 注釈更新・`docs/compat-feature-gap.md`／`docs/compat-api-scope.md` の該当行があれば更新する

## 9. 引き継ぎ（起票草案。本イシューでは起票しない）

- 「feat(facade): `ModelRegistry::download` の実装（HTTP クライアント依存承認後）」— 前提: §7 の 1〜3。内容: §8
- 必要に応じ「chore(deps): HTTP クライアント依存の追加とライセンス実測」を分離
- HF hub の `resolve` URL 形式・`ETag`／`X-Linked-Etag`・CDN リダイレクトの外部仕様確認（§5「要確認」印）

## 10. スコープ外

- 並列ダウンロード・resume／partial ダウンロード（イシュー明記）
- HF private repo 認証トークン（イシュー明記）
- ミラー／プロキシ設定
- レジストリ manifest の version 記法（#2087 側スコープ）
- ONNX モデルのダウンロード（safetensors 限定）

## 11. 前方参照の扱い

親 #2082（`docs/model-distribution-design.md`）・兄弟 #2087（`crates/facade/src/model.rs`・`ModelRegistry`・`ModelError`）は基準コミット時点で未マージ・PR 未作成（`git ls-remote --heads origin` に `2082`／`2087` を含むブランチなし・`gh pr list --search "2082 OR 2087"` が空・`crates/facade/src/model.rs` はリポジトリ内に存在しないことを 2026-09-23 に実測確認済み）。本ドキュメントは両者の内部構造（メソッドシグネチャ・enum バリアント）を確定事項として書かず、「並行イシュー #2082／#2087 の成果物（本 doc 作成時点で未マージ）。マージ状況により §5・§8 の記述を追従させる」ことを前提とする。

## 12. 出典一覧

`.claude/rules/deps-policy.md`・`.claude/rules/security.md`・`.claude/rules/coding-rust.md`・`docs/license-matrix.md`・`deny.toml`・`crates/facade/src/interop/safetensors.rs`・`crates/facade/tests/api_surface.rs`・`docs/facade-safetensors-exposure-decision.md`・`docs/facade-onnx-import-exposure-decision.md`・イシュー #2082／#2087／#2088。
