# リモートモデル取得（URL ダウンロード）の設計記録（#2088）

イシュー #2088「feat(facade): リモートモデル取得（URL ダウンロード）の実装（依存追加の承認後）」に対応する。親: #2082（`docs/model-distribution-design.md`。本 doc 作成後に作成済み）。兄弟: #2087（`crates/facade/src/model.rs`・`ModelRegistry`・`ModelError`。マージ済み）。

本ドキュメントは**コード変更を伴わない設計記録**を成果物とする。`Cargo.toml`／`Cargo.lock`／`deny.toml`／`docs/license-matrix.md`／`docs/spec/`（正本 submodule）・`crates/**` はいずれも変更しない。

基準コミット: 本 PR ブランチ作成時点の `origin/main`（`c498389f0e936c147ce31aa75e21d9808c751db0`・2026-09-23）。

**結論**: 本イシューの受け入れ条件は「依存追加のユーザー承認**取得時**」と「**未取得時**」の 2 系統に分かれる。自動運転モード（ユーザーへの質問・承認待ち不可）では HTTP クライアント依存の追加承認・facade 公開面拡張の承認を得られないため、本ドキュメントは「承認未取得時」の系統として、承認判断に必要な材料（候補クレート・ライセンス実測・技術仕様案・承認後の実装手順・セキュリティ設計）を 1 箇所に確定させる。実装（`ModelRegistry::download` 本体・`api_surface.rs` 到達性テスト・進捗ログ機構）は承認後の別イシューへ引き継ぐ。

**現在の状態（2026-10-07 追記・#2621）**: HTTP／TLS 依存は**承認待ちではなく再評価待ち**である。§14.3 の推奨案（`ureq =3.4.2`・`native-tls-no-default` 単独）は §15 の実測で HTTPS が成立しないと判明したため**取り下げ、承認対象から外した**。現時点の推奨は未定で、§14.5-1 の区分起案と §14.6 の承認チェックリストは現状のままでは承認できない。再評価で決める論点は §15.8 に整理した（新しい推奨は示していない。選び直しはユーザー判断）。

**再実測と推奨（2026-10-16 追記・#3122）**: §15.8 の選択肢を、決定記録にすでに載っている候補だけで、有効化しうる feature 組合せ・ターゲット・環境ごとに再実測した。結果と推奨（**`ureq =3.4.2`・`features = ["native-tls", "vendored"]` を第一候補、`minreq =3.0.0`・`https-native-tls` を次点**。いずれも**推奨止まりで承認は未取得**。最終承認は #3123）は §16 にある。§14.4・§14.5-1・§14.6 は書き換えず、§16.9 に書き直し案を別に置いた。本体の `Cargo.toml`／`Cargo.lock`／`deny.toml` は変更していない。

## 1. 背景

対応する他フレームワーク機能: `torch.hub.load_state_dict_from_url`・`tf.keras.utils.get_file` 相当（汎用 URL ダウンロード＋キャッシュ＋ハッシュ検証）。親 #2082 は学習済み重みの配布方式（ローカルレジストリ・リモート取得・HF hub 連携）の設計記録を、兄弟 #2087 は `~/.fandhe-ai/models/<name>/<version>/model.safetensors` レイアウトのローカルレジストリ実装を担う。本 #2088 はその「リモート取得」部分で、**HTTPS URL から safetensors をダウンロードしてローカルレジストリへキャッシュする汎用機構のみ**が対象である（`http`・`file`・`ftp` 等は §6 のとおり拒否。2026-09-24 追記で表記を「HTTP(S)」から「HTTPS」へ統一。詳細は本節末尾の追記参照）。

**2026-09-24 追記（HF hub 連携の分離。ユーザー決定「他ライブラリと同じにする」）**: PyTorch（`torch.hub` はコア／HF 連携は別パッケージ `huggingface_hub`）・TensorFlow-Keras（`tf.keras.utils.get_file` はコア／`tensorflow_hub`・Keras 3 の `hf://` は別パッケージ）と同型に、HuggingFace hub 固有の処理（リポジトリ ID → URL 変換・`resolve` 形式・`ETag`／`X-Linked-Etag`・CDN リダイレクト等）は facade（本 #2088）のスコープに含めない。別クレートとして #2243（#2244〜#2246 に分解済み）で扱う（詳細は `docs/model-distribution-design.md` §5）。本追記より前の本文（§5・§8・§9・§10 の該当箇所）は HF hub を対象に含む記述だったため、本追記に合わせて汎用ダウンロードのみを対象とする記述へ更新した。§2 の実測事実・§11 の前方参照記録は当時の記録のまま変更していない。

「承認未取得時」系統を採用した理由: 本セッションは自動運転モード（ユーザーへの質問・承認待ちが不可）で起動されており、依存追加は `.claude/rules/deps-policy.md`・CLAUDE.md Conventions により必ずユーザー承認を要する事項のため、この場では確定できない。

## 2. 現状のコード事実

| 事実 | 出典 |
|---|---|
| 本体 workspace の直接依存は許容依存 9 区分のうち第 1〜8 区分（CUDA／Metal／相互運用〈safetensors〉／相互運用〈prost〉／シリアライズ／CPU 並列／数値型／ベンチ〈criterion〉）に限られる（第 9 区分〈ベンチ比較対象〉は `scripts/bench/` 配下の独立 workspace 限定で本体 workspace には入らない）。HTTP クライアント・TLS スタックはいずれの区分にも属さない | `.claude/rules/deps-policy.md` 見出し・表 |
| `deny.toml` `[licenses].allow` は `MIT`・`Apache-2.0`・`Apache-2.0 WITH LLVM-exception`・`ISC`・`Zlib`・`Unicode-3.0`・`Unlicense`・`BSD-2-Clause` の 8 種。`[sources]` は `allow-registry = ["https://github.com/rust-lang/crates.io-index"]` で crates.io 限定（`unknown-registry`／`unknown-git` とも `deny`） | `deny.toml:43-52`,`deny.toml:55-60` |
| facade の safetensors 公開面（`load_safetensors_f32`／`load_safetensors_f32_from_bytes`／`require_keys`／`save_safetensors_f32`）は `onnx-interop::st_load`／`st_save` の純再エクスポートとして確定済み。REQ-7 契約（暗黙アダプタなし・無言 skip 禁止・F32 限定・決定的出力・一時ファイル + rename）を持つ | `crates/facade/src/interop/safetensors.rs:1-45`・`docs/facade-safetensors-exposure-decision.md` §11 |
| `crates/facade/tests/api_surface.rs` は facade の公開面・onnx-interop への依存形状（`path = "../onnx-interop"` 承認済み形状のみ）を機械的に固定する。新規 public 型・関数の追加は同テストへの到達性テスト追加とユーザー承認が前提 | `crates/facade/tests/api_surface.rs:974-1304` |
| `std` のみで実現できる範囲: `std::net::TcpStream` による平文 HTTP は可能だが、HF hub・実用上のモデル配布元は HTTPS 前提のため TLS なしでは要件を満たせない（依存追加が不可避である根拠） | `std::net` API 仕様（外部一般知識） |
| 親 #2082（`docs/model-distribution-design.md`）・兄弟 #2087（`crates/facade/src/model.rs`）は基準コミット時点で未マージ・PR 未作成（いずれも本 doc 作成後に作成・マージ済み。§11 追記参照） | 本 PR 作成時点の `find`／`git ls-remote`／`gh pr list` 実測（2026-09-23。§4） |

## 3. HTTP クライアント候補の比較

**ライセンスは推移的依存を含め実測が前提**（`docs/license-matrix.md` §1「feature 除外による回避を推定で記述しない」・旧 issue #2 の教訓）。本 PR（#2088）は依存を一切追加しないため `cargo tree` 実測は実施していなかった。以下は候補の一次情報に基づく当時の整理であり、**2026-10-03 の #2620 で候補 4 種を feature 組合せ×ターゲット別に実測し、結果を §13 に記録した**（表の「直接依存ライセンス」の 2 行〈`minreq`・`attohttpc`〉は実測で誤りと判明したため是正済み。推奨候補の選定と区分起案は #2621 の担当で、本節は書き換えない）。

| 候補 | 同期／非同期 | TLS 選択肢 | 直接依存ライセンス | 推移的依存の allow 集合適合 | 備考 |
|---|---|---|---|---|---|
| `ureq` | 同期 | `rustls`／`platform-verifier`／`native-tls` feature | MIT OR Apache-2.0 | 実測済み（§13。全 rustls 系組合せが現行 allow 外を含む） | 小規模・依存ツリーが薄いとされる |
| `minreq` | 同期 | `https`（= `https-rustls`）・`https-rustls-probe`・`https-native-tls`・`https-openssl` feature | ISC（2026-10-03 #2620 実測で是正。旧記載 Apache-2.0 は誤り） | 実測済み（§13。rustls 系は allow 外を含み、native-tls／openssl 系のみ allow 適合） | 最小構成志向 |
| `attohttpc` | 同期 | `tls-native`（既定）・`tls-rustls-webpki-roots` 等 | **MPL-2.0**（2026-10-03 #2620 実測で是正。旧記載 MIT OR Apache-2.0 は誤り） | 実測済み（§13。直接ライセンスが allow 外のため feature によらず不適合） | 同期専用設計 |
| `reqwest` | 非同期既定（`blocking` feature でも `tokio` 内包） | `rustls`（既定）／`native-tls` | MIT OR Apache-2.0 | 実測済み（§13。推移依存 90〜108 個で最大。allow 外を含む） | `tokio`／`hyper` ツリーを引き込むため依存ツリー肥大の懸念 |
| `std` のみ（TLS なし） | — | なし | — | — | 却下: HTTPS 不可 |
| 外部プロセス委譲（`curl`／`wget` を `std::process::Command` で起動） | — | — | — | — | 却下候補: 環境依存・OWASP A03（引数経由のインジェクション面）・エラー処理の不透明さ・ライブラリ利用者への暗黙の実行時要件 |

とくに実測が必要な点（承認後の実装手順 §8-1 で行う）:

- `rustls` 系スタックの証明書ストアクレート（`webpki-roots` と `rustls-native-certs` のどちらを選ぶ候補が使うかで、MPL-2.0 系ライセンスが allow 集合外になりうる）
- `ring`（`rustls` の暗号実装）のライセンス式
- `native-tls` 選択時の `openssl-sys` システムライブラリ依存（Linux で OpenSSL 実体を要求し、本リポの「環境非依存の開発コンテナ」方針〈README〉と相性を要検証）

推奨候補（**確定はユーザー承認事項。§7-1**）: facade は同期 API のみを公開する設計（`load_safetensors_f32` 等）であり、本リポの設計方針（feature フラグなし・cfg ベース・非同期ランタイム不使用）とも整合するため、同期専用の `ureq`／`minreq`／`attohttpc` のいずれかを推奨する。`reqwest` は非同期ランタイム（`tokio`）を推移的に引き込み設計方針と相性が悪いため非推奨とする。3 候補間の最終選定は §8-1 のライセンス実測結果で行う。

**追記（2026-10-03・#2621）**: 推奨 crate の選定は §14.3 で行った（未承認の起案）。本節の本文は書き換えない。

**追記（2026-10-07・#2621）**: §14.3 の推奨案は §15 の実測で HTTPS が成立しないと判明したため取り下げた。現時点の推奨は未定である（再評価の論点は §15.8）。

## 4. 依存の配置案（列挙のみ。確定は親 #2082 とユーザー承認）

| 案 | 内容 | 影響 |
|---|---|---|
| 案 A | `facade` の無条件依存 | crates.io 公開 `fandhe-ai` の依存ツリーが増え、全利用者に TLS スタックが載る |
| 案 B | cargo feature による opt-in | 本リポは「feature フラグなしの cfg ベース」方針（REQ-2・`.claude/rules/coding-rust.md`）で optional 依存の前例がなく、採否は方針判断を要する |
| 案 C | 非公開の別クレート（例: `model-hub`）へ切り出し | 公開クレート `facade` から非公開クレートへ通常依存できない構造問題（`docs/facade-onnx-import-exposure-decision.md` §3.1・`api_surface.rs` が監視する形状と同型）を再生産する。公開クレート 8 個目の新設とセットでないと成立しない |

各案の承認コスト・影響範囲は上記のとおりで、推奨は案 A（`onnx-interop` と同様、既存クレート構成に収まり構造問題を再生産しない）とするが、確定は親 #2082 の配布方式決定とユーザー承認による。

## 5. 技術仕様案（承認後にそのまま実装できる粒度）

- **公開 API**: `ModelRegistry::download(&self, url: &str, name: &str, version: &str) -> Result<(), ModelError>`（イシュー指定シグネチャ）。`url` は呼び出し側が解決済みの直接 HTTPS URL であり（§6 のとおり `http`・`file`・`ftp` 等は fail-closed で拒否。平文 HTTP は許容しない）、HF hub のリポジトリ ID → URL 変換のような特定ハブ固有の解決はここでは行わない（2026-09-24 追記。§1 参照）。`&self` か関連関数かは兄弟 #2087 が確定する `ModelRegistry` の形状に追従する。
- **呼び出し側が信頼値・進捗シンクを渡す公開面（2026-09-24 追記。PR #2242 codex-review P1）**: 上記のイシュー指定シグネチャだけでは、§6 (A08) が必須とする「呼び出し側が明示する信頼済み `sha256` pin」と、本節「進捗ログ」の進捗コールバックを渡す経路がない。このため次の対を公開面案とする（PyTorch `torch.hub.download_url_to_file(url, dst, hash_prefix=None, progress=True)`・Keras `get_file(..., file_hash=None)` と同じく、信頼値は省略可能な引数として呼び出し側から与える形）:
  - `ModelRegistry::download_with(&self, url: &str, name: &str, version: &str, options: DownloadOptions<'_>) -> Result<(), ModelError>`: pin・進捗コールバックを受け取る本体。
  - `DownloadOptions<'a>`: `#[non_exhaustive]` の構造体とし、`DownloadOptions::new()`（既定 = pin なし・進捗なし）と builder メソッド `expected_sha256(Sha256Pin)`・`progress(&'a mut dyn FnMut(DownloadProgress))` で組み立てる（将来のタイムアウト・サイズ上限の上書き追加を非破壊にするため、位置引数を増やさない）。
  - `Sha256Pin`: `Sha256Pin::from_hex(&str) -> Result<Sha256Pin, ModelError>` で 64 桁の 16 進文字列だけを受理する検証済み値型。形式不正は**ネットワーク接続・キャッシュ参照より前に** `ModelError::InvalidPin` で拒否する（fail-closed）。
  - `download(&self, url, name, version)` は `download_with(url, name, version, DownloadOptions::new())` と等価の簡易形として残す。この形は pin を持たないため、保証範囲は破損・部分改変の検出（`download.json` 保存値との照合）に限られ、意図的改竄に対する真正性は保証しないことを rustdoc に明記する。
  - **pin 指定時の検証順序**: (1) 新規取得では一時ファイルの SHA-256 を pin と照合し、不一致なら `ModelError::HashMismatch` で拒否して一時ファイルを削除する（`rename` しない）。(2) 304 応答・既存キャッシュの再利用では、`download.json` の保存値ではなく pin に対して既存 `model.safetensors` を再計算・照合する。不一致なら再利用せず無条件 GET で全量を再取得し、再取得後も pin と一致しなければ `HashMismatch` で拒否する。(3) `download.json` の保存値が pin と異なる場合も、そのキャッシュは再利用しない。
- **`ModelError` 拡張方針**: #2087 が定義する想定の `#[non_exhaustive]` enum へ追加バリアント（`Http { status: u16, url: String }`・`Network(io::Error 相当)`・`InvalidUrl`・`InvalidName`・`InvalidPin`・`HashMismatch`・`SizeLimitExceeded`・`StorageFull`・`Permission`・`CacheMetadata`・`Safetensors(LoadError)` 等）を追加する方針とする。#2087 が先にマージされた場合はその実際の定義に従う。
- **キャッシュ有効判定（etag）**: `<name>/<version>/download.json`（`serde_json`。許容区分内）に `url`・`etag`・`content_length`・`sha256`・`fetched_at` を保存する。再取得時は `If-None-Match` 条件付き GET（304 なら再利用）または HEAD の `ETag` 比較を行う。etag 非提供サーバは常に再取得する（安全側）。**304 応答時のローカルキャッシュ再検証（A08 整合性契約）**: 304 応答はリモート側の表現が変化していないことをサーバーが主張するのみであり、ローカルの `model.safetensors` が前回書き込み以降に別プロセス・別ユーザーにより改変・破損していないことは保証しない（キャッシュディレクトリが共有パスになりうる脅威モデルは §6 と同一）。このため 304 応答を無条件の再利用シグナルとして扱ってはならず、公開順序は次のとおりとする: (1) 304 受信時点でまず既存 `model.safetensors` を `download.json` 保存済みの `sha256` に対してストリーミング再計算・比較する、(2) 一致すれば「有効なキャッシュ」と確定し `download.json` の `fetched_at` のみを更新する（更新自体も REQ-7 と同型の一時ファイル + rename で行い、§6 の dirfd 契約に従う）、(3) 不一致であればキャッシュ破損・改変とみなし 304 を信頼せず破棄し、条件付きヘッダを外した無条件 GET で全量を再取得して §5 の新規取得フロー（一時ファイル書き込み → 検証 → §6 dirfd 契約での `model.safetensors`／`download.json` 再作成）をそのまま適用する（早期 return しない）。
- **書き込み**: 一時ファイル（同一ディレクトリ内）へストリーミング書き込み → 検証成功後に `rename` で `model.safetensors` へ公開する。REQ-7 の `st_save`（一時ファイル + rename）と同型の原子性契約とする。**既存キャッシュへの再公開（2 回目以降の fetch）**: `model.safetensors`／`download.json` が既に存在する場合（キャッシュ更新・§5 の 304 再検証で不一致となり破棄再取得する場合を含む）でも公開は失敗させず atomic replace で上書きする。§6 (b) のとおり `renameat2` の `RENAME_NOREPLACE`（宛先が既存の場合に失敗する意味論）は付与しない。`RENAME_NOREPLACE` を使うと 2 回目以降の fetch がすべて公開失敗するため、本節の原子性契約は「既存有無に関わらず成功する atomic replace」であることをここで確定する。
- **検証方式の決定点**（doc に両案を記録し推奨のみ書く。確定は承認後の実装イシューで行う）:
  - (i) 「F32 ロード可能性」: 既存 `load_safetensors_f32_from_bytes` をそのまま流用（新規コードなしだが、REQ-7 の F32 限定契約により bf16／f16 のモデルはキャッシュ不可・全量をメモリへ二重に載せる）
  - (ii) 「フォーマット妥当性」: ヘッダ長（先頭 8 byte）＋ヘッダ JSON パース＋オフセット整合のみ検証（dtype 非依存・ストリーミング可）。facade は `safetensors` クレートへ直接依存できないため、`onnx-interop::st_load` 側へ小さなヘッダ検証関数を追加する必要がある
  - 推奨: (ii)（用途に dtype 制約を持ち込まない）。採否は承認後の実装イシューで確定する
- **エラー処理**: HTTP 非 2xx（3xx はリダイレクト上限付きで追従）・接続／タイムアウト・`io::ErrorKind::StorageFull`／`PermissionDenied`・`Content-Length` と実受信長の不一致・サイズ上限超過を型付きエラーで返し、本番経路で `unwrap`／`expect` を使わない（`.claude/rules/coding-rust.md`）。
- **進捗ログ**: `log`／`tracing` も許容区分外のため新規依存を作らない。呼び出し側が `DownloadOptions::progress` で渡す `&mut dyn FnMut(DownloadProgress)` コールバックに受信バイト数／総バイト数を書く設計とし、既定は無出力とする（2026-09-24 追記: 渡し口を `DownloadOptions` に確定。代替案の `Option<&mut dyn std::io::Write>` シンクは不採用）。
- **スコープ**: 同期・単一接続。並列ダウンロード・resume・認証トークンはイシュー明記のスコープ外（§10）。
- **タイムアウト・上限の既定値**: 接続／読み取りタイムアウト・最大サイズ・リダイレクト回数を承認後の実装イシューで提案・確定する（本 doc では確定しない）。

## 6. セキュリティ設計（OWASP Top 10。`.claude/rules/security.md`）

承認後の実装が満たすべき受け入れ条件として、以下をチェックリスト形式で記録する。

- [ ] **A01／A03（パストラバーサル・インジェクション）**: `name`・`version` はキャッシュディレクトリ配下のパス要素になるため、空文字・`.`／`..`・パス区切り（`/`・`\`）・絶対パス・NUL・制御文字を fail-closed で拒否する allowlist 検証（英数字・`-`・`_`・`.` の限定集合、先頭 `.` 禁止）を行う
- [ ] **A01／A03（キャッシュ書き込み時のシンボリックリンク脱出防御）**: 上記の文字列検証だけでは、`<name>`／`<version>` ディレクトリ自体が既にキャッシュルート外を指すシンボリックリンクである場合（キャッシュディレクトリは他プロセス・他ユーザーが書き込み可能な共有パスになりうる）、検証済みの一時ファイル・`model.safetensors`・`download.json` をキャッシュルート外の任意ディレクトリへ作成できてしまう。**文字列パスの再解決を前提にした「検査してから rename」方式は、検査（stat／canonicalize）と rename の間に `<name>`／`<version>` またはその親要素をシンボリックリンクへ差し替えられる TOCTOU を構造的に防げない**（最終 rename の直前に宛先パスを再検証しても、その再検証自体が新たな文字列パス解決であり競合窓口になる）。承認後の実装は次のディレクトリハンドル基点の契約とし、書き込み確定まで文字列パスを再解決しない: (a) キャッシュルートから対象パスまでの各パス要素（`<name>`・`<version>` を含む中間ディレクトリ）を、ルートから 1 段ずつ dirfd 相対でオープンする。Linux では `openat2`（`RESOLVE_NO_SYMLINKS | RESOLVE_BENEATH` 相当）を優先し、`openat2` 非搭載の環境（対応カーネルを持たない Linux 含む）では各段を `openat(parent_dirfd, component, O_NOFOLLOW | O_DIRECTORY | O_CLOEXEC, ...)` で明示指定してオープンする代替経路を安全な同格手段として認める（`O_NOFOLLOW` でシンボリックリンクを拒否し、`O_DIRECTORY` で「その要素が通常ファイル等でありディレクトリでない」場合もオープン自体を失敗させる。いずれも dirfd 相対操作であり文字列パス再解決を伴わないため、この代替経路は (c) の fail-closed 対象ではない）。以降の一時ファイル作成・`model.safetensors`・`download.json` の作成・書き込み・`renameat`／`renameat2` による公開はすべてこの検証済みディレクトリファイルディスクリプタ（dirfd）相対の操作のみで完結させる。生成した dirfd は次のパス要素をオープンするまで保持し、途中で文字列パスへ戻して再オープンしない。初回ダウンロード等で `<name>`／`<version>` に対応するディレクトリが未作成の場合の手順を明記する: 各段の `openat(parent_dirfd, component, O_NOFOLLOW | O_DIRECTORY | O_CLOEXEC, ...)`（または `openat2`）が `ENOENT` を返した時点で、`mkdirat(parent_dirfd, component, 0o755)` によりその 1 段だけをディレクトリとして作成し、作成後に同じ `openat(parent_dirfd, component, O_NOFOLLOW | O_DIRECTORY | O_CLOEXEC, ...)` で改めてオープンして dirfd を得る（`std::fs::create_dir_all` 等の文字列パスベースの再帰作成 API は使わない。中間ディレクトリの作成もこの `mkdirat` 相対操作のみで 1 段ずつ行い、文字列パスの一括再帰作成へフォールバックしない。フォールバックすると symlink 追従が復活し、本節が防ごうとしている脱出経路が再発するため）。`mkdirat` が `EEXIST` を返した場合（並行する別プロセス・別スレッドとの作成競合）はエラーとせず、同じ `openat(parent_dirfd, component, O_NOFOLLOW | O_DIRECTORY | O_CLOEXEC, ...)` を再試行し、新規作成に成功した場合と同じ経路（オープン成功）へ合流させる。`mkdirat` 後の再オープンがなお失敗する場合（`ENOTDIR`＝該当要素が symlink 等の非ディレクトリへ差し替えられていた、`ELOOP` 等）はエラーとして処理を中断し、その段から先の作成・書き込みへは進まない（fail-closed）。(b) 一時ファイルは検証済み dirfd に対して `openat(..., O_CREAT | O_EXCL | O_NOFOLLOW, ...)` で作成し、最終公開は同じ dirfd を親として `renameat`（`renameat2` が使える環境でも `RENAME_NOREPLACE` は付与しない）で `model.safetensors`／`download.json` へ atomic replace する。§5 の etag／304 検証で再取得した内容を同名の既存ファイルへ上書き公開する経路があるため、`RENAME_NOREPLACE`（既存ファイルがあれば失敗する意味論）は本設計と両立しない。rename の直前に文字列パスを再解決して宛先の型を再確認する手順は行わない（dirfd 相対操作である時点でパス再解決自体が発生しないため TOCTOU 窓口が存在しない）。(c) 上記のいずれの dirfd 相対経路（`openat2`・`O_NOFOLLOW | O_DIRECTORY` 付き `openat`・`renameat`／`renameat2`）も安全に提供できない OS・実行環境（例: Windows。dirfd 相対 API を持たない）では、その環境向けの `download` 実装を fail-closed で非対応（明示エラーを返し書き込みを一切行わない）とし、文字列パス再解決＋事後検査による近似実装で代替しない。(d) 回帰テストとして、`<name>` または `<version>` に該当するディレクトリを事前にキャッシュルート外を指すシンボリックリンクへ差し替えたうえで `download` を呼び出し、キャッシュルート外への書き込みが発生せずエラーになることに加え、検査直後・rename 直前のタイミングでシンボリックリンクへ差し替える競合を模したケースでも同様にキャッシュルート外への書き込みが発生しないことを確認するテストを実装イシューの受け入れ基準に含める。加えて、`<name>`／`<version>` ディレクトリが存在しない初回ダウンロード（`mkdirat` 経路）についても、(i) 通常の新規作成が成功しキャッシュルート配下にのみディレクトリが作られること、(ii) `mkdirat` の `EEXIST`（作成競合）後の再オープンが正しく合流して同じ結果になること、(iii) `mkdirat` 実行前後に対象要素をキャッシュルート外を指すシンボリックリンクへ差し替える競合を模しても、後続の再オープンが `ENOTDIR`／`ELOOP` 等で失敗しキャッシュルート外への書き込みが発生しないこと、を確認するテストを同基準へ含める。**依存の注記**: `std::fs` は dirfd 相対のオープン・rename を表現できない（`OpenOptionsExt::custom_flags` はパスの最終要素のオープンにしか作用せず、中間ディレクトリの dirfd 相対解決には使えない）ため、本方式の実装には `libc` または `rustix` 等 OS 呼び出しラッパーの追加が必要になる見込みである。これは許容依存 9 区分（`.claude/rules/deps-policy.md`）のいずれにも属さないため、HTTP クライアント依存（§7-1）と同じユーザー承認ゲートを要する事項として §7 へ追記する
- [ ] URL は `https` スキームのみ許可する（`http`・`file`・`ftp` は拒否）。リダイレクト先にも同じ検証を通す
- [ ] **テスト経路の分離（2026-09-24 追記）**: §8 item 6 のローカル `TcpListener` 平文 HTTP モックは CI 実行可能性のためのテスト専用注入経路であり、本チェックリストの `https` スキーム限定検証を回避・弱体化するものではない。スキーム拒否ロジック自体（`http://` 入力を渡した際に `download` が接続前に拒否すること）は専用の単体テストで検証し、モックを使う転送系テストはスキーム検証を通過済みの内部コンポーネント（transport 層。スキームチェックを含まない）のみを対象とする設計とする
- [ ] **A05（設定不備）／A02**: TLS 証明書検証を無効化するオプションは設けない
- [ ] **A08（整合性）**: `Content-Length` 上限・実受信長一致を検証する。任意の `sha256` ピンを指定できるようにし（公開面は §5 の `download_with`＋`DownloadOptions::expected_sha256(Sha256Pin)`。2026-09-24 追記）、指定時は不一致で拒否して一時ファイルを削除する。ピンの形式不正（64 桁 16 進以外）は接続前に `InvalidPin` で拒否する。safetensors ヘッダ検証後にのみ `rename` する（検証前のファイルをキャッシュへ置かない）。**304 応答時も無検証で再利用しない**（§5「304 応答時のローカルキャッシュ再検証」参照。`download.json` 保存済み `sha256` によるストリーミング再検証を必須とし、不一致ならキャッシュを破棄して無条件 GET で再取得する）。呼び出し側が `sha256` ピンを指定している場合、304 時の再検証は `download.json` 保存値ではなくそのピンに対して行う（`download.json` 自体が改変対象になりうるため、ピンが与えられているときはピンを優先する。保存値は破損・部分改変の検出用、ピンは呼び出し側が信頼する真の値に対する検証用という役割分担とする）
- [ ] **A06（脆弱コンポーネント）**: 採用候補は `=x.y.z` 完全固定・`cargo deny check advisories bans licenses sources` 通過を採用条件に含める
- [ ] **A09（ログ）**: 進捗コールバックに URL のクエリ文字列・認証情報を流さない。URL に埋め込まれた資格情報（`user:pass@host` 形式）は拒否する
- [ ] **A10（SSRF）**: ライブラリ利用者が渡す URL をそのまま接続する性質上、内部ネットワーク宛先の判定・遮断はライブラリの責務としない（責務は呼び出し側）。`https` 限定と資格情報付き URL の拒否のみを担保範囲とすることを明記する
- [ ] 外部プロセス委譲案（`curl`／`wget` 起動）は A03 の観点で不採用とする（§3 と同一理由）

## 7. 承認事項（本イシュー時点ではいずれも未取得）

**追記（2026-10-03・#2621）**: 区分番号・承認項目の現行版は §14（特に §14.2 の対応表・§14.6 のチェックリスト）を正とする。以下の本文は作成時点（2026-09-23〜24）の記述であり書き換えない。

**追記（2026-10-07・#2621）**: §14.3 の推奨案を取り下げたため、§14.6 のチェックリストは「承認待ち」ではなく「再評価待ち」である（冒頭の「現在の状態」・§14.6・§15.8）。区分番号の読み替え（§14.2）は引き続き有効。

1. HTTP クライアント（＋ TLS スタック）を許容依存へ新規区分として追加すること（現行の許容依存 9 区分〈`.claude/rules/deps-policy.md`〉に続く第 10 区分相当。配置案（§4）の選択を含む）
2. facade 公開面の拡張: `ModelRegistry::download`・`ModelRegistry::download_with`・`DownloadOptions`・`Sha256Pin`・`DownloadProgress`（§5。2026-09-24 追記で pin・進捗の渡し口を追加）と `ModelError` の追加バリアントの追加、および `api_surface.rs` 到達性テストの追加
3. `docs/license-matrix.md` への行追加（依存追加とセット。承認前は行を増やさない）
4. 承認後の実装イシューの起票（`.claude/rules/out-of-scope-tracking.md` により起票自体もユーザー承認が必要なため本 PR では起票しない。§9 に起票草案を記録する）
5. **キャッシュ書き込みの dirfd 相対操作（§6 シンボリックリンク脱出防御）に必要な OS 呼び出しラッパー（`libc` または `rustix` 等）を許容依存へ新規区分として追加すること**（1 の第 10 区分に続く第 11 区分相当。両者は承認単位が異なるため区分番号を別に確定する）。`std::fs` は dirfd 相対のオープン・rename を提供しないため、§6 の TOCTOU 非構造化契約をそのまま実装するには 1 と同様の依存追加承認が別途必要（HTTP クライアント依存とは別クレート・別ゲート）

## 8. 承認後の実装手順（順序付き）

1. `Cargo.toml` `[workspace.dependencies]` へ HTTP クライアント候補（§7-1）・dirfd 相対操作用の OS 呼び出しラッパー（§6・§7-5）を `=x.y.z` 固定で追加（feature は §3 の実測構成どおり）・`Cargo.lock` 更新・`docs/license-matrix.md` 行追加（`cargo tree` 実測付き）・`cargo deny check` 通過確認・`scripts/check-forbidden-deps.sh` 通過確認
2. `crates/facade/Cargo.toml` へ結線（§4 の配置案に従う）
3. `crates/facade/src/model.rs`（#2087 成果物）へ `download`・`download_with`・`DownloadOptions`・`Sha256Pin` と `ModelError` 追加バリアントを実装。`name`／`version` 検証は #2087 の検証関数を再利用する
4. `download.json` メタデータ・条件付き GET（304 時は §5 の `sha256` ストリーミング再検証を経てから再利用可否を確定する）・§6 (a) の `mkdirat` 経路を含む dirfd 相対の一時ファイル + rename・サイズ上限・進捗コールバックを実装する（`url` は汎用 HTTPS URL のみ。HF hub URL 変換は含まない。2026-09-24 追記）
5. `api_surface.rs` に到達性テストと「facade が HTTP クレートの型を公開面に漏らさない」テストを追加する
6. テスト: ローカル `std::net::TcpListener` による最小 HTTP モック（200／304／404／リダイレクト／`Content-Length` 不一致／サイズ超過／pin 一致・不一致〈新規取得・304 再利用の両経路〉）で CI 実行可能にする。`Sha256Pin::from_hex` の形式不正拒否は接続前に失敗することを単体テストで検証する。実ネットワークを使うテスト・TLS 経路の実接続テストは `#[ignore]` 分離（`.claude/rules/coding-rust.md`）
7. docs: 本ドキュメントへ実装記録節を追記・`docs/README.md` 注釈更新・`docs/compat-feature-gap.md`／`docs/compat-api-scope.md` の該当行があれば更新する

## 9. 引き継ぎ（起票草案。本イシューでは起票しない）

- 「feat(facade): `ModelRegistry::download` の実装（HTTP クライアント依存承認後）」— 前提: §7 の 1〜3。内容: §8
- 必要に応じ「chore(deps): HTTP クライアント依存の追加とライセンス実測」を分離
- HF hub 連携（`resolve` URL 形式・`ETag`／`X-Linked-Etag`・CDN リダイレクトの外部仕様確認を含む）は #2088 のスコープ外。別クレートとして #2243（#2244〜#2246）で追跡する（2026-09-24 追記。§1 参照）

## 10. スコープ外

- 並列ダウンロード・resume／partial ダウンロード（イシュー明記）
- HF private repo 認証トークン（イシュー明記）
- ミラー／プロキシ設定
- レジストリ manifest の version 記法（#2087 側スコープ）
- ONNX モデルのダウンロード（safetensors 限定）
- **HF hub 連携全体**（リポジトリ ID → URL 変換・`resolve` API 呼び出し・LFS／CDN リダイレクト追従等）。facade（本 #2088）には入れず、別クレートとして #2243（#2244〜#2246）で扱う（2026-09-24 ユーザー決定「他ライブラリと同じにする」。詳細は `docs/model-distribution-design.md` §5）

## 11. 前方参照の扱い

親 #2082（`docs/model-distribution-design.md`）・兄弟 #2087（`crates/facade/src/model.rs`・`ModelRegistry`・`ModelError`）は基準コミット時点で未マージ・PR 未作成（`git ls-remote --heads origin` に `2082`／`2087` を含むブランチなし・`gh pr list --search "2082 OR 2087"` が空・`crates/facade/src/model.rs` はリポジトリ内に存在しないことを 2026-09-23 に実測確認済み）。本ドキュメントは両者の内部構造（メソッドシグネチャ・enum バリアント）を確定事項として書かず、「並行イシュー #2082／#2087 の成果物（本 doc 作成時点で未マージ）。マージ状況により §5・§8 の記述を追従させる」ことを前提とする。

**追記（#2082 作成時点）**: 親 #2082 は `docs/model-distribution-design.md` として作成済み（本 doc §4「facade API 契約案」が本 doc §7 の承認事項のうち依存追加 2 件を要約する）。兄弟 #2087 も実装済み・マージ済み（`crates/facade/src/model.rs`・`ModelRegistry`・`ModelError`）。本節（§11）自体は基準コミット時点の実測記録として不変のまま維持する。

**追記（2026-09-24。HF hub 連携の分離）**: ユーザー決定「他ライブラリと同じにする」（PyTorch `torch.hub`／TensorFlow-Keras `tf.keras.utils.get_file` と同型に、汎用ダウンロードはコア・HF hub 連携は別クレート）に基づき、本 #2088 の対象を汎用 URL ダウンロードのみへ確定し、HF hub 固有の処理（リポジトリ ID → URL 変換・`resolve` API・CDN リダイレクト等）を対象から除外した。既存追跡先だった §9 の「要確認」項目は #2243（#2244〜#2246）へ引き継いだ。本追記に伴い §1・§5・§8・§9・§10 の該当箇所を編集した（本節・§2 の実測事実は編集していない）。詳細・根拠出典は `docs/model-distribution-design.md` §5 を参照。

**追記（2026-09-24。取得元の表記統一と親文書の緩い要約の是正）**: `docs/model-distribution-design.md`（正本ではなく本 doc を子文書として参照するハブ文書）が §6 の `https` スキーム限定契約より緩い「HTTP(S) URL」という表記を使っていた codex-review 指摘を受け、本 doc 自身の §1・§5・§8 でも同じ表記揺れ（「HTTP(S) URL」）があったため「HTTPS URL」へ統一した（§6 のチェックリスト自体は当初から `https` 限定・`http` 拒否と正しく記述されており変更していない）。あわせて §6 に「テスト経路の分離」チェック項目を追加し、§8 item 6 のローカル HTTP モックが本番の `https` 限定契約を回避しないことを明記した。また、`docs/model-distribution-design.md` §2.2 の manifest 要約が本 doc §5・§6 (A08) の「保存済み `sha256` は破損・部分改変検出用、真正性は呼び出し側の信頼済み pin が根拠」という区別を「改竄・破損対策」と一括りにしていた点も是正した（本 doc §5・§6 (A08) 自体の記述は当初から当該区別を正しく持っており変更していない）。

**追記（2026-09-24。信頼済み pin・進捗コールバックの渡し口）**: イシュー指定シグネチャ `download(&self, url, name, version)` のままでは、§6 (A08) が必須とする呼び出し側の信頼済み `sha256` pin と、§5「進捗ログ」の進捗コールバックを渡す経路がなく、セキュリティ契約を実装できないという codex-review 指摘（PR #2242・P1）を受け、§5 に `download_with`＋`DownloadOptions`（`expected_sha256(Sha256Pin)`・`progress`）を公開面案として追加し、pin 指定時の検証順序（新規取得・304 再利用・保存値と pin の不一致）を確定した。同類の点検として、本 doc の他の受け入れ条件（タイムアウト・サイズ上限・リダイレクト回数）は既定値を実装側が持つ設計であり呼び出し側から渡す必要がないことを確認した（将来の上書きは `DownloadOptions` への非破壊追加で行う）。§6 (A08)・§7-2・§8 の手順 3・6 を合わせて更新し、親ハブ文書 `docs/model-distribution-design.md` §2.2・§4 も同じ契約へ揃えた。

## 12. 出典一覧

`.claude/rules/deps-policy.md`・`.claude/rules/security.md`・`.claude/rules/coding-rust.md`・`docs/license-matrix.md`・`deny.toml`・`crates/facade/src/interop/safetensors.rs`・`crates/facade/tests/api_surface.rs`・`docs/facade-safetensors-exposure-decision.md`・`docs/facade-onnx-import-exposure-decision.md`・イシュー #2082／#2087／#2088／#2620（§13 の実測）／#2621（§14 の起案）。HF hub 連携の追跡先: イシュー #2243（#2244／#2245／#2246）。

## 13. 候補 crate のライセンス・推移依存の実測（#2620）

HTTP クライアント候補 4 種を feature 組合せ×ターゲット別に実測し、§3 の「未実測」を埋める。親: #2619（Phase 5 #2606 配下）。**推奨 crate の選定・deps-policy の区分起案・ユーザー承認依頼は兄弟 #2621 の担当であり、本節は実測値の記録に徹する**（承認は未取得・依存は未追加）。

区分番号の注記: `libc` が 2026-09-28 に第 10 区分として新設済み（`.claude/rules/deps-policy.md`）のため、HTTP／TLS 依存は**第 11 区分相当**になる。§2 の「9 区分」・§7-1 の「第 10 区分相当」・§7-5 の `libc` 未承認扱いは本 doc 作成時点（2026-09-23〜24）の記述であり、§7 の本文は書き換えない。

### 13.1 実測環境と手順

| 項目 | 値 |
|---|---|
| 実測日 | 2026-10-03 |
| 基準コミット | `eb27531c7ea30354c58c3259d30f91526d60a856`（`origin/main`） |
| ツール | cargo 1.98.1・rustc 1.98.1・cargo-deny 0.19.8 |
| 固定版 | `ureq =3.4.2`・`minreq =3.0.0`・`attohttpc =0.31.0`・`reqwest =0.13.5`（いずれも `=x.y.z`。crates.io 由来） |
| 実測対象 | 本体 workspace の外に作った一時パッケージ（独自 `[workspace]`・`<tmp>` 配下・実測後に削除）。本体の `Cargo.toml`／`Cargo.lock`／`deny.toml` は変更していない |
| ターゲット | `x86_64-unknown-linux-gnu`・`aarch64-unknown-linux-gnu`・`aarch64-apple-darwin`（license-matrix §3 の軸 1）と、参考の `x86_64-pc-windows-msvc`（§6 (c) のとおり Windows の download は fail-closed 非対応のため、公開前提の参考値） |

手順（組合せごと。`<pkg>` は一時パッケージ）:

1. `cargo generate-lockfile`（`<pkg>` 内）で解決版を確定する
2. 推移依存数: `cargo tree --locked -e normal,build --target <t> --prefix none` の出力を `sort -u` し、ルート自身を除いた個数
3. ライセンス式: `cargo metadata --locked --format-version 1 --filter-platform <t>` の resolve に含まれる全パッケージの `license` を `jq` で抽出し、`MPL`・`GPL`・`CDLA`・`OpenSSL`・`BSD-3`・`AND` 結合・`0BSD` 等を機械的に洗い出す
4. `cargo deny --manifest-path <pkg>/Cargo.toml --locked check --config <deny.toml> licenses bans sources`。設定はルート `deny.toml` の複製に `[graph] targets = ["<t>"]` を足したもの（allow リストは不変。ターゲット別判定のため）。さらにルート `deny.toml` そのまま（全ターゲット）でも実行した
5. **`cargo build`／`cargo check` は実行していない**（未承認 crate の build script〈`aws-lc-sys`・`openssl-sys`・`ring`〉を走らせないため）。システムライブラリ要件は `cargo tree` の依存関係から判断した（**2026-10-07 追記**: `ureq` の 3 構成は使い捨てプロジェクトで実ビルド済み → §15.3）
6. `advisories` は未実施（承認後に実施する。依存を追加する PR では `deny-checks` に含まれ CI が検査する）（**2026-10-07 追記**: `ureq` の 3 構成は使い捨てプロジェクトで実施済み → §15.3）

ルート `deny.toml` の allow は `MIT`・`Apache-2.0`・`Apache-2.0 WITH LLVM-exception`・`ISC`・`Zlib`・`Unicode-3.0`・`Unlicense`・`BSD-2-Clause` の 8 種で、本実測では免除も緩和もしていない（fail は測定結果として記録）。

### 13.2 実測結果

推移依存数は「x86_64-linux／aarch64-linux／aarch64-darwin／x86_64-windows」の順（ルート自身を除く。normal＋build エッジ）。`deny` は `licenses`／`bans`／`sources` の 3 検査で、`bans`・`sources` は全組合せ ok、差が出るのは `licenses` のみ。ターゲット別判定と全ターゲット判定で allow 外 crate の集合は一致した（ターゲット間差なし）。

| 候補 | feature 組合せ | 推移依存数 | `deny licenses` | allow 外の crate と識別子 |
|---|---|---|---|---|
| `ureq` | ① 既定（`rustls`＋`gzip`） | 30／30／30／29 | FAILED | `subtle 2.6.1`（BSD-3-Clause）・`webpki-roots 1.0.9`（CDLA-Permissive-2.0） |
| `ureq` | ② `rustls` のみ（`--no-default-features`） | 25／25／25／24 | FAILED | 同上 |
| `ureq` | ③ `rustls-no-provider`＋`platform-verifier` | 20／20／24／20 | FAILED | `subtle 2.6.1`（BSD-3-Clause）のみ |
| `ureq` | ④ `native-tls`（`native-tls-webpki-roots` を含む） | 35／35／30／20 | FAILED | `webpki-root-certs 1.0.9`（CDLA-Permissive-2.0）のみ |
| `ureq` | ⑥ `native-tls-no-default` | 34／34／29／19 | **ok** | なし |
| `ureq` | ⑤ TLS なし（`--no-default-features`。基準線） | 10／10／10／10 | ok | なし |
| `minreq` | ① `https`（= `https-rustls`） | 21／21／21／21 | FAILED | `aws-lc-sys 0.45.0`（`ISC AND … AND BSD-3-Clause AND …`）・`subtle`（BSD-3-Clause）・`webpki-roots`（CDLA-Permissive-2.0） |
| `minreq` | ② `https-rustls-probe` | 23／23／26／23 | FAILED | `aws-lc-sys`・`subtle`・`webpki-root-certs`（CDLA-Permissive-2.0） |
| `minreq` | ③ `https-native-tls` | 21／21／15／5 | **ok** | なし |
| `minreq` | ④ `https-openssl` | 19／19／19／19 | **ok** | なし |
| `minreq` | ⑤ TLS なし（基準線） | 1／1／1／1 | ok | なし |
| `attohttpc` | ① 既定（`compress`＋`tls-native`） | 59／59／57／48 | FAILED | `attohttpc 0.31.0` 自身（**MPL-2.0**） |
| `attohttpc` | ② `tls-rustls-webpki-roots`（`--no-default-features`） | 57／57／57／57 | FAILED | `attohttpc`（MPL-2.0）・`aws-lc-sys`・`subtle`・`webpki-roots` |
| `attohttpc` | ③ TLS なし（`--no-default-features`） | 38／38／38／38 | FAILED | `attohttpc`（MPL-2.0）のみ |
| `reqwest` | ① `blocking`＋`rustls`（`--no-default-features`） | 90／90／92／90 | FAILED | `aws-lc-sys`・`subtle`・`webpki-root-certs` |
| `reqwest` | ② 既定＋`blocking`（`charset` を含む） | 104／103／108／107 | FAILED | 上記に加え `encoding_rs 0.8.42`（`(Apache-2.0 OR MIT) AND BSD-3-Clause`） |

読み取れること（事実のみ）:

- 現行 allow で通る組合せは「TLS なし」を除くと、`ureq`⑥（`native-tls-no-default`）・`minreq`③④（`https-native-tls`・`https-openssl`）だけである。いずれも OS の TLS（Linux は OpenSSL、macOS は Security.framework、Windows は schannel）を使う。（**2026-10-07 追記**: `ureq`⑥ についての「OS の TLS を使う」は依存関係からの読み取りで、実測と一致しなかった。⑥ 単独では TLS の実装が組み込まれず、HTTPS は実行時 panic、生成バイナリは `libssl`／`libcrypto` にリンクされない → §15.3・§15.4。`minreq`③④ は未ビルドのまま → §15.7）
- rustls 系は全組合せで現行 allow 外を含む。要因は `subtle`（BSD-3-Clause）、`webpki-roots`／`webpki-root-certs`（CDLA-Permissive-2.0）、`aws-lc-sys`（複合式中の BSD-3-Clause）、`encoding_rs`（BSD-3-Clause との AND）で、いずれもコピーレフトではなく許容的ライセンスだが、**allow リストへの追加はユーザー承認事項**（deps-policy・license-matrix §1）である。
- MPL 等コピーレフトの推移的混入は、`attohttpc` の**直接ライセンス**（MPL-2.0）以外では検出されなかった（全 feature 組合せ・全ターゲット。`GPL`／`LGPL`／`AGPL`／`MPL`／`EPL`／`CDDL` の式を機械抽出し、該当は `attohttpc` のみ）。
- `ring 0.17.14` は式 `Apache-2.0 AND ISC` で、両識別子が allow に含まれるため通る（`ureq` ①②はこの経路を含む）。`aws-lc-rs 1.18.1`（`ISC AND (Apache-2.0 OR ISC)`）も通るが、同 `aws-lc-sys` は複合式の中に BSD-3-Clause を含み落ちる。

**注記（2026-10-03・#2621）**: `ureq`③（`rustls-no-provider`＋`platform-verifier`）は暗号プロバイダを同梱しない。`cargo info ureq@3.4.2` の feature 定義では `rustls = [rustls-no-provider, _ring, rustls-webpki-roots]` で、`_ring`（`rustls` の `ring` 有効化）は内部 feature のため、③は単体では TLS が成立しない構成である。実使用には呼び出し側での別途のプロバイダ指定が要り、現行 allow で通るかの判定対象としては不完全（実測数値は変更しない）。

**注記（2026-10-07・#2621）**: `ureq`⑥（`native-tls-no-default`）も、単体では HTTPS が成立しないことを実ビルドで確認した（実行時 panic。native-tls の実装は feature `native-tls`〈④〉でのみ有効になる）。⑥ の「`deny licenses` ok」は HTTPS が成立しない構成に対する判定である（実測数値は変更しない）。詳細は §15.4。

### 13.3 重点検証点の結果

| 対象 | 解決版 | ライセンス式 | 判定 |
|---|---|---|---|
| `webpki-roots`（`ureq` `rustls`・`minreq` `https-rustls`・`attohttpc` `tls-rustls-webpki-roots`） | 1.0.9 | CDLA-Permissive-2.0 | allow 外 |
| `webpki-root-certs`（`ureq` `native-tls-webpki-roots`・`minreq` `https-rustls-probe`・`reqwest` `rustls`） | 1.0.9 | CDLA-Permissive-2.0 | allow 外 |
| `ring`（`ureq` `_ring`） | 0.17.14 | Apache-2.0 AND ISC | 適合 |
| `aws-lc-rs`／`aws-lc-sys`（`reqwest` `rustls`・`minreq` の rustls 系・`attohttpc` rustls） | 1.18.1／0.45.0 | `ISC AND (Apache-2.0 OR ISC)`／`ISC AND (Apache-2.0 OR ISC) AND Apache-2.0 AND MIT AND BSD-3-Clause AND …` | `-sys` が allow 外（BSD-3-Clause） |
| `subtle`（rustls 経路で共通） | 2.6.1 | BSD-3-Clause | allow 外 |
| `rustls-platform-verifier` 系 | 0.6.2（`minreq`）／0.7.1（`ureq`・`reqwest`） | 本体は適合。Linux は `rustls-native-certs`・`openssl-probe`、macOS は `security-framework` が加わり、Windows は追加なし | 推移集合がターゲットで異なる（件数差の要因）。ただし `ureq`③の allow 外は `subtle` のみ |
| `encoding_rs`（`reqwest` 既定の `charset`） | 0.8.42 | `(Apache-2.0 OR MIT) AND BSD-3-Clause` | allow 外 |
| `attohttpc` | 0.31.0 | MPL-2.0（直接） | feature によらず allow 外（TLS なし③でも fail） |
| `minreq` | 3.0.0 | ISC（直接） | allow 内 |

**2026-10-16 是正（#3124）**: 上の §13.2 系の表にある「`aws-lc-sys` は `cmake` 必須」は、`aws-lc-sys 0.46.0` のビルドでは観測されなかった（§16.5。`cmake` なしでビルドが成功した）。本節の旧記述は当時の記録として書き換えない。

### 13.4 システムライブラリ・ビルド要件・MSRV

ライセンスとは別の運用上の制約。`cargo build` は実行していないため依存関係と manifest からの判断であり、実ビルドでの確認は承認後の実装手順（§8）で行う。（**2026-10-07 追記**: `ureq` の `native-tls` 系・`rustls`〈`ring`〉の実ビルド結果は §15.3・§15.6。`aws-lc-sys`・`minreq`・`attohttpc`・`reqwest` は未ビルドのまま）

| 経路 | 要件 |
|---|---|
| `native-tls`／`https-native-tls`／`attohttpc` 既定 | Linux は `openssl-sys 0.9.117`（システム OpenSSL の開発ヘッダ／ライブラリ、または `vendored` feature）。macOS は `security-framework 3.7.0`、Windows は `schannel 0.1.29` で追加の C ライブラリ不要。§3 の「環境非依存の開発コンテナ」方針との整合は要検証（**2026-10-07 追記**: `Dockerfile` に `libssl-dev` が無いことを確認した。`ureq` の feature `native-tls` は実行時にも `libssl.so.3`／`libcrypto.so.3` へ動的リンクする → §15.3・§15.6） |
| `minreq` `https-openssl` | 全ターゲットで `openssl-sys`（`openssl/vendored` を feature に含むため、C ツールチェーンで OpenSSL をソースビルドする経路） |
| `aws-lc-sys`（`reqwest` `rustls`・`minreq` の rustls 系・`attohttpc` rustls） | `cmake`（`cmake 0.1.58`）と C コンパイラを要する build script。全ターゲットで依存に現れる |
| `ring`（`ureq` `rustls`） | C／アセンブリのビルド（`cc`）。cmake は不要 |
| TLS なし | なし |

MSRV（`cargo info` の直接 crate）: `ureq` 1.85・`minreq` 1.63・`reqwest` 1.85.0・`attohttpc` は未宣言。解決集合内で宣言された最大値は rustls 系が `zeroize` の 1.85、`reqwest`／`attohttpc` が `icu_provider` の 1.88（`url` 経由）、`openssl-sys` 経路が 1.80。toolchain は stable（`rust-toolchain.toml`）で、開発コンテナのベースは rust:1.88。

### 13.5 本記録の範囲と後続

- 本体 `Cargo.toml`／`Cargo.lock`／`deny.toml` は変更していない（差分なしを確認済み）。一時パッケージはリポジトリ外で実測し、削除済み。
- 実測値は 2026-10-03 時点の crates.io 解決版に依存する。承認後の依存追加 PR では再実測する。
- 推奨 crate、区分起案、承認依頼は #2621 で扱う。`docs/license-matrix.md` §10 に下書きの要約を置いた（未承認）。

## 14. 依存区分の起案と承認依頼（#2621）

### 14.1 位置づけ

- **本節は未承認の起案である。承認の代行は行わない**。依存追加・`.claude/rules/deps-policy.md`／`deny.toml`／`Cargo.toml`／`Cargo.lock` の変更は本 PR に含めない。
- 根拠は §13 の実測（#2620）のみで、新たな `cargo build`／実測は行っていない。API・feature の確認は `cargo info`・docs.rs の公開ドキュメントによる。固定版は 2026-10-03 時点の crates.io 解決版であり、承認後の依存追加 PR で再実測する。（**2026-10-07 追記**: 使い捨てプロジェクトでの実ビルド・挙動確認を §15 に記録した。**推奨案の構成は HTTPS が成立しなかった**〈§15.4・§15.8〉。このため §14.3 の推奨は取り下げて承認対象から外し、§14.5-1・§14.6 は再評価待ちとした。本節の本文は経緯として起案時点のまま残す）
- 承認を得ても本 doc は deps-policy を書き換えない。規約・`license-matrix.md`・`deny.toml` への正式反映は承認後の別 PR で行う。

### 14.2 区分番号の対応表（旧記述 → 現行）

| 旧記述 | 現行の読み方 |
|---|---|
| §2「許容依存 9 区分（第 1〜8）」 | 現行は 10 区分。本体 workspace の直接依存は第 1〜8 区分と第 10 区分（`libc`） |
| §7-1「第 10 区分相当」（HTTP／TLS） | **第 11 区分相当** |
| §7-5「`libc`／`rustix` の新規区分（第 11 区分相当）」 | 新規区分ではない。`libc` は 2026-09-28 に第 10 区分として承認済み（用途は onnx-interop の external data に限定）。**第 10 区分の用途拡張**、または onnx-interop ヘルパーの公開（§14.5-2） |
| `model-distribution-design.md` §4-1・§4-2 | 上の 2 行と同じ読み替え |
| `hf-hub-integration-design.md` の承認表（「第 10/11 区分相当」） | 旧番号の引用として同じく読み替える（同 doc は #2622 の担当範囲のため本 PR では編集しない） |

### 14.3 推奨 crate の選定

> **推奨の取り下げ（2026-10-07・#2621）**: 本節の推奨案（`ureq =3.4.2`・`default-features = false`・`features = ["native-tls-no-default"]`）は、§15 の実測（2026-10-07）で **HTTPS が成立しないと判明した**（実行時 panic。§15.4）。このため**推奨を取り下げ、承認対象から外す**。採用可能な構成は再評価が必要で、**現時点の推奨は未定**である。本節の「代替案」「次点」も、取り下げた推奨案との比較で付けた位置づけであり、新しい推奨を意味しない。再評価で決める論点は §15.8 にまとめた。以下の本文は起案時点（2026-10-03）の経緯として残す（承認依頼としては読まない）。

**除外**: `attohttpc`（直接ライセンスが MPL-2.0。§13.2）・`reqwest`（推移依存 90〜108 個で `tokio` を内包。現行 allow で通る組合せなし）。

現行 allow で `licenses` が通る 3 組合せの比較（§13.2・§13.4 の値を引用。`cargo build` は未実行。**2026-10-07 追記**: `ureq` ⑥ の実ビルド結果は §15.3）:

| 軸 | `ureq =3.4.2` ⑥ `native-tls-no-default` | `minreq =3.0.0` ③ `https-native-tls` | `minreq` ④ `https-openssl` |
|---|---|---|---|
| (a) 直接ライセンスと deps-policy の適合基準（MIT OR Apache-2.0 系） | MIT OR Apache-2.0。合致 | ISC。allow 内だが基準の記述から外れ、例外扱いの記録が要る | 同左 |
| (b) §5・§6 の API 要件 | 下表のとおり大半を公開 API で確認 | 未確認（承認後に確認） | 未確認（承認後に確認） |
| (c) 推移依存数（x86_64-linux／aarch64-linux／aarch64-darwin／x86_64-windows）・MSRV | 34／34／29／19・1.85 | 21／21／15／5・1.63 | 19／19／19／19・1.63 |
| (d) システムライブラリ | Linux は `openssl-sys`（OpenSSL 開発パッケージ）が要る（**2026-10-07 追記**: ⑥ はビルド時に OpenSSL 開発パッケージを使うが、生成バイナリは `libssl`／`libcrypto` にリンクされなかった。HTTPS が成立した feature `native-tls` は実行時にも OpenSSL の共有ライブラリが要る → §15.3） | 同左（未ビルド → §15.7） | `openssl/vendored` により C ツールチェーンでソースビルド（全ターゲット） |

(b) の確認結果（`ureq =3.4.2`。docs.rs の公開ドキュメントによる。実装時に再確認する）:

| 要件（§5・§6） | 確認結果 |
|---|---|
| リダイレクトの各ホップで `https` 限定を検証 | `max_redirects` の既定が 0（自動追従しない）。呼び出し側で `Location` を検証しながら手動追従できる。`https_only` 設定もある（**2026-10-07 追記**: 実測では既定は 10 で自動追従した。`max_redirects(0)` の明示が要る → §15.5） |
| 接続・読み取りのタイムアウト | `timeout_connect`／`timeout_recv_body`／`timeout_global` 等を確認（**2026-10-07 追記**: 実測では既定は `await_100` 以外すべて `None`〈無期限〉で、明示設定が要る。`timeout_recv_body` の発火は実測、`timeout_connect` は未検証 → §15.5・§15.7） |
| ストリーミング受信・`Content-Length`／`ETag` の取得 | ボディのリーダー経由の受信とヘッダ取得ができる設計。サイズ上限・ハッシュ計算を逐次行えるかは承認後に実装で確認（**2026-10-07 追記**: 逐次受信・サイズ上限の挙動を実測 → §15.5） |
| `If-None-Match` の付与と 304 判別 | 任意ヘッダ付与と 304 の取得は可能な想定。ただし `>=400` をエラー化する既定の扱いは実装時に確認（**2026-10-07 追記**: 304 は既定のまま `Ok`、`>=400` は `Err(StatusCode)` → §15.5） |
| 証明書検証を無効化しない既定 | 既定は検証有効。`disable_verification` が存在するため、facade からこの設定を出さないことを条件にする（**2026-10-07 追記**: 既定の `disable_verification: false` と、未知の発行者・ホスト名不一致の拒否を feature `rustls`・`native-tls` で実測 → §15.4） |
| `native-tls-no-default` での実行時設定 | TLS プロバイダ（native-tls）とルート証明書源（OS ストア）をどう指定するかは**未確認（承認後に確認）**。同 feature は `webpki-root-certs` を含まない（**2026-10-07 追記**: 同 feature 単独では HTTPS が成立しなかった → §15.4） |

**推奨案（取り下げ済み・2026-10-07。承認対象外）**: `ureq =3.4.2`・`default-features = false`・`features = ["native-tls-no-default"]`。直接ライセンスが MIT OR Apache-2.0 で、現行 allow を変えずに `licenses` が通る。`minreq` は依存数が最小だが、直接ライセンスが基準外で API 要件も未確認のため次点とする。（**2026-10-07 追記**: この推奨案の構成は実測で HTTPS が成立しなかったため、推奨を取り下げた。「現行 allow を変えずに `licenses` が通る」は HTTPS が成立しない構成に対する判定だった〈§15.4〉。新しい推奨は定めておらず、再評価はユーザー判断を仰ぐ → §15.8）

**代替案**（allow リスト変更の別途承認が要る。**2026-10-07 追記**: 推奨案の取り下げ後も、これを新しい推奨に繰り上げてはいない）: `ureq` ②（`rustls`＝`ring`＋`webpki-roots`、推移依存 25 個）。システム OpenSSL が不要になる代わり、allow に `BSD-3-Clause`（`subtle`）と `CDLA-Permissive-2.0`（`webpki-roots`）の追加承認が要る（いずれもコピーレフトではない。§13.3）。（**2026-10-07 追記**: 「システム OpenSSL が不要」は実ビルドで確認したが、`ring` の build script は Linux → `aarch64-apple-darwin` のクロス lib ビルドで失敗した → §15.3・§15.6）

**トレードオフ**（取り下げた推奨案と代替案の比較。経緯として残す）: 推奨案は Linux のビルドに OpenSSL 開発パッケージを要求する。現行の `Dockerfile` は `pkg-config`・`build-essential` 等は入れるが `libssl-dev` を入れていない。CI の `ubuntu-latest` の有無は未確認。開発コンテナ・CI への導入要否は承認後の実ビルドで確認する（推測で「入っている」と書かない）。代替案はこの要件を避けられるが、allow 追加の承認が要る。（**2026-10-07 追記**: `Dockerfile`・CI の事実確認と、代替案〈`ring`〉の Linux → `aarch64-apple-darwin` クロス lib ビルド失敗の実測は §15.6。`ubuntu-latest` の導入状況は未検証のまま）

### 14.4 配置案と `cfg` 範囲

§4 の案 A／B／C を TLS 方式との組合せで再評価する。

**2026-10-07 追記**: 下表の「推奨案（native-tls）」列は、取り下げた §14.3 の推奨案（`native-tls-no-default` 単独）を前提に書いたものである。列名と本文は経緯として残すが、TLS 方式が未定のため、配置案の比較は再評価で選ぶ構成に合わせてやり直す必要がある（論点は §15.8）。実測との差分は次の 2 点: (1) native-tls で HTTPS が成立した feature `native-tls` は、ビルド時の OpenSSL 開発パッケージに加えて実行時にも OpenSSL の共有ライブラリを要する（§15.3）。(2) rustls の列の「C コンパイラ程度」は、Linux → `aarch64-apple-darwin` のクロス lib ビルドが `ring` の build script で失敗した実測（§15.6）を含んでいない。`cfg(unix)` 限定の選択肢は TLS 方式に依存しない。

| 配置案 | 推奨案（native-tls）との組合せ | 代替案（rustls）との組合せ |
|---|---|---|
| A: `facade` の無条件依存 | 全 `fandhe-ai` 利用者（Linux）のビルドに OpenSSL 開発パッケージを要求する | 利用者の追加要件は C コンパイラ程度（`ring`）。依存数が増える |
| B: cargo feature で opt-in | 本リポに optional 依存の前例なし。要件を download 利用者に限定できる | 同左 |
| C: 非公開の別クレート | §4 の構造問題（公開クレートから非公開クレートへ依存できない）を再生産する | 同左 |

`cfg` 範囲の選択肢:

- **`cfg(unix)` 限定（推奨）**: §6 (c) で Windows の download は fail-closed の非対応と確定済み。`libc`（第 10 区分）の前例と同型で、Windows の依存ツリーが増えない。
- 全ターゲット: 将来 Windows 対応の余地を残すが、現状は使われない依存を載せる。

配置案と `cfg` 範囲の採否はユーザー承認事項とする。

### 14.5 deps-policy 表形式の起案

> **再評価待ち（2026-10-07・#2621）**: 下の「1. 第 11 区分」の起案は、取り下げた §14.3 の推奨案を前提にしている。前提の構成が取り下げられたため、**現状のままでは承認できない（承認待ちではなく再評価待ち）**。crate・feature が決まるまで、この行は deps-policy へ転記できる形になっていない（論点は §15.8）。「2. dirfd 相対操作」（B-1／B-2）は HTTP／TLS の構成に依存せず、今回の取り下げの影響を受けない。

**1. 第 11 区分（HTTP クライアント／TLS）**（前提の構成を取り下げ済み。表は経緯として残す）

| 区分 | クレート | 条件 |
|---|---|---|
| HTTP クライアント／TLS（第 11 区分） | ~~推奨: `ureq`（`default-features = false`・`native-tls-no-default`）~~（取り下げ。HTTPS 不成立 → §15.4。crate・feature は未定） | `=3.4.2` 完全固定。`cfg` 範囲は §14.4 の承認結果に従う。用途は `ModelRegistry::download`／`download_with` による HTTPS 取得に限る。HTTP crate の型を facade の公開面に出さない（`api_surface.rs` のテストで担保。§8-5）。TLS 証明書検証を無効化するオプションを設けない（§6）。非同期ランタイム（`tokio`／`hyper`）を推移依存に含めない。依存追加 PR で `docs/license-matrix.md` に行を追加し `cargo tree` を再実測する。`cargo deny check advisories bans licenses sources` と `scripts/check-forbidden-deps.sh` を通す。代替案（`rustls`）を採る場合は allow 追加（`BSD-3-Clause`・`CDLA-Permissive-2.0`）を別途承認する。HF hub 連携の別クレート（#2243・#2622）が同じ区分を再利用する場合も承認単位は別とする |

**2. dirfd 相対操作（§6・§7-5）**。第 10 区分 `libc` はすでに承認済みのため、新規区分ではなく次の二択にする。

- **案 B-1（推奨）**: 第 10 区分の用途を `crates/facade/src/model.rs` のキャッシュ書き込み（`openat`／`openat2`／`mkdirat`／`renameat`）へ拡張する。版は `=0.2.189` のまま `cfg(unix)` 限定で facade に直接依存させる。新規 `unsafe` は FFI 境界に限り、理由コメントと security-auditor のレビューを必須とする。公開クレートの公開面を広げず、既承認ピンと同型である。
- 案 B-2: `onnx-interop/src/onnx/external_data.rs` の `pub(super)` ヘルパー（`open_chain_openat2`・`openat_no_follow` 等）を公開して facade から使う。`libc` の用途拡張は不要だが、crates.io 公開クレート `fandhe-ai-onnx-interop` の公開面が増える。また `mkdirat`／`renameat` は既存ヘルパーにないため追加が要る。

どちらでも §6 の symlink 脱出防御と、Windows 非対応の fail-closed 契約は変更しない。

### 14.6 承認チェックリスト（ユーザーが選ぶ。すべて未承認）

> **再評価待ち（2026-10-07・#2621）**: 本チェックリストは、取り下げた §14.3 の推奨案を前提に作った。前提の構成が取り下げられたため、**現状のままでは承認できない（承認待ちではなく再評価待ち）**。とくに 1 項目めを現状の文面のまま承認すると、HTTPS 取得という設計目的を満たせない。各項目の扱いを【】で付した。再評価の論点は §15.8。

- [ ] HTTP／TLS 依存の第 11 区分を新設すること、および crate と feature（推奨案 `ureq` `native-tls-no-default`／代替案 `rustls`）【取り下げた構成を前提にしている。「推奨案」は選べない。crate・feature は再評価後に書き直す】
- [ ] 代替案を採る場合の allow 追加（`BSD-3-Clause`・`CDLA-Permissive-2.0`）【「代替案を採る場合」という条件が前提ごと変わった。HTTPS の成立を実測した `ureq =3.4.2` の構成は、いずれも allow の追加を要する（§15.4・§15.8）。追加する識別子は構成によって異なるため、再評価後に書き直す】
- [ ] 配置案（A／B／C）と `cfg` 範囲（`cfg(unix)` 限定／全ターゲット）【§14.4 の比較は取り下げた推奨案の列を含む。TLS 方式が決まってから判断する】
- [ ] dirfd 経路の扱い（B-1／B-2）【取り下げの影響なし。HTTP／TLS の構成に依存しない】
- [ ] facade 公開面の拡張（§7-2 のとおり）【取り下げの影響なし。ただし実装は依存の承認が前提】
- [ ] 承認後の実装 issue の起票（§9 の草案を基にする。起票自体もユーザー承認事項）【1〜3 項目めが決まった後の事項】
- [ ] deps-policy・`docs/license-matrix.md`・`deny.toml` への正式反映を承認後の別 PR で行うこと【1〜3 項目めが決まった後の事項。反映する内容は未定】

### 14.7 承認後の手順との対応

- §8-1 の「HTTP クライアント候補」は §14.5-1 の承認結果の crate・feature・`cfg` を指す。「OS 呼び出しラッパー」は新規依存ではなく、B-1 なら第 10 区分 `libc` の用途拡張、B-2 なら onnx-interop ヘルパーの公開を指す。（**2026-10-07 追記**: §14.5-1 は前提の構成を取り下げたため、承認結果として指せる crate・feature は現時点で存在しない）
- 依存追加 PR では、(1) §13 の実測を承認時点の解決版で再実測し、(2) 未実施の `cargo deny check advisories` を実施する。（**2026-10-07 追記**: 使い捨てプロジェクトでの先行実施結果は §15.3。依存追加 PR での再実施は引き続き必要）

## 15. 使い捨てプロジェクトでの検証結果（2026-10-07・#2621）

**依存の採用は未承認のまま。本節は判断材料の実測記録である。** ユーザーが 2026-10-07 に承認した範囲は「workspace 外の使い捨てプロジェクトでの検証だけを先行する」ことに限られ、§14.6 の承認チェックリストはいずれも未承認のままである。本体の `Cargo.toml`／`Cargo.lock`／`deny.toml`・`.claude/rules/deps-policy.md`・`docs/license-matrix.md`・`docs/spec/` は変更していない。

本節は実測した値だけを書く。実測していない事項は「未検証」と明記する（§15.7）。

### 15.1 実行環境と対象

| 項目 | 値 |
|---|---|
| 実測日 | 2026-10-07 |
| 基準コミット | `33e8bbf274904f071a7db9c82bc46b6b74480e3c`（`origin/main`） |
| 実行環境 | Linux・x86_64・rustc 1.98.1（cargo 1.98.1・cargo-deny 0.19.8） |
| ホストのシステム要件の状態 | C コンパイラ（`cc`／`gcc`）・`pkg-config` あり。OpenSSL 3.5.5 と開発パッケージ（`libssl-dev`）導入済み。**`cmake` は未導入**。システムパッケージの追加導入は行っていない |
| 実測対象 | 本体 workspace の外の使い捨てパッケージ（`target/` 配下〈git 管理外〉・空の `[workspace]` テーブル・構成ごとに別ディレクトリ・実測後に削除） |
| 構成 N（§14.3 推奨案・§13.2 ⑥） | `ureq = { version = "=3.4.2", default-features = false, features = ["native-tls-no-default"] }` |
| 構成 R（§14.3 代替案・§13.2 ②） | 同 `features = ["rustls"]` |
| 構成 N′（補助。§13.2 ④） | 同 `features = ["native-tls"]`。構成 N で HTTPS が成立しなかった（§15.4）ため、切り分け用に追加で実測した |

解決版（3 構成共通の `ureq 3.4.2`・`ureq-proto 0.6.4`・`rustls-pki-types 1.15.1` 以外）: 構成 N・N′ は `native-tls 0.2.18`・`openssl 0.10.81`・`openssl-sys 0.9.117`・`openssl-probe 0.2.1`、構成 R は `rustls 0.23.45`・`rustls-webpki 0.103.15`・`ring 0.17.14`・`subtle 2.6.1`・`webpki-roots 1.0.9`、構成 N′ のみ `webpki-root-certs 1.0.9`。

### 15.2 実行コマンド

構成ごとのディレクトリ（`<pkg>`）で次を実行した。

1. `cargo generate-lockfile`
2. `cargo build --locked -vv`（build script の実行有無と出力をログから確認）
3. `cargo tree --locked -e normal --target x86_64-unknown-linux-gnu --prefix none`、同 `-e normal,build`（§13.1 手順 2 と同じ数え方。`sort -u` しルート自身を除く）、`cargo tree --locked -e normal -f '{p} {f}'`（feature 表示）、`cargo tree --locked -d`（重複）
4. `cargo metadata --locked --format-version 1 --filter-platform x86_64-unknown-linux-gnu` の resolve に含まれる全パッケージの `license` を集計
5. `cargo deny --locked check --config deny.toml advisories licenses`。`deny.toml` は使い捨てパッケージ内に置いた最小構成で、`[licenses] allow` はルート `deny.toml` の 8 種（§13.1）と同一（追加・免除なし）。使い捨てパッケージ自身には `license = "MIT OR Apache-2.0"` を付けた（付けない初回実行はルートパッケージ自身の `unlicensed` で fail したため。依存の判定には影響しない）。advisories は cargo-deny が取得した RustSec advisory-db（取得済み DB の先頭コミット日付 2026-10-03）に対する判定である
6. `cargo build --locked --target aarch64-apple-darwin --lib`（CI の `cargo build (linux / aarch64-apple-darwin)` ジョブが行う Linux ホストからのクロス lib ビルドと同じ形。§15.6）
7. 挙動確認: Python 標準ライブラリ（`http.server`・`ssl`）で 127.0.0.1 の空きポートに立てたローカルサーバ（平文 HTTP と、`openssl` コマンドで作ったローカル CA 署名の証明書〈SAN は `DNS:localhost` のみ〉による HTTPS）に対し、1 ケース 1 プロセスの最小 Rust プログラムを実行した。サーバは実測後に終了を確認した

外部ネットワークへのアクセスは crates.io からの依存取得と、手順 5 の advisory-db 取得（cargo-deny の既定動作）だけである。

### 15.3 2 構成の比較（x86_64-unknown-linux-gnu）

| 軸 | 構成 N（`native-tls-no-default`） | 構成 R（`rustls`） | 構成 N′（`native-tls`。補助） |
|---|---|---|---|
| `cargo build`（ホスト） | 成功 | 成功 | 成功 |
| 実行された build script | `httparse`・`libc`・`native-tls`・`openssl`・`openssl-sys`・`proc-macro2`・`quote` | `httparse`・`libc`・`ring`・`rustls` | 構成 N と同じ |
| `links`（ネイティブリンク） | `openssl-sys` → システムの `libssl`／`libcrypto`（`cargo:rustc-link-lib=ssl`・`crypto`、`cargo:include=/usr/include`、`version_number=30500050`） | `ring` → 自前ビルドの静的ライブラリ（`rustc-link-lib=static=ring_core_0_17_14_`） | 構成 N と同じ |
| ビルドに使われたシステム要件 | C コンパイラ・`pkg-config`・OpenSSL 開発パッケージ（ヘッダとライブラリ）。`cmake` は不要だった | C コンパイラ（`cc`）。`cmake`・OpenSSL は不要だった | 構成 N と同じ |
| 生成バイナリの動的リンク（`ldd`） | `libssl`／`libcrypto` へのリンク**なし**（§15.4 のとおり native-tls のコードが組み込まれていない） | `libssl`／`libcrypto` へのリンクなし | `libssl.so.3`・`libcrypto.so.3` に動的リンク（実行環境にも OpenSSL の共有ライブラリが要る） |
| 推移依存数（`-e normal`／`-e normal,build`） | 29／34 | 22／25 | 30／35 |
| 重複バージョン（`cargo tree -d`） | なし | なし | なし |
| ライセンス集合（`cargo metadata`） | `MIT OR Apache-2.0` 系 30（表記ゆれ `Apache-2.0 OR MIT`・`MIT/Apache-2.0` を含む）・`MIT` 2（`bytes`・`openssl-sys`）・`Apache-2.0` 1（`openssl`）・`(MIT OR Apache-2.0) AND Unicode-3.0` 1（`unicode-ident`） | `MIT OR Apache-2.0` 系 18・`ISC` 2（`rustls-webpki`・`untrusted`）・`MIT` 1（`bytes`）・`Apache-2.0 AND ISC` 1（`ring`）・`Apache-2.0 OR ISC OR MIT` 1（`rustls`）・**`BSD-3-Clause` 1（`subtle`）**・**`CDLA-Permissive-2.0` 1（`webpki-roots`）** | 構成 N の集合に **`CDLA-Permissive-2.0` 1（`webpki-root-certs`）** が加わる（構成 N との差分はこの 1 crate のみ） |
| `cargo deny check licenses` | ok | FAILED（`subtle v2.6.1`〈BSD-3-Clause〉・`webpki-roots v1.0.9`〈CDLA-Permissive-2.0〉が allow 外） | FAILED（`webpki-root-certs v1.0.9`〈CDLA-Permissive-2.0〉が allow 外） |
| `cargo deny check advisories` | ok（該当 advisory なし） | ok（該当 advisory なし） | ok（該当 advisory なし） |
| HTTPS の成立（§15.4） | **不成立**（実行時 panic） | 成立 | 成立 |
| Linux → `aarch64-apple-darwin` のクロス lib ビルド（§15.6） | 成功 | **失敗**（`ring` の build script） | 成功 |

推移依存数（`-e normal,build`）は §13.2 の値（⑥ 34・② 25・④ 35）と一致した。`deny licenses` の結果も §13.2 と一致した。

### 15.4 (c) TLS プロバイダとルート証明書源の実行時設定

`Agent::new_with_defaults()` の `TlsConfig` を `{:?}` で表示した結果は、3 構成とも `TlsConfig { provider: Rustls, client_cert: None, root_certs: WebPki, use_sni: true, disable_verification: false }` だった。**有効にした feature によって既定値は変わらない**（native-tls 系の構成でも既定のプロバイダは `Rustls`、ルート証明書源は `WebPki`）。

ローカル HTTPS サーバ（`https://localhost:<port>`）への GET の結果:

| 設定（`TlsConfig::builder()`） | 構成 N | 構成 R | 構成 N′ |
|---|---|---|---|
| 既定（`Rustls`＋`WebPki`） | panic「`uri scheme is https, provider is Rustls but feature is not enabled: rustls`」 | `Err`（`InvalidCertificate(UnknownIssuer)`。ローカル CA は webpki-roots に無いため検証で拒否） | panic（構成 N と同じ文言） |
| `provider(NativeTls)`＋ルート既定（`WebPki`） | panic「`uri scheme is https, provider is NativeTls but feature is not enabled: native-tls`」 | panic（左と同じ文言） | `Err`（`certificate verify failed`・`unable to get local issuer certificate`） |
| `provider(NativeTls)`＋`RootCerts::PlatformVerifier` | panic（同上） | panic（同上） | `Err`（同上。OS の証明書ストアにローカル CA が無いため検証で拒否） |
| `provider(NativeTls)`＋`RootCerts::Specific(ローカル CA)` | panic（同上） | panic（同上） | **`Ok` 200**（ボディ 65536 バイト受信） |
| 上と同じ設定で `https://127.0.0.1:<port>`（証明書の SAN と不一致） | panic（同上） | panic（同上） | `Err`（`certificate verify failed`・`IP address mismatch`） |
| `provider(Rustls)`＋`RootCerts::PlatformVerifier` | panic（`…Rustls but feature is not enabled: rustls`） | panic「`Rustls + PlatformVerifier requires feature: platform-verifier`」 | panic（構成 N と同じ） |
| `provider(Rustls)`＋`RootCerts::Specific(ローカル CA)` | panic（同上） | **`Ok` 200**（ボディ 65536 バイト受信） | panic（同上） |
| 上と同じ設定で `https://127.0.0.1:<port>` | panic（同上） | `Err`（`InvalidCertificate(NotValidForNameContext …)`） | panic（同上） |

実測から言えること:

- **構成 N（推奨案の `native-tls-no-default` 単独）は、`ureq =3.4.2` では HTTPS が成立しなかった。** `TlsProvider::NativeTls` を明示しても panic し、生成バイナリは `libssl` にリンクされていない。`ureq 3.4.2` のソースでは、native-tls のコネクタとモジュールが `#[cfg(feature = "native-tls")]` で括られており（`src/tls/mod.rs:14`・`src/unversioned/transport/mod.rs:389`）、`native-tls-no-default` は `dep:native-tls`・`dep:der`・`_tls` を有効にするだけで feature `native-tls` 自体を有効にしない（同 crate の `Cargo.toml` の `[features]`）。このため構成 N は `native-tls`／`openssl-sys` を依存に引き込み build script も走らせるが、TLS の実装は組み込まれない。§13.2 ⑥ の「`deny licenses` ok」は、この HTTPS が成立しない構成に対する判定だったことになる。
- `ureq =3.4.2` で native-tls による HTTPS が成立したのは feature `native-tls`（構成 N′）で、この feature は `native-tls-webpki-roots`（`webpki-root-certs`・CDLA-Permissive-2.0）を必ず含む。**今回実測した範囲では、現行 allow のままで HTTPS が成立する `ureq =3.4.2` の構成は見つかっていない**（構成 N は HTTPS 不成立、構成 R・N′ は allow 外を含む）。
- native-tls を使う場合（構成 N′）、プロバイダは自動選択されず、`TlsConfig::builder().provider(TlsProvider::NativeTls)` の明示が要る。OS の証明書ストアを使うには `root_certs(RootCerts::PlatformVerifier)` を指定する。
- rustls（構成 R）は既定設定のまま動き、ルート証明書源は同梱の `webpki-roots` になる。OS ストア（`PlatformVerifier`）は feature `platform-verifier` が無いと panic する。
- 証明書検証は既定で有効で（`disable_verification: false`）、未知の発行者・ホスト名不一致はいずれも `Err` で拒否された（構成 R・N′）。
- 設定の不整合（feature 無効のプロバイダ指定・`PlatformVerifier` の feature 不足）は `Err` ではなく **panic** になる。facade から使う場合は、設定を固定し panic 経路に入らないことをテストで担保する必要がある。

### 15.5 (d) ストリーミング受信・`If-None-Match`／304・`>=400` の挙動

平文 HTTP のローカルサーバに対する観測（プロトコル挙動の確認であり TLS の確認ではない）。結果は 3 構成で同一だった。

| 観点 | 操作 | 観測結果 |
|---|---|---|
| 既定値 | `Agent::new_with_defaults().config()` | `max_redirects=10`・`max_redirects_will_error=true`・`https_only=false`・`http_status_as_error=true`。タイムアウトは `await_100: Some(1s)` 以外すべて `None`（接続・受信とも既定では無期限） |
| `Content-Length`／`ETag` の取得 | `Content-Length` 付き 200（65536 バイト） | `Ok` 200。`ETag` ヘッダ取得可。`body().content_length()` は `Some(65536)` |
| ストリーミング受信 | `body_mut().as_reader()` を 8192 バイトのバッファで `read` | 8 回の `read` で 65536 バイト（逐次読み出しでき、サイズ計数・ハッシュ計算を挟める） |
| チャンク応答 | サーバが 1000 バイト×5 チャンクを 0.2 秒間隔で送信 | `content_length()` は `None`。最初の `read` は 0 ms、全体は約 1000 ms で 5 回の `read`・計 5000 バイト（全体の到着を待たずに逐次受信している） |
| サイズ上限 | `into_body().into_with_config().limit(1000).reader()` で 65536 バイトの応答を読む | 1000 バイト読んだ後 `Err`（`BodyExceedsLimit(1000)`） |
| `If-None-Match` 一致 | `.header("If-None-Match", <ETag>)` | **`Ok` 304**（既定の `http_status_as_error=true` のままでもエラーにならない）。`ETag` 取得可・ボディ 0 バイト |
| `If-None-Match` 不一致 | 別の値を付与 | `Ok` 200・ボディ全量 |
| `>=400`（既定） | 404・500 | `Err(StatusCode(404))`・`Err(StatusCode(500))` |
| `>=400`（`http_status_as_error(false)`） | 404 | `Ok` 404（ボディ 9 バイト取得可） |
| リダイレクト（既定） | 302 → 同一サーバの別パス | **自動追従して `Ok` 200**（既定の `max_redirects` は 10） |
| リダイレクト（`max_redirects(0)`） | 同上 | `Ok` 302（追従しない。`max_redirects_will_error` が `true`／`false` のどちらでも `Ok` 302 で、`Location` ヘッダを取得できた） |
| `https_only(true)` | `http://` の URL | `Err(RequireHttpsOnly(<url>))` |
| 受信タイムアウト | `timeout_recv_body(Some(500ms))`、サーバがボディ途中で 3 秒停止 | 5 バイト受信後、約 500 ms で `read` が `Err`（`Timeout(RecvBody)`） |

§14.3 (b) の記述との差分（実測で判明した点）:

- §14.3 の「`max_redirects` の既定が 0（自動追従しない）」は実測と一致しなかった。**既定は 10 で自動追従する**。各ホップで `https` 限定を検証するには、`max_redirects(0)` を明示して `Location` を手動で追うか、`https_only(true)` を併用する必要がある（`https_only(true)` がリダイレクト先にも効くかは未検証。§15.7）。
- タイムアウトは既定で無効のため、§5・§6 の要件を満たすには明示設定が要る。
- 304 は既定設定のまま `Ok` で受け取れ、`>=400` だけが `Err(StatusCode)` になる。304 判別のために `http_status_as_error(false)` へ切り替える必要はなかった。

### 15.6 (e) OpenSSL 開発パッケージの有無と CI のクロスビルド

リポジトリ内のファイルから確認できた事実:

- `Dockerfile`（ベース `rust:1.88-slim-bookworm`）の `apt-get install` は `build-essential`・`pkg-config`・`git`・`make`・`curl`・`ca-certificates`・`python3` で、**`libssl-dev` は含まれない**。`cmake` も含まれない。
- `.github/workflows/` 配下に `apt`／`apt-get` によるパッケージ導入・`openssl`／`libssl` の記述・`container:` 指定はない。CI は `ubuntu-latest` の初期状態に依存する。**`ubuntu-latest` のイメージに OpenSSL 開発パッケージが入っているかは、本リポジトリのファイルからは確認できず未検証**である。
- `ci.yml` の `cargo build (linux / aarch64-apple-darwin)` ジョブは、Linux ホストから `cargo build --workspace --locked --target aarch64-apple-darwin --lib` を実行する。

クロス lib ビルドの実測（今回のホスト。macOS 向けの C クロスコンパイラは未導入）:

| 構成 | `cargo build --locked --target aarch64-apple-darwin --lib` |
|---|---|
| N | 成功（macOS 側は `openssl-sys` ではなく `security-framework` 系で、推移依存数 29〈`-e normal,build`〉） |
| R | **失敗**: `error: failed to run custom build command for ring v0.17.14`。`cc: error: unrecognized command-line option '-arch'`・`'-mmacosx-version-min=11.0'`（ホストの `cc` が macOS 向けオプションを受け付けない） |
| N′ | 成功（推移依存数 30） |

構成 R を facade の無条件依存（§14.4 の案 A）にすると、macOS 向け C クロスコンパイラの無いホストでは同ジョブと同じ形のビルドが `ring` の build script で失敗する（今回のホストでの実測）。`ubuntu-latest` 上での成否は未検証である。

### 15.7 未検証で残った点

- 公開インターネット上のサーバに対する TLS の実通信（今回の HTTPS はローカル CA・ローカルサーバのみ）。`RootCerts::PlatformVerifier`（OS ストア）で実在の証明書チェーンが検証に通ること、`webpki-roots` での同確認。
- OpenSSL 開発パッケージが**無い**環境での構成 N・N′ のビルド失敗の様子（今回のホストは導入済み。開発コンテナ・`ubuntu-latest` での実ビルドは未実施）。
- `ubuntu-latest` に OpenSSL 開発パッケージ・macOS 向け C クロスコンパイラが入っているか。
- `aarch64-unknown-linux-gnu`（DGX Spark GB10）・macOS 実機・Windows でのビルドと実行。`aarch64-apple-darwin` は lib のクロスビルドのみで、リンク・実行は未検証。
- `ureq` の feature 定義（`native-tls-no-default` 単独で native-tls の実装が有効にならないこと）が上流の意図した仕様か不具合か、および `3.4.2` 以外の版での挙動。crates.io の最新版は実測時点で `3.4.2` だった。
- `minreq`（§14.3 の次点）の実ビルドと API 挙動。`aws-lc-sys`（`cmake` 必須）を含む構成のビルド。
- `https_only(true)` がリダイレクト先のホップにも適用されるか、`Location` が `http://` のときの挙動。接続タイムアウト（`timeout_connect`）・名前解決タイムアウトの実挙動。
- gzip 等の圧縮応答（今回の構成は圧縮 feature を有効にしていない）、プロキシ環境変数の扱い、接続プールの挙動。
- MSRV での実ビルド（今回は rustc 1.98.1 のみ。解決集合内で宣言された `rust-version` の最大は 3 構成とも 1.85）。

### 15.8 判断材料としての要約（採否は未決定）

- §14.3 の推奨案（`ureq =3.4.2`・`native-tls-no-default`）は、ビルド・`deny`（advisories／licenses）は通るが、**HTTPS が成立しないことを実測した**。推奨案はこのままでは採用できる状態にない。
- 今回 HTTPS の成立を実測できた構成は R（`rustls`）と N′（`native-tls`）で、どちらも現行 allow 外のライセンスを含む（R: `BSD-3-Clause`・`CDLA-Permissive-2.0`、N′: `CDLA-Permissive-2.0`）。allow の追加はユーザー承認事項のままである。
- システム要件は、N′ がビルド時・実行時とも OpenSSL（Linux）、R が C コンパイラのみで、R は Linux → `aarch64-apple-darwin` のクロス lib ビルドが今回のホストでは失敗した。
- 上記を受けて、§14.3 の推奨は取り下げて承認対象から外し、§14.5-1・§14.6 は再評価待ちとした（各節の冒頭に明記）。**新しい推奨は本 doc では定めない**。構成の選び直しはユーザー判断である。

**再評価で決める論点**（選択肢と、それぞれの未解決条件の列挙。並び順は優先度を意味しない。実測値は各参照先を正とする）:

| 選択肢 | 実測で分かっていること | 決める前に残る条件 |
|---|---|---|
| `ureq =3.4.2`・feature `rustls`（構成 R） | ローカル CA を明示した HTTPS が成立（§15.4）。ホストのビルドは C コンパイラのみで成功、`cmake`・OpenSSL は不要（§15.3）。`deny advisories` は ok（§15.3） | allow に `BSD-3-Clause`（`subtle`）と `CDLA-Permissive-2.0`（`webpki-roots`）の追加承認が要る（§15.3）。Linux → `aarch64-apple-darwin` のクロス lib ビルドが `ring` の build script で失敗した（今回のホスト。`ubuntu-latest` では未検証。§15.6）。ルート証明書源は同梱の `webpki-roots` で、OS ストアを使うには feature `platform-verifier` が別途要る（その構成は未ビルド・未実測。§15.4） |
| `ureq =3.4.2`・feature `native-tls`（構成 N′） | `provider(NativeTls)` を明示し、ローカル CA を指定した HTTPS が成立（§15.4）。Linux → `aarch64-apple-darwin` のクロス lib ビルドは成功（§15.6）。`deny advisories` は ok（§15.3） | allow に `CDLA-Permissive-2.0`（`webpki-root-certs`）の追加承認が要る（§15.3）。Linux ではビルド時に OpenSSL 開発パッケージ、実行時に `libssl.so.3`／`libcrypto.so.3` が要る（§15.3）。`Dockerfile` に `libssl-dev` が無く、`ubuntu-latest` の導入状況と、開発パッケージが無い環境でのビルドは未検証（§15.6・§15.7）。プロバイダは自動選択されず明示が要る（§15.4） |
| `ureq` 以外（`minreq =3.0.0` の `https-native-tls`／`https-openssl`） | §13 のライセンス・推移依存数の実測のみ（現行 allow で `licenses` は ok。§13.2）。直接ライセンスは ISC（§13.3） | 実ビルド・HTTPS の成立・API 挙動（§5・§6 の要件）はいずれも未実測（§15.7）。`ureq` ⑥ と同じく「`licenses` ok」だけでは HTTPS の成立を意味しないため、同等の実測が要る |
| `ureq` の別の版・別の feature 組合せ | HTTPS の成立は実測なし（`3.4.2` の ①・③ はライセンス・推移依存数のみ実測。いずれも allow 外を含む。§13.2） | `native-tls-no-default` 単独で実装が有効にならないのが上流の意図した仕様かどうか、`3.4.2` 以外の版での挙動は未検証（§15.7） |

どの選択肢にも共通して残る条件:

- 公開インターネット上のサーバに対する TLS の実通信、OS の証明書ストアによる実在の証明書チェーンの検証は未検証（§15.7）。
- `aarch64-unknown-linux-gnu`（DGX Spark GB10）・macOS 実機でのビルドと実行は未検証（§15.7）。
- TLS 設定の不整合は `Err` ではなく panic になる（§15.4）。リダイレクトの既定は 10 回の自動追従、タイムアウトの既定は無期限である（§15.5）。どの構成でも、§5・§6 の要件を満たす明示設定と、そのテストでの担保が要る。
- 構成が決まった後に、§14.4（配置案と `cfg` 範囲）・§14.5-1（区分の起案）・§14.6（承認チェックリスト）をその構成で書き直す。§14.5-2 の dirfd 経路（B-1／B-2）は構成に依存しない。

## 16. TLS 構成候補の再実測と推奨（2026-10-16・#3122）

### 16.1 位置づけ

- **本節は推奨までを書く。依存の承認は取得していない**。最終承認は #3123（承認依頼）で得る。deps-policy・`deny.toml`・`Cargo.toml`／`Cargo.lock` は変更しない。`docs/license-matrix.md` への正式反映は #3124 の担当で、本節 §16.10 は追記案である。
- 対象は §13・§15 にすでに載っている候補（`ureq =3.4.2`・`minreq =3.0.0`）だけである。新しいクレートは選んでいない。
- 実測は 2026-10-16 に使い捨てプロジェクト（本体 workspace の外・空の `[workspace]`・構成ごとに別ディレクトリ・`=x.y.z` 完全固定・`default-features = false`）で行った。実測後に削除した（§16.11）。
- 実測していない事項は「未検証」と書く（§16.7）。数値は §16.2 の環境で実測した値で、版・環境が変われば変わる。
- 本実測にはサブエージェントへの委譲手段が無かったため、上流仕様の調査（`reference-researcher`）・レビュー（`reviewer`／`security-auditor`）は実行していない。上流の記述はローカルの crate ソース（`~/.cargo/registry`）の読み取りに限った。上流の issue・CHANGELOG・docs.rs は参照していない（未確認）。

### 16.2 実測環境と手順

| 項目 | 値 |
|---|---|
| 実測日 | 2026-10-16 |
| 基準コミット | `f6ff55cb5033fb57f299be464a00a73fcf0477ec`（`origin/main`） |
| ホスト | Linux 7.0.0-34-generic x86_64・Intel Core i7-13700K・gcc 15.2.0・`libssl-dev` 3.5.5-1ubuntu3.7・OpenSSL 3.5.5・`cmake` なし・`clang` なし・perl／make／pkg-config／strace あり。macOS 向け C クロスコンパイラなし。rustc 1.98.1（48a229cea・2026-09-01）・cargo 1.98.1・cargo-deny 0.19.8 |
| advisory-db | cargo-deny が取得した RustSec advisory-db の先頭コミット `479c888c`（2026-10-16T08:29:44+02:00） |
| 開発コンテナ相当 | リポジトリの `Dockerfile`（変更なし）を `docker build` した `rust:1.88-slim-bookworm` ベースのイメージ（Debian 12）。既定の `cargo` は 1.88.0（rustc 6b00bc388）、`+stable` は 1.99.1（rustc 58cf8f9c3・2026-10-13。イメージビルド時に導入された版）。`libssl-dev`・`cmake`・`clang` なし、perl・make・pkg-config あり。プローブは `/work` の外に置いたため `rust-toolchain.toml` は効いていない |
| 最小 Ubuntu | `ubuntu:24.04`（24.04.5 LTS・gcc 13）に `build-essential`・`curl`・`ca-certificates`・`pkg-config` だけを入れ、rustup の stable（1.99.1）で測った（`libssl-dev`・`cmake`・`clang` なし） |
| runner 相当の Ubuntu | 同 Ubuntu 24.04.5 に、runner-images の README が記載するパッケージのうち `libssl-dev`（3.0.13-0ubuntu3.16）・`cmake`・`clang` を足した環境。**実ランナーではなく、README の記載に合わせて作った環境である** |
| `ubuntu-latest` の実体（文書上の記載） | main の CI run 38866626169（2026-10-16）の「Set up job」ログ: `Image: ubuntu-24.04`・`Version: 20261012.334.1`。同版の runner-images README（タグ `ubuntu24/20261012.334`・タグ参照 SHA `7e7dc1bcc0ef464e5e315a2b4c33609590c6cffe`）の記載: Kernel 6.17.0-1022-azure・gcc 13.2.0・make 4.3・pkg-config 1.8.1・Perl 5.38.2・CMake 3.31.6（表では 3.31.5／4.1.2）・Clang 16.0.6／17.0.6／18.1.3・OpenSSL 3.0.13-0ubuntu3.16・**`libssl-dev` 3.0.13-0ubuntu3.16**。同 README の告知: 「`ubuntu-latest` ラベルは 2026 年 11 月に Ubuntu 26.04 を使う」（actions/runner-images#14748。本文は未参照で、README 上の見出しのみ確認）。**実ランナーでの実行は未検証**（`.github/workflows/**` を変えずに実ランナーを使う手段がないため） |
| クロス（macOS） | ホストで `cargo build --locked --target aarch64-apple-darwin --lib`（CI の `cargo build (linux / aarch64-apple-darwin)` ジョブと同じ形）。加えて Debian 12 コンテナに `clang` 14.0.6 を入れ `CC_aarch64_apple_darwin=clang` でも試した |
| クロス（aarch64 Linux） | Debian 12 コンテナ（rustc 1.88.0）に `gcc-aarch64-linux-gnu` 12.2.0・`libssl-dev:arm64` 3.0.22-1~deb12u1 を入れ、`--lib` のビルドのみ（リンク・実行はしていない）。ホストの binfmt 登録は行っていない |
| 構成（固定版） | `ureq =3.4.2`・`minreq =3.0.0`。解決版: `rustls 0.23.46`・`webpki-roots 1.0.9`・`ring 0.17.14`・`subtle 2.6.1`・`native-tls 0.2.18`・`openssl 0.10.81`・`openssl-sys 0.9.117`・`openssl-probe 0.2.1`・`webpki-root-certs 1.0.9`・`rustls-platform-verifier 0.7.1`（ureq。minreq の構成 M5 は 0.6.2）・`rustls-native-certs 0.8.5`・`aws-lc-sys 0.46.0`・`openssl-src 300.6.1+3.6.3`（同梱 OpenSSL 3.6.3） |

構成の ID（`default-features = false`。ureq は `features = [...]`）:

| ID | 構成 |
|---|---|
| R | `ureq` `["rustls"]` |
| R+PV | `ureq` `["rustls", "platform-verifier"]` |
| N | `ureq` `["native-tls-no-default"]`（§15 の構成 N。再確認のみ） |
| N′ | `ureq` `["native-tls"]` |
| N′+V | `ureq` `["native-tls", "vendored"]` |
| M1 | `minreq` `["https-native-tls"]` |
| M2 | `minreq` `["https-openssl"]`（`openssl/vendored`） |
| M3 | `minreq` `["https-openssl-probe"]` |
| M4 | `minreq` `["https-rustls"]`（`webpki-roots`。`aws-lc-rs`／`aws-lc-sys`） |
| M5 | `minreq` `["https-rustls-probe"]`（`rustls-platform-verifier`） |
| 参考 U1・U3 | `ureq` の既定（`rustls`＋`gzip`）・`["rustls-no-provider", "platform-verifier"]`（暗号プロバイダを別に足す必要があり単体では TLS が成立しない。ライセンスと依存数のみ測り、HTTPS・ビルド挙動は対象外） |

手順は §13.1・§15.2 と同じ数え方である。`cargo tree --locked -e normal`（および `-e normal,build`）を `--target {x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu, aarch64-apple-darwin}` で実行し、`--prefix none` の出力から「` (*)`」（重複表示）の行を除いて `sort -u` し、ルート自身を除いて数えた。`cargo metadata --locked --format-version 1 --filter-platform <t>` の resolve に含まれる全パッケージの `license` を集計し、`MPL`・`GPL`・`LGPL`・`AGPL`・`EPL`・`CDDL`・`CDLA`・`BSD-3`・`OpenSSL`・`AND`・`0BSD` を機械的に抜き出した。`cargo deny --locked check` は構成ごとに 2 回行った（§16.4）。HTTPS の挙動は 127.0.0.1 のローカルサーバ（Python 標準ライブラリ）とローカル CA（SAN は `DNS:localhost` のみ。システムの証明書ストアには入れず、`SSL_CERT_FILE` で注入）で、1 ケース 1 プロセスの最小 Rust プログラムを実行して測った。公開インターネットのサーバとの通信は行っていない。

**§15.3・§13.4 の値との整合**: 今回の推移依存数は R 22／25・N 29／34・N′ 30／35（x86_64-linux。`-e normal`／`-e normal,build`）で、§15.3 と一致した。M1 21（`-e normal,build`。x86_64・aarch64 linux）／15（aarch64-darwin）・M2 19 も §13.4 の値と一致した。解決版の変化（`rustls 0.23.45` → `0.23.46` など）による数の変化は観測されなかった。なお「` (*)`」行を除かずに数えると R 27・N 37・N′ 39 のように 2〜4 多くなる（`cargo tree --prefix none` が重複表示を別行にするため）。

### 16.3 構成別の比較表

推移依存数は `-e normal`／`-e normal,build`。順に x86_64-linux／aarch64-linux／aarch64-darwin。

| ID | 推移依存数（x86_64-linux／aarch64-linux／aarch64-darwin） | 直接ライセンス | 宣言 `rust-version` | HTTPS 成立（§16.6） |
|---|---|---|---|---|
| R | 22／25・22／25・22／25 | MIT OR Apache-2.0 | 1.85 | 成立（ローカル CA を明示指定。OS ストアは feature 不足で panic） |
| R+PV | 25／28・25／28・28／31 | MIT OR Apache-2.0 | 1.85 | 成立（`SSL_CERT_FILE` 経由の OS ストア検証・ローカル CA の明示指定とも `Ok`） |
| N | 29／34・29／34・29／29 | MIT OR Apache-2.0 | 1.85 | **不成立**（どのプロバイダ指定でも panic。§15.4 の再確認） |
| N′ | 30／35・30／35・30／30 | MIT OR Apache-2.0 | 1.85 | 成立（`NativeTls` の明示が要る。既定の `Rustls` は panic） |
| N′+V | 30／36・30／36・30／30 | MIT OR Apache-2.0 | 1.85 | 成立（N′ と同じ） |
| M1 | 16／21・16／21・15／15 | ISC | 1.63 | 成立（`SSL_CERT_FILE` 経由。CA の指定 API なし） |
| M2 | 13／19・13／19・13／19 | ISC | 1.63 | 成立（`SSL_CERT_FILE` 経由のみ。OS ストアは開かれない。§16.6） |
| M3 | 14／20・14／20・14／20 | ISC | 1.63 | 成立（`openssl-probe` が `/usr/lib/ssl/cert.pem`〈`/etc/ssl/certs/ca-certificates.crt` へのシンボリックリンク〉を開く） |
| M4 | 12／21・12／21・12／21 | ISC | 1.63 | ローカル CA を注入できず `UnknownIssuer`（`webpki-roots` のみ。環境変数は無視）。公開 CA は未検証 |
| M5 | 14／23・14／23・18／26 | ISC | 1.63 | 成立（`SSL_CERT_FILE` 経由） |
| 参考 U1 | 27／30・27／30・27／30 | MIT OR Apache-2.0 | 1.85 | 対象外 |
| 参考 U3 | 20／20・20／20・24／24 | MIT OR Apache-2.0 | 1.85 | 対象外 |

- `-e normal,build` の差分: N′+V は N′ に対して x86_64／aarch64 linux で `openssl-src` が +1 になる。aarch64-darwin では `openssl` を含む名前の crate が依存木に 0 個で、`vendored` は効かない。
- `cargo tree -d`（重複バージョン）は全構成で出力なし。
- 「宣言 `rust-version`」は `cargo info` が示すクレート自身の値である。解決集合内の最大値は集計していない。実ビルドは開発コンテナの rustc 1.88.0 で R・R+PV・N′+V・M2・M3・M4・M5 が成功した（N′・M1 は開発コンテナで `libssl-dev` が無く失敗するため 1.88.0 では未検証）。

### 16.4 ライセンスと `cargo deny` の結果

**コピーレフトの混入**: 全構成×3 ターゲットで、`MPL`・`GPL`・`LGPL`・`AGPL`・`EPL`・`CDDL` を含む宣言ライセンスは検出されなかった。次の宣言ライセンスが allow 外の候補になる（`cargo metadata` の実測）:

| 宣言ライセンス | クレート | 現れる構成 |
|---|---|---|
| `BSD-3-Clause` | `subtle 2.6.1` | R・R+PV・M4・M5・U1・U3 |
| `CDLA-Permissive-2.0` | `webpki-roots 1.0.9` | R・R+PV・M4・U1 |
| `CDLA-Permissive-2.0` | `webpki-root-certs 1.0.9` | N′・N′+V（native-tls feature が必ず含める）。R+PV・M5・U3 は全ターゲット指定の `cargo deny` でのみ現れる（上記 3 ターゲットの依存木には無い） |
| 複合式（`ISC AND (Apache-2.0 OR ISC) AND Apache-2.0 AND MIT AND BSD-3-Clause AND ...`） | `aws-lc-sys 0.46.0` | M4・M5（`BSD-3-Clause` を含む） |
| `ISC`（直接） | `minreq 3.0.0` | M1〜M5（既存 allow 内。ただし deps-policy の適合基準「MIT OR Apache-2.0 系」の記述から外れるため、例外の記録が要る） |
| `Apache-2.0 AND ISC` | `ring 0.17.14` | R・R+PV・U1・U3（allow 内） |
| `MIT/Apache-2.0`（表記ゆれ） | `openssl-src 300.6.1+3.6.3` など | N′+V・M2・M3 ほか（allow 内。同梱ソースのライセンスは §16.10） |

**`cargo deny --locked check advisories bans licenses sources`**（全ターゲット。構成ごとに 2 回）:

| 実行 | 設定 | 結果 |
|---|---|---|
| (i) | ルート `deny.toml` と同じ allow（8 種） | **ok**: N・M1・M2・M3。**licenses FAILED**: R（`subtle`〈BSD-3-Clause〉・`webpki-roots`〈CDLA-Permissive-2.0〉）・R+PV（左 2 件＋ `webpki-root-certs`〈CDLA-Permissive-2.0〉）・N′・N′+V（`webpki-root-certs`）・M4・M5（`subtle`・`aws-lc-sys` の複合式・`webpki-roots`／`webpki-root-certs`）・U1・U3 |
| (ii) | (i) に `BSD-3-Clause`・`CDLA-Permissive-2.0` を足した使い捨ての設定 | 全構成で **advisories ok・bans ok・licenses ok・sources ok**。構成ごとの追加は上表のとおり。N′・N′+V は `CDLA-Permissive-2.0` のみで足りる（`BSD-3-Clause` は不要）。M1・M2・M3・N は追加なし |

- advisories は全構成で ok（該当 advisory なし。advisory-db `479c888c`）。重複バージョンの警告（`syn`・`windows-sys`）は R+PV・M5・U3 のみ。
- 使い捨ての設定は使い捨てディレクトリにだけ置いた。本体の `deny.toml` は変更していない。
- **「`licenses` が ok」は HTTPS の成立を意味しない**（§15 の教訓）。N は (i) で ok だが HTTPS は成立しない。

### 16.5 システム要件とビルド結果

`cargo build --locked -vv`（debug）の成否。「失敗」の理由は各行の後の注を参照。

| ID | ホスト（libssl-dev あり） | 開発コンテナ（`libssl-dev` なし。1.88.0／1.99.1） | 最小 Ubuntu 24.04（`libssl-dev` なし） | runner 相当（`libssl-dev`・`cmake`・`clang` あり） | `aarch64-apple-darwin`（ホスト・gcc） | 同（Debian 12＋`clang` 14） | `aarch64-unknown-linux-gnu`（Debian 12＋`libssl-dev:arm64`） |
|---|---|---|---|---|---|---|---|
| R | 成功（7 s） | 成功 | 成功 | 成功 | **失敗**（`ring`） | **失敗**（`ring`） | 成功 |
| R+PV | 成功 | 成功 | 成功 | 成功 | **失敗**（`ring`） | **失敗**（`ring`） | 成功 |
| N | 成功 | **失敗**（`openssl-sys`） | **失敗**（`openssl-sys`） | 成功 | 成功 | 成功 | 成功 |
| N′ | 成功 | **失敗**（`openssl-sys`） | **失敗**（`openssl-sys`） | 成功 | 成功 | 成功 | 成功 |
| N′+V | 成功（29 s） | 成功（42 s） | 成功（22 s） | 成功（36 s） | 成功 | 成功 | 成功 |
| M1 | 成功 | **失敗**（`openssl-sys`） | **失敗**（`openssl-sys`） | 成功 | 成功 | 成功 | 成功 |
| M2 | 成功（21 s） | 成功 | 成功 | 成功 | **失敗**（`openssl-sys`） | **失敗**（`openssl-sys` のビルド中。根本原因の行は未抽出） | 成功 |
| M3 | 成功（41 s） | 成功 | 成功 | 成功 | **失敗**（`openssl-sys`） | **失敗**（同上） | 成功 |
| M4 | 成功（22 s） | 成功 | 成功 | 成功 | **失敗**（`aws-lc-sys`） | **失敗**（`aws-lc-sys`） | 成功 |
| M5 | 成功（32 s） | 成功 | 成功 | 成功 | **失敗**（`aws-lc-sys`） | **失敗**（`aws-lc-sys`） | 成功 |

- 開発コンテナの列: 1.99.1 では表の 10 構成すべてを測った。1.88.0（イメージ既定）では R・R+PV・N′+V・M2・M3・M4・M5 を測り、いずれも 1.99.1 と成否が同じだった（N・N′・M1 の 1.88.0 は未検証）。
- **`libssl-dev` の要否**: `openssl-sys` を動的リンクする N・N′・M1 は、`libssl-dev` が無い環境（`Dockerfile` そのまま・最小 Ubuntu）で `Could not find directory of OpenSSL installation, and this -sys crate cannot proceed without this knowledge ... Make sure you also have the development packages of openssl installed.` となり失敗した。`vendored`（N′+V・M2・M3）はソースから OpenSSL をビルドするため `libssl-dev` が無くても成功した。`ubuntu-latest` の実ランナーに `libssl-dev` があるかは文書上の記載（§16.2）で、実測は未検証。
- **`cmake` の要否**: `aws-lc-sys`（M4・M5）は、`cmake` が無いホスト・開発コンテナ・最小 Ubuntu でもビルドに成功した（`cmake` クレートはコンパイルされるが、`cmake` コマンドが無い環境で成功した理由は未調査）。**§13.2 と §15.7 の「`aws-lc-sys`（`cmake` 必須）」という記述は、今回の `aws-lc-sys 0.46.0` では観測されなかった**（§13.2 は未ビルド、§15.7 は未測定の項目だった）。
- **perl・make の要否**: `vendored`（N′+V・M2・M3）は `openssl-src` が perl で Configure し make でビルドする。開発コンテナ・最小 Ubuntu・ホストはいずれも perl と make があった環境で成功した。perl または make が無い環境は未検証。ビルド時間は debug で 21〜42 秒。
- **`aarch64-apple-darwin` のクロス lib ビルド（CI の同名ジョブと同じ形）**: `ring`（R・R+PV）・`aws-lc-sys`（M4・M5）・ソースビルドの OpenSSL（`https-openssl` が全ターゲットで `openssl/vendored` を有効にする M2・M3）は、`cc` が macOS 向けオプションを受け付けず失敗した（ホスト: `cc: error: unrecognized command-line option '-arch'`・`'-mmacosx-version-min=11.0'`）。`clang` を入れて `CC_aarch64_apple_darwin=clang` を指定しても、`ring`・`aws-lc-sys` は macOS SDK のヘッダ（`bits/libc-header-start.h`・`CoreServices/CoreServices.h`）が無く失敗した。**`ubuntu-latest` の実ランナーでの成否は未検証**（README が記載するのは gcc／clang で、macOS SDK の記載は確認できなかった。実ランナーで同じ失敗になるかは未検証）。成功したのは N・N′・N′+V・M1 で、macOS 側が `security-framework` 系（C のビルドなし・OpenSSL なし）になるためである。N′+V の `vendored` は macOS では効かない（依存木に `openssl` 系 crate が 0 個。実測）。
- **`aarch64-unknown-linux-gnu`**: 全構成で `--lib` のビルドは成功した。リンク形態: N・N′・M1 は `crypto`・`ssl` を動的、N′+V・M2・M3 は `static=crypto`・`static=ssl`、R は `static=ring_core_0_17_14_`、M4 は `static=aws_lc_0_46_0_crypto`（ビルド時のリンク指示。実機での実行は未検証）。
- **生成バイナリの `ldd`**（ホスト）: N′・M1 は `libssl.so.3`・`libcrypto.so.3` に動的リンクする（実行環境にも OpenSSL 3 の共有ライブラリが要る）。N′+V・M2・M3・R・R+PV・M4・M5 は `libssl`／`libcrypto` にリンクしない。N は `libssl` にリンクされない（TLS 実装が組み込まれていない。§15.4）。

### 16.6 HTTPS と §5・§6 の挙動

ホスト（x86_64 Linux）・ローカル CA・`https://localhost:<port>`。`Ok`／`Err`／panic のどれになったかを書く。

**ureq（`TlsConfig::builder()` の設定ごと）**

| 設定 | R | R+PV | N | N′ | N′+V |
|---|---|---|---|---|---|
| 既定のエージェント（プロバイダ `Rustls`＋`WebPki`） | `Err`（`UnknownIssuer`） | `Err`（`UnknownIssuer`） | panic（`provider is Rustls but feature is not enabled: rustls`） | 左と同じ panic | 左と同じ panic |
| `provider(Rustls)`＋`PlatformVerifier` | panic（`Rustls + PlatformVerifier requires feature: platform-verifier`） | `Err`（`UnknownIssuer`）。`SSL_CERT_FILE=<CA>` を付けると **`Ok` 200** | panic（`…Rustls but feature is not enabled`） | 同左 | 同左 |
| `provider(Rustls)`＋`Specific(CA)` | **`Ok` 200** | **`Ok` 200** | panic | panic | panic |
| `provider(NativeTls)`（ルート既定） | panic（`provider is NativeTls but feature is not enabled: native-tls`） | 同左 | 同左 | `Err`（`unable to get local issuer certificate`） | 同左 |
| `provider(NativeTls)`＋`PlatformVerifier` | panic | panic | panic | `Err`（同上）。`SSL_CERT_FILE=<CA>` を付けると **`Ok` 200** | 同左 |
| `provider(NativeTls)`＋`Specific(CA)` | panic | panic | panic | **`Ok` 200** | **`Ok` 200** |
| 上と同じ有効設定で `https://127.0.0.1`（SAN 不一致） | `Err`（`certificate not valid for name "127.0.0.1"`） | 同左 | （成立しない） | `Err`（`IP address mismatch`） | 同左 |
| 未知の発行者（ローカル CA を信頼しない設定） | `Err`（`UnknownIssuer`） | 同左 | （成立しない） | `Err`（`unable to get local issuer certificate`） | 同左 |

- **N（`native-tls-no-default` 単独）は再測しても HTTPS が成立しない**（どの設定でも panic）。上流の文書（`ureq-3.4.2/README.md:166`・`src/lib.rs:170`）は「`native-tls-no-default`: Enables native-tls, but does not enable webpki」と書く。一方 `Cargo.toml:94-98` の同 feature は `dep:native-tls`・`dep:der`・`_tls` を有効にするだけで、feature `native-tls` 自体は有効にしない。コード側は `#[cfg(feature = "native-tls")]`（`src/lib.rs:783`・`835`・`1265` ほか）で括られている。**文書の記述と実装が食い違っていることまでは確認した。上流が不具合として扱っているかどうかは未確認**（issue・CHANGELOG は参照していない）。
- `ureq` は feature で有効にしたプロバイダに関係なく**既定のプロバイダが `Rustls`** である（§15.4）。N′・N′+V で既定のまま使うと panic するため、`TlsProvider::NativeTls` の明示と、その設定を固定するテストが要る。設定の不整合は `Err` ではなく panic になる（§15.4 の再確認）。
- **`vendored` の効き方**: N′+V の挙動は N′ と同じで（上表）、リンク形態だけが静的になる。macOS では効かない（依存木に `openssl` 系 crate が 0 個）。
- **OS 証明書ストア**: N′・N′+V は、`SSL_CERT_FILE` なしでも `/etc/ssl/certs/ca-certificates.crt` を開いた（`strace -f -e trace=openat`。N′+V でも同じ）。これは `native-tls 0.2.18` の `src/imp/openssl.rs:19`（`openssl_probe::probe` を `LazyLock` で保持）・`:290-297`（`cert_file`／`cert_dir` を `load_verify_locations` へ渡す）による。**`vendored` でも `native-tls` 経由なので OS ストアを参照する**。R+PV は `rustls-native-certs` 経由で `SSL_CERT_FILE` を参照した。公開 CA のチェーン検証は未検証（§16.7）。

**minreq（M1〜M5。TLS 設定の API は無く、挙動はクレート feature と環境変数で決まる）**

| ケース | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| `SSL_CERT_FILE` なし | `Err`（`certificate verify failed`） | `Err` | `Err` | `Err`（`UnknownIssuer`） | `Err`（`UnknownIssuer`） |
| `SSL_CERT_FILE=<CA>` | **`Ok` 200** | **`Ok` 200** | **`Ok` 200** | `Err`（`UnknownIssuer`。環境変数は無視） | **`Ok` 200** |
| `https://127.0.0.1`（SAN 不一致・CA 注入あり） | `Err` | `Err` | `Err` | `Err` | `Err`（`NotValidForNameContext`） |
| OS 証明書ストアの参照（`strace`、`SSL_CERT_FILE` なし） | `/etc/ssl/certs/ca-certificates.crt` を開いた | **開かない**（`/usr/local/ssl/openssl.cnf`・`/usr/local/ssl/cert.pem` が ENOENT のみ） | `/usr/lib/ssl/cert.pem`（`/etc/ssl/certs/ca-certificates.crt` へのシンボリックリンク）を開いた | 未測定（`webpki-roots` のみ） | 未測定 |

- **M2（`https-openssl`）は vendored OpenSSL の既定ディレクトリ `/usr/local/ssl` しか見ない**ため、Debian／Ubuntu の `/etc/ssl/certs` は `SSL_CERT_FILE` を渡さない限り使われない。`https-openssl-probe`（M3）はこの穴を `openssl-probe` で埋める。M2 は macOS の cross 失敗（§16.5）と合わせて、単独では候補にならない。
- minreq は証明書検証を無効化する API も、カスタム CA を渡す API も持たない（`src/` に TLS 設定の公開 API なし）。テストで CA を差し込むには環境変数が要り、macOS の `security-framework` 経路（M1）では `SSL_CERT_FILE` が効かないため、TLS の結合テストは Linux 限定になる（macOS では効くかどうか未検証）。

**ストリーミング・条件付き取得・ステータス・リダイレクト・タイムアウト**（ureq は R・R+PV・N′・N′+V で同一。minreq は M1〜M3・M5 で同一。いずれも HTTPS で測った）

| 観点 | ureq（`Specific(CA)` で有効な設定） | minreq |
|---|---|---|
| ストリーミング受信・`Content-Length`／`ETag` | `as_reader()` で 8 回の `read`・65536 バイト。`content_length()`＝`Some(65536)`・`ETag` 取得可 | `send_lazy()` が返す `ResponseLazy`（`Read`）で逐次受信（8〜9 回の `read`）。ヘッダは `(String, String)` の列で取得 |
| サイズ上限 | `with_config().limit(1000)`: 1000 バイト読んだ後 `Err`（`the response body is larger than request limit: 1000`） | API なし（`Read` の側でバイト数を数えて打ち切る） |
| `If-None-Match` 一致 | `Ok` 304（既定の `http_status_as_error` のまま）・`ETag` 取得可・ボディ 0 | `Ok` 304 |
| `>=400` | 既定 `Err(http status: 404)`。`http_status_as_error(false)` で `Ok` 404（ボディ 9 バイト取得可） | 常に `Ok`（`status_code` を見る。404 で `Ok` 404） |
| リダイレクト（既定） | **`https` → `http://` へ自動追従して `Ok` 200**（下りリダイレクトを既定で許す） | 同左（`Ok` 200） |
| `https_only(true)` | **後続ホップにも効く**: `https` → `http://` の 302 は `Err(configured for https only: http://127.0.0.1:<port>/ok)`（§15.7 の未検証を解消） | API なし |
| 手動追従 | `max_redirects(0)` → `Ok` 302 と `Location` を取得できる（§15.5 の再確認） | `with_follow_redirects(false)` → `Ok` 302 と `Location`。`with_max_redirects(0)` は `Err(TooManyRedirections)` で手動追従には使えない |
| `https` → `https` のリダイレクト | `Ok` 200 | `Ok` 200 |
| 受信タイムアウト | `timeout_recv_body(500ms)`: ボディ途中で停止したケースで 5 バイト受信後に約 550 ms で `Err`（`timeout: receive body`） | `with_timeout(1)`（秒単位・整数）: 停止ケースで約 1.0 s で `Err`（`the timeout of the request was reached`） |
| **タイムアウトの種別** | **全体の期限であって無通信期限ではない**: 0.2 秒間隔でデータが届く 3 秒の応答に 500 ms を指定すると、データが途切れていないのに 300 バイト（約 550 ms）で打ち切られた（N′も同じ） | **同じく全体の期限**: 1 秒指定で 600 バイト（約 1.0 s）で打ち切られた |
| 接続タイムアウト | `timeout_connect(700ms)`: accept キューを満杯にした 127.0.0.1 への接続が 0.70 s で `Err(timeout: connect)`。**既定は無期限**（4 秒後も接続待ち） | 専用の設定なし。`with_timeout(1)` が 1.00 s で `Err`。**既定は無期限** |

- **設計上の入力（#3123・実装 Issue 向け）**: どちらの crate でも、受信タイムアウトは「リクエスト全体（またはボディ全体）の期限」で、無通信（アイドル）期限ではない。数 GB の safetensors を取得する §5 の仕様では、固定の全体期限を短くすると正常な遅い転送を打ち切り、長くすると停止検知が遅れる。アイドル検知を満たす設計（例: 読み取りのたびに期限を更新するラッパー、または ureq の別の設定の検証）は未解決で、実装 Issue で扱う。`ureq` の `timeout_per_call`・`timeout_resolve` 等の挙動は未検証。
- 接続タイムアウトの測定は、外部アドレスへの送信・iptables・netns を使わず、accept キューを満杯にしたローカルの待受ソケットで行った。

### 16.7 未検証の点

- 公開インターネット上のサーバに対する TLS の実通信と、OS の証明書ストアによる実在の証明書チェーンの検証（今回の HTTPS はローカル CA・ローカルサーバのみ）。#3173 に委ねる。
- `ubuntu-latest` の**実ランナー**でのビルド（`libssl-dev`・`clang` の導入状況は README の記載。macOS 向けクロスの成否は同じ環境のコンテナで再現しただけ）。README は 2026 年 11 月に `ubuntu-latest` が Ubuntu 26.04 になると告知している。26.04 での `libssl-dev` の有無は未確認。
- macOS 実機・Windows・DGX Spark GB10 実機でのビルド・リンク・実行（`aarch64-apple-darwin` は lib のクロスビルドのみ。`aarch64-unknown-linux-gnu` も `--lib` のビルドのみで、リンクと実行は未検証）。#3173 に委ねる。
- macOS の `security-framework` 経路（M1・N′・N′+V）での証明書検証の実挙動、および macOS で `SSL_CERT_FILE` が効くか。
- perl または make が無い環境での `vendored` のビルド。
- `cmake` が無い環境で `aws-lc-sys` が成功した仕組み（今回は成功の事実のみ）。
- `native-tls-no-default` が上流の意図した仕様か不具合か（文書と実装の食い違いまでは確認。上流の issue・CHANGELOG・docs.rs は未参照）、`ureq` の `3.4.2` 以外の版での挙動。crates.io の最新安定版は今回も `3.4.2` だった（2026-10-16 に crates.io API で確認）。古い版は、推奨の根拠にならないため実測しなかった。
- アイドル（無通信）タイムアウトを満たす設定、`timeout_per_call`・`timeout_resolve`・gzip 等の圧縮応答・プロキシ環境変数・接続プール。
- 解決集合内の `rust-version` の最大値（宣言値は §16.3）。

### 16.8 判定軸と推奨

判定軸（8 軸）での比較。○＝今回の実測で満たす・×＝満たさない・△＝条件付き。

| 軸 | R | R+PV | N′ | **N′+V** | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|---|---|---|---|
| (1) HTTPS 成立 | ○ | ○ | ○ | ○ | ○ | △（OS ストア不可） | ○ | △（ローカル CA 不可） | ○ |
| (2) §5・§6 を満たす明示設定（panic を固定設定で避けられる・https 限定・上限・タイムアウト） | ○ | ○ | △（プロバイダ明示が要る） | △（同左） | △（`https_only`・サイズ上限・接続タイムアウト・CA 指定 API なし。手動追従で https 限定は担保可） | △ | △ | △ | △ |
| (3) allow への追加 | `BSD-3-Clause`・`CDLA-Permissive-2.0` | 同左 | `CDLA-Permissive-2.0` | `CDLA-Permissive-2.0` | なし（ISC 直接の例外記録は要る） | なし | なし | `BSD-3-Clause`・`CDLA-Permissive-2.0` | `BSD-3-Clause`（全ターゲット指定では `CDLA` も） |
| (4) Linux のビルド・実行要件 | C コンパイラのみ | 同左 | ビルド `libssl-dev`・実行 `libssl.so.3` | perl・make（ビルド時間 21〜42 s）。実行時依存なし | ビルド `libssl-dev`・実行 `libssl.so.3` | perl・make | perl・make | C コンパイラ（`cmake` なしで成功） | 同左 |
| (5) `aarch64-apple-darwin` クロス lib（CI ジョブと同形） | × | × | ○ | ○ | ○ | × | × | × | × |
| (6) `aarch64-unknown-linux-gnu` の lib ビルド | ○ | ○ | ○（`libssl-dev:arm64` 要） | ○ | ○（同左） | ○ | ○ | ○ | ○ |
| (7) advisories | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| (8) 推移依存数（x86_64-linux・`normal,build`）／宣言 MSRV | 25／1.85 | 28／1.85 | 35／1.85 | 36／1.85 | 21／1.63 | 19／1.63 | 20／1.63 | 21／1.63 | 23／1.63 |

**読み取り**:

- (5) が判別軸になる。**`ring` 系（R・R+PV）・`aws-lc-sys` 系（M4・M5）・ソースビルド OpenSSL を全ターゲットで有効にする `https-openssl` 系（M2・M3）は、既存の CI ジョブ `cargo build (linux / aarch64-apple-darwin)` と両立しない**（`clang` を足しても macOS SDK が無く失敗）。両立させるには CI に macOS SDK を足す変更が要るが、`.github/workflows/**` の変更を伴うため本イシューの範囲外である。配置を `cfg(target_os = "linux")` 限定にして macOS を除く方法はあるが、macOS で `download` が使えなくなる（§14.4 の `cfg` 範囲の判断になる）。
- 残るのは N′・N′+V・M1 の 3 つである。N′・M1 は Linux で `libssl-dev` を要し、`Dockerfile` に無いこと（開発コンテナで失敗。`Dockerfile` の変更が要る）、実行時にも `libssl.so.3` が要ることが障害になる。N′+V は、**今回測った全環境（ホスト・開発コンテナ・最小 Ubuntu・runner 相当・クロス 2 種）で環境の変更なしに成功した**唯一の構成である。
- N′+V の代償は、(a) allow に `CDLA-Permissive-2.0` を足す承認（`webpki-root-certs`。N′ と同じ。コピーレフトではない〈§13.3〉）、(b) perl・make とビルド時間 21〜42 s（debug）、(c) OpenSSL 本体（Apache-2.0。§16.10）が静的に成果物へ入ること、(d) OpenSSL の脆弱性対応が `openssl-src` の更新頼みになること（システム OpenSSL なら OS の更新で直る。N′ はこの点で有利）、(e) ureq が `Rustls` を既定プロバイダにするため `TlsProvider::NativeTls` の明示が要ること、である。

**推奨（1 つ）: `ureq =3.4.2`・`default-features = false`・`features = ["native-tls", "vendored"]`**（構成 N′+V。承認は未取得。承認は #3123）。

- 根拠: 判定軸 (1)・(5)・(6)・(7) を満たし、(4) の環境要件が最も少なく（`Dockerfile`・CI の変更なしで測った全環境で成功）、(2) は ureq の API（`https_only`・`limit`・`timeout_connect`・`Specific(CA)`・手動追従）で §5・§6 の要件を明示設定で満たせる。macOS の CI クロスジョブと両立する。
- 条件（承認時に確認する）: 上記 (a)〜(e) の受容。`TlsConfig` を固定し panic 経路をテストで閉じること。リダイレクトは `https_only(true)` で各ホップを `https` に限ること（実測で後続ホップに効いた）。受信タイムアウトが全体期限である点（§16.6）の設計判断。
- 次点: `minreq =3.0.0`・`https-native-tls`（M1）。allow の追加が不要な点（`ISC` の直接ライセンスの例外記録は要る）と推移依存数が最小な点が長所で、`libssl-dev` を `Dockerfile` に足すこと、API の不足（`https_only`・サイズ上限・接続タイムアウト・CA 指定が無い）が代償である。**ただし、allow の追加を避けること自体を最優先にするなら M1 が候補になる。この選択は #3123 での判断事項で、本実測だけでは決まらない**。
- 推奨できない構成: R・R+PV（CI クロスと両立しない）・N（HTTPS 不成立）・M2（OS ストア不可かつクロス失敗）・M3・M4・M5（クロス失敗）。

### 16.9 推奨構成を前提にした書き直し案（§14.4・§14.5-1・§14.6）

既存の §14.4・§14.5-1・§14.6 は書き換えず、新しい案として並べる。すべて未承認の案である。

**§14.4 の書き直し案（配置案と `cfg` 範囲。N′+V を前提）**

| 配置案 | N′+V との組合せ |
|---|---|
| A: `facade` の無条件依存 | 全利用者が perl・make と OpenSSL のソースビルド（debug で 21〜42 s）を負う（Linux のみ。macOS は `vendored` が効かず `security-framework` 系になるためビルドは軽い）。`libssl-dev` は不要 |
| B: cargo feature で opt-in | 本リポに optional 依存の前例なし。要件を download 利用者に限定できる |
| C: 非公開の別クレート | §4 の構造問題を再生産する（変わらず） |

`cfg` 範囲: `cfg(unix)` 限定を引き続き第一案とする（Windows は fail-closed 非対応。§6 (c)）。`cfg(unix)` は macOS を含むため、**macOS を含む場合の CI クロスジョブとの両立は N′・N′+V・M1 でのみ実測で確認済み**（§16.5）。Linux 限定にすると macOS で `download` が使えない。採否はユーザー承認事項。

**§14.5-1 の書き直し案（第 11 区分の起案行）**

| 区分 | クレート | 条件 |
|---|---|---|
| HTTP クライアント／TLS（第 11 区分） | `ureq`（`default-features = false`・`features = ["native-tls", "vendored"]`） | `=3.4.2` 完全固定。`cfg` 範囲は §14.4 の承認結果に従う。用途は `ModelRegistry::download`／`download_with` による HTTPS 取得に限る。`TlsConfig` は `provider(TlsProvider::NativeTls)` を常に明示し、証明書検証を無効化する設定（`disable_verification`）を facade から出さない。`https_only(true)` を付け、リダイレクトの各ホップを `https` に限る。接続・受信のタイムアウトとサイズ上限を明示する（既定は無期限）。panic 経路（プロバイダ・ルート証明書の feature 不足）を固定設定とテストで閉じる。HTTP crate の型を facade の公開面に出さない（`api_surface.rs` のテストで担保。§8-5）。allow に `CDLA-Permissive-2.0`（`webpki-root-certs`）のみを追加する。perl・make をビルド要件として README・`Dockerfile` に反映する。依存追加 PR で `docs/license-matrix.md` に行を追加し `cargo tree`・`cargo deny check advisories bans licenses sources`・`scripts/check-forbidden-deps.sh` を再実測する。HF hub 連携の別クレート（#2243・#2622）が同じ区分を再利用する場合も承認単位は別とする |

**§14.6 の書き直し案（承認チェックリスト。すべて未承認）**

- [ ] HTTP／TLS 依存の第 11 区分を新設すること、および crate と feature（推奨: `ureq =3.4.2`・`native-tls`＋`vendored`。次点: `minreq =3.0.0`・`https-native-tls`）
- [ ] allow への `CDLA-Permissive-2.0` の追加（推奨構成の場合。次点の `minreq` を選ぶ場合は追加なし。`ISC` の直接ライセンスは例外として記録する）
- [ ] OpenSSL をソースから静的に取り込むこと（Apache-2.0 の同梱・脆弱性対応が `openssl-src` の更新頼みになる点・perl／make の要件。§16.10）の受容
- [ ] 配置案（A／B／C）と `cfg` 範囲（`cfg(unix)` 限定／Linux 限定／全ターゲット）
- [ ] 受信タイムアウトが全体期限である点（§16.6）の設計判断（アイドル検知を満たす設計）
- [ ] dirfd 経路の扱い（B-1／B-2。§14.5-2 のとおり。構成に依存しない）
- [ ] facade 公開面の拡張（§7-2 のとおり）
- [ ] 承認後の実装 issue の起票（§9 の草案を基にする）
- [ ] deps-policy・`docs/license-matrix.md`・`deny.toml` への正式反映を承認後の別 PR（#3124）で行うこと

### 16.10 `docs/license-matrix.md` 8 節系の追記案

反映は #3124 で行う（本 PR では `license-matrix.md` に追記案へのポインタ 1 行のみを足す）。

- 追記する行（推奨構成 `ureq =3.4.2` `["native-tls", "vendored"]`・ホスト実測 2026-10-16・advisory-db `479c888c`）: 推移依存数 30／36（x86_64-linux・aarch64-linux）・30／30（aarch64-darwin）。直接ライセンス MIT OR Apache-2.0。allow 外は `CDLA-Permissive-2.0`（`webpki-root-certs 1.0.9`）のみ。コピーレフトなし。`advisories` ok。
- §6「個別判断」相当の注記案:
  - **`webpki-root-certs 1.0.9`**（`CDLA-Permissive-2.0`）: Mozilla のルート証明書一覧を収めたデータクレート。許諾的なデータライセンスで、コピーレフトではない（§13.3）。`native-tls` feature が必ず含めるため、`RootCerts::PlatformVerifier` を使う運用でもビルドに入る。allow への追加はユーザー承認事項。
  - **`openssl-src 300.6.1+3.6.3`**（宣言は `MIT/Apache-2.0`）: **宣言ライセンスで `cargo deny` が通っても、成果物には OpenSSL 本体（同梱ソースの `openssl/LICENSE.txt` は Apache License 2.0）が静的にリンクされる**。Apache-2.0 は allow にあり適合する。OpenSSL の脆弱性対応は `openssl-src` の版更新（依存の更新は人間承認。`deps-policy.md`）に依存し、システム OpenSSL のように OS の更新では直らない。
  - **`minreq 3.0.0`**（直接ライセンス `ISC`）: allow 内だが deps-policy の適合基準（MIT OR Apache-2.0 系）の記述から外れ、次点として採る場合は例外の記録が要る。
- `docs/model-download-design.md` §13.2 の「`aws-lc-sys` は `cmake` 必須」に相当する記述は、`aws-lc-sys 0.46.0` では `cmake` なしでビルドが成功したため、#3124 の反映時に実測値（§16.5）で更新する。

### 16.11 後片付けと本体無変更の確認

- 使い捨てのディレクトリ（CA の秘密鍵・サーバ証明書・各構成のビルド成果物を含む）、Docker イメージ（`Dockerfile` から作ったイメージ・`ubuntu:24.04`）、Python サーバのプロセスは、実測後に削除・終了した。
- 本体の `Cargo.toml`・`Cargo.lock`・`deny.toml`・`crates/**`・`.github/**`・`Dockerfile`・`.claude/**`・`docs/spec/` は変更していない。変更したファイルは `docs/model-download-design.md`（追記のみ）・`docs/license-matrix.md`（§10 末尾にポインタ 1 行）・`docs/README.md`（索引の追記）の 3 つである。

## 17. 第 11 区分の確定記録（2026-10-16・#3123）

### 17.1 位置づけ

- 本節は #3123（所有者の承認取得）の記録である。**§1〜§16 は書き換えない**。§14.4・§14.5-1・§14.6・§16.9 の「未承認」「推奨」の記述は当時の状態であり、確定内容は本節が後続の正になる。
- **確定の性質**: 所有者コメント（#3123 コメント id `6111756315`・2026-10-12・「全て推奨で」の回答として記録されたもの）は次の 2 層からなる。
  - 推奨が本文に示されていた項目の確定: dirfd = B-1、`cfg` = `cfg(unix)` 限定、第 11 区分の新設。
  - 本文に推奨が無かった項目（クレート・版・feature・配置 A／B／C）への委任条項: 「#3122 の実測に基づく推奨を提示し、その推奨形で #3124 へ進めてよい」。差し戻し条件は 3 つ（§17.4）。
- 同コメントは §16（2026-10-16）より前に書かれており、**特定のクレート名を名指していない**。したがって構成と配置は「所有者が特定クレートを承認した」ものではなく、「委任条項に基づき §16.8 の推奨形で確定した」ものである。所有者が後から否認した場合の扱いは §17.5 に従う。
- 本イシューでコード・`Cargo.toml`・`Cargo.lock`・`deny.toml`・`deps-policy.md`・`CLAUDE.md` は変更していない。正式反映は #3124 の担当である。

### 17.2 確定内容

| 項目 | 確定値 | 根拠 |
|---|---|---|
| 区分 | 第 11 区分「HTTP クライアント／TLS」を新設 | 所有者コメントで確定 |
| クレート・版 | `ureq =3.4.2`（`=x.y.z` 完全固定） | 委任条項・§16.8 の推奨（構成 N′+V） |
| feature | `default-features = false`・`features = ["native-tls", "vendored"]` | 同上 |
| `cfg` 範囲 | `cfg(unix)` 限定（`[target.'cfg(unix)'.dependencies]`）。Windows は依存ツリーに入らず、download は fail-closed の非対応（§6 (c)） | 所有者コメントで確定。macOS のクロスジョブとの両立は §16.5 で実測済み |
| 配置 | **案 A（`facade` の無条件依存。ただし `cfg(unix)`）** | 委任条項。B（cargo feature での opt-in）は「feature フラグなし cfg ベース」方針と optional 依存の前例が無いことに反し、選ぶこと自体が新しい方針判断になる。C（非公開の別クレート）は公開クレートから非公開クレートへ依存できず不成立（§4・§14.4） |
| dirfd | B-1。第 10 区分 `libc =0.2.189` の用途を `crates/facade/src/model.rs` のキャッシュ書き込み（`openat`／`openat2`／`mkdirat`／`renameat`）へ広げる。facade は `[target.'cfg(unix)'.dependencies]` で `libc.workspace = true` を直接依存にする。新規 `unsafe` は FFI 境界に限り、理由コメントと security-auditor のレビューを必須とする | 所有者コメントで確定 |
| allow 追加 | `CDLA-Permissive-2.0` のみ（`webpki-root-certs 1.0.9`） | §16.4 |
| SHA-256 | 新規依存なし。`crates/self-repair/src/sha256.rs`（FIPS 180-4 自作）を facade へ移植する（`self-repair` は非公開で facade から依存できない）。`sha2` 等は使わない。実装は #3125 | 差し戻し条件 (3) の回避 |

利用者への影響:

- Linux ビルドに perl・make・C ツールチェーンが要る（debug で +21〜42 秒）。`libssl-dev` は不要で、実行時の `libssl.so.3` 依存も無い。
- macOS は `vendored` が効かず `security-framework` 系になるため、ビルドは軽い（§16.5）。
- perl・make が無い環境でのビルドは未検証（§16.7。#3173 で確認する）。
- facade の公開 API に HTTP crate の型を出さない（`api_surface.rs` で担保。§8-5）。

### 17.3 推奨形に付随する受容事項（§16.8 の (a)〜(e)）

- (a) allow に `CDLA-Permissive-2.0` を追加する。
- (b) Linux ビルドに perl・make が要り、ビルド時間が増える。
- (c) OpenSSL 本体（Apache-2.0）が成果物に静的に入る。
- (d) OpenSSL の脆弱性対応は `openssl-src` の版更新（人間承認）に頼る。検知は CI の `cargo deny check advisories` で行い、`=x.y.z` 完全固定と `Cargo.lock` のコミットで再現性を保つ。
- (e) `TlsProvider::NativeTls` の明示が必須（既定の Rustls は panic する）。

### 17.4 差し戻し 3 条件の判定

| 条件 | 判定 | 根拠 |
|---|---|---|
| (1) §14 に無い allow 識別子 | 該当しない | 追加は `CDLA-Permissive-2.0` のみで、§14.3・§14.5-1・§14.6 に同じ識別子が挙がっている。ただし §14 で挙がっていたのは `webpki-roots` で、N′+V の `webpki-root-certs 1.0.9` とはクレートが異なる（識別子は一致、クレートは別）。識別子は同一の範囲内と判断した |
| (2) 禁止リスト・コピーレフト | 該当しない | §16.4 で全構成 × 3 ターゲットのコピーレフト 0 件。本イシューで禁止リストを追加実測した（下記） |
| (3) 別の新規依存 | 該当しない | SHA-256 は自作実装の移植で足りる（§17.2） |

禁止リストの追加実測（2026-10-16・cargo 1.98.1・本体 workspace 外の使い捨てプロジェクト・空の `[workspace]`・`ureq = { version = "=3.4.2", default-features = false, features = ["native-tls", "vendored"] }`・`cargo generate-lockfile` で解決。51 パッケージ。`ureq 3.4.2`・`native-tls 0.2.18`・`openssl-src 300.6.1+3.6.3`・`webpki-root-certs 1.0.9`）:

- `scripts/check-forbidden-deps.sh lock <Cargo.lock>` は終了コード 0（「依存禁止リストの混入なし」）。`burn`・`cubecl`・`candle`・`tch`・`ndarray` で始まる名前の `[[package]]` は 0 件。
- `cargo metadata --locked` の宣言ライセンスのうち、`MPL`・`GPL`・`EPL`・`CDDL` を含むものは `r-efi 6.0.0`（`MIT OR Apache-2.0 OR LGPL-2.1-or-later`）の 1 件だけだった。これは OR 式で MIT を選べ、UEFI 専用のため §16.4 のターゲット別集計（unix 系 3 ターゲット）では出ない。LGPL を採用する依存ではなく、コピーレフトの混入には当たらない。#3124 の再実測では `--filter-platform` 付きで同じ結果になることを確認する。
- 使い捨てプロジェクトは測定後に削除した。

### 17.5 承認されなかった場合の扱い

差し戻し条件のいずれかに当たった場合、または所有者が推奨形を後から否認した場合:

- #3124 へ進まず、#3123 に差し戻しの理由を書いて所有者へ戻す。
- 第一の代替は次点 M1（`minreq =3.0.0`・`https-native-tls`）で再評価する。allow 追加は不要だが、代償が 3 つある。ISC の直接ライセンスの例外記録、`Dockerfile` への `libssl-dev` 追加、API の不足（`https_only`・サイズ上限・接続タイムアウト・CA 指定が無い。§16.8）。
- いずれの構成も承認されない場合は、HTTPS ダウンロード（#3124・#3125）を保留へ戻し、D-B の実装着手を止める。
- 再評価は新しい実測 issue として起票する。起票は所有者の判断による。

### 17.6 本イシューで決めない事項

- 受信タイムアウトが全体期限でありアイドル期限ではない点（§16.6）の設計判断は #3125 の設計事項とする。依存の承認には影響しない。
- 実機・実通信・perl／make が無い環境・`ubuntu-latest` の実ランナーでの確認は #3173 が担当する。
- facade 公開面の拡張（§7-2）と HTTPS 取得の実装は #3125 が担当する。
- HF Hub 連携の別クレート（#3129）が第 11 区分を再利用する場合は、承認単位を別とする（§14.5-1 を引き継ぐ）。

### 17.7 #3124 の作業項目

1. ルート `Cargo.toml` の `[workspace.dependencies]` に `ureq = { version = "=3.4.2", default-features = false, features = ["native-tls", "vendored"] }` を feature の理由コメント付きで追加する。
2. `crates/facade/Cargo.toml` に `[target.'cfg(unix)'.dependencies]` を新設し、`ureq.workspace = true`・`libc.workspace = true`（第 10 区分の用途拡張。B-1）を置く。`Cargo.lock` を更新してコミットする。
3. `deny.toml` の `[licenses].allow` に `CDLA-Permissive-2.0` を追加する（理由コメントは `webpki-root-certs`）。`[sources]` は crates.io 限定のまま。`cargo deny check advisories bans licenses sources` を通す。
4. `.claude/rules/deps-policy.md` の表に第 11 区分の行を追加し、第 10 区分 `libc` の行へ facade のキャッシュ書き込み用途を追記する。区分数の記述（`deps-policy.md` 冒頭・`CLAUDE.md`・`.claude/rules/coding-rust.md`・ルート `Cargo.toml` と `deny.toml` のコメント）の整合を取る。
5. `docs/license-matrix.md` の 8 節系に行を、6 節に個別判断（`webpki-root-certs`・`openssl-src` の静的同梱と脆弱性対応）を追加する（§16.10 の案を、その時点の解決版で再実測した値で書く）。§13.2 の「`aws-lc-sys` は `cmake` 必須」は §16.5 の実測で更新する。
6. 有効な feature の組合せごとに 3 ターゲットで `cargo tree` を再実測し、`scripts/check-forbidden-deps.sh lock-all`・`tree` が通ることを確認する。
7. facade の README とルート README に Linux のビルド要件（perl・make・C ツールチェーン）を書く。`Dockerfile` は perl・make が既にあるため変更不要の見込みだが、再確認する。
8. 公開 API・取得ロジックは追加しない（#3125）。未使用依存の警告は公開面を増やさない形で扱う。
9. security-auditor の監査を受ける（依存追加と `deny.toml` の変更）。
10. PR 本文と規約には「#3123 の所有者コメント 6111756315 の委任条項に基づく推奨形」と書く。「所有者が `ureq` を承認」とは書かない。

### 17.8 第 11 区分の deps-policy 起案行

§14.6 の起案行（`ureq`・`default-features = false`・`features = ["native-tls", "vendored"]`・`=3.4.2` 完全固定・`cfg(unix)` 限定・`TlsProvider::NativeTls` 明示・`disable_verification` を出さない・`https_only(true)`・タイムアウトとサイズ上限の明示・allow は `CDLA-Permissive-2.0` のみ・HF Hub は承認単位を別とする）を、本節の確定値でそのまま転記する。`cfg` 範囲の文言「§14.4 の承認結果に従う」は「`cfg(unix)` 限定」へ置き換える。転記は #3124 で行う。

## 18. #3124 での正式反映記録（2026-10-16）

### 18.1 反映した内容

- 承認根拠: #3123 所有者コメント 6111756315（2026-10-12）の委任条項（dirfd = B-1・`cfg(unix)` 限定・第 11 区分の新設を確定し、クレート・版・feature・配置は #3122 の実測推奨〈§16.8〉で進めてよい）。コメントは特定クレート名を挙げておらず、§16.8 の推奨形（`ureq =3.4.2`・`native-tls`＋`vendored`）はこの委任に基づく確定である（§17.1・§17.7）
- `Cargo.toml`（`[workspace.dependencies]` に `ureq`）・`crates/facade/Cargo.toml`（`[target.'cfg(unix)'.dependencies]` に `ureq`・`libc`）・`Cargo.lock`・`deny.toml`（allow に `CDLA-Permissive-2.0`）・`.claude/rules/deps-policy.md`（第 11 区分の行・第 10 区分 `libc` の用途拡張）・`docs/license-matrix.md`（追加実測）を更新した
- 取得ロジック・公開 API・dirfd への移行・SHA-256 の移植は追加していない（#3125）。`ureq`／`libc` はまだコードから参照されない（`unused_crate_dependencies` lint は有効でないため `clippy -D warnings` は通る）

### 18.2 再実測の結果

- 実解決版は §16 と同一（`ureq 3.4.2`・`openssl-src 300.6.1+3.6.3`・`webpki-root-certs 1.0.9` ほか）。数値と環境は `docs/license-matrix.md` の追加実測（#3124）を正とし、ここでは転記しない
- `cargo build --workspace --locked`（Linux・warm な依存キャッシュの環境）は成功。macOS クロス（`--lib`）・Windows クロス（clippy）も成功。Windows の依存ツリーには `ureq`・`libc`（facade 経由）が現れない

### 18.3 差し戻し 3 条件の再判定（§17.4）

1. allow に `CDLA-Permissive-2.0` 以外の識別子が要るか: 要らない。`cargo deny` の licenses は追加 1 識別子のみで ok
2. 禁止リストのクレート・コピーレフトの推移的混入: なし（`r-efi` は OR 式で MIT を選べる既存の判断）。`make deps-forbidden` は混入なし
3. 別の新規依存が必要か: 不要

### 18.4 Dockerfile

変更なし。開発コンテナは `build-essential`・`make` を導入済みで、perl はベースイメージに含まれる（§16.5 の開発コンテナ実測で N′+V が成功済み）。

## 19. #3125 実装記録（HTTPS 取得とハッシュ検証）

`ModelRegistry::download`／`download_with` の実装記録。承認根拠は §17.2（#3123 所有者コメント 6111756315 の委任条項に基づく推奨形。それ以上の承認は得ていない）。§1〜§18 は書き換えない。依存・`Cargo.lock`・`deny.toml` は変更なし（#3124 で反映済みの `ureq =3.4.2`・`libc =0.2.189` を使う）。

### 19.1 モジュール構成（`crates/facade/src/model/`。すべて private）

| ファイル | 役割 | cfg |
|---|---|---|
| `url.rs` | `ValidatedUrl`（検証関数からしか作れない newtype）・`validate_https_url`・`resolve_location` | `unix` |
| `sha256.rs` | FIPS 180-4 のストリーミング実装。`K`／`H0`／`compress` は `crates/self-repair/src/sha256.rs` から逐語移植（新規依存なし）。検証は NIST ベクタ・pad 方式の参照実装との境界（55/56/63/64/65 等）一致・分割投入の等価性 | `unix` |
| `transport.rs` | 内部 trait `Transport` と ureq 実装。ureq の型はこのファイルの外へ出ない | `unix` |
| `dirfd.rs` | libc による dirfd 連鎖（`openat`／`mkdirat`／`renameat`／`unlinkat`／`fstatat`）。新規 `unsafe` はここに限る | `unix` |
| `download.rs` | 取得手順（リダイレクト追従・ヘッダ検査・ストリーミング受信・検証・公開） | `unix` |

非 unix（Windows）は名前検証の後、ネットワーク・FS に触れる前に `Io(Unsupported)` を返す。

### 19.2 既定値（利用者が上書きする口はない）

| 項目 | 既定値 | 根拠 |
|---|---|---|
| 最大サイズ | 1 GiB（`MAX_MODEL_FILE_BYTES`） | `load` の上限と同じ。`Content-Length` の事前検査と受信中の計数の両方で守る。超過は既存の `TooLarge` |
| リダイレクト | 5 ホップ・手動追従（`max_redirects(0)`） | 各ホップで `Location` を再検証する。受理は絶対 `https://`・`//host`・`/path` のみ（`../x` 等の相対参照と `http://` は `InvalidUrl`） |
| 名前解決／接続（TLS 含む）／送信 | 各 30 秒 | 無期限の既定を潰す（§15.5） |
| 応答ヘッダ受信 | 60 秒 | 同上 |
| 全体期限 | 1 時間（全ホップ合算） | 呼び出し開始時に締め切りを決め、ホップごとに残りを `timeout_global` へ渡す |
| アイドル（無通信）検知 | 実装しない | §16.6: ureq の受信タイムアウトは全体期限でありアイドル期限ではない。停止は全体期限でのみ検知。上書き口は将来 `DownloadOptions` へ足す |
| `Content-Length` 無し | 受理し最大サイズで打ち切る | チャンク応答では `content_length() == None`（§15.5） |
| `Content-Encoding` | `identity` 以外は拒否 | 伸長しないためハッシュ・長さ検証が壊れる |
| proxy | `None`（環境変数を参照しない） | 暗黙の出口を作らない（A05／A10）。proxy 対応は範囲外（制約） |
| TLS | `NativeTls`＋`RootCerts::PlatformVerifier` | OS ストア。`disable_verification` は公開しない。実在チェーンでの確認は #3173 |

構築直後に `agent.config()` を読み戻して不変条件（provider・検証有効・https_only・リダイレクト 0・proxy なし・各タイムアウト）を照合する。ureq は provider と feature の不整合をリクエスト時に panic で報告しうるため、閉じたポート・TLS を話さないサーバへの要求が panic せず `Err` になる単体テストで feature 整合を CI 固定している。

### 19.3 §5 案との差分

- `ModelError::Http` は §5 案の `url` を持たない（A09。URL にクエリ・資格情報が含まれうる）。ureq のエラーは `Display` を通さず静的メッセージの `io::Error`（種別のみ保持）へ変換する
- `SizeLimitExceeded` は新設せず既存 `TooLarge` を再利用
- `DownloadOptions::new` は設けず `Default` のみ（api_surface の `pub fn new` 件数 1 の正ガードを保つ）

### 19.4 `openat2` を使わない理由

`openat` + `O_NOFOLLOW | O_DIRECTORY` の 1 段ずつの経路は §6 (a) が同格と認める経路で、Linux／macOS 共通・unsafe が最小で済む。一時ファイルは `O_CREAT | O_EXCL | O_NOFOLLOW`、公開は `renameat` 1 回（`RENAME_NOREPLACE` なし）。失敗時は自分の一時ファイルだけを `(dev, ino)` 照合の上 `unlinkat` する（`Drop` ガードで panic の unwind も覆う）。

### 19.5 #3126 との境界

`download.rs::run` の `verify_before_commit` へ、本 PR は §5 の方式 (i)「F32 ロード可能性」を実装した（`model.rs::verify_safetensors_before_commit` が一時ファイルを読み戻して `load_safetensors_f32_from_bytes` を呼ぶ）。方式 (ii)（dtype 非依存のヘッダのみ検証。推奨だった）は `onnx-interop::st_load` への新規関数追加を要するため採らなかった。帰結: (a) REQ-7 の F32 限定契約により F16／BF16 の妥当な safetensors も公開前検証で拒否される、(b) 検証のためファイル全量（上限 `MAX_MODEL_FILE_BYTES`）をメモリへ読み込み、さらに F32 テンソルを確保するため `load` と同等のメモリを一時的に使う。dtype 非依存化（方式 (ii)）は #3126 以降の課題とする。`download.json`・`If-None-Match`／304 は未実装（304 は異常として `Http`）。pin 無しの保証範囲は破損・部分改変の検出に限られる（rustdoc に明記）。

### 19.6 `save` について

`save` は `/proc/self/fd` 方式のまま（Linux x86_64／aarch64 限定。`docs/facade-model-registry-decision.md` §16.9）。dirfd への移行は本 PR の範囲外で、別イシューでの起票をユーザーに提案する（自動運転中は起票していない）。

### 19.7 新規 `unsafe`

`dirfd.rs` の libc 呼び出し（`openat`・`mkdirat`・`renameat`・`fstatat`・`fstat`・`unlinkat`）と `OwnedFd::from_raw_fd`・`mem::zeroed::<libc::stat>` のみ。各ブロックに SAFETY コメント（fd の生存期間・`CString` の寿命・戻り値検査）を付けた。

## 20. #3126 実装記録（ダウンロードのローカルキャッシュ・条件付き GET・304 再検証）

§19.5 で #3126 へ送った課題を実装した。§1〜§19 は書き換えず、本節で差分を記録する。

### 20.1 `download.json`

- 場所は `<cache_dir>/<name>/<version>/download.json`。`serde_json::Value` から手で組み立て・検証する（`serde` の derive は使わない）。実装は `crates/facade/src/model/cache.rs`
- 6 キーちょうど（未知キー・不足は拒否）: `url`（検証済み https URL・フラグメント除去済み）・`final_url`（リダイレクト追従後に `200` を返した最終取得 URL。`etag` はこの URL のリソースの検証子。検証済み https URL）・`etag`（文字列または `null`）・`content_length`（u64）・`sha256`（64 桁 16 進）・`fetched_at`（UNIX エポック秒 u64）
- 読み取り上限 64 KiB。不在・I/O エラー・スキーマ違反・シンボリックリンク・非通常ファイルはすべて「キャッシュなし」として無条件 GET へフォールバックする（要件 1。`ModelError::CacheMetadata` は読み出し側では呼び出し側へ届かず、書き込もうとしたメタデータが不正な場合にだけ返る）
- `etag` は印字可能 ASCII（0x20〜0x7E）・長さ 1〜1024 のみ。CR／LF／NUL・制御文字・非 ASCII は拒否する。保存時・読み出し時・送信直前（`transport.rs`）の 3 か所で同じ `valid_etag` を使う（ヘッダインジェクション対策）。応答の `ETag` が不正なら保存せず `null` を書く（古い etag を残さない）
- 保存値は破損・部分改変の検出用であり、真正性の根拠ではない（§6 A08・`model-distribution-design.md` §6）。`url` はクエリに署名トークンを含みうるため、0600 のファイルにだけ保存し、進捗・エラーには流さない（A09）

### 20.2 条件付き GET と 304 再検証

- 条件付き GET を送るのは、`download.json` が読めて、`url` が要求 URL と一致し、`etag` があり、pin 指定時は保存 `sha256` が pin と一致するときだけ（`model.rs::download_via` の参照段）。`If-None-Match` は**保存済み `final_url` と一致するホップにだけ**付け、そのホップの `304` だけを受理する（途中ホップや、リダイレクト先が A→B から A→C へ変わった場合は検証子を送らず無条件取得。別リソースが同じ ETag〈例 `"v1"`〉を返しても `304` で古いモデルを再利用しない。ローカル SHA の再計算ではこの取り違えを検出できないため。PR #3388 コードレビュー指摘）
- `304` は条件付きで送ったときだけ受理する。無条件 GET や、再検証失敗後にやり直した GET への `304` は `Http{304}`（サーバの異常応答を再利用のシグナルにしない）
- 再検証は既存 `model.safetensors` を `O_NOFOLLOW`・通常ファイル確認・サイズ上限付きで開き、SHA-256 をストリーミングで計算し直す。比較対象は pin があれば pin、なければ保存値。一致したときだけ `fetched_at` のみを更新して `download.json` を atomic に置き換える。一致しない・ファイルが無い・リンクなら、無条件 GET で全量を取り直す
- 参照段は `<name>/<version>` を作成しない（`open_root_existing`／`open_existing_dir`）。ヘッダ検査で失敗する取得は従来どおりディレクトリを作らない

### 20.3 公開順序と自然回復

`model.safetensors` を先に `renameat` し、次に `download.json` を `renameat` する。途中で止まって JSON が古いまま残っても、次回の `304` 再検証で SHA が合わず全量を取り直す。`download.json` の書き込みだけが失敗した場合もエラーを返す（モデル本体は検証済みで公開済み）。いずれも `RENAME_NOREPLACE` は付けず atomic replace で上書きする。`TmpGuard`（private staging 0700）は両方の公開で共用する。

### 20.4 検証方式 (ii) の採用

公開前検証を `onnx-interop::st_load::validate_safetensors_layout_from_bytes`（`SafeTensors::read_metadata` を呼ぶだけ。ヘッダ長・UTF-8・JSON・オフセット連続性・shape×dtype 長・全体長の一致を見て、dtype は限定しない）へ切り替えた。F16／BF16 の正当なファイルも公開できる（`load` は従来どおり F32 のみで `UnsupportedDtype`）。F32 テンソルを確保しないため、メモリはファイル全量 1 回分（上限 1 GiB）に減る。

- 公開クレート `onnx-interop` への変更は関数 1 つの追加のみ（既存関数・公開型は不変、依存は増えない）。facade は `safetensors` に直接依存できないため、この形にした
- facade 側は onnx-interop の参照を `src/interop/` に閉じ込める既存ガード（#2017）に合わせ、`src/interop/layout.rs` の `pub(crate)` ラッパー経由で呼ぶ（`safetensors.rs` は純再エクスポート契約のため触らない）。`api_surface.rs` の `mod.rs` 件数ガードを 3 から 4 に更新した。承認済み公開アイテム 16 件・公開メソッド 8 件は増やしていない

### 20.5 `serde_json` の結線

`crates/facade/Cargo.toml` の `[target.'cfg(unix)'.dependencies]` に `serde_json.workspace = true` を追加した。第 5 区分の承認済み依存で workspace 固定（`=1.0.151`）は変えず、`Cargo.lock` の解決結果も変わらない（dev-dependencies に既にあるため）。

### 20.6 エラー型

`ModelError` に `CacheMetadata { reason }`・`StorageFull(io::Error)`・`PermissionDenied(io::Error)` を追加した（`#[non_exhaustive]` のため非破壊。`Display` に URL・パスを含めない）。`map_fs_error` による割り当て（`StorageFull`／`QuotaExceeded` → `StorageFull`、`PermissionDenied` → `PermissionDenied`、他は `Io`）は**ダウンロード経路のみ**で、`save`／`load` は変えていない。

### 20.7 検証範囲と後続候補

- 単体テストは転送をモックし、実ファイルシステムで 304 再利用・改変後の取り直し・pin 優先・リンク拒否・破損 JSON のフォールバックを検証した。権限エラーは 0500 ディレクトリで実際に検証する（root 実行時はスキップ）。実際の ENOSPC は CI で再現できないため、`open_dir` への `io::Error` 注入による単体の水準にとどめた。実ネットワーク・TLS の実接続は #3173 で追跡済み
- 後続候補（起票はユーザー承認待ちのため未起票）: (a) ヘッダだけを読むストリーミング検証（`onnx-interop` の `serde_json` を通常依存へ格上げする判断が要る）、(b) `save`／`load` の FS エラーの型付き化と `save` の dirfd 化（§19.6）
