# リモートモデル取得（URL ダウンロード）の設計記録（#2088）

イシュー #2088「feat(facade): リモートモデル取得（URL ダウンロード）の実装（依存追加の承認後）」に対応する。親: #2082（`docs/model-distribution-design.md`。本 doc 作成後に作成済み）。兄弟: #2087（`crates/facade/src/model.rs`・`ModelRegistry`・`ModelError`。マージ済み）。

本ドキュメントは**コード変更を伴わない設計記録**を成果物とする。`Cargo.toml`／`Cargo.lock`／`deny.toml`／`docs/license-matrix.md`／`docs/spec/`（正本 submodule）・`crates/**` はいずれも変更しない。

基準コミット: 本 PR ブランチ作成時点の `origin/main`（`c498389f0e936c147ce31aa75e21d9808c751db0`・2026-09-23）。

**結論**: 本イシューの受け入れ条件は「依存追加のユーザー承認**取得時**」と「**未取得時**」の 2 系統に分かれる。自動運転モード（ユーザーへの質問・承認待ち不可）では HTTP クライアント依存の追加承認・facade 公開面拡張の承認を得られないため、本ドキュメントは「承認未取得時」の系統として、承認判断に必要な材料（候補クレート・ライセンス実測・技術仕様案・承認後の実装手順・セキュリティ設計）を 1 箇所に確定させる。実装（`ModelRegistry::download` 本体・`api_surface.rs` 到達性テスト・進捗ログ機構）は承認後の別イシューへ引き継ぐ。

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

- 現行 allow で通る組合せは「TLS なし」を除くと、`ureq`⑥（`native-tls-no-default`）・`minreq`③④（`https-native-tls`・`https-openssl`）だけである。いずれも OS の TLS（Linux は OpenSSL、macOS は Security.framework、Windows は schannel）を使う。
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

### 13.4 システムライブラリ・ビルド要件・MSRV

ライセンスとは別の運用上の制約。`cargo build` は実行していないため依存関係と manifest からの判断であり、実ビルドでの確認は承認後の実装手順（§8）で行う。（**2026-10-07 追記**: `ureq` の `native-tls` 系・`rustls`〈`ring`〉の実ビルド結果は §15.3・§15.6。`aws-lc-sys`・`minreq`・`attohttpc`・`reqwest` は未ビルドのまま）

| 経路 | 要件 |
|---|---|
| `native-tls`／`https-native-tls`／`attohttpc` 既定 | Linux は `openssl-sys 0.9.117`（システム OpenSSL の開発ヘッダ／ライブラリ、または `vendored` feature）。macOS は `security-framework 3.7.0`、Windows は `schannel 0.1.29` で追加の C ライブラリ不要。§3 の「環境非依存の開発コンテナ」方針との整合は要検証 |
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
- 根拠は §13 の実測（#2620）のみで、新たな `cargo build`／実測は行っていない。API・feature の確認は `cargo info`・docs.rs の公開ドキュメントによる。固定版は 2026-10-03 時点の crates.io 解決版であり、承認後の依存追加 PR で再実測する。（**2026-10-07 追記**: 使い捨てプロジェクトでの実ビルド・挙動確認を §15 に記録した。**推奨案の構成は HTTPS が成立しなかった**〈§15.4・§15.8〉。本節の本文は起案時点のまま残す）
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

**除外**: `attohttpc`（直接ライセンスが MPL-2.0。§13.2）・`reqwest`（推移依存 90〜108 個で `tokio` を内包。現行 allow で通る組合せなし）。

現行 allow で `licenses` が通る 3 組合せの比較（§13.2・§13.4 の値を引用。`cargo build` は未実行。**2026-10-07 追記**: `ureq` ⑥ の実ビルド結果は §15.3）:

| 軸 | `ureq =3.4.2` ⑥ `native-tls-no-default` | `minreq =3.0.0` ③ `https-native-tls` | `minreq` ④ `https-openssl` |
|---|---|---|---|
| (a) 直接ライセンスと deps-policy の適合基準（MIT OR Apache-2.0 系） | MIT OR Apache-2.0。合致 | ISC。allow 内だが基準の記述から外れ、例外扱いの記録が要る | 同左 |
| (b) §5・§6 の API 要件 | 下表のとおり大半を公開 API で確認 | 未確認（承認後に確認） | 未確認（承認後に確認） |
| (c) 推移依存数（x86_64-linux／aarch64-linux／aarch64-darwin／x86_64-windows）・MSRV | 34／34／29／19・1.85 | 21／21／15／5・1.63 | 19／19／19／19・1.63 |
| (d) システムライブラリ | Linux は `openssl-sys`（OpenSSL 開発パッケージ）が要る | 同左 | `openssl/vendored` により C ツールチェーンでソースビルド（全ターゲット） |

(b) の確認結果（`ureq =3.4.2`。docs.rs の公開ドキュメントによる。実装時に再確認する）:

| 要件（§5・§6） | 確認結果 |
|---|---|
| リダイレクトの各ホップで `https` 限定を検証 | `max_redirects` の既定が 0（自動追従しない）。呼び出し側で `Location` を検証しながら手動追従できる。`https_only` 設定もある（**2026-10-07 追記**: 実測では既定は 10 で自動追従した。`max_redirects(0)` の明示が要る → §15.5） |
| 接続・読み取りのタイムアウト | `timeout_connect`／`timeout_recv_body`／`timeout_global` 等を確認 |
| ストリーミング受信・`Content-Length`／`ETag` の取得 | ボディのリーダー経由の受信とヘッダ取得ができる設計。サイズ上限・ハッシュ計算を逐次行えるかは承認後に実装で確認（**2026-10-07 追記**: 逐次受信・サイズ上限の挙動を実測 → §15.5） |
| `If-None-Match` の付与と 304 判別 | 任意ヘッダ付与と 304 の取得は可能な想定。ただし `>=400` をエラー化する既定の扱いは実装時に確認（**2026-10-07 追記**: 304 は既定のまま `Ok`、`>=400` は `Err(StatusCode)` → §15.5） |
| 証明書検証を無効化しない既定 | 既定は検証有効。`disable_verification` が存在するため、facade からこの設定を出さないことを条件にする |
| `native-tls-no-default` での実行時設定 | TLS プロバイダ（native-tls）とルート証明書源（OS ストア）をどう指定するかは**未確認（承認後に確認）**。同 feature は `webpki-root-certs` を含まない（**2026-10-07 追記**: 同 feature 単独では HTTPS が成立しなかった → §15.4） |

**推奨案**: `ureq =3.4.2`・`default-features = false`・`features = ["native-tls-no-default"]`。直接ライセンスが MIT OR Apache-2.0 で、現行 allow を変えずに `licenses` が通る。`minreq` は依存数が最小だが、直接ライセンスが基準外で API 要件も未確認のため次点とする。（**2026-10-07 追記**: この推奨案の構成は実測で HTTPS が成立しなかった。推奨の見直しは未実施で、ユーザー判断を仰ぐ → §15.8）

**代替案**（allow リスト変更の別途承認が要る）: `ureq` ②（`rustls`＝`ring`＋`webpki-roots`、推移依存 25 個）。システム OpenSSL が不要になる代わり、allow に `BSD-3-Clause`（`subtle`）と `CDLA-Permissive-2.0`（`webpki-roots`）の追加承認が要る（いずれもコピーレフトではない。§13.3）。

**トレードオフ**: 推奨案は Linux のビルドに OpenSSL 開発パッケージを要求する。現行の `Dockerfile` は `pkg-config`・`build-essential` 等は入れるが `libssl-dev` を入れていない。CI の `ubuntu-latest` の有無は未確認。開発コンテナ・CI への導入要否は承認後の実ビルドで確認する（推測で「入っている」と書かない）。代替案はこの要件を避けられるが、allow 追加の承認が要る。（**2026-10-07 追記**: `Dockerfile`・CI の事実確認と、代替案〈`ring`〉の Linux → `aarch64-apple-darwin` クロス lib ビルド失敗の実測は §15.6。`ubuntu-latest` の導入状況は未検証のまま）

### 14.4 配置案と `cfg` 範囲

§4 の案 A／B／C を TLS 方式との組合せで再評価する。

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

**1. 第 11 区分（HTTP クライアント／TLS）**

| 区分 | クレート | 条件 |
|---|---|---|
| HTTP クライアント／TLS（第 11 区分） | 推奨: `ureq`（`default-features = false`・`native-tls-no-default`） | `=3.4.2` 完全固定。`cfg` 範囲は §14.4 の承認結果に従う。用途は `ModelRegistry::download`／`download_with` による HTTPS 取得に限る。HTTP crate の型を facade の公開面に出さない（`api_surface.rs` のテストで担保。§8-5）。TLS 証明書検証を無効化するオプションを設けない（§6）。非同期ランタイム（`tokio`／`hyper`）を推移依存に含めない。依存追加 PR で `docs/license-matrix.md` に行を追加し `cargo tree` を再実測する。`cargo deny check advisories bans licenses sources` と `scripts/check-forbidden-deps.sh` を通す。代替案（`rustls`）を採る場合は allow 追加（`BSD-3-Clause`・`CDLA-Permissive-2.0`）を別途承認する。HF hub 連携の別クレート（#2243・#2622）が同じ区分を再利用する場合も承認単位は別とする |

**2. dirfd 相対操作（§6・§7-5）**。第 10 区分 `libc` はすでに承認済みのため、新規区分ではなく次の二択にする。

- **案 B-1（推奨）**: 第 10 区分の用途を `crates/facade/src/model.rs` のキャッシュ書き込み（`openat`／`openat2`／`mkdirat`／`renameat`）へ拡張する。版は `=0.2.189` のまま `cfg(unix)` 限定で facade に直接依存させる。新規 `unsafe` は FFI 境界に限り、理由コメントと security-auditor のレビューを必須とする。公開クレートの公開面を広げず、既承認ピンと同型である。
- 案 B-2: `onnx-interop/src/onnx/external_data.rs` の `pub(super)` ヘルパー（`open_chain_openat2`・`openat_no_follow` 等）を公開して facade から使う。`libc` の用途拡張は不要だが、crates.io 公開クレート `fandhe-ai-onnx-interop` の公開面が増える。また `mkdirat`／`renameat` は既存ヘルパーにないため追加が要る。

どちらでも §6 の symlink 脱出防御と、Windows 非対応の fail-closed 契約は変更しない。

### 14.6 承認チェックリスト（ユーザーが選ぶ。すべて未承認）

- [ ] HTTP／TLS 依存の第 11 区分を新設すること、および crate と feature（推奨案 `ureq` `native-tls-no-default`／代替案 `rustls`）
- [ ] 代替案を採る場合の allow 追加（`BSD-3-Clause`・`CDLA-Permissive-2.0`）
- [ ] 配置案（A／B／C）と `cfg` 範囲（`cfg(unix)` 限定／全ターゲット）
- [ ] dirfd 経路の扱い（B-1／B-2）
- [ ] facade 公開面の拡張（§7-2 のとおり）
- [ ] 承認後の実装 issue の起票（§9 の草案を基にする。起票自体もユーザー承認事項）
- [ ] deps-policy・`docs/license-matrix.md`・`deny.toml` への正式反映を承認後の別 PR で行うこと

### 14.7 承認後の手順との対応

- §8-1 の「HTTP クライアント候補」は §14.5-1 の承認結果の crate・feature・`cfg` を指す。「OS 呼び出しラッパー」は新規依存ではなく、B-1 なら第 10 区分 `libc` の用途拡張、B-2 なら onnx-interop ヘルパーの公開を指す。
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
- 推奨案の見直し（構成の選び直し・`minreq` 等の再評価・allow 追加の要否）は本節では行わない。§14.6 の承認チェックリストの前提が変わるため、起案の更新とあわせてユーザー判断を仰ぐ。
