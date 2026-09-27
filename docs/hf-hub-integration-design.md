# HF hub 連携クレートの境界と取得 API 案の設計判断記録（#2244）

- イシュー #2244（親 #2243「HF hub 連携クレートの設計判断記録と依存追加の承認申請」・Phase 親 #2131）
- **コード変更なし・依存追加なし**。本 doc は設計案の記録であり、確定はユーザー承認後
- 基準コミット: 作業ブランチ作成時点の `origin/main`（`9fe4b523`。2026-09-27）
- 結論の要約: 本 doc は §1〜§3 に境界と API 案を記す設計案であり、正式決定ではない。**§4（認証トークン・セキュリティ）は #2245 で追記済み・§5・§6（他ライブラリ対応表・承認事項一覧）は #2246 で追記済み**

## 1. 背景

`docs/model-distribution-design.md` §5「HF hub 連携の方針（#2082 のスコープ外・ユーザー決定 2026-09-24）」により、役割分担は以下のとおり確定済みである（本節は同 §5 の要約であり、正本は同節）:

- **汎用の URL ダウンロード＋ローカルキャッシュ＋ハッシュ検証はコア（`facade`）側の責務**
- **特定ハブ（Hugging Face Hub）連携はコアに入れず、別クレート（または opt-in の別経路）の責務**

この分離は PyTorch（コア `torch.hub` と別パッケージ `huggingface_hub`）・Keras（コア `tf.keras.utils.get_file` と別パッケージ `huggingface_hub` の `hf://` 対応）と同型である（出典は `docs/model-distribution-design.md` §5 記載の一次情報 URL を参照。詳細な対応表は §5 に譲り、本 doc の §5 節〈#2246 追記〉に委ねる）。

### 前提状態

| イシュー | 内容 | 状態 |
|---|---|---|
| #2087 | ローカルモデルレジストリ（`ModelRegistry`） | 実装済み |
| #2088 | 汎用 HTTPS 取得（`download`／`download_with`） | 設計記録のみ。依存 2 件は未承認、未実装 |
| 本 #2244 | クレート境界・取得 API 案 | 設計のみ（本 doc） |
| #2245 | 認証トークン・セキュリティ設計 | 本 doc §4 に追記済み |
| #2246 | 他ライブラリ対応表・承認事項一覧 | 本 doc §5・§6 へ追記済み |

**前提チェーン**: 本 doc の API 案（§3）は、facade 側の汎用 HTTPS 取得（#2088）がユーザー承認・実装されるまで実装に着手できない。#2088 が提供する `download_with` 相当の機能に依存する設計であるため、本 doc の記述は現時点では机上案に留まる。

## 2. クレート境界

### 2.1 責務分割表

| 責務 | 担当 |
|---|---|
| 汎用 HTTPS 取得（リダイレクト先の検証・サイズ上限を含む） | facade（`ModelRegistry` 拡張。#2088） |
| 書き込み（一時ファイル＋rename・dirfd 相対書き込み） | facade（#2088） |
| 整合性（sha256 pin 検証・`download.json` manifest） | facade（#2088） |
| safetensors 読込（`ModelRegistry::load`） | facade（実装済み） |
| `repo_id` の検証 | 別クレート |
| revision 解決（branch・tag → commit sha） | 別クレート |
| ファイル一覧の取得 | 別クレート |
| HF の `resolve` 形式 URL の組み立て | 別クレート |
| キャッシュキーへの写像（§3.3） | 別クレート |
| push 系・認証・dataset／space リポ | いずれにも入れない（対象外。認証トークンの供給・秘匿契約・private repo 扱いは §4 で整理済み。初期状態は private/gated repo を fail-closed の非対応とする〈§4.1〉） |

### 2.2 依存方向

- 依存は**別クレートから `fandhe-ai`（facade）への一方向**とし、facade から別クレートへの依存は作らない
- facade の依存形状は `crates/facade/tests/api_surface.rs` が機械的に固定している。この向きであれば `docs/model-download-design.md` §4 の案 C（非公開の別クレートへ切り出す案）が指摘する構造問題（公開クレートから非公開クレートへ依存する向き）は生じない
- 別クレートが使うのは facade の公開 API（`ModelRegistry` と、#2088 承認後に追加される見込みの `download_with` 相当）だけであり、`tensor-core`／`autodiff`／`backend-*` 等の内部クレートには直接依存しない（`docs/compat-api-scope.md` §0）

### 2.3 中心論点: HF 固有の JSON API をどの HTTP 経路で呼ぶか

revision 解決とファイル一覧取得は、facade が提供する「バイト列をキャッシュへ取得する」機構だけでは賄えない（HF Hub API の JSON レスポンス解析が別途必要）。案を以下に並べ、推奨のみを記す。

- **案 A（推奨）**: 別クレートが自分で HTTP クライアント依存を持つ。#2088 の依存候補（`docs/model-download-design.md` §3）と同じクレートを再利用できる可能性はあるが、依存宣言と承認は別クレート単位で別途必要。詳細な依存列挙は §6（#2246）に委ねる
- **案 B**: facade が汎用の「HTTPS GET でバイト列／JSON を返す」プリミティブを公開し、別クレートはそれを使う。facade 公開面の拡張として別途承認が必要であり、HF 固有の要求（JSON API 呼び出し）がコアへ滲み出す懸念がある
- **案 C**: 別クレートは HF API を呼ばず、呼び出し側が commit sha とファイル名を既に知っている場合に `resolve` URL を組み立てるだけにする。最も小さいが、branch・tag の解決とファイル一覧取得ができず、受け入れ条件（revision 解決・ファイル一覧）を満たさない

推奨は**案 A**（コアを汚さず、PyTorch／Keras と同じ分離を保つ）。確定はユーザー承認による。

### 2.4 公開区分と名称

選択肢は次の 3 つ。

1. crates.io に公開する（8 件目。`fandhe-ai-` prefix、例 `fandhe-ai-hf-hub`。`docs/crates-io-naming-decision.md` の命名方針に従う）
2. `publish = false` の workspace 内クレート（前例: `docs-site`）
3. workspace 外の独立パッケージ

crates.io には第三者の `hf-hub` クレートが既に存在するため、名前の衝突・空き確認は**未実測**である。確認方法は `docs/crates-io-naming-decision.md` の手順に従い、承認後に実施する。

ディレクトリ名案（例 `crates/hf-hub`）を採用する場合、CLAUDE.md の「想定クレート 10 個＋docs-site」という記述の更新が必要になる（更新自体は承認後）。

選択肢 (1) を採ると `docs/compat-api-scope.md` §0「`facade` が唯一のサポートされる公開 API 面である」という記述と緊張関係が生じる（2 つ目の公開面が生まれる）。この扱いは承認事項（§2.6 項 6）とし、#2194 との文言整合は §6（#2246）へ申し送る。

### 2.5 既存 HF クライアントクレートを使う案（比較のみ）

crates.io の既存 `hf-hub` クレートを依存として使う案は**非推奨**として記録する。理由は、当該クレートが独自のキャッシュと HTTP スタックを持ち、facade の pin 検証（`docs/model-download-design.md` §5〜§6）と dirfd 書き込み契約を迂回してしまうため。依存内容・ライセンスの詳細確認は §6（#2246）に委ね、外部仕様は未実測として扱う。

### 2.6 境界に関わる承認事項

本 doc 時点ではすべて未取得・未実施（依存追加の網羅的な列挙・#2088 §7 との共有関係の整理は §6〈#2246〉を正とする）。

1. 新規クレートの workspace 追加（ルート `Cargo.toml` の `members`）
2. クレート名（ディレクトリ名と `[package] name`）
3. 公開区分（crates.io 公開と `release-all.yml` への追加の有無。§2.4）
4. HTTP 経路の選択（§2.3）
5. facade 公開面の拡張（§2.3 案 B を採る場合のみ）
6. `docs/compat-api-scope.md` §0 のサポート境界の扱い（§2.4）
7. 新規 `unsafe`（本設計では不要の見込み。dirfd 操作は facade 側 #2088 の責務であり別クレート側では発生しない）
8. 依存追加に伴う `scripts/check-forbidden-deps.sh`・`deny.toml`・`docs/license-matrix.md` への影響
9. 承認後の実装イシューの起票

## 3. 取得 API 案（案であり確定ではない）

### 3.1 API スケッチ

以下は Rust の疑似シグネチャであり、実装は行わない。

```rust
// 入力型
struct HfRepoId { /* namespace/name 形式。§3.2 で検証規則を規定 */ }
impl HfRepoId {
    fn parse(s: &str) -> Result<Self, HfError>;
}

enum Revision {
    CommitSha(String),  // 40 桁 16 進
    Ref(String),        // branch・tag の ref 名。既定は "main"
}
impl Revision {
    fn parse(s: &str) -> Result<Self, HfError>;
}

// エントリポイント
struct HfHub {
    registry: fandhe_ai::ModelRegistry,
}
impl HfHub {
    fn new(registry: fandhe_ai::ModelRegistry) -> Self;

    fn resolve_revision(&self, repo: &HfRepoId, rev: &Revision) -> Result<CommitSha, HfError>;
    fn list_files(&self, repo: &HfRepoId, rev: &CommitSha) -> Result<Vec<RepoFile>, HfError>;
    // 内部で facade の download_with（#2088 承認後）に委ね、pin をそのまま渡す
    fn download(
        &self,
        repo: &HfRepoId,
        rev: &CommitSha,
        filename: &str,
        options: DownloadOptions,
    ) -> Result<CachedModel, HfError>; // CachedModel { name, version }。
    // filename はキャッシュキー（name/version）に反映されない（§3.3 参照）ため、
    // 同一 repo・commit 内で異なる filename を渡す呼び出しはキャッシュ衝突になりうる。
    // §3.3 の是正方針（filename を単一許容値へ限定する案）が確定するまで、
    // 本シグネチャの filename 引数はこの制約付きの案として扱う
}

// #[non_exhaustive]。facade の ModelError を内部で包む
enum HfError { /* ... */ }
```

huggingface_hub（Python）の `hf_hub_download(repo_id, filename, revision)` との対応関係をこの位置に記す想定である（§5.2 に記載）。

HF 側では branch と tag が同じ ref 解決経路になる見込みであるため、上記スケッチでは型で区別しない案としている。この根拠は**外部仕様・要確認**（§3.5）であり、断定しない。

### 3.2 入力検証（fail-closed）

`docs/facade-model-registry-decision.md` §7 の allowlist 方針に倣い、次の 3 点を規定する。

- **`repo_id`**: `namespace/name` 形式・許容文字・長さ・`..`・`--` の扱いを allowlist で規定する
- **`revision`**: ref 名が `/` を含みうる（PR ref 等）ため、URL エンコードの方針を決める
- **`filename`**: 相対パス・`..`・絶対パスの先頭・NUL・制御文字を拒否する。サブフォルダを許容するかも決める

HF 側の命名規則（区切りに使う `--` が repo 名に出現しうるか、長さの上限等）は**外部仕様・要確認**であり、一次出典 URL を付けたうえで断定しない（§3.5 参照）。

### 3.3 キャッシュ配置と `ModelRegistry` の関係

`repo_id` は `/` を含むため、`ModelRegistry` の `name` にそのまま使えない（`docs/facade-model-registry-decision.md` §7 の `[A-Za-z0-9._-]+` allowlist・先頭 `.` 不可・`/` 不可）。候補を比較する。

- **候補 1（推奨案）**: `name` は repo_id を符号化したもの（例 `hf--<namespace>--<repo>`。区切りが一意になるかは §3.2 の要確認事項に依存する）とし、`version` は**解決済みの commit sha**（40 桁 16 進で allowlist に適合し、内容が変わらない）とする。branch・tag 名を `version` にしない理由: 同じ名前が時期によって別の内容を指しうるためキャッシュ無効化の問題が生じる。また `ModelRegistry` には refs（別名）の概念がない
- **候補 2**: `ModelRegistry` とは別に HF 専用キャッシュ（huggingface_hub の `blobs`／`snapshots`／`refs` 相当）を持つ。dirfd 書き込み契約と pin 検証を別クレートで作り直すことになるため非推奨

branch から sha への対応を覚えておく refs 相当の永続化は**スコープ外**とする（毎回解決するか、呼び出し側が sha を保持する）。オフライン時の扱いは §7 の未決事項に挙げる。

**単一ファイル制約**: `ModelRegistry` は `<cache_dir>/<name>/<version>/model.safetensors` の単一ファイルしか扱わない（`docs/model-distribution-design.md` §2.1）。このため `config.json`、分割 safetensors（index json と複数 shard）、サブフォルダ、safetensors 以外のファイルは本案の対象外とする。将来これらを扱う場合、`ModelRegistry` 自体の拡張が必要になり、それは facade 公開面拡張の承認事項（§2.6 項 5 相当）になる。

**キャッシュキーへの `filename` 反映（要是正・codex レビュー指摘）**: 候補 1 の `name`（repo_id 由来）・`version`（commit sha）は `filename` に由来しないため、`HfHub::download` が任意の `filename` を受け取れる §3.1 のスケッチのままでは、同一 repo・同一 commit 内の異なる safetensors ファイル（例: 分割 checkpoint の複数 shard 名を許してしまった場合）が同じ `<name>/<version>/model.safetensors` へ書き込まれ、後から取得した方が先の内容を無言で上書きする。返却する `CachedModel { name, version }` にも `filename` が含まれないため、呼び出し側はどのファイルがキャッシュされているか区別できない。是正方針は次の 2 択のいずれかとし、実装イシューで確定する（本 doc は設計判断記録であり、ここでは選択肢を提示するに留め断定しない）:
  - **是正案 A（推奨）**: `download` が受理する `filename` を単一の許容値（`ModelRegistry` の単一ファイル制約に合わせた canonical な safetensors ファイル名。例: リポジトリ内で解決したメインの safetensors ファイル 1 つのみ）に限定し、それ以外の `filename` は fail-closed で拒否する。単一ファイル制約（前段落）と整合し、キャッシュキーの衝突が構造的に発生しない
  - **是正案 B**: `filename` を検証済みの形で `name` の符号化（`hf--<namespace>--<repo>--<filename の符号化>` 等）へ衝突なく組み込み、`CachedModel` にも `filename` を含めて返す。複数ファイルの共存キャッシュを許すが、区切り文字の一意性（§3.2 の要確認事項）に依存する分だけ検証が複雑になる
  いずれの案でも、`CachedModel` から実際にキャッシュされたファイルを一意に特定できることを受け入れ条件とする。

### 3.4 safetensors 読込への接続

- `ModelRegistry::load(name, version)`（実装済み。`crates/facade/src/model.rs:648`）を経由し、`docs/facade-safetensors-exposure-decision.md` §11 の F32 限定契約をそのまま引き継ぐ。bf16／f16 の HF チェックポイントは fail-closed で失敗する
- HF レイアウトから `compat::Sequential` への変換は `docs/huggingface-safetensors-interop-guide.md` §3「レイアウト対応表」に任せる。別クレートは暗黙の変換をしない（REQ-7 の「暗黙アダプタなし」契約）
- `ModelRegistry::load`／`available_models` の Windows 非対応（fail-closed。`docs/facade-model-registry-decision.md` §14）もそのまま引き継ぐ

### 3.5 HF 固有の外部仕様（要確認リスト）

以下はいずれも一次出典 URL を付けて確認すべき事項であり、確認できなかった項目は**外部仕様・未実測**として扱い、実装イシューで確認し直す。

1. `resolve` URL の形式（例: `https://huggingface.co/<repo_id>/resolve/<revision>/<filename>`）
2. revision 一覧・ファイル一覧取得 API のエンドポイント形式
3. commit sha を返すレスポンスヘッダの名称
4. LFS の etag 系ヘッダの扱い
5. CDN リダイレクト先のドメイン

出典候補: https://huggingface.co/docs/hub/api （HF Hub API ドキュメント）。本 doc 作成時点では上記 5 点の詳細確認は行っておらず、実装イシュー側での reference-researcher 委譲確認を要する。

### 3.6 セキュリティ上の前提（境界レベルのみ。詳細は §4／#2245）

- `repo_id`・`revision`・`filename` は fail-closed で検証する（§3.2）
- キャッシュ書き込みは facade の dirfd 契約に任せ、別クレートではファイル I/O をしない。**唯一の例外はトークンファイルの読み取り**であり、読み取り専用・キャッシュルート外限定の条件付きで許容する（§4.1）
- HTTPS 限定はリダイレクト先にも及び、TLS 検証を無効化するオプションは設けない（`docs/model-download-design.md` §6 を継承）
- サーバーが返すハッシュ（etag 系）は信頼根にせず、真正性の根拠は呼び出し側が渡す pin とする

## 4. 認証トークンとセキュリティ設計（#2245）

**コード変更なし・依存追加なし**。本節も §1〜§3 と同じく設計案の記録であり、確定と案を明示的に区別する。HF 固有の外部仕様は §3.5 と同じ扱いとし、一次出典 URL を確認できた項目にはその URL を付け、確認できなかった項目は「外部仕様・要確認」と明記して断定しない。

### 4.1 認証トークン

**供給経路と優先順位（案）**: 明示引数（例 `HfHub::with_token(HfToken)` 相当）＞ 環境変数 ＞ トークンファイル、の順で解決する。

- 環境変数名は `HF_TOKEN`（新しい既定名）と `HUGGING_FACE_HUB_TOKEN`（旧名）の 2 種、トークンファイルの既定パスは `$HF_HOME/token`（`HF_TOKEN_PATH` による上書きを許容）、暗黙トークン送信の無効化スイッチは `HF_HUB_DISABLE_IMPLICIT_TOKEN` 相当を、huggingface_hub 互換の案として挙げる。出典候補: https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables 。本 doc 作成時点では実装イシュー側での確認を要する「外部仕様・要確認」として扱う
- **暗黙トークン（環境変数・ファイル由来）を送るかどうか**は次の 2 案を記録する
  - (i) 常に送る（huggingface_hub の既定と同型）
  - (ii) 明示 opt-in の場合だけ送る
  - **推奨は (ii)**。安全側に倒し、公開 repo 取得で意図せずトークンを送らないようにする

**トークンファイルの読み取り条件**（§3.6「別クレートはファイル I/O をしない」の唯一の例外。読み取り専用かつキャッシュルート外限定）:

- 最終要素は `std::os::unix::fs::OpenOptionsExt::custom_flags(O_NOFOLLOW)` 相当で開き、symlink を拒否する。std の範囲内で実現できる見込みだが、`O_NOFOLLOW` 定数の取得に `libc` が要るかは実装時に確認する。要る場合は依存追加として承認事項へ回す（§4.4）
- サイズ上限（例: 数 KiB。既定値は実装イシューで確定）を設ける
- Unix では所有者とパーミッション（group/other 読み取り可を拒否するか警告にとどめるか）を案として記録し、実装イシューで推奨を確定する
- 非 Unix（Windows 等）の扱いは fail-closed（非対応）か別経路を用意するかを案とし、`docs/facade-model-registry-decision.md` §14 の Windows fail-closed 方針との整合を確認する

**トークン値の検証（A03 ヘッダインジェクション対策）**: 前後の空白だけを除去したあと、印字可能 ASCII 以外（CR/LF・NUL・制御文字）を含むものや空文字列を、接続前に拒否する。

**秘匿契約（確定させたい契約として記述）**:

- `HfToken` newtype を設け、`Debug` は手書きで伏字（例 `HfToken(<redacted>)`）にする。`Display`・`Serialize` は実装しない
- `HfError` のどのバリアントにもトークン値を含めない
- ログ・進捗コールバック・facade のキャッシュメタデータ（`download.json` 相当）にもトークン値を書かない
- トークンは送信対象の `Authorization` ヘッダ構築時にのみ参照する
- メモリ上のゼロ化（zeroize）は許容依存 9 区分に該当クレートが無いため保証しない。これは残留リスクとして明記し、本節でも新規依存は追加しない

**private repo の扱い**: facade の `download_with`／`DownloadOptions`（`docs/model-download-design.md` §5）にはヘッダを渡す経路が無く、さらに同 §6 の A09 節は `user:pass@host` 形式の URL を拒否する。したがって現行設計のままでは、トークンを転送処理に届けられないという構造上の制約がある。選択肢を次のとおり記録する。

- (α) `DownloadOptions` に「送信先ホストに束縛された Bearer 資格情報」を追加する（facade 公開面の拡張 → 承認事項）
- (β) 別クレートが自前の HTTP クライアントで転送し、facade 側に「検証済みバイト列をキャッシュへ書き込む」入口を追加する（facade 公開面の拡張 → 承認事項）
- (γ) 承認までは private／gated repo を fail-closed の非対応にする

**推奨は、初期状態を (γ) とし、(α)／(β) の選択をユーザー承認の分岐として残すこと**。トークンを HF JSON API（revision 解決・ファイル一覧）にだけ付けられるかは §2.3 案 A の HTTP 経路に依存するため、この点も承認判断時に併記する。

**応答コードの扱い**: private／gated／存在しない repo に対する 401・403・404 の返り方は「外部仕様・要確認」として記録する。区別できない場合でも、エラーはトークンを含まない型付き `HfError`（例: `Unauthorized`／`NotFound`／`Gated`）で fail-closed にする。

**private 内容のキャッシュ権限（残留リスクと申し送り）**: `docs/model-download-design.md` §6 (a) の `mkdirat(..., 0o755)` のままだと、共有キャッシュ上で private repo の内容が他ユーザーから読める。private 内容については 0o700 ディレクトリと 0o600 ファイルとする案を、#2088 実装イシューへの申し送りおよび facade 側の承認事項として記録する（§4.5）。

### 4.2 セキュリティ設計（OWASP Top 10 対応表の形で）

**取得先ホストの固定と HTTPS 限定（A02／A05／A10）**:

- API と `resolve` の接続先は `https://huggingface.co` に固定する。利用者がベース URL を任意に差し替える口（エンドポイント上書き）は初期スコープ外とする。ミラー対応は `docs/model-download-design.md` §10 に倣う
- リダイレクト先は、HTTPS であることに加えて、ホストの allowlist（`huggingface.co` と CDN 系ドメイン。CDN ドメイン名は §3.5 の 5 番目の項目と同じく要確認）で検証する
- リダイレクト回数に上限を設ける
- **ホストが変わるリダイレクトでは、`Authorization` ヘッダを必ず除去する**
- TLS 検証を無効化するオプションは設けない（`docs/model-download-design.md` §6 を継承）

**ホスト検証をどこで実行するか**: facade（`docs/model-download-design.md`）はリダイレクト追従を持つが、検証はスキームだけである。そのため次の案を記録する。

- (a) facade にリダイレクト方針のフック、またはホスト allowlist 入力を追加する（facade 公開面の拡張 → 承認事項）
- (b) 別クレートが自前クライアントでリダイレクトを 1 段ずつ解決し、最終 URL を facade に渡す

**(b) を採る場合の必須契約（facade へ渡した後のリダイレクトも検証する）**: facade（`docs/model-download-design.md`）はリダイレクト追従を持つがスキーム検証のみでホスト allowlist を適用しない。そのため (b) で別クレートが検証済みの最終 URL を渡しても、facade がその応答をさらにリダイレクトとして追従すると、別クレート側の allowlist を経由せず任意の HTTPS ホストへ接続しうる。`DownloadOptions`（§5。2026-09-24 追記）にはリダイレクト追従を無効化する上書き口が現状無く、タイムアウト・サイズ上限・リダイレクト回数と同様に実装側の既定値に固定されている。したがって、この抜け道を塞ぐ手段はどちらも facade 公開面の拡張を伴い、(a) と同じく承認事項になる。(b) を単独では採用せず、次のいずれかを実装イシューでの必須契約として (a) 側の承認事項に含める。

- (b-1) facade 側にリダイレクト追従の無効化スイッチ（`DownloadOptions` への非破壊追加）を設け、別クレートから渡された最終 URL 1 本にのみ接続し、その応答が返すリダイレクトには追従しないようにする
- (b-2) facade 側の追従リダイレクトにも別クレートと同一のホスト allowlist を適用する

(b-1)／(b-2) のどちらも満たさない場合は (b) を単独の解決策として採用しない（fail-closed）。推奨は実装イシューで確定するが、(b) を採る場合の注意も記録する: 署名付き CDN URL には有効期限があり、クエリ文字列が資格情報と同等の機微情報であることに留意する。

**署名付き URL は資格情報と同等に扱う（A09）**: CDN の署名付きクエリを、エラー（`ModelError::Http { url }` 相当）・進捗・`download.json.url` に残さないことを要求する。クエリを除去（redaction）し、保存する URL は `resolve` URL とする。これを #2088 実装イシューへの申し送りとして記録する（§4.5）。

**パスと revision の検証（A01／A03）**:

- 第 1 層（別クレート）は文字列検証とする
  - `repo_id` と `filename` の規則は §3.2 を参照
  - `revision` は ref 名（`/` を含みうる。`refs/pr/N` 等）を URL の path 要素にするとき、パーセントエンコードする
  - NUL・制御文字・`..`・先頭 `/` は拒否する
- **構造上の保証**: branch／tag 名はパス要素にしない。キャッシュキーに入るのは解決済みの 40 桁 16 進 commit sha だけである（§3.3 候補 1）。このため、revision 文字列からキャッシュルートを脱出する経路は型の上で存在しない。サーバーが返した commit sha も 40 桁 16 進検証を通してから使う
- 第 2 層は facade の dirfd 契約（`docs/model-download-design.md` §6 (a)〜(d)。`openat2`／`O_NOFOLLOW|O_DIRECTORY`・`mkdirat` の段階的作成）とする。別クレートはキャッシュに対して I/O を行わず、この層に完全に委ねる（§3.6 の例外を除く）

**ハッシュ検証の必須化（A08）**:

- 案 1（**推奨**）: HF 側の取得オプションで `Sha256Pin` を型レベルの必須引数とし、facade `download_with` に必ず pin を渡す
- 案 2: 呼び出し側 pin が無い場合、HF API が返す LFS sha256 を整合性検出のみの pin として使う。「真正性ではない」と明記する（§3.6「サーバーのハッシュは信頼根にしない」と整合させる）
- いずれの案でも、ハッシュ無しで取得を完了する経路は作らない。呼び出し側 pin と API が返す sha256 が不一致なら拒否する（fail-closed）

**その他**: 依存（A06）は §6（#2246）の承認に委ねる。ログを出す依存は作らない（A09。`docs/model-download-design.md` と同様）。SSRF の責務範囲はホスト固定により `docs/model-download-design.md` の一般ケースより狭い。

### 4.3 テスト方針（実装イシューの受け入れ基準案）

**ネットワーク非依存（CI で実行）**:

- トークン供給元の優先順位解決は、環境変数とファイルの参照を注入可能なクロージャまたはトレイトとして設計する。edition 2024 では `std::env::set_var` が `unsafe` で並列テストと相性が悪いため、プロセス環境そのものは変更しない
- トークンファイルについて、symlink 拒否・サイズ上限・パーミッション検査（`#[cfg(unix)]`）・CR/LF や制御文字を含むトークンの拒否をテストする
- センチネル文字列のトークン（本物に見えないダミー値。例 `<token>` 相当）を使い、全 `HfError` バリアントの `Display`／`Debug`・`HfToken` の `Debug`・進捗出力・書き込まれた manifest にセンチネルが現れないことを検査する
- リダイレクト方針（ホスト allowlist 判定・ホストが変わったときの `Authorization` 除去・回数上限・https 以外の拒否）を純関数の表駆動テストにする
- `repo_id`・`revision`・`filename` の許可と拒否を表駆動で検査する。サーバーが返した sha の 40 桁 16 進検証もテストする
- ローカル `TcpListener` の平文 HTTP モックは、`docs/model-download-design.md` §8「テスト経路の分離」と同様にスキーム検査より下の transport 層だけに使う。これで `Authorization` ヘッダがホスト固定先にだけ付くこと、別ホストへのリダイレクトで付かないことを確認する
- pin 必須化が型または実行時に効いていることを確認する（pin 無しでは取得 API を呼べない、または接続前に拒否される）

**`#[ignore]` の実通信**: 公開 repo の revision 解決・ファイル一覧・取得を扱う。private repo の取得は手動実行時に環境変数からトークンを与える。CI では実行しない（**workflow へ secrets を追加しない**。`.claude/rules/ci.md` の fork PR 対策）。実通信テストのログにトークンや署名付き URL を出さない。

### 4.4 #2245 由来の承認事項（列挙のみ・未実施）

- facade 公開面の拡張
  - (α) ホストに束縛された資格情報を `DownloadOptions` に追加する
  - (β) 検証済みバイト列の書き込み入口を追加する
  - リダイレクト方針のフック、ホスト allowlist 入力、またはリダイレクト追従の無効化スイッチ（(b-1)／(b-2)。§4.2「ホスト検証をどこで実行するか」）を追加する
- private 内容のキャッシュ権限（0o700／0o600）を facade 側の契約として変更すること
- 別クレートの HTTP 経路（§2.3）に関わる依存
- `O_NOFOLLOW` 定数取得のための OS 呼び出しラッパー（必要な場合のみ。`docs/model-download-design.md` §7 の該当項目と共有できるかは実装時に確認する）
- 新規 `unsafe` は不要の見込みである（トークンファイル読み取りは std の `OpenOptionsExt` の範囲で実現できる見込みのため）
- 網羅的な一覧化は §6（#2246）を正とし、本節には書かない

### 4.5 #2088 実装イシューへの申し送り

- 署名付き URL のクエリ除去（エラー・進捗・`download.json`）
- private 内容のパーミッション（0o700／0o600）
- ホスト固定とリダイレクト方針の受け渡し
- `Authorization` ヘッダの受け渡し経路（(α) を採った場合）

これらは `docs/model-download-design.md` 自体を編集せず、本節に記録するに留める。

## 5. 他ライブラリ対応表（#2246）

本節の正本は `docs/model-distribution-design.md` §5「HF hub 連携の方針（#2082 のスコープ外・ユーザー決定 2026-09-24）」であり、本節はその詳細な対応表である。分離方針（汎用取得・キャッシュ・pin 検証はコア／HF hub 固有処理は別クレート）が PyTorch・TensorFlow/Keras の既存分離と同型であることを、出典 URL 付きで示す。

### 5.1 観点別対応表

| 観点 | PyTorch | TensorFlow/Keras | 本設計（fandhe-ai） |
|---|---|---|---|
| コア側の汎用 URL 取得＋キャッシュ | `torch.hub.load_state_dict_from_url`（`torch.hub.load` が内部で使用。`~/.cache/torch/hub/checkpoints`）[^1] | `tf.keras.utils.get_file`（`~/.keras` キャッシュ）[^2] | facade `ModelRegistry`（実装済み・#2087）＋ #2088 の `download`／`download_with` 案（未承認・未実装） |
| ハッシュ検証 | `load_state_dict_from_url` の `check_hash`（ファイル名の SHA256 prefix とダウンロード結果を突合）[^1] | `get_file` の `file_hash` 引数（`md5`／`sha256` を指定可能）[^2] | `Sha256Pin` を型レベルの必須引数とする案（§4.2 案 1・#2088 §5） |
| ハブ連携の提供元 | コア（`torch` 本体）ではなく別パッケージ `huggingface_hub` | コア（`tf.keras`）ではなく別パッケージ `tensorflow_hub`。Keras 3 の `hf://` は利用者が `huggingface_hub` を別途インストールする構成で Keras 本体の必須依存ではない[^2][^3] | facade（コア）ではなく別クレート（本 doc §2） |
| HF Hub 連携の入口 | `PyTorchModelHubMixin`（`from_pretrained`／`save_pretrained`／`push_to_hub`）[^4] | Keras 3 `hf://<repo_id>` によるモデル読込・KerasHub の `from_preset`（設定・語彙込みの preset 一式を取得）[^3][^5] | `HfHub::download`（§3.1 疑似シグネチャ）案 |
| HF Hub 連携の低レベル取得 API | `hf_hub_download(repo_id, filename, revision, ...)`。個別ファイル 1 本をキャッシュへ取得する[^6] | （同上。`huggingface_hub` を介して同じ API を利用） | `HfHub::download`（§3.1） |
| revision 指定 | `hf_hub_download` の `revision` 引数（branch・tag・commit sha のいずれも可）[^6] | 同左 | `Revision`（`CommitSha`／`Ref`）から `resolve_revision` で commit sha へ解決（§3.1・§3.3） |
| 認証トークン | `huggingface_hub` の `token` 引数・`HF_TOKEN` 環境変数・トークンファイル | 同左 | §4.1 に記載済み（本節では重複記載しない） |
| キャッシュ配置 | `huggingface_hub` 独自のキャッシュレイアウト（`blobs`／`snapshots`／`refs`。外部仕様・本 doc では未実測） | 同左（`huggingface_hub` 経由） | `ModelRegistry` の `<name>/<version>` 単一ファイル配置（§3.3 候補 1。`blobs`／`snapshots`／`refs` 相当は持たない） |

[^1]: https://docs.pytorch.org/docs/2.14/hub.html
[^2]: https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
[^3]: https://huggingface.co/docs/hub/keras
[^4]: https://huggingface.co/docs/huggingface_hub/package_reference/mixins
[^5]: https://keras.io/keras_hub/ （KerasHub `from_preset`。preset の内訳・API 詳細は外部仕様・要確認。§5.3 も参照）
[^6]: https://huggingface.co/docs/huggingface_hub/guides/download （`tensorflow_hub` パッケージ自体の一次ドキュメントは https://www.tensorflow.org/hub を候補としたが、tfhub.dev は 2023-11 に Kaggle Models へ移行済み〈`docs/model-distribution-design.md` §5 出典〉のため個別取得 API の対応関係は huggingface_hub 側の記述に委ねる）

### 5.2 API 対応表

huggingface_hub（Python）の `hf_hub_download(repo_id, filename, revision)` [^6] と、本設計の疑似シグネチャ（§3.1。「この位置に記す想定である」と記載していた箇所を本節で回収する）との対応は次のとおり。

| `hf_hub_download` 引数 | 本設計での対応 | 差分・注記 |
|---|---|---|
| `repo_id`（`namespace/name` 形式の文字列） | `HfRepoId::parse` が返す `HfRepoId` 型（§3.1） | 検証規則（allowlist・長さ・`..`・`--` の扱い）は §3.2 に規定。文字列そのままではなく検証済み型を経由する |
| `filename`（リポジトリ内の相対パス） | `HfHub::download` の `filename: &str` 引数（§3.1） | huggingface_hub はキャッシュキーに `filename` を含む（`snapshots/<sha>/<filename>` 相当）のに対し、本設計は `ModelRegistry` の単一ファイル制約（§3.3）により `filename` をキャッシュキーへ反映しない。このため §3.3 で記録済みの是正案 A（`filename` を単一許容値へ限定）／B（`filename` を `name` の符号化へ組み込む）のいずれかを実装イシューで確定する必要がある（未確定のまま） |
| `revision`（branch・tag・commit sha のいずれも可の文字列。既定 `"main"`） | `Revision`（`enum { CommitSha(String), Ref(String) }`）を `Revision::parse` で構築し、`resolve_revision` で commit sha へ解決（§3.1） | huggingface_hub は revision を型で区別しないが、本設計は型で区別する案としている。区別せず一括で ref 解決を試みる方が近い可能性もあり、HF 側の branch／tag 解決経路が同一かは外部仕様・要確認（§3.5） |
| （戻り値）取得済みファイルのローカルパス | `CachedModel { name, version }`（§3.1） | huggingface_hub はファイルパスを直接返すのに対し、本設計は `ModelRegistry` のキー（`name`／`version`）を返し、パス解決は `ModelRegistry::load` に委ねる。§3.3 の是正が未確定な間は `filename` が戻り値に含まれない制約が残る |

### 5.3 差分・非対応

次はいずれも本設計（facade コア・HF hub 連携別クレートとも）の対象外とする。§2.1 の責務分割表・§7 のスコープ外と整合させて記す。

- **push 系（アップロード）API**: `huggingface_hub` の `push_to_hub`・`upload_file` 等に相当する機能。読み取り専用（取得のみ）に限定する（§2.1）
- **dataset／space リポジトリ**: `repo_type` が `model` 以外のリポジトリへの対応
- **分割 safetensors・`config.json` の取り扱い**: `ModelRegistry` は `<cache_dir>/<name>/<version>/model.safetensors` の単一ファイルしか扱わない（§3.3「単一ファイル制約」）。index json と複数 shard から成る分割 checkpoint、`config.json`、サブフォルダはいずれも対象外
- **private／gated repo**: 初期状態は fail-closed の非対応（§4.1 の推奨 (γ)）。(α)／(β) の選択は承認事項
- **「モデル定義まで含む」配布形態（KerasHub preset 等）**: KerasHub の `from_preset` は重みに加えトークナイザ・前処理設定・モデルアーキテクチャ定義までを一括取得する[^5]。本設計はモデル定義（`compat::Sequential` 構築ロジック）を含まず、重みファイル（safetensors）の取得＋`docs/huggingface-safetensors-interop-guide.md` §3 のレイアウト対応表による手動復元に限定する（§3.4）

## 6. 承認事項（#2246）

本節は HF hub 連携クレートに関する承認事項の正本であり、`docs/model-distribution-design.md` §2.6・§4.4 に個別に記載されていた項目をここへ一覧化する。**本節に記載の事項はすべて未取得・未実施**である（本 doc §1 の前提状態表と同じ扱い）。実施（依存追加・クレート新設・facade 公開面拡張等）は本 doc では一切行わない。

### 6.1 承認事項一覧

| # | 区分 | 承認事項 | 出所 | #2088 §7 との関係 | 備考 |
|---|---|---|---|---|---|
| 1 | 依存追加 | HTTP クライアント＋TLS スタックを別クレートの依存として追加すること（§2.3 案 A） | §2.3・§2.6-8 | #2088 §7-1（第 10 区分相当）と**同じクレート選定・同じ deps-policy 区分を再利用しうるが、承認単位は別**。`docs/model-download-design.md` §3 の候補（`ureq`／`minreq`／`attohttpc`。`reqwest` は非推奨）を参照する | 依存宣言そのものは #2088 とは別のクレート（別 `Cargo.toml`）に対するものであり、#2088 の承認が本項の承認を兼ねない |
| 2 | 依存追加 | 証明書ストア（`webpki-roots`／`rustls-native-certs`）・`ring` のライセンス式確認（`cargo tree` 実測） | §2.3 | #2088 §3 の実測対象と共有 | 承認後に実測。推定で適合と記述しない（deps-policy.md） |
| 3 | 依存追加 | OS 呼び出しラッパー（`libc`／`rustix`）をトークンファイル読み取り（`O_NOFOLLOW`）用に追加すること（必要な場合のみ） | §4.1・§4.4 | #2088 §7-5（第 11 区分相当）と区分を共有しうるが、要否は実装時に確認する | `std::os::unix::fs::OpenOptionsExt::custom_flags` の範囲で完結する可能性もあり、その場合は新規依存不要 |
| 4 | 依存追加（不要の確認） | JSON 解析は `serde`／`serde_json`（既存の許容区分）を workspace 固定版のまま使う。新規区分は不要 | §2.6-8 | 該当なし | URL のパーセントエンコードは新規依存にせず自作する（依存を増やさない） |
| 5 | 依存追加（見送り） | `zeroize` は追加しない（メモリ上のゼロ化は保証しない残留リスクとして据え置く） | §4.1 | 該当なし | 新規依存の追加を伴わない現状維持の確認事項 |
| 6 | 依存追加（非推奨の確認） | crates.io の既存 `hf-hub` クレートを依存として使う案は非推奨（§2.5）。依存ツリー・ライセンスは未実測のまま | §2.5 | 該当なし | 独自キャッシュ・HTTP スタックを持ち facade の pin 検証・dirfd 契約を迂回するため |
| 7 | 依存追加（運用影響） | 上記 1〜3 の依存追加に伴う `.claude/rules/deps-policy.md` の区分追加・`docs/license-matrix.md` の行追加・`deny.toml`・`scripts/check-forbidden-deps.sh` への影響 | §2.6-8 | #2088 §7-3 と同じ運用 | 依存追加とセットで行う（deps-policy.md） |
| 8 | workspace 追加 | 新規クレートをルート `Cargo.toml` の `members` へ追加すること（ディレクトリ名・`[package] name` の決定を含む） | §2.6-1・§2.6-2 | 該当なし | ディレクトリ名案は `crates/hf-hub`（§2.4） |
| 9 | workspace 追加（運用影響） | 上記 8 に伴う CLAUDE.md「想定クレート 10 個＋docs-site」記述の更新、`.claude/rules/delegation-impl.md` の委譲マッピング（担当 builder）の更新 | §2.4 | 該当なし | 承認後に実施 |
| 10 | 公開範囲 | crates.io 公開の要否（8 件目・`fandhe-ai-` prefix）／`publish = false`／workspace 外の 3 択（§2.4） | §2.4・§2.6-3 | 該当なし | 公開する場合は名前の空き確認（第三者の `hf-hub` と衝突しないか）を承認後に `docs/crates-io-naming-decision.md` の手順で実施。現時点で未実測 |
| 11 | 公開範囲（運用影響） | crates.io 公開を選ぶ場合の `release-all.yml`・`docs/crates-io-publishing-order.md` への追加 | §2.4 | 該当なし | 上記 10 の決定に従属 |
| 12 | 公開範囲 | `docs/compat-api-scope.md` §0「facade が唯一のサポートされる公開 API 面」との整合の扱い（crates.io 公開を選ぶ場合、2 つ目の公開面が生じる） | §2.4・§2.6-6 | 該当なし | #2194 への申し送り（§6.2）とも関連 |
| 13 | HTTP 経路 | HTTP 経路の選択（§2.3 案 A〈推奨〉／案 B／案 C） | §2.3・§2.6-4 | 案 A 採用時は依存が上記 1 と重なる | 推奨は案 A。確定はユーザー承認による |
| 14 | facade 公開面拡張（前提共有） | #2088 §7-2（`download_with`・`DownloadOptions`・`Sha256Pin` 等）は本設計の前提として共有する | §2・§4 | #2088 §7-2 と**共有**（別クレートはこの機能に依存する設計） | #2088 自体が未承認のため、本設計の実装着手も #2088 承認・実装を待つ（本 doc §1「前提チェーン」） |
| 15 | facade 公開面拡張 | §2.3 案 B を採る場合のみ: facade が「HTTPS GET でバイト列／JSON を返す」プリミティブを公開すること | §2.3 | 該当なし | 案 A 推奨のため通常は不要 |
| 16 | facade 公開面拡張 | §4.1 (α)：`DownloadOptions` にホストへ束縛された Bearer 資格情報を追加すること | §4.1・§4.4 | 該当なし | private repo 対応の選択肢の 1 つ |
| 17 | facade 公開面拡張 | §4.1 (β)：別クレートが自前 HTTP クライアントで転送し、facade 側に「検証済みバイト列をキャッシュへ書き込む」入口を追加すること | §4.1・§4.4 | 該当なし | private repo 対応の選択肢のもう 1 つ |
| 18 | facade 公開面拡張 | §4.2 のリダイレクト方針フック・ホスト allowlist 入力、または追従無効化スイッチ（(b-1)／(b-2)） | §4.2・§4.4 | 該当なし | (b) を単独採用する場合の必須契約（§4.2） |
| 19 | facade 公開面拡張 | private 内容のキャッシュ権限を 0o700（ディレクトリ）／0o600（ファイル）へ変更すること | §4.1・§4.5 | #2088 実装イシューへの申し送り（§4.5）と対応 | 現行の `mkdirat(..., 0o755)` のままだと共有キャッシュ上で private repo 内容が他ユーザーから読める（残留リスク） |
| 20 | 新規 `unsafe` | 新規 `unsafe` は不要の見込み（トークンファイル読み取りは std の `OpenOptionsExt` の範囲で実現できる見込みのため） | §2.6-7・§4.4 | 該当なし | 「承認不要」の確認ではなく、実装時に不要と判明すれば承認事項から外れることの記録 |
| 21 | 実装着手 | 承認後の実装イシューの起票（`.claude/rules/out-of-scope-tracking.md` により起票自体もユーザー承認が必要） | §2.6-9 | #2088 §7-4 と同じ運用 | 本 doc・本イシューでは起票しない |

**#2088 の位置づけについての注記**: #2088 はイシューとしてはクローズ済みだが、`docs/model-download-design.md` §7 の承認事項は本 doc §1 の前提状態表と同じく「未取得」として扱う（同 doc は編集しない）。

### 6.2 #2194 への申し送り

`docs/spec/` 提案の起草に着手する際、次の文言整合を確認するよう申し送る（本節は記録のみであり、#2194 へのコメント・spec 側の編集はここでは行わない）。

- 提案する文言: 「モデルハブ: コア（facade）では非対応。特定ハブ（Hugging Face Hub）連携は別クレートで提供する（汎用 URL 取得＋キャッシュ＋ハッシュ検証はコアの責務〈#2088〉）」
- 根拠の所在: `docs/model-distribution-design.md` §5 と本 doc §5
- **気づいた不整合**: `docs/python-binding-tf-format-non-target-spec-proposal.md` §6 は「モデルハブは #1962（`docs/facade-inference-serving-scope-decision.md`）で整理済み」としているが、同 doc（`facade-inference-serving-scope-decision.md`）にはモデルハブの記述が見当たらない（見出し・本文ともに hub／ハブの記載なし）。#2194 の起草時は `model-distribution-design.md` §5 と本 doc を参照するよう申し送る。当該 doc（`python-binding-tf-format-non-target-spec-proposal.md`）の修正はスコープ外とし、本節への記録にとどめる

## 7. スコープ外・未決事項

### スコープ外

- push（アップロード）系 API
- 依存の詳細な列挙（§6／#2246 で完了済み）
- 依存のライセンス実測（承認後の実装イシューで実施。`docs/model-download-design.md` §8-1 と同じ手順）
- dataset／space リポ
- 分割 safetensors と `config.json` の取り扱い（§3.3）
- private／gated repo の取得（§4.1 の推奨は初期状態 (γ) の fail-closed 非対応。(α)／(β) の選択は承認事項）
- 実装そのもの

### 未決事項

- HF 命名規則の確認（§3.2）
- オフライン時の revision 解決（§3.3）
- 公開区分（§2.4・§6）
- HTTP 経路の選択（§2.3・§6）
- 認証トークンの環境変数名・トークンファイル既定パス・暗黙送信無効化スイッチの外部仕様確認（§4.1）
- 401／403／404 の返り方の外部仕様確認（§4.1）
- CDN リダイレクト先ドメインの allowlist 確定（§4.2）
- private repo 対応方式（(α)／(β)）の選択（§4.1）
- ハッシュ検証の pin 必須化案（案 1／案 2）の確定（§4.2）

## 8. 出典一覧

- 内部 docs: `docs/model-distribution-design.md`（§5）・`docs/model-download-design.md`（§3〜§7）・`docs/facade-model-registry-decision.md`（§7・§13・§14）・`docs/facade-safetensors-exposure-decision.md`（§11）・`docs/huggingface-safetensors-interop-guide.md`（§3）・`docs/compat-api-scope.md`（§0）・`docs/crates-io-naming-decision.md`
- 規約: `.claude/rules/deps-policy.md`・`.claude/rules/security.md`
- コード: `crates/facade/src/model.rs`（`ModelRegistry` 公開面）・`crates/facade/tests/api_surface.rs`（依存形状固定テスト）
- イシュー: #2243・#2244・#2245・#2246・#2088・#2087・#2194
- 外部 URL: https://docs.pytorch.org/docs/2.14/hub.html （PyTorch Hub）・https://huggingface.co/docs/huggingface_hub/guides/integrations （huggingface_hub integrations）・https://huggingface.co/docs/hub/keras （Keras at HF）・https://huggingface.co/docs/hub/api （HF Hub API ドキュメント）・https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables （huggingface_hub 環境変数リファレンス。トークン供給経路の案の出典。§4.1）・https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file （`tf.keras.utils.get_file`。§5.1）・https://blog.tensorflow.org/2023/03/tensorflow-hub-kaggle.html （tfhub.dev の Kaggle Models 移行。§5.1）・https://huggingface.co/docs/huggingface_hub/package_reference/mixins （`PyTorchModelHubMixin`。§5.1）・https://huggingface.co/docs/huggingface_hub/guides/download （`hf_hub_download`。§5.1・§5.2）・https://keras.io/keras_hub/ （KerasHub `from_preset`。§5.1・§5.3。preset 内訳の詳細は外部仕様・要確認）
