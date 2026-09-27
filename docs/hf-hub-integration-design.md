# HF hub 連携クレートの境界と取得 API 案の設計判断記録（#2244）

- イシュー #2244（親 #2243「HF hub 連携クレートの設計判断記録と依存追加の承認申請」・Phase 親 #2131）
- **コード変更なし・依存追加なし**。本 doc は設計案の記録であり、確定はユーザー承認後
- 基準コミット: 作業ブランチ作成時点の `origin/main`（`9fe4b523`。2026-09-27）
- 結論の要約: 本 doc は §1〜§3 に境界と API 案を記す設計案であり、正式決定ではない。§4〜§6 は兄弟イシュー #2245（認証トークン・セキュリティ）・#2246（他ライブラリ対応表・承認事項一覧）が追記する予約節

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
| #2245 | 認証トークン・セキュリティ設計 | 本 doc §4 へ追記予定 |
| #2246 | 他ライブラリ対応表・承認事項一覧 | 本 doc §5・§6 へ追記予定 |

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
| push 系・認証・dataset／space リポ | いずれにも入れない（対象外。認証は §4／#2245） |

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

huggingface_hub（Python）の `hf_hub_download(repo_id, filename, revision)` との対応関係をこの位置に記す想定である。

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
- キャッシュ書き込みは facade の dirfd 契約に任せ、別クレートではファイル I/O をしない
- HTTPS 限定はリダイレクト先にも及び、TLS 検証を無効化するオプションは設けない（`docs/model-download-design.md` §6 を継承）
- サーバーが返すハッシュ（etag 系）は信頼根にせず、真正性の根拠は呼び出し側が渡す pin とする

## 4. 認証トークンとセキュリティ設計（#2245 で追記）

（予約節。#2245 のマージにより本節へ追記される）

## 5. 他ライブラリ対応表（#2246 で追記）

（予約節。#2246 のマージにより本節へ追記される）

## 6. 承認事項（#2246 で追記）

（予約節。#2246 のマージにより、本 doc §2.6 の境界関連項目を含めて承認事項が一覧化される）

## 7. スコープ外・未決事項

### スコープ外

- push（アップロード）系 API
- 認証トークン（§4／#2245）
- 依存の詳細な列挙とライセンス実測（§6／#2246）
- dataset／space リポ
- 分割 safetensors と `config.json` の取り扱い（§3.3）
- 実装そのもの

### 未決事項

- HF 命名規則の確認（§3.2）
- オフライン時の revision 解決（§3.3）
- 公開区分（§2.4）
- HTTP 経路の選択（§2.3）

## 8. 出典一覧

- 内部 docs: `docs/model-distribution-design.md`（§5）・`docs/model-download-design.md`（§3〜§7）・`docs/facade-model-registry-decision.md`（§7・§13・§14）・`docs/facade-safetensors-exposure-decision.md`（§11）・`docs/huggingface-safetensors-interop-guide.md`（§3）・`docs/compat-api-scope.md`（§0）・`docs/crates-io-naming-decision.md`
- 規約: `.claude/rules/deps-policy.md`・`.claude/rules/security.md`
- コード: `crates/facade/src/model.rs`（`ModelRegistry` 公開面）・`crates/facade/tests/api_surface.rs`（依存形状固定テスト）
- イシュー: #2243・#2244・#2245・#2246・#2088・#2087・#2194
- 外部 URL: https://docs.pytorch.org/docs/2.14/hub.html （PyTorch Hub）・https://huggingface.co/docs/huggingface_hub/guides/integrations （huggingface_hub integrations）・https://huggingface.co/docs/hub/keras （Keras at HF）・https://huggingface.co/docs/hub/api （HF Hub API ドキュメント）
