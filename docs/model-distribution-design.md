# モデル配布機構（モデルハブ相当）の設計判断記録（#2082）

イシュー #2082「docs(facade): 配布機構（モデルハブ相当）の設計判断記録」。親: #2059（Phase 親「役割・機能の対応表の『ある』以外を埋める」）。ルート: #2058。

本ドキュメントは **コード変更を伴わない設計記録のみ**である。`crates/**`・`docs/spec/`（正本 submodule）・依存（`Cargo.toml`／`Cargo.lock`）・ガードレール閾値・数値一致許容誤差（tolerance／baseline）はいずれも変更しない。

学習済みモデル・プリトレイン重みの配布方式（ローカルレジストリ・リモート取得・HF hub 連携）を整理する親 issue として、子 issue の成果物を重複転記せず要点の要約＋リンクで束ねるハブ文書とする。個々の設計判断の正本は各子 doc（`docs/facade-model-registry-decision.md`・`docs/model-download-design.md`）であり、本文書と内容が食い違う場合は各子 doc を正とする。

## 1. 背景

対応する他フレームワークの機能: PyTorch `torch.hub`（`torch.hub.load`・`load_state_dict_from_url`）・TensorFlow/Keras `tf.keras.utils.get_file`／`keras.applications`・Hugging Face `huggingface_hub`（`hf_hub_download`・`PyTorchModelHubMixin`）。いずれも「プリトレイン重みをローカルキャッシュへ取得し、名前・バージョンで一元管理してロードする」入口を提供する。

本リポジトリの推論・fine-tuning シナリオでの配布要件:

- **推論**: 利用者が学習済み重み（`model.safetensors`）を一度取得すれば、以後は再ダウンロードなしにロードできること（ローカルキャッシュ）。
- **fine-tuning**: HF hub 等で配布されるプリトレイン重みを起点に `compat::Sequential::load_state_dict` へ繋ぎ込めること。facade には既に safetensors save／load（`fandhe_ai::interop::safetensors`。`docs/facade-safetensors-exposure-decision.md`。#2019）と `ModelCheckpoint::to_file`（#2073）があるが、「ホームディレクトリ配下で名前・バージョンごとに重みを一元管理し、同期ロードする」入口がなかった。
- **配布元の多様性**: 手動配置（ローカルレジストリ。#2087）・HTTP(S) URL からの明示ダウンロード（#2088）・HF hub のリポジトリ形式参照（facade には入れない別クレート。#2243）の 3 系統を想定する。facade（コア）が扱うのは前 2 者の汎用機構のみで、HF hub 固有の URL 変換・API 呼び出しは含まない（§5 参照）。いずれも数値経路（`Op`／`BackendOps`／VJP／カーネル）には一切触れず、ホスト側のファイル I/O と既存 `interop::safetensors` への委譲のみで完結する設計とし、REQ-1（完全自作コア）・REQ-12（利用者向け制御 API の限定）と抵触しない。

根拠: `docs/facade-inference-serving-scope-decision.md`（推論・サービング周辺のスコープ整理。モデル配布はここでは論点化されていないが、KV キャッシュ・トークナイザと同様「既存演算・既存フォーマット処理の合成に留める」方針を踏襲する）。

## 2. 配置案

配置の詳細（ディレクトリレイアウト・manifest・version 管理）は子 doc が確定済みであり、本節では要点のみを要約する。詳細・実装記録は各リンク先を参照する。

### 2.1 ローカルキャッシュ・ディレクトリレイアウト

`docs/facade-model-registry-decision.md`（#2087・実装済み）が確定・実装したレイアウト:

```
<cache_dir>/                      既定 $HOME/.fandhe-ai/models
                                   （Windows は $USERPROFILE。ルート解決のみ動作。下記注記参照）
  <name>/                         [A-Za-z0-9._-]+（先頭 '.' 不可）
    <version>/                    同上。semver 等の記法は解釈しない不透明な文字列
      model.safetensors           F32 テンソルのみ・キー = state_dict キー
```

`fandhe_ai::model::ModelRegistry::load(name, version)` で同期ロード（`HashMap<String, Tensor<f32>>` を返す。受入条件字面〈単一 `Tensor`〉からの変更理由は同 doc §5）・`available_models()` で一覧取得する読み取り専用レジストリであり、ディレクトリの作成・削除は一切行わない。配置（ダウンロード・手動コピー）は利用者側または §2.2 のリモート取得機構が担う。

**Windows 対応状況（重要）**: `ModelRegistry::new()` によるキャッシュルート解決（`$USERPROFILE/.fandhe-ai/models`）自体は Windows でも動作するが、`load`・`available_models` は **Windows では fail-closed で非対応**である。シンボリックリンク経由のキャッシュルート脱出を安全に防ぐ no-follow オープン実装（`dev`／`ino` 識別子照合）が Linux／macOS 限定で、Windows 向けの安全な実装（`file_index`／`volume_serial_number` 照合）が未確立なため。`load` は Windows では常に `Err(ModelError::Io)` を返し、`available_models` は常に空の一覧を返す（エラーにはならない）。出典: `docs/facade-model-registry-decision.md` §13・§14、`crates/facade/src/model.rs` モジュール doc「Windows 対応状況」節。

### 2.2 リモート取得・manifest

`docs/model-download-design.md`（#2088・設計記録のみ・コード未実装）が整理した要点:

- 公開 API 案: `ModelRegistry::download(&self, url: &str, name: &str, version: &str) -> Result<(), ModelError>`。
- **manifest（キャッシュ有効判定）**: `<name>/<version>/download.json`（`serde_json`。許容依存区分内）に `url`・`etag`・`content_length`・`sha256`・`fetched_at` を保存し、条件付き GET（`If-None-Match`）で再取得の要否を判定する。304 応答時も `sha256` によるローカル再検証を必須とし、無条件の再利用シグナルとしては扱わない（改竄・破損対策。A08）。
- **書き込みの原子性**: 一時ファイルへストリーミング書き込み → 検証成功後に `rename` で公開する、既存の safetensors 保存（`st_save`）と同型の契約。シンボリックリンク経由のキャッシュルート脱出対策として dirfd 相対操作（`openat`／`openat2`／`renameat`）による構造的防御を設計済み（未実装。§7 参照）。
- **version 管理**: version 文字列は semver 等の記法を解釈しない不透明な識別子として扱う（§2.1 と同一方針）。「最新版」解決・記法の意味論付けはスコープ外（`docs/facade-model-registry-decision.md` §9・`docs/model-download-design.md` §10）。

## 3. 子 issue 役割分担

| 役割 | 子 issue | 状態 | 成果物 |
|---|---|---|---|
| ローカルレジストリ | #2087 | CLOSED（実装済み） | `crates/facade/src/model.rs`（`ModelRegistry`／`ModelError`）・`docs/facade-model-registry-decision.md` |
| リモート取得（URL ダウンロード） | #2088 | 設計記録のみ（依存追加のユーザー承認待ち） | `docs/model-download-design.md`（HTTP クライアント候補比較・技術仕様案・OWASP セキュリティ設計・承認後の実装手順） |
| HF hub 連携 | #2082 のスコープ外。別クレートとしての設計は #2243（#2244〜#2246 に分解済み）で追跡 | #2243 OPEN | `docs/model-distribution-design.md` §5（本節） |

ローカルレジストリ（#2087）はリモート取得（#2088）がキャッシュへ書き込む先として設計されており、両者は「入口の分離」（配置元に依らず `model.safetensors` レイアウトへ収束させる）という一貫した構成を取る。HF hub 連携は §5 のとおり別クレート（または opt-in の別経路）の責務であり、#2082 のスコープ外として #2243（#2244 境界と取得 API 案・#2245 認証トークンとセキュリティ・#2246 他ライブラリ対応表と依存承認事項）へ追跡を引き継ぐ。

## 4. facade API 契約案

`docs/model-download-design.md` §7「承認事項（本イシュー時点ではいずれも未取得）」が列挙する 5 項目のうち、**依存追加のユーザー承認事項は次の 2 件**である（いずれも許容依存 9 区分〈`.claude/rules/deps-policy.md`〉に属さない新規区分の追加を要する）。**両方とも未承認であり、本ツリー（#2082 系列）では実施しない**。

1. **HTTP クライアント（＋ TLS スタック）の新規区分追加**（現行の許容依存 9 区分に続く第 10 区分相当）。`ModelRegistry::download` の実装に必須。候補は同期専用の `ureq`／`minreq`／`attohttpc`（`reqwest` は `tokio` を推移的に引き込むため非推奨。`docs/model-download-design.md` §3）。ライセンス（推移的依存を含む）は未実測のまま「承認後に `cargo tree` 実測を行う」と記録されている。
2. **キャッシュ書き込みの dirfd 相対操作用 OS 呼び出しラッパー（`libc` または `rustix` 等）の新規区分追加**（1 の第 10 区分に続く第 11 区分相当。両者は承認単位が異なるため別区分として扱う）。`std::fs` は dirfd 相対のオープン・rename を提供しないため、シンボリックリンク経由のキャッシュルート脱出対策（TOCTOU を構造的に閉じる設計。`docs/model-download-design.md` §6）の実装に必須。

上記 2 件に付随して、`docs/model-download-design.md` §7 は facade 公開面の拡張（`ModelRegistry::download` および進捗コールバック型の追加・`api_surface.rs` 到達性テストの追加）も承認事項として列挙しているが、これは依存追加そのものではなく、1・2 の承認を前提に実施する公開面拡張である。`docs/license-matrix.md` への行追加・承認後の実装イシュー起票（`.claude/rules/out-of-scope-tracking.md` によりユーザー承認が必要）も同様に 1・2 の後続事項として同 doc §7 に記録されている。

## 5. HF hub 連携の方針（#2082 のスコープ外・ユーザー決定 2026-09-24）

ユーザー決定（2026-09-24）「他ライブラリと同じにする」に基づき、HF hub（Hugging Face Hub）連携は本 issue（#2082）のスコープ外とする。方針は次のとおり: **汎用の URL ダウンロード＋ローカルキャッシュ＋ハッシュ検証はコア（facade）側の責務**とし（§2.2 のリモート取得機構がこれに当たる）、**特定ハブ（Hugging Face Hub）連携はコアに入れず、別クレート（または opt-in の別経路）の責務**とする。この別クレートとしての設計・依存承認事項は、ユーザー指示により起票済みの #2243「docs(interop): HF hub 連携クレートの設計判断記録と依存追加の承認申請」（親 #2131〈Phase 5〉配下。#2244「境界と取得 API 案」・#2245「認証トークンとセキュリティ」・#2246「他ライブラリ対応表と依存承認事項」へ 2h 以下の粒度で分解済み。#2245・#2246 は #2244 に依存）で追跡する。

根拠として、PyTorch・TensorFlow/Keras とも同型の分離を採用している:

- **PyTorch**: コア（`torch` 本体）は `torch.hub`（`hubconf.py` によるリポジトリ規約・`torch.hub.load`・`load_state_dict_from_url` の `~/.cache/torch/hub/checkpoints` キャッシュと `check_hash`〈ファイル名の SHA256 prefix 突合〉）を提供する。HF hub 連携は torch 本体ではなく別パッケージ `huggingface_hub` の `PyTorchModelHubMixin` 等が提供する。
  - 出典: https://docs.pytorch.org/docs/2.14/hub.html ・ https://huggingface.co/docs/huggingface_hub/guides/integrations
- **TensorFlow/Keras**: コア（`tf.keras`）は `tf.keras.utils.get_file`（`~/.keras` キャッシュ・`file_hash` 検証）を提供し、`keras.applications` の重み取得もこの上に構築されている。`tensorflow_hub` は別パッケージ（tfhub.dev は 2023-11 に Kaggle Models へ移行済み）。Keras 3 の `hf://` 対応は、利用者が `huggingface_hub` を別途インストールして使う構成であり、Keras 本体の必須依存ではない。
  - 出典: https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file ・ https://blog.tensorflow.org/2023/03/tensorflow-hub-kaggle.html ・ https://huggingface.co/docs/hub/keras

この分離は、本リポジトリの許容依存 9 区分（`.claude/rules/deps-policy.md`）が「必要最小限・`=x.y.z` 完全固定・ユーザー承認必須」を旨とする方針とも整合する。HTTP クライアント（§4-1）はコア（facade）が汎用ダウンロード機構として使う可能性があるため許容依存の新規区分として検討対象になりうるが、HF hub 専用の API 形状（リポジトリ ID・revision・LFS リダイレクト等）へ特化した処理はコアへ持ち込まない。

**将来 HF hub 連携を実装する場合**は、別クレートとして新設し、依存追加（HTTP クライアント等。§4-1 と同一区分を再利用できる可能性はあるが、クレート境界とユーザー承認は別途必要）はユーザー承認が前提であることを明記する。具体的な境界・API 案・認証トークンやセキュリティの検討・他ライブラリ対応表と依存承認事項の整理は #2243（#2244〜#2246）側で行い、本ドキュメントでは重複記載しない。

## 6. 契約整理（守るべき既存契約）

- **REQ-1 完全自作コア・許容依存 9 区分**: ローカルレジストリ（#2087）は既存 `interop::safetensors` への委譲のみで新規依存なし。リモート取得（#2088）は新規依存を要するため未承認のまま非実施。HF hub 連携（§5）は #2082 のスコープ外で、別クレート（#2243）側で依存追加の承認申請を行う。
- **security A03（パストラバーサル）**: `name`／`version` の allowlist 検証（`docs/facade-model-registry-decision.md` §7）・シンボリックリンク経由のキャッシュルート脱出対策（同 doc §12・§13、`docs/model-download-design.md` §6）はいずれもレジストリ・ダウンロード双方の共通契約として確立済み（後者は設計のみで未実装）。
- **`docs/compat-api-scope.md` §0 サポート境界**: `facade` が唯一のサポートされる公開 API 面。`ModelRegistry` の新規公開面はいずれも `crates/facade/tests/api_surface.rs` の機械固定・ユーザー承認の対象。
- **プラットフォーム差（Windows）**: `ModelRegistry::new()`（キャッシュルート解決）は Windows でも動作するが、`load`・`available_models` は Windows では fail-closed 非対応（§2.1 注記・`docs/facade-model-registry-decision.md` §13・§14）。リモート取得（#2088）の dirfd 相対書き込み契約も同じ理由で Windows 非対応として設計されている（`docs/model-download-design.md` §6 (c)）。

## 7. 対象外（本ドキュメントでは確定しない・実施しない事項）

- §4 に列挙した依存追加 2 件・facade 公開面拡張（`ModelRegistry::download` 等）の実施そのもの（未承認のため次の子 issue へ引き継ぐ。`docs/model-download-design.md` §9 に起票草案あり・未起票）
- HF hub 連携の実装（§5。#2082 のスコープ外。別クレートとしての設計は #2243〈#2244〜#2246〉で追跡）
- version 記法（semver・hash 等）の解釈・最新版解決（`docs/facade-model-registry-decision.md` §9）
- `compat::Sequential` へのレジストリ直結ラッパー（`Sequential::from_registry` 等）
- 非 F32 dtype・ファイルサイズ上限の値そのものの見直し（現行 1 GiB。`docs/facade-model-registry-decision.md` §15）

## 8. 出典一覧

- `docs/facade-model-registry-decision.md`（#2087・実装済み・正本）
- `docs/model-download-design.md`（#2088・設計記録のみ・正本）
- `docs/facade-inference-serving-scope-decision.md`（推論・サービング周辺スコープの姉妹整理）
- `docs/facade-safetensors-exposure-decision.md`（safetensors 公開面の正本）
- `.claude/rules/deps-policy.md`・`.claude/rules/security.md`・`.claude/rules/out-of-scope-tracking.md`
- イシュー #2082／#2087／#2088・親 #2059・ルート #2058
- HF hub 連携の追跡先: イシュー #2243（#2244／#2245／#2246。親 #2131〈Phase 5〉）
