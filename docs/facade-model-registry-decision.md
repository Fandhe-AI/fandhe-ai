# ローカルモデルレジストリの設計判断記録（#2087）

イシュー #2087「feat(facade): ローカルモデルレジストリ実装」。親: #2082。ルート: #2058（Phase 親 #2059）。

## 1. 背景・目的

facade には safetensors save／load（`fandhe_ai::interop::safetensors`。#2019）と `ModelCheckpoint::to_file`（#2073）があるが、「ホームディレクトリ配下に手動配置したプリトレイン重みを名前・バージョンで一元管理し、同期ロードする」入口がなかった。本イシューはこの入口として `fandhe_ai::model::ModelRegistry` を実装し、ディレクトリレイアウトを規定する。

リモート取得（#2088。HTTP クライアント依存の承認待ち）は本イシューのレジストリへキャッシュする側として後続で接続する想定であり、本イシューのスコープには含まない。数値経路（`Op`／`BackendOps`／VJP／カーネル）には一切触れず、ホスト側のパス管理と既存 `interop::safetensors` への委譲のみで完結する（REQ-12 と矛盾しない）。

## 2. 承認ゲートの確認

- `docs/facade-inference-serving-scope-decision.md`・`docs/compat-api-scope.md` に「モデル配布／レジストリ／プリトレイン重み」を段階 0 や承認待ちとして固定する記述はない（実装時点で grep 実測）
- Phase 親 #2059・ルート #2058・親 #2082・本イシュー #2087 いずれにもコメント（承認・保留）はなかった
- 先例として PR #2217（#2073）は決定記録 doc による保留指定がなく、受入条件に列挙された公開面に限定して実装している。本イシューも同型（doc によるゲートなし・受入条件が公開面を列挙）と判断し、**受入条件に列挙された公開面＋テスト可能性に必要な最小の構築子のみ**を実装した

## 3. ディレクトリレイアウト（規定）

```
<cache_dir>/                      既定 $HOME/.fandhe-ai/models
                                   （Windows は $USERPROFILE。ModelRegistry::new 参照）
  <name>/                         [A-Za-z0-9._-]+（先頭 '.' 不可）
    <version>/                    同上。semver 等の記法は解釈しない不透明な文字列
      model.safetensors           F32 テンソルのみ・キー = state_dict キー（転置・リネームなし。REQ-7）
```

配置は利用者（または将来のダウンロード側。#2088）が行う。`ModelRegistry` はディレクトリの作成・削除を一切行わない読み取り専用のレジストリである。

## 4. 公開面（承認事項。マージ前に確認されたい）

facade 新規公開面 7 件（`crates/facade/src/model.rs`。`crates/facade/tests/api_surface.rs::model_module_exposes_only_approved_surface` が機械固定）:

1. `fandhe_ai::model::ModelRegistry`（構造体。フィールドは private）
2. `fandhe_ai::model::ModelError`（`#[non_exhaustive]` enum。`Debug`／`Display`／`std::error::Error` 実装）
3. `ModelRegistry::new() -> Result<Self, ModelError>`（既定ルート `$HOME/.fandhe-ai/models`。`HOME`／`USERPROFILE` 未設定時は `ModelError::CacheDirUnavailable`）
4. `ModelRegistry::with_cache_dir(dir: impl Into<PathBuf>) -> Self`（ルート明示指定。統合テスト・CI・非標準配置向け）
5. `ModelRegistry::cache_dir(&self) -> &Path`
6. `ModelRegistry::load(&self, name: &str, version: &str) -> Result<HashMap<String, Tensor<f32>>, ModelError>`
7. `ModelRegistry::available_models(&self) -> Vec<(String, Vec<String>)>`

## 5. `load` 戻り値型の判断（受入条件字面からの逸脱）

イシュー受入条件の字面は `load` の戻り値要素型を単一 `Tensor` とするが、本実装は `HashMap<String, Tensor<f32>>`（state dict）を返す。理由:

- safetensors ファイルは複数テンソルを保持する（レイアウトが規定する `model.safetensors` は state dict 全体を 1 ファイルに収める設計）
- 消費先 `compat::Sequential::load_state_dict` も `HashMap<String, Tensor<f32>>` を取る
- 単一 `Tensor` に絞るには「どのキーを選ぶか」という恣意的な規則が追加で必要になり、レイアウト・キー命名を歪める

**代替案**: 単一テンソルの取り出しが必要な場合は `load_tensor(name, version, key) -> Result<Tensor<f32>, ModelError>` の追加で対応できる（本イシューでは未実装。必要になった時点で別イシューとして追跡する）。

## 6. `available_models` の列挙規則

- ルートが存在しない、または `read_dir` できない場合は空 `Vec`（レジストリが空とみなす。エラー型を返さないのは戻り値型の契約による。`load` の I/O エラー区別とは異なる扱い）
- ルート直下の**ディレクトリ**で名前が §7 の検証を通るものを `name` 候補とする（ファイル・検証不合格名はスキップ）
- `name` 直下のディレクトリで名前が検証を通り、かつ `model.safetensors` が通常ファイルとして存在するものだけを `version` として列挙する
- version が 1 件もない `name` は結果に含めない
- `name`・`versions` とも文字列昇順ソート（決定的出力。`save_safetensors_f32_to_bytes` のキー昇順と同じ方針）
- ディレクトリ名は UTF-8 変換できるもののみ対象（非 UTF-8 名はスキップ）

## 7. `name`／`version` の検証規則（OWASP A03 パストラバーサル対策）

許可文字集合は `[A-Za-z0-9._-]+`（1 文字以上）。拒否: 空文字・`.`・`..`・`/`・`\`・NUL・空白・その他の文字。**先頭が `.` の名前は拒否**（`.hidden` 等の隠しディレクトリ混入防止。`.`／`..` 排除と一貫した規則）。長さ上限は設けない（OS 側の制限に委ねる。上限値の導入はユーザー承認事項）。

検証は `load`・`available_models` の両方が同一の private 関数 `validate_component` を使う（allowlist 方式・fail-closed）。`load` は `name`・`version` をファイルシステムへ触れる前に検証してから存在確認へ進む（`crates/facade/tests/model_registry.rs::invalid_components_are_rejected_before_fs_access` が固定）。

`with_cache_dir` に渡されたルート自体は利用者の所有物として検証しない（相対パス可。正規化はしない）。シンボリックリンクは `is_file()`／`read_dir` の既定挙動どおり追従する（利用者自身のキャッシュディレクトリ配下のみを対象とするため、追従の拒否は行わない）。

## 8. safetensors 検証との関係

ファイル自体の検証（ヘッダ・dtype・shape）は `crate::interop::safetensors::load_safetensors_f32` に一元化されており、`model.rs` は複製・迂回しない（REQ-7 契約は `interop::safetensors` 側が正）。ファイルサイズ上限・改竄検知は未導入（既存の safetensors 公開面と同じ扱い。値の決定にはユーザー承認が必要）。

## 9. 対象外（Issue 追跡。新規起票はユーザー承認が必要なため本 doc への記載に留める）

- リモート取得・HF hub 連携（#2088 が既存）
- `docs/model-distribution-design.md` の作成（親 #2082 の成果物。本イシューでは作成しない）
- version 記法の解釈（semver・hash 等）・最新版解決
- `compat::Sequential` へのレジストリ直結ラッパー（`Sequential::from_registry` 等）
- 非 F32 dtype・ファイルサイズ上限・改竄検知

## 10. 実機 parity 非該当の整理

本イシューはホスト側のファイル I/O（パス組み立て・存在確認・列挙）と、数値経路に一切触れない既存 `interop::safetensors` への委譲のみで構成される。`crates/onnx-interop/src/st_load.rs`（#1752・#2019）と同じ整理により、CUDA／Metal 実機依存のテスト・ベンチは発生しない。`docs/perf/logs/` への申し送りは不要と判断した。

## 11. 出典

- `crates/facade/src/model.rs`（本体実装・モジュール doc）
- `crates/facade/src/interop/safetensors.rs`（委譲先・REQ-7 契約）
- `crates/facade/tests/model_registry.rs`・`crates/facade/tests/api_surface.rs`（受入テスト・公開面の機械固定）
- `docs/facade-safetensors-exposure-decision.md`・`docs/compat-api-scope.md`（既存の公開範囲決定記録）
