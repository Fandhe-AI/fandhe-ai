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

`with_cache_dir` に渡されたルート自体は利用者の所有物として検証しない（相対パス可。正規化はしない。ルート自体がシンボリックリンクであることは許容する。§12 参照）。

**2026-09-22 追記（PR #2226 codex-review 指摘・P0）**: レジストリ内部（`<root>/<name>`・`<name>/<version>`・葉ファイル `model.safetensors`）にシンボリックリンクが事前配置された場合の扱いは、当初「`is_file()`／`read_dir` の既定挙動どおり追従する」としていたが、これはキャッシュルート脱出（OWASP A03）を許してしまう欠陥だったため撤回した。現在の契約・実装は §12 を参照。

## 8. safetensors 検証との関係

ファイル自体の検証（ヘッダ・dtype・shape）は `crate::interop::safetensors::load_safetensors_f32` に一元化されており、`model.rs` は複製・迂回しない（REQ-7 契約は `interop::safetensors` 側が正）。ファイルサイズ上限・改竄検知は未導入（既存の safetensors 公開面と同じ扱い。値の決定にはユーザー承認が必要）。

## 9. 対象外（Issue 追跡。新規起票はユーザー承認が必要なため本 doc への記載に留める）

- リモート取得（#2088）・HF hub 連携（facade には入れず別クレート。#2243〈#2244〜#2246〉。2026-09-24 ユーザー決定。**2026-09-24 追記**: 当初は「#2088 が既存」としていたが、HF hub 連携は #2088 のスコープからも分離されたため訂正した。詳細は `docs/model-distribution-design.md` §5）
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

## 12. シンボリックリンク経由のキャッシュルート脱出対策（2026-09-22 追記・PR #2226 codex-review 指摘・P0）

§7 で策定した文字集合検証（`validate_component`）だけでは、レジストリ内部にシンボリックリンクを事前配置された場合のキャッシュルート脱出（OWASP A03）を防げない。`<root>/<name>`・`<name>/<version>`・葉ファイル `model.safetensors` のいずれかがシンボリックリンクであれば、`std::fs::metadata`（追従する）・`load_safetensors_f32` 経由でキャッシュルート外の任意 safetensors ファイルを読み込めてしまう。

**対策方針**: `load`・`available_models` の両方が内部の private 関数 `resolve_model_file` を経由する（列挙結果は必ず `load` が受理するパスのみを含む一貫性保証）。

1. `<root>/<name>`・`<name>/<version>`・葉ファイルの各段を [`std::fs::symlink_metadata`]（リンクを辿らない no-follow 検査）で検査し、いずれかがシンボリックリンク、または期待する型（ディレクトリ／通常ファイル）でなければ `NotFound` に丸める（`available_models` 側は `DirEntry::file_type()`（no-follow）で `name` 段を同様に検査する）。
2. 葉パスを `Path::canonicalize` し、キャッシュルートの canonicalize 結果配下であることを多層防御として再確認する（多段リンク・パス正規化差異対策）。

**本節時点（2026-09-22 初版）で採用しなかった対策・その後の撤回**: 当初は「`O_NOFOLLOW` での no-follow open（lstat→open 間の TOCTOU を完全に閉じる）には `libc` クレートの直接依存が要る」と判断し、std のみで構成して lstat 検査後・実際の open までの間の TOCTOU（差し替えレース）を対象外としていた。**この判断は同一 PR 内で撤回済み**（詳細は §13）: `libc` を追加せずとも `std::os::unix::fs::OpenOptionsExt::custom_flags` へ生の flag 値を渡すことで `O_NOFOLLOW`／`O_NONBLOCK` を実現でき、TOCTOU を実体識別子（`dev`／`ino`）照合で閉じられることが判明したため。

**ルート自体がシンボリックリンクの場合は許容する**（`load_succeeds_when_root_itself_is_a_symlink` で固定）。拒否対象はレジストリ内部からの脱出のみであり、`with_cache_dir` に渡すルート自体の間接参照は妨げない。

**Windows の既定ルート解決順序も同時に是正した**（P2・同 PR 指摘）: `ModelRegistry::new` は Windows では `USERPROFILE` を `HOME` より優先する（`cfg(windows)`）。従来は両 OS で `HOME` を先に見ており、公開ドキュメントが規定する Windows 既定ルート（`$USERPROFILE/.fandhe-ai/models`）と実装が乖離しうる欠陥だった。

固定テストは `crates/facade/tests/model_registry.rs` の `load_rejects_symlinked_leaf_file_escaping_root`・`load_rejects_symlinked_version_dir_escaping_root`・`load_rejects_symlinked_name_dir_escaping_root`・`available_models_excludes_all_symlink_escape_variants`（いずれも `#[cfg(unix)]`）・`load_succeeds_when_root_itself_is_a_symlink`（過剰拒否でないことの確認）。

## 13. TOCTOU（lstat〜open 間の差し替えレース）対策への切替（2026-09-22 追記・PR #2226 codex-review 指摘・P0／P2）

§12 は「lstat 検査後・実際の open までの TOCTOU は `libc` 直接依存が要るため対象外」としていたが、この判断は誤りだったため撤回し、以下へ差し替えた。`libc` を追加せずとも `std::os::unix::fs::OpenOptionsExt::custom_flags` に生の `open(2)` flag 値（`O_NOFOLLOW`・`O_NONBLOCK`。カーネル UAPI ヘッダ由来の固定値。`crates/facade/src/model.rs` の `open_flags` モジュール参照）を直接渡せば、`libc` クレートなしで no-follow open が実現できる。

**Linux／macOS の対策（実装済み）**:

1. 葉ファイルを `symlink_metadata`（no-follow）で検査し、`is_file() == true` を明示要求する（`!is_dir()` ではなく。FIFO・Unix ソケット・デバイスファイル等の非通常ファイルも拒否）。
2. 葉パスを canonicalize しキャッシュルート配下であることを多層防御として再確認する（§12 のスナップショット検査。単独では TOCTOU を閉じない）。
3. 葉を `open_leaf_no_follow`（private 関数）で開く。`O_NOFOLLOW`（シンボリックリンクへの差し替えをカーネルレベルで拒否。ELOOP で検出）・`O_NONBLOCK`（FIFO への差し替えによる `open` の無期限ブロックを防ぐ）を付与する。
4. 開いたハンドルの `fstat`（`File::metadata`）で `is_file()` を再確認したうえで、手順 1 の `symlink_metadata`（lstat）と `(dev, ino)` が一致することを検証する（「検査と open のハンドル一体化」）。手順 1〜3 の間に `name`／`version`／葉のいずれかが差し替えられても、開かれた実体の識別子は検査時点のものと一致しないため確実に検出できる（パスの再解決ではなく実体の同一性判定のため、中間ディレクトリの差し替えも同じ仕組みで捕捉する）。
5. 以降は同じ `File` ハンドルから読み取ったバイト列を `load_safetensors_f32_from_bytes` へ渡す（パスで再度 open すると手順 3〜4 で閉じた TOCTOU 窓が復活するため、ハンドルの使い回しは必須）。

**Linux／macOS 以外（Windows 等）の扱い（P0・2026-09-22 是正）**: 当初案は「対応する生 flag 値を持たないためプレーンな `File::open` にフォールバックし、`(dev, ino)` 照合は `#[cfg(unix)]` 限定で省略する」としていたが、これは公開ドキュメント（本 doc §3・crate doc）が明示的に Windows 配置（`$USERPROFILE/.fandhe-ai/models`）をサポートすると謳っているにもかかわらず、Windows では symlink_metadata 完了後〜`File::open` 前の TOCTOU 窓が無防備に残る欠陥だった（codex-review 指摘・PR #2226・P0）。許容依存 9 区分に `libc`／`windows-sys` が無く、Windows 向けの安全な no-follow open（`file_index`／`volume_serial_number` によるハンドル識別子照合を含む）を本 PR のスコープで確立できなかったため、**フォールバックではなく fail-closed 拒否**へ是正した: `open_leaf_no_follow` は Linux／macOS 以外では `ErrorKind::Unsupported` を返し、`resolve_model_file` はこれを `ModelError::Io` として伝播する。

この伝播の呼び出し元ごとの帰結は次のとおり: `load` は常に `Err(ModelError::Io(..))` を返す（Windows では成功しない）。`available_models` は各バージョンの存在確認を `resolve_model_file(&name, &version).is_ok()` の真偽値でのみ行い、エラー内容を伝播しないため、Windows では該当バージョンが静かに一覧から除外され、結果として常に空の一覧を返す（エラーにはならない）。キャッシュディレクトリ解決自体（`USERPROFILE` 優先。§12）は Windows でも従来どおり動作する。安全な Windows 実装の追加はスコープ外として別イシューで追跡する（本 doc 更新時点では未起票。実装対象外の追跡規約 `out-of-scope-tracking.md` に従いユーザー承認を経て追跡する）。

**対象外として残る経路（不変）**: レジストリ内に事前配置されたハードリンク（攻撃者が任意タイミングで作成できるのは同一ファイルシステム上の既存ファイルへのリンクのみであり、所有者・権限チェックを伴わない本レジストリの脅威モデル外）。

固定テストは §12 記載分に加え、`crates/facade/tests/model_registry.rs` の `load_rejects_non_regular_leaf_unix_socket`・`load_rejects_leaf_replaced_with_symlink_after_initial_write`（Unix ソケットの葉拒否・差し替え後の葉拒否。いずれも `#[cfg(unix)]`）、および `crates/facade/src/model.rs` の単体テスト（`non_empty_os_string_*`・`select_home_var_none_when_both_empty_strings`・`select_home_var_falls_back_when_preferred_is_empty_string`。空文字列 HOME／USERPROFILE の扱い。Cursor Bugbot 指摘・PR #2226）を追加した。

## 14. 公開説明と Windows 非対応実態の整合（2026-09-22 追記・PR #2226 codex-review 指摘・P1）

§13 で Windows の `load`／`available_models` を fail-closed 拒否へ是正したが、公開説明（`site/guides/interop.md` の「ローカルモデルレジストリ」節）と crate doc（`crates/facade/src/model.rs` のモジュール doc・`ModelRegistry::new` doc）が依然「Windows は `$USERPROFILE`」とだけ記載し、`load`／`available_models` が常に失敗する実態に触れていなかった（codex-review 指摘・PR #2226・P1）。

**是正内容**: `ModelRegistry::new`（キャッシュルート解決）は Windows でも動作するが、`load`・`available_models` は現時点で fail-closed 非対応であることを明記する形へ、次の 3 箇所を更新した。

1. `crates/facade/src/model.rs` モジュール doc: レイアウト規定ブロックの Windows 注記に「`load`／`available_models` は fail-closed 非対応」を追記し、独立した「Windows 対応状況」節を新設して §13 の帰結（`load` は常に `Err`、`available_models` は常に空の一覧）を crate doc 側にも明記した。
2. `ModelRegistry::new` のドキュメンテーションコメント: 「本関数によるキャッシュルート解決自体は Windows でも動作するが、`load`・`available_models` は fail-closed 非対応」という注記を追加した。
3. `site/guides/interop.md`「ローカルモデルレジストリ」節: 同内容の注記を追加した。

安全な Windows 実装（`file_index`／`volume_serial_number` によるハンドル識別子照合を含む）の追加は引き続きスコープ外（§13 に同じ）。

## 15. `model.safetensors` のファイルサイズ上限導入（2026-09-22 追記・PR #2226 codex-review 指摘・P0）

`load` は `resolve_model_file` が開いた `File` ハンドルから `read_to_end` で全バイトを無条件に `Vec` へ確保していた。`model.safetensors` は非信頼な外部フォーマット入力（利用者が手動配置するが、共有キャッシュディレクトリ経由で他プロセス・他ユーザーが書き込める場合もある）であり、サイズ検証なしの無制限確保は巨大ファイルによるメモリ枯渇（OWASP A03。AGENTS.md「外部フォーマットのパース検証（P0）」長さ事前検証要件）を招く（codex-review 指摘・PR #2226・P0）。

**対策**: `crates/facade/src/model.rs` に固定サイズ上限 `MAX_MODEL_FILE_BYTES`（当初 8 GiB。後述の再指摘を受け 1 GiB へ改定）を導入し、二段構えで検証する。

1. `resolve_model_file` の手順 4（fstat によるハンドル識別子照合）の直後に `open_meta.len()` を `MAX_MODEL_FILE_BYTES` と比較し、上回れば読み取りへ進む前に `ModelError::TooLarge` で拒否する。
2. `load` 側は fstat 完了後にファイルが差し替え・追記されて増大する TOCTOU にも備え、`std::io::Read::take(MAX_MODEL_FILE_BYTES + 1)` で読み取り自体を上限バイト数超で打ち切り、実際に読めたバイト数が上限を超えていれば同じく `ModelError::TooLarge` で拒否する（ちょうど上限バイト数で打ち切ると超過を検出できないため `+ 1` バイト分だけ多く読む）。事前確保サイズは実測ファイルサイズ（上限未満なら実測値）を用い、`Vec::try_reserve` で割り当て失敗を panic ではなく型付きエラーへ変換する。

両検証は private 純関数 `enforce_size_limit(name, version, len, max) -> Result<(), ModelError>` に集約し、`crates/facade/src/model.rs` の単体テストで境界値（`len == max` は許可・`len == max + 1` は拒否）・`Display` 出力・定数の非退化を検証する。加えて、上限値そのものを引数化した private ラッパー `resolve_model_file_with_limit`／`load_with_limit`（`resolve_model_file`／`load` は本番の固定 `MAX_MODEL_FILE_BYTES` でこれらへ委譲する薄いラッパー）を経由する単体テスト（`resolve_model_file_with_limit_rejects_over_bound`・`resolve_model_file_with_limit_accepts_at_bound`・`load_with_limit_rejects_over_bound`）を追加し、実際の fstat 経路・`Read::take` 二段構え経路の両方が `enforce_size_limit` と同じ判定へ到達することを、`MAX_MODEL_FILE_BYTES`（1 GiB）相当の実ファイルを用意せずに数バイトの実ファイル＋小さい `max` 引数で再現する（2026-09-23 追記・codex-review 再指摘への対応。1 GiB の実ファイル生成による検証は引き続き非現実的なため採用しない）。

**8 GiB を選定した理由（初回・撤回済み）**: 本レジストリが対象とする F32 のみの `compat::Sequential` 向けローカル重み（数百 MB〜数 GB 級を想定）に対して十分な余裕を持たせつつ、攻撃者が用意した巨大ファイルによる無制限確保を防ぐための固定安全域として選定した。

**1 GiB への改定（2026-09-22 追記・PR #2226 codex-review 再指摘・P0）**: 8 GiB は「攻撃者が用意した通常サイズのファイルだけで一般的な実行環境（GitHub ホステッド runner の既定 7 GiB RAM・開発者のノート PC 等）の OOM を引き起こせる」水準であり、AGENTS.md の長さ事前検証要件が求める「実質的な OOM 防止」を満たさないとの再指摘を受けた。加えて `load_safetensors_f32_from_bytes`（`crates/onnx-interop/src/st_load.rs`）は読み込んだ `Vec<u8>`（ファイルサイズ相当）と、デコード後の `Tensor<f32>` 群（safetensors の F32 データ部とほぼ同サイズ）を `bytes` の drop まで同時に保持するため、ピークメモリはおおよそ**ファイルサイズの 2 倍**になる。想定する最低限のホスト RAM を 4 GiB、単一モデルロードに許容する割合をその半分（2 GiB）とし、ピーク倍率 2 で割った `2 GiB ÷ 2 = 1 GiB` をファイルサイズ上限とした。本レジストリが対象とする F32 のみのローカル重み（数百 MB 級を主に想定。1 GiB 超のモデルは対象外）に対しては引き続き十分な余裕を持つ。1 GiB という値は本リポジトリ内で新規に導入したものではなく、`onnx-interop` の外部ファイル読み込み先例（`crates/onnx-interop/examples/model_zoo_probe.rs:45::MAX_READ_BYTES = 1024 * 1024 * 1024`・`crates/onnx-interop/tests/model_zoo_parity.rs:201` 同値。いずれも非信頼な外部フォーマット入力の読み込み上限として同じ 1 GiB を採用済み）と揃えたものであり、外部フォーマット読み込みの上限値の一貫性という観点でも妥当である。mmap／ストリーミング解析への変更は、許容依存 9 区分に mmap 相当のクレート（`libc`／`memmap2` 等）が無く、かつ `load_safetensors_f32_from_bytes` 側もデコード後にテンソル全量を保持する構造のためピークメモリの根本削減にはならないことから、本 PR のスコープでは採用しなかった（実装を要する場合は別途ユーザー承認・別イシューで追跡）。1 GiB 超のより大きなモデルへの対応は、単一 `Vec` への全量読み込みという現行方式のままでは上限引き上げが OOM 防止を後退させるため妥当ではなく、将来ストリーミング／mmap ベースの読み込み設計（別途ユーザー承認・別イシュー）を経て初めて扱う。

`crates/facade/src/interop/safetensors.rs`・`onnx.rs` はより汎用的な入口（他の呼び出し元からも使われる）のため上限値の決定を「値の決定にはユーザー承認が要る」としてスコープ外のまま維持しているが、本モジュールの `load` は単一の読み取り専用ローカルレジストリ入口に閉じており、固定の安全域値を導入・改定すること自体は AGENTS.md の長さ事前検証要件を満たすための実装判断であり、依存追加・ガードレール閾値・テスト許容誤差の変更（`.claude/rules/*.md` のユーザー承認必須事項）のいずれにも該当しない。将来より大きなモデルを扱う必要が生じた場合の値の見直しは別途ユーザー承認を経る。

`ModelError` は `#[non_exhaustive]` のため `TooLarge { name, version, len, max }` variant の追加は非破壊。`crates/facade/tests/api_surface.rs::model_types_are_reachable_via_facade` の `match` にも明示 arm を追加した。
