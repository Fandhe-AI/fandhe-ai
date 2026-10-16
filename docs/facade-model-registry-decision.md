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

配置は利用者（または将来のダウンロード側。#2088）が行う。`ModelRegistry` はディレクトリの作成・削除を一切行わない読み取り専用のレジストリである（書き込み側 `save` は §16 で追加決定。実装は #3119）。

## 4. 公開面（承認事項。マージ前に確認されたい）

facade 新規公開面 7 件（`crates/facade/src/model.rs`。`crates/facade/tests/api_surface.rs::model_module_exposes_only_approved_surface` が機械固定）:

1. `fandhe_ai::model::ModelRegistry`（構造体。フィールドは private）
2. `fandhe_ai::model::ModelError`（`#[non_exhaustive]` enum。`Debug`／`Display`／`std::error::Error` 実装）
3. `ModelRegistry::new() -> Result<Self, ModelError>`（既定ルート `$HOME/.fandhe-ai/models`。`HOME`／`USERPROFILE` 未設定時は `ModelError::CacheDirUnavailable`）
4. `ModelRegistry::with_cache_dir(dir: impl Into<PathBuf>) -> Self`（ルート明示指定。統合テスト・CI・非標準配置向け）
5. `ModelRegistry::cache_dir(&self) -> &Path`
6. `ModelRegistry::load(&self, name: &str, version: &str) -> Result<HashMap<String, Tensor<f32>>, ModelError>`
7. `ModelRegistry::available_models(&self) -> Vec<(String, Vec<String>)>`

（`save` の追加による 8 件目は §16 で決定。実装は #3119）

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
3. 葉を `open_leaf_no_follow`（`crate::fs_guard` の crate 内共有関数。#2364 で `model.rs` から抽出）で開く。`O_NOFOLLOW`（シンボリックリンクへの差し替えをカーネルレベルで拒否。ELOOP で検出）・`O_NONBLOCK`（FIFO への差し替えによる `open` の無期限ブロックを防ぐ）を付与する。
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

**対策**: `crates/facade/src/fs_guard.rs`（#2364 で `model.rs` から抽出）に固定サイズ上限 `MAX_MODEL_FILE_BYTES`（当初 8 GiB。後述の再指摘を受け 1 GiB へ改定）を導入し、二段構えで検証する。

1. `resolve_model_file` の手順 4（fstat によるハンドル識別子照合）の直後に `open_meta.len()` を `MAX_MODEL_FILE_BYTES` と比較し、上回れば読み取りへ進む前に `ModelError::TooLarge` で拒否する。
2. `load` 側は fstat 完了後にファイルが差し替え・追記されて増大する TOCTOU にも備え、`std::io::Read::take(MAX_MODEL_FILE_BYTES + 1)` で読み取り自体を上限バイト数超で打ち切り、実際に読めたバイト数が上限を超えていれば同じく `ModelError::TooLarge` で拒否する（ちょうど上限バイト数で打ち切ると超過を検出できないため `+ 1` バイト分だけ多く読む）。事前確保サイズは実測ファイルサイズ（上限未満なら実測値）を用い、`Vec::try_reserve` で割り当て失敗を panic ではなく型付きエラーへ変換する。確保失敗は `ModelError::Io` の `ErrorKind::OutOfMemory`（ヒープ確保なしで構築できる `io::Error::from(ErrorKind)` 表現）で返す（#2250・PR #2239 と同類型）。

両検証は private 純関数 `enforce_size_limit(name, version, len, max) -> Result<(), ModelError>` に集約し、`crates/facade/src/model.rs` の単体テストで境界値（`len == max` は許可・`len == max + 1` は拒否）・`Display` 出力・定数の非退化を検証する。加えて、上限値そのものを引数化した private ラッパー `resolve_model_file_with_limit`／`load_with_limit`（`resolve_model_file`／`load` は本番の固定 `MAX_MODEL_FILE_BYTES` でこれらへ委譲する薄いラッパー）を経由する単体テスト（`resolve_model_file_with_limit_rejects_over_bound`・`resolve_model_file_with_limit_accepts_at_bound`・`load_with_limit_rejects_over_bound`）を追加し、実際の fstat 経路・`Read::take` 二段構え経路の両方が `enforce_size_limit` と同じ判定へ到達することを、`MAX_MODEL_FILE_BYTES`（1 GiB）相当の実ファイルを用意せずに数バイトの実ファイル＋小さい `max` 引数で再現する（2026-09-23 追記・codex-review 再指摘への対応。1 GiB の実ファイル生成による検証は引き続き非現実的なため採用しない）。

**8 GiB を選定した理由（初回・撤回済み）**: 本レジストリが対象とする F32 のみの `compat::Sequential` 向けローカル重み（数百 MB〜数 GB 級を想定）に対して十分な余裕を持たせつつ、攻撃者が用意した巨大ファイルによる無制限確保を防ぐための固定安全域として選定した。

**1 GiB への改定（2026-09-22 追記・PR #2226 codex-review 再指摘・P0）**: 8 GiB は「攻撃者が用意した通常サイズのファイルだけで一般的な実行環境（GitHub ホステッド runner の既定 7 GiB RAM・開発者のノート PC 等）の OOM を引き起こせる」水準であり、AGENTS.md の長さ事前検証要件が求める「実質的な OOM 防止」を満たさないとの再指摘を受けた。加えて `load_safetensors_f32_from_bytes`（`crates/onnx-interop/src/st_load.rs`）は読み込んだ `Vec<u8>`（ファイルサイズ相当）と、デコード後の `Tensor<f32>` 群（safetensors の F32 データ部とほぼ同サイズ）を `bytes` の drop まで同時に保持するため、ピークメモリはおおよそ**ファイルサイズの 2 倍**になる。想定する最低限のホスト RAM を 4 GiB、単一モデルロードに許容する割合をその半分（2 GiB）とし、ピーク倍率 2 で割った `2 GiB ÷ 2 = 1 GiB` をファイルサイズ上限とした。本レジストリが対象とする F32 のみのローカル重み（数百 MB 級を主に想定。1 GiB 超のモデルは対象外）に対しては引き続き十分な余裕を持つ。1 GiB という値は本リポジトリ内で新規に導入したものではなく、`onnx-interop` の外部ファイル読み込み先例（`crates/onnx-interop/examples/model_zoo_probe.rs:45::MAX_READ_BYTES = 1024 * 1024 * 1024`・`crates/onnx-interop/tests/model_zoo_parity.rs:201` 同値。いずれも非信頼な外部フォーマット入力の読み込み上限として同じ 1 GiB を採用済み）と揃えたものであり、外部フォーマット読み込みの上限値の一貫性という観点でも妥当である。mmap／ストリーミング解析への変更は、許容依存 9 区分に mmap 相当のクレート（`libc`／`memmap2` 等）が無く、かつ `load_safetensors_f32_from_bytes` 側もデコード後にテンソル全量を保持する構造のためピークメモリの根本削減にはならないことから、本 PR のスコープでは採用しなかった（実装を要する場合は別途ユーザー承認・別イシューで追跡）。1 GiB 超のより大きなモデルへの対応は、単一 `Vec` への全量読み込みという現行方式のままでは上限引き上げが OOM 防止を後退させるため妥当ではなく、将来ストリーミング／mmap ベースの読み込み設計（別途ユーザー承認・別イシュー）を経て初めて扱う。

`crates/facade/src/interop/safetensors.rs`・`onnx.rs` はより汎用的な入口（他の呼び出し元からも使われる）のため上限値の決定を「値の決定にはユーザー承認が要る」としてスコープ外のまま維持しているが、本モジュールの `load` は単一の読み取り専用ローカルレジストリ入口に閉じており、固定の安全域値を導入・改定すること自体は AGENTS.md の長さ事前検証要件を満たすための実装判断であり、依存追加・ガードレール閾値・テスト許容誤差の変更（`.claude/rules/*.md` のユーザー承認必須事項）のいずれにも該当しない。将来より大きなモデルを扱う必要が生じた場合の値の見直しは別途ユーザー承認を経る。

`ModelError` は `#[non_exhaustive]` のため `TooLarge { name, version, len, max }` variant の追加は非破壊。`crates/facade/tests/api_surface.rs::model_types_are_reachable_via_facade` の `match` にも明示 arm を追加した。

## 16. 保存系メソッド `save` の公開形（イシュー #3118・親 #3117。D-A）

書き込み側（ローカル配置 API）の公開形を確定する。実装は #3119、テストと `api_surface` の正ガードは #3120。本節は決定記録であり、公開状態は**未公開**（コード変更なし）。§3 の「読み取り専用」・§4 の「公開面 7 件」は `save` の実装（#3119）が入るまで現状のまま有効である。

### 16.1 候補比較と推奨

| 案 | 入力 | 判断 |
|---|---|---|
| **A（推奨）** | `&HashMap<String, Tensor<f32>>` | `load` の戻り値と対称で、`save` → `load` の bit 一致往復がそのまま成り立つ。`compat::Sequential::state_dict` の出力をそのまま渡せる。F32 限定は型で保証され、エンコードは副作用のない `save_safetensors_f32_to_bytes(tensors, None)`（`crate::interop::safetensors`。決定的出力）へ委譲する |
| B | 検証済みバイト列（`&[u8]`） | 公開すると非信頼バイト列の経路が増え、F32 限定・REQ-7 の検証を再デコードしない限り迂回できる。ダウンロード（#3121）はバイト列ではなく一時ファイルへのストリーミング設計（`docs/model-download-design.md` §5）のため利点も薄い。不採用 |
| C | パス指定（既存 safetensors の取り込み） | 読み取り元パスという非信頼入力が増え symlink 対策が二重に要る。既存 `save_safetensors_f32(path, ..)` の一時ファイル（`std::fs::write`。最終要素の symlink に追従）を流用すると `docs/compat-model-io-decision.md` §12.3 で撤回済みの欠陥を継承する。`load_safetensors_f32` → `save` で代替できる。不採用 |

### 16.2 公開名・シグネチャ

```rust
pub fn save(
    &self,
    name: &str,
    version: &str,
    tensors: &HashMap<String, Tensor<f32>>,
) -> Result<(), ModelError>
```

- 名前は `load` と対になる `save`。戻り値は `()`（`compat::save_model` と同型）。保存先パスは `cache_dir()` とレイアウト（§3）から決まるため返さない（パス全文を出力に出さない既存方針とも揃う）
- 上書きフラグは持たない。既存の `model.safetensors` は原子的に置き換える（§16.4 手順 8）
- 追加する公開面は `save` の 1 件のみ（7 → 8 件）。`ModelError` の新 variant は §16.3

### 16.3 エラー型

`ModelError` は `#[non_exhaustive]` のまま variant の追加のみ行い、既存 variant は変えない（非破壊）。

| 事象 | variant | 備考 |
|---|---|---|
| `name`／`version` が allowlist 外 | 既存 `InvalidComponent` | ファイルシステムに触れる前に返す |
| テンソル集合が空 | **新規 `EmptyModel { name: String, version: String }`** | `to_bytes` は空マップでも成功しうるため明示的に拒否する |
| サイズ上限超過 | 既存 `TooLarge { name, version, len, max }` | `max` = `MAX_MODEL_FILE_BYTES`（1 GiB。§15。値は不変） |
| safetensors へのエンコード失敗 | **新規 `Save(SaveError)`** | `Load(LoadError)` と対称。`source()` は `Some(e)`。`SaveError` は `fandhe_ai::interop::safetensors::SaveError` として公開済みで、新しい型の露出はない |
| 書き込み前の読み戻し検証の失敗 | 既存 `Load(LoadError)` | デコード失敗は「`load` が読めないファイルを作らない」ことの帰結。キー集合・shape の不一致（内部不整合）は `Io(ErrorKind::InvalidData)` |
| `<name>`／`<version>` が symlink または非ディレクトリ | 既存 `Io(ErrorKind::InvalidInput)` | `compat::model_io` の no-follow 拒否と同じ表現。メッセージにパス全文を含めない |
| 非対応プラットフォーム | 既存 `Io(ErrorKind::Unsupported)` | 副作用の前に返す（§16.5） |
| 一時ファイル名を 8 回試行しても確保できない・その他の I/O | 既存 `Io` | |

新 variant の `Display`／`Debug` には `name`／`version` のみを出し、パスは出さない（OWASP A09。既存 variant と同方針）。

### 16.4 書き込みの安全契約（std のみ。`compat::model_io` と同水準）

レイアウトがファイル名を `model.safetensors` に固定するため、`compat::model_io` が safetensors に使う「一意な世代名へ `create_new` で直接書く」方式は使えない。manifest に使う「一時ファイル → `rename`」方式を採る。手順の順序は固定する。

1. **プラットフォーム判定**: §16.5 の範囲外なら何にも触れず `Io(Unsupported)` を返す
2. **書き込み前の検証**（ファイルシステムに一切触れない）: `validate_component("name")` → `validate_component("version")`（既存 private 関数を流用。allowlist は不変）→ 空集合の拒否 → 要素数 × 4 の合計を checked 演算で求めた概算を `MAX_MODEL_FILE_BYTES` と比較 → `save_safetensors_f32_to_bytes` でエンコード → `bytes.len() > MAX_MODEL_FILE_BYTES` なら `TooLarge` → `load_safetensors_f32_from_bytes(&bytes)` で読み戻してキー集合・shape の一致を確認し、デコード結果は直ちに drop する（`compat::model_io::verify_round_trip` 相当。「保存できたが `load` が拒否する」ファイルを作らない）。追加メモリのピークはバイト列とデコード結果の約 2 倍で、`load` のピーク（§15 の「ファイルサイズの 2 倍」）と同水準
3. **ルートの用意**: `create_dir_all(cache_dir)`。ルートは利用者の所有物として扱い、ルート自体が symlink でも許容する（§7・§12 と一貫）。「`ModelRegistry` はディレクトリを作らない」性質（§3）は `save` に限って変わる。`load`／`available_models` は従来どおり作らない
4. **`<name>`・`<version>` を 1 段ずつ用意**: 各段でまず `symlink_metadata`（追従しない）を取り、存在しなければ `std::fs::create_dir` で 1 段だけ作る（`create_dir_all` は使わない。`docs/model-download-design.md` §6 (a) が一括再帰作成を禁じる趣旨に揃える）。`AlreadyExists` は許容し改めて `symlink_metadata` を取る。最終的に symlink または非ディレクトリなら `Io(InvalidInput)` で拒否する。各段の `(dev, ino)` は手順 7 の再確認用に記録する
5. **多層防御**: `<version>` を canonicalize し、ルートを canonicalize した結果の配下にあることを確認する（§12 手順 2 と同じスナップショット検査）
6. **一時ファイルの作成と書き込み**: `<version>/` 内に `.model.safetensors.tmp-{pid}-{counter}-{nanos}` を `OpenOptions::write(true).create_new(true)` で作る。`AlreadyExists` は既存エントリに触れず名前を作り直し、最大 `MAX_TMP_NAME_ATTEMPTS` = 8 回試す。`create_new` は最終要素の symlink に追従しない。名前は先頭 `.` で `model.safetensors` と一致しないため `load`／`available_models` から見えない。続けて `write_all(&bytes)`
7. **確定直前の再確認**（窓を狭めるだけで閉じはしない）: 一時ファイルのハンドルの `fstat` と一時パスの `symlink_metadata` の `(dev, ino)` が一致し、`<name>`／`<version>` の `(dev, ino)` が手順 4 の記録から変わっていないこと。不一致なら自分の一時ファイルを削除して `Io(InvalidData)`
8. **公開**: `std::fs::rename(tmp, <version>/model.safetensors)` を唯一のコミット点とし、既存ファイルを原子的に置き換える。POSIX の `rename(2)` は宛先が既存 symlink でもリンクエントリ自体を置き換え参照先へは書かない（`docs/compat-model-io-decision.md` §12.3 手順 2 の整理）。宛先がディレクトリなら失敗するのでエラーとして扱う。上書き拒否方式（`RENAME_NOREPLACE` 相当）を採らないのは、ダウンロードの再取得が同名への置き換えを要するため（`docs/model-download-design.md` §5・§6 (b)）。std には NOREPLACE の手段もない
9. **保存後の検証**: 内部の `resolve_model_file` で開き直し（読み取りはしない）、`fstat` の `(dev, ino)` が手順 6 の一時ファイルと一致することを確認する。不一致なら `Io(InvalidData)`。`Ok` 時点で「`load` が読むファイルは今書いたもの」を保証する
10. **失敗時の後始末**: 手順 6 以降の失敗では、自分が作成し fd を保持する一時ファイルに限り、`(dev, ino)` 一致かつ通常ファイルの場合だけ削除する（`remove_own_tmp` と同型。`docs/compat-model-io-decision.md` §13.6）。作成したディレクトリは削除しない（空の `<version>/` は `available_models` が列挙しないため無害）。既存の `model.safetensors` は手順 8 より前の失敗では無傷
11. **耐久性と並行性**: fsync はしない（`save_model` と同じ。電源断時の保証なし）。同一 `name`/`version` への並行 `save` は最後に `rename` した側が勝ち、どちらのファイルも壊れない。`save` 中の `load` は原子的置換のため旧版か新版のどちらかを読める。ただし並行利用はサポート対象外として rustdoc に書く

**共有ヘルパーの配置**: `create_new_with_retry`・`remove_own_tmp`・`name_entropy` を、新設の crate 内モジュール `crates/facade/src/fs_write.rs`（`pub(crate)` のみ）へ挙動を変えずに移設し、`compat::model_io` と `model` の双方が使う。下位層の `model` が `compat` に依存する逆向き依存を避けるため、および `fs_guard.rs` 冒頭 doc が「書き込み側はこのモジュールに置かない」と定めているため。移設に合わせ `fs_guard.rs` 冒頭 doc の「書き込み側は `compat::model_io` が持つ」の一文を更新する（#3119 の作業）。

### 16.5 Windows・プラットフォームの扱い

- 書き込み側の対応範囲は読み取り側（`fs_guard::open_leaf_no_follow` の対応範囲）と同じ **Linux x86_64／aarch64 と macOS** に揃える。それ以外（Windows・他の unix・他の Linux アーキ）は、手順 3 のディレクトリ作成より前に `Io(Unsupported)` で fail-closed にする
- 理由: 手順 4・7・9・10 は `(dev, ino)`（`MetadataExt`）と `resolve_model_file` に依存する。`save` 成功後に `load` が拒否する環境を作らず往復契約を保つ。`compat::model_io` の「save は unix 全体、load はより狭い」非対称は `ModelRegistry` では採らない
- Windows の安全な実装（`file_index`／`volume_serial_number` 照合）は §13 と同じくスコープ外

### 16.6 dirfd 経路との範囲分離

- **本 issue 系（std のみ）で担保する範囲**: 書き込み前の allowlist 検証、各段の no-follow 型検査（symlink・非ディレクトリの拒否。#3120 の「`<name>` が symlink なら拒否」テストの根拠）、canonicalize によるルート配下確認、最終要素で symlink に追従しない `create_new`、エントリ自体を置換する `rename`、`(dev, ino)` で所有を確かめてからの一時ファイル削除、確定直前と保存後の `(dev, ino)` 再確認
- **残るリスク（実装前の想定。§16.9 で解消済み）**: 手順 4 の検査から手順 6 の作成・手順 8 の `rename` までの間に、キャッシュルートや `<name>` へ書き込める攻撃者（共有キャッシュディレクトリ・別のローカルユーザー）が `<name>`／`<version>` を symlink へ差し替える TOCTOU は、std では構造的に閉じられない。手順 7・9 は検出して `Err` を返せるが、ルート外へ書いた後になりうる。中間ディレクトリの作成も文字列パス解決に依存する。既定の `$HOME` 配下を単一ユーザーで使う限り該当する攻撃者は限られる。ルート自体の symlink は許容のまま
- **#3123（dirfd 経路の依存承認）の後に強化する範囲**: `docs/model-download-design.md` §6 (a)・(b) の dirfd 連鎖へ置き換える（`openat2`〈`RESOLVE_NO_SYMLINKS | RESOLVE_BENEATH`〉または `openat(O_NOFOLLOW | O_DIRECTORY | O_CLOEXEC)`、`ENOENT` 時の `mkdirat`、`openat(O_CREAT | O_EXCL | O_NOFOLLOW)`、`renameat`）。配置は `docs/model-download-design.md` §14.5 案 B-1（第 10 区分 `libc` の用途を facade へ拡張）を推奨とし、承認されるまで実施しない。強化後も公開シグネチャは変えない（内部置換のみ）
- ダウンロード（#3121）は §6 (c) により dirfd 経路が使えない環境では fail-closed にする必要がある。std のみの `save` は #3121 の安全な土台の代わりにならない

### 16.7 後続で更新する箇所（本 issue では実施しない）

- #3119: `crates/facade/src/model.rs`（`save`・新 variant・rustdoc〈Windows の扱い・`load` との往復契約・残る symlink リスク・並行性・耐久性〉）、`fs_write.rs`（新設）、`compat/model_io.rs`（helper 移設のみ）、`fs_guard.rs`（冒頭 doc の一文のみ）
- #3120: `crates/facade/tests/api_surface.rs`（`scan_unapproved_model_pub_items` の `ALLOWED_PUB_ITEMS` を 7 → 8 件にし `("fn", "save")` を追加／`model_types_are_reachable_via_facade` の `match` に `Save(_)`・`EmptyModel { .. }` の明示 arm／公開面ちょうど 8 件の正ガード／`ModelRegistry::save` の関数ポインタ型到達プローブ／利用例 doctest 実在検査）、本 doc への実装記録、`docs/model-distribution-design.md` §2.1・§6 への追記形での `save` 追加、`docs/compat-api-scope.md` §5 の適用記録、`docs/README.md` 索引、`site/guides/interop.md` の利用者向け説明（裁量）

### 16.8 承認・出典

- 承認: `docs/compat-api-scope.md` §5 の D-A（`https://github.com/Fandhe-AI/fandhe-ai/issues/3018#issuecomment-6111643437`。条件は「記録に形を書き、その形のまま実装へ進む。形を書けない点だけ所有者へ戻す」）。本節はすべての論点に形を与えており、**所有者へ戻す論点はない**。依存追加・`docs/spec/` 編集・tolerance／上限値の変更はない
- 出典: `crates/facade/src/model.rs`（`validate_component`・`resolve_model_file`）、`crates/facade/src/fs_guard.rs`（`MAX_MODEL_FILE_BYTES`・`open_leaf_no_follow`）、`crates/facade/src/compat/model_io.rs`（`create_new_with_retry`・`remove_own_tmp`・`MAX_TMP_NAME_ATTEMPTS`）、`docs/compat-model-io-decision.md` §12.3・§13.6、`docs/model-download-design.md` §5・§6・§14.5（案 B-1）

### 16.9 実装時の是正: ディレクトリ fd 固定（2026-10-16・PR #3377 codex-review P0。§16.4 手順 4〜7・§16.5・§16.6 を置き換える）

§16.6 の「残るリスク」（検査〜`create_dir`／`rename` 間のシンボリックリンク差し替えでルート外へ書ける）は、レビュー基準にシンボリックリンク脱出の許容例外がないため、実装（#3119）で次のとおり閉じた。依存の追加はない（std のみ）。

- **方式**: キャッシュルートをディレクトリとして開き、`<name>`・`<version>` を 1 段ずつ `/proc/self/fd/<親 fd>/<段>` 経由で `create_dir`（`AlreadyExists` 許容）→ `O_NOFOLLOW|O_DIRECTORY` で開いて fd を保持する。以後の一時ファイル作成（`create_new`）と `rename` は `/proc/self/fd/<version の fd>/` 配下だけで行い、パス文字列を再解決しない。検査ではなく「開いた fd」が基準なので、後からパスがリンクへ差し替えられても書き込み先は開いた実体に留まる（差し替えは確定後の `resolve_model_file` 検証が `Err` にする。ルート外への書き込みは起きない）。手順 5（canonicalize）と手順 7 のディレクトリ `(dev, ino)` 再確認は不要となり削除した（一時ファイルの `(dev, ino)` 再確認は残す）
- **プラットフォーム**: `/proc/self/fd` 方式は Linux 限定のため、`save` の対応範囲は **Linux x86_64／aarch64 のみ**へ狭めた（§16.5 の「macOS を含む」を置き換え）。macOS・Windows・他アーキは書き込み前に `Io(Unsupported)`。`/proc` が使えない Linux も書き込み前に `Err`。`load`／`available_models` の対応範囲は不変。macOS の `save` は `libc` による `openat`／`renameat`（§16.6 の #3123・案 B-1。第 10 区分の用途拡張はユーザー承認事項）の承認後に復帰させる
- **O_DIRECTORY の生値**: `fs_guard::open_flags::O_DIRECTORY`（x86_64 = 0o200000・aarch64 = 0o40000）を追加した（`O_NOFOLLOW` と同方針。libc は使わない）
- **テスト**: `<version>` 段・`<name>` 段を確定直前フックでシンボリックリンクへ差し替えても外部ディレクトリに何も書かれず `Err` になることを単体テストで検証する
- **残るリスク**: ルート自体のシンボリックリンクは許容のまま（利用者の選択）。`save` 後の読み戻し検証は差し替え検出用で、書き込みを取り消すものではない（書き込み先は fd 固定のため外部には出ない）

## 17. 実装記録（イシュー #3120）

§16.7 の #3120 項目の消化状況。公開面（公開アイテム）は 1 件も増やしていない。§3「ディレクトリを作らない」・§4「公開面 7 件」は `save` について §16・§17 が優先する（§3・§4 の本文は書き換えない。読み取り系の契約としては有効）。

### 17.1 追加したテスト

| ファイル | 内容（受入項目） |
|---|---|
| `crates/facade/tests/model_registry_save.rs`（新規・公開 API と std のみ） | R1 往復（複数テンソル・非連続 view・複数 name／version・上書き・`Sequential` の state dict）／R2 パス検証（`..`・絶対パス・区切り・NUL・制御文字・先頭 `.`・非 ASCII・空白、極端に長い名前、`<name>`／`<version>` のシンボリックリンク〈dangling 含む〉、葉のシンボリックリンクを辿らない置き換え、ルートのシンボリックリンク許容）／R3 障害（rename 失敗・上限超過〈stride 0 view で確保なし〉・空モデル・F16 ファイルの load 拒否と置き換え、いずれも既存が無傷で `.model.safetensors.tmp-*` が残らない）／エラー表示にパスが出ない |
| `crates/facade/tests/api_surface.rs` | `MODEL_APPROVED_PUB_ITEMS`（8 件）・`MODEL_REGISTRY_APPROVED_PUB_METHODS`（6 件）の定数化、正ガード `model_registry_surface_violations`（公開面ちょうど・欠落／重複も違反）と自己テスト、`src/` 全体の `impl ModelRegistry` の pub fn 集合の検査、関数ポインタ型の到達プローブ `model_registry_signatures_are_pinned`、`save` の doctest 実在検査 |
| `crates/facade/src/model.rs` | `save` の doc に実行される doctest（一時ディレクトリ使用・`Unsupported` では何もしない）を追加。挙動は不変 |

失敗注入・途中差し替え・一時名衝突は単体テスト（`model.rs` の `mod tests`）が担当し、結合テストは重複させない。

### 17.2 検出範囲と限界

- 非 F32 入力は `save` の引数型 `HashMap<String, Tensor<f32>>` で型として拒否される。実行時に再現できる検出は「F16 ファイルの `load` 拒否と `save` による置き換え」に限る
- `with_cache_dir` は `impl Into<PathBuf>` のため関数ポインタ化できず、呼び出し型のプローブに留まる
- 正ガードは `without_test_mod` を使わない。`model.rs` の `///` コメント内の `#[cfg(test)]` 文字列で切り出し位置がずれ、`save`／`available_models` が範囲から落ちるため。`tokens_of` はコメントを除き、`mod tests` 内に `impl ModelRegistry` は無い
- 結合テストの cfg は `save` の対応述語（Linux x86_64／aarch64）と同一。対応外には `Unsupported` を確認する対のテストを置く

### 17.3 長い名前の観測挙動

`validate_component` は長さを見ない。許可文字だけの 300 文字の `name`／`version` は OS の `ENAMETOOLONG` 等で `Io` になり、ルート直下に残骸は残らない（テストは `Err` と無残骸のみ固定し、`ErrorKind` は環境依存のため固定しない）。長さ上限の新設は公開挙動の変更であり本イシューでは行わない（別途扱い）。

### 17.4 承認

D-A の範囲内（§16.8）。新たな承認は得ていない。依存追加・tolerance／上限値・`validate_component` の許可文字集合の変更はない。

## 18. `download` の対応範囲と `save` の dirfd 移行（#3125）

`ModelRegistry::download`／`download_with` は libc の dirfd 相対操作（`cfg(unix)`＝Linux／macOS）で実装した（`docs/model-download-design.md` §19）。`save` は `/proc/self/fd` 方式のまま Linux x86_64／aarch64 限定で、dirfd への移行（macOS 対応を戻す）は §16.6／§16.9 が内部置換を許しているが本イシューの範囲外とした。別イシューでの扱いはユーザー判断（未起票）。

## 19. #3127: download テスト・公開面ガードの仕上げ

公開面の追加・変更はなく、承認事項もない。`download`／`download_with` の公開形（§18）は `api_surface.rs` の正ガード（`DownloadOptions`／`DownloadProgress` のフィールド・`#[non_exhaustive]`・HTTP／TLS 型の非漏洩・TLS 無効化手段の不在・private サブモジュール）で固定した。テスト経路（ループバックモック・実通信 opt-in）の設計と観測は `docs/model-download-design.md` §21 を正とする。

