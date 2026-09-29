# facade の Windows ビルド方針（NVRTC ディスクキャッシュの扱い）決定記録

イシュー #2389。関連: #2390（backend-cuda 実装）・#2391（facade・CI・doc 整合）・#2392（既存 kernel32 FFI 監査）・#2393（Windows 実機検証の受け皿）。

## 1. 背景・目的

- facade（`fandhe-ai`）は `backend-cuda` に無条件依存する。`crates/backend-cuda/src/nvrtc.rs:45-52` は非 unix で `compile_error!` を出す（#509／PR #677）ため、facade は Windows でビルドできない。
- そのため PR #2351 で Windows 対応済みの onnx-interop external data（`docs/onnx-external-data-decision.md`）に、facade `OnnxModel::from_path` からは Windows で到達できない。
- 本書は「Windows でビルド可能にする際、NVRTC ディスクキャッシュをどう扱うか」を 1 案に決める。コード変更は含まない（実装は #2390）。

## 2. 実測（イシュー #2389 記載・main 5764c54b 時点）

- `x86_64-pc-windows-msvc` 向け check は `backend-cuda` で停止する（`nvrtc.rs:46` の `compile_error!`、`std::os::unix` 使用箇所、`mode()`／`uid()`）。
- tensor-core・autodiff・backend-cpu・onnx-interop は同ターゲットで check が通る。cudarc 自体も msvc 向けにコンパイルでき、`Cargo.lock` は Windows 向け依存（libloading 系）を解決済み。
- facade 自身のコードの Windows コンパイル可否は **未検証**（`nvrtc.rs` を変更しないと検査できないため。#2391 で検証する）。

## 3. 判断材料

| 事実 | 出典 |
|---|---|
| ディスクキャッシュにヒットしても `kernel.ptx` は実行入力にしない。ヒットでもミスでも必ず `compile_ptx` を実行するため、**Linux／macOS でもディスクキャッシュによるプロセス再起動をまたいだコンパイル時間短縮は現状ゼロ** | `docs/cuda-jit-cache-design.md`「ディスク PTX を実行入力にしない判断（#511 PR #703）」、`crates/backend-cuda/src/module_cache.rs::load_function_cached` |
| ディスクキャッシュ失敗（workspace_root 解決不能・fs I/O 失敗）は fail-safe でキャッシュなしの縮退運転（NVRTC 直コンパイル＋プロセス内 LRU） | 同 doc「縮退方針」、`module_cache.rs:523`（`runtime_workspace_root().ok()`） |
| `load_function_cached` はディスク I/O の前に `runtime_workspace_root()` を呼ぶ。これが Err なら load も store も行われない（唯一の分岐点） | `module_cache.rs:523`、`nvrtc.rs:1282` |
| `ensure_cache_root`（:3153）・`store_cache_entry`（:3417）・`load_cache_entry`（:3681）の `pub(crate)` ラッパーは cfg 分けされていない。`has_workspace_root_marker`（:1167）は `MetadataExt` を cfg なしで使う | `nvrtc.rs` |
| アーキテクチャのゲート `compile_error!`（:1520）は `target_os = "linux"` 限定で Windows に影響しない | `nvrtc.rs:1516-1526` |
| `CudaError` は `#[non_exhaustive]` で `CacheDirUnavailable { detail }` が既にある | `crates/backend-cuda/src/error.rs` |
| driver／NVRTC 不在は `is_culib_present()` プローブで `DriverUnavailable`／`NvrtcUnavailable` の型付きエラーになる（panic 回避） | `crates/backend-cuda/src/device.rs`、`error.rs` |
| facade `model.rs` の `open_leaf_no_follow` は Linux／macOS 以外で既に `ErrorKind::Unsupported`（fail-closed） | `crates/facade/src/model.rs` |

## 4. 3 案の比較と決定

| 観点 | (i) kernel32 FFI で handle ベース移植 | (ii) Windows ではディスクキャッシュ無効・プロセス内メモリのみ（**採用**） | (iii) Windows では CUDA 経路を fail-closed |
|---|---|---|---|
| セキュリティ | ACL／所有者 SID・reparse point・原子的置換の TOCTOU 対策を Windows で作り直す。新たな攻撃面 | ファイル I/O 自体がなくなり TOCTOU・symlink 脱出・キャッシュ汚染の対象が消える（最強） | CUDA を使えないので攻撃面なし |
| 性能 | 現状の利得ゼロ（ディスク PTX を実行入力にしないため） | 現行比で劣化ゼロ（unix でも起動ごと再コンパイル）。プロセス内 LRU は維持 | GPU 性能をすべて失う |
| 実装量 | 大（FFI・ACL 照会・rename 置換・テスト一式。2h 単位の再分割要） | 小（cfg 分割と非 unix スタブ数個） | (ii) と同じ cfg 分割に加え実行時拒否・facade 分岐が必要（(ii) の上位集合） |
| `unsafe` | 追加が必要（security-auditor 監査・ユーザー承認必須） | 追加なし | 追加なし |
| 依存追加 | なし | なし | なし |
| 設計整合 | 整合 | 整合（cudarc 無条件依存＋動的ロード。toolkit は実行時のみ必要） | 無条件依存＋動的ロード構成と矛盾。Windows で CUDA が壊れている根拠もない |

**決定: (ii) を採る。** 非 unix（Windows）ではディスクキャッシュを無効にし、プロセス内 LRU と NVRTC 直コンパイルのみで動かす。

### 棄却理由

- **(i)**: ディスク PTX を実行入力にしない現行判断の下では移植しても性能利得がなく、`unsafe` 追加と新規攻撃面の不利益が大きい。加えて `unsafe` 追加はユーザー承認必須（`security.md`・`coding-rust.md`）で、自動運転中は承認を得られない。**再検討条件**: ディスク PTX を実行入力に戻す認証済み検証手段（`cuda-jit-cache-design.md` が将来案とする署名検証等）を導入する時に、Windows のディスクキャッシュを (i) で実装するか改めて判断する。
- **(iii)**: どの案でも `compile_error!` 撤去と unix I/O の cfg 分割が要るため、(iii) は (ii) に実行時拒否を足すだけ。拒否の根拠（Windows で CUDA が動かない実測）もない。driver／NVRTC 不在は既存プローブで型付きエラーにできる。

### (i) を選ばない場合の受入項目

(i) を選ばないため、`unsafe` の範囲・監査要件・承認の記録、および #2390 の 2h 再分割は **不要**（本書で明記）。

## 5. #2390 向けの実装形（指針）

1. `nvrtc.rs:45-52` の `compile_error!` を撤去し、冒頭「サポート対象 OS」節を本書出典に書き直す（`backend-switching-design.md` を引く記述の是正）。
2. `runtime_workspace_root()` は `cfg(not(unix))` で早期に `Err(CudaError::CacheDirUnavailable { detail })`（本書参照）を返す。`load_function_cached` は無変更で縮退運転になる。
3. `MetadataExt`／uid 依存関数（`has_workspace_root_marker`・`current_euid` 等）は `cfg(unix)` に閉じる。
4. `ensure_cache_root`／`store_cache_entry`／`load_cache_entry` の `pub(crate)` ラッパーは `cfg(unix)` とし、同シグネチャで `CacheDirUnavailable` を返す `cfg(not(unix))` スタブを置く（`module_cache.rs` の呼び出し側は cfg なしのまま）。
5. unix の I/O ヘルパー群（`nvrtc.rs` 約 :1484〜:3150）は `cfg(unix)` でまとめて囲む（Windows 向け clippy `-D warnings` の dead_code 回避。`#[allow]` で黙らせない: `coding-rust.md`）。
6. アーキテクチャゲート（:1516）は変更しない。
7. 受入範囲は `--lib`。unix 専用テスト（`jit_cache_regression_tests.rs` 等）は `cfg(unix)` のまま。unix の既存経路（TOCTOU 回帰テスト含む）は無改変。`unsafe` は追加しない。

## 6. サポート対象 OS の階層（本書を正とする）

- **Linux／macOS**: 全機能（ディスクキャッシュは fd pin の TOCTOU 対策付き）。
- **Windows（x86_64-pc-windows-msvc）**: #2390・#2391 後にビルド可能。ディスクキャッシュは設計上無効。CUDA 実行時動作は #2393 まで未検証。
- **その他の非 unix**: ビルド可能だが未検証。

`docs/backend-switching-design.md` にはサポート対象 OS の明文がなく、`nvrtc.rs` 冒頭コメントが同 doc を出典として引いているのは引用のずれである。同 doc へは本書への 1 行参照のみ追加した。

## 7. cudarc の Windows 動的ロード

cudarc 0.19.8 `src/lib.rs:204-243` の `get_lib_name_candidates` は、Windows では `DLL_PREFIX=""`・`DLL_SUFFIX=".dll"` で `{lib}{64}_{major}{minor}_0` 等を候補にする。CUDA 13.0（`cuda-13000`）では NVRTC 側の候補に `nvrtc64_130_0.dll`、driver 側は `nvcuda.dll` などが含まれる。ビルド可否は §2 の実測（cudarc が msvc 向けにコンパイル可）に基づく。DLL の実配置（`bin\` か `bin\x64\`）・`PATH` 要件は **未検証 → #2393**（推定を事実として書かない）。

## 8. CI の追加方法（required contexts を変えない）

- 新規ジョブは作らず、既存 `build` ジョブ（check context 名 `cargo build (linux / aarch64-apple-darwin)`）も改名しない。ruleset `main-protection`（20587668）の required contexts を更新せずに済む（`.claude/rules/ci.md`「運用上の注意」）。step の `name:` は check context ではない。
- `build` ジョブ内に既存 onnx-interop 行（`--lib --tests`）とは別のコマンド行を 1 行追加する。#2390 後: `cargo clippy -p fandhe-ai-backend-cuda --lib --locked --target x86_64-pc-windows-msvc -- -D warnings`。#2391 後は facade 行（`-p fandhe-ai`）にまとめてよい。同一行に足さないのは、unix 専用テストを持つクレートに `--tests` を強制しないため（`--tests` の要否は #2391 で判断）。
- Makefile `check-cross-windows-interop` にも同一コマンド行を追加する（ci.yml と共用の既存方針）。

## 9. #2393 への申し送り（Windows 実機で検証する項目）

1. CUDA 13.0 Windows toolkit／driver で `nvcuda.dll`・`nvrtc64_130_0.dll` が解決されること、配置ディレクトリと `PATH` 要件。
2. `nvrtc-builtins64_130.dll` が同時にロードできること。
3. DLL 不在時に `Device::Cuda` 選択が panic せず `DriverUnavailable`／`NvrtcUnavailable` の型付きエラーになること。
4. ディスクキャッシュが実際に無効であること（キャッシュディレクトリ非作成・`RUST_AI_CUDA_CACHE_DIR` 設定でも書き込まれない）とプロセス内 LRU の再利用。
5. facade `OnnxModel::from_path` から external data を読めること（#2391 後）。
6. GPU 搭載 Windows 実機で GEMM の CPU 参照との数値一致（統一複合判定）。

## 10. 他 doc・コードに残るずれ（別イシューで是正）

- `nvrtc.rs` 冒頭コメントの引用ずれ → #2390。
- `ci.yml`（build ジョブの Windows ステップ・clippy コメント等）と `Makefile:270-275` の「facade は Windows でビルドできない」記述 → #2391。
- `docs/onnx-external-data-decision.md` §7 → #2391。

## 11. OWASP Top 10 観点

- A08: ディスク PTX は同 uid の書き込み主体に対し認証できない。Windows ではディスク I/O 自体を行わず、キャッシュ汚染経由の任意 PTX 実行経路が構造的に存在しない。unix 側の現行判断は不変。
- A01／A04: ACL／SID の独自トラストモデルや reparse point の新規 TOCTOU 面を作らない。unix の fd pin は無改変。
- A03: `CUDA_INCLUDE_PATH`・`RUST_AI_CUDA_CACHE_DIR` の扱いは不変。Windows ではキャッシュルート用環境変数が書き込み先にならない。
- A06: 依存の追加・更新なし（`Cargo.toml`／`Cargo.lock` 不変）。
- `unsafe` の追加なし。

## 12. スコープ外

`nvrtc.rs` 実装変更（#2390）／facade の Windows クロス clippy・CI・Makefile の実変更・§10 の整合（#2391）／既存 kernel32 FFI の監査（#2392）／Windows 実機検証（#2393）／認証済みディスク PTX 検証の導入（将来。その際 (i) を再検討）。
