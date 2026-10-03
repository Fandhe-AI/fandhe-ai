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

### 実装時の差分（#2390）

- `ensure_cache_root` には非 unix スタブを置かず `cfg(unix)` のみとした（呼び出し元は同ファイルの store／load のみで、両スタブが呼ばないため `-D warnings` で dead_code になる。`#[allow]` での抑止は `coding-rust.md` が禁じる）。
- 非 unix スタブは `runtime_workspace_root`・`store_cache_entry`・`load_cache_entry` の 3 個で、いずれも `CudaError::CacheDirUnavailable` を返す。

## 6. サポート対象 OS の階層（本書を正とする）

- **Linux／macOS**: 全機能（ディスクキャッシュは fd pin の TOCTOU 対策付き）。
- **Windows（x86_64-pc-windows-msvc）**: #2390・#2391 によりビルド可能（CI の Windows クロス clippy で継続検査）。ディスクキャッシュは設計上無効。CUDA 実行時動作は #2393 まで未検証。**2026-10-03 実測・#2393**: Windows Server 2022（GCE VM・Tesla T4）で、`CUDA_INCLUDE_PATH` を明示すれば CUDA GEMM が動作し CPU 参照と統一複合判定で一致した。未指定では全カーネルのコンパイルが失敗する（§9 の実測結果）。**#2487**: `%CUDA_PATH%\include` を include 候補（`CUDA_INCLUDE_PATH` の直後）へ追加した。Toolkit 標準レイアウトでの実機確認は未実施で人手確認待ち。
- **その他の非 unix**: ビルド可能だが未検証。

`docs/backend-switching-design.md` にはサポート対象 OS の明文がなく、`nvrtc.rs` 冒頭コメントが同 doc を出典として引いているのは引用のずれである。同 doc へは本書への 1 行参照のみ追加した。

## 7. cudarc の Windows 動的ロード

cudarc 0.19.8 `src/lib.rs:204-243` の `get_lib_name_candidates` は、Windows では `DLL_PREFIX=""`・`DLL_SUFFIX=".dll"` で `{lib}{64}_{major}{minor}_0` 等を候補にする。CUDA 13.0（`cuda-13000`）では NVRTC 側の候補に `nvrtc64_130_0.dll`、driver 側は `nvcuda.dll` などが含まれる。ビルド可否は §2 の実測（cudarc が msvc 向けにコンパイル可）に基づく。DLL の実配置（`bin\` か `bin\x64\`）・`PATH` 要件は **未検証 → #2393**（推定を事実として書かない）。

**2026-10-03 実測・#2393（上の「未検証」への結果。出典 `docs/perf/logs/windows-onnx-external-data-2393/README.md`）**: Windows Server 2022 の GCE VM（Tesla T4・ドライバ 596.86）で、`nvcuda.dll` は `System32`、NVRTC は CUDA 13.0 redist 13.0.88 の `bin\x64\`（`nvrtc64_130_0.dll`・`nvrtc-builtins64_130.dll`）に配置されていた。NVRTC を含むディレクトリが `PATH` に無い場合は NVRTC がロードされず型付きエラーになり、`PATH` へ追加するとロードされた（`gpu/gpu-env.txt`・`gpu/g-cuda-probe-*.log`）。

## 8. CI の追加方法（required contexts を変えない）

- 新規ジョブは作らず、既存 `build` ジョブ（check context 名 `cargo build (linux / aarch64-apple-darwin)`）も改名しない。ruleset `main-protection`（20587668）の required contexts を更新せずに済む（`.claude/rules/ci.md`「運用上の注意」）。step の `name:` は check context ではない。
- `build` ジョブ内に既存 onnx-interop 行（`--lib --tests`）とは別のコマンド行を 1 行追加する。#2390 後: `cargo clippy -p fandhe-ai-backend-cuda --lib --locked --target x86_64-pc-windows-msvc -- -D warnings`。#2391 後は facade 行（`-p fandhe-ai`）にまとめてよい。同一行に足さないのは、unix 専用テストを持つクレートに `--tests` を強制しないため（`--tests` の要否は #2391 で判断）。
- Makefile `check-cross-windows-interop` にも同一コマンド行を追加する（ci.yml と共用の既存方針）。

### 実装時の確定（#2391・2026-09-29）

- facade 行 1 本（`cargo clippy -p fandhe-ai --lib --tests --locked --target x86_64-pc-windows-msvc -- -D warnings`）にまとめた。`-p fandhe-ai` は workspace の path 依存（backend-cuda 等の lib）にも clippy lint を適用することを、`cfg(windows)` 限定の lint 違反を仕込むプローブで実測した。backend-cuda 専用行は不要。
- `--tests` を採用した（`docs/compat-model-io-decision.md` §12.4 項目 5 の前提）。3 つの test target で `cfg(unix)` 専用の補助が dead_code になっていたため cfg を揃え、非 unix 向け `load_model_is_unsupported_on_non_unix` を追加した。
- ジョブ ID・name は不変で、ruleset の required contexts も変更していない。

## 9. #2393 への申し送り（Windows 実機で検証する項目）

1. CUDA 13.0 Windows toolkit／driver で `nvcuda.dll`・`nvrtc64_130_0.dll` が解決されること、配置ディレクトリと `PATH` 要件。
2. `nvrtc-builtins64_130.dll` が同時にロードできること。
3. DLL 不在時に `Device::Cuda` 選択が panic せず `DriverUnavailable`／`NvrtcUnavailable` の型付きエラーになること。
4. ディスクキャッシュが実際に無効であること（キャッシュディレクトリ非作成・`RUST_AI_CUDA_CACHE_DIR` 設定でも書き込まれない）とプロセス内 LRU の再利用。
5. facade `OnnxModel::from_path` から external data を読めること（#2391 後）。
6. GPU 搭載 Windows 実機で GEMM の CPU 参照との数値一致（統一複合判定）。

### 実測結果（2026-10-03・#2393）

出典: `docs/perf/logs/windows-onnx-external-data-2393/README.md`（CPU VM＝Windows Server 2022 Datacenter 21H2 build 20348.5622・GPU なし。GPU VM＝同 OS・Tesla T4〔compute capability 7.5・sm_75〕・ドライバ 596.86。いずれも GCE VM で物理機ではない。Rust 1.99.0）。T4 は本リポの主対象 GB10（sm_121）ではなく、TF32／mma 系（sm_80 以上）は対象外。

1. **確認（条件付き）**: `nvcuda.dll` は `System32` から解決された。`nvrtc64_130_0.dll` は CUDA 13.0 redist 13.0.88 の `bin\x64\` に置かれ、PATH へ追加するとロードされた。PATH に NVRTC が無い場合は `Device::Cuda` の選択自体は成功し、GEMM 実行時に `CudaUnavailable("CUDA NVRTC library unavailable ...")` の型付きエラーになった（`gpu/g-cuda-probe-no-nvrtc-path.log`）。
2. **確認**: `nvrtc-builtins64_130.dll` は `nvrtc64_130_0.dll` と同じ `bin\x64\` にあり、PATH 追加後に NVRTC のコンパイルが実行できた（`CUDA_INCLUDE_PATH` 指定後の `g2-*` で全 GEMM カーネルのコンパイルが成功。DLL のロード可否は個別観測ではなくコンパイル成功からの間接確認。`gpu/g2-cuda-probe-with-include.log`）。
3. **確認（エラー名の観測値）**: DLL 不在時は panic せず型付きエラー。CPU VM（ドライバなし）は `Device::Cuda` 選択時に `CudaUnavailable("CUDA driver library unavailable ...")`、GPU VM（NVRTC が PATH に無い）は GEMM 実行時に `CudaUnavailable("CUDA NVRTC library unavailable ...")`。想定していた `DriverUnavailable`／`NvrtcUnavailable` という名前ではなく、実際は `CudaUnavailable(<メッセージ>)` の 1 バリアントだった（`cpu/p4-cuda-probe-nogpu.log`）。
4. **確認**: `RUST_AI_CUDA_CACHE_DIR` を設定してもディレクトリ・ファイルは作成されなかった（CPU VM は `cache_dir_exists_after=False`、GPU VM は `created_dir=false files_written=false`）。プロセス内 LRU は 64x64x64 GEMM の 1 回目 約 2.6 s（NVRTC コンパイル）→ 2 回目以降 約 1.2〜1.5 ms で再利用された（`gpu/g2-cuda-probe-with-include.log`）。
5. **確認**: facade `OnnxModel::from_path` で external data を読めた（`interop_onnx_external_data` 2 passed。CPU VM の NTFS・ReFS・exFAT と GPU VM）。詳細は `docs/onnx-external-data-decision.md` §8 の 2026-10-03 追記。
6. **確認（`CUDA_INCLUDE_PATH` 指定が前提）**: PATH に NVRTC を追加しただけでは**全カーネルのコンパイルが `cuda_fp16.h` を開けず失敗**した（`NVRTC_ERROR_COMPILATION`）。原因は `crates/backend-cuda/src/nvrtc.rs` の `compile_ptx` の include 候補が `CUDA_INCLUDE_PATH` と Linux 固定パスのみで、Windows の CUDA ヘッダ位置を探さないこと（#2487 で `%CUDA_PATH%\include` を候補に追加。Toolkit 標準レイアウトでの実機確認は人手確認待ち）。CUDA 13.0 redist の cudart 13.0.96・crt 13.0.88・cccl 13.0.85 のヘッダ（版数と sha256 は `gpu/redist-manifest.txt`。取得スクリプトの指定値の転記）を 1 ディレクトリにまとめて `CUDA_INCLUDE_PATH` で指定すると、`cpu_cuda_parity` 2・`gemm_naive` 3・`gemm_tiled` 6・`gemm_transposed_parity` 5・`gemm_batched_parity` 2 が passed、`device_init` 1 passed（初回から）、64x64x64 GEMM が CPU 参照と統一複合判定で `fail_count=0`（`max_abs_diff` 0）だった。`gemm_f32_variants` は `internal-diagnostics` feature 必須のため未実行。

結論として、§9 の 6 項目はいずれも Windows 実機で確認できたが、6 は `CUDA_INCLUDE_PATH` の明示指定が必要で、既定構成のままでは Windows の CUDA が動かない（#2487 の対処は実機未確認。§6 の Windows 階層の「CUDA 実行時動作は未検証」は、この制約付きの動作確認済みに更新する）。

## 10. 他 doc・コードに残るずれ（別イシューで是正）

- `nvrtc.rs` 冒頭コメントの引用ずれ → #2390。
- `ci.yml`（build ジョブの Windows ステップ・clippy コメント等）と `Makefile:270-275` の「facade は Windows でビルドできない」記述 → #2391 で是正済み。
- `docs/onnx-external-data-decision.md` §7 → #2391 で是正済み。

## 11. OWASP Top 10 観点

- A08: ディスク PTX は同 uid の書き込み主体に対し認証できない。Windows ではディスク I/O 自体を行わず、キャッシュ汚染経由の任意 PTX 実行経路が構造的に存在しない。unix 側の現行判断は不変。
- A01／A04: ACL／SID の独自トラストモデルや reparse point の新規 TOCTOU 面を作らない。unix の fd pin は無改変。
- A03: `CUDA_INCLUDE_PATH`・`RUST_AI_CUDA_CACHE_DIR` の扱いは不変。`CUDA_PATH`（Windows のみ）も include パス文字列としてだけ使う（#2487）。Windows ではキャッシュルート用環境変数が書き込み先にならない。
- A06: 依存の追加・更新なし（`Cargo.toml`／`Cargo.lock` 不変）。
- `unsafe` の追加なし。

## 12. スコープ外

`nvrtc.rs` 実装変更（#2390）／facade の Windows クロス clippy・CI・Makefile の実変更・§10 の整合（#2391）／既存 kernel32 FFI の監査（#2392）／Windows 実機検証（#2393）／認証済みディスク PTX 検証の導入（将来。その際 (i) を再検討）。
