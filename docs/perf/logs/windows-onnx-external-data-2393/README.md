# Windows 実機での onnx external data 封じ込め・facade Windows 経路の実測ログ（イシュー #2393）

イシュー #2393「Windows 実機で onnx external data の封じ込めと facade の Windows 経路を検証する」の
実測記録。関連: #2349（`win_contained_open` 実装）・#2368（Windows `save_model` の rename 置換先）・
#2389〜#2392（facade の Windows ビルド可能化・CI・kernel32 FFI 監査）・PR #2351（flip-and-revert 是正）。
判断・結論の反映先は次の 3 つの決定記録で、本書は生ログの索引と実測値の出典を担う。

- [`docs/onnx-external-data-decision.md`](../../../onnx-external-data-decision.md) §5・§8・§10（「2026-10-03 実測・#2393」）
- [`docs/compat-model-io-decision.md`](../../../compat-model-io-decision.md) §12.4（W-save-2〜4）
- [`docs/facade-windows-build-decision.md`](../../../facade-windows-build-decision.md) §7・§9

## 位置づけ

- **実測日は 2026-10-03**。**物理機ではなく GCE の VM** で実施した。ファイルシステム意味論（NTFS・ReFS・exFAT）の
  確認としては物理機と同等とみなせるが、ハードウェア依存の差は対象外である。
- 検証後に一時 GCP プロジェクトごと削除した（再現にはプロジェクトの再作成が要る）。実ホスト名・ユーザー名・
  GCP プロジェクト ID は書かない。ログ中の伏せ字は `<user>`（ユーザー名）・`<win-vm>`（VM ホスト名）・
  `<lowpriv-user>`（検証用の非管理者ユーザー）。
- リポジトリのコード・テストは**一切変更していない**（hard link テストの cfg 変更だけ VM 上の作業コピーで一時的に行い、
  後で戻した。下記「注意」）。検証用の使い捨てハーネス（`harness-src/`）はビルド対象外として `.txt` 化して収録した。
- 起票候補は下記のとおり記録したが、**起票はしていない（ユーザー承認待ち。`.claude/rules/out-of-scope-tracking.md`）**。

## 実行環境

| 項目 | CPU VM（`cpu/`） | GPU VM（`gpu/`） |
|---|---|---|
| OS | Windows Server 2022 Datacenter 21H2 build 20348.5622 | 同左 |
| CPU | Intel Xeon 2.80GHz・8 論理 CPU（VM 作成時の指定は n2-standard-8。ログには現れない） | 同左 |
| GPU | なし（NVIDIA ドライバなし） | Tesla T4（compute capability 7.5）・NVIDIA ドライバ 596.86（VM 作成時の指定で Spot。ログには現れない） |
| ボリューム | C: NTFS（システム）／追加ディスクを R: ReFS（10 GiB）・X: exFAT（約 10 GiB）に初期化（`cpu/volumes.log`） | C: NTFS |
| ツールチェーン | Rust 1.99.0（x86_64-pc-windows-msvc）・MSVC 14.44.35207・Windows SDK 10.0.22621.0（`cpu/env_info.txt`） | 同左 |
| ソース | main `e2f54615` の `git archive` を転送（転送元で記録したコミットは `source-commit.txt`）。VM 上でのソース同一性の証跡は fixture 51 件の sha256 一致のみ（`cpu/fixtures-sha256.log`・`gpu/fixtures-sha256.log`: `ok=51 ng=0`） | 同左 |

- GPU VM の CUDA 資材（`gpu/gpu-env.txt`）: `nvcuda.dll` は System32。NVRTC は CUDA 13.0 redist 13.0.88 の
  `bin\x64\`（`nvrtc64_130_0.dll`・`nvrtc-builtins64_130.dll`）。ヘッダは redist の cudart 13.0.96・crt 13.0.88・
  cccl 13.0.85 を 1 ディレクトリへまとめて用意した。版数と照合した sha256 は `gpu/redist-manifest.txt`（取得スクリプトの指定値の転記）、
  照合結果は `gpu/gpu-env.txt` の `sha256_ok=True`。
- T4 は sm_75 で、本リポの主対象 GB10（sm_121）ではない。TF32／mma 系（sm_80 以上）は対象外。
- `gemm_f32_variants` は `internal-diagnostics` feature 必須（`gpu/g-cuda-gemm_f32_variants.log`）のため未実行。

## 実行コマンドと結果

全コマンドの終了コード・所要秒は `cpu/exitcodes.log`・`gpu/exitcodes.log`、各出力は同名ログ（`p3-*`・`p4-*`・`p4b-*`・`g-*`・`g2-*`）。

### ビルド・リンク（CPU VM）

| コマンド | 結果 | ログ |
|---|---|---|
| `cargo build -p fandhe-ai-onnx-interop --tests --locked` | 成功（MSVC リンク成功・55 秒） | `cpu/p3-build-interop.log` |
| `cargo build -p fandhe-ai --tests --locked` | 成功（160 秒） | `cpu/p3-build-facade.log` |

`#[link(name = "kernel32")]` の明示なしでも、`win_contained_open` の kernel32 手書き `extern "system"` 宣言は
リンクで解決された（`LNK` エラーなし）。`docs/onnx-external-data-decision.md` §10.2 の P2 は**実害なし**。
ただし明示化はコード修正の起票候補のまま。

### NTFS（C:。CPU VM）

| コマンド | 結果 | ログ |
|---|---|---|
| `cargo test -p fandhe-ai-onnx-interop --locked --test onnx_external_data` | 50 passed・3 ignored | `cpu/p3-interop-external-data.log` |
| 同 `-- --ignored`（symlink 最終・途中成分、UNC base_dir） | 3 passed（管理者実行・`\\localhost\C$` 到達可: `unc_reachable=True`） | `cpu/p3-interop-external-data-ignored.log` |
| `--test onnx_interp_pytorch_cnn_fixture` | 44 passed | `cpu/p3-interop-cnn-fixture.log` |
| `--lib` | 344 passed | `cpu/p3-interop-lib.log` |
| `cargo test -p fandhe-ai-onnx-interop --locked --no-fail-fast`（全体） | 全 target ok | `cpu/p3-interop-all.log` |
| facade `--test interop_onnx_external_data` | 2 passed（R1' facade `OnnxModel::from_path` で external data 読込を確認） | `cpu/p3-facade-external-data.log` |
| facade `--test interop_onnx_internal_parity` | 16 passed | `cpu/p3-facade-internal-parity.log` |
| facade `--test interop_onnx_external_data_limits` | 0 tests（Windows では対象テストが cfg で除外） | `cpu/p3-facade-external-data-limits.log` |
| facade `--test compat_sequential_model_io` | 2 passed（非 unix で `save_model`／`load_model` が `Unsupported`） | `cpu/p3-facade-model-io.log` |
| hard link（VM 上のみ `overlapping_regions_via_hard_link_are_rejected` の cfg を `any(unix, windows)` へ変更） | 1 passed | `cpu/p3-hardlink-temp-cfg.log` |

### ReFS（R:）・exFAT（X:）（`TMP` を各ボリュームへ向けて再実行）

| ボリューム | `onnx_external_data` | `-- --ignored` | cnn fixture | facade `interop_onnx_external_data` |
|---|---|---|---|---|
| ReFS | 51 passed・3 ignored | 3 passed | 44 passed | 2 passed |
| exFAT | 48 passed・**3 failed**・3 ignored | 1 passed・**2 failed** | 44 passed | 2 passed |

ログ: `cpu/p4-refs-*`・`cpu/p4-exfat-*`。exFAT の失敗はいずれも**テストの前提（reparse point・hard link）を
exFAT が持たないための fixture 作成失敗**で、封じ込め判定の失敗ではない。

- junction 2 件（`junction_as_final_directory_component_is_rejected`・`junction_intermediate_component_is_rejected`）:
  `mklink /J` が失敗（`ExitStatus(1)`）
- `overlapping_regions_via_hard_link_are_rejected`: `hard_link` が `Os { code: 1, kind: Uncategorized, message: "Incorrect function." }`
- `--ignored` の symlink 2 件: `symlink_file`／`symlink_dir` の作成が `Os { code: 1, ... "Incorrect function." }`（UNC base_dir は pass）

### FSCTL_SET_REPARSE_POINT（`reparse_nonempty`）

| ボリューム | 空ディレクトリ | 非空ディレクトリ | ログ |
|---|---|---|---|
| NTFS | 成功 | 145（`ERROR_DIR_NOT_EMPTY`） | `cpu/p4-reparse-nonempty-ntfs.log` |
| ReFS | 成功 | 145 | `cpu/p4-reparse-nonempty-refs.log` |
| exFAT | 1（`ERROR_INVALID_FUNCTION`＝reparse 非対応） | 1 | `cpu/p4-reparse-nonempty-exfat.log` |

`onnx-external-data-decision.md` §5 (d) の「非空ディレクトリは reparse 化できない」は ReFS でも成立した。

### 祖先ディレクトリ rename（`rename_race`）

攻撃スレッドが読み込み中の祖先ディレクトリ `a` を `a.renamed` へ rename し、成功したら即戻す。判定の要点は
`invalid_values`（不正値の件数。0 必須）。モデルは small＝8 MB、large＝256 MB（`gen_model ... --layout a/b`）。

**初回（`p4-rename-race-*`。参考扱い）**: 攻撃が連続 rename する方式。small で load 中の rename の
`code=32`（`ERROR_SHARING_VIOLATION`）が NTFS 67 件・ReFS 161 件観測され `invalid_values=0` だったが、
攻撃が load を邪魔して読み出し段階に届かず、実験として不十分だった。

**修正版（`p4b-rename-delay-*`）**: load ごとに遅延を掃引し、開始から遅延だけ待って rename を 1 回だけ試みる
（`--delay-us auto`。`calibration` は攻撃なしの `from_path` 中央値 T の 0〜1.3 倍を掃引）。

| モデル・FS | `from_path` 中央値（summary） | rename が失敗した区間（`rename_time` の試行時刻） | 失敗コード | 全体の `correct` | `invalid_values` |
|---|---|---|---|---|---|
| 8 MB・NTFS | 7983 µs | 約 0.4〜5.6 ms | 32（約 0.4〜0.8 ms 地点）→ 5（約 1.5 ms 以降） | 120/130 | 0 |
| 8 MB・ReFS | 7960 µs | 約 0.4〜5.1 ms | 32（約 0.4〜0.8 ms 地点）→ 5（約 1.1 ms 以降） | 119/130 | 0 |
| 256 MB・NTFS | 236276 µs | 約 13.8〜129.5 ms | 5 のみ | 48/52 | 0 |
| 256 MB・ReFS | 237471 µs | 約 12.8〜124.4 ms | 5 のみ | 50/52 | 0 |

- 中央値は各ログの `summary from_path_median_us`（`calibration` 行の値は 257657 µs〔NTFS large〕・248001 µs〔ReFS large〕・
  7905 µs／7757 µs〔small〕で、本表の値とは別）。
- **主に `ERROR_ACCESS_DENIED`(5) で失敗**した。`ERROR_SHARING_VIOLATION`(32) は small の最初期（開始後約 0.4〜0.8 ms）にのみ観測した。祖先チェーンを開いている段階と解釈されるが、
  `from_path` 内部の段階の時刻は測っていない（`cpu/p4b-rename-delay-small-ntfs.log` の
  `delay_us=395`: 32×10、`delay_us=790`: 32×2・5×8、`delay_us=1581` 以降は 5 のみ。ReFS は `delay_us=387`: 32×9）。
  256 MB は最初の遅延点が約 12.4〜12.9 ms（ReFS 12400 µs・NTFS 12882 µs）のため、それより早い時点は標本に入らなかった。
- 失敗区間より後（`from_path` はまだ実行中）は rename が**成功**し
  （`load_active_after=1 code=ok`。small NTFS は約 4.8 ms 以降、large NTFS は約 129 ms 以降、large ReFS は約 149 ms 以降）、
  load は**正しい値**で成功した。external data の読み出しが済んで祖先を保持しなくなった後と解釈されるが、読み出し完了の時刻は
  測っていない（実測は rename の試行時刻・エラーコード・`load_active_after` のみ）。いずれにせよ load の値は正しく、封じ込めの破れではない。
- 遅延 0（`from_path` 開始直後に rename 成立）: load は多くが型付きエラー（`Io:NotFound`・`Io:Uncategorized`）
  だった（NTFS small: NotFound 1・Uncategorized 9・正しい値 0、NTFS large: NotFound 1・Uncategorized 3、
  ReFS small: 正しい値 1・NotFound 6・Uncategorized 3、ReFS large: 正しい値 2・NotFound 2）。
- ReFS small の遅延 387／775 µs に、rename 成功と load の `Io:Uncategorized` が各 1 件ずつあった（フラグの立ち下がり競合。
  `load_active_after=0`）。いずれも型付きエラーで不正値ではない。
- **全条件で `invalid_values=0`（不正値ゼロ）**。load は「正しい値」か「型付きエラー」のいずれかだった。

### flip-and-revert（`flip_revert`）

途中成分 A を同じ深さの別ディレクトリ B への junction と入れ替える攻撃（`FSCTL_SET_REPARSE_POINT`）を load と独立に回し、
load が B の値を返す（`B_value_CONTAINMENT_BREACH`）か、その他の不正値（`OTHER_INVALID`）になるかを見る。

| FS | 攻撃サイクル | load 回数 | 内訳（型付きエラー） | `containment_breach_B` | `other_invalid` | ログ |
|---|---|---|---|---|---|---|
| NTFS | 475 | 2000 | reparse 拒否 566・`NotFound` 871・`PermissionDenied` 8・`Uncategorized` 555 | 0 | 0 | `cpu/p4b-flip-revert-ntfs.log` |
| ReFS | 764 | 2000 | reparse 拒否 942・`NotFound` 508・`PermissionDenied` 47・`Uncategorized` 503 | 0 | 0 | `cpu/p4b-flip-revert-refs.log` |

- 正しい値で成功した load は 0 件（攻撃の間隔が短く成功窓がなかった）。攻撃側は終了時に A を実ディレクトリへ戻した
  （`final_state_ok=true`）。
- 初回版（`p4-flip-revert-*`）はハーネス不具合（junction 削除失敗後に復元できず、NTFS は 10 サイクル・ReFS は 1 サイクルで
  攻撃が止まる。`restored=false`）のため**参考扱い**。ただし同版でも `containment_breach_B=0`・`other_invalid=0` だった。

### FILE_TRAVERSE 拒否 ACL（`p4-acl-*`）

- 実行主体は非管理者ユーザー `<win-vm>\<lowpriv-user>`（`BUILTIN\Users`・Medium Mandatory Level。
  **`SeChangeNotifyPrivilege`〈traverse チェックの回避〉は Enabled**。`cpu/p4-acl-whoami.log`）。SSH からの
  `Start-Process -Credential` は `0xC0000142` で起動不能だったため、タスクスケジューラ経由のバッチログオンで実行した
  （`cpu/exitcodes.log` の `baseline exit=-1073741502`）。
- ACL なし: 正しく読込（`pattern=match_seed_A`。`cpu/p4-acl-baseline.log`）。
- 途中ディレクトリ `a` に `(DENY)(S,X)` の ACE（`cpu/p4-acl-icacls-a.txt`）: `from_path` は
  `Io:PermissionDenied` で fail-closed（`cpu/p4-acl-deny-traverse-a.log`）。
- 同条件で通常の `copy` は**成功**した（`cpu/p4-acl-deny-traverse-plain-read.log`。traverse 回避特権による）。
- → 封じ込めは各祖先ディレクトリを明示的に開くため、traverse 拒否 ACE があると通常 API では読めるパスでも拒否する。
  可用性への副作用として記録した（`onnx-external-data-decision.md` §5）。

### W-save（`wsave`。`std::fs::rename`・Rust 1.99.0）

置換先 `manifest.json` を各種リンクにして `rename(tmp, dst)` を実行した（`target_written_through` は参照先が書き換わったか）。

| 置換先 | NTFS | ReFS |
|---|---|---|
| ファイル symlink | rename 成功・リンク自体が通常ファイルに置換（参照先は不変・`target_written_through=false`） | 同左 |
| ディレクトリ symlink | 同上（置換成功） | `ERROR_ACCESS_DENIED`(5)・リンク不変・tmp 残存 |
| junction | 同上（置換成功） | `ERROR_ACCESS_DENIED`(5)・リンク不変・tmp 残存 |
| dangling ファイル symlink | 同上（置換成功） | 同左 |

`create_new`（`OpenOptions::create_new`）: ファイル symlink・dangling は `AlreadyExists`(80)、**ディレクトリ symlink・junction は
`PermissionDenied`(5)（`AlreadyExists` ではない）**。NTFS・ReFS とも同じ。ログ: `cpu/p4-wsave-ntfs.log`・`cpu/p4-wsave-refs.log`。
ハーネスは `std::fs::rename` の最終結果しか観測できないため、`MoveFileExW` と `FileRenameInfoEx`（POSIX semantics）
フォールバックのどちらの段で成功・失敗したかは判別できない。

### CUDA（CPU VM＝GPU・ドライバなし）

`cuda_probe` で `Device::Cuda` を選択すると panic せず `CudaUnavailable("CUDA driver library unavailable: ...")` の型付きエラー
（`cpu/p4-cuda-probe-nogpu.log`）。`RUST_AI_CUDA_CACHE_DIR` を設定してもディレクトリは作られない
（`cache_dir_exists_after=False`）。

### CUDA（GPU VM。Tesla T4）

| 条件 | 結果 | ログ |
|---|---|---|
| PATH に NVRTC なし | `Device::Cuda` の選択は成功。GEMM 実行時に `CudaUnavailable("CUDA NVRTC library unavailable: ...")` の型付きエラー | `gpu/g-cuda-probe-no-nvrtc-path.log` |
| PATH に NVRTC（`bin\x64\`）を追加 | NVRTC はロードされるが、全カーネルのコンパイルが `cuda_fp16.h` を開けず `NVRTC_ERROR_COMPILATION`（`--gpu-architecture=compute_75`） | `gpu/g-cuda-probe-with-nvrtc-path.log`・`gpu/g-cuda-*.log` |
| 上記＋ヘッダを 1 ディレクトリへまとめ `CUDA_INCLUDE_PATH` で指定 | 全件成功（下表） | `gpu/g2-*` |

初回（`g-*`）の失敗原因は `crates/backend-cuda/src/nvrtc.rs` の `compile_ptx` の include 候補が `CUDA_INCLUDE_PATH` と
Linux 固定パスのみで、Windows の CUDA ヘッダ位置を探さないこと（起票候補 1）。初回は `device_init` 1 passed のみ成功し、
`cpu_cuda_parity`・`gemm_naive`・`gemm_tiled`・`gemm_transposed_parity`・`gemm_batched_parity` は全件
`cuda_fp16.h` 不在で失敗した。

`CUDA_INCLUDE_PATH` 指定後（`g2-*`。`--ignored --test-threads=1`）:

| テスト | 結果 |
|---|---|
| `cpu_cuda_parity` | 2 passed |
| `gemm_naive` | 3 passed |
| `gemm_tiled` | 6 passed |
| `gemm_transposed_parity` | 5 passed |
| `gemm_batched_parity` | 2 passed |
| `device_init`（初回 `g-*` から） | 1 passed |

`cuda_probe`（`gpu/g2-cuda-probe-with-include.log`）: 64x64x64 GEMM が CPU 参照と統一複合判定で `fail_count=0`
（`max_abs_diff` 0）。1 回目 約 2.6 s（NVRTC コンパイル）→ 2 回目以降 約 1.2〜1.5 ms（プロセス内再利用）。
`RUST_AI_CUDA_CACHE_DIR` を設定してもディレクトリ・ファイルは作られない（`created_dir=false files_written=false`）。
GPU VM でも facade `interop_onnx_external_data` は 2 passed（`gpu/g-facade-onnx-external-data.log`）。

### facade の全テスト（CPU VM）と失敗 9 件（起票候補 2）

`cargo test -p fandhe-ai --locked --no-fail-fast` は exit 101（`cpu/p3-facade-all.log`）。失敗は次の 9 件で、いずれも
Windows 固有のテスト分離漏れ（下記）であり、封じ込め・external data の失敗ではない。

| target | 失敗 | 原因（ログの panic メッセージ） |
|---|---|---|
| `--lib` | 5 件: `fs_guard::tests::open_leaf_checked_reads_regular_file_exactly`・`..._rejects_over_bound_without_reading`・`model::tests::load_with_limit_rejects_over_bound`・`resolve_model_file_with_limit_accepts_at_bound`・`..._rejects_over_bound`（196 passed） | 実装は非 Linux/macOS で設計どおり `Unsupported`（「葉ファイルのシンボリックリンク追跡なしオープンを実装していない」）を返すが、テストが cfg 分離されていない |
| `tests/model_registry.rs` | 3 件: `corrupted_file_is_reported_as_load_error`・`load_roundtrips_state_dict_bit_exact`・`available_models_lists_only_complete_entries_sorted`（5 passed） | 同上 |
| `tests/api_surface.rs` | 1 件: `tape_ref_declared_once_with_crate_private_field`（296 passed） | `src/lib.rs` の読み込み・パス処理（panic メッセージは「宣言は src/lib.rs: C:\work\src\crates\facade\src\lib.rs」。原因の詳細は未調査） |

## 検証ハーネス（`harness-src/`）

使い捨て（`win2393-harness`）。全 bin は `rec kind=... key=value` を 1 行 1 レコードで出し、最後に `summary ...` 行を出す。
詳細な使い方は [`harness-src/USAGE.md.txt`](./harness-src/USAGE.md.txt)（`Cargo.toml.txt`・`src__*.rs.txt` と併せて参照）。

| bin | 役割 |
|---|---|
| `gen_model` | external data 付きの決定的モデルを生成（`<layout>/w.bin` と `.onnx`。64 MiB チャンクの複数テンソル） |
| `load_once` | `OnnxModel::from_path` → `run` を 1 回行い値の正当性・エラー bucket を報告（ACL 検証用） |
| `rename_race` | 読み込み中に祖先ディレクトリを rename する攻撃（連続方式と `--delay-us` の遅延掃引方式） |
| `flip_revert` | 途中成分を同じ深さの別ディレクトリへの junction へ入れ替える flip-and-revert 攻撃 |
| `reparse_nonempty` | 空・非空ディレクトリへ mount point reparse を設定（`FSCTL_SET_REPARSE_POINT`） |
| `wsave` | `rename`・`create_new` の置換先が各種リンクのときの挙動（W-save-2〜4） |
| `cuda_probe` | `Device::Cuda` 選択・64x64x64 GEMM の CPU 参照との統一複合判定・NVRTC キャッシュ非作成の確認 |

## 注意

- **ReFS・exFAT の `onnx_external_data` 実行は、hard link テストを含むバイナリ（54 件）で行った**。NTFS（53 件）との差は、
  hard link の一時 cfg 変更を戻した後、ファイルの mtime が古いまま戻ったため cargo が再ビルドしなかったことによる
  （`cpu/p3-hardlink-temp-cfg.log` は `53 filtered out`＝全 54 件）。ReFS では hard link テストも pass、exFAT では上記のとおり
  hard link 作成が失敗した。
- facade は external data のエラーを `Io(Kind(..))` へ写像する際に raw OS エラーを落とすため、`Uncategorized` の内訳
  （共有違反等）はハーネスから判別できなかった（起票候補 5）。
- 物理機ではなく GCE VM。ファイルシステム意味論の確認としては同等だが、ハードウェア依存の差は対象外。
- `rename_race` は単発の遅延掃引（各遅延 4〜10 サンプル）で、タイミングを網羅したものではない。結論は「観測した範囲で
  不正値ゼロ」であり、全タイミングでの不在証明ではない。
- 攻撃・ACL 検証は管理者または検証用の非管理者ユーザーによる単一ホスト実験で、AppExecLink・クラウドプレースホルダ
  （OneDrive 等）の reparse タグは未検証。

## 期待と異なった挙動・起票候補（起票はしていない。ユーザー承認待ち）

1. **backend-cuda**: NVRTC の include 候補に Windows の CUDA ヘッダ位置（`CUDA_PATH\include` 等）がなく、
   `CUDA_INCLUDE_PATH` 未指定だと Windows で全 CUDA カーネルのコンパイルが失敗する（`gpu/g-*`）。
2. **facade**: `fs_guard`／`model` の単体テスト 5 件と `tests/model_registry.rs` 3 件が Windows で失敗する（実装は非 Linux/macOS
   で設計どおり `Unsupported` を返すがテストが cfg 分離されていない）。`tests/api_surface.rs` の
   `tape_ref_declared_once_with_crate_private_field` 1 件も失敗（`src/lib.rs` の読み込み・パス処理）。
3. リポジトリに Windows 予約名のディレクトリ `docs/perf/logs/cuda-gemm-vjp-transposed-entry-1590/aux` があり、Windows では
   作成できない（git clone／展開で失敗する）。
4. **onnx-interop**: `overlapping_regions_via_hard_link_are_rejected` を `cfg(any(unix, windows))` へ広げられる
   （NTFS・ReFS で pass）。
5. **facade**: external data のエラー写像で raw OS エラーが失われ診断しにくい（`Io(Kind(Uncategorized))`）。

既存の起票候補（`onnx-external-data-decision.md` §10.2 P2 の `#[link(name = "kernel32")]` 明示化・§10.4 の USN 変更検知・
`FileIdInfo`〈128 bit ID〉への切替）は本実測では変わらず、そのまま残る。

## 期待と異なった点（訂正）

- `onnx-external-data-decision.md` §5・§8 は「読み込み中の祖先 rename は共有違反で失敗」と記していたが、実測では
  **主に `ERROR_ACCESS_DENIED`(5)**（共有違反 32 は開始直後の約 1 ms 以内のみ）だった。いずれも rename が失敗する点は変わらない。
- `compat-model-io-decision.md` §12.4 の「`create_new` は symlink があれば `AlreadyExists` になる想定」は、ファイル symlink・
  dangling のみ成立し、ディレクトリ symlink・junction は `PermissionDenied`(5) だった。
- `facade-windows-build-decision.md` §9-3 の想定エラー名（`DriverUnavailable`／`NvrtcUnavailable`）は、実際には
  `CudaUnavailable("CUDA driver library unavailable ...")`／`CudaUnavailable("CUDA NVRTC library unavailable ...")` だった
  （panic せず型付きエラーである点は想定どおり）。
