# CUDA JIT コンパイルキャッシュ ディレクトリ解決規則

- 対応イシュー: 親 #503（Phase C: CUDA JIT shape 特化・コンパイルキャッシュ・静的タイル選定）／C-1 #504（キー型定義）／C-2 #506（自作非暗号ハッシュ・ディレクトリ命名規則。PR #659）／C-3 #509（一時ディレクトリコンパイル → アトミック rename によるキャッシュ書き込み。本節を本タスクで更新）
- 位置づけ: 本文書は `crates/backend-cuda/src/nvrtc.rs` のキャッシュルート解決・エントリパス組み立て・実 I/O ロジックの**利用者向け参照**である。実装本体のドキュメンテーションコメント（同ファイル）を正とし、本文書はそれを要約・横断参照可能な形にまとめたものにすぎない（二重管理を避けるため詳細ロジックはコードコメント側で保守する）。
- **現状（C-3・#509 実装後）**: `resolve_cache_root`／`cache_root`／`cache_entry_path`／`ensure_cache_root`／`store_cache_entry`／`load_cache_entry` はいずれも `pub(crate)`（crate 内限定）であり、`backend-cuda` クレートの外から呼び出す手段はまだない。C-3 でディスクへの実際の読み書き（`ensure_cache_root` の `create_dir_all`・symlink 解決込み containment 再検証、`store_cache_entry`／`load_cache_entry` の一時ディレクトリ書き込み・fsync・アトミック rename）を実装したことで、以下の環境変数は**実効化されている**（crate 内公開範囲の判断は `crates/backend-cuda/src/lib.rs` 直下 `pub use` から意図的に除外している。理由は同ファイル該当関数のドキュメンテーションコメント参照）。ただし GEMM 経路への結線（NVRTC コンパイル成功後に `store_cache_entry` を呼ぶ導線・プロセス内 LRU）は C-4（#511）のスコープであり、本タスク時点では未結線（先行スキャフォールディング）。

## 環境変数と優先順位

キャッシュのルートディレクトリは以下の優先順位で解決する（`nvrtc::resolve_cache_root` 実装。値は全て**絶対パスであることを要求**し、相対パス・空文字列は `CudaError::CacheDirUnavailable` として拒否する）。

1. `RUST_AI_CUDA_CACHE_DIR` — 明示的な上書き。DeepGEMM の `DG_JIT_CACHE_DIR` に相当する本リポジトリ独自の命名。値をそのままキャッシュルートとして使う
2. `XDG_CACHE_HOME` — 設定されていれば `${XDG_CACHE_HOME}/rust-ai-library/cuda` を使う
3. `HOME` — 上記 2 つが未設定の場合、`${HOME}/.cache/rust-ai-library/cuda` を使う（一般的な `~/.cache` 相当の Linux 慣行に対する最終フォールバック）

3 つとも未設定（環境変数が全欠落）の場合は `CudaError::CacheDirUnavailable` を返す（panic しない。呼び出し元がコンパイルキャッシュを使わない経路へフォールバックするか、エラーとして利用者へ伝播するかは C-3（#509）以降のスコープ）。

## 検証条件（A03 インジェクション対応）

`RUST_AI_CUDA_CACHE_DIR`・`XDG_CACHE_HOME`・`HOME` はいずれも外部プロセス環境変数由来の信頼できない入力として扱い、以下を満たさない値は三者とも同様に拒否する（`.claude/rules/security.md` A03 節）。

- 空文字列でないこと（`Some("")` のように変数自体は設定されているが値が空、というケースも「未設定」扱いでフォールバックせず明示的に拒否する。PR #659 codex-review P0 指摘: 旧実装は空文字列の `XDG_CACHE_HOME` を素通りさせ `HOME` へフォールバックしていた）
- 絶対パスであること（相対パスはリポジトリツリー内への意図しない書き込みを招きうるため拒否）

**`workspace_root` containment 検証（PR #659 codex-review P0 再指摘への対応。2 回目の設計変更）**: 1 回目の修正（P1 指摘対応）ではコンパイル時 `CARGO_MANIFEST_DIR` から導出する「ビルド時ワークスペースルート」（旧 `nvrtc::compile_time_workspace_root`）との比較を削除し、containment 検証自体を持たない許可リスト方式に置き換えていた。しかしこれは `RUST_AI_CUDA_CACHE_DIR` がリポジトリツリー内を指す絶対パス（例 `/workspace/repository/cache`）をそのまま受理してしまう回帰であり、codex-review に P0 として再指摘された。「ビルド時定数のハードコードは実行環境が変わると素通りする」という 1 回目の指摘の要点自体は正しいが、対策は検証を削除することではなく**信頼できる境界を呼び出し元から注入で受け取る**ことだった。

現行実装（`nvrtc::resolve_cache_root`）は `workspace_root: &Path` を**必須引数**（`Option` ではない）として受け取り、`RUST_AI_CUDA_CACHE_DIR`・`XDG_CACHE_HOME`・`HOME` の 3 分岐すべてで、解決結果が `workspace_root` 配下（`nvrtc::path_lexically_within` による `..` 折り畳み込みの字句正規化済み比較）でないことを検証し、配下であれば `CudaError::CacheDirUnavailable` で拒否する。`RUST_AI_CUDA_CACHE_DIR` だけを検証対象外にする例外は設けない（それが P0 再指摘の核心のため）。`workspace_root` を `Option` にして呼び出し元が `None` を渡せる余地を残すと検証を迂回できる構造が復活するため、必須引数として契約に組み込んでいる。

`workspace_root` を「どの値にするか」はビルド時定数のハードコードを避けつつ、かつプロセスのカレントディレクトリ（`std::env::current_dir()`）のような「プロセス起動時の作業ディレクトリ」を安易に使わない（`XDG_CACHE_HOME`／`HOME` 未設定でカレントディレクトリがたまたまホームディレクトリ配下だった場合、`~/.cache/...` という正当なフォールバック結果が誤って拒否される）。C-3（#509）が実際にディレクトリを作成・オープンする際に、実行時に確定する信頼できる境界（例: 明示設定される runtime workspace 設定値）をどう決定するかを含めて設計する。加えて、`canonicalize` 済みパスによる symlink 解決込みの再検証（symlink 解決込みの実在ベース検証はパスの実在を要求するため、fs I/O を行わない C-2 の純関数では原理的に実行できない）も C-3 のスコープとして残る。

さらに、キャッシュエントリパス（`cache_entry_path`）の組み立て結果は必ず解決済みルート配下（`starts_with(root)`）に収まることを保証する多層防御を持つ（第 1 層: `CudaKernelDescriptor::new` の構築時検証、第 2 層: `CudaKernelCacheKey::cache_entry_dir_name` 内の縦深防御検査、第 3 層: `cache_entry_path_in` のユニットテスト）。「ルート自体がリポジトリツリー外にある」ことは上記の `workspace_root` containment 検証（`resolve_cache_root` 側）が担う（第 0 層）。

## ディレクトリ命名規則（C-2・#506）

キャッシュエントリはキャッシュルート直下に `kernel.<name>.<hash>` の形式で配置する。`<hash>` は [`CudaKernelCacheKey::canonical_bytes`] を自作の非暗号ハッシュ（FNV-1a 64bit。std のみで実装。依存クレート追加なし）でハッシュ化した値の 16 桁 16 進表記。非暗号ハッシュを選んだ理由・改竄検知に使わない旨は `crates/backend-cuda/src/nvrtc.rs` の `fnv1a_64` ドキュメンテーションコメントを参照。

## エントリ内ファイル構成と書き込みプロトコル（C-3・#509）

最終エントリ（`<cache_root>/kernel.<name>.<hash16>/`）直下には `kernel.cu`（NVRTC へ渡したソース全文）と `kernel.ptx`（コンパイル結果の PTX アセンブリ全文）の 2 ファイルを置く。**不変条件**: 有効なエントリは両ファイルが存在する。片方欠落は破損とみなし、読み出し側（`load_cache_entry`）はミス（`Ok(None)`）として扱う（削除は行わない。置き換えは書き込み側に一元化）。

この不変条件の判定は本番経路とテスト経路で実装が分かれる（イシュー #509 PR #677 codex-review P0 指摘対応。両者は同じ判定基準〈エントリディレクトリ・両ファイルとも symlink を追跡しない実在確認〉を守るが、TOCTOU 耐性の作り方が異なる）:

- **本番経路**: `store_cache_entry_in`／`load_cache_entry_in`（いずれも Unix 版）は `nvrtc::validate_cache_entry_at`（`root_fd` からの fd 相対判定）を使う。検証と実際の作成・削除・読み取りを同一の pin 済みディレクトリ fd に結合することで、パスを再解決する隙（TOCTOU）を構造的に閉じる
- **テスト経路**: `nvrtc::validate_cache_entry`（パスベース版）は `#[cfg(test)]` 限定で、外部から symlink 差し替え等を観測するアサーション専用に残す。本番経路からは呼ばれない

読み出し側はさらに 2 つの追加検査を行う（実装計画 §3.1・§7 のハッシュ衝突安全弁）。いずれも通常運用で偽陽性にならない（`store_cache_entry_at` は非空バイト列を書き込んだ後に一時ディレクトリを fsync してから rename するため）ため、`Ok(None)`（ミス）扱いにしてよい:

1. **非空検査**: `kernel.cu`／`kernel.ptx` のいずれかが 0 バイトならミス（クラッシュ残骸 0 バイトファイルを有効なエントリと誤認しない）
2. **ソース照合**: `load_cache_entry`／`load_cache_entry_in` は呼び出し元がこれからコンパイルしようとしているソース全文（`expected_src`）を引数に取り、保存済み `kernel.cu` とバイト単位で照合する。不一致ならミス。エントリ名のハッシュ（FNV-1a 64bit・非暗号）が偶然衝突した場合に、別ソースの PTX を誤ってヒット扱いして GPU 上で実行してしまう経路を閉じる（C-4・#511 配線後、NVRTC 結線でこのフィールドが実際に GPU 上で実行される PTX になるため重要）

`store_cache_entry`（実体は `store_cache_entry_in` → `store_cache_entry_at`）は DeepGEMM の compiler（一時ディレクトリでビルド後 rename、先着プロセスがいた場合は rename 失敗を正常系として吸収）に倣うが、TOCTOU 対策のため全操作を「キャッシュルートを指す 1 個の pin 済みディレクトリ fd（`root_fd`）」からの fd 相対操作に統一している（イシュー #509 PR #677 codex-review P0 指摘対応。`root.join(..)` のようなパス再解決を経由すると、pin 後に `root` 自体が別ディレクトリへの symlink へ差し替えられた場合に symlink 先を誤って操作しうる）。

1. `ensure_cache_root` でキャッシュルートを実体化する。**祖先ディレクトリを検証してから作成する**順序を守る（作成してから検証する旧設計は、拒否確定前に workspace 内へ書き込みが発生しうる契約違反だった。イシュー #509 codex-review P0 再指摘対応）: `longest_existing_ancestor` で fs 上に実在する最長の祖先を求め、その祖先を `canonicalize`（symlink 解決込み）して `path_lexically_within` による containment 事前検証を行ってから、`create_dir_all_verified`（pin 済みディレクトリ fd 起点で 1 コンポーネントずつ作成・検証を結合する。Linux 版は `mkdirat`／`openat` の FFI 直接呼び出し、非 Linux〈macOS〉版は `openat_nofollow` 系の同等実装）で残りのコンポーネントを実体化する。最後に実体化後の `canonical_root` を再度 containment 検証する（事前検証と作成の間の TOCTOU に対する縦深防御）
2. `root_fd`（`ensure_cache_root` が返したルートを `open_dir_nofollow` で pin したディレクトリ fd）を起点に、一時ディレクトリ `.tmp.<final_entry_name>.<pid>.<seq>` を fd 相対（`create_subdir_pinned`）で排他作成する
3. `kernel.cu`／`kernel.ptx` を pin 済みの一時ディレクトリ fd から直接書き込み・fsync し（`write_child_file_pinned`。書き込みに使ったハンドルから直接 `sync_all` するためパス再オープンの TOCTOU 窓がない）、一時ディレクトリ自体も fsync する（rename 前にディスクへ反映されていることを保証。DeepGEMM `fsync_dir` 相当のボトムアップ fsync）
4. 最終パスへアトミックに配置する（`rename_pinned`。Linux 版は `/proc/self/fd/<fd>/<name>` 経由の `std::fs::rename`、macOS 版は `renameat(2)` の FFI 直接呼び出し〈`renameat_raw`〉——いずれも fd 相対で pin 済みディレクトリを起点にパスの 1 コンポーネントのみを解決し、`root` のパス再解決を挟まない）。失敗時は最終パスの既存エントリを `validate_cache_entry_at`／`entry_exists_at`（ディレクトリ・通常ファイル・symlink のいずれの占有も検出する。イシュー #509 PR #677 Bugbot 指摘対応）で検査し、有効なら「他プロセス先着」として正常系吸収（`Ok`）、破損なら退避名へ `rename_pinned` で一意に固定してから削除し一度だけ再試行、それでも失敗すれば `CudaError::CacheIo` で fail-closed に失敗する（無限リトライしない）。後始末（一時ディレクトリ・退避ディレクトリの削除）も `remove_cache_entry_pinned`（fd 相対）で行い、`fs::remove_dir_all` のようなパス再解決は使わない

## ソース断片の取り込み（C-5・#514）

`CudaKernelCacheKey` は descriptor・環境パラメータに加え、最終レンダー済みカーネルソース全体（`source: String`）をキーへ含める。

- **必要性判断**: DeepGEMM 型の「`#include` を正規表現抽出して再帰的にハッシュへ取り込む」機構は不要と判断した。本クレートのカーネルソースは `kernels_mma::render_mma_f16` 等がプロセス内で最終 `String` を確定させ、リポジトリ内ヘッダファイルへの `#include` 参照を持たないため（toolkit 標準ヘッダの変更は既存の `nvrtc_version` フィールドが追従する）。最終ソース文字列そのものをキーへ含めれば、断片（`kernels_mma.rs` の `*_BODY` 定数等）をどう編集しても推移的にキーが変わり、DeepGEMM の再帰ハッシュと同じ「ソース変更で確実にキャッシュミスする」性質を、ファイルパース・fs I/O ゼロで得られる。判断根拠の詳細は `crates/backend-cuda/src/nvrtc.rs` の `CudaKernelCacheKey` ドキュメンテーションコメントを正とする。
- **エンコーディング**: `canonical_bytes` の `ENCODING_VERSION` を `1` → `2` へ上げ、`compile_flags` の後段に `source` を長さプレフィクス付きで追記した。C-2（#506）時点のディスクキャッシュエントリ（C-3・#509 実装後に実体化）は本変更により全て無効化される契約（意図どおり）。
- **情報露出対策**: `source`（数十 KB になりうる）は `derive(Debug)` をやめ手動 `Debug` 実装とし、ログ・パニックメッセージには長さと非暗号な変更検知用フィンガープリント（FNV-1a 64bit。`stable_hash` と同一アルゴリズム）のみを出す（PR #676 codex-review P1 是正。当初案の先頭 40 文字平文要約はカーネル名・シグネチャ等の識別情報を含みうる部分的漏出だったため撤回した）。外部公開 getter は追加していない（`RenderedMmaKernel` がソース文字列を外部へ返さない設計〈PR #643〉と同じ理由）。

## 関連

- `crates/backend-cuda/src/nvrtc.rs`: 実装本体（`resolve_cache_root`／`cache_root`／`cache_entry_path`／`cache_entry_path_in`／`fnv1a_64`／`ensure_cache_root`／`ensure_cache_root_in`／`store_cache_entry`／`store_cache_entry_in`／`store_cache_entry_at`／`load_cache_entry`／`load_cache_entry_in`／`validate_cache_entry_at`（本番経路）／`validate_cache_entry`（`#[cfg(test)]` 限定）／`rename_pinned`／`create_dir_all_verified`／`CudaKernelCacheKey`）
- `crates/backend-cuda/src/error.rs`: `CudaError::CacheDirUnavailable`／`CudaError::CacheIo`
- C-4（#511）: プロセス内 LRU カーネルキャッシュ・GEMM 経路への結線（NVRTC コンパイル成功後に `store_cache_entry` を呼ぶ導線）
- C-10（#529）: ヒット/ミス・並行競合・破損検出の網羅的回帰テスト拡充（C-3 時点のユニットテストは受け入れ基準を直接検証する最小限に留める）
