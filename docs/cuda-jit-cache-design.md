# CUDA JIT shape 特化・コンパイルキャッシュ設計

- 対応イシュー: 親 #503（Phase C: CUDA JIT shape 特化・コンパイルキャッシュ・静的タイル選定）／C-1 #504（キー型定義）／C-2 #506（自作非暗号ハッシュ・ディレクトリ命名規則。PR #659）／C-3 #509（一時ディレクトリコンパイル → アトミック rename によるキャッシュ書き込み）／C-4 #511（プロセス内 LRU カーネルモジュールキャッシュ・GEMM 経路への結線）／C-5 #514（ソース断片の取り込み・`ENCODING_VERSION` v2 化）／C-6・C-7（#516・#519。テンプレート展開・次元別定数化選択・`ENCODING_VERSION` v3 化）／C-8・C-9a・C-9b（#521・#524・#527。SMEM 予算からの段数逆算・候補列挙・L1/L2 帯域コストモデル）／C-10（#529。網羅的回帰テスト）／C-12（#534。性能計測）／**C-14 #539（本タスク。JIT shape 特化・キャッシュ設計とコストモデル定数の実機再測定手順の記録。総括更新）**
- 位置づけ: 本文書は `crates/backend-cuda/src/nvrtc.rs` のキャッシュキー構成・キャッシュルート解決・エントリパス組み立て・実 I/O ロジック、および `gemm_auto.rs::cost_model` のコストモデル定数の**利用者向け参照**である。実装本体のドキュメンテーションコメント（`nvrtc.rs`・`gemm_auto.rs`）を正とし、本文書はそれを要約・横断参照可能な形にまとめたものにすぎない（二重管理を避けるため詳細ロジックはコードコメント側で保守する）。C-14（#539）はコード変更を伴わない docs のみのタスクであり、本節以下は既存記述の総括・ドリフト修正（下記「現状」の `ENCODING_VERSION` 訂正）と新設節（§キャッシュキー構成の総括・§キャッシュ無効化条件の総括・§shape 特化と静的タイル選定・§コストモデル定数の実機再測定手順）の追加のみを行う
- **現状（C-4・#511 実装後）**: `resolve_cache_root`／`cache_root`／`ensure_cache_root`／`store_cache_entry`／`load_cache_entry` はいずれも `pub(crate)`（crate 内限定）のまま（`backend-cuda` クレートの外から呼び出す手段はない）だが、C-4 で GEMM 経路への結線（`kernels_mma.rs::RenderedMmaKernel::compile` が `runtime_workspace_root()` で `workspace_root` を解決し、コンパイル前に `load_cache_entry` を引き・コンパイル成功後に `store_cache_entry` を呼ぶ導線）とプロセス内 LRU（`module_cache.rs::KernelModuleCache`。ロード済み `Arc<CudaModule>` の再利用）を実装した。これにより以下の環境変数は実際に GEMM 実行経路から**実効化されている**（`cache_entry_path` のみ、fd pin を経由しない便宜 API として crate 内呼び出し元を持たないまま残置。理由は同関数のドキュメンテーションコメント参照）。ディスクキャッシュ関連の失敗（`workspace_root` 解決不能・fs I/O 失敗）はコンパイル失敗にせず「ディスクキャッシュなしの縮退運転」（NVRTC 直コンパイル＋プロセス内 LRU のみ）へフォールバックする fail-safe 方針を採る（`RenderedMmaKernel::compile` ドキュメンテーションコメント参照）。

## キャッシュ配置ポリシー（要約）

キャッシュのルート（ディスク側）はリポジトリツリー外への配置を **fail-closed で強制**する（`workspace_root` containment 検証。詳細は下記「検証条件」節）。環境変数による上書きは `RUST_AI_CUDA_CACHE_DIR` > `XDG_CACHE_HOME` > `HOME` の優先順位（詳細は次節）。境界（`workspace_root`）自体が解決できない場合・全環境変数が未設定の場合は、コンパイル失敗にはせずディスクキャッシュなしの縮退運転（NVRTC 直コンパイル＋プロセス内 LRU のみ）へ fail-safe にフォールバックする（詳細は下記「プロセス内 LRU カーネルモジュールキャッシュと GEMM 経路への結線」節「縮退方針」）。

## 環境変数と優先順位

キャッシュのルートディレクトリは以下の優先順位で解決する（`nvrtc::resolve_cache_root` 実装。値は全て**絶対パスであることを要求**し、相対パス・空文字列は `CudaError::CacheDirUnavailable` として拒否する）。

1. `RUST_AI_CUDA_CACHE_DIR` — 明示的な上書き。DeepGEMM の `DG_JIT_CACHE_DIR` に相当する本リポジトリ独自の命名。値をそのままキャッシュルートとして使う
2. `XDG_CACHE_HOME` — 設定されていれば `${XDG_CACHE_HOME}/rust-ai-library/cuda` を使う
3. `HOME` — 上記 2 つが未設定の場合、`${HOME}/.cache/rust-ai-library/cuda` を使う（一般的な `~/.cache` 相当の Linux 慣行に対する最終フォールバック）

3 つとも未設定（環境変数が全欠落）の場合は `CudaError::CacheDirUnavailable` を返す（panic しない。呼び出し元がコンパイルキャッシュを使わない経路へフォールバックするか、エラーとして利用者へ伝播するかは C-3（#509）以降のスコープ）。

上記 3 つはいずれもディスクキャッシュ（`nvrtc.rs`）の配置先を決める。プロセス内 LRU カーネルモジュールキャッシュ（C-4・#511。下記「プロセス内 LRU カーネルモジュールキャッシュと GEMM 経路への結線」節）は別系統の環境変数 `RUST_AI_CUDA_MODULE_CACHE_CAPACITY`（容量。既定 `32`・許容範囲 `1..=1024`）を持つ。

## 検証条件（A03 インジェクション対応）

`RUST_AI_CUDA_CACHE_DIR`・`XDG_CACHE_HOME`・`HOME` はいずれも外部プロセス環境変数由来の信頼できない入力として扱い、以下を満たさない値は三者とも同様に拒否する（`.claude/rules/security.md` A03 節）。

- 空文字列でないこと（`Some("")` のように変数自体は設定されているが値が空、というケースも「未設定」扱いでフォールバックせず明示的に拒否する。PR #659 codex-review P0 指摘: 旧実装は空文字列の `XDG_CACHE_HOME` を素通りさせ `HOME` へフォールバックしていた）
- 絶対パスであること（相対パスはリポジトリツリー内への意図しない書き込みを招きうるため拒否）

**`workspace_root` containment 検証（PR #659 codex-review P0 再指摘への対応。2 回目の設計変更）**: 1 回目の修正（P1 指摘対応）ではコンパイル時 `CARGO_MANIFEST_DIR` から導出する「ビルド時ワークスペースルート」（旧 `nvrtc::compile_time_workspace_root`）との比較を削除し、containment 検証自体を持たない許可リスト方式に置き換えていた。しかしこれは `RUST_AI_CUDA_CACHE_DIR` がリポジトリツリー内を指す絶対パス（例 `/workspace/repository/cache`）をそのまま受理してしまう回帰であり、codex-review に P0 として再指摘された。「ビルド時定数のハードコードは実行環境が変わると素通りする」という 1 回目の指摘の要点自体は正しいが、対策は検証を削除することではなく**信頼できる境界を呼び出し元から注入で受け取る**ことだった。

現行実装（`nvrtc::resolve_cache_root`）は `workspace_root: &Path` を**必須引数**（`Option` ではない）として受け取り、`RUST_AI_CUDA_CACHE_DIR`・`XDG_CACHE_HOME`・`HOME` の 3 分岐すべてで、解決結果が `workspace_root` 配下（`nvrtc::path_lexically_within` による `..` 折り畳み込みの字句正規化済み比較）でないことを検証し、配下であれば `CudaError::CacheDirUnavailable` で拒否する。`RUST_AI_CUDA_CACHE_DIR` だけを検証対象外にする例外は設けない（それが P0 再指摘の核心のため）。`workspace_root` を `Option` にして呼び出し元が `None` を渡せる余地を残すと検証を迂回できる構造が復活するため、必須引数として契約に組み込んでいる。

`workspace_root` を「どの値にするか」はビルド時定数のハードコードを避ける（C-3・#509 で確定した方針）。C-4（#511）は結線時点でこの `workspace_root` を実際に決定する必要がある。

**`current_dir()` を境界そのものとしては使わない（イシュー #511 PR #703 codex-review P0 指摘への対応。設計変更）**: 当初案は `nvrtc::runtime_workspace_root()` が `std::env::current_dir()`（取得成功時は `canonicalize`）の結果をそのまま `workspace_root` として採用していた。しかしこれは「境界解決に失敗する方向」（false reject。誤拒否の帰結は「ディスクキャッシュが効かないだけ」で fail-safe）のみを正当化しており、「誤って広すぎる境界を受理する方向」（false accept）を防げていなかった。実際、プロセスの cwd をリポジトリツリー外に設定したまま `RUST_AI_CUDA_CACHE_DIR` にリポジトリツリー内の絶対パスを指定すると、`resolve_cache_root` の containment 検証は「cwd（＝誤った境界）の配下でないこと」しか確認できず、本来拒否すべき「リポジトリツリー内へのキャッシュ書き込み」を通過させてしまっていた。

現行実装の `nvrtc::runtime_workspace_root()` は、`canonicalize` 済みの cwd を起点に祖先方向へ実際の境界マーカー（`.git`、または `[workspace]` セクションを持つ `Cargo.toml`）を探索する（`nvrtc::find_workspace_root_from`／`nvrtc::has_workspace_root_marker`）。マーカー探索が cwd から到達できない場合（cwd がリポジトリツリー外にある等）は `None` として扱い、許容側へフォールバックせず `CudaError::CacheDirUnavailable` を返す（「境界を解決できないなら縮退運転〈ディスクキャッシュなし〉へ倒す。誤った境界を許容側で埋めない」という契約を境界解決自体にも一貫させる）。加えて `has_workspace_root_marker` はマーカー候補ディレクトリ自体の所有 uid（このプロセスの実効 uid と一致すること）・group／other 書き込みビット（`mode & 0o022 == 0` であること）を検査してから初めて `.git`／`Cargo.toml` の中身を見る（イシュー #511 PR #703 codex-review Bugbot 指摘〈Forgeable workspace root markers〉・P0 再指摘〈group 書き込みビット除外〉への対応。共有祖先ディレクトリ配下に攻撃者が偽のマーカーを仕込んで `workspace_root` を偽装する経路を遮断する）。この検査の帰結として、umask `002` の group 共有ワークツリー等では `workspace_root` 解決が常に失敗しディスクキャッシュが常に効かない縮退運転になりうるが、これは「他グループメンバーによる workspace boundary 偽装を許す」という P0 より安全側であるため受け入れている。判断の詳細は `nvrtc::runtime_workspace_root`／`nvrtc::has_workspace_root_marker` のドキュメンテーションコメントを正とする。加えて、`canonicalize` 済みパスによる symlink 解決込みの再検証（symlink 解決込みの実在ベース検証はパスの実在を要求するため、fs I/O を行わない C-2 の純関数では原理的に実行できない）は C-3（`ensure_cache_root`）が実ディレクトリのオープン時点で担う。

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
2. **ソース照合**: `load_cache_entry`／`load_cache_entry_in` は呼び出し元がこれからコンパイルしようとしているソース全文（`expected_src`）を引数に取り、保存済み `kernel.cu` とバイト単位で照合する。不一致ならミス。エントリ名のハッシュ（FNV-1a 64bit・非暗号）が偶然衝突した場合に、別ソースのエントリを誤ってヒット扱いする経路を閉じる。**ただし本照合は「保存済みソースが要求元ソースと一致すること」までしか保証せず、同一ディレクトリの `kernel.ptx` がそのソースから生成された成果物であることは保証しない**（この残余ギャップへの対応は下記「プロセス内 LRU カーネルモジュールキャッシュと GEMM 経路への結線（C-4・#511）」節「ディスク PTX を実行入力にしない判断」を参照。C-4 配線後の呼び出し元 `kernels_mma.rs::RenderedMmaKernel::compile` は disk hit を得ても `kernel_ptx` フィールドを `load_module` の実行入力へは使わない）

`store_cache_entry`（実体は `store_cache_entry_in` → `store_cache_entry_at`）は DeepGEMM の compiler（一時ディレクトリでビルド後 rename、先着プロセスがいた場合は rename 失敗を正常系として吸収）に倣うが、TOCTOU 対策のため全操作を「キャッシュルートを指す 1 個の pin 済みディレクトリ fd（`root_fd`）」からの fd 相対操作に統一している（イシュー #509 PR #677 codex-review P0 指摘対応。`root.join(..)` のようなパス再解決を経由すると、pin 後に `root` 自体が別ディレクトリへの symlink へ差し替えられた場合に symlink 先を誤って操作しうる）。

1. `ensure_cache_root` でキャッシュルートを実体化する。**祖先ディレクトリを検証してから作成する**順序を守る（作成してから検証する旧設計は、拒否確定前に workspace 内へ書き込みが発生しうる契約違反だった。イシュー #509 codex-review P0 再指摘対応）: `longest_existing_ancestor` で fs 上に実在する最長の祖先を求め、その祖先を `canonicalize`（symlink 解決込み）して `path_lexically_within` による containment 事前検証を行ってから、`create_dir_all_verified`（pin 済みディレクトリ fd 起点で 1 コンポーネントずつ作成・検証を結合する。Linux 版は `mkdirat`／`openat` の FFI 直接呼び出し、非 Linux〈macOS〉版は `openat_nofollow` 系の同等実装。拒否時の後始末〈`fs::remove_dir`／`unlinkat_raw`〉も同じ pin 済みディレクトリ fd 起点の magic path／dirfd 相対で行い、パスをルートから再解決しない。イシュー #509 PR #677 codex-review P0 再指摘対応）で残りのコンポーネントを実体化する。最後に、実体化後のルートを一度だけ `open_dir_nofollow` で pin し、その fd から得た実パス（`/proc/self/fd/<fd>`〈Linux〉／`F_GETPATH`〈macOS〉）で再度 containment を検証する（パスの再オープンではなく pin 済み fd 起点の再検証。事前検証と作成の間の TOCTOU に対する縦深防御）。`ensure_cache_root` は canonical パスとこの pin 済み fd の両方を返す
2. `root_fd`（`ensure_cache_root` が上記手順 1 で pin し、そのまま返した同一のディレクトリ fd。呼び出し元〈`store_cache_entry`／`load_cache_entry`〉は `root` をパスとして再オープンしない。イシュー #509 PR #677 codex-review P0 再指摘対応: 旧実装は `ensure_cache_root` が `PathBuf` のみを返し、`store_cache_entry_in`／`load_cache_entry_in` が改めて `root` を `open_dir_nofollow` で開き直していたため、検証と再オープンの間の窓で祖先を差し替えられるとその TOCTOU を再導入していた）を起点に、一時ディレクトリ `.tmp.<final_entry_name>.<pid>.<seq>` を fd 相対（`create_subdir_pinned`）で排他作成する
3. `kernel.cu`／`kernel.ptx` を pin 済みの一時ディレクトリ fd から直接書き込み・fsync し（`write_child_file_pinned`。書き込みに使ったハンドルから直接 `sync_all` するためパス再オープンの TOCTOU 窓がない）、一時ディレクトリ自体も fsync する（rename 前にディスクへ反映されていることを保証。DeepGEMM `fsync_dir` 相当のボトムアップ fsync）
4. 最終パスへアトミックに配置する（`rename_pinned`。Linux 版は `/proc/self/fd/<fd>/<name>` 経由の `std::fs::rename`、macOS 版は `renameat(2)` の FFI 直接呼び出し〈`renameat_raw`〉——いずれも fd 相対で pin 済みディレクトリを起点にパスの 1 コンポーネントのみを解決し、`root` のパス再解決を挟まない）。失敗時は最終パスの既存エントリを `validate_cache_entry_at`／`entry_exists_at`（ディレクトリ・通常ファイル・symlink のいずれの占有も検出する。イシュー #509 PR #677 Bugbot 指摘対応）で検査し、有効なら「他プロセス先着」として正常系吸収（`Ok`）、破損なら退避名へ `rename_pinned` で一意に固定してから削除し一度だけ再試行、それでも失敗すれば `CudaError::CacheIo` で fail-closed に失敗する（無限リトライしない）。後始末（一時ディレクトリ・退避ディレクトリの削除）も `remove_cache_entry_pinned`（fd 相対）で行い、`fs::remove_dir_all` のようなパス再解決は使わない

## ソース断片の取り込み（C-5・#514）

`CudaKernelCacheKey` は descriptor・環境パラメータに加え、最終レンダー済みカーネルソース全体（`source: String`）をキーへ含める。

- **必要性判断**: DeepGEMM 型の「`#include` を正規表現抽出して再帰的にハッシュへ取り込む」機構は不要と判断した。本クレートのカーネルソースは `kernels_mma::render_mma_f16` 等がプロセス内で最終 `String` を確定させ、リポジトリ内ヘッダファイルへの `#include` 参照を持たないため（toolkit 標準ヘッダの変更は既存の `nvrtc_version` フィールドが追従する）。最終ソース文字列そのものをキーへ含めれば、断片（`kernels_mma.rs` の `*_BODY` 定数等）をどう編集しても推移的にキーが変わり、DeepGEMM の再帰ハッシュと同じ「ソース変更で確実にキャッシュミスする」性質を、ファイルパース・fs I/O ゼロで得られる。判断根拠の詳細は `crates/backend-cuda/src/nvrtc.rs` の `CudaKernelCacheKey` ドキュメンテーションコメントを正とする。
- **エンコーディング**: `canonical_bytes` の `ENCODING_VERSION` を `1` → `2` へ上げ、`compile_flags` の後段に `source` を長さプレフィクス付きで追記した。C-2（#506）時点のディスクキャッシュエントリ（C-3・#509 実装後に実体化）は本変更により全て無効化される契約（意図どおり）。**現行は `ENCODING_VERSION = 3`**（C-7・#519・PR #674 でさらに引き上げ済み。詳細は下記「shape 特化と静的タイル選定」節・「キャッシュ無効化条件（総括）」節を参照。本節の `1 → 2` の記述は C-5 時点の変更履歴として維持し、v2 → v3 の変更理由は C-7 側の記述に譲る）。
- **情報露出対策**: `source`（数十 KB になりうる）は `derive(Debug)` をやめ手動 `Debug` 実装とし、ログ・パニックメッセージには長さと非暗号な変更検知用フィンガープリント（FNV-1a 64bit。`stable_hash` と同一アルゴリズム）のみを出す（PR #676 codex-review P1 是正。当初案の先頭 40 文字平文要約はカーネル名・シグネチャ等の識別情報を含みうる部分的漏出だったため撤回した）。外部公開 getter は追加していない（`RenderedMmaKernel` がソース文字列を外部へ返さない設計〈PR #643〉と同じ理由）。

## プロセス内 LRU カーネルモジュールキャッシュと GEMM 経路への結線（C-4・#511）

C-1〜C-3・C-5 が用意したキー型・ディレクトリ命名・ディスク I/O は、C-4 まで GEMM 経路へ未結線（`#[allow(dead_code)]` の先行スキャフォールディング）だった。C-4 は以下 2 点を実装する。

1. **プロセス内 LRU**（`crates/backend-cuda/src/module_cache.rs::KernelModuleCache`）: ロード済み `Arc<CudaModule>`（cudarc 0.19.8。`CudaContext::load_module` の戻り値）をプロセス内で再利用する容量上限つき LRU。std のみの自作実装（tick 方式・`HashMap<K, (V, tick)>`）で `lru` 等の追加依存はしない（許容依存 8 区分外。deps-policy.md）。キーは `(ctx_id, CudaKernelCacheKey)`（`ctx_id` は要求元 `Arc<CudaContext>` のポインタ識別。別 context への誤共有をキーレベルで遮断する）。容量は環境変数 `RUST_AI_CUDA_MODULE_CACHE_CAPACITY`（既定 `32`・許容範囲 `1..=1024`。不正値は `CudaError::InvalidModuleCacheCapacity` で fail-closed に拒否）で調整可能。プロセスワイドの唯一のインスタンス（`static` + `OnceLock`）として保持し、容量は初回利用時に一度だけ確定する。
2. **GEMM 経路への結線**（`crates/backend-cuda/src/kernels_mma.rs::RenderedMmaKernel::compile`）: 従来は呼び出しごとに NVRTC コンパイルしていた経路（`gemm_auto.rs::SpecializedMmaKernelHandle::compile` から到達する shape 特化経路）を、次のフォールバックへ結線した。
   1. プロセス内 LRU をキー（`self.cfg` から内部導出した `CudaKernelDescriptor`＋環境パラメータ＋`self.source`）で検索し、ヒットなら `cuModuleGetFunction` のみで済ませる
   2. ミスならディスクキャッシュ（`load_cache_entry`。ソース全文のバイト照合込み）を引く。ヒットしても、この段で得られる `kernel.ptx` は**実行入力としては使わない**（下記「ディスク PTX を実行入力にしない判断（イシュー #511 PR #703 codex-review P0 再指摘対応）」参照）。ヒット／ミスいずれの場合も `compile_ptx` を実行し、ミスの場合のみ成功後に `store_cache_entry` でディスクへ保存してから `load_module`
   いずれの段でロードした `Arc<CudaModule>` も最終的にプロセス内 LRU へ登録する。

**縮退方針（fail-safe）**: プロセス内 LRU（容量設定不正・`Mutex` poison）・ディスクキャッシュ（`workspace_root` 解決不能・fs I/O 失敗）いずれの失敗もコンパイル失敗にせず次の段（最終的には NVRTC 直コンパイル）へフォールバックする。両キャッシュは純粋な最適化であり、数値正しさは NVRTC 直コンパイル・ディスクキャッシュのソース全文照合のいずれでも独立に保たれるため、キャッシュ層の可用性低下が誤った PTX の実行につながることはない。

### ディスク PTX を実行入力にしない判断（イシュー #511 PR #703 codex-review P0 再指摘対応）

初回実装（本 PR #703 内の先行コミット）は上記 2 段目のディスクキャッシュヒット時、保存済み `kernel.ptx` を `Ptx::from_src` で直接 `load_module` していた。codex-review・Cursor Bugbot がいずれも P0 として指摘したとおり、`load_cache_entry` が検証するのは「保存済み `kernel.cu` が要求元ソースとバイト単位で一致すること」のみで、同一ディレクトリの `kernel.ptx` がそのソースから実際に生成された成果物であることまでは検証しない。追加した権限検査（`nvrtc::is_cache_entry_permission_untrusted`。エントリの mode が group／other 書き込み不可であり、かつ所有 uid がこのプロセスの実効 uid と一致することを要求）を経てもなお、**同一 uid の別プロセス・侵害プロセス**が `kernel.cu` を保ったまま `kernel.ptx` だけを任意の有効な PTX へ差し替える攻撃は防げない: ファイルを書き換えられる主体は同じ uid で新たな「正当に見える」エントリ一式（`kernel.cu`＋改竄済み `kernel.ptx`＋権限検査を通る mode）を作れてしまうため、暗号学的ダイジェストや署名をディスク上へ同居させても認証（authenticity の証明）にはならない（秘密鍵を持たないハッシュ／ダイジェストは完全性の検査にしかならず、同一 uid の書き込み主体に対しては無力）。許容依存 8 区分（`.claude/rules/deps-policy.md`）に署名検証用クレートは含まれず、新規追加はユーザー承認が要るため本 PR のスコープでは導入しない。

よって現行実装は disk hit を「ソース一致が確認できた」というシグナルとしてのみ扱い、実際にロードする PTX は常にこのプロセス内で NVRTC が生成したものに限る（ヒット・ミスいずれの分岐でも `compile_ptx` を経由する）。`load_cache_entry` の呼び出しそのもの（C-3・#509 の fs 配線・権限検査）は将来、認証済み検証手段（例: NVRTC を実行できる信頼済みプロセスのみが持つ鍵での署名検証）を導入した際に実行入力として再有効化できるよう維持する。この判断により、ディスクキャッシュはプロセス再起動をまたいだ NVRTC コンパイル時間の短縮には現時点で寄与しない（`store_cache_entry` によるディスク永続化自体は維持しており、将来の認証手段導入時に即座に活用できる）。プロセス内 LRU（1 段目）は本判断の影響を受けず、同一プロセス内での再利用は従来どおり `cuModuleGetFunction` のみで完結する。

**スコープ境界（イシュー #1024 で更新）**: 固定ソースの一回コンパイル経路のうち、`CudaGemm::new`（f32 GEMM 本番経路。naive f32/f16・tiled f32/f16・tiled_bias_act_f32・wmma_tf32・wmma_tf32_opt・wmma_tf32_staged の 8 カーネル＋サイズ条件付き swizzle 変種）はイシュー #1024 で上記 3 段フォールバックへ結線した。共通ロジックは `crates/backend-cuda/src/module_cache.rs::load_function_cached` へ抽出し、`RenderedMmaKernel::compile` と `CudaGemm::new` の双方がこれを呼ぶ（`crates/backend-cuda/src/gemm.rs::GemmKernelSpec::descriptor` 参照）。`CudaWmmaGemm::new`・`CudaMmaGemm::new`・elementwise/transpose 群は引き続き未結線（本イシューのスコープ外。拡大は効果に対しリスク過大と判断し、横展開は別イシューで判断する）。

## キャッシュキー構成（総括。C-14・#539）

キャッシュキーは `CudaKernelDescriptor`（カーネル形状・タイル構成側）と `CudaKernelCacheKey`（環境パラメータ・ソース側）の 2 層で構成される（`crates/backend-cuda/src/nvrtc.rs`）。各項目の正はコード側ドキュメンテーションコメントであり、本節は横断参照用の一覧にとどめる（詳細ロジックの複写はしない）。

- **`CudaKernelDescriptor`**（`nvrtc.rs`）:
  - `kernel_name`（`&'static str` 限定。実行時文字列を受け付けない設計で A03 対策済み）
  - `cache_key_shape`（`compiled_dims` で非選択（動的扱い）とした次元を sentinel `0` に正規化した shape。定数化対象として選択した次元は実値のまま保つ。実行に使う実 shape である `shape()` とはフィールド・意味とも分離しており、`Hash`／`Eq` 実装はキャッシュキーとしての同一性判定に `cache_key_shape` のみを用いる〈`shape` は含めない〉）
  - `block_m`／`block_n`／`block_k`／`stages`（静的タイル構成。C-6〜C-9b の選定結果を表す）
  - `dtype`
  - `compiled_dims`（`Option<CompiledDims>`。C-7・#519 で導入した次元別定数化選択。`None` は次元特化なし〈従来コンストラクタ経由〉を表す）
- **`CudaKernelCacheKey`** の環境パラメータ（`nvrtc.rs`）:
  - `compute_capability`
  - `nvrtc_version`
  - `compile_flags`（`canonical_bytes` へのエンコード順序を決定的に固定。詳細はコード側コメント参照）
  - `source`（最終レンダー済みカーネルソース全文。C-5・#514。外部公開 getter は追加していない〈`RenderedMmaKernel` がソース文字列を外部へ返さない設計と同一方針〉）

キー全体のバイト列表現は `CudaKernelCacheKey::canonical_bytes`（`ENCODING_VERSION = 3`。下記「キャッシュ無効化条件」節参照）が組み立て、`fnv1a_64`（自作 FNV-1a 64bit・非暗号ハッシュ）でハッシュ化した値がディレクトリ命名（`kernel.<name>.<hash16>`）へ使われる（詳細は上記「ディレクトリ命名規則（C-2・#506）」節）。

## キャッシュ無効化条件（総括。C-14・#539）

以下のいずれかが発生すると、対応するキャッシュエントリ（ディスク側）は無効化される（`Ok(None)` としてミス扱い、または別キーとして扱われる）。

1. **カーネルソース変更**: `source` がキーに含まれるため（C-5・#514）、`kernels_mma.rs` の `*_BODY` 定数等ソース断片の編集は最終レンダー済みソース全文へ推移的に反映され、確実にキャッシュミスする
2. **`ENCODING_VERSION` 引き上げによる全エントリ無効化**: `v1 → v2`（C-5・#514。`compile_flags` の後段に `source` を長さプレフィクス付きで追記）→ **`v2 → v3`（C-7・#519・PR #674。`cache_key_shape` の動的次元 sentinel 正規化＋ `compiled_dims` をキーへ追記。誤ヒット是正）**。現行は `v3`。バージョン引き上げのたびに旧バージョンのエントリは全て無効（意図どおりの契約）
3. **環境変化**: `compute_capability`・`nvrtc_version`・`compile_flags` のいずれかが変われば別エントリ化される（同一マシンでも toolkit 更新等で自動的に別キーへ切り替わる）
4. **shape・特化ポリシーの違い**: `cache_key_shape`（動的次元を正規化した shape）・`compiled_dims`（次元別定数化選択）の組み合わせが異なれば別エントリ化される。同一の実 `shape` でも `compiled_dims` の選択が異なれば別キーになる
5. **エントリ破損**: 片ファイル（`kernel.cu`／`kernel.ptx`）欠落・0 バイト・ソース照合不一致は「ミス」として扱う（削除はしない。置き換えは書き込み側〈`store_cache_entry`〉に一元化。詳細は上記「エントリ内ファイル構成と書き込みプロトコル（C-3・#509）」節）
6. **プロセス内 LRU の evict**: ディスク側の無効化とは独立に、`KernelModuleCache`（プロセス内 LRU）は容量（`RUST_AI_CUDA_MODULE_CACHE_CAPACITY`。既定 `32`・許容範囲 `1..=1024`）超過時に最も古いエントリを evict する（ディスクエントリ自体は消えない）

補足: ディスクキャッシュのヒットは「保存済みソースが要求元ソースと一致すること」までしか保証しない。同一ディレクトリの `kernel.ptx` がそのソースから実際に生成された成果物であることの保証は別問題であり、この残余ギャップへの対応は上記「ディスク PTX を実行入力にしない判断（イシュー #511 PR #703 codex-review P0 再指摘対応）」節を参照（ディスクヒット時も実行入力は常にプロセス内で NVRTC が生成した PTX に限る）。

## shape 特化と静的タイル選定（C-6〜C-9b）

Phase C は「テンプレート展開」「次元別定数化選択」「SMEM 予算からの段数逆算」「候補列挙」「コストモデルによる静的選定」の 5 段階で構成される。各段の詳細設計・実測記録は個別ドキュメントを正とし、本節は横断参照用の要約にとどめる。

- **C-6（テンプレート展開・#516）**: カーネルソースのタイル寸法（`DIM_M`／`DIM_N`／`DIM_K` 等）をマクロ間接化し、コンパイル時定数として展開可能にする。展開後も REQ-8 の手動境界チェック（`.claude/rules/coding-rust.md` カーネル境界検査節）は維持する。詳細は `docs/perf/cuda-jit-template-expansion.md`
- **C-7（次元別定数化選択・#519）**: `CompiledDims` により、shape の一部次元のみを定数化（残りは動的のまま）する選択を可能にした。`CudaKernelDescriptor::new_with_compiled_dims` 経由で構築し、`cache_key_shape` は選択に応じて動的次元を sentinel `0` に正規化する（上記「キャッシュキー構成」節参照）。詳細は `crates/backend-cuda/src/nvrtc.rs` の `CompiledDims`・`CudaKernelDescriptor::new_with_compiled_dims` ドキュメンテーションコメント
- **C-8（SMEM 予算からの段数逆算・#521）・C-9a（候補列挙・#524）**: 共有メモリ予算から実行可能な `stages` 段数を逆算し、`enumerate_tile_candidates` がタイル候補群を列挙する
- **C-9b（L1/L2 帯域コストモデル・#527）**: `gemm_auto.rs::cost_model` モジュールが実行時ベンチマークなしに候補群から最良 1 件を決定的に選ぶ。定数の実機再測定手順は下記「コストモデル定数の実機再測定手順」節を参照。詳細は `docs/perf/cuda-gemm-cost-model-selection.md`

**スコープ境界**: shape 特化・コストモデル選定はいずれも既定の本番 GEMM 経路（`CudaGemmAuto::run_f16`・`gemm_mma.rs::CudaMmaGemm` の `MMA_STAGES=3` 固定構成）へは**未結線**である。`SpecializedMmaKernelHandle`（shape 特化コンパイル導線）はテスト・ベンチ向けであり、既定経路の実行結果・parity ベースラインには影響しない（`docs/perf/cuda-gemm-cost-model-selection.md` §3・§4 参照）。適用判断は実機実測・補正判定完了後の後続タスクに委ねる。

## コストモデル定数の実機再測定手順（C-14・#539）

手順本体は複写せず `docs/perf/cuda-gemm-cost-model-selection.md` §1 を正として参照する（二重管理回避）。本節はトリガ・参照連鎖・関連ドキュメントとの相互参照のみを記録する。

- **現状**: `SM121_MEASURED_BANDWIDTH: Option<MeasuredBandwidth> = None`（`crates/backend-cuda/src/gemm_auto.rs::cost_model`）。`None` である限り `select_tile_config` はコストモデルを一切評価せず、固定選定テーブル（`MMA_BM=64`／`MMA_BN=128`／`MMA_BK=32`／`MMA_STAGES=3`。#494 実測記録が根拠）へ fail-closed にフォールバックする。DeepGEMM（`sm90.hpp`）の H100 実測定数は流用しない契約
- **再測定のトリガ**:
  1. 初回実測（`docs/perf/sm121-device-attributes.md`・A-2・#482 が未実測のまま安全側クローズしている）
  2. CUDA ドライバ／toolkit の更新
  3. 対象ハードウェアの変更（`SM121_COMPUTE_CAPABILITY = (12, 1)` との突合により、他アーキテクチャへの誤適用は構造的に遮断済み。`select_tile_config_for_device` が `device.compute_capability()` と一致する場合のみ `SM121_MEASURED_BANDWIDTH` を使う）
  4. カーネルのメモリトラフィック構造を変える変更（タイル構成そのものの変更を含む）
- **更新の流れ（参照連鎖）**:
  1. `docs/perf/sm121-device-attributes.md` へ L1/L2 帯域実測値を記入する
  2. `crates/backend-cuda/src/gemm_auto.rs::cost_model::SM121_MEASURED_BANDWIDTH` を `Some(MeasuredBandwidth { .. })` へ更新する
  3. `docs/perf/cuda-gemm-cost-model-selection.md` §1 の手順に従い、3 形状（M=N=K = 4096／2048／1024）でモデル選定と実測最良構成を照合する（**3 形状中 2 形状以上一致**で採用）
  4. 不一致の場合はモデル定数の実機補正を **1 回だけ** 行い再判定する（補正ループ禁止。2 回目の補正は行わない）。補正後も不一致なら `SM121_MEASURED_BANDWIDTH` を `None` に固定し、固定選定テーブル採用を確定する
  5. `docs/perf/cuda-gemm-cost-model-selection.md` §1 の記録欄（形状別の実測最良構成・コストモデル選定・一致／補正実施の表）へ結果を記入する
- **JIT キャッシュ効果の再計測**: 本節はコストモデル定数（タイル選定の質）の再測定手順であり、JIT キャッシュそのものの効果（初期化レイテンシ短縮効果）の計測手順とは別スコープ。後者は `docs/perf/cuda-jit-cache-benchmark.md` §3（計測方法）・§5（C-4 結線後の本番経路再計測への引き継ぎ）を参照
- **実機接続情報**: ホスト名等の実値は本ドキュメントへ書かず、`docs/real-hardware-verification-env.md`（実値はローカル管理外ファイル `docs/real-hardware-verification-env.local.md` を参照する方式）に従う

## 関連

- `crates/backend-cuda/src/nvrtc.rs`: 実装本体（`resolve_cache_root`／`cache_root`／`cache_entry_path`／`cache_entry_path_in`／`fnv1a_64`／`ensure_cache_root`／`ensure_cache_root_in`／`runtime_workspace_root`／`store_cache_entry`／`store_cache_entry_in`／`store_cache_entry_at`／`load_cache_entry`／`load_cache_entry_in`／`validate_cache_entry_at`（本番経路）／`validate_cache_entry`（`#[cfg(test)]` 限定）／`rename_pinned`／`create_dir_all_verified`／`CudaKernelDescriptor`／`CompiledDims`／`CudaKernelCacheKey`／`canonical_bytes`）
- `crates/backend-cuda/src/module_cache.rs`: プロセス内 LRU 本体（`LruCache`／`KernelModuleCache`／`resolve_module_cache_capacity`）
- `crates/backend-cuda/src/kernels_mma.rs`: GEMM 経路への結線（`RenderedMmaKernel::cache_descriptor`／`cache_key`／`compile`）
- `crates/backend-cuda/src/gemm_auto.rs`: `cost_model` モジュール（`SM121_MEASURED_BANDWIDTH`／`SM121_COMPUTE_CAPABILITY`／`estimate_candidate_cost`）・`select_tile_config`／`select_tile_config_for_device`・`enumerate_tile_candidates`・`SpecializedMmaKernelHandle`
- `crates/backend-cuda/src/error.rs`: `CudaError::CacheDirUnavailable`／`CudaError::CacheIo`／`CudaError::InvalidModuleCacheCapacity`／`CudaError::ModuleCacheUnavailable`
- C-10（#529）: ヒット/ミス・並行競合・破損検出の網羅的回帰テスト拡充。実装済み（`crates/backend-cuda/src/jit_cache_regression_tests.rs`。`nvrtc` モジュール直下の兄弟モジュールとして `#[path]` 属性で配置。キャッシュ API が `pub(crate)` にも満たない module-private のため `crates/backend-cuda/tests/`〈integration test〉ではなく in-crate ユニットテストとした判断理由は同ファイル冒頭のドキュメンテーションコメントを参照）
- C-9b（#527）: 最良構成選定。C-12（#534）: 性能計測（本キャッシュの hit/miss カウンタ〈`KernelModuleCache::hit_count`／`miss_count`〉を観測点として使う想定。計測コード・一次実測データの担当は #534 側であり、#539〈本文書〉は実機検証手順の文書化を担当。役割重複なし。`docs/perf/cuda-jit-cache-benchmark.md` §5 参照）
- `docs/perf/cuda-jit-template-expansion.md`: C-6／C-7 の設計要約・スコープ境界
- `docs/perf/cuda-gemm-cost-model-selection.md`: C-9b コストモデル定数の実機再測定手順の本体（本文書「コストモデル定数の実機再測定手順」節が参照する正）
- `docs/perf/sm121-device-attributes.md`: L1/L2 帯域の実測記入先（A-2・#482）
- `docs/perf/cuda-jit-cache-benchmark.md`: C-12 計測方法・C-4 結線後の本番経路再計測の引き継ぎ
