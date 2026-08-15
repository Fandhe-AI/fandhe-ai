# CUDA JIT コンパイルキャッシュ ディレクトリ解決規則

- 対応イシュー: 親 #503（Phase C: CUDA JIT shape 特化・コンパイルキャッシュ・静的タイル選定）／C-1 #504（キー型定義）／C-2 #506（自作非暗号ハッシュ・ディレクトリ命名規則。本文書はその一部として PR #659 で追加）
- 位置づけ: 本文書は `crates/backend-cuda/src/nvrtc.rs` のキャッシュルート解決・エントリパス組み立てロジックの**利用者向け参照**である。実装本体のドキュメンテーションコメント（同ファイル）を正とし、本文書はそれを要約・横断参照可能な形にまとめたものにすぎない（二重管理を避けるため詳細ロジックはコードコメント側で保守する）。
- **現状（PR #659 時点）**: 本文書が説明する解決ロジック（`resolve_cache_root`／`cache_root`／`cache_entry_path`）はいずれも `pub(crate)`（crate 内限定）であり、`backend-cuda` クレートの外から呼び出す手段はまだない。ディスクへの実際の読み書き（`create_dir_all`・アトミック rename 等）は C-3（#509）で実装される予定で、C-2（#506・本 PR）時点ではパス解決ロジックのみが存在し、実際のキャッシュ I/O はまだ発生しない。したがって以下の環境変数は **C-3 実装後に初めて実効化される**（内部実装の先行スキャフォールディングとして本 PR で導入。crate 内公開範囲の判断は `crates/backend-cuda/src/lib.rs` 直下 `pub use` から意図的に除外している。理由は同ファイル該当関数のドキュメンテーションコメント参照）。

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
- **（追加。PR #659 codex-review P0 指摘への対応）** 解決結果（3 分岐いずれも）が、コンパイル時 `CARGO_MANIFEST_DIR` から導出するワークスペースルート（`nvrtc::compile_time_workspace_root`）の配下に **字句上（lexically）** 収まらないこと。攻撃例 `RUST_AI_CUDA_CACHE_DIR=/path/to/repository/cache` はこの層で拒否される（`XDG_CACHE_HOME`・`HOME` 由来の解決結果も同一検証を課す。個別指摘対象だった 2 分岐を含む）

**上記 3 条件で保証できる範囲は限定的である**（PR #659 codex-review P0 指摘）。3 番目の containment 検証は `Path::starts_with` による**字句比較のみ**であり、シンボリックリンク解決を行わない。したがってシンボリックリンクでリポジトリツリー内へ迂回する経路は本層では検出できない。これを実効的に強制するには、信頼済みのワークスペースルートに対して `canonicalize` 済みパスで containment 検証を行う必要があるが、これは実際にディレクトリを作成・オープンする時点（C-3・#509）で行うのが正しい実装点である。C-2（本文書・`resolve_cache_root`）時点では fs I/O を一切行わない純粋なパス組み立てのみであり、`canonicalize` はパスの実在を前提とするため、シンボリックリンク対応の再検証は C-2 の責務外として C-3 側に委譲する。

さらに、キャッシュエントリパス（`cache_entry_path`）の組み立て結果は必ず解決済みルート配下（`starts_with(root)`）に収まることを保証する多層防御を持つ（第 1 層: `CudaKernelDescriptor::new` の構築時検証、第 2 層: `CudaKernelCacheKey::cache_entry_dir_name` 内の縦深防御検査、第 3 層: `cache_entry_path_in` のユニットテスト）。この多層防御は「エントリパスがルート配下に収まる」ことのみを保証し、「ルート自体がリポジトリツリー外にある」ことは保証しない（上記参照。C-3 のスコープ）。

## ディレクトリ命名規則（C-2・#506）

キャッシュエントリはキャッシュルート直下に `kernel.<name>.<hash>` の形式で配置する。`<hash>` は [`CudaKernelCacheKey::canonical_bytes`] を自作の非暗号ハッシュ（FNV-1a 64bit。std のみで実装。依存クレート追加なし）でハッシュ化した値の 16 桁 16 進表記。非暗号ハッシュを選んだ理由・改竄検知に使わない旨は `crates/backend-cuda/src/nvrtc.rs` の `fnv1a_64` ドキュメンテーションコメントを参照。

## 関連

- `crates/backend-cuda/src/nvrtc.rs`: 実装本体（`resolve_cache_root`／`cache_root`／`cache_entry_path`／`cache_entry_path_in`／`fnv1a_64`）
- `crates/backend-cuda/src/error.rs`: `CudaError::CacheDirUnavailable`
- C-3（#509）: 一時ディレクトリコンパイル → アトミック rename（本文書の環境変数が実際に I/O へ結び付くタスク）。**残課題**: 本 PR（#659）時点で #509 の受け入れ基準にはシンボリックリンク対応の `canonicalize` containment 再検証が明記されていない。実装時に本文書・`nvrtc.rs` のドキュメンテーションコメントを踏まえて追加すること
- C-4（#511）: プロセス内 LRU カーネルキャッシュ
