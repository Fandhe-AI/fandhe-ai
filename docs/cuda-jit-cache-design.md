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

**リポジトリツリー containment 検証を持たない理由（PR #659 codex-review P1 指摘への対応。設計変更）**: 旧実装はコンパイル時 `CARGO_MANIFEST_DIR` から導出する「ビルド時ワークスペースルート」（旧 `nvrtc::compile_time_workspace_root`）配下への字句上の containment を拒否する検証を持っていたが、ビルド環境と実行環境が異なる場合（別 checkout・コンテナ・配布先での実行）にはこの検証が無条件で素通りする構造的欠陥があった（`env!("CARGO_MANIFEST_DIR")` はビルド時のディレクトリ配置を焼き込む値であり、実行時に別パスへ配置されたリポジトリと比較しても一致しないため。`AGENTS.md` の実機固有パスのハードコード回避方針にも反する）。「ビルド時定数との一致で拒否する」ブロックリスト方式は実行時に一般には機能しないため削除し、代わりに以下の許可リスト方式に置き換えた:

- `XDG_CACHE_HOME`・`HOME` の 2 分岐は、外部入力へ本ライブラリ専有のサブパス（`rust-ai-library/cuda` または `.cache/rust-ai-library/cuda`）を必ず付加する。XDG Base Directory・`$HOME/.cache` はいずれも OS 標準のユーザーキャッシュ規約であり、ソースリポジトリのツリーとは独立した場所を指す設計上の慣習である
- `RUST_AI_CUDA_CACHE_DIR` は DeepGEMM の `DG_JIT_CACHE_DIR` と同様に「呼び出し元が明示的に指定した信頼済みキャッシュ配置先」として扱う。絶対パス・非空のみ検証し、それ以上の自動判定は行わない（呼び出し元がリポジトリツリー内を誤って指定した場合の防止はこの層の責務外）

実際にディレクトリを作成・オープンする時点（C-3・#509）で `canonicalize` 済みパスによる containment 再検証を行うのが正しい実装点である（symlink 解決込みの実在ベース検証はパスの実在を要求するため、fs I/O を行わない C-2 の純関数では原理的に実行できない）。C-3 はこの時点で「信頼できる runtime workspace 境界」をどう受け取るかを含めて設計する。`nvrtc::path_lexically_within`／`nvrtc::lexically_normalize`（`..` 折り畳み込みの字句正規化プリミティブ）は C-3 の実装に転用できるよう crate 内に残してある（`resolve_cache_root` からは呼ばない）。

さらに、キャッシュエントリパス（`cache_entry_path`）の組み立て結果は必ず解決済みルート配下（`starts_with(root)`）に収まることを保証する多層防御を持つ（第 1 層: `CudaKernelDescriptor::new` の構築時検証、第 2 層: `CudaKernelCacheKey::cache_entry_dir_name` 内の縦深防御検査、第 3 層: `cache_entry_path_in` のユニットテスト）。この多層防御は「エントリパスがルート配下に収まる」ことのみを保証し、「ルート自体がリポジトリツリー外にある」ことは保証しない（上記参照。C-3 のスコープ）。

## ディレクトリ命名規則（C-2・#506）

キャッシュエントリはキャッシュルート直下に `kernel.<name>.<hash>` の形式で配置する。`<hash>` は [`CudaKernelCacheKey::canonical_bytes`] を自作の非暗号ハッシュ（FNV-1a 64bit。std のみで実装。依存クレート追加なし）でハッシュ化した値の 16 桁 16 進表記。非暗号ハッシュを選んだ理由・改竄検知に使わない旨は `crates/backend-cuda/src/nvrtc.rs` の `fnv1a_64` ドキュメンテーションコメントを参照。

## 関連

- `crates/backend-cuda/src/nvrtc.rs`: 実装本体（`resolve_cache_root`／`cache_root`／`cache_entry_path`／`cache_entry_path_in`／`fnv1a_64`）
- `crates/backend-cuda/src/error.rs`: `CudaError::CacheDirUnavailable`
- C-3（#509）: 一時ディレクトリコンパイル → アトミック rename（本文書の環境変数が実際に I/O へ結び付くタスク）。**残課題**: 本 PR（#659）時点で #509 の受け入れ基準には、`canonicalize` 済みパスによる containment 検証（symlink 対応を含む）が明記されていない。C-2 のリポジトリツリー containment 検証（ビルド時定数依存の旧実装。PR #659 codex-review P1 指摘対応で削除済み）は実行時には機能しなかったため、C-3 では「信頼できる runtime workspace 境界をどう受け取るか」を含めて検証条件を新規設計する必要がある。実装時に本文書・`nvrtc.rs` のドキュメンテーションコメントを踏まえて追加すること
- C-4（#511）: プロセス内 LRU カーネルキャッシュ
