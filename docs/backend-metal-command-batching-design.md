# Metal コマンドバッファ共有と同期境界の設計

対応 Issue: #1016（イシューツリー #1015 の設計担当子イシュー）。
ツリー位置: #1008 → #1007 → #1015「複数 dispatch を 1 コマンドバッファにまとめ
`waitUntilCompleted` をホスト実体化時のみにする」→ 本文書（設計）／#1017（実装）。

**スコープ宣言**: 本文書は設計・契約の記述のみを行う。`crates/**` のコードは
変更しない。実装は #1017 が本文書の設計に基づいて行う。

**CUDA 側との対応関係**: 「1.3 の設計と同一契約」として参照される CUDA 側の
非同期実行設計は #1011（親）／#1012 が担当するが、本文書の執筆時点で #1012 の
成果物（`docs/backend-cuda-async-execution-design.md` 相当）は main にも
リモートブランチにも存在しない（`git ls-remote`・`gh pr list` で確認済み）。
そのため本文書は #1012 に依存せず、Metal 側の実装事実（現行コード）から契約を
自立して定義する。§3.8 の CUDA 対応表は現行 `backend-cuda` の実装（`stream`
API・`synchronize()` 呼び出し箇所）を根拠にしたものであり、#1012 がマージされ
次第、両文書の用語・境界を突き合わせて相互参照を追記する（forward reference）。

## §0 位置づけ・スコープ

- 本文書は #1015 ツリーの第 1 段（設計）。実装は #1017 が担当する。
- 本文書が確定するのは「どこでコマンドバッファ／エンコーダを共有してよいか」
  「どこで `waitUntilCompleted` に相当する同期を必ず挟むか」という契約であり、
  具体的な Rust API のシグネチャは #3.2 に「案」として示すが最終決定は #1017 が
  実装しながら行う。
- 実機（Apple Silicon Metal）でのベンチマーク実測は本 PR の対象外。§4 の効果
  測定欄は Mac セッションでの記入待ちとして空欄のまま残す（推定値で埋めない）。
- 数値一致複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）・カーネルの
  手動境界検査（REQ-8）・許容誤差・依存バージョン固定は一切変更しない。

## §1 背景・実測根拠

現行 Metal バックエンドは「演算 1 回 = コマンドバッファ 1 個 + コンピュート
エンコーダ 1 個 + `commit()` + `waitUntilCompleted()` + status 検査」の完全同期
実行である（`crates/backend-metal/src/context.rs:150-179` `MetalContext::
dispatch_sync`）。この固定費がボトルネックであることは以下で実測済み。

- `docs/perf/device-resident-update-bench.md` §3〜§4: デバイス常駐パラメータ
  更新（`DeviceParamStore::step`）が update フェーズ単体で cpu 比 約 132〜152
  倍遅い。原因は「コマンドバッファ生成・commit・完了待ちのディスパッチ単位
  あたり固定費 × パラメータ数」と分析されている（改善実装自体は同文書では
  スコープ外として記録）。
- `docs/perf/metal-fixed-overhead-diagnosis.md` §1: Metal GEMM の実行時間が
  サイズに依らず約 5 ms に張り付く。MLP 1 step あたり GEMM 呼び出しは 6 回
  発生するため、この固定費が積み上がる。
- `scripts/bench/framework-compare/results/summary.md`（環境 5・Apple M4
  Max・macOS）(b')「MLP 学習（デバイス常駐パラメータ更新モード）」表: metal
  の train は fresh 中央値 19.699 ms に対し reuse 中央値 20.381 ms（fresh/reuse
  比 0.97 倍。**reuse が速くなっていない**）。同文書の「環境 5 の備考」は
  この原因を「`register_resident_leaves` が毎 step 全パラメータを D2H
  download する経路が reuse でも残存する（#954 申し送り）」と記録しており、
  本イシューが対象とする「演算ごとの `waitUntilCompleted`」はこの残存コスト
  の構成要素の一つである。本文書はこの数値を「改善前の実測基準点」として
  引用するに留め、コマンドバッファ共有単独でどこまで縮むかは実機計測（§4）
  に委ねる。

## §2 現状整理（コードから確定できる事実）

### 2.1 `dispatch_sync` の手順

`MetalContext::dispatch_sync`（`crates/backend-metal/src/context.rs:150-174`）
は以下を 1 呼び出しごとに行う。

1. `autoreleasepool` で囲む（Cocoa アプリの周囲 pool が存在しない Rust バイナリ
   で `commandBuffer()`／`computeCommandEncoder()` の autoreleased オブジェクト
   がプロセス寿命分蓄積するのを防ぐため。同ファイルの doc コメント参照）。
2. `queue.commandBuffer()` → `computeCommandEncoder()` を新規生成。
3. 呼び出し元クロージャで dispatch を積む（`encode(&encoder)`）。
4. `endEncoding()` → `commit()` → `waitUntilCompleted()`。
5. `cmd_buf.status() == MTLCommandBufferStatus::Error` を検査し、エラーなら
   `MetalError::CommandBufferExecutionFailed { message }` を返す（`commit()`
   自体は成功を返すだけで GPU 側の fault・OOM・discarded work を検知しない
   ため、この status 検査を省くと出力バッファの古い／不完全な内容を読む
   無言の数値誤りにつながる、と doc コメントに明記されている）。

呼び出し点は `crates/backend-metal/src/gemm.rs:731,849,979,1042,1172,1187`・
`elementwise.rs:140,163`・`sgd.rs:126`・`rmsnorm.rs:210`・`softmax.rs:142`
の計 11 箇所（`grep -n dispatch_sync` で確認。実装時に再確認すること）。

### 2.2 非常駐経路はホスト実体化を挟む

`ops.rs::gemm`（`crates/backend-metal/src/ops.rs:303-334`）・`elementwise.rs::
run_binary` 系は「ホストスライス入力 → `MetalBuffer::new_with_data`（H2D 相当）
→ dispatch → `download`/`read_to_vec`（D2H 相当）→ `Tensor` 返却」という構造の
ため、**演算 1 回ごとにホスト実体化が構造的に発生する**。したがってバッチ化の
一次的な受益者は device-resident 経路（`ops.rs::sgd_step_device` →
`DeviceParamStore::step`。パラメータ数分ループしてバックエンドを呼ぶ。
`crates/autodiff/src/optim/device_store.rs`）であり、forward 側の常駐化
（別イシュー #1022 と紐づく想定）が入るまで GEMM 経路単体の受益は限定的で
ある。§4 の効果測定はこの前提を踏まえて解釈する。

### 2.3 コンテキストはプロセスワイド singleton

`context_cache::cached_context()`（`crates/backend-metal/src/context_cache.
rs:111-124`）は `OnceLock<Mutex<Option<Arc<MetalContext>>>>` によるプロセス
内キャッシュで、`Arc<MetalContext>` を複数スレッドから共有する（`Send + Sync`
を `assert_send_sync` で静的検証。同ファイル 88-98 行）。コマンドバッファの
バッチ状態を `MetalContext` に持たせる場合、内部可変性（`Mutex`）が必須になる
（複数スレッドが同時に同一 `MetalContext` を使いうるため）。

### 2.4 バッファは `StorageModeShared` 固定・同期を前提にした実装

`MetalBuffer` は `StorageModeShared`（CPU/GPU 共有メモリ）で確保している
（`crates/backend-metal/src/buffer.rs:88,104`。PoC-v2-4 実測に基づく判断が
同ファイル冒頭コメントに記録されている）。`MetalMemory::download`
（`crates/backend-metal/src/memory.rs:299`）はホストデータ確定を返り値の契約
とし、実装は内部で `download_inner`（同 269 行）を呼ぶだけで **明示的な同期
呼び出しを含まない**。これは「呼び出しに先立つ `dispatch_sync` が既に
`waitUntilCompleted` 済みだから安全」という**現状の暗黙の前提の上に成立して
いる**。同様に `PoolZeroFill::zero_fill`（同 324-325 行。`fandhe_ai_tensor_core::
pool::PoolZeroFill` の実装）は CPU 側からバッファへ直接書き込む処理であり、
これもホスト書き込みハザードの発生点である。**バッチ化してエンコーダをまたぐ
非同期実行にした瞬間、この前提が崩れる**。これが本設計の核心の課題である。

### 2.5 `Tape` と `DeviceParamStore` の寿命

`Tape`（`crates/autodiff/src/tape.rs:219-`）はステップごとに新規生成・破棄
される運用（同ファイル「学習ループでの運用」節）。一方 `DeviceParamStore`
（`crates/autodiff/src/optim/device_store.rs`）は `Tape` と独立した寿命で
パラメータ・velocity バッファをデバイス上に常駐させ、`poisoned` 状態機械
（`sgd_step_device` の実行時エラー後に遷移し、以降 `step`／`sync_to_host`／
`register_resident_leaves`／`snapshot_resident_leaves` を `BackendError::
StorePoisoned` で拒否する。同ファイル冒頭コメント）を持つ。バッチ化後の
エラー伝播（§3.7）はこの既存の poisoned 契約を土台にする。

## §3 設計（design decisions）

### 3.1 用語（CUDA 側と共有する語彙）

| 用語 | 意味 |
| --- | --- |
| 投入（encode） | エンコーダへ dispatch を積む操作。非同期。呼び出しはブロックしない |
| flush | 開いているエンコーダを `endEncoding()` し、コマンドバッファを `commit()` する。**完了は待たない** |
| 同期点（synchronize） | flush 済みでなければ flush したうえで `waitUntilCompleted()` を呼び、`status` を検査する。完了後に復帰する |
| ホスト実体化 | ホストがデバイス側の計算結果を読む操作全般（`read_to_vec`・`download`・`to_tensor` 等） |

CUDA 側で対応する語彙は §3.8 の対応表を参照。

### 3.2 共有単位と API 案（#1017 が実装する形）

`MetalContext` に「開いているバッチ」（open batch）の状態を追加する。

- 内容: 現在のコマンドバッファ・現在のコンピュートエンコーダ・投入済み op の
  ラベル列（エラー時の診断用）・そのバッチが参照する in-flight `MetalBuffer`
  の保持列・**そのバッチへ dispatch を投入した呼び出し元が登録する共有失敗
  トークン列（影響トークン登録。§3.7 (2) のエラー伝播が同期呼び出し元の同一性
  に依らず対象へ届くようにするための登録先）**。トークンの型は
  `tensor-core`（`crates/tensor-core`）が定義するバックエンド非依存の共有
  失敗セル（案: `DispatchFailureCell`。`Arc<Mutex<Option<BackendError>>>`
  相当。`clone()` は `Arc` の複製のみで軽量）とし、`backend-metal` はこの
  トークンを「`set()` できる不透明な共有ハンドル」としてのみ扱う。
  **`backend-metal` が `autodiff::DeviceParamStore` 型そのもの（`Weak`
  参照を含む）を保持・import することはない**（AGENTS.md「クレート境界の
  維持」。`autodiff` は `tensor-core` に依存するが `tensor-core`／
  `backend-metal` は `autodiff` を知らないという既存の依存方向を維持し、
  `backend-metal → autodiff` の逆依存／循環を作らない。旧案〈本節初版〉は
  `Batch` に `Weak<DeviceParamStore>` を持たせる設計だったが、レビュー指摘
  により本節の共有トークン方式へ改める）。
- 影響トークン登録の呼び出し規約: `sgd_step_device`（`autodiff` 側）は
  自身が保持する `DispatchFailureCell` を `clone()`（`Arc` 複製）して batch
  へ `encode` する際に渡し、バッチの登録列へ追加する（同一バッチへ複数
  トークンが登録されうる。バッチはどのコンテキスト呼び出し経路〈
  `synchronize()` の明示呼び出し・別スレッドの `download`／
  `sgd_step_device` が誘発する暗黙 flush 等〉で完了処理されても、登録済みの
  全トークンへ `set()` できる状態を保つ）。`DeviceParamStore` 自体は値として
  返され `Arc` で包まれていない（`facade::compat::Sequential::
  init_device_param_store` の戻り値。§2.5）ため `Weak<DeviceParamStore>` は
  構築できないが、共有トークンは `DeviceParamStore` 本体とは独立に発行する
  小さな `Arc` セルであるためこの制約に影響されない（`DeviceParamStore` が
  先に drop されてもトークンだけが `Batch` 側に残るだけで誰も読まなくなり
  実害がない。旧案「ストアが先に drop された場合は無視する」と同じ安全側の
  帰結を `Weak` なしで得る）。ポイズン遷移そのもの（`poisoned` フィールドへ
  の書き込み）は `backend-metal` は一切行わない。`backend-metal` が行うのは
  登録済みトークンへの `set()`（§3.7 (2)）のみであり、`poisoned` へ遷移
  させる判断とその実行は `DeviceParamStore` 自身（`autodiff` クレート内）が
  次回の `check_not_poisoned()` 呼び出し時に自分のトークンを検査して行う
  （下記 §3.7 (2) 改訂）。
- 保持方法: `Mutex<Option<Batch>>`（§2.3 の `Arc<MetalContext>` 複数スレッド
  共有契約と整合させるため）。
- `encode(label, |encoder| ...) -> Result<(), MetalError>`: open batch が
  無ければ新規に `commandBuffer()`／`computeCommandEncoder()` を生成してから
  クロージャを呼ぶ。ある場合はそのエンコーダへ積み増す。**呼び出しは
  待たない**。
- `flush(&self) -> Result<(), MetalError>`: 開いている場合のみ `endEncoding()`
  → `commit()` を行い、open batch を「未完了だが投入済み」の状態にする。
  Metal のコマンドバッファは一度 `commit()` すると再利用できない仕様のため、
  flush 後に新規 `encode` が呼ばれた場合は新しいコマンドバッファを生成する。
- `synchronize(&self) -> Result<(), MetalError>`: `flush()` → 直近でコミット
  した（複数あれば全ての未完了）コマンドバッファに対し `waitUntilCompleted()`
  → `status() == Error` ならバッチ内の op ラベル列を含めて
  `MetalError::CommandBufferExecutionFailed { message }` を返す（どの演算で
  失敗したかを診断メッセージから失わないようにする。既存の `dispatch_sync`
  の status 検査契約〈§2.1〉を拡張し、単一 dispatch から「バッチ内の複数
  dispatch のどれか」に一般化する）。
- `dispatch_sync` は `encode` + `synchronize` を組み合わせた薄いラッパーとして
  温存する。既存呼び出し元（PoC-v2-4 実測に基づく「ホスト転送を伴わない完了
  待ち」の計測境界）との互換性、および既存の数値一致回帰テストの不変を保つ
  ため、シグネチャ・戻り値の意味は変えない。
- エンコーダの dispatch type は既定（serial）を使う。同一エンコーダ内の
  dispatch は投入順に実行され、先行 dispatch の書き込みは後続 dispatch から
  可視という Metal の serial エンコーダの契約をそのまま利用する（`Concurrent`
  dispatch type とメモリバリアの明示制御は §5 で不採用と判断）。
- `MetalContext` は単一 `MTLCommandQueue` をプロセスワイド singleton として
  持つ（§2.3）ことを不変条件とする。同一 queue に commit したコマンドバッファ
  同士の実行順序は commit 順（in-order queue）で保証される。これは CUDA の
  単一ストリームによる in-order 実行契約と同型であり（§3.8）、複数
  `MTLCommandQueue`・`MTLEvent`／`MTLSharedEvent` によるクロスキュー同期は
  導入しない（`objc2-metal =0.3.2` に API 自体は存在するが、単一 queue で
  順序が閉じているため不要。§5(b)）。

### 3.3 エンコーダ・コマンドバッファの寿命

open batch は以下のいずれかで閉じる。

1. 明示的な同期点への到達（`synchronize()` 呼び出し、または `dispatch_sync`
   経由）。
2. 明示的な `flush()` 呼び出し（完了を待たずにコマンドキューへ投入したい
   場合。例えば複数の独立した部分グラフを investing してから 1 回だけ待つ、
   といった用途を #1017 が必要とすれば使う）。
3. `MetalContext` の drop 時: drop 実装で必ず `synchronize()` 相当を呼び、
   未完了の GPU work をプロセス終了・コンテキスト破棄後に残さない
   （`Arc<MetalContext>` はプロセスワイド singleton のため通常は drop されない
   が、テストでの明示的な破棄・将来のマルチデバイス対応に備えて契約として
   明記する）。
4. バッチ内 dispatch 数の安全弁上限（無限にエンコーダへ積み続けてコマンド
   バッファが肥大化するのを防ぐ）。具体的な上限値は #1017 の実装時に決定する
   （本文書では「上限を設ける」ことのみを契約として確定する）。

`autoreleasepool` はクロージャの字句スコープでのみ開始・終了できる（objc2
の API 契約）ため、「バッチ生成〜synchronize まで 1 pool を開いたまま保持
する」設計は採用できない。現行の「1 dispatch ごとに 1 pool」は維持せず、
代わりに `encode`／`flush`／`synchronize` の**各呼び出しをそれぞれ個別の
短命 `autoreleasepool` で囲む**設計に改める（呼び出し 1 回 = pool 1 回。
バッチをまたいで pool を持ち越さない）。

複数の `encode` 呼び出し・その後の `synchronize` にまたがって生存させる
必要があるオブジェクト（open batch が保持するコマンドバッファ・エンコーダ・
in-flight バッファ・§3.2 の共有失敗トークン `DispatchFailureCell`）は、生成した
`autoreleasepool` クロージャの内側で `Retained`（objc2 の所有権保持型）へ
変換したうえで `Batch` 構造体のフィールドとして pool の外（`encode` 呼び出し
自体の戻り値経由）へ持ち出し、次回以降の `encode`／`flush`／`synchronize`
呼び出し（＝別の pool）からはその `Retained` 経由でのみ参照する。これにより
「オブジェクトの寿命は個々の呼び出しの pool スコープを超えて `Batch` が
所有する」「pool 自体は各呼び出しの字句スコープに閉じる」の両方を矛盾なく
満たす（`autoreleasepool` は既定でスコープ終了時にプール内の autoreleased
オブジェクトを解放するため、persist させたいオブジェクトを暗黙のスコープ
依存にしない）。

### 3.4 バッファ寿命・再利用契約

Metal の `commandBuffer()` はリソースを retain する既定の動作のため、GPU
側からの参照自体は安全である。問題は **ホスト側からのアクセス**である。

契約: **in-flight のバッチが参照しうる `MetalBuffer` へのホスト読み書き
（`read_to_vec`・`zero_fill`・`contents()` の直接書き込み・`new_with_data`
によるプール再利用）は、必ず `synchronize()` を経由してから行う。**

個別バッファ単位の依存追跡（このバッファはこのコマンドバッファでしか参照
されていない、といった細粒度の判定）は行わない。ホストアクセスが発生する
たびにバッチ全体を同期する、過剰同期側（安全側）の設計に倒す。これは
§2.4 で述べた「暗黙の前提」を明示化したうえで、バッチ化後も安全性を維持する
ための最小の変更である。

演算内の一時バッファ（`a_buf`／`b_buf`／`c_buf` 等、GEMM・elementwise 各
関数がローカルに確保するバッファ）は、現行の「関数スコープで生存し
`dispatch_sync` の同期完了まで生きていることが保証される」契約から、
「バッチの in-flight 保持列に登録され、そのバッチが `synchronize()` される
まで解放されない」契約へ置き換える。

`PooledMemory`（`docs/memory-pool-design.md`・別イシュー #1018 が担当する
メモリプール）との関係: `zero_fill` は上記の通り同期点として扱う。メモリ
プールの導入後もこの契約（ゼロ埋めはホスト書き込みであり同期を要する）は
変更しない。

### 3.5 同期点一覧（受け入れ条件 (a)）

| API / 契機 | 種別 | 理由 | CUDA 対応（§3.8） |
| --- | --- | --- | --- |
| `MetalMemory::download`（`MemoryOps::download` の全バックエンド共通契約「復帰時点でホストデータ確定」。`crates/tensor-core/src/buffer.rs` 「download の同期契約」節） | 暗黙 | ホスト実体化 | `CudaMemory::download_inner` の `clone_dtoh` 後 `stream.synchronize()`（`crates/backend-cuda/src/memory.rs:255-280`） |
| `MetalBuffer::read_to_vec`／`HalfBuffer::read_to_vec` | 明示（呼び出し元が同期済みであることを保証する経路に限定するか、`synchronize` を内包するかは #1017 で選択） | ホスト実体化 | 同上 |
| 非常駐 op（`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`gemm_bias_act`／`run_fused_*` 等）の `Tensor` 返却 | 暗黙（構造的） | §2.2 の通り現状は演算ごとにホスト実体化する構造のため同期点になる。将来の forward 常駐化（別イシュー）で解消されうる | カーネル起動直後の readback パターンと同型 |
| `DeviceParamStore::sync_to_host`／`register_resident_leaves`（download を伴う）／`predict_resident` の `to_tensor` | 明示 | ホスト実体化 | 同上 |
| `PoolZeroFill::zero_fill` | 明示 | ホスト書き込み（§2.4・§3.4） | 該当なし（CUDA は zero-fill を GPU 側カーネルまたは `cuMemsetD32Async` で行う想定。本文書では確定させない） |
| `MetalContext` drop | 明示 | 未完了 GPU work を残さない（§3.3 (3)） | コンテキスト破棄時の `stream.synchronize()` 相当 |
| テスト・ベンチの計測境界（`dispatch_sync`／`synchronize`） | 明示 | spec の計測プロトコル（正本 `docs/spec/04-requirements.md` の性能計測境界の定義）と一致させる | 同型 |

**同期点にしないもの**: `sgd_step_device` の各呼び出し（encode のみで
synchronize しない。`DeviceParamStore::step` がパラメータ数分ループする際に
毎回待つと本イシューの目的〈固定費の削減〉が達成できないため）・`Tape` の
drop（`Tape` は計算グラフのノード管理のみを担い GPU リソースを直接持たない
ため同期対象外）・`flush()` 単体（完了を待たない操作として定義しているため）。

### 3.6 unified memory 下の可視性保証（受け入れ条件 (b)）

`StorageModeShared` は CPU と GPU が同一物理メモリを参照する（Apple Silicon
の unified memory アーキテクチャ。`crates/backend-metal/src/buffer.rs` 冒頭
コメント参照）。可視性の方向ごとに保証が異なる。

- **ホスト→GPU**: `commit()` より前に完了しているホスト側の書き込みは、
  同じバッファを参照する dispatch から可視である。`didModifyRange` の呼び出し
  は `StorageModeManaged`（Intel Mac 等）専用の契約であり、Apple Silicon の
  `StorageModeShared` では不要（`MTLResourceOptions::StorageModeManaged` は
  本リポジトリでは使用しない。§2.4 の採用理由と同じ）。
- **GPU→ホスト**: コマンドバッファの `status` が `Completed` に到達した後に
  のみ可視性が保証される。`commit()` 後・完了前にホストが同一バッファを
  読む、または上書きすることは未定義動作（データ競合）であり、§3.4 の契約
  （ホストアクセスは必ず `synchronize()` を経由する）で fail-closed に禁止
  する。
- `StorageModePrivate` + blit によるステージングは、`crates/backend-metal/
  src/buffer.rs` の既存の不採用判断（PoC-v2-4 実測に基づく `StorageModeShared`
  採用）を維持し、本設計でも変更しない。
- 複数スレッドからの利用: `Arc<MetalContext>` の open batch 状態は
  `Mutex`（§3.2）で直列化する。`synchronize()` はグローバル（呼び出し時点で
  その `MetalContext` が保持する全ての未完了バッチを対象とする）に倒し、
  スレッドごとの部分同期は行わない（過剰同期側の設計。§3.4 と同じ判断軸）。
  異なる `Tape`（学習ステップ）をまたぐ実行順序も、単一 `MTLCommandQueue` の
  commit 順によって保証される。

### 3.7 エラー伝播

遅延検出（`synchronize()` 時に初めて GPU 側のエラーが判明する）を前提とし、
以下の契約とする。

1. エラーが判明した時点で、そのバッチに含まれていた全ての dispatch の出力を
   **無効とみなす**（バッチ内のどの dispatch が実際に失敗要因かを個別に
   切り分けない。診断のためラベル列はエラーメッセージに含める。§3.2）。
2. `DeviceParamStore` の poison 遷移は、**`backend-metal` 側が直接行わず、
   `DeviceParamStore` 自身（`autodiff` クレート内）が §3.2 の共有失敗
   トークン（`DispatchFailureCell`）を検査して自己遷移する**ことで行う。
   `backend-metal` 側の役割は、エラーを検出した `synchronize()` 呼び出し
   （呼び出し元が `DeviceParamStore::step`・別スレッドの `download`・
   `MetalContext` drop のいずれであっても同一処理）が、そのバッチに登録
   された全てのトークンへ `set(err)` することのみに限定する（`tensor-core`
   が定義する型への操作であり、`autodiff::DeviceParamStore` の内部状態には
   一切触れない。§3.2 の「クレート境界の維持」）。`DeviceParamStore` 側は
   `check_not_poisoned()`（`step`／`sync_to_host`／`register_resident_leaves`
   ／`snapshot_resident_leaves` の共通入口）の先頭で自分のトークンを検査し、
   `set()` 済みであれば `poisoned` フィールドへ遷移させたうえで
   `BackendError::StorePoisoned` を返す。これにより、ストア自身が後から
   自発的に `synchronize()` を呼んで初めて poison されるのではなく、
   同一 `MetalContext` を共有する別スレッド・別 API のホスト実体化が先に
   synchronize してバッチのエラーを消費した場合でも、元の `DeviceParamStore`
   は「他者の synchronize」に依存せず自分のトークン経由で poison される
   （fail-closed の維持。対称なホスト実体化が一切発生しない経路 — バッチが
   登録したトークン全てに対応するストアが以降 `step`／`download`／
   `synchronize` を呼ばないまま放置される場合 — についても、次回いずれかの
   API 呼び出し時に §3.5 の暗黙・明示同期点を経由して同じバッチの status を
   再検査し、未消費のままだったエラーを遅延なく該当トークンへ伝播する。
   すなわちバッチの status 検査結果はバッチ側で一度確定させたら破棄せず、
   `Batch` が保持する `Result` としてキャッシュし、登録トークンが増える限り
   再送可能にする）。
   `poisoned` フィールドは現状 `bool`（内部可変性なし）だが、
   `sync_to_host`／`snapshot_resident_leaves` は `&self` で
   `check_not_poisoned()` を呼ぶ（§2.5 のとおり `step`／
   `register_resident_leaves` は `&mut self`）ため、自己遷移を実装するには
   `poisoned` を `Cell<bool>`（または同等の内部可変性）へ、`DispatchFailureCell`
   自体を保持するフィールドを追加する変更が `autodiff` 側に必要になる。
   いずれも `autodiff` クレート内で完結する変更であり、`backend-metal` 側の
   変更を要しない。
   これは既存の `StorePoisoned` 状態機械（§2.5・`.claude/rules/security.md`
   A08「部分的に更新されたデバイス側パラメータをそのまま学習継続・推論に
   使わせない」）の拡張であり、新しい状態を追加するものではない。
3. `BackendOps`（`crates/tensor-core/src/backend_ops.rs:122`）へ非破壊的な
   拡張として `synchronize(&self) -> Result<(), BackendError>`（デフォルト
   実装 `Ok(())`。CPU は no-op、CUDA は `stream.synchronize()`、Metal は
   本文書の `synchronize()`）を追加する案を、**CUDA 側 #1012 と要整合の
   共有契約案**として提示する。採否・最終シグネチャは #1012／#1017 の内容が
   揃った時点で main が判断する（本文書では「案」に留め、確定させない）。

### 3.8 CUDA 契約との対応表

| Metal | CUDA | 差分 |
| --- | --- | --- |
| `MTLCommandQueue`（単一 queue） | CUDA stream | 同型（in-order 実行契約） |
| `encode`（dispatch をエンコーダへ積む） | kernel `launch` | 同型（非同期投入） |
| `flush`（`endEncoding` + `commit`。待たない） | （CUDA には明示的な flush 概念はなく、stream への投入自体が非同期） | Metal はコマンドバッファ単位の commit が必要な点が異なる |
| `synchronize`（flush + `waitUntilCompleted` + status 検査） | `stream.synchronize()` | 同型の同期点 |
| `synchronize` 後の `read_to_vec`（コピー不要） | `clone_dtoh` + `stream.synchronize()`（`crates/backend-cuda/src/memory.rs:255-280`） | Metal は `StorageModeShared`（UMA）のためデバイス→ホストの明示コピーが不要。CUDA は `cuMemcpyDtoHAsync` による明示コピーが必要 |
| `status() == Error` | sticky context error（CUDA のエラーはコンテキスト全体に波及し以降の呼び出しも失敗し続けうる） | Metal のコマンドバッファエラーはそのコマンドバッファに閉じており、次のコマンドバッファは独立して実行できる点が異なる |
| （§3.7 (3) の `BackendOps::synchronize` 案） | 該当 API 未定（#1012 待ち） | forward reference |

## §4 期待効果と実機計測計画（Mac セッション記入欄）

**実測日・環境**: 2026-08-31・Apple M4 Max（macOS）。HEAD（#1017 実装
反映済み）でビルド・計測。計測は release ビルド（`cargo build --release`）
後に実施した。

### 4.1 計測境界の注意（比較の前提）

`scripts/bench/framework-compare/bench-fandhe`（`fandhe-ai =0.4.0` の
crates.io ピン。`.claude/rules/deps-policy.md` の第 9 区分）は本ワーク
ツリーの改善前コードしか計測できないため、以下の 2 種類の計測を分離した
（タスク指示・本文書冒頭の指示どおり実測前の推定値は記載しない）。

1. **#1017 の効果を隔離するマイクロベンチ**（HEAD 上・同一プロセス内で
   バッチ化前後の 2 経路を横並び計測。他の性能改善コミットの影響を
   受けない）。
2. **HEAD 絶対値としての MLP 学習 1 step 計測**（`bench-fandhe --task
   train --mode reuse` と同一モデル形状・シード・step 数・warmup 数の
   プロトコルを HEAD の `facade` crate（path 依存）で再現。crates.io
   0.4.0 基準点との比較は「#1017 単独の delta」ではなく「#1017 を含む
   HEAD までの全改善コミットの累積 delta」である点を明記する）。

### 4.2 コマンドバッファ生成数・`waitUntilCompleted` 呼び出し回数

**実行時カウンタは追加していない**（本 doc タスクの範囲外の本番コード
変更〈`crates/backend-metal/src/context.rs` への計測用計装〉を避けるため。
代わりに #1017 で変更された唯一の経路（`BackendOps::
sgd_step_device_tracked`。§7）をソースコードから静的に突き合わせた）。

- `gemm.rs`／`elementwise.rs`／`rmsnorm.rs`／`softmax.rs` の
  `dispatch_sync` 呼び出し（§2.1 の 11 箇所）は #1017 で**無変更**
  （§7 実装記録に明記）。MLP 1 step あたり GEMM 呼び出しは §1 の通り
  6 回発生し、各呼び出しは変更前・変更後とも独立した 1 コマンドバッファ
  ＋ 1 `waitUntilCompleted()` のまま。
- `sgd_step_device`（`token: None`。§3.7 (3) の非バッチ契約）は
  変更前・変更後とも「`encode` 直後に `ctx.synchronize()`」を維持
  （`ops.rs:375-390`・`sgd.rs` doc コメント）。
- 変更があるのは `sgd_step_device_tracked`（`token: Some`）のみ:
  変更前はデフォルト委譲で `sgd_step_device` へ流れ 1 呼び出し = 1 コマンド
  バッファ ＋ 1 `waitUntilCompleted()`（即時ブロッキング）。変更後は
  `encode` のみを行い、その場では待たない（`context.rs::encode`）。
- `DeviceParamStore::step` は #1023（#1017 以前）で既にパラメータ数に
  依らず 1 回の起動へ集約済み（`device_store.rs:43-44`）であるため、
  reuse ループの 1 step あたりの `sgd_step_device_tracked` 呼び出しは
  変更前後とも **1 回**。`register_resident_leaves`（forward 側の
  D2H download。§2.2）の直後に `ctx.synchronize()` が走り、まだ
  flush されていない SGD バッチがあればそこで `flush()` ＋
  `waitUntilCompleted()` される（`memory.rs::download_inner` が
  §3.5 の同期点）。
- 結論: **MLP 学習 1 step あたりのコマンドバッファ生成数・
  `waitUntilCompleted()` 呼び出し回数は変更前後で同数**（GEMM 6 回 +
  SGD 1 回 = 7 回、いずれもカウント不変）。#1017 が変えるのは「SGD の
  完了待ちが *いつ* 発生するか」であり、「起動直後の専用ブロッキング
  待機」から「次 step 冒頭の `register_resident_leaves` の download が
  必要とする同期点への遅延・合流」へ移した点にある（host 側は SGD 完了を
  待つ間に次 step のホスト側前処理を進められる。ただし本 doc 執筆時点で
  reuse ループが `sgd_step_device_tracked` を 1 step あたり 1 回しか
  呼ばないため、複数 `encode` が同一コマンドバッファへ積み増される
  効果〈§3.2 の本来の狙い〉はこの経路では実現していない）。

### 4.3 マイクロベンチ: #1017 の効果を隔離した計測（HEAD・新規追加）

`crates/backend-metal/tests/command_batching_bench.rs`
（`command_batching_micro_bench_untracked_vs_tracked`）で、同一 shape
（要素数 1024）・同一パラメータ／勾配バッファに対し、非バッチ経路
（`sgd_step_device`。呼び出しごとに同期）とバッチ経路
（`sgd_step_device_tracked`。100 回連続 `encode` の後に 1 回だけ
`download` で同期）を 100 回連続更新× 5 trial の中央値で比較した。

| 経路 | 100 回連続更新の中央値（Q1〜Q3） |
| --- | --- |
| 非バッチ（`sgd_step_device`。#1017 以前の `sgd_step_device_tracked` デフォルト委譲と同一の同期契約） | 10.857 ms（10.743〜13.596 ms） |
| バッチ（`sgd_step_device_tracked`。#1017） | 0.489 ms（0.476〜0.918 ms） |

speedup 約 22.2 倍（`tracked_faster=true`）。これは #1017 が導入した
「`encode` の連続投入 + 単一の遅延同期」機構そのものの効果を、他の
性能改善コミットと混同せず HEAD 上で単独に隔離した計測である。§4.2 で
述べた通り、現行の reuse ループ 1 step あたりの `sgd_step_device_tracked`
呼び出しは 1 回のみのため、この 22 倍という数値は「複数の SGD 起動を
1 コマンドバッファへ積み増す場合の上限効果」を示す指標であり、現行の
MLP 1 step の直接的な短縮幅とは別物である（§4.2 の結論と整合）。

### 4.4 HEAD 絶対値: MLP 学習 1 step（MNIST 規模 2 層 MLP・train・reuse）

`crates/facade/tests/mnist_scale_train_reuse_bench.rs`
（`mnist_scale_train_fresh_vs_reuse_metal`）で、`bench-fandhe` と同一の
モデル形状（`BATCH=64`／`D_IN=784`／`D_HIDDEN=256`／`D_OUT=10`）・
乱数シード・`TRAIN_STEPS=100`／`TRAIN_WARMUP=20`（先頭 20 step を捨て
残り 80 step の中央値）のプロトコルを HEAD の `facade` crate（path
依存）で再現した。

| | fresh（ホスト経由 SGD） | reuse（デバイス常駐 SGD） | reuse/fresh |
| --- | --- | --- | --- |
| HEAD（本実測。M4 Max） | 17.493 ms（17.094〜17.743 ms） | 8.756 ms（8.496〜9.117 ms） | 1.998 倍高速 |
| `fandhe-ai =0.4.0`（`results/summary.md` 環境 5 (b') 表。§1 引用） | 19.699 ms | 20.381 ms | 0.966 倍（reuse が遅い） |

**境界差の明記（§4.1）**: 上段（HEAD）と下段（0.4.0）は計測対象コード
のバージョンが異なる。HEAD の reuse 8.756 ms は #1017 単独の効果ではなく、
0.4.0 以降にマージされた性能改善コミット群（#1013・#1023・#1028・
#1043〜#1047・#1044・#1078〜#1082 等。冒頭コメント参照）を累積した結果
であり、§4.3 のマイクロベンチが #1017 単独の delta を担う。

### 4.5 #1015 受け入れ条件の充足判定

受け入れ条件「改善が見られること、かつ既存の数値一致 parity テストが
green のまま」を以下のとおり判定する。

- **改善**: (a) HEAD 絶対値として reuse が 0.4.0 の 20.381 ms から
  8.756 ms へ改善し、かつ 0.4.0 で発生していた「reuse が fresh より
  遅い」逆転（0.966 倍）が解消され reuse が fresh の 1.998 倍高速に
  なった（§4.4。ただし累積 delta であり #1017 単独の寄与は §4.3 の
  マイクロベンチ〈約 22 倍〉に限定して評価する）。(b) #1017 が導入した
  バッチ化機構そのものは、その機構を直接行使する経路（連続
  `sgd_step_device_tracked` 呼び出し）において約 22 倍の高速化を示した
  （§4.3）。現行の reuse ループはこの機構を 1 step あたり 1 回しか
  行使しないため、MLP 1 step の短縮幅への直接的寄与は §4.2 の結論
  （コマンドバッファ・wait 回数は不変、完了待ちの時間帯が後ろへ
  ずれるのみ）にとどまる点は正直に記録する。
- **parity green**: `make test-ignored-metal`（`cargo test -p
  fandhe-ai-backend-metal --release -- --ignored --nocapture` 相当）
  を実行し、既存の実機依存テスト（`command_batching.rs` の #1017
  受け入れ条件 1〜4 を含む）・parity テスト（`sgd_device_parity.rs`・
  `cpu_metal_parity.rs`・`gemm_*_parity.rs`・`rmsnorm_parity.rs`・
  `softmax_parity.rs` 等）を含む全 31 テストバイナリが `test result: ok`
  （failed 0）で完走したことを確認済み（2026-08-31 実測）。

## §5 代替案と採否

| 案 | 内容 | 採否 |
| --- | --- | --- |
| (a) | 演算ごとのコマンドバッファ + `waitUntilScheduled` のみで完了を待たない | 不採用。ホストからの可視性保証（§3.6）が得られない |
| (b) | 複数コマンドバッファを `MTLEvent`／`MTLSharedEvent` で連鎖させる | 不採用。単一 `MTLCommandQueue` の commit 順で順序保証が既に成立しており、クロスキュー同期の機構は不要（§3.2） |
| (c) | `MTLDispatchTypeConcurrent` + `memoryBarrierWithScope` による並列化 | 保留（スコープ外）。並列化の余地は将来課題として別イシュー提案の対象とする |
| (d) | `addCompletedHandler` によるコールバックベースの完了通知 | 保留。block FFI の呼び出しに伴う `unsafe` の増加に見合う利点が現時点では明確でない |
| (e) | `StorageModePrivate` + blit ステージング | 不採用（既存判断を維持。`crates/backend-metal/src/buffer.rs` 冒頭コメント） |

## §6 #1017 への引き渡し事項・#1012 との整合点・スコープ外

### 6.1 #1017 のテスト方針

実機依存テストは `#[ignore]` + `cfg(target_os = "macos")` の二重分離
（既存の `tests/sgd_device_parity.rs` 冒頭と同方針）を踏襲する。最低限含める
べきケース:

1. 同一バッチ内で A → B の依存がある dispatch（B が A の出力バッファを読む）
   の実行順序回帰（serial エンコーダの投入順実行契約が壊れていないことの
   確認）。
2. 複数パラメータに対する `sgd_step_device` の連続投入後の `sync_to_host`
   parity（既存の 100 step 累積判定パターンを流用し、許容誤差は変更しない）。
3. `synchronize` 時のエラーが（`backend-metal` が登録済みトークンへ
   `set()` した結果として）`DeviceParamStore` を次回の `check_not_poisoned()`
   呼び出し時に `poisoned` へ正しく自己遷移させること。加えて §3.7 (2) の
   共有失敗トークン登録契約を検証するケースとして、同一バッチへ `encode`
   した `DeviceParamStore` とは別の API（例: 別スレッドからの `download`）
   が先に synchronize してエラーを消費した場合でも、元の `DeviceParamStore`
   がトークン経由で poisoned へ遷移すること（呼び出し元の同一性に依らない
   ことの回帰確認）。
4. `dispatch_sync` 互換経路（既存呼び出し元を変更しない場合の後方互換）の
   既存 parity テストが不変であること。

Linux でも実行できる部分（バッチ状態機械のラベル列管理・投入数上限判定など
Metal API を直接呼ばない純粋なロジック）は `cfg(target_os = "macos")` の外側
のモジュールへ切り出すことを推奨する。

### 6.2 #1012 との整合点

#1012 マージ後に確認・追記すべき事項:

- §3.1 の用語（投入／flush／同期点／ホスト実体化）が CUDA 側の語彙と対応
  していること。
- §3.7 (3) の `BackendOps::synchronize` 案の採否・シグネチャ確定。
- §3.8 の対応表の記載内容が #1012 の記述と矛盾しないこと。

### 6.3 スコープ外（`out-of-scope-tracking.md` に従い記録のみ。起票はユーザー承認後）

- メモリプール本体の設計・実装（別イシュー #1018）。
- forward 経路の常駐化（別イシュー、#1022 と紐づく想定）。
- SGD の単一カーネル化（別イシュー、#1023 と紐づく想定）。
- `MTLDispatchTypeConcurrent` によるバッチ内並列化（§5(c)）。

## 7. 実装記録（#1017）

§0 で「最終決定は #1017 が行う」としていた事項の確定内容:

- **`Batch: Send`（§2.1）**: `unsafe impl Send for Batch`
  （`crates/backend-metal/src/context.rs`）を採用した。`Batch`
  （`cmd_buf`／`encoder`／`in_flight`／`tokens`）は `MetalContext::batch`
  （`Mutex<BatchSlots>`）のロック下でのみアクセスされ、`encode`／
  `flush`／`synchronize`／`Drop` の全経路がこの直列化を通る（Mutex に
  よる全アクセスの完全直列化）。§2.1 で検討した代替案（thread_local
  バッチ・個別コマンドバッファ即時 commit・`addCompletedHandler`）は
  いずれも不採用のまま（context.rs の SAFETY コメント参照）。
- **共有失敗トークンの経路（§3.7 (2)）**: `BackendOps` へ
  `sgd_step_device_tracked`（デフォルトメソッド。既定は
  `sgd_step_device` へ委譲。`crates/tensor-core/src/backend_ops.rs`）を
  追加する非破壊拡張を採用した。トークン
  （`fandhe_ai_tensor_core::DispatchFailureCell`。
  `crates/tensor-core/src/dispatch_failure.rs`）は `MetalContext::encode`
  と同一ロック区間でバッチへ登録する。「encode 後に別 API で登録」方式
  は不採用。
- **`BackendOps::synchronize`（§3.7 (3)）**: 本 PR では追加しない。
  `#1012`（CUDA 側）マージ後に main が trait 契約を判断する（§6.2 は
  未着手のまま残す）。
- **上限値**: `MAX_DISPATCHES_PER_BATCH = 256`
  （`crates/backend-metal/src/batch_state.rs`）。
- **同期点**: `memory.rs::download_inner`／`PoolZeroFill::zero_fill`・
  `MetalContext::Drop` に `synchronize()` を追加した（§3.5 の表どおり）。
  既存の `dispatch_sync` 呼び出し元（`gemm.rs`／`elementwise.rs`／
  `rmsnorm.rs`／`softmax.rs`）はシグネチャ・挙動とも無変更。
- **§4 実測**: 2026-08-31・Apple M4 Max 実機で実測済み（§4 参照）。
  `crates/backend-metal/tests/command_batching_bench.rs`
  （#1017 単独の効果を隔離するマイクロベンチ）・`crates/facade/tests/
  mnist_scale_train_reuse_bench.rs`（`bench-fandhe` 相当プロトコルの
  HEAD 絶対値計測）を新規追加した。`crates/backend-metal/tests/
  command_batching.rs` 冒頭コメント・`crates/facade/tests/
  device_param_store_bench.rs` の `legacy_vs_resident_per_step_metal`
  は正しさ検証・小規模ベンチの既存参照として引き続き有効。
