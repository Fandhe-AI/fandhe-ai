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

**§4.2 の Mac セッション記入時点（2026-08-31・HEAD）での再カウント（レビュー
指摘対応）**: 上記 11 箇所の列挙は #1017 着手前時点のものであり、以下の
2 点で HEAD とは既に乖離している。(1) `sgd.rs` は #1017 実装後
`dispatch_sync` を経由せず `ctx.encode` を直接呼ぶ（`token: None` の場合の
み `MetalSgd::run` 内で `ctx.synchronize()` を追加で呼ぶ。`sgd.rs:150,171`。
`grep -n '\.dispatch_sync(' crates/backend-metal/src/sgd.rs` は 0 件）ため、
そもそも「11 箇所」に `sgd.rs` を含めるのは #1017 以降誤り。(2) #1078
（MSE loss reduction 融合）で `mse.rs` に新規 `dispatch_sync` 呼び出し
（3 箇所）が追加されている。`grep -n '\.dispatch_sync(' crates/backend-metal/
src/*.rs`（2026-08-31 実測）による HEAD 時点の実件数は
`gemm.rs:780,906,1077,1207,1270,1406,1421`（7 箇所）・
`elementwise.rs:143,169`（2 箇所）・`mse.rs:132,145,184`（3 箇所）・
`rmsnorm.rs:216`（1 箇所）・`softmax.rs:145`（1 箇所）の**計 14 箇所**
（`sgd.rs` は含まれない）。§4.2 はこの HEAD 時点の実件数を根拠とする。

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

**#1099 追記**: 上記は `fandhe_ai_tensor_core::pool::PooledMemory<
MetalMemory>`（`memory.rs::PoolZeroFill for MetalMemory::zero_fill`）を
指しており、この経路は無変更のまま維持する。一方、`crate::pool::
MetalAllocator`（`buffer.rs::alloc_zeroed_pooled`／`alloc_uninit_pooled`
経由。`ops.rs` の GEMM／MSE 出力バッファ確保が使う、実際に本イシューが
問題視した経路）は、#1021 の pending-return 不変条件によりフリーリスト
上のバッファが常に GPU 未参照であることが構造的に保証されるため、
`synchronize()` を経由せずに「ホスト書き込み前に GPU 完了を待つ」という
一般契約を満たせることが判明し、`synchronize()` を除去した。証明・実測
は §8 を参照。

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

**再カウントの方法（codex-review PR #1097 P2 是正・2026-08-31 再実測）**:
前版は「GEMM 6 + SGD 1 = 7 回」という不完全な内訳から「境界で 1 組だけ
統合され 7→6」と静的に導出していたが、実測ベンチ（§4.4・
`mnist_scale_train_reuse_bench.rs` の 1 step）は `MseLoss::forward`／
`backward` も実行しており、Metal の `mse_loss`／`mse_loss_backward`
（`crates/backend-metal/src/mse.rs` の `dispatch_sync` 呼び出し 3 箇所・
132/145/184 行）を経由する。これが「7 回」に含まれていなかった。本節は
一時的な実行時カウンタ（`MetalContext::encode`／`synchronize` に
`AtomicUsize` を追加し、ラベル列も記録）を `context.rs`・
`crates/facade/tests/`（一時ファイル）へ**一時的に**追加し、実際の
reuse ループ 1 step を計測してから、計装を `git checkout` で完全に revert
した（本番コードへは計装を残していない。以下の数値はその実測ログを
根拠とする）。

**実測結果（steady-state・3 step warmup 後、単独 1 step と 10 step
平均の両方で一致。2026-08-31・M4 Max）**:

| 指標 | 実測値（1 step あたり） |
| --- | --- |
| `encode()` 呼び出し総数（= 個々のディスパッチ数） | **9**（10 step 平均でも 9.000 と完全一致） |
| 実際に生成されたコマンドバッファ数 | **9**（`encode()` 呼び出し数と完全一致） |
| `waitUntilCompleted()` 呼び出し数 | **9**（10 step 平均） |
| バッチあたりの dispatch 数（`BatchMeta::dispatch_count()`） | 全 9 バッチとも **1**（`batch_state.rs:55`） |

単独 1 step のラベル列は `["dispatch_sync" ×8, "sgd_step_f32" ×1]`
（`dispatch_sync` は `gemm.rs`／`elementwise.rs`／`mse.rs` の
`ctx.dispatch_sync` ラッパーが常に同一の文字列リテラルを渡すため
〈`context.rs:317` 付近の `self.encode("dispatch_sync", ...)`〉、ラベル
だけでは呼び出し元まで判別できない）。内訳は構造的に次のとおりと推定
される: forward の Linear1（`gemm_resident_rhs`。`ops.rs:571-` が既定実装
で呼ぶ `dispatch_strided_bias_act_prepared`）1 回 + forward の ReLU
（`elementwise.rs`）1 回 + forward の Linear2（`gemm_resident_rhs`）1 回
+ MSE forward（`mse.rs:132,145` の 2 段リダクション）2 回 + MSE
backward（`mse.rs:184`）1 回 + backward 側の resident GEMM 系呼び出し
（`gemm_resident_lhs`〈`ops.rs:777`〉等）2 回 = 8 回、に `sgd_step_f32`
（SGD tracked encode。`sgd.rs:150`）1 回を加えた計 9 回。この内訳の各
呼び出し元までの厳密な対応は、ラベルが区別できないため実測ログのみ
からは確定できない（総数 9 とバッチサイズ分布が全 1 であることが
主張の核心であり、個別呼び出し元の対応は補助的な推定に留める）。

**なぜ #1017 のバッチ化が全く発現しないか（file:line 根拠）**: 9 個の
バッチがいずれも dispatch 数 1（＝マージなし）だった理由は、GEMM・MSE
の出力バッファがプール経由の **zero-on-reuse** 確保
（`mem.alloc_zeroed(...)`。`crates/backend-metal/src/ops.rs:674`
〈`gemm_resident_rhs` の `c_dev_buf`〉・`ops.rs:777`
〈`gemm_resident_lhs`〉・`ops.rs:1114`〈別の resident GEMM 経路〉）で
確保されており、`SizeClassPool` がフリーリストの再利用バッファを返す際
（`crates/backend-metal/src/pool.rs:329,343-358`）、`zero_on_reuse ==
true` なら **`self.context.synchronize()` を無条件に呼んでからゼロ
クリアする**契約になっているため（「前利用者のバイト残留を防ぐため」。
`pool.rs:345-347` のコメント参照）。この `synchronize()` は当該 GEMM／
MSE 呼び出し自身の `dispatch_sync`（＝ `encode()` + `synchronize()`）が
走る**前**に発生する。したがって、直前の op（前 step の
`sgd_step_device_tracked` が残した未 flush バッチを含む）がまだ open の
ままであっても、次の GEMM／MSE 呼び出しの出力バッファ確保時点で
プール由来の `synchronize()` が先に走り、open batch を flush してしまう
——結果として当該 GEMM／MSE 自身の `encode()` が呼ばれる頃には常に
`slots.open.is_none()` であり、**マージの機会自体が構造的に発生しない**
（steady-state な MLP 学習では GEMM 出力サイズが毎 step 同一のため、
2 step 目以降はほぼ確実にプールのフリーリストが命中し、この
`synchronize()` が毎回発火する）。

- `sgd_step_device`（`token: None`。§3.7 (3) の非バッチ契約）は
  変更前・変更後とも「`encode` 直後に `ctx.synchronize()`」を維持
  （`ops.rs:375-390`・`sgd.rs` doc コメント）。無変更。
- `sgd_step_device_tracked`（`token: Some`）は #1017 で `encode` のみを
  行い、その場では待たない契約へ変更された（`context.rs::encode`）。
  `DeviceParamStore::step` は #1023（#1017 以前）で既にパラメータ数に
  依らず 1 回の起動へ集約済み（`device_store.rs:43-44`）であるため、
  reuse ループの 1 step あたりの `sgd_step_device_tracked` 呼び出しは
  常に **1 回**。
- **結論（実測により訂正）**: 上記のプール由来 `synchronize()` の介在に
  より、`sgd_step_device_tracked` が「待たない」契約になったこと自体は、
  **現行の MLP reuse 学習ループにおいてコマンドバッファ数・
  `waitUntilCompleted()` 呼び出し数を 1 つも削減しない**（実測: 9 回→
  9 回、削減率 0%）。#1017 が変えるのは、SGD 起動の完了待ちが「起動直後
  の専用ブロッキング待機」から「次の GEMM／MSE 呼び出しが自身の出力
  バッファをプールから再利用する際に副次的に要求する `synchronize()`
  への遅延」へ移った、という点のみであり、コマンドバッファ・wait の
  **回数**そのものへの寄与は本ワークロードでは実測上ゼロである。
  §4.3 のマイクロベンチ（GEMM／MSE のようなプール確保を挟まず
  `sgd_step_device_tracked` だけを連続投入する）が示す高速化は、この
  「プール確保が介在しない」特殊条件下でのみ現れる上限効果であり、
  実際の学習ループでは発現しないことを本節の実測が示す。

### 4.3 マイクロベンチ: #1017 の効果を隔離した計測（HEAD・新規追加）

`crates/backend-metal/tests/command_batching_bench.rs`
（`command_batching_micro_bench_untracked_vs_tracked`）で、同一 shape
（要素数 1024）・同一パラメータ／勾配バッファに対し、非バッチ経路
（`sgd_step_device`。呼び出しごとに同期）とバッチ経路
（`sgd_step_device_tracked`。100 回連続 `encode` の後に 1 回だけ
`download` で同期）を 100 回連続更新× 5 trial の中央値で比較した。
**各 trial で非バッチ経路とバッチ経路の最終パラメータ（`download` の
戻り値）を REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差
1e-5 未満）で突き合わせ、一致を確認したうえで時間を採用する**
（codex-review PR #1097 P2 是正。初版はバッチ経路の `download` 結果を
破棄しており 100 回更新の反映を未検証のまま計測していた）。

| 経路 | 100 回連続更新の中央値（Q1〜Q3） |
| --- | --- |
| 非バッチ（`sgd_step_device`。#1017 以前の `sgd_step_device_tracked` デフォルト委譲と同一の同期契約） | 12.129 ms（10.446〜13.435 ms） |
| バッチ（`sgd_step_device_tracked`。#1017） | 0.718 ms（0.705〜0.751 ms） |

speedup 約 16.9 倍（`tracked_faster=true`。5 trial とも数値一致検証
green）。これは #1017 が導入した「`encode` の連続投入 + 単一の遅延同期」
機構そのものの効果を、他の性能改善コミットと混同せず HEAD 上で単独に
隔離した計測である。**§4.2（実測により訂正済み）で確認したとおり、
現行の MLP reuse 学習ループでは GEMM／MSE 出力のプール確保
（zero-on-reuse）が毎 dispatch ごとに `synchronize()` を強制するため、
この約 17 倍のマージ効果は発現しない**（実測: MLP 1 step あたりの
コマンドバッファ・wait 回数削減は 0%。§4.2）。本節の数値は「GEMM／MSE
のようなプール確保を挟まない、`sgd_step_device_tracked` の連続投入」と
いう狭い条件下での #1017 機構自体の上限効果を示す指標であり、実際の
MLP 1 step の短縮への直接的寄与ではない。

### 4.4 HEAD 絶対値: MLP 学習 1 step（MNIST 規模 2 層 MLP・train・reuse）

`crates/facade/tests/mnist_scale_train_reuse_bench.rs`
（`mnist_scale_train_fresh_vs_reuse_metal`）で、`bench-fandhe` と同一の
モデル形状（`BATCH=64`／`D_IN=784`／`D_HIDDEN=256`／`D_OUT=10`）・
乱数シード・`TRAIN_STEPS=100`／`TRAIN_WARMUP=20`（先頭 20 step を捨て
残り 80 step の median/Q1/Q3 を取る、`bench-fandhe` と同一境界の 1 実行）
のプロトコルを HEAD の `facade` crate（path 依存）で再現した。**計測
境界の訂正（codex-review PR #1097 P1 是正）**: 初版は上記 1 実行の外側に
追加の全量 warmup 実行（`run_fresh`／`run_reuse` の破棄呼び出し）を
挟んでおり `bench-fandhe` と計測境界が異なっていた。本版はこの追加
warmup を削除し、`bench-fandhe` と同一境界の 1 実行を 5 trial 独立に
繰り返し、各実行の median をさらに中央値化した（各実行内の先頭 20 step
が `bench-fandhe` と同じ役割の warmup を兼ねる）。**数値検証（同 PR
P2 是正）**: `bench-fandhe` と同じく最終 step の loss の有限性、reuse
では終端同期後のパラメータ個数・全要素有限性を検証し、いずれかが破れた
場合は計測結果を採用せず失敗させる契約にした（初版は loss を読み捨て、
同期後パラメータも未検査だった）。

| | fresh（ホスト経由 SGD） | reuse（デバイス常駐 SGD） | reuse/fresh |
| --- | --- | --- | --- |
| HEAD（本実測。M4 Max。5 trial 中央値） | 19.070 ms（19.043〜20.405 ms） | 9.485 ms（9.358〜9.533 ms） | 2.011 倍高速 |
| `fandhe-ai =0.4.0`（`results/summary.md` 環境 5 (b') 表。§1 引用） | 19.699 ms | 20.381 ms | 0.966 倍（reuse が遅い） |

**境界差の明記（§4.1）**: 上段（HEAD）と下段（0.4.0）は計測対象コード
のバージョンが異なる。HEAD の reuse 9.485 ms は #1017 単独の効果ではなく、
0.4.0 以降にマージされた性能改善コミット群（#1013・#1023・#1028・
#1043〜#1047・#1044・#1078〜#1082 等。冒頭コメント参照）を累積した結果
であり、§4.3 のマイクロベンチが #1017 単独の delta を担う。

### 4.5 #1015 受け入れ条件の充足判定

受け入れ条件「改善が見られること、かつ既存の数値一致 parity テストが
green のまま」を以下のとおり判定する。

- **改善**: (a) HEAD 絶対値として reuse が 0.4.0 の 20.381 ms から
  9.485 ms へ改善し、かつ 0.4.0 で発生していた「reuse が fresh より
  遅い」逆転（0.966 倍）が解消され reuse が fresh の 2.011 倍高速に
  なった（§4.4。ただし累積 delta であり #1017 単独の寄与ではない）。
  (b) #1017 が導入したバッチ化機構そのものは、その機構を直接行使する
  経路（連続 `sgd_step_device_tracked` 呼び出し）において約 16.9 倍の
  高速化を示した（§4.3）。しかし §4.2 の実測（一時計装・revert 済み）
  により、**現行の MLP reuse 学習ループではこの機構が一切発現しない**
  ことが判明した: GEMM／MSE の出力バッファがプール経由の zero-on-reuse
  確保（`ops.rs:674,777,1114` の `mem.alloc_zeroed`）であり、プールの
  フリーリスト再利用が `pool.rs:329,343-358` の `self.context.
  synchronize()` を毎 dispatch ごとに強制するため、`sgd_step_device_
  tracked` が残す未 flush バッチは常にこの副次的な `synchronize()` で
  単独 flush され、後続 dispatch とマージする機会が構造的に存在しない
  （実測: 1 step あたりのコマンドバッファ・`waitUntilCompleted()`
  呼び出し数は 9 回のまま、削減率 0%）。したがって (a) の HEAD 絶対値
  改善は #1017 以外の性能改善コミット（§4.4 冒頭コメント参照）に
  ほぼ全面的に帰属し、**#1017 自身が本 MLP ワークロードの実行時間短縮に
  寄与した実測上の根拠は無い**点を正直に記録する。#1017 の効果が実際に
  現れうるのは、GEMM／MSE のような zero-on-reuse プール確保を挟まない
  経路（例: 複数パラメータの `sgd_step_device_tracked` を連続投入し
  最後にまとめて `download` する用途。§4.3 のマイクロベンチが示す条件）
  に限定される。
- **parity green**: `make test-ignored-metal`（`cargo test -p
  fandhe-ai-backend-metal --release -- --ignored --nocapture` 相当）
  を実行し、既存の実機依存テスト（`command_batching.rs` の #1017
  受け入れ条件 1〜4 を含む）・parity テスト（`sgd_device_parity.rs`・
  `cpu_metal_parity.rs`・`gemm_*_parity.rs`・`rmsnorm_parity.rs`・
  `softmax_parity.rs` 等）・`command_batching_bench.rs` の数値一致検証
  （§4.3）を含む全 31 個の `test result: ok`（failed 0）ブロックで完走
  したことを確認済み（2026-08-31 実測・codex-review PR #1097 の P1/P2
  是正後に再実測。内訳: lib unittests 1〈`src/lib.rs`〉+ `tests/*.rs`
  統合テスト 28 本〈本ファイル §4.3 の新規 `command_batching_bench.rs`
  を含む〉+ `[[example]]` の `test = true` 指定分 1〈`examples/
  gemm_splitk_shapes_bench.rs`。`Cargo.toml` コメント参照〉+ 末尾の
  `Doc-tests fandhe_ai_backend_metal` パス 1〈doctest 0 件〉= 31。
  `grep -c 'test result: ok'` と `grep -c '^running'` がいずれも 31 で
  一致することを実行ログから確認済み）。`cargo clippy --workspace
  --all-targets --all-features -- -D warnings` はネイティブ macOS ターゲット
  で exit 0（0 error）。Linux ターゲットでの dead_code 是正
  （`crates/facade/tests/mnist_scale_train_reuse_bench.rs` への
  `#![cfg(target_os = "macos")]` 追加）は `cargo check --target
  x86_64-unknown-linux-gnu -p fandhe-ai --tests` で当該ファイル由来の
  warning が 0 件になったことを確認済み（本 Mac に `x86_64-linux-gnu-gcc`
  クロスリンカが無いため `cargo clippy --workspace --target
  x86_64-unknown-linux-gnu` のフルビルドは再現不能。詳細は PR #1097
  対応コミットの報告を参照）。

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

## 8. 実装記録（#1099。§4.2・§4.4・§4.5・§3.4・§3.5 の追記）

§4.2・§4.5 が特定した「9 個のバッチがいずれも dispatch 数 1（マージ
なし）」の直接原因（`pool.rs::MetalAllocator::alloc_inner` の
zero-on-reuse 経路が無条件 `self.context.synchronize()` を呼ぶ）を
解消した（イシュー #1099）。

### 8.1 §3.4／§3.5 の契約改訂

§3.4 の「メモリプール導入後もこの契約（ゼロ埋めはホスト書き込みであり
同期を要する）は変更しない」は、**`crate::pool::MetalAllocator`
（`buffer.rs::alloc_zeroed_pooled`／`alloc_uninit_pooled` 経由。
`ops.rs` の GEMM／MSE 出力バッファ確保が使う経路）に限り本イシューで
変更する**。`fandhe_ai_tensor_core::pool::PooledMemory<MetalMemory>`
（`memory.rs::PoolZeroFill for MetalMemory::zero_fill`。§3.5 表の
`PoolZeroFill::zero_fill` 行）は本イシューのスコープ外の**別の**
プール機構であり、同期契約は無変更のまま維持する（`zero_fill` 自身は
「そのバッファを最後に読み書きした dispatch」の完了を待つ必要がある
という一般契約自体は変わらないが、`MetalAllocator` はこの一般契約を
満たすことを別の不変条件〈8.2〉で構造的に保証できるため、同期を
経由せずに満たす）。

`crate::pool::MetalAllocator::alloc_inner` の zero-on-reuse 経路
（`pool.rs`）から `self.context.synchronize()` を除去した。安全性の
根拠（#1021 の pending-return 不変条件）:

1. バッファの返却経路は `PooledMetalHandle::Drop` →
   `MetalContext::defer_pool_return` のみ（`context.rs:249-`）。
   `BatchSlots` の同一ロック区間内で `open`／`committed`（in-flight）
   バッチの有無を検査し、in-flight ならフリーリストへ入れず
   `pending_pool_returns` へ退避する（`pool_pending.rs::
   PendingReturns::defer_or_release`）。
2. 退避分の合流（`put_all_merged`）は `MetalContext::synchronize` の
   `waitUntilCompleted()` **完了後**、同一ロック保持のまま行う
   （`synchronize` は待機中も `BatchSlots` ロックを保持し続けるため、
   「committed を take した後・完了前」に別スレッドの Drop が即時
   返却へ抜ける TOCTOU 窓は存在しない）。
3. 即時返却（`in_flight == false`）が成立するのは open/committed
   バッチが 1 つも無い時点のみで、その時点でこのコンテキストの全
   GPU work は完了済み。

よって**フリーリスト上のバッファはいかなる open／committed／実行中
コマンドバッファからも参照されない**（GPU 未参照）ことが不変条件と
して成立し、`take()` で得た再利用バッファへのホスト書き込み
（`zero_fill_logical`）に同期は不要（§3.6 の「ホスト→GPU」可視性
保証はバッファを参照する dispatch の commit 前にホスト書き込みが
完了していることのみを要求し、本経路はその条件を満たす）。

**エラー伝播の位置づけの変化**: 従来は再利用確保時の `synchronize()`
が前バッチの実行エラーを alloc エラーとして早期に表面化させていたが、
除去後はエラー自体が失われるわけではなく、次のホスト実体化（各 op の
`dispatch_sync`／`download`／`Drop` が呼ぶ `synchronize()`）で
`DispatchFailureCell` 経由で登録済み全トークンへ伝播する契約（#1017・
§3.7 (2)）がそのまま働く。位置が「確保時」から「次の同期点」へ遅延
するのみで、エラーが握り潰される経路は増えない。

### 8.2 `PoolStats` 契約との非矛盾

`record_reuse` 後、`zero_on_reuse` 分岐は従来 `synchronize()` 失敗時に
`record_loan_end` で統計を巻き戻していたが、この分岐自体が
`synchronize()` 呼び出しと共に消滅した（`zero_fill_logical` はホスト
側のみの操作で失敗しない）。`pending_return_bytes`／
`record_pending_return`／`put_merged`（`docs/device-memory-pool-design.md`
§3.1・§3.3）の契約には一切触れておらず、返却経路（`defer_pool_return`）
・合流経路（`synchronize` 内の `drain_for_merge` → `put_all_merged`）は
無変更のまま。取得経路（`take()` → `record_reuse`）のみを変更した。

### 8.3 実測（2026-08-31・Apple M4 Max 実機）

**§4.2 の再実測（診断カウンタは本番コードへ恒久化。テスト・診断専用
`#[doc(hidden)]` API として `crates/backend-metal/src/context.rs::
MetalContext::diagnostic_batch_counters`／`lib.rs::
__diagnostic_batch_counters_snapshot` に残す。§4.2 が「一時計装・
`git checkout` で完全に revert」した方式から変更した）**:
`crates/facade/tests/mnist_scale_train_reuse_bench.rs::
mnist_scale_train_reuse_metal_batch_counters`（WARMUP=20 step 後の
steady-state 1 step）で実測。

| 指標 | before（§4.2・#1099 適用前） | after（#1099 適用後・本実測） |
| --- | --- | --- |
| `encode()` 呼び出し総数 | 9 | **9**（不変） |
| コマンドバッファ生成数 | 9 | **8** |
| `waitUntilCompleted()` 呼び出し数 | 9 | **8** |

§4.2 が静的に見込んでいた「境界で 1 組統合され 9→8」という想定どおり
の結果になった（前 step の `sgd_step_f32` tracked encode が、次 step
先頭の dispatch と同一バッチへ合流する）。

**§4.4 の再実測**（`mnist_scale_train_fresh_vs_reuse_metal`。5 trial
中央値）:

| | fresh | reuse | reuse/fresh |
| --- | --- | --- | --- |
| #1099 適用前（§4.4 表） | 19.070 ms（19.043〜20.405 ms） | 9.485 ms（9.358〜9.533 ms） | 2.011 倍高速 |
| #1099 適用後（本実測） | 18.148 ms（18.078〜18.244 ms） | 8.862 ms（8.725〜8.870 ms） | 2.048 倍高速 |

コマンドバッファ・wait 回数の 1 step あたり 1 回減（9→8、約 11%
削減）に対し、reuse 中央値の短縮は 9.485 ms → 8.862 ms（約 6.6%）。
§4.2 が既に指摘したとおり、この MLP ワークロードの支配項は
`docs/perf/train-step-phase-breakdown.md` の backward（83.6〜97.3%）
であり、コマンドバッファ数の削減はホスト側の固定費（コマンドバッファ
生成・`waitUntilCompleted` の呼び出しオーバーヘッド）のみに効くため、
削減率が 1 step 全体の短縮率より小さいのは整合的である。

**新規追加テスト**（実機実行済み・全 green。`--ignored --test-threads=1`）:

- `crates/backend-metal/tests/command_batching.rs::
  pool_reuse_zero_fill_does_not_synchronize_open_batch`（本イシューの
  核心となる回帰テスト。open バッチが存在する状態でのプール再利用
  ゼロ埋めが `waitUntilCompleted` を発火させないこと・再利用バッファが
  全ゼロであること・SGD 更新結果が正しいことを検証）。
- `crates/backend-metal/tests/command_batching_bench.rs::
  pool_reuse_interleaved_with_tracked_steps_preserves_batching`
  （50 step の tracked SGD + プール再利用の混在投入で、コマンドバッファ
  生成数〈1〉が `encode()` 呼び出し数〈50〉より大幅に少ないことを確認。
  受入条件 2 のもう一方の確認先）。
- `crates/facade/tests/mnist_scale_train_reuse_bench.rs::
  mnist_scale_train_reuse_metal_batch_counters`（上表の 9/8/8 を assert
  する回帰テスト）。

**既存 parity・回帰テストの green 確認**: `cargo test -p
fandhe-ai-backend-metal --release -- --ignored --test-threads=1`
（`pool_real_device.rs` の再利用時全ゼロ契約テストを含む全ての実機
ignored テスト）・`cargo test -p fandhe-ai --release --test
device_param_store_backend_parity -- --ignored --nocapture
--test-threads=1`（`device_resident_matches_host_sgd_on_metal_across_
100_steps`）が green（CUDA 側ケースは本 Mac に CUDA 実機が無いため
`CudaUnavailable` で失敗するが、#1099 の変更と無関係な既知の環境制約）。
tolerance（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）は変更していない。
`cargo fmt --all -- --check`／`cargo clippy -p fandhe-ai-backend-metal
--lib --tests --examples -- -D warnings` は 0 差分・0 warning
（`fandhe-ai-backend-cuda` の一部 dead-code warning は本 Mac に CUDA
toolkit 由来の `internal-diagnostics` feature 依存関数群が未使用に
なる既知の環境依存であり #1099 と無関係。`git stash` での前後比較で
本 PR の変更前から同一であることを確認済み）。

## 9. GPU タイムスタンプ診断オブザーバ（イシュー #1276）

`§3.5` の唯一の同期点である `MetalContext::synchronize` の契約
（ロック保持・エラー集約・`pending_pool_returns` 合流順序）はイシュー
#1276 でも不変。本体を「完了バッチごとに呼ばれるオブザーバ付き内部
関数」（`synchronize_observed`）へ切り出し、公開 `synchronize()` は
no-op オブザーバで呼ぶ薄いラッパーへ変更した。オブザーバは
`waitUntilCompleted()`・`status()` 判定の直後・`batch` drop 前に
`self.batch` のロックを保持したまま呼ばれるため、`Batch: Send` の
SAFETY コメントが要求する「`Mutex` 下でのみ `Batch` へ触れる」不変
条件は維持される。`#[cfg(test)] pub(crate) synchronize_with_gpu_
timestamps`（診断専用）が `MTLCommandBuffer::GPUStartTime`/
`GPUEndTime` を読むオブザーバの実装先。本番経路は追加の FFI 呼び出し
ゼロ（`docs/perf/metal-gemm-reuse-phase-breakdown.md` §2/§8 参照）。
