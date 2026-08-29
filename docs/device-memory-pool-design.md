# デバイスメモリ・プールアロケータの設計

対応: イシュー #1019（親 #1018 ツリー。上位ゴールは #1008「学習ループの固定費
削減」）。

## §0 位置づけ・スコープ

- 本文書は #1018 ツリーの**第 1 段（設計）**であり、`crates/**` は変更しない
  （docs-only）。API シグネチャは「案」として示し、最終決定は実装イシューで
  ある #1020（CUDA 実装）・#1021（Metal 実装）が実装しながら行う。
- 効果測定欄（§3.7）は Mac（M4 Max）・DGX Spark（GB10）実機セッションでの
  記入待ちの空欄とする。設計時点で「割当／解放区間」の実測値は存在せず、
  推定値では埋めない（`docs/backend-metal-buffer-pool-decision.md` §3 と
  同じ立場）。
- 依存状況（forward reference 方針）: 本文書は #1054
  （`docs/backend-metal-command-batching-design.md`。マージ済み）§3.4・§3.5
  の契約を前提として参照する。#1012（CUDA 側非同期実行設計）は本稿執筆時点
  で PR #1053（`docs/backend-cuda-async-execution-design.md`）として未マージ
  のため、依存せず現行コード事実（cudarc の既存挙動）から契約を自立記述し、
  マージ後に相互参照を追記する。
- 不変事項（変更しない）: バックエンド間数値一致の複合判定（相対誤差
  1e-3 未満または絶対誤差 1e-5 未満）・カーネル境界検査・依存の完全固定
  （`=x.y.z`）・REQ-14 のピークメモリ係数上限（2 倍以内）・ガードレール
  閾値／テスト許容誤差。緩和・厳格化はいずれもユーザー承認事項。

## §1 背景・実測根拠

- #1008 実測（学習 1 step。BATCH=64・D_IN=784・D_HIDDEN=256・D_OUT=10 の
  MLP）: CUDA 12.4 ms／Metal 18.6 ms／CPU 13.9〜17 ms 対 candle
  0.28〜0.81 ms。20〜40 倍差の主因として、演算ごとの同期・毎 step の
  D2H/H2D 転送・都度のバッファ割当／解放・不要なホストコピーが挙げられて
  いる。
- 内訳の確定は #1010（`docs/perf/train-step-phase-breakdown.md`。未着手）が
  担当する。本文書は #1009 で追加された `train --phases`
  （`scripts/bench/framework-compare/bench-fandhe`）の区間分解を計測手段
  として前提にするが、区間ごとの実測値そのものは本文書の対象外。
- したがって本設計の「効果がある見込み」は §3.7 に記す区間名との対応関係
  までであり、定量的な改善幅は記載しない。

## §2 現状整理（コードから確定できる事実）

| # | 事実 | 出典 |
|---|---|---|
| 1 | opt-in デコレータ `PooledMemory<M>`（バイトサイズ完全一致バケット・総量上限 128MiB 既定・グローバル LRU・`release_all_pooled`・`PoolZeroFill`）が既に存在する（#200/#201/#202 完了済み） | `crates/tensor-core/src/pool.rs`・`docs/memory-pool-design.md` |
| 2 | しかし `BackendOps::memory_ops()` が返すのは素の `CudaMemory`／`MetalMemory`／`CpuMemory`（`Box::leak` によるプロセス内 `'static` シングルトン）であり、`PooledMemory` はどの本番経路にも接続されていない | `crates/backend-cuda/src/ops.rs`（`static_cuda_memory`・`memory_ops`）・`crates/backend-metal/src/ops.rs`・`crates/backend-cpu/src/ops.rs` |
| 3 | 学習ホットパス（GEMM・elementwise・softmax）は `MemoryOps` を経由せず、演算ごとに CUDA では `stream.clone_htod`（入力アップロード）＋`stream.alloc_zeros`（出力確保）を直接呼ぶ | `crates/backend-cuda/src/gemm.rs`・`elementwise.rs`・`softmax.rs` |
| 4 | Metal も同様に `MetalBuffer::new_with_data`／`new_zeroed` を演算ごとに直接確保する（`gemm.rs`・`elementwise.rs` に多数箇所） | `crates/backend-metal/src/gemm.rs`・`elementwise.rs`・`docs/backend-metal-buffer-pool-decision.md` §1 |
| 5 | 完全一致方式にした理由は `download` 契約（ハンドル実長 = numel）を壊さないため。サイズクラス丸め（capacity と論理 numel の分離）は導入時点で申し送りのスコープ外事項として明記されていた | `docs/memory-pool-design.md`「サイズクラス方針」「スコープ外」節 |
| 6 | cudarc 0.19.8 は `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED`（`has_async_alloc`）が真なら `cuMemAllocAsync`／`cuMemFreeAsync`（driver 側 stream-ordered アロケータ）を使い、偽ならホスト同期を伴う通常の `cuMemAlloc`／`cuMemFree` に退化する。`CudaSlice::drop` は記録イベントをデバイス側で待ってから解放する。`cuMemPool*`（`get_mem_pool`／`set_mem_pool`／`result::mem_pool` 配下の attribute 操作・alloc_async 等）の raw binding も存在する | `cudarc-0.19.8/src/driver/safe/core.rs`（`has_async_alloc` フィールド・`malloc_async`／`free_async` 呼び出し箇所）・`src/driver/result.rs`（`mem_pool` モジュール・`get_default_mem_pool`／`get_mem_pool`／`set_mem_pool`） |
| 7 | `BufferHandle`／`DeviceBuffer` は `Send`/`Sync` を要求しない方針（v1 判断。「必要になった時点で再検討」）。`PooledMemory` の `Arc<Mutex<PoolCore>>` は `clippy::arc_with_non_send_sync` を明示 `allow` している | `crates/tensor-core/src/buffer.rs`「Send/Sync 境界」節・`crates/tensor-core/src/pool.rs`（`arc_with_non_send_sync` allow 箇所） |
| 8 | Metal 側の `MTLBuffer`／`MTLLibrary` 等 protocol は objc2-metal 0.3.2 で `Send + Sync` を supertrait に持つ（`context_cache.rs` のコンパイル時 `assert_send_sync` で固定）。#1054 §3.4 は「in-flight バッチが参照しうる `MetalBuffer` へのホスト読み書き（`zero_fill`・プール再利用の `new_with_data` を含む）は必ず `synchronize()` を経由してから行う」と契約し、プール導入後もこの契約を変えないと明記している | `crates/backend-metal/src/context_cache.rs`（`assert_send_sync` ブロック）・`docs/backend-metal-command-batching-design.md` §3.4・§3.5 |
| 9 | REQ-14 は「バッファプール等のキャッシュ機構を導入する場合、係数上限（2 倍以内）を維持できなければプール解放 API を提供する」ことを求める。係数 2.0（GEMM 4096³ で 384MiB 以内）は #179 で実測（対理論比 1.000）に基づき確定済み・変更なし | `docs/peak-memory-coefficient-decision.md` |
| 10 | 代表ワークロード（MLP 学習。`bench-fandhe`）の形状: `BATCH=64`・`D_IN=784`・`D_HIDDEN=256`・`D_OUT=10` → f32 でのバイトサイズは `x` 200,704 B・`W1` 802,816 B・`h` 65,536 B・`W2` 10,240 B・`out` 2,560 B。GEMM 4096² f32 = 64 MiB | `scripts/bench/framework-compare/bench-fandhe/src/main.rs` の `BATCH`／`D_IN`／`D_HIDDEN`／`D_OUT` 定数 |
| 11 | `train --phases` の区間名: `tape_build`／`leaf_register`／`forward`／`forward_resident`／`loss_readout`／`backward`／`param_readout`／`host_sgd`／`apply_params`／`device_update`／`tape_drop`／`step_total` | 同ファイルの `PHASE_*` 定数群 |

親 #1018 の「CPU も CUDA も都度確保（プールなし）」という記述は、上記
事実 1・2 に照らして正確化する: **プール機構（`PooledMemory`）自体は
既に存在するが、いずれの本番バックエンド経路にも接続されておらず、実質的
に「都度確保」のまま**、というのが正しい現状である。

## §3 設計（design decisions）

### 3.1 層構造（`SizeClassPool<H>`・`PoolConfig`・`PoolStats`。API 案）

**レビュー履歴（設計の 3 段階）**: 本節は codex-review PR #1056 で 2 度の
P1 指摘を経て以下の設計へ確定した。(1) 当初案は `BackendOps::allocator()`
が `&dyn DeviceAllocator`（`Box<dyn BufferHandle + Send>` を返す trait）を
公開する設計だった → `BackendOps` という公開ディスパッチ trait から内部
低水準バッファ表現へ到達できる、との P1 指摘（1 巡目）。(2) `BackendOps::
allocator()` を削除し `DeviceAllocator` の実装インスタンスをいずれの公開
関数からも返さない設計へ変更した → しかし `pub trait DeviceAllocator` と
`Box<dyn BufferHandle + Send>` という型定義自体が `fandhe-ai-tensor-core`
の公開面（`pub mod allocator`）に残り、外部コードがこのクレートへ直接
依存すれば型を参照・実装できてしまう、との P1 再指摘（2 巡目。「実装
インスタンスを返さない」だけでは型の到達可能性を防げない）。(3) 本節が
確定する設計は、**低水準 trait（`DeviceAllocator`）自体を廃止**し、
`tensor-core` にはハンドルの内部表現を一切含まない POD 型
（`PoolConfig`／`PoolStats`）とハンドル型非依存の generic なプール
実装（`SizeClassPool<H>`）のみを置く。加えて REQ-14（明示解放 API の
提供義務）を満たすため、公開 `BackendOps` trait に unit／POD のみを返す
2 メソッドを追加し、`facade` から薄く再公開する（後述）。

- **`tensor-core` に置く公開面（`crates/tensor-core/src/pool_core.rs`。
  新規モジュール。既存 `pool.rs`〈`PooledMemory`〉とは別モジュール）**:
  - `pub struct PoolConfig { .. }`（サイズクラス境界・総量上限
    `max_pool_bytes`・クラス別アイドル上限等の設定値。§3.2 の帯・丸め
    粒度に対応するフィールドを持つ）。POD（`Copy`・`Debug`・`PartialEq`）。
  - `pub struct PoolStats { .. }`（確保回数〈`alloc_count`〉・再利用
    回数〈`reuse_count`〉・解放回数・解放済みバイト数〈`released_bytes`〉・
    現在キャッシュ済みバイト数〈`cached_bytes`〉・返却待ちバイト数
    〈`pending_return_bytes`。Cursor Bugbot 指摘対応。§3.3「Metal」の
    `pending_pool_returns` を可視化するための新設フィールド。詳細は
    下記「フィールド更新契約」〉・`capacity_waste_bytes`〈§3.4 の内部
    断片化可視化〉等の数値のみ。ハンドル・ポインタ・trait object を
    一切含まない）。POD。各フィールドがどの操作でどう増減するかは
    下記「`PoolStats` フィールド更新契約」（codex-review PR #1056 P1
    是正）を正とする。
  - `pub struct SizeClassPool<H> { .. }`（`H` を型パラメータに取る
    ハンドル型非依存のフリーリスト・サイズクラス丸め・統計更新ロジック。
    **シグネチャに `BufferHandle`／`DeviceBuffer` を一切含めない**。`H`
    に trait 境界を課さない〈`H: Send` のみ。理由は §3.5〉ため、`H` が
    何であるかを `tensor-core` 側は一切知らない。`H` は各バックエンドの
    **生ハンドル型**（例: `CudaSliceHandle`。下記「各バックエンドクレート
    内に閉じる `pub(crate)` 面」参照）であり、
    貸出中の RAII ラッパー型（`PooledCudaHandle` 等。下記 `pub(crate)`
    面）とは別物である。

    **不変条件（codex-review PR #1056 P1 是正。所有権遷移の一意性）**:
    同一ハンドル（`H` の同一値）はフリーリスト（`SizeClassPool<H>` の
    内部状態）と貸出中（呼び出し元が保持する RAII ラッパー内の
    `Option<H>`）のいずれか一方にのみ存在し、両方に同時に存在すること
    はない。したがって「新規確保したハンドルをフリーリストへ記帳する」
    という操作（旧稿の `register`）は設けない。**新規確保直後のハンドルの
    所有権は最初から貸出中として扱う**（RAII ラッパーが排他的に所有し、
    `SizeClassPool<H>` はハンドルそのものの custody を一切持たない）。
    フリーリストへの遷移はラッパーの `Drop` 時にのみ発生する。ただし
    `SizeClassPool<H>` は**統計（`PoolStats`）に限っては**新規確保の
    発生を関知する（下記 `record_allocation`。ハンドルの custody を
    持つことと統計を更新することは別であり、後者はハンドルを一切
    経由しない`u64` 引数のみの呼び出しで実現できるため上記の不変条件と
    矛盾しない。codex-review PR #1056 P1 再指摘対応。旧稿は統計の更新
    経路が定義されておらず、新規確保の回数・内部断片化を `PoolStats`
    へ反映する手段が存在しなかった）。API 案:

    - `fn new(config: PoolConfig) -> Self`
    - `fn take(&self, bytes: u64) -> Option<H>`（フリーリストから適合
      クラスの空きハンドルを取り出す。**取り出した時点で所有権は完全に
      呼び出し元へ移り、フリーリストからは消える**。無ければ `None`
      〈呼び出し元が新規確保 FFI を呼ぶ〉。新規確保したハンドルは
      `take` を経由せず直接 RAII ラッパーで包む〈`take` は「既存の
      アイドルハンドルの再利用」専用〉。成功時は `bytes`〈今回の
      論理バイト数〉と適合クラス `class_bytes`〈§3.2 の丸め規則で
      `SizeClassPool<H>` が内部計算する〉から `PoolStats` を更新する。
      下記「フィールド更新契約」参照）。
    - `fn record_allocation(&self, logical_bytes: u64, class_bytes: u64)`
      （新規。codex-review PR #1056 P1 是正）: **ハンドルを一切受け取ら
      ない統計専用メソッド**。フリーリストが空で新規確保 FFI を呼んだ
      直後（`take` が `None` を返した場合の後続処理として）、具体
      アロケータが呼ぶ。`class_bytes` は呼び出し元（具体アロケータ）が
      §3.2 の丸め規則で自ら計算した値を渡す（`take` 内部の丸め計算と
      同じ規則だが、新規確保 FFI 自体〈確保サイズの指定〉に
      `class_bytes` が必要なため呼び出し元が既に計算済みであり、二重
      計算を避けるためここでは `SizeClassPool<H>` 側で再計算しない。
      両者が同じ §3.2 規則を参照することは実装時の回帰テストで担保する。
      §6.2）。`PoolStats` を更新するのみで、フリーリスト・ハンドルの
      custody には一切影響しない。
    - `fn put(&self, class_bytes: u64, handle: H)`（貸出中のハンドルを
      フリーリストへ返却する。旧稿の `register` の後継だが意味論が
      異なる: **RAII ラッパーの `Drop` からのみ呼ばれる**〈返却であり
      記帳ではない〉。新規確保直後には呼ばない。フリーリストへの記帳
      〈`cached_bytes += class_bytes`〉のみを行う純粋な操作であり、
      「今回の貸出がどれだけ内部断片化したか」〈`capacity_waste_bytes`
      の増減〉は関知しない。理由は本メソッドが 2 つの異なる文脈
      〈(a) RAII ラッパーの `Drop`〈貸出の終了〉・(b) `release_cached()`
      の個別解放失敗時の再挿入〈§3.6 (2) フェーズ (ii)。この場合その
      ハンドルは今回一度も貸し出されておらずアイドルのまま
      `take_one_for_release` に取り出されただけなので、貸出終了に伴う
      断片化の増減という概念自体が存在しない〉から呼ばれ、後者には
      `logical_bytes` という概念が存在しないため。(a) の断片化会計は
      下記 `record_loan_end` を **`Drop` の時点で必ず呼ぶ**ことで実現する
      （`put` 自体のタイミングとは独立。次項参照）。
    - `fn record_loan_end(&self, logical_bytes: u64, class_bytes: u64)`
      （新規。`record_allocation` と対になる統計専用メソッド。
      **ハンドルを一切受け取らない**）: RAII ラッパーの `Drop` が呼ぶ
      （`release_cached` の再挿入からは呼ばない。上記 `put` の注記
      参照）。**`put` 自体の呼び出しタイミングとは独立に、`Drop` の
      時点で常に即座に呼ぶ**（CUDA・Metal 共通。`capacity_waste_bytes`
      は「貸出中かどうか」の会計であり、`Drop` した時点で Rust 上の
      貸出（RAII ラッパーの生存期間）は終了しているため、GPU 側の
      実行完了を待つ必要がない。Metal の `pending_pool_returns`〈§3.3
      「Metal」〉によって `put` 自体〈フリーリストへの実際の記帳・
      `cached_bytes` の更新〉が `synchronize()` まで遅延される場合でも、
      `record_loan_end` はこの遅延に追従しない。§3.3「Metal」の
      `pending_pool_returns` は `(class_bytes, handle, pool)` の
      3 要素のみを保持すればよく〈`logical_bytes` を運ぶ必要がない〉
      のはこのためである）。`PoolStats` を更新するのみ。下記
      「フィールド更新契約」参照。
    - `fn take_one_for_release(&self) -> Option<(u64, H)>`（解放処理
      専用。フリーリストから 1 エントリ〈`(class_bytes, handle)`〉だけ
      取り出す。§3.6 の解放トランザクションで使う。**一括 `drain()` は
      設けない**〈part failure 時にキャッシュ全損させないため。理由は
      下記「解放時の所有権遷移」〉）。
    - `fn record_release(&self, class_bytes: u64)`（新規。統計専用。
      **ハンドルを一切受け取らない**）: `release_cached()`〈内部
      メソッド〉のフェーズ (ii) で `take_one_for_release` により取り
      出したハンドルの個別解放が成功するたびに呼ぶ（`cached_bytes` は
      `take_one_for_release` 自体が既に減算済みのため、本メソッドは
      `released_bytes` の加算のみを行う）。下記「フィールド更新契約」
      参照。
    - `fn record_pending_return(&self, class_bytes: u64)`（新規。Metal
      専用の統計専用メソッド。**ハンドルを一切受け取らない**。
      **`SizeClassPool<H>` 内部の `Mutex<PoolCore<H>>` を一切取らない
      lock-free メソッド**〈`AtomicU64::fetch_add`。§3.5「Metal
      `pending_pool_returns` の排他制御」参照。Cursor Bugbot・
      codex-review 再指摘対応。旧稿は `BatchSlots` のロックを解放した
      **後**に呼ぶ契約としていたが、これは `Drop` の push と本メソッド
      呼び出しの間に別スレッドの `synchronize()` が割り込む競合を生み、
      `pending_return_bytes` が恒久的にずれ得た。本節が確定する契約は
      `PooledMetalHandle::Drop` が `pending_pool_returns` への push と
      **同一の `BatchSlots` クリティカルセクション内**で本メソッドを
      呼ぶことである〉）: `pending_return_bytes` を `+class_bytes`
      する。
    - `fn record_pending_merge(&self, class_bytes: u64)`（新規。Metal
      専用の統計専用メソッド。**ハンドルを一切受け取らない**。
      `record_pending_return` と同じく **lock-free**〈`AtomicU64::
      fetch_sub`〉。`MetalContext::synchronize()` が `pending_pool_
      returns` から `std::mem::take` でエントリを取り出すのと**同一の
      `BatchSlots` クリティカルセクション内**で、取り出した各エントリに
      ついて呼ぶ（`put` 自体〈`SizeClassPool<H>` の
      `Mutex<PoolCore<H>>` を要する〉は `BatchSlots` のロックを解放
      した**後**に呼ぶ。§3.3「Metal」・§3.5 参照）: `pending_return_
      bytes` を `−class_bytes` する。
    - `fn stats(&self) -> PoolStats`／`fn config(&self) -> PoolConfig`
      （`PoolStats` は `Mutex<PoolCore<H>>` が保護するフィールド
      〈`alloc_count` 等〉と `pending_return_bytes`〈`AtomicU64`〉を
      合成して返す。両者を単一のロックで同時に読むわけではないため、
      **`stats()` 単発の呼び出し内で全フィールドが厳密に同一時刻の
      値であることは要求しない**〈診断用スナップショットとしての利用に
      留め、フィールド間の整合性を前提にした判定は行わない〉）。

    **統計専用メソッドの検証**: `record_allocation`／`record_loan_end`／
    `record_release`（`Mutex<PoolCore<H>>` で保護される群）と
    `record_pending_return`／`record_pending_merge`（`AtomicU64` による
    lock-free な群）とで検証方針が異なる:
    - **`Mutex<PoolCore<H>>` 群**: 実装バグにより減算対象のフィールドが
      `0` を下回る呼び出しが発生した場合は、`debug_assert!`（デバッグ・
      テストビルドで即座に検知する）に加え、リリースビルドでは
      `saturating_sub` により `0` に飽和させる（本番経路で `panic!`
      しない。`.claude/rules/coding-rust.md`）。この飽和は「バグを
      黙認する」ものではなく `debug_assert!` がテストで先に検知する
      前提の**多層防御**であり、統計フィールドの意味論的な正しさを
      設計上保証するものではない（あくまで診断用フィールドの表示が
      負値になって混乱を招くことを防ぐための保険）。
    - **`AtomicU64` 群（`pending_return_bytes`）**: `fetch_add`／
      `fetch_sub` の memory ordering は **`Ordering::Relaxed` で
      十分**とする（`pending_return_bytes` 自体の増減以外に、この
      atomic 操作を起点として他のメモリ位置の可視性を保証する必要が
      ない診断用カウンタであるため。`pending_pool_returns` 本体の
      custody・順序保証は `BatchSlots` の `Mutex` が別途担う）。
      **値が負になる（`u64` の下限を割り込む）遷移は設計上起こらない**:
      `record_pending_return` は push と、`record_pending_merge` は
      `take` と、それぞれ同一の `BatchSlots` クリティカルセクション内で
      対になって呼ばれる契約（下記・§3.3「Metal」）であるため、ある
      `class_bytes` 分の減算が対応する加算より先に起こることは構造的に
      起こり得ない。したがってこの 2 メソッドには `Mutex<PoolCore<H>>`
      群と同じ `debug_assert!`／`saturating_sub` の防御は適用しない
      （防ぐべき異常系がそもそも構造的に存在しないため。防御コードの
      不在自体が設計判断であることを明記する）。

    **`PoolStats` フィールド更新契約（codex-review PR #1056 P1 是正。
    どの操作でどう増減するかを一意に定める）**:

    | フィールド | 増加 | 減少 | 備考 |
    |---|---|---|---|
    | `alloc_count` | `record_allocation` 呼び出しごとに `+1` | なし（単調増加のカウンタ） | 新規物理確保の回数 |
    | `reuse_count` | `take` 成功（`Some` を返した）ごとに `+1` | なし（単調増加のカウンタ） | フリーリストからの再利用ヒット回数 |
    | `cached_bytes` | `put` で `+class_bytes` | `take` 成功で `−class_bytes`（取り出した分）／`take_one_for_release` で `−class_bytes`（取り出した分） | フリーリストが現在保持する総バイト数。恒等式 `cached_bytes == Σ(フリーリスト中の各エントリの class_bytes)` を常に満たす |
    | `pending_return_bytes`（`AtomicU64`。`Mutex<PoolCore<H>>` の外） | `record_pending_return(class_bytes)` 呼び出しごとに `+class_bytes`（`fetch_add`／`Relaxed`。`PooledMetalHandle::Drop` が `pending_pool_returns` へ push するのと**同一の `BatchSlots` クリティカルセクション内**で呼ぶ。§3.3「Metal」・§3.5） | `record_pending_merge(class_bytes)` 呼び出しごとに `−class_bytes`（`fetch_sub`／`Relaxed`。`MetalContext::synchronize()` が `pending_pool_returns` から `std::mem::take` するのと**同一の `BatchSlots` クリティカルセクション内**で呼ぶ。`record_loan_end` は既に `Drop` 時点で呼び出し済みのためここでは呼ばない。下記「Metal の返却待ちバイト数」参照） | CUDA・CPU では常に `0`（`pending_pool_returns` 相当の機構を持たないため呼ばれない）。push／`take` と加減算が同一クリティカルセクションで対になるため（Cursor Bugbot・codex-review 再指摘対応。旧稿は `BatchSlots` ロック解放後に加減算する契約だったため、別スレッドの `synchronize()` が先着すると `record_pending_merge` の減算が `record_pending_return` の加算より先に走り、`saturating_sub` で `0` に張り付いた後の加算が恒久的な誤差として残り得た）、`pending_return_bytes` に恒久的な誤差は生じない。ただし `record_pending_merge`〈ロック内〉と `put`〈ロック解放後〉の間の短い窓では、当該バイト数は `pending_return_bytes` にも `cached_bytes` にも含まれない中間状態になる（二重計上にはならず、`max_pool_bytes`／LRU の判定〈§3.4〉は実態より少なく見える保守側にのみ振れる） |
    | `capacity_waste_bytes` | `record_allocation` で `+(class_bytes − logical_bytes)`／`take` 成功で `+(class_bytes − bytes)`〈`bytes` はその回の `take` 呼び出しの論理バイト数〉 | 対応する貸出が終了する時（`record_loan_end(logical_bytes, class_bytes)` 呼び出しごとに `−(class_bytes − logical_bytes)`。**この `logical_bytes` はその貸出が `record_allocation` または `take` で開始した時点の値と同一でなければならない**〈RAII ラッパーが自身の `logical_bytes` フィールドに保持し続けることで担保する〉 | **意味論はストック量**（現在貸出中の全ハンドルについて `Σ(class_bytes − logical_bytes)` を表すリアルタイムの内部断片化実測値であり、`capacity_waste_bytes` 単体では「これまでの累積無駄」ではない）。`release_cached()` の再挿入（`put` のみ・`record_loan_end` を伴わない）はこのフィールドに影響しない（対象エントリは貸出中ではなくアイドルだったため） |
    | `released_bytes` | `record_release(class_bytes)` 呼び出しごとに `+class_bytes`（`release_cached()`〈内部メソッド。§3.1「2 段構成の命名規約」〉のフェーズ (ii) で個別解放が成功するたびに呼ぶ） | なし（単調増加のカウンタ） | §3.6 (2) 参照。プロセス起動からの累積解放バイト数（診断用） |

    **Metal の返却待ちバイト数（`pending_return_bytes`）と
    `capacity_waste_bytes` の関係**: `pending_pool_returns` へ委譲された
    時点（`PooledMetalHandle::Drop`）では、貸出は実質的に終了している
    ため `capacity_waste_bytes` は `record_loan_end` により**その時点で**
    減算する（`cached_bytes` への計上とは切り離す）。`pending_return_bytes`
    は「フリーリストへはまだ合流していないが GPU 完了待ちで既に貸出は
    終わっているバイト数」を表す独立した勘定であり、`capacity_waste_bytes`
    （貸出**中**のみを対象とする）とは対象期間が重複しない。

    **解放時の所有権遷移（codex-review PR #1056 P1 是正。トランザクション
    型 API）**: 旧稿の一括 `drain(&self) -> Vec<H>` は、呼び出し後は
    どのハンドルも `SizeClassPool<H>` の管理外になるため、§3.6 が要求
    する「stream 同期・driver トリム失敗時に未解放エントリをキャッシュへ
    残す」という fail-closed 契約を実現できない（一度 `drain` した時点で
    キャッシュは空になってしまい、失敗時に戻す先がない）。本設計は
    `take_one_for_release` による 1 件ずつの取り出しへ変更する。呼び出し
    元（各バックエンドの `pub(crate)` アロケータの内部 `release_cached()`
    実装。§3.1「2 段構成の命名規約」）が踏む手順（stream 同期 2 回・
    個別解放・driver トリムの 4 フェーズ）の詳細契約・各フェーズの
    `Err` 種別・状態一貫性は **§3.6 (2) を正とし、ここでは重複記載
    しない**（Cursor Bugbot 指摘対応で 3 → 4 フェーズへ改訂した際に
    本節と §3.6 (2) の二重管理が生じないようにするため）。
  - `SizeClassPool<H>: Send + Sync where H: Send`（`Mutex<PoolCore<H>>`
    で保護する内部実装。§3.5 参照）。
- **各バックエンドクレート内に閉じる `pub(crate)` 面（`backend-cuda`・
  `backend-metal` それぞれ独立に実装。`tensor-core` からは不可視）**:
  - **生ハンドル型（`H` を埋める具体型）**: 例
    `backend-cuda::pool::CudaSliceHandle`〈内部に `CudaSlice` を保持〉・
    `backend-metal::pool::MetalBufferHandle`〈内部に
    `Retained<ProtocolObject<dyn MTLBuffer>>` を保持〉。
    `SizeClassPool<H>` のフリーリストに格納される値そのものであり、
    `tensor-core` のどの公開シグネチャにも現れない。
  - **RAII 貸出ラッパー型（新設。codex-review PR #1056 P1 是正）**: 例
    `backend-cuda::pool::PooledCudaHandle { handle: Option<CudaSliceHandle>,
    class_bytes: u64, logical_bytes: u64, pool: Arc<SizeClassPool<
    CudaSliceHandle>> }`（`backend-metal::pool::PooledMetalHandle` も
    同型）。`logical_bytes`（新規。codex-review PR #1056 P1 是正）は
    `alloc_zeroed`／`alloc_uninit`（下記）呼び出し時に要求された論理
    バイト数（`class_bytes` への丸め前の値）を保持し、`Drop` 時の
    `record_loan_end(logical_bytes, class_bytes)` 呼び出し（下記）に
    使う。`alloc_zeroed`／`alloc_uninit`（下記）は生ハンドル型ではなく
    **この RAII ラッパーを返す**。二重返却・use-after-return の構造的
    防止は**両バックエンド共通で `Option<H>::take()` 方式に統一する**
    （codex-review PR #1056 P2 是正。旧稿は本節で `Option<H>::take()`、
    §3.3・§7 の一部記述で既存 `PooledBufferHandle` の `ManuallyDrop`
    ガード方式と、2 つの異なる二重防止方式が併記されており矛盾して
    いた。`ManuallyDrop` 方式への言及は削除し本節の記述を正とする）:
    `Drop` 実装は必ず `self.handle.take()` で `Option` から所有権を
    奪ってから後段の処理へ渡す（`take()` 後の `self.handle` は `None`
    になるため、何らかの経路で二重に `Drop` が走っても 2 回目は `None`
    を握って何もしない）。所有権を奪った直後、**`pool.put()` の
    呼び出しタイミングとは独立に**
    `pool.record_loan_end(self.logical_bytes, self.class_bytes)`
    を CUDA・Metal 共通で即座に呼ぶ（`PoolStats::capacity_waste_bytes`
    の会計。上記「フィールド更新契約」・下記「Metal の返却待ちバイト数」
    参照）。その後、`pool.put()` を呼ぶタイミングは CUDA と Metal で
    異なる（§3.3「返却の GPU 完了待ち契約」参照）:
    - **CUDA**: `record_loan_end` の直後に
      `pool.put(self.class_bytes, handle)` を呼ぶ（即時返却）。安全性の
      根拠は §3.3「CUDA」に既述のストリーム順序保証（同一 stream 上の
      後続再貸出しは前段の完了後にのみ実行される）であり、本節の
      変更なし。
    - **Metal**: `record_loan_end` の直後に**無条件で** `pool.put()` を
      呼んでよいわけではない（codex-review PR #1056 P1 是正。詳細契約は
      §3.3「Metal」参照）。ホットパス（GEMM 等）はこのラッパーが scope
      を抜けるまで生ハンドルを排他的に保持し続けるため、上記「不変
      条件」（フリーリストと貸出中の排他性）自体はいずれの経路でも
      保たれる。
  - 具体アロケータ型（例: `backend-cuda::pool::CudaAllocator`）が
    `SizeClassPool<H>` を保持し、`alloc_zeroed`／`alloc_uninit`（実際の
    確保 FFI・ゼロ初期化を行い `PooledCudaHandle` を返す）・内部
    `release_cached()`（driver トリム・stream 同期を含む実際の FFI。
    上記「解放時の所有権遷移」の手順を実装する）を実装する。いずれも
    `pub(crate)` 固有メソッドであり、`tensor-core` のどの trait にも
    属さない（sealed ではなく、そもそも `tensor-core` 側に対応する
    trait 自体が存在しない）。
  - ホットパス（§2 事実 3・4。`backend-cuda/src/{gemm,elementwise,
    softmax}.rs`・`backend-metal/src/{gemm,elementwise}.rs`）は同一
    クレート内で `CudaAllocator`／`MetalAllocator` インスタンスへ直接
    アクセスし `alloc_zeroed`／`alloc_uninit` を呼ぶ（`BackendOps` を
    経由しない。呼び出し箇所はいずれも各バックエンド自身の内部実装で
    あり、他クレートからの汎用呼び出しを必要としないため）。呼び出し元
    は返却された `PooledCudaHandle`／`PooledMetalHandle` を演算の間
    保持し、演算完了後（既存 `download` 契約の同期点を経てから）通常の
    スコープ終了で `Drop` させる（明示 `free()` は設けない。§3.3
    「返却は RAII のみ」を維持）。
  - プールは device 単位のプロセスワイド singleton とする
    （`static_cuda_memory`／同等の `Box::leak` 所有モデル。§2 事実 2 の
    計測系列単一化と整合させる）。CPU バックエンドは本イシューの
    対象外（別イシュー #1026 が担当）。

- **REQ-14 の到達経路（公開 `BackendOps` への追加。codex-review PR
  #1056 P1「利用者から到達できない」是正）**: 上記のとおり低水準
  アロケータ・ハンドル型は一切公開しないが、REQ-14 は「プール等の
  キャッシュ機構を導入する場合、係数上限を維持できなければプール解放
  API を提供する」ことを求めており、内部 OOM フォールバックから呼べる
  だけでは満たさない（`facade` を含む利用者側から到達できる必要がある）。
  `memory_ops()`（既存）と同型の非破壊拡張パターンで、`BackendOps` に
  戻り値が unit／POD のみの 2 メソッドを追加する:

```rust
pub trait BackendOps {
    // ...（既存メソッド）

    /// REQ-14 の明示解放 API。このバックエンドのデバイスメモリプールが
    /// アイドル保持しているバッファを全て解放する。CUDA で
    /// `has_async_alloc()` が真の環境では、自作プール層の解放に加え
    /// driver 側 memory pool のトリム（`cuMemPoolTrimTo(0)` 相当）・
    /// 2 回の対象 stream 同期を内部で行う（§3.6 (2) の 4 フェーズ）。
    /// Metal は driver トリムを持たないためフェーズが少ない
    /// （§3.6 (2)「バックエンド別の該当フェーズ」表参照）。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は `Ok(())`（プールを持たないバックエンドは解放対象なし。
    /// fail-open ではなく「対象が存在しないため自明に成功」という
    /// 意味）。CPU バックエンドは常にこのデフォルトのまま（本イシューの
    /// 対象外。#1026）。CUDA／Metal は §3.6 (2) の契約で実カーネルへ
    /// オーバーライドする。
    ///
    /// # エラー（codex-review PR #1056 P1 是正。旧稿は 2 種類に限定して
    /// おり §3.6 (2) の 4 フェーズ設計と矛盾していた）
    /// `Err` は §3.6 (2)「バックエンド別の該当フェーズ」表が定める
    /// フェーズのいずれかの失敗を表す。**実際に到達しうる `Err` の
    /// 種別・個数はバックエンドごとに異なり**（例: Metal はフェーズ
    /// (ii) が失敗しない設計のため実質的にフェーズ (i) 失敗の 1 種類
    /// のみへ到達しうる。CPU は本メソッドを常にデフォルト実装のまま
    /// 使うため到達しない）、本 doc comment では数を明記しない
    /// （codex-review PR #1056 P2 是正。旧稿はここでバックエンド別の
    /// 種類数を書き下し、§3.6 (2) の正本表と数が食い違っていた。数を
    /// 書かず正本表を参照する形にすることで今後の齟齬を構造的に防ぐ）。
    /// 黙殺・panic は禁止
    /// （fail-closed。`.claude/rules/coding-rust.md`）。**フェーズごとに
    /// 失敗時のプール内部状態が異なる**（あるフェーズでは未解放分が
    /// フリーリストへ再挿入されて残り、別のフェーズでは解放自体は
    /// 既に完了しておりトリム等の後続処理のみが未了となる）。呼び出し
    /// 元がこの違いを一律に「フリーリストへ戻る」と誤認しないよう、
    /// 各フェーズの失敗時状態は §3.6 (2)「バックエンド別の該当フェーズ」
    /// 表を正とする（本 doc comment では重複記載しない）。
    fn release_cached_device_memory(&self) -> Result<(), BackendError> {
        Ok(())
    }

    /// デバイスメモリプールの統計スナップショット（診断用）。
    /// `PoolStats`（POD。内部ハンドル表現を一切含まない）のみを返す。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は `None`（プールを持たないバックエンド）。
    fn device_memory_pool_stats(&self) -> Option<PoolStats> {
        None
    }
}
```

  - **2 段構成の命名規約（narrative 中の「`release_cached()`」の指示先を
    一意にする）**: 各バックエンドの `pub(crate)` 具体アロケータ型は
    内部実装として `pub(crate) fn release_cached(&self) -> Result<u64,
    BackendError>`（フリーリスト解放バイト数を返す。driver トリム・
    stream 同期を含む。§3.6 (2)）を持つ。公開 `BackendOps::
    release_cached_device_memory()` はこれを呼び出し、`Ok` 時はバイト数を
    捨てて `Ok(())` へ、`Err` はそのまま伝播する（バイト数は
    `device_memory_pool_stats()` が返す `PoolStats::cached_bytes`〈解放後
    は 0〉から確認できるため、公開 API 側では返す必要がない）。以降
    §3.3・§3.4・§3.6 の本文中で単に「`release_cached()`」と書く箇所は
    この内部メソッドを指し、利用者から到達可能な公開 API を指す場合は
    明示的に「`BackendOps::release_cached_device_memory()`」
    （facade 経由では `release_cached_memory()`）と書く。

  - **`facade` からの再公開（確定入口。`docs/compat-api-scope.md` §0 の
    確定入口一覧への追記は実装イシュー〈#1020／#1021〉側で行う。§6.1
    「ドキュメント追従」参照）**: `facade::Device` は識別子 enum（`Cpu`／`Cuda(ordinal)`／
    `Metal`）であり `tensor-core` 由来の**外部型**のため、facade は
    `Device` へ inherent メソッドを追加できない（Rust の orphan rule。
    `docs/facade-device-handle-design.md` が確定した「案 B のみ採用
    （新規ハンドル型を作らずバックエンド内部常駐化で解決する）」方針とも
    整合する）。したがって `tape`／`tape_for` と同型の**自由関数**として
    公開する:

    ```rust
    /// REQ-14 の明示解放 API。`device` に対応するバックエンドのデバイス
    /// メモリプールを全て解放する。プールを持たないバックエンド（CPU
    /// 等）は何もせず `Ok(())` を返す。
    pub fn release_cached_memory(device: Device) -> Result<(), BackendError> {
        resolve_ops(device)?.release_cached_device_memory()
    }

    /// デバイスメモリプールの統計スナップショット（診断用。POD
    /// `PoolStats` のみを返し、内部ハンドル表現は含まない）。
    pub fn memory_pool_stats(device: Device) -> Result<Option<PoolStats>, BackendError> {
        Ok(resolve_ops(device)?.device_memory_pool_stats())
    }
    ```

    `PoolStats` は `crate::{AutodiffError, Tensor}` 等と同じ「迂回経路を
    持たない値型」として `pub use fandhe_ai_tensor_core::PoolStats;` を
    facade クレート root へ追加する（`tests/api_surface.rs` が `pub use`
    を行単位で走査する制約〈`crates/facade/src/lib.rs` コメント〉に従い
    1 文 1 行を維持する）。`resolve_ops` は毎回新しい `Box<dyn
    BackendOps>` を構築する軽量値だが、実際のプール状態は §3.5 の
    プロセスワイド singleton が保持するため、呼び出しごとに新しい
    `BackendOps` インスタンスを介しても解放・統計は正しく機能する
    （`memory_ops()`／`CudaBackendOps` の `context_cache` 経由アクセスと
    同型の前提。`docs/facade-device-handle-design.md` §2.2）。

- **capacity と論理長の分離**: `DeviceBuffer` の `shape`（論理 numel）と、
  ハンドル内部の `capacity_bytes`（サイズクラス丸め後の実確保量。
  バックエンド内部のハンドル型が保持し `tensor-core` 側は関知しない）を
  分離する契約を導入する。`download` 側・カーネル側は常に論理長のみを読む
  （CUDA は `CudaSlice` の論理長ビュー相当のスライシング、Metal は
  `MTLBuffer::length()` ではなくディスクリプタに保持した論理長引数を使う）。
  これは §2 事実 5 の「完全一致」制約を解除する唯一の前提であり、本設計が
  確定させる中核の変更点である。
- 既存 `PooledMemory`（`crates/tensor-core/src/pool.rs`）との関係: 廃止・
  削除はしない。`SizeClassPool<H>` とは別モジュール（`pool_core.rs`）で
  共存させる。`PooledMemory` を新設計上の薄い互換アダプタとして残すか、
  実装後に非推奨化するかは実装イシュー（#1020）の判断に委ねる。本文書が
  確定するのは「既存公開 API（`PooledMemory`・`PoolZeroFill`）を壊さない」
  という制約のみ。
- ホットパスへの接続点（#1020／#1021 の作業対象。§2 事実 3・4 の箇所）:
  `backend-cuda/src/gemm.rs`・`elementwise.rs`・`softmax.rs` の
  `alloc_zeros` 呼び出しと、`backend-metal/src/gemm.rs`・`elementwise.rs`
  の `new_zeroed` 呼び出しを、いずれも各クレート内部の具体アロケータ型
  （`CudaAllocator`／`MetalAllocator`）への直接アクセサ経由（`BackendOps`
  を経由しない）へ置き換える: ゼロ初期化が必要な箇所は
  `alloc_zeroed`、カーネルが全要素を書き切る出力専用の箇所は
  `pub(crate) alloc_uninit` を呼ぶ。入力側のアップロード経路
  （`clone_htod`・`new_with_data`）は本イシューの対象外とする
  （`upload_into` の同期契約が前提であり、forward 常駐化〈別イシュー〉が
  進むことで直接確保の頻度は自然に減っていく）。

### 3.2 サイズクラス表（受け入れ条件）

jemalloc 型の方式を採用する: 2 の冪ごとの区間（オクターブ）を 4 段
（`1x`／`1.25x`／`1.5x`／`1.75x`）に分割し、切り上げ先の最小クラスへ丸める。

| 帯 | 範囲 | 丸め粒度 |
|---|---|---|
| 最小 | `bytes == 0` | プール非経由（従来どおり空ハンドルを即返す） |
| 極小 | 1 B 〜 255 B | 256 B（小帯の最小クラス）へ切り上げてプール経由とする（小帯と同一のフリーリストを使う。専用クラスは設けない） |
| 小 | 256 B 〜 1 MiB | 各オクターブを `1x`／`1.25x`／`1.5x`／`1.75x` の 4 段に分割（内部断片化の理論上限 25%） |
| 大 | 1 MiB 超 〜 64 MiB 未満 | 2 MiB 単位切り上げ（exclusive プール。1 バッファ 1 エントリ） |
| 巨大 | 64 MiB 以上 | 完全一致のみ・保持上限 1 エントリ／クラス |

空判定は要素数ではなくバイト数（`bytes == 0`）で行う（`shape` に 0 を含む
形状は要素数 0 でもバイト数 0 になるため実質的に等価だが、以降の丸め計算が
すべてバイト数ベースであることと表記を揃える）。1〜255 B の非空確保は
256 B の最小クラスへ切り上げることで小帯の丸め規則へ合流させ、専用の帯・
専用のフリーリストは設けない（分岐を増やさず内部断片化の上限も 256 B
以下に収まるため）。上記の閾値（1 MiB／2 MiB／64 MiB）は**案**であり、
#1010 の内訳実測後に見直す。`max_pool_bytes` 超の単一バッファはプールに
入れず即解放する（既存 `PooledMemory` の方針を踏襲）。64 MiB ちょうどは
「巨大」帯（64 MiB 以上）に属する（下記の GEMM 4096² 例を参照）。

代表ワークロード（§2 事実 10）の写像（各行は実際に計算した値。検算方法は
§5 検証方法を参照）:

| テンソル | 論理バイト数 | 丸め先クラス | 帯 |
|---|---|---|---|
| `x`（64×784, f32） | 200,704 B | 229,376 B（= 1.75 × 2^17） | 小 |
| `W1`（784×256, f32） | 802,816 B | 917,504 B（= 1.75 × 2^19） | 小 |
| `h`（64×256, f32） | 65,536 B | 65,536 B（= 1 × 2^16。既に丁度境界値） | 小 |
| `W2`（256×10, f32） | 10,240 B | 10,240 B（= 1.25 × 2^13） | 小 |
| `out`（64×10, f32） | 2,560 B | 2,560 B（= 1.25 × 2^11） | 小 |
| GEMM 4096²（f32） | 67,108,864 B（64 MiB） | 67,108,864 B（完全一致） | 巨大 |

MLP 学習の全テンソルは小クラス帯（≤ 1 MiB）に収まり、GEMM 4096² は巨大帯
（完全一致・1 エントリ保持）に該当することが確認できる。丸め計算は
`checked_mul`／`checked_add`（`u64`）で行い、オーバーフロー時は
`BackendError::DeviceAllocationFailed` を返す（既存
`pool.rs::checked_byte_len` の方針を踏襲）。

### 3.3 寿命・順序契約（stream / command buffer）

- 返却は RAII のみ（`Drop` でプールへ戻す）。明示 `free()` は設けない
  （`crates/tensor-core/src/buffer.rs`「解放方針」節を維持）。
- **返却の GPU 完了待ち契約（codex-review PR #1056 P1 是正。CUDA／
  Metal で異なる理由）**: §3.1「RAII 貸出ラッパー型」の `Drop` は
  `Option<H>::take()` で所有権を取り出した後、CUDA は即座に
  `pool.put()` する一方、Metal は GPU 完了まで `put()` を遅延させる
  （下記「Metal」参照）。この非対称の根拠を以下に明記する:
  - **CUDA が即時 `put()` で安全な理由**: 本設計は単一ストリームモデル
    （下記）を前提とし、`take()` で再取得したハンドルへの新規データ
    書き込みは常に `clone_htod`（`cuMemcpyHtoDAsync` 相当）による
    **同一 stream 上の非同期コピー**として発行される。CUDA の stream は
    in-order 実行契約を持つため、この新規コピーは同一 stream 上の
    先行 dispatch（返却前にそのバッファを使っていたカーネル・
    `cuMemFreeAsync` 相当の解放操作を含む）が完了した**後にのみ**
    実行されることが driver によって保証される（`cuMemAllocAsync`／
    `cuMemFreeAsync` の stream-ordered アロケータが「同一 stream 上の
    再利用は安全」を保証する契約と同型。§2 事実 6）。したがって
    ホスト側の `SizeClassPool<H>`（`take`／`put` の記帳）が GPU の完了を
    待たずに再貸出ししても、実際のデータ競合は stream の順序保証に
    よって発生しない。**複数 stream には非対応**（下記「単一ストリーム
    モデル」）: プールを `(ordinal, stream)` 単位に持ち別ストリームへの
    貸し出しを禁止しているのは、上記の安全性根拠が「同一 stream 上」
    という前提に依存するためであり、複数 stream 対応（#1012 の設計が
    確定した場合の将来対応）にはイベント fence 等の追加の順序保証機構
    が必要になる（下記参照）。
  - **Metal が即時 `put()` では安全でない理由**: #1054 §3.4 は
    「個別バッファ単位の依存追跡は行わない」「過剰同期側（安全側）の
    設計に倒す」という方針を確定済みであり、CUDA のような per-op の
    stream 順序保証に依拠した安全性論法を Metal には採用しない
    （バッファ再利用時のホスト側書き込み〈`new_with_data`〉・ゼロ初期化
    〈`zero_fill`〉のいずれも、in-flight のコマンドバッファに対して
    無条件に `synchronize()` を経由することを要求する契約が既に
    確定している。#1054 §3.4「契約」）。この確定済み契約と、RAII
    ラッパーの `Drop` で即座に `pool.put()` してしまう素朴な実装は
    矛盾する（`Drop` した直後にフリーリストへ戻り、`synchronize()` 前
    に別演算へ `take()` されうるため）。詳細な解決策は下記「Metal」を
    参照。
- **CUDA・単一ストリームモデル**: 単一ストリームモデルを前提とする
  （#1012 の設計が確定するまでの暫定前提）。同一ストリーム上での再利用は
  ストリーム順序により安全（前段カーネルの完了前に後続カーネルが同じ
  領域を書くことはない）。プールは `(ordinal, stream)` 単位に持つ。
  別ストリームへの貸し出しは v1 では禁止する（将来対応時はイベント
  fence を付与する）。ホストからの読み書きは既存の `download`（内部で
  `synchronize()` を伴う）経由に限る。
- **CUDA driver プールとの比較**（#1020 の判断材料）:

  | 案 | 内容 | 長所 | 短所 |
  |---|---|---|---|
  | A | driver の stream-ordered アロケータのみを使う（`cuMemPoolSetAttribute` で `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` を引き上げる） | 実装コストが最小・`unsafe` FFI 追加が 1 箇所（`result::mem_pool` の attribute 操作）に閉じる | Metal に相当物がなくバックエンド間で設計が割れる・`AllocationTracker` から保持量が直接見えない（`RESERVED_MEM_CURRENT` 等の別途取得が必要）・`has_async_alloc` が偽の環境（driver 側プール非対応）ではそもそも効果がない |
  | B | 自作サイズクラスプール（§3.1〜3.2） | backend 非依存・REQ-14 の計測系列（`AllocationTracker`）に自然に乗る・Metal と設計を共有できる | 実装コストが A より大きい |

  推奨は **B を正とする**。A は #1020 の実機計測を踏まえて追加検討する
  （採用する場合は release threshold を `max_pool_bytes` 以下に設定し、
  `release_cached()` から `cuMemPoolTrimTo(0)` 相当も併せて呼ぶ）。
  `has_async_alloc()` が偽の環境では `CudaSlice::drop` がホスト同期を伴う
  経路へ退化するため、自作プールによる「解放しない」ことの価値が相対的に
  高くなる点も判断材料として記録する。
  **B 採用時の driver プール予約メモリの扱い（レビュー指摘への対応）**:
  A/B いずれの設計を選んでも、`has_async_alloc()` が真の環境では cudarc
  自体が内部で `cuMemAllocAsync`／`cuMemFreeAsync`（driver 側 stream-ordered
  アロケータ。§2 事実 6）を経由する。つまり B（自作サイズクラスプール）を
  採用しても、自作プールの `release_cached()`／LRU 破棄で `CudaSlice` を
  drop するのは自作プール層の保持を解くだけであり、driver 側プールが
  release threshold の既定挙動により予約メモリ（reserved memory）を
  保持し続け得る。これは A 固有の懸念ではなく `has_async_alloc()` が真で
  ある限り常に該当するため、`release_cached()` の契約（§3.6 (2)）へ
  driver 側トリム呼び出しを組み込む。
- **Metal（codex-review PR #1056 P1 是正。返却を GPU 完了へ構造的に
  結び付ける）**: #1054 §3.4 の契約をそのまま採用する。すなわちプールへ
  返却されたバッファは、in-flight バッチの保持列から外れる（＝そのバッチ
  が `synchronize()` される）まで再貸出ししない。旧稿はこの契約を文章と
  してのみ述べており、`PooledMetalHandle::Drop` が実際にどう連動するかの
  機構が未定義だった（scope 終了で即座に `put()` すると契約に違反する）。
  本設計は以下の機構で契約を実現する:
  - `MetalContext` の `Mutex<BatchSlots>`（`crates/backend-metal/src/
    context.rs`。#1054/#1057 で実装済みの `open`／`committed` バッチ
    保持機構と同じロックで直列化する）に、**保留中のプール返却列**
    `pending_pool_returns: Vec<(u64, MetalBufferHandle,
    Arc<SizeClassPool<MetalBufferHandle>>)>`（`class_bytes`・生ハンドル・
    戻す先のプールの組）を新設する。
  - `PooledMetalHandle::Drop` は `self.handle.take()` で所有権を得た後
    （`record_loan_end` 呼び出しは §3.1「RAII 貸出ラッパー型」のとおり
    このタイミングとは独立に必ず実行済みとする）、`MetalContext` の
    現在の `BatchSlots` を検査する: **`open` バッチが存在する、または
    `committed`（commit 済みだが `waitUntilCompleted` 未実行）バッチが
    1 つ以上存在する場合**（＝直近の `synchronize()` 以降に GPU へ
    投入した work が残っている状態）は、`put()` を呼ばず、**`BatchSlots`
    の同一ロック区間内で** `pending_pool_returns` へ `(class_bytes,
    handle, pool)` を追加して所有権を委譲する**のと同時に**
    `pool.record_pending_return(class_bytes)`（§3.1。`SizeClassPool<H>`
    の `Mutex<PoolCore<H>>` を取らない lock-free な統計専用メソッド）を
    呼んで `pending_return_bytes` を加算する（codex-review PR #1056 P1
    是正・Cursor Bugbot 再指摘対応。旧稿は本メソッド呼び出しを
    `BatchSlots` のロックを解放した**後**に置いており、push と加算の
    間に別スレッドの `synchronize()` が割り込んで先に `record_pending_
    merge` を実行してしまう競合が生じ得た。`record_pending_return` は
    lock-free〈`AtomicU64::fetch_add`〉であるため、`BatchSlots` の
    ロックを保持したまま呼んでもロック順序のネスト・デッドロックには
    ならない。§3.5「Metal `pending_pool_returns` の排他制御」参照）。
    個々のバッファがどの dispatch に実際に参照されたかまでは追跡しない
    （#1054 §3.4 の「過剰同期側」方針と同じ粒度の保守的判定であり、
    実装コストと安全性のバランスを崩さない）。**`open`／`committed` の
    いずれも存在しない場合**（＝直前の `synchronize()` で全て完了済み、
    かつその後まだ何も encode されていない状態。「フラッシュ前に Drop
    されたラッパー」ではなく「in-flight なしの状態で Drop された
    ラッパー」に相当）は、`BatchSlots` のロックを解放した後、その場で
    `pool.put(class_bytes, handle)`（`Mutex<PoolCore<H>>` を要する。
    ロック解放後に呼ぶ理由は §3.5 参照）を呼んでよい（この経路では
    `pending_return_bytes` は変化しない）。
  - `MetalContext::synchronize()`（`waitUntilCompleted()` で全 committed
    バッチを完了させる箇所）は、`waitUntilCompleted()` 完了後、**まず
    `BatchSlots` の同一ロック区間内で** `pending_pool_returns` を
    `std::mem::take` して空にする**のと同時に**、取り出した各エントリ
    `(class_bytes, handle, pool)` について
    `pool.record_pending_merge(class_bytes)`（§3.1。lock-free。
    `pending_return_bytes` を減算する）を呼ぶ（codex-review PR #1056
    P1 是正・Cursor Bugbot 再指摘対応。`take` と同一ロック区間内で
    呼ぶことにより、`Drop` 側の push＋加算〈同じく単一クリティカル
    セクション〉との間で加算と減算の順序が入れ替わることが構造的に
    起こらなくなる。§3.1「統計専用メソッドの検証」参照）。**その後
    ロックを解放してから**、取り出した各エントリへ `pool.put(class_bytes,
    handle)`（`Mutex<PoolCore<H>>` を要するため。フリーリストへ実際に
    記帳し `cached_bytes` を加算する）を呼ぶ（`record_loan_end` は
    `Drop` 時点で既に呼び出し済みのためここでは呼ばない。§3.1
    「フィールド更新契約」`pending_return_bytes` 行）。**この合流
    （`pending_pool_returns` のロック内 `take`＋`record_pending_merge`
    と、ロック外での `put` 呼び出し）は、`waitUntilCompleted()` の
    実行時エラーの有無に関わらず、すなわち `synchronize()` が `Ok` を
    返す場合・`Err` を返す場合のいずれでも必ず行う**（codex-review PR
    #1056 P1 是正・Cursor Bugbot 指摘対応。理由: `waitUntilCompleted()`
    から復帰した時点で、対象コマンドバッファは成功・失敗いずれの結果
    であっても GPU 上での実行は完了しており、GPU 側がそのバッファを
    これ以上参照することはないため、合流〈フリーリストへの返却〉自体の
    安全性は実行結果の成否に依存しない。合流はフェーズ (i) 自体の
    一部であり、フェーズ (ii) が担う「解放処理」ではない。この区別が
    §3.6 (2)「バックエンド別の該当フェーズ」表・「フェーズ (i) 失敗時の
    共通契約」の前提である）。`synchronize()` が `Ok`／`Err` いずれで
    復帰する場合も、`pending_pool_returns` のロック内 `take`＋
    `record_pending_merge` が完了した時点で `pending_return_bytes` は
    合流対象分だけ確定的に `0` へ戻る（push／`take` と加減算が同一
    クリティカルセクションで対になるため、旧稿にあった「ロック解放後の
    短い窓での一時的なずれ」は本設計には存在しない。`record_pending_
    merge` の完了〈ロック内〉から対応する `put`〈ロック外〉までの短い
    窓では、当該バイト数は `pending_return_bytes` にも `cached_bytes`
    にも含まれない中間状態になるが、二重計上にはならず `max_pool_
    bytes`／LRU の判定〈§3.4〉は実態より少なく見える保守側にのみ振れる）。
    `take_one_for_release` によるフリーリスト走査を伴う操作
    （`release_cached_device_memory()` のフェーズ (ii)。§3.6 (2)）は、
    Metal では**必ずこの `synchronize()` の完了（`Ok`／`Err` いずれも
    含む）を先に待ってから開始する**契約とする（Cursor Bugbot 指摘
    「`pending_pool_returns` がフリーリスト外に置かれ
    `release_cached`／`take_one_for_release`／`cached_bytes`／LRU から
    見えない」への対応。§3.4・§3.6 (2) 参照。completion handler
    〈`addCompletedHandler`〉方式は採らない。既存 `synchronize()` が
    ブロッキングの `waitUntilCompleted()` ポーリング方式で実装済み
    〈#1054/#1057〉であり、その完了直後に処理するほうが既存のロック
    区間・エラー伝播〈`batch_state::propagate_failure`〉と一貫する
    ため）。
  - ゼロ初期化はホスト書き込み（`zero_fill`。#1054 §3.5 が定義する
    同期点）ではなく、`MTLBlitCommandEncoder::fillBuffer` によるデバイス
    側フィルをバッチへ encode する案を推奨する（同期点を新たに増やさ
    ないため）。最終選択は #1021 に委ねる。
- **二重返却・use-after-return の構造的防止（codex-review PR #1056 P2
  是正。方式の統一）**: §3.1「RAII 貸出ラッパー型」が定める
  `Option<H>::take()` 方式に統一する。旧稿は本節で既存
  `PooledBufferHandle` の `ManuallyDrop` によるガード方式
  （`crates/tensor-core/src/pool.rs`）を継承すると記載しており、§3.1 の
  `Option<H>::take()` 方式と矛盾していた。`PooledMemory`（既存）が
  `ManuallyDrop` を使うこと自体は変更しない（既存公開 API・実装への
  非破壊）が、本設計（`SizeClassPool<H>`・`PooledCudaHandle`／
  `PooledMetalHandle`）は新規実装であり `ManuallyDrop` を採用しない。

### 3.4 断片化

- **内部断片化**: 小クラス帯は理論上限 25%（実効はもっと小さい。§3.2 表
  参照）。大クラス帯は 2 MiB 未満／バッファ。`PoolStats::
  capacity_waste_bytes`（= Σ(capacity_bytes − 論理バイト数)。POD。
  `AllocatorStats` から改称。§3.1）で可視化する。
- **外部断片化**: v1 では slab／サブアロケーション（burn/cubecl の
  `SlicedPool` 相当。1 つの大きな確保を複数の論理バッファへオフセット
  分割する方式）を採用しない。`CudaSlice`／`MTLBuffer` のオフセットビュー
  と寿命結合の実装コストが大きく、影響範囲がホットパス全体に及ぶため
  （不採用理由・代替案は §4）。個々のプールエントリは driver／OS 側の
  独立確保であり、外部断片化の管理は driver に委ねる。
- **緩和策**: 総量上限＋グローバル LRU（既存踏襲）・クラス別アイドル
  上限・`release_cached()`（内部メソッド。§3.1「2 段構成の命名規約」）・
  OOM 時は `release_cached()` を 1 回実行してから再試行し、それでも
  失敗すれば `BackendError::DeviceAllocationFailed` を返す（fail-closed。
  無限リトライしない）。
  - **`max_pool_bytes`／LRU 判定の対象（Cursor Bugbot 指摘対応。§3.3
    「Metal」の `pending_pool_returns` を含める）**: `max_pool_bytes`
    超過判定・グローバル LRU（クラス別アイドル上限）はいずれも
    `PoolStats::cached_bytes + PoolStats::pending_return_bytes`
    （「アプリから見て idle だが保持中の総量」。フリーリストに実際に
    ある分＋ GPU 完了待ちで返却待ちの分）を対象に行う。**CUDA・CPU
    では `pending_return_bytes` が常に `0`（§3.3「Metal」）のため
    `cached_bytes` のみを見る現行の判定と実質的に同じ**であり、この
    変更は Metal にのみ意味を持つ。`pending_pool_returns` 中のエントリ
    は `synchronize()` による合流（§3.3「Metal」）を経るまでフリー
    リストへ実際には入らないため、**上限超過を検知した時点で
    `pending_pool_returns` 分をただちに LRU 解放することはできない**
    （フリーリスト外にあるためそもそも `take_one_for_release` の対象に
    ならない）。したがって Metal では、上限超過が検知されても実際の
    LRU 解放は次回の `synchronize()` 合流時点まで遅延しうる。この間、
    プールが実際に保持するデバイスメモリ量（`cached_bytes +
    pending_return_bytes`）が `max_pool_bytes` を**一時的に**超過する
    余地がある。この超過は「返却待ちだが同期待ちしている in-flight の
    作業セット」の分に限られ無制限には増大しない（同時に in-flight で
    あり得るバッチ数・バッファ数は §3.3「Metal」の安全弁〈#1054 §3.6
    「バッチ内 dispatch 数の安全弁上限」〉により有界であるため、REQ-14
    の A04〈資源枯渇〉が懸念する無制限成長には該当しない）。REQ-14 の
    係数上限（2 倍以内。`docs/peak-memory-coefficient-decision.md`）
    自体は変更しない。この一時超過の可能性自体は §0 の不変事項
    （REQ-14）に対する追加の注記として明記する。
  - **Metal の OOM フォールバック順序（Cursor Bugbot 指摘対応）**:
    Metal では OOM フォールバックが呼ぶ `release_cached()`（内部
    メソッド）が §3.6 (2) の Metal 版フェーズ (i)（`synchronize()`。
    `pending_pool_returns` の全件合流を含む。§3.3「Metal」）→ フェーズ
    (ii)（`take_one_for_release` によるフリーリスト走査・個別解放）の
    順で実行される契約になるため、`synchronize()` を経ずに
    `take_one_for_release` から直接開始することはない（上記「合流を
    経ずに LRU 解放できない」制約と整合する）。**`release_cached()` が `Result<u64,
  BackendError>` を返す契約（§3.1）により、この OOM フォールバック経路は
  2 種類の失敗を区別して扱う**: (i) `release_cached()` 自体が `Err` を
  返した場合（§3.6 (2) の 4 フェーズ〈stream 同期 2 回・個別解放・
  driver トリム〉のいずれかの失敗。理由文字列によるフェーズ区別は
  §3.6 (2)「`Err` の種別」参照）は、その `Err` をここで黙殺せず、確保の
  再試行を行わずに `BackendError::DeviceAllocationFailed` へ変換して
  即座に呼び出し元へ返す（driver 側の異常状態のまま再試行しても成功
  する見込みが薄く、黙殺すると fail-open になるため）。この場合も
  プール内部状態は一貫させる（§3.6 (2) の fail-closed 契約・§3.1
  「解放時の所有権遷移」の `take_one_for_release`／`put` トランザクション
  により、フェーズ (i)／(ii) の失敗では解放できた分のみフリーリストから
  外れ未解放分はキャッシュに残るため、直後の再試行は「フリーリストに
  残存する未着手エントリ」のみを対象に `take_one_for_release` を再開
  できる〈同一エントリを二重に解放しようとはしない〉。フェーズ (iii)／
  (iv) の失敗ではフリーリストは既に空であり、残存エントリを対象にした
  再試行という概念自体が該当しない〈§3.6 (2)「部分失敗時の状態一貫性」
  参照〉）。(ii) `release_cached()` が `Ok`
  を返した（解放自体は成功した）にもかかわらず再確保が失敗した場合
  のみ、上記のとおり `BackendError::DeviceAllocationFailed` を返す。
  この内部 `release_cached()` の `Err` は、公開 API
  `BackendOps::release_cached_device_memory()`（§3.1）が利用者から
  明示的に呼ばれた場合にも同じ契約でそのまま伝播する。

### 3.5 スレッド安全

- プール本体は `Mutex<PoolCore>`（既存方針を踏襲）で保護する。ロックを
  保持したまま FFI 呼び出し（`cuMemAllocAsync`／`newBuffer` 等）を行わない
  （フリーリスト操作のみをロック内で行い、確保が必要な場合はロック解放後
  に FFI を呼ぶ）。poison 時は既存方針（`into_inner` で継続。panic させ
  ない）を踏襲する。
- **`Send`/`Sync` 方針の更新**: `SizeClassPool<H>: Send + Sync where H:
  Send` とし、各バックエンドの**生ハンドル型** `H`（例:
  `CudaSliceHandle`・`MetalBufferHandle`。§3.1）は `Send` を実装する。
  RAII 貸出ラッパー型（`PooledCudaHandle`・`PooledMetalHandle`。§3.1）は
  `handle: Option<H>`（`H: Send`）と `pool: Arc<SizeClassPool<H>>`
  （`SizeClassPool<H>: Send + Sync`）のみを保持するため自動的に `Send`
  になる。根拠は §2 事実 6・8: `CudaSlice` は `Send + Sync`、Metal の
  `MTLBuffer` protocol も objc2-metal 0.3.2 で `Send + Sync` を
  supertrait に持つ。
  `crates/tensor-core/src/buffer.rs`「Send/Sync 境界」節が定めた「必要に
  なった時点で再検討する」の条件に、複数スレッド（学習ループのワーカー間
  でのプール共有）から確保・返却を行う本設計で到達したと位置づける。
  §2 事実 7 が定める「`BufferHandle`／`DeviceBuffer` は `Send`/`Sync` を
  要求しない」という既存方針自体は変更しない（`H` はバックエンド内部の
  独自ハンドル型であり、公開 `BufferHandle` trait とは別物。§3.1 の
  とおり `SizeClassPool<H>` は `tensor-core` の公開面に `BufferHandle`
  を一切持ち込まないため、`BufferHandle` trait 自体の supertrait 変更は
  不要かつ発生しない）。この変更により `crates/tensor-core/src/pool.rs`
  の `arc_with_non_send_sync` allow は解消できる見込みであり、
  #1020／#1021 へ申し送る。
- プールは device 単位のプロセスワイド singleton とする（`static_cuda_
  memory`／同等の `Box::leak` 所有モデル。§2 事実 2 の計測系列単一化と
  整合させる）。CPU バックエンドは本イシューの対象外（別イシュー #1026 が
  担当）だが、`SizeClassPool<H>` 自体は backend 非依存に設計されている
  ため、CPU 実装が必要になった際は恒等実装（`Vec` 確保をそのまま返す）
  で満たせる。
- **Metal `pending_pool_returns` の排他制御（codex-review PR #1056 P1
  是正・Cursor Bugbot 再指摘対応で精密化。§3.3「Metal」参照）**:
  `MetalContext` の `pending_pool_returns` は既存 `Mutex<BatchSlots>`
  （`open`／`committed` と同じロック。`crates/backend-metal/src/
  context.rs`）へ同居させ、専用の別ロックを設けない。理由は (i)
  `PooledMetalHandle::Drop` が「`open`／`committed` の有無を検査する」
  処理と「`pending_pool_returns` へ追加する」処理を同一ロック区間で
  行わないと、検査後・追加前に別スレッドが `synchronize()` を呼んで
  空の `pending_pool_returns` を drain してしまう TOCTOU（time-of-check
  to time-of-use）競合が生じうるため、(ii) `synchronize()` 自体も
  `waitUntilCompleted()` の間ロックを保持し続ける設計（#1054 §3.5
  「同時実行の扱い」・`context.rs::synchronize` コメント「他スレッドから
  の `encode`／`synchronize` はこの間ブロックされる」）と同じ「正しさ
  優先・並行性は最適化しない」方針に合わせるためである。

  **ロック順序規則（精密化）**: `BatchSlots` のロックを保持している間、
  `SizeClassPool<H>` の**`Mutex<PoolCore<H>>` を要する操作**
  （`take`／`put`／`take_one_for_release`／`stats()` によるスナップ
  ショット取得。以下「Mutex 系操作」）を呼ばない。これらは
  `BatchSlots` とは別ロックであり、保持中に呼ぶとロック順序がネスト
  しデッドロックしうるため。**`record_pending_return`／
  `record_pending_merge`（§3.1。以下「atomic 系メソッド」）はこの
  規則の例外とする**: いずれも `SizeClassPool<H>` 内部の `AtomicU64`
  を lock-free（`fetch_add`／`fetch_sub`。`Mutex<PoolCore<H>>` を
  一切取らない）に操作するため、`BatchSlots` のロックを保持したまま
  呼んでもロック順序のネスト・デッドロックが起こらない。

  この例外は単なる許容ではなく**必須の契約**である（Cursor Bugbot
  再指摘対応。旧稿は atomic 系メソッドも Mutex 系操作と同様に
  「ロックを解放した後に呼ぶ」としていたが、これは `pending_pool_
  returns` への push／`take`〈`BatchSlots` ロックで直列化される〉と
  `record_pending_return`／`record_pending_merge`〈ロック解放後に
  個別に呼ばれる〉の間に別スレッドが割り込む余地を残し、
  `record_pending_merge`〈減算〉が対応する `record_pending_return`
  〈加算〉より先に実行される競合を生んだ。`pending_return_bytes` は
  `u64` の下限を割り込むと `saturating_sub`〈仮に採用していた場合〉で
  `0` に張り付き、その後に本来先に走るはずだった加算が乗ることで
  `pending_return_bytes` が実態より恒久的に高い値のまま戻らなくなる
  〈`max_pool_bytes`／LRU 判定〈§3.4〉が実態より過大に見積もる方向へ
  壊れる。表示上の一時的なずれでは済まない〉）。是正内容:
  - `PooledMetalHandle::Drop`（§3.3「Metal」）: `BatchSlots` の**同一
    ロック区間内**で `pending_pool_returns` への push と
    `pool.record_pending_return(class_bytes)`（atomic 系メソッド）を
    **同時に**行う。`pool.put(class_bytes, handle)`（Mutex 系操作。
    `open`／`committed` がいずれも存在しない場合の即時返却経路）は
    ロックを解放した**後**に呼ぶ。
  - `MetalContext::synchronize()`（§3.3「Metal」）: `BatchSlots` の
    **同一ロック区間内**で `pending_pool_returns` を `std::mem::take`
    して空にするのと `pool.record_pending_merge(class_bytes)`
    （atomic 系メソッド。取り出した各エントリについて呼ぶ）を
    **同時に**行う。その後**ロックを解放してから**、取り出した各
    エントリへ `pool.put(class_bytes, handle)`（Mutex 系操作。
    フリーリストへ実際に記帳する）を呼ぶ（`flush_locked` が
    `slots.committed` を `waitUntilCompleted` ループの前に
    `std::mem::take` する既存パターンと同型。ロックを保持したまま FFI
    呼び出しを行わないという本節冒頭の方針を、Mutex 系操作〈フリー
    リスト操作のみで FFI ではないが、別ロック取得を伴うため同様に
    扱う〉にも適用する）。
  - **短い窓に残る非同期性（許容する範囲。恒久的な誤差ではない）**:
    `record_pending_merge`（ロック内。減算完了）から対応する `put`
    （ロック外。フリーリストへの記帳完了）までの間には短い窓が残る。
    この窓では、当該 `class_bytes` 分は `pending_return_bytes` にも
    `cached_bytes` にも含まれない中間状態になる。これは二重計上には
    ならず、この窓の間に `max_pool_bytes`／LRU（§3.4）を判定すると
    実態より**少なく**見える（保守側〈under-count〉にのみ振れる。
    over-count・恒久的な誤差にはならない）。custody（どのハンドルが
    フリーリスト・貸出中・返却待ちのいずれにあるか。§3.1「不変条件」）
    自体は常に `BatchSlots` ロックまたは `SizeClassPool<H>` 内部ロック
    のいずれかで正しく直列化されており、メモリ安全性・二重貸出防止
    には影響しない。

### 3.6 解放戦略（受け入れ条件）

1. 総量上限（既定 128 MiB を暫定継続）＋グローバル LRU。学習ワーキング
   セットとの関係は #1010 の内訳実測後に再評価する（変更する場合はユーザー
   承認事項）。
2. `release_cached()`（内部メソッド。§3.1「2 段構成の命名規約」。
   公開 API `BackendOps::release_cached_device_memory()` から呼ばれる。
   REQ-14 の要求する解放 API）。**手順は §3.1「解放時の所有権遷移」の
   トランザクション型 API（`take_one_for_release`／`put`）に従う
   （codex-review PR #1056 P1 是正。旧稿の一括 `drain()` を置き換えた）**。
   **以下は CUDA の 4 段構成を示す（Cursor Bugbot High 是正。旧稿は 3 段で、
   `cuMemFreeAsync` のエンキューのみで完了を待たずに driver トリムを
   呼んでいたため、トリムが未完了の解放を見落としていた）。Metal・CPU
   の該当フェーズは本項目末尾「バックエンド別の該当フェーズ」表を
   参照（codex-review PR #1056 P1 是正。公開 API のエラー契約が
   バックエンド別の対応関係を欠いていたとの指摘への対応）**:
   (i) 対象 stream を同期（`cuStreamSynchronize` 相当。cudarc の同期
   API 経由）し、これから行う `take_one_for_release`／drop に**先立って**
   現時点までの enqueue 済み GPU work が完了していることを確認する。
   **この同期は `take_one_for_release` を 1 度も呼ぶ前に行い、失敗した
   場合はフリーリストへ一切触れずに `Err` を返す**（キャッシュは全件
   無傷のまま残る。fail-closed）。(ii) 同期成功後、`take_one_for_release()`
   をフリーリストが空になるまでループで呼ぶ。取り出した `(class_bytes,
   handle)` ごとに実際の解放（`handle` の drop。`has_async_alloc()` が
   真の環境では `CudaSlice::drop` は `cuMemFreeAsync` を対象 stream へ
   **enqueue するのみで完了を待たない**。§2 事実 6）を行い、`PoolStats`
   （`cached_bytes` 等）を更新してから次のエントリへ進む（drop 自体は
   Rust の型システム上 `Result` を返せないインメモリ操作であり失敗しない。
   `put` による再挿入は、drop 以外の解放手段〈将来のバックエンド・
   想定外の実装変更〉に備えた防御として用意する）。(iii) **フリーリスト
   の全エントリの drop 完了後、`take_one_for_release` を一切呼ばずに
   対象 stream を再度同期する（2 回目の stream 同期。新設）**。これは
   ステップ (ii) で enqueue した全 `cuMemFreeAsync` が実際に完了し、
   解放領域が driver 側 memory pool（§3.3）へ返却済みであることを
   確認するためであり、この同期を経ずに (iv) のトリムを呼ぶと
   「プールが現在保持する空き領域のうち閾値超過分」しか対象にしない
   `cuMemPoolTrimTo(0)` がステップ (ii) の解放分をまだ観測できず、
   driver 予約メモリを取りこぼす（Cursor Bugbot 指摘の欠陥そのもの）。
   **この 2 回目の同期が失敗した場合**、ステップ (ii) で drop した
   ハンドルは既に Rust の値としては存在せず（drop 済みであり `put` で
   フリーリストへ戻す対象が無い）、`PoolStats::cached_bytes` も
   ステップ (ii) の時点で既に解放済みとして更新済みのため**そのまま
   確定**とする（黙殺で成功扱いにするのではなく、後述のとおり `Err`
   の種別で「解放は完了しているがトリム未実施」であることを呼び出し元へ
   伝える）。この場合ステップ (iv) は実行せず打ち切る（同期が失敗した
   ままトリム FFI を追加で呼んでも意味のある結果を保証できないため）。
   (iv) ステップ (iii) の同期が成功した場合のみ、driver 側 memory pool
   に残る予約メモリを `cuMemPoolTrimTo(0)` 相当（`result::mem_pool`
   経由。§2 事実 6）で 1 回解放する（A/B いずれの設計かに関わらず必須。
   §3.3 参照）。これにより `release_cached()` が REQ-14 の解放 API として
   自作プール保持分だけでなく driver 予約分も含めて解放する契約になる。
   このトリム呼び出しが失敗した場合も、自作プール層のフリーリストは
   既に空（`PoolStats::cached_bytes == 0`。ステップ (ii)・(iii) が
   完了済みのため）であり、`Err` を返す（driver 予約分の残存は下記
   「REQ-14 との整合」が別途扱う既知の差分）。

   **`Err` の種別（呼び出し元が状態を誤認しないための区別。Cursor
   Bugbot 指摘対応）**: `release_cached()` の戻り値は `Result<u64,
   BackendError>` とする（§3.1 の内部メソッド定義）。`Err` は次の
   いずれかであり、呼び出し元（LRU／OOM フォールバック。§3.4）は
   種別ごとに異なる状態を前提としてよい:
   - **フェーズ (i) 失敗**（`BackendError::DeviceAllocationFailed`
     に `"pre-free sync"` 等の識別可能な理由文字列を含める）: **共通
     契約は「解放対象としてのフリーリスト走査・取り出し（`take_one_
     for_release` の呼び出し）を一切開始しない」ことであり、「フリー
     リストへ一切触れない」ことまでは要求しない**（Cursor Bugbot 指摘
     対応。旧稿は後者〈CUDA にのみ当てはまる帰結〉を共通契約として
     誤って記載していた）。バックエンドごとの帰結は次のとおり:
     CUDA は `pending_pool_returns` 相当の合流機構を持たないため文字
     通りフリーリストは完全に無傷（全件キャッシュ残存）。Metal は
     `synchronize()`（フェーズ (i) そのもの）が `pending_pool_returns`
     の全件合流（`record_pending_merge`／`put`）を**同期の成否に
     関わらず**行うため（§3.3「Metal」）、フリーリストは合流分だけ
     増加しうる。この増加は「解放処理」ではなくフェーズ (i) 自体が
     担う入出金（返却待ちからフリーリストへの移動）であり、`take_
     one_for_release` によるフリーリストからの取り出しはいずれの
     バックエンドでも一切発生しない点は共通する。
   - **フェーズ (ii) 個別解放失敗**（現行 CUDA 実装では発生しない
     防御的経路。理由文字列に `"handle release"` 等を含める）: 失敗した
     エントリのみ `put` で再挿入され、以降の未着手エントリもフリー
     リストに残る。
   - **フェーズ (iii) 失敗**（理由文字列に `"post-free sync"` 等を
     含める）: フリーリストは既に空（`cached_bytes == 0`）。**再挿入は
     行わない**（ステップ (ii) で drop 済みのため再挿入対象が存在
     しない）。driver 予約分の解放が完了したかは未確認のまま処理を
     打ち切る（ステップ (iv) 未実行）。
   - **フェーズ (iv) 失敗**（理由文字列に `"driver trim"` 等を含める）:
     フリーリストは既に空（`cached_bytes == 0`）。driver 予約分の
     トリムのみ未完了。
   いずれの `Err` も黙殺（成功したものとして扱う）・`panic!` する・
   エラーを判別不能な値へ曖昧に符号化する（例: 失敗を `0` として返す）
   ことは、driver 予約分の残存を「解放成功」と誤認させ fail-open に
   なるため禁止する（本番経路の panic 禁止・fail-closed の維持。
   `.claude/rules/coding-rust.md`）。呼び出し元はこの `Err` を上位の
   `BackendError::DeviceAllocationFailed` 等の型付きエラーへそのまま
   伝播する契約とする（理由文字列は診断用であり型を分岐させる契約と
   まではしない。`BackendError` に新規 variant を追加すると破壊的変更
   になるため、既存 `DeviceAllocationFailed(String)` の文字列で区別する
   非破壊な方式を採る）。
   **部分失敗時の状態一貫性（fail-closed。§3.4 との整合）**: フェーズ
   (i) で失敗した場合、CUDA ではフリーリストは無傷（全件キャッシュ
   残存）。Metal では `synchronize()`（フェーズ (i)）自体が
   `pending_pool_returns` を同期の成否に関わらず全件合流させるため
   フリーリストは増加しうるが（上記「`Err` の種別」参照）、いずれの
   場合も `take_one_for_release` によるフリーリストからの取り出しは
   一切発生しない。フェーズ (ii) の途中で個別の解放処理が失敗した場合（上記のとおり
   CUDA の現行実装では発生しないが、防御として規定する）は、その
   `(class_bytes, handle)` のみ `put` で同じフリーリストへ再挿入し、
   以降の未着手エントリもフリーリストに残ったまま `Err` を返す（二重
   解放しない。解放済みエントリを再度解放しようとしない）。フェーズ
   (iii)／(iv) で失敗した場合は、フリーリストは既に空であり「一部解放
   済み・残りはキャッシュ保持中」という状態には当たらない（全件解放
   済みだが driver 予約分の可視化・トリムだけが未完了）。呼び出し元は
   `Err` を受け取った時点でこれらを区別し、フェーズ (i)／(ii) 失敗時の
   再試行（§3.4）のみ `take_one_for_release` を継続呼び出しすることで
   **フリーリストに残存する未着手エントリのみを対象に**行われる
   （フェーズ (iii)／(iv) 失敗時はフリーリストが既に空のため、再試行
   すべき残存エントリ自体が存在しない）。

   **バックエンド別の該当フェーズ（codex-review PR #1056 P1 是正。
   `BackendOps::release_cached_device_memory()` の doc comment〈§3.1〉が
   参照する正本）**:

   | バックエンド | フェーズ (i) | フェーズ (ii) | フェーズ (iii) | フェーズ (iv) |
   |---|---|---|---|---|
   | CUDA | `cuStreamSynchronize` 相当（トリム前の対象 stream 同期） | `take_one_for_release` ループでの個別解放（`CudaSlice` の drop。`cuMemFreeAsync` の enqueue のみで完了を待たない） | `cuStreamSynchronize` 相当（(ii) で enqueue した `cuMemFreeAsync` の完了待ち。新設） | `cuMemPoolTrimTo(0)` 相当（driver 予約メモリのトリム） |
   | Metal | `MetalContext::synchronize()`（`waitUntilCompleted()`。**`Ok`／`Err` いずれで復帰する場合も、復帰前に `pending_pool_returns`〈§3.3「Metal」〉を全件フリーリストへ合流させる**〈`record_pending_merge`／`put`。合流は「解放処理」ではなくフェーズ (i) 自体が担う。上記「`Err` の種別」参照〉。フェーズ (ii) 開始時点で必ず `pending_return_bytes == 0` であることが前提条件） | `take_one_for_release` ループでの個別解放（`Retained<ProtocolObject<dyn MTLBuffer>>` の drop による ObjC 参照カウント減算。呼び出し可能な解放 FFI が存在せず**失敗しない**） | なし（Metal に driver 側 memory pool・トリム相当の機構が存在しないため） | なし（同上） |
   | CPU | 該当なし（プール自体が本イシューの対象外。#1026） | 該当なし | 該当なし | 該当なし |

   Metal は driver トリムを持たないため `release_cached()` の `Err` は
   フェーズ (i)（`synchronize()` 失敗。§3.3「Metal」の
   `batch_state::propagate_failure` 経由で検出される実行時エラー等）
   のみに限定される（フェーズ (ii) は上記のとおり失敗しない設計のため、
   §3.6 (2) 冒頭「`Err` の種別」のフェーズ (ii) 行は Metal では実質
   到達不能）。CPU は `release_cached_device_memory()` のデフォルト
   実装（§3.1）のまま常に `Ok(())` を返す。
3. OOM フォールバック（§3.4）。
4. プロセス終了時は OS 側の回収に委ねる（既存 `Box::leak` 方針と同じ、
   意図的な「解放しないリーク」であることを明記する）。
5. Metal の `setPurgeableState` によるページ回収ヒントは不採用（スコープ
   外。§6）。

REQ-14 との整合: プール保持分は既存 `AllocationTracker::allocated_bytes`
に計上され続ける（既存機構をそのまま用いる）。ただし CUDA で
`has_async_alloc()` が真の環境では、driver 側 memory pool の予約メモリ
（§3.3）は自作プール層より下（cudarc 内部）で保持されるため
`AllocationTracker` の計上対象に含まれない。この差分を放置すると
内部 `release_cached()`／LRU 破棄後も実メモリは driver 側に残存し得るため、
GEMM 4096³ の係数 2.0 判定を `AllocationTracker` の値のみで行うと実際の
ピークメモリを過小評価するおそれがある。したがって本設計では
(i) 内部 `release_cached()`（公開 `BackendOps::release_cached_device_memory()`
からも到達可能）が §3.6 (2) の driver トリム呼び出しを含むことで
driver 予約分の残存を都度解消し、(ii) #1020 の実機計測では
`AllocationTracker::allocated_bytes` に加えて driver 側の実メモリ使用量
（`nvidia-smi` またはプロセスの実メモリ計測）も併せて確認し、両者の乖離が
無いことを係数 2.0 判定の受け入れ条件へ加える、の 2 点を契約とする。
GEMM 4096³ で係数 2.0（384 MiB 以内）を維持できることは、巨大帯を
「完全一致・1 エントリ保持」に限定した §3.2 の方針と、上記 driver 予約
メモリのトリム・実測併用で担保する。係数 2.0 自体は変更しない（緩和・
厳格化ともユーザー承認事項。`docs/peak-memory-coefficient-decision.md`）。

### 3.7 計測・受け入れ条件への写像

親 #1018 が期待する「割当／解放区間の大幅な削減」は、`train --phases`
の区間のうち `forward`／`backward`／`tape_drop` に主に効く見込みである
（演算ごとの確保・解放がこれらの区間内で発生するため）。ただしこれは
区間名からの対応関係の推定に留まり、定量的な改善幅は現時点で存在しない
（§1）。

実測記入欄（Mac M4 Max・DGX Spark GB10。5 回計測中央値・
`docs/real-hardware-verification-env.md` 準拠。プール導入前後 ×
fresh/reuse の 4 通り）:

| 環境 | 導入前 fresh | 導入前 reuse | 導入後 fresh | 導入後 reuse |
|---|---|---|---|---|
| Mac M4 Max（Metal） | （記入待ち） | （記入待ち） | （記入待ち） | （記入待ち） |
| DGX Spark GB10（CUDA） | （記入待ち） | （記入待ち） | （記入待ち） | （記入待ち） |

## §4 代替案と採否

| 案 | 内容 | 不採用理由 |
|---|---|---|
| driver プールのみ（案 A、§3.3） | `cuMemAllocAsync`/`cuMemFreeAsync` の release threshold 調整のみで済ませる | Metal に相当物がなくバックエンド間で設計思想が割れる。REQ-14 の計測系列に自然に乗らない。#1020 の実測後に追加検討する扱いに留める（完全に排除はしない） |
| slab／サブアロケーション（`SlicedPool` 相当） | 1 回の大きな確保を複数の論理バッファへオフセット分割して再利用する | オフセットビューと寿命結合の実装コストが大きく、`CudaSlice`／`MTLBuffer` 双方への影響範囲がホットパス全体に及ぶ。v1 では見送り、§3.4 の代替案として記録するに留める |
| バイトサイズ完全一致を継続（現状） | 既存 `PooledMemory` の方式をそのまま接続する | ワークロードごとにバイトサイズが微妙に異なるケース（バッチサイズ違い等）でヒット率が低く、サイズクラス化のメリット（内部断片化の許容と引き換えの高ヒット率）が得られない |
| `PooledMemory` を `memory_ops()` にそのまま差し込む | 新規 trait を作らず既存デコレータを本番経路へ接続するだけにする | §2 事実 5 の「完全一致でないと `download` 契約を壊す」制約が残ったまま、サイズクラス化（§3.2）が実現できない。層構造（§3.1）として `MemoryOps` の下にバイト単位の抽象を置く方が既存 API と非破壊に両立できる |
| 公開 `BackendOps` へ `allocator()` を追加する（旧稿・1 巡目） | `BackendOps::allocator(&self) -> Option<&dyn DeviceAllocator>` というデフォルトメソッドを `memory_ops()` と同型で公開 trait へ追加し、動的ディスパッチで各バックエンドの `DeviceAllocator` へ到達させる | crates.io 公開済み trait（`fandhe-ai-tensor-core::BackendOps`）から `Box<dyn BufferHandle + Send>` を返す低水準プール操作へ到達できてしまい、`docs/compat-api-scope.md` §0「`facade` が唯一のサポート対象公開 API 面」に反する（codex-review PR #1056 P1・1 巡目） |
| `pub trait DeviceAllocator` は維持し `BackendOps::allocator()` のみ削除する（旧稿・2 巡目） | `BackendOps` への到達経路（`allocator()`）を削除すれば、`DeviceAllocator` の実装インスタンスがいずれの公開関数からも返らなくなるため十分と考えた | 「実装インスタンスを公開関数から返さない」ことと「型定義自体が公開面から不可視であること」は別の防御であり、後者を満たさない。`pub trait DeviceAllocator`・`pub mod allocator` として `fandhe-ai-tensor-core` に残る限り、外部コードが本クレートへ直接依存すれば `Box<dyn BufferHandle + Send>` を含む型を直接参照・独自実装できてしまう（codex-review PR #1056 P1・2 巡目。「facade が唯一のサポート対象公開 API 面」という**運用上の制約**は Rust の可視性検査を代替しない）。本設計（§3.1）は低水準 trait 自体を廃し、ハンドル型非依存の `SizeClassPool<H>`（`H` に trait 境界を課さない）と POD 型（`PoolConfig`／`PoolStats`）のみを `tensor-core` の公開面に残すことでこれを解消する |
| REQ-14 の解放 API を内部 OOM フォールバックのみで満たす（旧稿） | `release_cached()` を「REQ-14 の明示解放 API」と位置づけつつ、実装インスタンスをいずれの公開関数からも返さない設計のままにする | 内部 OOM フォールバック（§3.4）から呼べるだけでは、`facade` を含む利用者側からプールを明示的に解放する経路が存在せず、REQ-14 が求める「プール解放 API の提供」を満たさない（codex-review PR #1056 P1）。本設計（§3.1）は低水準アロケータを一切露出せずに `BackendOps::release_cached_device_memory()`／`facade::release_cached_memory(device)`（unit のみを返す）を確定入口として追加することでこれを解消する |
| `SizeClassPool<H>::register(class_bytes, handle)` で新規確保ハンドルをフリーリストへ記帳する（旧稿） | 確保直後のハンドルを `register` でフリーリストへ登録し、`take` で取り出して貸し出す設計 | フリーリストへの登録と貸出の間で所有権移転が定義されておらず、貸出中のハンドルが `take()` で別演算へ再取得できてしまう（use-after-return・上書きの構造的欠陥。codex-review PR #1056 P1）。本設計（§3.1）は `register` を廃止し、新規確保直後は RAII 貸出ラッパーが排他的に所有し、`put` は `Drop` 時のみ呼ばれる契約へ変更した |
| `SizeClassPool<H>::drain() -> Vec<H>` で一括解放する（旧稿） | 解放時にフリーリストの全ハンドルを一括で取り出し、呼び出し元が順次解放する設計 | `drain` した時点で全ハンドルがプールの管理外になるため、途中で stream 同期・driver トリムが失敗しても取り出し済みハンドルをキャッシュへ戻す先がなく、§3.6 (2) の「未解放エントリはキャッシュへ残す」・§3.4 の「残存分のみ再試行」という fail-closed 契約を実現できない（codex-review PR #1056 P1）。本設計（§3.1）は `take_one_for_release` による 1 件ずつのトランザクション型取り出しと `put` による失敗時再挿入へ変更した |
| Metal も CUDA と同様に `PooledMetalHandle::Drop` で即座に `pool.put()` する（旧稿） | RAII ラッパーの `Drop` 実装を CUDA／Metal で共通化し、`Option<H>::take()` 後ただちに `pool.put()` する | Metal のコマンド実行は非同期であり、`Drop` 時点で GPU 側の読み書きが完了している保証がない。§3.3「Metal」（#1054 §3.4 由来）は「in-flight バッチの保持列から外れる（`synchronize()` される）まで再貸出ししない」契約を既に確定させており、即時 `put()` はこの契約に違反し、返却直後に別演算へ `take()` されたバッファが実行中カーネルの入出力を上書きしうる（codex-review PR #1056 P1）。本設計（§3.3・§3.5）は `MetalContext` の `pending_pool_returns`（`Mutex<BatchSlots>` 保護）へ所有権を一時的に委譲し、`synchronize()` 完了後にのみ `put()` する機構へ変更した。CUDA は stream 順序保証（§3.3「返却の GPU 完了待ち契約」）により即時 `put()` のままで安全なため、この変更は Metal 側にのみ適用する |
| release_cached() を stream 同期 1 回 → 個別解放 → driver トリムの 3 段のままにする（旧稿） | `cuMemFreeAsync` の enqueue（個別解放）直後に `cuMemPoolTrimTo` を呼ぶ | `cuMemFreeAsync` はエンキューのみで完了を待たないため、直後の trim はまだ driver 側に解放が反映されていないメモリを見落とす（Cursor Bugbot High 指摘）。本設計（§3.6 (2)）は個別解放後にもう 1 回 stream 同期を挟む 4 段構成へ変更し、この 2 回目の同期の失敗はハンドルが既に drop 済み（再挿入不可）であることを踏まえた専用の `Err` 種別で区別する |
| `SizeClassPool<H>` を `take`／`put`／解放系メソッドのみに留め、新規確保は統計を一切更新しない（旧稿） | 新規確保は `take` を経由せず RAII ラッパーが直接包むため、`SizeClassPool<H>` はハンドルの custody を持つ操作（`take`／`put`／`take_one_for_release`）でしか統計を更新できないという前提のまま設計する | 新規確保の発生・その内部断片化（`class_bytes − logical_bytes`）を記録する経路が存在せず、`alloc_count`／`capacity_waste_bytes` という `PoolStats` の宣言済みフィールドを実装不能にする（codex-review PR #1056 P1）。本設計（§3.1）は `record_allocation`／`record_loan_end`（いずれもハンドルを受け取らない統計専用メソッド）を追加し、ハンドルの custody とは独立に統計だけを更新できるようにした |
| 公開 `BackendOps::release_cached_device_memory()` の doc comment を「`Err` は 2 種類」のまま維持する（旧稿） | トリム前 stream 同期失敗・driver トリム失敗の 2 種類のみを doc comment に列挙する | §3.6 (2) の 4 フェーズ設計（post-free sync 失敗を含む）と矛盾し、実装者が post-free sync 失敗を表現・伝播できず fail-closed 契約を破るおそれがある（codex-review PR #1056 P1）。本設計（§3.1・§3.6 (2)）は doc comment を §3.6 (2)「バックエンド別の該当フェーズ」表を参照する形へ改め、CUDA／Metal／CPU 別の該当フェーズを明記した |
| `pending_pool_returns` の会計を `PoolStats` の外に置いたままにする（旧稿） | `cached_bytes`・`release_cached`・LRU 判定はいずれもフリーリストのみを見る設計のままにする | Metal では `Drop` 後 `synchronize()` されるまでのバッファが `cached_bytes`・LRU・`release_cached_device_memory()` のいずれからも不可視になり、`cached_bytes == 0` でも実際にはデバイスメモリが保持され得る（Cursor Bugbot High 指摘）。本設計（§3.1・§3.3・§3.4）は `PoolStats::pending_return_bytes` を新設して可視化し、`max_pool_bytes`／LRU 判定を `cached_bytes + pending_return_bytes` へ拡張し、`release_cached_device_memory()` の Metal 版フェーズ (i)（`synchronize()`）が `pending_pool_returns` を必ず全件フリーリストへ合流させてから後続のフリーリスト走査を行う契約とした |
| Metal フェーズ (i) 共通失敗契約を「フリーリストへ一切触れない」のまま CUDA・Metal 共通にする（旧稿） | `synchronize()`（Metal のフェーズ (i)）が `Err` を返した場合は `pending_pool_returns` の合流を行わず、フリーリストを CUDA と同様に完全に無傷のまま保つ | `synchronize()` は `Ok`／`Err` いずれで復帰する場合も `waitUntilCompleted()` の完了を経ており、GPU 側は対象バッファをこれ以上参照しないため、合流自体を `Err` 時に見送る安全上の理由がない。むしろ合流を見送ると、次回以降の `synchronize()` まで返却待ちバッファがフリーリストへ入らず回収が遅延し続ける（Cursor Bugbot Medium 指摘）。本設計（§3.3・§3.6 (2)）は共通契約を「解放対象としてのフリーリスト走査・取り出しは開始しない」へ narrow 化し、Metal の合流（`put` のみ・`take_one_for_release` を伴わない）はこの契約と両立すると整理した |
| Metal `Drop`／`synchronize()` が `BatchSlots` ロック保持中に直接 `PoolStats::pending_return_bytes` を増減する（旧稿） | `pending_pool_returns` への push／`take` と同一ロック区間内で `SizeClassPool<H>` の統計フィールドも直接書き換える | `SizeClassPool<H>` の統計は内部の別 `Mutex<PoolCore<H>>` で保護されており、`BatchSlots` のロックを保持したままこれに触れるとロック順序がネストし、§3.5 冒頭の「ロックを保持したまま外部呼び出しを行わない」方針・デッドロック回避方針と両立しない（Cursor Bugbot Medium 指摘）。本設計（§3.1・§3.3・§3.5）は `record_pending_return`／`record_pending_merge`（ハンドルを一切受け取らない統計専用メソッド）を新設し、いずれも `BatchSlots` のロックを解放した後に呼ぶ契約とした。ロック解放から統計反映までの短い窓で `pending_return_bytes` が実態と一時的にずれることは許容し明記した |
| `released_bytes`（backend の `release_cached()` が直接更新）を `SizeClassPool<H>` の API 一覧に含めない（旧稿） | `pending_return_bytes` 用の統計専用メソッドのみ追加し、`released_bytes` は backend 側が `PoolStats` の私有フィールドへ直接書き込む想定のままにする | 統計の所有者を `SizeClassPool<H>` に一本化する方針（§3.1）と矛盾し、backend が private 統計を直接触る記述が残ってしまう（codex-review 指摘）。本設計（§3.1）は `record_release(class_bytes)`（ハンドルを一切受け取らない統計専用メソッド）を追加し、`released_bytes` の更新経路も `SizeClassPool<H>` のメソッド呼び出しとして閉じた |
| `record_pending_return`／`record_pending_merge` を `Mutex<PoolCore<H>>` 経由にし `BatchSlots` ロック解放後に呼ぶ（前巡の旧稿） | 統計専用メソッドを他の `Mutex` 系メソッド（`record_allocation` 等）と同じ内部ロックで保護し、`BatchSlots` のロック順序規則（ロックを保持したまま `SizeClassPool<H>` を呼ばない）にそのまま従わせる | `pending_pool_returns` への push／`take`（`BatchSlots` ロックで直列化）と、対応する加算／減算（ロック解放後に個別に呼ばれる）が別々のクリティカルセクションになるため、別スレッドの `synchronize()` が先に `record_pending_merge`（減算）を実行し `pending_return_bytes` が `0` へ張り付いた後に本来先着すべきだった `record_pending_return`（加算）が乗ることで、`pending_return_bytes` が実態より恒久的に高い値のまま戻らなくなる（Cursor Bugbot Medium・codex-review P1 再指摘。旧稿は「表示上の一時的なずれ」と過小評価していたが `max_pool_bytes`／LRU 判定〈§3.4〉を恒久的に壊す実害があった）。本設計（§3.1・§3.5）は `pending_return_bytes` を `SizeClassPool<H>` 内の `AtomicU64` とし、`record_pending_return`／`record_pending_merge` を lock-free な `fetch_add`／`fetch_sub`（`Ordering::Relaxed`）へ再定義し、`BatchSlots` のロック保持中に呼べる例外として扱うことで、push／`take` と加減算を同一クリティカルセクション内に統合し、順序逆転を構造的に排除した |

## §5 検証方法（本文書内の記載の正しさ）

- §3.2 の丸め値は次の規則で計算する: バイト数 `n` に対し、`n` 以下となる
  最大の 2 の冪 `p` を求め、`p`・`1.25p`・`1.5p`・`1.75p`・`2p` の中から
  `n` 以上になる最小の値を丸め先とする。§3.2 表の各行はこの規則で検算
  済み（例: `x`=200,704 B → `p`=131,072（2^17）→ `1.75p`=229,376 ≥
  200,704 が最小の適合値）。
- 本文書が参照するファイル・関数名は origin/main 時点のコード事実として
  §2 表に列挙したもののみを用いる（未マージの #1053 のみ forward
  reference として明示的に区別する）。

## §6 #1020／#1021 への引き渡し・テスト方針・スコープ外

### 6.1 置換箇所一覧

§3.1「ホットパスへの接続点」に記載の `backend-cuda`／`backend-metal` の
各ファイルの直接確保呼び出しを対象とする。

**ドキュメント追従（実装 PR 側の作業）**: `docs/compat-api-scope.md` §0
の確定入口一覧（現在 4 項目）へ、5 項目目として
`fandhe_ai::release_cached_memory(Device)`／
`fandhe_ai::memory_pool_stats(Device)`（§3.1）を追記する。本設計文書
自体は docs-only の設計判断文書であり `docs/compat-api-scope.md` の
更新は対象外とする（実装が確定した時点で追記する）。

### 6.2 テスト方針（#1020／#1021 が実装）

- サイズクラス丸めの単体テスト（境界値・`u64` オーバーフロー）。
- 再利用時ゼロ初期化のテスト（A02 対策の回帰防止。§7）。
- LRU 破棄が `AllocationTracker::allocated_bytes` に正しく反映されること。
- **`SizeClassPool<H>` の所有権遷移の単体テスト（新規。codex-review PR
  #1056 P1 是正。§3.1「不変条件」の回帰防止）**:
  (a) `take(bytes)` → RAII ラッパーを `drop` → 再度 `take(bytes)` で
  **同一ハンドル**（`H` の同一性。テスト用の識別子を埋め込んだダミー
  `H` で検証する）が返ること。
  (b) `take(bytes)` で取り出したハンドルを RAII ラッパーが**保持中**の
  間に、同じクラスへ別スレッド／別呼び出しから `take(bytes)` しても
  貸出中のハンドルは返らない（フリーリストが空なら `None`）こと
  （use-after-return・二重貸出の構造的防止の回帰テスト）。
  (c) `put` は RAII ラッパーの `Drop` からのみ呼ばれ、新規確保直後の
  ハンドルに対しては呼ばれないこと（新規確保パスの単体テストで
  `put` 呼び出し回数が 0 のままであることを確認する）。
- **`PoolStats` フィールド更新契約の単体テスト（新規。codex-review PR
  #1056 P1 是正。§3.1「`PoolStats` フィールド更新契約」の回帰防止）**:
  (a) 新規確保パス（`take` が `None` を返した後、具体アロケータが
  確保 FFI に成功し `record_allocation(logical_bytes, class_bytes)` を
  呼ぶ経路）で `alloc_count` が `+1`、`capacity_waste_bytes` が
  `+(class_bytes − logical_bytes)` されること（`cached_bytes`・
  `reuse_count` は変化しないこと）。
  (b) 上記 (a) のハンドルを `drop`（`record_loan_end` → `put` の順に
  呼ばれる。§3.1「RAII 貸出ラッパー型」）すると、`capacity_waste_bytes`
  が (a) で加算した分だけ減算されて (a) 以前の値に戻り、`cached_bytes`
  が `+class_bytes` されること。
  (c) 上記 (b) の状態から `take(bytes2)`（`bytes2 != logical_bytes`）で
  再利用すると、`reuse_count` が `+1`、`cached_bytes` が `−class_bytes`、
  `capacity_waste_bytes` が `+(class_bytes − bytes2)`（(a) とは異なる
  論理バイト数に基づく値）されること（`capacity_waste_bytes` が
  「現在貸出中のバッファについてのストック量」であり `take` のたびに
  その回の論理バイト数で再計算されることの回帰防止）。
  (d) `release_cached()`（内部メソッド）のフェーズ (ii) で個別解放が
  成功するたびに `record_release(class_bytes)` が呼ばれ
  `released_bytes` が `+class_bytes` されること。
  (e) `release_cached()` の再挿入経路（フェーズ (ii) の防御的失敗
  ハンドリング。§3.6 (2)）では `put` のみが呼ばれ `record_loan_end` は
  呼ばれない（＝再挿入は `capacity_waste_bytes` に影響しない）こと。
- **Metal 返却の GPU 完了待ちテスト（新規。codex-review PR #1056 P1 是正。
  §3.3「Metal」・§3.5「Metal `pending_pool_returns` の排他制御」の
  回帰防止）**:
  (a) **CPU 単体テスト（`backend-metal` の `pending_pool_returns` 判定
  ロジックをハンドル型・`MTLCommandQueue` 等をフェイクにして macOS 実機
  非依存で検証する）**: 「`open`／`committed` バッチが存在する状態で
  `PooledMetalHandle` を `drop`」→ フェイク `SizeClassPool` の `put` が
  **呼ばれない**・`pending_pool_returns` に 1 件追加され
  `PoolStats::pending_return_bytes` が `+class_bytes` されること（同時に
  `record_loan_end` は `Drop` の時点で既に呼ばれているため
  `capacity_waste_bytes` は他のテストケースと同じ規則で即座に減算されて
  いること。§3.1「フィールド更新契約」）。続けてフェイク
  `synchronize()` を呼ぶと `pending_pool_returns` が空になり
  `PoolStats::pending_return_bytes` が `0` に戻り、`cached_bytes` へ
  `+class_bytes` が反映されたうえで `put` が 1 回呼ばれること
  （Cursor Bugbot 指摘「`cached_bytes == 0` のままメモリが保持され得る」
  の回帰防止。Drop→pending→synchronize→`cached_bytes` 反映の一連の
  流れを検証する）。
  (b) 同じ CPU 単体テストで「`open`／`committed` のいずれも存在しない
  状態（in-flight なし）で `PooledMetalHandle` を `drop`」→ `put` が
  **即座に**呼ばれ `pending_pool_returns`・`pending_return_bytes` は
  変化しない（`0` のまま）こと。
  (c) 同じ CPU 単体テストで**フェイク `release_cached_device_memory()`
  を呼ぶ前に必ず `pending_return_bytes == 0`（＝フェイク `synchronize()`
  が呼ばれ済み）であることをアサートする**（§3.3「Metal」の
  「フリーリスト走査を伴う操作を呼ぶ前には必ず `synchronize()` を先に
  完了させる契約」の回帰防止。Cursor Bugbot 指摘「`pending_pool_returns`
  がフリーリスト外に置かれ `release_cached`／`take_one_for_release`
  から見えない」を、プロセスの記述ではなく検証可能なアサーションとして
  固定する）。続けてフェイク `take_one_for_release` を呼ぶと、
  `synchronize()` で合流したエントリを含めてフリーリストから正しく
  取り出せる（＝`release_cached_device_memory()` 実行後に
  `cached_bytes` が `0` になる）ことを確認する。
  (d) **フェーズ (i) 失敗時も合流することのテスト（新規。Cursor
  Bugbot Medium 是正。§3.6 (2)「`Err` の種別」フェーズ (i) 行の回帰
  防止）**: 同じ CPU 単体テストで、フェイク `waitUntilCompleted` が
  実行時エラー（`MTLCommandBufferStatus::Error`）を返すよう設定した
  うえで `open`／`committed` バッチが存在する状態から `synchronize()`
  を呼ぶ。`synchronize()` が `Err` を返すにもかかわらず、
  `pending_pool_returns` が空になり `pending_return_bytes` が `0` に
  戻り `cached_bytes` へ合流分が反映されていることを確認する（＝
  合流はフェーズ (i) 自体の一部であり同期の成否に依存しないことの
  回帰防止）。同時に、この `Err` 復帰後に `take_one_for_release` を
  一切呼んでいない（フリーリストからの取り出しが発生していない）
  ことも確認する（「解放対象としての走査は開始しない」共通契約の
  回帰防止）。
  (e) `#[ignore]` Metal 実機テスト: 未完了バッチ中（`encode` 済み・
  `synchronize()` 未実行）に `PooledMetalHandle` を `drop` したハンドルが、
  `synchronize()` 呼び出し**前**の `take(class_bytes)` では返らない
  （フリーリストが空のまま `None`）ことを確認したうえで、
  `synchronize()` 呼び出し**後**の `take(class_bytes)` では返る
  （use-after-return の実機回帰防止）ことを確認する。
  (f) **`pending_return_bytes` の順序逆転レースの並行回帰テスト
  （新規。Cursor Bugbot Medium・codex-review P1 再指摘対応。§3.1
  「統計専用メソッドの検証」・§3.5「ロック順序規則（精密化）」の
  回帰防止）**: フェイクハンドル・フェイク `BatchSlots` を用いた CPU
  単体テストで、`PooledMetalHandle::Drop`（push＋`record_pending_
  return`）と `MetalContext::synchronize()`（`take`＋
  `record_pending_merge`＋`put`）を複数スレッド（または決定的な
  インターリーブを注入できるテストダブル）から多数回・順序を入れ替え
  ながら実行し、**どの実行順序でも最終的に `pending_return_bytes ==
  0` かつ `cached_bytes` が投入した全エントリの `class_bytes` 合計と
  一致する**ことを確認する（既存の `pooled_memory_integration.rs`
  の係数 2 倍回帰テストと同様、CI で決定的に再現できるようにする）。
  とくに「`Drop` が `pending_pool_returns` へ push した直後・別
  スレッドの `synchronize()` が同じロックを取得できるようになった
  瞬間」を注入できるテストダブル（例: ロック取得直前にバリアで
  同期する）で、`record_pending_merge` が対応する `record_pending_
  return` より先に走る余地が構造的に存在しない（同一クリティカル
  セクション内で対になっているため両者は同一スレッド・同一ロック
  区間でしか実行されない）ことを検証する。旧稿の設計（ロック解放後に
  加減算する方式。§4 該当行）ではこのテストが決定的に失敗しうる
  ケースを構成できたことをコメントに残し、回帰の再発を防ぐ。
- 内部 `release_cached()`（§3.1「2 段構成の命名規約」）が `Ok` を返す
  こと、および実行後に `PoolStats::cached_bytes` が 0 になること。
- **`take_one_for_release`／`put` によるトランザクション型解放の単体
  テスト（新規。codex-review PR #1056 P1 是正。§3.1「解放時の所有権
  遷移」・§3.6 (2) の回帰防止）**:
  (a) **フェーズ (i)（トリム前 1 回目の stream 同期）失敗**を模擬した
  フォールト注入で、`take_one_for_release` が**一度も呼ばれない**
  （＝フリーリストが無傷。`PoolStats::cached_bytes` が解放前と変わら
  ない）まま `Err` が返ること。
  (b) **フェーズ (ii)（個別解放。`take_one_for_release` 後・`put` 前）
  失敗**のフォールト注入（現行 CUDA 実装では発生しない防御的経路。
  §3.6 (2) 注記）で、失敗した `(class_bytes, handle)` が `put` により
  **同じフリーリストへ再挿入**され `PoolStats::cached_bytes` が**その
  分だけ減っていない**こと、かつ以降の未着手エントリもフリーリストに
  残ったまま `Err` が返ること。
  (c) **フェーズ (iii)（全ハンドル drop 後・トリム前の 2 回目の stream
  同期）失敗**を模擬したフォールト注入（Cursor Bugbot 指摘の回帰防止。
  §3.6 (2)）で、全エントリの `take_one_for_release`／drop 自体は完了
  済みのため `PoolStats::cached_bytes` は 0（フェーズ (ii) 完了時点の
  値）のままであり、**いずれのハンドルも `put` で再挿入されない**
  （drop 済みで再挿入対象が存在しないため）まま `Err` が返ること。
  driver トリム（フェーズ (iv)）は実行されない（呼び出し回数 0）ことも
  確認する。
  (d) **フェーズ (iv)（driver トリム）失敗**を模擬したフォールト注入
  （CPU バックエンド等 driver FFI を持たない実装では該当なしのため
  スキップ可）で、フェーズ (ii)・(iii) が全件成功したうえでトリムのみが
  失敗するケースでは `PoolStats::cached_bytes` が 0（フリーリスト解放は
  完了済み）のまま `Err` が返ること。
  (e) 上記 (b) の失敗後に `release_cached()` を再試行すると、
  `take_one_for_release` が**残存分（未着手エントリ）のみ**を対象に
  処理を再開すること（既に解放済みのエントリを二重に処理しない。
  §3.4 の再試行契約の回帰防止）。上記 (c)／(d) の失敗後は再試行対象と
  なる残存エントリ自体が存在しない（フリーリストが既に空）ことも
  確認する。OOM フォールバック（§3.4）がいずれの `Err` も黙殺・panic
  せず `BackendError::DeviceAllocationFailed` へそのまま伝播すること。
- `#[ignore]` 実機テスト（既存 `memory_real_device.rs`／
  `memory_roundtrip.rs` の拡張。`has_async_alloc()` プローブは #1012 の
  該当タスクと共有する）。
- 既存 `pooled_memory_integration.rs` の係数 2 倍回帰テストを新
  `SizeClassPool<H>` 経路（§3.1）へ移植する。
- `#[ignore]` 実機テスト（DGX Spark GB10・`has_async_alloc()` が真の
  環境）: 内部 `release_cached()` 実行後に
  `AllocationTracker::allocated_bytes` が 0 になることに加え、driver 側の
  実メモリ使用量（`nvidia-smi` 等）も併せて確認し、両者に有意な乖離が
  残らないことを確認する（§3.6 の driver トリム契約の回帰防止）。
- `crates/facade/tests/api_surface.rs`（既存のソース走査方式。§3.1 の
  「低水準 trait 自体を tensor-core の公開面へ出さない」不変条件を機械的
  に固定する。codex-review PR #1056 P1 是正）へ、次の 2 種のケースを
  追加する:
  (a) **低水準型の不在確認**: `DeviceAllocator`／`BufferHandle` のいずれも
  `facade`（および `tensor-core` の `pub` 宣言のうち facade が
  再エクスポートする範囲）の公開ソースに出現しないことを検査する。
  (b) **REQ-14 解放 API の到達確認**: `facade` の公開ソースに
  `release_cached_memory`／`memory_pool_stats`（§3.1）が `pub fn` として
  存在し、かつ `PoolStats` が `pub use` で再エクスポートされていること
  を検査する（REQ-14 の解放 API が facade から到達可能であることの
  機械的固定。§3.1「facade からの再公開」）。
- **facade 到達性の機能テスト（新規。REQ-14 是正の受け入れ条件）**:
  (a) CPU（プールなし）: `fandhe_ai::release_cached_memory(Device::Cpu)`
  が `Ok(())` を返し、`fandhe_ai::memory_pool_stats(Device::Cpu)` が
  `Ok(None)` を返すこと。
  (b) CUDA／Metal（`#[ignore]` 実機テスト）: 確保 → drop（プールへ返却）
  → `release_cached_memory(device)` → `memory_pool_stats(device)` の順で
  呼び、`PoolStats::cached_bytes` が 0 になることを確認する。
  (c) CUDA（`#[ignore]` 実機テスト。`has_async_alloc()` が真の環境）:
  stream 同期失敗を模擬したフォールト注入（内部 `release_cached()` の
  単体テストと同じ注入経路を `BackendOps::
  release_cached_device_memory()` 経由で駆動する）で `Err` が
  facade の `release_cached_memory()` までそのまま伝播し、かつ
  `memory_pool_stats()` の `PoolStats::cached_bytes` が「未解放分を含む
  一貫した値」であること（§3.6 (2) の部分失敗時状態一貫性の facade
  到達確認）。

### 6.3 スコープ外（`.claude/rules/out-of-scope-tracking.md` に従い記録のみ。起票はユーザー承認後）

- アップロード経路（`clone_htod`／`new_with_data`）の再利用。
- slab／サブアロケーション（§4）。
- 複数 CUDA ストリームをまたぐバッファ貸し出し。
- CPU バックエンドのホスト `Vec` プール化（別イシュー #1026）。
- driver プール（案 A）の release threshold 調整の実装可否判断（#1020 内
  で判断）。
- Metal `setPurgeableState` によるページ回収ヒント（§3.6 (5)）。

## §7 セキュリティ考慮事項（OWASP Top 10）

- **A02 暗号化の失敗／機微データ露出**: 再利用バッファに前利用者のデータ
  が残留するリスクがある。`alloc_zeroed`（各バックエンドの `pub(crate)`
  具体アロケータ型のメソッド。§3.1）経路は再利用時に必ずゼロ初期化
  （既存 `PoolZeroFill` 相当）を適用する。`alloc_uninit` はカーネルが
  確保領域の全要素を書き切る内部出力専用に限定する。**レビュー指摘の
  変遷（3 段階。§3.1「レビュー履歴」）**: (1) 当初案は「`facade` の
  公開 API へは露出しない」という文書上の注記のみで済ませていたが、
  `DeviceAllocator`（当時は公開 trait）のメソッドとして定義すると
  `BackendOps::allocator()` 経由で外部利用者が直接呼び出せてしまい注記が
  型システムで担保されないと指摘された。(2) `BackendOps::allocator()`
  自体を廃し実装インスタンスを公開関数から返さない設計へ変更したが、
  `pub trait DeviceAllocator` という型定義自体が `tensor-core` の公開面
  （`pub mod allocator`）に残るため、外部コードが本クレートへ直接依存
  すれば型を直接参照・独自実装できてしまう（「実装インスタンスを返さ
  ない」だけでは型の到達可能性を防げない）と再指摘された。(3) 本設計
  （現行。§3.1）は低水準 trait（`DeviceAllocator`）自体を廃止し、
  `alloc_zeroed`／`alloc_uninit` はいずれも各バックエンドクレート内の
  `pub(crate)` 固有メソッドとして実装する。`tensor-core` の公開面には
  ハンドル非依存の `SizeClassPool<H>`（`H` に trait 境界を課さない）と
  POD 型（`PoolConfig`／`PoolStats`）のみが残り、`BufferHandle` を含む
  低水準型は一切 `tensor-core` の公開面に現れない。ホットパスは同一
  クレート内から具体アロケータ型経由で直接呼び出す（§3.1「ホットパス
  への接続点」）。REQ-14 の解放 API は `alloc_zeroed`／`alloc_uninit`
  とは独立に、unit／POD のみを返す `BackendOps::
  release_cached_device_memory()`／`device_memory_pool_stats()`
  （§3.1）として別途確保する。Metal でデバイス側フィル（`fillBuffer`）
  を採用する場合も「貸出前に必ずフィルを encode する」ことを不変条件と
  する（§3.3）。
- **A03 インジェクション（入力検証）**: shape は safetensors／ONNX 経由で
  外部入力が流入しうる。サイズクラス丸め・capacity 計算は `checked_mul`／
  `checked_add`（`u64`）で行い、オーバーフローは型付きエラー
  （`BackendError::DeviceAllocationFailed`）として扱う（既存
  `checked_byte_len` の方針を踏襲。§3.2）。`bytes == 0` は FFI 非経由の
  空ハンドル契約を維持する。
- **A04 安全でない設計（資源枯渇）**: 無制限なプール成長（v1 が教訓とした
  candle の Metal プール無制限成長・17 倍蓄積の事例）を防ぐため、総量
  上限・グローバル LRU・クラス別上限・巨大帯 1 エントリ制限・OOM
  フォールバックを設計に組み込み、REQ-14 の係数上限（2 倍以内）を維持する
  （§3.4・§3.6）。
- **A05 セキュリティ設定ミス**: プール設定は型付き `PoolConfig` のみで
  受け取り、環境変数による無検証の上限上書きは設けない（ガードレール
  閾値と同様、既定値の変更はユーザー承認事項とする）。
- **A06 脆弱・古いコンポーネント**: 新規依存を追加しない。`cuMemPool*`
  系 API は既存 `cudarc =0.19.8` の binding のみを使用し、`Cargo.toml`／
  `Cargo.lock`／`docs/license-matrix.md` は変更しない。
- **A08 ソフトウェア・データ整合性**: 本イシューは docs-only 変更であり
  `docs/spec/`（正本 submodule）は編集しない。数値一致の複合判定・カーネル
  境界検査・許容誤差・REQ-14 の係数上限はいずれも変更しない（緩和は
  ユーザー承認必須）。`unsafe` 追加候補（`cuMemPoolSetAttribute`・
  `cuMemPoolTrimTo`（§3.6 の driver 予約メモリ解放契約）等の FFI・
  `fillBuffer` の encode 呼び出し）は FFI 境界に限定し、理由コメント
  と security-auditor によるレビューを必須とする。
- **A09 セキュリティログ・監視の不足**: `PoolStats`（`AllocatorStats`
  から改称。§3.1）はバイト数・カウンタのみを公開し、デバイスポインタ値を
  ログや統計に含めない。
- **スレッド安全性（メモリ安全上の前提）**: ロックを保持したまま FFI を
  呼ばない（§3.5）・poison 時は panic せず継続する・二重返却を
  `Option<H>::take()` で構造的に防止する（§3.1・§3.3。codex-review PR
  #1056 P2 是正で `ManuallyDrop` 方式から統一した）ことを契約として記す。

## §8 本設計文書で確定する事項／実装イシューで確定する事項

本 PR（#1056）は codex-review・Cursor Bugbot との往復が複数巡にわたり、
各巡の是正が次巡で新たな不整合（trim 前後の同期順序・所有権遷移の粒度・
統計更新経路・ロック順序等）を生む展開になった。以降のレビューで実装
レベルの指摘が出た場合に、本文書のどこまでを「設計判断として動かさない
線」とし、どこからを実装イシュー（#1020・#1021）側の裁量に委ねるかを
文書自身が示すため、区分を明記する。

### 8.1 本設計文書で確定する事項（変更にはユーザー承認・spec 整合の
再確認を要する）

- **公開面**: `tensor-core` に置く POD 型（`PoolConfig`／`PoolStats`）・
  ハンドル型非依存の `SizeClassPool<H>`（§3.1）。低水準 trait
  （`DeviceAllocator` 相当）を公開しないこと。`BackendOps` への追加は
  `release_cached_device_memory()`／`device_memory_pool_stats()` の
  unit／POD 返却 2 メソッドに限ること。`facade` からの
  `release_cached_memory(device)`／`memory_pool_stats(device)` 自由
  関数（§3.1「facade からの再公開」）。
- **所有権不変条件**: 同一ハンドルがフリーリストと貸出中に同時に存在
  しないこと（§3.1「不変条件」）。返却は RAII 限定・明示 `free()` を
  設けないこと（§3.3）。二重返却・use-after-return の構造的防止は
  `Option<H>::take()` 方式に統一すること（§3.1・§3.3。`ManuallyDrop`
  方式は採らない）。
- **REQ-14 の解放プロトコルの段階と失敗時状態**: `release_cached()` の
  フェーズ構成（CUDA 4 段・Metal 2 段。§3.6 (2)「バックエンド別の該当
  フェーズ」表）・各フェーズの `Err` 時のプール状態（フリーリストへの
  影響の有無。§3.6 (2)「`Err` の種別」「部分失敗時の状態一貫性」）・
  フェーズ (i) 失敗時の共通契約（「解放対象としての走査・取り出しは
  開始しない」。フリーリストへの合流〈Metal のみ〉はこれと矛盾しない
  ことを含む）。
- **Metal の返却待ち機構と合流順序**: `pending_pool_returns` による
  GPU 完了待ちの返却遅延（§3.3「Metal」）・`synchronize()` が同期の
  成否に関わらず全件合流させること・フリーリスト走査を伴う操作は
  Metal では必ず `synchronize()` 完了後に行うこと（§3.3・§3.6 (2)）。
- **統計フィールドの意味**: `PoolStats` の各フィールド（`alloc_count`・
  `reuse_count`・`cached_bytes`・`pending_return_bytes`・
  `capacity_waste_bytes`・`released_bytes`）が何を数える／表す量か
  （§3.1「フィールド更新契約」）。とくに `capacity_waste_bytes` が
  「現在貸出中のバッファについてのストック量」であり累積量ではない
  ことと、`pending_return_bytes` が Metal 以外で常に `0` であること。
- **不変事項**（§0 で既述のものを再掲。本設計文書のスコープ内で緩和・
  変更しない）: バックエンド間数値一致の複合判定・カーネル境界検査・
  依存の完全固定・REQ-14 のピークメモリ係数上限（2 倍以内）・ガード
  レール閾値／テスト許容誤差。

### 8.2 実装イシュー（#1020・#1021）で確定する事項（本文書は「案」を
示すに留め、実装時の判断に委ねる）

- **ロック粒度**: `SizeClassPool<H>` 内部の `Mutex<PoolCore<H>>` の
  分割粒度（単一ロックか、フリーリストと統計を別ロックにするか等）。
  §3.5 が定める「`BatchSlots` ロックを保持したまま `SizeClassPool<H>`
  を呼ばない」というロック**順序**の制約は 8.1 の確定事項だが、
  `SizeClassPool<H>` **内部**のロック実装詳細は実装時に決めてよい。
- **具体メソッドの引数形状の微調整**: `record_allocation`／
  `record_loan_end`／`record_release`／`record_pending_return`／
  `record_pending_merge`（§3.1）の引数の型・順序・まとめ方（例:
  `class_bytes`・`logical_bytes` を個別の `u64` にするか小さな構造体に
  まとめるか）。「ハンドルを一切受け取らない統計専用メソッドである
  こと」「どの操作から呼ばれるか」「どのフィールドをどう増減させる
  契約か」は 8.1 の確定事項だが、シグネチャの具体形は実装時に決めて
  よい。
- **統計更新の順序の細部**: 同一イベント内で複数の統計フィールドを
  更新する際の呼び出し順序（例: `record_pending_merge` と `put` の
  呼び出し順）。ロック**外**で行う限りにおいて、この順序自体が
  他の契約（不変条件・fail-closed 状態遷移）に影響しない範囲は実装時の
  裁量とする。
- **エラー種別の表現**: `BackendError::DeviceAllocationFailed(String)`
  の理由文字列によるフェーズ区別（§3.6 (2)）は本設計文書が定める
  **暫定の**非破壊な表現方式である。より型安全な表現（`BackendError`
  への新規 variant 追加等）が望ましいと判明した場合、それは crates.io
  公開済み trait への破壊的変更判断であり **REQ-1／deps-policy.md 系の
  通常のユーザー承認フローに従う**（本文書が先回りして承認するもの
  ではない）。

この区分は本文書のレビュー往復で生じた不整合の再発を防ぐための運用上の
指針であり、REQ・spec の記述そのものを変更するものではない。
