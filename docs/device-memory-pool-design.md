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

### 3.1 層構造と `DeviceAllocator` trait（API 案）

- 配置: `crates/tensor-core/src/allocator.rs`（新規モジュール。`pub mod
  allocator;` として `lib.rs` から re-export）。`MemoryOps`（shape 単位・
  `DeviceBuffer<f32>` を扱う既存の上位抽象）の**下位**に位置する、バイト
  単位のバックエンド非依存プール抽象として新設する。
- 結線: `BackendOps::allocator(&self) -> Option<&dyn DeviceAllocator> {
  None }`（デフォルト実装付き）を追加する。`memory_ops()` と同じ「既存
  trait への非破壊拡張」パターンを踏襲する。crates.io 公開済み trait
  （`BackendOps`）に**必須**メソッドを追加するのは破壊的変更であり、
  `BufferHandle::as_any_mut` 追加時（tensor-core 0.2.0 → 0.3.0）に生じた
  「既存実装者のビルドを壊す」問題と同種の教訓（`crates/tensor-core/src/
  buffer.rs`「破壊的変更」節）から、デフォルト実装で非対応バックエンドの
  ビルドを継続させる。
- trait 案（object-safe・`Send + Sync` を要求する。理由は §3.5）:

```rust
pub trait DeviceAllocator: Send + Sync {
    /// ゼロ初期化保証付きの確保。再利用バッファも必ずゼロ埋めしてから返す
    /// （A02 対策。§6 セキュリティ）。
    fn alloc_zeroed(&self, bytes: u64) -> Result<Box<dyn BufferHandle + Send>, BackendError>;

    /// ゼロ初期化を行わない確保。カーネルが確保領域の全要素を書き切る
    /// 出力専用の内部用途に限定し、`facade` の公開 API へは露出しない
    /// （§6 セキュリティ）。
    fn alloc_uninit(&self, bytes: u64) -> Result<Box<dyn BufferHandle + Send>, BackendError>;

    /// アイドル保持中のバッファを全て解放し、解放できたバイト数を返す
    /// （REQ-14 の明示解放 API。既存 `release_all_pooled` の後継）。
    fn release_cached(&self) -> u64;

    /// 統計（診断・受け入れ条件の検証用）。
    fn stats(&self) -> AllocatorStats;

    /// 現在有効な設定のスナップショット。
    fn config(&self) -> PoolConfig;
}
```

- **capacity と論理長の分離**: `DeviceBuffer` の `shape`（論理 numel）と、
  ハンドル内部の `capacity_bytes`（サイズクラス丸め後の実確保量）を分離
  する契約を導入する。`download` 側・カーネル側は常に論理長のみを読む
  （CUDA は `CudaSlice` の論理長ビュー相当のスライシング、Metal は
  `MTLBuffer::length()` ではなくディスクリプタに保持した論理長引数を使う）。
  これは §2 事実 5 の「完全一致」制約を解除する唯一の前提であり、本設計が
  確定させる中核の変更点である。
- 既存 `PooledMemory`（`crates/tensor-core/src/pool.rs`）との関係: 廃止・
  削除はしない。`DeviceAllocator` 上の薄い互換アダプタとして残すか、
  実装後に非推奨化するかは実装イシュー（#1020）の判断に委ねる。本文書が
  確定するのは「既存公開 API（`PooledMemory`・`PoolZeroFill`）を壊さない」
  という制約のみ。
- ホットパスへの接続点（#1020／#1021 の作業対象。§2 事実 3・4 の箇所）:
  `backend-cuda/src/gemm.rs`・`elementwise.rs`・`softmax.rs` の
  `alloc_zeros` 呼び出しと、`backend-metal/src/gemm.rs`・`elementwise.rs`
  の `new_zeroed` 呼び出しを、`allocator()` 経由（`alloc_zeroed`／
  `alloc_uninit`）へ置き換える。入力側のアップロード経路（`clone_htod`・
  `new_with_data`）は本イシューの対象外とする（`upload_into` の同期契約が
  前提であり、forward 常駐化〈別イシュー〉が進むことで直接確保の頻度は
  自然に減っていく）。

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
- **CUDA**: 単一ストリームモデルを前提とする（#1012 の設計が確定するまでの
  暫定前提）。同一ストリーム上での再利用はストリーム順序により安全（前段
  カーネルの完了前に後続カーネルが同じ領域を書くことはない）。プールは
  `(ordinal, stream)` 単位に持つ。別ストリームへの貸し出しは v1 では禁止
  する（将来対応時はイベント fence を付与する）。ホストからの読み書きは
  既存の `download`（内部で `synchronize()` を伴う）経由に限る。
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
- **Metal**: #1054 §3.4 の契約をそのまま採用する。すなわちプールへ返却
  されたバッファは、in-flight バッチの保持列から外れる（＝そのバッチが
  `synchronize()` される）まで再貸出ししない。ゼロ初期化はホスト書き込み
  （`zero_fill`。#1054 §3.5 が定義する同期点）ではなく、
  `MTLBlitCommandEncoder::fillBuffer` によるデバイス側フィルをバッチへ
  encode する案を推奨する（同期点を新たに増やさないため）。最終選択は
  #1021 に委ねる。
- **二重返却・use-after-return の構造的防止**: 既存 `PooledBufferHandle`
  の `ManuallyDrop` によるガード方式（`crates/tensor-core/src/pool.rs`）を
  継承する。

### 3.4 断片化

- **内部断片化**: 小クラス帯は理論上限 25%（実効はもっと小さい。§3.2 表
  参照）。大クラス帯は 2 MiB 未満／バッファ。`AllocatorStats::
  capacity_waste_bytes`（= Σ(capacity_bytes − 論理バイト数)）で可視化する。
- **外部断片化**: v1 では slab／サブアロケーション（burn/cubecl の
  `SlicedPool` 相当。1 つの大きな確保を複数の論理バッファへオフセット
  分割する方式）を採用しない。`CudaSlice`／`MTLBuffer` のオフセットビュー
  と寿命結合の実装コストが大きく、影響範囲がホットパス全体に及ぶため
  （不採用理由・代替案は §4）。個々のプールエントリは driver／OS 側の
  独立確保であり、外部断片化の管理は driver に委ねる。
- **緩和策**: 総量上限＋グローバル LRU（既存踏襲）・クラス別アイドル
  上限・`release_cached()`・OOM 時は `release_cached()` を 1 回実行して
  から再試行し、それでも失敗すれば `BackendError::DeviceAllocationFailed`
  を返す（fail-closed。無限リトライしない）。

### 3.5 スレッド安全

- プール本体は `Mutex<PoolCore>`（既存方針を踏襲）で保護する。ロックを
  保持したまま FFI 呼び出し（`cuMemAllocAsync`／`newBuffer` 等）を行わない
  （フリーリスト操作のみをロック内で行い、確保が必要な場合はロック解放後
  に FFI を呼ぶ）。poison 時は既存方針（`into_inner` で継続。panic させ
  ない）を踏襲する。
- **`Send`/`Sync` 方針の更新**: `DeviceAllocator: Send + Sync` とし、
  プールが格納するハンドルは `Box<dyn BufferHandle + Send>` を要求する。
  根拠は §2 事実 6・8: `CudaSlice` は `Send + Sync`、Metal の `MTLBuffer`
  protocol も objc2-metal 0.3.2 で `Send + Sync` を supertrait に持つ。
  `crates/tensor-core/src/buffer.rs`「Send/Sync 境界」節が定めた「必要に
  なった時点で再検討する」の条件に、複数スレッド（学習ループのワーカー間
  でのプール共有）から確保・返却を行う本設計で到達したと位置づける。ただし
  `BufferHandle` trait 自体の supertrait は変更しない（公開 trait の
  非破壊）。この変更により `crates/tensor-core/src/pool.rs` の
  `arc_with_non_send_sync` allow は解消できる見込みであり、#1020／#1021 へ
  申し送る。
- プールは device 単位のプロセスワイド singleton とする（`static_cuda_
  memory`／同等の `Box::leak` 所有モデル。§2 事実 2 の計測系列単一化と
  整合させる）。CPU バックエンドは本イシューの対象外（別イシュー #1026 が
  担当）だが、trait 自体は backend 非依存に設計し、CPU 実装が必要になった
  際は恒等実装（`Vec` 確保をそのまま返す）で満たせるようにする。

### 3.6 解放戦略（受け入れ条件）

1. 総量上限（既定 128 MiB を暫定継続）＋グローバル LRU。学習ワーキング
   セットとの関係は #1010 の内訳実測後に再評価する（変更する場合はユーザー
   承認事項）。
2. `release_cached()`（明示解放。REQ-14 の要求する解放 API）。CUDA で
   `has_async_alloc()` が真の環境では、自作プールのフリーリスト解放
   （`CudaSlice` の drop）に加え、driver 側 memory pool（§3.3）に残る
   予約メモリを `cuMemPoolTrimTo(0)` 相当（`result::mem_pool` 経由。
   §2 事実 6）で解放する呼び出しも `release_cached()` の内部契約に含める
   （A/B いずれの設計かに関わらず必須。§3.3 参照）。これにより
   `release_cached()` が REQ-14 の解放 API として自作プール保持分だけで
   なく driver 予約分も含めて解放する契約になる。
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
`release_cached()`／LRU 破棄後も実メモリは driver 側に残存し得るため、
GEMM 4096³ の係数 2.0 判定を `AllocationTracker` の値のみで行うと実際の
ピークメモリを過小評価するおそれがある。したがって本設計では
(i) `release_cached()` が §3.6 (2) の driver トリム呼び出しを含むことで
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

### 6.2 テスト方針（#1020／#1021 が実装）

- サイズクラス丸めの単体テスト（境界値・`u64` オーバーフロー）。
- 再利用時ゼロ初期化のテスト（A02 対策の回帰防止。§7）。
- LRU 破棄が `AllocationTracker::allocated_bytes` に正しく反映されること。
- `release_cached()` 実行後にプール保持分が 0 になること。
- `#[ignore]` 実機テスト（既存 `memory_real_device.rs`／
  `memory_roundtrip.rs` の拡張。`has_async_alloc()` プローブは #1012 の
  該当タスクと共有する）。
- 既存 `pooled_memory_integration.rs` の係数 2 倍回帰テストを新
  `DeviceAllocator` 経路へ移植する。
- `#[ignore]` 実機テスト（DGX Spark GB10・`has_async_alloc()` が真の
  環境）: `release_cached()` 実行後に `AllocationTracker::allocated_bytes`
  が 0 になることに加え、driver 側の実メモリ使用量（`nvidia-smi` 等）も
  併せて確認し、両者に有意な乖離が残らないことを確認する（§3.6 の driver
  トリム契約の回帰防止）。

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
  が残留するリスクがある。`alloc_zeroed` 経路は再利用時に必ずゼロ初期化
  （既存 `PoolZeroFill` 相当）を適用する。`alloc_uninit` はカーネルが
  確保領域の全要素を書き切る内部出力専用に限定し、`facade` の公開 API へ
  は露出しない（§3.1）。Metal でデバイス側フィル（`fillBuffer`）を採用
  する場合も「貸出前に必ずフィルを encode する」ことを不変条件とする
  （§3.3）。
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
- **A09 セキュリティログ・監視の不足**: `AllocatorStats` はバイト数・
  カウンタのみを公開し、デバイスポインタ値をログや統計に含めない。
- **スレッド安全性（メモリ安全上の前提）**: ロックを保持したまま FFI を
  呼ばない（§3.5）・poison 時は panic せず継続する・二重返却を
  `ManuallyDrop` で構造的に防止する（§3.3）ことを契約として記す。
