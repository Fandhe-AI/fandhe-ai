# CUDA `with_host_view` ホストステージング（イシュー #1336）

## 0. 要約

親イシュー #1334（「デバイス結果をホストへ 1 回だけ移し借用で返す」読み出し
API）の #1335（PR #1404）で `tensor-core::MemoryOps::with_host_view` を
新設し、CPU（ホストバッファ直接借用）・Metal（`synchronize` 後
`contents()` 直接借用）を実装した。CUDA は #1335 のスコープ外として
既定実装（`download` 経由の毎回新規 `Vec<f32>` 確保 + D2H）のまま残されて
いた。

本イシュー（#1336）は CUDA 側を実装する: 形状（要素数）ごとに再利用する
ホストステージングバッファへ `memcpy_dtoh` 1 回で D2H し、そのスライスを
呼び出し元クロージャへ借用として渡す。実装は完了し GPU 非依存の単体
テスト・受け入れ条件を検証する `#[ignore]` 実機テストを整備したが、
**本エージェント実行環境に CUDA 実機（DGX Spark GB10 等）がないため
D2H＋読み出し時間の before/after 実測（R3）は未完了**。既定の
`HOST_STAGING_KIND` は unsafe 経路（`Pinned`）を通さない安全側
（`Pageable`）に固定してある。

## 1. 背景（実測根拠）

`docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`
（§4.2・§6）が実測した「大容量バッファ per-call アロケーション＋転送」の
フェーズ分解によれば、D2H の宛先を**毎回新規確保**する構成（P4）のみが
31→32 MiB で 24〜33 倍の段差（glibc `M_MMAP_THRESHOLD` 動的上限）を示す
一方、**事前タッチ済み再利用 `Vec`（P5）は線形・段差なし**である。現状の
`MemoryOps::with_host_view` の CUDA 既定実装（`download` 経由）はまさに
P4 相当（`CudaStream::clone_dtoh` が毎回新規 `Vec<f32>` を確保）であり、
本イシューはこれを P5 相当（事前タッチ済み再利用バッファ）へ置き換える。

## 2. 設計

### 2.1 モジュール構成

- `crates/backend-cuda/src/host_staging.rs`（新規・非公開 `mod`）:
  `HostStagingCache`（形状ごとに再利用するホストバッファのキャッシュ。
  take/put 方式）・`HostStaging`（`Pinned`／`Pageable` の 2 種）・
  `HostStagingKind`・`HostStagingStats`（診断用スナップショット）。
- `crates/backend-cuda/src/memory.rs`: `CudaMemory` に
  `host_staging: Arc<Mutex<HostStagingCache>>` フィールドを追加し、
  `MemoryOps::with_host_view` を上書き実装。`take_or_alloc_staging`／
  `return_staging`（`host_staging` へのロック境界を限定するヘルパー）・
  `host_staging_stats`（`internal-diagnostics` feature 限定の診断
  アクセサ）・`release_host_staging`（REQ-14 型の明示解放 API。常時
  公開）を追加。

### 2.2 種別（`HostStagingKind`）

| 種別 | 実装 | unsafe | 特性 |
|------|------|--------|------|
| `Pageable`（既定） | 事前タッチ済み `Vec<f32>` | なし | 確保コストは通常の `Vec` 確保と同じ。ホスト読み出しは通常速度 |
| `Pinned` | cudarc `CudaContext::alloc_pinned`（`CU_MEMHOSTALLOC_WRITECOMBINED` 固定・page-locked） | **1 箇所**（`HostStaging::alloc` 内の `ctx.alloc_pinned::<f32>(numel)` 呼び出し） | GPU 側の DMA 転送は高速だが、WRITECOMBINED メモリは CPU 側の読み出しが著しく遅いことが知られている（cudarc-0.19.8 `core.rs:1406-1427` のドキュメンテーションコメント）。D2H 単体では有利でも「D2H＋ホスト読み出し」の合計で `Pageable` に劣る可能性があり、決め打ちしない |

`docs/backend-cuda-managed-placement-decision.md` が既存の設計判断で
「pinned host memory は対象外」としていた領域に本イシューで対応した
（`crates/backend-cuda/src/host_staging.rs` モジュール冒頭コメント参照）。

`cudarc-0.19.8` にはキャッシュ可能な pinned（`CU_MEMHOSTALLOC` フラグ
0）を得る手段として `driver::result::malloc_host` があるが、これは
`HostSlice` トレイトの実装・RAII 型の自作（`malloc_host`／`free_host`／
`from_raw_parts` を扱う unsafe 面の拡大）を要するため、本イシューでは
実装せず、実測で `Pinned`（WRITECOMBINED）が有望であった場合の後続候補
としてのみ記録する（ユーザー承認が必要な範囲）。

### 2.3 ロック方針（take/put）

`HostStagingCache` の `Mutex` は「データ構造操作のみ」の間だけ保持し、
`memcpy_dtoh`・`synchronize`・呼び出し元クロージャ `f` の実行中は保持
しない。これにより `f` の内部で別バッファの `with_host_view` を再入
しても deadlock しない（同一 numel の再入は新規確保で対応する）。

新規取得側（`take_or_alloc_staging`）は poison を fail-closed に拒否する
一方、返却側（`return_staging` → `host_staging::put_back`）は poison 後も
`PoisonError::into_inner` で panic せず返却を試みる（以降のアクセスは
`take_or_alloc_staging` 側の poison 検査が引き続き拒否する）。

### 2.4 世代検査

`context_cache::invalidate`（ordinal の poison 回復）はストリームの世代を
進めるが `CudaMemory` 自体は再生成されないため、ステージングエントリは
確保時点の ordinal 世代を刻印し、`take` 時に世代不一致なら破棄して
miss 扱いにする（fail-closed。イシュー #1013 設計文書 §9 item 7 と同じ
方針）。

### 2.5 分岐（`with_host_view` 本体）

1. `handle.storage == None`（空テンソル）: FFI を呼ばず `f(&[])`。
2. `CudaStorage::Managed`: `host_view_managed`（新設ヘルパー）へ委譲し、
   `UnifiedSlice::as_slice()` の借用をコピーなしでそのまま `f` へ渡す
   （`download` が `to_vec()` するのと異なり、managed 配置本来のゼロ
   コピー特性を保つ）。`host_staging` キャッシュは経由しない。
3. `CudaStorage::Device`: `host_staging` から確保・`memcpy_dtoh` で
   D2H・`synchronize` の後、`f` へ借用を渡す。使用後は成功・失敗
   いずれの経路でも `return_staging` でキャッシュへ返却する。

同期契約は `download`（`download_inner`）と同一で、`with_driver_call`
（poison／世代状態機械）を唯一の driver 呼び出し境界とする。

## 3. unsafe（1 箇所・security-auditor レビュー対象）

`crates/backend-cuda/src/host_staging.rs::HostStaging::alloc` の
`HostStagingKind::Pinned` 分岐のみ:

```rust
let pinned = unsafe { ctx.alloc_pinned::<f32>(numel)? };
```

- **理由**: `cudarc::driver::CudaContext::alloc_pinned` が unsafe な唯一の
  理由（cudarc-0.19.8 `core.rs:1410-1411`）は、返す `PinnedHostSlice` の
  内容が確保直後は不定であること。
- **安全性根拠**: 本関数はキャッシュ miss 時にのみ呼ばれ、呼び出し元
  （`memory.rs::CudaMemory::with_host_view`）は返ったバッファを呼び出し元
  クロージャ `f` へ渡す前に必ず `CudaStream::memcpy_dtoh` で要素数ぶん
  全域を D2H 上書きしてから `as_slice()` で読み出す（未初期化内容が外部へ
  露出する経路は存在しない）。要素型は `f32` であり全ビットパターンが
  有効な浮動小数点表現になるため（`cudarc::driver::ValidAsZeroBits` が
  `f32` に実装済み）、未初期化ビット列を無効値として解釈する余地もない。
  `memory.rs::alloc_zeroed_inner` の managed 確保分岐（`alloc_unified` +
  `memset_zeros`）・`pool.rs::CudaAllocator::alloc_uninit` と同一クラスの
  安全性根拠である。
- **既定では通らない**: `HOST_STAGING_KIND = HostStagingKind::Pageable`
  （既定）のため、本番経路は通常この unsafe ブロックへ到達しない。
  `#[ignore]` 実機テスト（`tests/host_view_real_device.rs`）でのみ
  `Pinned` 種の確保が実行される。

## 4. テスト

### 4.1 GPU 非依存単体テスト（`host_staging.rs::tests`・`memory.rs::tests`）

- `HostStagingCache::take`／`put` の hit／miss・世代不一致破棄・長さ
  不一致破棄・cap 超過破棄・`release_all`。
- `HostStaging::Pageable` の直接構築が bit 完全一致で読み出せること。
- Mutex poison からの復旧（`put_back` は panic しない／`take` 直接呼び出し
  経路は poison を観測できる）。
- `CudaMemory::with_host_view` のハンドル型不一致・device ordinal 不一致
  → `DeviceMismatch`（呼び出し元クロージャは呼ばれない）。
- 空バッファ（`numel == 0`）→ FFI を呼ばず空スライスを渡す。
- `CudaMemory: Send + Sync` の静的検査。

いずれも `cargo test -p fandhe-ai-backend-cuda --lib --all-features` で
green（本エージェント実行環境で確認済み）。

### 4.2 `#[ignore]` 実機テスト（`tests/host_view_real_device.rs`）

- `with_host_view` が `download().as_slice()`／`upload` の元データと
  NaN／inf／`MIN_POSITIVE` を含めて bit 完全一致すること。
- 同一形状の複数回呼び出しで `host_staging_stats().hits` が増加すること
  （キャッシュ再利用の確認）。
- `release_host_staging` がキャッシュを空にし、以降の呼び出しが miss から
  再開すること。
- `PooledMemory<CudaMemory>` 経由の `with_host_view` が透過的に動作し
  `download` と bit 同一であること。
- managed 配置（`crate::placement::set_managed_placement_enabled(true)`）
  での `with_host_view` が `download` と bit 同一であること（デバイスが
  managed memory 非対応の場合はスキップ）。

実行コマンド:

```sh
cargo test -p fandhe-ai-backend-cuda --release --all-features \
    --test host_view_real_device -- --ignored --nocapture --test-threads=1
```

## 5. 実機実測（R3・未実施）

**本エージェント実行環境に CUDA 実機（DGX Spark GB10 等）がないため、
D2H＋読み出し時間の before/after 5 回計測中央値は未実施のまま記入欄のみ
残す**（`docs/cuda-gemm-vjp-transposed-entry.md`・`docs/perf/
linear-forward-device-gpu.md` の CUDA 実測欄と同じ先例に従う）。

### 5.1 実行手順（実施時に埋める）

1. `docs/real-hardware-verification-env.md` §3・§4・§6 に従い DGX Spark
   へ rsync 転送する。
2. 計測前後に `nvidia-smi` で利用率 0%・`uptime`／load average を
   `docs/perf/logs/cuda-host-view-staging-1336/env_info.txt` へ記録する
   （内部ホスト名は書かない）。
3. `#[ignore]` 実機テスト（4.2 節）を実行し bit 同一を確認する。
4. D2H＋読み出し時間の A/B（`before` = 既定実装相当の `download`＋読み出し
   ／`after-Pageable`／`after-Pinned`）を N=1024/2048/4096（4/16/64 MiB）
   で計測する。各 run = 20 warmup + 20 計測の中央値、5 run の中央値を
   採用する。

### 5.2 実測値（未実施）

| N | before（`download`＋読み出し・ms） | after `Pageable`（ms） | after `Pinned`（ms） | 判定 |
|---|---|---|---|---|
| 1024 | 未実測 | 未実測 | 未実測 | 未実測 |
| 2048 | 未実測 | 未実測 | 未実測 | 未実測 |
| 4096 | 未実測 | 未実測 | 未実測 | 未実測 |

env_info（`uptime`／load average・実行時間帯）: 未実測。

## 6. 採否

- **本番既定は `HostStagingKind::Pageable`（unsafe 経路を通さない安全側）
  のまま維持する**。5 節の実測が完了し、`Pageable` を一貫して上回る種が
  確認できた場合にのみ、ユーザー承認を経て `HOST_STAGING_KIND` を切り替
  える（`.claude/rules/security.md`「unsafe は必要最小限」・本番結線の
  事前承認方針〈性能低下の可能性は前後比較を記録〉と整合させるため、
  実測なしの `Pinned` 既定化は行わない）。
- 実装自体（`HostStagingCache`・`with_host_view` の 3 分岐・GPU 非依存
  テスト・`#[ignore]` 実機テスト）は完了しており、既定 `Pageable` 種の
  下でも #1146 が示した「事前タッチ済み再利用 `Vec`」の段差回避効果は
  機構として反映されている（実測なしに効果量は主張しない）。

## 7. 引き継ぎ（対象外事項。ユーザー承認・別イシュー起票が必要）

1. **facade からの到達経路が現状存在しない**: `autodiff`／`facade` は
   `MemoryOps::with_host_view` を呼んでいない。`Var::matmul` の出力は
   `gemm` 内部の `readback`（`gemm.rs::run_f32_kernel`）で既にホスト
   常駐 `Tensor` になっており、`Var::host_view` は D2H を伴わない
   （`docs/public-api-design.md` §6 項目 10）。よって本イシューの CUDA
   上書きの直接の受益者は `DeviceBuffer<f32>` を保持する経路
   （`autodiff/src/optim/device_store.rs` の `mem.download(..)` 呼び出し・
   resident 系）であり、framework-compare の gemm reuse 計測（イシュー
   #1337 等）では本経路を通らない。
2. **`gemm` 内 `readback` のステージング化**: `gemm.rs::run_f32_kernel` の
   `readback` ヘルパー自体を `host_staging` 相当のキャッシュへ結線する
   ことは自然な後続候補だが、本イシューのスコープ外（自動運転のため
   Issue 起票はしない。将来 issue 化を検討）。
3. **キャッシュ可能 pinned（フラグ 0）**: `driver::result::malloc_host` +
   自作 `HostSlice` 実装は unsafe 面が広がるため本イシューでは実装しない
   （2.2 節）。実測で `Pinned`（WRITECOMBINED）が有望だった場合のみ
   ユーザー承認を得て検討する。
4. **5 節の実機実測そのもの**: 本エージェント実行環境に CUDA 実機がない
   ため未実施。DGX Spark GB10 実機にアクセス可能な環境で 5.1 節の手順を
   実行し、5.2 節の表・6 節の採否判断を更新する。
