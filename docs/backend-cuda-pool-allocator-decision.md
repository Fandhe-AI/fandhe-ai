# CUDA プールアロケータの採用判断・実測記入欄（#1020）

イシュー #1020「perf(backend-cuda): CUDA プールアロケータの実装とテスト」（親 #1018 ツリー。
上位ゴール #1008「学習ループの固定費削減」）の実装記録。設計の正本は
`docs/device-memory-pool-design.md`（#1019）であり、本文書はその「§8.2 実装裁量」の範囲内で
本イシューが確定した判断と、実装後に得られた実測（実機実測は 2026-08-31・DGX Spark GB10・
イシュー #1025 実装セッションで完了済み）を記録する。

## 1. 採用方式（案 A vs 案 B）

設計文書 §3.3 が示す比較表を踏まえ、以下を採用した。

| 論点 | 案 A（cudarc stream-ordered 確保。`cuMemAllocAsync`/`cuMemFreeAsync`） | 案 B（自作 `SizeClassPool<H>`） | 採用 |
|---|---|---|---|
| 制御性 | driver 任せ（`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` 等の driver 側パラメータに依存） | サイズクラス・LRU 破棄・総量上限を自前で完全制御（REQ-14 14-3 の係数上限〈2 倍以内〉を明示的に守れる） | B |
| 環境依存 | `has_async_alloc()`（`CudaContext::has_async_alloc()`。§4 参照）が偽の環境では素の同期 `cuMemAlloc`/`cuMemFree` に自動退化し効果が消える | 環境非依存（`cuMemAlloc`/`cuMemFree` の上に自前でキャッシュを被せるため、driver 側プールの有無に関わらず一貫して効く） | B |
| REQ-14 統計 | driver 内部の保持量を透過的に観測する API が乏しい | `PoolStats`（`alloc_count`／`reuse_count`／`cached_bytes` 等）を自前で持てる | B |
| 既存 `PooledMemory`（`crate::pool`）との整合 | 無関係 | 同じ「サイズクラス・総量上限・LRU」設計思想を踏襲しつつ、丸めありの別実装として共存（§5 参照） | B |

**結論**: 案 B（自作 `SizeClassPool<CudaSliceHandle>`）を正とする。案 A の driver プール自体は
「能動的には駆動しない」が、`release_cached` のフェーズ (iv) でのみ `cuMemPoolTrimTo` を呼び、
「(has_async_alloc な環境で) driver 側が裏で保持している分の解放も面倒を見る」位置づけとする
（`crates/backend-cuda/src/pool.rs::CudaAllocator::release_cached`）。

`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`（案 A 固有の driver 側パラメータ調整）は**本 PR では行わない**。
実機比較なしに driver 予約量を増やす変更は REQ-14 係数判定に影響しうるため、安全側の判断として見送る
（実機 A/B 後に判断する事項としてスコープ外に残す）。

## 2. `has_async_alloc()` の実装訂正（実装中に判明した事実）

実装計画は「cudarc 0.19.8 は `has_async_alloc()` のような判定ヘルパーを公開していない」という
前提で独自実装（`CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED` を直接クエリ）を予定していたが、
実装時に `cudarc-0.19.8/src/driver/safe/core.rs` を確認したところ **`CudaContext::has_async_alloc()`
が公開 API として既に存在する**ことが判明した（`CudaContext::new` 構築時に 1 回だけ
`CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED` をクエリしてキャッシュした値を返す。doc comment:
「Memory allocations performed through the default `CudaStream` will use `cuMemAllocAsync` over
`cuMemAlloc` if this method returns `true`」）。

この事実により、以下 2 点を訂正した:

- `crates/backend-cuda/src/pool.rs::has_async_alloc(ctx: &CudaContext) -> bool` は独自の属性
  クエリではなく `ctx.has_async_alloc()` へ単純委譲する
- **`stream.alloc_zeros`／`stream.alloc`（cudarc 既存 API）は `has_async_alloc()` が真の環境では
  内部で既に `cuMemAllocAsync`/`cuMemFreeAsync` を使っている**（`core.rs:809,1512,1535` の
  `if ctx.has_async_alloc { ... }` 分岐）。つまり本イシューが追加する `SizeClassPool`（アプリ層
  キャッシュ）は、driver 側の stream-ordered アロケータの**上に重ねて**追加のキャッシュ層を
  提供する構成になる（driver 側が既に速い環境でも、プールヒット時は cudarc 経由の呼び出し自体
  〈`cuMemAllocAsync` の発行コスト〉を完全にスキップできる点が引き続き価値を持つ）。

## 3. 対象 dtype・対象経路の範囲（設計書 §8.2 に委ねられた事項の確定）

- v1 は **f32 出力バッファのみ**対象とする。f16 出力（`gemm_wmma.rs`／`gemm_mma.rs`／
  `gemm_auto.rs` の `alloc_zeros::<f16>`）は直接確保のまま残す（型消去に伴う `unsafe` 追加を
  避ける安全側判断）。
- 対象は `gemm.rs::run_f32_kernel`（naive/tiled 共通）・`run_tiled_bias_act_f32`・
  `elementwise.rs::run_binary`／`run_unary`・`softmax.rs::run_softmax_f32_raw` の出力確保のみ。
  GEMM の WMMA(TF32) 系変種（`run_wmma_tf32`／`_opt`／`_staged`）は本イシューでは対象外とした
  （実装計画が対象を `run_f32_kernel`・`run_tiled_bias_act_f32` に明示限定していたため）。
- `MemoryOps`（`memory.rs::CudaMemory`）経由の `DeviceBuffer` は対象外（`CudaBufferHandle.slice.len()
  == numel` 前提の既存検証〈`ops.rs::gemm_resident_*`・`sgd.rs`〉への影響範囲が広いため）。

## 4. `alloc_uninit` 適用箇所の確認記録（OWASP A02）

カーネルソース（`kernels.rs`・`kernels_elementwise.rs`）を確認し、以下は出力の全要素を必ず
書き切る（`if (row < m && col < n)` / `if (idx < numel)` の書き込みガード内で該当インデックスに
必ず 1 回書き込む。境界外スレッドは何も書かない）ことを確認したため `alloc_uninit_f32` を使う:

- `gemm.rs::run_f32_kernel`（naive/tiled f32）
- `gemm.rs::run_tiled_bias_act_f32`（epilogue 融合。bias/activation も同一書き込みガード内）
- `elementwise.rs::run_binary`／`run_unary`（add/mul/relu/exp/tanh の全 5 カーネル）

一方、`softmax.rs::run_softmax_f32_raw` は persistent grid 方式（グリッドストライドで行を分担する
1 パス／2 パスカーネル）であり、本イシューの時間内でカーネルソースの全網羅性確認を完了できな
かったため、**安全側判断として `alloc_zeroed_f32`（ゼロ初期化を維持）に留めた**（プール接続の
効果〈確保コストの削減〉自体は zeroed でも得られる。全要素書き込みの確認が取れ次第、後続イシュー
で `alloc_uninit_f32` へ切り替える）。

## 5. 既存 `PooledMemory`（`crate::pool`）との関係

変更しない（非推奨化もしない）。`arc_with_non_send_sync` の allow も据え置く。`crate::pool_core`
（本イシューで新設）はハンドル非依存のサイズクラス丸めプール本体であり、`crate::pool` の
バイトサイズ完全一致プールとは別の型・別の関心事として共存する（`crate::pool_core` モジュール
コメント「命名の差異」参照。`pool_core::SizeClassPoolConfig` はクレートルートへ再エクスポートせず、
`PoolStats` のみを再公開する）。**PR #1063 マージ時の追記**: 本ドキュメント作成時点（PR #1061）の
独自 `pool_core` 実装（`PoolConfig` 等の命名）は、#1021（Metal）が独立実装した `pool_core.rs`
との統合時に Metal 側 API（`SizeClassPoolConfig`）へ一本化された（`crates/tensor-core/src/
pool_core.rs` モジュール doc「#1020／#1021 統合時の経緯」参照。`crates/backend-cuda/src/pool.rs`
はこの一本化後の API を呼ぶよう書き換え済み）。

## 6. 実装した公開面（facade 到達経路）

`docs/compat-api-scope.md` §0 の確定入口に以下 2 項目を追加した:

- `fandhe_ai::release_cached_memory(Device) -> Result<(), BackendError>`
- `fandhe_ai::memory_pool_stats(Device) -> Result<Option<PoolStats>, BackendError>`
- `fandhe_ai::PoolStats`（`fandhe_ai_tensor_core::pool_core::PoolStats` の再エクスポート）

いずれも `resolve_ops(device)?.release_cached_device_memory()` / `.device_memory_pool_stats()`
（`BackendOps` の新規デフォルトメソッド。CPU・Metal は既定実装のまま no-op / `None`）への薄い
委譲であり、`DeviceAllocator`／`BufferHandle`／`SizeClassPool` 等のプール実装型は facade から
一切到達できない（`crates/facade/tests/api_surface.rs::facade_does_not_expose_pool_implementation_types`
が機械的に固定する）。

## 7. 実測記入欄（実機実測完了・2026-08-31・DGX Spark GB10・イシュー #1025 実装セッション）

### 7.1 fresh N=2048 固有オーバーヘッド（#956/#1025）への効果

`crates/backend-cuda/src/fresh_overhead_diag_tests.rs::fresh_overhead_diag_v3_pooled_output`
（`#[ignore]`）が (c) C 確保フェーズをプール経由へ差し替えた変種として計測コードを用意した。

**正直な記録（実装計画 AC-3 の必須注記・2026-08-31 実機実測で更新）**: #956/#1025 が特定した
N=2048 固有の約 166〜184 ms は、イシュー #1025 の実装セッション（2026-08-31・HEAD `d6bd4ff`）
で実機実測した結果 **再現しなかった**（fresh N=2048 が reuse N=2048 と同水準。詳細・帰属先の
推定は `docs/perf/cuda-fresh-gemm-n2048-overhead-diagnosis.md` §6〜§7 を参照）。以下は当初の
仮説（本プールが直接効くのは**デバイス側 C 確保**（(c) フェーズ）**のみ**というホスト側解放
（(f3)）非対象の見立て）どおりの実測値である。**166 ms 規模の値が何によって解消していたかは
原因未特定のまま**（同 doc §7・§10 のとおり、#1061〈本プール導入〉・#1077・#1079・#1080 を
候補として挙げたが個々の PR の寄与を分離する追加実験は未実施。#1081〈tape ノードクリア API〉
は同 doc §7 で fresh 経路への直接の関与が確認できないとして候補から撤回済み）現 HEAD では
非再現だった、というのが正確な記述であり、本プール単独の寄与を主張するものではない。

**注意（測定区間が異なる。下表の 2 列を単純比較しない）**: 「GEMM 全体」列は `run_tiled_f32`
（H2D・launch・synchronize・D2H・(c) alloc を含む GEMM 全体、かつプールミス時の 1 回限りの
warmup）、「alloc 単体」列は `alloc_zeroed_f32`（(c) フェーズのみ、プールヒット時）であり
測定対象の範囲が異なる。両者の比を「短縮倍率」として主張することはできない
（`fresh_overhead_diag_v3_pooled_output` の実装参照。詳細は同診断 doc §10 の注意も参照）。

| 環境 | N | `run_tiled_f32`（GEMM 全体・プールミス時 warmup） | (c) `alloc_zeroed_f32` 単体（プールヒット・中央値） | 備考 |
|---|---|---|---|---|
| DGX Spark GB10（2026-08-31 実測。driver 580.173.02・CUDA 13.0） | 1024 | 2.616 ms | 0.002 ms | |
| DGX Spark GB10（2026-08-31 実測） | 2048 | 8.948 ms | 0.003 ms | |
| DGX Spark GB10（2026-08-31 実測） | 4096 | 54.981 ms | 0.003 ms | `release_cached`: `freed_bytes=88080384` |
| RTX 3060（参考値。Linux サーバー・`libcuda` 到達可能であれば） | 1024/2048/4096 | — | — | 正式値ではなく参考値。未実施のまま |

実行コマンド:

```bash
cargo test -p fandhe-ai-backend-cuda --release --lib -- --ignored --nocapture --test-threads=1 \
  fresh_overhead_diag_v3_pooled_output
```

### 7.2 `PoolStats` 恒等式・`release_cached` の実機確認

`crates/facade/tests/memory_pool_api.rs::release_cached_memory_on_cuda_drains_pool_after_gemm`
（`#[ignore]`）が「GEMM 1 回実行 → `release_cached_memory` → `cached_bytes == 0`」を実機で確認する。

| 環境 | 実行結果 | 備考 |
|---|---|---|
| DGX Spark GB10（2026-08-31 実測・イシュー #1025 実装セッション） | pass（`release_cached_memory_on_cuda_drains_pool_after_gemm ... ok`） | |

実行コマンド:

```bash
cargo test -p fandhe-ai --test memory_pool_api -- --ignored --nocapture
```

## 8. スコープ外事項（PR 本文で記録・起票はユーザー承認後）

- f16 出力バッファ（`gemm_wmma`／`gemm_mma`／`gemm_auto`）のプール化
- `MemoryOps`（`CudaMemory`／`DeviceBuffer`）経路のプール化
- 入力アップロード経路（`clone_htod`）の再利用・pinned ステージング
- 複数 CUDA ストリームをまたぐ貸し出し（#1012/#1013 の確定後）
- `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` の調整（実機比較後に判断。
  イシュー #1149 で release threshold 引き上げ・`cuMemAlloc` 同期割当
  との A/B 計測テスト（`crates/backend-cuda/tests/
  large_buffer_percall_alloc_ab_1149.rs`）を追加したが、本エージェント
  の実行環境に CUDA 実機がないため GB10 実測は未完了。実測完了までは
  本保留事項を継続する。実測手順・判定基準は
  `docs/perf/cuda-percall-alloc-pool-threshold-ab.md` を参照）
- `softmax.rs` の `alloc_uninit_f32` 化（persistent grid カーネルの全要素書き込み確認完了後）
- 既存 `PooledMemory` の非推奨化・`arc_with_non_send_sync` allow 解消
