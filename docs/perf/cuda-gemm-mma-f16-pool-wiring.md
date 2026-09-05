# `CudaMmaGemm::run_f16` の SizeClassPool 結線・before/after 計測（#1153）

イシュー #1153「`gemm_mma.rs` の `clone_htod`／`alloc_zeros` 直呼びを
`SizeClassPool` 経由へ結線する（A/B で有効な場合）」の実測記録。親
#1130（大容量バッファ per-call アロケーション病態の調査ツリー）配下。

## 状態

**実測完了・判定確定: 本番結線は見送り（G-1 相当）**。`pool.rs`／
`gemm_mma.rs` の実装（S-1 dtype 一般化・S-2 pooled API）は完了し
`internal-diagnostics` feature 限定の診断入口として残すが、
`CudaMmaGemm::run_f16`（本番経路）はプール API へは結線しない。GB10
実機での複数の A/B 計測（別バイナリ比較・同一バイナリ交互実行の両方）
で、dim4096（本経路の主要形状）においてプール経由化が生 API より
明確に不利であることを確認したため。

## 1. 背景

- #1146（`docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`）
  の結論: 約 260 ms・二峰性の主因は**ホスト側**（D2H 宛先 `Vec` の毎回
  新規確保・未タッチ → glibc mmap しきい値 32 MiB）。デバイス側で
  未解明なのは 32→33 MiB の P1/P2/P3（確保・`alloc_zeros`・H2D）約 2.2
  倍の段差のみ。
- #1149（`docs/perf/cuda-percall-alloc-pool-threshold-ab.md`）は上記段差
  に対する案 A（release threshold 引き上げ）・案 B（`cuMemAlloc` 同期
  割当）の A/B ハーネスを整備し、本イシューの Phase 0 として GB10 実機
  実測を完走させた。**孤立マイクロベンチマーク（P1〜P3）では案 A が
  段差を解消したが、`CudaMmaGemm::run_f16` の現実的なレプリカ（P7）では
  同じ案 A が median を約 10 倍悪化させるという、正反対の結果が出た**
  （§5 参照。unified memory 環境での物理メモリ競合と推定）。この結果を
  受け、driver 側の release threshold 調整（案 A）は不採用と確定した
  （`docs/perf/cuda-percall-alloc-pool-threshold-ab.md` §7）。
- 本イシューは、上記 driver 設定とは独立の**アプリ層**対策として、
  `gemm_mma.rs::CudaMmaGemm::run_f16` の per-call 確保（`upload_f16`／
  `alloc_output_f16` の `clone_htod`／`alloc_zeros::<f16>` 直呼び）を、
  自作 `SizeClassPool`（`pool.rs`。#1020 で f32 出力バッファ向けに導入
  済み）経由の確保へ置き換える段。

### 症状と対策の対応関係（過大な効果を主張しないための整理）

本イシューが対象とする「プール経由化」はデバイス側の per-call 確保
（`alloc`／`alloc_zeros`）を無くす施策であり、#1146 が特定したホスト側
D2H 宛先バッファの再利用（`run_f16 -> Vec<f16>` という API 形状に依存
する別施策）とは独立である。よって本イシューの結線だけで #1130 の元
症状（約 260 ms・二峰性）が解消するとは主張しない。判定は「結線後が
結線前に対し非後退（5 回計測中央値）」を基準とする（実装計画の
事前登録基準）。

## 2. 実装（完了。ただし `run_f16` へは未結線）

- `crates/backend-cuda/src/pool.rs`: `CudaSliceHandle` を dtype 付き
  `enum`（`F32`／`F16`）へ拡張し、`PoolDtype` トレイト（`f32`／`f16`
  の 2 impl のみ）で `PooledCudaHandle<T>` を generic 化した。**第 2 の
  プールは作らない**（`max_pool_bytes` 上限を実質 2 倍にしないため）。
  f32 のみのワークロード（既存の公開経路 6 箇所）は再型付けゼロの
  既存経路を通り続ける（詳細は `docs/backend-cuda-pool-allocator-
  decision.md` §3.1・§4.1）。`upload_f16`／`alloc_uninit_f16` を追加。
- `crates/backend-cuda/src/gemm_mma.rs`: `CudaMmaGemm` に
  `#[cfg(feature = "internal-diagnostics")] allocator: Arc<CudaAllocator>`
  フィールドを追加（`context_cache::cached_allocator` 経由）。カーネル
  起動実装を `launch_f16_views`（view ベース）へ集約し、raw API
  （`upload_f16`／`alloc_output_f16`／`launch_f16`／`download_f16`。
  常時公開・**本番経路**）とプール API（`upload_f16_pooled`／
  `alloc_output_f16_pooled`／`launch_f16_pooled`／`download_f16_pooled`。
  `pub`・`internal-diagnostics` feature 限定・**診断専用**）の両方から
  共有する。プール API の戻り値は `pool.rs::PooledF16Buffer`（`pub`
  不透明ラッパー。`PooledCudaHandle<f16>` 自体は `pub(crate)` のため
  `private_interfaces` lint を避ける薄い wrapper）。
- **`run_f16`（本番の唯一の入口）はプール API へ結線しない**。§5〜§7 の
  実測が根拠。

## 3. 計測方法

`examples/gemm_mma_f16_pool_wiring_bench.rs`（新規）が `CudaMmaGemm::
run_f16` と同一の呼び出し列（転送込み: H2D + カーネル起動 + D2H）を
`bench_harness::protocol::run`（warmup/計測 20 回以上）で計測する。
`--pooled` フラグ（`internal-diagnostics` feature 必須）で raw／pooled
の 2 経路を**同一バイナリ内で**切り替えられる（§6 の「同一バイナリ
交互実行」で使用）。

対象形状: 512／1024／2048／4096（正方形。イシュー #1153 実装計画の
受入基準）。

## 4. 環境

`docs/real-hardware-verification-env.md` §2〜§4・§6 準拠（内部ホスト名
は書かない）。

| 項目 | 値 |
|---|---|
| GPU | DGX Spark GB10（sm_121） |
| CUDA / rustc 版 | CUDA 13.0 系・rustc 1.97.0 |
| 検証対象コミット（before・別バイナリ比較） | `2250dce`（結線前・origin/main） |
| 検証対象コミット（after・別バイナリ比較） | イシュー #1153 作業ブランチの中間コミット（`run_f16` がプール API を呼ぶ実装時点） |
| 検証対象コミット（同一バイナリ比較） | イシュー #1153 作業ブランチ HEAD（`--pooled` フラグで raw／pooled 切替） |
| 実行日 | 2026-09-05 |
| GPU 占有状況 | 計測前後とも `nvidia-smi --query-gpu=utilization.gpu` 0% |

## 5. 実測結果

### 5.1 512／1024／2048（32 MiB 閾値未満。安定した比較が可能）

複数回計測（各 4 run 以上）の median（別バイナリ比較。ms）:

| size | before (raw) | after (pooled) | 比 |
|---|---|---|---|
| 512 | 0.0655〜0.0663（≈0.066） | 0.0591〜0.0599（≈0.0596） | **約 10% 改善** |
| 1024 | 0.1852〜0.1855（≈0.1853） | 0.1779〜0.1792（≈0.1786） | **約 3.6% 改善** |
| 2048 | 0.7905〜0.7969（≈0.793） | 0.7643〜0.7669（≈0.7656） | **約 3.5% 改善** |

これら 3 サイズは全 run で一貫して改善しており（run 間ばらつきが小さく
signal が明確）、プール経由化がデバイス側確保コストを確実に削減する
ことを裏付ける。生ログ: `docs/perf/logs/cuda-gemm-mma-f16-pool-1153/
run_f16_before_after_multi_binary.log`。

### 5.2 4096（32 MiB。D2H 宛先が glibc mmap しきい値と一致）

同サイズは #1146／#1149 が特定した「D2H 宛先フレッシュ `Vec` 確保が
確率的に約 10 倍遅くなる」二峰性の影響を強く受ける（プール経由化の
対象外のホスト側要因）。**別バイナリでの before/after 比較では
after（pooled 結線時点のバイナリ）が明確に不利**（before 中央値
median-of-medians ≈ 21 ms・4/15 run が slow〈約 27%〉、after ≈ 271 ms・
12/14 run が slow〈約 86%〉）だったが、**別バイナリ比較にはバイナリ
レイアウト差という交絡因子が残る**ため、同一バイナリでの追試を行った
（§5.3）。

### 5.3 同一バイナリでの raw／pooled 交互実行（交絡因子の排除）

`--pooled` フラグで raw/pooled を切り替えられる単一バイナリを用い、
`raw,after,raw,after,...` の順で 12 ペア（2 バッチ）を交互実行した
（GPU 占有 0% を確認済み）。

| mode | fast run（<50 ms） | slow run（>250 ms） | fast 率 |
|---|---|---|---|
| raw | 6 | 6 | 50% |
| pooled | 2 | 10 | 17% |

生ログ: `docs/perf/logs/cuda-gemm-mma-f16-pool-1153/
run_f16_same_binary_interleaved.log`。

**同一バイナリでも pooled が slow パスへ入る頻度が明確に高い**
（raw 50% vs pooled 17%）。これにより §5.2 の「後退」は単なるバイナリ
レイアウトの交絡ではなく、pooled 経路自体の性質に起因することが確認
できた。根本メカニズムは未特定だが、有力な仮説は次のとおり: raw 経路
は毎回 32 MiB のデバイスバッファを確保・解放する（`cuMemAllocAsync`/
`cuMemFreeAsync`）のに対し、pooled 経路は同一デバイスバッファを
`SizeClassPool` 内に保持し続けて再利用する。GB10 の unified memory
環境では、デバイス側バッファの保持パターンの違いが、何らかの経路で
ホスト側 D2H 宛先 `Vec` 確保時の glibc mmap しきい値挙動（動的に調整
される `M_MMAP_THRESHOLD`）に影響しうる（driver プール release
threshold 引き上げ〈#1149 案 A〉が同じ症状を引き起こしたことと整合的
——**2 つの独立したメモリ保持系の変更〈driver 側 release threshold・
アプリ層 SizeClassPool〉が、いずれも dim4096 で同じ方向の悪化を示した
ことは、単一の偶然ではなく GB10 の unified memory アーキテクチャに
起因する体系的な効果を示唆する**）。定量的な根本原因の特定（GPU
counters・メモリ帯域計測等）は未実施（§8）。

## 6. 実行手順

```sh
# 生 API（run_f16 の本番経路と同一）
cargo run -p fandhe-ai-backend-cuda --example gemm_mma_f16_pool_wiring_bench --release

# プール API（internal-diagnostics feature 必須。同一バイナリで比較する場合は
# 一度このビルドを作り、--pooled の有無だけを変えて実行する）
cargo build -p fandhe-ai-backend-cuda --release \
    --example gemm_mma_f16_pool_wiring_bench --features internal-diagnostics
target/release/examples/gemm_mma_f16_pool_wiring_bench            # raw
target/release/examples/gemm_mma_f16_pool_wiring_bench --pooled   # pooled
```

## 7. 判定

事前登録基準（実装計画）: 「全 dim で after の中央値が before 以下
（同等含む）」を満たせば ADOPT、いずれかの dim で 5% 超の後退があれば
差し戻し。

- 512／1024／2048: 全て改善（§5.1）。
- 4096: 別バイナリ比較・同一バイナリ交互実行のいずれでも pooled が
  slow パスへ入る頻度が raw より明確に高い（§5.2・§5.3）。中央値
  そのものの差は run 選択に依存し不安定だが、**分布として悪化して
  いることは同一バイナリ比較で確認済み**であり、5% を大きく超える
  後退と判断する。

**判定: `run_f16` へのプール API 結線は見送る（差し戻し。G-1 相当）**。
`upload_f16_pooled`／`alloc_output_f16_pooled`／`launch_f16_pooled`／
`download_f16_pooled` は `internal-diagnostics` feature 限定の診断専用
入口として実装を残し（`new_with_swizzle` 等と同じ位置づけ）、`run_f16`
は生 API（`upload_f16`／`alloc_output_f16`／`launch_f16`／
`download_f16`）のまま維持する。512〜2048（32 MiB 未満の形状）では
プール経由化に明確な利点があるため、**形状条件付きでの本番結線**
（dim4096 のみ raw・それ未満はプール経由）は後続イシューの検討候補と
して引き継ぐ（§8）。

案 A（driver release threshold）・本イシューのアプリ層 `SizeClassPool`
という独立した 2 つのメモリ保持最適化が、いずれも dim4096 で同方向に
悪化したという知見は、GB10（unified memory）上でのメモリ保持系最適化
全般に対する警鐘として `docs/backend-cuda-pool-allocator-decision.md`
§3.1 にも記録する。

## 8. スコープ外・引き継ぎ（PR 本文にも記載。起票はユーザー承認後）

- **形状条件付きでのプール経由化**（512〜2048 は pooled・4096 のみ
  raw。§7 で示した利点を活かす設計。閾値の正確な特定〈どこから悪化に
  転じるか〉が前提）
- dim4096 で pooled が slow パスへ入りやすくなるメカニズムの定量解明
  （GPU counters・メモリ帯域計測・glibc `M_MMAP_THRESHOLD` の実際の値
  の追跡等）
- D2H 宛先ホストバッファの再利用（#1146 §7 の最優先候補）: `run_f16 ->
  Vec<f16>` の契約変更（呼び出し元提供／常駐宛先・pinned）を要するため
  別イシュー
- `gemm_wmma.rs`（`:256-260,303-313`）・`gemm_auto.rs:671-673`
  （Dynamic 経路）への同型結線（今回の知見〈dim4096 で悪化しうる〉を
  踏まえ、結線前に同じ before/after 計測を要する）
- 案 B（`cuMemAlloc` 同期割当）のプール**ミス経路**への適用（cudarc
  0.19.8 の `CudaSlice::Drop`〈`has_async_alloc` の真偽で `free_async`／
  `free_sync` を自身で選ぶ〉API 不一致問題があるため所有型の新設が前提。
  `docs/perf/cuda-percall-alloc-pool-threshold-ab.md` 参照）
- `MMA_PRIORITY_PRODUCTION_ENABLED` はイシュー #1191 で有効化済み
  （`docs/perf/cuda-gemm-auto-f16-mma-switch.md` §0 の baseline ceiling
  承認・反映を経て `true` へ復帰）。これにより本イシューの結線可否
  判断の本番実効範囲（`CudaGemmAuto::run_f16` からの到達）が有効化
  された（cc>=8.0・整列形状・`mma` 構築済みなら `gemm_auto.rs` は
  `CudaMmaGemm` へ到達する）

## 9. 関連ファイル

- `crates/backend-cuda/src/pool.rs`（dtype 一般化・`PooledF16Buffer`）
- `crates/backend-cuda/src/gemm_mma.rs`（pooled API・feature ゲート）
- `crates/backend-cuda/examples/gemm_mma_f16_pool_wiring_bench.rs`（本
  イシュー計測バイナリ。`--pooled` フラグ）
- `docs/backend-cuda-pool-allocator-decision.md` §3.1／§4.1
- `docs/perf/cuda-percall-alloc-pool-threshold-ab.md`（Phase 0 実測完了。
  案 A〈release threshold〉不採用の判定根拠）
- `docs/perf/logs/cuda-gemm-mma-f16-pool-1153/`

## 10. 受入基準との対応

| 受入基準 | 対応 |
|---|---|
| #1149 で有効と確認された対策を `gemm_mma.rs` へ実装する | #1149（案 A）は不採用のため対象外。本イシュー独自のアプリ層 `SizeClassPool` を実装（§2）、before/after 計測の結果、本番結線は見送り（§7） |
| 変更後にバックエンド間数値一致回帰（REQ-2 複合判定）が通過する | `cpu_cuda_mma_parity`・`parity_nonregression`・`gemm_auto`・`dispatch_boundary`・`tensor_core_real_device`・facade `memory_pool_api` を GB10 実機実行。既知の pre-existing fail（`mma_f16_k4096_stress`・`tensor_core_parity_record`・`tensor_core_tflops_record`）は before/after で fail_count まで完全一致し非後退を確認 |
| GB10 で対策前後を複数回計測し病態の解消／緩和を確認して `docs/perf/` へ記録する | §5〜§7 |
| 本番結線は数値一致通過後（承認は事前済み。tolerance/baseline は不変） | 数値一致は確認済みだが、性能実測の結果 `run_f16` への結線は見送り |
| 有効な対策が確認できなかった場合は理由を記録しコード変更なし | `run_f16` のコードパス自体は raw のまま（無変更）。`pool.rs`／`gemm_mma.rs` の一般化・診断用 API 追加はコード変更として残るが、`internal-diagnostics` feature 限定のため既定ビルドの公開 API 面・本番経路には影響しない |
