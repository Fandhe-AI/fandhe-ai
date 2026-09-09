# イシュー #1478 実機実測ログ（`HOST_STAGING_KIND` の既定を `Pinned` へ切替）

DGX Spark GB10（sm_121）実機実測。2026-09-09。転送は `git archive`
経由（`env_info.txt` の「転送リビジョン」節参照。作業ツリー rsync を
避けた理由: 隔離 worktree 上で並走する別イシュー〈#1479〉の未コミット
WIP による汚染防止）。

## ファイル一覧

- `env_info.txt`: 実機環境（`nvidia-smi`／`uptime`／`rustc -V`／
  `nvcc --version`）・転送リビジョン・ビルドコマンド。
- `ignored-host_view_real_device.log`: ゲート A
  （`host_view_real_device` の `#[ignore]` 全 8 件。新規追加分
  `default_cuda_memory_uses_pinned_staging_and_matches_pageable_bit_exact`
  を含む）の実行ログ。
- `ab-run1.log`〜`ab-run5.log`: ゲート B
  （`host_view_staging_readout_ab_1336.rs::host_view_staging_readout_ab`）
  の独立プロセス起動 5 回の生ログ。各 run 冒頭に
  `default_kind,<kind>`（`CudaMemory::new` の本番既定コンストラクタが
  実際に解決した種別。自己証明。イシュー #1478 AC-2）を含む。
- `aggregate.py`: 上記 5 run の `median_ms` 列から N ごとの代表値
  （5 run の中央値）を算出する集計スクリプト（Python3 標準ライブラリ
  のみ）。
- `aggregate.md`: `aggregate.py` の決定的出力。

## 結果要約

### ゲート A（必須）

`host_view_real_device` の `#[ignore]` 全 8 件 pass
（`ignored-host_view_real_device.log`）。新規追加した
`default_cuda_memory_uses_pinned_staging_and_matches_pageable_bit_exact`
（`CudaMemory::new` が実際に `Pinned` へ解決され、かつ `Pageable`
対照腕・`download()` と bit 完全一致することを検証）も pass。

### ゲート B（本番切替の非後退。判定基準 `after_pinned/after_pageable ≤ 1.05`）

| N | before_med_ms | pageable_med_ms | pinned_med_ms | pinned/pageable | 判定 |
|---|---|---|---|---|---|
| 1024 | 0.4014 | 0.2606 | 0.2069 | 0.7939 | PASS |
| 2048 | 1.2494 | 0.9170 | 0.7838 | 0.8547 | PASS |
| 4096 | 30.1436 | 3.1983 | 3.0325 | 0.9482 | PASS |

全 N で判定基準（≤1.05）を満たし、いずれも改善方向（`Pinned` が
`Pageable` を約 6〜21% 上回る）。`docs/perf/cuda-host-view-staging-readout.md`
§5.2・§5.3（イシュー #1438 確定値）の傾向を本番既定切替後の系列構成
（`after_pageable` を明示種別で構築する F1 是正後）でも再現した。

`default_kind` は全 5 run とも `Pinned`（`CudaMemory::new` が本番既定
どおり `Pinned` へ解決されていることを確認済み）。

### 事前宣言ゲート（`docs/perf/cuda-host-view-staging-readout.md` §8 転記）

- ゲート A: `#[ignore]` 全件 pass — **達成**
- ゲート B: 全 N で `after_pinned/after_pageable ≤ 1.05` かつ
  `default_kind,Pinned` が 5 run すべてに存在 — **達成**
- ゲート C（framework-compare gemm cuda reuse 非後退ガード）は
  `docs/perf/logs/cuda-host-staging-pinned-default-1478/gate-c-sanity.md` を参照
  （実施した場合）。想定結果は「差なし」（`gemm` タスクは
  `MemoryOps::with_host_view` に到達しないため。イシュー #1478 計画
  F2 参照）。

## 再現手順

```sh
# ゲート A
cargo test -p fandhe-ai-backend-cuda --release --all-features \
    --test host_view_real_device -- --ignored --nocapture --test-threads=1

# ゲート B（5 回独立プロセス起動）
for i in 1 2 3 4 5; do
  cargo test -p fandhe-ai-backend-cuda --release --all-features \
      --test host_view_staging_readout_ab_1336 -- --ignored --nocapture --test-threads=1 \
      > ab-run$i.log 2>&1
done
python3 aggregate.py
```
