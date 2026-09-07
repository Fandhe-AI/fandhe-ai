# host_view_staging_readout_ab_1336 集計（5 run 中央値）

各セルは `median_ms`（1 run あたり 20/20 warmup+計測の中央値）を 5 プロセス起動ぶん集めた系列の中央値。`before` は `download()`（毎回新規確保）、`after_pageable` は本番既定（`HOST_STAGING_KIND=Pageable`）、`after_pinned` はキャッシュ経由 `Pinned`（`new_with_host_staging_kind`）。

| N | bytes(MiB) | before median_ms (5run) | after_pageable median_ms (5run) | after_pinned median_ms (5run) | pageable/before | pinned/before |
|---|---|---|---|---|---|---|
| 1024 | 4.0000 | 0.2616 (n=5) | 0.2598 (n=5) | 0.2076 (n=5) | 0.993x | 0.794x |
| 2048 | 16.0000 | 0.9016 (n=5) | 0.9180 (n=5) | 0.7780 (n=5) | 1.018x | 0.863x |
| 4096 | 64.0000 | 33.5578 (n=5) | 3.2441 (n=5) | 3.0391 (n=5) | 0.097x | 0.091x |

比率列は `median / before_median` で、1 未満は before に対する高速化（例: 0.10x は約 10 倍高速）を表す。
