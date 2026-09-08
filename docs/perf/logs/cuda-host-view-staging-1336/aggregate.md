# host_view_staging_readout_ab_1336 集計（5 run 中央値）

各セルは `median_ms`（1 run あたり 20/20 warmup+計測の中央値）を 5 プロセス起動ぶん集めた系列の中央値。`before` は `download()`（毎回新規確保）、`after_pageable` は本番既定（`HOST_STAGING_KIND=Pageable`）、`after_pinned` はキャッシュ経由 `Pinned`（`new_with_host_staging_kind`）。

| N | bytes(MiB) | before median_ms (5run) | after_pageable median_ms (5run) | after_pinned median_ms (5run) | pageable/before | pinned/before |
|---|---|---|---|---|---|---|
| 1024 | 4.0000 | 0.4252 (n=5) | 0.2680 (n=5) | 0.2108 (n=5) | 0.630x | 0.496x |
| 2048 | 16.0000 | 1.3366 (n=5) | 0.9338 (n=5) | 0.7565 (n=5) | 0.699x | 0.566x |
| 4096 | 64.0000 | 26.5788 (n=5) | 3.3413 (n=5) | 3.1270 (n=5) | 0.126x | 0.118x |

比率列は `median / before_median` で、1 未満は before に対する高速化（例: 0.10x は約 10 倍高速）を表す。
