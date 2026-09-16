# pinned_h2d_upload_ab_1585 集計（5 run 中央値）

各セルは `median_ms`（1 run あたり 20/20 warmup+計測の中央値）を 5 プロセス起動ぶん集めた系列の中央値。`pageable` はフラグ OFF（`clone_htod` 直呼び経路）、`pinned_staged` はフラグ ON（`H2dStagingCache` 経由の pinned ステージング）。いずれも「upload 発行 → `stream.synchronize()`」を 1 反復として計測する（`docs/perf/cuda-h2d-pinned-staging.md` §3）。

| N | bytes(MiB) | pageable median_ms (5run) | pinned_staged median_ms (5run) | pinned_staged/pageable | 判定 |
|---|---|---|---|---|---|
| 1024 | 4.0000 | 0.0813 (n=5) | 0.1951 (n=5) | 2.400x | 非改善 |
| 2048 | 16.0000 | 0.2942 (n=5) | 0.8802 (n=5) | 2.992x | 非改善 |
| 4096 | 64.0000 | 1.1449 (n=5) | 3.7747 (n=5) | 3.297x | 非改善 |

`pinned_staged/pageable < 1.00` を改善、`>= 1.00` を非改善として記録する規則（`docs/perf/cuda-h2d-pinned-staging.md` §3）。本規則は Layer B（改善根拠）単体の判定であり、Layer A（framework-compare 非後退）・ゲート A（`#[ignore]` 実機テスト全件 pass）と合わせて総合判断する。

全 N で改善（pinned_staged/pageable < 1.00）: False
