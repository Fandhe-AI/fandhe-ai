## N=256 集計（5 run 中央値, µs。括弧内は 5 run 間の min–max）


- HEAD path-patch reuse（AC-2）: 90.720／candle fresh: 75.088／gap（HEAD − candle。帰属の分母・RULE.txt 5）: 15.632
- 参考: registry =0.9.0 reuse（AC-2）: 89.769／gap（registry − candle。版差を含み帰属に使わない）: 14.681
- HEAD path-patch / registry（record_only・RULE.txt 4）: 1.011（要調査・別 issue）
- L0 ops_total: 52.737／L1 gemm_total: 48.752／L2 l2_sum: 52.208
- Layer A（HEAD path-patch）iter_total: 90.425／matmul: 56.216（参考 registry =0.9.0: iter_total 91.848／matmul 57.408）
- L0 ops_total の run 間 min–max: 52.464–53.136
- reuse が candle fresh より遅い逆転: あり（HEAD reuse − candle fresh = +15.632）
- nsys 補助診断プロセス（RULE.txt 1a。判定・統計に不使用）: 取得済み（nsys-cuda-api.log を参照）

- 注: 各値の中央値は run 単位の値の 5 run 中央値。仮説の µs は run 毎に導出した値の中央値
  （差・和は同一 run 内の区間から作る。H2 は kernel_wait と dev_kernel_b2b が別測定系列の推定を
  組み合わせた値で、厳密な区間ではない）。

### 床・launch overhead（RULE.txt 7。L0 比を併記）

| 項目 | µs (min–max) | L0 比 |
| --- | --- | --- |
| host→kernel 往復 (E tiny_roundtrip) | 6.096 (6.081–6.128) | 11.6% |
| device idle 推定 (dev_kernel_seg-b2b) | 2.268 (2.236–2.428) | 4.3% |
| host のみ時間 (l2_sum-dev_span, 0 未満は 0) | 0.400 (0.017–0.656) | 0.8% |

### H2 と床の照合（RULE.txt 6）

| 項目 | µs (min–max) | H2 中央値との比 |
| --- | --- | --- |
| H2 launch+同期の往復（推定） | 3.948 (3.948–4.032) | 100.0% |
| E tiny_roundtrip | 6.096 (6.081–6.128) | 154.4% |
| E sync_idle | 0.288 (0.288–0.288) | 7.3% |

| 仮説 | µs (min–max) | L0 比 | gap 比 | 判定 |
| --- | --- | --- | --- | --- |
| H1 毎反復 H2D | 22.896 (22.448–23.248) | 43.4% | 146.5% | 支持 |
| H2 launch+同期の往復（推定） | 3.948 (3.948–4.032) | 7.5% | 25.3% | 未確定 |
| H3 D2H readback | 25.056 (24.832–25.120) | 47.5% | 160.3% | 支持 |
| H4a host dispatch (L0-L1) | 4.144 (3.376–5.231) | 7.9% | 26.5% | 未確定 |
| H4b host dispatch (L1-l2_sum) | -3.776 (-4.303–-2.625) | -7.2% | -24.2% | 未確定 |
| H5 facade/tape (HEAD matmul-L0) | 3.625 (3.001–5.599) | 6.9% | 23.2% | 未確定 |
| 参考 registry =0.9.0 matmul-L0（版差を含む・帰属に使わない） | 4.743 (3.504–5.264) | 9.0% | 30.3% | 記録のみ |
| H6 checksum（記録のみ） | 33.696 (33.592–34.000) | 63.9% | 215.6% | 記録のみ |
| 補助 teardown | 1.536 (1.520–1.552) | 2.9% | 9.8% | 記録のみ |
| 補助 driver_scope（本番 with_driver_call 入退場） | 0.528 (0.512–0.528) | 1.0% | 3.4% | 記録のみ |
| 補助 alloc_c | 0.064 (0.064–0.064) | 0.1% | 0.4% | 記録のみ |
| 補助 h2d_clone_drop − h2d_prealloc（cudarc 内部の確保・解放） | 1.280 (1.232–1.520) | 2.4% | 8.2% | 記録のみ |

### 支配項（RULE.txt 8。L0 比が最大の項目）

| 順位 | 項目 | µs | L0 比 | iter_total 比 |
| --- | --- | --- | --- | --- |
| 1 | L2 readback | 25.408 | 48.2% | 28.1% |
| 2 | L2 h2d_a | 11.984 | 22.7% | 13.3% |
| 3 | L2 h2d_b | 10.880 | 20.6% | 12.0% |
| 4 | H4a host dispatch (L0-L1) | 4.144 | 7.9% | 4.6% |
| 5 | H5 facade/tape (HEAD matmul-L0) | 3.625 | 6.9% | 4.0% |
| 6 | L2 launch_issue | 1.664 | 3.2% | 1.8% |
| 7 | L2 teardown | 1.536 | 2.9% | 1.7% |
| 8 | L2 driver_scope | 0.528 | 1.0% | 0.6% |
| 9 | L2 alloc_c | 0.064 | 0.1% | 0.1% |
| 10 | H4b host dispatch (L1-l2_sum) | -3.776 | -7.2% | -4.2% |

- 支配項: **L2 readback**（25.408 µs／L0 比 48.2%）

