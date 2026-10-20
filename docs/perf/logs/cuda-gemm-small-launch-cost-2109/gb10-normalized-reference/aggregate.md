## N=256 集計（5 run 中央値, µs。括弧内は 5 run 間の min–max）


- HEAD path-patch reuse（AC-2）: 90.065／candle fresh: 75.865／gap（HEAD − candle。帰属の分母・RULE.txt 5）: 14.200
- 参考: registry =0.9.0 reuse（AC-2）: 91.816／gap（registry − candle。版差を含み帰属に使わない）: 15.951
- HEAD path-patch / registry（record_only・RULE.txt 4）: 0.981（非後退）
- L0 ops_total: 51.648／L1 gemm_total: 47.808／L2 l2_sum: 51.057
- Layer A（HEAD path-patch）iter_total: 91.984／matmul: 57.480（参考 registry =0.9.0: iter_total 91.848／matmul 57.264）
- L0 ops_total の run 間 min–max: 51.008–52.560
- reuse が candle fresh より遅い逆転: あり（HEAD reuse − candle fresh = +14.200）
- nsys 補助診断プロセス（RULE.txt 1a。判定・統計に不使用）: 取得済み（nsys-cuda-api.log を参照）

- 注: 各値の中央値は run 単位の値の 5 run 中央値。仮説の µs は run 毎に導出した値の中央値
  （差・和は同一 run 内の区間から作る。H2 は kernel_wait と dev_kernel_b2b が別測定系列の推定を
  組み合わせた値で、厳密な区間ではない）。

### 床・launch overhead（RULE.txt 7。L0 比を併記）

| 項目 | µs (min–max) | L0 比 |
| --- | --- | --- |
| host→kernel 往復 (E tiny_roundtrip) | 6.064 (5.984–6.112) | 11.7% |
| device idle 推定 (dev_kernel_seg-b2b) | 2.284 (2.204–2.452) | 4.4% |
| host のみ時間 (l2_sum-dev_span, 0 未満は 0) | 0.208 (0.000–0.656) | 0.4% |

### H2 と床の照合（RULE.txt 6）

| 項目 | µs (min–max) | H2 中央値との比 |
| --- | --- | --- |
| H2 launch+同期の往復（推定） | 3.916 (3.748–4.036) | 100.0% |
| E tiny_roundtrip | 6.064 (5.984–6.112) | 154.9% |
| E sync_idle | 0.288 (0.272–0.288) | 7.4% |

| 仮説 | µs (min–max) | L0 比 | gap 比 | 判定 |
| --- | --- | --- | --- | --- |
| H1 毎反復 H2D | 23.120 (22.448–23.312) | 44.8% | 162.8% | 支持 |
| H2 launch+同期の往復（推定） | 3.916 (3.748–4.036) | 7.6% | 27.6% | 未確定 |
| H3 D2H readback | 23.776 (23.617–25.056) | 46.0% | 167.4% | 支持 |
| H4a host dispatch (L0-L1) | 3.968 (3.200–4.783) | 7.7% | 27.9% | 未確定 |
| H4b host dispatch (L1-l2_sum) | -3.569 (-4.095–-3.248) | -6.9% | -25.1% | 未確定 |
| H5 facade/tape (HEAD matmul-L0) | 5.840 (4.184–6.681) | 11.3% | 41.1% | 未確定 |
| 参考 registry =0.9.0 matmul-L0（版差を含む・帰属に使わない） | 5.616 (3.368–19.249) | 10.9% | 39.5% | 記録のみ |
| H6 checksum（記録のみ） | 34.032 (33.952–34.328) | 65.9% | 239.7% | 記録のみ |
| 補助 teardown | 1.648 (1.488–1.904) | 3.2% | 11.6% | 記録のみ |
| 補助 driver_scope（本番 with_driver_call 入退場） | 0.528 (0.512–0.528) | 1.0% | 3.7% | 記録のみ |
| 補助 alloc_c | 0.064 (0.064–0.064) | 0.1% | 0.5% | 記録のみ |
| 補助 h2d_clone_drop − h2d_prealloc（cudarc 内部の確保・解放） | 1.280 (1.136–1.312) | 2.5% | 9.0% | 記録のみ |

### 支配項（RULE.txt 8。L0 比が最大の項目）

| 順位 | 項目 | µs | L0 比 | iter_total 比 |
| --- | --- | --- | --- | --- |
| 1 | L2 readback | 24.096 | 46.7% | 26.2% |
| 2 | L2 h2d_a | 11.984 | 23.2% | 13.0% |
| 3 | L2 h2d_b | 11.104 | 21.5% | 12.1% |
| 4 | H5 facade/tape (HEAD matmul-L0) | 5.840 | 11.3% | 6.3% |
| 5 | H4a host dispatch (L0-L1) | 3.968 | 7.7% | 4.3% |
| 6 | L2 launch_issue | 1.664 | 3.2% | 1.8% |
| 7 | L2 teardown | 1.648 | 3.2% | 1.8% |
| 8 | L2 driver_scope | 0.528 | 1.0% | 0.6% |
| 9 | L2 alloc_c | 0.064 | 0.1% | 0.1% |
| 10 | H4b host dispatch (L1-l2_sum) | -3.569 | -6.9% | -3.9% |

- 支配項: **L2 readback**（24.096 µs／L0 比 46.7%）

