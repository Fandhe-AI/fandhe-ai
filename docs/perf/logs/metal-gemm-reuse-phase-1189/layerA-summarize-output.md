# ベンチマーク集計（summarize.py 生成）

## 集計対象: results/raw/results-m4max-gemm-phases-0.6.0.jsonl

| フレームワーク | バージョン |
| --- | --- |
| fandhe-ai | 0.6.0 |
| candle | ? |
| burn | ? |

### (a) GEMM（C = A×B、f32、正方行列）

### (b) MLP 学習（784→256→10、ReLU、バッチ 64、MSE、SGD lr=0.01、1 ステップあたり時間）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 |
| --- | --- | --- | --- | --- |

### (a'') GEMM reuse 計測境界のフェーズ分解（イシュー #1182）

#### Metal / reuse / N=1024 / run 1/5

初期化(init_s): 48.423 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 2.067 ms | 1.815 ms | 2.103 ms | 71.5% |
| to_tensor | 0.1 µs | 0.1 µs | 0.1 µs | 0.0% |
| host_copy | 236.8 µs | 231.9 µs | 252.3 µs | 8.2% |
| checksum | 580.3 µs | 573.3 µs | 587.5 µs | 20.1% |
| iter_total | 2.891 ms | 2.624 ms | 2.939 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 2.884 ms

#### Metal / reuse / N=1024 / run 2/5

初期化(init_s): 39.614 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 2.039 ms | 1.466 ms | 2.130 ms | 71.3% |
| to_tensor | 0.0 µs | 0.0 µs | 0.1 µs | 0.0% |
| host_copy | 245.3 µs | 229.5 µs | 251.1 µs | 8.6% |
| checksum | 571.1 µs | 566.3 µs | 581.2 µs | 20.0% |
| iter_total | 2.859 ms | 2.278 ms | 2.935 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 2.855 ms

#### Metal / reuse / N=1024 / run 3/5

初期化(init_s): 40.108 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 1.989 ms | 1.398 ms | 2.142 ms | 71.3% |
| to_tensor | 0.1 µs | 0.0 µs | 0.1 µs | 0.0% |
| host_copy | 239.7 µs | 217.9 µs | 247.3 µs | 8.6% |
| checksum | 569.0 µs | 566.5 µs | 575.3 µs | 20.4% |
| iter_total | 2.791 ms | 2.203 ms | 2.966 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 2.798 ms

#### Metal / reuse / N=1024 / run 4/5

初期化(init_s): 39.317 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 2.058 ms | 1.437 ms | 2.190 ms | 71.6% |
| to_tensor | 0.0 µs | 0.0 µs | 0.0 µs | 0.0% |
| host_copy | 242.5 µs | 236.1 µs | 256.2 µs | 8.4% |
| checksum | 569.3 µs | 566.5 µs | 569.4 µs | 19.8% |
| iter_total | 2.875 ms | 2.236 ms | 3.012 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 2.869 ms

#### Metal / reuse / N=1024 / run 5/5

初期化(init_s): 41.614 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 1.485 ms | 1.404 ms | 2.177 ms | 65.1% |
| to_tensor | 0.0 µs | 0.0 µs | 0.0 µs | 0.0% |
| host_copy | 233.4 µs | 225.4 µs | 245.3 µs | 10.2% |
| checksum | 567.9 µs | 566.4 µs | 570.1 µs | 24.9% |
| iter_total | 2.280 ms | 2.209 ms | 2.999 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 2.286 ms

#### Metal / reuse / N=2048 / run 1/5

初期化(init_s): 52.146 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 5.587 ms | 5.131 ms | 6.657 ms | 64.8% |
| to_tensor | 0.1 µs | 0.1 µs | 0.1 µs | 0.0% |
| host_copy | 881.0 µs | 867.2 µs | 902.4 µs | 10.2% |
| checksum | 2.230 ms | 2.188 ms | 2.267 ms | 25.9% |
| iter_total | 8.622 ms | 8.254 ms | 9.814 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 8.699 ms

#### Metal / reuse / N=2048 / run 2/5

初期化(init_s): 51.046 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 4.832 ms | 4.668 ms | 9.949 ms | 59.7% |
| to_tensor | 0.1 µs | 0.0 µs | 0.1 µs | 0.0% |
| host_copy | 921.8 µs | 885.1 µs | 941.9 µs | 11.4% |
| checksum | 2.252 ms | 2.167 ms | 2.278 ms | 27.8% |
| iter_total | 8.093 ms | 7.797 ms | 13.047 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 8.006 ms

#### Metal / reuse / N=2048 / run 3/5

初期化(init_s): 51.777 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 5.549 ms | 4.707 ms | 9.884 ms | 63.3% |
| to_tensor | 0.0 µs | 0.0 µs | 0.1 µs | 0.0% |
| host_copy | 891.6 µs | 877.0 µs | 929.2 µs | 10.2% |
| checksum | 2.268 ms | 2.266 ms | 2.281 ms | 25.9% |
| iter_total | 8.766 ms | 7.867 ms | 13.032 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 8.709 ms

#### Metal / reuse / N=2048 / run 4/5

初期化(init_s): 50.332 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 6.306 ms | 5.354 ms | 7.956 ms | 66.2% |
| to_tensor | 0.1 µs | 0.0 µs | 0.2 µs | 0.0% |
| host_copy | 919.6 µs | 890.9 µs | 983.7 µs | 9.6% |
| checksum | 2.296 ms | 2.278 ms | 2.318 ms | 24.1% |
| iter_total | 9.530 ms | 8.541 ms | 11.332 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 9.522 ms

#### Metal / reuse / N=2048 / run 5/5

初期化(init_s): 51.075 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 4.967 ms | 4.558 ms | 10.055 ms | 60.6% |
| to_tensor | 0.1 µs | 0.0 µs | 0.2 µs | 0.0% |
| host_copy | 920.1 µs | 890.4 µs | 938.2 µs | 11.2% |
| checksum | 2.282 ms | 2.266 ms | 2.311 ms | 27.8% |
| iter_total | 8.197 ms | 7.810 ms | 13.367 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 8.169 ms

#### Metal / reuse / N=4096 / run 1/5

初期化(init_s): 94.205 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 34.687 ms | 25.356 ms | 37.202 ms | 72.3% |
| to_tensor | 0.2 µs | 0.1 µs | 0.2 µs | 0.0% |
| host_copy | 3.653 ms | 3.565 ms | 3.781 ms | 7.6% |
| checksum | 9.628 ms | 9.590 ms | 9.735 ms | 20.1% |
| iter_total | 47.944 ms | 38.574 ms | 50.481 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 47.967 ms

#### Metal / reuse / N=4096 / run 2/5

初期化(init_s): 90.261 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 34.800 ms | 23.907 ms | 36.665 ms | 73.1% |
| to_tensor | 0.2 µs | 0.1 µs | 0.2 µs | 0.0% |
| host_copy | 3.657 ms | 3.522 ms | 3.755 ms | 7.7% |
| checksum | 9.248 ms | 9.114 ms | 9.624 ms | 19.4% |
| iter_total | 47.626 ms | 37.148 ms | 49.749 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 47.706 ms

#### Metal / reuse / N=4096 / run 3/5

初期化(init_s): 91.408 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 36.491 ms | 34.038 ms | 37.058 ms | 73.2% |
| to_tensor | 0.2 µs | 0.1 µs | 0.2 µs | 0.0% |
| host_copy | 3.719 ms | 3.654 ms | 3.836 ms | 7.5% |
| checksum | 9.636 ms | 9.606 ms | 9.695 ms | 19.3% |
| iter_total | 49.824 ms | 47.708 ms | 50.510 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 49.847 ms

#### Metal / reuse / N=4096 / run 4/5

初期化(init_s): 92.901 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 33.861 ms | 25.431 ms | 36.281 ms | 71.8% |
| to_tensor | 0.2 µs | 0.1 µs | 0.2 µs | 0.0% |
| host_copy | 3.796 ms | 3.566 ms | 3.985 ms | 8.1% |
| checksum | 9.614 ms | 9.596 ms | 9.701 ms | 20.4% |
| iter_total | 47.132 ms | 38.908 ms | 49.626 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 47.270 ms

#### Metal / reuse / N=4096 / run 5/5

初期化(init_s): 89.189 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| matmul | 35.521 ms | 25.443 ms | 36.061 ms | 72.6% |
| to_tensor | 0.1 µs | 0.1 µs | 0.2 µs | 0.0% |
| host_copy | 3.769 ms | 3.671 ms | 3.933 ms | 7.7% |
| checksum | 9.740 ms | 9.708 ms | 9.841 ms | 19.9% |
| iter_total | 48.948 ms | 39.056 ms | 49.625 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 49.030 ms

### (c) 推論スループット（同 MLP forward のみ、バッチ 64。表のスループットはバッチ/秒 = 1/中央値。1 バッチ = 64 件）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 | バッチ/秒 |
| --- | --- | --- | --- | --- | --- |

#### データ有効性（checksum 突合・要素単位検証。イシュー #965・#970）

- 相互突合できた行なし（全 gemm 行が比較対象なしで突合不能）
- 要素単位検証済みの行なし（全 gemm 行が旧形式または対象外）

## 実行時失敗（skipped*.log）

- **skipped-m4max-0.4.0.log**: bench-burn task=gemm device=metal size=512 mode=fresh : MEASURE_ERROR: gemm checksum is degenerate (0) — result tensor i…
- **skipped-m4max-0.4.0.log**: bench-burn task=gemm device=metal size=1024 mode=fresh : MEASURE_ERROR: gemm checksum is degenerate (0) — result tensor …
- **skipped-m4max-0.4.0.log**: bench-burn task=gemm device=metal size=2048 mode=fresh : MEASURE_ERROR: gemm checksum is degenerate (0) — result tensor …
- **skipped-m4max-0.4.0.log**: bench-burn task=gemm device=metal size=4096 mode=fresh : MEASURE_ERROR: gemm checksum is degenerate (0) — result tensor …
- **skipped-m4max-0.5.0.log**: bench-burn task=gemm device=metal size=512 mode=fresh extra=none : MEASURE_ERROR: gemm checksum is degenerate (0) — resu…
- **skipped-m4max-0.5.0.log**: bench-burn task=gemm device=metal size=1024 mode=fresh extra=none : MEASURE_ERROR: gemm checksum is degenerate (0) — res…
- **skipped-m4max-0.5.0.log**: bench-burn task=gemm device=metal size=2048 mode=fresh extra=none : MEASURE_ERROR: gemm checksum is degenerate (0) — res…
- **skipped-m4max-0.5.0.log**: bench-burn task=gemm device=metal size=4096 mode=fresh extra=none : MEASURE_ERROR: gemm checksum is degenerate (0) — res…
- **skipped-m4max-0.6.0.log**: bench-burn task=gemm device=metal size=512 mode=fresh extra=none : MEASURE_ERROR: gemm checksum is degenerate (0) — resu…
- **skipped-m4max-0.6.0.log**: bench-burn task=gemm device=metal size=1024 mode=fresh extra=none : MEASURE_ERROR: gemm checksum is degenerate (0) — res…
- **skipped-m4max-0.6.0.log**: bench-burn task=gemm device=metal size=2048 mode=fresh extra=none : MEASURE_ERROR: gemm checksum is degenerate (0) — res…
- **skipped-m4max-0.6.0.log**: bench-burn task=gemm device=metal size=4096 mode=fresh extra=none : MEASURE_ERROR: gemm checksum is degenerate (0) — res…
- **skipped-rtx3060-train.log**: bench-candle BUILD FAILED (--features cuda): Err(Os { code: 2, kind: NotFound, message: "No such file or directory" }) —…
- **skipped-rtx3060-train.log**: bench-burn task=gemm/train/infer device=cuda : ビルドは成功したが実行時に cubecl-cuda が「CUDA installation not found」パニックを出し（CUDA_PATH…
- **skipped-rtx3060-train.log**: bench-fandhe task=train device=metal mode=reuse : NOT RUN — macOS 実機がこのエージェント環境（x86_64 Linux）から到達不能。再現: Apple Silicon 実機…
- **skipped-rtx3060-train.log**: bench-fandhe task=train device=cuda mode=reuse (DGX Spark GB10 / sm_121) : NOT RUN — docs/real-hardware-verification-env…

