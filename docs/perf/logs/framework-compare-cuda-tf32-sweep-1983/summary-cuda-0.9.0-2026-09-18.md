# ベンチマーク集計（summarize.py 生成）

## 集計対象: results/raw/results-cuda.jsonl

| フレームワーク | バージョン |
| --- | --- |
| fandhe-ai | 0.9.0 |
| candle | 0.11.0 |
| burn | 0.21.0 |

### (a) GEMM（C = A×B、f32、正方行列）

#### CPU

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 190.8 µs | 178.6 µs | 232.5 µs | 175.9 |
| 256 | candle | 314.3 µs | 308.9 µs | 346.3 µs | 106.8 |
| 256 | burn | 587.1 µs | 573.2 µs | 588.5 µs | 57.2 |
| 512 | fandhe-ai | 1.821 ms | 1.723 ms | 1.936 ms | 147.4 |
| 512 | candle | 1.755 ms | 1.707 ms | 2.032 ms | 152.9 |
| 512 | burn | 3.384 ms | 3.377 ms | 3.388 ms | 79.3 |
| 1024 | fandhe-ai | 4.558 ms | 4.352 ms | 4.726 ms | 471.1 |
| 1024 | candle | 5.520 ms | 5.391 ms | 5.666 ms | 389.1 |
| 1024 | burn | 22.136 ms | 22.115 ms | 22.227 ms | 97.0 |
| 2048 | fandhe-ai | 16.270 ms | 15.766 ms | 16.987 ms | 1055.9 |
| 2048 | candle | 34.583 ms | 33.651 ms | 35.246 ms | 496.8 |
| 2048 | burn | 156.989 ms | 156.582 ms | 157.369 ms | 109.4 |

#### CUDA

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 87.6 µs | 87.0 µs | 99.4 µs | 382.9 |
| 256 | candle | 75.0 µs | 74.8 µs | 75.4 µs | 447.3 |
| 256 | burn | 計測不可 | - | - | - |
| 512 | fandhe-ai | 258.5 µs | 258.2 µs | 258.9 µs | 1038.3 |
| 512 | candle | 237.2 µs | 236.3 µs | 238.0 µs | 1131.8 |
| 512 | burn | 計測不可 | - | - | - |
| 1024 | fandhe-ai | 955.1 µs | 954.2 µs | 956.9 µs | 2248.5 |
| 1024 | candle | 917.8 µs | 916.2 µs | 919.0 µs | 2339.8 |
| 1024 | burn | 計測不可 | - | - | - |
| 2048 | fandhe-ai | 4.193 ms | 4.190 ms | 4.217 ms | 4096.9 |
| 2048 | candle | 4.224 ms | 4.216 ms | 4.231 ms | 4067.5 |
| 2048 | burn | 計測不可 | - | - | - |
| 4096 | fandhe-ai | 39.922 ms | 39.754 ms | 40.012 ms | 3442.7 |
| 4096 | candle | 58.037 ms | 56.350 ms | 58.444 ms | 2368.1 |
| 4096 | burn | 計測不可 | - | - | - |

### (a') GEMM（デバイス/tape 再利用モード。初期化コストとカーネル実行の分離。イシュー #925）

#### CUDA

| N | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | GFLOP/s | fresh 中央値（参考） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 406.030 ms | 90.8 µs | 90.4 µs | 92.9 µs | 369.6 | 87.6 µs |
| 512 | fandhe-ai | 409.440 ms | 551.4 µs | 544.7 µs | 560.6 µs | 486.8 | 258.5 µs |
| 1024 | fandhe-ai | 415.691 ms | 2.331 ms | 2.155 ms | 2.397 ms | 921.4 | 955.1 µs |
| 2048 | fandhe-ai | 426.199 ms | 8.693 ms | 8.592 ms | 9.116 ms | 1976.3 | 4.193 ms |
| 4096 | fandhe-ai | 460.968 ms | 39.222 ms | 38.920 ms | 39.377 ms | 3504.2 | 39.922 ms |

### (a-tf32) GEMM TF32（--tf32 opt-in。REQ-2 統一複合判定。CUDA Tensor Core reduced precision）

#### CUDA

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai（無効: 要素誤差超過 fail=10538/65536, max_abs=1.581e-03, max_rel=1.554e+00, rescued=0, bound=0.000e+00） | 92.2 µs | 91.9 µs | 92.5 µs | - |
| 256 | candle（無効: 要素誤差超過 fail=10522/65536, max_abs=1.582e-03, max_rel=1.554e+00, rescued=0, bound=1.907e-06） | 64.7 µs | 64.4 µs | 65.7 µs | - |
| 256 | burn（無効: 要素誤差超過 fail=10538/65536, max_abs=1.581e-03, max_rel=1.554e+00, rescued=0, bound=1.907e-06） | 427.7 µs | 178.5 µs | 496.8 µs | - |
| 512 | fandhe-ai（無効: 要素誤差超過 fail=42361/262144, max_abs=2.343e-03, max_rel=1.972e+00, rescued=0, bound=0.000e+00） | 262.1 µs | 261.7 µs | 262.3 µs | - |
| 512 | candle（無効: 要素誤差超過 fail=42387/262144, max_abs=2.340e-03, max_rel=1.970e+00, rescued=0, bound=3.815e-06） | 223.3 µs | 222.9 µs | 223.8 µs | - |
| 512 | burn（無効: 要素誤差超過 fail=42361/262144, max_abs=2.343e-03, max_rel=1.972e+00, rescued=0, bound=3.815e-06） | 422.0 µs | 351.8 µs | 551.3 µs | - |
| 1024 | fandhe-ai（無効: 要素誤差超過 fail=169929/1048576, max_abs=3.643e-03, max_rel=1.965e+00, rescued=0, bound=0.000e+00） | 962.8 µs | 962.0 µs | 964.0 µs | - |
| 1024 | candle（無効: 要素誤差超過 fail=169971/1048576, max_abs=3.650e-03, max_rel=1.951e+00, rescued=0, bound=7.629e-06） | 864.6 µs | 863.8 µs | 865.6 µs | - |
| 1024 | burn（無効: 要素誤差超過 fail=169929/1048576, max_abs=3.643e-03, max_rel=1.965e+00, rescued=0, bound=7.629e-06） | 1.003 ms | 989.0 µs | 1.157 ms | - |
| 2048 | fandhe-ai（無効: 要素誤差超過 fail=681454/4194304, max_abs=4.941e-03, max_rel=1.987e+00, rescued=0, bound=0.000e+00） | 4.381 ms | 4.375 ms | 4.404 ms | - |
| 2048 | candle（無効: 要素誤差超過 fail=681418/4194304, max_abs=4.930e-03, max_rel=1.984e+00, rescued=33, bound=1.526e-05） | 3.713 ms | 3.701 ms | 3.729 ms | - |
| 2048 | burn（無効: 要素誤差超過 fail=681407/4194304, max_abs=4.941e-03, max_rel=1.987e+00, rescued=47, bound=1.526e-05） | 4.208 ms | 4.157 ms | 4.276 ms | - |
| 4096 | fandhe-ai（無効: 要素誤差超過 fail=2729050/16777216, max_abs=7.117e-03, max_rel=1.997e+00, rescued=0, bound=0.000e+00） | 33.867 ms | 33.609 ms | 33.922 ms | - |
| 4096 | candle（無効: 要素誤差超過 fail=2726683/16777216, max_abs=7.167e-03, max_rel=2.000e+00, rescued=646, bound=3.052e-05） | 58.698 ms | 58.622 ms | 59.161 ms | - |
| 4096 | burn（無効: 要素誤差超過 fail=2728488/16777216, max_abs=7.117e-03, max_rel=1.997e+00, rescued=562, bound=3.052e-05） | 39.682 ms | 39.574 ms | 39.829 ms | - |

### (b) MLP 学習（784→256→10、ReLU、バッチ 64、MSE、SGD lr=0.01、1 ステップあたり時間）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 |
| --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 1.193 ms | 955.0 µs | 1.548 ms |
| cpu | candle | 2.299 ms | 1.957 ms | 2.616 ms |
| cpu | burn | 963.3 µs | 958.6 µs | 967.4 µs |
| cuda | fandhe-ai | 518.2 µs | 500.5 µs | 535.5 µs |
| cuda | candle | 271.3 µs | 269.0 µs | 277.6 µs |
| cuda | burn（TF32） | 680.3 µs | 340.2 µs | 925.7 µs |

### (b') MLP 学習（デバイス常駐パラメータ更新モード。ホスト経由 SGD との分離。イシュー #957/#958/#959）

| デバイス | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | fresh 中央値（参考） | fresh/reuse 比 | 最終 loss 突合（fresh） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 926.5 µs | 972.9 µs | 894.5 µs | 1.209 ms | 1.193 ms | 1.23 倍 | 一致 |
| cuda | fandhe-ai | 213.606 ms | 315.5 µs | 313.8 µs | 316.5 µs | 518.2 µs | 1.64 倍 | 一致 |

### (b'') MLP 学習 1 step のフェーズ分解（イシュー #1009）

#### CPU / fresh

| フェーズ | 中央値 | Q1 | Q3 | step_total 比 |
| --- | --- | --- | --- | --- |
| tape_build | 0.3 µs | 0.2 µs | 0.3 µs | 0.0% |
| leaf_register | 1.8 µs | 1.4 µs | 2.0 µs | 0.2% |
| forward | 262.8 µs | 236.3 µs | 341.7 µs | 22.6% |
| loss_readout | 0.1 µs | 0.1 µs | 0.2 µs | 0.0% |
| backward | 442.7 µs | 401.4 µs | 644.5 µs | 38.1% |
| param_readout | 41.1 µs | 37.9 µs | 46.2 µs | 3.5% |
| host_sgd | 60.2 µs | 50.2 µs | 413.8 µs | 5.2% |
| apply_params | 0.7 µs | 0.6 µs | 1.0 µs | 0.1% |
| tape_drop | 4.2 µs | 3.3 µs | 211.2 µs | 0.4% |
| step_total | 1.162 ms | 976.2 µs | 1.435 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため step_total と一致しない場合がある）: 813.9 µs

#### CPU / reuse

初期化(init_s): 1.127 ms

| フェーズ | 中央値 | Q1 | Q3 | step_total 比 |
| --- | --- | --- | --- | --- |
| tape_build | 0.2 µs | 0.2 µs | 0.3 µs | 0.0% |
| leaf_register | 0.6 µs | 0.5 µs | 0.7 µs | 0.1% |
| forward_resident | 242.7 µs | 221.9 µs | 378.5 µs | 22.2% |
| loss_readout | 0.2 µs | 0.1 µs | 0.2 µs | 0.0% |
| backward | 485.0 µs | 417.5 µs | 807.8 µs | 44.3% |
| device_update | 263.9 µs | 229.1 µs | 286.2 µs | 24.1% |
| tape_drop | 1.9 µs | 1.6 µs | 2.2 µs | 0.2% |
| step_total | 1.094 ms | 929.6 µs | 1.476 ms | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため step_total と一致しない場合がある）: 994.5 µs

#### CUDA / fresh

| フェーズ | 中央値 | Q1 | Q3 | step_total 比 |
| --- | --- | --- | --- | --- |
| tape_build | 3.0 µs | 2.9 µs | 3.1 µs | 0.6% |
| leaf_register | 1.7 µs | 1.0 µs | 2.1 µs | 0.3% |
| forward | 176.2 µs | 170.9 µs | 178.7 µs | 34.2% |
| loss_readout | 0.0 µs | 0.0 µs | 0.0 µs | 0.0% |
| backward | 206.9 µs | 196.1 µs | 209.7 µs | 40.2% |
| param_readout | 80.6 µs | 66.8 µs | 92.4 µs | 15.6% |
| host_sgd | 44.8 µs | 37.5 µs | 57.6 µs | 8.7% |
| apply_params | 0.3 µs | 0.3 µs | 0.3 µs | 0.1% |
| tape_drop | 1.9 µs | 1.8 µs | 2.1 µs | 0.4% |
| step_total | 515.3 µs | 490.7 µs | 534.6 µs | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため step_total と一致しない場合がある）: 515.4 µs

#### CUDA / reuse

初期化(init_s): 219.877 ms

| フェーズ | 中央値 | Q1 | Q3 | step_total 比 |
| --- | --- | --- | --- | --- |
| tape_build | 2.4 µs | 2.3 µs | 2.4 µs | 0.7% |
| leaf_register | 0.3 µs | 0.2 µs | 0.3 µs | 0.1% |
| forward_resident | 154.8 µs | 153.9 µs | 155.7 µs | 49.3% |
| loss_readout | 0.0 µs | 0.0 µs | 0.0 µs | 0.0% |
| backward | 147.1 µs | 146.6 µs | 148.3 µs | 46.8% |
| device_update | 7.8 µs | 7.7 µs | 7.9 µs | 2.5% |
| tape_drop | 1.0 µs | 0.9 µs | 1.0 µs | 0.3% |
| step_total | 314.1 µs | 312.6 µs | 315.4 µs | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため step_total と一致しない場合がある）: 313.3 µs

### (c) 推論スループット（同 MLP forward のみ、バッチ 64。表のスループットはバッチ/秒 = 1/中央値。1 バッチ = 64 件）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 | バッチ/秒 |
| --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 180.5 µs | 169.3 µs | 190.0 µs | 5540 |
| cpu | candle | 249.9 µs | 218.7 µs | 259.4 µs | 4002 |
| cpu | burn | 234.0 µs | 232.7 µs | 235.8 µs | 4274 |
| cuda | fandhe-ai | 157.8 µs | 151.1 µs | 158.8 µs | 6339 |
| cuda | candle | 40.6 µs | 40.1 µs | 41.7 µs | 24636 |
| cuda | burn（TF32） | 376.3 µs | 195.0 µs | 466.5 µs | 2657 |

### (c') 推論スループット（デバイス常駐パラメータ・`predict_resident` reuse モード。イシュー #1217）

| デバイス | バッチ | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | バッチ/秒 | fresh 中央値（参考） | fresh/reuse 比 | checksum 突合（fresh） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | 64 | fandhe-ai | 993.7 µs | 195.4 µs | 186.6 µs | 236.5 µs | 5119 | 180.5 µs | 0.92 倍 | 一致 |
| cuda | 64 | fandhe-ai | 211.699 ms | 99.3 µs | 99.1 µs | 100.0 µs | 10069 | 157.8 µs | 1.59 倍 | 一致 |

### (c'') 推論 1 反復のフェーズ分解（イシュー #1217）

#### CPU / fresh / batch=64

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| predict | 198.9 µs | 182.5 µs | 290.3 µs | 99.5% |
| host_copy | 0.2 µs | 0.1 µs | 0.2 µs | 0.1% |
| checksum | 0.3 µs | 0.3 µs | 0.3 µs | 0.2% |
| iter_total | 199.9 µs | 183.2 µs | 291.4 µs | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 199.4 µs

#### CPU / reuse / batch=64

初期化(init_s): 962.0 µs

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| predict_resident | 188.4 µs | 168.6 µs | 208.9 µs | 99.7% |
| host_copy | 0.2 µs | 0.1 µs | 0.2 µs | 0.1% |
| checksum | 0.3 µs | 0.3 µs | 0.3 µs | 0.2% |
| iter_total | 189.0 µs | 169.2 µs | 210.3 µs | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 188.8 µs

#### CUDA / fresh / batch=64

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| leaf_register | 0.2 µs | 0.2 µs | 0.2 µs | 0.1% |
| forward | 136.9 µs | 131.8 µs | 138.7 µs | 87.1% |
| to_tensor | 19.2 µs | 19.1 µs | 19.4 µs | 12.3% |
| host_copy | 0.1 µs | 0.1 µs | 0.1 µs | 0.1% |
| checksum | 0.6 µs | 0.6 µs | 0.6 µs | 0.4% |
| iter_total | 157.1 µs | 152.3 µs | 159.0 µs | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 157.0 µs

#### CUDA / reuse / batch=64

初期化(init_s): 215.163 ms

| フェーズ | 中央値 | Q1 | Q3 | iter_total 比 |
| --- | --- | --- | --- | --- |
| predict_resident | 98.7 µs | 98.2 µs | 99.0 µs | 99.0% |
| host_copy | 0.1 µs | 0.0 µs | 0.1 µs | 0.1% |
| checksum | 0.7 µs | 0.6 µs | 0.8 µs | 0.7% |
| iter_total | 99.7 µs | 99.1 µs | 100.0 µs | 100.0% |

- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため iter_total と一致しない場合がある）: 99.5 µs

#### データ有効性（checksum 突合・要素単位検証。イシュー #965・#970）

- 不一致なし（相互突合できた 27 行の checksum が参照値と一致）
- **無効（要素誤差超過）**: burn/cuda/size=256/fresh — 要素誤差超過 fail=10538/65536, max_abs=1.581e-03, max_rel=1.554e+00, rescued=0, bound=1.907e-06
- **無効（要素誤差超過）**: burn/cuda/size=512/fresh — 要素誤差超過 fail=42361/262144, max_abs=2.343e-03, max_rel=1.972e+00, rescued=0, bound=3.815e-06
- **無効（要素誤差超過）**: burn/cuda/size=1024/fresh — 要素誤差超過 fail=169929/1048576, max_abs=3.643e-03, max_rel=1.965e+00, rescued=0, bound=7.629e-06
- **無効（要素誤差超過）**: burn/cuda/size=2048/fresh — 要素誤差超過 fail=681407/4194304, max_abs=4.941e-03, max_rel=1.987e+00, rescued=47, bound=1.526e-05
- **無効（要素誤差超過）**: burn/cuda/size=4096/fresh — 要素誤差超過 fail=2728488/16777216, max_abs=7.117e-03, max_rel=1.997e+00, rescued=562, bound=3.052e-05
- **無効（要素誤差超過）**: fandhe-ai/cuda/size=256/fresh — 要素誤差超過 fail=10538/65536, max_abs=1.581e-03, max_rel=1.554e+00, rescued=0, bound=0.000e+00
- **無効（要素誤差超過）**: fandhe-ai/cuda/size=512/fresh — 要素誤差超過 fail=42361/262144, max_abs=2.343e-03, max_rel=1.972e+00, rescued=0, bound=0.000e+00
- **無効（要素誤差超過）**: fandhe-ai/cuda/size=1024/fresh — 要素誤差超過 fail=169929/1048576, max_abs=3.643e-03, max_rel=1.965e+00, rescued=0, bound=0.000e+00
- **無効（要素誤差超過）**: fandhe-ai/cuda/size=2048/fresh — 要素誤差超過 fail=681454/4194304, max_abs=4.941e-03, max_rel=1.987e+00, rescued=0, bound=0.000e+00
- **無効（要素誤差超過）**: fandhe-ai/cuda/size=4096/fresh — 要素誤差超過 fail=2729050/16777216, max_abs=7.117e-03, max_rel=1.997e+00, rescued=0, bound=0.000e+00
- **無効（要素誤差超過）**: candle/cuda/size=256/fresh — 要素誤差超過 fail=10522/65536, max_abs=1.582e-03, max_rel=1.554e+00, rescued=0, bound=1.907e-06
- **無効（要素誤差超過）**: candle/cuda/size=512/fresh — 要素誤差超過 fail=42387/262144, max_abs=2.340e-03, max_rel=1.970e+00, rescued=0, bound=3.815e-06
- **無効（要素誤差超過）**: candle/cuda/size=1024/fresh — 要素誤差超過 fail=169971/1048576, max_abs=3.650e-03, max_rel=1.951e+00, rescued=0, bound=7.629e-06
- **無効（要素誤差超過）**: candle/cuda/size=2048/fresh — 要素誤差超過 fail=681418/4194304, max_abs=4.930e-03, max_rel=1.984e+00, rescued=33, bound=1.526e-05
- **無効（要素誤差超過）**: candle/cuda/size=4096/fresh — 要素誤差超過 fail=2726683/16777216, max_abs=7.167e-03, max_rel=2.000e+00, rescued=646, bound=3.052e-05

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
- **skipped-m4max-0.8.0.log**: bench-burn task=gemm device=metal size=512 mode=fresh extra=none : MEASURE_ERROR: gemm checksum is degenerate (0) — resu…
- **skipped-m4max-0.8.0.log**: bench-burn task=gemm device=metal size=1024 mode=fresh extra=none : MEASURE_ERROR: gemm checksum is degenerate (0) — res…
- **skipped-m4max-0.8.0.log**: bench-burn task=gemm device=metal size=2048 mode=fresh extra=none : MEASURE_ERROR: gemm checksum is degenerate (0) — res…
- **skipped-m4max-0.8.0.log**: bench-burn task=gemm device=metal size=4096 mode=fresh extra=none : MEASURE_ERROR: gemm checksum is degenerate (0) — res…
- **skipped-rtx3060-train.log**: bench-candle BUILD FAILED (--features cuda): Err(Os { code: 2, kind: NotFound, message: "No such file or directory" }) —…
- **skipped-rtx3060-train.log**: bench-burn task=gemm/train/infer device=cuda : ビルドは成功したが実行時に cubecl-cuda が「CUDA installation not found」パニックを出し（CUDA_PATH…
- **skipped-rtx3060-train.log**: bench-fandhe task=train device=metal mode=reuse : NOT RUN — macOS 実機がこのエージェント環境（x86_64 Linux）から到達不能。再現: Apple Silicon 実機…
- **skipped-rtx3060-train.log**: bench-fandhe task=train device=cuda mode=reuse (DGX Spark GB10 / sm_121) : NOT RUN — docs/real-hardware-verification-env…

