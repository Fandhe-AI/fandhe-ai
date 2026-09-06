# イシュー #1289 実行ログ集計（E2 特殊化版 反射値・kernel_gpu A/B）

生ログ: `reflection.log`（AC-1）・`kernel_gpu_run{1..5}.log`（AC-2。5 プロセス起動）・
`smoke_source_specialized.log`（#1288 非後退確認）・`env_info.txt`・
`uptime_before_run{1..5}.txt`・`pmset_therm_before.txt`・`pmset_therm_after.txt`。

## 反射値（AC-1）

`tile::CANDIDATES` 全 9 候補 × base/head（NN）を確認し、全候補・両側で
`requested_thread_count`／`max_total_threads_per_threadgroup`／
`thread_execution_width`／`static_threadgroup_memory_length` が完全一致
（base 側・head 側の値に一切差分なし）。`resolved_tile == requested`
（フォールバック非経由）も全 9 候補で成立。詳細値は `reflection.log`。

## kernel_gpu（AC-2。5 プロセス起動）

各 run は 20 warmup + 20 測定（trial 偶奇で base→head／head→base を反転する
interleave）。値は GPU タイムスタンプ（`kernel_gpu`）の 20 測定の中央値。

### N=1024

| run | base median (ms) | head median (ms) | head/base |
|---|---|---|---|
| 1 | 0.4759 | 0.5817 | 1.222397 |
| 2 | 0.5685 | 0.6994 | 1.230211 |
| 3 | 0.5600 | 0.5797 | 1.035268 |
| 4 | 0.6974 | 0.8504 | 1.219321 |
| 5 | 0.7167 | 0.6290 | 0.877732 |

5 run 比の中央値: **1.219321**（head が base より約 22% 遅い方向。5 run 中
4 run が比 >1.03、うち 3 run が比 >1.2）。base 絶対値自体のばらつきが大きく
（0.4759〜0.7167 ms）、`docs/perf/metal-gemm-reuse-phase-1277` の N=1024
分母（1.0267 ms）より小さい値域に集中している（実行環境・負荷条件の違いに
よるものと推定。本イシューのスコープ外のため深掘りしない）。

### N=2048

| run | base median (ms) | head median (ms) | head/base |
|---|---|---|---|
| 1 | 3.7546 | 3.8021 | 1.012651 |
| 2 | 2.6917 | 2.7227 | 1.011548 |
| 3 | 3.4585 | 3.4093 | 0.985772 |
| 4 | 2.4737 | 2.5985 | 1.050447 |
| 5 | 3.7149 | 3.3930 | 0.913344 |

5 run 比の中央値: **1.011548**（±5% 帯内。有意差なし）。base 絶対値は
2.47〜3.75 ms とばらつきが大きく、`docs/perf/metal-gemm-reuse-phase-1277`
§11 が記録した N=2048 の二峰性（1.60〜7.28 ms）と整合する範囲内。

### N=4096

| run | base median (ms) | head median (ms) | head/base |
|---|---|---|---|
| 1 | 14.1349 | 14.3358 | 1.014208 |
| 2 | 14.1904 | 14.4123 | 1.015641 |
| 3 | 14.3084 | 14.5638 | 1.017854 |
| 4 | 14.2794 | 14.4653 | 1.013023 |
| 5 | 14.1561 | 14.5383 | 1.027000 |

5 run 比の中央値: **1.015641**（±5% 帯内だが 5 run 全てで head が base を
上回り一貫して微後退の方向。base 絶対値は 14.13〜14.31 ms と安定しており、
`docs/perf/metal-gemm-reuse-phase-1277` §11 の分母 13.7051 ms（v0.6.0 ピン
当時の計測）と近い水準）。

## 経路切り替えの証跡

全 15 run（3 サイズ × 5 run）で `base_spec_cache_len=0`／
`base_fc_cache_len>0`・`head_spec_cache_len>0`／`head_fc_cache_len=0` を
確認（base は function constant 経路のみ、head はソーステキスト特殊化
経路のみを通ったことの独立証拠）。

## 負荷状況

各 run 直前の `uptime` load average（1 分値）は 6.18〜8.44（他セッション
並走中。`uptime_before_run{1..5}.txt`）。`pmset -g therm` は計測前後とも
サーマル・パフォーマンス警告記録なし。
