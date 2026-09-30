# CUDA gemm N=256 起動固定費の診断（イシュー #2109）

親 #2099（Phase 3 負けセル対処）の sub。**計装・診断テスト・事前登録規則・
計測スクリプトまでを本 PR で整備し、GB10 での 5 run 実測は実機セッションへ
申し送る**（開発機は NVRTC 非搭載で実機テスト不可）。修正実装は対象外。

## 1. 課題と数値の扱い

- スコアボード 2026-09-19 版（`docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.out:14`）で GB10 の CUDA gemm N=256（セル G-CUDA-G256）は 2 位・対 candle 0.83×。生値は fandhe-ai reuse 92.216 µs／fresh 89.272 µs、candle fresh 76.433 µs（gap 約 15.8 µs。reuse が fresh より遅い逆転もある）
- **Issue 本文の「0.72×」は G-CPU-G256 の値で転記違いの可能性がある**（`loss-attribution-matrix.md` §8）。本診断は G-CUDA-G256 = 0.83× を対象とする。Issue 本文は編集しない
- 小形状では固定費が支配的になりうるが内訳は未計測。`docs/analysis/candle-0.11.0-cuda-path.md` §9-3 の「dispatch 多段」は未検証の仮説

## 2. 計測境界の非対称（BOUND）

- candle 側ベンチは A/B を計測ループの前に 1 回だけ device へ置く（`bench-candle/src/main.rs:111-112`）。fandhe-ai 側は毎反復 A/B を H2D する。これは起動固定費ではなく計測境界の非対称であり、H1 として区間分離して記録する
- 両者とも checksum（ホスト側 O(n²) の和）を計測窓に含む（H6。記録のみ）

## 3. 計装と診断テスト

| 対象 | 内容 |
|---|---|
| `crates/backend-cuda/src/gemm.rs`（`GemmLaunchDiagCounters`） | `run_f32_kernel`・`CudaGemm::with_driver_call` の driver 境界回数（scope・H2D・pool alloc・launch・D2H・sync）。thread-local。`cfg(any(test, internal-diagnostics))` 限定で、非該当ビルドは空の `#[inline(always)]` 関数（#2299 の 2 層規約）。増分位置は成功後のみ、早期 return 経路は数えない |
| `lib.rs` `diagnostics::gemm_launch_counters_snapshot`／`reset_gemm_launch_counters` | crate 外向け入口（`internal-diagnostics` 限定。内部型を出さず `[u64; 8]`） |
| `crates/backend-cuda/src/gemm_small_launch_cost_diag_tests.rs` | N ∈ {128, 256, 512} の多層分解（L0 `ops_total`／L1 `gemm_total`／L2 h2d_a・h2d_b・alloc_c・launch_issue・readback〈本番と同じ同期 1 回〉・teardown・driver_scope〈本番 with_driver_call 入退場〉／L2S kernel_wait・d2h〈追加同期を挟む非本番の分離用補助系列。l2_sum に含めない〉／D device event／E floor）。`gemm_small_launch_counts_exact` は 1 回の GEMM の件数を厳密断言。実機テストは `#[ignore]` |

- 新規 `unsafe`・依存・公開 API 面の変更なし
- 観測できないもの: cudarc 内部の `cuEventCreate`／`cuEventDestroy`・`cuMemAllocAsync`／`cuMemFreeAsync`・`cuCtxGetCurrent`。cudarc 0.19.8 では `new_event` が `bind_to_thread` の後に `cuEventCreate`（`safe/core.rs:551`）、`record_event` が event 生成＋record（同 `:751`）、`memcpy_htod` が `bind_to_thread` の後に async memcpy（同 `:1602`）を行う。event tracking 有効時は alloc 系が slice ごとに event を作る。実数は nsys（任意）で確認する
- device event は `CU_EVENT_DEFAULT` を明示（`new_event(None)` は DISABLE_TIMING）。event は計測区間の外で事前生成する
- `dev_kernel_b2b` は「1 回起動」と「9 回連続キューイング」の device 時間差 /8 で求める推定値（launch latency を差し引く近似。厳密値ではない）
- D2H は本番 `memory::readback` を 1 区間として測るだけ。宛先方式の帰属は #2107 の担当

## 4. v0.9.0..HEAD の差分の扱い

- `memory.rs`／`host_staging.rs`／`pool.rs`: 差分は #2299 の cfg ゲート化とドキュメント整理のみ（実行文の変更なし。本 PR で `git diff v0.9.0..origin/main` を確認）
- `gemm.rs`（+953 行）・`ops.rs`（+763 行）・`context_cache.rs`（+261 行）: 多くは cfg ゲートと新機能と見られるが、N=256 NN 経路の実行文への影響は未分類
- 未分類が残るため RULE.txt 4 に従い、GB10 セッションでは **HEAD path-patch の Layer A を必須の正式な突合相手として実施する**（RULE.txt 4。H5 は HEAD の matmul − L0 で判定し、registry =0.9.0 は ratio の分母・参考値のみ）。分類結果は本節へ追記する

## 5. 仮説

| ID | 仮説 | 帰属に使う区間 |
|---|---|---|
| H1 | 毎反復 H2D（入力が resident でない境界の非対称） | L2 h2d_a + h2d_b |
| H2 | launch と同期の往復 | launch_issue + (L2S kernel_wait − dev_kernel_b2b)。floor と照合 |
| H3 | D2H readback | L2S d2h |
| H4 | host 側 dispatch | L0 − L1、L1 − l2_sum |
| H5 | facade・autodiff・tape | HEAD path-patch Layer A matmul − L0 |
| H6 | checksum | 記録のみ |
| 補助 | teardown（drop 時の free・event 破棄）・cudarc 内部 | L2 teardown、E h2d_clone_drop − h2d_prealloc、nsys |

判定閾値（gap の 50% 以上で「支持」）などの詳細は
[`RULE.txt`](logs/cuda-gemm-small-launch-cost-2109/RULE.txt) を正とし、本節では複製しない。

## 6. 再実行しない既存実験

| 実験 | 判定 | 出典 |
|---|---|---|
| H2D pinned staging（#1585） | REJECT（既定 OFF 維持） | `docs/perf/cuda-h2d-pinned-staging.md` |
| per-call alloc の release threshold・同期割当（#1149） | 差し戻し確定 | `docs/perf/cuda-percall-alloc-pool-threshold-ab.md` §7 |
| 都度同期の除去（#1011／#1013） | 実装済み（readback 1 点へ集約） | `docs/perf/cuda-async-sync-removal-*.md`・`docs/backend-cuda-async-execution-design.md` |
| CUDA Graph step capture（#1350） | 中立・既定 OFF | `docs/backend-cuda-graph-step-capture-design.md` §5 |
| Stream-K・persistent・TMA Stage 1 | REJECT／調査済み | `loss-attribution-matrix.md` §5 |
| reuse フェーズ分解（N=1024〜4096。#1182／#1973） | 記録済み（N=256 は本件で新規計測） | `cuda-gemm-reuse-phase-breakdown.md` §12 |

## 7. 結果（GB10 実測待ち）

未記入。`logs/cuda-gemm-small-launch-cost-2109/orchestrate.sh` → `aggregate.py` の
出力を本節へ転記する（受入基準 2: 5 run 中央値・単位 µs、受入基準 3: launch
overhead の定量化）。

## 8. 起票候補（列挙のみ。起票はユーザー承認後）

- 入力の resident 化による計測境界の対称化（H1 が支配的な場合。compare 側の設計判断を含む）
- Legacy stream での event tracking 無効化（`unsafe` を伴うため要承認）
- `teardown` 削減（drop 時の free・event 破棄。teardown が支配的な場合）
- readback 宛先の再利用（#2108）・CUDA 推論 Graph capture（#2115）は既存 issue
