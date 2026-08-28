# Metal 固定オーバーヘッド（約 5 ms）の内訳診断（#927）

イシュー #927「docs(perf): Metal 固定オーバーヘッド（約 5 ms）の内訳診断」の実測記録。
トラッキング #920 → Phase 1 親 #921 配下の診断タスク（兄弟 #925/#926/#928 と並列実行）。

## 状態: 構造分析・診断ハーネスは確定（Linux worktree）。実測は Mac 実機セッション待ち

本イシューは「実装変更を伴わない調査・計測・記録タスクのみ」（親 #921 本文）のため
`crates/backend-metal/src/`・`shaders/gemm.metal` は変更していない。実行環境は Mac 実機
（`docs/real-hardware-verification-env.md` §1・§7「ローカル直接実行」）だが本セッション環境は Linux
のため、#487・#384 の先例に従い、Linux 側で完了できる範囲（診断 example の実装・計測手順・記録
テンプレート・判定基準の確定）のみを本 PR で行う。**§5「実測結果」・§6「判定基準」・§7「既存実測との
整合」・§8「Phase 2 への示唆」は Mac 実機セッションで
`cargo run -p fandhe-ai-backend-metal --example fixed_overhead_diagnosis --release -- --size=256,512`
を実行してから記入する。**

## 1. 背景・出典

PR #915 のフレームワーク横並び実測（`scripts/bench/framework-compare/results/summary.md`
「(a) GEMM」Metal 節）で、fandhe-ai の Metal GEMM は N=256 中央値 5.441 ms・N=512 中央値
5.724 ms とサイズにほぼ非依存の約 5 ms に張り付く（同条件 candle は 257.6 µs／519.0 µs）。

| N | フレームワーク | 中央値 | GFLOP/s |
|---|---|---|---|
| 256 | fandhe-ai | 5.441 ms | 6.2 |
| 256 | candle | 257.6 µs | 130.2 |
| 512 | fandhe-ai | 5.724 ms | 46.9 |
| 512 | candle | 519.0 µs | 517.2 |

同 summary.md「(b) MLP 学習」「(c) 推論スループット」節でも、fandhe-ai の Metal 経路は
学習 48.845 ms/step・推論 24.125 ms（同 batch=64・784→256→10 MLP）と、CPU 経路（学習
18.185 ms・推論 505.7 µs）や candle/Burn の Metal 経路（学習 0.75〜1.6 ms・推論
0.25〜1.5 ms）に比べ大きく劣後する。これは「GEMM 呼び出し回数 × 約 5 ms の固定費」で
桁が整合する（推論は forward 1 回の GEMM 呼び出し、学習は forward + backward 相当の
複数回呼び出しを含む）。

## 2. 計測窓の定義（固定費の所在の切り分け）

`scripts/bench/framework-compare/bench-fandhe/src/main.rs` の GEMM 計測窓は
`tape.var`（テンソル生成。76〜79 行目）の**後**・`matmul` 呼び出し（81 行目）の直前に
`Instant::now()` を開始し、ホスト実体化（checksum。83 行目）で閉じる。したがって
この固定費は `fandhe_ai::tape_for`（デバイス選択・初期化）のコストではなく、
**演算メソッド呼び出しごとの Metal 資源の都度構築**に由来する。

`crates/backend-metal/src/ops.rs::MetalBackendOps::gemm` のドキュメンテーションコメント
（構造体 `MetalBackendOps` 冒頭）が明記するとおり:

> `MetalContext`／`MetalGemm` は各メソッド呼び出し時に都度構築する
> （`backend-cuda::ops::CudaBackendOps` と同じ設計判断。TASK-1.9b の
> デバイスハンドル常駐が未着地のため。ハンドル常駐化は TASK-1.9b／1.9d
> 以降の最適化対象）。

実際、`MetalBackendOps::gemm`（`ops.rs:151-180`）は呼び出しのたびに
`MetalContext::new()` → `MetalGemm::new(&ctx)` → `gemm.dispatch_auto(...)` を実行する。

## 3. 構造分析（フェーズ分解）

`matmul` 1 回あたりに毎回発生するコストを以下のフェーズへ分解する。

| フェーズ | 内容 | 出典 |
|---------|------|------|
| P1 | `MTLCreateSystemDefaultDevice` | `crates/backend-metal/src/context.rs::MetalContext::new` |
| P2 | `newCommandQueue` | 同上 |
| P3 | `MetalContext::new`（P1+P2 に加え `supportsFamily(Apple7)`・`MetalOccupancyInfo::probe`〈IOKit FFI〉照会の合算） | `crates/backend-metal/src/context.rs:70-100` |
| P4 | `newLibraryWithSource_options_error`（`shaders/gemm.metal` 全体の MSL 実行時コンパイル） | `crates/backend-metal/src/pipeline.rs::compile_gemm_library` |
| P5 | `MetalGemm::new(&ctx)`（ライブラリコンパイル＋固定 5 パイプライン構築〈naive/tiled/simdgroup/simdgroup_f16/tiled_bias_act〉の合算。tile 構成別特殊化パイプラインはインスタンス単位の遅延キャッシュのため都度構築下では毎回コールド） | `crates/backend-metal/src/gemm.rs::MetalGemm::new_with_swizzle_and_fine_barrier`・`pipeline_for_tile` |
| P6 | 都度構築 end-to-end（`MetalBackendOps::new()` + `BackendOps::gemm` を毎反復。framework-compare の 1 反復と同等条件） | `crates/backend-metal/src/ops.rs::MetalBackendOps::gemm` |
| P7 | 対照: 資源再利用（`MetalContext`/`MetalGemm` を 1 回構築して `dispatch_auto` を反復。転送・カーネル実行・同期のみ） | `crates/backend-metal/src/gemm.rs::MetalGemm::dispatch_auto` |

導出値: `P6 − P7 ≒ 都度構築固定費`。`P3 + P5`（デバイス/キュー/caps/occupancy 構築 +
ライブラリ/パイプライン構築の合算）との整合を突合し、残差（tile 特殊化パイプライン・
バッファ確保・A/B アップロード等）を記録する。

## 4. 計測手段・手順

`crates/backend-metal/examples/fixed_overhead_diagnosis.rs`（本イシューで新規作成）。

- P1・P2・P4 はサイズ非依存のため 1 度だけ計測する。P3・P5・P6・P7 は size ごとに計測する。
- P4（MSL ライブラリコンパイル）のみ初回と 2 回目以降（`--iters` 回。既定 20 回）を分離集計し、
  システム側 Metal コンパイラキャッシュの温存効果を観測する。
- それ以外のフェーズは `bench_harness::protocol::run`（`MeasurementConfig::default()` =
  warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1）を使う。
- 計測窓は A・B のホスト→デバイス転送・カーネル実行・`waitUntilCompleted` 同期・C の
  デバイス→ホスト readback を含む end-to-end 壁時計時間（`gemm_diagnosis.rs::
  wall_measurement` と同じ計測境界）。
- ノイズ対策: アイドル時間帯での計測・反復中央値の採用（既存 `docs/perf/
  metal-gemm-bottleneck-diagnosis.md` §2 のノイズ対策方針を踏襲）。

実行コマンド:

```sh
cargo run -p fandhe-ai-backend-metal --example fixed_overhead_diagnosis --release -- --size=256,512
```

`--iters=<N>`（`N` は 20 以上）で warmup・計測回数を引き上げられる（未指定時は既定 20/20）。

## 5. 実測結果（Mac 実機セッションでの記入待ち）

| フェーズ | median (ms) | Q1 (ms) | Q3 (ms) |
|---------|------------:|--------:|--------:|
| P1 device_create | (未計測) | | |
| P2 queue_create | | | |
| P4 library_compile_first | | (単一サンプルのため Q1/Q3 なし) | |
| P4 library_compile_rest | | | |
| P3 context_new (size=256) | | | |
| P5 gemm_new (size=256) | | | |
| P6 rebuild_each_call (size=256) | | | |
| P7 reused_dispatch (size=256) | | | |
| P3 context_new (size=512) | | | |
| P5 gemm_new (size=512) | | | |
| P6 rebuild_each_call (size=512) | | | |
| P7 reused_dispatch (size=512) | | | |

### 突合検算（Mac 実機セッションで記入）

- `P6 − P7`（fixed_cost, size=256）と framework-compare 実測（5.441 ms）の整合
- `P6 − P7`（fixed_cost, size=512）と framework-compare 実測（5.724 ms）の整合
- `P3 + P5` と `P6 − P7` の残差（tile 特殊化パイプライン・バッファ確保等の見積り）
- `P4 library_compile_first` と `P4 library_compile_rest` の比（システムコンパイラ
  キャッシュの温存効果）

## 6. 判定基準（Mac 実機セッションで記入）

支配的要因 = 固定費（P6−P7）に占める中央値シェアが最大のフェーズ。50% 超なら単独支配、
なければ上位要因を列挙する。

## 7. 既存実測との整合（Mac 実機セッションで記入）

参考ベースライン: `docs/perf/startup-cost-measurement.md`「Metal 実測結果」節（Apple M4 Max）は
プロセス初回（cold）で `device_init_secs` 中央値 約 35.280 ms（run1 cold）・`first_kernel_secs`
中央値 約 42.649 ms（run1 cold）。これは `MetalGemm::new` がプロセス起動後**最初**に
`gemm.metal` 全体を runtime コンパイルする経路であり、システム側 Metal コンパイラ
キャッシュが未温な状態でのコストを含む。本診断の P4/P5/P6/P7 はいずれもプロセス内
2 回目以降（warmup 済み・システムキャッシュ温存下）の都度構築費であり、上記
startup-cost の cold 値とは異なる母集団を計測している。この関係（cold 35〜43 ms 対
プロセス内 2 回目以降 約 5 ms）が本診断の実測値でどう再現されるかを記入する。

infer 24.125 ms（forward 1 回相当）・train 48.845 ms/step（forward+backward 相当の
複数回呼び出し）が「GEMM 呼び出し回数 × 固定費（P6−P7）」の概算とどの程度整合するかを
記入する。

## 8. Phase 2（#930）への示唆（Mac 実機セッションで記入）

キャッシュ化の対象階層（デバイス/キュー/ライブラリ/パイプライン/インスタンス単位 tile
キャッシュ）ごとの期待削減量。削減上限 ≒ P6 − P7 の実測値。
