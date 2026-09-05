# Metal GEMM N=4096 カーネル純境界の candle 比ギャップ調査（イシュー #1143）

## 0. 背景

`docs/perf/metal-gemm-bottleneck-rediagnosis.md` §7.1b（#1103）実測により、N=4096 NN の
カーネル純境界（`dispatch_tiled_prepared`。転送非計測）は `CANDIDATES[2]`
（32×64×16, wm2wn2）が最良で 9.76〜9.91 TFLOPS、candle Metal（MLX steel gemm）は
同一境界で 13.17 TFLOPS。到達率 約 74〜75%。

本 issue の中核仮説は「candle-core 0.11.0 が M4 Max・f32・大形状で選ぶ
`TILE_64_64_16_2_2`（本実装の `CANDIDATES[0]`〈64×64×16, wm2wn2〉と完全一致の
タイル形状）が本実装では 4096 で 1.22 TFLOPS（`CANDIDATES[2]` の約 12%）へ崩壊して
いる、つまり candle 比ギャップの主因はタイル形状選択差ではなく同一形状での実装差
（レジスタ配置・フラグメントロード方式・協調ロードの添字コストのいずれか）」
というものだった（イシュー本文・実装計画参照）。

## 1. 計測環境・プロトコル

- 実機: Apple M4 Max（GPU 40 コア）。`sw_vers` 実測値 `ProductVersion: 26.6.2`
  （`BuildVersion: 25G83`）
- `xcrun -sdk macosx metal` はオフライン LLVM IR proxy 用の Metal Toolchain
  コンポーネント未導入のため利用不能（`newLibraryWithSource` によるランタイム
  コンパイルには影響しない。計画 Phase A step 4 は「任意」のため本調査では実施
  せず、代わりに `MTLComputePipelineState` 反射値〈§2〉で代替した）
- 計測プロトコル: `docs/perf/metal-bench-noise-protocol.md` 準拠
  （`bench_harness::protocol::run`・`MeasurementConfig::default()`〈warmup 20・
  計測 20・中央値〉・決定的シード `0xC0FFEE`）
- `pmset -g therm`: 計測前後とも目視確認した結果は「サーマル・パフォーマンス
  警告記録なし」。ただし証跡ファイルとしてコミットしているのは計測後の実行結果
  （`docs/perf/logs/metal-gemm-n4096-kernel-gap-1143/pmset_therm_after.txt`）の
  みで、計測前の実行結果は保存していない（記録漏れ。実測手順・実施自体への疑義
  ではない）
- 生ログ: `docs/perf/logs/metal-gemm-n4096-kernel-gap-1143/sweep_run{1,2}.log`
  （`cargo run -p fandhe-ai-backend-metal --example gemm_transpose_tile_sweep
  --release` を別プロセスで 2 回実行）

## 2. Phase A: 反射値によるレジスタ圧仮説（H1）の検証

計測当時（イシュー #1143 実装時点）は `MetalGemm::tile_pipeline_reflection`
（`crates/backend-metal/src/gemm.rs`）・`TilePipelineReflection` 構造体・
`gemm_transpose_tile_sweep.rs::dump_reflection` という診断専用の入口を追加し、
これで `MTLComputePipelineState` 構築後の `maxTotalThreadsPerThreadgroup`／
`staticThreadgroupMemoryLength` を全候補 × NN で取得した（ディスパッチを伴わず
秒未満で完了）。**これらの入口は本 PR の後続コミット（`fix(backend-metal):
診断専用の TilePipelineReflection 入口を削除する`）で削除済みであり、HEAD の
`crates/backend-metal/src/gemm.rs`・`examples/gemm_transpose_tile_sweep.rs` には
存在しない**（`#[doc(hidden)] pub` でも内部表現の公開 API 漏出は AGENTS.md
規約上 P1 に該当するという codex-review 指摘・PR #1168 を受け、診断完了後に
入口を撤去したため）。したがって下表の実測値は当時のローカル変更を適用した
状態で取得したものであり、`cargo run -p fandhe-ai-backend-metal --example
gemm_transpose_tile_sweep --release` を HEAD 上でそのまま実行しても再現しない
（値そのものは §1 の生ログ・下表に記録済みのため再取得の必要はない。再現する
には削除前のコミット〈`938629c` またはその直前の追加コミット〉の
`gemm.rs`／`gemm_transpose_tile_sweep.rs` を一時的に復元する必要がある）。

| candidate | requested_thread_count | max_total_threads_per_threadgroup | static_threadgroup_memory_length |
|---|---|---|---|
| cand0（64×64,wm2wn2＝candle TILE_64_64_16_2_2 と同形状） | 128 | 1024 | 0 |
| cand1〜cand6 | 64〜128 | 1024 | 0 |
| cand7（single simdgroup） | 32 | 1024 | 0 |
| cand8（32×64,wm1wn2。新規） | 64 | 1024 | 0 |

**結果**: 全候補で `max_total_threads_per_threadgroup=1024`（デバイス上限のまま・
要求スレッド数に対する不足なし）。`static_threadgroup_memory_length` は全候補で
`0`（`gemm_simdgroup_tiled` の threadgroup メモリは `[[threadgroup(0)]]` 引数へ
`setThreadgroupMemoryLength_atIndex` で実行時に確保するランタイム動的方式であり、
コンパイラ静的確保ではないため、この値は本カーネルでは指標にならない）。

**判断**: cand0（candle と同一タイル形状）はスレッドグループレベルの占有率上限に
不足がない。よって **H1（レジスタ圧によるスレッドグループ占有率の低下）は、
`MTLComputePipelineState` 反射値のレベルでは支持されない**。cand0 の 4096 での
崩壊（実測 1.02〜1.03 TFLOPS。§3）は、スレッドグループの起動可否ではなく、
スレッドあたりの実行効率（真のレジスタ spill・メモリアクセスパターン等、この
反射 API では見えない要因）に起因する可能性が高い。

**本判断の限界（codex-review 指摘・PR #1168）**: `maxTotalThreadsPerThreadgroup`
はコンパイラがレジスタ／メモリ使用量から静的に算出する**スレッドグループ起動数の
理論上限**であり、全候補でデバイス上限 1024 のままという実測は「起動可能な
スレッドグループサイズを削るほどのレジスタ圧ではない」ことまでしか棄却しない。
以下は本反射値だけでは棄却できない：(a) 実行時のレジスタ spill（thread-local
メモリへの退避によるレイテンシ増）が理論上限に影響せずスループットのみを下げる
経路、(b) simdgroup 単位でのスケジューリング・ウェーブ数競合による実効
occupancy 低下。したがって「H1（レジスタ圧起因の occupancy 低下）が支持され
ない」は**この反射値ベースの検証手法の範囲内での結論**であり、レジスタ圧に
起因する性能劣化そのものを一般に棄却するものではない（§3 の cand0 崩壊自体が
「スレッドあたり実行効率」側の要因＝広い意味でのレジスタ圧の可能性を排除して
いない点は本文が明記するとおり）。この限界を前提に、E1/E2 見送り判断
（次段落）は「反射値で H1 が完全に反証された」ことではなく「低コストな反射値
検証で occupancy 側の明確な支持材料が得られず、E1/E2 の期待値がコスト
（共有カーネルへの変更・既存テスト回帰リスク）に見合わないという相対的判断」
として扱う。厳密なレジスタ spill 計測（Xcode Instruments 等）による棄却・
確証は §5 スコープ外へ引き継ぐ。

計画の判断基準（advisor 相談での確認）: H1 が反射値レベルで支持されない場合、
E1（`_Pragma("clang loop unroll(full)")` 付与）・E2（ソーステキスト特殊化による
厳密サイズ配列化）はいずれも低期待値（占有率制約ではない要因への対症療法に
なりうる）と判断し、E6（既存機構での候補追加測定）へ予算を振り向けた。
E3〜E5（フラグメントロード方式変更・協調ロード再構成・`FINE_BARRIER_ENABLED`／
`SWIZZLE_ENABLED` 切替）は本調査では実施しない（§5「スコープ外」）。

## 3. Phase B/E6: 新候補 `(32,64,16,1,2)` の追加測定

`tile.rs::CANDIDATES` の index 8（末尾追加。既存 index 0〜7 は不変）に
`(bm=32, bn=64, bk=16, wm=1, wn=2, staged=true)` を追加した（`cand2`
〈32×64、4 simdgroup 分担〉の `wm`/`wn` を 1/2〈2 simdgroup 分担〉へ落とした
MLX steel classic 経路の未収録構成）。`examples/gemm_transpose_tile_sweep.rs`
の `candidates()`（9 要素へ拡張）・`shapes()`（10 形状）で 2 回計測した。

N=4096 NN（正方立方。中核対象）:

| candidate | run1 (TFLOPS) | run2 (TFLOPS) |
|---|---|---|
| cand0（64×64,wm2wn2） | 1.0214 | 1.0193 |
| cand1（64×32,wm2wn2） | 8.1917 | 7.8563 |
| **cand2（32×64,wm2wn2。現行最良）** | **9.6479** | **7.6978** |
| cand3（32×32,wm2wn2） | 7.3033 | 6.3560 |
| cand4（64×64,wm1wn2） | 0.3641 | 0.3579 |
| cand5（64×32×32,wm2wn2） | 7.4861 | 7.0724 |
| cand6（64×32×8,wm4wn1） | 7.0722 | 6.6548 |
| cand7（single simdgroup 8×8） | 2.4420 | 2.3373 |
| **cand8（32×64,wm1wn2。新規）** | **0.9739** | **0.9556** |

**結果**: cand8 は cand2 の約 8〜10 分の 1 で明確に劣後する（プロセス間変動
〈`metal-gemm-tile-table.md` 冒頭の run1/run2 数% 差〉の範囲を大きく超える）。
他形状（512/1024/2048/2048×2048×64/2048×2048×512/1536×1024×1024/1024×1536×1536/
4096×1024×1024/1024×4096×1024。全 10 点 × NN/NT/TN/TT）でも cand8 は一度も
最良候補にならない（`sweep_run{1,2}.log` 全 129 行を確認）。

**注目すべき副次的観察**（本調査のスコープ外の追加知見）: `wm=1` 系構成
（cand4: 64×64,wm1wn2、cand8: 32×64,wm1wn2）はいずれも 4096 で著しく劣化する
（cand4 は cand0 よりさらに悪い 0.36 TFLOPS）。§2 の反射値はこれらも
`max_total_threads_per_threadgroup=1024`（不足なし）であるため、`wm=1`
（1 行のみの simdgroup 分担）に固有の別要因（simdgroup 間のロード共有パターン
差・スケジューリング干渉等）が疑われるが、本調査では未診断（§5「スコープ外」）。

## 4. 採否判断

計画 §3 step 12「改善が得られない場合」の判断基準に該当する:

- E6（新候補追加）は測定上の改善なし（明確な劣化）→ **不採用**。
  `select_with_occupancy_for_device` の `(4096,4096,4096)` は現行
  `CANDIDATES[2]` を維持する（変更なし）。
- E1〜E5（カーネル codegen 変更・フラグメントロード方式変更・協調ロード
  再構成・`FINE_BARRIER_ENABLED`／`SWIZZLE_ENABLED` 切替）は、H1 が反射値
  レベルで支持されないという §2 の結果を踏まえ、期待値対コスト（`gemm.metal`
  という全候補共有カーネルへの変更・既存 200 件超のテストへの回帰リスク）
  の比較から本調査では実施しない。

**本 PR での変更点**: (a) `crate::tile::CANDIDATES` への index 8 追加とその
収録・不変ガードテスト（HEAD に残存）、(b) 診断専用の
`MetalGemm::tile_pipeline_reflection`／`TilePipelineReflection` 構造体・
`gemm_transpose_tile_sweep.rs` の反射値ダンプ（`dump_reflection`）を計測用に
一時追加し §2 の実測後に削除（内部表現の公開 API 漏出という codex-review
指摘・PR #1168 への対応。HEAD には存在しない。上記「削除済み」の注記を参照）。
**本番選択ロジック（`tile::select`／`select_with_occupancy_for_device`）は
変更しない**（cand8 が劣後という測定結果のため）。

## 5. スコープ外（今後の切り出し候補）

- `wm=1` 系構成（cand4・cand8）が `wm=2` 系より一貫して大幅に劣化する原因の
  診断（§3 の副次的観察）。GPU counters による実測（`metal-gemm-bottleneck-
  rediagnosis.md` が既に「GPU Service が対象デバイス非対応」と報告済みのため、
  代替手段〈Xcode Instruments 等〉の要否を含め要検討）
- E1〜E5（ソーステキスト特殊化 codegen・フラグメントロード方式・協調ロード
  再構成・`FINE_BARRIER_ENABLED`／`SWIZZLE_ENABLED` の M4 Max 実測記録の空欄
  埋め）。`docs/backend-metal-async-copy-decision.md`・`metal-gemm-fine-
  barrier-ab.md`・`metal-gemm-tgid-swizzle-ab.md` は本 PR では更新しない
- cand0（candle と同一タイル形状）が 4096 で崩壊する根本原因の特定
  （スレッドあたり実行効率の直接計測手段が現状ない）
- §2 の限界に基づく厳密なレジスタ spill 計測（Xcode Instruments の GPU
  Counters・Shader Profiler 等）による H1 の確証・棄却。`maxTotalThreadsPerThreadgroup`
  はスレッドグループ起動数の理論上限のみを表し、spill によるスループット低下
  （起動数を削らない経路）を直接には検出できないため（§2 追記・codex-review
  指摘・PR #1168）

これらは本 PR 本文で読者へ提示し、Issue 起票の要否はユーザー判断に委ねる
（`.claude/rules/out-of-scope-tracking.md`「ユーザーの承認なしに勝手に Issue を
起票しない」）。

## 6. 関連 doc

- `docs/perf/metal-gemm-bottleneck-rediagnosis.md` §7.1b・§8（candle 比ギャップの
  一次診断。本 doc は §8「candle 側カーネル純境界差の追加切り分け」の実施記録）
- `docs/perf/metal-gemm-tile-table.md`（`CANDIDATES` 全候補の形状別実測テーブル。
  本 doc の新候補 cand8 は §5 へ追記済み）
- `docs/backend-metal-mlx-classic-nax-decision.md` §1（MLX steel classic の
  未収録構成一覧。`(32,64,16,1,2)` は同 §1 が記録していた未決事項）
- `docs/perf/metal-gemm-candle-gate-remeasurement.md`（#1147。end-to-end reuse ゲート
  #1037 の正式判定確定。本 doc のカーネル純境界差はその要因の一部）

## 7. E1（loop unroll pragma）軽量実験（イシュー #1188）

### 7.0 目的・切り出し範囲

§5「スコープ外」が挙げた「`wm=1` 系構成（cand4・cand8）が `wm=2` 系より
一貫して大幅に劣化する原因の診断」候補実験 E1〜E5 のうち、2h 以内に収まる
**E1（`gemm_simdgroup_tiled` のアキュムレータ系ループへの `#pragma clang
loop unroll(full)` 付与）のみ**を切り出して実機実測した。E2〜E5・Xcode
Instruments 系の GPU counters 実測は本イシューでも対象外のまま（§7.6）。

### 7.1 §1.2 の再整理（作業仮説の再定式化）

§3 の生ログ（`docs/perf/logs/metal-gemm-n4096-kernel-gap-1143/sweep_run{1,2}.log`）
を再確認すると、崩壊は「`wm=1` 系」でも「N=4096 固有」でもなく、**simdgroup
あたりのアキュムレータ格子（`acc_rows × acc_cols`）が 16 以上の候補**に共通して
全形状（512/1024/2048/4096）で起きていた:

| candidate | sub_bm×sub_bn | acc_rows×acc_cols | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| cand0（64×64, wm2wn2。candle と同一タイル） | 32×32 | 4×4=16 | 0.23 | 1.26 | 1.19 | 1.14 |
| cand8（32×64, wm1wn2） | 32×32 | 4×4=16 | 0.71 | 1.18 | 1.08 | 1.06 |
| cand4（64×64, wm1wn2） | 64×32 | 8×4=32 | 0.47 | 0.84 | 0.47 | 0.39 |
| cand2（32×64, wm2wn2。現行最良） | 16×32 | 2×4=8 | 0.75 | 6.02 | 9.13 | 8.59 |
| cand1/3/5/6（いずれも acc ≤8） | — | 4〜8 | 正常 | 正常 | 正常 | 6.5〜8.6 |

（TFLOPS。base_run1・NN。`docs/perf/logs/metal-gemm-n4096-e1-unroll-1188/base_run1.log`）

`gemm_simdgroup_tiled`（`crates/backend-metal/src/shaders/gemm.metal`）は
`simdgroup_float8x8 acc[MAX_ACC][MAX_ACC]`・`a_frag[MAX_ACC]`・`b_frag[MAX_ACC]`
を、function constant（`BM`/`WM` 等）から導いた `acc_rows`/`acc_cols` を上限と
する `for` ループで走査する。ループが展開されない場合、これらレジスタ配列への
添字が実行時変数のままになり、配列が thread-local メモリへ降格（spill）しうる。
これは CUDA 側 `docs/perf/cuda-gemm-mma-ldmatrix-double-buffer.md` が記録した
「`#pragma unroll` が cosmetic ではなく必須である」機構と同型であり、#1143 §2 が
「反射値では棄却できない」と留保した「起動数を削らない spill 経路」に該当する
（仮説 H2）。E1 はこれを loop unroll pragma の付与で直接検証する。

### 7.2 計測環境・プロトコル

- 実機: Apple M4 Max（macOS 26.6.2）。`docs/perf/metal-gemm-n4096-kernel-gap.md`
  §1 と同一機
- 変更範囲: `gemm_simdgroup_tiled`（f32 経路のみ。`gemm_simdgroup_tiled_f16`・
  `gemm_tiled`・`gemm_simdgroup` は不変）のアキュムレータ系ループ全 10 箇所
  （acc 初期化 2・staged フラグメントロード 2・staged MMA 発行 2・
  direct-load MMA 発行 2・エピローグストア 2）へ
  `#pragma clang loop unroll(full)` を挿入
- 計測 example: `cargo run -p fandhe-ai-backend-metal --example
  gemm_transpose_tile_sweep --release`（`CANDIDATES` 全 9 候補・NN/NT/TN/TT・
  warmup 20・計測 20・中央値・決定的シード `0xC0FFEE`。§3 と同一プロトコル）
  および `cargo run -p fandhe-ai-backend-metal --example gemm_bench --release`
  （本番 `dispatch_auto` 経路の前後比較）
- 実行順: base（HEAD 未改変）→ E1 適用 を 3 往復（base_run{1,2,3}／
  e1_run{1,2,3}）。`git stash` で pragma の適用・解除を切り替え、各 run 前に
  `git diff --stat` をログ冒頭へ記録して状態を確認できるようにした
  （`docs/perf/logs/metal-gemm-n4096-e1-unroll-1188/`）
- 計測中、run2/run3 にかけてマシン全体の TFLOPS 水準が低下する傾向が見られた
  （例: base cand2 4096 が 8.59→5.43→4.28 と単調減少）。`pmset -g therm`
  は計測前後とも warning レベル未記録だったが、比較は同時間帯に取得した
  base/E1 のペアで行い、絶対値の経時変化と混同しないよう注意した

### 7.3 結果: N=4096（全 9 候補・NN）

| candidate | base run1 | E1 run1 | base run2 | E1 run2 | base run3 | E1 run3 | 改善率（run1 基準） |
|---|---|---|---|---|---|---|---|
| cand0（acc=16） | 1.14 | **10.10** | 1.06 | 5.39 | 0.78 | 5.08 | **約 8.9 倍** |
| cand4（acc=32） | 0.39 | **8.21** | 0.38 | 4.84 | 0.32 | 5.12 | **約 21 倍** |
| cand8（acc=16） | 1.06 | **7.99** | 0.90 | 4.77 | 0.85 | 5.27 | **約 7.5 倍** |
| cand1（acc=8） | 8.59 | 8.62 | 5.85 | 3.45 | 3.63 | 3.63 | 約 1.0 倍（横ばい） |
| cand2（acc=8。本番採用） | 8.59 | 9.05 | 5.43 | 3.80 | 4.28 | 4.06 | run1: +5%／run2: -30%／run3: -5% |
| cand3（acc=8） | 6.76 | 7.48 | 4.99 | 3.60 | 3.68 | 3.61 | run1: +11%／以降は横ばい〜微減 |
| cand5（acc=8） | 6.48 | 6.99 | 5.02 | 4.46 | 4.45 | 4.27 | 横ばい |
| cand6（acc=8） | 7.03 | 7.00 | 4.81 | 3.87 | 4.43 | 3.64 | 横ばい〜微減 |
| cand7（acc=8。single simdgroup） | 2.24 | 2.26 | 1.72 | 1.53 | 1.53 | 1.63 | 横ばい |
| NT（strided classic） | 1.37 | 1.53 | 1.08 | 1.01 | 1.06 | 1.02 | 横ばい〜微増 |
| TN（strided classic） | 1.36 | 1.47 | 1.07 | 1.01 | 1.03 | 1.08 | 横ばい〜微増 |
| TT（strided classic） | 1.35 | 1.44 | 1.06 | 0.86 | 1.04 | 1.05 | 横ばい〜微減 |

（TFLOPS。全ログ `docs/perf/logs/metal-gemm-n4096-e1-unroll-1188/{base,e1}_run{1,2,3}.log`）

`acc_rows*acc_cols>=16` の 3 候補（cand0/4/8）は 3 run すべてで同方向・大幅
（4.8〜21 倍）に改善し、H2 を強く支持する結果を得た。`acc<=8` の候補（本番が
実際に返す cand2 を含む）は改善・劣化とも数 % 〜 30% 程度に収まり、run2 の
cand2 -30% を除けば §3 が記録した既存のプロセス間変動（4096 で最大約 20%）
の範囲内と解釈できる。cand2 の run2 の下振れは、同時刻の他候補（cand1/3/5/6）
も base run2 で軒並み run1 比 30〜40% 低い水準（マシン全体の熱・負荷起因の
可能性）にあることと整合し、E1 固有の劣化と断定する根拠はない。

### 7.4 結果: N=512/1024/2048（run1・NN。cand0/4/8 抜粋）

| candidate | 512 base→E1 | 1024 base→E1 | 2048 base→E1 |
|---|---|---|---|
| cand0（acc=16） | 0.23→0.66 | 1.26→5.58 | 1.19→9.85 |
| cand4（acc=32） | 0.47→1.03 | 0.84→5.82 | 0.47→8.09 |
| cand8（acc=16） | 0.71→0.57 | 1.18→5.00 | 1.08→9.03 |
| cand2（本番採用・参考） | 0.75→0.82 | 6.02→6.16 | 9.13→9.51 |

512 では cand8 のみ横ばい〜微減（他 2 候補は改善）だが、1024/2048 では
cand0/4/8 すべてで 4〜17 倍の大幅改善が確認でき、崩壊が「N=4096 固有」では
なく `acc_rows*acc_cols>=16` の候補に形状横断で共通する現象であることを
裏付ける（§7.1 の再整理どおり）。cand2（本番）は非後退。

### 7.5 本番 `dispatch_auto` 経路の before/after（N=4096・3 run 中央値）

`cargo run -p fandhe-ai-backend-metal --example gemm_bench --release` の
`dynamic_tile_auto_tflops`（`crate::tile::select` が実運用で返す構成。
N=4096 では cand2 相当）:

| 状態 | run1 | run2 | run3 | 中央値 |
|---|---|---|---|---|
| base | 4.168 | 4.083 | 3.887 | 4.083 |
| E1 | 4.310 | 4.157 | 4.422 | 4.310 |

E1 適用後の中央値は base 比 **約 +5.6%** で非後退（`docs/perf/logs/
metal-gemm-n4096-e1-unroll-1188/gemm_bench_{base,e1}_*.log`）。§7.3 の
cand2 単体計測で見えた run2 の下振れは、本番経路の 3 run では再現せず、
中央値ベースでは一貫して改善方向であることを確認した。

### 7.6 採否判断

計画 §5 の判定基準（有効: cand0/4/8 が base 比で明確に改善・2 run 以上同方向。
非後退: 本番選択候補が非後退）に照らし、**両条件を満たす（分岐 A）**:

- 有効性: cand0/4/8 は 3 run すべて・全計測形状（512〜4096）で同方向かつ
  大幅（4〜21 倍）に改善し、H2（unroll 未展開によるレジスタ配列の
  thread-local メモリへの降格）を強く支持する
- 非後退性: 本番 `dispatch_auto` 経路（N=4096）は 3 run 中央値で約 +5.6%
  改善。個別候補計測での cand2 run2 の -30% は、同時刻の他候補・base 側も
  同水準で低下しているマシン全体の熱・負荷変動と整合し、E1 固有の後退と
  結論づける根拠はない

上記に基づき **E1（unroll pragma）をコミットへ含める**。実施した追加検証:

- `cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture`
  のうち GEMM parity 系（`cpu_metal_parity`・`cpu_metal_f16_parity`・
  `cpu_metal_f16_tiled_parity`・`gemm_auto_parity`・`gemm_bias_act_parity`・
  `gemm_dynamic_tile_parity`・`gemm_f16_auto_parity`・`gemm_naive_parity`・
  `gemm_resident_parity`・`gemm_simdgroup_parity`・`gemm_strided_parity`）は
  E1 適用状態で全 green（loop unroll はアキュムレータの累算オペランド列・
  走査順を変更しないため、ビット同一という理論的期待どおり）。無関係な
  既存 flaky テスト（`command_batching_bench::pool_reuse_interleaved_with_
  tracked_steps_preserves_batching`）が HEAD（E1 未適用）でも同様に失敗する
  ことを個別に確認し、本変更に起因しないことを切り分けた
- `crates/backend-metal/tests/shader_source_evidence.rs` に
  `gemm_simdgroup_tiled_source_unrolls_accumulator_loops`（出現数 10 を固定）・
  `gemm_simdgroup_tiled_f16_source_does_not_unroll_accumulator_loops`（f16
  経路への波及なしを固定）を追加
- `crate::tile::select`／`select_with_occupancy_for_device`・`CANDIDATES` は
  本 PR で変更しない（受け入れ条件どおり）。4096 で cand0 が cand2 に迫る
  水準（10.10 対 9.05。run1）まで改善したが、候補選択ロジックの再チューニング
  （`CANDIDATES` 順位の再測定・`select` 分岐更新）は本 PR のスコープ外とし、
  後続 issue として整理する（§7.7）

### 7.7 スコープ外の更新

§5 の項目のうち、本イシューで扱った範囲を以下のとおり更新する:

- 「`wm=1` 系構成（cand4・cand8）が `wm=2` 系より一貫して大幅に劣化する
  原因の診断」→ **本イシューで H2（unroll 未展開によるレジスタ spill）を
  支持する強い実機実測根拠を得て解消**。ただし GPU counters 等による
  spill の直接確証（Xcode Instruments。§5 従来項目）は引き続き未実施
- E1 は本イシューで実施済み（採用）。E2〜E5（フラグメントロード方式変更・
  協調ロード再構成・`FINE_BARRIER_ENABLED`／`SWIZZLE_ENABLED` 切替との
  相互作用）は引き続き対象外
- **新規**: E1 採用により 4096 で cand0（candle と同一タイル形状）が cand2
  に迫る性能を示した。`CANDIDATES` の順位再測定・`select_with_occupancy_
  for_device` の分岐更新（cand0 を N=4096 の推奨候補へ格上げできるか）は
  本 PR では実施しない
- 上記いずれも Issue 起票の要否はユーザー判断に委ねる
  （`.claude/rules/out-of-scope-tracking.md`）

### 7.8 関連ログ

- `docs/perf/logs/metal-gemm-n4096-e1-unroll-1188/`: `base_run{1,2,3}.log`・
  `e1_run{1,2,3}.log`（`gemm_transpose_tile_sweep` 全形状・全候補）・
  `gemm_bench_base_{1,2,3}.log`・`gemm_bench_e1_{1,2,3}.log`（本番
  `dispatch_auto` 経路）・`pmset_therm_before.txt`・`pmset_therm_after.txt`
