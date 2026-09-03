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

`MetalGemm::tile_pipeline_reflection`（イシュー #1143 で追加。
`crates/backend-metal/src/gemm.rs`）で `MTLComputePipelineState` 構築後の
`maxTotalThreadsPerThreadgroup`／`staticThreadgroupMemoryLength` を全候補 × NN で
取得した（`gemm_transpose_tile_sweep.rs::dump_reflection`。ディスパッチを伴わず
秒未満で完了）。

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

**本 PR での変更点**: (a) `MetalGemm::tile_pipeline_reflection`（診断専用の
新規公開 API。`TilePipelineReflection` 構造体）の追加、(b)
`crate::tile::CANDIDATES` への index 8 追加とその収録・不変ガードテスト、(c)
`gemm_transpose_tile_sweep.rs` への反射値ダンプ・cand8 追加。**本番選択ロジック
（`tile::select`／`select_with_occupancy_for_device`）は変更しない**（cand8 が
劣後という測定結果のため）。

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
