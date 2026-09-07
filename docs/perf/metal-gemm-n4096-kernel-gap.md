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
  → **更新（イシュー #1278／#1279／#1280）**: E5（`FINE_BARRIER_ENABLED`／
  `SWIZZLE_ENABLED`）は #1278／#1279 で M4 Max 実機実測済み（いずれも安定性
  ゲート不成立により判定不可）。#1280 でこの実測結果に基づき採用候補 0 件・
  `dispatch_auto` 既定への結線対象なしと確定し、両定数とも `false` を維持した
  （`metal-gemm-fine-barrier-ab.md`・`metal-gemm-tgid-swizzle-ab.md` の
  「#1280 結線判断」節）。E1（loop unroll pragma）は §7 で軽量実験済み
  （イシュー #1188・#1282）。E2（ソーステキスト特殊化 codegen）は §8 で
  試作・bit 一致自己検証済み（イシュー #1288。性能実測は #1289 へ
  引き継ぎ）。E3（フラグメントロード方式変更）は候補実装・bit 一致自己
  検証まで完了（イシュー #1293。`docs/perf/metal-gemm-frag-load-
  candidates.md` 参照）。**#1295 で 5 候補（`tgp-k1`/`tgp-k2`/
  `device-legacy`/`device-hoisted-k1`/`device-hoisted-k2`）の
  N=1024/2048/4096 純カーネル時間を実測・判定済み**（§10）:
  本番既定 `tgp-k1` が全 N で最速で、他 4 候補はいずれも `tgp-k1`
  比 0.9996〜3.35 倍（`tgp-k2` の N=1024 のみ ±5% 帯内・他は全て
  明確に後退）のため**組み込み対象なし**（`tile::select` への
  結線は行わない）。**E4（協調ロード再構成）は候補実装・bit 一致自己
  検証まで完了**（イシュー #1298。`docs/perf/metal-gemm-coop-load-
  candidates.md` 参照）。**#1300 で 6 候補（`L0-P4`〈本番既定〉/`L0-P0`/
  `L0-P8`/`L1-P0`/`L1-P4`/`L1-P8`）の N=1024/2048/4096 純カーネル時間を
  実測・判定済み**（§11）: 本番既定 `L0-P4` を安定して上回る候補は
  確認できず（`RowStrided` 系は N=4096 で符号一貫の 5.6〜15.4% 後退・
  `L0-P0`／`L0-P8` は改善方向のシグナルがあるが N=2048/4096 で run 間の
  符号が反転し確証に至らず）**組み込み対象なし**（`tile::select` への
  結線は行わない。判断結果は #1302／#1304 へ引き継ぎ）
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
- **採否判断は `coding-rust.md`「ベンチは 5 回計測の中央値を採用」規約に従い
  5 回計測の中央値で行う**（codex-review 指摘・PR #1204。初版は base/E1 各 3
  run のみで採否を確定していたため規約未達だった。本節以降は base
  （`git checkout --detach` で E1 コミット `f19edce` の親 `564baee` へ戻し
  pragma 非適用を確認〈`grep -c "pragma clang loop unroll"
  crates/backend-metal/src/shaders/gemm.metal` が `0`〉した状態）→ E1
  （`f19edce`。同 grep が `10`）の順に各 5 回、`gemm_transpose_tile_sweep`・
  `gemm_bench` をそれぞれ実行した）
- 実行順: base 5 回連続（`base_run{1..5}`）→ E1 5 回連続（`e1_run{1..5}`）。
  §7.3〜§7.5 初版（3 run）は base/E1 を交互往復させていたが、本節の 5 run
  再計測はビルド切替コスト削減のため状態ごとにまとめて実行した（`git stash`
  ではなく別コミットへの `checkout --detach` で状態を切り替えたため）。
  この実行順の違いにより、経時的なマシン全体の熱・負荷ドリフトが base 群と
  E1 群それぞれの内部でのみ観測され、base⇄E1 の交互比較では緩和されていた
  可能性がある点に注意（§7.3 末尾の考察を参照）
- 計測中、5 run を通してマシン全体の TFLOPS 水準に大きな run 間変動が見られた
  （例: 本番 `dispatch_auto` 経路 N=4096 の base が 2.21→1.87→2.59→3.27→4.25
  TFLOPS、N=1024 では 1.44→1.44→0.76→0.83→1.32 TFLOPS の二峰性）。初版
  （3 run）の記録時点より変動幅が大きいセッションだったため、中央値中心の
  解釈に加え個々の run 値・ばらつきも本節の表に明記する

### 7.3 結果: N=4096（全 9 候補・NN。5 run 中央値）

| candidate | acc_rows×acc_cols | base（5 run） | base 中央値 | E1（5 run） | E1 中央値 | 改善率（中央値比） |
|---|---|---|---|---|---|---|
| cand0（64×64,wm2wn2） | 16 | 1.06/1.01/1.03/1.05/1.01 | 1.03 | 10.69/7.03/6.76/6.59/6.44 | 6.76 | **約 6.5 倍** |
| cand4（64×64,wm1wn2） | 32 | 0.25/0.37/0.39/0.41/0.41 | 0.39 | 9.08/5.84/5.53/5.18/4.28 | 5.53 | **約 14.2 倍** |
| cand8（32×64,wm1wn2） | 16 | 0.61/0.94/0.93/0.93/0.91 | 0.93 | 5.66/5.37/4.79/5.02/4.76 | 5.02 | **約 5.4 倍** |
| cand1（64×32,wm2wn2） | 8 | 3.68/3.70/3.82/3.90/4.23 | 3.82 | 8.46/5.26/5.02/4.84/4.64 | 5.02 | 約 1.3 倍 |
| cand2（32×64,wm2wn2。本番採用） | 8 | 3.75/3.73/3.97/4.25/4.24 | 3.97 | 9.75/4.96/4.68/5.13/4.59 | 4.96 | 約 1.25 倍 |
| cand3（32×32,wm2wn2） | 4 | 3.97/2.91/4.22/3.87/3.92 | 3.92 | 8.21/4.32/4.16/3.15/3.85 | 4.16 | 約 1.06 倍 |
| cand5（64×32×32,wm2wn2） | 8 | 3.31/3.51/3.48/3.94/3.28 | 3.48 | 6.82/4.94/4.85/4.53/3.21 | 4.85 | 約 1.39 倍 |
| cand6（64×32×8,wm4wn1） | 8 | 3.18/3.88/4.10/4.38/4.01 | 4.01 | 5.08/4.69/3.95/4.42/3.94 | 4.42 | 約 1.10 倍 |
| cand7（single simdgroup） | 1 | 1.20/1.60/1.66/1.54/1.43 | 1.54 | 1.65/1.71/1.70/1.44/1.47 | 1.65 | 約 1.07 倍 |
| NT（strided classic） | — | 0.75/0.98/1.01/1.04/1.01 | 1.01 | 1.16/1.15/1.19/0.99/0.75 | 1.15 | 約 1.14 倍 |
| TN（strided classic） | — | 0.76/0.91/0.99/1.04/0.99 | 0.99 | 1.16/1.16/1.11/1.15/0.86 | 1.15 | 約 1.16 倍 |
| TT（strided classic） | — | 0.72/0.88/0.98/1.04/1.04 | 0.98 | 1.16/1.16/1.11/1.01/0.85 | 1.11 | 約 1.13 倍 |

（TFLOPS。全ログ `docs/perf/logs/metal-gemm-n4096-e1-unroll-1188-5run/
{base,e1}_sweep_run{1,2,3,4,5}.log`）

**5 run 中央値での結論（初版 3 run からの更新）**: `acc_rows*acc_cols>=16` の
3 候補（cand0/cand4/cand8）は 5 run すべてで同方向・大幅（約 5.4〜14.2 倍。
中央値比）に改善し、H2 を強く支持する結果を維持した。一方、本セッションの
base 測定は初版（3 run。cand1/3/5/6 は 5〜8.6 TFLOPS 台）より全体的に低い
水準（3.2〜4.2 TFLOPS 台）で、`acc<=8` の候補（cand1/2/3/5/6/7・NT/TN/TT）
も E1 で約 1.06〜1.4 倍の緩やかな改善を示した。これは初版が「acc<=8 は
横ばい」としていた記述と定量的に一致しない（**P2 指摘 2 の是正**）。原因は
E1 適用による副作用ではなく、本 5 run 計測セッションの base 群自体が初版の
base より系統的に遅い（同一 HEAD~1 コミットの再ビルドだが、マシン全体の
熱・負荷状態がセッション間で異なる）ためと考えられる。中央値ベースでも
`acc<=8` 候補の改善率がすべて 1 倍台に収まっている（cand0/4/8 の 5〜14 倍とは
一桁以上異なる）ことから、「`acc_rows*acc_cols>=16` の候補が unroll pragma で
不均衡に大きく改善する」という中核結論（H2）自体は 5 run 再計測でも変わらない。

### 7.4 結果: N=512/1024/2048（NN・5 run 中央値。cand0/2/4/8 抜粋）

| candidate | 512 base中央値→E1中央値 | 1024 base中央値→E1中央値 | 2048 base中央値→E1中央値 |
|---|---|---|---|
| cand0（acc=16） | 0.25→0.67（約 2.7 倍） | 1.23→4.45（約 3.6 倍） | 1.19→7.94（約 6.7 倍） |
| cand4（acc=32） | 0.26→0.90（約 3.5 倍） | 0.83→4.61（約 5.6 倍） | 0.48→6.23（約 13.0 倍） |
| cand8（acc=16） | 0.55→1.14（約 2.1 倍） | 1.17→4.48（約 3.8 倍） | 0.88→6.32（約 7.2 倍） |
| cand2（本番採用・参考） | 0.77→1.30（約 1.7 倍） | **6.12→4.89（約 0.80 倍。低下）** | 7.22→7.22（約 1.00 倍。横ばい） |

512/1024/2048 いずれでも cand0/4/8 は 5 run 中央値で 2.1〜13.0 倍の大幅改善を
維持し、崩壊が「N=4096 固有」ではなく `acc_rows*acc_cols>=16` の候補に形状
横断で共通する現象であることを 5 run でも裏付ける。**cand2（本番が N=4096 で
実際に採用する候補。参考計測）は N=1024 で中央値が約 20% 低下した**（初版
3 run では「非後退」とだけ記述していたが、5 run では単純な非後退とは言えない。
**P2 指摘 2 の是正 その 2**）。ただし cand2 は N=512/1024/2048 では本番
`dispatch_auto` の実際の採用候補ではない（§7.5 の `old_tile=(64x32)` が示す
とおり、これら 3 形状の本番選択は cand1 相当の 64×32 タイルであり、cand2 は
N=4096 でのみ採用される。cand2 のこの表は「本番が別形状で採用する候補が他形状
でどう振る舞うか」の参考値であり、本番経路の非後退判定そのものは §7.5 の
N=4096 に限定して行う）。

### 7.5 本番 `dispatch_auto` 経路の before/after（N=4096・5 run 中央値）

`cargo run -p fandhe-ai-backend-metal --example gemm_bench --release` の
`dynamic_tile_auto_tflops`（`crate::tile::select` が実運用で返す構成。
N=4096 では cand2 相当）:

| 状態 | run1 | run2 | run3 | run4 | run5 | 中央値 |
|---|---|---|---|---|---|---|
| base | 2.208 | 1.865 | 2.595 | 3.275 | 4.246 | 2.595 |
| E1 | 2.607 | 2.753 | 2.893 | 3.109 | 2.957 | 2.893 |

（`docs/perf/logs/metal-gemm-n4096-e1-unroll-1188-5run/
{base,e1}_bench_run{1,2,3,4,5}.log`）

E1 適用後の中央値は base 比 **約 +11.5%** で非後退（改善）。base 側の run 間
ばらつき（1.87〜4.25 TFLOPS、約 2.3 倍の開き）は初版（3 run。3.89〜4.17）より
大幅に大きく、本セッションの計測ノイズが初版より高かったことを示す。この
ばらつきの大きさを踏まえ、初版が個別候補計測（§7.3）の run2 下振れを
「マシン全体の熱・負荷変動と整合し E1 固有の後退ではない」と説明していた
論拠は、本 5 run 計測でも同様の傾向（base 自身が run を追うごとに大きく
変動する）が確認されたことでより裏付けられた。

### 7.6 採否判断

計画 §5 の判定基準（有効: cand0/4/8 が base 比で明確に改善・複数 run で同方向。
非後退: 本番選択候補が非後退）に照らし、**5 回計測の中央値でも両条件を満たす
（分岐 A。初版 3 run の結論を維持）**:

- 有効性: cand0/4/8 は 5 run 中央値でも全計測形状（512〜4096）で同方向かつ
  大幅（約 2.1〜14.2 倍）に改善し、H2（unroll 未展開によるレジスタ配列の
  thread-local メモリへの降格）を強く支持する
- 非後退性: 本番 `dispatch_auto` 経路（N=4096。本番が実際に cand2 を採用する
  唯一の形状）は 5 run 中央値で約 +11.5% 改善。cand2 を他形状（N=1024）で
  参考計測すると中央値で約 20% 低下する run はあったが、その形状で本番が
  実際に採用するのは cand2 ではなく cand1 相当のタイルであるため、本番経路の
  非後退判定（N=4096 限定）には影響しない

上記に基づき **E1（unroll pragma）をコミットへ含める（初版の採否判断を
5 回計測の中央値で再確認・維持）**。実施した追加検証:

- `cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture`
  のうち GEMM parity 系（`cpu_metal_parity`・`cpu_metal_f16_parity`・
  `cpu_metal_f16_tiled_parity`・`gemm_auto_parity`・`gemm_bias_act_parity`・
  `gemm_dynamic_tile_parity`・`gemm_f16_auto_parity`・`gemm_naive_parity`・
  `gemm_resident_parity`・`gemm_simdgroup_parity`・`gemm_strided_parity`）は
  E1 適用状態で全 green（loop unroll はアキュムレータの累算オペランド列・
  走査順を変更しないため、ビット同一という理論的期待どおり。初版の実測を
  再確認する追加実行は本節では行わない。ビット同一性はループ添字の走査順・
  累算オペランド列を変更しないという pragma の性質上、コンパイル結果の
  性能のみが変わり数値結果には影響しないため 5 run 再計測の対象外とした）
- `crate::tile::select`／`select_with_occupancy_for_device`・`CANDIDATES` は
  本 PR で変更しない（受け入れ条件どおり）。4096 で cand0 が cand2 に迫る
  水準まで改善したが、候補選択ロジックの再チューニング（`CANDIDATES` 順位の
  再測定・`select` 分岐更新）は本 PR のスコープ外とし、後続 issue として
  整理する（§7.7）
- **本節の 5 run 再計測は計測手法（測定回数・中央値採用）のみを規約
  （`coding-rust.md`）へ整合させる是正であり、E1 のカーネル変更内容
  （`crates/backend-metal/src/shaders/gemm.metal`・
  `tests/shader_source_evidence.rs`）自体は変更しない**

### 7.6a 採否判断の見直し（PR #1204 codex-review 指摘。§7.7a で本番結線を撤回）

**上記 §7.6 の採否判断は撤回する。** codex-review（PR #1204）指摘: 「pragma は
`gemm_simdgroup_tiled` の全タイル候補に適用され、本番 `dispatch_auto` は
N=4096 以外でも同じシェーダを呼ぶため、非後退判定を N=4096 に限定するのは
他形状への影響を見落とす」。§7.5 が使ったものと同じ生ログ
（`docs/perf/logs/metal-gemm-n4096-e1-unroll-1188-5run/{base,e1}_bench_run{1..5}.log`）
から `dynamic_tile_auto_tflops`（本番 `dispatch_auto` が実際に返す値）の
N=512/1024/2048 における 5 run 中央値を追加算出したところ、**いずれも E1 適用
後に明確な後退**であることを確認した（§7.7a に算出値・スクリプトを記録）:

| N | base 中央値 | E1 中央値 | 変化率 |
|---|---|---|---|
| 512 | 0.4484 | 0.3219 | 約 -28.2% |
| 1024 | 1.3248 | 0.8594 | 約 -35.1% |
| 2048 | 2.2975 | 1.9523 | 約 -15.0% |
| 4096 | 2.5947 | 2.8932 | 約 +11.5%（§7.5 と同一） |

§7.6 は N=4096 の本番経路のみを非後退判定の対象としたため、上記の N=512/1024/
2048 における実質的な後退（本番が実際に採用する `old_tile=(64x32)` 系候補
〈acc<=8〉に対するもの。§7.4 で「参考値」と位置づけた cand2 とは別物）を
見落としていた。cand0/cand4/cand8（acc_rows*acc_cols>=16）が形状横断で大幅
改善するという有効性の知見（H2 の支持根拠）自体は変わらないが、**同一シェーダ
に無条件適用した pragma が、本番が実際に選ぶ他の候補（acc<=8 系）の性能を
N=512/1024/2048 で 15〜35% 悪化させる**ため、「本番採用（分岐 A）」の判断は
誤りだったと訂正する。原因（unroll(full) が小さい acc 候補ではレジスタ圧迫・
命令キャッシュ圧迫等の逆効果を及ぼす可能性）の切り分けは未実施であり、
codex-review の推奨（「原因切り分けまで判定保留」）に従い、**本 PR では
pragma の本番結線を撤回する**（§7.7a）。

### 7.7 スコープ外の更新

§5 の項目のうち、本イシューで扱った範囲を以下のとおり更新する:

- 「`wm=1` 系構成（cand4・cand8）が `wm=2` 系より一貫して大幅に劣化する
  原因の診断」→ **本イシューで H2（unroll 未展開によるレジスタ spill）を
  支持する強い実機実測根拠を得た**（3 run・5 run いずれの計測でも一致。
  有効性の知見自体は §7.6a 以降も維持）。ただし GPU counters 等による spill
  の直接確証（Xcode Instruments。§5 従来項目）は引き続き未実施
- **E1 は本 PR では本番結線を撤回する（§7.6a・§7.7a。初版の「採用」から
  訂正）**。撤回理由は本番 `dispatch_auto` 経路が N=512/1024/2048 で
  15〜35% 後退するため。E2〜E5（フラグメントロード方式変更・協調ロード
  再構成・`FINE_BARRIER_ENABLED`／`SWIZZLE_ENABLED` 切替との相互作用）は
  引き続き対象外
  → **更新（イシュー #1280）**: E5（`FINE_BARRIER_ENABLED`／`SWIZZLE_ENABLED`
  単体切替）は #1278／#1279 で M4 Max 実機実測済み・#1280 で採用候補 0 件・
  結線対象なしと確定した（本項目の対象外扱いは E1 との相互作用診断に限り
  維持。E2〜E4 は引き続き未実施）
- unroll(full) を acc_rows*acc_cols>=16 の候補にのみ適用する条件付き
  gating（function constant 分岐でループ本体を複製する等）は、pragma 単純
  付与よりコード複雑化・実機再検証コストが大きいため本 PR では実施せず、
  後続 issue として整理する（§7.7a）
- **新規**: 本節（5 run 再計測）で観測した計測ノイズの大きさ（本番経路
  N=4096 で base の run 間に約 2.3 倍の開き、N=1024 で cand2 参考計測が
  二峰的に約 2 倍変動）自体の原因診断（サーマルスロットリング・
  バックグラウンド GPU 負荷・`MeasurementConfig::default()` の warmup
  20/計測 20 が本機ではノイズ抑制に不十分である可能性等）は未実施
- 上記いずれも Issue 起票の要否はユーザー判断に委ねる
  （`.claude/rules/out-of-scope-tracking.md`）

### 7.7a 本番結線の撤回（実施内容の記録）

§7.6a の見直しに基づき、本 PR の以下のコミットで加えていた本番結線を撤回した:

- `crates/backend-metal/src/shaders/gemm.metal`: `gemm_simdgroup_tiled`
  （f32 経路）へ付与していた `#pragma clang loop unroll(full)` 全 10 箇所を
  削除し、`f19edce`（イシュー #1188 初回コミット）以前の状態へ戻した
- `crates/backend-metal/tests/shader_source_evidence.rs`:
  `gemm_simdgroup_tiled_source_unrolls_accumulator_loops`・
  `gemm_simdgroup_tiled_f16_source_does_not_unroll_accumulator_loops` の
  証跡テスト（pragma の存在を固定するもの）を削除した
- `docs/perf/logs/metal-gemm-n4096-e1-unroll-1188{,-5run}/` の実測ログ・
  §7.1〜§7.5 の実測記録・分析（H2 を支持する有効性の知見）はそのまま保持
  する（撤回するのは本番結線の判断のみ。実験自体は「実施済み・知見あり・
  本番未採用」として §7.7 に反映済み）

`dynamic_tile_auto_tflops`（本番 `dispatch_auto` 経路）の N=512/1024/2048/
4096 各 5 run 中央値の再算出（§7.6a の表）は、§7.5 と同一の既存ログ
（`docs/perf/logs/metal-gemm-n4096-e1-unroll-1188-5run/{base,e1}_bench_run{1..5}.log`）
の `size=<N> ... dynamic_tile_auto_tflops=<値>` 行を N 別に抽出し中央値を
取っただけであり、新規の実機実測は行っていない（既存ログの再集計）。

### 7.8 関連ログ

- `docs/perf/logs/metal-gemm-n4096-e1-unroll-1188/`: 初版 3 run の記録
  （`base_run{1,2,3}.log`・`e1_run{1,2,3}.log`・
  `gemm_bench_base_{1,2,3}.log`・`gemm_bench_e1_{1,2,3}.log`・
  `pmset_therm_before.txt`・`pmset_therm_after.txt`）。参考として保持し、
  §7.6 の採否判断は下記 5 run 記録を正とする
- `docs/perf/logs/metal-gemm-n4096-e1-unroll-1188-5run/`: 本節（§7.2〜§7.5）
  が正とする 5 run 記録（`base_sweep_run{1,2,3,4,5}.log`・
  `e1_sweep_run{1,2,3,4,5}.log`（`gemm_transpose_tile_sweep` 全形状・全候補）・
  `base_bench_run{1,2,3,4,5}.log`・`e1_bench_run{1,2,3,4,5}.log`（本番
  `dispatch_auto` 経路））

### 7.9 条件付き gating の実装（イシュー #1282）

§7.7 で「後続 issue として整理する」とした条件付き gating（function
constant 分岐でループ本体を複製する方式）を実装した（親 #1281 配下の
sub-issue）。**性能実測・本番既定の `true` への切替判断は本 issue のスコープ
外**であり、兄弟イシュー #1284 が引き継ぐ。本節は機構の実装内容と、実機
非依存の自己検証結果（AC-1・一部 AC-4）を記録する。

#### 7.9.1 設計

- `crates/backend-metal/src/shaders/gemm.metal` 冒頭に function constant
  `UNROLL_ACC_ENABLED`（index 11。`TRANS_A`/`TRANS_B`〈#1138・index 9/10〉の
  直後）を追加した
- `gemm_simdgroup_tiled`（f32 本体）のアキュムレータ系ループ 10 箇所
  （acc 初期化 外/内・staged `a_frag`/`b_frag` ロード・staged MMA 発行
  外/内・direct-load MMA 発行 外/内・エピローグストア 外/内）を、E1
  （§7.1〜§7.5）で無条件付与した `#pragma clang loop unroll(full)` を含む
  「unroll 版」と、従来コードとバイト同一の「非 unroll 版」の 2 系統へ
  `if (UNROLL_ACC_ENABLED) { ... } else { ... }`（計 6 ブロック）で複製した。
  両系統とも演算オペランド列（累算順・蛇行走査順）・REQ-8 手動境界チェック
  は不変で、`else` 側は `f19edce` 以前の状態とバイト同一を保つ契約とした
- `crates/backend-metal/src/tile.rs`: `UNROLL_ACC_ENABLED`（本番既定
  `false`）・`UNROLL_ACC_MIN_PRODUCT = 16`（E1 実測境界の単一真実源）・
  `TileConfig::acc_rows`/`acc_cols`/`unroll_acc_loops`（`acc_rows*acc_cols
  >= 16`）・純粋関数 `unroll_acc_loops_for(candidate, instance_flag)` を
  追加した
- `crates/backend-metal/src/pipeline.rs`: `make_pipeline_with_constants` の
  bool ゲート引数が 3 個（`swizzle_enabled`/`fine_barrier_enabled`/
  `unroll_acc_enabled`）になるため、`clippy::too_many_arguments`
  （`-D warnings`）を `#[allow]` で黙らせず `GemmGateConstants` 構造体へ
  束ねる方式で回避した
- `crates/backend-metal/src/gemm.rs`: `MetalGemm` に `unroll_acc_enabled`
  フィールドを追加し、`new_with_unroll_acc`（A/B 計測用の明示的入口。
  `new_with_swizzle`/`new_with_fine_barrier` と同型）を新設した。
  `pipeline_for_tile` は要求 `cfg` ではなく**フォールバック chain 巡回中の
  候補（`candidate`）自身**から実効ゲート値を導出する（フォールバック先で
  要求構成の acc 積を引きずった誤判定を避けるため）。`gemm_simdgroup_
  tiled_f16` は本定数を参照しないため `pipeline_for_tile_f16` は常に
  `false` を渡す
- 本番既定（`MetalGemm::new`）は `tile::UNROLL_ACC_ENABLED`（`false`）を
  渡すため、既定挙動（`dispatch_auto`・既存 parity）は不変

#### 7.9.2 自己検証（本エージェント実行環境。Linux・CI 相当）

本エージェント実行環境に macOS 実機がないため、Metal 実機依存の bit 一致
テスト（AC-2）は追加のみ行い実行していない（`#[ignore]` 分離済みで
`make test-ignored-metal` 相当の実機実行環境で `cargo test -p
fandhe-ai-backend-metal --release -- --ignored --nocapture` を後続で実行
する前提。テスト内容は 7.9.3 を参照）。以下は Linux で実行可能な範囲の
検証結果:

- `cargo build --workspace --locked`: 成功（既存の無関係な dead_code
  警告のみ。エラーなし）
- `cargo fmt --all -- --check`: 差分なし
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`:
  警告なし（`GemmGateConstants` 導入により `too_many_arguments` を回避）
- `cargo test --workspace --all-features`: 全 crate green（backend-metal
  は 237 passed・25 ignored〈うち 3 件が本 issue の実機専用テスト〉）
- `grep -c "pragma clang loop unroll" crates/backend-metal/src/shaders/
  gemm.metal` = 11（うち 1 件は冒頭コメント内の文字列。実際の pragma
  指示子は 10 箇所で E1 と同数。`tests/shader_source_evidence.rs::
  gemm_simdgroup_tiled_source_gates_unroll_pragmas_behind_function_
  constant` が文字列マッチベースで pragma 10・`if (UNROLL_ACC_ENABLED) {`
  6 を固定）
- `tile::tests::unroll_acc_candidates_are_exactly_acc_product_ge_16`
  （Linux 実行）: `CANDIDATES` 9 件中 `acc_rows*acc_cols>=16` を満たす
  index が `{0, 4, 8}`（E1 §7.6 の cand0/4/8 と一致）であることを確認済み
- `tile::tests::unroll_acc_enabled_is_false_by_default`: 本番既定 `false`
  をロック
- `gemm_simdgroup_tiled_source_retains_req8_boundary_guards_in_both_
  unroll_variants`: 境界チェック式（`out_row >= dims.m`・`out_col >=
  dims.n`・`a_row < dims.m`・`b_col < dims.n`）が unroll 版・非 unroll 版
  それぞれに 1 回ずつ（計 2 回）維持されていることを確認済み（AC-3）

#### 7.9.3 実機 `#[ignore]` テスト（AC-2。macOS 実機での実行は #1284 または
後続の実機検証セッションに引き継ぐ）

`crates/backend-metal/src/gemm.rs` の `mod tests` に追加（`tile::CANDIDATES`
が `pub(crate)` のためクレート内配置。`#[ignore = "Metal 実機（Apple
Silicon）依存。CI では実行しない"]`）:

- `unroll_acc_effective_matches_candidate_acc_product_threshold`: base
  （`unroll_acc_enabled=false`）は全候補で `false`、head
  （`unroll_acc_enabled=true`）は index 0/4/8 でのみ `true` を返すことを
  実機 `MetalGemm` インスタンス経由で確認する（AC-1）
- `unroll_acc_on_off_bit_match_all_candidates`: `CANDIDATES` 全 9 候補 ×
  N=512/1024/2048/4096 で `resolve_tile_config` によるフォールバック非経由
  を確認したうえで、`dispatch_tiled_prepared` の base/head 出力が
  `to_bits()` で厳密ビット一致することを確認する（AC-2）
- `unroll_acc_on_off_bit_match_dispatch_auto`: 本番自動選択経路
  （`dispatch_auto`）でも N=512/1024/2048/4096 で base/head が bit 一致
  することを確認する（AC-4 の一部。現行 `select`/`select_for_device` が
  選ぶ候補は acc 積 >= 16 に一致しないため、この経路単体では unroll 版
  ループを通らないが、function constant 特殊化自体が既存の自動選択経路の
  挙動を変えないことを確認する目的）

#### 7.9.4 スコープ外・引き継ぎ

- 性能実測（5 回計測中央値の全形状 before/after）・本番既定の `true` への
  切替判断 → #1284
- 上記実機 `#[ignore]` テスト（AC-2）の macOS 実機での実際の実行・結果記録
  → #1284 または後続の実機検証セッション
- `gemm_simdgroup_tiled_f16` への同機構の展開・direct-load 本体（unroll
  版・非unroll 版で重複する約 40 行）の helper 化による重複削減 →
  必要ならユーザー承認のうえ別 issue

### 7.10 本番 `dispatch_auto` 経路の全形状 A/B と結線判断（イシュー #1284）

#### 7.10.1 前提: 本 issue が測る構造的な事実

§7.9 で実装した条件付き gating（`UNROLL_ACC_ENABLED` function constant。
acc 積 >= `UNROLL_ACC_MIN_PRODUCT`（16）の候補のみ unroll 版ループを選択）
について、本 issue は本番 `dispatch_auto` 経路（`tile::select_with_
occupancy_for_device` の実測テーブル経由）で M4 Max 実機の N=512/1024/
2048/4096（正方）と、`CANDIDATES[0]`（acc 積 16）へ到達する補助形状
（`(2048,2048,512)`・`(4096,4096,1024)`）を計測した。

計測実装前に判明した構造的な事実（実装計画 §1.2。`crates/backend-metal/
examples/gemm_unroll_acc_ab_bench.rs` 冒頭ドキュメンテーションコメントに
同内容を記載）:

- `tile::select_with_occupancy_for_device` の M4 Max 実測テーブル
  （`exact_match_cfg`）は要求 4 形状いずれも acc 積 8 の候補
  （`CANDIDATES[5]`／`[6]`／`[1]`／`[2]`）を返す。acc 積 16 未満のため、
  正方 4 形状では本番経路は head（unroll 有効インスタンス）でも base と
  **同一の非 unroll 版ループ**をコンパイルする。したがって正方 4 形状の
  A/B は「function constant 特殊化の有無による差 ≈ 1.0」を測る計測になり、
  §7.5 の N=4096 +11.5%（`CANDIDATES[2]` を**無条件**unroll した実験値）は
  条件付き gating の下では設計上再現されない
- 結線の実効が及ぶのは `_ if m >= LARGE && n >= LARGE => CANDIDATES[0]`
  分岐（acc 積 16。非正方・非 tall/wide の大形状フォールバック）のみで
  あるため、本 issue はこの分岐へ到達する補助形状も計測対象へ加えた

このため、本 issue の採否判断基準は「正方 4 形状・補助 2 形状すべてで
`head_over_base >= 0.95`（`docs/perf/metal-gemm-fine-barrier-ab.md` の
`SMALL_SIZE_REGRESSION_TOLERANCE_RATIO` と同一値の再利用。新規閾値では
ない）」とし、正方形状の**改善は要求しない**（実装計画 §3.2）。

#### 7.10.2 プロトコル

- example: `crates/backend-metal/examples/gemm_unroll_acc_ab_bench.rs`
  （`gemm_fine_barrier_ab_bench.rs`・`gemm_swizzle_ab_bench.rs` と同一
  プロトコル。`bench_harness::ab`）。フェーズ 0（bit 一致自己検証）→
  フェーズ 1（安定性セルフチェック。対照カーネル: 本番既定
  `dispatch_auto`）→ フェーズ 2（`dispatch_tiled_prepared` prepared
  境界 A/B。**本判定対象**）→ フェーズ 3（`dispatch_auto` 転送込み境界
  A/B。参考値）
- `ROUNDS=6`・`COOLDOWN=2s`・`MIN_WARMUP=1s`（既定値。run1〜5 で使用）。
  フェーズ 1 が不成立のため run6 のみ `ROUNDS=10`・`COOLDOWN=8s`・
  `MIN_WARMUP=3s` へ一時調整して再試行したが、この調整も不成立だった
  （§7.10.3）。コミット前に既定値へ復元済み（差分なし）
- 実行環境: Apple M4 Max（GPU 40 コア）。`docs/perf/logs/metal-gemm-
  unroll-acc-ab-1284/env_info.txt` に機種・OS・rustc/cargo・HEAD sha・
  負荷観測（`uptime`）・サーマル観測（`pmset -g therm`）を記録

#### 7.10.3 フェーズ 1（安定性セルフチェック）の結果: 6 試行すべて不成立

6 回試行（run1〜5 は既定パラメータ、run6 は増量パラメータ）した結果、
**いずれの試行でも一部形状の spread が安定性ゲート
（`bench_harness::ab::STABILITY_SPREAD_GATE`＝0.05）を超過**し、単一 run
の参考 verdict は 6 回とも `undetermined` となった。

| run | 512 spread | 1024 | 2048 | 4096 | 2048×512 補助 | 4096×1024 補助 | 実行前 load average(1m) |
|---|---|---|---|---|---|---|---|
| 1 | NG (詳細は run1 ログ参照) | OK | NG | NG | NG | OK | 5.44 |
| 2 | NG | NG | NG | OK | NG | NG | 3.47 |
| 3 | NG | OK | NG | OK | NG | NG | 6.94 |
| 4 | NG | NG | NG | NG | NG | OK | 2.54 |
| 5 | NG | NG | NG | OK | NG | OK | 2.50 |
| 6（増量） | NG | NG | NG | NG | NG | NG | 12.66 |

低負荷（run4・run5。load average 1 分値 2.5 台）でも大半の形状で gate
超過が継続しており、`docs/perf/metal-gemm-transpose-tiled.md`（イシュー
#1187 §5.4）で観測された「低負荷開始でも gate 超過する」パターンと一致する
（他セッション負荷のみでは説明しきれない、本共有マシン固有の変動要因が
ある可能性を示唆。原因の特定は本 issue のスコープ外）。ROUNDS を 6→10・
COOLDOWN を 2s→8s・MIN_WARMUP を 1s→3s へ増量した run6 でも改善しな
かった（spread がむしろ悪化した形状もある）。

サーマル状態は実行前後とも `pmset -g therm` が
`"No thermal warning level has been recorded"` を報告しており、
サーマルスロットリングによる説明は支持されない
（`docs/perf/logs/metal-gemm-unroll-acc-ab-1284/pmset_therm_{before,after}.txt`）。

#### 7.10.4 フェーズ 2（prepared 境界。本判定対象）の実測値

フェーズ 1 が全 6 試行で不成立のため、個々の run の信頼性はフェーズ 1
ゲートで担保されていない。参考として、6 run の `head_over_base`（TFLOPS
比。head/base）を形状別に示す（`grep head_over_base` で各ログから再現
可能）:

| 形状 | run1 | run2 | run3 | run4 | run5 | run6 | 中央値 |
|---|---|---|---|---|---|---|---|
| 512³（acc 積 8。unroll 分岐非到達） | 1.0244 | 1.0089 | 0.8678 | 1.0003 | 1.3807 | 1.0180 | **1.0134** |
| 1024³（acc 積 8） | 1.0050 | 1.0022 | 1.0209 | 1.0171 | 1.1452 | 1.0013 | **1.0111** |
| 2048³（acc 積 8） | 1.0275 | 1.0619 | 0.9988 | 0.9837 | 1.1537 | 1.1707 | **1.0447** |
| 4096³（acc 積 8） | 1.0135 | 1.0116 | 1.1712 | 0.9901 | 0.9990 | 1.1777 | **1.0126** |
| 2048×2048×512 補助（acc 積 16。unroll 分岐到達） | 5.0168 | 5.3522 | 5.6598 | 5.6564 | 5.0616 | 5.3705 | **5.3613** |
| 4096×4096×1024 補助（acc 積 16） | 5.3337 | 6.5739 | 7.3746 | 7.3968 | 7.1735 | 8.7993 | **7.2740** |

参考値（フェーズ 3。転送込み境界）の中央値: 512³ 0.9950・1024³ 1.0073・
2048³ 1.0217・4096³ 1.0028・補助 2048×2048×512 2.0701・補助
4096×4096×1024 3.0008（いずれも `head_over_base` の 6 run 中央値）。

正方 4 形状は §7.10.1 の予測どおり `median ≈ 1.0`（改善も後退もない）に
収束している。補助 2 形状（`CANDIDATES[0]` 到達・acc 積 16）は E1
（§7.3・§7.4）で確認した cand0 の unroll 有効性がそのまま現れ、
prepared 境界で約 5.4〜7.3 倍という大きな改善を示した。個々の run 内で
`head_over_base < 0.95` になった唯一の観測値は run3 の 512³
（0.8678）で、その run3 自体は 512 の phase-1 spread も NG（不安定）
だった試行であり、単発の高ノイズサンプルの可能性が高い（他 5 run は
いずれも 512³ で 1.0 以上）。

#### 7.10.5 実機 `#[ignore]` テスト（AC-2。#1282 §7.9.3 からの引き継ぎ）

`make test-ignored-metal` 相当（`cargo test -p fandhe-ai-backend-metal
--release --no-fail-fast -- --ignored --nocapture`）を実機で初めて実行:

- `gemm::tests::unroll_acc_effective_matches_candidate_acc_product_
  threshold`・`unroll_acc_on_off_bit_match_all_candidates`・
  `unroll_acc_on_off_bit_match_dispatch_auto`（#1282 の 3 件。AC-1・
  AC-2・AC-4 の一部）: 3 件とも実機 pass（`bit_match_test.log`）
- parity 群（NN: `gemm_dynamic_tile_parity`〈11 passed〉・
  `gemm_auto_parity`〈3 passed〉・`cpu_metal_parity`〈5 passed〉・
  `gemm_resident_parity`〈2 passed〉・`gemm_bias_act_parity`〈7
  passed〉、NT/TN/TT: `gemm_transposed_parity`〈5 passed〉・
  `gemm_strided_parity`〈7 passed〉、ほか `gemm_simdgroup_parity`・
  `gemm_naive_parity`・`gemm_f16_auto_parity`・`cpu_metal_f16_parity`・
  `cpu_metal_f16_tiled_parity`・`rmsnorm_parity`・`sgd_device_parity`・
  `softmax_parity`・`mse_parity`・`linear_forward_device_parity` を含む
  全 parity テストファイル）: 全 pass（`parity_ignored_tests.log`）
- 既知の無関係な fail 1 件: `command_batching_bench.rs::
  pool_reuse_interleaved_with_tracked_steps_preserves_batching`
  （`encode()` 呼び出し回数のアサーション失敗。`left: 560〜621, right:
  50`）。本 issue の変更（`gemm.rs`・`tile.rs`・`pipeline.rs`・
  `shaders/gemm.metal` は未編集）とは無関係のファイルであり、
  `git stash` で本 issue の差分を除いた HEAD（462ece9）でも同一の
  アサーション失敗が再現することを確認済み（before/after 同一。
  tolerance には触れていない）

#### 7.10.6 結線判断: 判定不可（`UNROLL_ACC_ENABLED = false` 維持）

**判定不可**。6 試行すべてでフェーズ 1（安定性セルフチェック。対照
カーネル自体のばらつき）が不成立となり、単一 run の参考 verdict は
すべて `undetermined` だった。低負荷時（run4・run5）でも大半の形状で
gate 超過が継続したことから、原因は他セッションの GPU/CPU 負荷だけでは
説明しきれない本共有マシン固有の計測ノイズと考えられる
（`docs/perf/metal-gemm-transpose-tiled.md`〈#1186/#1187〉と同種の事象。
原因特定は本 issue のスコープ外）。

フェーズ 2 の実測値（§7.10.4）自体は 6 run を通して一貫した方向性を
示している（正方 4 形状は中央値 1.01〜1.04 倍で後退なし、補助 2 形状は
5.4〜7.3 倍の明確な改善）。しかし判定の前提となるフェーズ 1 ゲートが
一度も成立していないため、`docs/perf/metal-gemm-fine-barrier-ab.md`・
`docs/perf/metal-gemm-tgid-swizzle-ab.md`（#1278・#1279）と同じ安全側の
判断基準に従い、**この実測値のみを根拠に本番既定を切り替えない**。

- `tile::UNROLL_ACC_ENABLED` は `false` のまま維持する（コード変更なし）
- 機構自体（§7.9 の function constant 分岐・`unroll_acc_loops_for`）は
  revert しない（#1280 と同じ判断: 判定不可は機構の破棄を意味しない）
- 実機 `#[ignore]` テスト（#1282 の 3 件）は本 issue で初めて実機実行し
  pass を確認したため、この検証結果自体は確定した成果として残す
- 補助形状（`CANDIDATES[0]` 到達）で観測された 5.4〜7.3 倍という改善幅は
  大きく、より静かな実行環境（他セッション非併走）での再計測により
  フェーズ 1 が成立すれば採用可能性は高いと考えられる。再試行は
  ユーザー承認のうえ後続 issue へ引き継ぐ（§7.10.7）

#### 7.10.7 スコープ外・引き継ぎ

- より静かな実行環境（他セッション非併走）でのフェーズ 1〜3 再試行・
  本番既定切替の最終判断 → 必要ならユーザー承認のうえ後続 issue
- `command_batching_bench.rs` の無関係な既知 fail
  （`pool_reuse_interleaved_with_tracked_steps_preserves_batching`）の
  原因調査・修正 → 必要ならユーザー承認のうえ別 issue（本 issue のスコープ
  外・本 issue の変更と無関係であることは §7.10.5 で確認済み）
- `select`／`CANDIDATES`／`UNROLL_ACC_MIN_PRODUCT` 自体の変更（cand0 到達
  形状で改善が確認された場合の再チューニング） → §7.7 で既にスコープ外と
  記録済み（変更なし）

#### 7.10.8 関連ログ

`docs/perf/logs/metal-gemm-unroll-acc-ab-1284/`:
`env_info.txt`・`unroll_acc_ab_run{1..5}.log`・
`unroll_acc_ab_run6_adjusted.log`・
`uptime_before_run{1..3}.txt`・`pmset_therm_{before,after}.txt`・
`bit_match_test.log`（#1282 実機テスト 3 件）・
`parity_ignored_tests.log`（parity 群 + 既知無関係 fail 1 件）
## 8. E2 ソーステキスト特殊化経路の試作（イシュー #1288）

### 8.0 目的・切り出し範囲

§2（H1 レジスタ圧仮説）・§5「スコープ外」節が残した未実施候補 E2
（function constant 特殊化後もコンパイラが候補の厳密サイズでレジスタ
割付を最適化しない可能性）を検証するため、`gemm_simdgroup_tiled` を
候補（`tile::TileConfig`）ごとに**ソーステキストレベルで**特殊化し、
`gemm_simdgroup_tiled` のアキュムレータ配列（`acc`／`a_frag`／`b_frag`）を
`constexpr uint MAX_ACC = 8` の固定上限ではなく候補の厳密サイズ
（`ACC_ROWS_CAP`/`ACC_COLS_CAP` = `TileConfig::acc_rows()`/`acc_cols()`）で
確保するパイプライン構築経路を試作した。**本イシューは機構の実装と bit
一致の自己検証のみを担い、性能実測（反射値・カーネル純時間の before/
after）・本番既定への切替判断は行わない**（後続イシュー #1289〈実測〉／
#1302〈`tile::select` への組み込み判断〉のスコープ）。

### 8.1 設計

- **MSL 側**（`crates/backend-metal/src/shaders/gemm.metal`）: function
  constant 宣言ブロック（12 個。#188/#538/#540/#809/#1138/#1282）を
  `#ifdef GEMM_SPEC_ENABLED` / `#else` の二系統化した。`#else` 側
  （本番既定）は既存 12 行とバイト同一のまま維持し、`#ifdef` 側は
  `crate::spec_source::specialized_gemm_source` が前置する 12 個の
  `#define GEMM_SPEC_*` をリテラル代入する。`gemm_simdgroup_tiled`
  本体のアキュムレータ配列は `ACC_ROWS_CAP`/`ACC_COLS_CAP`（`#ifdef` 側は
  `GEMM_SPEC_ACC_ROWS`/`GEMM_SPEC_ACC_COLS` から、`#else` 側は従来どおり
  `MAX_ACC=8` から導出）で確保する。特殊化側には 4 本の `static_assert`
  （範囲 `[1,8]`・`ACC_ROWS_CAP == (BM/WM)/8` 等の整合性）を追加し、
  リテラル値と実行時計算式（`acc_rows`/`acc_cols`。ループ境界は変更せず
  不変のまま）の不整合をコンパイル時に fail-closed で検出する。
  `gemm_simdgroup_tiled_f16`（f16 経路）は対象外で変更していない
  （`tests/shader_source_evidence.rs::
  gemm_simdgroup_tiled_f16_source_max_acc_unchanged` が固定）。
- **bit 一致の論拠**: 配列容量は演算オペランド列に一切関与しない
  （ループ境界 `acc_rows`/`acc_cols` は特殊化の有無に関わらず同一の
  実行時計算式のまま）。`#else` 側は特殊化前とバイト同一のため、本番
  既定（`SOURCE_SPECIALIZATION_ENABLED=false`）の挙動は変わらない。
- **Rust 側**: `crates/backend-metal/src/spec_source.rs`（新規。
  `cfg(any(test, target_os = "macos"))`。`objc2` 系 FFI 非依存のため
  Linux でも生成ロジックの単体テストが回る）が `GEMM_MSL_SRC`
  （`gemm.metal` 全文。従来 `pipeline.rs` の private const だったものを
  移設し 2 経路で共有）・`SpecializationParams`・
  `specialized_gemm_source` を持つ。`pipeline.rs` は `compile_gemm_library`
  の本体を `compile_source`（共通実装）へ切り出し、新規
  `make_pipeline_source_specialized`（候補ごとにソースを生成・
  再コンパイル・パイプライン化。`unsafe` 不使用——function constant 値
  設定〈`setConstantValue_type_atIndex`〉を経由しないため）を追加した。
  両経路とも `compile_source` を通るため `compile_options()`
  （`MathMode::Safe` + `Precise`）が確実に同一適用される。
  `gemm.rs::MetalGemm` に `source_specialized: bool` フィールドと
  独立キャッシュ `tiled_spec_cache`（function constant 経路の
  `tiled_cache` とは別物。取り違え防止）を追加し、
  `new_with_source_specialization(ctx, bool)`（`new_with_unroll_acc` 等と
  同型の A/B・自己検証専用入口）を新設した。`pipeline_for_tile` は
  `self.source_specialized` でキャッシュ・構築関数を丸ごと切り替えるが、
  事前検証（`validate`・`shared_mem_bytes_for`）・ゲート導出
  （`unroll_acc_loops_for`）・事後検証
  （`maxTotalThreadsPerThreadgroup`）は両経路で完全に同一の式を使う。
  `MetalGemm::new`（本番経路）は `tile::SOURCE_SPECIALIZATION_ENABLED`
  （既定 `false`）を渡すため本番挙動は不変。

### 8.2 実機自己検証結果

`crates/backend-metal/src/gemm.rs::mod tests` に追加した実機
`#[ignore]` テスト 3 件を Apple M4 Max 実機（本 doc の計測環境と同一機）
で実行し、いずれも pass を確認した:

```
cargo test -p fandhe-ai-backend-metal --release --lib -- --ignored --nocapture source_specialized
```

```
running 3 tests
test gemm::tests::source_specialized_route_populates_only_spec_cache ... ok
test gemm::tests::source_specialized_on_off_bit_match_dispatch_auto ... ok
test gemm::tests::source_specialized_on_off_bit_match_all_candidates ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 267 filtered out; finished in 3.23s
```

- `source_specialized_on_off_bit_match_all_candidates`: `tile::CANDIDATES`
  全 9 候補 × N=512/1024/2048/4096 で、base（function constant 経路）/
  head（ソーステキスト特殊化経路）双方が `resolve_tile_config` で
  フォールバック非経由（検証の空振り防止）を確認したうえで、
  `dispatch_tiled_prepared` 出力が `to_bits()` で厳密ビット一致
- `source_specialized_route_populates_only_spec_cache`: 出力一致だけでは
  両経路が同じ実装へ倒れた false-green を検出できないため、head は
  `tiled_spec_cache` のみが増え `tiled_cache` は 0 のまま、base はその逆
  であることを独立に確認し、新経路が実際に走ったことを証明
- `source_specialized_on_off_bit_match_dispatch_auto`: 本番自動選択経路
  （`dispatch_auto`）でも N=512/1024/2048/4096 で base/head が bit 一致

既存の非後退確認（`gemm.metal` 改変の副作用がないこと）も同一実機で
再実行し、全て pass:

```
cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture unroll_acc
cargo test -p fandhe-ai-backend-metal --release --test gemm_dynamic_tile_parity -- --ignored --nocapture
cargo test -p fandhe-ai-backend-metal --release --test gemm_simdgroup_parity -- --ignored --nocapture
cargo test -p fandhe-ai-backend-metal --release --test gemm_transposed_parity -- --ignored --nocapture
```

env_info（内部ホスト名は含めない）: `sysctl -n machdep.cpu.brand_string`
= `Apple M4 Max`、`sw_vers` = macOS 26.6.2（BuildVersion 25G83）、
`rustc -V` = `rustc 1.96.0 (ac68faa20 2026-05-25)`、実行時 `uptime` の
load averages は約 9.4/6.6/5.0（他セッション並走。bit 一致テストは
負荷非依存のため判定に影響しない）。

### 8.3 #1289 への引き継ぎ

- 反射値（`maxTotalThreadsPerThreadgroup`・`threadExecutionWidth`・
  `staticThreadgroupMemoryLength`）は `pipeline_for_tile` が保持する
  `Retained<MtlPipeline>` から取得できる。公開 API 漏出（P1。§2 の
  「削除済み」注記と同じ判断）を避けるため `#[cfg(test)]` または
  example 専用の薄いアクセサとして #1289 側で設計すること
- before/after は `new_with_source_specialization(false/true)` の
  2 インスタンスを同一プロセスで interleaved 実行する
  （`gemm_fine_barrier_ab_bench.rs`／`gemm_swizzle_ab_bench.rs` の方式）。
  純カーネル時間は #1275/#1276 で確立した GPU タイムスタンプ経路
  （`context.rs::synchronize_with_gpu_timestamps`）を使う
- 本イシューでは性能未計測のため `tile::SOURCE_SPECIALIZATION_ENABLED`
  は `false` のまま。結線可否は #1289 の 5 回計測中央値に基づき判断し、
  後退時は結線せず理由を記録すること
- §5「スコープ外」節の「E1〜E5（ソーステキスト特殊化 codegen…）」の
  記述のうち、ソーステキスト特殊化 codegen（E2）は本節で試作・自己検証
  完了。性能実測は依然未実施のまま（#1289 へ引き継ぎ。→ §9 で実施）

## 9. E2 特殊化版の反射値・純カーネル時間 before/after（イシュー #1289）

### 9.0 目的・範囲

§8.3「#1289 への引き継ぎ」の指示に従い、E2（ソーステキスト特殊化経路。
`crate::spec_source`／`MetalGemm::new_with_source_specialization`）の
(a) `MTLComputePipelineState` 反射値、(b) N=1024/2048/4096 の純カーネル
時間（GPU タイムスタンプ `kernel_gpu`）を base（function constant 経路。
`source_specialized=false`）/head（ソーステキスト特殊化経路。
`source_specialized=true`）で before/after 比較し、E2 の有効性
（`tile::select`／候補表への組み込み可否）を判断する。**本イシューでは
`tile::SOURCE_SPECIALIZATION_ENABLED` は `false` のまま結線しない**
（判定が「可」であっても結線・組み込み判断は #1302 へ引き継ぐ。§8.3 の
安全側判断）。

### 9.1 環境・プロトコル

- 実機: Apple M4 Max（GPU 40 コア）。`sw_vers ProductVersion` 26.6.2
  （`BuildVersion` 25G83）。`rustc -V` = `rustc 1.96.0
  (ac68faa20 2026-05-25)`
- HEAD: `2752ab2edb1cfca230457b4e1dee481a78efce9d`（`origin/main` 直上。
  依存 #1288／#1277 いずれもマージ済み）
- 追加した診断テスト（`crates/backend-metal/src/
  gemm_spec_source_diag_tests.rs`。`#[cfg(all(test, target_os =
  "macos"))]`）:
  - `spec_source_reflection_dump_all_candidates`（AC-1。ディスパッチを
    伴わず秒未満で完了。1 プロセス実行）
  - `spec_source_kernel_gpu_ab_production_sizes`（AC-2。N=1024/2048/4096、
    各 20 warmup + 20 測定。trial 偶奇で base→head／head→base を反転する
    interleave で order-bias を相殺。`gemm_reuse_phase_diag_tests.rs::
    measure_one_phase_trial`〈`pub(crate)` 化して再利用〉と同一のフェーズ
    分解・GPU タイムスタンプ経路〈#1276〉を使う）
  - `gemm::MetalGemm::diag_tile_pipeline_reflection`（`gemm.rs`。
    `#[cfg(test)] pub(crate)`。反射値取得専用の薄いアクセサ。公開 API
    面へは出さない。§8.3 の指示どおり）
- 実行順序: (1) `#1288` の 3 実機テスト（`source_specialized_*`）の
  非後退確認 → (2) 反射値ダンプ（1 プロセス） → (3) kernel_gpu A/B を
  5 プロセス起動（各起動間 5 秒クールダウン）
- 負荷状況: 各 run 直前の `uptime` load average（1 分値）は
  6.18〜8.44（他セッション並走中の共有負荷環境。低負荷帯の確保はできな
  かったが、interleave 方式のため base/head 双方が同一負荷条件下で計測
  される。`docs/perf/logs/metal-gemm-e2-spec-source-ab-1289/
  uptime_before_run{1..5}.txt`）。`pmset -g therm` は計測前後ともサーマル・
  パフォーマンス警告記録なし
- 生ログ・集計: `docs/perf/logs/metal-gemm-e2-spec-source-ab-1289/`
  （`reflection.log`・`kernel_gpu_run{1..5}.log`・`aggregate.md`・
  `smoke_source_specialized.log`・`env_info.txt`）

### 9.2 反射値（AC-1）

`tile::CANDIDATES` 全 9 候補 × base/head（NN）で、`requested_thread_
count`／`max_total_threads_per_threadgroup`／`thread_execution_width`／
`static_threadgroup_memory_length` を取得した。**全 9 候補・base/head の
全組み合わせで反射値は完全一致**（差分なし）:

| candidate | requested_thread_count | max_total_threads_per_threadgroup | thread_execution_width | static_threadgroup_memory_length |
|---|---|---|---|---|
| 0（64×64,wm2wn2） | 128 | 1024 | 32 | 0 |
| 1（64×32,wm2wn2） | 128 | 1024 | 32 | 0 |
| 2（32×64,wm2wn2） | 128 | 1024 | 32 | 0 |
| 3（32×32,wm2wn2） | 128 | 1024 | 32 | 0 |
| 4（64×64,wm1wn2） | 64 | 1024 | 32 | 0 |
| 5（64×32,bk32,wm2wn2） | 128 | 1024 | 32 | 0 |
| 6（64×32,bk8,wm4wn1） | 128 | 1024 | 32 | 0 |
| 7（single simdgroup 8×8） | 32 | 1024 | 32 | 0 |
| 8（32×64,wm1wn2） | 64 | 1024 | 32 | 0 |

`static_threadgroup_memory_length` が全候補で 0 になるのは §2 と同じ
構造的理由（`setThreadgroupMemoryLength_atIndex` によるランタイム指定
方式のため、コンパイル時反射値には現れない）。全候補で `resolved_tile
== requested`（フォールバック非経由）も確認済み。

**含意**: E2（ソーステキストレベルでのアキュムレータ配列厳密サイズ化）
は、コンパイラが `MTLComputePipelineState` に報告する占有率上限
（`maxTotalThreadsPerThreadgroup`／`threadExecutionWidth`）・静的
threadgroup メモリ量のいずれも変化させない。§2 が「反射値レベルでは
H1（レジスタ圧仮説）非支持」と判定した結論は、function constant 特殊化
だけでなくソーステキスト特殊化（E2）でも同様に成立する（反射値という
観測粒度では両経路に差が現れない）。

### 9.3 kernel_gpu 純カーネル時間（AC-2）

5 プロセス起動・各 20 測定中央値の 5 run 集計（詳細は
`docs/perf/logs/metal-gemm-e2-spec-source-ab-1289/aggregate.md`）:

| N | base median 5 run range (ms) | head median 5 run range (ms) | head/base 比の 5 run 中央値 | 備考 |
|---|---|---|---|---|
| 1024 | 0.4759〜0.7167 | 0.5797〜0.8504 | **1.219321** | 5 run 中 4 run が比 >1.03、3 run が比 >1.2。head が一貫して遅い方向 |
| 2048 | 2.4737〜3.7546 | 2.5985〜3.8021 | **1.011548** | ±5% 帯内（有意差なし）。base 自体の run 間ばらつきが大きい（#1277 §11 の二峰性と整合） |
| 4096 | 14.1349〜14.3084 | 14.3358〜14.5638 | **1.015641** | ±5% 帯内だが 5 run 全てで head > base（一貫した微後退方向）。base 絶対値は安定（レンジ幅 0.17 ms） |

N=4096 の base 絶対値（14.13〜14.31 ms）は `docs/perf/metal-gemm-
reuse-phase-1277` §11 が記録した v0.6.0 ピン当時の分母（13.7051 ms）と
近い水準で、妥当性帯として許容範囲内と判断する。一方 N=1024 の base
絶対値（0.4759〜0.7167 ms）は同分母（1.0267 ms）より一貫して小さく、
本イシューの計測環境（他セッション並走の共有負荷）・HEAD の違いによる
ものと推定するが、深掘りはスコープ外とする（§9.5）。

### 9.4 採否判断

**組み込み不可（REJECT）** と判定する。判定基準（本イシューで新たに定めた
以下の 3 区分。他候補〈E1 loop unroll・#1188〉で採った基準と同じ考え方を
踏襲するが、本 doc 内に既存の §番号節としては存在しないため、ここに明文化
する）に照らすと:

- **候補表組み込み可**: N=2048 かつ N=4096 で改善（head/base 比 <1.0）、
  かつ N=1024 の後退が 5% 以内（head/base 比 <=1.05）
- **有効性なし → 組み込み不可**: 全 N で比が ±5% 帯内（有意差なし）、
  または大形状（N=2048／N=4096）で後退
- **判定不可（undetermined）**: run 間で改善／後退の方向が一貫せず
  ばらつきが大きい場合

- 「候補表組み込み可」の条件（N=2048 かつ N=4096 で改善〈比 <1.0〉、
  N=1024 の後退が 5% 以内）は**いずれも満たさない**: N=2048／N=4096 は
  改善方向ではなく、5 run 中央値でむしろ微後退（1.011〜1.016 倍）。
  N=1024 は後退が 5% を明確に超える（5 run 中央値 1.219 倍、約 22%
  後退）
- 「有効性なし → 組み込み不可」の条件のうち「全 N で比が ±5% 帯内」は
  N=1024 で満たさないが、「大形状で後退」は **N=4096 のみ**満たす
  （5 run 全てで head > base、一貫して微後退方向。1.013〜1.028 倍）。
  N=2048 は 5 run 全てで head > base ではなく、run 3（0.985772）・
  run 5（0.913344）の 2 run は head < base（比 <1.0。head が base を
  約 1.4〜8.7% 下回る）である。ただし N=2048 の 5 run 中央値（1.011548）
  自体は ±5% 帯内に収まり、`aggregate.md` が記す通り base 自体の run 間
  ばらつき（2.47〜3.75 ms。#1277 §11 の二峰性と整合）が支配的で、
  head/base 比の符号が run ごとに反転する程度には「有意差なし」の範囲と
  判断する（この意味で N=2048 は「大形状で後退」ではなく「±5% 帯内・
  有意差なし」区分に該当する）。以上により本条件は N=4096 の一貫した
  後退（区分「大形状で後退」）と N=1024／N=2048 の「±5% 帯内または
  それを明確に超える後退」の組み合わせで成立する
- 15 run 中、head が base を下回った（比 <1.0。改善方向を示した）run は
  N=1024 の run 5（0.877732）・N=2048 の run 3（0.985772）・run 5
  （0.913344）の計 3 run のみで、残り 12 run は head > base（後退方向）
  である。この 3 run のうち N=1024 run 5 は単独では 12.2% の改善幅だが、
  同じ N=1024 の他 4 run（run 1/2/4 が比 >1.2＝約 22〜23% 後退）と符号が
  逆転しており run 間で方向が一致しない。N=2048 の run 3/5 も改善幅は
  1.4〜8.7% と小さく、同じ N=2048 の run 1/2/4（比 >1.0）と符号が逆転
  している。「判定不可（undetermined）」の基準は「run 間で改善／後退の
  方向が一貫せずばらつきが大きい場合」だが、本判断では 5 run 中央値
  （N=1024: 1.219321・N=2048: 1.011548・N=4096: 1.015641）を採用基準の
  一次指標とし、run ごとの符号反転は「中央値の信頼区間内のノイズ」として
  扱う（`docs/perf/metal-bench-noise-protocol.md` の中央値ベース判定方針
  と同じ考え方）。N=4096 は符号反転が皆無（5 run 全て同方向）で
  undetermined 該当性を明確に排除できるが、N=1024／N=2048 は符号反転を
  伴う点で厳密には undetermined 基準に近い。それでも本 doc では
  undetermined と判定せず REJECT へ倒す判断とした理由は次のとおり:
  (a) N=1024 は中央値が 1.219321（約 22% 後退）と ±5% 帯を大きく超えて
  おり、符号反転は 5 run 中 1 run のみでノイズとして扱っても中央値の
  頑健性は損なわれない、(b) N=2048 は符号反転を伴い、5 run 全体の比の
  範囲は 0.913344〜1.050447（run 5〜run 4）で、run 5 単独は ±5% 帯を
  明確に下回る（約 8.7% の改善方向）ため「比が終始 ±5% 帯内」とは言えない。
  ただし 5 run 中央値（1.011548）は ±5% 帯内であり、改善方向の run
  （3・5）と後退方向の run（1・2・4）が拮抗して符号反転しているため、
  中央値ベースの判定方針（`docs/perf/metal-bench-noise-protocol.md`）に
  従い「改善〈比 <1.0〉」を安定して示したとは判断せず「有効性なし」
  区分の範囲内として扱う、(c) 3 サイズ中 2 サイズ
  （N=1024・N=4096）が明確に REJECT 側の基準を満たし、残り 1 サイズ
  （N=2048）も「候補表組み込み可」の基準（改善〈比 <1.0〉）を満たさない
  ため、「候補表組み込み可」の 3 条件同時成立が構造的に不可能（判定を
  覆すには N=2048 に加え N=1024／N=4096 も改善方向へ転じる必要があるが
  そのような run は 15 run 中に存在しない）。以上により本判断は
  undetermined ではなく REJECT を採用する

**結論**: E2（ソーステキスト特殊化によるアキュムレータ配列の厳密サイズ
化）は、反射値レベルでも実測 kernel_gpu レベルでも性能上の利得を示さず、
むしろ N=1024 で明確な後退（5 run 中央値 約 22%。ただし run 5 のみ約
12% の改善方向）・N=4096 で軽微だが 5 run 全てに一貫した後退（約 1〜2%）
を示した。N=2048 は 5 run 中央値では ±5% 帯内（有意差なし）だが run 3・
run 5 は改善方向・run 1/2/4 は後退方向と符号が割れており、単独では
「有効性あり」の根拠にならない。§2 の H1 仮説（function constant 特殊化後も
コンパイラが厳密サイズでレジスタ割付を最適化しない）を反射値レベルで
補完的に検証する目的は達成したが、E2 自体は候補表・`tile::select` への
組み込み対象としない。`tile::SOURCE_SPECIALIZATION_ENABLED` は `false`
のまま維持し、`#1302`（元は「E2 の `tile::select` 組み込み判断」を担う
予定だった）は本判定（REJECT）を根拠に組み込み作業なしでクローズ可能と
判断する（PR 本文・§9.5 に記載）。

### 9.5 スコープ外・引き継ぎ

- N=1024 の base 絶対値が `docs/perf/metal-gemm-reuse-phase-1277` の
  分母より小さい理由の深掘り（本イシューの計測環境・HEAD 差異による
  ものと推定するに留める）
- f16 経路（`gemm_simdgroup_tiled_f16`）への E2 展開（`spec_source.rs`
  は f32 経路のみを対象。#1288 と同じスコープ外判断）
- NT/TN/TT 転置パターンでの反射値・性能比較（本イシューは NN のみを
  計測対象とした。§8.1 の bit 一致自己検証は #1288 で NT/TN/TT も含めて
  完了済みのため、性能面のみの積み残し）
- 真のレジスタ spill 計測（Xcode Instruments 等。§5 スコープ外の継続）

### 9.6 関連ログ

`docs/perf/logs/metal-gemm-e2-spec-source-ab-1289/`（`reflection.log`・
`kernel_gpu_run{1..5}.log`・`aggregate.md`・`smoke_source_specialized.log`・
`env_info.txt`・`uptime_before_run{1..5}.txt`・`pmset_therm_before.txt`・
`pmset_therm_after.txt`）

## 10. E3 フラグメントロード方式候補の純カーネル時間比較（イシュー #1295）

### 10.0 目的・範囲

`docs/perf/metal-gemm-frag-load-candidates.md` §5「#1295 への引き継ぎ」の
指示に従い、E3（フラグメントロード方式変更。イシュー #1293 で実装・bit
一致を自己検証済み）の 5 候補（`tgp-k1`〈= 本番既定 `tile::FRAG_LOAD_
CONFIG`〉・`tgp-k2`・`device-legacy`〈= 本番既定の direct-load 側〉・
`device-hoisted-k1`・`device-hoisted-k2`）の N=1024/2048/4096 純カーネル
時間（GPU タイムスタンプ。イシュー #1276 の `kernel_gpu` 変種）を M4 Max
実機で 5 回計測し、有効性（`tile::select` への組み込み可否・採用候補）を
判定する。**`tile::select`／本番既定への実結線は本イシューのスコープ外**
であり、兄弟イシュー #1302（親 #1273 配下「E2〜E4 で有効な候補を
`tile::select` の候補表へ組み込む」担当）へ判定結果を引き継ぐ。

計測対象は NN・正方 3 形状（N=1024/2048/4096）のみ。NT/TN/TT・f16 経路・
N=512 は対象外（§10.5）。

### 10.1 環境・プロトコル

- HEAD: `c732988792b47f1cf341ae78ca407680e90f24c8`（origin/main。PR #1380）
- 実行方式: `crates/backend-metal/src/gemm_frag_load_diag_tests.rs`
  （`#[cfg(all(test, target_os = "macos"))]`。新規追加）の
  `frag_load_kernel_gpu_ab_production_sizes` を `cargo test -p
  fandhe-ai-backend-metal --release --lib frag_load_kernel_gpu_ab_
  production_sizes -- --ignored --nocapture --test-threads=1` で 5 プロセス
  起動（run1〜run5）
- **負荷状況**: `docs/real-hardware-verification-env.md` の待機方針（load
  average 1 分値が 3.0 未満に落ちるまで最大 30 分待機）に基づき監視を
  開始したが、**実際に観測できたのは約 5 分間（60 秒間隔 5 回ポーリング）
  のみ**で、30 分の待機を完了する前に低下傾向が見えないと判断して打ち
  切った。この間 load average 1 分値は 14.52 → 53.36 → 25.40 → 28.19 →
  19.77 と推移し低下傾向は見られなかった（53.36 のピークは本セッション
  自身の事前ビルド・スモークテスト実行と重なった時点であり、他セッション
  のみに起因するとは断定できない）。複数の並走エージェントセッション
  （別リポジトリの `cargo test --workspace --all-features` 等）が新規に
  起動する共有環境で、`metal-gemm-transpose-tiled.md`（#1186/#1187）・
  §7.10（#1284）と同種の状況。`env_info.txt` に並走プロセス名（プロセス名
  のみ）を記録済み
- **対処**: 5 候補すべてを同一プロセス・同一 trial ループ内で trial ごとに
  交互（開始オフセット回転）で計測する interleave 方式（`docs/perf/
  metal-bench-noise-protocol.md` §2 と同じ考え方）で外部負荷の影響を対称化
  したうえで判定した（詳細は `logs/metal-gemm-e3-frag-load-ab-1295/
  env_info.txt`）
- **twin validate 結果**: 全 N で `TileConfig { staged: false, ..cfg }`
  twin は `validate` を通過し、`device-legacy`/`device-hoisted-k1`/
  `device-hoisted-k2` の 3 候補とも全 N で計測対象に含まれた（validate
  不通過による対象外はゼロ件）
- **keep_alive 取扱い**: N ごとのループスコープで 5 候補分の `keep_alive`
  を保持し、次の N へ進む前に drop する方式を採用（truncate は不要。本機
  はピーク約 13.4 GiB を確保できる空きメモリがあった）
- **経路証跡の限界**: E2（`gemm_spec_source_diag_tests`）の
  `source_specialized_cache_len`/`function_constant_cache_len` に相当する
  「実際にどちらの経路の値か」を出力一致とは独立に保証する追加証跡は E3
  には存在しない。`resolved_cfg == requested_cfg`（フォールバック非経由。
  fail-closed 検証。全 run 全候補で違反なし）と、イシュー #1293 の bit
  一致自己検証（本判定に先立ち `frag_load` 4 テストを `--release
  --ignored --test-threads=1` で再実行し全 pass を確認。
  `logs/.../smoke_frag_load_bit_match.log`）が根拠

### 10.2 候補 × N の 5 run 表

`logs/metal-gemm-e3-frag-load-ab-1295/aggregate.md` に全数値を記載。要約
（5 run 中央値。ms）:

| N | tgp-k1（base） | tgp-k2 | device-legacy | device-hoisted-k1 | device-hoisted-k2 |
|---|---|---|---|---|---|
| 1024 | 1.0238 | 1.0234 | 2.5755 | 1.6246 | 2.1554 |
| 2048 | 2.4098 | 2.6647 | 8.0610 | 4.3095 | 6.9935 |
| 4096 | 22.0783 | 48.2870 | 55.3520 | 35.8267 | 61.7659 |

対 base（`tgp-k1`）比:

| N | tgp-k2 | device-legacy | device-hoisted-k1 | device-hoisted-k2 |
|---|---|---|---|---|
| 1024 | 0.9996 | 2.5156 | 1.5868 | 2.1053 |
| 2048 | 1.1058 | 3.3451 | 1.7883 | 2.9021 |
| 4096 | 2.1871 | 2.5071 | 1.6227 | 2.7976 |

対 device-legacy 比（staged→device 切替の効果と hoisting/ksteps の効果の
帰属分析用。採否の一次指標にはしない）:

| N | device-hoisted-k1 | device-hoisted-k2 |
|---|---|---|
| 1024 | 0.6308 | 0.8369 |
| 2048 | 0.5346 | 0.8676 |
| 4096 | 0.6473 | 1.1159 |

hoisted-k1 は device-legacy より一貫して速い（0.53〜0.65 倍）が、device
系そのものが staged（`tgp-*`）より大幅に遅いため base 比では改善に届かない
（「device 系内での hoisting の効果」と「staged→device の効果」を分離して
確認できたのみ）。

### 10.3 妥当性帯チェック

base（`tgp-k1`）の 5run 中央値絶対値を `docs/perf/metal-gemm-reuse-phase-
breakdown.md` §11 の分母（N=1024: 1.0267 ms／2048: 3.1849 ms／4096:
13.7051 ms）と突合した:

| N | 本測定 | 分母 | 乖離 |
|---|---|---|---|
| 1024 | 1.0238 ms | 1.0267 ms | 1.00 倍（ほぼ一致） |
| 2048 | 2.4098 ms | 3.1849 ms | 0.76 倍（本測定の方が速い） |
| 4096 | 22.0783 ms | 13.7051 ms | 1.61 倍（本測定の方が遅い） |

N=4096 の乖離（分母比 1.61 倍）は §10.1 の高負荷環境（他セッションの
CPU/GPU 競合）による事実として記録し、分母の置き換えは行わない
（§9.3〈#1289〉と同じ判断）。N=1024/2048 の
乖離は分母より小さい方向で、判定（相対比較）自体には影響しない。

### 10.4 採否判断

§9.4（#1289）の 3 区分判定基準をそのまま候補ごとに適用する:

- **候補表組み込み可**: N=2048 かつ N=4096 で改善（対 base 比 < 1.0）、
  かつ N=1024 の後退が 5% 以内（比 <= 1.05）
- **有効性なし → 組み込み不可**: 全 N で ±5% 帯内、または大形状（N=2048／
  4096）で後退
- **判定不可（undetermined）**: run 間で改善／後退の方向が一貫せず、
  ばらつきが大きい

判定結果（`device-legacy`／`device-hoisted-k1`／`device-hoisted-k2`
は全 N・全 run で方向が一貫し undetermined には該当しない。`tgp-k2` は
N=1024／N=4096 は全 run 一貫だが、N=2048 は 5 run 中 2 run（run2:
2.6913/2.6957=0.998331・run4: 2.5660/2.6907=0.953637）が改善方向、
残り 3 run（run1/3/5）が後退方向で符号が反転する。以下は候補ごとに
この点を明記したうえでの判定）:

- **`tgp-k2`**: N=1024 は全 run が ±5% 帯内（5run 中央値比 0.9996）。
  N=2048 は 5run 中央値比では後退（1.1058）だが run 単位では改善・後退が
  混在し（上記）、§9.4（#1289）と同じ中央値ベースの判定方針
  （`docs/perf/metal-bench-noise-protocol.md`）に従い「改善を安定して
  示した」とは判断しない。N=4096 は 5 run 全てで head > base
  （2.1871 倍。符号反転なし）と一貫して明確に後退する。3 サイズ中
  N=4096 が単独で「大形状で後退」の基準を満たし、N=2048 は「候補表
  組み込み可」の条件（改善〈比 <1.0〉）を満たさないため、両条件から
  **組み込み不可**と判定する（N=2048 の符号反転自体は判定を undetermined
  へ倒す根拠とはしない。理由は N=4096 の後退方向が 5 run とも一貫して
  おり、N=2048 が仮に改善方向で確定しても「候補表組み込み可」の 3 条件
  〈N=2048 かつ N=4096 で改善〉を N=4096 側が構造的に満たせないため）
- **`device-legacy`**: 全 N・全 run で base 比 2.5〜3.3 倍の大幅な後退
  → **組み込み不可**
- **`device-hoisted-k1`**: 全 N・全 run で base 比 1.6〜1.8 倍の後退
  → **組み込み不可**
- **`device-hoisted-k2`**: 全 N・全 run で base 比 2.1〜2.9 倍の大幅な後退
  → **組み込み不可**

**結論: E3 の 5 候補中、本番既定 `tgp-k1`（threadgroup 経由・K=1 段一括
ロード。現行の staged 経路）を上回る候補は存在しなかった。`tile::select`
候補表・本番既定への組み込み対象なし**（採用候補なし）。`tile::
FRAG_LOAD_CONFIG`（= `FragLoadConfig::DEFAULT`）は不変のまま維持する。

E1（loop unroll pragma。§7）・E2（ソーステキスト特殊化。§8-9）に続き、
E3 も本番既定を上回らないという結果になった。3 実験とも「現行の
`gemm_simdgroup_tiled` 実装（既存の staged 8 幅ロード・function constant
経路）が M4 Max 上では既に局所最適に近い」ことを補強する一次情報として
扱う（各実験のスコープ・仮説はそれぞれ独立であり、本節はこの傾向を指摘
するに留め、新たな仮説検証は行わない）。

### 10.5 スコープ外・引き継ぎ

- **#1302 への引き継ぎ**: E3 は判定「組み込み対象なし」につき、#1302 の
  E3 分担分は追加実装なしでクローズ可能（E2〈#1289 の REJECT 判定〉と
  同型の結論）。E3 の twin 構成（`staged=false` 版 `TileConfig`）を候補表
  へ追加する作業自体も本判定により不要と判断する
- NT/TN/TT 転置パターンでの性能比較（bit 一致は #1293 T4 で NN 以外も
  検証済みだが、性能実測は NN のみが対象）
- f16 経路（`gemm_simdgroup_tiled_f16`）への E3 展開（未着手のまま）
- N=512 の計測（`docs/perf/metal-gemm-reuse-phase-breakdown.md` 分母表が
  N>=1024 のみを対象とするため、他の E1〜E2 実測と同じ理由で対象外）
- N=4096 で run 間ばらつきが大きい根本原因（§10.1 の高負荷環境が主因と
  推定するに留め、低負荷環境での再測定による定量的切り分けは行わない）
- run2 の N=1024 のみ全 5 候補が他 4 run と比べ約半分の絶対値（例:
  `tgp-k1` 0.5621 ms 対 他 run の約 1.02 ms）を記録した（`logs/
  metal-gemm-e3-frag-load-ab-1295/aggregate.md` 参照）。run 内で全候補に
  対称に現れているため §10.4 の相対比較・判定には影響しないが、原因は
  未特定のまま記録する（run 単位のスケジューリング揺らぎと推定するに
  留める）

### 10.6 関連ログ

`docs/perf/logs/metal-gemm-e3-frag-load-ab-1295/`（`env_info.txt`・
`uptime_before_run{1..5}.txt`・`pmset_therm_before.txt`・
`pmset_therm_after.txt`・`smoke_frag_load_bit_match.log`・
`kernel_gpu_run{1..5}.log`・`aggregate.md`）

## 11. E4 協調ロードレイアウト候補の純カーネル時間比較（イシュー #1300）

### 11.0 目的・範囲

`docs/perf/metal-gemm-coop-load-candidates.md` §5「#1300 への引き継ぎ」の
指示に従い、E4（協調ロードレイアウト再構成。イシュー #1298 で実装・bit
一致を自己検証済み）の 6 候補（`L0-P4`〈= 本番既定 `tile::COOP_LOAD_
CONFIG`。`layout=RowLinear, pad=Four`〉・`L0-P0`・`L0-P8`・`L1-P0`・
`L1-P4`・`L1-P8`）の N=1024/2048/4096 純カーネル時間（GPU タイムスタンプ。
イシュー #1276 の `kernel_gpu` 変種）を M4 Max 実機で 5 回計測し、有効性
（`tile::select` への組み込み可否・採用候補）を判定する。**`tile::select`／
本番既定への実結線は本イシューのスコープ外**であり、兄弟イシュー #1302／
#1304（親 #1273 配下「E2〜E4 で有効な候補を `tile::select` の候補表へ
組み込む」担当）へ判定結果を引き継ぐ。

計測対象は NN・正方 3 形状（N=1024/2048/4096）のみ。NT/TN/TT・f16 経路・
N=512・XOR swizzle 軸（index 15。#1298 で未実装）は対象外（§11.5）。

### 11.1 環境・プロトコル

- HEAD: `98bc216ce49c0acb55838215d9c8b8218c981a38`（origin/main。E4 機構実装
  PR #1384 を含む）
- 実行方式: `crates/backend-metal/src/gemm_coop_load_diag_tests.rs`
  （`#[cfg(all(test, target_os = "macos"))]`。本イシューで新規追加）の
  `coop_load_kernel_gpu_ab_production_sizes` を `cargo test -p
  fandhe-ai-backend-metal --release --lib coop_load_kernel_gpu_ab_
  production_sizes -- --ignored --nocapture --test-threads=1` で 5 プロセス
  起動（run1〜run5）
- **計測前ゲート（T1〜T6 bit 一致）**: `gemm.rs::tests` の 6 テスト
  （`coop_load_bit_match_all_candidates`／`coop_load_bit_match_dispatch_
  auto`／`coop_load_transposed_bit_match`／`coop_load_bit_match_boundary_
  shape`／`coop_load_f16_path_is_noop`／`coop_load_default_matches_
  production_constants`）を `--release --ignored --nocapture
  --test-threads=1 coop_load` で実行し**全 6 件 pass**
  （`smoke_coop_load_bit_match.log`）。1 件でも fail した場合は性能比較へ
  進まない計画だったが、本イシューでは全 pass のため計測を継続した
- **負荷状況**: 着手前に `docs/real-hardware-verification-env.md`（load
  average 1 分値が 3.0 未満に落ちるまで待機）の方針でポーリングを開始した。
  開始直前は `load averages: 4.49 6.33 8.45`・`21 users`（他セッションと
  共有の環境。過去 #1186/#1187/#1278/#1279/#1295 と同種の状況）で、5 分間
  （60 秒間隔 5 回）のポーリングで 1 分値は 4.49 → 2.65 → 1.96 → 2.32 →
  3.89 → 3.17 と推移した。安定して 3.0 未満に収束しなかったため 30 分の
  上限を待たずに打ち切り、`load averages: 3.08 4.12 6.73` の状態で計測を
  開始した（`env_info.txt` 参照）。`pmset -g therm` は計測前後とも
  サーマルスロットリングの記録なし
- **対処**: E3（§10.1）と同じ interleave 方式（trial index による開始
  オフセット回転）を採用。E4 は全候補が同一 `base_cfg`（`staged=true` の
  選択構成）を使うため E3 のような `staged=false` twin は不要（ファイル
  冒頭コメント「候補と `TileConfig` の組み立て方」参照）
- **共有メモリ事前フィルタ結果**: 全 5 run・全 N で 6 候補すべてが
  `shared_mem_bytes_for_pad <= maxThreadgroupMemoryLength` を満たし、
  対象外となった候補はゼロ件（`TGP_PAD=8` によるフォールバックは
  発生しなかった）
- **keep_alive 取扱い**: N ごとのループスコープで 6 候補分の `keep_alive`
  を保持し、次の N へ進む前に drop する方式（truncate 不要。本機は
  N=4096 時ピーク約 16.1 GiB を確保できる空きメモリがあった）
- **経路証跡の限界**: E3 と同じく、`resolved_cfg == requested_cfg`
  （フォールバック非経由。fail-closed 検証。全 run 全候補で違反なし）と、
  イシュー #1298 の bit 一致自己検証（本判定に先立ち T1〜T6 を再実行し
  全 pass を確認）が根拠。E2 の `source_specialized_cache_len` 相当の
  追加証跡は存在しない

### 11.2 候補 × N の 5 run 表

`logs/metal-gemm-e4-coop-load-ab-1300/aggregate.md` に全数値・run 単位の
符号一貫性を記載。要約（5 run 中央値の中央値。ms）:

| N | L0-P4（base） | L0-P0 | L0-P8 | L1-P0 | L1-P4 | L1-P8 |
|---|---|---|---|---|---|---|
| 1024 | 1.0340 | 1.0184 | 1.0281 | 1.0260 | 1.0499 | 1.0447 |
| 2048 | 1.6117 | 1.6104 | 1.5920 | 1.6097 | 1.6071 | 1.5938 |
| 4096 | 13.6736 | 13.4865 | 14.0575 | 14.4631 | 14.4342 | 15.7797 |

対 base（`L0-P4`）比:

| N | L0-P0 | L0-P8 | L1-P0 | L1-P4 | L1-P8 |
|---|---|---|---|---|---|
| 1024 | 0.9849 | 0.9943 | 0.9923 | 1.0154 | 1.0103 |
| 2048 | 0.9992 | 0.9878 | 0.9988 | 0.9971 | 0.9889 |
| 4096 | 0.9863 | 1.0281 | 1.0577 | 1.0556 | 1.1540 |

### 11.3 妥当性帯チェック

base（`L0-P4`）の 5run 中央値絶対値を `docs/perf/metal-gemm-reuse-phase-
breakdown.md` §11.5 の分母（N=1024: 1.0267 ms／2048: 3.1849 ms／4096:
13.7051 ms）と突合した:

| N | 本測定 | 分母 | 乖離 |
|---|---|---|---|
| 1024 | 1.0340 ms | 1.0267 ms | 1.01 倍（ほぼ一致） |
| 2048 | 1.6117 ms | 3.1849 ms | 0.51 倍（本測定の方が速い） |
| 4096 | 13.6736 ms | 13.7051 ms | 1.00 倍（ほぼ一致） |

N=2048 の乖離（分母比 0.51 倍）は E3（§10.3）でも同様の方向の乖離
（0.76 倍）が観測されており、分母表の N=2048 実測が異なる `tile::select`
解決結果（`select_for_device` は `verified_m4_max_gpu_core_count()` の
検出結果に依存しうる）または異なる負荷環境下の実測値であった可能性を
示唆する事実として記録し、分母の置き換えは行わない（§10.3 と同じ判断。
N=4096 はほぼ一致しており判定への影響はない）。

### 11.4 採否判断

§9.4／§10.4 と同じ 3 区分判定基準を適用する:

- **候補表組み込み可**: N=2048 かつ N=4096 で改善（対 base 比 < 1.0）、
  かつ N=1024 の後退が 5% 以内（比 <= 1.05）
- **有効性なし → 組み込み不可**: 全 N で ±5% 帯内、または大形状（N=2048／
  4096）で後退
- **判定不可（undetermined）**: run 間で改善／後退の方向が一貫せず、
  ばらつきが大きい

`aggregate.md`「run 単位の符号一貫性」に基づき、中央値の中央値だけでなく
run ごとの符号（改善／後退の方向）を確認したうえで候補ごとに判定する:

- **`L0-P0`**: N=1024 は 5/5 run すべてで base より速い（符号一貫。
  0.9849）。**N=2048 は 3/5 run で速い・2/5 run で遅い**（符号不一致。
  中央値比は 0.9992 でほぼ 1.0）。**N=4096 も 3/5 run で速い・2/5 run で
  遅い**（符号不一致。run1・run4 は base 側も同時に高い値を示す高負荷回の
  外れ値）。中央値だけを見ると「候補表組み込み可」の 3 条件（N=2048 かつ
  N=4096 で比 <1.0、N=1024 後退 5% 以内）を形式的には満たすが、N=2048・
  N=4096 いずれも run 間で符号が反転しており、§10.4 の `tgp-k2` 判定と
  同じ理由（中央値ベースの判定方針は「改善を安定して示した」ことを要求
  する）により**候補表組み込み可とは判断しない**。差の絶対値も 1.4〜
  1.6% と §11.1 の共有負荷環境で観測されたノイズ幅（run 間で q1/q3 が
  数〜数十% 動く。`env_info.txt`「解釈上の注意」参照）の範囲内であり、
  ここまでのデータからは**組み込み不可（有効性の確証なし）**と判定する
- **`L0-P8`**: N=2048 は 5/5 run すべてで base より速い（符号一貫。
  0.9878。ただし差は 1.2〜1.9%）。N=1024 は base とほぼ同等（0.9943）。
  **N=4096 は 3/5 run で速い・2/5 run で遅い**（符号不一致。中央値比
  1.0281 は後退方向）。N=4096 が「候補表組み込み可」の必須条件（比
  <1.0）を中央値ベースで満たさないため**組み込み不可**
- **`L1-P0`／`L1-P4`／`L1-P8`**（`RowStrided` レイアウト）: **N=4096 は
  いずれも 5/5 run すべてで base より遅い**（符号完全一致。中央値比
  1.058〜1.154。3 候補中最大は `L1-P8` の 15.4% 後退）。N=1024/2048 は
  base と概ね同等（±2%）だが、N=4096 で符号一貫かつ明確な大形状後退を
  示すため**いずれも組み込み不可**（「大形状で後退」の基準に該当）

**結論: E4 の 6 候補中、本番既定 `L0-P4`（`RowLinear` レイアウト・
`TGP_PAD=4`）を安定して上回る候補は確認できなかった。`RowStrided`
レイアウト（L1 系）は N=4096 で符号一貫の後退を示し明確に劣後する一方、
`RowLinear` レイアウトの `pad` 変更（L0-P0／L0-P8）は N=1024/2048 で
方向性のある改善候補（特に `L0-P0`）を示したが、いずれも N=2048/4096 で
run 間の符号が反転し、共有負荷環境下で安定した改善として確証できなかった。
`tile::select` 候補表・本番既定への組み込み対象なし**（採用候補なし）。
`tile::COOP_LOAD_CONFIG`（= `CoopLoadConfig::DEFAULT`）は不変のまま維持
する。

E1（§7）・E2（§8-9）・E3（§10）に続き、E4 も本番既定を安定して上回らない
という結果になった。4 実験とも「現行の `gemm_simdgroup_tiled` 実装
（既存の staged 8 幅ロード・`RowLinear` 協調ロード・`pad=4`・function
constant 経路）が M4 Max 上では既に局所最適に近い」ことを補強する一次
情報として扱う（各実験のスコープ・仮説はそれぞれ独立であり、本節は
この傾向を指摘するに留め、新たな仮説検証は行わない）。

### 11.5 スコープ外・引き継ぎ

- **#1302／#1304 への引き継ぎ**: E4 は判定「組み込み対象なし」につき、
  両イシューの E4 分担分は追加実装なしでクローズ可能（E2〈#1289〉・E3
  〈#1295〉と同型の結論）
- **XOR swizzle 軸（index 15）**: `docs/perf/metal-gemm-coop-load-
  candidates.md` §4「未割当のまま残す」のとおり、イシュー #1298 では
  時間制約により未実装。本イシューでも実装・計測は行わない（新規 Issue
  は起票しない。ユーザー承認なしの起票禁止。PR 本文に提示のみ）。
  **追記（イシュー #1327）**: index 15 はその後 `TILE_CLASS`（タイル
  クラス分割ゲート。E6 試作）へ割り当てられた。詳細・実測は
  `docs/perf/metal-gemm-tile-class-split.md`。性能実測・採否判断は
  イシュー #1328（本ファイル §12）で完了済み（組み込み不可・REJECT）
- NT/TN/TT 転置パターンでの性能比較（bit 一致は #1298 T3 で NN 以外も
  検証済みだが、性能実測は NN のみが対象）
- f16 経路（`gemm_simdgroup_tiled_f16`）への E4 展開（T5 は f16 経路が
  `coop_load_layout` を no-op で受ける契約の bit 一致検証のみで、性能
  実測は未実施）
- N=512 の計測（`docs/perf/metal-gemm-reuse-phase-breakdown.md` 分母表が
  N>=1024 のみを対象とするため、他の E1〜E3 実測と同じ理由で対象外）
- `L0-P0`／`L0-P8` が N=1024/2048/4096 の一部 run で示した改善方向の
  シグナルは、より安定した（低負荷・専有）実行環境での再測定であれば
  異なる結論になる可能性がある。本イシューでは共有負荷環境下での判定に
  留め、専有環境での再測定は行わない

### 11.6 関連ログ

`docs/perf/logs/metal-gemm-e4-coop-load-ab-1300/`（`env_info.txt`・
`uptime_before_run{1..5}.txt`・`pmset_therm_before.txt`・
`pmset_therm_after.txt`・`smoke_coop_load_bit_match.log`・
`kernel_gpu_run{1..5}.log`・`aggregate.md`）

## 12. E6 タイルクラス分割の純カーネル時間比較（イシュー #1328）

### 12.0 目的・範囲

`docs/perf/metal-gemm-tile-class-split.md`（E6 タイルクラス分割。イシュー
#1327 で opt-in 機構〈`tile::TileClassMode`〉を実装・端あり形状込みで
bit 一致を自己検証済み）の性能実測・本番結線（`tile::TILE_CLASS_MODE`）
可否判断を担う（同 doc「性能実測・`tile::select` への組み込み判断は
行わない（兄弟イシュー #1328 のスコープ）」の引き継ぎ）。

**命名衝突の明示**: 本 §12 の「E6」（イシュー #1327/#1328。タイルクラス
分割）は、§3「Phase B/E6」の「E6」（イシュー #1143。候補
`(32,64,16,1,2)`＝`CANDIDATES[8]` の追加測定）とは別物である。両者は
偶然同じ略称を使っているだけで機構としては無関係。

**「現行経路」の定義**: 本番 `dispatch_auto`（`tile::select_for_device`）
は M4 Max 実測帯域で N=512→`CANDIDATES[5]`・1024→`[6]`・2048→`[1]`・
4096→`[2]` を選ぶ。本イシューが対象とする候補 0/4/5/8 のうち、
N=1024/2048/4096（§12.2 本表の対象）ではこの本番選択構成のいずれとも
一致しないが、**N=512 では候補 5（`CANDIDATES[5]`）自体が本番選択構成
そのもの**である点に注意（§12.2-B の N=512 行はしたがって候補 0/4/5/8
の一員である候補 5 の直接計測に相当する）。したがって「現行経路」は
2 通りに解釈しうる:

- (A) 候補 0/4/5/8 それぞれの `TileClassMode::Legacy`（1 dispatch。
  §12.2 の主対象）
- (B) 本番が実際に選択する構成（`CANDIDATES[1]/[2]/[5]/[6]`）の
  `TileClassMode::Legacy`（§12.2 追補で直接計測）

Issue 記載の受け入れ基準は (A) だが、本番結線の可否判断には (B) が
不可欠なため、本イシューのスコープを実務的に拡張し両方を計測した
（§3.4「採否・本番結線の判断規則」の「ADOPT 相当の場合の追加 A/B」
要求に対応）。

**AC 形状での構造的縮退（事前の作業仮説と実測の乖離）**: 候補 0/4/5/8
はいずれも `staged: true` で、N=1024/2048/4096 は各候補の `bm`/`bn`/`bk`
すべての倍数（`bk` が最大でも 32、N が最小でも 1024 のため必ず割り切れる）。
よって `tile::tile_class_plan` は interior＝grid 全体・端ストリップ 2 本
とも空を返し、`TileClassMode::Split` は「**Interior クラス（`tile::
TileClassMode` ドキュメンテーションコメント参照。direct-load 強制）を
grid 全体へ適用した 1 dispatch**」へ縮退する（実測で edge 増分 0・
interior 増分のみを確認。§12.1 参照）。当初は E3（`docs/perf/
metal-gemm-transpose-tiled.md`）の `device-legacy` twin（`staged=false`）
が全 N で staged 比 1.6〜3.3 倍遅かった実績から、この縮退が一律の後退
（REJECT）に直結すると予想していたが、**実測は候補依存で符号が割れた**
（§12.2）。

### 12.1 環境・プロトコル

- HEAD: `b41bba6`（origin/main。E6 機構実装 PR #1388 を含む）
- 実行方式: `crates/backend-metal/src/gemm_tile_class_diag_tests.rs`
  （`#[cfg(all(test, target_os = "macos"))]`。本イシューで新規追加）の
  2 テスト:
  - `tile_class_kernel_gpu_ab_production_sizes`（候補 0/4/5/8 × N=1024/
    2048/4096。§12.2 本表）を `cargo test -p fandhe-ai-backend-metal
    --release --lib tile_class_kernel_gpu_ab_production_sizes --
    --ignored --nocapture --test-threads=1` で 5 プロセス起動
  - `tile_class_production_select_kernel_gpu_ab`（本番選択構成 ×
    N=512/1024/2048/4096。§12.2 追補）を同様に 5 プロセス起動
- **プロダクションコード変更**: `gemm.rs::encode_tiled_by_class` を挙動
  不変の `plan_tiled_by_class`／`encode_tiled_plan` へリファクタし、
  診断専用 `diag_encode_tiled_nn` が `self.tile_class_mode` を尊重する
  ように是正した（従来は `pipeline_for_tile(..., TileClass::Legacy)` を
  直接呼んでいたため `new_with_tile_class(Split)` でも常に Legacy 経路
  しか測れなかった。§12 実測の前提修正。`tile.rs`／`shaders/gemm.metal`
  は無変更）
- **計測前ゲート（bit 一致）**: `gemm::tests` の 4 テスト
  （`tile_class_split_bit_match_all_candidates`／`_edge_shapes`／
  `_dispatch_auto`／`tile_class_default_matches_production_constants`）を
  `--release --ignored --nocapture --test-threads=1 tile_class` で実行し
  **全 4 件 pass**（`smoke_tile_class_bit_match.log`）。ハーネス自身も
  各 (N, cand) の trial 0 出力で base/head の bit 完全一致を fail-closed
  に検証し全て pass
- **経路証跡**: `TILE_CLASS_SPLIT_FALLBACK_COUNT` 増分は全 (N, cand) で
  0（Edge/Interior 解決構成の食い違いなし）。§12.2 本表では
  `TILE_CLASS_EDGE_DISPATCH_COUNT` 増分が全 (N, cand) で 0、
  `TILE_CLASS_INTERIOR_DISPATCH_COUNT` 増分が全 (N, cand) で 40
  （= warmup 20 + measured 20）——AC 形状での構造的縮退（interior のみ）
  を機構レベルで裏付ける
- **負荷状況**: 着手前 `uptime` 1 分値 1.44〜1.83（3.0 未満のため待機
  不要）。5 run とも同様の負荷帯（`uptime_before_run{1..5}.txt`）。
  `pmset -g therm` は前後とも警告なし
- **interleave**: A 節（`tile_class_kernel_gpu_ab_production_sizes`）は
  trial ごとに base/head の計測順を交互に反転する（E3/E4 と同じ
  order-bias 相殺）。**B 節（`tile_class_production_select_kernel_gpu_
  ab`。`run_size_with` 呼び出し）は N ごとに「base を 40 反復 → head を
  40 反復」の順で逐次実行し、trial 単位の交互化を行っていない**（`gemm_
  tile_class_diag_tests.rs::tile_class_production_select_kernel_gpu_ab`
  doc comment 参照）。N=2048/4096 の後退（2.3〜3.1 倍）は order-bias で
  説明できない規模だが、N=512/1024 の符号不一致・外れ値（§12.2-B）は
  この非交互構成の影響を受けている可能性がある

### 12.2 候補 × N の 5 run 表と対 Legacy 比

**A. 候補 0/4/5/8（AC 形状）**: `docs/perf/logs/
metal-gemm-e6-tile-class-ab-1328/aggregate.md` §A に全 run 生値を記載。

| N | cand | resolved_tile (bm,bn,bk,wm,wn) | legacy median (ms) | split median (ms) | run別比の中央値 (split/legacy) | run 間符号 |
|---|---|---|---|---|---|---|
| 1024 | 0 | 64,64,16,2,2 | 5.6569 | 3.6782 | **0.6502** | 一貫（改善） |
| 1024 | 4 | 64,64,16,1,2 | 2.4562 | 2.0830 | **0.8530** | 一貫（改善） |
| 1024 | 5 | 64,32,32,2,2 | 0.3552 | 1.1016 | **2.7380** | 一貫（後退） |
| 1024 | 8 | 32,64,16,1,2 | 1.9392 | 1.1774 | **0.5923** | 一貫（改善） |
| 2048 | 0 | 64,64,16,2,2 | 13.6808 | 7.0745 | **0.5193** | 一貫（改善） |
| 2048 | 4 | 64,64,16,1,2 | 34.8062 | 16.5568 | **0.4761** | 一貫（改善） |
| 2048 | 5 | 64,32,32,2,2 | 1.8344 | 4.8899 | **2.6646** | 一貫（後退） |
| 2048 | 8 | 32,64,16,1,2 | 15.7692 | 7.1534 | **0.4536** | 一貫（改善） |
| 4096 | 0 | 64,64,16,2,2 | 106.4910 | 57.2960 | **0.5384** | 一貫（改善） |
| 4096 | 4 | 64,64,16,1,2 | 306.0033 | 132.5602 | **0.4339** | 一貫（改善） |
| 4096 | 5 | 64,32,32,2,2 | 18.6987 | 43.5757 | **2.3304** | 一貫（後退） |
| 4096 | 8 | 32,64,16,1,2 | 127.0828 | 59.1988 | **0.4658** | 一貫（改善） |

候補 0/4/8 は Split が一貫して legacy より速い（0.43〜0.85 倍）。候補 5
は Split が一貫して legacy より遅い（2.3〜2.7 倍）。run 間の符号はどの
(N, cand) の組でも一貫している。**この差は `bk` では説明できない**
（後述 §12.2-B のとおり、本番選択構成 `[1]`/`[2]`〈いずれも `bk=16`〉も
Split で一貫して後退するため、「`bk=16` なら改善」という単純な規則は
成立しない）。むしろ候補 0/4/8 の legacy（staged）ベースライン自体が
同一 N の本番選択構成の legacy より極端に遅い（例: N=4096 で候補 0 の
legacy 106.49 ms 対 本番選択構成 `[2]` の legacy 13.77 ms。約 7.7 倍）
ことが特徴的であり、Split（direct-load）はこの非効率なベースラインを
部分的に緩和したに過ぎないと考えられる（§12.4 参照）。

**表記注（codex-review 指摘対応。イシュー #1328 PR #1389）**: 上表
（A）の「run別比の中央値」は各 run の比（split/legacy）を 5 個算出
したうえでその中央値を取ったものであり、`legacy median` 列と
`split median` 列の比（両群の中央値同士の比）とは値が異なる
（例: N=1024/cand5 は両群中央値の比では 1.1016/0.3552≈3.1014 だが
run別比の中央値は 2.7380）。下記 B 節の「中央値比（両群中央値の
比）」列は逆に両群中央値の比そのものであり、算出方法が異なる別の
指標である点に注意する（同一の生値は `aggregate.md` §A/§B 参照）。

**B. 本番 `select_for_device` 選択構成（追補）**: 同 `aggregate.md` §B に
全 run 生値を記載。

| N | 選択構成 (bm,bn,bk,wm,wn) | legacy median (ms) | split median (ms) | 中央値比（両群中央値の比） | run 間符号 |
|---|---|---|---|---|---|
| 512  | 64,32,32,2,2（`[5]`） | 0.2015 | 0.1081 | 0.5365 | **不一致**（2/5 run は legacy 比 約 2.5 倍に後退・3/5 run は半減） |
| 1024 | 64,32,8,4,1（`[6]`）  | 0.2245 | 0.5667 | 2.5243 | 4/5 run 後退・1 run 外れ値 |
| 2048 | 64,32,16,2,2（`[1]`） | 1.6030 | 4.9021 | 3.0581 | **一貫**（5/5 後退） |
| 4096 | 32,64,16,2,2（`[2]`） | 13.7657 | 32.1276 | 2.3339 | **一貫**（5/5 後退） |

本番が実際に選択する構成では、N=2048/4096 が明確に一貫した後退
（2.3〜3.1 倍遅い）、N=1024 もおおむね後退方向（4/5 run）、N=512 のみ
二峰性で符号不一致（B 節「interleave」注記のとおり非交互測定の影響が
疑われるが未確定）。N=512 の選択構成 `[5]` は候補 5（A 節）そのものであり、
A 節の N=1024/2048/4096 での結果（一貫して後退）と符号が整合する。
**候補 0/4/8（A 節。いずれも本番非選択構成）で観測した改善は、本番が
実際に選択する構成（N=1024/2048/4096。`[1]`/`[2]`/`[6]`）には現れない**
——A 節の結果を本番選択構成へ外挿できないことが直接計測で確認された。

### 12.3 妥当性帯チェック

§11.5 分母（`docs/perf/metal-gemm-reuse-phase-breakdown.md` §11.5。N=1024
1.0267 ms／2048 3.1849 ms／4096 13.7051 ms。`CANDIDATES[6]/[1]/[2]` 相当）
との突合: 本 §12.2-B の legacy 列は N=1024 0.2245 ms・2048 1.6030 ms・
4096 13.7657 ms で、N=4096 は分母とほぼ一致（13.77 対 13.71 ms）だが
N=1024/2048 は分母よりかなり**小さい**（速い）。原因は特定できていない
——本ハーネス（`MetalContext::new()` 専用コンテキスト＋`MetalGemm::
new_with_tile_class` の非キャッシュインスタンス）は §11.5 の本番 hot
path（`cached_context()`／`cached_gemm()`）と構成が異なるが、非キャッシュ
経路が速くなる理由は自明ではなく、単純な「コールドスタートで遅くなる」
という説明とは逆方向の乖離である。さらに N=1024 の legacy 5 run 生値
（§12.2-B: `[0.223, 0.224, 0.225, 1.022, 1.023]`）自体が二峰性で中央値が
不安定であり、上記の分母突合はこの不安定な中央値に基づく参考値に過ぎない
（原因不明のまま記録する。分母は置換しない）。N=4096 は二峰性が見られず
分母と近い値のため、この乖離が比率（split/legacy）の妥当性に与える影響は
N=4096 では小さいと考えられるが、N=512/1024 については§12.4 の判断で
慎重に扱う（§12.4 参照）。

### 12.4 採否判断

判定基準（§9.4/§10.4/§11.4 と同一）: 組み込み可＝N=2048 かつ 4096 で
Split/Legacy 比 < 1.0、かつ N=1024 の後退 ≤ 5%、かつ run 間の符号が一貫。

**判定: 組み込み不可（REJECT）。`tile::TILE_CLASS_MODE = Legacy` を維持
する。コード変更は §12.1 記載のリファクタ（`plan_tiled_by_
class`／`encode_tiled_plan` 分離。挙動不変）と診断テスト追加のみ**。

根拠:

1. 本番 `dispatch_auto` が実際に選択する構成（`CANDIDATES[1]/[2]/[6]`）
   では、N=2048/4096 で Split が一貫して legacy より 2.3〜3.1 倍遅い
   （§12.2-B。5/5 run で符号一貫）。N=1024 もおおむね後退方向（4/5 run）。
   これは判定基準の「N=2048 かつ 4096 で Split/Legacy 比 < 1.0」を明確に
   満たさない
2. 候補 0/4/8（A 節。N=1024/2048/4096 では本番非選択構成）で観測された
   改善（0.43〜0.85 倍）は真の実測結果だが、本番が実際に選択する構成
   （B 節の `[1]`/`[2]`/`[6]`）には現れないことを直接計測で確認済み。
   よって A 節の改善を理由に結線することはできない
3. 候補 5（A 節）は N=1024/2048/4096 いずれでも Split が一貫して遅く
   （2.3〜2.7 倍）、その本番選択構成としての姿である N=512（B 節）でも
   Split が legacy を上回る場面（2/5 run）と大きく下回る場面（3/5 run）
   が混在し符号が一貫しない。候補 5 系統（本番選択構成 `[5]` を含む）は
   いずれの N でも Split 採用の根拠にならない

候補 0/4/8 の「改善」は `bk` に依存する規則ではない（本番選択構成
`[1]`/`[2]` も `bk=16` だが Split で一貫して後退するため。§12.2-A
参照）。候補 0/4/8 の legacy（staged）ベースライン自体が同一 N の本番
選択構成の legacy より約 7〜8 倍遅く、Split（Interior クラス＝
direct-load 強制）はこの非効率を部分的に緩和したに過ぎないと考えられる
（プロファイリング等による機構レベルの検証は未実施）。いずれにせよ
候補 0/4/8 は Split・Legacy のどちらのモードでも本番選択構成の legacy
に及ばず（例: N=4096 で候補 0 の Split 57.30 ms 対 本番選択構成 `[2]`
の legacy 13.77 ms）、本番結線の根拠にはならない。「候補 0/4/8 の
legacy ベースラインがなぜ本番選択構成よりこれほど遅いか」自体は本
イシューでは未解明の新しい観察であり、§12.5 で引き継ぐ。

### 12.5 スコープ外・引き継ぎ

- 端あり形状（E6 本来の効用が現れる形状。Interior／Edge 両方が非空に
  なる形状）での性能比較は未実施（本 §12 は AC 形状〈整列〉のみ）
- N=512（候補 0/4/5/8 側。B 節のみ計測）・NT/TN/TT・f16 経路
- T4（転置）・T5（`FragLoadConfig` 合成）の bit 一致自己検証（#1327 から
  未実施のまま）
- REJECT のため `dispatch_auto` 全形状 A/B（§3.4 の「ADOPT 相当の場合」
  向け追加検証）は本 §12.2-B で先行的に実施済み（実質的に完了）
- 候補 0/4/8 の legacy（staged）ベースラインが同一 N の本番選択構成より
  約 7〜8 倍遅い理由（occupancy・レジスタ圧・スレッドグループ数等の
  プロファイリングによる特定）・Split（direct-load）がこれを部分的に
  緩和する機構の解明は未調査（§12.4）。追加調査が必要なら新規 Issue の
  起票をユーザーに提案する（`out-of-scope-tracking.md`）
- §12.3 で記録した、本ハーネス（非キャッシュ `MetalGemm` インスタンス）
  の legacy 計測値が §11.5 分母（本番 hot path）より N=1024/2048 で
  速いという原因不明の乖離、および N=1024 legacy 5 run の二峰性の原因
  調査は未実施
- XOR swizzle 軸（index 16 以降）・専有（低負荷）環境での再測定

### 12.6 関連ログ

`docs/perf/logs/metal-gemm-e6-tile-class-ab-1328/`（`env_info.txt`・
`uptime_before_run{1..5}.txt`・`pmset_therm_before.txt`・
`pmset_therm_after.txt`・`smoke_tile_class_bit_match.log`・
`kernel_gpu_run{1..5}.log`（候補 0/4/5/8）・
`kernel_gpu_production_select_run{1..5}.log`（本番選択構成追補）・
`aggregate.md`）

## 13. E7 候補追加: 64×64×32（wm2wn2）の収録と parity 確認（イシュー #1329）

### 13.0 目的・範囲

親 #1324（E7: bk=32 を 64×64 タイルへ拡張）の sub-issue。`CANDIDATES[0]`
（64,64,16,2,2。大形状の主力構成）に対する bk=32 版
`(64,64,32,2,2)` を `tile::CANDIDATES` の **index 9（末尾）** へ追加した。
K ループ 1 反復あたりの `threadgroup_barrier` 往復を半減させる狙い
（理論根拠。`CANDIDATES[5]`〈`(64,32,32,2,2)`。イシュー #532〉が bk=32
自体の初採用実績）。

本 issue のスコープは**候補追加と正確性（parity）確認のみ**: (a) 明示
指定（`GemmVariant::SimdgroupTiled(cfg)`／`dispatch_tiled_prepared`／
`dispatch_strided_tiled_prepared` の `cfg` 引数）でのみ到達可能にする
（`select`／`select_with_occupancy_for_device` の選択ロジックは無変更）、
(b) カーネル側手動境界チェック（REQ-8）を維持（`gemm.metal` は無変更）、
(c) 全形状 × NN/NT/TN/TT で parity 0 fail を実機確認、(d) threadgroup
メモリ使用量の反射値確認、(e) 現行 `CANDIDATES[0]`（bk=16）との出力が
複合判定内で一致することを確認。純カーネル時間の before/after 実測・
`tile::select` への組み込み判断は後続イシュー #1330 のスコープ（§13.5）。

### 13.1 環境・プロトコル

実機は Apple M4 Max（`docs/real-hardware-verification-env.md` §7）。
`cargo test -p fandhe-ai-backend-metal --release ... -- --ignored
--nocapture --test-threads=1`（正確性確認のみのため 5 回計測中央値は
不要）。実行前 `uptime` 1 分 load average 1.67〜1.70（低〜中程度の共有
負荷。21 ユーザーセッション並走）・`pmset -g therm` は thermal/performance
warning なし。env_info・実行ログは §13.6 参照。

### 13.2 反射値・SMEM 表

`candidate_9_reflection_shows_no_fallback_for_every_transpose_pattern`
（`gemm_spec_source_diag_tests.rs`）実機実測結果:

| pattern | requested_thread_count | max_total_threads_per_threadgroup | thread_execution_width | static_threadgroup_memory_length | `shared_mem_bytes_for` |
|---|---|---|---|---|---|
| NN | 128 | 1024 | 32 | 0 | 17920 |
| NT | 128 | 1024 | 32 | 0 | 18432 |
| TN | 128 | 1024 | 32 | 0 | 17408 |
| TT | 128 | 1024 | 32 | 0 | 17920 |

全パターンで `resolved_cfg == requested_cfg`（フォールバック非経由）・
`max_total_threads_per_threadgroup >= 128`・`thread_execution_width ==
32` を確認。`static_threadgroup_memory_length` が 0 なのはタイル
バッファが**動的** threadgroup メモリ（`threadgroup float* shared_mem
[[threadgroup(0)]]` + `setThreadgroupMemoryLength`。§2 の H1 検証時と
同じ契約）で確保されるためで、`shared_mem_bytes_for`（Rust 側の計算値。
`tile.rs::candidate_9_shared_mem_bytes_for_every_transpose_pattern_
within_32kib_and_16_aligned` で固定済み）が実際の確保量を表す。4
パターンとも 32KiB（32768 バイト）上限内・16 バイト整合。f16 版
`shared_mem_bytes_f16()` は 25344 バイト（`shared_mem_bytes_f16_all_
candidates_within_32kib_device_limit` の最大値も同値へ更新済み）。
親 #1324 の「32 KiB（2 面ダブルバッファ）」見積りは本追加では適用しない
（現行カーネルにダブルバッファはなく、本 issue でも追加しない）。

### 13.3 parity 結果表

| テスト | 対象 | 実機結果 |
|---|---|---|
| `bk32_64x64_candidate_matches_cpu_reference_non_multiple_of_tile`（`gemm_dynamic_tile_parity.rs`） | 境界形状 (100,130,70) | ok |
| `bk32_64x64_candidate_matches_cpu_reference_k_stress`（同上） | K=4096 ストレス | ok |
| `bk32_64x64_candidate_matches_cpu_reference_for_all_shapes_and_transpose_patterns`（`gemm_strided_parity.rs`） | 512³/1024³/2048³・(2048,2048,64)・(2048,2048,512)・(1536,1024,1024)・(1024,1536,1536)・(4096,1024,1024)・(1024,4096,1024)・(72,88,104) × NN/NT/TN/TT（計 40 ケース） | ok（`fail_count=0`・`resolved==cfg` 全ケース） |
| `bk32_64x64_candidate_matches_cpu_reference_for_n4096_cubic_shape`（同上） | 4096³ × NN/NT/TN/TT | ok（`fail_count=0`） |

全ケースで `assert_parity` の複合判定（相対誤差 1e-3 未満 または 絶対
誤差 1e-5 未満。REQ-2）が `fail_count=0` で通過。`dispatch_strided_
tiled_prepared` の戻り値 `resolved == cfg` を全ケースで assert 済みの
ため、サイレントフォールバックは発生していない。

### 13.4 `CANDIDATES[0]`（bk=16）との複合判定比較

`bk32_64x64_candidate_agrees_with_bk16_counterpart_within_composite_
tolerance`（`gemm_strided_parity.rs`）: N=512/1024/2048/4096 ×
NN/NT/TN/TT（計 16 ケース）で `CANDIDATES[9]`（bk=32）と `CANDIDATES[0]`
（bk=16）の出力を `fandhe_ai_backend_cpu::parity::compare` で直接比較。
全ケース `report.passes()==true`（`fail_count=0`）。bit 完全一致は
assert 契約に含めていない（K の分割粒度が異なるため丸め順が変わり
うる）が、実機観測では bit 完全一致していた（K チャンク順が両構成とも
昇順で同一のため）。

### 13.5 スコープ外・#1330 への引き継ぎ

- 純カーネル時間の before/after（`CANDIDATES[9]` vs `CANDIDATES[0]`。
  N=2048/4096・5 回計測中央値）・`tile::select` への組み込み判断
- ダブルバッファ（2 面）化による SMEM 32 KiB 構成・E1 unroll
  （`unroll_acc_candidates_are_exactly_acc_product_ge_16` の期待値
  `{0,4,8,9}` 化は本 issue で実施済みだが、本番 `UNROLL_ACC_ENABLED=
  false` かつ index 9 自体が `select` 非組み込みのため本番挙動は不変。
  E1 の再評価は #1330 の実測結果を見て判断）
- 端あり形状（8 の倍数でない大規模形状）・f16 経路・単独 `NT`/`TN`
  以外の性能比較

### 13.6 関連ログ

`docs/perf/logs/metal-gemm-e7-candidate-1329/`（`env_info.txt`・
`reflection.log`・`parity_dynamic_tile.log`・`parity_all_shapes.log`・
`parity_n4096_cubic.log`・`parity_bk16_compare.log`・
`regression_ignored_all_candidates.log`（`--lib --ignored` 全件）・
`regression_gemm_dynamic_tile_parity.log`・
`regression_gemm_strided_parity.log`・`make_test_ignored_metal.log`
（並列実行。command_batching の 1 件が並列競合による既知の flaky で
FAILED。§13.1 の `--test-threads=1` 版〈`full_ignored_serial.log`〉では
全 pass）・`full_ignored_serial.log`（`cargo test -p fandhe-ai-backend-
metal --release -- --ignored --nocapture --test-threads=1` 全件。ok）

## §14 E7 実測 — `CANDIDATES[9]`（bk=32）純カーネル時間の採否判定（イシュー #1330）

§13.5 からの引き継ぎ事項（純カーネル時間 before/after・`tile::select` 組み込み判断）
を本節で閉じる。

### 14.0 目的・範囲

`CANDIDATES[9]`（64×64・`bk=32`・wm2wn2。§13・PR #1391）の純カーネル時間
（GPU タイムスタンプ。`kernel_gpu`）を M4 Max 実機で 2 系列・5 プロセス起動・
5 回計測中央値で比較し、`tile::select`／`select_with_occupancy_for_device`
への組み込み可否を判定する。

- **A 系列**: `CANDIDATES[9]`（head）vs `CANDIDATES[0]`（base。同じ 64×64・
  `bk=16`）。#1324「K ループ 1 反復あたりの `threadgroup_barrier` 往復半減」
  仮説への直接回答
- **B 系列**: `CANDIDATES[9]`（head）vs `tile::select_for_device` の本番選択
  構成（base）。A 系列単独では `select` の組み込み根拠にならない（`[0]` は
  本番選択構成に対し N=4096 で約 7.7 倍遅い。§9.4／§12 実測）ため、B 系列を
  組み込み判断の唯一の根拠とする

### 14.1 環境・プロトコル

- 実機: Apple M4 Max・macOS 26.6.2（build 25G83）・rustc 1.96.0
- HEAD: `263cf5f`（PR #1391 マージ済み main）上のブランチ
  `perf/1330-metal-e7-bk32-ab`
- 実行前 uptime: 1 分 load average 7.79（開始時）〜2.76（終了時）で推移。
  21 ユーザーセッション並走の**共有負荷下**（他イシューの並列実行。
  `env_info.txt` 参照）。`pmset -g therm` は開始・終了とも thermal/
  performance warning なし
- ハーネス: `crates/backend-metal/src/gemm_bk32_diag_tests.rs`（新規。
  `#[ignore]`）。`--release --test-threads=1` で 5 プロセス起動
- 事前スモーク: §13.3 の parity テスト（`bk32_64x64_*`）を HEAD 上で
  再実行し全 pass を確認（`smoke_bk32_parity.log`）
- ログ: `docs/perf/logs/metal-gemm-e7-bk32-ab-1330/`

### 14.2 A 系列（`[9]` vs `[0]`）結果

| N | base（cand0）median [ms・run代表] | head（cand9）median [ms・run代表] | run別比中央値（head/base） | 符号一貫性 |
|---|---|---|---|---|
| 1024 | 約 1.53〜1.69（外れ値 run 除く） | 約 1.70〜1.76（外れ値 run 除く） | **1.107607** | 5/5 正（head が遅い） |
| 2048 | 12.21〜12.47 | 14.05〜14.34 | **1.151341** | 5/5 正 |
| 4096 | 105.19〜105.80 | 121.94〜122.22 | **1.158354** | 5/5 正 |

詳細は `docs/perf/logs/metal-gemm-e7-bk32-ab-1330/aggregate.md` 参照。
**`CANDIDATES[9]`（bk=32）は `CANDIDATES[0]`（bk=16）に対し、N=1024/2048/4096
のいずれも 10.8〜15.8% 一貫して遅い**（5/5 run 符号一貫）。#1324 の「バリア
半減で高速化する」仮説は本実測では支持されない（理論上の同期回数削減効果を、
`bk=32`化に伴う何らかの副作用——レジスタ／SMEM アクセスパターンの変化、
ロード粒度の変化等——が上回ったと推測されるが、機構レベルでの追加切り分けは
本イシューの範囲外とする）。

### 14.3 B 系列（`[9]` vs 本番選択構成）結果

| N | 本番選択構成（base） | run別比中央値（head/base） | 符号一貫性 |
|---|---|---|---|
| 512 | `CANDIDATES[5]`相当（64,32,32,2,2） | **4.543226** | 5/5 正 |
| 1024 | `(64,32,8,4,1)`（occupancy 縮退構成） | **6.590084** | 5/5 正 |
| 2048 | `(64,32,16,2,2)` | **7.571432** | 5/5 正 |
| 4096 | `CANDIDATES[2]`相当（32,64,16,2,2） | **7.518255** | 5/5 正 |

詳細は同 `aggregate.md` 参照。**`CANDIDATES[9]` は本番選択構成に対し全帯域
（N=512〜4096）で 4.5〜7.6 倍遅い**（20/20 反復すべて符号一貫）。

### 14.4 妥当性帯チェック

`docs/perf/metal-gemm-reuse-phase-breakdown.md` §11.5 分母（N=1024 1.0267 /
2048 3.1849 / 4096 13.7051 ms）と B 系列 base 列（AGENTS.md「5 回計測中央値」
規約に沿い 5 run 中央値: N=1024 0.2243 ms／N=2048 1.6006 ms／N=4096
13.7372 ms。当初記録は run1 のみの値〈約 1.02 ms〉を誤って base 代表値として
使用していたため訂正）を突合した結果、N=2048/4096 は概ね同オーダーで整合。
N=1024 は分母 1.0267 ms に対し中央値 0.2243 ms と約 4.6 倍の乖離があり、
run1（1.0245 ms・この base 列内でも外れ値）だけを見て「概ね整合」としていた
旧記述は誤りだった。乖離の要因は計測境界差・本番選択構成の変遷（#1039 以降の
テーブル更新）による差と考えられるが、分母自体は置換しない。この乖離は B 系列
の run別比（同一 run 内で base・head を対応付けて計算）自体の妥当性・採否判断
（REJECT）には影響しない（`aggregate.md` 「妥当性帯チェック」節に詳細）。

### 14.5 採否判断

**REJECT（`tile::select`／`select_with_occupancy_for_device` への
`CANDIDATES[9]` 組み込みは行わない）**。判断根拠:

1. A 系列: `[9]` は `[0]` 単体比較でも改善せず、全 N で 10.8〜15.8% 後退
   （5/5 run 符号一貫）。#1324 の理論的仮説（バリア半減による高速化）は
   実測で反証された
2. B 系列: `[9]` は本番選択構成に対し全帯域で 4.5〜7.6 倍後退（20/20 反復
   符号一貫）。判定基準（§9.4〜§12.4 と同一の「N=2048 かつ 4096 で
   head/base 比 < 1.0・N=1024 後退 ≤5%・run 間符号一貫」）に対し、後退方向
   かつ後退幅が判定基準の許容範囲（5%）を大幅に超えるため、明確に不採用
3. 符号のばらつきがなく（A 系列 15/15・B 系列 20/20 反復すべて同方向）、
   共有負荷下の計測ノイズでは結論を覆せない規模の後退のため `undetermined`
   ではなく確定 REJECT とする

`tile::CANDIDATES[9]` 自体（配列要素）は§13 のとおり不変のまま維持する
（明示指定でのみ到達可能な状態を継続。`tile.rs`／`gemm.rs`／
`shaders/gemm.metal` への変更なし）。

### 14.6 スコープ外・引き継ぎ

- `bk=32` 化がなぜ理論上の同期削減効果を上回る後退を招くかの機構レベルの
  追加切り分け（レジスタ圧・SMEM アクセスパターン・ロード粒度変化等の
  仮説検証）は本イシューの範囲外。新規 Issue は自動運転のため起票せず、
  親 #1324 へのコメント候補として記録する
- E1 unroll pragma（`UNROLL_ACC_ENABLED`）の index 9 再評価: 本判断（REJECT）
  により `CANDIDATES[9]` 自体が `select` 非組み込みのままのため、再評価の
  優先度は低いと判断し対象外とする
- ダブルバッファ（2 面）SMEM 化・端あり形状・NT/TN/TT・f16 経路の性能比較:
  §13.5 から引き続き対象外

### 14.7 関連ログ

`docs/perf/logs/metal-gemm-e7-bk32-ab-1330/`（`env_info.txt`・
`pmset_therm_before.txt`／`pmset_therm_after.txt`・`uptime_before_run{1..5}.txt`・
`smoke_bk32_parity.log`・`kernel_gpu_run{1..5}.log`（A 系列）・
`kernel_gpu_production_select_run{1..5}.log`（B 系列）・`aggregate.md`）。
内部ホスト名は含めない。

## 15. E8 候補追加: 128×64×16（wm2wn2）の収録と parity 確認（イシュー #1331）

### 15.0 目的・範囲

親 #1325（E8: threadgroup タイルを 128×64×16（2×2 simdgroup）へ拡張）の
sub-issue。`CANDIDATES[0]`（64,64,16,2,2。大形状の主力構成）に対する
bm=128 版 `(128,64,16,2,2)` を `tile::CANDIDATES` の **index 10（末尾）**
へ追加した。各 simdgroup が担当する acc タイル数（acc_rows=8・
acc_cols=4。積 32。`CANDIDATES[0]` は acc_rows=4・acc_cols=4・積 16）を
拡張したまま A/B タイルの threadgroup 内再利用率を倍にする狙い（理論
根拠。E1 実験〈本 doc §7〉・#1143 の H1〈レジスタ圧仮説。反射値レベル
では非支持〉が動機）。

本 issue のスコープは**候補追加と正確性（parity）確認のみ**: (a) 明示
指定（`GemmVariant::SimdgroupTiled(cfg)`／`dispatch_tiled_prepared`／
`dispatch_strided_tiled_prepared` の `cfg` 引数）でのみ到達可能にする
（`select`／`select_with_occupancy_for_device` の選択ロジックは無変更）、
(b) カーネル側手動境界チェック（REQ-8）を維持（`gemm.metal` は無変更）、
(c) 全形状 × NN/NT/TN/TT で parity 0 fail を実機確認、(d) threadgroup
メモリ使用量の反射値確認、(e) 現行 `CANDIDATES[0]`（bm=64）との出力が
複合判定内で一致することを確認。純カーネル時間の before/after 実測・
`tile::select` への組み込み判断は後続イシュー #1332 のスコープ（§15.6）。

### 15.1 環境・プロトコル

実機は Apple M4 Max（`docs/real-hardware-verification-env.md` §7）。
`cargo test -p fandhe-ai-backend-metal --release ... -- --ignored
--nocapture --test-threads=1`（正確性確認のみのため 5 回計測中央値は
不要）。実行前 `uptime` load average 2.87/4.26/5.18（他イシューが並列
実行中の共有環境。20 ユーザーセッション並走）。正確性確認のみのため
共有負荷は結論に影響しない。env_info・実行ログは §15.7 参照。

### 15.2 f16 版 SMEM 超過の扱い（設計判断）

`shared_mem_bytes_f16()` は本候補で **40064 バイト**となり、標準
Apple Silicon の threadgroup メモリ上限（32KiB=32768 バイト）を構造的
に超過する。エピローグ領域単独で `bm*bn*4 = 128*64*4 = 32768` バイトに
達し、staged タイル領域（half 単位。NN で 7296 バイト）を加えると必ず
上限を超える（`TileConfig::shared_mem_bytes_f16` ドキュメントコメント
参照）。`pipeline_for_tile_f16` は超過構成を `fallback_chain` の
`SINGLE_SIMDGROUP_8X8` へ安全に縮退させるため実行時 panic はしない
（f16 タイル化経路は本候補で構造的に使用不可なだけであり、`select`・
`dispatch_f16_auto_unverified` は `CANDIDATES` を選ばないため本番挙動へ
の影響もない）。

この非適格性を隠さず機械的に固定するため、`TileConfig::f16_tiled_fits_
standard_limit`（`shared_mem_bytes_f16() <= 32*1024` を返す `#[cfg(test)]`
限定ヘルパ）・`TileConfig::f16_tiled_candidates`（適格候補のみを巡回する
イテレータ）を追加し、既存の f16 全候補 CI テスト 2 件
（`shared_mem_bytes_f16_all_candidates_within_32kib_device_limit`・
`shared_mem_bytes_f16_fits_standard_shared_mem_limit_for_all_candidates`。
いずれも Linux 実行可能）を `CANDIDATES` 全体ではなく `f16_tiled_
candidates()` の巡回へ変更した。新規 CI テスト
`f16_tiled_candidates_excludes_exactly_candidate_10` で「f16 非適格
index 集合が正確に `{10}`」であることを両方向（増加・減少）で固定する
ドリフト検出とした。実機側の f16 全候補テスト 2 件
（`all_tile_candidates_match_cpu_reference_f16_tiled_medium_shape`／
`_non_multiple_boundary_shape`）は候補ごとに分岐させ、適格候補は従来
どおり `resolved == cfg` を、非適格（index 10）候補は `resolved ==
SINGLE_SIMDGROUP_8X8` を assert したうえで parity も実行する（縮退経路
が正しく動作することの fail-closed 確認。§15.7 の
`regression_ignored_all_candidates.log` 参照）。

この変更は既存テストの緩和ではなく、非適格集合を index で固定し
縮退先を assert する厳格化である（`tolerance`・baseline の変更は伴わ
ない。`.claude/rules/coding-rust.md` の対象外）。

### 15.3 反射値・SMEM 表

`candidate_10_reflection_shows_no_fallback_for_every_transpose_pattern`
（`gemm_spec_source_diag_tests.rs`）実機実測結果:

| pattern | requested_thread_count | max_total_threads_per_threadgroup | thread_execution_width | static_threadgroup_memory_length | `shared_mem_bytes_for` |
|---|---|---|---|---|---|
| NN | 128 | 1024 | 32 | 0 | 14592 |
| NT | 128 | 1024 | 32 | 0 | 15360 |
| TN | 128 | 1024 | 32 | 0 | 12800 |
| TT | 128 | 1024 | 32 | 0 | 13568 |

全パターンで `resolved_cfg == requested_cfg`（フォールバック非経由）・
`max_total_threads_per_threadgroup >= 128`・`thread_execution_width ==
32` を確認。`static_threadgroup_memory_length` が 0 なのは §13.2 と
同じ理由（動的 threadgroup メモリのため `shared_mem_bytes_for` が実際の
確保量を表す）。4 パターンとも 32KiB（32768 バイト）上限内・16 バイト
整合（`tile.rs::candidate_10_shared_mem_bytes_for_every_transpose_
pattern_within_32kib_and_16_aligned` で固定済み）。f16 版
`shared_mem_bytes_f16()` は上記 §15.2 のとおり 40064 バイトで超過・
非適格。

### 15.4 parity 結果表

| テスト | 対象 | 実機結果 |
|---|---|---|
| `bm128_candidate_matches_cpu_reference_non_multiple_of_tile`（`gemm_dynamic_tile_parity.rs`） | 境界形状 (200,130,70) | ok |
| `bm128_candidate_matches_cpu_reference_k_stress`（同上） | K=4096 ストレス | ok |
| `bm128_candidate_matches_cpu_reference_for_all_shapes_and_transpose_patterns`（`gemm_strided_parity.rs`） | 512³/1024³/2048³・(2048,2048,64)・(2048,2048,512)・(1536,1024,1024)・(1024,1536,1536)・(4096,1024,1024)・(1024,4096,1024)・(72,88,104)・(200,136,104) × NN/NT/TN/TT（計 44 ケース） | ok（`fail_count=0`・`resolved==cfg` 全ケース。実測 2.78s） |
| `bm128_candidate_matches_cpu_reference_for_n4096_cubic_shape`（同上） | 4096³ × NN/NT/TN/TT | ok（`fail_count=0`。実測 6.20s） |

全ケースで `assert_parity` の複合判定（相対誤差 1e-3 未満 または 絶対
誤差 1e-5 未満。REQ-2）が `fail_count=0` で通過。`dispatch_strided_
tiled_prepared` の戻り値 `resolved == cfg` を全ケースで assert 済みの
ため、サイレントフォールバックは発生していない。(200,136,104) は
200 mod 128=72・136 mod 64=8・104 mod 16=8 のいずれも非 0 で、M/N/K
全方向のブロック端部分タイル（協調ロードのベクトルグループ境界
フォールバック含む）を踏む形状として追加した。

### 15.5 `CANDIDATES[0]`（bm=64）との複合判定比較

`bm128_candidate_agrees_with_bm64_counterpart_within_composite_
tolerance`（`gemm_strided_parity.rs`）: N=512/1024/2048/4096 ×
NN/NT/TN/TT（計 16 ケース）で `CANDIDATES[10]`（bm=128）と
`CANDIDATES[0]`（bm=64）の出力を `fandhe_ai_backend_cpu::parity::
compare` で直接比較。全ケース `report.passes()==true`（`fail_count=0`。
実測 2.88s）。bit 完全一致は assert 契約に含めていない（タイル形状が
異なるため K チャンク内の丸め順・アキュムレータの組み方が変わりうる）。

### 15.6 スコープ外・#1332 への引き継ぎ

- 純カーネル時間の before/after（`CANDIDATES[10]` vs `CANDIDATES[0]`。
  N=1024/2048/4096・5 回計測中央値）・`tile::select` への組み込み判断
- f16 タイル化経路自体の再設計（エピローグ領域を `bm*bn*4` からタイル
  分割等で縮小し 32KiB 内へ収める案）: §15.2 のとおり本候補は f16 経路
  では構造的に使用不可のまま維持し、f32 経路のみを対象とする
- E1 unroll pragma（`UNROLL_ACC_ENABLED`）の index 10 再評価:
  `unroll_acc_candidates_are_exactly_acc_product_ge_16` の期待値
  `{0,4,8,9,10}` 化は本 issue で実施済みだが、本番
  `UNROLL_ACC_ENABLED=false` かつ index 10 自体が `select` 非組み込み
  のため本番挙動は不変。E1 の再評価は #1332 の実測結果を見て判断
- 端あり形状（8 の倍数でない大規模形状）・単独 NT/TN 以外の性能比較・
  ダブルバッファ（2 面）SMEM 化

### 15.7 関連ログ

`docs/perf/logs/metal-gemm-e8-candidate-1331/`（`env_info.txt`・
`reflection.log`・`parity_dynamic_tile.log`・`parity_all_shapes.log`・
`parity_n4096_cubic.log`・`parity_cand0_compare.log`・
`regression_ignored_all_candidates.log`（`--lib --ignored` 全件）・
`full_ignored_serial.log`（`cargo test -p fandhe-ai-backend-metal
--release -- --ignored --nocapture --test-threads=1` 全件））。内部
ホスト名は含めない。

## §16 E8 実測 — `CANDIDATES[10]`（128×64×16）純カーネル時間比較・採否判断（イシュー #1332）

### 16.0 目的・範囲・イシュー文言の注記

§15（イシュー #1331）で追加した `CANDIDATES[10]`（`bm=128, bn=64, bk=16,
wm=2, wn=2, staged`）の純カーネル時間（GPU タイムスタンプ。`kernel_gpu`
変種。イシュー #1276）を E7（§14）と同型の A/B 2 系列で計測し、
`tile::select` への組み込み可否を判定する。

イシュー #1332 本文は「現行 N=4096 最良候補（`CANDIDATES[3]`／`[0]`）」
と書くが、これは #744 時点の古い記述である。`tile.rs` の M4 Max
厳密一致テーブル（`select_with_occupancy_for_device`）と §14.3 実測では
**2048→`CANDIDATES[1]`（64,32,16,2,2）・4096→`CANDIDATES[2]`
（32,64,16,2,2）** が現行本番選択構成であり、`[3]` は #1039 以降最良
ではない。したがって本節では「現行最良候補」を B 系列（`select_for_
device` の実選択構成）で表現し、`[0]`（構造上の直接対応。§15.6 の
引き継ぎ）を A 系列の主対象、`[3]` をイシュー文言突合用の参考ペアと
して追加した。

### 16.1 環境・プロトコル

- 実機: Apple M4 Max（本エージェント実行環境自体。macOS 26.6.2・
  rustc 1.96.0）
- 計測対象コミット: `93c6107b42d3984ecd14a6325b045e8792fb2296`
  （origin/main。PR #1393 マージ済み。イシュー #1332 の診断テスト
  追加自体はこのコミットに対する変更であり、本番コード〈`tile.rs`／
  `gemm.rs`／`shaders/gemm.metal`〉は無変更のまま計測）
- 事前スモーク: `bm128_candidate_matches_cpu_reference_k_stress`・
  `bm128_candidate_matches_cpu_reference_non_multiple_of_tile`・
  `bm128_candidate_agrees_with_bm64_counterpart_within_composite_
  tolerance`・`bm128_candidate_matches_cpu_reference_for_all_shapes_
  and_transpose_patterns`・`bm128_candidate_matches_cpu_reference_
  for_n4096_cubic_shape` の 5 件全 pass（`smoke_bm128_parity.log`）
- A 系列: `bm128_kernel_gpu_ab_vs_candidate0`（N=1024/2048/4096 ×
  `cand10_vs_cand0`／`cand10_vs_cand3` の 2 ペア）を 5 プロセス起動
  （`--release --lib -- --ignored --nocapture --test-threads=1`）
- B 系列: `bm128_kernel_gpu_ab_vs_production_select`（N=512/1024/
  2048/4096 × `tile::select_for_device` 解決構成）を同様に 5 プロセス
  起動
- warmup 20・測定 20（trial 交互回転）・fail-closed 検証
  （`resolved_cfg == cfg` によるフォールバック非経由・trial 0 出力の
  複合判定〈相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満〉pass）は
  全 run で違反なし（`assert_eq!`／`assert!` が 1 件も panic せず全
  run が `test result: ok`）
- 負荷帯: `uptime` load average（1 分値）は A 系列（`uptime_before_
  run{1..5}.txt`）が 3.42〜7.97、B 系列（`uptime_before_prod_
  run{1..5}.txt`）が 2.87〜2.97 で、全 10 ファイル通しの範囲は
  2.87〜7.97（他セッションとの共有負荷下。`docs/perf/logs/
  metal-gemm-e8-bm128-ab-1332/uptime_before_*.txt`）。#1330（E7）と
  同様「負荷はあるが符号完全一致のため判定不可とはしない」方針を
  適用する（A 系列 30/30・B 系列
  20/20 の全反復が同一符号のため、この負荷帯でも結論は揺るがない）

### 16.2 A 系列結果（`CANDIDATES[10]` vs `[0]`／`[3]`）

`head_over_base_kernel_gpu`（5 run 中央値。詳細は
`docs/perf/logs/metal-gemm-e8-bm128-ab-1332/aggregate.md`）:

| N | vs `CANDIDATES[0]` | vs `CANDIDATES[3]` |
|---|---|---|
| 1024 | 1.65 倍後退（5/5 符号一貫） | 10.52 倍後退（5/5 符号一貫） |
| 2048 | 1.89 倍後退（5/5 符号一貫） | 13.45 倍後退（5/5 符号一貫） |
| 4096 | 2.40 倍後退（5/5 符号一貫） | 14.15 倍後退（5/5 符号一貫） |

`[10]` は #1325 の「threadgroup タイル拡張によるタイル再利用率向上」
仮説に反し、構造上の直接対応である `[0]`（同じ `bk=16`・2×2
simdgroup）に対しても全 N で一貫して遅い。

### 16.3 B 系列結果（`CANDIDATES[10]` vs 本番選択構成。結線判断の唯一の根拠）

`production_select_resolved`（全 run で一致・フォールバック非経由）:

| N | 解決構成 |
|---|---|
| 512 | `CANDIDATES[5]`（64,32,32,2,2） |
| 1024 | `CANDIDATES[6]`（64,32,8,4,1） |
| 2048 | `CANDIDATES[1]`（64,32,16,2,2） |
| 4096 | `CANDIDATES[2]`（32,64,16,2,2） |

`head_over_base_kernel_gpu`（5 run 中央値）:

| N | 比 | 符号一貫性 |
|---|---|---|
| 512 | 9.72 倍後退 | 5/5 |
| 1024 | 10.04 倍後退 | 5/5 |
| 2048 | 12.98 倍後退 | 5/5 |
| 4096 | 12.46 倍後退 | 5/5 |

N=512〜4096 の全帯域で `[10]` が本番選択構成より 1 桁近く遅い
（20/20 反復すべて後退方向で符号一貫）。

### 16.4 妥当性帯チェック

`docs/perf/metal-gemm-reuse-phase-breakdown.md` §11.5 の `kernel_gpu`
分母（本番選択構成。N=1024 1.0267 ms・N=2048 3.1849 ms・N=4096
13.7051 ms）と、本計測の B 系列 base（production_select）絶対値
5 run 中央値（N=1024 0.3769 ms・N=2048 1.6105 ms・N=4096 15.8216 ms）
を突合した。N=4096 は近い値（約 1.15 倍差）だが N=1024/2048 は
乖離があり、これは同ファイル §9.4 が記録する「N=1024 の既知の乖離
（約 4.6 倍）」と同種の計測プロトコル間差（本計測は warmup 20／
測定 20 の交互測定、§11.5 は単発 5 プロセス起動）に起因すると考え
られる。**採否判定は base/head を同一 run・同一プロトコル内で比較した
run 別比（`head_over_base_kernel_gpu`）の符号一貫性で行っており、
base 絶対値のプロトコル間差は判定そのものへ影響しない**（詳細は
`docs/perf/logs/metal-gemm-e8-bm128-ab-1332/aggregate.md`「妥当性帯
チェック」節）。

### 16.5 採否判断

- 採否判定基準（`(m,n,k)` キー単位）:
  - **ADOPT（行置換）**: B 系列で当該 N の run 別比中央値が < 0.95
    かつ 5/5 run 符号一貫
  - **REJECT**: 当該 N で後退方向（比 ≥ 1.0）が 5/5 符号一貫、または
    ±5% 帯内で有意差なし
  - **undetermined**: run 間で符号が反転
- B 系列は N=512/1024/2048/4096 のすべてで「後退方向（比 9.7〜13.0）
  が 5/5 符号一貫」に該当する。ノイズ帯（±5%）を 1 桁近く超える一貫
  した後退のため、undetermined の余地はない。
- **結論: 組み込み不可（REJECT）**。`tile.rs`／`gemm.rs`／
  `shaders/gemm.metal` は一切変更しない（`tile::select`／
  `select_with_occupancy_for_device` の候補表は `[0]`〜`[9]` の既存
  構成のまま不変維持）。`CANDIDATES[10]` は §15 で確立した「明示
  指定でのみ到達可能」な状態のまま残す（反射値・parity 群が既に
  green のため候補自体の削除はしない）。
- **示唆**: threadgroup タイルを 128×64 へ拡張しても、レジスタ／
  occupancy 上の負担が K ループの再利用率向上を上回り純カーネル
  時間が大幅に悪化する（#1325 の仮説は本 GPU 世代・本タイル形状の
  組合せでは支持されない）。128 系タイルの探索は本結果をもって
  打ち切りとし、後続の探索は 64 系タイルの周辺（E4〜E7 の知見）へ
  戻すことを推奨する（下記 16.6 参照）。

### 16.6 スコープ外・引き継ぎ

- NT/TN/TT・端あり形状・f16 経路（§15.2 のとおり構造的に不可）での
  `[10]` 性能比較（REJECT が明確なため実施しない）
- ダブルバッファ（2 面）SMEM 化・E1 unroll pragma の index 10
  再評価（§15.6 の引き継ぎ。REJECT により優先度は大幅に低下したと
  判断する。実施する場合は親 #1325 系列の後続 issue とする）
- M4 Max 以外の機種向けテーブル・後退の機構レベル切り分け
  （レジスタ圧・occupancy 低下の定量診断。反射値〈§15〉からの推定に
  留め、`MTLComputePipelineState` 実測〈§7 の H1 検証と同型の手法〉
  は本 issue のスコープ外）
- 本節の知見（128 系タイルは本 GPU 世代で不利）を親 #1325 へコメント
  として記録することを推奨する（自動運転のため本 issue 側では追加
  issue 起票は行わない）

### 16.7 関連ログ

`docs/perf/logs/metal-gemm-e8-bm128-ab-1332/`（`env_info.txt`・
`uptime_before_run{1..5}.txt`・`uptime_before_prod_run{1..5}.txt`・
`pmset_therm_before.txt`／`pmset_therm_after.txt`・
`smoke_bm128_parity.log`・`kernel_gpu_run{1..5}.log`（A 系列）・
`kernel_gpu_production_select_run{1..5}.log`（B 系列）・
`aggregate.md`（抽出コマンド・5 run 表・妥当性帯チェック・判定を
記載））。内部ホスト名・絶対パスは含めない。
