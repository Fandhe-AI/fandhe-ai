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
  結線は行わない）。E4（協調ロード再構成）は引き続き未着手のまま
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
