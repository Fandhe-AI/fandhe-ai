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
| cand3（32×32,wm2wn2） | 8 | 3.97/2.91/4.22/3.87/3.92 | 3.92 | 8.21/4.32/4.16/3.15/3.85 | 4.16 | 約 1.06 倍 |
| cand5（64×32×32,wm2wn2） | 8 | 3.31/3.51/3.48/3.94/3.28 | 3.48 | 6.82/4.94/4.85/4.53/3.21 | 4.85 | 約 1.39 倍 |
| cand6（64×32×8,wm4wn1） | 8 | 3.18/3.88/4.10/4.38/4.01 | 4.01 | 5.08/4.69/3.95/4.42/3.94 | 4.42 | 約 1.10 倍 |
| cand7（single simdgroup） | 8 | 1.20/1.60/1.66/1.54/1.43 | 1.54 | 1.65/1.71/1.70/1.44/1.47 | 1.65 | 約 1.07 倍 |
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
