# Metal GEMM 1024 以降スループット頭打ち context_cache 後の再診断（#1036）

イシュー #1036「docs(perf): Metal GEMM 1024 以降の頭打ちを context_cache 後の構成で M4 Max 実機診断する」の
実測記録。親 #1029「GEMM カーネルの candle 超え」配下。イシュー #1103「docs(perf): Metal GEMM 頭打ち診断の
残件（同一境界での candle 転送分離測定・GPU カウンタ表）を実施する」による追補を含む（§1.1・§5.3・§7.1b）。

`docs/perf/metal-gemm-bottleneck-diagnosis.md`（#487）・`docs/perf/metal-fixed-overhead-diagnosis.md`
（#927）はいずれも現行実装と前提が乖離している（#744 是正前の `tile::select`・`context_cache`（#930/#948）
導入前の固定費構成）ため、本 doc で現行構成（2026-08-30 時点の `main`）を前提に再診断する。

## 状態: 実測完了（M4 Max 実機・ローカル直接実行セッション）

`crates/backend-metal/src/`・`shaders/gemm.metal` は変更していない（診断タスク。#487/#927 の先例に従う）。

## 1. 計測環境

| 項目 | 値 |
|------|-----|
| チップ | Apple M4 Max（`sysctl -n hw.model` = `Mac16,6`） |
| OS | macOS 26.6.2（BuildVersion 25G83） |
| Xcode / xctrace | Xcode 26.6（17F113）／ xctrace 16.0（17F113） |
| rustc | 1.96.0（ac68faa20 2026-05-25） |
| 計測コミット SHA | `2a12c44294c707579254a90286573243293a0d40`（`origin/main` fetch 直後の HEAD） |
| 計測プロトコル | `bench-harness::protocol::run`（`MeasurementConfig::default()` = warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1）。§4 の追加スイープ（`gemm_tile_sweep` example）も同一 `MeasurementConfig::default()` を使用 |
| 決定的シード | `0xC0FFEE`（各 example の `SEED` 定数） |
| 計測衛生 | AC 電源接続・実行順は本 doc の記載順（§2→§3→§4→§5→§6）どおり連続実行。他 GPU 負荷アプリの明示終了は未実施（既定のバックグラウンドプロセスのみ）。**§3.4 で後述するとおり、別プロセス・別バイナリでは同一構成でも中央値が約 3 倍変動するプロセス間変動を観測した（原因はサーマル/クロック等の候補があるが未統制・未記録のため帰属は未確定）** — 詳細は §3.4 参照 |

生ログはリポジトリ内 `docs/perf/logs/metal-gemm-rediagnosis-1036/` に保存する（`step2_gemm_diagnosis.log`・
`step3_kernel_pure.log`・`step4_gemm_bench.log`・`step4b_extra_candidates.log`・
`step4b_extra_candidates_rerun.log`〈§4.4 追記のとおり `gemm_tile_sweep` example による再実行ログ〉・
`step6_fixed_overhead.log`・`durations_raw.txt`〈§5.1 の GPU 区間抽出〉。Review 指摘・#1036: 一時パスのみ
では後続セッションから監査・再現できないため収録。`metal_trace.trace` バンドル・`gpu_intervals.xml`
（約 720 KB）・`toc.xml` はサイズとバイナリ性のため未収録で、§5.1 の目視抽出値の一次出典は
`durations_raw.txt` とする）。

### 1.1 #1103 追補セッションの環境

イシュー #1103（本 doc §7.1「candle 転送分離測定」・§5「GPU counters」の未計測 2 点の解消）は
別セッションで実施した。環境は上表と同一実機（Apple M4 Max・macOS 26.6.2・AC 電源接続）だが、
計測コミット SHA・xctrace バージョンは以下のとおり更新されている。

| 項目 | 値 |
|------|-----|
| 計測コミット SHA | `b5c8f11c5a3061ff66c6acdddd47cf9c41c7d042`（`origin/main` fetch 直後の HEAD。#1104 マージ後） |
| xctrace | 16.0（17F113）。#1036 と同一バージョン |
| 追補範囲 | §7.1（candle transfer-split 測定・主因記述の更新）・§5.3（新設。GPU counters 再試行） |
| 生ログ | `step7_candle_transfer_split.log`・`step7_fandhe_gemm_bench_rerun.log`・`step8_gpu_counters_{1024,2048,4096}.log` を追加収録（下記 §5.3・§7.1 参照） |

## 2. 0.4.0 実測ベースライン（転記・出典明記）

親計画（Plan フェーズ）が `scripts/bench/framework-compare/results/raw/results-m4max-0.4.0.jsonl`
（2026-08-29・fandhe-ai 0.4.0）から抽出した値を転記する。当該ファイルは本セッションの worktree には
存在しない（untracked ファイルのため git worktree 間で共有されない）ため、本 doc 側では独立に開き直して
いない。数値は Plan フェーズの記載をそのまま転記し、独自の実測ではないことを明記する。

| size | mode | fandhe-ai 中央値 | fandhe-ai GFLOP/s | candle（MLX steel gemm）中央値 | candle GFLOP/s |
|------|------|-------------------|---------------------|-----------------------------------|------------------|
| 4096 | reuse | 45.339 ms | 3,031 | 27.27 ms | 5,040 |
| 4096 | fresh | 40.481 ms | 3,395 | — | — |

計測境界: reuse でも readback・checksum を含む end-to-end（`bench-fandhe` バイナリ経由。facade
`context_cache` を使った `tape_for` 再利用構成）。§3 以降の本 doc 実測（`MetalGemm` 直接呼び出し）とは
計測境界・呼び出し経路が異なる点に注意（§5「経路間の比較に関する注意」で扱う）。

## 3. 現行構成での解析値・壁時計再計測（旧 #487 doc §3.2 の是正）

### 3.1 `tile::select` の現行挙動（#744 是正後）

`crates/backend-metal/src/tile.rs:679-683` の `select(m, n, k)` は `m == n == k` かつ `m <= 4096` の
正方立方形状に対し一律 `CANDIDATES[3]`（32×32×16, wm=2, wn=2, staged=true）を返す
（`docs/perf/metal-tile-select-correction.md`・#744 実測確定）。旧 #487 doc §3.2 は「4 サイズとも
64×64×16, wm=wn=2 を選択する」という #744 是正前の前提で記述されており、現行 `main` とは一致しない。

### 3.2 現行構成での解析値・壁時計実測

```
cargo run -p fandhe-ai-backend-metal --example gemm_diagnosis --release -- \
  --gpu-core-count=40 --ideal-groups-multiplier=6 --iters=200
```

| size | tile | actual_groups | ideal_groups (=240) | actual/ideal | barriers_per_tg | arithmetic_intensity (FLOP/byte) | wall_ms（中央値） | wall_q1_ms | wall_q3_ms | tflops_lower_bound |
|------|------|----------------|----------------------|---------------|-------------------|-------------------------------------|----------------------|------------|------------|------------------------|
| 512  | 32×32×16, 2×2, staged | 256   | 240 | 1.067  | 64  | 7.7576 | 0.4265  | 0.3950 | 0.5049 | 0.6295 |
| 1024 | 32×32×16, 2×2, staged | 1,024 | 240 | 4.267  | 128 | 7.8769 | 1.2148  | 1.1790 | 1.2710 | 1.7677 |
| 2048 | 32×32×16, 2×2, staged | 4,096 | 240 | 17.067 | 256 | 7.9380 | 5.4939  | 5.4188 | 5.5905 | 3.1271 |
| 4096 | 32×32×16, 2×2, staged | 16,384| 240 | 68.267 | 512 | 7.9689 | 31.6801 | 31.3015| 32.0760| 4.3383 |

（生ログ: `step2_gemm_diagnosis.log`）

### 3.3 旧 #487 doc §3.3 の再評価

旧 doc §3.3 は「1024 以降 `actual_groups` が `ideal_groups` を大きく上回る領域でも頭打ちが観測される」
ことから「並列度〈concurrency/saturation〉不足の仮説では 1024 以降の頭打ちを説明できない」と暫定観察
していた。#744 是正後（32×32 タイル・§3.2 表）でも同じ構図が**より強く**現れる: `actual/ideal` 比は
1024 で 4.267 倍・4096 で **68.267 倍**（旧 64×64 構成の 17.067 倍よりさらに過飽和側）。この proxy
指標だけを見る限り、`ideal_groups`（MFA 経験式の concurrency/saturation 目標）に対する threadgroup
発行数の不足という意味での「並列度不足」は、少なくとも 1024 以降の頭打ちの説明にならない、という旧 doc
の暫定観察は現行構成でも変わらず支持される。ただし本 proxy はレジスタ・threadgroup memory 等の資源制約
を表さない一次指標に過ぎない（旧 doc §1 の限界注記のとおり）。真の occupancy（資源制約側）の実測は
§5「GPU counters 内訳表」参照（未計測。理由は同節）。

### 3.4 計測衛生上の所見: プロセス間の中央値変動（原因未確定）

§4.2（`gemm_f32_prepared_bench`。アイドル直後に単独実行）と §4.4（`gemm_tile_sweep` example。直前に §4.3
`gemm_bench` フルスイート実行済み）で同一構成（32×32×16, wm=2, wn=2, staged, N=1024）を計測したところ、
中央値 TFLOPS が 1.76 → 5.44（約 3.1 倍）に変動した。両者は別プロセス・別バイナリの計測で、直前負荷以外の
条件も統制しておらず温度・クロックも記録していないため、この差をサーマル/クロック状態に帰属することは
できない（**原因不明のプロセス間変動**として扱う。帰属の検証には同一ハーネスで負荷順序を入れ替えた反復と
温度・クロックの記録が必要で、本 doc ではスコープ外〈§8〉。Review 指摘・#1036）。プロセス間の絶対値比較に留め、本 doc の結論では **同一プロセス内で連続計測した値同士の相対比較**
（§4.3・§4.4 それぞれの候補間順位）のみを主要な根拠として扱い、プロセスをまたいだ絶対 TFLOPS の比較
（§2 の 0.4.0 ベースラインを含む）は参考値に留める。

## 4. カーネル純境界・タイル候補スイープ

### 4.1 転送境界の分離（受け入れ条件の前提整理）

`dispatch_auto`（`gemm_bench.rs::measure_auto`）は A・B のアップロード＋カーネル実行＋C readback を
含む end-to-end 境界。`dispatch_tiled_prepared`（`gemm_f32_prepared_bench.rs`・`gemm_bench.rs` の
`measure_tiled_prepared`）はバッファをループ外で 1 回だけアップロードし、計測対象をディスパッチ
（エンコード＋コマンドバッファ完了待ち）のみに絞る（readback 非計測）。両者を同一プロセス内・同一
`tile::select` 結果（32×32×16）で比較することで、転送（アップロード＋readback）の寄与を分離する。

### 4.2 カーネル純境界実測（`gemm_f32_prepared_bench`）

```
cargo run -p fandhe-ai-backend-metal --example gemm_f32_prepared_bench --release
```

| size | resolved tile | tflops（中央値） | q1 | q3 |
|------|-----------------|---------------------|------|------|
| 512  | 32×32×16, 2×2, staged | 0.8234 | 0.8062 | 0.8378 |
| 1024 | 32×32×16, 2×2, staged | 1.7623 | 1.7343 | 2.6785 |
| 2048 | 32×32×16, 2×2, staged | 7.8392 | 7.8225 | 7.8908 |
| 4096 | 32×32×16, 2×2, staged | **8.2923** | 8.2771 | 8.3144 |

（生ログ: `step3_kernel_pure.log`）

### 4.3 end-to-end 境界との同一プロセス内比較（`gemm_bench`）

```
cargo run -p fandhe-ai-backend-metal --example gemm_bench --release
```

`dispatch_auto`（end-to-end。転送込み）:

| size | naive | tiled | simdgroup | dynamic_tile_auto |
|------|-------|-------|-----------|----------------------|
| 256  | 0.0768 | 0.1008 | 0.1116 | 0.1205 |
| 512  | 0.1745 | 0.3260 | 0.4184 | 0.6489 |
| 1024 | 0.8200 | 1.0153 | 1.1757 | 1.7914 |
| 2048 | 1.0447 | 1.3624 | 1.6718 | 3.1999 |
| 4096 | 1.1047 | 1.5110 | 2.0186 | **4.3965** |

候補構成比較（`dispatch_tiled_prepared`。転送非計測。同一プロセス内で §4.2 と同じ 32×32×16 構成を
再確認しつつ、64×64・TM/TN 拡大系〈#745〉も比較）:

| size | candidate | tflops | resolved_matches_requested |
|------|-----------|--------|-------------------------------|
| 2048 | bm64_bn64_bk16_staged | 1.2140 | true |
| 2048 | **bm32_bn32_bk16_staged**（現行 CANDIDATES[3]） | **8.4334** | true |
| 2048 | bm32_bn32_bk16_direct（staged=false） | 3.0630 | true |
| 2048 | tm8_tn4_bm128_bn64_bk16_staged | 0.5557 | true |
| 2048 | tm4_tn8_bm64_bn128_bk16_staged | 0.8308 | true |
| 2048 | tm8_tn8_bm128_bn128_bk16_staged | 0.3921 | true |
| 4096 | bm64_bn64_bk16_staged | 1.2079 | true |
| 4096 | **bm32_bn32_bk16_staged**（現行 CANDIDATES[3]） | **7.1049** | true |
| 4096 | bm32_bn32_bk16_direct（staged=false） | 2.9603 | true |
| 4096 | tm8_tn4_bm128_bn64_bk16_staged | 0.5451 | true |
| 4096 | tm4_tn8_bm64_bn128_bk16_staged | 0.8879 | true |
| 4096 | tm8_tn8_bm128_bn128_bk16_staged | 0.3587 | true |

（生ログ: `step4_gemm_bench.log`。全出力〈非正方形状・occupancy 判定組み込み比較を含む〉は同ログ参照）

**end-to-end（`dynamic_tile_auto_tflops`）とカーネル純境界（`dispatch_tiled_prepared`）の同一プロセス内比較**:
同一 `gemm_bench` プロセス内で N=4096 は end-to-end 4.3965 TFLOPS に対し、同構成（32×32×16, staged）の
prepared 値 7.1049 TFLOPS。**転送（アップロード＋readback）を除くとスループットは約 1.62 倍になる**
（転送が end-to-end 時間の約 38% を占める）。これは同一プロセス・同一 `tile::select` 結果での比較であり、
転送コストの寄与を示す直接的な実測根拠である。なお §4.2 の 8.2923 TFLOPS は別コマンド
（`gemm_f32_prepared_bench`）・別プロセスの値のため、§3.4 の計測契約に従いプロセス間の参考値に留め、
本 doc の結論には用いない（Review 指摘・#1036: 別プロセスの値を「同一プロセス内比較」と誤記していた）。

### 4.4 追加タイル候補（CANDIDATES[4]/[5]/[6]）スイープ

`crate::tile::CANDIDATES` のうち §4.3 で未比較の候補（`[4]` 64×64×16 wm=1,wn=2 ／ `[5]` 64×32×32
wm=2,wn=2 ／ `[6]` 64×32×8 wm=4,wn=1）を、`[0]`（64×64×16 baseline）・`[3]`（32×32×16 現行選択）と
併せて `crates/backend-metal/examples/gemm_tile_sweep.rs`（`dispatch_tiled_prepared` を直接呼ぶ example。
初回計測時は一時パスのみで生成・削除していたが、codex-review 指摘〈PR #1096・P2〉を受け再現可能な
example として本 doc と同時に収録した）で比較した。

| size | candidate | tflops | resolved_matches_requested |
|------|-----------|--------|-------------------------------|
| 1024 | cand0_64x64x16_wm2wn2_staged | 1.2335 | true |
| 1024 | **cand3_32x32x16_wm2wn2_staged**（現行） | **5.4401** | true |
| 1024 | cand4_64x64x16_wm1wn2_staged | 0.8545 | true |
| 1024 | cand5_64x32x32_wm2wn2_staged | 5.4800 | true |
| 1024 | cand6_64x32x8_wm4wn1_staged | 5.3771 | true |
| 2048 | cand0_64x64x16_wm2wn2_staged | 1.2192 | true |
| 2048 | **cand3_32x32x16_wm2wn2_staged**（現行） | **8.8209** | true |
| 2048 | cand4_64x64x16_wm1wn2_staged | 0.5040 | true |
| 2048 | cand5_64x32x32_wm2wn2_staged | 7.4771 | true |
| 2048 | cand6_64x32x8_wm4wn1_staged | 7.4918 | true |
| 4096 | cand0_64x64x16_wm2wn2_staged | 1.1824 | true |
| 4096 | **cand3_32x32x16_wm2wn2_staged**（現行） | **5.3070** | true |
| 4096 | cand4_64x64x16_wm1wn2_staged | 0.4052 | true |
| 4096 | cand5_64x32x32_wm2wn2_staged | 4.5798 | true |
| 4096 | cand6_64x32x8_wm4wn1_staged | 4.8558 | true |

（生ログ: `step4b_extra_candidates.log`。§3.4 のとおり本スイープは §4.3 実行直後のため絶対値は §4.2/§4.3
と直接比較しない。1024 の cand3 が §4.2 の同構成〈1.76 TFLOPS〉と大きく乖離するのは §3.4 の原因未確定の
プロセス間変動によるもので、本スイープ内部の相対順位のみを根拠として用いる。再収録 example
〈`gemm_tile_sweep`。生ログ: `step4b_extra_candidates_rerun.log`〉による再実行ログ。順位傾向の確認用:
cand0／cand4（64×64 系）が cand3／cand5／cand6 より明確に劣る傾向・cand3／cand5／cand6 が互いに近接する
傾向は再現したが、2048 では cand3 と cand6 の順位（この再実行では cand6 がわずかに上回る。8.9620 vs
8.8894 TFLOPS、+0.8%）が入れ替わった。1024 も cand6 が cand3 をわずかに上回る（5.3265 vs 5.2399
TFLOPS）。いずれも §4.4 判断基準の 5% 未満の差である）

**観察**: 4096 では現行 `CANDIDATES[3]`（32×32×16, wm=2, wn=2, staged）がテストした 7 構成中最良で、
`cand5`（64×32×32, bk=32）・`cand6`（64×32×8, wm=4）は初回・再実行とも `cand3` の 84〜91% 程度に
留まる。2048 は初回計測では `cand3` 最良（`cand5`/`cand6` は 84〜85%）だが、再実行ログでは `cand6` が
`cand3` を +0.8% 上回っており、1024 と同じ判断基準（相対 5% 未満は同等）を適用して **2048 は `cand3` と
`cand6` を同等**と評価する（Review 指摘・#1036: 再実行結果にも同じ基準を一貫適用する）。
1024 では `cand5` 5.4800 TFLOPS が `cand3` 5.4401 TFLOPS を記載値上わずかに上回る（+0.7%）が、§3.4 で
同一構成の中央値がプロセス間で約 3 倍動く計測衛生の下で本スイープ内の 1% 未満の差は判断基準（候補間の
順位差として採用する閾値: 相対 5% 以上。統計的な有意差検定は行っておらず、本スイープの生ログ
〈`step4b_extra_candidates.log`〉には `MeasurementConfig::default()`〈計測 20 回〉の中央値のみを出力し
Q1/Q3 は記録していないため、5% は分位点に基づく値ではなく本 doc の工学的判断基準である）に満たず、**1024 は `cand3` と `cand5`
を同等**と評価する（`cand3` 最良と断定しない。Review 指摘・#1036）。`cand0`／`cand4`（64×64 系）・
TM/TN 拡大系（§4.3）はいずれも `cand3` の 15〜30% 程度に留まる。

## 5. GPU counters 内訳表（受け入れ条件 (a)）

### 5.1 採取試行と結果

```
xcrun xctrace record --template 'Metal System Trace' --launch -- \
  ./target/release/examples/gemm_f32_prepared_bench
xcrun xctrace export --input <trace> --toc
xcrun xctrace export --input <trace> \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'
```

上記でトレース採取自体は成功し（`metal_trace.trace`）、`--toc` は本テンプレートが持つテーブル一覧
（`gpu-counter-value`・`metal-gpu-counter-profile`・`metal-gpu-counter-intervals` を含む）を返した。
ただし `metal-gpu-counter-profile` テーブルの `counter-profile` 属性値は `0`（無効）であり、**「Metal
System Trace」テンプレートの既定設定では GPU Counters（occupancy・ALU/メモリ limiter 内訳）のカウンタ
プロファイルが有効化されていない**ことを確認した。GPU Counters を有効化するには Instruments GUI で
当該テンプレートのカウンタセットを明示選択する必要があり、非対話セッション（本サブエージェント実行環境）
からは操作できない。`MTLCounterSampleBuffer` のカーネル内組み込み（src 変更を要する）による代替採取は
本イシューのスコープ外（計画 §5「A08」・§6「スコープ外」節）と判断した。

`metal-gpu-intervals` テーブル（コマンドバッファ単位の実 GPU 実行区間）は採取できており、対象プロセス
（`gemm_f32_prepared_bench`）の `Compute` チャネルで個々の `Command Buffer N:Compute Command 0` 区間
（実 GPU 実行時間）を直接確認できた。例（N=512 帯・トレース冒頭付近の連続 10 件、`gpu_intervals.xml`
から目視抽出）: 167.58 µs・162.96 µs・165.04 µs・164.17 µs・164.67 µs・165.96 µs・164.79 µs・162.83 µs・
164.42 µs・163.42 µs（コマンドバッファあたりの実 GPU 実行時間。ホスト側ディスパッチオーバーヘッドを
含まない）。この値は §4.2 の N=512 中央値（0.8234 TFLOPS → 1 回あたり約 326 µs の壁時計時間。
`2*512^3/0.8234e12` 秒。Review 指摘・#1036: 旧記載の「約 654 µs」は換算誤り）よりも短く（約半分）、壁時計
時間には GPU 実行そのもの以外の待機・同期コスト（エンコード・コマンドバッファ完了待ち）が含まれること
を示唆するが、本表から `Compute` チャネルの区間を size ごとに機械的に紐付けるパーサ（`<row>` の
`ref` 属性による値の再利用〈XML 上の重複排除〉を解決する必要がある）を本セッションでは完成させられな
かったため、size 別の系統的な内訳表は**未計測**として扱う。

### 5.2 未計測部分の代替（proxy 証跡）

真の occupancy（レジスタ・threadgroup memory 使用率）・ALU/メモリ limiter の内訳は上記のとおり未計測
のため、§3・§4 で得た代替証跡を占有度合いの判断材料とする:

- **並列度〈concurrency/saturation〉proxy**（§3.2）: `actual_groups` は 1024 以降常に `ideal_groups`
  を大きく上回る（1024: 4.267 倍 〜 4096: 68.267 倍）。少なくとも「発行 threadgroup 数の不足」という
  意味での並列度不足は考えにくい
- **カーネル純境界 vs end-to-end の分離**（§4.3）: 同一 `gemm_bench` プロセス内で N=4096 のカーネル純
  境界（prepared 7.10 TFLOPS）は end-to-end（4.40 TFLOPS）の約 1.62 倍。転送（アップロード＋readback）が
  end-to-end 時間の約 38% を占める
- **タイル形状スイープ**（§4.3・§4.4）: `CANDIDATES` 全 7 構成中 `CANDIDATES[3]`（32×32×16）が 4096
  で最良（1024 は `cand5`/`cand6`、2048 は `cand6` と同等・差 +0.7〜1.7% は判定閾値未満）。より大きいブロック（64×64 系）は 1024〜4096
  のいずれでも `CANDIDATES[3]` の 15〜30% 程度に留まり、異なる K 刻み・wm/wn 配分（`cand5`/`cand6`）も
  4096 で `CANDIDATES[3]` を上回らず、1024/2048 でも閾値を超えて上回ることはない

### 5.3 GPU counters 再試行（#1103 追補）

イシュー #1103 は §5.1 が未計測のまま残した GPU counters（占有率・ALU/メモリ limiter 内訳）の
採取を再試行した。§5.1 は「Metal System Trace」テンプレート**既定**設定でカウンタプロファイルが
無効（`counter-profile=0`）だったことを原因としていたため、`xctrace record` の `--instrument`
オプションで `Metal GPU Counters` instrument を明示追加し、テンプレート既定を上書きする経路を
試した。

```
cargo build -p fandhe-ai-backend-metal --example gemm_counter_workload --release
xcrun xctrace record --template 'Metal System Trace' --instrument 'Metal GPU Counters' \
  --output counters_<N>.trace --launch -- \
  ./target/release/examples/gemm_counter_workload --size=<N> --iters=200
```

（`gemm_counter_workload` は本イシューで新規追加した size 固定・反復ディスパッチ専用 example。
§5.1 の課題「`gemm_f32_prepared_bench` は 512〜4096 を連続実行するため 1 トレースに複数 size が
混在し `<row>` の `ref` 属性解決パーサなしでは紐付けられない」を、1 プロセス実行 = 1 trace = 1 size
という対応にすることで回避する。`crates/backend-metal/examples/gemm_counter_workload.rs`）。

**結果（N=1024/2048/4096 とも同一の失敗パターン。生ログ:
`step8_gpu_counters_{1024,2048,4096}.log`）**:

| size | `xctrace record` の警告 | `--toc` の `counter-profile` | `gpu-counter-value`/`gpu-counter-info` のデータ行 |
|------|--------------------------|-------------------------------|----------------------------------------------------|
| 1024 | `GPU Service reported error: Selected counter profile is not supported on target device` | 3（`Counter Set: Performance Limiters`。§5.1 の 0=無効から前進） | 0 件 |
| 2048 | 同上 | 3（同上） | 0 件 |
| 4096 | 同上 | 3（同上） | 0 件 |

`--instrument 'Metal GPU Counters'` の明示指定は §5.1（既定テンプレートで `counter-profile=0`）
から前進し、カウンタセット（`Performance Limiters`）自体は `counter-profile=3` として活性化する。
しかし `xctrace record` 自体が warning で明示するとおり、**GPU Service がこのカウンタプロファイルを
本デバイス（Apple M4 Max・macOS 26.6.2）で非対応と報告**しており、`gpu-counter-info`（カウンタ名
の列挙）・`gpu-counter-value`（サンプル値）・`metal-gpu-counter-profile`（区間別プロファイル）の
いずれもデータ行が 0 件のままだった。

フォールバックとして代替テンプレート `Game Performance`（N=1024 で試行）も試したが、同一の
warning・同一の空データという結果だった。さらに `Metal GPU Counters` とは別の標準 instrument
`GPU`（`gpu-performance-state-info`/`gpu-performance-state-intervals` を提供）も試したが、
これは GPU の動作状態〈performance state〉の遷移情報のみで、占有率・ALU/メモリ limiter 内訳
（本イシューが求める GPU counters）は提供しない（詳細は `step8_gpu_counters_4096.log` 末尾）。

**判断（受け入れ条件 (a) のフォールバック分岐: フォールバック実施済み・非対話環境では確定不能）**:
GUI（Instruments.app）でのカウンタセット選択操作が必要な可能性が高く、`xctrace` の CLI オプション
には対象カウンタセットを個別指定する手段がない（`xcrun xctrace record --help` にも該当オプション
なし）。非対話セッション（本サブエージェント実行環境）ではこれ以上の解消ができないため、真の
occupancy・ALU/メモリ limiter 内訳は**引き続き未計測**として扱う。§5.2 の proxy 証跡（並列度
proxy・カーネル純境界分離・タイル形状スイープ）を占有度合いの判断材料とする方針は変更しない。
`MTLCounterSampleBuffer` のカーネル内組み込みによる代替採取は引き続き本イシューのスコープ外
（§8 参照）。

## 6. 固定費解消の裏取り（受け入れ条件外・context_cache 前提確認）

```
cargo run -p fandhe-ai-backend-metal --example fixed_overhead_diagnosis --release -- --size=256,512
```

| size | P6 rebuild_each_call（中央値） | P7 reused_dispatch（中央値） | P6-P7 |
|------|-----------------------------------|----------------------------------|-------|
| 256  | 0.2599 ms | 0.2852 ms | -0.0253 ms |
| 512  | 0.4738 ms | 0.5214 ms | -0.0476 ms |

（生ログ: `step6_fixed_overhead.log`）

#927 doc が診断対象とした「都度構築（P6）」の約 5 ms 固定費は、`P6 - P7`（都度構築の追加コスト）が両
サイズとも**負**（-0.03〜-0.05 ms・計測ノイズの範囲内でほぼゼロ）であり、`context_cache`（#930/#948）
導入後は解消済みであることを実測で確認した。#927 doc §5〜§8 のテンプレートは埋めず、本節に前提の違いを
明記して扱う（旧 doc への追補は §7 参照）。

## 7. 頭打ち要因の主要候補・タイル候補の確定（受け入れ条件 (b)）

### 7.1 頭打ち要因の主要候補

以下の実測から、N=1024 以降の頭打ちの要因のうち、**fandhe-ai 自身の系列内では転送
（アップロード＋readback）オーバーヘッドが無視できない寄与を持つ**ことは確定した。ただし
#1103 で追加実施した candle 側の同一境界での転送分離測定（下記「candle 転送分離測定」節）の
結果、**candle 比ギャップの主因を転送オーバーヘッドと確定することはできない**（むしろ candle
側の転送寄与率が fandhe-ai より一貫して大きく、単純な「fandhe-ai 側の転送コストが candle に
劣後する主因」という仮説は実測で支持されなかった。詳細は同節）。占有度合い（並列度 proxy）・
タイル形状のいずれも主因としての説明力が実測で否定された点は #1036 時点から変わらない:

1. **並列度〈concurrency/saturation〉proxy は「不足」を示さない**（§3.3・§5.2）: 1024 以降
   `actual_groups` は `ideal_groups` を常に大きく上回る
2. **タイル形状は現行選択（`CANDIDATES[3]`）が 4096 で最良・1024/2048 で同等**（§4.3・§4.4・§5.2）: 全
   7 候補構成中、64×64 系・TM/TN 拡大系は大きく劣り、異なる K 刻み（`cand5`/`cand6`）も 4096 で
   32×32×16 を上回らない（1024 の `cand5` +0.7%・再実行の 2048 `cand6` +0.8% は判定閾値未満で同等）。タイル形状の変更では頭打ちを
   解消できない
3. **自系列内の境界分離で転送コストの寄与が直接示される**（§4.3）: 同一 `gemm_bench` プロセス内実測で、
   N=4096 のカーネル純境界（転送除き・prepared）は 7.10 TFLOPS、転送込み end-to-end（`dispatch_auto`）は
   4.40 TFLOPS に留まり、同一計測境界チェーン内の差（約 1.62 倍・転送が約 38%）が転送コストの寄与を
   直接示す（§4.2 の別プロセス値 8.29 TFLOPS は参考値に留め根拠に用いない）。
   なお §2 の candle 5,040 GFLOP/s（5.04 TFLOPS）は転送込み end-to-end 値であり、転送除きの
   カーネル純境界と直接比較できない（計測境界が揃っていないため「カーネル純境界は candle を上回る」
   とは主張しない。candle との比較は転送込み同士の §7.2 の注意に従う）

真の occupancy（レジスタ・threadgroup memory 使用率、§5.1・§5.3 で未計測）が別途頭打ちに寄与している
可能性は排除できないが、上記 3 点（proxy 不足なし・タイル形状は現行が最良〜同等・自系列の転送分離で
約 1.62 倍差）は「タイル形状の再選択（#1037/#1039 のスコープ）だけでは頭打ちを解消できない」ことを
強く示唆する。fandhe-ai 自身の end-to-end 経路（facade／`MetalGemm::dispatch_auto`）の転送削減
（既にデバイス常駐化されたバッファの再利用範囲拡大等）は依然として有力な改善候補だが、**candle 比の
ギャップを埋める主因**としての位置づけは下記の実測により後退した。

### 7.1b candle 転送分離測定（#1103 追補・受け入れ条件 1）

`docs/perf/metal-gemm-bottleneck-rediagnosis.md` §7.1（#1036 時点）は「candle 比ギャップの主因は
転送オーバーヘッド」と確定するには candle 側の同一境界での転送分離測定が必要、と留保していた
（§8 の「candle 比ギャップの主因確定に必要な…測定」項目）。#1103 はこれを実施した。

`scripts/bench/framework-compare/bench-candle/src/main.rs` に新タスク `gemm-transfer-split`
（`--device metal` 限定）を追加し、fandhe-ai の `dispatch_auto`（転送込み）／
`dispatch_tiled_prepared`（転送除外）と同一の 2 境界を candle 側でも**同一プロセス内**で計測した:

- **転送込み境界**（`gemm_transfer_incl`）: 計測クロージャ内で毎反復 `Tensor::from_vec`（A・B の
  ホスト→デバイス転送）→ `matmul` → `to_vec2` readback
- **転送除外境界**（`gemm_transfer_excl`）: A・B をループ外で 1 回だけ転送し、計測クロージャ内は
  `matmul` + `Device::synchronize()` のみ（readback は計測窓外で 1 回のみ）

実測（M4 Max・生ログ `step7_candle_transfer_split.log`。fandhe-ai 側は同一セッション内で
`gemm_bench` を再実行した値〈`step7_fandhe_gemm_bench_rerun.log`〉を併記。fandhe-ai の transfer
込みは `dynamic_tile_auto_tflops`、転送除外は `bm32_bn32_bk16_staged` 候補〈`CANDIDATES[3]` と
同一構成〉の同一プロセス内値。N=1024 は `gemm_bench` の候補比較が 2048/4096 のみのため fandhe-ai
側の同一プロセス転送除外値がなく、参考掲載に留める）:

| size | framework | 転送込み TFLOPS | 転送除外 TFLOPS | 除外/込み比 | 転送寄与率 |
|------|-----------|-------------------|-------------------|----------------|--------------|
| 1024 | candle | 0.7062 | 6.6878 | 9.47× | 89.4% |
| 1024 | fandhe-ai | 1.8254 | （同一プロセス値なし。§4.2 参考値 1.7623〈別プロセス〉） | — | — |
| 2048 | candle | 1.3024 | 8.4533 | 6.49× | 84.6% |
| 2048 | fandhe-ai | 3.3005 | 8.8313 | 2.68× | 62.6% |
| 4096 | candle | 3.2046 | 13.1686 | 4.11× | 75.7% |
| 4096 | fandhe-ai | 4.4289 | 8.2179 | 1.86× | 46.1% |

（転送寄与率 = 1 − 転送込み/転送除外。両フレームワークとも「転送を除外すると大幅に速くなる」こと
自体は共通するが、寄与率は N=2048/4096 のいずれも **candle の方が fandhe-ai より高い**〈2048:
84.6% vs 62.6%・4096: 75.7% vs 46.1%〉。これは「fandhe-ai が candle より転送コストの相対負担が
重い」という仮説とは逆方向の結果である）

**判断（S5 の分岐: candle 側の転送寄与率が fandhe-ai と同程度またはそれ以上 → 主因記述を後退させる）**:
candle 自身も転送込み/転送除外の差が大きく（むしろ fandhe-ai より寄与率が高い）、この同一プロセス内
比較からは「fandhe-ai の転送オーバーヘッドが candle に対して相対的に大きい」という主張は支持されない。
むしろ転送除外（カーネル純境界）同士を比較すると、N=2048 では fandhe-ai（8.83 TFLOPS）が candle
（8.45 TFLOPS）をわずかに上回り、N=4096 では candle（13.17 TFLOPS）が fandhe-ai（8.22 TFLOPS）を
上回るという非一貫な結果になっている。したがって**candle 比ギャップの主因を「転送オーバーヘッド」
と確定することはできない**。

ただし N=4096 のこの差（13.17/8.22 ≈ 1.6 倍）について、転送以外の要因（カーネル実装・タイル形状・
MLX steel gemm 側の最適化差等）に起因すると方向性のある結論を出すことは**本追補の計測衛生では
できない**: 本節の数値は fandhe-ai（`gemm_bench` 再実行）と candle（`bench-candle`）という**別プロセス**
間の絶対 TFLOPS 比較であり、§3.4 は同一構成・同一ハーネスでも別プロセス・別バイナリでは中央値が
約 3 倍変動する原因未確定のプロセス間変動を観測している。観測された 1.6 倍の差は §3.4 が示す変動幅
（約 3 倍）に収まる範囲であり、この非一貫な結果が「転送以外の要因による candle 優位」を示すのか
「§3.4 のプロセス間変動」を示すのかを、本追補の範囲では**切り分けられない**。よって「候補: 転送以外の
要因（カーネル実装差等）」への言及は撤回し、「非一貫な結果が観測されたが、§3.4 のプロセス間変動の
範囲内であり要因を特定できない」旨を確定記述とする（要因の切り分けが必要であれば §3.4 のプロセス間
変動の帰属検証〈同一ハーネスでの負荷順序入れ替え反復・温度／クロック記録〉と合わせて別 Issue で扱う。
§8 参照）。

なお本節の数値はいずれも fandhe-ai／candle それぞれの**プロセス内**での転送込み/転送除外比較
（§3.4 の計測契約に従う）であり、fandhe-ai と candle 間の絶対 TFLOPS 比較はプロセス間の参考値に
留める（§3.4 の原因未確定のプロセス間変動が候補として残っているため）。

### 7.2 経路間の比較に関する注意

§2 の 0.4.0 ベースライン（facade `context_cache` 経由の `bench-fandhe` バイナリ。reuse 3.03 TFLOPS）と
§4.3 の `dispatch_auto`（同一プロセス内実測。4.40 TFLOPS）はいずれも「転送込み end-to-end」だが値が
異なる（4.40 vs 3.03 TFLOPS）。両者は計測プロセス・呼び出し経路（`MetalGemm::dispatch_auto` 直接呼び出し
vs facade 経由の `bench-fandhe`）が異なるため単純に差分を「facade 側の追加オーバーヘッド」と結論づける
ことはできない（§3.4 の原因未確定のプロセス間変動も未統制のため）。この経路間ギャップの定量診断は本
イシューのスコープ外とし、§8 の対象外事項に記録する。

### 7.3 #1037/#1039 へ引き渡すタイル候補の確定リスト

| 優先順位 | 候補 | 判断 | 根拠 |
|----------|------|------|------|
| 1（現状維持） | `CANDIDATES[3]`（32×32×16, wm=2, wn=2, staged） | 昇格不要。現行選択を維持 | §4.3・§4.4 で 4096 は最良、1024 は `[5]`/`[6]`・2048 は `[6]`（再実行）と同等（差 +0.7〜1.7% は判定閾値 5% 未満）。同等帯で現行を置き換える根拠はない |
| 保留（要 K 支配的形状での追実測） | `CANDIDATES[5]`（64×32×32, wm=2, wn=2, staged） | 1024 では `[3]` と同等（記載値上 +0.7%）、2048/4096 では `[3]` の 84〜91% に留まる対抗候補。K 方向刻みが大きい（bk=32）ため、K 支配的な非正方形状（縦長・横長）での優位性は未検証 | §4.4 |
| 保留（要 K 支配的形状での追実測） | `CANDIDATES[6]`（64×32×8, wm=4, wn=1, staged） | 初回計測では `[5]` とほぼ同水準（2048/4096 で 84〜91%）だが、再実行では 1024（+1.7%）・2048（+0.8%）で `[3]` をわずかに上回り同等帯。wm=4 の縦分担が K 支配的形状で有利かは未検証 | §4.4 |
| 却下（正方立方 512〜4096 帯） | `CANDIDATES[0]`（64×64×16, wm=2, wn=2, staged） | 全サイズ帯で `[3]` の 15〜30% 程度に留まる。#744 是正の判断（正方立方帯は `[3]` へ一律縮退）を再確認 | §4.3・§4.4 |
| 却下 | `CANDIDATES[4]`（64×64×16, wm=1, wn=2, staged） | `[0]` よりさらに低い（`[3]` の 7〜16%） | §4.4 |
| 却下 | TM/TN 拡大系（128×64・64×128・128×128, bk=16, wm=2, wn=2, staged） | `[3]` の 4〜10% 程度と大幅に劣る | §4.3 |
| 却下 | `bm32_bn32_bk16_direct`（staged=false） | staged 版（`[3]`）の 36〜43% 程度。協調ロードを外すと明確に劣化 | §4.3 |

**#1037/#1039（テーブル駆動タイル選択）への申し送り**: 正方立方 512〜4096 帯では現行 `CANDIDATES[3]`
一律選択（#744 是正）を変更する根拠は本再診断でも得られなかった。テーブル駆動化の主な価値は正方立方
帯以外（縦長・横長・`m == n` かつ `k != m` 等。`tile.rs` 冒頭コメントが「引き続き暫定値のまま」とする
範囲）の閾値確定にあり、その場合の候補集合として `CANDIDATES[5]`／`[6]` を優先追試対象とすることを推奨
する。

## 8. スコープ外（記録のみ）

- タイル選択テーブルの実装変更（#1037/#1038/#1039 のスコープ。本 doc は候補の確定リストを引き渡すのみ）
- `MTLCounterSampleBuffer` の src 組み込みによるカーネル内 GPU counters 採取（§5.1・§5.3 で不採用と
  判断。必要であれば `out-of-scope-tracking.md` に従いユーザー承認のうえ別 Issue で起票）
- §7.2 の経路間ギャップ（`dispatch_auto` 直接呼び出し vs facade `context_cache` 経由）の定量診断
- `xctrace` エクスポートの `<row>` `ref` 属性解決による size 別 GPU 実行時間の機械的な内訳表生成
- §3.4 のプロセス間変動（約 3 倍）の帰属検証（同一ハーネスでの負荷順序入れ替え反復・温度／クロック記録）
- ~~candle 比ギャップの主因確定に必要な、同一プロセス・同一入出力境界で fandhe-ai と candle（MLX steel
  gemm）の転送時間を分離した測定~~ → **#1103 で実施済み**（§7.1b「candle 転送分離測定」節）。
  結果は「主因を転送オーバーヘッドと確定できない」という判断であり、N=4096 のカーネル純境界での
  非一貫な結果（1.6 倍）は §3.4 のプロセス間変動（約 3 倍）の範囲内で要因を特定できない。§3.4 の
  プロセス間変動の帰属検証と合わせた追加調査が必要（未着手・別 Issue 検討）
- ~~§5.1 の GPU counters 未計測（「Metal System Trace」テンプレート既定でカウンタプロファイル無効）~~
  → **#1103 で `--instrument 'Metal GPU Counters'` を追加試行済み**（§5.3）。カウンタプロファイルは
  活性化（`counter-profile=0`→`3`）したが GPU Service が本デバイスで非対応と報告し、実データは
  引き続き未計測。非対話環境ではこれ以上の解消不可と判断（§5.3 参照）
- ~~上記 candle 側カーネル純境界差（N=4096 で candle が fandhe-ai を上回る要因）の追加切り分け
  （カーネル実装・タイル形状・MLX steel gemm 側最適化差等。#1103 では未着手）~~ →
  **#1143 で実施済み**（`docs/perf/metal-gemm-n4096-kernel-gap.md`）。candle が選ぶ
  `TILE_64_64_16_2_2` と同一形状の `CANDIDATES[0]` を `MTLComputePipelineState`
  反射値で調査した結果、スレッドグループレベルの占有率上限には不足がなく（レジスタ圧
  仮説は反射値レベルでは支持されない）、MLX steel classic 経路の未収録構成
  `(32,64,16,1,2)` を追加測定したが劣後（現行 `CANDIDATES[2]` の約 8〜10 分の 1）で
  あったため選択ロジックは変更していない。カーネル codegen 側の変更（ソーステキスト
  特殊化・フラグメントロード方式・協調ロード再構成）は未着手のまま引き続きスコープ外
  （同 doc §5「スコープ外」）
- framework-compare `summary.md` の 0.4.0 正式更新・REQ-8 下限の再確定（人間承認タスク。計画 §6 のとおり
  本イシューのスコープ外）
- #1147 で end-to-end reuse ゲート（#1037「N=1024/2048/4096 reuse で candle 超え」）の正式判定を
  確定した（Apple M4 Max 実機実測。正式系列・参考系列いずれも未達成）。結果と残差は
  `docs/perf/metal-gemm-candle-gate-remeasurement.md` を参照

## 9. 参照

- `docs/perf/metal-gemm-bottleneck-diagnosis.md`（#487。本 doc が是正・引き継ぐ旧診断）
- `docs/perf/metal-fixed-overhead-diagnosis.md`（#927。本 doc §6 が固定費解消を裏取り）
- `docs/perf/metal-tile-select-correction.md`（#744。`tile::select` 現行挙動の確定根拠）
- `crates/backend-metal/src/tile.rs`（`TileConfig`・`CANDIDATES`・`select`）
- `crates/backend-metal/examples/gemm_diagnosis.rs`・`gemm_f32_prepared_bench.rs`・`gemm_bench.rs`・
  `fixed_overhead_diagnosis.rs`・`gemm_tile_sweep.rs`（§4.4 の追加タイル候補スイープ。本診断の計測本体）
- `crates/backend-metal/examples/gemm_counter_workload.rs`（§5.3。#1103 で追加した GPU counters
  採取用の size 固定・反復ディスパッチワークロード）
- `scripts/bench/framework-compare/bench-candle/src/main.rs`（`run_gemm_transfer_split`。§7.1
  「candle 転送分離測定」節。#1103 で追加した `gemm-transfer-split` タスク）
- 親 #1029・トラッキング系譜 #480 → Phase D #530 → D-2 #533／D-7 #541/#542（旧診断の後続タスク）・
  #1036（本 doc の初版）・#1103（本節の追補元イシュー）
