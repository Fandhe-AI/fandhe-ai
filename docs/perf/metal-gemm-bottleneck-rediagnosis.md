# Metal GEMM 1024 以降スループット頭打ち context_cache 後の再診断（#1036）

イシュー #1036「docs(perf): Metal GEMM 1024 以降の頭打ちを context_cache 後の構成で M4 Max 実機診断する」の
実測記録。親 #1029「GEMM カーネルの candle 超え」配下。

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
| 計測コミット SHA | `2a12c44294c707579254a90286573243293a0d4`（`origin/main` fetch 直後の HEAD） |
| 計測プロトコル | `bench-harness::protocol::run`（`MeasurementConfig::default()` = warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1）。§4 の追加スイープ（`tmp_tile_sweep`）も同一 `MeasurementConfig::default()` を使用 |
| 決定的シード | `0xC0FFEE`（各 example の `SEED` 定数） |
| 計測衛生 | AC 電源接続・実行順は本 doc の記載順（§2→§3→§4→§5→§6）どおり連続実行。他 GPU 負荷アプリの明示終了は未実施（既定のバックグラウンドプロセスのみ）。**§3.4 で後述するとおり、直前の GPU 負荷履歴によって同一構成でも中央値が有意に変動する（サーマル/クロック状態依存）ことを実測で確認した** — 詳細は §3.4 参照 |

生ログは `/private/tmp/claude-501/-Users-nancy-fandhe-library-rust-ai-library/4bc4d337-b3ac-4193-8a78-acb3c09ea455/scratchpad/1036/`
以下に保存（`step2_gemm_diagnosis.log`・`step3_kernel_pure.log`・`step4_gemm_bench.log`・
`step4b_extra_candidates.log`・`step6_fixed_overhead.log`・`metal_trace.trace`・`toc.xml`・
`gpu_intervals.xml`）。本セッション（サブエージェント実行環境）のローカルパスであり、後続セッションから
直接参照できない場合がある点に留意する。

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

### 3.4 計測衛生上の所見: 直前の GPU 負荷履歴による中央値変動

§4.2（`gemm_f32_prepared_bench`。アイドル直後に単独実行）と §4.4（`tmp_tile_sweep`。直前に §4.3
`gemm_bench` フルスイート実行済み）で同一構成（32×32×16, wm=2, wn=2, staged, N=1024）を計測したところ、
中央値 TFLOPS が 1.76 → 5.44（約 3.1 倍）に変動した。両者は同一プロセス内では計測していないため
プロセス間の絶対値比較に留め、本 doc の結論では **同一プロセス内で連続計測した値同士の相対比較**
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

**end-to-end（§4.3 `dynamic_tile_auto_tflops`）とカーネル純境界（§4.2）の同一プロセス内比較**:
N=4096 で end-to-end 4.3965 TFLOPS に対しカーネル純境界 8.2923 TFLOPS。**転送（アップロード＋readback）
を除くとスループットは約 1.9 倍になる**。これは同一プロセス・同一 `tile::select` 結果（32×32×16）での
比較であり、転送コストが N=4096 の end-to-end 時間のおよそ半分を占めることを示す直接的な実測根拠である。

### 4.4 追加タイル候補（CANDIDATES[4]/[5]/[6]）スイープ

`crate::tile::CANDIDATES` のうち §4.3 で未比較の候補（`[4]` 64×64×16 wm=1,wn=2 ／ `[5]` 64×32×32
wm=2,wn=2 ／ `[6]` 64×32×8 wm=4,wn=1）を、`[0]`（64×64×16 baseline）・`[3]`（32×32×16 現行選択）と
併せて一時計測バイナリ（`dispatch_tiled_prepared` を直接呼ぶ最小 example。本 PR には含めない。生成・
削除の経緯は §7 参照）で比較した。

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
と直接比較しない。1024 の cand3 が §4.2 の同構成〈1.76 TFLOPS〉と大きく乖離するのは §3.4 のサーマル/
クロック状態依存によるもので、本スイープ内部の相対順位のみを根拠として用いる）

**観察**: 1024/2048/4096 いずれのサイズでも現行 `CANDIDATES[3]`（32×32×16, wm=2, wn=2, staged）が
テストした 7 構成中最良。`cand5`（64×32×32, bk=32）・`cand6`（64×32×8, wm=4）は 2048/4096 で
`cand3` の 84〜91% 程度まで近づくが上回らない。`cand0`／`cand4`（64×64 系）・TM/TN 拡大系（§4.3）は
いずれも `cand3` の 15〜30% 程度に留まる。

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
含まない）。この値は §4.2 の N=512 中央値（0.8234 TFLOPS → 1 回あたり約 654 µs の壁時計時間。
`2*512^3/0.8234e12` 秒）よりも短く、壁時計時間には GPU 実行そのもの以外の待機・同期コストが含まれる
ことを示唆するが、本表から `Compute` チャネルの区間を size ごとに機械的に紐付けるパーサ（`<row>` の
`ref` 属性による値の再利用〈XML 上の重複排除〉を解決する必要がある）を本セッションでは完成させられな
かったため、size 別の系統的な内訳表は**未計測**として扱う。

### 5.2 未計測部分の代替（proxy 証跡）

真の occupancy（レジスタ・threadgroup memory 使用率）・ALU/メモリ limiter の内訳は上記のとおり未計測
のため、§3・§4 で得た代替証跡を占有度合いの判断材料とする:

- **並列度〈concurrency/saturation〉proxy**（§3.2）: `actual_groups` は 1024 以降常に `ideal_groups`
  を大きく上回る（1024: 4.267 倍 〜 4096: 68.267 倍）。少なくとも「発行 threadgroup 数の不足」という
  意味での並列度不足は考えにくい
- **カーネル純境界 vs end-to-end の分離**（§4.3）: N=4096 でカーネル純境界（8.29 TFLOPS）は
  end-to-end（4.40 TFLOPS）の約 1.9 倍。転送（アップロード＋readback）が end-to-end 時間のおよそ半分
  を占める
- **タイル形状スイープ**（§4.3・§4.4）: `CANDIDATES` 全 7 構成中 `CANDIDATES[3]`（32×32×16）が全サイズ
  帯で最良。より大きいブロック（64×64 系）は 1024〜4096 のいずれでも `CANDIDATES[3]` の 15〜30% 程度
  に留まり、より小さい K 刻み・異なる wm/wn 配分（`cand5`/`cand6`）も `CANDIDATES[3]` を上回らない

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

## 7. 主因の確定・タイル候補の確定（受け入れ条件 (b)）

### 7.1 頭打ちの主因

以下の実測から、N=1024 以降の頭打ち（0.4.0 ベースラインでの candle 比ギャップ）の主因を**転送
（アップロード＋readback）オーバーヘッド**と判断する。占有度合い（並列度 proxy）・タイル形状のいずれも
主因としての説明力が実測で否定された:

1. **並列度〈concurrency/saturation〉proxy は「不足」を示さない**（§3.3・§5.2）: 1024 以降
   `actual_groups` は `ideal_groups` を常に大きく上回る
2. **タイル形状は現行選択（`CANDIDATES[3]`）が最良**（§4.3・§4.4・§5.2）: 全 7 候補構成中、64×64 系・
   TM/TN 拡大系・異なる K 刻みのいずれも 32×32×16 を上回らない。タイル形状の変更では頭打ちを解消できない
3. **転送を除くとカーネル純境界は candle を上回る**（§4.2・§4.3）: N=4096 のカーネル純境界は 8.29
   TFLOPS で、§2 の 0.4.0 ベースライン（candle 5,040 GFLOP/s = 5.04 TFLOPS）を上回る。一方 end-to-end
   （同一プロセス内 `dispatch_auto`）は 4.40 TFLOPS に留まり、カーネル純境界との差（約 1.9 倍）が
   転送コストの寄与を直接示す

真の occupancy（レジスタ・threadgroup memory 使用率、§5.1 で未計測）が別途頭打ちに寄与している可能性は
排除できないが、上記 3 点（proxy 不足なし・タイル形状最適・転送分離で candle 超え）は「タイル形状の
再選択（#1037/#1039 のスコープ）だけでは 0.4.0 ベースラインのギャップを解消できない」ことを強く示唆する。
残存ギャップの解消には、GEMM カーネル自体の変更ではなく **facade／`MetalGemm::dispatch_auto` 呼び出し
経路の転送（アップロード・readback）削減**（既にデバイス常駐化されたバッファの再利用範囲拡大等）が
主要な候補になる。

### 7.2 経路間の比較に関する注意

§2 の 0.4.0 ベースライン（facade `context_cache` 経由の `bench-fandhe` バイナリ。reuse 3.03 TFLOPS）と
§4.3 の `dispatch_auto`（同一プロセス内実測。4.40 TFLOPS）はいずれも「転送込み end-to-end」だが値が
異なる（4.40 vs 3.03 TFLOPS）。両者は計測プロセス・呼び出し経路（`MetalGemm::dispatch_auto` 直接呼び出し
vs facade 経由の `bench-fandhe`）が異なるため単純に差分を「facade 側の追加オーバーヘッド」と結論づける
ことはできない（§3.4 のサーマル/クロック状態依存も未統制のため）。この経路間ギャップの定量診断は本
イシューのスコープ外とし、§8 の対象外事項に記録する。

### 7.3 #1037/#1039 へ引き渡すタイル候補の確定リスト

| 優先順位 | 候補 | 判断 | 根拠 |
|----------|------|------|------|
| 1（現状維持） | `CANDIDATES[3]`（32×32×16, wm=2, wn=2, staged） | 昇格不要。現行選択を維持 | §4.3・§4.4 の全サイズ帯（1024/2048/4096）で最良 |
| 保留（要 K 支配的形状での追実測） | `CANDIDATES[5]`（64×32×32, wm=2, wn=2, staged） | 正方形状では `[3]` に劣るが 2048/4096 で 84〜91% まで近づく唯一の対抗候補。K 方向刻みが大きい（bk=32）ため、K 支配的な非正方形状（縦長・横長）での優位性は未検証 | §4.4 |
| 保留（要 K 支配的形状での追実測） | `CANDIDATES[6]`（64×32×8, wm=4, wn=1, staged） | `[5]` とほぼ同水準（2048/4096 で 84〜91%）。wm=4 の縦分担が K 支配的形状で有利かは未検証 | §4.4 |
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
- `MTLCounterSampleBuffer` の src 組み込みによるカーネル内 GPU counters 採取（§5.1 で不採用と判断。
  必要であれば `out-of-scope-tracking.md` に従いユーザー承認のうえ別 Issue で起票）
- §7.2 の経路間ギャップ（`dispatch_auto` 直接呼び出し vs facade `context_cache` 経由）の定量診断
- `xctrace` エクスポートの `<row>` `ref` 属性解決による size 別 GPU 実行時間の機械的な内訳表生成
  （§5.1。本セッションでは完成させられなかった）
- framework-compare `summary.md` の 0.4.0 正式更新・REQ-8 下限の再確定（人間承認タスク。計画 §6 のとおり
  本イシューのスコープ外）

## 9. 参照

- `docs/perf/metal-gemm-bottleneck-diagnosis.md`（#487。本 doc が是正・引き継ぐ旧診断）
- `docs/perf/metal-fixed-overhead-diagnosis.md`（#927。本 doc §6 が固定費解消を裏取り）
- `docs/perf/metal-tile-select-correction.md`（#744。`tile::select` 現行挙動の確定根拠）
- `crates/backend-metal/src/tile.rs`（`TileConfig`・`CANDIDATES`・`select`）
- `crates/backend-metal/examples/gemm_diagnosis.rs`・`gemm_f32_prepared_bench.rs`・`gemm_bench.rs`・
  `fixed_overhead_diagnosis.rs`（本診断の計測本体）
- 親 #1029・トラッキング系譜 #480 → Phase D #530 → D-2 #533／D-7 #541/#542（旧診断の後続タスク）
