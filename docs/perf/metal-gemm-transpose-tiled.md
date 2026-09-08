# Metal GEMM 転置ロード拡張（`gemm_simdgroup_tiled`）の実装・実測記録

イシュー #1138（`gemm_simdgroup_tiled` の転置ロードを NT/TN/TT パターンへ
拡張しタイル variant 選択を適用する）の実装記録。設計判断・結線判断は本
ドキュメントを正とする。

## 1. 計測環境

- 機種: Apple M4 Max（`sysctl machdep.cpu.brand_string` 実測）
- OS: macOS 26.6.2（`sw_vers` 実測）
- `docs/perf/metal-gemm-tile-table.md` §1 と同一機種

（内部ホスト名は記載しない。`docs/real-hardware-verification-env.md` 方針）

## 2. 実装内容（AC-1〜AC-4）

- `crates/backend-metal/src/shaders/gemm.metal::gemm_simdgroup_tiled` に
  `TRANS_A`/`TRANS_B` function constant（index 9/10。`FINE_BARRIER_ENABLED`
  〈index 8〉の直後）を追加し、staged 協調ロード・direct-load 双方の
  フラグメントロードを `simdgroup_load(..., transpose_matrix=true)` で
  転置対応させた（設計は §3 参照）。
- 新規境界検査ヘルパ 4 関数（`tiled_at_group_in_bounds`／
  `tiled_at_elem_in_bounds`／`tiled_bt_group_in_bounds`／
  `tiled_bt_elem_in_bounds`）を追加し、REQ-8 手動境界チェックを転置ロード
  側でも維持した（既存 5 ヘルパの本体・シグネチャは無変更。
  `tests/shader_source_evidence.rs::gemm_metal_boundary_helpers_retain_req8_condition_expressions`
  が既存 5 関数を厳密固定していることを確認済み）。
- Rust 側: `TileConfig::shared_mem_bytes_for(pattern)`（パターン別
  threadgroup 共有メモリ量）・`crate::gemm::strided_tiled_eligibility`
  （純粋関数の適格性ゲート）・`MetalGemm::dispatch_strided_tiled_prepared`
  （新規公開入口）・`MetalError::StridedTiledIneligible`（新規 variant）を
  追加した。`pipeline_for_tile` のキャッシュキーを
  `(TileConfig, TransposePattern)` へ拡張し、`gemm_simdgroup_tiled` の
  MSL コンパイルをパターンごとに特殊化する。
- `dispatch_strided_tiled_prepared` は `tile::select`／
  `select_with_occupancy` 等が選んだ `TileConfig` を NN だけでなく
  NT/TN/TT でも使う（AC-4「タイル variant 選択の適用」）。

## 3. 設計方針（要約）

threadgroup タイルは「転置後の物理レイアウトのまま」格納し、フラグメント
ロード（`simdgroup_load`）の `transpose_matrix` 引数で転置する
（MLX steel 型の設計。`docs/backend-metal-transpose-collapse-design.md`
§2 の設計継承）。NN（`TRANS_A=false && TRANS_B=false`）では既存の
アドレス式・threadgroup 配置・フラグメントロード順・MMA 発行順が完全に
不変（テキストレベルでも既存の needle 文字列を全て保持）であり、以下の
実機テストでビット同一を確認済み。

## 4. 実機実測（正確性）

`cargo test -p fandhe-ai-backend-metal --release --test gemm_strided_parity -- --ignored --nocapture`
（2026-09-03・本セッション実行）:

```
test dispatch_strided_tiled_prepared_nn_is_bit_identical_to_dispatch_tiled_prepared ... ok
test dispatch_strided_tiled_prepared_matches_cpu_reference_for_all_transpose_patterns ... ok
test dispatch_strided_tiled_prepared_rejects_non_eight_divisible_shape_while_classic_succeeds ... ok
test dispatch_strided_bias_act_prepared_matches_cpu_reference_for_all_transpose_patterns ... ok
test dispatch_bias_act_prepared_nn_is_bit_identical_to_strided_nn ... ok
test gemm_collapsed_lhs_matches_per_batch_cpu_reference ... ok

test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

- NN 非後退（`dispatch_strided_tiled_prepared` の NN 経路が既存
  `dispatch_tiled_prepared` と `assert_eq!` でビット完全一致）を確認した。
- NT/TN/TT parity（`assert_parity`。REQ-2 統一複合判定）を形状
  (64,64,64)・(72,88,104)〈8 整除だがタイル非整除〉・(256,256,256) の
  3 点 × 4 パターンで確認した。
- 適格性ゲート（非 8 整除形状）は `Err(StridedTiledIneligible)` を返し、
  同入力で classic strided 経路（`dispatch_strided_bias_act_prepared`）は
  引き続き成功することを確認した（fail-closed フォールバックの健全性）。
- `cargo test -p fandhe-ai-backend-metal --release --lib -- --ignored --nocapture`
  で `tile::tests::all_tile_candidates_resolve_without_fallback_for_every_transpose_pattern`
  （`CANDIDATES` 全 8 構成 × NN/NT/TN/TT の計 32 通りがサイレントな
  `SINGLE_SIMDGROUP_8X8` へのフォールバックなしにパイプライン構築できる
  ことの確認）も green を確認した。

既存回帰スイート（`gemm_resident_parity.rs`・`gemm_bias_act_parity.rs`・
`gemm_dynamic_tile_parity.rs`・`ops.rs` の
`gemm_resident_lhs_transposed_b_does_not_increment_repack_counter`・
`tile.rs` の `all_tile_candidates_match_cpu_reference_*` 系。全て
`--ignored` 実機テスト）はすべて非後退（green）のまま維持されている
ことを本セッションで確認済み。

## 5. 性能実測（A/B）と結線判断（イシュー #1186）

イシュー #1186 で `crates/backend-metal/examples/
gemm_transpose_route_ab_bench.rs`（`docs/perf/metal-bench-noise-protocol.md`
準拠。`bench_harness::ab::run_ab`。同一プロセス内で A/B 2 クロージャを
ラウンド交互に interleaved 計測する方式——本節旧稿の「2 プロセス」表記は
`run_ab` の実態と異なる誤記のため本節で訂正する）を追加し、A（base=
`MetalGemm::dispatch_strided_bias_act_prepared`。現状の本番経路）と
B（head=`MetalGemm::dispatch_strided_tiled_prepared`。`tile::
select_for_device` が選ぶ構成——`dispatch_auto` の本番既定経路と同一の
選択ロジック）を、`gemm_transpose_tile_sweep.rs::shapes()` と同一の
10 形状 × NT/TN/TT（計 30 セル）で計測を試みた。

`gemm_transpose_route_ab_bench.rs` は#1249/#1251 で `--phase1-only`
モード（フェーズ 1〈安定性セルフチェック〉のみ実行してフェーズ 2 へ
進まず終了する）を追加した:

```sh
cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench --release -- --phase1-only
```

出力にはサイズごとに機械可読な 1 行 `phase1_round_stats`（`grep
'^phase1_round_stats '` で抽出。キー: `size`・`rounds`・`spread`・
`gate`・`within_gate`・`median_secs`・`min_secs`／`min_round_idx`・
`max_secs`／`max_round_idx`〈秒基準・0 始まり。TFLOPS では大小が逆転
するため注意〉・`round_medians_secs`〈カンマ区切り〉）と、末尾に総括
`phase1_summary`（`sizes_gate_exceeded` 一覧）が出る。§5.4 の 4 試行が
示す単発スパイク型の再現条件を排他環境／負荷環境で切り分ける用途
（#1253・#1255）で、1 回ごとに別プロセスで起動する運用を想定する。

### 5.1 計測環境

- 機種・OS: §1 と同一（Apple M4 Max・macOS 26.6.2）
- 実行日: 2026-09-04
- 生ログ・env_info: `docs/perf/logs/metal-gemm-transpose-route-ab-1186/`
  （`route_ab_run1.log` は最終実行の stdout+stderr そのまま——`cargo run`
  のビルド出力〈本体と無関係な `backend-cuda` の未使用コード警告を含む。
  本 PR の変更とは無関係な既存事象〉が先頭に混在する。`env_info.txt` に
  実行前後の `pmset -g therm`／`uptime` を記録）

### 5.2 結果: フェーズ 1（安定性セルフチェック）が不成立・判定不可

`gemm_transpose_route_ab_bench.rs` のフェーズ 1（対照カーネル
`dispatch_auto` を `bench_harness::ab::run_stability` で計測し、
`STABILITY_SPREAD_GATE`〈0.05〉以内かを確認する自己検査。§5 本文の
A/B 計測はこのゲートを全サイズで通過しない限り実行しない設計——安全側
判断: 判定を無効化して中断する方向のみ許す）が、本セッション中の
実行環境では**一度も成立しなかった**。

- `uptime` 実測で load average 3.4〜8.6（同一マシンで並走する他セッション
  の GPU 計測負荷。実装計画 §4 ステップ 5 が事前に想定していた「兄弟
  イシューの GPU 計測が並走しうる」状況が実際に発生した）
- `pmset -g therm` はサーマル警告なし（"No thermal warning level has
  been recorded"）——スロットリングではなく、他プロセスとの GPU リソース
  競合が spread 悪化の要因と考えられる
- ROUNDS/COOLDOWN/MIN_WARMUP を許容される方向（増やす）のみ 3 段階で
  調整して計 4 回計測を試みたが、いずれもいずれかのサイズで gate 超過
  だった（プロトコル §5「調整手順」に従う。判定閾値
  `STABILITY_SPREAD_GATE` 自体は変更していない）:

| 試行 | ROUNDS | COOLDOWN | MIN_WARMUP | gate 超過サイズ（spread） |
|---|---|---|---|---|
| 1 | 6 | 2s | 1s | 256(0.107)・1024(0.066)・2048(0.464) |
| 2 | 10 | 4s | 2s | 256(0.322)・512(0.085)・4096(0.343) |
| 3 | 10 | 4s | 2s（再実行） | 256(0.262)・512(0.135)・1024(1.182)・2048(0.251)・4096(0.362) |
| 4 | 10 | 8s | 3s | 256(0.332)・1024(0.726)・4096(0.059) |

（試行ごとに gate 超過するサイズ・spread が異なる——固定パターンの
バグではなく、実行のたびに変動する外部負荷〈他プロセスの GPU 競合〉が
原因であることを示す。ログは `route_ab_run1.log` が最終試行〈試行 4〉の
出力を保持する）

- 全試行を通じて 1024 前後・4096 で単発の低速ラウンド（サーマル/他
  プロセス起因の一過性スパイクと推定）が spread を押し上げるパターンが
  繰り返し観測された。これは対照カーネル `dispatch_auto` 自体の計測
  であり、B（`dispatch_strided_tiled_prepared`）固有の問題ではない

### 5.3 判定: `undetermined`（判定不可）

**フェーズ 2（30 セルの A/B 本計測）は一度も実行できておらず、
「全形状 × NT/TN/TT で B/A（TFLOPS 比）≥ 1.0」という結線可否の判断基準
（イシュー #1186 本文）を満たすかどうかは実測できていない。**

実装計画の fail-closed 方針（安定性ゲート不成立が解消しなければ
「判定不可」を記録し結線可否を確定しない）に従い、本ドキュメントでは
**`verdict=undetermined`** として記録する。**添付ログ `route_ab_run1.log`
（本節 5.1 の最終試行の生 stdout+stderr）はこの修正前のコードでの実行結果
のため `verdict=` を含まない**——当時の `gemm_transpose_route_ab_bench.rs`
はフェーズ 1 不成立の早期 return で verdict 行を出力せず、フェーズ 2 到達時
とログ形式が非対称だった。この非対称は codex-review 指摘（PR #1198）を受けて
その後のコミットで解消済みであり、**現在の `gemm_transpose_route_ab_bench.rs`
（`crates/backend-metal/examples/gemm_transpose_route_ab_bench.rs:519-530`）は
フェーズ 1 不成立の早期 return でも `println!("verdict=undetermined ...")` を
明示的に出力する**——ログ・添付済みの `route_ab_run1.log` はコード修正前の
実行結果であるためこの出力を含まないが、現在のコードを再実行すれば
`verdict=` grep で判定を機械的に読み取れる。`MetalGemm::
dispatch_strided_bias_act_prepared` への自動ルーティング結線は
（§5 旧稿と同じく）行わない——判定根拠が得られていない以上、性能低下の
可能性がある変更を無根拠に本番経路へ入れない安全側の判断は変わらない。

`MetalGemm::dispatch_strided_tiled_prepared` は明示入口として引き続き
利用可能であり、AC-4（NT/TN/TT へのタイル variant 選択適用）はこの明示
入口で満たされている（§4 の実機正確性実測が根拠）。

## 5.4 再計測（イシュー #1187。4 試行とも `verdict=undetermined`）

イシュー #1187 で §6（旧稿）の引き継ぎ事項に従い、`uptime` の load
average が低いタイミングを選びつつ `gemm_transpose_route_ab_bench.rs`
のフェーズ 2 A/B 本計測を再実行した。**4 試行とも、フェーズ 1（安定性
セルフチェック）が全サイズで成立せず、`verdict=undetermined`（判定不可）
のまま終了した**。30 セルの A/B 本計測は本イシューでも未実行のまま。

### 計測環境・実行日

- 機種・OS: §1 と同一（Apple M4 Max・macOS 26.6.2）
- 実行日: 2026-09-05
- 生ログ・env_info: `docs/perf/logs/metal-gemm-transpose-route-ab-1187/`
  （`route_ab_run1.log`〜`route_ab_run4.log`・`uptime_during_run4.txt`・
  `env_info.txt`）

### 試行表

| 試行 | ROUNDS | COOLDOWN | MIN_WARMUP | 実行直前 uptime load average | gate 超過サイズ数 |
|---|---|---|---|---|---|
| 1 | 10 | 8s | 3s（既定値。#1186 試行 4 と同一） | 4.19 4.93 6.02 | 5/5（256〜4096 全サイズ。spread 最大 1.6356@4096） |
| 2 | 14 | 10s | 4s（増やす方向のみ一時調整。実行中に他プロセス負荷が再上昇した体感はあるが、実行中の `uptime` 生出力は未記録のため具体的な上振れ幅は出典なしの伝聞として扱う） | 5.73 7.54 6.97 | 5/5（spread 最大 1.2495@4096） |
| 3 | 14 | 10s | 4s（試行 2 と同一設定で再試行） | 9.53 10.39 9.61 | 5/5（spread 最大 3.3070@256） |
| 4 | 10 | 8s | 3s（既定値へ戻し再試行。イシュー #1187 継続分） | 1.38 2.41 3.30 | 5/5（256=0.2877・512=0.5298・1024=2.3028・2048=1.6631・4096=0.9392。spread 最大 2.3028@1024） |

（試行 2・3 で使用した ROUNDS=14/COOLDOWN=10s/MIN_WARMUP=4s は実行時の
一時調整であり、`gemm_transpose_route_ab_bench.rs` 本体へは反映していない
——`undetermined` が確定した以上、判定不可のまま定数変更のみをコードへ
残さない安全側の判断。`pmset -g therm` は 4 試行ともサーマル警告なし
「No thermal warning level has been recorded」で、スロットリングではなく
他セッション並走・並行実行タスクによる GPU/CPU リソース競合が spread
悪化の主因と推定される点は #1186 の分析と同じ）

試行 4 は実行直前の load average が 1〜3 台と低く、他セッションの並走
プロセス（`pgrep -fl "cargo|bench-|rustc"`）も検出されなかったため
「並走 GPU 負荷が無い状態」を狙って開始したが、**それでも 5/5 サイズで
gate（spread ≤0.05）を超過**した。実行中 30 秒間隔で `uptime` を記録した
`uptime_during_run4.txt`（前回までの課題「実行中の生出力未記録」を是正）
によると、実行序盤は load average 1.5〜2 台で安定していたが、中盤
（4096 計測時間帯）に 7〜11 台まで上昇している。この上昇は、本実装
エージェント自身が同一マシン上で並行して実行した完了条件検証作業
（`cargo check --workspace`・`cargo clippy --workspace --all-targets
--all-features`・`cargo test --workspace`）が寄与した可能性が高く、
他セッション由来かどうかは切り分けられていない。いずれにせよ、
**低負荷開始でも spread gate を安定して満たせないこと**が試行 4 で
新たに確認された（256 サイズでも spread=0.2877 と gate の約 5.75 倍
超過しており、単に他プロセス負荷だけが原因とは断定できない可能性を
示唆する。詳細は #1187 の後続課題として §6 に記録する）。

### 判定: `verdict=undetermined`（4 試行とも判定不可のまま）

**フェーズ 2（30 セルの A/B 本計測）は本イシューでも一度も実行できて
いない。**「全形状 × NT/TN/TT で B/A（TFLOPS 比）≥ 1.0」という結線可否
の判断基準（イシュー #1186 本文）を満たすかどうかは、引き続き実測できて
いない。実装計画の fail-closed 方針（安定性ゲート不成立が解消しなければ
判定を確定しない）に従い、**`MetalGemm::dispatch_strided_bias_act_prepared`
への自動ルーティング結線は本イシューでも行わない**（判定根拠が得られて
いない以上、性能低下の可能性がある変更を無根拠に本番経路へ入れない安全側
の判断は #1186 から変わらず）。

`MetalGemm::dispatch_strided_tiled_prepared` は引き続き明示入口として
利用可能であり、AC-4 は §4 の実機正確性実測のとおり満たされている。

## 5.5 排他環境での phase 1 spread 分布記録（イシュー #1253）

イシュー #1253 は §6（旧稿）の引き継ぎ事項「実行自体が spread へ与える影響の
切り分け」に向けた前段として、`--phase1-only` モード（#1249/#1251）を
排他環境（実行直前・実行中の load average < 2・他 GPU プロセスなし）で
3 回実行し、サイズ別 spread・単発スパイクの有無を記録することを目的とする。
`STABILITY_SPREAD_GATE` 等の判定閾値・統計量は変更していない。

### attempt 1（TIMEOUT・valid_runs=0）

2026-09-08 19:04〜22:04 JST の 3 時間（elapsed=10828s）、排他ゲートの
成立を待機したが、ログ上 `gate_ok=1` の行は 1 行もなく `consecutive_ok`
も一度も 1 以上にならないまま TIMEOUT した（`docs/perf/logs/
metal-gemm-transpose-route-ab-1242/wait_gate.log`・`orchestrate.log`）。
phase1-only の実測は 1 回も実行できていない。

`wait_gate.log`（ポーリング 335 行）を実際に集計した結果は次のとおり
（集計コマンドは後述。初稿の「最低でも load1 ≈ 3.3 台までしか下がらず」
という記述は誤りだったため本節で訂正する）:

| 項目 | 実測値 |
|------|--------|
| `gate_ok=1` の行数 | 0 / 335 |
| `consecutive_ok` の最大値 | 0 |
| `load1` の最小値 | 1.86（elapsed=3702s・20:06:08 JST。`load5=3.50 util=8% proc_count=7 procs=[python3,] gate_ok=0`） |
| `load1` < 2.0 の行 | 上記と 1.92（elapsed=9396s・21:41:02 JST。`load5=3.19 util=17% proc_count=7 procs=[python3,] gate_ok=0`）の 2 行のみ |
| `load1` < 3.0 の行数 | 38 |
| `load5` の最小値 | 2.87（全行で 2.0 以上） |
| `proc_count` の最小値 | 7（全行で 1 以上。`python3` が全行で検出） |
| `util` の最小値 | 1%（全行で 1% 以上） |

すなわち **load1 が一時的に 2.0 未満（1.86／1.92）へ低下した瞬間は 2 回
あるが、いずれも `gate_ok=0` のまま記録されており**、attempt 1 のゲート
判定条件（閾値・比較対象が load1 か load5 か・連続回数・プロセス条件の
有無）はログからは確定できない。attempt 1 を駆動した実スクリプトは
失われており（commit ccede10 で `wait_gate.sh`／`orchestrate.sh` を
再構成した経緯を同コミットメッセージに明記）、ログの出力形式も再構成版と
一致しない——attempt 1 のログには `util=` 欄があり、elapsed=9494s の行では
`procs=[gemm_transpose_route_ab_bench,python3,]` と当該ベンチバイナリ
自体を検出しているが、再構成版 `wait_gate.sh` は `util=` を出力せず
`pgrep` 対象も `cargo`／`rustc`／`python3` の 3 種のみである（attempt 2 の
`wait_gate_attempt2.log` は再構成版の形式と一致する）。したがって
再構成版の判定条件（`GATE_THRESHOLD=2.0`・load1 比較・`CONSEC_REQUIRED=2`）
を attempt 1 の判定条件と同一とみなす根拠はない。

load1 < 2.0 の 2 行が `gate_ok=0` となった理由として、ログ上の他欄と整合
する仮説は次の 3 つである（いずれも**推測**であり、ログからは確定できない）:

- 仮説 (a): 「他 GPU プロセスなし」条件として `proc_count = 0` も
  要求していた（全行で `proc_count >= 7`・`python3` 常駐のため不成立）。
  イシュー本文のゲート定義（load average < 2 **かつ** 他 GPU プロセス
  なし）とは最も整合する
- 仮説 (b): 比較対象が load1 ではなく load5 だった（load5 の最小値は
  2.87 で全行 2.0 以上）
- 仮説 (c): GPU 使用率 `util = 0%` も条件だった（全行で 1% 以上）

なお elapsed=9494s の行で他セッションが `gemm_transpose_route_ab_bench`
（本 A/B ベンチのバイナリ自体）を実行していたことは、同一 GPU 上での
計測競合が実際に生じていた直接の証跡である。

集計コマンド（Python3 標準ライブラリのみ。リポジトリルートで実行）:

```sh
python3 - <<'EOF'
import re
L="docs/perf/logs/metal-gemm-transpose-route-ab-1242/wait_gate.log"
P=re.compile(r"load1=([\d.]+) load5=([\d.]+) util=(\d+)% proc_count=(\d+) procs=\[([^\]]*)\] gate_ok=(\d) consecutive_ok=(\d+)")
rows=[m.groups() for m in map(P.search, open(L)) if m]
l1=[float(r[0]) for r in rows]; l5=[float(r[1]) for r in rows]
print("rows",len(rows),"gate_ok=1",sum(int(r[5]) for r in rows),"max consec",max(int(r[6]) for r in rows))
print("min load1",min(l1),"min load5",min(l5),"load1<2",sum(v<2 for v in l1),"load1<3",sum(v<3 for v in l1))
print("min proc_count",min(int(r[3]) for r in rows),"min util",min(int(r[2]) for r in rows))
EOF
```

### attempt 2（TIMEOUT・valid_runs=0。有限待機）

attempt 1 が最大 3 時間の無限定待機で TIMEOUT したことを受け、attempt 2
は本実装エージェントのセッション実行時間制約に合わせ待機上限を有限
（`wait_gate.sh` の `MAX_WAIT_SECS=600`・実際の打ち切りは elapsed≈201s
時点）に区切って再試行した。2026-09-09 00:58 JST に開始し、load average
は 4.2〜12.6 台で推移して**一度も 2 未満へ近づく気配を示さず**、
`gate_ok=0`（不成立）が続いた（`wait_gate_attempt2.log`）。他セッションの
`cargo`／`rustc`／`python3` が attempt 1 と同様に検出され続けており、
本 worktree 環境自体が複数イシューの並列実行を常時抱える構造であることを
裏付けている。持続的な非収束トレンドを確認した時点で待機を打ち切り
`TIMEOUT`（`DONE_TIMEOUT_ATTEMPT2`）として記録した——プログラム上の
`MAX_WAIT_SECS` 到達を待たなかったが、判定条件・ゲート閾値そのものは
変更していない（打ち切り理由は `orchestrate.log`・`wait_gate_attempt2.log`
末尾に明記）。attempt 2 でも phase1-only の実測は 0 回のまま。

### 結論（本イシューでの到達点）

**排他環境（load average < 2・他 GPU プロセスなし）の確保に 2 回とも
失敗し、phase1-only の実測（サイズ別 spread・単発スパイクの記録）は
1 件も取得できていない。** 本イシューが動作した worktree 環境自体が、
複数イシューの並列実行セッション（他 worktree の cargo ビルド・python3
集計スクリプト）を常時抱える共有環境であり、attempt 1（3 時間待機）・
attempt 2（有界待機）のいずれも load average が 2 未満へ収束しなかった。
`STABILITY_SPREAD_GATE`・統計量は変更していない。

実装計画の fail-closed 方針（#1187／#1284 の前例と同様）に従い、
本ドキュメントでは AC1／AC2（3 回分の実測ログ・表化）を**未達のまま**
記録する。ゲート閾値を緩めて排他条件を弱めることは行わない——排他環境
での再試行は、他イシューの並列実行が実際に止まる時間帯（`docs/
real-hardware-verification-env.md` の実機予約運用）を確保したうえで
改めて行う必要がある。

## 6. 引き継ぎ事項

- **`gemm_transpose_route_ab_bench.rs` によるフェーズ 2 A/B 本計測の
  再実行**: #1186〜#1187 の 4 試行とも、実行直前の `uptime` load average
  （4〜10 台で変動する試行、1〜3 台と低い試行の双方を含む）に関わらず
  フェーズ 1 の安定性ゲートを一度も通過できていない。試行 4（§5.4）は
  低負荷・並走プロセスなしで開始したにも関わらず 5/5 サイズで gate 超過
  しており、実行中に負荷が再上昇した事実（`uptime_during_run4.txt` 実測）
  はあるものの、それだけでは説明が難しい可能性（256 サイズの spread が
  gate の約 5.75 倍というベースライン自体の高さ）が新たに示唆されている。
  再実行は、他プロセスが一切並走しない時間帯の確保に加え、**ベンチ実行
  自体（コンパイル・ディスパッチオーバーヘッド等）が spread へ与える
  影響の切り分け**も併せて調査したうえで行う必要がある（イシュー #1187
  の後続。まだ結線可否の実測材料が得られていないため引き続き前提条件）。
- `dispatch_strided_bias_act_prepared` への自動ルーティングの結線
  可否判断（性能実測込み）は上記の実測が前提のため持ち越し。
- パターン別タイル選択テーブル（`tile::select` を NT/TN/TT 専用へ拡張
  する要否）の判断も、上記ベンチ実測が前提のため持ち越し。
- `examples/gemm_transpose_tile_sweep.rs` の NT/TN/TT tiled 候補計測
  （タイル variant 別のスイープ。現状は classic strided 固定候補のみ）
  は引き続きスコープ外。
- **排他環境での phase 1 spread 分布記録（§5.5・イシュー #1253）の再試行**:
  2 回とも load average < 2 のゲートに到達できず TIMEOUT した。本 worktree
  環境が複数イシューの並列実行セッションを常時抱える構造上の制約であり、
  他イシューの並走が実際に止まる時間帯を確保しない限り再現性のある排他
  計測は困難である。なお `docs/perf/logs/
  metal-gemm-transpose-route-ab-1242/orchestrate.sh`／`wait_gate.sh` は
  attempt 1 の実行後に再構成したもの（attempt 2 で使用）であり、attempt 1
  の判定条件を再現する保証はない（§5.5 attempt 1 節: attempt 1 のログは
  load1 < 2.0 の行でも `gate_ok=0` であり、出力形式も再構成版と一致しない）。
  再試行時は再構成版をそのまま流用してよいが、実際に用いた判定条件
  （`GATE_THRESHOLD`・比較対象〈load1〉・`CONSEC_REQUIRED`・プロセス条件
  〈`pgrep` 対象と `proc_count` の扱い〉・`MAX_WAIT_SECS`）を `env_info.txt`
  と本ドキュメントへ明示して記録し、attempt 1 と同一条件であるとは
  記述しないこと。
