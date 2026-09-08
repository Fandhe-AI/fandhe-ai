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
cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench --release --features internal-diagnostics -- --phase1-only
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

## 5.5 GPU タイムスタンプ分離計測モードの追加（イシュー #1259）

§5.4 まででフェーズ 1 の単発スパイクが 8 試行（#1186〜#1187）を通じて
一度も安定性ゲートを満たせず、原因が **GPU 実行時間（純カーネル時間）**
に乗るのか **host 側時間**（upload・alloc・encode・commit_wait・readback）
に乗るのかが未切り分けのまま残っていた。本イシューはこの切り分けを行う
ための計測機構を `gemm_transpose_route_ab_bench.rs` へ opt-in
（`--gpu-timestamps`）で追加する（実測・原因切り分け自体は #1261 へ
引き継ぐ。本イシューは機構追加と短時間動作確認に限定）。

### 実行方法

```sh
cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench --release --features internal-diagnostics -- --phase1-only --gpu-timestamps
```

`--phase1-only` と順序不問で併用可。引数なし（既定）の壁時計判定・出力
（`phase1_round_stats`／`phase1_summary`／`verdict=` 等）はバイト単位で
不変——`--gpu-timestamps` はフェーズ 1 の対照ワークロードを `dispatch_auto`
から計装版（`run_stability_gpu_host`）へ**置換**するのみで、追加パスを
走らせて ROUNDS を倍増させることはしない。

### 機構

- `MetalContext::synchronize_with_gpu_timestamps`／`BatchGpuTimestamps`
  （イシュー #1276 で `#[cfg(test)] pub(crate)` として新設）を `pub` へ
  公開化した。本番 `MetalContext::synchronize()` は引き続き no-op
  オブザーバのままで、公開化自体は本番経路の FFI 呼び出し回数・挙動を
  変えない（AC-2 は不変。`crates/backend-metal/src/context.rs`）
- `MetalGemm::encode_tiled_prepared`（`pub`。ラベル `"gemm_tiled_prepared"`）
  を新設し、`dispatch_tiled_prepared`（encode + 即時 synchronize）・
  `diag_encode_tiled_nn`（`#[cfg(test)]` 限定・ラベル
  `"diag_encode_tiled_nn"` 不変）と共通の private ヘルパ
  `encode_tiled_nn_recorded` へ委譲するよう整理した。診断テスト
  （`gemm_reuse_phase_diag_tests.rs`）が assert するラベル契約は不変
  （`crates/backend-metal/src/gemm.rs`）
- **追補（同イシュー #1259。codex-review Medium 指摘対応）**: 上記 2 点の
  `pub` 化は当初「無条件 `pub`」で行ったが、crates.io 公開クレート
  （`fandhe-ai-backend-metal`）の恒久的な公開 API 面へ診断・ベンチ専用の
  内部到達手段（GPU タイムスタンプ収集・encode/synchronize 分離計測用
  エンコード専用入口）を与えてしまう懸念が指摘された。`backend-cuda` の
  `CudaDevice::context`/`stream`（`internal-diagnostics` feature ゲート。
  #1390）と同じ解決パターンを適用し、両 API とも `internal-diagnostics`
  feature（既定 OFF）限定の `pub` とし、既定ビルドでは `pub(crate)` に
  絞った（`crates/backend-metal/Cargo.toml` の feature コメント参照）。
  本 example（`gemm_transpose_route_ab_bench`）自体も `required-features
  = ["internal-diagnostics"]` を要求するよう変更した。CI の `cargo test
  --workspace --all-features`（rust-ci test ジョブ・`make test`）は常に
  この feature を含むため、上記の実行方法・出力契約・テストカバレッジは
  不変。
- `bench_harness::ab::run_stability_observed`（ラウンド完了オブザーバ付き
  変種。`run_stability` はこれを no-op フックで呼ぶ薄いラッパーへ変更）
  を新設し、ラウンド境界（「直前 `measurement.iters` 件が測定対象」）を
  example 側へ通知できるようにした（`crates/bench-harness/src/ab.rs`）
- opt-in 時のワークロードは `dispatch_auto`（`GemmVariant::
  SimdgroupTiled` 分岐）と同一組成（upload → alloc → encode →
  commit_wait → readback）を上記公開 API で再現しつつ、`Instant` による
  host 側フェーズ内訳と `kernel_gpu`（`GPUEndTime − GPUStartTime`）を
  呼び出しごとに記録する。対象サイズ（256〜4096）は全て 8 の倍数のため
  `pad_matrix`/`unpad_matrix` は no-op・`c_buf` 確保は `dispatch_auto` と
  同じ専有確保（`MetalBuffer::new_zeroed`。本 example の
  `MetalContext` はプロセスワイド singleton ではないため
  `alloc_uninit_pooled` は元々 `new_zeroed` へフォールバックする）で、
  既定組成との実質差は (i) バッチラベル、(ii) `encode` の resources 3 本
  retain、(iii) タイムスタンプ取得 2 回、(iv) 入力検証経路
  （`validate_dims` on slice → `validate_prepared_inputs_f32`）のみ
- `wall_minus_gpu`／`commit_wait_minus_gpu` は**サンプルごとに差を
  取ってから**中央値化する（`median(a) − median(b)` ではない。
  `gemm_reuse_phase_diag_tests.rs`〈PR #1371 レビュー教訓〉と同じ理由）
- `MTLCommandBuffer` の不変条件（`batches.len()==1`・タイムスタンプ
  `Some`・`0 ≤ kernel_gpu ≤ commit_wait ≤ wall`）違反は fail-closed で
  当該ラウンドを `valid=false`・関連する差分／spread を `NA` として
  報告し、既定の壁時計判定出力を失わせない

### 出力キー（機械可読）

- 冒頭マーカー: `phase1_workload=gpu_timestamps`（opt-in 時のみ）
- `grep '^phase1_gpu_host_round '`: ラウンド別（`size`・`round`・`iters`・
  `kernel_gpu_median_secs`・`wall_median_secs`・
  `wall_minus_gpu_median_secs`・`commit_wait_median_secs`・
  `commit_wait_minus_gpu_median_secs`・`upload_median_secs`・
  `alloc_median_secs`・`encode_median_secs`・`readback_median_secs`・
  `resolved_cfg`・`valid`）
- `grep '^phase1_gpu_host_stats '`: サイズ別総括（`size`・`rounds`・
  `valid`〈valid=true だったラウンド数〉・`spread_kernel_gpu`・
  `max_round_idx_kernel_gpu`・`spread_wall`・`max_round_idx_wall`・
  `spread_wall_minus_gpu`・`max_round_idx_wall_minus_gpu`・
  `kernel_gpu_round_medians_secs`・`wall_round_medians_secs`。値なしは
  `NA`）

フェーズ 2（30 セルの A/B 本計測）は非計装のまま（本イシューのスコープ
外。§6 参照）。

### 短時間動作確認（M4 Max 実機。ROUNDS=2・COOLDOWN=1s・MIN_WARMUP=1s
への一時ローカル編集・コミットせず revert 済み）

- `--phase1-only --gpu-timestamps` 実行で `phase1_workload=gpu_timestamps`
  マーカー・5 サイズ × 2 ラウンドの `phase1_gpu_host_round` 行・各サイズの
  `phase1_gpu_host_stats` 行を確認（全ラウンド〈5 サイズ × 2 ラウンド〉
  とも `valid=true`・`kernel_gpu ≤ commit_wait ≤ wall` を満たす）
- `--phase1-only`（`--gpu-timestamps` なし）実行で既存出力形式が変化しない
  ことを確認（`phase1_gpu_host_*` 行・`phase1_workload=` マーカーとも
  出力されない）
- 引数エラー系（`--gpu-timestamps --gpu-timestamps`・`--bogus`）が
  `MetalContext::new()` 到達前に fail-closed でエラー終了することを確認
- 実測値自体は共有負荷下の短時間確認のため参考記録に留め、`docs/perf/
  logs/` には残さない（正式な原因切り分け・実測記録は #1261 の担当）

## 5.6 排他環境での phase 1 spread 分布記録（イシュー #1253）

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

**PR #1459 codex-review／Cursor Bugbot 指摘の是正（2026-09-09）**:
両 attempt とも ゲート不通過（`gate_ok=0` 継続）で `cargo run` ループへ
到達しなかったため、本記録の実測値・TIMEOUT 判定そのものへの影響はない
が、`orchestrate.sh`／`wait_gate.sh` 自体に将来の再試行を壊す 2 件の
不備が指摘された。(1) `orchestrate.sh` の `cargo run` が
`gemm_transpose_route_ab_bench` の `required-features =
["internal-diagnostics"]`（`crates/backend-metal/Cargo.toml`）を満たさず
計測開始前に失敗しうる状態だったため、`--features internal-diagnostics`
を明示指定した。(2) `wait_gate.sh` の `gate_ok` 判定が
`proc_count`（cargo/rustc/python3 の有無）を記録するのみで判定式へ
反映しておらず、load average のみで PASSED 判定していたため、
`proc_count == 0` を必須条件へ追加した。加えて `orchestrate.sh` の
3 回の run 実行中は排他計測契約を実行前後の静的確認・終了コードのみで
判定していたため、run 実行中もバックグラウンドで load average・他
GPU/build 系プロセス（自 run のプロセスツリーは除外）をポーリング監視
し、逸脱を検出した run は終了コードに関わらず `valid_runs` から除外する
よう変更した（`phase1_run${n}_monitor.log` に記録）。修正後スクリプトは
`docs/perf/logs/metal-gemm-transpose-route-ab-1242/orchestrate.sh`／
`wait_gate.sh` を正とする。

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

## 5.7 排他環境での分離計測・原因候補 (a)(b)(c) の切り分け（イシュー #1261）

§5.5 が追加した `--gpu-timestamps` 分離計測モードを用い、排他環境
（load average < 2.0・他 GPU プロセスなし）で正式な原因切り分け計測を
実施した（判定規則は計測前に事前宣言し、計測後に緩めていない
——`docs/perf/logs/metal-gemm-transpose-route-ab-1242/1261-aggregate.py`
の docstring 参照）。

### 排他環境ゲート・実行環境

- 機種・OS: §1 と同一（Apple M4 Max・macOS 26.6.2）
- 実行日: 2026-09-08
- ゲート条件: 1 分 load average < 2.0 を 60 秒間隔で 2 回連続。最大 30 回
  （約 30 分）待機
- 生ログ・env_info: `docs/perf/logs/metal-gemm-transpose-route-ab-1242/`
  （`1261-` プレフィックス。#1253/#1255 と同一ディレクトリを共有するが
  ファイル名衝突なし）

### ゲート結果: 30 回試行すべて不通過（`gate_not_passed`）

排他環境ゲートは**30 回の試行すべてで不通過**だった（生ログ
`docs/perf/logs/metal-gemm-transpose-route-ab-1242/1261-gate.log`・
`1261-env_info.txt`）。

- 1 分 load average は 30 サンプル全てで 2.0 を上回った（最小 2.40・
  最大 6.92・平均 4.28。`count_below_2.0=0`）
- GPU プロセス検出（watchlist: `cargo|rustc|bench|gemm_|python|torch|mlx`）
  は 30 試行すべてで `proc_count=0` と記録されていたが、**この値は
  「該当プロセスなし」の根拠にならない（訂正。codex-review 指摘・
  PR #1457）**: 当時の `1261-orchestrate.sh` は `pgrep -fl -E "${GPU_WATCH_PATTERN}"`
  を呼んでいたところ、macOS（BSD）の `pgrep` に `-E` オプションは存在
  せず毎回コマンド自体が失敗していた（`pgrep` は既定で拡張正規表現を
  解釈するため `-E` は不要かつ無効な引数）。その失敗（非ゼロ終了・空
  stdout）が後続の `| grep -v ...` パイプラインに吸収され、実際には
  他 GPU プロセスの有無を一切検査できないまま機械的に `proc_count=0`
  が記録されていた。ただしこの回の**ゲート判定結果自体（30 回とも
  不通過）は本バグの影響を受けない**——`ok=1` は load average 条件
  （30 サンプル全てで 2.0 超）と `proc_count==0` 条件の両方を要求する
  AND 条件であり、load average 側が単独で全試行を不通過にしているため、
  `pgrep` 側の検査可否に関わらず判定結果は変わらない。`pgrep` の
  `-E` 除去・失敗検知（`pgrep_check_status`）は PR #1457 で是正済みで
  あり、以降の計測では GPU プロセス側の検査も正しく機能する
- セッションが「複数のイシューが並列に実行されるワークフロー運用下」
  にあるという実行前提（本イシューの起動プロンプトが明示）と整合する
  結果であり、実装計画（§4 ステップ 5・§1）が事前に想定していたリスク
  がそのまま発生した

計画 §3.4「排他ゲート不通過のまま終了した場合: 計測せず
`undetermined`」に従い、`1261-orchestrate.sh` は `1261-GATE_NOT_PASSED.marker`
を出力して**計測本体（E1〜E3・W1〜W2）を一切実行せずに終了**した
（fail-closed。共有負荷下の値を「排他環境の結果」として提示しない）。

### 計測マトリクス（計画。未実行）

| # | 引数 | 目的 |
|---|---|---|
| E1〜E3 | `--phase1-only --gpu-timestamps` | 排他環境 3 回の分離計測（正式） |
| W1〜W2 | `--phase1-only --gpu-timestamps --min-warmup-secs=9` | 原因候補 (b) 検証: MIN_WARMUP を 3 倍に増やす方向のみの追加試行 |

ゲート不通過のため、上記いずれも実行されていない。

### スパイク所在の判定結果

**評価対象なし**——排他環境が確保できず E1〜E3／W1〜W2 のいずれも実行
していないため、`phase1_gpu_host_round`／`phase1_gpu_host_stats` の生データ
自体が存在しない（`1261-aggregate.py` の判定規則を適用できるログが
0 件）。

### 原因候補 (a)(b)(c) の判定

計画 §3.4「排他環境でスパイクが消えた場合: (a)(b)(c) は排他環境では
発生せず評価対象なし」の対偶——**排他環境そのものが確保できなかった**
ため、(a) コマンドバッファのスケジューリング・(b) ウォームアップ不足・
(c) pmset 電源状態のいずれについても、支持・不支持を判定する材料が
得られていない。3 候補とも **「排他環境が確保できず評価対象なし」**
（判定不能。§5.2〜§5.4 の共有負荷下での undetermined と同じ結論だが、
今回は分離計測データすら得られていない点でさらに手前の段階）。

### Phase 2（#1246: #1263〜#1266）への反映事項

- **#1264（実行前ガード）**: 本イシューで実測した閾値（load average(1 分)
  < 2.0・60 秒間隔で 2 回連続）は、本セッションの実行環境（load average
  2.4〜6.9 台で持続）に対しては**一度も成立しなかった**——閾値自体が
  厳しすぎるのではなく、この種の並列ワークフロー運用下では「他プロセス
  が一切並走しない時間帯」が実質的に存在しないことを示す実測事実として
  記録する（`crates/bench-harness::env_guard`〈#1264 で実装済みの API 層〉
  の `EnvGuardConfig::max_load_avg_1min` へ同じ閾値〈2.0〉を設定した場合、
  同様に `Fail` が持続する見込み）。GPU プロセス watchlist
  （`cargo|rustc|bench|gemm_|python|torch|mlx`）・`ioreg` の
  `Device Utilization %` 手法は #1264 の env_guard 実装と重複するため、
  #1265 の結線時は本イシュー独自の bash 実装ではなく env_guard 側を
  正として使うべき（本イシューの `1261-orchestrate.sh` は #1264/#1265 の
  スコープと衝突しない一回限りの計測実行用として意図的に独立実装した
  ——イシュー #1261 実装計画 §0 参照）
- **#1265（バックオフ再試行・env_info 自動記録）**: 待機間隔 60 秒・
  上限 30 回（合計約 30 分）の実績を得た。持続的な高負荷環境では
  30 回の再試行では不十分な可能性があり、#1265 は (i) より長い待機上限
  を選択可能にする、(ii) ゲート不通過が一定回数続いた場合に「低負荷
  時間帯を後から探す」運用（例: cron 的な定期リトライ）を検討する、の
  いずれかを設計判断として持ち越すべき。env_info へ自動記録すべき項目
  として、実行前後の `uptime`・`pmset -g`／`pmset -g therm`／
  `pmset -g batt`・GPU 使用率・実効 `ROUNDS`/`COOLDOWN`/`MIN_WARMUP`
  に加え、**ゲート開始から終了までの経過秒数**（本イシューの
  `1261-gate.log` には `gate_start_unix` はあるが終了時刻がなく、
  経過時間を正確に算出できなかった。§ env_info「実行環境の背景」参照）
  を追加すべきと分かった
- **#1266（ロバスト統計の設計提案）**: 分離計測データ自体が得られな
  かったため、E1〜E3 のラウンド別中央値を使ったロバスト統計案の実データ
  検証は持ち越し。合成ログによる `1261-aggregate.py --self-test` の
  判定ロジック自体（スパイク所在判定・(b) 先頭偏在判定）は実装・自己
  テスト済みであり、実データが得られた際にそのまま適用できる
- **#1263 全体**: 今回の実測は「排他環境ゲート自体が、本セッションの
  実行環境では 30 分間一度も通過しない」という重い制約を明らかにした。
  Phase 2 配下のイシューは、この制約を前提に「排他環境の確保」を
  待つのではなく、共有負荷下でも解釈可能な統計手法（#1266）や、
  低負荷時間帯を検出して自動的にリトライするスケジューリング（#1265）
  の優先度を上げて設計することを推奨する
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
- **排他環境での phase 1 spread 分布記録（§5.6・イシュー #1253）の再試行**:
  2 回とも load average < 2 のゲートに到達できず TIMEOUT した。本 worktree
  環境が複数イシューの並列実行セッションを常時抱える構造上の制約であり、
  他イシューの並走が実際に止まる時間帯を確保しない限り再現性のある排他
  計測は困難である。なお `docs/perf/logs/
  metal-gemm-transpose-route-ab-1242/orchestrate.sh`／`wait_gate.sh` は
  attempt 1 の実行後に再構成したもの（attempt 2 で使用）であり、attempt 1
  の判定条件を再現する保証はない（§5.6 attempt 1 節: attempt 1 のログは
  load1 < 2.0 の行でも `gate_ok=0` であり、出力形式も再構成版と一致しない）。
  再試行時は再構成版をそのまま流用してよいが、実際に用いた判定条件
  （`GATE_THRESHOLD`・比較対象〈load1〉・`CONSEC_REQUIRED`・プロセス条件
  〈`pgrep` 対象と `proc_count` の扱い〉・`MAX_WAIT_SECS`）を `env_info.txt`
  と本ドキュメントへ明示して記録し、attempt 1 と同一条件であるとは
  記述しないこと。
- **#1261（排他環境での分離計測）は「排他環境ゲートが 30 回試行すべて
  不通過」という結果に終わり、分離計測データ自体が得られなかった
  （§5.7）。フェーズ 2 A/B 本計測の再実行に必要な「他プロセスが一切
  並走しない時間帯」は、本リポジトリの並列ワークフロー運用下では
  実質的に確保が困難であることが 30 分間の実測で裏付けられた。
  §5.7「Phase 2 への反映事項」で #1263〜#1266 への引き継ぎ事項を整理
  済み——とくに #1266（共有負荷下でも解釈可能なロバスト統計）の優先度
  を上げることを推奨する。