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
**`verdict=undetermined`** として記録する（`gemm_transpose_route_ab_bench.rs`
自体はフェーズ 2 の `verdict=` 行を出力していない——フェーズ 1 で中断した
ため到達していない。この呼称は本ドキュメントの判定表記であり、
`route_ab_run1.log` を検索してもヒットしない）。`MetalGemm::
dispatch_strided_bias_act_prepared` への自動ルーティング結線は
（§5 旧稿と同じく）行わない——判定根拠が得られていない以上、性能低下の
可能性がある変更を無根拠に本番経路へ入れない安全側の判断は変わらない。

`MetalGemm::dispatch_strided_tiled_prepared` は明示入口として引き続き
利用可能であり、AC-4（NT/TN/TT へのタイル variant 選択適用）はこの明示
入口で満たされている（§4 の実機正確性実測が根拠）。

## 6. 引き継ぎ事項

- **`gemm_transpose_route_ab_bench.rs` によるフェーズ 2 A/B 本計測の
  再実行**: 他セッションの GPU 計測負荷が小さい時間帯（`uptime` の
  load average が低いタイミング）を選んで再実行し、フェーズ 1 の
  安定性ゲートを通過させたうえで 30 セルの実測・`verdict` 判定を得る
  必要がある（イシュー #1187 の前提条件）。
- `dispatch_strided_bias_act_prepared` への自動ルーティングの結線
  可否判断（性能実測込み。イシュー #1187）は上記の実測が前提のため
  持ち越し。
- パターン別タイル選択テーブル（`tile::select` を NT/TN/TT 専用へ拡張
  する要否）の判断も、上記ベンチ実測が前提のため持ち越し。
- `examples/gemm_transpose_tile_sweep.rs` の NT/TN/TT tiled 候補計測
  （タイル variant 別のスイープ。現状は classic strided 固定候補のみ）
  は引き続きスコープ外。
