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

## 5. 性能実測（ベンチマーク A/B）と結線判断

**本セッションでは `docs/perf/metal-bench-noise-protocol.md` 準拠の
2 プロセス interleaved A/B 計測（`bench_harness::ab::run_ab`。6 ラウンド・
安定性ゲート・サーマル記録付き）を実施していない。** 実装・正確性検証
（実機テスト全 green）を優先したため、性能比較（classic strided
`gemm_tiled_bias_act` vs 本イシューの tiled 転置ロード経路）の
厳密な計測は後続セッションへ持ち越す。

この状態を踏まえ、**`MetalGemm::dispatch_strided_bias_act_prepared` への
自動ルーティング（bias/act 無し・適格な入力を無条件に
`dispatch_strided_tiled_prepared` へ委譲する変更）は本 PR では行わない**
（未実装のまま）。実装計画 §4 ステップ 8 の判断基準（「全形状 ×
NT/TN/TT で B/A（TFLOPS 比）が 1.0 以上」を実測で確認できた場合のみ
結線する）を満たす実測根拠がないため、性能低下の可能性がある変更を
無根拠に本番経路へ入れない（安全側の判断）。

`MetalGemm::dispatch_strided_tiled_prepared` は明示入口として常に
利用可能であり、AC-4（NT/TN/TT へのタイル variant 選択適用）はこの明示
入口で満たされる。自動ルーティングの可否判断（性能実測込み）は別イシュー
で引き継ぐ。

## 6. 引き継ぎ事項

- `docs/perf/logs/metal-gemm-transpose-tiled-1138/` への生ログ配置・
  `examples/gemm_transpose_tile_sweep.rs` の NT/TN/TT tiled 候補計測・
  `examples/gemm_transpose_route_ab_bench.rs`（結線前後 A/B）の追加・
  実行は未実施（§5 参照）。
- `dispatch_strided_bias_act_prepared` への自動ルーティングの可否判断
  （性能実測が前提）。
- パターン別タイル選択テーブル（`tile::select` を NT/TN/TT 専用へ拡張
  する要否）の判断も、上記ベンチ実測が前提のため持ち越し。
