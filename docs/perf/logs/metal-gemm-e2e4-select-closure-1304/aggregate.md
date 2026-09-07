# イシュー #1304 実機検証結果集計（M4 Max）

`env_info.txt`・`uptime_before_run.txt` 参照（内部ホスト名は含めない）。

## 0. 前提

E2（ソーステキスト特殊化。#1289）・E3（フラグメントロード方式。#1295）・
E4（協調ロードレイアウト。#1300）はいずれも実機実測で**組み込み不可
（REJECT）**と確定済み（`docs/perf/metal-gemm-n4096-kernel-gap.md`
§9.4／§10.4／§11.4）。本イシュー（#1304）は「有効候補 0 件」の受け入れ
条件分岐に従い、`tile::select`・`tile::CANDIDATES`・opt-in 3 定数
（`SOURCE_SPECIALIZATION_ENABLED`／`FRAG_LOAD_CONFIG`／
`COOP_LOAD_CONFIG`）を一切変更せず、本番選択構成の全形状 × NN/NT/TN/TT
parity・既存 bit 一致テスト群を実機で確認した。

## 1. 新規テスト（(a)）: 本番選択構成 × 10 形状 × 4 転置パターン

`cargo test -p fandhe-ai-backend-metal --release --test gemm_strided_parity -- --ignored --nocapture production_select_matches_cpu_reference`

| test | 結果 |
|------|------|
| `production_select_matches_cpu_reference_for_all_shapes_and_transpose_patterns` | ok |
| `production_select_matches_cpu_reference_for_n4096_cubic_shape` | ok |

全 10 形状 × 4 パターン（計 40 ケース）+ 4096³ × 4 パターン（計 4 ケース）
= **44 ケースすべて `assert_parity` pass（fail_count=0）・`resolved == cfg`
（サイレントフォールバックなし）**。

`select_for_device` が形状ごとに返した本番構成（`--nocapture` 診断出力。
`parity_all_shapes_and_n4096.log` 参照）:

| (m, n, k) | 選択構成（`TileConfig`） |
|---|---|
| (512, 512, 512) | `{bm:64, bn:32, bk:32, wm:2, wn:2, staged:true}` |
| (1024, 1024, 1024) | `{bm:64, bn:32, bk:8, wm:4, wn:1, staged:true}` |
| (2048, 2048, 2048) | `{bm:64, bn:32, bk:16, wm:2, wn:2, staged:true}` |
| (2048, 2048, 64) | `{bm:64, bn:64, bk:16, wm:2, wn:2, staged:true}` |
| (2048, 2048, 512) | `{bm:64, bn:64, bk:16, wm:2, wn:2, staged:true}` |
| (1536, 1024, 1024) | `{bm:64, bn:32, bk:16, wm:2, wn:2, staged:true}` |
| (1024, 1536, 1536) | `{bm:64, bn:64, bk:16, wm:2, wn:2, staged:true}` |
| (4096, 1024, 1024) | `{bm:64, bn:32, bk:16, wm:2, wn:2, staged:true}` |
| (1024, 4096, 1024) | `{bm:32, bn:64, bk:16, wm:2, wn:2, staged:true}` |
| (72, 88, 104) | `{bm:32, bn:32, bk:16, wm:2, wn:2, staged:true}` |
| (4096, 4096, 4096) | `{bm:32, bn:64, bk:16, wm:2, wn:2, staged:true}` |

複数の異なる `TileConfig`（境界形状縮退〈(72,88,104)〉を含む）が本番選択
経路で実際に到達し、いずれも 4 転置パターンで正確であることを確認した。

## 2. 既存 E2〜E4 bit 一致自己検証（(b)）

`cargo test -p fandhe-ai-backend-metal --release --lib -- --ignored --nocapture bit_match --skip kernel_gpu --skip reflection`

`bit_match` フィルタは E1・E6 にも一致するため、対象外のフィルタ副産物
として参考実行に含めた（いずれも短時間の bit 一致テストで害はない）。

| test | 対象 | 結果 |
|------|------|------|
| `source_specialized_on_off_bit_match_all_candidates` | E2 | ok |
| `source_specialized_on_off_bit_match_dispatch_auto` | E2 | ok |
| `frag_load_on_off_bit_match_all_candidates` | E3 | ok |
| `frag_load_tgp_vs_device_same_shape_bit_match` | E3 | ok |
| `frag_load_on_off_bit_match_dispatch_auto` | E3 | ok |
| `frag_load_transposed_bit_match` | E3 | ok |
| `coop_load_bit_match_all_candidates` | E4 | ok |
| `coop_load_bit_match_dispatch_auto` | E4 | ok |
| `coop_load_transposed_bit_match` | E4 | ok |
| `coop_load_bit_match_boundary_shape` | E4 | ok |
| `unroll_acc_on_off_bit_match_all_candidates` | E1（参考） | ok |
| `unroll_acc_on_off_bit_match_dispatch_auto` | E1（参考） | ok |
| `tile_class_split_bit_match_all_candidates` | E6（参考） | ok |
| `tile_class_split_bit_match_dispatch_auto` | E6（参考） | ok |
| `tile_class_split_bit_match_edge_shapes` | E6（参考） | ok |

E2〜E4 対象 10 本すべて `ok`。全 15 本 pass（0 fail）。

## 3. 既存 integration bit 一致・parity（(c)）

`cargo test -p fandhe-ai-backend-metal --release --test gemm_swizzle_bit_match --test gemm_fine_barrier_bit_match --test gemm_transposed_parity --test gemm_dynamic_tile_parity --test gemm_strided_parity -- --ignored --nocapture --skip bk32_64x64 --skip bm128 --skip production_select`

E7（`bk32_64x64_*`）・E8（`bm128_*`）・本イシュー新規（`production_select_*`。
(a) で実行済み）を `--skip` で除外（対象外・重複実行回避であり、カバレッジ
削減ではない）。

| test ファイル | 結果 |
|---|---|
| `gemm_dynamic_tile_parity`（11 件） | 全 ok |
| `gemm_fine_barrier_bit_match`（2 件） | 全 ok |
| `gemm_strided_parity`（残り 7 件） | 全 ok |
| `gemm_swizzle_bit_match`（2 件） | 全 ok |
| `gemm_transposed_parity`（5 件） | 全 ok |

合計 27 件すべて pass（0 fail）。

## 4. 既定値ガード（(d)）

`cargo test -p fandhe-ai-backend-metal --release --lib -- tile::tests::source_specialization_enabled_is_false_by_default tile::tests::frag_load_config_default_is_current_path tile::tests::coop_load_config_default_is_current_path`

| test | 結果 |
|------|------|
| `source_specialization_enabled_is_false_by_default` | ok |
| `frag_load_config_default_is_current_path` | ok |
| `coop_load_config_default_is_current_path` | ok |

`SOURCE_SPECIALIZATION_ENABLED=false`・`FRAG_LOAD_CONFIG=DEFAULT`・
`COOP_LOAD_CONFIG=DEFAULT` がコミット状態のまま維持されていることを
実機でも確認。

## 5. 総括

- (a)〜(d) すべて pass（fail 0 件）。本番コード（`tile.rs`・`gemm.rs` の
  実体・`shaders/gemm.metal`）は無変更のまま、本番選択構成の全形状 ×
  転置 4 種 parity・既存 bit 一致群の非後退を実機で確認した
- `resolved != cfg`（サイレントフォールバック）は 1 件も観測されなかった
- 本節が想定していた「fail 発見時は §18 へ発見事項として記録・緩和禁止」
  の分岐には至らなかった
