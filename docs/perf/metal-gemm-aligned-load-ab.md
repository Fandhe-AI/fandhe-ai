# Metal GEMM アラインメント特化ロード分岐 A/B 計測記録（#752）

イシュー #752「perf(backend-metal): アラインメント特化ロード分岐（align_M/N/K function constant 方式）」の
A/B 計測手順・記録テンプレート。`gemm_simdgroup_tiled` へ `ALIGN_M`/`ALIGN_N`/`ALIGN_K` の 3 つの MSL
function constant を追加し、実効次元が `TileConfig` のブロック幅（`bm`/`bn`/`bk`）へ厳密に整除することが
ホスト側（`crate::tile::AlignFlags::for_dims`）で証明できた場合に、staged 協調ロード・direct-load 経路双方の
境界チェックをコンパイル時定数畳み込みで恒真化しデッドコード除去する（MLX steel `matmul.cpp` の
align_M/N/K を参考にした技法。技法の参照のみでコード転記はしない）。

## 状態: 未計測。実機セッションで消化

本ファイルは Linux worktree で作成され、Metal 実機（Apple Silicon）が同一セッションで使用できないため
計測手順・記録テンプレートのみを整備した状態。`crates/backend-metal/tests/shader_source_evidence.rs` の
`gemm_metal_source_declares_align_function_constants`・
`gemm_simdgroup_tiled_source_uses_align_flags_in_staged_guards`・
`gemm_simdgroup_tiled_source_uses_align_flags_in_direct_load_guards` により function constant 宣言・
OR 合成ガードの実在・検査式の残存（REQ-8）は Linux CI（ubuntu-latest）上で機械検査済みだが、数値一致の
実測（恒真化がビット単位一致を崩していないことの検証）・性能効果の実測・採否判断（下記「判断基準」）は
Mac 実機セッションでの後続対応が必要。

## 計測手順（Apple Silicon 実機）

base（変更前）と head（変更後）それぞれについて計測し、5 回計測の中央値 TFLOPS を比較する
（`bench-harness::protocol::run` が中央値計測を実装済み。`coding-rust.md` 準拠。手順自体は
`docs/real-hardware-verification-env.md` の接続・転送手順に従う。実ホスト名はローカル管理外ファイル参照）。

```sh
git fetch origin

# base（変更前。align_M/N/K 導入前の直近コミット）
git checkout <base-sha>
cargo run -p backend-metal --example gemm_bench --release > /tmp/gemm_bench_base.txt
cargo run -p backend-metal --example gemm_f32_prepared_bench --release > /tmp/gemm_f32_prepared_bench_base.txt

# head（本イシューの実装ブランチ）
git checkout feat/752-metal-aligned-load
cargo run -p backend-metal --example gemm_bench --release > /tmp/gemm_bench_head.txt
cargo run -p backend-metal --example gemm_f32_prepared_bench --release > /tmp/gemm_f32_prepared_bench_head.txt
```

出力形式（`examples/gemm_bench.rs`／`examples/gemm_f32_prepared_bench.rs` 参照）は
`docs/perf/metal-gemm-dynamic-tile.md` と同一（`size=<N>` 行・`shape=(<M>x<N>x<K>)` 行・
`size=<N> candidate=<label>` 行）。整列形状（512/1024/2048/4096 の正方立方 `m == n == k`。いずれも
`CANDIDATES[3]`〈32x32/bk16/staged〉の bm/bn/bk=32/32/16 を整除し `ALIGN_M=ALIGN_N=ALIGN_K=true` へ
畳み込まれる）の TFLOPS を base/head で突き合わせる。

数値一致確認（採否判断より前に必須。恒真化のみでビット単位一致が理論上成立するはずの前提を検証する）:

```sh
cargo test -p backend-metal --release --all-features -- --ignored --nocapture
```

`gemm_dynamic_tile_parity`（新規追加した `staged_path_matches_cpu_reference_fully_aligned`・
`staged_path_matches_cpu_reference_partially_aligned`・`direct_load_path_matches_cpu_reference_fully_aligned`
を含む）・`cpu_metal_parity`・`gemm_auto_parity` 等が green であること（tolerance は変更しない。coding-rust.md）。

## 判断基準

- base に対し head の中央値 TFLOPS が非劣化かつ整列形状（512/1024/2048/4096）で改善していれば「採用」とし、
  本ドキュメントへ実測結果を追記する
- 改善が確認できなければ**採用しない**と判断し、アラインメント特化ロード分岐一式（`gemm.metal` の
  `ALIGN_M`/`ALIGN_N`/`ALIGN_K` 宣言・OR 合成ガード、`crate::tile::AlignFlags`、`pipeline.rs`・`gemm.rs`
  の関連変更、`shader_source_evidence.rs`・`gemm_dynamic_tile_parity.rs` の追加テスト）を revert PR で
  撤去し、その判断と実測値を本ドキュメントへ記録する

## 実測結果

（未計測。実機セッションで本節へ追記する）
