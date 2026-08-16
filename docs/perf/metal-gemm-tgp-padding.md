# Metal GEMM threadgroup memory パディング（bank conflict 回避）A/B 計測記録（#538）

イシュー #538「perf(backend-metal): threadgroup memory のパディング（bank conflict 回避）を導入」の A/B 計測手順・記録テンプレート。
`gemm_simdgroup_tiled` の協調ロード（`USE_TGP_STAGING=true`）経路の共有メモリタイル（A: BM×BK、B: BK×BN）の行末へ
パディング（`TGP_PAD` function constant。既定 `pad: 4` 要素 = 16 バイト、`f32` の `16/sizeof(f32)`）を加算し、
`simdgroup_load` の列方向アクセスが threadgroup メモリのバンク境界と整合してしまうことによるバンクコンフリクトを
回避する変更の効果を計測する（MLX steel `gemm.h` の `tgp_padding_a`/`tgp_padding_b`・metal-flash-attention の
leadingBlockDimensions 実値指定・TileKernels の `TILE_X + TILE_K` 確保と同族の技法。CUDA 側 B-7 と同族）。

## 状態: 未計測。実機セッションで消化

本ファイルは Linux worktree で作成され、Metal 実機（Apple Silicon）が同一セッションで使用できないため計測手順・
記録テンプレートのみを整備した状態（#533・#536 と同方式）。`crates/backend-metal/tests/shader_source_evidence.rs` の
`gemm_metal_source_declares_tgp_pad_function_constant`・`gemm_simdgroup_tiled_source_uses_tgp_padding_stride`・
`gemm_simdgroup_tiled_source_retains_boundary_guard_with_padding`、および `crates/backend-metal/src/tile.rs` の
`TileConfig` 単体テスト（`shared_mem_bytes_includes_derived_pad_in_both_tile_strides_when_staged`・
`pad_is_derived_purely_from_staged`・`shared_mem_bytes_saturates_instead_of_wrapping_on_overflow` 等。`pad` は
イシュー #538 codex-review 指摘 P1 再指摘対応〈PR #673〉で `TileConfig` の `pub` フィールドから `staged` 由来の
導出メソッド `TileConfig::pad()` へ設計変更したため、テスト名もこれに合わせて変更済み）により、パディング機構の
実在・SMEM 使用量計算への反映・
REQ-8 境界チェック維持は Linux CI（ubuntu-latest）上で機械検査済みだが、数値一致の実測（ビット単位一致の理論前提の
検証）・性能効果の実測・採否判断（下記「判断基準」）は Mac 実機セッションでの後続対応が必要。

## 計測手順（Apple Silicon 実機）

base（変更前）と head（変更後）それぞれについて計測し、5 回計測の中央値 TFLOPS を比較する
（`bench-harness::protocol::run` が中央値計測を実装済み。`coding-rust.md` 準拠。手順自体は
`docs/real-hardware-verification-env.md` の接続・転送手順に従う。実ホスト名はローカル管理外ファイル参照）。

```sh
git fetch origin

# base（変更前。TGP_PAD 導入前の直近コミット）
git checkout <base-sha>
cargo run -p backend-metal --example gemm_bench --release > /tmp/gemm_bench_base.txt

# head（本イシューの実装ブランチ）
git checkout perf/538-metal-tgp-padding
cargo run -p backend-metal --example gemm_bench --release > /tmp/gemm_bench_head.txt
```

出力形式（`examples/gemm_bench.rs` 参照）は `docs/perf/metal-gemm-dynamic-tile.md` と同一（`size=<N>` 行・
`shape=(<M>x<N>x<K>)` 行・`size=<N> candidate=<label>` 行）。`gemm_simdgroup_tiled` 系（`GemmVariant::SimdgroupTiled`
候補構成のうち `USE_TGP_STAGING=true` を選ぶ構成・`dispatch_auto` 経由でこの経路を通る形状）の TFLOPS を
base/head で突き合わせる。`bm64_bn64_bk16_staged`・`bm32_bn32_bk16_staged` の 2 候補（`examples/gemm_bench.rs`）は
staged 経路のためパディングの影響を直接受ける（`bm32_bn32_bk16_direct` は `staged=false` のため無関係・変化なしの
はずである点も併せて確認する）。

数値一致確認（採否判断より前に必須。パディング導入がビット単位一致を崩さないはずの理論前提を検証する。
`gemm.metal` の staged 経路コメント「パディング列は simdgroup_load が一切読まないため 0 埋め不要」の実機検証）:

```sh
cargo test -p backend-metal --release -- --ignored --nocapture
```

`gemm_dynamic_tile_parity`・`cpu_metal_parity`・`all_tile_candidates_match_cpu_reference_medium_shape` 等が
green であること（tolerance は変更しない。coding-rust.md）。

## A-7 診断結果との突合

イシュー #487（A-7。`docs/perf/metal-gemm-bottleneck-diagnosis.md`）は本イシュー時点で実測（§4「実測結果」・
§5「結論」）が未実施のため、本ドキュメントの実測時点で A-7 側の実測が完了していれば、以下の観点で突合する:

- A-7 が「ロード律速」（実効帯域が理論帯域に対し低い）と診断していた場合、バンクコンフリクト回避によるロード
  効率改善は A-7 の律速要因を直接緩和する候補であり、改善幅が大きいことが期待される
- A-7 が「演算律速」（occupancy・arithmetic intensity 側が支配的）と診断していた場合、本変更の効果は限定的
  である可能性が高い（バンクコンフリクトはロード側のみに影響するため）
- A-7 が未実測のままの場合は本欄に「A-7 未実測のため突合不可」と記録し、判断基準（下記）は本ドキュメント単独の
  A/B 結果のみで下す

## 判断基準

- base に対し head の中央値 TFLOPS が改善していれば「採用」とし、本ドキュメントへ実測結果を追記する
- 改善が確認できなければ**採用しない**と判断し、パディングの変更（`gemm.metal` の `TGP_PAD`・`lda`/`ldb` 関連
  箇所、`tile.rs` の `TileConfig::pad()`（PR #673 で `staged` からの導出メソッドへ変更済み）・
  `shared_mem_bytes()`・`validate` 規則、`pipeline.rs` の index 6 定数、`shader_source_evidence.rs` の関連テスト）
  を revert PR で撤去し、その判断と実測値を本ドキュメントへ記録する

## 実測結果

（未計測。実機セッションで本節へ追記する）
