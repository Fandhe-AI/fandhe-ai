# Metal GEMM staged ロード float4 ベクトル化 A/B 計測記録（#533）

イシュー #533「perf(backend-metal): gemm_simdgroup_tiled の staged ロードを float4 ベクトル読み出しへ変更」の A/B 計測手順・記録テンプレート。
`gemm_simdgroup_tiled` の協調ロード（`USE_TGP_STAGING=true`）経路を、A タイル（BM×BK）・B タイル（BK×BN）の
1 要素ずつのスカラーロードから `float4` 単位のベクトルロード（境界外グループは要素単位スカラー読み出し + 0 埋めへ
フォールバック）へ変更した効果を計測する（MLX `BlockLoader`〈`loader.h`〉が 128bit 幅相当の単位で読み出す方式を参考にした適用）。

## 状態: 未計測。実機セッションで消化

本ファイルは Linux worktree で作成され、Metal 実機（Apple Silicon）が同一セッションで使用できないため計測手順・
記録テンプレートのみを整備した状態。`crates/backend-metal/tests/shader_source_evidence.rs` の
`gemm_simdgroup_tiled_source_uses_float4_staged_load`・`gemm_simdgroup_tiled_source_retains_float4_load_boundary_fallback`
により float4 ベクトルロードの実在・境界チェック維持は Linux CI（ubuntu-latest）上で機械検査済みだが、数値一致の
実測（ビット単位一致の理論前提の検証）・性能効果の実測・採否判断（下記「判断基準」）は Mac 実機セッションでの
後続対応が必要。

## 計測手順（Apple Silicon 実機）

base（変更前）と head（変更後）それぞれについて計測し、5 回計測の中央値 TFLOPS を比較する
（`bench-harness::protocol::run` が中央値計測を実装済み。`coding-rust.md` 準拠。手順自体は
`docs/real-hardware-verification-env.md` の接続・転送手順に従う。実ホスト名はローカル管理外ファイル参照）。

```sh
git fetch origin

# base（変更前。float4 ベクトル化導入前の直近コミット）
git checkout <base-sha>
cargo run -p fandhe-ai-backend-metal --example gemm_bench --release > /tmp/gemm_bench_base.txt

# head（本イシューの実装ブランチ）
git checkout perf/533-metal-gemm-float4-staged-load
cargo run -p fandhe-ai-backend-metal --example gemm_bench --release > /tmp/gemm_bench_head.txt
```

出力形式（`examples/gemm_bench.rs` 参照）は `docs/perf/metal-gemm-dynamic-tile.md` と同一（`size=<N>` 行・
`shape=(<M>x<N>x<K>)` 行・`size=<N> candidate=<label>` 行）。`gemm_simdgroup_tiled` 系（`GemmVariant::SimdgroupTiled`
候補構成のうち `USE_TGP_STAGING=true` を選ぶ構成・`dispatch_auto` 経由でこの経路を通る形状）の TFLOPS を
base/head で突き合わせる。

数値一致確認（採否判断より前に必須。ロード手段の変更のみでビット単位一致が理論上成立するはずの前提を検証する）:

```sh
cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture
```

`gemm_dynamic_tile_parity`・`cpu_metal_parity` 等が green であること（tolerance は変更しない。coding-rust.md）。

## 判断基準

- base に対し head の中央値 TFLOPS が改善していれば「採用」とし、本ドキュメントへ実測結果を追記する
- 改善が確認できなければ**採用しない**と判断し、float4 ベクトルロードの変更（`gemm.metal` の該当箇所・
  `shader_source_evidence.rs` の `gemm_simdgroup_tiled_source_uses_float4_staged_load`・
  `gemm_simdgroup_tiled_source_retains_float4_load_boundary_fallback` テスト）を revert PR で撤去し、
  その判断と実測値を本ドキュメントへ記録する

## 実測結果

（未計測。実機セッションで本節へ追記する）
