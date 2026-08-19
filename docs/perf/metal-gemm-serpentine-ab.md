# Metal GEMM 蛇行（serpentine）走査順 A/B 計測記録（#536）

イシュー #536「perf(backend-metal): acc_rows/acc_cols ループへ蛇行（serpentine）走査順を移植」の A/B 計測手順・記録テンプレート。
`gemm_simdgroup_tiled` の MMA 発行ループ（staged 経路・直接ロード経路）を、内側 `acc_cols` ループの行優先走査から
蛇行走査（奇数行 r で列を逆順に辿る。`c_ = (r % 2 == 1) ? (acc_cols - 1 - ci) : ci`）へ変更した効果を計測する
（MLX `tile_matmad`〈`mma.h`〉・CUTLASS `mma_tensor_op.h` 同型。CUDA 側イシュー #497 B-6 と同一技法の Metal 適用）。

## 状態: 未計測。実機セッションで消化

本ファイルは Linux worktree で作成され、Metal 実機（Apple Silicon）が同一セッションで使用できないため計測手順・
記録テンプレートのみを整備した状態。`crates/backend-metal/tests/shader_source_evidence.rs` の
`gemm_simdgroup_tiled_source_uses_serpentine_scan_order` により蛇行走査式の実在は Linux CI（ubuntu-latest）上で
機械検査済みだが、性能効果の実測・採否判断（下記「判断基準」）は Mac 実機セッションでの後続対応が必要。

**イシュー #745 による構造変化（重要）**: #745 で `gemm_simdgroup_tiled` の staged 経路を「kk ステップ先頭で
A/B フラグメントを一括ロードしてからレジスタ常駐のまま MMA を発行する」構造（MLX steel `mma.h` 型）へ再構成した
結果、staged 経路の蛇行走査（本ドキュメントが対象としていたもの）は「`b_tile` を内側ループで毎回再ロードする
構造」という前提自体が消滅したため撤去した。本ドキュメントが計測対象とする蛇行走査は、以後 **direct-load 経路
（`staged=false`。本番では `SINGLE_SIMDGROUP_8X8` のみで使用）にのみ残存する**。staged 経路（`CANDIDATES` の
本番候補すべて）の A/B 比較は `docs/perf/metal-gemm-register-accumulator-ab.md`（#745）が引き継ぐ。本ドキュメントの
実機計測・採否判断は direct-load 経路（影響範囲が狭い）に限定して行う。

## 計測手順（Apple Silicon 実機）

base（変更前）と head（変更後）それぞれについて計測し、5 回計測の中央値 TFLOPS を比較する
（`bench-harness::protocol::run` が中央値計測を実装済み。`coding-rust.md` 準拠。手順自体は
`docs/real-hardware-verification-env.md` の接続・転送手順に従う。実ホスト名はローカル管理外ファイル参照）。

```sh
git fetch origin

# base（変更前。蛇行走査導入前の直近コミット）
git checkout <base-sha>
cargo run -p backend-metal --example gemm_bench --release > /tmp/gemm_bench_base.txt

# head（本イシューの実装ブランチ）
git checkout perf/536-metal-gemm-serpentine
cargo run -p backend-metal --example gemm_bench --release > /tmp/gemm_bench_head.txt
```

出力形式（`examples/gemm_bench.rs` 参照）は `docs/perf/metal-gemm-dynamic-tile.md` と同一（`size=<N>` 行・
`shape=(<M>x<N>x<K>)` 行・`size=<N> candidate=<label>` 行）。`gemm_simdgroup_tiled` 系（`GemmVariant::SimdgroupTiled`
候補構成・`dispatch_auto` 経由でこの経路を通る形状）の TFLOPS を base/head で突き合わせる。

数値一致確認（採否判断より前に必須。走査順の並べ替えのみでビット単位一致が理論上成立するはずの前提を検証する）:

```sh
cargo test -p backend-metal --release -- --ignored --nocapture
```

`gemm_dynamic_tile_parity`・`cpu_metal_parity` 等が green であること（tolerance は変更しない。coding-rust.md）。

## 判断基準

- base に対し head の中央値 TFLOPS が改善していれば「採用」とし、本ドキュメントへ実測結果を追記する
- 改善が確認できなければ**採用しない**と判断し、蛇行走査の変更（`gemm.metal` の該当箇所・
  `shader_source_evidence.rs` の `gemm_simdgroup_tiled_source_uses_serpentine_scan_order` テスト）を revert PR で撤去し、
  その判断と実測値を本ドキュメントへ記録する

## 実測結果

（未計測。実機セッションで本節へ追記する）
