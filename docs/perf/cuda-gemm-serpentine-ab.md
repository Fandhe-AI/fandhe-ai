# CUDA GEMM 蛇行（serpentine）走査順 A/B 計測記録（#497）

イシュー #497「perf(backend-cuda): mma 発行ループへ蛇行（serpentine）走査順を導入」の A/B 計測手順・記録テンプレート。
`crates/backend-cuda/src/kernels_mma.rs` の `MMA_F16` カーネル（f16 `mma.sync`/`ldmatrix`/`cp.async` 経路）の mma
発行 2 重ループ（mi/nj。#493 で導入した warp あたり 2x2 レジスタブロッキング）へ、内側 nj ループの蛇行走査
（外側 mi が奇数のとき nj を逆順に辿る。`int nj_s = (mi % 2) ? (WARP_TILES_N - 1 - nj) : nj;`）を導入した効果を
計測する（CUTLASS `mma_tensor_op.h`・MLX `mma.h` `tile_matmad` の `n_serp = (m % 2) ? (N - 1 - n) : n` と同型。
Metal 側イシュー #536・PR #652 の CUDA 適用）。

## 状態: 未計測。実機（DGX Spark spark-dbd9・sm_121）セッションで消化

本ファイルは Linux worktree（NVRTC 非搭載環境）で作成され、CUDA 実機が同一セッションで使用できないため
計測手順・記録テンプレートのみを整備した状態（PR #655 と同状況）。`crates/backend-cuda/src/kernels_mma.rs`
の `mma_f16_source_uses_serpentine_scan_order` により蛇行走査式の実在・mma 発行オペランドへの反映は Linux CI
（ubuntu-latest。GPU 不要のソース証跡テスト）上で機械検査済みだが、性能効果の実測・採否判断（下記
「判断基準」）は実機セッションでの後続対応が必要。

## 計測手順（DGX Spark GB10・sm_121 実機）

base（変更前）と head（変更後）それぞれについて計測し、5 回計測の中央値 TFLOPS を比較する
（`bench-harness::protocol::run` が中央値計測を実装済み。`coding-rust.md` 準拠。接続・転送手順は
`docs/real-hardware-verification-env.md` に従う。実ホスト名はローカル管理外ファイル参照）。

```sh
git fetch origin

# base（変更前。蛇行走査導入前の直近コミット = origin/main 0cd3c87 相当）
git checkout <base-sha>
cargo run -p backend-cuda --example gemm_mma_bench --release > /tmp/gemm_mma_bench_base.txt

# head（本イシューの実装ブランチ）
git checkout perf/497-cuda-mma-serpentine
cargo run -p backend-cuda --example gemm_mma_bench --release > /tmp/gemm_mma_bench_head.txt
```

出力形式（`examples/gemm_mma_bench.rs` 参照）の `MMA_F16` 経路（f16 `mma.sync`/`ldmatrix`/`cp.async`）の
TFLOPS を base/head で突き合わせる。

数値一致確認（採否判断より前に必須。走査順の並べ替えのみでビット単位一致が理論上成立するはずの前提を検証する。
本ファイル冒頭・kernels_mma.rs 該当コメント「数値不変性」参照）:

```sh
cargo test -p backend-cuda --release -- --ignored --nocapture
```

`cpu_cuda_mma_parity`・`parity_nonregression`（tolerance pin テスト含む）等が green であること
（tolerance 定数〈`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`〉・`parity_baseline.rs` は本イシューで
一切変更していない）。

レジスタスピル確認（TFLOPS 比較の前に必須。`nj_s` は `#pragma unroll` 下の実行時計算インデックスで `d`/
`b_frag`〈asm オペランド〉へのアクセスに使うため、コンパイラが定数畳み込みしきれない場合はローカルメモリへ
スピルし得る。スピルが起きると効果測定が「改善なし」ではなく「性能後退」として現れるため、両者を切り分ける）:

```sh
# NVRTC の -Xptxas -v 相当（レジスタ使用量ログ）で base/head 間の register 数・
# local memory 使用量に差がないことを確認してから TFLOPS を比較する
```

## 判断基準

- base に対し head の中央値 TFLOPS が改善していれば「採用」とし、本ドキュメントへ実測結果を追記する
- 改善が確認できなければ**採用しない**と判断し、蛇行走査の変更（`kernels_mma.rs` の該当箇所・
  `mma_f16_source_uses_serpentine_scan_order` テスト）を revert PR で撤去し、その判断と実測値を
  本ドキュメントへ記録する

## 実測結果

（未計測。実機セッションで本節へ追記する）
