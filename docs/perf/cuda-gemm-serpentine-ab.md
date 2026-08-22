# CUDA GEMM 蛇行（serpentine）走査順 A/B 計測記録（#497）

イシュー #497「perf(backend-cuda): mma 発行ループへ蛇行（serpentine）走査順を導入」の A/B 計測手順・記録テンプレート。
`crates/backend-cuda/src/kernels_mma.rs` の `MMA_F16` カーネル（f16 `mma.sync`/`ldmatrix`/`cp.async` 経路）の mma
発行 2 重ループ（mi/nj。#493 で導入した warp あたり 2x2 レジスタブロッキング）へ、内側 nj ループの蛇行走査
（外側 mi が奇数のとき nj を逆順に辿る。`int nj_s = (mi % 2) ? (WARP_TILES_N - 1 - nj) : nj;`）を導入する案の
効果を計測する（CUTLASS `mma_tensor_op.h`・MLX `mma.h` `tile_matmad` の `n_serp = (m % 2) ? (N - 1 - n) : n`
と同型。Metal 側イシュー #536・PR #652 の CUDA 適用）。

## 状態: 撤回済み（未計測のまま本番カーネルへ導入しない）。実機（DGX Spark GB10・sm_121）セッションで計測後に再提案

`kernels_mma.rs` への蛇行走査コード（`nj_s` 変数・ソース証跡テスト `mma_f16_source_uses_serpentine_scan_order`）
は、PR #657 codex-review 指摘（P1: 性能改善を実測せずに `MMA_F16` 本番カーネルへ変更を導入している。下記
「判断基準」の採否契約を満たさない）を受けて revert した。Linux worktree（NVRTC 非搭載環境）では実機計測が
できないため（PR #655 と同状況）、本 PR の時点では計測手順・記録テンプレートのみを整備し、カーネル変更自体は
未導入の状態へ戻す。実機ツリー #408 側のセッションで下記手順に従い計測し、判断基準を満たした場合にのみ
別 PR で再導入する。

## 計測手順（DGX Spark GB10・sm_121 実機）

base（変更前）と head（変更後）それぞれについて計測し、5 回計測の中央値 TFLOPS を比較する
（`bench-harness::protocol::run` が中央値計測を実装済み。`coding-rust.md` 準拠。接続・転送手順は
`docs/real-hardware-verification-env.md` に従う。実ホスト名はローカル管理外ファイル参照）。

```sh
git fetch origin

# base（変更前。蛇行走査導入前 = 本 PR の revert 後の状態。origin/main 相当）
git checkout <base-sha>
cargo run -p fandhe-ai-backend-cuda --example gemm_mma_bench --release > /tmp/gemm_mma_bench_base.txt

# head（蛇行走査を再導入した実験ブランチ。上記「状態」節の revert 前コミット・
# または本ドキュメントの記述に従い再実装したもの）
git checkout <serpentine-experiment-branch>
cargo run -p fandhe-ai-backend-cuda --example gemm_mma_bench --release > /tmp/gemm_mma_bench_head.txt
```

出力形式（`examples/gemm_mma_bench.rs` 参照）の `MMA_F16` 経路（f16 `mma.sync`/`ldmatrix`/`cp.async`）の
TFLOPS を base/head で突き合わせる。

数値一致確認（採否判断より前に必須。走査順の並べ替えのみでビット単位一致が理論上成立するはずの前提を検証する。
上記蛇行走査式の設計意図「数値不変性」参照）:

```sh
cargo test -p fandhe-ai-backend-cuda --release -- --ignored --nocapture
```

`cpu_cuda_mma_parity`・`parity_nonregression`（tolerance pin テスト含む）等が green であること
（tolerance 定数〈`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`〉・`parity_baseline.rs` は変更しない）。

レジスタスピル確認（TFLOPS 比較の前に必須。`nj_s` は `#pragma unroll` 下の実行時計算インデックスで `d`/
`b_frag`〈asm オペランド〉へのアクセスに使うため、コンパイラが定数畳み込みしきれない場合はローカルメモリへ
スピルし得る。スピルが起きると効果測定が「改善なし」ではなく「性能後退」として現れるため、両者を切り分ける）:

```sh
# NVRTC の -Xptxas -v 相当（レジスタ使用量ログ）で base/head 間の register 数・
# local memory 使用量に差がないことを確認してから TFLOPS を比較する
```

## 判断基準

- base に対し head の中央値 TFLOPS が改善していれば「採用」とし、`kernels_mma.rs` へ蛇行走査コード
  （`nj_s` 変数・ソース証跡テスト）を再実装する PR を起票し、本ドキュメントへ実測結果を追記する
- 改善が確認できなければ**採用しない**と判断し、その判断と実測値を本ドキュメントへ記録して本イシュー
  （#497）をクローズする
- **未計測の間は「採用済み」として扱わない**。本番カーネルへの変更導入は、上記いずれかの判断が
  実機計測をもって確定してから行う（暫定導入は行わない）

## 実測結果

（未計測。実機セッションで本節へ追記する）
