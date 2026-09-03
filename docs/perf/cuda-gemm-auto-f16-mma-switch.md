# CudaGemmAuto::run_f16 の MatrixUnit 分岐 mma 優先化 前後比較（#1156）

イシュー #1156 の受け入れ条件 R5（ユーザー承認条件）「結線前後で同一プロトコル・
5 回計測中央値の比較を取り、後退確認時は結線しない」に対応する記録。設計の正は
`docs/dispatch-rules-design.md` §5.6。

## 状態: 未実測（本エージェント実行環境に CUDA 実機なし）

本ドキュメントを執筆したエージェント実行環境（macOS worktree）には CUDA driver・
実機がないため、切替前後（base `0c91218` vs HEAD）の `CudaGemmAuto::run_f16`
（転送込み・dim 512/1024/2048/4096・5 回計測中央値）の実機比較は未実施である。

実測値を捏造せず「未実測」と明記して安全側で結線する（`docs/perf/cuda-tf32-
optin-parity.md`「実機なしのため未実測明記」と同じ先例方針）。未実測は「後退
確認」ではないため #1156 の結線撤回条件（後退が確認された場合は結線しない）には
該当しない。

先行根拠として、`docs/perf/cuda-wmma-f16-perf-triage.md` §3.1/§4.1（イシュー
#1123・2026-09-03 GB10 実機実測）のカーネル単体計測が `mma_sync_f16` が
`wmma_f16_opt`/`wmma_f16_basic` に対し形状依存で約 4.1〜10.8 倍高速であることを
既に確認済みである。この結果は `CudaGemmAuto::run_f16` を経由しないカーネル単体
プロトコルでの計測であり、本ドキュメントが記録すべき「auto 経路（転送込み）での
切替前後比較」とは計測対象が異なる点に注意（正式な TFLOPS 記録・auto 経路実測は
#1160 が引き継ぐ）。

## 実機実行手順（DGX Spark GB10 想定。持ち帰り用）

転送は rsync 方式（`docs/real-hardware-verification-env.md`。ホスト名等はローカル
管理値を使い本ドキュメントには書かない）。

計測用の `#[ignore]` テスト（`bench_harness::protocol::run`。warmup 20・計測 20）は
**本 PR では未追加**であり、下記コマンドは既存の parity テスト（`tests/gemm_auto.rs`）
を実行するのみで TFLOPS 数値は得られない。5 回計測中央値の before/after 測定用テストは
#1160 が追加する。

```bash
# 参考: 既存 parity テストの実行（数値一致確認。TFLOPS 計測ではない）。
cargo test -p fandhe-ai-backend-cuda --release --all-features --test gemm_auto \
  -- --ignored --nocapture
```

before/after 比較の本実測手順は #1160 で計測テストを追加した後に本ドキュメントへ
追記する。

## 実測結果

| dim | base（結線前）TFLOPS 中央値 | HEAD（結線後）TFLOPS 中央値 | 判定 |
|---|---|---|---|
| 512  | 未実測 | 未実測 | — |
| 1024 | 未実測 | 未実測 | — |
| 2048 | 未実測 | 未実測 | — |
| 4096 | 未実測 | 未実測 | — |

実機セッションが持ち帰り次第、本表を更新する。
