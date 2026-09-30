# CUDA 推論チェーン CUDA Graph capture（#2115）実測ログ置き場

判定規則は [`RULE.txt`](./RULE.txt)（実測前に固定・事後に緩めない）、記録本文は
[`../../infer-chain-graphcapture-cuda-ab.md`](../../infer-chain-graphcapture-cuda-ab.md)。
**現時点の verdict は `undetermined`**（実装セッションに DGX Spark GB10 がなく、
R0〜R2・5 run A/B は未実行）。

## GB10 での実行手順（別セッション・ユーザー承認後）

前提: 本ブランチを GB10 へチェックアウトし、`<repo>` をそのルートとする。
`FANDHE_AI_CUDA_GRAPH_INFER` は**最初の CUDA デバイス初期化より前**に環境変数で与える
（created stream は ordinal ごとに sticky）。

```sh
# R0: 前提ゲート（非後退）
cargo test -p fandhe-ai-backend-cuda --release --test linear_forward_device_real_device -- --ignored --nocapture
cargo test -p fandhe-ai-backend-cuda --release --test graph_capture_real_device -- --ignored --nocapture --test-threads=1

# R1: capture の正当性（"INFER_CAPTURE_R1: PASS" が 5 行出ること。SKIP は不合格）
FANDHE_AI_CUDA_GRAPH_INFER=1 cargo test -p fandhe-ai-backend-cuda --release \
  --test infer_graph_capture_real_device -- --ignored --nocapture --test-threads=1
FANDHE_AI_CUDA_GRAPH_INFER=1 cargo test -p fandhe-ai --release \
  --test predict_device_chain_cuda_bit_identity -- --ignored --nocapture

# R2: OFF/ON の bit ダンプ diff（batch ごとに --exact で単独実行。各 2 プロセス）
for b in 64 1024 4096; do
  for mode in off on; do
    if [ "$mode" = on ]; then export FANDHE_AI_CUDA_GRAPH_INFER=1; else unset FANDHE_AI_CUDA_GRAPH_INFER; fi
    cargo test -p fandhe-ai --release --test predict_device_chain_cuda_graph_bit_identity \
      -- --ignored --exact --nocapture "predict_resident_bit_dump_cuda_graph_batch_$b" \
      | grep '^out\[' > "dump-$mode-$b.txt"
  done
  diff "dump-off-$b.txt" "dump-on-$b.txt" && echo "R2-PASS batch=$b"
done
unset FANDHE_AI_CUDA_GRAPH_INFER

# 5 run A/B（専有ゲート付き。GB10 が空いていること）
AB_PATCH_FACADE_PATH=<repo>/crates/facade \
  bash scripts/bench/framework-compare/run_ab_infer_graph_cuda.sh 2115-gb10-run1
```

期待行数: R2 のダンプは `batch * 10` 行（640 / 10240 / 40960）。
A/B の出力は `scripts/bench/framework-compare/compare-infer-graph-<label>.md` と
`results/raw/`（ログ・JSONL・ゲートログ・GPU 情報）。採否は RULE.txt の
ADOPT/REJECT/undetermined に機械的に従い、結果は perf 記録の記入欄へ転記する。
内部ホスト名・絶対パスは `<home>` 等にマスクしてからコミットする。

## 実装セッション（RTX 3060）の参考記録

[`env_info.txt`](./env_info.txt) 参照。GB10 判定の代わりにはならない。本機には CUDA
toolkit ヘッダがなく GEMM カーネルをコンパイルできないため、capture 経路の実機テストは
未実行（入力検査系の 1 テストのみ実行）。
