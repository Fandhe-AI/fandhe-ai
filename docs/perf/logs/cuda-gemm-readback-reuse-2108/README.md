# CUDA readback 宛先再利用 A/B（イシュー #2108）実行手順

GB10 実機セッション向け。判定は同ディレクトリの `RULE.txt`（実測前固定）に従う。設計は
`docs/perf/cuda-gemm-readback-reuse-2108.md`。**本ディレクトリ時点で GB10 実測は未実施**。

## 0. 順序（必須）

1. #2107 の `orchestrate.sh gb10` を **本 PR より前のコミット**（例: `8bba6b57`）の checkout で実行し、
   `aggregate.py` の判定（N ごとの支持／棄却／未確定）を得る（`docs/perf/logs/cuda-gemm-readback-attribution-2107/README.md`）。
2. 全 N が「棄却」なら本 A/B は実施せず `not applicable` と記録する。
3. それ以外は HEAD（本 PR 以降）で次節を実行する。

## 1. 実機テスト（bit 同一・staging 再利用。driver のみで実行可・NVRTC 不要）

```sh
cargo test -p fandhe-ai-backend-cuda --release --all-features \
    --test readback_reuse_bit_match_2108 -- --ignored --nocapture --test-threads=1
```

## 2. A/B 計測

```sh
cd scripts/bench/framework-compare
AB_PATCH_FACADE_PATH="$(cd ../../../crates/facade && pwd)" \
  bash run_ab_readback_reuse_cuda.sh head-<short sha>-2108
```

- 出力: `results/raw/results-dgx-readback-reuse-ab-<label>-{before,after}.jsonl`・manifest・gate／uptime ログ・
  `compare-readback-reuse-ab-<label>.md`（`compare_gemm_ab.py --threshold 1.00 --require-checksum-exact` の判定表）
- 呼び出し元シェルに `FANDHE_AI_CUDA_READBACK_DEST` が設定されていると停止する（before 腕は未設定が契約）
- `AB_LOAD_GATE_MODE=record_only` は非正式系列（ADOPT 不可）。exclusive の閾値は load1 < 1.0 固定

## 3. 収録（このディレクトリ）

`env_info.txt`（`nvidia-smi`・`uptime`）・生 JSONL・比較表を置く。内部ホスト名・絶対パスは `<home>` 等へマスクする。
判定後に `docs/perf/cuda-gemm-readback-reuse-2108.md` §5 を埋める。ADOPT の場合のみ、ユーザー承認後の別 PR で
`memory::READBACK_DEST` を切り替える。
