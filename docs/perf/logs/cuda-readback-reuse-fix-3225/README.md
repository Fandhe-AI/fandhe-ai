# CUDA readback 是正（背景事前タッチ宛先プール）A/B（イシュー #3225）

- 判定規則: `RULE.txt`（計測前に固定）。記録先の設計・経緯: `docs/perf/cuda-gemm-reuse-phase-breakdown.md` §15。
- 実行: `scripts/bench/framework-compare/run_ab_readback_prefetch_cuda.sh head-<short sha>-3225`
  （GB10 実機。`AB_PATCH_FACADE_PATH` は本リポジトリ HEAD の `crates/facade`）。
- 出力（`scripts/bench/framework-compare/results/raw/`）: before／after の JSONL・manifest・gate ログ・
  uptime ログ・`compare-readback-prefetch-ab-<label>.md`（4 群の比較。n256 は record_only）。
- 状態: **GB10 実測は未実施**（この PR の作成時点）。結果が入るまで判定は undetermined。
