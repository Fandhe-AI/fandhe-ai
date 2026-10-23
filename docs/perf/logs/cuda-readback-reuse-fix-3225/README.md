# CUDA readback 是正（背景事前タッチ宛先プール）A/B（イシュー #3225）

- 判定規則: `RULE.txt`（計測前に固定）。記録先の設計・経緯: `docs/perf/cuda-gemm-reuse-phase-breakdown.md` §15。
- 実行: `scripts/bench/framework-compare/run_ab_readback_prefetch_cuda.sh head-<short sha>-3225`
  （GB10 実機。`AB_PATCH_FACADE_PATH` は本リポジトリ HEAD の `crates/facade`）。
- 出力（`scripts/bench/framework-compare/results/raw/`）: before／after の JSONL・manifest・gate ログ・
  uptime ログ・`compare-readback-prefetch-ab-<label>.md`（4 群の比較。n256 は record_only）。
- 状態: **GB10 実測済み・判定 ADOPT**（2026-10-23。ラベル `head-73768af-3225`・計測 sha `73768af83c0f45b482f1a209d4539a4f06b8e0bc`・専有ゲート exclusive 成立・
  `checksum_gate: OK`・`#[ignore]` bit 一致テスト pass・`.rev-stamp` 突合一致）。全セルの表・後退セルの併記は `docs/perf/cuda-gemm-reuse-phase-breakdown.md` §15.2。
  既定値 `READBACK_DEST` の反転は #3227 の方針に従う別 PR（本記録では変えない）。
- 収録物（ホスト名・絶対パスは `<home>` へマスク済み）: before／after の JSONL・`manifest-*.json`・`gate-*.log`・`uptime-*.log`・`skipped-*.log`（空）・
  `compare-readback-prefetch-ab-head-73768af-3225.md`・`ab-3225.log`（実行ログ）・`bitmatch-3225.log`（bit 一致テスト）・`env_info.txt`・`rev-stamp-verification-3225.md`。
- GB10 での実行手順（bundle 方式。`.git` が要るため rsync ではなく git bundle を送る）: 手元で `git bundle create` → ノードへ転送 →
  ノード側で `git init`・`git fetch <bundle> <branch>`・`git checkout --detach FETCH_HEAD`・`git rev-parse HEAD > .rev-stamp` →
  先に `cargo test -p fandhe-ai-backend-cuda --release --all-features --test readback_prefetch_bit_match_3225 -- --ignored --nocapture --test-threads=1` →
  `cd scripts/bench/framework-compare && AB_PATCH_FACADE_PATH="$(cd ../../../crates/facade && pwd)" bash run_ab_readback_prefetch_cuda.sh head-<sha7>-3225`。
