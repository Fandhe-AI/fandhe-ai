# GB10 unified memory・Grace CPU 考察（イシュー #2123）の実行手順

状態: **未実測**（実行ホストが x86_64 で DGX Spark GB10 に到達できない）。
判定規則は `RULE.txt`（実測前に固定）。考察は `docs/gb10-unified-memory-grace-cpu-consideration.md`。

## GB10 での実行手順

1. 属性ダンプ: `cargo run -p fandhe-ai-backend-cuda --release --features internal-diagnostics --example unified_memory_probe`
2. SVE2 検出: `EXPECT_GRACE_SVE2=1 cargo test -p fandhe-ai-backend-cpu --release --test grace_sve2_probe_report -- --ignored --nocapture`
3. train reuse 再実測（R-UM-train）: `AB_PATCH_FACADE_PATH=<HEAD の crates/facade 絶対パス> bash scripts/bench/framework-compare/run_ab_managed_cuda.sh head-<sha>`（5 回中央値。`compare_managed_ab.py` の出力を使う）
4. 帯域（R-UM-bw）: `docs/perf/cuda-managed-placement-ab.md` §6 の手順で `managed_placement_bandwidth_real_device` を実行
5. env_info（`nvidia-smi`・`uptime`・`lscpu`）を取得し、ホスト名・`$HOME`・ユーザー名をマスクして本ディレクトリの `gb10/` へ収録
6. `RULE.txt` に従い判定し、考察 doc §6・§7 の記入欄を更新する

## 記入欄（未計測）

| 項目 | 値 |
|---|---|
| unified_memory_probe（GB10） | 未計測 |
| grace_sve2_probe 3 経路・sve_default_vl_bytes（参考値） | 未計測（cpuinfo 経路のみ既存ログで確認済み） |
| sve_running_vl_bytes（R-SVE2-kernel の判定値） | 未取得（現 probe は取得しない。取得手段は別イシュー・unsafe 承認事項）。当面 SVE2 カーネルは判定不能 |
| train reuse on/off 中央値 | 未計測 |
| 帯域 readback／upload／download | 未計測 |

## 参考（開発機の動作確認。GB10 の値ではない）

RTX 3060 の x86_64 開発機で `unified_memory_probe` が動作することのみ確認した
（MANAGED_MEMORY=1・CONCURRENT_MANAGED_ACCESS=1・PAGEABLE_MEMORY_ACCESS=0・INTEGRATED=0）。統合 GPU ではないため GB10 の判断材料にはならない。
