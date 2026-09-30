# metal-tape-build-infer-phase-2114

イシュー #2114（Metal `tape_build` 削減の opt-in 機構と infer GPU 起動固定費の診断）の計測ログ置き場。
背景・仮説・限界は `docs/perf/metal-tape-build-infer-fixedcost.md`。

## 構成

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録の判定規則（実測前に固定。事後に緩めない） |
| `orchestrate.sh` | テストバイナリを 1 回ビルドし、独立 5 プロセスで実行して `run{1..5}.jsonl`・`env_info.txt`・`load_gate.log`・`load_gate_status.txt` を残す（m4max のみ。record_only） |
| `aggregate.py` | 5 run を集計（python3 標準ライブラリのみ。欠落・不正・n≠80・非有限/非正の median_s・checksum 不一致・ゲート状態欠落は fail-closed。`--self-test` あり） |
| `m4max/` | 実機実測時の生ログ（現状は未実測のため存在しない） |

## 実行手順（Mac 実機セッション）

```bash
# 1. 分解と opt-in キャッシュの bit 一致（hard assert）
cargo test --release -p fandhe-ai --test metal_infer_tape_build_fixedcost_diag \
  -- --ignored --test-threads=1 --nocapture metal_infer_decomposition_matches_public_api_bit_exact
# 2. 独立 5 run とその集計
bash docs/perf/logs/metal-tape-build-infer-phase-2114/orchestrate.sh m4max
python3 docs/perf/logs/metal-tape-build-infer-phase-2114/aggregate.py \
  docs/perf/logs/metal-tape-build-infer-phase-2114/m4max > docs/perf/logs/metal-tape-build-infer-phase-2114/m4max/aggregate.md
# 3. wall と GPU busy の差（record-only。出力は m4max/gpu_busy.txt へ手動転記）
cargo test --release -p fandhe-ai-backend-metal infer_fixed_cost_diag \
  -- --ignored --nocapture --test-threads=1
```

- 既存の `run{N}.jsonl` があると `orchestrate.sh` は中止する（差し替え禁止）
- 収録前にホスト名は `masked`、`$HOME` は `<home>` へ置換される（`env_info.txt`・`run*.err`）。手順 3 の転記時も同じ置換を行う
- 結果は `docs/perf/metal-tape-build-infer-fixedcost.md` §6 に RULE.txt の規則で記入する
