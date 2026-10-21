# metal-infer-bimodal-macmini-3183

イシュー #3183（Mac mini の infer Metal reuse の二峰性の原因特定。診断のみ）の計測ログ置き場。
背景・仮説・限界・結果は `docs/perf/metal-infer-bimodal-macmini.md`。

## 構成

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録の判定規則（実測前に単独コミットで固定。事後に緩めない） |
| `orchestrate.sh` | テストバイナリ 2 本を 1 回だけビルドし、run ごとに腕の起動順をローテーションして独立 5 回実行する。`run{1..5}.jsonl`・`env_info.txt`・`env_run{N}.txt`・`load_gate*.{log,txt}`・`power_run{N}.txt` を残す（`macmini` のみ受け付ける。既存 run があれば中止） |
| `aggregate.py` | 5 run を集計（python3 標準ライブラリのみ。n≠500・非有限／負値・JSON 不正・腕／run／ゲート状態の欠落・checksum 不一致は fail-closed。`--self-test` あり） |
| `macmini/` | Mac mini の実測生ログ（2026-10-21。`rev-stamp.txt`・`prereq_bit_exact.log`・`aggregate.md`・`bench_control/` を含む） |

## 実行手順（Mac mini セッション）

転送・SSH は `docs/real-hardware-verification-env.md` §7 に従う（`METAL_NODE` の実値は `docs/real-hardware-verification-env.local.md`）。計測中はほかのビルド・計測を並走させない。

```bash
# 0. 前提確認: 分解と公開 API の bit 一致（既存テスト）
cargo test --release -p fandhe-ai --test metal_infer_tape_build_fixedcost_diag \
  -- --ignored --test-threads=1 --nocapture metal_infer_decomposition_matches_public_api_bit_exact
# 1. 独立 5 run（nohup で切り離し、生存確認する。macOS には setsid がない）
bash docs/perf/logs/metal-infer-bimodal-macmini-3183/orchestrate.sh macmini
# 2. 集計
python3 -I docs/perf/logs/metal-infer-bimodal-macmini-3183/aggregate.py \
  docs/perf/logs/metal-infer-bimodal-macmini-3183/macmini \
  > docs/perf/logs/metal-infer-bimodal-macmini-3183/macmini/aggregate.md
```

- 同セッションで bench-fandhe（crates.io 0.11.0・変更なし）の `infer metal 64 reuse` を 5 回実行し `macmini/bench_control/` へ記録する（記録のみ）
- 収録前にホスト名は `masked`、`$HOME` は `<home>` に置換する（`orchestrate.sh` の `mask`）。実ホスト名はコミットしない
- 開発機（M4 Max）でのスモークは出力形式の確認用で、計測として扱わずコミットしない
