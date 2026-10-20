# Mac mini の SME ON／OFF A/B ログ（イシュー #3176）

`SME_PRODUCTION_ENABLED` を反転した A/B（framework-compare の CPU セル）・R4 格子・MLP 長方形 4 形状の事前登録規則・patch・オーケストレーション・集計ツール・実行手順。
結論は [`docs/perf/cpu-gemm-sme-ab-macmini.md`](../../cpu-gemm-sme-ab-macmini.md)。

## 収録物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に固定。セル分類・前提ゲート・判定語彙・負荷ゲート 3.0） |
| `on-arm.patch` | after 腕（`gemm_blis/mod.rs` の `SME_PRODUCTION_ENABLED` 1 行。sha256 は RULE 記載） |
| `orchestrate_macmini.sh` | P-TREE → P-R0 → R4 5 run → 長方形 5 run → A/B の一括実行（`--dry-run` あり） |
| `aggregate.py` | 厳密集計（`--self-test` あり。異常入力は undetermined へ写像） |
| `macmini/` | 実測ログ（マスク済み）。**未実測のため未生成** |

A/B 本体は `scripts/bench/framework-compare/run_ab_sme_macmini_cpu.sh`、長方形の診断は `crates/backend-cpu/src/gemm_blis/mod.rs` の `sme_vs_neon_ab_mlp_rect`（`#[ignore]`）。

## 実行手順（Mac mini 実機。ホスト名と手順は `docs/real-hardware-verification-env.md` §7・Git 管理外の `.local.md`）

1. ローカルで `git bundle create <tmp>/3176.bundle <branch>` を作り rsync で送る。Mac mini 上で `git clone <bundle> before`（差分検証に git が必要なため展開ツリー方式は使わない）。target と WORK は同期ツリーの外に置く。
2. `RAYON_NUM_THREADS` を unset し、他の重い処理を止める。torch venv はリポ外の既存 venv を使う。
3. 計画確認: `BEFORE=… TORCH_PY=… WORK=… bash docs/perf/logs/sme-macmini-3176/orchestrate_macmini.sh --dry-run`
4. macOS には `setsid` がないので `nohup` のサブシェルで `zsh -lc` 経由で起動し、`ps` で生存確認する（`--dry-run` を外す）。
5. 完了後に `macmini/` を回収し、`python3 -I aggregate.py <macmini> > aggregate.md`。終わったら before/after の `target-ab-*` と probe の target を削除する（SSD は 228 GiB）。

## 状態

本 PR を作った実行環境から Mac mini へ到達していないため、計測は**未実施**（判定は undetermined）。
基盤（診断テスト・A/B スクリプト・オーケストレーション・集計ツール）は M4 Max 開発機で `bash -n`・`shellcheck`・`--dry-run`・`aggregate.py --self-test`・診断テストの smoke（記録に使わない）まで確認した。
