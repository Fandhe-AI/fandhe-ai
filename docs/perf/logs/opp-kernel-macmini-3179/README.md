# opp-kernel-macmini-3179（イシュー #3179）

Mac mini で PyTorch（CPU／MPS）の `torch.mm` 純カーネル時間と Python 側 checksum を測る診断の資産。

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に固定） |
| `run-macmini.sh` | Mac mini 用の実行ループ（負荷ゲート・起動順ローテーション・ログ非上書き） |
| `aggregate_rust.py` | Rust 逐次総和（`macmini/rust/`）の ns／要素の集計（5 サイズ × 5 run の完備を fail-closed で検査） |
| `macmini/` | 実測ログ（2026-10-21 UTC）。`run1`〜`run5/`・`rust/`・`aggregate.{json,md}`（Python 側）・`aggregate_rust.{json,md}`（Rust 側）・`gate.log`・`env.txt`・`versions.txt`・`rev-stamp.txt` |

診断本体は `scripts/bench/framework-compare/opp_kernel_torch_diag.py`、結論は `docs/perf/opp-kernel-macmini.md`。

## 再現手順

手順は `docs/real-hardware-verification-env.md` §7 に従う（`METAL_NODE` へ rsync、`zsh -lc`、macOS に setsid はないので nohup のサブシェルで切り離す）。
`run-macmini.sh` の冒頭コメントの環境変数を設定して実行し、生ログを回収する。`.rev-stamp` を `macmini/` に残す。

## 記録（実測時に埋める）

- 実行日時: 2026-10-21 05:24 UTC 開始（`macmini/env.txt`）。1 系列を一度で完走（GATE-TIMEOUT／GATE-INVALID／途中失敗なし。run の差し替えなし）
- 負荷: 65 プロセスすべて GATE-PASS（load1 は 1.54〜2.72。`macmini/gate.log`）
- `.rev-stamp`: `4c9abce7ed8e2f05561ef4b42e047602f34dca0c`（`macmini/rev-stamp.txt`）。bench-fandhe は fandhe-ai v0.11.0（`macmini/rust/tree.txt`）
- venv: Mac mini の `~/work/thread-3178/venv-torch`（Python 3.12.15・torch 2.14.0・numpy 2.5.4）
- RULE 項 2 の逸脱: 0.11.0 計測の作業ディレクトリは Mac mini に残っておらず、fill_vec のキャッシュを同じ seed で作り直した（checksum の系列 B 照合は通過）
- 入口の追加: 計測前に `run-macmini.sh` へ `REV_STAMP`・`TREE_TXT` の検査と複写、`RAYON_NUM_THREADS` 設定時の停止、`env.txt` への日時・負荷・熱状態の追記を足した（計測ロジック・腕・回数・ゲート閾値は不変）
- マスク: 回収後に `env.txt`・`build_bench.log`・`launch.log` の絶対ホームパス（`/Users/<名前>`）を `~` に置換した（値は不変。内部ホスト名・`uname -n` の出力は含まれない）
- スモーク（系列に数えない）: Rust CPU の N=4096 を 1 回、`opp_kernel_torch_diag.py --self-test` を 1 回、fill_vec のキャッシュ生成を実施
