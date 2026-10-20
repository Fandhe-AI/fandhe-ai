# opp-kernel-macmini-3179（イシュー #3179）

Mac mini で PyTorch（CPU／MPS）の `torch.mm` 純カーネル時間と Python 側 checksum を測る診断の資産。

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に固定） |
| `run-macmini.sh` | Mac mini 用の実行ループ（負荷ゲート・起動順ローテーション・ログ非上書き） |
| `macmini/` | 実測ログ。**未生成**（unavailable。実行環境から Mac mini へ到達できなかった） |

診断本体は `scripts/bench/framework-compare/opp_kernel_torch_diag.py`、結論は `docs/perf/opp-kernel-macmini.md`。

## 再現手順

手順は `docs/real-hardware-verification-env.md` §7 に従う（`METAL_NODE` へ rsync、`zsh -lc`、macOS に setsid はないので nohup のサブシェルで切り離す）。
`run-macmini.sh` の冒頭コメントの環境変数を設定して実行し、生ログを回収する。`.rev-stamp` を `macmini/` に残す。

## 記録（実測時に埋める）

- 実行日時・負荷・`.rev-stamp`: unavailable
