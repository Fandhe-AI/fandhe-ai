# opp-kernel-gb10-3185（イシュー #3185）

GB10 で PyTorch（cuBLAS）の `torch.mm` 純カーネル時間（FP32・TF32）と Python 側 checksum を測る診断の資産。
兄弟 #3179（Mac mini 版、`../opp-kernel-macmini-3179/`）の基盤を `importlib` で再利用し、#3179 側のファイルは変更しない。

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に単独コミットで固定） |
| `run-gb10.sh` | GB10 用の実行ループ（負荷ゲート・起動順ローテーション・ログ非上書き・版固定） |
| `gb10/` | 実測ログ。**未生成**（unavailable。下記） |

診断本体は `scripts/bench/framework-compare/opp_kernel_cuda_diag.py`、結論は `docs/perf/opp-kernel-gb10.md`。

## 実測の状況

- 実装時点で GB10 には SSH 到達できたが、RULE 項 2 が要求する `~/work/.venv-bench`（torch 2.14.0+cu130）が
  ノード上に存在せず、他の venv の torch は 2.13.0／2.11.0 のみだった。別版へ読み替えない規則（fail-closed）と、
  共有ノードへの大型パッケージ導入は承認事項であることから計測を行わず、unavailable とした。

## 再現手順

`docs/real-hardware-verification-env.md` §3〜4 に従い rsync で転送（`.local.md` 除外）し、`.rev-stamp` を残す。
ビルド（`cargo build --release -p fandhe-ai-backend-cuda --tests`）を先に済ませ、`run-gb10.sh` 冒頭コメントの
環境変数を設定して `setsid nohup` で切り離して実行し、`gb10/` を回収する。回収後に内部ホスト名・ユーザー名・
`/home`・`/Users` で始まるパスが無いことを grep で確認する。

## 記録（実測時に埋める）

- 実行日時・負荷・`.rev-stamp`: unavailable
