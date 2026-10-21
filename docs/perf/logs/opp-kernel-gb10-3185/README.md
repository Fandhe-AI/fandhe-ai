# opp-kernel-gb10-3185（イシュー #3185）

GB10 で PyTorch（cuBLAS）の `torch.mm` 純カーネル時間（FP32・TF32）と Python 側 checksum を測る診断の資産。
兄弟 #3179（Mac mini 版、`../opp-kernel-macmini-3179/`）の基盤を `importlib` で再利用し、#3179 側のファイルは変更しない。

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に単独コミットで固定） |
| `run-gb10.sh` | GB10 用の実行ループ（負荷ゲート・起動順ローテーション・ログ非上書き・版固定） |
| `gb10/` | 実測ログ（2026-10-21 UTC。5 run 分の生ログ・`env.txt`・`gate.log`・`rev-stamp.txt`・`aggregate.{json,md}`・`cpu-blas/`・`fandhe/`・`venv-install.txt`・`build.log`・`smoke.txt`。絶対ホームパスは `~` にマスク済み） |

診断本体は `scripts/bench/framework-compare/opp_kernel_cuda_diag.py`、結論は `docs/perf/opp-kernel-gb10.md`。

## 実測の状況

- 2026-10-21（UTC）に GB10 で実測済み。5 run × 3 腕・`cpu-blas-probe`・fandhe テスト 5 run × 2 種が完走し、`aggregate` の fail-closed 照合を通過した。結論は `docs/perf/opp-kernel-gb10.md`

## 再現手順

`docs/real-hardware-verification-env.md` §3〜4 に従い rsync で転送（`.local.md` 除外）し、`.rev-stamp` を残す。
ビルド（`cargo build --release -p fandhe-ai-backend-cuda --tests`）を先に済ませ、`run-gb10.sh` 冒頭コメントの
環境変数を設定して `setsid nohup` で切り離して実行し、`gb10/` を回収する。回収後に内部ホスト名・ユーザー名・
`/home`・`/Users` で始まるパスが無いことを grep で確認する。

## 記録

- 実行日時・負荷: 2026-10-21 UTC。`gate.log` は 26 件すべて GATE-PASS（load1 0.56〜0.99・GPU 利用率 0%）。ノードには常駐の他サービスがあったが、GPU 利用率 0% のゲートを毎プロセス前に通した
- `.rev-stamp`: `c22b8c26e4ebd171cebd5c65346dc03435430e4c`（`gb10/rev-stamp.txt`）。転送は専用ツリー、ビルドは専用の `CARGO_TARGET_DIR`（`gb10/build.log`）
- venv: `~/work/.venv-bench` は所有者承認（2026-10-21、イシューコメント）に基づき新規導入。`torch==2.14.0`（`--index-url https://download.pytorch.org/whl/cu130`）、numpy は PyPI の最新（2.5.4。版は記録のみ）。`torch.__version__` = `2.14.0+cu130`、`torch.cuda.is_available()` = True（`gb10/venv-install.txt`）
- `bench_py.py`: ツリー外の専用ディレクトリに複写して使用（`fill_vec` のキャッシュが rsync で消える・誤コミットされるのを避けるため）。sha256 は RULE 項 3 と一致。ノード上の既存 `~/work/bench_py.py` は無かった
- スモーク（系列に数えない。`gb10/smoke.txt`）: `--self-test`・cpu 腕 1 回・fandhe 2 テストの `parse-fandhe` が通ることを確認。`measure --run 0` は「run が不正」で拒否されたため `--run 1` で出力先を分けて再実行した。計測ロジック・腕・回数・閾値は変更していない
- 逸脱: なし。スクリプト（`run-gb10.sh`・`opp_kernel_cuda_diag.py`）は無修正で実行した
- マスク: 絶対ホームパス（`/home/<ユーザー名>`）を `~` に置換（`launch.log`・`build.log`・`fandhe/*.log`・`venv-install.txt`・`smoke.txt`）。数値は書き換えていない。内部ホスト名・ユーザー名・`/home`・`/Users` が `gb10/` に無いことを grep で確認済み
