# 常駐チェーン CPU train の REGRESS 切り分け（イシュー #3611）

## 状態

- **実施 2（r2。2026-10-28）**: 正規化を直した `DIAG-r2.txt` を計測前にコミットし、両機体で 7 腕 x 5 round を完走。分類は Mac mini・GB10 とも (i) ノイズ幅内（N 0.1104／0.0938、D1 -0.0003／-0.0093）。数値・解釈・所有者判断事項は決定記録 §17.5「実施 2（r2）」、ログは `macmini-r2/`・`gb10-r2/`（`summary-3611r2-*.txt`）。以下の「2026-10-27」の記述は r1 の記録で、そのまま残す。
- **2026-10-27 に Mac mini・GB10 で実行したが、両機体とも計測前の precheck で停止し、計測は 1 本も走っていない（未実測。分類なし）**。停止理由は `layout_symbol_diff=0`（B1／B2 のシンボルのアドレス差を確認できない。v0 mangling のクレート識別子がビルドパス依存で B1／B2 の名前が一致しないため）。詳細と所有者判断事項は決定記録 `docs/compat-resident-train-step-decision.md` §17.5。
- 背景: #3257 の参考系列で CPU の train セルが両機体とも REGRESS（Mac mini ratio_ab 1.004・GB10 1.059。`../resident-chain-ab-3257/judge-3257.md`）。
- 本診断は分類（ノイズ幅内／配置の影響／チェーン化の後退／chain が速い／判定不能）であり、#3257 の RULE・判定は変えない。是正の要否は所有者が判断する。

## 実施メモ（2026-10-27）

- runner: `08902dbf3`（main の HEAD。DIAG・パッチ・ランナーは #3626 から変更なし。DIAG の sha256 は `b233e73b…` で両ノードのクローンと一致）。B0＝`786d18b50`・B1＝`a8e91f0c9`。bundle を各ノードの専用ディレクトリへ転送して clone し、`.rev-stamp`（`dirty=0`）を作ってパッチを適用した（GitHub 認証はノードに置いていない）。
- 機体ロック: 両ノードで `#3611 2026-10-27T05:46:48Z` を取得し、収録後の 2026-10-27T05:52:26Z に解放した（他のロックは無かった）。起動前の環境変数 `BENCH_LAYOUT_PAD_3611`・`RAYON_NUM_THREADS` は未設定。機体の同定（Mac mini＝`Apple M4`、GB10＝aarch64・GPU 名 `NVIDIA GB10`）は通過した。
- 結果: 両機体とも B0・B1・B2 のビルドと tree／sha の確認は通り、`precheck FAILED: ... layout_symbol_diff=0` で停止した。`env_info`・`results-*.jsonl`・`digest-*`・`load_gate-*` は出力されていない。集計（`summarize_...py`）は入力が無いため実行していない。
- マスク: ホームの絶対パスを `<home>` に置換した。sha・nm のシンボル名とアドレスは置換していない。
- 収録しないもの: bench バイナリ（`bench-fandhe-diag-*`）、`target-diag-*`、B2 の worktree、`.local.md`・`.env*`。ノード上の作業ディレクトリ（target を含む）は削除済み。

## 収録物

| ファイル | 内容 |
|---|---|
| `DIAG.txt` | 事前登録手順（計測前に固定。対象 SHA・パッチ sha256・7 腕・巡回・ゲート・hard 条件・分類規則・仮説・限界） |
| `DIAG-r2.txt` | 2 回目の事前登録（r1 の停止原因だった主要シンボルの正規化のみを直した。runner／helper の sha256 を固定） |
| `scripts/bench/framework-compare/diag_layout_symbols.py` | r2 の主要シンボル突き合わせ helper（legacy／v0 mangling を正規化。stdlib のみ） |
| `macmini-r2/`・`gb10-r2/` | r2 の生ログ（マスク済み）と集計出力（`summary-3611r2-*.txt`。末尾に終了コード） |
| `layout-shift.patch` | 配置の対照腕 B2 にだけ当てる計測専用パッチ（bench-fandhe の `main.rs` のみ。コミットは本パッチファイルだけで `crates/` には触れない） |
| `scripts/bench/framework-compare/run_diag_resident_chain_cpu.sh` | 両機体共用ランナー（B0・B1・B2 の 3 ツリー・7 腕・5 round） |
| `scripts/bench/framework-compare/summarize_diag_resident_chain_cpu.py` | DIAG の分類規則を fail-closed に適用する集計（`..._test.py` は CI で実行） |

## 実行手順（実測時）

1. B0（`786d18b50`）と B1（DIAG の `b1`）を bundle で各ノードの専用ディレクトリへ clone し、`.rev-stamp`（1 行目 HEAD・2 行目 `dirty=0`）を作って `.git/info/exclude` へ足し、`bench-resident-chain.patch` を `git apply` する。ランナー（DIAG・パッチを含むコミット）も別 clone で置く。
2. `docs/real-hardware-verification-env.md` §7 に従い機体ロックを取る。
3. `AB_PROFILE=macmini|gb10 AB_B0_FACADE_PATH=<b0>/crates/facade AB_B1_FACADE_PATH=<b1>/crates/facade bash scripts/bench/framework-compare/run_diag_resident_chain_cpu.sh <label>`（GB10 は `setsid nohup`、Mac mini は nohup サブシェル）。B2 は B1 から自動で作られ終了時に削除される。
4. 生ログを回収し、ホスト名・`$HOME`・ユーザー名をマスクして `macmini/`・`gb10/` に置く（sha・digest・JSONL の数値は置換しない）。
5. `python3 -I scripts/bench/framework-compare/summarize_diag_resident_chain_cpu.py --diag DIAG-r2.txt（r1 の記録は DIAG.txt） --dir <dir> --label <label> --profile <macmini|gb10>` の出力を終了コードつきで保存する。

## 注意

- r2 時点の補足: #3273 でピンは `=0.12.0` へ更新済みで、b1 `a8e91f0c9` と main の `crates/` は約 2000 行異なる（下記の「`.rs` は main と同一」は r1 時点の記述）。
- HEAD の main は `fandhe-ai =0.11.0` ピンが path patch に解決されないため（#3623 の 0.12.0 版上げ以降）、B1 は `a8e91f0c9`（`.rs` は main と同一）。#3273 のピン更新後に HEAD を使う場合は別名の DIAG を計測前に登録する。
- 開発機（M4 Max）の値は記録に使わない。
