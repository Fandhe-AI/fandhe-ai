# 常駐チェーン CPU train の REGRESS 切り分け（イシュー #3611）

## 状態

- **手順の登録のみ。実機計測（Mac mini・GB10）は未実測**。結果は決定記録 `docs/compat-resident-train-step-decision.md` §17.5 へ追記する。
- 背景: #3257 の参考系列で CPU の train セルが両機体とも REGRESS（Mac mini ratio_ab 1.004・GB10 1.059。`../resident-chain-ab-3257/judge-3257.md`）。原因は未調査。
- 本診断は分類（ノイズ幅内／配置の影響／チェーン化の後退／chain が速い／判定不能）であり、#3257 の RULE・判定は変えない。是正の要否は所有者が判断する。

## 収録物

| ファイル | 内容 |
|---|---|
| `DIAG.txt` | 事前登録手順（計測前に固定。対象 SHA・パッチ sha256・7 腕・巡回・ゲート・hard 条件・分類規則・仮説・限界） |
| `layout-shift.patch` | 配置の対照腕 B2 にだけ当てる計測専用パッチ（bench-fandhe の `main.rs` のみ。コミットは本パッチファイルだけで `crates/` には触れない） |
| `scripts/bench/framework-compare/run_diag_resident_chain_cpu.sh` | 両機体共用ランナー（B0・B1・B2 の 3 ツリー・7 腕・5 round） |
| `scripts/bench/framework-compare/summarize_diag_resident_chain_cpu.py` | DIAG の分類規則を fail-closed に適用する集計（`..._test.py` は CI で実行） |

## 実行手順（実測時）

1. B0（`786d18b50`）と B1（DIAG の `b1`）を bundle で各ノードの専用ディレクトリへ clone し、`.rev-stamp`（1 行目 HEAD・2 行目 `dirty=0`）を作って `.git/info/exclude` へ足し、`bench-resident-chain.patch` を `git apply` する。ランナー（DIAG・パッチを含むコミット）も別 clone で置く。
2. `docs/real-hardware-verification-env.md` §7 に従い機体ロックを取る。
3. `AB_PROFILE=macmini|gb10 AB_B0_FACADE_PATH=<b0>/crates/facade AB_B1_FACADE_PATH=<b1>/crates/facade bash scripts/bench/framework-compare/run_diag_resident_chain_cpu.sh <label>`（GB10 は `setsid nohup`、Mac mini は nohup サブシェル）。B2 は B1 から自動で作られ終了時に削除される。
4. 生ログを回収し、ホスト名・`$HOME`・ユーザー名をマスクして `macmini/`・`gb10/` に置く（sha・digest・JSONL の数値は置換しない）。
5. `python3 -I scripts/bench/framework-compare/summarize_diag_resident_chain_cpu.py --diag DIAG.txt --dir <dir> --label <label> --profile <macmini|gb10>` の出力を終了コードつきで保存する。

## 注意

- HEAD の main は `fandhe-ai =0.11.0` ピンが path patch に解決されないため（#3623 の 0.12.0 版上げ以降）、B1 は `a8e91f0c9`（`.rs` は main と同一）。#3273 のピン更新後に HEAD を使う場合は別名の DIAG を計測前に登録する。
- 開発機（M4 Max）の値は記録に使わない。
