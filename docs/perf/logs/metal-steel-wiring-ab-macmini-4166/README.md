# steel 候補の結線の Mac mini 前後比較（イシュー #4166）

結線（PR #4255。`tile::STEEL_LARGE_N_ENABLED = true`。M4Base の正方 1024／2048／4096 だけ）の後に、Mac mini（`<metal-node>`）の gemm Metal セルで後退していないことを確かめる前後比較の事前登録規則と道具。結論の記入先は `docs/perf/metal-gemm-steel-candidates.md` §11.5・§11.7。

## 状態: 事前登録・未実測

- 規則: `RULE.txt`（本 PR で固定。以後は本文を変えず、変更は `RULE.v2.txt` で行う）。
- 判定式・閾値・ゲート・NB-RANGE5／NB-SEP は #4160 の `RULE.txt`（`../metal-steel-large-n-ab-macmini-4160/RULE.txt`）のセル A/B の部分を参照する。本ディレクトリでは書き写さない。
- 計測は本 PR のマージ後。系列ごとに所有者の準備の連絡を受けてから起動する（RULE 項 4）。

## ファイル

| ファイル | 役割 |
|---|---|
| `RULE.txt` | 事前登録規則（対象 `9eba21749`・腕・指紋・道具の sha256・判定の写像・準備・系列の数え方・開示） |
| `deploy.sh` | Mac mini 上の展開。`9eba21749` の archive から展開ツリー・after（clean）・before（定数 1 行だけ false）を作り、指紋と manifest を残す |
| `fp.sh` | ツリー指紋（#4160 の RULE.txt 項 14 (a) と同じ求め方。`ab` で runner の出力を除く） |
| `prereq_tests.sh` | 前提の確認（RULE 項 3）。steel の 2 件・Mac mini 表の診断・parity 59 本の ignored を、件数を数えてから流す |
| `drive_series.sh` | 系列 1 本（RULE 項 4〜6）。準備の確認（ゲート・mediaanalysisd・hw-lock・load1・ツリー・腕の checkout）→ `AB_DIRECTION=post-wiring` で runner → 全ゲート通過の判定（値は見ない） |
| `judge_ab.py` | 判定。#4160 の `aggregate.py`（sha256 固定）の `load_ab`・`scoreboard` だけを呼び、PASS／FAIL／UNDETERMINED に写す |
| `test_drive_series.sh` | `drive_series.sh`・`prereq_tests.sh`・`fp.sh` の self-test（ps・sysctl・cargo などを偽物に差し替える） |

## 手順（Mac mini。bash 3.2 互換）

1. 送信側で `git archive --format=tar 9eba21749 > tree.tar` と、規則を固定したコミットの `git archive --format=tar <commit> docs/perf/logs/metal-steel-wiring-ab-macmini-4166 > rule.tar` を作り、Mac mini へ送る。
2. `bash deploy.sh <W> tree.tar rule.tar <RULE の tree_fingerprint>`。`<W>` は名前が `.noindex` で終わる新しいディレクトリ。`W/rule/` の各ファイルの sha256 を RULE 項 1 (d) と照合する。
3. `bash W/rule/prereq_tests.sh <W>`（hw-lock を取る）。`W/prereq/result.txt` が `all-pass` でなければ測らずに報告する。
4. 系列 k（1〜3）ごとに: 所有者へ「準備をお願いします」と伝える → 返事を受けたら `nohup bash W/rule/drive_series.sh <W> <k> <試みの番号> run > W/formal-<k>-<試み>.drive.out 2>&1 &`（サブシェルで切り離す）→ `W/drive.log` の最後の行で判定を見る。始めなかった試みは `W/prestart-<k>-<n>.txt`。
5. 全ゲート通過（`verdict=pass`）の系列が出たら止める。その系列の `results/raw` のうちラベル `formal-<k>` の記録を `<系列>/ab/` へ、`W/formal-<k>.status` を `<系列>/status.txt` へ写し、マスクしてから `python3 -I -B judge_ab.py <系列>` で判定する。

## 検証状況（固定前）

- `python3 -I -B judge_ab.py --self-test`: #4160 の `series-formal-v2-2/ab` を見本に、同系列の `aggregate.md` のセル A/B 表とスコアボード比（NB-SEP は 2048 だけ成立）を再現した。変異（5/5 round の後退・4/5 round の後退・checksum 不一致・セルのゲート・round 先頭の timeout・rc=1・skipped・`gate=pass-reference`・ラベル混在・sha256 不一致）で各分岐を確かめた。コードへの 4 個の変異（anomaly を PASS に・ゲート不通過を無視・`gate=pass` の部分一致・sha256 照合の無効化）は、どれも self-test で落ちた。
- `bash test_drive_series.sh`: 始めない分岐（ゲート FOREIGN／失敗・Safari・mediaanalysisd の境界 5.0／最大値／非数値／部分一致の名前・hw-lock 使用中・2 回目の確認・`__pycache__` によるツリー不一致・after の未追跡・before の余分な差分／差分なし／HEAD 違い・cargo が見つからない）、全ゲート通過の判定、runner へ渡す環境変数、`prereq_tests.sh` の件数照合。12 個の変異を入れ、どれも self-test が落ちることを確かめた。`/bin/bash`（3.2）でも通る。
- 手元（開発機。計測ではない）で `9eba21749` の archive に `deploy.sh` を通し、偽の cargo で runner をビルドの手前まで動かした。post-wiring の checkout 検査は通り、pre-wiring では止まった。
- 手元で本物の cargo に `--ignored --list` だけをさせ（GPU では流していない）、`--exact` の 2 件・Mac mini 表の 3 件・`gemm_splitk_parity` の 1 件が数えられ、`linalg_parity_inputs_guard` が 0 件になること、一覧のビルドの後も after が clean のままであることを確かめた。
- Mac mini で dry 用の別ディレクトリへ `deploy.sh` を通し（ノード側の指紋は RULE の値と一致）、`drive_series.sh` の dry を 1 回通した（全段 dry-ok。計測は起動していない。dry 用ディレクトリは消した）。
