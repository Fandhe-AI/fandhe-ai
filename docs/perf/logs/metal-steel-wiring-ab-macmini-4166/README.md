# steel 候補の結線の Mac mini 前後比較（イシュー #4166）

結線（PR #4255。`tile::STEEL_LARGE_N_ENABLED = true`。M4Base の正方 1024／2048／4096 だけ）の後に、Mac mini（`<metal-node>`）の gemm Metal セルで後退していないことを確かめる前後比較の事前登録規則と道具。結論の記入先は `docs/perf/metal-gemm-steel-candidates.md` §11.5・§11.7。

## 状態: 実測済み・判定 PASS（2026-11-12・系列 `formal-1`）

- 規則: `RULE.txt`（PR #4260 で固定。マージコミット `b18fc055c`。本文は測った後も変えていない）。
- 判定式・閾値・ゲート・NB-RANGE5／NB-SEP は #4160 の `RULE.txt`（`../metal-steel-large-n-ab-macmini-4160/RULE.txt`）のセル A/B の部分を参照する。本ディレクトリでは書き写さない。
- 前提の確認（RULE 項 3）は all-pass。系列は `formal-1` の 1 本で全ゲート通過し、RULE 項 6 によりそこで止めて判定した（`formal-2`・`formal-3` は回していない）。判定は **PASS**（`series-formal-1/judge.md`）。

## 結果（2026-11-12・Mac mini `<metal-node>`）

### 前提の確認（RULE 項 3）

- `prereq/result.txt`: `all-pass 2026-11-12T07:07:24Z`。`prereq/summary.tsv` は 61 単位（pass 58・ignored テスト 0 件の `pass-no-ignored` 3）。steel の 2 件（`--exact`）と Mac mini 表の 3 件を含め、単位ごとに一覧の件数と passed が一致し、failed 0・終了コード 0。
- hw-lock は 1 回目で取れた（ロックが取れずにやり直した試みはない）。記録 PR には項 3 のとおり `summary.tsv` と `result.txt` だけを収める。

### 系列 `formal-1`

- 準備（RULE 項 4）: 所有者の準備完了の返事を受けて起動した。止めてほしいプロセスの名指しはなく、計測セッションは何も止めていない。`series-formal-1/series-start.txt` のとおり、他ジョブのゲート 2 回とも rc=0（`Safari` 本体なし）・mediaanalysisd 3 回とも %CPU 0・hw-lock 取得 08:03:31Z・load1=1.24・ツリーの照合が一致・checkout の照合（after clean、before は tile.rs の 1 行だけ、HEAD 一致、道具あり）が成り立った。
- 実行: 2026-11-12T08:03:39Z〜08:28:25Z。`drive/drive.log` の判定行は `formal-1 verdict=pass ab_rc0=1 ab_gate=5/5 skipped_bytes=0`（`ab_rc0=1` は状態ファイルに runner の `ab rc=0` の行が 1 本あるという意味）。他ジョブのゲートは 259 回とも CLEAN（`series-formal-1/ab/foreign-gate-summary.tsv`）、load ゲートは 5 round とも pass。
- 判定（`judge_ab.py`）: 全ゲート通過・load_ab の状態 ok・**PASS**。checksum は全セルで完全一致。5/5 round で一貫して 2% を超えて後退するセルはない。
- after/before の中央値比（round 別の比の中央値。`judge.md`）: N=2048 fresh 0.942・reuse 0.933、N=4096 fresh 0.906・reuse 0.918（いずれも 5/5 round で after が速い）。N=256／512／1024 は 0.977〜1.004 で、round ごとに向きが揃わない。
- **開示（compare の終了コード 3）**: runner の終了コードは 0 だが、`compare-exit-4160-metal-formal-1.log` は `compare exit=3`。`compare_gemm_ab.py`（`--threshold 1.00`）は 5 run を合わせた中央値どうしの比で判定し、N=1024 fresh を「後退」（3.487 ms ÷ 3.313 ms = 1.0527）と表示した。同じセルの round 別の比は 0.962・1.140・0.977・1.103・0.970（中央値 0.977、2% 超は 2/5）。RULE 項 2 が参照する #4160 の判定条件は「5/5 round で一貫して 2% を超える後退」で、このセルは当たらない。runner はこの終了コードを、checksum 不一致・判定不能でない限り記録だけにする（`skipped-4160-metal-formal-1.log` は 0 バイト）。#4160 の `formal-v2-2` でも同じ表示が 2 セル（256/fresh・512/reuse）に出ていた。N=1024 が「速くなった」とは書かない。
- スコアボード比（candle 相手。記録のみで PASS／FAIL には使わない）: NB-SEP は 3 点とも**不成立**。N=2048 は before 0.932〜0.940 → after 1.000〜1.004（#4160 の `formal-v2-2` では after 1.005〜1.014 で成立していた）、N=1024 は before 0.885〜1.044 → after 0.906〜1.057、G4096（記録のみ）は before 0.854〜0.924 → after 0.951〜1.006。この系列からは「1 位に変わった」とは書けない。

### 記録の配置

| パス | 中身 |
|---|---|
| `prereq/summary.tsv`・`prereq/result.txt` | 前提の確認（RULE 項 3） |
| `drive/drive.log` | 系列の判定行 |
| `drive/deploy.txt`・`drive/fp-deploy.txt`・`drive/fp-deploy-ab.txt` | 展開の記録（ノード側の指紋が RULE と一致） |
| `drive/precheck-dry-1.txt` | 固定前の dry（RULE 項 9。固定前の版の `drive_series.sh` で行った） |
| `series-formal-1/series-start.txt`・`status.txt`・`ab.out` | 系列の準備の確認・状態ファイル・runner の出力 |
| `series-formal-1/ab/` | runner の `results/raw` のうちラベル `formal-1` の記録（実行ファイルは除く）と、他ジョブのゲートの出力の `foreign-gate.tar.gz`・要約 `foreign-gate-summary.tsv` |
| `series-formal-1/judge.md` | `python3 -I -B judge_ab.py series-formal-1` の出力 |

回収時にホスト名を `<metal-node>`、`$HOME` を `<home>`、ユーザー名を `<user>` に置き換えた（RULE 項 8）。

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
3. `bash W/rule/prereq_tests.sh <W>`（hw-lock を取る）。`W/prereq/result.txt` が `all-pass` でなければ測らずに報告する。hw-lock が取れなかったとき（終了コード 5）は `W/prereq` を作らず `W/prereq-attempts.log` に残すので、ロックが外れたら同じ展開のまま再実行する。
4. 系列 k（1〜3）ごとに: 所有者へ「準備をお願いします」と伝える → 返事を受けたら `nohup bash W/rule/drive_series.sh <W> <k> <試みの番号> run > W/formal-<k>-<試み>.drive.out 2>&1 &`（サブシェルで切り離す）→ `W/drive.log` の最後の行で判定を見る。始めなかった試みは `W/prestart-<k>-<n>.txt`。`verdict=notpass` の系列は、その系列のラベルの記録を `<系列>/ab/` へ、`W/formal-<k>.status` を `<系列>/status.txt` へ写し `python3 -I -B W/rule/judge_ab.py --checksum-only <系列>` を流す（値は出ない）。`checksum=fail` なら止めてその系列を FAIL と判定する（RULE 項 6）。途中で止まった系列でも、`judge_ab.py <系列>` は記録済みの行の確かな checksum 不一致で FAIL を出す（速度表は出さない）。
5. 全ゲート通過（`verdict=pass`）の系列が出たら止める。その系列の `results/raw` のうちラベル `formal-<k>` の記録を `<系列>/ab/` へ、`W/formal-<k>.status` を `<系列>/status.txt` へ写し、マスクしてから `python3 -I -B judge_ab.py <系列>` で判定する。`aggregate.py` は、リポジトリでは隣の `metal-steel-large-n-ab-macmini-4160/`、Mac mini の展開では `W/tree/docs/perf/logs/metal-steel-large-n-ab-macmini-4160/` から読む（どちらも sha256 を照合する）。

## 検証状況（固定前）

- `python3 -I -B judge_ab.py --self-test`: #4160 の `series-formal-v2-2/ab` を見本に、同系列の `aggregate.md` のセル A/B 表とスコアボード比（NB-SEP は 2048 だけ成立）を再現した。変異（5/5 round の後退・4/5 round の後退・checksum 不一致（ゲート不通過の系列を含む）・`--checksum-only` の各状態（途中で止まった系列の記録済みの不一致・checksum 欠損を含む）・ゲート不通過の系列の NB-SEP・展開の配置での `aggregate.py` の探索・セルのゲート・round 先頭の timeout・rc=1・skipped・`gate=pass-reference`・ラベル混在・sha256 不一致）で各分岐を確かめた。コードへの 12 個の変異（anomaly を PASS に・ゲート不通過を無視・`gate=pass` の部分一致・sha256 照合の無効化・NB-SEP の無効化・`--checksum-only` の fail／unknown の取りこぼし・部分記録の照合の無効化・通常の判定での部分記録の FAIL の無効化・ゲート関連の入力の欠落で checksum の FAIL を失うこと・展開の配置の探索の削除）は、どれも self-test で落ちた。
- `bash test_drive_series.sh`: 始めない分岐（ゲート FOREIGN／失敗・Safari・mediaanalysisd の境界 5.0／最大値／非数値／部分一致の名前・hw-lock 使用中・2 回目の確認・`__pycache__` によるツリー不一致・after の未追跡・before の余分な差分／差分なし／HEAD 違い・cargo が見つからない・使用済みの試み番号での再実行）、全ゲート通過の判定、runner へ渡す環境変数、`prereq_tests.sh` の件数照合と、hw-lock が取れなかった後の再試行。14 個の変異を入れ、どれも self-test が落ちることを確かめた。`/bin/bash`（3.2）でも通る。
- 手元（開発機。計測ではない）で `9eba21749` の archive に `deploy.sh` を通し、偽の cargo で runner をビルドの手前まで動かした。post-wiring の checkout 検査は通り、pre-wiring では止まった。
- 手元で本物の cargo に `--ignored --list` だけをさせ（GPU では流していない）、`--exact` の 2 件・Mac mini 表の 3 件・`gemm_splitk_parity` の 1 件が数えられ、`linalg_parity_inputs_guard` が 0 件になること、一覧のビルドの後も after が clean のままであることを確かめた。
- Mac mini で dry 用の別ディレクトリへ `deploy.sh` を通し（ノード側の指紋は RULE の値と一致）、`drive_series.sh` の dry を 1 回通した（固定前の版。その後の変更は使用済みの試み番号を拒否する起動時の 1 行だけ。全段 dry-ok。計測は起動していない。dry 用ディレクトリは消した）。
