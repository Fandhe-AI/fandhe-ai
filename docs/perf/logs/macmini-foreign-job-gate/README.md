# Mac mini の他ジョブの前提ゲート（#4151・#4153・#4154・#4160 の測り直し用）

## 状態

部品だけを置く（実測なし）。各 issue の新しい RULE（#4151・#4154 は `RULE.v2.txt`、#4153 は `RULE.v3.txt`、#4160 は固定前の `RULE.txt`）が
本ディレクトリのスクリプトの sha256 を固定し、round ごとの記録を各 issue の系列ディレクトリの `env/foreign-gate/` に残す。

## なぜ足すか

2026-11-10 の <metal-node>（Mac mini）の計測区間の大半が、hw-lock を取らずに動いていた別のジョブ（wgpu の GPU GEMM・CPU 100%・caffeinate 付き）と
重なった（#4153 は round 2〜5、#4151・#4154 は全 round）。load1 のゲート（3.0 未満）は通っていたが、このゲートでは重なりを検出できない。
所有者判断（#4151・#4153・#4154 の 2026-11-10 のコメント「他ジョブと重なった系列は測り直す」）で、各 round の開始前と終了後に
他ジョブがいないことを確かめる前提ゲートを足すことになった。**ゲートを厳しくする方向だけの追加で、load1 のゲート・閾値・分類式・確度の規則は変えない。**

## 他ジョブとみなす基準（`foreign_job_gate.py`。RULE と同じ文言）

ps（`ps -A -o pid=,ppid=,uid=,pcpu=,comm=`）と GPU 使用率（`ioreg -r -d 1 -w 0 -c IOAccelerator` の `"Device Utilization %"`）を
1 回の判定で読む。次のどれか 1 つでも当たれば FOREIGN。

| 記号 | 基準 | 対象 |
|---|---|---|
| G1 | 実行ファイルの basename が `caffeinate` | ユーザーを問わない |
| G2 | %CPU（ps の pcpu）が 25.0 以上 | ユーザーを問わない |
| G3 | basename（小文字化）が `wgpu`・`python`・`cargo`・`rustc`・`xctrace`・`torch`・`bench`・`caffeinate` のどれかを含む | ユーザー・%CPU を問わない |
| G4 | `"Device Utilization %"` を 1 秒間隔で 5 回読み、10 以上の回が 2 回以上 | 機体全体 |

- **除外**: ゲート自身・その祖先（ラッパー・ドライバ）・その子孫（ps・ioreg）、pid 0（kernel_task）と pid 1（launchd）。名前では除外しない
  （自分の計測が python や cargo を動かしても、round の前後には生きていない。取り残されて生きていれば他ジョブと同じく当てる）。
- **G4 を「5 回中 2 回」にした理由**: 他ジョブの GPU 使用は断続的で、2026-11-10 に実機で稼働中の他ジョブへ 3 回読んだ値は 97・0・0 だった。
  最小値で判定すると見逃し、1 回だけで判定すると直前の自分の GPU 処理の残りで当たりうる。
- **基準の決め方**: 2026-11-10 に稼働中の他ジョブへ当てて、G1（caffeinate）・G2（99.3%）・G3（名前に wgpu）で FOREIGN になることを確かめた
  （ロック・計測なしの読み取りだけ）。このとき他ジョブ以外のプロセスの %CPU は最大 15.9%（コンテナ基盤のバックエンド）・11.7%（ssh のセッション）だった。
- **平常時の確認（G2 の 25.0 の根拠）**: 他ジョブの終了後（2026-11-10T06:50:45Z に driver が無いことを確認）、2 分待ってから 06:52:49〜06:57:59Z に
  30 秒間隔でゲートを 10 回当て、10 回とも CLEAN（終了コード 0）だった（ロックなしの読み取りだけ）。計測中と同じく、この間は Claude Code を待機させ、
  ssh も含めて Mac mini で何も動かしていない。除外対象（ゲート自身の系譜と pid 0・1）を除いた %CPU の最大は 10 回を通して 4.3%
  （s8 の backgroundtaskmanagementd）で、各回の最大は 0.5〜4.3%。G2 の 25.0 はこの最大の約 5.8 倍で、他ジョブの 99.3% との間にある。
  load1 は 0.92〜1.35。GPU 使用率は 50 回の読み取りのうち 1 回（s1 の 1 回目）だけ 85 で、ほかは 0。1 回だけの値では G4 に当たらない
  （「5 回中 2 回」の理由と同じ）。記録は `idle-check-20261110/`（`s1.txt`〜`s10.txt`・`status.txt`。マスク済み）。
- **限界**: ゲートは round の前後だけを確かめるので、round の途中で始まって終わるジョブは検出できない。
  ps の %CPU は減衰平均で、起動直後のジョブは低く出る（G3・G4 で補う）。GPU 使用率は機体全体の値で、どのプロセスかは分からない。

## 開始前・終了後の扱い（`round_with_foreign_gate.sh`）

| 時点 | CLEAN | FOREIGN | ゲート自身の失敗 |
|---|---|---|---|
| 開始前 | run_round.sh を始める | 30 秒間隔で最大 60 回（30 分）待つ。60 回とも FOREIGN なら run_round.sh を呼ばず `gate.txt` を `TIMEOUT foreign-job before <UTC>`（exit 1） | `gate.txt` を `INVALID foreign-gate-error before`（exit 1） |
| 終了後 | run_round.sh の終了コードを返す | `gate.txt` の 1 語目が PASS なら `INVALID foreign-after <UTC>` に書き換え（元の行は `# was: ` で残す）、exit 4 で系列を止める | FOREIGN と同じ（`INVALID foreign-gate-error after`・exit 4） |

- `gate.txt` の 1 語目は既存の PASS／TIMEOUT／INVALID だけを使う。集計（`aggregate_diag.py`・`aggregate_4151.py`・`aggregate_4154.py` 等）は
  既存のとおり PASS 以外の round を正式系列から外す。差し替え・再実行・除外はしない。
- 各回の結果は `<env>/foreign-gate/r<i>-before-a<k>.txt`・`r<i>-after.txt`（マスク済み: ユーザー名・$HOME・ホスト名を置換し、
  自分以外の uid は数値のまま。プロセスは basename だけでパスは書かない）。

## ファイル

| ファイル | 役割 |
|---|---|
| `foreign_job_gate.py` | 判定（1 回）。`--phase before|after --out <file>`・`--self-test`（合成 fixture。境界値・除外・GPU の回数・解釈の失敗） |
| `round_with_foreign_gate.sh` | 1 round を前後のゲートで挟む。`<run_round.sh> <round 番号> <round ディレクトリ> <env ディレクトリ>` |
| `test_round_with_foreign_gate.sh` | ラッパーの経路の self-test（スタブのゲートで CLEAN・待ち・TIMEOUT・終了後 FOREIGN・ゲート失敗・round 失敗） |
| `idle-check-20261110/` | 平常時の確認の記録（10 回分のゲート出力と時刻・load1。上の「平常時の確認」） |

## 確認

- `python3 -I foreign_job_gate.py --self-test` → `self-test ok`
- `bash test_round_with_foreign_gate.sh` → `self-test ok`（macOS の `/bin/bash` 3.2 と bash 5 の両方）
- 実機での形式確認（値は見ない捨てる前提の試走）は、各 issue の RULE を固定する前に行い、その RULE の PR に記録する。
