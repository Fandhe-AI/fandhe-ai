# #4266 GB10 推論限定 jpw1 の測り直し（train の判定を変えた RULE.v2）

状態: **実測済み・ADOPT**（2026-11-12 `<cuda-node>`〈GB10〉。下の「結果」）。`RULE.v2.txt` は PR #4268（マージコミット `c214644dd`）で固定し、そのマージの後に測った。

- #4161（`docs/perf/logs/cpu-infer-jpw1-gb10-ab-4161/`。REJECT）の A/B を、train セルの判定だけを変えて測り直す。#4161 のディレクトリと記録は変えない。
- 判定の変更は所有者判断（#4266 本文「所有者判断（2026-11-12・チャット）」）。
- **本規則は #4161 の値を見た後に書いた。** 何を見たかは `RULE.v2.txt` 項 6。

## 判定（`RULE.v2.txt` 項 3）

| セル | 判定 | 出典 |
|---|---|---|
| infer fresh・infer reuse | 中央値比 < 1.00 で合格、>= 1.00 で REJECT | #4161 `RULE.txt` 項 8（変えていない） |
| train fresh・train reuse | 次の 2 つがどちらも成り立てば非後退。どちらかが外れたら REJECT（条件 1 が外れたときは前提も外れるので undetermined） | 所有者判断 #4266 |

- train の条件 1: 前提の probe で、inf-jpw1 の腕でも `train_<mode>` が `applied=0`（機構が発火していない）。
- train の条件 2: 5 ラウンドの round 別比（inf-jpw1 / off）の最小が 1.00 以下（1.00 ちょうどは非後退）。
- 新しい数値の閾値は足していない。train の中央値比は記録のみ。
- ゲート・前提・checksum・round 数・優先順位（FAIL > undetermined > REJECT > ADOPT）は #4161 `RULE.txt` のまま。

## ファイル

| ファイル | 役割 |
|---|---|
| `RULE.v2.txt` | 事前登録規則 v2。PARAMS に計測対象・tar の指紋・道具 3 つの sha256 を置く |
| `aggregate.py` | 集計。#4161 の `aggregate.py` から派生し、train の判定（`train_nonregress`）・PARAMS の検査・出力の列を変えた |
| `orchestrate.sh` | 計測。#4161 の `orchestrate.sh` から派生し、読む規則ファイル名と、本スクリプト自身・patch の sha256 照合だけを変えた |
| `ab_patch_inf_jpw1.diff` | inf-jpw1 の腕に当てる patch。#4161 の同名ファイルとバイト単位で同じ（sha256 `4152567b…1075`） |
| `test_orchestrate_4266.sh` | `orchestrate.sh` の前提検査を、偽の `nvidia-smi`・python と小さな tar で通す試験（実機なし） |

## 手順（実機セッション。`RULE.v2.txt` 項 4・5）

1. 送信側で `git archive --format=tar 85498f32be4bb8ea70302bd85ec2b8b4329d862f > tree.tar` を作り、sha256 が PARAMS の `tree_fingerprint` と一致することを確かめる。道具 4 本は、本規則を固定した PR のマージ後の main から取り、PARAMS の sha256 と照合する。どのコミットから取ったかを記録する。
2. `<cuda-node>` へ tar と道具を送る（clone・fetch はしない）。新しい展開先に tar を展開して BASE とし、`.rev-stamp` に完全 SHA を書く。LOGDIR（新しいディレクトリ）に道具 4 本を置く。WORK も新しいディレクトリにする。
3. 他の計測・ビルドが動いていないことと、ロックが空いていることを確かめる。`~/work/.fandhe-hw-lock` を手で取る（owner は `#4266 measure <UTC>` の 3 語）。
4. 次で起動する（nohup で切り離す）。`RAYON_NUM_THREADS` は unset。
   ```sh
   BASE=<展開先> LOGDIR=<記録先> WORK=<作業> ARCHIVE=<tree.tar> \
   PY=<torch 2.14.0 の python> BP=<ツリー外の bench_py.py> \
   HW_LOCK_OWNER_FILE=<home>/work/.fandhe-hw-lock/owner HW_LOCK_HELD=1 bash <記録先>/orchestrate.sh
   ```
   - `BP` は sha256 `9fd2c08f…409b` の `bench_py.py`（計測対象 tree の `docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py` と同じ）を、BASE の外に置いたもの。`bench_py.py` は自身の隣に `cache/` を書くため、BASE 内のものは使わない（`RULE.v2.txt` 項 5 (f)）。
5. 系列の途中は値を見ない（見るのは終了コードと `gate.txt` だけ）。終わったら（失敗時も）、計測のプロセスが 0 件であることを確かめてからロックを外す。
6. `python3 -I <記録先>/aggregate.py <記録先>` で集計する。
7. マスクして収録する（ホスト名は `<cuda-node>`、`$HOME` は `<home>`、ユーザー名は `<user>`）。漏れがないことを確かめてから入れる。`env/` には次を置く。
   - `.rev-stamp`
   - `git diff v0.14.0 <tree_sha> -- crates/` の一覧
   - 道具を取ったコミット
   - 使った `bench_py.py` の置き場所（`<home>` に置換）と sha256、torch の版
   - ロックの取得・解放の時刻

## 検証状況（固定前・開発機）

本節は RULE.v2 を固定した時点の記述で、そのまま残す。実測の結果と実行の記録は下の「結果」節。

- `python3 -I aggregate.py --self-test`: OK。#4161 から変えていない部分（推論セル・ゲート・checksum・欠損）に加え、train の新しい判定を確かめる。
  - 中央値比 > 1.00 でも最小 <= 1.00 なら非後退
  - 全 round > 1.00 は REJECT
  - 最小がちょうど 1.00 は非後退
  - round が欠けた比の列は条件 2 不成立
  - probe の `applied` の読み取り（値・重複・欠落）と、条件 1 の不成立
  - PARAMS の検査と `patch_sha256.txt` の照合
  - `run()` を最後まで通す（集計・判定・NB-SEP・`aggregate.md`／`aggregate.tsv` の書き出しと列数）
- `bash test_orchestrate_4266.sh`: OK。本物の `RULE.v2.txt`・patch で前提検査をすべて通り、2 腕のツリーを作って patch を当て（定数 false／true）、ビルドの最初で止まる。次の場合は、何も書かずに止まる。
  - `orchestrate.sh` を 1 行変える
  - patch を変える、または消す
  - PARAMS のピンが欠ける
  - `.rev-stamp` が不一致
  - ロックが未宣言
- 変異の確認: 次の 8 つの変異のどれでも、いずれかの試験が落ちることを確かめた（捨てコピーで。コミットしていない）。
  - train の最小を中央値に戻す
  - `<=` を `<` にする
  - 条件 1 を外す
  - train の非後退を REJECT に使わない
  - round 数を見ない
  - `orchestrate.sh` の自身の照合を外す
  - patch の照合を外す
  - `RULE.txt` を読む
- `ab_patch_inf_jpw1.diff` は `85498f32b` で `git apply --check` が通る。
- **未検証**: GB10 での実行（ビルド・前提 report・5 ラウンド）。本規則の固定前に、実機での捨て試走はしていない（`RULE.v2.txt` 項 6 (c)）。

## 結果（2026-11-12・`<cuda-node>`〈GB10〉）

**verdict: ADOPT**（`aggregate.md`）。正式系列（ゲート 5/5 PASS）・前提成立・checksum 一致。

| セル | 中央値比（inf-jpw1 / off） | round 別比 | 最小 | inf-jpw1 の勝ち | 判定 |
|---|---|---|---|---|---|
| train fresh | 0.9312 | 0.9732・0.9232・0.9312・0.8644・0.9530 | 0.8644 | 5/5 | 非後退（条件 1・2 とも成立） |
| train reuse | 1.0225 | 1.0225・1.1695・1.0066・0.9221・1.0412 | 0.9221 | 1/5 | 非後退（条件 1・2 とも成立） |
| infer fresh | 0.9247 | 0.9831・0.8174・0.9773・0.8432・0.9247 | 0.8174 | 5/5 | 合格（< 1.00） |
| infer reuse | 0.9102 | 0.9776・0.9102・0.8262・0.9148・0.8909 | 0.8262 | 5/5 | 合格（< 1.00） |

- **train reuse は中央値比 1.0225 で、#4161 の `RULE.txt` の式（中央値比 < 1.00）なら REJECT になる値**。本規則では train の中央値比は記録のみで、条件 2（round 別比の最小 0.9221 <= 1.00）により非後退とした。この判定の変更は所有者判断（#4266）による。
- **開示**（`RULE.v2.txt` 項 6 と同じ）: 本規則は #4161 の値を見た後に書いた。本規則の train の判定を #4161 の値に当てても、train の 2 セルは非後退になる（round 別比の最小は train fresh 0.9147・train reuse 0.8691）。
- train の条件 1: `prereq/inf-jpw1.log` で `train_fresh`・`train_reuse` とも `applied=0`（infer は `infer_fresh` 2・`infer_reuse` 3 で発火）。off 腕は 4 パスとも `gate=false applied=0`。`gb10_affinity` は両腕とも `enabled=false`。
- ゲート: 5 ラウンドとも PASS（load1 0.34〜0.70・compute-apps 0・GPU util 0 ×3・hw-lock held。`round<i>/gate.txt`）。
- NB-SEP（記録のみ。採否とは独立）: infer fresh・infer reuse とも不成立（`aggregate.md`）。
- ADOPT でも既定は反転しない（`RULE.v2.txt` 項 5 (e)）。本結果は #4165（CPU 系の結線）の材料になる。

### 実行の記録

- 計測: 2026-11-12T05:39:01Z〜05:50:25Z（`orchestrate.sh` rc=0。`env/status.log`）。
- 実機ロック: 05:39:01Z に手で取得（owner `#4266 measure 2026-11-12T05:39:01Z`）。取得前に、計測系のプロセス（cargo・rustc・python 等）0 件・GPU の compute プロセス 0 件・load1 0.22 を確かめた。終了後、計測のプロセスが 0 件であることを確かめてから 05:51:00Z に解放した。#4161 の起動スクリプトにあった終了時の自動解放は使っていない。
- 計測対象: `85498f32be4bb8ea70302bd85ec2b8b4329d862f`（`env/rev-stamp.txt`）。`git archive` の tar の sha256 は PARAMS の `tree_fingerprint` と送信側・ノード側の両方で一致し、`orchestrate.sh` の前提検査（展開内容と BASE の一致・`.rev-stamp`）も通った。ノード上で clone・fetch はしていない。
- 道具: PR #4268 のマージコミット `c214644dd77f6cd796f24fbdcd1436820f3d13cf` から取った（`env/tools-commit.txt`）。送信側・ノード側の両方で PARAMS の 3 つの sha256 と一致した。本ディレクトリの道具 4 本は、マージ後の main のものから変えていない。
- `bench_py.py`: BASE の外（`<home>/work/jpw1-4266-m/bp/`）に複写したものを使った。sha256 `9fd2c08f…409b`（`env/bench_py.txt`）。torch 2.14.0+cu130（`env/torch-version.txt`）。#4161 では BASE 内の同じファイルを `BP` に渡していた（#4161 の起動記録による。`bench_py.py` は隣に `cache/` を書くので BASE 内に書いたはずだが、#4161 のノード側ディレクトリは残っておらず確かめていない。BASE と tar の一致検査は計測前に行うので、#4161 の判定には影響しない）。
- 集計: ノード上で `python3 -I aggregate.py <記録先>` を実行した（rc=0）。収録後に開発機で、収録物の複製から再集計し、`aggregate.md`・`aggregate.tsv` がバイト単位で一致することを確かめた。
- 系列の起動は 1 回だけ（05:39:01Z）。やり直し・捨てた系列・実機での試走はない。
- 系列の途中では値を見ていない（見たのは `status.log` の終了行・`gate.txt`・残りのプロセス数だけ）。
- 収録: `$HOME` を `<home>` に置換した。ホスト名・ユーザー名・IP が含まれないことを grep で確かめた。
- ノードの作業ディレクトリは `<home>/work/jpw1-4266-m`（#4156 の作業ディレクトリとは別。そちらには触れていない）。

