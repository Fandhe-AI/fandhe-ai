# #4266 GB10 推論限定 jpw1 の測り直し（train の判定を変えた RULE.v2）

状態: **事前登録（RULE.v2）・未実測**。`RULE.v2.txt` を本ディレクトリの PR で固定し、マージの後に `<cuda-node>`（GB10）で測る。

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
