# 0.15.0 の 6 フレームワーク同席計測 — GB10（DGX Spark）

**状態: 計測済み（5/5）・5 run 中央値を集計済み。判定一覧は #4179。** crates.io `fandhe-ai =0.15.0` を GB10（DGX Spark）で 5 ラウンド測った #4178 の収録物と計測キット。
RULE は `RULE.txt` で固定済み（計測 SHA `4fce735fb57d97c09ce96b7c6b77eff915c24229`）。判定一覧（`gen_0150.py`・`check_0150.py`）と勝敗・前版比の表は #4179 の範囲で、本ディレクトリには置かない。本 README には数値の表・比・勝敗を書かない。
Mac mini は #4177 が `../framework-compare-0.15.0-macmini-remeasure/` で行った。

## 計測の事実（収録物で確認済み）

- 系列: 2026-11-14T16:36:06Z〜17:09:51Z。`gb10/loop.log` が `ALL-DONE completed=5/5`。
- 専有ゲート: 各 round の前に load1<1.0 かつ GPU 0% を 3 サンプル連続で確認（`gb10/gate.log`）。系列の前後の compute プロセスは 0 件（`gb10/compute_apps_before_series.txt`・`compute_apps_after_series.txt`）。
- 環境変数検査: none（`gb10/env_guard_before_series.txt`・各 run の記録）。
- venv: `venv-bench` の freeze は 0.14.0 の収録と byte 一致（`env/freeze-cmp.txt`）。
- 計測 SHA: `rev-stamp.txt` は 5 round とも `4fce735fb…`（RULE と一致）。
- 同一性検査: `aggregate/check_identity.py`（0.14.0 の同名版から版の文字列だけを 0.15.0 に変えたもの）が rc=0。行数は stage1 112・stage2（extra）8・stage3（Python）28（5 run とも）。

## 5 run 中央値の集計

`aggregate/median_rounds.py`（0.14.0 の同名版と byte 一致）で run1..5 から、セルごとに `median_s` の中央値 round の行を選んで 1 本にした（全セルで 5 run 揃い。2 回実行して byte 一致）。stage1 の 112 行のうち `train_phases`・`infer_phases` の行は集約の対象外（0.14.0 と同じ）のため、集約後は 58 セルになる。判定・勝敗・集計表は含まない（#4179）。

| ファイル | セル数 | sha256 |
|---|---|---|
| `gb10/results-dgx-0.15.0-median5.jsonl` | 58 | `cca61103bf3d25d6896775657994de8f02b8fe09407c96ef31b8c425e170f143` |
| `gb10/results-dgx-0.15.0-extra-median5.jsonl` | 8 | `14d10b49b13f095c3191bc6bfaf21cd2fb0d97696d7bf4428b02db3dea076ea3` |
| `gb10/results-dgx-py-0.15.0-median5.jsonl` | 28 | `1d4b3dc5164429a34134d84207cca6c1cff0b996b7cf882ebc7360381374631d` |

スクリプトの sha256: `aggregate/check_identity.py` は `1e8e4064d1b7609e341ca7beb3b13a4b4acbf2fdbb939098909b337ba526de32`、`aggregate/median_rounds.py` は `8b74f22339ee94a280617a209214134f6c55698eb86ef01bae3f9d657724306f`。

再現（本ディレクトリで）:

```bash
PIN=4fce735fb57d97c09ce96b7c6b77eff915c24229; g=gb10
python3 -I aggregate/check_identity.py $PIN results-dgx-0.15.0.jsonl=112,results-dgx-0.15.0-extra.jsonl=8,results-dgx-py-0.15.0.jsonl=28 $g/run{1..5}
python3 -I aggregate/median_rounds.py $g/results-dgx-0.15.0-median5.jsonl $g/run{1..5}/results-dgx-0.15.0.jsonl
python3 -I aggregate/median_rounds.py $g/results-dgx-0.15.0-extra-median5.jsonl $g/run{1..5}/results-dgx-0.15.0-extra.jsonl
python3 -I aggregate/median_rounds.py $g/results-dgx-py-0.15.0-median5.jsonl $g/run{1..5}/results-dgx-py-0.15.0.jsonl
sha256sum -c SHA256SUMS
```

以降の節は計測前に書いた計測キットの説明（0.14.0 からの変更・手順・自己テスト）である。

## 0.14.0 からの変更

写し元は `../framework-compare-0.14.0-remeasure/scripts/` の 2 本。判定式・閾値・N=5・期待行数（stage1 112・stage2 8・stage3 28・skipped 0）・`bench_py.py` の sha256（`9fd2c08f…`）・専有ゲート（load1<1.0 かつ GPU 0% を 3 サンプル連続・30 秒間隔・最大 30 分、系列の前と各 round の前に compute プロセス 0 件）は変えない。GB10 に他ジョブゲートは足さない。

| # | 変更 | 場所 |
|---|---|---|
| 版 | ラベル・出力先（`~/work/dgx-0150-logs`）・ファイル名の 0140 → 0150、`tree.txt` のピン検査を `fandhe-ai v0.15.0$` に、ロック owner の正規表現を `#4178 rel-0\.15\.0 <UTC>` に（0.14.0 は `#3875 rel-0\.14\.0`） | 2 本とも |
| 2 | **環境変数の検査**。系列の前（`dgx-0150-loop.sh`）と各 run の前（`dgx-run-0150.sh`）に `FANDHE_AI_*` と `RAYON_NUM_THREADS` が 1 つも無いことを確かめ、あれば名前だけを出して非 0 で止める。結果（なし）は `env_info.txt` の末尾（`hw-lock-owner:` の後ろ。先頭行と `compute-apps:` の形式は 0.14.0 のまま）と `env_guard_before_series.txt` に残る。GB10 は #4299 で env=0/1 を使った直後のため残留を疑う | `scripts/env_guard.sh`・loop・run |
| 3 | **収録スクリプト** `collect.sh`（語境界つきマスク・漏れ検査・`SHA256SUMS`・計測後の freeze 収録と cmp） | `collect.sh`・`scripts/mask_lib.sh` |
| 4 | **事前検査**。`~/work/.venv-bench` の freeze（`pip freeze` の直後に `python --version`。0.14.0 と同じ形式）が 0.14.0 の `env/pip-freeze-venv-bench.txt` と byte 一致するか。不一致・不在なら測らず止まる（結果は `venv_preflight.txt`） | `scripts/venv_preflight.sh`・loop |
| 5 | 自己テスト | `scripts/self-test.sh` |

その他の違い: 呼び先を本スクリプトの隣（`SD`）から取る（0.14.0 は `~/work` 直下）。

**`run_0140.sh` について**: 依頼文の写し元にあった `run_0140.sh` は GB10 の測定道具ではなく `scoreboard/run_0140.sh`（スコアボード生成）で、#4179 の範囲のため写していない。

**意図して 0.14.0 の文字列を残している箇所**: スクリプト冒頭のコメント（写し元・変更点の説明）、run の `bench-fandhe が 0.14.0 から変わっている` のメッセージ、`venv_preflight.sh`・`collect.sh` が参照する 0.14.0 の収録 freeze（`../framework-compare-0.14.0-remeasure/env/`）。

## 0.15.0 の既定（RULE で書き換える。環境変数では変えない）

0.14.0 の「SME OFF・Graph OFF の既定」を、0.15.0 の既定に合わせる。GB10 では推論スコープ jpw1（aarch64 Linux 限定）と CUDA 推論 Graph 既定 ON（既定ストリームが created stream になり学習にも効く）が効く見込み。GB10 は SME 非搭載で CPU は NEON のまま。環境変数で既定を変えないことは `env_guard.sh` が担保する。

## 手順（機体側）

1. 計測 SHA は道具 PR マージ後の main。その SHA で clone し、`.rev-stamp`（40 桁 1 行）を作る。`ROOT` はその clone。
2. 事前検査: `bash scripts/venv_preflight.sh gb10 "${HOME}/work/.venv-bench"`（rc=0 かつ `bench identical`）。スクリプトを平置きした場合は `FREEZE_REF_DIR` に 0.14.0 の `env/` を指す。
3. `~/work/bench_py.py`（sha256 は run ごとに照合）を置く。
4. dry-run の前に実機ロックを取る（`mkdir ~/work/.fandhe-hw-lock` が成功したときだけ、`owner` に `#4178 rel-0.15.0 <UTC>`〈`date -u +%FT%TZ`〉を 1 行で書く。`mkdir` が失敗したら他のジョブが使用中なので待つ）。`dgx-run-0150.sh` は owner がこの形式でないと計測前に止まる。
5. dry-run（ロックを持ったまま `dgx-run-0150.sh` で 1 round を通して rc=0・行数 112／8／28・skipped 0。値は見ない・残さない）。終わったら owner が自分のものであることを確かめてからロックを外す（`rm -rf ~/work/.fandhe-hw-lock`）。RULE の固定（別 PR）を待つ間はロックを持たない。
6. RULE の固定後、もう一度ロックを取る（owner は新しい時刻で同じ形式）。系列:

   ```bash
   mkdir -p ~/work/dgx-0150-logs
   ROOT=<clone> EXPECT_SHA=<計測 SHA> setsid nohup bash <clone>/docs/perf/logs/framework-compare-0.15.0-remeasure/scripts/dgx-0150-loop.sh > ~/work/dgx-0150-logs/loop.log 2>&1 &
   ```

   `loop.log` に `ALL-DONE completed=5/5` が出たものだけを採る。`SERIES-ABORTED` で打ち切ったら報告して止める（取り直しは別名の RULE）。系列のロックは `dgx-0150-loop.sh` が終了時（打ち切り・失敗を含む）に自動で外す（owner が `#4178 rel-0.15.0 ` で始まるときだけ。`lock-released` が出る）。
7. 収録（clone の本ディレクトリで）: `SERIES_DIR=~/work/dgx-0150-logs bash collect.sh`。rc=0 なら漏れ 0 件・`SHA256SUMS` 生成済みで、freeze は 0.14.0 と byte 一致（`env/freeze-cmp.txt`）。rc=5 は収録完了だが計測後の freeze が不一致（所有者へ報告）。

## 自己テスト

```bash
bash scripts/self-test.sh      # rc=0 なら全件通過。実機なしでも Linux で動く
shellcheck -s bash -S warning -x scripts/*.sh collect.sh
```

環境変数検査・マスクの語境界・venv の freeze 検査（macmini・gb10 の両形式）・`SHA256SUMS` を検査する（Mac mini 側のディレクトリと同一ファイル。他ジョブゲートの検査は `foreign_gate_wrapper.sh` が無いためこちらでは飛ばす）。

## ファイル

| パス | 内容 |
|---|---|
| `scripts/dgx-0150-loop.sh` | 5 ラウンドのループ（専有ゲート・環境変数検査・venv 事前検査・ロック解除） |
| `scripts/dgx-run-0150.sh` | 1 round |
| `scripts/env_guard.sh`・`venv_preflight.sh`・`mask_lib.sh`・`self-test.sh` | Mac mini 側ディレクトリと同一内容 |
| `collect.sh` | 収録 |

## 未解決・申し送り

- `RULE.txt`（別 PR）に、道具の sha256 と計測 SHA を書き込む。0.14.0 の RULE との差は変更点 2〜3 と「SME OFF・Graph OFF」の文言だけにする。
- 0.14.0 → 0.15.0 の差には、既定の変更（jpw1・Graph ほか）が含まれる。#4179 のスコアボードの注記に書く。
- 入力: `run1..5/`、median5 の jsonl（GB10 は 3 本）、各 run の `env_info.txt`。

## 運用メモ・既知の制限

- **打ち切り後の再実行**: `~/work/dgx-0150-logs` は系列のログ置き場で、同じ場所へ重ねて回すと run のログが混ざる。打ち切り系列は採らず、再実行（取り直し）の判断は所有者が行い、新しいログ置き場（別名の RULE）を使う。なお Mac mini 側は `fg_require` が既存の出力ディレクトリを拒否するため、同じ `D` での再実行は `FOREIGN-GATE PRECHECK FAILED` で止まる（そちらの README と self-test）。
- **マスクの既知の制限**: ユーザー名が `home`・`user`・`node` のような語のとき、置換先（`<home>`・`<user>`・`<cuda-node>`）の中身と衝突して `<<user>>` のようになりうる。漏れは残さない側（fail-closed）なので直さず許容する。
