# 0.15.0 の 6 フレームワーク同席計測 — Mac mini（Apple M4）

**状態: 道具のみ（RULE 未固定・未計測）。** crates.io `fandhe-ai =0.15.0`（framework-compare のピンは #4305 で更新済み）を Mac mini の系列 B で測る #4177 の計測キット。
実測値・判定・`RULE.txt` はまだ無い。`RULE.txt` は別 PR で道具の sha256 と計測 SHA を書き込んで固定する（#4168 の「道具 #4292 → RULE #4294」と同じ分離）。
スコアボードの生成器・検査器（`gen_0150.py`・`check_0150.py`）は #4179 の範囲で、本ディレクトリには置かない。GB10 は #4178 が別ディレクトリ `../framework-compare-0.15.0-remeasure/` で行う。
計測の実行と値の確認はこの PR の範囲外。

## 0.14.0 からの変更

写し元は `../framework-compare-0.14.0-macmini-remeasure/scripts/` の 3 本。判定式・閾値・N=5・期待行数（Rust 48・Python 32・skipped は burn Metal gemm N=512〜4096 fresh の 4 件だけ）・`bench_py.py` の sha256（`9fd2c08f…`）・機体検査（`Apple M4` の完全一致）・系列 B（系列 A は行わず、ノイズ帯は na）は変えない。

| # | 変更 | 場所 |
|---|---|---|
| 版 | ラベル・出力先・ファイル名の 0140 → 0150、`tree.txt` のピン検査を `fandhe-ai v0.15.0$` に、ロック owner の形式を `#4177 rel-0.15.0 <UTC>` に | 3 本すべて |
| 1 | **他ジョブゲート**。`foreign_gate_wrapper.sh` が sha256 固定済みの `docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py`（`d8cf0fd6…`）と `scripts/bench/framework-compare/ab_foreign_gate.sh`（`b7ae69bc…`）の sha256 を検査してから、中身を変えずに `fg_wait_before`・`fg_after` を呼ぶ。各ラウンドの前は CLEAN かつ load1<3.0 になるまで 30 秒間隔・最大 30 分待ち（超えたら GATE-TIMEOUT）、後は FOREIGN（またはゲート自身の失敗）ならそのラウンドを無効にして系列を打ち切る。5/5 完走だけを採る。扱いは #4168 と同じ（`prepost-0150-4168/RULE.txt` の [負荷ゲート]・`run_ab_prepost_0150_4168.sh` の `gate_wait`）。gate.log の文法も #4168 と同じ | `scripts/foreign_gate_wrapper.sh`・`macmini-0150b-loop.sh` |
| 1' | 0.14.0 の load1<3.0 の run 前待ち（1 分間隔）は、上の待ち（30 秒間隔）に含めた。閾値と上限（30 分）は同じ。系列 B のまま | `macmini-0150b-loop.sh` |
| 2 | **環境変数の検査**。各 run の前に `FANDHE_AI_*` と `RAYON_NUM_THREADS` が 1 つも無いことを確かめ、あれば名前だけを出して非 0 で止める。結果（なし）は run ごとの `env_guard.txt` と、系列の前後の `env/env_info_*.txt` に残る | `scripts/env_guard.sh`・run・series |
| 3 | **収録スクリプト** `collect.sh`（語境界つきマスク・漏れ検査・`SHA256SUMS`・計測後の freeze 収録と cmp） | `collect.sh`・`scripts/mask_lib.sh` |
| 4 | **事前検査**。比較相手の venv の freeze が 0.14.0 の `env/pip-freeze-venv-{torch,tf}.txt` と byte 一致するか。不一致・不在なら測らず止まる（作り直しは別の承認事項） | `scripts/venv_preflight.sh`・series |
| 5 | 自己テスト | `scripts/self-test.sh` |

その他の違い: ロック owner の形式検査（0.14.0 は文字列一致のみ）、`env_info_*.txt` に他ジョブゲート 2 ファイルの sha256 と環境変数検査の結果を追記、呼び先を本スクリプトの隣（`SD`）から取る（0.14.0 は作業ディレクトリ直下）、venv の場所を `PY_VENV_DIR` で指定可能。

**意図して 0.14.0 の文字列を残している箇所**: スクリプト冒頭のコメント（写し元・変更点の説明）、run の `bench-fandhe が 0.14.0 から変わっている` のメッセージ（`bench-fandhe` を 0.14.0 から変えない前提の検査）、`venv_preflight.sh`・`collect.sh` が参照する 0.14.0 の収録 freeze（`../framework-compare-0.14.0-macmini-remeasure/env/`）。

## 0.15.0 の既定（RULE で書き換える。環境変数では変えない）

0.14.0 の「SME OFF・Graph OFF の既定」を、0.15.0 の既定に合わせる。Mac mini M4 では SME 既定 ON（SME small sched ON）と Metal STEEL large-N（正方 1024/2048/4096）が効く見込み。jpw1（aarch64 Linux 限定）と CUDA の Graph は GB10 側。環境変数で既定を変えないことは `env_guard.sh` が担保する。

## 手順（機体側）

1. 計測 SHA は道具 PR マージ後の main。その SHA で clone し、`.rev-stamp`（40 桁 1 行）を作る。`ROOT` はその clone。
2. 比較相手の venv を決める。**候補は 2 つ**: 0.14.0 の作業ディレクトリに残っている `venv-torch`・`venv-tf`、無ければ `<home>/work/venv4168.noindex/`（#4168 で作り直した venv。直下に `venv-torch`・`venv-tf` がある前提）。使う方を **`PY_VENV_DIR`** に指定する（既定は `D`）。
3. 事前検査: `bash scripts/venv_preflight.sh macmini "${PY_VENV_DIR}"`（rc=0 かつ `torch identical`・`tf identical`）。スクリプトを平置きした場合は `FREEZE_REF_DIR` に 0.14.0 の `env/` を指す。
4. 作業ディレクトリ `D` に `py/bench_py.py` を置く（sha256 は run ごとに照合）。dry-run の前に実機ロックを取る（`mkdir ~/work/.fandhe-hw-lock` が成功したときだけ、`owner` に `#4177 rel-0.15.0 <UTC>`〈`date -u +%FT%TZ`〉を 1 行で書く。`mkdir` が失敗したら他のジョブが使用中なので待つ）。
5. dry-run: `BUILD_ONLY` ではなく 1 round を通して rc=0・行数（Rust 48・Python 32・skipped 4）を確かめる（結果の値は見ない・残さない）。終わったら owner が自分のものであることを確かめてからロックを外す（`rm -rf ~/work/.fandhe-hw-lock`）。この行数の確認のあとに RULE を固定する（別 PR）。固定を待つ間はロックを持たない。
6. RULE の固定後、もう一度ロックを取り（owner は新しい時刻で同じ形式）、その owner を引数に渡して系列を始める。系列のロックは `macmini-0150-series.sh` が終了時（打ち切り・失敗を含む）に自動で外す（owner が引数と一致するときだけ）。系列（bash 3.2・`setsid` なし。`ROOT`・`D`・`EXPECT_SHA`・任意の `PY_VENV_DIR` を export）:

   ```bash
   (nohup bash "<ROOT>/docs/perf/logs/framework-compare-0.15.0-macmini-remeasure/scripts/macmini-0150-series.sh" "<ロックの owner>" > "${D}/loop-macmini.log" 2>&1 < /dev/null &)
   ```

   `loop-macmini.log` に `ALL-DONE completed=5/5` が出たものだけを採る。`SERIES-ABORTED`・`GATE-TIMEOUT`・`GATE-INVALID` で打ち切ったら報告して止める（再開・取り直しは別判断で、別名の RULE）。
7. 収録（clone の本ディレクトリで。`SERIES_DIR=${D}/macmini-0150b`・`LOOP_LOG=${D}/loop-macmini.log`・`ENV_SERIES_DIR=${D}/env-series`）:

   ```bash
   SERIES_DIR=... LOOP_LOG=... ENV_SERIES_DIR=... PY_VENV_DIR=... bash collect.sh
   ```

   rc=0 なら漏れ 0 件・`SHA256SUMS` 生成済みで、freeze は 0.14.0 と byte 一致（`env/freeze-cmp.txt`）。rc=5 は収録完了だが計測後の freeze が不一致（所有者へ報告）。

## 自己テスト

```bash
bash scripts/self-test.sh      # rc=0 なら全件通過。実機・Mac なしでも Linux で動く（python3・sysctl・sleep・uv をモック）
shellcheck -s bash -S warning -x scripts/*.sh collect.sh
```

ゲートのラッパーの判定分岐（CLEAN／FOREIGN／FOREIGN の後に CLEAN／load1 超過／ゲート自身の失敗 3 種／load1 取得不可／タイムアウトの待ち回数／sha256 不一致・欠落・相対パス・出力ディレクトリ既存）、マスクの語境界、環境変数検査、venv の freeze 検査、`SHA256SUMS` を検査する。
ラッパーの固定値が #4168 の `RULE.txt` の値と一致することも見る。

## ファイル

| パス | 内容 |
|---|---|
| `scripts/macmini-0150-series.sh` | 起動役（owner 形式検査・環境変数検査・venv 事前検査・env 記録・ロック解放） |
| `scripts/macmini-0150b-loop.sh` | 系列 B（他ジョブゲートつき） |
| `scripts/macmini-run-0150.sh` | 1 round |
| `scripts/foreign_gate_wrapper.sh` | 他ジョブゲートの薄いラッパー（Mac mini 側のみ） |
| `scripts/env_guard.sh`・`venv_preflight.sh`・`mask_lib.sh`・`self-test.sh` | 両機体で同一内容（GB10 側ディレクトリにも同じものがある） |
| `collect.sh` | 収録 |

## 未解決・申し送り

- `RULE.txt`（別 PR）に、道具の sha256（上のスクリプト群と、固定済みの `foreign_job_gate.py`・`ab_foreign_gate.sh`）と計測 SHA を書き込む。0.14.0 の RULE との差は変更点 1〜3 と「SME OFF・Graph OFF」の文言だけにする。
- `venv4168.noindex` の中身の配置（直下に `venv-torch`・`venv-tf`）はこの PR では確認していない（機体側で `venv_preflight.sh` を当てて確かめる）。
- 0.14.0 → 0.15.0 の差には、既定の変更（SME・STEEL ほか）と、Mac mini に足した他ジョブゲートの影響が含まれる。#4179 のスコアボードの注記に書く。

## 運用メモ・既知の制限

- **打ち切り後の再実行**: 他ジョブゲートの出力ディレクトリ `${D}/macmini-0150b/foreign-gate/` は、`fg_require` が既存を拒否する（差し替え禁止）。そのため、打ち切った系列と同じ `D` で `macmini-0150-series.sh` を再実行すると `FOREIGN-GATE PRECHECK FAILED` で止まる。これは意図した動きで、打ち切り系列は採らず、再実行（取り直し）の判断は所有者が行い、その場合は新しい `D`（別名の RULE）を使う。self-test が確認している。
- **env_info の環境変数検査**: `env_info_before.txt`／`env_info_after.txt` は、それぞれの時点で `env_guard_check` を呼び直した結果を書く（`env_guard_checked_at=` の行つき）。
- **マスクの既知の制限**: ユーザー名が `home`・`user`・`node` のような語のとき、置換先（`<home>`・`<user>`・`<metal-node>`）の中身と衝突して `<<user>>` のようになりうる。漏れは残さない側（fail-closed）なので直さず許容する。
