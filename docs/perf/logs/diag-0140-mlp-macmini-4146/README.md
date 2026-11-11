# 診断 (3): <metal-node> の MLP 2 セル（M-CPU-TRN・M-CPU-INF）（#4146）

## 状態

**PR-2（2026-11-11 実測）: 実測済み。** 本系列は `series-formal-1/`（5 round・全 PASS・正式）。規則は `RULE.v2.txt`（2026-11-11T01:20:17Z 固定・PR #4240）。
`RULE.txt` では計測していない（残置）。分類・確度は下の「結果」と `series-formal-1/aggregate.md`。分類の scoreboard への反映はこの issue では行わない（#4168）。

雛形と規則の一般論は `docs/perf/logs/diag-0140-template/README.md`、式の定義は `docs/perf/loss-attribution-matrix.md` §10.1。

## 対象

| セル | 機体 | scoreboard の比 | 現分類（確度低） | 相手 |
|---|---|---|---|---|
| M-CPU-TRN | <metal-node> | 0.24 | 自分たちの制約 | PyTorch |
| M-CPU-INF | <metal-node> | 0.15 | 自分たちの制約 | PyTorch |

出典: `docs/perf/logs/framework-compare-0.14.0-remeasure/scoreboard/unreached_0140.tsv`。分類の反映はこの issue では行わない（#4168）。

## 注意（読み方）

- tree は 0.14.0 計測の腕 `18a7f4b87fe61029a179cb1922cc82928440f2c9`。`crates/backend-cpu` は v0.14.0 と差 0 件で、workspace の `cpu_jobgrid_bench` と crates.io 版 `fandhe-ai =0.14.0` の CPU カーネルは同一コード。
- `tree_fingerprint` はこの SHA の git archive から求めた値（`aggregate_diag.py --fingerprint`）。<metal-node> の展開ツリーで計測前に同値であることを確認し、1 件でも違えば停止する。
- scoreboard の train reuse は常駐チェーン（`train_step_resident`。第 1 層の入力勾配 dX を計算しない。#3245）。fresh の窓（`run_train`）は入力が `tape.var` のため **L1 の dX を計算している**ので、F2 の fresh 窓にはその余分な仕事が含まれる。
- 純カーネルは GEMM のみ（ReLU・MSE・SGD・readout は両側とも含まない）。学習は l1_fwd・l2_fwd・l2_dw・l2_dx・l1_dw の 5 形状（l1_dx は含めない）、推論は l1_fwd・l2_fwd。時計は両側 `wall`（単発 GEMM を温めた状態で測った値の和で、step 内の実行時の状態とは一致しない。F1 の注意）。
- `kernel_owner=hardware` は測る前の宣言（PyTorch CPU の GEMM は閉源の Accelerate、REQ-1 により fandhe-ai は使わない）。実測時に `torch.__config__.show()` の BLAS 経路を `env/` に記録し、Accelerate でなくても宣言は動かさず、ここに注記する。
- ノイズ帯 NB-RANGE5 は `docs/perf/noise-band-decision.md`（案・所有者承認待ち）。承認後に定義が変わったら RULE.v2.txt で反映する。
- **RULE.txt は #4208 の修正前の集計スクリプトを固定しているため使わない。実測の前に RULE.v2.txt を単独コミットで固定する。**`RULE.txt` の `aggregate_script_sha256`（`d55f295d…`）は、時計不一致の F1 セルを窓超過で異常終了させていた旧版の `aggregate_diag.py` の値。#4208 で雛形を直してコピーも合わせたため、旧 RULE のままでは集計が sha256 不一致で止まる（fail-closed）。旧 RULE は消さず・上書きせずに残す（雛形 README「RULE の改訂は `RULE.v2.txt` を別ファイルで置き、旧版を残す」）。`RULE.v2.txt` は測る直前に <metal-node> 側で `RULE.txt` から作り、`fixed-at-utc`（1 行目と PARAMS）・`tree_sha`・`tree_fingerprint`・`aggregate_script_sha256`（`sha256sum aggregate_diag.py`）を入れ直す。
- issue 本文の内訳（forward 206・backward 452・更新 118 µs）は 0.11.0 時点（MM-180）の値。0.14.0 の内訳は record_only の `--phases`（Tape 経路）で取り直す。分類には使わない。

## ファイル

| ファイル | 役割 |
|---|---|
| `RULE.txt` | #4208 より前に固定した規則。修正前の `aggregate_diag.py`（sha256 `d55f295d…`）を固定しているため使わない（残置。上の注意） |
| `RULE.v2.txt` | 実測の前に単独コミットで固定する規則（未作成。測る直前に <metal-node> 側で作る） |
| `RULE.template.txt`・`aggregate_diag.py` | 雛形のコピー。#4146 の試走を受けて窓超過を比較不能にした版（雛形と同一。`aggregate_diag.py` は無改変で雛形と byte 一致。sha256 を `RULE.v2.txt` の PARAMS に固定） |
| `to_values_4146.py` | `raw/` → `values.tsv` の変換（厳格パース・`--self-test`） |
| `run_round.sh` | 1 round の起動（ゲート → 計測 → `gate.txt`・`raw/`） |

## 手順（PR-2・<metal-node>）

0. `RULE.v2.txt` を作り（上の注意）、`python3 -I aggregate_diag.py --self-test` を通してから**単独コミット**して push する。以降の手順の「RULE」は `RULE.v2.txt` を指す。
1. hw-lock を取り、ツリーの指紋を `RULE.v2.txt` と照合する。framework-compare と workspace ルート（`cargo build --release -p fandhe-ai-backend-cpu --example cpu_jobgrid_bench`）を事前ビルドする。
2. `ROOT=… D=… EXPECT_SHA=<RULE.v2.txt の tree_sha> HW_LOCK_HELD=1 bash run_round.sh <i> <記録先>/round<i>` を 1〜5 で回す。run の差し替え・除外・上書きは禁止。
3. 各 round で `python3 -I to_values_4146.py --round-dir <記録先>/round<i>`（`raw/rust.jsonl` は bench-fandhe、`raw/py.jsonl` は bench_py、`raw/jobgrid.txt`、`raw/opp_kernel.jsonl`）。
4. 回収時にパス上のユーザー名・ホスト名をマスクする。`env/` にツリー指紋・torch 設定・`.rev-stamp` を置く。
5. `python3 -I aggregate_diag.py --dir <記録先> --rule RULE.v2.txt` で `aggregate.md`・`aggregate.tsv` を生成する。
6. #3244 の確認を下に書く: train reuse の窓が #3247 の after（約 0.50 ms）と整合し before（約 0.67 ms）でないこと、record_only の `train --phases --mode reuse` の backward が 0.11.0 の 451.5 µs より小さいこと。効いていなければその事実を書いて停止する。
7. 結果欄に 2 セルの分類と確度を書き、`Closes #4146` の PR-2 を出す。

## 結果

記録先は `series-formal-1/`（`aggregate.md`・`aggregate.tsv`・`round1〜5/`（`values.tsv` は手順 3 の変換）・`round<i>.log`・`drive.status`・`env/`・`RULE.v2.txt` の写し）。
集計は `python3 -I aggregate_diag.py --dir series-formal-1 --rule RULE.v2.txt`（rc=0）。

### 系列

- 系列の開始: 2026-11-11T01:48:51Z、終了: 01:49:47Z。round 1〜5 とも開始前・終了後の他ジョブのゲートが CLEAN で、`gate.txt` は PASS（`drive.status`）。
- 系列の前の確認（`env/preflight.txt`）: `mediaanalysisd` は 2 分間隔で 3 回とも %CPU 0.0、spotlightknowledged の「Submitting text processing」は直近 10 分で 0 件。
- ビルドの前と後のツリーの指紋は RULE.v2 の値と一致し（`env/build/`）、計測後も一致した（`env/trees.txt`。`__pycache__` は 0 件）。bench-fandhe の fandhe-ai は crates.io の 0.14.0（`env/build/tree-fandhe-ai.txt`）。
- torch の BLAS 経路は `BLAS_INFO=accelerate`（`env/torch-config.txt`。系列の前に取得）で、`kernel_owner=hardware` の宣言どおり Accelerate。
- 系列の途中で値は見ていない（見たのは `drive.status` と `gate.txt` だけ）。固定前の試走で値を見たこと・ビルドを hw-lock なしで行ったことは `RULE.v2.txt` 項 8 (e) に記録した。
- 本系列では、純カーネルが窓を上回る round（窓超過）はなかった（`aggregate.md` に窓超過の行なし）。

### 分類（`aggregate.md` から転記）

| セル | F1 中央値（NB-RANGE5） | F2 中央値（NB-RANGE5） | 観測比 中央値 | 分類／確度 | 前分類との異同 |
|---|---|---|---|---|---|
| M-CPU-TRN | 0.9162（0.8573〜1.0316・閾値をまたぐ） | 0.1765（0.1717〜0.1976・またがない） | 0.2453 | 分類保留／低（どれも届かない。残差は固定費など別区間。別診断へ） | 変更（自分たちの制約 → 分類保留） |
| M-CPU-INF | 1.1559（0.9806〜1.2750・閾値をまたぐ） | 0.1530（0.1460〜0.1559・またがない） | 0.1550 | ハードの上限／低（F1 が届く。相手の純カーネルに置換すると届く） | 変更（自分たちの制約 → ハードの上限） |

量ごとの 5 round の中央値（ms。`round<i>/values.tsv`）:

| セル | fandhe の窓（reuse） | fandhe の fresh 窓 | 相手の窓 | fandhe の純カーネル | 相手の純カーネル |
|---|---|---|---|---|---|
| M-CPU-TRN | 0.4999（0.4975〜0.5084） | 0.7010 | 0.1220 | 0.4092 | 0.0430 |
| M-CPU-INF | 0.2045（0.2015〜0.2080） | 0.2069 | 0.0317 | 0.1972 | 0.0212 |

- 純カーネル（GEMM のみ・時計は両側 wall）は fandhe が相手の約 9.5 倍（TRN）・9.3 倍（INF）。F1 は GEMM を相手の水準へ置換した仮想上限で、Accelerate（閉源）の再現ではない。
- TRN は GEMM を置換しても F1 が 1.0 に届かない run がある（中央値 0.92）。窓の残り（約 0.09 ms）は GEMM 以外の区間。

### #3244 の確認（手順 6）

- train reuse の窓は 5 round で 0.4975〜0.5084 ms で、#3247 の after（約 0.50 ms）と整合し、before（約 0.67 ms）ではない。
- record_only の `train --phases --mode reuse` の backward は 5 round で 256.2〜264.2 µs で、0.11.0 の 451.5 µs より小さい（`round<i>/raw/phases-train.jsonl`）。
  step_total は 492.9〜504.3 µs、forward_resident 205.8〜210.9 µs、device_update 29.5〜31.0 µs。
- 以上から #3244（入力勾配 dX の省略）は本系列のツリーで効いている。
