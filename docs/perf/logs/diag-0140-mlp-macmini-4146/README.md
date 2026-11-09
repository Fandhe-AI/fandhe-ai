# 診断 (3): <metal-node> の MLP 2 セル（M-CPU-TRN・M-CPU-INF）（#4146）

## 状態

**PR-1 時点: 実測なし。** 本ディレクトリは測る前に固定する部品（RULE・変換スクリプト・起動キット）だけを置く。
`round{i}/`・`env/`・`aggregate.md`・`aggregate.tsv` は <metal-node> での実測（PR-2）で追加する。
実測値・分類・確度は、実測するまで書かない（推定値で埋めない）。#4146 は PR-2 で閉じる。

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
- issue 本文の内訳（forward 206・backward 452・更新 118 µs）は 0.11.0 時点（MM-180）の値。0.14.0 の内訳は record_only の `--phases`（Tape 経路）で取り直す。分類には使わない。

## ファイル

| ファイル | 役割 |
|---|---|
| `RULE.txt` | 測る前に固定した規則（単独コミット） |
| `RULE.template.txt`・`aggregate_diag.py` | 雛形のコピー（`aggregate_diag.py` は無改変。sha256 を RULE の PARAMS に固定） |
| `to_values_4146.py` | `raw/` → `values.tsv` の変換（厳格パース・`--self-test`） |
| `run_round.sh` | 1 round の起動（ゲート → 計測 → `gate.txt`・`raw/`） |

## 手順（PR-2・<metal-node>）

1. hw-lock を取り、ツリーの指紋を RULE と照合する。framework-compare と workspace ルート（`cargo build --release -p fandhe-ai-backend-cpu --example cpu_jobgrid_bench`）を事前ビルドする。
2. `ROOT=… D=… EXPECT_SHA=… HW_LOCK_HELD=1 bash run_round.sh <i> <記録先>/round<i>` を 1〜5 で回す。run の差し替え・除外・上書きは禁止。
3. 各 round で `python3 -I to_values_4146.py --round-dir <記録先>/round<i>`（`raw/rust.jsonl` は bench-fandhe、`raw/py.jsonl` は bench_py、`raw/jobgrid.txt`、`raw/opp_kernel.jsonl`）。
4. 回収時にパス上のユーザー名・ホスト名をマスクする。`env/` にツリー指紋・torch 設定・`.rev-stamp` を置く。
5. `python3 -I aggregate_diag.py --dir <記録先>` で `aggregate.md`・`aggregate.tsv` を生成する。
6. #3244 の確認を下に書く: train reuse の窓が #3247 の after（約 0.50 ms）と整合し before（約 0.67 ms）でないこと、record_only の `train --phases --mode reuse` の backward が 0.11.0 の 451.5 µs より小さいこと。効いていなければその事実を書いて停止する。
7. 結果欄に 2 セルの分類と確度を書き、`Closes #4146` の PR-2 を出す。

## 結果

（実測後に追記。PR-1 時点は空。）
